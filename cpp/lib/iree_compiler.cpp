#include "kashi/iree_compiler.h"
#include "kashi/cuda_manifest.h"
#include "kashi/iree_partition.h"

#include <atomic>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <unistd.h>

#include "iree/compiler/Dialect/VM/Target/Bytecode/BytecodeModuleTarget.h"
#include "iree/compiler/Pipelines/Pipelines.h"
#include "iree/compiler/embedding_api.h"
#include "iree/compiler/mlir_interop.h"
#include "mlir/CAPI/IR.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/raw_ostream.h"

namespace kashi {
namespace {
using namespace mlir;
namespace fs = std::filesystem;

template <typename T, void (*Destroy)(T *)>
using Handle = std::unique_ptr<T, decltype(Destroy)>;

// Set before entering the static initializer: a fork can inherit its guard
// while another thread holds it, as well as an initialized LLVM worker pool.
static_assert(std::atomic<pid_t>::is_always_lock_free);
std::atomic<pid_t> compilerOwner{0};

struct CompilerRuntime {
  /// Initialize the private compiler before any session is created.
  CompilerRuntime() { ireeCompilerGlobalInitialize(); }
  /// Only the initializing process can safely join the compiler's workers.
  ~CompilerRuntime() {
    if (compilerOwner.load(std::memory_order_relaxed) == getpid())
      ireeCompilerGlobalShutdown();
  }
};

struct Diagnostics {
  /// Open the log before any invocation can emit diagnostics.
  explicit Diagnostics(fs::path path) : path(std::move(path)), log(this->path) {
    if (!log)
      throw std::runtime_error("cannot open IREE diagnostic log: " +
                               this->path.string());
  }
  fs::path path;
  std::ofstream log;
  std::mutex mutex;

  // Compiler worker threads may deliver diagnostics concurrently.
  static void receive(iree_compiler_diagnostic_severity_t severity,
                      const char *text, size_t size, void *data) {
    auto &self = *static_cast<Diagnostics *>(data);
    std::lock_guard lock(self.mutex);
    if (severity == IREE_COMPILER_DIAGNOSTIC_SEVERITY_ERROR)
      self.log << "error: ";
    self.log.write(text, size) << '\n';
  }

  /// Record a stage failure and throw without obscuring its exception category.
  template <typename Error = std::runtime_error>
  [[noreturn]] void fail(const std::string &stage,
                         const char *message = nullptr) {
    std::lock_guard lock(mutex);
    log << "IREE " << stage << " failed";
    if (message)
      log << ": " << message;
    log << '\n' << std::flush;
    throw Error("IREE " + stage + " failed; " +
                (log ? "see " : "cannot write diagnostic log ") +
                path.string());
  }

  /// Consume an embedding API error and translate it into a stage failure.
  void check(iree_compiler_error_t *error, const std::string &stage) {
    Handle<iree_compiler_error_t, ireeCompilerErrorDestroy> scoped(
        error, ireeCompilerErrorDestroy);
    if (error)
      fail(stage, ireeCompilerErrorGetMessage(error));
  }
};

/// Write an artifact directly and remove it if emission or close fails.
template <typename Emit>
void writeArtifact(const fs::path &path, Diagnostics &diagnostics,
                   const std::string &stage, Emit emit) {
  std::error_code error;
  llvm::raw_fd_ostream out(path.string(), error);
  if (error)
    diagnostics.fail(stage, error.message().c_str());
  // A buffered write may first fail on close. Clear that error before the
  // LLVM stream destructor can turn it into a fatal usage error.
  const auto result = emit(out);
  out.close();
  error = out.error();
  if (failed(result) || error) {
    out.clear_error();
    // Do not register a process-wide signal handler to remove this file.
    std::error_code ignored;
    fs::remove(path, ignored);
    diagnostics.fail(stage, error ? error.message().c_str() : nullptr);
  }
}

/// Write parseable module assembly with locations for the Python reader.
void writeModule(ModuleOp module, const fs::path &path,
                 Diagnostics &diagnostics) {
  writeArtifact(path, diagnostics, "write " + path.filename().string(),
                [&](llvm::raw_ostream &out) {
                  module.print(out, OpPrintingFlags().enableDebugInfo());
                  out << '\n';
                  return success();
                });
}

/// Partition and lower a Torch module in one private IREE invocation.
std::optional<std::string>
compile(const std::string &assembly, const IreeCompileOptions &options,
        std::optional<IreeTestPipeline> testPipeline) {
  const pid_t pid = getpid();
  pid_t owner = 0;
  if (!compilerOwner.compare_exchange_strong(owner, pid,
                                             std::memory_order_relaxed) &&
      owner != pid)
    throw std::runtime_error(
        "cannot use an inherited IREE compiler after fork; use spawn or exec");
  static const CompilerRuntime runtime;
  fs::path dir = fs::absolute(options.artifactDir);
  Diagnostics diagnostics(dir / "iree-compile.log");
  const bool cuda = options.target == "cuda";
  if (!cuda && options.target != "vmvx")
    diagnostics.fail("configuration", "expected target cuda or vmvx");
  if (cuda != options.cudaTarget.has_value())
    diagnostics.fail("configuration", "cuda_target is required only for CUDA");

  Handle<iree_compiler_session_t, ireeCompilerSessionDestroy> session(
      ireeCompilerSessionCreate(), ireeCompilerSessionDestroy);
  const char *backend = cuda ? "--iree-hal-target-device=cuda"
                             : "--iree-hal-target-backends=vmvx";
  llvm::SmallString<32> arch("--iree-cuda-target=");
  if (options.cudaTarget)
    arch += *options.cudaTarget;
  // The private compiler does not ship a host HAL driver for constant
  // evaluation.
  const char *flags[] = {"--iree-plugin=kashi",
                         "--iree-input-type=torch",
                         "--iree-opt-const-eval=false",
                         backend,
                         arch.c_str(),
                         "--iree-cuda-use-ptxas=false"};
  diagnostics.check(
      ireeCompilerSessionSetFlags(session.get(), cuda ? 6 : 4, flags),
      "configuration");

  iree_compiler_source_t *wrapped = nullptr;
  auto *sourceError = ireeCompilerSourceWrapBuffer(
      session.get(), "torch.mlir", assembly.c_str(), assembly.size() + 1, true,
      &wrapped);
  Handle<iree_compiler_source_t, ireeCompilerSourceDestroy> source(
      wrapped, ireeCompilerSourceDestroy);
  diagnostics.check(sourceError, "source");
  Handle<iree_compiler_invocation_t, ireeCompilerInvocationDestroy> invocation(
      ireeCompilerInvocationCreate(session.get()),
      ireeCompilerInvocationDestroy);
  auto *inv = invocation.get();
  ireeCompilerInvocationEnableCallbackDiagnostics(inv, 0, Diagnostics::receive,
                                                  &diagnostics);
  if (!ireeCompilerInvocationParseSource(inv, source.get()))
    diagnostics.fail("parse");
  OwningOpRef<ModuleOp> original(
      cast<ModuleOp>(unwrap(ireeCompilerInvocationExportStealModule(inv))));
  auto partition = partitionTorchModule(*original);
  if (failed(partition))
    diagnostics.fail<UnsupportedPartition>("partition");
  writeModule(*partition->segments, dir / "partitioned.mlir", diagnostics);
  writeModule(*partition->driver, dir / "driver.mlir", diagnostics);
  if (!ireeCompilerInvocationImportStealModule(
          inv, wrap(partition->segments.release().getOperation())))
    diagnostics.fail("verify partitioned module");

  auto run = [&](const char *from, const char *to) {
    ireeCompilerInvocationSetCompileFromPhase(inv, from);
    ireeCompilerInvocationSetCompileToPhase(inv, to);
    if (!ireeCompilerInvocationPipeline(inv, IREE_COMPILER_PIPELINE_STD))
      diagnostics.fail(std::string("compile through ") + to);
  };
  auto passes = [&](const char *checkpoint, const char *pipeline) {
    if (!ireeCompilerInvocationRunPassPipeline(inv, pipeline))
      diagnostics.fail(std::string("pass at ") + checkpoint);
  };

  using Phase = mlir::iree_compiler::IREEVMPipelinePhase;
  const bool probes = testPipeline == IreeTestPipeline::Checkpoints;
  const bool split = testPipeline != IreeTestPipeline::Uninterrupted;
  const char *from = "start";
  std::vector<KernelIdentity> identities;
  auto takeModule = [&] {
    return OwningOpRef<ModuleOp>(
        cast<ModuleOp>(unwrap(ireeCompilerInvocationExportStealModule(inv))));
  };
  auto putModule = [&](OwningOpRef<ModuleOp> module, const char *stage) {
    if (!ireeCompilerInvocationImportStealModule(
            inv, wrap(module.release().getOperation())))
      diagnostics.fail(stage);
  };

  // The invocation API appends to its existing root pass manager. Wrap only
  // nested operations: an outer builtin.module would skip the root module.
  mlir::iree_compiler::enumerateIREEVMPipelinePhases(
      [&](Phase phase, StringRef name, StringRef) {
        const char *pipeline = nullptr;
        switch (phase) {
        case Phase::Start:
          pipeline = probes ? "kashi-verify-segments,kashi-probe-torch"
                            : "kashi-verify-segments";
          break;
        case Phase::Flow:
          if (testPipeline == IreeTestPipeline::FailAtFlow)
            pipeline = "kashi-fail-flow-probe";
          else if (probes)
            pipeline = cuda ? "kashi-probe-flow,kashi-collide-flow-functions"
                            : "kashi-probe-flow";
          break;
        case Phase::ExecutableConfigurations:
          if (probes && cuda)
            pipeline = "kashi-probe-configurations";
          break;
        case Phase::ExecutableTargets:
          if (probes && cuda)
            pipeline = "hal.executable(hal.executable.variant(builtin.module("
                       "kashi-probe-nvvm)))";
          break;
        case Phase::VM:
          if (probes)
            pipeline = "kashi-probe-vm";
          break;
        default:
          break;
        }
        const bool snapshot = split && phase == Phase::Flow;
        const bool manifest = split && cuda && phase == Phase::HAL;
        if (!pipeline && !snapshot && !manifest)
          return;
        if (phase != Phase::Start) {
          run(from, name.data());
          from = name.data();
        }
        if (pipeline)
          passes(name.data(), pipeline);
        if (snapshot) {
          auto flow = takeModule();
          writeArtifact(dir / "flow.mlir", diagnostics, "write Flow snapshot",
                        [&](llvm::raw_ostream &out) {
                          flow->print(out);
                          return success();
                        });
          if (cuda) {
            auto result = identifyCudaKernels(*flow);
            if (failed(result))
              diagnostics.fail("identify Flow exports");
            identities = std::move(*result);
          }
          putModule(std::move(flow), "verify Flow checkpoint");
        }
        if (manifest) {
          auto hal = takeModule();
          if (failed(writeCudaManifest(*hal, dir, identities)))
            diagnostics.fail("resolve linked CUDA exports");
          putModule(std::move(hal), "verify linked HAL module");
        }
      });
  // Finish lowering after the last requested checkpoint. The uninterrupted
  // test has only the start verifier, so it lowers from start to end here.
  run(from, "end");
  auto vm = takeModule();
  // No session VM or bytecode flags are overridden, so the pinned
  // serializer's defaults match the invocation's options.
  writeArtifact(
      dir / "module.vmfb", diagnostics, "serialize VM bytecode",
      [&](llvm::raw_ostream &out) {
        return mlir::iree_compiler::IREE::VM::translateModuleToBytecode(
            *vm, {}, {}, out);
      });
  return std::move(partition->sourceDevice);
}
} // namespace

/// Compile with production checkpoints and return the original source device.
std::optional<std::string> compileIree(const std::string &assembly,
                                       const IreeCompileOptions &options) {
  return compile(assembly, options, std::nullopt);
}

/// Run a native test pipeline without exposing pass selection to Python.
void compileIreeForTesting(const std::string &assembly,
                           const IreeCompileOptions &options,
                           IreeTestPipeline pipeline) {
  compile(assembly, options, pipeline);
}
} // namespace kashi
