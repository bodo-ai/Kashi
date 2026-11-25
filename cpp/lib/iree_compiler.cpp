// Kashi embeds IREE to compile partitioned Torch MLIR in process. Only the
// segment module enters IREE's standard pipeline; the driver stays available
// for Python to rebuild FX. At selected phase boundaries we run registered
// Kashi passes and capture artifacts, then resume on the same invocation
// without serializing IR between phases.
//
//   Python: import FX as Torch MLIR
//                  |
//                  v
//   Native: parse and partition --------------------> driver MLIR ---+
//                  |                                                 |
//           segment module                                           |
//                  v                                                 |
//   +--> IREE: lower to next checkpoint                              |
//   |              |                                                 |
//   |              v                                                 |
//   |    Run checkpointPasses() <--- static Kashi plugin             |
//   |              |                 registers pass names            |
//   |              v                                                 |
//   |    Capture artifacts as needed --> Flow MLIR and CUDA PTX      |
//   |              |                     manifest -------------------+
//   +------ resume from this phase                                   |
//                                                                    v
//                                                        Python: rebuild FX

#include "kashi/iree_compiler.h"
#include "kashi/cuda_manifest.h"
#include "kashi/iree_partition.h"

#include <cassert>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <stdexcept>

#include "iree/compiler/Dialect/VM/Target/Bytecode/BytecodeModuleTarget.h"
#include "iree/compiler/Pipelines/Pipelines.h"
#include "iree/compiler/embedding_api.h"
#include "iree/compiler/mlir_interop.h"
#include "mlir/CAPI/IR.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"

namespace kashi {
namespace {
using namespace mlir;
using llvm::StringLiteral;
namespace fs = std::filesystem;

template <auto Destroy> struct Deleter {
  template <typename T> void operator()(T *handle) const { Destroy(handle); }
};

template <typename T, void (*Destroy)(T *)>
using Handle = std::unique_ptr<T, Deleter<Destroy>>;
using Session = Handle<iree_compiler_session_t, ireeCompilerSessionDestroy>;
using Source = Handle<iree_compiler_source_t, ireeCompilerSourceDestroy>;
using InvocationHandle =
    Handle<iree_compiler_invocation_t, ireeCompilerInvocationDestroy>;
using CompilerError = Handle<iree_compiler_error_t, ireeCompilerErrorDestroy>;

struct CompilerRuntime {
  CompilerRuntime() { ireeCompilerGlobalInitialize(); }
  ~CompilerRuntime() { ireeCompilerGlobalShutdown(); }
};

struct Diagnostics {
  /// Open the log before any invocation can emit diagnostics.
  explicit Diagnostics(fs::path path) : path(std::move(path)), log(this->path) {
    if (!log) {
      throw std::runtime_error("cannot open IREE diagnostic log: " +
                               this->path.string());
    }
  }
  fs::path path;
  std::ofstream log;
  std::mutex mutex;

  // IREE may deliver diagnostics from worker threads. Keep each record intact
  // and prevent concurrent writes to this invocation's stream, including
  // writes from fail(). This does not lock or serialize compilation.
  static void receive(iree_compiler_diagnostic_severity_t severity,
                      const char *text, size_t size, void *data) {
    auto &self = *static_cast<Diagnostics *>(data);
    std::lock_guard lock(self.mutex);
    if (severity == IREE_COMPILER_DIAGNOSTIC_SEVERITY_ERROR) {
      self.log << "error: ";
    }
    self.log.write(text, static_cast<std::streamsize>(size)) << '\n';
  }

  /// Record a stage failure and throw without obscuring its exception category.
  template <typename Error = std::runtime_error>
  [[noreturn]] void
  fail(const llvm::Twine &stage,
       std::optional<std::string_view> message = std::nullopt) {
    const std::string failure = ("IREE " + stage + " failed").str();
    std::lock_guard lock(mutex);
    log << failure;
    if (message) {
      log << ": " << *message;
    }
    log << '\n' << std::flush;
    throw Error(failure + "; " +
                (log ? "see " : "cannot write diagnostic log ") +
                path.string());
  }

  /// Consume an embedding API error and translate it into a stage failure.
  void check(CompilerError error, const llvm::Twine &stage) {
    if (error) {
      fail(stage, ireeCompilerErrorGetMessage(error.get()));
    }
  }

  template <typename Error = std::runtime_error, typename T>
  T require(FailureOr<T> result, const llvm::Twine &stage) {
    if (failed(result)) {
      fail<Error>(stage);
    }
    // Keep the MLIR predicate: FailureOr hides std::optional::has_value().
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    return std::move(*result);
  }
};

struct BackendConfig {
  IreeBackend backend;
  std::string_view name;
  StringLiteral deviceFlag;
  std::optional<StringLiteral> architectureFlag;
  llvm::ArrayRef<StringLiteral> extraFlags;
};

constexpr StringLiteral cudaExtraFlags[] = {"--iree-cuda-use-ptxas=false"};
constexpr BackendConfig backendConfigs[] = {
    {IreeBackend::CUDA, "cuda", "--iree-hal-target-device=cuda",
     StringLiteral("--iree-cuda-target="), cudaExtraFlags},
    {IreeBackend::VMVX,
     "vmvx",
     "--iree-hal-target-backends=vmvx",
     std::nullopt,
     {}},
};

// Used only with StringLiteral flags and IREE's literal phase names. The
// embedding API accepts nul-terminated strings, not pointer/length pairs.
const char *terminatedData(llvm::StringRef literal) {
  assert(literal.data()[literal.size()] == '\0');
  return literal.data();
}

const BackendConfig &getBackendConfig(const IreeCompileOptions &options,
                                      Diagnostics &diagnostics) {
  for (const auto &config : backendConfigs) {
    if (options.target == config.backend) {
      return config;
    }
  }
  diagnostics.fail("configuration", "expected target cuda or vmvx");
}

Session createSession(Diagnostics &diagnostics) {
  // Keep the embedded pipeline from running IREE's JIT of constant
  // initializers on the host. This disables runtime-backed evaluation via
  // the local-task CPU driver, not ordinary IR constant folding.
  llvm::SmallVector<const char *, 3> flags = {"--iree-plugin=kashi",
                                              "--iree-input-type=torch",
                                              "--iree-opt-const-eval=false"};
  Session session(ireeCompilerSessionCreate());
  diagnostics.check(
      CompilerError(ireeCompilerSessionSetFlags(
          session.get(), static_cast<int>(flags.size()), flags.data())),
      "configuration");
  return session;
}

/// Configure the target only after native partitioning finds compilable work.
void configureBackend(const Session &session, const BackendConfig &backend,
                      const std::optional<std::string> &architecture,
                      Diagnostics &diagnostics) {
  llvm::SmallVector<const char *, 3> flags = {
      terminatedData(backend.deviceFlag)};
  llvm::SmallString<32> archFlag;
  if (backend.architectureFlag && architecture) {
    archFlag += *backend.architectureFlag;
    archFlag += *architecture;
    flags.push_back(archFlag.c_str());
  }
  for (StringLiteral flag : backend.extraFlags) {
    flags.push_back(terminatedData(flag));
  }
  diagnostics.check(
      CompilerError(ireeCompilerSessionSetFlags(
          session.get(), static_cast<int>(flags.size()), flags.data())),
      "configuration");
}

Source wrapSource(const Session &session, const std::string &assembly,
                  Diagnostics &diagnostics) {
  iree_compiler_source_t *wrapped = nullptr;
  CompilerError error(ireeCompilerSourceWrapBuffer(
      session.get(), "torch.mlir", assembly.c_str(), assembly.size() + 1, true,
      &wrapped));
  Source source(wrapped);
  diagnostics.check(std::move(error), "source");
  return source;
}

class Invocation {
public:
  Invocation(const Session &session, Diagnostics &diagnostics)
      : diagnostics(diagnostics),
        handle(ireeCompilerInvocationCreate(session.get())) {
    ireeCompilerInvocationEnableCallbackDiagnostics(
        handle.get(), 0, Diagnostics::receive, &diagnostics);
  }

  void parse(const Source &source) {
    if (!ireeCompilerInvocationParseSource(handle.get(), source.get())) {
      diagnostics.fail("parse");
    }
  }

  OwningOpRef<ModuleOp> takeModule() {
    return OwningOpRef<ModuleOp>(cast<ModuleOp>(
        unwrap(ireeCompilerInvocationExportStealModule(handle.get()))));
  }

  void putModule(OwningOpRef<ModuleOp> module, llvm::StringRef stage) {
    if (!ireeCompilerInvocationImportStealModule(
            handle.get(), wrap(module.release().getOperation()))) {
      diagnostics.fail(stage);
    }
  }

  // Phase names are null-terminated literals from IREE's phase enumeration.
  void lowerTo(llvm::StringRef to) {
    ireeCompilerInvocationSetCompileFromPhase(handle.get(),
                                              terminatedData(nextPhase));
    ireeCompilerInvocationSetCompileToPhase(handle.get(), terminatedData(to));
    if (!ireeCompilerInvocationPipeline(handle.get(),
                                        IREE_COMPILER_PIPELINE_STD)) {
      diagnostics.fail("compile through " + to);
    }
    nextPhase = to;
  }

  void runPasses(llvm::StringRef checkpoint, llvm::StringLiteral pipeline) {
    if (!ireeCompilerInvocationRunPassPipeline(handle.get(),
                                               terminatedData(pipeline))) {
      diagnostics.fail("pass at " + checkpoint);
    }
  }

private:
  Diagnostics &diagnostics;
  InvocationHandle handle;
  llvm::StringRef nextPhase = "start";
};

/// Write an artifact directly and remove it if emission or close fails.
template <typename Emit>
void writeArtifact(const fs::path &path, Diagnostics &diagnostics,
                   const llvm::Twine &stage, Emit emit) {
  std::error_code error;
  llvm::raw_fd_ostream out(path.string(), error);
  if (error) {
    diagnostics.fail(stage, error.message());
  }
  // A buffered write may first fail on close. Clear that error before the
  // LLVM stream destructor can turn it into a fatal usage error.
  const auto result = emit(out);
  out.close();
  error = out.error();
  if (failed(result) || error) {
    out.clear_error();
    std::error_code ignored;
    fs::remove(path, ignored);
    if (error) {
      diagnostics.fail(stage, error.message());
    }
    diagnostics.fail(stage);
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

struct PartitionResult {
  std::optional<std::string> sourceDevice;
  bool empty;
};

PartitionResult partitionInput(Invocation &invocation, const fs::path &dir,
                               const IreeCompileOptions &options,
                               Diagnostics &diagnostics) {
  auto original = invocation.takeModule();
  auto partition = diagnostics.require<UnsupportedPartition>(
      partitionTorchModule(*original, options.schema), "partition");
  writeModule(*partition.segments, dir / "partitioned.mlir", diagnostics);
  writeModule(*partition.driver, dir / "driver.mlir", diagnostics);
  bool empty = partition.segments->getBody()->empty();
  if (empty)
    return {std::move(partition.sourceDevice), true};
  invocation.putModule(std::move(partition.segments),
                       "verify partitioned module");
  return {std::move(partition.sourceDevice), false};
}

using Phase = mlir::iree_compiler::IREEVMPipelinePhase;

std::optional<StringLiteral>
checkpointPasses(Phase phase, IreeBackend backend,
                 std::optional<IreeTestPipeline> testPipeline) {
  const bool probes = testPipeline == IreeTestPipeline::Checkpoints;
  const bool cuda = backend == IreeBackend::CUDA;
  switch (phase) {
  case Phase::Start:
    if (probes) {
      return StringLiteral("kashi-verify-segments,kashi-probe-torch");
    }
    return StringLiteral("kashi-verify-segments");
  case Phase::Flow:
    if (testPipeline == IreeTestPipeline::FailAtFlow) {
      return StringLiteral("kashi-fail-flow-probe");
    }
    if (probes && cuda) {
      return StringLiteral("kashi-probe-flow,kashi-collide-flow-functions");
    }
    if (probes) {
      return StringLiteral("kashi-probe-flow");
    }
    break;
  case Phase::ExecutableConfigurations:
    if (probes && cuda) {
      return StringLiteral("kashi-probe-configurations");
    }
    break;
  case Phase::ExecutableTargets:
    // The API appends to the root pass manager. Wrap only nested operations:
    // an outer builtin.module would skip the root module.
    if (probes && cuda) {
      return StringLiteral(
          "hal.executable(hal.executable.variant(builtin.module("
          "kashi-probe-nvvm)))");
    }
    break;
  case Phase::VM:
    if (probes) {
      return StringLiteral("kashi-probe-vm");
    }
    break;
  default:
    break;
  }
  return std::nullopt;
}

std::vector<KernelIdentity> captureFlow(Invocation &invocation,
                                        const fs::path &dir,
                                        IreeBackend backend,
                                        Diagnostics &diagnostics) {
  auto flow = invocation.takeModule();
  // FX reassembly reads dispatch references from this snapshot. Save it before
  // renaming the in-memory symbols so those references match the manifest keys.
  writeArtifact(dir / "flow.mlir", diagnostics, "write Flow snapshot",
                [&](llvm::raw_ostream &out) {
                  flow->print(out);
                  return success();
                });
  writeModule(*flow, dir / "phases" / "module.flow.mlir", diagnostics);
  std::vector<KernelIdentity> identities;
  if (backend == IreeBackend::CUDA) {
    identities = diagnostics.require(identifyCudaKernels(*flow),
                                     "identify Flow exports");
  }
  invocation.putModule(std::move(flow), "verify Flow checkpoint");
  return identities;
}

void emitCudaManifest(Invocation &invocation, const fs::path &dir,
                      const std::vector<KernelIdentity> &identities,
                      Diagnostics &diagnostics) {
  auto hal = invocation.takeModule();
  if (failed(writeCudaManifest(*hal, dir, identities))) {
    diagnostics.fail("resolve linked CUDA exports");
  }
  invocation.putModule(std::move(hal), "verify linked HAL module");
}

void capturePhase(Invocation &invocation, const fs::path &path,
                  Diagnostics &diagnostics) {
  auto module = invocation.takeModule();
  // The launch ABI reader consumes textual operation forms; location suffixes
  // are not part of that ABI. Keep provenance on partitioned/driver MLIR.
  writeArtifact(path, diagnostics, "write " + path.filename().string(),
                [&](llvm::raw_ostream &out) {
                  module->print(out);
                  out << '\n';
                  return success();
                });
  invocation.putModule(std::move(module), "verify phase checkpoint");
}

void emitVmBytecode(Invocation &invocation, const fs::path &dir,
                    Diagnostics &diagnostics) {
  auto module = invocation.takeModule();
  writeArtifact(dir / "module.vmfb", diagnostics, "write VM bytecode",
                [&](llvm::raw_ostream &out) {
                  namespace VM = mlir::iree_compiler::IREE::VM;
                  return VM::translateModuleToBytecode(
                      *module, VM::TargetOptions{}, VM::BytecodeTargetOptions{},
                      out);
                });
}

void lowerWithCheckpoints(Invocation &invocation, const fs::path &dir,
                          IreeBackend backend,
                          std::optional<IreeTestPipeline> testPipeline,
                          Diagnostics &diagnostics) {
  std::vector<KernelIdentity> identities;
  mlir::iree_compiler::enumerateIREEVMPipelinePhases(
      [&](Phase phase, llvm::StringRef name, llvm::StringRef) {
        const auto pipeline = checkpointPasses(phase, backend, testPipeline);
        const bool flowSnapshot = phase == Phase::Flow;
        const bool phaseSnapshot =
            phase == Phase::Stream || phase == Phase::ExecutableTargets;
        const bool manifest =
            backend == IreeBackend::CUDA && phase == Phase::HAL;
        if (!pipeline && !flowSnapshot && !phaseSnapshot && !manifest) {
          return;
        }
        if (phase != Phase::Start) {
          invocation.lowerTo(name);
        }
        if (pipeline) {
          invocation.runPasses(name, *pipeline);
        }
        if (flowSnapshot) {
          identities = captureFlow(invocation, dir, backend, diagnostics);
        }
        if (phaseSnapshot) {
          capturePhase(invocation,
                       dir / "phases" / ("module." + name.str() + ".mlir"),
                       diagnostics);
        }
        if (manifest) {
          emitCudaManifest(invocation, dir, identities, diagnostics);
        }
      });
  invocation.lowerTo("end");
  emitVmBytecode(invocation, dir, diagnostics);
}

/// Partition and lower a Torch module in one private IREE invocation.
std::optional<std::string>
compile(const std::string &assembly, const IreeCompileOptions &options,
        std::optional<IreeTestPipeline> testPipeline) {
  static const CompilerRuntime runtime;
  const fs::path dir = fs::absolute(options.artifactDir);
  Diagnostics diagnostics(dir / "iree-compile.log");
  const auto &backend = getBackendConfig(options, diagnostics);
  if (!backend.architectureFlag && options.cudaTarget)
    diagnostics.fail("configuration", "cuda_target is required only for CUDA");
  // CPU-only IREE builds do not register CUDA flags. Parse and partition with
  // common flags first, so driver-only graphs need no CUDA backend at all.
  const auto session = createSession(diagnostics);
  const auto source = wrapSource(session, assembly, diagnostics);
  Invocation invocation(session, diagnostics);
  invocation.parse(source);
  auto partition = partitionInput(invocation, dir, options, diagnostics);
  if (partition.empty)
    return std::move(partition.sourceDevice);
  if (backend.architectureFlag && !options.cudaTarget)
    diagnostics.fail("configuration", "cuda_target is required only for CUDA");
  configureBackend(session, backend, options.cudaTarget, diagnostics);
  std::error_code error;
  fs::create_directories(dir / "phases", error);
  if (error)
    diagnostics.fail("create phase directory", error.message());
  lowerWithCheckpoints(invocation, dir, backend.backend, testPipeline,
                       diagnostics);
  return std::move(partition.sourceDevice);
}
} // namespace

std::optional<IreeBackend> parseIreeBackend(std::string_view target) {
  for (const auto &config : backendConfigs) {
    if (config.name == target) {
      return config.backend;
    }
  }
  return std::nullopt;
}

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
