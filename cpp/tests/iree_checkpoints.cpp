#include "kashi/cuda_manifest.h"
#include "kashi/iree_compiler.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <random>
#include <string>
#include <string_view>
#include <utility>

#ifdef __linux__
#include <cstdlib>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Utils/FlatbufferUtils.h"
#include "iree/schemas/cuda_executable_def_builder.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"

namespace {
namespace fs = std::filesystem;

struct Artifacts {
  fs::path directory =
      fs::temp_directory_path() /
      ("kashi-checkpoints-" + std::to_string(std::random_device{}()));
  /// Create an isolated directory for the invocation's artifacts.
  Artifacts() { REQUIRE(fs::create_directory(directory)); }
  /// Remove test artifacts without throwing during failure unwinding.
  ~Artifacts() {
    std::error_code error;
    fs::remove_all(directory, error);
  }
  /// Read an entire artifact, including binary VM bytecode.
  std::string read(const char *file) const {
    std::ifstream input(directory / file, std::ios::binary);
    REQUIRE(input.good());
    return {std::istreambuf_iterator<char>(input),
            std::istreambuf_iterator<char>()};
  }
};

constexpr auto fixture = R"mlir(
module {
  func.func @main(%x: !torch.vtensor<[1024],f32>) -> !torch.vtensor<[1024],f32> {
    %y = torch.aten.relu %x : !torch.vtensor<[1024],f32> -> !torch.vtensor<[1024],f32>
    %z = torch.aten.neg %y : !torch.vtensor<[1024],f32> -> !torch.vtensor<[1024],f32>
    return %z : !torch.vtensor<[1024],f32>
  }
}
)mlir";

#ifdef __linux__
constexpr auto twoSegments = R"mlir(
module {
  func.func @main(%x: !torch.vtensor<[1024],f32>) -> !torch.vtensor<[1024],f32> {
    %y = torch.aten.relu %x : !torch.vtensor<[1024],f32> -> !torch.vtensor<[1024],f32>
    torch.operator "torch.test.barrier"(%y) : (!torch.vtensor<[1024],f32>) -> ()
    %z = torch.aten.neg %y : !torch.vtensor<[1024],f32> -> !torch.vtensor<[1024],f32>
    return %z : !torch.vtensor<[1024],f32>
  }
}
)mlir";
#endif
} // namespace

// Test that resuming at typed checkpoints is byte-for-byte transparent.
TEST_CASE("Typed checkpoint probes preserve VMVX compilation output") {
  Artifacts split;
  Artifacts uninterrupted;
  kashi::compileIreeForTesting(fixture, {split.directory, "vmvx", {}},
                               kashi::IreeTestPipeline::Checkpoints);
  kashi::compileIreeForTesting(fixture, {uninterrupted.directory, "vmvx", {}},
                               kashi::IreeTestPipeline::Uninterrupted);
  REQUIRE(split.read("module.vmfb") == uninterrupted.read("module.vmfb"));
  REQUIRE(split.read("flow.mlir").find("flow.dispatch") != std::string::npos);
}

// Check that a deliberately failed pass leaves the next invocation usable.
TEST_CASE("A Flow pass failure does not poison the next embedded invocation") {
  Artifacts failed;
  Artifacts healthy;
  REQUIRE_THROWS_WITH(
      kashi::compileIreeForTesting(fixture, {failed.directory, "vmvx", {}},
                                   kashi::IreeTestPipeline::FailAtFlow),
      Catch::Matchers::ContainsSubstring("pass at flow"));
  const auto log = failed.read("iree-compile.log");
  const auto message = log.find("Kashi Flow probe failure");
  REQUIRE(message != std::string::npos);
  const auto previousLine = log.rfind('\n', message);
  const auto lineStart =
      previousLine == std::string::npos ? 0 : previousLine + 1;
  REQUIRE(log.compare(lineStart, 7, "error: ") == 0);
  REQUIRE_FALSE(fs::exists(failed.directory / "module.vmfb"));
  kashi::compileIree(fixture, {healthy.directory, "vmvx", {}});
  REQUIRE(healthy.read("module.vmfb").size() > 0);
}

// Exercise the exact image-count boundary using a serialized CUDA export.
TEST_CASE("CUDA manifest validates serialized export module ordinals") {
  namespace HAL = mlir::iree_compiler::IREE::HAL;
  using mlir::iree_compiler::FlatbufferBuilder;

  constexpr char ptx[] = ".visible .entry kernel() { ret; }\n";
  for (uint32_t ordinal : {0u, 1u}) {
    CAPTURE(ordinal);
    Artifacts artifacts;
    mlir::MLIRContext context;
    context.loadDialect<HAL::HALDialect>();
    mlir::OpBuilder builder(&context);
    auto loc = builder.getUnknownLoc();
    mlir::OwningOpRef<mlir::ModuleOp> module(mlir::ModuleOp::create(loc));
    builder.setInsertionPointToStart(module->getBody());
    auto executable = HAL::ExecutableOp::create(builder, loc, "linked");
    builder.setInsertionPointToStart(&executable.getBlock());

    FlatbufferBuilder fb;
    REQUIRE(iree_hal_cuda_ExecutableDef_start_as_root(fb) == 0);
    auto image = iree_hal_cuda_ModuleDef_create(fb, fb.createString(ptx));
    auto kernel = fb.createString("kernel");
    auto name = fb.createString("stable");
    iree_hal_cuda_BlockDims_t dims{1, 1, 1};
    // Omit optional offsets rather than passing null offsets to create().
    REQUIRE(iree_hal_cuda_ExportDef_start(fb) == 0);
    REQUIRE(iree_hal_cuda_ExportDef_module_ordinal_add(fb, ordinal) == 0);
    REQUIRE(iree_hal_cuda_ExportDef_kernel_name_add(fb, kernel) == 0);
    REQUIRE(iree_hal_cuda_ExportDef_block_dims_add(fb, &dims) == 0);
    REQUIRE(iree_hal_cuda_ExportDef_export_name_add(fb, name) == 0);
    auto entry = iree_hal_cuda_ExportDef_end(fb);
    REQUIRE(image != 0);
    REQUIRE(entry != 0);
    auto modules = fb.createOffsetVec({image});
    auto exports = fb.createOffsetVec({entry});
    REQUIRE(iree_hal_cuda_ExecutableDef_modules_add(fb, modules) == 0);
    REQUIRE(iree_hal_cuda_ExecutableDef_exports_add(fb, exports) == 0);
    REQUIRE(iree_hal_cuda_ExecutableDef_end_as_root(fb) != 0);
    HAL::ExecutableBinaryOp::create(
        builder, loc, "cuda", builder.getStringAttr("cuda-nvptx-fb"),
        fb.getHeaderPrefixedBufferAttr(
            &context, iree_hal_cuda_ExecutableDef_file_identifier, 0));

    std::string diagnostic;
    mlir::ScopedDiagnosticHandler handler(
        &context, [&](mlir::Diagnostic &d) { diagnostic = d.str(); });
    auto result = kashi::writeCudaManifest(
        *module, artifacts.directory, {{"original", "exported", "stable"}});
    REQUIRE(artifacts.read("kernel_0.ptx") == ptx);
    auto manifest = artifacts.directory / "kernel_manifest.json";
    if (ordinal == 0) {
      REQUIRE(mlir::succeeded(result));
      REQUIRE(diagnostic.empty());
      REQUIRE(fs::exists(manifest));
    } else {
      REQUIRE(mlir::failed(result));
      REQUIRE(diagnostic == "serialized CUDA export has no matching PTX entry");
      REQUIRE_FALSE(fs::exists(manifest));
    }
  }
}

#ifdef __linux__
// Exit normally in the child so the test also exercises static destruction.
TEST_CASE("A forked child cannot use or shut down the parent compiler") {
  Artifacts warm;
  Artifacts child;
  Artifacts healthy;
  kashi::compileIree(twoSegments, {warm.directory, "vmvx", {}});
  const pid_t pid = fork();
  REQUIRE(pid >= 0);
  if (pid == 0) {
    alarm(5);
    try {
      kashi::compileIree(twoSegments, {child.directory, "vmvx", {}});
    } catch (const std::runtime_error &error) {
      std::exit(std::string_view(error.what()).find("spawn or exec") ==
                std::string_view::npos);
    }
    std::exit(2);
  }
  int status;
  REQUIRE(waitpid(pid, &status, 0) == pid);
  REQUIRE(WIFEXITED(status));
  REQUIRE(WEXITSTATUS(status) == 0);
  kashi::compileIree(fixture, {healthy.directory, "vmvx", {}});
  REQUIRE(healthy.read("module.vmfb").size() > 0);
}

// Force real write failures at the Flow snapshot and VM bytecode stages.
TEST_CASE("Compiler artifact write errors are recoverable") {
  for (const auto &[file, stage] :
       {std::pair{"flow.mlir", "write Flow snapshot"},
        std::pair{"module.vmfb", "serialize VM bytecode"}}) {
    CAPTURE(file);
    Artifacts broken;
    fs::create_symlink("/dev/full", broken.directory / file);
    REQUIRE_THROWS_WITH(
        kashi::compileIree(fixture, {broken.directory, "vmvx", {}}),
        Catch::Matchers::ContainsSubstring(stage));
    REQUIRE(broken.read("iree-compile.log").find(stage) != std::string::npos);
    REQUIRE(fs::exists(broken.directory / "driver.mlir"));
    Artifacts healthy;
    kashi::compileIree(fixture, {healthy.directory, "vmvx", {}});
    REQUIRE(healthy.read("module.vmfb").size() > 0);
  }
}

// Inspect the OS disposition rather than Python's cached signal handler.
TEST_CASE("Compiler artifact writes preserve the host signal handler") {
  struct sigaction before, after;
  REQUIRE(sigaction(SIGTERM, nullptr, &before) == 0);
  Artifacts artifacts;
  kashi::compileIree(fixture, {artifacts.directory, "vmvx", {}});
  REQUIRE(sigaction(SIGTERM, nullptr, &after) == 0);
  REQUIRE(after.sa_flags == before.sa_flags);
  if (before.sa_flags & SA_SIGINFO)
    REQUIRE(after.sa_sigaction == before.sa_sigaction);
  else
    REQUIRE(after.sa_handler == before.sa_handler);
}

// An opened but unwritable log must not be presented as a diagnostic source.
TEST_CASE("Compiler errors identify an unwritable diagnostic log") {
  Artifacts artifacts;
  fs::create_symlink("/dev/full", artifacts.directory / "iree-compile.log");
  REQUIRE_THROWS_WITH(
      kashi::compileIree("module { invalid", {artifacts.directory, "vmvx", {}}),
      Catch::Matchers::ContainsSubstring(
          "parse failed; cannot write diagnostic log"));
}

// Colliding local names must link to distinct entries.
TEST_CASE("CUDA checkpoints preserve colliding Flow export identities") {
  Artifacts split;
  kashi::compileIreeForTesting(twoSegments, {split.directory, "cuda", "sm_80"},
                               kashi::IreeTestPipeline::Checkpoints);
  auto flow = split.read("flow.mlir");
  auto first = flow.find("func.func @shared_kernel");
  REQUIRE(first != std::string::npos);
  REQUIRE(flow.find("func.func @shared_kernel", first + 1) !=
          std::string::npos);
  auto manifest = split.read("kernel_manifest.json");
  REQUIRE(manifest.find("kashi_kernel_0") != std::string::npos);
  REQUIRE(manifest.find("kashi_kernel_1") != std::string::npos);
  auto ptx = split.read("kernel_0.ptx");
  REQUIRE(ptx.find(".entry kashi_kernel_0(") != std::string::npos);
  REQUIRE(ptx.find(".entry kashi_kernel_1(") != std::string::npos);
}
#endif
