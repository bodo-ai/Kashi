#include "kashi/cuda_manifest.h"
#include "kashi/iree_compiler.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <optional>
#include <random>
#include <string>
#include <string_view>

#ifdef __linux__
#include <signal.h>
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
using kashi::IreeBackend;

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

#ifdef KASHI_TEST_CUDA_CODEGEN
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

mlir::OwningOpRef<mlir::ModuleOp>
makeCudaExecutable(mlir::MLIRContext &context,
                   std::optional<std::string_view> ptx, uint32_t ordinal,
                   std::string_view kernelName = "kernel",
                   bool duplicateExport = false) {
  namespace HAL = mlir::iree_compiler::IREE::HAL;
  using mlir::iree_compiler::FlatbufferBuilder;
  context.loadDialect<HAL::HALDialect>();
  mlir::OpBuilder builder(&context);
  auto loc = builder.getUnknownLoc();
  mlir::OwningOpRef<mlir::ModuleOp> module(mlir::ModuleOp::create(loc));
  builder.setInsertionPointToStart(module->getBody());
  auto executable = HAL::ExecutableOp::create(builder, loc, "linked");
  builder.setInsertionPointToStart(&executable.getBlock());

  FlatbufferBuilder fb;
  REQUIRE(iree_hal_cuda_ExecutableDef_start_as_root(fb) == 0);
  auto ptxString = ptx ? fb.createString(*ptx) : 0;
  REQUIRE(iree_hal_cuda_ModuleDef_start(fb) == 0);
  if (ptx) {
    REQUIRE(ptxString != 0);
    REQUIRE(iree_hal_cuda_ModuleDef_ptx_image_add(fb, ptxString) == 0);
  }
  auto image = iree_hal_cuda_ModuleDef_end(fb);
  auto kernel = fb.createString(kernelName);
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
  auto exports = duplicateExport ? fb.createOffsetVec({entry, entry})
                                 : fb.createOffsetVec({entry});
  REQUIRE(iree_hal_cuda_ExecutableDef_modules_add(fb, modules) == 0);
  REQUIRE(iree_hal_cuda_ExecutableDef_exports_add(fb, exports) == 0);
  REQUIRE(iree_hal_cuda_ExecutableDef_end_as_root(fb) != 0);
  HAL::ExecutableBinaryOp::create(
      builder, loc, "cuda", builder.getStringAttr("cuda-nvptx-fb"),
      fb.getHeaderPrefixedBufferAttr(
          &context, iree_hal_cuda_ExecutableDef_file_identifier, 0));
  return module;
}

} // namespace

// Check that a deliberately failed pass leaves the next invocation usable.
TEST_CASE("A Flow pass failure does not poison the next embedded invocation") {
  Artifacts failed;
  Artifacts healthy;
  REQUIRE_THROWS_WITH(kashi::compileIreeForTesting(
                          fixture, {failed.directory, IreeBackend::VMVX, {}},
                          kashi::IreeTestPipeline::FailAtFlow),
                      Catch::Matchers::ContainsSubstring("pass at flow"));
  const auto log = failed.read("iree-compile.log");
  const auto message = log.find("Kashi Flow probe failure");
  REQUIRE(message != std::string::npos);
  const auto previousLine = log.rfind('\n', message);
  const auto lineStart =
      previousLine == std::string::npos ? 0 : previousLine + 1;
  REQUIRE(log.compare(lineStart, 7, "error: ") == 0);
  REQUIRE(!fs::exists(failed.directory / "flow.mlir"));
  kashi::compileIree(fixture, {healthy.directory, IreeBackend::VMVX, {}});
  REQUIRE(healthy.read("flow.mlir").find("flow.dispatch") != std::string::npos);
}

TEST_CASE("VMVX checkpoints produce a Flow snapshot without CUDA artifacts") {
  Artifacts artifacts;
  kashi::compileIreeForTesting(fixture,
                               {artifacts.directory, IreeBackend::VMVX, {}},
                               kashi::IreeTestPipeline::Checkpoints);
  REQUIRE(artifacts.read("flow.mlir").find("flow.dispatch") !=
          std::string::npos);
  REQUIRE(!fs::exists(artifacts.directory / "kernel_manifest.json"));
}

TEST_CASE(
    "Compiler reports configuration and artifact failures at their stage") {
  Artifacts artifacts;
  const auto dir = artifacts.directory;
  REQUIRE(kashi::parseIreeBackend("vmvx") == IreeBackend::VMVX);
  REQUIRE(kashi::parseIreeBackend("cuda") == IreeBackend::CUDA);
  REQUIRE_FALSE(kashi::parseIreeBackend("unknown"));

  SECTION("CUDA architecture cannot be passed to VMVX") {
    REQUIRE_THROWS_WITH(
        kashi::compileIree(fixture, {dir, IreeBackend::VMVX, "sm_80"}),
        Catch::Matchers::ContainsSubstring("configuration failed"));
    REQUIRE(artifacts.read("iree-compile.log")
                .find("cuda_target is required only for CUDA") !=
            std::string::npos);
  }
  SECTION("the diagnostic log cannot be opened") {
    fs::create_directory(dir / "iree-compile.log");
    REQUIRE_THROWS_WITH(
        kashi::compileIree(fixture, {dir, IreeBackend::VMVX, {}}),
        Catch::Matchers::ContainsSubstring("cannot open IREE diagnostic log"));
  }
  SECTION("the driver cannot be opened") {
    fs::create_directory(dir / "driver.mlir");
    REQUIRE_THROWS_WITH(
        kashi::compileIree(fixture, {dir, IreeBackend::VMVX, {}}),
        Catch::Matchers::ContainsSubstring("write driver.mlir failed"));
  }
  SECTION("the entry point is not supported") {
    REQUIRE_THROWS_AS(
        kashi::compileIree("module { func.func private @other() }",
                           {dir, IreeBackend::VMVX, {}}),
        kashi::UnsupportedPartition);
    REQUIRE(artifacts.read("iree-compile.log")
                .find("expected a single block in func.func @main") !=
            std::string::npos);
  }
  SECTION("the backend is not recognized") {
    REQUIRE_THROWS_WITH(
        kashi::compileIree(fixture,
                           {dir, kashi::parseIreeBackend("unknown"), {}}),
        Catch::Matchers::ContainsSubstring("configuration failed"));
    REQUIRE(artifacts.read("iree-compile.log")
                .find("expected target cuda or vmvx") != std::string::npos);
  }
}

TEST_CASE("CUDA manifest resolves entries in the selected serialized image") {
  constexpr auto entry = ".visible .entry/* split */\n kernel() { ret; }\n";
  struct Scenario {
    std::string_view ptx;
    uint32_t ordinal;
    bool resolves;
    std::string_view kernel = "kernel";
  };
  for (const auto &[ptx, ordinal, resolves, kernel] :
       {Scenario{entry, 0, true}, Scenario{entry, 1, false},
        Scenario{"/* .entry kernel() */\n.entry other() { ret; }\n", 0, false},
        Scenario{".file 1 \"a .entry kernel() b\"\n", 0, false},
        Scenario{".entrypoint kernel() {}", 0, false},
        Scenario{".entry x", 0, true, "x"},
        Scenario{".entry _x() {}", 0, true, "_x"},
        Scenario{".entry _() {}", 0, false, "_"},
        Scenario{"// .entry kernel() {}", 0, false},
        Scenario{".func other() {{ } .entry kernel() {}}", 0, false},
        Scenario{".func other() {{}} .entry kernel() {}", 0, true},
        Scenario{"} .entry kernel() {}", 0, true},
        Scenario{R"ptx(.file 1 "a\" .entry kernel() {}")ptx", 0, false},
        Scenario{R"ptx(.file 1 "a\\" .entry kernel() {})ptx", 0, true},
        Scenario{R"ptx(.file 1 "a .entry kernel() {}\)ptx", 0, false}}) {
    CAPTURE(ptx, ordinal, kernel);
    Artifacts artifacts;
    mlir::MLIRContext context;
    auto module = makeCudaExecutable(context, ptx, ordinal, kernel);

    std::string diagnostic;
    mlir::ScopedDiagnosticHandler handler(
        &context, [&](mlir::Diagnostic &d) { diagnostic = d.str(); });
    auto result = kashi::writeCudaManifest(
        *module, artifacts.directory, {{"original", "exported", "stable"}});
    REQUIRE(artifacts.read("kernel_0.ptx") == ptx);
    auto manifest = artifacts.directory / "kernel_manifest.json";
    if (resolves) {
      REQUIRE(mlir::succeeded(result));
      REQUIRE(diagnostic.empty());
      REQUIRE(fs::exists(manifest));
    } else {
      REQUIRE(mlir::failed(result));
      REQUIRE(diagnostic == "serialized CUDA export has no matching PTX entry");
      REQUIRE(!fs::exists(manifest));
    }
  }
}

TEST_CASE("CUDA manifest reports malformed serialized executable data") {
  namespace HAL = mlir::iree_compiler::IREE::HAL;
  Artifacts artifacts;
  mlir::MLIRContext context;
  auto module = makeCudaExecutable(context, ".entry kernel() {}", 0);
  auto executable = *module->getOps<HAL::ExecutableOp>().begin();
  auto binary = *executable.getOps<HAL::ExecutableBinaryOp>().begin();
  const auto data = mlir::cast<mlir::DenseIntElementsAttr>(binary.getData());
  const auto raw = data.getRawData();
  std::string expected;

  SECTION("data is not an i8 array") {
    binary.setDataAttr(mlir::Builder(&context).getI32TensorAttr({0, 1}));
    expected = "expected a byte array for the CUDA executable";
  }
  SECTION("invalid outer header magic") {
    std::string bytes(raw.begin(), raw.end());
    bytes.front() = '\0';
    binary.setDataAttr(mlir::DenseElementsAttr::getFromRawBuffer(
        data.getType(), {bytes.data(), bytes.size()}));
    expected = "invalid serialized CUDA executable header";
  }
  SECTION("valid header but out-of-bounds FlatBuffer root offset") {
    std::string bytes(raw.begin(), raw.end());
    bytes.replace(sizeof(iree_flatbuffer_file_header_t), sizeof(uint32_t),
                  sizeof(uint32_t), '\xff');
    binary.setDataAttr(mlir::DenseElementsAttr::getFromRawBuffer(
        data.getType(), {bytes.data(), bytes.size()}));
    expected = "invalid serialized CUDA executable FlatBuffer";
  }
  SECTION("image table has no PTX string") {
    auto missing = makeCudaExecutable(context, std::nullopt, 0);
    auto linked = *missing->getOps<HAL::ExecutableOp>().begin();
    auto missingBinary = *linked.getOps<HAL::ExecutableBinaryOp>().begin();
    binary.setDataAttr(missingBinary.getData());
    expected = "missing serialized CUDA PTX image";
  }

  std::string diagnostic;
  mlir::ScopedDiagnosticHandler handler(
      &context, [&](mlir::Diagnostic &d) { diagnostic = d.str(); });
  REQUIRE(
      mlir::failed(kashi::writeCudaManifest(*module, artifacts.directory, {})));
  CHECK(diagnostic == expected);
  CHECK(!fs::exists(artifacts.directory / "kernel_0.ptx"));
  CHECK(!fs::exists(artifacts.directory / "kernel_manifest.json"));
}

TEST_CASE("CUDA manifest errors identify the failed join or output") {
  Artifacts artifacts;
  mlir::MLIRContext context;
  bool duplicate = false;
  std::string expected;
  SECTION("duplicate serialized export") {
    duplicate = true;
    expected = "ambiguous serialized CUDA export: stable";
  }
  SECTION("Flow export is unmatched") {
    expected = "missing serialized CUDA export for original::exported";
  }
  SECTION("PTX destination cannot be opened") {
    REQUIRE(fs::create_directory(artifacts.directory / "kernel_0.ptx"));
    expected = "cannot open " + (artifacts.directory / "kernel_0.ptx").string();
  }
#ifdef __linux__
  SECTION("PTX destination cannot be written") {
    fs::create_symlink("/dev/full", artifacts.directory / "kernel_0.ptx");
    expected =
        "cannot write " + (artifacts.directory / "kernel_0.ptx").string();
  }
#endif
  auto module =
      makeCudaExecutable(context, ".entry kernel() {}", 0, "kernel", duplicate);
  std::string diagnostic;
  mlir::ScopedDiagnosticHandler handler(
      &context, [&](mlir::Diagnostic &d) { diagnostic = d.str(); });
  REQUIRE(mlir::failed(kashi::writeCudaManifest(
      *module, artifacts.directory, {{"original", "exported", "missing"}})));
  CHECK(diagnostic.compare(0, expected.size(), expected) == 0);
  CHECK(!fs::exists(artifacts.directory / "kernel_manifest.json"));
}

#ifdef __linux__
// Force a real write failure at the Flow snapshot stage.
TEST_CASE("Compiler artifact write errors are recoverable") {
  constexpr auto stage = "write Flow snapshot";
  Artifacts broken;
  fs::create_symlink("/dev/full", broken.directory / "flow.mlir");
  REQUIRE_THROWS_WITH(
      kashi::compileIree(fixture, {broken.directory, IreeBackend::VMVX, {}}),
      Catch::Matchers::ContainsSubstring(stage));
  REQUIRE(broken.read("iree-compile.log").find(stage) != std::string::npos);
  REQUIRE(fs::exists(broken.directory / "driver.mlir"));
  Artifacts healthy;
  kashi::compileIree(fixture, {healthy.directory, IreeBackend::VMVX, {}});
  REQUIRE(healthy.read("flow.mlir").find("flow.dispatch") != std::string::npos);
}

// Inspect the OS disposition rather than Python's cached signal handler.
TEST_CASE("Compiler artifact writes preserve the host signal handler") {
  struct sigaction before, after;
  REQUIRE(sigaction(SIGTERM, nullptr, &before) == 0);
  Artifacts artifacts;
  kashi::compileIree(fixture, {artifacts.directory, IreeBackend::VMVX, {}});
  REQUIRE(sigaction(SIGTERM, nullptr, &after) == 0);
  REQUIRE(after.sa_flags == before.sa_flags);
  if (before.sa_flags & SA_SIGINFO) {
    REQUIRE(after.sa_sigaction == before.sa_sigaction);
  } else {
    REQUIRE(after.sa_handler == before.sa_handler);
  }
}

// An opened but unwritable log must not be presented as a diagnostic source.
TEST_CASE("Compiler errors identify an unwritable diagnostic log") {
  Artifacts artifacts;
  fs::create_symlink("/dev/full", artifacts.directory / "iree-compile.log");
  REQUIRE_THROWS_WITH(
      kashi::compileIree("module { invalid",
                         {artifacts.directory, IreeBackend::VMVX, {}}),
      Catch::Matchers::ContainsSubstring(
          "parse failed; cannot write diagnostic log"));
}
#endif

#ifdef KASHI_TEST_CUDA_CODEGEN
// Colliding local names must link to distinct entries.
TEST_CASE("CUDA checkpoints preserve colliding Flow export identities") {
  Artifacts split;
  kashi::compileIreeForTesting(twoSegments,
                               {split.directory, IreeBackend::CUDA, "sm_80"},
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
