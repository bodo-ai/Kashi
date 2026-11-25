#include "kashi/iree_partition.h"

#include <string>
#include <utility>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

namespace {
using namespace mlir;
using Catch::Matchers::ContainsSubstring;

struct PartitionFixture {
  MLIRContext context;

  PartitionFixture() {
    context.loadDialect<func::FuncDialect, torch::Torch::TorchDialect>();
  }

  OwningOpRef<ModuleOp> parse(llvm::StringRef text) {
    auto module = parseSourceString<ModuleOp>(text, &context);
    REQUIRE(static_cast<bool>(module));
    return module;
  }

  kashi::Partition partition(ModuleOp module) {
    auto result = kashi::partitionTorchModule(module);
    REQUIRE(succeeded(result));
    // FailureOr hides has_value(); Catch aborts this test on failure above.
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    return std::move(*result);
  }

  void reject(llvm::StringRef text, const char *expected) {
    auto module = parse(text);
    std::string diagnostics;
    ScopedDiagnosticHandler handler(&context, [&](Diagnostic &diagnostic) {
      llvm::raw_string_ostream stream(diagnostics);
      diagnostic.print(stream);
      stream << '\n';
      return success();
    });
    CHECK(failed(kashi::partitionTorchModule(*module)));
    CHECK_THAT(diagnostics, ContainsSubstring(expected));
  }
};

std::string print(ModuleOp module) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  module.print(stream);
  return text;
}
} // namespace

TEST_CASE_METHOD(PartitionFixture,
                 "Allocation stays in the driver with the original device",
                 "[partition]") {
  auto source = parse(R"mlir(
module {
  func.func @main() -> !torch.vtensor<[4],f32> {
    %four = torch.constant.int 4
    %dtype = torch.constant.int 6
    %none = torch.constant.none
    %cuda = torch.constant.device "cuda:2"
    %shape = torch.prim.ListConstruct %four : (!torch.int) -> !torch.list<int>
    %buffer = torch.aten.empty.memory_format %shape, %dtype, %none, %cuda, %none, %none : !torch.list<int>, !torch.int, !torch.none, !torch.Device, !torch.none, !torch.none -> !torch.vtensor<[4],f32>
    return %buffer : !torch.vtensor<[4],f32>
  }
}
)mlir");
  const auto before = print(*source);
  auto result = partition(*source);
  CHECK(succeeded(verify(*result.segments)));
  CHECK(succeeded(verify(*result.driver)));
  CHECK(result.sourceDevice == "cuda:2");
  CHECK(print(*source) == before);
  CHECK(result.segments->getBody()->empty());
  CHECK_THAT(print(*result.driver),
             ContainsSubstring("torch.aten.empty.memory_format"));
  CHECK_THAT(print(*result.driver),
             ContainsSubstring("torch.constant.device \"cuda:2\""));
}

TEST_CASE_METHOD(PartitionFixture,
                 "Partition preserves a non-CUDA device clone", "[partition]") {
  auto source = parse(R"mlir(
module {
  func.func @main() -> !torch.vtensor<[4],f32> {
    %four = torch.constant.int 4
    %dtype = torch.constant.int 6
    %none = torch.constant.none
    %device = torch.constant.device "meta"
    %shape = torch.prim.ListConstruct %four : (!torch.int) -> !torch.list<int>
    %buffer = torch.aten.empty.memory_format %shape, %dtype, %none, %device, %none, %none : !torch.list<int>, !torch.int, !torch.none, !torch.Device, !torch.none, !torch.none -> !torch.vtensor<[4],f32>
    return %buffer : !torch.vtensor<[4],f32>
  }
}
)mlir");
  auto result = partition(*source);
  CHECK(result.sourceDevice == "meta");
  CHECK(result.segments->getBody()->empty());
  CHECK_THAT(print(*result.driver),
             ContainsSubstring("torch.constant.device \"meta\""));
}

TEST_CASE_METHOD(
    PartitionFixture,
    "Partition clones transitive list constants once in definition order",
    "[partition]") {
  auto source = parse(R"mlir(
module {
  func.func @main() -> !torch.list<int> {
    %one = torch.constant.int 1
    %row = torch.prim.ListConstruct %one, %one : (!torch.int, !torch.int) -> !torch.list<int>
    %rows = torch.prim.ListConstruct %row, %row : (!torch.list<int>, !torch.list<int>) -> !torch.list<list<int>>
    %first, %second = torch.prim.ListUnpack %rows : !torch.list<list<int>> -> !torch.list<int>, !torch.list<int>
    return %second : !torch.list<int>
  }
}
)mlir");
  auto result = partition(*source);
  CHECK_FALSE(result.sourceDevice.has_value());
  CHECK(succeeded(verify(*result.segments)));
  CHECK(succeeded(verify(*result.driver)));
  auto segment = result.segments->lookupSymbol<func::FuncOp>("segment_0");
  REQUIRE(segment);
  CHECK(segment.getNumArguments() == 0);
  CHECK(segment.getFunctionType().getNumResults() == 1);
  auto &body = segment.getBody().front();
  REQUIRE(llvm::range_size(body) == 5);
  auto op = body.begin();
  Value one = op++->getResult(0);
  Operation &row = *op++;
  REQUIRE(row.getNumOperands() == 2);
  CHECK(row.getOperand(0) == one);
  CHECK(row.getOperand(1) == one);
  Operation &rows = *op++;
  REQUIRE(rows.getNumOperands() == 2);
  CHECK(rows.getOperand(0) == row.getResult(0));
  CHECK(rows.getOperand(1) == row.getResult(0));
  auto &unpack = *op++;
  CHECK(unpack.getOperand(0) == rows.getResult(0));
  CHECK(op->getOperand(0) == unpack.getResult(1));
}

TEST_CASE_METHOD(PartitionFixture,
                 "Partition rejects unsupported types crossing its ABI",
                 "[partition]") {
  SECTION("live in") {
    reject(R"mlir(
module {
  func.func @main(%f: !torch.float) -> !torch.list<float> {
    %list = torch.prim.ListConstruct %f : (!torch.float) -> !torch.list<float>
    return %list : !torch.list<float>
  }
}
)mlir",
           "unsupported live-in for segment_0");
  }
  SECTION("live out with both a local and an external user") {
    reject(R"mlir(
module {
  func.func @main(%x: !torch.vtensor<[],f32>) -> !torch.list<float> {
    %f = torch.aten.Float.Tensor %x : !torch.vtensor<[],f32> -> !torch.float
    %list = torch.prim.ListConstruct %f : (!torch.float) -> !torch.list<float>
    torch.operator "torch.test.scalar"(%f) : (!torch.float) -> ()
    return %list : !torch.list<float>
  }
}
)mlir",
           "unsupported live-out for segment_0");
  }
}

TEST_CASE_METHOD(PartitionFixture,
                 "Partition validates the source module shape and operations",
                 "[partition]") {
  SECTION("empty module") {
    reject("module {}", "expected a module containing only func.func @main");
  }
  SECTION("extra function") {
    reject(
        "module { func.func @main() { return } func.func @other() { return } }",
        "expected a module containing only func.func @main");
  }
  SECTION("non-function top-level operation") {
    reject("module { module {} }",
           "expected a single block in func.func @main");
  }
  SECTION("wrong entry name") {
    reject("module { func.func @other() { return } }",
           "expected a single block in func.func @main");
  }
  SECTION("declaration without a body") {
    reject("module { func.func private @main() }",
           "expected a single block in func.func @main");
  }
  SECTION("nested region") {
    context.allowUnregisteredDialects();
    reject("module { func.func @main() { \"test.region\"() ({}) : () -> () "
           "return } }",
           "nested regions are not supported");
  }
  SECTION("non-Torch source operation") {
    reject(
        "module { func.func @main() { func.call @main() : () -> () return } }",
        "unsupported source operation: func.call");
  }
  SECTION("conflicting devices, even if unused") {
    reject(R"mlir(
module {
  func.func @main() {
    %cuda = torch.constant.device "cuda:2"
    %cpu = torch.constant.device "cpu"
    return
  }
}
)mlir",
           "multiple source devices are unsupported: cuda:2 and cpu");
  }
}

TEST_CASE_METHOD(PartitionFixture,
                 "Partition skips leading adjacent and trailing empty runs",
                 "[partition]") {
  auto source = parse(R"mlir(
module {
  func.func @main(%x: !torch.vtensor<[4],f32>) -> !torch.vtensor<[4],f32> {
    %a = torch.operator "torch.test.first"(%x) : (!torch.vtensor<[4],f32>) -> !torch.vtensor<[4],f32>
    %b = torch.operator "torch.test.second"(%a) : (!torch.vtensor<[4],f32>) -> !torch.vtensor<[4],f32>
    %c = torch.aten.relu %b : !torch.vtensor<[4],f32> -> !torch.vtensor<[4],f32>
    %d = torch.operator "torch.test.last"(%c) : (!torch.vtensor<[4],f32>) -> !torch.vtensor<[4],f32>
    return %d : !torch.vtensor<[4],f32>
  }
}
)mlir");
  auto result = partition(*source);
  CHECK(succeeded(verify(*result.segments)));
  CHECK(succeeded(verify(*result.driver)));
  CHECK(llvm::hasSingleElement(result.segments->getBody()->getOperations()));
  auto segment = result.segments->lookupSymbol<func::FuncOp>("segment_2");
  REQUIRE(segment);
  CHECK(llvm::range_size(result.driver->getBody()->getOperations()) == 2);
  auto declaration = result.driver->lookupSymbol<func::FuncOp>("segment_2");
  REQUIRE(declaration);
  CHECK(declaration.isPrivate());
  CHECK(declaration.isExternal());
  CHECK(declaration.getFunctionType() == segment.getFunctionType());

  auto main = result.driver->lookupSymbol<func::FuncOp>("main");
  REQUIRE(main);
  auto &body = main.getBody().front();
  REQUIRE(llvm::range_size(body) == 5);
  auto op = body.begin();
  Operation &first = *op++;
  Operation &second = *op++;
  auto call = dyn_cast<func::CallOp>(&*op++);
  REQUIRE(call);
  CHECK(call.getCallee() == "segment_2");
  Operation &last = *op++;
  CHECK(first.getOperand(0) == body.getArgument(0));
  CHECK(second.getOperand(0) == first.getResult(0));
  CHECK(call.getOperand(0) == second.getResult(0));
  CHECK(last.getOperand(0) == call.getResult(0));
  CHECK(op->getOperand(0) == last.getResult(0));
}

TEST_CASE_METHOD(PartitionFixture,
                 "View and split preserve storage across mutation",
                 "[partition]") {
  auto source = parse(R"mlir(
module {
  func.func @main(%x: !torch.vtensor<[4],f32>) -> (!torch.vtensor<[4],f32>, !torch.vtensor<[2],f32>) {
    %two = torch.constant.int 2
    %zero = torch.constant.int 0
    %sizes = torch.prim.ListConstruct %two, %two : (!torch.int, !torch.int) -> !torch.list<int>
    %parts = torch.aten.split_with_sizes %x, %sizes, %zero : !torch.vtensor<[4],f32>, !torch.list<int>, !torch.int -> !torch.list<vtensor>
    %left, %right = torch.prim.ListUnpack %parts : !torch.list<vtensor> -> !torch.vtensor<[2],f32>, !torch.vtensor<[2],f32>
    torch.operator "torch.test.mutate"(%left) : (!torch.vtensor<[2],f32>) -> ()
    return %x, %left : !torch.vtensor<[4],f32>, !torch.vtensor<[2],f32>
  }
}
)mlir");
  auto result = partition(*source);
  CHECK(result.segments->getBody()->empty());
  auto text = print(*result.driver);
  CHECK_THAT(text, ContainsSubstring("torch.aten.split_with_sizes"));
  CHECK_THAT(text, ContainsSubstring("torch.prim.ListUnpack"));
  CHECK_THAT(text, ContainsSubstring("torch.test.mutate"));
  auto main = result.driver->lookupSymbol<func::FuncOp>("main");
  REQUIRE(main);
  auto &body = main.getBody().front();
  Operation *split = nullptr;
  Operation *unpack = nullptr;
  Operation *mutation = nullptr;
  for (Operation &op : body) {
    auto name = op.getName().getStringRef();
    if (name == "torch.aten.split_with_sizes")
      split = &op;
    else if (name == "torch.prim.ListUnpack")
      unpack = &op;
    else if (name == "torch.operator")
      mutation = &op;
  }
  REQUIRE(split);
  REQUIRE(unpack);
  REQUIRE(mutation);
  CHECK(split->getOperand(0) == body.getArgument(0));
  CHECK(unpack->getOperand(0) == split->getResult(0));
  CHECK(mutation->getOperand(0) == unpack->getResult(0));
  auto &ret = body.back();
  CHECK(ret.getOperand(0) == body.getArgument(0));
  CHECK(ret.getOperand(1) == unpack->getResult(0));
  CHECK(succeeded(verify(*result.driver)));
}

TEST_CASE_METHOD(PartitionFixture,
                 "Dynamic shapes and lists stay in the driver", "[partition]") {
  auto source = parse(R"mlir(
module {
  func.func @main(%x: !torch.vtensor<[?],f32>, %dim: !torch.int) -> !torch.vtensor<[?],f32> {
    %size = torch.aten.size.int %x, %dim : !torch.vtensor<[?],f32>, !torch.int -> !torch.int
    %shape = torch.prim.ListConstruct %size : (!torch.int) -> !torch.list<int>
    %view = torch.aten.view %x, %shape : !torch.vtensor<[?],f32>, !torch.list<int> -> !torch.vtensor<[?],f32>
    return %view : !torch.vtensor<[?],f32>
  }
}
)mlir");
  auto result = partition(*source);
  CHECK(result.segments->getBody()->empty());
  auto text = print(*result.driver);
  CHECK_THAT(text, ContainsSubstring("torch.aten.size.int"));
  CHECK_THAT(text, ContainsSubstring("torch.prim.ListConstruct"));
  CHECK_THAT(text, ContainsSubstring("torch.aten.view"));
  CHECK(succeeded(verify(*result.driver)));
}

TEST_CASE_METHOD(PartitionFixture, "An empty graph has no outlined segments",
                 "[partition]") {
  auto source = parse("module { func.func @main() { return } }");
  auto result = partition(*source);
  CHECK(result.segments->getBody()->empty());
  CHECK(succeeded(verify(*result.driver)));
}

TEST_CASE_METHOD(PartitionFixture,
                 "Schema metadata is cached and controls boundaries",
                 "[partition]") {
  auto source = parse(R"mlir(
module {
  func.func @main(%x: !torch.vtensor<[4],f32>) -> !torch.vtensor<[4],f32> {
    %a = torch.aten.relu %x : !torch.vtensor<[4],f32> -> !torch.vtensor<[4],f32>
    %b = torch.aten.relu %a : !torch.vtensor<[4],f32> -> !torch.vtensor<[4],f32>
    return %b : !torch.vtensor<[4],f32>
  }
}
)mlir");
  int queries = 0;
  auto result =
      kashi::partitionTorchModule(*source, [&](std::string_view name) {
        CHECK(name == "torch.aten.relu");
        ++queries;
        return std::optional<bool>(true);
      });
  REQUIRE(succeeded(result));
  CHECK(queries == 1);
  // FailureOr hides has_value(); REQUIRE above aborts if the value is absent.
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
  auto partition = std::move(*result);
  CHECK(partition.segments->getBody()->empty());
  CHECK(succeeded(verify(*partition.driver)));

  std::string diagnostics;
  ScopedDiagnosticHandler handler(&context, [&](Diagnostic &diagnostic) {
    diagnostics = diagnostic.str();
    return success();
  });
  auto unknown = kashi::partitionTorchModule(
      *source,
      [](std::string_view) -> std::optional<bool> { return std::nullopt; });
  CHECK(failed(unknown));
  CHECK_THAT(diagnostics, ContainsSubstring("unknown ATen schema"));
}

TEST_CASE_METHOD(PartitionFixture, "Clone remains a driver allocation",
                 "[partition]") {
  auto source = parse(R"mlir(
module {
  func.func @main(%x: !torch.vtensor<[4],f32>) -> !torch.vtensor<[4],f32> {
    %none = torch.constant.none
    %copy = torch.aten.clone %x, %none : !torch.vtensor<[4],f32>, !torch.none -> !torch.vtensor<[4],f32>
    return %copy : !torch.vtensor<[4],f32>
  }
}
)mlir");
  auto result = partition(*source);
  CHECK(result.segments->getBody()->empty());
  CHECK_THAT(print(*result.driver), ContainsSubstring("torch.aten.clone"));
  CHECK(succeeded(verify(*result.driver)));
}
