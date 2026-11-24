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

TEST_CASE_METHOD(
    PartitionFixture,
    "Partition clones only needed constants and rewrites only CUDA clones",
    "[partition]") {
  auto source = parse(R"mlir(
module {
  func.func @main() -> !torch.vtensor<[4],f32> {
    %four = torch.constant.int 4
    %unused = torch.constant.int 99
    %dtype = torch.constant.int 6
    %none = torch.constant.none
    %cuda = torch.constant.device "cuda:2"
    %same = torch.constant.device "cuda:2"
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
  CHECK_THAT(print(*result.driver),
             ContainsSubstring("torch.constant.device \"cuda:2\""));

  auto segment = result.segments->lookupSymbol<func::FuncOp>("segment_0");
  REQUIRE(segment);
  CHECK(segment.getNumArguments() == 0);
  auto &body = segment.getBody().front();
  REQUIRE(llvm::range_size(body) == 7);
  auto op = body.begin();
  Value four = op++->getResult(0);
  Value dtype = op++->getResult(0);
  Value none = op++->getResult(0);
  Operation &device = *op++;
  auto deviceName = device.getAttrOfType<StringAttr>("value");
  REQUIRE(deviceName);
  CHECK(deviceName.getValue() == "cpu");
  Operation &shape = *op++;
  REQUIRE(shape.getNumOperands() == 1);
  CHECK(shape.getOperand(0) == four);
  Operation &buffer = *op++;
  REQUIRE(buffer.getNumOperands() == 6);
  CHECK(buffer.getOperand(0) == shape.getResult(0));
  CHECK(buffer.getOperand(1) == dtype);
  CHECK(buffer.getOperand(2) == none);
  CHECK(buffer.getOperand(3) == device.getResult(0));
  CHECK(buffer.getOperand(4) == none);
  CHECK(buffer.getOperand(5) == none);
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
  CHECK_THAT(print(*result.segments),
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
