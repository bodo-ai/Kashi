// Outline value-semantic Torch runs and replay aliasing, mutation, allocation,
// shape and container boundaries in the driver in their original order.

#include "kashi/iree_partition.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"

#include <utility>

namespace kashi {
namespace {
using namespace mlir;
namespace Torch = torch::Torch;

struct Segment {
  SmallVector<Operation *> ops;
  SmallVector<Value> inputs;
  SmallVector<Value> outputs;
  func::FuncOp function;
};

using ConstantOps = llvm::SmallSetVector<Operation *, 8>;

struct PartitionPlan {
  // Keep empty runs: each driver boundary ends a run, and its original index
  // determines the segment symbol even if some runs are not outlined.
  SmallVector<Segment> segments;
  llvm::DenseMap<Operation *, size_t> membership;
  ConstantOps constants;
  llvm::DenseSet<Operation *> boundaries;
  std::optional<std::string> sourceDevice;
};

/// Return whether the Python driver can transport this type across a segment.
bool supportsBoundaryType(Type type) {
  return isa<Torch::BaseTensorType, Torch::ListType, Torch::IntType>(type);
}

bool isCustomOp(Operation &op) {
  return op.getName().getStringRef() == Torch::OperatorOp::getOperationName();
}

bool containsTensor(Type type) {
  if (isa<Torch::BaseTensorType>(type))
    return true;
  if (auto list = dyn_cast<Torch::ListType>(type))
    return containsTensor(list.getContainedType());
  return false;
}

bool containsInt(Type type) {
  if (isa<Torch::IntType>(type))
    return true;
  if (auto list = dyn_cast<Torch::ListType>(type))
    return containsInt(list.getContainedType());
  return false;
}

bool isDriverList(Operation &op, const PartitionPlan &plan) {
  StringRef name = op.getName().getStringRef();
  if (name == Torch::PrimListConstructOp::getOperationName()) {
    auto list = dyn_cast<Torch::ListType>(op.getResult(0).getType());
    if (!list)
      return false;
    Type element = list.getContainedType();
    return containsTensor(element) ||
           (containsInt(element) &&
            !llvm::all_of(op.getOperands(), [&](Value value) {
              return plan.constants.contains(value.getDefiningOp());
            }));
  }
  if (name == Torch::PrimListUnpackOp::getOperationName()) {
    auto list = dyn_cast<Torch::ListType>(op.getOperand(0).getType());
    return list && (containsTensor(list.getContainedType()) ||
                    !plan.constants.contains(op.getOperand(0).getDefiningOp()));
  }
  return false;
}

bool isExplicitBoundary(StringRef name) {
  return name == "torch.aten.size" || name.starts_with("torch.aten.size.") ||
         name == "torch.aten.sym_size" ||
         name.starts_with("torch.aten.sym_size.") ||
         name == "torch.aten._unsafe_view" ||
         name.starts_with("torch.aten._unsafe_view.") ||
         name == "torch.aten.unsafe_split" ||
         name.starts_with("torch.aten.unsafe_split.") ||
         name == "torch.aten.unsafe_split_with_sizes" ||
         name.starts_with("torch.aten.unsafe_split_with_sizes.") ||
         name == "torch.aten.clone" || name.starts_with("torch.aten.clone.") ||
         // Copies can change devices or consume strided inputs, neither of
         // which the compiled segment ABI can represent faithfully.
         name == "torch.aten._to_copy" ||
         name.starts_with("torch.aten._to_copy.") ||
         name == "torch.aten.empty" || name.starts_with("torch.aten.empty.") ||
         name.starts_with("torch.aten.empty_") ||
         name == "torch.aten.new_empty" ||
         name.starts_with("torch.aten.new_empty.") ||
         name.starts_with("torch.aten.new_empty_");
}

/// Determine whether the PyTorch driver must retain this operation.
FailureOr<bool> isDriverBoundary(
    Operation &op, const PartitionPlan &plan,
    const std::function<std::optional<bool>(std::string_view)> &schema,
    llvm::StringMap<std::optional<bool>> &schemaCache) {
  StringRef name = op.getName().getStringRef();
  bool boundary =
      isCustomOp(op) || isExplicitBoundary(name) || isDriverList(op, plan);
  if (name.starts_with("torch.aten.") && schema) {
    auto [it, inserted] = schemaCache.try_emplace(name);
    if (inserted)
      it->second = schema(std::string_view(name.data(), name.size()));
    if (!it->second)
      return op.emitError("unknown ATen schema: ") << name;
    boundary = boundary || *it->second;
  } else if (name.starts_with("torch.aten.")) {
    if (!op.getName().isRegistered())
      return op.emitError("unknown ATen operation: ") << name;
    boundary = boundary || !op.hasTrait<Torch::OpTrait::HasValueSemantics>();
  }
  if (!boundary && !name.starts_with("torch.aten.") &&
      !name.starts_with("torch.constant.") &&
      name != Torch::ValueTensorLiteralOp::getOperationName() &&
      name != Torch::PrimListConstructOp::getOperationName() &&
      name != Torch::PrimListUnpackOp::getOperationName()) {
    if (!op.getName().isRegistered())
      return op.emitError("unknown Torch operation: ") << name;
    boundary = !op.hasTrait<Torch::OpTrait::HasValueSemantics>();
  }
  return boundary;
}

/// Resolve values in order; a missing mapping is a partition construction bug.
SmallVector<Value> mapped(ValueRange values, const IRMapping &mapping) {
  return llvm::map_to_vector(
      values, [&](Value value) { return mapping.lookup(value); });
}

/// Classify source operations in block order without changing the source.
LogicalResult classifySourceOps(
    Block &body, PartitionPlan &plan,
    const std::function<std::optional<bool>(std::string_view)> &schema) {
  llvm::StringMap<std::optional<bool>> schemaCache;
  plan.segments.emplace_back();
  for (Operation &op : body.without_terminator()) {
    StringRef name = op.getName().getStringRef();
    if (op.getNumRegions()) {
      return op.emitError("nested regions are not supported");
    }
    if (!name.starts_with("torch.")) {
      return op.emitError("unsupported source operation: ") << name;
    }
    auto boundary = isDriverBoundary(op, plan, schema, schemaCache);
    if (failed(boundary))
      return failure();
    // FailureOr hides has_value(); the failed() check above proves success.
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    if (*boundary) {
      plan.boundaries.insert(&op);
      plan.segments.emplace_back();
      continue;
    }

    bool constant = name.starts_with("torch.constant.") ||
                    name == Torch::ValueTensorLiteralOp::getOperationName();
    bool constantList =
        name == Torch::PrimListConstructOp::getOperationName() &&
        llvm::all_of(op.getOperands(), [&](Value value) {
          return plan.constants.contains(value.getDefiningOp());
        });
    if (!constant && !constantList) {
      plan.membership.try_emplace(&op, plan.segments.size() - 1);
      plan.segments.back().ops.push_back(&op);
      continue;
    }

    if (constant && op.getNumOperands()) {
      return op.emitError("constant with operands: ") << name;
    }
    if (name == Torch::ConstantDeviceOp::getOperationName()) {
      StringRef device = op.getAttrOfType<StringAttr>("value").getValue();
      if (plan.sourceDevice && device != *plan.sourceDevice) {
        return op.emitError("multiple source devices are unsupported: ")
               << *plan.sourceDevice << " and " << device;
      }
      if (!plan.sourceDevice) {
        plan.sourceDevice = device.str();
      }
    }
    plan.constants.insert(&op);
  }
  return success();
}

/// Collect a constant and all the constants it needs.
void collectConstantClosure(Operation &op,
                            llvm::DenseSet<Operation *> &needed) {
  if (!needed.insert(&op).second) {
    return;
  }
  // Only constant lists have operands, all defined by earlier constants.
  for (Value element : op.getOperands()) {
    collectConstantClosure(*element.getDefiningOp(), needed);
  }
}

LogicalResult computeSegmentABI(const PartitionPlan &plan, size_t index,
                                StringRef name, Segment &segment,
                                llvm::DenseSet<Operation *> &neededConstants) {
  llvm::DenseSet<Value> seenInputs;
  auto isLocal = [&](Operation *op) {
    auto it = plan.membership.find(op);
    return it != plan.membership.end() && it->second == index;
  };

  // Scan operations in block order and operands in operand order. The first
  // use of each nonlocal value fixes its argument position: if the run uses
  // %b before %a, it takes (%b, %a) even if @main declares (%a, %b).
  // Constants are cloned instead of taking up argument positions.
  //
  // Values used outside the run are returned in definition order (result
  // number within each op). A run defining %a then %b returns (%a, %b) even
  // if the custom op that consumes them takes (%b, %a).
  for (Operation *op : segment.ops) {
    for (Value operand : op->getOperands()) {
      Operation *def = operand.getDefiningOp();
      if (def && plan.constants.contains(def)) {
        collectConstantClosure(*def, neededConstants);
        continue;
      }
      if (isLocal(def) || !seenInputs.insert(operand).second) {
        continue;
      }
      if (!supportsBoundaryType(operand.getType())) {
        return op->emitError("unsupported live-in for ")
               << name << ": " << operand.getType();
      }
      segment.inputs.push_back(operand);
    }
    for (Value result : op->getResults()) {
      if (llvm::all_of(result.getUsers(), isLocal)) {
        continue;
      }
      if (!supportsBoundaryType(result.getType())) {
        return op->emitError("unsupported live-out for ")
               << name << ": " << result.getType();
      }
      segment.outputs.push_back(result);
    }
  }
  return success();
}

void populateSegmentBody(Segment &segment, const ConstantOps &constants,
                         const llvm::DenseSet<Operation *> &neededConstants,
                         Location returnLoc) {
  Block &entry = *segment.function.addEntryBlock();
  OpBuilder builder = OpBuilder::atBlockEnd(&entry);
  IRMapping mapping;
  mapping.map(segment.inputs, entry.getArguments());

  // Clone only the needed constants, in source order so list elements are
  // mapped before the list itself. Device rewriting is segment-local.
  for (Operation *op : constants) {
    if (!neededConstants.contains(op)) {
      continue;
    }
    Operation &clone = *builder.clone(*op, mapping);
    if (op->getName().getStringRef() ==
            Torch::ConstantDeviceOp::getOperationName() &&
        op->getAttrOfType<StringAttr>("value").getValue().starts_with("cuda")) {
      clone.setAttr("value", builder.getStringAttr("cpu"));
    }
  }
  for (Operation *op : segment.ops) {
    builder.clone(*op, mapping);
  }
  func::ReturnOp::create(builder, returnLoc, mapped(segment.outputs, mapping));
}

/// Outline each nonempty segment as a definition in the segment module and a
/// matching private declaration in the driver. Compute its ordered SSA
/// interface once so the definition and its driver calls share the same ABI.
LogicalResult outlineSegments(PartitionPlan &plan, Partition &result,
                              func::FuncOp main, func::ReturnOp ret) {
  OpBuilder segmentBuilder = OpBuilder::atBlockEnd(result.segments->getBody());
  OpBuilder driverBuilder = OpBuilder::atBlockEnd(result.driver->getBody());
  for (auto [index, segment] : llvm::enumerate(plan.segments)) {
    if (segment.ops.empty()) {
      continue;
    }

    std::string name = "segment_" + std::to_string(index);
    llvm::DenseSet<Operation *> neededConstants;
    if (failed(
            computeSegmentABI(plan, index, name, segment, neededConstants))) {
      return failure();
    }

    auto signature =
        segmentBuilder.getFunctionType(TypeRange(ValueRange(segment.inputs)),
                                       TypeRange(ValueRange(segment.outputs)));
    segment.function =
        func::FuncOp::create(segmentBuilder, main.getLoc(), name, signature);
    populateSegmentBody(segment, plan.constants, neededConstants, ret.getLoc());
    func::FuncOp::create(driverBuilder, main.getLoc(), name, signature)
        .setPrivate();
  }
  return success();
}

void callSegment(OpBuilder &builder, Location loc, const Segment &segment,
                 IRMapping &mapping) {
  if (!segment.function) {
    return;
  }
  auto call = func::CallOp::create(builder, loc, segment.function,
                                   mapped(segment.inputs, mapping));
  // The same ABI order binds return operand i to call result i and restores
  // its source SSA identity for custom ops and later segment calls.
  mapping.map(segment.outputs, call.getResults());
}

void buildDriver(ModuleOp driver, const PartitionPlan &plan, func::FuncOp main,
                 func::ReturnOp ret) {
  OpBuilder builder = OpBuilder::atBlockEnd(driver.getBody());
  auto driverMain = func::FuncOp::create(builder, main.getLoc(), "main",
                                         main.getFunctionType());
  Block &entry = *driverMain.addEntryBlock();
  builder.setInsertionPointToEnd(&entry);
  IRMapping mapping;
  Block &sourceBody = main.getBody().front();
  mapping.map(sourceBody.getArguments(), entry.getArguments());

  size_t index = 0;
  for (Operation &op : sourceBody.without_terminator()) {
    if (plan.boundaries.contains(&op)) {
      callSegment(builder, op.getLoc(), plan.segments[index++], mapping);
    } else if (!plan.constants.contains(&op)) {
      continue;
    }
    builder.clone(op, mapping);
  }
  callSegment(builder, ret.getLoc(), plan.segments[index], mapping);
  builder.clone(*ret, mapping);
}
} // namespace

FailureOr<Partition> partitionTorchModule(
    ModuleOp source,
    const std::function<std::optional<bool>(std::string_view)> &schema) {
  auto &operations = source.getBody()->getOperations();
  if (!llvm::hasSingleElement(operations)) {
    return source.emitError(
        "expected a module containing only func.func @main");
  }
  auto main = dyn_cast<func::FuncOp>(operations.front());
  if (!main || main.getSymName() != "main" ||
      !llvm::hasSingleElement(main.getBody())) {
    return source.emitError("expected a single block in func.func @main");
  }
  Block &body = main.getBody().front();
  auto ret = dyn_cast_or_null<func::ReturnOp>(body.getTerminator());
  if (!ret) {
    return main.emitError("expected func.return at the end of @main");
  }

  PartitionPlan plan;
  if (failed(classifySourceOps(body, plan, schema))) {
    return failure();
  }

  Partition result{ModuleOp::create(source.getLoc()),
                   ModuleOp::create(source.getLoc()),
                   std::move(plan.sourceDevice)};
  if (failed(outlineSegments(plan, result, main, ret))) {
    return failure();
  }

  buildDriver(*result.driver, plan, main, ret);
  if (failed(verify(*result.driver))) {
    return failure();
  }
  return result;
}
} // namespace kashi
