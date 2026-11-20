#include "kashi/iree_partition.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"
#include "torch-mlir/Dialect/Torch/IR/TorchTypes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

namespace kashi {
namespace {
using namespace mlir;

struct Segment {
  SmallVector<Operation *> ops;
  SmallVector<Value> inputs;
  SmallVector<Value> outputs;
  func::FuncOp function;
};

/// Return whether the Python driver can transport this type across a segment.
bool supported(Type type) {
  return isa<torch::Torch::BaseTensorType, torch::Torch::ListType,
             torch::Torch::IntType>(type);
}

/// Materialize the types in ABI order for building a function signature.
SmallVector<Type> types(ValueRange values) {
  return llvm::to_vector(values.getTypes());
}

/// Resolve values in order; a missing mapping is a partition construction bug.
SmallVector<Value> mapped(ValueRange values, const IRMapping &mapping) {
  return llvm::map_to_vector(
      values, [&](Value value) { return mapping.lookup(value); });
}

/// Collect a clonable constant and its transitive clonable dependencies.
void collectConstants(Operation *op,
                      const llvm::DenseSet<Operation *> &clonable,
                      llvm::DenseSet<Operation *> &needed) {
  if (!op || !clonable.contains(op) || !needed.insert(op).second)
    return;
  for (auto operand : op->getOperands())
    collectConstants(operand.getDefiningOp(), clonable, needed);
}
} // namespace

/// Outline supported runs while retaining custom operations in the driver.
FailureOr<Partition> partitionTorchModule(ModuleOp source) {
  auto &operations = source.getBody()->getOperations();
  if (!llvm::hasSingleElement(operations))
    return source.emitError(
        "expected a module containing only func.func @main");
  auto main = dyn_cast<func::FuncOp>(operations.front());
  if (!main || main.getSymName() != "main" ||
      !llvm::hasSingleElement(main.getBody()))
    return source.emitError("expected a single block in func.func @main");
  Block &body = main.getBody().front();
  auto ret = dyn_cast_or_null<func::ReturnOp>(body.getTerminator());
  if (!ret)
    return main.emitError("expected func.return at the end of @main");

  Partition result{
      ModuleOp::create(source.getLoc()), ModuleOp::create(source.getLoc()), {}};
  SmallVector<Segment> segments(1);
  llvm::DenseMap<Operation *, size_t> membership;
  llvm::DenseSet<Operation *> clonable;
  SmallVector<Operation *> constantOrder;
  for (auto &op : body.without_terminator()) {
    auto name = op.getName().getStringRef();
    if (op.getNumRegions())
      return op.emitError("nested regions are not supported");
    if (!name.starts_with("torch."))
      return op.emitError("unsupported source operation: ") << name;
    if (name == "torch.operator") {
      segments.emplace_back();
      continue;
    }
    bool constant =
        name.starts_with("torch.constant.") || name == "torch.vtensor.literal";
    bool constantList = name == "torch.prim.ListConstruct" &&
                        llvm::all_of(op.getOperands(), [&](Value value) {
                          return clonable.contains(value.getDefiningOp());
                        });
    if (constant || constantList) {
      if (constant && op.getNumOperands())
        return op.emitError("constant with operands: ") << name;
      if (name == "torch.constant.device") {
        auto device = op.getAttrOfType<StringAttr>("value").getValue();
        if (result.sourceDevice && device != *result.sourceDevice)
          return op.emitError("multiple source devices are unsupported: ")
                 << *result.sourceDevice << " and " << device;
        result.sourceDevice = device.str();
      }
      clonable.insert(&op);
      constantOrder.push_back(&op);
    } else {
      membership.try_emplace(&op, segments.size() - 1);
      segments.back().ops.push_back(&op);
    }
  }

  OpBuilder builder(source.getContext());
  for (size_t index = 0; index < segments.size(); ++index) {
    auto &segment = segments[index];
    if (segment.ops.empty())
      continue;
    std::string name = "segment_" + std::to_string(index);
    llvm::DenseSet<Value> inputs;
    llvm::DenseSet<Operation *> needed;
    auto local = [&](Operation *op) {
      auto it = membership.find(op);
      return it != membership.end() && it->second == index;
    };
    // Keep ABI order independent of the hash sets: first use for inputs,
    // definition order for outputs.
    for (auto *op : segment.ops) {
      for (auto operand : op->getOperands()) {
        auto *def = operand.getDefiningOp();
        if (clonable.contains(def)) {
          collectConstants(def, clonable, needed);
        } else if (!local(def) && inputs.insert(operand).second) {
          if (!supported(operand.getType()))
            return op->emitError("unsupported live-in for ")
                   << name << ": " << operand.getType();
          segment.inputs.push_back(operand);
        }
      }
      for (auto value : op->getResults()) {
        if (llvm::all_of(value.getUsers(), local))
          continue;
        if (!supported(value.getType()))
          return op->emitError("unsupported live-out for ")
                 << name << ": " << value.getType();
        segment.outputs.push_back(value);
      }
    }

    auto signature =
        builder.getFunctionType(types(segment.inputs), types(segment.outputs));
    builder.setInsertionPointToEnd(result.segments->getBody());
    segment.function =
        func::FuncOp::create(builder, main.getLoc(), name, signature);
    auto *entry = segment.function.addEntryBlock();
    builder.setInsertionPointToEnd(entry);
    IRMapping mapping;
    mapping.map(segment.inputs, entry->getArguments());
    // Clone only the constants needed here, in source order so list elements
    // are mapped before the list itself. Device rewriting is segment-local.
    for (auto *op : constantOrder) {
      if (!needed.contains(op))
        continue;
      auto *clone = builder.clone(*op, mapping);
      if (op->getName().getStringRef() == "torch.constant.device" &&
          op->getAttrOfType<StringAttr>("value").getValue().starts_with("cuda"))
        clone->setAttr("value", builder.getStringAttr("cpu"));
    }
    for (auto *op : segment.ops)
      builder.clone(*op, mapping);
    func::ReturnOp::create(builder, ret.getLoc(),
                           mapped(segment.outputs, mapping));

    builder.setInsertionPointToEnd(result.driver->getBody());
    func::FuncOp::create(builder, main.getLoc(), name, signature).setPrivate();
  }

  builder.setInsertionPointToEnd(result.driver->getBody());
  auto driverMain = func::FuncOp::create(builder, main.getLoc(), "main",
                                         main.getFunctionType());
  auto *entry = driverMain.addEntryBlock();
  builder.setInsertionPointToEnd(entry);
  IRMapping mapping;
  mapping.map(body.getArguments(), entry->getArguments());
  size_t index = 0;
  // Flush each compiled segment before cloning the next custom op. The
  // custom op and following segment then see the same mapped SSA values.
  for (auto &op : body) {
    bool custom = op.getName().getStringRef() == "torch.operator";
    if (custom || &op == ret.getOperation()) {
      const auto &segment = segments[index++];
      if (segment.function) {
        auto call = func::CallOp::create(builder, op.getLoc(), segment.function,
                                         mapped(segment.inputs, mapping));
        mapping.map(segment.outputs, call.getResults());
      }
      builder.clone(op, mapping);
    } else if (clonable.contains(&op)) {
      builder.clone(op, mapping);
    }
  }
  if (failed(verify(*result.driver)))
    return failure();
  return result;
}
} // namespace kashi
