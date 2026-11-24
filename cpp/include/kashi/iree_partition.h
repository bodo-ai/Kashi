#pragma once

#include <optional>
#include <string>

#include "mlir/IR/BuiltinOps.h"

namespace kashi {

struct Partition {
  mlir::OwningOpRef<mlir::ModuleOp> segments;
  mlir::OwningOpRef<mlir::ModuleOp> driver;
  std::optional<std::string> sourceDevice;
};

// Outlines Torch runs between custom ops and builds a driver that calls them
// in source order. Both modules use the source context; the source is
// unchanged.
mlir::FailureOr<Partition> partitionTorchModule(mlir::ModuleOp source);

} // namespace kashi
