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

// Constructs both modules in the source context without modifying the source.
mlir::FailureOr<Partition> partitionTorchModule(mlir::ModuleOp source);

} // namespace kashi
