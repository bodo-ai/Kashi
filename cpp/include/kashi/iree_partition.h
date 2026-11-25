#pragma once

#include <functional>
#include <optional>
#include <string>
#include <string_view>

#include "mlir/IR/BuiltinOps.h"

namespace kashi {

struct Partition {
  mlir::OwningOpRef<mlir::ModuleOp> segments;
  mlir::OwningOpRef<mlir::ModuleOp> driver;
  std::optional<std::string> sourceDevice;
};

// The optional schema query returns whether an ATen operation aliases or
// writes an argument; nullopt means the operation has no known schema.
// Without it, registered Torch traits are used conservatively.
mlir::FailureOr<Partition> partitionTorchModule(
    mlir::ModuleOp source,
    const std::function<std::optional<bool>(std::string_view)> &schema = {});

} // namespace kashi
