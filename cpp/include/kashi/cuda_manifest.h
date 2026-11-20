#pragma once

#include <filesystem>
#include <vector>

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/StringRef.h"

namespace kashi {

// Names are interned in the invocation context, which outlives the manifest.
struct KernelIdentity {
  llvm::StringRef executable;
  llvm::StringRef exportName;
  llvm::StringRef stableName;
};

// Assign collision-free internal symbols after saving the public Flow snapshot.
mlir::FailureOr<std::vector<KernelIdentity>>
identifyCudaKernels(mlir::ModuleOp module);

// Extract the PTX and its entry names from the linked, serialized HAL binaries.
mlir::LogicalResult
writeCudaManifest(mlir::ModuleOp module, const std::filesystem::path &directory,
                  const std::vector<KernelIdentity> &identities);

} // namespace kashi
