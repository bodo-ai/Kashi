#pragma once

#include <filesystem>
#include <vector>

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/StringRef.h"

namespace kashi {

// Names returned by identifyCudaKernels are interned in the invocation's MLIR
// context and remain valid while that context lives.
struct KernelIdentity {
  // Original Flow executable symbol, the outer manifest key.
  llvm::StringRef executable;
  // Original Flow export symbol, the key within that executable.
  llvm::StringRef exportName;
  // Link-unique name used to match serialized CUDA exports.
  llvm::StringRef stableName;
};

// Preserve the original Flow keys, then rename functions and exports so those
// keys can still be resolved after CUDA linking merges executable namespaces.
// Call only after saving the original Flow snapshot.
mlir::FailureOr<std::vector<KernelIdentity>>
identifyCudaKernels(mlir::ModuleOp module);

// Resolve the Flow keys using serialized CUDA export metadata, then write the
// manifest and its PTX images.
mlir::LogicalResult
writeCudaManifest(mlir::ModuleOp module, const std::filesystem::path &directory,
                  const std::vector<KernelIdentity> &identities);

} // namespace kashi
