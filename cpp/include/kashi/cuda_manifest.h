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

// For each dispatch target (Flow executable symbol + export symbol), record
// the original pair and assign a unique name to the function and export so
// the kernel is still identifiable after CUDA links the executables together.
// Save flow.mlir first: FX reassembly reads the original references from that
// file and uses them as manifest keys. Only the in-memory IR is renamed.
mlir::FailureOr<std::vector<KernelIdentity>>
identifyCudaKernels(mlir::ModuleOp module);

// Resolve the Flow keys using serialized CUDA export metadata, then write the
// manifest and its PTX images.
mlir::LogicalResult
writeCudaManifest(mlir::ModuleOp module, const std::filesystem::path &directory,
                  const std::vector<KernelIdentity> &identities);

} // namespace kashi
