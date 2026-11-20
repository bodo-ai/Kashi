#include "kashi/cuda_manifest.h"

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/hal/utils/executable_header.h"
#include "iree/schemas/cuda_executable_def_reader.h"
#include "iree/schemas/cuda_executable_def_verifier.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

namespace kashi {
namespace {
using namespace mlir;
using namespace mlir::iree_compiler;

struct KernelLocation {
  std::string file;
  std::string entry;
};

/// Emit an artifact and report open, write, or close failures on its IR owner.
LogicalResult writeFile(Operation *op, const std::filesystem::path &path,
                        llvm::function_ref<void(llvm::raw_ostream &)> write) {
  std::error_code error;
  llvm::raw_fd_ostream out(path.string(), error);
  if (error)
    return op->emitError("cannot open ")
           << path.string() << ": " << error.message();
  write(out);
  out.close();
  if (out.has_error()) {
    out.clear_error();
    return op->emitError("cannot write ") << path.string();
  }
  return success();
}

/// Collect PTX entry declarations, excluding names found only in comments.
llvm::StringSet<> readPtxEntries(StringRef image) {
  llvm::StringSet<> entries;
  while (!image.empty()) {
    auto [line, rest] = image.split('\n');
    image = rest;
    auto code = line.split("//").first;
    auto pos = code.find(".entry ");
    if (pos == StringRef::npos)
      pos = code.find(".entry\t");
    if (pos == StringRef::npos)
      continue;
    auto name = code.drop_front(pos + 7).ltrim().split('(').first.rtrim();
    if (!name.empty())
      entries.insert(name);
  }
  return entries;
}
} // namespace

/// Preserve original Flow keys while assigning names that survive CUDA linking.
FailureOr<std::vector<KernelIdentity>> identifyCudaKernels(ModuleOp module) {
  llvm::DenseSet<StringAttr> names;
  module.walk([&](SymbolOpInterface symbol) {
    if (auto name = symbol.getNameAttr())
      names.insert(name);
  });
  llvm::DenseMap<Operation *, StringAttr> assigned;
  std::vector<KernelIdentity> identities;
  // Stream resolves the function using the export symbol, so rename both
  // and retarget qualified dispatch references after saving the Flow snapshot.
  // Reserve names against all original symbols.
  size_t nextName = 0;
  for (auto executable : module.getOps<IREE::Flow::ExecutableOp>()) {
    auto inner = executable.getInnerModule();
    if (!inner)
      return executable.emitError("expected a Flow executable with a body");
    SymbolTable symbols(inner);
    SmallVector<std::pair<IREE::Flow::ExecutableExportOp, Operation *>> exports;
    for (auto exportOp : executable.getOps<IREE::Flow::ExecutableExportOp>()) {
      auto *function = symbols.lookup(exportOp.getFunctionRef());
      if (!function)
        return exportOp.emitError("cannot resolve the exported Flow function");
      exports.emplace_back(exportOp, function);
    }
    for (auto [exportOp, function] : exports) {
      auto [it, inserted] = assigned.try_emplace(function);
      if (inserted) {
        do {
          llvm::SmallString<32> buffer;
          llvm::raw_svector_ostream stream(buffer);
          stream << "kashi_kernel_" << nextName++;
          it->second = StringAttr::get(module.getContext(), buffer);
        } while (!names.insert(it->second).second);
        if (failed(symbols.rename(function, it->second)))
          return exportOp.emitError("cannot rename the exported Flow function");
      }
      exportOp.setFunctionRefAttr(FlatSymbolRefAttr::get(it->second));
      if (failed(
              SymbolTable::replaceAllSymbolUses(exportOp, it->second, module)))
        return exportOp.emitError("cannot retarget Flow dispatch references");
      identities.push_back({executable.getSymName(), exportOp.getSymName(),
                            it->second.getValue()});
      exportOp.setSymNameAttr(it->second);
    }
  }
  return identities;
}

/// Map original Flow exports to verified linked CUDA images and entries.
LogicalResult writeCudaManifest(ModuleOp module,
                                const std::filesystem::path &directory,
                                const std::vector<KernelIdentity> &identities) {
  llvm::StringMap<KernelLocation> locations;
  size_t nextImage = 0;
  for (auto executable : module.getOps<IREE::HAL::ExecutableOp>()) {
    for (auto binary : executable.getOps<IREE::HAL::ExecutableBinaryOp>()) {
      if (binary.getFormat() != "cuda-nvptx-fb")
        continue;
      auto data = dyn_cast<DenseIntElementsAttr>(binary.getData());
      if (!data || !data.getElementType().isInteger(8) || data.isSplat())
        return binary.emitError(
            "expected a byte array for the CUDA executable");
      auto raw = data.getRawData();
      iree_const_byte_span_t flatbuffer{};
      auto status = iree_hal_read_executable_flatbuffer_header(
          iree_make_const_byte_span(raw.data(), raw.size()), false,
          iree_hal_cuda_ExecutableDef_file_identifier, &flatbuffer);
      if (!iree_status_is_ok(status)) {
        iree_status_ignore(status);
        return binary.emitError("invalid serialized CUDA executable header");
      }
      if (iree_hal_cuda_ExecutableDef_verify_as_root(flatbuffer.data,
                                                     flatbuffer.data_length))
        return binary.emitError(
            "invalid serialized CUDA executable FlatBuffer");
      auto root = iree_hal_cuda_ExecutableDef_as_root(flatbuffer.data);
      auto modules = iree_hal_cuda_ExecutableDef_modules(root);
      auto exports = iree_hal_cuda_ExecutableDef_exports(root);
      struct Image {
        std::string file;
        llvm::StringSet<> entries;
      };
      SmallVector<Image> images;
      for (size_t i = 0, e = iree_hal_cuda_ModuleDef_vec_len(modules); i < e;
           ++i) {
        auto ptx = iree_hal_cuda_ModuleDef_ptx_image(
            iree_hal_cuda_ModuleDef_vec_at(modules, i));
        if (!ptx)
          return binary.emitError("missing serialized CUDA PTX image");
        StringRef text(ptx, flatbuffers_string_len(ptx));
        auto file = "kernel_" + std::to_string(nextImage++) + ".ptx";
        if (failed(writeFile(binary, directory / file,
                             [&](auto &out) { out << text; })))
          return failure();
        images.push_back({std::move(file), readPtxEntries(text)});
      }
      for (size_t i = 0, e = iree_hal_cuda_ExportDef_vec_len(exports); i < e;
           ++i) {
        auto entry = iree_hal_cuda_ExportDef_vec_at(exports, i);
        auto ordinal = iree_hal_cuda_ExportDef_module_ordinal(entry);
        auto exportName = iree_hal_cuda_ExportDef_export_name(entry);
        auto kernelName = iree_hal_cuda_ExportDef_kernel_name(entry);
        if (!exportName || !kernelName || ordinal >= images.size() ||
            !images[ordinal].entries.contains(kernelName))
          return binary.emitError(
              "serialized CUDA export has no matching PTX entry");
        if (!locations
                 .try_emplace(exportName,
                              KernelLocation{images[ordinal].file, kernelName})
                 .second)
          return binary.emitError("ambiguous serialized CUDA export: ")
                 << exportName;
      }
    }
  }
  llvm::json::Object manifest;
  for (const auto &identity : identities) {
    auto it = locations.find(identity.stableName);
    if (it == locations.end())
      return module.emitError("missing serialized CUDA export for ")
             << identity.executable << "::" << identity.exportName;
    if (!manifest.getObject(identity.executable))
      manifest[identity.executable] = llvm::json::Object{};
    (*manifest.getObject(identity.executable))[identity.exportName] =
        llvm::json::Object{{"file", it->second.file},
                           {"entry", it->second.entry}};
  }
  return writeFile(module, directory / "kernel_manifest.json", [&](auto &out) {
    out << llvm::json::Value(std::move(manifest)) << '\n';
  });
}
} // namespace kashi
