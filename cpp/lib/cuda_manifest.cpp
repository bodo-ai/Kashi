#include "kashi/cuda_manifest.h"

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/hal/utils/executable_header.h"
#include "iree/schemas/cuda_executable_def_reader.h"
#include "iree/schemas/cuda_executable_def_verifier.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

namespace kashi {
namespace {
using namespace mlir;
using namespace mlir::iree_compiler;

struct KernelLocation {
  // PTX image filename relative to the artifact directory.
  std::string file;
  // Actual PTX entry symbol, borrowed from the HAL binary.
  StringRef entry;
};

/// Emit an artifact and report open, write, or close failures on its IR owner.
LogicalResult writeFile(OpState op, const std::filesystem::path &path,
                        llvm::function_ref<void(llvm::raw_ostream &)> write) {
  std::error_code error;
  llvm::raw_fd_ostream out(path.string(), error);
  if (error) {
    return op.emitError("cannot open ")
           << path.string() << ": " << error.message();
  }
  write(out);
  out.close();
  if (out.has_error()) {
    out.clear_error();
    return op.emitError("cannot write ") << path.string();
  }
  return success();
}

bool isPtxWordChar(char c) {
  return llvm::isAlnum(c) || c == '_' || c == '$' || c == '%' || c == '.';
}

/// Read a token, skipping comments as whitespace and keeping literals opaque.
StringRef nextPtxToken(StringRef &image) {
  while (!(image = image.ltrim()).empty()) {
    if (image.consume_front("//")) {
      image = image.split('\n').second;
      continue;
    }
    if (image.consume_front("/*")) {
      image = image.split("*/").second;
      continue;
    }
    char first = image.front();
    size_t length = 1;
    if (first == '"' || first == '\'') {
      while (length < image.size()) {
        char c = image[length++];
        if (c == first)
          break;
        if (c == '\\' && length < image.size())
          ++length;
      }
    } else if (isPtxWordChar(first)) {
      // Keep dotted words together so a suffix cannot look like a directive.
      while (length < image.size() && isPtxWordChar(image[length]))
        ++length;
    }
    auto token = image.take_front(length);
    image = image.drop_front(length);
    return token;
  }
  return {};
}

/// Collect the identifiers after top-level .entry directives. The result
/// borrows names from the original, unmodified PTX image.
llvm::DenseSet<StringRef> readPtxEntries(StringRef image) {
  llvm::DenseSet<StringRef> entries;
  size_t depth = 0;
  bool expectName = false;
  for (auto token = nextPtxToken(image); !token.empty();
       token = nextPtxToken(image)) {
    if (expectName) {
      char first = token.front();
      if ((llvm::isAlpha(first) ||
           (token.size() > 1 && StringRef("_$%").contains(first))) &&
          token.drop_front().find_first_of(".%") == StringRef::npos)
        entries.insert(token);
    }
    if (token == "{")
      ++depth;
    else if (token == "}" && depth != 0)
      --depth;
    expectName = depth == 0 && token == ".entry";
  }
  return entries;
}

LogicalResult extractCudaBinary(IREE::HAL::ExecutableBinaryOp binary,
                                const std::filesystem::path &directory,
                                size_t &nextImage,
                                llvm::StringMap<KernelLocation> &locations) {
  auto data = dyn_cast<DenseIntElementsAttr>(binary.getData());
  if (!data || !data.getElementType().isInteger(8) || data.isSplat())
    return binary.emitError("expected a byte array for the CUDA executable");
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
    return binary.emitError("invalid serialized CUDA executable FlatBuffer");
  auto root = iree_hal_cuda_ExecutableDef_as_root(flatbuffer.data);
  auto modules = iree_hal_cuda_ExecutableDef_modules(root);
  auto exports = iree_hal_cuda_ExecutableDef_exports(root);
  struct Image {
    std::string file;
    llvm::DenseSet<StringRef> entries;
  };
  SmallVector<Image> images;
  for (size_t i = 0, e = iree_hal_cuda_ModuleDef_vec_len(modules); i < e; ++i) {
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
  for (size_t i = 0, e = iree_hal_cuda_ExportDef_vec_len(exports); i < e; ++i) {
    auto entry = iree_hal_cuda_ExportDef_vec_at(exports, i);
    auto ordinal = iree_hal_cuda_ExportDef_module_ordinal(entry);
    auto exportName = iree_hal_cuda_ExportDef_export_name(entry);
    auto kernelName = iree_hal_cuda_ExportDef_kernel_name(entry);
    if (!exportName || !kernelName || ordinal >= images.size())
      return binary.emitError(
          "serialized CUDA export has no matching PTX entry");
    const auto &image = images[ordinal];
    StringRef entryName(kernelName, flatbuffers_string_len(kernelName));
    if (!image.entries.contains(entryName))
      return binary.emitError(
          "serialized CUDA export has no matching PTX entry");
    StringRef stableName(exportName, flatbuffers_string_len(exportName));
    if (!locations
             .try_emplace(stableName, KernelLocation{image.file, entryName})
             .second)
      return binary.emitError("ambiguous serialized CUDA export: ")
             << stableName;
  }
  return success();
}
} // namespace

/// CUDA linking merges executable-local symbols, and the serialized export
/// name does not include the original executable. Use a globally unique name
/// for the join while keeping the original executable/export pair for callers.
FailureOr<std::vector<KernelIdentity>> identifyCudaKernels(ModuleOp module) {
  llvm::DenseSet<StringAttr> names;
  module.walk([&](SymbolOpInterface symbol) {
    if (auto name = symbol.getNameAttr())
      names.insert(name);
  });
  std::vector<KernelIdentity> identities;
  // CUDA serialization matches functions and exports by name, so rename both.
  // Reserve names against all original symbols, including unexported ones.
  size_t nextName = 0;
  for (auto executable : module.getOps<IREE::Flow::ExecutableOp>()) {
    auto inner = executable.getInnerModule();
    if (!inner)
      return executable.emitError("expected a Flow executable with a body");
    SymbolTable symbols(inner);
    llvm::SmallDenseMap<StringAttr, StringAttr, 4> assigned;
    for (auto exportOp : executable.getOps<IREE::Flow::ExecutableExportOp>()) {
      auto originalName = exportOp.getFunctionRefAttr().getRootReference();
      auto [it, inserted] = assigned.try_emplace(originalName);
      if (inserted) {
        auto function = symbols.lookup<SymbolOpInterface>(originalName);
        if (!function)
          return exportOp.emitError(
              "cannot resolve the exported Flow function");
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
      // Dispatches still reference the old qualified export symbol.
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
      if (binary.getFormat() == "cuda-nvptx-fb" &&
          failed(extractCudaBinary(binary, directory, nextImage, locations)))
        return failure();
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
        llvm::json::Object{{"file", StringRef(it->second.file)},
                           {"entry", it->second.entry}};
  }
  return writeFile(module, directory / "kernel_manifest.json", [&](auto &out) {
    out << llvm::json::Value(std::move(manifest)) << '\n';
  });
}
} // namespace kashi
