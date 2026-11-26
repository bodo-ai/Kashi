#include "kashi/iree_compiler.h"

#include <nanobind/nanobind.h>
// These headers provide nanobind type-caster specializations, not named APIs.
#include <nanobind/stl/filesystem.h> // IWYU pragma: keep
#include <nanobind/stl/optional.h>   // IWYU pragma: keep
#include <nanobind/stl/string.h>     // IWYU pragma: keep

namespace {
namespace nb = nanobind;

// Query only schema alias/write metadata; segmentation itself remains native.
std::optional<bool> schemaBoundary(std::string_view name) {
  nb::gil_scoped_acquire acquire;
  constexpr std::string_view prefix = "torch.aten.";
  name.remove_prefix(prefix.size());
  auto dot = name.find('.');
  std::string base(name.substr(0, dot));
  std::string overload = dot == std::string_view::npos
                             ? "default"
                             : std::string(name.substr(dot + 1));
  nb::object aten = nb::module_::import_("torch").attr("ops").attr("aten");
  nb::object packet = nb::getattr(aten, base.c_str(), nb::none());
  if (packet.is_none() || !nb::hasattr(packet, "overloads"))
    return std::nullopt;
  bool found = false;
  for (nb::handle item : packet.attr("overloads")()) {
    if (nb::cast<std::string>(item) == overload) {
      found = true;
      break;
    }
  }
  if (!found)
    return std::nullopt;
  nb::object schema = nb::getattr(packet, overload.c_str()).attr("_schema");
  for (nb::handle result : nb::getattr(schema, "returns")) {
    if (!nb::getattr(result, "alias_info").is_none())
      return true;
  }
  // Python exposes the top-level alias info but not nested contained-type
  // annotations. Treat any annotated argument as a boundary too, so list
  // element aliasing or writes cannot be mistaken for pure value semantics.
  for (nb::handle argument : nb::getattr(schema, "arguments")) {
    if (!nb::getattr(argument, "alias_info").is_none())
      return true;
  }
  return false;
}
} // namespace

/// Define the private Python interface to the embedded IREE compiler.
NB_MODULE(_native, m) {
  namespace nb = nanobind;
  const nb::exception<kashi::UnsupportedPartition> unsupportedPartition(
      m, "_UnsupportedPartition", PyExc_NotImplementedError);
  m.def(
      "compile_iree",
      [](const std::string &assembly, std::filesystem::path artifactDir,
         const std::string &target, std::optional<std::string> cudaTarget) {
        kashi::IreeCompileOptions options{std::move(artifactDir),
                                          kashi::parseIreeBackend(target),
                                          std::move(cudaTarget)};
        options.schema = schemaBoundary;
        nb::gil_scoped_release release;
        return kashi::compileIree(assembly, options);
      },
      "Compile Torch MLIR and return the original tensor device.",
      nb::arg("assembly"), nb::arg("artifact_dir"), nb::kw_only(),
      nb::arg("target"), nb::arg("cuda_target") = nb::none());
}
