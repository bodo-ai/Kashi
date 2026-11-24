#include "kashi/iree_compiler.h"

#include <nanobind/nanobind.h>
// These headers provide nanobind type-caster specializations, not named APIs.
#include <nanobind/stl/filesystem.h> // IWYU pragma: keep
#include <nanobind/stl/optional.h>   // IWYU pragma: keep
#include <nanobind/stl/string.h>     // IWYU pragma: keep

/// Define the private Python interface to the embedded IREE compiler.
NB_MODULE(_native, m) {
  namespace nb = nanobind;
  const nb::exception<kashi::UnsupportedPartition> unsupportedPartition(
      m, "_UnsupportedPartition", PyExc_NotImplementedError);
  m.def(
      "compile_iree",
      [](const std::string &assembly, std::filesystem::path artifactDir,
         const std::string &target, std::optional<std::string> cudaTarget) {
        return kashi::compileIree(assembly, {std::move(artifactDir),
                                             kashi::parseIreeBackend(target),
                                             std::move(cudaTarget)});
      },
      "Compile Torch MLIR and return the original tensor device.",
      nb::arg("assembly"), nb::arg("artifact_dir"), nb::kw_only(),
      nb::arg("target"), nb::arg("cuda_target") = nb::none(),
      nb::call_guard<nb::gil_scoped_release>());
}
