#pragma once

#include <filesystem>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

namespace kashi {

struct UnsupportedPartition : std::runtime_error {
  using std::runtime_error::runtime_error;
};

enum class IreeBackend { CUDA, VMVX };

std::optional<IreeBackend> parseIreeBackend(std::string_view target);

struct IreeCompileOptions {
  std::filesystem::path artifactDir;
  // An unrecognized caller target is reported in the compilation log.
  std::optional<IreeBackend> target;
  std::optional<std::string> cudaTarget;
  std::function<std::optional<bool>(std::string_view)> schema;
};

std::optional<std::string> compileIree(const std::string &assembly,
                                       const IreeCompileOptions &options);

enum class IreeTestPipeline { Checkpoints, FailAtFlow };

void compileIreeForTesting(const std::string &assembly,
                           const IreeCompileOptions &options,
                           IreeTestPipeline pipeline);

} // namespace kashi
