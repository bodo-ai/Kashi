#pragma once

#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>

namespace kashi {

struct UnsupportedPartition : std::runtime_error {
  using std::runtime_error::runtime_error;
};

struct IreeCompileOptions {
  std::filesystem::path artifactDir;
  std::string target;
  std::optional<std::string> cudaTarget;
};

std::optional<std::string> compileIree(const std::string &assembly,
                                       const IreeCompileOptions &options);

enum class IreeTestPipeline { Checkpoints, FailAtFlow, Uninterrupted };

void compileIreeForTesting(const std::string &assembly,
                           const IreeCompileOptions &options,
                           IreeTestPipeline pipeline);

} // namespace kashi
