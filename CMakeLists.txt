cmake_minimum_required(VERSION 3.29)
if(NOT DEFINED CMAKE_LINKER_TYPE AND DEFINED ENV{CMAKE_LINKER_TYPE})
  set(CMAKE_LINKER_TYPE "$ENV{CMAKE_LINKER_TYPE}")
endif()
project(kashi LANGUAGES C CXX)

include(cmake/KashiBuildOptions.cmake)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  # Avoid the newer TLSDESC ABI chosen by nanobind on some build sysroots.
  set(NB_HAS_MTLS_GNU2 OFF)
endif()
find_package(nanobind CONFIG REQUIRED)
include(cmake/KashiIreeSetup.cmake)

set(KASHI_LIB_SOURCES
  cpp/lib/cuda_manifest.cpp
  cpp/lib/iree_compiler.cpp
  cpp/lib/iree_partition.cpp)
if(KASHI_COVERAGE)
  # Keep every translation unit in the test binary, even if no test calls it.
  add_library(kashi_core OBJECT ${KASHI_LIB_SOURCES})
else()
  add_library(kashi_core STATIC ${KASHI_LIB_SOURCES})
endif()
set_target_properties(kashi_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(KASHI_LINT)
  set_property(TARGET kashi_core PROPERTY CXX_CLANG_TIDY "${KASHI_CLANG_TIDY}")
endif()
target_include_directories(kashi_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/cpp/include")
target_compile_features(kashi_core PUBLIC cxx_std_17)
if(KASHI_IREE_SDK_DIR)
  target_link_libraries(kashi_core PRIVATE KashiIree::core)
else()
  target_link_libraries(kashi_core PRIVATE
    iree::compiler::API::StaticImpl
    iree::compiler::Dialect::Flow::IR
    iree::compiler::Dialect::HAL::IR
    iree::hal::utils::executable_header
    iree::schemas::cuda_executable_def_c_fbs
    iree::schemas::executable_debug_info_c_fbs
    iree::compiler::plugins::input::Torch::torch-mlir::TorchDialectIR)
  set(KASHI_IREE_INCLUDE_DIRS
    "${KASHI_IREE_SOURCE_DIR}/compiler/bindings/c"
    "${KASHI_IREE_SOURCE_DIR}/compiler/src"
    "${CMAKE_CURRENT_BINARY_DIR}/iree/compiler/src"
    "${KASHI_IREE_SOURCE_DIR}/third_party/llvm-project/llvm/include"
    "${KASHI_IREE_SOURCE_DIR}/third_party/llvm-project/mlir/include"
    "${CMAKE_CURRENT_BINARY_DIR}/iree/llvm-project/include"
    "${CMAKE_CURRENT_BINARY_DIR}/iree/llvm-project/tools/mlir/include")
  target_include_directories(kashi_core SYSTEM PRIVATE ${KASHI_IREE_INCLUDE_DIRS})
endif()
if(KASHI_MUTATION)
  target_compile_options(kashi_core PRIVATE
    -O0 -g -UNDEBUG -grecord-command-line "-fpass-plugin=${KASHI_MULL_PLUGIN}")
  set_property(SOURCE ${KASHI_LIB_SOURCES} APPEND PROPERTY OBJECT_DEPENDS
    "${KASHI_MULL_CONFIG};${KASHI_MULL_PLUGIN}")
endif()
if(KASHI_COVERAGE)
  target_compile_options(kashi_core PRIVATE ${KASHI_COVERAGE_FLAGS})
  # Both the test executable and the extension link these object files.
  target_link_options(kashi_core INTERFACE -fprofile-instr-generate)
endif()

include(cmake/KashiPythonExtension.cmake)

if(KASHI_BUILD_TESTS)
  include(cmake/KashiNativeTests.cmake)
endif()
