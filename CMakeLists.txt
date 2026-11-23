cmake_minimum_required(VERSION 3.29)
if(NOT DEFINED CMAKE_LINKER_TYPE AND DEFINED ENV{CMAKE_LINKER_TYPE})
  set(CMAKE_LINKER_TYPE "$ENV{CMAKE_LINKER_TYPE}")
endif()
project(kashi LANGUAGES C CXX)

if(NOT DEFINED CMAKE_C_COMPILER_LAUNCHER OR NOT DEFINED CMAKE_CXX_COMPILER_LAUNCHER)
  find_program(KASHI_SCCACHE NAMES sccache)
  if(KASHI_SCCACHE)
    foreach(lang IN ITEMS C CXX)
      if(NOT DEFINED CMAKE_${lang}_COMPILER_LAUNCHER)
        set(CMAKE_${lang}_COMPILER_LAUNCHER "${KASHI_SCCACHE}")
      endif()
    endforeach()
  endif()
endif()

option(KASHI_BUILD_TESTS "Build the native Catch2 tests" OFF)
option(KASHI_MUTATION "Instrument project native code for Mull" OFF)
option(KASHI_LINT "Run clang-tidy on project C++ sources" OFF)
option(KASHI_COVERAGE "Instrument native tests for LLVM coverage" OFF)
set(KASHI_MULL_PLUGIN "" CACHE FILEPATH "Mull LLVM frontend plugin")
set(KASHI_MULL_CONFIG "${CMAKE_CURRENT_SOURCE_DIR}/mull.yml" CACHE FILEPATH "Mull mutation policy")

if(KASHI_MUTATION)
  if(NOT KASHI_BUILD_TESTS)
    message(FATAL_ERROR "KASHI_MUTATION requires KASHI_BUILD_TESTS=ON")
  endif()
  if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    message(FATAL_ERROR "KASHI_MUTATION requires Clang")
  endif()
  if(NOT IS_ABSOLUTE "${KASHI_MULL_PLUGIN}"
     OR NOT EXISTS "${KASHI_MULL_PLUGIN}"
     OR IS_DIRECTORY "${KASHI_MULL_PLUGIN}")
    message(FATAL_ERROR "KASHI_MULL_PLUGIN must name an existing plugin file")
  endif()
endif()

if(KASHI_LINT)
  find_program(KASHI_CLANG_TIDY NAMES clang-tidy REQUIRED)
endif()

if(KASHI_COVERAGE)
  if(NOT KASHI_BUILD_TESTS OR KASHI_MUTATION)
    message(FATAL_ERROR "KASHI_COVERAGE requires tests and cannot be combined with Mull")
  endif()
  if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    message(FATAL_ERROR "KASHI_COVERAGE requires Clang")
  endif()
  set(KASHI_COVERAGE_FLAGS -fprofile-instr-generate -fcoverage-mapping)
endif()

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  # Avoid the newer TLSDESC ABI chosen by nanobind on some build sysroots.
  set(NB_HAS_MTLS_GNU2 OFF)
endif()
find_package(nanobind CONFIG REQUIRED)
include(cmake/KashiIree.cmake)

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
target_link_libraries(kashi_core PRIVATE
  iree::compiler::API::StaticImpl
  iree::compiler::Dialect::Flow::IR
  iree::compiler::Dialect::HAL::IR
  iree::hal::utils::executable_header
  iree::compiler::Dialect::VM::Target::Bytecode
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

nanobind_add_module(_native cpp/bindings/module.cpp)
target_link_libraries(_native PRIVATE kashi_core)
if(APPLE)
  target_link_options(_native PRIVATE "LINKER:-exported_symbol,_PyInit__native")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set(exports "${CMAKE_CURRENT_BINARY_DIR}/_native.exports")
  file(GENERATE OUTPUT "${exports}" CONTENT "{ global: PyInit__native; local: *; };\n")
  target_link_options(_native PRIVATE "LINKER:--version-script=${exports}")
  set_property(TARGET _native APPEND PROPERTY LINK_DEPENDS "${exports}")
else()
  message(FATAL_ERROR "The private compiler supports Linux and macOS")
endif()
if(KASHI_LINT)
  set_property(TARGET _native PROPERTY CXX_CLANG_TIDY "${KASHI_CLANG_TIDY}")
endif()
install(TARGETS _native LIBRARY DESTINATION kashi RUNTIME DESTINATION kashi)
install(FILES "${KASHI_IREE_SOURCE_DIR}/third_party/llvm-project/mlir/LICENSE.TXT"
  DESTINATION kashi/licenses RENAME MLIR.txt)
install(FILES "${KASHI_IREE_SOURCE_DIR}/third_party/flatcc/LICENSE"
  DESTINATION kashi/licenses RENAME FlatCC.txt)
if(IREE_TARGET_BACKEND_CUDA)
  find_file(KASHI_CUDA_NOTICE NAMES EULA.txt LICENSE
    HINTS "$ENV{IREE_CUDA_DEPS_DIR}" "${CUDAToolkit_BIN_DIR}/.."
    NO_DEFAULT_PATH REQUIRED)
  install(FILES "${KASHI_CUDA_NOTICE}" DESTINATION kashi/licenses RENAME CUDA.txt)
endif()
install(FILES "${KASHI_IREE_SOURCE_DIR}/LICENSE" DESTINATION kashi/licenses RENAME IREE.txt)
install(FILES "${KASHI_IREE_SOURCE_DIR}/third_party/llvm-project/llvm/LICENSE.TXT"
  DESTINATION kashi/licenses RENAME LLVM.txt)
install(FILES "${KASHI_IREE_SOURCE_DIR}/third_party/torch-mlir/LICENSE"
  DESTINATION kashi/licenses RENAME Torch-MLIR.txt)

if(KASHI_BUILD_TESTS)
  enable_testing()
  find_package(Catch2 3 CONFIG REQUIRED)
  list(APPEND CMAKE_MODULE_PATH "${Catch2_DIR}")
  include(Catch)

  file(GLOB_RECURSE KASHI_TEST_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/cpp/tests/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/cpp/tests/*.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/cpp/tests/*.cxx")
  if(KASHI_TEST_SOURCES)
    add_executable(kashi_native_tests ${KASHI_TEST_SOURCES})
    target_include_directories(kashi_native_tests SYSTEM PRIVATE ${KASHI_IREE_INCLUDE_DIRS})
    target_link_libraries(kashi_native_tests PRIVATE
      kashi_core Catch2::Catch2WithMain
      iree::compiler::Dialect::HAL::IR
      iree::compiler::Utils
      iree::schemas::cuda_executable_def_c_fbs
      iree::schemas::executable_debug_info_c_fbs)
    if(IREE_TARGET_BACKEND_CUDA)
      target_compile_definitions(kashi_native_tests PRIVATE KASHI_TEST_CUDA_CODEGEN)
    endif()
    if(KASHI_LINT)
      set_property(TARGET kashi_native_tests PROPERTY CXX_CLANG_TIDY "${KASHI_CLANG_TIDY}")
    endif()
    if(KASHI_MUTATION)
      # The Mull policy excludes test sources while permitting inline project headers.
      target_compile_options(kashi_native_tests PRIVATE
        -O0 -g -grecord-command-line "-fpass-plugin=${KASHI_MULL_PLUGIN}")
      set_property(SOURCE ${KASHI_TEST_SOURCES} APPEND PROPERTY OBJECT_DEPENDS
        "${KASHI_MULL_CONFIG};${KASHI_MULL_PLUGIN}")
    endif()
    if(KASHI_COVERAGE)
      # Instrument test translation units for coverage of inline project headers.
      target_compile_options(kashi_native_tests PRIVATE ${KASHI_COVERAGE_FLAGS})
      target_link_options(kashi_native_tests PRIVATE -fprofile-instr-generate)
    endif()
    catch_discover_tests(kashi_native_tests)
  endif()
endif()
