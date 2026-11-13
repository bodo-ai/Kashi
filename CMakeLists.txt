cmake_minimum_required(VERSION 3.20)
project(ml_compiler LANGUAGES CXX)
option(ML_COMPILER_BUILD_TESTS "Build the native Catch2 tests" OFF)
option(ML_COMPILER_MUTATION "Instrument project native code for Mull" OFF)
set(ML_COMPILER_MULL_PLUGIN "" CACHE FILEPATH "Mull LLVM frontend plugin")

if(ML_COMPILER_MUTATION)
  if(NOT ML_COMPILER_BUILD_TESTS)
    message(FATAL_ERROR "ML_COMPILER_MUTATION requires ML_COMPILER_BUILD_TESTS=ON")
  endif()
  if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    message(FATAL_ERROR "ML_COMPILER_MUTATION requires Clang")
  endif()
  if(NOT IS_ABSOLUTE "${ML_COMPILER_MULL_PLUGIN}"
     OR NOT EXISTS "${ML_COMPILER_MULL_PLUGIN}"
     OR IS_DIRECTORY "${ML_COMPILER_MULL_PLUGIN}")
    message(FATAL_ERROR "ML_COMPILER_MULL_PLUGIN must name an existing plugin file")
  endif()
endif()

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind CONFIG REQUIRED)

file(GLOB_RECURSE ML_COMPILER_LIB_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/cpp/lib/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/cpp/lib/*.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/cpp/lib/*.cxx")

if(ML_COMPILER_LIB_SOURCES)
  add_library(ml_compiler_core STATIC ${ML_COMPILER_LIB_SOURCES})
  set_target_properties(ml_compiler_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
  target_include_directories(ml_compiler_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/cpp/include")
  if(ML_COMPILER_MUTATION)
    target_compile_options(ml_compiler_core PRIVATE
      -O0 -g -grecord-command-line "-fpass-plugin=${ML_COMPILER_MULL_PLUGIN}")
  endif()
else()
  add_library(ml_compiler_core INTERFACE)
  target_include_directories(ml_compiler_core INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/cpp/include")
endif()

nanobind_add_module(_native cpp/bindings/module.cpp)
target_link_libraries(_native PRIVATE ml_compiler_core)
install(TARGETS _native LIBRARY DESTINATION ml_compiler RUNTIME DESTINATION ml_compiler)

if(ML_COMPILER_BUILD_TESTS)
  enable_testing()
  find_package(Catch2 3 CONFIG REQUIRED)
  list(APPEND CMAKE_MODULE_PATH "${Catch2_DIR}")
  include(Catch)

  file(GLOB_RECURSE ML_COMPILER_TEST_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/cpp/tests/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/cpp/tests/*.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/cpp/tests/*.cxx")
  if(ML_COMPILER_TEST_SOURCES)
    add_executable(ml_compiler_native_tests ${ML_COMPILER_TEST_SOURCES})
    target_link_libraries(ml_compiler_native_tests PRIVATE ml_compiler_core Catch2::Catch2WithMain)
    if(ML_COMPILER_MUTATION)
      # The Mull policy excludes test sources while permitting inline project headers.
      target_compile_options(ml_compiler_native_tests PRIVATE
        -O0 -g -grecord-command-line "-fpass-plugin=${ML_COMPILER_MULL_PLUGIN}")
    endif()
    catch_discover_tests(ml_compiler_native_tests)
  endif()
endif()
