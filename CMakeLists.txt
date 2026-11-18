cmake_minimum_required(VERSION 3.20)
project(kashi LANGUAGES CXX)
option(KASHI_BUILD_TESTS "Build the native Catch2 tests" OFF)
option(KASHI_MUTATION "Instrument project native code for Mull" OFF)
option(KASHI_LINT "Run clang-tidy on project C++ sources" OFF)
set(KASHI_MULL_PLUGIN "" CACHE FILEPATH "Mull LLVM frontend plugin")

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

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind CONFIG REQUIRED)

file(GLOB_RECURSE KASHI_LIB_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/cpp/lib/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/cpp/lib/*.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/cpp/lib/*.cxx")

if(KASHI_LIB_SOURCES)
  add_library(kashi_core STATIC ${KASHI_LIB_SOURCES})
  set_target_properties(kashi_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
  if(KASHI_LINT)
    set_property(TARGET kashi_core PROPERTY CXX_CLANG_TIDY "${KASHI_CLANG_TIDY}")
  endif()
  target_include_directories(kashi_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/cpp/include")
  if(KASHI_MUTATION)
    target_compile_options(kashi_core PRIVATE
      -O0 -g -grecord-command-line "-fpass-plugin=${KASHI_MULL_PLUGIN}")
  endif()
else()
  add_library(kashi_core INTERFACE)
  target_include_directories(kashi_core INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/cpp/include")
endif()

nanobind_add_module(_native cpp/bindings/module.cpp)
target_link_libraries(_native PRIVATE kashi_core)
if(KASHI_LINT)
  set_property(TARGET _native PROPERTY CXX_CLANG_TIDY "${KASHI_CLANG_TIDY}")
endif()
install(TARGETS _native LIBRARY DESTINATION kashi RUNTIME DESTINATION kashi)

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
    target_link_libraries(kashi_native_tests PRIVATE kashi_core Catch2::Catch2WithMain)
    if(KASHI_LINT)
      set_property(TARGET kashi_native_tests PROPERTY CXX_CLANG_TIDY "${KASHI_CLANG_TIDY}")
    endif()
    if(KASHI_MUTATION)
      # The Mull policy excludes test sources while permitting inline project headers.
      target_compile_options(kashi_native_tests PRIVATE
        -O0 -g -grecord-command-line "-fpass-plugin=${KASHI_MULL_PLUGIN}")
    endif()
    catch_discover_tests(kashi_native_tests)
  endif()
endif()
