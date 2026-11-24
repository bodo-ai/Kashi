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
  if(KASHI_IREE_SDK_DIR)
    target_link_libraries(kashi_native_tests PRIVATE
      kashi_core Catch2::Catch2WithMain kashi_iree_plugin
      KashiIree::core KashiIree::tests KashiIree::test_link)
  else()
    target_include_directories(kashi_native_tests SYSTEM PRIVATE ${KASHI_IREE_INCLUDE_DIRS})
    target_link_libraries(kashi_native_tests PRIVATE
      kashi_core Catch2::Catch2WithMain
      iree::compiler::Dialect::HAL::IR
      iree::compiler::Utils
      iree::compiler::plugins::input::Torch::torch-mlir::TorchDialectIR
      MLIRFuncDialect
      MLIRParser
      iree::schemas::cuda_executable_def_c_fbs
      iree::schemas::executable_debug_info_c_fbs)
  endif()
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
