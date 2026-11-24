if(TARGET KashiIree::plugin)
  # The packaged IREE plugin manager calls Kashi's builtin registration symbol.
  # Link the consumer's object directly to resolve it regardless of archive order.
  add_library(kashi_iree_plugin OBJECT "${CMAKE_CURRENT_LIST_DIR}/registration.cpp")
  set_target_properties(kashi_iree_plugin PROPERTIES POSITION_INDEPENDENT_CODE ON)
  target_compile_features(kashi_iree_plugin PRIVATE cxx_std_17)
else()
  add_library(kashi_iree_plugin STATIC "${CMAKE_CURRENT_LIST_DIR}/registration.cpp")
endif()
if(KASHI_LINT)
  set_property(TARGET kashi_iree_plugin PROPERTY CXX_CLANG_TIDY "${KASHI_CLANG_TIDY}")
endif()
# The bundled IREE compiler and MLIR are built without C++ RTTI.
target_compile_options(kashi_iree_plugin PRIVATE -fno-rtti)
if(TARGET KashiIree::plugin)
  target_link_libraries(kashi_iree_plugin PRIVATE KashiIree::plugin)
  return()
endif()
target_link_libraries(kashi_iree_plugin PRIVATE
  iree::compiler::PluginAPI iree::compiler::PluginAPI::PluginEntryPoint
  iree::compiler::Dialect::Flow::IR iree::compiler::Dialect::HAL::IR
  iree::compiler::Dialect::VM::IR
  iree::compiler::plugins::input::Torch::torch-mlir::TorchDialectIR
  MLIRFuncDialect MLIRNVVMDialect MLIRPass)
iree_compiler_register_plugin(PLUGIN_ID kashi TARGET kashi_iree_plugin)
