add_library(kashi_iree_plugin STATIC "${CMAKE_CURRENT_LIST_DIR}/registration.cpp")
# The bundled IREE compiler and MLIR are built without C++ RTTI.
target_compile_options(kashi_iree_plugin PRIVATE -fno-rtti)
target_link_libraries(kashi_iree_plugin PRIVATE
  iree::compiler::PluginAPI iree::compiler::PluginAPI::PluginEntryPoint
  iree::compiler::Dialect::Flow::IR iree::compiler::Dialect::HAL::IR
  iree::compiler::Dialect::VM::IR
  iree::compiler::plugins::input::Torch::torch-mlir::TorchDialectIR
  MLIRFuncDialect MLIRNVVMDialect MLIRPass)
iree_compiler_register_plugin(PLUGIN_ID kashi TARGET kashi_iree_plugin)
