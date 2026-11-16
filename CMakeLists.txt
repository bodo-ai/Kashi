cmake_minimum_required(VERSION 3.20)
project(ml_compiler LANGUAGES CXX)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(_native cpp/bindings/module.cpp)
install(TARGETS _native LIBRARY DESTINATION ml_compiler RUNTIME DESTINATION ml_compiler)
