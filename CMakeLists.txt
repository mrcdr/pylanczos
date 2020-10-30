cmake_minimum_required(VERSION 3.11)
project (pylanczos CXX)

include_directories("./src")
include_directories("./external/lambda-lanczos/include/lambda_lanczos")

## Pybind configuration
add_subdirectory("external/pybind11")
file(GLOB pybind_src "cpp/*.cpp")
pybind11_add_module(pylanczoscpp ${pybind_src})
