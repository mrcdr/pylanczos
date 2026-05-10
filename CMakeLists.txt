cmake_minimum_required(VERSION 3.15)

project(
  ${SKBUILD_PROJECT_NAME}
  VERSION ${SKBUILD_PROJECT_VERSION}
  LANGUAGES CXX)

include_directories("./include")
include_directories("./external/lambda-lanczos/include/lambda_lanczos")

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)
pybind11_add_module(pylanczoscpp MODULE "cpp/pylanczos.cpp" WITH_SOABI)
install(TARGETS pylanczoscpp DESTINATION "${SKBUILD_PROJECT_NAME}")