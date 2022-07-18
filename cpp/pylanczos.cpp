#include <string>
#include <complex>
#include <tuple>
#include <pybind11/pybind11.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pylanczos.hpp>


namespace pylanczos {

template<typename T>
void declare_template(pybind11::module &m, const std::string& suffix) {
  using namespace lambda_lanczos;

  std::string pyclass_name = std::string("PyLanczosCpp") + suffix;

  pybind11::class_<PyLanczosCpp<T>>(m, pyclass_name.c_str())
    .def_readwrite("matrix_size", &PyLanczosCpp<T>::matrix_size)
    .def_readwrite("find_maximum", &PyLanczosCpp<T>::find_maximum)
    .def_readwrite("eigenvalue_offset", &PyLanczosCpp<T>::eigenvalue_offset)
    .def(pybind11::init<std::function<void (pybind11::array_t<T>, pybind11::array_t<T>)>, size_t, bool>())
    .def("run", &PyLanczosCpp<T>::run);
}

PYBIND11_MODULE(pylanczoscpp, m) {
  declare_template<float>(m, "Float");
  declare_template<double>(m, "Double");
  declare_template<long double>(m, "LongDouble");
  declare_template<std::complex<float>>(m, "ComplexFloat");
  declare_template<std::complex<double>>(m, "ComplexDouble");
  declare_template<std::complex<long double>>(m, "ComplexLongDouble");
}

} /* namespace pylanczos */
