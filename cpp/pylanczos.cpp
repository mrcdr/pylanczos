#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>

#include <complex>
#include <pylanczos.hpp>
#include <string>
#include <tuple>

namespace pylanczos {

template <typename T>
void declare_template_lanczos(pybind11::module& m, const std::string& suffix) {
  using namespace lambda_lanczos;

  std::string pyclass_name = std::string("PyLanczosCpp") + suffix;

  pybind11::class_<PyLanczosCpp<T>>(m, pyclass_name.c_str())
      .def_readwrite("matrix_size", &PyLanczosCpp<T>::matrix_size)
      .def_readwrite("find_maximum", &PyLanczosCpp<T>::find_maximum)
      .def_readwrite("eigenvalue_offset", &PyLanczosCpp<T>::eigenvalue_offset)
      .def(pybind11::init<std::function<void(pybind11::array_t<T>, pybind11::array_t<T>)>, size_t, bool, size_t>())
      .def("run", &PyLanczosCpp<T>::run);
}

template <typename T>
void declare_template_exponentiator(pybind11::module& m, const std::string& suffix) {
  using namespace lambda_lanczos;

  std::string pyclass_name = std::string("PyExponentiatorCpp") + suffix;

  pybind11::class_<PyExponentiatorCpp<T>>(m, pyclass_name.c_str())
      .def_readwrite("matrix_size", &PyExponentiatorCpp<T>::matrix_size)
      .def(pybind11::init<std::function<void(pybind11::array_t<T>, pybind11::array_t<T>)>, size_t>())
      .def("run", &PyExponentiatorCpp<T>::run);
}

PYBIND11_MODULE(pylanczoscpp, m) {
  declare_template_lanczos<float>(m, "Float");
  declare_template_lanczos<double>(m, "Double");
  declare_template_lanczos<long double>(m, "LongDouble");
  declare_template_lanczos<std::complex<float>>(m, "ComplexFloat");
  declare_template_lanczos<std::complex<double>>(m, "ComplexDouble");
  declare_template_lanczos<std::complex<long double>>(m, "ComplexLongDouble");

  declare_template_exponentiator<float>(m, "Float");
  declare_template_exponentiator<double>(m, "Double");
  declare_template_exponentiator<long double>(m, "LongDouble");
  declare_template_exponentiator<std::complex<float>>(m, "ComplexFloat");
  declare_template_exponentiator<std::complex<double>>(m, "ComplexDouble");
  declare_template_exponentiator<std::complex<long double>>(m, "ComplexLongDouble");
}

} /* namespace pylanczos */
