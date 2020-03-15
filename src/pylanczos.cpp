#include <pybind11/pybind11.h>
#include "pylanczos.hpp"


namespace pylanczos {

PYBIND11_MODULE(pylanczoscpp, m) {
    pybind11::class_<PyLanczosBase<double>, PyLanczosBaseInheritanceHelper<double>>(m, "PyLanczosBase")
      .def_readwrite("matrix_size", &PyLanczosBase<double>::matrix_size)
      .def_readwrite("find_maximum", &PyLanczosBase<double>::find_maximum)
      .def_readwrite("eigenvalue_offset", &PyLanczosBase<double>::eigenvalue_offset)
      .def(pybind11::init<size_t, bool>())
      .def("run", &PyLanczosBase<double>::run)
      .def("mv_mul", &PyLanczosBase<double>::mv_mul);
}

} /* namespace pylanczos */
