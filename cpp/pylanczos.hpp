#ifndef PYLANCZOS_H_
#define PYLANCZOS_H_

#include <tuple>
#include <functional>
#include <lambda_lanczos.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace pylanczos {

template <typename T>
class PyLanczosCpp : public lambda_lanczos::LambdaLanczos<T> {
public:
  PyLanczosCpp(std::function<void(pybind11::array_t<T>, pybind11::array_t<T>)> mv_mul,
               size_t matrix_size,
               bool find_maximum)
    : lambda_lanczos::LambdaLanczos<T>([=](const std::vector<T>& in, std::vector<T>& out) {
      mv_mul(pybind11::array_t<T>(this->matrix_size, in.data(), pybind11::capsule(&in)),
	     pybind11::array_t<T>(this->matrix_size, out.data(), pybind11::capsule(&out)));
    },
      matrix_size,
      find_maximum) {}

  // This tuple will be interpreted as multiple-value return (python tuple) in Python.
  std::tuple<lambda_lanczos::util::real_t<T>,
             pybind11::array_t<T>,
             int>
  run() {
    lambda_lanczos::util::real_t<T> eigenvalue;
    auto eigenvector = new std::vector<T>(this->matrix_size);
    auto cap = pybind11::capsule(eigenvector, [](void* v) { delete reinterpret_cast<std::vector<T>*>(v); });
    auto itern = this->lambda_lanczos::LambdaLanczos<T>::run(eigenvalue, *eigenvector);

    return std::make_tuple(eigenvalue, pybind11::array(eigenvector->size(), eigenvector->data(), cap), itern);
  }
};

} /* namespace pylanczos */

#endif /* PYLANCZOS_H_ */
