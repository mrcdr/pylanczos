#ifndef PYLANCZOS_H_
#define PYLANCZOS_H_

#include <tuple>
#include <functional>
#include <lambda_lanczos.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace pylanczos {

template <typename T>
class PyLanczosBase {
public:
  size_t matrix_size;
  bool find_maximum;
  lambda_lanczos::util::real_t<T> eigenvalue_offset = 0.0;

  PyLanczosBase(size_t matrix_size, bool find_maximum)
    : matrix_size(matrix_size), find_maximum(find_maximum) {}
  virtual ~PyLanczosBase() {};

  // This tuple will be interpreted as multiple-value return (python tuple) in Python.
  std::tuple<lambda_lanczos::util::real_t<T>,
	     pybind11::array_t<T>,
	     int>
  run() {
    auto mv_mul = [&](const std::vector<double>& in, std::vector<double>& out) {
      this->mv_mul(pybind11::array_t<T>(this->matrix_size, in.data(), pybind11::capsule(&in)),
		   pybind11::array_t<T>(this->matrix_size, out.data(), pybind11::capsule(&out)));
    };

    lambda_lanczos::LambdaLanczos<T> engine(mv_mul, this->matrix_size, this->find_maximum);
    engine.eigenvalue_offset = this->eigenvalue_offset;
    lambda_lanczos::util::real_t<T> eigenvalue;
    auto eigenvector = new std::vector<T>(this->matrix_size);
    auto cap = pybind11::capsule(eigenvector, [](void* v) { delete reinterpret_cast<std::vector<T>*>(v); });
    auto itern = engine.run(eigenvalue, *eigenvector);

    return std::make_tuple(eigenvalue, pybind11::array(eigenvector->size(), eigenvector->data(), cap), itern);
  }

  virtual void mv_mul(pybind11::array_t<T>, pybind11::array_t<T>) = 0;
};


template <typename T>
class PyLanczosBaseInheritanceHelper : public PyLanczosBase<T> {
public:
  /* Inherit the constructors */
  using PyLanczosBase<T>::PyLanczosBase;

  /* Trampoline (need one for each virtual function) */
  void mv_mul(pybind11::array_t<T> in, pybind11::array_t<T> out) override {
    PYBIND11_OVERLOAD_PURE(void, PyLanczosBase<T>, mv_mul, in, out);
  }
};


} /* namespace pylanczos */

#endif /* PYLANCZOS_H_ */
