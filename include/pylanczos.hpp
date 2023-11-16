#ifndef PYLANCZOS_H_
#define PYLANCZOS_H_

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <exponentiator.hpp>
#include <functional>
#include <lambda_lanczos.hpp>
#include <tuple>

namespace pylanczos {

template <typename T>
class PyLanczosCpp : public lambda_lanczos::LambdaLanczos<T> {
 public:
  PyLanczosCpp(std::function<void(pybind11::array_t<T>, pybind11::array_t<T>)> mv_mul,
               size_t matrix_size,
               bool find_maximum,
               size_t num_eigs)
      : lambda_lanczos::LambdaLanczos<T>(
            [=](const std::vector<T>& in, std::vector<T>& out) {
              mv_mul(pybind11::array_t<T>(this->matrix_size, in.data(), pybind11::capsule(&in)),
                     pybind11::array_t<T>(this->matrix_size, out.data(), pybind11::capsule(&out)));
            },
            matrix_size,
            find_maximum,
            num_eigs) {}

  // This tuple will be interpreted as multiple-value return (python tuple) in Python.
  std::tuple<pybind11::array_t<T>, pybind11::array_t<T>, std::vector<size_t>> run() {
    using RT = lambda_lanczos::util::real_t<T>;
    std::vector<RT> eigenvalues;
    std::vector<std::vector<T>> eigenvectors;

    this->lambda_lanczos::LambdaLanczos<T>::run(eigenvalues, eigenvectors);

    auto eigenvalues_py = pybind11::array_t<RT>(std::vector<size_t>{this->num_eigs});
    auto eigenvectors_py =
        pybind11::array_t<T, pybind11::array::f_style>(std::vector<size_t>{this->matrix_size, this->num_eigs});

    /* Copy result to numpy arrays */
    for (size_t j = 0; j < this->num_eigs; ++j) {
      *(eigenvalues_py.mutable_data(j)) = eigenvalues[j];
      for (size_t i = 0; i < this->matrix_size; ++i) {
        *(eigenvectors_py.mutable_data(i, j)) = eigenvectors[j][i];  // Transpose
      }
    }

    return std::make_tuple(eigenvalues_py, eigenvectors_py, this->getIterationCounts());
  }
};

template <typename T>
class PyExponentiatorCpp : public lambda_lanczos::Exponentiator<T> {
 public:
  PyExponentiatorCpp(std::function<void(pybind11::array_t<T>, pybind11::array_t<T>)> mv_mul, size_t matrix_size)
      : lambda_lanczos::Exponentiator<T>(
            [=](const std::vector<T>& in, std::vector<T>& out) {
              mv_mul(pybind11::array_t<T>(this->matrix_size, in.data(), pybind11::capsule(&in)),
                     pybind11::array_t<T>(this->matrix_size, out.data(), pybind11::capsule(&out)));
            },
            matrix_size) {}

  // This tuple will be interpreted as multiple-value return (python tuple) in Python.
  std::tuple<pybind11::array_t<T>, size_t> run(const T a, pybind11::array_t<T> input_py) {
    std::vector<T> input(this->matrix_size), output(this->matrix_size);
    for (size_t i = 0; i < input.size(); ++i) {
      input[i] = *input_py.data(i);
    }
    const auto iter_count = this->lambda_lanczos::Exponentiator<T>::run(a, input, output);

    /* Copy result to numpy arrays */
    auto output_py = pybind11::array_t<T>(std::vector<size_t>{this->matrix_size});
    for (size_t i = 0; i < this->matrix_size; ++i) {
      *(output_py.mutable_data(i)) = output[i];
    }

    return std::make_tuple(output_py, iter_count);
  }
};

} /* namespace pylanczos */

#endif /* PYLANCZOS_H_ */
