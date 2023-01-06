![CI](https://github.com/mrcdr/pylanczos/workflows/CI/badge.svg)
[![codecov](https://codecov.io/gh/mrcdr/pylanczos/branch/master/graph/badge.svg?token=CLVRQ8PN1J)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()


# PyLanczos
## Overview
**PyLanczos** is a Lanczos diagonalization library.
Its core part is written in C++ as [LambdaLanczos](https://github.com/mrcdr/lambda-lanczos).

## Usage
All samples are available [here](https://github.com/mrcdr/pylanczos/tree/master/sample).

### NumPy and SciPy matrix
``` python
matrix = np.array([[2.0, 1.0, 1.0],
                   [1.0, 2.0, 1.0],
                   [1.0, 1.0, 2.0]])

engine = PyLanczos(matrix, True, 2)  # Find 2 maximum eigenpairs
eigenvalues, eigenvectors = engine.run()
print("Eigenvalue: {}".format(eigenvalues))
print("Eigenvector:")
print(eigenvectors)
```
Note: Use of SciPy sparse matrix is recommended to take full advantage of Lanczos algorithm.

### Customized operation
You can also attach your customized function:
```python
tensor = np.zeros((2, 2, 2, 2), dtype='float64')
tensor[0, 0, 0, 0] = 1
# and so on...

## Matrix-vector (or tensor-matrix) multiplication function
def mv_mul(v_in, v_out):
    v_in.shape = (2, 2)
    v_out.shape = (2, 2)

    np.einsum("ijkl, kl -> ij", tensor, v_in, out=v_out, optimize="optimal")

## Calculate an "eigenmatrix" for the 4th-order tensor.
engine = PyLanczos.create_custom(mv_mul, n, 'float64', False, 1) # Find 1 minimum eigenpair
eigenvalues, eigenmatrices = engine.run()
eigenmatrices.shape = (2, 2)
print("Eigenvalue: {}".format(eigenvalues))
print("Eigenmatrix:")
print(eigenmatrices)
```
There is [a full sample](https://github.com/mrcdr/pylanczos/tree/master/sample/sample3_custom.py) with detailed description.

## Installation
```sh
pip install pylanczos
```

## Requirements

C++11 compatible environment

## License

[MIT](https://github.com/mrcdr/lambda-lanczos/blob/master/LICENSE)

## Author

[mrcdr](https://github.com/mrcdr)

## PyPI repository

[pylanczos](https://pypi.org/project/pylanczos/)
