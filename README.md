![CI](https://github.com/mrcdr/pylanczos/workflows/CI/badge.svg)
[![codecov](https://codecov.io/gh/mrcdr/pylanczos/branch/master/graph/badge.svg?token=CLVRQ8PN1J)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()


# PyLanczos
## Overview
**PyLanczos** is a Lanczos diagonalization library.
Its core part is written in C++ as [LambdaLanczos](https://github.com/mrcdr/lambda-lanczos).

## Usage
Samples are available [here](https://github.com/mrcdr/pylanczos/tree/master/sample).
``` python
import numpy as np
from pylanczos import PyLanczos


matrix = np.array([[2.0, 1.0, 1.0],
                   [1.0, 2.0, 1.0],
                   [1.0, 1.0, 2.0]])

engine = PyLanczos(matrix, True)  # True to calculate the maximum eigenvalue.
eigval, eigvec = engine.run()

print("Eigenvalue: {}".format(eigval))
print("Eigenvector: {}".format(eigvec))
```


## Installation

`pip install pylanczos`

## Requirements

C++11 compatible environment

## License

[MIT](https://github.com/mrcdr/lambda-lanczos/blob/master/LICENSE)

## Author

[mrcdr](https://github.com/mrcdr)
