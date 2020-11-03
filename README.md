![CI](https://github.com/mrcdr/pylanczos/workflows/CI/badge.svg)
[![codecov](https://codecov.io/gh/mrcdr/pylanczos/branch/master/graph/badge.svg?token=CLVRQ8PN1J)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()


# PyLanczos
## Overview
**PyLanczos** is a wrapper library of [LambdaLanczos](https://github.com/mrcdr/lambda-lanczos),
which is written in C++.

Currently PyLanczos is a beta version.
The Lanczos algorithm itself is well tested, but the interface would be changed in future.

## Usage
Samples are available [here](https://github.com/mrcdr/pylanczos/tree/master/sample).
``` python
import numpy as np
from pylanczos import PyLanczos


matrix = np.array([[2.0, 1.0, 1.0],
                   [1.0, 2.0, 1.0],
                   [1.0, 1.0, 2.0]])

engine = PyLanczos(matrix, True)  # True to calculate the maximum eigenvalue.
eigval, eigvec, itern = engine.run()

print("Eigenvalue: {}".format(eigval))
print("Eigenvector: {}".format(eigvec))
```


## Installation
1. Clone or download the latest version from [Github](https://github.com/mrcdr/pylanczos/).
1. In the cloned directory, run `pip install .`


## License

[MIT](https://github.com/mrcdr/lambda-lanczos/blob/master/LICENSE)

## Author

[mrcdr](https://github.com/mrcdr)
