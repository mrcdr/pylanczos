import numpy as np
from pylanczos import PyLanczos

arr = np.matrix([[2, 1, 1], [1, 2, 1], [1, 1, 2]], dtype='float64')
engine = PyLanczos(arr, True)
engine.eigenvalue_offset = 10.0
eigval, eigvec, itern = engine.run()
