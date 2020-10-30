import numpy as np
from pylanczos import PyLanczos
matrix = np.array([[2.0, 1.0, 1.0],
                   [1.0, 2.0, 1.0],
                   [1.0, 1.0, 2.0]])

engine = PyLanczos(matrix, True)  # True to calculate the maximum eigenvalue.
eigval, eigvec, itern = engine.run()
print("Eigenvalue: {}".format(eigval))
print("Eigenvector: {}".format(eigvec))
