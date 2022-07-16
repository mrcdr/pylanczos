import numpy as np
from pylanczos import PyLanczos


def exec_sample():
    matrix = np.array([[2.0, 1.0, 1.0],
                       [1.0, 2.0, 1.0],
                       [1.0, 1.0, 2.0]])

    engine = PyLanczos(matrix, True)  # True to calculate the maximum eigenvalue.
    eigval, eigvec = engine.run()
    print("Eigenvalue: {}".format(eigval))
    print("Eigenvector: {}".format(eigvec))


if __name__ == "__main__":
    exec_sample()
