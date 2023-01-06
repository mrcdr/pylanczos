import numpy as np
from pylanczos import PyLanczos


def exec_sample():
    matrix = np.array([[2.0, 1.0, 1.0],
                       [1.0, 2.0, 1.0],
                       [1.0, 1.0, 2.0]])

    engine = PyLanczos(matrix, True, 2)  # Find 2 maximum eigenpairs
    eigenvalues, eigenvectors = engine.run()
    print("Eigenvalue: {}".format(eigenvalues))
    print("Eigenvector:")
    print(eigenvectors)


if __name__ == "__main__":
    exec_sample()
