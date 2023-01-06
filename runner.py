import numpy as np
from pylanczos import PyLanczos

if __name__ == "__main__":
    matrix = np.array([[2.0, 1.0, 1.0],
                       [1.0, 2.0, 1.0],
                       [1.0, 1.0, 2.0]])

    engine = PyLanczos(matrix, True, 2) # Find 2 maximum eigenpairs
    eigenvalues, eigenvectors = engine.run()
    print("Eigenvalues: {}".format(eigenvalues))
    print("Eigenvectors:")
    print(eigenvectors)
