import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from pylanczos import PyLanczos


def exec_sample():
    n = 10

    matrix = lil_matrix((n, n), dtype='float64')

    for i in range(n-1):
        matrix[(i, i+1)] = 1
        matrix[(i+1, i)] = 1

    # Equivalent to the following n by n matrix
    #      0  1  0  0  ..  0  0
    #      1  0  1  0  ..  0  0
    #      0  1  0  1  ..  0  0
    #      0  0  1  0  ..  0  0
    #      :  :  :  :      :  :
    #      0  0  0  0  ..  0  1
    #      0  0  0  0  ..  1  0
    # Its eigenvalues are 2*cos(k*pi/(n+1)), where k = 1, 2, ... , n.
    # So k = 1 is the largest one and k = n is the smallest one.

    engine = PyLanczos(csr_matrix(matrix), True, 1) # Find 1 maximum eigenpair
    eigenvalues, eigenvectors = engine.run()
    print("Eigenvalue: {}".format(eigenvalues))
    print("Eigenvector:")
    print(eigenvectors)


if __name__ == "__main__":
    exec_sample()
