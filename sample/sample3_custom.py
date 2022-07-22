import numpy as np
from pylanczos import PyLanczos


def exec_sample():
    n = 4

    ## Express the following matrix as a 4th-order tensor:
    ## [[1  0  0  1]
    ##  [0  0  1  1]
    ##  [0  1  0  1]
    ##  [1  1  1  0]]
    ## (Its eigenvalues are (1-sqrt(13))/2, -1, 1, and (1+sqrt(13))/2).
    tensor = np.zeros((2, 2, 2, 2), dtype='float64')
    tensor[0, 0, 0, 0] = 1
    tensor[0, 0, 1, 1] = 1
    tensor[0, 1, 1, 0] = 1
    tensor[0, 1, 1, 1] = 1
    tensor[1, 0, 0, 1] = 1
    tensor[1, 0, 1, 1] = 1
    tensor[1, 1, 0, 0] = 1
    tensor[1, 1, 0, 1] = 1
    tensor[1, 1, 1, 0] = 1

    ## Matrix-vector (or tensor-matrix) multiplication function.
    ## Note the followings:
    ## (1) Don't use reshape() for the input/output vectors. 
    ##     reshape() may duplicate elements rather than simply create a view of the original vector.
    ##     v.shape=... can prevent such implicit copying (it raises an error when copying is inevitable).
    ## (2) Use the "out" keyword argument of np.einsum to store the result directly into v_out.
    ## (Optional) For some complex tensor, a pre-calculated contraction list may improve performance (see np.einsum_path).
    def mv_mul(v_in, v_out):
        v_in.shape = (2, 2)
        v_out.shape = (2, 2)

        np.einsum("ijkl, kl -> ij", tensor, v_in, out=v_out, optimize="optimal")

    ## Calculate an "eigenmatrix" for the 4th-order tensor.
    engine = PyLanczos.create_custom(mv_mul, n, 'float64')
    eigval, eigmat = engine.run()
    eigmat.shape = (2, 2)
    print("Eigenvalue: {}".format(eigval))
    print("Eigenmatrix: {}".format(eigmat))


if __name__ == "__main__":
    exec_sample()
