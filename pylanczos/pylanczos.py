import numpy as np
from pylanczoscpp import PyLanczosBase

class PyLanczos(PyLanczosBase):
    def __init__(self, matrix, find_maximum = False):
        super(PyLanczos, self).__init__(matrix.shape[0], find_maximum)
        self.matrix = matrix
        self.find_maximum = find_maximum

        self.__is_sparse = False
        try:
            from scipy import sparse
            self.__is_sparse = sparse.issparse(matrix)
        except ImportError:
            pass

    def mv_mul(self, v_in, v_out):
        # This "if" statement is required due to
        # undefined behavior of numpy.dot with scipy.sparse.
        if self.__is_sparse:
            result = self.matrix*v_in
        else:
            result = self.matrix.dot(v_in)
        np.copyto(v_out, result)
        #np.dot(self.matrix, v_in, v_out)  # I need something like this...
