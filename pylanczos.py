import numpy as np
from pylanczoscpp import PyLanczosBase

class PyLanczos(PyLanczosBase):
    def __init__(self, matrix, find_maximum = False):
        super(PyLanczos, self).__init__(len(matrix), find_maximum)
        self.matrix = matrix
        self.find_maximum = find_maximum

    def mv_mul(self, v_in, v_out):
        np.dot(self.matrix, v_in, v_out)
