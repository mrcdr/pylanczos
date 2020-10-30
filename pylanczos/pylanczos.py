import numpy as np
from pylanczoscpp import *
from .pylanczos_exception import PyLanczosException


class PyLanczos():
    _dtype_to_suffix = {np.dtype(np.float32): PyLanczosCppFloat,
                        np.dtype(np.float64): PyLanczosCppDouble,
                        np.dtype(np.float128): PyLanczosCppLongDouble,
                        np.dtype(np.complex64): PyLanczosCppComplexFloat,
                        np.dtype(np.complex128): PyLanczosCppComplexDouble,
                        np.dtype(np.complex256): PyLanczosCppComplexLongDouble}

    def __init__(self, matrix, find_maximum=False):
        if matrix.dtype not in PyLanczos._dtype_to_suffix:
            raise PyLanczosException('Unsupported dtype: {}'.format(matrix.dtype))

        self._matrix = matrix
        self._find_maximum = find_maximum

    def run(self):
        def mv_mul(v_in, v_out):
            result = self._matrix.dot(v_in)
            np.copyto(v_out, result)
            # np.dot(self.matrix, v_in, v_out)  # I need something like this...

        klass = PyLanczos._dtype_to_suffix[self._matrix.dtype]
        engine = klass(mv_mul, self._matrix.shape[0], self._find_maximum)

        return engine.run()
