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
        """Constructs Lanczos calculation engine.

        Parameters
        ----------
        matrix : Any numpy or scipy matrix type
            The symmetric (Hermitian) matrix to be diagonalized.
            This matrix will never be changed.
        find_maximum : bool
            `True` to calculate the maximum eigenvalue,
            `False` to calculate the minimum one.

        Raises
        ------
        PyLanczosException
            Unsupported `dtype` is specified.

        Note
        ----
        Matrices which has the following attributes/methods are
        actually acceptable:

        - `dtype` attribute, which should be `numpy.dtype`
        - `dot` method, which accepts `numpy.ndarray` and
          returns the dot product (matrix-vector multiplication) result
          as `numpy.ndarray`

        `dtype` should be numpy `float32`, `float64`, `float128`, `complex64`,
        `complex128`, or `complex256`.
        """

        if matrix.dtype not in PyLanczos._dtype_to_suffix:
            raise PyLanczosException('Unsupported dtype: {}'.format(matrix.dtype))

        self._dtype = matrix.dtype
        self._n = matrix.shape[0]
        self._find_maximum = find_maximum

        def mv_mul(v_in, v_out):
            result = matrix.dot(v_in)
            np.copyto(v_out, result)
            # np.dot(self.matrix, v_in, v_out)  # I need something like this...

        self._mv_mul = mv_mul

    @staticmethod
    def create_custom(mv_mul, n, dtype, find_maximum=False):
        """Constructs Lanczos calculation engine
        with a custom matrix-vector multiplication function.

        Parameters
        ----------
        mv_mul : lambda(numpy.array, numpy.array)
            The matrix-vector multiplication function, which should multiply
            a symmetric (Hermitian) matrix to an input array (the first
            parameter) and store the result into an output array (the second
            parameter). Each arrays are passed as `numpy.array`.
        n : int
            The dimension of the matrix, i.e. n for n by n matrix.
        dtype : numpy.dtype or str
            The type of the matrix (acceptable type is described in a note of
            `__init__` section).
        find_maximum : bool
            `True` to calculate the maximum eigenvalue,
            `False` to calculate the minimum one.

        Raises
        ------
        PyLanczosException
            Unsupported `dtype` is specified.

        Note
        ----
        Since the matrix-vector multiplication function usually spends
        the most part of the Lanczos calculation time, the function should be
        optimized enough.
        """

        dummy_matrix = np.array([], dtype=dtype)
        pylanczos_obj = PyLanczos(dummy_matrix, find_maximum)
        pylanczos_obj._n = n
        pylanczos_obj._mv_mul = mv_mul

        return pylanczos_obj

    def run(self):
        """Executes the Lanczos algorithm.

        Returns
        -------
        float
            Calculated eigenvalue
        numpy.ndarray
            Calculated eigenvector
        int
            Lanczos iteration count
        """

        klass = PyLanczos._dtype_to_suffix[self._dtype]
        engine = klass(self._mv_mul, self._n, self._find_maximum)

        return engine.run()
