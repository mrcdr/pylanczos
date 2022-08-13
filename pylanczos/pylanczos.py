import numpy as np
from pylanczoscpp import *
from .pylanczos_exception import PyLanczosException


def create_suffix_dict():
    dict = {}
    pairs = [
        ("single", PyLanczosCppFloat),
        ("double", PyLanczosCppDouble),
        ("longdouble", PyLanczosCppLongDouble),
        ("csingle", PyLanczosCppComplexFloat),
        ("cdouble", PyLanczosCppComplexDouble),
        ("clongdouble", PyLanczosCppComplexLongDouble)
    ]

    for (keystr, val) in pairs:
        dict[np.dtype(keystr)] = val

    return dict


class PyLanczos():
    _dtype_to_suffix = create_suffix_dict()

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
            If unsupported `dtype` is specified.

        Note
        ----
        A matrix that has the following attributes/methods are
        actually acceptable:

        - `dtype` attribute, which should be `numpy.dtype`
        - `dot` method, which accepts `numpy.ndarray` and
          returns the dot product (matrix-vector multiplication) result
          as `numpy.ndarray`

        Following `dtype` (and their equivalents) are acceptable: `single`, `double`, `longdouble`, `csingle`,
        `cdouble`, and `clongdouble`.
        """

        if matrix.dtype not in PyLanczos._dtype_to_suffix:
            raise PyLanczosException('Unsupported dtype: {}'.format(matrix.dtype))

        self._dtype = matrix.dtype
        self._n = matrix.shape[0]
        self._find_maximum = find_maximum
        self._iteration_count = 0

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
        mv_mul : lambda(numpy.ndarray, numpy.ndarray)
            The matrix-vector multiplication function, which should multiply
            a symmetric (Hermitian) matrix to an input array (the first
            parameter) and store the result into an output array (the second
            parameter). Each arrays are passed as `numpy.ndarray`.
        n : int
            The dimension of the matrix, i.e. n for n by n matrix.
        dtype : numpy.dtype or str
            The type of the matrix. Following `dtype` (and their equivalents) are acceptable:
            `single`, `double`, `longdouble`, `csingle`, `cdouble`, or `clongdouble`.
        find_maximum : bool
            `True` to calculate the maximum eigenvalue,
            `False` to calculate the minimum one.

        Raises
        ------
        PyLanczosException
            If unnsupported `dtype` is specified.

        Note
        ----
        Since the matrix-vector multiplication usually spends
        the most part of the Lanczos calculation time, `mv_mul` should be well optimized.
        It is commonly good to define `mv_mul` with binary-level optimized functions,
        such as `numpy.dot`, `numpy.einsum`, etc.
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
        """

        klass = PyLanczos._dtype_to_suffix[self._dtype]
        engine = klass(self._mv_mul, self._n, self._find_maximum)

        eigen_value, eigen_vector, iteration_count = engine.run()

        self._iteration_count = iteration_count

        return eigen_value, eigen_vector

    @property
    def iteration_count(self):
        return self._iteration_count
