import numpy as np
import unittest
from pylanczos import PyLanczos
from scipy.sparse import lil_matrix, csr_matrix


class PyLanczosTest(unittest.TestCase):
    def test_simple_matrix(self):
        matrix = np.array([[2, 1, 1],
                           [1, 2, 1],
                           [1, 1, 2]], dtype='float64')

        engine = PyLanczos(matrix, True)
        eigval, eigvec, itern = engine.run()

        np.testing.assert_almost_equal(eigval, 4.0)
        sign = np.sign(eigvec[0])
        correct_eigvec = sign/np.sqrt(len(matrix))*np.array([1, 1, 1])
        np.testing.assert_allclose(eigvec, correct_eigvec)

    def test_simple_matrix_single_float(self):
        matrix = np.array([[2, 1, 1],
                           [1, 2, 1],
                           [1, 1, 2]], dtype='float32')

        engine = PyLanczos(matrix, True)
        eigval, eigvec, itern = engine.run()

        np.testing.assert_almost_equal(eigval, 4.0, decimal=3)
        sign = np.sign(eigvec[0])
        correct_eigvec = sign/np.sqrt(len(matrix))*np.array([1, 1, 1], dtype='float32')
        np.testing.assert_allclose(eigvec, correct_eigvec, rtol=1e-3)

    def test_sparse_matrix_numpy(self):
        n = 10

        matrix = np.zeros((n, n), dtype='float64')
        for i in range(n-1):
            matrix[i, i+1] = -1
            matrix[i+1, i] = -1

        engine = PyLanczos(matrix, False)
        eigval, eigvec, itern = engine.run()

        np.testing.assert_almost_equal(eigval, -2.0*np.cos(np.pi/(n+1)))
        sign = np.sign(eigvec[0])

        correct_eigvec = sign * np.sin((1+np.array(range(n)))*np.pi/(n+1))
        correct_eigvec /= np.linalg.norm(correct_eigvec)

        np.testing.assert_allclose(eigvec, correct_eigvec)

    def test_sparse_matrix_scipy(self):
        n = 10

        matrix = lil_matrix((n, n), dtype='float64')

        for i in range(n-1):
            matrix[(i, i+1)] = -1
            matrix[(i+1, i)] = -1

        engine = PyLanczos(csr_matrix(matrix), False)
        eigval, eigvec, itern = engine.run()

        np.testing.assert_almost_equal(eigval, -2.0*np.cos(np.pi/(n+1)))
        sign = np.sign(eigvec[0])

        correct_eigvec = sign * np.sin((1+np.array(range(n)))*np.pi/(n+1))
        correct_eigvec /= np.linalg.norm(correct_eigvec)

        np.testing.assert_allclose(eigvec, correct_eigvec)

    def test_complex_matrix(self):
        matrix = np.array([[  0, 1j,  1],
                           [-1j,  0, 1j],
                           [  1, -1j, 0]], dtype='complex128')

        engine = PyLanczos(matrix, False)
        eigval, eigvec, itern = engine.run()

        np.testing.assert_almost_equal(eigval, -2.0)
        phase = np.angle(eigvec[0])
        correct_eigvec = np.array([1, 1j, -1])
        correct_eigvec = np.exp(1j*phase)/np.linalg.norm(correct_eigvec)*correct_eigvec
        np.testing.assert_allclose(eigvec, correct_eigvec)


if __name__ == '__main__':
    unittest.main()
