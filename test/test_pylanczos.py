import numpy as np
import unittest
from pylanczos import PyLanczos
from scipy.sparse import lil_matrix, csr_matrix


class PyLanczosTest(unittest.TestCase):
    def test_simple_matrix(self):
        matrix = np.array([[2, 1, 1],
                           [1, 2, 1],
                           [1, 1, 2]], dtype='float64')
        n = len(matrix)

        engine = PyLanczos(matrix, True, 1)
        eigenvalues, eigenvectors = engine.run()
        eigval = eigenvalues[0]
        eigvec = eigenvectors[:, 0]

        np.testing.assert_almost_equal(eigval, 4.0)
        sign = np.sign(eigvec[0])
        correct_eigvec = sign/np.sqrt(len(matrix))*np.array([1, 1, 1])
        np.testing.assert_allclose(eigvec, correct_eigvec)

    def test_simple_matrix_single_float(self):
        matrix = np.array([[2, 1, 1],
                           [1, 2, 1],
                           [1, 1, 2]], dtype='float32')

        engine = PyLanczos(matrix, True, 1)
        eigenvalues, eigenvectors = engine.run()
        eigval = eigenvalues[0]
        eigvec = eigenvectors[:, 0]

        np.testing.assert_almost_equal(eigval, 4.0, decimal=3)
        sign = np.sign(eigvec[0])
        correct_eigvec = sign/np.sqrt(len(matrix))*np.array([1, 1, 1], dtype='float32')
        np.testing.assert_allclose(eigvec, correct_eigvec, rtol=1e-3)

    def test_simple_matrix_multiple_eigenpairs(self):
        matrix = np.array(
            [[         6, np.sqrt(2), np.sqrt(2)],
             [np.sqrt(2),          1,          5],
             [np.sqrt(2),          5,          1]],
            dtype='float64') / 4
        n = len(matrix)

        engine = PyLanczos(matrix, True, 3)
        eigenvalues, eigenvectors = engine.run()

        correct_eigenvalues = np.array([2, 1, -1], dtype="float64")
        correct_eigenvectors = np.array(
            [[ np.sqrt(2), -np.sqrt(2),  0],
             [          1,          1,   1],
             [          1,          1,  -1]],
            dtype="float64") # Elements are defined so that v[1, :] are positive.

        np.testing.assert_allclose(eigenvalues, correct_eigenvalues)

        ## Adjust sign, adujust norm, and then compare
        for k in range(correct_eigenvectors.shape[1]):
            sign = 1 if eigenvectors[1, k] >= 0 else -1
            norm = np.linalg.norm(correct_eigenvectors[:, k])
            correct_eigenvectors[:, k] *= sign / norm

        np.testing.assert_allclose(eigenvectors, correct_eigenvectors, atol=np.finfo("float64").eps*1e1)

    def test_sparse_matrix_numpy(self):
        n = 10

        matrix = np.zeros((n, n), dtype='float64')
        for i in range(n-1):
            matrix[i, i+1] = -1
            matrix[i+1, i] = -1

        engine = PyLanczos(matrix, False, 1)
        eigenvalues, eigenvectors = engine.run()
        eigval = eigenvalues[0]
        eigvec = eigenvectors[:, 0]

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

        engine = PyLanczos(csr_matrix(matrix), False, 1)
        eigenvalues, eigenvectors = engine.run()
        eigval = eigenvalues[0]
        eigvec = eigenvectors[:, 0]

        np.testing.assert_almost_equal(eigval, -2.0*np.cos(np.pi/(n+1)))
        sign = np.sign(eigvec[0])

        correct_eigvec = sign * np.sin((1+np.array(range(n)))*np.pi/(n+1))
        correct_eigvec /= np.linalg.norm(correct_eigvec)

        np.testing.assert_allclose(eigvec, correct_eigvec)

    def test_sparse_matrix_dynamic(self):
        n = 10

        def mv_mul(v_in, v_out):
            for i in range(n-1):
                v_out[i] += -1.0*v_in[i+1]
                v_out[i+1] += -1.0*v_in[i]

        engine = PyLanczos.create_custom(mv_mul, n, 'float64', False, 1)
        eigenvalues, eigenvectors = engine.run()
        eigval = eigenvalues[0]
        eigvec = eigenvectors[:, 0]

        np.testing.assert_almost_equal(eigval, -2.0*np.cos(np.pi/(n+1)))
        sign = np.sign(eigvec[0])

        correct_eigvec = sign * np.sin((1+np.array(range(n)))*np.pi/(n+1))
        correct_eigvec /= np.linalg.norm(correct_eigvec)

        np.testing.assert_allclose(eigvec, correct_eigvec)


    def test_complex_matrix(self):
        matrix = np.array([[  0, 1j,  1],
                           [-1j,  0, 1j],
                           [  1, -1j, 0]], dtype='complex128')

        engine = PyLanczos(matrix, False, 1)
        eigenvalues, eigenvectors = engine.run()
        eigval = eigenvalues[0]
        eigvec = eigenvectors[:, 0]

        np.testing.assert_almost_equal(eigval, -2.0)
        phase = np.angle(eigvec[0])
        correct_eigvec = np.array([1, 1j, -1])
        correct_eigvec = np.exp(1j*phase)/np.linalg.norm(correct_eigvec)*correct_eigvec
        np.testing.assert_allclose(eigvec, correct_eigvec)

    def test_unsupported_dtype(self):
        matrix = np.array([[2, 1, 1],
                           [1, 2, 1],
                           [1, 1, 2]], dtype='int64')

        with self.assertRaises(Exception):
            PyLanczos(matrix, True)


if __name__ == '__main__':
    unittest.main()
