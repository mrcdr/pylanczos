import numpy as np
import unittest
from pylanczos import Exponentiator
from scipy.sparse import lil_matrix, csr_matrix


class ExponentiatorTest(unittest.TestCase):
    def test_exponentiate_real(self):
        matrix = np.array([[2, 1, 1], [1, 2, 1], [1, 1, 2]], dtype="float64")
        input = np.array([1, 0, 0], dtype="float64")
        a = 3
        engine = Exponentiator(matrix)
        output = engine.run(a, input)

        w, v = np.linalg.eig(matrix)
        expected_output = v @ np.diag(np.exp(a * w)) @ v.T @ input
        overlap = np.abs(
            np.dot(output, expected_output)
            / np.linalg.norm(output)
            / np.linalg.norm(expected_output)
        )

        np.testing.assert_almost_equal(overlap, 1.0)

    def test_exponentiate_large_matrix(self):
        t = -1
        n = 100
        matrix = lil_matrix((n, n), dtype="complex128")

        for i in range(n - 1):
            matrix[i, i + 1] = t
            matrix[i + 1, i] = t

        matrix[-1, 0] = t
        matrix[0, -1] = t

        input = np.zeros(n, dtype="complex128")
        input[0] = 1 + 2j
        input[n - 1] = 1 + 2j
        input[n // 2] = 8 + 2j
        input = input / np.linalg.norm(input)  # normalize

        a = 3j

        engine = Exponentiator(csr_matrix(matrix))
        output = engine.run(a, input)

        w, v = ExponentiatorTest.make_plane_wave(t, n)
        expected_output = v @ np.diag(np.exp(a * w)) @ v.conj().T @ input

        overlap = np.abs(
            np.dot(output.conj(), expected_output)
            / np.linalg.norm(output)
            / np.linalg.norm(expected_output)
        )

        np.testing.assert_almost_equal(overlap, 1.0)

    def test_exponentiate_zero_delta(self):
        t = -1
        n = 100
        matrix = lil_matrix((n, n), dtype="complex128")

        for i in range(n - 1):
            matrix[i, i + 1] = t
            matrix[i + 1, i] = t

        matrix[-1, 0] = t
        matrix[0, -1] = t

        input = np.zeros(n, dtype="complex128")
        input[0] = 1 + 2j
        input[n - 1] = 1 + 2j
        input[n // 2] = 8 + 2j
        input = input / np.linalg.norm(input)  # normalize

        a = 0 + 0j

        engine = Exponentiator(csr_matrix(matrix))
        output = engine.run(a, input)

        expected_output = input

        overlap = np.abs(
            np.dot(output.conj(), expected_output)
            / np.linalg.norm(output)
            / np.linalg.norm(expected_output)
        )

        np.testing.assert_almost_equal(overlap, 1.0)

    def test_exponentiate_large_matrix_dynamic(self):
        t = -1
        n = 100

        def mv_mul(v_in, v_out):
            for i in range(n - 1):
                v_out[i] += t * v_in[i + 1]
                v_out[i + 1] += t * v_in[i]

            v_out[-1] += t * v_in[0]
            v_out[0] += t * v_in[-1]

        input = np.zeros(n, dtype="complex128")
        input[0] = 1 + 2j
        input[n - 1] = 1 + 2j
        input[n // 2] = 8 + 2j
        input = input / np.linalg.norm(input)  # normalize

        a = 3j

        engine = Exponentiator.create_custom(mv_mul, n, dtype="complex128")
        output = engine.run(a, input)

        w, v = ExponentiatorTest.make_plane_wave(t, n)
        expected_output = v @ np.diag(np.exp(a * w)) @ v.conj().T @ input

        overlap = np.abs(
            np.dot(output.conj(), expected_output)
            / np.linalg.norm(output)
            / np.linalg.norm(expected_output)
        )

        np.testing.assert_almost_equal(overlap, 1.0)

    def make_plane_wave(t, n):
        ks = 2 * np.pi / n * np.arange(n)
        es = 2 * t * np.cos(ks)

        u = np.ndarray((n, n), dtype="complex128")

        for j, k in enumerate(ks):
            u[:, j] = np.exp(1j * k * np.arange(n)) / np.sqrt(n)

        return (es, u)


if __name__ == "__main__":
    unittest.main()
