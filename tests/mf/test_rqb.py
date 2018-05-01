import numpy as np

from ristretto.mf import rqb


atol_float32 = 1e-4
atol_float64 = 1e-8


def test_rqb_float64():
    m, k = 100, 10
    A = np.array(np.random.randn(m, k), np.float64)
    A = A.dot( A.T )
    Q, B = rqb(A, k=k, p=5, q=2)
    Ak = Q.dot(B)
    percent_error = 100 * np.linalg.norm(A - Ak) / np.linalg.norm(A)

    assert percent_error < atol_float64


def test_rqb_complex128():
    m, k = 100, 10
    A = np.array(np.random.randn(m, k), np.float64) + 1j * \
            np.array(np.random.randn(m, k), np.float64)
    A = A.dot(A.conj().T)
    Q, B = rqb(A, k=k, p=5, q=2)
    Ak = Q.dot(B)
    percent_error = 100 * np.linalg.norm(A - Ak) / np.linalg.norm(A)

    assert percent_error < atol_float64
