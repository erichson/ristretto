import numpy as np

from ristretto.mf import rsvd_single


atol_float32 = 1e-4
atol_float64 = 1e-8


def test_rsvd_single_float64():
    m, k = 100, 10
    A = np.array(np.random.randn(m, k), np.float64)
    A = A.dot( A.T )
    U, s, Vt = rsvd_single(A, k=k, p=5)
    Ak = U.dot(np.diag(s).dot(Vt))
    percent_error = 100 * np.linalg.norm(A - Ak) / np.linalg.norm(A)

    assert percent_error < atol_float64


def test_rsvd_single_complex128():
    m, k = 100, 10
    A = np.array(np.random.randn(m, k), np.float64) + 1j * \
            np.array(np.random.randn(m, k), np.float64)
    A = A.dot(A.conj().T)
    U, s, Vh = rsvd_single(A, k=k, p=5)
    Ak = U.dot(np.diag(s).dot(Vh))
    percent_error = 100 * np.linalg.norm(A - Ak) / np.linalg.norm(A)


def test_rsvd_single_orthogonal_complex128():
    m, k = 100, 10
    A = np.array(np.random.randn(m, k), np.float64) + 1j * \
            np.array(np.random.randn(m, k), np.float64)
    A = A.dot(A.conj().T)
    U, s, Vh = rsvd_single(A, k=k, p=5, sdist='orthogonal')
    Ak = U.dot(np.diag(s).dot(Vh))
    percent_error = 100 * np.linalg.norm(A - Ak) / np.linalg.norm(A)
