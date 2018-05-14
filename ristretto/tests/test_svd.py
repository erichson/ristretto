import numpy as np

from ristretto.svd import rsvd

atol_float32 = 1e-4
atol_float64 = 1e-8


# =============================================================================
# rsvd function
# =============================================================================
def test_rsvd_float64():
    m, k = 100, 10
    A = np.array(np.random.randn(m, k), np.float64)
    A = A.dot( A.T )
    U, s, Vt = rsvd(A, k=k, p=5, q=2)
    Ak = U.dot(np.diag(s).dot(Vt))
    percent_error = 100 * np.linalg.norm(A - Ak) / np.linalg.norm(A)

    assert percent_error < atol_float64


def test_rsvd_complex128():
    m, k = 100, 10
    A = np.array(np.random.randn(m, k), np.float64) + 1j * \
            np.array(np.random.randn(m, k), np.float64)
    A = A.dot(A.conj().T)
    U, s, Vh = rsvd(A, k=k, p=5, q=2)
    Ak = U.dot(np.diag(s).dot(Vh))
    percent_error = 100 * np.linalg.norm(A - Ak) / np.linalg.norm(A)

    assert percent_error < atol_float64


def test_rsvd_fliped_float64():
    m, k = 100, 10
    A = np.array(np.random.randn(m, k), np.float64)
    A = A.dot(A.T)
    A = A[:,0:50]
    U, s, Vh = rsvd(A.T, k=k, p=5, q=2)
    Ak = U.dot(np.diag(s).dot(Vh))
    percent_error = 100 * np.linalg.norm(A.T - Ak) / np.linalg.norm(A.T)

    assert percent_error < atol_float64


def test_rsvd_fliped_complex128():
    m, k = 100, 10
    A = np.array(np.random.randn(m, k), np.float64) + 1j * \
            np.array(np.random.randn(m, k), np.float64)
    A = A.dot(A.conj().T)
    A = A[:,0:50]
    U, s, Vh = rsvd(A.conj().T, k=k, p=5, q=2)
    Ak = U.dot(np.diag(s).dot(Vh))
    percent_error = 100 * np.linalg.norm(A.conj().T - Ak) / np.linalg.norm(A.conj().T)

    assert percent_error < atol_float64


def test_rsvd_single_float64():
    m, k = 100, 10
    A = np.array(np.random.randn(m, k), np.float64)
    A = A.dot( A.T )
    U, s, Vt = rsvd(A, k=k, p=5, single_pass=True)
    Ak = U.dot(np.diag(s).dot(Vt))
    percent_error = 100 * np.linalg.norm(A - Ak) / np.linalg.norm(A)

    assert percent_error < atol_float64


def test_rsvd_single_complex128():
    m, k = 100, 10
    A = np.array(np.random.randn(m, k), np.float64) + 1j * \
            np.array(np.random.randn(m, k), np.float64)
    A = A.dot(A.conj().T)
    U, s, Vh = rsvd(A, k=k, p=5, single_pass=True)
    Ak = U.dot(np.diag(s).dot(Vh))
    percent_error = 100 * np.linalg.norm(A - Ak) / np.linalg.norm(A)


def test_rsvd_single_orthogonal_complex128():
    m, k = 100, 10
    A = np.array(np.random.randn(m, k), np.float64) + 1j * \
            np.array(np.random.randn(m, k), np.float64)
    A = A.dot(A.conj().T)
    U, s, Vh = rsvd(A, k=k, p=5, sdist='orthogonal', single_pass=True)
    Ak = U.dot(np.diag(s).dot(Vh))
    percent_error = 100 * np.linalg.norm(A - Ak) / np.linalg.norm(A)
