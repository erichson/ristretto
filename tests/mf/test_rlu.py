import numpy as np

from ristretto.mf import rlu


atol_float32 = 1e-4
atol_float64 = 1e-8


def test_rlu_float64():
    m, k = 100, 10
    A = np.array(np.random.randn(m, k), np.float64)
    A = A.dot(A.T)
    A = A[:,0:50]
    P, L, U, Q = rlu(A, permute=False, k=k, p=5, q=2)
    Ak = P.dot(L.dot(U)).dot(Q)
    percent_error = 100 * np.linalg.norm(A - Ak) / np.linalg.norm(A)

    assert percent_error < atol_float64


def test_rlu_complex128():
    m, k = 100, 10
    A = np.array(np.random.randn(m, k), np.float64) + 1j * \
            np.array(np.random.randn(m, k), np.float64)
    A = A.dot(A.conj().T)
    A = A[:,0:50]
    P, L, U, Q = rlu(A, permute=False, k=k, p=5, q=2)
    Ak = P.dot(L.dot(U)).dot(Q)
    percent_error = 100 * np.linalg.norm(A - Ak) / np.linalg.norm(A)

    assert percent_error < atol_float64

def test_rlu_permute_float64():
    m, k = 100, 10
    A = np.array(np.random.randn(m, k), np.float64)
    A = A.dot(A.T)
    A = A[:,0:50]
    L, U, = rlu(A, permute=True, k=k, p=5, q=2)
    Ak = L.dot(U)
    percent_error = 100 * np.linalg.norm(A - Ak) / np.linalg.norm(A)

    assert percent_error < atol_float64


def test_rlu_permute_complex128():
    m, k = 100, 10
    A = np.array(np.random.randn(m, k), np.float64) + 1j * np.array(np.random.randn(m, k), np.float64)
    A = A.dot(A.conj().T)
    A = A[:,0:50]
    L, U= rlu(A, permute=True, k=k, p=5, q=2)
    Ak = L.dot(U)
    percent_error = 100 * np.linalg.norm(A - Ak) / np.linalg.norm(A)

    assert percent_error < atol_float64
