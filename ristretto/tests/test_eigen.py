import numpy as np

from ristretto.eigen import reigh
from ristretto.eigen import reigh_nystroem
from ristretto.eigen import reigh_nystroem_col

atol_float32 = 1e-4
atol_float64 = 1e-8


# =============================================================================
# reigh function
# =============================================================================
def test_reigh_float64():
    m, k = 100, 10
    A = np.array(np.random.randn(m, k), np.float64)
    A = A.dot(A.T)

    w, v = reigh(A, k=k, p=5, q=2)
    Ak = (v*w).dot(v.T)

    percent_error = 100 * np.linalg.norm(A - Ak) / np.linalg.norm(A)
    assert percent_error < atol_float64


def test_reigh_complex128():
    m, k = 100, 10
    A = np.array(np.random.randn(m, k), np.float64) + 1j * \
            np.array(np.random.randn(m, k), np.float64)
    A = A.dot(A.conj().T)

    w, v = reigh(A, k=k, p=10, q=2)
    Ak = (v*w).dot(v.conj().T)

    percent_error = 100 * np.linalg.norm(A - Ak) / np.linalg.norm(A)
    assert percent_error < atol_float64


# =============================================================================
# reig_nystroem function
# =============================================================================
def test_reig_nystroem_float64():
    m, k = 20, 10
    A = np.array(np.random.randn(m, k), np.float64)
    A = A.dot(A.T)

    w, v = reigh_nystroem(A, k=k, p=0, q=2)
    Ak = (v*w).dot(v.T)

    percent_error = 100 * np.linalg.norm(A - Ak) / np.linalg.norm(A)
    assert percent_error < atol_float64


def test_reig_nystroem_complex128():
    m, k = 20, 10
    A = np.array(np.random.randn(m, k), np.float64) + 1j * np.array(np.random.randn(m, k), np.float64)
    A = A.dot(A.conj().T)

    w, v = reigh_nystroem(A, k=k, p=0, q=2)
    Ak = (v*w).dot(v.conj().T)

    percent_error = 100 * np.linalg.norm(A - Ak) / np.linalg.norm(A)
    assert percent_error < atol_float64


# =============================================================================
# reig_nystroem_col function
# =============================================================================
def test_reig_nystroem_col_float64():
    m, k = 20, 10
    A = np.array(np.random.randn(m, k), np.float64)
    A = A.dot(A.T)

    w, v = reigh_nystroem_col(A, k=k, p=0)
    Ak = (v*w).dot(v.T)

    percent_error = 100 * np.linalg.norm(A - Ak) / np.linalg.norm(A)
    assert percent_error < atol_float64


def test_reig_nystroem_col_complex128():
    m, k = 20, 10
    A = np.array(np.random.randn(m, k), np.float64) + 1j * np.array(np.random.randn(m, k), np.float64)
    A = A.dot(A.conj().T)

    w, v = reigh_nystroem_col(A, k=k, p=0)
    Ak = (v*w).dot(v.conj().T)

    percent_error = 100 * np.linalg.norm(A - Ak) / np.linalg.norm(A)
    assert percent_error < atol_float64
