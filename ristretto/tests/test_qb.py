import numpy as np

from ristretto.qb import compute_rqb

from .utils import relative_error

atol_float32 = 1e-4
atol_float64 = 1e-8


# =============================================================================
# rqb function
# =============================================================================
def test_rqb_float64():
    m, k = 100, 10
    A = np.random.randn(m, k).astype(np.float64)
    A = A.dot(A.T)

    Q, B = compute_rqb(A, k, oversample=5, n_subspace=2)
    Ak = Q.dot(B)

    assert relative_error(A, Ak) < atol_float64


def test_rqb_complex128():
    m, k = 100, 10
    A = np.random.randn(m, k).astype(np.float64) + \
            1j * np.random.randn(m, k).astype(np.float64)
    A = A.dot(A.conj().T)

    Q, B = compute_rqb(A, k, oversample=5, n_subspace=2)
    Ak = Q.dot(B)

    assert relative_error(A, Ak) < atol_float64


# =============================================================================
# blocked rqb function
# =============================================================================
def test_rqb_block_float64():
    m, k = 100, 10
    A = np.random.randn(m, k).astype(np.float64)
    A = A.dot(A.T)

    Q, B = compute_rqb(A, k, oversample=5, n_subspace=2, n_blocks=4)
    Ak = Q.dot(B)

    assert relative_error(A, Ak) < atol_float64


def test_rqb_block_wide_float64():
    m, k = 100, 10
    A = np.random.randn(m, k).astype(np.float64)
    A = A.dot(A.T)
    A = A[0:80,:]

    Q, B = compute_rqb(A, k, oversample=5, n_subspace=2, n_blocks=4)
    Ak = Q.dot(B)

    assert relative_error(A, Ak) < atol_float64


def test_rqb_block_complex128():
    m, k = 100, 10
    A = np.random.randn(m, k).astype(np.float64) + \
            1j * np.random.randn(m, k).astype(np.float64)
    A = A.dot(A.conj().T)

    Q, B = compute_rqb(A, k, oversample=5, n_subspace=2, n_blocks=4)
    Ak = Q.dot(B)

    assert relative_error(A, Ak) < atol_float64


def test_rqb_block_wide_complex128():
    m, k = 100, 10
    A = np.random.randn(m, k).astype(np.float64) + \
            1j * np.random.randn(m, k).astype(np.float64)
    A = A.dot(A.conj().T)
    A = A[0:40,:]

    Q, B = compute_rqb(A, k, oversample=5, n_subspace=2, n_blocks=4)
    Ak = Q.dot(B)

    assert relative_error(A, Ak) < atol_float64
