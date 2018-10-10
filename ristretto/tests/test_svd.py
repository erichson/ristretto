import numpy as np

from ristretto.svd import compute_rsvd

from .utils import relative_error

atol_float32 = 1e-4
atol_float64 = 1e-8


# =============================================================================
# compute_rsvd function
# =============================================================================
def test_compute_rsvd_float64():
    m, k = 100, 10
    A = np.random.randn(m, k).astype(np.float64)
    A = A.dot(A.T)

    # ------------------------------------------------------------------------
    # test normal
    U, s, Vt = compute_rsvd(A, k, oversample=5, n_subspace=2)
    Ak = U.dot(np.diag(s).dot(Vt))

    assert relative_error(A, Ak) < atol_float64

    # ------------------------------------------------------------------------
    # test transposed
    A = A[:, :50].T

    U, s, Vt = compute_rsvd(A, k, oversample=5, n_subspace=2)
    Ak = U.dot(np.diag(s).dot(Vt))

    assert relative_error(A, Ak) < atol_float64


def test_compute_rsvd_complex128():
    m, k = 100, 10
    A = np.random.randn(m, k).astype(np.float64) + \
            1j * np.random.randn(m, k).astype(np.float64)
    A = A.dot(A.conj().T)

    # ------------------------------------------------------------------------
    # test normal
    U, s, Vt = compute_rsvd(A, k, oversample=5, n_subspace=2)
    Ak = U.dot(np.diag(s).dot(Vt))

    assert relative_error(A, Ak) < atol_float64

    # ------------------------------------------------------------------------
    # test transposed
    A = A[:, :50].conj().T

    U, s, Vt = compute_rsvd(A, k, oversample=5, n_subspace=2)
    Ak = U.dot(np.diag(s).dot(Vt))

    assert relative_error(A, Ak) < atol_float64
