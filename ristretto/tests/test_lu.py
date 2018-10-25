import numpy as np

from ristretto.lu import compute_rlu

from .utils import relative_error

atol_float32 = 1e-4
atol_float64 = 1e-8


# =============================================================================
# compute_rlu function
# =============================================================================
def test_compute_rlu_float64():
    m, k = 100, 10
    A = np.random.randn(m, k).astype(np.float64)
    A = A.dot(A.T)[:, :50]

    # ------------------------------------------------------------------------
    # test wth permute == False
    P, L, U, Q = compute_rlu(A, k, oversample=5, n_subspace=2, permute=False)
    Ak = P.dot(L.dot(U)).dot(Q)

    assert relative_error(A, Ak) < atol_float64

    # ------------------------------------------------------------------------
    # test wth permute == True
    L, U = compute_rlu(A, k, oversample=5, n_subspace=2, permute=True)
    Ak = L.dot(U)

    assert relative_error(A, Ak) < atol_float64


def test_compute_rlu_complex128():
    m, k = 100, 10
    A = np.random.randn(m, k).astype(np.float64) + \
            1j * np.random.randn(m, k).astype(np.float64)
    A = A.dot(A.conj().T)[:, :50]

    # ------------------------------------------------------------------------
    # test wth permute == False
    P, L, U, Q = compute_rlu(A, k, oversample=5, n_subspace=2, permute=False)
    Ak = P.dot(L.dot(U)).dot(Q)

    assert relative_error(A, Ak) < atol_float64

    # ------------------------------------------------------------------------
    # test wth permute == True
    L, U = compute_rlu(A, k, oversample=5, n_subspace=2, permute=True)
    Ak = L.dot(U)

    assert relative_error(A, Ak) < atol_float64
