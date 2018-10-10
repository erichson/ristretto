from __future__ import division

import numpy as np

from ristretto.cur import compute_cur, compute_rcur

from .utils import relative_error

atol_float32 = 1e-4
atol_float64 = 1e-8


# =============================================================================
# compute_cur function
# =============================================================================
def test_compute_cur():
    m, k = 100, 10
    A = np.random.randn(m, k).astype(np.float64)
    A = A.dot(A.T)[:, :50]

    # index_set = False
    C, U, R = compute_cur(A, rank=k+2, index_set=False)
    A_cur = C.dot(U).dot(R)
    assert relative_error(A, A_cur) < atol_float32

    # index_set = True
    C, U, R = compute_cur(A, rank=k+2, index_set=True)
    A_cur = A[:, C].dot(U).dot(A[R])
    assert relative_error(A, A_cur) < atol_float32


# =============================================================================
# compute_rcur function
# =============================================================================
def test_compute_rcur():
    m, k = 100, 10
    A = np.random.randn(m, k).astype(np.float64)
    A = A.dot(A.T)[:, :50]

    # index_set = False
    C, U, R = compute_rcur(A, k+2, index_set=False)
    A_cur = C.dot(U).dot(R)
    assert relative_error(A, A_cur) < atol_float32

    # index_set = True
    C, U, R = compute_rcur(A, k+2, index_set=True)
    A_cur = A[:, C].dot(U).dot(A[R])
    assert relative_error(A, A_cur) < atol_float32
