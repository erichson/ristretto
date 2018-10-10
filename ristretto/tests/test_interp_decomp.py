from __future__ import division

import numpy as np
from scipy import linalg

from ristretto.interp_decomp import compute_interp_decomp
from ristretto.interp_decomp import compute_rinterp_decomp

from .utils import relative_error

atol_float32 = 1e-4
atol_float64 = 1e-8


# =============================================================================
# compute_interp_decomp function
# =============================================================================
def test_id_col():
    m, k = 100, 10
    A = np.random.randn(m, k).astype(np.float64)
    A = A.dot(A.T)[:, :50]

    # ------------------------------------------------------------------------
    # test index_set == False
    C, V = compute_interp_decomp(A, rank=k+2, mode='column', index_set=False)
    A_id = C.dot(V)
    assert relative_error(A, A_id) < atol_float32

    # ------------------------------------------------------------------------
    # test index_set == True
    C, V = compute_interp_decomp(A, rank=k+2, mode='column', index_set=True)
    A_id = A[:, C].dot(V)
    assert relative_error(A, A_id) < atol_float32


def test_id_row():
    m, k = 100, 10
    A = np.random.randn(m, k).astype(np.float64)
    A = A.dot(A.T)[:, :50]

    # ------------------------------------------------------------------------
    # test index_set == False
    Z, R = compute_interp_decomp(A, k+2, mode='row', index_set=False)
    A_id = Z.dot(R)
    assert relative_error(A, A_id) < atol_float32

    # ------------------------------------------------------------------------
    # test index_set == True
    Z, R = compute_interp_decomp(A, k+2, mode='row', index_set=True)
    A_id = Z.dot(A[R, :])
    assert relative_error(A, A_id) < atol_float32


# =============================================================================
# compute_rinterp_decomp function
# =============================================================================
def test_rid_col():
    m, k = 100, 10
    A = np.random.randn(m, k).astype(np.float64)
    A = A.dot(A.T)[:, :50]

    # index_set = False
    C, V = compute_rinterp_decomp(A, k+2, mode='column', index_set=False)
    A_id = C.dot(V)
    assert relative_error(A, A_id) < atol_float32

    # index_set = True
    C, V = compute_rinterp_decomp(A, k+2, mode='column', index_set=True)
    A_id = A[:, C].dot(V)
    assert relative_error(A, A_id) < atol_float32


def test_rid_row():
    m, k = 100, 10
    A = np.random.randn(m, k).astype(np.float64)
    A = A.dot(A.T)[:, :50]

    # ------------------------------------------------------------------------
    # test index_set == False
    Z, R = compute_rinterp_decomp(A, k+2, mode='row', index_set=False)
    A_id = Z.dot(R)
    assert relative_error(A, A_id) < atol_float32

    # ------------------------------------------------------------------------
    # test index_set == True
    Z, R = compute_rinterp_decomp(A, k+2, mode='row', index_set=True)
    A_id = Z.dot(A[R, :])
    assert relative_error(A, A_id) < atol_float32
