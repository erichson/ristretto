import numpy as np

from ristretto.cur import cur
from ristretto.cur import rcur

atol_float32 = 1e-4
atol_float64 = 1e-8


# =============================================================================
# cur function
# =============================================================================
def test_cur():
    m, k = 100, 10
    A = np.array(np.random.randn(m, k), np.float64)
    A = A.dot(A.T)
    A = A[:,0:50]

    # index_set = False
    C, U, R = cur(A, k=k+2, index_set=False)
    relative_error = (np.linalg.norm(A - C.dot(U).dot(R)) / np.linalg.norm(A))
    assert relative_error < 1e-4

    # index_set = True
    C, U, R = cur(A, k=k+2, index_set=True)
    relative_error = (np.linalg.norm(A - A[:,C].dot(U).dot(A[R,:])) / np.linalg.norm(A))
    assert relative_error < 1e-4


# =============================================================================
# rcur function
# =============================================================================
def test_rcur():
    m, k = 100, 10
    A = np.array(np.random.randn(m, k), np.float64)
    A = A.dot(A.T)
    A = A[:,0:50]

    # index_set = False
    C, U, R = rcur(A, k=k+2, index_set=False)
    relative_error = (np.linalg.norm(A - C.dot(U).dot(R)) / np.linalg.norm(A))
    assert relative_error < 1e-4

    # index_set = True
    C, U, R = rcur(A, k=k+2, index_set=True)
    relative_error = (np.linalg.norm(A - A[:,C].dot(U).dot(A[R,:])) / np.linalg.norm(A))
    assert relative_error < 1e-4
