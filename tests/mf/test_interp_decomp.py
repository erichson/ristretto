import numpy as np

from ristretto.mf import interp_decomp, rinterp_decomp, rinterp_decomp_qb


atol_float32 = 1e-4
atol_float64 = 1e-8


def test_id_col():
    m, k = 100, 10
    A = np.array(np.random.randn(m, k), np.float64)
    A = A.dot(A.T)
    A = A[:,0:50]

    # index_set = False
    C, V = interp_decomp(A, k=k+2, mode='column', index_set=False)
    relative_error = (np.linalg.norm(A - C.dot(V)) / np.linalg.norm(A))
    assert relative_error < 1e-4

    # index_set = True
    C, V = interp_decomp(A, k=k+2, mode='column', index_set=True)
    relative_error = (np.linalg.norm(A - A[:,C].dot(V)) / np.linalg.norm(A))
    assert relative_error < 1e-4

def test_id_row():
    m, k = 100, 10
    A = np.array(np.random.randn(m, k), np.float64)
    A = A.dot(A.T)
    A = A[:,0:50]

    # index_set = False
    Z, R = interp_decomp(A, k=k+2, mode='row', index_set=False)
    relative_error = (np.linalg.norm(A - Z.dot(R)) / np.linalg.norm(A))
    assert relative_error < 1e-4

    # index_set = True
    Z, R = interp_decomp(A, k=k+2, mode='row', index_set=True)
    relative_error = (np.linalg.norm(A - Z.dot(A[R,:])) / np.linalg.norm(A))
    assert relative_error < 1e-4


def test_rid_col():
    m, k = 100, 10
    A = np.array(np.random.randn(m, k), np.float64)
    A = A.dot(A.T)
    A = A[:,0:50]

    # index_set = False
    C, V = rinterp_decomp(A, k=k+2, mode='column', index_set=False)
    relative_error = (np.linalg.norm(A - C.dot(V)) / np.linalg.norm(A))
    assert relative_error < 1e-4

    # index_set = True
    C, V = rinterp_decomp(A, k=k+2, mode='column', index_set=True)
    relative_error = (np.linalg.norm(A - A[:,C].dot(V)) / np.linalg.norm(A))
    assert relative_error < 1e-4


def test_rid_row():
    m, k = 100, 10
    A = np.array(np.random.randn(m, k), np.float64)
    A = A.dot(A.T)
    A = A[:,0:50]

    # index_set = False
    Z, R = rinterp_decomp(A, k=k+2, mode='row', index_set=False)
    relative_error = (np.linalg.norm(A - Z.dot(R)) / np.linalg.norm(A))
    assert relative_error < 1e-4

    # index_set = True
    Z, R = rinterp_decomp(A, k=k+2, mode='row', index_set=True)
    relative_error = (np.linalg.norm(A - Z.dot(A[R,:])) / np.linalg.norm(A))
    assert relative_error < 1e-4


def test_ridqb_col():
    m, k = 100, 10
    A = np.array(np.random.randn(m, k), np.float64)
    A = A.dot(A.T)
    A = A[:,0:50]

    # index_set = False
    C, V = rinterp_decomp_qb(A, k=k+2, mode='column', index_set=False)
    relative_error = (np.linalg.norm(A - C.dot(V)) / np.linalg.norm(A))
    assert relative_error < 1e-4

    # index_set = True
    C, V = rinterp_decomp_qb(A, k=k+2, mode='column', index_set=True)
    relative_error = (np.linalg.norm(A - A[:,C].dot(V)) / np.linalg.norm(A))
    assert relative_error < 1e-4


def test_ridqb_row():
    m, k = 100, 10
    A = np.array(np.random.randn(m, k), np.float64)
    A = A.dot(A.T)
    A = A[:,0:50]

    # index_set = False
    Z, R = rinterp_decomp_qb(A, k=k+2, mode='row', index_set=False)
    relative_error = (np.linalg.norm(A - Z.dot(R)) / np.linalg.norm(A))
    assert relative_error < 1e-4

    # index_set = True
    Z, R = rinterp_decomp_qb(A, k=k+2, mode='row', index_set=True)
    relative_error = (np.linalg.norm(A - Z.dot(A[R,:])) / np.linalg.norm(A))
    assert relative_error < 1e-4
