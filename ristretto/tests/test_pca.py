import numpy as np

from ristretto.spca import compute_spca
from ristretto.spca import compute_rspca

from .utils import relative_error

atol_float16 = 1e-2
atol_float32 = 1e-4
atol_float64 = 1e-8

alpha = 1e-3
beta = 1e-5

def get_A():
    # Create Data
    m = 10000
    V1 = np.array(np.random.standard_normal(m) * 290).reshape(-1, 1)
    V2 = np.array(np.random.standard_normal(m) * 399).reshape(-1, 1)
    V3 = -0.1*V1 + 0.1*V2 + np.array(np.random.standard_normal(m) * 100).reshape(-1, 1)

    return np.concatenate((V1,V1,V1,V1, V2,V2,V2,V2, V3,V3), axis=1)


# =============================================================================
# spca function
# =============================================================================
def test_spca():
    A = get_A()

    Bstar, Astar, eigvals, obj = compute_spca(A, n_components=3, max_iter=100,
                                      alpha=alpha, beta=beta)

    A_pca = A.dot(Bstar).dot(Astar.T)
    assert relative_error(A, A_pca) < atol_float16


# =============================================================================
# robspca function
# =============================================================================
def test_robspca():
    gamma  = 10
    A = get_A()

    Bstar, Astar, eigvals, obj = compute_spca(A, n_components=3, max_iter=100, robust=True,
                                              alpha=alpha, beta=beta, gamma=gamma)

    A_pca = A.dot(Bstar).dot(Astar.T)
    assert relative_error(A, A_pca) < atol_float16


# =============================================================================
# rspca function
# =============================================================================
def test_rspca():
    A = get_A()

    Bstar, Astar, eigvals, obj = compute_rspca(A, n_components=3, oversample=10,
                                       n_subspace=2, max_iter=100,
                                       alpha=alpha, beta=beta)

    A_pca = A.dot(Bstar).dot(Astar.T)
    assert relative_error(A, A_pca) < atol_float16
