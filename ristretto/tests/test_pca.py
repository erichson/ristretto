import numpy as np

from ristretto.pca import spca
from ristretto.pca import rspca
from ristretto.pca import robspca

atol_float32 = 1e-4
atol_float64 = 1e-8


# =============================================================================
# spca function
# =============================================================================
def test_spca():
    # Create Data
    m = 10000
    V1 = np.array(np.random.standard_normal(m) * 290).reshape(-1, 1)
    V2 = np.array(np.random.standard_normal(m) * 399).reshape(-1, 1)
    V3 = -0.1*V1 + 0.1*V2 + np.array(np.random.standard_normal(m) * 100).reshape(-1, 1)

    A = np.concatenate((V1,V1,V1,V1, V2,V2,V2,V2, V3,V3), axis=1)

    alpha  = 1e-3
    beta  = 1e-5


    Bstar, Astar, eigvals, obj = spca(A, n_components=3, max_iter=100,
                                      alpha=alpha, beta=beta, verbose=0)

    relative_error = (np.linalg.norm(A - A.dot(Bstar).dot(Astar.T)) / np.linalg.norm(A))
    assert relative_error < 1e-2

# =============================================================================
# rspca function
# =============================================================================
def test_rspca():

    # Create Data
    m = 10000
    V1 = np.array(np.random.standard_normal(m) * 290).reshape(-1, 1)
    V2 = np.array(np.random.standard_normal(m) * 399).reshape(-1, 1)
    V3 = -0.1*V1 + 0.1*V2 + np.array(np.random.standard_normal(m) * 100).reshape(-1, 1)

    A = np.concatenate((V1,V1,V1,V1, V2,V2,V2,V2, V3,V3), axis=1)

    alpha  = 1e-3
    beta  = 1e-5

    Bstar, Astar, eigvals, obj = rspca(A, n_components=3, p=10, q=2, max_iter=100,
                                       alpha=alpha, beta=beta, verbose=0)

    relative_error = (np.linalg.norm(A - A.dot(Bstar).dot(Astar.T)) / np.linalg.norm(A))
    assert relative_error < 1e-2


# =============================================================================
# robspca function
# =============================================================================
def test_robspca():
    # Create Data
    m = 10000
    V1 = np.array(np.random.standard_normal(m) * 290).reshape(-1, 1)
    V2 = np.array(np.random.standard_normal(m) * 399).reshape(-1, 1)
    V3 = -0.1*V1 + 0.1*V2 + np.array(np.random.standard_normal(m) * 100).reshape(-1, 1)

    A = np.concatenate((V1,V1,V1,V1, V2,V2,V2,V2, V3,V3), axis=1)

    alpha  = 1e-3
    beta  = 1e-5
    gamma  = 10


    Bstar, Astar, S, eigvals, obj = robspca(A, n_components=3, max_iter=100,
                                            alpha=alpha, beta=beta, gamma=gamma, verbose=0)

    relative_error = (np.linalg.norm(A - A.dot(Bstar).dot(Astar.T)) / np.linalg.norm(A))
    assert relative_error < 1e-2
