"""
Module containing utility functions for 
"""
from scipy import linalg


def orthonormalize(A, overwrite_a=True, check_finite=False):
    """orthonormalize the columns of A via QR decomposition"""
    # NOTE: for A(m, n) 'economic' returns Q(m, k), R(k, n) where k is min(m, n)
    Q, _ = linalg.qr(A, overwrite_a=overwrite_a, check_finite=check_finite,
                     mode='economic', pivoting=False)
    return Q


def perform_subspace_iterations(A, Q, n_iter=1, axis=0):
    """perform subspace iterations on Q"""
    # orthonormalize Y, overwriting
    Q = orthonormalize(Q)

    # perform subspace iterations
    for _ in range(n_iter):
        if axis == 0:
            Z = orthonormalize(A.dot(Q.T))
            Q = orthonormalize(A.T.dot(Z))
        else:
            Z = orthonormalize(A.T.dot(Q))
            Q = orthonormalize(A.dot(Z))

    return Q
