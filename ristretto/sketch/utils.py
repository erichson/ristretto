"""
Module containing utility functions for
"""
from scipy import linalg


def orthonormalize(A, overwrite_a=True, check_finite=False):
    """orthonormalize the columns of A via QR decomposition"""
    # NOTE: for A(m, n) 'economic' returns Q(m, k), R(k, n) where k is min(m, n)
    # TODO: when does overwrite_a even work? (fortran?)
    Q, _ = linalg.qr(A, overwrite_a=overwrite_a, check_finite=check_finite,
                     mode='economic', pivoting=False)
    return Q


def perform_subspace_iterations(A, Q, n_iter=2, axis=1):
    """perform subspace iterations on Q"""
    # TODO: can we figure out how not to transpose for row wise
    if axis == 0:
        Q = Q.T

    # orthonormalize Y, overwriting
    Q = orthonormalize(Q)

    # perform subspace iterations
    for _ in range(n_iter):
        if axis == 0:
            Z = orthonormalize(A.dot(Q))
            Q = orthonormalize(A.T.dot(Z))
        else:
            Z = orthonormalize(A.T.dot(Q))
            Q = orthonormalize(A.dot(Z))

    if axis == 0:
        return Q.T
    return Q
