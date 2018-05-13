"""
Randomized Singular Value Decomposition
"""
# TODO: Add option for sparse random test matrices.
# TODO:  Modify algorithm to allow for the streaming model.
# Authors: N. Benjamin Erichson
#          Joseph Knox
# License: GNU General Public License v3.0

import numpy as np
from scipy import linalg

from .rqb import rqb_single
from ..utils import conjugate_transpose


def rsvd_single(A, k=None, p=10, l=None, sdist='uniform'):
    """Randomized Singular Value Decomposition Single-View.

    Randomized algorithm for computing the approximate low-rank singular value
    decomposition of a rectangular `(m, n)` matrix `A`, with target rank `k << min{m, n}`.
    The input matrix is factored as `A = U * diag(s) * Vt`. The right singular
    vectors are the columns of the real or complex unitary matrix `U`. The left
    singular vectors are the columns of the real or complex unitary matrix `V`.
    The singular values `s` are non-negative and real numbers.

    This algorithms implements a (pseudo) single pass algorithm.


    Parameters
    ----------
    A : array_like, shape `(m, n)`.
        Real nonnegative input matrix.

    k : integer, `k << min{m,n}`.
        Target rank.

    p : integer, default: `p=10`.
        Parameter to control oversampling of column space.

    l : integer, default: `l=2*p`.
        Parameter to control oversampling of row space.

    sdist : str `{'uniform', 'normal', 'orthogonal'}`, default: `sdist='uniform'`.
        'uniform' : Random test matrices with uniform distributed elements.

        'normal' : Random test matrices with normal distributed elements.

        'orthogonal' : Orthogonalized random test matrices with uniform distributed elements.


    Returns
    -------
    U:  array_like, shape `(m, k)`.
        Right singular values.

    s : array_like, 1-d array of length `k`.
        Singular values.

    Vt : array_like, shape `(k, n)`.
        Left singular values.


    References
    ----------
    Tropp, Joel A., et al.
    "Randomized single-view algorithms for low-rank matrix approximation" (2016).
    (available at `arXiv <https://arxiv.org/abs/1609.00048>`_).
    """
    # converts A to array, raise ValueError if A has inf or nan
    A = np.asarray_chkfinite(A)
    m, n = A.shape

    flipped = False
    if m < n:
        A = conjugate_transpose(A)
        m , n = A.shape
        flipped = True

    # compute QB decomposition
    Q, B = rqb_single(A, k=k, p=p, l=l, sdist=sdist)

    # Singular Value Decomposition
    # NOTE: B = U" * S * Vt
    U, s, Vt = linalg.svd(B, compute_uv=True, full_matrices=False,
                          overwrite_a=True, check_finite=False)

    # Recover right singular vectors
    U = Q.dot(U)

    # Return Trunc
    if flipped:
        return conjugate_transpose(Vt)[:, :k], s[:k], conjugate_transpose(U)[:k, :]

    return U[:, :k], s[:k], Vt[:k , :]
