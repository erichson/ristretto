"""
Interpolative decomposition (ID)
"""
# Authors: N. Benjamin Erichson
#          Joseph Knox
# License: GNU General Public License v3.0
from __future__ import division

import numpy as np
from scipy import linalg

from .qb import compute_rqb
from .utils import conjugate_transpose

_VALID_MODES = ('row', 'column')


def compute_interp_decomp(A, rank, mode='column', index_set=False):
    """Interpolative decomposition (ID).

    Algorithm for computing the low-rank ID
    decomposition of a rectangular `(m, n)` matrix `A`, with target rank `k << min{m, n}`.
    Input matrix is factored as `A = C * V`, using the column pivoted QR decomposition.
    The factor matrix `C` is formed of a subset of columns of `A`,
    also called the partial column skeleton. The factor matrix `V` contains
    a `(rank, rank)` identity matrix as a submatrix, and is well-conditioned.

    If `mode='row'`, then the input matrix is factored as `A = Z * R`, using the
    row pivoted QR decomposition. The factor matrix `R` is now formed as
    a subset of rows of `A`, also called the partial row skeleton.

    Parameters
    ----------
    A : array_like, shape `(m, n)`.
        Input array.

    rank : integer
        Target rank. Best if `rank << min{m,n}`

    mode: str `{'column', 'row'}`, default: `mode='column'`.
        'column' : ID using column pivoted QR.
        'row' : ID using row pivoted QR.

    index_set: str `{'True', 'False'}`, default: `index_set='False'`.
        'True' : Return column/row index set instead of `C` or `R`.


    Returns
    -------
    If `mode='column'`:
        C:  array_like, shape `(m, rank)`.
            Partial column skeleton.

        V : array_like, shape `(rank, n)`.
            Well-conditioned matrix.

    If `mode='row'`:
        Z:  array_like, shape `(m, rank)`.
            Well-conditioned matrix.

        R : array_like, shape `(rank, n)`.
            Partial row skeleton.

    References
    ----------
    S. Voronin and P.Martinsson.
    "RSVDPACK: Subroutines for computing partial singular value
    decompositions via randomized sampling on single core, multi core,
    and GPU architectures" (2015).
    (available at `arXiv <http://arxiv.org/abs/1502.05366>`_).
    """
    if mode not in _VALID_MODES:
        raise ValueError('mode must be one of %s, not %s'
                         % (' '.join(_VALID_MODES), mode))

    # converts A to array, raise ValueError if A has inf or nan
    A = np.asarray_chkfinite(A)
    if mode=='row':
        A = conjugate_transpose(A)

    m, n = A.shape
    if rank < 1 or rank > min(m, n):
        raise ValueError("Target rank rank must be >= 1 or < min(m, n), not %d" % rank)

    #Pivoted QR decomposition
    Q, R, P = linalg.qr(A, mode='economic', overwrite_a=False, pivoting=True,
                        check_finite=False)

    # Select column subset
    C = A[:, P[:rank]]

    # Compute V
    T =  linalg.pinv2(R[:rank, :rank]).dot(R[:rank, rank:n])
    V = np.bmat([[np.eye(rank), T]])
    V = V[:, np.argsort(P)]

    # Return ID
    if mode == 'column':
        if index_set:
            return P[:rank], V
        return C, V
    # mode == row
    elif index_set:
        return conjugate_transpose(V), P[:rank]

    return conjugate_transpose(V), conjugate_transpose(C)


def compute_rinterp_decomp(A, rank, oversample=10, n_subspace=2, mode='column',
                   index_set=False, random_state=None):
    """Randomized interpolative decomposition (rID).

    Algorithm for computing the approximate low-rank ID
    decomposition of a rectangular `(m, n)` matrix `A`, with target rank `k << min{m, n}`.
    The input matrix is factored as `A = C * V`. The factor matrix `C`is formed
    of a subset of columns of `A`, also called the partial column skeleton.
    The factor matrix `V`contains a `(rank, rank)` identity matrix as a submatrix,
    and is well-conditioned.

    If `mode='row'`, then the input matrix is factored as `A = Z * R`, using the
    row pivoted QR decomposition. The factor matrix `R` is now formed as
    a subset of rows of `A`, also called the partial row skeleton.

    The quality of the approximation can be controlled via the oversampling
    parameter `oversample` and `n_subspace` which specifies the number of
    subspace iterations.


    Parameters
    ----------
    A : array_like, shape `(m, n)`.
        Input array.

    rank : integer
        Target rank. Best if `rank << min{m,n}`

    oversample : integer, optional (default: 10)
        Controls the oversampling of column space. Increasing this parameter
        may improve numerical accuracy.

    n_subspace : integer, default: 2.
        Parameter to control number of subspace iterations. Increasing this
        parameter may improve numerical accuracy.

    mode: str `{'column', 'row'}`, default: `mode='column'`.
        'column' : Column ID.
        'row' : Row ID.

    index_set: str `{'True', 'False'}`, default: `index_set='False'`.
        'True' : Return column/row index set instead of `C` or `R`.

    random_state : integer, RandomState instance or None, optional (default ``None``)
        If integer, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used by np.random.


    Returns
    -------
    If `mode='column'`:
        C:  array_like, shape `(m, rank)`.
            Partial column skeleton.

        V : array_like, shape `(rank, n)`.
            Well-conditioned matrix.

    If `mode='row'`:
        Z:  array_like, shape `(m, rank)`.
            Well-conditioned matrix.

        R : array_like, shape `(rank, n)`.
            Partial row skeleton.

    References
    ----------
    S. Voronin and P.Martinsson.
    "RSVDPACK: Subroutines for computing partial singular value
    decompositions via randomized sampling on single core, multi core,
    and GPU architectures" (2015).
    (available at `arXiv <http://arxiv.org/abs/1502.05366>`_).
    """
    if mode not in _VALID_MODES:
        raise ValueError('mode must be one of %s, not %s'
                         % (' '.join(_VALID_MODES), mode))

    # converts A to array, raise ValueError if A has inf or nan
    A = np.asarray_chkfinite(A)
    if mode == 'row':
        A = conjugate_transpose(A)

    # compute QB factorization
    Q, B = compute_rqb(A, rank, oversample=oversample, n_subspace=n_subspace,
                       random_state=random_state)

    # Deterministic ID
    J, V = compute_interp_decomp(B, rank, mode='column', index_set=True)
    J = J[:rank]

    # Return ID
    if mode == 'column':
        if index_set:
            return J, V
        return A[:, J], V
    # mode == 'row'
    elif index_set:
        return conjugate_transpose(V), J
    return conjugate_transpose(V), conjugate_transpose(A[:, J])
