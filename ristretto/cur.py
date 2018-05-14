"""
CUR-ID
"""
# Authors: N. Benjamin Erichson
#          Joseph Knox
# License: GNU General Public License v3.0

from __future__ import division

import numpy as np
from scipy import linalg

from .interp_decomp import interp_decomp, rinterp_decomp


def cur(A, k=None, index_set=False):
    """CUR decomposition.

    Algorithm for computing the low-rank CUR
    decomposition of a rectangular `(m, n)` matrix `A`, with target rank `k << min{m, n}`.
    Input matrix is factored as `A = C * U * R`, using the column/row pivoted QR decomposition.
    The factor matrix `C` is formed of a subset of columns of `A`,
    also called the partial column skeleton. The factor matrix `R` is formed as
    a subset of rows of `A` also called the partial row skeleton.
    The factor matrix `U` is formed so that `U = C**-1 * A * R**-1` is satisfied.


    Parameters
    ----------
    A : array_like, shape `(m, n)`.
        Input matrix.

    k : integer, `k << min{m,n}`.
        Target rank.

    index_set: str `{'True', 'False'}`, default: `index_set='False'`.
        'True' : Return column/row index set instead of `C` and `R`.

    Returns
    -------
    C:  array_like, shape `(m, k)`.
            Partial column skeleton.

    U : array_like, shape `(k, k)`.
            Well-conditioned matrix.

    R : array_like, shape `(k, n)`.
            Partial row skeleton.


    References
    ----------
    S. Voronin and P.Martinsson.
    "RSVDPACK: Subroutines for computing partial singular value
    decompositions via randomized sampling on single core, multi core,
    and GPU architectures" (2015).
    (available at `arXiv <http://arxiv.org/abs/1502.05366>`_).
    """
    # compute column ID
    J, V = interp_decomp(A, k=k, mode='column', index_set=True)

    # select column subset
    C = A[:, J]

    # compute row ID of C
    Z, I = interp_decomp(C, k=k, mode='row', index_set=True)

    # select row subset
    R = A[I, :]

    # compute U
    U = V.dot(linalg.pinv2(R))

    # return ID
    if index_set:
        return J, U, I
    return C, U, R


def rcur(A, k=None, p=10, q=1, index_set=False, random_state=None):
    """Randomized CUR decomposition.

    Randomized algorithm for computing the approximate low-rank CUR
    decomposition of a rectangular `(m, n)` matrix `A`, with target rank `k << min{m, n}`.
    Input matrix is factored as `A = C * U * R`, using the column/row pivoted QR decomposition.
    The factor matrix `C` is formed of a subset of columns of `A`,
    also called the partial column skeleton. The factor matrix `R` is formed as
    a subset of rows of `A` also called the partial row skeleton.
    The factor matrix `U` is formed so that `U = C**-1 * A * R**-1` is satisfied.


    Parameters
    ----------
    A : array_like, shape `(m, n)`.
        Input matrix.

    k : integer, `k << min{m,n}`.
        Target rank.

    p : integer, default: `p=10`.
        Parameter to control oversampling.

    q : integer, default: `q=1`.
        Parameter to control number of power (subspace) iterations.

    index_set: str `{'True', 'False'}`, default: `index_set='False'`.
        'True' : Return column/row index set instead of `C` and `R`.

    random_state : integer, RandomState instance or None, optional (default ``None``)
        If integer, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used by np.random.


    Returns
    -------
    C:  array_like, shape `(m, k)`.
            Partial column skeleton.

    U : array_like, shape `(k, k)`.
            Well-conditioned matrix.

    R : array_like, shape `(k, n)`.
            Partial row skeleton.


    References
    ----------
    S. Voronin and P.Martinsson.
    "RSVDPACK: Subroutines for computing partial singular value
    decompositions via randomized sampling on single core, multi core,
    and GPU architectures" (2015).
    (available at `arXiv <http://arxiv.org/abs/1502.05366>`_).
    """
    # Compute column ID
    J, V = rinterp_decomp(A, k=k, p=p, q=q, mode='column', index_set=True,
                          random_state=random_state)

    # Select column subset
    C = A[:, J]

    # Compute row ID of C
    Z, I = rinterp_decomp(C, k=k, p=p, q=q,  mode='row', index_set=True,
                          random_state=random_state)

    # Select row subset
    R = A[I, :]

    # Compute U
    U = V.dot(linalg.pinv2(R))

    # Return ID
    if index_set:
        return J, U, I
    return C, U, R
