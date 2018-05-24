"""
Random Singular Value Decomposition.
"""

# Authors: N. Benjamin Erichson
#          Joseph Knox
# License: GNU General Public License v3.0

from __future__ import division

import numpy as np
from scipy import linalg
from scipy import sparse

from .qb import rqb
from .utils import conjugate_transpose


def rsvd(A, rank, oversample=10, n_subspace=2, sparse=False, random_state=None):
    """Randomized Singular Value Decomposition.

    Randomized algorithm for computing the approximate low-rank singular value
    decomposition of a rectangular `(m, n)` matrix `A`, with target rank `k << min{m, n}`.
    The input matrix is factored as `A = U * diag(s) * Vt`. The right singular
    vectors are the columns of the real or complex unitary matrix `U`. The left
    singular vectors are the columns of the real or complex unitary matrix `V`.
    The singular values `s` are non-negative and real numbers.

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

    sparse : boolean, optional (default: False)
        If sparse == True, perform compressed rsvd.

    random_state : integer, RandomState instance or None, optional (default ``None``)
        If integer, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used by np.random.


    Returns
    -------
    U:  array_like
        Right singular values, array of shape `(m, rank)`.

    s : array_like
        Singular values, 1-d array of length `rank`.

    Vh : array_like
        Left singular values, array of shape `(rank, n)`.


    Notes
    -----
    If rank > (n/1.5), partial SVD or truncated SVD might be faster.


    References
    ----------
    N. Halko, P. Martinsson, and J. Tropp.
    "Finding structure with randomness: probabilistic
    algorithms for constructing approximate matrix
    decompositions" (2009).
    (available at `arXiv <http://arxiv.org/abs/0909.4061>`_).

    S. Voronin and P.Martinsson.
    "RSVDPACK: Subroutines for computing partial singular value
    decompositions via randomized sampling on single core, multi core,
    and GPU architectures" (2015).
    (available at `arXiv <http://arxiv.org/abs/1502.05366>`_).
    """
    # converts A to array, raise ValueError if A has inf or nan
    A = np.asarray_chkfinite(A)
    m, n = A.shape

    flipped = False
    if m < n:
        A = conjugate_transpose(A)
        m , n = A.shape
        flipped = True

    # Compute QB decomposition
    Q, B = rqb(A, rank, oversample=oversample, n_subspace=n_subspace,
               sparse=sparse, random_state=random_state)

    # Compute SVD
    U, s, Vt = linalg.svd(B, compute_uv=True, full_matrices=False,
                          overwrite_a=True, check_finite=False)

    # Recover right singular vectors
    U = Q.dot(U)

    # Return Trunc
    if flipped:
        return conjugate_transpose(Vt)[:, :rank], s[:rank], conjugate_transpose(U)[:rank, :]

    return U[:, :rank], s[:rank], Vt[:rank, :]
