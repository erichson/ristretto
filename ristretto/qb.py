"""
Randomized QB Decomposition
"""
# Authors: N. Benjamin Erichson
#          Joseph Knox
# License: GNU General Public License v3.0

import numpy as np
from scipy import linalg

from .sketch.transforms import johnson_lindenstrauss, sparse_johnson_lindenstrauss
from .utils import conjugate_transpose


def rqb(A, rank, oversample=10, n_subspace=1, sparse=False, random_state=None):
    """Randomized QB Decomposition.

    Randomized algorithm for computing the approximate low-rank QB
    decomposition of a rectangular `(m, n)` matrix `A`, with target rank
    `rank << min{m, n}`. The input matrix is factored as `A = Q * B`.

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

    n_subspace : integer, default: 1.
        Parameter to control number of subspace iterations. Increasing this
        parameter may improve numerical accuracy.

    sparse : boolean, optional (default: False)
        If sparse == True, perform compressed random qr decomposition.

    random_state : integer, RandomState instance or None, optional (default ``None``)
        If integer, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used by np.random.


    Returns
    -------
    Q:  array_like, shape `(m, rank + oversample)`.
        Orthonormal basis matrix.

    B : array_like, shape `(rank + oversample, n)`.
        Smaller matrix.


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
    if sparse:
        Q = johnson_lindenstrauss(A, rank + oversample, n_subspace=n_subspace,
                                  random_state=random_state)
    else:
        Q = sparse_johnson_lindenstrauss(
            A, rank + oversample, n_subspace=n_subspace, random_state=random_state)

    #Project the data matrix a into a lower dimensional subspace
    B = conjugate_transpose(Q).dot(A)

    return Q, B
