"""
Randomized QB Decomposition
"""
# Authors: N. Benjamin Erichson
#          Joseph Knox
# License: GNU General Public License v3.0

import numpy as np
from scipy import linalg

from .sketch.transforms import johnson_lindenstrauss, sparse_johnson_lindenstrauss
from .sketch.utils import perform_subspace_iterations, orthonormalize
from .utils import conjugate_transpose


def _compute_rqb(A, rank, oversample, n_subspace, sparse, random_state):
    if sparse:
        Q = sparse_johnson_lindenstrauss(A, rank + oversample,
                                         random_state=random_state)
    else:
        Q = johnson_lindenstrauss(A, rank + oversample, random_state=random_state)

    if n_subspace > 0:
        Q = perform_subspace_iterations(A, Q, n_iter=n_subspace, axis=1)
    else:
        Q = orthonormalize(Q)

    # Project the data matrix a into a lower dimensional subspace
    B = conjugate_transpose(Q).dot(A)

    return Q, B


def compute_rqb(A, rank, oversample=20, n_subspace=2, n_blocks=1, sparse=False,
                random_state=None):
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

    n_subspace : integer, default: 2.
        Parameter to control number of subspace iterations. Increasing this
        parameter may improve numerical accuracy. Every additional subspace
        iterations requires an additional full pass over the data matrix.

    n_blocks : integer, default: 1.
        If `n_blocks > 1` a column blocked QB decomposition procedure will be
        performed. A larger number requires less fast memory, while it
        leads to a higher computational time.

    sparse : boolean, optional (default: False)
        If sparse == True, perform compressed random qr decomposition.

    random_state : integer, RandomState instance or None, optional (default `None`)
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
    if n_blocks > 1:
        m, n = A.shape

        # index sets
        row_sets = np.array_split(range(m), n_blocks)

        Q_block = []
        K = []

        nblock = 1
        for rows in row_sets:
            # converts A to array, raise ValueError if A has inf or nan
            Qtemp, Ktemp = _compute_rqb(np.asarray_chkfinite(A[rows, :]), 
                rank=rank, oversample=oversample, n_subspace=n_subspace, 
                sparse=sparse, random_state=random_state)

            Q_block.append(Qtemp)
            K.append(Ktemp)
            nblock += 1

        Q_small, B = _compute_rqb(
            np.concatenate(K, axis=0), rank=rank, oversample=oversample,
            n_subspace=n_subspace, sparse=sparse, random_state=random_state)

        Q_small = np.vsplit(Q_small, n_blocks)

        Q = [Q_block[i].dot(Q_small[i]) for i in range(n_blocks)]
        Q = np.concatenate(Q, axis=0)

    else:
        Q, B = _compute_rqb(np.asarray_chkfinite(A), 
            rank=rank, oversample=oversample, n_subspace=n_subspace,
            sparse=sparse, random_state=random_state)

    return Q, B
