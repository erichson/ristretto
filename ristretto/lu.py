"""
Randomized LU Decomposition
"""
# Authors: N. Benjamin Erichson
#          Joseph Knox
# License: GNU General Public License v3.0
from __future__ import division

import numpy as np
from scipy import linalg
from scipy import sparse

from .sketch.transforms import johnson_lindenstrauss
from .sketch.utils import perform_subspace_iterations
from .utils import conjugate_transpose


def compute_rlu(A, rank, oversample=10, n_subspace=2, permute=False, random_state=None):
    """Randomized LU Decomposition.

    Randomized algorithm for computing the approximate low-rank LU
    decomposition of a rectangular `(m, n)` matrix `A`, with target rank
    `rank << min{m, n}`. The input matrix is factored as `A = P * L * U * C`, where
    `L` and `U` are the lower and upper triangular matrices, respectively.
    And `P` and `C` are the row and column permutation matrices.

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

    permute : bool, default: `permute=False`.
        If `True`, perform the multiplication P*L and U*C.

    random_state : integer, RandomState instance or None, optional (default ``None``)
        If integer, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used by np.random.

    Returns
    -------
    P : array_like, shape `(m, m)`.
        Row permutation matrix, only returned if `permute == False`.

    L :  array_like, shape `(m, rank)`.
        Lower triangular matrix.

    U : array_like, shape `(rank, n)`.
        Upper triangular matrix.

    C : array_like, shape `(n, n)`.
        Column permutation matrix, only returned if `permute == False`.


    References
    ----------
    Halko, Nathan, Per-Gunnar Martinsson, and Joel A. Tropp.
    "Finding structure with randomness: Probabilistic algorithms for
    constructing approximate matrix decompositions."
    SIAM review 53.2 (2011): 217-288.
    (available at `arXiv <http://arxiv.org/abs/0909.4061>`_).

    Shabat, Gil, et al.
    "Randomized LU decomposition."
    Applied and Computational Harmonic Analysis (2016).
    (available at `arXiv <https://arxiv.org/abs/1310.7202>`_).
    """
    # get random sketch
    S = johnson_lindenstrauss(A, rank + oversample, random_state=random_state)

    if n_subspace > 0:
        S = perform_subspace_iterations(A, S, n_iter=n_subspace, axis=1)

    # Compute pivoted LU decompostion of the orthonormal basis matrix Q.
    # Q = P * L * U
    P, L_tilde, _ = linalg.lu(S, permute_l=False, overwrite_a=True, check_finite=False)
    _, r ,_ = sparse.find(P.T)

    # Truncate L_tilde
    L_tilde = L_tilde[:, :rank]

    # Form smaller matrix B
    U, s, Vt = linalg.svd(L_tilde, compute_uv=True, full_matrices=False,
                          overwrite_a=False, check_finite=False)

    B = (conjugate_transpose(Vt) / s).dot(conjugate_transpose(U)).dot(A[r,:])

    # Compute LU decompostion of B.
    C, L, U = linalg.lu(conjugate_transpose(B), permute_l=False,
                        overwrite_a=True, check_finite=False)

    #Return
    if permute:
        _, r ,_ = sparse.find(P)
        _, c ,_ = sparse.find(C)
        return L_tilde.dot(conjugate_transpose(U))[r,:], conjugate_transpose(L)[:,c]

    return P, L_tilde.dot(conjugate_transpose(U)), conjugate_transpose(L),\
        conjugate_transpose(C)
