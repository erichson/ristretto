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

from ..sketch import sketch
from ..utils import conjugate_transpose


def rlu(A, permute=False, k=None, p=10, q=1, sdist='uniform'):
    """Randomized LU Decomposition.

    Randomized algorithm for computing the approximate low-rank LU
    decomposition of a rectangular `(m, n)` matrix `A`, with target rank `k << min{m, n}`.
    The input matrix is factored as `A = P * L * U * C`, where
    `L` and `U` are the lower and upper triangular matrices, respectively.
    And `P` and `C` are the row and column permutation matrices.

    The quality of the approximation can be controlled via the oversampling
    parameter `p` and the parameter `q` which specifies the number of
    subspace iterations.


    Parameters
    ----------
    A : array_like, shape `(m, n)`.
        Real nonnegative input matrix.

    permute : bool, default: `permute=False`.
        If `True`, perform the multiplication P*L and U*C.

    k : integer, `k << min{m,n}`.
        Target rank.

    p : integer, default: `p=10`.
        Parameter to control oversampling.

    q : integer, default: `q=1`.
        Parameter to control number of power (subspace) iterations.

    sdist : str `{'uniform', 'normal'}`, default: `sdist='uniform'`.
        'uniform' : Random test matrix with uniform distributed elements.

        'normal' : Random test matrix with normal distributed elements.


    Returns
    -------
    P : array_like, shape `(m, m)`.
        Row permutation matrix, if `permute_l=False`.

    L :  array_like, shape `(m, k)`.
        Lower triangular matrix.

    U : array_like, shape `(k, n)`.
        Upper triangular matrix.

    C : array_like, shape `(n, n)`.
        Column Permutation matrix, if `permute=False`.


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
    S = sketch(A, output_rank=k, n_oversample=p, n_iter=q, distribution=sdist,
               axis=1, check_finite=True)

    # Compute pivoted LU decompostion of the orthonormal basis matrix Q.
    # Q = P * L * U
    P, L_tilde, _ = linalg.lu(S, permute_l=False, overwrite_a=True, check_finite=False)
    _, r ,_ = sparse.find(P.T)

    # Truncate L_tilde
    L_tilde = L_tilde[:, :k]

    # Form smaller matrix B
    U, s, Vt = linalg.svd(L_tilde, compute_uv=True, full_matrices=False,
                          overwrite_a=False, check_finite=False)

    B = (conjugate_transpose(Vt) / s).dot(conjugate_transpose(U)).dot(A[r,:])

    # Compute LU decompostion of B.
    C, L, U = linalg.lu(conjugate_transpose(B), permute_l=False,
                        overwrite_a=True, check_finite=False)

    #Return
    if permute:
        _, r, _ = sparse.find(P)
        _, c, _ = sparse.find(C)
        return L_tilde.dot(conjugate_transpose(U))[r,:], conjugate_transpose(L)[:,c]

    return P, L_tilde.dot(conjugate_transpose(U)), conjugate_transpose(L), conjugate_transpose(C)
