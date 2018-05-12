"""
Randomized Singular Value Decomposition
"""
# Authors: N. Benjamin Erichson
#          Joseph Knox
# License: GNU General Public License v3.0

import numpy as np
from scipy import linalg

from .rqb import rqb
from ..utils import conjugate_transpose


def rsvd(A, k=None, p=10, q=1, sdist='uniform'):
    """Randomized Singular Value Decomposition.

    Randomized algorithm for computing the approximate low-rank singular value
    decomposition of a rectangular `(m, n)` matrix `A`, with target rank `k << min{m, n}`.
    The input matrix is factored as `A = U * diag(s) * Vt`. The right singular
    vectors are the columns of the real or complex unitary matrix `U`. The left
    singular vectors are the columns of the real or complex unitary matrix `V`.
    The singular values `s` are non-negative and real numbers.

    The quality of the approximation can be controlled via the oversampling
    parameter `p` and the parameter `q` which specifies the number of
    subspace iterations.

    If k > (n/1.5), partial SVD or truncated SVD might be faster.


    Parameters
    ----------
    A : array_like, shape `(m, n)`.
        Real nonnegative input matrix.

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
    U:  array_like, shape `(m, k)`.
        Right singular values.

    s : array_like, 1-d array of length `k`.
        Singular values.

    Vt : array_like, shape `(k, n)`.
        Left singular values.


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

    Q, B = rqb(A, k=k, p=p, q=q, sdist=sdist)

    # Compute SVD
    U, s, Vt = linalg.svd(B, compute_uv=True, full_matrices=False,
                          overwrite_a=True, check_finite=False)

    # Recover right singular vectors
    U = Q.dot(U)

    # Return Trunc
    if flipped:
        return conjugate_transpose(Vt)[:, :k], s[:k], conjugate_transpose(U)[:k, :]

    return U[:, :k], s[:k], Vt[:k, :]
