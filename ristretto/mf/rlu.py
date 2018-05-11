"""
Randomized LU Decomposition
"""
# Authors: N. Benjamin Erichson
#          Joseph Knox
# License: GNU General Public License v3.0
from __future__ import division, print_function

import numpy as np
from scipy import linalg
from scipy import sparse

from ..dmd.utils import conjugate_transpose
from ..dmd.rdmd import _get_sdist_func

_VALID_DTYPES = (np.float32, np.float64, np.complex64, np.complex128)
_VALID_SDISTS = ('uniform', 'normal')


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
    # converts A to array, raise ValueError if A has inf or nan
    A = np.asarray_chkfinite(A)
    m, n = A.shape

    if A.dtype not in _VALID_DTYPES:
        raise ValueError('A.dtype must be one of %s, not %s'
                         % (' '.join(_VALID_DTYPES), A.dtype))

    if sdist not in _VALID_SDISTS:
        raise ValueError('sdists must be one of %s, not %s'
                         % (' '.join(_VALID_SDISTS), sdist))

    if k is None:
        # default
        k = min(m, n)

    if k < 1 or k > min(m, n):
        raise ValueError("Target rank k must be >= 1 or < min(m, n), not %d" % k)

    # distribution to draw random samples
    sdist_func = _get_sdist_func(sdist)

    #Generate a random test matrix Omega
    Omega = sdist_func(size=(n, k+p)).astype(A.dtype)

    if A.dtype == np.complexfloating:
        real_type = np.float32 if A.dtype == np.complex64 else np.float64
        Omega += 1j * sdist_func(size=(n, k+p)).astype(real_type)

    #Build sample matrix Y : Y = A * Omega (Y approximates range of A)
    Y = A.dot(Omega)
    del Omega

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Orthogonalize Y using economic QR decomposition: Y=QR
    #If q > 0 perfrom q subspace iterations
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    for _ in range(q):
        Y, _ = linalg.qr(Y, mode='economic', check_finite=False, overwrite_a=True)
        Z, _ = linalg.qr(conjugate_transpose(A).dot(Y), mode='economic',
                         check_finite=False, overwrite_a=True)
        Y = A.dot(Z)

    Q, _ = linalg.qr(Y, mode='economic', check_finite=False, overwrite_a=True)
    del Y, Z

    # Compute pivoted LU decompostion of the orthonormal basis matrix Q.
    # Q = P * L * U
    P, L_tilde, _ = linalg.lu(Q, permute_l=False, overwrite_a=True, check_finite=False)
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
        _, r ,_ = sparse.find(P)
        _, c ,_ = sparse.find(C)
        return L_tilde.dot(conjugate_transpose(U))[r,:], conjugate_transpose(L)[:,c]

    return P, L_tilde.dot(conjugate_transpose(U)), conjugate_transpose(L), conjugate_transpose(C)
