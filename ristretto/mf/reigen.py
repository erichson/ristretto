"""
Randomized Singular Value Decomposition
"""
# Authors: N. Benjamin Erichson
#          Joseph Knox
# License: GNU General Public License v3.0

from __future__ import division, print_function

import numpy as np
from scipy import linalg

from ..sketch import sketch
from ..utils import conjugate_transpose

_VALID_DTYPES = (np.float32, np.float64, np.complex64, np.complex128)


def reigh(A, k, p=20, q=2, sdist='normal'):
    """Randomized eigendecompostion.


    Parameters
    ----------
    A : array_like, shape `(n, n)`.
        Hermitian matrix.

    k : integer, `k << n`.
        Target rank.

    p : integer, default: `p=10`.
        Parameter to control oversampling.

    q : integer, default: `q=2`.
        Parameter to control number of power (subspace) iterations.

    sdist : str `{'uniform', 'normal'}`, default: `sdist='uniform'`.
        'uniform' : Random test matrix with uniform distributed elements.

        'normal' : Random test matrix with normal distributed elements.


    Returns
    -------
    w : array_like, 1-d array of length `k`.
        The eigenvalues.

    v: array_like, shape `(n, k)`.
        The normalized selected eigenvector corresponding to the
        eigenvalue w[i] is the column v[:,i].


    References
    ----------
    N. Halko, P. Martinsson, and J. Tropp.
    "Finding structure with randomness: probabilistic
    algorithms for constructing approximate matrix
    decompositions" (2009).
    (available at `arXiv <http://arxiv.org/abs/0909.4061>`_).
    """
    # get random sketch
    Q = sketch(A, output_rank=k, n_oversample=p, n_iter=q, distribution=sdist,
               axis=1, check_finite=True)

    #Project the data matrix a into a lower dimensional subspace
    B = A.dot(Q)
    B = conjugate_transpose(Q).dot(B)
    B = (B + conjugate_transpose(B)) / 2 # Symmetry

    # Eigendecomposition
    w, v = linalg.eigh(B, eigvals_only=False, overwrite_a=True,
                       turbo=True, eigvals=None, type=1, check_finite=False)

    v[:, :A.shape[1]] = v[:, A.shape[1] - 1::-1]
    w = w[::-1]

    return w[:k], Q.dot(v)[:,:k]


def reigh_nystroem(A, k, p=10, q=2, sdist='normal'):
    """Randomized eigendecompostion using the Nystroem method.


    Parameters
    ----------
    A : array_like, shape `(n, n)`.
        Positive-definite matrix (PSD) input matrix.

    k : integer, `k << n`.
        Target rank.

    p : integer, default: `p=10`.
        Parameter to control oversampling.

    q : integer, default: `q=2`.
        Parameter to control number of power (subspace) iterations.

    sdist : str `{'uniform', 'normal'}`, default: `sdist='uniform'`.
        'uniform' : Random test matrix with uniform distributed elements.

        'normal' : Random test matrix with normal distributed elements.

    Returns
    -------
    w : array_like, 1-d array of length `k`.
        The eigenvalues.

    v: array_like, shape `(n, k)`.
        The normalized selected eigenvector corresponding
        to the eigenvalue w[i] is the column v[:,i].


    References
    ----------
    N. Halko, P. Martinsson, and J. Tropp.
    "Finding structure with randomness: probabilistic
    algorithms for constructing approximate matrix
    decompositions" (2009).
    (available at `arXiv <http://arxiv.org/abs/0909.4061>`_).
    """
    # get random sketch
    S = sketch(A, output_rank=k, n_oversample=p, n_iter=q, distribution=sdist,
               axis=1, check_finite=True)

    #Project the data matrix a into a lower dimensional subspace
    B1 = A.dot(S)
    B2 = conjugate_transpose(S).dot(B1)
    B2 = (B2 + conjugate_transpose(B2)) / 2 # Symmetry

    try:
        # Cholesky factorizatoin
        C = linalg.cholesky(B2, lower=True, overwrite_a=True, check_finite=False)
    except:
        print("Cholesky factorizatoin has failed, because array is not positive definite.")
        # Eigendecompositoin
        w, v = linalg.eigh(B2, eigvals_only=False, overwrite_a=True,
                           turbo=True, eigvals=None, type=1, check_finite=False)


        v[:, :A.shape[1]] = v[:, A.shape[1]-1::-1]
        w = w[::-1]

        return w[:k], S.dot(v)[:,:k]

    # Upper triangular solve
    F = linalg.solve_triangular(C, conjugate_transpose(B1), lower=True,
                                unit_diagonal=False, overwrite_b=True,
                                debug=None, check_finite=False)

    #Compute SVD
    v, w, _ = linalg.svd(conjugate_transpose(F), compute_uv=True, full_matrices=False,
                         overwrite_a=True, check_finite=False)

    return w[:k]**2, v[:,:k]


def reigh_nystroem_col(A, k, p=0):
    """Randomized eigendecompostion using the Nystroem method.


    Parameters
    ----------
    A : array_like, shape `(n, n)`.
        Positive-definite matrix (PSD) input matrix.

    k : integer, `k << n`.
        Target rank.

    p : integer, default: `p=0`.
        Parameter to control oversampling.


    Returns
    -------
    w : array_like, 1-d array of length `k`.
        The eigenvalues.

    v: array_like, shape `(n, k)`.
        The normalized selected eigenvector corresponding
        to the eigenvalue w[i] is the column v[:,i].


    References
    ----------
    N. Halko, P. Martinsson, and J. Tropp.
    "Finding structure with randomness: probabilistic
    algorithms for constructing approximate matrix
    decompositions" (2009).
    (available at `arXiv <http://arxiv.org/abs/0909.4061>`_).
    """
    # converts A to array, raise ValueError if A has inf or nan
    A = np.asarray_chkfinite(A)
    m, n = A.shape

    if A.dtype not in _VALID_DTYPES:
        raise ValueError('A.dtype must be one of %s, not %s'
                         % (' '.join(_VALID_DTYPES), A.dtype))

    if k is None:
        # default
        k = min(m,n)

    if k < 1 or k > min(m, n):
        raise ValueError("Target rank k must be >= 1 or < min(m, n), not %d" % k)

    #Generate a random test matrix Omega
    idx = np.sort(np.random.choice(n, size=(k+p), replace=False))

    #Project the data matrix a into a lower dimensional subspace
    B1 = A[:,idx]
    B2 = B1[idx,:].copy()

    B2 = (B2 + conjugate_transpose(B2)) / 2 # Symmetry

    try:
        # Cholesky factorizatoin
        C = linalg.cholesky(B2, lower=True, overwrite_a=True, check_finite=False)
    except:
        print("Cholesky factorizatoin has failed, because array is not positive definite.")
        # Eigendecompositoin
        U, s, _ = linalg.svd(B2, full_matrices=False, overwrite_a=True, check_finite=False)

        U = B1.dot(U / s)
        U = U[:, :k] * np.sqrt(k / n)
        s = s[:k] * (n / k)

        return s[:k], U

    # Upper triangular solve
    F = linalg.solve_triangular(C, conjugate_transpose(B1), lower=True,
                                unit_diagonal=False, overwrite_b=True,
                                debug=None, check_finite=False)

    #Compute SVD
    v, w, _ = linalg.svd(conjugate_transpose(F), compute_uv=True,
                         full_matrices=False, overwrite_a=True, check_finite=False)

    return w[:k]**2, v[:, :k]
