"""
Randomized Singular Value Decomposition
"""
# Authors: N. Benjamin Erichson
#          Joseph Knox
# License: GNU General Public License v3.0

# TODO: repace nystroem_col with random uniform sampling
# TODO: conform functions to return like scipy.linalg.eig and rename
from __future__ import division
import warnings

import numpy as np
from scipy import linalg
from sklearn.utils import check_random_state

from .sketch.transforms import johnson_lindenstrauss, randomized_uniform_sampling
from .sketch.utils import perform_subspace_iterations
from .utils import conjugate_transpose

_VALID_DTYPES = (np.float32, np.float64, np.complex64, np.complex128)


def compute_reigh(A, rank, oversample=10, n_subspace=2, random_state=None):
    """Randomized eigendecompostion.

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

    random_state : integer, RandomState instance or None, optional (default ``None``)
        If integer, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used by np.random.


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
    Q = johnson_lindenstrauss(A, rank + oversample, axis=1, random_state=random_state)

    if n_subspace > 0:
        Q = perform_subspace_iterations(A, Q, n_iter=n_subspace, axis=1)

    #Project the data matrix a into a lower dimensional subspace
    B = A.dot(Q)
    B = conjugate_transpose(Q).dot(B)
    B = (B + conjugate_transpose(B)) / 2 # Symmetry

    # Eigendecomposition
    w, v = linalg.eigh(B, eigvals_only=False, overwrite_a=True,
                       turbo=True, eigvals=None, type=1, check_finite=False)

    v[:, :A.shape[1]] = v[:, A.shape[1] - 1::-1]
    w = w[::-1]

    return w[:rank], Q.dot(v)[:,:rank]


def compute_reigh_nystroem(A, rank, oversample=10, n_subspace=2, random_state=None):
    """Randomized eigendecompostion using the Nystroem method.

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

    random_state : integer, RandomState instance or None, optional (default ``None``)
        If integer, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used by np.random.


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
    S = johnson_lindenstrauss(A, rank + oversample, axis=1, random_state=random_state)

    if n_subspace > 0:
        S = perform_subspace_iterations(A, S, n_iter=n_subspace, axis=1)

    #Project the data matrix a into a lower dimensional subspace
    B1 = A.dot(S)
    B2 = conjugate_transpose(S).dot(B1)
    B2 = (B2 + conjugate_transpose(B2)) / 2 # Symmetry

    try:
        # Cholesky factorizatoin
        C = linalg.cholesky(B2, lower=True, overwrite_a=True, check_finite=False)
    except linalg.LinAlgError:
        warnings.warn("Cholesky factorizatoin has failed, because array is not "
                      "positive definite. Using SVD instead.")
        # Eigendecompositoin
        w, v = linalg.eigh(B2, eigvals_only=False, overwrite_a=True,
                           turbo=True, eigvals=None, type=1, check_finite=False)


        v[:, :A.shape[1]] = v[:, A.shape[1]-1::-1]
        w = w[::-1]

        return w[:rank], S.dot(v)[:,:rank]

    # Upper triangular solve
    F = linalg.solve_triangular(C, conjugate_transpose(B1), lower=True,
                                unit_diagonal=False, overwrite_b=True,
                                debug=None, check_finite=False)

    #Compute SVD
    v, w, _ = linalg.svd(conjugate_transpose(F), compute_uv=True, full_matrices=False,
                         overwrite_a=True, check_finite=False)

    return w[:rank]**2, v[:,:rank]


def compute_reigh_nystroem_col(A, rank, oversample=0, random_state=None):
    """Randomized eigendecompostion using the Nystroem method.

    The quality of the approximation can be controlled via the oversampling
    parameter `oversample`.


    Parameters
    ----------
    A : array_like, shape `(m, n)`.
        Input array.

    rank : integer
        Target rank. Best if `rank << min{m,n}`

    oversample : integer, optional (default: 10)
        Controls the oversampling of column space. Increasing this parameter
        may improve numerical accuracy.

    random_state : integer, RandomState instance or None, optional (default ``None``)
        If integer, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used by np.random.


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

    # TODO: repace with random uniform sampling

    random_state = check_random_state(random_state)

    # converts A to array, raise ValueError if A has inf or nan
    A = np.asarray_chkfinite(A)
    m, n = A.shape

    if A.dtype not in _VALID_DTYPES:
        raise ValueError('A.dtype must be one of %s, not %s'
                         % (' '.join(_VALID_DTYPES), A.dtype))

    if rank < 1 or rank > min(m, n):
        raise ValueError("Target rank must be >= 1 or < min(m, n), not %d" % rank)

    #Generate a random test matrix Omega
    idx = np.sort(random_state.choice(n, size=(rank+oversample), replace=False))

    #Project the data matrix a into a lower dimensional subspace
    B1 = A[:,idx]
    B2 = B1[idx,:].copy()

    B2 = (B2 + conjugate_transpose(B2)) / 2 # Symmetry

    try:
        # Cholesky factorizatoin
        C = linalg.cholesky(B2, lower=True, overwrite_a=True, check_finite=False)
    except linalg.LinAlgError:
        warnings.warn("Cholesky factorizatoin has failed, because array is not "
                      "positive definite. Using SVD instead.")
        # Eigendecompositoin
        U, s, _ = linalg.svd(B2, full_matrices=False, overwrite_a=True, check_finite=False)

        U = B1.dot(U / s)
        U = U[:, :rank] * np.sqrt(rank / n)
        s = s[:rank] * (n / rank)

        return s[:rank], U

    # Upper triangular solve
    F = linalg.solve_triangular(C, conjugate_transpose(B1), lower=True,
                                unit_diagonal=False, overwrite_b=True,
                                debug=None, check_finite=False)

    #Compute SVD
    v, w, _ = linalg.svd(conjugate_transpose(F), compute_uv=True,
                         full_matrices=False, overwrite_a=True, check_finite=False)

    return w[:rank]**2, v[:, :rank]
