"""
Randomized Singular Value Decomposition
"""
# Authors: N. Benjamin Erichson
#          Joseph Knox
# License: GNU General Public License v3.0

from __future__ import division, print_function

import numpy as np
from scipy import linalg
from scipy.sparse import linalg as splinalg

from ..dmd.utils import conjugate_transpose
from ..dmd.rdmd import _get_sdist_func

_VALID_DTYPES = (np.float32, np.float64, np.complex64, np.complex128)
_VALID_SDISTS = ('uniform', 'normal')


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
        k = min(m,n)

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

    #Project the data matrix a into a lower dimensional subspace
    B = A.dot(Q)
    B = conjugate_transpose(Q).dot(B)
    B = (B + conjugate_transpose(B)) / 2 # Symmetry

    # Eigendecompositoin
    w, v = linalg.eigh(B, eigvals_only=False, overwrite_a=True,
                       turbo=True, eigvals=None, type=1, check_finite=False)

    v[:, :n ] = v[: , n-1::-1]
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
        k = min(m,n)

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

    #Project the data matrix a into a lower dimensional subspace
    B1 = A.dot(Q)
    B2 = conjugate_transpose(Q).dot(B1)

    B2 = (B2 + conjugate_transpose(B2)) / 2 # Symmetry

    try:
        # Cholesky factorizatoin
        C = linalg.cholesky(B2, lower=True, overwrite_a=True, check_finite=False)
    except:
        print("Cholesky factorizatoin has failed, because array is not positive definite.")
        # Eigendecompositoin
        w, v = linalg.eigh(B2, eigvals_only=False, overwrite_a=True,
                           turbo=True, eigvals=None, type=1, check_finite=False)


        v[:, :n] = v[:, n-1::-1]
        w = w[::-1]

        return w[:k], Q.dot(v)[:,:k]

    # Upper triangular solve
    F = linalg.solve_triangular(a=C, b=conjugate_transpose(B1), lower=True,
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
        U, s, _ = linalg.svd(B2,  full_matrices=False, overwrite_a=True, check_finite=False)

        U = B1.dot(U  * s  **-1)
        U = U[:,0:k] * np.sqrt(k / n)
        s = s[0:k] * (n / k)

        return (s[0:k], U)

    # Upper triangular solve
    F = linalg.solve_triangular(a=C, b=conjugate_transpose(B1), lower=True,
                                unit_diagonal=False, overwrite_b=True,
                                debug=None, check_finite=False)

    #Compute SVD
    v, w, _ = linalg.svd(conjugate_transpose(F), compute_uv=True,
                         full_matrices=False, overwrite_a=True, check_finite=False)

    return w[:k]**2 , v[:,:k]
