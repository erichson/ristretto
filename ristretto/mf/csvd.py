"""
Compressed Singular Value Decomposition.
"""
# Authors: N. Benjamin Erichson
#          Joseph Knox
# License: GNU General Public License v3.0

from __future__ import division

import numpy as np
from scipy import linalg
from scipy import sparse

from ..utils import conjugate_transpose

_VALID_DTYPES = (np.float32, np.float64, np.complex64, np.complex128)


def csvd(A, k=None, p=10, sdist='sparse', formatS='csr'):
    """Compressed Singular Value Decomposition.

    Row compressed algorithm for computing the approximate low-rank singular value
    decomposition of a rectangular (m, n) matrix `A` with target rank `k << n`.
    The input matrix a is factored as `A = U * diag(s) * Vt`. The left singular
    vectors are the columns of the real or complex unitary matrix `U`. The right
    singular vectors are the columns of the real or complex unitary matrix `V`.
    The singular values `s` are non-negative and real numbers.

    The parameter `c` specifies the number of measurements and is required to
    be `c>k`.


    Parameters
    ----------
    A : array_like
        Real/complex input matrix  `A` with dimensions `(m, n)`.

    k : int
        `k` is the target rank of the low-rank decomposition, k << min(m,n).

    c : int
        `c` sets the number of measurments.

    sdist : str `{gaussian', 'spixel', 'sparse'}`
        Defines the sampling distribution.

    ortho : str `{True, False}`
        If `True` the left singular values are orthonormalized.

    method :   `{SVD, QR}`
        Defines the method to compute the orthnormalization step.

    scaled : str `{True, False}`
        If `True` the singular values are rescaled.

    fortmatS : str `{csr, coo}`
        Defines the format of the sparse measurement matrix.


    Returns
    -------
    U:  array_like
        Right singular values, array of shape `(m, k)`.

    s : array_like
        Singular values, 1-d array of length `k`.

    Vh : array_like
        Left singular values, array of shape `(k, n)`.


    Notes
    -----
    If the option `ortho=True` is selected, then the approximation is more
    accurate.

    If the sparse sampling distribution is used, the appropriate format for
    the sparse measurement matrix is crucial. In generall `csr` is the optimal
    format, but sometimes `coo` gives a better performance. Sparse matricies
    are computational efficient if the leading dimension is m>5000.
    """
    # converts A to array, raise ValueError if A has inf or nan
    A = np.asarray_chkfinite(A)
    m, n = A.shape

    if A.dtype not in _VALID_DTYPES:
        raise ValueError('A.dtype must be one of %s, not %s'
                         % (' '.join(_VALID_DTYPES), A.dtype))

    if A.dtype == np.complexfloating:
        real_type = np.float32 if A.dtype == np.complex64 else np.float64

    # Generate random measurement matrix and compress input matrix
    if sdist=='gaussian':
        C = np.random.standard_normal(size=(k+p, m)).astype(A.dtype)

        if A.dtype == np.complexfloating:
            C += 1j * np.random.standard_normal(size=(k+p , m)).astype(real_type)
        Y = C.dot(A)

    elif sdist=='spixel':
        C = np.random.sample(m, k+p)
        Y =  A[C , :]

    elif sdist=='sparse':
        density = m / np.log(m)
        C = sparse.rand(k+p, m, density=density**-1, format=formatS,
                        dtype=real_type, random_state=None)
        C.data = np.array(np.where(C.data >= 0.5, 1, -1), dtype=A.dtype)
        Y = C.dot(A)

    else:
        raise ValueError('Sampling distribution is not supported.')

    # Compute singular value decomposition
    _ , s , Vh = linalg.svd(Y, full_matrices=False, overwrite_a=True, check_finite=False)

    # truncate
    if k is not None:
        s = s[:k]
        Vh = Vh[:k, :]

    # Recover left-singular vectors
    U, s, Q = linalg.svd(A.dot(conjugate_transpose(Vh)),
                         full_matrices=False, overwrite_a=True, check_finite=False)

    return U, s, Q.dot(Vh)


def csvd2(A, k=None, p=10, sdist='sparse', formatS='csr'):
    """Compressed Singular Value Decomposition.

    Row compressed algorithm for computing the approximate low-rank singular value
    decomposition of a rectangular (m, n) matrix `A` with target rank `k << n`.
    The input matrix a is factored as `A = U * diag(s) * Vt`. The left singular
    vectors are the columns of the real or complex unitary matrix `U`. The right
    singular vectors are the columns of the real or complex unitary matrix `V`.
    The singular values `s` are non-negative and real numbers.

    The parameter `c` specifies the number of measurements and is required to
    be `c>k`.


    Parameters
    ----------
    A : array_like
        Real/complex input matrix  `A` with dimensions `(m, n)`.

    k : int
        `k` is the target rank of the low-rank decomposition, k << min(m,n).

    c : int
        `c` sets the number of measurments.

    sdist : str `{gaussian', 'spixel', 'sparse'}`
        Defines the sampling distribution.

    ortho : str `{True, False}`
        If `True` the left singular values are orthonormalized.

    method :   `{SVD, QR}`
        Defines the method to compute the orthnormalization step.

    scaled : str `{True, False}`
        If `True` the singular values are rescaled.

    fortmatS : str `{csr, coo}`
        Defines the format of the sparse measurement matrix.

    Returns
    -------
    U:  array_like
        Right singular values, array of shape `(m, k)`.

    s : array_like
        Singular values, 1-d array of length `k`.

    Vh : array_like
        Left singular values, array of shape `(k, n)`.


    Notes
    -----
    If the option `ortho=True` is selected, then the approximation is more
    accurate.

    If the sparse sampling distribution is used, the appropriate format for
    the sparse measurement matrix is crucial. In generall `csr` is the optimal
    format, but sometimes `coo` gives a better performance. Sparse matricies
    are computational efficient if the leading dimension is m>5000.
    """
    # converts A to array, raise ValueError if A has inf or nan
    A = np.asarray_chkfinite(A)
    m, n = A.shape

    if A.dtype not in _VALID_DTYPES:
        raise ValueError('A.dtype must be one of %s, not %s'
                         % (' '.join(_VALID_DTYPES), A.dtype))

    if A.dtype == np.complexfloating:
        real_type = np.float32 if A.dtype == np.complex64 else np.float64

    # Generate random measurement matrix and compress input matrix
    if sdist=='gaussian':
        C = np.random.standard_normal(size=(k+p, m)).astype(A.dtype)

        if A.dtype == np.complexfloating:
            C += 1j * np.random.standard_normal(size=(k+p , m)).astype(real_type)
        Y = C.dot(A)

    elif sdist=='spixel':
        C = np.random.sample(m, k+p)
        Y =  A[C , :]

    elif sdist=='sparse':
        density = m / np.log(m)
        C = sparse.rand(k+p, m, density=density**-1, format=formatS,
                        dtype=real_type, random_state=None)
        C.data = np.array(np.where(C.data >= 0.5, np.sqrt(density), -np.sqrt(density)),
                          dtype=A.dtype)
        Y = C.dot(A)

    else:
        raise ValueError('Sampling distribution is not supported.')

    # Compute singular value decomposition
    B = Y.dot(conjugate_transpose(Y))
    B = 0.5 * (B + conjugate_transpose(B))

    l = k+p
    lo, hi = (l-k), (l-1) # truncate
    s, T = linalg.eigh(B, b=None, lower=True, eigvals_only=False,
                       overwrite_a=True, overwrite_b=False, turbo=True, eigvals=None,
                       type=1, check_finite=False)


    # reverse the n first columns of u, and s
    T[:, :l] = T[:, l-1::-1]
    s = s[::-1]

    # truncate
    if k is not None:
        s = s[:k]
        T = T[:, :k]

    mask = s > 0
    s[mask] = s[mask]**0.5

    V = conjugate_transpose(Y).dot(T[:,mask] / s[mask])

    # Recover left-singular vectors
    U, s, Vhstar = linalg.svd(A.dot(V), full_matrices=False,
                              overwrite_a=True, check_finite=False)

    return U, s, Vhstar.dot(conjugate_transpose(V))


def csvd_double(A, k=None, p=10, sdist='sparse', formatS='csr'):
    """Compressed Singular Value Decomposition.

    Row compressed algorithm for computing the approximate low-rank singular value
    decomposition of a rectangular (m, n) matrix `A` with target rank `k << n`.
    The input matrix a is factored as `A = U * diag(s) * Vt`. The left singular
    vectors are the columns of the real or complex unitary matrix `U`. The right
    singular vectors are the columns of the real or complex unitary matrix `V`.
    The singular values `s` are non-negative and real numbers.

    The parameter `c` specifies the number of measurements and is required to
    be `c>k`.


    Parameters
    ----------
    A : array_like
        Real/complex input matrix  `A` with dimensions `(m, n)`.

    k : int
        `k` is the target rank of the low-rank decomposition, k << min(m,n).

    c : int
        `c` sets the number of measurments.

    sdist : str `{gaussian', 'spixel', 'sparse'}`
        Defines the sampling distribution.

    ortho : str `{True, False}`
        If `True` the left singular values are orthonormalized.

    method :   `{SVD, QR}`
        Defines the method to compute the orthnormalization step.

    scaled : str `{True, False}`
        If `True` the singular values are rescaled.

    fortmatS : str `{csr, coo}`
        Defines the format of the sparse measurement matrix.

    Returns
    -------
    U:  array_like
        Right singular values, array of shape `(m, k)`.

    s : array_like
        Singular values, 1-d array of length `k`.

    Vh : array_like
        Left singular values, array of shape `(k, n)`.


    Notes
    -----
    If the option `ortho=True` is selected, then the approximation is more
    accurate.

    If the sparse sampling distribution is used, the appropriate format for
    the sparse measurement matrix is crucial. In generall `csr` is the optimal
    format, but sometimes `coo` gives a better performance. Sparse matricies
    are computational efficient if the leading dimension is m>5000.
    """
    # converts A to array, raise ValueError if A has inf or nan
    A = np.asarray_chkfinite(A)
    m, n = A.shape

    if A.dtype not in _VALID_DTYPES:
        raise ValueError('A.dtype must be one of %s, not %s'
                         % (' '.join(_VALID_DTYPES), A.dtype))

    # Generate random measurement matrix and compress input matrix
    # Generate a random test matrix Omega
    Omega = random.sample(n, k+p)
    Psi = random.sample(m, k+p)

    L, _ = linalg.qr(A[:, Omega], mode='economic', check_finite=False, overwrite_a=False)
    R, _ = linalg.qr(A[Psi, :].T, mode='economic', check_finite=False, overwrite_a=False)

    #Project the data matrix a into a lower dimensional subspace
    #B = Q.T * A
    D = conjugate_transpose(L).dot(A)
    D = D.dot(R)

    # Compute singular value decomposition
    U, s, Vh = linalg.svd(D, full_matrices=False, overwrite_a=True, check_finite=False)

    return L.dot(U), s, Vh.dot(R.T)
