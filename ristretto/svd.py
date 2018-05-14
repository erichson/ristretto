"""
Compressed and Random Singular Value Decompositions.
"""
# TODO: implement 'ortho' option
# TODO: implement 'method' option
# TODO: implement 'scaled' option
# TODO; implement 'sdists' for csvd_double

# Authors: N. Benjamin Erichson
#          Joseph Knox
# License: GNU General Public License v3.0

from __future__ import division

import numpy as np
from scipy import linalg
from scipy import sparse

from .qb import rqb
from .utils import check_random_state, conjugate_transpose

_VALID_DTYPES = (np.float32, np.float64, np.complex64, np.complex128)
_VALID_SDISTS = ('gaussian', 'spixel', 'sparse')


def _sparse_sample(A, k, p, sdist, formatS, check_finite=False, random_state=None):
    random_state = check_random_state(random_state)

    # converts A to array, raise ValueError if A has inf or nan
    A = np.asarray_chkfinite(A) if check_finite else np.asarray(A)
    m, n = A.shape

    if A.dtype not in _VALID_DTYPES:
        raise ValueError('A.dtype must be one of %s, not %s'
                         % (' '.join(_VALID_DTYPES), A.dtype))

    if sdist not in _VALID_SDISTS:
        raise ValueError('sdists must be one of %s, not %s'
                         % (' '.join(_VALID_SDISTS), sdist))

    if A.dtype == np.complexfloating:
        real_type = np.float32 if A.dtype == np.complex64 else np.float64

    # Generate random measurement matrix and compress input matrix
    if sdist=='gaussian':
        C = random_state.standard_normal(size=(k+p, m)).astype(A.dtype)

        if A.dtype == np.complexfloating:
            C += 1j * random_state.standard_normal(size=(k+p , m)).astype(real_type)
        Y = C.dot(A)

    elif sdist=='spixel':
        C = random_state.sample(m, k+p)
        Y = A[C, :]

    else:
        # sdist == 'sparse'
        density = m / np.log(m)
        C = sparse.rand(k+p, m, density=density**-1, format=formatS,
                        dtype=real_type, random_state=random_state)
        C.data = np.array(np.where(C.data >= 0.5, 1, -1), dtype=A.dtype)
        Y = C.dot(A)

    return Y


def csvd(A, k=None, p=10, sdist='sparse', formatS='csr', random_state=None):
    """Compressed Singular Value Decomposition.

    Row compressed algorithm for computing the approximate low-rank singular value
    decomposition of a rectangular (m, n) matrix `A` with target rank `k << n`.
    The input matrix a is factored as `A = U * diag(s) * Vt`. The left singular
    vectors are the columns of the real or complex unitary matrix `U`. The right
    singular vectors are the columns of the real or complex unitary matrix `V`.
    The singular values `s` are non-negative and real numbers.


    Parameters
    ----------
    A : array_like
        Real/complex input matrix  `A` with dimensions `(m, n)`.

    k : int
        `k` is the target rank of the low-rank decomposition, k << min(m,n).

    p : int
        `p` oversampling parameter. The number of measurements will be `p+k`

    sdist : str `{gaussian', 'spixel', 'sparse'}`
        Defines the sampling distribution.

    fortmatS : str `{csr, coo}`
        Defines the format of the sparse measurement matrix.

    random_state : integer, RandomState instance or None, optional (default ``None``)
        If integer, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used by np.random.


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
    If the sparse sampling distribution is used, the appropriate format for
    the sparse measurement matrix is crucial. In generall `csr` is the optimal
    format, but sometimes `coo` gives a better performance. Sparse matricies
    are computational efficient if the leading dimension is m>5000.
    """
    # get sparse sketch of A
    Y = _sparse_sample(A, k, p, sdist, formatS, check_finite=True,
                       random_state=random_state)

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


def csvd2(A, k=None, p=10, sdist='sparse', formatS='csr', random_state=None):
    """Compressed Singular Value Decomposition.

    Row compressed algorithm for computing the approximate low-rank singular value
    decomposition of a rectangular (m, n) matrix `A` with target rank `k << n`.
    The input matrix a is factored as `A = U * diag(s) * Vt`. The left singular
    vectors are the columns of the real or complex unitary matrix `U`. The right
    singular vectors are the columns of the real or complex unitary matrix `V`.
    The singular values `s` are non-negative and real numbers.


    Parameters
    ----------
    A : array_like
        Real/complex input matrix  `A` with dimensions `(m, n)`.

    k : int
        `k` is the target rank of the low-rank decomposition, k << min(m,n).

    p : int
        `p` oversampling parameter. The number of measurements will be `p+k`

    sdist : str `{gaussian', 'spixel', 'sparse'}`
        Defines the sampling distribution.

    fortmatS : str `{csr, coo}`
        Defines the format of the sparse measurement matrix.

    random_state : integer, RandomState instance or None, optional (default ``None``)
        If integer, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used by np.random.


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
    If the sparse sampling distribution is used, the appropriate format for
    the sparse measurement matrix is crucial. In generall `csr` is the optimal
    format, but sometimes `coo` gives a better performance. Sparse matricies
    are computational efficient if the leading dimension is m>5000.
    """
    # get sparse sketch of A
    Y = _sparse_sample(A, k, p, sdist, formatS, check_finite=True,
                       random_state=random_state)

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
    s[mask] = np.sqrt(s[mask])

    V = conjugate_transpose(Y).dot(T[:,mask] / s[mask])

    # Recover left-singular vectors
    U, s, Vhstar = linalg.svd(A.dot(V), full_matrices=False,
                              overwrite_a=True, check_finite=False)

    return U, s, Vhstar.dot(conjugate_transpose(V))


def csvd_double(A, k=None, p=10, formatS='csr', random_state=None):
    """Compressed Singular Value Decomposition.

    Row compressed algorithm for computing the approximate low-rank singular value
    decomposition of a rectangular (m, n) matrix `A` with target rank `k << n`.
    The input matrix a is factored as `A = U * diag(s) * Vt`. The left singular
    vectors are the columns of the real or complex unitary matrix `U`. The right
    singular vectors are the columns of the real or complex unitary matrix `V`.
    The singular values `s` are non-negative and real numbers.


    Parameters
    ----------
    A : array_like
        Real/complex input matrix  `A` with dimensions `(m, n)`.

    k : int
        `k` is the target rank of the low-rank decomposition, k << min(m,n).

    p : int
        `p` oversampling parameter. The number of measurements will be `p+k`

    fortmatS : str `{csr, coo}`
        Defines the format of the sparse measurement matrix.

    random_state : integer, RandomState instance or None, optional (default ``None``)
        If integer, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used by np.random.


    Returns
    -------
    U:  array_like
        Right singular values, array of shape `(m, k)`.

    s : array_like
        Singular values, 1-d array of length `k`.

    Vh : array_like
        Left singular values, array of shape `(k, n)`.
    """
    random_state = check_random_state(random_state)

    # converts A to array, raise ValueError if A has inf or nan
    A = np.asarray_chkfinite(A)
    m, n = A.shape

    if A.dtype not in _VALID_DTYPES:
        raise ValueError('A.dtype must be one of %s, not %s'
                         % (' '.join(_VALID_DTYPES), A.dtype))

    # Generate random measurement matrix and compress input matrix
    # Generate a random test matrix Omega
    Omega = random_state.sample(n, k+p)
    Psi = random_state.sample(m, k+p)

    L, _ = linalg.qr(A[:, Omega], mode='economic', check_finite=False, overwrite_a=False)
    R, _ = linalg.qr(A[Psi, :].T, mode='economic', check_finite=False, overwrite_a=False)

    #Project the data matrix a into a lower dimensional subspace
    #B = Q.T * A
    D = conjugate_transpose(L).dot(A)
    D = D.dot(R)

    # Compute singular value decomposition
    U, s, Vh = linalg.svd(D, full_matrices=False, overwrite_a=True, check_finite=False)

    return L.dot(U), s, Vh.dot(R.T)


def rsvd(A, k=None, p=10, l=None, q=1, sdist='uniform', single_pass=False,
         random_state=None):
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
        Parameter to control oversampling of column space.

    l : integer, default: `l=2*p`.
        Parameter to control oversampling of row space. Only relevant if
        single_pass == True.

    q : integer, default: `q=1`.
        Parameter to control number of power (subspace) iterations. Only
        relevant if single_pass == False.

    sdist : str `{'uniform', 'normal'}`, default: `sdist='uniform'`.
        'uniform' : Random test matrix with uniform distributed elements.

        'normal' : Random test matrix with normal distributed elements.

    single_pass : bool
        If single_pass == True, perfom single pass of algorithm.

    random_state : integer, RandomState instance or None, optional (default ``None``)
        If integer, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used by np.random.


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

    # Compute QB decomposition
    Q, B = rqb(A, k=k, p=p, l=l, q=q, sdist=sdist, single_pass=single_pass,
               random_state=random_state)

    # Compute SVD
    U, s, Vt = linalg.svd(B, compute_uv=True, full_matrices=False,
                          overwrite_a=True, check_finite=False)

    # Recover right singular vectors
    U = Q.dot(U)

    # Return Trunc
    if flipped:
        return conjugate_transpose(Vt)[:, :k], s[:k], conjugate_transpose(U)[:k, :]

    return U[:, :k], s[:k], Vt[:k, :]
