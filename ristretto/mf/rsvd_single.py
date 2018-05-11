"""
Randomized Singular Value Decomposition
"""
# TODO: Add option for sparse random test matrices.
# TODO:  Modify algorithm to allow for the streaming model.
# Authors: N. Benjamin Erichson
#          Joseph Knox
# License: GNU General Public License v3.0

from __future__ import division

import numpy as np
from scipy import linalg

from ..dmd.utils import conjugate_transpose
from ..dmd.rdmd import _get_sdist_func

_VALID_DTYPES = (np.float32, np.float64, np.complex64, np.complex128)
_VALID_SDISTS = ('uniform', 'normal', 'orthogonal')


def rsvd_single(A, k=None, p=10, l=None, sdist='uniform'):
    """Randomized Singular Value Decomposition Single-View.

    Randomized algorithm for computing the approximate low-rank singular value
    decomposition of a rectangular `(m, n)` matrix `A`, with target rank `k << min{m, n}`.
    The input matrix is factored as `A = U * diag(s) * Vt`. The right singular
    vectors are the columns of the real or complex unitary matrix `U`. The left
    singular vectors are the columns of the real or complex unitary matrix `V`.
    The singular values `s` are non-negative and real numbers.

    This algorithms implements a (pseudo) single pass algorithm.


    Parameters
    ----------
    A : array_like, shape `(m, n)`.
        Real nonnegative input matrix.

    k : integer, `k << min{m,n}`.
        Target rank.

    p : integer, default: `p=10`.
        Parameter to control oversampling of column space.

    l : integer, default: `l=2*p`.
        Parameter to control oversampling of row space.

    sdist : str `{'uniform', 'normal', 'orthogonal'}`, default: `sdist='uniform'`.
        'uniform' : Random test matrices with uniform distributed elements.

        'normal' : Random test matrices with normal distributed elements.

        'orthogonal' : Orthogonalized random test matrices with uniform distributed elements.


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
    Tropp, Joel A., et al.
    "Randomized single-view algorithms for low-rank matrix approximation" (2016).
    (available at `arXiv <https://arxiv.org/abs/1609.00048>`_).
    """
    # converts A to array, raise ValueError if A has inf or nan
    A = np.asarray_chkfinite(A)
    m, n = A.shape

    flipped = False
    if m < n:
        A = conjugate_transpose(A)
        m , n = A.shape
        flipped = True

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

    if l is None:
        # defalut
        l = 2*p

    # distribution to draw random samples
    sdist_func = _get_sdist_func(sdist)

    #Generate a random test matrix Omega
    Omega = sdist_func(size=(n, k+p)).astype(A.dtype)
    Psi = sdist_func(size=(k+l, m)).astype(A.dtype)

    if A.dtype == np.complexfloating:
        real_type = np.float32 if A.dtype == np.complex64 else np.float64
        Omega += 1j * sdist_func(size=(n, k+p)).astype(real_type)
        Psi += 1j * sdist_func(size=(k+l, m)).astype(real_type)

    if sdist=='orthogonal':
        Omega, _ = linalg.qr(Omega,  mode='economic', check_finite=False, overwrite_a=True)
        Psi, _ = linalg.qr(Psi.T,  mode='economic', check_finite=False, overwrite_a=True)
        Psi = Psi.T

    #Build sample matrix Y = A * Omega and W = Psi * A
    # NOTE: Y should approximate the column space and W the row space of A
    Y = A.dot(Omega)
    W = Psi.dot(A)
    del Omega

    #Orthogonalize Y using economic QR decomposition: Y=QR
    Q, _ = linalg.qr(Y,  mode='economic', check_finite=False, overwrite_a=True)
    U, T = linalg.qr(Psi.dot(Q),  mode='economic', check_finite=False, overwrite_a=False)

    # Form a smaller matrix
    B = linalg.solve(a=T, b=conjugate_transpose(U).dot(W))

    # Singular Value Decomposition
    # NOTE: B = U" * S * Vt
    U, s, Vt = linalg.svd(B, compute_uv=True, full_matrices=False,
                          overwrite_a=True, check_finite=False)

    # Recover right singular vectors
    U = Q.dot(U)

    # Return Trunc
    if flipped:
        return conjugate_transpose(Vt)[:, :k], s[:k], conjugate_transpose(U)[:k, :]

    return U[:, :k], s[:k], Vt[:k , :]
