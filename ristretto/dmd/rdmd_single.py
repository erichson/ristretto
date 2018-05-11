"""
Randomized Singular Value Decomposition.
"""
# Authors: N. Benjamin Erichson
#          Joseph Knox
# License: GNU General Public License v3.0

# TODO: Add option for sparse random test matrices.
# TODO: Modify algorithm to allow for the streaming model.

from __future__ import division

import numpy as np
from scipy import linalg

from .dmd import dmd, _get_amplitudes
from .rdmd import _get_sdist_func
from .utils import conjugate_transpose

_VALID_DTYPES = (np.float32, np.float64, np.complex64, np.complex128)
_VALID_SDISTS = ('uniform', 'normal', 'orthogonal')


def rdmd_single(A, dt = 1, k=None, p=10, l=None, sdist='uniform',
                return_amplitudes=False, return_vandermonde=False, order=True):
    """Randomized Dynamic Mode Decomposition Single-View.

    Dynamic Mode Decomposition (DMD) is a data processing algorithm which
    allows to decompose a matrix `A` in space and time. The matrix `A` is
    decomposed as `A = F * B * V`, where the columns of `F` contain the dynamic modes.
    The modes are ordered corresponding to the amplitudes stored in the diagonal
    matrix `B`. `V` is a Vandermonde matrix describing the temporal evolution.

    This algorithms implements a single pass algorithm.

    Parameters
    ----------
    A : array_like
        Real/complex input matrix  `a` with dimensions `(m, n)`.

    dt : scalar or array_like
        Factor specifying the time difference between the observations.

    k : int
        If `k < (n-1)` low-k Dynamic Mode Decomposition is computed.

    p : integer, default: `p=10`.
        Parameter to control oversampling of column space.

    l : integer, default: `l=2*p`.
        Parameter to control oversampling of row space.

    sdist : str `{'uniform', 'normal', 'orthogonal'}`, default: `sdist='uniform'`.
        'uniform' : Random test matrices with uniform distributed elements.

        'normal' : Random test matrices with normal distributed elements.

        'orthogonal' : Orthogonalized random test matrices with uniform distributed elements.

    return_amplitudes : bool `{True, False}`
        True: return amplitudes in addition to dynamic modes.

    return_vandermonde : bool `{True, False}`
        True: return Vandermonde matrix in addition to dynamic modes and amplitudes.

    order :  bool `{True, False}`
        True: return modes sorted.


    Returns
    -------
    F : array_like
        Matrix containing the dynamic modes of shape `(m, n-1)`  or `(m, k)`.

    b : array_like, if `return_amplitudes=True`
        1-D array containing the amplitudes of length `min(n-1, k)`.

    V : array_like, if `return_vandermonde=True`
        Vandermonde matrix of shape `(n-1, n-1)`  or `(k, n-1)`.

    omega : array_like
        Time scaled eigenvalues: `ln(l)/dt`.


    References
    ----------
    Tropp, Joel A., et al.
    "Randomized single-view algorithms for low-k matrix approximation" (2016).
    (available at `arXiv <https://arxiv.org/abs/1609.00048>`_).
    """
    # converts A to array, raise ValueError if A has inf or nan
    A = np.asarray_chkfinite(A)
    m, n = A.shape

    if sdist not in _VALID_SDISTS:
        raise ValueError('sdist must be one of %s, not %s'
                         % (' '.join(_VALID_SDISTS), sdist))

    if A.dtype not in _VALID_DTYPES:
        raise ValueError('A.dtype must be one of %s, not %s'
                         % (' '.join(_VALID_DTYPES), A.dtype))

    if k > min(m,n) or k < 1:
        raise ValueError('If specified, k must be < min(n,m) and > 1')

    if k is None:
        # defualt
        k = min(m, n)

    # distribution to draw random samples
    sdist_func = _get_sdist_func(sdist)

    if l is None:
        # default to twice the column oversampling rate.
        l = 2*p

    #Generate a random test matrix Omega
    Omega = sdist_func(size=(n, k+p)).astype(A.dtype)
    Psi = sdist_func(size=(k+l, m)).astype(A.dtype)

    if A.dtype == np.complexfloating:
        real_type = np.float32 if A.dtype == np.complex64 else np.float64
        Omega += 1j * sdist_func(size=(n, k+p)).astype(real_type)
        Psi += 1j * sdist_func(size=(k+l, m)).astype(real_type)

    if sdist == 'orthogonal':
        Omega, _ = linalg.qr(Omega, mode='economic', check_finite=False, overwrite_a=True)
        Psi , _ = linalg.qr(Psi.T, mode='economic', check_finite=False, overwrite_a=True)
        Psi = Psi.T

    #Build sample matrix Y = A * Omega and W = Psi * A
    #Note: Y should approximate the column space and W the row space of A
    Y = A.dot(Omega)
    W = Psi.dot(A)
    del Omega

    #Orthogonalize Y using economic QR decomposition: Y=QR
    Q, _ = linalg.qr(Y, mode='economic', check_finite=False, overwrite_a=True )
    U, T = linalg.qr(Psi.dot(Q), mode='economic', check_finite=False, overwrite_a=False )

    # Form a smaller matrix
    B = linalg.solve(a=T, b=conjugate_transpose(U).dot(W))

    # only difference is we need to premultiply F from dmd
    # vandermonde is basically already computed
    # TODO: factor out the rest so no code is repeated
    F, V, omega = dmd(B, dt=dt, k=k, modes='standard',return_amplitudes=False,
                      return_vandermonde=True, order=order)

    #Compute DMD Modes
    F = Q.dot(F)

    result = [F]
    if return_amplitudes:
        #Compute amplitueds b using least-squares: Fb=x1
        b = _get_amplitudes(F, A)
        result.append(b)

    if return_vandermonde:
        result.append(V)

    result.append(omega)
    return result
