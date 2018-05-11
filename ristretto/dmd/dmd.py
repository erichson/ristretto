"""
Dynamic Mode Decomposition (DMD).
"""
# Authors: N. Benjamin Erichson
#          Joseph Knox
# License: GNU General Public License v3.0

from __future__ import division

import numpy as np
from scipy import linalg

from .utils import conjugate_transpose

_VALID_DTYPES = (np.float32, np.float64, np.complex64, np.complex128)
_VALID_MODES = ('standard', 'exact', 'exact_scaled')


def _get_amplitudes(F, A):
    """return amplitudes given dynamic modes F and original array A"""
    result = linalg.lstsq(F, A[:, 0])
    return result[0]


def dmd(A, dt=1, k=None, modes='exact', return_amplitudes=False,
        return_vandermonde=False, order=True):
    """Dynamic Mode Decomposition.

    Dynamic Mode Decomposition (DMD) is a data processing algorithm which
    allows to decompose a matrix `A` in space and time. The matrix `A` is
    decomposed as `A = F * B * V`, where the columns of `F` contain the dynamic modes.
    The modes are ordered corresponding to the amplitudes stored in the diagonal
    matrix `B`. `V` is a Vandermonde matrix describing the temporal evolution.


    Parameters
    ----------
    A : array_like
        Real/complex input matrix  `a` with dimensions `(m, n)`.

    dt : scalar or array_like
        Factor specifying the time difference between the observations.

    k : int, optional
        If `k < (n-1)` low-rank Dynamic Mode Decomposition is computed.

    modes : str `{'standard', 'exact', 'exact_scaled'}`
        - 'standard' : uses the standard definition to compute the dynamic modes, `F = U * W`.
        - 'exact' : computes the exact dynamic modes, `F = Y * V * (S**-1) * W`.
        - 'exact_scaled' : computes the exact dynamic modes, `F = (1/l) * Y * V * (S**-1) * W`.

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
    J. H. Tu, et al.
    "On Dynamic Mode Decomposition: Theory and Applications" (2013).
    (available at `arXiv <http://arxiv.org/abs/1312.0041>`_).

    N. B. Erichson and C. Donovan.
    "Randomized Low-Rank Dynamic Mode Decomposition for Motion Detection" (2015).
    Under Review.
    """
    # converts A to array, raise ValueError if A has inf or nan
    A = np.asarray_chkfinite(A)
    m, n = A.shape

    if modes not in _VALID_MODES:
        raise ValueError('modes must be one of %s, not %s'
                         % (' '.join(_VALID_MODES), modes))

    if A.dtype not in _VALID_DTYPES:
        raise ValueError('A.dtype must be one of %s, not %s'
                         % (' '.join(_VALID_DTYPES), A.dtype))

    if k is not None and (k < 1 or k > n - 1):
        raise ValueError('k must be > 1 and less than n - 1')

    #Split data into lef and right snapshot sequence
    X = A[:, :(n-1)] #pointer
    Y = A[:, 1:n] #pointer

    #Singular Value Decomposition
    U, s, Vh = linalg.svd(X, compute_uv=True, full_matrices=False,
                          overwrite_a=False, check_finite=True)

    if k is not None:
        U = U[:, :k]
        s = s[:k]
        Vh = Vh[:k, :]

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Solve the LS problem to find estimate for M using the pseudo-inverse
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #real: M = U.T * Y * Vt.T * S**-1
    #complex: M = U.H * Y * Vt.H * S**-1
    #Let G = Y * Vt.H * S**-1, hence M = M * G
    G = np.dot(Y, conjugate_transpose(Vh)) / s
    M = np.dot(conjugate_transpose(U), G)

    #Eigen Decomposition
    l, W = linalg.eig(M, right=True, overwrite_a=True)
    omega = np.log(l) / dt

    if order:
        # return ordered result
        sort_idx = np.argsort(np.abs(omega))
        W = W[:, sort_idx]
        l = l[sort_idx]
        omega = omega[sort_idx]

    #Compute DMD Modes
    if modes == 'standard':
        F = np.dot(U, W)
    else:
        F = np.dot(G, W)
        if modes == 'exact_scaled':
            F /= l

    result = [F]
    if return_amplitudes:
        #Compute amplitueds b using least-squares: Fb=x1
        b = _get_amplitudes(F, A)
        result.append(b)

    if return_vandermonde:
        #Compute Vandermonde matrix
        V = np.fliplr(np.vander(l , N=n))
        result.append(V)

    result.append(omega)
    return result
