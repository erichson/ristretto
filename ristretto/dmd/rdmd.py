"""
Randomized Dynamic Mode Decomposition (DMD).
"""
# Authors: N. Benjamin Erichson
#          Joseph Knox
# License: GNU General Public License v3.0

from __future__ import division

from .dmd import dmd, _get_amplitudes
from ..mf import rqb


def rdmd(A, dt=1, k=None, p=10, q=2, sdist='uniform', single_pass=False,
         return_amplitudes=False, return_vandermonde=False, order=True):
    """
    Randomized Dynamic Mode Decomposition.

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

    k : int
        If `k < (n-1)` low-k Dynamic Mode Decomposition is computed.

    p : int, optional
        Oversampling paramater.

    q : int, optional
        Number of subspace iterations to perform.

    sdist : str `{'uniform', 'normal'}`
        Specify the distribution of the sensing matrix `S`.

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
    """
    # Compute QB decomposition
    Q, B = rqb(A, k=k, p=p, q=q, sdist=sdist)

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
