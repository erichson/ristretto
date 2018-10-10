# TODO: add docs for DMD and RDMD
# TODO: complete class methods (fit_transform, transform, inverse_transform)
# TODO: change function/class attribute names to be consistent
#       (ie l == amplitudes)
"""
Dynamic Mode Decomposition (DMD).
"""
# Authors: N. Benjamin Erichson
#          Joseph Knox
# License: GNU General Public License v3.0
from __future__ import division

import numpy as np
from scipy import linalg
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from .qb import compute_rqb
from .utils import conjugate_transpose

_VALID_MODES = ('standard', 'exact', 'exact_scaled')


def compute_dmd(A, rank=None, dt=1, modes='exact', order=True):
    """Dynamic Mode Decomposition.

    Dynamic Mode Decomposition (DMD) is a data processing algorithm which
    allows to decompose a matrix `A` in space and time. The matrix `A` is
    decomposed as `A = F * B * V`, where the columns of `F` contain the dynamic modes.
    The modes are ordered corresponding to the amplitudes stored in the diagonal
    matrix `B`. `V` is a Vandermonde matrix describing the temporal evolution.


    Parameters
    ----------
    A : array_like, shape `(m, n)`.
        Input array.

    rank : int
        If `rank < (n-1)` low-rank Dynamic Mode Decomposition is computed.

    dt : scalar or array_like, optional (default: 1)
        Factor specifying the time difference between the observations.

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
        1-D array containing the amplitudes of length `min(n-1, rank)`.

    V : array_like, if `return_vandermonde=True`
        Vandermonde matrix of shape `(n-1, n-1)`  or `(rank, n-1)`.

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

    if rank is not None and (rank < 1 or rank > n):
        raise ValueError('rank must be > 1 and less than n')

    # Split data into lef and right snapshot sequence
    X = A[:, :(n-1)]  # pointer
    Y = A[:, 1:n]     # pointer

    # Singular Value Decomposition
    U, s, Vh = linalg.svd(X, compute_uv=True, full_matrices=False,
                          overwrite_a=False, check_finite=True)

    if rank is not None:
        U = U[:, :rank]
        s = s[:rank]
        Vh = Vh[:rank, :]

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Solve the LS problem to find estimate for M using the pseudo-inverse
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # real: M = U.T * Y * Vt.T * S**-1
    # complex: M = U.H * Y * Vt.H * S**-1
    # Let G = Y * Vt.H * S**-1, hence M = M * G
    G = np.dot(Y, conjugate_transpose(Vh)) / s
    M = np.dot(conjugate_transpose(U), G)

    # Eigen Decomposition
    l, W = linalg.eig(M, right=True, overwrite_a=True)
    omega = np.log(l) / dt

    if order:
        # return ordered result
        sort_idx = np.argsort(np.abs(omega))
        W = W[:, sort_idx]
        l = l[sort_idx]
        omega = omega[sort_idx]

    # Compute DMD Modes
    if modes == 'standard':
        F = np.dot(U, W)
    else:
        F = np.dot(G, W)
        if modes == 'exact_scaled':
            F /= l

    return F, l, omega


def compute_rdmd(A, rank, dt=1, oversample=10, n_subspace=2, modes='standard',
                 order=True, random_state=None):
    """Randomized Dynamic Mode Decomposition.

    Dynamic Mode Decomposition (DMD) is a data processing algorithm which
    allows to decompose a matrix `A` in space and time. The matrix `A` is
    decomposed as `A = F * B * V`, where the columns of `F` contain the dynamic modes.
    The modes are ordered corresponding to the amplitudes stored in the diagonal
    matrix `B`. `V` is a Vandermonde matrix describing the temporal evolution.

    The quality of the approximation can be controlled via the oversampling
    parameter `oversample` and `n_subspace` which specifies the number of
    subspace iterations.


    Parameters
    ----------
    A : array_like, shape `(m, n)`.
        Input array.

    rank : integer
        Target rank. Best if `rank << min{m,n}`

    dt : scalar or array_like
        Factor specifying the time difference between the observations.

    oversample : integer, optional (default: 10)
        Controls the oversampling of column space. Increasing this parameter
        may improve numerical accuracy.

    n_subspace : integer, default: 2.
        Parameter to control number of subspace iterations. Increasing this
        parameter may improve numerical accuracy.

    return_amplitudes : bool `{True, False}`
        True: return amplitudes in addition to dynamic modes.

    return_vandermonde : bool `{True, False}`
        True: return Vandermonde matrix in addition to dynamic modes and amplitudes.

    order :  bool `{True, False}`
        True: return modes sorted.

    random_state : integer, RandomState instance or None, optional (default ``None``)
        If integer, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used by np.random.


    Returns
    -------
    F : array_like
        Matrix containing the dynamic modes of shape `(m, rank)`.

    b : array_like, if `return_amplitudes=True`
        1-D array containing the amplitudes of length `min(n-1, rank)`.

    V : array_like, if `return_vandermonde=True`
        Vandermonde matrix of shape `(rank, n-1)`.

    omega : array_like
        Time scaled eigenvalues: `ln(l)/dt`.


    References
    ----------
    Tropp, Joel A., et al.
    "Randomized single-view algorithms for low-k matrix approximation" (2016).
    (available at `arXiv <https://arxiv.org/abs/1609.00048>`_).
    """
    # Compute QB decomposition
    Q, B = compute_rqb(A, rank, oversample=oversample, n_subspace=n_subspace,
                       random_state=random_state)

    # only difference is we need to premultiply F from dmd
    # vandermonde is basically already computed
    F, l, omega = compute_dmd(B, rank=rank, dt=dt, modes=modes, order=order)

    # Compute DMD Modes
    F = Q.dot(F)

    return F, l, omega


def get_amplitudes(A, F):
    '''Compute amplitueds b using least-squares: Fb=x1'''
    return linalg.lstsq(F, A[:, 0])[0]


def get_vandermonde(A, l):
    '''Compute Vandermonde matrix'''
    return np.fliplr(np.vander(l, N=A.shape[1]))


class DMD(BaseEstimator):

    def __init__(self, rank=None, dt=1, modes='exact', order=True):
        self.rank = rank
        self.dt = dt
        self.modes = modes
        self.order = order

    def fit(self, X, y=None):
        '''Fits DMD'''
        self.X_ = np.asarray_chkfinite(X)
        self.F_, self.l_, self.omega_ = compute_dmd(
            self.X_, rank=self.rank, dt=self.dt, modes=self.modes,
            order=self.order)
        return self

    @property
    def amplitudes_(self):
        '''Compute amplitueds b using least-squares: Fb=x1'''
        check_is_fitted(self, ['X_', 'F_'])
        return get_amplitudes(self.X_, self.F_)

    @property
    def vandermonde_(self):
        '''Compute Vandermonde matrix'''
        check_is_fitted(self, ['X_', 'l_'])
        return get_vandermonde(self.X_, self.l_)

    def fit_transform(self, X):
        '''TODO: Not Implemented'''
        raise NotImplementedError

    def transform(self, X):
        '''TODO: Not Implemented'''
        raise NotImplementedError

    def inverse_transform(self, X):
        '''TODO: Not Implemented'''
        raise NotImplementedError


class RDMD(DMD):

    def __init__(self, rank=None, dt=1, oversample=10, n_subspace=2,
                 modes='standard', order=True, random_state=None):
        super(RDMD, self).__init__(rank=rank, dt=dt, modes=modes, order=order)
        self.oversample = oversample
        self.n_subspace = n_subspace
        self.random_state = random_state

    def fit(self, X):
        self.X_ = np.asarray_chkfinite(X)
        self.F_, self.l_, self.omega_ = compute_rdmd(
            self.X_, rank=self.rank, dt=self.dt, oversample=self.oversample,
            n_subspace=self.n_subspace, modes=self.modes, order=self.order,
            random_state=self.random_state)
        return self
