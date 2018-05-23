"""
Interpolative decomposition (ID)
"""
# Authors: N. Benjamin Erichson
#          Joseph Knox
# License: GNU General Public License v3.0

from __future__ import division

import numpy as np
from scipy import linalg

from .qb import rqb
from .utils import check_random_state, conjugate_transpose

_VALID_DTYPES = (np.float32, np.float64, np.complex64, np.complex128)
_VALID_SDISTS = ('uniform', 'normal')
_VALID_MODES = ('row', 'column')

def _get_distribution_func(distribution, random_state):
    if distribution == 'uniform':
        return partial(random_state.uniform, -1, 1)
    return random_state.standard_normal


def interp_decomp(A, k=None, mode='column', index_set=False):
    """Interpolative decomposition (ID).

    Algorithm for computing the low-rank ID
    decomposition of a rectangular `(m, n)` matrix `A`, with target rank `k << min{m, n}`.
    Input matrix is factored as `A = C * V`, using the column pivoted QR decomposition.
    The factor matrix `C` is formed of a subset of columns of `A`,
    also called the partial column skeleton. The factor matrix `V` contains
    a `(k, k)` identity matrix as a submatrix, and is well-conditioned.

    If `mode='row'`, then the input matrix is factored as `A = Z * R`, using the
    row pivoted QR decomposition. The factor matrix `R` is now formed as
    a subset of rows of `A`, also called the partial row skeleton.

    Parameters
    ----------
    A : array_like, shape `(m, n)`.
        Input matrix.

    k : integer, `k << min{m,n}`.
        Target rank.

    mode: str `{'column', 'row'}`, default: `mode='column'`.
        'column' : ID using column pivoted QR.
        'row' : ID using row pivoted QR.

    index_set: str `{'True', 'False'}`, default: `index_set='False'`.
        'True' : Return column/row index set instead of `C` or `R`.

    Returns
    -------
    If `mode='column'`:
        C:  array_like, shape `(m, k)`.
            Partial column skeleton.

        V : array_like, shape `(k, n)`.
            Well-conditioned matrix.

    If `mode='row'`:
        Z:  array_like, shape `(m, k)`.
            Well-conditioned matrix.

        R : array_like, shape `(k, n)`.
            Partial row skeleton.

    References
    ----------
    S. Voronin and P.Martinsson.
    "RSVDPACK: Subroutines for computing partial singular value
    decompositions via randomized sampling on single core, multi core,
    and GPU architectures" (2015).
    (available at `arXiv <http://arxiv.org/abs/1502.05366>`_).
    """
    if mode not in _VALID_MODES:
        raise ValueError('mode must be one of %s, not %s'
                         % (' '.join(_VALID_MODES), mode))

    # converts A to array, raise ValueError if A has inf or nan
    A = np.asarray_chkfinite(A)
    if mode=='row':
        A = conjugate_transpose(A)
    m, n = A.shape

    if A.dtype not in _VALID_DTYPES:
        raise ValueError('A.dtype must be one of %s, not %s'
                         % (' '.join(_VALID_DTYPES), A.dtype))

    if k is None:
        # default
        k = min(m,n)

    if k < 1 or k > min(m, n):
        raise ValueError("Target rank k must be >= 1 or < min(m, n), not %d" % k)

    #Pivoted QR decomposition
    Q, R, P = linalg.qr(A, mode='economic', overwrite_a=False, pivoting=True,
                        check_finite=False)

    # Select column subset
    C = A[:, P[:k]]

    # Compute V
    T =  linalg.pinv2(R[:k, :k]).dot(R[:k, k:n])
    V = np.bmat([[np.eye(k), T]])
    V = V[:, np.argsort(P)]

    # Return ID
    if mode == 'column':
        if index_set:
            return P[:k], V
        return C, V
    # mode == row
    elif index_set:
        return conjugate_transpose(V), P[:k]

    return conjugate_transpose(V), conjugate_transpose(C)


def rinterp_decomp(A, k=None, mode='column', p=10, q=1, sdist='normal',
                   index_set=False, random_state=None):
    """Randomized interpolative decomposition (rID).

    Algorithm for computing the approximate low-rank ID
    decomposition of a rectangular `(m, n)` matrix `A`, with target rank `k << min{m, n}`.
    The input matrix is factored as `A = C * V`. The factor matrix `C`is formed
    of a subset of columns of `A`, also called the partial column skeleton.
    The factor matrix `V`contains a `(k, k)` identity matrix as a submatrix,
    and is well-conditioned.

    If `mode='row'`, then the input matrix is factored as `A = Z * R`, using the
    row pivoted QR decomposition. The factor matrix `R` is now formed as
    a subset of rows of `A`, also called the partial row skeleton.

    The quality of the approximation can be controlled via the oversampling
    parameter `p` and the parameter `q` which specifies the number of
    subspace iterations.


    Parameters
    ----------
    A : array_like, shape `(m, n)`.
        Input matrix.

    k : integer, `k << min{m,n}`.
        Target rank.

    mode: str `{'column', 'row'}`, default: `mode='column'`.
        'column' : Column ID.
        'row' : Row ID.

    p : integer, default: `p=10`.
        Parameter to control oversampling.

    q : integer, default: `q=1`.
        Parameter to control number of power (subspace) iterations.

    sdist : str `{'uniform', 'normal'}`, default: `sdist='uniform'`.
        'uniform' : Random test matrix with uniform distributed elements.

        'normal' : Random test matrix with normal distributed elements.

    index_set: str `{'True', 'False'}`, default: `index_set='False'`.
        'True' : Return column/row index set instead of `C` or `R`.

    random_state : integer, RandomState instance or None, optional (default ``None``)
        If integer, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used by np.random.

    Returns
    -------
    If `mode='column'`:
        C:  array_like, shape `(m, k)`.
            Partial column skeleton.

        V : array_like, shape `(k, n)`.
            Well-conditioned matrix.

    If `mode='row'`:
        Z:  array_like, shape `(m, k)`.
            Well-conditioned matrix.

        R : array_like, shape `(k, n)`.
            Partial row skeleton.

    References
    ----------
    S. Voronin and P.Martinsson.
    "RSVDPACK: Subroutines for computing partial singular value
    decompositions via randomized sampling on single core, multi core,
    and GPU architectures" (2015).
    (available at `arXiv <http://arxiv.org/abs/1502.05366>`_).
    """
    random_state=check_random_state(random_state)

    # converts A to array, raise ValueError if A has inf or nan
    A = np.asarray_chkfinite(A)

    if mode=='row':
        A = conjugate_transpose(A)

    if A.dtype not in _VALID_DTYPES:
        raise ValueError('A.dtype must be one of %s, not %s'
                         % (' '.join(_VALID_DTYPES), A.dtype))

    if mode not in _VALID_MODES:
        raise ValueError('mode must be one of %s, not %s'
                         % (' '.join(_VALID_MODES), mode))

    if sdist not in _VALID_SDISTS:
        raise ValueError('sdists must be one of %s, not %s'
                         % (' '.join(_VALID_SDISTS), sdist))

    m, n = A.shape
    if k is None:
        # default
        k = min(m, n)

    if k < 1 or k > min(m, n):
        raise ValueError("Target rank k must be >= 1 or < min(m, n), not %d" % k)

    # distribution to draw random samples
    sdist_func = _get_distribution_func(sdist, random_state)

    #Generate a random test matrix Omega
    Omega = sdist_func(size=(k+p, m)).astype(A.dtype)

    if A.dtype == np.complexfloating:
        real_type = np.float32 if A.dtype == np.complex64 else np.float64
        Omega += 1j * sdist_func(size=(k+p, m)).astype(real_type)

    #Build sample matrix Y : Y = A * Omega (Y approximates range of A)
    Y = Omega.dot(A)
    del Omega

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Orthogonalize Y using economic QR decomposition: Y=QR
    #If q > 0 perfrom q subspace iterations
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    for _ in range(q):
        Y, _ = linalg.qr(conjugate_transpose(Y), mode='economic',
        check_finite=False, overwrite_a=True)
        Z, _ = linalg.qr(A.dot(Y), mode='economic', check_finite=False, overwrite_a=True)
        Y = conjugate_transpose(Z).dot(A)

    del Z

    # Deterministic ID
    J, V = interp_decomp(Y, k=k, mode='column', index_set=True)
    J = J[0:k]

    if mode == 'column':
        if index_set:
            return J, V
        return A[:,J], V
    # mode == 'row'
    elif index_set:
        return conjugate_transpose(V), J
    return conjugate_transpose(V), conjugate_transpose(A[:,J])


def rinterp_decomp_qb(A, k=None, mode='column', p=10, q=1, sdist='normal',
                      index_set=False, random_state=None):
    r"""Randomized interpolative decomposition (rID).

    Algorithm for computing the approximate low-rank ID
    decomposition of a rectangular `(m, n)` matrix `A`, with target rank `k << min{m, n}`.
    The input matrix is factored as `A = C * V`. The factor matrix $\mathbf{C}$ is formed
    of a subset of columns of $\mathbf{A}$, also called the partial column skeleton.
    The factor matrix $\mathbf{V}$ contains a $k\times k$ identity matrix as a submatrix,
    and is well-conditioned.

    If `mode='row'`, then the input matrix is factored as `A = Z * R`, using the
    row pivoted QR decomposition. The factor matrix $\mathbf{C}$ is now formed as
    a subset of rows of $\mathbf{A}$, also called the partial row skeleton.

    The quality of the approximation can be controlled via the oversampling
    parameter `p` and the parameter `q` which specifies the number of
    subspace iterations.


    Parameters
    ----------
    A : array_like, shape `(m, n)`.
        Input matrix.

    k : integer, `k << min{m,n}`.
        Target rank.

    mode: str `{'column', 'row'}`, default: `mode='column'`.
        'column' : Column ID.
        'row' : Row ID.

    p : integer, default: `p=10`.
        Parameter to control oversampling.

    q : integer, default: `q=1`.
        Parameter to control number of power (subspace) iterations.

    sdist : str `{'uniform', 'normal'}`, default: `sdist='uniform'`.
        'uniform' : Random test matrix with uniform distributed elements.

        'normal' : Random test matrix with normal distributed elements.

    index_set: str `{'True', 'False'}`, default: `index_set='False'`.
        'True' : Return column/row index set.

    random_state : integer, RandomState instance or None, optional (default ``None``)
        If integer, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used by np.random.


    Returns
    -------
    If `mode='column'`:
        C:  array_like, shape `(m, k)`.
            Partial column skeleton.

        V : array_like, shape `(k, n)`.
            Well-conditioned matrix.

    If `mode='row'`:
        Z:  array_like, shape `(m, k)`.
            Well-conditioned matrix.

        R : array_like, shape `(k, n)`.
            Partial row skeleton.


    J : array_like, shape `(k, n)`.
        Column/row index set.


    References
    ----------
    S. Voronin and P.Martinsson.
    "RSVDPACK: Subroutines for computing partial singular value
    decompositions via randomized sampling on single core, multi core,
    and GPU architectures" (2015).
    (available at `arXiv <http://arxiv.org/abs/1502.05366>`_).
    """
    if mode not in _VALID_MODES:
        raise ValueError('mode must be one of %s, not %s'
                         % (' '.join(_VALID_MODES), mode))

    # converts A to array, raise ValueError if A has inf or nan
    A = np.asarray_chkfinite(A)
    if mode=='row':
        A = conjugate_transpose(A)

    # compute QB factorization
    Q, B = rqb(A, k=k, p=p, q=q, sdist=sdist, random_state=random_state)

    # Deterministic ID
    J, V = interp_decomp(B, k=k, mode='column', index_set=True)
    J = J[:k]

    # Return ID
    if mode=='column':
        if index_set:
            return J, V
        return A[:,J], V
    # mode == 'row'
    elif index_set:
        return conjugate_transpose(V), J
    return conjugate_transpose(V), conjugate_transpose(A[:,J])
