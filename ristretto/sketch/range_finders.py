"""
Functions for approximating the range of a matrix A.
"""
from functools import reduce
import operator as op

import numpy as np
from scipy import linalg
from scipy import fftpack
from scipy import stats
from scipy import sparse

from ..utils import check_random_state


def _axis_wrapper(func):
    """Wraps range finders, dealing with axis keyword arguments.

    If axis == 1: return function
    If axis == 0: transpose A, return transpose of results
    else: raise error
    """
    def check_axis_and_call(*args, **kwargs):
        axis = kwargs.get('axis', 1)
        if axis not in (0, 1):
            raise ValueError('If supplied, axis must be in (0, 1)')
        if axis == 0:
            A, *args = args
            result = func(A, *args, **kwargs)
            if isinstance(result, tuple):
                return tuple(map(np.transpose, result))
            return result.T
        return func(*args, **kwargs)
    return check_axis_and_call


def _ortho(X):
    """orthonormalize the columns of X via QR decomposition"""
    Q, _ = linalg.qr(X, overwrite_a=True, mode='economic', pivoting=False,
                     check_finite=False)
    return Q


@_axis_wrapper
def randomized_uniform_sampling(A, l, axis=1, random_state=None):
    """Uniform randomized sampling transform.

    Given an m x n matrix A, and an integer l, this returns an m x l
    random subset of the range of A.

    """
    random_state = check_random_state(random_state)

    A = np.asarray_chfinite(A)
    m, n = A.shape

    # sample l columns with equal probability
    idx = random_state.choice(n, size=l, replace=False)
    return A[:, idx]


@_axis_wrapper
def randomized_basic(A, l, axis=1, random_state=None):
    """

    Given an m x n matrix A, and an integer l, this scheme computes an m x l
    orthonormail matrix Q whose range approximates the range of A

    Notes
    -----
    Also known as randQB
    """
    random_state = check_random_state(random_state)

    A = np.asarray_chfinite(A)
    m, n = A.shape

    # construct gaussian random matrix
    Omega = random_state.standard_normal(size=(n, l)).astype(A.dtype)

    # project A onto Omega
    Y = A.dot(Omega)

    # orthonormalize Y
    Q = _ortho(Y)
    B = Q.T.dot(A)

    return Q, B


@_axis_wrapper
def randomized_blocked_adaptive(A, r=10, tol=1e-3, axis=1, random_state=None):
    """

    Given an m x n matrix A, a tolerance tol, and an iteger r,
    adaptive_randomized_range_finder computes an orthonormal matrix Q such
    that `` | I - Q Q^*) A | <= tol `` holds with at least probability
    ``1 - min(m, n)*10^-r``.

    Notes
    -----
    Also known as randQB_b
    """
    random_state = check_random_state(random_state)

    A = np.asarray_chfinite(A)
    m, n = A.shape

    Q_iters, QQT_iters, B_iters = [], [], []
    for _ in range(r):
        # construct gaussian random matrix
        Omega = random_state.standard_normal(size=(n, r)).astype(A.dtype)

        Q = _ortho(A.dot(Omega))

        if Q_iters:
            # Qi = orth(Qi - sum(Qj Qj^T Qi))
            Q -= reduce(op.add, map(lambda x: x.dot(Q), QQT_iters))

        Q = _ortho(Q)

        # compute QB decomposition
        B = Q.T.dot(A)

        # break if we reach desired rank
        if linalg.norm(A - Q.dot(B)) < tol:
            break

        # update
        Q_iters.append(Q)
        B_iters.append(B)
        QQT_iters.append(Q.dot(Q.T))

    return np.hstack(Q_iters), np.vstack(B_iters)


@_axis_wrapper
def randomized_subspace_iteration(A, l, n_iter=10, axis=1, random_state=None):
    """

    Given an m x n matrix A, and an integer l, this scheme computes an m x l
    orthonormail matrix Q whose range approximates the range of A.

    """
    random_state = check_random_state(random_state)

    A = np.asarray_chfinite(A)
    m, n = A.shape

    # construct gaussian random matrix
    Omega = random_state.standard_normal(size=(n, l)).astype(A)

    # project A onto Omega
    Y = A.dot(Omega)

    Q = _ortho(Y)
    for _ in range(n_iter):
        Z = _ortho(A.T.dot(Q))
        Q = _ortho(A.dot(Z))

    B = Q.T.dot(A)

    return Q, B


# TODO: need to verify
#def fast_johnson_lindenstrauss_transform(A, l, random_state=None):
#    """Fast Johnson-Lindenstrauss Transform.
#
#    Given an m x n matrix A, and an integer l, this scheme computes an m x l
#    orthonormail matrix Q whose range approximates the range of A
#
#    """
#    m, n = A.shape
#
#    d = random_state.choice((-1, 1), size=l)
#    d = sparse.spdiags(d, 0, n, n)
#
#    # project A onto d, compute DCT
#    Ad = A.dot(d)
#    Adf = fftpack.dct(Ad, axis=1, norm='ortho')
#
#    # uniformly sample
#    Q = randomized_uniform_sampling_transform(Adf, l)
