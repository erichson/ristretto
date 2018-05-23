"""
Functions for approximating the range of a matrix A.
"""
from __future__ import division
from math import log

import numpy as np
from scipy import fftpack
from scipy import sparse

from . import _sketches
from .utils import perform_subspace_iterations
from ..utils import check_random_state, safe_sparse_dot


def randomized_uniform_sampling(A, l, axis=1, random_state=None):
    """Uniform randomized sampling transform.

    Given an m x n matrix A, and an integer l, this returns an m x l
    random subset of the range of A.

    """
    random_state = check_random_state(random_state)
    A = np.asarray(A)

    # sample l rows/columns with equal probability
    idx = _sketches.random_axis_sample(A, l, axis, random_state)

    return np.take(A, idx, axis=axis)


def johnson_lindenstrauss(A, l, n_subspace=None, axis=1, random_state=None):
    """

    Given an m x n matrix A, and an integer l, this scheme computes an m x l
    orthonormal matrix Q whose range approximates the range of A

    """
    random_state = check_random_state(random_state)

    A = np.asarray(A)
    if A.ndim != 2:
        raise ValueError('A must be a 2D array, not %dD' % A.ndim)

    if axis not in (0, 1):
        raise ValueError('If supplied, axis must be in (0, 1)')

    # construct gaussian random matrix
    Omega = _sketches.random_gaussian_map(A, l, axis, random_state)

    # project A onto Omega
    if axis == 0:
        Q = Omega.T.dot(A)
    else:
        Q = A.dot(Omega)

    if n_subspace is not None:
        Q = perform_subspace_iterations(A, Q, n_iter=n_subspace, axis=axis)

    return Q


def sparse_johnson_lindenstrauss(A, l, n_subspace=1, density=None, axis=1,
                                 random_state=None):
    """

    Given an m x n matrix A, and an integer l, this scheme computes an m x l
    orthonormal matrix Q whose range approximates the range of A

    Parameters
    ----------
    density : sparse matrix density

    """
    random_state = check_random_state(random_state)

    A = np.asarray(A)
    if A.ndim != 2:
        raise ValueError('A must be a 2D array, not %dD' % A.ndim)

    if axis not in (0, 1):
        raise ValueError('If supplied, axis must be in (0, 1)')

    if density is None:
        density = A.shape[0] / log(A.shape[0])

    # construct sparse sketch
    Omega = _sketches.sparse_random_map(A, l, axis, density, random_state)

    # project A onto Omega
    if axis == 0:
        Q = safe_sparse_dot(Omega.T, A)
    else:
        Q = safe_sparse_dot(A, Omega)

    if n_subspace is not None:
        Q = perform_subspace_iterations(A, Q, n_iter=n_subspace, axis=axis)

    return Q

def fast_johnson_lindenstrauss(A, l, axis=1, random_state=None):
    """

    Given an m x n matrix A, and an integer l, this scheme computes an m x l
    orthonormal matrix Q whose range approximates the range of A

    """
    random_state = check_random_state(random_state)

    A = np.asarray_chkfinite(A)
    if A.ndim != 2:
        raise ValueError('A must be a 2D array, not %dD' % A.ndim)

    if axis not in (0, 1):
        raise ValueError('If supplied, axis must be in (0, 1)')

    # TODO: Find name for sketch and put in _sketches
    # construct gaussian random matrix
    diag = random_state.choice((-1, 1), size=A.shape[axis]).astype(A.dtype)

    if axis == 0:
        diag = diag[:, np.newaxis]
    
    # discrete fourier transform of AD (or DA)
    FDA = fftpack.dct(A * diag, axis=axis, norm='ortho')

    # randomly sample axis
    return randomized_uniform_sampling(
        FDA, l, axis=axis, random_state=random_state)
