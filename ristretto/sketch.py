"""
Module containing random sketch generating object.
"""
# Authors: N. Benjamin Erichson
#          Joseph Knox
# License: GNU General Public License v3.0
from functools import partial
import warnings

import numpy as np
from scipy import linalg

from .utils import conjugate_transpose

_VALID_DISTRIBUTIONS = ('uniform', 'normal', None)
_VALID_SINGLE_DISTRIBUTIONS = ('uniform', 'normal', 'orthogonal')
_VALID_DTYPES = (np.float32, np.float64, np.complex64, np.complex128)
_QR_KWARGS = dict(mode='economic', check_finite=False, overwrite_a=True)

def _output_rank_check(A, output_rank):
    n, m = A.shape
    rank = min(n, m) if output_rank is None else output_rank

    if rank > min(n, m):
        warnings.warn('output_rank %d is greater than the minimum '
                      'dimension of input array A (shape %s). The '
                      'minimum dimension will be chosen instead',
                      output_rank, n, m, min(n, m))
        rank = min(n, m)
    return rank

def _get_distribution_func(distribution):
    """Helper function returning numpy sampling distribution function"""
    if distribution is None:
        def wrapped_rand(size=None):
            # have to wrap np.random.rand to take size kwarg
            return np.random.rand(*size)
        return wrapped_rand
    elif distribution == 'uniform':
        return partial(np.random.uniform, -1, 1)
    return np.random.standard_normal


def sketch(A, out=None, output_rank=None, n_oversample=10, n_iter=2,
           distribution=None, axis=0, check_finite=False):
    # converts A to array, raise ValueError if A has inf or nan
    A = np.asarray_chkfinite(A) if check_finite else np.asarray(A)

    if A.dtype not in _VALID_DTYPES:
        raise ValueError('A.dtype must be one of %s, not %s'
                         % (' '.join(_VALID_DTYPES), A.dtype))

    if distribution not in _VALID_DISTRIBUTIONS:
        raise ValueError('distribution must be one of %s, not %s'
                         % (' '.join(_VALID_DISTRIBUTIONS), distribution))

    if axis not in (0, 1):
        raise ValueError('If specified, axis must be 0 or 1, not %s' % axis)

    # check rank
    rank = _output_rank_check(A, output_rank)

    n_oversample += rank
    size = (n_oversample, A.shape[0]) if axis == 0 else (A.shape[1], n_oversample)

    # get numpy random func
    dist_func = _get_distribution_func(distribution)

    #Generate a random test matrix Omega
    Omega = dist_func(size=size).astype(A.dtype)

    if A.dtype == np.complexfloating:
        real_type = np.float32 if A.dtype == np.complex64 else np.float64
        Omega += 1j * dist_func(size=size).astype(real_type)

    #Build sample matrix Y : Y = A * Omega
    # Y approximates range of A
    Y = A.dot(Omega)
    del Omega

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Orthogonalize Y using economic QR decomposition: Y=QR
    #If q > 0 perfrom q subspace iterations
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    for _ in range(n_iter):
        Y, _ = linalg.qr(Y, **_QR_KWARGS)
        Z, _ = linalg.qr(conjugate_transpose(A).dot(Y), **_QR_KWARGS)
        Y = A.dot(Z)

    # compute sketch
    S, _ = linalg.qr(Y, **_QR_KWARGS)

    #if axis == 0:
    #    return np.dot(conjugate_transpose(S), A, out=out)
    #return np.dot(A, S, out=out)
    return S


def single_pass_sketch(A, output_rank=None, row_oversample=None,
                       column_oversample=10, distribution='uniform',
                       check_finite=False):
    # converts A to array, raise ValueError if A has inf or nan
    A = np.asarray_chkfinite(A) if check_finite else np.asarray(A)

    if A.dtype not in _VALID_DTYPES:
        raise ValueError('A.dtype must be one of %s, not %s'
                         % (' '.join(_VALID_DTYPES), A.dtype))

    if distribution not in _VALID_SINGLE_DISTRIBUTIONS:
        raise ValueError('distribution must be one of %s, not %s'
                         % (' '.join(_VALID_SINGLE_DISTRIBUTIONS), distribution))

    if row_oversample is None:
        row_oversample = 2 * column_oversample

    # check rank
    rank = _output_rank_check(A, output_rank)
    row_oversample += rank
    column_oversample += rank

    # get numpy random func
    dist_func = _get_distribution_func(distribution)

    #Generate a random test matrix Omega
    Omega = dist_func(size=(A.shape[1], column_oversample)).astype(A.dtype)
    Psi = dist_func(size=(row_oversample, A.shape[0])).astype(A.dtype)

    if A.dtype == np.complexfloating:
        real_type = np.float32 if A.dtype == np.complex64 else np.float64
        Omega += 1j * dist_func(size=(A.shape[1], column_oversample)).astype(real_type)
        Psi += 1j * dist_func(size=(row_oversample, A.shape[0])).astype(real_type)

    if distribution == 'orthogonal':
        Omega, _ = linalg.qr(Omega, **_QR_KWARGS)
        Psi , _ = linalg.qr(conjugate_transpose(Psi), **_QR_KWARGS)
        Psi = conjugate_transpose(Psi)

    #Build sample matrix Y = A * Omega and W = Psi * A
    #Note: Y should approximate the column space and W the row space of A
    #Y = A.dot(Omega)
    #W = Psi.dot(A)
    #del Omega

    #Orthogonalize Y using economic QR decomposition: Y=QR
    #Q, _ = linalg.qr(Y, **_QR_KWARGS)
    #U, T = linalg.qr(Psi.dot(Q), **_QR_KWARGS)

    # Form a smaller matrix
    #return linalg.solve(T, conjugate_transpose(U).dot(W))
    return Omega, Psi
