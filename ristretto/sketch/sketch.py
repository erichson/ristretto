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

from .utils import check_random_state, conjugate_transpose

_VALID_DISTRIBUTIONS = ('uniform', 'normal')
_VALID_SINGLE_DISTRIBUTIONS = ('uniform', 'normal', 'orthogonal')
_VALID_DTYPES = (np.float32, np.float64, np.complex64, np.complex128)
_QR_KWARGS = dict(mode='economic', check_finite=False, overwrite_a=True)

def sketch(A, method='randomized_subspace_iteration', *args, **kwargs):
    try:
        func = getattr(range_finders, method)
    except AttributeError:
        # TODO: write better error message
        raise ValueError('incorrect method %s passed' )

    return func(*args, **kwargs)


def single_pass_sketch(A, output_rank=None, row_oversample=None,
                       column_oversample=10, distribution='uniform',
                       check_finite=False, random_state=None):
    random_state = check_random_state(random_state)

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
    dist_func = _get_distribution_func(distribution, random_state)

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
