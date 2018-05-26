"""
Module containing sketching funcitons.
"""
from functools import partial
from math import sqrt

from scipy import sparse


def random_axis_sample(A, l, axis, random_state):
    """randomly sample the index of axis"""
    return random_state.choice(A.shape[axis], size=l, replace=False)


def random_gaussian_map(A, l, axis, random_state):
    """generate random gaussian map"""
    return random_state.standard_normal(size=(A.shape[axis], l)).astype(A.dtype)


def random_uniform_map(A, l, axis, random_state):
    """generate random uniform map"""
    return random_state.uniform(-1, 1, size=(A.shape[axis], l)).astype(A.dtype)


def sparse_random_map(A, l, axis, density, random_state):
    """generate sparse random sampling"""
    # TODO: evaluete random_state paramter: we want to pass it to sparse.random
    #       most definitely to sparsely sample the nnz elements of Omega, but
    #       is using random_state in data_rvs redundant?
    values = (-sqrt(1. / density), sqrt(1. / density))
    data_rvs = partial(random_state.choice, values)

    return sparse.random(A.shape[axis], l, density=density, data_rvs=data_rvs,
                         random_state=random_state, dtype=A.dtype)
