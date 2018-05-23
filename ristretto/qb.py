"""
Randomized QB Decomposition
"""
# Authors: N. Benjamin Erichson
#          Joseph Knox
# License: GNU General Public License v3.0

# NOTE: we should depricate single_pass because:
#       we get the same performance by perfoming a single subspace
#       iteration, the single pass construct doesn't really help us
#       here unless we write it in C/Cython and compute the products
#              Y = A * Omega, Y_tilde = A * Omega_tilde 
#       in sinlge passes

import numpy as np
from scipy import linalg

from .sketch.transforms import johnson_lindenstrauss
from .utils import conjugate_transpose


def rqb(A, k=None, p=10, l=None, q=1, sdist='normal', single_pass=False,
        random_state=None):
    """Randomized QB Decomposition.

    Randomized algorithm for computing the approximate low-rank QB
    decomposition of a rectangular `(m, n)` matrix `A`, with target rank `k << min{m, n}`.
    The input matrix is factored as `A = Q * B`.

    The quality of the approximation can be controlled via the oversampling
    parameter `p` and the parameter `q` which specifies the number of
    subspace iterations.


    Parameters
    ----------
    A : array_like, shape `(m, n)`.
        Real nonnegative input matrix.

    k : integer, `k << min{m,n}`.
        Target rank.

    p : integer, default: `p=10`.
        Parameter to control oversampling of column space.

    l : integer, default: `l=2*p`.
        Parameter to control oversampling of row space. Only relevant if
        single_pass == True.

    q : integer, default: `q=1`.
        Parameter to control number of power (subspace) iterations. Only
        relevant if single_pass == False.

    sdist : str `{'uniform', 'normal'}`, default: `sdist='uniform'`.
        'uniform' : Random test matrix with uniform distributed elements.

        'normal' : Random test matrix with normal distributed elements.

    single_pass : bool
        If single_pass == True, perfom single pass of algorithm, meaning that
        in the algorithm only accesses A directly a single time. Beneficial A
        is large.

    random_state : integer, RandomState instance or None, optional (default ``None``)
        If integer, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used by np.random.


    Returns
    -------
    Q:  array_like, shape `(m, k+p)`.
        Orthonormal basis matrix.

    B : array_like, shape `(k+p, n)`.
        Smaller matrix.


    References
    ----------
    N. Halko, P. Martinsson, and J. Tropp.
    "Finding structure with randomness: probabilistic
    algorithms for constructing approximate matrix
    decompositions" (2009).
    (available at `arXiv <http://arxiv.org/abs/0909.4061>`_).

    S. Voronin and P.Martinsson.
    "RSVDPACK: Subroutines for computing partial singular value
    decompositions via randomized sampling on single core, multi core,
    and GPU architectures" (2015).
    (available at `arXiv <http://arxiv.org/abs/1502.05366>`_).
    """
    # get random sketch
    if single_pass:
        # NOTE: we get the same performance by perfoming a single subspace
        #       iteration, the single pass construct doesn't really help us
        #       here unless we write it in C/Cython and compute the products
        #              Y = A * Omega, Y_tilde = A * Omega_tilde 
        #       in sinlge passes
        q = 1

    Q = johnson_lindenstrauss(A, k + p, n_subspace=q, random_state=random_state)

    #Project the data matrix a into a lower dimensional subspace
    B = conjugate_transpose(Q).dot(A)

    return Q, B
