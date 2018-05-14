"""
Randomized QB Decomposition
"""
# Authors: N. Benjamin Erichson
#          Joseph Knox
# License: GNU General Public License v3.0

import numpy as np
from scipy import linalg

from .sketch import sketch, single_pass_sketch
from .utils import conjugate_transpose


def rqb(A, k=None, p=10, q=1, sdist='normal', random_state=None):
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
        Parameter to control oversampling.

    q : integer, default: `q=1`.
        Parameter to control number of power (subspace) iterations.

    sdist : str `{'uniform', 'normal'}`, default: `sdist='uniform'`.
        'uniform' : Random test matrix with uniform distributed elements.

        'normal' : Random test matrix with normal distributed elements.

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
    Q = sketch(A, output_rank=k, n_oversample=p, n_iter=q, distribution=sdist,
               axis=1, check_finite=True, random_state=random_state)

    #Project the data matrix a into a lower dimensional subspace
    B = conjugate_transpose(Q).dot(A)

    return Q, B


def rqb_single(A, k=None, p=10, l=None, q=1, sdist='normal', random_state=None):
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
        Parameter to control oversampling.

    q : integer, default: `q=1`.
        Parameter to control number of power (subspace) iterations.

    sdist : str `{'uniform', 'normal'}`, default: `sdist='uniform'`.
        'uniform' : Random test matrix with uniform distributed elements.

        'normal' : Random test matrix with normal distributed elements.

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
    # Form a smaller matrix
    Omega, Psi = single_pass_sketch(
        A, output_rank=k, column_oversample=p, row_oversample=l,
        distribution=sdist, check_finite=True, random_state=random_state)

    #Build sample matrix Y = A * Omega and W = Psi * A
    #Note: Y should approximate the column space and W the row space of A
    Y = A.dot(Omega)
    W = Psi.dot(A)
    del Omega

    #Orthogonalize Y using economic QR decomposition: Y=QR
    Q, _ = linalg.qr(Y, mode='economic', check_finite=False, overwrite_a=True )
    U, T = linalg.qr(Psi.dot(Q), mode='economic', check_finite=False, overwrite_a=False )

    # Form a smaller matrix
    B = linalg.solve(T, conjugate_transpose(U).dot(W), check_finite=False,
                     overwrite_a=True, overwrite_b=True)

    return Q, B
