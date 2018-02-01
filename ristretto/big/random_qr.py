# Authors: Ben Erichson erichson@uw.edu
#          Joseph Knox josephk@alleninstitute.org
# License:

# NOTE : linalg.qr returns FORTRAN contiguous arrays!!!

from functools import partial
from scipy import linalg
import numpy as np

def _random_sketch(A, extended_rank, n_blocks=0):
    """Constructs sketch of A as the sum of randomly weighted columns of A.

    More

    Parameters
    ---------
    A

    extended_rank

    n_blocks

    Returns
    -------
    """
    # random matrix ..
    Omega = np.random.standard_normal( (A.shape[1], extended_rank) )
    Omega = Omega.astype( A.dtype )

    # initialize result array
    out = np.zeros( (A.shape[0], extended_rank), dtype=A.dtype )

    if n_blocks:
        # do the computation blocked
        idx_sets = np.array_split( np.arange(A.shape[1]), n_blocks)
        for idx in idx_sets:

            # inplace addition
            np.add( A[:,idx].dot(Omega[idx,:]), out, out)
    else:
        # overwrite out
        np.dot( A, Omega, out)

    return out

def _blocked_power_iters(qr_func, A, Y, Z, n_iters, n_blocks):
    """Performs power iterations in blocks.

    see _power_iterations for more
    """
    # index sets
    row_sets, col_sets = map(lambda x : np.array_split(np.arange(x),
                                                       n_blocks), A.shape)
    for _ in range(n_iters):

        # qr of sketch
        Y, _ = qr_func(Y)

        # project ...
        Z = Z.T
        for idx in col_sets:
            Z[:, idx] = Y.T.dot(A[:,idx])

        # qr of projection
        Z, _ = qr_func(Z.T)

        # project back
        for idx in row_sets:
            Y[idx, :] = A[idx,:].dot(Z)

    return Y

def _power_iters(qr_func, A, Y, Z, n_iters):
    """Performs power iterations.

    see _power_iterations for more
    """
    for _ in range(n_iters):

        # qr of sketch
        Y, _ = qr_func(Y)
        Z = np.dot(A.T, Y)

        # qr of projection
        Z, _ = qr_func(Z)
        Y = np.dot(A, Z)

    return Y

def _power_iterations(A, Y, n_iters=0, n_blocks=0):
    """Performs power iterations on Y.

    ...

    Parameters
    ----------
    A

    Y

    n_iters

    n_blocks

    Returns
    -------
    C
    """


    # QR
    qr_func = partial(linalg.qr, mode="economic", check_finite=False,
                      overwrite_a=True)

    # initialize array ...
    Z = np.empty( (A.shape[1], Y.shape[1]), dtype=A.dtype )

    if n_blocks:
        return _blocked_power_iters(qr_func, A, Y, Z, n_iters, n_blocks)

    return _power_iters(qr_func, A, Y, Z, n_iters)


def randomized_qr_sample( A, target_rank=1, oversample=0, n_iters=0,
                          n_blocks=0):
    """Generate column subset using randomized QR factorization.

    ...

    Parameters
    ----------
    A

    target_rank

    oversample_size

    n_iters

    n_blocks

    Returns
    -------
    C

    Examples
    --------
    """
    if n_blocks > min(A.shape):
        raise ValueError( "n_blocks must be less than the smallest "
                          "dimension of A ({})".format(min(A.shape)) )

    # extended rank :: rank of spaces we compute to give us a better estimate
    extended_rank = target_rank + oversample

    # construct random sketch
    Y = _random_sketch(A, extended_rank, n_blocks)

    # perform power iterations for stability
    if n_iters:
        Y = _power_iterations(A, Y, n_iters, n_blocks)

    # Now compute pivoted QR of Y.T
    Q, R, P = linalg.qr(Y.T , mode='economic', overwrite_a=True, pivoting=True)

    # Select column subset
    return P[:target_rank]
