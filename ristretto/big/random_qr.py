# Authors: Ben Erichson erichson@uw.edu
#          Joseph Knox josephk@alleninstitute.org
# License:
import numpy as np
from scipy.linalg import qr
from numpy.random import standard_normal

def _random_sketch(A, Omega, out, n_blocks):
    """Constructs sketch of A as the sum of randomly weighted columns of A.

    More

    Parameters
    ---------
    """
    # split indices into n_blocks
    idx_sets = np.array_split( np.arange(A.shape[1], dtype=np.int), n_blocks)
    for idx in idx_sets:

        # inplace
        np.add( A[:,idx].dot(Omega[idx,:]), out, out)

def _power_iterations(A, Y, Z, n_iterations, n_blocks):
    """Performs power iterations"""

    # index sets
    row_sets = np.array_split( np.arange(A.shape[0]), n_blocks )
    col_sets = np.array_split( np.arange(A.shape[1]), n_blocks )

    for i in range(n_iterations):

        # qr of sketch
        Y, _ = qr(Y, mode='economic', check_finite=False, overwrite_a=True)

        # project ...
        Z = Z.T
        for idx in col_sets:
            Z[:, idx] = Y.T.dot(A[:,idx])

        # qr of projection
        Z, _ = qr(Z.T, mode='economic', check_finite=False, overwrite_a=True)

        # project back
        for idx in row_sets:
            Y[idx, :] = A[idx,:].dot(Z)

    return Y

def randomized_qr_sample( A, target_rank=1, oversample_size=0,
                          n_power_iterations=0, n_blocks=1):
    # for convienence
    m, n = A.shape

    # random matrix ..
    Omega = standard_normal( (n, target_rank + oversample_size) )

    # construct random sketch
    Y = np.zeros( (m, target_rank + oversample_size), dtype=np.float32 )
    _random_sketch(A, Omega, Y, n_blocks)

    del(Omega)

    # perform power iterations for stability
    Z = np.empty( (n, target_rank + oversample_size), dtype=np.float32 )
    Y = _power_iterations(A, Y, Z, n_power_iterations, n_blocks)

    del(Z)

    # Now compute pivoted QR of Y.T
    Q, R, P = qr(Y.T , mode='economic', overwrite_a=True, pivoting=True)

    # Select column subset
    return P[:target_rank]
