# Authors: Ben Erichson erichson@uw.edu
#          Joseph Knox josephk@alleninstitute.org
# License:
from __future__ import print_function
import time
import numpy as np
from scipy.linalg import qr

from ristretto.random_qr import randomized_qr_sample

def deterministic_qr_subset(arr, **kwargs):
    # qr
    Q, R, P = qr( arr.T,  mode='economic', overwrite_a=False, pivoting=True)

    # Select column subset
    k = kwargs["target_rank"]
    return P[:k]

if __name__ == "__main__":

    # keyword arguments for random_qr_sample
    qr_kwargs = {
        "target_rank" : 10,
        "oversample_size" : 50,
        "n_power_iterations" : 5,
        "n_blocks" : 20
    }

    # for repeatablility
    np.random.seed(123)

    # initialize intricmodel from 'data'
    m, n, k = 2000, 3500, 10
    arr = np.dot(np.random.rand(m,k), np.random.rand(k,n))

    print("\nrandom algorithm")

    # start timer
    start_time = time.time()

    # compute column sample
    columns = randomized_qr_sample( arr, **qr_kwargs)

    # print output
    print("columns    :", columns)
    print("time taken : {:.3f} s".format(time.time() - start_time) )

    check = 1
    if check:
        print("\ndeterministic sanity check")

        # start timer
        start_time = time.time()

        # Select column subset
        columns = deterministic_qr_subset( arr, **qr_kwargs )

        # print output
        print("columns    :", columns)
        print("time taken : {:.3f} s".format( time.time() - start_time) )
