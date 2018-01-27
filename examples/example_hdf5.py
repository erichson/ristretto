# Authors: Ben Erichson erichson@uw.edu
#          Joseph Knox josephk@alleninstitute.org
# License:
from __future__ import print_function
import os
import time
import h5py
import numpy as np
from scipy.linalg import qr

from ristretto.util.io import use_hdf5
from ristretto.random_qr import randomized_qr_sample

def write_hdf5(filepath, dataset, arr):
    f = h5py.File(filepath, mode="w")
    f.create_dataset(dataset, data=arr)
    f.close()

if __name__ == "__main__":

    # names
    filepath = "data/random_data.hdf5"
    dataset = "random_data"

    # keyword arguments for random_qr_sample
    qr_kwargs = {
        "target_rank" : 10,
        "oversample_size" : 50,
        "n_power_iterations" : 5,
        "n_blocks" : 20
    }

    # generate fake data, write to file
    m, n = 2000, 4000
    arr = np.random.rand(m,n)

    if not os.path.exists(filepath):
        # write to disk
        write_hdf5(filepath, dataset, arr)
        del(arr)

    # 'load' data
    arr = use_hdf5(filepath, dataset)

    # start timer
    start_time = time.time()

    # compute column sample
    columns = randomized_qr_sample( arr, **qr_kwargs)

    # print output
    print("columns    :", columns)
    print("time taken : {:.3f} s".format(time.time() - start_time) )
