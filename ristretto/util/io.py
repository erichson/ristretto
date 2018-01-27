# Authors: Ben Erichson erichson@uw.edu
#          Joseph Knox josephk@alleninstitute.org
# License:

import h5py

def use_hdf5(filepath, dataset, libver="latest"):
    try:
        extension = filepath.split(".")[-1]
        if extension != "hdf5":
            raise ValueError("Must provide filepath with .hdf5 extension")
    except AttributeError:
        raise ValueError("filepath must be a string!!")

    f = h5py.File(filepath, mode="r", libver=libver)

    if dataset in f.keys():
        return f.get(dataset)
    else:
        raise ValueError("dataset not in file!")
