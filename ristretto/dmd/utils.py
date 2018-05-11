"""
Utility functions for :mod:`dmd` module.
"""
# Authors: Joseph Knox
# License: GNU General Public License v3.0
import numpy as np
from scipy import linalg


def conjugate_transpose(A):
    """Performs conjugate transpose of A"""
    if A.dtype == np.complexfloating:
        return A.conj().T
    return A.T
