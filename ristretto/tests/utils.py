from __future__ import division

from scipy import linalg


def relative_error(A, B):
    return linalg.norm(A - B) / linalg.norm(A)
