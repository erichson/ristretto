"""
Principal Component Analysis (PCA).
"""
# Authors: Fengzhe Shi
from __future__ import division

import numpy as np
from scipy import linalg
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from .svd import compute_rsvd


def compute_pca(X, n_components=None, svd_type="original"):
    r"""Principal Component Analysis (PCA).

    Given a mean centered rectangular matrix `A` with shape `(m, n)`, PCA
    computes a set of components that can optimally reconstruct the
    input data.


    Parameters
    ----------
    X : array_like, shape `(m, n)`.
        Input array.

    n_components : integer, `n_components <= min{m,n}`.
        Target rank, i.e., number of components to be computed.

    svd_type : {"original"}, (default ``svd_type = "original"``).
        When "original", the SVD solver from sklearn will be called to solve the SVD.

    Returns
    -------
    B:  array_like, `(n, n_components)`.
        Sparse components extracted from the data.

    A : array_like, `(n, n_components)`.
        Orthogonal components extracted from the data.

    Notes
    -----
    Variable Projection for SPCA solves the following optimization problem:
    minimize :math:`1/2 \| X - X B A^T \|^2 + \alpha \|B\|_1 + 1/2 \beta \|B\|^2`
    """

    X -= np.mean(X)

    # TODO Different from Sparse PCA
    m, n = X.shape
    if n_components is not None:
        if n_components > n:
            raise ValueError('n_components must be less than the number '
                             'of columns of X (%d)' % n)

    else:
        n_components = min(m, n)

    # Initialization of Variable Projection Solver
    if svd_type == "original":
        U, D, Vt = linalg.svd(X, full_matrices=False, overwrite_a=False)
    elif svd_type == "randomized":
        U, D, Vt = compute_rsvd(X, n_components)

    A = Vt[:n_components].T
    B = Vt[:n_components].T

    return B, A


class PCA(BaseEstimator):

    def __init__(self, n_components=None, svd_type="originial"):
        self.n_components = n_components
        self.svd_type = svd_type

        # TODO Define all the attributes inside the __init__ function
        self.B_ = None
        self.A_ = None

    def _transform(self, X, B):
        # TODO: CHECK!
        return X.dot(B)

    def fit(self, X):
        # TODO Should we standardize the input X?
        self.B_, self.A_ = compute_pca(
            X, n_components=self.n_components, svd_type=self.svd_type)
        return self

    def fit_transform(self, X):
        self.fit(X)
        return self._transform(X, self.B_)

    def transform(self, X):
        check_is_fitted(self, ['B_'])
        return self._transform(X, self.B_)

    def inverse_transform(self, X):
        check_is_fitted(self, ['A_'])
        # TODO: CHECK!
        return X.dot(self.A_.T)
