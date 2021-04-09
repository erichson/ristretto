"""
Principal Component Analysis (PCA).
"""
# Authors: Fengzhe Shi
from __future__ import division

import numpy as np
from scipy import linalg
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.extmath import svd_flip

from .svd import compute_rsvd


def compute_pca(X, n_components=None, svd_type="original", oversample=10, n_subspace=2, n_blocks=1, sparse=False, random_state=None):
    r"""Principal Component Analysis (PCA).

    Given a centered rectangular matrix `A` with shape `(m, n)`, PCA
    computes a set of components that can optimally reconstruct the
    input data.


    Parameters
    ----------
    X : array_like, shape `(m, n)`.
        Input array.

    n_components : integer, `n_components <= min{m,n}`.
        Target rank, i.e., number of components to be computed.

    svd_type : {"original", "randomized"}, (default ``svd_type = "original"``).
        If "original", the SVD solver from sklearn will be called to solve the SVD.
        If "randomized",

    The following parameters are for randomized SVD.
    oversample : integer, optional (default: 10)
        Controls the oversampling of column space. Increasing this parameter
        may improve numerical accuracy.

    n_subspace : integer, default: 2.
        Parameter to control number of subspace iterations. Increasing this
        parameter may improve numerical accuracy.

    n_blocks : integer, default: 1.
        If `n_blocks > 1` a column blocked QB decomposition procedure will be
        performed. A larger number requires less fast memory, while it
        leads to a higher computational time.

    sparse : boolean, optional (default: False)
        If sparse == True, perform compressed rsvd.

    random_state : integer, RandomState instance or None, optional (default ``None``)
        If integer, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used by np.random.

    Returns
    -------
    A : array_like, `(n, n_components)`.
        Orthogonal components extracted from the data.

    """

    m, n = X.shape
    if n_components is not None:
        if n_components > n:
            raise ValueError('n_components must be less than the number '
                             'of columns of X (%d)' % n)

    else:
        n_components = min(m, n)

    # Initialization of Variable Projection Solver
    if svd_type == "original":
        U, D, Vt = linalg.svd(X, full_matrices=False)
    elif svd_type == "randomized":
        U, D, Vt = compute_rsvd(X, n_components, oversample, n_subspace, n_blocks, sparse, random_state)

    U, Vt = svd_flip(U, Vt)
    explained_variance_ = (D ** 2) / (m - 1)

    return Vt[:n_components], explained_variance_[:n_components]


class PCA(BaseEstimator):

    def __init__(self, n_components=None, svd_type="original", oversample=10,
                 n_subspace=2, n_blocks=1, sparse=False, random_state=None):
        self.n_components = n_components
        self.svd_type = svd_type
        self.mean_ = 0
        self.explained_variance_ = 0
        self.oversample = oversample
        self.n_subspace = n_subspace
        self.n_blocks = n_blocks
        self.sparse = sparse
        self.random_state = random_state
        self.A_ = None

    def _transform(self, X, A):
        if self.mean_ is not None:
            X -= self.mean_

        Z = np.dot(X, A.T)
        Z /= np.sqrt(self.explained_variance_)
        return Z

    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        X -= self.mean_
        self.A_, self.explained_variance_ = compute_pca(
            X, n_components=self.n_components, svd_type=self.svd_type,
            oversample=self.oversample, n_subspace=self.n_subspace, n_blocks=self.n_blocks,
            sparse=self.sparse, random_state=self.random_state)
        return self

    def fit_transform(self, X):
        self.fit(X)
        return self._transform(X, self.A_)

    def transform(self, X):
        check_is_fitted(self, ['A_'])
        return self._transform(X, self.A_)

    def inverse_transform(self, X):
        check_is_fitted(self, ['A_'])
        return X.dot(self.A_.T)
