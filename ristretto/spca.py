# TODO: improve docs (especially for classes)
# TODO: evaluate whether returning S is necessary when robust == True
#    S : array_like, `(m, n)`.
#        Sparse component which captures grossly corrupted entries in the data
#        matrix.
"""
Sparse Principal Component Analysis (SPCA).
"""
# Authors: N. Benjamin Erichson
#          Joseph Knox
# License: GNU General Public License v3.0
from __future__ import division

import numpy as np
from scipy import linalg
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from .qb import compute_rqb
from .utils import soft_l0, soft_l1


def compute_spca(X, n_components=None, alpha=0.1, beta=1e-5, gamma=0.1,
                 robust=False, regularizer='l1', max_iter=1e3, tol=1e-5):
    r"""Sparse Principal Component Analysis (SPCA).

    Given a mean centered rectangular matrix `A` with shape `(m, n)`, SPCA
    computes a set of sparse components that can optimally reconstruct the
    input data.  The amount of sparseness is controllable by the coefficient
    of the L1 penalty, given by the parameter alpha. In addition, some ridge
    shrinkage can be applied in order to improve conditioning.


    Parameters
    ----------
    X : array_like, shape `(m, n)`.
        Input array.

    n_components : integer, `n_components << min{m,n}`.
        Target rank, i.e., number of sparse components to be computed.

    alpha : float, (default ``alpha = 0.1``).
        Sparsity controlling parameter. Higher values lead to sparser components.

    beta : float, (default ``beta = 1e-5``).
        Amount of ridge shrinkage to apply in order to improve conditionin.

    regularizer : string {'l0', 'l1'}.
        Type of sparsity-inducing regularizer. The l1 norm (also known as LASSO)
        leads to softhreshold operator (default).  The l0 norm is implemented
        via a hardthreshold operator.

    robust : bool ``{'True', 'False'}``, optional (default ``False``).
        Use a robust algorithm to compute the sparse PCA.

    max_iter : integer, (default ``max_iter = 500``).
        Maximum number of iterations to perform before exiting.

    tol : float, (default ``tol = 1e-5``).
        Stopping tolerance for reconstruction error.

    Returns
    -------
    B:  array_like, `(n, n_components)`.
        Sparse components extracted from the data.

    A : array_like, `(n, n_components)`.
        Orthogonal components extracted from the data.

    eigvals : array_like, `(n_components)`.
        Eigenvalues correspnding to the extracted components.

    obj : array_like, `(n_iter)`.
        Objective value at the i-th iteration.

    Notes
    -----
    Variable Projection for PCA solves the following optimization problem:
    minimize :math:`1/2 \| X - X B A^T \|^2 + \alpha \|B\|_1 + 1/2 \beta \|B\|^2`
    """
    def compute_residual(X, B, A):
        return X - X.dot(B).dot(A.T)

    if regularizer == 'l1':
        regularizer_func = soft_l1
    elif regularizer == 'l0':
        if robust:
            raise NotImplementedError('l0 regularization is not supported for '
                                      'robust sparse pca')
        regularizer_func = soft_l0
    else:
        raise ValueError('regularizer must be one of ("l1", "l0"), not '
                         '%s.' % regularizer)

    m, n = X.shape
    if n_components is not None:
        if n_components > n:
            raise ValueError('n_components must be less than the number '
                             'of columns of X (%d)' % n)
    else:
        n_components = n

    # Initialization of Variable Projection Solver
    U, D, Vt = linalg.svd(X, full_matrices=False, overwrite_a=False)
    Dmax = D[0]  # l2 norm

    A = Vt[:n_components].T
    B = Vt[:n_components].T

    if robust:
        U = U[:, :n_components]
        Vt = Vt[:n_components]
        S = np.zeros_like(X)
    else:
        # compute outside the loop
        VD = Vt.T * D
        VD2 = Vt.T * D**2

    # Set Tuning Parameters
    alpha *= Dmax**2
    beta *= Dmax**2
    nu = 1.0 / (Dmax**2 + beta)
    kappa = nu * alpha

    obj = []  # values of objective function
    n_iter = 0

    #   Apply Variable Projection Solver
    while max_iter > n_iter:

        # Update A:
        # X'XB = UDV'
        # Compute X'XB via SVD of X
        if robust:
            XS = X - S
            XB = X.dot(B)
            Z = (XS).T.dot(XB)
        else:
            Z = VD2.dot(Vt.dot(B))

        Utilde, Dtilde, Vttilde = linalg.svd(Z, full_matrices=False, overwrite_a=True)
        A = Utilde.dot(Vttilde)

        # Proximal Gradient Descent to Update B
        if robust:
            R = XS - XB.dot(A.T)
            G = X.T.dot(R.dot(A)) - beta * B
        else:
            G = VD2.dot(Vt.dot(A - B)) - beta * B

        B = regularizer_func(B + nu * G, kappa)

        if robust:
            R = compute_residual(X, B, A)
            S = soft_l1(R, gamma)
            R -= S
        else:
            R = compute_residual(VD.T, B, A)

        objective = 0.5*np.sum(R**2) + alpha*np.sum(np.abs(B)) + 0.5*beta*np.sum(B**2)
        if robust:
            objective += gamma * np.sum(np.abs(S))

        obj.append(objective)

        # Break if obj is not improving anymore
        if n_iter > 0 and abs(obj[-2] - obj[-1]) / obj[-1] < tol:
            break

        # Next iter
        n_iter += 1

    eigen_values = Dtilde / (m-1)

    return B, A, eigen_values, obj


def compute_rspca(X, n_components, alpha=0.1, beta=0.1, max_iter=1000,
                  regularizer='l1', tol=1e-5, oversample=50, n_subspace=2,
                  n_blocks=1, robust=False, random_state=None):
    r"""Randomized Sparse Principal Component Analysis (rSPCA).

    Given a mean centered rectangular matrix `A` with shape `(m, n)`, SPCA
    computes a set of sparse components that can optimally reconstruct the
    input data.  The amount of sparseness is controllable by the coefficient
    of the L1 penalty, given by the parameter alpha. In addition, some ridge
    shrinkage can be applied in order to improve conditioning.

    This algorithm uses randomized methods for linear algebra to accelerate
    the computations.

    The quality of the approximation can be controlled via the oversampling
    parameter `oversample` and `n_subspace` which specifies the number of
    subspace iterations.


    Parameters
    ----------
    X : array_like, shape `(m, n)`.
        Real nonnegative input matrix.

    n_components : integer, `n_components << min{m,n}`.
        Target rank, i.e., number of sparse components to be computed.

    alpha : float, (default ``alpha = 0.1``).
        Sparsity controlling parameter. Higher values lead to sparser components.

    beta : float, (default ``beta = 0.1``).
        Amount of ridge shrinkage to apply in order to improve conditionin.

    regularizer : string {'l0', 'l1'}.
        Type of sparsity-inducing regularizer. The l1 norm (also known as LASSO)
        leads to softhreshold operator (default).  The l0 norm is implemented
        via a hardthreshold operator.

    max_iter : integer, (default ``max_iter = 500``).
        Maximum number of iterations to perform before exiting.

    tol : float, (default ``tol = 1e-5``).
        Stopping tolerance for reconstruction error.

    verbose : bool ``{'True', 'False'}``, optional (default ``verbose = True``).
        Display progress.

    oversample : integer, optional (default: 10)
        Controls the oversampling of column space. Increasing this parameter
        may improve numerical accuracy.

    n_subspace : integer, default: 2.
        Parameter to control number of subspace iterations. Increasing this
        parameter may improve numerical accuracy.

    n_blocks : integer, default: 2.
        Paramter to control in how many blocks of columns the input matrix
        should be split. A larger number requires less fast memory, while it
        leads to a higher computational time.

    random_state : integer, RandomState instance or None, optional (default ``None``)
        If integer, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used by np.random.

    Returns
    -------
    B:  array_like, `(n, n_components)`.
        Sparse components extracted from the data.

    A : array_like, `(n, n_components)`.
        Orthogonal components extracted from the data.

    eigvals : array_like, `(n_components)`.
        Eigenvalues correspnding to the extracted components.

    S : array_like, `(m, n)`.
        Sparse component which captures grossly corrupted entries in the data
        matrix. Returned only if `robust == True`

    obj : array_like, `(n_iter)`.
        Objective value at the i-th iteration.

    Notes
    -----
    Variable Projection for SPCA solves the following optimization problem:
    minimize :math:`1/2 \| X - X B A^T \|^2 + \alpha \|B\|_1 + 1/2 \beta \|B\|^2`
    """
    # Shape of data matrix
    m = X.shape[0]

    # Compute QB decomposition
    Q, Xcompressed = compute_rqb(
        X, rank=n_components, oversample=oversample, n_subspace=n_subspace,
        n_blocks=n_blocks, random_state=random_state)

    # Compute Sparse PCA
    B, A, eigen_values, obj = compute_spca(
        Xcompressed, n_components=n_components, alpha=alpha, beta=beta,
        regularizer=regularizer, max_iter=max_iter, tol=tol, robust=robust)

    # rescale eigen values
    eigen_values *= (n_components + oversample - 1) / (m-1)

    return B, A, eigen_values, obj


class SPCA(BaseEstimator):

    def __init__(self, n_components=None, alpha=0.1, beta=1e-5, gamma=0.1,
                 robust=False, regularizer='l1', max_iter=1e3, tol=1e-5):
        self.n_components = n_components
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.robust = robust
        self.regularizer = regularizer
        self.max_iter = max_iter
        self.tol = tol

    def _transform(self, X, B):
        # TODO: CHECK!
        return X.dot(B)

    def fit(self, X):
        self.B_, self.A_, self.eigen_values_, self.obj_ = compute_spca(
            X, n_components=self.n_components, alpha=self.alpha, beta=self.beta,
            gamma=self.gamma, robust=self.robust, regularizer=self.regularizer,
            max_iter=self.max_iter, tol=self.tol)
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


class RSPCA(SPCA):

    def __init__(self, n_components=None, alpha=0.1, beta=1e-5,
                 robust=False, regularizer='l1', max_iter=1e3, tol=1e-5,
                 oversample=50, n_subspace=2, n_blocks=1, random_state=None):
        super(RSPCA, self).__init__(
            n_components=n_components, alpha=alpha,  beta=beta,
            robust=robust, regularizer=regularizer, max_iter=max_iter, tol=tol)
        self.oversample = oversample
        self.n_subspace = n_subspace
        self.n_blocks = n_blocks
        self.random_state = random_state

    def fit(self, X):
        self.B_, self.A_, self.eigen_values_, self.obj_ = compute_rspca(
            X, n_components=self.n_components, alpha=self.alpha, beta=self.beta,
            robust=self.robust, regularizer=self.regularizer,
            max_iter=self.max_iter, tol=self.tol, oversample=self.oversample,
            n_subspace=self.n_subspace, n_blocks=self.n_blocks,
            random_state=self.random_state)
        return self
