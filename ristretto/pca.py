"""
Sparse Principal Component Analysis (SPCA).
"""
# Authors: N. Benjamin Erichson
#          Joseph Knox
# License: GNU General Public License v3.0

from __future__ import division, print_function

import numpy as np
from scipy import linalg

from .qb import rqb


def spca(X, n_components, alpha=0.1, beta=0.01, max_iter=500, tol=1e-5,
        verbose=True):
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

    beta : float, (default ``beta = 0.1``).
        Amount of ridge shrinkage to apply in order to improve conditionin.

    max_iter : integer, (default ``max_iter = 500``).
        Maximum number of iterations to perform before exiting.

    tol : float, (default ``tol = 1e-5``).
        Stopping tolerance for reconstruction error.

    verbose : bool ``{'True', 'False'}``, optional (default ``verbose = True``).
        Display progress.

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
    Variable Projection for SPCA solves the following optimization problem:
    minimize :math:`1/2 \| X - X B A^T \|^2 + \alpha \|B\|_1 + 1/2 \beta \|B\|^2`
    """
    # Shape of input matrix
    m = X.shape[0]

    #--------------------------------------------------------------------
    #   Initialization of Variable Projection Solver
    #--------------------------------------------------------------------
    _, D, Vt = linalg.svd(X, full_matrices=False, overwrite_a=False)
    Dmax = D[0] # l2 norm

    A = Vt.T[:, 0:n_components]
    B = Vt.T[:, 0:n_components]

    VD = Vt.T * D
    VD2 = Vt.T * D**2

    # Set Tuning Parameters
    alpha *= Dmax**2
    beta *= Dmax**2

    n_iter = 0
    nu = 1.0 / (Dmax**2 + beta)
    kappa = nu * alpha

    #   Apply Variable Projection Solver
    obj = []
    while max_iter > n_iter:

        # Update A:
        # X'XB = UDV'
        # Compute X'XB via SVD of X
        Z = VD2.dot(Vt.dot(B))

        Utilde, Dtilde, Vttilde = linalg.svd(Z, full_matrices=False, overwrite_a=True)

        A = Utilde.dot(Vttilde)

        # Proximal Gradient Descent to Update B
        #G = XtX.dot(A-B) - beta * B
        G = VD2.dot(Vt.dot(A - B)) - beta * B
        B_temp = B + nu * G

        # l1 soft-threshold
        idxH = B_temp > kappa
        idxL = B_temp <= -kappa
        B = np.zeros_like(B)
        B[idxH] = B_temp[idxH] - kappa
        B[idxL] = B_temp[idxL] + kappa

        if n_iter % 5 == 0:
            # compute residual
            R = VD.T - VD.T.dot(B).dot(A.T)

            # Compute objective function
            obj.append(0.5*np.sum(R**2) + alpha*np.sum(np.abs(B)) +
                       0.5*beta*np.sum(B**2))

            # Verbose
            if verbose:
                print("Iteration:  %s, Objective value:  %s" % (n_iter, obj[-1]))

            # Break if obj is not improving anymore
            if n_iter > 0 and abs(obj[-2] - obj[-1]) / obj[-1] < tol:
                break

        # Next iter
        n_iter += 1

    eigvals = Dtilde / (m - 1)
    return(B, A, eigvals, obj)


def robspca(X, n_components, alpha=0.1, beta=0.1, gamma=0.1, max_iter=1000,
            tol=1e-5, verbose=True):
    r"""Robust Sparse Principal Component Analysis (Robust SPCA).

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

    beta : float, (default ``beta = 0.1``)
        Amount of ridge shrinkage to apply in order to improve conditionin.

    gamma : float, (default ``gamma = 0.1``).
        Sparsity controlling parameter for the error matrix S.
        Smaller values lead to a larger amount of n_iterse removeal.

    max_iter : integer, (default ``max_iter = 500``).
        Maximum number of iterations to perform before exiting.

    tol : float, (default ``tol = 1e-5``).
        Stopping tolerance for reconstruction error.

    verbose : bool ``{'True', 'False'}``, optional (default ``verbose = True``).
        Display progress.


    Returns
    -------
    B:  array_like, `(n, n_components)`.
        Sparse components extracted from the data.

    A : array_like, `(n, n_components)`.
        Orthogonal components extracted from the data.

    S : array_like, `(m, n)`.
        Sparse component which captures grossly corrupted entries in the data
        matrix.

    eigvals : array_like, `(n_components)`.
        Eigenvalues correspnding to the extracted components.

    obj : array_like, `(n_iter)`.
        Objective value at the i-th iteration.


    Notes
    -----
    Variable Projection for SPCA solves the following optimization problem:
    minimize :math:`1/2 \| X - X B A^T \|^2 + \alpha \|B\|_1 + 1/2 \beta \|B\|^2`
    """
    # Shape of input matrix
    m = X.shape[0]

    # Initialization of Variable Projection Solver
    U, D, Vt = linalg.svd(X, full_matrices=False, overwrite_a=False)

    Dmax = D[0] #l2 norm

    U = U[:,0:n_components]
    Vt = Vt[0:n_components,:]

    A = Vt.T
    B = Vt.T

    # Set Tuning Parameters
    alpha *= Dmax**2
    beta *= Dmax**2
    nu   = 1.0 / (Dmax**2 + beta)
    kappa = nu * alpha
    S = np.zeros_like(X)

    # Apply Variable Projection Solver
    n_iter = 0
    obj = []
    while max_iter > n_iter:

        # Update A:
        # X'XB = UDV'
        # Compute X'XB via SVD of X
        XS = X - S
        XB = X.dot(B)
        Z = (XS).T.dot(XB)

        Utilde, Dtilde, Vttilde = linalg.svd( Z , full_matrices=False, overwrite_a=True)
        A = Utilde.dot(Vttilde)


        # Proximal Gradient Descent to Update B
        R = XS - XB.dot(A.T)
        G = X.T.dot(R.dot(A)) - beta * B
        B_temp = B + nu * G


        # l1 soft-threshold
        idxH = B_temp > kappa
        idxL = B_temp <= -kappa
        B = np.zeros_like(B)
        B[idxH] = B_temp[idxH] - kappa
        B[idxL] = B_temp[idxL] + kappa

        # compute residual
        R = X - X.dot(B).dot(A.T)

        # l1 soft-threshold
        idxH = R > gamma
        idxL = R <= -gamma
        S = np.zeros_like(S)
        S[idxH] = R[idxH] - gamma
        S[idxL] = R[idxL] + gamma


        # Compute objective function
        obj.append(0.5 * np.sum((R-S)**2) + alpha * np.sum(abs(B)) +
                   0.5 * beta * np.sum(B**2) + gamma * np.sum(abs(S)))



        # Verbose
        if verbose and n_iter % 10 == 0:
            print("Iteration:  %s, Objective:  %s" % (n_iter, obj[n_iter]))


        # Break if obj is not improving anymore
        if n_iter > 0 and abs(obj[-1]-obj[-2]) / obj[-1] < tol:
            break

        # Next iter
        n_iter += 1

    eigvals = Dtilde / (m-1)
    return B, A, S, eigvals, obj


def rspca(X, n_components, alpha=0.1, beta=0.1, max_iter=1000, tol=1e-5,
          verbose=0, oversample=10, n_subspace=1, random_state=None):
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

    max_iter : integer, (default ``max_iter = 500``).
        Maximum number of iterations to perform before exiting.

    tol : float, (default ``tol = 1e-5``).
        Stopping tolerance for reconstruction error.

    verbose : bool ``{'True', 'False'}``, optional (default ``verbose = True``).
        Display progress.

    oversample : integer, optional (default: 10)
        Controls the oversampling of column space. Increasing this parameter
        may improve numerical accuracy.

    n_subspace : integer, default: 1.
        Parameter to control number of subspace iterations. Increasing this
        parameter may improve numerical accuracy.

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
    Q, Xcompressed = rqb(X, rank=n_components, oversample=oversample,
                         n_subspace=n_subspace, random_state=random_state)

    # Compute Sparse PCA
    B, A, eigvals, obj = spca(Xcompressed, n_components=n_components,
                              alpha=alpha, beta=beta,
                              max_iter=max_iter, tol=tol, verbose=verbose)

    # rescale eigen values
    eigvals = eigvals * (n_components + oversample - 1) / (m-1)

    return B, A, eigvals, obj
