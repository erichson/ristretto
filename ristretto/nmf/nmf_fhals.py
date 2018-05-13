"""
Nonnegative Matrix Factorization.
"""
# Authors: N. Benjamin Erichson
#          Joseph Knox
# License: GNU General Public License v3.0

from __future__ import division, print_function

import numpy as np
from scipy import linalg

from ..mf.rqb import rqb
from ..externals.cdnmf_fast import _update_cdnmf_fast as _fhals_update_shuffle
from ..externals.nmf import _initialize_nmf
from ..externals.utils import check_random_state, check_non_negative


def nmf(A, k, init='nndsvd', shuffle=False,
        l2_reg_H=0.0, l2_reg_W=0.0, l1_reg_H=0.0, l1_reg_W=0.0,
        tol=1e-5, maxiter=200, random_state=None, verbose=False):
    """Nonnegative Matrix Factorization.

    Hierarchical alternating least squares algorithm
    for computing the approximate low-rank nonnegative matrix factorization of
    a rectangular `(m, n)` matrix `A`. Given the target rank `k << min{m,n}`,
    the input matrix `A` is factored as `A = W H`. The nonnegative factor
    matrices `W` and `H` are of dimension `(m, k)` and `(k, n)`, respectively.


    Parameters
    ----------
    A : array_like, shape `(m, n)`.
        Real nonnegative input matrix.

    k : integer, `k << min{m,n}`.
        Target rank.

    init :  'random' | 'nndsvd' | 'nndsvda' | 'nndsvdar'
        Method used to initialize the procedure. Default: 'nndsvd'.
        Valid options:
        - 'random': non-negative random matrices, scaled with:
            sqrt(X.mean() / n_components)
        - 'nndsvd': Nonnegative Double Singular Value Decomposition (NNDSVD)
            initialization (better for sparseness)
        - 'nndsvda': NNDSVD with zeros filled with the average of X
            (better when sparsity is not desired)
        - 'nndsvdar': NNDSVD with zeros filled with small random values
            (generally faster, less accurate alternative to NNDSVDa
            for when sparsity is not desired)

    shuffle : boolean, default: False
        If true, randomly shuffle the update order of the variables.

    l2_reg_H : float, (default ``l2_reg_H = 0.1``).
        Amount of ridge shrinkage to apply to `H` to improve conditionin.

    l2_reg_W : float, (default ``l2_reg_W = 0.1``).
        Amount of ridge shrinkage to apply to `W` to improve conditionin.

    l1_reg_H : float, (default ``l1_reg_H = 0.0``).
        Sparsity controlling parameter on `H`.
        Higher values lead to sparser components.

    l1_reg_W : float, (default ``l1_reg_W = 0.0``).
        Sparsity controlling parameter on `W`.
        Higher values lead to sparser components.

    tol : float, default: `tol=1e-4`.
        Tolerance of the stopping condition.

    maxiter : integer, default: `maxiter=100`.
        Number of iterations.

    random_state : integer, RandomState instance or None, optional (default ``None``)
        If integer, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used by np.random.

    verbose : boolean, default: `verbose=False`.
        The verbosity level.


    Returns
    -------
    W:  array_like, `(m, k)`.
        Solution to the non-negative least squares problem.

    H : array_like, `(k, n)`.
        Solution to the non-negative least squares problem.


    Notes
    -----
    This HALS update algorithm written in cython is adapted from the
    scikit-learn implementation for the deterministic NMF. We also have
    adapted the initilization scheme.

    See: https://github.com/scikit-learn/scikit-learn


    References
    ----------
    [1] Cichocki, Andrzej, and P. H. A. N. Anh-Huy. "Fast local algorithms for
    large scale nonnegative matrix and tensor factorizations."
    IEICE transactions on fundamentals of electronics, communications and
    computer sciences 92.3: 708-721, 2009.

    [2] C. Boutsidis, E. Gallopoulos: SVD based initialization: A head start for
    nonnegative matrix factorization - Pattern Recognition, 2008
    http://tinyurl.com/nndsvd


    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[1,1], [2, 1], [3, 1.2], [4, 1], [5, 0.8], [6, 1]])
    >>> import ristretto as ro
    >>> W, H = ro.rnm(X, k=2, p=0)
    """
    # checks
    check_non_negative(A, 'ristretto.nmf.nmf')
    rns = check_random_state(random_state)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialization methods for factor matrices W and H
    # 'normal': nonnegative standard normal random init
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    W, H = _initialize_nmf(A, k, init=init, eps=1e-6, random_state=rns)
    Ht = np.array(H.T, order='C')

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Iterate the HALS algorithm until convergence or maxiter is reached
    # i)   Update factor matrix H and normalize columns
    # ii)  Update low-dimensional factor matrix W
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    for niter in range(maxiter):
        violation = 0.0

        # Update factor matrix H with regularization
        WtW = W.T.dot(W)
        WtW.flat[::k + 1] += l2_reg_H # adds l2_reg only on the diagonal
        AtW = A.T.dot(W) - l1_reg_H

        # compute violation update
        permutation = rns.permutation(k) if shuffle else np.arange(k)
        violation += _fhals_update_shuffle(Ht, WtW, AtW, permutation)

        # Update factor matrix W with regularization
        HHt = Ht.T.dot(Ht)
        HHt.flat[::k + 1] += l2_reg_W # adds l2_reg only on the diagonal
        AHt = A.dot(Ht) - l1_reg_W

        # compute violation update
        permutation = rns.permutation(k) if shuffle else np.arange(k)
        violation += _fhals_update_shuffle(W, HHt, AHt, permutation)

        # Compute stopping condition.
        if niter == 0:
            violation_init = violation

        if violation_init == 0:
            break

        fitchange = violation / violation_init

        if verbose == True and niter % 10 == 0:
            print('Iteration: %s fit: %s, fitchange: %s' %(niter, violation, fitchange))

        if fitchange <= tol:
            break

    # Return factor matrices
    return W, Ht.T


def rnmf(A, k, p=20, q=2, init='nndsvd', shuffle=False,
         l2_reg_H=0.0, l2_reg_W=0.0, l1_reg_H=0.0, l1_reg_W=0.0,
         tol=1e-5, maxiter=200, random_state=None, verbose=False):
    """
    Randomized Nonnegative Matrix Factorization.

    Randomized hierarchical alternating least squares algorithm
    for computing the approximate low-rank nonnegative matrix factorization of
    a rectangular `(m, n)` matrix `A`. Given the target rank `k << min{m,n}`,
    the input matrix `A` is factored as `A = W H`. The nonnegative factor
    matrices `W` and `H` are of dimension `(m, k)` and `(k, n)`, respectively.

    The quality of the approximation can be controlled via the oversampling
    parameter `p` and the parameter `q` which specifies the number of
    subspace iterations.


    Parameters
    ----------
    A : array_like, shape `(m, n)`.
        Real nonnegative input matrix.

    k : integer, `k << min{m,n}`.
        Target rank, i.e., number of components to extract from the data

    p : integer, default: `p=20`.
        Parameter to control oversampling.

    q : integer, default: `q=2`.
        Parameter to control number of power (subspace) iterations.

    init :  'random' | 'nndsvd' | 'nndsvda' | 'nndsvdar'
        Method used to initialize the procedure. Default: 'nndsvd'.
        Valid options:
        - 'random': non-negative random matrices, scaled with:
            sqrt(X.mean() / n_components)
        - 'nndsvd': Nonnegative Double Singular Value Decomposition (NNDSVD)
            initialization (better for sparseness)
        - 'nndsvda': NNDSVD with zeros filled with the average of X
            (better when sparsity is not desired)
        - 'nndsvdar': NNDSVD with zeros filled with small random values
            (generally faster, less accurate alternative to NNDSVDa
            for when sparsity is not desired)

    shuffle : boolean, default: False
        If true, randomly shuffle the update order of the variables.

    l2_reg_H : float, (default ``l2_reg_H = 0.1``).
        Amount of ridge shrinkage to apply to `H` to improve conditioning.

    l2_reg_W : float, (default ``l2_reg_W = 0.1``).
        Amount of ridge shrinkage to apply to `W` to improve conditioning.

    l1_reg_H : float, (default ``l1_reg_H = 0.0``).
        Sparsity controlling parameter on `H`.
        Higher values lead to sparser components.

    l1_reg_W : float, (default ``l1_reg_W = 0.0``).
        Sparsity controlling parameter on `W`.
        Higher values lead to sparser components.

    tol : float, default: `tol=1e-5`.
        Tolerance of the stopping condition.

    maxiter : integer, default: `maxiter=200`.
        Number of iterations.

    random_state : integer, RandomState instance or None, optional (default ``None``)
        If integer, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used by np.random.

    verbose : boolean, default: `verbose=False`.
        The verbosity level.


    Returns
    -------
    W:  array_like, `(m, k)`.
        Solution to the non-negative least squares problem.

    H : array_like, `(k, n)`.
        Solution to the non-negative least squares problem.


    Notes
    -----
    This HALS update algorithm written in cython is adapted from the
    scikit-learn implementation for the deterministic NMF.  We also have
    adapted the initilization scheme.

    See: https://github.com/scikit-learn/scikit-learn


    References
    ----------
    [1] Erichson, N. Benjamin, Ariana Mendible, Sophie Wihlborn, and J. Nathan Kutz.
    "Randomized Nonnegative Matrix Factorization."
    Pattern Recognition Letters (2018).

    [2] Cichocki, Andrzej, and P. H. A. N. Anh-Huy. "Fast local algorithms for
    large scale nonnegative matrix and tensor factorizations."
    IEICE transactions on fundamentals of electronics, communications and
    computer sciences 92.3: 708-721, 2009.

    [3] C. Boutsidis, E. Gallopoulos: SVD based initialization: A head start for
    nonnegative matrix factorization - Pattern Recognition, 2008
    http://tinyurl.com/nndsvd

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[1,1], [2, 1], [3, 1.2], [4, 1], [5, 0.8], [6, 1]])
    >>> import ristretto as ro
    >>> W, H = ro.rnmf(X, k=2, p=0)
    """
    # checks
    check_non_negative(A, 'ristretto.nmf.nmf')
    rns = check_random_state(random_state)
    m, n = A.shape

    flipped = False
    if n > m :
        A = A.T
        m, n = A.shape
        flipped = True

    # Compute QB decomposition
    Q, B = rqb(A, k=k, p=p, q=q, sdist=None)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialization methods for factor matrices W and H
    # 'normal': nonnegative standard normal random init
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    W, H = _initialize_nmf(A, k, init=init, eps=1e-6, random_state=rns)
    Ht = np.array(H.T, order='C')
    W_tilde = Q.T.dot(W)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Iterate the HALS algorithm until convergence or maxiter is reached
    # i)   Update factor matrix H
    # ii)  Update factor matrix W
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    for niter in range(maxiter):
        violation = 0.0

        # Update factor matrix H
        WtW = W.T.dot(W)
        WtW.flat[::k + 1] += l2_reg_H # adds l2_reg only on the diagonal
        BtW = B.T.dot(W_tilde) - l1_reg_H

        # compute violation update
        permutation = rns.permutation(k) if shuffle else np.arange(k)
        violation += _fhals_update_shuffle(Ht, WtW, BtW, permutation)

        # Update factor matrix W
        HHt = Ht.T.dot(Ht)
        HHt.flat[::k + 1] += l2_reg_W # adds l2_reg only on the diagonal

        # Rotate AHt back to high-dimensional space
        BHt = Q.dot(B.dot(Ht)) - l1_reg_W

        # compute violation update
        permutation = rns.permutation(k) if shuffle else np.arange(k)
        violation += _fhals_update_shuffle(W, HHt, BHt, permutation)

        # Project W to low-dimensional space
        W_tilde = Q.T.dot(W)

        # Compute stopping condition.
        if niter == 0:
            violation_init = violation

        if violation_init == 0:
            break

        fitchange = violation / violation_init

        if verbose:
            show = 10 if niter < 100 else 50
            if niter % show == 0:
                print('Iteration: %s fit: %s, fitchange: %s'
                      % (niter+1, violation, fitchange))

        if fitchange <= tol:
            break

    if verbose:
        print('Iteration: %d, fit: %s, fitchange: %s' % (niter, violation, fitchange))

    # Return factor matrices
    if flipped:
        return(Ht, W.T)
    return(W, Ht.T)
