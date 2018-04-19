""" 
Nonnegative Matrix Factorization
"""
# Author: N. Benjamin Erichson
# License: GNU General Public License v3.0


from __future__ import division

import numpy as np
import scipy as sci
from scipy import linalg

from ristretto._fhals_update_shuffle import _fhals_update_shuffle
from ristretto import rsvd



#import pyximport; pyximport.install()
#from _fhals_update_shuffle import _fhals_update_shuffle





# The rutine to initialize the NMF is adapted from the sci-learn package, see
# https://github.com/scikit-learn/scikit-learn/blob/a24c8b46/sklearn/decomposition/nmf.py#L1040
# All credits go the sci-learn team, thx :)
def _initialize_nmf(X, n_components, init=None, eps=1e-6, random_state=None):
    """Algorithms for NMF initialization.
    Computes an initial guess for the non-negative
    rank k matrix approximation for X: X = WH
    Parameters
    ----------
    
    X : array-like, shape (n_samples, n_features)
        The data matrix to be decomposed.
    
    n_components : integer
        The number of components desired in the approximation.
    
    init :  'random' | 'nndsvd' | 'nndsvda' | 'nndsvdar'
        Method used to initialize the procedure.
        Default: 'nndsvd' if n_components < n_features, otherwise 'random'.
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
    
    eps : float
        Truncate all values less then this in output to zero.
    
    random_state : int, RandomState instance or None, optional, default: None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`. Used when ``random`` == 'nndsvdar' or 'random'.
    
    Returns
    -------
    W : array-like, shape (n_samples, n_components)
        Initial guesses for solving X ~= WH
    H : array-like, shape (n_components, n_features)
        Initial guesses for solving X ~= WH
    
    References
    ----------
    C. Boutsidis, E. Gallopoulos: SVD based initialization: A head start for
    nonnegative matrix factorization - Pattern Recognition, 2008
    http://tinyurl.com/nndsvd
    """

    n_samples, n_features = X.shape

    if init is None:
        if n_components < n_features:
            init = 'nndsvd'
        else:
            init = 'random'

    # Random initialization
    if init == 'random':
        avg = np.sqrt(X.mean() / n_components)
        rng = random_state
        H = avg * rng.randn(n_components, n_features)
        W = avg * rng.randn(n_samples, n_components)
        # we do not write np.abs(H, out=H) to stay compatible with
        # numpy 1.5 and earlier where the 'out' keyword is not
        # supported as a kwarg on ufuncs
        np.abs(H, H)
        np.abs(W, W)
        return W, H

    # NNDSVD initialization
    U, S, V = rsvd(X, n_components, p=20, q=2)
    W, H = np.zeros(U.shape), np.zeros(V.shape)

    # The leading singular triplet is non-negative
    # so it can be used as is for initialization.
    W[:, 0] = np.sqrt(S[0]) * np.abs(U[:, 0])
    H[0, :] = np.sqrt(S[0]) * np.abs(V[0, :])

    for j in range(1, n_components):
        x, y = U[:, j], V[j, :]

        # extract positive and negative parts of column vectors
        x_p, y_p = np.maximum(x, 0), np.maximum(y, 0)
        x_n, y_n = np.abs(np.minimum(x, 0)), np.abs(np.minimum(y, 0))

        # and their norms
        x_p_nrm, y_p_nrm = np.sum(x_p**2)**0.5, np.sum(y_p**2)**0.5
        x_n_nrm, y_n_nrm = np.sum(x_n**2)**0.5, np.sum(y_n**2)**0.5

        m_p, m_n = x_p_nrm * y_p_nrm, x_n_nrm * y_n_nrm

        # choose update
        if m_p > m_n:
            u = x_p / x_p_nrm
            v = y_p / y_p_nrm
            sigma = m_p
        
        else:
            u = x_n / x_n_nrm
            v = y_n / y_n_nrm
            sigma = m_n

        lbd = np.sqrt(S[j] * sigma)
        W[:, j] = lbd * u
        H[j, :] = lbd * v

    W[W < eps] = 0
    H[H < eps] = 0

    if init == "nndsvd":
        pass
    
    elif init == "nndsvda":
        avg = X.mean()
        W[W == 0] = avg
        H[H == 0] = avg
        
    elif init == "nndsvdar":
        rng = random_state
        avg = X.mean()
        W[W == 0] = abs(avg * rng.randn(len(W[W == 0])) / 100)
        H[H == 0] = abs(avg * rng.randn(len(H[H == 0])) / 100)
    
    else:
        raise ValueError(
            'Invalid init parameter: got %r instead of one of %r' %
            (init, (None, 'random', 'nndsvd', 'nndsvda', 'nndsvdar')))

    return(W, H)




def rnmf(A, k, p=20, q=2, init='nndsvd', shuffle=False,
         l2_reg_H = 0.0, l2_reg_W = 0.0, l1_reg_H = 0.0, l1_reg_W = 0.0,
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
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Error catching
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~     
    m, n = A.shape  

    
    flipped = False
    if n > m :
        A = A.T
        m, n = A.shape
        flipped = True
    
    if (A < 0).any():
        raise ValueError("Input matrix with nonnegative elements is required.")    
    
    if  A.dtype == sci.float32: 
        data_type = sci.float32
        
    elif A.dtype == sci.float64: 
        data_type = sci.float64  

    else:
        raise ValueError("A.dtype is not supported.")    
    
    
    if random_state is None or isinstance(random_state, int):
            rns = sci.random.RandomState(random_state)
    
    elif isinstance(random_state, sci.random.RandomState):
            rns = random_state
    
    else:
        raise ValueError('Seed should be None, int or np.random.RandomState')    
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute QB decomposition
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~            

    #Build sample matrix Y : Y = A * Omega, where Omega is a random test matrix 
    Omega = sci.array( rns.rand( n, k+p ) , dtype = data_type ) 
    Y = sci.dot(A, Omega)
    del(Omega)

    #If q > 0 perfrom q subspace iterations  
    if q > 0:
        for i in range(q):
            Y , _ = sci.linalg.qr( Y, mode='economic', check_finite=False, overwrite_a=True)
            Z , _ = sci.linalg.qr( A.T.dot(Y), mode='economic', check_finite=False, overwrite_a=True)
            Y = sci.dot(A, Z)
        #End for
        del(Z)

     #End if       
    
    #Orthogonalize Y using economic QR decomposition: Y = QR          
    Q , _ = sci.linalg.qr( Y,  mode='economic', check_finite=False, overwrite_a=True) 
    
    #Project input data to low-dimensional space
    B = sci.dot(Q.T, A) 
    
    

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~                            
    # Initialization methods for factor matrices W and H
    # 'normal': nonnegative standard normal random init
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
    W, H = _initialize_nmf(A, k, init=init, eps=1e-6, random_state=rns)
    Ht = np.array(H.T, order='C')
    W_tilde = sci.dot(Q.T, W)
    
    del(A)


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Iterate the HALS algorithm until convergence or maxiter is reached
    # i)   Update factor matrix H
    # ii)  Update factor matrix W
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
        
    for niter in range(maxiter):
        violation = 0.0

        # Update factor matrix H
        #WtW = W_tilde.T.dot(W_tilde)        
        WtW = sci.dot(W.T, W)

        BtW = sci.dot(B.T, W_tilde)        
        
        # L2 regularization of H
        if l2_reg_H != 0.0:
            # adds l2_reg only on the diagonal
            WtW.flat[::k + 1] += l2_reg_H
        
        # L1 regularization  of H
        if l1_reg_H != 0.0:
            BtW -= l1_reg_H
        
        if shuffle:
            permutation = rns.permutation(k)
        else:
            permutation = np.arange(k)
        
        violation += _fhals_update_shuffle(Ht, WtW, BtW, permutation)                        
         

        # Update factor matrix W
        HHt = sci.dot(Ht.T, Ht)
        
        # Rotate AHt back to high-dimensional space
        BHt = sci.dot(Q, sci.dot(B, Ht)) 

        # L2 regularization of W
        if l2_reg_W != 0.0:
            # adds l2_reg only on the diagonal
            HHt.flat[::k + 1] += l2_reg_W
        
        # L1 regularization of W
        if l1_reg_W != 0.0:
            BHt -= l1_reg_W


        if shuffle:
            permutation = rns.permutation(k)
        else:
            permutation = np.arange(k)


        violation += _fhals_update_shuffle(W, HHt, BHt, permutation)


 
        # Project W to low-dimensional space
        W_tilde = sci.dot(Q.T, W)  
       
 

        # Compute stopping condition.
        if niter == 0:
            violation_init = violation
    
        if violation_init == 0:
            break       
    
        fitchange = violation / violation_init
        
        if niter < 100:
            show = 10
        else:
            show = 50
            
        if verbose == True and niter % show == 0:
            print('Iteration: %s fit: %s, fitchange: %s' %(niter+1, violation, fitchange))        
    
        if fitchange <= tol:
            break        
           
            
    #End for

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~                            
    # Return factor matrices
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  
    if verbose == True:
            print('Iteration: %s fit: %s, fitchange: %s' %(niter+1, violation, fitchange))        
       
    if flipped == False:
        return( W, Ht.T)
    else:
        return( Ht, W.T)





def nmf(A, k, init='nndsvd', shuffle=False, 
         l2_reg_H = 0.0, l2_reg_W = 0.0, l1_reg_H = 0.0, l1_reg_W = 0.0,
          tol=1e-5, maxiter=200, random_state=None, verbose=False):
    """
    Nonnegative Matrix Factorization.
    
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
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Error catching
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~     
    m, n = A.shape  
    
    
    if (A < 0).any():
        raise ValueError("Input matrix with nonnegative elements is required.")    
       

    if random_state is None or isinstance(random_state, int):
            rns = sci.random.RandomState(random_state)
    
    elif isinstance(random_state, sci.random.RandomState):
            rns = random_state
    
    else:
        raise ValueError('Seed should be None, int or np.random.RandomState')        

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
        
        # Update factor matrix H
        WtW = sci.dot(W.T, W)
        AtW = sci.dot(A.T, W)
        
        
         # L2 regularization of H
        if l2_reg_H != 0.0:
            # adds l2_reg only on the diagonal
            WtW.flat[::k + 1] += l2_reg_H
        
        # L1 regularization  of H
        if l1_reg_H != 0.0:
            AtW -= l1_reg_H
        
        if shuffle:
            permutation = rns.permutation(k)
        else:
            permutation = np.arange(k)

        
        violation += _fhals_update_shuffle(Ht, WtW, AtW, permutation)           
        
        

        # Update factor matrix W
        HHt = sci.dot(Ht.T, Ht)
        AHt = sci.dot(A, Ht) 


        # L2 regularization of W
        if l2_reg_W != 0.0:
            # adds l2_reg only on the diagonal
            HHt.flat[::k + 1] += l2_reg_W
        
        # L1 regularization of W
        if l1_reg_W != 0.0:
            AHt -= l1_reg_W


        if shuffle:
            permutation = rns.permutation(k)
        else:
            permutation = np.arange(k)


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

    #End for

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~                            
    # Return factor matrices
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~     
    return( W, Ht.T)
