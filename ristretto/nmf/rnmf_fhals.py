""" 
Nonnegative Matrix Factorization
"""
# Author: N. Benjamin Erichson
# License: GNU General Public License v3.0


from __future__ import division

import numpy as np
import scipy as sci
from scipy import linalg

from .._rfhals_update import _rfhals_update

#import pyximport; pyximport.install()
#from _rfhals_update import _rfhals_update

epsi = np.finfo(np.float32).eps

def rnmf_fhals(A, k, p=20, q=2, init='normal', tol=1e-4, maxiter=100, verbose=False):
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
        Target rank.
    
    p : integer, default: `p=20`.
        Parameter to control oversampling.
    
    q : integer, default: `q=2`.
        Parameter to control number of power (subspace) iterations.
    
    init : str `{'normal'}`. 
        'normal' : Factor matrices are initialized with nonnegative 
                   Gaussian random numbers.
            
    tol : float, default: `tol=1e-4`.
        Tolerance of the stopping condition.
        
    maxiter : integer, default: `maxiter=100`.
        Number of iterations.   
        
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
    This implementation is based on the scikit-learn implementation of the
    deterministic NMF. See: https://github.com/scikit-learn/scikit-learn
    
    
    References
    ----------
    Cichocki, Andrzej, and P. H. A. N. Anh-Huy. "Fast local algorithms for
    large scale nonnegative matrix and tensor factorizations."
    IEICE transactions on fundamentals of electronics, communications and
    computer sciences 92.3: 708-721, 2009.
    
    
    Examples
    --------    
    >>> import numpy as np
    >>> X = np.array([[1,1], [2, 1], [3, 1.2], [4, 1], [5, 0.8], [6, 1]])
    >>> import dimly.nmf.rnmf_fhals as rnmf
    >>> W, H = rnmf(X, k=2, p=0)
    
    """    
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Error catching
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~     
    m, n = A.shape  
    
    if (A < 0).any():
        raise ValueError("Input matrix with nonnegative elements is required.")    
    
    if  A.dtype == sci.float32: 
        data_type = sci.float32
        
    elif A.dtype == sci.float64: 
        data_type = sci.float64  

    else:
        raise ValueError("A.dtype is not supported.")    
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute QB decomposition
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~            

    #Build sample matrix Y : Y = A * Omega, where Omega is a random test matrix 
    Omega = sci.array( sci.random.rand( n, k+p ) , dtype = data_type ) 
    Y = A.dot( Omega )
    
    #If q > 0 perfrom q subspace iterations  
    if q > 0:
        for i in range(q):
            Y , _ = sci.linalg.qr( Y, mode='economic', check_finite=False, overwrite_a=True)
            Z , _ = sci.linalg.qr( A.T.dot(Y), mode='economic', check_finite=False, overwrite_a=True)
            Y = A.dot( Z )
        #End for
     #End if       
    
    #Orthogonalize Y using economic QR decomposition: Y = QR          
    Q , _ = sci.linalg.qr( Y,  mode='economic', check_finite=False, overwrite_a=True) 
 
    #Project input data to low-dimensional space
    A = Q.T.dot( A )                        

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~                            
    # Initialization methods for factor matrices W and H
    # 'normal': nonnegative standard normal random init
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
    
    if init == 'normal':
        m, n = A.shape
        W = sci.maximum(0.0, sci.random.standard_normal((m, k)))
        Ht = sci.maximum(0.0, sci.random.standard_normal((n, k)))
        #Ht /= sci.linalg.norm(Ht, axis=0) 
    else:
        raise ValueError('Initialization method is not supported.')
    #End if
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Iterate the HALS algorithm until convergence or maxiter is reached
    # i)   Update factor matrix H and normalize columns   
    # ii)  Update low-dimensional factor matrix W
    # iii) Compute fit log( ||A-WH|| )
    #   -> break if fit <-5 or fit_change < tol
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
    Wtemp = np.empty(W.shape)
    
    for niter in range(maxiter):

        # Pointer to updated factor matrix 
        if niter != 0: W = Wtemp    

        # Update factor matrix H
        WtW = W.T.dot(W)
        AtW = A.T.dot(W)
        
        _ = _rfhals_update(Ht, WtW, AtW)                        
        Ht /= sci.maximum(epsi, sci.linalg.norm(Ht, axis=0))

        # Update factor matrix W
        HHt = Ht.T.dot(Ht)
        AHt = Q.dot(A.dot(Ht)) # Rotate AHt back to high-dimensional space

        W = Q.dot(W) # Rotate W back to high-dimensional space
        _ = _rfhals_update(W, HHt, AHt)
        
        # Project W to low-dimensional space
        Wtemp = Q.T.dot(W)  
        
        # Compute stopping condition.
        if niter % 10 == 0 and verbose == True:
            fit = np.log10(sci.linalg.norm(A - Wtemp.dot( Ht.T )))

            if niter == 0: fitold = fit               
            
            fitchange = abs(fitold - fit)
            fitold = fit
            
            if verbose == True:
                print('Iteration: %s fit: %s, fitchange: %s' %(niter, fit, fitchange))        
            #End if        
 
            if niter > 1 and (fit <= -5 or fitchange <= tol):      
                break       
            #End if 
        #End if                  
    #End for

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~                            
    # Return factor matrices
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
    if verbose == True:
        fit = np.log10(sci.linalg.norm(A - (Q.T.dot(W)).dot( Ht.T)))
        print('Final Iteration: %s fit: %s' %(niter, fit)) 
        
    return( W, Ht.T )