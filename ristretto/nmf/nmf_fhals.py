""" 
Nonnegative Matrix Factorization
"""
# Author: N. Benjamin Erichson
# License: GNU General Public License v3.0


from __future__ import division

import numpy as np
import scipy as sci
from scipy import linalg

from .._fhals_update import _fhals_update

#import pyximport; pyximport.install()
#from _fhals_update import _fhals_update

epsi = np.finfo(np.float32).eps

def nmf_fhals(A, k, init='normal', tol=1e-4, maxiter=100, verbose=False):
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
    This implementation mirrors the scikit-learn implementation of the
    NMF. See: https://github.com/scikit-learn/scikit-learn
    
    
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
    >>> import dimly.nmf.nmf_fhals as nmf
    >>> W, H = nmf(X, k=2, p=0)
    
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
    # Initialization methods for factor matrices W and H
    # 'normal': nonnegative standard normal random init
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
    
    if init == 'normal':
        m, n = A.shape
        W = sci.maximum(0.0, sci.random.standard_normal((m, k)))
        Ht = sci.maximum(0.0, sci.random.standard_normal((n, k)))
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
    
    for niter in range(maxiter): 
        violation = 0.0
        
        # Update factor matrix H
        WtW = W.T.dot(W)
        AtW = A.T.dot(W)
        
        violation += _fhals_update(Ht, WtW, AtW)                        
        Ht /= sci.maximum(epsi, sci.linalg.norm(Ht, axis=0))

        # Update factor matrix W
        HHt = Ht.T.dot(Ht)
        AHt = A.dot(Ht) # Rotate AHt back to high-dimensional space

        violation += _fhals_update(W, HHt, AHt)
        
        
        # Compute stopping condition.
        if niter == 0:
            violation_init = violation

        if violation_init == 0:
            break       

        fitchange = violation / violation_init
        
        if verbose == True:
            print('Iteration: %s fit: %s, fitchange: %s' %(niter, violation, fitchange))        

        if fitchange <= tol:
            break

    #End for

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~                            
    # Return factor matrices
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
    if verbose == True:
        print('Final Iteration: %s fit: %s' %(niter, violation)) 
        
    return( W, Ht.T )