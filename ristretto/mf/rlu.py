""" 
Randomized LU Decomposition
"""
# Author: N. Benjamin Erichson
# License: GNU General Public License v3.0

from __future__ import division

import numpy as np
import scipy as sci
from scipy import linalg
from scipy import sparse
 
#matrix transpose for real matricies
def rT(A): 
    return A.T
    
#matrix transpose for complex matricies
def cT(A): 
    return A.conj().T      



def rlu(A, permute=False, k=None, p=10, q=1, sdist='uniform'):
    """
    Randomized LU Decomposition.
    
    Randomized algorithm for computing the approximate low-rank LU 
    decomposition of a rectangular `(m, n)` matrix `A`, with target rank `k << min{m, n}`. 
    The input matrix is factored as `A = P * L * U * C`, where
    `L` and `U` are the lower and upper triangular matrices, respectively. 
    And `P` and `C` are the row and column permutation matrices.

    The quality of the approximation can be controlled via the oversampling 
    parameter `p` and the parameter `q` which specifies the number of 
    subspace iterations. 
        
    
    Parameters
    ----------
    A : array_like, shape `(m, n)`.
        Real nonnegative input matrix.
        
    permute : bool, default: `permute=False`.
        If `True`, perform the multiplication P*L and U*C.
        
    k : integer, `k << min{m,n}`.
        Target rank.
    
    p : integer, default: `p=10`.
        Parameter to control oversampling.
    
    q : integer, default: `q=1`.
        Parameter to control number of power (subspace) iterations.
                  
    sdist : str `{'uniform', 'normal'}`, default: `sdist='uniform'`.
        'uniform' : Random test matrix with uniform distributed elements.
        
        'normal' : Random test matrix with normal distributed elements.     

    
    Returns
    -------
    P : array_like, shape `(m, m)`.
        Row permutation matrix, if `permute_l=False`.
    
    L :  array_like, shape `(m, k)`.
        Lower triangular matrix.
    
    U : array_like, shape `(k, n)`.
        Upper triangular matrix. 

    C : array_like, shape `(n, n)`.
        Column Permutation matrix, if `permute=False`.

    Notes
    -----   
    


    References
    ----------
    Halko, Nathan, Per-Gunnar Martinsson, and Joel A. Tropp.
    "Finding structure with randomness: Probabilistic algorithms for
    constructing approximate matrix decompositions." 
    SIAM review 53.2 (2011): 217-288.
    (available at `arXiv <http://arxiv.org/abs/0909.4061>`_).
    
    Shabat, Gil, et al. 
    "Randomized LU decomposition." 
    Applied and Computational Harmonic Analysis (2016).
    (available at `arXiv <https://arxiv.org/abs/1310.7202>`_).
    
    
    Examples
    --------


    
    """

    # Shape of input matrix 
    m , n = A.shape   
    dat_type =  A.dtype   

    if  dat_type == sci.float32: 
        isreal = True
        real_type = sci.float32
        fT = rT
    elif dat_type == sci.float64: 
        isreal = True
        real_type = sci.float64  
        fT = rT
    elif dat_type == sci.complex64:
        isreal = False 
        real_type = sci.float32
        fT = cT
    elif dat_type == sci.complex128:
        isreal = False 
        real_type = sci.float64
        fT = cT
    else:
        raise ValueError( "A.dtype is not supported" )
    
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Generate a random test matrix Omega
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if sdist=='uniform':   
        Omega = np.array( sci.random.uniform( -1 , 1 , size=( n, k+p ) ) , dtype = dat_type ) 
        if isreal==False: 
            Omega += 1j * sci.array( sci.random.uniform(-1 , 1 , size=( n, k+p  ) ) , dtype = real_type )
      
    elif sdist=='normal':   
        Omega = np.array( sci.random.standard_normal( size=( n, k+p  ) ) , dtype = dat_type ) 
        if isreal==False: 
            Omega += 1j * sci.array( sci.random.standard_normal( size=( n, k+p  ) ) , dtype = real_type )     

    else: 
        raise ValueError('Sampling distribution is not supported.')    
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Build sample matrix Y : Y = A * Omega
    #Note: Y should approximate the range of A
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
    Y = A.dot( Omega )
    del( Omega )
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Orthogonalize Y using economic QR decomposition: Y=QR
    #If q > 0 perfrom q subspace iterations
    #Note: check_finite=False may give a performance gain
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~      
    s=1 #control parameter for number of orthogonalizations
    if q > 0:
        for i in np.arange( 1, q+1 ):
            if( (2*i-2) % s == 0 ):
                Y , _ = sci.linalg.qr( Y , mode='economic', check_finite=False, overwrite_a=True )
                        
            if( (2*i-1) % s == 0 ):
                Z , _ = sci.linalg.qr( fT( A ).dot( Y ) , mode='economic', check_finite=False, overwrite_a=True)
       
            Y = A.dot( Z )
        #End for
        del(Z)
     #End if       
        
    Q , _ = sci.linalg.qr( Y ,  mode='economic' , check_finite=False, overwrite_a=True ) 
    del(Y)


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute pivoted LU decompostion of the orthonormal basis matrix Q.
    # Q = P * L * U 
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 

    P, L_tilde, _ = sci.linalg.lu(Q, permute_l=False, overwrite_a=True, check_finite=True)

    _, r ,_ = sci.sparse.find(P.T)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Truncate L_tilde
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  

    L_tilde = L_tilde[:, 0:k]

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Form smaller matrix B
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
    #B, _, _, _ = np.linalg.lstsq(a=L_tilde, b=A[r,:] ) 
    
    U, s, Vt = sci.linalg.svd( L_tilde ,  compute_uv=True,full_matrices=False, 
                              overwrite_a=False, check_finite=False)
    
    B = (fT(Vt)*s**-1).dot(fT(U)).dot(A[r,:])
    
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute LU decompostion of B. 
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
    C, L, U = sci.linalg.lu(fT(B), permute_l=False, overwrite_a=True, check_finite=True)
    
 
    #Return
    if permute == False:
        #_, r ,_ = sci.sparse.find(P)
        #_,c,_ = sci.sparse.find(C.T)        
        return  ( P, L_tilde.dot(fT(U)), fT(L), fT(C)) 
    
    else:
        _, r ,_ = sci.sparse.find(P)
        _, c ,_ = sci.sparse.find(C)

        return ( L_tilde.dot(fT(U))[r,:], fT(L)[:,c] ) 

           
