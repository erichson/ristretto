""" 
Randomized Singular Value Decomposition
"""
# Author: N. Benjamin Erichson
# License: GNU General Public License v3.0

from __future__ import division

import numpy as np
import scipy as sci
from scipy import linalg
import scipy.sparse.linalg as scislin
 
#matrix transpose for real matricies
def rT(A): 
    return A.T
    
#matrix transpose for complex matricies
def cT(A): 
    return A.conj().T      



def rsvd_single(A, k=None, p=10, l=None, sdist='uniform'):
    """
    Randomized Singular Value Decomposition Single-View.
    
    Randomized algorithm for computing the approximate low-rank singular value 
    decomposition of a rectangular `(m, n)` matrix `A`, with target rank `k << min{m, n}`. 
    The input matrix is factored as `A = U * diag(s) * Vt`. The right singular 
    vectors are the columns of the real or complex unitary matrix `U`. The left 
    singular vectors are the columns of the real or complex unitary matrix `V`. 
    The singular values `s` are non-negative and real numbers.

    This algorithms implements a (pseudo) single pass algorithm. 
     
    
    Parameters
    ----------
    A : array_like, shape `(m, n)`.
        Real nonnegative input matrix.
    
    k : integer, `k << min{m,n}`.
        Target rank.
    
    p : integer, default: `p=10`.
        Parameter to control oversampling of column space.
        
    l : integer, default: `l=2*p`.
        Parameter to control oversampling of row space.        
                    
    sdist : str `{'uniform', 'normal', 'orthogonal'}`, default: `sdist='uniform'`.
        'uniform' : Random test matrices with uniform distributed elements.
        
        'normal' : Random test matrices with normal distributed elements.     

        'orthogonal' : Orthogonalized random test matrices with uniform distributed elements.     

    
    Returns
    -------
    U:  array_like, shape `(m, k)`.
        Right singular values.
    
    s : array_like, 1-d array of length `k`.
        Singular values. 
    
    Vt : array_like, shape `(k, n)`.
        Left singular values.


    Notes
    -----   
    * Add option for sparse random test matrices.
    
    * Modify algorithm to allow for the streaming model. 
    

    References
    ----------
    Tropp, Joel A., et al.
    "Randomized single-view algorithms for low-rank matrix approximation" (2016). 
    (available at `arXiv <https://arxiv.org/abs/1609.00048>`_).


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
    
    if m < n:
        A = fT( A )
        m , n = A.shape 
        flipped = True
    else: 
       flipped = False 
    
    if l is None:
        l = 2*p
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Generate a random test matrix Omega
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if sdist=='uniform':   
        Omega = np.array( sci.random.uniform( -1 , 1 , size=( n, k+p ) ) , dtype = dat_type ) 
        Psi = np.array( sci.random.uniform( -1 , 1 , size=( k+l, m ) ) , dtype = dat_type ) 
        if isreal==False: 
            Omega += 1j * sci.array( sci.random.uniform(-1 , 1 , size=( n, k+p  ) ) , dtype = real_type )
            Psi += 1j * sci.array( sci.random.uniform(-1 , 1 , size=( k+l, m  ) ) , dtype = real_type )
    
    elif sdist=='normal':   
        Omega = np.array( sci.random.standard_normal( size=( n, k+p  ) ) , dtype = dat_type ) 
        Psi = np.array( sci.random.standard_normal( size=( k+l, m  ) ) , dtype = dat_type )  
        if isreal==False: 
            Omega += 1j * sci.array( sci.random.standard_normal( size=( n, k+p  ) ) , dtype = real_type )     
            Psi += 1j * sci.array( sci.random.standard_normal( size=( k+l, m  ) ) , dtype = real_type )     

    elif sdist=='orthogonal':   
        Omega = np.array( sci.random.standard_normal( size=( n, k+p  ) ) , dtype = dat_type ) 
        Psi = np.array( sci.random.standard_normal( size=( k+l, m  ) ) , dtype = dat_type )  
        if isreal==False: 
            Omega += 1j * sci.array( sci.random.standard_normal( size=( n, k+p  ) ) , dtype = real_type )     
            Psi += 1j * sci.array( sci.random.standard_normal( size=( k+l, m  ) ) , dtype = real_type )  

        Omega , _ = sci.linalg.qr( Omega ,  mode='economic' , check_finite=False, overwrite_a=True ) 
        Psi , _ = sci.linalg.qr( Psi.T ,  mode='economic' , check_finite=False, overwrite_a=True ) 
        Psi = Psi.T

    else: 
        raise ValueError('Sampling distribution is not supported.')    
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Build sample matrix Y = A * Omega and W = Psi * A 
    #Note: Y should approximate the column space and W the row space of A
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
    Y = A.dot( Omega )
    W = Psi.dot(A)
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Orthogonalize Y using economic QR decomposition: Y=QR
    #Note: check_finite=False may give a performance gain
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~      
    Q , _ = sci.linalg.qr( Y ,  mode='economic' , check_finite=False, overwrite_a=True ) 
   
    U , T = sci.linalg.qr( Psi.dot(Q) ,  mode='economic' , check_finite=False, overwrite_a=False ) 

    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Form a smaller matrix
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
  
    B = sci.linalg.solve(a=T, b=fT(U).dot(W))

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Singular Value Decomposition
    #Note: B = U" * S * Vt
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~      
    #Compute SVD
    U , s , Vt = sci.linalg.svd( B ,  compute_uv=True,
                              full_matrices=False, 
                              overwrite_a=True,
                              check_finite=False)
     
    #Recover right singular vectors
    U = Q.dot(U)

    #Return Trunc
    if flipped==True:
        return ( fT( Vt )[ : , 0:k] , s[ 0:k ] , fT(U)[ 0:k, : ] ) 
    else: 
        return ( U[ : , 0:k] , s[ 0:k ] , Vt[ 0:k , : ] ) 

    

    #**************************************************************************   
    #End rsvd
    #**************************************************************************       
