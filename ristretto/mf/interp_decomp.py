""" 
Interpolative decomposition (ID)
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



def interp_decomp(A, k=None, mode='column', index_set=False):
    """
    Interpolative decomposition (ID).
    
    Algorithm for computing the low-rank ID 
    decomposition of a rectangular `(m, n)` matrix `A`, with target rank `k << min{m, n}`. 
    Input matrix is factored as `A = C * V`, using the column pivoted QR decomposition.
    The factor matrix `C` is formed of a subset of columns of `A`, 
    also called the partial column skeleton. The factor matrix `V` contains 
    a `(k, k)` identity matrix as a submatrix, and is well-conditioned. 

    If `mode='row'`, then the input matrix is factored as `A = Z * R`, using the 
    row pivoted QR decomposition. The factor matrix `R` is now formed as
    a subset of rows of `A`, also called the partial row skeleton.      
    
    Parameters
    ----------
    A : array_like, shape `(m, n)`.
        Input matrix.
    
    k : integer, `k << min{m,n}`.
        Target rank.

    mode: str `{'column', 'row'}`, default: `mode='column'`.
        'column' : ID using column pivoted QR.
        'row' : ID using row pivoted QR.

    index_set: str `{'True', 'False'}`, default: `index_set='False'`.
        'True' : Return column/row index set instead of `C` or `R`.     
        
    Returns
    -------
    If `mode='column'`:
        C:  array_like, shape `(m, k)`.
            Partial column skeleton.
        
        V : array_like, shape `(k, n)`.
            Well-conditioned matrix. 

    If `mode='row'`:
        Z:  array_like, shape `(m, k)`.
            Well-conditioned matrix.
        
        R : array_like, shape `(k, n)`.
            Partial row skeleton.


    Notes
    -----   


    References
    ----------    
    S. Voronin and P.Martinsson. 
    "RSVDPACK: Subroutines for computing partial singular value 
    decompositions via randomized sampling on single core, multi core, 
    and GPU architectures" (2015).
    (available at `arXiv <http://arxiv.org/abs/1502.05366>`_).


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
    
    if k is None:
        raise ValueError( "Target rank k is required." )

    if k < 0:
        raise ValueError( "Target rank k not valid." )
        
    if k > min(m,n):
        k = min(m,n)  

    
    if mode=='row':
        A = rT(A)
        m,n = A.shape
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Pivoted QR decomposition
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
    Q, R, P = sci.linalg.qr( A ,  mode='economic', overwrite_a=False, pivoting=True) 
        
    # Select column subset
    C = A[:,P[0:k]]
    
    # Compute V
    T =  sci.linalg.pinv2(R[0:k , 0:k]).dot(R[0:k , k:n]) 
    V = sci.bmat([[np.eye(k), T]])
    V = V[:,sci.argsort(P)]

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Return ID
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
    if mode=='column':
        if index_set==False:
            return ( C, V ) 
        else:
            return ( P[0:k], V  ) 

    if mode=='row':
        if index_set==False:
            return ( rT(V), rT(C) ) 
        else:
            return ( rT(V), P[0:k]  )
           



def rinterp_decomp(A, k=None, mode='column', p=10, q=1, sdist='normal', index_set=False):
    """
    Randomized interpolative decomposition (rID).
    
    Algorithm for computing the approximate low-rank ID 
    decomposition of a rectangular `(m, n)` matrix `A`, with target rank `k << min{m, n}`. 
    The input matrix is factored as `A = C * V`. The factor matrix `C`is formed 
    of a subset of columns of `A`, also called the partial column skeleton. 
    The factor matrix `V`contains a `(k, k)` identity matrix as a submatrix,
    and is well-conditioned. 
    
    If `mode='row'`, then the input matrix is factored as `A = Z * R`, using the 
    row pivoted QR decomposition. The factor matrix `R` is now formed as
    a subset of rows of `A`, also called the partial row skeleton.      

    The quality of the approximation can be controlled via the oversampling 
    parameter `p` and the parameter `q` which specifies the number of 
    subspace iterations. 
        
    
    Parameters
    ----------
    A : array_like, shape `(m, n)`.
        Input matrix.
    
    k : integer, `k << min{m,n}`.
        Target rank.

    mode: str `{'column', 'row'}`, default: `mode='column'`.
        'column' : Column ID.
        'row' : Row ID. 
    
    p : integer, default: `p=10`.
        Parameter to control oversampling.
    
    q : integer, default: `q=1`.
        Parameter to control number of power (subspace) iterations.
                  
    sdist : str `{'uniform', 'normal'}`, default: `sdist='uniform'`.
        'uniform' : Random test matrix with uniform distributed elements.
        
        'normal' : Random test matrix with normal distributed elements.     

    index_set: str `{'True', 'False'}`, default: `index_set='False'`.
        'True' : Return column/row index set instead of `C` or `R`. 
    
    Returns
    -------
    If `mode='column'`:
        C:  array_like, shape `(m, k)`.
            Partial column skeleton.
        
        V : array_like, shape `(k, n)`.
            Well-conditioned matrix. 
        
    If `mode='row'`:
        Z:  array_like, shape `(m, k)`.
            Well-conditioned matrix.
        
        R : array_like, shape `(k, n)`.
            Partial row skeleton.        


    Notes
    -----   
    


    References
    ----------
    S. Voronin and P.Martinsson. 
    "RSVDPACK: Subroutines for computing partial singular value 
    decompositions via randomized sampling on single core, multi core, 
    and GPU architectures" (2015).
    (available at `arXiv <http://arxiv.org/abs/1502.05366>`_).


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
    
    if k is None:
        raise ValueError( "Target rank k is required." )

    if k < 0:
        raise ValueError( "Target rank k not valid." )
        
    if k > min(m,n):
        k = min(m,n)  
 

    if mode=='row':
        A = rT(A)
        m,n = A.shape
       
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Generate a random test matrix Omega
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if sdist=='uniform':   
        Omega = np.array( sci.random.uniform( -1 , 1 , size=( k+p, m ) ) , dtype = dat_type ) 
        if isreal==False: 
            Omega += 1j * sci.array( sci.random.uniform(-1 , 1 , size=( k+p, m  ) ) , dtype = dat_type )
      
    elif sdist=='normal':   
        Omega = np.array( sci.random.standard_normal( size=( k+p, m  ) ) , dtype = dat_type ) 
        if isreal==False: 
            Omega += 1j * sci.array( sci.random.standard_normal( size=( k+p, m  ) ) , dtype = dat_type )     

    else: 
        raise ValueError('Sampling distribution is not supported.')    
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Build sample matrix Y : Y = A * Omega
    #Note: Y should approximate the range of A
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
    Y = Omega.dot(A)
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
                Y , _ = sci.linalg.qr( fT(Y) , mode='economic', check_finite=False, overwrite_a=True )
                        
            if( (2*i-1) % s == 0 ):
                Z , _ = sci.linalg.qr( A.dot(Y)  , mode='economic', check_finite=False, overwrite_a=True)
       
            Y = fT(Z).dot(A)
        #End for
        del(Z)
     #End if       
        
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Deterministic ID
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
    J, V = interp_decomp(Y, k=k, mode='column', index_set=True)
    
    J = J[0:k]     
   
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Return ID
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
    
    if mode=='column':
        if index_set==False:
            return ( A[:,J], V ) 
        else:
            return ( J, V ) 

    if mode=='row':
        if index_set==False:
            return ( rT(V), rT(A[:,J]) ) 
        else:
            return ( rT(V), J )
           







def rinterp_decomp_qb(A, k=None, mode='column', p=10, q=1, sdist='normal', index_set=False):
    """
    Randomized interpolative decomposition (rID).
    
    Algorithm for computing the approximate low-rank ID 
    decomposition of a rectangular `(m, n)` matrix `A`, with target rank `k << min{m, n}`. 
    The input matrix is factored as `A = C * V`. The factor matrix $\mathbf{C}$ is formed 
    of a subset of columns of $\mathbf{A}$, also called the partial column skeleton. 
    The factor matrix $\mathbf{V}$ contains a $k\times k$ identity matrix as a submatrix,
    and is well-conditioned. 
    
    If `mode='row'`, then the input matrix is factored as `A = Z * R`, using the 
    row pivoted QR decomposition. The factor matrix $\mathbf{C}$ is now formed as
    a subset of rows of $\mathbf{A}$, also called the partial row skeleton.      

    The quality of the approximation can be controlled via the oversampling 
    parameter `p` and the parameter `q` which specifies the number of 
    subspace iterations. 
        
    
    Parameters
    ----------
    A : array_like, shape `(m, n)`.
        Input matrix.
    
    k : integer, `k << min{m,n}`.
        Target rank.

    mode: str `{'column', 'row'}`, default: `mode='column'`.
        'column' : Column ID.
        'row' : Row ID. 
    
    p : integer, default: `p=10`.
        Parameter to control oversampling.
    
    q : integer, default: `q=1`.
        Parameter to control number of power (subspace) iterations.
                  
    sdist : str `{'uniform', 'normal'}`, default: `sdist='uniform'`.
        'uniform' : Random test matrix with uniform distributed elements.
        
        'normal' : Random test matrix with normal distributed elements.     

    index_set: str `{'True', 'False'}`, default: `index_set='False'`.
        'True' : Return column/row index set.
    
    Returns
    -------
    If `mode='column'`:
        C:  array_like, shape `(m, k)`.
            Partial column skeleton.
        
        V : array_like, shape `(k, n)`.
            Well-conditioned matrix. 
        
    If `mode='row'`:
        Z:  array_like, shape `(m, k)`.
            Well-conditioned matrix.
        
        R : array_like, shape `(k, n)`.
            Partial row skeleton.        
        

    J : array_like, shape `(k, n)`.
        Column/row index set.

    Notes
    -----   
    


    References
    ----------
    S. Voronin and P.Martinsson. 
    "RSVDPACK: Subroutines for computing partial singular value 
    decompositions via randomized sampling on single core, multi core, 
    and GPU architectures" (2015).
    (available at `arXiv <http://arxiv.org/abs/1502.05366>`_).


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
    
    if k is None:
        raise ValueError( "Target rank k is required." )

    if k < 0:
        raise ValueError( "Target rank k not valid." )
        
    if k > min(m,n):
        k = min(m,n)  
 

    if mode=='row':
        A = rT(A)
        m,n = A.shape
       
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Generate a random test matrix Omega
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if sdist=='uniform':   
        Omega = np.array( sci.random.uniform( -1 , 1 , size=( n, k+p ) ) , dtype = dat_type ) 
        if isreal==False: 
            Omega += 1j * sci.array( sci.random.uniform(-1 , 1 , size=( n, k+p ) ) , dtype = dat_type )
      
    elif sdist=='normal':   
        Omega = np.array( sci.random.standard_normal( size=( n, k+p  ) ) , dtype = dat_type ) 
        if isreal==False: 
            Omega += 1j * sci.array( sci.random.standard_normal( size=( n, k+p  ) ) , dtype = dat_type )     

    else: 
        raise ValueError('Sampling distribution is not supported.')    
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Build sample matrix Y : Y = A * Omega
    #Note: Y should approximate the range of A
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
    Y = A.dot(Omega)
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
     #End if       
        
    Q , _ = sci.linalg.qr( Y ,  mode='economic' , check_finite=False, overwrite_a=True ) 
    del(Y)    

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Project the data matrix a into a lower dimensional subspace
    #B = Q.T * A 
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
    B = fT( Q ).dot( A )         
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Deterministic ID
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
    J, V = interp_decomp(B, k=k, mode='column', index_set=True)
    
    J = J[0:k]   
 
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Return ID
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
    if mode=='column':
        if index_set==False:
            return ( A[:,J], V ) 
        else:
            return ( J, V ) 

    if mode=='row':
        if index_set==False:
            return ( rT(V), rT(A[:,J]) ) 
        else:
            return ( rT(V), J  )