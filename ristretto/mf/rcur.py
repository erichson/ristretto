""" 
CUR-ID
"""
# Author: N. Benjamin Erichson
# License: GNU General Public License v3.0

from __future__ import division

import numpy as np
import scipy as sci
from scipy import linalg
import scipy.sparse.linalg as scislin
 
from . interp_decomp import interp_decomp, rinterp_decomp


#matrix transpose for real matricies
def rT(A): 
    return A.T
    
#matrix transpose for complex matricies
def cT(A): 
    return A.conj().T    



def cur(A, k=None, index_set=False):
    """
    CUR decomposition.
    
    Algorithm for computing the low-rank CUR 
    decomposition of a rectangular `(m, n)` matrix `A`, with target rank `k << min{m, n}`. 
    Input matrix is factored as `A = C * U * R`, using the column/row pivoted QR decomposition.
    The factor matrix `C` is formed of a subset of columns of `A`, 
    also called the partial column skeleton. The factor matrix `R` is formed as
    a subset of rows of `A` also called the partial row skeleton.   
    The factor matrix `U` is formed so that `U = C**-1 * A * R**-1` is satisfied. 
 
    
    Parameters
    ----------
    A : array_like, shape `(m, n)`.
        Input matrix.
    
    k : integer, `k << min{m,n}`.
        Target rank.

    index_set: str `{'True', 'False'}`, default: `index_set='False'`.
        'True' : Return column/row index set instead of `C` and `R`.     
        
    Returns
    -------
    C:  array_like, shape `(m, k)`.
            Partial column skeleton.
        
    U : array_like, shape `(k, k)`.
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

    
    
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Compute column ID
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
    J, V = interp_decomp(A, k=k, mode='column', index_set=True)

        
    # Select column subset
    C = A[:, J]
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Compute row ID of C
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Z, I = interp_decomp(C, k=k, mode='row', index_set=True)

    # Select row subset
    R = A[I, :]

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Compute U
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    U = V.dot(sci.linalg.pinv2( R ))


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Return ID
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
    if index_set==False:
            return ( C, U, R ) 
    else:
            return ( J, U, I ) 


           





def rcur(A, k=None, p=10, q=1, index_set=False):
    """
    Randomized CUR decomposition.
    
    Randomized algorithm for computing the approximate low-rank CUR 
    decomposition of a rectangular `(m, n)` matrix `A`, with target rank `k << min{m, n}`. 
    Input matrix is factored as `A = C * U * R`, using the column/row pivoted QR decomposition.
    The factor matrix `C` is formed of a subset of columns of `A`, 
    also called the partial column skeleton. The factor matrix `R` is formed as
    a subset of rows of `A` also called the partial row skeleton.   
    The factor matrix `U` is formed so that `U = C**-1 * A * R**-1` is satisfied. 
 
    
    Parameters
    ----------
    A : array_like, shape `(m, n)`.
        Input matrix.
    
    k : integer, `k << min{m,n}`.
        Target rank.

    p : integer, default: `p=10`.
        Parameter to control oversampling.
    
    q : integer, default: `q=1`.
        Parameter to control number of power (subspace) iterations.
                  
    index_set: str `{'True', 'False'}`, default: `index_set='False'`.
        'True' : Return column/row index set instead of `C` and `R`.     
        
    Returns
    -------
    C:  array_like, shape `(m, k)`.
            Partial column skeleton.
        
    U : array_like, shape `(k, k)`.
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

    
    
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Compute column ID
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
    J, V = rinterp_decomp(A, k=k, p=p, q=q, mode='column', index_set=True)

        
    # Select column subset
    C = A[:, J]
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Compute row ID of C
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Z, I = rinterp_decomp(C, k=k, p=p, q=q,  mode='row', index_set=True)

    # Select row subset
    R = A[I, :]

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Compute U
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    U = V.dot(sci.linalg.pinv2( R ))


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Return ID
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
    if index_set==False:
            return ( C, U, R ) 
    else:
            return ( J, U, I ) 