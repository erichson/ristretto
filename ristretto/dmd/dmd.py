"""

Dynamic Mode Decomposition (DMD) python function.

"""
# Author: N. Benjamin Erichson
# License: GNU General Public License v3.0

from __future__ import division

import numpy as np
import scipy as sci
from scipy import linalg
 

#matrix transpose for real matricies
def rT(A): 
    return A.T
    
#matrix transpose for complex matricies
def cT(A): 
    return A.conj().T      


def dmd(A, dt = 1, k=None, modes='exact', return_amplitudes=False, return_vandermonde=False, order=True):
    """
    Dynamic Mode Decomposition.

    Dynamic Mode Decomposition (DMD) is a data processing algorithm which
    allows to decompose a matrix `A` in space and time. The matrix `A` is 
    decomposed as `A = F * B * V`, where the columns of `F` contain the dynamic modes.
    The modes are ordered corresponding to the amplitudes stored in the diagonal 
    matrix `B`. `V` is a Vandermonde matrix describing the temporal evolution.


    Parameters
    ----------
    A : array_like
        Real/complex input matrix  `a` with dimensions `(m, n)`.
    
    dt : scalar or array_like  
        Factor specifying the time difference between the observations.      
    
    k : int, optional
        If `k < (n-1)` low-rank Dynamic Mode Decomposition is computed.
    
    modes : str `{'standard', 'exact', 'exact_scaled'}`
        'standard' : uses the standard definition to compute the dynamic modes, `F = U * W`.
        
        'exact' : computes the exact dynamic modes, `F = Y * V * (S**-1) * W`.    
        
        'exact_scaled' : computes the exact dynamic modes, `F = (1/l) * Y * V * (S**-1) * W`.
    
    return_amplitudes : bool `{True, False}` 
        True: return amplitudes in addition to dynamic modes. 
    
    return_vandermonde : bool `{True, False}`
        True: return Vandermonde matrix in addition to dynamic modes and amplitudes.
            
    order :  bool `{True, False}`
        True: return modes sorted.


    Returns
    -------
    F : array_like
        Matrix containing the dynamic modes of shape `(m, n-1)`  or `(m, k)`.
    
    b : array_like, if `return_amplitudes=True`
        1-D array containing the amplitudes of length `min(n-1, k)`.
    
    V : array_like, if `return_vandermonde=True`
        Vandermonde matrix of shape `(n-1, n-1)`  or `(k, n-1)`.

    omega : array_like
        Time scaled eigenvalues: `ln(l)/dt`. 



    Notes
    -----


    References
    ----------
    J. H. Tu, et al.
    "On Dynamic Mode Decomposition: Theory and Applications" (2013).
    (available at `arXiv <http://arxiv.org/abs/1312.0041>`_).   
    
    N. B. Erichson and C. Donovan.
    "Randomized Low-Rank Dynamic Mode Decomposition for Motion Detection" (2015).
    Under Review.    
    
    
    Examples
    --------


    """

 
    #Shape of D
    m, n = A.shape   
    dat_type =  A.dtype
    if  dat_type == np.float32: 
        isreal = True
        real_type = np.float32
        fT = rT
    elif dat_type == np.float64: 
        isreal = True
        real_type = np.float64  
        fT = rT
    elif dat_type == np.complex64:
        isreal = False 
        real_type = np.float32
        fT = cT
    elif dat_type == np.complex128:
        isreal = False 
        real_type = np.float64
        fT = cT
    else:
        raise ValueError('A.dtype is not supported')
        
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Split data into lef and right snapshot sequence
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    X = A[ : , 0:(n-1)] #pointer
    Y = A[ : , 1:n] #pointer   
     
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Singular Value Decomposition
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~        

    U, s, Vh = sci.linalg.svd( X ,  compute_uv=True,
                                  full_matrices=False, 
                                  overwrite_a=False,
                                  check_finite=True)
                                
    if k != None:
            U = U[ : , 0:k ]
            s = s[ 0:k ]
            Vh = Vh[ 0:k , : ]
            
            
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Solve the LS problem to find estimate for M using the pseudo-inverse    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
    #real: M = U.T * Y * Vt.T * S**-1
    #complex: M = U.H * Y * Vt.H * S**-1
    #Let G = Y * Vt.H * S**-1, hence M = M * G

    Vscaled = fT(Vh)  * s**-1
    G = np.dot( Y , Vscaled ) 
    M = np.dot( fT(U), G )
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Eigen Decomposition
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    l, W = sci.linalg.eig( M , right=True, overwrite_a=True )    

    omega = np.log(l) / dt
 
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Order
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if order==True: 
        sort_idx = np.argsort(np.abs(omega))
        W = W[  :, sort_idx ]
        l = l[ sort_idx ] 
        omega = omega[ sort_idx ]  
 
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
    #Compute DMD Modes 
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
    if modes=='standard': 
        F = np.dot( U , W )    
    elif modes=='exact': 
        F = np.dot( G , W )
    elif modes=='exact_scaled':     
        F = np.dot((1/l) * G , W )
    else: 
        raise ValueError('Type of modes is not supported, choose "exact" or "standard".')
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Compute amplitueds b using least-squares: Fb=x1
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~     
    if return_amplitudes==True:   
        b , _ , _ , _ = sci.linalg.lstsq( F , A[ : , 0 ])


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Compute Vandermonde matrix
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if return_vandermonde==True: 
        V = np.fliplr(np.vander( l , N =  n ))     
        

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Return 
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  
    if return_amplitudes==True and return_vandermonde==True:
        return F, b, V, omega
    elif return_amplitudes==True and return_vandermonde==False:
        return F, b, omega
    else:
        return F, omega
  
    #**************************************************************************   
    #End dmd
    #**************************************************************************  
     
    
