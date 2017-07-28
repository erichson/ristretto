"""
Randomized Dynamic Mode Decomposition (DMD).

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



def rdmd(A, dt = 1, k=None, p=10, q=2, sdist='uniform',
         return_amplitudes=False, return_vandermonde=False, order=True):
    """
    Randomized Dynamic Mode Decomposition.

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
    
    k : int
        If `k < (n-1)` low-rank Dynamic Mode Decomposition is computed.
    
    p : int, optional
        Oversampling paramater.         
    
    sdist : str `{'uniform', 'normal'}`  
        Specify the distribution of the sensing matrix `S`. 
    
    
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
  

    
    Examples
    --------



    
    """
 
    #Shape of A
    m, n = A.shape   
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
        raise ValueError('A.dtype is not supported')


    if k > min(m,n):
        k = min(m,n)       
    if k is None:
        raise ValueError('Target rank needs to be specified.')

    
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
     #End if       
        
    Q , _ = sci.linalg.qr( Y ,  mode='economic' , check_finite=False, overwrite_a=True ) 
    del(Y)
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Project the data matrix a into a lower dimensional subspace
    #B = Q.T * A 
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
    B = fT( Q ).dot( A )   



    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Split data into lef and right snapshot sequence
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
    X = B[ : , 0:(n-1)] #pointer
    Y = B[ : , 1:n] #pointer   
     
     
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Singular Value Decomposition
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~        
    U, s, Vh = sci.linalg.svd( X ,  compute_uv=True,
                                  full_matrices=False, 
                                  overwrite_a=False,
                                  check_finite=True)
    U = U[ : , 0:k ]
    s = s[ 0:k  ]
    Vh = Vh[ 0:k  , : ]

    #EndIf    
     
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Solve the LS problem to find estimate for M using the pseudo-inverse    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
    #real: M = U.T * Y * Vt.T * S**-1
    #complex: M = U.H * Y * Vt.H * S**-1
    #Let G = Y * Vt.H * S**-1, hence M = M * G
    Vscaled = fT(Vh) * s**-1
    G = np.dot( Y , Vscaled )
    M = np.dot( fT(U) , G ) 
     
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Eigen Decomposition
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    l, W = sci.linalg.eig( M , right=True, overwrite_a=True )    
 
    omega = np.log(l) / dt
       
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Order
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if order==True: 
        sort_idx = sci.argsort(np.abs(omega)) 
        W = W[  :, sort_idx ]
        l = l[ sort_idx ] 
        omega = omega[ sort_idx ]  
          
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
    #Compute DMD Modes 
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
    F = Q.dot( U.dot( W))    


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Compute amplitueds b using least-squares: Fb=x1
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~     
    if return_amplitudes==True:   
        b , _ , _ , _ = sci.linalg.lstsq( F , A[ : , 0 ])


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Compute Vandermonde matrix (CPU)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if return_vandermonde==True: 
        V = sci.fliplr(sci.vander( l , N =  n ))     
        

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Return
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
    
    if return_amplitudes==True and return_vandermonde==True:
        return F, b, V, omega
    elif return_amplitudes==True and return_vandermonde==False:
        return F, b, omega
    else:
        return F , omega   
  
    #**************************************************************************   
    #End rDMD
    #**************************************************************************  

