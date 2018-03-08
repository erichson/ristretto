"""

Compressed Singular Value Decomposition

"""

from __future__ import division
import numpy as np
import scipy as sci
import random


#matrix transpose for real matricies
def rT(A): 
    return A.T
    
#matrix transpose for complex matricies
def cT(A): 
    return A.conj().T 

epsi = np.finfo(np.float64).eps





def csvd(A, k=None, p=10, sdist='sparse', formatS='csr'):
    """
    Compressed Singular Value Decomposition.

    Row compressed algorithm for computing the approximate low-rank singular value 
    decomposition of a rectangular (m, n) matrix `A` with target rank `k << n`. 
    The input matrix a is factored as `A = U * diag(s) * Vt`. The left singular 
    vectors are the columns of the real or complex unitary matrix `U`. The right 
    singular vectors are the columns of the real or complex unitary matrix `V`. 
    The singular values `s` are non-negative and real numbers.

    The parameter `c` specifies the number of measurements and is required to 
    be `c>k`.
        
    
    Parameters
    ----------
    A : array_like
        Real/complex input matrix  `A` with dimensions `(m, n)`.

    k : int
        `k` is the target rank of the low-rank decomposition, k << min(m,n). 

    c : int
        `c` sets the number of measurments.
    
    sdist : str `{gaussian', 'spixel', 'sparse'}`
        Defines the sampling distribution.

    ortho : str `{True, False}`
        If `True` the left singular values are orthonormalized.         

    method :   `{SVD, QR}` 
        Defines the method to compute the orthnormalization step.     
        
    scaled : str `{True, False}`
        If `True` the singular values are rescaled.
        
    fortmatS : str `{csr, coo}`
        Defines the format of the sparse measurement matrix. 
    
    Returns
    -------
    U:  array_like
        Right singular values, array of shape `(m, k)`.
    
    s : array_like
        Singular values, 1-d array of length `k`.
    
    Vh : array_like
        Left singular values, array of shape `(k, n)`.


    Notes
    -----   
    If the option `ortho=True` is selected, then the approximation is more
    accurate.    
    
    If the sparse sampling distribution is used, the appropriate format for
    the sparse measurement matrix is crucial. In generall `csr` is the optimal
    format, but sometimes `coo` gives a better performance. Sparse matricies 
    are computational efficient if the leading dimension is m>5000. 

    References
    ----------


    Examples
    --------


    
    """
    #*************************************************************************
    #***        Author: N. Benjamin Erichson <nbe@st-andrews.ac.uk>        ***
    #***                              <2016>                               ***
    #*************************************************************************
    
    # Shape and type of input matrix 
    A = sci.asarray(A)    
    m , n = A.shape   
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
    # Generate random measurement matrix and compress input matrix
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if sdist=='gaussian':   
        C = sci.array(sci.random.standard_normal( size=( k+p , m  ) ) , dtype=dat_type) 
        if isreal==False: 
            C += 1j * sci.array(sci.random.standard_normal( size=( k+p , m  ) ) , dtype=real_type)     
        Y = C.dot(A)
    
    elif sdist=='spixel':
        C = random.sample(range(m), k+p)
        Y =  A[ C , : ]
        
    elif sdist=='sparse':   
       # density = np.sqrt(m)
        density = m/np.log(m)
        C = sci.sparse.rand(k+p, m, density=density**-1, format=formatS, dtype=real_type, random_state=None)
        C.data = sci.array(sci.where(C.data >= 0.5, 1 , -1 ) , dtype=dat_type)
        Y = C.dot( A )    
        
    else:   
        raise ValueError('Sampling distribution is not supported.')
        
    # del(C)
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute singular value decomposition    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~   
    _ , s , Vh = sci.linalg.svd( Y, full_matrices=False,  
                                  overwrite_a=True,
                                  check_finite=True)
    
    # truncate
    if k is not None:        
        s = s[ 0:k ]
        Vh = Vh[ 0:k , : ]
    
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Recover left-singular vectors
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
    U , s , Q = sci.linalg.svd( A.dot(fT(Vh)) , full_matrices=False,  
                                  overwrite_a=True,
                                  check_finite=True)
                                  
    Vh = Q.dot(Vh)  
        

    # End if
               
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Return
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~        
    return ( U , s , Vh ) 
    
    #**************************************************************************   
    #End csvd
    #**************************************************************************       






def csvd2(A, k=None, p=10, sdist='sparse', formatS='csr'):
    """
    Compressed Singular Value Decomposition.

    Row compressed algorithm for computing the approximate low-rank singular value 
    decomposition of a rectangular (m, n) matrix `A` with target rank `k << n`. 
    The input matrix a is factored as `A = U * diag(s) * Vt`. The left singular 
    vectors are the columns of the real or complex unitary matrix `U`. The right 
    singular vectors are the columns of the real or complex unitary matrix `V`. 
    The singular values `s` are non-negative and real numbers.

    The parameter `c` specifies the number of measurements and is required to 
    be `c>k`.
        
    
    Parameters
    ----------
    A : array_like
        Real/complex input matrix  `A` with dimensions `(m, n)`.

    k : int
        `k` is the target rank of the low-rank decomposition, k << min(m,n). 

    c : int
        `c` sets the number of measurments.
    
    sdist : str `{gaussian', 'spixel', 'sparse'}`
        Defines the sampling distribution.

    ortho : str `{True, False}`
        If `True` the left singular values are orthonormalized.         

    method :   `{SVD, QR}` 
        Defines the method to compute the orthnormalization step.     
        
    scaled : str `{True, False}`
        If `True` the singular values are rescaled.
        
    fortmatS : str `{csr, coo}`
        Defines the format of the sparse measurement matrix. 
    
    Returns
    -------
    U:  array_like
        Right singular values, array of shape `(m, k)`.
    
    s : array_like
        Singular values, 1-d array of length `k`.
    
    Vh : array_like
        Left singular values, array of shape `(k, n)`.


    Notes
    -----   
    If the option `ortho=True` is selected, then the approximation is more
    accurate.    
    
    If the sparse sampling distribution is used, the appropriate format for
    the sparse measurement matrix is crucial. In generall `csr` is the optimal
    format, but sometimes `coo` gives a better performance. Sparse matricies 
    are computational efficient if the leading dimension is m>5000. 

    References
    ----------


    Examples
    --------


    
    """
    #*************************************************************************
    #***        Author: N. Benjamin Erichson <nbe@st-andrews.ac.uk>        ***
    #***                              <2016>                               ***
    #*************************************************************************
    
    # Shape and type of input matrix 
    A = sci.asarray(A)    
    m , n = A.shape   
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
    # Generate random measurement matrix and compress input matrix
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if sdist=='gaussian':   
        C = sci.array(sci.random.standard_normal( size=( k+p , m  ) ) , dtype=dat_type) 
        if isreal==False: 
            C += 1j * sci.array(sci.random.standard_normal( size=( k+p , m  ) ) , dtype=real_type)     
        Y = C.dot(A)
    
    elif sdist=='spixel':
        C = random.sample(xrange(m), k+p)
        Y =  A[ C , : ]
        
    elif sdist=='sparse':   
       # density = np.sqrt(m)
        density = m/np.log(m)
        C = sci.sparse.rand(k+p, m, density=density**-1, format=formatS, dtype=real_type, random_state=None)
        C.data = sci.array(sci.where(C.data >= 0.5, np.sqrt(density) , -np.sqrt(density) ) , dtype=dat_type)
        Y = C.dot( A )    
        
    else:   
        raise ValueError('Sampling distribution is not supported.')
        
    # del(C)
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute singular value decomposition    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~   
    #B = Y.dot(fT(Y))
    
    #s, Vh = sci.linalg.eig(Y.dot(fT(Y)), b=None, left=False, right=True, overwrite_a=True, 
    #                       overwrite_b=False, check_finite=True)    
    
    B = Y.dot(fT(Y))
    B=0.5*(B+fT(B))
    
    l = k+p
    lo, hi = (l-k), (l-1) # truncate
    s, T = sci.linalg.eigh(B, b=None, lower=True, eigvals_only=False, 
                  overwrite_a=True, overwrite_b=False, turbo=True, eigvals=None, 
                  type=1, check_finite=True)    


    # reverse the n first columns of u
    
    T[ : , :l ] = T[ : , l-1::-1 ]
    # reverse s
    s = s[ ::-1 ]
    
    # truncate
    if k is not None:        
        s = s[ xrange(k) ]
        T = T[ :, xrange(k) ]    
    
    mask = s > 0
    s[mask] = s[mask]**0.5
    
    V = fT(Y).dot(T[:,mask] * s[mask]**-1)


    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Recover left-singular vectors
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
    U , s , Vhstar = sci.linalg.svd( A.dot( V ) , full_matrices=False,  
                                  overwrite_a=True,
                                  check_finite=True)
                                  
    Vh = Vhstar.dot(fT(V))  
        
  
    # End if
               
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Return
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~        
    return ( U , s , Vh ) 
    
    #**************************************************************************   
    #End csvd
    #**************************************************************************       








def csvd_double(A, k=None, p=10, sdist='sparse', formatS='csr'):
    """
    Compressed Singular Value Decomposition.

    Row compressed algorithm for computing the approximate low-rank singular value 
    decomposition of a rectangular (m, n) matrix `A` with target rank `k << n`. 
    The input matrix a is factored as `A = U * diag(s) * Vt`. The left singular 
    vectors are the columns of the real or complex unitary matrix `U`. The right 
    singular vectors are the columns of the real or complex unitary matrix `V`. 
    The singular values `s` are non-negative and real numbers.

    The parameter `c` specifies the number of measurements and is required to 
    be `c>k`.
        
    
    Parameters
    ----------
    A : array_like
        Real/complex input matrix  `A` with dimensions `(m, n)`.

    k : int
        `k` is the target rank of the low-rank decomposition, k << min(m,n). 

    c : int
        `c` sets the number of measurments.
    
    sdist : str `{gaussian', 'spixel', 'sparse'}`
        Defines the sampling distribution.

    ortho : str `{True, False}`
        If `True` the left singular values are orthonormalized.         

    method :   `{SVD, QR}` 
        Defines the method to compute the orthnormalization step.     
        
    scaled : str `{True, False}`
        If `True` the singular values are rescaled.
        
    fortmatS : str `{csr, coo}`
        Defines the format of the sparse measurement matrix. 
    
    Returns
    -------
    U:  array_like
        Right singular values, array of shape `(m, k)`.
    
    s : array_like
        Singular values, 1-d array of length `k`.
    
    Vh : array_like
        Left singular values, array of shape `(k, n)`.


    Notes
    -----   
    If the option `ortho=True` is selected, then the approximation is more
    accurate.    
    
    If the sparse sampling distribution is used, the appropriate format for
    the sparse measurement matrix is crucial. In generall `csr` is the optimal
    format, but sometimes `coo` gives a better performance. Sparse matricies 
    are computational efficient if the leading dimension is m>5000. 

    References
    ----------


    Examples
    --------


    
    """
    #*************************************************************************
    #***        Author: N. Benjamin Erichson <nbe@st-andrews.ac.uk>        ***
    #***                              <2016>                               ***
    #*************************************************************************
    
    # Shape and type of input matrix 
    A = sci.asarray(A)    
    m , n = A.shape   
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
    # Generate random measurement matrix and compress input matrix
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Generate a random test matrix Omega
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Omega = random.sample(xrange(n), k+p)
    Psi = random.sample(xrange(m), k+p)

    L , _ = sci.linalg.qr( A[ :, Omega ] ,  mode='economic' , check_finite=False, overwrite_a=False ) 
    R , _ = sci.linalg.qr( A[ Psi, : ].T ,  mode='economic' , check_finite=False, overwrite_a=False ) 
    
        
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Project the data matrix a into a lower dimensional subspace
    #B = Q.T * A 
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
    D = fT(L).dot(A)
    D = D.dot(R)
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute singular value decomposition    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~   
    U , s , Vh = sci.linalg.svd( D, full_matrices=False,  
                                  overwrite_a=True,
                                  check_finite=True)
    
               
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Return
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~        
    return ( L.dot(U) , s , Vh.dot(R.T) ) 
    
    #**************************************************************************   
    #End csvd
    #**************************************************************************       


