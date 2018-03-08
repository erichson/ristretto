import numpy as np
import scipy as sci

from ristretto.mf import rsvd, rqb



def spca(X, n_components=None, alpha = 0.1, beta = 0.01, 
         max_iter = 500, tol = 1e-5, verbose = True):

    """
    Sparse Principal Component Analysis (SPCA).
    
    Given a mean centered rectangular matrix `A` with shape `(m, n)`, SPCA 
    computes a set of sparse components that can optimally reconstruct the 
    input data.  The amount of sparseness is controllable by the coefficient
    of the L1 penalty, given by the parameter alpha. In addition, some ridge 
    shrinkage can be applied in order to improve conditioning.   
    
    
    
    Parameters
    ----------
    A : array_like, shape `(m, n)`.
        Real nonnegative input matrix.
    
    n_components : integer, `n_components << min{m,n}`.
        Target rank, i.e., number of sparse components to be computed.
        
    alpha : float, (default ``alpha = 0.1``).
        Sparsity controlling parameter. Higher values lead to sparser components.
    
    beta : float, (default ``beta = 0.1``).
        Amount of ridge shrinkage to apply in order to improve conditionin.

    max_iter : integer, (default ``max_iter = 500``).
        Maximum number of iterations to perform before exiting.
            
    tol : float, (default ``tol = 1e-5``).
        Stopping tolerance for reconstruction error.
            
    verbose : bool ``{'True', 'False'}``, optional (default ``verbose = True``).
        Display progress.
    
    
    Returns
    -------
    B:  array_like, `(n, n_components)`.
        Sparse components extracted from the data.
    
    A : array_like, `(n, n_components)`.
        Orthogonal components extracted from the data.

    eigvals : array_like, `(n_components)`.
        Eigenvalues correspnding to the extracted components. 
        
    obj : array_like, `(n_iter)`.
        Objective value at the i-th iteration.         
        
    Notes
    -----   
    Variable Projection for SPCA solves the following optimization problem:
    minimize 1/2⋅‖X - X⋅B⋅Aᵀ‖² + α⋅‖B‖₁ + 1/2⋅β‖B‖² 
    
    
    References
    ----------

    
    
    Examples
    --------    

    
    """        
    
    
    # Shape of input matrix
    m, n = X.shape

    #--------------------------------------------------------------------
    #   Initialization of Variable Projection Solver
    #--------------------------------------------------------------------    
    _, D, Vt = sci.linalg.svd( X , full_matrices=False, overwrite_a=False)
    
    Dmax = D[0] # l2 norm

    A = Vt.T[:, 0:n_components]
    B = Vt.T[:, 0:n_components]
    
 
    #--------------------------------------------------------------------
    #   Set Tuning Parameters
    #--------------------------------------------------------------------  
    alpha *= Dmax**2
    beta *= Dmax**2
    
    noi = 0
    nu   = 1.0 / (Dmax**2 + beta)
    kappa = nu * alpha
        
    
    obj = []
    
    #--------------------------------------------------------------------
    #   Apply Variable Projection Solver
    #--------------------------------------------------------------------  
    while max_iter > noi:
  
        # Update A: 
        # X'XB = UDV'
        # Compute X'XB via SVD of X
        #Z = XtX.dot(B)
        Z = (Vt.T * D**2).dot( Vt.dot(B) )
    
        Utilde, Dtilde, Vttilde = sci.linalg.svd( Z , full_matrices=False, overwrite_a=True)
        
        A = Utilde.dot(Vttilde)
        
        
        # Proximal Gradient Descent to Update B
        #G = XtX.dot(A-B) - beta * B
        G = (Vt.T * D**2).dot(Vt.dot(A - B)) - beta * B
        
        B_temp = B + nu * G
        
        # l1 soft-threshold
        idxH = B_temp > kappa
        idxL = B_temp <= -kappa
        B = np.zeros( B.shape )
        B[idxH] = B_temp[idxH] - kappa    
        B[idxL] = B_temp[idxL] + kappa


        # compute residual
        DV = Vt.T * D
        R = DV.T - DV.T.dot(B).dot(A.T)
      
        # Compute objective function
        obj.append( 0.5 * sci.sum(R**2) + alpha * sci.sum(np.abs(B)) + 0.5 * beta * sci.sum(B**2) )  

            
        # Verbose
        if verbose == True and noi%10==0: print("Iteration:  %s, Objective:  %s" % (noi, obj[noi]))
    
       
        # Break if obj is not improving anymore
        if noi>0 and abs(obj[noi-1]-obj[noi]) / obj[noi] < tol: break        

        # Next iter
        noi += 1 
  
    eigvals = Dtilde / (m-1)
    return(B, A, eigvals, obj)




def rspca(X, n_components, alpha = 0.1, beta = 0.1, 
         max_iter = 1000, tol = 1e-5, verbose = 0, p = 20, q = 2):

    
    """
    Randomized Sparse Principal Component Analysis (rSPCA).
    
    Given a mean centered rectangular matrix `A` with shape `(m, n)`, SPCA 
    computes a set of sparse components that can optimally reconstruct the 
    input data.  The amount of sparseness is controllable by the coefficient
    of the L1 penalty, given by the parameter alpha. In addition, some ridge 
    shrinkage can be applied in order to improve conditioning.   
    
    This algorithm uses randomized methods for linear algebra to accelerate 
    the computations. 
    
    
    Parameters
    ----------
    A : array_like, shape `(m, n)`.
        Real nonnegative input matrix.
    
    n_components : integer, `n_components << min{m,n}`.
        Target rank, i.e., number of sparse components to be computed.
        
    alpha : float, (default ``alpha = 0.1``).
        Sparsity controlling parameter. Higher values lead to sparser components.
    
    beta : float, (default ``beta = 0.1``).
        Amount of ridge shrinkage to apply in order to improve conditionin.

    max_iter : integer, (default ``max_iter = 500``).
        Maximum number of iterations to perform before exiting.
            
    tol : float, (default ``tol = 1e-5``).
        Stopping tolerance for reconstruction error.
            
    verbose : bool ``{'True', 'False'}``, optional (default ``verbose = True``).
        Display progress.

    p : integer, (default: `p=20`).
        Parameter to control oversampling.
    
    q : integer, (default: `q=2`).
        Parameter to control number of power (subspace) iterations.    
    
    Returns
    -------
    B:  array_like, `(n, n_components)`.
        Sparse components extracted from the data.
    
    A : array_like, `(n, n_components)`.
        Orthogonal components extracted from the data.

    eigvals : array_like, `(n_components)`.
        Eigenvalues correspnding to the extracted components. 
        
    obj : array_like, `(n_iter)`.
        Objective value at the i-th iteration.         
        
    Notes
    -----   
    Variable Projection for SPCA solves the following optimization problem:
    minimize 1/2⋅‖X - X⋅B⋅Aᵀ‖² + α⋅‖B‖₁ + 1/2⋅β‖B‖² 
    
    
    References
    ----------

    
    
    Examples
    --------    

    
    """ 
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Shape of data matrix
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~     
    m, n = X.shape     
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute QB decomposition
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~            
    Q, Xcompressed = rqb(X, k = n_components, p=p, q=q )
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute Sparse PCA
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~        
    B, A, eigvals, obj = spca(Xcompressed, n_components=n_components, 
                              alpha = alpha, beta = beta, 
                              max_iter = max_iter, tol = tol, verbose = verbose)   
  
    
    # rescale eigen values
    eigvals = eigvals * (n_components+p - 1) / (m-1)
    
    return(B, A, eigvals, obj)







def robspca(X, n_components, alpha  = 0.1, beta  = 0.1, gamma  = 0.1,
         max_iter = 1000, tol = 1e-5, verbose = True):
    
    """
    Robust Sparse Principal Component Analysis (Robust SPCA).
    
    Given a mean centered rectangular matrix `A` with shape `(m, n)`, SPCA 
    computes a set of sparse components that can optimally reconstruct the 
    input data.  The amount of sparseness is controllable by the coefficient
    of the L1 penalty, given by the parameter alpha. In addition, some ridge 
    shrinkage can be applied in order to improve conditioning.   
    

    
    Parameters
    ----------
    A : array_like, shape `(m, n)`.
        Real nonnegative input matrix.
    
    n_components : integer, `n_components << min{m,n}`.
        Target rank, i.e., number of sparse components to be computed.
        
    alpha : float, (default ``alpha = 0.1``).
        Sparsity controlling parameter. Higher values lead to sparser components.
    
    beta : float, (default ``beta = 0.1``)
        Amount of ridge shrinkage to apply in order to improve conditionin.
        
    gamma : float, (default ``gamma = 0.1``).
        Sparsity controlling parameter for the error matrix S. 
        Smaller values lead to a larger amount of noise removeal.        

    max_iter : integer, (default ``max_iter = 500``).
        Maximum number of iterations to perform before exiting.
            
    tol : float, (default ``tol = 1e-5``).
        Stopping tolerance for reconstruction error.
            
    verbose : bool ``{'True', 'False'}``, optional (default ``verbose = True``).
        Display progress.
   
    
    Returns
    -------
    B:  array_like, `(n, n_components)`.
        Sparse components extracted from the data.
    
    A : array_like, `(n, n_components)`.
        Orthogonal components extracted from the data.
        
    S : array_like, `(m, n)`.
        Sparse component which captures grossly corrupted entries in the data 
        matrix.        

    eigvals : array_like, `(n_components)`.
        Eigenvalues correspnding to the extracted components. 
        
    obj : array_like, `(n_iter)`.
        Objective value at the i-th iteration.  
        
        
    Notes
    -----   
    Variable Projection for SPCA solves the following optimization problem:
    minimize 1/2⋅‖X - X⋅B⋅Aᵀ - S‖² + α⋅‖B‖₁ + 1/2⋅β‖B‖²  + γ‖S‖₁
    
    
    References
    ----------

    
    
    Examples
    --------    

    
    """ 

    # Shape of input matrix
    m,n = X.shape
    
    #--------------------------------------------------------------------
    #   Initialization of Variable Projection Solver
    #--------------------------------------------------------------------    
    U, D, Vt = sci.linalg.svd( X , full_matrices=False, overwrite_a=False)
    #U, D, Vt = rsvd( X , k=n_components, p=p, q=q)
    
    Dmax = D[0] #l2 norm

    U = U[:,0:n_components]
    Vt = Vt[0:n_components,:]

    A = Vt.T
    B = Vt.T

    #--------------------------------------------------------------------
    #   Set Tuning Parameters
    #--------------------------------------------------------------------   
    alpha *= Dmax**2
    beta *= Dmax**2
    gamma *= Dmax**2    
    
    
    noi = 0
    nu   = 1.0 / (Dmax**2 + beta)
    kappa = nu * alpha 
    
    obj = []
    
    S = sci.zeros(X.shape)    

    #--------------------------------------------------------------------
    #   Apply Variable Projection Solver
    #--------------------------------------------------------------------
    while max_iter > noi:
  
        # Update A: 
        # X'XB = UDV'
        # Compute X'XB via SVD of X
        XS = X - S
        XB = X.dot(B)
        Z = (XS).T.dot(XB)
    
        Utilde, Dtilde, Vttilde = sci.linalg.svd( Z , full_matrices=False, overwrite_a=True)
        #Utilde, Dtilde, Vttilde = rsvd( Z , k=n_components, p=p, q=q)

        A = Utilde.dot(Vttilde)

        
        # Proximal Gradient Descent to Update B
        R = (XS) - (XB).dot(A.T)
        G = X.T.dot(R.dot(A)) - beta * B
    
        B_temp = B + nu * G
    
            
        # l1 soft-threshold
        idxH = B_temp > kappa
        idxL = B_temp <= -kappa
        B = np.zeros( B.shape )
        B[idxH] = B_temp[idxH] - kappa    
        B[idxL] = B_temp[idxL] + kappa
   
        
        # compute residual
        R = X - (X.dot(B)).dot(A.T)   

        # l1 soft-threshold
        idxH = R > gamma
        idxL = R <= -gamma
        S = np.zeros( S.shape )
        S[idxH] = R[idxH] - gamma    
        S[idxL] = R[idxL] + gamma    
 
    
        # Compute objective function
        obj.append(0.5 * sci.sum((R-S)**2) + alpha * sci.sum(abs(B)) + 
                   0.5 * beta * sci.sum(B**2) + gamma * sci.sum(abs(S)))


    
        # Verbose
        if verbose == True and noi%10==0: print("Iteration:  %s, Objective:  %s" % (noi, obj[noi]))
    
        
        # Break if obj is not improving anymore
        if noi>0 and abs(obj[noi-1]-obj[noi]) / obj[noi] < tol: break        

        # Next iter
        noi += 1 
    
    eigvals = Dtilde / (m-1)
    return(B, A, S, eigvals, obj)





