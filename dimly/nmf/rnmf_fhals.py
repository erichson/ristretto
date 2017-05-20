from __future__ import division

import numpy as np
import scipy as sci

from .._fhals_update import _fhals_update
#import pyximport; pyximport.install()
#from _fhals_update import _fhals_update

epsi = np.finfo(np.float32).eps

def _rnqb(A, k, p=10, q=2):
    m,n = A.shape
     
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Build sample matrix Y : Y = A * O, where O is arandom sampling matrix
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
    Y = A.dot(sci.random.rand( n, k+p ))
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Orthogonalize Y using economic QR decomposition: Y=QR
    #If q > 0 perfrom q subspace iterations
    #Note: check_finite=False may give a performance gain
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~      
    if q > 0:
        for i in xrange(q):
            Y , _ = sci.linalg.qr( Y, mode='economic', check_finite=False, overwrite_a=True)
            Z , _ = sci.linalg.qr( A.T.dot(Y), mode='economic', check_finite=False, overwrite_a=True)
            Y = A.dot( Z )
        #End for
     #End if       
        
    Q , _ = sci.linalg.qr( Y,  mode='economic', check_finite=False, overwrite_a=True) 
    del(Y)
 
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Project the data matrix a into a lower dimensional subspace
    #B = Q.T * A 
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
    B = Q.T.dot( A )     
    
    return(Q, B)



def rnmf_fhals(A, k, p=10, q=2, init='rand',
               tol=1e-4, maxiter=100, 
               verbose=False):
            
    fit_out = []

    Q, A = _rnqb(A, k, p=p, q=q)

    #traceAtA = sci.trace(A.T.dot(A))
    #traceAtA = np.dot(A.ravel(), A.ravel())                      
                           
    # Initialization methods for factor matrices W and H
    # 'rand': uniform random init
    if init == 'rand':
        m,n = A.shape
        W = sci.maximum(0.0, sci.random.standard_normal((m, k)))
        Ht = sci.maximum(0.0, sci.random.standard_normal((n, k)))
        #Ht /= Ht.max(axis=0)
        Ht /= sci.linalg.norm(Ht, axis=0)
    else:
        raise ValueError('Initialization method is not supported.')
    
 
    for niter in xrange(maxiter):

        if niter != 0:
            W = Q.T.dot(W)    

        # Update factor matrix H
        WtW = W.T.dot(W)
        AtW = A.T.dot(W)
        
        _ = _fhals_update(Ht, WtW, AtW)                        
        #Ht /= sci.maximum(epsi, Ht.max(axis=0))
        Ht /= sci.maximum(epsi, sci.linalg.norm(Ht, axis=0))

        # Update factor matrix W
        HHt = Ht.T.dot(Ht)
        AHt = Q.dot(A.dot(Ht))

        W = Q.dot(W) 
        _ = _fhals_update(W, HHt, AHt)
        

        
        if niter % 10 == 0 and verbose == True:
            fit = np.log10(sci.linalg.norm(A - (Q.T.dot(W)).dot( Ht.T)))
            #fit = traceAtA -2 * sci.trace(AtW.dot(Ht.T)) + sci.trace(Ht.dot(WtW).dot(Ht.T)) 
            #fit = sci.log10( np.sqrt(fit) )
            if niter == 0:  
                fitold = fit               
            fitchange = abs(fitold - fit)
            fitold = fit
            fit_out.append( fit )
            
            if verbose == True:
                print('Iteration: %s fit: %s, fitchange: %s' %(niter, fit, fitchange))        
 
            if niter > 1 and (fit <= -5 or fitchange <= tol):      
                break       
        

    if verbose == True:
        fit = np.log10(sci.linalg.norm(A - (Q.T.dot(W)).dot( Ht.T)))
        #fit = traceAtA -2 * sci.trace(AtW.dot(Ht.T)) + sci.trace(Ht.dot(WtW).dot(Ht.T)) 
        #fit = sci.log10( np.sqrt(fit) )
        print('Final Iteration: %s fit: %s' %(niter, fit)) 
        
    return( W, Ht.T, fit_out)