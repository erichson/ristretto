import numpy as np
import scipy as sci


def nmf_data(m, n, k, factor_type='normal', noise_type='normal', noiselevel=0):

    if factor_type == 'normal':
        #Normal
        Wtue = np.maximum(0, np.random.standard_normal((m,k)))
        Htrue = np.maximum(0, np.random.standard_normal((k,n)))
    
    elif factor_type == 'unif':
        #Unif
        Wtue = np.random.rand(m,k)
        Htrue =  np.random.rand(k,n)
    else:
        raise ValueError("factor_type not supported")    
    
    
    A = Anoisy = Wtue.dot(Htrue)
    
    if noise_type == 'normal':
        Anoisy = A + np.maximum(0, np.random.standard_normal((m,n))) * noiselevel
    else:
        raise ValueError("noise_type not supported")    

           
    return (A, Anoisy)