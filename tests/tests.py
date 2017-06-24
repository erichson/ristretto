from __future__ import division
import numpy as np
import scipy as sci

from ristretto import *
from ristretto.nmf import *
from ristretto.mf import *
from ristretto.util import *



from unittest import main, makeSuite, TestCase, TestSuite
from numpy.testing import assert_raises, assert_equal, assert_allclose

atol_float32 = 1e-4
atol_float64 = 1e-8

#
#******************************************************************************
#
class test_mf(TestCase):
    def setUp(self):
        np.random.seed(123)

    def test_rqb_float64(self):
        m, k = 100, 10
        A = np.array(np.random.randn(m, k), np.float64)
        A = A.dot( A.T )
        Q, B = rqb(A, k=k, p=5, q=2)
        Ak = Q.dot(B)
        percent_error = 100 * np.linalg.norm(A - Ak) / np.linalg.norm(A)
        
        assert percent_error < atol_float64
           

    def test_rqb_complex128(self):
        m, k = 100, 10
        A = np.array(np.random.randn(m, k), np.float64) + 1j * np.array(np.random.randn(m, k), np.float64)
        A = A.dot(A.conj().T)
        Q, B = rqb(A, k=k, p=5, q=2)
        Ak = Q.dot(B)
        percent_error = 100 * np.linalg.norm(A - Ak) / np.linalg.norm(A)
        
        assert percent_error < atol_float64

    def test_rsvd_float64(self):
        m, k = 100, 10
        A = np.array(np.random.randn(m, k), np.float64)
        A = A.dot( A.T )
        U, s, Vt = rsvd(A, k=k, p=5, q=2)
        Ak = U.dot(np.diag(s).dot(Vt))
        percent_error = 100 * np.linalg.norm(A - Ak) / np.linalg.norm(A)
        
        assert percent_error < atol_float64
           

    def test_rsvd_complex128(self):
        m, k = 100, 10
        A = np.array(np.random.randn(m, k), np.float64) + 1j * np.array(np.random.randn(m, k), np.float64)
        A = A.dot(A.conj().T)
        U, s, Vh = rsvd(A, k=k, p=5, q=2)
        Ak = U.dot(np.diag(s).dot(Vh))
        percent_error = 100 * np.linalg.norm(A - Ak) / np.linalg.norm(A)
        
        assert percent_error < atol_float64

		  
    def test_rsvd_fliped_float64(self):
        m, k = 100, 10
        A = np.array(np.random.randn(m, k), np.float64)
        A = A.dot(A.T)
        A = A[:,0:50]
        U, s, Vh = rsvd(A.T, k=k, p=5, q=2)
        Ak = U.dot(np.diag(s).dot(Vh))        
        percent_error = 100 * np.linalg.norm(A.T - Ak) / np.linalg.norm(A.T)
        
        assert percent_error < atol_float64  
  
		    
    def test_rsvd_fliped_complex128(self):
        m, k = 100, 10
        A = np.array(np.random.randn(m, k), np.float64) + 1j * np.array(np.random.randn(m, k), np.float64)
        A = A.dot(A.conj().T)
        A = A[:,0:50]
        U, s, Vh = rsvd(A.conj().T, k=k, p=5, q=2)
        Ak = U.dot(np.diag(s).dot(Vh))
        percent_error = 100 * np.linalg.norm(A.conj().T - Ak) / np.linalg.norm(A.conj().T)
        
        assert percent_error < atol_float64 






#
#******************************************************************************
#
class test_nmf(TestCase):
    def setUp(self):
        np.random.seed(123)

    def test_rnmf_fhals(self):
		A, Anoisy = nmf_data(100, 100, 10, factor_type='normal', noise_type='normal',  noiselevel=0)
		W, H = rnmf_fhals(Anoisy, k=10)
		
		relative_error = (np.linalg.norm(A - W.dot(H)) / np.linalg.norm(A))
		assert relative_error < 1e-4  


    def test_nmf_fhals(self):
		A, Anoisy = nmf_data(100, 100, 10, factor_type='normal', noise_type='normal',  noiselevel=0)
		W, H = nmf_fhals(Anoisy, k=10)
		
		relative_error = (np.linalg.norm(A - W.dot(H)) / np.linalg.norm(A))
		assert relative_error < 1e-4  
     


#
#******************************************************************************
#
        
def suite():
    s = TestSuite()
    s.addTest(test_nmf('test_rnmf_fhals'))

    
    return s

if __name__ == '__main__':
    main(defaultTest = 'suite')
