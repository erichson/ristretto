# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

# The following code is written by Mathieu Blondel, Tom Dupre la Tour
# and only slightly modified. 
# See https://github.com/scikit-learn/scikit-learn
# License: BSD 3 clause

cimport cython
from libc.math cimport fabs

def _fhals_update(double[:, ::1] W, double[:, :] HHt, double[:, :] AHt):
    
    cdef Py_ssize_t m = W.shape[0]  
    cdef Py_ssize_t n = W.shape[1]
    cdef Py_ssize_t i, r, t
    cdef double gradient, projected_gradient
    cdef double violation = 0
    
    with nogil:
        for t in xrange(n):
            for i in xrange(m):
                
                # Gradient
                gradient = -AHt[i, t]

                for r in xrange(n):
                    gradient += HHt[t, r] * W[i, r]

                # projected gradient
                projected_gradient = min(0.0, gradient) if W[i, t] == 0 else gradient
                violation += fabs(projected_gradient)    

                # Update W or H, respectively. 
                if HHt[t, t] != 0:
                    W[i, t] = max(0.0, W[i, t] - gradient / HHt[t, t])   

    return violation