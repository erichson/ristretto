# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

# Author: N. Benjamin Erichson
# Based on code by Mathieu Blondel, Tom Dupre la Tour
# License: GNU General Public License v3.

cimport cython
from libc.math cimport fabs


def _rfhals_update(double[:, ::1] W, double[:, :] HHt, double[:, :] AHt):
    
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
