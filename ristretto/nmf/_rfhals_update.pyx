# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

# Author: N. Benjamin Erichson
# Based on code by Mathieu Blondel, Tom Dupre la Tour
# License: GNU General Public License v3.

cimport cython

def _rfhals_update(double[:, ::1] W, double[:, :] HHt, double[:, :] AHt):
    
    cdef Py_ssize_t m = W.shape[0]  
    cdef Py_ssize_t n = W.shape[1]
    cdef Py_ssize_t i, r, t
    cdef double gradient

    with nogil:
        for t in xrange(n):
            for i in xrange(m):
                
                # Gradient
                gradient = -AHt[i, t]

                for r in xrange(n):
                    gradient += HHt[t, r] * W[i, r]

                # Update W or H, respectively. 
                if HHt[t, t] != 0:
                    W[i, t] = max(0.0, W[i, t] - gradient / HHt[t, t])
                
    return 0
