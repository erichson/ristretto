# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

# Based on code by Mathieu Blondel, Tom Dupre la Tour
# License: GNU General Public License v3.

cimport cython
from libc.math cimport fabs


def _fhals_update_shuffle(double[:, ::1] W, double[:, :] HHt, double[:, :] XHt,
                       Py_ssize_t[::1] permutation):
    cdef double violation = 0
    cdef Py_ssize_t n_components = W.shape[1]
    cdef Py_ssize_t n_samples = W.shape[0]  # n_features for H update
    cdef double  gradient, projected_gradient
    cdef Py_ssize_t i, r, s, t

    with nogil:
        for s in range(n_components):
            t = permutation[s]

            for i in range(n_samples):
                gradient = -XHt[i, t]

                for r in range(n_components):
                    gradient += HHt[t, r] * W[i, r]

                # projected gradient
                if W[i, t] == 0.0:
                    projected_gradient = min(0.0, gradient)

                elif W[i, t] > 0.0:
                    projected_gradient = gradient

                else:
                    projected_gradient = 0.0

                violation += fabs(projected_gradient)

                if HHt[t, t] != 0:
                    W[i, t] = max(0.0, W[i, t] - gradient / HHt[t, t])

    return violation
