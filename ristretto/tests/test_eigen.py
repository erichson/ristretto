from __future__ import division

import numpy as np

from ristretto.eigen import compute_reigh
from ristretto.eigen import compute_reigh_nystroem
from ristretto.eigen import compute_reigh_nystroem_col

from .utils import relative_error

atol_float32 = 1e-4
atol_float64 = 1e-8


# =============================================================================
# compute_reigh function
# =============================================================================
def test_compute_reigh_float64():
    m, k = 100, 10
    A = np.random.randn(m, k).astype(np.float64)
    A = A.dot(A.T)

    w, v = compute_reigh(A, k, oversample=5, n_subspace=2)
    Ak = (v * w).dot(v.T)

    assert relative_error(A, Ak) < atol_float64


def test_compute_reigh_complex128():
    m, k = 100, 10
    A = np.random.randn(m, k).astype(np.float64) + \
            1j * np.random.randn(m, k).astype(np.float64)
    A = A.dot(A.conj().T)

    w, v = compute_reigh(A, k, oversample=10, n_subspace=2)
    Ak = (v * w).dot(v.conj().T)

    assert relative_error(A, Ak) < atol_float64


# =============================================================================
# reig_nystroem function
# =============================================================================
def test_reig_nystroem_float64():
    m, k = 20, 10
    A = np.random.randn(m, k).astype(np.float64)
    A = A.dot(A.T)

    w, v = compute_reigh_nystroem(A, k, oversample=0, n_subspace=2)
    Ak = (v * w).dot(v.T)

    assert relative_error(A, Ak) < atol_float64


def test_reig_nystroem_complex128():
    m, k = 20, 10
    A = np.random.randn(m, k).astype(np.float64) + \
            1j * np.random.randn(m, k).astype(np.float64)
    A = A.dot(A.conj().T)

    w, v = compute_reigh_nystroem(A, k, oversample=0, n_subspace=2)
    Ak = (v * w).dot(v.conj().T)

    assert relative_error(A, Ak) < atol_float64


# =============================================================================
# reig_nystroem_col function
# =============================================================================
def test_reig_nystroem_col_float64():
    m, k = 20, 10
    A = np.random.randn(m, k).astype(np.float64)
    A = A.dot(A.T)

    w, v = compute_reigh_nystroem_col(A, k, oversample=0)
    Ak = (v * w).dot(v.T)

    assert relative_error(A, Ak) < atol_float64


def test_reig_nystroem_col_complex128():
    m, k = 20, 10
    A = np.random.randn(m, k).astype(np.float64) + \
            1j * np.random.randn(m, k).astype(np.float64)
    A = A.dot(A.conj().T)

    w, v = compute_reigh_nystroem_col(A, k, oversample=0)
    Ak = (v * w).dot(v.conj().T)

    assert relative_error(A, Ak) < atol_float64
