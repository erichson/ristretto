import numpy as np

from ristretto.dmd import \
    (DMD, RDMD, compute_dmd, compute_rdmd, get_amplitudes, get_vandermonde)

atol_float32 = 1e-4
atol_float64 = 1e-8

def get_A():
    # Define time and space discretizations
    x = np.linspace( -10, 10, 100)
    t = np.linspace(0, 8*np.pi , 60)
    X, T = np.meshgrid(x, t)

    # Create two patio-temporal patterns
    F1 = 0.5 * np.cos(X) * (1. + 0. * T)
    F2 = ((1./np.cosh(X)) * np.tanh(X)) * (2 * np.exp(1j * 2.8 * T))

    return np.array((F1 + F2).T, order='C')


def A_tilde(A, Fmodes, l):
    b = get_amplitudes(A, Fmodes)
    V = get_vandermonde(A, l)

    return Fmodes.dot(np.diag(b).dot(V))

# =============================================================================
# dmd function
def test_dmd():
    A = get_A()

    # ------------------------------------------------------------------------
    # tests mode == standard
    Fmodes, l, omega = compute_dmd(A, rank=2, modes='standard')
    assert np.allclose(A, A_tilde(A, Fmodes, l), atol_float64)

    # ------------------------------------------------------------------------
    # tests mode == exact, rank == A.shape[1]
    Fmodes, l, omega = compute_dmd(A, rank=A.shape[1], modes='exact')
    assert np.allclose(A, A_tilde(A, Fmodes, l), atol_float64)

    # ------------------------------------------------------------------------
    # tests mode == exact_scaled, rank == None
    Fmodes, l, omega = compute_dmd(A, modes='exact_scaled')
    assert np.allclose(A, A_tilde(A, Fmodes, l), atol_float64)


# =============================================================================
# rdmd function
def test_compute_rdmd():
    A = get_A()

    Fmodes, l, omega = compute_rdmd(A, rank=2, oversample=10, n_subspace=2)
    assert np.allclose(A, A_tilde(A, Fmodes, l), atol_float64)


# =============================================================================
# DMD class
def test_DMD():
    A = get_A()

    dmd = DMD(rank=2)
    dmd.fit(A)

    assert np.allclose(A, A_tilde(dmd.X_, dmd.F_, dmd.l_), atol_float64)
    assert np.allclose(dmd.amplitudes_, get_amplitudes(dmd.X_, dmd.F_))
    assert np.allclose(dmd.vandermonde_, get_vandermonde(dmd.X_, dmd.l_))


# =============================================================================
# RDMD class
def test_RDMD():
    A = get_A()

    rdmd = RDMD(rank=2)
    rdmd.fit(A)

    assert np.allclose(A, A_tilde(rdmd.X_, rdmd.F_, rdmd.l_), atol_float64)
