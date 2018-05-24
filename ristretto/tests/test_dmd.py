import numpy as np

from ristretto.dmd import dmd
from ristretto.dmd import rdmd

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


def A_tilde(Fmodes, b, V):
    return Fmodes.dot(np.diag(b).dot(V))

# =============================================================================
# dmd function
# =============================================================================
def test_dmd():
    A = get_A()

    # ------------------------------------------------------------------------
    # tests mode == standard
    Fmodes, b, V, omega = dmd(A, rank=2, modes='standard',
                              return_amplitudes=True, return_vandermonde=True)
    assert np.allclose(A, A_tilde(Fmodes, b, V), atol_float64)

    # ------------------------------------------------------------------------
    # tests mode == exact, rank == A.shape[1]
    Fmodes, b, V, omega = dmd(A, rank=A.shape[1], modes='exact',
                              return_amplitudes=True, return_vandermonde=True)
    assert np.allclose(A, A_tilde(Fmodes, b, V), atol_float64)

    # ------------------------------------------------------------------------
    # tests mode == exact_scaled, rank == None
    Fmodes, b, V, omega = dmd(A, modes='exact_scaled', return_amplitudes=True,
                              return_vandermonde=True)
    assert np.allclose(A, A_tilde(Fmodes, b, V), atol_float64)


# =============================================================================
# rdmd function
# =============================================================================
def test_rdmd():
    A = get_A()

    Fmodes, b, V, omega = rdmd(A, 2, oversample=10, n_subspace=2,
                               return_amplitudes=True, return_vandermonde=True)
    assert np.allclose(A, A_tilde(Fmodes, b, V), atol_float64)
