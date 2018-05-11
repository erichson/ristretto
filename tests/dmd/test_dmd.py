import numpy as np

from ristretto.dmd import dmd


atol_float32 = 1e-4
atol_float64 = 1e-8


def test_dmd():
    # Define time and space discretizations
    x=np.linspace( -10, 10, 100)
    t=np.linspace(0, 8*np.pi , 60)
    dt=t[2]-t[1]
    X, T = np.meshgrid(x,t)
    # Create two patio-temporal patterns
    F1 = 0.5* np.cos(X)*(1.+0.* T)
    F2 = ( (1./np.cosh(X)) * np.tanh(X)) *(2.*np.exp(1j*2.8*T))
    A = np.array((F1+F2).T, order='C')

    Fmodes, b, V, omega = dmd(A, k=2, modes='standard',
                              return_amplitudes=True, return_vandermonde=True)
    Atilde = Fmodes.dot( np.dot(np.diag(b), V))
    assert np.allclose(A, Atilde, atol_float64)

    Fmodes, b, V, omega = dmd(A, modes='standard', return_amplitudes=True,
                              return_vandermonde=True)
    Atilde = Fmodes.dot( np.dot(np.diag(b), V))
    assert np.allclose(A, Atilde, atol_float64)

    Fmodes, b, V, omega = dmd(A, modes='exact', return_amplitudes=True,
                              return_vandermonde=True)
    Atilde = Fmodes.dot( np.dot(np.diag(b), V))
    assert np.allclose(A, Atilde, atol_float64)

    Fmodes, b, V, omega = dmd(A, modes='exact_scaled', return_amplitudes=True,
                              return_vandermonde=True)
    Atilde = Fmodes.dot( np.dot(np.diag(b), V))
    assert np.allclose(A, Atilde, atol_float64)
