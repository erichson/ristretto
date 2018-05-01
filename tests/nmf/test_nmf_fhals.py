import numpy as np

from ristretto.nmf import nmf, rnmf
from ristretto.util import nmf_data


atol_float32 = 1e-4
atol_float64 = 1e-8


def test_nmf_fhals():
    A, Anoisy = nmf_data(100, 100, 10, factor_type='normal', noise_type='normal',  noiselevel=0)
    W, H = nmf(Anoisy, k=10)

    relative_error = (np.linalg.norm(A - W.dot(H)) / np.linalg.norm(A))
    assert relative_error < 1e-4


def test_rnmf_fhals():
    A, Anoisy = nmf_data(100, 100, 10, factor_type='normal', noise_type='normal',  noiselevel=0)
    W, H = rnmf(Anoisy, k=10)

    relative_error = (np.linalg.norm(A - W.dot(H)) / np.linalg.norm(A))
    assert relative_error < 1e-4
