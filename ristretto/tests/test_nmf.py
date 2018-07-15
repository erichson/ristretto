import numpy as np

from ristretto.nmf import nmf
from ristretto.nmf import rnmf
from ristretto.utils import nmf_data

atol_float32 = 1e-4
atol_float64 = 1e-8


# =============================================================================
# nmf function
# =============================================================================
def test_nmf_fhals():
    A, Anoisy = nmf_data(100, 100, 10, factor_type='normal', noise_type='normal',  noiselevel=0)
    W, H = nmf(Anoisy, rank=10)

    relative_error = (np.linalg.norm(A - W.dot(H)) / np.linalg.norm(A))
    assert relative_error < 1e-4


# =============================================================================
# rnmf function
# =============================================================================
def test_rnmf_fhals():
    A, Anoisy = nmf_data(100, 100, 10, factor_type='normal', noise_type='normal',  noiselevel=0)
    W, H = rnmf(Anoisy, rank=10)

    relative_error = (np.linalg.norm(A - W.dot(H)) / np.linalg.norm(A))
    assert relative_error < 1e-4
