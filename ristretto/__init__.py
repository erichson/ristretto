"""
Randomized Dimension Reduction Library
======================================

The Python software library ristretto provides a collection of randomized
matrix algorithms which can be used for dimension reduction.

See https://github.com/erichson/ristretto for more information.
"""
# Authors: N. Benjamin Erichson
#          Joseph Knox
# License: GNU General Public License v3.0

__version__ = '0.1.2'

# Matrix Factorization
from ristretto.mf import rsvd
from ristretto.mf import rsvd_single
from ristretto.mf import csvd

from ristretto.mf import cur
from ristretto.mf import rcur

from ristretto.mf import interp_decomp
from ristretto.mf import rinterp_decomp
from ristretto.mf import rinterp_decomp_qb

from ristretto.mf import rlu

from ristretto.mf import rqb

from ristretto.mf import reigh
from ristretto.mf import reigh_nystroem
from ristretto.mf import reigh_nystroem_col

# Dynamic Mode Decomposition
from ristretto.dmd import dmd
from ristretto.dmd import rdmd
from ristretto.dmd import rdmd_single

#Nonnegative MF
from ristretto.nmf import nmf
from ristretto.nmf import rnmf

#PCA
from ristretto.pca import robspca
from ristretto.pca import rspca
from ristretto.pca import spca

#Utilities
from ristretto import utils
