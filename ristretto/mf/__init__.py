"""
Module containg algorithms for matrix factorization.
"""
# Authors: N. Benjamin Erichson
#          Joseph Knox
# License: GNU General Public License v3.0

from .csvd import csvd

from .interp_decomp import interp_decomp
from .interp_decomp import rinterp_decomp
from .interp_decomp import rinterp_decomp_qb

from .reigen import reigh
from .reigen import reigh_nystroem
from .reigen import reigh_nystroem_col

from .rcur import cur
from .rcur import rcur

from .rlu import rlu

from .rqb import rqb

from .rsvd import rsvd

from .rsvd_single import rsvd_single


__all__ = ['interp_decomp',
           'rinterp_decomp',
           'rinterp_decomp_qb',
           'reigh',
           'reigh_nystroem',
           'reigh_nystroem_col',
           'cur',
           'rcur',
           'rlu',
           'rqb',
           'rsvd',
           'rsvd_single']
