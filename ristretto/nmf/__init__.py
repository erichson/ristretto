"""
Module containing non-negative matrix factorization algorithms.
"""
# Authors: N. Benjamin Erichson
#          Joseph Knox
# License: GNU General Public License v3.0

from .nmf_fhals import nmf
from .nmf_fhals import rnmf


__all__ = ['nmf', 'rnmf']
