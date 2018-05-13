"""
Module :mod"`dmd` contains functions for computing the Dynamic Mode Decomposition.
"""
# Authors: N. Benjamin Erichson
#          Joseph Knox
# License: GNU General Public License v3.0

from .dmd import dmd
from .rdmd import rdmd
from .rdmd_single import rdmd_single

__all__ = ['dmd', 'rdmd', 'rdmd_single']
