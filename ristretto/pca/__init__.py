"""
Module containing algorithms for sparse Principal Component Analysis.
"""
# Authors: N. Benjamin Erichson
#          Joseph Knox
# License: GNU General Public License v3.0

from .spca import robspca
from .spca import rspca
from .spca import spca


__all__ = ['robspca', 'rspca', 'spca']
