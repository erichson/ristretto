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
import sys

# PEP0440 compatible ristretto version number
# see https://www.python.org/dev/peps/pep-0440/
__version__ = '0.1.2'

try:
    __RISTRETTO_SETUP__
except NameError:
    __RISTRETTO_SETUP__ = False

if __RISTRETTO_SETUP__:
    # not yet importing rest of ristretto during build process, as it may not
    # yet be completed
    sys.stderr.write('Partial import of ristretto during the build process.\n')
else:
    __all__ = ['cur', 'dmd', 'eigen', 'interp_decomp', 'lu', 'nmf', 'pca',
               'qb', 'sketch', 'svd', 'utils']
