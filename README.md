<img src="https://raw.githubusercontent.com/Benli11/ristretto/master/ristretto.png" width="550">

The idea of randomized low-rank matrix approximations is to restrict the high-dimensional input data matrix to a low-dimensional space. In plain words, the aim is to find a smaller matrix which captures the essential information of the input matrix. This smaller matrix can then be used to extract (learn) the coherent structure of the data. Probabilistic algorithms considerably reduce the computational demands of traditional (deterministic) algorithms, and the computational advantage becomes pronounced with increasing matrix dimensions.

# ristretto: Package Overview [![Build Status](https://travis-ci.org/Benli11/ristretto.svg?branch=master)](https://travis-ci.org/Benli11/ristretto)
*************************************************

The Python software library ristretto provides a collection of randomized matrix algorithms which can be used for dimension reduction. Overview of implemented routines:
* Randomized singular value decomposition: ``from ristretto.mf import rsvd``.
* Randomized single-view singular value decomposition: ``from ristretto.mf import rsvd_single``.
* Randomized interpolative decomposition:``from ristretto.mf import rinterp_decomp``.
* Randomized CUR decomposition: ``from ristretto.mf import rcur``.
* Randomized LU decompositoin: ``from ristretto.mf import rlu``.
* Randomized nonnegative matrix factorization: ``from ristretto.nmf import rnmf_fhals``.

# Get started
******************

### Quick Installation via PIP 
To install the latest stable release of ristretto run:

``pip install ristretto``

### Obtaining the Latest Software via GIT 
To get the latest stable and development versions of ristretto run:

``git clone https://github.com/Benli11/ristretto``

Then, to build and install the package, run from within the main directory in the release:

``python setup.py install``

After successfully installing the ristretto library, the unit tests can be run by:

``python setup.py test``



# References
*************
* [N. Benjamin Erichson, et al. `Randomized Matrix Decompositions using R.' (2016)](http://arxiv.org/abs/1608.02148)
* [Sergey Voronin, Per-Gunnar Martinsson. `RSVDPACK: Subroutines for computing partial singular value decompositions via randomized sampling on single core, multi core, and GPU architectures.' (2015)](https://arxiv.org/abs/1502.05366)
* [Nathan Halko, et al. Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions.' (2011)](https://arxiv.org/abs/0909.4061)

