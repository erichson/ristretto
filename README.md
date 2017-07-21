![ristretto](https://raw.githubusercontent.com/Benli11/ristretto/master/ristretto.png)

Package Description [![Build Status](https://travis-ci.org/Benli11/ristretto.svg?branch=master)](https://travis-ci.org/Benli11/ristretto)
*************************************************

Ristretto means 'limited' or 'restricted' in Italian, and it is traditionally a short shot of espresso. The taste is predominate by the faster-to-extract compounds, which are extracted by forcing a small amount of water under highy pressure through ground coffee beans.

The idea of randomized low-rank matrix approximations is to restrict the high-dimensional input data matrix to a low-dimensional space. In plain words, the aim is to find a smaller matrix which captures the essential information of the input matrix. This smaller matrix can then be used to extract (learn) the coherent structure of the data. Probabilistic algorithms considerably reduce the computational demands of traditional (deterministic) algorithms, and the computational advantage becomes pronounced with increasing matrix dimensions.

The software library ``ristretto`` provides a collection of randomized matrix algorithms which can be used for dimension reduction. Overview of implemented routines:
* Randomized singular value decomposition (rsvd).
* Single-view randomized singular value decomposition (rsvd_single).
* Randomized LU decompositoin (rlu).
* Randomized nonnegative matrix factorization (rnmf_fhals).



Installation
************
Get the latest version
``git clone https://github.com/Benli11/ristretto``

To build and install ristretto, run from within the main directory in the release:
``python setup.py install``

After successfully installing ristretto, the unit tests can be run by:
``python setup.py test``


References
*************
* [N. Benjamin Erichson, et al. `Randomized Matrix Decompositions using R.' (2016)](http://arxiv.org/abs/1608.02148)
* [Sergey Voronin, Per-Gunnar Martinsson. `RSVDPACK: Subroutines for computing partial singular value decompositions via randomized sampling on single core, multi core, and GPU architectures.' (2015)](https://arxiv.org/abs/1502.05366)
* [Nathan Halko, et al. 1Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions.' (2011)](https://arxiv.org/abs/0909.4061)

