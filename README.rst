.. -*- mode: rst -*-

.. image:: ristretto.png
    :width: 500px


The idea of randomized low-rank matrix approximations is to restrict the
high-dimensional input data matrix to a low-dimensional space. In plain words,
the aim is to find a smaller matrix which captures the essential information of
the input matrix. This smaller matrix can then be used to extract (learn) the
coherent structure of the data. Probabilistic algorithms considerably reduce
the computational demands of traditional (deterministic) algorithms, and the
computational advantage becomes pronounced with increasing matrix dimensions.


ristretto: Package Overview  |Travis|_ |Codecov|_ |Readthedocs|_
=================================================================

.. |Travis| image:: https://travis-ci.org/eirchson/ristretto.svg?branch=master
.. _Travis: https://travis-ci.org/erichson/ristretto

.. |Codecov| image:: https://codecov.io/gh/erichson/ristretto/branch/master/graph/badge.svg
.. _Codecov: https://codecov.io/gh/erichson/ristretto

.. |Readthedocs| image:: https://readthedocs.org/projects/ristretto/badge/?version=latest
.. _Readthedocs: http://ristretto.readthedocs.io/en/latest/?badge=latest

The Python software library ristretto provides a collection of randomized matrix
algorithms which can be used for dimension reduction. Overview of implemented routines:

* Randomized singular value decomposition:``from ristretto.svd import compute_rsvd``.
* Randomized interpolative decomposition:``from ristretto.interp_decomp import compute_rinterp_decomp``.
* Randomized CUR decomposition: ``from ristretto.cur import compute_rcur``.
* Randomized LU decompositoin: ``from ristretto.lu import compute_rlu``.
* Randomized nonnegative matrix factorization: ``from ristretto.nmf import compute_rnmf_fhals``.

Get started
-----------

Obtaining the Latest Software via GIT
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
To get the latest stable and development versions of ristretto run::

   $ git clone https://github.com/erichson/ristretto
   

Then, to build and install the package, run from within the main directory in
the release::

   $ python setup.py install

**Note** you will need the following 3rd party packages installed in your environment:

* numpy
* scipy
* Cython
* scikit-learn
* nose

After successfully installing the ristretto library, the unit tests can be run by::

   $ python setup.py test



References
----------
- `N. Benjamin Erichson, et al. 'Randomized Matrix Decompositions using R.' (2016)
  <http://arxiv.org/abs/1608.02148>`_
- `Sergey Voronin, Per-Gunnar Martinsson. 'RSVDPACK: Subroutines for computing
  partial singular value decompositions via randomized sampling on single core,
  multi core, and GPU architectures.' (2015)
  <https://arxiv.org/abs/1502.05366>`_
- `Nathan Halko, et al. 'Finding structure with randomness: Probabilistic
  algorithms for constructing approximate matrix decompositions.' (2011)
  <https://arxiv.org/abs/0909.4061>`_
