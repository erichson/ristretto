.. _api_ref:

=============
API Reference
=============

This is the reference for the functions contained in `ristretto
<https://github.com/erichson/ristretto>`_.


.. _cur_ref:

:mod:`ristretto.cur`: CUR Decomposition
=======================================

.. automodule:: ristretto.cur
   :no-members:
   :no-inherited-members:

.. currentmodule:: ristretto

.. autosummary::
   :toctree: generated/

   cur.cur
   cur.rcur


.. _dmd_ref:

:mod:`ristretto.dmd`: DMD Decomposition
=======================================

.. automodule:: ristretto.dmd
   :no-members:
   :no-inherited-members:

.. currentmodule:: ristretto

.. autosummary::
   :toctree: generated/

   dmd.dmd
   dmd.rdmd


.. _eigen_ref:

:mod:`ristretto.eigen`: Eigenvalue Decompositions
=================================================

.. automodule:: ristretto.eigen
   :no-members:
   :no-inherited-members:

.. currentmodule:: ristretto

.. autosummary::
   :toctree: generated/

   eigen.reigh
   eigen.reigh_nystroem
   eigen.reigh_nystroem_col


.. _interp_decomp_ref:

:mod:`ristretto.interp_decomp`: Interpolation Decompositions
============================================================

.. automodule:: ristretto.interp_decomp
   :no-members:
   :no-inherited-members:

.. currentmodule:: ristretto

.. autosummary::
   :toctree: generated/

   interp_decomp.interp_decomp
   interp_decomp.rinterp_decomp


.. _lu_ref:

:mod:`ristretto.lu`: LU Decomposition
=====================================

.. automodule:: ristretto.lu
   :no-members:
   :no-inherited-members:

.. currentmodule:: ristretto

.. autosummary::
   :toctree: generated/

   lu.rlu


.. _nmf_ref:

:mod:`ristretto.nmf`: Non-negative Matrix Factorization
=======================================================

.. automodule:: ristretto.nmf
   :no-members:
   :no-inherited-members:

.. currentmodule:: ristretto

.. autosummary::
   :toctree: generated/

   nmf.nmf
   nmf.rnmf


.. _pca_ref:

:mod:`ristretto.pca`: Principal Component Analysis
==================================================

.. automodule:: ristretto.pca
   :no-members:
   :no-inherited-members:

.. currentmodule:: ristretto

.. autosummary::
   :toctree: generated/

   pca.robspca
   pca.rspca
   pca.spca


.. _qb_ref:

:mod:`ristretto.qb`: QB Decomposition
=====================================

.. automodule:: ristretto.qb
   :no-members:
   :no-inherited-members:

.. currentmodule:: ristretto

.. autosummary::
   :toctree: generated/

   qb.rqb


.. _svd_ref:

:mod:`ristretto.svd`: SVD Decomposition
=======================================

.. automodule:: ristretto.svd
   :no-members:
   :no-inherited-members:

.. currentmodule:: ristretto

.. autosummary::
   :toctree: generated/

   svd.rsvd


.. _utils_ref:

:mod:`ristretto.utils`: Utility Functions
=========================================

.. automodule:: ristretto.utils
   :no-members:
   :no-inherited-members:

.. currentmodule:: ristretto

.. autosummary::
   :toctree: generated/

   utils.check_non_negative
   utils.check_random_state
   utils.conjugate_transpose
   utils.safe_sparse_dot
   utils.nmf_data


.. _sketch_ref:

:mod:`ristretto.sketch`: Sketching related functions
====================================================

Transforms
----------
.. automodule:: ristretto.sketch.transforms
   :no-members:
   :no-inherited-members:

.. currentmodule:: ristretto

.. autosummary::
   :toctree: generated/

   sketch.transforms.randomized_uniform_sampling
   sketch.transforms.johnson_lindenstrauss
   sketch.transforms.sparse_johnson_lindenstrauss
   sketch.transforms.fast_johnson_lindenstrauss

Utility Functions
-----------------
.. automodule:: ristretto.sketch.utils
   :no-members:
   :no-inherited-members:

.. currentmodule:: ristretto

.. autosummary::
   :toctree: generated/

   sketch.utils.orthonormalize
   sketch.utils.perform_subspace_iterations
