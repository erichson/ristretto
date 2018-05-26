import numpy as np
from numpy.testing import assert_raises

from ristretto.sketch.transforms import randomized_uniform_sampling
from ristretto.sketch.transforms import johnson_lindenstrauss
from ristretto.sketch.transforms import sparse_johnson_lindenstrauss
from ristretto.sketch.transforms import fast_johnson_lindenstrauss


def test_randomized_uniform_sampling():
    # ------------------------------------------------------------------------
    # tests return correct size
    m, n = 30, 10
    A = np.ones((m, n))
    l = 3

    row_trans = randomized_uniform_sampling(A, l, axis=0)
    col_trans = randomized_uniform_sampling(A, l, axis=1)

    assert row_trans.shape == (l, n)
    assert col_trans.shape == (m, l)

    # ------------------------------------------------------------------------
    # tests raises incompatible axis
    assert_raises(IndexError, randomized_uniform_sampling, A, l, axis=2)

    # ------------------------------------------------------------------------
    # tests raises incompatible A dimensions
    assert_raises(IndexError, randomized_uniform_sampling, A[5], l)


def test_johnson_linderstrauss():
    # ------------------------------------------------------------------------
    # tests return correct size
    m, n = 30, 10
    A = np.ones((m, n))
    l = 3

    row_trans = johnson_lindenstrauss(A, l, axis=0)
    col_trans = johnson_lindenstrauss(A, l, axis=1)

    assert row_trans.shape == (l, n)
    assert col_trans.shape == (m, l)

    # ------------------------------------------------------------------------
    # tests raises incompatible axis
    assert_raises(ValueError, johnson_lindenstrauss, A, l, axis=2)

    # ------------------------------------------------------------------------
    # tests raises incompatible A dimensions
    assert_raises(ValueError, johnson_lindenstrauss, A[5], l)


def test_sparse_johnson_linderstrauss():
    # ------------------------------------------------------------------------
    # tests return correct size
    m, n = 30, 10
    A = np.ones((m, n))
    l = 3

    row_trans = sparse_johnson_lindenstrauss(A, l, axis=0)
    col_trans = sparse_johnson_lindenstrauss(A, l, axis=1)

    assert row_trans.shape == (l, n)
    assert col_trans.shape == (m, l)

    # ------------------------------------------------------------------------
    # tests raises incompatible axis
    assert_raises(ValueError, sparse_johnson_lindenstrauss, A, l, axis=2)

    # ------------------------------------------------------------------------
    # tests raises incompatible A dimensions
    assert_raises(ValueError, sparse_johnson_lindenstrauss, A[5], l)


def test_fast_johnson_linderstrauss():
    # ------------------------------------------------------------------------
    # tests return correct size
    m, n = 30, 10
    A = np.ones((m, n))
    l = 3

    row_trans = fast_johnson_lindenstrauss(A, l, axis=0)
    col_trans = fast_johnson_lindenstrauss(A, l, axis=1)

    assert row_trans.shape == (l, n)
    assert col_trans.shape == (m, l)

    # ------------------------------------------------------------------------
    # tests raises incompatible axis
    assert_raises(ValueError, fast_johnson_lindenstrauss, A, l, axis=2)

    # ------------------------------------------------------------------------
    # tests raises incompatible A dimensions
    assert_raises(ValueError, fast_johnson_lindenstrauss, A[5], l)
