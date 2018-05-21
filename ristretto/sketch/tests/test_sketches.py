import numpy as np
from numpy.testing import assert_raises

from ristretto.sketch import _sketches


def test_random_axis_sample():
    # ------------------------------------------------------------------------
    # tests return correct size
    m, n = 30, 10
    A = np.ones((m, n))
    l = 3
    random_state = np.random.RandomState(123)

    row_idx = _sketches.random_axis_sample(A, l, 0, random_state)
    col_idx = _sketches.random_axis_sample(A, l, 1, random_state)

    assert len(row_idx) == l
    assert len(col_idx) == l

    # ------------------------------------------------------------------------
    # tests returns unique indices
    assert len(np.unique(row_idx)) == len(row_idx)
    assert len(np.unique(col_idx)) == len(col_idx)

    # ------------------------------------------------------------------------
    # tests return unique in axis
    assert all(np.isin(row_idx, np.arange(m), assume_unique=True))
    assert all(np.isin(col_idx, np.arange(n), assume_unique=True))


def test_random_gaussian_map():
    # ------------------------------------------------------------------------
    # tests return correct shape
    m, n = 30, 10
    A = np.ones((m, n))
    l = 3
    random_state = np.random.RandomState(123)

    row_sketch = _sketches.random_gaussian_map(A, l, 0, random_state)
    col_sketch = _sketches.random_gaussian_map(A, l, 1, random_state)

    assert row_sketch.shape == (m, l)
    assert col_sketch.shape == (n, l)

    # ------------------------------------------------------------------------
    # tests return correct data type
    assert row_sketch.dtype == A.dtype
    assert col_sketch.dtype == A.dtype


def test_sparse_random_map():
    # ------------------------------------------------------------------------
    # tests return correct shape
    m, n = 30, 10
    A = np.ones((m, n))
    l = 3
    density = 1./3
    random_state = np.random.RandomState(123)

    row_sketch = _sketches.sparse_random_map(A, l, 0, density, random_state)
    col_sketch = _sketches.sparse_random_map(A, l, 1, density, random_state)

    assert row_sketch.shape == (m, l)
    assert col_sketch.shape == (n, l)

    # ------------------------------------------------------------------------
    # tests return correct data type
    assert row_sketch.dtype == A.dtype
    assert col_sketch.dtype == A.dtype

    # ------------------------------------------------------------------------
    # tests returns correct density
    assert row_sketch.nnz == int(density*m*l)
    assert col_sketch.nnz == int(density*n*l)

    # ------------------------------------------------------------------------
    # tests raises error when density not in [0,1]
    assert_raises(ValueError, _sketches.sparse_random_map, A, l, 0, -1, random_state)
    assert_raises(ValueError, _sketches.sparse_random_map, A, l, 0, 1.1, random_state)
