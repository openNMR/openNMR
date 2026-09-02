import numpy as np
import pytest

from opennmr.plotting.style import downsample_2d


# ============================================================
# downsample_2d -- max-magnitude-preserving, not evenly-spaced
# index-skipping (see the function's own docstring for the real
# EXSY-diagonal bug this fixed)
# ============================================================

def test_downsample_2d_preserves_a_thin_diagonal_ridge():
    # The exact real-world failure mode: an evenly-spaced index pick
    # can step clean over a 1-point-wide ridge at an irrational slope
    # and lose most of it. Deliberately asymmetric shape (matching a
    # real 2D's own F1<<F2), irrational-ish slope so no bin/ridge
    # alignment happens to save the old naive method by coincidence.
    n_y, n_x = 256, 4096
    data = np.zeros((n_y, n_x))
    slope = (n_x / n_y) * 0.9137
    ridge_cols = set()
    for i in range(n_y):
        j = int(round(i * slope)) % n_x
        data[i, j] = 1.0
        ridge_cols.add(j)
    x = np.arange(n_x, dtype=float)
    y = np.arange(n_y, dtype=float)

    data_ds, x_ds, y_ds = downsample_2d(data, x, y, max_points_x=1024, max_points_y=64)

    # Every original row had exactly one ridge point -- almost all of
    # them should still be individually visible (some collisions are
    # expected/acceptable once multiple original rows collapse into
    # the same output row).
    assert np.count_nonzero(data_ds) >= n_y * 0.8


def test_downsample_2d_naive_index_skip_would_have_lost_the_same_ridge():
    # Confirms the test above is actually discriminating -- the OLD
    # evenly-spaced-index approach loses the vast majority of the same
    # ridge, so "preserves most of it" above is a real improvement,
    # not a tautology that would pass either way.
    n_y, n_x = 256, 4096
    data = np.zeros((n_y, n_x))
    slope = (n_x / n_y) * 0.9137
    for i in range(n_y):
        j = int(round(i * slope)) % n_x
        data[i, j] = 1.0

    x_idx = np.linspace(0, n_x - 1, 1024, dtype=int)
    y_idx = np.linspace(0, n_y - 1, 64, dtype=int)
    naive_ds = data[np.ix_(y_idx, x_idx)]

    assert np.count_nonzero(naive_ds) < n_y * 0.8 * (64 / n_y)  # far fewer survive than the fixed version


def test_downsample_2d_preserves_sign():
    data = np.zeros((64, 64))
    data[10, 10] = -5.0
    data[10, 11] = 3.0  # smaller magnitude, same bin-ish neighborhood
    x = np.arange(64, dtype=float)
    y = np.arange(64, dtype=float)

    data_ds, _, _ = downsample_2d(data, x, y, max_points_x=16, max_points_y=16)

    # The largest-magnitude value in the bin containing column 10/11
    # is -5.0 -- must survive as negative, not get sign-dropped.
    assert data_ds.min() == pytest.approx(-5.0)


def test_downsample_2d_returns_unchanged_when_already_within_limits():
    data = np.random.default_rng(0).normal(size=(10, 12))
    x = np.arange(12, dtype=float)
    y = np.arange(10, dtype=float)

    data_ds, x_ds, y_ds = downsample_2d(data, x, y, max_points_x=2000, max_points_y=500)

    np.testing.assert_array_equal(data_ds, data)
    np.testing.assert_array_equal(x_ds, x)
    np.testing.assert_array_equal(y_ds, y)


def test_downsample_2d_handles_axis_length_not_evenly_divisible_by_target():
    # Real acquisition sizes rarely divide evenly into a target point
    # count -- must not crash, and must still land on the requested
    # output shape (trimming a few trailing points rather than failing).
    data = np.random.default_rng(1).normal(size=(1002, 39418))
    x = np.arange(39418, dtype=float)
    y = np.arange(1002, dtype=float)

    data_ds, x_ds, y_ds = downsample_2d(data, x, y, max_points_x=16384, max_points_y=256)

    assert data_ds.shape == (256, 16384)
    assert len(x_ds) == 16384
    assert len(y_ds) == 256


def test_downsample_2d_output_shape_matches_requested_caps():
    data = np.random.default_rng(2).normal(size=(1024, 65536))
    x = np.arange(65536, dtype=float)
    y = np.arange(1024, dtype=float)

    data_ds, x_ds, y_ds = downsample_2d(data, x, y, max_points_x=16384, max_points_y=256)

    assert data_ds.shape == (256, 16384)
    assert len(x_ds) == 16384
    assert len(y_ds) == 256
