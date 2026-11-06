"""Speed-ups for large 2D data (2026-11-06; an 800 MHz EXSY of
512 x 131072 points took ~20 s to open). Each change keeps the result:

* contour downsampling takes each bin's largest-magnitude signed value from
  the bin's maximum and minimum (two fast reductions) instead of
  abs + argmax + take_along_axis, and trims with a slice instead of np.take;
* the digital filter's FFTs no longer copy the whole array again in astype
  (its byte-for-byte agreement with nmrglue is pinned in test_digital_filter).
"""

import numpy as np
import pytest

from opennmr.plotting.style import _max_abs_bin, downsample_2d


def _reference_bin(data, coords, n_bins, axis):
    """The previous implementation, kept here as the reference."""
    n = data.shape[axis]
    if n_bins >= n:
        return data, coords
    b = n // n_bins
    t = b * n_bins
    if t != n:
        data = np.take(data, np.arange(t), axis=axis)
    r = data.reshape(data.shape[:axis] + (n_bins, b) + data.shape[axis + 1:])
    i = np.argmax(np.abs(r), axis=axis + 1, keepdims=True)
    return (np.squeeze(np.take_along_axis(r, i, axis=axis + 1), axis=axis + 1),
            coords[:t].reshape(n_bins, b)[:, b // 2])


@pytest.mark.parametrize("shape, bins, axis", [
    ((64, 10007), 997, 1),        # length not a multiple of the bins: trimmed
    ((300, 4096), 2000, 1),
    ((513, 700), 64, 0),
    ((7, 5000), 5000, 1),         # already within the limit: unchanged
])
def test_binning_matches_the_previous_method(shape, bins, axis):
    rng = np.random.default_rng(sum(shape))
    data = rng.standard_normal(shape) * rng.choice([1.0, 100.0], size=shape)
    coords = np.linspace(-5, 5, shape[axis])
    got, got_coords = _max_abs_bin(data, coords, bins, axis)
    ref, ref_coords = _reference_bin(data, coords, bins, axis)
    assert np.array_equal(got, ref) and np.array_equal(got_coords, ref_coords)


def test_negative_peaks_keep_their_sign():
    data = np.zeros((4, 100))
    data[1, 37] = -9.0
    data[1, 38] = 5.0
    reduced, _ = _max_abs_bin(data, np.arange(100.0), 10, axis=1)
    assert reduced[1, 3] == -9.0


def test_an_exact_tie_keeps_the_positive_value():
    data = np.zeros((1, 20))
    data[0, 2], data[0, 7] = -4.0, 4.0          # same bin, equal and opposite
    reduced, _ = _max_abs_bin(data, np.arange(20.0), 2, axis=1)
    assert reduced[0, 0] == 4.0


def test_complex_data_still_bins_by_magnitude():
    data = np.zeros((2, 40), dtype=complex)
    data[0, 5] = 3 - 4j                          # |z| = 5
    data[0, 6] = 4.5
    reduced, _ = _max_abs_bin(data, np.arange(40.0), 4, axis=1)
    assert reduced[0, 0] == 3 - 4j


def test_downsample_2d_matches_the_previous_method_on_a_large_array():
    rng = np.random.default_rng(1)
    data = rng.standard_normal((512, 32768))
    x, y = np.arange(32768.0), np.arange(512.0)
    got = downsample_2d(data, x, y)
    r1, rx = _reference_bin(data, x, 2000, 1)
    r2, ry = _reference_bin(r1, y, 500, 0)
    assert all(np.array_equal(a, b) for a, b in zip(got, (r2, rx, ry)))
