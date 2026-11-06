"""The automatic spline baseline on noise-free points (2026-11-06).

A perfectly smooth baseline (a simulated spectrum, or the straight-line data
of the GUI tests) has a robust noise estimate at floating-point rounding.
Its automatic smoothing target, len * noise**2, was then far below what
arithmetic can resolve, and FITPACK warned that its result was
"theoretically impossible" (s too small). Such points are now interpolated
(s = 0) with nothing rejected; noisy baselines are smoothed as before.
"""

import warnings

import numpy as np
import pytest

from opennmr.processing.fitting import estimate_baseline_spline
from opennmr.processing.ft import baseline_correct


@pytest.mark.parametrize("make", [
    lambda x: 0.5 * x - 16.0,                       # the GUI tests' straight line
    lambda x: 3e5 + 40.0 * x + 0.8 * x ** 2,        # a smooth curve at real intensities
    lambda x: np.zeros_like(x),                     # nothing at all
])
def test_noise_free_baseline_fits_exactly_without_a_warning(make):
    x = np.arange(64, dtype=float)
    y = make(x)
    with warnings.catch_warnings():
        warnings.simplefilter("error")              # any FITPACK warning fails
        spline = estimate_baseline_spline(x, y)
    edges = np.r_[x[:3], x[-3:]]
    assert np.allclose(spline(edges), make(edges), rtol=1e-9, atol=1e-9)


def test_baseline_correct_spline_on_a_straight_line_is_quiet():
    freq = np.linspace(10, -10, 64)
    spectrum = np.arange(64, dtype=float) - 32
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        corrected = baseline_correct(freq, spectrum, method="spline")
    assert np.allclose(corrected, 0.0, atol=1e-8)


def test_a_noisy_baseline_is_still_smoothed_not_interpolated():
    rng = np.random.default_rng(3)
    x = np.linspace(0, 1, 4000)
    trend = 200.0 * (x - 0.5) ** 2
    y = trend + rng.normal(0, 5.0, x.size)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        spline = estimate_baseline_spline(x, y, frac=0.1)
    edges = np.r_[x[:400], x[-400:]]
    # Smoothed: it follows the trend far more closely than the noise (sigma 5)
    # -- an interpolating spline would be off by about 5 everywhere.
    assert np.sqrt(np.mean((spline(edges) - 200.0 * (edges - 0.5) ** 2) ** 2)) < 1.5


def test_small_but_real_noise_is_not_mistaken_for_none():
    rng = np.random.default_rng(4)
    x = np.linspace(0, 1, 2000)
    y = 1e6 + rng.normal(0, 1e-2, x.size)           # 1e-8 of the level: real, if tiny
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        spline = estimate_baseline_spline(x, y)
    edges = np.r_[x[:100], x[-100:]]
    assert np.std(spline(edges) - 1e6) < 5e-3      # smoothed, not following each point
