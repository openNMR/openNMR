import numpy as np
import pytest

from opennmr.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
from opennmr.processing import arithmetic


def _make_1d(n, lo, hi, fn):
    ppm = np.linspace(lo, hi, n)
    metadata = NMRMetadata(vendor="bruker", nucleus="1H", sfo1=400.0)
    axis = NMRAxis(kind="frequency", ppm=ppm, hz=ppm * 400.0)
    return NMRSpectrum(data=fn(ppm), axes=[axis], metadata=metadata)


def test_combine_spectra_sum_default_ops():
    a = _make_1d(50, 0, 10, lambda x: np.ones_like(x))
    b = _make_1d(50, 0, 10, lambda x: 2 * np.ones_like(x))
    combined = arithmetic.combine_spectra([a, b], npoints=50)
    np.testing.assert_allclose(combined.data, 3.0, atol=1e-6)


def test_combine_spectra_subtraction():
    a = _make_1d(50, 0, 10, lambda x: 3 * np.ones_like(x))
    b = _make_1d(50, 0, 10, lambda x: 1 * np.ones_like(x))
    combined = arithmetic.combine_spectra([a, b], ops=["+", "-"], npoints=50)
    np.testing.assert_allclose(combined.data, 2.0, atol=1e-6)


def test_combine_spectra_scaling_factors():
    a = _make_1d(50, 0, 10, lambda x: np.ones_like(x))
    b = _make_1d(50, 0, 10, lambda x: np.ones_like(x))
    combined = arithmetic.combine_spectra([a, b], scaling_factors=[1, 2], npoints=50)
    np.testing.assert_allclose(combined.data, 3.0, atol=1e-6)


def test_combine_spectra_intersection_range():
    a = _make_1d(50, 0, 10, lambda x: np.ones_like(x))
    b = _make_1d(50, 5, 15, lambda x: np.ones_like(x))
    combined = arithmetic.combine_spectra([a, b], common_range="intersection", npoints=20)
    assert combined.axes[0].ppm.min() >= 5.0 - 1e-9
    assert combined.axes[0].ppm.max() <= 10.0 + 1e-9


def test_combine_spectra_union_range_zero_fills_outside():
    a = _make_1d(50, 0, 10, lambda x: np.ones_like(x))
    b = _make_1d(50, 5, 15, lambda x: np.ones_like(x))
    combined = arithmetic.combine_spectra([a, b], common_range="union", npoints=200)
    assert combined.axes[0].ppm.min() < 1e-6
    assert combined.axes[0].ppm.max() > 14.9
    # Near ppm=1 only `a` contributes (b's range doesn't cover it).
    idx = np.argmin(np.abs(combined.axes[0].ppm - 1.0))
    assert combined.data[idx] == pytest.approx(1.0, abs=0.1)


def test_combine_spectra_no_overlap_raises():
    a = _make_1d(50, 0, 5, lambda x: np.ones_like(x))
    b = _make_1d(50, 10, 15, lambda x: np.ones_like(x))
    with pytest.raises(ValueError):
        arithmetic.combine_spectra([a, b], common_range="intersection")


def test_combine_spectra_requires_at_least_two():
    a = _make_1d(50, 0, 10, lambda x: np.ones_like(x))
    with pytest.raises(ValueError):
        arithmetic.combine_spectra([a])
