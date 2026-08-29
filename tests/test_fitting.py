import numpy as np
import pytest

from nmrsuite.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
from nmrsuite.processing import fitting, slicing
from nmrsuite.processing.fitting import gaussian, lorentzian


def _make_1d(x0=5.0, fwhm=0.05, A=1.0, kind="gaussian", b0=0.02, b1=0.0,
             n=800, lo=3.0, hi=7.0):
    ppm = np.linspace(lo, hi, n)
    y = gaussian(ppm, x0, fwhm, A) if kind == "gaussian" else lorentzian(ppm, x0, fwhm, A)
    x_ref = np.mean(ppm)
    y = y + b0 + b1 * (ppm - x_ref)
    metadata = NMRMetadata(vendor="bruker", nucleus="1H", sfo1=400.0)
    axis = NMRAxis(kind="frequency", ppm=ppm, hz=ppm * 400.0)
    return NMRSpectrum(data=y, axes=[axis], metadata=metadata)


def _make_2d_for_fitting(n_pts=1200, x0=5.0, fwhm=0.04, amps=(0.5, 1.0, 1.5), b0=0.01):
    ppm = np.linspace(3.0, 7.0, n_pts)
    n_rows = len(amps)
    data = np.zeros((n_rows, n_pts))
    for i, A in enumerate(amps):
        data[i] = gaussian(ppm, x0, fwhm, A) + b0
    y_ppm = np.linspace(100, 90, n_rows)
    f1_axis = NMRAxis(kind="frequency", ppm=y_ppm, hz=y_ppm * 100.0)
    f2_axis = NMRAxis(kind="frequency", ppm=ppm, hz=ppm * 400.0)
    metadata = NMRMetadata(vendor="bruker", nucleus="1H", nucleus_indirect="13C",
                            sfo1=400.0, sfo1_indirect=100.0)
    return NMRSpectrum(data=data, axes=[f1_axis, f2_axis], metadata=metadata)


# ============================================================
# Baseline / region mask
# ============================================================

def test_build_region_mask_union_of_regions():
    x = np.linspace(0, 10, 11)
    mask = fitting.build_region_mask(x, [(2, 4), (7, 8)])
    expected = ((x >= 2) & (x <= 4)) | ((x >= 7) & (x <= 8))
    np.testing.assert_array_equal(mask, expected)


def test_build_region_mask_none_selects_everything():
    x = np.linspace(0, 10, 11)
    assert np.all(fitting.build_region_mask(x, None))


def test_estimate_baseline_robust_recovers_clean_linear_baseline():
    x = np.linspace(-5, 5, 200)
    b0_true, b1_true = 0.5, 0.1
    y = b0_true + b1_true * x

    b0, b1, x_ref = fitting.estimate_baseline_robust(x, y, baseline_regions=(-5, 5), sigma=3.0)

    assert b1 == pytest.approx(b1_true, abs=1e-6)
    assert (b0 + b1 * (0.0 - x_ref)) == pytest.approx(b0_true, abs=1e-6)


def test_estimate_baseline_robust_rejects_outliers():
    # nmrtools2's original convergence check compared this iteration's
    # fit against the previous one using the *old* point-selection
    # mask, before the newly-computed outlier mask was ever applied to
    # a refit -- on the first pass those two fits are bit-identical,
    # so sigma-clipping was silently a no-op on every call. Fixed here
    # (see estimate_baseline_robust) to refit against the updated mask
    # before checking convergence, so a spurious "peak" sitting in a
    # baseline candidate region is actually excluded.
    x = np.linspace(-5, 5, 200)
    b0_true, b1_true = 0.5, 0.1
    y = b0_true + b1_true * x
    y[95:105] += 50.0  # a "peak" sitting inside the baseline candidate region

    b0, b1, x_ref = fitting.estimate_baseline_robust(x, y, baseline_regions=(-5, 5), sigma=3.0)

    assert b1 == pytest.approx(b1_true, abs=1e-2)
    assert (b0 + b1 * (0.0 - x_ref)) == pytest.approx(b0_true, abs=1e-2)


# ============================================================
# Integrals
# ============================================================

def test_compute_integrals_matches_analytic_gaussian_and_lorentzian():
    x = np.linspace(-100, 100, 200001)
    A, fwhm = 2.0, 0.5

    y_gauss = gaussian(x, 0.0, fwhm, A)
    y_lorentz = lorentzian(x, 0.0, fwhm, A)
    baseline = np.zeros_like(x)

    integrals, integrals_norm = fitting.compute_integrals(x, [y_gauss, y_lorentz, baseline])

    expected_gauss = A * fwhm * np.sqrt(np.pi / (4 * np.log(2)))
    expected_lorentz = A * fwhm * np.pi / 2

    assert integrals[0] == pytest.approx(expected_gauss, rel=1e-3)
    assert integrals[1] == pytest.approx(expected_lorentz, rel=1e-2)


# ============================================================
# fit_spectrum
# ============================================================

def test_fit_spectrum_recovers_single_gaussian_peak():
    x0_true, fwhm_true, A_true = 5.0, 0.05, 1.0
    b0_true, b1_true = 0.02, 0.01
    spec = _make_1d(x0=x0_true, fwhm=fwhm_true, A=A_true, kind="gaussian", b0=b0_true, b1=b1_true)

    peak_defs = [{"type": "gaussian", "shift": 5.02, "fwhm": 0.06}]
    popt, pcov_free, is_free, scale, x_ref = fitting.fit_spectrum(spec, peak_defs, use_normalised=False)

    assert popt[0] == pytest.approx(x0_true, abs=1e-3)
    assert popt[1] == pytest.approx(fwhm_true, abs=1e-3)
    assert popt[2] == pytest.approx(A_true, abs=1e-2)


def test_fit_spectrum_recovers_two_peaks_gaussian_and_lorentzian():
    ppm = np.linspace(0, 10, 2000)
    x_ref_true = np.mean(ppm)
    y = (
        gaussian(ppm, 3.0, 0.04, 0.8)
        + lorentzian(ppm, 7.0, 0.06, 1.2)
        + 0.01 + 0.002 * (ppm - x_ref_true)
    )
    metadata = NMRMetadata(vendor="bruker", nucleus="1H", sfo1=400.0)
    axis = NMRAxis(kind="frequency", ppm=ppm, hz=ppm * 400.0)
    spec = NMRSpectrum(data=y, axes=[axis], metadata=metadata)

    peak_defs = [
        {"type": "gaussian", "shift": 3.05, "fwhm": 0.05},
        {"type": "lorentzian", "shift": 6.95, "fwhm": 0.05},
    ]
    popt, pcov_free, is_free, scale, x_ref = fitting.fit_spectrum(spec, peak_defs, use_normalised=False)

    assert popt[0] == pytest.approx(3.0, abs=1e-3)
    assert popt[1] == pytest.approx(0.04, abs=2e-3)
    assert popt[2] == pytest.approx(0.8, abs=1e-2)
    assert popt[3] == pytest.approx(7.0, abs=1e-3)
    assert popt[4] == pytest.approx(0.06, abs=2e-3)
    assert popt[5] == pytest.approx(1.2, abs=1e-2)


# ============================================================
# 2D slice selection / fitting
# ============================================================

def test_select_2d_traces_by_shift_range():
    spec = _make_2d_for_fitting()
    indices = fitting.select_2d_traces(spec, axis="f1", shift_range=(89, 101))
    assert indices == [1, 2, 3]


def test_select_2d_traces_fit_all():
    spec = _make_2d_for_fitting()
    assert fitting.select_2d_traces(spec, axis="f1", fit_all=True) == [1, 2, 3]


def test_nmr2d_fit_slices_recovers_amplitudes_via_get_slices():
    amps = (0.5, 1.0, 1.5)
    spec = _make_2d_for_fitting(amps=amps)

    slices = slicing.get_slices(spec, rows=[1, 2, 3])
    peak_defs = [{"type": "gaussian", "shift": 5.02, "fwhm": 0.05}]

    fitted = fitting.NMR2D_fit_slices(
        slices, peak_defs, mode="rows", normalize=False,
        show_report=False, show_warnings=False,
    )

    rows = fitted["rows"]
    assert len(rows) == 3
    assert fitted["columns"] == []

    recovered_amps = [r["popt"][2] for r in rows]
    np.testing.assert_allclose(recovered_amps, amps, atol=1e-2)

    for expected_index, r in zip([1, 2, 3], rows):
        assert r["index"] == expected_index
        assert r["slice_type"] == "row"
        assert r["slice"].metadata.raw["slice"]["index"] == expected_index
        assert r["integrals"][0] > 0
