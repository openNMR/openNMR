import numpy as np
import pytest

from opennmr.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
from opennmr.processing import fitting, slicing
from opennmr.processing.fitting import gaussian, lorentzian


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

# ============================================================
# Auto peak-picking
# ============================================================

def test_auto_pick_peaks_finds_single_peak():
    spec = _make_1d(x0=5.0, fwhm=0.05, A=1.0, b0=0.0)
    peaks = fitting.auto_pick_peaks(spec, prominence=0.1)
    assert len(peaks) == 1
    assert peaks[0]["type"] == "pseudo_voigt"
    assert peaks[0]["shift"] == pytest.approx(5.0, abs=0.02)
    assert peaks[0]["fwhm"] > 0
    assert peaks[0]["eta"] == pytest.approx(0.5)


def _make_two_peak_1d():
    ppm = np.linspace(3.0, 8.0, 2000)
    y = gaussian(ppm, 4.0, 0.05, 1.0) + gaussian(ppm, 6.5, 0.05, 0.6)
    metadata = NMRMetadata(vendor="bruker", nucleus="1H", sfo1=400.0)
    axis = NMRAxis(kind="frequency", ppm=ppm, hz=ppm * 400.0)
    return NMRSpectrum(data=y, axes=[axis], metadata=metadata)


def test_auto_pick_peaks_finds_two_separated_peaks_sorted_by_position():
    peaks = fitting.auto_pick_peaks(_make_two_peak_1d(), prominence=0.1)
    assert len(peaks) == 2
    assert [p["shift"] for p in peaks] == sorted(p["shift"] for p in peaks)
    assert peaks[0]["shift"] == pytest.approx(4.0, abs=0.02)
    assert peaks[1]["shift"] == pytest.approx(6.5, abs=0.02)


def test_auto_pick_peaks_respects_max_peaks_strongest_first():
    peaks = fitting.auto_pick_peaks(_make_two_peak_1d(), prominence=0.1, max_peaks=1)
    assert len(peaks) == 1
    assert peaks[0]["shift"] == pytest.approx(4.0, abs=0.02)  # the taller peak (A=1.0 vs 0.6)


def test_auto_pick_peaks_high_prominence_finds_nothing_in_pure_noise():
    ppm = np.linspace(0, 10, 500)
    rng = np.random.default_rng(0)
    y = rng.normal(0, 0.01, size=ppm.shape)
    metadata = NMRMetadata(vendor="bruker", nucleus="1H", sfo1=400.0)
    axis = NMRAxis(kind="frequency", ppm=ppm, hz=ppm * 400.0)
    spec = NMRSpectrum(data=y, axes=[axis], metadata=metadata)

    assert fitting.auto_pick_peaks(spec, prominence=0.9) == []


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


def test_estimate_baseline_polynomial_degree_1_matches_robust():
    # estimate_baseline_robust is documented as a thin degree=1 wrapper
    # around estimate_baseline_polynomial -- confirm they actually agree,
    # not just that each independently looks reasonable.
    x = np.linspace(-5, 5, 200)
    y = 0.5 + 0.1 * x + np.random.default_rng(0).normal(scale=0.01, size=x.shape)

    b0, b1, x_ref = fitting.estimate_baseline_robust(x, y, baseline_regions=(-5, 5), sigma=3.0)
    coeffs, poly_x_ref = fitting.estimate_baseline_polynomial(x, y, baseline_regions=(-5, 5), sigma=3.0, degree=1)

    assert poly_x_ref == pytest.approx(x_ref)
    assert coeffs[0] == pytest.approx(b1, abs=1e-9)
    assert coeffs[1] == pytest.approx(b0, abs=1e-9)


def test_estimate_baseline_polynomial_recovers_quadratic_curvature():
    # A degree-1 (linear) fit structurally cannot follow real
    # curvature regardless of which points it's fit to -- degree=2
    # should recover it directly.
    x = np.linspace(-5, 5, 300)
    true_coeffs = [0.2, -0.3, 1.0]  # a*x^2 + b*x + c
    y = np.polyval(true_coeffs, x)

    coeffs, x_ref = fitting.estimate_baseline_polynomial(x, y, baseline_regions=(-5, 5), sigma=3.0, degree=2)
    recovered = np.polyval(coeffs, x - x_ref)

    np.testing.assert_allclose(recovered, y, atol=1e-6)


def test_estimate_baseline_polynomial_rejects_a_peak():
    x = np.linspace(-5, 5, 300)
    y = 0.2 * x ** 2 - 0.3 * x + 1.0
    y_with_peak = y.copy()
    y_with_peak[145:155] += 50.0  # a "peak" sitting inside the baseline candidate region

    coeffs, x_ref = fitting.estimate_baseline_polynomial(
        x, y_with_peak, baseline_regions=(-5, 5), sigma=3.0, degree=2,
    )
    recovered = np.polyval(coeffs, x - x_ref)

    np.testing.assert_allclose(recovered, y, atol=0.1)


def test_estimate_baseline_sinusoidal_recovers_a_known_wave():
    x = np.linspace(-10, 10, 400)
    period = 40.0  # twice the x-range -- matches the default period estimate
    true = 2.0 + 3.0 * np.cos(2 * np.pi * x / period) - 1.5 * np.sin(2 * np.pi * x / period)

    coeffs, fitted_period = fitting.estimate_baseline_sinusoidal(x, true, baseline_regions=(-10, 10), sigma=3.0)
    recovered = fitting.evaluate_baseline_sinusoidal(coeffs, fitted_period, x)

    np.testing.assert_allclose(recovered, true, atol=1e-6)


def test_estimate_baseline_sinusoidal_rejects_a_peak():
    x = np.linspace(-10, 10, 400)
    period = 40.0
    true = 2.0 + 3.0 * np.cos(2 * np.pi * x / period)
    y_with_peak = true.copy()
    y_with_peak[195:205] += 50.0

    coeffs, fitted_period = fitting.estimate_baseline_sinusoidal(x, y_with_peak, baseline_regions=(-10, 10), sigma=3.0)
    recovered = fitting.evaluate_baseline_sinusoidal(coeffs, fitted_period, x)

    np.testing.assert_allclose(recovered, true, atol=0.5)


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


def test_fit_spectrum_clips_data_driven_initial_guess_into_a_tight_manual_bound():
    # Real bug, found while building amplitude_bound_regions (below):
    # initialize_params_with_map's own initial amplitude guess is read
    # straight off the data (the y-value at the peak's own starting
    # shift), with no knowledge of whatever bounds will actually be
    # used -- a manual "Set Bounds..." amplitude entry (or
    # amplitude_bound_regions) tighter than that raw data value used to
    # crash scipy's bounded curve_fit outright ("Initial guess is
    # outside of provided bounds"), rather than being respected.
    x0_true, fwhm_true, A_true = 5.0, 0.05, 2.0
    spec = _make_1d(x0=x0_true, fwhm=fwhm_true, A=A_true, kind="gaussian", b0=0.0, b1=0.0)

    tight_upper = 0.5  # well below the true peak height (2.0) the raw-data guess would read
    peak_defs = [{"type": "gaussian", "shift": x0_true, "fwhm": fwhm_true,
                  "bounds": {"amplitude": (-np.inf, tight_upper)}}]

    popt, _pcov_free, _is_free, _scale, _x_ref = fitting.fit_spectrum(spec, peak_defs, use_normalised=False)

    assert popt[2] == pytest.approx(tight_upper, rel=1e-3)  # pinned at the bound, not a crash


def test_fit_spectrum_fixed_dict_locks_only_the_named_parameter():
    # peak_defs' own "fixed" dict is per-parameter (e.g. {"shift":
    # True} locks only the shift, leaving FWHM/amplitude free) -- the
    # GUI now exposes this directly (individual parameters, not just
    # an entire peak, fixable), but nothing at this backend level had
    # ever actually verified a *partial* fixed dict numerically before
    # (only the GUI's own all-four-True "Fixed" checkbox case had any
    # coverage at all). Deliberately wrong starting shift (5.2, true
    # 5.0) with only "shift" fixed: if fixed were silently ignored,
    # the fit would pull the shift back toward 5.0; locked correctly,
    # it must stay pinned exactly at its own starting value while FWHM
    # still converges close to the true value around it.
    x0_true, fwhm_true, A_true = 5.0, 0.05, 1.0
    spec = _make_1d(x0=x0_true, fwhm=fwhm_true, A=A_true, kind="gaussian", b0=0.0, b1=0.0)

    wrong_shift = 5.02
    peak_defs = [{"type": "gaussian", "shift": wrong_shift, "fwhm": 0.08, "fixed": {"shift": True}}]
    popt, _pcov_free, is_free, _scale, _x_ref = fitting.fit_spectrum(spec, peak_defs, use_normalised=False)

    assert popt[0] == pytest.approx(wrong_shift, abs=1e-9)  # locked, not pulled toward x0_true
    assert not is_free[0]
    assert is_free[1] and is_free[2]  # FWHM/amplitude still free
    assert popt[1] == pytest.approx(fwhm_true, abs=0.03)  # widens slightly to still cover the off-center true peak


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


def _make_drifting_peak_series(n_slices=6, drift_per_slice=0.15, x0=5.0, fwhm=0.05, A=1.0, n_pts=800):
    """A pseudo-2D series where the peak's own true position drifts
    slice to slice -- e.g. a temperature-dependent shift -- used to
    exercise seed_mode, which today's default ("independent", every
    slice fit from the same user-entered guess) has no way to track."""
    ppm = np.linspace(3.0, 7.0, n_pts)
    true_shifts = [x0 + drift_per_slice * i for i in range(n_slices)]
    data = np.array([gaussian(ppm, s, fwhm, A) for s in true_shifts])
    f1_axis = NMRAxis(kind="index", index=np.arange(1, n_slices + 1))
    f2_axis = NMRAxis(kind="frequency", ppm=ppm, hz=ppm * 400.0)
    metadata = NMRMetadata(vendor="bruker", nucleus="1H", sfo1=400.0, pseudo_2d=True)
    spectrum = NMRSpectrum(data=data, axes=[f1_axis, f2_axis], metadata=metadata)
    return spectrum, true_shifts


def test_nmr2d_fit_slices_seed_mode_independent_can_lose_track_of_drift():
    # The problem seed_mode exists to solve: fitting-wizard roadmap
    # item "fit one slice first, then... use them as each next slice's
    # own starting guess... rather than every slice fit independently
    # from scratch, today's behaviour." Confirms that behaviour is
    # real -- a peak that drifts several FWHM away from the table's
    # own single starting guess isn't reliably recovered by later
    # slices under the default mode.
    spectrum, true_shifts = _make_drifting_peak_series()
    slices = slicing.get_slices(spectrum, rows=list(range(1, len(true_shifts) + 1)))
    peak_defs = [{"type": "gaussian", "shift": true_shifts[0], "fwhm": 0.05}]

    fitted = fitting.NMR2D_fit_slices(slices, peak_defs, mode="rows",
                                       show_report=False, show_warnings=False, seed_mode="independent")

    fitted_shifts = [r["popt"][0] for r in fitted["rows"]]
    assert fitted_shifts[0] == pytest.approx(true_shifts[0], abs=1e-2)  # first slice always recovers fine
    assert fitted_shifts[-1] != pytest.approx(true_shifts[-1], abs=1e-2)  # last slice does not


def test_nmr2d_fit_slices_seed_mode_propagate_tracks_drift():
    spectrum, true_shifts = _make_drifting_peak_series()
    slices = slicing.get_slices(spectrum, rows=list(range(1, len(true_shifts) + 1)))
    peak_defs = [{"type": "gaussian", "shift": true_shifts[0], "fwhm": 0.05}]

    fitted = fitting.NMR2D_fit_slices(slices, peak_defs, mode="rows",
                                       show_report=False, show_warnings=False, seed_mode="propagate")

    fitted_shifts = [r["popt"][0] for r in fitted["rows"]]
    np.testing.assert_allclose(fitted_shifts, true_shifts, atol=1e-2)


def test_nmr2d_fit_slices_seed_mode_apply_first_fixes_shift_and_fwhm():
    spectrum, true_shifts = _make_drifting_peak_series()
    slices = slicing.get_slices(spectrum, rows=list(range(1, len(true_shifts) + 1)))
    peak_defs = [{"type": "gaussian", "shift": true_shifts[0], "fwhm": 0.05}]

    fitted = fitting.NMR2D_fit_slices(slices, peak_defs, mode="rows",
                                       show_report=False, show_warnings=False, seed_mode="apply_first")

    rows = fitted["rows"]
    first_shift, first_fwhm = rows[0]["popt"][0], rows[0]["popt"][1]
    for r in rows[1:]:
        assert r["popt"][0] == pytest.approx(first_shift, abs=1e-9)  # locked, not tracking the true drift
        assert r["popt"][1] == pytest.approx(first_fwhm, abs=1e-9)
        assert not r["is_free"][0] and not r["is_free"][1]  # shift/FWHM genuinely marked not-free
        assert r["is_free"][2]  # amplitude still free


def test_nmr2d_fit_slices_invalid_seed_mode_raises():
    spec = _make_2d_for_fitting()
    slices = slicing.get_slices(spec, rows=[1])
    with pytest.raises(ValueError):
        fitting.NMR2D_fit_slices(slices, [{"type": "gaussian", "shift": 5.0, "fwhm": 0.04}],
                                  mode="rows", seed_mode="not-a-real-mode")


def _make_single_gaussian_slice(x0=5.0, fwhm=0.2, A=2.0, n_pts=4000):
    """A one-slice pseudo-2D wrapping a single, clean, isolated Gaussian
    -- used to test amplitude_bound_regions, where the exact area under
    a known analytic peak needs to be hand-computable."""
    ppm = np.linspace(3.0, 7.0, n_pts)
    y = gaussian(ppm, x0, fwhm, A)
    f1_axis = NMRAxis(kind="index", index=np.array([1]))
    f2_axis = NMRAxis(kind="frequency", ppm=ppm, hz=ppm * 400.0)
    metadata = NMRMetadata(vendor="bruker", nucleus="1H", sfo1=400.0, pseudo_2d=True)
    spectrum = NMRSpectrum(data=y[np.newaxis, :], axes=[f1_axis, f2_axis], metadata=metadata)
    return spectrum, ppm, y


def test_nmr2d_fit_slices_amplitude_bound_regions_computes_correct_bound_and_constrains_the_fit():
    # Real fitting-wizard-roadmap item: "bounding possibly from
    # integrals." Region deliberately narrower than the true peak's own
    # support (±0.05 ppm around a 0.2 ppm-FWHM peak) -- captures only a
    # fraction of its true area, so the derived bound must sit below
    # the true amplitude, and curve_fit (given no other way to reduce
    # residual further) should converge to that bound's own edge.
    x0, fwhm, true_A = 5.0, 0.2, 2.0
    spectrum, ppm, y = _make_single_gaussian_slice(x0, fwhm, true_A)
    slices = slicing.get_slices(spectrum, rows=[1])
    region = (4.95, 5.05)
    peak_defs = [{"type": "gaussian", "shift": x0, "fwhm": fwhm}]

    fitted = fitting.NMR2D_fit_slices(slices, peak_defs, mode="rows", show_report=False, show_warnings=False,
                                       amplitude_bound_regions=[region])

    result = fitted["rows"][0]
    lo, hi = result["peak_defs"][0]["bounds"]["amplitude"]
    assert lo == -np.inf

    mask = (ppm >= region[0]) & (ppm <= region[1])
    expected_integral = fitting._trapz(y[mask], ppm[mask])
    expected_bound = expected_integral / (fwhm * np.sqrt(np.pi / (4 * np.log(2))))
    assert hi == pytest.approx(expected_bound, rel=1e-6)
    assert expected_bound < true_A  # confirms the region really is the tighter constraint here

    fitted_A = result["popt"][2]
    assert fitted_A == pytest.approx(expected_bound, rel=1e-3)
    assert fitted_A < true_A  # genuinely capped, not coincidentally already there


def test_nmr2d_fit_slices_amplitude_bound_regions_respects_explicit_user_bound():
    # A peak's own explicit bounds (e.g. from the GUI's "Set Bounds..."
    # dialog) must always win outright -- this feature only ever fills
    # in an upper bound where none already exists.
    spectrum, ppm, y = _make_single_gaussian_slice()
    slices = slicing.get_slices(spectrum, rows=[1])
    peak_defs = [{"type": "gaussian", "shift": 5.0, "fwhm": 0.2, "bounds": {"amplitude": (-np.inf, 7.0)}}]

    fitted = fitting.NMR2D_fit_slices(slices, peak_defs, mode="rows", show_report=False, show_warnings=False,
                                       amplitude_bound_regions=[(4.5, 5.5)])

    assert fitted["rows"][0]["peak_defs"][0]["bounds"]["amplitude"] == (-np.inf, 7.0)


def test_nmr2d_fit_slices_amplitude_bound_regions_off_by_default():
    spectrum, ppm, y = _make_single_gaussian_slice()
    slices = slicing.get_slices(spectrum, rows=[1])
    peak_defs = [{"type": "gaussian", "shift": 5.0, "fwhm": 0.2}]

    fitted = fitting.NMR2D_fit_slices(slices, peak_defs, mode="rows", show_report=False, show_warnings=False)

    assert "amplitude" not in fitted["rows"][0]["peak_defs"][0].get("bounds", {})


def test_nmr2d_fit_slices_amplitude_bound_regions_uses_the_tightest_overlapping_region():
    x0, fwhm, true_A = 5.0, 0.2, 2.0
    spectrum, ppm, y = _make_single_gaussian_slice(x0, fwhm, true_A)
    slices = slicing.get_slices(spectrum, rows=[1])
    narrow, wide = (4.95, 5.05), (4.0, 6.0)
    peak_defs = [{"type": "gaussian", "shift": x0, "fwhm": fwhm}]

    # Order shouldn't matter -- pass the looser region first to confirm
    # the tighter one isn't just "whichever came first."
    fitted = fitting.NMR2D_fit_slices(slices, peak_defs, mode="rows", show_report=False, show_warnings=False,
                                       amplitude_bound_regions=[wide, narrow])

    _, hi = fitted["rows"][0]["peak_defs"][0]["bounds"]["amplitude"]

    narrow_mask = (ppm >= narrow[0]) & (ppm <= narrow[1])
    narrow_integral = fitting._trapz(y[narrow_mask], ppm[narrow_mask])
    expected = narrow_integral / (fwhm * np.sqrt(np.pi / (4 * np.log(2))))
    assert hi == pytest.approx(expected, rel=1e-6)
    assert hi < true_A  # sanity: the narrow region is indeed the tighter of the two


def test_nmr2d_fit_slices_amplitude_bound_regions_skips_a_nonpositive_integral():
    # A genuinely negative-going peak (e.g. an inversion-recovery slice
    # before the null point) has a negative raw integral over its own
    # region -- must not silently force a near-zero amplitude cap here;
    # explicit S/N gating (noise_region/sn_threshold) is the deliberate
    # tool for "pin amplitude to zero," not a side effect of this one.
    x0, fwhm, true_A = 5.0, 0.2, -1.5
    spectrum, ppm, y = _make_single_gaussian_slice(x0, fwhm, true_A)
    slices = slicing.get_slices(spectrum, rows=[1])
    peak_defs = [{"type": "gaussian", "shift": x0, "fwhm": fwhm}]

    fitted = fitting.NMR2D_fit_slices(slices, peak_defs, mode="rows", show_report=False, show_warnings=False,
                                       amplitude_bound_regions=[(4.5, 5.5)],
                                       positive_intensity=False, allow_negative_intensity=True)

    result = fitted["rows"][0]
    assert "amplitude" not in result["peak_defs"][0].get("bounds", {})
    assert result["popt"][2] == pytest.approx(true_A, abs=1e-2)  # unaffected, correctly recovers the true value
