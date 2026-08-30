import numpy as np
import pytest

from opennmr.processing import ft


def test_roll_fid_integer_shift():
    fid = np.array([1.0, 2.0, 3.0, 4.0], dtype=complex)
    rolled = ft.roll_fid(fid, 1)
    np.testing.assert_allclose(rolled, [4.0, 1.0, 2.0, 3.0])


def test_zero_fill_pads_and_truncates():
    fid = np.array([1.0, 2.0, 3.0], dtype=complex)
    assert len(ft.zero_fill(fid, 5)) == 5
    np.testing.assert_allclose(ft.zero_fill(fid, 5)[3:], [0, 0])
    assert len(ft.zero_fill(fid, 2)) == 2


def test_apodize_exponential_decays():
    fid = np.ones(100, dtype=complex)
    windowed = ft.apodize(fid, dt=1e-3, window="exponential", lb=50.0)
    assert windowed[0] == 1.0
    assert np.all(np.diff(np.abs(windowed)) <= 0)  # monotonically decaying


def test_apodize_unknown_window_raises():
    fid = np.ones(10, dtype=complex)
    try:
        ft.apodize(fid, dt=1e-3, window="not-a-window")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


def test_fft_spectrum_peak_at_zero_freq_for_dc_fid():
    dt = 1e-4
    fid = np.ones(256, dtype=complex)  # pure DC -> single peak at 0 Hz
    freq_hz, spectrum = ft.fft_spectrum(fid, dt)
    peak_freq = freq_hz[np.argmax(np.abs(spectrum))]
    assert abs(peak_freq) < 1e-6


def test_apply_phase_zero_order_rotates_real_to_imag():
    spectrum = np.array([1.0 + 0j, 1.0 + 0j])
    freq = np.array([0.0, 0.0])
    rotated = ft.apply_phase(spectrum, freq, phase0_deg=90, phase1_deg=0)
    np.testing.assert_allclose(np.real(rotated), [0, 0], atol=1e-10)


def test_apply_phase_at_pivot_equals_phase0_only():
    # At the pivot itself, phase1's contribution should vanish
    # entirely regardless of where the pivot sits in the window.
    freq = np.linspace(0, 100, 1001)
    spectrum = np.ones(freq.shape, dtype=complex)
    pivot_ppm = 30.0
    pivot_frac = (pivot_ppm - freq.min()) / (freq.max() - freq.min())

    phased = ft.apply_phase(spectrum, freq, phase0_deg=10.0, phase1_deg=40.0, pivot_frac=pivot_frac)

    idx_pivot = np.argmin(np.abs(freq - pivot_ppm))
    angle_at_pivot = np.degrees(np.angle(phased[idx_pivot]))
    assert angle_at_pivot == pytest.approx(-10.0, abs=1e-6)


def test_apply_phase_edge_to_edge_difference_equals_phase1():
    # phase1_deg is the *total* phase swing from one edge of the axis
    # to the other, regardless of pivot position -- the standard NMR
    # convention (matches e.g. Bruker's PHC1).
    freq = np.linspace(0, 100, 1001)
    spectrum = np.ones(freq.shape, dtype=complex)

    for pivot_frac in (0.0, 0.3, 0.5, 1.0):
        phased = ft.apply_phase(spectrum, freq, phase0_deg=0.0, phase1_deg=40.0, pivot_frac=pivot_frac)
        angle_max = np.degrees(np.angle(phased[-1]))
        angle_min = np.degrees(np.angle(phased[0]))
        assert (angle_max - angle_min) == pytest.approx(-40.0, abs=1e-6)


def test_apply_phase_pivot_outside_window_still_linear():
    # A pivot outside the visible window is a legitimate extrapolation
    # of the same linear ramp, not a special case.
    freq = np.linspace(0, 100, 1001)
    spectrum = np.ones(freq.shape, dtype=complex)
    pivot_frac = -0.5  # 50 ppm below freq.min()

    phased = ft.apply_phase(spectrum, freq, phase0_deg=0.0, phase1_deg=40.0, pivot_frac=pivot_frac)
    angle_max = np.degrees(np.angle(phased[-1]))
    angle_min = np.degrees(np.angle(phased[0]))
    assert (angle_max - angle_min) == pytest.approx(-40.0, abs=1e-6)


def test_apc0_recovers_known_phase_on_noisy_low_snr_spectrum():
    # Regression test: apc0 used to score by raw sum(real(corrected))
    # over the whole array, which a wide, mostly-noise/baseline window
    # with only a small real peak (e.g. a late PFG gradient-step
    # slice, reported directly as "Autophase... zeroing everything")
    # can dominate, picking a phase driven by noise rather than the
    # peak. Now scores by phase_score's "magnitude" penalty instead
    # (phase-independent by construction), which recovers the true
    # phase far more reliably at the same noise level -- confirmed via
    # a direct side-by-side synthetic comparison before making this
    # change (old scoring: off by 11 deg at this noise level; new: 2).
    n, sw = 4096, 4000.0
    dt = 1.0 / sw
    t = np.arange(n) * dt
    fid = np.exp(-15.0 * t) * np.exp(2j * np.pi * 200.0 * t)
    rng = np.random.default_rng(0)
    fid = fid + rng.normal(0, 0.1, n) + 1j * rng.normal(0, 0.1, n)
    freq, spectrum = ft.fft_spectrum(fid, dt)

    true_phase0 = 73.0
    scrambled = spectrum * np.exp(1j * np.deg2rad(true_phase0))
    recovered = ft.apc0(scrambled, freq, phase0_inc=0.5)

    assert abs(((recovered - true_phase0 + 180) % 360) - 180) < 5.0


def test_autophase01_recovers_known_phase():
    # A single absorption-mode Lorentzian, phased by a known amount --
    # autophase01 should recover (close to) the negative of that phase.
    freq = np.linspace(-500, 500, 2001)
    lineshape = 1.0 / (1.0 + (freq / 20.0) ** 2)
    true_phase0 = 37.0
    phased = ft.apply_phase(lineshape.astype(complex), freq, -true_phase0, 0.0)
    phase0, phase1 = ft.autophase01(phased, freq, phase0_inc=1.0)
    assert abs(((phase0 - true_phase0 + 180) % 360) - 180) < 2.0


def test_linear_predict_extends_length():
    fid = np.exp(-np.arange(50) * 0.05).astype(complex)
    extended = ft.linear_predict(fid, order=10, n_predict=5, direction="forward")
    assert len(extended) == 55


def test_align_echo_defaults_to_index_zero():
    # Matches leonmr's xpulse_imaging.py (validated on real gradient-echo
    # data): the echo top rolls to t=0, not the array centre -- so what
    # follows (window/zero-fill/FFT) sees an ordinary one-sided decay.
    n = 64
    fid = np.zeros(n, dtype=complex)
    fid[10] = 1.0  # echo peak off-center
    aligned = ft.align_echo(fid)
    assert np.argmax(np.abs(aligned)) == 0


def test_align_echo_respects_explicit_target_index():
    n = 64
    fid = np.zeros(n, dtype=complex)
    fid[10] = 1.0
    aligned = ft.align_echo(fid, target_index=n // 2)
    assert np.argmax(np.abs(aligned)) == n // 2


def _sloped_baseline_with_peak(n=2000):
    freq = np.linspace(-10.0, 10.0, n)
    true_baseline = 5.0 + 0.3 * freq
    peak = 80.0 * np.exp(-0.5 * ((freq - 1.0) / 0.2) ** 2)
    return freq, true_baseline, peak


def test_baseline_correct_is_not_biased_by_a_tall_peak():
    # A single unweighted polyfit over the whole spectrum (this
    # function's own original implementation) gets pulled toward a
    # tall peak, systematically under-correcting the true baseline --
    # this is the "isn't achieving much" complaint the sigma-clipped
    # rewrite fixes. Compare against the naive fit directly to prove
    # the new version does substantially better, not just differently.
    freq, true_baseline, peak = _sloped_baseline_with_peak()
    spectrum = true_baseline + peak

    naive_coeffs = np.polyfit(freq, spectrum, 1)
    naive_corrected = spectrum - np.polyval(naive_coeffs, freq)

    corrected = ft.baseline_correct(freq, spectrum)

    # Away from the peak, the robust correction should sit much closer
    # to zero (true baseline fully removed) than the naive one.
    away_from_peak = np.abs(freq - 1.0) > 2.0
    naive_residual = np.abs(naive_corrected[away_from_peak]).mean()
    robust_residual = np.abs(corrected[away_from_peak]).mean()
    assert robust_residual < naive_residual * 0.1


def test_baseline_correct_preserves_complex_dtype():
    freq, true_baseline, peak = _sloped_baseline_with_peak()
    spectrum = (true_baseline + peak) + 1j * np.zeros_like(freq)
    corrected = ft.baseline_correct(freq, spectrum)
    assert np.iscomplexobj(corrected)


def test_baseline_correct_handles_a_short_spectrum():
    # A small synthetic/test-fixture-sized spectrum (far fewer points
    # than any real acquisition) shouldn't hit estimate_baseline_robust's
    # "not enough points" floor via frac=0.05 rounding down to nothing.
    freq = np.linspace(-5.0, 5.0, 32)
    spectrum = 2.0 + 0.1 * freq
    corrected = ft.baseline_correct(freq, spectrum)
    assert np.allclose(corrected, 0.0, atol=1e-6)


def test_linear_predict_backward_reconstructs_points_before_the_start():
    # Backward LP fits the same AR model on a reversed copy of the FID,
    # extends it, then reverses back -- reconstructing points *before*
    # the measured start (e.g. lost to a dead-time delay) rather than
    # extending past the measured end.
    n = 60
    t = np.arange(n)
    true_signal = np.exp(-0.05 * t) * np.exp(1j * 0.3 * t)
    fid = true_signal[10:]  # first 10 points "missing"

    extended = ft.linear_predict(fid, order=15, n_predict=10, direction="backward")

    assert len(extended) == len(fid) + 10
    np.testing.assert_allclose(extended[:10], true_signal[:10], atol=1e-6)
    np.testing.assert_allclose(extended[10:], fid)  # the real, measured points are untouched


def test_linear_predict_rejects_unknown_direction():
    with pytest.raises(ValueError):
        ft.linear_predict(np.ones(20, dtype=complex), order=5, n_predict=3, direction="sideways")


def test_baseline_correct_respects_explicit_baseline_regions():
    freq, true_baseline, peak = _sloped_baseline_with_peak()
    spectrum = true_baseline + peak
    corrected = ft.baseline_correct(freq, spectrum, baseline_regions=[(-10.0, -2.0), (4.0, 10.0)])
    away_from_peak = np.abs(freq - 1.0) > 2.0
    assert np.abs(corrected[away_from_peak]).mean() < 1.0


def _curved_baseline_with_peak(n=2000):
    freq = np.linspace(-10.0, 10.0, n)
    true_baseline = 3.0 - 0.1 * freq + 0.02 * freq ** 2  # genuinely curved, not just sloped
    peak = 5.0 * np.exp(-0.5 * ((freq - 1.0) / 0.3) ** 2)
    return freq, true_baseline, peak


def test_baseline_correct_spline_handles_curved_baseline_linear_cannot():
    # Explicit user feedback: linear-only baseline correction "doesn't
    # work so well" -- a straight line structurally cannot follow a
    # curved (rolling) baseline no matter which points it's fit to,
    # common on wideline/solid-state spectra. Spline can.
    freq, true_baseline, peak = _curved_baseline_with_peak()
    spectrum = true_baseline + peak

    corrected_linear = ft.baseline_correct(freq, spectrum, method="linear")
    corrected_spline = ft.baseline_correct(freq, spectrum, method="spline")

    away_from_peak = np.abs(freq - 1.0) > 2.0
    linear_residual = np.abs(corrected_linear[away_from_peak]).mean()
    spline_residual = np.abs(corrected_spline[away_from_peak]).mean()
    assert spline_residual < linear_residual * 0.05
    assert spline_residual < 0.05  # genuinely removed, not just "better than linear"


def test_baseline_correct_spline_smoothing_override_is_respected():
    # Proof the smoothing parameter is actually threaded through (not
    # silently ignored): a noisy baseline fit with s=0 (exact
    # interpolation through every candidate point) must differ
    # measurably from the same fit with a large, heavily-smoothing s.
    rng = np.random.default_rng(3)
    freq, true_baseline, peak = _curved_baseline_with_peak(n=500)
    noisy_baseline = true_baseline + rng.normal(0.0, 0.05, size=freq.shape)
    spectrum = noisy_baseline + peak

    corrected_exact = ft.baseline_correct(freq, spectrum, method="spline", smoothing=0.0)
    corrected_smooth = ft.baseline_correct(freq, spectrum, method="spline", smoothing=1.0e6)
    assert not np.allclose(corrected_exact, corrected_smooth)


def test_baseline_correct_rejects_unknown_method():
    freq, true_baseline, peak = _sloped_baseline_with_peak()
    with pytest.raises(ValueError):
        ft.baseline_correct(freq, true_baseline + peak, method="quadratic")
