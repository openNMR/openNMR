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


def test_fft_spectrum_2d_input_matches_last_axis_length():
    # Real bug: a 2D array (a batch of FIDs stacked along axis 0, e.g.
    # F1 rows each holding a length-64 FID) used to produce a freq_hz
    # matching the *row count* (len(fid) reads axis 0) instead of the
    # row length -- a shape this project's own X-Pulse read_2d hit
    # directly (F1=128 rows, F2=2048 points -- freq_hz for the F1 FFT
    # came out length 2048, not 128, causing a real crash rendering
    # the resulting spectrum).
    dt = 1e-4
    fid_2d = np.ones((5, 64), dtype=complex)  # 5 rows, 64 points each
    freq_hz, spectrum = ft.fft_spectrum(fid_2d, dt)
    assert freq_hz.shape == (64,)
    assert spectrum.shape == (5, 64)


def test_fft_spectrum_2d_input_ffts_each_row_independently():
    # Each row should FFT exactly as it would on its own, not get
    # mixed with the others (a plain np.fft.fftshift with no axes=
    # argument would incorrectly shift axis 0 -- the row index -- too).
    dt = 1e-4
    row0 = np.exp(1j * 2 * np.pi * 1000.0 * np.arange(64) * dt)
    row1 = np.ones(64, dtype=complex)
    fid_2d = np.array([row0, row1])
    freq_hz_2d, spectrum_2d = ft.fft_spectrum(fid_2d, dt)
    freq_hz_0, spectrum_0 = ft.fft_spectrum(row0, dt)
    freq_hz_1, spectrum_1 = ft.fft_spectrum(row1, dt)
    np.testing.assert_allclose(freq_hz_2d, freq_hz_0)
    np.testing.assert_allclose(spectrum_2d[0], spectrum_0)
    np.testing.assert_allclose(spectrum_2d[1], spectrum_1)


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


def test_prepare_fid_clamps_lp_order_instead_of_raising():
    # A UI/caller-chosen order (e.g. a fixed default of 20) can easily
    # exceed what's valid once truncate has shrunk the FID -- prepare_fid
    # should still produce a real LP result at the largest valid order
    # rather than propagating linear_predict's ValueError.
    fid = np.array([1.0, 0.9, 0.8], dtype=complex)  # len 3 -> max valid order is 1
    extended = ft.prepare_fid(fid, linear_predict_points=4, linear_predict_order=20)
    assert len(extended) == 7
    np.testing.assert_allclose(extended[:3], fid)


def test_prepare_fid_skips_lp_when_no_order_is_valid():
    # Fewer than 3 points: even order=1 isn't < len(fid)-1, so LP can't
    # run at all -- prepare_fid should leave the FID untouched rather
    # than raising.
    fid = np.array([1.0, 0.9], dtype=complex)
    result = ft.prepare_fid(fid, linear_predict_points=4, linear_predict_order=20)
    np.testing.assert_allclose(result, fid)


def test_decode_hypercomplex_f1_passes_through_unpaired_modes():
    data = np.arange(12, dtype=complex).reshape(4, 3)
    for fn_mode in (0, 1, 2, 3):
        np.testing.assert_array_equal(ft.decode_hypercomplex_f1(data, fn_mode), data)


def test_decode_hypercomplex_f1_states():
    # Row 2k is the real/"cosine" row, 2k+1 the imaginary/"sine" row,
    # for the same t1 increment k -- decoded[k] = row[2k] + 1j*row[2k+1].
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=complex)
    decoded = ft.decode_hypercomplex_f1(data, fn_mode=4)
    expected = np.array([[1 + 4j, 2 + 5j, 3 + 6j], [7 + 10j, 8 + 11j, 9 + 12j]])
    np.testing.assert_allclose(decoded, expected)


def test_decode_hypercomplex_f1_states_tppi_alternates_sign():
    # Same row-pairing as plain States, plus every other hypercomplex
    # point negated -- displaces the axial artifact peak to the edge
    # of F1 instead of its centre.
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=complex)
    decoded = ft.decode_hypercomplex_f1(data, fn_mode=5)
    expected = np.array([[1 + 4j, 2 + 5j, 3 + 6j], [-(7 + 10j), -(8 + 11j), -(9 + 12j)]])
    np.testing.assert_allclose(decoded, expected)


def test_decode_hypercomplex_f1_echo_antiecho():
    echo = np.array([1, 2, 3], dtype=complex)
    antiecho = np.array([4, 5, 6], dtype=complex)
    decoded = ft.decode_hypercomplex_f1(np.array([echo, antiecho]), fn_mode=6)
    expected = (echo + antiecho) + 1j * (echo - antiecho)
    np.testing.assert_allclose(decoded, [expected])


def test_decode_hypercomplex_f1_drops_trailing_unpaired_row():
    data = np.arange(15, dtype=complex).reshape(5, 3)  # 5 rows: 2 full pairs + 1 leftover
    decoded = ft.decode_hypercomplex_f1(data, fn_mode=4)
    assert decoded.shape == (2, 3)


def _synthetic_states_pair_dataset(f1_hz, n1=16, n2=8, dt1=1e-3, states_tppi=False):
    # Forward model matching decode_hypercomplex_f1's own docstring:
    # row_R(t1) = cos(2*pi*f1_hz*t1) * f2signal, row_I(t1) = sin(...) *
    # f2signal -- a States (or States-TPPI, if the receiver-phase sign
    # alternation is also applied going in) encoded raw row-pair series
    # for a single, known F1 frequency.
    t2 = np.arange(n2)
    f2signal = np.exp(1j * 2 * np.pi * 0.1 * t2)  # arbitrary fixed F2 content, same every row
    rows = []
    for k in range(n1):
        t1 = k * dt1
        phase = 2 * np.pi * f1_hz * t1
        sign = -1.0 if (states_tppi and k % 2) else 1.0
        rows.append(sign * np.cos(phase) * f2signal)
        rows.append(sign * np.sin(phase) * f2signal)
    return np.array(rows, dtype=complex), dt1


def test_decode_hypercomplex_f1_states_recovers_known_f1_frequency():
    # End-to-end physical check, not just arithmetic: a States-encoded
    # series built for a specific, known F1 frequency should FFT (after
    # decoding) to a peak at that frequency -- and decoding first is
    # what makes this work, not incidental (see the "regression" test
    # below for the contrast).
    n1, dt1 = 16, 1e-3
    sw1 = 1.0 / dt1
    f1_hz = 3 * (sw1 / n1)  # an exact FFT bin, safe to check by bin index
    data, dt1 = _synthetic_states_pair_dataset(f1_hz, n1=n1, dt1=dt1)
    decoded = ft.decode_hypercomplex_f1(data, fn_mode=4)
    assert decoded.shape == (n1, data.shape[1])
    spectrum = np.fft.fftshift(np.fft.fft(decoded, axis=0), axes=0)
    power = np.abs(spectrum).sum(axis=1)
    freqs = np.fft.fftshift(np.fft.fftfreq(n1, d=dt1))
    assert abs(freqs[np.argmax(power)] - f1_hz) < (sw1 / n1) / 2


def test_decode_hypercomplex_f1_skipping_decode_would_be_wrong():
    # Regression guard for the exact bug this feature fixes: FFT-ing
    # the raw, still-row-paired data directly (as read_2d did before
    # decode_hypercomplex_f1 existed) does NOT recover the true F1
    # frequency -- confirms the decode step is load-bearing, not
    # cosmetic, for this synthetic case.
    n1, dt1 = 16, 1e-3
    sw1 = 1.0 / dt1
    f1_hz = 3 * (sw1 / n1)
    data, dt1 = _synthetic_states_pair_dataset(f1_hz, n1=n1, dt1=dt1)
    undecoded_spectrum = np.fft.fftshift(np.fft.fft(data, axis=0), axes=0)
    undecoded_power = np.abs(undecoded_spectrum).sum(axis=1)
    undecoded_freqs = np.fft.fftshift(np.fft.fftfreq(data.shape[0], d=dt1))
    assert abs(undecoded_freqs[np.argmax(undecoded_power)] - f1_hz) > (sw1 / n1)


def test_decode_hypercomplex_f1_states_tppi_recovers_known_f1_frequency():
    n1, dt1 = 16, 1e-3
    sw1 = 1.0 / dt1
    f1_hz = 3 * (sw1 / n1)
    data, dt1 = _synthetic_states_pair_dataset(f1_hz, n1=n1, dt1=dt1, states_tppi=True)
    decoded = ft.decode_hypercomplex_f1(data, fn_mode=5)
    spectrum = np.fft.fftshift(np.fft.fft(decoded, axis=0), axes=0)
    power = np.abs(spectrum).sum(axis=1)
    freqs = np.fft.fftshift(np.fft.fftfreq(n1, d=dt1))
    assert abs(freqs[np.argmax(power)] - f1_hz) < (sw1 / n1) / 2


def test_decode_hypercomplex_f1_echo_antiecho_recovers_a_consistent_frequency():
    # Self-consistency check only (see decode_hypercomplex_f1's own
    # docstring: no real Echo-Antiecho data exists in this project to
    # validate the absolute sign/handedness convention against). Builds
    # an echo/antiecho pair from the standard PEP forward model
    # (P-type/N-type complex exponentials sharing the same F2 content)
    # and checks *a* single, well-defined frequency comes out at the
    # expected magnitude -- not that it lands on the "positive" side.
    n1, dt1 = 16, 1e-3
    sw1 = 1.0 / dt1
    f1_hz = 3 * (sw1 / n1)
    t2 = np.arange(8)
    f2signal = np.exp(1j * 2 * np.pi * 0.1 * t2)
    rows = []
    for k in range(n1):
        t1 = k * dt1
        phase = 2 * np.pi * f1_hz * t1
        echo = np.exp(1j * phase) * f2signal
        antiecho = np.exp(-1j * phase) * f2signal
        rows.append(echo)
        rows.append(antiecho)
    data = np.array(rows, dtype=complex)
    decoded = ft.decode_hypercomplex_f1(data, fn_mode=6)
    spectrum = np.fft.fftshift(np.fft.fft(decoded, axis=0), axes=0)
    power = np.abs(spectrum).sum(axis=1)
    freqs = np.fft.fftshift(np.fft.fftfreq(n1, d=dt1))
    peak_freq = abs(freqs[np.argmax(power)])
    assert abs(peak_freq - f1_hz) < (sw1 / n1) / 2


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


def test_baseline_correct_polynomial_handles_curved_baseline_linear_cannot():
    # Same real-curvature case spline was added for -- a degree-2
    # polynomial should do it too, without needing spline's own
    # "follows anything, including noise between sparse points" risk.
    freq, true_baseline, peak = _curved_baseline_with_peak()
    spectrum = true_baseline + peak

    corrected_linear = ft.baseline_correct(freq, spectrum, method="linear")
    corrected_poly = ft.baseline_correct(freq, spectrum, method="polynomial", degree=2)

    away_from_peak = np.abs(freq - 1.0) > 2.0
    linear_residual = np.abs(corrected_linear[away_from_peak]).mean()
    poly_residual = np.abs(corrected_poly[away_from_peak]).mean()
    assert poly_residual < linear_residual * 0.05
    assert poly_residual < 1e-6  # the true baseline genuinely *is* degree 2 -- near-exact recovery


def test_baseline_correct_polynomial_degree_1_matches_linear():
    freq, true_baseline, peak = _sloped_baseline_with_peak()
    spectrum = true_baseline + peak

    corrected_linear = ft.baseline_correct(freq, spectrum, method="linear")
    corrected_poly1 = ft.baseline_correct(freq, spectrum, method="polynomial", degree=1)

    np.testing.assert_allclose(corrected_linear, corrected_poly1, atol=1e-8)


def _rolling_wave_baseline_with_peak(n=2000):
    freq = np.linspace(-10.0, 10.0, n)
    true_baseline = 2.0 + 3.0 * np.cos(2 * np.pi * freq / 40.0)  # one smooth rise-and-fall
    peak = 5.0 * np.exp(-0.5 * ((freq - 1.0) / 0.3) ** 2)
    return freq, true_baseline, peak


def test_baseline_correct_sinusoidal_handles_a_rolling_wave_baseline():
    freq, true_baseline, peak = _rolling_wave_baseline_with_peak()
    spectrum = true_baseline + peak

    corrected_linear = ft.baseline_correct(freq, spectrum, method="linear")
    corrected_sinusoidal = ft.baseline_correct(freq, spectrum, method="sinusoidal")

    away_from_peak = np.abs(freq - 1.0) > 2.0
    linear_residual = np.abs(corrected_linear[away_from_peak]).mean()
    sinusoidal_residual = np.abs(corrected_sinusoidal[away_from_peak]).mean()
    assert sinusoidal_residual < linear_residual * 0.1
    assert sinusoidal_residual < 1e-6  # the true baseline genuinely *is* one sinusoid
