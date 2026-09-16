import os

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


def test_zero_fill_batches_2d_along_last_axis():
    # Same result as calling per-row, since io.bruker.read_2d switched
    # from an np.apply_along_axis loop to calling this directly on the
    # whole array -- this is the property that fix actually depends on.
    fid = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=complex)
    batched = ft.zero_fill(fid, 5)
    per_row = np.stack([ft.zero_fill(row, 5) for row in fid])
    np.testing.assert_array_equal(batched, per_row)
    assert batched.shape == (2, 5)
    truncated = ft.zero_fill(fid, 2)
    np.testing.assert_array_equal(truncated, fid[:, :2])


def test_apodize_exponential_decays():
    fid = np.ones(100, dtype=complex)
    windowed = ft.apodize(fid, dt=1e-3, window="exponential", lb=50.0)
    assert windowed[0] == 1.0
    assert np.all(np.diff(np.abs(windowed)) <= 0)  # monotonically decaying


def test_apodize_batches_2d_along_last_axis():
    fid = np.array([[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]], dtype=complex)
    batched = ft.apodize(fid, dt=1e-3, window="exponential", lb=50.0)
    per_row = np.stack(
        [ft.apodize(row, dt=1e-3, window="exponential", lb=50.0) for row in fid]
    )
    np.testing.assert_allclose(batched, per_row)


def test_apodize_unknown_window_raises():
    fid = np.ones(10, dtype=complex)
    try:
        ft.apodize(fid, dt=1e-3, window="not-a-window")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


def test_apodize_exponential_matches_nmrglue_em_hz_convention():
    # _window_exponential switched from a hand-rolled np.exp call to
    # nmrglue.process.proc_base.em -- confirm the Hz-based lb
    # convention survived the switch unchanged (algebraically
    # equivalent, but worth a real regression test, not just a
    # derivation in a comment).
    fid = np.ones(200, dtype=complex)
    dt = 1.0 / 5000.0
    windowed = ft.apodize(fid, dt=dt, window="exponential", lb=20.0)
    sw_hz = 1.0 / dt
    t = np.arange(200) * dt
    expected = np.exp(-t * np.pi * 20.0)
    np.testing.assert_allclose(np.abs(windowed), expected, atol=1e-10)


def test_apodize_gaussian_zero_gb_is_plain_exponential():
    # gb=0.0 (default) disables the Gaussian term entirely -- should
    # degenerate to exactly the same shape as the plain exponential
    # window for the same lb.
    fid = np.ones(200, dtype=complex)
    dt = 1.0 / 5000.0
    gaussian_windowed = ft.apodize(fid, dt=dt, window="gaussian", lb=-10.0, gb=0.0)
    exponential_windowed = ft.apodize(fid, dt=dt, window="exponential", lb=-10.0)
    # Gaussian's own lb uses the *same* sign sense as exponential's
    # (see _window_gaussian's own docstring: negative -> sharpen/grow,
    # same as passing exponential a negative lb) -- lb=-10 for both
    # should match exactly once gb=0 removes the Gaussian term.
    np.testing.assert_allclose(np.abs(gaussian_windowed), np.abs(exponential_windowed), atol=1e-8)


def test_apodize_gaussian_negative_lb_grows_then_gaussian_tapers_to_near_zero():
    # Real user report: "Gaussian I think need[s] a negative lb" -- a
    # negative lb should cancel/reverse the FID's own decay (growth
    # early on), with the gb-controlled Gaussian term eventually
    # dominating and tapering the window back toward zero well before
    # the end -- not a monotonic decay (that would just be a
    # differently-signed exponential, not a real Lorentz-to-Gauss
    # shape) and not unbounded growth all the way to the last point
    # (an earlier draft of this window, using a too-small or
    # wrongly-scaled gb, produced exactly that -- see the function's
    # own docstring for the real debugging story).
    n = 4096
    dt = 1.0 / 8000.0
    fid = np.ones(n, dtype=complex)
    windowed = ft.apodize(fid, dt=dt, window="gaussian", lb=-2.0, gb=10.0)
    mag = np.abs(windowed)
    peak_index = int(np.argmax(mag))
    assert 0 < peak_index < n - 1  # peak is genuinely interior, not pinned to either edge
    assert mag[0] < mag[peak_index]  # grows away from the start...
    assert mag[-1] < 0.1 * mag[peak_index]  # ...then tapers back down well before the end


def test_apodize_sine_ssb_two_is_a_pure_cosine_bell():
    fid = np.ones(64, dtype=complex)
    windowed = ft.apodize(fid, dt=1e-3, window="sine", ssb=2.0)
    # off = 1/ssb = 0.5 -- sin(pi*0.5) = 1 at the very first point,
    # i.e. a cosine-shaped bell, not a plain sine starting at 0.
    assert windowed[0] == pytest.approx(1.0, abs=1e-6)


def test_apodize_sine_default_ssb_matches_plain_sine_bell():
    # ssb=0.0 (below Bruker's own ssb>=2 threshold for a shift) should
    # be the same plain 0-starting sine bell _window_sine/_window_qsine
    # used before the ssb parameter existed.
    n = 64
    fid = np.ones(n, dtype=complex)
    sine = ft.apodize(fid, dt=1e-3, window="sine", ssb=0.0)
    qsine = ft.apodize(fid, dt=1e-3, window="qsine", ssb=0.0)
    expected_sine = np.sin(np.pi * np.arange(n) / (n - 1))
    np.testing.assert_allclose(np.abs(sine), expected_sine, atol=1e-8)
    np.testing.assert_allclose(np.abs(qsine), expected_sine ** 2, atol=1e-8)


def test_apodize_qsine_ssb_matches_bruker_apply_window_convention():
    # Same off=1/ssb formula opennmr.io.bruker._apply_bruker_window
    # already uses (and has real-data validation for) on Bruker's own
    # recorded WDW=4/QSINE -- confirms the generic registry's ssb
    # parameter produces the identical window, not a second,
    # independently-drifting formula.
    from opennmr.io import bruker as bru

    n = 128
    dt = 1.0 / 4000.0
    fid = np.ones(n, dtype=complex)
    generic = ft.apodize(fid, dt=dt, window="qsine", ssb=3.0)
    recorded = bru._apply_bruker_window(fid.copy(), {"WDW": 4, "SSB": 3.0}, 1.0 / dt)
    np.testing.assert_allclose(np.abs(generic), np.abs(recorded), atol=1e-8)


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
    # Solved against TopSpin's own recombined F1 interferogram, not
    # guessed -- see decode_hypercomplex_f1's docstring for the fit
    # (residual 0.0172, per-point |corr| 0.997-1.0000, and a scale
    # whose phase is a constant +90 degrees, which is the 1j here).
    echo = np.array([1 + 2j, 2 - 1j, 3 + 0j])
    antiecho = np.array([4 + 1j, 5 + 3j, 6 - 2j])
    decoded = ft.decode_hypercomplex_f1(np.array([echo, antiecho]), fn_mode=6)
    np.testing.assert_allclose(decoded, [1j * (np.conj(echo) - antiecho)])

    # The two components the hypercomplex F1 step needs must each be a
    # single *real* F2 plane -- that is the whole point, and what the
    # superseded whole-complex-row form got wrong (it left F2's
    # dispersion mixed into F1, i.e. the hourglass lineshape).
    np.testing.assert_allclose(np.real(decoded[0]), np.imag(echo) + np.imag(antiecho))
    np.testing.assert_allclose(np.imag(decoded[0]), np.real(echo) - np.real(antiecho))


def test_decode_hypercomplex_f1_states_uses_only_the_f2_real_parts():
    # Same error class as Echo-Antiecho above: combining whole complex
    # rows as a + 1j*b silently carries each row's F2 imaginary part
    # into F1's quadrature, so the F1 transform's real part comes out
    # as RR - II (the phase twist) instead of RR.
    cos_row = np.array([1 + 7j, 2 - 5j, 3 + 4j])
    sin_row = np.array([4 + 1j, 5 + 3j, 6 - 2j])
    data = np.array([cos_row, sin_row])
    decoded = ft.decode_hypercomplex_f1(data, fn_mode=4)
    np.testing.assert_allclose(decoded, [np.real(cos_row) + 1j * np.real(sin_row)])

    # Discriminates against the superseded form, which these rows' own
    # imaginary parts would otherwise dominate.
    assert np.max(np.abs(decoded[0] - (cos_row + 1j * sin_row))) > 1.0

    # States-TPPI is the same combination plus (-1)**k.
    four_rows = np.array([cos_row, sin_row, cos_row, sin_row])
    tppi = ft.decode_hypercomplex_f1(four_rows, fn_mode=5)
    np.testing.assert_allclose(tppi[0], np.real(cos_row) + 1j * np.real(sin_row))
    np.testing.assert_allclose(tppi[1], -(np.real(cos_row) + 1j * np.real(sin_row)))


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


def test_fft_spectrum_is_bit_identical_to_numpy_at_every_worker_count(monkeypatch):
    # fft_spectrum switched from numpy.fft to scipy.fft purely for its
    # `workers` parameter. Threading splits the batch across
    # independent 1D transforms, so no individual transform's own
    # summation order changes and the result must be *bit*-identical --
    # not merely close. Asserted with array_equal rather than allclose
    # deliberately: a tolerance here would hide exactly the kind of
    # thread-count-dependent numerical drift this check exists to rule
    # out.
    rng = np.random.default_rng(0)
    fid = rng.standard_normal((6, 512)) + 1j * rng.standard_normal((6, 512))
    reference = np.fft.fftshift(np.fft.fft(fid, axis=-1), axes=-1)

    for workers in (1, 2, 4, 8):
        monkeypatch.setenv("OPENNMR_FFT_WORKERS", str(workers))
        _, spectrum = ft.fft_spectrum(fid, dt=1e-4)
        assert np.array_equal(spectrum, reference), f"differs at workers={workers}"


def test_fft_workers_env_override_pins_the_count(monkeypatch):
    monkeypatch.setenv("OPENNMR_FFT_WORKERS", "1")
    assert ft.fft_workers() == 1  # the supported "threads off" configuration
    monkeypatch.setenv("OPENNMR_FFT_WORKERS", "3")
    assert ft.fft_workers() == 3


def test_fft_workers_ignores_a_bad_override_instead_of_raising(monkeypatch):
    # A malformed or non-positive environment value must not stop a
    # spectrum from loading -- it falls back to the automatic choice.
    for bad in ("garbage", "0", "-2", ""):
        monkeypatch.setenv("OPENNMR_FFT_WORKERS", bad)
        assert ft.fft_workers() >= 1


def test_fft_workers_is_capped_and_never_exceeds_available_cpus(monkeypatch):
    # Never workers=-1: scipy/numpy's own BLAS/OpenMP pools are already
    # live in this process, and the measured F2 plateau is 4, so
    # uncapped threading costs more than it buys on a many-core box.
    monkeypatch.delenv("OPENNMR_FFT_WORKERS", raising=False)
    monkeypatch.setattr(ft, "_available_cpus", lambda: 128)
    assert ft.fft_workers() == ft._FFT_WORKER_CAP

    # ...and on a constrained machine it follows the allowance down,
    # rather than pinning the cap regardless.
    monkeypatch.setattr(ft, "_available_cpus", lambda: 1)
    assert ft.fft_workers() == 1


def test_available_cpus_prefers_the_process_allowance_over_machine_cores(monkeypatch):
    # os.cpu_count() reports the machine, not this process's own
    # allowance -- wrong under cgroup limits and CPU affinity masks.
    # os.process_cpu_count() is the correct API but is 3.13+, so the
    # fallback chain has to work on 3.10-3.12 too; both paths are
    # pinned here rather than only whichever this interpreter happens
    # to take.
    monkeypatch.setattr(os, "cpu_count", lambda: 64)
    monkeypatch.setattr(os, "process_cpu_count", lambda: 2, raising=False)
    assert ft._available_cpus() == 2

    # Without process_cpu_count (i.e. Python < 3.13), sched_getaffinity
    # is the next-best source where it exists (Linux); it is absent on
    # macOS/Windows, which is why it is probed rather than assumed.
    monkeypatch.delattr(os, "process_cpu_count", raising=False)
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: {0, 1, 2}, raising=False)
    assert ft._available_cpus() == 3

    # Neither available -> os.cpu_count() is the last resort.
    monkeypatch.delattr(os, "sched_getaffinity", raising=False)
    assert ft._available_cpus() == 64


def test_decode_hypercomplex_f1_is_vendor_agnostic():
    # This module is deliberately vendor-free: the F1 quadrature sense
    # differs between Bruker and X-Pulse, and that difference is
    # normalised by the *reader* (io.bruker negates the sine-modulated
    # rows), not by a flag here. A `conjugate_f1` parameter existed for
    # one release and was removed for exactly that reason.
    import inspect

    params = inspect.signature(ft.decode_hypercomplex_f1).parameters
    assert set(params) == {"data", "fn_mode"}

    rows = np.array([[1 + 0j, 2 + 0j], [3 + 0j, 4 + 0j]])
    np.testing.assert_allclose(
        ft.decode_hypercomplex_f1(rows, fn_mode=4), rows[0::2] + 1j * rows[1::2]
    )


def test_negating_sine_rows_matches_the_opposite_quadrature_sense():
    # The identity the reader-side normalisation relies on:
    # a - 1j*b == a + 1j*(-b). Pinned so the two can never drift.
    rows = np.arange(8, dtype=complex).reshape(4, 2)
    a, b = rows[0::2], rows[1::2]

    normalised = rows.copy()
    normalised[1::2] *= -1
    got = ft.decode_hypercomplex_f1(normalised, fn_mode=5)

    expected = (a - 1j * b) * np.array([1.0, -1.0])[:, np.newaxis]
    np.testing.assert_allclose(got, expected)


def test_linear_predict_stabilises_a_divergent_ar_fit():
    # Real failure on real data: an unconstrained least-squares AR fit
    # can put poles outside the unit circle, so the "prediction" grows
    # exponentially instead of extrapolating a decaying signal. Three of
    # five Bruker datasets whose proc2s asks for F1 linear prediction
    # came back with peak magnitudes of 1e85 to 1e104 against an
    # expected ~1e11, which drove their correlation with TopSpin to
    # exactly 0.0000.
    #
    # A short noisy record with a growing component is the reliable way
    # to provoke it.
    rng = np.random.default_rng(3)
    n = 64
    t = np.arange(n)
    fid = np.exp(0.02 * t) * np.exp(2j * np.pi * 0.05 * t)
    fid = fid + 0.05 * (rng.standard_normal(n) + 1j * rng.standard_normal(n))

    unstable = ft.linear_predict(fid, order=24, n_predict=192, stabilise=False)
    stabilised = ft.linear_predict(fid, order=24, n_predict=192, stabilise=True)

    input_max = np.max(np.abs(fid))
    # The stabilised prediction must not blow up past the input's own
    # scale by orders of magnitude...
    assert np.max(np.abs(stabilised)) < 100 * input_max
    # ...and must be dramatically better behaved than the raw fit.
    assert np.max(np.abs(stabilised)) < np.max(np.abs(unstable))
    assert np.isfinite(stabilised).all()
    # The measured points themselves are never altered.
    np.testing.assert_allclose(stabilised[:n], fid)


def test_reflect_unstable_roots_leaves_a_stable_model_alone():
    # A model whose poles are already inside the unit circle must come
    # back untouched -- the guard should cost nothing but a root-find
    # in the common case.
    decaying = np.exp(-0.1 * np.arange(48)) * np.exp(2j * np.pi * 0.1 * np.arange(48))
    with_guard = ft.linear_predict(decaying, order=8, n_predict=16, stabilise=True)
    without = ft.linear_predict(decaying, order=8, n_predict=16, stabilise=False)
    np.testing.assert_allclose(with_guard, without, rtol=1e-6, atol=1e-9)


# ============================================================
# Echo position offset
# ============================================================

def _echo(n=256, top=90, width=12.0):
    """A symmetric full echo peaking at ``top``."""
    t = np.arange(n)
    return (np.exp(-((t - top) / width) ** 2)).astype(complex)


def test_find_echo_top_offset_shifts_the_reported_top():
    fid = _echo(top=90)
    assert ft.find_echo_top(fid) == pytest.approx(90.0, abs=0.05)
    assert ft.find_echo_top(fid, offset=4.0) == pytest.approx(94.0, abs=0.05)
    assert ft.find_echo_top(fid, offset=-4.0) == pytest.approx(86.0, abs=0.05)


def test_find_echo_top_offset_accepts_fractions():
    fid = _echo(top=90)
    assert ft.find_echo_top(fid, offset=0.5) == pytest.approx(90.5, abs=0.05)


def test_find_echo_top_offset_is_clamped_to_the_array():
    # A large offset must not roll the top outside the data it indexes.
    fid = _echo(n=128, top=60)
    assert ft.find_echo_top(fid, offset=1e6) == pytest.approx(127.0)
    assert ft.find_echo_top(fid, offset=-1e6) == pytest.approx(0.0)


def test_align_echo_offset_moves_where_the_peak_lands():
    fid = _echo(top=90)
    aligned = ft.align_echo(fid)
    # Without an offset the peak lands at index 0.
    assert int(np.argmax(np.abs(aligned))) == 0
    # An offset of +4 rolls 4 points further, so the true peak ends up
    # 4 samples from the end (wrapped), i.e. deliberately mis-set --
    # which is the point: the control exists to correct a detector that
    # landed in the wrong place.
    offset_aligned = ft.align_echo(fid, echo_offset=4.0)
    assert int(np.argmax(np.abs(offset_aligned))) == len(fid) - 4


def test_prepare_fid_uses_the_same_top_for_the_roll_and_the_wrap_cap():
    # The wrap limit that caps truncate has to come from the offset
    # top, or the cap no longer matches where the pre-echo portion
    # actually wrapped to.
    fid = _echo(n=256, top=90)
    plain = ft.prepare_fid(fid, align_echo_enabled=True)
    nudged = ft.prepare_fid(fid, align_echo_enabled=True, echo_offset=10.0)
    assert len(plain) == 256 - 90
    assert len(nudged) == 256 - 100


def test_prepare_fid_truncate_still_applies_alongside_an_offset():
    fid = _echo(n=256, top=90)
    result = ft.prepare_fid(fid, align_echo_enabled=True, echo_offset=10.0, truncate=50)
    assert len(result) == 50


def test_prepare_fid_echo_offset_defaults_to_no_change():
    fid = _echo()
    np.testing.assert_allclose(ft.prepare_fid(fid, align_echo_enabled=True),
                               ft.prepare_fid(fid, align_echo_enabled=True, echo_offset=0.0))


# ============================================================
# Whittaker (asymmetrically reweighted) baseline
# ============================================================

def _rolling_baseline_spectrum(n=8192, solvent=100.0, seed=0):
    """A spectrum with a rolling baseline, a dominant solvent peak and
    three weak analyte peaks -- the case the edge-based estimators
    cannot see, because the roll is in the middle and the edges are
    not representative of it."""
    rng = np.random.default_rng(seed)
    ppm = np.linspace(-2.0, 12.0, n)

    def lor(centre, width, amp):
        return amp / (1.0 + ((ppm - centre) / width) ** 2)

    baseline = 0.35 * np.sin(2 * np.pi * (ppm + 2.0) / 22.0) + 0.05 * ppm
    peaks = (lor(4.7, 0.02, solvent) + lor(1.2, 0.01, 1.0)
             + lor(7.3, 0.01, 0.8) + lor(2.6, 0.015, 0.5))
    noise = rng.normal(0.0, 0.004, n)
    return ppm, peaks, baseline, peaks + baseline + noise


def test_whittaker_smooth_matches_a_dense_solve():
    """The banded solve is an optimisation, not a different answer."""
    from opennmr.processing.fitting import whittaker_smooth

    n = 150
    rng = np.random.default_rng(1)
    y = rng.normal(size=n)
    w = rng.uniform(0.1, 1.0, size=n)
    lam = 12.5

    z = whittaker_smooth(y, w, lam, d=2)
    D = np.diff(np.eye(n), 2, axis=0)
    expected = np.linalg.solve(np.diag(w) + lam * D.T @ D, w * y)
    np.testing.assert_allclose(z, expected, atol=1e-8)


def test_whittaker_recovers_a_rolling_baseline_under_a_dominant_peak():
    ppm, _peaks, true_baseline, spectrum = _rolling_baseline_spectrum()
    from opennmr.processing.fitting import estimate_baseline_whittaker

    fitted = estimate_baseline_whittaker(ppm, spectrum)
    error = np.sqrt(np.mean((fitted - true_baseline) ** 2))
    # The baseline itself spans ~0.9; recovering it to a few percent of
    # that is the point of the method.
    assert error < 0.05 * np.ptp(true_baseline)


def test_whittaker_beats_the_edge_based_estimators_on_that_spectrum():
    """The reason to add it: every other method assumes the first and
    last few percent of the spectrum are representative baseline."""
    ppm, _peaks, true_baseline, spectrum = _rolling_baseline_spectrum()

    def rms(method, **kwargs):
        corrected = ft.baseline_correct(ppm, spectrum, method=method, **kwargs)
        return float(np.sqrt(np.mean(((spectrum - corrected) - true_baseline) ** 2)))

    whittaker = rms("whittaker")
    assert whittaker < rms("linear")
    assert whittaker < rms("polynomial", degree=3)


def test_whittaker_does_not_eat_the_peaks_it_runs_under():
    """A baseline that follows the peaks is not a baseline."""
    ppm, _peaks, _true, spectrum = _rolling_baseline_spectrum()
    corrected = np.real(ft.baseline_correct(ppm, spectrum, method="whittaker"))

    for centre, expected in ((4.7, 100.0), (1.2, 1.0), (7.3, 0.8), (2.6, 0.5)):
        index = int(np.argmin(np.abs(ppm - centre)))
        window = corrected[max(0, index - 30):index + 30]
        # Every peak survives with most of its height -- including the
        # weak ones, which a baseline dragged up by the solvent peak
        # would have flattened.
        assert window.max() > 0.8 * expected


def test_whittaker_lam_controls_stiffness():
    ppm, _peaks, _true, spectrum = _rolling_baseline_spectrum()
    from opennmr.processing.fitting import estimate_baseline_whittaker

    stiff = estimate_baseline_whittaker(ppm, spectrum, lam=1e14)
    loose = estimate_baseline_whittaker(ppm, spectrum, lam=1e6)
    # Stiffer means straighter: less curvature left in the result.
    assert np.sum(np.abs(np.diff(stiff, 2))) < np.sum(np.abs(np.diff(loose, 2)))


def test_whittaker_asls_is_available_and_also_finds_the_baseline():
    ppm, _peaks, true_baseline, spectrum = _rolling_baseline_spectrum()
    from opennmr.processing.fitting import estimate_baseline_whittaker

    fitted = estimate_baseline_whittaker(ppm, spectrum, method="asls", p=0.001)
    assert np.sqrt(np.mean((fitted - true_baseline) ** 2)) < 0.15 * np.ptp(true_baseline)

    with pytest.raises(ValueError, match="arpls"):
        estimate_baseline_whittaker(ppm, spectrum, method="nonsense")


def test_whittaker_ignores_baseline_regions_rather_than_misusing_them():
    """It finds the baseline itself; a candidate region is not part of
    how it works, and quietly restricting the fit to one would be a
    different method wearing this one's name."""
    ppm, _peaks, _true, spectrum = _rolling_baseline_spectrum()
    with_regions = ft.baseline_correct(
        ppm, spectrum, method="whittaker", baseline_regions=[(0.0, 0.5)])
    without = ft.baseline_correct(ppm, spectrum, method="whittaker")
    np.testing.assert_allclose(np.real(with_regions), np.real(without))


def test_whittaker_handles_an_all_zero_spectrum():
    ppm = np.linspace(0.0, 10.0, 256)
    from opennmr.processing.fitting import estimate_baseline_whittaker
    np.testing.assert_allclose(estimate_baseline_whittaker(ppm, np.zeros(256)), 0.0)


def test_baseline_correct_rejects_an_unknown_method_by_name():
    ppm = np.linspace(0.0, 10.0, 256)
    with pytest.raises(ValueError, match="whittaker"):
        ft.baseline_correct(ppm, np.zeros(256), method="nope")


def test_compute_baseline_and_baseline_correct_agree():
    """The overlay draws compute_baseline's output and the correction
    subtracts baseline_correct's -- if they ever disagreed, the preview
    would be a lie in exactly the case it exists to catch."""
    ppm, _peaks, _true, spectrum = _rolling_baseline_spectrum(n=2048)
    for method, kwargs in (("linear", {}), ("polynomial", {"degree": 3}),
                            ("sinusoidal", {"n_harmonics": 2}), ("spline", {}),
                            ("whittaker", {})):
        baseline = ft.compute_baseline(ppm, spectrum, method=method, **kwargs)
        corrected = ft.baseline_correct(ppm, spectrum, method=method, **kwargs)
        np.testing.assert_allclose(np.real(spectrum - baseline), np.real(corrected))


def test_compute_baseline_rejects_an_unknown_method():
    ppm = np.linspace(0.0, 10.0, 256)
    with pytest.raises(ValueError, match="whittaker"):
        ft.compute_baseline(ppm, np.zeros(256), method="nope")


def test_sinusoidal_harmonics_and_period_change_the_fit():
    """Both were accepted by baseline_correct all along and neither had
    any way to be set."""
    ppm, _peaks, _true, spectrum = _rolling_baseline_spectrum(n=2048)
    one = ft.compute_baseline(ppm, spectrum, method="sinusoidal", n_harmonics=1)
    three = ft.compute_baseline(ppm, spectrum, method="sinusoidal", n_harmonics=3)
    assert not np.allclose(one, three)

    auto = ft.compute_baseline(ppm, spectrum, method="sinusoidal")
    pinned = ft.compute_baseline(ppm, spectrum, method="sinusoidal", period=4.0)
    assert not np.allclose(auto, pinned)


def test_whittaker_penalty_bands_are_cached_but_not_shared_mutably():
    """The cache is what makes a pseudo-2D affordable; a caller adding
    its weights to the shared array would corrupt every later row."""
    from opennmr.processing.fitting import _whittaker_penalty_bands, whittaker_smooth

    first = _whittaker_penalty_bands(64, 10.0, 2)
    before = first.copy()
    y = np.linspace(0.0, 1.0, 64)
    whittaker_smooth(y, np.ones(64), 10.0, d=2)
    np.testing.assert_array_equal(_whittaker_penalty_bands(64, 10.0, 2), before)
