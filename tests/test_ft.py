import numpy as np
import pytest

from nmrsuite.processing import ft


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
