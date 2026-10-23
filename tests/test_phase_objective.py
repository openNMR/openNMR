"""Autophase objective: baseline robustness, strongest-row reference and the
sign-aware, noise-aware per-slice phasing of a pseudo-2D."""
import numpy as np
import pytest

from opennmr.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
from opennmr.notebook import autophase_slices
from opennmr.processing import ft

N = 2048
PPM = np.linspace(10, 0, N)


def _lines(peaks=((5.0, 0.05, 1.0), (3.0, 0.08, 0.6), (7.0, 0.05, 0.3))):
    s = np.zeros(N, complex)
    for c, w, a in peaks:
        x = (PPM - c) / w
        s += a * (1 + 1j * x) / (1 + x ** 2)
    return s


def _wrap(d):
    return (d + 180.0) % 360.0 - 180.0


def test_apc0_on_clean_spectrum():
    for phase in (-150, -40, 0, 33, 120):
        s = _lines() * np.exp(1j * np.deg2rad(phase))
        assert abs(_wrap(ft.apc0(s, PPM, 0.5) - phase)) < 1.0


def test_baseline_hump_does_not_pull_the_phase():
    rng = np.random.default_rng(0)
    for phase in (-120, -30, 50, 140):
        s = _lines() * np.exp(1j * np.deg2rad(phase))
        s = s + 0.1 * np.exp(-((PPM - 5) / 5.0) ** 2) + rng.normal(0, 0.002, N)
        assert abs(_wrap(ft.apc0(s, PPM, 0.5) - phase)) < 3.0
        # Without the baseline step the same hump is off by many degrees.
        assert abs(_wrap(ft.apc0(s, PPM, 0.5, remove_baseline=False) - phase)) > 3.0


def test_clean_spectrum_has_no_phasing_baseline():
    assert not np.any(ft.phasing_baseline(_lines().real + 0j))


def test_reference_row_is_the_strongest_not_the_mean():
    rows = np.array([_lines() * a for a in (1.0, -1.0, 0.2)])
    assert ft.reference_row(rows) in (0, 1)
    assert ft.reference_row(np.zeros(5)) == 0


def test_signless_score_ignores_sign():
    s = _lines()
    assert ft.phase_score_signless(s) == pytest.approx(ft.phase_score_signless(-s))


def _series(amplitudes, phases, noise=0.0, seed=1):
    rng = np.random.default_rng(seed)
    rows = [_lines() * a * np.exp(1j * np.deg2rad(p))
            + noise * (rng.normal(size=N) + 1j * rng.normal(size=N))
            for a, p in zip(amplitudes, phases)]
    data = np.array(rows)
    return NMRSpectrum(
        data=data.real, data_imag=data.imag,
        axes=[NMRAxis(kind="index", index=np.arange(1, len(rows) + 1)),
              NMRAxis(kind="frequency", ppm=PPM, hz=PPM * 400.0)],
        metadata=NMRMetadata(vendor="bruker", nucleus="1H", sfo1=400.0, pseudo_2d=True))


def _check_absorptive(spectrum, amplitudes, tol=0.08):
    for row, a in enumerate(amplitudes):
        y = spectrum.data[row]
        ideal = _lines().real * a
        assert np.max(np.abs(y - ideal)) < tol * max(abs(a), 1e-9) * 1.0 + 1e-6


def test_each_slice_is_phased_on_its_own():
    amps = [1.0, 0.8, 0.6, 0.4]
    phases = [20.0, 70.0, -40.0, 85.0]
    sp = _series(amps, phases)
    found = autophase_slices(sp)
    assert not any(found["inherited"])
    _check_absorptive(sp, amps)


def test_negative_slices_stay_negative():
    amps = [-1.0, -0.5, 0.0001 + 0.3, 0.7, 1.0]
    phases = [30.0] * 5
    sp = _series(amps, phases)
    autophase_slices(sp, window=90.0)
    signs = [np.sign(np.sum(sp.data[k])) for k in range(5)]
    assert signs[0] < 0 and signs[1] < 0 and signs[3] > 0 and signs[4] > 0


def test_noisy_slice_takes_its_neighbours_phase():
    amps = [1.0, 0.9, 0.0, 0.7, 0.6]
    sp = _series(amps, [40.0] * 5, noise=0.02)
    # Row 2 holds nothing but noise: its own phase would be a guess.
    found = autophase_slices(sp, min_snr=10.0)
    assert found["inherited"][2] is True
    assert abs(_wrap(found["phase0"][2] - found["phase0"][1])) < 5.0
    assert found["reference"] == 0


def test_wide_drift_without_a_window_phases_every_row():
    amps = [1.0, 1.0, 1.0, 1.0]
    sp = _series(amps, [0.0, 60.0, 130.0, 170.0])
    autophase_slices(sp)
    _check_absorptive(sp, amps)


def test_recorded_per_slice_phases():
    sp = _series([1.0, 1.0], [10.0, 80.0])
    found = autophase_slices(sp)
    recorded = sp.metadata.raw["slice_phase0"]
    assert len(recorded) == 2
    assert abs(_wrap(recorded[0] - found["phase0"][0])) < 1e-6


def test_a_2d_matrix_can_be_scored_without_a_baseline():
    from opennmr.processing import ft
    matrix = np.ones((3, 150), dtype=complex)
    assert ft.phasing_baseline(matrix).shape == matrix.shape
    assert np.isfinite(ft.apc0(matrix, np.linspace(-10, 10, 150)))
