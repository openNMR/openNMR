"""Reference deconvolution -- opennmr.processing.refdeconv.

The tests build a spectrum the honest way -- construct an FID, distort
it, transform -- so "the truth" is a real undistorted spectrum rather
than an assumption about what the answer should look like.
"""

import numpy as np
import pytest

from opennmr.processing import ft, refdeconv

N, DT, HZ_PER_PPM = 8192, 1e-4, 100.0
T2 = 0.02  # natural FWHM = 1/(pi*T2) = 15.9 Hz = 0.159 ppm


def _time():
    return np.arange(N) * DT


def _fid_at(shift_ppm, amplitude, t=None):
    """One resonance's FID. Negated frequency because
    ``ft.fft_spectrum``'s axis puts a ``+f`` FID at ``-f``."""
    t = _time() if t is None else t
    return amplitude * np.exp(-2j * np.pi * shift_ppm * HZ_PER_PPM * t) * np.exp(-t / T2)


def _three_peak_fid():
    """A reference at 0 ppm plus two other resonances."""
    return _fid_at(0.0, 1.0) + _fid_at(4.0, 0.7) + _fid_at(6.0, 0.5)


def _spectrum(fid):
    freq_hz, spec = ft.fft_spectrum(fid, DT)
    return freq_hz / HZ_PER_PPM, spec


def _fwhm(spec, ppm, centre, halfwidth=0.9):
    """Absorption-mode FWHM in ppm. The *real* part deliberately: the
    magnitude of a Lorentzian is sqrt(3) wider, which silently makes
    every width look 1.73x too big."""
    mask = (ppm > centre - halfwidth) & (ppm < centre + halfwidth)
    y, x = np.real(spec[mask]), ppm[mask]
    y = y / y.max()
    above = x[y >= 0.5]
    return float(above.max() - above.min())


def _distorted(distortion, noise=3e-4, seed=0):
    """``(ppm, clean_spectrum, distorted_spectrum)``."""
    clean = _three_peak_fid()
    rng = np.random.default_rng(seed)
    measured = clean * distortion + (rng.normal(0, noise, N) + 1j * rng.normal(0, noise, N))
    ppm, spec_clean = _spectrum(clean)
    _ppm, spec_meas = _spectrum(measured)
    return ppm, spec_clean, spec_meas


#: A field wobble that decays away -- a lineshape distortion that does
#: *not* destroy the signal, which is what a shim error or an eNMR
#: cell's electrodes actually do.
def _field_wobble():
    t = _time()
    return np.exp(1j * 2 * np.pi * 25.0 * t * np.exp(-t / 0.03))


# ============================================================
# The ideal reference FID
# ============================================================

def test_ideal_reference_lands_at_the_requested_shift_and_width():
    ppm = np.linspace(10.0, -10.0, 16384)
    for shift, width in ((0.0, 0.2), (3.0, 0.5), (-4.0, 0.3)):
        spec = ft.fft_spectrum(refdeconv.ideal_reference(ppm, shift, width), 1.0)[1]
        assert ppm[np.argmax(np.abs(spec))] == pytest.approx(shift, abs=0.01)
        assert _fwhm(spec, ppm, shift, halfwidth=2.0) == pytest.approx(width, rel=0.05)


def test_ideal_reference_is_causal():
    # Inverse-transforming a *real* Lorentzian profile gives a signal
    # symmetric about t=0, half of which wraps to the end of the array
    # and corrupts exactly the late points the regularisation protects.
    # An FID starts at t=0 and decays.
    ppm = np.linspace(10.0, -10.0, 2048)
    fid = refdeconv.ideal_reference(ppm, 0.0, 0.3)
    assert abs(fid[0]) == pytest.approx(np.abs(fid).max())
    assert abs(fid[-1]) < abs(fid[0]) * 1e-3


def test_ideal_reference_gaussian_decays_faster_than_lorentzian_late_on():
    ppm = np.linspace(10.0, -10.0, 4096)
    lorentz = np.abs(refdeconv.ideal_reference(ppm, 0.0, 0.3, lineshape="lorentzian"))
    gauss = np.abs(refdeconv.ideal_reference(ppm, 0.0, 0.3, lineshape="gaussian"))
    assert gauss[500] < lorentz[500]


def test_ideal_reference_rejects_a_bad_width_or_lineshape():
    ppm = np.linspace(10.0, -10.0, 512)
    with pytest.raises(ValueError, match="must be positive"):
        refdeconv.ideal_reference(ppm, 0.0, 0.0)
    with pytest.raises(ValueError, match="Unknown lineshape"):
        refdeconv.ideal_reference(ppm, 0.0, 0.3, lineshape="voigt")


# ============================================================
# Isolating the measured reference
# ============================================================

def test_isolate_reference_keeps_only_the_chosen_region():
    ppm, _clean, spec = _distorted(_field_wobble())
    isolated = refdeconv.isolate_reference(spec, ppm, (-1.0, 1.0))
    # Transforming it back must show the reference and nothing else.
    back = ft.fft_spectrum(isolated, DT)[1]
    assert np.abs(back[(ppm > -1.0) & (ppm < 1.0)]).max() > 0
    for other in (4.0, 6.0):
        near = (ppm > other - 0.3) & (ppm < other + 0.3)
        assert np.abs(back[near]).max() < np.abs(back).max() * 0.01


def test_isolate_reference_rejects_an_empty_region():
    ppm, _clean, spec = _distorted(_field_wobble())
    with pytest.raises(ValueError, match="No spectral points"):
        refdeconv.isolate_reference(spec, ppm, (100.0, 200.0))


def test_isolate_reference_rejects_a_length_mismatch():
    with pytest.raises(ValueError, match="spectral points against"):
        refdeconv.isolate_reference(np.zeros(10), np.zeros(5), (0.0, 1.0))


# ============================================================
# The correction, and its regularisation
# ============================================================

def test_correction_of_an_undistorted_reference_is_flat():
    ppm = np.linspace(10.0, -10.0, 2048)
    fid = refdeconv.ideal_reference(ppm, 0.0, 0.3)
    factor = refdeconv.correction(fid, fid, method="none")
    # ideal / ideal = 1 wherever there is any signal at all.
    early = factor[:200]
    np.testing.assert_allclose(np.real(early), 1.0, atol=1e-6)


def test_correction_regularisation_tames_the_late_points():
    # The whole difficulty: past the reference's own decay the
    # denominator is noise, and an undamped reciprocal grows without
    # bound.
    ppm = np.linspace(10.0, -10.0, 4096)
    ideal = refdeconv.ideal_reference(ppm, 0.0, 0.2)
    rng = np.random.default_rng(0)
    measured = refdeconv.ideal_reference(ppm, 0.0, 0.8) + rng.normal(0, 1e-6, 4096)

    bare = refdeconv.correction(measured, ideal, method="none")
    wiener = refdeconv.correction(measured, ideal, method="wiener", eps=1e-3)
    truncated = refdeconv.correction(measured, ideal, method="truncate", eps=1e-3)

    assert np.abs(bare).max() > 1e3 * np.abs(wiener).max()
    assert np.abs(wiener[-100:]).max() < 1.0
    # truncate hands back exactly 1 (no correction) past its threshold
    np.testing.assert_allclose(np.abs(truncated[-10:]), 1.0)


def test_correction_eps_trades_correction_against_noise():
    ppm = np.linspace(10.0, -10.0, 4096)
    ideal = refdeconv.ideal_reference(ppm, 0.0, 0.2)
    measured = refdeconv.ideal_reference(ppm, 0.0, 0.8)
    loose = refdeconv.correction(measured, ideal, method="wiener", eps=1e-1)
    tight = refdeconv.correction(measured, ideal, method="wiener", eps=1e-5)
    # A larger eps backs off sooner, so its correction is smaller.
    assert np.abs(loose).max() < np.abs(tight).max()


def test_correction_rejects_bad_input():
    fid = np.ones(8, dtype=complex)
    with pytest.raises(ValueError, match="Unknown regularisation"):
        refdeconv.correction(fid, fid, method="magic")
    with pytest.raises(ValueError, match="differ in shape"):
        refdeconv.correction(fid, np.ones(4, dtype=complex))
    with pytest.raises(ValueError, match="entirely zero"):
        refdeconv.correction(np.zeros(8, dtype=complex), fid)


# ============================================================
# End to end, against a known truth
# ============================================================

def test_reference_deconvolution_recovers_the_undistorted_linewidth():
    ppm, spec_clean, spec_meas = _distorted(_field_wobble())
    truth = _fwhm(spec_clean, ppm, 0.0)
    assert _fwhm(spec_meas, ppm, 0.0) > truth * 1.5   # genuinely distorted

    result = refdeconv.reference_deconvolve(
        spec_meas, ppm, (-1.9, 1.9), fwhm=truth, method="wiener", eps=1e-3)
    assert _fwhm(result["spectrum"], ppm, 0.0) == pytest.approx(truth, rel=0.15)


def test_reference_deconvolution_corrects_peaks_outside_the_reference_region():
    # The method's entire claim: the distortion is common-mode, so a
    # correction derived from one resonance fixes all of them.
    ppm, spec_clean, spec_meas = _distorted(_field_wobble())
    truth = _fwhm(spec_clean, ppm, 4.0)
    assert _fwhm(spec_meas, ppm, 4.0) > truth * 1.5

    result = refdeconv.reference_deconvolve(
        spec_meas, ppm, (-1.9, 1.9), fwhm=_fwhm(spec_clean, ppm, 0.0),
        method="wiener", eps=1e-3)
    for centre in (4.0, 6.0):
        assert _fwhm(result["spectrum"], ppm, centre) == pytest.approx(truth, rel=0.15)


def test_reference_deconvolution_recovery_tracks_surviving_signal():
    # The limit is information, not implementation: a distortion that
    # destroys the late FID cannot be undone, and the result degrades
    # gracefully rather than pretending otherwise.
    t = _time()
    ppm, spec_clean, _m = _distorted(_field_wobble())
    truth = _fwhm(spec_clean, ppm, 0.0)

    widths = []
    for tau in (0.06, 0.03, 0.012):
        distortion = np.exp(-((t / tau) ** 2)) * _field_wobble()
        ppm, _c, spec_meas = _distorted(distortion)
        result = refdeconv.reference_deconvolve(
            spec_meas, ppm, (-1.9, 1.9), fwhm=truth, method="wiener", eps=1e-3)
        widths.append(_fwhm(result["spectrum"], ppm, 0.0))

    # Harsher distortion (shorter tau) -> less complete recovery.
    assert widths[0] < widths[1] < widths[2]
    assert widths[0] == pytest.approx(truth, rel=0.35)


def test_reference_deconvolution_does_not_move_the_spectrum():
    # The default target position is the measured centroid, so a
    # correction changes lineshape and not chemical shift.
    ppm, _clean, spec_meas = _distorted(_field_wobble())
    result = refdeconv.reference_deconvolve(spec_meas, ppm, (-1.9, 1.9), fwhm=0.15)
    before = ppm[np.argmax(np.abs(spec_meas))]
    after = ppm[np.argmax(np.abs(result["spectrum"]))]
    assert after == pytest.approx(before, abs=0.1)


def test_reference_deconvolution_preserves_relative_intensities():
    # It should change lineshape, not concentration: the 0.7 and 0.5
    # resonances must stay in that ratio to the reference.
    ppm, _clean, spec_meas = _distorted(_field_wobble())
    result = refdeconv.reference_deconvolve(spec_meas, ppm, (-1.9, 1.9), fwhm=0.15)

    def area(spec, centre, halfwidth=1.5):
        mask = (ppm > centre - halfwidth) & (ppm < centre + halfwidth)
        return float(np.abs(np.trapezoid(np.real(spec[mask]), ppm[mask])))

    before = area(spec_meas, 4.0) / area(spec_meas, 0.0)
    after = area(result["spectrum"], 4.0) / area(result["spectrum"], 0.0)
    assert after == pytest.approx(before, rel=0.2)


def test_reference_deconvolution_reports_what_it_did():
    ppm, _clean, spec_meas = _distorted(_field_wobble())
    result = refdeconv.reference_deconvolve(
        spec_meas, ppm, (-1.0, 1.0), fwhm=0.15, method="truncate", eps=1e-2)
    assert result["method"] == "truncate"
    assert result["eps"] == pytest.approx(1e-2)
    assert result["fwhm"] == pytest.approx(0.15)
    assert result["region"] == (-1.0, 1.0)
    assert result["shift"] == pytest.approx(0.0, abs=0.2)
    for key in ("spectrum", "correction", "fid_measured", "fid_ideal"):
        assert len(result[key]) == ppm.size


def test_reference_deconvolution_honours_an_explicit_shift():
    ppm, _clean, spec_meas = _distorted(_field_wobble())
    result = refdeconv.reference_deconvolve(
        spec_meas, ppm, (-1.0, 1.0), fwhm=0.15, shift=0.4)
    assert result["shift"] == pytest.approx(0.4)


def test_reference_position_finds_the_measured_line():
    ppm, _clean, spec_meas = _distorted(_field_wobble())
    assert refdeconv.reference_position(spec_meas, ppm, (-1.9, 1.9)) == pytest.approx(0.0, abs=0.1)


def test_reference_position_beats_a_centroid_on_an_asymmetric_line():
    # Why this is a maximum and not a centroid: these distortions drag
    # a tail to one side and a centroid follows it.
    ppm, _clean, spec_meas = _distorted(_field_wobble())
    mask = (ppm > -1.9) & (ppm < 1.9)
    weights = np.abs(spec_meas[mask])
    centroid = float(np.sum(ppm[mask] * weights) / weights.sum())

    maximum = refdeconv.reference_position(spec_meas, ppm, (-1.9, 1.9))
    assert abs(maximum) < abs(centroid) / 5.0


def test_reference_position_of_an_empty_region_is_its_middle():
    ppm = np.linspace(10.0, -10.0, 256)
    assert refdeconv.reference_position(np.zeros(256), ppm, (2.0, 4.0)) == pytest.approx(3.0)
