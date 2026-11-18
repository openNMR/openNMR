"""Model applied to every slice (2026-11-17): relative amplitudes above 1 are
warned about when the lineshapes came from a slice that is not the strongest,
and "Fitted peak amplitude (raw)" gives the height in the spectrum's own
units."""
import numpy as np
import pytest

from opennmr.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
from opennmr.notebook import fit_advice
from opennmr.notebook import series_fitting as sf
from opennmr.processing import relaxation

X = np.linspace(10, -10, 1024)
DELAYS = np.array([0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0])
T1, M0 = 0.8, 1e6


def _lorentz(height, width=0.25):
    return height * width ** 2 / (X ** 2 + width ** 2)


def _series():
    """Saturation-recovery-like rise: the LAST slice is the strongest."""
    rng = np.random.default_rng(3)
    data = np.array([_lorentz(M0 * (1 - np.exp(-d / T1))) + rng.normal(0, 500, X.size)
                     for d in DELAYS])
    meta = NMRMetadata(vendor="bruker", nucleus="19F", sfo1=376.0, pseudo_2d=True)
    return NMRSpectrum(data=data, axes=[NMRAxis(kind="index", index=np.arange(1, 9)),
                                        NMRAxis(kind="frequency", ppm=X, hz=X * 376.0)],
                       metadata=meta)


def _fit(reference_height, reference_row):
    return sf.fit_series_with_lineshapes(
        _series(), X, [_lorentz(reference_height)], ["Site 1"], reference_row=reference_row)


def test_raw_amplitude_is_the_height_in_the_spectrums_units():
    results = _fit(0.5e6, 4)                       # lineshape taken from a weak slice
    raw = [r["raw_amplitudes"][0] for r in results]
    expected = M0 * (1 - np.exp(-DELAYS / T1))
    assert np.allclose(raw, expected, rtol=0.02, atol=3e3)
    # relative amplitude is that over the reference height: above 1 at the end
    assert results[-1]["amplitudes"][0] == pytest.approx(expected[-1] / 0.5e6, rel=0.02)
    assert results[-1]["reference_heights"][0] == pytest.approx(0.5e6, rel=0.01)


def test_raw_amplitude_does_not_depend_on_which_slice_gave_the_lineshape():
    weak, strong = _fit(0.5e6, 4), _fit(1.0e6, 8)
    assert np.allclose([r["raw_amplitudes"][0] for r in weak],
                       [r["raw_amplitudes"][0] for r in strong], rtol=0.02, atol=3e3)
    # whereas the relative ones do
    assert weak[-1]["amplitudes"][0] > 1.5 > strong[-1]["amplitudes"][0]


def test_a_reference_that_is_not_the_strongest_slice_is_warned_about():
    note = sf.reference_slice_note(_fit(0.5e6, 4))
    assert note and "slice 4" in note and "slice 8" in note and "above 1" in note
    assert "raw" in note
    advice = fit_advice.advise_series(_fit(0.5e6, 4))
    assert any("above 1" in item["text"] for item in advice)


def test_no_warning_when_the_reference_is_the_strongest_slice():
    assert sf.reference_slice_note(_fit(1.0e6, 8)) is None
    assert sf.reference_slice_note([]) is None


def test_curve_fit_of_raw_amplitude_recovers_t1_and_the_true_m0():
    results = _fit(0.5e6, 4)
    fit = relaxation.NMR2D_fit_relaxation(
        results, model="saturation_recovery", vdlist=DELAYS, show_report=False,
        value_source="amplitude_raw")[0]
    assert fit["params"]["T1"] == pytest.approx(T1, rel=0.05)
    assert fit["params"]["M0"] == pytest.approx(M0, rel=0.05)         # real units, not ~2
    relative = relaxation.NMR2D_fit_relaxation(
        results, model="saturation_recovery", vdlist=DELAYS, show_report=False,
        value_source="amplitude")[0]
    assert relative["params"]["M0"] == pytest.approx(M0 / 0.5e6, rel=0.05)


def test_results_made_before_raw_amplitudes_existed_say_to_fit_again():
    results = _fit(0.5e6, 4)
    for r in results:
        r.pop("raw_amplitudes")
    with pytest.raises(ValueError, match="again"):
        relaxation.NMR2D_fit_relaxation(results, model="saturation_recovery", vdlist=DELAYS,
                                        show_report=False, value_source="amplitude_raw")


def test_a_peak_fit_raw_amplitude_is_its_own_amplitude():
    from opennmr.processing import fitting, slicing
    rows = slicing.get_slices(_series(), rows=list(range(1, 9)))["rows"]
    peaks = [{"type": "lorentzian", "shift": 0.0, "fwhm": 0.5}]
    fitted = fitting.NMR2D_fit_slices({"rows": rows}, peaks, mode="rows", show_report=False,
                                      show_warnings=False)["rows"]
    a = relaxation.NMR2D_fit_relaxation(fitted, model="saturation_recovery", vdlist=DELAYS,
                                        show_report=False, value_source="amplitude")[0]
    b = relaxation.NMR2D_fit_relaxation(fitted, model="saturation_recovery", vdlist=DELAYS,
                                        show_report=False, value_source="amplitude_raw")[0]
    assert np.allclose(a["y"], b["y"])
