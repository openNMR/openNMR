"""The seam between NMRSpectrum and the notebook API's results dict."""

import numpy as np
import pytest

from opennmr.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
from opennmr.notebook.bridge import RESULTS_KEYS, results_from_spectrum


def _spectrum(n=64, ndim=1, kind="frequency"):
    ppm = np.linspace(100.0, -100.0, n)
    axis = (NMRAxis(kind="frequency", ppm=ppm, hz=ppm * 50.0) if kind == "frequency"
            else NMRAxis(kind="index", index=np.arange(1, n + 1)))
    data = np.exp(-((ppm - 10.0) ** 2) / 50.0)
    metadata = NMRMetadata(vendor="test", nucleus="13C", sfo1=50.0, source_path="/tmp/x")
    if ndim == 2:
        return NMRSpectrum(data=np.vstack([data, 2 * data]),
                           axes=[NMRAxis(kind="index", index=np.arange(1, 3)), axis],
                           metadata=metadata, data_imag=np.vstack([data, data]))
    return NMRSpectrum(data=data, axes=[axis], metadata=metadata, data_imag=0.5 * data)


def test_the_dict_has_exactly_the_keys_the_notebook_api_expects():
    """Every notebook entry point takes what ``NMR1dimport`` returns.
    Naming the contract here means a mismatch fails as a test rather
    than as a KeyError three frames into a fit."""
    results = results_from_spectrum(_spectrum())
    assert tuple(results) == RESULTS_KEYS


def test_values_carry_through_and_normalisation_peaks_at_one():
    spectrum = _spectrum()
    results = results_from_spectrum(spectrum)
    np.testing.assert_allclose(results["real_spectrum"], spectrum.data)
    np.testing.assert_allclose(results["x_ppm"], spectrum.axes[0].ppm)
    assert np.max(np.abs(results["normalised_real_spectrum"])) == pytest.approx(1.0)
    assert results["nuclide"] == "13C"
    assert results["SFO1"] == pytest.approx(50.0)


def test_the_ppm_axis_is_not_re_sorted():
    """Both this application and ``NMR1dimport`` hold ppm descending,
    and the notebook's fitters sort ascending themselves where they
    need to. Sorting here as well would be a second, silent convention
    for the same data."""
    spectrum = _spectrum()
    results = results_from_spectrum(spectrum)
    assert results["x_ppm"][0] > results["x_ppm"][-1]


def test_a_2d_needs_a_row_chosen_explicitly():
    """There is no single "the" spectrum in a 2D dataset, and picking
    one silently is how a fit ends up describing a slice nobody
    chose."""
    spectrum = _spectrum(ndim=2)
    with pytest.raises(ValueError, match="needs a row"):
        results_from_spectrum(spectrum)
    with pytest.raises(IndexError):
        results_from_spectrum(spectrum, row=5)
    np.testing.assert_allclose(results_from_spectrum(spectrum, row=1)["real_spectrum"],
                               spectrum.data[1])


def test_an_index_axis_is_refused_with_a_useful_message():
    """A pseudo-2D's F1 is a slice number, not a shift -- a fitter
    handed one would fit a lineshape against an index."""
    spectrum = _spectrum(kind="index")
    with pytest.raises(ValueError, match="frequency axis"):
        results_from_spectrum(spectrum)


def test_a_missing_imaginary_part_still_gives_the_full_dict():
    """The dict's shape is fixed; consumers that touch the imaginary
    part check it for content themselves."""
    spectrum = _spectrum()
    spectrum.data_imag = None
    results = results_from_spectrum(spectrum)
    assert tuple(results) == RESULTS_KEYS
    assert np.allclose(results["imag_spectrum"], 0.0)


def test_it_drives_the_csa_fitter_on_a_known_lineshape():
    """The point of the bridge: a spectrum already open in the
    application can be handed to the notebook's fitters without being
    re-read from disk in a different format."""
    from opennmr.notebook.csa_fitting import fit_csa_spectrum
    from opennmr.notebook.simulations import csa_definition, make_static_csa_spectrum

    # Wide enough that the powder pattern is not clipped -- a clipped
    # pattern fits back with a badly shrunken span (measured: 126 ppm
    # against a true 200), which is a property of the data, not a bug.
    ascending = np.linspace(-200.0, 500.0, 512)
    simulated = make_static_csa_spectrum(csa_definition(100.0, 200.0, 0.3),
                                         ascending, fwhm=5.0, n_theta=120, n_phi=240)
    ppm = ascending[::-1]
    spectrum = NMRSpectrum(
        data=simulated["real_spec"][::-1],
        axes=[NMRAxis(kind="frequency", ppm=ppm, hz=ppm * 100.0)],
        metadata=NMRMetadata(vendor="test", nucleus="13C", sfo1=100.0),
    )
    popt, _pcov, _is_free, _scale, info = fit_csa_spectrum(
        results_from_spectrum(spectrum),
        [{"d_iso": 85.0, "span_or_Delta": 170.0, "skew_or_eta": 0.45, "fwhm": 7.0}],
        convention="Haeberlen", static=True, n_theta=100, n_phi=200,
    )
    assert info["n_sites"] == 1
    assert popt[0] == pytest.approx(100.0, abs=3.0)    # d_iso
    assert popt[1] == pytest.approx(200.0, abs=8.0)    # Delta
    assert popt[2] == pytest.approx(0.30, abs=0.08)    # eta
