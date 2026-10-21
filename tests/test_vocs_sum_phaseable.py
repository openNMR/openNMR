"""A VOCS sum keeps an imaginary part, so Phase and Baseline stay available
for it (reported 2026-10-21: a saved VOCS sum could not be phased or
baseline-corrected)."""
import numpy as np

from opennmr.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
from opennmr.processing import vocs


def _spectrum(offset_ppm, with_imag=True, n=2048, sfo1=100.0):
    ppm = np.linspace(offset_ppm + 40.0, offset_ppm - 40.0, n)
    line = 1.0 / (1.0 + ((ppm - offset_ppm) / 2.0) ** 2)
    dispersion = ((ppm - offset_ppm) / 2.0) * line
    axis = NMRAxis(kind="frequency", ppm=ppm, hz=ppm * sfo1)
    meta = NMRMetadata(nucleus="17O", sfo1=sfo1, vendor="bruker")
    return NMRSpectrum(data=line, axes=[axis], metadata=meta,
                       data_imag=dispersion if with_imag else None)


def test_vocs_sum_carries_imaginary_part():
    spectra = [_spectrum(-20.0), _spectrum(0.0), _spectrum(20.0)]
    result = vocs.vocs_sum(spectra, pulse=10.0, npoints=1024)
    assert result["vocs_sum_imag"].shape == result["vocs_sum"].shape
    assert np.any(result["vocs_sum_imag"] != 0.0)
    assert result["imag_rebuilt"] == [False, False, False]


def test_vocs_sum_imag_follows_real_weighting():
    """Real and imaginary parts are weighted and normalised alike, so a
    pure phase rotation of every input rotates the sum the same way."""
    base = [_spectrum(-10.0), _spectrum(10.0)]
    angle = np.deg2rad(37.0)
    rotated = []
    for s in base:
        z = (s.data + 1j * s.data_imag) * np.exp(1j * angle)
        rotated.append(NMRSpectrum(data=z.real, axes=s.axes, metadata=s.metadata,
                                   data_imag=z.imag))
    a = vocs.vocs_sum(base, pulse=10.0, npoints=1024)
    b = vocs.vocs_sum(rotated, pulse=10.0, npoints=1024)
    za = (a["vocs_sum"] + 1j * a["vocs_sum_imag"]) * np.exp(1j * angle)
    zb = b["vocs_sum"] + 1j * b["vocs_sum_imag"]
    assert np.allclose(za, zb, atol=1e-9)


def test_input_without_imaginary_part_is_rebuilt():
    spectra = [_spectrum(-10.0, with_imag=False), _spectrum(10.0)]
    result = vocs.vocs_sum(spectra, pulse=10.0, npoints=1024)
    assert result["imag_rebuilt"] == [True, False]
    assert np.all(np.isfinite(result["vocs_sum_imag"]))


# --- in the GUI -------------------------------------------------------------

import pytest  # noqa: E402

from test_gui import (qapp, _load_spectrum_directly, _select_rows,  # noqa: E402,F401
                      _make_1d_peak_spectrum_with_pulse)


def _complex_peak(x0):
    s = _make_1d_peak_spectrum_with_pulse(x0=x0)
    return NMRSpectrum(data=s.data, axes=s.axes, metadata=s.metadata,
                       data_imag=np.zeros_like(s.data))


def test_gui_vocs_sum_can_be_phased_and_baseline_corrected(qapp):
    from opennmr.gui.main_window import MainWindow
    window = MainWindow()
    for offset in (-2.0, 0.0, 2.0):
        _load_spectrum_directly(window, _complex_peak(offset))
    window.spectrum_list.selectAll()
    window._on_vocs_sum_selection_action()

    summed = window.session.spectra[-1]
    assert summed.metadata.vendor == "derived"
    assert summed.data_imag is not None
    # Phase is on offer for it, not greyed out with "needs retained imaginary data".
    assert window._phase_mode == "1d"
    assert window.phase0_spin.isEnabled()

    before = summed.data.copy()
    window.phase0_spin.setValue(60.0)
    assert not np.allclose(summed.data, before)          # the rotation reached the data

    window.baseline_correct_button.setChecked(True)      # baseline correction switches on
    assert summed.metadata.raw.get("baseline_correct") is True
    window.close()
