"""The plain-language advice in the Fit and Dynamics panels (explicit user
request, 2026-10-06): shown under each fit's results, kept with the result
and its log, and each suggestion's button doing what it says."""

import numpy as np
import pytest
from PySide6.QtWidgets import QMessageBox

from opennmr.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
from opennmr.gui.main_window import MainWindow
from opennmr.notebook.combined_hamiltonian import Site, simulate_combined

from test_gui import (qapp, _load_spectrum_directly, _make_pseudo2d_relaxation_spectrum,  # noqa: F401
                      _set_delay_unit)


@pytest.fixture(autouse=True)
def _dialogs(monkeypatch):
    shown = []
    for name in ("warning", "information", "critical"):
        monkeypatch.setattr(QMessageBox, name,
                            lambda *a, _n=name, **k: shown.append((_n, a[2] if len(a) > 2 else "")))
    return shown


def _two_peak_spectrum(window, second=True):
    x = np.linspace(15, -5, 8192)
    sites = [Site("1H", 2.0, label="A", fwhm=0.05, lineshape="lorentzian")]
    if second:
        sites.append(Site("1H", 7.0, label="B", fwhm=0.05, lineshape="lorentzian", amount=0.5))
    y = simulate_combined(sites, x[::-1], 400e6)["real_spec"][::-1]
    y = y / y.max() * 1000 + np.random.default_rng(0).normal(0, 1, x.size)
    spectrum = NMRSpectrum(data=y, axes=[NMRAxis(kind="frequency", ppm=x)],
                           metadata=NMRMetadata(vendor="simulated", nucleus="1H", sfo1=400.0))
    return window._add_derived_spectrum(spectrum, "two peaks")


def _one_peak_model(window):
    panel = window.combined_fit_panel
    panel.seed_from_spectrum(window.session.spectra[window._selected_index()])
    panel.regime_combo.setCurrentIndex(panel.regime_combo.findData("static"))
    panel.diso_spin.setValue(2.0)
    panel.fwhm_spin.setValue(0.05)
    panel.lineshape_combo.setCurrentIndex(panel.lineshape_combo.findData("lorentzian"))
    return panel


def test_combined_fit_shows_advice_and_its_buttons_act(qapp, _dialogs):
    window = MainWindow()
    _two_peak_spectrum(window)
    panel = _one_peak_model(window)
    assert not panel.advice_box.isVisibleTo(panel)
    assert panel.fit(), _dialogs
    advice = panel.result["advice"]
    add = next(it["action"] for it in advice if it.get("action")
               and it["action"]["type"] == "add_peak")
    assert add["delta_iso"] == pytest.approx(7.0, abs=0.05)
    assert panel.advice_box.isVisibleTo(panel)
    assert "What to try next" in panel.advice_box.text()
    assert "no peak in the model accounts for" in panel.advice_box.text()
    # The advice goes into the fit's record (the processing log).
    assert "What to try next:" in panel.report_text()
    # "Add this peak" adds it where the data say.
    button = next(b for b in panel.advice_box.action_buttons if b.text() == "Add this peak")
    button.click()
    assert len(panel.site_dicts) == 2
    assert panel.site_dicts[1]["delta_iso"] == pytest.approx(7.0, abs=0.05)
    assert not button.isEnabled()
    assert "Done: added Site 2" in panel.advice_box.done_label.text()
    assert panel.fit(), _dialogs
    assert panel.result["advice"][0]["tone"] == "good"
    # Restored with the result, as kept fits and saved states do.
    encoded = panel.result_state()
    panel.clear_result()
    assert not panel.advice_box.isVisibleTo(panel)
    panel.restore_result(encoded, window.session.spectra[0])
    assert panel.advice_box.isVisibleTo(panel)


def test_every_combined_action(qapp):
    window = MainWindow()
    panel = window.combined_fit_panel
    panel.nucleus_combo.setCurrentText("27Al")
    panel.add_interaction("quadrupolar")
    label = panel.site_dicts[0]["label"]
    assert panel.apply_advice_action({"type": "add_interaction", "site": label, "kind": "csa"})
    assert {i["type"] for i in panel.site_dicts[0]["interactions"]} == {"quadrupolar", "csa"}
    assert panel.apply_advice_action({"type": "free_parameter", "site": label, "name": "beta"})
    assert all(panel._constraint(label, n)["free"] for n in ("alpha", "beta", "gamma"))
    assert panel.apply_advice_action({"type": "fix_parameter", "site": label, "name": "eta_Q"})
    assert not panel._constraint(label, "eta_Q")["free"]
    assert panel.apply_advice_action({"type": "widen_bound", "site": label, "name": "C_Q",
                                      "side": "upper", "value": 12.5})
    assert panel._constraint(label, "C_Q")["max"] == "12.5"
    assert panel.apply_advice_action({"type": "set_lineshape", "site": label,
                                      "lineshape": "pseudo_voigt"})
    assert panel.site_dicts[0]["lineshape"] == "pseudo_voigt"
    panel.regime_combo.setCurrentIndex(panel.regime_combo.findData("mas"))
    assert panel.apply_advice_action({"type": "widen_bound", "site": "", "name": "mas_Hz",
                                      "side": "lower", "value": 9000})
    assert panel.mas_constraint()["min"] == "9000"
    panel.add_site()
    assert panel.apply_advice_action({"type": "remove_site", "site": label})
    assert [s["label"] for s in panel.site_dicts] == ["Site 2"]
    assert panel.apply_advice_action({"type": "add_interaction", "site": "nope",
                                      "kind": "csa"}) is None


def test_the_advice_compares_with_kept_fits(qapp, _dialogs, monkeypatch):
    window = MainWindow()
    index = _two_peak_spectrum(window)
    panel = _one_peak_model(window)
    assert panel.fit(), _dialogs
    window._last_fit_key = "combined"
    window._keep_shown_fit(name="one peak", canvas=window.fit_canvas)
    assert [m["name"] for m in window._fit_models(index)] == ["one peak"]
    panel.apply_advice_action({"type": "add_peak", "delta_iso": 7.0, "fwhm": 0.05})
    panel.lineshape_combo.setCurrentIndex(panel.lineshape_combo.findData("lorentzian"))
    assert panel.fit(), _dialogs
    text = panel.advice_box.text()
    assert "This fit (2 peaks, isotropic) describes your data clearly better than “one peak”" \
        in text


def test_isotropic_peak_fit_suggests_the_missing_peak(qapp, _dialogs):
    window = MainWindow()
    _two_peak_spectrum(window)
    window._add_peak_row(shift=2.0, fwhm=0.05, eta=1.0)
    window._on_fit_spectrum_clicked()
    box = window.spectral_fit_advice_box
    assert box.isVisibleTo(box.parentWidget())
    assert "near 7 ppm" in box.text() or "near 7.0" in box.text()
    button = next(b for b in box.action_buttons if b.text() == "Add this peak")
    button.click()
    assert window.peak_table.rowCount() == 2
    assert float(window.peak_table.item(1, 0).text()) == pytest.approx(7.0, abs=0.05)


def test_fit_curve_advice_on_delays_and_model(qapp, _dialogs):
    delays = np.geomspace(0.001, 0.06, 8)            # stop at 1.2 × T2
    window = MainWindow()
    _load_spectrum_directly(window, _make_pseudo2d_relaxation_spectrum(delays, T2=0.05))
    window._select_single_row(0)
    window._add_region_row(4.5, 5.5)
    window._on_integrate_regions_clicked()
    window.delay_source_combo.setCurrentIndex(window._MANUAL_LIST_SOURCE)
    _set_delay_unit(window, "s")
    window.manual_delays_edit.setText(", ".join(f"{d:.6g}" for d in delays))
    window.relaxation_model_combo.setCurrentIndex(2)  # exponential decay (T2)
    window._on_fit_curve_button_clicked()
    box = window.curve_fit_advice_box
    assert box.isVisibleTo(box.parentWidget())
    assert "Your longest delay" in box.text() and "Add delays up to about 5× T2" in box.text()
    # The model switch the advice can offer.
    assert window._apply_curve_advice({"type": "set_relaxation_model",
                                       "model": "stretched_decay"})
    assert window.relaxation_model_combo.currentData() == "stretched_decay"
    assert window._apply_curve_advice({"type": "set_relaxation_model", "model": "nope"}) is None


def test_series_box_says_whether_one_shape_fits_every_slice(qapp, _dialogs):
    window = MainWindow()
    delays = np.geomspace(0.001, 0.3, 6)
    _load_spectrum_directly(window, _make_pseudo2d_relaxation_spectrum(delays, T2=0.05))
    window._select_single_row(0)
    window.combined_fit_panel.seed_from_spectrum(window.session.spectra[0])
    panel = window.combined_fit_panel
    panel.regime_combo.setCurrentIndex(panel.regime_combo.findData("static"))
    panel.nucleus_combo.setCurrentText("1H")
    panel.larmor_spin.setValue(400.0)
    panel.diso_spin.setValue(5.0)
    panel.fwhm_spin.setValue(0.05)
    window.slice_spin.setValue(1)
    assert panel.fit(), _dialogs
    results = window._on_model_series_clicked("combined")
    assert results
    box = window._series_advice_boxes["combined"]
    assert box.isVisibleTo(box.parentWidget())
    assert "only the intensities change" in box.text()
