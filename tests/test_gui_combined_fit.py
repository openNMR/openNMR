"""Fit > Solid-state NMR (gui/combined_panels.CombinedFitPanel): the
site editor of Simulate > Solid-state NMR plus a parameter table
(start / free / bounds / multi-start values), run through
notebook.combined_fitting against synthetic 27Al spectra with known
parameters, and kept with the spectrum like every other fit."""

import json

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMessageBox

from opennmr.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
from opennmr.gui.combined_panels import (_P_FREE, _P_MAX, _P_MIN, _P_NAME, _P_SITE, _P_START,
                                         _P_TRY, CombinedFitPanel)
from opennmr.gui.main_window import MainWindow
from opennmr.io import history as io_history
from opennmr.notebook.combined_hamiltonian import Site, simulate_combined

from test_gui import qapp, _settle_sidecar_save  # noqa: F401 -- qapp is a fixture

NU0 = 130.3e6
MAS = 20e3


@pytest.fixture(autouse=True)
def _dialogs(monkeypatch):
    shown = []
    for name in ("warning", "information", "critical"):
        monkeypatch.setattr(QMessageBox, name,
                            lambda *a, _n=name, **k: shown.append((_n, a[2] if len(a) > 2 else "")))
    return shown


def _truth():
    return [Site("27Al", 12, label="Al1", fwhm=0.6, amount=2.0).add_quadrupolar(4.2, 0.35),
            Site("27Al", -8, label="Al2", fwhm=0.8, amount=1.0).add_quadrupolar(6.0, 0.7)]


def _load(window, sites=None, name="27Al MAS", seed=0):
    x = np.linspace(250, -250, 8192)
    y = simulate_combined(sites or _truth(), x[::-1], NU0, mas_Hz=MAS, transitions="ct",
                          n_zcw=6765, order="exact")["real_spec"][::-1]
    y = y / y.max() * 1000 + np.random.default_rng(seed).standard_normal(x.size) * 3
    spectrum = NMRSpectrum(data=y, axes=[NMRAxis(kind="frequency", ppm=x)],
                           metadata=NMRMetadata(vendor="simulated", nucleus="27Al", sfo1=NU0 / 1e6,
                                                raw={"acqus": {"MASR": MAS}}))
    return window._add_derived_spectrum(spectrum, name)


def _set_up(panel):
    """Two sites near (not at) the truth, CT only, a quick powder set."""
    panel.transitions_combo.setCurrentIndex(panel.transitions_combo.findData("ct"))
    panel.orientations_spin.setValue(987)
    panel.set_state({"widgets": {}, "current": 0, "sites": [
        {"label": "Al1", "delta_iso": 10.0, "amount": 1.0, "fwhm": 1.0, "lineshape": "gaussian",
         "eta": 0.5, "interactions": [{"type": "quadrupolar", "C_Q": 4.0, "eta_Q": 0.5}]},
        {"label": "Al2", "delta_iso": -5.0, "amount": 1.0, "fwhm": 1.0, "lineshape": "gaussian",
         "eta": 0.5, "interactions": [{"type": "quadrupolar", "C_Q": 5.6, "eta_Q": 0.5}]}]})


def _row(panel, label, name):
    t = panel.parameter_table
    for r in range(t.rowCount()):
        if t.item(r, _P_SITE).text() == label and t.item(r, _P_NAME).data(Qt.ItemDataRole.UserRole) == name:
            return r
    raise KeyError((label, name))


def _values(panel):
    return {f"{p['label']}.{p['name']}": p for p in panel.result["parameters"]}


def test_the_section_sits_in_fit_and_is_seeded_from_the_spectrum(qapp):
    window = MainWindow()
    panel = window.combined_fit_panel
    assert isinstance(panel, CombinedFitPanel)
    _load(window)
    assert panel.nucleus_combo.currentText() == "27Al"
    assert panel.larmor_spin.value() == pytest.approx(130.3)
    assert panel.regime_combo.currentData() == "mas" and panel.mas_spin.value() == MAS


def test_the_parameter_table_follows_the_sites(qapp):
    window = MainWindow()
    panel = window.combined_fit_panel
    _set_up(panel)
    names = [(panel.parameter_table.item(r, _P_SITE).text(),
              panel.parameter_table.item(r, _P_NAME).data(Qt.ItemDataRole.UserRole))
             for r in range(panel.parameter_table.rowCount())]
    assert names == [("Al1", n) for n in ("delta_iso", "amount", "fwhm", "C_Q", "eta_Q")] + \
                    [("Al2", n) for n in ("delta_iso", "amount", "fwhm", "C_Q", "eta_Q")]
    # Adding a CSA to a quadrupolar site adds its tensor and the Euler
    # angles -- which start fixed.
    panel.site_list.setCurrentRow(0)
    panel.add_interaction("csa", aniso=50.0, asym=0.3)
    r = _row(panel, "Al1", "beta")
    assert panel.parameter_table.item(r, _P_FREE).checkState() == Qt.CheckState.Unchecked
    assert panel.parameter_table.item(_row(panel, "Al1", "aniso"), _P_START).text() == "50"
    # Editing the site shows in the table, and typing a start edits the site.
    panel.diso_spin.setValue(11.5)
    assert panel.parameter_table.item(_row(panel, "Al1", "delta_iso"), _P_START).text() == "11.5"
    panel.parameter_table.item(_row(panel, "Al1", "C_Q"), _P_START).setText("4.4")
    assert panel.site_dicts[0]["interactions"][0]["C_Q"] == 4.4
    # A rename keeps the site's choices.
    panel.parameter_table.item(_row(panel, "Al1", "fwhm"), _P_FREE).setCheckState(Qt.CheckState.Unchecked)
    panel.label_edit.setText("AlO6")
    panel._on_label_edited("AlO6")
    assert panel.parameter_table.item(_row(panel, "AlO6", "fwhm"), _P_FREE).checkState() == \
        Qt.CheckState.Unchecked


def test_the_fit_recovers_both_sites_and_reports_them(qapp, _dialogs):
    window = MainWindow()
    panel = window.combined_fit_panel
    _load(window)
    _set_up(panel)
    assert panel.fit()
    assert not _dialogs, _dialogs
    v = _values(panel)
    assert v["Al1.C_Q"]["value"] == pytest.approx(4.2, abs=0.02)
    assert v["Al1.eta_Q"]["value"] == pytest.approx(0.35, abs=0.02)
    assert v["Al1.delta_iso"]["value"] == pytest.approx(12.0, abs=0.05)
    assert v["Al2.C_Q"]["value"] == pytest.approx(6.0, abs=0.03)
    assert panel.result["derived"]["Al1"]["share_%"] == pytest.approx(200 / 3, abs=1.0)
    # A table: a row per site, a column per parameter (2026-10-08).
    t = panel.results_table
    rows = [t.verticalHeaderItem(r).text() for r in range(t.rowCount())]
    columns = [t.horizontalHeaderItem(c).text() for c in range(t.columnCount())]
    assert rows[:2] == ["Al1", "Al2"]
    assert {"C_Q (MHz)", "Share (%)", "P_Q (MHz)", "CT c.o.g. (ppm)"} <= set(columns)
    assert "±" in t.item(rows.index("Al1"), columns.index("C_Q (MHz)")).text()
    text = panel.results_label.text()
    assert "Solid-state NMR fit — MAS 20 kHz" in text and "R² 0.99" in text
    assert "COMBINED-HAMILTONIAN FIT REPORT" in panel.text_report()
    assert window.canvas_tabs.currentWidget() is window.fit_canvas


def test_fixed_and_bounded_parameters_are_honoured(qapp, _dialogs):
    window = MainWindow()
    panel = window.combined_fit_panel
    _load(window)
    _set_up(panel)
    t = panel.parameter_table
    t.item(_row(panel, "Al2", "eta_Q"), _P_START).setText("0.7")
    t.item(_row(panel, "Al2", "eta_Q"), _P_FREE).setCheckState(Qt.CheckState.Unchecked)
    t.item(_row(panel, "Al1", "C_Q"), _P_MIN).setText("4.25")
    assert panel.fit()
    v = _values(panel)
    assert v["Al2.eta_Q"]["value"] == 0.7 and not v["Al2.eta_Q"]["free"]
    assert v["Al1.C_Q"]["value"] >= 4.25 and v["Al1.C_Q"]["lower"] == 4.25
    t = panel.results_table
    assert "(fixed)" in t.item(
        [t.verticalHeaderItem(r).text() for r in range(t.rowCount())].index("Al2"),
        [t.horizontalHeaderItem(c).text() for c in range(t.columnCount())].index("η_Q")).text()


def test_bad_cells_are_named_not_fitted(qapp, _dialogs):
    window = MainWindow()
    panel = window.combined_fit_panel
    _load(window)
    _set_up(panel)
    panel.parameter_table.item(_row(panel, "Al1", "C_Q"), _P_MAX).setText("big")
    assert not panel.fit()
    assert "Al1, C_Q (MHz): Min and Max must be numbers" in _dialogs[-1][1]
    panel.parameter_table.item(_row(panel, "Al1", "C_Q"), _P_MAX).setText("")
    panel.window_combo.setCurrentIndex(panel.window_combo.findData("regions"))
    assert not panel.fit()
    assert "Integration regions" in _dialogs[-1][1]


def test_try_starts_run_a_multistart_and_keep_the_best(qapp, _dialogs):
    window = MainWindow()
    panel = window.combined_fit_panel
    _load(window, sites=[_truth()[0]])
    _set_up(panel)
    panel.site_list.setCurrentRow(1)
    panel.remove_site()
    # A deliberately poor start for eta_Q, rescued by the extra start.
    panel.parameter_table.item(_row(panel, "Al1", "eta_Q"), _P_START).setText("0.95")
    panel.parameter_table.item(_row(panel, "Al1", "eta_Q"), _P_TRY).setText("0.3, 0.6")
    panel.screen_orientations_spin.setValue(377)
    assert panel.fit()
    assert len(panel.result["starts"]) == 3
    assert "Multi-start: 3 starts" in panel.results_label.text()
    assert _values(panel)["Al1.eta_Q"]["value"] == pytest.approx(0.35, abs=0.02)


def test_use_fitted_values_and_open_in_simulate(qapp, _dialogs):
    window = MainWindow()
    panel = window.combined_fit_panel
    _load(window)
    _set_up(panel)
    panel.parameter_table.item(_row(panel, "Al1", "fwhm"), _P_FREE).setCheckState(Qt.CheckState.Unchecked)
    panel.fit()
    fitted = _values(panel)["Al1.C_Q"]["value"]
    panel.use_fitted_as_start()
    assert panel.site_dicts[0]["interactions"][0]["C_Q"] == pytest.approx(fitted)
    assert panel.parameter_table.item(_row(panel, "Al1", "fwhm"), _P_FREE).checkState() == \
        Qt.CheckState.Unchecked  # choices kept
    panel._on_send()
    sim = window.combined_sim_panel
    assert sim.site_dicts[0]["interactions"][0]["C_Q"] == pytest.approx(fitted)
    assert sim.regime_combo.currentData() == "mas" and sim.mas_spin.value() == MAS
    assert sim.transitions_combo.currentData() == "ct"
    assert sim.spectrum is not None


def test_the_fit_is_kept_with_its_spectrum_and_logged(qapp, _dialogs):
    window = MainWindow()
    panel = window.combined_fit_panel
    index = _load(window)
    _set_up(panel)
    panel.parameter_table.item(_row(panel, "Al2", "eta_Q"), _P_TRY).setText("0.8")
    panel.fit()
    _settle_sidecar_save(window)
    shown = (panel.results_label.text(), panel.report_text())
    saved = window.session.spectra[index].metadata.raw["fits"]["combined"]
    assert saved["label"] == "Solid-state NMR"
    json.dumps(saved)  # all plain data
    text = io_history.render_log(window.session.info(index)["history"])
    assert "Solid-state NMR fit of 27Al MAS" in text
    # The values table (a row per site, a column per parameter) is in the log.
    assert "| C_Q (MHz) |" in text and "Al1 | " in text and "Al2 | " in text
    # The parameter choices are part of the settings.
    assert "η_Q free, also tried from 0.8" in text and "β (°)" not in text

    _load(window, name="other")
    assert panel.result is None
    panel.remove_site()
    window._select_single_row(index)
    assert (panel.results_label.text(), panel.report_text()) == shown
    assert [s["label"] for s in panel.site_dicts] == ["Al1", "Al2"]
    assert panel.constraints["Al2"]["eta_Q"]["try"] == "0.8"
    panel.use_fitted_as_start()  # works from a restored result
    assert panel.site_dicts[0]["interactions"][0]["C_Q"] == pytest.approx(
        _values(panel)["Al1.C_Q"]["value"])


def test_panel_state_round_trips(qapp):
    panel = MainWindow().combined_fit_panel
    _set_up(panel)
    panel.parameter_table.item(_row(panel, "Al1", "C_Q"), _P_MIN).setText("3")
    before = panel.get_state()
    panel.set_state(json.loads(json.dumps(before)))
    assert panel.get_state() == before
