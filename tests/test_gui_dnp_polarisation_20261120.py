"""Dynamics ▸ DNP Polarisation (gui/dnp_polarisation_panel.py)."""
import math

import numpy as np
import pytest
from PySide6.QtWidgets import QMessageBox

from opennmr.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
from opennmr.gui.main_window import MainWindow
from opennmr.notebook import dnp_polarisation as dp
from opennmr.processing import slicing

from test_gui import qapp  # noqa: F401 -- qapp is a fixture

PPM = np.linspace(-20.0, 20.0, 2001)
DELAYS = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 16.0])


def _series(amplitude, tau):
    traces = []
    for i, t in enumerate(DELAYS):
        height = amplitude * (1 - math.exp(-t / tau))
        y = height / (1.0 + ((PPM - 3.0) / 0.25) ** 2) + 0.001 * np.random.default_rng(i).standard_normal(PPM.size)
        md = NMRMetadata(vendor="simulated", nucleus="1H", sfo1=400.0, bf1=400.0)
        traces.append(NMRSpectrum(data=y, axes=[NMRAxis(kind="frequency", ppm=PPM, hz=PPM * 400.0)],
                                  metadata=md))
    return slicing.build_pseudo2d(traces)


@pytest.fixture
def window(qapp, monkeypatch):
    shown = []
    for name in ("warning", "information", "critical"):
        monkeypatch.setattr(QMessageBox, name, lambda *a, _n=name, **k: shown.append((_n, a)))
    w = MainWindow()
    w._messages = shown
    w._add_derived_spectrum(_series(10.0, 4.0), "mw on")
    w._add_derived_spectrum(_series(2.0, 6.0), "mw off")
    w.dnp_polarisation_panel.refresh()
    return w


def _panel(window):
    panel = window.dnp_polarisation_panel
    panel.source_combo.setCurrentIndex(panel.source_combo.findData("list"))
    panel.list_edit.setText(", ".join(str(v) for v in DELAYS))
    panel.region_edit.setText("0, 6")
    panel.per_scan_check.setChecked(False)
    panel.divide_rg_check.setChecked(False)
    return panel


def test_the_section_is_in_dynamics_and_is_an_optional_section(window):
    assert "dnp_polarisation" in window.OPTIONAL_SECTIONS
    assert window.OPTIONAL_SECTIONS["dnp_polarisation"][1] == ("_dnp_polarisation_section",)
    assert not window._dnp_polarisation_section.isExpanded()


def test_both_pseudo_2d_experiments_are_listed_and_b_defaults_to_the_second(window):
    panel = window.dnp_polarisation_panel
    names = [panel.a_combo.itemText(i) for i in range(panel.a_combo.count())]
    assert len(names) == 2 and names[0].startswith("mw on") and names[1].startswith("mw off")
    assert panel.a_combo.currentIndex() == 0 and panel.b_combo.currentIndex() == 1


def test_a_one_d_spectrum_is_not_offered(window):
    ppm = np.linspace(10, 0, 100)
    window._add_derived_spectrum(
        NMRSpectrum(data=np.zeros(100), axes=[NMRAxis(kind="frequency", ppm=ppm, hz=ppm * 400)],
                    metadata=NMRMetadata(vendor="simulated", nucleus="1H", sfo1=400.0)), "one-d")
    window.dnp_polarisation_panel.refresh()
    assert window.dnp_polarisation_panel.a_combo.count() == 2


def test_compute_matches_the_notebook_engine(window):
    panel = _panel(window)
    result = panel.compute()
    a = dp.measure_series(window.session.spectra[0], DELAYS, "integral", region=(0, 6),
                          per_scan=False, divide_rg=False)
    b = dp.measure_series(window.session.spectra[1], DELAYS, "integral", region=(0, 6),
                          per_scan=False, divide_rg=False)
    expected = dp.polarisation(a, b, "difference")
    np.testing.assert_allclose(result["y"], expected["y"])
    np.testing.assert_allclose(result["x"], DELAYS)


def test_plot_draws_on_the_fit_view_with_the_chosen_options(window):
    panel = _panel(window)
    panel.individual_check.setChecked(True)
    panel.connect_check.setChecked(False)
    panel.plot()
    figure = window.fit_canvas.figure
    assert len(figure.axes) == 2
    assert not [ln for ln in figure.axes[-1].lines if ln.get_linestyle() == "-"]
    assert "A: mw on" in panel.report_label.text()
    panel.individual_check.setChecked(False)
    panel.connect_check.setChecked(True)
    panel.plot()
    assert len(window.fit_canvas.figure.axes) == 1


def test_start_stop_step_and_start_step_sources(window):
    panel = window.dnp_polarisation_panel
    panel.source_combo.setCurrentIndex(panel.source_combo.findData("start_stop_step"))
    panel.start_edit.setText("1"); panel.stop_edit.setText("6"); panel.step_edit.setText("1")
    values, _ = panel._delays_for(panel._entry(panel.a_combo))
    assert values.tolist() == [1, 2, 3, 4, 5, 6]
    panel.unit_combo.setCurrentIndex(panel.unit_combo.findData("ms"))
    values, _ = panel._delays_for(panel._entry(panel.a_combo))
    assert values == pytest.approx(np.array([1, 2, 3, 4, 5, 6]) * 1e-3)
    panel.source_combo.setCurrentIndex(panel.source_combo.findData("start_step"))
    panel.unit_combo.setCurrentIndex(panel.unit_combo.findData("s"))
    panel.start_edit.setText("1"); panel.step_edit.setText("1"); panel.multiplier_edit.setText("2")
    values, _ = panel._delays_for(panel._entry(panel.a_combo))
    assert values.tolist() == [1, 2, 4, 8, 16, 32]
    assert "6 delays" in panel.delays_label.text()


def test_mistakes_are_explained_not_raised(window):
    panel = window.dnp_polarisation_panel
    panel.source_combo.setCurrentIndex(panel.source_combo.findData("experiment"))
    panel.plot()                                      # no recorded list
    assert window._messages and "records no delay list" in window._messages[-1][1][2]
    panel = _panel(window)
    panel.region_edit.setText("3")
    panel.plot()
    assert "two numbers" in window._messages[-1][1][2]
    panel.b_combo.setCurrentIndex(panel.a_combo.currentIndex())
    panel.region_edit.setText("0, 6")
    panel.plot()
    assert "same experiment" in window._messages[-1][1][2]


def test_stored_fit_quantity_needs_a_stored_fit_and_uses_it_when_present(window):
    panel = _panel(window)
    panel.quantity_combo.setCurrentIndex(panel.quantity_combo.findData("fit_area"))
    panel.plot()
    assert "no stored per-slice fit" in window._messages[-1][1][2]
    for k, scale in ((0, 3.0), (1, 1.0)):
        window.session.spectra[k].metadata.raw["slice_fit_results"] = [
            {"index": i + 1, "integrals": [scale * (i + 1)], "integral_errors": [0.1]}
            for i in range(len(DELAYS))]
    panel.b_combo.setCurrentIndex(1)
    result = panel.compute()
    np.testing.assert_allclose(result["y"], 2.0 * np.arange(1, len(DELAYS) + 1))


def test_export_csv_and_copy_code(window, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QFileDialog
    from PySide6.QtGui import QGuiApplication
    panel = _panel(window)
    path = tmp_path / "pol.csv"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(path), "")))
    panel.export_csv()
    text = path.read_text(encoding="utf-8-sig")
    assert "delay_s,result,result_error,A,A_error,B,B_error" in text
    panel.copy_code()
    code = QGuiApplication.clipboard().text()
    assert "dp.measure_series(on, delays" in code and "region=(0.0, 6.0)" in code
    compile(code, "<copied>", "exec")           # the copied code is valid Python
    assert "dp.plot_polarisation" in code and "np.arange" not in code
