"""Fit > Quadrupolar (gui/quadrupolar_panels.py): table -> notebook
fitter -> results table, against a synthetic 27Al spectrum with known
parameters loaded as an ordinary spectrum."""
import numpy as np
import pytest
from PySide6.QtWidgets import QMessageBox

from opennmr.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
from opennmr.gui.main_window import MainWindow
from opennmr.notebook import simulations as S


def _close(got, want, tol):
    got, want, tol = (np.asarray(v, dtype=float) for v in (got, want, tol))
    assert np.all(np.abs(got - want) <= tol), f"{got} vs {want} (tol {tol})"

from test_gui import qapp  # noqa: F401 -- fixture

NU0 = 130.3e6


@pytest.fixture(autouse=True)
def _dialogs(monkeypatch):
    shown = []
    for name in ("warning", "information", "critical"):
        monkeypatch.setattr(QMessageBox, name,
                            lambda *a, _n=name, **k: shown.append((_n, a[2] if len(a) > 2 else "")))
    return shown


def _load(window, mas=0.0, masr=None):
    x = np.linspace(60, -80, 4096)
    y = 1000 * S.quadrupolar_spectrum(2.5, 4.2e6, 0.35, NU0, x[::-1], mas_Hz=mas,
                                      transitions="ct", delta_iso=12.0, fwhm=0.6,
                                      n_zcw=28657)["normalised_real_spec"][::-1]
    y += np.random.default_rng(0).standard_normal(x.size) * 3 + 2
    raw = {"acqus": {"MASR": masr}} if masr else {}
    spectrum = NMRSpectrum(data=y, axes=[NMRAxis(kind="frequency", ppm=x)],
                           metadata=NMRMetadata(vendor="simulated", nucleus="27Al",
                                                sfo1=NU0 / 1e6, raw=raw))
    window._add_derived_spectrum(spectrum, "27Al synthetic")
    return spectrum


def test_seeded_estimated_fitted_and_reported(qapp, _dialogs):  # noqa: F811
    window = MainWindow()
    panel = window.quad_fit_panel
    _load(window)
    assert panel.nucleus_combo.currentText() == "27Al" and panel.spin_label.text() == "2.5"
    assert panel.larmor_spin.value() == pytest.approx(130.3)
    assert panel.regime_combo.currentData() == "static"
    panel.estimate()
    assert float(panel.sites_table.item(0, 2).text()) == pytest.approx(4.2, rel=0.15)
    panel.fit()
    assert not _dialogs, _dialogs
    p = panel.result["popt"]
    _close(p[:3], [12.0, 4.2, 0.35], [0.05, 0.02, 0.02])
    t = panel.results_table
    headers = [t.horizontalHeaderItem(c).text() for c in range(t.columnCount())]
    assert headers[:5] == ["δiso\n(ppm)", "C_Q\n(MHz)", "η_Q", "P_Q\n(MHz)", "CT c.o.g.\n(ppm)"]
    assert float(t.item(0, 1).text().split("\n")[0]) == pytest.approx(4.2, abs=0.01)
    assert "R² 0.99" in panel.results_label.text()
    assert "Site\tδiso (ppm)" in panel.table_tsv()
    assert "QUADRUPOLAR FIT REPORT" in panel.text_report()
    assert window.canvas_tabs.currentWidget() is window.fit_canvas


def test_mas_rate_is_seeded_and_mas_fit_runs(qapp, _dialogs):  # noqa: F811
    window = MainWindow()
    panel = window.quad_fit_panel
    _load(window, mas=15e3, masr=15000)
    assert panel.regime_combo.currentData() == "mas" and panel.mas_spin.value() == 15000
    panel.orientations_spin.setValue(1597)
    panel.render_orientations_spin.setValue(2584)
    panel.estimate()
    panel.fit()
    assert not _dialogs, _dialogs
    assert panel.result["popt"][1] == pytest.approx(4.2, abs=0.05)
    assert "MAS 15 kHz" in panel.results_label.text()


def test_fixed_parameter_and_send_to_simulation(qapp, _dialogs):  # noqa: F811
    window = MainWindow()
    panel = window.quad_fit_panel
    _load(window)
    for column, value in ((1, "12"), (2, "4.1"), (3, "0.35"), (4, "1")):
        panel.sites_table.item(0, column).setText(value)
    panel.set_constraints(0, {}, {"eta_Q": True})
    assert panel.sites_table.item(0, 3).font().italic()
    panel.fit()
    assert panel.result["popt"][2] == 0.35 and "(fixed)" in panel.results_table.item(0, 2).text()
    assert panel.send_button.isEnabled()
    panel.send_button.click()
    assert window.sim_quad_cq_spin.value() == pytest.approx(panel.result["popt"][1], abs=1e-3)
    assert window.sim_quad_diso_spin.value() == pytest.approx(panel.result["popt"][0], abs=1e-3)
    assert window.sim_quad_transitions_combo.currentData() == "ct"
    assert window.side_tabs.tabText(window.side_tabs.currentIndex()) == "Simulate"


def test_refusals(qapp, _dialogs):  # noqa: F811
    window = MainWindow()
    panel = window.quad_fit_panel
    panel.fit()                                         # nothing selected / no nucleus
    assert _dialogs
    _dialogs.clear()
    _load(window)
    panel.nucleus_combo.setCurrentText("2H")            # integer spin -> satellites
    assert panel.transitions_combo.currentData() == "all"
    panel.transitions_combo.setCurrentIndex(panel.transitions_combo.findData("ct"))
    panel.fit()
    assert _dialogs and "no central transition" in _dialogs[-1][1]
