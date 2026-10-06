"""2026-09-29 (fourth round): values between Fit and Simulate -- "Send to
Simulate" for every lineshape fit (CSA added; the site selected in the
results is sent), "Send to Fit as Starting Values" from every lineshape
simulation, and "Use Fitted Values as Start" in every lineshape fit.
Synthetic data only."""
import numpy as np
import pytest

from opennmr.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
from opennmr.notebook import simulations as S


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(qapp, tmp_path, monkeypatch):
    from opennmr.gui.main_window import MainWindow
    from PySide6.QtWidgets import QMessageBox
    monkeypatch.setattr(MainWindow, "_FIGURE_STYLES_PATH", str(tmp_path / "figure_styles.json"))
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: pytest.fail(str(a[1:])))
    return MainWindow()


def _cells(table, row, columns):
    return [float(table.item(row, c).text()) for c in columns]


# ---- notebook: CSA conventions ------------------------------------------------

@pytest.mark.parametrize("args", [(10, -80, 0.4), (0, 100, 0.0), (3, -30, 0.9), (50, 20, 0.2)])
def test_csa_conversion_keeps_the_tensor(args):
    hb = S.convert_csa_parameters(*args, "Haeberlen", "Herzfeld-Berger")
    back = S.convert_csa_parameters(*hb, "Herzfeld-Berger", "Haeberlen")
    np.testing.assert_allclose(np.sort(np.diag(S.csa_definition(*args, "Haeberlen"))),
                               np.sort(np.diag(S.csa_definition(*hb, "Herzfeld-Berger"))))
    np.testing.assert_allclose(back, args, atol=1e-9)
    assert hb[1] >= 0                                       # a span is never negative
    with pytest.raises(ValueError):
        S.convert_csa_parameters(*args, "Haeberlen", "Mehring")


# ---- quadrupolar and Czjzek -----------------------------------------------------

def _fake_quad_result(panel):
    popt = np.array([10.0, 3.2, 0.4, 2.0, 100.0, 0.0,
                     -5.0, 1.5, 0.8, 3.0, 50.0, 0.0, 0.0, 0.0])
    panel.result = {"popt": popt, "info": {"n_sites": 2, "labels": ["A", "B"]},
                    "settings": {"larmor_Hz": 130.3e6, "mas_Hz": 20000.0, "transitions": "ct",
                                 "order": "second"},
                    "nucleus": "27Al", "x": np.linspace(-100, 100, 11)}
    return popt


def test_quadrupolar_use_fitted_and_send_selected_site(window, monkeypatch):
    panel = window.quad_fit_panel
    popt = _fake_quad_result(panel)
    panel.use_fitted_as_start()
    assert panel.sites_table.rowCount() == 2
    assert _cells(panel.sites_table, 1, (1, 2, 3, 4)) == [-5.0, 1.5, 0.8, 3.0]
    assert panel.sites_table.item(1, 0).text() == "B"
    sent = []
    panel._send_to_simulation = sent.append
    panel.results_table.setRowCount(2)
    panel.results_table.setColumnCount(1)
    panel.results_table.setCurrentCell(1, 0)
    panel._on_send()
    assert sent[-1]["C_Q_MHz"] == pytest.approx(1.5) and sent[-1]["delta_iso"] == -5.0


def test_quadrupolar_simulation_to_fit_keeps_the_spectrums_field(window):
    panel = window.quad_fit_panel
    panel.nucleus_combo.setCurrentText("27Al")
    panel.larmor_spin.setValue(156.4)
    window.sim_quad_nucleus_combo.setCurrentText("27Al")
    window.sim_quad_larmor_spin.setValue(130.3)
    window.sim_quad_cq_spin.setValue(4.5)
    window.sim_quad_eta_spin.setValue(0.3)
    window.sim_quad_diso_spin.setValue(62.0)
    window.sim_quad_fwhm_spin.setValue(2.5)
    window.sim_quad_regime_combo.setCurrentIndex(window.sim_quad_regime_combo.findData("mas"))
    window.sim_quad_mas_spin.setValue(25000.0)
    window.sim_quad_method_combo.setCurrentIndex(
        window.sim_quad_method_combo.findData("second_order"))
    window.sim_quad_transitions_combo.setCurrentIndex(
        window.sim_quad_transitions_combo.findData("ct"))
    window._send_quadrupolar_simulation_to_fit()
    assert _cells(panel.sites_table, 0, (1, 2, 3, 4)) == [62.0, 4.5, 0.3, 2.5]
    assert panel.regime_combo.currentData() == "mas" and panel.mas_spin.value() == 25000.0
    assert panel.order_combo.currentData() == "second"
    assert panel.larmor_spin.value() == pytest.approx(156.4)        # the spectrum's, kept
    assert "130.3" in window.statusBar().currentMessage()


def test_czjzek_fit_simulate_round_trip(window):
    x = np.linspace(200, -200, 1024)
    sim = __import__("opennmr.notebook.czjzek", fromlist=["x"]).czjzek_spectrum(
        2.5, 2.0, 130.3e6, x, delta_iso=40, fwhm=4)
    y = 1000 * sim["real_spec"] / sim["real_spec"].max()
    spectrum = NMRSpectrum(y, [NMRAxis("frequency", ppm=x, hz=x * 130.3)],
                           NMRMetadata(vendor="bruker", nucleus="27Al", sfo1=130.3))
    window._add_derived_spectrum(spectrum, "glass")
    fit = window.czjzek_fit_panel
    fit.sites_table.item(0, 1).setText("35")
    fit.sites_table.item(0, 2).setText("1.5")
    fit.grid_cq_spin.setValue(24)
    fit.grid_eta_spin.setValue(8)
    fit.fit()
    assert fit.use_fitted_button.isEnabled()
    fitted = fit.result["report"][0]
    fit.sites_table.item(0, 2).setText("9")
    fit.use_fitted_as_start()
    assert float(fit.sites_table.item(0, 2).text()) == pytest.approx(fitted["sigma"], rel=1e-5)
    # Simulate ▸ Czjzek (extended) -> Fit: model, mean EFG and grid follow.
    s = window.czjzek_sim_panel
    s.model_combo.setCurrentIndex(s.model_combo.findData("extended"))
    s.cq0_spin.setValue(4.0)
    s.eta0_spin.setValue(0.25)
    s.sigma_spin.setValue(0.7)
    s.grid_cq_spin.setValue(40)
    s.send_to_fit()
    assert fit.model() == "extended"
    assert _cells(fit.sites_table, 0, (2, 3, 6, 7)) == [0.7, 5.0, 4.0, 0.25]
    assert fit.grid_cq_spin.value() == 40


# ---- CSA ---------------------------------------------------------------------------

def test_csa_use_fitted_send_and_back(window):
    window._csa_fit_result = {
        "popt": np.array([12.0, -90.0, 0.3, 4.0, 10.0, 0.5, 40.0, 60.0, 0.7, 3.0, 5.0, 0.2,
                          0.0, 0.0]),
        "info": {"n_sites": 2}, "convention": "Haeberlen", "static": True, "mas_Hz": None,
        "x": np.linspace(-200, 250, 11), "spectrum_index": None,
        "sites": [{"label": "C1"}, {"label": "C2"}]}
    window._report_csa_fit = lambda: None
    window._csa_use_fitted_as_start()
    t = window.csa_sites_table
    assert t.rowCount() == 2
    assert _cells(t, 1, (1, 2, 3, 4)) == [40.0, 60.0, 0.7, 3.0]
    window.csa_results_table.setRowCount(2)
    window.csa_results_table.setColumnCount(1)
    window.csa_results_table.setCurrentCell(1, 0)
    window._send_csa_fit_to_simulation()
    # Every fitted site goes (2026-09-30), the one selected in the results
    # table selected in Simulate.
    sites = window.sim_csa_sites
    assert [s["label"] for s in sites.site_dicts] == ["C1", "C2"]
    assert sites.current() == 1
    assert window.sim_csa_diso_spin.value() == 40.0 and window.sim_csa_aniso_spin.value() == 60.0
    assert window.sim_csa_ppm_max_spin.value() == 250.0
    # Straight back: every row as it was.
    window._send_csa_simulation_to_fit()
    assert t.rowCount() == 2
    assert _cells(t, 0, (1, 2, 3, 4)) == [12.0, -90.0, 0.3, 4.0]
    assert _cells(t, 1, (1, 2, 3, 4)) == [40.0, 60.0, 0.7, 3.0]
    # One site back, into fit row 1, with the simulation in the other
    # convention: converted, since site 2 stays in the fit's.
    sites.select(0)
    sites.remove_site()
    window.sim_csa_convention_combo.setCurrentIndex(
        window.sim_csa_convention_combo.findData("Herzfeld-Berger"))
    window.sim_csa_diso_spin.setValue(10.0)
    window.sim_csa_aniso_spin.setValue(136.0)
    window.sim_csa_skeweta_spin.setValue(0.529412)
    t.setCurrentCell(0, 1)
    window._send_csa_simulation_to_fit()
    assert window.csa_convention_combo.currentData() == "Haeberlen"
    assert t.rowCount() == 2
    d_iso, delta, eta = _cells(t, 0, (1, 2, 3))
    # (the skew spin box holds three decimals)
    assert (d_iso, delta, eta) == pytest.approx((10.0, -80.0, 0.4), abs=2e-2)


# ---- J-coupling and combined --------------------------------------------------------

def test_jcoupling_simulation_to_fit(window):
    sim = window.jcoupling_sim_panel
    # First order is the panel's restorable mode (it opens on spin systems).
    sim.model_combo.setCurrentIndex(sim.model_combo.findData("first_order"))
    sim.send_to_fit()
    fit = window.jcoupling_fit_panel
    assert fit.editor.sites_table.rowCount() == sim.editor.sites_table.rowCount()
    assert fit.editor.sites_table.item(0, 1).text() == sim.editor.sites_table.item(0, 1).text()
    assert fit.model_combo.currentData() == "first_order"
    sim.model_combo.setCurrentIndex(sim.model_combo.findData("spin_system"))
    sim.send_to_fit()
    assert fit.model_combo.currentData() == "spin_system"
    assert fit.spin_editor.labels() == sim.spin_editor.labels()
    assert window.combined_fit_model.model() == "spin_system"


def test_combined_simulation_to_fit(window):
    sim = window.combined_sim_panel
    assert sim.to_fit_button.isEnabled()
    sim.send_to_fit()
    fit = window.combined_fit_panel
    assert [s["label"] for s in fit.site_dicts] == [s["label"] for s in sim.site_dicts]
    assert fit.site_dicts is not sim.site_dicts and fit.site_dicts[0] is not sim.site_dicts[0]
    assert fit.parameter_table.rowCount() > 0
