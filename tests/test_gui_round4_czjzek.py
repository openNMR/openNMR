"""2026-09-29 (fourth round), Czjzek: the extended model (a mean EFG
C_Q0, η_Q0 plus a random part) in Fit and Simulate, and the live line
saying where a distribution sits. Synthetic data only."""
import numpy as np
import pytest

from opennmr.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
from opennmr.notebook import czjzek as cz


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(qapp, tmp_path, monkeypatch):
    from opennmr.gui.main_window import MainWindow
    monkeypatch.setattr(MainWindow, "_FIGURE_STYLES_PATH", str(tmp_path / "figure_styles.json"))
    return MainWindow()


def _ordered_glass(window):
    x = np.linspace(200, -200, 2048)
    sim = cz.czjzek_spectrum(2.5, 0.8, 130.3e6, x, delta_iso=60, fwhm=3, cq0_MHz=5.0, eta0=0.3)
    y = 1000 * sim["real_spec"] / sim["real_spec"].max()
    y = y + np.random.default_rng(4).normal(0, 3, x.size)
    spectrum = NMRSpectrum(y, [NMRAxis("frequency", ppm=x, hz=x * 130.3)],
                           NMRMetadata(vendor="bruker", nucleus="27Al", sfo1=130.3))
    return window._add_derived_spectrum(spectrum, "ordered glass")


def test_fit_panel_says_where_plain_czjzek_sits(window):
    panel = window.czjzek_fit_panel
    assert panel.model() == "czjzek"
    assert panel.sites_table.isColumnHidden(6) and panel.sites_table.isColumnHidden(7)
    panel.sites_table.item(0, 2).setText("2")
    panel._describe_sites()
    text = panel.distribution_label.text()
    mean = cz.czjzek_statistics(2.0)["mean_CQ"]
    assert f"{mean:.3g}" in text and "centred on zero" in text
    panel.sites_table.item(0, 2).setText("oops")
    panel._describe_sites()
    assert panel.distribution_label.text().endswith("—")


def test_extended_model_columns_d_and_validation(window):
    panel = window.czjzek_fit_panel
    panel.sites_table.item(0, 3).setText("4")
    panel.model_combo.setCurrentIndex(panel.model_combo.findData("extended"))
    assert not panel.sites_table.isColumnHidden(6)
    assert panel.sites_table.item(0, 3).text() == "5"          # d put back to 5, held
    assert panel._constraints(0)[1].get("d")
    panel.sites_table.item(0, 6).setText("4")
    panel.sites_table.item(0, 7).setText("0.2")
    site = panel.read_sites()[0]
    assert site["model"] == "extended" and site["CQ0"] == 4 and site["eta0"] == 0.2
    panel._describe_sites()
    assert "C_Q0 = 4 MHz" in panel.distribution_label.text()
    panel.sites_table.item(0, 7).setText("1.5")
    with pytest.raises(ValueError, match="η_Q0"):
        panel.read_sites()
    panel.sites_table.item(0, 7).setText("0.2")
    panel.set_constraints(0, {}, {})                          # d freed by hand
    with pytest.raises(ValueError, match="d = 5"):
        panel.read_sites()
    # The model choice is part of the saved inputs.
    state = panel.get_state()
    panel.model_combo.setCurrentIndex(0)
    panel.set_state(state)
    assert panel.model() == "extended"


def test_extended_fit_in_the_gui_reports_and_sends_the_mean_efg(window):
    _ordered_glass(window)
    panel = window.czjzek_fit_panel
    panel.model_combo.setCurrentIndex(panel.model_combo.findData("extended"))
    for column, value in ((1, "55"), (2, "1.2"), (4, "4"), (6, "4"), (7, "0.5")):
        panel.sites_table.item(0, column).setText(value)
    panel.grid_cq_spin.setValue(48)
    panel.fit()
    row = panel.result["report"][0]
    assert row["extended"]
    assert row["CQ0"] == pytest.approx(5.0, rel=0.02)
    assert row["sigma"] == pytest.approx(0.8, rel=0.06)
    headers = [panel.results_table.horizontalHeaderItem(c).text().replace("\n", " ")
               for c in range(panel.results_table.columnCount())]
    assert "C_Q0 (MHz)" in headers and "η_Q0" in headers and "d" not in headers
    assert "mean EFG" in panel.results_label.text()
    panel.show_map()
    labels = [l.get_label() for l in window.fit_canvas.figure.czjzek_axes["map"].get_lines()]
    assert any("Q0" in l for l in labels)
    panel._on_send()
    sim = window.czjzek_sim_panel
    assert sim.model_combo.currentData() == "extended"
    assert sim.cq0_spin.value() == pytest.approx(row["CQ0"], abs=1e-3)


def test_sim_panel_extended_model(window):
    sim = window.czjzek_sim_panel
    assert not sim.cq0_spin.isVisibleTo(sim)
    assert "centred on zero" in sim.distribution_label.text()
    sim.d_spin.setValue(4.0)
    sim.model_combo.setCurrentIndex(sim.model_combo.findData("extended"))
    assert sim.cq0_spin.isVisibleTo(sim) and not sim.d_spin.isEnabled()
    assert sim.d_spin.value() == 5.0
    sim.cq0_spin.setValue(4.0)
    sim.eta0_spin.setValue(0.3)
    sim.sigma_spin.setValue(0.5)
    assert "C_Q0 = 4 MHz" in sim.distribution_label.text()
    sim.grid_cq_spin.setValue(32)
    result = sim.simulate()
    assert result["extended"] and result["cq0"] == 4.0
    assert "extended Czjzek" in sim.results_label.text()
    n = len(window.session)
    sim.export()
    assert len(window.session) == n + 1
    raw = window.session.spectra[-1].metadata.raw["simulation"]
    assert raw["model"] == "Extended Czjzek" and raw["CQ0_MHz"] == 4.0
