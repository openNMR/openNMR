"""Simulate ▸ CSA and Simulate ▸ Quadrupolar with several sites, and the
Fit <-> Simulate transfers carrying every site (explicit user request,
2026-09-30: "if you have multiple sites that you fit to in Fit, you can
then simulate the full spectrum in Simulate, adjust things, then hit Send
to Fit as Starting Values and all sites will appear correctly")."""
import json

import numpy as np
import pytest
from PySide6.QtWidgets import QMessageBox

from opennmr.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
from opennmr.gui.main_window import MainWindow
from opennmr.io import history as io_history
from opennmr.notebook import simulations as S

from test_gui import qapp  # noqa: F401 -- fixture

NU0 = 130.3e6


@pytest.fixture(autouse=True)
def _dialogs(monkeypatch):
    shown = []
    for name in ("warning", "information", "critical"):
        monkeypatch.setattr(QMessageBox, name,
                            lambda *a, _n=name, **k: shown.append((_n, a[2] if len(a) > 2 else "")))
    return shown


def _cells(table, row, columns):
    return [float(table.item(row, c).text()) for c in columns]


def _csa_window():
    window = MainWindow()
    window.sim_csa_nucleus_edit.setText("13C")
    window.sim_csa_orientations_spin.setValue(800)
    window.sim_csa_points_spin.setValue(1024)
    return window


# ---- Simulate ▸ CSA ---------------------------------------------------------------

def test_csa_sites_are_added_edited_and_summed(qapp, _dialogs):  # noqa: F811
    window = _csa_window()
    sites = window.sim_csa_sites
    assert len(sites.site_dicts) == 1
    window.sim_csa_diso_spin.setValue(20.0)
    sites.add_site()
    assert sites.current() == 1 and len(sites.site_dicts) == 2
    window.sim_csa_diso_spin.setValue(120.0)
    window.sim_csa_aniso_spin.setValue(-50.0)
    sites.amount_spin.setValue(3.0)
    sites.label_edit.setText("C=O")
    sites.label_edit.textEdited.emit("C=O")
    assert sites.site_dicts[0]["d_iso"] == 20.0                    # site 1 untouched
    assert (sites.site_dicts[1]["d_iso"], sites.site_dicts[1]["span_or_Delta"],
            sites.site_dicts[1]["amount"], sites.site_dicts[1]["label"]) == (120.0, -50.0, 3.0, "C=O")
    # Selecting a site shows its values in the editors.
    sites.select(0)
    assert window.sim_csa_diso_spin.value() == 20.0 and sites.amount_spin.value() == 1.0
    # The auto axis spans every site.
    assert window.sim_csa_ppm_min_spin.value() < 20.0 - 100
    assert window.sim_csa_ppm_max_spin.value() > 120.0 + 60
    window._on_simulate_csa_clicked()
    assert not _dialogs, _dialogs
    spectrum = window._simulated_csa_spectrum
    record = spectrum.metadata.raw["simulation"]
    assert record["n_sites"] == 2 and [s["label"] for s in record["sites"]] == ["Site 1", "C=O"]
    text = window.sim_csa_results_label.text()
    assert "2 sites" in text and "C=O (δiso = 120 ppm, 75%)" in text
    # The same spectrum as the notebook function gives.
    ppm = spectrum.axes[0].ppm
    expected = S.csa_multisite_spectrum(
        sites.sites(), ppm, **window._csa_grid_kwargs("zcw", 800), powder_scheme="zcw")
    np.testing.assert_allclose(spectrum.data, expected["real_spec"])
    # Each site drawn on the Simulation plot, with a legend.
    assert window.canvas_tabs.currentWidget() is window.simulation_canvas
    labels = [t.get_text() for t in window.simulation_canvas.figure.axes[0].get_legend().get_texts()]
    assert labels == ["Total", "Site 1", "C=O"]


def test_one_site_simulates_as_before(qapp, _dialogs):  # noqa: F811
    window = _csa_window()
    window.sim_csa_diso_spin.setValue(42.0)
    window._on_simulate_csa_clicked()
    record = window._simulated_csa_spectrum.metadata.raw["simulation"]
    assert record["d_iso"] == 42.0 and "n_sites" not in record
    assert window.sim_csa_results_label.text().startswith(
        "Simulated static CSA pattern, Haeberlen convention, δiso = 42 ppm.")


def test_duplicate_and_remove_keep_one_site(qapp):  # noqa: F811
    window = _csa_window()
    sites = window.sim_csa_sites
    assert not sites.remove_button.isEnabled()
    window.sim_csa_aniso_spin.setValue(77.0)
    sites.duplicate_site()
    assert sites.site_dicts[1]["span_or_Delta"] == 77.0 and sites.site_dicts[1]["label"] == "Site 2"
    sites.remove_site()
    sites.remove_site()
    assert len(sites.site_dicts) == 1


def test_switching_convention_converts_every_site(qapp):  # noqa: F811
    window = _csa_window()
    sites = window.sim_csa_sites
    sites.set_sites([dict(d_iso=10.0, span_or_Delta=-80.0, skew_or_eta=0.4, fwhm=2, eta=0.5),
                     dict(d_iso=50.0, span_or_Delta=60.0, skew_or_eta=0.2, fwhm=2, eta=0.5)])
    before = [np.diag(S.csa_definition(s["d_iso"], s["span_or_Delta"], s["skew_or_eta"]))
              for s in sites.site_dicts]
    window.sim_csa_convention_combo.setCurrentIndex(
        window.sim_csa_convention_combo.findData("Herzfeld-Berger"))
    after = [np.diag(S.csa_definition(s["d_iso"], s["span_or_Delta"], s["skew_or_eta"],
                                      convention="Herzfeld-Berger")) for s in sites.site_dicts]
    for a, b in zip(before, after):
        np.testing.assert_allclose(np.sort(a), np.sort(b), atol=1e-9)
    assert "span" in sites.site_list.item(0).text()


def test_csa_fit_to_simulation_and_back_with_every_site(qapp):  # noqa: F811
    window = _csa_window()
    x = np.linspace(-200, 250, 1024)
    sim_sites = [dict(label="C1", d_iso=12.0, span_or_Delta=-90.0, skew_or_eta=0.3, fwhm=4.0,
                      eta=0.5, amount=1.0),
                 dict(label="C2", d_iso=140.0, span_or_Delta=60.0, skew_or_eta=0.7, fwhm=3.0,
                      eta=0.5, amount=2.0)]
    # A fit record as fit_csa_spectrum leaves it: amplitudes scale each
    # site's peak-normalised shape, so they are NOT in the ratio 1 : 2.
    kw = dict(powder_scheme="zcw", n_zcw=1500)
    heights = [S.csa_multisite_spectrum([s | {"amount": 1.0}], x, **kw)["real_spec"].max()
               for s in sim_sites]
    popt = []
    for site, height in zip(sim_sites, heights):
        popt += [site["d_iso"], site["span_or_Delta"], site["skew_or_eta"], site["fwhm"],
                 site["amount"] * height * 1e4, 0.0]
    window._csa_fit_result = {
        "popt": np.array(popt + [0.0, 0.0]),
        "info": {"n_sites": 2, "labels": ["C1", "C2"], "x_ref": 0.0,
                 "sim_kwargs": {"convention": "Haeberlen", "static": True, **kw}},
        "convention": "Haeberlen", "static": True, "mas_Hz": None, "x": x,
        "spectrum_index": None, "sites": [{"label": "C1"}, {"label": "C2"}]}
    window._send_csa_fit_to_simulation()
    sites = window.sim_csa_sites
    assert [s["label"] for s in sites.site_dicts] == ["C1", "C2"]
    np.testing.assert_allclose([s["amount"] for s in sites.site_dicts], [0.5, 1.0], rtol=1e-6)
    assert window.sim_csa_ppm_min_spin.value() == -200.0

    # Adjust one site, and send them all back: one fit row per site.
    sites.select(1)
    window.sim_csa_diso_spin.setValue(141.5)
    sites.add_site({"d_iso": 60.0, "span_or_Delta": 30.0, "skew_or_eta": 0.1, "fwhm": 2.0,
                    "label": "C3"})
    table = window.csa_sites_table
    window._csa_use_fitted_as_start()
    table.item(0, 0).setData(window._CSA_FIXED_ROLE, {"fwhm": True})   # a constraint to keep
    window._send_csa_simulation_to_fit()
    assert table.rowCount() == 3
    assert [table.item(r, 0).text() for r in range(3)] == ["C1", "C2", "C3"]
    assert _cells(table, 1, (1, 2, 3, 4)) == [141.5, 60.0, 0.7, 3.0]
    assert _cells(table, 2, (1, 2, 3, 4)) == [60.0, 30.0, 0.1, 2.0]
    assert table.item(0, 0).data(window._CSA_FIXED_ROLE) == {"fwhm": True}
    assert "all 3 site rows" in window.statusBar().currentMessage()


def test_csa_sites_are_journalled_and_restored(qapp, _dialogs):  # noqa: F811
    window = _csa_window()
    sites = window.sim_csa_sites
    window.sim_csa_diso_spin.setValue(5.0)
    sites.add_site({"d_iso": 90.0, "label": "B"})
    window._on_simulate_csa_clicked()
    entry = io_history.load_simulation_journal()["entries"][-1]
    assert any(line.startswith("Site 2 (B): δiso 90 ppm") for line in entry["inputs_text"])
    sites.set_sites([{"d_iso": 0.0}])
    window._restore_simulation(entry)
    assert [s["label"] for s in sites.site_dicts] == ["Site 1", "B"]
    assert [s["d_iso"] for s in sites.site_dicts] == [5.0, 90.0]
    assert window.sim_csa_diso_spin.value() == sites.current_site()["d_iso"]
    assert "2 sites" in window.sim_csa_results_label.text()


def test_state_saved_before_sites_restores_as_one_site(qapp):  # noqa: F811
    window = _csa_window()
    section = window._simulation_panels()["csa"]
    state = json.loads(json.dumps(section.get_state()))
    state.pop("sites")
    state.pop("current")
    # An older record kept the spin boxes as widgets.
    state["widgets"]["sim_csa_diso_spin"] = {"kind": "spin", "value": 33.0}
    window.sim_csa_sites.add_site({"d_iso": 1.0})
    section.set_state(state)
    assert len(window.sim_csa_sites.site_dicts) == 1
    assert window.sim_csa_sites.site_dicts[0]["d_iso"] == 33.0


# ---- Simulate ▸ Quadrupolar --------------------------------------------------------

def _quad_window():
    window = MainWindow()
    window.sim_quad_nucleus_combo.setCurrentText("27Al")
    window.sim_quad_larmor_spin.setValue(NU0 / 1e6)
    window.sim_quad_transitions_combo.setCurrentIndex(
        window.sim_quad_transitions_combo.findData("ct"))
    window.sim_quad_method_combo.setCurrentIndex(
        window.sim_quad_method_combo.findData("second_order"))
    window.sim_quad_orientations_spin.setValue(2000)
    window.sim_quad_points_spin.setValue(1024)
    return window


def test_quadrupolar_sites_simulate_and_compare_orders(qapp, _dialogs):  # noqa: F811
    window = _quad_window()
    sites = window.sim_quad_sites
    window.sim_quad_cq_spin.setValue(3.0)
    window.sim_quad_diso_spin.setValue(10.0)
    sites.add_site({"C_Q_MHz": 6.0, "eta_Q": 0.6, "d_iso": 60.0, "fwhm": 3.0, "amount": 3.0,
                    "label": "Al(IV)"})
    window._on_simulate_quadrupolar_clicked()
    assert not _dialogs, _dialogs
    text = window.sim_quad_results_label.text()
    assert "2 sites" in text and "Al(IV) (C_Q = 6 MHz, 75%, CT c.o.g." in text
    record = window._simulated_quadrupolar_spectrum.metadata.raw["simulation"]
    assert record["n_sites"] == 2 and record["lineshape"] == "gaussian"
    # The auto axis frames both sites.
    assert window.sim_quad_ppm_max_spin.value() > 60.0
    window.sim_quad_compare_check.setChecked(True)
    window._on_simulate_quadrupolar_clicked()
    assert not _dialogs, _dialogs


def _quad_fit_spectrum(window):
    x = np.linspace(120, -100, 2048)
    sites = [dict(label="Al1", C_Q_MHz=3.0, eta_Q=0.2, d_iso=10.0, fwhm=1.5, amount=1.0),
             dict(label="Al2", C_Q_MHz=6.0, eta_Q=0.6, d_iso=60.0, fwhm=3.0, amount=3.0)]
    y = S.quadrupolar_multisite_spectrum(sites, 2.5, NU0, x[::-1], order="second",
                                         transitions="ct", n_zcw=4181)["real_spec"][::-1] * 1e5
    spectrum = NMRSpectrum(data=y, axes=[NMRAxis(kind="frequency", ppm=x)],
                           metadata=NMRMetadata(vendor="simulated", nucleus="27Al",
                                                sfo1=NU0 / 1e6, raw={}))
    window._add_derived_spectrum(spectrum, "27Al two sites")


def test_quadrupolar_fit_to_simulation_and_back_with_every_site(qapp, _dialogs):  # noqa: F811
    window = _quad_window()
    panel = window.quad_fit_panel
    _quad_fit_spectrum(window)
    panel.transitions_combo.setCurrentIndex(panel.transitions_combo.findData("ct"))
    panel.order_combo.setCurrentIndex(panel.order_combo.findData("second"))
    for column, value in ((0, "Al1"), (1, "11"), (2, "3.1"), (3, "0.25"), (4, "1.5")):
        panel.sites_table.item(0, column).setText(value)
    panel.add_site(("Al2", "58", "5.8", "0.55", "3", "0.5"))
    panel.fit()
    assert not _dialogs, _dialogs
    panel.results_table.setCurrentCell(1, 0)
    panel.send_button.click()
    sites = window.sim_quad_sites
    assert [s["label"] for s in sites.site_dicts] == ["Al1", "Al2"]
    assert sites.current() == 1
    np.testing.assert_allclose([s["amount"] for s in sites.site_dicts], [1 / 3, 1.0], rtol=0.02)
    assert window.sim_quad_cq_spin.value() == pytest.approx(6.0, abs=0.02)
    assert window.side_tabs.tabText(window.side_tabs.currentIndex()) == "Simulate"

    # Adjust and send back: both rows, labels and adjusted values.
    window.sim_quad_diso_spin.setValue(59.0)
    window._send_quadrupolar_simulation_to_fit()
    table = panel.sites_table
    assert table.rowCount() == 2
    assert [table.item(r, 0).text() for r in range(2)] == ["Al1", "Al2"]
    assert float(table.item(1, 1).text()) == 59.0
    assert float(table.item(0, 2).text()) == pytest.approx(3.0, abs=0.02)
    # One site sent back goes into the selected row only.
    sites.select(0)
    sites.remove_site()
    table.setCurrentCell(0, 1)
    window.sim_quad_cq_spin.setValue(7.5)
    window._send_quadrupolar_simulation_to_fit()
    assert table.rowCount() == 2 and float(table.item(0, 2).text()) == 7.5


def test_quadrupolar_pseudo_voigt_reaches_the_fit(qapp):  # noqa: F811
    window = _quad_window()
    window.sim_quad_lineshape_combo.setCurrentIndex(
        window.sim_quad_lineshape_combo.findData("pseudo_voigt"))
    assert window.sim_quad_pv_eta_spin.isEnabled()
    window.sim_quad_pv_eta_spin.setValue(0.3)
    window._send_quadrupolar_simulation_to_fit()
    panel = window.quad_fit_panel
    assert panel.broadening_combo.currentData() == "pseudo_voigt"
    assert float(panel.sites_table.item(0, 5).text()) == 0.3
