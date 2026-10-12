"""CSA, Quad & J-coupling: select several peaks and give them the current
peak's interactions (Copy to Peaks; explicit user request, 2026-10-12),
and many peaks handled in one go (bulk), in Simulate and in Fit."""
import time

import numpy as np
import pytest
from PySide6.QtWidgets import QMessageBox

from test_gui import qapp  # noqa: F401 -- fixture


@pytest.fixture(autouse=True)
def _dialogs(monkeypatch):
    shown = []
    for name in ("warning", "information", "critical"):
        monkeypatch.setattr(QMessageBox, name,
                            lambda *a, _n=name, **k: shown.append((_n, a[2] if len(a) > 2 else "")))
    return shown


@pytest.fixture
def window(qapp):  # noqa: F811
    from opennmr.gui.main_window import MainWindow
    return MainWindow()


def _sites(n, start=0.0):
    rng = np.random.default_rng(1)
    return [{"label": f"Li {k}", "delta_iso": float(start + d), "amount": float(a),
             "fwhm": 8.0, "lineshape": "gaussian", "eta": 0.5, "interactions": []}
            for k, (d, a) in enumerate(zip(rng.normal(300, 150, n), rng.uniform(0.2, 1, n)))]


def _sim_panel(window, n):
    window._export_pnmr_sites(_sites(n), [], "7Li", 155.5, "simulate", False, "test")
    return window.combined_sim_panel


def test_sites_list_is_multi_select_with_a_copy_menu(window):
    p = _sim_panel(window, 4)
    from PySide6.QtWidgets import QAbstractItemView
    assert p.site_list.selectionMode() == QAbstractItemView.SelectionMode.ExtendedSelection
    assert p.copy_button.isEnabled()
    p.site_list.setCurrentRow(2)
    p._name_copy_actions()
    texts = [a.text() for a in p._copy_actions]
    assert texts[0] == "Li 2's interactions → selected peaks"
    assert texts[3].endswith("and broadening → all peaks")


def test_copy_to_selected_then_to_all(window):
    p = _sim_panel(window, 6)
    p.site_list.setCurrentRow(0)
    p.add_interaction("csa", aniso=-600.0, asym=0.2)
    for row in (2, 4):
        p.site_list.item(row).setSelected(True)
    assert "3 of 6 peaks selected" in p.selection_label.text()
    report = p.copy_interactions("selected")
    assert report["copied"] == 2
    assert [bool(s["interactions"]) for s in p.site_dicts] == [True, False, True, False, True,
                                                              False]
    assert p.site_dicts[2]["interactions"][0]["aniso"] == -600.0
    assert p.site_dicts[2]["interactions"][0] is not p.site_dicts[0]["interactions"][0]
    assert p.site_dicts[2]["delta_iso"] != p.site_dicts[0]["delta_iso"]   # own δiso kept
    assert "CSA copied to 2 peak(s)" in p.selection_label.text().replace("site(s)", "peak(s)")
    p.site_dicts[0]["fwhm"] = 12.0
    report = p.copy_interactions("all", broadening=True)
    assert report["copied"] == 5
    assert all(s["fwhm"] == 12.0 and s["interactions"] for s in p.site_dicts)
    # Clearing the selected ones makes them isotropic again.
    p.site_list.clearSelection()
    p.site_list.item(5).setSelected(True)
    assert p.clear_interactions("selected") == 1 and p.site_dicts[5]["interactions"] == []


def test_nothing_selected_says_how(window):
    p = _sim_panel(window, 3)
    p.site_list.setCurrentRow(0)
    p.site_list.clearSelection()
    report = p.copy_interactions("selected")
    assert report["copied"] == 0 and "Ctrl/Shift-click" in report["text"]


def test_quadrupolar_is_not_given_to_spin_half_or_linked_peaks(window):
    p = window.combined_sim_panel
    p.nucleus_combo.setCurrentText("23Na")
    p.set_state({"widgets": {}, "sites": [
        {"label": "A", "delta_iso": 0.0, "amount": 1.0, "fwhm": 1.0, "lineshape": "gaussian",
         "eta": 0.5, "interactions": [{"type": "quadrupolar", "C_Q": 2.0, "eta_Q": 0.3}]},
        {"label": "B", "delta_iso": 5.0, "amount": 1.0, "fwhm": 1.0, "lineshape": "gaussian",
         "eta": 0.5, "interactions": []},
        {"label": "coil", "isotope": "63Cu", "delta_iso": 0.0, "amount": 1.0, "fwhm": 1.0,
         "lineshape": "gaussian", "eta": 0.5, "interactions": []},
        {"label": "H", "isotope": "1H", "delta_iso": 0.0, "amount": 1.0, "fwhm": 1.0,
         "lineshape": "gaussian", "eta": 0.5, "interactions": []}], "current": 0})
    p.site_list.setCurrentRow(0)
    report = p.copy_interactions("all")
    assert report["copied"] == 3
    assert p.site_dicts[1]["interactions"][0]["type"] == "quadrupolar"
    assert p.site_dicts[2]["interactions"] == [] and p.site_dicts[3]["interactions"] == []
    assert sorted(report["no_quadrupolar"]) == ["H", "coil"]


def test_twelve_hundred_peaks_export_copy_and_simulate_quickly(window):
    t = time.perf_counter()
    p = _sim_panel(window, 1200)
    assert len(p.site_dicts) == 1200 and p.site_list.count() == 1200
    p.regime_combo.setCurrentIndex(p.regime_combo.findData("mas"))
    p.mas_spin.setValue(30000.0)
    p.site_list.setCurrentRow(0)
    p.add_interaction("csa", aniso=-600.0, asym=0.2)
    p.site_list.selectAll()
    assert p.copy_interactions("selected")["copied"] == 1199
    assert p.simulate()
    assert time.perf_counter() - t < 60          # ~3 s; one by one it took minutes
    log = p.results_label.text()
    assert "1200 sites (7Li)" in log and "simulated once" in log
    assert len(p.result["result"]["sites"]) == 1200


def test_fit_panel_takes_many_peaks_in_one_go(window):
    t = time.perf_counter()
    window._export_pnmr_sites(_sites(600), [], "7Li", 155.5, "fit", False, "test")
    fit = window.combined_fit_panel
    assert len(fit.site_dicts) == 600
    assert time.perf_counter() - t < 30           # was ~6 min for 1200 (table rebuilt per peak)
    rows = fit.parameter_table.rowCount()
    fit.site_list.setCurrentRow(0)
    fit.add_interaction("csa", aniso=-600.0, asym=0.2)
    fit.copy_interactions("all")
    assert fit.parameter_table.rowCount() > rows      # CSA rows for every peak, one rebuild
    assert all(s["interactions"] for s in fit.site_dicts)
