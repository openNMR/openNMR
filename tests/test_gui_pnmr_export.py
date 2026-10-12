"""Export pNMR sites to CSA, Quad & J-coupling (gui/pnmr_export_row.py on
Simulate ▸ pNMR (CIF) and Fit ▸ pNMR (CIF bond pathways); explicit user
request, 2026-10-12), and the rename of Combined Hamiltonian.

The rock-salt Na(Cl,Br) model of test_gui_pnmr_fit.py: seven isotropic
peaks at -60..+60 ppm with binomial(6, 1/2) weights."""
import numpy as np
import pytest
from PySide6.QtCore import QItemSelectionModel

pytest.importorskip("pymatgen")

from test_gui import qapp  # noqa: F401,E402 -- fixture
from test_gui_pnmr_fit import (  # noqa: F401,E402 -- fixtures
    SHIFTS, _no_modal_dialogs, fitted, settings_file, _window_with_model,
)

NAME = "CSA, Quad & J-coupling"


def test_rename_in_fit_and_simulate(qapp):  # noqa: F811
    from opennmr.gui.main_window import MainWindow
    window = MainWindow()
    assert window._combined_fit_section.title() == NAME
    assert window._combined_sim_section.title() == NAME
    # The "&" is shown, not taken as a keyboard mnemonic.
    assert "Quad && J-coupling" in window._combined_fit_section._header.text()
    assert window.combined_fit_panel.fit_button.text() == "Fit CSA, Quad && J-coupling"


def test_simulation_exports_every_group_to_simulate(qapp, settings_file):  # noqa: F811
    window, _fit = _window_with_model(settings_file)
    sim = window.pnmr_panel
    assert not sim.send_sites_row.export_all_button.isEnabled()
    assert sim.enumerate_sites()
    assert sim.send_sites_row.export_all_button.isEnabled()
    sim.export_min_spin.setValue(0.0)
    sites = sim.export_sites("all")
    assert sorted(s["delta_iso"] for s in sites) == pytest.approx(SHIFTS, abs=1e-6)
    panel = window.combined_sim_panel
    assert [s["delta_iso"] for s in panel.site_dicts] == pytest.approx(
        [s["delta_iso"] for s in sites])
    assert all("nucleus" not in s for s in panel.site_dicts)
    assert panel.nucleus_combo.currentText() == "23Na"
    assert panel.larmor_spin.value() == pytest.approx(sim._result[0]["BF1"], rel=1e-6)
    assert window._combined_sim_section.isExpanded()
    assert "Exported 7 site(s) to Simulate" in sim.log_edit.toPlainText()
    # Interactions can now be added to a site and it simulates.
    panel.site_list.setCurrentRow(0)
    panel.add_interaction("quadrupolar", C_Q=1.0, eta_Q=0.0)
    assert panel.site_dicts[0]["interactions"][0]["type"] == "quadrupolar"


def test_simulation_exports_selected_configurations_and_appends(qapp, settings_file):  # noqa: F811
    window, _fit = _window_with_model(settings_file)
    sim = window.pnmr_panel
    assert sim.enumerate_sites()
    sim.export_min_spin.setValue(0.0)
    sim.export_sites("all")
    view = sim.shift_table_view
    view.selectionModel().select(view.model().index(0, 0),
                                 QItemSelectionModel.SelectionFlag.Select
                                 | QItemSelectionModel.SelectionFlag.Rows)
    sim.send_sites_row.append_check.setChecked(True)
    added = sim.export_sites("selected")
    assert len(added) == 1
    # The most probable configuration is shown first: 0 ppm (20/64).
    assert added[0]["delta_iso"] == pytest.approx(0.0, abs=1e-6)
    labels = [s["label"] for s in window.combined_sim_panel.site_dicts]
    assert len(labels) == 8 and len(set(labels)) == 8               # appended, renamed unique


def test_selected_needs_a_selection(qapp, settings_file, _no_modal_dialogs):  # noqa: F811
    window, _fit = _window_with_model(settings_file)
    assert window.pnmr_panel.enumerate_sites()
    assert window.pnmr_panel.export_sites("selected") is None
    assert any("Select one or more" in text for _kind, text in _no_modal_dialogs)


def test_fit_exports_fitted_peaks_as_the_fit_start(fitted):
    window, panel = fitted
    assert panel.send_sites_row.destination() == "fit"
    sites = panel.export_sites("all")
    assert sorted(s["delta_iso"] for s in sites) == pytest.approx(SHIFTS, abs=0.05)
    fit_panel = window.combined_fit_panel
    assert sorted(s["delta_iso"] for s in fit_panel.site_dicts) == pytest.approx(SHIFTS, abs=0.05)
    assert fit_panel.nucleus_combo.currentText() == "23Na"
    assert window._combined_fit_section.isExpanded()
    assert "23Na is quadrupolar" in panel.status_log.toPlainText()
    assert "CSA, Quad & J-coupling" in window.statusBar().currentMessage()


def test_fit_exports_selected_rows_to_simulate(fitted):
    window, panel = fitted
    table = panel.results_peaks_table
    table.selectionModel().select(table.model().index(1, 0),
                                  QItemSelectionModel.SelectionFlag.Select
                                  | QItemSelectionModel.SelectionFlag.Rows)
    table.selectionModel().select(table.model().index(4, 0),
                                  QItemSelectionModel.SelectionFlag.Select
                                  | QItemSelectionModel.SelectionFlag.Rows)
    assert panel.selected_peak_indices() == [1, 4]
    panel.send_sites_row.destination_combo.setCurrentIndex(
        panel.send_sites_row.destination_combo.findData("simulate"))
    sites = panel.export_sites("selected")
    assert len(sites) == 2
    expected = [panel.result["peaks"][i]["fitted_shift"] for i in (1, 4)]
    assert [s["delta_iso"] for s in window.combined_sim_panel.site_dicts] == pytest.approx(expected)
    assert "2 of 7 peaks exported" in panel.status_log.toPlainText()
