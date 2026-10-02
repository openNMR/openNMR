"""Fit > pNMR (CIF bond pathways): gui/pnmr_fit_panel.py.

The structural model comes from Simulate > pNMR (CIF) (one copy of it); the
spectrum fitted is that panel's own simulation of the rock-salt Na(Cl,Br)
cell (test_pnmr_simulator.py), so every expected number -- seven peaks at
-60..+60 ppm with binomial(6, 1/2) areas -- is known exactly. These tests
check the wiring: the model, the fit, the tables, hover and click, the
overlay surviving restyling, refit, picking on the plot, saving/keeping.
"""
import ast
import csv
import json
import math

import numpy as np
import pytest
from matplotlib.backend_bases import MouseEvent
from PySide6.QtWidgets import QMessageBox

pytest.importorskip("pymatgen")

from opennmr.gui.main_window import MainWindow  # noqa: E402
from opennmr.gui.pnmr_fit_panel import PnmrFitPanel, parse_shifts  # noqa: E402
from opennmr.notebook import pnmr_simulator as pnmr  # noqa: E402

from test_gui import qapp  # noqa: F401,E402 -- fixture
from test_pnmr_simulator import ROCKSALT_CIF, ROCKSALT_SHELLS, ROCKSALT_SHIFTS  # noqa: E402

BINOMIAL_PCT = 100 * np.array([1, 6, 15, 20, 15, 6, 1]) / 64.0
SHIFTS = [-60.0, -40.0, -20.0, 0.0, 20.0, 40.0, 60.0]


@pytest.fixture(autouse=True)
def _no_modal_dialogs(monkeypatch):
    shown = []
    for name in ("warning", "information", "critical"):
        monkeypatch.setattr(QMessageBox, name,
                            lambda *a, _n=name, **k: shown.append((_n, a[2] if len(a) > 2 else "")))
    return shown


@pytest.fixture
def settings_file(tmp_path):
    cif = tmp_path / "rocksalt.cif"
    cif.write_text(ROCKSALT_CIF)
    path = tmp_path / "rocksalt_parameters.json"
    pnmr.save_simulation_parameters(
        str(path), cif_filename=str(cif), atom_of_interest="Na+", shell_bins=ROCKSALT_SHELLS,
        shift_dict=ROCKSALT_SHIFTS, nuclide="23Na", B0_tesla=9.4, broadening=3.0,
        n_points=4096, xlims=[-100, 100])
    return str(path)


def _window_with_model(settings_file):
    window = MainWindow()
    assert window.pnmr_panel.load_settings(settings_file)
    return window, window.pnmr_fit_panel


@pytest.fixture
def fitted(qapp, settings_file):  # noqa: F811
    """A window whose Open experiments holds the simulated rock-salt
    spectrum (selected), fitted by the pNMR fit panel."""
    window, panel = _window_with_model(settings_file)
    assert window.pnmr_panel.enumerate_sites()
    window._add_derived_spectrum(window.pnmr_panel._spectrum, "rock salt (simulated)")
    assert window._selected_index() == 0
    assert panel.run_fit()
    return window, panel


def _mouse(canvas, name, x_ppm, button=None, dx=0.0):
    """A matplotlib event at x_ppm on the fit's axes, through the canvas's
    own callback registry (as a real mouse event is)."""
    ax = canvas.ax
    y_mid = float(np.mean(ax.get_ylim()))
    px, py = ax.transData.transform((x_ppm, y_mid))
    event = MouseEvent(name, canvas.canvas, px + dx, py, button=button)
    canvas.canvas.callbacks.process(name, event)
    return event


# ---------------------------------------------------------------------------
# Where it lives, and the model it uses
# ---------------------------------------------------------------------------
def test_panel_is_a_section_of_the_fit_tile(qapp):  # noqa: F811
    window = MainWindow()
    panel = window.pnmr_fit_panel
    assert isinstance(panel, PnmrFitPanel)
    assert window._pnmr_fit_section.isAncestorOf(panel)
    assert window._fit_panels()["pnmr"] is panel
    assert window._FIT_LABELS["pnmr"] == "pNMR (CIF)"
    assert window.keep_fit_buttons.get("pnmr") is not None


def test_without_a_model_it_says_what_is_missing(qapp, _no_modal_dialogs):  # noqa: F811
    window = MainWindow()
    panel = window.pnmr_fit_panel
    assert not panel.refresh_model()
    assert "no CIF loaded" in panel.model_summary_label.text()
    assert not panel.run_fit()
    assert any("not ready" in text for _, text in _no_modal_dialogs)


def test_the_model_is_simulate_pnmrs_own(qapp, settings_file):  # noqa: F811
    window, panel = _window_with_model(settings_file)
    assert panel.refresh_model()
    model = panel.model_kwargs()
    full = window.pnmr_panel.generate_kwargs()
    assert set(model) == set(window.pnmr_panel.FIT_MODEL_KEYS)
    assert all(model[k] == full[k] for k in model)
    assert model["shift_dict"]["NN"] == {"Cl-": 10.0, "Br-": -10.0, **{
        k: v for k, v in model["shift_dict"]["NN"].items() if k not in ("Cl-", "Br-")}}
    summary = panel.model_summary_label.text()
    assert "rocksalt.cif" in summary and "Na+" in summary and "Cl- +10" in summary
    # Automatic tolerance: the smallest non-zero pathway shift.
    assert panel.tolerance_spin.value() == pytest.approx(10.0)


def test_changing_the_sites_after_initialise_makes_the_model_stale(qapp, settings_file):  # noqa: F811
    window, panel = _window_with_model(settings_file)
    window.pnmr_panel.max_r_spin.setValue(window.pnmr_panel.max_r_spin.value() + 1.0)
    assert not panel.refresh_model()
    assert "Initialise" in panel.model_summary_label.text()


def test_set_up_goes_to_simulate_pnmr(qapp):  # noqa: F811
    window = MainWindow()
    window.pnmr_fit_panel.setup_button.click()
    assert window.side_tabs.tabText(window.side_tabs.currentIndex()) == "Simulate"
    assert window._pnmr_sim_section.isExpanded()


# ---------------------------------------------------------------------------
# The fit, its tables and its plot
# ---------------------------------------------------------------------------
def test_fit_finds_the_seven_binomial_peaks(fitted):
    window, panel = fitted
    peaks = panel.result["peaks"]
    assert [p["fitted_shift"] for p in peaks] == pytest.approx(SHIFTS)
    assert [p["integral_pct"] for p in peaks] == pytest.approx(BINOMIAL_PCT, rel=1e-5)
    assert panel.result["rel_rmse"] < 1e-6
    table = panel.results_peaks_table
    assert table.rowCount() == 7
    assert table.item(1, 0).text() == "-40.00 (fixed)"
    assert table.item(3, 3).text().startswith("31.2")
    assert table.item(1, 5).text() == "Na1: NN:5Br-+1Cl- NNN:12Na+ (100%)"
    assert "7 peak(s) from 28 predicted environment(s)" in panel.results_label.text()


def test_fit_is_drawn_on_the_fit_view_and_saved_with_the_spectrum(fitted):
    window, panel = fitted
    assert window.canvas_tabs.currentWidget() is window.fit_canvas
    assert panel.is_showing()
    assert window.fit_canvas.fit_component_labels() == [f"{s:.1f} ppm" for s in SHIFTS]
    fits = window.session.spectra[0].metadata.raw["fits"]
    assert fits["pnmr"]["label"] == "pNMR (CIF)"
    assert "Peak 2 (-40.00 ppm): Na1: NN:5Br-+1Cl- NNN:12Na+ 100%" in fits["pnmr"]["report"]


def test_peak_names_follow_the_chosen_mode_and_stay_unique(fitted):
    window, panel = fitted
    panel.peak_names_combo.setCurrentIndex(panel.peak_names_combo.findData("environment"))
    labels = window.fit_canvas.fit_component_labels()
    assert labels[0] == "Na1: NN:6Br- NNN:12Na+"
    assert len(set(labels)) == 7
    panel.peak_names_combo.setCurrentIndex(panel.peak_names_combo.findData("number"))
    assert window.fit_canvas.fit_component_labels()[2] == "Peak 3"
    from opennmr.gui.pnmr_fit_panel import _unique
    assert _unique(["a", "b", "a", "a"]) == ["a", "b", "a (2)", "a (3)"]


# ---------------------------------------------------------------------------
# Hover and click
# ---------------------------------------------------------------------------
def test_hovering_lists_the_environments_under_the_cursor(fitted):
    window, panel = fitted
    _mouse(window.fit_canvas, "motion_notify_event", -40.0)
    assert panel.hover_text.splitlines()[0] == "δ = -40.0 ppm"
    assert "100.0%  Na1  NN:5Br-+1Cl- NNN:12Na+" in panel.hover_text
    assert "NN:5Br-+1Cl-" in panel.hover_label.text()
    # Between peaks, on baseline: nothing is named.
    _mouse(window.fit_canvas, "motion_notify_event", -90.0)
    assert panel.hover_text == "No fitted peak here."


def test_hover_is_off_when_unticked(fitted):
    window, panel = fitted
    panel.hover_check.setChecked(False)
    panel.hover_text = ""
    _mouse(window.fit_canvas, "motion_notify_event", -40.0)
    assert panel.hover_text == ""


def test_clicking_a_peak_selects_it_and_lists_its_environments(fitted):
    window, panel = fitted
    _mouse(window.fit_canvas, "button_press_event", 19.0, button=1)
    _mouse(window.fit_canvas, "button_release_event", 19.0, button=1)
    assert panel.selected_peak == 4
    assert panel.results_peaks_table.selectionModel().selectedRows()[0].row() == 4
    assert "Peak 5" in panel.results_env_title.text()
    env = panel.results_env_table
    assert env.rowCount() == 1
    assert env.item(0, 0).text() == "Na1 ×4"
    assert env.item(0, 1).text() == "NN:2Br-+4Cl- NNN:12Na+"
    assert float(env.item(0, 3).text()) == pytest.approx(100.0)
    assert float(env.item(0, 4).text()) == pytest.approx(BINOMIAL_PCT[4], abs=0.01)
    # The selected peak is outlined on the plot.
    assert any(a.get_label() == "_selected_peak" for a in window.fit_canvas.ax.lines)


def test_a_drag_is_a_pan_not_a_click(fitted):
    window, panel = fitted
    _mouse(window.fit_canvas, "button_press_event", 19.0, button=1)
    _mouse(window.fit_canvas, "button_release_event", 19.0, button=1, dx=40.0)
    assert panel.selected_peak is None


def test_selecting_a_table_row_selects_the_peak(fitted):
    window, panel = fitted
    panel.results_peaks_table.selectRow(2)
    assert panel.selected_peak == 2
    assert panel.results_env_table.item(0, 1).text() == "NN:4Br-+2Cl- NNN:12Na+"
    outlines = [a for a in window.fit_canvas.ax.lines if a.get_label() == "_selected_peak"]
    assert len(outlines) == 1
    assert np.argmax(outlines[0].get_ydata()) == np.argmax(panel.result["components"][2])
    panel.results_peaks_table.selectRow(5)
    outlines = [a for a in window.fit_canvas.ax.lines if a.get_label() == "_selected_peak"]
    assert len(outlines) == 1           # the previous outline was replaced, not added to


# ---------------------------------------------------------------------------
# The overlay: labels and sticks, surviving restyling, dropped with the fit
# ---------------------------------------------------------------------------
def test_labels_and_sticks_are_drawn_and_survive_a_component_restyle(fitted):
    window, panel = fitted
    canvas = window.fit_canvas
    panel.max_labels_spin.setValue(3)
    panel.annotate_check.setChecked(True)
    panel.sticks_check.setChecked(True)
    texts = [t.get_text() for t in canvas.ax.texts if t.get_text().startswith("Shift =")]
    assert len(texts) == 3
    assert any(t.startswith("Shift = 0.0 ppm") for t in texts)        # the largest peak
    sticks = [c for c in canvas.ax.collections if len(getattr(c, "get_segments", lambda: [])())
              == 7]
    assert sticks
    canvas.set_fit_component_style(fill=False)
    assert len([t for t in canvas.ax.texts if t.get_text().startswith("Shift =")]) == 3
    assert panel.is_showing()


def test_another_plot_drops_the_overlay_and_hover_stops(fitted):
    window, panel = fitted
    canvas = window.fit_canvas
    canvas.show_spectrum(window.session.spectra[0])
    assert canvas.slice_fit_overlay() is None
    assert not panel.is_showing()
    panel.hover_text = ""
    _mouse(canvas, "motion_notify_event", -40.0)
    assert panel.hover_text == ""
    # A plot option change does not take the view back over.
    panel.annotate_check.setChecked(True)
    assert canvas.slice_fit_overlay() is None


def test_a_failing_overlay_never_takes_the_fit_with_it(qapp):  # noqa: F811
    window = MainWindow()
    canvas = window.fit_canvas
    x = np.linspace(-10, 10, 201)
    peak = np.exp(-x ** 2)

    def broken(ax):
        raise RuntimeError("boom")

    with pytest.warns(RuntimeWarning, match="boom"):
        canvas.show_slice_fit(x, peak, peak, components=[peak, np.zeros_like(x)],
                              overlay=broken)
    assert canvas.shown_fit() is not None
    assert canvas.slice_fit_overlay() is broken


# ---------------------------------------------------------------------------
# Simplify and refit; picking shifts on the plot
# ---------------------------------------------------------------------------
def test_refit_onto_chosen_shifts(fitted):
    window, panel = fitted
    assert panel.refit_window_spin.value() == pytest.approx(10.0)   # the tolerance
    panel.merge_mode_combo.setCurrentIndex(panel.merge_mode_combo.findData("picks"))
    panel.picks_edit.setText("-50, 0, 50")
    assert panel.run_refit()
    peaks = panel.result["peaks"]
    assert len(peaks) == 3 and panel.result["stage"] == "refit"
    middle = sorted(m["fraction_in_group"] for m in peaks[1]["members"])
    assert middle == pytest.approx([15 / 50, 15 / 50, 20 / 50])
    assert "Simplified refit" in panel.results_label.text()
    assert window.fit_canvas.fit_component_labels() == panel.peak_names()


def test_refit_by_merging_low_area_peaks(fitted):
    window, panel = fitted
    # The 1/64 outer peaks fall below 10% of the 20/64 centre; their
    # neighbours are 20 ppm away, beyond the 10 ppm grouping tolerance.
    panel.merge_relative_spin.setValue(10.0)
    assert panel.run_refit()
    assert len(panel.result["peaks"]) == 7              # nothing near enough: left alone
    panel.merge_tol_auto_check.setChecked(False)
    panel.merge_tol_spin.setValue(25.0)
    assert panel.run_refit()
    assert len(panel.result["peaks"]) == 5
    assert "(2 merge(s))" in panel.results_label.text()
    outer = panel.result["peaks"][0]
    assert sorted(m["label"] for m in outer["members"]) == [
        "NN:5Br-+1Cl- NNN:12Na+", "NN:6Br- NNN:12Na+"]


def test_refit_needs_a_maximal_fit_first(qapp, settings_file, _no_modal_dialogs):  # noqa: F811
    window, panel = _window_with_model(settings_file)
    assert not panel.refit_button.isEnabled()
    assert not panel.run_refit()


def test_picking_on_the_plot_adds_shifts(fitted):
    window, panel = fitted
    panel.merge_mode_combo.setCurrentIndex(panel.merge_mode_combo.findData("picks"))
    panel.pick_button.setChecked(True)
    assert window.fit_canvas._pick_mode
    window.fit_canvas.peakPicked.emit(12.345)
    window.fit_canvas.peakPicked.emit(-3.0)
    assert parse_shifts(panel.picks_edit.text()) == pytest.approx([12.35, -3.0])
    # A click while picking picks; it does not select a peak.
    _mouse(window.fit_canvas, "button_press_event", 19.0, button=1)
    _mouse(window.fit_canvas, "button_release_event", 19.0, button=1)
    assert panel.selected_peak is None
    window.fit_canvas.set_pick_mode(False)
    assert not panel.pick_button.isChecked()
    window.fit_canvas.peakPicked.emit(99.0)
    assert "99" not in panel.picks_edit.text()


def test_parse_shifts():
    assert parse_shifts("120, 45.5 -10;3") == [120.0, 45.5, -10.0, 3.0]
    assert parse_shifts("  ") == []
    with pytest.raises(ValueError, match="'abc'"):
        parse_shifts("1, abc")


# ---------------------------------------------------------------------------
# Saving, keeping, restoring, exporting
# ---------------------------------------------------------------------------
def test_result_round_trips_through_its_saved_state(fitted):
    window, panel = fitted
    state = panel.result_state()
    json.dumps(state)                                   # plain JSON
    table_before = [[panel.results_peaks_table.item(r, c).text() for c in range(6)]
                    for r in range(7)]
    panel.clear_result()
    assert panel.results_peaks_table.rowCount() == 0
    panel.restore_result(state, window.session.spectra[0])
    assert [[panel.results_peaks_table.item(r, c).text() for c in range(6)]
            for r in range(7)] == table_before
    assert panel.environments_at(-40.0).iloc[0]["environment"] == "NN:5Br-+1Cl- NNN:12Na+"
    assert not panel.refit_button.isEnabled()          # needs the fit run again


def test_keep_this_fit(fitted):
    window, panel = fitted
    model = window._keep_panel_fit("pnmr", "rock salt 7 peaks")
    assert model is not None
    kept = window._fit_models(0)
    assert [m["name"] for m in kept] == ["rock salt 7 peaks"]
    assert kept[0]["kind"] == "pnmr"


def test_save_environment_report(fitted, tmp_path):
    window, panel = fitted
    path = tmp_path / "envs.csv"
    panel.save_report(str(path))
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 7
    assert rows[3]["environment"] == "NN:3Br-+3Cl- NNN:12Na+"
    assert float(rows[3]["pct_of_total"]) == pytest.approx(BINOMIAL_PCT[3], rel=1e-5)
    assert rows[3]["copies"] == "4" and rows[3]["site"] == "Na1"


def test_notebook_code_is_valid_python_naming_the_same_model(fitted):
    window, panel = fitted
    code = panel.notebook_code()
    ast.parse(code)
    assert "fit_predicted_spectrum(" in code and "rocksalt.cif" in code
    assert "'Cl-': 10.0" in code and "environments_at_shift(fit" in code


def test_a_spectrum_of_another_nucleus_is_flagged(fitted):
    window, panel = fitted
    window.session.spectra[0].metadata.nucleus = "7Li"
    assert panel.run_fit()
    assert "the spectrum is 7Li but the model observes Na+" in panel.status_log.toPlainText()


def test_fit_only_the_integration_regions(fitted, _no_modal_dialogs):
    window, panel = fitted
    panel.use_regions_check.setChecked(True)
    assert not panel.run_fit()          # no regions yet
    assert any("no regions" in text for _, text in _no_modal_dialogs)
    window.region_table.setRowCount(1)
    from PySide6.QtWidgets import QTableWidgetItem
    window.region_table.setItem(0, 0, QTableWidgetItem("-30"))
    window.region_table.setItem(0, 1, QTableWidgetItem("30"))
    assert panel.run_fit()
    assert [p["fitted_shift"] for p in panel.result["peaks"]] == pytest.approx([-20, 0, 20])
    assert any(d["reason"] == "outside fit_regions" for d in panel.result["dropped"])
    assert panel.result["x"].min() >= -30 - 0.1 and panel.result["x"].max() <= 30 + 0.1
