"""Round of 2026-10-08 (explicit user requests):

1. the Combined-Hamiltonian fitted values as a table (row per site, column
   per parameter), not a list;
2. "Put back in its panel" -> "Re-plot and show “What's next?”";
3. "Save the fitted values" (values and errors as readable JSON) beside
   each fit, for kept fits, and with a saved openNMR state;
4. the crosshair runs through every row of a stacked Compare;
5. kept fits stay listed after Compare Ticked (selecting a fit's Open
   experiment no longer empties the list); "Send to Open Experiments";
6. Residuals / Components of redraw a comparison at once;
7. Settings ▸ Saving: the default folder for openNMR states.
Synthetic data only."""
import json
import os

import numpy as np
import pytest
from matplotlib.backend_bases import MouseEvent

from opennmr.io import state as io_state
from opennmr.notebook import fit_values as fv

from test_gui_round4_fits import _window_with_kept_fits, qapp  # noqa: F401 -- qapp is a fixture


@pytest.fixture
def window(qapp, tmp_path, monkeypatch):
    from opennmr.gui.main_window import MainWindow
    monkeypatch.setattr(MainWindow, "_FIGURE_STYLES_PATH", str(tmp_path / "figure_styles.json"))
    monkeypatch.setenv("OPENNMR_STATES_DIR", str(tmp_path / "states"))
    yield MainWindow()
    io_state.set_default_state_root(None)


# ============================================================
# Fitted values: notebook
# ============================================================

@pytest.mark.parametrize("text, expected", [
    ("12.01 ± 0.03", (12.01, 0.03, False)),
    ("-1.5e3 ± 2e1", (-1500.0, 20.0, False)),
    ("0.7 (fixed)", (0.7, None, True)),
    ("1230 ± 150", (1230.0, 150.0, False)),
    ("4.2\n± 0.1", (4.2, 0.1, False)),
    ("−3.5 +/- 0.2", (-3.5, 0.2, False)),
    ("42", (42.0, None, False)),
])
def test_value_and_error_are_read_from_the_table_text(text, expected):
    got = fv.parse_value_error(text)
    value, error, fixed = expected
    assert got["value"] == pytest.approx(value) and got["fixed"] is fixed
    assert got["error"] == (None if error is None else pytest.approx(error))


def test_text_that_is_not_a_number_is_kept_as_text():
    got = fv.parse_value_error("Al(IV)")
    assert got["value"] is None and got["text"] == "Al(IV)"


def _combined_result():
    return {
        "parameters": [
            {"label": "Al1", "name": "delta_iso", "value": 12.0123456, "error": 0.0312, "free": True},
            {"label": "Al1", "name": "C_Q", "value": 4.2, "error": 0.02, "free": True},
            {"label": "Al2", "name": "delta_iso", "value": 60.5, "error": 0.1, "free": True},
            {"label": "Al2", "name": "eta_Q", "value": 0.7, "error": None, "free": False},
            {"label": "", "name": "mas_Hz", "value": 20000.0, "error": None, "free": False},
        ],
        "derived": {"Al1": {"share_%": 66.7, "share_%_err": 1.0, "P_Q": 4.29, "P_Q_err": 0.02},
                    "Al2": {"share_%": 33.3, "share_%_err": 1.0}},
        "quality": {"r2": 0.9991, "reduced_chi2": 1.04, "rmse": np.float64(0.01)},
    }


def test_a_fit_result_gives_a_site_by_parameter_table_at_full_precision(tmp_path):
    record = fv.fitted_values_record(fit="Combined Hamiltonian", spectrum="27Al",
                                     result=_combined_result())
    v = record["values"]
    assert v["Al1"]["δiso (ppm)"] == {"value": 12.0123456, "error": 0.0312, "fixed": False}
    assert v["Al2"]["η_Q"] == {"value": 0.7, "error": None, "fixed": True}
    assert v["(spectrum)"]["MAS rate (Hz)"]["fixed"] is True
    assert v["Al1"]["Share (%)"]["value"] == 66.7 and v["Al1"]["P_Q (MHz)"]["error"] == 0.02
    assert record["quality"]["r2"] == 0.9991
    path = fv.save_fitted_values(record, tmp_path / "values")          # .json added
    assert path.endswith("values.json")
    text = open(path, encoding="utf-8").read()
    assert "δiso (ppm)" in text and "\n  " in text                       # readable, indented
    back = fv.load_fitted_values(path)
    assert back["values"]["Al1"]["C_Q (MHz)"]["value"] == 4.2
    # A result straight in, as a notebook would.
    path2 = fv.save_fitted_values(_combined_result(), tmp_path / "direct.json", fit="x")
    assert fv.load_fitted_values(path2)["fit"] == "x"
    (tmp_path / "other.json").write_text("{}")
    with pytest.raises(ValueError):
        fv.load_fitted_values(tmp_path / "other.json")


def test_the_table_as_shown_is_kept_and_exact_values_win():
    record = fv.fitted_values_record(
        columns=["δiso (ppm)", "FWHM (ppm)"],
        rows=[("Al1", ["12.01 ± 0.03", "2.0 (fixed)"]), ("Al2", ["60.5 ± 0.1", "—"])],
        result=_combined_result())
    assert record["as shown"]["Al1"]["δiso (ppm)"] == "12.01 ± 0.03"
    assert record["values"]["Al1"]["δiso (ppm)"]["value"] == 12.0123456       # exact
    assert record["values"]["Al1"]["FWHM (ppm)"] == {"value": 2.0, "error": None, "fixed": True}
    assert "FWHM (ppm)" not in record["values"]["Al2"]                       # blank cell left out


def test_notebook_exports_the_module():
    import opennmr.notebook as nb
    assert nb.fit_values is fv
    assert "fit_values" in nb.__all__ and "save_fitted_values" in nb.__all__


# ============================================================
# 1. The Combined fit's values as a table
# ============================================================

def test_combined_results_are_a_table_with_a_row_per_site(window):
    panel = window.combined_fit_panel
    panel.result = {**_combined_result(), "log": "Combined-Hamiltonian fit: test", "warnings": [],
                    "quality": {"r2": 0.99, "rmse": 0.1, "reduced_chi2": 1.0}, "advice": []}
    panel._report()
    t = panel.results_table
    rows = [t.verticalHeaderItem(r).text() for r in range(t.rowCount())]
    cols = [t.horizontalHeaderItem(c).text() for c in range(t.columnCount())]
    assert rows == ["Al1", "Al2", "(spectrum)"]
    assert cols[:2] == ["δiso (ppm)", "C_Q (MHz)"]
    assert {"η_Q", "MAS rate (Hz)", "Share (%)", "P_Q (MHz)"} <= set(cols)
    assert "(fixed)" in t.item(rows.index("Al2"), cols.index("η_Q")).text()
    assert t.item(rows.index("Al2"), cols.index("C_Q (MHz)")).text() == "—"   # not this site's
    # Saved from the panel: the same numbers, exact.
    record = window._fit_values_record("combined")
    assert record["values"]["Al1"]["δiso (ppm)"]["value"] == 12.0123456
    assert record["as shown"]["Al2"]["η_Q"].endswith("(fixed)")


# ============================================================
# 2, 3. Re-plot and show "What's next?"; save a fit's values
# ============================================================

def test_a_kept_fit_is_re_plotted_with_whats_next_and_its_values_saved(window, tmp_path):
    from opennmr.gui.fit_models_panel import RESTORE_TEXT
    from test_state_and_fit_models import _czjzek_glass
    fp = window.fit_models_panel
    assert fp.restore_button.text() == RESTORE_TEXT == "Re-plot and Show “What's Next?”"
    index = _czjzek_glass(window)
    panel = window.czjzek_fit_panel
    panel.sites_table.item(0, 1).setText("40")
    panel.sites_table.item(0, 2).setText("2")
    panel.grid_cq_spin.setValue(32)
    panel.fit()
    # Beside Keep This Fit…, every fit panel can save its values.
    assert set(window.save_fit_values_buttons) == set(window.keep_fit_buttons)
    path = window._save_fit_values("czjzek", path=str(tmp_path / "czjzek.json"))
    record = fv.load_fitted_values(path)
    assert record["fit"] == "Czjzek" and record["spectrum"] == window.session.label(index)
    assert record["values"] and record["columns"]
    first = next(iter(record["values"].values()))
    assert any(isinstance(c.get("value"), float) for c in first.values())
    model = window._keep_shown_fit(name="1 site")
    assert model.get("values_table", {}).get("columns")
    panel.clear_result()
    window._show_side_tab("Simulate")
    window._restore_fit_model(index, model["id"])
    assert window.side_tabs.tabText(window.side_tabs.currentIndex()) == "Fit"
    assert window._czjzek_fit_section.isExpanded()
    assert panel.result is not None
    # The kept fit's values, from Kept fit models.
    kept = window._save_fit_model_values(index, [model["id"]], path=str(tmp_path / "kept.json"))
    kept = fv.load_fitted_values(kept)
    assert kept["fit"].endswith("1 site") and kept["values"] and "r2" in kept["quality"]


# ============================================================
# 5, 6. Kept fits stay; Send to Open Experiments; live residuals
# ============================================================

def test_kept_fits_stay_listed_when_their_open_experiment_is_selected(window):
    index, ids = _window_with_kept_fits(window)
    window._select_single_row(index)
    fp = window.fit_models_panel
    fp.refresh()
    fp.set_ticked(ids)
    fp.layout_combo.setCurrentIndex(fp.layout_combo.findData("stack"))
    fp.compare()
    experiment = window._find_fit_model_experiment(window.session.label(index),
                                                   window._fit_models(index)[0])
    assert experiment is not None
    window._select_single_row(experiment)      # as clicking it in Compare does
    fp.refresh()
    assert fp.table.rowCount() == 2
    assert window.session.label(index) in fp.owner_label.text()
    assert window._fit_models_index() == index


def test_send_to_open_experiments_button(window):
    index, ids = _window_with_kept_fits(window)
    window._select_single_row(index)
    fp = window.fit_models_panel
    fp.refresh()
    assert fp.to_open_button.isVisibleTo(fp) or not fp.to_open_button.isHidden()
    n = len(window.session)
    fp.set_ticked(ids)
    fp.to_open_button.click()
    assert len(window.session) == n + 2
    fp.refresh()
    assert fp.table.rowCount() == 2                 # still there
    fp.to_open_button.click()                       # reused, not added again
    assert len(window.session) == n + 2


def test_residuals_switch_redraws_the_comparison(window):
    index, ids = _window_with_kept_fits(window)
    window._select_single_row(index)
    fp = window.fit_models_panel
    fp.refresh()
    fp.set_ticked(ids)
    fp.layout_combo.setCurrentIndex(fp.layout_combo.findData("stack"))
    fp.residuals_check.setChecked(True)
    fp.compare()

    def residual_lines():
        return [l for ax in window.compare_canvas._compare_axes for l in ax.lines
                if l.get_label().endswith("residual") and l.get_visible()]
    assert residual_lines()
    fp.residuals_check.setChecked(False)            # no Compare Ticked needed
    assert not residual_lines()
    fp.residuals_check.setChecked(True)
    assert residual_lines()


# ============================================================
# 4. Crosshair through every row of a stacked Compare
# ============================================================

def test_crosshair_runs_through_every_stacked_row(window, qapp):
    index, ids = _window_with_kept_fits(window)
    window._select_single_row(index)
    fp = window.fit_models_panel
    fp.refresh()
    fp.set_ticked(ids)
    fp.layout_combo.setCurrentIndex(fp.layout_combo.findData("stack"))
    fp.compare()
    canvas = window.compare_canvas
    canvas.canvas.draw()
    rows = canvas._compare_axes
    assert len(rows) == 2
    canvas.set_crosshair_enabled(True)
    canvas._remove_crosshair()
    top = rows[0]
    x, y = top.transData.transform((5.0, np.mean(top.get_ylim())))
    event = MouseEvent("motion_notify_event", canvas.canvas, x, y)
    for _ in range(2):          # the first move creates it, as on screen
        canvas.canvas.callbacks.process("motion_notify_event", event)
    cross = canvas._crosshair
    assert cross is not None
    vertical = {ax: line.get_visible() for ax, line in zip(cross._vline_axes, cross._vlines)}
    assert all(vertical.get(ax) for ax in rows)       # through both rows
    horizontal = {ax: line.get_visible() for ax, line in zip(cross._hline_axes, cross._hlines)}
    assert horizontal[top] and not horizontal[rows[1]]   # only the row under the cursor


# ============================================================
# 7. Settings ▸ Saving; fitted values with a state
# ============================================================

def test_settings_choose_where_states_are_saved(window, tmp_path):
    from opennmr.gui.main_window import MainWindow
    from opennmr.gui.state_dialog import SaveStateDialog
    chosen = str(tmp_path / "my states")
    assert window.state_folder_edit.text() == io_state.builtin_state_root()
    window.state_folder_edit.setText(chosen)
    window.state_folder_edit.editingFinished.emit()
    assert io_state.default_state_root() == chosen
    with open(MainWindow._UI_SETTINGS_PATH, encoding="utf-8") as handle:
        assert json.load(handle)["state_folder"] == chosen
    dialog = SaveStateDialog(window, [], default_name="run 1")
    assert dialog.folder_edit.text().startswith(chosen)
    io_state.set_default_state_root(None)            # as a fresh launch would start
    window2 = MainWindow()
    assert io_state.default_state_root() == chosen == window2.state_folder_edit.text()
    window2.state_folder_reset_button.click()
    assert io_state.default_state_root() == io_state.builtin_state_root()
    with open(MainWindow._UI_SETTINGS_PATH, encoding="utf-8") as handle:
        assert "state_folder" not in json.load(handle)


def test_a_file_is_refused_as_the_state_folder(window, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: None)
    a_file = tmp_path / "file.txt"
    a_file.write_text("x")
    window._set_state_folder(str(a_file))
    assert io_state.default_state_root() != str(a_file)


def test_a_saved_state_carries_the_fitted_values(window, tmp_path):
    from opennmr.gui.main_window import FIT_VALUES_FILE
    index, ids = _window_with_kept_fits(window)
    folder = window._save_state(str(tmp_path / "s.opennmr"), "s", [index], fit_values=True)
    path = os.path.join(folder, FIT_VALUES_FILE)
    record = fv.load_fitted_values(path)
    kept = record["spectra"][window.session.label(index)]["kept fits"]
    assert set(kept) == {"1 site", "2 sites"}
    assert kept["2 sites"]["quality"]["k"] == 6
    folder2 = window._save_state(str(tmp_path / "t.opennmr"), "t", [index])
    assert not os.path.exists(os.path.join(folder2, FIT_VALUES_FILE))
