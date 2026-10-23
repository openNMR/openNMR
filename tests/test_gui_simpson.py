"""Simulate > SIMPSON, Fit > SIMPSON spectrum, File > Export Data (Ctrl/Cmd+Shift+X),
and the Welcome wizard's extra settings (explicit user request, 2026-10-23)."""
import numpy as np
import pytest
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import QFileDialog, QMessageBox

from opennmr.gui.export_dialog import ExportDialog, export_many
from opennmr.gui.main_window import MainWindow
from opennmr.gui.welcome_wizard import WelcomeWizard
from opennmr.io import export as io_export

import simpson_files as sf
from test_gui import qapp  # noqa: F401 -- fixture


@pytest.fixture(autouse=True)
def _no_modal_dialogs(monkeypatch):
    shown = []
    for name in ("warning", "information", "critical"):
        monkeypatch.setattr(QMessageBox, name,
                            lambda *a, _n=name, **k: shown.append((_n, a[2] if len(a) > 2 else "")))
    return shown


def _window_with(tmp_path, *names, kind="fid"):
    window = MainWindow()
    panel = window.simpson_panel
    paths = [(sf.write_fid if kind == "fid" else sf.write_spe)(tmp_path / n) for n in names]
    panel.add_paths(paths)
    return window, panel, paths


# ---- sections ----------------------------------------------------------------

def test_sections_exist_and_can_be_hidden_and_restored(qapp):  # noqa: F811
    window = MainWindow()
    assert "simpson" in window.OPTIONAL_SECTIONS
    sim, fit = window._simpson_sim_section, window._simpson_fit_section
    assert not sim.isHidden() and not fit.isHidden()
    window.section_shown_checkboxes["simpson"].setChecked(False)
    assert sim.isHidden() and fit.isHidden()
    window.section_shown_checkboxes["simpson"].setChecked(True)
    assert not sim.isHidden() and not fit.isHidden()


def test_simpson_is_in_the_welcome_wizard_layout_page(qapp):  # noqa: F811
    window = MainWindow()
    wizard = WelcomeWizard(window)
    assert "simpson" in wizard.section_checks
    wizard.section_checks["simpson"].setChecked(False)
    wizard.apply()
    assert window.section_shown_checkboxes["simpson"].isChecked() is False
    assert window._simpson_sim_section.isHidden()


# ---- spectra and FIDs --------------------------------------------------------

def test_files_become_open_experiments_that_process_like_any_1d(qapp, tmp_path):  # noqa: F811
    window, panel, paths = _window_with(tmp_path, "a.fid", "b.fid")
    panel.nuclide_combo.setCurrentText("31P")
    panel.field_spin.setValue(9.4)
    panel.file_list.selectAll()
    panel.open_button.click()
    assert len(window.session) == 2
    spectrum = window.session.spectra[0]
    assert spectrum.metadata.vendor == "simpson" and spectrum.ndim == 1
    assert spectrum.axes[0].ppm is not None
    assert window._fid_reprocessable(spectrum)             # window, zero-fill, phase act live


def test_plot_shows_the_spectrum_without_adding_it(qapp, tmp_path):  # noqa: F811
    window, panel, _ = _window_with(tmp_path, "a.spe", kind="spe")
    panel.plot_button.click()
    assert len(window.session) == 0
    assert panel.spectrum is not None
    assert window.fit_canvas.simulation_spectrum is panel.spectrum


def test_unreadable_file_is_reported_not_raised(qapp, tmp_path, _no_modal_dialogs):  # noqa: F811
    window = MainWindow()
    bad = tmp_path / "bad.dat"
    bad.write_text("not a simpson file\n")
    window.simpson_panel.add_paths([str(bad)])
    window.simpson_panel.plot_button.click()
    assert _no_modal_dialogs


# ---- points ------------------------------------------------------------------

def test_points_plot_uses_the_chosen_name_and_label(qapp, tmp_path):  # noqa: F811
    window = MainWindow()
    panel = window.simpson_panel
    paths = []
    for amp in (10, 20, 30):
        paths.append(sf.write_point_fid(tmp_path / f"b1_{amp}.fid",
                                        [amp, amp * 0.5, amp * 0.1, 1, 1]))
    panel.add_paths(paths)
    panel.n_points_spin.setValue(3)
    panel.x_name_edit.setText("B1")
    panel.x_label_edit.setText("RF amplitude (kHz)")
    panel.expressions_edit.setPlainText("A\nA + B - 0.3*C")
    panel.plot_points_button.click()
    ax = window.fit_canvas.figure.axes[0]
    assert "RF amplitude" in ax.get_xlabel()
    assert len(ax.lines) == 2
    assert np.allclose(ax.lines[1].get_ydata(), [10 + 5 - 0.3, 20 + 10 - 0.6, 30 + 15 - 0.9][0:0] or
                       [a + a * 0.5 - 0.3 * a * 0.1 for a in (10, 20, 30)])
    assert np.allclose(ax.lines[0].get_xdata(), [10, 20, 30])


def test_points_x_from_steps_and_typed(qapp, tmp_path):  # noqa: F811
    window = MainWindow()
    panel = window.simpson_panel
    panel.add_paths([sf.write_point_fid(tmp_path / f"p{k}.fid", [k, 0, 0]) for k in (1, 2, 3)])
    panel.x_source_combo.setCurrentIndex(panel.x_source_combo.findData("steps"))
    panel.x_start_spin.setValue(0.5)
    panel.x_step_spin.setValue(0.25)
    assert panel.x_values() == pytest.approx([0.5, 0.75, 1.0])
    panel.x_source_combo.setCurrentIndex(panel.x_source_combo.findData("typed"))
    panel.x_table.item(1, 1).setText("7,5")                 # a decimal comma is accepted
    assert panel.x_values()[1] == pytest.approx(7.5)
    panel.x_table.item(2, 1).setText("abc")
    with pytest.raises(ValueError):
        panel.x_values()


def test_a_bad_expression_is_reported(qapp, tmp_path, _no_modal_dialogs):  # noqa: F811
    window = MainWindow()
    panel = window.simpson_panel
    panel.add_paths([sf.write_point_fid(tmp_path / "p1.fid", [1, 2, 3])])
    panel.expressions_edit.setPlainText("A + Q")
    panel.plot_points_button.click()
    assert _no_modal_dialogs


def test_copied_notebook_code_reproduces_the_series(qapp, tmp_path):  # noqa: F811
    window = MainWindow()
    panel = window.simpson_panel
    panel.add_paths([sf.write_point_fid(tmp_path / f"q_{k}.fid", [k, 2 * k, 0]) for k in (1, 2)])
    panel.n_points_spin.setValue(2)
    panel.expressions_edit.setPlainText("A + B")
    code = panel.notebook_code()
    scope = {}
    import matplotlib
    matplotlib.use("Agg")
    exec(compile(code, "<copied>", "exec"), scope)        # noqa: S102 -- our own generated code
    assert np.allclose(scope["ax"].lines[0].get_ydata(), [3, 6])


# ---- send to Fit and fit -----------------------------------------------------

def test_send_to_fit_and_fit_an_experiment(qapp, tmp_path):  # noqa: F811
    window, panel, paths = _window_with(tmp_path, "sim.fid")
    panel.nuclide_combo.setCurrentText("31P")
    panel.field_spin.setValue(9.4)
    panel.plot_button.click()
    panel.send_button.click()
    fit_panel = window.simpson_fit_panel
    assert len(fit_panel.templates) == 1
    # the "experiment": the same lines, twice as strong, on a small offset
    experiment = sf.write_fid(tmp_path / "exp.fid", fid=2.0 * sf.make_fid())
    window._load(experiment, "simpson", extra_kwargs={"nucleus": "31P", "field": 9.4})
    window._select_single_row(len(window.session) - 1)
    fit_panel.float_shift_check.setChecked(False)
    result = fit_panel.fit()
    assert result is not None
    assert result["params"][0]["amplitude"] == pytest.approx(2.0, rel=0.02)
    assert window.fit_canvas._slice_fit_active
    assert fit_panel.results_table.rowCount() == 1
    assert "simpson" in (window.session.spectra[-1].metadata.raw.get("fits") or {})


def test_a_transformed_fid_opened_here_can_be_sent_to_fit(qapp, tmp_path):  # noqa: F811
    window, panel, paths = _window_with(tmp_path, "sim.fid")
    panel.file_list.selectAll()
    panel.open_button.click()
    window._select_single_row(0)
    panel.send_selected_button.click()
    assert len(window.simpson_fit_panel.templates) == 1


def test_fit_without_templates_says_so(qapp, _no_modal_dialogs):  # noqa: F811
    window = MainWindow()
    assert window.simpson_fit_panel.fit() is None
    assert _no_modal_dialogs


def test_simpson_fit_state_round_trips(qapp, tmp_path):  # noqa: F811
    window, panel, _ = _window_with(tmp_path, "sim.fid")
    fit_panel = window.simpson_fit_panel
    fit_panel.float_broadening_check.setChecked(True)
    fit_panel.shift_window_spin.setValue(2.5)
    state = fit_panel.get_state()
    fit_panel.float_broadening_check.setChecked(False)
    fit_panel.shift_window_spin.setValue(9.0)
    fit_panel.set_state(state)
    assert fit_panel.float_broadening_check.isChecked()
    assert fit_panel.shift_window_spin.value() == pytest.approx(2.5)
    assert window._fit_panels()["simpson"] is fit_panel
    assert "simpson" in window._simulation_panels()


# ---- export ------------------------------------------------------------------

def test_export_menu_item_and_shortcut(qapp):  # noqa: F811
    window = MainWindow()
    assert window.export_data_action.shortcut() == QKeySequence("Ctrl+Shift+X")
    assert window.export_data_action in window.menuBar().actions()[0].menu().actions()
    others = [a for a in window.findChildren(type(window.export_data_action))
              if a is not window.export_data_action and a.shortcut() == QKeySequence("Ctrl+Shift+X")]
    assert not others                                      # no clash with another action


def test_export_one_spectrum_to_each_format(qapp, tmp_path, monkeypatch):  # noqa: F811
    window, panel, _ = _window_with(tmp_path, "a.fid")
    panel.file_list.selectAll()
    panel.open_button.click()
    items = [(window.session.label(0), window.session.spectra[0])]
    for fmt, ext in (("csv", ".csv"), ("xy", ".xy"), ("jdx", ".jdx")):
        dialog = ExportDialog(items, checked=[0])
        dialog.format_combo.setCurrentIndex(dialog.format_combo.findData(fmt))
        target = str(tmp_path / f"out{ext}")
        monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, _t=target, **k: (_t, ""))
        dialog._on_export()
        assert dialog.written == [target]
    back = io_export.read_jcamp_spectrum(str(tmp_path / "out.jdx"))
    assert len(back["real"]) == window.session.spectra[0].data.size


def test_export_several_to_a_folder_without_clashes(qapp, tmp_path):  # noqa: F811
    window, panel, _ = _window_with(tmp_path, "a.fid", "b.fid")
    panel.file_list.selectAll()
    panel.open_button.click()
    items = [("same name", window.session.spectra[0]), ("same name", window.session.spectra[1])]
    written = export_many(items, str(tmp_path / "out"), "csv")
    assert len(set(written)) == 2 and all(p.endswith(".csv") for p in written)


def test_export_action_runs_the_dialog(qapp, tmp_path, monkeypatch):  # noqa: F811
    window, panel, _ = _window_with(tmp_path, "a.fid")
    panel.file_list.selectAll()
    panel.open_button.click()
    window._select_single_row(0)
    target = str(tmp_path / "viaaction.csv")
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (target, ""))
    monkeypatch.setattr(ExportDialog, "exec", lambda self: (self._on_export(), 1)[1])
    window.export_data_action.trigger()
    assert (tmp_path / "viaaction.csv").exists()


def test_export_with_nothing_open_says_so(qapp, _no_modal_dialogs):  # noqa: F811
    window = MainWindow()
    window.export_data_action.trigger()
    assert _no_modal_dialogs


# ---- wizard ------------------------------------------------------------------

def test_wizard_applies_the_extra_settings(qapp):  # noqa: F811
    window = MainWindow()
    wizard = WelcomeWizard(window)
    wizard.crosshair_width_spin.setValue(2.0)
    wizard._crosshair_color = "#aa0000"
    wizard.zoom_box_width_spin.setValue(3.5)
    wizard.auto_styles_check.setChecked(not window.auto_apply_type_styles_checkbox.isChecked())
    wizard.curve_top_check.setChecked(True)
    wizard.update_check.setChecked(True)
    wizard.apply()
    color, width = window._all_canvases()[0].crosshair_style()
    assert width == pytest.approx(2.0) and str(color).lower() in ("#aa0000", "#a00")
    assert window.zoom_box_width_spin.value() == pytest.approx(3.5)
    assert window.settings_curve_top_spine_checkbox.isChecked()
    assert window.updates_panel.startup_checkbox.isChecked()


def test_wizard_skip_leaves_the_extra_settings_alone(qapp):  # noqa: F811
    window = MainWindow()
    before = (window.zoom_box_width_spin.value(), window.crosshair_width_spin.value(),
              window.auto_apply_type_styles_checkbox.isChecked())
    wizard = WelcomeWizard(window)
    wizard.zoom_box_width_spin.setValue(wizard.zoom_box_width_spin.value() + 1)
    wizard.skip()
    assert before == (window.zoom_box_width_spin.value(), window.crosshair_width_spin.value(),
                      window.auto_apply_type_styles_checkbox.isChecked())


# ---- other SIMPSON file shapes (user request, 2026-10-23) ----------------------

import os as _os

_SAMPLES = _os.path.join(_os.path.dirname(__file__), "data", "simpson")


def _sample(name):
    return _os.path.join(_SAMPLES, name)


def test_a_script_fills_in_nuclide_and_field_and_is_not_listed(qapp):  # noqa: F811
    window = MainWindow()
    panel = window.simpson_panel
    panel.add_paths([_sample("inv_pow_wurst.tcl")])
    assert panel.paths() == []
    assert panel.nuclide_combo.currentText() == "31P"
    assert panel._proton_spin.value() == pytest.approx(500.0)
    assert "31P" in panel.status_label.text()


def test_a_multi_element_file_adds_every_element(qapp, tmp_path):  # noqa: F811
    import shutil
    copy = tmp_path / "example.fid"
    shutil.copy(_sample("example.fid"), copy)
    # a sidecar left by an earlier session must not leak one element into the other
    (tmp_path / "example.fid.opennmr_processing.json").write_text(
        '{"opennmr_processing_version": 2, "state": {"title": "example.fid [1/2]"}}')
    window = MainWindow()
    panel = window.simpson_panel
    panel.add_paths([str(copy)])
    panel.open_button.click()
    assert len(window.session) == 2
    titles = [s.metadata.raw["title"] for s in window.session.spectra]
    assert titles == ["example.fid [1/2]", "example.fid [2/2]"]
    assert "2 spectrum/FID" in panel.status_label.text()
    assert not any(n.endswith(".opennmr_processing.json.new") for n in _os.listdir(tmp_path))


def test_plot_picks_the_chosen_element(qapp):  # noqa: F811
    window = MainWindow()
    panel = window.simpson_panel
    panel.add_paths([_sample("example.fid")])
    panel.plot_button.click()
    assert panel.element_spin.maximum() == 2
    first = panel.spectrum.data.copy()
    panel.element_spin.setValue(2)
    panel.plot_button.click()
    assert not np.allclose(first, panel.spectrum.data)


def test_a_scan_table_plots_but_is_not_added(qapp):  # noqa: F811
    window = MainWindow()
    panel = window.simpson_panel
    panel.add_paths([_sample("quad_invert_output.dat")])
    panel.plot_button.click()
    assert "parameter scan" in panel.status_label.text()
    panel.open_button.click()
    assert len(window.session) == 0
    assert "Not spectra or FIDs" in panel.status_label.text()


def test_a_pulse_shape_plots(qapp):  # noqa: F811
    window = MainWindow()
    panel = window.simpson_panel
    panel.add_paths([_sample("inv_pow.shape")])
    panel.plot_button.click()
    assert "pulse shape" in panel.status_label.text()
    assert len(window.fit_canvas.figure.axes) == 2


def test_a_headerless_xreim_dat_opens_as_a_complex_spectrum(qapp, tmp_path):  # noqa: F811
    window = MainWindow()
    panel = window.simpson_panel
    panel.add_paths([sf.write_xreim(tmp_path / "s.dat", "spe")])
    panel.open_button.click()                           # kind left on "detect"
    spectrum = window.session.spectra[0]
    assert spectrum.data_imag is not None and np.max(np.abs(spectrum.data_imag)) > 0
