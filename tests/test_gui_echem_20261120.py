"""Apply ▸ Electrochemistry (gui/echem_panel.py, Plot Style section)."""
import numpy as np
import pytest
from PySide6.QtWidgets import QMessageBox

from opennmr.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
from opennmr.gui.main_window import MainWindow

from test_gui import qapp  # noqa: F401 -- qapp is a fixture
from test_echem_20261120 import _profile, MASS_MG

PPM = np.linspace(100.0, -100.0, 400)


def _spec(k, path=None):
    y = np.exp(-((PPM - (50 - 5 * k)) / 6.0) ** 2)
    md = NMRMetadata(vendor="simulated", nucleus="23Na", sfo1=100.0, bf1=100.0)
    if path:
        md.source_path = path
    return NMRSpectrum(data=y, axes=[NMRAxis(kind="frequency", ppm=PPM, hz=PPM * 100.0)], metadata=md)


@pytest.fixture
def window(qapp, monkeypatch, tmp_path):
    for name in ("warning", "information", "critical"):
        monkeypatch.setattr(QMessageBox, name, lambda *a, **k: None)
    w = MainWindow()
    for k in range(6):
        w._add_derived_spectrum(_spec(k, f"/data/sample/{10 + k}"), f"exp {10 + k}")
    csv = tmp_path / "cell.csv"
    _profile().to_csv(csv, index=False)
    p = w.echem_panel
    p.path_edit.setText(str(csv))
    p.mass_edit.setText(str(MASS_MG))
    p.molar_edit.setText("100")
    p.refresh()
    # Derived spectra carry their label, not a folder; real experiments carry their folder, so
    # the range selection is tested with the folders a Bruker load would give.
    real = p._spectra
    def with_folders():
        out = real()
        for k, e in enumerate(out):
            e["path"] = f"/data/sample/{10 + k}"
        return out
    p._spectra = with_folders
    p.refresh()
    return w


def test_section_is_in_apply_and_optional(window):
    assert window.OPTIONAL_SECTIONS["echem"][1] == ("_echem_section",)
    assert "echem" in window.APPLY_TILE_SECTIONS
    assert not window._echem_section.isExpanded()


def test_load_reports_cycles_and_fills_column_choices(window):
    p = window.echem_panel
    assert p.load()
    assert "2 cycle" in p.info_label.text()
    assert p.x_combo.findData("voltage_V") >= 0 and p.position_combo.findData("index") >= 0


def test_bad_file_is_reported_not_raised(window):
    p = window.echem_panel
    p.path_edit.setText("/no/such/file.mpr")
    assert not p.load()
    assert "Could not read" in p.report_label.text()


def test_range_of_experiment_numbers_picks_open_spectra(window):
    p = window.echem_panel
    p.select_combo.setCurrentIndex(p.select_combo.findData("range"))
    p.exp_start.setValue(10); p.exp_stop.setValue(15); p.exp_step.setValue(2)
    chosen, notes = p._chosen()
    assert [e["label"].split()[1] for e in chosen] == ["10", "12", "14"]
    p.exp_stop.setValue(40); p.exp_step.setValue(10)
    chosen, notes = p._chosen()
    assert [e["label"].split()[1] for e in chosen] == ["10"] and any("Not open" in n for n in notes)


def test_operando_plot_draws_cell_beside_spectra_and_replots_on_style(window):
    p = window.echem_panel
    p.select_combo.setCurrentIndex(p.select_combo.findData("range"))
    p.exp_start.setValue(10); p.exp_stop.setValue(15); p.exp_step.setValue(1)
    p.per_edit.setText("24"); p.time_unit_combo.setCurrentText("h")
    p.display_combo.setCurrentIndex(p.display_combo.findData("heatmap"))
    p.refresh()
    p.load()
    p.plot()
    assert p.plotted and p.link is not None and len(p.link) == 6
    fig = window.compare_canvas.figure
    assert hasattr(fig, "_echem_crosshair") and fig.axes_cell is not None
    window.echem_cell_xlabel_edit.setText("Potential (V)")
    window._on_echem_style_changed()
    assert window.compare_canvas.figure.axes_cell.get_xlabel() == "Potential (V)"
    window.echem_cell_xticks_combo.setCurrentIndex(window.echem_cell_xticks_combo.findData("none"))
    cell = window.compare_canvas.figure.axes_cell
    assert cell.xaxis._major_tick_kw["label1On"] is False
    spectra = window.compare_canvas.figure.axes_spectra
    assert spectra.xaxis._major_tick_kw.get("label1On", True) is True   # the other plot, the other axis
    window.echem_cell_top_spine_check.setChecked(True)
    assert window.compare_canvas.figure.axes_cell.spines["top"].get_visible()


def test_ex_situ_table_and_missing_value_message(window):
    p = window.echem_panel
    p.load()
    p.mode_combo.setCurrentIndex(p.mode_combo.findData("ex_situ"))
    p.select_combo.setCurrentIndex(p.select_combo.findData("range"))
    p.exp_start.setValue(10); p.exp_stop.setValue(12); p.exp_step.setValue(1)
    p.refresh()
    assert p.ex_table.rowCount() == 3
    p.plot()
    assert "needs a value" in p.report_label.text()
    for r, v in enumerate(("3.4", "3.6", "3.8")):
        p.ex_table.item(r, 1).setText(v)
    p.plot()
    assert p.plotted and len(p.link) == 3 and abs(p.link["voltage_V"].iloc[1] - 3.6) < 0.03


def test_no_spectra_chosen_says_so(window):
    p = window.echem_panel
    p.load()
    window.spectrum_list.clearSelection()
    p.refresh()
    p.plot()
    assert "No spectra chosen" in p.report_label.text()


def test_notebook_code_runs(window):
    p = window.echem_panel
    p.load()
    p.per_edit.setText("24")
    p.select_combo.setCurrentIndex(p.select_combo.findData("range"))
    p.refresh()
    code = p.notebook_code()
    assert "plot_echem_nmr" in code and "operando_times" in code
    compile(code, "<code>", "exec")


def _ex_situ(window, values=("3.4", "3.6", "3.8")):
    p = window.echem_panel
    p.load()
    p.mode_combo.setCurrentIndex(p.mode_combo.findData("ex_situ"))
    p.select_combo.setCurrentIndex(p.select_combo.findData("range"))
    p.exp_start.setValue(10); p.exp_stop.setValue(12); p.exp_step.setValue(1)
    p.refresh()
    for r, v in enumerate(values):
        p.ex_table.item(r, 1).setText(v)
    return p


def test_unplaceable_samples_give_a_message_not_a_traceback(window):
    p = _ex_situ(window, ("9.0", "9.5", "9.9"))
    p.plot()
    assert "No spectrum could be placed" in p.report_label.text()


def test_even_spacing_mass_factor_and_colours_from_the_table(window):
    p = _ex_situ(window)
    p.distribute_combo.setCurrentIndex(p.distribute_combo.findData("even"))
    p.by_mass_check.setChecked(True)
    p.plot()
    assert "sample mass" in p.report_label.text()        # a mass is needed first
    for r, (m, f) in enumerate((("1", "1"), ("2", "2"), ("4", "1"))):
        p.ex_table.item(r, 5).setText(m); p.ex_table.item(r, 6).setText(f)
    spec = p.compute()
    peaks = spec["matrix"].max(axis=1)
    assert np.allclose(peaks, [1.0, 0.5 * 2, 0.25], atol=2e-3)
    p.plot()
    fig = window.compare_canvas.figure
    fig.canvas.draw()
    assert len(fig.axes_spectra.get_yticks()) == 3
    assert "scaling_factors" in p.notebook_code() and "distribute='even'" in p.notebook_code()



def test_mass_from_the_right_click_nucleus_and_colour_cells(window):
    p = window.echem_panel
    p.select_combo.setCurrentIndex(p.select_combo.findData("range"))
    p.exp_start.setValue(10); p.exp_stop.setValue(12); p.exp_step.setValue(1)
    window.session.set_mass(0, 0.0125)            # the right-click "Set sample mass" action: grams
    p.refresh()
    assert p.ex_table.item(0, 5).text() == "12.5" and p.ex_table.item(1, 5).text() == ""
    p.ex_table.item(1, 5).setText("3")             # a typed value is kept
    window.session.set_mass(0, 0.02)
    p.refresh()
    assert p.ex_table.item(0, 5).text() == "20" and p.ex_table.item(1, 5).text() == "3"
    p.per_edit.setText("24"); p.time_unit_combo.setCurrentText("h")
    p.display_combo.setCurrentIndex(p.display_combo.findData("superimposed"))
    p.plot()
    fig = window.compare_canvas.figure
    from opennmr.core.registry import NUC_label
    assert fig.axes_spectra.get_xlabel() == NUC_label("23Na", x_unit="ppm")


def test_colours_of_spectra_and_points_are_in_the_series_list(window):
    p = _ex_situ(window)
    p.display_combo.setCurrentIndex(p.display_combo.findData("superimposed"))
    p.plot()
    canvas = window.compare_canvas
    entries = {e.key: e for e in canvas._series["external"]}
    name = p.ex_table.item(1, 0).text()
    assert name in entries and "cell" in entries           # one entry per spectrum, one for the curve
    entries[name].colour = "#ff0000"
    entries[name].colour_overridden = True
    canvas._redraw_external()
    fig = canvas.figure
    from matplotlib.colors import to_hex
    lines = [l for l in fig.axes_spectra.lines if l.get_label() == name]
    points = [l for l in fig.axes_cell.lines if l.get_label() == name]
    assert lines and points                                  # linked: one setting for both
    assert to_hex(lines[0].get_color()) == "#ff0000" and to_hex(points[0].get_color()) == "#ff0000"
    # a name can be switched off in Plot Style
    window.echem_spectra_ylabel_show_check.setChecked(False)
    assert window.compare_canvas.figure.axes_spectra.get_ylabel() == ""


@pytest.mark.parametrize("side", ["left", "top"])
def test_switching_sides_keeps_the_spectra_and_measure_gives_ppm_and_hz(window, side):
    p = window.echem_panel
    p.select_combo.setCurrentIndex(p.select_combo.findData("range"))
    p.exp_start.setValue(10); p.exp_stop.setValue(15); p.exp_step.setValue(1)
    p.per_edit.setText("6"); p.time_unit_combo.setCurrentText("h")
    p.refresh(); p.load()
    p.plot()                                              # cell on the right first
    p.side_combo.setCurrentIndex(p.side_combo.findData(side))
    p.plot()
    canvas = window.compare_canvas
    fig = canvas.figure
    assert canvas.ax is fig.axes_spectra and len(fig.axes_spectra.lines) >= 6
    assert fig.axes_spectra.get_xlim() == (100.0, -100.0)
    got = []
    canvas.measurementCompleted.connect(got.append)
    canvas._measurement_points = [(fig.axes_spectra, 10.0, 0.5), (fig.axes_spectra, 0.0, 0.5)]
    canvas._complete_measurement()
    assert got, "no measurement reported"
    text = str(got[0])
    assert "ppm" in text.lower() and "hz" in text.lower()
    dx = got[0]["dx"]
    assert dx.get("ppm") == pytest.approx(10.0) and dx.get("hz") == pytest.approx(1000.0)   # 100 MHz


def test_gap_and_left_spine_controls_in_plot_style(window):
    p = window.echem_panel
    p.select_combo.setCurrentIndex(p.select_combo.findData("range"))
    p.exp_start.setValue(10); p.exp_stop.setValue(15); p.exp_step.setValue(1)
    p.per_edit.setText("24"); p.time_unit_combo.setCurrentText("h")
    p.display_combo.setCurrentIndex(p.display_combo.findData("superimposed"))
    p.refresh(); p.load(); p.plot()
    f0 = window.compare_canvas.figure
    before = f0.axes_cell.get_position().x0 - f0.axes_spectra.get_position().x1
    window.echem_spectra_left_spine_check.setChecked(False)
    assert not window.compare_canvas.figure.axes_spectra.spines["left"].get_visible()
    window.echem_gap_spin.setValue(0.4)
    fig = window.compare_canvas.figure
    assert fig.axes_cell.get_position().x0 - fig.axes_spectra.get_position().x1 > before + 0.1
