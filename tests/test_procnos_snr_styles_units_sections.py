"""Explicit user requests, 2026-10-14 (third round):

* right-click an experiment (Open experiments, or the Open tab's
  browser) to see its proc numbers as a checklist and open them;
* computing signal-to-noise no longer throws away the plot's names and
  colours;
* Settings ▸ Interface ▸ Export size in: inches (default) or cm;
* more optional sections: Czjzek (Fit and Simulate), eNMR, Imaging and
  the whole Advanced Processing tile.
"""

import json

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QTreeWidgetItemIterator

from opennmr.gui.main_window import MainWindow
from opennmr.gui.procno_dialog import PROCNO_ROLE, ProcnoDialog
from opennmr.gui.widgets.typed_spinbox import LengthSpinBox
from opennmr.io import bruker
from opennmr.io import persistence

from test_gui import _make_raw_1d_dataset, _write_params, qapp  # noqa: F401 -- qapp is a fixture
from test_gui_snr import _load as _load_snr  # noqa: E402


def _saved(window) -> dict:
    with open(window._UI_SETTINGS_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def _two_procnos(tmp_path, monkeypatch):
    path = _make_raw_1d_dataset(tmp_path, monkeypatch, n=64, name="expt")
    pdata2 = tmp_path / "expt" / "pdata" / "2"
    pdata2.mkdir(parents=True)
    _write_params(pdata2 / "procs", {"SW_p": 8000.0, "SI": 64, "SF": 400.13, "WDW": 1,
                                     "LB": 50.0, "SSB": 0.0, "PHC0": 12.0})
    (pdata2 / "title").write_text("broadened\n")
    return path


# ---- proc numbers ------------------------------------------------------------------

def test_available_procnos_lists_each_with_what_topspin_did(tmp_path, monkeypatch):
    path = _two_procnos(tmp_path, monkeypatch)
    entries = bruker.available_procnos(path)
    assert [e["procno"] for e in entries] == [1, 2]
    assert entries[1]["title"] == "broadened"
    assert "EM LB 50" in entries[1]["summary"] and "PHC0 12.0" in entries[1]["summary"]
    assert entries[0]["processed"] == []          # no 1r written by the fixture
    assert bruker.available_procnos(str(tmp_path / "nowhere")) == []


def test_dialog_locks_what_is_open_and_returns_the_rest(qapp, tmp_path, monkeypatch):
    path = _two_procnos(tmp_path, monkeypatch)
    entries = bruker.available_procnos(path)
    dialog = ProcnoDialog("expt", entries, open_procnos=[1], preselect=[2])
    first, second = dialog.list.item(0), dialog.list.item(1)
    assert first.checkState() == Qt.CheckState.Checked
    assert not (first.flags() & Qt.ItemFlag.ItemIsEnabled)
    assert second.data(PROCNO_ROLE) == 2 and "broadened" in second.text()
    assert dialog.to_open() == [2]
    dialog.set_checked([])
    assert dialog.to_open() == []


def _answer(monkeypatch, procnos):
    seen = {}

    def run(self, dialog):
        seen["rows"] = [dialog.list.item(i).text() for i in range(dialog.list.count())]
        dialog.set_checked(procnos)
        return True
    monkeypatch.setattr(MainWindow, "_run_procno_dialog", run)
    return seen


def test_open_experiments_right_click_opens_other_procnos(qapp, tmp_path, monkeypatch):
    path = _two_procnos(tmp_path, monkeypatch)
    window = MainWindow()
    window._load(path, vendor="bruker")
    _answer(monkeypatch, [2])
    rows = window._choose_procnos(path, "bruker")
    assert len(window.session) == 2 and window.session.procno(rows[0]) == 2
    assert window.session.index_for_path(path, procno=1) == 0
    assert window.session.index_for_path(path, procno=2) == 1
    assert window.session.spectra[1].metadata.raw["procs"]["WDW"] == 1
    assert not np.allclose(window.session.spectra[0].data, window.session.spectra[1].data)
    # Asked again, nothing new: both are open (and locked in the list).
    assert window._choose_procnos(path, "bruker") == []
    assert len(window.session) == 2


def test_the_browser_offers_proc_numbers(qapp, tmp_path, monkeypatch):
    path = _two_procnos(tmp_path, monkeypatch)
    window = MainWindow()
    window.data_browser.set_matches(str(tmp_path), [(path, "bruker")])
    iterator = QTreeWidgetItemIterator(window.data_browser._tree)
    item = None
    while iterator.value() is not None:
        if iterator.value().data(0, Qt.ItemDataRole.UserRole):
            item = iterator.value()
        iterator += 1
    actions = dict(window.data_browser.context_actions(item))
    assert set(actions) == {"Open", "Proc Numbers…"}
    _answer(monkeypatch, [1, 2])
    actions["Proc Numbers…"]()
    assert sorted(window.session.procno(i) for i in range(len(window.session))) == [1, 2]


def test_another_procno_neither_reads_nor_overwrites_the_saved_processing(qapp, tmp_path,
                                                                         monkeypatch):
    path = _two_procnos(tmp_path, monkeypatch)
    window = MainWindow()
    window._load(path, vendor="bruker")
    spec = window.session.spectra[0]
    spec.metadata.raw["live_window"] = "qsine"
    window._run_sidecar_save_now(0)
    assert persistence.load_processing_sidecar(path)["live_window"] == "qsine"
    _answer(monkeypatch, [2])
    row = window._choose_procnos(path, "bruker")[0]
    other = window.session.spectra[row]
    assert other.metadata.raw.get("live_window") != "qsine"     # its own processing
    other.metadata.raw["live_window"] = "exponential"
    window._run_sidecar_save_now(row)
    assert persistence.load_processing_sidecar(path)["live_window"] == "qsine"   # untouched


# ---- S/N keeps the plot's names and colours ------------------------------------------

def test_snr_recalculation_keeps_names_and_colours(qapp, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: None)
    window = MainWindow()
    panel = _load_snr(window)
    canvas = window.compare_canvas
    panel.set_signal_region(5, -5)
    panel.set_noise_region([(80, 40)])
    panel.calculate(explicit=True)
    key = canvas._series["external"][0].key
    canvas.set_series_style(key, label="Renamed", colour="#ff0000")
    canvas.set_style(labelsize=21)
    panel.set_signal_region(4, -4)
    panel.calculate(explicit=True)
    lines = [ln for ln in canvas.figure.axes[0].lines if not ln.get_label().startswith("_")]
    assert lines[0].get_label() == "Renamed" and lines[0].get_color() == "#ff0000"
    assert canvas.figure.axes[0].xaxis.label.get_fontsize() == 21


def test_external_plots_of_another_kind_start_fresh(qapp):
    from opennmr.gui.widgets.canvas import MplCanvas

    def draw(figure):
        figure.add_subplot(111).plot([0, 1], [0, 1], label="a")[0].axes.legend()
    canvas = MplCanvas()
    canvas.show_external_plot(draw, kind="compare", series_group="snr")
    canvas.set_series_style("a", colour="#00ff00")
    canvas.show_external_plot(draw, kind="compare", series_group="snr")
    assert canvas._series["external"][0].colour == "#00ff00"
    canvas.show_external_plot(draw, kind="compare", series_group="dnp")
    assert not canvas._series["external"][0].colour_overridden


# ---- export size unit ------------------------------------------------------------------

def test_length_box_shows_cm_and_keeps_inches(qapp):
    box = LengthSpinBox()
    box.setRange(1.0, 40.0)
    box.setValue(8.0)
    assert box.text() == "8.00 in"
    box.set_unit("cm")
    assert box.text() == "20.32 cm" and box.value() == 8.0
    box.lineEdit().setText("21 cm")
    box.interpretText()
    assert box.value() == pytest.approx(21 / 2.54, abs=1e-6)
    box.set_unit("in")
    assert box.text() == "8.27 in"


def test_export_unit_setting_is_remembered(qapp):
    window = MainWindow()
    assert window.export_unit_combo.currentData() == "in"
    window.output_fig_width_spin.setValue(10.0)
    window.export_unit_combo.setCurrentIndex(window.export_unit_combo.findData("cm"))
    assert window.output_fig_width_spin.text() == "25.40 cm"
    assert window._current_output_export_kwargs()["fig_width"] == 10.0      # inches inside
    assert _saved(window)["export_unit"] == "cm"
    again = MainWindow()
    assert again.output_fig_height_spin.unit() == "cm"


# ---- more optional sections ----------------------------------------------------------

def test_new_optional_sections_hide_and_come_back(qapp):
    window = MainWindow()
    for key in ("czjzek", "enmr", "imaging", "advanced"):
        assert key in window.section_shown_checkboxes
    window.section_shown_checkboxes["czjzek"].setChecked(False)
    window.section_shown_checkboxes["enmr"].setChecked(False)
    window.section_shown_checkboxes["imaging"].setChecked(False)
    assert window._czjzek_fit_section.isHidden() and window._czjzek_sim_section.isHidden()
    assert window._enmr_section.isHidden() and window._imaging_section.isHidden()
    window.section_shown_checkboxes["czjzek"].setChecked(True)
    assert not window._czjzek_fit_section.isHidden()


def test_the_whole_advanced_processing_tile_is_hidden(qapp):
    window = MainWindow()
    tile = next(i for i in range(window.side_tabs.count())
                if window.side_tabs.tabText(i) == "Adv. Proc.")
    window._show_side_tab("Adv. Proc.")
    window.section_shown_checkboxes["advanced"].setChecked(False)
    assert not window.side_tabs.isTabVisible(tile)
    assert window.side_tabs.currentIndex() != tile                # left for Open
    assert not window._window_panel_actions["Adv. Proc."].isVisible()
    assert "advanced" in _saved(window)["hidden_sections"]
    again = MainWindow()                                           # a restart
    assert not again.side_tabs.isTabVisible(tile)
    assert not again._window_panel_actions["Adv. Proc."].isVisible()
    again.section_shown_checkboxes["advanced"].setChecked(True)
    assert again.side_tabs.isTabVisible(tile)
    assert again._window_panel_actions["Adv. Proc."].isVisible()
