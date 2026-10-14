"""Settings ▸ Data / Sections shown / Click and drag, and typed-freely
number boxes (explicit user requests, 2026-10-14):

* a default data folder the Open dialogs start in (still overridable);
* the pNMR, DNP and field-drift sections can be hidden, and stay hidden
  after a restart;
* click-and-drag zooms by default (or pans, by choice, remembered), and
  the plot toolbar shows which by checking that button;
* the DPI box goes past 1200 and, like Process ▸ Truncate to, can be
  typed into past three figures.
"""

import json
import os

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QFileDialog, QSpinBox

from opennmr.gui import data_directory
from opennmr.gui.main_window import MainWindow
from opennmr.gui.widgets.canvas import MplCanvas
from opennmr.gui.widgets.typed_spinbox import TypedSpinBox

from test_gui import _make_raw_1d_dataset, qapp  # noqa: F401 -- qapp is a fixture


def _saved(window) -> dict:
    with open(window._UI_SETTINGS_PATH, encoding="utf-8") as fh:
        return json.load(fh)


# ---- default data folder ------------------------------------------------------

def test_data_folder_is_saved_and_open_dialogs_start_there(qapp, tmp_path, monkeypatch):
    data = tmp_path / "nmr data"
    data.mkdir()
    window = MainWindow()
    assert window.data_folder_edit.text() == ""
    window._set_data_folder(str(data))
    assert _saved(window)["data_folder"] == str(data)
    assert data_directory.default_data_directory() == str(data)

    seen = {}

    def fake_dir(parent, caption, directory="", options=None):
        seen.setdefault("dirs", []).append(directory)
        return ""

    def fake_files(parent, caption, directory="", filter="", options=None):
        seen["files"] = directory
        return [], ""
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", staticmethod(fake_dir))
    monkeypatch.setattr(QFileDialog, "getOpenFileNames", staticmethod(fake_files))
    window._on_open_clicked()
    window._on_open_files_clicked()
    window._on_browse_folder_clicked()
    assert seen["dirs"] == [str(data), str(data)] and seen["files"] == str(data)

    # A new window (a restart) reads it back.
    again = MainWindow()
    assert again.data_folder_edit.text() == str(data)
    again._set_data_folder("")
    assert "data_folder" not in _saved(again)
    assert data_directory.start_directory() == ""


def test_a_missing_data_folder_is_refused_and_a_vanished_one_is_ignored(qapp, tmp_path,
                                                                       monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    warned = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warned.append(a[2]))
    window = MainWindow()
    window._set_data_folder(str(tmp_path / "no such folder"))
    assert warned and "not a folder" in warned[0]
    assert "data_folder" not in window._ui_settings
    gone = tmp_path / "unplugged"
    gone.mkdir()
    window._set_data_folder(str(gone))
    gone.rmdir()
    # Kept as set, but dialogs no longer point at it.
    assert data_directory.stored_default_data_directory() == str(gone)
    assert data_directory.start_directory(str(tmp_path)) == str(tmp_path)


def test_the_browser_lists_the_data_folder_at_start_up(qapp, tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    path = _make_raw_1d_dataset(data, monkeypatch, n=32)
    window = MainWindow()
    window._set_data_folder(str(data))
    calls = []
    monkeypatch.setattr(window.data_browser, "set_matches",
                        lambda directory, matches, **k: calls.append((directory, matches)))
    window._browse_default_data_folder()
    assert calls and calls[0][0] == str(data)
    assert len(calls[0][1]) == 1 and os.path.basename(path) in str(calls[0][1][0])
    window.data_folder_browse_at_start_checkbox.setChecked(False)
    assert _saved(window)["data_folder_browse_at_start"] is False


# ---- sections shown ---------------------------------------------------------------

def test_sections_can_be_hidden_and_stay_hidden_after_a_restart(qapp):
    window = MainWindow()
    for key in MainWindow.OPTIONAL_SECTIONS:
        assert window.section_shown_checkboxes[key].isChecked()
    window._set_section_shown("pnmr", False)
    window.section_shown_checkboxes["drift"].setChecked(False)
    assert window._pnmr_fit_section.isHidden() and window._pnmr_sim_section.isHidden()
    assert window.drift_section.isHidden()
    assert not window._dnp_section.isHidden()
    assert _saved(window)["hidden_sections"] == ["drift", "pnmr"]

    again = MainWindow()
    again.show()
    assert not again._pnmr_fit_section.isVisible() and not again.drift_section.isVisible()
    assert not again.section_shown_checkboxes["pnmr"].isChecked()
    # ...and back.
    again.section_shown_checkboxes["pnmr"].setChecked(True)
    again.section_shown_checkboxes["drift"].setChecked(True)
    assert not again._pnmr_fit_section.isHidden() and not again.drift_section.isHidden()
    assert "hidden_sections" not in _saved(again)


# ---- click and drag ----------------------------------------------------------------

class _Event:
    def __init__(self, canvas, xdata, ydata, button=1):
        ax = canvas.ax
        self.inaxes = ax
        self.button = button
        self.xdata, self.ydata = xdata, ydata
        self.x, self.y = ax.transData.transform((xdata, ydata))


def _line_canvas():
    canvas = MplCanvas()
    canvas.resize(600, 400)
    canvas.ax = canvas.figure.add_subplot(111)
    x = np.linspace(10, 0, 200)
    canvas.ax.plot(x, np.sin(x))
    canvas.ax.set_xlim(10, 0)               # NMR: high ppm on the left
    canvas.ax.set_ylim(-1.5, 1.5)
    canvas.canvas.draw()
    return canvas


def test_box_zoom_drag_keeps_the_axis_direction(qapp):
    canvas = _line_canvas()
    canvas.set_drag_mode("zoom")
    canvas._on_press(_Event(canvas, 8.0, -1.0))
    canvas._on_motion(_Event(canvas, 4.0, 0.5))
    canvas._on_release(_Event(canvas, 4.0, 0.5))
    lo, hi = canvas.ax.get_xlim()
    assert lo == pytest.approx(8.0, abs=1e-6) and hi == pytest.approx(4.0, abs=1e-6)
    assert canvas.ax.get_ylim() == pytest.approx((-1.0, 0.5), abs=1e-6)
    assert canvas._zoom_start is None


def test_a_flat_box_zooms_x_only_and_a_click_does_nothing(qapp):
    canvas = _line_canvas()
    canvas.set_drag_mode("zoom")
    canvas._on_press(_Event(canvas, 9.0, 0.0))
    canvas._on_release(_Event(canvas, 9.0, 0.0))        # a click
    assert canvas.ax.get_xlim() == pytest.approx((10, 0))
    canvas._on_press(_Event(canvas, 9.0, 0.0))
    canvas._on_release(_Event(canvas, 6.0, 0.001))       # no height: x only
    assert canvas.ax.get_xlim() == pytest.approx((9.0, 6.0), abs=1e-6)
    assert canvas.ax.get_ylim() == pytest.approx((-1.5, 1.5))


def test_pan_drag_is_unchanged(qapp):
    canvas = _line_canvas()
    canvas.set_drag_mode("pan")
    canvas._on_press(_Event(canvas, 5.0, 0.0))
    canvas._on_motion(_Event(canvas, 4.0, 0.0))
    canvas._on_release(_Event(canvas, 4.0, 0.0))
    lo, hi = canvas.ax.get_xlim()
    assert lo == pytest.approx(11.0, abs=1e-6) and hi == pytest.approx(1.0, abs=1e-6)


def test_toolbar_checks_what_a_drag_will_do(qapp):
    canvas = _line_canvas()
    actions = canvas.toolbar._actions
    canvas.set_drag_mode("zoom")
    assert actions["zoom"].isChecked() and not actions["pan"].isChecked()
    canvas.set_drag_mode("pan")
    assert actions["pan"].isChecked() and not actions["zoom"].isChecked()
    # A click mode owns the drag: neither is checked while it is on.
    canvas.set_pick_mode(True)
    assert not actions["pan"].isChecked() and not actions["zoom"].isChecked()
    canvas.set_pick_mode(False)
    assert actions["pan"].isChecked()
    # matplotlib's own tool still shows as itself.
    canvas.toolbar.zoom()
    assert actions["zoom"].isChecked() and not actions["pan"].isChecked()
    canvas.toolbar.zoom()
    assert actions["pan"].isChecked()
    with pytest.raises(ValueError):
        canvas.set_drag_mode("rotate")


def test_window_drag_mode_defaults_to_zoom_and_is_remembered(qapp):
    window = MainWindow()
    assert window.drag_mode_combo.currentData() == "zoom"
    assert {c.drag_mode() for c in window._all_canvases()} == {"zoom"}
    window.drag_mode_combo.setCurrentIndex(window.drag_mode_combo.findData("pan"))
    assert {c.drag_mode() for c in window._all_canvases()} == {"pan"}
    assert _saved(window)["drag_mode"] == "pan"
    again = MainWindow()
    assert again.drag_mode_combo.currentData() == "pan"
    assert again.canvas.drag_mode() == "pan"


# ---- typed number boxes ----------------------------------------------------------------

def _type_after(box, text):
    box.show()
    box.lineEdit().setFocus()
    box.lineEdit().end(False)
    QTest.keyClicks(box.lineEdit(), text)


def test_a_fourth_digit_can_be_typed_after_three(qapp):
    plain, typed = QSpinBox(), TypedSpinBox()
    for box in (plain, typed):
        box.setRange(1, 4096)
        box.setValue(999)
    _type_after(plain, "1")
    _type_after(typed, "1")
    assert plain.text() == "999"                # Qt's own box refuses 9991 (over 4096)
    assert typed.text() == "9991"               # kept while typing...
    typed.interpretText()
    assert typed.value() == 4096                # ...and brought inside the range on Enter


def test_dpi_and_truncation_boxes(qapp, tmp_path, monkeypatch):
    window = MainWindow()
    assert isinstance(window.output_dpi_spin, TypedSpinBox)
    assert window.output_dpi_spin.maximum() == MainWindow.MAX_EXPORT_DPI >= 2400
    window.output_dpi_spin.setValue(2400)
    assert window.output_dpi_spin.value() == 2400
    for name in ("tdeff_spin", "tdeff_f1_spin", "zero_fill_custom_spin", "zero_fill_f1_spin"):
        assert isinstance(getattr(window, name), TypedSpinBox), name
    window._load(_make_raw_1d_dataset(tmp_path, monkeypatch, n=4096), vendor="bruker")
    box = window.tdeff_spin
    box.setValue(999)
    _type_after(box, "")
    box.lineEdit().selectAll()
    QTest.keyClicks(box.lineEdit(), "2500")
    box.interpretText()
    assert box.value() == 2500
