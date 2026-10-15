"""Follow-ups of 2026-10-15 (evening):

* the extracted-slice panel beside a contour drew its bottom trace's
  baseline on the panel's axis line -- every trace now floats above it
  (Plot Style's gaps, which also space the traces);
* Settings ▸ Double-click a plot to reset its view (on by default);
* File ▸ Clear All History… deletes openNMR's files beside an
  experiment and re-opens it with no earlier processing;
* the Compare list lets go of its current row (and the plot's tint) on
  a click in its blank space or Escape, and an export never shows the tint.
"""

import json
import os
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest

from opennmr.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
from opennmr.gui.main_window import MainWindow
from opennmr.gui.widgets.canvas import MplCanvas
from opennmr.io import history as io_history
from opennmr.io import persistence as io_persistence
from opennmr.plotting import twod
from opennmr.plotting.twod import SLICE_PANEL_GID

from test_gui import (  # noqa: F401 -- qapp is a fixture
    _make_1d_dataset, _make_raw_1d_dataset, _select_rows, qapp,
)


def _cosy_like(n1=16, n2=512):
    """A 2D whose rows sit on a negative baseline with one peak each."""
    f2 = np.linspace(10, 0, n2)
    data = np.full((n1, n2), -0.2)
    for r in range(n1):
        data[r] += np.exp(-((f2 - (2 + 0.3 * r)) / 0.05) ** 2)
    data[:, ::7] -= 0.05                       # a little "noise" below the baseline
    axes = [NMRAxis(kind="frequency", ppm=np.linspace(10, 0, n1)), NMRAxis(kind="frequency", ppm=f2)]
    metadata = NMRMetadata(vendor="derived", nucleus="1H", sfo1=80.0, nucleus_indirect="1H",
                           sfo1_indirect=80.0, source_path="synthetic")
    return NMRSpectrum(data=data, axes=axes, metadata=metadata)


def _panel(fig):
    return next(ax for ax in fig.axes if ax.get_gid() == SLICE_PANEL_GID)


# ---- slice panel ------------------------------------------------------------

@pytest.mark.parametrize("gap", [0.05, 0.3])
def test_every_slice_floats_above_the_panel_axis(gap):
    fig, _ax, _proj = twod.plot_2d_contour(_cosy_like(), slices=[{"axis": "f1", "index": 3},
                                                         {"axis": "f1", "index": 12}],
                                    slice_bottom_gap=gap)
    panel = _panel(fig)
    low = panel.get_ylim()[0]
    lowest = min(float(np.min(line.get_ydata())) for line in panel.get_lines())
    heights = [float(np.ptp(line.get_ydata())) for line in panel.get_lines()]
    assert lowest == pytest.approx(0.0)          # the bottom trace starts at its slot
    assert low == pytest.approx(-gap * min(heights), rel=1e-6)
    assert low < lowest


def test_the_gui_slice_panel_takes_plot_styles_gaps(qapp):
    canvas = MplCanvas()
    canvas.set_contour_slices([{"axis": "f1", "index": 3, "label": ""},
                               {"axis": "f1", "index": 12, "label": ""}])
    canvas.show_spectrum(_cosy_like())
    trace_height = float(np.ptp(_panel(canvas.figure).get_lines()[0].get_ydata()))
    assert _panel(canvas.figure).get_ylim()[0] == pytest.approx(-MplCanvas._BASELINE_GAP * trace_height)
    canvas.set_style(baseline_gap=0.5, stack_gap=1.0)
    panel = _panel(canvas.figure)
    lines = panel.get_lines()
    height = float(np.ptp(lines[0].get_ydata()))
    assert height == pytest.approx(0.5 * 0.5)           # slot 1/2, trace fills 1/(1+1) of it
    assert panel.get_ylim()[0] == pytest.approx(-0.5 * height)


def test_default_gap_below_a_1d_is_ten_percent(qapp):
    assert MplCanvas._BASELINE_GAP == pytest.approx(0.10)
    window = MainWindow()
    assert window.output_baseline_gap_spin.value() == pytest.approx(10.0)


# ---- double-click resets the view ----------------------------------------------

def _double_click(canvas):
    ax = canvas.ax
    return SimpleNamespace(button=1, dblclick=True, inaxes=ax, x=10.0, y=10.0,
                           xdata=float(np.mean(ax.get_xlim())), ydata=float(np.mean(ax.get_ylim())))


def _spectrum_1d():
    x = np.linspace(10, 0, 400)
    metadata = NMRMetadata(vendor="derived", nucleus="1H", sfo1=400.0, source_path="synthetic")
    return NMRSpectrum(data=np.exp(-((x - 4) / 0.1) ** 2), axes=[NMRAxis(kind="frequency", ppm=x)],
                       metadata=metadata)


def test_double_click_resets_the_view(qapp):
    canvas = MplCanvas()
    canvas.show_spectrum(_spectrum_1d())
    full_x, full_y = canvas.ax.get_xlim(), canvas.ax.get_ylim()
    canvas.note_natural_view()
    canvas.ax.set_xlim(6, 3)
    canvas.ax.set_ylim(0.2, 0.4)
    canvas._on_press(_double_click(canvas))
    assert canvas.ax.get_xlim() == pytest.approx(full_x)
    assert canvas.ax.get_ylim() == pytest.approx(full_y)


def test_double_click_can_be_switched_off(qapp):
    canvas = MplCanvas()
    canvas.show_spectrum(_spectrum_1d())
    canvas.set_double_click_resets(False)
    canvas.note_natural_view()
    canvas.ax.set_xlim(6, 3)
    canvas._on_press(_double_click(canvas))
    assert canvas.ax.get_xlim() == pytest.approx((6, 3))


def test_double_click_setting_is_remembered_and_reaches_every_plot(qapp):
    window = MainWindow()
    assert window.double_click_reset_checkbox.isChecked()
    window.double_click_reset_checkbox.setChecked(False)
    assert all(not c.double_click_resets() for c in window._all_canvases())
    with open(window._UI_SETTINGS_PATH, encoding="utf-8") as fh:
        assert json.load(fh)["double_click_reset"] is False
    again = MainWindow()
    assert not again.double_click_reset_checkbox.isChecked()
    assert not again.canvas.double_click_resets()


# ---- clear all history ---------------------------------------------------------

def test_clear_experiment_files_deletes_openNMRs_files_only(tmp_path):
    expt = tmp_path / "3"
    expt.mkdir()
    (expt / "fid").write_bytes(b"data")
    for name in ("opennmr_history.json", "opennmr_log.txt", "opennmr_processing.json"):
        (expt / name).write_text("{}")
    (expt / "opennmr_metadata.json").write_text(json.dumps({"notes": "My sample", "mass_g": 2.0}))
    (expt / io_history.SNAPSHOT_DIRNAME).mkdir()
    (expt / io_history.SNAPSHOT_DIRNAME / "savepoint_1.npz").write_bytes(b"x")
    deleted = io_history.clear_experiment_files(str(expt))
    assert len(deleted) == 5
    assert sorted(os.listdir(expt)) == ["fid", "opennmr_metadata.json"]     # data kept, notes kept
    assert json.loads((expt / "opennmr_metadata.json").read_text()) == {
        "notes": "My sample", "opennmr_metadata_version": 1}
    (expt / "opennmr_processing.json").write_text("{}")
    io_history.clear_experiment_files(str(expt), keep_notes=False)
    assert os.listdir(expt) == ["fid"]


def test_clear_all_history_reopens_without_earlier_processing(qapp, tmp_path, monkeypatch):
    path = _make_raw_1d_dataset(tmp_path, monkeypatch, n=128)
    window = MainWindow()
    window._load(path, vendor="bruker")
    index = window._selected_index()
    original = np.array(window.session.spectra[index].data)
    window.phase0_spin.setValue(40.0)
    window._flush_pending_state(index)
    window.session.set_nucleus(index, "19F")
    window._run_sidecar_save_now(index)
    window._add_savepoint(index, name="Kept")
    window._write_history(index)
    assert os.path.exists(io_persistence.processing_sidecar_path(path))
    assert os.path.exists(io_history.log_path(path))
    assert window.can_undo(index)

    deleted = window._clear_all_history(index, keep_notes=False)
    qapp.processEvents()
    assert deleted
    assert io_history.experiment_files(path) == []
    spectrum = window.session.spectra[index]
    assert float(spectrum.metadata.raw.get("phase0", 0.0) or 0.0) == 0.0
    assert np.allclose(spectrum.data, original)
    assert spectrum.metadata.nucleus == "1H"            # the reader's, not the override
    assert not window.can_undo(index)
    history = window._history_info(index)["history"]
    assert not io_history.savepoints(history)
    # A reopen starts clean as well.
    other = MainWindow()
    other._load(path, vendor="bruker")
    assert float(other.session.spectra[-1].metadata.raw.get("phase0", 0.0) or 0.0) == 0.0


def test_clear_all_history_is_in_the_file_menu(qapp):
    window = MainWindow()
    assert window.clear_history_action.text().replace("&", "") == "Clear All History…"


# ---- Compare list: click away to deselect ------------------------------------------

def _compare_two(window, tmp_path):
    window._load(_make_1d_dataset(tmp_path, name="a"), vendor="bruker")
    window._load(_make_1d_dataset(tmp_path, name="b"), vendor="bruker")
    _select_rows(window, [0, 1])
    window.compare_list.resize(300, 400)
    window.compare_list.setCurrentRow(0)
    assert window.compare_canvas._selected_compare_index == 0


def test_clicking_blank_space_in_the_compare_list_deselects(qapp, tmp_path):
    window = MainWindow()
    _compare_two(window, tmp_path)
    viewport = window.compare_list.viewport()
    blank = QPoint(10, viewport.height() - 5)
    assert window.compare_list.itemAt(blank) is None
    QTest.mouseClick(viewport, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, blank)
    assert window.compare_list.currentRow() == -1
    assert window.compare_canvas._selected_compare_index is None
    # A click on a row still selects it.
    rect = window.compare_list.visualItemRect(window.compare_list.item(1))
    QTest.mouseClick(viewport, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, rect.center())
    assert window.compare_list.currentRow() == 1


def test_escape_in_the_compare_list_deselects(qapp, tmp_path):
    window = MainWindow()
    _compare_two(window, tmp_path)
    QTest.keyClick(window.compare_list, Qt.Key.Key_Escape)
    assert window.compare_list.currentRow() == -1
    assert window.compare_canvas._selected_compare_index is None


def test_unselecting_on_the_plot_lets_the_list_go_too(qapp, tmp_path):
    window = MainWindow()
    _compare_two(window, tmp_path)
    window.compare_canvas._toggle_compare_selection(window.compare_canvas._compare_axes[0])
    assert window.compare_canvas._selected_compare_index is None
    assert window.compare_list.currentRow() == -1


def test_an_export_never_shows_the_selection_tint(qapp, tmp_path):
    window = MainWindow()
    _compare_two(window, tmp_path)
    canvas = window.compare_canvas
    seen = []
    canvas.export_full_resolution(lambda: seen.append([ax.patch.get_alpha() for ax in canvas._compare_axes]))
    assert seen == [[0.0, 0.0]]
    assert canvas._compare_axes[0].patch.get_alpha() == 1.0      # back on screen afterwards
