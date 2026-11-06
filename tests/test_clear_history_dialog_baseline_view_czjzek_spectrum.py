"""Fixes of 2026-11-06:

* File ▸ Clear All History… crashed the Windows build natively inside
  its confirmation (a QMessageBox with ``setCheckBox``); it is now a
  plain dialog, opened once the menu has closed;
* a baseline correction no longer leaves the corrected trace under a
  kept y view (baseline halfway up, peaks cut off): the view follows
  the trace -- refitted if the whole trace was on show, the same crop
  of it if the user had zoomed;
* Simulate ▸ Czjzek shows the spectrum alone by default; the map is a
  choice (``layout="spectrum"`` in the notebook too).
"""

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication, QDialog, QDialogButtonBox

from opennmr.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
from opennmr.gui import progress as progress_dialogs
from opennmr.gui.main_window import MainWindow
from opennmr.gui.widgets.canvas import MplCanvas
from opennmr.notebook import czjzek as cz

from test_gui import (  # noqa: F401 -- qapp is a fixture
    _make_raw_2d_dataset,
    _write_params,
    qapp,
)


# ---- Clear All History -------------------------------------------------------------

def test_clear_history_dialog_keeps_notes_by_default_and_cancel_is_the_default(qapp):
    dialog = progress_dialogs.ClearHistoryDialog(
        None, label="expt", files=["/a/opennmr_history.json", "/a/opennmr_log.txt"])
    try:
        assert dialog.keep_notes() is True
        assert dialog.keep_notes_check.isVisibleTo(dialog)
        assert dialog.keep_notes_check.parent() is dialog     # owned from the start
        buttons = dialog.findChild(QDialogButtonBox)
        cancel = buttons.button(QDialogButtonBox.StandardButton.Cancel)
        assert cancel.isDefault() and not dialog.clear_button.isDefault()
        texts = " ".join(w.text() for w in dialog.findChildren(type(dialog.clear_button))
                         if hasattr(w, "text"))
        assert "Clear All History" in texts
        dialog.keep_notes_check.setChecked(False)
        assert dialog.keep_notes() is False
    finally:
        dialog.deleteLater()


def test_clear_history_dialog_without_a_folder_has_no_notes_choice(qapp):
    dialog = progress_dialogs.ClearHistoryDialog(None, label="sim", files=None)
    try:
        assert not dialog.keep_notes_check.isVisibleTo(dialog)
    finally:
        dialog.deleteLater()


def _derived(window):
    x = np.linspace(10, 0, 128)
    spectrum = NMRSpectrum(np.exp(-(x - 5) ** 2), [NMRAxis("frequency", ppm=x, hz=x * 400.0)],
                           NMRMetadata(vendor="simulated", nucleus="1H", sfo1=400.0))
    return window._add_derived_spectrum(spectrum, "derived")


def test_clear_history_click_runs_through_the_new_dialog(qapp, monkeypatch):
    window = MainWindow()
    _derived(window)
    calls = []
    monkeypatch.setattr(progress_dialogs.ClearHistoryDialog, "exec",
                        lambda self: QDialog.DialogCode.Accepted)
    monkeypatch.setattr(window, "_clear_all_history",
                        lambda index, keep_notes=True: calls.append((index, keep_notes)) or [])
    window._on_clear_history_clicked()
    assert calls == [(window._selected_index(), True)]


def test_clear_history_cancel_clears_nothing(qapp, monkeypatch):
    window = MainWindow()
    _derived(window)
    calls = []
    monkeypatch.setattr(progress_dialogs.ClearHistoryDialog, "exec",
                        lambda self: QDialog.DialogCode.Rejected)
    monkeypatch.setattr(window, "_clear_all_history",
                        lambda index, keep_notes=True: calls.append(index) or [])
    window._on_clear_history_clicked()
    assert calls == []


def test_clear_history_action_opens_after_the_menu_closes(qapp, monkeypatch):
    window = MainWindow()
    opened = []
    monkeypatch.setattr(window, "_on_clear_history_clicked", lambda: opened.append(True))
    window.clear_history_action.trigger()
    assert opened == []                  # not from inside the menu's dispatch
    for _ in range(5):
        QApplication.processEvents()
    assert opened == [True]


# ---- Baseline correction keeps the trace in view -----------------------------------

def _offset_spectrum(offset):
    x = np.linspace(10, 0, 512)
    y = offset + np.exp(-((x - 5.0) / 0.1) ** 2)
    return NMRSpectrum(y, [NMRAxis("frequency", ppm=x, hz=x * 400.0)],
                       NMRMetadata(vendor="simulated", nucleus="1H", sfo1=400.0))


def test_whole_trace_view_is_refitted_after_a_baseline_shift(qapp):
    canvas = MplCanvas()
    spectrum = _offset_spectrum(3.0)                 # drawn normalised: baseline at 0.75
    canvas.show_spectrum(spectrum)
    captured = canvas.capture_intensity_view()
    assert captured["whole"] and captured["level"] == pytest.approx(0.75, abs=0.01)
    spectrum.data = spectrum.data - 3.0               # corrected: baseline at 0
    canvas.update_data(spectrum)
    low_before_follow = canvas.ax.get_ylim()[0]
    assert low_before_follow > 0.1                    # the reported fault: baseline off-view
    assert canvas.follow_intensity_view(captured)
    low, high = canvas.ax.get_ylim()
    assert low < 0.0 < 1.0 < high                     # all of it on show again


def test_an_intensity_zoom_stays_on_the_same_part_of_the_trace(qapp):
    canvas = MplCanvas()
    spectrum = _offset_spectrum(3.0)
    canvas.show_spectrum(spectrum)
    canvas.ax.set_ylim(0.74, 0.80)                    # zoomed into the baseline
    captured = canvas.capture_intensity_view()
    assert not captured["whole"]
    spectrum.data = spectrum.data - 3.0
    canvas.update_data(spectrum)
    canvas.follow_intensity_view(captured)
    low, high = canvas.ax.get_ylim()
    old_fraction = (captured["level"] - 0.74) / (0.80 - 0.74)
    new_level = float(np.median(spectrum.normalized()))
    assert (new_level - low) / (high - low) == pytest.approx(old_fraction, abs=0.02)
    # The trace was rescaled by its own normalisation (0.25 -> 1 tall),
    # and the zoom with it: the same view of the same noise and wings.
    assert (high - low) == pytest.approx((0.80 - 0.74) * 4.0, rel=0.05)


def test_following_twice_is_the_same_view(qapp):
    canvas = MplCanvas()
    spectrum = _offset_spectrum(3.0)
    canvas.show_spectrum(spectrum)
    captured = canvas.capture_intensity_view()
    spectrum.data = spectrum.data - 3.0
    canvas.update_data(spectrum)
    canvas.follow_intensity_view(captured)
    first = canvas.ax.get_ylim()
    canvas.follow_intensity_view(captured)
    assert canvas.ax.get_ylim() == pytest.approx(first)


def test_no_intensity_axes_means_nothing_to_follow(qapp):
    canvas = MplCanvas()
    assert canvas.capture_intensity_view() is None
    assert canvas.follow_intensity_view(None) is False


def _offset_1d_dataset(tmp_path, n=512):
    expt = tmp_path / "offset1d"
    pdata = expt / "pdata" / "1"
    pdata.mkdir(parents=True)
    _write_params(expt / "acqus", {"NUC1": "1H", "BF1": 400.13, "SFO1": 400.13, "O1": 2000.0,
                                   "NS": 4})
    _write_params(pdata / "procs", {"SW_p": 8000.0, "SI": n, "SF": 400.13, "NC_proc": 0})
    i = np.arange(n)
    y = 300000 + 100000 * np.exp(-((i - n // 2) / 3.0) ** 2)
    y.astype("int32").tofile(pdata / "1r")
    np.zeros(n, dtype="int32").tofile(pdata / "1i")
    return str(expt)


def test_baseline_correct_on_a_1d_keeps_the_spectrum_in_view(qapp, tmp_path):
    window = MainWindow()
    window._load(_offset_1d_dataset(tmp_path), vendor="bruker")
    window.baseline_correct_button.setChecked(False)
    window.canvas.show_spectrum(window.session.spectra[window._selected_index()])
    window.baseline_correct_button.setChecked(True)
    y = window.session.spectra[window._selected_index()].normalized()
    low, high = window.canvas.ax.get_ylim()
    assert low <= float(np.min(y)) and high >= float(np.max(y))
    assert window._baseline_view_follow is None       # a 1D is done at once


def test_baseline_correct_on_a_pseudo2d_follows_at_the_settle(qapp, tmp_path, monkeypatch):
    window = MainWindow()
    window._load(_make_raw_2d_dataset(tmp_path, monkeypatch, n_f1=4), vendor="bruker")
    window.side_tabs.setCurrentIndex(window._processing_tab_index)
    window.apply_all_slices_checkbox.setChecked(True)
    window.baseline_correct_button.setChecked(True)
    # The recompute that applies it is still to come: the capture waits.
    assert window._baseline_view_follow is not None
    window._live_reprocess_timer.stop()
    window._run_debounced_multi_row_reprocess()
    assert window._baseline_view_follow is None


def test_a_new_selection_drops_a_pending_follow(qapp):
    window = MainWindow()
    _derived(window)
    second = _derived(window)
    window._baseline_view_follow = {"index": 0, "canvas": None, "preview": None}
    window._on_selection_changed(second)
    assert window._baseline_view_follow is None


# ---- Czjzek: the spectrum alone -----------------------------------------------------

def test_notebook_czjzek_plot_can_draw_the_spectrum_alone():
    x = np.linspace(50, -50, 256)
    fig = cz.plot_czjzek_distribution({"sigma": 1.0}, spectrum={"x": x, "y": np.exp(-x ** 2)},
                                      layout="spectrum")
    axes = fig.czjzek_axes
    assert axes["map"] is None and axes["colorbar"] is None
    assert axes["spectrum"] is not None and len(axes["spectrum"].get_lines()) == 1
    assert len(fig.axes) == 1
    assert "mathit" in axes["spectrum"].get_xlabel() or axes["spectrum"].get_xlabel()
    with pytest.raises(ValueError):
        cz.plot_czjzek_distribution({"sigma": 1.0}, layout="spectrum")


def test_czjzek_simulation_shows_the_spectrum_alone_by_default(qapp):
    window = MainWindow()
    sim = window.czjzek_sim_panel
    assert sim.layout_combo.currentData() == "spectrum"
    assert sim.layout_combo.currentIndex() == 0
    sim.grid_cq_spin.setValue(12)
    sim.grid_eta_spin.setValue(4)
    assert sim.simulate() is not None
    axes = window.simulation_canvas.figure.czjzek_axes
    assert axes["map"] is None and axes["spectrum"] is not None
    assert sim.plot_kind() == "compare"
    assert window.simulation_canvas.simulation_spectrum is not None   # Keep still works
    # The map is still one choice away.
    sim.layout_combo.setCurrentIndex(sim.layout_combo.findData("top"))
    sim.simulate()
    assert window.simulation_canvas.figure.czjzek_axes["map"] is not None
    assert sim.plot_kind() == "2d"


def test_czjzek_fit_map_choices_are_unchanged(qapp):
    window = MainWindow()
    fit = window.czjzek_fit_panel
    assert [fit.map_layout_combo.itemData(i) for i in range(fit.map_layout_combo.count())] \
        == ["right", "top"]
