"""2026-11-22: drag-and-drop onto the plot -- an experiment dragged from Open experiments, or a
file / folder dragged from the file explorer. Spectrum view: show it. Compare: add it to the overlay."""
import os

import numpy as np
import pytest
from PySide6.QtWidgets import QMessageBox
from PySide6.QtCore import QEvent, QMimeData, QPointF, QUrl, Qt
from PySide6.QtGui import QDragEnterEvent, QDropEvent

from opennmr.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
from opennmr.gui.main_window import MainWindow

from test_gui import qapp  # noqa: F401 -- qapp is a fixture

PPM = np.linspace(10.0, -10.0, 200)


def _spec(k, path):
    md = NMRMetadata(vendor="simulated", nucleus="1H", sfo1=400.0, bf1=400.0)
    md.source_path = path
    return NMRSpectrum(data=np.exp(-((PPM - k) / 0.5) ** 2),
                       axes=[NMRAxis(kind="frequency", ppm=PPM, hz=PPM * 400.0)], metadata=md)


@pytest.fixture(autouse=True)
def _no_dialogs(monkeypatch):
    for name in ("warning", "information", "critical"):
        monkeypatch.setattr(QMessageBox, name, lambda *a, **k: None)


def _window(tmp_path):
    w = MainWindow()
    for k in range(3):
        d = tmp_path / str(k + 1)
        d.mkdir()
        w._add_derived_spectrum(_spec(k, str(d)), f"exp {k + 1}")
        # derived spectra do not keep a folder; give each the one a real load would
        w.session.spectra[-1].metadata.source_path = str(d)
    return w


def _compare_entries(w):
    return [w.compare_list.item(i).data(Qt.ItemDataRole.UserRole) for i in range(w.compare_list.count())]


def test_both_main_plots_accept_dropped_files_and_experiments(qapp, tmp_path):
    w = _window(tmp_path)
    assert w.canvas.accept_row_drops and not w.canvas.accept_file_drops   # files: the window opens them
    assert w.compare_canvas.accept_row_drops and w.compare_canvas.accept_file_drops


def test_other_canvases_leave_file_drops_to_the_window(qapp):
    w = MainWindow()
    assert not w.fid_canvas.accept_file_drops and not w.fid_canvas.accept_row_drops


def _url_event(kind, mime):
    cls = QDragEnterEvent if kind == QEvent.Type.DragEnter else QDropEvent
    return cls(QPointF(10, 10).toPoint(), Qt.DropAction.CopyAction, mime,
               Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)


def test_url_drop_on_the_plot_emits_paths(qapp, tmp_path):
    w = _window(tmp_path)
    got = []
    w.compare_canvas.pathsDropped.connect(got.append)
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(tmp_path))])
    ev = _url_event(QEvent.Type.DragEnter, mime)
    assert w.compare_canvas.eventFilter(w.compare_canvas.canvas, ev) is True and ev.isAccepted()
    drop = _url_event(QEvent.Type.Drop, mime)
    w.compare_canvas.eventFilter(w.compare_canvas.canvas, drop)
    qapp.processEvents()
    from PySide6.QtTest import QTest
    QTest.qWait(30)
    assert [[os.path.normpath(p) for p in paths] for paths in got] == [[os.path.normpath(str(tmp_path))]]


def test_dragged_rows_are_added_to_compare(qapp, tmp_path):
    w = _window(tmp_path)
    w._on_experiments_dropped_on_compare([0, 2], -1)
    assert _compare_entries(w) == [(0, None), (2, None)]


def test_dropped_rows_on_the_spectrum_plot_select_them(qapp, tmp_path):
    w = _window(tmp_path)
    w._on_rows_dropped_on_spectrum_plot([1, 2])
    selected = {i for i in range(w.spectrum_list.count()) if w.spectrum_list.item(i).isSelected()}
    assert selected == {1, 2}
    w._on_rows_dropped_on_spectrum_plot([99])                # out of range: ignored
    assert {i for i in range(w.spectrum_list.count()) if w.spectrum_list.item(i).isSelected()} == {1, 2}


def test_a_folder_already_open_dropped_on_compare_is_added(qapp, tmp_path):
    w = _window(tmp_path)
    w._on_paths_dropped_on_compare([str(tmp_path / "2")])
    assert _compare_entries(w) == [(1, None)]
    assert not getattr(w, "_compare_after_open", [])


def test_a_parent_folder_adds_every_open_experiment_under_it(qapp, tmp_path):
    """What happens once a dropped folder has finished opening (the add-when-loaded step)."""
    w = _window(tmp_path)
    w._compare_after_open = [str(tmp_path)]
    w._add_opened_paths_to_compare()
    assert sorted(_compare_entries(w)) == [(0, None), (1, None), (2, None)]
    assert w._compare_after_open == []


def test_nonexistent_paths_are_ignored(qapp, tmp_path):
    w = _window(tmp_path)
    w._on_paths_dropped_on_compare([str(tmp_path / "nope")])
    assert _compare_entries(w) == []
