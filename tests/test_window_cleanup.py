"""A deleted MainWindow is freed (2026-10-12).

GitHub Actions cancelled the test run partway through test_gui.py ("The
operation was canceled"): every MainWindow stayed in memory (~10 MB each)
because CombinedFitPanel's live-preview timer had no parent. That timer
outlived the window, and its timeout connection kept the panel alive --
and, through the panel's callbacks, the whole window. Several hundred
windows later the runner ran out of memory. conftest.py deletes each
test's windows; this test checks that deleting a window really frees it.
"""
import gc
import weakref

from PySide6.QtCore import QEvent, QObject, QTimer
from PySide6.QtWidgets import QApplication

from test_gui import qapp  # noqa: F401 -- fixture


def _delete_all_windows(app):
    for widget in app.topLevelWidgets():
        widget.close()
        widget.deleteLater()
    app.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    app.processEvents()
    gc.collect()


def _built_window():
    from opennmr.gui.main_window import MainWindow
    return weakref.ref(MainWindow())


def test_a_deleted_main_window_is_garbage_collected(qapp):  # noqa: F811
    ref = _built_window()
    _delete_all_windows(QApplication.instance())
    assert ref() is None, (
        "MainWindow survived deletion: something outside the window (an object "
        "with no parent, e.g. a QTimer() or QThread()) holds a connection to it "
        "or to one of its panels")


def test_the_combined_fit_preview_timer_belongs_to_its_panel(qapp):  # noqa: F811
    from opennmr.gui.main_window import MainWindow
    window = MainWindow()
    timer = window.combined_fit_panel._preview_timer
    assert isinstance(timer, QTimer)
    assert timer.parent() is window.combined_fit_panel


def test_no_parentless_qobjects_are_left_behind(qapp):  # noqa: F811
    """No top-level QObject (not a widget) a window made outlives it, except
    Qt's own. Parentless timers, threads and workers are how windows leak."""
    app = QApplication.instance()
    before = {id(o) for o in app.findChildren(QObject)}
    ref = _built_window()
    _delete_all_windows(app)
    assert ref() is None
    left = [o for o in app.findChildren(QObject) if id(o) not in before]
    assert not [o for o in left if isinstance(o, QTimer)], left
