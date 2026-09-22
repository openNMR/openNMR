import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# When another Qt binding (e.g. PyQt6) is also installed in this
# environment, its Qt plugins can shadow PySide6's own -- at a
# different Qt version, which then fails to load at all ("Could not
# find the Qt platform plugin"). Point Qt at PySide6's own bundled
# plugins explicitly, before anything imports PySide6.
import PySide6

os.environ.setdefault(
    "QT_PLUGIN_PATH", os.path.join(os.path.dirname(PySide6.__file__), "Qt", "plugins"),
)

# Loads run synchronously under test unless a test says otherwise.
# The threading contract requires this to be pinned rather than
# inherited: an async load lands via the Qt event loop, so a test that
# asserts straight after opening a file would race it -- passing on a
# fast machine and flaking on a slow one. tests/test_background_load.py
# exercises the threaded path deliberately, driving the event loop.
os.environ.setdefault("OPENNMR_BACKGROUND_LOAD", "0")

import matplotlib

matplotlib.use("Agg")

import gc

import pytest


@pytest.fixture(autouse=True)
def _delete_leftover_windows():
    """Delete every top-level widget a test leaves behind.

    Without this, each MainWindow outlives its test: its signal
    connections hold Python references from the C++ side, where the
    garbage collector cannot see them, so ~8.5 MB per window is never
    freed. test_gui.py builds several hundred windows, and that piles up
    past the ~7 GB of a GitHub Actions runner, which then kills the job
    ("The operation was canceled") partway through the suite.
    Deleting the C++ side drops those references; measured at ~0.6 MB
    retained per window afterwards."""
    yield
    from PySide6.QtCore import QEvent
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        return
    for widget in app.topLevelWidgets():
        widget.close()
        widget.deleteLater()
    app.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    app.processEvents()
    gc.collect()
