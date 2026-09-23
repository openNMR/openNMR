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
import sys

import pytest


@pytest.fixture(autouse=True)
def _isolate_interface_settings(tmp_path, monkeypatch):
    """Keep the GUI's interface settings (Settings tab: font size,
    per-type style auto-apply) out of the real ~/.opennmr, and put the
    application font back after every test.

    The font is process-wide -- one QApplication serves the whole run
    -- so a test that enlarges it would otherwise enlarge every widget
    in every later test. Only touches the GUI if a test already
    imported it: non-GUI tests pay nothing."""
    main_window = sys.modules.get("opennmr.gui.main_window")
    if main_window is not None:
        monkeypatch.setattr(main_window.MainWindow, "_UI_SETTINGS_PATH",
                            str(tmp_path / "ui_settings.json"))
    app = None
    font = None
    if "PySide6.QtWidgets" in sys.modules:
        from PySide6.QtGui import QFont
        from PySide6.QtWidgets import QApplication

        app = QApplication.instance()
        font = QFont(app.font()) if app is not None else None
    yield
    if app is not None and font is not None:
        app.setFont(font)


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
    if "PySide6.QtWidgets" not in sys.modules:
        return  # no Qt test has run -- nothing to clean up
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
