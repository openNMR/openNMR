import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# Qt's offscreen platform does not find Windows' own fonts ("QFontDatabase:
# Cannot find font directory ... Qt no longer ships fonts"), and lays text
# out with fallback metrics -- side panels then measured far wider than
# on screen (test_gui_reflow failed only on Windows, 2026-10-16).
if os.name == "nt":
    _fonts = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")
    if os.path.isdir(_fonts):
        os.environ.setdefault("QT_QPA_FONTDIR", _fonts)

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
# Folder scans too (gui/folder_scan.py); tests/test_folder_scan.py runs
# the background path itself.
os.environ.setdefault("OPENNMR_BACKGROUND_SCAN", "0")
# And long fits and simulations (gui/jobs.py): run at once, with the modal
# pop-up, as before; tests/test_background_jobs.py runs them in the
# background itself.
os.environ.setdefault("OPENNMR_BACKGROUND_JOBS", "0")

# No update check at start-up under test: a MainWindow is built
# thousands of times, and none of them should touch the network.
# tests/test_gui_updates.py drives the check with a patched source.
os.environ.setdefault("OPENNMR_UPDATE_CHECK", "0")

# No crash log under test: tests call opennmr_launch.main(), which would
# otherwise take over sys.excepthook and pytest's own fault handler, and
# write to the real ~/.opennmr/logs. tests/test_crash_log.py turns it on
# for itself, into a temporary folder.
os.environ.setdefault("OPENNMR_CRASH_LOG", "0")

# No run-time estimate before Solid-state NMR simulations and fits (and so
# never a "this will take ..." question waiting for an answer); the
# estimate's own tests (test_run_time_estimates.py) turn it back on.
os.environ.setdefault("OPENNMR_CONFIRM_LONG_RUNS", "0")

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
    # Simulation runs are journalled (io.history.record_simulation) in
    # the simulations folder; never the real ~/Documents during tests.
    monkeypatch.setenv("OPENNMR_SIMULATIONS_DIR", str(tmp_path / "simulations"))
    # Cmd/Ctrl+E's fact list: never the user's own ~/.opennmr/facts.txt.
    monkeypatch.setenv("OPENNMR_FACTS_FILE", str(tmp_path / "facts.txt"))
    # The update helper's log and result (read, and removed, when Settings
    # opens): never the user's own, which a test run would otherwise consume.
    monkeypatch.setenv("OPENNMR_STATE_DIR", str(tmp_path / "update_state"))
    # Saved copies of slow reads (io.load_cache): never the user's own
    # cache folder, and never one test's copy served to another.
    monkeypatch.setenv("OPENNMR_CACHE_DIR", str(tmp_path / "load_cache"))
    # Remembered folder scans (io.scan.ScanIndex): never the user's own.
    monkeypatch.setenv("OPENNMR_SCAN_INDEX_DIR", str(tmp_path / "scan_index"))
    # Autosave and crash recovery (io.recovery): never the user's own
    # recovery folder -- a test window would otherwise take a running
    # app's lock for a crashed session and move its autosave aside.
    monkeypatch.setenv("OPENNMR_RECOVERY_DIR", str(tmp_path / "recovery"))
    main_window = sys.modules.get("opennmr.gui.main_window")
    if main_window is not None:
        monkeypatch.setattr(main_window.MainWindow, "_UI_SETTINGS_PATH",
                            str(tmp_path / "ui_settings.json"))
        # The saved per-type styles, Plot Style config and probe config too:
        # read from the real ~/.opennmr they made results depend on whoever
        # ran the tests (a saved default changed what Plot Style showed --
        # test_per_type_default_follows_the_plot_on_show, 2026-10-16), and
        # a test that saves one would overwrite the user's own.
        for attr, name in (("_FIGURE_STYLES_PATH", "figure_styles.json"),
                           ("_PLOT_STYLE_CONFIG_PATH", "plot_style.json"),
                           ("_PROBE_CONFIG_PATH", "probe_config.json")):
            monkeypatch.setattr(main_window.MainWindow, attr, str(tmp_path / name))
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
    Deleting the C++ side drops those references; measured at ~0.4 MB
    retained per window afterwards -- *if* nothing outside the window
    holds one of its objects. A parentless QObject (a ``QTimer()`` made
    before ``QWidget.__init__``, a ``QThread()``) is not deleted with it,
    and its signal connections keep the window alive: that brought the
    leak back at ~10 MB per window in October 2026 (CombinedFitPanel's
    preview timer). tests/test_window_cleanup.py checks a deleted window
    is really freed."""
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


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "full_2d_autophase_prompt: let Autophase on a 2-D show its real "
        "confirmation (tests patch QMessageBox.question themselves)")


@pytest.fixture(autouse=True)
def _confirm_full_2d_autophase(request, monkeypatch):
    """Answer *Yes* to the "auto-phasing a 2-D spectrum can take a long
    time" question, so tests written before it existed keep exercising
    the full 2-D autophase -- an unanswered modal would hang a headless
    run. Tests about the question itself opt out with
    ``@pytest.mark.full_2d_autophase_prompt``."""
    main_window = sys.modules.get("opennmr.gui.main_window")
    if main_window is not None and request.node.get_closest_marker("full_2d_autophase_prompt") is None:
        monkeypatch.setattr(main_window.MainWindow, "_confirm_full_2d_autophase", lambda self: True)
    yield


@pytest.fixture(autouse=True)
def _no_blocking_dialogs(monkeypatch):
    """Never let a test wait on a dialog nobody can answer.

    Under the offscreen platform a real ``QMessageBox.critical`` (or a file
    or input dialog) never returns: an error the code reports that way
    froze the whole run instead of failing one test (2026-10-16: a curve-fit
    export failing on Windows hung at ``QMessageBox.critical``). Here every
    such dialog returns at once -- message boxes are printed to stderr, so
    the failing test's output says what the dialog said; file and input
    dialogs return "cancelled". A test that wants a particular answer
    patches the dialog itself, which overrides this."""
    if "PySide6.QtWidgets" not in sys.modules:
        yield
        return
    from PySide6.QtWidgets import QFileDialog, QInputDialog, QMessageBox

    def _box(kind):
        def show(*args, **kwargs):
            title = args[1] if len(args) > 1 else kwargs.get("title", "")
            text = args[2] if len(args) > 2 else kwargs.get("text", "")
            print(f"[QMessageBox.{kind}] {title}: {text}", file=sys.stderr)
            if kind == "question":
                default = args[4] if len(args) > 4 else kwargs.get("defaultButton")
                return default if default not in (None, QMessageBox.StandardButton.NoButton) \
                    else QMessageBox.StandardButton.No
            return QMessageBox.StandardButton.Ok
        return staticmethod(show)

    for kind in ("critical", "warning", "information", "question"):
        monkeypatch.setattr(QMessageBox, kind, _box(kind))
    for name in ("getSaveFileName", "getOpenFileName"):
        monkeypatch.setattr(QFileDialog, name, staticmethod(lambda *a, **k: ("", "")))
    monkeypatch.setattr(QFileDialog, "getOpenFileNames", staticmethod(lambda *a, **k: ([], "")))
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", staticmethod(lambda *a, **k: ""))
    for name in ("getText", "getItem"):
        monkeypatch.setattr(QInputDialog, name, staticmethod(lambda *a, **k: ("", False)))
    for name in ("getDouble", "getInt"):
        monkeypatch.setattr(QInputDialog, name, staticmethod(lambda *a, **k: (0, False)))
    yield
