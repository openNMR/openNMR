"""The crash log: every unexpected error, written where a user can find it.

Why
---
The installers build a *windowed* app (PyInstaller ``--windowed``): there is
no console, ``sys.stderr`` is ``None``, and an error raised in a button's
handler -- or a hard crash inside Qt or numpy -- used to vanish without a
trace (stability audit, 2026-11-01). :func:`install` makes each one leave a
record in ``~/.opennmr/logs/crash.log``:

* **Python errors** anywhere on the GUI thread (``sys.excepthook``; PySide6
  sends errors raised in slots there) and in worker threads
  (``threading.excepthook``): the traceback, with the openNMR, Python, Qt
  and platform versions. On the GUI thread the user is also told, once per
  burst, in a non-blocking message with the details and a button that opens
  the log folder -- the action may not have finished, and they should know.
* **Hard crashes** (segmentation faults, aborts) via :mod:`faulthandler`,
  which writes every thread's Python stack into the same file before the
  process dies.
* **Qt's own critical and fatal messages** (``qInstallMessageHandler``).

Nothing is written when nothing goes wrong. The file is kept below
:data:`MAX_BYTES`; when it grows past that it is renamed ``crash.log.1``
(replacing an older one) and a new one is started.

Why a module of its own, outside the ``opennmr`` package: the hook must be in
place before ``import opennmr`` -- several seconds of numpy/scipy/matplotlib
imports, any of which can fail on a broken install -- and importing anything
inside the package runs that first. Like :mod:`opennmr_launch` it imports
only the standard library (PySide6 lazily, if it is already loaded).

``OPENNMR_LOG_DIR`` moves the log (tests, portable installs);
``OPENNMR_CRASH_DIALOG=0`` keeps the record but shows no message;
``OPENNMR_CRASH_LOG=0`` turns the whole thing off (the test suite, so pytest
keeps its own fault handler).
"""

from __future__ import annotations

import datetime
import faulthandler
import os
import platform
import sys
import threading
import time
import traceback

#: Environment variable that moves the log folder.
LOG_DIR_ENV = "OPENNMR_LOG_DIR"
#: Set to "0" to install nothing at all (the test suite).
ENABLE_ENV = "OPENNMR_CRASH_LOG"
#: Set to "0" to log errors without telling the user (tests, batch runs).
DIALOG_ENV = "OPENNMR_CRASH_DIALOG"
#: The log's file name inside :func:`log_dir`.
LOG_NAME = "crash.log"
#: Above this size the log is rotated to ``crash.log.1`` at start-up.
MAX_BYTES = 1_000_000
#: Seconds after one error message during which further errors are only logged
#: (a failure inside a redraw can repeat on every paint).
DIALOG_QUIET_S = 10.0

_state = {"installed": False, "file": None, "previous": {}, "last_dialog": 0.0,
          "dialog": None, "count": 0}
_lock = threading.Lock()


def log_dir() -> str:
    """Folder the crash log lives in (created on first write)."""
    return os.environ.get(LOG_DIR_ENV) or os.path.join(os.path.expanduser("~"), ".opennmr", "logs")


def log_path() -> str:
    """Full path of the crash log."""
    return os.path.join(log_dir(), LOG_NAME)


def _version() -> str:
    try:
        from importlib import metadata
        return metadata.version("opennmr")
    except Exception:  # noqa: BLE001 -- a version is a nicety, never a reason to fail
        return "unknown"


def _qt_version() -> str:
    mod = sys.modules.get("PySide6")
    return getattr(mod, "__version__", "") if mod is not None else ""


def environment_text() -> str:
    """One line: openNMR, Python, Qt and platform versions, frozen or not."""
    qt = _qt_version()
    return (f"openNMR {_version()} | Python {platform.python_version()} | "
            + (f"PySide6 {qt} | " if qt else "")
            + f"{platform.platform()} | " + ("installed app" if getattr(sys, "frozen", False)
                                              else "from source/pip"))


def _rotate(path: str) -> None:
    try:
        if os.path.getsize(path) > MAX_BYTES:
            os.replace(path, path + ".1")
    except OSError:
        pass


def _open_log():
    """The log, opened for appending (rotated first if too large); None if the
    folder cannot be written."""
    try:
        os.makedirs(log_dir(), exist_ok=True)
        path = log_path()
        _rotate(path)
        return open(path, "a", encoding="utf-8", errors="replace")
    except OSError:
        return None


def write_entry(kind: str, text: str) -> str | None:
    """Append one dated entry to the log; returns its path (None if the log
    could not be written). Safe to call from any thread."""
    stamp = datetime.datetime.now().isoformat(timespec="seconds")
    entry = (f"\n===== {stamp} | {kind} =====\n{environment_text()}\n"
             + text.rstrip() + "\n")
    with _lock:
        handle = _state["file"]
        try:
            if handle is None or handle.closed:
                handle = _open_log()
                _state["file"] = handle
            if handle is None:
                return None
            handle.write(entry)
            handle.flush()
            _state["count"] += 1
            return log_path()
        except (OSError, ValueError):
            return None


def _format(exc_type, exc, tb) -> str:
    return "".join(traceback.format_exception(exc_type, exc, tb))


def _excepthook(exc_type, exc, tb):
    if issubclass(exc_type, KeyboardInterrupt):
        _state["previous"]["excepthook"](exc_type, exc, tb)
        return
    text = _format(exc_type, exc, tb)
    path = write_entry("Unhandled error", text)
    previous = _state["previous"].get("excepthook")
    if previous is not None and sys.stderr is not None:
        try:
            previous(exc_type, exc, tb)
        except Exception:  # noqa: BLE001
            pass
    _tell_user(exc_type, exc, text, path)


def _threading_hook(args):
    if args.exc_type is SystemExit:
        return
    name = getattr(args.thread, "name", "?")
    write_entry(f"Unhandled error in thread {name!r}",
                _format(args.exc_type, args.exc_value, args.exc_traceback))
    previous = _state["previous"].get("threading")
    if previous is not None and sys.stderr is not None:
        try:
            previous(args)
        except Exception:  # noqa: BLE001
            pass


def _qt_message_handler(mode, context, message):
    try:
        from PySide6.QtCore import QtMsgType
        serious = mode in (QtMsgType.QtCriticalMsg, QtMsgType.QtFatalMsg)
    except Exception:  # noqa: BLE001
        serious = False
    if serious:
        where = ""
        if context is not None and getattr(context, "file", None):
            where = f" ({context.file}:{context.line})"
        write_entry("Qt critical message", f"{message}{where}")
    if sys.stderr is not None:
        try:
            sys.stderr.write(f"{message}\n")
        except Exception:  # noqa: BLE001
            pass


def _dialog_wanted() -> bool:
    return os.environ.get(DIALOG_ENV, "1").strip() != "0"


def _tell_user(exc_type, exc, text, path) -> None:
    """A non-blocking message on the GUI thread, at most one per
    :data:`DIALOG_QUIET_S` and never two at once."""
    if not _dialog_wanted() or threading.current_thread() is not threading.main_thread():
        return
    qtwidgets = sys.modules.get("PySide6.QtWidgets")
    if qtwidgets is None or qtwidgets.QApplication.instance() is None:
        return
    now = time.monotonic()
    shown = _state["dialog"]
    if now - _state["last_dialog"] < DIALOG_QUIET_S:
        return
    try:
        if shown is not None and shown.isVisible():
            return
    except RuntimeError:            # the C++ object is gone
        pass
    _state["last_dialog"] = now
    try:
        _state["dialog"] = show_error_message(exc_type, exc, text, path)
    except Exception:  # noqa: BLE001 -- the report must never raise in its turn
        _state["dialog"] = None


def show_error_message(exc_type, exc, text, path):
    """The "something went wrong" message: modeless, with the traceback as
    details and a button that opens the log folder. Returns the box."""
    from PySide6.QtCore import Qt, QUrl
    from PySide6.QtGui import QDesktopServices
    from PySide6.QtWidgets import QApplication, QMessageBox

    parent = QApplication.activeWindow()
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Warning)
    box.setWindowTitle("openNMR – unexpected error")
    summary = f"{exc_type.__name__}: {exc}".strip()
    if len(summary) > 300:
        summary = summary[:300] + "…"
    where = (f"The details were saved to\n{path}\n\nPlease send that file, with a note of "
             "what you were doing, when you report this." if path else
             "The crash log could not be written (the folder is not writable).")
    box.setText("Something went wrong, and the last action may not have finished.\n\n"
                + summary)
    box.setInformativeText(where)
    box.setDetailedText(text)
    box.setStandardButtons(QMessageBox.StandardButton.Close)
    folder = box.addButton("Open Log Folder", QMessageBox.ButtonRole.ActionRole)
    folder.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(log_dir())))
    box.setWindowModality(Qt.WindowModality.NonModal)
    box.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
    box.show()
    return box


def install(qt: bool = True) -> bool:
    """Start recording. Idempotent; returns True the first time.

    ``qt``: also route Qt's critical/fatal messages here (only if PySide6 is
    importable). Call as early as possible -- :func:`opennmr_launch.main`
    does it before the splash."""
    if _state["installed"] or os.environ.get(ENABLE_ENV, "1").strip() == "0":
        return False
    _state["installed"] = True
    _state["previous"] = {"excepthook": sys.excepthook or sys.__excepthook__,
                          "threading": threading.excepthook}
    sys.excepthook = _excepthook
    threading.excepthook = _threading_hook
    handle = _open_log()
    _state["file"] = handle
    if handle is not None:
        try:
            # Native crashes: written straight to the file descriptor as the
            # process dies, so the handle stays open for the whole session.
            faulthandler.enable(file=handle, all_threads=True)
        except (OSError, ValueError, RuntimeError):
            pass
    if qt:
        try:
            from PySide6.QtCore import qInstallMessageHandler
            _state["previous"]["qt"] = qInstallMessageHandler(_qt_message_handler)
        except Exception:  # noqa: BLE001 -- PySide6 missing: Python errors are still logged
            pass
    return True


def uninstall() -> None:
    """Undo :func:`install` (tests)."""
    if not _state["installed"]:
        return
    sys.excepthook = _state["previous"].get("excepthook", sys.__excepthook__)
    threading.excepthook = _state["previous"].get("threading", threading.__excepthook__)
    try:
        faulthandler.disable()
    except Exception:  # noqa: BLE001
        pass
    if "qt" in _state["previous"]:
        try:
            from PySide6.QtCore import qInstallMessageHandler
            qInstallMessageHandler(_state["previous"]["qt"])
        except Exception:  # noqa: BLE001
            pass
    with _lock:
        if _state["file"] is not None:
            try:
                _state["file"].close()
            except OSError:
                pass
    _state.update(installed=False, file=None, previous={}, last_dialog=0.0, dialog=None,
                  count=0)
