"""Scanning a data folder in the background, from where the last scan
left off (explicit user request, 2026-11-13: opening openNMR with a
large default data folder took minutes, on older machines up to an hour).

The window never waits for a scan. :class:`FolderScanner` runs
:func:`opennmr.io.scan.scan_for_spectra` on its own thread with the
folder's :class:`~opennmr.io.scan.ScanIndex`, so only folders that
changed since last time are examined, and reports as it goes. The Open
panel shows the remembered list at once and brings it up to date when
the scan finishes.

The same pattern as :mod:`opennmr.gui.loader`: a plain ``QObject`` moved
to a ``QThread``; results come back as queued signals, so the GUI only
ever touches them on its own thread. ``OPENNMR_BACKGROUND_SCAN=0`` makes
scans synchronous (tests pin it, as they do loads).
"""

from __future__ import annotations

import os
import threading

from PySide6.QtCore import Q_ARG, QMetaObject, QObject, Qt, QThread, Signal, Slot

from ..io import scan as io_scan

BACKGROUND_SCAN_ENV = "OPENNMR_BACKGROUND_SCAN"


def background_scan_enabled() -> bool:
    """Whether folder scans run off the GUI thread (on unless the
    environment says otherwise)."""
    value = os.environ.get(BACKGROUND_SCAN_ENV)
    if value is None:
        return True
    return value.strip().lower() not in ("0", "false", "no", "off")


def run_scan(root: str, *, full: bool = False, max_matches: int | None = None,
             cancel=None, progress=None) -> io_scan.ScanResult:
    """One scan of ``root`` with its remembered index, saved afterwards.
    ``full``: ignore what is remembered and examine everything again
    (the index is rebuilt). The same on either thread."""
    index = io_scan.ScanIndex(root) if full else io_scan.ScanIndex.load(root)
    result = io_scan.scan_for_spectra(root, max_matches=max_matches, index=index,
                                      cancel=cancel, progress=progress)
    index.save(result)
    return result


class _ScanWorker(QObject):
    progress = Signal(int, str, int, object)     # request, root, folders, found so far
    finished = Signal(int, str, object)          # request, root, ScanResult
    failed = Signal(int, str, str)

    def __init__(self) -> None:
        super().__init__()
        self._cancelled: dict[int, threading.Event] = {}
        self._lock = threading.Lock()

    def event_for(self, request: int) -> threading.Event:
        with self._lock:
            return self._cancelled.setdefault(request, threading.Event())

    def cancel_all(self) -> None:
        with self._lock:
            for event in self._cancelled.values():
                event.set()

    @Slot(int, str, bool, int)
    def run(self, request: int, root: str, full: bool, max_matches: int) -> None:
        event = self.event_for(request)
        if event.is_set():
            self.finished.emit(request, root, None)
            return

        def progress(folders, found) -> None:
            self.progress.emit(request, root, int(folders), list(found))

        try:
            result = run_scan(root, full=full, max_matches=max_matches,
                              cancel=event.is_set, progress=progress)
        except Exception as exc:  # noqa: BLE001 -- reported, never kills the thread
            self.failed.emit(request, root, str(exc))
        else:
            self.finished.emit(request, root, result)
        finally:
            with self._lock:
                self._cancelled.pop(request, None)


class FolderScanner(QObject):
    """One background thread for folder scans; a new scan stops the one
    before it. Signals arrive on the GUI thread and carry the root, so a
    late answer about an earlier folder can be told apart."""

    progress = Signal(str, int, object)      # root, folders examined, found so far
    finished = Signal(str, object)           # root, ScanResult (cancelled=True if stopped)
    failed = Signal(str, str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._thread = QThread()
        self._worker = _ScanWorker()
        self._worker.moveToThread(self._thread)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._request = 0
        self._root: str | None = None
        self._thread.start()

    def is_scanning(self) -> bool:
        return self._root is not None

    def scanning_root(self) -> str | None:
        return self._root

    def scan(self, root: str, *, full: bool = False, max_matches: int | None = None) -> None:
        """Start scanning ``root`` (stopping any scan in progress)."""
        self.cancel()
        self._request += 1
        self._root = root
        limit = int(max_matches if max_matches is not None else io_scan.SCAN_LIMIT)
        self._worker.event_for(self._request)
        QMetaObject.invokeMethod(
            self._worker, "run", Qt.ConnectionType.QueuedConnection,
            Q_ARG(int, self._request), Q_ARG(str, root), Q_ARG(bool, bool(full)),
            Q_ARG(int, limit))

    def cancel(self) -> None:
        """Stop the scan in progress; it answers with a cancelled result."""
        self._worker.cancel_all()

    def shutdown(self) -> None:
        """Stop the thread and wait for it. Safe to call twice."""
        self._worker.cancel_all()
        if self._thread.isRunning():
            self._thread.quit()
            self._thread.wait()

    @Slot(int, str, int, object)
    def _on_progress(self, request, root, folders, found) -> None:
        if request == self._request:
            self.progress.emit(root, folders, found)

    @Slot(int, str, object)
    def _on_finished(self, request, root, result) -> None:
        if request != self._request:
            return
        self._root = None
        if result is None:
            result = io_scan.ScanResult()
            result.cancelled = True
        self.finished.emit(root, result)

    @Slot(int, str, str)
    def _on_failed(self, request, root, message) -> None:
        if request != self._request:
            return
        self._root = None
        self.failed.emit(root, message)
