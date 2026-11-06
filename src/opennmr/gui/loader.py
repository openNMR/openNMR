"""
Reading a spectrum off the GUI thread.

The first concurrency in this project outside the FFT worker pool.
What it buys is **responsiveness, not speed**: a large load spends
roughly half its time in ``plot_2d_contour``, and matplotlib figure and
Qt widget work must stay on the GUI thread, so moving the *read* off it
leaves the window alive and cancellable without making anything
finish sooner. See TODO.md's threading plan for the measurements.

Design points, each with the reason:

* **Threads, never ``multiprocessing``.** ``spawn`` (macOS, Windows)
  re-imports the parent module and would need every spectrum to pickle
  across a process boundary; it also behaves differently again inside
  a PyInstaller bundle, which this project already builds. numpy and
  scipy release the GIL during their C-level work, so threads deliver
  here.
* **A plain ``QObject`` moved with ``moveToThread``**, not a ``QThread``
  subclass overriding ``run``. The worker stays an ordinary object that
  can be called directly in a test with no thread at all.
* **The worker only reads.** It calls :func:`opennmr.io.load` and hands
  the finished spectrum back by signal. It never touches
  :class:`~opennmr.gui.session.Session`, whose lists the GUI thread
  mutates, and never touches a widget or a figure.
* **Cancellation is between stages, not mid-FFT.** The reader reports
  its stages (:mod:`opennmr.core.load_progress`) and a cancelled read
  stops at the next one; interrupting a running transform is not
  honest, so a stage already under way finishes first.
* **Slow reads are kept** (:mod:`opennmr.io.load_cache`): opening the
  same experiment, read the same way, again uses the saved copy.
"""

from __future__ import annotations

import os
import threading

from PySide6.QtCore import Q_ARG, QMetaObject, QObject, QThread, Qt, Signal, Slot

from .. import io
from ..core import load_progress
from ..io import load_cache

#: Set to ``0``/``false``/``no`` to make every load synchronous. Worth
#: having for CI determinism and for debugging a load that misbehaves --
#: the synchronous path is the same code, just called directly.
BACKGROUND_LOAD_ENV = "OPENNMR_BACKGROUND_LOAD"


def background_load_enabled() -> bool:
    """Whether loads should run off the GUI thread.

    On unless the environment says otherwise. An unparseable value is
    treated as "on" rather than raising -- a bad environment variable
    should not stop a spectrum loading, the same rule
    :func:`opennmr.processing.ft.fft_workers` follows."""
    value = os.environ.get(BACKGROUND_LOAD_ENV)
    if value is None:
        return True
    return value.strip().lower() not in ("0", "false", "no", "off")


class SpectrumReader(QObject):
    """Reads one spectrum, off the GUI thread.

    Emits exactly one of :attr:`finished` or :attr:`failed` per
    :meth:`read` call -- never both, and never neither, so a caller can
    always tear down after either.
    """

    #: ``(path, vendor, spectrum)`` -- the spectrum is handed over, not
    #: added to any session; the GUI thread owns that.
    finished = Signal(str, str, object)
    #: ``(path, vendor, message)``. The message, not the exception, so
    #: nothing exception-shaped crosses the thread boundary.
    failed = Signal(str, str, str)
    #: ``(path, text)``: the stage the read has reached.
    progress = Signal(str, str)
    #: ``(path, vendor)``: a read cancelled by request (see
    #: :meth:`BackgroundLoader.cancel_pending`), emitted instead of the others.
    cancelled = Signal(str, str)
    #: ``(path, info)`` just before :attr:`finished`: ``info`` from
    #: :func:`opennmr.io.load_cache.load` (seconds, from_copy ...).
    timing = Signal(str, object)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._cancelled = False
        #: Requests to drop, by id (see BackgroundLoader). Shared with the
        #: GUI thread, hence the lock.
        self._dropped: set[int] = set()
        self._lock = threading.Lock()

    def _is_dropped(self, request) -> bool:
        if request is None:
            return False
        with self._lock:
            return request in self._dropped

    def cancel(self) -> None:
        """Ask for the in-flight read to be discarded.

        Checked between stages -- a read already inside a transform
        runs to completion and its result is dropped, which is what
        "cancel" can honestly mean without interrupting numpy
        mid-operation."""
        self._cancelled = True

    @Slot(str, str, "QVariant")
    def read(self, path: str, vendor: str, kwargs: dict | None = None) -> None:
        """Load ``path`` and emit the result. Never raises: a failure
        becomes :attr:`failed`, because an exception escaping a slot on
        a worker thread has nowhere useful to go."""
        kwargs = dict(kwargs or {})
        request = kwargs.pop("_request_id", None)
        if self._cancelled:
            return
        if self._is_dropped(request):
            self.cancelled.emit(path, vendor)
            return

        def report(text: str) -> None:
            if self._cancelled or self._is_dropped(request):
                raise load_progress.LoadCancelled()
            self.progress.emit(path, text)

        try:
            with load_progress.reporting(report):
                spectrum, info = load_cache.load(path, vendor, kwargs)
        except load_progress.LoadCancelled:
            if not self._cancelled:
                self.cancelled.emit(path, vendor)
            return
        except Exception as exc:  # noqa: BLE001 -- reported, not raised, see docstring
            if not self._cancelled:
                self.failed.emit(path, vendor, str(exc))
            return
        if self._cancelled:
            return
        if self._is_dropped(request):
            self.cancelled.emit(path, vendor)
            return
        self.timing.emit(path, info)
        self.finished.emit(path, vendor, spectrum)


class BackgroundLoader(QObject):
    """Owns the worker thread and its lifecycle.

    One long-lived thread reused across loads rather than one per load:
    thread creation is not free, and a single owned thread is far
    easier to shut down correctly. :meth:`shutdown` must be called
    before the owning window is destroyed, or Qt prints ``QThread:
    Destroyed while thread is still running`` and can abort the
    process on exit.
    """

    #: Re-emitted from the worker, on the GUI thread.
    finished = Signal(str, str, object)
    failed = Signal(str, str, str)
    progress = Signal(str, str)
    cancelled = Signal(str, str)
    timing = Signal(str, object)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._thread = QThread()
        self._reader = SpectrumReader()
        self._reader.moveToThread(self._thread)
        self._reader.finished.connect(self.finished)
        self._reader.failed.connect(self.failed)
        self._reader.progress.connect(self.progress)
        self._reader.cancelled.connect(self.cancelled)
        self._reader.timing.connect(self.timing)
        self._next_request = 0
        self._queued: set[int] = set()
        # Bound slots of this (GUI-thread) object: queued to the GUI
        # thread, so _queued is only ever touched there.
        self._reader.finished.connect(self._settle)
        self._reader.failed.connect(self._settle)
        self._reader.cancelled.connect(self._settle)
        self._thread.start()

    @Slot()
    def _settle(self) -> None:
        # One request answered (they are answered in order, one thread).
        if self._queued:
            self._queued.discard(min(self._queued))

    def load(self, path: str, vendor: str, kwargs: dict | None = None) -> None:
        """Queue a read. Returns immediately; watch the signals."""
        self._reader._cancelled = False
        self._next_request += 1
        request = self._next_request
        self._queued.add(request)
        kwargs = dict(kwargs or {})
        kwargs["_request_id"] = request
        # QueuedConnection by virtue of the thread affinity difference:
        # invoking through the event loop is what actually gets the work
        # onto the worker thread rather than running it here.
        QMetaObject.invokeMethod(
            self._reader, "read", Qt.ConnectionType.QueuedConnection,
            Q_ARG(str, path), Q_ARG(str, vendor), Q_ARG("QVariant", kwargs),
        )

    def cancel_pending(self) -> None:
        """Drop every read queued or running now: each stops at its next
        stage and answers with :attr:`cancelled`. Reads queued after this
        are not affected."""
        with self._reader._lock:
            self._reader._dropped |= set(self._queued)

    def cancel(self) -> None:
        self._reader.cancel()

    def shutdown(self) -> None:
        """Stop the thread and wait for it. Safe to call twice."""
        self._reader.cancel()
        if self._thread.isRunning():
            self._thread.quit()
            self._thread.wait()
