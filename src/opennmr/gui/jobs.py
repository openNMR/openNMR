"""Long fits and simulations in the background, one at a time, the rest
waiting their turn (explicit user request, 2026-11-13: "is it possible to
run a fit/simulation and then also be able to do something else at the
same time?" -- one job in the background, the others queued and the user
told so; the queue shown from the status bar, Ctrl/Cmd+R).

A job is split where every slow panel already splits it:

* **inputs**, read from the panel when the job is started (so editing the
  panel while it runs, or setting up the next model, changes nothing);
* **compute**, which touches no widget and runs on a worker thread,
  reporting through :mod:`opennmr.core.fit_progress` as before;
* **apply**, back on the GUI thread: show the result, store the fit.

A fit belongs to the spectrum it was started on, never to whichever is on
screen when it finishes. If that spectrum is selected, the result is shown
at once; if not, it waits and is shown when the spectrum is next selected
(the status bar says so, and the Background jobs window has **Show**). If
the spectrum was removed, the result is discarded and the user told. If
its processing changed while the fit ran, the result is still shown and
the user is told it was fitted on the processing as it was.

``OPENNMR_BACKGROUND_JOBS=0`` runs every job at once on the GUI thread,
with the modal Stop-able pop-up -- what tests pin, and what happens for a
panel that is not inside the main window.
"""

from __future__ import annotations

import itertools
import os
import threading
from collections import deque

from PySide6.QtCore import Q_ARG, QMetaObject, QObject, Qt, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QApplication, QDialog, QDialogButtonBox, QHBoxLayout, QLabel, QListWidget, QMessageBox, QProgressBar,
    QPushButton, QToolButton, QVBoxLayout,
)

from ..core import fit_progress

BACKGROUND_JOBS_ENV = "OPENNMR_BACKGROUND_JOBS"


def background_jobs_enabled() -> bool:
    """Whether long jobs run in the background (on unless the
    environment says otherwise)."""
    value = os.environ.get(BACKGROUND_JOBS_ENV)
    if value is None:
        return True
    return value.strip().lower() not in ("0", "false", "no", "off")


_IDS = itertools.count(1)


class Job:
    """One fit or simulation.

    ``compute()`` runs on the worker thread and returns the result;
    ``apply(result, stopped_early)`` shows it (GUI thread); ``fail(exc)``
    reports an error (GUI thread). ``spectrum_index``: the spectrum a fit
    belongs to (None for a simulation). ``inputs``/``restore_inputs``: the
    panel's inputs when started, put back before the result is shown if
    they were changed meanwhile. ``watched``: the compute reports to
    fit_progress, so it can be stopped (simulations cannot: Stop discards
    their result when they finish). ``keepable``: Stop and Keep Best means
    something (a fit; not a simulation that only reports its stage).
    ``on_end()``: called on the GUI thread once the job is over, however it
    ended (shown, waiting for its spectrum, failed, stopped or removed from
    the queue) -- for a panel that marks itself busy while its job runs."""

    def __init__(self, title: str, compute, apply, *, fail=None, panel=None,
                 spectrum_index: int | None = None, inputs=None, restore_inputs=None,
                 current_inputs=None, watched: bool = True, keepable: bool | None = None,
                 on_end=None) -> None:
        self.id = next(_IDS)
        self.title = title
        self.compute = compute
        self.apply = apply
        self.fail = fail
        self.panel = panel
        self.spectrum_index = spectrum_index
        self.inputs = inputs
        self.restore_inputs = restore_inputs
        self.current_inputs = current_inputs
        self.watched = watched
        self.keepable = watched if keepable is None else bool(keepable)
        self.on_end = on_end
        # Filled in by the manager.
        self.window = None          # the main window it was submitted to
        self.target = None          # the spectrum's session bookkeeping (identity)
        self.spectrum = None        # the spectrum object when started
        self.label = ""             # the spectrum's name, for messages
        self.answer = None          # None, "discard" or fit_progress.KEEP
        self.status = None          # the latest FitStatus
        self._lock = threading.Lock()

    def describe(self) -> str:
        return f"{self.title} of {self.label}" if self.label else self.title

    def current_index(self) -> int | None:
        """The row of the spectrum this job belongs to *now* (rows move
        when spectra are added or removed while it runs); None if it has
        been removed or the job has no spectrum."""
        if self.target is not None and self.window is not None:
            return self.window.session.index_of_info(self.target)
        return self.spectrum_index

    def end(self) -> None:
        """Call ``on_end`` once (GUI thread)."""
        callback, self.on_end = self.on_end, None
        if callback is not None:
            try:
                callback()
            except Exception:  # noqa: BLE001 -- bookkeeping must not sink the queue
                pass


# ---------------------------------------------------------------------------
# The worker
# ---------------------------------------------------------------------------

class _Worker(QObject):
    progress = Signal(int)                       # job id: its status changed
    finished = Signal(int, object, bool, bool)   # job id, result, stopped_early, cancelled
    failed = Signal(int, object)                 # job id, exception

    def __init__(self) -> None:
        super().__init__()
        self.jobs: dict[int, Job] = {}

    @Slot(int)
    def run(self, job_id: int) -> None:
        job = self.jobs.get(job_id)
        if job is None:
            return
        if job.answer == "discard":
            self.finished.emit(job_id, None, False, True)
            return

        def report(status):
            with job._lock:
                job.status = status
            self.progress.emit(job_id)
            if job.answer == "discard":
                return False
            if job.answer == fit_progress.KEEP:
                return fit_progress.KEEP
            return True

        try:
            if job.watched:
                with fit_progress.watch(report, label=job.title, min_interval=0.25) as status:
                    result = job.compute()
                stopped_early = bool(getattr(status, "stopped_early", False))
            else:
                result = job.compute()
                stopped_early = False
        except fit_progress.FitCancelled:
            self.finished.emit(job_id, None, False, True)
        except Exception as exc:  # noqa: BLE001 -- reported on the GUI thread
            self.failed.emit(job_id, exc)
        else:
            cancelled = job.answer == "discard"
            self.finished.emit(job_id, None if cancelled else result, stopped_early, cancelled)


# ---------------------------------------------------------------------------
# The manager
# ---------------------------------------------------------------------------

class JobManager(QObject):
    """Runs one job at a time on its own thread; the rest wait in order."""

    changed = Signal()

    def __init__(self, window) -> None:
        super().__init__(window)
        self.window = window
        self.current: Job | None = None
        self.queue: deque[Job] = deque()
        #: Finished jobs whose result waits for their spectrum to be selected.
        self.waiting: list[Job] = []
        self._thread: QThread | None = None
        self._worker: _Worker | None = None
        self.indicator = JobIndicator(self)
        self.dialog: JobsDialog | None = None

    # -- starting --------------------------------------------------------

    def submit(self, job: Job) -> bool:
        """Run ``job`` now, or after the jobs before it. Returns whether
        it ran to a result now (only when jobs run on the GUI thread) or
        was started or queued (in the background)."""
        window = self.window
        job.window = window
        index = job.spectrum_index
        if index is not None and 0 <= index < len(window.session):
            job.target = window._history_info(index)
            job.spectrum = window.session.spectra[index]
            job.label = window.session.label(index)
        if not background_jobs_enabled():
            return run_inline(job)
        if self.current is None:
            self._start(job)
        else:
            self.queue.append(job)
            before = len(self.queue) - 1
            window.statusBar().showMessage(
                f"{job.describe()} will start when {self.current.describe()} has finished"
                + (f" and the {before} waiting before it" if before else "")
                + ". Ctrl+R shows the queue.", 10000)
        self.changed.emit()
        return True

    def _ensure_thread(self) -> None:
        if self._thread is not None:
            return
        self._thread = QThread()
        self._worker = _Worker()
        self._worker.moveToThread(self._thread)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._thread.start()

    def _start(self, job: Job) -> None:
        self._ensure_thread()
        self.current = job
        self._worker.jobs[job.id] = job
        self.window.statusBar().showMessage(
            f"{job.describe()} is running in the background: you can carry on working. "
            "Ctrl+R shows its progress.", 8000)
        QMetaObject.invokeMethod(self._worker, "run", Qt.ConnectionType.QueuedConnection,
                                 Q_ARG(int, job.id))
        self.changed.emit()

    def _next(self) -> None:
        self.current = None
        if self.queue:
            self._start(self.queue.popleft())
        self.changed.emit()

    # -- stopping --------------------------------------------------------

    def stop_current(self, keep: bool) -> None:
        job = self.current
        if job is not None and job.answer is None:
            job.answer = fit_progress.KEEP if keep and job.keepable else "discard"
            self.changed.emit()

    def remove_queued(self, job_id: int) -> None:
        for job in list(self.queue):
            if job.id == job_id:
                self.queue.remove(job)
                job.end()
        self.changed.emit()

    def shutdown(self) -> None:
        """Stop the running job and the thread (the window is closing)."""
        for job in self.queue:
            job.end()
        self.queue.clear()
        if self.current is not None:
            self.current.answer = "discard"
        if self._thread is not None and self._thread.isRunning():
            self._thread.quit()
            self._thread.wait(10000)

    # -- results ---------------------------------------------------------

    @Slot(int)
    def _on_progress(self, job_id: int) -> None:
        if self.current is not None and self.current.id == job_id:
            self.changed.emit()

    @Slot(int, object)
    def _on_failed(self, job_id: int, exc) -> None:
        job = self._worker.jobs.pop(job_id, None)
        self._next()
        if job is None:
            return
        job.end()
        if job.fail is not None:
            job.fail(exc)
        else:
            QMessageBox.warning(self.window, job.title, f"{job.describe()} did not succeed:\n\n{exc}")

    @Slot(int, object, bool, bool)
    def _on_finished(self, job_id: int, result, stopped_early: bool, cancelled: bool) -> None:
        job = self._worker.jobs.pop(job_id, None)
        self._next()
        if job is None:
            return
        job.end()
        bar = self.window.statusBar()
        if cancelled:
            bar.showMessage(f"{job.describe()} stopped. Nothing was changed.", 10000)
            return
        if job.target is not None:
            index = self.window.session.index_of_info(job.target)
            if index is None:
                bar.showMessage(f"{job.describe()} finished, but that spectrum has been "
                                "removed: the result was not kept.", 12000)
                return
            if not self.window._is_selected_info(job.target):
                job.target.setdefault("pending_job_results", []).append(
                    (job, result, stopped_early))
                bar.showMessage(f"{job.describe()} has finished: select {job.label} to see the "
                                "result (or Ctrl+R ▸ Show).", 15000)
                self._finished_waiting(job)
                return
        deliver(self.window, job, result, stopped_early)

    def _finished_waiting(self, job: Job) -> None:
        self.waiting.append(job)
        self.changed.emit()

    def deliver_pending(self, info) -> None:
        """Show the results waiting for the spectrum just selected."""
        if not isinstance(info, dict):
            return
        pending = info.pop("pending_job_results", None) or []
        for job, result, stopped_early in pending:
            if job in self.waiting:
                self.waiting.remove(job)
            deliver(self.window, job, result, stopped_early)
        if pending:
            self.changed.emit()

    def show_waiting(self, job: Job) -> None:
        """Select the spectrum a finished job belongs to (its result is
        then shown)."""
        if job.target is None:
            return
        index = self.window.session.index_of_info(job.target)
        if index is None:
            if job in self.waiting:
                self.waiting.remove(job)
            self.changed.emit()
            return
        self.window._select_single_row(index)

    # -- the window ------------------------------------------------------

    def show_dialog(self) -> None:
        if self.dialog is None:
            self.dialog = JobsDialog(self, self.window)
        self.dialog.refresh()
        self.dialog.show()
        self.dialog.raise_()
        self.dialog.activateWindow()


def deliver(window, job: Job, result, stopped_early: bool) -> None:
    """Show a finished job's result (GUI thread)."""
    notes = []
    if job.restore_inputs is not None and job.current_inputs is not None \
            and job.inputs is not None:
        try:
            if job.current_inputs() != job.inputs:
                job.restore_inputs(job.inputs)
                notes.append("The panel's inputs were changed while it ran; they have been "
                             "put back to those it was run with.")
        except Exception:  # noqa: BLE001 -- the result is still shown
            pass
    if job.target is not None and job.spectrum is not None:
        index = window.session.index_of_info(job.target)
        if index is not None and window.session.spectra[index] is not job.spectrum:
            notes.append("The spectrum's processing changed while it ran: the result is for "
                         "the processing as it was when it started.")
    job.apply(result, stopped_early)
    if stopped_early:
        notes.insert(0, "It was stopped early: showing the best parameters it reached, "
                        "without uncertainties (it had not converged).")
    if notes:
        text = f"{job.describe()}: " + " ".join(notes)
        window.statusBar().showMessage(text, 15000)
        QTimer.singleShot(0, lambda: window.statusBar().showMessage(text, 15000))


def run_inline(job: Job) -> bool:
    """Run ``job`` now on the GUI thread with the modal Stop-able pop-up
    (the behaviour before background jobs; tests, and panels outside the
    main window). Returns whether its result was shown."""
    try:
        return _run_inline(job)
    finally:
        job.end()


def _run_inline(job: Job) -> bool:
    from .fit_progress_dialog import FitProgress

    def failed(exc) -> bool:
        while QApplication.overrideCursor() is not None:
            QApplication.restoreOverrideCursor()
        if job.fail is not None:
            job.fail(exc)
        else:
            QMessageBox.warning(job.panel, job.title, f"It did not succeed:\n\n{exc}")
        return False

    QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
    if job.watched:
        with FitProgress(job.panel, job.title) as box:
            try:
                result = job.compute()
            except Exception as exc:  # noqa: BLE001 -- reported, as in the background
                return failed(exc)
        if box.stopped:
            return False
        stopped_early = box.stopped_early
    else:
        try:
            result = job.compute()
        except Exception as exc:  # noqa: BLE001
            return failed(exc)
        stopped_early = False
    while QApplication.overrideCursor() is not None:
        QApplication.restoreOverrideCursor()
    job.apply(result, stopped_early)
    return True


def in_background(panel) -> bool:
    """Whether a job submitted from ``panel`` now would run in the
    background (rather than at once, with the modal pop-up)."""
    window = panel.window() if panel is not None else None
    return getattr(window, "job_manager", None) is not None and background_jobs_enabled()


def submit(panel, job: Job) -> bool:
    """Run ``job`` through the main window's job manager, or at once when
    ``panel`` is not inside one. Returns what :meth:`JobManager.submit` does."""
    window = panel.window() if panel is not None else None
    manager = getattr(window, "job_manager", None)
    if job.panel is None:
        job.panel = panel
    if manager is None:
        return run_inline(job)
    return manager.submit(job)


# ---------------------------------------------------------------------------
# The status-bar indicator and the window
# ---------------------------------------------------------------------------

class JobIndicator(QToolButton):
    """Bottom right of the status bar while a job runs or waits: what is
    running and how many wait. Click, double-click or right-click (or
    Ctrl/Cmd+R) for the Background jobs window. Hidden when idle."""

    def __init__(self, manager: JobManager) -> None:
        super().__init__()
        self._manager = manager
        self.setAutoRaise(True)
        self.setToolTip("Background jobs: click (or Ctrl+R) for progress, Stop, and the queue.")
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(lambda _pos: manager.show_dialog())
        self.clicked.connect(manager.show_dialog)
        manager.changed.connect(self.refresh)
        self.setVisible(False)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802 -- Qt override
        self._manager.show_dialog()

    def refresh(self) -> None:
        manager = self._manager
        job = manager.current
        waiting = len(manager.queue)
        done = len(manager.waiting)
        if job is None and not waiting and not done:
            self.setVisible(False)
            return
        parts = []
        if job is not None:
            text = f"⏳ {job.describe()}"
            status = job.status
            if status is not None and status.evaluations:
                text += f" · {status.evaluations:,} evaluations"
            elif status is not None and status.stage:
                text += f" · {status.stage}"
            if job.answer is not None:
                text += " · stopping…"
            parts.append(text)
        if waiting:
            parts.append(f"{waiting} waiting")
        if done:
            parts.append(f"{done} finished, not yet shown")
        self.setText("  ·  ".join(parts))
        self.setVisible(True)


class JobsDialog(QDialog):
    """The running job (its progress, Stop and Keep Best, Discard), the
    jobs waiting their turn (Remove), and finished results waiting for
    their spectrum to be selected (Show). Not modal: work carries on."""

    def __init__(self, manager: JobManager, parent=None) -> None:
        super().__init__(parent)
        self._manager = manager
        self.setWindowTitle("Background jobs")
        self.setMinimumWidth(460)
        layout = QVBoxLayout(self)
        self.current_label = QLabel("")
        self.current_label.setWordWrap(True)
        layout.addWidget(self.current_label)
        self.busy = QProgressBar()
        self.busy.setRange(0, 0)
        layout.addWidget(self.busy)
        row = QHBoxLayout()
        self.keep_button = QPushButton("Stop and Keep Best")
        self.keep_button.setToolTip(
            "End the fit now and show the best parameters it has reached, without "
            "uncertainties (it has not converged).")
        self.keep_button.clicked.connect(lambda: manager.stop_current(keep=True))
        row.addWidget(self.keep_button)
        self.discard_button = QPushButton("Discard")
        self.discard_button.setToolTip("End it now and keep nothing.")
        self.discard_button.clicked.connect(lambda: manager.stop_current(keep=False))
        row.addWidget(self.discard_button)
        row.addStretch(1)
        layout.addLayout(row)

        layout.addWidget(QLabel("Waiting their turn:"))
        self.queue_list = QListWidget()
        layout.addWidget(self.queue_list)
        self.remove_button = QPushButton("Remove from Queue")
        self.remove_button.clicked.connect(self._remove)
        layout.addWidget(self.remove_button)

        layout.addWidget(QLabel("Finished, waiting for their spectrum to be selected:"))
        self.done_list = QListWidget()
        self.done_list.itemDoubleClicked.connect(lambda _item: self._show())
        layout.addWidget(self.done_list)
        self.show_button = QPushButton("Show")
        self.show_button.setToolTip("Select the spectrum and show the result.")
        self.show_button.clicked.connect(self._show)
        layout.addWidget(self.show_button)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.hide)
        layout.addWidget(buttons)
        manager.changed.connect(self.refresh)

    def refresh(self) -> None:
        from .fit_progress_dialog import status_text
        manager = self._manager
        job = manager.current
        if job is None:
            self.current_label.setText("Nothing is running.")
        elif job.status is not None:
            self.current_label.setText(status_text(job.describe(), job.status)
                                       + ("\n\nStopping…" if job.answer else ""))
        else:
            self.current_label.setText(job.describe() + ("\n\nStopping…" if job.answer else ""))
        running = job is not None and job.answer is None
        self.busy.setVisible(job is not None)
        self.keep_button.setEnabled(running and job.keepable)
        self.discard_button.setEnabled(running)
        self.queue_list.clear()
        for queued in manager.queue:
            self.queue_list.addItem(queued.describe())
            self.queue_list.item(self.queue_list.count() - 1).setData(Qt.ItemDataRole.UserRole,
                                                                       queued.id)
        self.remove_button.setEnabled(bool(manager.queue))
        self.done_list.clear()
        for done in manager.waiting:
            self.done_list.addItem(done.describe())
        self.show_button.setEnabled(bool(manager.waiting))

    def _remove(self) -> None:
        item = self.queue_list.currentItem() or (self.queue_list.item(0)
                                                 if self.queue_list.count() else None)
        if item is not None:
            self._manager.remove_queued(item.data(Qt.ItemDataRole.UserRole))

    def _show(self) -> None:
        waiting = self._manager.waiting
        row = self.done_list.currentRow()
        if not waiting:
            return
        job = waiting[row if 0 <= row < len(waiting) else 0]
        self._manager.show_waiting(job)


def make_action(manager: JobManager, parent) -> QAction:
    """Window ▸ Background Jobs… (Ctrl/Cmd+R)."""
    action = QAction("Background Jobs…", parent)
    action.setShortcut(QKeySequence("Ctrl+R"))
    action.setToolTip("The fit or simulation running in the background, and those waiting.")
    action.triggered.connect(manager.show_dialog)
    return action
