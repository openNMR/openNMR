"""The Stop-able pop-up a spectrum fit shows while it runs.

Explicit user request (2026-10-01): the Czjzek library's update-as-you-go
pop-up, "when fitting spectra, too ... not just for the Czjzek, but also
for quadrupole, CSA, J etc". The numbers come from
:mod:`opennmr.core.fit_progress`, which every fitter reports to.

Use around a fit::

    with FitProgress(self, "CSA fit") as progress:
        try:
            ... the fit and what uses its result ...
        except Exception as exc:
            ...
    if progress.stopped:
        return

Two buttons (explicit user request, 2026-11-12: "rather than erasing all
progress midway, the current fit ... can be displayed"):

* **Stop and Keep Best** ends the fit at once and keeps the best
  parameters it had reached, as its result (``core.fit_progress.KEEP``):
  the block carries on as for a finished fit, ``stopped_early`` is set,
  and the status bar says the result had not converged and has no
  uncertainties. A fit made of several (starts, slices) keeps what it has
  and runs no more. A fitter that cannot hand back a partial result
  discards it, as Discard does, and says so.
* **Discard** raises :class:`~opennmr.core.fit_progress.FitCancelled`
  inside the fit, which skips the rest of the block; this context manager
  absorbs it, says in the status bar that nothing was changed, and sets
  ``stopped``.
The fit runs on the GUI thread, as the Czjzek library build does: the
pop-up keeps the window responsive (``processEvents``) and is window-
modal, so nothing else can be started underneath it.
"""

from __future__ import annotations

import math
import time

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication, QDialog, QDialogButtonBox, QLabel, QProgressBar, QVBoxLayout,
)

from ..core import fit_progress


def _duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f} s"
    minutes, secs = divmod(int(round(seconds)), 60)
    return f"{minutes} min {secs:02d} s"


def status_text(title: str, status) -> str:
    """The pop-up's lines for ``status`` (a FitStatus)."""
    head = title + (f" — {status.stage}" if status.stage else "")
    lines = [head, f"{status.evaluations:,} model evaluations, {_duration(status.elapsed)}"]
    iterations = getattr(status, "iterations", None)
    limit = getattr(status, "max_iterations", None)
    if iterations is not None and limit:
        lines.append(f"About {iterations:,} iterations of at most {limit:,} "
                     f"(Settings ▸ Fitting)")
    if math.isfinite(status.best_cost):
        misfit = f"Misfit (sum of squared residuals): {status.best_cost:.4g}"
        if math.isfinite(status.relative):
            misfit += f", {100 * status.relative:.3g} % of the starting value"
        lines.append(misfit)
    if math.isfinite(status.recent_fall):
        lines.append(f"Still improving: {100 * status.recent_fall:.2g} % per second"
                     if status.recent_fall > 1e-6 else
                     "No longer improving: the fit is converging")
    return "\n".join(lines)


class FitProgressDialog(QDialog):
    """The pop-up: what the fit is doing, a busy bar, and the two buttons.
    ``answer`` is None while it runs, ``fit_progress.KEEP`` or ``"discard"``
    once a button is pressed (closing the window is Discard)."""

    def __init__(self, title: str, text: str, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setWindowModality(Qt.WindowModality.WindowModal)
        self.setMinimumWidth(440)
        self.answer = None
        layout = QVBoxLayout(self)
        self.label = QLabel(text)
        self.label.setWordWrap(True)
        layout.addWidget(self.label)
        bar = QProgressBar()
        bar.setRange(0, 0)
        layout.addWidget(bar)
        buttons = QDialogButtonBox()
        self.keep_button = buttons.addButton("Stop and Keep Best",
                                             QDialogButtonBox.ButtonRole.AcceptRole)
        self.keep_button.setToolTip(
            "End the fit now and show the best parameters it has reached. It has not "
            "converged, so they come without uncertainties; fit again to carry on from them.")
        self.discard_button = buttons.addButton("Discard",
                                                QDialogButtonBox.ButtonRole.RejectRole)
        self.discard_button.setToolTip("End the fit now and keep nothing: the parameters stay as "
                                       "they were before the fit.")
        self.keep_button.clicked.connect(lambda: self._answer(fit_progress.KEEP))
        self.discard_button.clicked.connect(lambda: self._answer("discard"))
        layout.addWidget(buttons)

    def _answer(self, answer) -> None:
        if self.answer is None:
            self.answer = answer
            self.label.setText(self.label.text() + "\n\nStopping…")

    def reject(self) -> None:  # Esc or the window's close button
        self._answer("discard")


class FitProgress:
    """A context manager: the pop-up for every fit run inside it."""

    #: Seconds before the pop-up appears: a quick fit shows nothing.
    DELAY = 0.4

    def __init__(self, parent, title: str) -> None:
        self._parent, self.title = parent, title
        self._dialog = None
        self._watch = None
        self.stopped = False
        #: The fit was stopped with Stop and Keep Best, and its best
        #: parameters were kept as its result.
        self.stopped_early = False
        self.last_status = None

    def __enter__(self) -> "FitProgress":
        self._started = time.perf_counter()
        self._watch = fit_progress.watch(self._update, label=self.title)
        self._watch.__enter__()
        return self

    def _say(self, text: str) -> None:
        window = self._parent.window() if self._parent is not None else None
        bar = getattr(window, "statusBar", None)
        if callable(bar):
            bar().showMessage(text, 15000)

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        status = self.last_status
        self._watch.__exit__(None, None, None)
        if exc_type is not None and issubclass(exc_type, fit_progress.FitCancelled):
            self.stopped = True
            # The Stop skipped the caller's own restore (it is in an
            # ``except Exception``, which a FitCancelled passes by).
            while QApplication.overrideCursor() is not None:
                QApplication.restoreOverrideCursor()
            if status is not None and status.stopped_early:
                self._say(f"{self.title} stopped. This fit cannot hand back a partial result, so "
                          "nothing was changed: the parameters are as they were before the fit.")
            else:
                self._say(f"{self.title} stopped. Nothing was changed: the parameters are as "
                          f"they were before the fit.")
            return True
        if status is not None and status.stopped_early:
            self.stopped_early = True
            fall = (f", the misfit at {100 * status.relative:.3g} % of its starting value"
                    if math.isfinite(status.relative) else "")
            text = (f"{self.title} stopped early: showing the best parameters it reached"
                    f"{fall}. It had not converged, so there are no uncertainties; fit again "
                    "to carry on from here.")
            # After the caller has shown the result (which may post its own
            # status message), so this one is the one left on screen.
            from PySide6.QtCore import QTimer
            self._say(text)
            QTimer.singleShot(0, lambda: self._say(text))
        return False

    def _update(self, status):
        self.last_status = status
        if self._dialog is None:
            if time.perf_counter() - self._started < self.DELAY:
                return True
            self._dialog = FitProgressDialog(self.title, status_text(self.title, status),
                                             self._parent)
            self._dialog.show()
        if self._dialog.answer is None:
            self._dialog.label.setText(status_text(self.title, status))
        QApplication.processEvents()
        answer = self._dialog.answer
        if answer == "discard":
            return False
        if answer == fit_progress.KEEP:
            return fit_progress.KEEP
        return True

    @property
    def shown(self) -> bool:
        return self._dialog is not None

    def close(self) -> None:
        if self._dialog is not None:
            self._dialog.answer = self._dialog.answer or "closed"
            self._dialog.hide()
            self._dialog.deleteLater()
            self._dialog = None
