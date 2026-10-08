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

Stop raises :class:`~opennmr.core.fit_progress.FitCancelled` inside the
fit, which skips the rest of the block; this context manager absorbs it,
says in the status bar that nothing was changed, and sets ``stopped``.
The fit runs on the GUI thread, as the Czjzek library build does: the
pop-up keeps the window responsive (``processEvents``) and is window-
modal, so nothing else can be started underneath it.
"""

from __future__ import annotations

import math
import time

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QProgressDialog

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


class FitProgress:
    """A context manager: the pop-up for every fit run inside it."""

    #: Seconds before the pop-up appears: a quick fit shows nothing.
    DELAY = 0.4

    def __init__(self, parent, title: str) -> None:
        self._parent, self.title = parent, title
        self._dialog = None
        self._watch = None
        self.stopped = False
        self.last_status = None

    def __enter__(self) -> "FitProgress":
        self._started = time.perf_counter()
        self._watch = fit_progress.watch(self._update, label=self.title)
        self._watch.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        self._watch.__exit__(None, None, None)
        if exc_type is not None and issubclass(exc_type, fit_progress.FitCancelled):
            self.stopped = True
            # The Stop skipped the caller's own restore (it is in an
            # ``except Exception``, which a FitCancelled passes by).
            while QApplication.overrideCursor() is not None:
                QApplication.restoreOverrideCursor()
            window = self._parent.window() if self._parent is not None else None
            bar = getattr(window, "statusBar", None)
            if callable(bar):
                bar().showMessage(f"{self.title} stopped. Nothing was changed: the parameters "
                                  f"are as they were before the fit.", 10000)
            return True
        return False

    def _update(self, status) -> bool:
        self.last_status = status
        if self._dialog is None:
            if time.perf_counter() - self._started < self.DELAY:
                return True
            self._dialog = QProgressDialog(status_text(self.title, status), "Stop", 0, 0,
                                           self._parent)
            self._dialog.setWindowTitle(self.title)
            self._dialog.setWindowModality(Qt.WindowModality.WindowModal)
            self._dialog.setMinimumDuration(0)
            self._dialog.setMinimumWidth(420)
        self._dialog.setLabelText(status_text(self.title, status))
        self._dialog.setValue(0)
        QApplication.processEvents()
        return not self._dialog.wasCanceled()

    @property
    def shown(self) -> bool:
        return self._dialog is not None

    def close(self) -> None:
        if self._dialog is not None:
            self._dialog.reset()
            self._dialog.close()
            self._dialog.deleteLater()
            self._dialog = None
