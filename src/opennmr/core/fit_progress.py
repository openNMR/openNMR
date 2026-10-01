"""
Live progress of a running fit -- for every fitter in openNMR.

Explicit user request (2026-10-01): "I really like the little
update-as-you-go pop-up when computing the Czjzek distribution. Is there
a good way to do something similar when fitting spectra, too? Not just
for the Czjzek, but also for quadrupole, CSA, J etc".

A least-squares fit has no known number of steps, so what is reported is
what *is* known while it runs: how many times the model has been
evaluated, the misfit (residual sum of squares, weighted when the fit is)
-- the best so far and how far it has come down from the start -- how
fast it is still falling, and the time taken. Whoever watches can stop
the fit; it then raises :class:`FitCancelled` and returns nothing, so a
half-finished fit is never mistaken for a result.

Each fitter passes its model through :func:`tracked` (``curve_fit``) or
its residual function through :func:`tracked_residuals`
(``least_squares``). Outside :func:`watch` both return the function
unchanged, so an unwatched fit costs nothing.

In a notebook::

    from opennmr.core import fit_progress
    with fit_progress.watch(fit_progress.print_progress):
        popt, pcov, *rest = fit_csa_spectrum(results, p0, ...)

The GUI shows the same numbers in a Stop-able pop-up
(``opennmr.gui.fit_progress_dialog.FitProgress``).
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np


class FitCancelled(BaseException):
    """Raised inside a fit when whoever watches it asks it to stop.

    A ``BaseException``, like ``KeyboardInterrupt``: fitters catch
    ``RuntimeError``/``Exception`` to skip a start that does not converge,
    and a Stop must not be mistaken for one of those (it would carry on
    with the next start, and end "no starting point converged")."""


@dataclass
class FitStatus:
    """What a watcher is told, at most every ``min_interval`` seconds."""

    label: str = "Fit"
    #: Model (or residual) evaluations so far, finite-difference ones
    #: included -- curve_fit spends (free parameters + 1) per step.
    evaluations: int = 0
    elapsed: float = 0.0
    #: Residual sum of squares: at the first evaluation, the best so far,
    #: and the latest.
    start_cost: float = float("nan")
    best_cost: float = float("nan")
    cost: float = float("nan")
    #: Fractional fall of the best misfit over about the last second
    #: (0.02 = 2 %); ``nan`` until a second has passed.
    recent_fall: float = float("nan")
    #: What the fit is doing now, e.g. "start 3 of 12" or "slice 2 of 5".
    stage: str = ""
    #: The fit's own free parameters' count, when known.
    n_free: int | None = None
    history: list = field(default_factory=list, repr=False)

    @property
    def relative(self) -> float:
        """Best misfit as a fraction of the starting one."""
        if not np.isfinite(self.start_cost) or self.start_cost <= 0:
            return float("nan")
        return self.best_cost / self.start_cost

    def summary(self) -> str:
        """One line of plain text -- what :func:`print_progress` prints."""
        parts = [f"{self.label}"]
        if self.stage:
            parts.append(self.stage)
        parts.append(f"{self.evaluations:,} evaluations")
        parts.append(f"{self.elapsed:.1f} s")
        if np.isfinite(self.best_cost):
            parts.append(f"misfit {self.best_cost:.4g}")
            if np.isfinite(self.relative):
                parts.append(f"{100 * self.relative:.3g} % of the start")
        if np.isfinite(self.recent_fall):
            parts.append(f"falling {100 * self.recent_fall:.2g} %/s"
                         if self.recent_fall > 1e-6 else "no longer falling")
        return " | ".join(parts)


class _Watcher:
    def __init__(self, callback, label, min_interval):
        self.callback = callback
        self.min_interval = float(min_interval)
        self.status = FitStatus(label=label)
        self.started = time.perf_counter()
        self.last_report = -np.inf
        self.cancelled = False

    def record(self, cost: float) -> None:
        if self.cancelled:
            raise FitCancelled("The fit was stopped before it finished.")
        st = self.status
        st.evaluations += 1
        now = time.perf_counter()
        st.elapsed = now - self.started
        if np.isfinite(cost):
            st.cost = cost
            if not np.isfinite(st.start_cost):
                st.start_cost = cost
            if not np.isfinite(st.best_cost) or cost < st.best_cost:
                st.best_cost = cost
        if now - self.last_report >= self.min_interval:
            self.last_report = now
            st.history.append((st.elapsed, st.best_cost))
            # The fall over about the last second, from the history.
            older = [(t, c) for t, c in st.history if st.elapsed - t >= 1.0]
            if older and np.isfinite(older[-1][1]) and older[-1][1] > 0:
                t0, c0 = older[-1]
                st.recent_fall = max(0.0, (c0 - st.best_cost) / c0) / max(st.elapsed - t0, 1e-9)
            del st.history[:-200]
            if self.callback(st) is False:
                self.cancelled = True
                raise FitCancelled("The fit was stopped before it finished.")


_ACTIVE: list[_Watcher] = []


@contextmanager
def watch(callback, label: str = "Fit", min_interval: float = 0.1):
    """Report every fit run inside this block to ``callback(status)``
    (a :class:`FitStatus`), at most every ``min_interval`` seconds.
    Returning ``False`` from the callback stops the fit with
    :class:`FitCancelled`. Nested blocks report to the outermost one, so
    a multi-start fit made of single fits is watched as one."""
    if _ACTIVE:
        yield _ACTIVE[0].status
        return
    watcher = _Watcher(callback, label, min_interval)
    _ACTIVE.append(watcher)
    try:
        yield watcher.status
    finally:
        _ACTIVE.remove(watcher)


def active() -> bool:
    return bool(_ACTIVE)


def set_stage(text: str) -> None:
    """Say what the watched fit is doing now ("start 3 of 12"); a no-op
    when nothing watches."""
    if _ACTIVE:
        _ACTIVE[0].status.stage = str(text)
        # Shown at the next report; a new stage is worth one straight away.
        _ACTIVE[0].last_report = -np.inf


def check_cancelled() -> None:
    """Raise :class:`FitCancelled` if the watcher has asked to stop --
    for loops between fits (slices, starts) that evaluate no model."""
    if _ACTIVE and _ACTIVE[0].cancelled:
        raise FitCancelled("The fit was stopped before it finished.")


def tracked(model, ydata, sigma=None, n_free: int | None = None):
    """``model`` (a ``curve_fit`` model, ``model(x, *p)``) reporting to
    the active :func:`watch` -- or ``model`` itself when nothing watches.
    The misfit is ``sum(((model - ydata) / sigma)**2)``, as curve_fit
    minimises it."""
    if not _ACTIVE:
        return model
    watcher = _ACTIVE[0]
    if n_free is not None:
        watcher.status.n_free = int(n_free)
    y = np.asarray(ydata, dtype=float)
    w = None if sigma is None else 1.0 / np.asarray(sigma, dtype=float)

    def watched(x, *params):
        out = model(x, *params)
        try:
            r = np.asarray(out, dtype=float) - y
            if w is not None:
                r = r * w
            cost = float(np.dot(r.ravel(), r.ravel()))
        except (TypeError, ValueError):
            cost = float("nan")
        watcher.record(cost)
        return out

    return watched


def tracked_residuals(fun, n_free: int | None = None):
    """``fun`` (a ``least_squares`` residual function, ``fun(p)``)
    reporting ``sum(fun(p)**2)`` to the active :func:`watch` -- or
    ``fun`` itself when nothing watches."""
    if not _ACTIVE:
        return fun
    watcher = _ACTIVE[0]
    if n_free is not None:
        watcher.status.n_free = int(n_free)

    def watched(p, *args, **kwargs):
        out = fun(p, *args, **kwargs)
        try:
            r = np.asarray(out, dtype=float).ravel()
            cost = float(np.dot(r, r))
        except (TypeError, ValueError):
            cost = float("nan")
        watcher.record(cost)
        return out

    return watched


def print_progress(status: FitStatus, every: float = 1.0) -> bool:
    """A ready-made notebook callback: prints :meth:`FitStatus.summary`
    about once a second, on one line. Interrupt the kernel to stop."""
    last = getattr(print_progress, "_last", None)
    if last is not None and status.elapsed - last < every and status.elapsed > last:
        return True
    print_progress._last = status.elapsed
    print("\r" + status.summary(), end="", flush=True)
    return True
