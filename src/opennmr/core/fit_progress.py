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

**Maximum iterations** (explicit user request, 2026-10-08: "a maximum
number of iterations when fitting ... user-editable ... starting value of
2000. If it hasn't converged after that, provide suggestions for where to
go next"). One app-wide limit, :data:`DEFAULT_MAX_ITERATIONS` to start
with, read by every fitter through :func:`limited_curve_fit` /
:func:`limited_least_squares`::

    fit_progress.set_max_iterations(5000)          # from now on
    with fit_progress.iteration_limit(500):        # just these fits
        ...

An *iteration* is one trial step of the optimiser (scipy's ``max_nfev``
for the trust-region fits; the finite-difference evaluations for the
Jacobian are not counted). A fit that reaches the limit raises
:class:`FitNotConverged`, whose message says what to try next and which
carries the best parameters found (``best_params``) so a notebook can
restart from there.

**Stopping and keeping the best so far** (explicit user request,
2026-11-12: "rather than erasing all progress midway, the current fit ...
can be displayed"). A watcher's callback may return :data:`KEEP` instead
of ``False``: the fit in progress then ends at once and
:func:`limited_curve_fit` / :func:`limited_least_squares` return the
best parameters it had reached, as its result -- with no uncertainty
(``pcov`` all NaN; the solution's ``success`` False), since a fit stopped
part-way has not converged and its covariance would mean nothing -- and
the watcher's status says ``stopped_early``. Any later fit inside the
same watch (the next start of a multi-start fit, the next slice of a
series) raises :class:`FitStopped`, a ``RuntimeError`` like a fit that
did not converge, so those loops skip it and keep what they have. A fitter
that does not go through these two functions cannot keep a partial
result: the stop then discards it, as ``False`` does.

**Convergence tolerance** (same request: "control the tolerance where
the spectrum is considered converged"). The optimiser stops when an
iteration changes the misfit (``ftol``) or the parameters (``xtol``) by
less than this relative amount: scipy's own defaults (about 1e-8,
:data:`DEFAULT_TOLERANCE`) unless set with :func:`set_tolerance`
(Settings ▸ Fitting). Looser
is quicker and less exact; tighter is slower. A fitter that asks for its
own tolerance (the spin-system fit's 1e-12) keeps it.

**Fitting algorithm** (explicit user request, 2026-11-13: "allow users
to switch the fitting algorithm ... chosen in the Settings menu"). One
app-wide choice, :data:`DEFAULT_METHOD` ("auto") to start with, read by
every fitter through :func:`limited_curve_fit` /
:func:`limited_least_squares` (and :func:`least_squares` for the few that
need no iteration limit)::

    fit_progress.set_method("dogbox")              # from now on
    with fit_progress.fitting_method("differential_evolution"):
        ...                                        # just these fits

* ``"auto"`` -- what every fit did before the choice existed: scipy's
  Trust Region Reflective when a parameter has a bound, Levenberg-
  Marquardt (``curve_fit``) otherwise; ``least_squares`` fits use TRF.
* ``"trf"`` -- Trust Region Reflective: bounds respected; robust.
* ``"dogbox"`` -- dogleg in a box: bounds respected; can do better than
  TRF when many parameters sit on their bounds.
* ``"lm"`` -- Levenberg-Marquardt: fast, but cannot apply bounds -- a
  fit that has bounds uses TRF instead, and :func:`last_method` says so.
* ``"differential_evolution"`` / ``"dual_annealing"`` -- global searches
  for the best minimum within the bounds, for a starting point that is
  far off or a misfit with several minima. Each needs a finite lower and
  upper bound on every free parameter (a fit without them uses the local
  default instead, and says so), is much slower, and is always finished
  by a TRF fit from its best point, so the result carries the usual
  uncertainties. Seeded, so the same fit gives the same answer.

A fitter that asks for its own method keeps it. What the last fit
actually used, and why when that differs from the choice, is
:func:`last_method`.
"""

from __future__ import annotations

import math
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np


#: What a watcher's callback returns to stop the fit and keep the best
#: parameters reached (see the module notes); ``False`` discards it.
KEEP = "keep"


class FitCancelled(BaseException):
    """Raised inside a fit when whoever watches it asks it to stop.

    A ``BaseException``, like ``KeyboardInterrupt``: fitters catch
    ``RuntimeError``/``Exception`` to skip a start that does not converge,
    and a Stop must not be mistaken for one of those (it would carry on
    with the next start, and end "no starting point converged")."""


class _StopKeepingBest(FitCancelled):
    """Raised inside the model when the watcher asks to stop and keep the
    best so far; caught by :func:`limited_curve_fit` /
    :func:`limited_least_squares`. A fitter that does not use them sees a
    :class:`FitCancelled`, and nothing is kept."""


class FitStopped(RuntimeError):
    """A fit that would have begun after a stop that kept the best so far:
    it is not run. A ``RuntimeError``, as :class:`FitNotConverged` is, so a
    multi-start or per-slice loop skips it and keeps what it has."""


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
    #: The iteration limit in force (see :func:`max_iterations`).
    max_iterations: int | None = None
    #: ``evaluations`` when the current single fit began -- a multi-start
    #: or multi-pass fit counts iterations per fit, as the limit applies.
    iteration_offset: int = 0
    #: The fit was stopped, keeping the best parameters it had reached.
    stopped_early: bool = False
    #: The fitting algorithm in use (see :func:`last_method`).
    method: str = ""
    history: list = field(default_factory=list, repr=False)

    @property
    def iterations(self) -> int | None:
        """Approximate optimiser iterations so far: each one costs one
        trial evaluation plus one per free parameter for the Jacobian.
        ``None`` while the number of free parameters is unknown."""
        if not self.n_free:
            return None
        return int(max(self.evaluations - self.iteration_offset, 0) // (self.n_free + 1))

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
        #: Stop asked for, keeping the best so far: ``keep_pending`` until
        #: the next evaluation ends the fit, then ``keep_requested``.
        self.keep_pending = False
        self.keep_requested = False

    def stop_keeping_best(self) -> None:
        self.keep_requested = True
        self.status.stopped_early = True
        raise _StopKeepingBest("The fit was stopped; keeping the best so far.")

    def record(self, cost: float) -> None:
        if self.cancelled:
            raise FitCancelled("The fit was stopped before it finished.")
        if self.keep_requested:
            return            # finishing up after a keep-stop: count nothing
        if self.keep_pending:
            # At the evaluation *after* the request, so the one that was
            # being reported when Stop came has been noted as a candidate
            # best by the limited_* wrappers.
            self.stop_keeping_best()
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
            answer = self.callback(st)
            if answer is False:
                self.cancelled = True
                raise FitCancelled("The fit was stopped before it finished.")
            if answer == KEEP:
                self.keep_pending = True


class _PerThread:
    """The watched fit, per thread: a fit running in the background
    (gui/jobs.py) and one run on the GUI thread meanwhile must not report
    to, or be stopped by, each other's watcher. Behaves as the list it
    replaced."""

    def __init__(self) -> None:
        self._local = threading.local()

    def _list(self) -> list:
        items = getattr(self._local, "items", None)
        if items is None:
            items = self._local.items = []
        return items

    def __bool__(self) -> bool:
        return bool(self._list())

    def __len__(self) -> int:
        return len(self._list())

    def __getitem__(self, index):
        return self._list()[index]

    def __iter__(self):
        return iter(list(self._list()))

    def append(self, item) -> None:
        self._list().append(item)

    def remove(self, item) -> None:
        self._list().remove(item)


_ACTIVE = _PerThread()


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
    watcher.status.max_iterations = max_iterations()
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


def report_stage(text: str) -> None:
    """Say what the watched run is doing and report it now (at most every
    ``min_interval``), for long steps that evaluate no model -- a
    lineshape library, a field calculation. Raises :class:`FitCancelled`
    when the watcher asks to stop; a Stop and Keep Best asked for here
    ends the fit at its first evaluation (keeping its starting point).
    A no-op when nothing watches."""
    if not _ACTIVE:
        return
    watcher = _ACTIVE[0]
    if watcher.cancelled:
        raise FitCancelled("The fit was stopped before it finished.")
    status = watcher.status
    status.stage = str(text)
    now = time.perf_counter()
    status.elapsed = now - watcher.started
    if now - watcher.last_report < watcher.min_interval:
        return
    watcher.last_report = now
    answer = watcher.callback(status)
    if answer is False:
        watcher.cancelled = True
        raise FitCancelled("The fit was stopped before it finished.")
    if answer == KEEP:
        watcher.keep_pending = True


def _short_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f} s"
    minutes, secs = divmod(int(round(seconds)), 60)
    return f"{minutes} min {secs:02d} s"


def stage_callback(what: str = "lineshape"):
    """A ``progress(done, total)`` callback (the lineshape-library
    builders' protocol) that reports through :func:`report_stage` --
    "lineshape 40 of 400, about 1 min left" -- and returns False (stop)
    when the watcher asks to stop. Safe on any thread."""
    started = time.perf_counter()

    def progress(done, total) -> bool:
        done, total = int(done), int(total)
        text = f"{what} {done:,} of {total:,}"
        if 0 < done < total:
            left = (time.perf_counter() - started) / done * (total - done)
            text += f", about {_short_duration(left)} left"
        try:
            report_stage(text)
        except FitCancelled:
            return False
        return True
    return progress


def stopped_early() -> bool:
    """Whether the watched fit has been stopped keeping the best so far
    (or asked to be)."""
    return bool(_ACTIVE) and (_ACTIVE[0].keep_requested or _ACTIVE[0].keep_pending)


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


# ---------------------------------------------------------------------------
# Maximum iterations
# ---------------------------------------------------------------------------

#: The starting (suggested) limit -- explicit user request.
DEFAULT_MAX_ITERATIONS = 2000
_MAX_ITERATIONS = [DEFAULT_MAX_ITERATIONS]


#: The starting convergence tolerance: scipy's own default.
DEFAULT_TOLERANCE = 1e-8
#: Accepted range of the tolerance.
TOLERANCE_RANGE = (1e-15, 1e-2)
_TOLERANCE = [DEFAULT_TOLERANCE]


def tolerance(value: float | None = None) -> float:
    """The convergence tolerance to use: ``value`` when given, else the one
    set with :func:`set_tolerance` (1e-8 unless changed)."""
    if value is None:
        return _TOLERANCE[-1]
    value = float(value)
    lo, hi = TOLERANCE_RANGE
    if not lo <= value <= hi:
        raise ValueError(f"The convergence tolerance must be between {lo:g} and {hi:g} "
                         f"(got {value:g}).")
    return value


def set_tolerance(value: float | None) -> float:
    """Set the app-wide tolerance (``None`` restores 1e-8). Returns it."""
    _TOLERANCE[0] = DEFAULT_TOLERANCE if value is None else tolerance(value)
    del _TOLERANCE[1:]
    return _TOLERANCE[0]


@contextmanager
def convergence_tolerance(value: float):
    """Use ``value`` as the tolerance for the fits inside this block only."""
    _TOLERANCE.append(tolerance(value))
    try:
        yield _TOLERANCE[-1]
    finally:
        _TOLERANCE.pop()


def _apply_tolerance(kwargs: dict) -> None:
    """Pass the tolerance on -- only once it has been changed, so that left
    alone every fit is exactly as before (scipy's defaults: 1e-8 for the
    trust-region fits, about 1.5e-8 for Levenberg-Marquardt)."""
    tol = tolerance()
    if tol == DEFAULT_TOLERANCE:
        return
    kwargs.setdefault("ftol", tol)
    kwargs.setdefault("xtol", tol)


# ---------------------------------------------------------------------------
# Fitting algorithm
# ---------------------------------------------------------------------------

#: Every choice, with its name as Settings ▸ Fitting shows it.
METHODS = {
    "auto": "Automatic (TRF with bounds, Levenberg-Marquardt without)",
    "trf": "Trust Region Reflective (TRF)",
    "dogbox": "Dogbox (dogleg in a box)",
    "lm": "Levenberg-Marquardt (no bounds)",
    "differential_evolution": "Differential evolution (global), then TRF",
    "dual_annealing": "Dual annealing (global), then TRF",
}
#: The starting choice: what every fit did before there was one.
DEFAULT_METHOD = "auto"
GLOBAL_METHODS = ("differential_evolution", "dual_annealing")
#: A global search's own iteration cap (generations for differential
#: evolution); the TRF fit that finishes it is held to max_iterations.
GLOBAL_MAX_ITERATIONS = 1000
_METHOD = [DEFAULT_METHOD]
_LAST_METHOD: dict = {"method": None, "note": ""}


def method(value: str | None = None) -> str:
    """The fitting algorithm to use: ``value`` when given (checked), else
    the one set with :func:`set_method` ("auto" unless changed)."""
    if value is None:
        return _METHOD[-1]
    value = str(value)
    if value not in METHODS:
        raise ValueError(f"Unknown fitting method {value!r}; use one of {list(METHODS)}.")
    return value


def set_method(value: str | None) -> str:
    """Set the app-wide fitting algorithm (``None`` restores "auto")."""
    _METHOD[0] = DEFAULT_METHOD if value is None else method(value)
    del _METHOD[1:]
    return _METHOD[0]


@contextmanager
def fitting_method(value: str):
    """Use ``value`` as the fitting algorithm for the fits inside this block."""
    _METHOD.append(method(value))
    try:
        yield _METHOD[-1]
    finally:
        _METHOD.pop()


def last_method() -> dict:
    """``{"method": ..., "note": ...}`` for the most recent fit: the
    algorithm it actually used (a key of :data:`METHODS`, "auto" for
    scipy's own choice, or the one the fitter asked for), and why, when
    that is not the one chosen."""
    return dict(_LAST_METHOD)


def _bounds_arrays(bounds, n: int):
    """``(lo, hi)`` as float arrays of length ``n`` (or None)."""
    if bounds is None:
        return None
    lo, hi = (bounds.lb, bounds.ub) if hasattr(bounds, "lb") else bounds
    try:
        lo = np.broadcast_to(np.asarray(lo, dtype=float), (n,)).copy()
        hi = np.broadcast_to(np.asarray(hi, dtype=float), (n,)).copy()
    except ValueError:
        return None
    return lo, hi


def _choose_method(kwargs: dict, bounds, n: int | None) -> str | None:
    """Put the chosen local method into ``kwargs`` and return the global
    search to run first (or None). Records :func:`last_method`."""
    if "method" in kwargs:
        _LAST_METHOD.update(method=kwargs["method"], note="")
        return None
    chosen = method()
    bounded = _bounded(bounds)
    note = ""
    search = None
    lm_blocker = ""
    if chosen == "lm":
        if bounded:
            lm_blocker = "cannot apply bounds, and this fit has them"
        elif kwargs.get("loss", "linear") != "linear":
            lm_blocker = "cannot use a robust loss, and this fit does"
        elif kwargs.get("jac_sparsity") is not None or kwargs.get("tr_solver") not in (None, "exact"):
            lm_blocker = "cannot use this fit's sparse solver"
    if lm_blocker:
        kwargs["method"] = "trf"
        note = (f"Levenberg-Marquardt {lm_blocker}: it used Trust Region Reflective (TRF).")
        used = "trf"
    elif chosen in GLOBAL_METHODS:
        limits = _bounds_arrays(bounds, n) if n else None
        if limits is None or not (np.all(np.isfinite(limits[0])) and np.all(np.isfinite(limits[1]))):
            note = (f"{METHODS[chosen]} needs a finite lower and upper bound on every free "
                    f"parameter, and this fit does not have them: it used the automatic "
                    f"local method.")
            used = "auto"
        else:
            kwargs["method"] = "trf"
            search = chosen
            used = chosen
    elif chosen == "auto":
        used = "auto"
    else:
        kwargs["method"] = chosen
        used = chosen
    _LAST_METHOD.update(method=used, note=note)
    if _ACTIVE:
        _ACTIVE[0].status.method = used
    return search


def _global_search(cost, x0, bounds, search: str) -> np.ndarray:
    """The best point a global ``search`` finds for ``cost`` within the
    (finite) ``bounds``, starting from ``x0`` (clipped into them)."""
    from scipy.optimize import differential_evolution, dual_annealing

    lo, hi = _bounds_arrays(bounds, len(x0))
    x0 = np.clip(np.asarray(x0, dtype=float), lo, hi)
    pairs = list(zip(lo, hi))
    limit = min(max_iterations(), GLOBAL_MAX_ITERATIONS)
    if search == "differential_evolution":
        result = differential_evolution(cost, pairs, x0=x0, maxiter=limit, seed=0,
                                        polish=False, updating="immediate")
    else:
        result = dual_annealing(cost, pairs, x0=x0, maxiter=limit, seed=0)
    return np.asarray(result.x, dtype=float)


#: Notebook-friendly names (``nmr.set_fit_method("dogbox")``): the bare
#: ``method`` / ``set_method`` are too generic for ``from ... import *``.
FIT_METHODS = METHODS
fit_method = method
set_fit_method = set_method
last_fit_method = last_method


def least_squares(fun, x0, **kwargs):
    """``scipy.optimize.least_squares`` with the chosen fitting algorithm
    (:func:`method`) -- for fitters with their own ``max_nfev`` that do
    not go through :func:`limited_least_squares`."""
    from scipy.optimize import least_squares as _least_squares

    search = _choose_method(kwargs, kwargs.get("bounds"), len(np.atleast_1d(x0)))
    if search is not None:
        args, kw = kwargs.get("args", ()), kwargs.get("kwargs", {}) or {}

        def cost(p):
            try:
                r = np.asarray(fun(p, *args, **kw), dtype=float).ravel()
                value = float(np.dot(r, r))
            except (ArithmeticError, ValueError, TypeError):
                return 1e300              # a trial point the model cannot evaluate
            return value if math.isfinite(value) else 1e300

        x0 = _global_search(cost, x0, kwargs["bounds"], search)
    return _least_squares(fun, x0, **kwargs)


def _refuse_after_keep_stop(label: str) -> None:
    if stopped_early():
        watcher = _ACTIVE[0]
        watcher.keep_requested = True
        watcher.status.stopped_early = True
        raise FitStopped(f"{label} was not run: the fit was stopped, keeping the best so far.")


def max_iterations(value: int | None = None) -> int:
    """The iteration limit to use: ``value`` when given, else the one set
    with :func:`set_max_iterations` (2000 unless changed)."""
    if value is None:
        return _MAX_ITERATIONS[-1]
    value = int(value)
    if value < 1:
        raise ValueError(f"The maximum number of iterations must be at least 1 (got {value}).")
    return value


def set_max_iterations(value: int | None) -> int:
    """Set the app-wide limit (``None`` restores 2000). Returns it."""
    _MAX_ITERATIONS[0] = DEFAULT_MAX_ITERATIONS if value is None else max_iterations(value)
    del _MAX_ITERATIONS[1:]
    return _MAX_ITERATIONS[0]


@contextmanager
def iteration_limit(value: int):
    """Use ``value`` as the limit for the fits inside this block only."""
    _MAX_ITERATIONS.append(max_iterations(value))
    try:
        yield _MAX_ITERATIONS[-1]
    finally:
        _MAX_ITERATIONS.pop()


def not_converged_suggestions(limit: int, *, n_free: int | None = None,
                              start_cost: float = float("nan"),
                              best_cost: float = float("nan")) -> list[str]:
    """What to try when a fit stops at ``limit`` iterations without
    converging -- plain language, most useful first."""
    out = []
    if math.isfinite(start_cost) and math.isfinite(best_cost) and start_cost > 0:
        frac = best_cost / start_cost
        if frac < 0.999:
            out.append(f"It was still improving (the misfit had come down to {100 * frac:.3g} % "
                       f"of its starting value), so it may only need longer: raise the maximum "
                       f"number of iterations (Settings ▸ Fitting in the app; "
                       f"max_iterations= or set_max_iterations() in a notebook), e.g. to "
                       f"{2 * int(limit)}.")
        else:
            out.append("The misfit hardly moved from its starting value, so more iterations are "
                       "unlikely to help on their own: the starting point is probably too far "
                       "from the answer.")
    else:
        out.append(f"Raise the maximum number of iterations (Settings ▸ Fitting in the app; "
                   f"max_iterations= or set_max_iterations() in a notebook), e.g. to "
                   f"{2 * int(limit)}, if it was still improving.")
    out.append("Start closer to the answer: check the starting values against the spectrum "
               "(shift, width, anisotropy), or fit a simpler model first and start the full "
               "model from its result.")
    many = f" ({n_free} are free now)" if n_free else ""
    out.append(f"Free fewer parameters{many}: fix or link the ones the data cannot pin down "
               "(e.g. η with weak sidebands, Euler angles, a minor site's width), fit, then "
               "free them one at a time.")
    out.append("Check the limits: a parameter pushing against a bound can stall a fit – "
               "widen that bound, or fix the parameter if the bound is physical.")
    out.append("Check the data: phase and baseline errors, or a fit window that cuts off part "
               "of the signal (sidebands included), leave no clean minimum to converge to.")
    out.append("Try several starting points (multi-start), in case this one sits in a poor "
               "local minimum.")
    return out


class FitNotConverged(RuntimeError):
    """A fit reached its maximum number of iterations without converging.

    A ``RuntimeError``, as scipy's own non-convergence is, so a
    multi-start fit still skips such a start and carries on.
    ``str()`` gives the message *and* the suggestions, so anywhere that
    shows an error shows what to try next.

    ``best_params`` -- the best (free) parameters reached, to restart from.
    """

    def __init__(self, limit: int, *, label: str = "The fit", n_free: int | None = None,
                 best_params=None, start_cost: float = float("nan"),
                 best_cost: float = float("nan")) -> None:
        self.max_iterations = int(limit)
        self.label = label
        self.n_free = n_free
        self.best_params = None if best_params is None else np.array(best_params, dtype=float)
        self.start_cost = float(start_cost)
        self.best_cost = float(best_cost)
        self.suggestions = not_converged_suggestions(
            self.max_iterations, n_free=n_free, start_cost=self.start_cost,
            best_cost=self.best_cost)
        super().__init__(self._text())

    @property
    def headline(self) -> str:
        return (f"{self.label} did not converge within the maximum of "
                f"{self.max_iterations:,} iteration{'' if self.max_iterations == 1 else 's'}.")

    def _text(self) -> str:
        return self.headline + "\n\nWhat to try next:\n" + "\n".join(
            f"• {line}" for line in self.suggestions)


def _begin_limited_fit(limit: int) -> None:
    """Tell the watcher (if any) a new single fit starts, held to ``limit``."""
    if _ACTIVE:
        st = _ACTIVE[0].status
        st.max_iterations = int(limit)
        st.iteration_offset = st.evaluations


def _is_limit_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return ("maximum number of function evaluations" in text
            or "number of calls to function has reached maxfev" in text)


class _Best:
    """Remembers the lowest-misfit parameters a model was called with."""

    def __init__(self, ydata, sigma=None):
        self.y = np.asarray(ydata, dtype=float)
        self.w = None if sigma is None else 1.0 / np.asarray(sigma, dtype=float)
        self.start = float("nan")
        self.cost = float("nan")
        self.params = None

    def note(self, params, residual) -> None:
        try:
            r = np.asarray(residual, dtype=float).ravel()
            cost = float(np.dot(r, r))
        except (TypeError, ValueError):
            return
        if not math.isfinite(cost):
            return
        if not math.isfinite(self.start):
            self.start = cost
        if not math.isfinite(self.cost) or cost < self.cost:
            self.cost, self.params = cost, np.array(params, dtype=float)


def _bounded(bounds) -> bool:
    if bounds is None:
        return False
    lo, hi = bounds
    return bool(np.any(np.isfinite(lo)) or np.any(np.isfinite(hi)))


def limited_curve_fit(f, xdata, ydata, p0=None, *, bounds=None, max_iterations: int | None = None,
                      label: str = "The fit", **kwargs):
    """``scipy.optimize.curve_fit`` held to :func:`max_iterations`.

    Reaching the limit raises :class:`FitNotConverged` (with suggestions)
    in place of scipy's bare ``RuntimeError``. Any ``maxfev``/``max_nfev``
    passed is replaced by the limit."""
    from scipy.optimize import curve_fit

    limit = globals()["max_iterations"](max_iterations)
    kwargs.pop("maxfev", None)
    kwargs.pop("max_nfev", None)
    n_free = len(np.atleast_1d(p0)) if p0 is not None else None
    search = _choose_method(kwargs, bounds, n_free)
    method = kwargs.get("method")
    if method == "lm" or (method is None and not _bounded(bounds)):
        # Levenberg-Marquardt counts every call, Jacobian ones included.
        kwargs["maxfev"] = limit * ((n_free or 1) + 1)
    else:
        kwargs["max_nfev"] = limit
    if bounds is not None:
        kwargs["bounds"] = bounds
    _apply_tolerance(kwargs)
    _refuse_after_keep_stop(label)
    best = _Best(ydata, kwargs.get("sigma"))
    _begin_limited_fit(limit)

    def remembered(x, *params):
        out = f(x, *params)
        try:
            r = np.asarray(out, dtype=float) - best.y
            if best.w is not None:
                r = r * best.w
        except (TypeError, ValueError):
            return out
        best.note(params, r)
        return out

    try:
        if search is not None:
            sigma = kwargs.get("sigma")
            weight = None if sigma is None else 1.0 / np.asarray(sigma, dtype=float)
            y = np.asarray(ydata, dtype=float)

            def cost(params):
                try:
                    r = np.asarray(remembered(xdata, *params), dtype=float) - y
                    if weight is not None:
                        r = r * weight
                    value = float(np.dot(r.ravel(), r.ravel()))
                except (ArithmeticError, ValueError, TypeError):
                    return 1e300          # a trial point the model cannot evaluate
                return value if math.isfinite(value) else 1e300

            p0 = _global_search(cost, p0, bounds, search)
        return curve_fit(remembered, xdata, ydata, p0=p0, **kwargs)
    except _StopKeepingBest:
        # Stopped part-way, keeping the best reached: no uncertainty.
        if best.params is None:
            raise FitCancelled("The fit was stopped before it had evaluated anything.")
        n = best.params.size
        return best.params.copy(), np.full((n, n), np.nan)
    except RuntimeError as exc:
        if isinstance(exc, FitNotConverged) or not _is_limit_error(exc):
            raise
        raise FitNotConverged(limit, label=label, n_free=n_free, best_params=best.params,
                              start_cost=best.start, best_cost=best.cost) from exc


def limited_least_squares(fun, x0, *, max_iterations: int | None = None, label: str = "The fit",
                          raise_on_limit: bool = True, **kwargs):
    """``scipy.optimize.least_squares`` held to :func:`max_iterations`.

    Reaching the limit raises :class:`FitNotConverged` unless
    ``raise_on_limit=False``, when the solution is returned with
    ``solution.not_converged`` set to that exception (so its suggestions
    can go in the fit's warnings) -- for fitters that already report an
    unconverged fit rather than discarding it."""
    from scipy.optimize import least_squares

    limit = globals()["max_iterations"](max_iterations)
    kwargs.pop("max_nfev", None)
    search = _choose_method(kwargs, kwargs.get("bounds"), len(np.atleast_1d(x0)))
    _apply_tolerance(kwargs)
    _refuse_after_keep_stop(label)
    start = [float("nan")]
    best = _Best(np.zeros(0))
    _begin_limited_fit(limit)

    def remembered(p, *args, **kw):
        out = fun(p, *args, **kw)
        try:
            r = np.asarray(out, dtype=float).ravel()
            cost = float(np.dot(r, r))
        except (TypeError, ValueError):
            return out
        if not math.isfinite(start[0]):
            start[0] = cost
        if math.isfinite(cost) and (best.params is None or cost < best.cost):
            best.cost, best.params = cost, np.array(p, dtype=float)
        return out

    try:
        if search is not None:
            args, kw = kwargs.get("args", ()), kwargs.get("kwargs", {}) or {}

            def cost(p):
                try:
                    r = np.asarray(remembered(p, *args, **kw), dtype=float).ravel()
                    value = float(np.dot(r, r))
                except (ArithmeticError, ValueError, TypeError):
                    return 1e300          # a trial point the model cannot evaluate
                return value if math.isfinite(value) else 1e300

            x0 = _global_search(cost, x0, kwargs["bounds"], search)
        solution = least_squares(remembered, x0, max_nfev=limit, **kwargs)
    except _StopKeepingBest:
        if best.params is None:
            raise FitCancelled("The fit was stopped before it had evaluated anything.")
        return _stopped_solution(fun, best.params, kwargs, start[0], best.cost, label)
    solution.not_converged = None
    if solution.status == 0:
        exc = FitNotConverged(limit, label=label, n_free=len(np.atleast_1d(x0)),
                              best_params=solution.x, start_cost=start[0],
                              best_cost=2.0 * float(solution.cost))
        if raise_on_limit:
            raise exc
        solution.not_converged = exc
    return solution


def _stopped_solution(fun, x, kwargs: dict, start_cost: float, best_cost: float, label: str):
    """What :func:`limited_least_squares` returns for a fit stopped part-way
    keeping the best so far: an ``OptimizeResult`` at ``x`` with the
    residuals and a finite-difference Jacobian there (fitters read them),
    ``success`` False and a message saying what happened. The watcher
    counts none of these evaluations."""
    from scipy.optimize import OptimizeResult

    args, kw = kwargs.get("args", ()), kwargs.get("kwargs", {}) or {}
    x = np.asarray(x, dtype=float)
    r0 = np.asarray(fun(x, *args, **kw), dtype=float).ravel()
    jac = np.empty((r0.size, x.size))
    for k in range(x.size):
        h = 1e-7 * max(abs(x[k]), 1.0)
        step = x.copy()
        step[k] += h
        jac[:, k] = (np.asarray(fun(step, *args, **kw), dtype=float).ravel() - r0) / h
    fall = (f" (the misfit at {100 * best_cost / start_cost:.3g} % of its starting value)"
            if math.isfinite(start_cost) and start_cost > 0 else "")
    solution = OptimizeResult(
        x=x, fun=r0, jac=jac, cost=0.5 * float(np.dot(r0, r0)), status=-2, success=False,
        nfev=0, message=f"{label} was stopped before converging, keeping the best parameters "
                        f"reached{fall}; its uncertainties do not apply.")
    solution.not_converged = None
    solution.stopped_early = True
    return solution
