"""Ask before a long simulation or fit, with how long it will take
(explicit user request, 2026-11-01; the estimates are
:mod:`opennmr.notebook.run_time`).

Each Solid-state NMR "Simulate" and "Fit" first works out how long the run
will take. A quick run is checked with one small trial simulation (~20 ms
on a desktop) and starts at once; a run estimated above
:data:`~opennmr.notebook.run_time.SIMULATION_WARN_S` (simulation) or with
an upper estimate above :data:`~opennmr.notebook.run_time.FIT_WARN_S`
(fit) asks first -- "Run" or "Cancel", with what would make it faster. The
estimate and the time actually taken go into the result's report and log.

``OPENNMR_CONFIRM_LONG_RUNS=0`` skips the estimate entirely (the test
suite, so ordinary tests never wait on a question; tests of this module
turn it back on).
"""

from __future__ import annotations

import math
import os

from PySide6.QtWidgets import QMessageBox

from ..notebook import run_time

#: Environment switch: "0" turns estimates and questions off.
ENV = "OPENNMR_CONFIRM_LONG_RUNS"

SIMULATION_TIPS = (
    "To make it faster: lower Orientations; under MAS, a peak without a quadrupolar "
    "interaction needs far fewer with Lebedev (Powder averaging); 'Exact' quadrupolar "
    "theory is the slowest level.")
FIT_TIPS = (
    "To make it faster: fit with fewer Orientations first and refine with more; fix the "
    "parameters you already know; with several starts, set 'Screen starts at' to a "
    "smaller orientation count.")


def enabled() -> bool:
    return os.environ.get(ENV, "1").strip() != "0"


def ask(parent, title: str, what: str, text: str, tips: str) -> bool:
    """The question; True to run. 'Run' is the default button."""
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Question)
    box.setWindowTitle(title)
    box.setText(f"This {what} will take {text}.")
    box.setInformativeText(tips + "\n\nRun it now?")
    run = box.addButton("Run", QMessageBox.ButtonRole.AcceptRole)
    box.addButton(QMessageBox.StandardButton.Cancel)
    box.setDefaultButton(run)
    box.exec()
    return box.clickedButton() is run


def check_simulation(parent, sites, axis, settings: dict, runs: int = 1):
    """Estimate a combined simulation (``runs`` of them: the order
    comparison runs three). Returns the estimate dict (or None when
    estimates are off or failed), or False if the user cancelled."""
    if not enabled():
        return None
    try:
        est = run_time.estimate_simulation_time(
            sites, axis, stop_below=run_time.SIMULATION_WARN_S / max(runs, 1), **settings)
    except Exception:  # noqa: BLE001 -- an estimate must never stop the run itself
        return None
    est = dict(est, seconds=est["seconds"] * runs)
    if est["seconds"] >= run_time.SIMULATION_WARN_S and not est["bound"]:
        what = "simulation" if runs == 1 else f"comparison ({runs} simulations)"
        text = est["text"].replace(run_time.format_duration(est["seconds"] / runs),
                                   run_time.format_duration(est["seconds"]), 1)
        if not ask(parent, "Solid-state NMR", what, text, SIMULATION_TIPS):
            return False
    return est


def n_starts(starts) -> int:
    """How many starting points a multi-start fit tries."""
    if not starts:
        return 1
    return max(math.prod(len(v) for per in starts.values() for v in per.values()), 1)


def check_fit(parent, results, sites, kwargs: dict, starts=None, screen_n_zcw=None,
              title: str = "Solid-state NMR fit", experiments: int = 1, joint: bool = False):
    """Estimate a combined fit (``experiments`` of them in a row: the
    multi-spectrum fit, estimated from its first spectrum; ``joint``: then
    refined all together, about as long again). Returns the estimate,
    None, or False (cancelled)."""
    if not enabled():
        return None
    times = experiments * (2 if joint else 1)
    try:
        est = run_time.estimate_fit_time(
            results, sites, starts=n_starts(starts), screen_n_zcw=screen_n_zcw,
            stop_below=run_time.FIT_WARN_S / max(times, 1), **kwargs)
    except Exception:  # noqa: BLE001
        return None
    if times > 1:
        how = (f"{experiments} spectra fitted one by one" if experiments > 1 else "one spectrum")
        if joint:
            how += ", then refined together"
        est = dict(est, low=est["low"] * times, high=est["high"] * times,
                   text=(f"about {run_time.format_duration(est['low'] * times)} to "
                         f"{run_time.format_duration(est['high'] * times)} "
                         f"({how}; the first spectrum alone: {est['text']})"))
    if est["high"] >= run_time.FIT_WARN_S and not est["bound"]:
        what = "fit" if experiments == 1 else "multi-spectrum fit"
        if not ask(parent, title, what, est["text"], FIT_TIPS):
            return False
    return est


def report_line(estimate, elapsed: float) -> str:
    """'Took 42 s (estimated about 30 s to 1 min 50 s).' -- or '' if no
    estimate was made or it was only a quick bound."""
    if not estimate or estimate.get("bound"):
        return ""
    if "low" in estimate:
        guess = (f"{run_time.format_duration(estimate['low'])} to "
                 f"{run_time.format_duration(estimate['high'])}")
    else:
        guess = run_time.format_duration(estimate["seconds"])
    return f"Took {run_time.format_duration(elapsed)} (estimated {guess})."
