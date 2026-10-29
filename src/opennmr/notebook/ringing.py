"""
Is a 2D spectrum's F2 ringing (truncation "wiggles")? And what window removes it?

Rows of a 2D spectrum sometimes show wiggles beside strong lines that look like a multiplet
and are not in the TopSpin spectrum. The usual cause is an FID cut off before it has decayed
and transformed with no (or too gentle) a window; a sine-bell removes it. The Process tab's
Direct dimension ▸ "Check for truncation wiggles" button runs exactly this.

    from opennmr.notebook import ringing
    d = ringing.check_f2_ringing(spectrum)       # a raw-FID Bruker 2D (or 1D) NMRSpectrum
    print(d["report"])
    d["severity"], d["ripple_windowed"], d["period_hz"], d["suggestion"]

and to apply what it suggests::

    s = d["suggestion"]
    spectrum = bruker.read_2d(path, window=s["window"], window_kwargs={"ssb": s["ssb"]})

It looks at the F2 FID as it is processed now (the live Process settings, else TopSpin's
recorded window and TDeff), so "window in use" in the report is what the spectrum on screen has.
It also says so when a window TopSpin recorded could not be reproduced here -- the one case where
openNMR's F2 differs from TopSpin's by construction.
"""

from __future__ import annotations

import numpy as np

from ..processing import ringing as _ringing
from ..processing.ringing import (  # noqa: F401 -- the same functions, for notebooks
    diagnose_truncation, suggest_window, ripple_fraction, window_weights, noise_bandwidth,
)

_WDW_NAMES = {0: "none", 1: "EM", 2: "GM", 3: "SINE", 4: "QSINE", 5: "TRAF", 6: "user-defined"}


def _recorded_window(procs: dict, n: int, dt: float):
    """``(weights, description)`` of the window TopSpin recorded in ``procs`` (``WDW``, ``LB``,
    ``SSB``) for ``n`` points; weights ``None`` if openNMR cannot reproduce it."""
    wdw = int(float(procs.get("WDW", 0) or 0))
    name = _WDW_NAMES.get(wdw, f"WDW {wdw}")
    if wdw == 0:
        return np.ones(n), "none (TopSpin recorded no window)"
    if wdw == 1:
        lb = float(procs.get("LB", 0.0) or 0.0)
        return window_weights("exponential", n, dt, lb=lb), f"EM, LB {lb:g} Hz (TopSpin's own)"
    if wdw in (3, 4):
        ssb = float(procs.get("SSB", 0.0) or 0.0)
        key = "sine" if wdw == 3 else "qsine"
        return window_weights(key, n, dt, ssb=ssb), f"{name}, SSB {ssb:g} (TopSpin's own)"
    return None, f"{name} (TopSpin's own; openNMR cannot reproduce it)"


def _live_window(raw: dict, n: int, dt: float):
    """``(weights, description)`` of the window the live Process settings give, else the recorded."""
    key = raw.get("live_window")
    if key in (None, "__recorded__"):
        return _recorded_window(raw.get("procs") or {}, n, dt)
    if key == "none":
        return np.ones(n), "none (chosen in Process)"
    params = {}
    if key == "exponential":
        params = {"lb": float(raw.get("live_lb", 0.0) or 0.0)}
    elif key == "gaussian":
        params = {"lb": float(raw.get("live_lb", 0.0) or 0.0), "gb": float(raw.get("live_gb", 0.0) or 0.0)}
    elif key in ("sine", "qsine"):
        params = {"ssb": float(raw.get("live_ssb", 0.0) or 0.0)}
    text = ", ".join(f"{k} {v:g}" for k, v in params.items())
    return window_weights(key, n, dt, **params), f"{key}{' (' + text + ')' if text else ''} (chosen in Process)"


def check_f2_ringing(spectrum, *, procno: int = 1, window="current", window_params: dict | None = None,
                     truncate: int | None = None, **options) -> dict:
    """Diagnose truncation wiggles in the direct dimension of ``spectrum``.

    ``spectrum``: a raw-FID :class:`~opennmr.core.spectrum.NMRSpectrum` (a Bruker 2D read by
    ``read_2d``, or a 1D) -- or a path, read with ``bruker.read_2d``. ``window``: ``"current"``
    (the window the spectrum is processed with now, the default), ``"none"``, or a window
    name (``"sine"``, ``"qsine"``, ``"exponential"``...) with ``window_params`` (``{"ssb": 2}``)
    to ask "what would this window leave?". ``truncate``: points to use (default: what the
    spectrum uses -- Process's Truncate, else TopSpin's TDeff, else all). ``options``:
    :func:`opennmr.processing.ringing.diagnose_truncation`'s (``tail_fraction``,
    ``strongest_rows``).

    Returns :func:`~opennmr.processing.ringing.diagnose_truncation`'s dict plus
    ``window_in_use`` (a description), ``window_unsupported`` (the reader's note when a
    recorded TopSpin window could not be applied, else ``None``) and ``report`` (sentences)."""
    if isinstance(spectrum, str):
        from ..io import bruker
        spectrum = bruker.read_2d(spectrum, procno=procno)
    raw = (spectrum.metadata.raw or {})
    fid = raw.get("fid_raw")
    if fid is None:
        raise ValueError("this spectrum has no raw FID to check (it was not read from a raw FID, "
                         "or is a processed-only dataset)")
    fid = np.asarray(fid)
    dt = float(raw.get("fid_dt") or 0.0)
    if dt <= 0:
        raise ValueError("the spectrum does not record its dwell time (fid_dt)")
    n_total = fid.shape[-1]
    if truncate is None:
        truncate = raw.get("live_truncate") or raw.get("tdeff_f2_points") or None
    n = min(int(truncate), n_total) if truncate else n_total

    if window == "current":
        weights, description = _live_window(raw, n, dt)
    elif window in (None, "none"):
        weights, description = np.ones(n), "none"
    else:
        weights = window_weights(window, n, dt, **(window_params or {}))
        text = ", ".join(f"{k} {v:g}" for k, v in (window_params or {}).items())
        description = f"{window}{' (' + text + ')' if text else ''} (as asked)"
    unsupported = raw.get("window_unsupported") if window == "current" else None

    diagnosis = diagnose_truncation(
        fid, 1.0 / dt, sf_mhz=getattr(spectrum.metadata, "sfo1", None), truncate=n,
        window_weights=weights, **options)
    diagnosis["window_in_use"] = description
    diagnosis["window_unsupported"] = unsupported
    header = [f"Window in use: {description}."]
    if weights is None and unsupported is None:
        header.append("That window cannot be reproduced here, so F2 was left without one.")
    if unsupported:
        header.append(f"{unsupported} F2 is therefore processed WITHOUT a window here but with one in "
                      f"TopSpin, which would explain wiggles that TopSpin does not show.")
    diagnosis["report"] = "\n".join(header + [diagnosis["report"]])
    return diagnosis
