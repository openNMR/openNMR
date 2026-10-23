"""
Field drift along the slices of a pseudo-2D, measured from the data and removed.

A magnet's field drifts, and without a lock every peak of a long series
creeps in frequency. Here the drift is **seen first**: the position of a
chosen peak is read off every slice and plotted against the slice number; a
smooth curve (a line, a polynomial, a spline, a sinusoid) is fitted through
those positions, and each slice is shifted back by what the curve says it
drifted. Nothing is assumed about the vendor: it works on the spectra, so it
serves Bruker, X-Pulse, Varian, JEOL, Spinsolve, SIMPSON and files alike.

    from opennmr.notebook import field_drift as fd

    track = fd.track_peak(spectrum, region_ppm=(5.2, 4.4))        # position per slice
    fit = fd.fit_drift(track["row"], track["hz"], model="polynomial", degree=3)
    fd.apply_drift(spectrum, fit["shifts_hz"], record=fit)           # shift every slice back
    fd.plot_drift(track, fit)                                       # to look at it
    fd.remove_drift(spectrum)                                       # undo

or in one call: ``fd.correct_field_drift(spectrum, (5.2, 4.4), model="spline")``.

Sign convention (as in :mod:`opennmr.notebook.drift_correction`): a shift is
``reference - observed`` in Hz of the detected nucleus, and a positive shift
moves a peak to higher frequency (left on the usual plot).

For a series whose raw FIDs are still at hand, the time-domain correction of
:mod:`opennmr.notebook.drift_correction` is exact and survives reprocessing;
this one shifts the spectra themselves (a phase ramp on the inverse transform of
each row, so a shift of a fraction of a point is exact, not interpolated).
"""

from __future__ import annotations

import numpy as np

MODELS = ("linear", "polynomial", "spline", "spline_points", "sinusoidal")
MODEL_LABELS = {
    "linear": "Linear",
    "polynomial": "Polynomial",
    "spline": "Spline (automatic)",
    "spline_points": "Spline (chosen slices)",
    "sinusoidal": "Sinusoidal (+ linear)",
}

#: ``metadata.raw`` keys: the record of what was done, and the rows as they were.
RECORD_KEY = "field_drift"
ORIGINAL_KEY = "_field_drift_original"


# ----------------------------------------------------------------------
# Tracking a peak along the slices
# ----------------------------------------------------------------------

def _row_axis_hz(spectrum) -> tuple[np.ndarray, np.ndarray]:
    """``(ppm axis, Hz axis)`` of the detected dimension."""
    ppm = np.asarray(spectrum.axes[-1].ppm, dtype=float)
    sfo1 = float(spectrum.metadata.sfo1 or 1.0)
    hz = spectrum.axes[-1].hz
    hz = np.asarray(hz, dtype=float) if hz is not None else ppm * sfo1
    return ppm, hz


def track_peak(spectrum, region_ppm: tuple[float, float], method: str = "peak",
               rows=None, original: bool = False) -> dict:
    """Position of the peak in ``region_ppm`` in every slice of a pseudo-2D.

    ``method`` : ``"peak"`` the tallest point refined by a parabola through
    its top three (the default), or ``"centroid"`` the intensity-weighted
    centre of the part of the region above half the peak height (steadier
    for a noisy or broad line). The tallest *magnitude* is used, so a peak
    that changes sign along the series is still followed.

    Returns ``{"row", "hz", "ppm", "intensity", "method", "region_ppm"}``:
    0-based slice numbers, and the position of each in Hz (on the axis of the
    spectrum as it stands, so an SR applied to it enters equally in all
    slices) and in ppm. Choose a region that holds the peak in every slice and
    no other peak.

    ``original=True`` reads the slices as they were before :func:`apply_drift`,
    so the drift that was corrected can still be looked at."""
    if spectrum.ndim != 2:
        raise ValueError("track_peak needs a pseudo-2D (one spectrum per row).")
    if method not in ("peak", "centroid"):
        raise ValueError("method must be 'peak' or 'centroid'.")
    ppm, hz = _row_axis_hz(spectrum)
    lo, hi = sorted(float(v) for v in region_ppm)
    mask = (ppm >= lo) & (ppm <= hi)
    idx = np.flatnonzero(mask)
    if idx.size < 3:
        raise ValueError(f"Fewer than three points between {lo} and {hi} ppm.")
    kept = spectrum.metadata.raw.get(ORIGINAL_KEY) if original else None
    data = np.asarray(kept[0] if kept is not None else spectrum.data, dtype=float)
    selected = np.arange(data.shape[0]) if rows is None else np.asarray(list(rows), dtype=int)
    out_hz, out_ppm, out_i = [], [], []
    for r in selected:
        y = data[r]
        mag = np.abs(y[idx])
        k = int(idx[int(np.argmax(mag))])
        if method == "centroid":
            weights = np.abs(y[idx])
            keep = weights >= 0.5 * weights.max()
            w = weights[keep]
            hz_pos = float(np.sum(hz[idx][keep] * w) / np.sum(w))
            ppm_pos = float(np.sum(ppm[idx][keep] * w) / np.sum(w))
        else:
            hz_pos, ppm_pos = float(hz[k]), float(ppm[k])
            if idx[0] < k < idx[-1]:
                ya, yb, yc = abs(y[k - 1]), abs(y[k]), abs(y[k + 1])
                denom = ya - 2 * yb + yc
                offset = 0.5 * (ya - yc) / denom if denom else 0.0
                offset = float(np.clip(offset, -1.0, 1.0))
                hz_pos = float(hz[k] + offset * (hz[k + 1] - hz[k]))
                ppm_pos = float(ppm[k] + offset * (ppm[k + 1] - ppm[k]))
        out_hz.append(hz_pos)
        out_ppm.append(ppm_pos)
        out_i.append(float(abs(y[k])))
    return {"row": selected.astype(float), "hz": np.asarray(out_hz), "ppm": np.asarray(out_ppm),
            "intensity": np.asarray(out_i), "method": method, "region_ppm": (lo, hi)}


# ----------------------------------------------------------------------
# Fitting the drift curve
# ----------------------------------------------------------------------

def _basis_sine(x, period):
    w = 2.0 * np.pi / period
    return np.column_stack([np.ones_like(x), x, np.sin(w * x), np.cos(w * x)])


def _fit_sinusoid(x, y, period=None):
    """``a + b x + A sin(2 pi x / P + phi)``: P by a scan, then least squares."""
    from scipy.optimize import least_squares

    span = float(x.max() - x.min()) or 1.0
    step = float(np.median(np.diff(np.sort(x)))) if x.size > 1 else 1.0
    periods = [float(period)] if period else list(np.geomspace(max(3 * step, span / 50), 3 * span, 200))
    best = None
    for p in periods:
        basis = _basis_sine(x, p)
        coef, *_ = np.linalg.lstsq(basis, y, rcond=None)
        rss = float(np.sum((basis @ coef - y) ** 2))
        if best is None or rss < best[0]:
            best = (rss, p, coef)
    _rss, p, coef = best
    if period:
        return p, coef

    def residual(theta):
        return _basis_sine(x, theta[0]) @ theta[1:] - y

    solution = least_squares(residual, np.concatenate([[p], coef]))
    return float(solution.x[0]), solution.x[1:]


def fit_drift(x, y_hz, model: str = "linear", degree: int = 3, n_points: int = 6,
              node_x=None, period: float | None = None, reference="first") -> dict:
    """Fit a smooth curve through the peak positions ``y_hz`` (Hz) at slice numbers ``x``.

    ``model`` : ``"linear"``; ``"polynomial"`` (``degree``); ``"spline"`` an
    automatic smoothing spline (its smoothness chosen from the data by
    cross-validation, so it follows a real curve without chasing the scatter);
    ``"spline_points"`` a spline through ``node_x`` (the slice numbers you chose;
    each node takes the median of its neighbours' positions, ``n_points``
    equally spaced nodes if none are given); ``"sinusoidal"`` a line plus
    one sinusoid (``period`` in slices if you know it, else found), for a drift
    that oscillates, e.g. with the temperature of the room.

    ``reference`` : which position the slices are brought to: ``"first"``
    (the fitted position of the first slice; default), ``"mean"``, ``"last"``,
    or a number of Hz.

    Returns ``{"model", "x", "y", "fit", "residual", "rms", "reference_hz",
    "shifts_hz", "params", "predict"}``. ``shifts_hz[i]`` is
    ``reference - fit[i]``: what to add to slice ``i``'s frequency to put its
    peak where the reference is. ``predict(x)`` evaluates the curve anywhere."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y_hz, dtype=float)
    if x.size != y.size or x.size < 2:
        raise ValueError("fit_drift needs matching x and y with at least two points.")
    if model not in MODELS:
        raise ValueError(f"model must be one of {MODELS}.")
    params: dict = {}
    if model == "linear":
        coef = np.polyfit(x, y, 1)
        predict = lambda q, c=coef: np.polyval(c, np.asarray(q, float))  # noqa: E731
        params = {"slope_hz_per_slice": float(coef[0]), "offset_hz": float(coef[1])}
    elif model == "polynomial":
        degree = int(max(1, min(degree, x.size - 1)))
        coef = np.polyfit(x, y, degree)
        predict = lambda q, c=coef: np.polyval(c, np.asarray(q, float))  # noqa: E731
        params = {"degree": degree, "coefficients": [float(c) for c in coef]}
    elif model == "spline":
        order = np.argsort(x)
        xs, ys = x[order], y[order]
        if x.size < 5:
            coef = np.polyfit(x, y, 1)
            predict = lambda q, c=coef: np.polyval(c, np.asarray(q, float))  # noqa: E731
            params = {"note": "fewer than five slices: a line was used"}
        else:
            from scipy.interpolate import make_smoothing_spline
            spline = make_smoothing_spline(xs, ys)
            predict = lambda q, s=spline: s(np.asarray(q, float))  # noqa: E731
            params = {"lambda": float(getattr(spline, "lam", np.nan))
                      if hasattr(spline, "lam") else None}
    elif model == "spline_points":
        from scipy.interpolate import CubicSpline
        if node_x is None or len(node_x) < 2:
            nodes = np.linspace(x.min(), x.max(), max(int(n_points), 2))
        else:
            nodes = np.unique(np.clip(np.asarray(node_x, float), x.min(), x.max()))
        span = float(x.max() - x.min()) or 1.0
        half = max(span / (4 * max(len(nodes), 2)), float(np.median(np.diff(np.sort(x)))) * 1.01)
        values = [float(np.median(y[np.abs(x - n) <= half])) if np.any(np.abs(x - n) <= half)
                  else float(np.interp(n, np.sort(x), y[np.argsort(x)])) for n in nodes]
        if len(nodes) >= 3:
            spline = CubicSpline(nodes, values, bc_type="natural", extrapolate=True)
            predict = lambda q, s=spline: s(np.asarray(q, float))  # noqa: E731
        else:
            predict = lambda q, n=nodes, v=values: np.interp(np.asarray(q, float), n, v)  # noqa: E731
        params = {"nodes": [float(n) for n in nodes], "values_hz": values}
    else:  # sinusoidal
        p, coef = _fit_sinusoid(x, y, period)
        predict = lambda q, p=p, c=coef: _basis_sine(np.atleast_1d(np.asarray(q, float)), p) @ c  # noqa: E731
        params = {"period_slices": float(p), "offset_hz": float(coef[0]),
                  "slope_hz_per_slice": float(coef[1]),
                  "amplitude_hz": float(np.hypot(coef[2], coef[3]))}

    fit = np.asarray(predict(x), dtype=float)
    residual = y - fit
    if reference == "first":
        ref_hz = float(np.ravel(predict(x.min()))[0])
    elif reference == "last":
        ref_hz = float(np.ravel(predict(x.max()))[0])
    elif reference == "mean":
        ref_hz = float(np.mean(fit))
    else:
        ref_hz = float(reference)
    return {"model": model, "x": x, "y": y, "fit": fit, "residual": residual,
            "rms": float(np.sqrt(np.mean(residual ** 2))), "reference_hz": ref_hz,
            "shifts_hz": ref_hz - fit, "params": params, "predict": predict}


# ----------------------------------------------------------------------
# Shifting spectra
# ----------------------------------------------------------------------

def hz_per_point(ppm_axis, sfo1: float) -> float:
    """Signed step of the frequency axis in Hz per point (negative for the
    usual high-to-low ppm axis)."""
    ppm = np.asarray(ppm_axis, dtype=float)
    if ppm.size < 2:
        raise ValueError("The axis needs at least two points.")
    return float((ppm[-1] - ppm[0]) / (ppm.size - 1) * float(sfo1 or 1.0))


def shift_rows(data, shifts_hz, step_hz: float):
    """Shift each row of ``data`` (complex or real, ``(n_rows, n_points)`` or one
    row) by ``shifts_hz[row]`` Hz toward higher frequency, given the axis step
    ``step_hz`` (signed, Hz per point). Returns a new array.

    A complex row is shifted exactly, by a phase ramp on its inverse transform;
    the row is padded by replicating its ends first, so nothing wraps round from
    one side to the other. A real-only row is shifted by cubic interpolation."""
    arr = np.asarray(data)
    single = arr.ndim == 1
    rows = arr[None, :] if single else arr
    shifts = np.broadcast_to(np.asarray(shifts_hz, dtype=float), (rows.shape[0],))
    # Content moves toward higher frequency, i.e. to lower index when the axis falls.
    points = -shifts / step_hz
    out = np.empty_like(rows)
    pad = int(np.ceil(np.max(np.abs(points)))) + 16 if rows.size else 0
    for r in range(rows.shape[0]):
        n = float(points[r])
        if n == 0.0:
            out[r] = rows[r]
            continue
        if np.iscomplexobj(rows):
            padded = np.pad(rows[r], pad, mode="edge")
            f = np.fft.fftfreq(padded.size)
            shifted = np.fft.ifft(np.fft.fft(padded) * np.exp(2j * np.pi * f * n))
            out[r] = shifted[pad:pad + rows.shape[1]]
        else:
            from scipy.ndimage import shift as nd_shift
            out[r] = nd_shift(rows[r], -n, order=3, mode="nearest")
    return out[0] if single else out


def apply_drift(spectrum, shifts_hz, record: dict | None = None) -> dict:
    """Shift every slice of a pseudo-2D ``spectrum`` back by ``shifts_hz`` (Hz,
    one per slice; see :func:`fit_drift`), in place.

    The slices as they were are kept, so :func:`remove_drift` undoes it
    exactly and a second call starts from the original rather than correcting
    twice. What was done is recorded in ``spectrum.metadata.raw["field_drift"]``
    (which a saved state keeps and a reopened spectrum re-applies). Returns the
    record."""
    if spectrum.ndim != 2:
        raise ValueError("Field drift is corrected along the slices of a pseudo-2D.")
    raw = spectrum.metadata.raw
    shifts = np.asarray(shifts_hz, dtype=float).ravel()
    if shifts.size != spectrum.data.shape[0]:
        raise ValueError(f"{shifts.size} shifts for {spectrum.data.shape[0]} slices.")
    if ORIGINAL_KEY in raw:
        remove_drift(spectrum)
    raw[ORIGINAL_KEY] = (np.array(spectrum.data, copy=True),
                         None if spectrum.data_imag is None else np.array(spectrum.data_imag, copy=True))
    step = hz_per_point(spectrum.axes[-1].ppm, spectrum.metadata.sfo1)
    if spectrum.data_imag is not None:
        moved = shift_rows(spectrum.data + 1j * spectrum.data_imag, shifts, step)
        spectrum.data = np.real(moved)
        spectrum.data_imag = np.imag(moved)
    else:
        spectrum.data = shift_rows(spectrum.data, shifts, step)
    info = {"model": (record or {}).get("model", "custom"),
            "shifts_hz": [float(s) for s in shifts],
            "params": _plain((record or {}).get("params", {})),
            "reference_hz": float((record or {}).get("reference_hz", 0.0)),
            "region_ppm": [float(v) for v in (record or {}).get("region_ppm", ())],
            "rms_hz": float((record or {}).get("rms", 0.0))}
    raw[RECORD_KEY] = info
    return info


def _plain(value):
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def remove_drift(spectrum) -> bool:
    """Undo :func:`apply_drift`. ``False`` if nothing had been applied."""
    raw = spectrum.metadata.raw
    kept = raw.pop(ORIGINAL_KEY, None)
    if kept is None:
        raw.pop(RECORD_KEY, None)
        return False
    spectrum.data = kept[0]
    spectrum.data_imag = kept[1]
    raw.pop(RECORD_KEY, None)
    return True


def reapply_record(spectrum) -> bool:
    """Re-apply the ``field_drift`` record of a freshly loaded spectrum (from a
    saved state). ``False`` if there is none, it was already applied, or it
    does not fit this spectrum."""
    raw = spectrum.metadata.raw
    record = raw.get(RECORD_KEY)
    if not record or ORIGINAL_KEY in raw or spectrum.ndim != 2:
        return False
    shifts = record.get("shifts_hz") or []
    if len(shifts) != spectrum.data.shape[0]:
        return False
    apply_drift(spectrum, shifts, record={**record, "rms": record.get("rms_hz", 0.0)})
    raw[RECORD_KEY] = {**raw[RECORD_KEY], **{k: v for k, v in record.items() if k not in ("shifts_hz",)}}
    return True


def correct_field_drift(spectrum, region_ppm, model: str = "linear", method: str = "peak",
                        degree: int = 3, n_points: int = 6, node_x=None,
                        period: float | None = None, reference="first",
                        apply: bool = True) -> dict:
    """Track a peak, fit its drift, and (unless ``apply=False``) correct ``spectrum``.

    Returns ``{"track", "fit"}`` (see :func:`track_peak`, :func:`fit_drift`)."""
    if apply:
        remove_drift(spectrum)        # measure the series as acquired, never an earlier correction
    track = track_peak(spectrum, region_ppm, method=method)
    fit = fit_drift(track["row"], track["hz"], model=model, degree=degree, n_points=n_points,
                    node_x=node_x, period=period, reference=reference)
    fit["region_ppm"] = track["region_ppm"]
    if apply:
        apply_drift(spectrum, fit["shifts_hz"], record=fit)
    return {"track": track, "fit": fit}


# ----------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------

def plot_drift(track: dict, fit: dict | None = None, ax=None, unit: str = "hz",
               colour: str = "#0072B2", fit_colour: str = "#D55E00", corrected: dict | None = None):
    """Peak position against slice number, with the fitted curve.

    ``unit`` : ``"hz"`` or ``"ppm"`` for the vertical axis. ``corrected`` : the
    positions measured again after the correction, drawn as open circles.
    Returns the axes."""
    import matplotlib.pyplot as plt

    if ax is None:
        _fig, ax = plt.subplots(figsize=(5.5, 3.6))
    key = "hz" if unit == "hz" else "ppm"
    x = np.asarray(track["row"]) + 1
    y = np.asarray(track[key])
    ax.plot(x, y, "o", ms=4, color=colour, label="peak position")
    if fit is not None:
        scale = 1.0 if key == "hz" else 1.0 / max(abs(np.mean(track["hz"]) / np.mean(track["ppm"]))
                                                  if np.mean(track["ppm"]) else 1.0, 1e-12)
        ax.plot(np.asarray(fit["x"]) + 1, np.asarray(fit["fit"]) * (1.0 if key == "hz" else scale),
                "-", lw=1.6, color=fit_colour, label=MODEL_LABELS.get(fit["model"], fit["model"]))
    if corrected is not None:
        ax.plot(np.asarray(corrected["row"]) + 1, np.asarray(corrected[key]), "o", ms=4,
                mfc="none", mec="#009E73", label="after correction")
    ax.set_xlabel("Slice")
    ax.set_ylabel("Peak position (Hz)" if key == "hz" else "Peak position (ppm)")
    ax.legend(frameon=False)
    return ax


# ----------------------------------------------------------------------
# For a pipeline that rebuilds the spectra (the GUI's reprocessing)
# ----------------------------------------------------------------------

def record_shifts(spectrum) -> np.ndarray | None:
    """The per-slice shifts (Hz) recorded on ``spectrum``, if they fit it."""
    record = spectrum.metadata.raw.get(RECORD_KEY)
    if not record:
        return None
    shifts = np.asarray(record.get("shifts_hz") or [], dtype=float)
    if spectrum.ndim == 2 and shifts.size == spectrum.data.shape[0]:
        return shifts
    return None


def shift_rebuilt_rows(spectrum, complex_rows, ppm_axis, remember: bool = True):
    """Apply the recorded drift to ``complex_rows`` (the slices rebuilt from the
    raw FIDs, ``(n_rows, n_points)``) and, with ``remember``, keep them as the
    uncorrected originals :func:`remove_drift` restores. Returns the shifted
    rows, or ``complex_rows`` unchanged when nothing is recorded."""
    shifts = record_shifts(spectrum)
    rows = np.asarray(complex_rows)
    if shifts is None or rows.ndim != 2 or rows.shape[0] != shifts.size:
        return complex_rows
    if remember:
        spectrum.metadata.raw[ORIGINAL_KEY] = (np.real(rows).copy(), np.imag(rows).copy())
    return shift_rows(rows, shifts, hz_per_point(ppm_axis, spectrum.metadata.sfo1))
