"""
DNP polarisation against a delay: the difference between two pseudo-2D series.

Two experiments are recorded with the same delay list, typically one with the
microwaves on and one with them off, or two different polarisation times. Each
slice is measured (a region integral, or the height of a peak), the two series
are subtracted (or divided), and the result is plotted against the delay with
nothing fitted. The individual series can be drawn with it, and the points joined
by lines.

Measuring is shared with the rest of the DNP tools
(:func:`opennmr.notebook.dnp.region_integral`, :func:`~opennmr.notebook.dnp.peak_height`),
so an integral here is the same number as in Integrate ▸ DNP. Values already
fitted slice by slice (the output of ``NMR2D_fit_slices``, a CSA or quadrupolar
series fit, or ``NMR2D_integrate_slices``) can be used instead, for a fitted peak
height or area.

Notebook::

    from opennmr.notebook import dnp_polarisation as dp
    on = dp.measure_series(spectrum_on, delays, quantity="integral", region=(120, 140))
    off = dp.measure_series(spectrum_off, delays, quantity="integral", region=(120, 140))
    result = dp.polarisation(on, off, mode="difference")
    dp.plot_polarisation(result, show_individual=True, connect=True)
    dp.polarisation_to_csv(result, "polarisation.csv")

``delays`` is anything :func:`delay_values` accepts: a list, or
``start=…, stop=…, step=…``, or ``start=…, step=…, multiplier=…``. Delays are in
seconds throughout.
"""

from __future__ import annotations

import csv
import math

import numpy as np

from ..core.spectrum import NMRSpectrum
from ..processing.relaxation import start_step_values
from ..processing.slicing import get_trace
from . import intensity_scaling
from .dnp import peak_height, region_integral

QUANTITIES = ("integral", "height", "fit_area", "fit_height")
MODES = ("difference", "ratio", "enhancement", "sum")

_MODE_TEXT = {
    "difference": "A − B",
    "ratio": "A / B",
    "enhancement": "(A − B) / B",
    "sum": "A + B",
}
_QUANTITY_TEXT = {
    "integral": "integral",
    "height": "peak height",
    "fit_area": "fitted peak area",
    "fit_height": "fitted peak height",
}


# ============================================================
# Delays
# ============================================================

def delay_values(n: int, values=None, start: float | None = None, stop: float | None = None,
                 step: float | None = None, multiplier: float = 1.0, scale: float = 1.0) -> np.ndarray:
    """The ``n`` delays (seconds) for the slices of a series.

    Give one of

    * ``values``: a list, one per slice (longer is fine, the first ``n`` are
      used; shorter is an error);
    * ``start``, ``stop`` and ``step``: ``start, start + step, … ≤ stop``
      (``stop`` included when the step reaches it);
    * ``start`` and ``step`` (and ``multiplier``): one value per slice, each
      step ``multiplier`` times the last, as the Fit panel's *Start + step*;
    * ``start`` and ``stop`` alone: ``n`` equally spaced values from one to the other.

    ``scale`` converts what was typed to seconds (1e-3 for milliseconds).
    The count must equal ``n`` for start/stop/step, since a list that does
    not match the data is the one mistake that silently shifts every point."""
    n = int(n)
    if values is not None:
        arr = np.asarray(values, dtype=float).ravel() * scale
        if arr.size < n:
            raise ValueError(
                f"The delay list has {arr.size} values but the series has {n} slices; "
                "a shorter list cannot be used. Give one value per slice (a longer list "
                "is fine: the first values are used).")
        return arr[:n]
    if start is None:
        raise ValueError("Give a list of delays, or a start value.")
    if stop is not None and step is not None:
        if step == 0:
            raise ValueError("Step must be non-zero.")
        count = int(math.floor((stop - start) / step + 1e-9)) + 1
        if count < 1:
            raise ValueError("Stop is before start for this step.")
        arr = (start + step * np.arange(count)) * scale
        if arr.size != n:
            raise ValueError(
                f"start {start:g}, stop {stop:g}, step {step:g} gives {arr.size} values "
                f"but the series has {n} slices.")
        return arr
    if stop is not None:
        return np.linspace(start, stop, n) * scale
    if step is None:
        raise ValueError("Give a step, or a stop, as well as a start.")
    return start_step_values(start * scale, step * scale, multiplier, n)



def experiment_delays(spectrum: NMRSpectrum) -> tuple[np.ndarray | None, str | None]:
    """The delays (seconds) the experiment itself records for its slices, and
    where they came from: a JEOL/other reader's own list, or a Bruker ``vdlist``
    next to the data. ``(None, None)`` when there is none to read."""
    raw = spectrum.metadata.raw or {}
    n = spectrum.data.shape[0] if spectrum.ndim == 2 else None
    recorded = raw.get("indirect_values")
    if isinstance(recorded, dict) and recorded.get("unit") == "s":
        values = np.asarray(recorded.get("values"), dtype=float).ravel()
        if values.size and np.all(np.isfinite(values)) and (n is None or values.size >= n):
            return values[:n], recorded.get("source") or "the data file's own delays"
    path = spectrum.metadata.source_path
    if spectrum.metadata.vendor == "bruker" and path:
        try:
            from ..io import bruker as io_bruker
            listed = io_bruker.read_vdlist(path)
        except (FileNotFoundError, OSError, ValueError):
            return None, None
        values = np.asarray(listed["seconds"] if isinstance(listed, dict) else listed,
                            dtype=float).ravel()
        if values.size and (n is None or values.size >= n):
            return values[:n], "this experiment's vdlist"
    return None, None

# ============================================================
# Measuring a series
# ============================================================

def _row_spectra(spectrum: NMRSpectrum, rows=None) -> tuple[list[int], list[NMRSpectrum]]:
    if spectrum.ndim != 2:
        raise ValueError("A pseudo-2D series is needed: this spectrum is 1D.")
    n = spectrum.data.shape[0]
    chosen = list(range(1, n + 1)) if rows is None else [int(r) for r in rows]
    bad = [r for r in chosen if not (1 <= r <= n)]
    if bad:
        raise ValueError(f"Slice {bad[0]} does not exist: the series has {n} slices (1..{n}).")
    return chosen, [get_trace(spectrum, "f1", index=r) for r in chosen]


def measure_series(spectrum: NMRSpectrum, delays, quantity: str = "integral", region=None,
                   peak: float | None = None, baseline: str = "local", rows=None,
                   label: str | None = None, per_scan: bool = True, divide_rg: bool = True,
                   snap: float | None = None) -> dict:
    """Measure every slice of a pseudo-2D series.

    ``quantity``: ``"integral"`` (``region=(lo, hi)`` ppm) or ``"height"``
    (``peak=`` ppm: the largest signal within ``snap`` of it, height above a
    local baseline). ``delays``: one per slice in seconds (see
    :func:`delay_values`); with ``rows`` (1-based slice numbers) the delays
    are those of the chosen slices, in the same order. ``per_scan`` and
    ``divide_rg``: scale to per scan / per unit receiver gain where these are
    recorded, so two experiments with different NS or RG are comparable.

    Returns ``{"x", "y", "y_err", "rows", "label", "quantity", "region",
    "peak", "notes", "unit", "source"}`` in the order given (not re-sorted:
    a delay list need not be monotonic)."""
    if quantity not in ("integral", "height"):
        raise ValueError("quantity must be 'integral' or 'height' here; use series_from_fits "
                         "for a fitted area or height.")
    if quantity == "integral":
        if region is None:
            raise ValueError("An integral needs a region=(lo, hi) in ppm.")
        lo, hi = sorted(float(v) for v in region)
    elif peak is None:
        raise ValueError("A peak height needs a peak position in ppm.")
    chosen, traces = _row_spectra(spectrum, rows)
    delays = np.asarray(delays, dtype=float).ravel()
    if delays.size == spectrum.data.shape[0] and rows is not None:
        delays = delays[np.array(chosen) - 1]
    if delays.size != len(chosen):
        raise ValueError(f"{len(chosen)} slices but {delays.size} delays.")
    notes: list[str] = []
    factor = 1.0
    if per_scan:
        info = intensity_scaling.comparison_factors(spectrum, "per_scan", divide_rg)
        notes.extend(f"{n}." for n in info["notes"])
        factor = info["factor"]
    elif divide_rg:
        rg = intensity_scaling.receiver_gain(spectrum)
        if rg:
            factor = 1.0 / rg
        else:
            notes.append("Receiver gain not recorded (not divided).")
    y, e = [], []
    for r, trace in zip(chosen, traces):
        try:
            if quantity == "integral":
                m = region_integral(trace, lo, hi, baseline=baseline)
            else:
                m = peak_height(trace, float(peak), snap=snap, baseline=baseline)
        except ValueError as exc:
            raise ValueError(f"Slice {r}: {exc}") from None
        y.append(m["value"] * factor)
        e.append(abs(m["error"] * factor))
    unit = _QUANTITY_TEXT[quantity] + (" per scan" if per_scan else "") + (
        " per unit RG" if divide_rg else "") + " (a.u.)"
    return {"x": delays, "y": np.array(y, float), "y_err": np.array(e, float), "rows": chosen,
            "label": label or f"{_QUANTITY_TEXT[quantity]}", "quantity": quantity,
            "region": (lo, hi) if quantity == "integral" else None,
            "peak": None if quantity == "integral" else float(peak), "notes": notes,
            "unit": unit, "source": "measured"}


def series_from_fits(fits, delays, value_source: str = "area", peak_index: int = 0,
                     label: str | None = None) -> dict:
    """One peak's (or region's) values from slice-by-slice fits already made.

    ``fits``: ``fitted["rows"]`` from ``NMR2D_fit_slices`` (a peak fit, or a CSA /
    quadrupolar series fit) or from ``NMR2D_integrate_slices``. ``value_source``:
    ``"area"`` (the fitted area, or the integral of an integrated region),
    ``"amplitude"`` (the fitted height). ``delays``: indexed by each slice's own
    1-based number (``fit["index"]``), as for ``NMR2D_fit_relaxation``, or one
    per fit when the lengths match. ``peak_index``: which peak, 0-based."""
    if not fits:
        raise ValueError("No fitted slices given.")
    if value_source not in ("area", "amplitude"):
        raise ValueError("value_source must be 'area' or 'amplitude'.")
    delays = np.asarray(delays, dtype=float).ravel()
    indices = [int(f["index"]) for f in fits]
    if max(indices) <= delays.size:
        # The NMR2D_fit_relaxation convention: each fit looks up its own slice number.
        x = np.array([delays[i - 1] for i in indices], float)
    elif delays.size == len(fits):
        x = delays.copy()
    else:
        raise ValueError(f"The delay list has {delays.size} values but a fit covers slice "
                         f"{max(indices)}; give one delay per slice (or one per fit).")
    y, e = [], []
    model_fit = bool(fits[0].get("model_fit"))
    fitted = "popt" in fits[0] or model_fit
    for fit in fits:
        if value_source == "area" or not fitted:
            values = list(fit.get("integrals", []))
            errors = list(fit.get("integral_errors") or [None] * len(values))
        elif model_fit:
            values = list(fit["amplitudes"])
            errors = list(fit.get("amplitude_errors") or [None] * len(values))
        else:
            from ..processing.fitting import extract_params_and_errors
            per_peak = extract_params_and_errors(
                fit["popt"], fit["pcov"], fit["peak_defs"], fit["is_free"])
            k = [p["names"].index("amplitude") for p in per_peak]
            values = [p["params"][i] for p, i in zip(per_peak, k)]
            errors = [p["errors"][i] for p, i in zip(per_peak, k)]
        if peak_index >= len(values):
            y.append(np.nan)
            e.append(np.nan)
        else:
            y.append(float(values[peak_index]))
            err = errors[peak_index] if peak_index < len(errors) else None
            e.append(float(err) if err is not None and np.isfinite(err) else 0.0)
    if value_source == "amplitude" and not fitted:
        raise ValueError("These slices were integrated, not fitted: they have no peak height. "
                         "Use value_source='area', or fit the peaks first.")
    quantity = "fit_height" if value_source == "amplitude" else "fit_area"
    return {"x": x, "y": np.array(y, float), "y_err": np.array(e, float),
            "rows": indices, "label": label or f"peak {peak_index + 1}", "quantity": quantity,
            "region": None, "peak": None, "notes": [],
            "unit": _QUANTITY_TEXT[quantity] + " (a.u.)", "source": "fitted"}


# ============================================================
# Combining two series
# ============================================================

def _matched(a: dict, b: dict, interpolate: bool, rel_tol: float = 1e-6):
    """B's values at A's delays: by value where the lists agree, otherwise
    only when ``interpolate`` is on."""
    xa, xb = np.asarray(a["x"], float), np.asarray(b["x"], float)
    scale = max(float(np.max(np.abs(xa))), float(np.max(np.abs(xb))), 1e-300)
    if xa.size == xb.size and np.allclose(xa, xb, rtol=rel_tol, atol=rel_tol * scale):
        return np.asarray(b["y"], float), np.asarray(b["y_err"], float), []
    if not interpolate:
        raise ValueError(
            f"The two series do not share their delays ({xa.size} values from "
            f"{xa.min():g} to {xa.max():g} s against {xb.size} from {xb.min():g} to "
            f"{xb.max():g} s). Subtracting needs a value of B at each delay of A: use the "
            "same list for both, or switch on interpolation.")
    order = np.argsort(xb)
    yb = np.interp(xa, xb[order], np.asarray(b["y"], float)[order])
    eb = np.interp(xa, xb[order], np.asarray(b["y_err"], float)[order])
    outside = int(np.count_nonzero((xa < xb.min()) | (xa > xb.max())))
    notes = ["B was interpolated linearly onto the delays of A."]
    if outside:
        notes.append(f"{outside} delay(s) of A lie outside B's range: B is held at its end value there.")
    return yb, eb, notes


def polarisation(a: dict, b: dict, mode: str = "difference", interpolate: bool = False,
                 sort: bool = True) -> dict:
    """Combine two measured series, delay by delay.

    ``mode``: ``"difference"`` (A − B), ``"ratio"`` (A / B), ``"enhancement"``
    ((A − B) / B) or ``"sum"``. Errors are propagated from the two series'
    own (independent). ``interpolate``: put B onto A's delays when the lists
    differ. Returns ``{"x", "y", "y_err", "a", "b", "mode", "label", "unit",
    "notes"}`` with ``a`` and ``b`` the two series (B at A's delays)."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, not {mode!r}")
    yb, eb, notes = _matched(a, b, interpolate)
    ya, ea = np.asarray(a["y"], float), np.asarray(a["y_err"], float)
    x = np.asarray(a["x"], float)
    with np.errstate(divide="ignore", invalid="ignore"):
        if mode == "difference":
            y, e = ya - yb, np.hypot(ea, eb)
        elif mode == "sum":
            y, e = ya + yb, np.hypot(ea, eb)
        elif mode == "ratio":
            y = ya / yb
            e = np.abs(y) * np.sqrt(np.where(ya != 0, (ea / ya) ** 2, 0.0) + (eb / yb) ** 2)
        else:
            y = (ya - yb) / yb
            num_e = np.hypot(ea, eb)
            num = ya - yb
            e = np.abs(y) * np.sqrt(np.where(num != 0, (num_e / num) ** 2, 0.0) + (eb / yb) ** 2)
    if mode in ("ratio", "enhancement") and np.any(yb == 0):
        notes.append("B is zero at some delays: those points are undefined (shown as gaps).")
    b_at_a = dict(b, x=x.copy(), y=yb, y_err=eb)
    if a.get("quantity") != b.get("quantity"):
        notes.append(f"A is a {_QUANTITY_TEXT.get(a.get('quantity'), 'value')} but B is a "
                     f"{_QUANTITY_TEXT.get(b.get('quantity'), 'value')}: check this is intended.")
    if sort:
        order = np.argsort(x, kind="stable")
        x, y, e = x[order], y[order], e[order]
        a = dict(a, x=np.asarray(a["x"], float)[order], y=ya[order], y_err=ea[order])
        b_at_a = dict(b_at_a, x=b_at_a["x"][order], y=yb[order], y_err=eb[order])
    unit = ("(A / B)" if mode == "ratio" else "(A − B) / B" if mode == "enhancement"
            else a.get("unit", "a.u."))
    return {"x": x, "y": y, "y_err": e, "a": a, "b": b_at_a, "mode": mode,
            "label": _MODE_TEXT[mode], "unit": unit, "notes": notes}


# ============================================================
# Output
# ============================================================

def _time_unit(x, unit: str = "auto") -> tuple[float, str]:
    """(divisor, label) for seconds shown as ``unit``."""
    table = {"s": (1.0, "s"), "ms": (1e-3, "ms"), "us": (1e-6, "µs"), "µs": (1e-6, "µs")}
    if unit != "auto":
        if unit not in table:
            raise ValueError("time unit must be 'auto', 's', 'ms' or 'µs'.")
        return table[unit]
    finite = np.abs(np.asarray(x, float)[np.isfinite(x)])
    top = float(finite.max()) if finite.size else 1.0
    return table["us"] if top < 1e-3 else table["ms"] if top < 1.0 else table["s"]


def polarisation_to_csv(result: dict, path=None) -> str:
    """CSV text of the result, with both individual series alongside (and
    written to ``path`` when given). Delays in seconds."""
    lines = [
        f"# openNMR DNP polarisation: {result['label']}",
        f"# A: {result['a']['label']} ({result['a']['unit']})",
        f"# B: {result['b']['label']} ({result['b']['unit']})",
    ]
    lines += [f"# note: {n}" for n in result["notes"]]
    import io
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["delay_s", "result", "result_error", "A", "A_error", "B", "B_error"])
    for i in range(len(result["x"])):
        w.writerow([repr(float(result["x"][i])), repr(float(result["y"][i])),
                    repr(float(result["y_err"][i])),
                    repr(float(result["a"]["y"][i])), repr(float(result["a"]["y_err"][i])),
                    repr(float(result["b"]["y"][i])), repr(float(result["b"]["y_err"][i]))])
    text = "\n".join(lines) + "\n" + buf.getvalue()
    if path is not None:
        with open(path, "w", encoding="utf-8-sig", newline="") as fh:
            fh.write(text)
    return text


def polarisation_report(result: dict) -> str:
    """A short plain-language account of what was plotted."""
    a, b = result["a"], result["b"]
    x = result["x"]
    out = [f"{result['label']}  ({_MODE_TEXT[result['mode']]})",
           f"A: {a['label']} [{a['source']}] – {_QUANTITY_TEXT.get(a['quantity'], '')}"
           + (f", region {a['region'][0]:g} to {a['region'][1]:g} ppm" if a.get("region") else "")
           + (f", peak at {a['peak']:g} ppm" if a.get("peak") is not None else ""),
           f"B: {b['label']} [{b['source']}] – {_QUANTITY_TEXT.get(b['quantity'], '')}"
           + (f", region {b['region'][0]:g} to {b['region'][1]:g} ppm" if b.get("region") else "")
           + (f", peak at {b['peak']:g} ppm" if b.get("peak") is not None else ""),
           f"{len(x)} delays, {x.min():g} to {x.max():g} s."]
    finite = np.isfinite(result["y"])
    if finite.any():
        k = int(np.nanargmax(np.abs(np.where(finite, result["y"], np.nan))))
        out.append(f"Largest |{result['label']}| = {result['y'][k]:.4g} ± {result['y_err'][k]:.2g} "
                   f"at {x[k]:g} s.")
    out += [f"Note: {n}" for n in result["notes"]]
    out += [f"A: {n}" for n in a["notes"]] + [f"B: {n}" for n in b["notes"]]
    return "\n".join(out)


def plot_polarisation(result: dict, show_individual: bool = False, connect: bool = True,
                      time_unit: str = "auto", figure=None, ax=None, fontsize: float = 12.0,
                      colour: str = "black", colour_a: str = "tab:blue", colour_b: str = "tab:red",
                      marker: str = "o", marker_a: str = "s", marker_b: str = "^",
                      markersize: float = 5.0, linewidth: float = 1.0, error_bars: bool = True,
                      zero_line: bool = True, log_x: bool = False,
                      xlabel: str | None = None, ylabel: str | None = None,
                      show_top_spine: bool = False, show_right_spine: bool = False):
    """The result against the delay. ``show_individual`` also draws A and B
    (on a second axis above, sharing the delay axis, when the two differ
    greatly from the result; always above, in one figure). ``connect`` joins the
    points with lines. Returns the figure."""
    import matplotlib.pyplot as plt
    from ..plotting.style import apply_spine_style, apply_tick_style

    divisor, unit_label = _time_unit(result["x"], time_unit)
    x = np.asarray(result["x"], float) / divisor
    if ax is None:
        if figure is None:
            figure = plt.figure(figsize=(3.375 * 1.4, 3.0 * (2 if show_individual else 1)))
        else:
            figure.clear()
        if show_individual:
            top, bottom = figure.subplots(2, 1, sharex=True)
            axes = [(top, "individual"), (bottom, "result")]
        else:
            axes = [(figure.subplots(1, 1), "result")]
    else:
        figure = ax.figure
        axes = [(ax, "result")]

    def draw(axis, xs, ys, es, colour_, marker_, label):
        style = "-" if connect else "none"
        axis.plot(xs, ys, linestyle=style, color=colour_, linewidth=linewidth, zorder=1)
        if error_bars:
            axis.errorbar(xs, ys, yerr=es, fmt=marker_, color=colour_, markersize=markersize,
                          capsize=2.5, linewidth=linewidth, zorder=2, label=label)
        else:
            axis.plot(xs, ys, marker_, color=colour_, markersize=markersize, zorder=2, label=label)

    for axis, kind in axes:
        if kind == "individual":
            draw(axis, x, result["a"]["y"], result["a"]["y_err"], colour_a, marker_a,
                 result["a"]["label"] + " (A)")
            draw(axis, x, result["b"]["y"], result["b"]["y_err"], colour_b, marker_b,
                 result["b"]["label"] + " (B)")
            axis.set_ylabel(ylabel_for_individual(result), fontsize=fontsize)
            axis.legend(frameon=False, fontsize=fontsize * 0.75)
        else:
            draw(axis, x, result["y"], result["y_err"], colour, marker, result["label"])
            if zero_line and np.nanmin(result["y"]) < 0 < np.nanmax(result["y"]):
                axis.axhline(0, color="0.5", linewidth=0.8, zorder=0)
            axis.set_ylabel(ylabel if ylabel is not None else _result_label(result),
                            fontsize=fontsize)
        if log_x:
            axis.set_xscale("log")
        apply_spine_style(axis, top=show_top_spine, right=show_right_spine, bottom=True, left=True)
        apply_tick_style(axis, labelsize=fontsize * 0.9)
        axis.tick_params(which="both", top=show_top_spine, right=show_right_spine)
    axes[-1][0].set_xlabel(xlabel if xlabel is not None else f"delay ({unit_label})",
                           fontsize=fontsize)
    figure.tight_layout()
    return figure


def ylabel_for_individual(result: dict) -> str:
    return result["a"]["unit"]


def _result_label(result: dict) -> str:
    if result["mode"] in ("ratio", "enhancement"):
        return f"{result['label']}"
    return f"{result['label']}  {result['a']['unit']}"
