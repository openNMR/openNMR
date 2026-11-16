"""
Copying a fit from one experiment to another (explicit user request,
2026-11-16): a CSA / quadrupolar / solid-state fit made on an extracted
slice of one experiment, used on a slice of another; and a T1 (or T2, or
any) curve fit of one profile, used on another profile.

Nothing here is the fit itself. Copying a fit means *starting from it*:

* **Slice (lineshape) fits.** The fitted site parameters (shift, CSA,
  C_Q, widths, relative intensities...) of a panel fit become the starting
  values of the same panel for the other spectrum. The baseline and the
  noise differ between experiments, so the values are refitted rather than
  imposed. (To apply one slice's *fixed* shapes to every slice of a series
  -- no refit of the shape -- use :mod:`opennmr.notebook.series_fitting`.)
* **Profile (curve) fits.** The fitted model and its parameter values
  become the starting guesses (with the fixed flags and bounds the source
  used) of the same model on the other profile.

Notebook::

    from opennmr.notebook import fit_transfer as ft

    guesses = ft.curve_fit_start(source_results)          # one dict per peak
    new = ft.refit_curve_like(source_results, target_slice_results, vdlist,
                              value_source="area")
    print(ft.compare_curve_fits(source_results, new))

The GUI actions (right-click a fit in Open experiments ▸ Copy Fit To…) are
built on the same functions.
"""

from __future__ import annotations

import copy
import math

import numpy as np

__all__ = ["curve_fit_start", "guess_rows", "widen_bounds", "refit_curve_like",
           "compare_curve_fits", "slice_fit_start", "describe_transfer"]


def _finite(value) -> bool:
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def curve_fit_start(results, context=None) -> list[dict]:
    """One start per peak of a curve fit: ``{"model", "peak_label",
    "p0", "bounds", "fixed", "param_names"}``.

    ``results`` : the curve fit's per-peak results (``relaxation_fit_results``).
    ``p0`` is ``{name: fitted value}``; ``bounds`` is ``{name: (lo, hi)}``
    and ``fixed`` ``{name: True}`` for each held fixed, from the fit's own record (and, where
    given, ``context``'s per-peak entries -- the settings the fit was run
    with, which say what was held fixed). Non-finite fitted values are
    left out (the target's automatic guess is used for them)."""
    per_peak = (context or {}).get("per_peak") or []
    starts = []
    for k, result in enumerate(results or []):
        if not isinstance(result, dict):
            continue
        names = list(result.get("param_names") or (result.get("params") or {}))
        params = result.get("params") or {}
        p0 = {n: float(params[n]) for n in names if _finite(params.get(n))}
        lo_hi = result.get("bounds")
        bounds = {}
        if lo_hi is not None and len(lo_hi) == 2 and len(lo_hi[0]) == len(names):
            for n, lo, hi in zip(names, lo_hi[0], lo_hi[1]):
                bounds[n] = (float(lo), float(hi))
        # A fit's ``fixed`` is ``{name: the value it was held at}``: a name
        # being there is what makes it fixed (its value may well be 0).
        fixed = {n: True for n in (result.get("fixed") or {})}
        if k < len(per_peak):
            _p0, _bounds, ctx_fixed = (list(per_peak[k]) + [None, None, None])[:3]
            if isinstance(ctx_fixed, dict):
                fixed.update({n: True for n in ctx_fixed})
        starts.append({"model": result.get("model"), "peak_label": result.get("peak_label"),
                       "param_names": names, "p0": p0, "bounds": bounds, "fixed": fixed})
    return starts


def widen_bounds(guess: float, low: float, high: float) -> tuple[float, float]:
    """``(low, high)`` widened, if need be, to contain ``guess``: the other
    experiment's automatic bounds were worked out from *its* data and may
    sit either side of the copied value. Widened by a decade (to a tenth /
    ten times the guess) -- never narrowed."""
    low, high = float(low), float(high)
    guess = float(guess)
    if guess < low:
        low = guess / 10.0 if guess > 0 else (guess * 10.0 if guess < 0 else -1.0)
    if guess > high:
        high = guess * 10.0 if guess > 0 else (guess / 10.0 if guess < 0 else 1.0)
    return low, high


def guess_rows(start: dict, param_names, *, scale=None, auto=None,
               fmt=lambda v: f"{v:.6g}") -> list[list[str]]:
    """The GUI guess table's rows for ``start`` (one peak of
    :func:`curve_fit_start`): ``[[name, guess, min, max, fixed], ...]``,
    strings, in ``param_names`` order.

    ``scale(name)`` : the display unit's factor (the table shows ms where
    the model works in s). ``auto`` : ``{name: (lo, hi)}``, the target's
    own automatic bounds (seconds), used where the source recorded none --
    widened to hold the copied value."""
    scale = scale or (lambda _name: 1.0)
    rows = []
    for name in param_names:
        s = float(scale(name)) or 1.0
        guess = start["p0"].get(name)
        if guess is None:
            continue                                   # the table keeps its automatic value
        lo_hi = start["bounds"].get(name) or (auto or {}).get(name)
        lo_cell = hi_cell = ""
        if lo_hi is not None and all(np.isfinite(lo_hi)):
            lo, hi = widen_bounds(guess, *lo_hi)
            lo_cell, hi_cell = fmt(lo / s), fmt(hi / s)
        elif lo_hi is not None:
            lo, hi = lo_hi
            lo_cell = fmt(lo / s) if np.isfinite(lo) else "-inf"
            hi_cell = fmt(hi / s) if np.isfinite(hi) else "inf"
        rows.append([name, fmt(guess / s), lo_cell, hi_cell,
                     "1" if start["fixed"].get(name) else "0"])
    return rows


def _by_peak(values, n_peaks: int, source: list[dict], key: str):
    """``{peak: value}`` for the target's ``n_peaks``, each taking the
    source's own peak (by position); extra target peaks take the first."""
    if not source:
        return None
    out = {}
    for p in range(n_peaks):
        out[p] = source[p][key] if p < len(source) else source[0][key]
    return out


def refit_curve_like(source_results, target_slice_results, vdlist, *, value_source="area",
                     context=None, custom_model=None, use_errors=True, keep_fixed=True,
                     show_report=False):
    """Fit ``target_slice_results`` (per-slice measurements of another
    profile) with the model of ``source_results``, starting from its fitted
    values. Returns the new per-peak results.

    ``vdlist`` : the target's delays (seconds). ``context`` : the source's
    ``relaxation_fit_context``, for what was held fixed (and, for a custom
    model, the model itself; or give ``custom_model``). ``keep_fixed=False``
    fits everything the source held fixed too. Starting values outside the
    model's own bounds are widened, as in :func:`guess_rows`.
    """
    from ..processing import relaxation as pr

    starts = curve_fit_start(source_results, context)
    if not starts:
        raise ValueError("The source has no curve fit to copy.")
    model = starts[0]["model"]
    custom = custom_model or (context or {}).get("custom_model")
    if model in ("custom", "diffusion_n") and custom is None:
        raise ValueError("That fit used a custom model; give it as custom_model.")
    spec = custom if model in ("custom", "diffusion_n") else pr.RELAXATION_MODELS[model]
    names = list(spec["params"])
    n_peaks = len(target_slice_results[0].get("integrals", [])) if target_slice_results else 0
    first = target_slice_results[0] if target_slice_results else {}
    if "peak_defs" in first:
        n_peaks = len(first["peak_defs"])
    elif first.get("model_fit"):
        n_peaks = len(first.get("site_labels") or first.get("integrals", []))
    n_peaks = max(n_peaks, 1)

    p0_map, bounds_map, fixed_map = {}, {}, {}
    for p in range(n_peaks):
        start = starts[p] if p < len(starts) else starts[0]
        p0_map[p] = [start["p0"].get(n, float("nan")) for n in names]
        bounds_map[p] = _bounds_for(start, names, spec, target_slice_results, vdlist, p0_map[p])
        fixed_map[p] = ({n: start["p0"][n] for n in names
                         if start["fixed"].get(n) and n in start["p0"]}
                        if keep_fixed else None) or None
        # a NaN (a parameter the source never fitted) falls back to automatic
        if any(not np.isfinite(v) for v in p0_map[p]):
            p0_map[p] = [v if np.isfinite(v) else _auto_guess(spec, i, target_slice_results, vdlist)
                         for i, v in enumerate(p0_map[p])]
    if not any(fixed_map.values()):
        fixed_map = None
    return pr.NMR2D_fit_relaxation(
        target_slice_results, model=model, vdlist=vdlist, show_report=show_report,
        value_source=value_source, custom_model=custom if model in ("custom", "diffusion_n")
        else None, use_errors=use_errors, p0=p0_map, bounds=bounds_map, fixed=fixed_map)


def _auto_guess(spec, index, slice_results, vdlist):
    try:
        return float(spec["p0"](np.asarray(vdlist, dtype=float), np.ones(len(vdlist)))[index])
    except Exception:  # noqa: BLE001 -- a one-off fallback
        return 1.0


def _bounds_for(start, names, spec, slice_results, vdlist, p0):
    """``(lo, hi)`` lists in ``names`` order: the source's bounds, widened to
    hold the copied values (the target's own model bounds otherwise)."""
    try:
        lo_auto, hi_auto = spec["bounds"](np.asarray(vdlist, dtype=float),
                                          np.ones(len(vdlist)))
    except Exception:  # noqa: BLE001 -- unbounded
        lo_auto, hi_auto = [-np.inf] * len(names), [np.inf] * len(names)
    lo, hi = [], []
    for i, name in enumerate(names):
        a, b = start["bounds"].get(name, (lo_auto[i], hi_auto[i]))
        if np.isfinite(p0[i]) and np.isfinite(a) and np.isfinite(b):
            a, b = widen_bounds(p0[i], a, b)
        lo.append(a)
        hi.append(b)
    return lo, hi


def compare_curve_fits(source_results, new_results) -> list[dict]:
    """Peak by peak: each parameter of the source and of the copy, side by
    side -- ``[{"peak", "parameter", "source", "copy", "copy_error"}]``."""
    rows = []
    for k, new in enumerate(new_results or []):
        old = source_results[k] if k < len(source_results) else (source_results or [{}])[0]
        for name in new.get("param_names") or new.get("params", {}):
            rows.append({"peak": new.get("peak_label") or f"Peak {k + 1}", "parameter": name,
                         "source": (old.get("params") or {}).get(name),
                         "copy": (new.get("params") or {}).get(name),
                         "copy_error": (new.get("errors") or {}).get(name)})
    return rows


def slice_fit_start(panel_inputs: dict, fitted_sites: list[dict] | None = None) -> dict:
    """A panel's saved inputs (``get_state()``), as the starting point for
    another spectrum: the same inputs, with ``fitted_sites`` (the fit's
    ``site_dicts``, fitted values) in place of the starting sites when
    given. A copy -- the source is not changed."""
    state = copy.deepcopy(panel_inputs or {})
    if fitted_sites:
        state["sites"] = copy.deepcopy(fitted_sites)
    return state


def describe_transfer(kind: str, source: str, target: str, mode: str = "start") -> str:
    """One line for the processing log: what was copied, from where to
    where, and how it was used."""
    how = {"start": "used as the starting values (not yet refitted)",
           "refit": "used as the starting values and refitted"}.get(mode, mode)
    return f"{kind} copied from {source} to {target}: {how}."
