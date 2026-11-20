"""Lineshape (peak) fits for PFG series: one shared shape set for every row.

``fit_series_with_lineshapes`` takes its shapes from a single reference slice, but in a
PFG series no one slice works: a weak membrane peak is hard to see in row 1 and the bulk peak
has gone by the last rows. :func:`fit_series_shared_shapes` instead fits **all rows at once**
(improvement notes §5):

* peak positions, widths and pseudo-Voigt mixing ``eta`` are **shared** by every row;
* each row's amplitudes (**non-negative**, NNLS) and a linear baseline are solved *linearly*
  for every trial shape -- variable projection -- so only 3 numbers per peak are searched;
* a peak may be held **downfield of another** (``anchors={membrane: bulk}``), which stops
  two pseudo-Voigts collapsing onto each other with opposite-sign amplitudes;
* :func:`flag_crosstalk` warns when one peak's decay contains a component that belongs to
  another peak (a susceptibility-broadened bulk line is not a pseudo-Voigt, so some bulk
  signal ends up under the membrane peak).

Areas and their errors are returned per row, ready for
:func:`opennmr.processing.relaxation.fit_relaxation` with ``build_diffusion_model``.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import least_squares, nnls

__all__ = ["pseudo_voigt", "fit_series_shared_shapes", "flag_crosstalk"]

_LN2 = math.log(2.0)


def pseudo_voigt(x, position: float, fwhm: float, eta: float = 0.5):
    """Unit-height pseudo-Voigt: ``eta`` Lorentzian + ``1 - eta`` Gaussian, FWHM ``fwhm``."""
    u = (np.asarray(x, dtype=float) - position) / fwhm
    return eta / (1.0 + 4.0 * u * u) + (1.0 - eta) * np.exp(-4.0 * _LN2 * u * u)


def _area_per_height(fwhm: float, eta: float) -> float:
    """Area of a unit-height pseudo-Voigt: ``w (eta pi/2 + (1-eta) sqrt(pi / (4 ln 2)))``."""
    return fwhm * (eta * math.pi / 2.0 + (1.0 - eta) * math.sqrt(math.pi / (4.0 * _LN2)))


def _baseline_columns(x, baseline: str):
    if baseline == "none":
        return []
    cols = [np.ones_like(x)]
    if baseline == "linear":
        cols.append(x - float(np.mean(x)))
    return cols


def _solve_rows(design_peaks, base_cols, data, non_negative: bool):
    """Per-row linear solve. Returns amplitudes (m, P), baseline coefficients (m, B)."""
    n_peaks = design_peaks.shape[1]
    base = np.column_stack(base_cols) if base_cols else np.zeros((design_peaks.shape[0], 0))
    n_base = base.shape[1]
    amps = np.zeros((data.shape[0], n_peaks))
    bases = np.zeros((data.shape[0], n_base))
    if non_negative:
        aug = np.column_stack([design_peaks, base, -base]) if n_base else design_peaks
        for r, y in enumerate(data):
            sol, _ = nnls(aug, y)
            amps[r] = sol[:n_peaks]
            if n_base:
                bases[r] = sol[n_peaks:n_peaks + n_base] - sol[n_peaks + n_base:]
    else:
        full = np.column_stack([design_peaks, base]) if n_base else design_peaks
        sol, *_ = np.linalg.lstsq(full, data.T, rcond=None)
        amps = sol[:n_peaks].T
        bases = sol[n_peaks:].T
    return amps, bases


def fit_series_shared_shapes(ppm, data, peaks, window=None, non_negative: bool = True,
                             baseline: str = "linear", anchors=None, fit_eta: bool = True,
                             max_nfev: int = 300) -> dict:
    """Fit every row of ``data`` (rows x points) with the same peak shapes.

    Parameters
    ----------
    ppm, data : the axis (either direction) and the real series, one row per gradient level.
    peaks : list of ``{"position", "fwhm", "eta" (default 0.5), "label" (optional),
        "position_bounds" (optional (lo, hi) ppm), "fwhm_bounds" (optional)}``.
    window : ``(lo, hi)`` ppm to fit within (default the whole axis).
    non_negative : amplitudes are >= 0 (NNLS; the baseline stays free in sign).
    baseline : ``"linear"`` (offset + slope per row), ``"constant"`` or ``"none"``.
    anchors : ``{i: j}`` -- peak ``i`` must lie at or **downfield** (higher ppm) of peak ``j``
        (it is searched as ``position_j + d`` with ``d >= 0``). ``j`` must not itself be anchored.
    fit_eta : search the pseudo-Voigt mixing too (otherwise held at each peak's ``eta``).

    Returns a dict with ``positions``, ``fwhm``, ``eta``, ``labels`` (shared); ``amplitudes``,
    ``amplitude_errors``, ``areas``, ``area_errors``, ``baseline_coefficients`` (per row);
    ``x`` / ``fit`` / ``data`` over the window; ``rms``, ``r_squared`` (per row), ``success``,
    ``log``.
    """
    if baseline not in ("linear", "constant", "none"):
        raise ValueError("baseline must be 'linear', 'constant' or 'none'.")
    x = np.asarray(ppm, dtype=float)
    Y = np.atleast_2d(np.asarray(data, dtype=float))
    if Y.shape[1] != x.size:
        raise ValueError("data must have one column per ppm point.")
    order = np.argsort(x)
    x, Y = x[order], Y[:, order]
    if window is not None:
        keep = (x >= min(window)) & (x <= max(window))
        x, Y = x[keep], Y[:, keep]
    if x.size < 8:
        raise ValueError("The fit window holds too few points.")
    peaks = [dict(p) for p in peaks]
    if not peaks:
        raise ValueError("Give at least one peak.")
    n_peaks = len(peaks)
    anchors = {int(i): int(j) for i, j in (anchors or {}).items()}
    for i, j in anchors.items():
        if not (0 <= i < n_peaks and 0 <= j < n_peaks) or i == j or j in anchors:
            raise ValueError("anchors must map one peak to a different, un-anchored peak.")
    labels = [p.get("label") or f"Peak {k + 1}" for k, p in enumerate(peaks)]

    # parameter vector: per peak [position (or offset from its anchor), fwhm, (eta)]
    p0, lo, hi = [], [], []
    for k, p in enumerate(peaks):
        pos, w = float(p["position"]), float(p["fwhm"])
        if w <= 0:
            raise ValueError(f"Peak {k + 1}: fwhm must be positive.")
        if k in anchors:
            ref = float(peaks[anchors[k]]["position"])
            span = max(4.0 * w, 3.0 * abs(pos - ref))
            p0.append(min(max(pos - ref, 0.0), span))
            lo.append(0.0)
            hi.append(span)
        else:
            blo, bhi = p.get("position_bounds") or (pos - 2.0 * w, pos + 2.0 * w)
            p0.append(min(max(pos, blo), bhi))
            lo.append(float(blo))
            hi.append(float(bhi))
        flo, fhi = p.get("fwhm_bounds") or (0.2 * w, 5.0 * w)
        p0.append(min(max(w, flo), fhi))
        lo.append(float(flo))
        hi.append(float(fhi))
        if fit_eta:
            p0.append(float(p.get("eta", 0.5)))
            lo.append(0.0)
            hi.append(1.0)
    stride = 3 if fit_eta else 2

    def unpack(theta):
        pos, wid, eta = [], [], []
        for k in range(n_peaks):
            base = k * stride
            pos.append(float(theta[base]))
            wid.append(float(theta[base + 1]))
            eta.append(float(theta[base + 2]) if fit_eta else float(peaks[k].get("eta", 0.5)))
        for i, j in anchors.items():
            pos[i] = pos[j] + pos[i]
        return pos, wid, eta

    base_cols = _baseline_columns(x, baseline)

    def design(theta):
        pos, wid, eta = unpack(theta)
        return np.column_stack([pseudo_voigt(x, pos[k], wid[k], eta[k]) for k in range(n_peaks)])

    def residual(theta):
        d = design(theta)
        amps, bases = _solve_rows(d, base_cols, Y, non_negative)
        fit = amps @ d.T
        if bases.shape[1]:
            fit = fit + bases @ np.column_stack(base_cols).T
        return (Y - fit).ravel()

    sol = least_squares(residual, p0, bounds=(lo, hi), x_scale="jac", max_nfev=max_nfev)
    pos, wid, eta = unpack(sol.x)
    d = design(sol.x)
    amps, bases = _solve_rows(d, base_cols, Y, non_negative)
    base_mat = np.column_stack(base_cols) if base_cols else np.zeros((x.size, 0))
    fit = amps @ d.T + (bases @ base_mat.T if base_mat.shape[1] else 0.0)
    resid = Y - fit

    # errors: unconstrained least-squares covariance over the columns that are active
    amp_err = np.zeros_like(amps)
    for r in range(Y.shape[0]):
        active = [k for k in range(n_peaks) if amps[r, k] > 0] or list(range(n_peaks))
        cols = np.column_stack([d[:, active], base_mat]) if base_mat.shape[1] else d[:, active]
        dof = max(x.size - cols.shape[1], 1)
        s2 = float(resid[r] @ resid[r]) / dof
        cov = s2 * np.linalg.pinv(cols.T @ cols)
        err = np.sqrt(np.clip(np.diag(cov), 0.0, None))
        # a peak held at zero by the constraint gets the error it would have had if free
        full = np.column_stack([d, base_mat]) if base_mat.shape[1] else d
        cov_full = s2 * np.linalg.pinv(full.T @ full)
        err_full = np.sqrt(np.clip(np.diag(cov_full), 0.0, None))
        for k in range(n_peaks):
            amp_err[r, k] = err[active.index(k)] if k in active else err_full[k]

    per_height = np.array([_area_per_height(wid[k], eta[k]) for k in range(n_peaks)])
    ss_tot = np.sum((Y - Y.mean(axis=1, keepdims=True)) ** 2, axis=1)
    r2 = 1.0 - np.sum(resid ** 2, axis=1) / np.where(ss_tot > 0, ss_tot, np.nan)
    log = (f"Shared-shape fit of {Y.shape[0]} rows, {n_peaks} peak(s): "
           + "; ".join(f"{labels[k]} at {pos[k]:.3f} ppm, FWHM {wid[k]:.3f}, η {eta[k]:.2f}"
                       for k in range(n_peaks))
           + f". Amplitudes {'non-negative (NNLS)' if non_negative else 'unconstrained'}, "
             f"{baseline} baseline per row; rms residual {float(np.sqrt(np.mean(resid ** 2))):.3g}.")
    if anchors:
        log += " " + "; ".join(f"{labels[i]} held at or downfield of {labels[j]}"
                               for i, j in anchors.items()) + "."
    return {
        "positions": np.array(pos), "fwhm": np.array(wid), "eta": np.array(eta), "labels": labels,
        "amplitudes": amps, "amplitude_errors": amp_err,
        "areas": amps * per_height, "area_errors": amp_err * per_height,
        "baseline_coefficients": bases, "x": x, "data": Y, "fit": fit, "residual": resid,
        "rms": float(np.sqrt(np.mean(resid ** 2))), "r_squared": r2,
        "success": bool(sol.success), "message": sol.message, "log": log,
    }


def flag_crosstalk(result: dict, b, tol: float = 0.3, min_fraction: float = 0.05) -> list[dict]:
    """Peaks whose decay contains a component that belongs to *another* peak.

    Each peak's area-vs-b curve is run through
    :func:`~opennmr.notebook.pfg_selection.select_diffusion_model` (up to two components, with
    and without an offset). Peak ``i`` is flagged when a non-dominant component carrying at
    least ``min_fraction`` of its signal has a D within ``tol`` (natural-log units, ~35 % for
    0.3) of another peak's dominant D. Returns one dict per flagged pair:
    ``{"peak", "other", "D_component", "D_other", "fraction"}``."""
    from .pfg_selection import select_diffusion_model

    b = np.asarray(b, dtype=float)
    areas, errs = result["areas"], result["area_errors"]
    info = []
    for k in range(areas.shape[1]):
        y = areas[:, k]
        sig = errs[:, k] if np.all(errs[:, k] > 0) else None
        sel = select_diffusion_model(b, y, sig, n_max=2, offsets=(False, True))
        fit = sel["fit"]
        if fit is None:
            info.append({"D": [], "amp": []})
            continue
        names = fit["param_names"]
        if "D" in names:
            ds, amps = [fit["params"]["D"]], [fit["params"]["I0"]]
        else:
            ds = [fit["params"][nm] for nm in names if nm.startswith("D")]
            amps = [fit["params"][nm] for nm in names if nm.startswith("I")]
        info.append({"D": ds, "amp": amps})
    flags = []
    dominant = [d["D"][int(np.argmax(d["amp"]))] if d["D"] else None for d in info]
    for i, d in enumerate(info):
        if len(d["D"]) < 2:
            continue
        total = sum(d["amp"])
        main = int(np.argmax(d["amp"]))
        for c, (Dc, a) in enumerate(zip(d["D"], d["amp"])):
            if c == main or total <= 0 or a / total < min_fraction:
                continue
            for j, Dj in enumerate(dominant):
                if j != i and Dj and abs(math.log(Dc / Dj)) < tol:
                    flags.append({"peak": result["labels"][i], "other": result["labels"][j],
                                  "D_component": Dc, "D_other": Dj, "fraction": a / total})
    return flags


def shared_shapes_slice_results(result: dict, rows, slices=None) -> list[dict]:
    """A :func:`fit_series_shared_shapes` result as per-slice results Fit Curve can read
    (the same layout the lineshape-series fits use: ``Value source`` Area / Amplitude).

    ``rows`` are the 1-based slice indices the fitted rows came from, in order; ``slices``
    optionally the matching trace spectra (kept as ``"slice"`` for the viewer)."""
    rows = list(rows)
    n = result["amplitudes"].shape[0]
    if len(rows) != n:
        raise ValueError("rows must name one slice per fitted row.")
    per_height = np.array([_area_per_height(w, e) for w, e in zip(result["fwhm"], result["eta"])])
    x = result["x"]
    out = []
    for r in range(n):
        comps = [result["amplitudes"][r, k] * pseudo_voigt(x, result["positions"][k],
                                                           result["fwhm"][k], result["eta"][k])
                 for k in range(len(result["labels"]))]
        base = result["fit"][r] - np.sum(comps, axis=0)
        out.append({
            "index": int(rows[r]), "slice": None if slices is None else slices[r],
            "x": x, "y": result["data"][r], "y_fit": result["fit"][r], "components": comps,
            "baseline": base, "fit_mask": np.ones(x.size, dtype=bool),
            "amplitudes": [float(v) for v in result["amplitudes"][r]],
            "amplitude_errors": [float(v) for v in result["amplitude_errors"][r]],
            "raw_amplitudes": [float(v) for v in result["amplitudes"][r]],
            "raw_amplitude_errors": [float(v) for v in result["amplitude_errors"][r]],
            "reference_heights": [1.0] * len(per_height),
            "integrals": [float(v) for v in result["areas"][r]],
            "integral_errors": [float(v) for v in result["area_errors"][r]],
            "site_labels": list(result["labels"]),
            "reference_areas": [float(v) for v in per_height],
            "baseline_coefficients": [float(v) for v in result["baseline_coefficients"][r]],
            "r_squared": float(result["r_squared"][r]), "rmse": float(np.sqrt(np.mean(result["residual"][r] ** 2))),
            "model": "shared_shapes", "model_fit": True, "reference_row": None,
            "baseline_mode": "linear", "fit_regions": None,
        })
    return out
