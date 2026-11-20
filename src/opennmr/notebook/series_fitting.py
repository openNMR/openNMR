"""
Fit every slice of a pseudo-2D series with a lineshape model fitted to
one slice -- T1/T2/T1rho (or any series) with CSA, quadrupolar,
Czjzek (distributed quadrupolar) or Solid-state NMR (mixed) sites.

Explicit user request (2026-09-28): "When I try to fit T1/T2 pseudo-2d
data using CSA or quad models (or mixed), I extract a slice, then fit
the single slice, but then the 'Export to pseudo2d' button remains
greyed out. This prevents T1/T2 fitting using CSA/quad models."

**The method.** In a relaxation series the *shape* of each site's
pattern does not change from slice to slice -- only how much of it
there is. So once one slice has been fitted with the full nonlinear
model, every slice is fitted as a *linear* combination of that fit's
per-site lineshapes plus a baseline::

    y_slice(x) = sum_k a_k * c_k(x) + b0 + b1 * (x - x_ref)

where ``c_k`` is site k's fitted curve on the reference slice. This is
exact least squares (no starting guesses, no convergence to check),
takes milliseconds per slice, stays well defined on slices whose signal
has decayed into the noise -- where a free nonlinear refit would chase
noise -- and gives each site's intensity a proper standard error from
the covariance ``s^2 (A^T A)^-1``.

For each slice and site it reports:

``amplitudes`` : ``a_k`` -- the site's intensity relative to the
    reference slice (1 there, up to noise and baseline). Above 1 wherever
    a slice is stronger than the reference (see :func:`reference_slice_note`).
``raw_amplitudes`` : ``a_k`` times the site's height on the reference
    curve -- the peak height in the spectrum's own units, not normalised.
``integrals`` : ``a_k`` times the area of site k's reference curve --
    the site's fitted area in this slice, in the spectrum's intensity
    units x ppm. The natural T1/T2 quantity.

The results have the same shape as :func:`opennmr.processing.fitting.
NMR2D_fit_slices`' (``index``, ``slice``, ``x``, ``y``, ``y_fit``, plus
``integrals``/``amplitudes`` and their errors), marked
``model_fit=True``, so :func:`opennmr.processing.relaxation.
NMR2D_fit_relaxation` fits them with ``value_source="area"`` or
``"amplitude"`` exactly as it does a peak fit::

    from opennmr.notebook.series_fitting import (
        model_components, fit_series_with_lineshapes, series_fit_report)

    labels, x, comps = model_components("csa", csa_fit)     # from one slice's fit
    series = fit_series_with_lineshapes(pseudo2d, x, comps, labels,
                                        fit_regions=[(-200, 400)])
    print(series_fit_report(series))
    NMR2D_fit_relaxation(series, model="inversion_recovery", vdlist=vd,
                         value_source="area")

Amplitudes are **not** bounded at zero: inversion recovery crosses
zero, and a series must be allowed to. (The CSA-only
:func:`opennmr.notebook.csa_relaxation.NMR2D_fit_csa_slices`, which fits
its own reference slice from notebook dicts with the same frozen-shape
method, holds amplitudes >= 0 by default -- pass
``allow_negative_intensity=True`` there for inversion recovery.) This
module generalises that method to the quadrupolar and Combined-
Hamiltonian fits, and starts from a fit already made.

**Not done here.** A per-slice refit of the lineshape parameters
themselves (a CSA or C_Q changing along the series). If a site's shape
really changes, model it as two sites with different relaxation, or fit
chosen slices individually and compare.
"""

from __future__ import annotations

import numpy as np

# numpy 2 renamed trapz; keep older installs working.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz

#: The model kinds :func:`model_components` understands.
MODEL_KINDS = ("csa", "quadrupolar", "czjzek", "combined")

_KIND_LABELS = {"csa": "CSA", "quadrupolar": "Quadrupolar", "czjzek": "Czjzek",
                "combined": "Solid-state NMR"}


def model_components(kind: str, fit: dict, x=None):
    """``(labels, x, components)`` -- one fitted curve per site on
    ascending ppm ``x`` -- from a single-spectrum fit.

    ``kind`` : "csa" -- ``fit`` holds ``popt`` and ``info`` from
    :func:`opennmr.notebook.csa_fitting.fit_csa_spectrum` (and ``x``,
    the ppm axis, unless given); "quadrupolar" -- ``popt``/``info``
    from :func:`opennmr.notebook.quadrupolar_fitting.
    fit_quadrupolar_spectrum` (or precomputed ``components``);
    "czjzek" -- ``popt``/``info`` from :func:`opennmr.notebook.czjzek.
    fit_czjzek_spectrum` (its ``info`` keeps the lineshape library, so
    the curves are exactly the fitted ones); "combined" -- the dict :func:`opennmr.notebook.combined_fitting.
    fit_combined` returns (its ``site_spectra`` on its ``x``).

    The baseline is not a component: each slice gets its own."""
    if kind not in MODEL_KINDS:
        raise ValueError(f"kind must be one of {MODEL_KINDS}, not {kind!r}.")
    if kind == "combined":
        xs = np.asarray(fit["x"], dtype=float)
        spectra = fit["site_spectra"]
        labels = list(spectra)
        comps = [np.asarray(spectra[k], dtype=float) for k in labels]
        order = np.argsort(xs)
        xs = xs[order]
        comps = [c[order] for c in comps]
        if x is not None:
            x = np.sort(np.asarray(x, dtype=float))
            comps = [np.interp(x, xs, c) for c in comps]
            xs = x
        return labels, xs, comps

    xs = np.sort(np.asarray(fit["x"] if x is None else x, dtype=float))
    info = fit["info"]
    labels = list(info.get("labels") or [f"Site {k + 1}" for k in range(info["n_sites"])])
    if kind == "csa":
        from .csa_fitting import evaluate_csa_fit
        _total, comps = evaluate_csa_fit(xs, fit["popt"], info, return_components=True)
    elif kind == "czjzek":
        from .czjzek import evaluate_czjzek_fit
        _total, comps = evaluate_czjzek_fit(xs, fit["popt"], info)
    else:
        from .quadrupolar_fitting import evaluate_quad_fit
        _total, comps = evaluate_quad_fit(xs, fit["popt"], info, return_components=True)
    comps = [np.asarray(c, dtype=float) for c in comps[:len(labels)]]   # last is the baseline
    return labels, xs, comps


def _region_mask(x, fit_regions):
    if not fit_regions:
        return np.ones(x.size, dtype=bool)
    mask = np.zeros(x.size, dtype=bool)
    for lo, hi in fit_regions:
        mask |= (x >= min(lo, hi)) & (x <= max(lo, hi))
    return mask


def _as_rows(spectrum, rows):
    """``[(1-based index, 1D NMRSpectrum), ...]`` from a pseudo-2D
    spectrum (or a list of 1D spectra, indexed from 1)."""
    if isinstance(spectrum, (list, tuple)):
        items = list(spectrum)
        wanted = rows or list(range(1, len(items) + 1))
        return [(i, items[i - 1]) for i in wanted]
    from ..processing import slicing
    if spectrum.ndim != 2:
        raise ValueError("Needs a pseudo-2D spectrum (or a list of 1D slices).")
    n = spectrum.data.shape[0]
    wanted = list(rows) if rows else list(range(1, n + 1))
    return [(i, slicing.get_trace(spectrum, axis="f1", index=i)) for i in wanted]


def _nonnegative_coefficients(design, y, n_sites):
    """Least squares with the first ``n_sites`` coefficients >= 0 and the rest (baseline) free."""
    from scipy.optimize import nnls
    free = design[:, n_sites:]
    aug = np.column_stack([design[:, :n_sites], free, -free]) if free.shape[1] else design
    sol, _ = nnls(aug, y)
    k = free.shape[1]
    return np.concatenate([sol[:n_sites], sol[n_sites:n_sites + k] - sol[n_sites + k:n_sites + 2 * k]])


def fit_series_with_lineshapes(spectrum, x_components, components, labels=None, rows=None,
                               fit_regions=None, baseline: str = "linear",
                               kind: str = "model", reference_row: int | None = None,
                               non_negative: bool = False):
    """Fit every slice as a linear combination of fixed site lineshapes.

    Parameters
    ----------
    spectrum : a pseudo-2D :class:`~opennmr.core.spectrum.NMRSpectrum`,
        or a list of 1D spectra sharing one ppm axis.
    x_components, components : ascending ppm axis and one curve per site
        on it -- :func:`model_components` output. Interpolated onto each
        slice's own axis if that differs.
    labels : one name per site.
    rows : 1-based slice indices to fit (default every slice).
    fit_regions : ``[(lo, hi), ...]`` ppm -- the window the fit uses
        (use the one the reference fit used). Default the whole axis.
    baseline : "linear" (default; offset and slope, per slice),
        "constant" or "none".
    non_negative : amplitudes are constrained to be >= 0 (NNLS; the baseline stays free in
        sign). Without it a weak or vanished site (e.g. the bulk peak at high b) can come back
        with a negative amplitude. The quoted errors are the unconstrained ones, which are
        conservative for a site held at zero.
    kind : recorded in each result ("csa", "quadrupolar", "combined").
    reference_row : the slice the lineshapes came from (1-based),
        recorded so the report can say so.

    Returns
    -------
    list of dict, one per slice -- see the module docstring.
    """
    if baseline not in ("linear", "constant", "none"):
        raise ValueError("baseline must be 'linear', 'constant' or 'none'.")
    xc = np.asarray(x_components, dtype=float)
    order_c = np.argsort(xc)
    xc = xc[order_c]
    comps = [np.asarray(c, dtype=float)[order_c] for c in components]
    if not comps:
        raise ValueError("No site lineshapes to fit with.")
    labels = list(labels) if labels else [f"Site {k + 1}" for k in range(len(comps))]
    # Each site's area on the reference, over the whole axis: what turns
    # a relative amplitude into an area.
    # Signed: an inverted reference slice (early in inversion recovery)
    # has negative site curves, and the area must keep that sign.
    areas = np.array([float(_trapezoid(c, xc)) for c in comps])
    # Each site's height on the reference (the value at its largest |value|,
    # so the sign of an inverted reference is kept): what turns a relative
    # amplitude into a peak height in the spectrum's own units.
    heights = np.array([float(c[int(np.argmax(np.abs(c)))]) if c.size else 0.0 for c in comps])

    results = []
    for index, trace in _as_rows(spectrum, rows):
        x_native = np.asarray(trace.axis_values(0, unit="ppm"), dtype=float)
        y_native = np.real(np.asarray(trace.data)).astype(float)
        order = np.argsort(x_native)
        x = x_native[order]
        y = y_native[order]
        c_here = comps if (x.size == xc.size and np.allclose(x, xc)) else \
            [np.interp(x, xc, c) for c in comps]
        mask = _region_mask(x, fit_regions)
        x_ref = float(np.mean(x[mask])) if mask.any() else 0.0
        columns = list(c_here)
        if baseline in ("linear", "constant"):
            columns.append(np.ones_like(x))
        if baseline == "linear":
            columns.append(x - x_ref)
        design = np.column_stack(columns)
        a_mask, y_mask = design[mask], y[mask]
        n_obs, n_par = a_mask.shape
        if n_obs <= n_par:
            raise ValueError(f"Slice {index}: the fit window holds only {n_obs} points.")
        coef, *_ = np.linalg.lstsq(a_mask, y_mask, rcond=None)
        if non_negative:
            coef = _nonnegative_coefficients(a_mask, y_mask, len(comps))
        fitted_mask = a_mask @ coef
        residual = y_mask - fitted_mask
        dof = n_obs - n_par
        s2 = float(residual @ residual) / dof
        gram = a_mask.T @ a_mask
        try:
            cov = s2 * np.linalg.inv(gram)
        except np.linalg.LinAlgError:
            cov = s2 * np.linalg.pinv(gram)
        errors = np.sqrt(np.clip(np.diag(cov), 0.0, None))
        n_sites = len(comps)
        amps, amp_err = coef[:n_sites], errors[:n_sites]
        ss_tot = float(((y_mask - y_mask.mean()) ** 2).sum())
        r2 = 1.0 - float(residual @ residual) / ss_tot if ss_tot > 0 else float("nan")

        y_fit_sorted = design @ coef
        site_sorted = [a * c for a, c in zip(amps, c_here)]
        base_sorted = y_fit_sorted - np.sum(site_sorted, axis=0)

        def native(a):
            out = np.empty_like(a)
            out[order] = a
            return out

        raw = dict(getattr(trace.metadata, "raw", None) or {})
        raw.setdefault("slice", {"slice_type": "row", "index": index})
        import dataclasses
        slice_spec = dataclasses.replace(trace, metadata=dataclasses.replace(trace.metadata, raw=raw))
        results.append({
            "index": int(index), "slice": slice_spec,
            "x": x_native, "y": y_native, "y_fit": native(y_fit_sorted),
            "components": [native(c) for c in site_sorted], "baseline": native(base_sorted),
            "fit_mask": native(mask),
            "amplitudes": [float(v) for v in amps],
            "amplitude_errors": [float(v) for v in amp_err],
            # The same amplitudes in the spectrum's own units (a_k times the
            # site's reference height): what a peak fit's own amplitude is.
            "raw_amplitudes": [float(v) for v in amps * heights],
            "raw_amplitude_errors": [float(v) for v in amp_err * np.abs(heights)],
            "reference_heights": [float(v) for v in heights],
            "integrals": [float(v) for v in amps * areas],
            "integral_errors": [float(v) for v in amp_err * areas],
            "site_labels": labels, "reference_areas": [float(v) for v in areas],
            "baseline_coefficients": [float(v) for v in coef[n_sites:]],
            "r_squared": r2, "rmse": float(np.sqrt(s2)),
            "model": kind, "model_fit": True, "reference_row": reference_row,
            "baseline_mode": baseline, "fit_regions": fit_regions, "non_negative": bool(non_negative),
        })
    return results


def reference_slice_note(results) -> str | None:
    """A warning when the relative amplitudes of a model fitted to every slice
    exceed 1, or ``None``.

    ``amplitudes`` are relative to the slice the lineshapes came from (1
    there). If that is not the slice with the most signal, the stronger slices
    read above 1 and a relaxation curve drawn from them runs above 1 too --
    correct, but easy to mistake for an error. Says which slice is the
    strongest, and what to do."""
    if not results or not results[0].get("model_fit"):
        return None
    amps = [r.get("amplitudes") for r in results]
    if any(a is None for a in amps):
        return None
    n_sites = min(len(a) for a in amps)
    if not n_sites:
        return None
    table = np.abs(np.array([list(a)[:n_sites] for a in amps], float))
    labels = list(results[0].get("site_labels") or [])
    high = []
    for k in range(n_sites):
        column = table[:, k]
        if not np.any(np.isfinite(column)):
            continue
        i = int(np.nanargmax(column))
        if column[i] > 1.05:
            high.append((labels[k] if k < len(labels) else f"Site {k + 1}",
                         float(column[i]), int(results[i]["index"])))
    if not high:
        return None
    reference = results[0].get("reference_row")
    where = f"slice {reference}" if reference else "the reference slice"
    reached = "; ".join(f"{name} {value:.2f} (slice {index})" for name, value, index in high[:4])
    strongest = high[0][2]
    return (f"The lineshapes were taken from {where}, which is not the strongest slice: "
            f"relative amplitudes reach {reached}, so the curve goes above 1. Take the "
            f"lineshapes from slice {strongest} to have 1 at the maximum, or choose Value "
            "source ▸ Fitted peak amplitude (raw) for values in the spectrum's own units.")


def series_fit_report(results) -> str:
    """A plain-text table of every slice: each site's area and relative
    amplitude with 1-sigma errors, and the slice's R^2."""
    if not results:
        return "No slices fitted."
    first = results[0]
    labels = first["site_labels"]
    kind = _KIND_LABELS.get(first.get("model"), first.get("model") or "Model")
    ref = first.get("reference_row")
    lines = [f"{kind} lineshapes applied to {len(results)} slice(s)"
             + (f", taken from slice {ref}" if ref else "") + ".",
             "Each slice fitted as a linear combination of the fixed site lineshapes "
             f"plus a {first.get('baseline_mode', 'linear')} baseline.",
             ""]
    header = ["Slice"]
    for lab in labels:
        header += [f"{lab} area", "±", f"{lab} rel.", "±"]
    header.append("R²")
    lines.append("\t".join(header))
    for r in results:
        row = [str(r["index"])]
        for k in range(len(labels)):
            row += [f"{r['integrals'][k]:.5g}", f"{r['integral_errors'][k]:.2g}",
                    f"{r['amplitudes'][k]:.4f}", f"{r['amplitude_errors'][k]:.2g}"]
        row.append(f"{r['r_squared']:.4f}" if np.isfinite(r["r_squared"]) else "—")
        lines.append("\t".join(row))
    low = [r["index"] for r in results if np.isfinite(r["r_squared"]) and r["r_squared"] < 0.9]
    if low:
        lines += ["", f"R² below 0.9 in slice(s) {', '.join(map(str, low))} – expected where "
                      "the signal has decayed into the noise, a sign of a changing lineshape "
                      "otherwise."]
    return "\n".join(lines)
