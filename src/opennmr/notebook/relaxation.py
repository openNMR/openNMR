"""
Relaxation-curve fitting (T1, T2, T1rho) for nmrtools.

This module is deliberately decoupled from the peak-fitting machinery
in ``fitting.py``: it operates on plain ``(t, y[, y_err])`` arrays, so
it can fit a relaxation curve to *any* series of intensities/integrals
against delay time, not only ones produced by
:func:`nmrtools.fitting.NMR2D_fit_slices`. A convenience wrapper,
:func:`NMR2D_fit_relaxation`, is provided for the common case of
fitting directly from that function's output.

Models provided
----------------
- ``saturation_recovery`` / ``inversion_recovery``  (T1)
- ``exponential_decay``                              (T2 / T1rho)
- ``biexponential_recovery`` / ``biexponential_decay``
- ``triexponential_recovery`` / ``triexponential_decay`` (three components)
- ``stretched_recovery`` / ``stretched_decay``        (stretched/KWW,
  for distributions of relaxation times)

An intensity series can come from either fitted peaks
(:func:`nmrtools.fitting.NMR2D_fit_slices`) or, when individual peaks
aren't cleanly resolved enough to be worth modelling, direct
region integration (:func:`NMR2D_integrate_slices`) -- both produce
output shaped the same way, so :func:`NMR2D_fit_relaxation` accepts
either interchangeably.

Workflow
--------

    NMR2D_get_slices -> [NMR2D_fit_slices | NMR2D_integrate_slices]
                      -> NMR2D_fit_relaxation -> NMR2D_plot_relaxation

or, end-to-end, :func:`NMR2D_relaxation_workflow`. Alternatively, fit
an arbitrary (t, y) array directly with :func:`fit_relaxation`.
"""

import numpy as np
import matplotlib.pyplot as plt
from math import floor, log10
from scipy.optimize import curve_fit

from .utils import read_vdlist
from .plotting import setup_plotting, apply_tick_style, apply_spine_style, save_figure
from .fitting import build_region_mask, estimate_baseline_robust, estimate_noise_level, NMR2D_fit_slices
from .two_d import NMR2dimport, NMR2D_get_slices, NMR2D_plot_slices, prepare_trace_for_fit
from ..processing import relaxometry as _relaxometry
from ..processing.relaxometry import triexponential_decay, triexponential_recovery  # noqa: F401

# numpy >= 2.0 renamed trapz -> trapezoid
_trapz = getattr(np, "trapezoid", None) or np.trapz


# ============================================================
# Models
# ============================================================

def saturation_recovery(t, M0, T1):
    """Longitudinal recovery from zero magnetisation: ``M0*(1 - exp(-t/T1))``."""
    return M0 * (1.0 - np.exp(-t / T1))


def inversion_recovery(t, M0, T1, C=2.0):
    """
    Longitudinal recovery from an inverted state:
    ``M0*(1 - C*exp(-t/T1))``.

    ``C`` defaults to 2 (perfect inversion) but is a free parameter in
    :func:`fit_relaxation` unless you fix it -- real inversion pulses
    are rarely perfectly calibrated, and fixing C=2 when it isn't will
    bias the fitted T1.
    """
    return M0 * (1.0 - C * np.exp(-t / T1))


def exponential_decay(t, M0, T2):
    """Mono-exponential decay: ``M0*exp(-t/T2)``. Used for T2 and T1rho."""
    return M0 * np.exp(-t / T2)


def biexponential_decay(t, A1, T2a, A2, T2b):
    """Sum of two independent decaying components."""
    return A1 * np.exp(-t / T2a) + A2 * np.exp(-t / T2b)


def biexponential_recovery(t, A1, T1a, A2, T1b, C=2.0):
    """
    Sum of two independent recovering components, sharing a common
    inversion efficiency ``C``. Set ``C=1`` (fixed) for a
    saturation-recovery biexponential.
    """
    return A1 * (1.0 - C * np.exp(-t / T1a)) + A2 * (1.0 - C * np.exp(-t / T1b))


def stretched_decay(t, M0, T2, beta):
    """
    Stretched-exponential (Kohlrausch-Williams-Watts) decay:
    ``M0*exp(-(t/T2)**beta)``. ``beta`` (0 < beta <= 1) parameterises
    a distribution of relaxation times rather than a single value;
    beta=1 recovers a mono-exponential.
    """
    return M0 * np.exp(-(t / T2) ** beta)


def stretched_recovery(t, M0, T1, beta, C=2.0):
    """Stretched-exponential recovery: ``M0*(1 - C*exp(-(t/T1)**beta))``."""
    return M0 * (1.0 - C * np.exp(-(t / T1) ** beta))


# Registry: name -> (function, param names, default p0 fn, default bounds fn)
# p0/bounds functions take (t, y) and return a list matching param order.

def _p0_mono(t, y):
    M0 = np.max(np.abs(y)) if np.max(np.abs(y)) > 0 else 1.0
    T = np.median(t[t > 0]) if np.any(t > 0) else 1.0
    return [M0, T]


def _bounds_mono(t, y):
    return ([-np.inf, 1e-9], [np.inf, np.inf])


def _p0_mono_C(t, y):
    return _p0_mono(t, y) + [2.0]


def _bounds_mono_C(t, y):
    lo, hi = _bounds_mono(t, y)
    return (lo + [0.0], hi + [4.0])


def _p0_biexp_decay(t, y):
    M0 = np.max(np.abs(y)) if np.max(np.abs(y)) > 0 else 1.0
    T = np.median(t[t > 0]) if np.any(t > 0) else 1.0
    return [M0 * 0.5, T * 0.3, M0 * 0.5, T * 2.0]


def _bounds_biexp_decay(t, y):
    return ([-np.inf, 1e-9, -np.inf, 1e-9], [np.inf, np.inf, np.inf, np.inf])


def _p0_biexp_recovery(t, y):
    return _p0_biexp_decay(t, y) + [2.0]


def _bounds_biexp_recovery(t, y):
    lo, hi = _bounds_biexp_decay(t, y)
    return (lo + [0.0], hi + [4.0])


def _p0_stretched(t, y):
    return _p0_mono(t, y) + [0.8]


def _bounds_stretched(t, y):
    lo, hi = _bounds_mono(t, y)
    return (lo + [0.05], hi + [1.5])


def _p0_stretched_C(t, y):
    return _p0_mono(t, y) + [0.8, 2.0]


def _bounds_stretched_C(t, y):
    lo, hi = _bounds_mono(t, y)
    return (lo + [0.05, 0.0], hi + [1.5, 4.0])


RELAXATION_MODELS = {
    "saturation_recovery": {
        "func": saturation_recovery, "params": ["M0", "T1"],
        "p0": _p0_mono, "bounds": _bounds_mono, "rate_param": "T1",
    },
    "inversion_recovery": {
        "func": inversion_recovery, "params": ["M0", "T1", "C"],
        "p0": _p0_mono_C, "bounds": _bounds_mono_C, "rate_param": "T1",
    },
    "exponential_decay": {
        "func": exponential_decay, "params": ["M0", "T2"],
        "p0": _p0_mono, "bounds": _bounds_mono, "rate_param": "T2",
    },
    "biexponential_decay": {
        "func": biexponential_decay, "params": ["A1", "T2a", "A2", "T2b"],
        "p0": _p0_biexp_decay, "bounds": _bounds_biexp_decay, "rate_param": None,
    },
    "biexponential_recovery": {
        "func": biexponential_recovery, "params": ["A1", "T1a", "A2", "T1b", "C"],
        "p0": _p0_biexp_recovery, "bounds": _bounds_biexp_recovery, "rate_param": None,
    },
    "stretched_decay": {
        "func": stretched_decay, "params": ["M0", "T2", "beta"],
        "p0": _p0_stretched, "bounds": _bounds_stretched, "rate_param": "T2",
    },
    "stretched_recovery": {
        "func": stretched_recovery, "params": ["M0", "T1", "beta", "C"],
        "p0": _p0_stretched_C, "bounds": _bounds_stretched_C, "rate_param": "T1",
    },
    # Three components (explicit user request, 2026-10-06): grid-searched
    # starts, multi-start fit, fastest component first -- see
    # opennmr.processing.relaxometry.
    **_relaxometry.TRIEXP_MODELS,
}

# Convenience aliases: T1rho experiments use the same functional forms
# as T2 (decay) or T1 (recovery) depending on the pulse sequence used;
# no separate model is needed, just fit with the appropriate decay/
# recovery model and relabel the rate constant "T1rho" yourself.


# ============================================================
# Fitting
# ============================================================

def fit_relaxation(t, y, model="inversion_recovery", y_err=None, p0=None,
                    bounds=None, fixed=None, maxfev=20000):
    """
    Fit a relaxation curve ``y(t)`` with a named model from
    :data:`RELAXATION_MODELS`.

    Parameters
    ----------
    t, y : array-like
        Delay (seconds) and intensity/integral at each delay.

    model : str
        One of :data:`RELAXATION_MODELS`.

    y_err : array-like, optional
        Per-point uncertainty (e.g. ``integral_errors`` from
        :func:`nmrtools.fitting.NMR2D_fit_slices`). If given, used as
        ``sigma`` in the least-squares fit (points are weighted by
        1/error^2) and reported back as parameter uncertainties that
        account for the measurement errors rather than only the
        fit residual scatter.

    p0 : list, optional
        Initial parameter guess, in the order given by
        ``RELAXATION_MODELS[model]["params"]``. If None, a data-driven
        default is used.

    bounds : (lower, upper), optional
        Parameter bounds. If None, sensible defaults are used
        (relaxation times constrained positive; ``beta`` in (0.05,
        1.5]; inversion efficiency ``C`` in [0, 4]).

    fixed : dict, optional
        ``{param_name: value}`` to hold fixed during the fit (e.g.
        ``{"C": 2.0}`` to force perfect inversion).

    Returns
    -------
    dict with keys:
        ``params``       : {name: value} fitted (and fixed) parameters.
        ``errors``       : {name: 1-sigma uncertainty} (nan for fixed params).
        ``popt``, ``pcov`` : raw arrays for the free parameters, in
            model parameter order (fixed params excluded).
        ``model``, ``param_names``, ``rate_param`` : bookkeeping.
        ``t``, ``y``, ``y_err``, ``y_fit`` : the data and fitted curve.
        ``fit_quality`` : dict with rmse, chi2, reduced_chi2.
    """

    if model not in RELAXATION_MODELS:
        raise ValueError(
            f"Unknown relaxation model '{model}'. Available: {sorted(RELAXATION_MODELS)}"
        )

    spec = RELAXATION_MODELS[model]
    func, param_names = spec["func"], spec["params"]

    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    y_err = None if y_err is None else np.asarray(y_err, dtype=float)

    order = np.argsort(t)
    t, y = t[order], y[order]
    if y_err is not None:
        y_err = y_err[order]

    fixed = fixed or {}
    free_names = [p for p in param_names if p not in fixed]

    if not free_names:
        raise ValueError("Every parameter is fixed -- nothing to fit.")

    p0_full = spec["p0"](t, y) if p0 is None else list(p0)
    lo_full, hi_full = spec["bounds"](t, y) if bounds is None else bounds

    free_idx = [i for i, p in enumerate(param_names) if p not in fixed]
    p0_free = [p0_full[i] for i in free_idx]
    lo_free = [lo_full[i] for i in free_idx]
    hi_free = [hi_full[i] for i in free_idx]

    def wrapper(t, *free_vals):
        full = dict(fixed)
        full.update(zip(free_names, free_vals))
        return func(t, *[full[p] for p in param_names])

    sigma = y_err if (y_err is not None and np.all(np.isfinite(y_err)) and np.all(y_err > 0)) else None

    fit_kwargs = dict(spec.get("curve_fit_kwargs") or {})

    def fit_once(start):
        popt_, pcov_ = curve_fit(
            wrapper, t, y, p0=start, bounds=(lo_free, hi_free),
            sigma=sigma, absolute_sigma=sigma is not None, maxfev=maxfev, **fit_kwargs,
        )
        r = (y - wrapper(t, *popt_)) / (sigma if sigma is not None else 1.0)
        return popt_, pcov_, float(np.sum(r ** 2))

    starts = _relaxometry.candidate_starts(spec, t, y, p0_full, p0, free_idx)
    if len(starts) == 1:
        popt, pcov, _cost = fit_once(p0_free)
    else:
        popt, pcov = _relaxometry.fit_from_starts(fit_once, starts, lo_free, hi_free)

    perr = np.sqrt(np.diag(pcov))

    params = dict(fixed)
    errors = {p: np.nan for p in fixed}
    for name, val, err in zip(free_names, popt, perr):
        params[name] = val
        errors[name] = err

    y_fit = func(t, *[params[p] for p in param_names])
    residual = y - y_fit
    rmse = np.sqrt(np.mean(residual ** 2))
    chi2 = np.sum((residual / sigma) ** 2) if sigma is not None else np.sum(residual ** 2)
    red_chi2 = chi2 / max(len(y) - len(free_names), 1)

    result = {
        "model": model, "param_names": param_names, "rate_param": spec["rate_param"],
        "params": params, "errors": errors,
        "popt": popt, "pcov": pcov, "free_names": free_names,
        "fixed": dict(fixed), "bounds": (list(lo_full), list(hi_full)),
        "t": t, "y": y, "y_err": y_err, "y_fit": y_fit,
        "fit_quality": {"rmse": rmse, "chi2": chi2, "reduced_chi2": red_chi2},
    }
    return _relaxometry.sort_components(result, spec.get("components"))


def format_time_value(value, error=None, sig_figs=3):
    """
    Format a time value (assumed given in seconds) with an
    automatically chosen unit (s, ms, us, or ns) for display, e.g.
    ``format_time_value(0.00152, 0.00003)`` -> ``"1.52 \u00b1 0.03 ms"``.

    Parameters
    ----------
    value : float
        Time in seconds.
    error : float, optional
        1-sigma uncertainty, in seconds -- converted to the same unit
        as ``value`` and appended.
    sig_figs : int, optional
        Significant figures used to round ``value``. ``error`` is
        rounded to the same number of decimal places as the rounded
        ``value``, so the two line up. Default 3.

    Returns
    -------
    str
    """
    if value is None or not np.isfinite(value):
        return "n/a"

    abs_val = abs(value)
    if abs_val == 0 or abs_val >= 1.0:
        unit, factor = "s", 1.0
    elif abs_val >= 1e-3:
        unit, factor = "ms", 1e3
    elif abs_val >= 1e-6:
        unit, factor = "\u00b5s", 1e6
    else:
        unit, factor = "ns", 1e9

    val_scaled = value * factor

    if val_scaled == 0:
        decimals = max(sig_figs - 1, 0)
    else:
        decimals = max(sig_figs - int(floor(log10(abs(val_scaled)))) - 1, 0)

    val_str = f"{round(val_scaled, decimals):.{decimals}f}"

    if error is not None and np.isfinite(error):
        err_scaled = error * factor
        err_str = f"{round(err_scaled, decimals):.{decimals}f}"
        return f"{val_str} \u00b1 {err_str} {unit}"

    return f"{val_str} {unit}"


def print_relaxation_report(fit):
    """Print a human-readable summary of a :func:`fit_relaxation` result."""
    print("\n" + "=" * 50)
    print(f"RELAXATION FIT REPORT ({fit['model']})")
    print("=" * 50)
    for name in fit["param_names"]:
        val, err = fit["params"][name], fit["errors"][name]
        if np.isnan(err):
            print(f"  {name:<8}= {val:.6g} (fixed)")
        else:
            print(f"  {name:<8}= {val:.6g} \u00b1 {err:.4g}")

    rate_param = fit["rate_param"]
    if rate_param:
        print(f"\n  {rate_param} = {format_time_value(fit['params'][rate_param], fit['errors'][rate_param])}")

    mq = fit["fit_quality"]
    print(f"\n  RMSE          = {mq['rmse']:.4g}")
    print(f"  Reduced chi^2 = {mq['reduced_chi2']:.4g}")
    print("=" * 50 + "\n")


def _select_series(fit_results, show_series):
    """
    Filter a list of :func:`fit_relaxation`-style results (each with
    a ``"peak_label"`` key) down to a requested subset, given either
    0-based indices or label strings. Returns all results unchanged if
    ``show_series`` is None.
    """
    if show_series is None:
        return list(fit_results)

    selected = []
    for item in show_series:
        if isinstance(item, str):
            match = next((f for f in fit_results if f.get("peak_label") == item), None)
            if match is None:
                available = [f.get("peak_label") for f in fit_results]
                raise ValueError(f"No series labelled '{item}'. Available: {available}")
            selected.append(match)
        else:
            selected.append(fit_results[item])
    return selected


# ============================================================
# Region-based integration (peak-fitting-free alternative)
# ============================================================

def NMR2D_integrate_slices(slices, regions, mode="rows", region_labels=None,
                            normalise=False, x_unit='ppm',
                            subtract_baseline=True, baseline_regions=None,
                            baseline_sigma=2.5, baseline_max_iter=10,
                            noise_region=None, method="trapezoid"):
    """
    Integrate one or more extracted 2D NMR slices (rows and/or
    columns) over user-specified x-axis regions -- a peak-fitting-free
    alternative to :func:`nmrtools.fitting.NMR2D_fit_slices`, useful
    when individual peaks are too poorly resolved or overlapping to
    be worth modelling, and a bulk-region integral (e.g. an aromatic
    region ``[6.5, 8.0]`` vs an aliphatic region ``[0.5, 3.0]``) is
    all that's needed for a relaxation curve.

    Parameters
    ----------
    slices : dict
        Output of :func:`nmrtools.two_d.NMR2D_get_slices`.
    regions : list of (float, float)
        x-axis regions to integrate, e.g. ``[[3.4, 4.5], [4.5, 8.0]]``
        -- as many as you like. Order is preserved and matches the
        ``"integrals"`` order in the output (and hence
        ``peak_labels``/``region_labels`` in downstream calls).
    mode : {'rows', 'columns', 'both'}, optional
    region_labels : list of str, optional
        Labels for the region series -- forwarded to downstream
        functions as ``peak_labels`` (the argument name is unchanged
        there since it accepts either kind of series).
    normalise : bool, optional
        Normalise each trace (to its own max) before integrating.
    subtract_baseline : bool, optional
        Estimate and subtract a robust linear baseline (see
        :func:`nmrtools.fitting.estimate_baseline_robust`) before
        integrating each region. Recommended: an unsubtracted baseline
        offset biases every region's integral by an amount that
        depends on the region's width, which then propagates directly
        into a distorted relaxation curve. Default True.
    baseline_regions, baseline_sigma, baseline_max_iter : optional
        See :func:`nmrtools.fitting.estimate_baseline_robust`.
    noise_region : tuple or list of tuples, optional
        Region(s) used to estimate each slice's noise level for
        integral error propagation (see
        :func:`nmrtools.fitting.estimate_noise_level`). If None, the
        first/last 5% of each slice is used. Errors are propagated
        analytically through the trapezoidal rule (exact for
        independent per-point Gaussian noise), not just assumed
        constant across regions of different widths.
    method : {'trapezoid', 'simpson'}, optional
        The integration rule (see
        :func:`opennmr.processing.relaxometry.integration_weights`).
        Which is more accurate depends on the region's edges, not on
        Simpson's higher order: over a whole peak whose region reaches
        the baseline the trapezoid rule is already all but exact (its
        error terms are the curve's derivatives at the two ends, which
        vanish there) and Simpson's is no better -- worse on a peak only
        ~2 points per FWHM. Where a region edge cuts through signal,
        Simpson's is better (measured 4-30× smaller error at 3-12 points
        per FWHM). Simpson's alternating weights carry about 5% more
        noise. Either way the error is propagated through the rule's
        own weights.

    Returns
    -------
    dict with keys ``"rows"`` and ``"columns"``, shaped closely enough
    to match :func:`nmrtools.fitting.NMR2D_fit_slices`'s output to be
    used interchangeably with :func:`NMR2D_fit_relaxation`: each entry
    has ``slice``, ``slice_type``, ``index``, ``shift``, ``x``, ``y``,
    ``integrals``, ``integral_errors``, ``integrals_norm``, ``regions``.
    There is no fit, so ``popt``/``pcov``/``components`` are absent.
    """

    if mode not in ["rows", "columns", "both"]:
        raise ValueError("mode must be 'rows', 'columns', or 'both'")

    selected = []
    if mode in ["rows", "both"]:
        selected.extend(("row", s) for s in slices.get("rows", []))
    if mode in ["columns", "both"]:
        selected.extend(("column", s) for s in slices.get("columns", []))

    if not selected:
        raise ValueError("No slices available for integration.")

    integrated = {"rows": [], "columns": []}

    for slice_type, s in selected:

        results = prepare_trace_for_fit(s, normalise=normalise)
        x = results["x_ppm"] if x_unit == "ppm" else results["x_Hz"]
        y = results["real_spectrum"]

        if subtract_baseline:
            b0, b1, x_ref = estimate_baseline_robust(
                x, y, baseline_regions=baseline_regions,
                sigma=baseline_sigma, max_iter=baseline_max_iter,
            )
            y_corr = y - (b0 + b1 * (x - x_ref))
        else:
            y_corr = y

        noise = estimate_noise_level(x, y_corr, noise_region=noise_region)

        integrals, errors = [], []

        for lo, hi in regions:

            mask = build_region_mask(x, [(lo, hi)])
            x_r, y_r = x[mask], y_corr[mask]

            if len(x_r) < 2:
                raise ValueError(f"Region ({lo}, {hi}) contains fewer than 2 points.")

            # x_ppm (and x_Hz) run in descending order by nmrtools'
            # convention (see NMR1dimport); sort ascending first so the
            # integral's sign reflects the data, not the axis direction
            # (np.trapz flips sign under a descending x otherwise).
            sort_idx = np.argsort(x_r)
            x_r, y_r = x_r[sort_idx], y_r[sort_idx]

            # NOT abs()'d, unlike the peak-fit route's integrals: a raw
            # region integral in a signed series (e.g. an inversion-
            # recovery slice before the null point) is genuinely
            # negative, and abs() here would destroy exactly the sign
            # information the relaxation fit needs.
            # The chosen rule's own weights give the integral and,
            # being linear, its noise exactly: I = sum(w_i y_i), so
            # Var(I) = noise^2 sum(w_i^2) for i.i.d. Gaussian noise, for
            # any spacing (relaxometry.integration_weights).
            integral, error = _relaxometry.integrate(x_r, y_r, method=method, noise=noise)

            integrals.append(integral)
            errors.append(error)

        total = np.sum(integrals)
        integrals_norm = [100 * a / total for a in integrals] if total > 0 else [np.nan] * len(integrals)

        integrated_result = {
            "slice": s, "slice_type": slice_type,
            "index": s["index"], "shift": s["shift"],
            "x": x, "y": y_corr,
            "integrals": integrals, "integral_errors": errors, "integrals_norm": integrals_norm,
            "regions": list(regions), "region_labels": region_labels,
            "integration_method": method,
        }

        integrated["rows" if slice_type == "row" else "columns"].append(integrated_result)

    return integrated


# ============================================================
# Convenience: fit directly from NMR2D_fit_slices output
# ============================================================

def _get_delays(fits, vdlist=None):
    """Resolve delay times (seconds), sorted with the fits, for a
    ``fitted["rows"]``/``fitted["columns"]`` list from
    :func:`nmrtools.fitting.NMR2D_fit_slices`."""

    if vdlist is None:
        datapath = next(
            (fit.get("slice", {}).get("datapath") for fit in fits
             if fit.get("slice", {}).get("datapath")), None,
        )
        if datapath is None:
            raise ValueError(
                "No vdlist given and none could be recovered from the "
                "fitted slices -- pass vdlist explicitly (a Bruker "
                "experiment path, the dict from "
                "nmrtools.utils.read_vdlist(), or an array of delays "
                "in seconds indexed by 1-based F1 slice index)."
            )
        vdlist = datapath

    if isinstance(vdlist, str):
        vdlist_seconds = read_vdlist(vdlist)["seconds"]
    elif isinstance(vdlist, dict):
        vdlist_seconds = np.asarray(vdlist["seconds"], dtype=float)
    else:
        vdlist_seconds = np.asarray(vdlist, dtype=float)

    indices = [fit["index"] for fit in fits]
    try:
        delays = np.array([vdlist_seconds[int(i) - 1] for i in indices], dtype=float)
    except IndexError:
        raise ValueError(
            f"vdlist has {len(vdlist_seconds)} entries, but a fitted "
            f"slice index falls outside that range -- check the vdlist "
            f"matches this dataset."
        )

    return delays


def NMR2D_fit_relaxation(fits, model="inversion_recovery", vdlist=None,
                          peak_labels=None, normalised=False, use_errors=True,
                          fixed=None, show_report=True):
    """
    Fit relaxation curves to every peak's integral series across a
    :func:`nmrtools.fitting.NMR2D_fit_slices` output (T1/T2/T1rho
    pseudo-2D series).

    Parameters
    ----------
    fits : list
        ``fitted["rows"]`` or ``fitted["columns"]`` from
        :func:`nmrtools.fitting.NMR2D_fit_slices`. Every entry must
        have been fitted against the *same* ``peak_defs``, in the
        same order, or the per-peak series below are meaningless.
    model : str
        See :data:`RELAXATION_MODELS`.
    vdlist : optional
        See :func:`nmrtools.fitting.NMR2D_plot_integrals` -- same
        resolution rules apply.
    peak_labels : list of str, optional
    normalised : bool, optional
        Fit ``integrals_norm`` (percentage) rather than raw
        integrals. Not recommended for absolute T1/T2 amplitude
        interpretation, but harmless for the relaxation *time*
        itself.
    use_errors : bool, optional
        Weight the fit by each slice's ``integral_errors`` when
        available (see ``y_err`` in :func:`fit_relaxation`).
    fixed : dict, optional
        Forwarded to :func:`fit_relaxation` for every peak.
    show_report : bool, optional
        Print a report for each fitted peak.

    Returns
    -------
    list of dict
        One :func:`fit_relaxation` result per peak, with an added
        ``"peak_label"`` key.
    """

    if not fits:
        raise ValueError("No fitted slices available.")

    t = _get_delays(fits, vdlist=vdlist)
    order = np.argsort(t)

    source = "integrals_norm" if normalised else "integrals"
    n_peaks = max(len(fit.get(source, [])) for fit in fits)

    results = []

    for peak_i in range(n_peaks):

        y = np.array([fit[source][peak_i] if len(fit.get(source, [])) > peak_i
                      else np.nan for fit in fits])[order]
        t_peak = t[order]

        y_err = None
        if use_errors:
            errs = [fit.get("integral_errors", [None])[peak_i]
                    if len(fit.get("integral_errors", [])) > peak_i else None
                    for fit in fits]
            if all(e is not None and np.isfinite(e) for e in errs):
                y_err = np.array(errs)[order]

        valid = np.isfinite(y)
        if y_err is not None:
            valid &= np.isfinite(y_err)

        fit_result = fit_relaxation(t_peak[valid], y[valid], model=model,
                                     y_err=None if y_err is None else y_err[valid],
                                     fixed=fixed)

        label = (peak_labels[peak_i] if peak_labels is not None and peak_i < len(peak_labels)
                 else f"Peak {peak_i + 1}")
        fit_result["peak_label"] = label
        results.append(fit_result)

        if show_report:
            print(f"\n--- {label} ---")
            print_relaxation_report(fit_result)

    return results


def _lighten_colour(colour, amount=0.45):
    """
    Blend ``colour`` toward white by ``amount`` (0 = unchanged, 1 =
    white). Used to derive a relaxation fit curve's colour from its
    data points' colour, so the two are visibly related rather than
    identical or (as previously) uncoordinated.
    """
    import matplotlib.colors as mcolors
    r, g, b = mcolors.to_rgb(colour)
    amount = min(max(amount, 0.0), 1.0)
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)


def _broadcast(value, n, name):
    """
    Resolve a per-series plot-styling argument. ``None`` or a scalar
    (a single colour, string, or a plain ``(dx, dy)`` 2-tuple) applies
    to every series; a ``list`` gives one entry per series and must
    have length ``n`` (i.e. match the number of series being plotted,
    after any ``show_series`` filtering). A tuple is always treated as
    one scalar value (e.g. one RGB colour, or one annotation offset),
    never as "one value per series" -- pass a *list* of tuples for
    that.
    """
    if value is None:
        return [None] * n
    if isinstance(value, list):
        if len(value) != n:
            raise ValueError(
                f"{name} has {len(value)} entries but {n} series are being "
                f"plotted -- give one entry per series (after any "
                f"show_series filtering), or a single value to apply to all."
            )
        return value
    return [value] * n


def _resolve_annotate_anchor(position, t_curve, y_curve):
    """
    Resolve an ``annotate_position`` entry to an (x, y) point on the
    plotted fit curve. Accepts ``'start'``, ``'middle'``, ``'end'``,
    or a float fraction in [0, 1] along the curve (0 = start, 1 = end).
    """
    if isinstance(position, str):
        mapping = {'start': 0.0, 'middle': 0.5, 'end': 1.0}
        if position not in mapping:
            raise ValueError(
                f"annotate_position must be 'start', 'middle', 'end', or a "
                f"float in [0, 1], got {position!r}."
            )
        frac = mapping[position]
    else:
        frac = float(position)
        if not (0.0 <= frac <= 1.0):
            raise ValueError("annotate_position as a float must be in [0, 1].")

    idx = int(round(frac * (len(t_curve) - 1)))
    return t_curve[idx], y_curve[idx]


def NMR2D_plot_relaxation(fit_results, show_series=None, annotate=False,
                           xlabel="Delay (s)", ylabel="Integral",
                           logplot=False,
                           pl_width=3.375, pl_height=3.375, font='Arial', fontsize=12,
                           output_format='',
                           colours=None, fit_colours=None, fit_shade=0.45,
                           annotate_position='end', annotate_offset=(5, 0),
                           annotate_ha='left', annotate_va='center',
                           tick_dir='in', tick_length=4.0, tick_width=1.0, tick_pad=4.0,
                           tick_number=None, show_top_spine=False, show_right_spine=False,
                           n_curve_points=300,
                           save_fig=False, fig_name='nmr_relaxation_plot', save_format='png'):
    """
    Plot one or more :func:`fit_relaxation` results (data + fitted
    curve) on a single axis -- typically the output of
    :func:`NMR2D_fit_relaxation`.

    Parameters
    ----------
    fit_results : list of dict
        Output of :func:`NMR2D_fit_relaxation` (or a list of
        :func:`fit_relaxation` results with a ``"peak_label"`` key
        added).
    show_series : list, optional
        Which series to plot -- a list of 0-based indices into
        ``fit_results``, or of the ``peak_label`` strings. If None
        (default), every series is plotted.
    annotate : bool, optional
        Label each curve directly on the plot with its fitted rate
        constant and uncertainty, formatted in a natural unit
        (s/ms/us/ns -- see :func:`format_time_value`). Off by default;
        the legend already names each series, so this is for when you
        specifically want the relaxation time visible on the figure
        itself (e.g. for a talk slide). Positioned per
        ``annotate_position``/``annotate_offset``, and always drawn in
        the series' (bolder) data colour rather than the paler fit
        curve colour, for legibility.
    xlabel, ylabel : str, optional
    logplot : {False, 'x', 'y', 'both'}, optional
    pl_width, pl_height, font, fontsize, output_format : optional
        See :func:`nmrtools.plotting.setup_plotting`.
    colours : colour or list of colours, optional
        Colour for each series' data points (markers/error bars). A
        single colour applies to every series; a list gives one colour
        per series (must match the number of series being plotted). If
        None (default), colours are drawn from matplotlib's colour
        cycle -- resolved once per series up front, so a series' data
        and fit curve are reliably matched rather than each
        independently advancing the cycle (which is what happened
        previously when both were plotted with ``color=None``).
    fit_colours : colour or list of colours, optional
        Explicit colour for each series' fitted curve, overriding the
        automatic pale tint. If None (default), each fit curve is
        drawn in a paler shade of that series' own data colour (see
        ``fit_shade``) -- so the fit is visually tied to its data
        without being identical to it. Use this to override that
        default for specific series while leaving others automatic
        (pass ``None`` for those entries in the list).
    fit_shade : float, optional
        How much lighter the automatic fit-curve colour is than its
        data colour, in [0, 1] (0 = identical, 1 = white). Default
        0.45. Ignored for any series with an explicit ``fit_colours``
        entry.
    annotate_position : {'start', 'middle', 'end'}, float, or list, optional
        Where each label sits along its fit curve: ``'start'``,
        ``'middle'``, ``'end'`` (default), or a float fraction in
        [0, 1] (0 = start, 1 = end). A single value applies to every
        series; a list gives one per series -- useful when curves
        overlap and different series need labels in different places
        to stay legible.
    annotate_offset : (dx, dy) or list of (dx, dy), optional
        Pixel offset (points) from the anchor point set by
        ``annotate_position`` to where the text is actually drawn.
        Default ``(5, 0)`` (5 points to the right). A single ``(dx,
        dy)`` tuple applies to every series; a *list* of tuples gives
        one per series.
    annotate_ha, annotate_va : str or list of str, optional
        Text alignment relative to its anchor point (matplotlib
        ``ha``/``va`` values, e.g. ``'left'``/``'center'``). Defaults
        suit ``annotate_position='end'``; for ``'start'`` you'll
        usually want ``annotate_ha='right'`` so the label doesn't run
        back over the curve. Single value or per-series list.
    tick_dir, tick_length, tick_width, tick_pad, tick_number : optional
    show_top_spine, show_right_spine : bool, optional
    n_curve_points : int, optional
        Resolution of the plotted fitted curve.
    save_fig, fig_name, save_format : optional
        See :func:`nmrtools.plotting.save_figure`.

    Returns
    -------
    fig, ax
    """

    fit_results = _select_series(fit_results, show_series)
    n = len(fit_results)

    data_colours = _broadcast(colours, n, "colours")
    fit_colours = _broadcast(fit_colours, n, "fit_colours")
    positions = _broadcast(annotate_position, n, "annotate_position")
    offsets = _broadcast(annotate_offset, n, "annotate_offset")
    ha_list = _broadcast(annotate_ha, n, "annotate_ha")
    va_list = _broadcast(annotate_va, n, "annotate_va")

    style = setup_plotting(font=font, fontsize=fontsize, pl_width=pl_width,
                            pl_height=pl_height, output_format=output_format)
    pl_width, pl_height = style['pl_width'], style['pl_height']

    fig, ax = plt.subplots()

    prop_cycle_colours = plt.rcParams['axes.prop_cycle'].by_key().get('color', ['C0'])

    for i, fit in enumerate(fit_results):

        # Resolved once per series -- not passed as color=None to two
        # separate plot calls -- so the data points and fit curve for
        # a given series are reliably related, rather than each
        # silently advancing matplotlib's colour cycle independently.
        data_colour = data_colours[i] if data_colours[i] is not None \
            else prop_cycle_colours[i % len(prop_cycle_colours)]
        curve_colour = fit_colours[i] if fit_colours[i] is not None \
            else _lighten_colour(data_colour, fit_shade)

        label = fit.get("peak_label", f"Fit {i + 1}")

        if fit["y_err"] is not None:
            ax.errorbar(fit["t"], fit["y"], yerr=fit["y_err"], marker='o',
                        linestyle='none', capsize=3, color=data_colour, label=None)
        else:
            ax.plot(fit["t"], fit["y"], marker='o', linestyle='none', color=data_colour, label=None)

        spec = RELAXATION_MODELS[fit["model"]]
        t_curve = np.linspace(fit["t"].min(), fit["t"].max(), n_curve_points)
        y_curve = spec["func"](t_curve, *[fit["params"][p] for p in fit["param_names"]])

        ax.plot(t_curve, y_curve, linestyle='-', color=curve_colour, label=label)

        rate_param = fit["rate_param"]
        if rate_param == "T1":
            rate_param_ = r"$T_{1}$"
        elif rate_param == "T2":
            rate_param_ = r"$T_{2}$"
        else:
            rate_param_ = rate_param

        if annotate and rate_param:
            time_str = format_time_value(fit['params'][rate_param], fit['errors'][rate_param])
            anchor_x, anchor_y = _resolve_annotate_anchor(positions[i], t_curve, y_curve)
            ax.annotate(
                fr"{rate_param_} = {time_str}",
                xy=(anchor_x, anchor_y), xytext=offsets[i],
                textcoords='offset points', ha=ha_list[i], va=va_list[i],
                fontsize=0.8 * style['fontsize'], color=data_colour,
            )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if logplot:
        if logplot == 'x':
            ax.set_xscale('log')
            ax.set_yscale('linear')
        elif logplot == 'y':
            ax.set_yscale('log')
            ax.set_xscale('linear')
        elif logplot == 'both':
            ax.set_xscale('log')
            ax.set_yscale('log')
        else:
            raise ValueError(f"Please set option logplot to False, 'x', 'y', or 'both'. Currently: {logplot}")

    apply_spine_style(ax, top=show_top_spine, right=show_right_spine)
    apply_tick_style(ax, direction=tick_dir, length=tick_length, width=tick_width,
                      pad=tick_pad, num_ticks=tick_number, axis='both')
    ax.legend()

    fig.set_figheight(pl_height)
    fig.set_figwidth(pl_width)
    fig.tight_layout()

    if save_fig:
        save_figure(fig, fig_name=fig_name, save_format=save_format)

    return fig, ax


# ============================================================
# End-to-end relaxation workflow
# ============================================================

def NMR2D_relaxation_workflow(datapath, procno=1,
                               rows=None, columns=None, row_width=1, column_width=1,
                               row_shifts=None, column_shifts=None,
                               integration="peaks", peak_defs=None, regions=None,
                               peak_labels=None, region_labels=None,
                               fit_regions=None, baseline_regions=None, baseline_sigma=2.5,
                               baseline_max_iter=10, subtract_baseline=True,
                               normalise=False, positive_intensity=True,
                               allow_negative_intensity=False, sn_threshold=3,
                               noise_region=None, integration_method="trapezoid",
                               region_threshold=5.0, region_threshold_mode="noise",
                               model="inversion_recovery", vdlist=None, fixed=None,
                               use_errors=True, normalised_integrals=False,
                               show_series=None,
                               show_slices=False, show_fit=True, annotate=True,
                               show_report=True, show_warnings=True,
                               colour_spectra="coolwarm", offset_scale=1.05, xlims=None,
                               logplot=False,
                               colours=None, fit_colours=None, fit_shade=0.45,
                               annotate_position='end', annotate_offset=(5, 0),
                               annotate_ha='left', annotate_va='center',
                               font='Arial', fontsize=12, pl_width=3.375, pl_height=3.375,
                               tick_dir='in', tick_length=4.0, tick_width=1.0, tick_pad=4.0,
                               tick_number=None, show_top_spine=False, show_right_spine=False,
                               save_fig=False, fig_name='nmr_relaxation', save_format='png'):
    """
    High-level, end-to-end T1/T2/T1rho relaxation workflow for a
    Bruker pseudo-2D dataset: import once, extract slices, build an
    intensity series per peak or per region, fit a relaxation model to
    each, and optionally report and plot the result.

    Workflow: ``NMR2dimport -> NMR2D_get_slices ->
    [NMR2D_fit_slices | NMR2D_integrate_slices] ->
    NMR2D_fit_relaxation -> NMR2D_plot_relaxation``.

    Only one indirect-dimension direction (rows OR columns) is fitted
    per call -- a T1/T2/T1rho series lives along one dimension, and
    keeping this to a single direction keeps the return shape simple.
    Call it twice for a dataset with both.

    Parameters
    ----------
    datapath : str
        Bruker pseudo-2D experiment path.
    procno : int, optional
    rows, columns, row_shifts, column_shifts, row_width, column_width : optional
        Slice selection -- see :func:`nmrtools.two_d.NMR2D_get_slices`.
        Exactly one of the row-/column- pairs should be given (e.g.
        ``rows=list(range(1, n_slices + 1))`` for every row of a
        pseudo-2D series).
    integration : {'peaks', 'regions'}, optional
        ``'peaks'`` (default): fit ``peak_defs`` to each slice and use
        the fitted peak integrals (see
        :func:`nmrtools.fitting.NMR2D_fit_slices`).
        ``'regions'``: skip peak fitting and directly integrate each
        slice over the x-ranges in ``regions`` (see
        :func:`NMR2D_integrate_slices`) -- for when individual peaks
        aren't cleanly resolved enough to be worth modelling.
    peak_defs : list of dict
        Required if ``integration='peaks'``. See
        :func:`nmrtools.fitting.fit_spectrum`.
    regions : list of (float, float), or ``"auto"``
        Required if ``integration='regions'``, e.g.
        ``[[3.4, 4.5], [4.5, 8.0]]`` -- as many as you like. ``"auto"``
        finds them from a threshold over every slice
        (:func:`opennmr.processing.relaxometry.find_integration_regions`:
        ``region_threshold`` × the noise σ, or × each slice's maximum with
        ``region_threshold_mode="fraction"``); what was found is returned
        as ``"auto_regions"``.
    integration_method : {'trapezoid', 'simpson'}, optional
        The rule for ``integration='regions'`` (see
        :func:`NMR2D_integrate_slices`).
    peak_labels, region_labels : list of str, optional
        Series labels; whichever matches ``integration`` is used.
    fit_regions : list of tuple, optional
        Only used when ``integration='peaks'``.
    baseline_regions, baseline_sigma, baseline_max_iter : optional
        Baseline estimation, used by both integration modes.
    subtract_baseline : bool, optional
        Only used when ``integration='regions'`` -- see
        :func:`NMR2D_integrate_slices`.
    normalise : bool, optional
        Normalise each slice before fitting/integrating.
    positive_intensity, allow_negative_intensity, sn_threshold : optional
        Only used when ``integration='peaks'``.
    noise_region : tuple or list of tuples, optional
        Used for S/N-based amplitude gating (``integration='peaks'``)
        or integral error propagation (``integration='regions'``).
    model : str, optional
        See :data:`RELAXATION_MODELS`.
    vdlist : optional
        See :func:`NMR2D_fit_relaxation` / :func:`_get_delays`.
    fixed : dict, optional
        Parameters held fixed in the relaxation fit (e.g.
        ``{"C": 2.0}``), forwarded to :func:`fit_relaxation` for every
        series.
    use_errors : bool, optional
        Weight the relaxation fit by each slice's integral error.
    normalised_integrals : bool, optional
        Fit percentage-normalised integrals rather than raw ones.
    show_series : list, optional
        Which peak/region series to report and plot -- 0-based indices
        (matching ``peak_defs``/``regions`` order) or the labels in
        ``peak_labels``/``region_labels``. Every series is still
        fitted regardless; this only controls what's displayed. If
        None (default), every series is shown.
    show_slices : bool, optional
        Also plot the raw extracted slices (see
        :func:`nmrtools.two_d.NMR2D_plot_slices`) -- off by default,
        since the point of this workflow is the relaxation curve;
        turn on to sanity-check the extraction.
    show_fit : bool, optional
        Plot the fitted relaxation curve(s) (see
        :func:`NMR2D_plot_relaxation`).
    annotate : bool, optional
        When ``show_fit=True``, label each curve on the plot with its
        fitted rate constant, nicely formatted in s/ms/us/ns (see
        :func:`format_time_value`). Default True.
    show_report, show_warnings : bool, optional
    colour_spectra, offset_scale, xlims : optional
        Forwarded to :func:`nmrtools.two_d.NMR2D_plot_slices` when
        ``show_slices=True``.
    colours, fit_colours, fit_shade, annotate_position, annotate_offset,
    annotate_ha, annotate_va : optional
        Forwarded to :func:`NMR2D_plot_relaxation` when
        ``show_fit=True`` -- see that function for details. By default
        each series gets an automatically-cycled data colour with its
        fit curve drawn as a paler shade of the same colour.
    font, fontsize : optional
    save_fig, fig_name, save_format : optional
        See :func:`nmrtools.plotting.save_figure`. When both
        ``show_slices`` and ``show_fit`` are True and ``save_fig`` is
        True, the slices figure is saved as ``f"{fig_name}_slices"``.

    Returns
    -------
    dict
        ``"dataset"``, ``"slices"`` (raw extracted traces), ``"series"``
        (the peak-fit or region-integral result feeding the relaxation
        fit), ``"relaxation_fits"`` (output of
        :func:`NMR2D_fit_relaxation`, filtered to ``show_series`` if
        given), ``"figure"``, ``"axes"`` (relaxation plot, or None),
        ``"slices_figure"``, ``"slices_axes"`` (or None).
    """

    if integration not in ("peaks", "regions"):
        raise ValueError("integration must be 'peaks' or 'regions'")
    if integration == "peaks" and peak_defs is None:
        raise ValueError("peak_defs is required when integration='peaks'")
    if integration == "regions" and regions is None:
        raise ValueError("regions is required when integration='regions'")
    auto = None

    row_requested = rows is not None or row_shifts is not None
    col_requested = columns is not None or column_shifts is not None

    if row_requested and col_requested:
        raise ValueError(
            "NMR2D_relaxation_workflow fits one relaxation series "
            "(one direction) at a time -- pass rows/row_shifts OR "
            "columns/column_shifts, not both. Call it twice for a "
            "dataset needing both."
        )
    if not row_requested and not col_requested:
        raise ValueError("No rows or columns were requested.")

    mode = "rows" if row_requested else "columns"

    dataset = NMR2dimport(datapath, procno=procno)

    if mode == "rows":
        slices = NMR2D_get_slices(dataset, rows=rows, row_shifts=row_shifts,
                                   row_width=row_width, normalise=False)
    else:
        slices = NMR2D_get_slices(dataset, columns=columns, column_shifts=column_shifts,
                                   column_width=column_width, normalise=False)

    labels = peak_labels if integration == "peaks" else region_labels

    if integration == "peaks":
        series = NMR2D_fit_slices(
            slices, peak_defs=peak_defs, mode=mode,
            fit_regions=fit_regions, baseline_regions=baseline_regions,
            baseline_sigma=baseline_sigma, baseline_max_iter=baseline_max_iter,
            normalise=normalise, positive_intensity=positive_intensity,
            allow_negative_intensity=allow_negative_intensity,
            sn_threshold=sn_threshold, noise_region=noise_region,
            show_report=False, show_warnings=show_warnings,
        )[mode]
    else:
        if isinstance(regions, str):
            if regions != "auto":
                raise ValueError("regions must be a list of (lo, hi) or 'auto'.")
            traces = [prepare_trace_for_fit(t, normalise=normalise) for t in slices[mode]]
            auto = _relaxometry.find_integration_regions(
                traces[0]["x_ppm"], np.vstack([t["real_spectrum"] for t in traces]),
                threshold=region_threshold, mode=region_threshold_mode,
                subtract_baseline=subtract_baseline)
            if not auto["regions"]:
                raise ValueError(auto["notes"][0])
            regions = auto["regions"]
            if show_report:
                print("Integration regions: " + " ".join(auto["notes"]))
        series = NMR2D_integrate_slices(
            slices, regions=regions, mode=mode, region_labels=region_labels,
            normalise=normalise, subtract_baseline=subtract_baseline,
            baseline_regions=baseline_regions, baseline_sigma=baseline_sigma,
            baseline_max_iter=baseline_max_iter, noise_region=noise_region,
            method=integration_method,
        )[mode]

    relaxation_fits = NMR2D_fit_relaxation(
        series, model=model, vdlist=vdlist, peak_labels=labels,
        normalised=normalised_integrals, use_errors=use_errors,
        fixed=fixed, show_report=False,
    )

    relaxation_fits_shown = _select_series(relaxation_fits, show_series)

    if show_report:
        for fit in relaxation_fits_shown:
            print(f"\n--- {fit['peak_label']} ---")
            print_relaxation_report(fit)

    fig_slices = ax_slices = None
    if show_slices:
        fig_slices, ax_slices = NMR2D_plot_slices(
            slices, mode=mode, colour_spectra=colour_spectra,
            offset_scale=offset_scale, xlims=xlims, font=font, fontsize=fontsize,
        )
        if save_fig:
            save_figure(fig_slices, fig_name=f"{fig_name}_slices", save_format=save_format)

    fig = ax = None
    if show_fit:
        fig, ax = NMR2D_plot_relaxation(
            relaxation_fits_shown, annotate=annotate, font=font, fontsize=fontsize, logplot=logplot,
            colours=colours, fit_colours=fit_colours, fit_shade=fit_shade,
            annotate_position=annotate_position, annotate_offset=annotate_offset,
            annotate_ha=annotate_ha, annotate_va=annotate_va,
            pl_width=pl_width, pl_height=pl_height, tick_dir=tick_dir, tick_length=tick_length, 
            tick_width=tick_width, tick_pad=tick_pad, tick_number=tick_number, 
            show_top_spine=show_top_spine, show_right_spine=show_right_spine,
        )
        if save_fig:
            save_figure(fig, fig_name=fig_name, save_format=save_format)

    return {
        "dataset": dataset, "slices": slices, "series": series,
        "relaxation_fits": relaxation_fits_shown, "auto_regions": auto,
        "figure": fig, "axes": ax,
        "slices_figure": fig_slices, "slices_axes": ax_slices,
    }
