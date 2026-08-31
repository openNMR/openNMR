"""
Relaxation-curve fitting (T1, T2, T1rho) for opennmr.

Ported from ``nmrtools2``'s ``relaxation.py``. The plotting/workflow
functions there (``NMR2D_plot_relaxation``, ``NMR2D_relaxation_workflow``,
and their private plot-styling helpers) are deliberately left out here,
for the same reason as ``opennmr.processing.fitting``'s deferred
functions -- they're one-shot import+fit+plot+report convenience
wrappers, deferred to a future GUI fitting panel.

This module is deliberately decoupled from the peak-fitting machinery
in ``fitting.py``: it operates on plain ``(t, y[, y_err])`` arrays, so
it can fit a relaxation curve to *any* series of intensities/integrals
against delay time, not only ones produced by
:func:`opennmr.processing.fitting.NMR2D_fit_slices`. A convenience
function, :func:`NMR2D_fit_relaxation`, is provided for the common
case of fitting directly from that function's output (or from
:func:`NMR2D_integrate_slices`'s -- both are shaped the same way, so
either can be passed to :func:`NMR2D_fit_relaxation` interchangeably).

Models provided
----------------
- ``saturation_recovery`` / ``inversion_recovery``  (T1)
- ``exponential_decay``                              (T2 / T1rho)
- ``biexponential_recovery`` / ``biexponential_decay``
- ``stretched_recovery`` / ``stretched_decay``        (stretched/KWW,
  for distributions of relaxation times)

Workflow
--------

    opennmr.processing.slicing.get_slices
        -> [opennmr.processing.fitting.NMR2D_fit_slices | NMR2D_integrate_slices]
        -> NMR2D_fit_relaxation

Alternatively, fit an arbitrary (t, y) array directly with
:func:`fit_relaxation`.

This module never reads a delay list (``vdlist``) file itself --
:func:`NMR2D_fit_relaxation`/:func:`_get_delays` take an already-loaded
vdlist (the dict from :func:`opennmr.io.bruker.read_vdlist`, or a
plain array of delays in seconds), keeping processing/ modules IO-free.
"""

from __future__ import annotations

import re
from math import floor, log10

import numpy as np
from scipy.optimize import curve_fit

from .fitting import build_region_mask, estimate_baseline_robust, estimate_noise_level, extract_params_and_errors
from ..core.spectrum import NMRSpectrum

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


def stejskal_tanner_decay(b, I0, D):
    """Stejskal-Tanner PFG diffusion attenuation: ``I0*exp(-b*D)``.

    ``b`` : diffusion weighting (s/m^2) -- e.g. from
    :func:`opennmr.io.bruker.read_diff_xml`'s own ``"B"``, which
    already has gamma/delta/DELTA (and TopSpin's own ``DELTA-delta/3``
    correction) baked in; don't fold those in again here. Same
    functional shape as :func:`exponential_decay` with ``b``
    substituted for ``t`` and ``D`` for the rate constant -- fits via
    the same :func:`fit_relaxation` machinery, just relabelled.
    """
    return I0 * np.exp(-b * D)


def stejskal_tanner_decay_stretched(b, I0, D, beta):
    """Stretched Stejskal-Tanner decay: ``I0*exp(-(b*D)**beta)`` -- for
    a distribution of diffusion coefficients (e.g. a polydisperse
    sample) rather than a single value; ``beta=1`` recovers the plain
    exponential."""
    return I0 * np.exp(-(b * D) ** beta)


def stejskal_tanner_decay_biexponential(b, I0a, Da, I0b, Db):
    """Sum of two independent Stejskal-Tanner components -- for two
    genuinely distinct diffusing populations (e.g. bound/free water,
    or two chemically distinct species) resolved as two separate
    diffusion coefficients, rather than :func:`stejskal_tanner_decay_
    stretched`'s single smeared-out distribution. Same shape as
    :func:`biexponential_decay` with ``b`` substituted for ``t`` and
    ``D`` for the rate constant -- see :func:`stejskal_tanner_decay`'s
    own docstring for why that substitution is exact here."""
    return I0a * np.exp(-b * Da) + I0b * np.exp(-b * Db)


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


def _p0_diffusion(b, y):
    # D is orders of magnitude smaller than a relaxation time
    # constant, so _p0_mono's "median of the x-values" heuristic
    # (right for a delay series) would be wildly wrong here -- guess
    # instead from where a decay to ~1/e over the full b-range would
    # put D.
    I0 = np.max(np.abs(y)) if np.max(np.abs(y)) > 0 else 1.0
    b_max = np.max(b[b > 0]) if np.any(b > 0) else 1.0
    return [I0, 1.0 / b_max]


def _bounds_diffusion(b, y):
    return ([-np.inf, 1e-15], [np.inf, np.inf])


def _p0_diffusion_stretched(b, y):
    return _p0_diffusion(b, y) + [0.8]


def _bounds_diffusion_stretched(b, y):
    lo, hi = _bounds_diffusion(b, y)
    return (lo + [0.05], hi + [1.5])


def _p0_diffusion_biexp(b, y):
    # Same reasoning as _p0_diffusion (D needs a b-range-scaled guess,
    # not _p0_biexp_decay's own delay-median-based one) applied to two
    # components -- deliberately far apart (a 10x spread, not
    # _p0_biexp_decay's own narrower 0.3x/2x) so the fit starts from
    # two genuinely distinct components, and centred higher than a
    # first attempt at this guess (20/b_max, 2/b_max, not 5/b_max,
    # 0.5/b_max): tuned empirically against fit_relaxation's own actual
    # curve_fit call (bounds + x_scale="jac", not a bare unconstrained
    # fit) -- confirmed a materially different convergence outcome for
    # the exact same starting point once real (even very loose) bounds
    # are present, since that changes which scipy solver runs at all.
    # The "fast" guess in particular has to decay substantially within
    # the b-range actually sampled, not just mildly -- a guess that's
    # still nearly flat across the whole sampled range gives curve_fit
    # no curvature to distinguish it from a constant/offset early on,
    # which is enough to derail convergence even for well-separated
    # true values.
    I0 = np.max(np.abs(y)) if np.max(np.abs(y)) > 0 else 1.0
    b_max = np.max(b[b > 0]) if np.any(b > 0) else 1.0
    return [I0 * 0.5, 20.0 / b_max, I0 * 0.5, 2.0 / b_max]


def _bounds_diffusion_biexp(b, y):
    return ([-np.inf, 1e-15, -np.inf, 1e-15], [np.inf, np.inf, np.inf, np.inf])


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
    "stejskal_tanner": {
        "func": stejskal_tanner_decay, "params": ["I0", "D"],
        "p0": _p0_diffusion, "bounds": _bounds_diffusion, "rate_param": "D",
    },
    "stejskal_tanner_stretched": {
        "func": stejskal_tanner_decay_stretched, "params": ["I0", "D", "beta"],
        "p0": _p0_diffusion_stretched, "bounds": _bounds_diffusion_stretched, "rate_param": "D",
    },
    "stejskal_tanner_biexponential": {
        "func": stejskal_tanner_decay_biexponential, "params": ["I0a", "Da", "I0b", "Db"],
        "p0": _p0_diffusion_biexp, "bounds": _bounds_diffusion_biexp, "rate_param": None,
    },
}

def build_custom_model(equation: str, independent_var: str = "x",
                        p0_overrides: list[float] | tuple[float, ...] | None = None) -> dict:
    """Build a :data:`RELAXATION_MODELS`-shaped spec dict from a user-
    typed equation string (e.g. ``"A*exp(-x/T1)+C"``), for
    :func:`fit_relaxation`'s/:func:`NMR2D_fit_relaxation`'s own
    ``custom_model`` parameter -- an escape hatch for a functional form
    not already in :data:`RELAXATION_MODELS`, without needing a new
    named Python function (and p0/bounds helpers) for every one-off
    model a user wants to try.

    Parsed with sympy: every free symbol besides ``independent_var``
    becomes a fitted parameter, in the order it first appears in
    ``equation`` reading left to right -- also the order
    ``p0_overrides`` (if given) is read in, so what the user typed and
    what they set initial guesses for line up the same way.

    ``p0_overrides`` : initial guesses, positionally matching the
    detected parameter order. Any parameter beyond the given values
    (or all of them, if omitted entirely) defaults to 1.0 -- the same
    default ``scipy.optimize.curve_fit`` itself uses when no p0 is
    given at all. No bounds are inferred beyond a very wide (but
    finite -- literal +/-inf bounds push ``curve_fit`` onto a solver
    that doesn't support ``fit_relaxation``'s own ``x_scale="jac"``)
    range, since there's no general way to guess, e.g., that a symbol
    named ``T1`` should be positive from the equation alone.

    Raises
    ------
    ValueError
        If the equation doesn't parse, doesn't reference
        ``independent_var``, or has no free parameters to fit.
    """
    import sympy

    try:
        expr = sympy.sympify(equation, evaluate=True)
    except (sympy.SympifyError, TypeError, SyntaxError) as exc:
        raise ValueError(f"Could not parse equation '{equation}': {exc}") from exc

    x_symbol = sympy.Symbol(independent_var)
    if x_symbol not in expr.free_symbols:
        raise ValueError(
            f"Equation '{equation}' doesn't reference the independent variable '{independent_var}'."
        )

    seen: list[str] = []
    for token in re.findall(r"[A-Za-z_][A-Za-z_0-9]*", equation):
        symbol = sympy.Symbol(token)
        if symbol in expr.free_symbols and symbol != x_symbol and token not in seen:
            seen.append(token)
    param_names = seen

    if not param_names:
        raise ValueError(f"Equation '{equation}' has no free parameters to fit (only '{independent_var}').")

    param_symbols = [sympy.Symbol(name) for name in param_names]
    numeric_func = sympy.lambdify((x_symbol, *param_symbols), expr, modules="numpy")

    def func(t, *params):
        t = np.asarray(t, dtype=float)
        result = numeric_func(t, *params)
        # An equation that (unusually) doesn't actually vary with x --
        # e.g. a plain constant -- lambdifies to a bare scalar rather
        # than an array matching t; curve_fit needs the latter.
        return np.broadcast_to(result, t.shape).astype(float) if np.ndim(result) == 0 else np.asarray(result, dtype=float)

    overrides = list(p0_overrides) if p0_overrides else []
    p0_values = [overrides[i] if i < len(overrides) else 1.0 for i in range(len(param_names))]

    return {
        "func": func, "params": param_names,
        "p0": lambda t, y: list(p0_values),
        "bounds": lambda t, y: ([-1.0e12] * len(param_names), [1.0e12] * len(param_names)),
        "rate_param": None,
    }


# Convenience aliases: T1rho experiments use the same functional forms
# as T2 (decay) or T1 (recovery) depending on the pulse sequence used;
# no separate model is needed, just fit with the appropriate decay/
# recovery model and relabel the rate constant "T1rho" yourself.
#
# PFG diffusion (stejskal_tanner*) reuses this same registry/fitting
# machinery too -- despite the module name, nothing here is actually
# T1/T2-specific; it's a generic decay/recovery curve fitter keyed by
# whatever independent variable the caller passes as `t` (a delay in
# seconds for relaxation, a b-value in s/m^2 for diffusion).


# ============================================================
# Fitting
# ============================================================

def fit_relaxation(t, y, model: str = "inversion_recovery", y_err=None, p0=None,
                    bounds=None, fixed=None, maxfev=20000, custom_model: dict | None = None) -> dict:
    """
    Fit a relaxation curve ``y(t)`` with a named model from
    :data:`RELAXATION_MODELS`, or an arbitrary user-supplied one.

    Parameters
    ----------
    t, y : array-like
        Delay (seconds) and intensity/integral at each delay.

    model : str
        One of :data:`RELAXATION_MODELS`. Ignored (only used as the
        result's own ``"model"`` label) when ``custom_model`` is given.

    custom_model : dict, optional
        A :data:`RELAXATION_MODELS`-shaped spec (``func``, ``params``,
        ``p0``, ``bounds``, ``rate_param``) to fit instead of looking
        ``model`` up in :data:`RELAXATION_MODELS` -- see
        :func:`build_custom_model` for building one from a typed
        equation string.

    y_err : array-like, optional
        Per-point uncertainty (e.g. ``integral_errors`` from
        :func:`opennmr.processing.fitting.NMR2D_fit_slices`). If
        given, used as ``sigma`` in the least-squares fit (points are
        weighted by 1/error^2) and reported back as parameter
        uncertainties that account for the measurement errors rather
        than only the fit residual scatter.

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

    if custom_model is not None:
        spec = custom_model
    elif model not in RELAXATION_MODELS:
        raise ValueError(
            f"Unknown relaxation model '{model}'. Available: {sorted(RELAXATION_MODELS)}"
        )
    else:
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

    # x_scale="jac": without it, curve_fit's default parameter scaling
    # (fine for e.g. M0~1-10 alongside T1/T2~0.01-10 s) badly
    # under-serves a diffusion coefficient D~1e-9 to 1e-13 sitting next
    # to I0~1-10 -- found via a Stejskal-Tanner fit reporting bogus
    # xtol "convergence" after 11 function evals, off by 30%+ from the
    # true (synthetic, known) D. Auto-scaling from the Jacobian fixes
    # that case and is a no-op for already-well-scaled models.
    popt, pcov = curve_fit(
        wrapper, t, y, p0=p0_free, bounds=(lo_free, hi_free),
        sigma=sigma, absolute_sigma=sigma is not None, maxfev=maxfev, x_scale="jac",
    )

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

    return {
        "model": model, "param_names": param_names, "rate_param": spec["rate_param"],
        "params": params, "errors": errors,
        "popt": popt, "pcov": pcov, "free_names": free_names,
        "t": t, "y": y, "y_err": y_err, "y_fit": y_fit,
        "fit_quality": {"rmse": rmse, "chi2": chi2, "reduced_chi2": red_chi2},
    }


def fit_curve_uncertainty(fit_result: dict, n_points: int = 200, sigma: float = 3.0):
    """A smooth model curve over ``fit_result``'s own delay range, plus
    a linearized (delta-method) uncertainty band at ``sigma`` standard
    deviations -- the same idea as ``lmfit``'s ``eval_uncertainty``
    (which the user's own PFG diffusion-fitting code uses, shaded via
    ``ax.fill`` around the fit curve), hand-rolled here since this
    module fits with plain ``scipy.curve_fit`` rather than ``lmfit``.

    The band is propagated from :func:`fit_relaxation`'s own
    ``pcov`` (parameter covariance) via a numerically-estimated
    Jacobian of the model curve with respect to each free parameter,
    evaluated on a fine grid -- standard linear error propagation
    (var(f) = J @ pcov @ J.T), valid near the fitted optimum the same
    way the covariance-derived parameter errors already reported by
    :func:`fit_relaxation` are.

    Returns ``(t_fine, y_fine, y_band)`` -- plot the shaded region as
    ``y_fine - y_band`` to ``y_fine + y_band``.
    """
    spec = RELAXATION_MODELS[fit_result["model"]]
    func, param_names = spec["func"], spec["params"]
    free_names = fit_result["free_names"]
    params = fit_result["params"]
    popt = np.asarray(fit_result["popt"], dtype=float)
    pcov = np.asarray(fit_result["pcov"], dtype=float)
    t = fit_result["t"]

    t_fine = np.linspace(float(np.min(t)), float(np.max(t)), n_points)

    def curve(free_vals):
        full = dict(params)
        full.update(zip(free_names, free_vals))
        return func(t_fine, *[full[p] for p in param_names])

    y_fine = curve(popt)

    n_free = len(popt)
    jac = np.zeros((n_points, n_free))
    for i in range(n_free):
        step = 1e-6 * max(abs(popt[i]), 1e-12)
        p_plus, p_minus = popt.copy(), popt.copy()
        p_plus[i] += step
        p_minus[i] -= step
        jac[:, i] = (curve(p_plus) - curve(p_minus)) / (2 * step)

    variance = np.einsum("ij,jk,ik->i", jac, pcov, jac)
    y_band = sigma * np.sqrt(np.clip(variance, 0.0, None))

    return t_fine, y_fine, y_band


def format_time_value(value, error=None, sig_figs=3):
    """
    Format a time value (assumed given in seconds) with an
    automatically chosen unit (s, ms, us, or ns) for display, e.g.
    ``format_time_value(0.00152, 0.00003)`` -> ``"1.52 ± 0.03 ms"``.

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
        unit, factor = "µs", 1e6
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
        return f"{val_str} ± {err_str} {unit}"

    return f"{val_str} {unit}"


def format_diffusion_value(value, error=None, sig_figs=3):
    """
    Format a diffusion coefficient (m^2/s) for display, e.g.
    ``format_diffusion_value(2.03e-9, 0.05e-9)`` ->
    ``"2.03e-09 ± 0.05e-09 m^2/s"``.

    Diffusion coefficients don't have a clean SI-prefix convention the
    way times do (ns/us/ms/s span a small, well-known set) -- typical
    values range from ~1e-8 (small molecules, low viscosity) to
    ~1e-13 m^2/s (large aggregates/restricted diffusion), so this
    stays in scientific notation (mantissa/exponent) rather than
    guessing at a unit prefix.
    """
    if value is None or not np.isfinite(value):
        return "n/a"

    exponent = int(floor(log10(abs(value)))) if value != 0 else 0
    decimals = max(sig_figs - 1, 0)
    val_str = f"{value / 10 ** exponent:.{decimals}f}e{exponent:+03d}"

    if error is not None and np.isfinite(error):
        err_str = f"{error / 10 ** exponent:.{decimals}f}e{exponent:+03d}"
        return f"{val_str} ± {err_str} m^2/s"

    return f"{val_str} m^2/s"


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
            print(f"  {name:<8}= {val:.6g} ± {err:.4g}")

    rate_param = fit["rate_param"]
    if rate_param:
        print(f"\n  {rate_param} = {format_time_value(fit['params'][rate_param], fit['errors'][rate_param])}")

    mq = fit["fit_quality"]
    print(f"\n  RMSE          = {mq['rmse']:.4g}")
    print(f"  Reduced chi^2 = {mq['reduced_chi2']:.4g}")
    print("=" * 50 + "\n")


# ============================================================
# Region-based integration (peak-fitting-free alternative)
# ============================================================

def NMR2D_integrate_slices(slices: dict[str, list[NMRSpectrum]], regions, mode: str = "rows", region_labels=None,
                            normalize=False, x_unit='ppm',
                            subtract_baseline=True, baseline_regions=None,
                            baseline_sigma=2.5, baseline_max_iter=10,
                            noise_region=None) -> dict:
    """
    Integrate one or more extracted 2D NMR slices (rows and/or
    columns) over user-specified x-axis regions -- a peak-fitting-free
    alternative to :func:`opennmr.processing.fitting.NMR2D_fit_slices`,
    useful when individual peaks are too poorly resolved or
    overlapping to be worth modelling, and a bulk-region integral
    (e.g. an aromatic region ``[6.5, 8.0]`` vs an aliphatic region
    ``[0.5, 3.0]``) is all that's needed for a relaxation curve.

    Parameters
    ----------
    slices : dict
        Output of :func:`opennmr.processing.slicing.get_slices`:
        ``{"rows": [NMRSpectrum, ...], "columns": [NMRSpectrum, ...]}``.
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
    normalize : bool, optional
        Normalise each trace (to its own max) before integrating.
    subtract_baseline : bool, optional
        Estimate and subtract a robust linear baseline (see
        :func:`opennmr.processing.fitting.estimate_baseline_robust`)
        before integrating each region. Recommended: an unsubtracted
        baseline offset biases every region's integral by an amount
        that depends on the region's width, which then propagates
        directly into a distorted relaxation curve. Default True.
    baseline_regions, baseline_sigma, baseline_max_iter : optional
        See :func:`opennmr.processing.fitting.estimate_baseline_robust`.
    noise_region : tuple or list of tuples, optional
        Region(s) used to estimate each slice's noise level for
        integral error propagation (see
        :func:`opennmr.processing.fitting.estimate_noise_level`). If
        None, the first/last 5% of each slice is used. Errors are
        propagated analytically through the trapezoidal rule (exact
        for independent per-point Gaussian noise), not just assumed
        constant across regions of different widths.

    Returns
    -------
    dict with keys ``"rows"`` and ``"columns"``, shaped closely enough
    to match :func:`opennmr.processing.fitting.NMR2D_fit_slices`'s
    output to be used interchangeably with
    :func:`NMR2D_fit_relaxation`: each entry has ``slice`` (the trace
    ``NMRSpectrum`` itself), ``slice_type``, ``index``, ``shift``,
    ``x``, ``y``, ``integrals``, ``integral_errors``,
    ``integrals_norm``, ``regions``. There is no fit, so
    ``popt``/``pcov``/``components`` are absent.
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

    for slice_type, trace in selected:

        x = trace.axis_values(0, unit=x_unit.lower())
        y = trace.normalized() if normalize else trace.data

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

            # x_ppm (and x_Hz) run in descending order by opennmr's
            # Bruker convention (see opennmr.io.bruker's
            # _ppm_hz_axis); sort ascending first so the integral's
            # sign reflects the data, not the axis direction (np.trapz
            # flips sign under a descending x otherwise).
            sort_idx = np.argsort(x_r)
            x_r, y_r = x_r[sort_idx], y_r[sort_idx]

            # NOT abs()'d, unlike the peak-fit route's integrals: a raw
            # region integral in a signed series (e.g. an inversion-
            # recovery slice before the null point) is genuinely
            # negative, and abs() here would destroy exactly the sign
            # information the relaxation fit needs.
            integral = _trapz(y_r, x_r)

            # Trapezoidal-rule error propagation: I = sum(y_i * w_i),
            # so Var(I) = noise^2 * sum(w_i^2) for i.i.d. Gaussian
            # per-point noise. Exact for the trapezoidal weights
            # (works for non-uniform spacing too).
            w = np.empty_like(x_r)
            w[1:-1] = 0.5 * np.abs(x_r[2:] - x_r[:-2])
            w[0] = 0.5 * np.abs(x_r[1] - x_r[0])
            w[-1] = 0.5 * np.abs(x_r[-1] - x_r[-2])
            error = noise * np.sqrt(np.sum(w ** 2))

            integrals.append(integral)
            errors.append(error)

        total = np.sum(integrals)
        integrals_norm = [100 * a / total for a in integrals] if total > 0 else [np.nan] * len(integrals)

        slice_info = trace.metadata.raw["slice"]
        integrated_result = {
            "slice": trace, "slice_type": slice_type,
            "index": slice_info["index"], "shift": slice_info["shift"],
            "x": x, "y": y_corr,
            "integrals": integrals, "integral_errors": errors, "integrals_norm": integrals_norm,
            "regions": list(regions), "region_labels": region_labels,
        }

        integrated["rows" if slice_type == "row" else "columns"].append(integrated_result)

    return integrated


# ============================================================
# Convenience: fit directly from NMR2D_fit_slices/NMR2D_integrate_slices output
# ============================================================

def _get_delays(fits, vdlist):
    """
    Resolve delay times (seconds), sorted with the fits, for a
    ``fitted["rows"]``/``fitted["columns"]``-shaped list from
    :func:`opennmr.processing.fitting.NMR2D_fit_slices` or
    :func:`NMR2D_integrate_slices`.

    ``vdlist`` : the dict returned by
    :func:`opennmr.io.bruker.read_vdlist` (its ``"seconds"`` key is
    used), or a plain array of delays in seconds indexed by 1-based F1
    slice index. Unlike ``nmrtools2``, this never reads a vdlist file
    itself -- read it once with
    :func:`opennmr.io.bruker.read_vdlist` and pass the result in, so
    this processing module stays IO-free.
    """

    if vdlist is None:
        raise ValueError(
            "vdlist is required -- pass the dict from "
            "opennmr.io.bruker.read_vdlist(), or an array of delays "
            "(seconds) indexed by 1-based F1 slice index."
        )

    if isinstance(vdlist, dict):
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


def NMR2D_fit_relaxation(fits, model: str = "inversion_recovery", vdlist=None,
                          peak_labels=None, normalised=False, value_source="integral",
                          use_errors=True, fixed=None, show_report=True,
                          custom_model: dict | None = None) -> list:
    """
    Fit relaxation curves to every peak's/region's integral series
    across a :func:`opennmr.processing.fitting.NMR2D_fit_slices` (or
    :func:`NMR2D_integrate_slices`) output (T1/T2/T1rho pseudo-2D
    series).

    Parameters
    ----------
    fits : list
        ``fitted["rows"]`` or ``fitted["columns"]`` from
        :func:`opennmr.processing.fitting.NMR2D_fit_slices`, or the
        equivalent from :func:`NMR2D_integrate_slices`. Every entry
        must have been fitted/integrated against the *same*
        ``peak_defs``/``regions``, in the same order, or the per-peak
        series below are meaningless.
    model : str
        See :data:`RELAXATION_MODELS`.
    vdlist : dict or array-like
        The dict from :func:`opennmr.io.bruker.read_vdlist`, or a
        plain array of delays (seconds) indexed by 1-based F1 slice
        index. Required -- see :func:`_get_delays`.
    peak_labels : list of str, optional
    normalised : bool, optional
        Only used when ``value_source="integral"``: fit
        ``integrals_norm`` (percentage) rather than raw integrals. Not
        recommended for absolute T1/T2 amplitude interpretation, but
        harmless for the relaxation *time* itself.
    value_source : {'integral', 'amplitude'}, optional
        ``'integral'`` (default): each peak's integrated area (see
        ``normalised``) -- more robust to per-slice lineshape/linewidth
        changes than a single point estimate, since it sums over the
        whole peak rather than reading one point off it.
        ``'amplitude'``: the peak's own fitted pseudo-Voigt/Gaussian/
        Lorentzian amplitude instead -- simpler "peak height" reading,
        but more sensitive to a linewidth change across slices being
        mistaken for an intensity change.
    use_errors : bool, optional
        Weight the fit by each slice's per-peak uncertainty when
        available (``integral_errors`` for ``value_source="integral"``,
        the fitted amplitude's own covariance-derived error for
        ``"amplitude"`` -- see ``y_err`` in :func:`fit_relaxation`).
    fixed : dict, optional
        Forwarded to :func:`fit_relaxation` for every peak.
    show_report : bool, optional
        Print a report for each fitted peak.
    custom_model : dict, optional
        Forwarded to :func:`fit_relaxation` for every peak -- see its
        own ``custom_model`` parameter and :func:`build_custom_model`.

    Returns
    -------
    list of dict
        One :func:`fit_relaxation` result per peak, with an added
        ``"peak_label"`` key.
    """

    if value_source not in ("integral", "amplitude"):
        raise ValueError("value_source must be 'integral' or 'amplitude'")

    if not fits:
        raise ValueError("No fitted slices available.")

    t = _get_delays(fits, vdlist)
    order = np.argsort(t)

    if value_source == "amplitude":
        per_fit_amplitudes = []
        per_fit_amplitude_errors = []
        for fit in fits:
            per_peak = extract_params_and_errors(fit["popt"], fit["pcov"], fit["peak_defs"], fit["is_free"])
            amp_indices = [p["names"].index("amplitude") for p in per_peak]
            per_fit_amplitudes.append([p["params"][i] for p, i in zip(per_peak, amp_indices)])
            per_fit_amplitude_errors.append([p["errors"][i] for p, i in zip(per_peak, amp_indices)])
        n_peaks = max(len(v) for v in per_fit_amplitudes)
    else:
        source = "integrals_norm" if normalised else "integrals"
        n_peaks = max(len(fit.get(source, [])) for fit in fits)

    results = []

    for peak_i in range(n_peaks):

        if value_source == "amplitude":
            y = np.array([vals[peak_i] if len(vals) > peak_i else np.nan
                          for vals in per_fit_amplitudes])[order]
        else:
            y = np.array([fit[source][peak_i] if len(fit.get(source, [])) > peak_i
                          else np.nan for fit in fits])[order]
        t_peak = t[order]

        y_err = None
        if use_errors:
            if value_source == "amplitude":
                errs = [errs_i[peak_i] if len(errs_i) > peak_i else None
                        for errs_i in per_fit_amplitude_errors]
            else:
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
                                     fixed=fixed, custom_model=custom_model)

        label = (peak_labels[peak_i] if peak_labels is not None and peak_i < len(peak_labels)
                 else f"Peak {peak_i + 1}")
        fit_result["peak_label"] = label
        results.append(fit_result)

        if show_report:
            print(f"\n--- {label} ---")
            print_relaxation_report(fit_result)

    return results
