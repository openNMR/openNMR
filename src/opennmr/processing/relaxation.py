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


#: A PFG amplitude is a peak intensity, and an intensity cannot be
#: negative. Leaving ``I0`` free below zero let a Stejskal-Tanner fit
#: settle on a mirror-image decay -- a negative amplitude with a
#: perfectly plausible D -- whenever the series was noisy, badly
#: phased or over-attenuated at high b. That fit reports a diffusion
#: coefficient that looks fine in isolation and is meaningless.
#: Explicit user request: "for PFG there should be no possibility of
#: negative intensities".
_I0_MIN = 0.0


def _bounds_diffusion(b, y):
    return ([_I0_MIN, 1e-15], [np.inf, np.inf])


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
    # Both component amplitudes, for the same reason as _I0_MIN: a
    # two-component fit is *more* prone to this, not less -- it will
    # happily describe a monoexponential decay as a large positive
    # component plus a small negative one.
    return ([_I0_MIN, 1e-15, _I0_MIN, 1e-15], [np.inf, np.inf, np.inf, np.inf])


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
        "log_params": ("D",),
    },
    "stejskal_tanner_stretched": {
        "func": stejskal_tanner_decay_stretched, "params": ["I0", "D", "beta"],
        "p0": _p0_diffusion_stretched, "bounds": _bounds_diffusion_stretched, "rate_param": "D",
        "log_params": ("D",),
    },
    "stejskal_tanner_biexponential": {
        "func": stejskal_tanner_decay_biexponential, "params": ["I0a", "Da", "I0b", "Db"],
        "p0": _p0_diffusion_biexp, "bounds": _bounds_diffusion_biexp, "rate_param": None,
        "log_params": ("Da", "Db"),
    },
}

def evaluate_expression(expression: str, constants: dict | None = None) -> dict:
    """Evaluate a **fully-determined** numeric expression over named
    acquisition parameters -- ``"TAU*2"``, ``"d1"``, ``"p90*1e-6"``,
    ``"tau*ne"``.

    The sibling of :func:`build_custom_model` for the case where the
    answer must be a *number*, not a curve: the Values-source factor
    for a list file, where the user's request was "it would be nice if
    the list file x factor had an option to use an experimental
    parameter (or a combination) -- for X-Pulse I might want the TAU
    value to multiply with the arrayed number of echoes to fit the T2".

    The same sympy parsing and the same ``constants`` dict
    (:func:`opennmr.core.acquisition.acquisition_constants`) as a
    custom fitting equation, so a name means the same thing in both
    places. The one difference is the failure mode, and it is
    deliberate: an unknown symbol here is an **error naming it**, not a
    free parameter. There is nothing to fit a factor against, so a
    mistyped ``TAO`` must not quietly become anything at all.

    ``constants`` values are used **exactly as the vendor recorded
    them** -- an X-Pulse ``TAU`` in microseconds stays in microseconds.
    Converting is the expression's job (``TAU*1e-6``), because only the
    person reading the pulse programme knows which unit it was written
    in.

    Returns ``{"value", "used"}`` -- the number, and the constants that
    actually went into it (so the GUI can show *which* parameters a
    factor picked up, the same way a custom equation does).

    Raises
    ------
    ValueError
        If the expression doesn't parse, names something the
        experiment doesn't record, or doesn't reduce to a finite real
        number.
    """
    import sympy

    text = (expression or "").strip()
    if not text:
        raise ValueError("Enter a factor -- a number, or an expression in this experiment's parameters.")
    try:
        expr = sympy.sympify(text, evaluate=True)
    except (sympy.SympifyError, TypeError, SyntaxError) as exc:
        raise ValueError(f"Could not parse factor '{text}': {exc}") from exc

    substitutions = {}
    for name, value in (constants or {}).items():
        if value is None:
            continue
        symbol = sympy.Symbol(str(name))
        if symbol in expr.free_symbols:
            substitutions[symbol] = float(value)
    if substitutions:
        expr = expr.subs(substitutions)

    unknown = sorted(str(symbol) for symbol in expr.free_symbols)
    if unknown:
        # Named, and the available names offered alongside: "TAU is not
        # recorded" is only half an answer when the file spells it
        # "Tau" or the user is on a vendor that records nothing at all.
        available = sorted(constants or {})
        hint = (f" This experiment records: {', '.join(available[:20])}"
                + (", …" if len(available) > 20 else "") + ".") if available else (
            " This experiment has no recorded acquisition parameters to use.")
        raise ValueError(
            f"Factor '{text}' uses {', '.join(unknown)}, which this experiment does not record.{hint}"
        )

    try:
        value = float(expr)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Factor '{text}' did not reduce to a number.") from exc
    if not np.isfinite(value):
        raise ValueError(f"Factor '{text}' evaluated to {value}.")
    return {"value": value,
            "used": {str(symbol): float(val) for symbol, val in substitutions.items()}}


def build_custom_model(equation: str, independent_var: str = "x",
                        p0_overrides: list[float] | tuple[float, ...] | None = None,
                        constants: dict | None = None) -> dict:
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

    ``constants`` : named values substituted into the equation before
    anything is fitted, so a symbol that is *known* doesn't become a
    free parameter. Explicit user request: "it would be good if I can
    make fitting equations using values from the input data (e.g.
    delays and constants and L or whatever)" -- so an equation can be
    written as ``A*exp(-x*d1/T2)`` with ``d1`` taken from the
    acquisition rather than typed in as a magic number, or fitted for
    when nobody supplies it.

    Only symbols the equation actually uses are substituted, and a
    constant never shadows ``independent_var``. Names not present in
    the equation are ignored, which matters because the caller
    generally passes *every* acquisition parameter it has rather than
    guessing which ones a given equation wants.

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

    # Substituted before the free symbols are counted, so a known value
    # is a number in the expression rather than something to fit. The
    # independent variable is never substitutable, even if a vendor
    # happens to record a parameter of the same name.
    substitutions = {}
    if constants:
        for name, value in constants.items():
            if name == independent_var or value is None:
                continue
            symbol = sympy.Symbol(str(name))
            if symbol in expr.free_symbols:
                substitutions[symbol] = float(value)
    if substitutions:
        expr = expr.subs(substitutions)
    substituted_names = {str(symbol) for symbol in substitutions}

    seen: list[str] = []
    for token in re.findall(r"[A-Za-z_][A-Za-z_0-9]*", equation):
        symbol = sympy.Symbol(token)
        if symbol in expr.free_symbols and symbol != x_symbol and token not in seen:
            seen.append(token)
    param_names = seen

    if not param_names:
        known = f" (only '{independent_var}'"
        if substituted_names:
            known += " and the supplied " + ", ".join(sorted(substituted_names))
        raise ValueError(f"Equation '{equation}' has no free parameters to fit{known}).")

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
        "func": func, "params": param_names, "constants": dict(
            (name, float(value)) for name, value in
            ((str(sym), val) for sym, val in substitutions.items())
        ),
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

#: What a reported "+/- x" from a least-squares fit actually *is*.
#: There are two genuinely different quantities here and they answer
#: different questions, so every result carries which one it holds
#: (see :func:`describe_uncertainty`) and every report says so.
#:
#: ``measured``  -- per-point measurement errors were supplied, so
#:   scipy is run with ``absolute_sigma=True`` and the covariance is
#:   on the scale those errors set. This is the uncertainty **on the
#:   parameter given the known noise**: it answers "how well does my
#:   data pin this number down", and it does not shrink just because
#:   the model happens to fit tidily.
#:
#: ``scatter``   -- no measurement errors were supplied, so scipy is
#:   run with ``absolute_sigma=False`` and rescales the covariance by
#:   the reduced chi-squared. This is the uncertainty **assuming the
#:   model is correct** and attributing all residual scatter to noise:
#:   it answers "how tightly is this number determined by the shape of
#:   the curve", and it is optimistic whenever the model is wrong,
#:   because a systematic misfit is absorbed as if it were noise.
#:
#: The distinction is not pedantic. The same dataset can report a D
#: with a 2% "error" one way and 8% the other, and only one of those
#: is a claim about the measurement.
UNCERTAINTY_KINDS = {
    "measured": {
        "short": "1σ (from supplied point errors)",
        "long": (
            "1σ, propagated from the per-point measurement errors supplied with the data "
            "(absolute_sigma=True). Independent of how well the model happens to fit."
        ),
    },
    "scatter": {
        "short": "1σ (from residual scatter)",
        "long": (
            "1σ, from the scatter of the data about the fitted curve, rescaled by the reduced "
            "chi-squared (absolute_sigma=False). Assumes the model is correct -- any systematic "
            "misfit is absorbed into this number as if it were noise, making it optimistic."
        ),
    },
}


#: Parameters that a model spec may ask to be **searched in log10
#: space**, listed in its own ``log_params`` key.
#:
#: A diffusion coefficient is ~1e-9 sitting in the same parameter
#: vector as an amplitude of ~1, and the interesting range spans five
#: decades. Searched linearly, that is close to hopeless: measured
#: directly on noise-free two-component synthetic data with the
#: starting guess *already at the true values*, a linear search
#: converged to D = 5.5e-09 / 6.2e-10 against a truth of 3e-09 /
#: 2e-10 (R² 0.93), and moving the guess a decade away made it worse.
#: The same fits in log10 space recovered 3.000e-09 / 2.000e-10
#: exactly, from every starting point tried, and a three-component fit
#: recovered all three components to 6 significant figures.
#:
#: Nothing outside :func:`fit_relaxation`'s own curve_fit call sees
#: this: guesses, bounds, reported values, errors and covariance are
#: all in natural units, with the covariance pushed back through the
#: transform's Jacobian (see :func:`_undo_log_space`). It is a change
#: of search coordinates, not a change of what is reported.
LOG_FITTED_PARAMS = ("D", "Da", "Db", "D1", "D2", "D3", "D4", "D5", "D6")

#: Smallest value a log-space lower bound is allowed to reach. A
#: bound of 0 (or -inf) has no logarithm, and a diffusion coefficient
#: below 1e-20 m^2/s is not a physical answer anyone is looking for --
#: it is the fit having collapsed a component to nothing.
_LOG_FLOOR = 1e-20


def _to_log_space(values, log_indices, what: str, floor: bool = False):
    """``values`` with the entries at ``log_indices`` replaced by their
    base-10 logarithms. ``floor``: clamp a non-positive lower bound up
    to :data:`_LOG_FLOOR` rather than raising, since ``0`` is the
    natural way to write "no lower bound" for a positive quantity."""
    out = list(values)
    for i in log_indices:
        value = float(out[i])
        if value <= 0:
            if floor:
                value = _LOG_FLOOR
            else:
                raise ValueError(
                    f"{what} for a log-fitted parameter must be positive, got {value:g}."
                )
        out[i] = np.log10(value) if np.isfinite(value) else np.inf
    return out


def _from_log_space(values, log_indices):
    """Inverse of :func:`_to_log_space` for a single parameter vector."""
    out = list(values)
    for i in log_indices:
        out[i] = 10.0 ** out[i]
    return out


def _undo_log_space(popt, pcov, log_indices):
    """Take a fitted vector and its covariance back to natural units.

    The values are just ``10**x``. The covariance needs the Jacobian
    of that map -- ``d(10**x)/dx = 10**x * ln(10)`` -- applied as
    ``J @ pcov @ J.T`` with ``J`` diagonal, which is ordinary linear
    error propagation and exactly what makes the reported ± come out
    in m^2/s rather than in decades."""
    if not log_indices:
        return np.asarray(popt, dtype=float), np.asarray(pcov, dtype=float)
    popt = np.asarray(popt, dtype=float).copy()
    jacobian = np.ones_like(popt)
    for i in log_indices:
        popt[i] = 10.0 ** popt[i]
        jacobian[i] = popt[i] * np.log(10.0)
    pcov = np.asarray(pcov, dtype=float) * np.outer(jacobian, jacobian)
    return popt, pcov


def describe_uncertainty(from_measurement_errors: bool, dof: int) -> dict:
    """The ``uncertainty`` block every :func:`fit_relaxation` result
    carries: which of :data:`UNCERTAINTY_KINDS` its ``errors`` are,
    spelled out, plus the degrees of freedom they were computed on."""
    kind = "measured" if from_measurement_errors else "scatter"
    return {
        "kind": kind,
        "sigma_level": 1.0,
        "from_measurement_errors": bool(from_measurement_errors),
        "dof": int(dof),
        "short": UNCERTAINTY_KINDS[kind]["short"],
        "description": UNCERTAINTY_KINDS[kind]["long"],
    }



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

    # Parameters the spec asks to be *searched* in log10 space (see
    # LOG_FITTED_PARAMS). Everything outside this block -- p0, bounds,
    # the returned params/errors, the report -- stays in natural units;
    # only the numbers handed to curve_fit are transformed.
    log_free = [i for i, name in enumerate(free_names) if name in set(spec.get("log_params", ()))]
    p0_free = _to_log_space(p0_free, log_free, "initial guess")
    lo_free = _to_log_space(lo_free, log_free, "lower bound", floor=True)
    hi_free = _to_log_space(hi_free, log_free, "upper bound")

    def wrapper(t, *free_vals):
        full = dict(fixed)
        full.update(zip(free_names, _from_log_space(free_vals, log_free)))
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

    popt, pcov = _undo_log_space(popt, pcov, log_free)
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
    dof = max(len(y) - len(free_names), 1)
    red_chi2 = chi2 / dof
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - float(np.sum(residual ** 2)) / ss_tot if ss_tot > 0 else np.nan

    return {
        "model": model, "param_names": param_names, "rate_param": spec["rate_param"],
        "params": params, "errors": errors,
        "popt": popt, "pcov": pcov, "free_names": free_names,
        "fixed": dict(fixed), "p0": list(p0_full), "bounds": (list(lo_full), list(hi_full)),
        # The spec itself, not just its name: a custom_model (a typed
        # equation, or build_diffusion_model's N-component spec) exists
        # nowhere in RELAXATION_MODELS, so anything wanting to
        # re-evaluate the curve later -- fit_curve_uncertainty, the
        # report, a refit after excluding a point -- cannot get at the
        # function by looking the name up. Carrying it costs nothing
        # and removes a whole class of "works for named models only".
        "spec": spec,
        "t": t, "y": y, "y_err": y_err, "y_fit": y_fit,
        "uncertainty": describe_uncertainty(sigma is not None, dof),
        "fit_quality": {"rmse": rmse, "chi2": chi2, "reduced_chi2": red_chi2,
                         "r_squared": r_squared, "dof": dof, "n_points": int(len(y)),
                         "decay_fraction": decay_fraction(y)},
    }


def decay_fraction(y) -> float:
    """How much signal is left at the end of a series -- the last
    measured point over the first, ``I_final / I_0``.

    **A fit-quality number that R² cannot give you.** A diffusion or
    relaxation series that only decayed to 80% of its starting
    intensity constrains its rate poorly however tidily a curve passes
    through the points -- the fit is extrapolating past the data. The
    conventional target for a PFG series is to reach roughly 10-20% or
    below, and this is the number that says whether it did.

    **Measured, not fitted**: it states what the experiment actually
    achieved. Taking it off the fitted curve would report what the
    model claims the attenuation was, which is the thing being judged.

    Assumes ``y`` is ordered by the independent variable, which
    :func:`fit_relaxation` guarantees (it sorts by ``t``). Returns NaN
    for an empty series or a zero first point -- an undefined ratio,
    reported as such rather than as a misleading infinity.

    Sign is kept. An inversion recovery starts negative and ends
    positive, so a negative fraction there says the inversion worked;
    forcing it positive would throw that away.
    """
    y = np.asarray(y, dtype=float)
    if y.size < 2 or y[0] == 0 or not np.isfinite(y[0]):
        return float("nan")
    return float(y[-1] / y[0])


def fit_relaxation_global(datasets, model: str = "exponential_decay", shared=(),
                          custom_model: dict | None = None, p0=None, bounds=None,
                          fixed=None, maxfev=40000, labels=None) -> dict:
    """Fit several curves **at once**, with some parameters tied
    across all of them and the rest free per curve.

    The multivariate fit named on the roadmap for PFG and T1/T2. The
    question it answers is the one a per-curve fit cannot: *these
    peaks belong to the same molecule, so they share one diffusion
    coefficient -- what is it, and how well determined is it by all of
    the data together?* Fitting each peak separately and averaging the
    answers is not the same thing, and understates the uncertainty,
    because it throws away the constraint that they were never allowed
    to disagree in the first place.

    The same shape covers the other standard cases: one T2 across
    several integration regions of one species; a shared inversion
    efficiency ``C`` across every peak of an inversion-recovery series
    (the pulse was equally imperfect for all of them) while each peak
    keeps its own ``M0`` and ``T1``; a shared stretch exponent with
    per-peak ``D``.

    Parameters
    ----------
    datasets : sequence
        One entry per curve. Each is either ``(t, y)``, ``(t, y,
        y_err)``, or a dict with ``t``/``y``/optional ``y_err`` and
        ``label`` keys. Every curve is fit with the same model, but
        they need not share an x-grid or even a point count.
    model : str
        A :data:`RELAXATION_MODELS` key, as :func:`fit_relaxation`.
    shared : sequence of str
        Parameter names held to a single common value across every
        dataset. Anything not listed gets its own value per dataset.
        An empty ``shared`` is legal and simply fits them all
        independently in one call -- useful as the null comparison.
    custom_model : dict, optional
        As :func:`fit_relaxation` -- e.g.
        :func:`opennmr.processing.diffusion.build_diffusion_model`'s
        N-component spec.
    p0, bounds : optional
        Starting guesses and ``(lo, hi)`` bounds **in model parameter
        order**, applied to the shared parameters and used as each
        dataset's starting point for the per-dataset ones. Omit for
        the model's own data-driven defaults, computed from the
        concatenated data.
    fixed : dict, optional
        ``{param: value}`` held fixed for every dataset.
    labels : sequence of str, optional
        Names for the datasets in the result. Defaults to the dicts'
        own ``label`` keys, else ``"1"``, ``"2"``, ...

    Returns
    -------
    dict with:
        ``shared`` / ``shared_errors`` : the tied parameters and their
            uncertainties -- **one** number each, determined by all
            the data jointly.
        ``per_dataset`` : list of per-curve dicts, each shaped like a
            :func:`fit_relaxation` result (``params``, ``errors``,
            ``t``, ``y``, ``y_fit``, ``fit_quality``) so every
            existing consumer -- the report, the plot, the CSV -- works
            on one unchanged.
        ``uncertainty``, ``fit_quality`` : for the joint fit, over
            every point of every curve.

    Raises
    ------
    ValueError
        If ``datasets`` is empty, a ``shared`` name isn't a model
        parameter, or every parameter is both shared and fixed
        (nothing left to fit).
    """
    if custom_model is not None:
        spec = custom_model
    elif model not in RELAXATION_MODELS:
        raise ValueError(f"Unknown relaxation model '{model}'. Available: {sorted(RELAXATION_MODELS)}")
    else:
        spec = RELAXATION_MODELS[model]

    func, param_names = spec["func"], spec["params"]
    prepared = _prepare_global_datasets(datasets, labels)
    if not prepared:
        raise ValueError("fit_relaxation_global needs at least one dataset.")

    fixed = dict(fixed or {})
    shared = [name for name in dict.fromkeys(shared)]
    unknown = [name for name in shared if name not in param_names]
    if unknown:
        raise ValueError(
            f"shared names {unknown} are not parameters of model '{model}' "
            f"(it has {param_names})."
        )
    shared_free = [name for name in shared if name not in fixed]
    local_free = [name for name in param_names if name not in fixed and name not in shared]
    if not shared_free and not local_free:
        raise ValueError("Every parameter is fixed -- nothing to fit.")

    # Defaults come from the pooled data: a per-dataset default would
    # give each curve a different starting point for a parameter they
    # are about to be forced to agree on, which is a needlessly bad
    # place to start a joint fit from.
    t_all = np.concatenate([d["t"] for d in prepared])
    y_all = np.concatenate([d["y"] for d in prepared])
    order_all = np.argsort(t_all)
    p0_full = list(spec["p0"](t_all[order_all], y_all[order_all])) if p0 is None else list(p0)
    if bounds is None:
        lo_full, hi_full = spec["bounds"](t_all[order_all], y_all[order_all])
    else:
        lo_full, hi_full = bounds
    lo_full, hi_full = list(lo_full), list(hi_full)
    index_of = {name: i for i, name in enumerate(param_names)}

    # Vector layout: every shared free parameter once, then every
    # local free parameter repeated per dataset. Kept as an explicit
    # index map rather than reconstructed by arithmetic at each use,
    # so unpacking the result cannot drift out of step with packing.
    vector_names: list[str] = list(shared_free)
    slots: list[tuple[int, str]] = [(-1, name) for name in shared_free]
    for d_index in range(len(prepared)):
        for name in local_free:
            vector_names.append(name)
            slots.append((d_index, name))

    guess = [p0_full[index_of[name]] for _d, name in slots]
    lower = [lo_full[index_of[name]] for _d, name in slots]
    upper = [hi_full[index_of[name]] for _d, name in slots]

    # Same log10 search space as the single-curve fit -- see
    # LOG_FITTED_PARAMS. It matters *more* here: a joint fit has to
    # hold one shared D against several amplitudes at once, so the
    # conditioning that makes a single Stejskal-Tanner fit merely
    # imprecise makes this one land on the wrong answer outright
    # (measured: a shared D of 1.05e-09 recovered against a synthetic
    # truth of 8.50e-10, with every amplitude ~14% high to match).
    log_names = set(spec.get("log_params", ()))
    log_slots = [i for i, (_d, name) in enumerate(slots) if name in log_names]
    guess = _to_log_space(guess, log_slots, "initial guess")
    lower = _to_log_space(lower, log_slots, "lower bound", floor=True)
    upper = _to_log_space(upper, log_slots, "upper bound")

    def unpack(values, from_log: bool = True):
        """The flat parameter vector -> one full {name: value} dict per
        dataset. ``from_log`` undoes the log10 search transform on the
        way out; the error-vector caller passes ``False``, since a
        standard deviation is not exponentiated the way a value is (it
        is propagated through the Jacobian instead -- see
        :func:`_undo_log_space`)."""
        if from_log:
            values = _from_log_space(values, log_slots)
        shared_values = {name: values[i] for i, name in enumerate(shared_free)}
        out = []
        cursor = len(shared_free)
        for _d in prepared:
            full = dict(fixed)
            full.update(shared_values)
            for name in local_free:
                full[name] = values[cursor]
                cursor += 1
            out.append(full)
        return out

    # One concatenated "x" over every dataset, evaluated piecewise --
    # curve_fit works on a single 1-D residual, and the datasets are
    # deliberately allowed to have different x-grids and lengths.
    lengths = [len(d["t"]) for d in prepared]
    x_concat = np.concatenate([d["t"] for d in prepared])
    y_concat = np.concatenate([d["y"] for d in prepared])
    errs = [d.get("y_err") for d in prepared]
    if all(e is not None and np.all(np.isfinite(e)) and np.all(e > 0) for e in errs):
        sigma_concat = np.concatenate(errs)
    else:
        sigma_concat = None

    def joint(_x, *values):
        per_dataset = unpack(values)

        pieces = []
        for data, full in zip(prepared, per_dataset):
            pieces.append(func(data["t"], *[full[name] for name in param_names]))
        return np.concatenate(pieces)

    popt_log, pcov_log = curve_fit(
        joint, x_concat, y_concat, p0=guess, bounds=(lower, upper),
        sigma=sigma_concat, absolute_sigma=sigma_concat is not None,
        maxfev=maxfev, x_scale="jac",
    )
    popt, pcov = _undo_log_space(popt_log, pcov_log, log_slots)
    perr = np.sqrt(np.diag(pcov))

    resolved = unpack(popt, from_log=False)
    error_vectors = unpack(perr, from_log=False)
    # A fixed parameter has no uncertainty, and unpack() filled it with
    # its *value*; overwrite so nothing downstream reads a held-constant
    # number as an error bar.
    for errors in error_vectors:
        for name in fixed:
            errors[name] = np.nan

    y_fit_concat = joint(x_concat, *popt_log)
    residual_all = y_concat - y_fit_concat
    n_free = len(popt)
    dof = max(len(y_concat) - n_free, 1)
    chi2 = float(np.sum((residual_all / sigma_concat) ** 2)) if sigma_concat is not None \
        else float(np.sum(residual_all ** 2))
    ss_tot_all = float(np.sum((y_concat - np.mean(y_concat)) ** 2))

    per_dataset = []
    cursor = 0
    for data, full, errors, length in zip(prepared, resolved, error_vectors, lengths):
        y_fit = y_fit_concat[cursor:cursor + length]
        cursor += length
        residual = data["y"] - y_fit
        ss_tot = float(np.sum((data["y"] - np.mean(data["y"])) ** 2))
        per_dataset.append({
            "label": data["label"], "peak_label": data["label"],
            "model": model, "param_names": param_names, "rate_param": spec["rate_param"],
            "spec": spec, "params": dict(full), "errors": dict(errors),
            "t": data["t"], "y": data["y"], "y_err": data.get("y_err"), "y_fit": y_fit,
            "shared": list(shared_free), "fixed": dict(fixed),
            "uncertainty": describe_uncertainty(sigma_concat is not None, dof),
            "fit_quality": {
                "rmse": float(np.sqrt(np.mean(residual ** 2))),
                "chi2": float(np.sum(residual ** 2)),
                "reduced_chi2": chi2 / dof,
                "r_squared": 1.0 - float(np.sum(residual ** 2)) / ss_tot if ss_tot > 0 else np.nan,
                "dof": dof, "n_points": int(length),
                "decay_fraction": decay_fraction(data["y"]),
            },
        })

    return {
        "model": model, "param_names": param_names, "spec": spec,
        "shared": {name: float(popt[i]) for i, name in enumerate(shared_free)},
        "shared_errors": {name: float(perr[i]) for i, name in enumerate(shared_free)},
        "local_names": local_free, "fixed": dict(fixed),
        "per_dataset": per_dataset, "labels": [d["label"] for d in prepared],
        "popt": popt, "pcov": pcov, "vector_names": vector_names, "log_slots": log_slots,
        "uncertainty": describe_uncertainty(sigma_concat is not None, dof),
        "fit_quality": {
            "rmse": float(np.sqrt(np.mean(residual_all ** 2))),
            "chi2": chi2, "reduced_chi2": chi2 / dof,
            "r_squared": 1.0 - float(np.sum(residual_all ** 2)) / ss_tot_all if ss_tot_all > 0 else np.nan,
            "dof": dof, "n_points": int(len(y_concat)), "n_datasets": len(prepared),
        },
    }


def _prepare_global_datasets(datasets, labels) -> list[dict]:
    """Normalise :func:`fit_relaxation_global`'s several accepted
    dataset spellings into one list of ``{t, y, y_err, label}``, each
    sorted by t (the same ordering :func:`fit_relaxation` imposes, so
    a joint fit and a per-curve one see the data the same way)."""
    prepared = []
    for i, entry in enumerate(datasets):
        if isinstance(entry, dict):
            t, y = entry["t"], entry["y"]
            y_err = entry.get("y_err")
            label = entry.get("label") or entry.get("peak_label")
        else:
            items = list(entry)
            if len(items) == 2:
                (t, y), y_err = items, None
            elif len(items) == 3:
                t, y, y_err = items
            else:
                raise ValueError(
                    f"dataset {i} has {len(items)} elements -- expected (t, y) or (t, y, y_err)."
                )
            label = None
        t = np.asarray(t, dtype=float)
        y = np.asarray(y, dtype=float)
        if t.size != y.size:
            raise ValueError(f"dataset {i}: {t.size} x-values against {y.size} y-values.")
        y_err = None if y_err is None else np.asarray(y_err, dtype=float)
        order = np.argsort(t)
        prepared.append({
            "t": t[order], "y": y[order],
            "y_err": None if y_err is None else y_err[order],
            "label": (labels[i] if labels is not None and i < len(labels) else None) or label or f"{i + 1}",
        })
    return prepared


def format_global_report(result: dict, width: int = 66) -> str:
    """A readable summary of a :func:`fit_relaxation_global` result:
    the shared parameters first (they are the answer the joint fit
    exists to produce), then one line per dataset for the parameters
    that stayed local."""
    rule = "=" * width
    n_shared = len(result["shared"])
    lines = [rule, f"GLOBAL CURVE FIT — {result['model']}".center(width).rstrip(), rule]

    quality = result["fit_quality"]
    lines.append("")
    lines.append(f"  {quality['n_datasets']} curve(s), {quality['n_points']} points, "
                  f"{n_shared} shared parameter(s)")

    if result["shared"]:
        lines.append("")
        lines.append("  Shared across every curve")
        lines.append("  " + "-" * (width - 4))
        for name, value in result["shared"].items():
            error = result["shared_errors"].get(name)
            lines.append(f"    {name:<10} {format_param_value(name, value, error)}")
    else:
        lines.append("")
        lines.append("  No shared parameters — every curve fitted independently.")

    if result["local_names"]:
        lines.append("")
        lines.append("  Per curve")
        lines.append("  " + "-" * (width - 4))
        for entry in result["per_dataset"]:
            # Amplitudes as shares, for the same reason the per-peak
            # summary uses them: this block exists to compare curves
            # against each other, and each curve's absolute intensity
            # is the one number that cannot be.
            shares = amplitude_fractions(entry["params"])
            parts = [
                f"{name}={shares[name] * 100:.1f}%" if name in shares
                else f"{name}={format_param_value(name, entry['params'][name], entry['errors'].get(name))}"
                for name in result["local_names"]
            ]
            lines.append(f"    {entry['label']}: " + ",  ".join(parts))

    uncertainty = result["uncertainty"]
    lines.append("")
    lines.append("  Uncertainty")
    lines.append("  " + "-" * (width - 4))
    lines.append(f"    {uncertainty['short']}")
    for chunk in _wrap(uncertainty["description"], width - 6):
        lines.append(f"    {chunk}")

    lines.append("")
    lines.append("  Fit quality (joint, over every curve)")
    lines.append("  " + "-" * (width - 4))
    if np.isfinite(quality.get("r_squared", np.nan)):
        lines.append(f"    R²             = {quality['r_squared']:.5f}")
    lines.append(f"    RMSE           = {quality['rmse']:.4g}")
    lines.append(f"    reduced χ²     = {quality['reduced_chi2']:.4g}")
    lines.append(f"    points / d.o.f = {quality['n_points']} / {quality['dof']}")
    # **Per curve, not over the joint series.** The joint fit's own
    # y-vector is every dataset concatenated, so "last over first"
    # across it would divide one curve's end by another curve's start
    # -- a number with no meaning. Each curve's own attenuation is the
    # thing worth reading, and a joint fit is exactly where one curve
    # having barely decayed is easy to miss.
    fractions = [(entry.get("label") or f"curve {i + 1}",
                  entry.get("fit_quality", {}).get("decay_fraction", float("nan")))
                 for i, entry in enumerate(result.get("per_dataset") or [])]
    fractions = [(label, value) for label, value in fractions if np.isfinite(value)]
    if fractions:
        lines.append(f"    final I/I₀     = " + ", ".join(
            f"{label} {value:.3g}" for label, value in fractions))
    lines.append(rule)
    return "\n".join(lines)


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
    spec = fit_result.get("spec") or RELAXATION_MODELS[fit_result["model"]]
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


def _csv_param_unit(name: str) -> str:
    """The unit a fitted parameter's **raw** value is in, for a CSV
    column header, or ``""`` where it has none.

    Raw, not display-scaled: the panel shows a T2 in whatever unit
    suits it, but a file that silently switched units with a combo box
    would be unreadable a month later. Times stay in seconds and
    diffusion coefficients in m^2/s, with the unit named in the header
    so the file says what it holds."""
    if name in _TIME_PARAMS:
        return "s"
    if name.startswith(_DIFFUSION_PARAM_PREFIX) and name[1:].isdigit() or name in ("D", "Da", "Db"):
        return "m^2/s"
    return ""


def curve_fit_parameters_to_csv(results) -> str:
    """Every fitted peak's parameters and fit quality as a CSV table --
    one row per peak, one column per parameter with its error beside
    it, then the quality metrics.

    Explicit user request: "Export curve fit needs to include the fit
    report parameters, like D/T1/T2 and the fit quality". The series
    export alone gave the data behind the plot but not the numbers the
    fit produced, which are the reason the fit was run.

    **A table, not the formatted report.** ``format_relaxation_report``
    already renders those numbers for reading; this is the same content
    in a form a spreadsheet can sort and a script can parse. Values are
    raw (see :func:`_csv_param_unit`) with the unit named in the
    header.

    Columns are the union of every peak's parameter names, so a
    multi-component fit and a single-exponential one can sit in the
    same file -- a peak with no such parameter gets an empty cell
    rather than a zero.
    """
    import csv
    from io import StringIO

    results = list(results or [])
    if not results:
        raise ValueError("No curve fit to export -- run Fit Curve first.")

    names: list[str] = []
    for result in results:
        for name in (result.get("params") or {}):
            if name not in names:
                names.append(name)

    quality_columns = [
        ("r_squared", "R^2"), ("reduced_chi2", "reduced chi^2"),
        ("rmse", "RMSE"), ("dof", "dof"), ("n_points", "points"),
        # How far the series actually decayed -- see decay_fraction.
        # Alongside R² on purpose: the two answer different questions,
        # and a tidy fit through a series that barely moved is exactly
        # the case R² alone flatters.
        ("decay_fraction", "final I/I0"),
    ]

    buffer = StringIO()
    writer = csv.writer(buffer)
    header = ["peak", "model"]
    for name in names:
        unit = _csv_param_unit(name)
        header.append(f"{name} ({unit})" if unit else name)
        header.append(f"{name} error ({unit})" if unit else f"{name} error")
    header.extend(label for _key, label in quality_columns)
    header.append("uncertainty")
    writer.writerow(header)

    for index, result in enumerate(results, start=1):
        params = result.get("params") or {}
        errors = result.get("errors") or {}
        quality = result.get("fit_quality") or {}
        row = [result.get("peak_label") or result.get("label") or f"peak {index}",
               result.get("model", "")]
        for name in names:
            value = params.get(name)
            error = errors.get(name)
            row.append("" if value is None else f"{float(value):.10g}")
            row.append("" if error is None else f"{float(error):.10g}")
        for key, _label in quality_columns:
            value = quality.get(key)
            row.append("" if value is None else f"{float(value):.10g}")
        # The one-line form, not the whole describe_uncertainty dict --
        # its `description` is a paragraph meant for a report panel and
        # would swamp the row. What the column has to say is *which
        # kind* of error the ± is, which is exactly `short`.
        uncertainty = result.get("uncertainty") or {}
        row.append(str(uncertainty.get("short", "") if isinstance(uncertainty, dict)
                        else uncertainty))
        writer.writerow(row)
    return buffer.getvalue()


def curve_fit_to_csv(results, x_label: str = "x", x_scale: float = 1.0,
                      value_label: str = "value", include_fit: bool = True,
                      include_parameters: bool = True) -> str:
    """A curve fit as CSV: the x-axis, then each peak's **measured**
    value at every point, with the fitted curve alongside.

    Explicit user request: "I need the curve fit results to be
    exportable as integral/peak height vs whatever x is". That is the
    data behind the plot -- the per-slice intensities the Spectral
    Fitting tab produced and the curve that was fitted through them --
    rather than only the fitted parameters, which the report already
    gives.

    ``results`` : the list :func:`NMR2D_fit_relaxation` returns, one
    entry per peak. Each already carries its own ``t``/``y``/``y_fit``
    (and ``y_err`` where the measurement supplied one), so nothing is
    recomputed here and the file cannot disagree with what was fitted.

    **Peaks are not assumed to share an x-axis.** They normally do, but
    a point excluded from one peak's fit (see the GUI's own
    point-exclusion) leaves that peak shorter than the others. The
    union of every peak's x-values is used, and a peak with no
    measurement at some x is written as an empty cell rather than
    being silently dropped or padded with a number.

    ``x_scale`` : seconds per unit of the *written* x column, so a
    delay series exports in the same millisecond unit the panel
    displays. 1.0 leaves the values as fitted.
    ``value_label`` : what the measured column is -- "integral" or
    "amplitude", per the Value-source setting.

    ``include_parameters`` : prepend the fitted parameters and fit
    quality as their own table (see
    :func:`curve_fit_parameters_to_csv`), separated from the series by
    a blank line. On by default -- the numbers the fit produced are
    the reason it was run, and a file holding only the points behind
    the plot left them to be copied out of the GUI by hand. Pass
    ``False`` for the bare series table.

    A pure string builder (no file I/O), so it can be tested against
    its own output rather than against "did a file appear".
    """
    import csv
    from io import StringIO

    results = list(results or [])
    if not results:
        raise ValueError("No curve fit to export -- run Fit Curve first.")

    preamble = ""
    if include_parameters:
        # Two labelled sections rather than one wide table: the
        # parameters are per *peak* and the series is per *point*, so
        # they have genuinely different row meanings and forcing them
        # into one grid would leave most of it empty. The section
        # titles are what let a reader (or a script skipping to the
        # blank line) tell which is which.
        preamble = ("Curve fit parameters\n"
                    + curve_fit_parameters_to_csv(results)
                    + "\nCurve fit series\n")

    x_values = sorted({round(float(value), 12)
                       for result in results for value in np.asarray(result["t"], dtype=float)})
    has_errors = any(result.get("y_err") is not None for result in results)

    buffer = StringIO()
    writer = csv.writer(buffer)
    header = [x_label]
    for result in results:
        label = result.get("peak_label") or result.get("label") or "peak"
        header.append(f"{label} {value_label}")
        if has_errors:
            header.append(f"{label} {value_label} error")
        if include_fit:
            header.append(f"{label} fit")
    writer.writerow(header)

    columns = []
    for result in results:
        t = np.asarray(result["t"], dtype=float)
        y_err = result.get("y_err")
        columns.append((
            {round(float(value), 12): index for index, value in enumerate(t)},
            np.asarray(result["y"], dtype=float),
            np.asarray(result["y_fit"], dtype=float),
            np.asarray(y_err, dtype=float) if y_err is not None else None,
        ))

    for x in x_values:
        row = [f"{x / x_scale:.10g}"]
        for index_map, y, y_fit, y_err in columns:
            position = index_map.get(x)
            row.append("" if position is None else f"{y[position]:.10g}")
            if has_errors:
                row.append("" if position is None or y_err is None else f"{y_err[position]:.10g}")
            if include_fit:
                row.append("" if position is None else f"{y_fit[position]:.10g}")
        writer.writerow(row)
    return preamble + buffer.getvalue()


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


#: Parameters whose value is a *time* in seconds, and so should be
#: shown through :func:`format_time_value`'s automatic ms/us/ns
#: scaling rather than as a bare float. Matched by exact name against
#: a model's own parameter list.
_TIME_PARAMS = frozenset({"T1", "T2", "T1rho", "T1a", "T1b", "T2a", "T2b", "tau", "tau_c"})

#: Parameters whose value is a *diffusion coefficient* in m^2/s, and
#: so belong in scientific notation with the unit attached (see
#: :func:`format_diffusion_value`). Covers the named registry models'
#: ``D``/``Da``/``Db`` and :func:`opennmr.processing.diffusion.build_diffusion_model`'s
#: numbered ``D1``, ``D2``, ... alike.
_DIFFUSION_PARAM_PREFIX = "D"


#: Parameters whose value is an **amplitude** -- a signal intensity on
#: the data's own arbitrary scale. Covers every amplitude the named
#: registry models use (``M0``; ``A``/``A1``/``A2``; ``I0`` and its
#: ``I0a``/``I0b`` pair) and
#: :func:`opennmr.processing.diffusion.build_diffusion_model`'s
#: numbered ``I1``, ``I2``, ....
#:
#: Matched by name, the same way :data:`_TIME_PARAMS` and
#: :func:`_is_diffusion_param` are, and for the same reason: this
#: decides how a number is *presented*, and a presentation rule has to
#: be predictable from the model you picked. The cost is that a
#: **custom equation gets no ratios** -- its parameters are whatever
#: the user named them, and there is no way to know which of them is an
#: amplitude without guessing. Silently treating the wrong symbol as
#: one would be worse than showing nothing.
_AMPLITUDE_PARAM_RE = re.compile(r"^(?:M0|A\d*|I0[ab]?|I\d+)$")


def _is_amplitude_param(name: str) -> bool:
    return bool(_AMPLITUDE_PARAM_RE.match(name))


def amplitude_fractions(params: dict) -> dict[str, float]:
    """Each amplitude parameter as a fraction of the total amplitude.

    Explicit user request: "it would be good to have relative ratios
    instead of absolute intensity values so numbers can be more easily
    compared in multi-component fits". A fitted ``A1 = 0.0532`` means
    nothing on its own -- it is on whatever arbitrary scale the spectra
    happened to be processed on -- while "63% / 37%" is the form a
    multi-component result is actually quoted and compared in.

    Returns ``{}`` unless the model has **more than one** amplitude:
    a single-amplitude fit would report a constant 100%, which is not
    a result. Also ``{}`` when the amplitudes sum to zero, and for a
    custom equation (see :data:`_AMPLITUDE_PARAM_RE`).

    The denominator is the **signed** sum, matching
    :func:`opennmr.processing.diffusion.component_weights` so the two
    cannot quote different numbers for the same fit. A fit whose
    components have opposite signs can therefore produce a fraction
    outside 0-100%; that is a real signal -- usually an
    over-parameterised model -- and rounding it away into an absolute
    value would hide it.
    """
    amplitudes = {name: float(value) for name, value in params.items()
                  if _is_amplitude_param(name) and value is not None and np.isfinite(value)}
    if len(amplitudes) < 2:
        return {}
    total = sum(amplitudes.values())
    if not total:
        return {}
    return {name: value / total for name, value in amplitudes.items()}


def _is_diffusion_param(name: str) -> bool:
    return name == "D" or (name.startswith(_DIFFUSION_PARAM_PREFIX) and name[1:].isdigit()) \
        or name in ("Da", "Db")


def format_param_value(name: str, value, error=None, sig_figs: int = 3) -> str:
    """One parameter, formatted the way its own physical dimension
    wants: a relaxation time through :func:`format_time_value` (auto
    ms/us/ns), a diffusion coefficient through
    :func:`format_diffusion_value` (scientific, m^2/s), anything else
    -- amplitudes, stretch exponents, inversion efficiency -- as a
    plain significant-figures number, because those are dimensionless
    or on the data's own arbitrary intensity scale and inventing a
    unit for them would be worse than showing none.

    The point of routing every printed number through one function is
    that a value and its unit can never come apart: the unit is chosen
    from the same name that selected the value."""
    if value is None or not np.isfinite(value):
        return "n/a"
    if name in _TIME_PARAMS:
        return format_time_value(value, error, sig_figs=sig_figs)
    if _is_diffusion_param(name):
        return format_diffusion_value(value, error, sig_figs=sig_figs)
    if error is None or not np.isfinite(error):
        return f"{value:.{sig_figs + 1}g}"
    return f"{value:.{sig_figs + 1}g} ± {error:.{sig_figs}g}"


#: How each amplitude's name maps to the rate parameter of the same
#: component, for the named registry models. ``A1`` belongs with
#: ``T2a``/``T1a``, ``I0a`` with ``Da``, ``I3`` with ``D3`` -- the
#: pairing is by the vendor of the model, not by position, because the
#: parameter order differs between them.
_AMPLITUDE_RATE_SUFFIX = {"A1": ("a", "1"), "A2": ("b", "2"),
                           "I0a": ("a",), "I0b": ("b",)}


def _paired_rate_param(amplitude: str, param_names) -> str | None:
    """The rate parameter belonging to the same component as
    ``amplitude``, or ``None`` when there is no unambiguous pairing.

    ``I3`` -> ``D3`` and ``A1`` -> ``T2a``/``T1a`` are the two real
    conventions in this registry. ``None`` rather than a guess when
    neither applies: a wrongly-paired D next to a population share is
    worse than no D at all, since it reads as a measured fact.
    """
    names = list(param_names)
    if amplitude.startswith("I") and amplitude[1:].isdigit():
        candidate = f"D{amplitude[1:]}"
        return candidate if candidate in names else None
    for suffix in _AMPLITUDE_RATE_SUFFIX.get(amplitude, ()):
        for stem in ("T1", "T2", "D"):
            candidate = f"{stem}{suffix}"
            if candidate in names:
                return candidate
    return None


def format_relaxation_report(fit, title: str | None = None, width: int = 66) -> str:
    """A readable, self-describing summary of one
    :func:`fit_relaxation` result, as a string.

    Deliberately a **string builder**, not a printer: the GUI shows
    this in a panel, the CSV export embeds it, and a test can assert
    on it directly. :func:`print_relaxation_report` is now a one-line
    wrapper that prints what this returns.

    Three things the old one-line-per-parameter dump left the reader
    to work out for themselves, and this states outright:

    * **What kind of uncertainty** the ± is (see
      :data:`UNCERTAINTY_KINDS`). Two fits of the same data can quote
      very different errors depending only on whether per-point errors
      were supplied, and nothing said which was which.
    * **Units.** Every value goes through :func:`format_param_value`,
      so a T2 reads "12.4 ± 0.3 ms" and a D reads
      "2.03e-09 ± 0.05e-09 m^2/s" instead of both being bare floats.
    * **Whether the fit is any good** -- R², reduced chi-squared and
      the point count together, rather than RMSE on the data's own
      arbitrary intensity scale as the only quality number.

    Any model with **more than one amplitude** additionally gets each
    one as a fraction of the total (see :func:`amplitude_fractions`) --
    on the parameter's own row and again in a section pairing it with
    the rate it belongs to. That is the form a multi-component result
    is actually quoted and compared in; the raw amplitudes are on
    whatever arbitrary intensity scale the spectra were processed on
    and say nothing on their own.
    """
    rule = "=" * width
    lines = [rule, (title or f"CURVE FIT REPORT — {fit['model']}").center(width).rstrip(), rule]

    quality = fit.get("fit_quality", {})
    uncertainty = fit.get("uncertainty") or describe_uncertainty(fit.get("y_err") is not None,
                                                                 quality.get("dof", 1))

    rate_param = fit.get("rate_param")
    if rate_param and rate_param in fit["params"]:
        lines.append("")
        lines.append(f"  {rate_param} = {format_param_value(rate_param, fit['params'][rate_param], fit['errors'].get(rate_param))}")

    spec = fit.get("spec") or {}
    n_components = spec.get("n_components")
    # One source for the ratios, whatever shape the model is: a
    # generalised N-component diffusion fit (I1..In), a biexponential
    # (A1/A2, I0a/I0b), anything with more than one amplitude. The
    # numbers agree with component_weights by construction -- same
    # signed-sum denominator -- so the two cannot disagree about the
    # same fit.
    fractions = amplitude_fractions(fit["params"])

    lines.append("")
    lines.append("  Parameters")
    lines.append("  " + "-" * (width - 4))
    for name in fit["param_names"]:
        value = fit["params"].get(name)
        error = fit["errors"].get(name)
        fixed_note = "  (fixed)" if error is None or not np.isfinite(error) else ""
        # The ratio sits on the amplitude's own row, not only in the
        # section below, so the comparison is where the number is.
        share = fractions.get(name)
        share_note = f"   ({share * 100:.1f}%)" if share is not None else ""
        lines.append(
            f"    {name:<10} {format_param_value(name, value, None if fixed_note else error)}"
            f"{fixed_note}{share_note}"
        )

    if fractions:
        lines.append("")
        # Both terms, deliberately. "Population weights" is what a
        # multi-component diffusion result is actually quoted as and is
        # the phrase a reader looks for; "relative amplitudes" is what
        # the section shows for every other multi-amplitude model,
        # where "population" is a claim about the sample that the
        # numbers alone do not make.
        lines.append("  Relative amplitudes (population weights — fraction of the total)")
        lines.append("  " + "-" * (width - 4))
        # In the model's own parameter order, not alphabetical: a
        # 10-component fit sorts as I1, I10, I2, ... alphabetically,
        # and the model's order is the one the rest of the report uses.
        for name in [n for n in fit["param_names"] if n in fractions]:
            share = fractions[name]
            # Paired with whatever rate the same component was fitted
            # with -- a population share is only meaningful next to the
            # D or T it belongs to, and hunting for the pairing in the
            # block above is exactly the work this section removes.
            rate_name = _paired_rate_param(name, fit["param_names"])
            rate = ""
            if rate_name is not None:
                rate = f"   {rate_name} = {format_param_value(rate_name, fit['params'].get(rate_name))}"
            lines.append(f"    {name:<10} {share * 100:6.2f}%{rate}")
        if any(share < 0 or share > 1 for share in fractions.values()):
            # Not arithmetic noise: components of opposite sign are
            # usually an over-parameterised model, and an absolute
            # amplitude would have hidden it.
            lines.append("")
            for chunk in _wrap(
                "A share outside 0-100% means the components have opposite signs, "
                "which usually indicates more components than the data supports.",
                width - 6,
            ):
                lines.append(f"    {chunk}")

    lines.append("")
    lines.append("  Uncertainty")
    lines.append("  " + "-" * (width - 4))
    lines.append(f"    {uncertainty['short']}")
    for chunk in _wrap(uncertainty["description"], width - 6):
        lines.append(f"    {chunk}")

    lines.append("")
    lines.append("  Fit quality")
    lines.append("  " + "-" * (width - 4))
    if np.isfinite(quality.get("r_squared", np.nan)):
        lines.append(f"    R²             = {quality['r_squared']:.5f}")
    lines.append(f"    RMSE           = {quality.get('rmse', float('nan')):.4g}")
    lines.append(f"    reduced χ²     = {quality.get('reduced_chi2', float('nan')):.4g}")
    lines.append(f"    points / d.o.f = {quality.get('n_points', len(fit.get('t', [])))} / {quality.get('dof', '?')}")
    # How far the series actually decayed -- the one quality number
    # that says whether the *experiment* constrained the rate, as
    # opposed to how well a curve fits whatever it produced. See
    # decay_fraction.
    fraction = quality.get("decay_fraction")
    if fraction is None:
        fraction = decay_fraction(fit.get("y", []))
    if np.isfinite(fraction):
        lines.append(f"    final I/I₀     = {fraction:.4g}  ({fraction * 100:.1f}%)")
    lines.append(rule)
    return "\n".join(lines)


def _wrap(text: str, width: int) -> list[str]:
    """Greedy word wrap. Local rather than ``textwrap`` only so the
    report's own line width is the single thing controlling layout."""
    words, lines, current = text.split(), [], ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) > width and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def print_relaxation_report(fit):
    """Print :func:`format_relaxation_report`'s summary of a
    :func:`fit_relaxation` result."""
    print("\n" + format_relaxation_report(fit) + "\n")


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

#: What a slice's per-peak value can be, for a relaxation/diffusion
#: curve fit.
#:
#: ``"area"`` and ``"region"`` are both integrals, and telling them
#: apart is the point: ``NMR2D_fit_slices`` integrates each *fitted
#: lineshape* (see :func:`opennmr.processing.fitting.compute_integrals`),
#: while ``NMR2D_integrate_slices`` sums the raw data over a region
#: nobody fitted anything to. Both used to be called ``"integral"``,
#: which meant the same setting silently measured two different things
#: depending on which function produced the slices. Reported directly:
#: "the two options of peak height or integral don't make it clear if
#: a fully fitted spectrum is being used, this should be a third,
#: distinct option (where it should use the integrated peak area, not
#: the integrated region)."
#:
#: ``"integral"`` survives as the historical name and resolves to
#: whichever of the two the slices actually carry.
_VALUE_SOURCES = ("integral", "region", "area", "amplitude")


def _default_peak_label(fits, peak_index: int, value_source: str) -> str:
    """``"Peak 3 (4.213 ppm)"`` -- or ``"Region 2 (2.42-4.55 ppm)"``
    for an integrated series -- rather than a bare ``"Peak 3"``.

    Explicit user request: the fitting canvases should say where each
    peak actually is. A curve fit reports one number per peak, and
    without its position there is nothing tying that number back to
    the spectrum it came from -- which matters most in exactly the
    case the labels are needed for, a fit over several peaks whose
    order in the table is not obvious from the plot.

    Taken from the **first slice's own fitted shift**, not from the
    starting guess in ``peak_defs``: the guess is what the user typed
    and the fitted value is where the peak actually is, and with the
    default across-slices mode every slice shares one shift anyway.
    Falls back to the bare number whenever the position cannot be
    recovered -- a label is never worth raising for.
    """
    number = peak_index + 1
    if not fits:
        return f"Peak {number}"
    first = fits[0]

    # "region" is the new, explicit name; "integral" is the historical
    # one that meant this whenever the slices carried regions.
    if value_source in ("integral", "region") and "regions" in first:
        regions = first.get("regions") or []
        labels = first.get("region_labels") or []
        if peak_index < len(labels) and labels[peak_index]:
            return str(labels[peak_index])
        if peak_index < len(regions):
            low, high = regions[peak_index][:2]
            return f"Region {number} ({min(low, high):.4g}-{max(low, high):.4g} ppm)"
        return f"Region {number}"

    try:
        per_peak = extract_params_and_errors(
            first["popt"], first["pcov"], first["peak_defs"], first["is_free"])
        values = dict(zip(per_peak[peak_index]["names"], per_peak[peak_index]["params"]))
        return f"Peak {number} ({values['shift']:.4g} ppm)"
    except (KeyError, IndexError, TypeError, ValueError):
        return f"Peak {number}"


def _per_peak(value, peak_index: int):
    """One peak's own ``p0``/``bounds``/``fixed``, from a value that may
    be either shared across every peak or a ``{peak_index: value}``
    dict of per-peak overrides.

    A ``fixed`` argument is itself a dict, so "is this a per-peak
    mapping or one shared value?" cannot be answered by type alone --
    it is decided on the keys: a per-peak mapping is keyed by integers,
    a shared ``fixed`` by parameter *names*. Mixed keys would be
    ambiguous and are treated as shared, which is the older, safer
    reading."""
    if isinstance(value, dict) and value and all(isinstance(key, int) for key in value):
        return value.get(peak_index)
    return value


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

    indices = [int(fit["index"]) for fit in fits]
    # A list *longer* than the series is not a problem and never was:
    # each slice looks up its own entry and the surplus is simply never
    # read. That is Bruker's own convention -- a 24-entry vdlist with F1
    # TD set to 10 uses the first 10 -- and it falls out of indexing by
    # slice number for free.
    #
    # Shorter is the real failure, and it used to surface as a bare
    # IndexError turned into "check the vdlist matches this dataset",
    # which says what is wrong but not what to do. Name both counts and
    # the way out.
    missing = [i for i in indices if not (1 <= i <= len(vdlist_seconds))]
    if missing:
        raise ValueError(
            f"The value list has {len(vdlist_seconds)} entries but this fit "
            f"covers slice {max(missing)} -- {len(missing)} slice(s) have no "
            f"value. A longer list would be fine (the surplus is ignored); a "
            f"shorter one cannot be fitted. Limit the analysis to the first "
            f"{len(vdlist_seconds)} slices, or use a list that covers them all."
        )

    return np.array([vdlist_seconds[i - 1] for i in indices], dtype=float)


def NMR2D_fit_relaxation(fits, model: str = "inversion_recovery", vdlist=None,
                          peak_labels=None, normalised=False, value_source="integral",
                          use_errors=True, fixed=None, show_report=True,
                          custom_model: dict | None = None, p0=None, bounds=None,
                          return_series: bool = False):
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
    p0, bounds, fixed : optional
        Starting guesses, ``(lo, hi)`` bounds and held-fixed values,
        forwarded to :func:`fit_relaxation`.

        Each may be given **once for every peak**, or **per peak** as a
        ``{peak_index: value}`` dict keyed by 0-based peak number. A
        dict may name only some peaks; the rest fall back to the
        model's own automatic defaults.

        Per-peak matters more than it first looks: the peaks of one
        series often are *not* the same physics. A solvent line and a
        solute line in the same PFG series diffuse at genuinely
        different rates, so one shared starting D is a poor guess for
        at least one of them; and a peak that has decayed into the
        noise by mid-series may need its own bounds to stop the fit
        chasing it.
    return_series : bool
        Return ``(results, series)`` instead of just ``results``, where
        ``series`` is the ``[{t, y, y_err, label}, ...]`` this
        assembled on the way -- exactly the shape
        :func:`fit_relaxation_global` takes. Lets a caller do a joint
        fit over the same extracted series without re-deriving them
        from the slice results a second time.

    Returns
    -------
    list of dict
        One :func:`fit_relaxation` result per peak, with an added
        ``"peak_label"`` key. Or ``(results, series)`` when
        ``return_series``.
    """

    if value_source not in _VALUE_SOURCES:
        raise ValueError(f"value_source must be one of {sorted(_VALUE_SOURCES)}")

    if not fits:
        raise ValueError("No fitted slices available.")

    fitted_slices = "popt" in fits[0]
    if value_source in ("area", "amplitude") and not fitted_slices:
        raise ValueError(
            f"value_source={value_source!r} needs peaks fitted with NMR2D_fit_slices -- "
            f"these slices came from NMR2D_integrate_slices, which measures a region "
            f"without fitting a lineshape to it. Use value_source='region', or fit the "
            f"peaks first."
        )
    if value_source == "region" and fitted_slices:
        raise ValueError(
            "value_source='region' means a raw integral over a region, but these slices "
            "carry fitted peaks. Use value_source='area' for the area under the fitted "
            "lineshape, or 'amplitude' for its height."
        )
    # "integral" is the historical name from before these were told
    # apart, and meant whichever of the two the slices happened to
    # carry. Kept working, resolved to whichever it would have been.
    if value_source == "integral":
        value_source = "area" if fitted_slices else "region"

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
        # Both "area" and "region" read the same key; which one it holds
        # is decided by the function that produced the slices, and the
        # checks above have already established they agree.
        source = "integrals_norm" if normalised else "integrals"
        n_peaks = max(len(fit.get(source, [])) for fit in fits)

    results = []
    series = []

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
                                     fixed=_per_peak(fixed, peak_i),
                                     custom_model=custom_model,
                                     p0=_per_peak(p0, peak_i),
                                     bounds=_per_peak(bounds, peak_i))

        label = (peak_labels[peak_i] if peak_labels is not None and peak_i < len(peak_labels)
                 else _default_peak_label(fits, peak_i, value_source))
        fit_result["peak_label"] = label
        results.append(fit_result)
        series.append({
            "t": t_peak[valid], "y": y[valid],
            "y_err": None if y_err is None else y_err[valid], "label": label,
        })

        if show_report:
            print(f"\n--- {label} ---")
            print_relaxation_report(fit_result)

    return (results, series) if return_series else results
