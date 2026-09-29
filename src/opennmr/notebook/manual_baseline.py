"""
Manual baseline correction with an explicit, user-visible model.

The automatic estimators in :mod:`opennmr.processing.fitting` pick their
own baseline points and return a curve; this module is for the case the
user wants to *say* what the baseline is:

* ``"polynomial"`` -- ``A x^4 + B x^3 + C x^2 + D x + E``
* ``"sine"``       -- ``A sin(B x + C) + D``

with ``x`` the chemical shift in ppm (so ``B`` is in rad/ppm) and the
result in the spectrum's own intensity units. The model is fitted by
least squares to the points inside user-chosen baseline regions (ppm
ranges that contain no signal); any parameter can be held fixed (e.g.
``A = B = 0`` for a quadratic), and the fitted coefficients can be
edited by hand and evaluated directly.

Every function works on plain arrays, and :func:`manual_baseline_correct`
on the ``{"x_ppm", "real_spectrum", ...}`` result dicts the rest of
``opennmr.notebook`` uses, so a notebook and the GUI (Process ▸
Baseline ▸ "Manual: ...") compute exactly the same numbers -- the GUI
calls :func:`evaluate_manual_baseline` / :func:`fit_manual_baseline`
through :func:`opennmr.processing.ft.compute_baseline`.

Example
-------
>>> fit = fit_manual_baseline(x, y, "polynomial",
...                           regions=[(-50, -20), (180, 220)],
...                           fixed={"A": 0.0, "B": 0.0})   # quadratic
>>> fit["params"]            # {'A': 0.0, 'B': 0.0, 'C': ..., 'D': ..., 'E': ...}
>>> corrected = y - evaluate_manual_baseline(x, "polynomial", fit["params"])
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit

#: Parameter names, in order, for each model.
MANUAL_BASELINE_PARAMS = {
    "polynomial": ("A", "B", "C", "D", "E"),
    "sine": ("A", "B", "C", "D"),
}

#: The model written out, for labels, logs and reports.
MANUAL_BASELINE_EQUATIONS = {
    "polynomial": "A·x⁴ + B·x³ + C·x² + D·x + E",
    "sine": "A·sin(B·x + C) + D",
}

_ALIASES = {
    "polynomial": "polynomial", "poly": "polynomial", "quartic": "polynomial",
    "manual_polynomial": "polynomial",
    "sine": "sine", "sin": "sine", "sinusoid": "sine", "manual_sine": "sine",
}


def _model_name(model: str) -> str:
    try:
        return _ALIASES[str(model).lower()]
    except KeyError:
        raise ValueError(f"model must be 'polynomial' or 'sine', got {model!r}") from None


def _as_param_list(model: str, params) -> list[float]:
    """``params`` as an ordered list of floats, from a dict keyed by
    parameter name or a sequence in :data:`MANUAL_BASELINE_PARAMS` order."""
    names = MANUAL_BASELINE_PARAMS[model]
    if isinstance(params, dict):
        return [float(params.get(n, 0.0)) for n in names]
    values = [float(v) for v in params]
    if len(values) != len(names):
        raise ValueError(f"the {model} baseline has {len(names)} parameters "
                         f"({', '.join(names)}); got {len(values)}")
    return values


def polynomial_baseline(x, A=0.0, B=0.0, C=0.0, D=0.0, E=0.0):
    """``A x^4 + B x^3 + C x^2 + D x + E``."""
    x = np.asarray(x, dtype=float)
    return (((A * x + B) * x + C) * x + D) * x + E


def sine_baseline(x, A=0.0, B=0.0, C=0.0, D=0.0):
    """``A sin(B x + C) + D`` (``B`` in rad per unit of ``x``)."""
    x = np.asarray(x, dtype=float)
    return A * np.sin(B * x + C) + D


def evaluate_manual_baseline(x, model: str, params) -> np.ndarray:
    """The baseline ``model`` with ``params`` (dict or ordered sequence)
    evaluated on ``x`` (ppm)."""
    model = _model_name(model)
    values = _as_param_list(model, params)
    if model == "polynomial":
        return polynomial_baseline(x, *values)
    return sine_baseline(x, *values)


def region_mask(x, regions) -> np.ndarray:
    """Boolean mask of the points of ``x`` inside any ``(lo, hi)`` of
    ``regions`` (order within a pair does not matter). ``None`` or an
    empty list selects nothing."""
    x = np.asarray(x, dtype=float)
    mask = np.zeros(x.shape, dtype=bool)
    if not regions:
        return mask
    if np.isscalar(regions[0]):
        regions = [regions]
    for lo, hi in regions:
        lo, hi = sorted((float(lo), float(hi)))
        mask |= (x >= lo) & (x <= hi)
    return mask


def _fixed_dict(model, fixed) -> dict[str, float]:
    """``fixed`` normalised to ``{name: value}``. Accepts a dict of
    values, or (with ``initial``) handled by the caller."""
    names = MANUAL_BASELINE_PARAMS[model]
    if not fixed:
        return {}
    out = {}
    for name, value in dict(fixed).items():
        if name not in names:
            raise ValueError(f"{name!r} is not a parameter of the {model} baseline "
                             f"({', '.join(names)})")
        out[name] = float(value)
    return out


def _fit_polynomial(xs, ys, fixed):
    """Exact linear least squares for the free coefficients, with the
    fixed ones moved to the right-hand side. Solved on a scaled,
    centred variable (for conditioning -- x^4 at 200 ppm is 1.6e9) and
    mapped back exactly to coefficients of plain ``x``."""
    names = MANUAL_BASELINE_PARAMS["polynomial"]
    powers = {"A": 4, "B": 3, "C": 2, "D": 1, "E": 0}
    target = ys - sum(value * xs ** powers[n] for n, value in fixed.items())
    free = [n for n in names if n not in fixed]
    result = dict(fixed)
    if not free:
        return result, np.zeros((0, 0))
    if len(xs) < len(free):
        raise ValueError(f"{len(xs)} baseline point(s) cannot determine {len(free)} "
                         f"free coefficient(s) -- widen the baseline regions or fix more")
    # Column scaling only (no centring, which would mix fixed and free
    # coefficients): each column divided by its own max |value|.
    columns = np.column_stack([xs ** powers[n] for n in free])
    scale = np.max(np.abs(columns), axis=0)
    scale[scale == 0] = 1.0
    design = columns / scale
    solution, _res, rank, _sv = np.linalg.lstsq(design, target, rcond=None)
    if rank < len(free):
        raise ValueError("the baseline points do not determine every free coefficient "
                         "(too few distinct x values) -- widen the regions or fix more")
    coeffs = solution / scale
    dof = max(len(xs) - len(free), 1)
    residual = target - design @ solution
    s2 = float(residual @ residual) / dof
    try:
        cov = s2 * np.linalg.inv(design.T @ design) / np.outer(scale, scale)
    except np.linalg.LinAlgError:
        cov = np.full((len(free), len(free)), np.nan)
    for n, c in zip(free, coeffs):
        result[n] = float(c)
    return result, cov


def _sine_initial_guess(xs, ys, fixed):
    """Starting values for ``A sin(Bx + C) + D``: for a set of trial
    frequencies (from an FFT of the baseline points resampled onto a
    uniform grid, and 0.25-4 cycles across their x-range), the model is
    linear in ``a sin(Bx) + b cos(Bx) + D`` -- solve that exactly and
    keep the best."""
    span = float(np.ptp(xs)) or 1.0
    trials = [2.0 * np.pi * c / span for c in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0)]
    if len(xs) >= 16:
        order = np.argsort(xs)
        grid = np.linspace(xs[order[0]], xs[order[-1]], 512)
        resampled = np.interp(grid, xs[order], ys[order])
        spectrum = np.abs(np.fft.rfft(resampled - resampled.mean()))
        freqs = np.fft.rfftfreq(grid.size, d=grid[1] - grid[0])
        if spectrum.size > 1:
            k = int(np.argmax(spectrum[1:]) + 1)
            trials.append(2.0 * np.pi * freqs[k])
    if "B" in fixed:
        trials = [fixed["B"]]
    best = None
    for B in trials:
        design = np.column_stack([np.sin(B * xs), np.cos(B * xs), np.ones_like(xs)])
        sol, *_ = np.linalg.lstsq(design, ys, rcond=None)
        cost = float(np.sum((design @ sol - ys) ** 2))
        if best is None or cost < best[0]:
            best = (cost, B, sol)
    _cost, B, (a, b, D) = best
    A = float(np.hypot(a, b))
    C = float(np.arctan2(b, a))   # a sin + b cos = A sin(Bx + C)
    guess = {"A": A, "B": float(B), "C": C, "D": float(D)}
    guess.update(fixed)
    return guess


def _canonical_sine(params: dict) -> dict:
    """Same curve, conventional signs: ``A >= 0``, ``B >= 0``,
    ``C`` in (-pi, pi]."""
    A, B, C, D = (params[n] for n in "ABCD")
    if B < 0:          # A sin(-|B|x + C) = -A sin(|B|x - C)
        B, C, A = -B, -C, -A
    if A < 0:          # -|A| sin(t) = |A| sin(t + pi)
        A, C = -A, C + np.pi
    C = float(np.angle(np.exp(1j * C)))
    if np.isclose(C, -np.pi):
        C = np.pi
    return {"A": float(A), "B": float(B), "C": C, "D": float(D)}


def _fit_sine(xs, ys, fixed, initial):
    names = MANUAL_BASELINE_PARAMS["sine"]
    free = [n for n in names if n not in fixed]
    if len(xs) < len(free):
        raise ValueError(f"{len(xs)} baseline point(s) cannot determine {len(free)} "
                         f"free parameter(s) -- widen the baseline regions or fix more")
    start = dict(initial) if initial else _sine_initial_guess(xs, ys, fixed)
    start.update(fixed)
    if not free:
        return dict(start), np.zeros((0, 0))

    def model(x, *values):
        p = dict(fixed)
        p.update(zip(free, values))
        return sine_baseline(x, *(p[n] for n in names))

    p0 = [start[n] for n in free]
    popt, pcov = curve_fit(model, xs, ys, p0=p0, maxfev=20000)
    result = dict(fixed)
    result.update({n: float(v) for n, v in zip(free, popt)})
    if not fixed.keys() & {"A", "B", "C"}:
        result = _canonical_sine(result)
    return result, np.atleast_2d(pcov)


def fit_manual_baseline(x, y, model: str = "polynomial", regions=None, fixed=None,
                        initial=None) -> dict:
    """Fit ``model`` to the points of ``(x, y)`` inside ``regions``.

    Parameters
    ----------
    x, y : array
        Chemical shift (ppm) and intensity. A complex ``y`` is fitted on
        its real part.
    model : {"polynomial", "sine"}
        ``A x^4 + B x^3 + C x^2 + D x + E`` or ``A sin(B x + C) + D``.
    regions : list of (lo, hi), optional
        ppm ranges that hold only baseline. ``None`` uses the first and
        last 5 % of the points, the same default the automatic methods
        use -- but choose regions yourself where you can: that is the
        point of a manual baseline.
    fixed : dict, optional
        ``{name: value}`` for parameters held at a value rather than
        fitted, e.g. ``{"A": 0, "B": 0}`` for a quadratic.
    initial : dict, optional
        Starting values for the sine fit (the polynomial fit is linear
        and needs none). By default they are found automatically.

    Returns
    -------
    dict with ``model``, ``params`` (``{name: value}``), ``errors``
    (1-sigma standard errors, 0 for fixed), ``fixed`` (names),
    ``regions``, ``n_points``, ``rms`` (residual RMS over the baseline
    points), ``equation`` and ``baseline`` (the fitted curve on all of
    ``x``).
    """
    model = _model_name(model)
    x = np.asarray(x, dtype=float)
    y = np.real(np.asarray(y))
    if x.shape != y.shape:
        raise ValueError(f"x and y differ in shape: {x.shape} vs {y.shape}")
    if regions:
        mask = region_mask(x, regions)
    else:
        n = x.size
        k = max(3, int(round(0.05 * n)))
        mask = np.zeros(n, dtype=bool)
        mask[:k] = True
        mask[-k:] = True
    mask &= np.isfinite(y)
    xs, ys = x[mask], y[mask]
    if xs.size == 0:
        raise ValueError("no points fall inside the baseline regions")
    fixed_values = _fixed_dict(model, fixed)
    if model == "polynomial":
        params, cov = _fit_polynomial(xs, ys, fixed_values)
    else:
        params, cov = _fit_sine(xs, ys, fixed_values, initial)
    names = MANUAL_BASELINE_PARAMS[model]
    free = [n for n in names if n not in fixed_values]
    errors = {n: 0.0 for n in names}
    diag = np.diag(cov) if cov.size else np.array([])
    for n, var in zip(free, diag):
        errors[n] = float(np.sqrt(var)) if np.isfinite(var) and var >= 0 else float("nan")
    params = {n: float(params[n]) for n in names}
    fitted_points = evaluate_manual_baseline(xs, model, params)
    rms = float(np.sqrt(np.mean((ys - fitted_points) ** 2)))
    return {
        "model": model,
        "params": params,
        "errors": errors,
        "fixed": sorted(fixed_values, key=names.index),
        "regions": [tuple(sorted(map(float, r))) for r in (regions or [])],
        "n_points": int(xs.size),
        "rms": rms,
        "equation": MANUAL_BASELINE_EQUATIONS[model],
        "baseline": evaluate_manual_baseline(x, model, params),
    }


def describe_manual_baseline(model: str, params, digits: int = 6) -> str:
    """One line for a log or methods section, e.g.
    ``"A·sin(B·x + C) + D with A = 12.3, B = 0.0314, C = -1.2, D = 4"``
    (x in ppm)."""
    model = _model_name(model)
    values = _as_param_list(model, params)
    parts = ", ".join(f"{n} = {v:.{digits}g}" for n, v in zip(MANUAL_BASELINE_PARAMS[model], values))
    return f"{MANUAL_BASELINE_EQUATIONS[model]} (x in ppm) with {parts}"


def manual_baseline_correct(result: dict, model: str = "polynomial", regions=None, params=None,
                            fixed=None, initial=None) -> dict:
    """Baseline-correct a notebook result dict (``x_ppm`` and
    ``real_spectrum``, e.g. from :func:`opennmr.notebook.NMR1dimport`).

    With ``params`` the baseline is evaluated from them as given (a
    hand-set baseline); otherwise it is fitted to ``regions`` with
    :func:`fit_manual_baseline`. Returns a *copy* of ``result`` with
    ``real_spectrum`` corrected, plus ``baseline`` (the subtracted
    curve), ``uncorrected_spectrum`` and ``baseline_fit`` (the fit
    dict, or the evaluated parameters). The input is not modified.
    """
    model = _model_name(model)
    x = np.asarray(result["x_ppm"], dtype=float)
    y = np.real(np.asarray(result["real_spectrum"]))
    if params is not None:
        values = dict(zip(MANUAL_BASELINE_PARAMS[model], _as_param_list(model, params)))
        baseline = evaluate_manual_baseline(x, model, values)
        fit = {"model": model, "params": values, "equation": MANUAL_BASELINE_EQUATIONS[model],
               "regions": [tuple(sorted(map(float, r))) for r in (regions or [])],
               "baseline": baseline}
    else:
        fit = fit_manual_baseline(x, y, model, regions=regions, fixed=fixed, initial=initial)
        baseline = fit["baseline"]
    out = dict(result)
    out["uncorrected_spectrum"] = y
    out["baseline"] = baseline
    out["real_spectrum"] = y - baseline
    out["baseline_fit"] = fit
    return out


def plot_manual_baseline(x, y, fit_or_baseline, regions=None, ax=None, show_corrected=True,
                         xlabel="δ / ppm", colors=("0.15", "tab:blue", "tab:orange")):
    """Publication-ready check plot: the spectrum, the baseline (dashed)
    and, optionally, the corrected spectrum, with the baseline regions
    shaded. ``fit_or_baseline`` is a :func:`fit_manual_baseline` result
    or a baseline array. Returns the Axes."""
    import matplotlib.pyplot as plt

    x = np.asarray(x, dtype=float)
    y = np.real(np.asarray(y))
    if isinstance(fit_or_baseline, dict):
        baseline = np.asarray(fit_or_baseline["baseline"], dtype=float)
        regions = regions if regions is not None else fit_or_baseline.get("regions")
    else:
        baseline = np.asarray(fit_or_baseline, dtype=float)
    if ax is None:
        _fig, ax = plt.subplots(figsize=(6.5, 3.5), constrained_layout=True)
    for lo, hi in regions or []:
        ax.axvspan(min(lo, hi), max(lo, hi), color="0.85", alpha=0.6, lw=0, zorder=0)
    ax.plot(x, y, color=colors[0], lw=1.0, label="spectrum")
    ax.plot(x, baseline, color=colors[1], lw=1.4, ls="--", label="baseline")
    if show_corrected:
        ax.plot(x, y - baseline, color=colors[2], lw=1.0, label="corrected")
    ax.set_xlim(max(x), min(x))    # NMR convention: shift increases to the left
    ax.set_xlabel(xlabel)
    ax.set_yticks([])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False)
    return ax
