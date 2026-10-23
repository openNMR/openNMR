"""
Fit a measured spectrum against SIMPSON spectra.

A simulated spectrum has no parameters to fit -- the physics is already in it
-- but a measured one is rarely at exactly its position, intensity or width.
This fits the *simulation as a template*:

    y(x) = sum_k  a_k * [ T_k  convolved with a line of width (lb_k, gb_k) ](x - d_k)
           + c0 + c1 * x

for the templates ``T_k`` you supply (one SIMPSON spectrum, or several that
add up -- two sites, say), with

* ``a_k``  the amplitude (always fitted),
* ``d_k``  a shift in ppm (fitted by default, within +/- ``shift_window``),
* ``lb_k`` an extra Lorentzian and ``gb_k`` an extra Gaussian broadening, in
  Hz (fixed at zero unless ``float_broadening``): SIMPSON's own lines are
  as narrow as the simulation made them, a real sample's are not,
* ``c0``, ``c1`` a flat or sloping baseline (``baseline="offset"``/``"slope"``;
  ``"none"`` for neither).

The model is linear in ``a_k`` and ``c``; those are solved exactly at every
step (variable projection), so the optimiser only searches shifts and
broadenings and has far fewer ways to get stuck. Uncertainties are the usual
least-squares ones from the Jacobian, scaled by the residual variance.

Everything here is plain NumPy/SciPy on arrays, so a notebook gets the same
numbers the GUI shows::

    templates = [{"x": sim["x_ppm"], "y": sim["real_spectrum"], "name": "site A"}]
    out = fit_simpson_templates(exp["x_ppm"], exp["real_spectrum"], templates,
                                fit_regions=[(40, -40)])
    out["y_model"], out["params"]
"""

from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve
from scipy.special import wofz

from ..core import fit_progress

_FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def _uniform_ascending(x, y):
    """``(x, y)`` ascending on a uniform grid (interpolating if it is not)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    order = np.argsort(x)
    x, y = x[order], y[order]
    if x.size < 2:
        raise ValueError("A template needs at least two points.")
    step = np.diff(x)
    if np.allclose(step, step.mean(), rtol=1e-3, atol=0):
        return x, y
    grid = np.linspace(x[0], x[-1], x.size)
    return grid, np.interp(grid, x, y)


def voigt_kernel(n_half: int, dx_hz: float, lb: float, gb: float) -> np.ndarray:
    """A unit-sum line shape on ``2*n_half + 1`` points ``dx_hz`` apart:
    Lorentzian of full width ``lb`` Hz convolved with a Gaussian of full width
    ``gb`` Hz (pure Lorentzian or pure Gaussian when the other is zero)."""
    nu = np.arange(-n_half, n_half + 1) * dx_hz
    sigma = gb * _FWHM_TO_SIGMA
    gamma = lb / 2.0
    if sigma <= 0 and gamma <= 0:
        kernel = np.zeros_like(nu)
        kernel[n_half] = 1.0
        return kernel
    if sigma <= 0:
        kernel = gamma / np.pi / (nu ** 2 + gamma ** 2)
    else:
        z = (nu + 1j * gamma) / (sigma * np.sqrt(2.0))
        kernel = np.real(wofz(z)) / (sigma * np.sqrt(2.0 * np.pi))
    total = kernel.sum()
    return kernel / total if total > 0 else kernel


def broaden(x_ppm, y, sfo1_mhz: float, lb: float = 0.0, gb: float = 0.0):
    """``y`` (on the uniform ascending grid ``x_ppm``) broadened by a line of
    ``lb`` Hz Lorentzian and ``gb`` Hz Gaussian width. Area is preserved; the
    ends are treated as zero (no wrap-around), so a pattern that touches the
    edge of the window loses a little there."""
    if lb <= 0 and gb <= 0:
        return np.asarray(y, dtype=float)
    dx_hz = float(x_ppm[1] - x_ppm[0]) * sfo1_mhz
    reach = max(lb * 20.0, gb * 4.0, 3 * dx_hz)         # Lorentzian tails: wide
    n_half = int(min(np.ceil(reach / dx_hz), 4 * len(y)))
    return fftconvolve(np.asarray(y, dtype=float), voigt_kernel(n_half, dx_hz, lb, gb),
                       mode="same")


def template_on(x, template: dict, shift: float = 0.0, lb: float = 0.0, gb: float = 0.0,
                sfo1_mhz: float = 1.0) -> np.ndarray:
    """A template evaluated on ``x`` (ppm), moved by ``shift`` ppm and
    broadened by ``lb``/``gb`` Hz. Beyond the template's own range it keeps
    its edge values (a flat continuation): zero there would put a step at the
    edge as the template shifts, which a fit cannot differentiate."""
    tx, ty = template["_grid"]
    smooth = broaden(tx, ty, sfo1_mhz, lb, gb)
    return np.interp(np.asarray(x, dtype=float) - shift, tx, smooth,
                     left=float(smooth[0]), right=float(smooth[-1]))


def _prepare(templates):
    out = []
    for i, t in enumerate(templates):
        if "x" not in t or "y" not in t:
            raise ValueError("Each template needs 'x' (ppm) and 'y' (intensity) arrays.")
        grid = _uniform_ascending(t["x"], t["y"])
        out.append({**t, "_grid": grid, "name": t.get("name") or f"Template {i + 1}"})
    if not out:
        raise ValueError("Give at least one template spectrum.")
    return out


def _baseline_columns(x, baseline: str):
    if baseline == "none":
        return []
    if baseline == "offset":
        return [np.ones_like(x)]
    if baseline == "slope":
        centre = 0.5 * (x.max() + x.min())
        return [np.ones_like(x), x - centre]
    raise ValueError("baseline must be 'none', 'offset' or 'slope'.")


def fit_simpson_templates(x_ppm, y, templates, *, sfo1_mhz: float = 1.0, fit_regions=None,
                          float_shift: bool = True, shift_window: float = 5.0,
                          float_broadening: bool = False, max_broadening_hz: float = 2000.0,
                          start_shifts=None, start_lb=None, baseline: str = "offset",
                          non_negative: bool = False) -> dict:
    """Fit ``y(x_ppm)`` with the SIMPSON ``templates``.

    Parameters
    ----------
    x_ppm, y : array-like
        The measured spectrum (any order).
    templates : list of dict
        Each ``{"x": ppm, "y": intensity, "name": str}``.
    sfo1_mhz : float
        The measured spectrum's observe frequency, converting Hz of
        broadening to ppm. 1 means the broadening values are in ppm.
    fit_regions : list of (lo, hi), optional
        Fit only inside these ppm ranges (either order). Default: everything
        the templates and the data share.
    float_shift, shift_window
        Fit each template's shift, within +/- ``shift_window`` ppm of its
        start (``start_shifts``, default 0).
    float_broadening, max_broadening_hz, start_lb
        Fit an extra Lorentzian and Gaussian width per template (Hz, between
        0 and ``max_broadening_hz``); ``start_lb`` is the Lorentzian start (default: one point spacing of the template; a width of zero has no slope to search from).
    baseline : {"none", "offset", "slope"}
    non_negative : bool
        Force every amplitude >= 0 (a template is a spectrum, so it usually
        is). Solved with non-negative least squares.

    Returns
    -------
    dict with ``x`` (the fitted points, ppm), ``y_data``, ``y_model``,
    ``components`` (one array per template, amplitude included), ``baseline``
    (array), ``params`` (list of one dict per template: ``name``, ``amplitude``
    + ``amplitude_err``, ``shift`` + ``shift_err``, ``lb``, ``gb`` and their
    errors, ``fraction`` of the total area), ``baseline_params``, ``rmse``,
    ``r_squared``, ``n_points``, ``n_free``, ``message``.
    """
    from scipy.optimize import nnls

    x_all = np.asarray(x_ppm, dtype=float)
    y_all = np.asarray(y, dtype=float)
    if x_all.shape != y_all.shape or x_all.ndim != 1:
        raise ValueError("x_ppm and y must be 1D arrays of the same length.")
    templates = _prepare(templates)
    mask = np.ones(x_all.shape, dtype=bool)
    if fit_regions:
        mask = np.zeros(x_all.shape, dtype=bool)
        for lo, hi in fit_regions:
            a, b = (lo, hi) if lo <= hi else (hi, lo)
            mask |= (x_all >= a) & (x_all <= b)
    if mask.sum() < 3 + len(templates):
        raise ValueError("Too few points inside the fit regions for this many templates.")
    order = np.argsort(x_all[mask])
    x = x_all[mask][order]
    y_fit = y_all[mask][order]
    n_t = len(templates)
    base_cols = _baseline_columns(x, baseline)
    start_shifts = np.zeros(n_t) if start_shifts is None else np.asarray(start_shifts, float)
    start_lb = np.zeros(n_t) if start_lb is None else np.asarray(start_lb, float)
    # A line much narrower than the template's point spacing changes it only
    # in proportion to its width squared, so a search started at zero width
    # sees no slope and never leaves. Start broadening at one point spacing.
    grid_hz = np.array([abs(float(t["_grid"][0][1] - t["_grid"][0][0])) * float(sfo1_mhz)
                        for t in templates])

    # Nonlinear parameters: shift_k (if floating), lb_k and gb_k (if floating).
    names, x0, lower, upper = [], [], [], []
    for k in range(n_t):
        if float_shift:
            names.append(("shift", k))
            x0.append(float(start_shifts[k]))
            lower.append(float(start_shifts[k]) - shift_window)
            upper.append(float(start_shifts[k]) + shift_window)
        if float_broadening:
            names.append(("lb", k))
            x0.append(float(min(max(start_lb[k], grid_hz[k]), 0.5 * max_broadening_hz)))
            lower.append(0.0)
            upper.append(float(max_broadening_hz))
            names.append(("gb", k))
            x0.append(float(min(grid_hz[k], 0.5 * max_broadening_hz)))
            lower.append(0.0)
            upper.append(float(max_broadening_hz))

    def unpack(p):
        shifts, lbs, gbs = start_shifts.copy(), start_lb.copy(), np.zeros(n_t)
        for value, (what, k) in zip(p, names):
            if what == "shift":
                shifts[k] = value
            elif what == "lb":
                lbs[k] = value
            else:
                gbs[k] = value
        return shifts, lbs, gbs

    def design(p):
        shifts, lbs, gbs = unpack(p)
        cols = [template_on(x, templates[k], shifts[k], lbs[k], gbs[k], sfo1_mhz)
                for k in range(n_t)]
        return np.column_stack(cols + base_cols)

    def solve_linear(a):
        if non_negative:
            # Amplitudes >= 0; the baseline may take either sign.
            n_amp = n_t
            extra = a[:, n_amp:]
            augmented = np.column_stack([a[:, :n_amp], extra, -extra]) if extra.size else a
            sol, _ = nnls(augmented, y_fit)
            coef = sol[:n_amp].tolist()
            if extra.size:
                m = extra.shape[1]
                coef += (sol[n_amp:n_amp + m] - sol[n_amp + m:]).tolist()
            return np.asarray(coef)
        coef, *_ = np.linalg.lstsq(a, y_fit, rcond=None)
        return coef

    def residual(p):
        a = design(p)
        return a @ solve_linear(a) - y_fit

    message = "linear fit (nothing nonlinear to search)"
    if x0:
        solution = fit_progress.limited_least_squares(
            fit_progress.tracked_residuals(residual, n_free=len(x0) + n_t + len(base_cols)),
            np.asarray(x0), bounds=(np.asarray(lower), np.asarray(upper)),
            label="The SIMPSON fit", x_scale=np.maximum(np.asarray(upper) - np.asarray(lower), 1e-9),
            diff_step=1e-3)
        p = solution.x
        message = solution.message
        jac = solution.jac
    else:
        p, jac = np.zeros(0), None

    a = design(p)
    coef = solve_linear(a)
    model = a @ coef
    resid = y_fit - model
    n_points = int(y_fit.size)
    n_free = len(p) + a.shape[1]
    dof = max(n_points - n_free, 1)
    variance = float(resid @ resid) / dof

    # Uncertainties: the full Jacobian over every parameter (amplitudes and
    # baseline enter linearly, the others through the columns).
    errors = np.full(len(p) + a.shape[1], np.nan)
    try:
        step = 1e-4
        columns = [a]
        full_jac = []
        for j in range(len(p)):
            h = step * max(abs(p[j]), 1.0)
            q = p.copy()
            q[j] += h
            full_jac.append((design(q) @ coef - model) / h)
        for j in range(a.shape[1]):
            full_jac.append(a[:, j])
        full_jac = np.column_stack(full_jac)
        cov = np.linalg.pinv(full_jac.T @ full_jac) * variance
        errors = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    except np.linalg.LinAlgError:
        pass

    shifts, lbs, gbs = unpack(p)
    err_of = {}
    for j, (what, k) in enumerate(names):
        err_of[(what, k)] = float(errors[j])
    amp_err = errors[len(p):len(p) + n_t]
    components = []
    for k in range(n_t):
        components.append(coef[k] * template_on(x, templates[k], shifts[k], lbs[k], gbs[k],
                                                sfo1_mhz))
    integrate = getattr(np, "trapezoid", None) or np.trapz       # NumPy 1.x or 2.x
    areas = np.array([abs(integrate(c, x)) for c in components])
    total_area = areas.sum() or 1.0
    params = []
    for k in range(n_t):
        params.append({
            "name": templates[k]["name"], "amplitude": float(coef[k]),
            "amplitude_err": float(amp_err[k]), "shift": float(shifts[k]),
            "shift_err": err_of.get(("shift", k), 0.0 if not float_shift else float("nan")),
            "lb": float(lbs[k]), "lb_err": err_of.get(("lb", k), 0.0),
            "gb": float(gbs[k]), "gb_err": err_of.get(("gb", k), 0.0),
            "fraction": float(areas[k] / total_area)})
    base = np.zeros_like(x)
    base_params = []
    for j, column in enumerate(base_cols):
        base += coef[n_t + j] * column
        base_params.append({"name": ("offset", "slope")[j], "value": float(coef[n_t + j]),
                            "error": float(errors[len(p) + n_t + j])})
    total = float(np.sum((y_fit - y_fit.mean()) ** 2)) or 1.0
    return {
        "x": x, "y_data": y_fit, "y_model": model, "components": components, "baseline": base,
        "params": params, "baseline_params": base_params,
        "rmse": float(np.sqrt(np.mean(resid ** 2))),
        "r_squared": float(1.0 - np.sum(resid ** 2) / total),
        "n_points": n_points, "n_free": int(n_free), "message": str(message),
        "settings": {"float_shift": bool(float_shift), "shift_window": float(shift_window),
                     "float_broadening": bool(float_broadening),
                     "baseline": baseline, "non_negative": bool(non_negative),
                     "sfo1_mhz": float(sfo1_mhz),
                     "fit_regions": [list(map(float, r)) for r in (fit_regions or [])]},
    }


# ---------------------------------------------------------------------------
# Notebook-friendly front end: accept spectra as the app holds them
# ---------------------------------------------------------------------------

def _as_xy(item, name=None):
    """``(x_ppm, y, sfo1 or None, name)`` from an NMRSpectrum, a results dict
    (``NMR1dimport``/``SIMPSONimport``), or an ``(x, y)`` pair."""
    from ..core.spectrum import NMRSpectrum

    if isinstance(item, NMRSpectrum):
        if item.ndim != 1:
            raise ValueError("Give a 1D spectrum (select one slice of a 2D first).")
        label = name or (item.metadata.raw or {}).get("title") or item.metadata.source_path
        return (np.asarray(item.axes[0].ppm, dtype=float), np.asarray(item.data, dtype=float),
                item.metadata.sfo1, label)
    if isinstance(item, dict):
        x = item.get("x_ppm")
        if x is None:
            raise ValueError("That spectrum has no ppm axis.")
        return (np.asarray(x, dtype=float), np.asarray(item["real_spectrum"], dtype=float),
                item.get("SFO1"), name or item.get("path"))
    x, y = item
    return np.asarray(x, dtype=float), np.asarray(y, dtype=float), None, name


def SIMPSONfit_spectrum(experiment, simulations, names=None, **kwargs) -> dict:
    """Fit a measured spectrum with one or more SIMPSON spectra.

    ``experiment`` and each of ``simulations`` may be an
    :class:`~opennmr.core.spectrum.NMRSpectrum` (what ``SIMPSONload`` or the
    GUI's Open experiments hold), a results dict from ``NMR1dimport`` /
    ``SIMPSONimport``, or an ``(x_ppm, y)`` pair. ``names`` labels the
    simulations. A single simulation may be passed on its own (an NMRSpectrum or
    a dict); for an ``(x, y)`` pair, put it in a list. The other keyword arguments are those of
    :func:`fit_simpson_templates`; ``sfo1_mhz`` defaults to the experiment's
    observe frequency.

    Returns :func:`fit_simpson_templates`' dict."""
    if isinstance(simulations, dict) or hasattr(simulations, "axes"):
        simulations = [simulations]          # one spectrum; a list of (x, y) pairs is already a list
    simulations = list(simulations)
    x, y, sfo1, _ = _as_xy(experiment)
    templates = []
    for i, sim in enumerate(simulations):
        sx, sy, _s, label = _as_xy(sim, names[i] if names else None)
        templates.append({"x": sx, "y": sy, "name": label or f"Simulation {i + 1}"})
    kwargs.setdefault("sfo1_mhz", float(sfo1) if sfo1 else 1.0)
    return fit_simpson_templates(x, y, templates, **kwargs)


def SIMPSONplot_fit(result: dict, ax=None, show_components: bool = True, pl_width: float = 3.375,
                    pl_height: float = 3.0, xlabel: str = "δ (ppm)"):
    """Data, fit, residual and each simulated component of
    :func:`SIMPSONfit_spectrum`. Returns ``(fig, ax)``."""
    import matplotlib.pyplot as plt
    from ..plotting.accessibility import default_series_colours
    from ..plotting.labels import italicise_symbols

    if ax is None:
        fig, ax = plt.subplots(figsize=(pl_width, pl_height))
    else:
        fig = ax.figure
    x = result["x"]
    ax.plot(x, result["y_data"], color="0.35", linewidth=1.0, label="Data")
    colours = default_series_colours(max(len(result["components"]), 1))
    if show_components:
        for i, comp in enumerate(result["components"]):
            ax.plot(x, comp + result["baseline"], color=colours[i], linewidth=1.0,
                    linestyle="--", label=result["params"][i]["name"])
    ax.plot(x, result["y_model"], color="black", linewidth=1.2, label="Fit")
    ax.set_xlabel(italicise_symbols(xlabel))
    ax.set_yticks([])
    ax.invert_xaxis()
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False)
    return fig, ax
