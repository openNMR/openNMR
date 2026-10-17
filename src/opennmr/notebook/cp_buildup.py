"""
Cross-polarisation (CP) build-up curves: the integral of a region in
each spectrum of a variable-contact-time series, against the contact
time, and fits of the CP kinetics models.

Explicit user request (2026-10-16): "plot cross polarisation build-up
curves ... provide a reference (ideally non-CP, but not necessarily)
spectrum, then integrate a region of interest and apply across all other
spectra, then plot the integral as a function of the contact time
(extracted either from the pulse sequence, such as P15 ... or typed as a
number (units of microseconds)) ... fit their CP buildup curves given
initial guesses for the buildup time, T1rho and other relevant
parameters."

Models -- all from W. Kolodziejski and J. Klinowski, "Kinetics of
Cross-Polarization in Solid-State NMR: A Guide for Chemists", *Chem. Rev.*
**2002**, 102, 613-628 (equation numbers are theirs). Times in µs::

    "rise"         eq 11   I(t) = I0 [1 - exp(-t/T_IS)]
    "is"           eq 10   I(t) = I0 (1 - T_IS/T1ρ^I)^-1 [exp(-t/T1ρ^I) - exp(-t/T_IS)]
    "is_t1rho_s"   eq 9    I(t) = I0 (1 + T_IS/T1ρ^S - T_IS/T1ρ^I)^-1
                                  {exp(-t/T1ρ^I) - exp[-t(1/T_IS + 1/T1ρ^S)]}
    "two_is"       eq 10 twice, sharing T1ρ^I (two overlapping sites, or two
                   CP rates within one line)
    "iis"          eq 28   I(t) = I0 exp(-t/T1ρ^I) [1 - λ exp(-t/T_df)
                                  - (1-λ) exp(-3t/(2T_df)) exp(-t²/(2T2²))]
                   (I-I*-S, powders under MAS; λ = 1/(n+1) for an SIₙ group:
                   ½ for CH, ⅓ for CH2.)

Eqs 9-10 are the classical I-S model in the dilute-S limit (εα² ≈ 0): I0
is the amplitude without rotating-frame relaxation, proportional to the
amount of S in the sample (their eq 8). They are written here with
``expm1`` so they stay exact when T_IS ≈ T1ρ (where the textbook form is
0/0; the limit is I0·(t/T_IS)·exp(-t/T_IS)).

A reference spectrum (ideally direct excitation, or any spectrum) can be
given: its region integral is reported and the curve can be expressed
relative to it. Integrals are taken per scan (JEOL divided by its scans;
see :mod:`.intensity_scaling`) and, by default, per unit receiver gain,
since a variable-contact-time series is often recorded with the RG
changed between experiments.
"""

from __future__ import annotations

import math

import numpy as np

from ..core.acquisition import acquisition_constants
from ..core.spectrum import NMRSpectrum
from . import intensity_scaling
from .dnp import region_integral

REFERENCE = ("W. Kolodziejski and J. Klinowski, Chem. Rev. 2002, 102, 613-628")


# ============================================================
# Contact times
# ============================================================

def _as_number_us(value) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        v = float(value)
    else:
        text = str(value).strip().lower().replace("µ", "u").replace("μ", "u")
        scale = 1.0
        for suffix, s in (("us", 1.0), ("ms", 1e3), ("s", 1e6)):
            if text.endswith(suffix) and not text[: -len(suffix)].strip().isalpha():
                head = text[: -len(suffix)].strip()
                try:
                    float(head)
                except ValueError:
                    break
                text, scale = head, s
                break
        try:
            v = float(text) * scale
        except ValueError:
            return None
    if not math.isfinite(v) or v < 0:
        raise ValueError(f"A contact time must be a non-negative number of µs, not {value!r}.")
    return v


def contact_time_us(spectrum: NMRSpectrum, source) -> float:
    """The contact time (µs) of ``spectrum``: ``source`` itself when it is
    a number (``1000``, ``"1000"``, ``"1 ms"``), else the named
    acquisition parameter as the pulse programme names it -- ``P15``
    (Bruker pulses are recorded in µs), ``D20`` (delays in s, converted),
    ``CNST..``/others taken as µs. Raises ValueError naming what is
    available when the parameter is missing or zero."""
    direct = _as_number_us(source)
    if direct is not None:
        return direct
    name = str(source).strip()
    key = name.lower()
    constants = acquisition_constants(spectrum)
    value = constants.get(key)
    if value is None:
        value = constants.get(name)
    if value is None or not value > 0:
        if not constants:
            raise ValueError(f"This spectrum has no recorded acquisition parameters, so {name} "
                             "cannot be looked up: type the contact time in µs instead.")
        used = sorted(int(k[1:]) for k, v in constants.items()
                      if k.startswith("p") and k[1:].isdigit() and v and v > 0)
        listed = ", ".join(f"P{i} ({constants[f'p{i}']:g} µs)" for i in used[:12])
        zero = "" if value is None else " (recorded as 0, i.e. unused)"
        raise ValueError(f"{name} is not a usable contact time on this spectrum{zero}. "
                         f"Pulse lengths set: {listed or 'none'}.")
    if key.startswith("d") and key[1:].isdigit():
        return float(value) * 1e6
    return float(value)


# ============================================================
# Integrals against contact time
# ============================================================

def cp_series(spectra, region, contact_times, reference: NMRSpectrum | None = None,
              labels=None, baseline: str = "local", divide_rg: bool = True,
              relative_to: str = "none") -> dict:
    """The region integral of each spectrum against its contact time.

    ``region``: (lo, hi) ppm. ``contact_times``: one per spectrum -- a
    number (µs) or a parameter name (``"P15"``), or a single name/number
    used for all. ``reference``: optional spectrum integrated over the
    same region. ``baseline``: as :func:`opennmr.notebook.dnp.region_integral`
    ('local', 'global' or 'none'). ``divide_rg``: per unit receiver gain
    (where recorded). ``relative_to``: 'none', 'reference' (÷ the
    reference integral) or 'max' (÷ the largest).

    Returns ``{"t_us", "y", "y_err", "labels", "raw", "reference",
    "region", "relative_to", "notes", "unit"}`` sorted by contact time."""
    spectra = list(spectra)
    if not spectra:
        raise ValueError("No spectra given.")
    labels = list(labels) if labels is not None else [f"spectrum {i + 1}" for i in range(len(spectra))]
    if isinstance(contact_times, (str, int, float)):
        contact_times = [contact_times] * len(spectra)
    contact_times = list(contact_times)
    if len(contact_times) != len(spectra):
        raise ValueError(f"{len(spectra)} spectra but {len(contact_times)} contact times.")
    lo, hi = sorted(float(v) for v in region)
    notes: list[str] = []

    def integral(spectrum, label):
        if spectrum.ndim != 1:
            raise ValueError(f"{label} is not a 1D spectrum (keep a slice first).")
        r = region_integral(spectrum, lo, hi, baseline=baseline)
        info = intensity_scaling.comparison_factors(spectrum, "per_scan", divide_rg)
        notes.extend(f"{label}: {n}." for n in info["notes"])
        f = info["factor"]
        return r["value"] * f, abs(r["error"] * f), info

    rows = []
    for spectrum, label, source in zip(spectra, labels, contact_times):
        try:
            t = contact_time_us(spectrum, source)
        except ValueError as exc:
            raise ValueError(f"{label}: {exc}") from None
        value, error, info = integral(spectrum, label)
        rows.append({"label": label, "t_us": t, "integral": value, "error": error,
                     "ns_divisor": info["ns_divisor"], "rg": info["rg"],
                     "source": str(source)})
    rows.sort(key=lambda r: r["t_us"])
    ref = None
    if reference is not None:
        value, error, info = integral(reference, "reference")
        ref = {"integral": value, "error": error, "rg": info["rg"]}
    if not divide_rg:
        gains = {r["rg"] for r in rows if r["rg"] is not None}
        if len(gains) > 1:
            notes.append("Receiver gains differ across the series (RG "
                         + ", ".join(f"{g:g}" for g in sorted(gains))
                         + ") and are not divided: the curve mixes gains.")
    y = np.array([r["integral"] for r in rows], float)
    e = np.array([r["error"] for r in rows], float)
    unit = "integral per scan" + (" per unit RG" if divide_rg else "") + " (a.u.)"
    if relative_to == "reference":
        if ref is None or ref["integral"] == 0:
            raise ValueError("'Relative to the reference' needs a reference spectrum with a "
                             "non-zero integral in the region.")
        rel = ref["error"] / abs(ref["integral"])
        e = np.sqrt(e ** 2 + (y * rel) ** 2) / abs(ref["integral"])
        y = y / ref["integral"]
        unit = "integral / reference integral"
    elif relative_to == "max":
        m = y[np.argmax(np.abs(y))]
        y, e = y / m, e / abs(m)
        unit = "integral / largest integral"
    elif relative_to != "none":
        raise ValueError("relative_to must be 'none', 'reference' or 'max'.")
    if len(rows) != len({r["t_us"] for r in rows}):
        notes.append("Some contact times repeat: those points are fitted as repeats.")
    return {"t_us": np.array([r["t_us"] for r in rows], float), "y": y, "y_err": e,
            "labels": [r["label"] for r in rows], "raw": rows, "reference": ref,
            "region": (lo, hi), "relative_to": relative_to, "notes": notes, "unit": unit,
            "baseline": baseline, "divide_rg": divide_rg}


# ============================================================
# Models
# ============================================================

def _is_kernel(t, a, b):
    """exp(-b t) * (1 - exp(-(a-b) t)) / (a - b), exact as a -> b."""
    t = np.asarray(t, float)
    d = a - b
    if abs(d) * max(float(np.max(t)) if t.size else 0.0, 1e-300) < 1e-9:
        return t * np.exp(-b * t)
    return np.exp(-b * t) * (-np.expm1(-d * t)) / d


def cp_rise(t, I0, T_IS):
    """Eq 11: I0 [1 - exp(-t/T_IS)] (no rotating-frame relaxation)."""
    return I0 * (-np.expm1(-np.asarray(t, float) / T_IS))


def cp_is(t, I0, T_IS, T1rho_I):
    """Eq 10: classical I-S, I0 (1 - T_IS/T1ρ^I)^-1 [exp(-t/T1ρ^I) - exp(-t/T_IS)]."""
    return I0 / T_IS * _is_kernel(t, 1.0 / T_IS, 1.0 / T1rho_I)


def cp_is_t1rho_s(t, I0, T_IS, T1rho_I, T1rho_S):
    """Eq 9: classical I-S with T1ρ of S as well."""
    return I0 / T_IS * _is_kernel(t, 1.0 / T_IS + 1.0 / T1rho_S, 1.0 / T1rho_I)


def cp_two_is(t, I0_a, T_IS_a, I0_b, T_IS_b, T1rho_I):
    """Two eq-10 components sharing T1ρ^I."""
    return cp_is(t, I0_a, T_IS_a, T1rho_I) + cp_is(t, I0_b, T_IS_b, T1rho_I)


def cp_iis(t, I0, T_df, T2, T1rho_I, lam):
    """Eq 28: I-I*-S for a powder under MAS,
    I0 exp(-t/T1ρ^I) [1 - λ exp(-t/T_df) - (1-λ) exp(-3t/(2T_df)) exp(-t²/(2T2²))]."""
    t = np.asarray(t, float)
    return I0 * np.exp(-t / T1rho_I) * (
        1.0 - lam * np.exp(-t / T_df) - (1.0 - lam) * np.exp(-1.5 * t / T_df - t ** 2 / (2 * T2 ** 2)))


#: key -> description; parameters in order with (unit, lower, upper).
MODELS = {
    "is": {"name": "Classical I–S (eq 10): T_IS, T1ρ(I)", "func": cp_is, "equation": "eq 10",
           "params": ("I0", "T_IS", "T1rho_I")},
    "rise": {"name": "Simple rise, no T1ρ (eq 11)", "func": cp_rise, "equation": "eq 11",
             "params": ("I0", "T_IS")},
    "is_t1rho_s": {"name": "I–S with T1ρ(S) too (eq 9)", "func": cp_is_t1rho_s,
                   "equation": "eq 9", "params": ("I0", "T_IS", "T1rho_I", "T1rho_S")},
    "two_is": {"name": "Two I–S components, shared T1ρ(I)", "func": cp_two_is,
               "equation": "eq 10 × 2", "params": ("I0_a", "T_IS_a", "I0_b", "T_IS_b", "T1rho_I")},
    "iis": {"name": "Non-classical I–I*–S, powder/MAS (eq 28)", "func": cp_iis,
            "equation": "eq 28", "params": ("I0", "T_df", "T2", "T1rho_I", "lam")},
}

#: Parameter -> (label, unit). Times are µs.
PARAMETERS = {
    "I0": ("I₀", ""), "I0_a": ("I₀ (a)", ""), "I0_b": ("I₀ (b)", ""),
    "T_IS": ("T_IS (T_CP)", "µs"), "T_IS_a": ("T_IS (a)", "µs"), "T_IS_b": ("T_IS (b)", "µs"),
    "T1rho_I": ("T1ρ (I, e.g. ¹H)", "µs"), "T1rho_S": ("T1ρ (S)", "µs"),
    "T_df": ("T_df (spin diffusion)", "µs"), "T2": ("T₂ (I–S dephasing)", "µs"),
    "lam": ("λ = 1/(n+1)", ""),
}


def default_start(t_us, y, model: str = "is") -> dict:
    """Starting values read off the data: I0 from the maximum, T_IS from
    the rise to it, T1ρ from the decay after it (log-linear), else long."""
    t = np.asarray(t_us, float)
    y = np.asarray(y, float)
    if t.size == 0:
        raise ValueError("No points.")
    sign = 1.0 if y[np.argmax(np.abs(y))] >= 0 else -1.0
    ys = sign * y
    k = int(np.argmax(ys))
    t_peak = float(t[k]) if t[k] > 0 else float(np.max(t)) / 3 or 100.0
    peak = float(ys[k])
    t1rho = 20.0 * float(np.max(t)) or 1e4
    after = (t > t[k]) & (ys > 0.05 * peak)
    if np.count_nonzero(after) >= 2:
        slope = np.polyfit(t[after], np.log(ys[after]), 1)[0]
        if slope < 0:
            t1rho = -1.0 / slope
    t_is = max(t_peak / 3.0, 1e-3)
    i0 = sign * peak * 1.2
    start = {"I0": i0, "T_IS": t_is, "T1rho_I": max(t1rho, 3 * t_is), "T1rho_S": 50 * t1rho,
             "I0_a": 0.6 * i0, "T_IS_a": t_is / 3, "I0_b": 0.6 * i0, "T_IS_b": 3 * t_is,
             "T_df": max(t_peak / 2.0, 1e-3), "T2": max(t_peak / 10.0, 1.0), "lam": 0.5}
    return {p: start[p] for p in MODELS[model]["params"]}


def _bounds_for(name, start):
    if name.startswith("I0"):
        # the sign of the signal: components of one curve share it
        return (0.0, np.inf) if start >= 0 else (-np.inf, 0.0)
    if name == "lam":
        return (0.0, 1.0)
    return (1e-6, np.inf)


def fit_cp_buildup(t_us, y, model: str = "is", start: dict | None = None,
                   fixed: dict | None = None, bounds: dict | None = None, y_err=None) -> dict:
    """Fit a CP kinetics model (see :data:`MODELS`) to ``y`` against the
    contact time ``t_us`` (µs).

    ``start``: starting values (missing ones from :func:`default_start`);
    ``fixed``: {name: value} held, not fitted; ``bounds``: {name: (lo,
    hi)}; ``y_err``: per-point uncertainties (weights; their scale is
    kept only for the reduced χ²).

    Returns ``{"model", "params", "errors", "fixed", "free_names", "pcov",
    "t", "y", "y_fit", "curve_t", "curve_y", "r_squared", "rmse",
    "t_max_us", "warnings", "equation", "reference"}``."""
    from scipy.optimize import curve_fit

    if model not in MODELS:
        raise ValueError(f"Unknown model {model!r}; choose from {sorted(MODELS)}.")
    t = np.asarray(t_us, float)
    y = np.asarray(y, float)
    if t.shape != y.shape or t.ndim != 1:
        raise ValueError("t and y must be matching 1D arrays.")
    names = list(MODELS[model]["params"])
    func = MODELS[model]["func"]
    s0 = default_start(t, y, model)
    s0.update({k: float(v) for k, v in (start or {}).items() if k in names and v is not None})
    fixed = {k: float(v) for k, v in (fixed or {}).items() if k in names}
    free = [n for n in names if n not in fixed]
    if not free:
        raise ValueError("Every parameter is fixed: free at least one.")
    if t.size < len(free) + 1:
        raise ValueError(f"{t.size} points cannot fix {len(free)} free parameters: add contact "
                         "times or fix some parameters.")
    lo, hi = [], []
    for n in free:
        b = (bounds or {}).get(n) or _bounds_for(n, s0[n])
        lo.append(b[0]); hi.append(b[1])
        if not (b[0] < s0[n] < b[1]):
            s0[n] = min(max(s0[n], b[0] + 1e-9 * (abs(b[0]) + 1)), b[1] - 1e-9 * (abs(b[1]) + 1))

    def f(tt, *p):
        values = dict(fixed)
        values.update(zip(free, p))
        return func(tt, *[values[n] for n in names])

    sigma = None
    if y_err is not None:
        e = np.asarray(y_err, float)
        if e.shape == y.shape and np.all(e > 0):
            sigma = e
    try:
        p, pcov = curve_fit(f, t, y, p0=[s0[n] for n in free], bounds=(lo, hi), sigma=sigma,
                            absolute_sigma=False, maxfev=20000, x_scale="jac")
    except RuntimeError as exc:
        raise ValueError(f"The fit did not converge ({exc}). Try other starting values, or "
                         "fix a parameter.") from None
    errors = np.sqrt(np.clip(np.diag(pcov), 0, np.inf)) if np.all(np.isfinite(pcov)) \
        else np.full(len(free), np.nan)
    params = dict(fixed)
    params.update(zip(free, map(float, p)))
    errs = {n: float(e) for n, e in zip(free, errors)}
    y_fit = f(t, *p)
    ss_res = float(np.sum((y - y_fit) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    t_hi = max(float(np.max(t)) * 1.1, 1.0)
    curve_t = np.linspace(0.0, t_hi, 600)
    curve_y = f(curve_t, *p)
    k = int(np.argmax(np.abs(curve_y)))
    t_max = float(curve_t[k]) if 0 < k < curve_t.size - 1 else None
    warnings = []
    for n, (a, b) in zip(free, zip(lo, hi)):
        v = params[n]
        if (np.isfinite(a) and abs(v - a) <= 1e-6 * (abs(a) + 1)) or \
                (np.isfinite(b) and abs(v - b) <= 1e-6 * (abs(b) + 1)):
            warnings.append(f"{PARAMETERS[n][0]} ended at its limit ({v:.4g}).")
        if n in errs and np.isfinite(errs[n]) and v != 0 and errs[n] > 0.5 * abs(v):
            warnings.append(f"{PARAMETERS[n][0]} is poorly determined ({v:.4g} ± {errs[n]:.2g}).")
    for n in ("T1rho_I", "T1rho_S"):
        if n in free and params[n] > 3 * float(np.max(t)):
            warnings.append(f"{PARAMETERS[n][0]} = {params[n]:.4g} µs is longer than the "
                            "longest contact time × 3: the decay is barely sampled, so it is "
                            "uncertain -- add longer contact times.")
    if t_max is None and model != "rise":
        warnings.append("The fitted curve has not turned over within the contact times: the "
                        "maximum (and T1ρ) lie beyond the data.")
    if model == "iis" and "lam" in params:
        n_eff = 1.0 / params["lam"] - 1.0 if params["lam"] > 0 else float("inf")
        warnings.append(f"λ = {params['lam']:.3g} corresponds to n ≈ {n_eff:.2g} directly bonded "
                        "I spins (λ = 1/(n+1): ½ for CH, ⅓ for CH₂).")
    return {"model": model, "model_name": MODELS[model]["name"], "params": params,
            "errors": errs, "fixed": fixed, "free_names": free, "pcov": pcov,
            "t": t, "y": y, "y_fit": y_fit, "curve_t": curve_t, "curve_y": curve_y,
            "r_squared": r2, "rmse": math.sqrt(ss_res / t.size), "t_max_us": t_max,
            "warnings": warnings, "equation": MODELS[model]["equation"], "reference": REFERENCE}


def _time_text(us: float) -> str:
    if us is None or not math.isfinite(us):
        return "—"
    return f"{us / 1000:.4g} ms" if abs(us) >= 1000 else f"{us:.4g} µs"


def cp_report(fit: dict, series: dict | None = None) -> str:
    """The fit in plain text: each parameter with its error (times in µs
    and ms), the contact time of maximum signal, R², and the warnings."""
    lines = [f"CP build-up fit: {fit['model_name']} ({fit['equation']} of {fit['reference']})."]
    for n in MODELS[fit["model"]]["params"]:
        label, unit = PARAMETERS[n]
        v = fit["params"][n]
        if n in fit["fixed"]:
            err = "fixed"
        else:
            e = fit["errors"].get(n, float("nan"))
            err = f"± {_time_text(e) if unit else f'{e:.3g}'}"
        value = _time_text(v) if unit else f"{v:.4g}"
        lines.append(f"  {label} = {value} {err}")
    if fit.get("t_max_us") is not None:
        lines.append(f"  Signal is largest at a contact time of {_time_text(fit['t_max_us'])}.")
    lines.append(f"  R² = {fit['r_squared']:.4f}; {fit['t'].size} contact times "
                 f"({_time_text(float(np.min(fit['t'])))} to {_time_text(float(np.max(fit['t'])))}).")
    if series is not None:
        lo, hi = series["region"]
        lines.append(f"  Region {hi:g} to {lo:g} ppm; y = {series['unit']}.")
        if series.get("reference") is not None:
            lines.append(f"  Reference integral (same region): {series['reference']['integral']:.4g}"
                         f" ± {series['reference']['error']:.2g}.")
        for n in series.get("notes", []):
            lines.append(f"  Note: {n}")
    for w in fit["warnings"]:
        lines.append(f"  ⚠ {w}")
    return "\n".join(lines)


def plot_cp_buildup(series: dict, fit: dict | None = None, ax=None, time_unit: str = "auto",
                    colour=None, label: str | None = None, figure=None, fit_colour=None):
    """The build-up curve: points (with their integral errors) against
    contact time, and the fitted curve when given. Returns the Axes.
    ``figure``: draw on this (cleared) figure (the GUI's)."""
    if ax is None:
        if figure is not None:
            ax = figure.add_subplot(111)
        else:
            import matplotlib.pyplot as plt
            _fig, ax = plt.subplots(figsize=(5, 3.5))
    t = np.asarray(series["t_us"], float)
    if time_unit == "auto":
        time_unit = "ms" if t.size and float(np.max(t)) >= 10000 else "µs"
    scale = 1e-3 if time_unit == "ms" else 1.0
    ax.errorbar(t * scale, series["y"], yerr=series.get("y_err"), fmt="o", ms=5, capsize=2,
                color=colour or "black", label=label or "Integral", zorder=3)
    if fit is not None:
        ax.plot(np.asarray(fit["curve_t"]) * scale, fit["curve_y"], "-",
                color=fit_colour or "#d55e00", lw=1.5, label=f"Fit ({fit['equation']})", zorder=2)
    ax.set_xlabel(_italic_label(f"Contact time ({time_unit})"))
    ax.set_ylabel(_italic_label(series.get("unit", "Integral")))
    ax.axhline(0.0, color="0.6", lw=0.6, zorder=1)
    ax.set_xlim(left=0)
    ax.legend(frameon=False)
    return ax


def cp_series_regions(spectra, regions, contact_times, names=None, **kwargs) -> list[dict]:
    """:func:`cp_series` for several regions at once (explicit user request,
    2026-10-16: "as many integral regions as they so choose"): one series
    per ``(lo, hi)`` region, each with its ``"name"`` (default "Region 1",
    ...). Keyword arguments are :func:`cp_series`'s."""
    regions = list(regions)
    if not regions:
        raise ValueError("Add at least one region.")
    names = list(names) if names is not None else [f"Region {i + 1}" for i in range(len(regions))]
    out = []
    for name, region in zip(names, regions):
        try:
            series = cp_series(spectra, region, contact_times, **kwargs)
        except ValueError as exc:
            raise ValueError(f"{name}: {exc}") from None
        series["name"] = name
        out.append(series)
    return out


def plot_cp_buildups(series_list, fits=None, ax=None, time_unit: str = "auto", figure=None,
                     colours=None):
    """Several regions' build-up curves on one Axes, each in its own colour
    with its fit (``fits``: {series name: fit}) as a line of the same
    colour. Returns the Axes."""
    if ax is None:
        if figure is not None:
            ax = figure.add_subplot(111)
        else:
            import matplotlib.pyplot as plt
            _fig, ax = plt.subplots(figsize=(5, 3.5))
    if colours is None:
        from ..plotting.accessibility import ordered_qualitative_colours
        colours = ordered_qualitative_colours("okabe_ito")
    fits = fits or {}
    t_all = np.concatenate([np.asarray(s["t_us"], float) for s in series_list]) \
        if series_list else np.array([])
    if time_unit == "auto":
        time_unit = "ms" if t_all.size and float(np.max(t_all)) >= 10000 else "µs"
    scale = 1e-3 if time_unit == "ms" else 1.0
    for k, series in enumerate(series_list):
        colour = colours[k % len(colours)]
        name = series.get("name") or f"Region {k + 1}"
        ax.errorbar(np.asarray(series["t_us"]) * scale, series["y"], yerr=series.get("y_err"),
                    fmt="o", ms=5, capsize=2, color=colour, label=name, zorder=3)
        fit = fits.get(name)
        if fit is not None:
            ax.plot(np.asarray(fit["curve_t"]) * scale, fit["curve_y"], "-", color=colour, lw=1.5,
                    label=f"{name}: fit ({fit['equation']})", zorder=2)
    ax.set_xlabel(_italic_label(f"Contact time ({time_unit})"))
    if series_list:
        ax.set_ylabel(_italic_label(series_list[0].get("unit", "Integral")))
    ax.axhline(0.0, color="0.6", lw=0.6, zorder=1)
    ax.set_xlim(left=0)
    ax.legend(frameon=False)
    return ax


def _italic_label(text):
    """Quantity symbols italic (plotting.labels.italicise_symbols)."""
    from opennmr.plotting.labels import italicise_symbols
    return italicise_symbols(text)
