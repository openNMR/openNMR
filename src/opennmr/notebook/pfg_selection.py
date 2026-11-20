"""Choosing, checking and quoting multi-component PFG diffusion fits.

From the PEGDA-membrane PFG rerun (improvement notes, November 2026):

* :func:`select_diffusion_model` -- fit 1...``n_max`` components, each with and without a
  constant offset ``C``, rank by AICc and BIC, and let only fits that pass physical validity
  gates be selected (a component is added only when AICc improves by more than 10).
* :func:`inflated_errors` -- the Birge-ratio / PDG scale-factor errors, measured sigma times
  sqrt(max(1, reduced chi-squared)).
* :func:`component_weights_with_errors` -- component fractions with errors from the full
  covariance, and ratios of two parameters (e.g. D_membrane / D_water).
* :func:`combine_measurements` -- repeats averaged per sample first, then over samples.
* :func:`free_water_D`, :func:`water_temperature_from_D`, :func:`residual_d2o_limit`,
  :func:`tail_residual` -- PFG quality checks.

Every function takes plain arrays / fit dictionaries, so it works the same from a notebook
and from the GUI.
"""

from __future__ import annotations

import itertools
import math

import numpy as np

from ..processing import diffusion as _diff
from ..processing.relaxation import fit_relaxation

__all__ = [
    "aicc", "bic", "select_diffusion_model", "format_model_selection", "inflated_errors",
    "component_weights_with_errors", "parameter_ratio", "combine_measurements",
    "free_water_D", "water_temperature_from_D", "residual_d2o_limit", "tail_residual",
]


# ----------------------------------------------------------------------
# Information criteria
# ----------------------------------------------------------------------
def aicc(chi2: float, n: int, k: int) -> float:
    """Small-sample AIC: ``n ln(chi2/n) + 2k + 2k(k+1)/(n-k-1)`` (inf when ``n - k - 1 <= 0``)."""
    if n - k - 1 <= 0 or not chi2 >= 0:
        return math.inf
    chi2 = max(chi2, 1e-12 * n)          # a numerically exact fit must not give ln(0)
    return n * math.log(chi2 / n) + 2 * k + 2.0 * k * (k + 1) / (n - k - 1)


def bic(chi2: float, n: int, k: int) -> float:
    """``n ln(chi2/n) + k ln n``."""
    if n <= 0 or not chi2 >= 0:
        return math.inf
    chi2 = max(chi2, 1e-12 * n)
    return n * math.log(chi2 / n) + k * math.log(n)


# ----------------------------------------------------------------------
# Model selection
# ----------------------------------------------------------------------
def _names(n_components: int, offset: bool) -> tuple[list[str], list[str]]:
    base = _diff.diffusion_component_names(n_components)
    amps = [nm for nm in base if nm.startswith("I")]
    ds = [nm for nm in base if nm.startswith("D")]
    return amps, ds


def _start_sets(b, n_components: int, max_sets: int = 40):
    """Starting ``D`` lists (descending, one per component) on a log grid spanning the b range."""
    b = np.asarray(b, dtype=float)
    positive = b[b > 0]
    if not positive.size:
        return [None]
    lo = 0.1 / float(positive.max())
    hi = min(3.0 / float(positive.min()), 3e-8)
    if hi <= lo:
        hi = lo * 1e3
    grid = np.logspace(math.log10(lo), math.log10(hi), 7)
    sets = [sorted(c, reverse=True) for c in itertools.combinations(grid, n_components)]
    # components a decade apart at least are the useful starts; keep the spread ones first
    sets.sort(key=lambda s_: -(math.log10(s_[0] / s_[-1]) if len(s_) > 1 else 0.0))
    return sets[:max_sets] if n_components > 1 else sets


def _chi2(fit) -> float:
    return float(fit["fit_quality"]["chi2"])


def _validate(fit, n_components: int, offset: bool, amp_sigma: float, amp_frac: float,
              d_rel_err: float, d_ratio_min: float) -> list[str]:
    """Reasons a fit fails the validity gates (empty list = valid)."""
    p, e = fit["params"], fit["errors"]
    lo, hi = fit["bounds"]
    names = list(fit["param_names"])
    amps, ds = _names(n_components, offset)
    reasons = []
    total = sum(float(p[a]) for a in amps) + (float(p["C"]) if offset else 0.0)
    for a in amps:
        sig = e.get(a, np.nan)
        if not np.isfinite(sig):
            reasons.append(f"{a}: uncertainty undetermined")
        elif p[a] <= amp_sigma * sig:
            reasons.append(f"{a} = {p[a]:.3g} is not above {amp_sigma:g}σ ({sig:.2g})")
        if total > 0 and p[a] <= amp_frac * total:
            reasons.append(f"{a} is below {100 * amp_frac:g}% of I0")
    for d in ds:
        sig = e.get(d, np.nan)
        if not np.isfinite(sig):
            reasons.append(f"{d}: uncertainty undetermined")
        elif p[d] <= 0 or sig / p[d] > d_rel_err:
            reasons.append(f"{d} is determined only to {100 * sig / max(p[d], 1e-300):.0f}%"
                           f" (> {100 * d_rel_err:g}%)")
        i = names.index(d)
        for bound, label in ((lo[i], "lower"), (hi[i], "upper")):
            if np.isfinite(bound) and bound > 0 and abs(math.log(p[d] / bound)) < 0.02 * math.log(10):
                reasons.append(f"{d} sits at its {label} bound")
    values = sorted((float(p[d]) for d in ds), reverse=True)
    for a_, b_ in zip(values[:-1], values[1:]):
        if b_ > 0 and a_ / b_ <= d_ratio_min:
            reasons.append(f"adjacent D values differ by only ×{a_ / b_:.2f} (≤ {d_ratio_min:g})")
    if offset:
        sig = e.get("C", np.nan)
        if not np.isfinite(sig) or p["C"] <= amp_sigma * sig:
            reasons.append(f"C = {p['C']:.3g} is not significant (≤ {amp_sigma:g}σ)")
    return reasons


def select_diffusion_model(b, y, y_err=None, n_max: int = 3, offsets=(False, True),
                           prefer_simpler: bool = True, delta_aicc: float = 10.0,
                           amp_sigma: float = 2.0, amp_fraction: float = 1e-3,
                           d_rel_err: float = 0.5, d_ratio_min: float = 1.5,
                           expected_D=None) -> dict:
    """Fit 1...``n_max`` Stejskal-Tanner components, each with and without a constant ``C``
    (``offsets``), rank them by AICc and BIC, and select one.

    Only fits passing **all** the gates can be selected: each amplitude above
    ``amp_sigma`` σ and ``amp_fraction`` of I0; each D determined to better than ``d_rel_err``;
    adjacent D ratios above ``d_ratio_min``; no D at a bound; C significant when included.
    With ``prefer_simpler`` the candidates are walked from few parameters to many, and a more
    complex one replaces the current choice only if its AICc is lower by more than
    ``delta_aicc``; otherwise the lowest AICc among the valid fits is chosen.

    Each starting point set is a log grid of D values across the b range (or ``expected_D``
    when given); the best chi-squared of the starts is kept.

    Returns ``{"rows": [...ranked by AICc...], "selected": row | None, "fit": fit | None,
    "log": str}``; a row has ``n_components``, ``offset``, ``k``, ``chi2``, ``aicc``, ``bic``,
    ``delta_aicc`` (from the lowest), ``valid``, ``reasons`` and ``fit``."""
    b = np.asarray(b, dtype=float)
    y = np.asarray(y, dtype=float)
    if b.shape != y.shape or b.ndim != 1:
        raise ValueError("b and y must be 1D arrays of the same length.")
    n = b.size
    rows = []
    for n_comp in range(1, int(n_max) + 1):
        for offset in offsets:
            row = {"n_components": n_comp, "offset": bool(offset), "fit": None, "valid": False,
                   "reasons": [], "chi2": math.inf, "aicc": math.inf, "bic": math.inf, "k": None}
            starts = [expected_D] if expected_D is not None and \
                np.size(expected_D) in (1, n_comp) else _start_sets(b, n_comp)
            best = None
            for guess in starts:
                try:
                    spec = _diff.build_diffusion_model(n_comp, expected_D=guess, offset=bool(offset))
                    fit = fit_relaxation(b, y, y_err=y_err, custom_model=spec)
                except Exception:  # noqa: BLE001 -- a failed start is just a worse start
                    continue
                if best is None or _chi2(fit) < _chi2(best):
                    best = fit
            if best is None:
                row["reasons"] = ["the fit did not converge from any starting point"]
                rows.append(row)
                continue
            k = len(best["free_names"])
            chi2 = _chi2(best)
            row.update(fit=best, k=k, chi2=chi2, aicc=aicc(chi2, n, k), bic=bic(chi2, n, k))
            row["reasons"] = _validate(best, n_comp, bool(offset), amp_sigma, amp_fraction,
                                       d_rel_err, d_ratio_min)
            row["valid"] = not row["reasons"] and np.isfinite(row["aicc"])
            rows.append(row)

    finite = [r["aicc"] for r in rows if np.isfinite(r["aicc"])]
    best_aicc = min(finite) if finite else math.nan
    for r in rows:
        r["delta_aicc"] = r["aicc"] - best_aicc if np.isfinite(r["aicc"]) else math.nan
    valid = [r for r in rows if r["valid"]]
    selected = None
    if valid:
        if prefer_simpler:
            valid_sorted = sorted(valid, key=lambda r: (r["k"], r["aicc"]))
            selected = valid_sorted[0]
            for r in valid_sorted[1:]:
                if r["k"] > selected["k"] and selected["aicc"] - r["aicc"] > delta_aicc:
                    selected = r
                elif r["k"] == selected["k"] and r["aicc"] < selected["aicc"]:
                    selected = r
        else:
            selected = min(valid, key=lambda r: r["aicc"])
    rows.sort(key=lambda r: (not np.isfinite(r["aicc"]), r["aicc"]))
    out = {"rows": rows, "selected": selected, "fit": None if selected is None else selected["fit"]}
    out["log"] = format_model_selection(out, delta_aicc=delta_aicc, prefer_simpler=prefer_simpler)
    return out


def format_model_selection(result: dict, delta_aicc: float = 10.0, prefer_simpler: bool = True) -> str:
    """The ranked table, with the reason each rejected fit failed, as plain text."""
    lines = [f"{'comp.':>5} {'C':>2} {'k':>3} {'chi2':>11} {'AICc':>9} {'dAICc':>7} {'BIC':>9}  status"]
    for r in result["rows"]:
        status = "SELECTED" if r is result["selected"] else ("valid" if r["valid"] else "rejected")
        lines.append(f"{r['n_components']:>5} {'+' if r['offset'] else '-':>2} "
                     f"{'-' if r['k'] is None else r['k']:>3} {r['chi2']:>11.4g} {r['aicc']:>9.2f} "
                     f"{r['delta_aicc']:>7.2f} {r['bic']:>9.2f}  {status}")
        if not r["valid"]:
            for reason in r["reasons"]:
                lines.append(f"{'':>10}· {reason}")
    rule = (f"a component (or C) is added only when AICc improves by more than {delta_aicc:g}"
            if prefer_simpler else "the lowest AICc among the valid fits")
    lines.append("Rule: " + rule + ".")
    if result["selected"] is None:
        lines.append("No fit passed every validity gate.")
    return "\n".join(lines)


# ----------------------------------------------------------------------
# Errors
# ----------------------------------------------------------------------
def inflated_errors(fit: dict) -> dict:
    """Parameter errors scaled by sqrt(max(1, reduced chi-squared)) -- the Birge ratio /
    PDG scale-factor convention. A fit with measured point errors but a reduced
    chi-squared of 9-880 (noise-only integral errors) otherwise understates the scatter
    about the curve by a factor of 3-30. Returns ``{"scale", "errors", "measured_errors",
    "label"}``; with no measured errors (scatter-based already) the scale is 1."""
    unc = fit.get("uncertainty", {})
    red = float(fit["fit_quality"]["reduced_chi2"])
    scale = math.sqrt(max(1.0, red)) if unc.get("from_measurement_errors") else 1.0
    errors = {k: (v * scale if np.isfinite(v) else v) for k, v in fit["errors"].items()}
    return {"scale": scale, "errors": errors, "measured_errors": dict(fit["errors"]),
            "label": ("1σ (measured errors × √χ²red, Birge ratio)" if scale > 1.0
                      else fit.get("uncertainty", {}).get("short", "1σ"))}


def component_weights_with_errors(fit: dict, inflate: bool = False) -> dict:
    """Fractions ``w_k = I_k / T`` with ``T = sum I + C`` and their 1σ errors from the
    full covariance (``dw_k/dI_j = (delta_kj T - I_k) / T^2``; C enters ``T`` only).
    Returns ``{"names", "weights", "errors", "offset_fraction", "offset_error"}``."""
    spec = fit["spec"]
    names = list(fit["param_names"])
    amps = [nm for nm in names if nm.startswith("I")]
    has_c = "C" in names
    members = amps + (["C"] if has_c else [])
    idx = {nm: i for i, nm in enumerate(fit["free_names"])}
    cov = np.asarray(fit["pcov"], dtype=float)
    if inflate:
        cov = cov * inflated_errors(fit)["scale"] ** 2
    values = np.array([float(fit["params"][m]) for m in members])
    total = float(values.sum())
    out_w, out_e = [], []
    for k, name in enumerate(members):
        grad = np.array([((1.0 if k == j else 0.0) * total - values[k]) / total ** 2
                         for j in range(len(members))])
        var = 0.0
        for a, ma in enumerate(members):
            for c, mc in enumerate(members):
                if ma in idx and mc in idx:
                    var += grad[a] * grad[c] * cov[idx[ma], idx[mc]]
        out_w.append(values[k] / total)
        out_e.append(math.sqrt(max(var, 0.0)))
    result = {"names": members, "weights": dict(zip(members, out_w)),
              "errors": dict(zip(members, out_e)), "offset_fraction": None, "offset_error": None}
    if has_c:
        result["offset_fraction"], result["offset_error"] = out_w[-1], out_e[-1]
    return result


def parameter_ratio(fit: dict, numerator: str, denominator: str, inflate: bool = False) -> dict:
    """``p_num / p_den`` with its 1σ error from the full covariance (e.g. ``D_membrane /
    D_water``, which cancels temperature and gradient-calibration offsets between sessions)."""
    free = list(fit["free_names"])
    cov = np.asarray(fit["pcov"], dtype=float)
    if inflate:
        cov = cov * inflated_errors(fit)["scale"] ** 2
    a, b = float(fit["params"][numerator]), float(fit["params"][denominator])
    ratio = a / b
    grad = {numerator: 1.0 / b, denominator: -a / b ** 2}
    var = sum(grad[x] * grad[y] * cov[free.index(x), free.index(y)]
              for x in grad for y in grad if x in free and y in free)
    return {"ratio": ratio, "error": math.sqrt(max(var, 0.0))}


def combine_measurements(values, errors=None, groups=None) -> dict:
    """Combine per-sample fitted values (e.g. D at each cross-link density).

    ``groups`` gives each measurement's *sample*; repeats of one sample are averaged first
    (error ``sqrt(sum sigma_i^2) / n``, or the SEM of the repeats when larger), then the
    samples are averaged with error ``sqrt(SEM^2 + (sqrt(sum sigma_i^2)/n)^2)``, SEM being the
    standard error of the sample means. For a single sample there is no SEM: the pooled
    relative SD of the sample groups with n >= 2 is used instead (and said so in ``note``).
    Returns ``{"mean", "error", "n_samples", "note"}``."""
    values = np.asarray(values, dtype=float)
    errors = np.zeros_like(values) if errors is None else np.asarray(errors, dtype=float)
    if groups is None:
        groups = np.arange(values.size)
    groups = np.asarray(groups)
    labels = list(dict.fromkeys(groups.tolist()))
    means, sig = [], []
    pooled_num, pooled_den = 0.0, 0
    for lab in labels:
        m = groups == lab
        v = values[m]
        means.append(float(v.mean()))
        sig.append(float(np.sqrt(np.sum(errors[m] ** 2)) / v.size))
        if v.size >= 2 and v.mean() != 0:
            pooled_num += float(np.var(v, ddof=1)) / float(v.mean()) ** 2 * (v.size - 1)
            pooled_den += v.size - 1
    means, sig = np.asarray(means), np.asarray(sig)
    n = means.size
    mean = float(means.mean())
    fit_part = float(np.sqrt(np.sum(sig ** 2)) / n)
    note = ""
    if n >= 2:
        sem = float(np.std(means, ddof=1) / math.sqrt(n))
    elif pooled_den:
        rel = math.sqrt(pooled_num / pooled_den)
        sem = rel * abs(mean)
        note = "single sample: spread taken from the pooled relative SD of groups with n ≥ 2"
    else:
        sem = 0.0
        note = "single measurement: only the fit error is available"
    return {"mean": mean, "error": math.sqrt(sem ** 2 + fit_part ** 2), "n_samples": n, "note": note}


# ----------------------------------------------------------------------
# PFG quality checks
# ----------------------------------------------------------------------
_HHS = (1.635e-8, 215.05, 2.063)      # Holz, Heil & Sacco, PCCP 2000, 2, 4740 (H2O)


def free_water_D(T_K) -> np.ndarray:
    """Self-diffusion of H2O (m^2/s): ``D0 [(T/Ts) - 1]^gamma`` (Holz, Heil & Sacco 2000)."""
    d0, ts, g = _HHS
    return d0 * (np.asarray(T_K, dtype=float) / ts - 1.0) ** g


def water_temperature_from_D(D) -> np.ndarray:
    """The temperature (K) at which free H2O has diffusion coefficient ``D`` -- the
    temperature a measured water component *implies* (catches a mislabelled session)."""
    d0, ts, g = _HHS
    return ts * ((np.asarray(D, dtype=float) / d0) ** (1.0 / g) + 1.0)


def residual_d2o_limit(T_K: float = 298.15) -> float:
    """The D2O self-diffusion coefficient (1.872e-9 m^2/s at 25 °C), scaled with T through
    the H2O law: a water component below this in an exchanged sample may be residual D2O."""
    return 1.872e-9 * float(free_water_D(T_K) / free_water_D(298.15))


def tail_residual(fit: dict, n_last: int = 3) -> dict:
    """The mean normalised residual ``(y - fit)/sigma`` of the last ``n_last`` points
    (sigma = supplied errors, else the rms residual). A large positive value means a
    non-decaying plateau is missing from the model: the fit needs ``C``."""
    y, yf = np.asarray(fit["y"], dtype=float), np.asarray(fit["y_fit"], dtype=float)
    err = fit.get("y_err")
    sigma = np.asarray(err, dtype=float) if err is not None else np.full_like(
        y, float(np.sqrt(np.mean((y - yf) ** 2))) or 1.0)
    z = (y - yf) / sigma
    value = float(np.mean(z[-int(n_last):]))
    return {"value": value, "n_last": int(n_last),
            "needs_offset": bool(value > 3.0 and "C" not in fit["param_names"])}
