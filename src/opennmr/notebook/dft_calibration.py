"""
Shielding → shift scales from reference compounds (explicit user request,
2026-10-07): several reference calculations, each with the experimental
shift you measured for it, give a straight line of calculated isotropic
shielding against experimental shift for one element,

    σ_calc = a + b δ_exp,

and the sample's shifts are read off it, δ = (σ − a) / b. With one
reference the slope is fixed at −1 (the plain δ = δ_ref + σ_ref − σ);
with two or more it is fitted, or fixed at a value you choose.

    from opennmr.notebook import dft_nmr, dft_calibration as dc
    ala = dft_nmr.read_calculation("L-alanine.magres")
    gly = dft_nmr.read_calculation("glycine.magres")
    pts = [dc.reference_point(ala, ["H1", "H2", "H3"], delta_exp=1.4, name="Ala CH3"),
           dc.reference_point(ala, ["H4"], delta_exp=3.9, name="Ala CH"), ...]
    cal_H = dc.fit_calibration(pts)
    dc.plot_calibration(cal_H)
    dc.save_calibration(cal_H, "1H, CASTEP PBE, amino acids")
    model = dft_nmr.dft_sites(sample, calibration={"H": cal_H, "Na": cal_Na}, element="H")

Choices made, and why
---------------------
* **Which way round.** σ_calc is regressed on δ_exp (y on x), as asked,
  and that is also the statistically sound direction: the experimental
  shifts are known to a few hundredths of a ppm, the calculated
  shieldings scatter by tenths to ppm, so the error is in y.
* **Uncertainty of a predicted shift.** For a sample shielding σ₀, the
  shift is δ₀ = (σ₀ − a)/b with standard uncertainty (inverse
  prediction, e.g. Draper & Smith, *Applied Regression Analysis*, §3.2)

      s_δ = (s / |b|) √(1 + 1/n + (δ₀ − δ̄)² / S_xx)

  where s is the residual standard deviation of σ about the line, δ̄ the
  mean reference shift and S_xx = Σ(δ_i − δ̄)². The leading 1 is the
  scatter of a *new* compound about the line, which dominates; it is
  what makes this an honest error bar rather than the error of the line.
  With the slope fixed the last term is absent and s has n − 1 degrees
  of freedom. Two references with a fitted slope (or one with a fixed
  slope) define the line exactly: no uncertainty is given.
* **Groups of atoms.** A reference point is the *mean* isotropic shielding
  of the atoms you choose: symmetry copies of one site, or atoms that
  exchange fast on the NMR time scale (the three H of a CH₃ or NH₃⁺
  group) and so give one line.
* **What must match.** Shieldings from different codes, functionals,
  cutoffs, pseudopotentials or basis sets are not on one scale, and
  neither are VASP shieldings with and without the G=0 term. Each point
  keeps its calculation's method; mismatches are reported, between the
  points and between the calibration and the sample.

Saved calibrations live in ``~/.opennmr/dft_calibrations.json`` (each
keeps its points, so it stands without the files).
"""

from __future__ import annotations

import datetime as _dt
import json
import os

import numpy as np

from . import dft_nmr
from ..core.atomic_write import write_json_atomic

# ============================================================
# Reference points
# ============================================================


def _iso(sigma) -> float:
    return float(np.trace(dft_nmr.symmetric(sigma))) / 3.0


def reference_point(calc, labels=None, element=None, delta_exp=None, name=None,
                    use: bool = True) -> dict:
    """One point of a calibration: the mean isotropic shielding of
    ``labels`` (or of every ``element`` atom) in ``calc``, and the
    experimental shift ``delta_exp`` (ppm; may be None until known)."""
    if isinstance(calc, (str, os.PathLike)):
        calc = dft_nmr.read_calculation(calc)
    atoms = [a for a in dft_nmr.select_atoms(calc, element, labels) if a["sigma"] is not None]
    if not atoms:
        raise ValueError("None of these atoms has a shielding tensor.")
    elements = {a["species"] for a in atoms}
    if len(elements) > 1:
        raise ValueError("A reference point is one element (got "
                         + ", ".join(sorted(elements)) + ").")
    values = [_iso(a["sigma"]) for a in atoms]
    labs = [a["label"] for a in atoms]
    return {"name": name or f"{os.path.basename(calc['path'])} "
                            f"{dft_nmr._group_label(labs)}",
            "file": calc["path"], "element": elements.pop(), "labels": labs,
            "sigma": float(np.mean(values)), "sigma_values": values,
            "delta_exp": None if delta_exp is None else float(delta_exp), "use": bool(use),
            "code": calc["code"], "method": dict(calc.get("method") or {}),
            "g0": calc.get("g0", "none")}


def typed_point(element, sigma, delta_exp=None, name=None, use: bool = True, method=None,
                g0="none") -> dict:
    """A reference point from a shielding you have (a published value, or
    a saved point whose file is gone) rather than a file."""
    return {"name": name or f"{element} σ = {float(sigma):.3f}", "file": "", "element": element,
            "labels": [], "sigma": float(sigma), "sigma_values": [float(sigma)],
            "delta_exp": None if delta_exp is None else float(delta_exp), "use": bool(use),
            "code": (method or {}).get("code", ""), "method": dict(method or {}), "g0": g0}


def refresh_point(point, calc) -> dict:
    """``point`` with its shielding taken again from ``calc`` (the same
    file re-read, e.g. with the G=0 term now in or out); name, δ_exp and
    use are kept."""
    fresh = reference_point(calc, point["labels"], point["element"])
    return dict(fresh, name=point["name"], delta_exp=point.get("delta_exp"),
                use=point.get("use", True))


def equivalent_groups(calc, element, tol: float = 0.01) -> list[list[str]]:
    """``element``'s atoms (with a shielding) grouped by equal principal
    shieldings (within ``tol`` ppm): symmetry copies of one site. Groups
    are in order of increasing σ_iso."""
    rows = []
    for a in calc["atoms"]:
        if a["species"] == element and a["sigma"] is not None:
            rows.append((np.sort(np.linalg.eigvalsh(dft_nmr.symmetric(a["sigma"]))), a["label"]))
    groups = []
    for w, label in sorted(rows, key=lambda r: float(np.mean(r[0]))):
        for g in groups:
            if np.max(np.abs(g[0] - w)) <= tol:
                g[1].append(label)
                break
        else:
            groups.append((w, [label]))
    return [labels for _w, labels in groups]


def suggest_points(calc, element, tol: float = 0.01) -> list[dict]:
    """One point per group of equivalent ``element`` sites, with no
    experimental shift yet: the rows a user fills in (and merges, for a
    CH₃)."""
    return [reference_point(calc, labels) for labels in equivalent_groups(calc, element, tol)]


def merge_points(points, name=None) -> dict:
    """One point from several of the same file and element (e.g. the
    three groups of a CH₃): the mean over all their atoms."""
    points = list(points)
    if len(points) < 2:
        raise ValueError("Choose at least two points to merge.")
    if len({p["file"] for p in points}) > 1 or len({p["element"] for p in points}) > 1:
        raise ValueError("Only points of one file and one element can be merged.")
    labels = [lab for p in points for lab in p["labels"]]
    values = [v for p in points for v in p["sigma_values"]]
    known = [p["delta_exp"] for p in points if p["delta_exp"] is not None]
    out = dict(points[0])
    out.update(name=name or f"{os.path.basename(points[0]['file'])} "
                            f"{dft_nmr._group_label(labels)}",
               labels=labels, sigma_values=values, sigma=float(np.mean(values)),
               delta_exp=known[0] if len(set(known)) == 1 else None,
               use=all(p["use"] for p in points))
    return out


# ============================================================
# Fitting
# ============================================================


def _usable(points, element=None):
    return [p for p in points if p.get("use", True) and p.get("delta_exp") is not None
            and (element is None or p["element"] == element)]


def fit_calibration(points, element=None, slope=None, name=None) -> dict:
    """The line σ_calc = a + b δ_exp through the usable points (ticked,
    with an experimental shift) of ``element``.

    ``slope`` : None fits it (two or more points; one point fixes it at
    −1); a number fixes it there. Returns a calibration: ``intercept``
    (a), ``slope`` (b), their standard errors (None where undetermined),
    ``n``, ``dof``, ``s_sigma`` (residual SD of σ), ``r2``, ``rmse_delta``
    (RMS of δ_calc − δ_exp over the points), the points with their
    residuals, the method, and plain-language ``notes``."""
    pts = _usable(points, element)
    if not pts:
        raise ValueError("No usable reference: tick at least one and give its experimental "
                         "shift.")
    elements = {p["element"] for p in pts}
    if len(elements) > 1:
        raise ValueError("One calibration is one element (got " + ", ".join(sorted(elements))
                         + "): choose the element.")
    element = elements.pop()
    x = np.array([p["delta_exp"] for p in pts], dtype=float)
    y = np.array([p["sigma"] for p in pts], dtype=float)
    n = len(pts)
    notes = []
    free = slope is None and n >= 2
    xbar, ybar = float(x.mean()), float(y.mean())
    sxx = float(np.sum((x - xbar) ** 2))
    if slope is None and n == 1:
        slope = -1.0
        notes.append("One reference: the slope is fixed at −1 (δ = δ_ref + σ_ref − σ). Add "
                     "references across the shift range to fit it.")
    if free:
        if sxx < 1e-12:
            raise ValueError("Every reference has the same experimental shift: the slope is "
                             "undetermined. Fix the slope, or add references at other shifts.")
        b = float(np.sum((x - xbar) * (y - ybar)) / sxx)
        a = ybar - b * xbar
        dof = n - 2
    else:
        b = float(slope)
        a = float(np.mean(y - b * x))
        dof = n - 1
    if abs(b) < 1e-9:
        raise ValueError("The slope is zero: shielding does not change with shift across these "
                         "references, so no shift can be read off.")
    resid = y - (a + b * x)
    sse = float(np.sum(resid ** 2))
    s = float(np.sqrt(sse / dof)) if dof > 0 else None
    se_b = s / np.sqrt(sxx) if (free and s is not None) else None
    if s is None:
        se_a = None
    elif free:
        se_a = s * np.sqrt(1.0 / n + xbar ** 2 / sxx)
    else:
        se_a = s / np.sqrt(n)
    sst = float(np.sum((y - ybar) ** 2))
    r2 = (1.0 - sse / sst) if (n >= 3 and sst > 0 and free) else None
    d_calc = (y - a) / b
    d_resid = d_calc - x
    rmse = float(np.sqrt(np.mean(d_resid ** 2)))
    if free and dof == 0:
        notes.append("Two references define the line exactly: no uncertainty can be given. A "
                     "third shows how well the line holds.")
    if b > 0:
        notes.append(f"The slope is positive ({b:.3f}): shielding should fall as the shift "
                     "rises. Check the experimental shifts and which atoms each point uses.")
    elif free and abs(b + 1.0) > 0.2:
        notes.append(f"The slope is {b:.3f}, far from −1. Calculated shielding scales are often "
                     "off by 5–15 %, but this much may mean a misassigned reference or one "
                     "outside the range of the others.")
    if se_b is not None and abs(b + 1.0) < 2 * se_b and free:
        notes.append(f"The slope is −1 within its uncertainty (±{se_b:.3f}): a slope fixed at −1 "
                     "fits as well, with one fewer parameter.")
    if s is not None and n >= 4:
        big = [p["name"] for p, r in zip(pts, d_resid) if abs(r) > 3.0 * s / abs(b)]
        if big:
            notes.append("Far off the line (more than 3 s): " + ", ".join(big)
                         + ". Check its assignment, the atoms averaged, or its δ_exp.")
    notes += consistency_notes(pts)
    used = []
    for p, r, dc_ in zip(pts, d_resid, d_calc):
        q = dict(p)
        q.update(delta_calc=float(dc_), residual=float(r))
        used.append(q)
    cal = {"name": name or "", "element": element, "intercept": float(a), "slope": float(b),
           "intercept_err": None if se_a is None else float(se_a),
           "slope_err": None if se_b is None else float(se_b), "free_slope": bool(free),
           "n": n, "dof": dof, "s_sigma": s, "r2": r2, "rmse_delta": rmse,
           "x_mean": xbar, "Sxx": sxx, "x_range": (float(x.min()), float(x.max())),
           "points": used, "method": dict(pts[0].get("method") or {}),
           "g0": sorted({p.get("g0", "none") for p in pts}), "notes": notes, "manual": False}
    return cal


def consistency_notes(points) -> list[str]:
    """Method and G=0 differences between reference points."""
    notes = []
    if not points:
        return notes
    first = points[0]
    element = first["element"]
    for p in points[1:]:
        diffs = dft_nmr.method_differences(first.get("method"), p.get("method"), element)
        if diffs:
            notes.append(f"{p['name']} and {first['name']} differ in " + "; ".join(diffs)
                         + ": their shieldings are not on one scale.")
    g0 = {p.get("g0", "none") for p in points} - {"none"}
    if "excluded" in g0 and ("included" in g0 or "fixed" in g0):
        notes.append("Some references include the G=0 term and some do not: it shifts every "
                     "shielding by several ppm, so the points are not on one scale.")
    return notes


def manual_calibration(element, sigma_ref, delta_ref=0.0, slope_m=1.0) -> dict:
    """A calibration from typed numbers: δ = δ_ref + m (σ_ref − σ), i.e.
    σ = (σ_ref + δ_ref/m) − δ/m."""
    m = float(slope_m)
    if m == 0:
        raise ValueError("m cannot be 0.")
    return {"name": "typed", "element": element,
            "intercept": float(sigma_ref) + float(delta_ref) / m, "slope": -1.0 / m,
            "intercept_err": None, "slope_err": None, "free_slope": False, "n": 0, "dof": 0,
            "s_sigma": None, "r2": None, "rmse_delta": None, "x_mean": float(delta_ref),
            "Sxx": 0.0, "x_range": None, "points": [], "method": {}, "g0": [],
            "notes": [f"Typed: σ_ref = {float(sigma_ref):.4f} ppm, δ_ref = {float(delta_ref):.3f} "
                      f"ppm, m = {m:g}."], "manual": True,
            "typed": {"sigma_ref": float(sigma_ref), "delta_ref": float(delta_ref), "m": m}}


def shift_from_sigma(cal, sigma):
    """δ = (σ − a)/b: for a number (isotropic shielding) or a 3×3 tensor
    (its symmetric part)."""
    a, b = cal["intercept"], cal["slope"]
    sigma = np.asarray(sigma, dtype=float)
    if sigma.shape == (3, 3):
        return (dft_nmr.symmetric(sigma) - a * np.eye(3)) / b
    return (sigma - a) / b


def shift_uncertainty(cal, delta) -> float | None:
    """Standard uncertainty of a shift read off the line (see the module
    docstring); None when the references cannot give one."""
    s = cal.get("s_sigma")
    if s is None or not cal.get("n"):
        return None
    term = 1.0 + 1.0 / cal["n"]
    if cal.get("free_slope") and cal.get("Sxx"):
        term += (float(delta) - cal["x_mean"]) ** 2 / cal["Sxx"]
    return float(s / abs(cal["slope"]) * np.sqrt(term))


def _pm(value, err, digits):
    return f"{value:.{digits}f}" + (f" ± {err:.{digits}f}" if err is not None else "")


def equation_text(cal) -> str:
    """σ_calc = a + b δ_exp (with errors), and the inverse used on the
    sample."""
    a, b = cal["intercept"], cal["slope"]
    sign = "−" if b < 0 else "+"
    fwd = (f"σ_calc = {_pm(a, cal.get('intercept_err'), 3)} {sign} "
           f"{_pm(abs(b), cal.get('slope_err'), 4)} δ_exp")
    inv_sign = "−" if b < 0 else "+"
    inv = f"δ = {-a / b:.3f} {inv_sign} {abs(1.0 / b):.4f} σ"
    return f"{fwd};  so {inv}"


def summary_text(cal) -> str:
    """One line: element, equation, n, fit quality."""
    if cal.get("manual"):
        return f"{cal['element']}: typed scale. {equation_text(cal)}"
    bits = [f"{cal['element']}: {equation_text(cal)}", f"n = {cal['n']}"]
    bits.append("slope fitted" if cal["free_slope"]
                else f"slope fixed at {cal['slope']:g}".replace("-", "−"))
    if cal.get("r2") is not None:
        bits.append(f"R² = {cal['r2']:.4f}")
    if cal.get("s_sigma") is not None:
        bits.append(f"s = {cal['s_sigma'] / abs(cal['slope']):.3g} ppm in δ")
    if cal.get("rmse_delta") is not None and cal["n"] > 1:
        bits.append(f"RMS(δ_calc − δ_exp) = {cal['rmse_delta']:.3g} ppm")
    return "; ".join(bits) + "."


def sample_notes(cal, sample, deltas=()) -> list[str]:
    """What to check when using ``cal`` on ``sample``: method and G=0
    differences, and shifts outside the references' range."""
    notes = []
    if cal.get("manual"):
        return notes
    diffs = dft_nmr.method_differences(cal.get("method"), sample.get("method"), cal["element"])
    if diffs:
        notes.append("The calibration's references and the sample differ in " + "; ".join(diffs)
                     + ": the calibration does not carry over.")
    cal_g0 = set(cal.get("g0") or []) - {"none"}
    s_g0 = sample.get("g0", "none")
    if s_g0 != "none" and cal_g0:
        if (s_g0 == "excluded") != ("excluded" in cal_g0):
            notes.append(f"The sample's G=0 term is {s_g0} but the calibration's references' "
                         f"is {', '.join(sorted(cal_g0))}: not on one scale.")
    lo_hi = cal.get("x_range")
    if lo_hi and cal["n"] >= 2 and len(deltas):
        lo, hi = lo_hi
        pad = 0.1 * (hi - lo)
        out = [d for d in deltas if d < lo - pad or d > hi + pad]
        if out:
            notes.append(f"{len(out)} shift(s) lie outside the references' range ({lo:.1f} to "
                         f"{hi:.1f} ppm): an extrapolation, less certain than the error bars "
                         "say if the line is not straight out there.")
    return notes


# ============================================================
# Library of saved calibrations
# ============================================================


def library_path() -> str:
    return os.path.join(os.path.expanduser("~"), ".opennmr", "dft_calibrations.json")


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def load_library(path=None) -> list[dict]:
    """Every saved calibration (newest first)."""
    path = path or library_path()
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    cals = data.get("calibrations", []) if isinstance(data, dict) else []
    for c in cals:
        if c.get("x_range") is not None:
            c["x_range"] = tuple(c["x_range"])
    return cals


def save_calibration(cal, name, path=None) -> dict:
    """Save ``cal`` under ``name`` (replacing one of the same name and
    element). Returns the saved entry."""
    if not str(name).strip():
        raise ValueError("Give the calibration a name.")
    path = path or library_path()
    cals = [c for c in load_library(path)
            if not (c.get("name") == name and c.get("element") == cal["element"])]
    entry = _jsonable(dict(cal, name=str(name),
                           saved=_dt.datetime.now().isoformat(timespec="seconds")))
    cals.insert(0, entry)
    write_json_atomic(path, {"version": 1, "calibrations": cals}, indent=1, ensure_ascii=False)
    return entry


def delete_calibration(name, element=None, path=None) -> bool:
    path = path or library_path()
    cals = load_library(path)
    keep = [c for c in cals if not (c.get("name") == name
                                    and (element is None or c.get("element") == element))]
    if len(keep) == len(cals):
        return False
    write_json_atomic(path, {"version": 1, "calibrations": keep}, indent=1, ensure_ascii=False)
    return True


def find_calibration(name, element=None, path=None) -> dict:
    for c in load_library(path):
        if c.get("name") == name and (element is None or c.get("element") == element):
            return c
    raise KeyError(f"No saved calibration {name!r}" + (f" for {element}" if element else "")
                   + ".")


# ============================================================
# Pictures and tables
# ============================================================


def _fit_lines(cal) -> list[str]:
    """The fit's numbers as mathtext lines for a plot."""
    a, b = cal["intercept"], cal["slope"]
    ea, eb = cal.get("intercept_err"), cal.get("slope_err")
    minus = "\u2212"
    lines = [r"$\mathit{\sigma}_\mathrm{calc} = \mathit{a} + \mathit{b}\,"
             r"\mathit{\delta}_\mathrm{exp}$",
             r"$\mathit{a}$ = " + f"{a:.2f}" + (f" ± {ea:.2f}" if ea is not None else "")
             + " ppm",
             r"$\mathit{b}$ = " + f"{b:.3f}".replace("-", minus)
             + (f" ± {eb:.3f}" if eb is not None else ("" if cal["free_slope"] else " (fixed)"))]
    if cal.get("r2") is not None:
        lines.append(r"$\mathit{R}^2$ = " + f"{cal['r2']:.4f}")
    if cal.get("s_sigma") is not None:
        lines.append(r"$\mathit{s}_\delta$ = " + f"{cal['s_sigma'] / abs(b):.2f} ppm")
    lines.append(r"$\mathit{n}$ = " + str(cal["n"]))
    return lines


def _place_labels(ax, xy, names, fontsize, avoid_line=None):
    """Put each name next to its point where it overlaps nothing already
    placed (labels, points, the fitted line), trying eight directions at
    two distances; the least-bad spot otherwise."""
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    to_px = ax.transData.transform
    pts_px = to_px(np.asarray(xy, dtype=float))
    taken = []
    r_pt = 5.0
    for (x0, y0), (px, py) in zip(xy, pts_px):
        taken.append((px - r_pt, py - r_pt, px + r_pt, py + r_pt))
    for t in ax.texts:                       # e.g. the fit's numbers
        bb = t.get_window_extent(renderer)
        taken.append((bb.x0 - 3, bb.y0 - 3, bb.x1 + 3, bb.y1 + 3))
    line_px = None
    if avoid_line is not None:
        xs = np.linspace(*ax.get_xlim(), 200)
        line_px = to_px(np.column_stack([xs, avoid_line(xs)]))
    axbox = ax.get_window_extent(renderer)
    dirs = [(1, 1), (1, -1), (-1, 1), (-1, -1), (1, 0), (-1, 0), (0, 1), (0, -1)]
    texts = []
    for (x0, y0), name in zip(xy, names):
        best = None
        for dist in (6, 14):
            for dx, dy in dirs:
                ha = "left" if dx > 0 else ("right" if dx < 0 else "center")
                va = "bottom" if dy > 0 else ("top" if dy < 0 else "center")
                t = ax.annotate(name, (x0, y0), xytext=(dx * dist, dy * dist),
                                textcoords="offset points", ha=ha, va=va, fontsize=fontsize)
                bb = t.get_window_extent(renderer)
                box = (bb.x0 - 1, bb.y0 - 1, bb.x1 + 1, bb.y1 + 1)
                cost = 0.0
                for o in taken:
                    w = min(box[2], o[2]) - max(box[0], o[0])
                    h = min(box[3], o[3]) - max(box[1], o[1])
                    if w > 0 and h > 0:
                        cost += w * h
                if line_px is not None:
                    inside = ((line_px[:, 0] > box[0]) & (line_px[:, 0] < box[2])
                              & (line_px[:, 1] > box[1]) & (line_px[:, 1] < box[3]))
                    cost += 40.0 * int(np.sum(inside))
                out = (max(0, axbox.x0 - box[0]) + max(0, box[2] - axbox.x1)
                       + max(0, axbox.y0 - box[1]) + max(0, box[3] - axbox.y1))
                cost += 500.0 * out + 0.01 * dist
                t.remove()
                if best is None or cost < best[0]:
                    best = (cost, (dx * dist, dy * dist), ha, va, box)
                if cost < 0.1 * dist + 1e-9:
                    break
            if best[0] < 1.0:
                break
        _c, offset, ha, va, box = best
        texts.append(ax.annotate(name, (x0, y0), xytext=offset, textcoords="offset points",
                                 ha=ha, va=va, fontsize=fontsize))
        taken.append(box)
    return texts


def plot_calibration(cal, figure=None, residuals: bool = True, annotate: bool = True,
                     fontsize: float = 11.0, color=None, line_color=None, marker_size=36.0):
    """σ_calc against δ_exp with the fitted line and its numbers; below it
    (``residuals``), δ_calc − δ_exp of each reference. Point names are
    placed clear of each other, the points and the line; the numbers go
    in the corner the line leaves empty. Returns ``(figure, notes)``."""
    import matplotlib.pyplot as plt
    if cal.get("manual") or not cal.get("points"):
        raise ValueError("A typed scale has no reference points to plot.")
    fig = figure if figure is not None else plt.figure(figsize=(6.0, 5.2 if residuals else 4.2))
    fig.clear()
    if residuals:
        gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.08)
        ax = fig.add_subplot(gs[0])
        axr = fig.add_subplot(gs[1], sharex=ax)
    else:
        ax, axr = fig.add_subplot(111), None
    cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["#0072B2", "#D55E00"])
    color = color or cycle[0]
    line_color = line_color or (cycle[1] if len(cycle) > 1 else "k")
    pts = cal["points"]
    x = np.array([p["delta_exp"] for p in pts])
    y = np.array([p["sigma"] for p in pts])
    lo, hi = float(x.min()), float(x.max())
    pad = 0.08 * (hi - lo) if hi > lo else 1.0
    xs = np.linspace(lo - pad, hi + pad, 50)
    line = (lambda v: cal["intercept"] + cal["slope"] * np.asarray(v))
    ax.plot(xs, line(xs), color=line_color, lw=1.4, zorder=1)
    ax.scatter(x, y, s=marker_size, color=color, edgecolor="none", zorder=2)
    ax.set_xlim(lo - pad, hi + pad)
    ylo, yhi = float(min(y.min(), line(xs).min())), float(max(y.max(), line(xs).max()))
    ypad = 0.1 * (yhi - ylo) if yhi > ylo else 1.0
    ax.set_ylim(ylo - ypad, yhi + ypad)
    ax.set_ylabel(_italic_label(r"$\mathit{\sigma}_\mathrm{calc}$ (ppm)"), fontsize=fontsize)
    # The line runs corner to corner; its numbers go in an empty corner.
    if cal["slope"] < 0:
        pos = (0.03, 0.04, "left", "bottom")
    else:
        pos = (0.97, 0.04, "right", "bottom")
    ax.text(pos[0], pos[1], "\n".join(_fit_lines(cal)), transform=ax.transAxes, ha=pos[2],
            va=pos[3], fontsize=fontsize * 0.8, linespacing=1.35)
    xlabel = r"$\mathit{\delta}_\mathrm{exp}$ (ppm)"
    if axr is not None:
        r = np.array([p["residual"] for p in pts])
        axr.axhline(0.0, color=line_color, lw=1.0)
        axr.scatter(x, r, s=marker_size * 0.7, color=color, edgecolor="none")
        lim = max(float(np.max(np.abs(r))) * 1.3, 0.05)
        axr.set_ylim(-lim, lim)
        axr.set_ylabel(_italic_label(r"$\mathit{\delta}_\mathrm{calc} - \mathit{\delta}_\mathrm{exp}$"
                       "\n(ppm)"), fontsize=fontsize * 0.85)
        axr.set_xlabel(_italic_label(xlabel), fontsize=fontsize)
        plt.setp(ax.get_xticklabels(), visible=False)
    else:
        ax.set_xlabel(_italic_label(xlabel), fontsize=fontsize)
    if annotate:
        _place_labels(ax, list(zip(x, y)), [p["name"] for p in pts], fontsize * 0.72,
                      avoid_line=line)
    return fig, list(cal.get("notes") or [])


def points_table(cal_or_points) -> list[list[str]]:
    """Rows: name, atoms, σ_calc, δ_exp, δ_calc, residual."""
    pts = cal_or_points["points"] if isinstance(cal_or_points, dict) else cal_or_points
    rows = []
    for p in pts:
        rows.append([p["name"], dft_nmr._group_label(p["labels"]), f"{p['sigma']:.4f}",
                     "—" if p.get("delta_exp") is None else f"{p['delta_exp']:.3f}",
                     "—" if p.get("delta_calc") is None else f"{p['delta_calc']:.3f}",
                     "—" if p.get("residual") is None else f"{p['residual']:+.3f}"])
    return rows


def calibration_to_python(points, element, slope=None, include_g0=True, var="cal",
                          header: bool = True) -> str:
    """Notebook code reproducing a calibration from its points (files that
    no longer exist become typed points with their stored σ)."""
    points = [p for p in points if p["element"] == element]
    files = {}
    for p in points:
        if p.get("file") and os.path.exists(p["file"]):
            files.setdefault(p["file"], f"ref{len(files) + 1}")
    lines = (["from opennmr.notebook import dft_nmr, dft_calibration as dc"] if header else [])
    lines += [f"{v} = dft_nmr.read_calculation({path!r}, include_g0={bool(include_g0)})"
              for path, v in files.items()]
    lines.append(f"points_{element} = [")
    for p in points:
        if p.get("file") in files:
            lines.append(f"    dc.reference_point({files[p['file']]}, {p['labels']!r}, "
                         f"delta_exp={p.get('delta_exp')!r}, name={p['name']!r}, "
                         f"use={bool(p.get('use', True))}),")
        else:
            lines.append(f"    dc.typed_point({element!r}, {p['sigma']!r}, "
                         f"delta_exp={p.get('delta_exp')!r}, name={p['name']!r}, "
                         f"use={bool(p.get('use', True))}),")
    lines.append("]")
    lines.append(f"{var} = dc.fit_calibration(points_{element}, element={element!r}, "
                 f"slope={slope!r})")
    lines.append(f"print(dc.summary_text({var}))")
    return "\n".join(lines) + "\n"



def shift_table(sample, calibrations) -> list[dict]:
    """Every atom of every calibrated element in ``sample``: label,
    element, σ_iso, δ_iso and its uncertainty -- one table across nuclei
    (e.g. ¹H on an alanine scale and ²³Na on a NaCl one)."""
    if isinstance(sample, (str, os.PathLike)):
        sample = dft_nmr.read_calculation(sample)
    if "intercept" in calibrations:
        calibrations = {calibrations["element"]: calibrations}
    rows = []
    for a in sample["atoms"]:
        cal = calibrations.get(a["species"])
        if cal is None or a["sigma"] is None:
            continue
        sigma = _iso(a["sigma"])
        delta = float(shift_from_sigma(cal, sigma))
        rows.append({"label": a["label"], "element": a["species"], "sigma_iso": sigma,
                     "delta_iso": delta, "delta_err": shift_uncertainty(cal, delta)})
    return rows


def _italic_label(text):
    """Quantity symbols italic (plotting.labels.italicise_symbols)."""
    from opennmr.plotting.labels import italicise_symbols
    return italicise_symbols(text)
