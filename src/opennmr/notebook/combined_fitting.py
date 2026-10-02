"""
Fit spectra with Combined-Hamiltonian sites: any mix of CSA,
quadrupolar and J interactions per site, static or MAS -- the model of
:mod:`opennmr.notebook.combined_hamiltonian`, fitted by least squares.

Workflow
--------

    sites = [Site("27Al", 10, label="Al1").add_quadrupolar(3.0, 0.3).add_csa(40, 0.5),
             Site("27Al", -5, label="Al2").add_quadrupolar(5.5, 0.6)]
    fit = fit_combined(results, sites, mas_Hz=15e3, transitions="ct")
    print_combined_fit_report(fit)
    plot_combined_fit(fit, nuclide="27Al")
    fit["sites"]          # fitted Site objects -- simulate, plot or refit them

or, from a Bruker folder, :func:`NMRCombinedFitWorkflow`.

Parameters
----------
Each site's fittable parameters are named as below; which exist depends
on the interactions the site has:

    delta_iso, amount, fwhm       every site (amount = relative number of nuclei)
    eta                           pseudo-Voigt broadening only (Lorentzian fraction)
    aniso, asym                   CSA (Delta/eta or span/skew, per its convention)
    alpha, beta, gamma            CSA Euler angles in the EFG frame (site has both)
    C_Q, eta_Q                    quadrupolar (C_Q in MHz)
    J1, J2, ...                   the site's J groups, in the order they were added

By default every one of them is free **except the Euler angles**, which
a single 1D lineshape rarely determines -- free them with ``vary``.
``fixed`` removes names from the free set; ``vary`` replaces it. Both
take a list (every site) or ``{site label: [names]}``. ``bounds`` takes
``{name: (lo, hi)}`` (every site) and/or ``{site label: {name: (lo, hi)}}``.

Amounts are started from a non-negative least-squares fit of the
starting shapes to the data (``estimate_amounts=True``), so they need
not be guessed; the reported shares (%) are then numbers of nuclei --
all transitions of every site count, even when only the central
transition is fitted -- provided excitation was uniform (non-selective).
"""

import copy
import csv

import numpy as np
from scipy.optimize import curve_fit, nnls

from .fitting import build_region_mask, estimate_baseline_robust, fit_quality
from .combined_hamiltonian import (Site, as_sites, _site_sticks, _broaden_site,
                                   simulate_combined)
from .simulations import quadrupolar_ct_isotropic_shift_Hz, QUADRUPOLAR_ORDERS
from ..core import fit_progress as _fit_progress

_BROADENING = ("amount", "fwhm", "eta")


# ============================================================
# Parameter access
# ============================================================

def site_parameter_names(site):
    """Every fittable parameter name of ``site`` (a Site or dict)."""
    d = site.to_dict() if isinstance(site, Site) else site
    types = [i["type"] for i in d["interactions"]]
    names = ["delta_iso", "amount", "fwhm"]
    if d.get("lineshape") == "pseudo_voigt":
        names.append("eta")
    if "csa" in types:
        names += ["aniso", "asym"]
        if "quadrupolar" in types:
            names += ["alpha", "beta", "gamma"]
    if "quadrupolar" in types:
        names += ["C_Q", "eta_Q"]
    names += [f"J{k + 1}" for k in range(types.count("jcoupling"))]
    return names


def _find(d, kind, k=0):
    found = [i for i in d["interactions"] if i["type"] == kind]
    return found[k]


def _get(d, name):
    if name in ("delta_iso", "amount", "fwhm", "eta"):
        return float(d.get(name, 0.0))
    if name in ("aniso", "asym"):
        return float(_find(d, "csa")[name])
    if name in ("alpha", "beta", "gamma"):
        return float(_find(d, "csa")["euler"]["alpha beta gamma".split().index(name)])
    if name in ("C_Q", "eta_Q"):
        return float(_find(d, "quadrupolar")[name])
    if name.startswith("J"):
        return float(_find(d, "jcoupling", int(name[1:]) - 1)["J_Hz"])
    raise KeyError(name)


def _set(d, name, value):
    if name in ("delta_iso", "amount", "fwhm", "eta"):
        d[name] = float(value)
    elif name in ("aniso", "asym"):
        _find(d, "csa")[name] = float(value)
    elif name in ("alpha", "beta", "gamma"):
        csa = _find(d, "csa")
        e = list(csa["euler"])
        e["alpha beta gamma".split().index(name)] = float(value)
        csa["euler"] = tuple(e)
    elif name in ("C_Q", "eta_Q"):
        _find(d, "quadrupolar")[name] = float(value)
    elif name.startswith("J"):
        _find(d, "jcoupling", int(name[1:]) - 1)["J_Hz"] = float(value)
    else:
        raise KeyError(name)


def _default_bounds(d, name):
    if name == "amount":
        return 0.0, np.inf
    if name == "fwhm":
        return 1e-4, np.inf
    if name in ("eta", "eta_Q"):
        return 0.0, 1.0
    if name == "asym":
        return (0.0, 1.0) if _find(d, "csa")["convention"] == "Haeberlen" else (-1.0, 1.0)
    if name == "C_Q":
        return 0.0, np.inf
    if name in ("alpha", "beta", "gamma"):
        return -360.0, 360.0
    return -np.inf, np.inf


def _per_site(spec, labels, what):
    """``spec`` as {label: value} for every site: a list/dict applying to all,
    or keyed by label (or both, for bounds)."""
    out = {lab: None for lab in labels}
    if spec is None:
        return out
    if isinstance(spec, (list, tuple, set)):
        return {lab: list(spec) for lab in labels}
    unknown = [k for k in spec if k not in labels and not (what == "bounds" and isinstance(k, str)
                                                           and not isinstance(spec[k], dict))]
    if unknown:
        raise KeyError(f"{what}: {unknown} are not site labels ({labels}).")
    return {lab: spec.get(lab) for lab in labels}


# ============================================================
# Fit
# ============================================================

def fit_combined(results, sites, vary=None, fixed=None, bounds=None, larmor_Hz=None,
                 mas_Hz=0.0, order="second", transitions="all", rotor_angle="magic",
                 n_zcw=None, fit_regions=None, baseline_regions=None, fit_baseline=False,
                 estimate_amounts=True, max_nfev=2000):
    """
    Fit Combined-Hamiltonian sites to a 1D spectrum.

    Parameters
    ----------
    results : dict with ``x_ppm``, ``real_spectrum``, ``SFO1`` (MHz) --
        NMR1dimport output, or the GUI's bridge.results_from_spectrum.
    sites : Site(s) (or dicts) -- the starting model. Not modified.
    vary, fixed, bounds : which parameters are free and their bounds --
        see the module docstring for names and forms.
    larmor_Hz : default ``results['SFO1'] * 1e6``.
    mas_Hz, order, transitions, rotor_angle, n_zcw : as simulate_combined.
    fit_regions : [(lo, hi), ...] ppm, or None for the whole spectrum.
    baseline_regions, fit_baseline : a linear baseline is estimated from
        ``baseline_regions`` (default the outer 5% of the spectrum) and
        held, unless ``fit_baseline``.
    estimate_amounts : start every site's amount from a non-negative
        least-squares fit of the starting shapes (recommended).

    Returns
    -------
    dict with
        ``sites`` -- fitted Site objects (raw intensity units for amount);
        ``parameters`` -- one row per parameter: site, name, value, error
        (1 sigma), free, lower, upper;
        ``pcov`` (free parameters, in raw units), ``free_names``;
        ``derived`` -- per site: share (%) and, with a quadrupolar
        interaction, P_Q and the CT centre of gravity, with errors;
        ``x``, ``y``, ``y_fit``, ``site_spectra``, ``baseline`` (ascending
        ppm; the fit window is ``mask``);
        ``quality`` (R^2, RMSE, reduced chi^2), ``warnings``, ``log``,
        ``settings``.
    """
    sites = as_sites(sites)
    labels = [s.label for s in sites]
    if len(set(labels)) != len(labels):
        raise ValueError(f"Site labels must be unique, got {labels}.")
    if order not in QUADRUPOLAR_ORDERS:
        raise ValueError(f"order must be one of {QUADRUPOLAR_ORDERS}.")
    if larmor_Hz is None:
        larmor_Hz = float(results["SFO1"]) * 1e6
    larmor_Hz = float(larmor_Hz)
    mas_Hz = float(mas_Hz or 0.0)
    if n_zcw is None:
        n_zcw = 2584 if mas_Hz else 28657
    beta_r = (np.arccos(1 / np.sqrt(3)) if rotor_angle == "magic"
              else np.deg2rad(float(rotor_angle)))
    nuc = results.get("nuclide")
    warnings = []
    if nuc and all(nuc != s.nucleus for s in sites) and nuc.lstrip("0123456789") != "":
        warnings.append(f"The spectrum's nucleus ({nuc}) is not the sites' ({sites[0].nucleus}).")

    # ---- data, window, baseline --------------------------------------------------
    x_full = np.asarray(results["x_ppm"], dtype=float)
    idx = np.argsort(x_full)
    x = x_full[idx]
    y_raw = np.asarray(results["real_spectrum"], dtype=float)[idx]
    scale = float(np.max(np.abs(y_raw))) or 1.0
    y = y_raw / scale
    mask = build_region_mask(x, fit_regions)
    if mask.sum() < 8:
        raise ValueError("The fit window holds fewer than 8 points.")
    x_ref = float(np.mean(x[mask]))
    b0, b1, _ = estimate_baseline_robust(x, y, baseline_regions=baseline_regions, x_ref=x_ref)

    # ---- parameter table ------------------------------------------------------------
    dicts = [s.to_dict() for s in sites]
    vary_m, fixed_m = _per_site(vary, labels, "vary"), _per_site(fixed, labels, "fixed")
    gbounds = {k: v for k, v in (bounds or {}).items() if k not in labels}
    sbounds = {k: v for k, v in (bounds or {}).items() if k in labels}
    table = []
    for i, d in enumerate(dicts):
        names = site_parameter_names(d)
        if vary_m[d["label"]] is not None:
            free_set = set(vary_m[d["label"]])
        else:
            free_set = set(names) - {"alpha", "beta", "gamma"}
        free_set -= set(fixed_m[d["label"]] or [])
        bad = (free_set | set(fixed_m[d["label"]] or [])) - set(names)
        if bad:
            raise ValueError(f"{d['label']}: no parameter(s) {sorted(bad)}; it has {names}.")
        if "C_Q" in names:
            _set(d, "C_Q", abs(_get(d, "C_Q")))
        for name in names:
            lo, hi = _default_bounds(d, name)
            if name in gbounds:
                lo, hi = gbounds[name]
            if name in (sbounds.get(d["label"]) or {}):
                lo, hi = sbounds[d["label"]][name]
            lo, hi = min(lo, hi), max(lo, hi)
            value = _get(d, name)
            if name in free_set and not lo <= value <= hi:
                warnings.append(f"{d['label']}: {name} start {value:g} outside ({lo:g}, {hi:g}); "
                                "moved inside")
                value = float(np.clip(value, lo, hi))
                _set(d, name, value)
            table.append({"site": i, "label": d["label"], "name": name, "value": value,
                          "free": name in free_set, "lower": lo, "upper": hi})
    if fit_baseline:
        table += [{"site": None, "label": "", "name": "b0", "value": b0, "free": True,
                   "lower": -np.inf, "upper": np.inf},
                  {"site": None, "label": "", "name": "b1", "value": b1, "free": True,
                   "lower": -np.inf, "upper": np.inf}]

    sim = dict(larmor_Hz=larmor_Hz, mas_Hz=mas_Hz, order=order, transitions=transitions,
               beta_r=beta_r, n_zcw=int(n_zcw))
    cache = {}

    def sticks(i, d):
        key = (i, repr({k: v for k, v in d.items() if k not in _BROADENING}))
        if key not in cache:
            if len(cache) > 256:
                cache.clear()
            comp, _p = _site_sticks(Site.from_dict(d), x, sim["larmor_Hz"], sim["mas_Hz"],
                                    sim["order"], sim["transitions"], sim["beta_r"], sim["n_zcw"])
            cache[key] = comp
        return cache[key]

    def site_curves(ds):
        return [np.sum(list(_broaden_site(Site.from_dict(d), sticks(i, d), x).values()), axis=0)
                for i, d in enumerate(ds)]

    def apply(values):
        ds = copy.deepcopy(dicts)
        base = (b0, b1)
        for row, v in zip(table, values):
            if row["site"] is None:
                base = (v, base[1]) if row["name"] == "b0" else (base[0], v)
            else:
                _set(ds[row["site"]], row["name"], v)
        return ds, base

    # ---- starting amounts ----------------------------------------------------------------
    if estimate_amounts:
        for d in dicts:
            d["amount"] = 1.0
        shapes = site_curves(dicts)
        A = np.column_stack([s[mask] for s in shapes])
        target = y[mask] - (b0 + b1 * (x[mask] - x_ref))
        amounts, _res = nnls(A, target)
        for d, a, row in zip(dicts, amounts, [r for r in table if r["name"] == "amount"]):
            a = max(a, 1e-6)
            d["amount"] = a
            row["value"] = a

    free = np.array([r["free"] for r in table])
    p_all = np.array([r["value"] for r in table], dtype=float)
    start_curves = site_curves(apply(p_all)[0])

    def model(_x, *free_values):
        values = p_all.copy()
        values[free] = free_values
        ds, (c0, c1) = apply(values)
        return (np.sum(site_curves(ds), axis=0) + c0 + c1 * (x - x_ref))[mask]

    if free.any():
        lo = np.array([r["lower"] for r in table])[free]
        hi = np.array([r["upper"] for r in table])[free]
        popt, pcov = curve_fit(_fit_progress.tracked(model, y[mask], n_free=int(free.sum())),
                               x[mask], y[mask], p0=p_all[free], bounds=(lo, hi),
                               diff_step=1e-4, max_nfev=max_nfev)
        p_all[free] = popt
    else:
        pcov = np.zeros((0, 0))

    # ---- back to raw units, outputs ------------------------------------------------------------
    unit = np.array([scale if r["name"] in ("amount", "b0", "b1") else 1.0 for r in table])
    pcov = np.asarray(pcov) * np.outer(unit[free], unit[free])
    errors = np.zeros(len(table))
    errors[free] = np.sqrt(np.clip(np.diag(pcov), 0, None))
    ds, (c0, c1) = apply(p_all)
    curves = site_curves(ds)
    baseline = (c0 + c1 * (x - x_ref)) * scale
    y_fit = np.sum(curves, axis=0) * scale + baseline
    fitted_sites = []
    for d in ds:
        d = copy.deepcopy(d)
        d["amount"] *= scale
        fitted_sites.append(Site.from_dict(d))
    parameters = []
    for r, v, e, u in zip(table, p_all, errors, unit):
        parameters.append({**r, "value": float(v * u), "error": float(e) if r["free"] else None,
                           "lower": r["lower"] * u, "upper": r["upper"] * u})
    free_names = [f"{r['label']}: {r['name']}" if r["label"] else r["name"]
                  for r in table if r["free"]]

    ym, fm = y_raw[mask], y_fit[mask]
    q = fit_quality(ym, fm, int(free.sum()))
    tot = float(np.sum((ym - ym.mean()) ** 2))
    q["r2"] = 1 - float(np.sum((ym - fm) ** 2)) / tot if tot else float("nan")

    derived = _derived(parameters, pcov, free_names, fitted_sites, larmor_Hz)
    if q["r2"] < 0.9:
        warnings.append(f"Poor fit (R^2 {q['r2']:.3f}): most likely a local minimum -- try "
                        "other starting values (fit_combined_multistart), or check the model.")
    warnings += _warnings(parameters, pcov, free_names, curves, x, mask, baseline_regions,
                          fit_baseline, start_curves)
    regime = "static" if not mas_Hz else f"MAS {mas_Hz / 1e3:g} kHz"
    log = ("Combined-Hamiltonian fit: " + f"{regime}, nu0 = {larmor_Hz / 1e6:.7g} MHz, "
           f"quadrupolar {order}" + ("" if order == "exact" else "-order")
           + f", transitions {transitions}, {n_zcw} orientations, window "
           + ("whole spectrum" if fit_regions is None else
              ", ".join(f"{max(a, b):g} to {min(a, b):g} ppm" for a, b in fit_regions))
           + (", baseline fitted" if fit_baseline else ", baseline estimated and held") + ".\n"
           + "\n".join(s.describe() for s in fitted_sites))
    return {"sites": fitted_sites, "parameters": parameters, "pcov": pcov,
            "free_names": free_names, "derived": derived, "x": x, "y": y_raw, "y_fit": y_fit,
            "site_spectra": {s.label: c * scale for s, c in zip(fitted_sites, curves)},
            "baseline": baseline, "mask": mask, "quality": q, "warnings": warnings, "log": log,
            "settings": {"larmor_Hz": larmor_Hz, "mas_Hz": mas_Hz, "order": order,
                         "transitions": transitions, "rotor_angle": rotor_angle,
                         "n_zcw": int(n_zcw), "fit_regions": fit_regions,
                         "fit_baseline": fit_baseline},
            "nuclide": results.get("nuclide") or sites[0].nucleus}


def fit_combined_multistart(results, sites, starts, screen_n_zcw=None, max_starts=64,
                            verbose=True, **fit_kwargs):
    """
    :func:`fit_combined` from every combination of starting values in
    ``starts`` -- ``{site label: {parameter: [values, ...]}}`` -- keeping
    the fit with the lowest chi^2.

    Powder lineshapes are multimodal: a static CSA fitted from the wrong
    side (e.g. Delta started at -40 when it is +60) settles in a
    different, wrong minimum with a poor R^2. Scanning a handful of
    starts is the reliable remedy. ``screen_n_zcw``: fit every start
    with this (coarser, faster) powder set, then refit only the best at
    the full ``n_zcw``.

    Returns the best fit (as fit_combined), with ``fit["starts"]``: one
    row per start -- the start values, R^2, chi^2 -- best first.
    """
    import itertools
    sites = as_sites(sites)
    labels = [s.label for s in sites]
    axes = [(lab, name, list(values)) for lab, per in starts.items() for name, values in per.items()]
    for lab, name, _v in axes:
        if lab not in labels:
            raise KeyError(f"{lab!r} is not a site label ({labels}).")
    combos = list(itertools.product(*[v for _l, _n, v in axes])) or [()]
    if len(combos) > max_starts:
        raise ValueError(f"{len(combos)} starting points (max_starts={max_starts}); "
                         "use fewer values.")
    rows, best, best_start = [], None, None
    screen = dict(fit_kwargs)
    if screen_n_zcw:
        screen["n_zcw"] = screen_n_zcw
    for k, combo in enumerate(combos):
        _fit_progress.set_stage(f"start {k + 1} of {len(combos)}")
        dicts = [s.to_dict() for s in sites]
        for (lab, name, _v), value in zip(axes, combo):
            _set(dicts[labels.index(lab)], name, value)
        start_label = ", ".join(f"{lab}: {name} = {v:g}" for (lab, name, _), v in zip(axes, combo))
        try:
            fit = fit_combined(results, dicts, **screen)
        except (RuntimeError, ValueError) as exc:          # a start that does not converge
            rows.append({"start": start_label, "r2": float("nan"), "chi2": float("inf"),
                         "error": str(exc)})
            continue
        rows.append({"start": start_label, "r2": fit["quality"]["r2"],
                     "chi2": fit["quality"]["chi2"]})
        if verbose:
            print(f"start {start_label}: R^2 {fit['quality']['r2']:.5f}")
        if best is None or fit["quality"]["chi2"] < best["quality"]["chi2"]:
            best, best_start = fit, dicts
    if best is None:
        raise RuntimeError("No starting point converged.")
    if screen_n_zcw:
        _fit_progress.set_stage("refining the best start")
        best = fit_combined(results, best["sites"], **fit_kwargs)
    best["starts"] = sorted(rows, key=lambda r: r["chi2"])
    n_good = sum(1 for r in rows if np.isfinite(r["chi2"])
                 and r["chi2"] <= 1.01 * best["starts"][0]["chi2"])
    best["log"] += (f"\nMulti-start: {len(combos)} starts; {n_good} reached the best chi^2 "
                    f"(within 1%).")
    if n_good == 1 and len(combos) > 1:
        best["warnings"].append("Only one start reached the best minimum -- scan more starting "
                                "values before trusting it.")
    return best


def _cov_of(names, pcov, free_names):
    """Covariance sub-matrix for ``names`` (zeros for fixed ones)."""
    n = len(names)
    out = np.zeros((n, n))
    pos = [free_names.index(k) if k in free_names else None for k in names]
    for a, pa in enumerate(pos):
        for b, pb in enumerate(pos):
            if pa is not None and pb is not None:
                out[a, b] = pcov[pa, pb]
    return out


def _derived(parameters, pcov, free_names, sites, larmor_Hz):
    amounts = {p["label"]: p for p in parameters if p["name"] == "amount"}
    total = sum(p["value"] for p in amounts.values()) or 1.0
    out = {}
    for s in sites:
        a = amounts[s.label]
        row = {"share_%": 100 * a["value"] / total}
        # share error from the amounts' covariance
        names = [f"{lab}: amount" for lab in amounts]
        cov = _cov_of(names, pcov, free_names)
        g = np.array([(-a["value"] / total ** 2 + (1 / total if lab == s.label else 0.0))
                      for lab in amounts]) * 100
        row["share_%_err"] = float(np.sqrt(max(g @ cov @ g, 0.0)))
        if s.get("quadrupolar"):
            vals = {p["name"]: p["value"] for p in parameters if p["label"] == s.label}
            cq, eq, d = abs(vals["C_Q"]), vals["eta_Q"], vals["delta_iso"]
            cov3 = _cov_of([f"{s.label}: {k}" for k in ("delta_iso", "C_Q", "eta_Q")], pcov,
                           free_names)
            fac = np.sqrt(1 + eq ** 2 / 3)
            gp = np.array([0.0, fac, cq * eq / (3 * fac)])
            row.update(P_Q=cq * fac, P_Q_err=float(np.sqrt(max(gp @ cov3 @ gp, 0.0))))
            if not float(s.spin).is_integer():
                k = quadrupolar_ct_isotropic_shift_Hz(s.spin, 1e6, 0.0, larmor_Hz) / larmor_Hz * 1e6
                gc = np.array([1.0, 2 * k * cq * fac ** 2, k * cq ** 2 * 2 * eq / 3])
                row.update(delta_cg=d + k * cq ** 2 * fac ** 2,
                           delta_cg_err=float(np.sqrt(max(gc @ cov3 @ gc, 0.0))))
        out[s.label] = row
    return out


def _warnings(parameters, pcov, free_names, curves, x, mask, baseline_regions, fit_baseline,
              start_curves=None):
    out = []
    err = {f"{p['label']}: {p['name']}": p for p in parameters}
    for p in parameters:
        if not p["free"]:
            continue
        e = (p["error"] or 0.0) + 1e-9 * max(1.0, abs(p["value"]))
        at_upper = (np.isfinite(p["upper"]) and p["value"] + e >= p["upper"]
                    and p["name"] != "amount")
        at_lower = (np.isfinite(p["lower"]) and p["value"] - e <= p["lower"]
                    and p["name"] not in ("C_Q",))
        if at_upper and at_lower:
            # +/- 1 sigma covers the whole allowed range: the data do not
            # determine it at all, which is not the same as "on a bound".
            out.append(f"{p['label']}: {p['name']} is not determined by the data -- its 1σ "
                       f"uncertainty spans its whole range ({p['lower']:g} to {p['upper']:g}); "
                       "fix it, or add data that constrain it")
        elif at_upper:
            out.append(f"{p['label']}: {p['name']} at its upper bound {p['upper']:g}")
        elif at_lower:
            out.append(f"{p['label']}: {p['name']} at its lower bound {p['lower']:g}")
        if p["name"] == "C_Q" and p["value"] < 0.02:
            out.append(f"{p['label']}: C_Q fitted to ~0 -- no quadrupolar splitting is resolved")
    d = np.sqrt(np.clip(np.diag(pcov), 1e-300, None))
    corr = pcov / np.outer(d, d) if pcov.size else pcov
    for a in range(len(free_names)):
        for b in range(a + 1, len(free_names)):
            if abs(corr[a, b]) > 0.95 and "amount" not in free_names[a] + free_names[b]:
                out.append(f"{free_names[a]} and {free_names[b]} are {abs(corr[a, b]):.0%} "
                           "correlated -- the data do not separate them; fix one, or add data "
                           "(another field or spinning rate)")
    total = np.sum(curves, axis=0)
    peak = float(np.max(np.abs(total[mask]))) if mask.any() else 0.0
    if peak > 0:
        edges = np.flatnonzero(np.diff(np.concatenate([[False], mask, [False]]).astype(int)))
        ends = np.clip(np.concatenate([edges[0::2], edges[1::2] - 1]), 0, x.size - 1)
        if np.max(np.abs(total[ends])) > 0.05 * peak:
            out.append("The fitted pattern still has intensity at an edge of the fit window: it "
                       "is clipped -- widen the window")
        if not fit_baseline:
            if baseline_regions is not None:
                zone = build_region_mask(x, [baseline_regions] if np.isscalar(baseline_regions[0])
                                         else baseline_regions)
            else:
                k = max(1, int(x.size * 0.05))
                zone = np.zeros(x.size, dtype=bool)
                zone[:k] = zone[-k:] = True
            # The starting model too: a fit dragged off by a baseline sitting
            # on the pattern can move its own pattern out of the zone.
            inside = float(np.max(np.abs(total[zone]))) / peak if zone.any() else 0.0
            if start_curves is not None and zone.any():
                start = np.sum(start_curves, axis=0)
                if np.max(np.abs(start[mask])) > 0:
                    inside = max(inside, float(np.max(np.abs(start[zone])))
                                 / float(np.max(np.abs(start[mask]))))
            if inside > 0.01:
                out.append(f"The baseline was estimated where the pattern (as started or as "
                           f"fitted) still has {inside:.0%} of its height: give baseline_regions "
                           "clear of the pattern, or fit_baseline=True")
    return out


# ============================================================
# Report, plot, workflow
# ============================================================

def _fmt(value, error):
    if error is None:
        return f"{value:.6g} (fixed)"
    # An error at round-off level (a single site's share: 100 % with an
    # error of ~1e-16 from the covariance) is no error, not 17 decimals.
    if not np.isfinite(error) or error <= 1e-12 * max(abs(value), 1.0):
        return f"{value:.6g}"
    decimals = max(0, 1 - int(np.floor(np.log10(error))))
    return f"{value:.{decimals}f} ± {error:.{decimals}f}"


_UNITS = {"delta_iso": "ppm", "fwhm": "ppm", "aniso": "ppm", "C_Q": "MHz", "alpha": "deg",
          "beta": "deg", "gamma": "deg"}


def print_combined_fit_report(fit, save_txt=None):
    """Print (and optionally save to ``save_txt``) the fitted parameters
    with 1-sigma errors (rounded to the digits the error supports), the
    derived quantities, fit quality, the setup and every warning.
    Returns the text."""
    lines = ["=" * 68, "COMBINED-HAMILTONIAN FIT REPORT", "=" * 68, fit["log"], ""]
    for label in dict.fromkeys(p["label"] for p in fit["parameters"] if p["label"]):
        lines.append(label)
        for p in fit["parameters"]:
            if p["label"] != label:
                continue
            unit = _UNITS.get(p["name"], "Hz" if p["name"].startswith("J") else "")
            name = f"{p['name']} ({unit})" if unit else p["name"]
            lines.append(f"  {name:<18}= {_fmt(p['value'], p['error'])}")
        d = fit["derived"][label]
        lines.append(f"  {'share (%)':<18}= {_fmt(d['share_%'], d['share_%_err'] or None)}")
        if "P_Q" in d:
            lines.append(f"  {'P_Q (MHz)':<18}= {_fmt(d['P_Q'], d['P_Q_err'] or None)}")
        if "delta_cg" in d:
            lines.append(f"  {'CT c.o.g. (ppm)':<18}= {_fmt(d['delta_cg'], d['delta_cg_err'] or None)}")
        lines.append("")
    q = fit["quality"]
    lines.append(f"R^2 = {q['r2']:.5f}   RMSE = {q['rmse']:.4g}   reduced chi^2 = "
                 f"{q['reduced_chi2']:.4g}")
    if fit["warnings"]:
        lines += ["", "Warnings:"] + [f"  - {w}" for w in fit["warnings"]]
    lines.append("=" * 68)
    text = "\n".join(lines)
    print(text)
    if save_txt:
        with open(save_txt, "w", encoding="utf-8") as f:
            f.write(text)
    return text


def plot_combined_fit(fit, nuclide=None, show_sites=True, show_residual=True, xlims=None,
                      pl_width=3.375, pl_height=3.375, fontsize=12, font=None, fname=None):
    """Data, fit, each site (dashed, on the baseline) and the residual,
    on a descending ppm axis over the fit window; Okabe-Ito colours."""
    import matplotlib.pyplot as plt
    from .plotting import _apply_axis_style
    from .utils import NUC_label
    from ..plotting.accessibility import ordered_qualitative_colours
    if font:
        plt.rcParams["font.family"] = font
    x, m = fit["x"], fit["mask"]
    scale = float(np.max(np.abs(fit["y"][m]))) or 1.0
    colours = ordered_qualitative_colours("okabe_ito")
    if show_residual:
        fig, (ax, rax) = plt.subplots(2, 1, sharex=True, figsize=(pl_width, pl_height * 1.25),
                                      gridspec_kw={"height_ratios": [4, 1], "hspace": 0.05})
    else:
        fig, ax = plt.subplots(figsize=(pl_width, pl_height))
        rax = None
    ax.plot(x[m], fit["y"][m] / scale, color=colours[0], lw=1.0, label="Data")
    ax.plot(x[m], fit["y_fit"][m] / scale, color=colours[7], lw=1.3, label="Fit")
    if show_sites and len(fit["site_spectra"]) > 1:
        site_colours = [colours[k] for k in (3, 5, 4, 6, 2, 1)]   # clear of the fit's vermilion
        for i, (label, c) in enumerate(fit["site_spectra"].items()):
            ax.plot(x[m], (c + fit["baseline"])[m] / scale, ls=(0, (4, 2)), lw=1.0,
                    color=site_colours[i % len(site_colours)], label=label)
    ax.legend(frameon=False, fontsize=fontsize * 0.8)
    _apply_axis_style(ax, "in", 4.0, 1.0, 4.0, None, fontsize, False)
    last = ax
    if rax is not None:
        rax.plot(x[m], (fit["y"] - fit["y_fit"])[m] / scale, color="0.4", lw=0.8)
        rax.axhline(0, color="0.7", lw=0.6)
        _apply_axis_style(rax, "in", 4.0, 1.0, 4.0, None, fontsize, False)
        last = rax
    lo, hi = (min(xlims), max(xlims)) if xlims else (x[m].min(), x[m].max())
    ax.set_xlim(hi, lo)
    nuclide = nuclide or fit.get("nuclide")
    last.set_xlabel(NUC_label(nuclide, x_unit="ppm") if nuclide else "Shift (ppm)",
                    fontsize=fontsize)
    if fname:
        fig.savefig(fname, dpi=300, bbox_inches="tight")
    return fig, ax


def NMRCombinedFitWorkflow(datapath, sites, procno=1, show_plot=True, show_report=True,
                           save_report=None, save_csv=None, fname=None, **fit_kwargs):
    """Import a processed 1D Bruker spectrum, fit ``sites`` (see
    :func:`fit_combined` for ``fit_kwargs``), report, plot and optionally
    save the report / a CSV of data, fit, sites and baseline."""
    from .one_d import NMR1dimport
    results = NMR1dimport(datapath, procno=procno)
    fit = fit_combined(results, sites, **fit_kwargs)
    if show_report or save_report:
        fit["report"] = print_combined_fit_report(fit, save_txt=save_report)
    if show_plot:
        fit["figure"], fit["axes"] = plot_combined_fit(fit, fname=fname)
    if save_csv:
        m = fit["mask"]
        with open(save_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["x_ppm", "y", "y_fit"] + list(fit["site_spectra"]) + ["baseline"])
            for i in np.flatnonzero(m):
                w.writerow([fit["x"][i], fit["y"][i], fit["y_fit"][i]]
                           + [c[i] for c in fit["site_spectra"].values()] + [fit["baseline"][i]])
    return fit
