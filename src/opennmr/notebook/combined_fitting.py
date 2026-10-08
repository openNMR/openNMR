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

Several nuclei on one spectrum
------------------------------
A site may be of another nucleus than the spectrum's -- it is placed by
the IUPAC frequency ratios (:mod:`opennmr.notebook.isotopes`). A site
*linked* to another isotope's site (``Site.isotope_partner`` /
``Site.link_to``) has no parameters of its own except those named in its
link's ``free``: everything else follows its parent, so 47Ti and 49Ti
are fitted as one chemistry::

    ti = Site("47Ti", -800, label="Ti1").add_quadrupolar(4.0, 0.2)
    fit = fit_combined(results, [ti, ti.isotope_partner()], transitions="ct")

A site of another element with no link (a 63Cu coil signal on a 23Na
spectrum) is fitted with its own parameters. Shares (%) are worked out
per nucleus; a linked partner reports its parent's.

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
                                   simulate_combined, resolve_link_dicts, observed_nucleus,
                                   site_axes, placement_notes)
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


def _amount_owners(dicts):
    """Per site, the index of the site whose amount sets its amount:
    itself, or the parent of a partner whose amount is linked."""
    labels = [d["label"] for d in dicts]
    out = []
    for i, d in enumerate(dicts):
        link = d.get("link")
        if link and "amount" not in (link.get("free") or []) and link.get("to") in labels:
            out.append(labels.index(link["to"]))
        else:
            out.append(i)
    return out


def _owner_columns(curves, owners):
    """(owner indices, one summed curve per owner) -- each column is what
    one independent amount multiplies."""
    keys = list(dict.fromkeys(owners))
    return keys, [np.sum([c for c, o in zip(curves, owners) if o == k], axis=0) for k in keys]


def _spectrum_nucleus(results, sites, observe, warnings):
    """The observed nucleus for a fit: ``observe``, else the spectrum's
    own when a site is of it, else the first unlinked site's."""
    if observe:
        return observed_nucleus(sites, observe)
    nuc = results.get("nuclide")
    if nuc and any(s.nucleus == nuc for s in sites):
        return str(nuc)
    first = observed_nucleus(sites)
    if nuc and str(nuc).lstrip("0123456789") != "":
        warnings.append(f"The spectrum's nucleus ({nuc}) is not the sites' ({first}).")
    return first


def _linked_sources(dicts):
    """{(label, name): (parent label, factor)} for every parameter of a
    linked site that follows its parent -- for errors and shares."""
    from .isotopes import partner_factors
    out = {}
    by_label = {d["label"]: d for d in dicts}
    for d in dicts:
        link = d.get("link")
        if not link or link.get("to") not in by_label:
            continue
        parent = by_label[link["to"]]
        f = partner_factors(parent["nucleus"], d["nucleus"])
        own = set(link.get("free") or [])
        for name in site_parameter_names(d):
            if name in own:
                continue
            factor = (f["C_Q"] if name == "C_Q" else f["amount"] if name == "amount"
                      else f["J"] if name.startswith("J") else 1.0)
            out[(d["label"], name)] = (parent["label"], factor or 1.0)
    return out


# ============================================================
# Fit
# ============================================================

def fit_combined(results, sites, vary=None, fixed=None, bounds=None, larmor_Hz=None,
                 mas_Hz=0.0, order="second", transitions="all", rotor_angle="magic",
                 n_zcw=None, fit_regions=None, baseline_regions=None, fit_baseline=False,
                 estimate_amounts=True, max_nfev=None, fit_mas=False, mas_window_Hz=500.0,
                 mas_scan_points=41, mas_bounds_Hz=None, observe=None):
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
    fit_mas : let the MAS rate vary, within ``mas_Hz`` ± ``mas_window_Hz``
        (default False: held at ``mas_Hz``). The recorded spinning rate
        is often a few Hz to a few hundred Hz off the true one, and a
        sideband pattern is very sensitive to it. Freed, the rate is
        first scanned over the window at ``mas_scan_points`` points with
        a coarse powder average (each point with its own least-squares
        amounts) -- a start more than a linewidth off puts every sideband
        beside its data, a minimum a local fit cannot climb out of --
        and then refined with everything else. Reported as the
        parameter ``mas_Hz`` (no site label) with its 1σ.
        ``mas_bounds_Hz=(lo, hi)`` replaces the symmetric window.
    observe : the spectrum's nucleus (default: ``results['nuclide']``
        when a site is of it, else the first unlinked site's). Sites of
        other nuclei are placed on its scale (module docstring).

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
    mas_start = mas_Hz
    if fit_mas:
        if mas_Hz <= 0:
            raise ValueError("fit_mas needs a spinning rate to start from: a static fit has no "
                             "MAS rate to vary.")
        mas_window_Hz = abs(float(mas_window_Hz))
        if mas_bounds_Hz is not None:
            mas_lo, mas_hi = sorted(float(v) for v in mas_bounds_Hz)
            if not mas_lo <= mas_Hz <= mas_hi or mas_hi <= mas_lo:
                raise ValueError(f"mas_bounds_Hz {mas_bounds_Hz} must contain the starting "
                                 f"rate {mas_Hz:g} Hz.")
        elif not mas_window_Hz:
            raise ValueError("mas_window_Hz must be > 0 to let the MAS rate vary.")
        else:
            mas_lo, mas_hi = mas_Hz - mas_window_Hz, mas_Hz + mas_window_Hz
    if n_zcw is None:
        n_zcw = 2584 if mas_Hz else 28657
    beta_r = (np.arccos(1 / np.sqrt(3)) if rotor_angle == "magic"
              else np.deg2rad(float(rotor_angle)))
    warnings = []
    observe = _spectrum_nucleus(results, sites, observe, warnings)

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
    resolve_link_dicts(dicts)            # a link that cannot hold fails here, by name
    axes = site_axes(sites, x, larmor_Hz, observe)
    owners = _amount_owners(dicts)
    vary_m, fixed_m = _per_site(vary, labels, "vary"), _per_site(fixed, labels, "fixed")
    gbounds = {k: v for k, v in (bounds or {}).items() if k not in labels}
    sbounds = {k: v for k, v in (bounds or {}).items() if k in labels}
    table = []
    for i, d in enumerate(dicts):
        names = site_parameter_names(d)
        linked = d.get("link")
        if linked:
            own = list(linked.get("free") or [])
            linked_names = [n for n in names if n not in own]
            names = [n for n in names if n in own]
        if vary_m[d["label"]] is not None:
            free_set = set(vary_m[d["label"]])
        else:
            free_set = set(names) - {"alpha", "beta", "gamma"}
        free_set -= set(fixed_m[d["label"]] or [])
        bad = (free_set | set(fixed_m[d["label"]] or [])) - set(names)
        if bad and linked:
            raise ValueError(f"{d['label']}: {sorted(bad)} follow {linked['to']} (linked); "
                             f"to fit its own, name them in the link's free list "
                             f"(it keeps {names or 'none'} of its own; follows {linked_names}).")
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
    if fit_mas:
        table.append({"site": None, "label": "", "name": "mas_Hz", "value": mas_Hz, "free": True,
                      "lower": max(mas_lo, 1.0), "upper": mas_hi})

    sim = dict(larmor_Hz=larmor_Hz, mas_Hz=mas_Hz, order=order, transitions=transitions,
               beta_r=beta_r, n_zcw=int(n_zcw))
    cache = {}

    def sticks(i, d, mas=None, n=None):
        mas = sim["mas_Hz"] if mas is None else float(mas)
        n = sim["n_zcw"] if n is None else int(n)
        key = (i, repr({k: v for k, v in d.items() if k not in _BROADENING}), mas, n)
        if key not in cache:
            if len(cache) > 256:
                cache.clear()
            comp, _p = _site_sticks(Site.from_dict(d), axes[i][0], axes[i][1], mas,
                                    sim["order"], sim["transitions"], sim["beta_r"], n)
            cache[key] = comp
        return cache[key]

    def site_curves(ds, mas=None, n=None):
        ds = resolve_link_dicts(ds)
        return [np.sum(list(_broaden_site(Site.from_dict(d), sticks(i, d, mas, n),
                                          axes[i][0]).values()), axis=0)
                for i, d in enumerate(ds)]

    def apply(values):
        ds = copy.deepcopy(dicts)
        base = (b0, b1)
        mas = mas_Hz
        for row, v in zip(table, values):
            if row["site"] is None:
                if row["name"] == "mas_Hz":
                    mas = v
                else:
                    base = (v, base[1]) if row["name"] == "b0" else (base[0], v)
            else:
                _set(ds[row["site"]], row["name"], v)
        return ds, base, mas

    # ---- the MAS rate: scanned over its window before the local fit ------------------------
    mas_scan = None
    if fit_mas and int(mas_scan_points) > 1:
        row = next(r for r in table if r["name"] == "mas_Hz")
        grid = np.linspace(row["lower"], row["upper"], int(mas_scan_points))
        coarse = min(int(n_zcw), 233)
        target = y[mask] - (b0 + b1 * (x[mask] - x_ref))
        trial = [dict(d, amount=1.0) for d in dicts]
        chi2 = []
        for m in grid:
            _k, cols = _owner_columns(site_curves(trial, mas=m, n=coarse), owners)
            A = np.column_stack([c[mask] for c in cols])
            a, _r = nnls(A, target)
            chi2.append(float(np.sum((A @ a - target) ** 2)))
        best = float(grid[int(np.argmin(chi2))])
        row["value"] = best
        mas_Hz = best
        sim["mas_Hz"] = best
        mas_scan = {"grid_Hz": grid, "chi2": np.asarray(chi2), "best_Hz": best,
                    "n_zcw": coarse}

    # ---- starting amounts ----------------------------------------------------------------
    if estimate_amounts:
        for d in dicts:
            d["amount"] = 1.0
        keys, shapes = _owner_columns(site_curves(dicts), owners)
        A = np.column_stack([s[mask] for s in shapes])
        target = y[mask] - (b0 + b1 * (x[mask] - x_ref))
        amounts, _res = nnls(A, target)
        rows = {r["site"]: r for r in table if r["name"] == "amount"}
        for k, a in zip(keys, amounts):
            a = max(a, 1e-6)
            dicts[k]["amount"] = a
            rows[k]["value"] = a

    free = np.array([r["free"] for r in table])
    p_all = np.array([r["value"] for r in table], dtype=float)
    start_curves = site_curves(apply(p_all)[0])

    def local_fit(which, **kw):
        def model(_x, *free_values):
            values = p_all.copy()
            values[which] = free_values
            ds, (c0, c1), mas = apply(values)
            return (np.sum(site_curves(ds, mas=mas), axis=0) + c0 + c1 * (x - x_ref))[mask]
        lo = np.array([r["lower"] for r in table])[which]
        hi = np.array([r["upper"] for r in table])[which]
        popt, pcov = _fit_progress.limited_curve_fit(
            _fit_progress.tracked(model, y[mask], n_free=int(which.sum())),
            x[mask], y[mask], p0=p_all[which], bounds=(lo, hi),
            diff_step=1e-4, max_iterations=max_nfev, label="The Combined Hamiltonian fit", **kw)
        p_all[which] = popt
        return pcov

    if free.any():
        if fit_mas:
            # Two passes. Refining the rate together with everything from
            # the start lets a shape parameter run off while the sidebands
            # are still slightly misplaced (measured: a 13C CSA started
            # 150 Hz off went to Δ = -690 ppm, R² 0.84, though the rate
            # itself came out right). Hold the scanned rate first, then
            # free it from there, steps scaled by the Jacobian.
            held = free & np.array([r["name"] != "mas_Hz" for r in table])
            if held.any():
                local_fit(held)
            pcov = local_fit(free, x_scale="jac")
        else:
            pcov = local_fit(free)
    else:
        pcov = np.zeros((0, 0))

    # ---- back to raw units, outputs ------------------------------------------------------------
    unit = np.array([scale if r["name"] in ("amount", "b0", "b1") else 1.0 for r in table])
    pcov = np.asarray(pcov) * np.outer(unit[free], unit[free])
    errors = np.zeros(len(table))
    errors[free] = np.sqrt(np.clip(np.diag(pcov), 0, None))
    ds, (c0, c1), mas_Hz = apply(p_all)
    curves = site_curves(ds, mas=mas_Hz)
    baseline = (c0 + c1 * (x - x_ref)) * scale
    y_fit = np.sum(curves, axis=0) * scale + baseline
    fitted_sites = []
    for d in resolve_link_dicts(ds):
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

    derived = _derived(parameters, pcov, free_names, fitted_sites,
                       {s.label: nu0 for s, (_a, nu0) in zip(fitted_sites, axes)},
                       sources=_linked_sources(dicts))
    if q["r2"] < 0.9:
        warnings.append(f"Poor fit (R^2 {q['r2']:.3f}): most likely a local minimum -- try "
                        "other starting values (fit_combined_multistart), or check the model.")
    warnings += _warnings(parameters, pcov, free_names, curves, x, mask, baseline_regions,
                          fit_baseline, start_curves)
    regime = "static" if not mas_Hz else f"MAS {mas_Hz / 1e3:g} kHz"
    if fit_mas:
        mas_row = next(p for p in parameters if p["name"] == "mas_Hz")
        regime = (f"MAS {mas_Hz:.1f} ± {mas_row['error'] or 0.0:.1f} Hz (fitted; started at "
                  f"{mas_start:g} Hz, allowed {mas_row['lower']:g} to {mas_row['upper']:g} Hz)")
    log = ("Combined-Hamiltonian fit: " + f"{regime}, nu0 = {larmor_Hz / 1e6:.7g} MHz, "
           f"quadrupolar {order}" + ("" if order == "exact" else "-order")
           + f", transitions {transitions}, {n_zcw} orientations, window "
           + ("whole spectrum" if fit_regions is None else
              ", ".join(f"{max(a, b):g} to {min(a, b):g} ppm" for a, b in fit_regions))
           + (", baseline fitted" if fit_baseline else ", baseline estimated and held") + ".\n"
           + "\n".join(s.describe() for s in fitted_sites)
           + "".join("\n" + n for n in placement_notes(fitted_sites, observe, x)))
    if any("gyromagnetic ratios" in n for n in placement_notes(fitted_sites, observe)):
        warnings.append("A nucleus without a tabulated IUPAC frequency ratio was placed by its "
                        "gyromagnetic ratio, which ignores the reference compounds' shielding: "
                        "its position may be off by up to thousands of ppm -- free its δiso.")
    return {"sites": fitted_sites, "parameters": parameters, "pcov": pcov,
            "free_names": free_names, "derived": derived, "x": x, "y": y_raw, "y_fit": y_fit,
            "site_spectra": {s.label: c * scale for s, c in zip(fitted_sites, curves)},
            "baseline": baseline, "mask": mask, "quality": q, "warnings": warnings, "log": log,
            "settings": {"larmor_Hz": larmor_Hz, "mas_Hz": mas_Hz, "order": order,
                         "transitions": transitions, "rotor_angle": rotor_angle,
                         "n_zcw": int(n_zcw), "fit_regions": fit_regions,
                         "fit_baseline": fit_baseline, "fit_mas": bool(fit_mas),
                         "mas_Hz_start": mas_start,
                         "mas_window_Hz": float(mas_window_Hz) if fit_mas else None,
                         "observe": observe},
            "mas_scan": mas_scan,
            "nuclide": results.get("nuclide") or observe}


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
    last_limit = None
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
            if isinstance(exc, _fit_progress.FitNotConverged):
                last_limit = exc
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
        if last_limit is not None:
            # Every start ran out of iterations: say so, with what to try.
            raise last_limit
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


def starting_model(results, sites, larmor_Hz=None, mas_Hz=0.0, order="second",
                   transitions="all", rotor_angle="magic", n_zcw=None, fit_regions=None,
                   baseline_regions=None, estimate_amounts=True, observe=None):
    """
    The model :func:`fit_combined` starts from, on the data's own axis --
    to see a starting guess against the spectrum before fitting (the
    GUI's live preview), or to check one in a notebook.

    It is the fit's own first step, not an approximation of it: the same
    window, the same held linear baseline, and (with
    ``estimate_amounts``, the fit's default) the same non-negative least
    squares amounts. With every parameter fixed, ``fit_combined``
    returns exactly this curve as its ``y_fit``.

    ``n_zcw`` : powder orientations (default as fit_combined: 28657
    static, 2584 MAS). A preview can use fewer -- the shape converges
    long before the fit's default.

    Returns a dict: ``x`` (ascending ppm), ``y`` (data), ``mask`` (the
    fit window), ``start`` (total, with baseline), ``site_spectra``
    ({label: curve}), ``baseline``, ``amounts`` ({label: amount}, raw
    units as fit_combined reports them), ``n_zcw``.

        guess = starting_model(results, sites, mas_Hz=15e3, transitions="ct")
        plt.plot(guess["x"], guess["y"], guess["x"], guess["start"])
    """
    sites = as_sites(sites)
    if order not in QUADRUPOLAR_ORDERS:
        raise ValueError(f"order must be one of {QUADRUPOLAR_ORDERS}.")
    if larmor_Hz is None:
        larmor_Hz = float(results["SFO1"]) * 1e6
    mas_Hz = float(mas_Hz or 0.0)
    if n_zcw is None:
        n_zcw = 2584 if mas_Hz else 28657
    beta_r = (np.arccos(1 / np.sqrt(3)) if rotor_angle == "magic"
              else np.deg2rad(float(rotor_angle)))
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
    observe = _spectrum_nucleus(results, sites, observe, [])
    axes = site_axes(sites, x, float(larmor_Hz), observe)
    dicts = [s.to_dict() for s in sites]
    owners = _amount_owners(dicts)
    for d in dicts:
        if "C_Q" in site_parameter_names(d):
            _set(d, "C_Q", abs(_get(d, "C_Q")))
        if estimate_amounts:
            d["amount"] = 1.0
    dicts = resolve_link_dicts(dicts)
    curves = []
    for d, (axis, nu0) in zip(dicts, axes):
        site = Site.from_dict(d)
        comp, _p = _site_sticks(site, axis, nu0, mas_Hz, order, transitions, beta_r,
                                int(n_zcw))
        curves.append(np.sum(list(_broaden_site(site, comp, axis).values()), axis=0))
    if estimate_amounts:
        keys, cols = _owner_columns(curves, owners)
        A = np.column_stack([c[mask] for c in cols])
        amounts, _res = nnls(A, y[mask] - (b0 + b1 * (x[mask] - x_ref)))
        amounts = np.maximum(amounts, 1e-6)
        by_owner = dict(zip(keys, amounts))
        curves = [c * by_owner[o] for c, o in zip(curves, owners)]
        for d, o in zip(dicts, owners):
            d["amount"] = float(d["amount"] * by_owner[o])
    baseline = (b0 + b1 * (x - x_ref)) * scale
    site_spectra = {d["label"]: c * scale for d, c in zip(dicts, curves)}
    start = np.sum(list(site_spectra.values()), axis=0) + baseline
    return {"x": x, "y": y_raw, "mask": mask, "start": start, "site_spectra": site_spectra,
            "baseline": baseline, "n_zcw": int(n_zcw),
            "amounts": {d["label"]: d["amount"] * scale for d in dicts}}


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


def _derived(parameters, pcov, free_names, sites, larmor_Hz, sources=None):
    """Shares (per nucleus: amounts of different nuclei are not numbers of
    the same nuclei) and quadrupolar products. ``sources``: from
    :func:`_linked_sources`, for a linked partner's parameters (which have
    no rows of their own): it reports its parent's share, and its P_Q and
    centre of gravity from its own (derived) values with the parent's
    errors scaled."""
    sources = sources or {}
    nucleus = {s.label: s.nucleus for s in sites}
    amounts = {p["label"]: p for p in parameters if p["name"] == "amount"}
    out = {}
    for s in sites:
        if s.label not in amounts:
            continue
        group = {lab: p for lab, p in amounts.items() if nucleus.get(lab) == s.nucleus}
        total = sum(p["value"] for p in group.values()) or 1.0
        a = amounts[s.label]
        row = {"share_%": 100 * a["value"] / total}
        # share error from the amounts' covariance
        names = [f"{lab}: amount" for lab in group]
        cov = _cov_of(names, pcov, free_names)
        g = np.array([(-a["value"] / total ** 2 + (1 / total if lab == s.label else 0.0))
                      for lab in group]) * 100
        row["share_%_err"] = float(np.sqrt(max(g @ cov @ g, 0.0)))
        out[s.label] = row
    for s in sites:
        row = out.get(s.label)
        if row is None:          # a partner with a linked amount: its parent's share
            parent = sources.get((s.label, "amount"), (None,))[0]
            row = dict(out.get(parent, {"share_%": float("nan"), "share_%_err": 0.0}))
            row = {"share_%": row["share_%"], "share_%_err": row["share_%_err"],
                   "linked_to": parent}
            out[s.label] = row
        if s.get("quadrupolar"):
            q = s.get("quadrupolar")[0]
            cq, eq, d = abs(q["C_Q"]), q["eta_Q"], s.delta_iso
            cov_names, scale = [], []
            for k in ("delta_iso", "C_Q", "eta_Q"):
                src, factor = sources.get((s.label, k), (s.label, 1.0))
                cov_names.append(f"{src}: {k}")
                scale.append(factor)
            cov3 = _cov_of(cov_names, pcov, free_names) * np.outer(scale, scale)
            fac = np.sqrt(1 + eq ** 2 / 3)
            gp = np.array([0.0, fac, cq * eq / (3 * fac)])
            row.update(P_Q=cq * fac, P_Q_err=float(np.sqrt(max(gp @ cov3 @ gp, 0.0))))
            if not float(s.spin).is_integer():
                nu0 = larmor_Hz.get(s.label) if isinstance(larmor_Hz, dict) else larmor_Hz
                k = quadrupolar_ct_isotropic_shift_Hz(s.spin, 1e6, 0.0, nu0) / nu0 * 1e6
                gc = np.array([1.0, 2 * k * cq * fac ** 2, k * cq ** 2 * 2 * eq / 3])
                row.update(delta_cg=d + k * cq ** 2 * fac ** 2,
                           delta_cg_err=float(np.sqrt(max(gc @ cov3 @ gc, 0.0))))
    return {s.label: out[s.label] for s in sites}


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


def print_combined_fit_report(fit, save_txt=None, advice=True):
    """Print (and optionally save to ``save_txt``) the fitted parameters
    with 1-sigma errors (rounded to the digits the error supports), the
    derived quantities, fit quality, the setup and every warning, then
    (``advice``) what the fit tells you and what to try next, in plain
    words (:mod:`opennmr.notebook.fit_advice`). Returns the text."""
    lines = ["=" * 68, "COMBINED-HAMILTONIAN FIT REPORT", "=" * 68, fit["log"], ""]
    sites = {(s.label if hasattr(s, "label") else s["label"]): s for s in fit.get("sites") or []}
    labels = list(dict.fromkeys([p["label"] for p in fit["parameters"] if p["label"]]
                                + list(sites)))
    labels = [lab for lab in sites if lab in labels] + [lab for lab in labels if lab not in sites]
    for label in labels:
        site = sites.get(label)
        sd = site.to_dict() if hasattr(site, "to_dict") else (site or {})
        link = sd.get("link")
        lines.append(f"{label}" + (f"   [{sd['nucleus']}, linked to {link['to']}]" if link else ""))
        for p in fit["parameters"]:
            if p["label"] != label:
                continue
            unit = _UNITS.get(p["name"], "Hz" if p["name"].startswith("J") else "")
            name = f"{p['name']} ({unit})" if unit else p["name"]
            lines.append(f"  {name:<18}= {_fmt(p['value'], p['error'])}")
        if link:
            own = set(link.get("free") or [])
            for n in site_parameter_names(sd):
                if n in own:
                    continue
                unit = _UNITS.get(n, "Hz" if n.startswith("J") else "")
                name = f"{n} ({unit})" if unit else n
                lines.append(f"  {name:<18}= {_get(sd, n):.6g} (from {link['to']})")
        d = fit["derived"].get(label)
        if d is None:
            lines.append("")
            continue
        lines.append(f"  {'share (%)':<18}= {_fmt(d['share_%'], d['share_%_err'] or None)}")
        if "P_Q" in d:
            lines.append(f"  {'P_Q (MHz)':<18}= {_fmt(d['P_Q'], d['P_Q_err'] or None)}")
        if "delta_cg" in d:
            lines.append(f"  {'CT c.o.g. (ppm)':<18}= {_fmt(d['delta_cg'], d['delta_cg_err'] or None)}")
        lines.append("")
    q = fit["quality"]
    lines.append(f"R^2 = {q['r2']:.5f}   RMSE = {q['rmse']:.4g}   reduced chi^2 = "
                 f"{q['reduced_chi2']:.4g}")
    mas = next((p for p in fit["parameters"] if p["name"] == "mas_Hz"), None)
    if mas is not None:
        lines.append(f"MAS rate (Hz)      = {_fmt(mas['value'], mas['error'])}")
    if fit["warnings"]:
        lines += ["", "Warnings:"] + [f"  - {w}" for w in fit["warnings"]]
    if advice:
        from .fit_advice import advice_text, advise_combined_fit
        items = fit.get("advice")
        if items is None:
            items = advise_combined_fit(fit)
        if items:
            lines += ["", advice_text(items)]
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
