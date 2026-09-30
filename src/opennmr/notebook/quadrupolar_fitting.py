"""
Fit static or MAS quadrupolar powder lineshapes -- one or more sites,
the central transition alone or with the satellites -- to a 1D
spectrum, using the simulator in :mod:`opennmr.notebook.simulations`
(``quadrupolar_spectrum``) as the model.

Workflow
--------

    NMR1dimport -> estimate_quadrupolar_start (optional)
                -> fit_quadrupolar_spectrum -> NMR1dplot_with_quad_fit
                                            -> print_quadrupolar_fit_report

or, end to end, :func:`NMRQuadFitWorkflow`. From the GUI: Fit >
Quadrupolar, which calls the same functions.

Parameters per site: ``d_iso`` (ppm, the isotropic *chemical* shift),
``C_Q`` (MHz), ``eta_Q`` (0-1), ``fwhm`` (ppm, broadening), ``amplitude``
and ``eta`` (the pseudo-Voigt Lorentzian fraction of the broadening --
NOT eta_Q). Every site shares the spin, Larmor frequency, MAS rate,
level of theory and transitions, and one linear baseline (estimated
and held fixed, as in the CSA fit).

Two things the data cannot tell you, and the report says so:

* **the sign of C_Q** -- a lineshape depends on C_Q^2 (second order)
  or is symmetric under C_Q -> -C_Q (first order), so |C_Q| is fitted;
* **d_iso and C_Q separately from one CT at one field** are strongly
  correlated when eta_Q is free and the lineshape is poorly resolved
  (the CT sits at d_iso + an isotropic shift proportional to
  C_Q^2 (1 + eta_Q^2/3) / nu0). The report gives the CT centre of
  gravity and P_Q = C_Q sqrt(1 + eta_Q^2 / 3), which are well
  determined even then; a second field pins them apart.

A multi-site ``p0`` is a list of dicts; ``fixed``/``bounds`` work as in
:mod:`opennmr.notebook.csa_fitting`: global (applied to every site) or
per site (keyed by index or label), or inside each site's own dict.
"""

import csv

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from .utils import find_gamma, find_isotope_info, NUC_label
from .plotting import setup_plotting, apply_tick_style, apply_spine_style, save_figure
from .one_d import NMR1dimport
from .fitting import build_region_mask, estimate_baseline_robust, fit_quality, expand_pcov
from .simulations import (quadrupolar_spectrum, quadrupolar_ct_isotropic_shift_Hz,
                          quadrupolar_transition_table, QUADRUPOLAR_ORDERS)

QUAD_PARAM_NAMES = ["d_iso", "C_Q", "eta_Q", "fwhm", "amplitude", "eta"]
_N = len(QUAD_PARAM_NAMES)
_QUAD_PARAM_LABELS = {"d_iso": "d_iso (ppm)", "C_Q": "C_Q (MHz)", "eta_Q": "eta_Q",
                      "fwhm": "fwhm (ppm)", "amplitude": "amplitude",
                      "eta": "pv_mix_eta"}


# ============================================================
# Model
# ============================================================

def default_transitions(I):
    """'ct' for a half-integer spin (the usual measurement), 'all' for an
    integer spin, which has no central transition."""
    return "all" if float(I).is_integer() else "ct"


def quad_model_full(x_ppm_sorted, d_iso, C_Q_MHz, eta_Q, fwhm, eta=0.0, *, I, larmor_Hz,
                    mas_Hz=0.0, order="second", transitions=None, rotor_angle="magic",
                    n_zcw=4181, lineshape="gaussian", normalised=True):
    """
    One site's lineshape on ``x_ppm_sorted`` (ascending, evenly spaced),
    normalised to a maximum of 1 (``normalised=False``: the simulator's
    absolutely scaled ``real_spec`` instead -- see quad_site_amounts).

    ``C_Q_MHz`` in MHz; its sign is irrelevant (see the module notes).
    ``mas_Hz`` = 0 gives the static pattern.
    """
    if transitions is None:
        transitions = default_transitions(I)
    spec = quadrupolar_spectrum(
        I, abs(C_Q_MHz) * 1e6, float(np.clip(eta_Q, 0.0, 1.0)), larmor_Hz, x_ppm_sorted,
        mas_Hz=mas_Hz, order=order, transitions=transitions, delta_iso=d_iso,
        fwhm=max(float(fwhm), 1e-9), lineshape=lineshape, eta=float(np.clip(eta, 0.0, 1.0)),
        n_zcw=n_zcw, rotor_angle=rotor_angle)
    return spec["normalised_real_spec"] if normalised else spec["real_spec"]


def build_quad_model(x_ppm_sorted, params, n_sites, x_ref=None, **sim_kwargs):
    """``sum_i amplitude_i * quad_model_full(site_i) + b0 + b1 (x - x_ref)``;
    ``params`` = 6 values per site (``QUAD_PARAM_NAMES`` order) + (b0, b1)."""
    params = np.asarray(params, dtype=float)
    x_ppm_sorted = np.asarray(x_ppm_sorted, dtype=float)
    if params.shape[0] != _N * n_sites + 2:
        raise ValueError(f"params has length {params.shape[0]}, expected {_N * n_sites + 2}.")
    x_ref = np.mean(x_ppm_sorted) if x_ref is None else x_ref
    total = params[_N * n_sites] + params[_N * n_sites + 1] * (x_ppm_sorted - x_ref)
    for d_iso, cq, eta_q, fwhm, amp, eta in params[:_N * n_sites].reshape(n_sites, _N):
        total = total + amp * quad_model_full(x_ppm_sorted, d_iso, cq, eta_q, fwhm, eta,
                                              **sim_kwargs)
    return total


# ============================================================
# p0 / fixed / bounds
# ============================================================

def _normalise_sites(p0):
    if isinstance(p0, dict):
        p0 = [p0]
    if not isinstance(p0, (list, tuple)) or not p0:
        raise TypeError("p0 must be a dict (one site) or a non-empty list of dicts.")
    labels = []
    for i, site in enumerate(p0):
        if not isinstance(site, dict):
            raise TypeError(f"p0[{i}] must be a dict, got {type(site)}.")
        missing = [k for k in ("d_iso", "C_Q", "eta_Q", "fwhm") if k not in site]
        if missing:
            raise ValueError(f"p0[{i}] (label={site.get('label')!r}) is missing {missing}.")
        labels.append(site.get("label", f"Site {i + 1}"))
    if len(set(labels)) != len(labels):
        raise ValueError(f"Site labels must be unique, got {labels}.")
    return list(p0), labels


def _site_index(key, labels):
    if isinstance(key, (int, np.integer)) and not isinstance(key, bool):
        if 0 <= key < len(labels):
            return int(key)
        raise KeyError(f"Site index {key} out of range for {len(labels)} site(s).")
    if key in labels:
        return labels.index(key)
    raise KeyError(f"{key!r} is not a site label or index; labels are {labels}.")


def _fixed_matrix(fixed, sites, labels):
    matrix = [set() for _ in sites]
    if fixed:
        if isinstance(fixed, (list, tuple)):
            fixed = {name: True for name in fixed}
        for key, val in fixed.items():
            if key in QUAD_PARAM_NAMES:
                if val:
                    for m in matrix:
                        m.add(key)
            else:
                names = ([n for n, on in val.items() if on] if isinstance(val, dict) else list(val))
                matrix[_site_index(key, labels)].update(names)
    for i, site in enumerate(sites):
        own = site.get("fixed")
        if own:
            matrix[i].update([n for n, on in own.items() if on] if isinstance(own, dict)
                             else list(own))
    for i, names in enumerate(matrix):
        bad = names - set(QUAD_PARAM_NAMES)
        if bad:
            raise ValueError(f"Unknown parameter(s) {sorted(bad)} fixed for site {labels[i]!r}.")
    return matrix


def build_quad_bounds(sites, labels, bounds=None, positive_intensity=True):
    """Lower/upper bounds per site (then the caller appends the baseline's).
    Defaults: C_Q >= 0 (its sign is not observable), eta_Q in [0, 1],
    fwhm > 0, amplitude >= 0, pseudo-Voigt eta in [0, 1]."""
    base = {"d_iso": (-np.inf, np.inf), "C_Q": (0.0, np.inf), "eta_Q": (0.0, 1.0),
            "fwhm": (1e-6, np.inf),
            "amplitude": (0.0 if positive_intensity else -np.inf, np.inf), "eta": (0.0, 1.0)}
    per_site = [dict(base) for _ in sites]
    bounds = dict(bounds or {})
    for key, val in bounds.items():
        if key in QUAD_PARAM_NAMES:
            for b in per_site:
                b[key] = tuple(val)
    for key, val in bounds.items():
        if key not in QUAD_PARAM_NAMES:
            per_site[_site_index(key, labels)].update({k: tuple(v) for k, v in val.items()})
    for i, site in enumerate(sites):
        per_site[i].update({k: tuple(v) for k, v in (site.get("bounds") or {}).items()})
    lower, upper = [], []
    for b in per_site:
        for name in QUAD_PARAM_NAMES:
            lo, hi = b[name]
            lower.append(min(lo, hi))
            upper.append(max(lo, hi))
    return np.array(lower, dtype=float), np.array(upper, dtype=float)


# ============================================================
# Starting values from the data
# ============================================================

def _pattern_width(x, y, threshold=0.2):
    """Width (ppm) of the region where y exceeds ``threshold`` of its max,
    and the centre of gravity of y over that region."""
    y = np.clip(np.asarray(y, dtype=float), 0.0, None)
    if not y.size or y.max() <= 0:
        return 0.0, float(np.mean(x))
    keep = y >= threshold * y.max()
    xs = np.asarray(x)[keep]
    cg = float(np.sum(xs * y[keep]) / np.sum(y[keep]))
    return float(xs.max() - xs.min()), cg


def estimate_quadrupolar_start(results, I, larmor_Hz=None, mas_Hz=0.0, eta_Q=0.5, fwhm=None,
                               transitions=None, fit_regions=None, order="second",
                               n_zcw=2584, threshold=0.2, n_iter=6):
    """
    Starting values for a one-site fit, from the data: the pattern's
    width above ``threshold`` of its maximum sets C_Q (by matching the
    same width measured on simulations -- the CT width scales as C_Q^2,
    a first-order pattern's as C_Q), and its centre of gravity, less
    the CT's isotropic quadrupolar shift, sets d_iso.

    Returns a p0 dict ``{d_iso, C_Q (MHz), eta_Q, fwhm}``. A start, not
    a result: eta_Q is only a guess (``eta_Q``), and the fit refines all
    of them.
    """
    if transitions is None:
        transitions = default_transitions(I)
    if larmor_Hz is None:
        larmor_Hz = float(results["SFO1"]) * 1e6
    x = np.asarray(results["x_ppm"], dtype=float)
    order_idx = np.argsort(x)
    x, y = x[order_idx], np.asarray(results["real_spectrum"], dtype=float)[order_idx]
    mask = build_region_mask(x, fit_regions)
    x, y = x[mask], y[mask]
    y = y - np.percentile(y, 5)
    width_data, cg = _pattern_width(x, y, threshold)
    step = abs(x[1] - x[0]) if x.size > 1 else 1.0
    if fwhm is None:
        fwhm = max(3 * step, 0.02 * width_data)
    ct_led = transitions in ("ct",) or (transitions == "all" and not float(I).is_integer())
    power = 2.0 if ct_led else 1.0
    # simulate on a generous grid around the data's centre
    grid = np.linspace(cg - 3 * width_data - 10 * fwhm, cg + 3 * width_data + 10 * fwhm, 4096)
    cq = 1.0  # MHz
    for _ in range(n_iter):
        sim = quad_model_full(grid, 0.0, cq, eta_Q, fwhm, I=I, larmor_Hz=larmor_Hz,
                              mas_Hz=mas_Hz, order=order, transitions=transitions, n_zcw=n_zcw)
        width_sim, _ = _pattern_width(grid, sim, threshold)
        if width_sim <= 0 or width_data <= 0:
            break
        new = cq * (width_data / width_sim) ** (1.0 / power)
        if abs(new - cq) < 1e-3 * cq:
            cq = new
            break
        cq = float(np.clip(new, 1e-3, 1e3))
    shift = 0.0
    if not float(I).is_integer() and transitions in ("ct", "all"):
        shift = quadrupolar_ct_isotropic_shift_Hz(I, cq * 1e6, eta_Q, larmor_Hz) / larmor_Hz * 1e6
    return {"d_iso": cg - shift, "C_Q": cq, "eta_Q": eta_Q, "fwhm": fwhm}


# ============================================================
# Fitting
# ============================================================

def fit_quadrupolar_spectrum(results, p0, I=None, nucleus=None, larmor_Hz=None, mas_Hz=0.0,
                             order="second", transitions=None, rotor_angle="magic",
                             n_zcw=None, lineshape="gaussian", fit_regions=None,
                             baseline_regions=None, baseline_sigma=2.5, baseline_max_iter=10,
                             bounds=None, fixed=None, positive_intensity=True,
                             fit_baseline=False):
    """
    Fit one or more quadrupolar sites to a 1D spectrum.

    Parameters
    ----------
    results : dict
        :func:`opennmr.notebook.one_d.NMR1dimport` output (or the GUI's
        ``bridge.results_from_spectrum``).
    p0 : dict or list of dict
        Per site ``{"d_iso": ppm, "C_Q": MHz, "eta_Q": ..., "fwhm": ppm,
        ["amplitude"], ["eta"], ["label"], ["fixed"], ["bounds"]}``.
        See :func:`estimate_quadrupolar_start` for a data-driven start.
    I, nucleus : the spin, given directly or read from the nuclide
        (default: ``results['nuclide']``).
    larmor_Hz : default ``results['SFO1'] * 1e6``.
    mas_Hz : 0 for a static spectrum, else the spinning rate (Hz).
    order : 'second' (default: fast) or 'exact' (slower, esp. under
        MAS; worth it only at low field / very large C_Q), or 'first'.
    transitions : 'ct' (default for half-integer spins), 'all', 'st', ...
    n_zcw : powder orientations while fitting. Default 28657 static,
        4181 MAS (each MAS orientation costs an FFT). Too few leaves
        powder "ripple" that the fit absorbs into a larger fwhm; check
        the result is unchanged at a larger value.
    lineshape : broadening kernel ('gaussian', 'lorentzian', 'pseudo_voigt').
    fit_regions : [(lo, hi), ...] ppm, or None for the whole spectrum.
    bounds, fixed : global and/or per site, as in csa_fitting.
    fit_baseline : bool
        Also fit the linear baseline (default: estimated from the ends
        of the spectrum and held, as the CSA fit does). Worth it when a
        broad pattern leaves little baseline at the ends -- an estimate
        taken from pattern, not baseline, biases every parameter.

    Returns
    -------
    popt, pcov_free, is_free, scale, fit_info
        As :func:`opennmr.notebook.csa_fitting.fit_csa_spectrum`:
        ``popt`` in raw intensity units, 6 values per site
        (``QUAD_PARAM_NAMES``) then (b0, b1); ``fit_info`` carries
        ``sim_kwargs``, ``labels``, ``x_ref``, ``fit_regions``,
        ``warnings`` and a one-line ``log``.
    """
    if I is None:
        nucleus = nucleus or results.get("nuclide")
        if not nucleus:
            raise ValueError("Give I, or a nucleus (results has no 'nuclide').")
        I = float(find_isotope_info(nucleus)["spin"])
    I = float(I)
    if I < 1:
        raise ValueError(f"I = {I:g} has no quadrupole moment.")
    if larmor_Hz is None:
        larmor_Hz = float(results["SFO1"]) * 1e6
    if order not in QUADRUPOLAR_ORDERS:
        raise ValueError(f"order must be one of {QUADRUPOLAR_ORDERS}.")
    if transitions is None:
        transitions = default_transitions(I)
    lineshape = str(lineshape).lower().replace("-", "_")
    if lineshape == "pv":
        lineshape = "pseudo_voigt"
    if lineshape not in ("gaussian", "lorentzian", "pseudo_voigt"):
        raise ValueError(f"Unknown lineshape {lineshape!r}.")
    mas_Hz = float(mas_Hz or 0.0)
    if n_zcw is None:
        n_zcw = 4181 if mas_Hz else 28657

    sites, labels = _normalise_sites(p0)
    n_sites = len(sites)

    x_full = np.asarray(results["x_ppm"], dtype=float)
    y_raw = np.asarray(results["real_spectrum"], dtype=float)
    scale = float(np.max(np.abs(y_raw))) or 1.0
    idx = np.argsort(x_full)
    x_sorted, y_sorted = x_full[idx], y_raw[idx] / scale
    mask = build_region_mask(x_sorted, fit_regions)
    x_fit, y_fit = x_sorted[mask], y_sorted[mask]
    if x_fit.size < 8:
        raise ValueError("The fit window holds fewer than 8 points.")
    x_ref = float(np.mean(x_fit))
    b0, b1, _ = estimate_baseline_robust(x_sorted, y_sorted, baseline_regions=baseline_regions,
                                         sigma=baseline_sigma, max_iter=baseline_max_iter,
                                         x_ref=x_ref)

    values = []
    for site in sites:
        amp = (float(site["amplitude"]) / scale if "amplitude" in site
               else max(float(np.max(y_fit) - b0), 1e-3) / n_sites)
        default_eta = 0.5 if lineshape == "pseudo_voigt" else (1.0 if lineshape == "lorentzian" else 0.0)
        values += [float(site["d_iso"]), abs(float(site["C_Q"])), float(site["eta_Q"]),
                   float(site["fwhm"]), amp, float(site.get("eta", default_eta))]
    p0_arr = np.array(values + [b0, b1], dtype=float)

    fixed_m = _fixed_matrix(fixed, sites, labels)
    is_free = np.array([name not in fixed_m[s] for s in range(n_sites) for name in QUAD_PARAM_NAMES]
                       + [bool(fit_baseline)] * 2)
    eta_idx = np.arange(n_sites) * _N + 5
    if lineshape != "pseudo_voigt":
        p0_arr[eta_idx] = 1.0 if lineshape == "lorentzian" else 0.0
        is_free[eta_idx] = False

    lower, upper = build_quad_bounds(sites, labels, bounds, positive_intensity)
    lower = np.append(lower, [-np.inf, -np.inf])
    upper = np.append(upper, [np.inf, np.inf])
    amp_idx = np.arange(n_sites) * _N + 4
    lower[amp_idx] /= scale
    upper[amp_idx] /= scale
    notes = []
    names = [f"{lab}: {n}" for lab in labels for n in QUAD_PARAM_NAMES] + ["b0", "b1"]
    for i in np.flatnonzero(is_free):
        if not lower[i] <= p0_arr[i] <= upper[i]:
            moved = float(np.clip(p0_arr[i], lower[i], upper[i]))
            notes.append(f"{names[i]}: starting value {p0_arr[i]:g} is outside its bounds "
                         f"({lower[i]:g}, {upper[i]:g}); started at {moved:g} instead")
            p0_arr[i] = moved

    sim_kwargs = dict(I=I, larmor_Hz=float(larmor_Hz), mas_Hz=mas_Hz, order=order,
                      transitions=transitions, rotor_angle=rotor_angle, n_zcw=int(n_zcw),
                      lineshape=lineshape)

    def model(_x, *free):
        full = p0_arr.copy()
        full[is_free] = free
        return build_quad_model(x_sorted, full, n_sites, x_ref=x_ref, **sim_kwargs)[mask]

    # Relative steps sized to the physics: the default (~1.5e-8 of the
    # value) is below the powder pattern's own bin-to-bin roughness.
    popt_free, pcov_free = curve_fit(model, x_fit, y_fit, p0=p0_arr[is_free],
                                     bounds=(lower[is_free], upper[is_free]),
                                     diff_step=1e-4, maxfev=5000)
    popt = p0_arr.copy()
    popt[is_free] = popt_free
    unit = np.ones(len(popt))
    unit[amp_idx] = scale
    unit[-2:] = scale
    popt = popt * unit
    pcov_free = np.asarray(pcov_free) * np.outer(unit[is_free], unit[is_free])

    warnings = list(notes)
    if fit_baseline:
        zone = None
    elif baseline_regions is not None:
        zone = build_region_mask(x_sorted, [baseline_regions] if np.isscalar(baseline_regions[0])
                                 else baseline_regions)
    else:
        k = max(1, int(x_sorted.size * 0.05))
        zone = np.zeros(x_sorted.size, dtype=bool)
        zone[:k] = zone[-k:] = True
    warnings += _edge_warnings(x_sorted, mask, popt / unit, n_sites, x_ref, sim_kwargs, zone)
    warnings += _physics_warnings(popt, pcov_free, is_free, labels, I, larmor_Hz, mas_Hz,
                                  transitions, order)
    regime = "static" if not mas_Hz else f"MAS {mas_Hz / 1e3:g} kHz"
    groups = sorted({r["group"] for r in quadrupolar_transition_table(I)
                     if transitions == "all" or (transitions == "ct" and r["group"] == "CT")
                     or (transitions == "st" and r["group"] != "CT")
                     or r["group"] == str(transitions).upper()}, key=lambda g: (g != "CT", g))
    log = (f"Quadrupolar fit: I = {I:g}, nu0 = {larmor_Hz / 1e6:.6g} MHz, {regime}, "
           f"{order}-order model" if order != "exact" else
           f"Quadrupolar fit: I = {I:g}, nu0 = {larmor_Hz / 1e6:.6g} MHz, {regime}, exact model")
    log += (f", transitions {', '.join(groups)}, {lineshape} broadening, {n_sites} site(s), "
            f"{n_zcw} orientations, window "
            + ("whole spectrum" if fit_regions is None else
               ", ".join(f"{max(a, b):g} to {min(a, b):g} ppm" for a, b in fit_regions)) + ".")
    fit_info = {"sim_kwargs": sim_kwargs, "n_sites": n_sites, "labels": labels, "x_ref": x_ref,
                "fit_regions": (None if fit_regions is None else
                                [(float(min(a, b)), float(max(a, b))) for a, b in fit_regions]),
                "warnings": warnings, "log": log, "lower": lower * unit, "upper": upper * unit}
    return popt, pcov_free, is_free, scale, fit_info


def _edge_warnings(x_sorted, mask, params, n_sites, x_ref, sim_kwargs, baseline_zone=None):
    """A fitted pattern that runs past the fit window is clipped -- and the
    baseline, estimated from the spectrum's ends, may then be sitting on
    it. Both bias every parameter. Checked two ways: the site's full
    second-order frequency range against the window, and the fitted
    intensity at the window's edges."""
    from .simulations import quadrupolar_axis_limits
    x_win = x_sorted[mask]
    if not x_win.size:
        return []
    w_lo, w_hi = float(x_win.min()), float(x_win.max())
    out = []
    for s in range(n_sites):
        d, cq, eta_q = params[s * _N], abs(params[s * _N + 1]), float(np.clip(params[s * _N + 2], 0, 1))
        if cq <= 0:
            continue
        try:
            lo, hi = quadrupolar_axis_limits(sim_kwargs["I"], cq * 1e6, eta_q,
                                             sim_kwargs["larmor_Hz"],
                                             transitions=sim_kwargs["transitions"],
                                             delta_iso=d, margin=0.0)
        except ValueError:
            continue
        lo, hi = lo + 1.0, hi - 1.0          # undo quadrupolar_axis_limits' 1 ppm minimum pad
        beyond = max(w_lo - lo, hi - w_hi, 0.0)
        if beyond > 0.02 * max(hi - lo, 1e-9):
            out.append(f"Site {s + 1}: its fitted pattern spans {hi:.4g} to {lo:.4g} ppm, beyond "
                       f"the fit window ({w_hi:.4g} to {w_lo:.4g} ppm) -- it is clipped, and the "
                       "baseline (estimated from the spectrum's ends) may be sitting on it. "
                       "Widen the window/spectrum, or fit the baseline (fit_baseline=True).")
    base = params[_N * n_sites] + params[_N * n_sites + 1] * (x_sorted - x_ref)
    sites = build_quad_model(x_sorted, params, n_sites, x_ref=x_ref, **sim_kwargs) - base
    peak = np.max(np.abs(sites[mask]))
    edges = np.flatnonzero(np.diff(np.concatenate([[False], mask, [False]]).astype(int)))
    idx = np.clip(np.concatenate([edges[0::2], edges[1::2] - 1]), 0, x_sorted.size - 1)
    if not out and peak > 0 and np.max(np.abs(sites[idx])) > 0.05 * peak:
        out.append(f"The fitted pattern still has {np.max(np.abs(sites[idx])) / peak:.0%} of its "
                   "height at an edge of the fit window: it is clipped. Widen the window, or "
                   "fit the baseline (fit_baseline=True).")
    if baseline_zone is not None and baseline_zone.any() and peak > 0:
        inside = float(np.max(np.abs(sites[baseline_zone]))) / peak
        if inside > 0.01:
            out.append(f"The baseline was estimated where the fitted pattern still has "
                       f"{inside:.0%} of its height (the spectrum's outer 5% at each end, or "
                       "baseline_regions): it is partly the pattern's tail, which biases "
                       "amplitude, d_iso and C_Q. Give baseline_regions clear of the pattern, "
                       "or fit the baseline (fit_baseline=True).")
    return out


def _physics_warnings(popt, pcov_free, is_free, labels, I, larmor_Hz, mas_Hz, transitions, order):
    warnings = []
    pcov = expand_pcov(pcov_free, is_free, len(popt))
    for s, label in enumerate(labels):
        cq = popt[s * _N + 1]
        if is_free[s * _N + 1] and cq < 0.02:
            warnings.append(f"{label}: C_Q fitted to ~0 -- the lineshape is not resolving a "
                            "quadrupolar splitting; the broadening is doing the work.")
        nu_q = 3 * cq * 1e6 / (2 * I * (2 * I - 1))
        if order != "exact" and nu_q > 0.1 * larmor_Hz:
            warnings.append(f"{label}: nu_Q is {nu_q / larmor_Hz:.0%} of the Larmor frequency; "
                            "perturbation theory is no longer reliable -- refit with the exact model.")
        i_d, i_c = s * _N, s * _N + 1
        if is_free[i_d] and is_free[i_c] and pcov[i_d, i_d] > 0 and pcov[i_c, i_c] > 0:
            r = pcov[i_d, i_c] / np.sqrt(pcov[i_d, i_d] * pcov[i_c, i_c])
            if abs(r) > 0.95:
                warnings.append(f"{label}: d_iso and C_Q are {abs(r):.0%} correlated -- quote "
                                "the CT centre of gravity and P_Q, or add a second field.")
    return warnings


def quad_site_amounts(x_ppm, popt, fit_info, normalise=True):
    """Each fitted site's ``amount`` (relative number of nuclei, as
    quadrupolar_multisite_spectrum weights it) from the fit's
    amplitudes: amplitude / the site's un-normalised peak height on the
    fit's own axis (see csa_fitting.csa_site_amounts). Largest = 1 with
    ``normalise``."""
    from .simulations import site_amounts_from_amplitudes
    xs = np.sort(np.asarray(x_ppm, dtype=float))
    n = fit_info["n_sites"]
    params = np.asarray(popt[:_N * n], dtype=float).reshape(n, _N)
    heights = [float(np.max(quad_model_full(xs, p[0], p[1], p[2], p[3], p[5], normalised=False,
                                            **fit_info["sim_kwargs"])))
               for p in params]
    return site_amounts_from_amplitudes(params[:, 4], heights, normalise=normalise)


def evaluate_quad_fit(x_ppm, popt, fit_info, return_components=False, n_zcw=None):
    """The fitted model at ``x_ppm`` (any order; must be evenly spaced).
    ``n_zcw`` overrides the orientation count (e.g. finer for a final
    curve). With ``return_components``: ``(y, [site_1, ..., baseline])``."""
    x = np.asarray(x_ppm, dtype=float)
    idx = np.argsort(x)
    xs = x[idx]
    kw = dict(fit_info["sim_kwargs"])
    if n_zcw:
        kw["n_zcw"] = int(n_zcw)
    n = fit_info["n_sites"]
    base = popt[_N * n] + popt[_N * n + 1] * (xs - fit_info["x_ref"])
    comps = [p[4] * quad_model_full(xs, p[0], p[1], p[2], p[3], p[5], **kw)
             for p in np.asarray(popt[:_N * n]).reshape(n, _N)]
    total = base + np.sum(comps, axis=0) if comps else base

    def unsort(a):
        out = np.empty_like(a)
        out[idx] = a
        return out
    if not return_components:
        return unsort(total)
    return unsort(total), [unsort(c) for c in comps] + [unsort(base)]


# ============================================================
# Derived quantities, report
# ============================================================

def quadrupolar_derived(popt, pcov_full, fit_info):
    """Per site: P_Q (MHz) = C_Q sqrt(1 + eta_Q^2/3), and the CT centre
    of gravity delta_cg (ppm) = d_iso + the CT isotropic quadrupolar
    shift (half-integer spins), each with a 1-sigma error propagated
    from the covariance. Returns a list of dicts."""
    I = fit_info["sim_kwargs"]["I"]
    nu0 = fit_info["sim_kwargs"]["larmor_Hz"]
    out = []
    for s in range(fit_info["n_sites"]):
        d, cq, eta_q = popt[s * _N], abs(popt[s * _N + 1]), popt[s * _N + 2]
        cov = pcov_full[np.ix_([s * _N, s * _N + 1, s * _N + 2], [s * _N, s * _N + 1, s * _N + 2])]
        fac = np.sqrt(1 + eta_q ** 2 / 3)
        pq = cq * fac
        g_pq = np.array([0.0, fac, cq * eta_q / (3 * fac)])
        row = {"P_Q": pq, "P_Q_err": float(np.sqrt(max(g_pq @ cov @ g_pq, 0.0)))}
        if not float(I).is_integer():
            k = quadrupolar_ct_isotropic_shift_Hz(I, 1e6, 0.0, nu0) / nu0 * 1e6  # ppm per MHz^2
            cg = d + k * cq ** 2 * (1 + eta_q ** 2 / 3)
            g_cg = np.array([1.0, 2 * k * cq * (1 + eta_q ** 2 / 3), k * cq ** 2 * 2 * eta_q / 3])
            row.update({"delta_cg": cg, "delta_cg_err": float(np.sqrt(max(g_cg @ cov @ g_cg, 0.0)))})
        out.append(row)
    return out


def check_quad_bound_hits(popt, pcov_full, is_free, fit_info):
    """Free parameters sitting on (within 1 sigma of) a bound."""
    warnings = []
    lower, upper = fit_info.get("lower"), fit_info.get("upper")
    if lower is None:
        return warnings
    for s, label in enumerate(fit_info["labels"]):
        for j, name in enumerate(QUAD_PARAM_NAMES):
            i = s * _N + j
            if not is_free[i] or name == "C_Q" and lower[i] == 0.0 and popt[i] > 0.05:
                continue
            err = np.sqrt(max(pcov_full[i, i], 0.0))
            err += 1e-9 * max(1.0, abs(popt[i]))
            if np.isfinite(upper[i]) and popt[i] + err >= upper[i]:
                warnings.append(f"{label}: {name} at its upper bound {upper[i]:g}")
            if np.isfinite(lower[i]) and popt[i] - err <= lower[i]:
                warnings.append(f"{label}: {name} at its lower bound {lower[i]:g}")
    return warnings


def print_quadrupolar_fit_report(x, y, y_fit, popt, pcov_full, is_free, fit_info,
                                 save_txt=False, txt_filename="quadrupolar_fit_report.txt"):
    """Print (and optionally save) the fitted parameters with 1-sigma errors,
    P_Q and the CT centre of gravity, fit quality, the setup line and
    every warning. Returns the report as a string."""
    errors = np.sqrt(np.clip(np.diag(pcov_full), 0.0, None))
    metrics = fit_quality(np.asarray(y), np.asarray(y_fit), int(np.sum(is_free)))
    total = float(np.sum((np.asarray(y) - np.mean(y)) ** 2))
    r2 = 1 - float(np.sum((np.asarray(y) - np.asarray(y_fit)) ** 2)) / total if total else np.nan
    derived = quadrupolar_derived(popt, pcov_full, fit_info)
    lines = ["=" * 64, "QUADRUPOLAR FIT REPORT", "=" * 64, fit_info.get("log", ""), ""]
    for s, label in enumerate(fit_info["labels"]):
        lines.append(f"{label}")
        for j, name in enumerate(QUAD_PARAM_NAMES):
            i = s * _N + j
            if name == "eta" and fit_info["sim_kwargs"]["lineshape"] != "pseudo_voigt":
                continue
            text = (f"{popt[i]:.6g} (fixed)" if not is_free[i]
                    else f"{popt[i]:.6g} ± {errors[i]:.2g}")
            lines.append(f"  {_QUAD_PARAM_LABELS[name]:<16}= {text}")
        d = derived[s]
        lines.append(f"  {'P_Q (MHz)':<16}= {d['P_Q']:.6g} ± {d['P_Q_err']:.2g}")
        if "delta_cg" in d:
            lines.append(f"  {'CT c.o.g. (ppm)':<16}= {d['delta_cg']:.6g} ± {d['delta_cg_err']:.2g}")
        lines.append("")
    lines += [f"R^2 = {r2:.5f}   RMSE = {metrics['rmse']:.4g}   reduced chi^2 = "
              f"{metrics['reduced_chi2']:.4g}",
              "Sign of C_Q is not determined by a lineshape; |C_Q| is reported."]
    warns = list(fit_info.get("warnings", [])) + check_quad_bound_hits(popt, pcov_full, is_free,
                                                                       fit_info)
    if warns:
        lines += ["", "Warnings:"] + [f"  - {w}" for w in warns]
    lines.append("=" * 64)
    report = "\n".join(lines)
    print(report)
    if save_txt:
        with open(txt_filename, "w", encoding="utf-8") as f:
            f.write(report)
    return report


# ============================================================
# Plot and workflow
# ============================================================

def NMR1dplot_with_quad_fit(results, popt, fit_info, fit_regions=None, show_components=True,
                            show_residual=True, xlims='default', xlabel=None,
                            pl_width=3.375, pl_height=3.375, font='Arial', fontsize=12,
                            output_format='', colour_data='black', colour_fit='#D55E00',
                            colour_components=None, tick_dir='in', tick_length=4.0,
                            tick_width=1.0, tick_pad=4.0, tick_number=None, n_zcw_final=None,
                            save_fig=False, fig_name='nmr_quad_fit_plot', save_format='png',
                            component_style=None):
    """Data, fit, each site (dashed) and, below, the residual -- on the
    spectrum's own descending ppm axis, normalised to the data maximum.

    ``component_style`` : a :func:`opennmr.notebook.fit_display.component_style`
    dict (or keyword dict) -- show/hide/highlight the sites, fill them or
    not, their colours -- drawn exactly as the GUI's Fit view draws it
    (``canvas.fit_component_style()`` gives the GUI's current one). When
    given it replaces ``show_components``/``colour_*``."""
    style = setup_plotting(font=font, fontsize=fontsize, pl_width=pl_width,
                           pl_height=pl_height, output_format=output_format)
    x = np.asarray(results["x_ppm"], dtype=float)
    idx = np.argsort(x)
    x = x[idx]
    y = np.asarray(results["real_spectrum"], dtype=float)[idx]
    scale = float(np.max(np.abs(y))) or 1.0
    y_fit, comps = evaluate_quad_fit(x, popt, fit_info, return_components=True, n_zcw=n_zcw_final)
    mask = build_region_mask(x, fit_regions if fit_regions is not None else fit_info.get("fit_regions"))
    if show_residual:
        fig, (ax, rax) = plt.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [4, 1],
                                                                       "hspace": 0.05})
    else:
        fig, ax = plt.subplots()
        rax = None
    if component_style is not None:
        from .fit_display import draw_fit_components
        draw_fit_components(ax, x[mask], y[mask] / scale, y_fit[mask] / scale,
                            components=[np.asarray(c)[mask] / scale for c in comps],
                            labels=list(fit_info["labels"]), style=component_style,
                            auto_hide_single=True, fit_label="Fit")
    else:
        ax.plot(x[mask], y[mask] / scale, color=colour_data, lw=1.2, label="Data")
        ax.plot(x[mask], y_fit[mask] / scale, color=colour_fit, lw=1.5, label="Fit")
    if component_style is None and show_components and fit_info["n_sites"] > 1:
        from ..plotting.accessibility import ordered_qualitative_colours
        colours = colour_components or ordered_qualitative_colours("okabe_ito")[2:]
        for i, (c, label) in enumerate(zip(comps[:-1], fit_info["labels"])):
            ax.plot(x[mask], (c + comps[-1])[mask] / scale, ls="--", lw=1.0,
                    color=colours[i % len(colours)], label=label)
    ax.legend(frameon=False, fontsize=fontsize * 0.8)
    ax.set_yticks([])
    apply_spine_style(ax, top=False, right=False, bottom=True, left=False)
    apply_tick_style(ax, direction=tick_dir, length=tick_length, width=tick_width, pad=tick_pad,
                     num_ticks=tick_number, axis='x')
    last = ax
    if rax is not None:
        rax.plot(x[mask], (y - y_fit)[mask] / scale, color="0.4", lw=0.8)
        rax.axhline(0, color="0.7", lw=0.6)
        rax.set_yticks([])
        apply_spine_style(rax, top=False, right=False, bottom=True, left=False)
        apply_tick_style(rax, direction=tick_dir, length=tick_length, width=tick_width,
                         pad=tick_pad, num_ticks=tick_number, axis='x')
        last = rax
    xm = x[mask]
    if xlims == 'default':
        ax.set_xlim(xm.max(), xm.min())
    else:
        ax.set_xlim(max(xlims), min(xlims))
    if xlabel is None:
        nuc = results.get("nuclide", "")
        try:
            nuc = find_gamma(nuc, SFO1=results.get("SFO1", 0))["Nuclide"]
        except Exception:  # noqa: BLE001 -- a label must not sink a plot
            pass
        xlabel = NUC_label(nuc, x_unit="ppm") if nuc else "Shift (ppm)"
    last.set_xlabel(xlabel)
    fig.set_figwidth(style["pl_width"])
    fig.set_figheight(style["pl_height"] * (1.25 if rax is not None else 1.0))
    if save_fig:
        save_figure(fig, fig_name=fig_name, save_format=save_format)
    return fig, ax


def NMRQuadFitWorkflow(datapath, p0=None, procno=1, I=None, nucleus=None, larmor_Hz=None,
                       mas_Hz=0.0, order="second", transitions=None, rotor_angle="magic",
                       n_zcw=None, n_zcw_final=28657, lineshape="gaussian", fit_regions=None,
                       bounds=None, fixed=None, eta_Q_start=0.5, show_plot=True,
                       show_report=True, save_csv=False, csv_filename="quad_fit_results.csv",
                       save_report=False, report_filename="quad_fit_report.txt",
                       save_fig=False, fig_name='nmr_quad_fit_plot', save_format='png',
                       **plot_kwargs):
    """
    Import a processed 1D Bruker spectrum, fit quadrupolar site(s), and
    plot/report. ``p0=None`` starts a single site from
    :func:`estimate_quadrupolar_start`. Returns a dict with ``popt``,
    ``pcov`` (full), ``fit_info``, ``derived``, ``x``, ``y``, ``y_fit``,
    ``components``, ``report``, ``figure``, ``axes``.
    """
    results = NMR1dimport(datapath, procno=procno)
    if I is None:
        I = float(find_isotope_info(nucleus or results["nuclide"])["spin"])
    if p0 is None:
        p0 = estimate_quadrupolar_start(results, I, larmor_Hz=larmor_Hz, mas_Hz=mas_Hz,
                                        eta_Q=eta_Q_start, transitions=transitions,
                                        fit_regions=fit_regions, order="second")
    popt, pcov_free, is_free, _scale, info = fit_quadrupolar_spectrum(
        results, p0, I=I, larmor_Hz=larmor_Hz, mas_Hz=mas_Hz, order=order,
        transitions=transitions, rotor_angle=rotor_angle, n_zcw=n_zcw, lineshape=lineshape,
        fit_regions=fit_regions, bounds=bounds, fixed=fixed)
    pcov = expand_pcov(pcov_free, is_free, len(popt))
    x = np.asarray(results["x_ppm"], dtype=float)
    idx = np.argsort(x)
    x = x[idx]
    y = np.asarray(results["real_spectrum"], dtype=float)[idx]
    y_fit, comps = evaluate_quad_fit(x, popt, info, return_components=True, n_zcw=n_zcw_final)
    mask = build_region_mask(x, fit_regions)
    report = None
    if show_report or save_report:
        report = print_quadrupolar_fit_report(x[mask], y[mask], y_fit[mask], popt, pcov, is_free,
                                              info, save_txt=save_report,
                                              txt_filename=report_filename)
    fig = ax = None
    if show_plot:
        fig, ax = NMR1dplot_with_quad_fit(results, popt, info, fit_regions=fit_regions,
                                          n_zcw_final=n_zcw_final, save_fig=save_fig,
                                          fig_name=fig_name, save_format=save_format,
                                          **plot_kwargs)
    if save_csv:
        with open(csv_filename, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["x_ppm", "y", "y_fit"] + [f"{l}_component" for l in info["labels"]] + ["baseline"]
            writer.writerow(header)
            for i in np.flatnonzero(mask):
                writer.writerow([x[i], y[i], y_fit[i]] + [c[i] for c in comps])
    return {"popt": popt, "pcov": pcov, "is_free": is_free, "fit_info": info,
            "derived": quadrupolar_derived(popt, pcov, info), "x": x[mask], "y": y[mask],
            "y_fit": y_fit[mask], "components": [c[mask] for c in comps], "report": report,
            "figure": fig, "axes": ax}
