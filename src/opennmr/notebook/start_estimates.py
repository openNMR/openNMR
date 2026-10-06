"""
Good starting values for a fit, estimated from the spectrum (explicit user
request, 2026-10-06: "given a certain model -- one type of interaction --
estimate what good starting values for a fit would be").

One interaction type at a time:

* ``"isotropic"`` -- peaks picked from the smoothed spectrum: shift, width
  (FWHM at half height) and amount (height × width) of each.
* ``"jcoupling"`` -- one first-order multiplet: its centre, J from the
  median line spacing, the number of equivalent partners from the number
  of lines (n + 1 for spin-1/2 partners), its linewidth.
* ``"csa"`` and ``"quadrupolar"`` -- one site, static or MAS, from two
  properties a powder pattern provably keeps:

  - its **first moment** (centre of mass) is δiso, plus, for a
    quadrupolar nucleus, the second-order quadrupolar shift -- so δiso
    is read off once the coupling is known;
  - its **second moment** (spread) grows with the size of the
    interaction, and MAS does not change it: the sideband manifold keeps
    the static second moment. So the size -- Δ, or C_Q -- is whatever
    makes a simulation's spread match the data's, found in a few quick
    low-resolution simulations, for each asymmetry η (and, for a CSA,
    each sign of Δ).

  The (η, sign) whose shape then matches the data best is chosen. This
  needs no rule of thumb about edges or horns, which broadening,
  sidebands and second-order shapes all defeat; the simulations use the
  fit's own model, regime and transitions.

Moments are taken only over the stretches where there is signal (above 3σ
of the noise, widened a little), so noise elsewhere in the window does not
inflate the spread. The data's baseline is removed first (robust linear).

    from opennmr.notebook import start_estimates as se
    est = se.estimate_start(results, "csa", nucleus="13C", mas_Hz=12e3)
    print("\\n".join(est["notes"]))
    fit = fit_combined(results, est["sites"], mas_Hz=12e3)
"""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks, peak_widths

from .combined_hamiltonian import Site, simulate_combined
from .fitting import build_region_mask, estimate_baseline_robust

KINDS = ("isotropic", "csa", "quadrupolar", "jcoupling")
#: Asymmetries tried for each anisotropic interaction; the best is then
#: refined on a finer step around it.
ETA_GRID = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


def _prepare(results, fit_regions):
    x = np.asarray(results["x_ppm"], dtype=float)
    order = np.argsort(x)
    x = x[order]
    y = np.asarray(results["real_spectrum"], dtype=float)[order]
    b0, b1, _ = estimate_baseline_robust(x, y, x_ref=float(np.mean(x)))
    y = y - (b0 + b1 * (x - float(np.mean(x))))
    mask = build_region_mask(x, fit_regions)
    return x, y, mask


def _noise(y):
    d = np.diff(y)
    return max(1.4826 * float(np.median(np.abs(d - np.median(d)))) / np.sqrt(2.0), 1e-300)


def _support(x, y, mask, widen=5):
    """Where there is signal: whole stretches whose smoothed intensity
    rises clearly out of the noise (a top above 6σ), each widened by
    ``widen`` points, inside ``mask``. Whole stretches, not single points
    above a threshold: picking points one by one keeps only the noise that
    happens to push upwards at a faint edge, which inflates the spread
    (measured: 73 ppm rms against the truth's 64 at S/N 50)."""
    sigma = _noise(y)
    k = np.ones(5) / 5
    ys = np.convolve(y, k, mode="same")
    s_noise = sigma / np.sqrt(5)
    hot = (ys > 2 * s_noise) & mask
    keep = np.zeros(y.size, dtype=bool)
    edges = np.flatnonzero(np.diff(np.concatenate([[0], hot.astype(int), [0]])))
    for a, b in zip(edges[0::2], edges[1::2]):
        if float(np.max(ys[a:b])) > 6 * s_noise:
            keep[max(a - widen, 0):min(b + widen, y.size)] = True
    keep &= mask
    return keep if keep.sum() >= 8 else mask.copy()


def _moments(x, y, where):
    w = y[where]
    s = float(np.sum(w))
    if s <= 0:
        raise ValueError("No positive signal in the window to estimate from.")
    m1 = float(np.sum(x[where] * w) / s)
    m2 = float(np.sum((x[where] - m1) ** 2 * w) / s)
    return m1, max(m2, 0.0)


def _nucleus_of(results, nucleus):
    nucleus = nucleus or results.get("nuclide")
    if not nucleus:
        raise ValueError("Give the nucleus (e.g. '13C').")
    return nucleus


def _site_dict(label, delta, fwhm, interactions, lineshape="gaussian", amount=1.0):
    return {"label": label, "delta_iso": float(delta), "amount": float(amount),
            "fwhm": float(fwhm), "lineshape": lineshape, "eta": 0.5,
            "interactions": interactions}


# ============================================================
# Isotropic peaks and J multiplets: peak picking
# ============================================================

def _peaks(x, y, mask, max_peaks, min_prominence=0.05):
    dx = float(np.median(np.diff(x)))
    k = np.ones(3) / 3
    ys = np.convolve(np.where(mask, y, 0.0), k, mode="same")
    top = float(np.max(ys[mask])) if mask.any() else 0.0
    if top <= 0:
        raise ValueError("No positive signal in the window to estimate from.")
    sigma = _noise(y)
    idx, props = find_peaks(ys, prominence=max(min_prominence * top, 5 * sigma), height=0)
    idx = idx[mask[idx]]
    if not idx.size:
        raise ValueError("No peaks stand out of the noise in the window.")
    widths = peak_widths(ys, idx, rel_height=0.5)[0] * dx
    order = np.argsort(-ys[idx])[:max_peaks]
    idx, widths = idx[order], widths[order]
    # Each top between grid points: a parabola through the three highest.
    centres = x[idx].astype(float)
    inner = (idx > 0) & (idx < ys.size - 1)
    a, b, c = ys[idx[inner] - 1], ys[idx[inner]], ys[idx[inner] + 1]
    denom = a - 2 * b + c
    offset = np.where(denom != 0, 0.5 * (a - c) / np.where(denom != 0, denom, 1), 0.0)
    centres[inner] += np.clip(offset, -0.5, 0.5) * dx
    keep = np.argsort(centres)
    return centres[keep], ys[idx][keep], np.maximum(widths[keep], 2 * dx)


def _estimate_isotropic(x, y, mask, max_peaks, lineshape):
    centres, heights, widths = _peaks(x, y, mask, max_peaks)
    sites = [_site_dict(f"Site {k + 1}", c, w, [], lineshape=lineshape, amount=h * w)
             for k, (c, h, w) in enumerate(zip(centres, heights, widths))]
    notes = [f"Found {len(sites)} peak{'s' if len(sites) != 1 else ''} standing out of the "
             "noise (at least 5% of the tallest): "
             + ", ".join(f"{c:.4g} ppm (FWHM {w:.3g} ppm)" for c, w in zip(centres, widths))
             + ". Each peak's shift is its top, its width the full width at half height."]
    if len(centres) == max_peaks:
        notes.append(f"Only the {max_peaks} tallest were kept; raise the number of peaks to "
                     "keep more.")
    return sites, notes


def _estimate_jcoupling(x, y, mask, larmor_Hz, partner, lineshape):
    centres, heights, widths = _peaks(x, y, mask, 40)
    if centres.size < 2:
        raise ValueError("Only one line in the window: no splitting to measure a J from. "
                         "Narrow the window to one multiplet with its lines resolved.")
    spacings_Hz = np.diff(centres) * larmor_Hz / 1e6
    J = float(np.median(spacings_Hz))
    n = int(centres.size - 1)
    centre = float(np.sum(centres * heights) / np.sum(heights))
    from math import comb
    expected = np.array([comb(n, k) for k in range(n + 1)], float)
    expected /= expected.max()
    seen = heights / heights.max()
    binomial = bool(np.all(np.abs(seen - expected) < 0.25))
    uneven = float(np.std(spacings_Hz) / J) if J else 0.0
    site = _site_dict("Site 1", centre, float(np.median(widths)),
                      [{"type": "jcoupling", "nuclide": partner, "n": n, "J_Hz": J}],
                      lineshape=lineshape, amount=float(np.sum(heights) * np.median(widths)))
    notes = [f"{centres.size} lines, {J:.3g} Hz apart (median spacing): J ≈ {J:.3g} Hz to "
             f"{n} equivalent {partner} partner{'s' if n != 1 else ''} (n + 1 lines for n "
             "spin-½ partners), centred at {:.4g} ppm.".format(centre)]
    if not binomial:
        notes.append("The line heights are not in the binomial ratio n partners give "
                     f"({', '.join(f'{v:.2f}' for v in expected)}): it may be several "
                     "couplings, or overlapping multiplets -- check the J groups.")
    if uneven > 0.1:
        notes.append(f"The spacings vary by {uneven:.0%}: more than one J, or strong "
                     "coupling (then use the strongly coupled peaks model).")
    return [site], notes


# ============================================================
# CSA and quadrupolar: moments matched by simulation
# ============================================================

def _sim(kind, eta, size, x, settings, fwhm, nucleus, transitions, n_zcw, delta=0.0):
    site = Site(nucleus, float(delta), label="S", fwhm=fwhm)
    if kind == "csa":
        site.add_csa(size, eta)
    else:
        site.add_quadrupolar(size, eta)
    return np.asarray(simulate_combined([site], x, settings["larmor_Hz"],
                                        mas_Hz=settings["mas_Hz"], order=settings["order"],
                                        transitions=transitions, n_zcw=n_zcw)["real_spec"],
                      float)


def _size_for_m2(kind, eta, sign, target_m2, delta, x, where, settings, fwhm, nucleus,
                 transitions, n_zcw):
    """The interaction size (|Δ| ppm, or C_Q MHz) whose simulation, made at
    δiso = ``delta`` and measured over the same stretch ``where`` as the
    data, has the data's second moment. Secant steps on log size against
    log m2, kept inside a bracket and falling back to bisection, so a
    spread that saturates (a pattern running off the axis) or a simulation
    that cannot be made (sidebands too many for the spinning rate) cannot
    send it off to absurd sizes."""
    def m2_of(p):
        # Simulated at δiso = delta itself: a pattern simulated at 0 and
        # shifted loses whatever ran off the axis before the shift (the
        # outer sidebands -- measured: m1 off by 10 ppm).
        try:
            curve = _sim(kind, eta, sign * p, x, settings, fwhm, nucleus, transitions, n_zcw,
                         delta)
        except ValueError:
            return np.inf                      # too large to simulate: treat as too wide
        if float(np.sum(curve[where])) <= 0:
            return np.inf
        return _moments(x, curve, where)[1]
    span = float(x[where].max() - x[where].min()) if where.any() else float(np.ptp(x))
    if kind == "csa":
        p = np.sqrt(max(5.0 * target_m2 / (1 + eta ** 2 / 3), 1e-12))
        lo_lim, hi_lim, exponent = 1e-4, 3.0 * span, 2.0
    else:
        p, exponent = 2.0, 4.0 if transitions == "ct" else 2.0
        lo_lim, hi_lim = 1e-3, 200.0
    p = float(np.clip(p, lo_lim, hi_lim))
    lo = hi = None                             # sizes known to give too little / too much
    prev = None
    for _ in range(14):
        m2 = m2_of(p)
        if np.isfinite(m2) and abs(m2 - target_m2) <= 0.005 * target_m2:
            return p
        if m2 < target_m2:
            lo = p if lo is None else max(lo, p)
        else:
            hi = p if hi is None else min(hi, p)
        if prev is not None and np.isfinite(m2) and np.isfinite(prev[1]) and m2 > 0 \
                and prev[1] > 0 and abs(np.log(p / prev[0])) > 1e-9:
            exponent = float(np.clip(np.log(m2 / prev[1]) / np.log(p / prev[0]), 0.5, 6.0))
        prev = (p, m2)
        if np.isfinite(m2) and m2 > 0:
            step = p * (target_m2 / m2) ** (1.0 / exponent)
        else:
            step = p / 2.0
        if lo is not None and hi is not None and not lo < step < hi:
            step = float(np.sqrt(lo * hi))     # bisect (in log) inside the bracket
        p = float(np.clip(step, lo_lim, hi_lim))
        if (lo is not None and hi is not None and hi / lo < 1.002) or \
                (p in (lo_lim, hi_lim) and prev[0] == p):
            break
    return p


def _estimate_anisotropic(kind, x, y, mask, nucleus, settings, transitions, n_zcw,
                          ct_only=False):
    where = _support(x, y, mask)
    if ct_only:
        # The central transition only: the stretch holding the tallest
        # point (its centreband). Satellites usually run far past the
        # axis, where their spread says little about C_Q.
        top = int(np.argmax(np.where(where, y, -np.inf)))
        edges = np.flatnonzero(np.diff(np.concatenate([[0], where.astype(int), [0]])))
        for a, b in zip(edges[0::2], edges[1::2]):
            if a <= top < b:
                where = np.zeros_like(where)
                where[a:b] = True
                break
    m1, m2 = _moments(x, y, where)
    dx = float(np.median(np.diff(x)))
    mas = settings["mas_Hz"]
    # Starting linewidth: the narrowest line under MAS (a sideband), else a
    # small fraction of the pattern's spread.
    try:
        _c, _h, widths = _peaks(x, y, where, 5)
        fwhm = float(np.min(widths)) if mas else max(2 * dx, 0.02 * np.sqrt(m2))
    except ValueError:
        fwhm = max(2 * dx, 0.02 * np.sqrt(m2))
    signs = (1.0, -1.0) if kind == "csa" else (1.0,)
    target = y[where]
    # Under MAS a CSA's centreband sits exactly at δiso: the data peak
    # nearest the centre of mass (which a pattern cut off at the window's
    # edge pulls away from it).
    snap = None
    if mas and kind == "csa":
        try:
            tops, _h, _w = _peaks(x, y, where, 64)
            snap = tops
        except ValueError:
            snap = None

    def trial(eta, sign):
        delta = m1
        if snap is not None and snap.size:
            delta = float(snap[np.argmin(np.abs(snap - m1))])
        size = 0.0
        # A CSA's δiso is its centre of mass (or its centreband): nothing to
        # iterate. A quadrupolar CT is shifted by an amount that depends on
        # C_Q, so δiso and C_Q are refined together.
        for _ in range(1 if kind == "csa" else 3):
            size = _size_for_m2(kind, eta, sign, m2, delta, x, where, settings, fwhm, nucleus,
                                transitions, n_zcw)
            curve = _sim(kind, eta, sign * size, x, settings, fwhm, nucleus, transitions,
                         n_zcw, delta)
            if kind != "csa" and float(np.sum(curve[where])) > 0:
                delta += m1 - _moments(x, curve, where)[0]
        return _score(kind, eta, sign * size, delta, x, where, target, settings, fwhm,
                      nucleus, transitions, n_zcw)

    trials = [trial(eta, sign) for eta in ETA_GRID for sign in signs]
    best = min(trials, key=lambda t: t["chi2"])
    # A finer look around the best asymmetry.
    for eta in (best["eta"] - 0.1, best["eta"] - 0.05, best["eta"] + 0.05, best["eta"] + 0.1):
        if 0.0 <= eta <= 1.0:
            trials.append(trial(eta, np.sign(best["size"]) or 1.0))
    best = min(trials, key=lambda t: t["chi2"])
    ranked = sorted(trials, key=lambda t: t["chi2"])
    rival = next((t for t in ranked[1:] if np.sign(t["size"]) != np.sign(best["size"])), None)
    if kind == "csa":
        inter = {"type": "csa", "aniso": round(best["size"], 3), "asym": round(best["eta"], 3),
                 "convention": "Haeberlen", "euler": (0.0, 0.0, 0.0)}
        size_text = f"Δ ≈ {best['size']:.4g} ppm"
    else:
        inter = {"type": "quadrupolar", "C_Q": round(abs(best["size"]), 4),
                 "eta_Q": round(best["eta"], 3)}
        size_text = f"C_Q ≈ {abs(best['size']):.3g} MHz"
    site = _site_dict("Site 1", best["delta_iso"], fwhm, [inter])
    notes = [
        f"Centre of mass of the pattern: {m1:.4g} ppm"
        + (f"; with the quadrupolar shift taken off, δiso ≈ {best['delta_iso']:.4g} ppm."
           if kind == "quadrupolar" else
           f"; the centreband nearest it puts δiso at {best['delta_iso']:.4g} ppm."
           if snap is not None else f", which is δiso for a CSA: δiso ≈ "
           f"{best['delta_iso']:.4g} ppm."),
        f"Spread of the pattern (second moment, which spinning does not change): "
        f"{np.sqrt(m2):.3g} ppm rms, which needs {size_text}.",
        f"Of the asymmetries tried ({', '.join(f'{e:g}' for e in ETA_GRID)}, then ±0.05 and "
        f"±0.1 around the best){' and both signs of Δ' if kind == 'csa' else ''}, "
        f"{'η' if kind == 'csa' else 'η_Q'} ≈ {best['eta']:.2f} matches the shape best.",
        f"Starting linewidth: {fwhm:.3g} ppm "
        + ("(the narrowest line)." if mas else "(the fit will refine it)."),
    ]
    if rival is not None and rival["chi2"] < 1.1 * best["chi2"]:
        notes.append("The opposite sign of Δ fits almost as well (Δ ≈ "
                     f"{rival['size']:.4g} ppm, η ≈ {rival['eta']:.2f}): try both as starts "
                     "(Try starts) if the fit looks wrong.")
    if kind == "csa" and abs(best["size"]) * settings["larmor_Hz"] / 1e6 < 0.5 * mas and mas:
        notes.append("The anisotropy is small next to the spinning rate: few sidebands, so "
                     "these values are rough.")
    return [site], notes, best


def _score(kind, eta, size, delta, x, where, target, settings, fwhm, nucleus, transitions,
           n_zcw):
    curve = _sim(kind, eta, size, x, settings, fwhm, nucleus, transitions, n_zcw, delta)
    s = curve[where]
    a = float(np.dot(s, target) / (np.dot(s, s) or 1.0))
    chi2 = float(np.sum((target - a * s) ** 2))
    return {"eta": float(eta), "size": float(size), "delta_iso": float(delta), "chi2": chi2,
            "curve": a * curve}


# ============================================================
# Entry point
# ============================================================

def estimate_start(results, kind, nucleus=None, larmor_Hz=None, mas_Hz=0.0, order="second",
                   transitions="all", fit_regions=None, max_peaks=8, partner="1H",
                   lineshape="gaussian", n_zcw=None):
    """
    Starting values for a fit of ``results`` (``x_ppm``, ``real_spectrum``,
    ``SFO1``) with one interaction type, ``kind``: ``"isotropic"``,
    ``"jcoupling"``, ``"csa"`` or ``"quadrupolar"`` (one site; give a
    ``fit_regions`` window around it when the spectrum holds several).

    Returns a dict:
        ``sites`` -- site dicts, ready for :func:`fit_combined` (or
        Fit ▸ Combined Hamiltonian);
        ``notes`` -- how each value was arrived at, in plain words;
        ``start_r2`` -- R² of the starting model (amounts by least
        squares), a check that the start is near the data;
        ``kind``, ``nucleus``.
    """
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}.")
    nucleus = _nucleus_of(results, nucleus)
    larmor_Hz = float(larmor_Hz) if larmor_Hz else float(results["SFO1"]) * 1e6
    x, y, mask = _prepare(results, fit_regions)
    settings = {"larmor_Hz": larmor_Hz, "mas_Hz": float(mas_Hz or 0.0),
                "order": "second" if order == "exact" else order}
    if kind == "isotropic":
        sites, notes = _estimate_isotropic(x, y, mask, int(max_peaks), lineshape)
    elif kind == "jcoupling":
        sites, notes = _estimate_jcoupling(x, y, mask, larmor_Hz, partner, lineshape)
    else:
        if kind == "quadrupolar":
            from .utils import find_isotope_info
            if float(find_isotope_info(nucleus)["spin"]) < 1:
                raise ValueError(f"{nucleus} is spin-1/2: it has no quadrupolar coupling.")
        n = int(n_zcw) if n_zcw else (233 if settings["mas_Hz"] else 987)
        ct_only = False
        est_transitions = transitions
        if kind == "quadrupolar" and transitions != "ct":
            from .utils import find_isotope_info
            spin = float(find_isotope_info(nucleus)["spin"])
            if not spin.is_integer():
                ct_only, est_transitions = True, "ct"
        sites, notes, _best = _estimate_anisotropic(kind, x, y, mask, nucleus, settings,
                                                    est_transitions, n, ct_only=ct_only)
        if ct_only:
            notes.append("Estimated from the central transition alone (its centreband), "
                         "which C_Q shapes most clearly; the fit then uses every transition "
                         "you chose.")
        if order == "exact":
            notes.append("Estimated with second-order theory (fast); the fit uses exact.")
    from .combined_fitting import starting_model
    try:
        guess = starting_model(results, [Site.from_dict({**s, "nucleus": nucleus}) for s in sites],
                               larmor_Hz=larmor_Hz, mas_Hz=mas_Hz, order=order,
                               transitions=transitions, fit_regions=fit_regions,
                               n_zcw=(233 if mas_Hz else 4181))
        m = guess["mask"]
        res = guess["y"][m] - guess["start"][m]
        tot = float(np.sum((guess["y"][m] - guess["y"][m].mean()) ** 2))
        start_r2 = 1 - float(np.sum(res ** 2)) / tot if tot else float("nan")
        for s in sites:
            s["amount"] = float(guess["amounts"].get(s["label"], s["amount"]))
    except (ValueError, KeyError):
        start_r2 = float("nan")
    if np.isfinite(start_r2):
        notes.append(f"The starting model already matches the data with R² {start_r2:.3f}; "
                     "the fit takes it from there.")
    return {"sites": sites, "notes": notes, "start_r2": start_r2, "kind": kind,
            "nucleus": nucleus}
