"""
Plain-language advice on a finished fit: what the fit tells you, what it
cannot tell you, and what to try next (explicit user request,
2026-10-06: "comments about what to try next ... and what the fits (to
individual spectra and to any pseudo-2D data) can and cannot say ...
in straightforward, easy-to-understand, plain language").

Every check is measured, not guessed:

* **Noise** is estimated from the spectrum itself (:func:`estimate_noise`),
  from point-to-point scatter where the fitted model is flat, so "the fit
  is good" means "what is left over is no bigger than the noise" -- not
  an R² threshold, which says little about a noisy spectrum.
* **Misfits** are located by smoothing the residual over about one
  linewidth and keeping only stretches well above the smoothed noise:
  signal no peak accounts for ("try adding a peak"), a shape that is not
  quite right on a peak (which interaction to add), and spinning
  sidebands that are too weak or strong (anisotropy under- or
  over-estimated) or shifted progressively (MAS rate off).
* **Parameters** are read with their limits and uncertainties: at a limit,
  not determined, only roughly determined, or trading off with another.
* **Models** are compared by the Bayesian information criterion, which
  charges each extra parameter; the verdicts ("clearly better", "no real
  difference: prefer the simpler") use the usual ΔBIC bands.

Each item is a dict:

    {"section": "tells" | "next",     # what the fit says | what to try next
     "tone": "good" | "note" | "warn",
     "text": "...",                   # plain language, one or two sentences
     "action": None | {"type": ..., ...}}  # optional one-click follow-up

Actions a GUI can offer: ``add_peak`` (delta_iso, fwhm), ``add_interaction``
(site, kind), ``set_lineshape`` (site, lineshape), ``free_mas``,
``free_parameter`` (site, name), ``fix_parameter`` (site, name),
``widen_bound`` (site, name, side, value), ``remove_site`` (site),
``set_relaxation_model`` (model).

    from opennmr.notebook import fit_advice
    fit = fit_combined(results, sites, mas_Hz=12e3)
    print(fit_advice.advice_text(fit_advice.advise_combined_fit(fit)))
"""

from __future__ import annotations

import numpy as np

#: Friendly names for parameters, as people say them.
_NAMES = {"delta_iso": "the shift (δiso)", "amount": "the amount", "fwhm": "the linewidth",
          "eta": "the Lorentzian fraction", "aniso": "the anisotropy (Δ or span)",
          "asym": "the asymmetry (η or skew)", "alpha": "the Euler angle α",
          "beta": "the Euler angle β", "gamma": "the Euler angle γ",
          "C_Q": "the quadrupolar coupling C_Q", "eta_Q": "the quadrupolar asymmetry η_Q",
          "mas_Hz": "the MAS rate", "b0": "the baseline offset", "b1": "the baseline slope"}
_UNITS = {"delta_iso": " ppm", "fwhm": " ppm", "aniso": " ppm", "C_Q": " MHz", "alpha": "°",
          "beta": "°", "gamma": "°", "mas_Hz": " Hz"}
#: Parameters whose limits are physical (0..1 for an asymmetry): ending
#: on one is a statement about the tensor, not a limit to widen.
_PHYSICAL_LIMITS = ("asym", "eta_Q", "eta")


def _name(label, name):
    base = _NAMES.get(name, f"J ({name})" if name.startswith("J") else name)
    return f"{base} of {label}" if label else base


def _fmt(value, name=""):
    unit = _UNITS.get(name, " Hz" if name.startswith("J") else "")
    return f"{value:.4g}{unit}"


def _item(section, tone, text, action=None):
    text = text[:1].upper() + text[1:] if text else text
    return {"section": section, "tone": tone, "text": text, "action": action}


# ============================================================
# Noise and residual structure
# ============================================================

def estimate_noise(x, y, y_fit=None, baseline=None):
    """The noise level (1σ) of a spectrum, from point-to-point differences
    where the fitted model is flat (below 1% of its height), robustly
    (median absolute deviation): a smooth signal hardly changes from one
    point to the next, noise does. Falls back to every point when there
    are too few flat ones."""
    y = np.asarray(y, dtype=float)
    if y.size < 8:
        return float(np.std(y)) or 1.0
    quiet = np.ones(y.size, dtype=bool)
    if y_fit is not None:
        model = np.asarray(y_fit, dtype=float) - (0.0 if baseline is None
                                                  else np.asarray(baseline, dtype=float))
        top = float(np.max(np.abs(model))) or 1.0
        quiet = np.abs(model) < 0.01 * top
    d = np.diff(y)
    ok = quiet[1:] & quiet[:-1]
    if ok.sum() < 32:
        ok = np.ones(d.size, dtype=bool)
    d = d[ok]
    sigma = 1.4826 * float(np.median(np.abs(d - np.median(d)))) / np.sqrt(2.0)
    if not sigma > 0:
        sigma = float(np.std(d)) / np.sqrt(2.0)
    return sigma if sigma > 0 else 1e-12


def _smooth(r, w):
    w = max(1, int(w))
    if w <= 1:
        return r.copy()
    kernel = np.ones(w) / w
    return np.convolve(r, kernel, mode="same")


def residual_lumps(x, y, y_fit, mask=None, sigma=None, width_points=5, threshold=4.0):
    """Stretches of the spectrum the model misses: the residual smoothed
    over ``width_points`` (about a linewidth), kept where it stays above
    ``threshold`` × the smoothed noise. Returns a list, biggest first, of
    {"centre", "low", "high" (ppm), "sign" (+1: data above the fit),
    "height" (largest smoothed residual), "area", "significance" (height /
    smoothed noise)}."""
    x = np.asarray(x, dtype=float)
    order = np.argsort(x)
    x, y, f = x[order], np.asarray(y, float)[order], np.asarray(y_fit, float)[order]
    m = np.ones(x.size, bool) if mask is None else np.asarray(mask, bool)[order]
    sigma = estimate_noise(x, y, f) if sigma is None else sigma
    r = np.where(m, y - f, 0.0)
    rs = _smooth(r, width_points)
    s_noise = sigma / np.sqrt(max(1, int(width_points)))
    hot = (np.abs(rs) > threshold * s_noise) & m
    lumps = []
    if not hot.any():
        return lumps
    dx = float(np.median(np.abs(np.diff(x)))) if x.size > 1 else 1.0
    edges = np.flatnonzero(np.diff(np.concatenate([[0], hot.astype(int), [0]])))
    for a, b in zip(edges[0::2], edges[1::2]):
        seg = slice(a, b)
        part = rs[seg]
        k = int(np.argmax(np.abs(part)))
        weights = np.abs(r[seg]) + 1e-300
        lumps.append({"centre": float(np.sum(x[seg] * weights) / np.sum(weights)),
                      "low": float(x[a]), "high": float(x[b - 1]),
                      "sign": 1 if part[k] > 0 else -1, "height": float(part[k]),
                      "area": float(np.sum(r[seg]) * dx),
                      "significance": float(abs(part[k]) / s_noise)})
    lumps.sort(key=lambda d: -abs(d["area"]))
    return lumps


def fit_verdict(ratio):
    """(tone, text) for the leftover-to-noise ratio of a fit."""
    if ratio <= 1.3:
        return "good", ("The model describes your spectrum to within the noise: what the fit "
                        "leaves behind is just noise.")
    if ratio <= 2.5:
        return "note", (f"A good fit, but not a perfect one: what is left over is about "
                        f"{ratio:.1f}× the noise, so the model misses some detail (see below).")
    return "warn", (f"The model misses real features of your spectrum: what is left over is "
                    f"about {ratio:.0f}× the noise. The numbers below are the best this model "
                    "can do, not necessarily the right ones.")


# ============================================================
# Parameters
# ============================================================

def _parameter_items(parameters, pcov=None, free_names=None, natural_bounds=None):
    """Limits, undetermined and rough parameters, and trade-offs."""
    out = []
    reliable, rough = [], []
    natural_bounds = natural_bounds or {}
    for p in parameters:
        if not p.get("free") or p["name"] in ("b0", "b1"):
            continue
        label, name, value = p.get("label") or "", p["name"], float(p["value"])
        err = p.get("error")
        err = float(err) if err is not None and np.isfinite(err) else None
        lo, hi = float(p.get("lower", -np.inf)), float(p.get("upper", np.inf))
        e = (err or 0.0) + 1e-9 * max(1.0, abs(value))
        at_hi = np.isfinite(hi) and value + e >= hi
        at_lo = np.isfinite(lo) and value - e <= lo
        who = _name(label, name)
        if at_hi and at_lo:
            out.append(_item("tells", "warn",
                             f"Your data cannot pin down {who}: its uncertainty covers its whole "
                             f"allowed range ({lo:g} to {hi:g}). Whatever value it shows is not "
                             "a measurement."))
            out.append(_item("next", "note",
                             f"Fix {who} at a sensible value (untick Free) -- a literature "
                             "value, or one from another measurement -- so the other numbers "
                             "stop depending on it.", {"type": "fix_parameter", "site": label,
                                                       "name": name}))
            continue
        if name == "amount" and at_lo:
            continue                  # reported as "peak contributes nothing" elsewhere
        if name == "C_Q" and at_lo:
            out.append(_item("tells", "note",
                             f"{label} shows no measurable quadrupolar effect (C_Q went to "
                             "about 0): at this field and linewidth, its coupling is too small "
                             "to see, or the peak does not need it."))
            continue
        nat = natural_bounds.get((label, name))
        side = "upper" if at_hi else "lower" if at_lo else None
        if side is not None:
            limit = hi if side == "upper" else lo
            physical = name in _PHYSICAL_LIMITS and nat is not None and \
                np.isclose(limit, nat[1] if side == "upper" else nat[0])
            if name == "mas_Hz":
                out.append(_item("tells", "warn",
                                 f"The MAS rate stopped at the edge of the range it was allowed "
                                 f"({limit:g} Hz): the true rate may lie further out."))
                out.append(_item("next", "note",
                                 "Check the spinning rate recorded for this spectrum, or widen "
                                 "the MAS rate's Min/Max and fit again.",
                                 {"type": "widen_bound", "site": "", "name": "mas_Hz",
                                  "side": side, "value": _widened(value, lo, hi, side)}))
            elif physical:
                out.append(_item("tells", "note",
                                 f"{who[0].upper() + who[1:]} ended at {limit:g}, the edge of "
                                 f"what is physically possible. That can be real (an asymmetry "
                                 "of 0 is an axially symmetric tensor), but if you did not "
                                 "expect it, the model may be compensating for something "
                                 "else."))
            else:
                out.append(_item("tells", "warn",
                                 f"{who[0].upper() + who[1:]} stopped at its "
                                 f"{'upper' if side == 'upper' else 'lower'} limit "
                                 f"({_fmt(limit, name)}): the best value may lie beyond it, so "
                                 "this number is a bound, not a measurement."))
                out.append(_item("next", "note",
                                 f"Widen the limit on {who} (its {'Max' if side == 'upper' else 'Min'}"
                                 ") and fit again. If it runs away again, the model is probably "
                                 "missing something rather than the limit being too tight.",
                                 {"type": "widen_bound", "site": label, "name": name,
                                  "side": side, "value": _widened(value, lo, hi, side)}))
            continue
        if name == "mas_Hz":
            out.append(_item("tells", "good" if err is not None and err < 20 else "note",
                             f"The MAS rate came out at {value:.1f}"
                             + (f" ± {err:.1f}" if err is not None else "") + " Hz."))
            continue
        # A shift's relative error means nothing (0.01 ± 0.01 ppm is a fine
        # measurement): shifts are not graded here.
        if err is None or name in ("amount", "eta", "delta_iso"):
            continue
        rel = err / abs(value) if value else np.inf
        scale = {"alpha": 180.0, "beta": 180.0, "gamma": 180.0}.get(name)
        if scale:
            rel = err / scale
        if rel > 0.3:
            rough.append(f"{who} ({_fmt(value, name)} ± {_fmt(err, name)})")
        elif rel < 0.05:
            reliable.append(who)
    if reliable:
        out.append(_item("tells", "good", "Well determined (better than ±5%): "
                         + _join(reliable) + "."))
    if rough:
        out.append(_item("tells", "warn", "Only roughly determined (more than ±30%): "
                         + _join(rough) + ". Treat these as estimates."))
    if pcov is not None and free_names:
        pcov = np.asarray(pcov, dtype=float)
        if pcov.ndim == 2 and pcov.shape[0] == len(free_names) and pcov.size:
            d = np.sqrt(np.clip(np.diag(pcov), 1e-300, None))
            corr = pcov / np.outer(d, d)
            seen = 0
            for a in range(len(free_names)):
                for b in range(a + 1, len(free_names)):
                    na, nb = free_names[a], free_names[b]
                    if any(k in na + nb for k in ("amount", "b0", "b1")):
                        continue
                    if abs(corr[a, b]) > 0.95 and seen < 3:
                        seen += 1
                        out.append(_item(
                            "tells", "note",
                            f"{_pretty(na)} and {_pretty(nb)} trade off against each other "
                            f"({abs(corr[a, b]):.0%} correlated): the data cannot tell them "
                            "apart well. Fixing one makes the other reliable; another field or "
                            "spinning rate would separate them."))
    return out


def _widened(value, lo, hi, side):
    span = max(abs(hi - lo) if np.isfinite(hi - lo) else abs(value), abs(value) * 0.5, 1e-6)
    return float(hi + 0.5 * span) if side == "upper" else float(lo - 0.5 * span)


def _pretty(free_name):
    if ": " in free_name:
        label, name = free_name.split(": ", 1)
        return _name(label, name)
    return _name("", free_name)


def _join(parts):
    parts = list(parts)
    if len(parts) <= 1:
        return "".join(parts)
    return ", ".join(parts[:-1]) + " and " + parts[-1]


# ============================================================
# Combined-Hamiltonian fits (one spectrum)
# ============================================================

def _site_info(fit):
    """[{label, kinds (set), spin, delta_iso, fwhm, lineshape, centre (δcg
    or δiso), aniso, C_Q}] from a fit's sites (Site objects or dicts)."""
    out = []
    for s in fit.get("sites") or [{"label": k} for k in fit.get("site_spectra", {})]:
        d = s.to_dict() if hasattr(s, "to_dict") else dict(s)
        inter = d.get("interactions") or []
        kinds = {i.get("type") for i in inter}
        csa = next((i for i in inter if i.get("type") == "csa"), {})
        quad = next((i for i in inter if i.get("type") == "quadrupolar"), {})
        derived = (fit.get("derived") or {}).get(d.get("label"), {})
        out.append({"label": d.get("label"), "kinds": kinds, "spin": d.get("spin"),
                    "delta_iso": float(d.get("delta_iso", 0.0)), "fwhm": float(d.get("fwhm", 0.0)),
                    "lineshape": d.get("lineshape"), "aniso": csa.get("aniso"),
                    "convention": csa.get("convention"), "C_Q": quad.get("C_Q"),
                    "eta_Q": quad.get("eta_Q"),
                    "centre": float(derived.get("delta_cg", d.get("delta_iso", 0.0))),
                    "share": derived.get("share_%"), "share_err": derived.get("share_%_err")})
    return out


def _sideband_items(fit, sites, x, y, base, mask, sigma, larmor_MHz):
    """Each anisotropic site's sidebands: too weak or strong in the fit
    (anisotropy under-/overestimated), or shifted progressively (MAS rate
    off)."""
    out, flagged = [], set()
    settings = fit.get("settings") or {}
    mas = float(settings.get("mas_Hz") or 0.0)
    curves = fit.get("site_spectra") or {}
    if not mas or not larmor_MHz or not curves:
        return out, flagged
    spacing = mas / larmor_MHz                                   # ppm
    dx = float(np.median(np.abs(np.diff(x))))
    total_sites = np.sum([np.asarray(c, float) for c in curves.values()], axis=0)
    shift_reported = False
    for s in sites:
        if not s["kinds"] & {"csa", "quadrupolar"} or s["label"] not in curves:
            continue
        mine = np.asarray(curves[s["label"]], float)
        data_mine = y - base - (total_sites - mine)               # the data, others removed
        k_max = int(np.floor((x[mask].max() - x[mask].min()) / spacing)) + 1
        # Each window is as wide as this site's centreband (plus a margin
        # for a slightly wrong MAS rate), centred where its sideband should
        # be -- not a whole rotor period wide, which adds up baseline offsets
        # and noise from hundreds of empty points, and not only where the
        # fit has intensity, which misses sidebands the fit has squashed.
        near = np.abs(x - s["centre"]) <= spacing / 2
        hot = near & (mine > 0.01 * (float(np.max(mine[near])) if near.any() else 1.0))
        half = (float(x[hot].max() - x[hot].min()) / 2 if hot.sum() > 1 else 0.0) + \
            2 * max(s["fwhm"], dx)
        rows = []
        for k in range(-k_max, k_max + 1):
            centre = s["centre"] + k * spacing
            reach = min(half + 0.01 * abs(k) * spacing, spacing / 2)
            win = mask & (np.abs(x - centre) <= reach)
            if win.sum() < 3:
                continue
            d_k, f_k = float(np.sum(data_mine[win]) * dx), float(np.sum(mine[win]) * dx)
            noise_k = sigma * dx * np.sqrt(win.sum())
            cd = float(np.sum(x[win] * np.clip(data_mine[win], 0, None))
                       / max(np.sum(np.clip(data_mine[win], 0, None)), 1e-300))
            cf = float(np.sum(x[win] * np.clip(mine[win], 0, None))
                       / max(np.sum(np.clip(mine[win], 0, None)), 1e-300))
            rows.append((k, d_k, f_k, noise_k, cd - cf))
        centre_rows = [r for r in rows if r[0] == 0]
        side_rows = [r for r in rows if r[0] != 0]
        if not centre_rows or not side_rows:
            continue
        _k, d0, f0, n0, _ = centre_rows[0]
        sb_d = sum(r[1] for r in side_rows)
        sb_f = sum(r[2] for r in side_rows)
        sb_noise = float(np.sqrt(sum(r[3] ** 2 for r in side_rows)))
        if f0 <= 0 or d0 <= 0:
            continue
        what = ("the anisotropy" if s["kinds"] == {"csa"} else "C_Q" if s["kinds"] == {"quadrupolar"}
                else "the anisotropy (CSA) or C_Q")
        centre_ok = abs(d0 - f0) < max(0.05 * d0, 4 * n0)
        diff = sb_d - sb_f
        if abs(diff) > max(4 * sb_noise, 0.1 * max(abs(sb_d), abs(sb_f), 1e-300)) and \
                max(sb_d, sb_f) > 0.03 * d0:
            weaker = diff > 0
            flagged.add(s["label"])
            pct_f, pct_d = 100 * sb_f / f0, 100 * sb_d / d0
            out.append(_item(
                "tells", "warn",
                ("The centreband of " if centre_ok else "For ") + f"{s['label']}"
                + (" fits well, but the" if centre_ok else ", the") + " spinning sidebands do not "
                f"match yet: in the fit they hold {pct_f:.0f}% of the centreband's intensity, in "
                f"your data {pct_d:.0f}%. So {what} is probably "
                + ("underestimated." if weaker else "overestimated.")))
            held = [p for p in fit.get("parameters") or [] if p.get("label") == s["label"]
                    and p["name"] in ("aniso", "C_Q") and not p.get("free")]
            if held:
                names = _join([_name(s["label"], p["name"]) for p in held])
                out.append(_item(
                    "next", "note",
                    f"{names[0].upper() + names[1:]} is held fixed at "
                    + _join([_fmt(p["value"], p["name"]) for p in held])
                    + ": let it vary (tick Free).",
                    {"type": "free_parameter", "site": s["label"], "name": held[0]["name"]}))
            else:
                out.append(_item(
                    "next", "note",
                    f"Try a {'larger' if weaker else 'smaller'} starting value for {what} of "
                    f"{s['label']} (or a few values in Try starts)."))
        # Progressive shift of the sidebands: the MAS rate is off.
        strong = [r for r in side_rows if min(r[1], r[2]) > 5 * r[3]]
        if len(strong) >= 2 and not settings.get("fit_mas") and not shift_reported:
            ks = np.array([r[0] for r in strong], float)
            sh = np.array([r[4] for r in strong], float)
            w = np.array([r[2] for r in strong], float)
            a = float(np.sum(w * ks * sh) / np.sum(w * ks * ks))
            pred = a * ks
            ss_res = float(np.sum(w * (sh - pred) ** 2))
            ss_tot = float(np.sum(w * sh ** 2)) or 1e-300
            explained = 1 - ss_res / ss_tot
            if explained > 0.7 and abs(a) * np.max(np.abs(ks)) > 0.25 * max(s["fwhm"], dx):
                delta_hz = a * larmor_MHz
                shift_reported = True
                out.insert(0, _item(
                    "tells", "warn",
                    f"The spinning sidebands in your data sit slightly "
                    f"{'further from' if a > 0 else 'closer to'} the centreband than in the fit, "
                    "more so for the outer ones: the MAS rate is probably "
                    f"{'higher' if a > 0 else 'lower'} than the {mas:g} Hz used (by roughly "
                    f"{abs(delta_hz):.0f} Hz or more). With the sidebands misplaced, the fit "
                    "tends to shrink the anisotropy to hide them, so fix this first."))
                out.insert(1, _item(
                    "next", "note",
                    "Let the MAS rate vary: tick Free on the MAS rate row of the parameter "
                    "table (it may move by up to 500 Hz).", {"type": "free_mas"}))
        # Too few sidebands to measure the anisotropy.
        if "csa" in s["kinds"] and s["aniso"] is not None:
            width_Hz = abs(float(s["aniso"])) * larmor_MHz
            if width_Hz < 0.5 * mas:
                out.append(_item(
                    "tells", "note",
                    f"{s['label']}'s anisotropy (about {width_Hz / 1e3:.1g} kHz) is small next to "
                    f"the spinning rate ({mas / 1e3:g} kHz), so it has only weak sidebands: its "
                    "Δ and especially η are hard to measure here. Spinning slower, or a static "
                    "spectrum, would pin them down."))
    return out, flagged


def _adjacent_site(lump, sites, curves, x):
    """The site whose own fitted pattern reaches (above 2% of its height)
    to within five of its linewidths of ``lump``, or None."""
    best, best_level = None, 0.0
    for s in sites:
        c = curves.get(s["label"])
        if c is None:
            continue
        reach = 5 * max(s["fwhm"], 0.0) + (lump["high"] - lump["low"])
        near = (x >= lump["low"] - reach) & (x <= lump["high"] + reach)
        if not near.any():
            continue
        level = float(np.max(np.abs(c[near]))) / (float(np.max(np.abs(c))) or 1.0)
        if level > 0.02 and level > best_level:
            best, best_level = s, level
    return best


def _sideband_owner(centre, sites, mas, larmor_MHz, dx):
    """The site with a spinning sideband (k ≠ 0) at ``centre`` ppm, allowing
    for a slightly wrong MAS rate (1% of a rotor period per order), or
    None."""
    if not mas or not larmor_MHz:
        return None
    spacing = mas / larmor_MHz
    best, best_miss = None, np.inf
    for s in sites:
        k = round((centre - s["centre"]) / spacing)
        if not k:
            continue
        miss = abs(centre - (s["centre"] + k * spacing))
        if miss < 2 * max(s["fwhm"], 3 * dx) + 0.02 * abs(k) * spacing and miss < best_miss:
            best, best_miss = s, miss
    return best


def _shape_suggestions(s, nucleus_spin, euler_free):
    """What to change on a peak whose shape is not quite right."""
    kinds = s["kinds"]
    out = []
    if not kinds:
        if s["lineshape"] != "pseudo_voigt":
            out.append(_item("next", "note",
                             f"Try a pseudo-Voigt line shape for {s['label']} (a mix of Gaussian "
                             "and Lorentzian): real lines are often between the two.",
                             {"type": "set_lineshape", "site": s["label"],
                              "lineshape": "pseudo_voigt"}))
        if nucleus_spin and nucleus_spin >= 1:
            out.append(_item("next", "note",
                             f"If {s['label']} looks lopsided or has shoulders, it may be "
                             "broadened by a quadrupolar coupling: try adding one.",
                             {"type": "add_interaction", "site": s["label"],
                              "kind": "quadrupolar"}))
        else:
            out.append(_item("next", "note",
                             f"If {s['label']} looks lopsided or has sidebands, try adding a "
                             "CSA to it.", {"type": "add_interaction", "site": s["label"],
                                            "kind": "csa"}))
    elif kinds == {"quadrupolar"}:
        out.append(_item("next", "note",
                         f"Try adding a CSA to {s['label']}: quadrupolar sites often carry some "
                         "chemical-shift anisotropy too, which makes the pattern lopsided and "
                         "changes the sidebands.",
                         {"type": "add_interaction", "site": s["label"], "kind": "csa"}))
    elif kinds == {"csa"} and nucleus_spin and nucleus_spin >= 1:
        out.append(_item("next", "note",
                         f"Try adding a quadrupolar coupling to {s['label']}: the nucleus has "
                         f"spin {nucleus_spin:g}, so it has a quadrupole moment, and a CSA alone "
                         "cannot give the second-order shape.",
                         {"type": "add_interaction", "site": s["label"],
                          "kind": "quadrupolar"}))
    elif {"csa", "quadrupolar"} <= kinds and not euler_free:
        out.append(_item("next", "note",
                         f"Try freeing the Euler angles of {s['label']}: they set how the CSA "
                         "and quadrupolar tensors are oriented to each other, and they start "
                         "fixed at 0 (aligned).",
                         {"type": "free_parameter", "site": s["label"], "name": "beta"}))
    out.append(_item("next", "note",
                     f"{s['label']} may also really be two overlapping peaks: try a second peak "
                     "close to it.",
                     {"type": "add_peak", "delta_iso": s["delta_iso"] + max(s["fwhm"], 0.1),
                      "fwhm": s["fwhm"] or 1.0}))
    return out


def advise_combined_fit(fit, alternatives=None, nucleus_spin=None):
    """Plain-language advice on a :func:`~.combined_fitting.fit_combined`
    result (or one experiment of a multi-spectrum fit).

    ``alternatives`` : ``[(name, fit), ...]`` -- other fits of the same
    spectrum (kept fit models) to compare with this one.
    ``nucleus_spin`` : the observed nucleus's spin, when the sites do not
    carry it.
    """
    items = []
    if fit.get("x") is None or fit.get("y") is None or fit.get("y_fit") is None:
        return items
    x = np.asarray(fit["x"], float)
    y = np.asarray(fit["y"], float)
    yf = np.asarray(fit["y_fit"], float)
    base = np.asarray(fit.get("baseline") if fit.get("baseline") is not None else 0.0, float) \
        * np.ones_like(y)
    mask = np.asarray(fit["mask"], bool) if fit.get("mask") is not None else np.ones(y.size, bool)
    sites = _site_info(fit)
    if nucleus_spin is None:
        nucleus_spin = next((s["spin"] for s in sites if s["spin"] is not None), None)
    settings = fit.get("settings") or {}
    larmor_MHz = float(settings.get("larmor_Hz") or fit.get("larmor_Hz") or 0.0) / 1e6
    sigma = estimate_noise(x, y, yf, base)
    resid = (y - yf)[mask]
    ratio = float(np.sqrt(np.mean(resid ** 2)) / sigma) if resid.size else np.nan
    tone, text = fit_verdict(ratio)
    items.append(_item("tells", tone, text))

    # ---- where the model misses ----
    dx = float(np.median(np.abs(np.diff(x)))) if x.size > 1 else 1.0
    widths = [s["fwhm"] for s in sites if s["fwhm"] > 0]
    w_pts = int(np.clip(round((min(widths) if widths else 5 * dx) / dx), 3, 101))
    model = yf - base
    top = float(np.max(np.abs(model[mask]))) if mask.any() else 1.0
    curves = {k: np.asarray(v, float) for k, v in (fit.get("site_spectra") or {}).items()}
    mas = float(settings.get("mas_Hz") or 0.0)
    suggested = set()
    sb_items, sb_flagged = _sideband_items(fit, sites, x, y, base, mask, sigma, larmor_MHz)
    items += sb_items
    if ratio > 1.3:
        lumps = [l for l in residual_lumps(x, y, yf, mask, sigma, w_pts)
                 if abs(l["height"]) > 0.02 * top][:4]
        sideband_lumps = {}            # owner label -> [lump centres]
        for lump in lumps:
            i = int(np.argmin(np.abs(x - lump["centre"])))
            here = abs(model[i]) / (top or 1.0)
            if lump["sign"] > 0 and here < 0.15:
                owner = _sideband_owner(lump["centre"], sites, mas, larmor_MHz, dx)
                if owner is not None:
                    sideband_lumps.setdefault(owner["label"], []).append(lump["centre"])
                    continue
                # Touching a peak's own pattern (its wings, a pattern edge that
                # reaches further): that peak's shape, not a new peak.
                neighbour = _adjacent_site(lump, sites, curves, x)
                if neighbour is not None:
                    if neighbour["label"] in suggested or neighbour["label"] in sb_flagged:
                        continue
                    suggested.add(neighbour["label"])
                    items.append(_item(
                        "tells", "note",
                        f"Next to {neighbour['label']}, around {lump['centre']:.4g} ppm, your data "
                        f"have more intensity than the fit: {neighbour['label']}'s pattern reaches "
                        "further out than the model's (its shape is not quite right yet)."))
                    euler_free = any(p.get("free") and p["name"] in ("alpha", "beta", "gamma")
                                     and p.get("label") == neighbour["label"]
                                     for p in fit.get("parameters") or [])
                    items += _shape_suggestions(neighbour, nucleus_spin, euler_free)
                    continue
                pct = 100 * lump["height"] / (top or 1.0)
                width = max(lump["high"] - lump["low"], 2 * dx)
                items.append(_item(
                    "tells", "warn",
                    f"There is signal near {lump['centre']:.4g} ppm (about {pct:.0f}% of the "
                    "tallest peak) that no peak in the model accounts for."))
                items.append(_item(
                    "next", "note",
                    f"Try adding a peak at about {lump['centre']:.4g} ppm, then compare the "
                    "two fits (keep both and use Compare in Kept fit models).",
                    {"type": "add_peak", "delta_iso": round(lump["centre"], 4),
                     "fwhm": round(width, 4)}))
                continue
            # On a peak: whose, and what to change.
            owner_label = max(curves, key=lambda k: abs(curves[k][i])) if curves else None
            owner = next((s for s in sites if s["label"] == owner_label), None)
            if owner is None or owner["label"] in suggested:
                continue
            if owner["label"] in sb_flagged:
                continue                      # its sidebands are the problem, said above
            if _sideband_owner(lump["centre"], [owner], mas, larmor_MHz, dx) is not None:
                # On one of its sidebands, not its centreband: the sideband
                # checks speak for it.
                sideband_lumps.setdefault(owner["label"], []).append(lump["centre"])
                continue
            suggested.add(owner["label"])
            items.append(_item(
                "tells", "note",
                f"The biggest mismatch is around {lump['centre']:.4g} ppm, on {owner['label']}: "
                "its shape is not quite right yet."))
            euler_free = any(p.get("free") and p["name"] in ("alpha", "beta", "gamma")
                             and p.get("label") == owner["label"]
                             for p in fit.get("parameters") or [])
            items += _shape_suggestions(owner, nucleus_spin, euler_free)
        for label, centres in sideband_lumps.items():
            owner = next(s for s in sites if s["label"] == label)
            if label in sb_flagged:
                continue                      # already said, with numbers
            where = _join([f"{c:.4g}" for c in sorted(centres)]) + " ppm"
            if owner["kinds"] & {"csa", "quadrupolar"}:
                what = ("the anisotropy" if "csa" in owner["kinds"] else "C_Q")
                items.append(_item(
                    "tells", "warn",
                    f"There is signal near {where} that the fit leaves out, where spinning "
                    f"sidebands of {label} should be: the fit's sidebands for {label} are too "
                    f"weak, so {what} is probably underestimated (or the MAS rate is off)."))
                items.append(_item(
                    "next", "note",
                    f"Try a larger starting value for {what} of {label} (or a few values in Try "
                    "starts)."))
            else:
                kind = "quadrupolar" if (nucleus_spin or 0) >= 1 and not owner["kinds"] else "csa"
                items.append(_item(
                    "tells", "warn",
                    f"There is signal near {where} that the model leaves out, where spinning "
                    f"sidebands of {label} would be -- but {label} has no anisotropy, so the model "
                    "gives it none."))
                items.append(_item(
                    "next", "note",
                    f"Try adding a {'CSA' if kind == 'csa' else 'quadrupolar coupling'} to "
                    f"{label}: an isotropic peak has no sidebands.",
                    {"type": "add_interaction", "site": label, "kind": kind}))
            suggested.add(label)


    # ---- parameters ----
    natural = {}
    try:
        from .combined_fitting import _default_bounds
        for s in fit.get("sites") or []:
            d = s.to_dict() if hasattr(s, "to_dict") else dict(s)
            for p in fit.get("parameters") or []:
                if p.get("label") == d.get("label"):
                    try:
                        natural[(d["label"], p["name"])] = _default_bounds(d, p["name"])
                    except (KeyError, ValueError, IndexError):
                        pass
    except ImportError:
        pass
    params = _parameter_items(fit.get("parameters") or [], fit.get("pcov"),
                              fit.get("free_names"), natural)
    root = any(it["action"] and it["action"]["type"] in ("free_mas", "add_peak", "add_interaction")
               and it["section"] == "next" for it in items)
    if root:
        # A model that is missing something (a peak, an interaction, the
        # right MAS rate) makes its other numbers move to compensate:
        # their uncertainties and trade-offs would change on refitting, so
        # listing them now only buries what to do first. Limits stay.
        limits = [it for it in params
                  if (it["action"] or {}).get("type") == "widen_bound"
                  or any(w in it["text"] for w in ("stopped at", "edge of", "limit ("))]
        items += limits
        if len(limits) < len(params):
            items.append(_item("tells", "note",
                               "How well each number is determined will change once the "
                               "suggestions below are tried, so it is not listed yet: fit again "
                               "after the first suggestion."))
    else:
        items += params

    # ---- peaks that are not needed ----
    for s in sites:
        share, err = s["share"], s["share_err"]
        if share is None or len(sites) < 2:
            continue
        if share < 2.0 or (err is not None and err > share):
            items.append(_item(
                "tells", "note",
                f"{s['label']} makes up only {share:.1f}% of the signal"
                + (" -- not distinguishable from zero" if err is not None and err > share else "")
                + ": your data may not need it."))
            items.append(_item(
                "next", "note",
                f"Try removing {s['label']} and fitting again; keep both fits and compare them "
                "(Kept fit models ▸ Compare).", {"type": "remove_site", "site": s["label"]}))
    for a in range(len(sites)):
        for b in range(a + 1, len(sites)):
            sa, sb = sites[a], sites[b]
            if sa["kinds"] != sb["kinds"]:
                continue
            close = abs(sa["delta_iso"] - sb["delta_iso"]) < 0.5 * max(min(sa["fwhm"], sb["fwhm"]),
                                                                         dx)
            same = all(_similar(sa[k], sb[k]) for k in ("fwhm", "aniso", "C_Q"))
            if close and same:
                items.append(_item(
                    "tells", "note",
                    f"{sa['label']} and {sb['label']} have ended up almost identical: one peak "
                    "probably describes them both."))
                items.append(_item("next", "note",
                                   f"Try removing {sb['label']} and compare the fits.",
                                   {"type": "remove_site", "site": sb["label"]}))

    # ---- window and baseline (fit_combined's own checks, in plain words) ----
    for w in fit.get("warnings") or []:
        if "clipped" in w:
            items.append(_item("next", "note",
                               "The fitted pattern runs off the edge of the fit window: widen "
                               "the window so the whole pattern, sidebands included, is inside."))
        elif "baseline was estimated" in w:
            items.append(_item("next", "note",
                               "The baseline was measured where the pattern still has "
                               "intensity, which pulls the fit: tick 'Fit the baseline too', "
                               "or widen the spectrum so its ends are flat."))

    # ---- other models of the same spectrum ----
    if alternatives:
        items += compare_items(fit, alternatives)
    return _dedupe(items)


def _similar(a, b, rel=0.15):
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    a, b = float(a), float(b)
    return abs(a - b) <= rel * max(abs(a), abs(b), 1e-9)


def _dedupe(items):
    seen, out = set(), []
    for it in items:
        key = (it["section"], it["text"])
        if key not in seen:
            seen.add(key)
            out.append(it)
    return out


# ============================================================
# Comparing models of the same spectrum
# ============================================================

def _n_free(fit):
    params = fit.get("parameters")
    if params is not None:
        return sum(1 for p in params if p.get("free"))
    k = fit.get("n_free")
    return None if k is None else int(k)


def _mask_of(fit, n):
    m = fit.get("mask")
    return np.ones(n, bool) if m is None else np.asarray(m, bool)


def common_points(fit, other):
    """The two fits on the points they share -- same ppm, same data,
    inside both fit windows: ``(y, this fit, other fit)``, or None when
    they are not fits of the same data (another spectrum or slice) or
    share too few points. A kept fit stores only its displayed range, so
    exact equality of the axes is not required."""
    xa, xb = np.asarray(fit["x"], float), np.asarray(other["x"], float)
    ya, yb = np.asarray(fit["y"], float), np.asarray(other["y"], float)
    fa, fb = np.asarray(fit["y_fit"], float), np.asarray(other["y_fit"], float)
    ma, mb = _mask_of(fit, xa.size), _mask_of(other, xb.size)
    order = np.argsort(xa)
    xs = xa[order]
    j = np.clip(np.searchsorted(xs, xb), 1, xs.size - 1)
    nearest = np.where(np.abs(xs[j - 1] - xb) <= np.abs(xs[j] - xb), j - 1, j)
    step = float(np.median(np.abs(np.diff(xs)))) if xs.size > 1 else 1.0
    ok = np.abs(xs[nearest] - xb) <= 1e-6 * max(step, 1e-12) + 1e-9
    if ok.sum() < max(16, 0.5 * xb.size):
        return None
    ia = order[nearest[ok]]
    scale = float(np.max(np.abs(ya))) or 1.0
    if not np.allclose(ya[ia], yb[ok], rtol=1e-6, atol=1e-9 * scale):
        return None
    keep = ma[ia] & mb[ok]
    if keep.sum() < 16:
        return None
    return ya[ia][keep], fa[ia][keep], fb[ok][keep]


def information_criterion(fit):
    """``(BIC, n points, k free parameters)`` of a fit over its window,
    or None when the fit lacks its curves or its parameter count."""
    if fit.get("y") is None or fit.get("y_fit") is None:
        return None
    y, f = np.asarray(fit["y"], float), np.asarray(fit["y_fit"], float)
    mask = _mask_of(fit, y.size)
    n, k = int(mask.sum()), _n_free(fit)
    rss = float(np.sum((y[mask] - f[mask]) ** 2))
    if k is None or n <= k + 1 or rss <= 0:
        return None
    return n * np.log(rss / n) + k * np.log(n), n, k


def _bic(y, f, k):
    n = y.size
    rss = float(np.sum((y - f) ** 2))
    if rss <= 0 or n <= k + 1:
        return None
    return n * np.log(rss / n) + k * np.log(n)


def _describe_model(fit):
    source = fit.get("sites") or fit.get("site_dicts")
    if source or fit.get("site_spectra"):
        sites = _site_info({**fit, "sites": source} if source else fit)
        kinds = sorted({k for s in sites for k in s["kinds"]})
        names = {"csa": "CSA", "quadrupolar": "quadrupolar", "jcoupling": "J"}
        return (f"{len(sites)} peak{'s' if len(sites) != 1 else ''}"
                + (" with " + "/".join(names.get(k, k) for k in kinds) if kinds else ", isotropic"))
    k = _n_free(fit)
    labels = fit.get("labels") or []
    if labels:
        return f"{len(labels)} component{'s' if len(labels) != 1 else ''}, {k} free parameters"
    return f"{k} free parameters" if k is not None else "unknown parameters"


def compare_items(fit, alternatives, this_name="this fit"):
    """Advice comparing ``fit`` with each ``(name, other fit)`` of the same
    data, on the points both fitted, by the Bayesian information criterion
    (BIC) -- which charges each extra parameter, so a model with more peaks
    must earn them. ΔBIC bands: < 2 no real difference, 2-6 positive,
    6-10 strong, > 10 very strong. ``other`` needs ``x``, ``y``,
    ``y_fit``, and ``parameters`` or ``n_free`` (a kept fit model has all)."""
    out = []
    k_this = _n_free(fit)
    if fit.get("y") is None or k_this is None:
        return out
    for name, other in alternatives:
        k_other = _n_free(other)
        if other.get("y") is None or other.get("y_fit") is None or k_other is None:
            continue
        shared = common_points(fit, other)
        if shared is None:
            continue                  # another spectrum, slice or window: not comparable
        y, f_this, f_other = shared
        b_this, b_other = _bic(y, f_this, k_this), _bic(y, f_other, k_other)
        if b_this is None or b_other is None:
            continue
        d = b_other - b_this          # > 0: this fit is better
        desc_mine, desc_theirs = _describe_model(fit), _describe_model(other)
        if abs(d) <= 2:
            simpler = None
            if k_other < k_this:
                simpler = f"“{name}”"
            elif k_this < k_other:
                simpler = "this fit"
            out.append(_item("tells", "note",
                             f"This fit ({desc_mine}) and “{name}” ({desc_theirs}) describe "
                             "your data equally well"
                             + (f": prefer the simpler one ({simpler}, with fewer parameters)."
                                if simpler else ".")))
            continue
        strength = ("clearly better" if abs(d) > 10 else "better" if abs(d) > 6
                    else "slightly better")
        if d > 0:
            extra = k_this - k_other
            why = (" -- by far more than its extra parameters would give by chance"
                   if extra > 0 and d > 10 else "")
            out.append(_item("tells", "good",
                             f"This fit ({desc_mine}) describes your data {strength} than "
                             f"“{name}” ({desc_theirs}){why}."))
        else:
            out.append(_item("tells", "warn",
                             f"“{name}” ({desc_theirs}) describes your data {strength} than "
                             f"this fit ({desc_mine})."))
    return out


# ============================================================
# Pseudo-2D series: every slice of one series
# ============================================================

def advise_series(results):
    """Advice on a pseudo-2D series measured slice by slice: a model
    applied to every slice (``model_fit``), an isotropic peak fit, or
    region integrals. Each result needs ``index`` and, for the shape
    checks, ``x``, ``y``, ``y_fit``."""
    items = []
    if not results:
        return items
    first = results[0]
    is_model = bool(first.get("model_fit"))
    has_fit = all(r.get("y_fit") is not None and r.get("x") is not None for r in results)
    labels = first.get("site_labels") or [f"Peak {k + 1}" for k in
                                          range(len(first.get("integrals") or []))]
    if has_fit:
        ratios = []
        for r in results:
            x, y, f = (np.asarray(r[k], float) for k in ("x", "y", "y_fit"))
            m = np.asarray(r["fit_mask"], bool) if r.get("fit_mask") is not None \
                else np.ones(y.size, bool)
            sigma = estimate_noise(x, y, f)
            res = (y - f)[m]
            ratios.append(float(np.sqrt(np.mean(res ** 2)) / sigma) if res.size else np.nan)
        ratios = np.asarray(ratios)
        bad = [r["index"] for r, q in zip(results, ratios) if q > 2.0]
        if not bad:
            items.append(_item(
                "tells", "good",
                ("The same lineshapes fit every slice to within the noise: only the intensities "
                 "change along the series, which is what a relaxation series should do."
                 if is_model else "Every slice is fitted to within its noise.")))
        else:
            worst = int(np.nanargmax(ratios))
            items.append(_item(
                "tells", "warn",
                ("The fixed lineshapes leave real signal behind in slice"
                 if is_model else "The fit leaves real signal behind in slice")
                + f"{'s' if len(bad) > 1 else ''} {_ranges(bad)} (worst, slice "
                f"{results[worst]['index']}: about {ratios[worst]:.0f}× the noise). "
                + ("The shape changes along the series -- for example, one part of the "
                   "pattern relaxing faster than another -- so the intensities from those "
                   "slices are less reliable." if is_model else
                   "The intensities from those slices are less reliable.")))
            items.append(_item(
                "next", "note",
                ("Try a model with more peaks, so parts that relax differently can change "
                 "independently; or fit the strongest of those slices on its own and compare."
                 if is_model else
                 "Look at those slices (Spectral Fit view) and add or adjust peaks there.")))
        # Leftover signal in the strongest slice: a missing peak, or a
        # peak whose shape is not right.
        strongest = max(results, key=lambda r: float(np.max(np.abs(np.asarray(r["y_fit"],
                                                                              float)))))
        items += _generic_lump_items(strongest, len(results) > 1)
    # Intensities in the noise.
    amps = [r.get("amplitudes") or r.get("integrals") for r in results]
    errs = [r.get("amplitude_errors") or r.get("integral_errors") for r in results]
    if all(a is not None for a in amps) and all(e is not None for e in errs):
        for k, label in enumerate(labels):
            weak = [r["index"] for r, a, e in zip(results, amps, errs)
                    if k < len(a) and k < len(e) and e[k] and abs(a[k]) < 3 * abs(e[k])]
            if weak and len(weak) < len(results):
                items.append(_item(
                    "tells", "note",
                    f"{label} is within the noise in slice{'s' if len(weak) > 1 else ''} "
                    f"{_ranges(weak)}: those points carry little information (they count in "
                    "the curve fit with large error bars). That is expected where a signal "
                    "passes through zero or has decayed."))
            elif weak:
                items.append(_item(
                    "tells", "warn",
                    f"{label} is within the noise in every slice: no relaxation curve can be "
                    "measured from it."))
    # Isotropic fits' own limits.
    hits = [r["index"] for r in results if any("bound" in str(w) for w in r.get("warnings") or [])]
    if hits:
        items.append(_item(
            "tells", "note",
            f"In slice{'s' if len(hits) > 1 else ''} {_ranges(hits)} a peak's position, width or "
            "height stopped at its limit; those slices' values are less certain."))
        items.append(_item("next", "note",
                           "Loosen the position tolerance (or the peak's bounds) and fit again."))
    return _dedupe(items)


def _generic_lump_items(r, is_series):
    """Missing peaks / wrong shapes in one fitted slice, without knowing the
    model's interactions: signal far from every fitted peak is a missing
    peak; signal on a peak's flank is its shape."""
    out = []
    x, y, f = (np.asarray(r[k], float) for k in ("x", "y", "y_fit"))
    order = np.argsort(x)
    x, y, f = x[order], y[order], f[order]
    m = (np.asarray(r["fit_mask"], bool)[order] if r.get("fit_mask") is not None
         else np.ones(x.size, bool))
    base = np.asarray(r["baseline"], float)[order] if r.get("baseline") is not None \
        else np.zeros(x.size)
    widths = [float(p.get("fwhm")) for p in r.get("peak_defs") or [] if p.get("fwhm")]
    dx = float(np.median(np.abs(np.diff(x)))) if x.size > 1 else 1.0
    width = min(widths) if widths else 10 * dx
    sigma = estimate_noise(x, y, f, base)
    res = (y - f)[m]
    if not res.size or np.sqrt(np.mean(res ** 2)) / sigma <= 1.3:
        return out
    model = f - base
    top = float(np.max(np.abs(model[m]))) or 1.0
    where = f" (slice {r.get('index')})" if is_series else ""
    for lump in [l for l in residual_lumps(x, y, f, m, sigma,
                                           int(np.clip(round(width / dx), 3, 101)))
                 if abs(l["height"]) > 0.02 * top][:3]:
        reach = 5 * width + (lump["high"] - lump["low"])
        near = (x >= lump["low"] - reach) & (x <= lump["high"] + reach)
        on_peak = float(np.max(np.abs(model[near]))) / top > 0.02 if near.any() else False
        if lump["sign"] > 0 and not on_peak:
            out.append(_item("tells", "warn",
                             f"There is signal near {lump['centre']:.4g} ppm{where} (about "
                             f"{100 * lump['height'] / top:.0f}% of the tallest peak) that no "
                             "fitted peak accounts for."))
            out.append(_item("next", "note",
                             f"Try adding a peak at about {lump['centre']:.4g} ppm and fit again.",
                             {"type": "add_peak", "delta_iso": round(lump["centre"], 4),
                              "fwhm": round(max(lump["high"] - lump["low"], 2 * dx), 4)}))
        else:
            out.append(_item("tells", "note",
                             f"Around {lump['centre']:.4g} ppm{where} the fitted peak's shape does "
                             "not quite match the data."))
            out.append(_item("next", "note",
                             "Let the peak's Gaussian/Lorentzian mix (eta) vary, or try two "
                             "peaks there if it looks like it has a shoulder."))
    return out


def _ranges(indices):
    indices = sorted(int(i) for i in indices)
    out, start, prev = [], None, None
    for i in indices:
        if start is None:
            start = prev = i
        elif i == prev + 1:
            prev = i
        else:
            out.append(f"{start}–{prev}" if prev > start else f"{start}")
            start = prev = i
    if start is not None:
        out.append(f"{start}–{prev}" if prev > start else f"{start}")
    return ", ".join(out)


# ============================================================
# Relaxation / diffusion curves (Dynamics ▸ Fit Curve)
# ============================================================

_RATE_WORDS = {"T1": ("recovery", "T1"), "T2": ("decay", "T2"), "T1rho": ("decay", "T1ρ"),
               "D": ("decay", "D")}


def advise_relaxation(result, label=None):
    """Advice on one :func:`opennmr.processing.relaxation.fit_relaxation`
    result: whether the delays span the curve, how well the rate is
    determined, whether the points scatter randomly about the curve, and
    limits."""
    items = []
    t = np.asarray(result.get("t"), float)
    y = np.asarray(result.get("y"), float)
    f = result.get("y_fit")
    if t.size == 0 or f is None:
        return items
    f = np.asarray(f, float)
    who = f" for {label}" if label else ""
    names = list(result.get("param_names") or [])
    params = result.get("params") or {}
    errors = result.get("errors") or {}
    rate = result.get("rate_param")
    model = str(result.get("model") or "")
    n_par = len(result.get("free_names") or names)
    if t.size < n_par + 3:
        items.append(_item("tells", "warn",
                           f"Only {t.size} points{who} for {n_par} fitted parameters: the fit "
                           "has little to go on, and its uncertainties are themselves uncertain."))
    resid = y - f
    scatter = float(np.std(resid)) if resid.size > 1 else 0.0
    # Runs of same-signed residuals: few long runs = a systematic misfit.
    signs = np.sign(resid[np.abs(resid) > 1e-12 * (np.max(np.abs(y)) or 1)])
    systematic = False
    if signs.size >= 8:
        n_pos, n_neg = int((signs > 0).sum()), int((signs < 0).sum())
        runs = 1 + int(np.sum(signs[1:] != signs[:-1]))
        if n_pos and n_neg:
            mu = 2 * n_pos * n_neg / (n_pos + n_neg) + 1
            var = (mu - 1) * (mu - 2) / (n_pos + n_neg - 1)
            z = (runs - mu) / np.sqrt(var) if var > 0 else 0.0
            systematic = z < -2.0
    if rate in params and params[rate] and np.isfinite(params[rate]):
        value = float(params[rate])
        err = errors.get(rate)
        word, sym = _RATE_WORDS.get(rate, ("change", rate))
        order = np.argsort(t)
        t_s = t[order]
        extrapolated = False
        if rate == "D":
            span_hi = float(t_s[-1]) * value
            if span_hi < 1.5:
                extrapolated = True
                items.append(_item(
                    "tells", "warn",
                    f"The signal has only decayed to about {np.exp(-span_hi):.0%} of its start "
                    f"by the strongest gradient{who}: D is extrapolated from the first part of "
                    "the decay."))
                items.append(_item("next", "note",
                                   "Add stronger gradients (or a longer diffusion time Δ) so "
                                   "the signal falls to about 10% or less."))
        elif rate in ("T1", "T2", "T1rho") and value > 0:
            longest, shortest = float(t_s[-1]), float(t_s[t_s > 0][0]) if np.any(t_s > 0) else 0.0
            if longest < 3 * value:
                extrapolated = True
                items.append(_item(
                    "tells", "warn",
                    f"Your longest delay ({longest:.3g} s) is only {longest / value:.1f}× {sym} "
                    f"({value:.3g} s){who}: the {word} has not "
                    f"{'levelled off' if word == 'recovery' else 'died away'} yet, so {sym} (and "
                    f"the {'final' if word == 'recovery' else 'baseline'} intensity) rest on an "
                    "extrapolation."))
                items.append(_item("next", "note",
                                   f"Add delays up to about 5× {sym} ({5 * value:.3g} s)."))
            if shortest > 0.3 * value:
                items.append(_item(
                    "tells", "note",
                    f"Your shortest delay ({shortest:.3g} s) is already {shortest / value:.1f}× "
                    f"{sym}{who}: the start of the {word} is not measured."))
                items.append(_item("next", "note",
                                   f"Add shorter delays, down to about {sym}/10 "
                                   f"({value / 10:.3g} s)."))
        if err is not None and np.isfinite(err) and value:
            rel = abs(float(err) / value)
            if rel > 0.2:
                items.append(_item("tells", "warn",
                                   f"{sym}{who} = {value:.3g} ± {float(err):.2g} ({rel:.0%}): only "
                                   "roughly determined."))
            elif rel < 0.05 and not systematic and extrapolated:
                items.append(_item("tells", "note",
                                   f"{sym}{who} = {value:.4g} ± {float(err):.2g} (±{rel:.1%}) -- "
                                   "but that small uncertainty assumes the curve carries on as "
                                   "the model says beyond your last point."))
            elif rel < 0.05 and not systematic:
                items.append(_item("tells", "good",
                                   f"{sym}{who} = {value:.4g} ± {float(err):.2g} is well "
                                   f"determined (±{rel:.1%})."))
    if systematic:
        items.append(_item(
            "tells", "warn",
            f"The points{who} curve away from the fit in a systematic way instead of scattering "
            "randomly about it: one simple exponential does not describe them."))
        if "stretched" not in model and "biexp" not in model and "triexp" not in model:
            target = ("stretched_recovery" if "recovery" in model else
                      "stejskal_tanner_stretched" if "stejskal" in model else "stretched_decay")
            items.append(_item(
                "next", "note",
                "Try a stretched-exponential model (a spread of rates, common in solids) or a "
                "two-component one (two distinct environments), and compare.",
                {"type": "set_relaxation_model", "model": target}))
        elif model in ("biexponential_decay", "biexponential_recovery"):
            target = model.replace("biexponential", "triexponential")
            items.append(_item(
                "next", "note",
                "Two components still leave a pattern: try three, then check whether the third "
                "is supported (the advice for that fit compares it with this one).",
                {"type": "set_relaxation_model", "model": target}))
    elif scatter and t.size >= 6:
        items.append(_item("tells", "good",
                           f"The points{who} scatter randomly about the curve: the model "
                           "describes their shape."))
    # Components that are not distinct.
    rates = [n for n in names if n.startswith(("T1", "T2", "D")) and n != rate]
    if model.startswith(("biexp", "stejskal_tanner_biexp")) and len(rates) >= 2:
        a, b = (abs(float(params.get(n, np.nan))) for n in rates[:2])
        if np.isfinite(a) and np.isfinite(b) and max(a, b) < 2.5 * min(a, b):
            items.append(_item(
                "tells", "note",
                f"The two components{who} have similar rates ({rates[0]} = {a:.3g}, "
                f"{rates[1]} = {b:.3g}): the data may not support two -- compare with a "
                "single-component or stretched fit."))
    if model.startswith("triexp"):
        items += _triexp_items(result, names, params, errors, who)
    items += _correlation_items(result, who)
    # Limits.
    bounds = result.get("bounds")
    fixed = result.get("fixed") or {}
    if bounds:
        lo, hi = bounds
        for name, l, h in zip(names, lo, hi):
            if name in fixed or name not in params:
                continue
            v = float(params[name])
            e = abs(float(errors.get(name) or 0.0))
            if not np.isfinite(e):
                e = 0.0

            def at(bound):
                # Within its 1σ of the bound *and* close to it in its own
                # terms: a poorly determined value far from a bound (a
                # 0.1 s time constant ± 0.3 s, near the 1e-9 floor only
                # through its error bar) is not "at a limit".
                gap = abs(v - bound)
                return gap <= e + 1e-9 * max(1.0, abs(v)) and \
                    gap <= 0.01 * max(abs(v), abs(bound)) + 1e-12
            if (np.isfinite(h) and (v >= h or at(h))) or (np.isfinite(l) and (v <= l or at(l))):
                items.append(_item(
                    "tells", "warn",
                    f"{name}{who} stopped at a limit of its range ({v:.3g}): treat it as a "
                    "bound, not a value."))
                items.append(_item("next", "note",
                                   f"Widen {name}'s range in Starting values (Min/Max) and fit "
                                   "again."))
    return _dedupe(items)


def _weighted_bic(y, f, k, y_err=None):
    """BIC of a curve fit: χ² + k ln n when the points carry their own
    errors, n ln(RSS/n) + k ln n when they do not."""
    y, f = np.asarray(y, float), np.asarray(f, float)
    n = y.size
    if n <= k + 1:
        return None
    if y_err is not None:
        e = np.asarray(y_err, float)
        if e.shape == y.shape and np.all(np.isfinite(e)) and np.all(e > 0):
            return float(np.sum(((y - f) / e) ** 2)) + k * np.log(n)
    return _bic(y, f, k)


def _triexp_items(result, names, params, errors, who):
    """Whether three components are supported: distinct time constants,
    determined amplitudes, and a lower BIC than two components on the
    same points."""
    items = []
    times = [n for n in names if n[:2] in ("T1", "T2") and n[-1:] in ("a", "b", "c")]
    amps = [n for n in names if n in ("A1", "A2", "A3")]
    vals = [abs(float(params.get(n, np.nan))) for n in times]
    for i in range(len(vals) - 1):
        a, b = vals[i], vals[i + 1]
        if np.isfinite(a) and np.isfinite(b) and max(a, b) < 2.5 * min(a, b):
            items.append(_item(
                "tells", "warn",
                f"{times[i]} = {a:.3g} s and {times[i + 1]} = {b:.3g} s{who} are within a "
                "factor of 2.5: a sum of exponentials cannot separate time constants that "
                "close, so these two are one component split in half."))
    for n in amps:
        v, e = params.get(n), errors.get(n)
        if v is not None and e is not None and np.isfinite(e) and v and abs(e / v) > 0.5:
            items.append(_item(
                "tells", "warn",
                f"{n}{who} = {float(v):.3g} ± {float(e):.2g}: that component's size is not "
                "determined by these data."))
    # Against two components, on the same points.
    try:
        from ..processing.relaxation import fit_relaxation
        two_model = str(result.get("model")).replace("triexponential", "biexponential")
        t = np.asarray(result["t"], float)
        y = np.asarray(result["y"], float)
        y_err = result.get("y_err")
        fixed = {k: v for k, v in (result.get("fixed") or {}).items() if k == "C"}
        two = fit_relaxation(t, y, model=two_model, y_err=y_err, fixed=fixed or None)
        k3 = len(result.get("free_names") or names)
        k2 = len(two.get("free_names") or two["param_names"])
        b3 = _weighted_bic(y, result["y_fit"], k3, y_err)
        b2 = _weighted_bic(y, two["y_fit"], k2, y_err)
    except Exception:  # noqa: BLE001 -- advice is best-effort
        return items
    if b3 is None or b2 is None:
        return items
    d = b3 - b2
    if d > -2:
        items.append(_item(
            "tells", "warn",
            f"Two components describe these points{who} as well as three once the extra "
            f"parameters are counted (ΔBIC = {d:+.1f}): the third component is not supported. "
            "Quote the two-component fit.",
            {"type": "set_relaxation_model", "model": two_model}))
    elif d > -10:
        items.append(_item(
            "tells", "note",
            f"Three components fit{who} somewhat better than two (ΔBIC = {d:+.1f}), but not "
            "decisively: more points, or less noise, would settle it."))
    else:
        items.append(_item(
            "tells", "good",
            f"Three components fit{who} clearly better than two (ΔBIC = {d:+.1f}) -- far "
            "more than the extra parameters would give by chance."))
    return items


def _correlation_items(result, who, limit=0.95):
    """Parameters the fit could not tell apart (|r| ≥ ``limit``)."""
    try:
        from ..processing.relaxometry import strong_correlations
        pairs = strong_correlations(result, limit)
    except Exception:  # noqa: BLE001 -- no covariance: nothing to say
        return []
    if not pairs:
        return []
    text = ", ".join(f"{a} and {b} (r = {r:+.3f})" for a, b, r in pairs[:3])
    return [_item(
        "tells", "note",
        f"Strongly correlated parameters{who}: {text}. Each can trade off against the other "
        "with little change in the fit, so their individual errors understate how loosely "
        "they are pinned together; the correlation matrix shows them all.")]


# ============================================================
# Multi-spectrum fits
# ============================================================

def advise_multi_fit(result, per_experiment=2):
    """Advice on a :func:`~.multi_fitting.fit_multi` result: the fit of
    each spectrum (its verdict and its ``per_experiment`` most useful
    next steps), then the series as a whole."""
    items = []
    fits = [f for f in result.get("experiments") or [] if f.get("y") is not None]
    ratios = []
    for f in fits:
        own = advise_combined_fit(f)
        if not own:
            continue
        verdict = own[0]
        ratios.append((f.get("label"), verdict["tone"]))
        items.append(_item("tells", verdict["tone"], f"{f.get('label')}: {verdict['text']}"))
        nexts = [it for it in own if it["section"] == "next"][:per_experiment]
        items += [_item("next", it["tone"], f"{f.get('label')}: {it['text']}", it.get("action"))
                  for it in nexts]
    if fits and len(fits) < len(result.get("experiments") or []):
        items.append(_item("tells", "note",
                           "Only the experiment selected carries its curves here; select "
                           "another to see the advice on its fit."))
    poor = [lab for lab, tone in ratios if tone == "warn"]
    if len(ratios) > 1 and poor and len(poor) < len(ratios):
        items.append(_item("tells", "warn",
                           f"Most spectra fit, but not {_join(poor)}: what is true of the others "
                           "may not hold there (a new peak, or a change in shape)."))
    for w in result.get("warnings") or []:
        if "R² fell" in w:
            items.append(_item("next", "note",
                               "A fit got worse along the series: start each spectrum from the "
                               "starting model (Start each fit: from the starting model) to "
                               "check that an earlier bad fit was not carried forward."))
            break
    return _dedupe(items)


# ============================================================
# Output
# ============================================================

_MARK = {"good": "✓", "note": "•", "warn": "⚠"}


def advice_text(items):
    """The advice as plain text: what the fit tells you, then what to
    try next -- for reports, logs and notebooks."""
    tells = [it for it in items if it["section"] == "tells"]
    nexts = [it for it in items if it["section"] == "next"]
    lines = []
    if tells:
        lines.append("What the fit tells you:")
        lines += [f"  {_MARK.get(it['tone'], '•')} {it['text']}" for it in tells]
    if nexts:
        lines.append("What to try next:")
        lines += [f"  → {it['text']}" for it in nexts]
    return "\n".join(lines)


def print_advice(items):
    text = advice_text(items)
    print(text)
    return text
