"""Lanthanide-induced shifts: Bleaney constants, ⟨S_z⟩ and the Reilley
separation of contact and pseudocontact shifts across an isostructural
lanthanide series.

For a nucleus i in a series of isostructural Ln complexes, the paramagnetic
shift is (Reilley, Good & Desreux 1975; Bleaney 1972; Golding & Halton 1972)

    δ_para(i, Ln) = F_i ⟨S_z⟩_Ln + G_i C_J,Ln ,

with ⟨S_z⟩ the spin expectation value of the Ln³⁺ ion (contact term; F_i ∝
the hyperfine coupling of nucleus i) and C_J Bleaney's constant
(pseudocontact term; G_i ∝ the geometry and the crystal-field parameter).
F_i and G_i are the same for every Ln *if* the structure, the crystal
field (Bleaney's assumption) and the hyperfine coupling stay the same along
the series. The two linear forms

    δ/⟨S_z⟩ = F + G (C_J/⟨S_z⟩)        δ/C_J = G + F (⟨S_z⟩/C_J)

make this a straight line; a break in either (commonly between the light
and heavy Ln, around Nd–Sm or Tb–Dy) signals a structural change, a change
of crystal field, or a change of F along the series.

Constants
---------
* :data:`BLEANEY_CJ`: Bleaney's C_J (Bleaney, J. Magn. Reson. 8, 91 (1972)),
  scaled to Dy = −100, at 300 K (including excited J multiplets for Sm, Eu).
* :data:`GOLDING_HALTON_SZ`: ⟨S_z⟩ of Golding & Halton (Aust. J. Chem. 25,
  2577 (1972)), 300 K, in the usual units (−⟨S_z⟩ per μ_B B/3kT).
* :func:`ground_state` / :func:`bleaney_ground_state` /
  :func:`sz_ground_state`: the same quantities from the free-ion ground J
  multiplet alone (Hund's rules, Landé g_J and Stevens' α_J), as a check on
  the tables: they agree for Ce–Nd and Tb–Yb to the precision tabulated
  (within ~10 %, the excited-multiplet admixture), and not for Sm and Eu,
  whose excited multiplets are populated at room temperature, nor for Pm,
  whose ground-multiplet C_J is almost zero (+0.1) so that J mixing sets
  the tabulated +2.0. Pm (radioactive) is rarely measured; check its
  entries against your source if you use it.

Tables differ between sources (Pinkerton et al. 1985 revised ⟨S_z⟩, notably
Sm and Eu): give your own with ``constants=`` if your reference values
differ, and say which you used.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np

__all__ = [
    "SERIES", "BLEANEY_CJ", "GOLDING_HALTON_SZ", "STEVENS_ALPHA", "ground_state",
    "bleaney_ground_state", "sz_ground_state", "constants_table", "reilley_analysis",
    "plot_reilley", "lanthanide_to_python",
]

#: Ln³⁺ with their 4f electron count.
SERIES = {"La": 0, "Ce": 1, "Pr": 2, "Nd": 3, "Pm": 4, "Sm": 5, "Eu": 6, "Gd": 7, "Tb": 8,
          "Dy": 9, "Ho": 10, "Er": 11, "Tm": 12, "Yb": 13, "Lu": 14}

BLEANEY_CJ = {"Ce": -6.3, "Pr": -11.0, "Nd": -4.2, "Pm": 2.0, "Sm": -0.7, "Eu": 4.0, "Gd": 0.0,
              "Tb": -86.0, "Dy": -100.0, "Ho": -39.0, "Er": 33.0, "Tm": 53.0, "Yb": 22.0}

GOLDING_HALTON_SZ = {"Ce": -0.98, "Pr": -2.97, "Nd": -4.49, "Pm": -4.01, "Sm": 0.06,
                     "Eu": 10.68, "Gd": 31.50, "Tb": 31.82, "Dy": 28.55, "Ho": 22.63,
                     "Er": 15.37, "Tm": 8.21, "Yb": 2.59}

#: Stevens' second-order factor α_J = ⟨J‖α‖J⟩ of the ground multiplet.
STEVENS_ALPHA = {"Ce": Fraction(-2, 35), "Pr": Fraction(-52, 2475), "Nd": Fraction(-7, 1089),
                 "Pm": Fraction(14, 40455), "Sm": Fraction(13, 315), "Eu": Fraction(0),
                 "Gd": Fraction(0), "Tb": Fraction(-1, 99), "Dy": Fraction(-2, 315),
                 "Ho": Fraction(-1, 450), "Er": Fraction(4, 1575), "Tm": Fraction(1, 99),
                 "Yb": Fraction(2, 63)}


def ground_state(ln) -> dict:
    """L, S, J and the Landé g_J of the free Ln³⁺ ground multiplet (Hund's
    rules)."""
    if ln not in SERIES:
        raise ValueError(f"{ln} is not a lanthanide (La–Lu).")
    n = SERIES[ln]
    ml = [3, 2, 1, 0, -1, -2, -3]
    up = min(n, 7)
    down = max(0, n - 7)
    S = Fraction(up - down, 2)
    L = abs(sum(ml[:up]) + sum(ml[:down]))
    J = abs(L - S) if n < 7 else L + S
    if J == 0:
        g = Fraction(0)
    else:
        g = 1 + (J * (J + 1) + S * (S + 1) - L * (L + 1)) / (2 * J * (J + 1))
    return {"n": n, "L": int(L), "S": float(S), "J": float(J), "g_J": float(g),
            "_J": J, "_g": g}


def bleaney_ground_state(ln, normalise=True) -> float:
    """C_J = g_J² J(J+1)(2J−1)(2J+3) α_J for the ground multiplet alone,
    scaled to Dy = −100 (``normalise``)."""
    def raw(x):
        s = ground_state(x)
        J, g = s["_J"], s["_g"]
        if J == 0:
            return 0.0
        return float(g ** 2 * J * (J + 1) * (2 * J - 1) * (2 * J + 3) * STEVENS_ALPHA[x])
    value = raw(ln)
    return -100.0 * value / raw("Dy") if normalise else value


def sz_ground_state(ln) -> float:
    """⟨S_z⟩ (Golding–Halton units) of the ground multiplet alone:
    (g_J − 1) g_J J(J+1)."""
    s = ground_state(ln)
    J, g = s["_J"], s["_g"]
    return float((g - 1) * g * J * (J + 1))


def constants_table(constants=None) -> dict:
    """{Ln: {"C_J", "Sz", "C_J_ground", "Sz_ground"}} for Ce–Yb: the
    tabulated values (or ``constants`` = {Ln: (C_J, ⟨S_z⟩)} overriding
    them) with the ground-multiplet values beside them."""
    out = {}
    for ln in BLEANEY_CJ:
        cj, sz = BLEANEY_CJ[ln], GOLDING_HALTON_SZ[ln]
        if constants and ln in constants:
            cj, sz = map(float, constants[ln])
        out[ln] = {"C_J": cj, "Sz": sz, "C_J_ground": bleaney_ground_state(ln),
                   "Sz_ground": sz_ground_state(ln)}
    return out


def _linfit(x, y):
    X = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    dof = len(x) - 2
    s2 = float(res @ res / dof) if dof > 0 else float("nan")
    cov = s2 * np.linalg.pinv(X.T @ X)
    ss = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(res @ res) / ss if ss > 0 else float("nan")
    return beta, np.sqrt(np.clip(np.diag(cov), 0, None)), r2, res


def reilley_analysis(shifts, *, delta_dia=0.0, constants=None, exclude=(), groups=None) -> dict:
    """Separate contact (F) and pseudocontact (G) contributions of one
    nucleus over a lanthanide series.

    ``shifts`` : {Ln: δ_obs (ppm)}. ``delta_dia`` : the diamagnetic shift
    (the La, Lu or Y analogue) taken off, giving δ_para; or a {Ln: value}.
    ``constants`` : {Ln: (C_J, ⟨S_z⟩)} to override the tables.
    ``exclude`` : Ln left out of the fits (still listed). ``groups`` :
    lists of Ln fitted separately (e.g. ``[["Ce", "Pr", "Nd", "Sm", "Eu"],
    ["Tb", "Dy", "Ho", "Er", "Tm", "Yb"]]``) to test for a break.

    Each fit gives F and G three ways: the direct two-parameter least
    squares δ_para = F⟨S_z⟩ + G C_J (no intercept; the best estimate) and
    the two linear plots, whose agreement is the test of the model."""
    table = constants_table(constants)
    unknown = [ln for ln in shifts if ln not in table]
    if unknown:
        raise ValueError(f"No constants for {', '.join(unknown)} (paramagnetic Ce–Yb only; "
                         "La and Lu are diamagnetic references).")
    rows = []
    for ln in sorted(shifts, key=lambda x: SERIES[x]):
        dd = delta_dia.get(ln, 0.0) if isinstance(delta_dia, dict) else float(delta_dia)
        dp = float(shifts[ln]) - dd
        rows.append({"Ln": ln, "delta_obs": float(shifts[ln]), "delta_para": dp,
                     "C_J": table[ln]["C_J"], "Sz": table[ln]["Sz"],
                     "used": ln not in exclude})
    group_list = groups or [[r["Ln"] for r in rows]]
    fits = []
    for members in group_list:
        sel = [r for r in rows if r["Ln"] in members and r["used"]]
        fits.append(_fit_group(sel, members))
    notes = []
    if groups and len(fits) == 2 and all(f["direct"] for f in fits):
        a, b = fits
        for key in ("F", "G"):
            va, ea = a["direct"][key], a["direct"][key + "_err"]
            vb, eb = b["direct"][key], b["direct"][key + "_err"]
            if np.isfinite(ea) and np.isfinite(eb) and abs(va - vb) > 2 * np.hypot(ea, eb):
                notes.append(f"{key} differs between the two groups by more than twice its "
                             f"combined error ({va:.4g} vs {vb:.4g}): a change along the series.")
    for f in fits:
        notes += f["notes"]
    for r in rows:
        f = next((f for f in fits if r["Ln"] in f["members"] and f["direct"]), None)
        r["contact"] = f["direct"]["F"] * r["Sz"] if f else None
        r["pseudocontact"] = f["direct"]["G"] * r["C_J"] if f else None
        r["residual"] = (r["delta_para"] - r["contact"] - r["pseudocontact"]) if f else None
    return {"rows": rows, "fits": fits, "notes": notes, "constants": table}


def _fit_group(sel, members) -> dict:
    out = {"members": list(members), "used": [r["Ln"] for r in sel], "direct": None,
           "sz_form": None, "cj_form": None, "notes": []}
    if len(sel) < 2:
        out["notes"].append(f"{', '.join(members)}: fewer than two lanthanides to fit.")
        return out
    d = np.array([r["delta_para"] for r in sel])
    sz = np.array([r["Sz"] for r in sel])
    cj = np.array([r["C_J"] for r in sel])
    X = np.column_stack([sz, cj])
    beta, *_ = np.linalg.lstsq(X, d, rcond=None)
    res = d - X @ beta
    dof = len(d) - 2
    s2 = float(res @ res / dof) if dof > 0 else float("nan")
    cov = s2 * np.linalg.pinv(X.T @ X)
    err = np.sqrt(np.clip(np.diag(cov), 0, None))
    out["direct"] = {"F": float(beta[0]), "G": float(beta[1]), "F_err": float(err[0]),
                     "G_err": float(err[1]), "rms": float(np.sqrt(np.mean(res ** 2)))}
    m = np.abs(sz) > 0.5          # δ/⟨S_z⟩ is ill-defined near ⟨S_z⟩ = 0 (Sm)
    if m.sum() >= 2:
        b, e, r2, _r = _linfit(cj[m] / sz[m], d[m] / sz[m])
        out["sz_form"] = {"F": float(b[0]), "G": float(b[1]), "F_err": float(e[0]),
                          "G_err": float(e[1]), "r2": r2, "x": (cj[m] / sz[m]).tolist(),
                          "y": (d[m] / sz[m]).tolist(),
                          "Ln": [r["Ln"] for r, k in zip(sel, m) if k]}
    m = np.abs(cj) > 0.5          # δ/C_J is undefined for Gd (C_J = 0)
    if m.sum() >= 2:
        b, e, r2, _r = _linfit(sz[m] / cj[m], d[m] / cj[m])
        out["cj_form"] = {"G": float(b[0]), "F": float(b[1]), "G_err": float(e[0]),
                          "F_err": float(e[1]), "r2": r2, "x": (sz[m] / cj[m]).tolist(),
                          "y": (d[m] / cj[m]).tolist(),
                          "Ln": [r["Ln"] for r, k in zip(sel, m) if k]}
    if dof == 0:
        out["notes"].append(f"{', '.join(out['used'])}: two lanthanides fix F and G exactly; "
                            "no test of the model.")
    for name, form in (("δ/⟨S_z⟩ plot", out["sz_form"]), ("δ/C_J plot", out["cj_form"])):
        if form and np.isfinite(form["r2"]) and form["r2"] < 0.95 and len(form["x"]) > 2:
            out["notes"].append(f"{', '.join(out['used'])}: the {name} is not straight "
                                f"(R² = {form['r2']:.3f}); check for a break along the "
                                "series (fit light and heavy Ln separately).")
    return out


def plot_reilley(result, figure=None, fontsize=11.0, fit_index=None):
    """The two Reilley plots side by side: δ/⟨S_z⟩ against C_J/⟨S_z⟩ and
    δ/C_J against ⟨S_z⟩/C_J, each Ln labelled, each group's line drawn."""
    import matplotlib.pyplot as plt
    from opennmr.plotting.labels import italicise_symbols as it
    fig = figure if figure is not None else plt.figure(figsize=(9.0, 4.0))
    fig.clear()
    axes = fig.subplots(1, 2, gridspec_kw={"wspace": 0.4})
    cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2"])
    fits = result["fits"] if fit_index is None else [result["fits"][fit_index]]
    excluded = [r for r in result["rows"] if not r["used"]]
    for k, f in enumerate(fits):
        c = cycle[k % len(cycle)]
        for ax, key in zip(axes, ("sz_form", "cj_form")):
            form = f[key]
            if not form:
                continue
            x, y = np.array(form["x"]), np.array(form["y"])
            ax.plot(x, y, "o", color=c, ms=5)
            for xi, yi, ln in zip(x, y, form["Ln"]):
                ax.annotate(ln, (xi, yi), xytext=(4, 3), textcoords="offset points",
                            fontsize=fontsize * 0.7)
            xs = np.linspace(x.min(), x.max(), 2) if len(x) > 1 else x
            a0 = form["F"] if key == "sz_form" else form["G"]
            a1 = form["G"] if key == "sz_form" else form["F"]
            ax.plot(xs, a0 + a1 * xs, "-", color=c, lw=1.2)
    for r in excluded:
        if abs(r["Sz"]) > 0.5:
            axes[0].plot(r["C_J"] / r["Sz"], r["delta_para"] / r["Sz"], "x", color="0.4")
        if abs(r["C_J"]) > 0.5:
            axes[1].plot(r["Sz"] / r["C_J"], r["delta_para"] / r["C_J"], "x", color="0.4")
    axes[0].set_xlabel(it(r"$C_J$ / $\langle S_z \rangle$"), fontsize=fontsize)
    axes[0].set_ylabel(it(r"$\delta_\mathrm{para}$ / $\langle S_z \rangle$ (ppm)"),
                       fontsize=fontsize)
    axes[1].set_xlabel(it(r"$\langle S_z \rangle$ / $C_J$"), fontsize=fontsize)
    axes[1].set_ylabel(it(r"$\delta_\mathrm{para}$ / $C_J$ (ppm)"), fontsize=fontsize)
    return fig


def lanthanide_to_python(shifts, delta_dia=0.0, exclude=(), groups=None, constants=None) -> str:
    return ("from opennmr.notebook import lanthanides as ln\n"
            f"res = ln.reilley_analysis({dict(shifts)!r}, delta_dia={delta_dia!r}, "
            f"exclude={tuple(exclude)!r}, groups={groups!r}, constants={constants!r})\n"
            "for f in res['fits']:\n    print(f['members'], f['direct'])\n"
            "fig = ln.plot_reilley(res)\n")
