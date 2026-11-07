"""Bond-pathway shift contributions from DFT, for the CIF pNMR simulator.

The pNMR (CIF) simulator (:mod:`opennmr.notebook.pnmr_simulator`) builds a
disordered material's spectrum from additive contributions: each
paramagnetic neighbour of species s in shell k adds P(k, s) ppm to the
observed nucleus. This module works those P out from periodic hyperfine
calculations (CRYSTAL, or any calculation openNMR can turn into hyperfine
shifts), in either of the two usual ways:

* ``"spin_flip"`` (Middlemiss et al., Chem. Mater. 2013): a ferromagnetic
  reference, and calculations in each of which one paramagnetic ion j has
  its spin reversed. Reversing j turns its contribution from +P_j to −P_j,
  so P_jn = (δ_FM,n − δ_flip_j,n)/2 for every observed nucleus n.
* ``"single"``: calculations with one paramagnetic ion each, the others
  replaced by a diamagnetic analogue (Ga³⁺ for Fe³⁺, Sc³⁺, Zn²⁺ …); every
  hyperfine shift in such a cell is that ion's: P_jn = δ_n.

Each (ion, nucleus) pair is put in a shell by the shortest ion–nucleus
distance (minimum image). In a periodic cell the nucleus also feels every
*other* image of the same ion; when another image lies inside the shells,
the pair is flagged as ambiguous -- the cell is too small for that pathway
to be separated, and a larger supercell is needed.

The result is a table of every pair, the mean P (and spread) per shell and
species, ``shift_dict`` in the simulator's own format, and an additivity
check: the reference cell's own shifts predicted from the mean P and its
neighbour counts (spin flip only), which shows how well the additive
model holds before it is used on a disordered material.
"""

from __future__ import annotations

import os

import numpy as np

__all__ = ["hyperfine_shifts", "pathway_contributions", "plot_pathways", "pathways_to_python",
           "MODES"]

MODES = {"spin_flip": "Ferromagnetic reference + one calculation per flipped ion",
         "single": "One paramagnetic ion per calculation (others diamagnetic)"}


def _read(c):
    if isinstance(c, (str, os.PathLike)):
        from .dft_nmr import read_calculation
        return read_calculation(c)
    return c


def hyperfine_shifts(calc, paramagnetic, components="both") -> dict:
    """{label: isotropic hyperfine shift (ppm)} of every atom of ``calc``
    that has one, after scaling by ``paramagnetic`` (the same dictionary as
    :func:`dft_nmr.dft_sites`'s: φ, χ or θ/μ_eff/S for CRYSTAL; ``model``
    and its parameters for an ORCA EPR output)."""
    calc = _read(calc)
    args = dict(paramagnetic or {})
    args.pop("components", None)
    if calc.get("code") == "CRYSTAL":
        from . import crystal_nmr
        c = crystal_nmr.apply_scaling(calc, crystal_nmr.magnetic_scaling(**args),
                                      components=components)
        return {a["label"]: float(np.trace(a["delta_hf"]) / 3.0) for a in c["atoms"]
                if a.get("delta_hf") is not None}
    if calc.get("epr"):
        from . import paramagnetic_nmr as pm
        c = pm.apply_paramagnetic(calc, components=components, **args)
        key = {"both": "total", "contact": "contact", "dipolar": "pseudocontact"}[components]
        return {a["label"]: float(a["para"][key]) for a in c["atoms"] if a.get("para")}
    raise ValueError(f"{os.path.basename(str(calc.get('path')))}: no hyperfine data (a CRYSTAL "
                     "properties output, or an ORCA EPR/NMR output with hyperfine tensors).")


def _min_image(d_cart, lattice):
    if lattice is None:
        return d_cart
    L = np.asarray(lattice, dtype=float)
    f = np.linalg.solve(L.T, d_cart)
    f -= np.round(f)
    return L.T @ f


def _image_distances(p_from, p_to, lattice, rmax):
    d = np.asarray(p_to, dtype=float) - np.asarray(p_from, dtype=float)
    if lattice is None:
        return np.array([np.linalg.norm(d)])
    from .pseudocontact import _translations
    d0 = _min_image(d, lattice)
    r = np.linalg.norm(d0[None, :] + _translations(lattice, rmax + 1.0), axis=1)
    return np.sort(r[r <= rmax + 1e-9])


def _atoms(calc, species):
    return [a for a in calc["atoms"] if a["species"] in species]


def _shell(d, bins):
    for name, lo, hi in bins:
        if lo < d <= hi:
            return name
    return None


def _flipped_ion(ref, pert, magnetic):
    """The magnetic ion whose spin density at the nucleus changed sign."""
    rho_ref = {a["label"]: a.get("rho") for a in ref["atoms"]}
    out = []
    for a in _atoms(pert, magnetic):
        r0, r1 = rho_ref.get(a["label"]), a.get("rho")
        if r0 is not None and r1 is not None and r0 * r1 < 0:
            out.append(a["label"])
    return out


def _single_ion(calc, magnetic):
    ions = _atoms(calc, magnetic)
    if len(ions) == 1:
        return [ions[0]["label"]]
    rho = [(abs(a.get("rho") or 0.0), a["label"]) for a in ions]
    top = max((r for r, _ in rho), default=0.0)
    if top > 0:
        return [lab for r, lab in rho if r > 0.05 * top]
    return [a["label"] for a in ions]


def pathway_contributions(calcs, *, observed, magnetic, shell_bins, mode="spin_flip",
                          reference=None, flipped=None, paramagnetic=None,
                          components="both") -> dict:
    """Bond-pathway contributions P (ppm) per shell and paramagnetic species.

    ``calcs`` : the perturbed calculations (paths or dicts): one flipped ion
    each (``mode="spin_flip"``, with the ferromagnetic ``reference``), or
    one paramagnetic ion each (``mode="single"``). ``observed`` : the
    observed element ("Na"); ``magnetic`` : the paramagnetic species
    (["Mn", "Fe"]). ``shell_bins`` : [(name, r_min, r_max)] in Å, as in
    the CIF simulator. ``flipped`` : the ion of each calculation by label
    (``["Mn3", "Fe7"]``), when it cannot be found from the spin densities.
    ``paramagnetic`` : the magnetic scaling (see :func:`hyperfine_shifts`),
    the same for every calculation.

    Returns ``rows`` (one per ion–nucleus pair), ``summary`` (per shell and
    species: mean, sd, n, min, max), ``shift_dict`` ({shell: {species:
    mean}}), ``additivity`` (spin flip) and ``notes``."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {', '.join(MODES)}.")
    if isinstance(magnetic, str):
        magnetic = [magnetic]
    magnetic = list(magnetic)
    bins = [(str(n), float(lo), float(hi)) for n, lo, hi in shell_bins]
    if not bins:
        raise ValueError("Give at least one shell (name, r_min, r_max in Å).")
    rmax = max(hi for _n, _lo, hi in bins)
    calcs = [_read(c) for c in calcs]
    if not calcs:
        raise ValueError("Give at least one calculation.")
    if flipped is not None and len(flipped) != len(calcs):
        raise ValueError("Give one ion label per calculation in flipped=.")
    notes = []
    ref = None
    ref_shift = None
    if mode == "spin_flip":
        if reference is None:
            raise ValueError("The spin-flip method needs the ferromagnetic reference calculation.")
        ref = _read(reference)
        ref_shift = hyperfine_shifts(ref, paramagnetic, components)
    rows = []
    for k, c in enumerate(calcs):
        name = os.path.basename(str(c.get("path", f"calc {k + 1}")))
        if flipped is not None:
            ions = [flipped[k]] if isinstance(flipped[k], str) else list(flipped[k])
        elif mode == "spin_flip":
            ions = _flipped_ion(ref, c, magnetic)
        else:
            ions = _single_ion(c, magnetic)
        if len(ions) != 1:
            raise ValueError(f"{name}: cannot tell which {'/'.join(magnetic)} ion is "
                             + ("flipped" if mode == "spin_flip" else "the paramagnetic one")
                             + f" ({len(ions)} candidates"
                             + (f": {', '.join(ions)}" if ions else "")
                             + "). Name it with flipped=[...].")
        by_label = {a["label"]: a for a in c["atoms"]}
        if ions[0] not in by_label:
            raise ValueError(f"{name}: no atom {ions[0]}.")
        ion = by_label[ions[0]]
        shifts = hyperfine_shifts(c, paramagnetic, components)
        lattice = c.get("lattice")
        for a in c["atoms"]:
            if a["species"] != observed or a["label"] not in shifts:
                continue
            if mode == "spin_flip":
                if a["label"] not in ref_shift:
                    continue
                P = 0.5 * (ref_shift[a["label"]] - shifts[a["label"]])
            else:
                P = shifts[a["label"]]
            dists = _image_distances(ion["position"], a["position"], lattice, rmax)
            d = float(dists[0]) if len(dists) else float(np.linalg.norm(
                _min_image(np.asarray(a["position"]) - ion["position"], lattice)))
            inside = [x for x in dists if _shell(x, bins) is not None]
            rows.append({"calc": name, "ion": ion["label"], "species": ion["species"],
                         "nucleus": a["label"], "distance": d, "shell": _shell(d, bins),
                         "P": float(P), "ambiguous": len(inside) > 1,
                         "images_in_shells": len(inside)})
    if not rows:
        raise ValueError(f"No {observed} nuclei with hyperfine shifts in the calculations.")
    summary = {}
    for r in rows:
        if r["shell"] is None:
            continue
        summary.setdefault(r["shell"], {}).setdefault(r["species"], []).append(r["P"])
    shift_dict, stats = {}, {}
    for name, _lo, _hi in bins:
        shift_dict[name] = {}
        for sp in magnetic:
            vals = np.array(summary.get(name, {}).get(sp, []))
            shift_dict[name][sp] = float(vals.mean()) if len(vals) else 0.0
            stats[(name, sp)] = {"mean": float(vals.mean()) if len(vals) else None,
                                 "sd": float(vals.std(ddof=1)) if len(vals) > 1 else None,
                                 "n": int(len(vals)),
                                 "min": float(vals.min()) if len(vals) else None,
                                 "max": float(vals.max()) if len(vals) else None}
    n_amb = sum(r["ambiguous"] for r in rows)
    if n_amb:
        notes.append(f"{n_amb} of {len(rows)} pairs see another image of the same ion inside the "
                     "shells: the cell is too small to separate those pathways (their P is the "
                     "sum over images). Use a larger supercell for them.")
    beyond = [r for r in rows if r["shell"] is None and abs(r["P"]) > 0.0]
    if beyond:
        big = max(beyond, key=lambda r: abs(r["P"]))
        notes.append(f"{len(beyond)} pairs lie beyond the last shell; the largest is "
                     f"{big['P']:+.3g} ppm ({big['ion']}–{big['nucleus']}, {big['distance']:.2f} Å)"
                     ": add a shell if that is not negligible.")
    for (name, sp), s in stats.items():
        if s["n"] > 1 and s["sd"] is not None and abs(s["mean"]) > 0 and \
                s["sd"] > 0.25 * abs(s["mean"]):
            notes.append(f"{name}/{sp}: the {s['n']} pathways spread widely (mean {s['mean']:.3g}, "
                         f"sd {s['sd']:.3g} ppm): one shell holds geometrically different "
                         "pathways (e.g. 90° and 180°); split it by distance if you can.")
    additivity = None
    if mode == "spin_flip":
        additivity = _additivity(ref, ref_shift, observed, magnetic, bins, shift_dict)
        if additivity["rows"]:
            notes.append(f"Additivity check on the ferromagnetic reference: the mean pathways "
                         f"reproduce its {observed} shifts to an rms of "
                         f"{additivity['rms']:.3g} ppm (largest {additivity['max']:.3g} ppm).")
    return {"rows": rows, "summary": stats, "shift_dict": shift_dict, "shell_bins": bins,
            "mode": mode, "observed": observed, "magnetic": magnetic,
            "additivity": additivity, "notes": notes}


def _additivity(ref, ref_shift, observed, magnetic, bins, shift_dict) -> dict:
    lattice = ref.get("lattice")
    rmax = max(hi for _n, _lo, hi in bins)
    ions = _atoms(ref, magnetic)
    out = []
    for a in ref["atoms"]:
        if a["species"] != observed or a["label"] not in ref_shift:
            continue
        pred = 0.0
        for ion in ions:
            for d in _image_distances(ion["position"], a["position"], lattice, rmax):
                sh = _shell(d, bins)
                if sh is not None:
                    pred += shift_dict[sh].get(ion["species"], 0.0)
        out.append({"nucleus": a["label"], "calculated": ref_shift[a["label"]],
                    "predicted": pred, "difference": pred - ref_shift[a["label"]]})
    diffs = np.array([r["difference"] for r in out]) if out else np.zeros(0)
    return {"rows": out, "rms": float(np.sqrt(np.mean(diffs ** 2))) if len(diffs) else 0.0,
            "max": float(np.max(np.abs(diffs))) if len(diffs) else 0.0}


def plot_pathways(result, figure=None, fontsize=11.0):
    """Each pathway's P against its distance, by paramagnetic species, with
    the shells shaded and each shell's mean marked."""
    import matplotlib.pyplot as plt
    from opennmr.plotting.labels import italicise_symbols as it
    fig = figure if figure is not None else plt.figure(figsize=(6.0, 4.2))
    fig.clear()
    ax = fig.add_subplot(111)
    cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2"])
    for k, (name, lo, hi) in enumerate(result["shell_bins"]):
        ax.axvspan(lo, hi, color="0.9" if k % 2 == 0 else "0.96", zorder=0)
        ax.text(0.5 * (lo + hi), 1.0, name, transform=ax.get_xaxis_transform(), ha="center",
                va="bottom", fontsize=fontsize * 0.75)
    for j, sp in enumerate(result["magnetic"]):
        c = cycle[j % len(cycle)]
        rs = [r for r in result["rows"] if r["species"] == sp]
        if not rs:
            continue
        ok = [r for r in rs if not r["ambiguous"]]
        amb = [r for r in rs if r["ambiguous"]]
        ax.plot([r["distance"] for r in ok], [r["P"] for r in ok], "o", color=c, ms=5, label=sp)
        if amb:
            ax.plot([r["distance"] for r in amb], [r["P"] for r in amb], "o", mfc="none", mec=c,
                    ms=6, label=f"{sp} (images overlap)")
        for name, lo, hi in result["shell_bins"]:
            s = result["summary"].get((name, sp))
            if s and s["mean"] is not None:
                ax.hlines(s["mean"], lo, hi, colors=c, lw=1.4)
    ax.axhline(0.0, color="0.3", lw=0.8)
    ax.set_xlabel(it(r"$r$ (Å)"), fontsize=fontsize)
    ax.set_ylabel(it(r"$P$ (ppm)"), fontsize=fontsize)
    ax.legend(fontsize=fontsize * 0.8, frameon=False)
    return fig


def pathways_to_python(calcs, *, reference=None, **kw) -> str:
    args = ", ".join(f"{k}={v!r}" for k, v in kw.items())
    return ("from opennmr.notebook import bond_pathways as bp\n"
            f"res = bp.pathway_contributions({list(map(str, calcs))!r}, "
            f"reference={reference!r}, {args})\n"
            "print(res['shift_dict'])\n"
            "for n in res['notes']:\n    print(n)\n")
