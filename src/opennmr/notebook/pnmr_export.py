"""
pNMR (CIF bond pathways) peaks as CSA, Quad & J-coupling sites.

The pNMR simulation and fit (:mod:`.pnmr_simulator`, :mod:`.pnmr_fit`)
compute **isotropic** peaks only -- one shift per environment (group).
These functions turn those peaks into site dicts for
:mod:`.combined_hamiltonian` (the GUI's "CSA, Quad & J-coupling"
panels), where CSA, quadrupolar and J interactions can be added to each
(explicit user request, 2026-10-12: "Export All (or selected) sites to
'Combined Hamiltonian'").

* :func:`sites_from_pnmr_fit` -- each fitted peak becomes a site: δiso
  the fitted centre, amount its fitted area (fraction of the total),
  FWHM and lineshape as fitted.
* :func:`sites_from_pnmr_table` -- a simulation's shift table, grouped
  as the fit groups environments (:func:`.pnmr_fit.group_environments`),
  each group a site: δiso its probability-weighted shift, amount its
  probability, FWHM the simulation's broadening.

Example::

    from opennmr.notebook import pnmr_export, combined_hamiltonian as ch
    dicts, notes = pnmr_export.sites_from_pnmr_fit(fit, "7Li", return_notes=True)
    sites = [ch.Site.from_dict(d) for d in dicts]
    sites[0].add_csa(aniso=-450.0, asym=0.3)     # e.g. the electron-nuclear dipolar part
    spectrum = ch.simulate_combined(sites, larmor_Hz=..., mas_Hz=...)

Every site dict has ``nucleus``, ``label``, ``delta_iso``, ``amount``,
``fwhm``, ``lineshape``, ``eta`` and ``interactions`` (empty unless
``interactions=`` adds the same ones to every site).
"""

from __future__ import annotations

import copy
import math
from typing import Iterable, Sequence

import numpy as np

_SIGMA_TO_FWHM = 2.0 * math.sqrt(2.0 * math.log(2.0))
_LINESHAPES = {"gaussian": "gaussian", "lorentzian": "lorentzian",
               "pseudo_voigt": "pseudo_voigt", "pseudovoigt": "pseudo_voigt",
               "voigt": "pseudo_voigt"}


def _spin(nucleus: str) -> float | None:
    from .utils import find_isotope_info
    try:
        return float(find_isotope_info(nucleus)["spin"])
    except (KeyError, TypeError, ValueError):
        return None


def _unique(labels: Sequence[str]) -> list[str]:
    seen: dict[str, int] = {}
    out = []
    for name in labels:
        seen[name] = seen.get(name, 0) + 1
        out.append(name if seen[name] == 1 else f"{name} ({seen[name]})")
    return out


def _label(members, shift: float, k: int) -> str:
    site = None
    if members:
        best = max(members, key=lambda m: (m.get("fraction_in_group") or 0.0, m.get("p") or 0.0))
        site = best.get("site")
    return f"{site} {shift:.4g} ppm" if site else f"pNMR {k + 1} ({shift:.4g} ppm)"


def _with_interactions(site: dict, interactions) -> dict:
    if interactions:
        site["interactions"] = copy.deepcopy(list(interactions))
    return site


def _common_notes(nucleus: str, source: str) -> list[str]:
    notes = [
        "Each site has its isotropic shift only. Add CSA (e.g. the electron-nuclear dipolar "
        "anisotropy, often hundreds of ppm in paramagnets), quadrupolar or J interactions to "
        "it in CSA, Quad & J-coupling."]
    spin = _spin(nucleus)
    if spin is not None and spin > 0.5:
        if source == "fit":
            notes.append(
                f"{nucleus} is quadrupolar (I = {spin:g}). δiso here is the fitted peak centre; "
                "with second-order quadrupolar broadening the centre of the central transition "
                "sits below δiso by the quadrupolar-induced shift, so raise δiso once C_Q is "
                "added (or fit it).")
        else:
            notes.append(
                f"{nucleus} is quadrupolar (I = {spin:g}). δiso here is the bond-pathway "
                "isotropic shift; adding C_Q will move the central transition's centre below it "
                "(the quadrupolar-induced shift), as it should.")
    return notes


def _selected(n: int, which) -> list[int]:
    if which is None:
        return list(range(n))
    out = sorted({int(i) for i in which})
    bad = [i for i in out if not 0 <= i < n]
    if bad:
        raise ValueError(f"No peak(s) {bad}: there are {n} (0..{n - 1}).")
    if not out:
        raise ValueError("No peaks selected.")
    return out


# ---------------------------------------------------------------------------
# From a fit
# ---------------------------------------------------------------------------

def _fit_peaks(fit) -> list[dict]:
    peaks = fit.get("peaks") if isinstance(fit, dict) else None
    if isinstance(peaks, list) and (not peaks or isinstance(peaks[0], dict)):
        return peaks
    from .pnmr_fit import summarise_peaks
    return summarise_peaks(fit)


def _fit_nucleus(fit) -> str | None:
    spectrum = fit.get("spectrum") if isinstance(fit, dict) else None
    nucleus = getattr(getattr(spectrum, "metadata", None), "nucleus", None)
    if nucleus:
        return str(nucleus)
    results = fit.get("results") if isinstance(fit, dict) else None
    if isinstance(results, dict) and results.get("nuclide"):
        return str(results["nuclide"])
    return None


def sites_from_pnmr_fit(fit, nucleus: str | None = None, which: Iterable[int] | None = None,
                        interactions=None, return_notes: bool = False):
    """Fitted pNMR peaks -> site dicts (see the module docstring).

    ``fit``: :func:`.pnmr_fit.fit_predicted_spectrum`'s or
    :func:`.pnmr_fit.refit_merged_groups`' output, or a dict with
    ``"peaks"`` from :func:`.pnmr_fit.summarise_peaks` (the GUI's). ``which``:
    peak indices (default all). ``nucleus``: the observed nuclide (taken
    from the fitted spectrum when it is there). Amounts are the fitted
    areas as fractions of the *whole* fit, so a selection keeps its
    proportions to the rest."""
    peaks = _fit_peaks(fit)
    if not peaks:
        raise ValueError("The fit has no peaks.")
    nucleus = nucleus or _fit_nucleus(fit)
    if not nucleus:
        raise ValueError("Give the nucleus (e.g. '7Li'): the fit does not record it.")
    chosen = _selected(len(peaks), which)
    sites = []
    for k in chosen:
        p = peaks[k]
        shift = float(p["fitted_shift"])
        pct = p.get("integral_pct")
        amount = float(pct) / 100.0 if pct is not None and np.isfinite(pct) else 1.0
        lineshape = _LINESHAPES.get(str(p.get("type", "gaussian")).lower(), "gaussian")
        eta = p.get("eta")
        sites.append(_with_interactions({
            "nucleus": nucleus, "label": _label(p.get("members"), shift, k),
            "delta_iso": shift, "amount": max(amount, 1e-6),
            "fwhm": float(p.get("fwhm", 1.0)), "lineshape": lineshape,
            "eta": float(eta) if eta is not None and np.isfinite(eta) else 0.5,
            "interactions": [],
        }, interactions))
    for site, name in zip(sites, _unique([s["label"] for s in sites])):
        site["label"] = name
    if not return_notes:
        return sites
    notes = _common_notes(nucleus, "fit")
    notes.append(
        "Amounts are the fitted isotropic peak areas (fraction of the whole fit). Once an "
        "anisotropy spreads a site into spinning sidebands, the centreband alone holds less "
        "than its amount: refit the amounts over a region that includes the sidebands.")
    if len(chosen) < len(peaks):
        left = sum(float(peaks[k].get("integral_pct") or 0.0) for k in range(len(peaks))
                   if k not in chosen)
        notes.append(f"{len(chosen)} of {len(peaks)} peaks exported; the other "
                     f"{len(peaks) - len(chosen)} hold {left:.3g} % of the fitted area.")
    return sites, notes


# ---------------------------------------------------------------------------
# From a simulation's shift table
# ---------------------------------------------------------------------------

def _with_labels(table, site_labels):
    """The simulation's shift table with the ``label`` (composition) and
    ``Site_label`` columns :func:`.pnmr_fit.group_environments` reads."""
    table = table.copy()
    fixed = {"Site", "Config", "Total_shift", "p", "label", "composition", "Site_label"}
    count_cols = [c for c in table.columns if c not in fixed and "_" in str(c)]
    if "label" not in table.columns:
        def _row_label(row):
            parts: dict[str, list[str]] = {}
            for col in count_cols:
                n = row[col]
                if n and np.isfinite(n):
                    shell, species = str(col).rsplit("_", 1)
                    parts.setdefault(shell, []).append(f"{int(n)}{species}")
            return " ".join(f"{s}:" + "+".join(v) for s, v in parts.items()) or "bare"
        table["label"] = table.apply(_row_label, axis=1)
    if "Site_label" not in table.columns:
        def _site(i):
            if site_labels is None:
                return f"site {i}"
            try:
                return site_labels[int(i)] if not isinstance(site_labels, dict) \
                    else site_labels.get(int(i), f"site {i}")
            except (IndexError, TypeError, ValueError):
                return f"site {i}"
        table["Site_label"] = [_site(i) for i in table["Site"]] if "Site" in table.columns \
            else None
    return table


def sites_from_pnmr_table(table, nucleus: str | None = None, broadening_sigma: float | None = None,
                          tolerance: float | None = None, which_rows: Iterable[int] | None = None,
                          min_fraction: float = 0.01, max_sites: int = 64, site_labels=None,
                          method: str = "probability", interactions=None,
                          return_notes: bool = False):
    """A pNMR simulation's shift table (``generate_spectrum_dict``'s, or
    :func:`.pnmr_fit.predict_environments`') -> site dicts, one per group
    of environments within ``tolerance`` ppm (as the fit groups them; the
    default is the smallest bond-pathway shift, from the table's recorded
    parameters).

    ``which_rows``: only these rows (positions in ``table``) -- e.g. the
    configurations selected in the GUI -- grouped among themselves.
    ``min_fraction``: groups holding less than this fraction of the
    exported probability are left out (said in the notes, with how much
    they held), and at most ``max_sites`` of the most probable are kept:
    a site list of thousands of tiny groups is no use to add tensors to.
    ``broadening_sigma`` (ppm, Gaussian σ as the simulation's
    ``broadening``) sets each FWHM; it and ``nucleus`` are read from the
    table's recorded parameters when not given."""
    from .pnmr_fit import default_tolerance, group_environments

    params = dict(getattr(table, "attrs", {}).get("parameters", {}) or {})
    nucleus = nucleus or params.get("nuclide")
    if not nucleus:
        raise ValueError("Give the nucleus (e.g. '7Li'): the table does not record it.")
    if broadening_sigma is None:
        broadening_sigma = params.get("broadening_sigma_ppm", 1.0)
    fwhm = float(broadening_sigma) * _SIGMA_TO_FWHM
    if which_rows is not None:
        rows = sorted({int(i) for i in which_rows})
        if not rows:
            raise ValueError("No rows selected.")
        table = table.iloc[rows]
    if len(table) == 0:
        raise ValueError("The shift table is empty.")
    if tolerance is None:
        try:
            tolerance = default_tolerance(params.get("shift_dict") or {})
        except ValueError:
            tolerance = max(fwhm / 2.0, 0.1)
    groups, tolerance = group_environments(_with_labels(table, site_labels), tolerance=tolerance,
                                           method=method)
    total = float(groups["group_p"].sum()) or 1.0
    order = np.argsort(-groups["group_p"].to_numpy(), kind="stable")
    keep = [int(i) for i in order if groups["group_p"].iloc[int(i)] / total >= min_fraction]
    keep = keep[:max_sites] or [int(order[0])]
    keep.sort(key=lambda i: -float(groups["group_shift"].iloc[i]))      # high ppm first
    sites = []
    for k, i in enumerate(keep):
        g = groups.iloc[i]
        shift = float(g["group_shift"])
        sites.append(_with_interactions({
            "nucleus": nucleus, "label": _label(g["members"], shift, k),
            "delta_iso": shift, "amount": float(g["group_p"]) / total,
            "fwhm": fwhm, "lineshape": "gaussian", "eta": 0.5, "interactions": [],
        }, interactions))
    for site, name in zip(sites, _unique([s["label"] for s in sites])):
        site["label"] = name
    if not return_notes:
        return sites
    notes = _common_notes(nucleus, "simulation")
    notes.append(
        f"{len(table)} configuration(s) grouped within {tolerance:.3g} ppm into {len(groups)} "
        f"group(s); amounts are each group's probability (fraction of what was exported); FWHM "
        f"{fwhm:.3g} ppm from the simulation's broadening (σ {float(broadening_sigma):.3g} ppm).")
    dropped = len(groups) - len(keep)
    if dropped:
        held = 100.0 * (1.0 - sum(float(groups["group_p"].iloc[i]) for i in keep) / total)
        notes.append(f"{dropped} group(s) below {100 * min_fraction:.3g} % each (or beyond "
                     f"{max_sites} sites) left out; together they held {held:.3g} % of the "
                     "intensity.")
    if which_rows is not None:
        notes.append("Only the selected configurations were exported.")
    return sites, notes


def sites_python(sites: Sequence[dict]) -> str:
    """Notebook code that rebuilds ``sites`` as Site objects."""
    lines = ["from opennmr.notebook import combined_hamiltonian as ch", "sites = ["]
    for s in sites:
        d = {k: s[k] for k in ("nucleus", "label", "delta_iso", "amount", "fwhm", "lineshape",
                               "eta", "interactions") if k in s}
        lines.append(f"    ch.Site.from_dict({d!r}),")
    lines.append("]")
    lines.append("# e.g. sites[0].add_csa(aniso=..., asym=...); sites[0].add_quadrupolar(C_Q=..., "
                 "eta_Q=...)")
    return "\n".join(lines)
