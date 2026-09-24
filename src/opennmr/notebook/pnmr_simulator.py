"""
pnmr_spectrum_simulator.py
===========================

CIF-based paramagnetic-NMR (pNMR) shift-prediction pipeline, extended so that
its output is a dictionary with the same shape as an experimental spectrum:

    dict_keys(['x_ppm', 'x_Hz', 'real_spectrum', 'normalised_real_spectrum',
               'imag_spectrum', 'normalised_imag_spectrum',
               'nuclide', 'BF1', 'SFO1', 'path'])

This lets a CIF-predicted spectrum be plotted/compared directly against
experimental spectra that already come in that format.

Pipeline
--------
1. read_cif_with_occupancies   - parse the CIF, insert vacancies once
2. shell_distance_histogram    - (diagnostic) inspect raw neighbor distances
3. get_neighbors_by_shell      - bin neighbors of the site of interest into
                                  user-defined shells (e.g. NN, NNN, ...)
4. enumerate_local_environments- enumerate every disordered occupancy
                                  configuration and its probability, per site
5. build_shift_table           - turn each configuration into a predicted
                                  total shift, weighted by its probability
6. generate_spectrum_dict      - broaden the shift table into a spectrum and
                                  package it into the experimental-style dict


Referencing
-----------
`find_gamma`, `NUC_label` and `compute_BF1_SFO1` now live in nmrtools.utils
(the same helpers used by NMR1dimport / NMR1dplot in one_d.py, and by
generate_bms_spectrum_dict in bms_spectrum.py), imported from there below,
so a CIF-simulated spectrum references (BF1/SFO1) and labels its axis
identically to an imported experimental one -- from one copy of the lookup
table, not a private one. `compute_BF1_SFO1` wraps `find_gamma` to turn a
nuclide + field strength into (BF1, SFO1); pass `gamma_MHz_per_T=` to
override the table lookup entirely if needed.

(Previously this module carried its own private copies of `find_gamma`,
`NUC_label` and `compute_BF1_SFO1`, despite this same docstring already
claiming they were "ported as-is" from nmrtools.utils -- they weren't
imported, they were duplicated. Fixed here: this module now imports them,
same as bms_spectrum.py does.)

Plotting
--------
`plot_spectrum_dict`, `plot_comparison` and `plot_stacked_spectra` have
moved to nmrtools.plotting (nothing in them was CIF-specific; they work on
any dict of the shared spectrum shape, including generate_bms_spectrum_dict's
output) and are re-imported below for backwards compatibility with existing
`from nmrtools.pnmr_simulator import plot_spectrum_dict`-style code.
"""

from __future__ import annotations

import os
import re
import warnings
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .utils import find_gamma, NUC_label, compute_BF1_SFO1
from .plotting import (
    plot_spectrum_dict, plot_comparison, plot_stacked_spectra,  # noqa: F401 (re-exported)
)


class PnmrWarning(UserWarning):
    """A warning about the simulation itself (e.g. peaks outside the
    requested axis), as distinct from the CIF parser's own notes."""


# ===========================================================================
# 1. Read CIF, insert vacancies ONCE, flag sites of interest
# ===========================================================================
def read_cif_with_occupancies(filename, atom_of_interest=None, vacancy_tol=5,
                               insert_vacancies=True, verbose=False,
                               sites_of_interest=None):
    """
    Returns (structure, site_data).
    site_data[i]['occupancies'] is guaranteed to sum to 1 (Va inserted if not),
    and this insertion happens exactly once, at the source, rather than lazily
    inside neighbor traversal.

    site_data[i]['label'] is the CIF's own site label (``_atom_site_label``,
    e.g. "Na1", "C3"); every symmetry-equivalent copy of a site carries the
    label of the site it was generated from. A site shared by several
    species (one position, several CIF rows) carries the label pymatgen
    kept for it -- one of those rows' labels.

    sites_of_interest : optional list of labels. When given, only sites
        that contain `atom_of_interest` *and* carry one of these labels are
        flagged -- e.g. atom_of_interest="C", sites_of_interest=["C3", "C5"]
        builds the spectrum of C3 and C5 only. None (the default) flags
        every site containing `atom_of_interest`.
    """
    from pymatgen.io.cif import CifParser

    parser = CifParser(filename)
    structures = parser.parse_structures(primitive=False)
    if not structures:
        raise ValueError("No structures found in CIF.")
    if len(structures) > 1 and verbose:
        print(f"Note: {len(structures)} structures in CIF; using the first.")
    structure = structures[0]

    site_data = []
    for i, site in enumerate(structure):
        occ = [{"species": str(el), "probability": float(p)}
               for el, p in site.species.items()]
        site_data.append({"site_index": i, "coordinates": site.coords,
                           "occupancies": occ, "label": _site_label(site)})

    if insert_vacancies:
        for s in site_data:
            p_tot = sum(o["probability"] for o in s["occupancies"])
            if round(p_tot, vacancy_tol) != 1.0:
                if verbose:
                    print(f"Site {s['site_index']}: occ sums to {p_tot:.5f} "
                          f"-> inserting Va ({1.0 - p_tot:.5f})")
                s["occupancies"].append({"species": "Va", "probability": 1.0 - p_tot})

    if atom_of_interest is not None:
        mark_sites_of_interest(site_data, atom_of_interest, sites_of_interest)
    elif sites_of_interest:
        raise ValueError("sites_of_interest needs an atom_of_interest too.")

    return structure, site_data


def _site_label(site) -> str:
    """The CIF label of a pymatgen site. pymatgen only carries labels
    from 2023.7 on; older versions fall back to the species string, so
    selecting sites by label then just means selecting by species."""
    label = getattr(site, "label", None)
    return str(label) if label else site.species_string


def mark_sites_of_interest(site_data, atom_of_interest, sites_of_interest=None):
    """(Re)set site_data[i]['contains <atom_of_interest>'] -- the flag every
    later step keys on -- without re-reading the CIF.

    A site is flagged when one of its occupants is exactly
    `atom_of_interest` (the species string as the CIF gives it, e.g.
    "Na+" in a CIF with oxidation states, "Na" in one without) and, when
    `sites_of_interest` is given, its label is in that list. Returns the
    flag's key. Raises ValueError for a label that is not in the
    structure, or that names a site not containing `atom_of_interest`,
    rather than silently simulating fewer sites than asked for.
    """
    key = f"contains {atom_of_interest}"
    wanted = None if not sites_of_interest else {str(x) for x in sites_of_interest}
    if wanted is not None:
        labels = {s.get("label") for s in site_data}
        unknown = sorted(wanted - labels)
        if unknown:
            raise ValueError(f"No site labelled {', '.join(unknown)} in this structure "
                             f"(labels: {', '.join(sorted(l for l in labels if l))}).")
        wrong = sorted(
            lab for lab in wanted
            if not any(o["species"] == atom_of_interest
                       for s in site_data if s.get("label") == lab for o in s["occupancies"]))
        if wrong:
            raise ValueError(f"Site(s) {', '.join(wrong)} contain no {atom_of_interest}.")
    for s in site_data:
        has = any(o["species"] == atom_of_interest for o in s["occupancies"])
        s[key] = has and (wanted is None or s.get("label") in wanted)
    return key


def _is_ignored(site, ignore_species, ignore_sites) -> bool:
    """A neighbour is dropped entirely when any of its occupants is an
    ignored species, or its label is an ignored site."""
    if ignore_sites and site.get("label") in ignore_sites:
        return True
    return bool(ignore_species) and any(
        o["species"] in ignore_species for o in site["occupancies"])


# ===========================================================================
# 1b. Describing a structure before simulating it (used by the GUI's
#     step-by-step pNMR panel, equally usable from a notebook)
# ===========================================================================
_ELEMENT_RE = re.compile(r"^([A-Z][a-z]?)")

#: Default NMR isotope where an element has more than one in the table --
#: the one normally measured. Anything else falls back to the table order.
_PREFERRED_NUCLIDE = {
    "H": "1H", "Li": "7Li", "B": "11B", "N": "15N", "Mg": "25Mg", "Cl": "35Cl",
    "K": "39K", "Cu": "63Cu", "Ga": "71Ga", "Br": "81Br", "Rb": "87Rb",
    "Ag": "109Ag", "Cd": "113Cd", "In": "115In", "Sn": "119Sn", "Sb": "121Sb",
    "Te": "125Te", "Xe": "129Xe", "Ba": "137Ba", "Hg": "199Hg", "Tl": "205Tl",
    "Pb": "207Pb", "Si": "29Si", "O": "17O", "C": "13C", "S": "33S", "Ti": "47Ti",
}


def species_element(species: str) -> Optional[str]:
    """Element symbol of a CIF species string: "Na+" -> "Na", "Mn3+" ->
    "Mn", "O2-" -> "O". None for a vacancy ("Va") or anything unparsable."""
    if species == "Va":
        return None
    m = _ELEMENT_RE.match(str(species))
    return m.group(1) if m else None


def candidate_nuclides(species: str) -> List[str]:
    """NMR isotopes of `species`' element in the gyromagnetic-ratio table,
    the usually-measured one first (e.g. "Li+" -> ["7Li", "6Li"])."""
    from ..core.registry import _GAMMA_DF
    element = species_element(species)
    if element is None:
        return []
    found = [str(n) for n in _GAMMA_DF["Nuclide"]
             if re.sub(r"^\d+", "", str(n)) == element]
    preferred = _PREFERRED_NUCLIDE.get(element)
    if preferred in found:
        found.remove(preferred)
        found.insert(0, preferred)
    return found


def structure_species(site_data, include_vacancies=False) -> List[str]:
    """Every distinct species in the structure, in order of first
    appearance."""
    seen = []
    for s in site_data:
        for o in s["occupancies"]:
            sp = o["species"]
            if sp not in seen and (include_vacancies or sp != "Va"):
                seen.append(sp)
    return seen


def site_label_table(site_data) -> pd.DataFrame:
    """One row per CIF site label: its occupants and how many copies the
    symmetry expansion made (its multiplicity in the cell)."""
    rows = {}
    for s in site_data:
        label = s.get("label", "?")
        if label not in rows:
            rows[label] = {
                "Label": label,
                "Species": [o["species"] for o in s["occupancies"]],
                "Occupancy": ", ".join(f"{o['species']} {o['probability']:.3g}"
                                        for o in s["occupancies"]),
                "Multiplicity": 0,
            }
        rows[label]["Multiplicity"] += 1
    return pd.DataFrame(list(rows.values()),
                        columns=["Label", "Species", "Occupancy", "Multiplicity"])


def contributing_species(site_data, ignore_species=None, ignore_sites=None) -> List[str]:
    """Species that can contribute to a shift once `ignore_species` and
    `ignore_sites` are removed: every occupant of every site that is not
    ignored, vacancies excluded (they contribute 0 by convention)."""
    ignore_species = set(ignore_species or [])
    ignore_sites = set(ignore_sites or [])
    out = []
    for s in site_data:
        if _is_ignored(s, ignore_species, ignore_sites):
            continue
        for o in s["occupancies"]:
            if o["species"] != "Va" and o["species"] not in out:
                out.append(o["species"])
    return out


def neighbour_distance_table(structure, site_data, atom_of_interest, max_r=6.0,
                             ignore_species=None, ignore_sites=None) -> pd.DataFrame:
    """Every (site of interest, neighbour) pair within `max_r`, after
    ignored species/sites are removed -- the detailed form of
    shell_distance_histogram. Columns: Site, Site_label, Neighbour,
    Neighbour_label, Distance (Angstrom), Occupancy ({species: p}).
    Sorted by distance. Use it to choose shell boundaries, and with
    shell_composition() to see what each shell contains."""
    ignore_species = set(ignore_species or [])
    ignore_sites = set(ignore_sites or [])
    key = f"contains {atom_of_interest}"
    interest_idx = {i for i, s in enumerate(site_data) if s.get(key, False)}
    c_idx, p_idx, _, dists = structure.get_neighbor_list(r=max_r)
    rows = []
    for c, p, d in zip(c_idx, p_idx, dists):
        if c not in interest_idx or _is_ignored(site_data[p], ignore_species, ignore_sites):
            continue
        rows.append({
            "Site": int(c), "Site_label": site_data[c].get("label"),
            "Neighbour": int(p), "Neighbour_label": site_data[p].get("label"),
            "Distance": float(d),
            "Occupancy": {o["species"]: o["probability"] for o in site_data[p]["occupancies"]},
        })
    df = pd.DataFrame(rows, columns=["Site", "Site_label", "Neighbour", "Neighbour_label",
                                     "Distance", "Occupancy"])
    return df.sort_values("Distance", kind="stable").reset_index(drop=True)


def shell_composition(distance_table: pd.DataFrame, shell_bins, n_sites=None) -> pd.DataFrame:
    """Mean number of each neighbouring species per site of interest, in
    each shell -- occupancy-weighted, so a half-occupied Ni/Mn neighbour
    counts 0.5 Ni + 0.5 Mn. A species absent from a shell has no row for
    it. Columns: Shell, Species, Mean_count, Neighbours (the mean number
    of neighbour *positions* in that shell, the same on every row of a
    shell). `n_sites` defaults to the number of distinct sites of interest
    in the table."""
    validate_shell_bins(shell_bins)
    if n_sites is None:
        n_sites = distance_table["Site"].nunique() if len(distance_table) else 0
    rows = []
    for name, dmin, dmax in shell_bins:
        sel = distance_table[(distance_table["Distance"] > dmin)
                             & (distance_table["Distance"] <= dmax)]
        totals: Dict[str, float] = {}
        for occ in sel["Occupancy"]:
            for sp, p in occ.items():
                totals[sp] = totals.get(sp, 0.0) + p
        positions = len(sel) / n_sites if n_sites else 0.0
        for sp, t in totals.items():
            rows.append({"Shell": name, "Species": sp,
                         "Mean_count": t / n_sites if n_sites else 0.0,
                         "Neighbours": positions})
        if not totals:
            rows.append({"Shell": name, "Species": None, "Mean_count": 0.0, "Neighbours": 0.0})
    return pd.DataFrame(rows, columns=["Shell", "Species", "Mean_count", "Neighbours"])


def suggest_shell_bins(distances, n_shells=None, min_gap=0.2, max_r=None):
    """Shell boundaries suggested from a distance list (e.g.
    shell_distance_histogram's output): distances closer together than
    `min_gap` Angstrom form one cluster, and each boundary sits halfway
    across the gap between two clusters. Returns [(d_min, d_max), ...],
    nearest first, the first starting at 0. A cluster that reaches `max_r`
    (the search radius the distances were collected within) may be cut off
    by it, so it is not suggested. A starting point to check against the
    histogram, not a substitute for looking at it."""
    d = np.sort(np.asarray(distances, dtype=float))
    if d.size == 0:
        return []
    breaks = np.where(np.diff(d) > min_gap)[0]
    starts = np.concatenate(([0], breaks + 1))
    ends = np.concatenate((breaks, [d.size - 1]))
    clusters = [(d[a], d[b]) for a, b in zip(starts, ends)]
    out = []
    for k, (lo, hi) in enumerate(clusters):
        if k + 1 < len(clusters):
            upper = 0.5 * (hi + clusters[k + 1][0])
        elif max_r is not None and hi > max_r - min_gap:
            break
        else:
            upper = hi + 0.5 * min_gap
        lower = 0.0 if not out else out[-1][1]
        out.append((round(float(lower), 3), round(float(upper), 3)))
        if n_shells is not None and len(out) >= n_shells:
            break
    return out


#: Names a shell cannot take: they are column names / keys of the shift table.
_RESERVED_SHELL_NAMES = {"p", "Site", "Config", "Total_shift"}


def validate_shell_bins(shell_bins) -> None:
    """Raise ValueError for shell definitions that would silently give a
    wrong answer: overlapping shells (a neighbour would be assigned to
    whichever is listed first), d_min >= d_max, negative distances,
    duplicate or empty names, or a name that collides with a shift-table
    column ("p", "Site", "Config", "Total_shift")."""
    if not shell_bins:
        raise ValueError("Define at least one shell.")
    names = []
    for entry in shell_bins:
        if len(entry) != 3:
            raise ValueError(f"Shell {entry!r}: expected (name, d_min, d_max).")
        name, dmin, dmax = entry
        if not str(name).strip():
            raise ValueError("Every shell needs a name.")
        if name in _RESERVED_SHELL_NAMES:
            raise ValueError(f"'{name}' cannot be a shell name (it is a shift-table column).")
        if name in names:
            raise ValueError(f"Two shells are called '{name}'.")
        names.append(name)
        if dmin < 0 or dmax <= dmin:
            raise ValueError(f"Shell '{name}': need 0 <= d_min < d_max, got {dmin} to {dmax} Å.")
    ordered = sorted(shell_bins, key=lambda b: b[1])
    for (n1, _, hi1), (n2, lo2, _) in zip(ordered, ordered[1:]):
        if lo2 < hi1:
            raise ValueError(f"Shells '{n1}' and '{n2}' overlap ({lo2} Å < {hi1} Å): a neighbour "
                             f"in the overlap would count in only one of them.")


def site_weights(site_data, atom_of_interest, weight_by_occupancy=True) -> Dict[int, float]:
    """Weight of each site of interest in the spectrum, summing to 1.

    weight_by_occupancy=True (the default) weights each site by its
    occupancy of `atom_of_interest`: the signal from a site is
    proportional to how many nuclei are on it, so a site 30% occupied by
    Na contributes 0.3/1.0 as much as a fully occupied one (P2 Na_e/Na_f,
    Li on transition-metal sites, ...). False weights every site equally --
    the behaviour before 2026-09-24, identical whenever all sites of
    interest have the same occupancy."""
    key = f"contains {atom_of_interest}"
    raw = {}
    for i, s in enumerate(site_data):
        if not s.get(key, False):
            continue
        raw[i] = (sum(o["probability"] for o in s["occupancies"]
                      if o["species"] == atom_of_interest)
                  if weight_by_occupancy else 1.0)
    total = sum(raw.values())
    if total <= 0:
        return {}
    return {i: w / total for i, w in raw.items()}


# ===========================================================================
# 2. Diagnostic: inspect the raw distance distribution before picking bins
# ===========================================================================
def shell_distance_histogram(structure, site_data, atom_of_interest,
                              max_r=6.0, ignore_species=None, ignore_sites=None):
    """Sorted array of distances from every atom_of_interest site to every
    non-ignored neighbor within max_r. Use np.diff() on the result to find
    the gaps that should become your shell boundaries.
    neighbour_distance_table() gives the same pairs with labels and
    occupancies attached."""
    table = neighbour_distance_table(structure, site_data, atom_of_interest, max_r=max_r,
                                     ignore_species=ignore_species, ignore_sites=ignore_sites)
    return np.sort(table["Distance"].to_numpy(dtype=float))


# ===========================================================================
# 3. Build neighbor lists, binned into user-defined shells
# ===========================================================================
def get_neighbors_by_shell(structure, site_data, shell_bins, ignore_species=None,
                           ignore_sites=None):
    """
    shell_bins: list of (name, d_min, d_max), e.g.
        [("NN", 0.0, 2.65), ("NNN", 2.65, 3.15)]
    Distances outside all bins (deliberate gaps) are dropped, not misassigned.
    A neighbour is counted in a shell when d_min < d <= d_max. Overlapping
    shells are refused (see validate_shell_bins).
    ignore_sites: CIF site labels to leave out, like ignore_species.
    """
    validate_shell_bins(shell_bins)
    ignore_species = set(ignore_species or [])
    ignore_sites = set(ignore_sites or [])
    max_r = max(dmax for _, _, dmax in shell_bins)
    c_idx, p_idx, _, dists = structure.get_neighbor_list(r=max_r)

    neighbors_info = {i: {name: [] for name, _, _ in shell_bins}
                       for i in range(len(structure))}

    def assign(d):
        for name, dmin, dmax in shell_bins:
            if dmin < d <= dmax:
                return name
        return None

    for c, p, d in zip(c_idx, p_idx, dists):
        occ = site_data[p]["occupancies"]
        if _is_ignored(site_data[p], ignore_species, ignore_sites):
            continue
        shell = assign(d)
        if shell is None:
            continue
        neighbors_info[c][shell].append({"index": int(p), "species": occ,
                                          "distance": float(d)})
    return neighbors_info


# ===========================================================================
# 4. Per-shell occupancy distribution, then joint distribution across shells
# ===========================================================================
def local_environment_distribution(neighbor_list):
    dist = {(): 1.0}
    for neighbor in neighbor_list:
        opts = [(s["species"], s["probability"]) for s in neighbor["species"]]
        new_dist = defaultdict(float)
        for comp, p in dist.items():
            for sp, sp_p in opts:
                new_dist[tuple(sorted(comp + (sp,)))] += p * sp_p
        dist = new_dist
    return dict(dist)


def combine_labeled_shells(shell_dists):
    """shell_dists: {shell_name: {comp_tuple: prob}}. Order of shell_names
    is preserved (dict order), works for any number of shells."""
    names = list(shell_dists.keys())
    combined = {(): 1.0}
    for name in names:
        new_combined = {}
        for key, p1 in combined.items():
            for comp, p2 in shell_dists[name].items():
                nk = key + (comp,)
                new_combined[nk] = new_combined.get(nk, 0.0) + p1 * p2
        combined = new_combined
    return {tuple(zip(names, k)): p for k, p in combined.items()}


# ===========================================================================
# 5. Enumerate configs per site of interest
# ===========================================================================
def _environment_key(neighbors_info_i, shell_names):
    """What a site's configuration list depends on: per shell, the
    multiset of its neighbours' occupancies. Two sites with equal keys
    have identical configurations and probabilities."""
    return tuple(
        tuple(sorted(tuple(sorted((o["species"], o["probability"]) for o in n["species"]))
                     for n in neighbors_info_i[name]))
        for name in shell_names)


def _configs_from_key(key, shell_names):
    shell_dists = {}
    for name, shell in zip(shell_names, key):
        neighbours = [{"species": [{"species": sp, "probability": p} for sp, p in occ]}
                      for occ in shell]
        shell_dists[name] = local_environment_distribution(neighbours)
    joint = combine_labeled_shells(shell_dists)
    configs = []
    for env, p in joint.items():
        cfg = {"p": p}
        for shell_name, comp in env:
            cfg[shell_name] = dict(Counter(comp))
        configs.append(cfg)
    return configs


def enumerate_local_environments(site_data, neighbors_info, atom_of_interest, shell_names,
                                 progress=None):
    """{site_index: [config, ...]} for every flagged site, each config
    {"p": probability, <shell>: {species: count}}.

    Sites whose shells hold the same multiset of neighbour occupancies --
    all the symmetry-equivalent copies of a site, typically, and every
    copy in a supercell of a disordered cell -- are enumerated once and
    share the one configuration list (the same list object: treat it as
    read-only). Neighbours are taken in a canonical (sorted) order, so
    two such sites list their configurations in the same order.

    progress: optional callable(stage, done, total), called after each
        site; it may raise to abandon the run (see EnumerationCancelled).
    """
    key = f"contains {atom_of_interest}"
    interest = [i for i, site in enumerate(site_data) if site.get(key, False)]
    results = {}
    cache = {}
    for done, i in enumerate(interest, 1):
        env_key = _environment_key(neighbors_info[i], shell_names)
        if env_key not in cache:
            cache[env_key] = _configs_from_key(env_key, shell_names)
        results[i] = cache[env_key]
        if progress is not None:
            progress("Enumerating environments", done, len(interest))
    return results


# ===========================================================================
# 6. Shift table
# ===========================================================================
def build_shift_table(local_envs, shift_dict, warn_missing=True, print_table=False,
                      site_weights=None, report=None, progress=None):
    """
    shift_dict: {shell_name: {species: shift_value}}. 'Va' and unlisted
    species contribute 0 by convention.
    site_weights: {site_index: weight} summing to 1 (see site_weights()).
        None weights every site equally.
    report: where warnings about missing shift constants go -- a
        callable taking one string. Default print.
    progress: optional callable(stage, done, total), called after each
        site; it may raise to abandon the run.
    """
    report = print if report is None else report
    warned = set()
    templates = {}          # id(config list) -> rows without "Site"

    def _rows_for(configs):
        rows = []
        for config_idx, cfg in enumerate(configs):
            row = {"Config": config_idx}
            total_shift = 0.0
            for shell_name, counts in cfg.items():
                if shell_name == "p":
                    continue
                lookup = shift_dict.get(shell_name, {})
                for species, n in counts.items():
                    row[f"{shell_name}_{species}"] = n
                    if (warn_missing and species != "Va"
                            and species not in lookup and (shell_name, species) not in warned):
                        report(f"Warning: no shift constant for '{species}' in shell "
                               f"'{shell_name}' — treated as 0.0. Check this isn't a typo.")
                        warned.add((shell_name, species))
                    total_shift += n * lookup.get(species, 0.0)
            row["Total_shift"] = total_shift
            row["p"] = cfg["p"]
            rows.append(row)
        return rows

    rows = []
    for done, (site_idx, configs) in enumerate(local_envs.items(), 1):
        # Sites sharing a configuration list (see enumerate_local_
        # environments) share its rows too -- only "Site" differs.
        template = templates.get(id(configs))
        if template is None:
            template = templates[id(configs)] = _rows_for(configs)
        rows.extend({"Site": site_idx, **row} for row in template)
        if progress is not None:
            progress("Building the shift table", done, len(local_envs))

    if progress is not None:
        progress("Assembling the table", 0, 1)
    df = pd.DataFrame(rows).fillna(0)
    if len(df):
        # A fixed column order -- Site, Config, each shell's species
        # (shells in order, species alphabetical), Total_shift, p --
        # rather than whichever order the first configuration listed them.
        first = next(iter(local_envs.values()))[0]
        shell_order = [k for k in first if k != "p"]
        count_cols = [c for c in df.columns if c not in ("Site", "Config", "Total_shift", "p")]
        count_cols.sort(key=lambda c: (next((n for n, sh in enumerate(shell_order)
                                              if c.startswith(f"{sh}_")), len(shell_order)), c))
        df = df[["Site", "Config", *count_cols, "Total_shift", "p"]]

    # The configuration probabilities are normalised independently
    # for each crystallographic/site instance. Convert them to weights
    # normalised over the complete shift table so that sum(p) == 1.
    n_sites = df["Site"].nunique() if len(df) else 0
    if n_sites > 0:
        if site_weights is None:
            df["p"] /= n_sites
        else:
            missing = set(df["Site"]) - set(site_weights)
            if missing:
                raise ValueError(f"No weight given for site(s) {sorted(missing)}.")
            df["p"] *= df["Site"].map(site_weights)

    if print_table:
        with pd.option_context('display.max_rows', None,
                                'display.max_columns', None,
                                'display.precision', 3):
            print(df)

    # Sanity check: probabilities/weights should now sum to 1.
    p_sum = df["p"].sum() if len(df) else 1.0
    if not np.isclose(p_sum, 1.0, atol=1e-12):
        raise RuntimeError(
            f"Shift-table probabilities sum to {p_sum:.12g}, expected 1.0."
        )

    return df


# ===========================================================================
# 7. Referencing: nuclide -> gamma -> BF1 / SFO1, and axis labelling
# ===========================================================================
# find_gamma, NUC_label and compute_BF1_SFO1 are imported from nmrtools.utils
# at the top of this file (see module docstring) -- no longer defined here.


# ===========================================================================
# 8. Shift table -> broadened spectrum arrays
# ===========================================================================
#: Distinct shifts broadened per block in _broadened_spectrum -- bounds
#: its temporary (block x n_points) array to ~64 MB at 8192 points.
_BROADEN_BLOCK = 1024


def _broadened_spectrum(shift_table: pd.DataFrame, broadening: float,
                         n_points: int, xlims: Optional[Sequence[float]],
                         report=None, progress=None,
                         ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gaussian-broadened stick spectrum on a ppm axis. Each stick's height is
    its row's p, which build_shift_table has already normalised over the
    whole table (sum(p) == 1, per-site weights included), so the sticks
    are used as they are. (Until 2026-09-24 the result was divided by the
    number of sites a second time: the normalised spectrum was unaffected,
    but real_spectrum's absolute scale depended on how many sites the
    cell held.)

    Rows with exactly the same Total_shift are summed first -- many
    configurations share a shift -- and the distinct shifts are then
    broadened in vectorised blocks. Exact (no truncated Gaussian tails, no
    binning): the same sum as one Gaussian per row, in a different order.

    report: callable taking a string for the peaks-outside-xlims notes;
        default warnings.warn(..., PnmrWarning).
    progress: optional callable(stage, done, total), called per block.
    """
    def _note(message):
        if report is None:
            warnings.warn(message, PnmrWarning)
        else:
            report(message)

    shifts, inverse = np.unique(shift_table["Total_shift"].to_numpy(dtype=float),
                                return_inverse=True)
    weights = np.bincount(inverse.ravel(), weights=shift_table["p"].to_numpy(dtype=float),
                          minlength=shifts.size)

    if xlims is not None:
        x_min, x_max = min(xlims), max(xlims)
        if x_min > shifts.min() - 5 * broadening:
            _note("Some peaks lie outside the specified xlims (use a smaller minimum).")
        if x_max < shifts.max() + 5 * broadening:
            _note("Some peaks lie outside the specified xlims (use a larger maximum).")
    else:
        x_min = shifts.min() - 5 * broadening
        x_max = shifts.max() + 5 * broadening

    x_ppm = np.linspace(x_min, x_max, n_points)
    y = np.zeros_like(x_ppm)
    n_blocks = max(1, -(-shifts.size // _BROADEN_BLOCK))
    if progress is not None:
        progress("Broadening", 0, n_blocks)
    for b in range(n_blocks):
        sl = slice(b * _BROADEN_BLOCK, (b + 1) * _BROADEN_BLOCK)
        z = (x_ppm[None, :] - shifts[sl, None]) / broadening
        y += weights[sl] @ np.exp(-0.5 * z * z)
        if progress is not None:
            progress("Broadening", b + 1, n_blocks)

    return x_ppm, y


def _normalise(arr: np.ndarray) -> np.ndarray:
    peak = np.max(np.abs(arr)) if arr.size else 0.0
    if peak == 0:
        return np.zeros_like(arr)
    return arr / peak


def _resolve_output_dir(output_dir: Optional[str], cif_filename: str,
                        create: bool = True) -> str:
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(cif_filename)),
                                   "pnmr_simulation")
    if create:
        os.makedirs(output_dir, exist_ok=True)
    return output_dir


# ===========================================================================
# 9. Top-level driver: CIF -> experimental-style spectrum dict
# ===========================================================================
class EnumerationCancelled(Exception):
    """Raise from a `progress` callback to abandon a run cleanly."""


class TooManyConfigurations(ValueError):
    """The shift table would exceed `max_configurations` rows. Carries the
    estimate (see estimate_configurations) so a caller can ask whether to
    go ahead anyway."""

    def __init__(self, estimate: Dict, limit: int):
        self.estimate = estimate
        self.limit = limit
        super().__init__(
            f"This would build a shift table of {estimate['n_rows']:,} configurations "
            f"(~{estimate['est_bytes'] / 1e9:.2g} GB), above the limit of {limit:,}. "
            f"Narrow the shells or the sites, or raise max_configurations.")


def _neighbours_for(structure, site_data, atom_of_interest, shell_bins, ignore_species,
                    ignore_sites, sites_of_interest):
    mark_sites_of_interest(site_data, atom_of_interest, sites_of_interest)
    return get_neighbors_by_shell(structure, site_data, shell_bins,
                                  ignore_species=ignore_species, ignore_sites=ignore_sites)


def _estimate_from_neighbours(site_data, neighbors_info, atom_of_interest, shell_bins) -> Dict:
    shell_names = [name for name, _, _ in shell_bins]
    key = f"contains {atom_of_interest}"
    interest = [i for i, s in enumerate(site_data) if s.get(key, False)]
    per_env, n_rows, columns = {}, 0, set()
    for i in interest:
        env_key = _environment_key(neighbors_info[i], shell_names)
        if env_key not in per_env:
            count = 1
            for name, shell in zip(shell_names, env_key):
                dist = local_environment_distribution(
                    [{"species": [{"species": sp, "probability": p} for sp, p in occ]}
                     for occ in shell])
                count *= len(dist)
                for comp in dist:
                    columns.update(f"{name}_{sp}" for sp in comp)
            per_env[env_key] = count
        n_rows += per_env[env_key]
    n_cols = len(columns) + 4            # + Site, Config, Total_shift, p
    return {"n_sites": len(interest), "n_distinct_environments": len(per_env),
            "n_rows": int(n_rows), "n_columns": n_cols,
            # 8-byte columns, plus pandas' own overhead (~2x while building)
            "est_bytes": int(n_rows * n_cols * 8 * 2)}


def estimate_configurations(cif_filename, atom_of_interest, shell_bins, ignore_species=None,
                            ignore_sites=None, sites_of_interest=None, vacancy_tol=5,
                            structure=None, site_data=None) -> Dict:
    """How big a run would be, without enumerating it: the number of
    sites of interest, how many distinct environments they have (sites
    with the same neighbour occupancies are enumerated once), the number
    of shift-table rows and a rough memory estimate. Cheap -- it counts
    each shell's compositions, not their product."""
    if structure is None or site_data is None:
        structure, site_data = read_cif_with_occupancies(
            cif_filename, vacancy_tol=vacancy_tol, insert_vacancies=True)
    neighbors_info = _neighbours_for(structure, site_data, atom_of_interest, shell_bins,
                                     ignore_species, ignore_sites, sites_of_interest)
    return _estimate_from_neighbours(site_data, neighbors_info, atom_of_interest, shell_bins)


def shift_table_from_structure(structure, site_data, atom_of_interest, shell_bins,
                               shift_dict, ignore_species=None, ignore_sites=None,
                               sites_of_interest=None, weight_by_occupancy=True,
                               warn_missing_shifts=True, progress=None, report=None,
                               max_configurations=None) -> pd.DataFrame:
    """Steps 3-6 on an already-read structure: flag the sites of
    interest, bin their neighbours into shells, enumerate every
    occupancy configuration and turn it into a weighted shift table.
    Shared by generate_spectrum_dict and pnmr_fit.predict_environments
    so the two can never weight or filter differently.

    max_configurations: refuse (TooManyConfigurations) before enumerating
        if the table would have more rows than this. None: no limit."""
    neighbors_info = _neighbours_for(structure, site_data, atom_of_interest, shell_bins,
                                     ignore_species, ignore_sites, sites_of_interest)
    if max_configurations is not None:
        estimate = _estimate_from_neighbours(site_data, neighbors_info, atom_of_interest,
                                             shell_bins)
        if estimate["n_rows"] > max_configurations:
            raise TooManyConfigurations(estimate, max_configurations)
    shell_names = [name for name, _, _ in shell_bins]
    local_envs = enumerate_local_environments(
        site_data, neighbors_info, atom_of_interest, shell_names, progress=progress,
    )
    if not local_envs:
        raise ValueError(
            f"No sites containing '{atom_of_interest}' were found"
            + (f" among {', '.join(sites_of_interest)}" if sites_of_interest else "")
            + " - check atom_of_interest and sites_of_interest.")
    weights = site_weights(site_data, atom_of_interest, weight_by_occupancy)
    return build_shift_table(
        local_envs, shift_dict, warn_missing=warn_missing_shifts, print_table=False,
        site_weights=weights, report=report, progress=progress,
    )


#: Identifies a saved simulation-parameters JSON.
PARAMETERS_FORMAT = "opennmr.pnmr_simulation_parameters"
PARAMETERS_VERSION = 1

#: generate_spectrum_dict arguments a parameters file records, and what
#: load_simulation_parameters gives back (with the file's own names where
#: they differ: "broadening_sigma_ppm" -> broadening).
_RECORDED_ARGUMENTS = (
    "cif_filename", "atom_of_interest", "sites_of_interest", "ignore_species",
    "ignore_sites", "shell_bins", "shift_dict", "nuclide", "B0_tesla", "offset_ppm",
    "gamma_MHz_per_T", "broadening", "n_points", "xlims", "weight_by_occupancy",
)


def simulation_parameters(cif_filename, atom_of_interest, shell_bins, shift_dict, nuclide,
                          B0_tesla, ignore_species=None, offset_ppm=0.0,
                          gamma_MHz_per_T=None, broadening=2.0, n_points=8192, xlims=None,
                          sites_of_interest=None, ignore_sites=None,
                          weight_by_occupancy=True, output_dir=None, **_ignored) -> Dict:
    """The JSON-ready record of a simulation's inputs -- what
    generate_spectrum_dict writes beside its shift table, and what
    load_simulation_parameters reads back. Accepts generate_spectrum_dict's
    own keyword arguments (others are ignored), so it can also save a
    set-up that has not been run yet."""
    validate_shell_bins(shell_bins)
    return {
        "format": PARAMETERS_FORMAT, "version": PARAMETERS_VERSION,
        "cif_filename": os.path.abspath(cif_filename),
        "atom_of_interest": atom_of_interest,
        "sites_of_interest": list(sites_of_interest) if sites_of_interest else None,
        "ignore_species": list(ignore_species or []),
        "ignore_sites": list(ignore_sites or []),
        "shell_bins": [[n, float(a), float(b)] for n, a, b in shell_bins],
        "shift_dict": {sh: {sp: float(v) for sp, v in d.items()} for sh, d in shift_dict.items()},
        "nuclide": nuclide, "B0_tesla": float(B0_tesla), "offset_ppm": float(offset_ppm),
        "gamma_MHz_per_T": None if gamma_MHz_per_T is None else float(gamma_MHz_per_T),
        "broadening_sigma_ppm": float(broadening),
        "n_points": int(n_points),
        "xlims": [float(x) for x in xlims] if xlims is not None else None,
        "weight_by_occupancy": bool(weight_by_occupancy),
        "output_dir": None if output_dir is None else os.path.abspath(output_dir),
    }


def save_simulation_parameters(path, **kwargs) -> Dict:
    """Write simulation_parameters(**kwargs) to `path` as JSON; returns it."""
    import json
    params = simulation_parameters(**kwargs)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(params, fh, indent=2)
    return params


def _find_cif(recorded, json_path):
    """The recorded CIF if it exists; else a file of the same name next to
    the JSON or one folder up (the default output folder is
    <cif folder>/pnmr_simulation) -- so a project folder that has been
    moved or copied to another computer still loads."""
    if recorded and os.path.isfile(recorded):
        return recorded
    name = os.path.basename(str(recorded or "").replace("\\", "/"))
    if name:
        here = os.path.dirname(os.path.abspath(json_path))
        for folder in (here, os.path.dirname(here)):
            candidate = os.path.join(folder, name)
            if os.path.isfile(candidate):
                return candidate
    return None


def load_simulation_parameters(path, cif_filename=None) -> Dict:
    """Read a parameters JSON (written by generate_spectrum_dict, the GUI's
    Save table as / Save settings, or save_simulation_parameters) and
    return generate_spectrum_dict's keyword arguments, so that

        nmr.generate_spectrum_dict(**nmr.load_simulation_parameters(path))

    reruns it. output_dir and save_shift_table are not included: where to
    write a rerun is the caller's choice (the file records the original
    folder as "output_dir"; pass it explicitly to overwrite that run).

    cif_filename overrides the recorded CIF. Otherwise the recorded path is
    used if it exists, then a file of the same name beside the JSON or one
    folder up; FileNotFoundError if none is found."""
    import json
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    if not isinstance(raw, dict):
        raise ValueError(f"{os.path.basename(path)} is not a pNMR parameters file.")
    fmt = raw.get("format")
    if fmt is not None and fmt != PARAMETERS_FORMAT:
        raise ValueError(f"{os.path.basename(path)} is a '{fmt}' file, not pNMR simulation "
                         f"parameters.")
    if raw.get("version", 1) > PARAMETERS_VERSION:
        raise ValueError(f"{os.path.basename(path)} was written by a newer openNMR "
                         f"(version {raw['version']}); update to read it.")
    raw = dict(raw)
    if "broadening_sigma_ppm" in raw:
        raw["broadening"] = raw.pop("broadening_sigma_ppm")
    missing = [k for k in ("cif_filename", "atom_of_interest", "shell_bins", "shift_dict",
                           "nuclide", "B0_tesla") if k not in raw]
    if missing:
        raise ValueError(f"{os.path.basename(path)} is missing {', '.join(missing)} -- "
                         f"not a pNMR parameters file?")
    kwargs = {k: raw[k] for k in _RECORDED_ARGUMENTS if k in raw}
    kwargs["shell_bins"] = [(str(n), float(a), float(b)) for n, a, b in kwargs["shell_bins"]]
    validate_shell_bins(kwargs["shell_bins"])
    if cif_filename is not None:
        kwargs["cif_filename"] = cif_filename
    else:
        found = _find_cif(raw["cif_filename"], path)
        if found is None:
            raise FileNotFoundError(
                f"The CIF this simulation used, {raw['cif_filename']}, was not found (nor a "
                f"file of that name beside {os.path.basename(path)}). Pass cif_filename=.")
        kwargs["cif_filename"] = found
    return kwargs


def generate_spectrum_dict(
    cif_filename: str,
    atom_of_interest: str,
    shell_bins: List[Tuple[str, float, float]],
    shift_dict: Dict[str, Dict[str, float]],
    nuclide: str,
    B0_tesla: float,
    ignore_species: Optional[Sequence[str]] = None,
    offset_ppm: float = 0.0,
    gamma_MHz_per_T: Optional[float] = None,
    broadening: float = 2.0,
    n_points: int = 8192,
    xlims: Optional[Sequence[float]] = None,
    output_dir: Optional[str] = None,
    save_shift_table: bool = True,
    vacancy_tol: int = 5,
    warn_missing_shifts: bool = True,
    verbose: bool = False,
    sites_of_interest: Optional[Sequence[str]] = None,
    ignore_sites: Optional[Sequence[str]] = None,
    weight_by_occupancy: bool = True,
    save_parameters: bool = True,
    progress=None,
    report=None,
    max_configurations: Optional[int] = None,
) -> Tuple[Dict, pd.DataFrame]:
    """
    Run the full CIF -> predicted-spectrum pipeline and package the result
    in the same dictionary shape as an experimental spectrum:

        {'x_ppm', 'x_Hz', 'real_spectrum', 'normalised_real_spectrum',
         'imag_spectrum', 'normalised_imag_spectrum',
         'nuclide', 'BF1', 'SFO1', 'path'}

    Parameters
    ----------
    cif_filename : path to the CIF to simulate.
    atom_of_interest : species string of the crystallographic site to build
        the spectrum for, exactly as the CIF gives it (e.g. "Li", or "Li+"
        in a CIF carrying oxidation states). This is a *structural* label
        matched against the CIF occupancies, independent of `nuclide` below.
    shell_bins : [(name, d_min, d_max), ...] coordination shells, e.g.
        [("NN", 0.0, 3.10), ("NNN", 3.50, 4.20)]. A neighbour at distance d
        belongs to a shell when d_min < d <= d_max; shells may not overlap.
    shift_dict : {shell_name: {species: shift_ppm}} contribution per
        neighbor species per shell. Species not listed (and 'Va') contribute 0.
    nuclide : NMR-isotope label used for referencing, e.g. "7Li", "27Al".
        Independent of `atom_of_interest` so you can simulate, say, the 6Li
        or 7Li spectrum of the same Li sites.
    B0_tesla : magnetic field strength of the (simulated) spectrometer, used
        with the nuclide's gyromagnetic ratio to compute BF1/SFO1.
    ignore_species : neighbor species to exclude entirely from the shell
        search (e.g. ["O", "O2-"] to skip the anion sublattice).
    offset_ppm : carrier offset from BF1, in ppm (0 => SFO1 == BF1).
    gamma_MHz_per_T : override the built-in gyromagnetic-ratio lookup.
    broadening : Gaussian sigma (standard deviation, NOT FWHM), in ppm,
        applied to each stick. FWHM = 2.3548 * sigma.
    n_points : number of points on the ppm axis.
    xlims : (min, max) in ppm to force the axis range; default spans the
        shift distribution +/- 5*broadening.
    output_dir : directory used as the returned 'path' and (if enabled) to
        save the shift table. Defaults to <cif directory>/pnmr_simulation.
    save_shift_table : write the shift table to
        <output_dir>/<nuclide>_shift_table.csv.
    vacancy_tol : decimal places used when deciding whether site occupancies
        already sum to 1 (see read_cif_with_occupancies).
    warn_missing_shifts : print a warning for neighbor species with no entry
        in shift_dict (treated as 0 contribution).
    verbose : pass through to read_cif_with_occupancies.
    sites_of_interest : CIF site labels to restrict the spectrum to (e.g.
        ["C3", "C5", "C7"]); None uses every site containing
        atom_of_interest.
    ignore_sites : CIF site labels to exclude from the shell search, like
        ignore_species but for individual sites.
    weight_by_occupancy : weight each site of interest by its occupancy of
        atom_of_interest (see site_weights). False gives every site equal
        weight, as before 2026-09-24; the two agree whenever every site of
        interest has the same occupancy.
    save_parameters : with save_shift_table, also write every input above
        to <output_dir>/<nuclide>_simulation_parameters.json, so a saved
        table always says how it was made. load_simulation_parameters reads
        it back.
    progress : optional callable(stage, done, total) called as the run
        proceeds (per site while enumerating and tabulating, per block while
        broadening). It may raise -- EnumerationCancelled, say -- to abandon
        the run; nothing is written to disk before the run completes.
    report : optional callable taking a string, for the run's warnings
        (missing shift constants, peaks outside xlims) instead of print /
        warnings.warn -- how the GUI collects them from a worker thread.
    max_configurations : refuse with TooManyConfigurations, before
        enumerating, if the shift table would exceed this many rows. None
        (the default): no limit. See estimate_configurations.

    Returns
    -------
    (spectrum_dict, shift_table) : the experimental-style dict, and the
        underlying per-site/per-configuration shift table (for inspection;
        it's also written to disk when save_shift_table=True). The inputs
        are recorded in shift_table.attrs["parameters"].
    """
    if not broadening > 0:
        raise ValueError(f"broadening (Gaussian sigma) must be > 0 ppm, got {broadening}.")
    if B0_tesla is None or not B0_tesla > 0:
        raise ValueError(f"B0_tesla must be > 0, got {B0_tesla}.")
    if xlims is not None and float(min(xlims)) == float(max(xlims)):
        raise ValueError("xlims must span a non-zero range.")
    # Referencing first: an unknown nuclide should fail before a long run.
    BF1, SFO1, nuclide = compute_BF1_SFO1(nuclide, B0_tesla, offset_ppm, gamma_MHz_per_T)

    if progress is not None:
        progress("Reading the CIF", 0, 1)
    structure, site_data = read_cif_with_occupancies(
        cif_filename, atom_of_interest=None,
        vacancy_tol=vacancy_tol, insert_vacancies=True, verbose=verbose,
    )
    shift_table = shift_table_from_structure(
        structure, site_data, atom_of_interest, shell_bins, shift_dict,
        ignore_species=ignore_species, ignore_sites=ignore_sites,
        sites_of_interest=sites_of_interest, weight_by_occupancy=weight_by_occupancy,
        warn_missing_shifts=warn_missing_shifts, progress=progress, report=report,
        max_configurations=max_configurations,
    )

    x_ppm, real_spectrum = _broadened_spectrum(shift_table, broadening, n_points, xlims,
                                               report=report, progress=progress)
    imag_spectrum = np.zeros_like(real_spectrum)
    x_Hz = x_ppm * BF1

    normalised_real_spectrum = _normalise(real_spectrum)
    normalised_imag_spectrum = imag_spectrum.copy()  # all-zero; normalising is a no-op

    path = _resolve_output_dir(output_dir, cif_filename, create=save_shift_table)
    parameters = simulation_parameters(
        cif_filename, atom_of_interest, shell_bins, shift_dict, nuclide, B0_tesla,
        ignore_species=ignore_species, offset_ppm=offset_ppm, gamma_MHz_per_T=gamma_MHz_per_T,
        broadening=broadening, n_points=n_points, xlims=xlims,
        sites_of_interest=sites_of_interest, ignore_sites=ignore_sites,
        weight_by_occupancy=weight_by_occupancy, output_dir=path)
    parameters.update({
        "BF1_MHz": float(BF1), "SFO1_MHz": float(SFO1),
        "n_sites": int(shift_table["Site"].nunique()),
        "n_configurations": int(len(shift_table)),
    })
    shift_table.attrs["parameters"] = parameters

    if save_shift_table:
        shift_table.to_csv(os.path.join(path, f"{nuclide}_shift_table.csv"), index=False)
        if save_parameters:
            import json
            with open(os.path.join(path, f"{nuclide}_simulation_parameters.json"), "w",
                      encoding="utf-8") as fh:
                json.dump(parameters, fh, indent=2)

    spectrum_dict = {
        "x_ppm": x_ppm,
        "x_Hz": x_Hz,
        "real_spectrum": real_spectrum,
        "normalised_real_spectrum": normalised_real_spectrum,
        "imag_spectrum": imag_spectrum,
        "normalised_imag_spectrum": normalised_imag_spectrum,
        "nuclide": nuclide,
        "BF1": BF1,
        "SFO1": SFO1,
        "path": path,
    }
    return spectrum_dict, shift_table
