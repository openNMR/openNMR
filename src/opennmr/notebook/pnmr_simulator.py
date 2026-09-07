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


# ===========================================================================
# 1. Read CIF, insert vacancies ONCE, flag sites of interest
# ===========================================================================
def read_cif_with_occupancies(filename, atom_of_interest=None, vacancy_tol=5,
                               insert_vacancies=True, verbose=False):
    """
    Returns (structure, site_data).
    site_data[i]['occupancies'] is guaranteed to sum to 1 (Va inserted if not),
    and this insertion happens exactly once, at the source, rather than lazily
    inside neighbor traversal.
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
                           "occupancies": occ})

    if insert_vacancies:
        for s in site_data:
            p_tot = sum(o["probability"] for o in s["occupancies"])
            if round(p_tot, vacancy_tol) != 1.0:
                if verbose:
                    print(f"Site {s['site_index']}: occ sums to {p_tot:.5f} "
                          f"-> inserting Va ({1.0 - p_tot:.5f})")
                s["occupancies"].append({"species": "Va", "probability": 1.0 - p_tot})

    if atom_of_interest is not None:
        for s in site_data:
            s[f"contains {atom_of_interest}"] = any(
                o["species"] == atom_of_interest for o in s["occupancies"])

    return structure, site_data


# ===========================================================================
# 2. Diagnostic: inspect the raw distance distribution before picking bins
# ===========================================================================
def shell_distance_histogram(structure, site_data, atom_of_interest,
                              max_r=6.0, ignore_species=None):
    """Sorted array of distances from every atom_of_interest site to every
    non-ignored neighbor within max_r. Use np.diff() on the result to find
    the gaps that should become your shell boundaries."""
    ignore_species = set(ignore_species or [])
    key = f"contains {atom_of_interest}"
    interest_idx = {i for i, s in enumerate(site_data) if s.get(key, False)}

    c_idx, p_idx, _, dists = structure.get_neighbor_list(r=max_r)
    out = []
    for c, p, d in zip(c_idx, p_idx, dists):
        if c not in interest_idx:
            continue
        if any(o["species"] in ignore_species for o in site_data[p]["occupancies"]):
            continue
        out.append(d)
    return np.sort(np.array(out))


# ===========================================================================
# 3. Build neighbor lists, binned into user-defined shells
# ===========================================================================
def get_neighbors_by_shell(structure, site_data, shell_bins, ignore_species=None):
    """
    shell_bins: list of (name, d_min, d_max), e.g.
        [("NN", 0.0, 2.65), ("NNN", 2.65, 3.15)]
    Distances outside all bins (deliberate gaps) are dropped, not misassigned.
    """
    ignore_species = set(ignore_species or [])
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
        if ignore_species and any(o["species"] in ignore_species for o in occ):
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
def enumerate_local_environments(site_data, neighbors_info, atom_of_interest, shell_names):
    key = f"contains {atom_of_interest}"
    results = {}
    for i, site in enumerate(site_data):
        if not site.get(key, False):
            continue
        shell_dists = {name: local_environment_distribution(neighbors_info[i][name])
                        for name in shell_names}
        joint = combine_labeled_shells(shell_dists)
        configs = []
        for env, p in joint.items():
            cfg = {"p": p}
            for shell_name, comp in env:
                cfg[shell_name] = dict(Counter(comp))
            configs.append(cfg)
        results[i] = configs
    return results


# ===========================================================================
# 6. Shift table
# ===========================================================================
def build_shift_table(local_envs, shift_dict, warn_missing=True, print_table=False):
    """
    shift_dict: {shell_name: {species: shift_value}}. 'Va' and unlisted
    species contribute 0 by convention.
    """
    rows = []
    warned = set()
    for site_idx, configs in local_envs.items():
        for config_idx, cfg in enumerate(configs):
            row = {"Site": site_idx, "Config": config_idx}
            total_shift = 0.0
            for shell_name, counts in cfg.items():
                if shell_name == "p":
                    continue
                lookup = shift_dict.get(shell_name, {})
                for species, n in counts.items():
                    row[f"{shell_name}_{species}"] = n
                    if (warn_missing and species != "Va"
                            and species not in lookup and (shell_name, species) not in warned):
                        print(f"Warning: no shift constant for '{species}' in shell "
                              f"'{shell_name}' — treated as 0.0. Check this isn't a typo.")
                        warned.add((shell_name, species))
                    total_shift += n * lookup.get(species, 0.0)
            row["Total_shift"] = total_shift
            row["p"] = cfg["p"]
            rows.append(row)

    df = pd.DataFrame(rows).fillna(0)
    
    # The configuration probabilities are normalized independently
    # for each crystallographic/site instance. Convert them to weights
    # normalized over the complete shift table so that sum(p) == 1.
    n_sites = df["Site"].nunique()
    if n_sites > 0:
        df["p"] /= n_sites

    if print_table:
        with pd.option_context('display.max_rows', None,
                                'display.max_columns', None,
                                'display.precision', 3):
            print(df)

    # Sanity check: probabilities/weights should now sum to 1.
    p_sum = df["p"].sum()
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
def _broadened_spectrum(shift_table: pd.DataFrame, broadening: float,
                         n_points: int, xlims: Optional[Sequence[float]]
                         ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gaussian-broadened stick spectrum on a ppm axis, pooled over sites with
    equal weight per site (correct Wyckoff-multiplicity weighting, since
    primitive=False keeps symmetry-equivalent sites as separate rows).
    Mirrors the original notebook's plot_spectrum() maths, but returns
    arrays instead of drawing an Axes.
    """
    shifts = shift_table["Total_shift"].to_numpy()
    weights = shift_table["p"].to_numpy()
    n_sites = shift_table["Site"].nunique()

    if xlims is not None:
        x_min, x_max = min(xlims), max(xlims)
        if x_min > shifts.min() - 5 * broadening:
            warnings.warn("Some peaks lie outside the specified xlims "
                           "(use a smaller minimum).")
        if x_max < shifts.max() + 5 * broadening:
            warnings.warn("Some peaks lie outside the specified xlims "
                           "(use a larger maximum).")
    else:
        x_min = shifts.min() - 5 * broadening
        x_max = shifts.max() + 5 * broadening

    x_ppm = np.linspace(x_min, x_max, n_points)
    y = np.zeros_like(x_ppm)
    for s, w in zip(shifts, weights):
        y += w * np.exp(-0.5 * ((x_ppm - s) / broadening) ** 2)
    y /= n_sites

    return x_ppm, y


def _normalise(arr: np.ndarray) -> np.ndarray:
    peak = np.max(np.abs(arr)) if arr.size else 0.0
    if peak == 0:
        return np.zeros_like(arr)
    return arr / peak


def _resolve_output_dir(output_dir: Optional[str], cif_filename: str) -> str:
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(cif_filename)),
                                   "pnmr_simulation")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


# ===========================================================================
# 9. Top-level driver: CIF -> experimental-style spectrum dict
# ===========================================================================
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
    atom_of_interest : element symbol of the crystallographic site to build
        the spectrum for (e.g. "Li"). This is a *structural* label matched
        against the CIF occupancies, independent of `nuclide` below.
    shell_bins : [(name, d_min, d_max), ...] coordination shells, e.g.
        [("NN", 0.0, 3.10), ("NNN", 3.50, 4.20)].
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
    broadening : Gaussian sigma, in ppm, applied to each stick.
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

    Returns
    -------
    (spectrum_dict, shift_table) : the experimental-style dict, and the
        underlying per-site/per-configuration shift table (for inspection;
        it's also written to disk when save_shift_table=True).
    """
    structure, site_data = read_cif_with_occupancies(
        cif_filename, atom_of_interest=atom_of_interest,
        vacancy_tol=vacancy_tol, insert_vacancies=True, verbose=verbose,
    )

    shell_names = [name for name, _, _ in shell_bins]
    neighbors_info = get_neighbors_by_shell(
        structure, site_data, shell_bins, ignore_species=ignore_species,
    )
    local_envs = enumerate_local_environments(
        site_data, neighbors_info, atom_of_interest, shell_names,
    )
    shift_table = build_shift_table(
        local_envs, shift_dict, warn_missing=warn_missing_shifts, print_table=False,
    )
    if shift_table.empty:
        raise ValueError(
            f"No sites containing '{atom_of_interest}' were found (or none "
            f"had any neighbors assigned to a shell) - check atom_of_interest, "
            f"shell_bins and ignore_species."
        )

    x_ppm, real_spectrum = _broadened_spectrum(shift_table, broadening, n_points, xlims)
    imag_spectrum = np.zeros_like(real_spectrum)

    BF1, SFO1, nuclide = compute_BF1_SFO1(nuclide, B0_tesla, offset_ppm, gamma_MHz_per_T)
    x_Hz = x_ppm * BF1

    normalised_real_spectrum = _normalise(real_spectrum)
    normalised_imag_spectrum = imag_spectrum.copy()  # all-zero; normalising is a no-op

    path = _resolve_output_dir(output_dir, cif_filename)
    if save_shift_table:
        shift_table.to_csv(os.path.join(path, f"{nuclide}_shift_table.csv"), index=False)

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


