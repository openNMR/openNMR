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
`find_gamma` and `NUC_label` are ported as-is from nmrtools.utils (the same
helpers used by NMR1dimport / NMR1dplot in one_d.py), so a CIF-simulated
spectrum references (BF1/SFO1) and labels its axis identically to an
imported experimental one. `compute_BF1_SFO1` wraps `find_gamma` to turn a
nuclide + field strength into (BF1, SFO1); pass `gamma_MHz_per_T=` to
override the table lookup entirely if needed.
"""

from __future__ import annotations

import os
import re
import warnings
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


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
# find_gamma() and NUC_label() below are ported as-is from nmrtools.utils
# (the same helpers NMR1dimport / NMR1dplot use in one_d.py), so a simulated
# spectrum references and labels identically to an imported experimental one.
_GAMMA_TABLE = [
    ['Name', 'Nuclide', 'Spin', 'Magnetic Moment', 'Gyromagnetic Ratio (MHz/T)', 'Quadrupole Moment (fm^2)'],
    ['Hydrogen', '1H', '0.5', '4.83735', '42.57746460132430', '---'],
    ['Deuterium', '2H', '1', '1.21260', '6.53590463949470', '0.28600'],
    ['Helium', '3He', '0.5', '-3.68515', '-32.43603205003720', '---'],
    ['Tritium', '3H', '0.5', '5.15971', '45.41483118028370', '---'],
    ['Lithium', '6Li', '1', '1.16256', '6.26620067293118', '-0.08080'],
    ['Lithium', '7Li', '1.5', '4.20408', '16.54845000000000', '-4.01000'],
    ['Beryllium', '9Be', '1.5', '-1.52014', '-5.98370064894306', '5.28800'],
    ['Boron', '10B', '3', '2.07921', '4.57519531807410', '8.45900'],
    ['Boron', '11B', '1.5', '3.47103', '13.66297000000000', '4.05900'],
    ['Carbon', '13C', '0.5', '1.21661', '10.70839020506340', '---'],
    ['Nitrogen', '14N', '1', '0.57100', '3.07770645852245', '2.04400'],
    ['Nitrogen', '15N', '0.5', '-0.49050', '-4.31726881729937', '---'],
    ['Oxygen', '17O', '2.5', '-2.24077', '-5.77426865932844', '-2.55800'],
    ['Fluorine', '19F', '0.5', '4.55333', '40.07757016369700', '---'],
    ['Neon', '21Ne', '1.5', '-0.85438', '-3.36307127148622', '10.15500'],
    ['Sodium', '23Na', '1.5', '2.86298', '11.26952278792250', '10.40000'],
    ['Magnesium', '25Mg', '2.5', '-1.01220', '-2.60834261585015', '19.94000'],
    ['Aluminum', '27Al', '2.5', '4.30869', '11.10307854843700', '14.66000'],
    ['Silicon', '29Si', '0.5', '-0.96179', '-8.46545000000000', '---'],
    ['Phosphorus', '31P', '0.5', '1.95999', '17.25144000000000', '---'],
    ['Sulfur', '33S', '1.5', '0.83117', '3.27171633415147', '-6.78000'],
    ['Chlorine', '35Cl', '1.5', '1.06103', '4.17654000000000', '-8.16500'],
    ['Chlorine', '37Cl', '1.5', '0.88320', '3.47653283041643', '-6.43500'],
    ['Potassium', '39K', '1.5', '0.50543', '1.98953228161455', '5.85000'],
    ['Potassium', '40K', '4', '-1.45132', '-2.47372936498302', '-7.30000'],
    ['Potassium', '41K', '1.5', '0.27740', '1.09191431807057', '7.11000'],
    ['Calcium', '43Ca', '3.5', '-1.49407', '-2.86967503240704', '-4.08000'],
    ['Scandium', '45Sc', '3.5', '5.39335', '10.35908000000000', '-22.00000'],
    ['Titanium', '47Ti', '2.5', '-0.93294', '-2.40404000000000', '30.20000'],
    ['Titanium', '49Ti', '3.5', '-1.25201', '-2.40475161264699', '24.70000'],
    ['Vanadium', '50V', '6', '3.61376', '4.25047148768370', '21.00000'],
    ['Vanadium', '51V', '3.5', '5.83808', '11.21327743103380', '-5.20000'],
    ['Chromium', '53Cr', '1.5', '-0.61263', '-2.41152000000000', '-15.00000'],
    ['Manganese', '55Mn', '2.5', '4.10424', '10.57624385581420', '33.00000'],
    ['Iron', '57Fe', '0.5', '0.15696', '1.38156039900351', '---'],
    ['Cobalt', '59Co', '3.5', '5.24700', '10.07769000000000', '42.00000'],
    ['Nickel', '61Ni', '1.5', '-0.96827', '-3.81144000000000', '16.20000'],
    ['Copper', '63Cu', '1.5', '2.87549', '11.31876532731510', '-22.00000'],
    ['Copper', '65Cu', '1.5', '3.07465', '12.10269891500850', '-20.40000'],
    ['Zinc', '67Zn', '2.5', '1.03556', '2.66853501532750', '15.00000'],
    ['Gallium', '69Ga', '1.5', '2.60340', '10.24776396876680', '17.10000'],
    ['Gallium', '71Ga', '1.5', '3.30787', '13.02073645775120', '10.70000'],
    ['Germanium', '73Ge', '4.5', '-0.97229', '-1.48973801382307', '-19.60000'],
    ['Arsenic', '75As', '1.5', '1.85835', '7.31501583241246', '31.40000'],
    ['Selenium', '77Se', '0.5', '0.92678', '8.15731153773769', '---'],
    ['Bromine', '79Br', '1.5', '2.71935', '10.70415668357710', '31.30000'],
    ['Bromine', '81Br', '1.5', '2.93128', '11.53838323328760', '26.20000'],
    ['Krypton', '83Kr', '4.5', '-1.07311', '-1.64423000000000', '25.90000'],
    ['Rubidium', '85Rb', '2.5', '1.60131', '4.12642612503788', '27.60000'],
    ['Strontium', '87Sr', '4.5', '-1.20902', '-1.85246804462381', '33.50000'],
    ['Rubidium', '87Rb', '1.5', '3.55258', '13.98399000000000', '13.35000'],
    ['Yttrium', '89Y', '0.5', '-0.23801', '-2.09492468493000', '---'],
    ['Zirconium', '91Zr', '2.5', '-1.54246', '-3.97478329525992', '-17.60000'],
    ['Niobium', '93Nb', '4.5', '6.82170', '10.45234000000000', '-32.00000'],
    ['Molybdenium', '95Mo', '2.5', '-1.08200', '-2.78680000000000', '-2.20000'],
    ['Molybdenium', '97Mo', '2.5', '-1.10500', '-2.84569000000000', '25.50000'],
    ['Ruthenium', '99Ru', '2.5', '-0.75880', '-1.95601000000000', '7.90000'],
    ['Technetium', '99Tc', '4.5', '6.28100', '9.62251000000000', '-12.90000'],
    ['Ruthenium', '101Ru', '2.5', '-0.85050', '-2.19156000000000', '45.70000'],
    ['Rhodium', '103Rh', '0.5', '-0.15310', '-1.34772000000000', '---'],
    ['Palladium', '105Pd', '2.5', '-0.76000', '-1.95761000000000', '66.00000'],
    ['Silver', '107Ag', '0.5', '-0.19690', '-1.73307000631627', '---'],
    ['Silver', '109Ag', '0.5', '-0.22636', '-1.99239707059020', '---'],
    ['Cadmium', '111Cd', '0.5', '-1.03037', '-9.06914203769978', '---'],
    ['Indium', '113In', '4.5', '6.11240', '9.36547000000000', '79.90000'],
    ['Cadmium', '113Cd', '0.5', '-1.07786', '-9.48709883375341', '---'],
    ['Indium', '115In', '4.5', '6.12560', '9.38569000000000', '81.00000'],
    ['Tin', '115Sn', '0.5', '-1.59150', '-14.00770000000000', '---'],
    ['Tin', '117Sn', '0.5', '-1.73385', '-15.26103326770140', '---'],
    ['Tin', '119Sn', '0.5', '-1.81394', '-15.96595000000000', '---'],
    ['Antimony', '121Sb', '2.5', '3.97960', '10.25515000000000', '-36.00000'],
    ['Antimony', '123Sb', '3.5', '2.89120', '5.55323000000000', '-49.00000'],
    ['Tellurium', '123Te', '0.5', '-1.27643', '-11.23491000000000', '---'],
    ['Tellurium', '125Te', '0.5', '-1.53894', '-13.54542255864230', '---'],
    ['Iodine', '127I', '2.5', '3.32871', '8.57776706639786', '-71.00000'],
    ['Xenon', '129Xe', '0.5', '-1.34749', '-11.86039000000000', '---'],
    ['Xenon', '131Xe', '1.5', '0.89319', '3.51586001685444', '-11.40000'],
    ['Cesium', '133Cs', '3.5', '2.92774', '5.62334202679439', '-0.34300'],
    ['Barium', '135Ba', '1.5', '1.08178', '4.25819000000000', '16.00000'],
    ['Barium', '137Ba', '1.5', '1.21013', '4.76342786926888', '24.50000'],
    ['Lanthanum', '138La', '5', '4.06809', '5.66152329764214', '45.00000'],
    ['Lanthanum', '139La', '3.5', '3.15568', '6.06114544425158', '20.00000'],
    ['Praseodymium', '141Pr', '2.5', '5.05870', '13.03590000000000', '-5.89000'],
    ['Neodymium', '143Nd', '3.5', '-1.20800', '-2.31889000000000', '-63.00000'],
    ['Neodymium', '145Nd', '3.5', '-0.74400', '-1.42921000000000', '-33.00000'],
    ['Samarium', '147Sm', '3.5', '-0.92390', '-1.77458000000000', '-25.90000'],
    ['Samarium', '149Sm', '3.5', '-0.76160', '-1.46295000000000', '7.40000'],
    ['Europium', '151Eu', '2.5', '4.10780', '10.58540000000000', '90.30000'],
    ['Europium', '153Eu', '2.5', '1.81390', '4.67422000000000', '241.20000'],
    ['Gadolinium', '155Gd', '1.5', '-0.33208', '-1.30717137860235', '127.00000'],
    ['Gadolinium', '157Gd', '1.5', '-0.43540', '-1.71394000000000', '135.00000'],
    ['Terbium', '159Tb', '1.5', '2.60000', '10.23525000000000', '143.20000'],
    ['Dysprosium', '161Dy', '0.5', '-0.56830', '-1.46438000000000', '250.70000'],
    ['Dysprosium', '163Dy', '1', '0.79580', '2.05151000000000', '264.80000'],
    ['Holmium', '165Ho', '0.5', '4.73200', '9.08775000000000', '358.00000'],
    ['Erbium', '167Er', '0.5', '-0.63935', '-1.22799179441414', '356.50000'],
    ['Thulium', '169Tm', '1', '-0.40110', '-3.53006000000000', '---'],
    ['Ytterbium', '171Yb', '1.5', '0.85506', '7.52612000000000', '---'],
    ['Ytterbium', '173Yb', '1.5', '-0.80446', '-2.07299000000000', '280.00000'],
    ['Lutetium', '175Lu', '3', '2.53160', '4.86250000000000', '349.00000'],
    ['Lutetium', '176Lu', '1.5', '3.38800', '3.45112000000000', '497.00000'],
    ['Hafnium', '177Hf', '0.5', '0.89970', '1.72842000000000', '336.50000'],
    ['Hafnium', '179Hf', '1', '-0.70850', '-1.08560000000000', '379.30000'],
    ['Tantalum', '181Ta', '0.5', '2.68790', '5.16267000000000', '317.00000'],
    ['Tungsten', '183W', '2.5', '0.20401', '1.79564972994000', '---'],
    ['Rhenium', '185Re', '0.5', '3.77100', '9.71752000000000', '218.00000'],
    ['Osmium', '187Os', '1.5', '0.11198', '0.98563064707380', '---'],
    ['Rhenium', '187Re', '1.5', '3.80960', '9.81700000000000', '207.00000'],
    ['Osmium', '189Os', '2.5', '0.85197', '3.35360155237225', '85.60000'],
    ['Iridium', '191Ir', '2.5', '0.19460', '0.76585000000000', '81.60000'],
    ['Iridium', '193Ir', '0.5', '0.21130', '0.83190000000000', '75.10000'],
    ['Platinum', '195Pt', '0.5', '1.05570', '9.29226000000000', '---'],
    ['Gold', '197Au', '1.5', '0.19127', '0.75289837379052', '54.70000'],
    ['Mercury', '199Hg', '1.5', '0.87622', '7.71231431685275', '---'],
    ['Mercury', '201Hg', '1.5', '-0.72325', '-2.84691587554490', '38.60000'],
    ['Thallium', '203Tl', '1.5', '2.80983', '24.73161181836180', '---'],
    ['Thallium', '205Tl', '4', '2.83747', '24.97488014887780', '---'],
    ['Lead', '207Pb', '1.5', '1.00906', '8.88157793726598', '---'],
    ['Bismuth', '209Bi', '3.5', '4.54440', '6.96303000000000', '-51.60000'],
    ['Uranium', '235U', '3.5', '-0.43000', '-0.82761000000000', '493.60000'],
]

_GAMMA_DF = pd.DataFrame(_GAMMA_TABLE[1:], columns=_GAMMA_TABLE[0])


def find_gamma(isotope, field='', SFO1=''):
    """
    results = find_gamma(isotope, field='', SFO1='')

    isotope : str
        Nuclide as '#X', e.g. '17O', '27Al'. Read from Bruker input as
        NUC = results['nuclide'].

    field : '' or '100'
        Set to '100' to search for the actual nuclide observed on
        100 MHz (1H-equivalent 400 MHz) data, given a 500 MHz-equivalent
        SFO1. Converts from the 500 MHz equivalent SFO1 back to the
        actual nuclide/frequency at the true field.

    SFO1 : float
        Transmitter frequency. Read from Bruker input as
        SFO1 = results['SFO1'].

    Returns
    -------
    dict with keys 'Nuclide' and 'gamma' (in Hz/T, except when field='100',
    where 'gamma' is left in MHz/T - see the field='100' branch below).
    """

    df = _GAMMA_DF

    if df['Nuclide'].isin([isotope]).any():
        gamma = df.loc[df['Nuclide'] == isotope, "Gyromagnetic Ratio (MHz/T)"]
        gamma = float(gamma.iloc[0]) * 1e6
    else:
        print(
            "Isotope string not recognised, please input a string of "
            "format e.g., '1H' or '27Al' and ensure it is an NMR active "
            "nuclide."
        )
        return

    if field == '100':
        gamma = abs(gamma / 1e6)
        field_strength = 11.75
        freqactual = SFO1 / (field_strength / 5)
        sortdf = df.sort_values("Gyromagnetic Ratio (MHz/T)")
        df_closest = sortdf.iloc[
            (sortdf['Gyromagnetic Ratio (MHz/T)'].astype(float) - freqactual).abs().argsort()[:1]
        ]

        gamma = df_closest["Gyromagnetic Ratio (MHz/T)"].iloc[0]
        isotope = str(df_closest["Nuclide"].iloc[0])

    return {'Nuclide': isotope, 'gamma': gamma}


def NUC_label(NUC, x_unit='ppm'):
    """
    labeltext = NUC_label(NUC, x_unit='ppm')

    NUC : str
        Nuclide as '#X', e.g. '17O', '27Al'. Should be read from Bruker
        input.

    x_unit : 'ppm' or 'Hz'
        Assigns the x unit for the plot. Default = 'ppm'.
    """
    NUCalpha = NUC.strip('0123456789')
    NUCnum = NUC[:len(NUC) - len(NUCalpha)]
    labeltext = (
        r'$\mathregular{\delta \,\, ^{' + NUCnum + r'}}$'
        + NUCalpha + " (" + x_unit + ")"
    )
    return labeltext


def compute_BF1_SFO1(nuclide: str, B0_tesla: float, offset_ppm: float = 0.0,
                      gamma_MHz_per_T: Optional[float] = None
                      ) -> Tuple[float, float, str]:
    """
    Compute (BF1, SFO1, nuclide) in MHz for a given nuclide and field
    strength, using the same gyromagnetic-ratio lookup (find_gamma) as
    NMR1dimport / NMR1dplot in one_d.py - so a simulated spectrum
    references identically to an imported experimental one.

    BF1  - "basic frequency": the reference frequency defining 0 ppm at this
           field, i.e. gamma/(2*pi) * B0.
    SFO1 - the carrier/transmitter frequency actually used, i.e. BF1 shifted
           by `offset_ppm` (0 by default -> SFO1 == BF1, no carrier offset).
    nuclide - the (possibly canonicalised) nuclide string, as returned by
           find_gamma; identical to the input unless you also pass a
           field='100' remapping through **kwargs (not exposed here - call
           find_gamma directly for that case).

    gamma_MHz_per_T overrides the find_gamma table lookup entirely
    (useful for an isotope not in the table, or a literature-checked value).
    """
    if gamma_MHz_per_T is None:
        nucgamma = find_gamma(nuclide, field='', SFO1='')
        if nucgamma is None:
            raise KeyError(
                f"No entry for nuclide '{nuclide}' in the gyromagnetic-ratio "
                f"table (find_gamma). Pass gamma_MHz_per_T=<value> explicitly "
                f"(gamma/2*pi, in MHz/T) to override."
            )
        nuclide = nucgamma['Nuclide']
        gamma_MHz_per_T = nucgamma['gamma'] / 1e6  # find_gamma returns Hz/T here

    BF1 = gamma_MHz_per_T * B0_tesla
    SFO1 = BF1 * (1.0 + offset_ppm * 1e-6)
    return float(BF1), float(SFO1), nuclide


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


# ===========================================================================
# 10. Plotting helpers (work on the shared dict format)
# ===========================================================================
#
# All three plotting functions below share the same "usual plot modifiers":
#
#   tick_dir='in', tick_length=4.0, tick_width=1.0, tick_pad=4.0,
#   tick_number=None, xlims='default', pl_width=3.375, pl_height=3.375,
#   font='Arial', fontsize=12, fname=True
#
# xlims : 'default' auto-ranges from the plotted data (inverted per NMR
#         convention when invert_x=True); pass an explicit (xmin, xmax) to
#         override (used exactly as given, via ax.set_xlim(*xlims)).
# pl_width / pl_height : figure size in inches, only used when `ax=None`
#         (i.e. this call creates its own figure).
# fname : True  -> auto-name and save next to the spectrum's own 'path'
#                   (falls back to the current directory if unavailable);
#         str   -> save to that exact path;
#         False/None -> don't save.

def _apply_axis_style(ax, tick_dir, tick_length, tick_width, tick_pad,
                       tick_number, fontsize, show_yaxis):
    from matplotlib.ticker import MaxNLocator

    ax.tick_params(axis='x', direction=tick_dir, length=tick_length,
                    width=tick_width, pad=tick_pad, labelsize=fontsize)
    if tick_number is not None:
        ax.xaxis.set_major_locator(MaxNLocator(tick_number))

    if show_yaxis:
        ax.tick_params(axis='y', direction=tick_dir, length=tick_length,
                        width=tick_width, pad=tick_pad, labelsize=fontsize)
    else:
        ax.yaxis.set_visible(False)
        for spine in ("left", "right", "top"):
            ax.spines[spine].set_visible(False)


def _save_figure(fig, fname, default_dir, default_name):
    if not fname:
        return
    out_path = os.path.join(default_dir or ".", default_name) if fname is True else fname
    fig.savefig(out_path, dpi=300, bbox_inches="tight")


def plot_spectrum_dict(spectrum_dict, ax=None, axis="ppm", channel="real",
                        normalised=True, invert_x=True, offset=0.0, label=None,
                        show_yaxis=False,
                        tick_dir='in', tick_length=4.0, tick_width=1.0, tick_pad=4.0,
                        tick_number=None, xlims='default', pl_width=3.375, pl_height=3.375,
                        font='Arial', fontsize=12, fname=True, **plot_kwargs):
    """
    Plot a spectrum dict (simulated or experimental - same shape) on a
    Matplotlib Axes.

    axis     : "ppm" or "Hz"
    channel  : "real" or "imag"
    normalised : use the normalised_* arrays
    offset   : constant added to y after selecting the channel/normalisation
               (e.g. to stack several plot_spectrum_dict calls on one Axes)
    """
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = font

    x_key = "x_ppm" if axis == "ppm" else "x_Hz"
    prefix = "normalised_" if normalised else ""
    y_key = f"{prefix}{channel}_spectrum"

    x = spectrum_dict[x_key]
    y = spectrum_dict[y_key] + offset

    if ax is None:
        fig, ax = plt.subplots(figsize=(pl_width, pl_height))
    else:
        fig = ax.figure

    ax.plot(x, y, label=label, **plot_kwargs)

    _apply_axis_style(ax, tick_dir, tick_length, tick_width, tick_pad,
                       tick_number, fontsize, show_yaxis)

    if xlims == 'default':
        if invert_x:
            ax.set_xlim(max(x), min(x))
    else:
        ax.set_xlim(xlims[0], xlims[1])

    nuclide = spectrum_dict.get("nuclide", "")
    ax.set_xlabel(NUC_label(nuclide, x_unit=axis), fontsize=fontsize)

    if label is not None:
        ax.legend(fontsize=fontsize * 0.85)

    safe_nuclide = re.sub(r"[^\w\-]", "", nuclide)
    _save_figure(fig, fname, spectrum_dict.get("path"),
                 f"{safe_nuclide}_{channel}_{axis}_spectrum.png")

    return ax


def plot_comparison(simulated_dict, experimental_dict, ax=None, axis="ppm",
                     channel="real", normalised=True, invert_x=True,
                     sim_label="Simulated", exp_label="Experimental", offset=0.0,
                     show_yaxis=False,
                     tick_dir='in', tick_length=4.0, tick_width=1.0, tick_pad=4.0,
                     tick_number=None, xlims='default', pl_width=3.375, pl_height=3.375,
                     font='Arial', fontsize=12, fname=True):
    """
    Overlay a simulated spectrum dict and an experimental spectrum dict.

    offset : added to the simulated spectrum's y-values only, so the two
             traces can be separated vertically (e.g. offset=0.2 with
             normalised=True plots the simulated trace 0.2 above the
             experimental one).
    """
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = font

    if ax is None:
        fig, ax = plt.subplots(figsize=(pl_width, pl_height))
    else:
        fig = ax.figure

    common = dict(axis=axis, channel=channel, normalised=normalised, invert_x=False,
                   show_yaxis=show_yaxis, tick_dir=tick_dir, tick_length=tick_length,
                   tick_width=tick_width, tick_pad=tick_pad, tick_number=tick_number,
                   xlims='default', pl_width=pl_width, pl_height=pl_height,
                   font=font, fontsize=fontsize, fname=False)

    plot_spectrum_dict(experimental_dict, ax=ax, offset=0.0, label=exp_label,
                        color="tab:gray", **common)
    plot_spectrum_dict(simulated_dict, ax=ax, offset=offset, label=sim_label,
                        color="tab:red", **common)

    x_key = "x_ppm" if axis == "ppm" else "x_Hz"
    if xlims == 'default':
        if invert_x:
            x_all = np.concatenate([simulated_dict[x_key], experimental_dict[x_key]])
            ax.set_xlim(x_all.max(), x_all.min())
    else:
        ax.set_xlim(xlims[0], xlims[1])

    nuclide = simulated_dict.get("nuclide") or experimental_dict.get("nuclide") or ""
    safe_nuclide = re.sub(r"[^\w\-]", "", nuclide)
    path = simulated_dict.get("path") or experimental_dict.get("path")
    _save_figure(fig, fname, path, f"{safe_nuclide}_comparison_{channel}_{axis}.png")

    return ax


def plot_stacked_spectra(spectra, labels=None, offset_step=0.2, colors=None,
                          axis="ppm", channel="real", normalised=True, invert_x=True,
                          ax=None, show_yaxis=False,
                          tick_dir='in', tick_length=4.0, tick_width=1.0, tick_pad=4.0,
                          tick_number=None, xlims='default', pl_width=3.375, pl_height=3.375,
                          font='Arial', fontsize=12, fname=True):
    """
    Stack any number of spectrum dicts (simulated and/or experimental,
    freely mixed) vertically, offsetting spectrum i by i * offset_step.
    Handy for comparing a simulated spectrum against several experimental
    ones (or several simulated candidate models) at once.
    """
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = font

    if ax is None:
        fig, ax = plt.subplots(figsize=(pl_width, pl_height))
    else:
        fig = ax.figure

    labels = labels or [None] * len(spectra)
    colors = colors or [None] * len(spectra)

    common = dict(axis=axis, channel=channel, normalised=normalised, invert_x=False,
                   show_yaxis=show_yaxis, tick_dir=tick_dir, tick_length=tick_length,
                   tick_width=tick_width, tick_pad=tick_pad, tick_number=tick_number,
                   xlims='default', pl_width=pl_width, pl_height=pl_height,
                   font=font, fontsize=fontsize, fname=False)

    for i, (spec, lab, col) in enumerate(zip(spectra, labels, colors)):
        plot_spectrum_dict(spec, ax=ax, offset=i * offset_step, label=lab,
                            color=col, **common)

    x_key = "x_ppm" if axis == "ppm" else "x_Hz"
    if xlims == 'default':
        if invert_x and spectra:
            x_all = np.concatenate([s[x_key] for s in spectra])
            ax.set_xlim(x_all.max(), x_all.min())
    else:
        ax.set_xlim(xlims[0], xlims[1])

    nuclide = spectra[0].get("nuclide", "") if spectra else ""
    safe_nuclide = re.sub(r"[^\w\-]", "", nuclide)
    path = spectra[0].get("path") if spectra else None
    _save_figure(fig, fname, path, f"{safe_nuclide}_stack_{channel}_{axis}.png")

    return ax
