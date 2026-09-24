"""
General-purpose utilities shared by the 1D, 2D, fitting, pNMR-simulation
and BMS-simulation modules.

Contains:

- ``find_gamma``            : NMR isotope / gyromagnetic-ratio lookup table
- ``compute_BF1_SFO1``       : nuclide + field -> (BF1, SFO1) referencing,
                                built on top of find_gamma. Previously this
                                lived only inside pnmr_simulator.py (its own
                                private copy of find_gamma/NUC_label included)
                                -- moved here so any *_spectrum_dict producer
                                (experimental, CIF-pNMR, or BMS) references a
                                simulated spectrum identically, from one
                                place, instead of drifting copies.
- ``NUC_label``              : axis-label formatting for a given nuclide
- ``read_bruker_metadata``   : single source of truth for parsing Bruker
                                ``acqus``-style JCAMP-DX parameter files
"""

import os
from typing import Optional, Tuple

import pandas as pd
import numpy as np

from ..core.registry import NUC_label as _core_NUC_label


# ============================================================
# Gyromagnetic ratio table
# ============================================================

_GAMMA_TABLE = [
    ['Name', 'Nuclide', 'Spin', 'Magnetic Moment', 'Gyromagnetic Ratio (MHz/T)', 'Quadrupole Moment (fm^2)', 'Natural Abundance (%)'],
    ['Hydrogen', '1H', '0.5', '4.83735', '42.57746460132430', '---', '99.9885'],
    ['Deuterium', '2H', '1', '1.21260', '6.53590463949470', '0.28600', '0.0115'],
    ['Helium', '3He', '0.5', '-3.68515', '-32.43603205003720', '---', '0.0001'],
    ['Tritium', '3H', '0.5', '5.15971', '45.41483118028370', '---', '0.0'],
    ['Lithium', '6Li', '1', '1.16256', '6.26620067293118', '-0.08080', '7.59'],
    ['Lithium', '7Li', '1.5', '4.20408', '16.54845000000000', '-4.01000', '92.41'],
    ['Beryllium', '9Be', '1.5', '-1.52014', '-5.98370064894306', '5.28800', '100.0'],
    ['Boron', '10B', '3', '2.07921', '4.57519531807410', '8.45900', '19.9'],
    ['Boron', '11B', '1.5', '3.47103', '13.66297000000000', '4.05900', '80.1'],
    ['Carbon', '13C', '0.5', '1.21661', '10.70839020506340', '---', '1.07'],
    ['Nitrogen', '14N', '1', '0.57100', '3.07770645852245', '2.04400', '99.632'],
    ['Nitrogen', '15N', '0.5', '-0.49050', '-4.31726881729937', '---', '0.368'],
    ['Oxygen', '17O', '2.5', '-2.24077', '-5.77426865932844', '-2.55800', '0.038'],
    ['Fluorine', '19F', '0.5', '4.55333', '40.07757016369700', '---', '100.0'],
    ['Neon', '21Ne', '1.5', '-0.85438', '-3.36307127148622', '10.15500', '0.27'],
    ['Sodium', '23Na', '1.5', '2.86298', '11.26952278792250', '10.40000', '100.0'],
    ['Magnesium', '25Mg', '2.5', '-1.01220', '-2.60834261585015', '19.94000', '10.0'],
    ['Aluminum', '27Al', '2.5', '4.30869', '11.10307854843700', '14.66000', '100.0'],
    ['Silicon', '29Si', '0.5', '-0.96179', '-8.46545000000000', '---', '4.6832'],
    ['Phosphorus', '31P', '0.5', '1.95999', '17.25144000000000', '---', '100.0'],
    ['Sulfur', '33S', '1.5', '0.83117', '3.27171633415147', '-6.78000', '0.76'],
    ['Chlorine', '35Cl', '1.5', '1.06103', '4.17654000000000', '-8.16500', '75.78'],
    ['Chlorine', '37Cl', '1.5', '0.88320', '3.47653283041643', '-6.43500', '24.22'],
    ['Potassium', '39K', '1.5', '0.50543', '1.98953228161455', '5.85000', '93.2581'],
    ['Potassium', '40K', '4', '-1.45132', '-2.47372936498302', '-7.30000', '0.0117'],
    ['Potassium', '41K', '1.5', '0.27740', '1.09191431807057', '7.11000', '6.7302'],
    ['Calcium', '43Ca', '3.5', '-1.49407', '-2.86967503240704', '-4.08000', '0.135'],
    ['Scandium', '45Sc', '3.5', '5.39335', '10.35908000000000', '-22.00000', '100.0'],
    ['Titanium', '47Ti', '2.5', '-0.93294', '-2.40404000000000', '30.20000', '7.44'],
    ['Titanium', '49Ti', '3.5', '-1.25201', '-2.40475161264699', '24.70000', '5.41'],
    ['Vanadium', '50V', '6', '3.61376', '4.25047148768370', '21.00000', '0.25'],
    ['Vanadium', '51V', '3.5', '5.83808', '11.21327743103380', '-5.20000', '99.75'],
    ['Chromium', '53Cr', '1.5', '-0.61263', '-2.41152000000000', '-15.00000', '9.501'],
    ['Manganese', '55Mn', '2.5', '4.10424', '10.57624385581420', '33.00000', '100.0'],
    ['Iron', '57Fe', '0.5', '0.15696', '1.38156039900351', '---', '2.119'],
    ['Cobalt', '59Co', '3.5', '5.24700', '10.07769000000000', '42.00000', '100.0'],
    ['Nickel', '61Ni', '1.5', '-0.96827', '-3.81144000000000', '16.20000', '1.1399'],
    ['Copper', '63Cu', '1.5', '2.87549', '11.31876532731510', '-22.00000', '69.17'],
    ['Copper', '65Cu', '1.5', '3.07465', '12.10269891500850', '-20.40000', '30.83'],
    ['Zinc', '67Zn', '2.5', '1.03556', '2.66853501532750', '15.00000', '4.1'],
    ['Gallium', '69Ga', '1.5', '2.60340', '10.24776396876680', '17.10000', '60.108'],
    ['Gallium', '71Ga', '1.5', '3.30787', '13.02073645775120', '10.70000', '39.892'],
    ['Germanium', '73Ge', '4.5', '-0.97229', '-1.48973801382307', '-19.60000', '7.73'],
    ['Arsenic', '75As', '1.5', '1.85835', '7.31501583241246', '31.40000', '100.0'],
    ['Selenium', '77Se', '0.5', '0.92678', '8.15731153773769', '---', '7.63'],
    ['Bromine', '79Br', '1.5', '2.71935', '10.70415668357710', '31.30000', '50.69'],
    ['Bromine', '81Br', '1.5', '2.93128', '11.53838323328760', '26.20000', '49.31'],
    ['Krypton', '83Kr', '4.5', '-1.07311', '-1.64423000000000', '25.90000', '11.49'],
    ['Rubidium', '85Rb', '2.5', '1.60131', '4.12642612503788', '27.60000', '72.17'],
    ['Strontium', '87Sr', '4.5', '-1.20902', '-1.85246804462381', '33.50000', '7.0'],
    ['Rubidium', '87Rb', '1.5', '3.55258', '13.98399000000000', '13.35000', '27.83'],
    ['Yttrium', '89Y', '0.5', '-0.23801', '-2.09492468493000', '---', '100.0'],
    ['Zirconium', '91Zr', '2.5', '-1.54246', '-3.97478329525992', '-17.60000', '11.22'],
    ['Niobium', '93Nb', '4.5', '6.82170', '10.45234000000000', '-32.00000', '100.0'],
    ['Molybdenium', '95Mo', '2.5', '-1.08200', '-2.78680000000000', '-2.20000', '15.92'],
    ['Molybdenium', '97Mo', '2.5', '-1.10500', '-2.84569000000000', '25.50000', '9.55'],
    ['Ruthenium', '99Ru', '2.5', '-0.75880', '-1.95601000000000', '7.90000', '12.76'],
    ['Technetium', '99Tc', '4.5', '6.28100', '9.62251000000000', '-12.90000', '0.0'],
    ['Ruthenium', '101Ru', '2.5', '-0.85050', '-2.19156000000000', '45.70000', '17.06'],
    ['Rhodium', '103Rh', '0.5', '-0.15310', '-1.34772000000000', '---', '100.0'],
    ['Palladium', '105Pd', '2.5', '-0.76000', '-1.95761000000000', '66.00000', '22.33'],
    ['Silver', '107Ag', '0.5', '-0.19690', '-1.73307000631627', '---', '51.839'],
    ['Silver', '109Ag', '0.5', '-0.22636', '-1.99239707059020', '---', '48.161'],
    ['Cadmium', '111Cd', '0.5', '-1.03037', '-9.06914203769978', '---', '12.8'],
    ['Indium', '113In', '4.5', '6.11240', '9.36547000000000', '79.90000', '4.29'],
    ['Cadmium', '113Cd', '0.5', '-1.07786', '-9.48709883375341', '---', '12.22'],
    ['Indium', '115In', '4.5', '6.12560', '9.38569000000000', '81.00000', '95.71'],
    ['Tin', '115Sn', '0.5', '-1.59150', '-14.00770000000000', '---', '0.34'],
    ['Tin', '117Sn', '0.5', '-1.73385', '-15.26103326770140', '---', '7.68'],
    ['Tin', '119Sn', '0.5', '-1.81394', '-15.96595000000000', '---', '8.59'],
    ['Antimony', '121Sb', '2.5', '3.97960', '10.25515000000000', '-36.00000', '57.21'],
    ['Antimony', '123Sb', '3.5', '2.89120', '5.55323000000000', '-49.00000', '42.79'],
    ['Tellurium', '123Te', '0.5', '-1.27643', '-11.23491000000000', '---', '0.89'],
    ['Tellurium', '125Te', '0.5', '-1.53894', '-13.54542255864230', '---', '7.07'],
    ['Iodine', '127I', '2.5', '3.32871', '8.57776706639786', '-71.00000', '100.0'],
    ['Xenon', '129Xe', '0.5', '-1.34749', '-11.86039000000000', '---', '26.44'],
    ['Xenon', '131Xe', '1.5', '0.89319', '3.51586001685444', '-11.40000', '21.18'],
    ['Cesium', '133Cs', '3.5', '2.92774', '5.62334202679439', '-0.34300', '100.0'],
    ['Barium', '135Ba', '1.5', '1.08178', '4.25819000000000', '16.00000', '6.592'],
    ['Barium', '137Ba', '1.5', '1.21013', '4.76342786926888', '24.50000', '11.232'],
    ['Lanthanum', '138La', '5', '4.06809', '5.66152329764214', '45.00000', '0.09'],
    ['Lanthanum', '139La', '3.5', '3.15568', '6.06114544425158', '20.00000', '99.91'],
    ['Praseodymium', '141Pr', '2.5', '5.05870', '13.03590000000000', '-5.89000', '100.0'],
    ['Neodymium', '143Nd', '3.5', '-1.20800', '-2.31889000000000', '-63.00000', '12.2'],
    ['Neodymium', '145Nd', '3.5', '-0.74400', '-1.42921000000000', '-33.00000', '8.3'],
    ['Samarium', '147Sm', '3.5', '-0.92390', '-1.77458000000000', '-25.90000', '14.99'],
    ['Samarium', '149Sm', '3.5', '-0.76160', '-1.46295000000000', '7.40000', '13.82'],
    ['Europium', '151Eu', '2.5', '4.10780', '10.58540000000000', '90.30000', '47.81'],
    ['Europium', '153Eu', '2.5', '1.81390', '4.67422000000000', '241.20000', '52.19'],
    ['Gadolinium', '155Gd', '1.5', '-0.33208', '-1.30717137860235', '127.00000', '14.8'],
    ['Gadolinium', '157Gd', '1.5', '-0.43540', '-1.71394000000000', '135.00000', '15.65'],
    ['Terbium', '159Tb', '1.5', '2.60000', '10.23525000000000', '143.20000', '100.0'],
    ['Dysprosium', '161Dy', '2.5', '-0.56830', '-1.46438000000000', '250.70000', '18.91'],
    ['Dysprosium', '163Dy', '2.5', '0.79580', '2.05151000000000', '264.80000', '24.9'],
    ['Holmium', '165Ho', '3.5', '4.73200', '9.08775000000000', '358.00000', '100.0'],
    ['Erbium', '167Er', '3.5', '-0.63935', '-1.22799179441414', '356.50000', '22.93'],
    ['Thulium', '169Tm', '0.5', '-0.40110', '-3.53006000000000', '---', '100.0'],
    ['Ytterbium', '171Yb', '0.5', '0.85506', '7.52612000000000', '---', '14.28'],
    ['Ytterbium', '173Yb', '2.5', '-0.80446', '-2.07299000000000', '280.00000', '16.13'],
    ['Lutetium', '175Lu', '3.5', '2.53160', '4.86250000000000', '349.00000', '97.41'],
    ['Lutetium', '176Lu', '7', '3.38800', '3.45112000000000', '497.00000', '2.59'],
    ['Hafnium', '177Hf', '3.5', '0.89970', '1.72842000000000', '336.50000', '18.6'],
    ['Hafnium', '179Hf', '4.5', '-0.70850', '-1.08560000000000', '379.30000', '13.62'],
    ['Tantalum', '181Ta', '3.5', '2.68790', '5.16267000000000', '317.00000', '99.988'],
    ['Tungsten', '183W', '0.5', '0.20401', '1.79564972994000', '---', '14.31'],
    ['Rhenium', '185Re', '2.5', '3.77100', '9.71752000000000', '218.00000', '37.4'],
    ['Osmium', '187Os', '0.5', '0.11198', '0.98563064707380', '---', '1.96'],
    ['Rhenium', '187Re', '2.5', '3.80960', '9.81700000000000', '207.00000', '62.6'],
    ['Osmium', '189Os', '1.5', '0.85197', '3.35360155237225', '85.60000', '16.15'],
    ['Iridium', '191Ir', '1.5', '0.19460', '0.76585000000000', '81.60000', '37.3'],
    ['Iridium', '193Ir', '1.5', '0.21130', '0.83190000000000', '75.10000', '62.7'],
    ['Platinum', '195Pt', '0.5', '1.05570', '9.29226000000000', '---', '33.832'],
    ['Gold', '197Au', '1.5', '0.19127', '0.75289837379052', '54.70000', '100.0'],
    ['Mercury', '199Hg', '0.5', '0.87622', '7.71231431685275', '---', '16.87'],
    ['Mercury', '201Hg', '1.5', '-0.72325', '-2.84691587554490', '38.60000', '13.18'],
    ['Thallium', '203Tl', '0.5', '2.80983', '24.73161181836180', '---', '29.524'],
    ['Thallium', '205Tl', '0.5', '2.83747', '24.97488014887780', '---', '70.476'],
    ['Lead', '207Pb', '0.5', '1.00906', '8.88157793726598', '---', '22.1'],
    ['Bismuth', '209Bi', '4.5', '4.54440', '6.96303000000000', '-51.60000', '100.0'],
    ['Uranium', '235U', '3.5', '-0.43000', '-0.82761000000000', '493.60000', '0.72'],
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
    dict with keys 'Nuclide' and 'gamma' (in Hz/T).
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
        field_strength = 11.75
        freqactual = SFO1 / (field_strength / 5)
        sortdf = df.sort_values("Gyromagnetic Ratio (MHz/T)")
        df_closest = sortdf.iloc[
            (sortdf['Gyromagnetic Ratio (MHz/T)'].astype(float) - freqactual).abs().argsort()[:1]
        ]

        gamma = float(df_closest["Gyromagnetic Ratio (MHz/T)"].iloc[0]) * 1e6
        isotope = str(df_closest["Nuclide"].iloc[0])

    return {'Nuclide': isotope, 'gamma': gamma}


def find_isotope_info(isotope):
    """
    info = find_isotope_info(isotope)

    Convenience lookup returning everything the J-coupling multiplet
    machinery (see :func:`nmrtools.simulations.jcoupled_multiplet`)
    needs for a given nuclide in one call, drawn from the same
    ``_GAMMA_TABLE`` used by :func:`find_gamma` (so it never drifts out
    of sync with it) plus the natural-abundance column added alongside
    it.

    isotope : str
        Nuclide as '#X', e.g. '1H', '31P', '19F', '13C'.

    Returns
    -------
    dict with keys:

    ``nuclide``   : the nuclide string, as given.
    ``spin``      : nuclear spin quantum number I (float, e.g. 0.5 for
                    spin-1/2).
    ``gamma``     : gyromagnetic ratio in Hz/T (same convention as
                    :func:`find_gamma`).
    ``abundance`` : natural isotopic abundance, in % (0-100).

    Raises
    ------
    KeyError
        If ``isotope`` is not in the table.
    """
    df = _GAMMA_DF
    row = df.loc[df['Nuclide'] == isotope]
    if row.empty:
        raise KeyError(
            f"No entry for nuclide '{isotope}' in the gyromagnetic-ratio "
            f"table. Please use a string of format e.g. '1H' or '31P'."
        )
    row = row.iloc[0]
    return {
        'nuclide': isotope,
        'spin': float(row['Spin']),
        'gamma': float(row['Gyromagnetic Ratio (MHz/T)']) * 1e6,
        'abundance': float(row['Natural Abundance (%)']),
    }


def find_abundance(isotope):
    """
    abundance = find_abundance(isotope)

    Natural isotopic abundance (in %) for a given nuclide, e.g. for
    working out relative J-coupling multiplet/satellite intensities
    (see :func:`nmrtools.simulations.jcoupled_multiplet`): a spin
    coupled to a partner of low natural abundance (e.g. 13C, 1.07%)
    shows that coupling as a pair of small satellites either side of a
    much taller uncoupled centreband, whereas a partner at ~100%
    abundance (e.g. 31P, 19F, 1H) shows the full multiplet with no
    separate centreband.

    isotope : str
        Nuclide as '#X', e.g. '13C', '31P'.

    Returns
    -------
    float
        Natural abundance in %, 0-100.
    """
    return find_isotope_info(isotope)['abundance']


def compute_BF1_SFO1(nuclide: str, B0_tesla: float, offset_ppm: float = 0.0,
                      gamma_MHz_per_T: Optional[float] = None
                      ) -> Tuple[float, float, str]:
    """
    Compute (BF1, SFO1, nuclide) in MHz for a given nuclide and field
    strength, from the same gyromagnetic-ratio lookup (find_gamma) used
    by NMR1dimport/NMR1dplot (one_d.py) -- the single place every
    *_spectrum_dict producer in nmrtools (experimental import, CIF-pNMR
    prediction, BMS shift simulation) goes for referencing, so a
    simulated spectrum references identically to an imported experimental
    one, and adding a new predictor never means adding a new copy of this
    logic.

    BF1  - "basic frequency": the reference frequency defining 0 ppm at
           this field, i.e. gamma/(2*pi) * B0.
    SFO1 - the carrier/transmitter frequency actually used, i.e. BF1
           shifted by `offset_ppm` (0 by default -> SFO1 == BF1, no
           carrier offset).
    nuclide - the (possibly canonicalised) nuclide string, as returned by
           find_gamma; identical to the input unless you also pass a
           field='100' remapping through find_gamma directly (not
           exposed here).

    gamma_MHz_per_T overrides the find_gamma table lookup entirely
    (useful for an isotope not in the table, or a literature-checked
    value).

    Raises
    ------
    KeyError
        If `nuclide` is not in the gyromagnetic-ratio table and no
        `gamma_MHz_per_T` override is given.
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


def NUC_label(NUC, x_unit='ppm', style='brackets', template=None):
    """
    labeltext = NUC_label(NUC, x_unit='ppm', style='brackets', template=None)

    Thin wrapper around :func:`opennmr.core.registry.NUC_label` -- kept
    importable from here (``opennmr.notebook.utils``/``opennmr.notebook``)
    since every notebook-side plotting function already imports it from
    this module, but no longer a second implementation of its own: this
    and the GUI's axis labels were quietly drifting apart (this one had
    no ``style``/``slash`` option at all, and its mathtext wrapped the
    element symbol outside \mathregular{}, unlike the GUI's).

    NUC : str
        Nuclide as '#X', e.g. '17O', '27Al'. Should be read from Bruker
        input.

    x_unit : 'ppm' or 'Hz'
        Assigns the x unit for the plot. Default = 'ppm'.

    style / template : see :func:`opennmr.core.registry.NUC_label`.
    """
    return _core_NUC_label(NUC, x_unit=x_unit, style=style, template=template)


def read_vdlist(datapath, vdlist_name="vdlist"):
    """
    Read a Bruker variable-delay list (``vdlist``), as used by
    pseudo-2D relaxation experiments to label each slice with the
    delay actually used.

    Parameters
    ----------
    datapath : str
        Either:

        - the Bruker experiment directory, in which case the file
          named ``vdlist_name`` is read from directly inside it (and,
          failing that, from ``lists/vd/<vdlist_name>``, the location
          TopSpin itself uses for named delay lists); or
        - a direct path to a delay-list file anywhere on disk (e.g.
          one you keep outside the experiment directory), in which
          case it is read as-is and ``vdlist_name`` is ignored.

    vdlist_name : str, optional
        Name of the list file to look for when ``datapath`` is a
        directory. Default ``"vdlist"`` -- the conventional Bruker
        filename. Set this if your list has a different name and
        still lives inside the experiment directory; use a direct
        file path in ``datapath`` instead if it lives elsewhere.

    Returns
    -------
    dict
        ``"name"``    : resolved list filename.
        ``"raw"``     : list of the original strings, e.g. ``"10m"``.
        ``"seconds"`` : ndarray of delays converted to seconds.

    Raises
    ------
    FileNotFoundError
        If the list file cannot be found.

    Notes
    -----
    Bruker delay lists use a single trailing unit character per line:
    ``n`` (ns), ``u`` (\u00b5s), ``m`` (ms), ``s`` (s), or no suffix
    (assumed seconds). Blank lines are ignored.

    This no longer reads ``acqus`` for a ``VDLIST`` parameter -- the
    list filename is either the default ``"vdlist"``, an explicit
    ``vdlist_name``, or implied by a direct file path passed as
    ``datapath``.
    """

    if os.path.isfile(datapath):
        list_path = datapath
        vdlist_name = os.path.basename(datapath)
    else:
        candidates = [
            os.path.join(datapath, vdlist_name),
            os.path.join(datapath, "lists", "vd", vdlist_name),
        ]

        list_path = next((p for p in candidates if os.path.exists(p)), None)

        if list_path is None:
            raise FileNotFoundError(
                f"Could not find vdlist '{vdlist_name}' in {datapath} "
                f"(looked in: {candidates}). Pass a direct path to the "
                "list file as 'datapath' if it lives elsewhere."
            )

    unit_scale = {"n": 1e-9, "u": 1e-6, "m": 1e-3, "s": 1.0}

    raw, seconds = [], []

    with open(list_path, "r", errors="ignore") as inp:
        for line in inp:
            line = line.strip()
            if not line:
                continue
            raw.append(line)
            if line[-1] in unit_scale:
                value = float(line[:-1]) * unit_scale[line[-1]]
            else:
                value = float(line)
            seconds.append(value)

    return {"name": vdlist_name, "raw": raw, "seconds": np.array(seconds, dtype=float)}


#: Bruker JCAMP-DX parameters that are stored as an indexed array
#: (``##$X= (0..N)`` followed by N+1 whitespace-separated values on
#: the following line(s)) rather than as a single scalar. Shared by
#: :func:`read_bruker_metadata` for every array-valued parameter it
#: knows how to parse -- add an entry here (not a bespoke branch) to
#: pick up another one (e.g. ``"VCLIST"``, ``"VDLIST"`` if ever stored
#: inline rather than as a separate list file).
_ARRAY_PARAMS = ("P", "D", "CNST", "L")


def read_bruker_metadata(acqus_path):
    """
    Single source of truth for Bruker acquisition metadata.

    Parses a Bruker JCAMP-DX parameter file (``acqus``, ``acqu2s``,
    ``acqu3s``, ...) and returns the parameters needed elsewhere in
    nmrtools.

    Returns
    -------
    dict with keys:

    ``NUC``, ``OBS``, ``SFO1``, ``O1``, ``NS``
        As before.

    ``TD``, ``SW_h``, ``BYTORDA``, ``DTYPA``
        Time-domain point count, spectral width in Hz, and the raw
        binary FID's byte order (0 little-endian / 1 big-endian) and
        data type (0 = int32, 2 = float64) -- needed to read the raw
        ``fid``/``ser`` file directly, e.g. via
        :func:`read_bruker_fid`, rather than the already-processed
        ``1r``/``1i``. ``None`` if not present in this file.

    ``P``, ``D``, ``CNST``, ``L`` (list) and individual ``P0``,
    ``P1``, ..., ``D0``, ``D1``, ..., ``CNST0``, ``CNST1``, ...,
    ``L0``, ``L1``, ... entries
        Pulse lengths, delays, constants, and loop counters, each as
        both the full array (indexable in the natural Bruker way,
        i.e. ``meta["D"][6]`` for ``d6``) and as flattened
        ``{name}{i}`` keys. Only present if that parameter appears in
        this particular acqus file.
    """

    meta = {
        "NUC": None,
        "OBS": None,
        "SFO1": None,
        "O1": None,
        "NS": None,
        "TD": None,
        "SW_h": None,
        "BYTORDA": None,
        "DTYPA": None,
        "P": {},
    }

    with open(acqus_path, "r", errors="ignore") as inp:

        lines = iter(inp)

        for line in lines:

            line = line.strip()

            if line.startswith("##$NUC1="):
                meta["NUC"] = (
                    line.split("=")[1]
                    .strip()
                    .replace("<", "")
                    .replace(">", "")
                )

            elif line.startswith("##$BF1="):
                meta["OBS"] = float(line.split("=")[1])

            elif line.startswith("##$SFO1="):
                meta["SFO1"] = float(line.split("=")[1])

            elif line.startswith("##$O1="):
                meta["O1"] = float(line.split("=")[1])

            elif line.startswith("##$NS="):
                meta["NS"] = float(line.split("=")[1])

            elif line.startswith("##$TD="):
                meta["TD"] = int(line.split("=")[1])

            elif line.startswith("##$SW_h="):
                meta["SW_h"] = float(line.split("=")[1])

            elif line.startswith("##$BYTORDA="):
                meta["BYTORDA"] = int(line.split("=")[1])

            elif line.startswith("##$DTYPA="):
                meta["DTYPA"] = int(line.split("=")[1])

            else:
                for name in _ARRAY_PARAMS:
                    if line.startswith(f"##${name}="):

                        # Example:
                        # ##$D= (0..31)

                        nmax = int(line.split("..")[1].split(")")[0])

                        values = []

                        while len(values) < nmax + 1:
                            values.extend(next(lines).split())

                        values = [float(v) for v in values[:nmax + 1]]

                        for i, p in enumerate(values):
                            meta[f"{name}{i}"] = p

                        meta[name] = values

                        break

    if meta["NUC"] is None:
        raise ValueError("NUC not found in acqus")

    return meta


def read_bruker_fid(data_path, meta=None, fid_name="fid"):
    """
    Read a raw Bruker time-domain FID (or single-row ``ser``) file.

    Unlike :func:`nmrtools.one_d.NMR1dimport`, which reads the
    already-processed ``1r``/``1i`` spectrum from ``pdata``, this
    reads the raw digitised time-domain signal directly out of the
    top-level experiment directory -- needed for anything that has to
    reprocess the FID itself, such as CPMG echo-train splitting (see
    :func:`nmrtools.one_d.CPMGprocess`).

    Parameters
    ----------
    data_path : str
        Path to the top-level Bruker experiment directory (the one
        containing ``acqus`` and, directly inside it, the raw FID
        file -- not ``pdata``).

    meta : dict, optional
        Pre-parsed acqus metadata, as returned by
        :func:`read_bruker_metadata`. If not given, ``acqus`` is read
        from ``data_path`` to determine ``BYTORDA``/``DTYPA``.

    fid_name : str, optional
        Name of the raw data file inside ``data_path``. Default
        ``"fid"``; pass ``"ser"`` for a multi-dimensional acquisition
        being read one row at a time.

    Returns
    -------
    ndarray of complex128
        The FID as a single complex array, real and imaginary parts
        interleaved in the file as consecutive samples (the standard
        Bruker on-disk convention).

    Notes
    -----
    No digital-filter (group-delay) correction is applied -- if the
    acquisition used the usual oversampled digital receiver, the FID
    returned here still has the corresponding leading delay/artifact
    in it. This matters most for the very first points of the FID
    (e.g. when locating the first CPMG echo).
    """

    fid_path = os.path.join(data_path, fid_name)

    if meta is None:
        meta = read_bruker_metadata(os.path.join(data_path, "acqus"))

    byte_order = ">" if meta.get("BYTORDA") == 1 else "<"
    dtypa = meta.get("DTYPA") or 0

    dtype = np.dtype(byte_order + ("f8" if dtypa == 2 else "i4"))

    raw = np.fromfile(fid_path, dtype=dtype).astype(float)

    return raw[0::2] + 1j * raw[1::2]