"""
The NMR periodic table and Larmor-frequency calculator.

Every element, and the spin, natural abundance, gyromagnetic ratio,
quadrupole moment and receptivity of each of its NMR-active isotopes,
plus the Larmor frequency of any of them at any field. Explicit user
request (2026-09-28): an "NMR Periodic Table" in the GUI, modelled on
https://euannbassey.github.io/periodic_table.html -- search for and
highlight an element, click it for its nuclides -- with a Larmor
calculator beside it.

**One source of nuclear data.** The isotope values are *not* stored
here. They are read from ``notebook.utils._GAMMA_DF`` -- the table
:func:`~opennmr.notebook.utils.find_gamma` and
:func:`~opennmr.notebook.utils.compute_BF1_SFO1` use to reference every
spectrum openNMR imports or simulates. So a frequency this module
reports is, by construction, the one the rest of openNMR would use for
that nucleus; the two cannot drift apart. This module only adds what
that table lacks: the element itself (Z, name, atomic weight, position,
electron configuration), from the same data as the website table.

**What the Larmor frequency here is.** nu0 = |gamma / 2 pi| * B0, with
gamma the tabulated (bare-nucleus) value. The frequency at which a real
sample's 0 ppm reference resonates differs from this by that reference
compound's shielding: parts per million for 1H and 13C, but up to ~1 %
for heavy nuclei (195Pt, 207Pb, 199Hg...). For spectrometer
referencing use the reference compound's IUPAC Xi ratio; for "where
will this nucleus be, roughly, on my magnet" and for tuning ranges, this
is the right number.

Everything here works identically in a notebook and in the GUI (which
calls these functions; see ``gui/periodic_table_panel.py``)::

    from opennmr.notebook.periodic_table import (
        element, isotopes, larmor_frequency, larmor_table, plot_periodic_table)

    element("Al")["isotopes"][0]            # 27Al: I = 5/2, 100 %, ...
    larmor_frequency("27Al", B0_T=9.3947)   # 104.31 MHz
    larmor_frequency("27Al", proton_MHz=400)
    larmor_table(proton_MHz=600, spin=0.5)  # every spin-1/2 nucleus at 600 MHz
    plot_periodic_table(show="larmor", B0_T=14.1)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from .utils import _GAMMA_DF

#: Spin -> colour, the same (ColorBrewer "Paired") scheme as the website
#: table, so the two read the same. Keys are floats (0.5, 1.0, ...).
SPIN_COLOURS = {
    0.5: "#a6cee3", 1.0: "#1f78b4", 1.5: "#b2df8a", 2.0: "#33a02c",
    2.5: "#fb9a99", 3.0: "#e31a1c", 3.5: "#fdbf6f", 4.0: "#ff7f00",
    4.5: "#cab2d6", 5.0: "#6a3d9a", 6.0: "#2b2b2b", 7.0: "#b15928",
    9.0: "#656565",
}
#: Colour for an element with no NMR-active isotope in the table.
NO_NMR_COLOUR = "#e7e8e6"

#: Nuclides in the table that are radioactive *and* not meaningfully
#: present in natural samples (tabulated at 0 % abundance), or of an
#: element with no stable isotope at all. Marked with a dagger. The many
#: extremely long-lived primordial nuclides (e.g. 40K, 50V, 115In, 187Re)
#: are deliberately not flagged: for NMR they behave as stable nuclei at
#: their natural abundance.
RADIOACTIVE = frozenset({"3H", "99Tc", "235U"})

#: Magnets people actually have, as (1H frequency in MHz, label).
COMMON_MAGNETS = (
    (60.0, "60 MHz (benchtop)"), (80.0, "80 MHz (benchtop)"), (100.0, "100 MHz"),
    (200.0, "200 MHz"), (300.0, "300 MHz"), (400.0, "400 MHz"), (500.0, "500 MHz"),
    (600.0, "600 MHz"), (700.0, "700 MHz"), (800.0, "800 MHz"), (850.0, "850 MHz"),
    (900.0, "900 MHz"), (950.0, "950 MHz"), (1000.0, "1.0 GHz"), (1100.0, "1.1 GHz"),
    (1200.0, "1.2 GHz"),
)

#: Other spellings a search should accept, lower case -> symbol.
_NAME_ALIASES = {
    "aluminum": "Al", "cesium": "Cs", "sulphur": "S", "deuterium": "H",
    "tritium": "H", "proton": "H", "wolfram": "W", "niton": "Rn",
}

# (Z, symbol, name, standard atomic weight, category, period, group,
#  ground-state electron configuration). Lanthanides and actinides carry
#  group 3; :func:`grid_position` puts them in the f-block rows.
_ELEMENTS = (
    (1, 'H', 'Hydrogen', '1.008', 'nonmetal', 1, 1, '1s1'),
    (2, 'He', 'Helium', '4.0026', 'noble-gas', 1, 18, '1s2'),
    (3, 'Li', 'Lithium', '6.94', 'alkali-metal', 2, 1, '[He] 2s1'),
    (4, 'Be', 'Beryllium', '9.0122', 'alkaline-earth', 2, 2, '[He] 2s2'),
    (5, 'B', 'Boron', '10.81', 'metalloid', 2, 13, '[He] 2s2 2p1'),
    (6, 'C', 'Carbon', '12.011', 'nonmetal', 2, 14, '[He] 2s2 2p2'),
    (7, 'N', 'Nitrogen', '14.007', 'nonmetal', 2, 15, '[He] 2s2 2p3'),
    (8, 'O', 'Oxygen', '15.999', 'nonmetal', 2, 16, '[He] 2s2 2p4'),
    (9, 'F', 'Fluorine', '18.998', 'halogen', 2, 17, '[He] 2s2 2p5'),
    (10, 'Ne', 'Neon', '20.180', 'noble-gas', 2, 18, '[He] 2s2 2p6'),
    (11, 'Na', 'Sodium', '22.990', 'alkali-metal', 3, 1, '[Ne] 3s1'),
    (12, 'Mg', 'Magnesium', '24.305', 'alkaline-earth', 3, 2, '[Ne] 3s2'),
    (13, 'Al', 'Aluminium', '26.982', 'post-transition-metal', 3, 13, '[Ne] 3s2 3p1'),
    (14, 'Si', 'Silicon', '28.085', 'metalloid', 3, 14, '[Ne] 3s2 3p2'),
    (15, 'P', 'Phosphorus', '30.974', 'nonmetal', 3, 15, '[Ne] 3s2 3p3'),
    (16, 'S', 'Sulfur', '32.06', 'nonmetal', 3, 16, '[Ne] 3s2 3p4'),
    (17, 'Cl', 'Chlorine', '35.45', 'halogen', 3, 17, '[Ne] 3s2 3p5'),
    (18, 'Ar', 'Argon', '39.948', 'noble-gas', 3, 18, '[Ne] 3s2 3p6'),
    (19, 'K', 'Potassium', '39.098', 'alkali-metal', 4, 1, '[Ar] 4s1'),
    (20, 'Ca', 'Calcium', '40.078', 'alkaline-earth', 4, 2, '[Ar] 4s2'),
    (21, 'Sc', 'Scandium', '44.956', 'transition-metal', 4, 3, '[Ar] 3d1 4s2'),
    (22, 'Ti', 'Titanium', '47.867', 'transition-metal', 4, 4, '[Ar] 3d2 4s2'),
    (23, 'V', 'Vanadium', '50.942', 'transition-metal', 4, 5, '[Ar] 3d3 4s2'),
    (24, 'Cr', 'Chromium', '51.996', 'transition-metal', 4, 6, '[Ar] 3d5 4s1'),
    (25, 'Mn', 'Manganese', '54.938', 'transition-metal', 4, 7, '[Ar] 3d5 4s2'),
    (26, 'Fe', 'Iron', '55.845', 'transition-metal', 4, 8, '[Ar] 3d6 4s2'),
    (27, 'Co', 'Cobalt', '58.933', 'transition-metal', 4, 9, '[Ar] 3d7 4s2'),
    (28, 'Ni', 'Nickel', '58.693', 'transition-metal', 4, 10, '[Ar] 3d8 4s2'),
    (29, 'Cu', 'Copper', '63.546', 'transition-metal', 4, 11, '[Ar] 3d10 4s1'),
    (30, 'Zn', 'Zinc', '65.38', 'transition-metal', 4, 12, '[Ar] 3d10 4s2'),
    (31, 'Ga', 'Gallium', '69.723', 'post-transition-metal', 4, 13, '[Ar] 3d10 4s2 4p1'),
    (32, 'Ge', 'Germanium', '72.630', 'metalloid', 4, 14, '[Ar] 3d10 4s2 4p2'),
    (33, 'As', 'Arsenic', '74.922', 'metalloid', 4, 15, '[Ar] 3d10 4s2 4p3'),
    (34, 'Se', 'Selenium', '78.971', 'nonmetal', 4, 16, '[Ar] 3d10 4s2 4p4'),
    (35, 'Br', 'Bromine', '79.904', 'halogen', 4, 17, '[Ar] 3d10 4s2 4p5'),
    (36, 'Kr', 'Krypton', '83.798', 'noble-gas', 4, 18, '[Ar] 3d10 4s2 4p6'),
    (37, 'Rb', 'Rubidium', '85.468', 'alkali-metal', 5, 1, '[Kr] 5s1'),
    (38, 'Sr', 'Strontium', '87.62', 'alkaline-earth', 5, 2, '[Kr] 5s2'),
    (39, 'Y', 'Yttrium', '88.906', 'transition-metal', 5, 3, '[Kr] 4d1 5s2'),
    (40, 'Zr', 'Zirconium', '91.224', 'transition-metal', 5, 4, '[Kr] 4d2 5s2'),
    (41, 'Nb', 'Niobium', '92.906', 'transition-metal', 5, 5, '[Kr] 4d4 5s1'),
    (42, 'Mo', 'Molybdenum', '95.95', 'transition-metal', 5, 6, '[Kr] 4d5 5s1'),
    (43, 'Tc', 'Technetium', '[98]', 'transition-metal', 5, 7, '[Kr] 4d5 5s2'),
    (44, 'Ru', 'Ruthenium', '101.07', 'transition-metal', 5, 8, '[Kr] 4d7 5s1'),
    (45, 'Rh', 'Rhodium', '102.91', 'transition-metal', 5, 9, '[Kr] 4d8 5s1'),
    (46, 'Pd', 'Palladium', '106.42', 'transition-metal', 5, 10, '[Kr] 4d10'),
    (47, 'Ag', 'Silver', '107.87', 'transition-metal', 5, 11, '[Kr] 4d10 5s1'),
    (48, 'Cd', 'Cadmium', '112.41', 'transition-metal', 5, 12, '[Kr] 4d10 5s2'),
    (49, 'In', 'Indium', '114.82', 'post-transition-metal', 5, 13, '[Kr] 4d10 5s2 5p1'),
    (50, 'Sn', 'Tin', '118.71', 'post-transition-metal', 5, 14, '[Kr] 4d10 5s2 5p2'),
    (51, 'Sb', 'Antimony', '121.76', 'metalloid', 5, 15, '[Kr] 4d10 5s2 5p3'),
    (52, 'Te', 'Tellurium', '127.60', 'metalloid', 5, 16, '[Kr] 4d10 5s2 5p4'),
    (53, 'I', 'Iodine', '126.90', 'halogen', 5, 17, '[Kr] 4d10 5s2 5p5'),
    (54, 'Xe', 'Xenon', '131.29', 'noble-gas', 5, 18, '[Kr] 4d10 5s2 5p6'),
    (55, 'Cs', 'Caesium', '132.91', 'alkali-metal', 6, 1, '[Xe] 6s1'),
    (56, 'Ba', 'Barium', '137.33', 'alkaline-earth', 6, 2, '[Xe] 6s2'),
    (57, 'La', 'Lanthanum', '138.91', 'lanthanide', 6, 3, '[Xe] 5d1 6s2'),
    (58, 'Ce', 'Cerium', '140.12', 'lanthanide', 6, 3, '[Xe] 4f1 5d1 6s2'),
    (59, 'Pr', 'Praseodymium', '140.91', 'lanthanide', 6, 3, '[Xe] 4f3 6s2'),
    (60, 'Nd', 'Neodymium', '144.24', 'lanthanide', 6, 3, '[Xe] 4f4 6s2'),
    (61, 'Pm', 'Promethium', '[145]', 'lanthanide', 6, 3, '[Xe] 4f5 6s2'),
    (62, 'Sm', 'Samarium', '150.36', 'lanthanide', 6, 3, '[Xe] 4f6 6s2'),
    (63, 'Eu', 'Europium', '151.96', 'lanthanide', 6, 3, '[Xe] 4f7 6s2'),
    (64, 'Gd', 'Gadolinium', '157.25', 'lanthanide', 6, 3, '[Xe] 4f7 5d1 6s2'),
    (65, 'Tb', 'Terbium', '158.93', 'lanthanide', 6, 3, '[Xe] 4f9 6s2'),
    (66, 'Dy', 'Dysprosium', '162.50', 'lanthanide', 6, 3, '[Xe] 4f10 6s2'),
    (67, 'Ho', 'Holmium', '164.93', 'lanthanide', 6, 3, '[Xe] 4f11 6s2'),
    (68, 'Er', 'Erbium', '167.26', 'lanthanide', 6, 3, '[Xe] 4f12 6s2'),
    (69, 'Tm', 'Thulium', '168.93', 'lanthanide', 6, 3, '[Xe] 4f13 6s2'),
    (70, 'Yb', 'Ytterbium', '173.05', 'lanthanide', 6, 3, '[Xe] 4f14 6s2'),
    (71, 'Lu', 'Lutetium', '174.97', 'lanthanide', 6, 3, '[Xe] 4f14 5d1 6s2'),
    (72, 'Hf', 'Hafnium', '178.49', 'transition-metal', 6, 4, '[Xe] 4f14 5d2 6s2'),
    (73, 'Ta', 'Tantalum', '180.95', 'transition-metal', 6, 5, '[Xe] 4f14 5d3 6s2'),
    (74, 'W', 'Tungsten', '183.84', 'transition-metal', 6, 6, '[Xe] 4f14 5d4 6s2'),
    (75, 'Re', 'Rhenium', '186.21', 'transition-metal', 6, 7, '[Xe] 4f14 5d5 6s2'),
    (76, 'Os', 'Osmium', '190.23', 'transition-metal', 6, 8, '[Xe] 4f14 5d6 6s2'),
    (77, 'Ir', 'Iridium', '192.22', 'transition-metal', 6, 9, '[Xe] 4f14 5d7 6s2'),
    (78, 'Pt', 'Platinum', '195.08', 'transition-metal', 6, 10, '[Xe] 4f14 5d9 6s1'),
    (79, 'Au', 'Gold', '196.97', 'transition-metal', 6, 11, '[Xe] 4f14 5d10 6s1'),
    (80, 'Hg', 'Mercury', '200.59', 'transition-metal', 6, 12, '[Xe] 4f14 5d10 6s2'),
    (81, 'Tl', 'Thallium', '204.38', 'post-transition-metal', 6, 13, '[Xe] 4f14 5d10 6s2 6p1'),
    (82, 'Pb', 'Lead', '207.2', 'post-transition-metal', 6, 14, '[Xe] 4f14 5d10 6s2 6p2'),
    (83, 'Bi', 'Bismuth', '208.98', 'post-transition-metal', 6, 15, '[Xe] 4f14 5d10 6s2 6p3'),
    (84, 'Po', 'Polonium', '[209]', 'metalloid', 6, 16, '[Xe] 4f14 5d10 6s2 6p4'),
    (85, 'At', 'Astatine', '[210]', 'halogen', 6, 17, '[Xe] 4f14 5d10 6s2 6p5'),
    (86, 'Rn', 'Radon', '[222]', 'noble-gas', 6, 18, '[Xe] 4f14 5d10 6s2 6p6'),
    (87, 'Fr', 'Francium', '[223]', 'alkali-metal', 7, 1, '[Rn] 7s1'),
    (88, 'Ra', 'Radium', '[226]', 'alkaline-earth', 7, 2, '[Rn] 7s2'),
    (89, 'Ac', 'Actinium', '[227]', 'actinide', 7, 3, '[Rn] 6d1 7s2'),
    (90, 'Th', 'Thorium', '232.04', 'actinide', 7, 3, '[Rn] 6d2 7s2'),
    (91, 'Pa', 'Protactinium', '231.04', 'actinide', 7, 3, '[Rn] 5f2 6d1 7s2'),
    (92, 'U', 'Uranium', '238.03', 'actinide', 7, 3, '[Rn] 5f3 6d1 7s2'),
    (93, 'Np', 'Neptunium', '[237]', 'actinide', 7, 3, '[Rn] 5f4 6d1 7s2'),
    (94, 'Pu', 'Plutonium', '[244]', 'actinide', 7, 3, '[Rn] 5f6 7s2'),
    (95, 'Am', 'Americium', '[243]', 'actinide', 7, 3, '[Rn] 5f7 7s2'),
    (96, 'Cm', 'Curium', '[247]', 'actinide', 7, 3, '[Rn] 5f7 6d1 7s2'),
    (97, 'Bk', 'Berkelium', '[247]', 'actinide', 7, 3, '[Rn] 5f9 7s2'),
    (98, 'Cf', 'Californium', '[251]', 'actinide', 7, 3, '[Rn] 5f10 7s2'),
    (99, 'Es', 'Einsteinium', '[252]', 'actinide', 7, 3, '[Rn] 5f11 7s2'),
    (100, 'Fm', 'Fermium', '[257]', 'actinide', 7, 3, '[Rn] 5f12 7s2'),
    (101, 'Md', 'Mendelevium', '[258]', 'actinide', 7, 3, '[Rn] 5f13 7s2'),
    (102, 'No', 'Nobelium', '[259]', 'actinide', 7, 3, '[Rn] 5f14 7s2'),
    (103, 'Lr', 'Lawrencium', '[266]', 'actinide', 7, 3, '[Rn] 5f14 7s2 7p1'),
    (104, 'Rf', 'Rutherfordium', '[267]', 'transition-metal', 7, 4, '[Rn] 5f14 6d2 7s2'),
    (105, 'Db', 'Dubnium', '[268]', 'transition-metal', 7, 5, '[Rn] 5f14 6d3 7s2'),
    (106, 'Sg', 'Seaborgium', '[269]', 'transition-metal', 7, 6, '[Rn] 5f14 6d4 7s2'),
    (107, 'Bh', 'Bohrium', '[270]', 'transition-metal', 7, 7, '[Rn] 5f14 6d5 7s2'),
    (108, 'Hs', 'Hassium', '[269]', 'transition-metal', 7, 8, '[Rn] 5f14 6d6 7s2'),
    (109, 'Mt', 'Meitnerium', '[278]', 'unknown', 7, 9, '[Rn] 5f14 6d7 7s2'),
    (110, 'Ds', 'Darmstadtium', '[281]', 'unknown', 7, 10, '[Rn] 5f14 6d8 7s2'),
    (111, 'Rg', 'Roentgenium', '[282]', 'unknown', 7, 11, '[Rn] 5f14 6d9 7s2'),
    (112, 'Cn', 'Copernicium', '[285]', 'transition-metal', 7, 12, '[Rn] 5f14 6d10 7s2'),
    (113, 'Nh', 'Nihonium', '[286]', 'post-transition-metal', 7, 13, '[Rn] 5f14 6d10 7s2 7p1'),
    (114, 'Fl', 'Flerovium', '[289]', 'post-transition-metal', 7, 14, '[Rn] 5f14 6d10 7s2 7p2'),
    (115, 'Mc', 'Moscovium', '[290]', 'post-transition-metal', 7, 15, '[Rn] 5f14 6d10 7s2 7p3'),
    (116, 'Lv', 'Livermorium', '[293]', 'post-transition-metal', 7, 16, '[Rn] 5f14 6d10 7s2 7p4'),
    (117, 'Ts', 'Tennessine', '[294]', 'halogen', 7, 17, '[Rn] 5f14 6d10 7s2 7p5'),
    (118, 'Og', 'Oganesson', '[294]', 'noble-gas', 7, 18, '[Rn] 5f14 6d10 7s2 7p6'),)


# ============================================================
# Data classes
# ============================================================

@dataclass(frozen=True)
class Isotope:
    """One NMR-active nuclide, as tabulated in ``notebook.utils``."""

    nuclide: str                 #: "27Al"
    symbol: str                  #: "Al"
    A: int                       #: mass number
    spin: float                  #: nuclear spin quantum number I
    gamma_MHz_per_T: float       #: gamma / 2 pi, signed, MHz/T
    abundance: float             #: natural abundance, % (0-100)
    quadrupole_fm2: Optional[float]  #: Q in fm^2; None for I = 1/2
    magnetic_moment: Optional[float]  #: mu / mu_N, as tabulated
    radioactive: bool = False

    @property
    def spin_label(self) -> str:
        return spin_label(self.spin)

    @property
    def receptivity_1H(self) -> float:
        """Receptivity relative to 1H (D^p): |gamma|^3 x abundance x I(I+1)."""
        return _receptivity(self) / _receptivity(_ISOTOPES["1H"])

    @property
    def receptivity_13C(self) -> float:
        """Receptivity relative to 13C (D^C), the usual quoted number."""
        return _receptivity(self) / _receptivity(_ISOTOPES["13C"])

    def larmor_MHz(self, B0_T: float, signed: bool = False) -> float:
        nu = self.gamma_MHz_per_T * float(B0_T)
        return nu if signed else abs(nu)

    def as_dict(self) -> dict:
        return {
            "nuclide": self.nuclide, "symbol": self.symbol, "A": self.A,
            "spin": self.spin, "gamma_MHz_per_T": self.gamma_MHz_per_T,
            "abundance": self.abundance, "quadrupole_fm2": self.quadrupole_fm2,
            "magnetic_moment": self.magnetic_moment, "radioactive": self.radioactive,
            "receptivity_1H": self.receptivity_1H, "receptivity_13C": self.receptivity_13C,
        }


@dataclass(frozen=True)
class Element:
    Z: int
    symbol: str
    name: str
    mass: str                    #: standard atomic weight, as a string ("[98]" if none)
    category: str
    period: int
    group: int
    econfig: str
    isotopes: tuple = field(default_factory=tuple)  #: NMR isotopes, most abundant first

    @property
    def has_nmr(self) -> bool:
        return bool(self.isotopes)

    @property
    def principal_isotope(self) -> Optional[Isotope]:
        """The most abundant NMR-active isotope (the one its tile is coloured by)."""
        return self.isotopes[0] if self.isotopes else None

    def as_dict(self) -> dict:
        return {
            "Z": self.Z, "symbol": self.symbol, "name": self.name, "mass": self.mass,
            "category": self.category, "period": self.period, "group": self.group,
            "econfig": self.econfig, "isotopes": [i.as_dict() for i in self.isotopes],
        }


# ============================================================
# Building the index from the shared gamma table
# ============================================================

def _receptivity(iso: Isotope) -> float:
    return abs(iso.gamma_MHz_per_T) ** 3 * iso.abundance * iso.spin * (iso.spin + 1.0)


def _float_or_none(value) -> Optional[float]:
    try:
        text = str(value).strip()
        if text in ("", "---", "nan", "None"):
            return None
        return float(text)
    except (TypeError, ValueError):
        return None


_NUCLIDE_RE = re.compile(r"^(\d+)([A-Z][a-z]?)$")


def _build():
    isotopes: dict[str, Isotope] = {}
    for _i, row in _GAMMA_DF.iterrows():
        nuclide = str(row["Nuclide"]).strip()
        match = _NUCLIDE_RE.match(nuclide)
        if not match:
            continue
        quad = _float_or_none(row.get("Quadrupole Moment (fm^2)"))
        spin = float(row["Spin"])
        isotopes[nuclide] = Isotope(
            nuclide=nuclide, symbol=match.group(2), A=int(match.group(1)), spin=spin,
            gamma_MHz_per_T=float(row["Gyromagnetic Ratio (MHz/T)"]),
            abundance=_float_or_none(row.get("Natural Abundance (%)")) or 0.0,
            quadrupole_fm2=None if spin == 0.5 else quad,
            magnetic_moment=_float_or_none(row.get("Magnetic Moment")),
            radioactive=nuclide in RADIOACTIVE,
        )
    by_symbol: dict[str, list] = {}
    for iso in isotopes.values():
        by_symbol.setdefault(iso.symbol, []).append(iso)
    elements = []
    for Z, symbol, name, mass, category, period, group, econfig in _ELEMENTS:
        own = sorted(by_symbol.get(symbol, ()), key=lambda i: (-i.abundance, i.A))
        elements.append(Element(Z, symbol, name, mass, category, period, group, econfig,
                                tuple(own)))
    return isotopes, tuple(elements)


_ISOTOPES, _ELEMENT_LIST = _build()
_BY_SYMBOL = {e.symbol: e for e in _ELEMENT_LIST}
_BY_SYMBOL_LOWER = {e.symbol.lower(): e for e in _ELEMENT_LIST}
_BY_NAME = {e.name.lower(): e for e in _ELEMENT_LIST}


# ============================================================
# Lookups
# ============================================================

_SUPERSCRIPTS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")


def spin_label(spin: float) -> str:
    """0.5 -> "1/2", 1.0 -> "1", 3.5 -> "7/2"."""
    frac = Fraction(float(spin)).limit_denominator(2)
    return str(frac.numerator) if frac.denominator == 1 else f"{frac.numerator}/{frac.denominator}"


def normalise_nuclide(text: str) -> Optional[str]:
    """ "27Al", "27 al", "Al27", "Al-27", "²⁷Al" -> "27Al"; None if the text
    is not shaped like a nuclide (it need not be NMR-active)."""
    t = (text or "").translate(_SUPERSCRIPTS).strip().replace(" ", "").replace("-", "")
    m = re.match(r"^(\d+)([A-Za-z]{1,2})$", t) or None
    if m:
        mass, sym = m.group(1), m.group(2)
    else:
        m = re.match(r"^([A-Za-z]{1,2})(\d+)$", t)
        if not m:
            return None
        sym, mass = m.group(1), m.group(2)
    sym = sym[0].upper() + sym[1:].lower()
    return f"{int(mass)}{sym}"


def elements() -> list:
    """All 118 elements, by Z."""
    return list(_ELEMENT_LIST)


def element(query) -> Element:
    """An element by symbol ("Al"), name ("aluminium"/"aluminum"), Z (13)
    or one of its nuclides ("27Al"). Raises KeyError if nothing matches."""
    if isinstance(query, Element):
        return query
    if isinstance(query, (int, np.integer)):
        if 1 <= int(query) <= len(_ELEMENT_LIST):
            return _ELEMENT_LIST[int(query) - 1]
        raise KeyError(f"No element with Z = {query}.")
    text = str(query).strip()
    if text.isdigit():
        return element(int(text))
    if text in _BY_SYMBOL:
        return _BY_SYMBOL[text]
    low = text.lower()
    if low in _BY_SYMBOL_LOWER:
        return _BY_SYMBOL_LOWER[low]
    if low in _BY_NAME:
        return _BY_NAME[low]
    if low in _NAME_ALIASES:
        return _BY_SYMBOL[_NAME_ALIASES[low]]
    nuc = normalise_nuclide(text)
    if nuc is not None:
        sym = _NUCLIDE_RE.match(nuc).group(2)
        if sym in _BY_SYMBOL:
            return _BY_SYMBOL[sym]
    raise KeyError(f"'{query}' is not an element symbol, name, atomic number or nuclide.")


def isotope(nuclide: str) -> Isotope:
    """One NMR-active nuclide ("27Al", "Al-27", ...). KeyError if it is not
    in openNMR's table (i.e. not NMR-active, or not tabulated)."""
    key = normalise_nuclide(nuclide)
    if key is None or key not in _ISOTOPES:
        raise KeyError(
            f"'{nuclide}' is not an NMR-active nuclide in openNMR's table "
            f"(try e.g. '27Al' or '1H').")
    return _ISOTOPES[key]


def isotopes(query=None) -> list:
    """NMR-active isotopes of one element (most abundant first), or of every
    element (by Z, then abundance) when ``query`` is None."""
    if query is None:
        return [iso for el in _ELEMENT_LIST for iso in el.isotopes]
    return list(element(query).isotopes)


def search(query: str) -> list:
    """Elements matching a search box entry, best match first.

    Matches, in this order: a nuclide ("27Al", "Al27" -> Al); an atomic
    number; an exact symbol; symbols starting with the text; names (and
    alternative spellings) starting with it. An empty query matches
    nothing -- the table then shows everything undimmed."""
    q = (query or "").strip()
    if not q:
        return []
    out: list = []

    def add(el):
        if el not in out:
            out.append(el)

    nuc = normalise_nuclide(q)
    if nuc is not None and nuc[-1].isalpha():
        try:
            add(element(nuc))
        except KeyError:
            pass
    if q.isdigit():
        try:
            add(element(int(q)))
        except KeyError:
            pass
        return out
    low = q.lower()
    for el in _ELEMENT_LIST:
        if el.symbol.lower() == low:
            add(el)
    for el in _ELEMENT_LIST:
        if el.symbol.lower().startswith(low):
            add(el)
    for el in _ELEMENT_LIST:
        if el.name.lower().startswith(low):
            add(el)
    for alias, sym in _NAME_ALIASES.items():
        if alias.startswith(low):
            add(_BY_SYMBOL[sym])
    return out


def grid_position(el) -> tuple:
    """(row, column), 1-based, in the 18-column table: lanthanides on row
    9 and actinides on row 10 (row 8 is the gap), columns 4-18."""
    el = element(el)
    if 57 <= el.Z <= 71:
        return 9, el.Z - 57 + 4
    if 89 <= el.Z <= 103:
        return 10, el.Z - 89 + 4
    return el.period, el.group


def spins_present() -> list:
    """Every spin any tabulated nuclide has, ascending."""
    return sorted({iso.spin for iso in _ISOTOPES.values()})


# ============================================================
# Field and Larmor frequency
# ============================================================

def gamma_1H_MHz_per_T() -> float:
    return _ISOTOPES["1H"].gamma_MHz_per_T


def proton_frequency_to_field(proton_MHz: float) -> float:
    """B0 in tesla of a magnet named by its 1H frequency ("400 MHz")."""
    return float(proton_MHz) / gamma_1H_MHz_per_T()


def field_to_proton_frequency(B0_T: float) -> float:
    """The 1H Larmor frequency, MHz, at B0 (tesla)."""
    return float(B0_T) * gamma_1H_MHz_per_T()


def _resolve_field(B0_T, proton_MHz) -> float:
    if (B0_T is None) == (proton_MHz is None):
        raise ValueError("Give exactly one of B0_T (tesla) or proton_MHz (the magnet's 1H frequency).")
    B0 = float(B0_T) if B0_T is not None else proton_frequency_to_field(proton_MHz)
    if not np.isfinite(B0) or B0 < 0:
        raise ValueError(f"The field must be a finite, non-negative number (got {B0}).")
    return B0


def larmor_frequency(nuclide: str, B0_T: float | None = None, *,
                     proton_MHz: float | None = None, signed: bool = False) -> float:
    """Larmor frequency of ``nuclide`` in MHz: |gamma/2pi| x B0.

    Give the field either in tesla (``B0_T``) or as the magnet's 1H
    frequency (``proton_MHz=400``). ``signed=True`` keeps the sign of
    gamma (negative for e.g. 15N, 29Si, 119Sn). Uses the same table as
    :func:`~opennmr.notebook.utils.compute_BF1_SFO1`."""
    B0 = _resolve_field(B0_T, proton_MHz)
    return isotope(nuclide).larmor_MHz(B0, signed=signed)


def larmor_working(nuclide: str, B0_T: float | None = None, *,
                   proton_MHz: float | None = None) -> str:
    """The calculation written out, for a log or a notebook cell:
    "nu0(27Al) = |gamma/2pi| x B0 = 11.103079 MHz/T x 9.394700 T = 104.3095 MHz"."""
    B0 = _resolve_field(B0_T, proton_MHz)
    iso = isotope(nuclide)
    return (f"ν₀({iso.nuclide}) = |γ/2π| × B₀ = {abs(iso.gamma_MHz_per_T):.6f} MHz/T × "
            f"{B0:.6f} T = {iso.larmor_MHz(B0):.4f} MHz")


def larmor_table(B0_T: float | None = None, *, proton_MHz: float | None = None,
                 element=None, spin: float | None = None,
                 min_abundance: float | None = None,
                 sort_by: str = "frequency") -> pd.DataFrame:
    """Every NMR-active nuclide (optionally of one ``element``, with one
    ``spin``, or at least ``min_abundance`` %) and its Larmor frequency at
    the given field, as a DataFrame.

    Columns: nuclide, element, Z, spin, spin_label, abundance_pct,
    gamma_MHz_per_T, quadrupole_fm2, larmor_MHz, ratio_to_1H_pct,
    receptivity_1H, receptivity_13C, radioactive. ``sort_by`` is
    "frequency" (descending, the order a probe's tuning range reads in),
    "Z", or "receptivity"."""
    B0 = _resolve_field(B0_T, proton_MHz)
    pool = isotopes(element) if element is not None else isotopes()
    rows = []
    for iso in pool:
        if spin is not None and abs(iso.spin - float(spin)) > 1e-9:
            continue
        if min_abundance is not None and iso.abundance < float(min_abundance):
            continue
        el = _BY_SYMBOL[iso.symbol]
        rows.append({
            "nuclide": iso.nuclide, "element": el.name, "Z": el.Z,
            "spin": iso.spin, "spin_label": iso.spin_label,
            "abundance_pct": iso.abundance, "gamma_MHz_per_T": iso.gamma_MHz_per_T,
            "quadrupole_fm2": iso.quadrupole_fm2, "larmor_MHz": iso.larmor_MHz(B0),
            "ratio_to_1H_pct": 100.0 * abs(iso.gamma_MHz_per_T) / abs(gamma_1H_MHz_per_T()),
            "receptivity_1H": iso.receptivity_1H, "receptivity_13C": iso.receptivity_13C,
            "radioactive": iso.radioactive,
        })
    df = pd.DataFrame(rows, columns=[
        "nuclide", "element", "Z", "spin", "spin_label", "abundance_pct", "gamma_MHz_per_T",
        "quadrupole_fm2", "larmor_MHz", "ratio_to_1H_pct", "receptivity_1H",
        "receptivity_13C", "radioactive"])
    if sort_by == "frequency":
        df = df.sort_values(["larmor_MHz", "Z"], ascending=[False, True])
    elif sort_by == "receptivity":
        df = df.sort_values("receptivity_1H", ascending=False)
    elif sort_by == "Z":
        df = df.sort_values(["Z", "abundance_pct"], ascending=[True, False])
    else:
        raise ValueError("sort_by must be 'frequency', 'Z' or 'receptivity'.")
    df.attrs["B0_T"] = B0
    df.attrs["proton_MHz"] = field_to_proton_frequency(B0)
    return df.reset_index(drop=True)


def nearest_nuclides(frequency_MHz: float, B0_T: float | None = None, *,
                     proton_MHz: float | None = None, n: int = 5) -> pd.DataFrame:
    """The ``n`` nuclides whose Larmor frequency at this field is closest to
    ``frequency_MHz`` -- "what else resonates near here?" (probe tuning,
    an unexpected signal, a background from the probe's materials)."""
    df = larmor_table(B0_T, proton_MHz=proton_MHz)
    df["offset_MHz"] = df["larmor_MHz"] - float(frequency_MHz)
    order = df["offset_MHz"].abs().sort_values().index[:int(n)]
    return df.loc[order].reset_index(drop=True)


# ============================================================
# Figure
# ============================================================

def _text_colour_on(hex_colour: str) -> str:
    h = hex_colour.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return "white" if (0.299 * r + 0.587 * g + 0.114 * b) < 140 else "black"


def plot_periodic_table(ax=None, *, show: str = "mass", B0_T: float | None = None,
                        proton_MHz: float | None = None, highlight: Iterable | None = None,
                        spin: float | None = None, fill: bool = False,
                        figsize=(13.0, 7.4), title: str | None = None):
    """The NMR periodic table as a matplotlib figure.

    Each tile has a bar in the colour of its most abundant NMR isotope's
    spin (``fill=True`` fills the whole tile instead); grey tiles have
    no NMR-active isotope.

    show : "mass" | "larmor" | "receptivity" | "isotope"
        What the tile's bottom line says: atomic weight; the principal
        isotope's Larmor frequency in MHz at the given field (needs
        ``B0_T`` or ``proton_MHz``); its receptivity relative to 13C; or
        which isotope that is ("²⁷Al, 5/2").
    highlight : elements to outline (symbols, names, Z or nuclides).
    spin : dim every element without an isotope of this spin.

    Returns ``(fig, ax)``.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, Rectangle

    B0 = None
    if show == "larmor":
        B0 = _resolve_field(B0_T, proton_MHz)
    elif show not in ("mass", "receptivity", "isotope"):
        raise ValueError("show must be 'mass', 'larmor', 'receptivity' or 'isotope'.")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    highlighted = {element(h).symbol for h in (highlight or ())}

    w, h, gap = 1.0, 1.0, 0.06
    for el in _ELEMENT_LIST:
        row, col = grid_position(el)
        # The f-block sits a third of a tile below period 7, not a row.
        y = -(row - 1) * h if row < 9 else -(row - 2) * h - 0.35
        x = (col - 1) * w
        principal = el.principal_isotope
        colour = SPIN_COLOURS.get(principal.spin, "#999999") if principal else NO_NMR_COLOUR
        dim = spin is not None and not any(abs(i.spin - spin) < 1e-9 for i in el.isotopes)
        alpha = 0.22 if dim else 1.0
        face = colour if (fill and principal) else ("#ffffff" if principal else NO_NMR_COLOUR)
        ax.add_patch(FancyBboxPatch(
            (x + gap / 2, y - h + gap / 2), w - gap, h - gap,
            boxstyle="round,pad=0,rounding_size=0.05", facecolor=face,
            edgecolor="#0f6b72" if el.symbol in highlighted else "#9aa3a5",
            linewidth=2.0 if el.symbol in highlighted else 0.6, alpha=alpha, zorder=2))
        if principal and not fill:
            ax.add_patch(Rectangle((x + gap / 2, y - h + gap / 2), 0.07, h - gap,
                                   facecolor=colour, edgecolor="none", alpha=alpha, zorder=3))
        ink = _text_colour_on(face) if fill and principal else "#1b1e22"
        soft = ink if fill and principal else "#565c62"
        ax.text(x + 0.12, y - 0.1, str(el.Z), fontsize=5.5, color=soft, va="top",
                ha="left", alpha=alpha, zorder=4)
        ax.text(x + w / 2, y - 0.46, el.symbol, fontsize=11, fontweight="bold", color=ink,
                ha="center", va="center", alpha=alpha, zorder=4)
        if show == "mass":
            bottom = el.mass
        elif principal is None:
            bottom = "—"
        elif show == "larmor":
            bottom = f"{principal.larmor_MHz(B0):.1f}"
        elif show == "receptivity":
            r = principal.receptivity_13C
            bottom = f"{r:.2g}" if r > 0 else "0"
        else:
            sup = str(principal.A).translate(str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹"))
            bottom = f"{sup}{el.symbol} {principal.spin_label}"
        ax.text(x + w / 2, y - 0.8, bottom, fontsize=5.5, color=soft, ha="center",
                va="center", alpha=alpha, zorder=4)

    # Where the f-block belongs, in period 6 and 7's group-3 slot.
    for row, text in ((6, "57–71"), (7, "89–103")):
        y = -(row - 1) * h
        ax.add_patch(FancyBboxPatch(
            (2 + gap / 2, y - h + gap / 2), w - gap, h - gap,
            boxstyle="round,pad=0,rounding_size=0.05", facecolor="none",
            edgecolor="#9aa3a5", linewidth=0.6, linestyle=(0, (2, 2)), zorder=2))
        ax.text(2.5, y - 0.5, text, fontsize=5.5, color="#9a9f9f", ha="center", va="center")

    # Legend: one swatch per spin that colours at least one tile.
    shown = sorted({e.principal_isotope.spin for e in _ELEMENT_LIST if e.principal_isotope})
    lx, ly = 2.3, -0.25
    ax.text(lx, ly + 0.05, "Spin of the most abundant NMR isotope", fontsize=7,
            color="#565c62", va="bottom")
    for k, s in enumerate(shown + [None]):
        cx = lx + (k % 7) * 1.3
        cy = ly - 0.45 - (k // 7) * 0.45
        ax.add_patch(Rectangle((cx, cy - 0.15), 0.28, 0.28,
                               facecolor=SPIN_COLOURS[s] if s is not None else NO_NMR_COLOUR,
                               edgecolor="#9aa3a5", linewidth=0.4))
        ax.text(cx + 0.36, cy, f"I = {spin_label(s)}" if s is not None else "none",
                fontsize=6.5, va="center", color="#1b1e22")

    if title is None:
        title = {"mass": "NMR periodic table",
                 "larmor": (f"Larmor frequency (MHz) of the most abundant NMR isotope at "
                            f"{B0:.3f} T ({field_to_proton_frequency(B0) if B0 else 0:.0f} MHz ¹H)"
                            if B0 is not None else ""),
                 "receptivity": "Receptivity relative to ¹³C of the most abundant NMR isotope",
                 "isotope": "Most abundant NMR-active isotope and its spin"}[show]
    if title:
        ax.set_title(title, fontsize=10, loc="left")
    ax.set_xlim(-0.1, 18.1)
    ax.set_ylim(-9.5, 0.3)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax
