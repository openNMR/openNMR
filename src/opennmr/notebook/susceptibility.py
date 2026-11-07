"""
Magnetic susceptibility in the units people have (explicit user request,
2026-11-07: "supply the susceptibility in a variety of units -- emu mol⁻¹,
cgs, or something else"), converted to the SI volume susceptibility the BMS
simulator uses.

The SI volume susceptibility χ_v is dimensionless (M = χ_v H). Every other
common form converts to it with at most a density or a molar volume:

=================  ===============================  ==========================
unit                what it is                       χ_v (SI) =
=================  ===============================  ==========================
``si_volume``       SI volume (dimensionless)        χ
``si_volume_ppm``   the same, in ppm                 χ × 10⁻⁶
``cgs_volume``      cgs/Gaussian volume              4π χ
                    (emu cm⁻³ Oe⁻¹, dimensionless)
``cgs_mass``        emu g⁻¹ Oe⁻¹ (= cm³ g⁻¹)         4π χ ρ        (ρ in g cm⁻³)
``si_mass``         m³ kg⁻¹                          χ ρ           (ρ in kg m⁻³)
``cgs_molar``       emu mol⁻¹ Oe⁻¹ (= cm³ mol⁻¹)     4π χ / V_m    (V_m in cm³ mol⁻¹)
``si_molar``        m³ mol⁻¹                         χ / V_m       (V_m in m³ mol⁻¹)
``mu_eff``          μ_eff (μ_B) per paramagnetic     4π χ_mol / V_m, with
                    ion, at T (Curie–Weiss θ)        χ_mol = n C/(T − θ) + χ_dia
=================  ===============================  ==========================

C = N_A μ_B² μ_eff² / (3 k_B) = 0.12505 μ_eff² emu K mol⁻¹ (CODATA 2018
constants); n paramagnetic ions per formula unit; χ_dia an optional
diamagnetic (core) term in emu mol⁻¹.

The molar volume of one formula unit, V_m = M/ρ, or N_A V_cell / Z from
the unit cell (V_cell in Å³, Z formula units per cell).

Use the density of the solid itself (the crystal density), not a tap or
packing density: the packing is what the BMS simulation builds, and its
volume fraction is applied there.

Check: water, χ_mol = −12.97 × 10⁻⁶ cm³ mol⁻¹, M = 18.015 g mol⁻¹,
ρ = 0.998 g cm⁻³ → χ_v = −9.03 × 10⁻⁶ (SI), the tabulated value.
"""

from __future__ import annotations

import numpy as np

N_A = 6.02214076e23                # mol⁻¹ (exact)
MU_B_CGS = 9.2740100783e-21        # erg G⁻¹
K_B_CGS = 1.380649e-16             # erg K⁻¹ (exact)
#: N_A μ_B² / (3 k_B): emu K mol⁻¹ per μ_B² (≈ 0.12505; often written 1/7.997)
CURIE_CGS = N_A * MU_B_CGS ** 2 / (3.0 * K_B_CGS)

UNITS = {
    "si_volume": "SI volume (dimensionless)",
    "si_volume_ppm": "SI volume, ppm (× 10⁻⁶)",
    "cgs_volume": "cgs volume (emu cm⁻³ Oe⁻¹)",
    "cgs_mass": "cgs mass (emu g⁻¹ Oe⁻¹ = cm³ g⁻¹)",
    "si_mass": "SI mass (m³ kg⁻¹)",
    "cgs_molar": "cgs molar (emu mol⁻¹ Oe⁻¹ = cm³ mol⁻¹)",
    "si_molar": "SI molar (m³ mol⁻¹)",
    "mu_eff": "μ_eff (μ_B) per paramagnetic ion, at T",
}

#: What each unit needs besides the value: "density" or "molar_volume".
NEEDS = {
    "si_volume": (), "si_volume_ppm": (), "cgs_volume": (),
    "cgs_mass": ("density",), "si_mass": ("density",),
    "cgs_molar": ("molar_volume",), "si_molar": ("molar_volume",),
    "mu_eff": ("molar_volume", "temperature"),
}


def molar_volume(*, molar_mass=None, density=None, cell_volume=None, Z=None) -> float:
    """The volume of one mole of formula units, cm³ mol⁻¹: ``molar_mass``
    (g mol⁻¹) / ``density`` (g cm⁻³), or N_A ``cell_volume`` (Å³) / ``Z``."""
    if cell_volume is not None or Z is not None:
        if not cell_volume or not Z or cell_volume <= 0 or Z <= 0:
            raise ValueError("From the unit cell: give its volume (Å³) and Z, the formula "
                             "units per cell (both positive).")
        return N_A * float(cell_volume) * 1e-24 / float(Z)
    if not molar_mass or not density or molar_mass <= 0 or density <= 0:
        raise ValueError("A molar unit needs the molar volume: the molar mass (g mol⁻¹) and the "
                         "density (g cm⁻³) of the solid, or its unit-cell volume and Z.")
    return float(molar_mass) / float(density)


def curie_molar(mu_eff, T, theta=0.0, n_ions=1.0, chi_dia=0.0):
    """χ_mol (emu mol⁻¹) of n ions of ``mu_eff`` (μ_B) per formula unit at
    ``T`` (K), Curie–Weiss ``theta`` (K), plus ``chi_dia`` (emu mol⁻¹)."""
    mu = np.asarray(mu_eff, dtype=float)
    if T is None or T - theta <= 0:
        raise ValueError(f"μ_eff needs a temperature above θ (T = {T}, θ = {theta:g} K): the "
                         "Curie–Weiss law holds only above the ordering temperature.")
    return n_ions * CURIE_CGS * mu ** 2 / (float(T) - float(theta)) + float(chi_dia)


def to_si_volume(value, unit="si_volume", *, density=None, molar_mass=None, cell_volume=None,
                 Z=None, T=None, theta=0.0, n_ions=1.0, chi_dia=0.0):
    """``value`` (a number or the three principal values) in ``unit``
    (:data:`UNITS`) as the SI volume susceptibility (dimensionless, the
    same shape). ``density`` in g cm⁻³ (the solid's own); the molar volume
    from ``molar_mass`` and ``density`` or from ``cell_volume`` (Å³) and
    ``Z``; for ``mu_eff``, ``T``, ``theta``, ``n_ions`` and ``chi_dia``
    (emu mol⁻¹)."""
    if unit not in UNITS:
        raise ValueError(f"Unknown unit {unit!r}: one of {', '.join(UNITS)}.")
    x = np.asarray(value, dtype=float)
    if unit == "si_volume":
        out = x
    elif unit == "si_volume_ppm":
        out = x * 1e-6
    elif unit == "cgs_volume":
        out = 4.0 * np.pi * x
    elif unit in ("cgs_mass", "si_mass"):
        if not density or density <= 0:
            raise ValueError("A mass susceptibility needs the density of the solid (g cm⁻³).")
        out = 4.0 * np.pi * x * density if unit == "cgs_mass" else x * density * 1e3
    else:
        Vm = molar_volume(molar_mass=molar_mass, density=density, cell_volume=cell_volume, Z=Z)
        if unit == "cgs_molar":
            out = 4.0 * np.pi * x / Vm
        elif unit == "si_molar":
            out = x / (Vm * 1e-6)
        else:
            out = 4.0 * np.pi * curie_molar(x, T, theta, n_ions, chi_dia) / Vm
    return float(out) if np.ndim(out) == 0 else out


def describe(value, unit, **kw) -> str:
    """One line for the log: the input and the SI volume value it gives."""
    chi = np.atleast_1d(to_si_volume(value, unit, **kw))
    vals = ", ".join(f"{v:.4g}" for v in np.atleast_1d(np.asarray(value, float)))
    extra = []
    if kw.get("density"):
        extra.append(f"ρ = {kw['density']:g} g cm⁻³")
    if kw.get("molar_mass"):
        extra.append(f"M = {kw['molar_mass']:g} g mol⁻¹")
    if kw.get("cell_volume"):
        extra.append(f"V_cell = {kw['cell_volume']:g} Å³, Z = {kw.get('Z'):g}")
    if unit == "mu_eff":
        extra.append(f"T = {kw.get('T'):g} K, θ = {kw.get('theta', 0):g} K, "
                     f"{kw.get('n_ions', 1):g} ion(s) per formula unit")
        if kw.get("chi_dia"):
            extra.append(f"χ_dia = {kw['chi_dia']:g} emu mol⁻¹")
    tail = f" ({'; '.join(extra)})" if extra else ""
    return (f"χ = {vals} {UNITS[unit]}{tail} → χ_v (SI) = "
            + ", ".join(f"{v:.4g}" for v in chi))
