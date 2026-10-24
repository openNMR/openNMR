"""Paramagnetic shifts and quadrupolar tensors from CRYSTAL calculations.

CRYSTAL (``PROPERTIES`` output, any extension: ``.out``, ``.outp``, ``.log``
...) prints three things that matter for the NMR of a paramagnetic solid:

* ``SPIN DENSITY AT THE NUCLEAR POSITIONS`` (bohr⁻³) -- the Fermi contact
  interaction;
* ``HYPERFINE ELECTRON-NUCLEAR SPIN INTERACTION TENSOR`` -- the traceless
  anisotropic (spin dipolar) part, in MHz for each isotope, with its
  nuclear g factor;
* ``TRACELESS ELECTRIC FIELD GRADIENT TENSOR`` (atomic units) -- the
  quadrupolar interaction (C_Q, η_Q).

The calculation is of the fully spin-polarised (ferromagnetic) state. At
the temperature and field of the NMR experiment the electron spin is only
partly polarised, and the hyperfine interaction is scaled by

    φ = ⟨S_z⟩ / S  =  B0 μ_eff² μ_B / (3 k_B g S (T − θ))      (Curie–Weiss)

(``S``: spin quantum number per formula unit; ``θ``: Weiss constant, < 0
antiferromagnetic; ``μ_eff`` in μ_B). Equivalently, from a measured
molar susceptibility, φ = χ_M B0 / (μ0 N_A g μ_B S). The shift of a
nucleus (diamagnetic convention, δ > 0 to high frequency) is then

    δ = (φ / 2) · A / ν0 ,

with ``A`` the hyperfine coupling (Hz) and ν0 the Larmor frequency, so
that the Fermi contact shift is δ_FC = (φ/3) μ0 g_e μ_B ρ(0) / (a0³ B0)
and the spin dipolar shift tensor is δ_dip = (φ/2) A_dip / ν0. Both are
independent of the isotope (A and ν0 both scale with the nuclear g factor)
and, in the Curie–Weiss regime, of the field (φ ∝ B0).

What this gives is the *hyperfine* shift: the orbital (diamagnetic) shift
is not in the calculation. Add it with ``delta_ref`` (the orbital shift of
that nucleus in a diamagnetic analogue, say), or leave it at 0.

The results are put into the same atoms dictionary as the other codes in
:mod:`opennmr.notebook.dft_nmr`, with σ = −δ (so ``dft_sites`` with
``sigma_ref=0`` returns δ): use ``dft_sites(sample, paramagnetic={...})``
or :func:`apply_scaling` directly.
"""

from __future__ import annotations

import copy
import math
import os
import re

import numpy as np

__all__ = [
    "read_crystal", "is_crystal", "read_spin_density_table", "magnetic_scaling",
    "curie_weiss_phi", "phi_from_susceptibility", "apply_scaling", "fermi_contact_shift",
    "crystal_summary", "SCALING_ROUTES",
]

# ---------------------------------------------------------------------------
# Constants (CODATA 2018)
# ---------------------------------------------------------------------------

MU_0 = 1.25663706212e-6        # N A^-2
MU_B = 9.2740100783e-24        # J T^-1
MU_N = 5.0507837461e-27        # J T^-1
K_B = 1.380649e-23             # J K^-1
N_A = 6.02214076e23            # mol^-1
H_PLANCK = 6.62607015e-34      # J s
A_0 = 5.29177210903e-11        # m
G_E = 2.00231930436256         # free-electron g factor (the contact term)
#: Atomic unit of electric-field gradient, V m^-2 (CODATA 2018).
AU_EFG = 9.7173624292e21
#: Default g factor in φ (the notebook scripts use 2.0).
G_DEFAULT = 2.0
#: Proton Larmor frequency per tesla (MHz), for converting a spectrometer
#: ¹H frequency to a field.
PROTON_MHZ_PER_T = 42.577478518

SCALING_ROUTES = {
    "phi": "A scaling factor φ = ⟨S_z⟩/S you give directly",
    "susceptibility": "A molar magnetic susceptibility χ_M (experimental or computed) and the "
                      "spin S",
    "curie_weiss": "A Weiss constant θ, an effective moment μ_eff and the spin S "
                   "(Curie–Weiss law)",
}

_FLOAT = r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[EeDd][-+]?\d+)?"
_NUM = re.compile(_FLOAT)


def _f(text: str) -> float:
    return float(text.replace("D", "E").replace("d", "e"))


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def is_crystal(text: str) -> bool:
    """True if ``text`` (the start of a file) is CRYSTAL output."""
    return bool(re.search(r"\*\s+CRYSTAL(?:\d{2})?\s+\*|CRYSTAL\d{2}|CRYSTAL\s+-\s+PROPERTIES",
                          text))


_ATOM_ROW = re.compile(r"^\s*(\d+)\s+(\d+)\s+([A-Za-z]{1,3})\s+(\d+)\s+(" + _FLOAT + r")\s+("
                       + _FLOAT + r")\s+(" + _FLOAT + r")\s+" + _FLOAT + r"\s+" + _FLOAT + r"\s*$")
_SPIN_ROW = re.compile(r"^\s*(\d+)\s+(\d+)\s+([A-Za-z]{1,3})\s+(" + _FLOAT + r")\s+(" + _FLOAT
                       + r")\s+(" + _FLOAT + r")\s+(" + _FLOAT + r")\s*$")
_HF_HEAD = re.compile(r"^\s*(\d+)\s+(\d+)\s+([A-Za-z]{1,3})\s+(\d+)\s+(" + _FLOAT + r")\s*$")
_EFG_HEAD = re.compile(r"POINT\s+(\d+)\s+POSITION")
_TENSOR_LINE = re.compile(r"(XX|YY|ZZ|XY|XZ|YZ)\s+(" + _FLOAT + r")")


def _symbol(sym: str) -> str:
    return sym[:1].upper() + sym[1:].lower()


def _tensor(line: str) -> np.ndarray:
    v = {k: _f(x) for k, x in _TENSOR_LINE.findall(line)}
    if len(v) != 6:
        raise ValueError(f"Cannot read the tensor line {line.strip()!r}.")
    return np.array([[v["XX"], v["XY"], v["XZ"]],
                     [v["XY"], v["YY"], v["YZ"]],
                     [v["XZ"], v["YZ"], v["ZZ"]]])


def _parse(text: str) -> dict:
    """One CRYSTAL output: every block this module uses."""
    lines = text.splitlines()
    out = {"lattice": None, "atoms": {}, "spin": {}, "hf": {}, "efg": {}, "energy": None,
           "title": "", "functional": "", "hybrid": None, "spin_total": None,
           "conv_mT": None, "conv_MHz": None, "n_spin_sections": 0}
    mode = None
    i = 0
    n = len(lines)
    current_hf = None
    current_efg = None
    while i < n:
        line = lines[i]
        s = line.strip()
        i += 1
        if not s:
            continue
        if "DIRECT LATTICE VECTOR COMPONENTS" in s and out["lattice"] is None:
            rows = [[_f(x) for x in _NUM.findall(lines[i + k])[:3]] for k in range(3)
                    if i + k < n]
            if len(rows) == 3 and all(len(r) == 3 for r in rows):
                out["lattice"] = np.array(rows)
            continue
        if "ATOM N.AT." in s and "SHELL" in s and not out["atoms"]:
            mode = "atoms"
            continue
        if "SPIN POLARIZED SYSTEM: SUMMED SPIN DENSITY" in s:
            m = _NUM.findall(s.split("DENSITY")[-1])
            if m:
                out["spin_total"] = _f(m[0])
            continue
        if s.startswith("TOTAL ENERGY") and out["energy"] is None:
            m = re.search(r"TOTAL ENERGY\s+(" + _FLOAT + ")", s)
            if m:
                out["energy"] = _f(m.group(1))
            continue
        if "FUNCTIONAL:" in s and not out["functional"]:
            out["functional"] = s.split("FUNCTIONAL:", 1)[1].strip()
            continue
        if "PERCENTAGE OF FOCK EXCHANGE" in s and out["hybrid"] is None:
            m = _NUM.findall(s.split("EXCHANGE")[-1])
            if m:
                out["hybrid"] = _f(m[-1])
            continue
        if s == "SPIN DENSITY AT THE NUCLEAR POSITIONS":
            mode = "spin"
            out["n_spin_sections"] += 1
            continue
        if s == "HYPERFINE ELECTRON-NUCLEAR SPIN INTERACTION TENSOR":
            mode = "hf"
            continue
        if s == "TRACELESS ELECTRIC FIELD GRADIENT TENSOR":
            mode = "efg"
            continue
        if s.startswith("TOTAL DENSITY AT THE NUCLEAR") or s.startswith("CONVERSION FACTORS") \
                or s.startswith("TOTAL ELECTROSTATIC POTENTIAL"):
            if mode == "spin" and s.startswith("CONVERSION FACTORS"):
                pass                      # the spin section's factors follow; keep reading them
            else:
                mode = None
            continue
        if s.startswith("AN (MILLI T)"):
            m = re.search(r"(" + _FLOAT + r")\s*\*\s*GN", s)
            if m:
                out["conv_mT"] = _f(m.group(1))
            continue
        if s.startswith("AN (MHZ)"):
            m = re.search(r"(" + _FLOAT + r")\s*\*\s*AN", s)
            if m:
                out["conv_MHz"] = _f(m.group(1))
            continue
        if mode == "atoms":
            if s.startswith("*"):
                if out["atoms"]:
                    mode = None
                continue
            m = _ATOM_ROW.match(line)
            if m:
                point = int(m.group(1))
                out["atoms"][point] = {"Z": int(m.group(2)) % 100,
                                       "species": _symbol(m.group(3)),
                                       "position": np.array([_f(m.group(5)), _f(m.group(6)),
                                                             _f(m.group(7))])}
            continue
        if mode == "spin":
            m = _SPIN_ROW.match(line)
            if m:
                out["spin"][int(m.group(1))] = {"species": _symbol(m.group(3)),
                                                "rho": _f(m.group(7))}
            continue
        if mode == "hf":
            if s.startswith("POINT") and "MASS NO" in s:
                current_hf = "head"
                continue
            if current_hf == "head":
                m = _HF_HEAD.match(line)
                if m:
                    current_hf = (int(m.group(1)), int(m.group(4)), _f(m.group(5)))
                    out["hf"].setdefault(int(m.group(1)), {})[int(m.group(4))] = {
                        "g_N": _f(m.group(5)), "tensor": None}
                continue
            if isinstance(current_hf, tuple) and "TENSOR IN ORIGINAL CARTESIAN AXES" in s:
                while i < n and not lines[i].strip().startswith("XX"):
                    i += 1
                if i < n:
                    point, mass, _g = current_hf
                    out["hf"][point][mass]["tensor"] = _tensor(lines[i])
                    i += 1
                current_hf = None
            continue
        if mode == "efg":
            m = _EFG_HEAD.search(s)
            if m:
                current_efg = int(m.group(1))
                continue
            if current_efg is not None and "TENSOR IN ORIGINAL CARTESIAN AXES" in s:
                while i < n and not lines[i].strip().startswith("XX"):
                    i += 1
                if i < n:
                    out["efg"][current_efg] = _tensor(lines[i])
                    i += 1
                current_efg = None
            continue
    # the title: the line after the second row of stars following "STARTING DATE"
    for k, raw in enumerate(lines[:400]):
        if "CRYSTAL - PROPERTIES" in raw or "CRYSTAL - SCF" in raw:
            j = k - 1
            while j >= 0 and not lines[j].strip():
                j -= 1
            if j >= 0 and not lines[j].strip().startswith("*"):
                out["title"] = lines[j].strip()
            break
    return out


def read_spin_density_table(path) -> dict:
    """A file holding only the ``SPIN DENSITY AT THE NUCLEAR POSITIONS``
    table (the ``spin_dens_table`` file made with ``grep -A`` from a
    CRYSTAL output, with or without its header lines): ``{point: {"species",
    "rho"}}``."""
    with open(path, "r", errors="replace") as f:
        text = f.read()
    rows = {}
    for line in text.splitlines():
        m = _SPIN_ROW.match(line)
        if m:
            rows[int(m.group(1))] = {"species": _symbol(m.group(3)), "rho": _f(m.group(7))}
    if not rows:
        raise ValueError(f"{os.path.basename(str(path))}: no spin-density rows (POINT ATOM X Y Z "
                         "BOHR**(-3)) found.")
    return rows


def read_crystal(path, *more_paths, spin_density_file=None) -> dict:
    """A CRYSTAL output (any extension), as a calculation dictionary like the
    other codes' (:func:`opennmr.notebook.dft_nmr.read_calculation`).

    ``more_paths`` : further CRYSTAL outputs of the *same* calculation to
    read the other blocks from (the hyperfine, EFG and spin-density blocks
    can be in separate runs). ``spin_density_file`` : a bare
    ``spin_dens_table`` file, if the spin density is not in the output.

    Each atom (``label`` = element + CRYSTAL's atom number, e.g. ``Na15``)
    has ``efg`` (V/m², from the printed atomic units), ``rho`` (spin
    density at the nucleus, bohr⁻³ or None), ``hf`` (``{mass number:
    {"g_N", "tensor"}}``, the traceless hyperfine tensor in MHz) and
    ``sigma`` None until :func:`apply_scaling` sets it from the scaling
    factor."""
    paths = [path, *more_paths]
    parsed = []
    for p in paths:
        with open(p, "r", errors="replace") as f:
            text = f.read()
        if not is_crystal(text[:200000]) and not is_crystal(text):
            raise ValueError(f"{os.path.basename(str(p))}: not CRYSTAL output.")
        parsed.append(_parse(text))
    first = parsed[0]
    atoms_tbl = next((p["atoms"] for p in parsed if p["atoms"]), {})
    if not atoms_tbl:
        raise ValueError(f"{os.path.basename(str(path))}: no atom table (ATOM N.AT. SHELL X(A) "
                         "Y(A) Z(A)) in the CRYSTAL output.")
    spin, hf, efg = {}, {}, {}
    for p in parsed:
        spin.update(p["spin"])
        for k, v in p["hf"].items():
            hf.setdefault(k, {}).update(v)
        efg.update(p["efg"])
    notes = []
    if spin_density_file is not None:
        spin.update(read_spin_density_table(spin_density_file))
    atoms = []
    for point in sorted(atoms_tbl):
        a = atoms_tbl[point]
        label = f"{a['species']}{point}"
        rec = {"species": a["species"], "index": point, "label": label,
               "position": a["position"], "sigma": None,
               "efg": None if point not in efg else efg[point] * AU_EFG,
               "efg_au": None if point not in efg else efg[point],
               "rho": spin.get(point, {}).get("rho"),
               "hf": {m: {"g_N": d["g_N"], "tensor": d["tensor"]}
                      for m, d in hf.get(point, {}).items() if d["tensor"] is not None}}
        atoms.append(rec)
    if not spin:
        notes.append("No spin density at the nuclei in the file(s): the Fermi contact shift "
                     "cannot be computed (give spin_density_file=, or the PROPERTIES run that "
                     "prints SPIN DENSITY AT THE NUCLEAR POSITIONS).")
    if not hf:
        notes.append("No hyperfine tensors in the file(s): the spin dipolar shift cannot be "
                     "computed.")
    if not efg:
        notes.append("No EFG in the file(s): no quadrupolar coupling.")
    energy = next((p["energy"] for p in parsed if p["energy"] is not None), None)
    functional = next((p["functional"] for p in parsed if p["functional"]), "")
    hybrid = next((p["hybrid"] for p in parsed if p["hybrid"] is not None), None)
    method = {"code": "CRYSTAL", "functional": functional
              + (f", {hybrid:g}% Fock exchange" if hybrid is not None else ""),
              "cutoff": "", "pseudopotentials": {}, "periodic": True}
    calc = {"code": "CRYSTAL", "path": str(path), "lattice": next(
        (p["lattice"] for p in parsed if p["lattice"] is not None), None),
        "atoms": atoms, "notes": notes, "info": {"title": first["title"]}, "method": method,
        "g0": "none", "paramagnetic": True, "energy_Eh": energy,
        "energy_meV": None if energy is None else energy * 27211.386245988,
        "spin_total": next((p["spin_total"] for p in parsed if p["spin_total"] is not None),
                           None),
        "conversion": {"mT_per_gN_rho": next((p["conv_mT"] for p in parsed
                                              if p["conv_mT"]), None),
                       "MHz_per_mT": next((p["conv_MHz"] for p in parsed
                                           if p["conv_MHz"]), None)},
        "scaling": None}
    return calc


# ---------------------------------------------------------------------------
# The magnetic scaling factor
# ---------------------------------------------------------------------------

def curie_weiss_phi(B0, T, theta, spin, mu_eff, g=G_DEFAULT) -> float:
    """φ = B0 μ_eff² μ_B / (3 k_B g S (T − θ)): the thermal average of the
    electron spin along B0, ⟨S_z⟩/S, in the Curie–Weiss regime.

    ``B0`` in tesla, ``T`` and ``theta`` in kelvin (θ < 0:
    antiferromagnetic), ``spin`` the spin quantum number S per formula
    unit, ``mu_eff`` in μ_B per formula unit."""
    _check_positive(B0=B0, T=T, spin=spin, mu_eff=mu_eff, g=g)
    if T - theta <= 0:
        raise ValueError(f"T − θ = {T - theta:g} K is not positive: the Curie–Weiss law holds "
                         "only above the Weiss temperature (T > θ); the sample is ordered at "
                         "this temperature, so φ cannot be estimated this way.")
    return float(B0 * mu_eff ** 2 * MU_B / (3.0 * K_B * g * spin * (T - theta)))


def phi_from_susceptibility(chi, B0, spin, g=G_DEFAULT, units="cm3/mol") -> float:
    """φ = χ_M B0 / (μ0 N_A g μ_B S) from the molar susceptibility χ_M of
    the sample at the field and temperature of the experiment.

    ``units``: ``"cm3/mol"`` (cgs, emu/mol: the usual experimental unit;
    SI = 4π × 10⁻⁶ × this, in m³/mol) or ``"m3/mol"`` (SI)."""
    _check_positive(B0=B0, spin=spin, g=g)
    if units in ("cm3/mol", "emu/mol", "cgs"):
        chi_si = 4.0 * math.pi * 1e-6 * float(chi)
    elif units in ("m3/mol", "si", "SI"):
        chi_si = float(chi)
    else:
        raise ValueError("chi units must be 'cm3/mol' (cgs, emu/mol) or 'm3/mol' (SI).")
    if chi_si <= 0:
        raise ValueError("χ_M must be positive for a paramagnet (a negative value means the "
                         "diamagnetic background was not removed).")
    return float(chi_si * B0 / (MU_0 * N_A * g * MU_B * spin))


def _check_positive(**kw) -> None:
    for name, value in kw.items():
        if value is None:
            raise ValueError(f"{name} is required.")
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a positive number (got {value}).")


def magnetic_scaling(phi=None, *, chi=None, chi_units="cm3/mol", theta=None, mu_eff=None,
                     spin=None, T=298.0, B0=11.7, g=G_DEFAULT) -> dict:
    """The scaling factor φ from whichever the user has, in this order of
    preference: ``phi`` itself; a susceptibility ``chi`` (with ``spin``,
    ``B0``); a Weiss constant ``theta``, ``mu_eff`` and ``spin`` (with
    ``T``, ``B0``).

    Returns ``{"phi", "route", "B0", "T", "text", ...inputs}``. φ is
    proportional to ``B0``, so a φ you give directly belongs to the field
    ``B0`` you give with it."""
    _check_positive(B0=B0)
    if phi is not None:
        if not np.isfinite(phi) or phi == 0:
            raise ValueError("φ must be a non-zero number.")
        route, value = "phi", float(phi)
        text = f"φ = {value:.6g} (given), at B0 = {B0:g} T"
    elif chi is not None:
        value = phi_from_susceptibility(chi, B0, spin, g=g, units=chi_units)
        route = "susceptibility"
        text = (f"φ = χ_M B0 / (μ0 N_A g μ_B S) = {value:.6g} from χ_M = {chi:g} {chi_units}, "
                f"S = {spin:g}, g = {g:g}, B0 = {B0:g} T")
    elif theta is not None or mu_eff is not None:
        if theta is None or mu_eff is None or spin is None:
            raise ValueError("The Curie–Weiss route needs all of the Weiss constant θ, the "
                             "effective moment μ_eff and the spin S (and T, B0).")
        value = curie_weiss_phi(B0, T, theta, spin, mu_eff, g=g)
        route = "curie_weiss"
        text = (f"φ = B0 μ_eff² μ_B / (3 k_B g S (T − θ)) = {value:.6g} for θ = {theta:g} K, "
                f"μ_eff = {mu_eff:g} μ_B, S = {spin:g}, T = {T:g} K, g = {g:g}, B0 = {B0:g} T")
    else:
        raise ValueError("Give φ, or a magnetic susceptibility (with the spin S), or a Weiss "
                         "constant θ with μ_eff and S.")
    return {"phi": float(value), "route": route, "B0": float(B0), "T": float(T), "g": float(g),
            "chi": chi, "chi_units": chi_units, "theta": theta, "mu_eff": mu_eff, "spin": spin,
            "text": text}


# ---------------------------------------------------------------------------
# Shifts
# ---------------------------------------------------------------------------

def fermi_contact_shift(rho, phi, B0) -> float:
    """Fermi contact shift (ppm) for a spin density ``rho`` (bohr⁻³) at the
    nucleus: δ_FC = 10⁶ (φ/3) μ0 g_e μ_B ρ / (a0³ B0)."""
    return float(1e6 * (phi / 3.0) * MU_0 * G_E * MU_B * rho / (A_0 ** 3 * B0))


def _dipolar_shift_tensor(tensor_MHz, g_N, phi, B0) -> np.ndarray:
    """δ_dip (ppm) = 10⁶ (φ/2) A / ν0 with ν0 = g_N μ_N B0 / h (signed, so
    the sign of g_N cancels)."""
    nu0 = g_N * MU_N * B0 / H_PLANCK
    t = np.asarray(tensor_MHz, dtype=float)
    t = t - np.trace(t) / 3.0 * np.eye(3)       # printed traceless only to its rounding
    return 1e6 * 0.5 * phi * (t * 1e6) / nu0


def apply_scaling(calc, scaling=None, *, components="both", **kwargs) -> dict:
    """A copy of ``calc`` with each atom's shielding ``sigma`` = −δ_hyperfine
    (ppm tensor: Fermi contact + spin dipolar), so ``dft_sites(...,
    sigma_ref=0)`` returns the paramagnetic shift.

    ``scaling`` : the dictionary of :func:`magnetic_scaling`, or give its
    arguments here (``phi=``, ``chi=``, ``theta=`` ...). ``components`` :
    ``"both"``, ``"contact"`` (Fermi contact only) or ``"dipolar"``.

    Per atom, ``delta_fc`` (ppm), ``delta_dip`` (3×3 tensor, ppm) and
    ``delta_hf`` (the sum) are kept for inspection."""
    if calc.get("code") != "CRYSTAL":
        raise ValueError("apply_scaling is for CRYSTAL calculations.")
    if components not in ("both", "contact", "dipolar"):
        raise ValueError("components must be 'both', 'contact' or 'dipolar'.")
    if scaling is None:
        scaling = magnetic_scaling(**kwargs)
    elif kwargs:
        raise ValueError("Give a scaling dictionary or its arguments, not both.")
    phi, B0 = scaling["phi"], scaling["B0"]
    out = copy.deepcopy(calc)
    n_fc = n_dip = 0
    for a in out["atoms"]:
        fc = None if a.get("rho") is None else fermi_contact_shift(a["rho"], phi, B0)
        dip = None
        if a.get("hf"):
            mass = sorted(a["hf"])[0]
            h = a["hf"][mass]
            if abs(h["g_N"]) > 0:
                dip = _dipolar_shift_tensor(h["tensor"], h["g_N"], phi, B0)
        delta = np.zeros((3, 3))
        have = False
        if components in ("both", "contact") and fc is not None:
            delta += fc * np.eye(3)
            have = True
            n_fc += 1
        if components in ("both", "dipolar") and dip is not None:
            delta += dip
            have = True
            n_dip += 1
        a["delta_fc"], a["delta_dip"] = fc, dip
        a["delta_hf"] = delta if have else None
        a["sigma"] = -delta if have else None
    out["scaling"] = dict(scaling, components=components)
    out["notes"] = list(out["notes"])
    out["notes"].append(
        f"Paramagnetic (hyperfine) shift from CRYSTAL: {scaling['text']}. "
        + {"both": "Fermi contact + spin dipolar", "contact": "Fermi contact only",
           "dipolar": "spin dipolar only"}[components]
        + f" ({n_fc} contact, {n_dip} dipolar). This is the hyperfine shift only: add the orbital "
          "(diamagnetic) shift through δ_ref.")
    if out.get("spin_total") is not None:
        out["notes"].append(f"The CRYSTAL calculation has {out['spin_total']:g} unpaired "
                            "electron(s) per cell (the fully polarised state that φ scales).")
    return out


def crystal_summary(calc) -> str:
    """One paragraph on what was read."""
    n = len(calc["atoms"])
    n_rho = sum(a.get("rho") is not None for a in calc["atoms"])
    n_hf = sum(bool(a.get("hf")) for a in calc["atoms"])
    n_efg = sum(a.get("efg") is not None for a in calc["atoms"])
    return (f"CRYSTAL: {n} atoms; spin density for {n_rho}, hyperfine tensors for {n_hf}, "
            f"EFG for {n_efg}.")
