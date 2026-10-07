"""
DFT-calculated NMR parameters into simulations (explicit user request,
2026-10-06): read the shielding and electric-field-gradient tensors of a
sample and a reference calculation -- CASTEP ``.magres`` or VASP ``OUTCAR``
-- reference the shifts, work out the CSA and quadrupolar parameters and
their relative orientation, and simulate the spectrum of any set of sites.

    from opennmr.notebook import dft_nmr
    sample = dft_nmr.read_calculation("sample.magres")
    ref = dft_nmr.read_calculation("NaCl.magres")
    model = dft_nmr.dft_sites(sample, ref, element="Na", delta_ref=7.21)
    print(dft_nmr.format_table(model))
    spec = dft_nmr.simulate_dft(model, larmor_Hz=105.8e6, mas_Hz=12.5e3)
    dft_nmr.plot_dft_tensors(model["atoms"][0], sample, frame="cell")

What is computed, and the conventions
-------------------------------------
**Shifts.** Every code gives absolute shielding tensors σ. The shift
tensor is

    δ = δ_ref + m (σ_ref − σ)          (m = 1 by default)

applied to the whole (symmetrised) tensor, where σ_ref is the isotropic
shielding the *reference* calculation gives for the same element
(averaged over the reference sites you choose) and δ_ref is that
compound's *experimental* shift on your scale (e.g. solid NaCl for
²³Na). With m ≠ 1 the expression is the linear regression of a set of
compounds, δ = δ_ref − m (σ − σ_ref). Only the symmetric part of σ enters
an NMR spectrum at high field; the antisymmetric part is reported but
not used.

**CSA.** From the principal values of δ:

* Herzfeld–Berger (Mason): δ11 ≥ δ22 ≥ δ33, span Ω = δ11 − δ33,
  skew κ = 3(δ22 − δiso)/Ω.
* Haeberlen: |δzz − δiso| ≥ |δxx − δiso| ≥ |δyy − δiso|, reduced
  anisotropy δ_aniso = δzz − δiso, η_CSA = (δyy − δxx)/δ_aniso, and the
  anisotropy Δδ = δzz − (δxx + δyy)/2 = 1.5 δ_aniso.

The simulation uses the Haeberlen form (:meth:`Site.add_csa` with
``aniso`` = δ_aniso), whose axes are the tensor's own principal axes.

**EFG.** Principal values ordered |V_zz| ≥ |V_yy| ≥ |V_xx|;
η_Q = (V_xx − V_yy)/V_zz and C_Q = e Q V_zz / h, with the nuclear
quadrupole moment Q (signed) of the observed isotope from openNMR's
isotope table -- not the code's own (VASP uses QUAD_EFG, 1 mb by
default; CASTEP has its own table). The sign of C_Q is kept; spectra do
not depend on it.

**Relative orientation.** The CSA principal axes in the EFG frame, as
the Euler angles (α, β, γ) of :mod:`opennmr.notebook.combined_hamiltonian`
(active ZYZ: δ = R diag(δxx, δyy, δzz) Rᵀ, R = Rz(α) Ry(β) Rz(γ)). A
principal axis has no direction, so 16 sets of angles describe the same
pair of tensors; the one with the smallest β, then α, then γ is
reported.

**VASP.** The OUTCAR's "Absolute Chemical Shift tensors" are −σ of the
valence electrons only. The total is σ = −(valence + G=0 + core): the
G=0 (macroscopic susceptibility) tensor is added unless
``include_g0=False``, and the core term of each species is added. With
G=0 included the result reproduces VASP's own "(absolute, valence and
core) INCLUDING G=0 CONTRIBUTION" shifts: the printed G=0 tensor is the
valence susceptibility's (k χ_pGv), and VASP's total also carries the
core susceptibility's k χ_core, which is added here the same way. The G=0 term depends
on the sample's shape (VASP assumes a sphere) and its susceptibility, so
it does not cancel between sample and reference unless the two are
alike: treat both the same way.
"""

from __future__ import annotations

import os
import re
from itertools import product

import numpy as np

from . import combined_hamiltonian as ch

# ============================================================
# Physical constants and units
# ============================================================

_E = 1.602176634e-19              # C (exact)
_H = 6.62607015e-34               # J s (exact)
#: e/h in Hz per V.
E_OVER_H = _E / _H
#: One atomic unit of electric-field gradient, E_h/(e a0²), in V/m² (CODATA 2018).
AU_EFG = 9.7173624292e21
#: One V/Å² in V/m².
V_PER_A2 = 1e20


def quadrupole_moment_barn(nucleus) -> float | None:
    """Signed nuclear quadrupole moment of ``nucleus`` in barn (openNMR's
    isotope table, fm² → barn), or None for I = 1/2."""
    from .utils import _GAMMA_DF
    rows = _GAMMA_DF[_GAMMA_DF["Nuclide"] == str(nucleus)]
    if rows.empty:
        raise KeyError(f"{nucleus!r} is not in the isotope table.")
    try:
        return float(rows["Quadrupole Moment (fm^2)"].iloc[0]) / 100.0
    except (TypeError, ValueError):
        return None


def nmr_isotopes(element: str) -> list[str]:
    """The NMR-active isotopes of ``element``, most receptive first
    (natural abundance × |γ|³ × I(I + 1): ¹¹³Cd before ¹¹¹Cd)."""
    from .utils import _GAMMA_DF
    rows = []
    for _i, r in _GAMMA_DF.iterrows():
        m = re.fullmatch(r"(\d+)([A-Za-z]+)", str(r["Nuclide"]))
        if not m or m.group(2) != element:
            continue
        try:
            abundance = float(r["Natural Abundance (%)"])
            gamma = float(r["Gyromagnetic Ratio (MHz/T)"])
            spin = float(r["Spin"])
        except (TypeError, ValueError):
            continue
        if spin <= 0:
            continue
        rows.append((abundance * abs(gamma) ** 3 * spin * (spin + 1), str(r["Nuclide"])))
    return [n for _a, n in sorted(rows, key=lambda t: -t[0])]


def default_isotope(element: str) -> str:
    """The most receptive NMR-active isotope of ``element``."""
    found = nmr_isotopes(element)
    if not found:
        raise KeyError(f"No NMR-active isotope of {element!r} in the isotope table.")
    return found[0]


# ============================================================
# Reading calculations
# ============================================================

def _atom(species, index, label, position):
    name = f"{species}{index}" if label in (None, "", species) else f"{label}_{index}"
    return {"species": species, "index": int(index), "label": name,
            "position": np.asarray(position, dtype=float), "sigma": None, "efg": None}


def read_magres(path) -> dict:
    """A CASTEP (or any CCP-NC) ``.magres`` file: lattice, atoms (Å),
    shielding tensors ``ms`` (ppm) and EFG tensors ``efg`` (converted to
    V/m²). Tensors are taken as written (row-major); the CASTEP summary
    in ``[magres_old]``, if present, is kept as ``castep_summary`` for
    checking."""
    with open(path, "r", errors="replace") as f:
        text = f.read()
    if "[atoms]" not in text and "[magres]" not in text:
        raise ValueError(f"{os.path.basename(str(path))}: not a magres file (no [atoms] or "
                         "[magres] block).")
    calc = {"code": "CASTEP", "path": str(path), "lattice": None, "atoms": [], "notes": [],
            "info": {}}
    section = None
    units = {}
    by_key = {}
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("[") and line.endswith("]"):
            section = None if line.startswith("[/") else line[1:-1]
            continue
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if section == "calculation":
            if parts[0] in ("calc_code", "calc_code_version", "calc_xcfunctional",
                            "calc_comment", "calc_cutoffenergy"):
                calc["info"][parts[0]] = " ".join(parts[1:])
                if parts[0] == "calc_code":
                    calc["code"] = parts[1]
            elif parts[0] == "calc_pspot" and len(parts) >= 3:
                calc["info"].setdefault("pspots", {})[parts[1]] = " ".join(parts[2:])
        elif section in ("atoms", "magres"):
            if parts[0] == "units":
                units[parts[1]] = parts[2]
            elif parts[0] == "lattice":
                calc["lattice"] = np.array([float(v) for v in parts[1:10]]).reshape(3, 3)
            elif parts[0] == "atom":
                species, label, index = parts[1], parts[2], int(parts[3])
                atom = _atom(species, index, label, [float(v) for v in parts[4:7]])
                calc["atoms"].append(atom)
                by_key[(label, index)] = atom
            elif parts[0] in ("ms", "efg"):
                key = (parts[1], int(parts[2]))
                tensor = np.array([float(v) for v in parts[3:12]]).reshape(3, 3)
                atom = by_key.get(key)
                if atom is None:
                    continue
                if parts[0] == "ms":
                    if units.get("ms", "ppm") != "ppm":
                        raise ValueError(f"Shielding in {units['ms']}, expected ppm.")
                    atom["sigma"] = tensor
                else:
                    unit = units.get("efg", "au")
                    if unit != "au":
                        raise ValueError(f"EFG in {unit}, expected atomic units.")
                    atom["efg"] = tensor * AU_EFG
    if units.get("atom", "Angstrom") != "Angstrom":
        raise ValueError(f"Positions in {units['atom']}, expected Angstrom.")
    calc["castep_summary"] = _castep_summary(text)
    info = calc["info"]
    calc["method"] = {"code": calc["code"], "functional": info.get("calc_xcfunctional", ""),
                      "cutoff": info.get("calc_cutoffenergy", ""),
                      "pseudopotentials": dict(info.get("pspots") or {}),
                      "periodic": calc["lattice"] is not None}
    # The .magres σ is the code's total; CASTEP's GIPAW (Pickard & Mauri)
    # includes the G=0 term for a spherical sample, and the file carries no
    # susceptibility to take it out with.
    calc["g0"] = "fixed" if calc["lattice"] is not None else "none"
    if "[magres]" not in text:
        calc["notes"].append("No [magres] block: this file has positions but no tensors.")
    if calc["lattice"] is None:
        calc["notes"].append("No lattice: a non-periodic (molecular) calculation; tensors are "
                             "shown in the file's x, y, z frame.")
    return calc


def _castep_summary(text):
    """CASTEP's own printed numbers (``[magres_old]``), per atom label, for
    checking: ``{"Na1": {"iso", "aniso", "asym", "Cq", "eta"}}``."""
    out = {}
    pattern = re.compile(r"^\s*([A-Za-z]+)\s+(\d+)\s+(Isotropic|Anisotropy|Asymmetry|Cq|Eta):"
                         r"\s+([-+0-9.Ee]+|N/A)", re.M)
    names = {"Isotropic": "iso", "Anisotropy": "aniso", "Asymmetry": "asym", "Cq": "Cq",
             "Eta": "eta"}
    for species, index, what, value in pattern.findall(text):
        if value == "N/A":
            continue
        out.setdefault(f"{species}{index}", {})[names[what]] = float(value)
    return out


def _blocks_after(lines, marker):
    return [i for i, line in enumerate(lines) if marker in line]


def read_outcar(path, include_g0: bool = True) -> dict:
    """A VASP ``OUTCAR`` from an NMR calculation (LCHIMAG and/or LEFG):
    lattice, atoms, total shielding tensors σ (ppm) and EFG tensors (V/m²).

    σ = −(valence shift tensor + G=0 tensor + core shift): see the module
    docstring. The last occurrence of each block is used (the final ionic
    step). ``vasp_summary`` keeps VASP's own CSA table (iso, span, skew,
    with and without G=0, valence and core) for checking."""
    with open(path, "r", errors="replace") as f:
        lines = f.read().splitlines()
    if not any("vasp." in line for line in lines[:5]):
        raise ValueError(f"{os.path.basename(str(path))}: not a VASP OUTCAR (no 'vasp.' "
                         "version line at the top).")
    calc = {"code": "VASP", "path": str(path), "lattice": None, "atoms": [], "notes": [],
            "info": {"version": lines[0].split()[0]}}
    # Species, in POTCAR order.
    species = []
    for line in lines:
        m = re.match(r"\s*VRHFIN\s*=\s*([A-Za-z]+)\s*:", line)
        if m:
            species.append(m.group(1))
    if not species:
        for line in lines:
            m = re.match(r"\s*TITEL\s*=\s*\S+\s+([A-Za-z]+)", line)
            if m:
                species.append(re.sub(r"_.*", "", m.group(1)))
    counts = None
    for line in lines:
        if "ions per type" in line:
            counts = [int(v) for v in line.split("=")[1].split()]
    if counts is None or len(counts) != len(species):
        raise ValueError("Could not read the species and their counts (VRHFIN / 'ions per type').")
    symbols = [s for s, n in zip(species, counts) for _ in range(n)]
    n_ions = len(symbols)
    # Lattice: the last block.
    idx = _blocks_after(lines, "direct lattice vectors")
    if idx:
        i = idx[-1]
        calc["lattice"] = np.array([[float(v) for v in lines[i + k].split()[:3]]
                                    for k in (1, 2, 3)])
    idx = _blocks_after(lines, "position of ions in cartesian coordinates")
    if not idx:
        raise ValueError("No 'position of ions in cartesian coordinates' block.")
    i = idx[-1]
    positions = [[float(v) for v in lines[i + 1 + k].split()[:3]] for k in range(n_ions)]
    seen = {}
    for sym, pos in zip(symbols, positions):
        seen[sym] = seen.get(sym, 0) + 1
        calc["atoms"].append(_atom(sym, seen[sym], sym, pos))
    # EFG (V/Å²): V_xx V_yy V_zz V_xy V_xz V_yz.
    idx = _blocks_after(lines, "Electric field gradients (V/A^2)")
    if idx:
        i = idx[-1] + 4
        for k in range(n_ions):
            parts = lines[i + k].split()
            xx, yy, zz, xy, xz, yz = (float(v) for v in parts[1:7])
            calc["atoms"][int(parts[0]) - 1]["efg"] = np.array(
                [[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]]) * V_PER_A2
        calc["notes"].append("EFG read in V/Å² (VASP prints 3 decimals: ±0.0005 V/Å², i.e. "
                             f"±{0.0005 * V_PER_A2 * E_OVER_H * 0.1e-28 / 1e3:.2g} kHz of C_Q "
                             "per 0.1 b of Q).")
    # Shielding.
    idx = _blocks_after(lines, "Absolute Chemical Shift tensors")
    if idx:
        start = idx[-1]
        sym_at = [j for j in range(start, len(lines)) if "SYMMETRIZED TENSORS" in lines[j]
                  and "UNSYMMETRIZED" not in lines[j]]
        unsym_at = [j for j in range(start, len(lines)) if "UNSYMMETRIZED TENSORS" in lines[j]]
        j0 = (sym_at or unsym_at)[0]
        valence = {}
        j = j0 + 1
        while j < len(lines):
            m = re.match(r"\s*ion\s+(\d+)", lines[j])
            if not m:
                break
            valence[int(m.group(1))] = np.array([[float(v) for v in lines[j + k].split()[:3]]
                                                 for k in (1, 2, 3)])
            j += 4
        g0 = np.zeros((3, 3))
        g_idx = _blocks_after(lines, "G=0 CONTRIBUTION TO CHEMICAL SHIFT")
        if g_idx and include_g0:
            i = g_idx[-1]
            rows = []
            for line in lines[i:i + 12]:
                m = re.match(r"\s*([123])\s+([-0-9.Ee+]+)\s+([-0-9.Ee+]+)\s+([-0-9.Ee+]+)\s*$",
                             line)
                if m:
                    rows.append([float(m.group(k)) for k in (2, 3, 4)])
            if len(rows) == 3:
                g0 = np.array(rows)
            via = "pGv" if "pGv" in " ".join(lines[i:i + 3]) else "vGv"
            # The printed G=0 tensor is the valence susceptibility's,
            # k·χ; VASP's own total adds the core susceptibility's
            # k·χ_core too (measured: it is what makes its "valence and
            # core, INCLUDING G=0" column).
            chi = _susceptibility(lines, via)
            chi_core = _core_susceptibility(lines)
            if chi is not None and chi_core is not None and abs(np.trace(chi)) > 0:
                k = float(np.trace(g0) / np.trace(chi))
                g0 = g0 + k * chi_core * np.eye(3)
            else:
                calc["notes"].append("The core susceptibility's G=0 term was not found; only "
                                     "the valence one is included.")
            calc["notes"].append(f"VASP G=0 (macroscopic susceptibility, {via}, spherical "
                                 f"sample) included: {np.trace(g0) / 3:+.3f} ppm on every shift.")
        elif g_idx:
            calc["notes"].append("VASP G=0 contribution left out (include_g0=False).")
        core = {}
        c_idx = _blocks_after(lines, "Core NMR properties")
        if c_idx:
            for line in lines[c_idx[-1]:c_idx[-1] + 4 + len(species)]:
                m = re.match(r"\s*\d+\s+([A-Za-z]+)\s+([-0-9.Ee+]+)\s*$", line)
                if m:
                    core[m.group(1)] = float(m.group(2))
        else:
            calc["notes"].append("No core NMR contribution found: shieldings are valence only.")
        for n, tensor in valence.items():
            atom = calc["atoms"][n - 1]
            total_shift = tensor + g0 + core.get(atom["species"], 0.0) * np.eye(3)
            atom["sigma"] = -total_shift
        calc["info"]["g0_tensor"] = g0
        calc["info"]["core_shift"] = core
        calc["info"]["include_g0"] = bool(include_g0)
        calc["g0"] = ("included" if include_g0 else "excluded") if g_idx else "none"
    else:
        calc["g0"] = "none"
    calc["vasp_summary"] = _vasp_summary(lines)
    calc["method"] = _vasp_method(lines, species)
    return calc


def _vasp_method(lines, species):
    tags = {}
    for line in lines:
        m = re.match(r"\s*(GGA|METAGGA|LHFCALC|ENCUT|IVDW)\s*=\s*(\S+)", line)
        if m and m.group(1) not in tags:
            tags[m.group(1)] = m.group(2)
    titles = [re.sub(r"\s+", " ", line.split("=", 1)[1]).strip()
              for line in lines if re.match(r"\s*TITEL\s*=", line)]
    pspots = dict(zip(species, titles))
    functional = "GGA " + tags.get("GGA", "?")
    meta = tags.get("METAGGA", "")
    if meta and meta.upper() not in ("--", "NONE", "F", ".FALSE."):
        functional += f", meta-GGA {meta}"
    if tags.get("LHFCALC", "F").upper().startswith("T"):
        functional += ", hybrid"
    return {"code": "VASP", "functional": functional,
            "cutoff": f"{tags['ENCUT']} eV" if "ENCUT" in tags else "",
            "pseudopotentials": pspots, "periodic": True}


def _susceptibility(lines, which="pGv"):
    """VASP's approximate valence susceptibility tensor (10⁻⁶ cm³/mol)."""
    idx = _blocks_after(lines, f"Approximate magnetic susceptibility, {which}")
    if not idx:
        return None
    i = idx[-1]
    try:
        return np.array([[float(v) for v in lines[i + k].split()[1:4]] for k in (1, 2, 3)])
    except (ValueError, IndexError):
        return None


def _core_susceptibility(lines):
    for line in reversed(lines):
        m = re.search(r"Core contribution to magnetic susceptibility:\s*([-0-9.Ee+]+)", line)
        if m:
            return float(m.group(1))
    return None


def _vasp_summary(lines):
    """VASP's CSA table: ``{(scope, ion): (iso_ex, span_ex, skew_ex, iso_in,
    span_in, skew_in)}``, scope "valence" or "total"."""
    out = {}
    idx = _blocks_after(lines, "CSA tensor (J. Mason")
    if not idx:
        return out
    scope = None
    for line in lines[idx[-1]:idx[-1] + 400]:
        if "valence only" in line:
            scope = "valence"
            continue
        if "valence and core" in line:
            scope = "total"
            continue
        parts = line.split()
        if scope and len(parts) == 7 and parts[0].isdigit():
            out[(scope, int(parts[0]))] = tuple(float(v) for v in parts[1:])
        if "IF SPAN.EQ.0" in line:
            break
    return out


_FLOAT = r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[EeDd][-+]?\d+)?"


def _floats(line):
    return [float(v.replace("D", "E").replace("d", "e")) for v in re.findall(_FLOAT, line)]


def _symbol(Z):
    from . import periodic_table as pt
    return pt.element(int(Z)).symbol


def read_orca(path) -> dict:
    """An ORCA output (``.out``) with NMR shieldings (``! NMR`` or
    ``%eprnmr ... shift``) and/or EFGs (``fgrad``): atoms (Å, the last
    "CARTESIAN COORDINATES (ANGSTROEM)" block), the "Total shielding tensor
    (ppm)" of each nucleus and its "Raw EFG matrix" (atomic units → V/m²).

    ORCA reports eigenvalues of sᵀs; the symmetric part of σ is used here
    (the part a high-field spectrum sees), so principal values can differ
    from ORCA's in the last digits when σ is asymmetric. Each EFG is
    checked against ORCA's own printed e²qQ with ORCA's Q: a mismatch is a
    note, not silence."""
    with open(path, "r", errors="replace") as f:
        lines = f.read().splitlines()
    if not any("O   R   C   A" in line for line in lines[:200]):
        raise ValueError(f"{os.path.basename(str(path))}: not an ORCA output (no ORCA banner).")
    calc = {"code": "ORCA", "path": str(path), "lattice": None, "atoms": [], "notes": [],
            "info": {}, "g0": "none"}
    idx = _blocks_after(lines, "CARTESIAN COORDINATES (ANGSTROEM)")
    if not idx:
        raise ValueError("No 'CARTESIAN COORDINATES (ANGSTROEM)' block.")
    seen = {}
    for line in lines[idx[-1] + 2:]:
        parts = line.split()
        if len(parts) != 4:
            break
        sym = re.sub(r"[^A-Za-z]", "", parts[0])
        sym = sym[:1].upper() + sym[1:].lower()
        seen[sym] = seen.get(sym, 0) + 1
        calc["atoms"].append(_atom(sym, seen[sym], sym, [float(v) for v in parts[1:4]]))
    n_atoms = len(calc["atoms"])
    current = None
    summary = {}
    i = 0
    while i < len(lines):
        line = lines[i]
        m = re.match(r"\s*Nucleus\s+(\d+)\s*([A-Z][a-z]?)\b", line)
        if m:
            current = int(m.group(1))
            if current >= n_atoms or calc["atoms"][current]["species"] != m.group(2):
                raise ValueError(f"ORCA nucleus {m.group(1)}{m.group(2)} does not match the "
                                 "coordinates read.")
        elif "Total shielding tensor (ppm)" in line and current is not None:
            rows = [_floats(lines[i + k]) for k in (1, 2, 3)]
            calc["atoms"][current]["sigma"] = np.array(rows, dtype=float)
            i += 3
        elif "Raw EFG matrix" in line and current is not None:
            rows = []
            j = i + 1
            while len(rows) < 3 and j < i + 8:
                # a separator is all dashes; a row may start with a minus sign
                vals = [] if re.fullmatch(r"-+", lines[j].strip()) else _floats(lines[j])
                if len(vals) == 3:
                    rows.append(vals)
                j += 1
            calc["atoms"][current]["efg"] = np.array(rows, dtype=float) * AU_EFG
            entry = summary.setdefault(calc["atoms"][current]["label"], {})
            for line2 in lines[j:j + 30]:
                q = re.search(r"Quadrupole tensor eigenvalues.*Q\s*=\s*(" + _FLOAT + ")", line2)
                if q:
                    entry["Q"] = float(q.group(1))
                e = re.search(r"e\*\*2qQ\s*=\s*(" + _FLOAT + r")\s*MHz", line2)
                if e:
                    entry["e2qQ"] = float(e.group(1))
                et = re.match(r"\s*eta\s*=\s*(" + _FLOAT + ")", line2)
                if et:
                    entry["eta"] = float(et.group(1))
                if "Raw EFG matrix" in line2 or re.match(r"\s*Nucleus\s+\d+", line2):
                    break
            i = j - 1
        i += 1
    calc["orca_summary"] = summary
    # Self-check: our C_Q from the raw matrix with ORCA's Q vs ORCA's e²qQ.
    for atom in calc["atoms"]:
        s = summary.get(atom["label"], {})
        if atom["efg"] is None or "Q" not in s or "e2qQ" not in s:
            continue
        ours = efg_parameters(atom["efg"], s["Q"])["C_Q"]
        if abs(ours - s["e2qQ"]) > max(0.002, 0.01 * abs(s["e2qQ"])):
            calc["notes"].append(f"{atom['label']}: C_Q from ORCA's raw EFG with ORCA's Q is "
                                 f"{ours:.4f} MHz, but ORCA prints e²qQ = {s['e2qQ']:.4f} MHz. "
                                 "Check this file's EFG before trusting C_Q.")
    keywords = []
    for line in lines:
        m = re.match(r"\|\s*\d+>\s*!(.*)", line)
        if m:
            keywords.append(m.group(1).strip())
    calc["method"] = {"code": "ORCA", "functional": " ".join(keywords), "cutoff": "",
                      "pseudopotentials": {}, "periodic": False}
    version = next((ln.split("Version")[1].split("-")[0].strip() for ln in lines[:200]
                    if "Program Version" in ln), "")
    calc["info"]["version"] = version
    calc["notes"].append("Molecular (non-periodic) calculation: tensors are in the file's "
                         "x, y, z frame.")
    return calc


_GAUSSIAN_SHIELD = re.compile(r"^\s*(\S+)\s+(GIAO|CSGT|IGAIM|Single-origin|SingleOrigin)\s+"
                              r"Magnetic shielding tensor \(ppm\)")


def read_gaussian(path) -> dict:
    """A Gaussian log with an NMR calculation (``NMR`` keyword): atoms from
    the orientation the tensors are in (the last "Standard orientation"
    before them, or "Input orientation" with NoSymm) and each atom's
    shielding tensor (ppm). With MP2 (or another correlated) NMR, its block
    is used rather than the SCF one.

    Gaussian's EFG tables are not read: none was available to check the
    layout and sign against, so no C_Q is given for Gaussian files."""
    with open(path, "r", errors="replace") as f:
        lines = f.read().splitlines()
    if not any("Gaussian, Inc." in line or "Entering Gaussian System" in line
               for line in lines[:400]):
        raise ValueError(f"{os.path.basename(str(path))}: not a Gaussian log.")
    calc = {"code": "Gaussian", "path": str(path), "lattice": None, "atoms": [], "notes": [],
            "info": {}, "g0": "none"}
    blocks = [i for i, line in enumerate(lines) if _GAUSSIAN_SHIELD.match(line)]
    if not blocks:
        raise ValueError("No 'Magnetic shielding tensor (ppm)' block: was NMR requested?")
    methods = [_GAUSSIAN_SHIELD.match(lines[i]).group(1) for i in blocks]
    chosen = blocks[-1]
    for i, meth in zip(blocks, methods):
        if meth.upper() != "SCF":
            chosen = i
    method = _GAUSSIAN_SHIELD.match(lines[chosen]).group(1)
    if method.upper() != "SCF":
        calc["notes"].append(f"{method} shieldings used (the file also has SCF ones).")
    orient = [i for i, line in enumerate(lines[:chosen]) if "Standard orientation:" in line]
    which = "Standard"
    if not orient:
        orient = [i for i, line in enumerate(lines[:chosen]) if "Input orientation:" in line]
        which = "Input"
    if not orient:
        raise ValueError("No 'Standard orientation' or 'Input orientation' block before the "
                         "shieldings.")
    seen = {}
    for line in lines[orient[-1] + 5:]:
        parts = line.split()
        if len(parts) != 6 or not parts[0].isdigit():
            break
        sym = _symbol(parts[1])
        seen[sym] = seen.get(sym, 0) + 1
        calc["atoms"].append(_atom(sym, seen[sym], sym, [float(v) for v in parts[3:6]]))
    calc["info"]["orientation"] = which
    j = chosen + 1
    pattern = re.compile(r"([XYZ])([XYZ])=\s*(" + _FLOAT + ")")
    while j < len(lines):
        m = re.match(r"\s*(\d+)\s+([A-Z][a-z]?)\s+Isotropic\s*=", lines[j])
        if not m:
            break
        n = int(m.group(1))
        tensor = np.zeros((3, 3))
        found = 0
        for line in lines[j + 1:j + 4]:
            for a, b, v in pattern.findall(line):
                tensor["XYZ".index(a), "XYZ".index(b)] = float(v.replace("D", "E"))
                found += 1
        if found != 9:
            raise ValueError(f"Gaussian atom {n}: could not read its shielding tensor.")
        atom = calc["atoms"][n - 1]
        if atom["species"] != m.group(2):
            raise ValueError(f"Gaussian atom {n}{m.group(2)} does not match the orientation "
                             "block read.")
        atom["sigma"] = tensor
        j += 4
        while j < len(lines) and lines[j].strip().startswith("Eigenvalues"):
            j += 1
    # The route of the job step that printed the shieldings (a log can
    # hold several steps: opt, then NMR after --Link1--).
    route = []
    starts = [k for k, line in enumerate(lines[:chosen])
              if line.strip().startswith("#") and lines[k - 1].strip().startswith("---")]
    if starts:
        for line2 in lines[starts[-1]:starts[-1] + 6]:
            if line2.strip().startswith("---"):
                break
            route.append(line2.strip())
    calc["method"] = {"code": "Gaussian", "functional": " ".join(route), "cutoff": "",
                      "pseudopotentials": {}, "periodic": False}
    if any("Electric Field Gradient" in line for line in lines):
        calc["notes"].append("This Gaussian file has EFG tables, which openNMR does not read yet "
                             "(no C_Q from Gaussian).")
    calc["notes"].append("Molecular (non-periodic) calculation: tensors are in the file's "
                         f"{which.lower()}-orientation x, y, z frame.")
    return calc


#: The codes :func:`read_calculation` recognises.
SUPPORTED_CODES = ("CASTEP (.magres)", "VASP (OUTCAR)", "ORCA (.out)", "Gaussian (.log)")


def read_calculation(path, include_g0: bool = True) -> dict:
    """A CASTEP ``.magres``, VASP ``OUTCAR``, ORCA output or Gaussian log,
    recognised by content. ``include_g0`` applies to VASP only (the other
    codes give a total σ that cannot be split, or have no G=0 term)."""
    with open(path, "r", errors="replace") as f:
        head = f.read(4000)
        more = head + f.read(60000)
    if "[atoms]" in head or "#$magres" in head or "[magres]" in head or \
            "[calculation]" in head:
        return read_magres(path)
    if "vasp." in head:
        return read_outcar(path, include_g0=include_g0)
    if "O   R   C   A" in more:
        return read_orca(path)
    if "Gaussian, Inc." in more or "Entering Gaussian System" in more:
        return read_gaussian(path)
    raise ValueError(f"{os.path.basename(str(path))}: not a calculation openNMR reads ("
                     + ", ".join(SUPPORTED_CODES) + ").")


def describe_method(method) -> str:
    """One line: code, functional/route, cutoff."""
    if not method:
        return "unknown method"
    bits = [method.get("code", "")]
    if method.get("functional"):
        bits.append(method["functional"])
    if method.get("cutoff"):
        bits.append(f"cutoff {method['cutoff']}")
    return ", ".join(b for b in bits if b)


def _norm(text):
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def method_differences(a, b, element=None) -> list[str]:
    """Where two calculations' methods differ in ways that put their
    shieldings on different scales: code, functional (or route/keywords),
    cutoff, and the pseudopotential of ``element``."""
    if not a or not b:
        return []
    out = []
    if a.get("code") != b.get("code"):
        return [f"code ({a.get('code')} vs {b.get('code')})"]
    for key, name in (("functional", "functional / keywords"), ("cutoff", "cutoff")):
        if _norm(a.get(key)) and _norm(b.get(key)) and _norm(a.get(key)) != _norm(b.get(key)):
            out.append(f"{name} ({a.get(key)} vs {b.get(key)})")
    if element:
        pa = (a.get("pseudopotentials") or {}).get(element)
        pb = (b.get("pseudopotentials") or {}).get(element)
        if pa and pb and _norm(pa) != _norm(pb):
            out.append(f"{element} pseudopotential ({pa} vs {pb})")
    return out


def species_of(calc) -> list[str]:
    """The elements in a calculation, in order of first appearance."""
    return list(dict.fromkeys(a["species"] for a in calc["atoms"]))


# ============================================================
# Tensors to NMR parameters
# ============================================================

def symmetric(t):
    t = np.asarray(t, dtype=float)
    return 0.5 * (t + t.T)


def _right_handed(U):
    U = np.array(U, dtype=float)
    if np.linalg.det(U) < 0:
        U[:, 2] = -U[:, 2]
    return U


def shift_tensor(sigma, sigma_ref, delta_ref=0.0, slope=1.0):
    """δ = δ_ref + m (σ_ref − σ), on the symmetric part of σ (ppm)."""
    return float(delta_ref) * np.eye(3) + float(slope) * (float(sigma_ref) * np.eye(3)
                                                          - symmetric(sigma))


def csa_parameters(delta) -> dict:
    """Everything a (symmetric) shift tensor says, in both conventions.
    ``axes``: columns the Haeberlen x, y, z principal axes (right-handed)."""
    d = symmetric(delta)
    w, U = np.linalg.eigh(d)
    iso = float(np.mean(w))
    dev = w - iso
    order = np.argsort(-np.abs(dev), kind="stable")
    iz, ix, iy = order
    dzz, dxx, dyy = w[iz], w[ix], w[iy]
    aniso = float(dzz - iso)
    eta = float((dyy - dxx) / aniso) if abs(aniso) > 1e-12 else 0.0
    eta = min(max(eta, 0.0), 1.0)
    d11, d22, d33 = np.sort(w)[::-1]
    span = float(d11 - d33)
    skew = float(3.0 * (d22 - iso) / span) if span > 1e-12 else 0.0
    return {"iso": iso, "principal_haeberlen": (float(dxx), float(dyy), float(dzz)),
            "principal_hb": (float(d11), float(d22), float(d33)),
            "aniso": aniso, "eta": eta, "anisotropy": 1.5 * aniso,
            "span": span, "skew": min(max(skew, -1.0), 1.0),
            "axes": _right_handed(U[:, [ix, iy, iz]])}


def efg_parameters(V, Q_barn) -> dict:
    """EFG tensor (V/m²) → V_zz, η_Q, C_Q (MHz, signed; None without Q),
    P_Q. ``axes``: columns x, y, z with |V_xx| ≤ |V_yy| ≤ |V_zz|."""
    V = symmetric(V)
    # An EFG is traceless; a printed one is only to its rounding (VASP: 3
    # decimals in V/Å²). Its trace would otherwise leak into η_Q.
    V = V - np.trace(V) / 3.0 * np.eye(3)
    w, U = np.linalg.eigh(V)
    order = np.argsort(np.abs(w), kind="stable")
    ix, iy, iz = order
    vxx, vyy, vzz = w[ix], w[iy], w[iz]
    eta = float((vxx - vyy) / vzz) if abs(vzz) > 0 else 0.0
    eta = min(max(eta, 0.0), 1.0)
    cq = float(E_OVER_H * Q_barn * 1e-28 * vzz / 1e6) if Q_barn is not None else None
    return {"Vzz": float(vzz), "principal": (float(vxx), float(vyy), float(vzz)), "eta_Q": eta,
            "C_Q": cq, "P_Q": None if cq is None else float(abs(cq) * np.sqrt(1 + eta ** 2 / 3)),
            "axes": _right_handed(U[:, [ix, iy, iz]])}


def euler_zyz(R):
    """(α, β, γ) degrees with R = Rz(α) Ry(β) Rz(γ)."""
    R = np.asarray(R, dtype=float)
    cb = float(np.clip(R[2, 2], -1.0, 1.0))
    beta = np.arccos(cb)
    if np.sin(beta) > 1e-8:
        alpha = np.arctan2(R[1, 2], R[0, 2])
        gamma = np.arctan2(R[2, 1], -R[2, 0])
    elif cb > 0:
        alpha, gamma = np.arctan2(R[1, 0], R[0, 0]), 0.0
    else:
        alpha, gamma = np.arctan2(-R[1, 0], -R[0, 0]), 0.0
    return tuple(float(np.degrees(a)) % 360.0 for a in (alpha, beta, gamma))


_FLIPS = [np.diag(s) for s in ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1))]


def relative_euler(csa_axes, efg_axes, return_frame=False):
    """Euler angles (degrees) of the CSA principal frame in the EFG frame,
    as :class:`~opennmr.notebook.combined_hamiltonian.Site` uses them: the
    smallest (β, α, γ) of the 16 sign choices of the two sets of axes.

    The sign choice of the EFG axes is part of the answer: the angles hold
    in that frame. With ``return_frame`` the EFG axes they were taken in are
    returned too, ``(angles, efg_axes)`` -- the site frame of the
    simulation, for anything that maps a direction into it."""
    best = None
    for s_e, s_c in product(_FLIPS, _FLIPS):
        R = (efg_axes @ s_e).T @ (csa_axes @ s_c)
        a, b, g = euler_zyz(R)
        key = (round(b, 6), round(a % 360.0, 6), round(g % 360.0, 6))
        if best is None or key < best[0]:
            best = (key, (a, b, g), efg_axes @ s_e)
    angles = tuple(0.0 if abs(v) < 1e-9 or abs(v - 360.0) < 1e-9 else round(v, 9)
                   for v in best[1])
    return (angles, best[2]) if return_frame else angles


# ============================================================
# Choosing sites, referencing, sites for simulation
# ============================================================

def select_atoms(calc, element=None, labels=None) -> list[dict]:
    """``element="Na"`` (every Na), ``labels=["Na1", "Na3"]``, or both
    (labels within the element). Raises with what is available when
    nothing matches."""
    atoms = calc["atoms"]
    if isinstance(labels, str):
        labels = [s.strip() for s in re.split(r"[,\s]+", labels) if s.strip()]
    if element:
        atoms = [a for a in atoms if a["species"] == element]
    if labels:
        wanted = {str(s) for s in labels}
        atoms = [a for a in atoms if a["label"] in wanted]
        missing = wanted - {a["label"] for a in atoms}
        if missing:
            raise ValueError(f"No site {', '.join(sorted(missing))} in "
                             f"{os.path.basename(calc['path'])}.")
    if not atoms:
        raise ValueError(f"No {element or 'atoms'} in {os.path.basename(calc['path'])}: it has "
                         + ", ".join(species_of(calc)) + ".")
    return atoms


def reference_shielding(ref_calc, element, labels=None) -> dict:
    """The reference calculation's isotropic shielding for ``element``:
    the mean over its sites (or ``labels``), with their spread."""
    atoms = [a for a in select_atoms(ref_calc, element, labels) if a["sigma"] is not None]
    if not atoms:
        raise ValueError(f"The reference has no shielding for {element}.")
    values = np.array([float(np.trace(symmetric(a["sigma"]))) / 3.0 for a in atoms])
    return {"sigma_ref": float(values.mean()), "spread": float(np.ptp(values)),
            "labels": [a["label"] for a in atoms], "values": values.tolist()}


def _merge_key(rec, tol):
    def q(v, step):
        return None if v is None else round(v / step)
    return (q(rec["iso"], tol["iso"]), q(rec.get("aniso"), tol["aniso"]),
            q(rec.get("eta"), tol["eta"]), q(rec.get("C_Q"), tol["C_Q"]),
            q(rec.get("eta_Q"), tol["eta_Q"]),
            tuple(q(v, tol["euler"]) for v in rec.get("euler") or ()))


def dft_sites(sample, reference=None, element=None, labels=None, nucleus=None, delta_ref=0.0,
              sigma_ref=None, reference_labels=None, slope=1.0, fwhm=1.0,
              include_csa=True, include_quadrupolar=True, merge=True, calibration=None) -> dict:
    """Simulation sites from a DFT calculation.

    ``sample`` / ``reference`` : from :func:`read_calculation` (or paths).
    ``element`` / ``labels`` : which atoms -- "Na" for all of them, or
    labels such as ``["Na1", "Na2"]`` (see :func:`select_atoms`).
    ``nucleus`` : the observed isotope (default the most receptive one).
    ``delta_ref`` : the reference compound's experimental shift (ppm).
    ``sigma_ref`` : the reference shielding to use instead of the
    reference calculation's (e.g. when it has no such element).
    ``reference_labels`` : which reference sites to average.
    ``slope`` : m in δ = δ_ref + m(σ_ref − σ).
    ``calibration`` : a shielding scale from
    :func:`opennmr.notebook.dft_calibration.fit_calibration` (or a dict of
    them by element, ``{"H": cal_H, "Na": cal_Na}``), used instead of
    ``reference`` / ``sigma_ref`` / ``delta_ref`` / ``slope``: δ = (σ − a)/b,
    with each isotropic shift's uncertainty (``iso_err``) when the
    references give one.
    ``merge`` : sites whose parameters all agree (to 0.01 ppm, 0.001 in η,
    0.1 kHz, 0.1°) become one site with their count as its amount -- the
    same lineshape, simulated once.

    Returns ``{"sites", "site_dicts", "atoms", "nucleus", "reference",
    "notes"}``; each entry of ``atoms`` has the shift tensor (Cartesian,
    ppm), both CSA conventions, the EFG and C_Q, the Euler angles and
    ``site_axes`` (columns: the simulation frame's axes in the
    calculation's Cartesian frame)."""
    if isinstance(sample, (str, os.PathLike)):
        sample = read_calculation(sample)
    if isinstance(reference, (str, os.PathLike)):
        reference = read_calculation(reference)
    atoms = select_atoms(sample, element, labels)
    elements = {a["species"] for a in atoms}
    if len(elements) > 1:
        raise ValueError("Choose sites of one element: one spectrum is one nucleus (got "
                         + ", ".join(sorted(elements)) + ").")
    element = elements.pop()
    nucleus = nucleus or default_isotope(element)
    if re.sub(r"^\d+", "", nucleus) != element:
        raise ValueError(f"{nucleus} is not an isotope of {element}.")
    spin = float(ch.find_isotope_info(nucleus)["spin"])
    Q = quadrupole_moment_barn(nucleus) if spin > 0.5 else None
    notes = []
    cal = None
    if calibration is not None:
        from . import dft_calibration as dcal
        cal = calibration
        if "intercept" not in cal:           # a mapping by element
            if element not in cal:
                raise ValueError(f"No calibration for {element} (have "
                                 + ", ".join(sorted(cal)) + ").")
            cal = cal[element]
        if cal["element"] != element:
            raise ValueError(f"That calibration is for {cal['element']}, not {element}.")
        sigma_ref, delta_ref, slope = cal["intercept"], 0.0, -1.0 / cal["slope"]
        notes.append("Shift scale: " + dcal.summary_text(cal))
        notes += [f"Calibration: {n}" for n in cal.get("notes") or [] if not cal.get("manual")]
    ref_info = {"delta_ref": float(delta_ref), "slope": float(slope)}
    if cal is not None:
        ref_info.update(sigma_ref=float(sigma_ref), source="calibration",
                        calibration=cal.get("name") or "", equation=dcal.equation_text(cal))
    elif sigma_ref is None:
        if reference is None:
            raise ValueError("Give a reference calculation, or the reference shielding σ_ref.")
        try:
            info = reference_shielding(reference, element, reference_labels)
        except ValueError as exc:
            raise ValueError(f"{exc} Give σ_ref for {element} yourself (sigma_ref=…), or a "
                             f"reference calculation that contains {element}.") from None
        sigma_ref = info["sigma_ref"]
        ref_info.update(info, source=os.path.basename(reference["path"]))
        notes.append(f"σ_ref = {sigma_ref:.4f} ppm: mean of {len(info['labels'])} {element} "
                     f"site(s) of {ref_info['source']}"
                     + (f" (they differ by up to {info['spread']:.3g} ppm)"
                        if info["spread"] > 0.05 else "") + ".")
        if reference["code"] != sample["code"]:
            notes.append(f"The reference is a {reference['code']} calculation and the sample a "
                         f"{sample['code']} one: shieldings from different codes, "
                         "pseudopotentials or functionals are not on one scale.")
    else:
        ref_info.update(sigma_ref=float(sigma_ref), source="typed")
        notes.append(f"σ_ref = {float(sigma_ref):.4f} ppm, as typed.")
    if float(delta_ref) == 0.0 and cal is None:
        notes.append("δ_ref is 0: shifts are relative to the reference compound itself, not "
                      "to your experimental scale. Enter the reference's measured shift.")
    records = []
    for a in atoms:
        if a["sigma"] is None:
            raise ValueError(f"{a['label']} has no shielding tensor in "
                             f"{os.path.basename(sample['path'])}.")
        delta = shift_tensor(a["sigma"], sigma_ref, delta_ref, slope)
        csa = csa_parameters(delta)
        antisym = 0.5 * (np.asarray(a["sigma"]) - np.asarray(a["sigma"]).T)
        rec = {"label": a["label"], "species": a["species"], "position": a["position"],
               "shift_tensor": delta, "iso": csa["iso"], "aniso": csa["aniso"],
               "eta": csa["eta"], "anisotropy": csa["anisotropy"], "span": csa["span"],
               "skew": csa["skew"], "principal_hb": csa["principal_hb"],
               "principal_haeberlen": csa["principal_haeberlen"], "csa_axes": csa["axes"],
               "antisymmetric_ppm": float(np.max(np.abs(antisym))),
               "efg_tensor": a["efg"], "C_Q": None, "eta_Q": None, "P_Q": None, "Vzz": None,
               "efg_axes": None, "euler": (0.0, 0.0, 0.0)}
        use_quad = include_quadrupolar and Q is not None and a["efg"] is not None
        if include_quadrupolar and Q is not None and a["efg"] is None:
            notes.append(f"{a['label']}: no EFG in the file, so no quadrupolar coupling.")
        if a["efg"] is not None and Q is not None:
            efg = efg_parameters(a["efg"], Q)
            rec.update(C_Q=efg["C_Q"], eta_Q=efg["eta_Q"], P_Q=efg["P_Q"], Vzz=efg["Vzz"],
                       efg_axes=efg["axes"], efg_principal=efg["principal"])
        if use_quad and include_csa:
            rec["euler"], rec["efg_axes"] = relative_euler(csa["axes"], rec["efg_axes"],
                                                           return_frame=True)
        rec["iso_err"] = dcal.shift_uncertainty(cal, csa["iso"]) if cal is not None else None
        rec["uses_quadrupolar"] = bool(use_quad)
        rec["uses_csa"] = bool(include_csa)
        rec["site_axes"] = rec["efg_axes"] if use_quad else csa["axes"]
        records.append(rec)
    if spin > 0.5 and Q is None:
        notes.append(f"No quadrupole moment for {nucleus} in the isotope table.")
    if cal is not None:
        notes += dcal.sample_notes(cal, sample, [r["iso"] for r in records])
    # Sites, merging the equivalent ones.
    tol = {"iso": 0.01, "aniso": 0.01, "eta": 0.001, "C_Q": 1e-4, "eta_Q": 0.001, "euler": 0.1}
    groups = {}
    for rec in records:
        key = _merge_key({**rec, "C_Q": rec["C_Q"] if rec["uses_quadrupolar"] else None,
                          "eta_Q": rec["eta_Q"] if rec["uses_quadrupolar"] else None,
                          "aniso": rec["aniso"] if include_csa else None,
                          "eta": rec["eta"] if include_csa else None,
                          "euler": rec["euler"] if rec["uses_quadrupolar"] and include_csa
                          else ()}, tol) if merge else rec["label"]
        groups.setdefault(key, []).append(rec)
    site_dicts, sites = [], []
    for members in groups.values():
        rec = members[0]
        label = rec["label"] if len(members) == 1 else _group_label([m["label"] for m in members])
        site = ch.Site(nucleus, delta_iso=rec["iso"], label=label, amount=float(len(members)),
                       fwhm=float(fwhm))
        if include_csa and abs(rec["aniso"]) > 1e-9:
            site.add_csa(rec["aniso"], rec["eta"], convention="Haeberlen",
                         euler=rec["euler"] if rec["uses_quadrupolar"] else (0.0, 0.0, 0.0))
        if rec["uses_quadrupolar"] and abs(rec["C_Q"]) > 0:
            site.add_quadrupolar(rec["C_Q"], rec["eta_Q"])
        d = site.to_dict()
        d["dft"] = {"atoms": [m["label"] for m in members], "file": os.path.basename(
            sample["path"]), "code": sample["code"]}
        site_dicts.append(d)
        sites.append(site)
    if merge and len(sites) < len(records):
        notes.append(f"{len(records)} {element} sites → {len(sites)} distinct "
                     f"(equivalent ones merged; their count is the amount).")
    spread = max((r["antisymmetric_ppm"] for r in records), default=0.0)
    if spread > 1.0:
        notes.append(f"Antisymmetric shielding of up to {spread:.3g} ppm left out: it does not "
                     "affect a high-field spectrum to first order.")
    return {"sites": sites, "site_dicts": site_dicts, "atoms": records, "nucleus": nucleus,
            "element": element, "reference": ref_info, "notes": notes + list(sample["notes"]),
            "sample": {"path": sample["path"], "code": sample["code"]}}


def _group_label(labels):
    """``Na1, Na2, Na3, Na4`` → ``Na1–Na4`` when consecutive, else listed;
    one label is itself."""
    labels = list(labels)
    if len(labels) == 1:
        return labels[0]
    nums = []
    stem = None
    for lab in labels:
        m = re.fullmatch(r"([A-Za-z_]+?)(\d+)", lab)
        if not m or (stem is not None and m.group(1) != stem):
            return ", ".join(labels)
        stem = m.group(1)
        nums.append(int(m.group(2)))
    if nums == list(range(nums[0], nums[0] + len(nums))):
        return f"{stem}{nums[0]}–{stem}{nums[-1]}"
    return ", ".join(labels)


# ============================================================
# Reporting and simulation
# ============================================================

TABLE_COLUMNS = (
    ("label", "Site", "{}"), ("iso", "δiso (ppm)", "{:.2f}"), ("iso_err", "± δiso (ppm)", "{:.2f}"),
    ("span", "Ω (ppm)", "{:.2f}"), ("skew", "κ", "{:.3f}"),
    ("aniso", "δaniso (ppm)", "{:.2f}"), ("anisotropy", "Δδ (ppm)", "{:.2f}"),
    ("eta", "ηCSA", "{:.3f}"), ("C_Q", "C_Q (MHz)", "{:.4f}"), ("eta_Q", "ηQ", "{:.3f}"),
    ("P_Q", "P_Q (MHz)", "{:.4f}"), ("euler", "α, β, γ (°)", "{}"),
)


def table_rows(model) -> list[list[str]]:
    rows = []
    for rec in model["atoms"]:
        row = []
        for key, _name, fmt in TABLE_COLUMNS:
            v = rec.get(key)
            if key == "euler":
                row.append(", ".join(f"{x:.1f}" for x in v) if rec.get("uses_quadrupolar")
                           and rec.get("uses_csa") else "—")
            elif v is None:
                row.append("—")
            else:
                row.append(fmt.format(v))
        rows.append(row)
    return rows


def format_table(model) -> str:
    """The per-atom parameters as aligned text."""
    head = [name for _k, name, _f in TABLE_COLUMNS]
    rows = table_rows(model)
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(head)]
    lines = ["  ".join(h.rjust(w) for h, w in zip(head, widths))]
    lines += ["  ".join(c.rjust(w) for c, w in zip(r, widths)) for r in rows]
    return "\n".join(lines)


def table_csv(model) -> str:
    import csv
    import io
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow([name for _k, name, _f in TABLE_COLUMNS])
    w.writerows(table_rows(model))
    return buf.getvalue()


def simulate_dft(model, ppm_axis=None, larmor_Hz=None, B0_T=None, mas_Hz=0.0, order="second",
                 transitions="all", n_zcw=None, n_points=4096):
    """:func:`~opennmr.notebook.combined_hamiltonian.simulate_combined` of a
    :func:`dft_sites` model; the axis is chosen to hold every site when not
    given."""
    sites = model["sites"]
    if larmor_Hz is None:
        if B0_T is None:
            raise ValueError("Give the Larmor frequency or the field B0.")
        larmor_Hz = abs(ch.find_isotope_info(model["nucleus"])["gamma"]) * float(B0_T)
    if ppm_axis is None:
        lo, hi = ch.combined_axis_limits(sites, larmor_Hz, transitions=transitions,
                                         mas_Hz=mas_Hz, order=order)
        ppm_axis = np.linspace(lo, hi, int(n_points))
    return ch.simulate_combined(sites, ppm_axis, larmor_Hz=larmor_Hz, mas_Hz=mas_Hz,
                                order=order, transitions=transitions, n_zcw=n_zcw)


# ============================================================
# Frames for the tensor pictures
# ============================================================

def standard_cell_rotation(lattice) -> np.ndarray:
    """The rotation taking the calculation's Cartesian frame to the
    crystallographic standard setting: a along x, b in the xy plane, c
    with a positive z component."""
    a, b, _c = (np.asarray(v, dtype=float) for v in lattice)
    x = a / np.linalg.norm(a)
    z = np.cross(a, b)
    z /= np.linalg.norm(z)
    y = np.cross(z, x)
    return np.vstack([x, y, z])


def display_frame(calc, record, frame="cell") -> dict:
    """What the tensor picture needs to draw ``record``'s tensors in the
    calculation's frame: ``rotation`` (site frame → display frame) and the
    reference ``axes`` to draw, ``[(vector, label), ...]``.

    ``frame="cell"`` (periodic only): the standard setting (a along x, b in
    the xy plane), with the cell vectors a, b, c drawn. ``"cartesian"``:
    the file's own x, y, z."""
    R_site = np.asarray(record["site_axes"], dtype=float)
    lattice = calc.get("lattice")
    if frame == "cell":
        if lattice is None:
            raise ValueError("This calculation has no cell (not periodic): use the Cartesian "
                             "frame.")
        Q = standard_cell_rotation(lattice)
        axes = [(Q @ np.asarray(v, dtype=float), name) for v, name in zip(lattice, "abc")]
        return {"rotation": Q @ R_site, "axes": axes, "name": "cell",
                "title": "cell frame (a ∥ x, b in xy)"}
    if frame == "cartesian":
        return {"rotation": R_site, "axes": [], "name": "cartesian",
                "title": f"{calc['code']} Cartesian frame"}
    raise ValueError("frame must be 'cell' or 'cartesian'.")


def axis_angles(calc, record) -> list[str]:
    """Plain-language directions of the unique axes (δzz, V_zz): angles to
    a, b, c (periodic) or x, y, z."""
    out = []
    lattice = calc.get("lattice")
    refs = ([(np.asarray(v, float), n) for v, n in zip(lattice, "abc")] if lattice is not None
            else [(e, n) for e, n in zip(np.eye(3), "xyz")])
    for name, axes in (("δzz", record.get("csa_axes")), ("V_zz", record.get("efg_axes"))):
        if axes is None:
            continue
        v = np.asarray(axes)[:, 2]
        angles = []
        for ref, n in refs:
            c = abs(float(np.dot(v, ref) / np.linalg.norm(ref)))
            angles.append(f"{np.degrees(np.arccos(min(c, 1.0))):.1f}° to {n}")
        out.append(f"{record['label']}: {name} axis at " + ", ".join(angles) + ".")
    return out


def plot_dft_tensors(record, calc, frame="cell", style="ellipsoid", larmor_Hz=None,
                     nucleus=None, figure=None, **kwargs):
    """One atom's tensors in the calculation's frame (cell or Cartesian):
    :func:`opennmr.notebook.tensor_views.plot_tensors` with the display
    frame of :func:`display_frame`. Returns ``(figure, notes)``."""
    from . import tensor_views as tv
    site = _record_site(record, nucleus)
    look = display_frame(calc, record, frame)
    fig, notes = tv.plot_tensors(site, larmor_Hz, style=style, figure=figure, frame=look,
                                 **kwargs)
    return fig, notes + axis_angles(calc, record)


def _record_site(record, nucleus=None):
    element = record["species"]
    nucleus = nucleus or default_isotope(element)
    site = ch.Site(nucleus, delta_iso=record["iso"], label=record["label"], fwhm=0.0)
    if record.get("uses_csa", True) and abs(record["aniso"]) > 1e-9:
        site.add_csa(record["aniso"], record["eta"], convention="Haeberlen",
                     euler=record["euler"] if record.get("uses_quadrupolar") else (0, 0, 0))
    if record.get("uses_quadrupolar") and record.get("C_Q"):
        site.add_quadrupolar(record["C_Q"], record["eta_Q"])
    return site


def dft_to_python(sample_path, reference_path=None, **kwargs) -> str:
    """Notebook code that reproduces a GUI DFT import."""
    args = ", ".join(f"{k}={v!r}" for k, v in kwargs.items() if v is not None)
    ref = (f"ref = dft_nmr.read_calculation({reference_path!r})\n" if reference_path
           else "ref = None\n")
    return ("from opennmr.notebook import dft_nmr\n"
            f"sample = dft_nmr.read_calculation({sample_path!r})\n" + ref
            + f"model = dft_nmr.dft_sites(sample, ref{', ' + args if args else ''})\n"
            "print(dft_nmr.format_table(model))\n")
