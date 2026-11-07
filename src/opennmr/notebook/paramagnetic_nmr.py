"""Paramagnetic NMR shifts from g, zero-field-splitting and hyperfine tensors.

This is the molecular ("ORCA-style") route to the paramagnetic shift, as in
ORCA's ``orca_pnmr`` (Moon & Patchkovskii; Soncini & Van den Heuvel; Vaara):

    σ_para = −(μ_B / (g_N μ_N)) · g · Z · A ,                      (1)

with ``g`` the electronic g tensor ("BgS" convention, H = μ_B B·g·S),
``A`` the hyperfine tensor of the nucleus ("SAI" convention, H = S·A·I, in
energy units: h × the MHz that ORCA prints), ``g_N μ_N`` the nuclear
magnetogyric factor (γ_N ħ), and ``Z`` the electron spin's static
(isothermal) susceptibility tensor at zero field, ⟨⟨S_k ; S_l⟩⟩ (J⁻¹):

    Z_kl = Σ_{n,m} ⟨n|S_k|m⟩⟨m|S_l|n⟩ w_nm ,
    w_nm = p_n / k_BT                          if E_n = E_m
         = (p_n − p_m) / (E_m − E_n)           otherwise,

over the eigenstates |n⟩ (energies E_n, Boltzmann populations p_n) of the
zero-field spin Hamiltonian H = S·D·S. Without zero-field splitting every
level is degenerate and Z = S(S+1)/(3 k_B T) 1 -- the Curie law. With it,
Z is anisotropic and has a temperature-independent (second-order, "van
Vleck") part, which gives a pseudocontact shift even with an isotropic g.

The isotropic shift splits exactly into

    δ_contact      = +(μ_B / g_N μ_N) · tr(g Z) A_iso / 3      (Fermi contact)
    δ_pseudocontact = +(μ_B / g_N μ_N) · tr(g Z A_dip) / 3      (spin dipolar),

with A = A_iso 1 + A_dip; the tensors (not only their traces) are kept, as
the spin dipolar tensor makes the large spinning-sideband manifolds of a
paramagnetic solid. δ = −σ (diamagnetic convention, δ > 0 to high
frequency).

``Z`` can also come from the sample's magnetism rather than the calculated
spin Hamiltonian (:data:`MODELS`): a Curie–Weiss law, a measured effective
moment or susceptibility, or the scaling factor φ = ⟨S_z⟩/S used for
periodic calculations (:mod:`opennmr.notebook.crystal_nmr`). For a
coupled solid, a single-ion Curie law overestimates the shift whenever the
ions are antiferromagnetically coupled; the measured magnetism is then the
right input.

What is *not* here: the orbital (diamagnetic) shielding. ORCA computes it in
a separate (NMR) run; add it as ``orbital_shift`` or through a reference
calculation / calibration in :func:`opennmr.notebook.dft_nmr.dft_sites`.

Readers: :func:`attach_orca_epr` adds the g, D and hyperfine tensors of an
ORCA EPR/NMR output (called by :func:`dft_nmr.read_orca`), and
:func:`read_orca_pnmr_input` / :func:`read_orca_pnmr_output` read the input
and output of ``orca_pnmr``, so a result can be checked against ORCA's own.
"""

from __future__ import annotations

import copy
import math
import os
import re

import numpy as np

from .crystal_nmr import H_PLANCK, K_B, MU_0, MU_B, MU_N, N_A, G_E, A_0

__all__ = [
    "MODELS", "spin_matrices", "spin_susceptibility", "spin_response", "paramagnetic_shielding",
    "shielding_contributions", "attach_orca_epr", "is_paramagnetic", "apply_paramagnetic",
    "read_orca_pnmr_input", "read_orca_pnmr_output", "compare_with_orca", "contribution_rows",
    "zfs_tensor", "D_E_from_tensor", "g_nuclear_from_pfac", "PFAC_PER_GN",
    "shift_vs_temperature", "fit_temperature", "TEMPERATURE_MODELS", "plot_shift_temperature",
    "plot_contributions", "paramagnetic_to_python",
]

C_CM = 29979245800.0                      # speed of light, cm s^-1
CM1_TO_J = H_PLANCK * C_CM                # 1 cm^-1 in J

#: ORCA's hyperfine prefactor P (MHz bohr³) per unit nuclear g factor:
#: P = (μ0/4π) g_e μ_B g_N μ_N / (h a0³). ORCA prints P for each nucleus;
#: P / PFAC_PER_GN is the g_N that its A tensor was computed with.
PFAC_PER_GN = 1e-7 * G_E * MU_B * MU_N / (H_PLANCK * A_0 ** 3) / 1e6

MODELS = {
    "curie": "Calculated spin Hamiltonian: Curie law with the calculated g, plus the "
             "zero-field splitting D if there is one (ORCA's orca_pnmr model)",
    "curie_weiss": "Curie–Weiss law with the calculated S: Z = S(S+1)/3k(T − θ)",
    "mu_eff": "Measured effective moment μ_eff (and Weiss constant θ)",
    "susceptibility": "Measured molar susceptibility χ_M at the temperature T",
    "phi": "Scaling factor φ = ⟨S_z⟩/S at the field B0 (as for CRYSTAL)",
}

_FLOAT = r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[EeDd][-+]?\d+)?"


def _floats(line):
    return [float(v.replace("D", "E").replace("d", "e")) for v in re.findall(_FLOAT, line)]


# ---------------------------------------------------------------------------
# The electron spin
# ---------------------------------------------------------------------------

def spin_matrices(S):
    """(S_x, S_y, S_z) for spin ``S`` (2S+1 square, complex), basis m = S … −S."""
    S = float(S)
    n = int(round(2 * S + 1))
    if n < 2 or abs(n - (2 * S + 1)) > 1e-9:
        raise ValueError(f"S = {S:g} is not a positive half-integer or integer.")
    m = S - np.arange(n)
    sz = np.diag(m).astype(complex)
    sp = np.zeros((n, n), dtype=complex)
    for i in range(1, n):
        sp[i - 1, i] = math.sqrt(S * (S + 1) - m[i] * (m[i] + 1))
    sx = 0.5 * (sp + sp.conj().T)
    sy = -0.5j * (sp - sp.conj().T)
    return sx, sy, sz


def zfs_tensor(D_cm=0.0, E_cm=0.0, axes=None) -> np.ndarray:
    """The traceless D tensor (cm⁻¹) of the axial and rhombic parameters:
    diag(−D/3 + E, −D/3 − E, 2D/3) in its own frame, rotated by ``axes``
    (columns: its x, y, z in the molecular frame)."""
    t = np.diag([-D_cm / 3.0 + E_cm, -D_cm / 3.0 - E_cm, 2.0 * D_cm / 3.0])
    if axes is not None:
        R = np.asarray(axes, dtype=float)
        t = R @ t @ R.T
    return t


def D_E_from_tensor(D_cm) -> dict:
    """D and E (cm⁻¹) of a D tensor, with |E/D| ≤ 1/3 (the usual choice of
    axes: z the principal axis furthest from the mean)."""
    t = np.asarray(D_cm, dtype=float)
    t = 0.5 * (t + t.T)
    t = t - np.trace(t) / 3.0 * np.eye(3)
    w, v = np.linalg.eigh(t)
    order = np.argsort(np.abs(w))          # smallest |.| first, z last
    w, v = w[order], v[:, order]
    D = 1.5 * w[2]
    E = 0.5 * (w[0] - w[1])
    if D != 0 and E / D < 0:
        w[[0, 1]] = w[[1, 0]]
        v[:, [0, 1]] = v[:, [1, 0]]
        E = -E
    return {"D": float(D), "E": float(E), "E_over_D": float(E / D) if D else 0.0,
            "axes": v, "trace": float(np.trace(np.asarray(D_cm, dtype=float)))}


def spin_susceptibility(S, T, D_cm=None) -> np.ndarray:
    """Z (J⁻¹): the zero-field static susceptibility of the spin ``S`` at
    ``T`` (K) under H = S·D·S (``D_cm`` 3×3 in cm⁻¹, or None for no
    zero-field splitting -- then exactly S(S+1)/(3 k_B T) 1)."""
    if T <= 0:
        raise ValueError("The temperature must be positive (K).")
    kT = K_B * float(T)
    S = float(S)
    if D_cm is None or not np.any(np.asarray(D_cm, dtype=float)):
        spin_matrices(S)                      # validates S
        return S * (S + 1.0) / (3.0 * kT) * np.eye(3)
    D = np.asarray(D_cm, dtype=float) * CM1_TO_J
    D = 0.5 * (D + D.T)
    ops = spin_matrices(S)
    H = sum(D[k, l] * ops[k] @ ops[l] for k in range(3) for l in range(3))
    E, U = np.linalg.eigh(H)
    E = E - E.min()
    p = np.exp(-E / kT)
    p /= p.sum()
    M = [U.conj().T @ o @ U for o in ops]
    dE = E[None, :] - E[:, None]               # E_m − E_n
    # Energies are in joules (~1e-21): the tolerance must scale with them.
    degenerate = np.abs(dE) < 1e-9 * (kT + float(np.max(np.abs(E))))
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.where(degenerate, p[:, None] / kT, (p[:, None] - p[None, :]) / dE)
    Z = np.empty((3, 3))
    for k in range(3):
        for l in range(3):
            Z[k, l] = float(np.real(np.sum(M[k] * M[l].T * w)))
    return 0.5 * (Z + Z.T)


def _g_iso(g) -> float:
    g = np.asarray(g, dtype=float)
    return float(math.sqrt(np.trace(g @ g.T) / 3.0))


def _chi_si(chi, units) -> float:
    if units in ("cm3/mol", "emu/mol", "cgs"):
        return 4.0 * math.pi * 1e-6 * float(chi)
    if units in ("m3/mol", "si", "SI"):
        return float(chi)
    raise ValueError("χ units must be 'cm3/mol' (cgs, emu/mol) or 'm3/mol' (SI).")


def spin_response(S, T=298.0, model="curie", *, g=None, D_cm=None, theta=0.0, mu_eff=None,
                  chi=None, chi_units="cm3/mol", phi=None, B0=None) -> dict:
    """The spin susceptibility tensor Z (J⁻¹) for one of the :data:`MODELS`.

    * ``"curie"``: from S, T and the zero-field splitting ``D_cm`` (if any)
      -- the calculated spin Hamiltonian, as orca_pnmr.
    * ``"curie_weiss"``: S(S+1)/(3 k_B (T − θ)).
    * ``"mu_eff"``: μ_eff² / (3 k_B (T − θ) g_iso²), μ_eff in μ_B.
    * ``"susceptibility"``: χ_M / (μ0 N_A μ_B² g_iso²), the measured molar
      susceptibility at T (cm³/mol cgs by default).
    * ``"phi"``: φ S / (g_iso μ_B B0), φ = ⟨S_z⟩/S at the field B0 (T).

    g_iso = √(tr(g gᵀ)/3) of the calculated g (g_e without one): the
    measured magnetism is μ_B² g² Z, so Z is what is left once g is
    divided out. Returns ``{"Z", "model", "text", "notes", ...}``."""
    if model not in MODELS:
        raise ValueError(f"model must be one of {', '.join(MODELS)}.")
    gi = _g_iso(g) if g is not None else G_E
    notes = []
    has_D = D_cm is not None and np.any(np.asarray(D_cm, dtype=float))
    if has_D and model != "curie":
        notes.append("The zero-field splitting is used only by the calculated-spin-Hamiltonian "
                     "model; this model takes the magnetism as measured, which already "
                     "includes its effect on the average spin (isotropically).")
    if model in ("curie_weiss", "mu_eff") and T - float(theta or 0.0) <= 0:
        raise ValueError(f"T − θ = {T - float(theta or 0.0):g} K is not positive: the "
                         "Curie–Weiss law holds only above the ordering (Weiss) temperature.")
    if model == "curie":
        Z = spin_susceptibility(S, T, D_cm if has_D else None)
        text = (f"Z from the spin Hamiltonian: S = {S:g}, T = {T:g} K"
                + (", with zero-field splitting" if has_D else ", Curie law (no ZFS)"))
    elif model == "curie_weiss":
        Z = S * (S + 1.0) / (3.0 * K_B * (T - theta)) * np.eye(3)
        text = f"Curie–Weiss: Z = S(S+1)/3k(T − θ), S = {S:g}, T = {T:g} K, θ = {theta:g} K"
    elif model == "mu_eff":
        if not mu_eff or mu_eff <= 0:
            raise ValueError("Give the effective moment μ_eff (μ_B, positive).")
        Z = mu_eff ** 2 / (3.0 * K_B * (T - theta) * gi ** 2) * np.eye(3)
        text = (f"μ_eff = {mu_eff:g} μ_B, θ = {theta:g} K, T = {T:g} K (g_iso = {gi:.5f}); "
                f"equivalent S(S+1) = {mu_eff ** 2 / gi ** 2:.4g}")
    elif model == "susceptibility":
        if chi is None or chi <= 0:
            raise ValueError("Give the molar susceptibility χ_M (positive: the paramagnetic "
                             "part, with the diamagnetic background removed).")
        Z = _chi_si(chi, chi_units) / (MU_0 * N_A * MU_B ** 2 * gi ** 2) * np.eye(3)
        text = f"χ_M = {chi:g} {chi_units} at T = {T:g} K (g_iso = {gi:.5f})"
    else:
        if phi is None or B0 is None or B0 <= 0:
            raise ValueError("Give φ and the field B0 (T) it belongs to.")
        Z = float(phi) * S / (gi * MU_B * float(B0)) * np.eye(3)
        text = f"φ = {phi:g} at B0 = {B0:g} T, S = {S:g} (g_iso = {gi:.5f})"
    curie = S * (S + 1.0) / (3.0 * K_B * T)
    ratio = float(np.trace(Z) / 3.0 / curie)
    text += f"; Z_iso / Curie(S, T) = {ratio:.4g}"
    return {"Z": Z, "model": model, "S": float(S), "T": float(T), "theta": float(theta or 0.0),
            "mu_eff": mu_eff, "chi": chi, "chi_units": chi_units, "phi": phi, "B0": B0,
            "g_iso": gi, "has_zfs": bool(has_D and model == "curie"), "text": text,
            "curie_ratio": ratio, "notes": notes}


# ---------------------------------------------------------------------------
# The shielding
# ---------------------------------------------------------------------------

def _prefactor(g_N) -> float:
    """μ_B h 10⁶ / (g_N μ_N), times 10⁶ for ppm: A in MHz, Z in J⁻¹."""
    if not g_N:
        raise ValueError("The nuclear g factor is zero.")
    return MU_B * H_PLANCK * 1e6 / (float(g_N) * MU_N) * 1e6


def paramagnetic_shielding(g, Z, A_MHz, g_N) -> np.ndarray:
    """σ_para (ppm, 3×3, not symmetrised) = −(μ_B/(g_N μ_N)) g·Z·A, with
    A in MHz (of the isotope whose ``g_N`` is given) and Z in J⁻¹."""
    g = G_E * np.eye(3) if g is None else np.asarray(g, dtype=float)
    return -_prefactor(g_N) * (g @ np.asarray(Z, dtype=float) @ np.asarray(A_MHz, dtype=float))


def shielding_contributions(g, Z, A_MHz, g_N) -> dict:
    """The Fermi contact (A_iso) and spin dipolar (A − A_iso) parts of σ_para
    (tensors, ppm) and the isotropic shifts δ = −σ_iso: ``contact``,
    ``pseudocontact``, ``total``."""
    A = np.asarray(A_MHz, dtype=float)
    a_iso = float(np.trace(A) / 3.0)
    s_con = paramagnetic_shielding(g, Z, a_iso * np.eye(3), g_N)
    s_dip = paramagnetic_shielding(g, Z, A - a_iso * np.eye(3), g_N)
    tot = s_con + s_dip
    return {"sigma": tot, "sigma_contact": s_con, "sigma_dipolar": s_dip, "A_iso_MHz": a_iso,
            "contact": -float(np.trace(s_con)) / 3.0, "pseudocontact": -float(np.trace(s_dip)) / 3.0,
            "total": -float(np.trace(tot)) / 3.0}


def g_nuclear_from_pfac(pfac_MHz) -> float:
    """g_N from ORCA's printed P (MHz bohr³)."""
    return float(pfac_MHz) / PFAC_PER_GN


# ---------------------------------------------------------------------------
# ORCA
# ---------------------------------------------------------------------------

_HFC_HEAD = re.compile(r"^\s*Nucleus\s+(\d+)\s*([A-Z][a-z]?)\s*:\s*A:ISTP=\s*(\d+)\s+I=\s*("
                       + _FLOAT + r")\s+P=\s*(" + _FLOAT + ")")
_MULT = re.compile(r"^\s*Multiplicity\b[^\n]*?\.{3,}\s*(\d+)\s*$")


def _matrix_after(lines, i, limit=8):
    """The first three rows of three numbers after line ``i`` (separators
    of dashes skipped -- a row may itself start with a minus sign)."""
    rows = []
    j = i + 1
    while len(rows) < 3 and j < min(len(lines), i + 1 + limit):
        s = lines[j].strip()
        vals = [] if (not s or re.fullmatch(r"-+", s)) else _floats(s)
        if len(vals) == 3 and re.fullmatch(r"[\s\d.eEdD+-]+", s):
            rows.append(vals)
        elif rows:
            break
        j += 1
    return (np.array(rows, dtype=float), j) if len(rows) == 3 else (None, j)


def parse_orca_epr(lines) -> dict:
    """The EPR parts of an ORCA output: ``g`` (3×3), ``multiplicity``,
    ``D_cm`` (3×3 or None) and ``hfc`` {ORCA atom index: {"element",
    "mass", "I", "P", "g_N", "A_MHz"}} (the "Raw HFC matrix" of each
    nucleus). The last of each block wins, as ORCA prints a block again in
    a later job step."""
    out = {"g": None, "multiplicity": None, "D_cm": None, "hfc": {}, "notes": []}
    for i, line in enumerate(lines):
        m = _MULT.match(line)
        if m:
            out["multiplicity"] = int(m.group(1))
        if "The g-matrix:" in line:
            g, _j = _matrix_after(lines, i)
            if g is not None:
                out["g"] = g
        elif re.search(r"ZERO[- ]FIELD[- ]SPLITTING", line, re.I):
            for k in range(i + 1, min(len(lines), i + 60)):
                if re.search(r"raw[- ]?matrix", lines[k], re.I):
                    D, _j = _matrix_after(lines, k)
                    if D is not None:
                        out["D_cm"] = D
                    break
        m = _HFC_HEAD.match(line)
        if m:
            idx = int(m.group(1))
            for k in range(i + 1, min(len(lines), i + 8)):
                if "Raw HFC matrix" in lines[k]:
                    A, _j = _matrix_after(lines, k)
                    if A is not None:
                        P = float(m.group(5))
                        out["hfc"][idx] = {"element": m.group(2), "mass": int(m.group(3)),
                                           "I": float(m.group(4)), "P": P,
                                           "g_N": g_nuclear_from_pfac(P), "A_MHz": A}
                    break
    if out["D_cm"] is not None:
        out["notes"].append("A zero-field-splitting tensor was read from the ORCA output "
                            "(its 'raw matrix'). This reader has only been checked on "
                            "synthetic text: compare D and E with ORCA's printout.")
    return out


def attach_orca_epr(calc, lines) -> dict:
    """Add the g, D and hyperfine tensors of an ORCA output to ``calc`` (from
    :func:`dft_nmr.read_orca`, in place): ``calc["epr"]`` and, per atom,
    ``hfc``. Returns ``calc``."""
    epr = parse_orca_epr(lines)
    if not epr["hfc"]:
        return calc
    atoms = calc["atoms"]
    for idx, h in epr["hfc"].items():
        if idx >= len(atoms) or atoms[idx]["species"] != h["element"]:
            raise ValueError(f"ORCA hyperfine nucleus {idx}{h['element']} does not match the "
                             "coordinates read.")
        atoms[idx]["hfc"] = dict(h)
        atoms[idx]["orca_label"] = f"{idx}{h['element']}"
    mult = epr["multiplicity"]
    calc["epr"] = {"g": epr["g"], "multiplicity": mult,
                   "S": None if mult is None else (mult - 1) / 2.0, "D_cm": epr["D_cm"]}
    calc["notes"] = list(calc.get("notes") or []) + epr["notes"]
    if epr["g"] is None:
        calc["notes"].append("No g-matrix in the ORCA output: the free-electron g_e is used "
                             "(add 'gtensor true' to %eprnmr for the calculated one).")
    calc["notes"].append(f"ORCA EPR: hyperfine tensors for {len(epr['hfc'])} nuclei"
                         + (f", multiplicity {mult} (S = {(mult - 1) / 2:g})" if mult else "")
                         + (", g-matrix" if epr["g"] is not None else "")
                         + (", D tensor" if epr["D_cm"] is not None else "") + ".")
    return calc


def is_paramagnetic(calc) -> bool:
    """True for a calculation whose shifts need a magnetic model: CRYSTAL
    hyperfine data, or an ORCA output with hyperfine tensors."""
    return calc is not None and (calc.get("code") == "CRYSTAL" or bool(calc.get("epr")))


def apply_paramagnetic(calc, *, model="curie", T=298.0, S=None, D_cm="file", g="file",
                       components="both", **model_kwargs) -> dict:
    """A copy of an ORCA EPR ``calc`` with each hyperfine nucleus's
    paramagnetic shielding (eq. 1). Per atom: ``sigma_para`` (ppm tensor of
    the chosen ``components``: "both", "contact" or "dipolar"),
    ``para`` (:func:`shielding_contributions`). An atom with no orbital
    shielding in the file gets ``sigma`` = ``sigma_para`` (so the shift is
    the paramagnetic one alone, plus whatever orbital shift is added); one
    with an orbital shielding keeps it as ``sigma``.

    ``S`` : default from the multiplicity. ``D_cm`` / ``g`` : ``"file"``
    (what the output has), None (none / g_e), or a 3×3. The other keywords
    go to :func:`spin_response`."""
    if not calc.get("epr"):
        raise ValueError("No hyperfine tensors in this calculation (ORCA %eprnmr with "
                         "Nuclei = all X {aiso, adip}).")
    if components not in ("both", "contact", "dipolar"):
        raise ValueError("components must be 'both', 'contact' or 'dipolar'.")
    epr = calc["epr"]
    S = epr.get("S") if S is None else float(S)
    if S is None or S <= 0:
        raise ValueError("Give the spin S (no multiplicity in the file).")
    gt = epr.get("g") if isinstance(g, str) and g == "file" else g
    Dt = epr.get("D_cm") if isinstance(D_cm, str) and D_cm == "file" else D_cm
    resp = spin_response(S, T, model, g=gt, D_cm=Dt, **model_kwargs)
    out = copy.deepcopy(calc)
    n = 0
    have_orbital = sum(a.get("sigma") is not None for a in out["atoms"] if a.get("hfc"))
    for a in out["atoms"]:
        h = a.get("hfc")
        if not h:
            continue
        parts = shielding_contributions(gt, resp["Z"], h["A_MHz"], h["g_N"])
        sp = {"both": parts["sigma"], "contact": parts["sigma_contact"],
              "dipolar": parts["sigma_dipolar"]}[components]
        a["para"] = parts
        a["sigma_para"] = sp
        a["delta_contact"] = parts["contact"]
        a["delta_pseudocontact"] = parts["pseudocontact"]
        if a.get("sigma") is None:
            a["sigma"] = sp
            a["sigma_para"] = None          # already the whole of sigma
            a["orbital_in_file"] = False
        else:
            a["orbital_in_file"] = True
        n += 1
    out["paramagnetic"] = True
    out["scaling"] = dict(resp, components=components, Z=resp["Z"].tolist(),
                          g=None if gt is None else np.asarray(gt).tolist(),
                          D_cm=None if Dt is None else np.asarray(Dt).tolist())
    out["notes"] = list(out.get("notes") or []) + list(resp["notes"])
    what = {"both": "Fermi contact + spin dipolar", "contact": "Fermi contact only",
            "dipolar": "spin dipolar only"}[components]
    out["notes"].append(f"Paramagnetic shielding σ = −(μ_B/g_Nμ_N) g·Z·A for {n} nuclei ({what}); "
                        + resp["text"] + ".")
    if have_orbital and have_orbital < n:
        out["notes"].append("Some nuclei have an orbital shielding in the file and some do not: "
                            "their shifts are not on one scale.")
    elif not have_orbital:
        out["notes"].append("No orbital (diamagnetic) shielding in this output: the shifts are the "
                            "paramagnetic ones; add the orbital shift (e.g. of a diamagnetic "
                            "analogue, or from an ORCA NMR run of the same structure).")
    return out


def contribution_rows(calc_or_model, labels=None) -> list[dict]:
    """Per nucleus (that has a paramagnetic part): label, contact,
    pseudocontact and total paramagnetic isotropic shift (ppm)."""
    atoms = calc_or_model.get("atoms") or []
    rows = []
    for a in atoms:
        members = a.get("members") or [a.get("label")]
        if labels is not None and not set(members) & set(labels):
            continue
        p = a.get("para")
        if p is None:
            continue
        rows.append({"label": a["label"], "contact": p["contact"],
                     "pseudocontact": p["pseudocontact"], "total": p["total"]})
    return rows


# ---- orca_pnmr's own files -----------------------------------------------------

def read_orca_pnmr_input(path) -> dict:
    """An ``orca_pnmr`` input (``.pnmr.inp``): multiplicity, temperatures, g,
    D (cm⁻¹) and the A tensors (MHz) with each nucleus's P."""
    with open(path, "r", errors="replace") as f:
        raw = [ln.split("#")[0].strip() for ln in f.read().splitlines()]
    lines = [ln for ln in raw if ln]
    it = iter(lines)
    try:
        mult = int(float(next(it)))
        t0, t1, dt = float(next(it)), float(next(it)), float(next(it))
        g = None
        if int(float(next(it))):
            g = np.array([_floats(next(it)) for _ in range(3)])
        D = None
        if int(float(next(it))):
            D = np.array([_floats(next(it)) for _ in range(3)])
        n = int(float(next(it)))
        hfc = {}
        for _ in range(n):
            label = next(it)
            P = float(next(it))
            A = np.array([_floats(next(it)) for _ in range(3)])
            m = re.match(r"(\d+)\s*([A-Z][a-z]?)", label)
            if not m:
                raise ValueError(f"Cannot read the nucleus label {label!r}.")
            hfc[int(m.group(1))] = {"element": m.group(2), "label": label, "P": P,
                                    "g_N": g_nuclear_from_pfac(P), "A_MHz": A}
    except StopIteration:
        raise ValueError(f"{os.path.basename(str(path))}: the orca_pnmr input ends early.") from None
    temps = list(np.arange(t0, t1 + 0.5 * dt, dt)) if dt > 0 else [t0]
    return {"multiplicity": mult, "S": (mult - 1) / 2.0, "temperatures": temps, "g": g,
            "D_cm": D, "hfc": hfc}


def read_orca_pnmr_output(path) -> dict:
    """``orca_pnmr``'s printed paramagnetic shielding tensors:
    ``{temperature: {"0H": {"iso", "tensor"}}}``."""
    with open(path, "r", errors="replace") as f:
        lines = f.read().splitlines()
    out = {}
    T = None
    in_block = False
    row = re.compile(r"^\s*(\d+[A-Z][a-z]?)\s+(" + _FLOAT + r")\s+(" + _FLOAT + r")\s+("
                     + _FLOAT + r")\s+(" + _FLOAT + r")\s*$")
    for i, line in enumerate(lines):
        if "Paramagnetic shielding tensors" in line:
            in_block = True
            continue
        if "Isotropic paramagnetic shielding summary" in line:
            in_block = False
            continue
        if not in_block:
            continue
        m = re.match(r"\s*Temperature:\s*(" + _FLOAT + r")\s*K", line)
        if m:
            T = float(m.group(1))
            out.setdefault(T, {})
            continue
        m = row.match(line)
        if m and T is not None and i > 0 and i + 1 < len(lines):
            above, below = _floats(lines[i - 1]), _floats(lines[i + 1])
            if len(above) == 3 and len(below) == 3:
                out[T][m.group(1)] = {"iso": float(m.group(2)), "tensor": np.array(
                    [above, [float(m.group(k)) for k in (3, 4, 5)], below])}
    if not out:
        raise ValueError(f"{os.path.basename(str(path))}: no 'Paramagnetic shielding tensors' "
                         "block (orca_pnmr output).")
    return out


def read_orca_pnmr_echo(path) -> dict:
    """What ``orca_pnmr`` printed of its own input: ``S``, ``g``, ``D_cm``
    (None when it says D is not used), and ``hfc`` {ORCA atom index:
    {"element", "P", "g_N", "A_MHz"}} -- enough to recompute its result."""
    with open(path, "r", errors="replace") as f:
        lines = f.read().splitlines()
    out = {"S": None, "g": None, "D_cm": None, "hfc": {}}
    a_row = re.compile(r"^\s*(\d+)([A-Z][a-z]?)\s+(" + _FLOAT + r")\s+(" + _FLOAT + r")\s+("
                       + _FLOAT + r")\s+(" + _FLOAT + r")\s+(" + _FLOAT + r")\s*$")
    in_a = False
    for i, line in enumerate(lines):
        m = re.search(r"Electronic spin quantum number:\s*(" + _FLOAT + ")", line)
        if m:
            out["S"] = float(m.group(1))
        if line.strip() == "g tensor:":
            out["g"] = np.array([_floats(lines[i + k]) for k in (2, 3, 4)])
        if line.strip().startswith("D tensor (cm-1):") and "will not" not in line:
            D = np.array([_floats(lines[i + k]) for k in (2, 3, 4)])
            out["D_cm"] = D
        if "D tensor will not be used" in line:
            out["D_cm"] = None
        if "Nucleus  PFAC" in line:
            in_a = True
            continue
        if "Paramagnetic shielding tensors" in line:
            break
        if in_a:
            m = a_row.match(line)
            if m:
                above, below = _floats(lines[i - 1]), _floats(lines[i + 1])
                P = float(m.group(3))
                out["hfc"][int(m.group(1))] = {
                    "element": m.group(2), "P": P, "g_N": g_nuclear_from_pfac(P),
                    "A_MHz": np.array([above, [float(m.group(k)) for k in (5, 6, 7)], below])}
    # A "zero" D printed when none was given means none.
    if out["D_cm"] is not None and not np.any(out["D_cm"]):
        out["D_cm"] = None
    return out


def compare_with_orca(calc, orca_output, T=None) -> dict:
    """The largest difference (ppm) between this module's σ_para for every
    nucleus of ``calc`` (after :func:`apply_paramagnetic` with
    ``components="both"``) and ``orca_pnmr``'s printed tensors (a path or
    :func:`read_orca_pnmr_output`'s dict)."""
    ref = read_orca_pnmr_output(orca_output) if isinstance(orca_output, (str, os.PathLike)) \
        else orca_output
    if T is None:
        T = min(ref, key=lambda t: abs(t - float((calc.get("scaling") or {}).get("T", t))))
    table = ref[T]
    worst_iso = worst_el = 0.0
    n = 0
    for a in calc["atoms"]:
        key = a.get("orca_label")
        if key is None or key not in table or a.get("para") is None:
            continue
        mine = a["para"]["sigma"]
        worst_el = max(worst_el, float(np.max(np.abs(mine - table[key]["tensor"]))))
        worst_iso = max(worst_iso, abs(float(np.trace(mine)) / 3.0 - table[key]["iso"]))
        n += 1
    if not n:
        raise ValueError("No nucleus in common with the orca_pnmr output.")
    return {"T": T, "n": n, "max_iso_ppm": worst_iso, "max_element_ppm": worst_el}


# ---------------------------------------------------------------------------
# Temperature
# ---------------------------------------------------------------------------

def shift_vs_temperature(calc, temperatures, *, labels=None, element=None, model="curie",
                         S=None, D_cm="file", g="file", orbital_shift=0.0, **model_kwargs) -> dict:
    """The isotropic shift of each chosen nucleus over ``temperatures`` (K):
    ``{"T", "sites": {label: {"total", "contact", "pseudocontact"}}}`` (ppm;
    ``total`` includes ``orbital_shift``). ``"susceptibility"`` and
    ``"phi"`` describe one temperature and are refused here."""
    if model in ("susceptibility", "phi"):
        raise ValueError("A measured χ_M or φ belongs to one temperature: use the calculated "
                         "spin Hamiltonian, a Curie–Weiss law or μ_eff (with θ) for a "
                         "temperature dependence.")
    T = np.asarray(temperatures, dtype=float)
    if T.ndim != 1 or len(T) < 2 or np.any(T <= 0):
        raise ValueError("Give at least two positive temperatures (K).")
    sites = {}
    for t in T:
        c = apply_paramagnetic(calc, model=model, T=float(t), S=S, D_cm=D_cm, g=g,
                               **model_kwargs)
        for a in c["atoms"]:
            if a.get("para") is None:
                continue
            if element and a["species"] != element:
                continue
            if labels and a["label"] not in labels:
                continue
            s = sites.setdefault(a["label"], {"total": [], "contact": [], "pseudocontact": []})
            s["total"].append(a["para"]["total"] + float(orbital_shift))
            s["contact"].append(a["para"]["contact"])
            s["pseudocontact"].append(a["para"]["pseudocontact"])
    if not sites:
        raise ValueError("No nucleus with a hyperfine tensor matches.")
    for s in sites.values():
        for k in s:
            s[k] = np.asarray(s[k])
    return {"T": T, "sites": sites, "model": model, "orbital_shift": float(orbital_shift)}


#: How shifts known at one temperature scale with T (explicit user request,
#: 2026-11-07: shift against temperature for any calculation, not only ORCA).
SCALING_MODELS = {
    "curie": "Curie: contact and dipolar ∝ 1/T",
    "curie_weiss": "Curie–Weiss: contact and dipolar ∝ 1/(T − θ)",
    "curie_bleaney": "Curie + Bleaney: contact ∝ 1/T, dipolar (pseudocontact) ∝ 1/T²",
}


def scaled_shift_vs_temperature(sites, temperatures, *, T_ref, model="curie", theta=0.0) -> dict:
    """Shift against temperature from shifts calculated (or measured) at one
    temperature ``T_ref`` -- by any code: VASP, CASTEP, CRYSTAL, Gaussian, ORCA.

    ``sites``: ``{label: {"dia": δ_orb, "contact": δ_c, "dipolar": δ_d}}``
    (ppm at ``T_ref``; "dipolar" the isotropic pseudocontact part, 0 for a
    traceless spin-dipolar tensor). ``model`` (:data:`SCALING_MODELS`):
    both paramagnetic parts ∝ 1/T (Curie), ∝ 1/(T − θ) (Curie–Weiss), or the
    contact ∝ 1/T and the dipolar ∝ 1/T² (Bleaney's leading pseudocontact
    term for an anisotropic ion). The diamagnetic shift does not change.

    Returns the format of :func:`shift_vs_temperature` (``"T"``, ``"sites"``
    with ``total``, ``contact``, ``pseudocontact``)."""
    if model not in SCALING_MODELS:
        raise ValueError(f"model: one of {', '.join(SCALING_MODELS)}.")
    T = np.asarray(temperatures, dtype=float)
    if T.ndim != 1 or len(T) < 2 or np.any(T <= 0):
        raise ValueError("Give at least two positive temperatures (K).")
    T_ref = float(T_ref)
    if T_ref <= 0:
        raise ValueError("The reference temperature must be positive.")
    if model == "curie_weiss":
        if np.any(T - theta <= 0) or T_ref - theta <= 0:
            raise ValueError(f"T − θ must stay positive (θ = {theta:g} K): the Curie–Weiss law "
                             "holds only above the ordering temperature.")
        f_c = f_d = (T_ref - theta) / (T - theta)
    elif model == "curie":
        f_c = f_d = T_ref / T
    else:
        f_c, f_d = T_ref / T, (T_ref / T) ** 2
    if not sites:
        raise ValueError("Give at least one site.")
    out = {}
    for label, v in sites.items():
        dia = float(v.get("dia", 0.0))
        c = float(v.get("contact", 0.0)) * f_c
        d = float(v.get("dipolar", 0.0)) * f_d
        out[str(label)] = {"total": dia + c + d, "contact": c, "pseudocontact": d}
    return {"T": T, "sites": out, "model": model, "T_ref": T_ref, "theta": float(theta)}


def crystal_shift_vs_temperature(calc, temperatures, *, scaling, labels=None, element=None,
                                 orbital_shift=0.0, components="both", T_ref=None,
                                 theta=None) -> dict:
    """Shift against temperature of a CRYSTAL calculation with its magnetic
    scaling (``scaling``: the arguments of :func:`crystal_nmr.magnetic_scaling`).
    The shift is linear in φ, so δ(T) = δ(T_ref) φ(T)/φ(T_ref): with a
    Weiss constant θ and μ_eff (the Curie–Weiss route) φ ∝ 1/(T − θ) from
    the scaling itself; a φ or χ_M belongs to one temperature, ``T_ref``,
    and is carried to others by a Curie–Weiss law with ``theta`` (0: Curie)."""
    from . import crystal_nmr
    args = dict(scaling)
    args.pop("components", None)
    sc = crystal_nmr.magnetic_scaling(**args)
    if sc["route"] == "curie_weiss":
        T_ref, theta = sc["T"], float(sc["theta"])
    else:
        if T_ref is None:
            raise ValueError("A φ or χ_M belongs to one temperature: give the temperature it "
                             "was measured at (T_ref), and θ (0 for a Curie law).")
        theta = 0.0 if theta is None else float(theta)
    c = crystal_nmr.apply_scaling(calc, sc, components=components)
    sites = {}
    for a in c["atoms"]:
        if a.get("delta_hf") is None:
            continue
        if element and a["species"] != element:
            continue
        if labels and a["label"] not in labels:
            continue
        fc = a["delta_fc"] if (a.get("delta_fc") is not None
                               and components in ("both", "contact")) else 0.0
        dip = (float(np.trace(a["delta_dip"]) / 3.0) if a.get("delta_dip") is not None
               and components in ("both", "dipolar") else 0.0)
        sites[a["label"]] = {"dia": float(orbital_shift), "contact": float(fc), "dipolar": dip}
    if not sites:
        raise ValueError("No nucleus with a hyperfine shift matches.")
    out = scaled_shift_vs_temperature(sites, temperatures, T_ref=T_ref,
                                      model="curie_weiss" if theta else "curie", theta=theta)
    out["orbital_shift"] = float(orbital_shift)
    out["source"] = "CRYSTAL"
    return out


TEMPERATURE_MODELS = {
    "curie": ("δ = δ_dia + a/T", "Curie law: a contact (and pseudocontact) shift ∝ 1/T"),
    "curie_bleaney": ("δ = δ_dia + a/T + b/T²",
                      "Curie plus Bleaney's 1/T² term: the leading pseudocontact shift of an "
                      "ion with magnetic anisotropy (lanthanides); a ≈ contact, b ≈ pseudocontact"),
    "curie_weiss": ("δ = δ_dia + C/(T − θ)", "Curie–Weiss: a coupled paramagnet above its "
                                             "ordering temperature"),
}


def fit_temperature(T, delta, model="curie", delta_dia=None, sigma=None) -> dict:
    """Fit isotropic shifts δ(T) (ppm) to one of :data:`TEMPERATURE_MODELS`.

    ``delta_dia`` : fix the diamagnetic shift (e.g. from a diamagnetic
    analogue) rather than fit it. ``sigma`` : uncertainties of δ (ppm),
    for weighted fits. Linear models are solved exactly; Curie–Weiss by
    least squares started from the Curie fit. Returns parameters, standard
    errors, R², residuals and notes."""
    T = np.asarray(T, dtype=float)
    y = np.asarray(delta, dtype=float)
    if T.shape != y.shape or T.ndim != 1:
        raise ValueError("T and δ must be equal-length lists.")
    if np.any(T <= 0):
        raise ValueError("Temperatures must be positive (K).")
    if model not in TEMPERATURE_MODELS:
        raise ValueError(f"model must be one of {', '.join(TEMPERATURE_MODELS)}.")
    w = None if sigma is None else 1.0 / np.asarray(sigma, dtype=float)
    fixed = delta_dia is not None
    notes = []
    if model in ("curie", "curie_bleaney"):
        cols = [1.0 / T] + ([1.0 / T ** 2] if model == "curie_bleaney" else [])
        names = ["a"] + (["b"] if model == "curie_bleaney" else [])
        if not fixed:
            cols = [np.ones_like(T)] + cols
            names = ["delta_dia"] + names
        X = np.column_stack(cols)
        yy = y - (float(delta_dia) if fixed else 0.0)
        Xw, yw = (X, yy) if w is None else (X * w[:, None], yy * w)
        n, p = X.shape
        if n < p:
            raise ValueError(f"{model} needs at least {p} temperatures here.")
        beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
        res_w = yw - Xw @ beta
        dof = n - p
        s2 = float(res_w @ res_w / dof) if dof > 0 else float("nan")
        if w is not None and dof > 0:
            s2 = 1.0  # absolute weights: the covariance comes from sigma
        cov = s2 * np.linalg.pinv(Xw.T @ Xw)
        params = dict(zip(names, beta.tolist()))
        errs = dict(zip(names, np.sqrt(np.clip(np.diag(cov), 0, None)).tolist()))
        if fixed:
            params["delta_dia"], errs["delta_dia"] = float(delta_dia), 0.0

        def f(t, P=params):
            t = np.asarray(t, dtype=float)
            return P["delta_dia"] + P["a"] / t + (P.get("b", 0.0) / t ** 2)
        if dof == 0:
            notes.append("As many parameters as points: the curve passes through every point "
                         "and no uncertainty can be given.")
        if model == "curie_bleaney" and len(np.unique(T)) >= 3:
            span = T.max() / T.min()
            if span < 1.3:
                notes.append(f"T spans only a factor {span:.2f}: 1/T and 1/T² are nearly "
                             "collinear, so a and b (contact and pseudocontact) are poorly "
                             "separated; see their errors.")
    else:
        from scipy.optimize import curve_fit
        start = fit_temperature(T, y, "curie", delta_dia=delta_dia)
        p0 = [start["params"]["a"], min(0.0, 0.5 * T.min())]
        if fixed:
            def model_f(t, C, th):
                return float(delta_dia) + C / (t - th)
        else:
            p0 = [start["params"]["delta_dia"]] + p0

            def model_f(t, d0, C, th):
                return d0 + C / (t - th)
        upper = [np.inf] * len(p0)
        upper[-1] = 0.999 * T.min()
        lower = [-np.inf] * len(p0)
        if len(T) < len(p0):
            raise ValueError(f"Curie–Weiss needs at least {len(p0)} temperatures here.")
        popt, pcov = curve_fit(model_f, T, y, p0=p0, sigma=None if sigma is None else sigma,
                               absolute_sigma=sigma is not None, bounds=(lower, upper),
                               maxfev=20000)
        names = (["C", "theta"] if fixed else ["delta_dia", "C", "theta"])
        params = dict(zip(names, map(float, popt)))
        errs = dict(zip(names, np.sqrt(np.clip(np.diag(pcov), 0, None)).tolist()))
        if fixed:
            params["delta_dia"], errs["delta_dia"] = float(delta_dia), 0.0
        if params["theta"] > 0.99 * T.min() * 0.999:
            notes.append("θ reached its bound just below the lowest temperature: the data do "
                         "not follow a Curie–Weiss law there.")

        def f(t, P=params):
            t = np.asarray(t, dtype=float)
            return P["delta_dia"] + P["C"] / (t - P["theta"])
        dof = len(T) - len(popt)
    fitted = f(T)
    resid = y - fitted
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid ** 2)) / ss_tot if ss_tot > 0 else float("nan")
    return {"model": model, "equation": TEMPERATURE_MODELS[model][0], "params": params,
            "errors": errs, "r2": r2, "residuals": resid, "T": T, "delta": y, "dof": int(dof),
            "function": f, "fixed_delta_dia": fixed, "notes": notes}


#: Legend forms of the equations (mathtext, symbols italic).
_EQUATION_TEX = {
    "curie": r"$\delta = \delta_\mathrm{dia} + a/T$",
    "curie_bleaney": r"$\delta = \delta_\mathrm{dia} + a/T + b/T^2$",
    "curie_weiss": r"$\delta = \delta_\mathrm{dia} + C/(T - \theta)$",
}


def _italic(text):
    from opennmr.plotting.labels import italicise_symbols
    return italicise_symbols(text)


def plot_shift_temperature(curves=None, data=None, fits=None, figure=None, x_axis="T",
                           fontsize=11.0, components=False):
    """Shift against temperature: ``curves`` (from :func:`shift_vs_temperature`),
    measured ``data`` ({name: (T, δ)}) and ``fits`` ({name: fit_temperature
    result}). ``x_axis`` : ``"T"`` or ``"1/T"`` (1000/T, K⁻¹: a Curie law is a
    straight line). ``components`` : also the contact and pseudocontact
    parts of each calculated curve (dashed, dotted)."""
    import matplotlib.pyplot as plt
    if x_axis not in ("T", "1/T"):
        raise ValueError("x_axis must be 'T' or '1/T'.")
    fig = figure if figure is not None else plt.figure(figsize=(6.0, 4.4))
    fig.clear()
    ax = fig.add_subplot(111)
    cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2"])
    k = 0

    def X(t):
        t = np.asarray(t, dtype=float)
        return 1000.0 / t if x_axis == "1/T" else t
    if curves:
        for label, s in curves["sites"].items():
            c = cycle[k % len(cycle)]
            k += 1
            ax.plot(X(curves["T"]), s["total"], color=c, lw=1.6, label=f"{label} (calc.)")
            if components:
                ax.plot(X(curves["T"]), s["contact"], color=c, lw=1.0, ls="--",
                        label=f"{label} contact")
                ax.plot(X(curves["T"]), s["pseudocontact"], color=c, lw=1.0, ls=":",
                        label=f"{label} pseudocontact")
    for name, (t, d) in (data or {}).items():
        c = cycle[k % len(cycle)]
        k += 1
        ax.plot(X(t), d, "o", color=c, ms=5, label=name)
        fit = (fits or {}).get(name)
        if fit is not None:
            tt = np.linspace(np.min(t), np.max(t), 200)
            ax.plot(X(tt), fit["function"](tt), color=c, lw=1.2,
                    label=f"{name}: " + _EQUATION_TEX[fit["model"]])
    for name, fit in (fits or {}).items():
        if name in (data or {}):
            continue
        c = cycle[k % len(cycle)]
        k += 1
        tt = np.linspace(fit["T"].min(), fit["T"].max(), 200)
        ax.plot(X(fit["T"]), fit["delta"], "o", color=c, ms=5, label=name)
        ax.plot(X(tt), fit["function"](tt), color=c, lw=1.2,
                    label=f"{name}: " + _EQUATION_TEX[fit["model"]])
    ax.set_xlabel(_italic(r"$T$ (K)") if x_axis == "T" else _italic(r"1000/$T$ (K$^{-1}$)"),
                  fontsize=fontsize)
    ax.set_ylabel(_italic(r"$\delta$ (ppm)"), fontsize=fontsize)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=fontsize * 0.8, frameon=False)
    return fig


def plot_contributions(rows, figure=None, fontsize=11.0, orbital=None):
    """Bars of the contact and pseudocontact isotropic shifts of each site
    (``rows`` from :func:`contribution_rows`), and the orbital shift if
    given ({label: ppm})."""
    import matplotlib.pyplot as plt
    if not rows:
        raise ValueError("No paramagnetic sites to show.")
    fig = figure if figure is not None else plt.figure(figsize=(max(4.5, 0.6 * len(rows) + 2),
                                                                4.2))
    fig.clear()
    ax = fig.add_subplot(111)
    cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2"])
    labels = [r["label"] for r in rows]
    x = np.arange(len(rows))
    series = [("contact", "Fermi contact"), ("pseudocontact", "Pseudocontact (spin dipolar)")]
    if orbital:
        series.append(("orbital", "Orbital"))
    width = 0.8 / len(series)
    for j, (key, name) in enumerate(series):
        vals = [(orbital or {}).get(r["label"], 0.0) if key == "orbital" else r[key] for r in rows]
        ax.bar(x + (j - (len(series) - 1) / 2) * width, vals, width, label=name,
               color=cycle[j % len(cycle)])
    ax.axhline(0.0, color="0.3", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45 if len(rows) > 6 else 0,
                       ha="right" if len(rows) > 6 else "center")
    ax.set_ylabel(_italic(r"$\delta_\mathrm{iso}$ (ppm)"), fontsize=fontsize)
    ax.legend(fontsize=fontsize * 0.8, frameon=False)
    return fig


def paramagnetic_to_python(path, *, element, labels=None, nucleus=None, paramagnetic=None,
                           orbital_shift=0.0, **site_kwargs) -> str:
    """Notebook code for an ORCA EPR import: read, apply, sites."""
    kw = ", ".join(f"{k}={v!r}" for k, v in dict(element=element, labels=labels,
                                                  nucleus=nucleus, **site_kwargs).items()
                   if v is not None)
    return ("from opennmr.notebook import dft_nmr, paramagnetic_nmr as pm\n"
            f"sample = dft_nmr.read_calculation({str(path)!r})\n"
            f"model = dft_nmr.dft_sites(sample, paramagnetic={dict(paramagnetic or {})!r}, "
            f"orbital_shift={float(orbital_shift)!r}, {kw})\n"
            "print(dft_nmr.format_table(model))\n"
            "for row in pm.contribution_rows(model):\n"
            "    print(row)\n")
