"""Pseudocontact shifts from a magnetic susceptibility tensor (point dipole).

A paramagnetic centre with molecular susceptibility tensor χ (m³ per
molecule, SI) at the origin shifts a nucleus at r by the dipolar field of
its induced moment, m = χ B0 / μ0:

    δ (tensor) = (1 / 4π r³) (3 r̂ r̂ᵀ − 1) χ ,
    δ_PCS (isotropic) = (1 / 12π r³) [3 r̂ᵀ χ r̂ − tr χ]
                      = (1 / 12π r³) [Δχ_ax (3cos²θ − 1) + 3/2 Δχ_rh sin²θ cos 2φ],

with Δχ_ax = χ_zz − (χ_xx + χ_yy)/2 and Δχ_rh = χ_xx − χ_yy in χ's principal
frame (θ, φ the nucleus's polar angles there). Only the anisotropy of χ
gives an isotropic shift; the isotropic χ still gives the (traceless)
dipolar shift *tensor* -- the spinning-sideband pattern of a paramagnetic
solid. This is the metal-centred ("point dipole") approximation: no spin
delocalisation (contact shift) and no ligand-centred dipolar part, both of
which the hyperfine route (:mod:`opennmr.notebook.paramagnetic_nmr`) has.

χ can come from: an ORCA CASSCF/NEVPT2 run (``dosusceptibility``: the
molar χT tensor against temperature, :func:`read_orca_susceptibility`);
Δχ_ax, Δχ_rh and Euler angles (:func:`chi_from_axial_rhombic`, as fitted in
protein PCS work, 10⁻³² m³); g and the spin susceptibility Z
(:func:`chi_from_g`); or a fit to measured shifts (:func:`fit_chi_tensor`).

In a periodic solid the sum over the paramagnetic ions converges only
conditionally. :func:`pcs_shifts` sums the ions inside a sphere of radius
R (a Lorentz sphere) and adds, optionally, the bulk term of the magnetised
medium outside it, δ_bulk = (1/3 − D) χ_v, with D the demagnetising
factor of the sample along B0 (a sphere, or a long rotor at the magic
angle: 1/3, so δ_bulk = 0; a long cylinder along B0: 0). This bulk term
is what the BMS simulator (:mod:`opennmr.notebook.bms`) computes in detail
for a real particle packing; here only its mean.
"""

from __future__ import annotations

import math
import os
import re

import numpy as np

from .crystal_nmr import MU_0, MU_B, N_A

__all__ = [
    "chi_molecule_from_molar_cgs", "chi_from_axial_rhombic", "axial_rhombic", "chi_from_g",
    "read_orca_susceptibility", "read_orca_sus_file", "chi_at_temperature", "pcs_tensor",
    "pcs_iso", "pcs_shifts", "fit_chi_tensor", "pcs_map", "plot_pcs_map", "SAMPLE_SHAPES",
    "rotation_zyz", "pcs_to_python",
]

#: Δχ unit customary in PCS work.
UNIT_1E32 = 1e-32

SAMPLE_SHAPES = {
    None: ("Near field only (no bulk term)", None),
    "sphere": ("Sphere (D = 1/3): bulk term 0", 1.0 / 3.0),
    "mas_rotor": ("Long rotor at the magic angle (D = 1/3 along B0): bulk term 0", 1.0 / 3.0),
    "cylinder_parallel": ("Long cylinder along B0 (static, D = 0): +χ_v/3", 0.0),
    "cylinder_perpendicular": ("Long cylinder perpendicular to B0 (D = 1/2): −χ_v/6", 0.5),
}

_FLOAT = r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[EeDd][-+]?\d+)?"


def _floats(line):
    return [float(v.replace("D", "E")) for v in re.findall(_FLOAT, line)]


# ---------------------------------------------------------------------------
# χ tensors
# ---------------------------------------------------------------------------

def chi_molecule_from_molar_cgs(chi_cgs):
    """Molar susceptibility (cm³/mol, cgs; scalar or tensor) → m³ per
    molecule (SI): × 4π × 10⁻⁶ / N_A."""
    return 4.0 * math.pi * 1e-6 * np.asarray(chi_cgs, dtype=float) / N_A


def rotation_zyz(alpha, beta, gamma) -> np.ndarray:
    """Active rotation R = Rz(α) Ry(β) Rz(γ), degrees: its columns are the
    tensor frame's axes in the molecular frame."""
    a, b, c = np.radians([alpha, beta, gamma])

    def rz(t):
        return np.array([[math.cos(t), -math.sin(t), 0], [math.sin(t), math.cos(t), 0], [0, 0, 1]])
    ry = np.array([[math.cos(b), 0, math.sin(b)], [0, 1, 0], [-math.sin(b), 0, math.cos(b)]])
    return rz(a) @ ry @ rz(c)


def chi_from_axial_rhombic(dchi_ax, dchi_rh=0.0, euler=(0.0, 0.0, 0.0), chi_iso=0.0,
                           unit=UNIT_1E32) -> np.ndarray:
    """χ (m³ per molecule) from Δχ_ax, Δχ_rh (in ``unit``, default 10⁻³²
    m³), the ZYZ Euler angles (degrees) of its frame and an isotropic part."""
    ax, rh = float(dchi_ax) * unit, float(dchi_rh) * unit
    d = np.diag([-ax / 3.0 + rh / 2.0, -ax / 3.0 - rh / 2.0, 2.0 * ax / 3.0])
    R = rotation_zyz(*euler)
    return R @ d @ R.T + float(chi_iso) * unit * np.eye(3)


def axial_rhombic(chi, unit=UNIT_1E32) -> dict:
    """Δχ_ax, Δχ_rh (in ``unit``), the isotropic part and the principal axes
    of χ, in the usual convention: z the axis of the largest |anisotropic|
    eigenvalue, 0 ≤ Δχ_rh/Δχ_ax ≤ 2/3."""
    c = np.asarray(chi, dtype=float)
    c = 0.5 * (c + c.T)
    iso = float(np.trace(c) / 3.0)
    w, v = np.linalg.eigh(c - iso * np.eye(3))
    order = np.argsort(np.abs(w))
    w, v = w[order], v[:, order]
    ax = 1.5 * w[2]
    rh = w[0] - w[1]
    if ax != 0 and rh / ax < 0:
        w[[0, 1]] = w[[1, 0]]
        v[:, [0, 1]] = v[:, [1, 0]]
        rh = -rh
    if np.linalg.det(v) < 0:
        v[:, 0] = -v[:, 0]
    return {"dchi_ax": ax / unit, "dchi_rh": rh / unit, "chi_iso": iso / unit, "axes": v,
            "rhombicity": float(rh / ax) if ax else 0.0, "unit": unit}


def chi_from_g(g, Z) -> np.ndarray:
    """χ (m³ per molecule) = μ0 μ_B² g Z gᵀ, from the g tensor and the spin
    susceptibility Z (J⁻¹, :func:`paramagnetic_nmr.spin_response`)."""
    g = np.asarray(g, dtype=float)
    return MU_0 * MU_B ** 2 * g @ np.asarray(Z, dtype=float) @ g.T


# ---------------------------------------------------------------------------
# ORCA susceptibility
# ---------------------------------------------------------------------------

def _orca_atoms(lines):
    idx = [i for i, ln in enumerate(lines) if "CARTESIAN COORDINATES (ANGSTROEM)" in ln]
    atoms = []
    if not idx:
        return atoms
    seen = {}
    for line in lines[idx[-1] + 2:]:
        parts = line.split()
        if len(parts) != 4:
            break
        sym = re.sub(r"[^A-Za-z]", "", parts[0])
        sym = sym[:1].upper() + sym[1:].lower()
        seen[sym] = seen.get(sym, 0) + 1
        atoms.append({"species": sym, "label": f"{sym}{seen[sym]}", "index": len(atoms),
                      "position": np.array([float(v) for v in parts[1:]])})
    return atoms


def read_orca_susceptibility(path) -> dict:
    """The "TEMPERATURE DEPENDENT MOLAR MAGNETIC SUSCEPTIBILITY TENSOR"
    blocks of an ORCA output (CASSCF/NEVPT2 ``dosusceptibility``): for each
    block, the temperatures and the χT tensors (cm³ K/mol, molecular
    frame), with the method heading that precedes it; and the atoms (Å).
    Returns ``{"blocks": [{"method", "T", "chiT"}], "atoms", "path"}``."""
    with open(path, "r", errors="replace") as f:
        lines = f.read().splitlines()
    blocks = []
    method = ""
    current = None
    i = 0
    while i < len(lines):
        line = lines[i]
        m = re.search(r"(QDPT WITH\s+\S+|NEVPT2|CASSCF)", line)
        if m and "TENSOR" not in line:
            method = m.group(1).strip()
        if "TEMPERATURE DEPENDENT MOLAR MAGNETIC SUSCEPTIBILITY TENSOR" in line:
            current = {"method": method, "T": [], "chiT": []}
            blocks.append(current)
        elif current is not None:
            mt = re.match(r"\s*TEMPERATURE/K:\s*(" + _FLOAT + ")", line)
            if mt and i + 4 < len(lines) and "Tensor in molecular frame" in lines[i + 1]:
                rows = [_floats(lines[i + k]) for k in (2, 3, 4)]
                if all(len(r) == 3 for r in rows):
                    current["T"].append(float(mt.group(1)))
                    current["chiT"].append(rows)
                    i += 4
            elif re.match(r"\s*-{20,}", line) and current["T"] and \
                    "TEMPERATURE/K" not in "".join(lines[i + 1:i + 4]):
                current = None
        i += 1
    blocks = [b for b in blocks if b["T"]]
    for b in blocks:
        b["T"] = np.asarray(b["T"])
        b["chiT"] = np.asarray(b["chiT"])
    if not blocks:
        raise ValueError(f"{os.path.basename(str(path))}: no 'TEMPERATURE DEPENDENT MOLAR "
                         "MAGNETIC SUSCEPTIBILITY TENSOR' block (ORCA %casscf/%nevpt2 with "
                         "rel/soc dosusceptibility).")
    return {"blocks": blocks, "atoms": _orca_atoms(lines), "path": str(path)}


def read_orca_sus_file(path) -> dict:
    """An ORCA ``.sus`` file: field (G), T (K) and χT (cm³ K/mol, the
    isotropic average) per row."""
    T, chiT, B = [], [], []
    with open(path, "r", errors="replace") as f:
        for line in f:
            v = _floats(line.replace("-" * 15, " "))
            if len(v) >= 3:
                B.append(v[0])
                T.append(v[1])
                chiT.append(v[-1])
    if not T:
        raise ValueError(f"{os.path.basename(str(path))}: no rows of field, T, χT.")
    return {"field_G": np.asarray(B), "T": np.asarray(T), "chiT": np.asarray(chiT)}


def chi_at_temperature(sus, T, block=-1) -> dict:
    """The molar χ tensor (cm³/mol) at ``T`` from :func:`read_orca_susceptibility`
    (χT interpolated linearly between the printed temperatures), and the
    molecular one (m³)."""
    b = sus["blocks"][block]
    Ts = b["T"]
    if T < Ts.min() or T > Ts.max():
        raise ValueError(f"T = {T:g} K is outside the calculated range "
                         f"{Ts.min():g}–{Ts.max():g} K.")
    chiT = np.array([[np.interp(T, Ts, b["chiT"][:, k, l]) for l in range(3)] for k in range(3)])
    molar = chiT / float(T)
    molar = 0.5 * (molar + molar.T)
    return {"T": float(T), "molar_cgs": molar, "chi": chi_molecule_from_molar_cgs(molar),
            "method": b["method"], "chiT_iso": float(np.trace(chiT) / 3.0)}


# ---------------------------------------------------------------------------
# Shifts
# ---------------------------------------------------------------------------

def pcs_tensor(r_A, chi) -> np.ndarray:
    """δ (ppm tensor) at r (Å, from the paramagnetic centre) for χ (m³)."""
    r = np.asarray(r_A, dtype=float) * 1e-10
    d = float(np.linalg.norm(r))
    if d <= 0:
        raise ValueError("The nucleus is at the paramagnetic centre.")
    u = r / d
    return 1e6 * (3.0 * np.outer(u, u) - np.eye(3)) @ np.asarray(chi, dtype=float) / (
        4.0 * math.pi * d ** 3)


def pcs_iso(r_A, chi):
    """δ_PCS (ppm) at r (Å; one vector or an (n, 3) array) for χ (m³)."""
    r = np.atleast_2d(np.asarray(r_A, dtype=float)) * 1e-10
    d = np.linalg.norm(r, axis=1)
    c = np.asarray(chi, dtype=float)
    c = 0.5 * (c + c.T)
    with np.errstate(divide="ignore", invalid="ignore"):
        u = r / d[:, None]
        val = 1e6 * (3.0 * np.einsum("ni,ij,nj->n", u, c, u) - np.trace(c)) / (
            12.0 * math.pi * d ** 3)
    out = np.where(d > 0, val, np.nan)
    return out if np.ndim(r_A) > 1 else float(out[0])


def _translations(lattice, radius):
    L = np.asarray(lattice, dtype=float)
    inv = np.linalg.inv(L)
    # how many cells along each vector can lie within the radius
    n = [int(math.ceil(radius * np.linalg.norm(inv[:, k]))) + 1 for k in range(3)]
    grid = np.array(np.meshgrid(*[np.arange(-k, k + 1) for k in n], indexing="ij")).reshape(3, -1).T
    return grid @ L


def pcs_shifts(nuclei, metals, chi, *, lattice=None, radius=40.0, sample_shape=None,
               demag=None, min_distance=0.3) -> dict:
    """Point-dipole shifts at ``nuclei`` from paramagnetic ``metals``.

    ``nuclei`` / ``metals`` : lists of atoms (dicts with ``label`` and
    ``position``, Å) or (n, 3) arrays. ``chi`` : one χ (m³) for every
    metal, or one per metal. ``lattice`` : rows a, b, c (Å) for a periodic
    structure -- every image of every metal within ``radius`` (Å) of the
    nucleus is summed, and the convergence is checked against 0.75 R.
    ``sample_shape`` (:data:`SAMPLE_SHAPES`) or ``demag`` (D along B0) adds
    the bulk term (1/3 − D) χ_v, χ_v the isotropic volume susceptibility
    of the paramagnetic ions in the cell (periodic only).

    Returns per nucleus: ``iso`` (near field + bulk), ``near_iso``,
    ``tensor`` (near field), ``bulk``, ``n_terms``, ``convergence``
    (|iso(R) − iso(0.75 R)|, ppm) and notes."""
    def as_list(objs, kind):
        if isinstance(objs, np.ndarray) or (objs and not isinstance(objs[0], dict)):
            arr = np.atleast_2d(np.asarray(objs, dtype=float))
            return [{"label": f"{kind}{k + 1}", "position": p} for k, p in enumerate(arr)]
        return list(objs)
    nuc = as_list(nuclei, "n")
    met = as_list(metals, "M")
    if not nuc or not met:
        raise ValueError("Give at least one nucleus and one paramagnetic centre.")
    chis = np.asarray(chi, dtype=float)
    chis = np.repeat(chis[None], len(met), axis=0) if chis.ndim == 2 else chis
    if len(chis) != len(met):
        raise ValueError("Give one χ, or one per paramagnetic centre.")
    mpos = np.array([m["position"] for m in met], dtype=float)
    notes = []
    if lattice is not None:
        T = _translations(lattice, radius + float(np.ptp(mpos, axis=0).max() if len(mpos) else 0)
                          + 1.0)
    else:
        T = np.zeros((1, 3))
    D = demag
    if D is None and sample_shape is not None:
        if sample_shape not in SAMPLE_SHAPES:
            raise ValueError(f"sample_shape must be one of {[k for k in SAMPLE_SHAPES if k]}.")
        D = SAMPLE_SHAPES[sample_shape][1]
    bulk = 0.0
    if D is not None:
        if lattice is None:
            notes.append("A bulk (demagnetising) term needs a periodic structure; none added.")
        else:
            V = abs(float(np.linalg.det(np.asarray(lattice, dtype=float)))) * 1e-30
            chi_v = float(sum(np.trace(c) / 3.0 for c in chis)) / V
            bulk = 1e6 * (1.0 / 3.0 - float(D)) * chi_v
            notes.append(f"Bulk term (1/3 − D) χ_v = {bulk:.4g} ppm (D = {D:g}, χ_v = "
                         f"{chi_v:.4g} SI).")
    out = []
    for n in nuc:
        p = np.asarray(n["position"], dtype=float)
        tens = np.zeros((3, 3))
        iso_far = iso_near = 0.0
        terms = 0
        for k in range(len(met)):
            r = mpos[k][None, :] + T - p[None, :]
            d = np.linalg.norm(r, axis=1)
            keep = (d >= min_distance) & (d <= radius)
            if not np.any(keep):
                continue
            rr = r[keep]
            # vector from the metal to the nucleus is −rr
            iso_vals = pcs_iso(-rr, chis[k])
            iso_far += float(np.sum(iso_vals))
            iso_near += float(np.sum(iso_vals[d[keep] <= 0.75 * radius]))
            for v in -rr:
                tens += pcs_tensor(v, chis[k])
            terms += int(np.sum(keep))
        rec = {"label": n.get("label"), "near_iso": iso_far, "iso": iso_far + bulk, "bulk": bulk,
               "tensor": tens, "n_terms": terms,
               "convergence": abs(iso_far - iso_near) if lattice is not None else 0.0}
        out.append(rec)
    if lattice is not None:
        worst = max((r["convergence"] for r in out), default=0.0)
        notes.append(f"Lattice sum over a sphere of radius {radius:g} Å; the isotropic shifts "
                     f"change by up to {worst:.3g} ppm between 0.75 R and R.")
    return {"sites": out, "notes": notes, "bulk": bulk, "radius": float(radius),
            "periodic": lattice is not None}


def fit_chi_tensor(nuclei, metal, delta_pcs, *, weights=None) -> dict:
    """The anisotropic χ (5 parameters, m³) that best gives the measured
    pseudocontact shifts ``delta_pcs`` (ppm) of ``nuclei`` (Å, (n, 3) or
    atoms) around one centre at ``metal`` (Å). Linear least squares (the
    shift is linear in χ), so the answer is unique when n ≥ 5 and the
    nuclei are not all on a cone. Returns χ, Δχ_ax/Δχ_rh/axes, errors, the
    back-calculated shifts and Q = rms(res)/rms(δ)."""
    pos = np.array([n["position"] if isinstance(n, dict) else n for n in nuclei], dtype=float)
    y = np.asarray(delta_pcs, dtype=float)
    if len(pos) != len(y):
        raise ValueError("One shift per nucleus.")
    if len(y) < 5:
        raise ValueError("At least five shifts are needed for the five parameters of Δχ.")
    basis = [np.diag([1.0, 0, -1.0]), np.diag([0, 1.0, -1.0])]
    for (i, j) in ((0, 1), (0, 2), (1, 2)):
        b = np.zeros((3, 3))
        b[i, j] = b[j, i] = 1.0
        basis.append(b)
    r = pos - np.asarray(metal, dtype=float)[None, :]
    X = np.column_stack([pcs_iso(r, b * 1e-32) for b in basis])   # ppm per 1e-32 m³
    w = np.ones_like(y) if weights is None else np.asarray(weights, dtype=float)
    beta, *_ = np.linalg.lstsq(X * w[:, None], y * w, rcond=None)
    pred = X @ beta
    res = y - pred
    dof = len(y) - 5
    s2 = float(np.sum((w * res) ** 2) / dof) if dof > 0 else float("nan")
    cov = s2 * np.linalg.pinv((X * w[:, None]).T @ (X * w[:, None]))
    chi = sum(b * c for b, c in zip(basis, beta)) * 1e-32
    ar = axial_rhombic(chi)
    q = float(np.sqrt(np.mean(res ** 2)) / np.sqrt(np.mean(y ** 2))) if np.any(y) else float("nan")
    notes = []
    if np.linalg.matrix_rank(X) < 5:
        notes.append("The nuclei do not determine all five parameters (too few, or too "
                     "symmetric an arrangement).")
    return {"chi": chi, **ar, "parameters": beta, "cov": cov, "predicted": pred,
            "residuals": res, "Q": q, "dof": dof, "notes": notes}


# ---------------------------------------------------------------------------
# Pictures
# ---------------------------------------------------------------------------

def pcs_map(chi, plane="xz", extent=10.0, n=241, frame="principal"):
    """δ_PCS (ppm) on a square grid (±``extent`` Å) through the centre, in
    the plane of two axes (``"xy"``, ``"xz"``, ``"yz"``) of χ's principal
    frame (``frame="principal"``) or the molecular frame. Returns (u, v,
    δ, axes) with ``axes`` the 3×3 whose first two columns span the plane."""
    if plane not in ("xy", "xz", "yz"):
        raise ValueError("plane must be 'xy', 'xz' or 'yz'.")
    R = axial_rhombic(chi)["axes"] if frame == "principal" else np.eye(3)
    i, j = "xyz".index(plane[0]), "xyz".index(plane[1])
    k = 3 - i - j
    axes = R[:, [i, j, k]]
    u = np.linspace(-extent, extent, n)
    U, V = np.meshgrid(u, u, indexing="xy")
    pts = U.ravel()[:, None] * axes[:, 0][None, :] + V.ravel()[:, None] * axes[:, 1][None, :]
    vals = pcs_iso(pts, chi).reshape(U.shape)
    return u, u, vals, axes


def _level_text(x):
    return f"{x:.0f}" if abs(x) >= 10 else f"{x:.2g}"


def plot_pcs_map(chi, figure=None, plane="xz", extent=10.0, frame="principal", atoms=None,
                 centre=(0.0, 0.0, 0.0), slab=1.0, levels=None, fontsize=11.0, cmap="RdBu_r",
                 label_atoms=True):
    """Contours of δ_PCS in a plane through the paramagnetic centre (the
    familiar double cone), with the atoms within ±``slab`` Å of the plane
    marked. Positive (deshielded) red, negative blue, by default; levels
    symmetric and logarithmically spaced (the shift falls as r⁻³)."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import SymLogNorm
    fig = figure if figure is not None else plt.figure(figsize=(5.6, 5.0))
    fig.clear()
    ax = fig.add_subplot(111)
    u, v, d, axes = pcs_map(chi, plane, extent, frame=frame)
    # Inside the ion the point dipole means nothing; leave a hole there.
    UU, VV = np.meshgrid(u, v, indexing="xy")
    hole = min(1.5, 0.15 * float(extent))
    d = np.where(np.hypot(UU, VV) < hole, np.nan, d)
    finite = np.abs(d[np.isfinite(d)])
    # Clip close to the centre, where r⁻³ diverges.
    top = float(np.nanpercentile(finite, 99.0)) if finite.size else 1.0
    if levels is None:
        lo = max(top * 1e-3, 1e-6)
        pos = np.geomspace(lo, top, 6)
        levels = np.concatenate([-pos[::-1], pos])
    dd = np.clip(d, -top, top)
    norm = SymLogNorm(linthresh=max(top * 1e-3, 1e-9), vmin=-top, vmax=top)
    cf = ax.contourf(u, v, dd, levels=np.concatenate([[-top * 1.0001], levels, [top * 1.0001]])
                     if levels[0] > -top else levels, cmap=cmap, norm=norm, extend="both")
    cs = ax.contour(u, v, dd, levels=levels, colors="k", linewidths=0.5)
    # Label the outer half of the levels only: the inner ones crowd.
    outer = [lv for lv in levels if abs(lv) <= np.sort(np.abs(levels))[len(levels) // 2]]
    ax.clabel(cs, levels=outer, fmt=_level_text, fontsize=fontsize * 0.6)
    ax.plot(0, 0, "o", color="k", ms=6)
    if atoms:
        c = np.asarray(centre, dtype=float)
        for a in atoms:
            r = np.asarray(a["position"], dtype=float) - c
            x, y, z = r @ axes[:, 0], r @ axes[:, 1], r @ axes[:, 2]
            if abs(z) <= slab and abs(x) <= extent and abs(y) <= extent and np.linalg.norm(r) > 0.1:
                ax.plot(x, y, "o", mfc="white", mec="k", ms=5)
                if label_atoms:
                    ax.annotate(a.get("label", ""), (x, y), xytext=(3, 3),
                                textcoords="offset points", fontsize=fontsize * 0.6)
    sub = r"_{\chi}" if frame == "principal" else ""
    ax.set_xlabel(rf"$\mathit{{{plane[0]}}}{sub}$ (Å)", fontsize=fontsize)
    ax.set_ylabel(rf"$\mathit{{{plane[1]}}}{sub}$ (Å)", fontsize=fontsize)
    ax.set_aspect("equal")
    cb = fig.colorbar(cf, ax=ax, ticks=levels)
    cb.ax.set_yticklabels([_level_text(lv) for lv in levels])
    from opennmr.plotting.labels import italicise_symbols
    cb.set_label(italicise_symbols(r"$\delta_\mathrm{PCS}$ (ppm)"), fontsize=fontsize)
    return fig


def pcs_to_python(**kw) -> str:
    """Notebook code for a point-dipole calculation (``kw``: the arguments
    of the GUI: ``source``, ``chi`` / ``axial`` … , ``structure``)."""
    lines = ["from opennmr.notebook import pseudocontact as pcs, dft_nmr", "import numpy as np"]
    src = kw.get("source")
    if src == "orca":
        lines.append(f"sus = pcs.read_orca_susceptibility({kw['path']!r})")
        lines.append(f"chi = pcs.chi_at_temperature(sus, {kw['T']!r}, block={kw.get('block', -1)!r})['chi']")
    elif src == "axial":
        lines.append(f"chi = pcs.chi_from_axial_rhombic({kw['dchi_ax']!r}, {kw['dchi_rh']!r}, "
                     f"euler={tuple(kw['euler'])!r})")
    else:
        lines.append(f"chi = np.array({np.asarray(kw['chi']).tolist()!r})")
    if kw.get("structure"):
        lines.append(f"calc = dft_nmr.read_calculation({kw['structure']!r})")
        lines.append(f"metals = [a for a in calc['atoms'] if a['label'] in {kw['metals']!r}]")
        lines.append(f"nuclei = [a for a in calc['atoms'] if a['species'] == {kw['element']!r}]")
        lines.append(f"res = pcs.pcs_shifts(nuclei, metals, chi, lattice=calc.get('lattice'), "
                     f"radius={kw.get('radius', 40.0)!r}, sample_shape={kw.get('sample_shape')!r})")
        lines.append("for s in res['sites']:\n    print(s['label'], round(s['iso'], 3))")
    return "\n".join(lines) + "\n"
