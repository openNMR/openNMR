"""
Fast bulk magnetic susceptibility (BMS) shifts of a particle packing
(explicit user request, 2026-11-07: "a fast version of the BMS shift
simulator ... maximise efficiency, however that is best done").

Why this is fast, and still the same physics
--------------------------------------------
:func:`opennmr.notebook.bms.main.run_bms` computes every particle's
demagnetising field on its own: per particle, the shape on the whole grid,
one forward and six inverse 3-d FFTs, then a trilinear interpolation of all
grid points. Its cost grows with the number of particles.

The shift is linear in the magnetisation (first order in χ ≪ 1), so

    Σ_p χ_p · N_p(r) = IFFT[ FFT(X)(k) · k kᵀ/|k|² ](r),   X(r) = Σ_p χ_p s_p(r),

where s_p is particle p's shape function and χ_p its susceptibility
tensor in the laboratory frame. All particles share one set of FFTs: at
most six forward transforms of X (one if every particle has an isotropic
χ), and one inverse transform per output component (one for the static
δ_zz or for δ_iso, six for the full tensor that MAS sidebands need). The
cost depends on the grid only, not on the number of particles. In
addition:

* each particle's shape is evaluated only on the small block of grid
  around it (its bounding radius plus a margin), not on the whole grid;
* real-to-complex FFTs (``scipy.fft.rfftn``) on all cores, in single
  precision by default (``dtype="float32"``: shifts of ~10³ ppm to
  ~10⁻⁴ ppm; ``"float64"`` is there for checking);
* the spectrum uses only the voxels that carry signal.

The conventions (quaternions, χ tensor rotation, shape functions) are
those of the original code, reused from ``io_utils.inp_part`` and
``geometry`` (the shapes are evaluated by :func:`block_shape`, equal to
``geometry.shape_function`` to ~1e-13). The Lorentz-sphere term −χ_p/3
inside particle p (``shift.shielding_bms``) is applied in proportion to
each voxel's occupancy by default; see ``lorentz`` in :func:`bms_field`.

Checks against the original ``run_bms``: the same field (to ~0.05 ppm)
for a particle centred on a grid node. For a particle off the nodes --
every particle in a real packing -- the original interpolates each
particle's field trilinearly across its sharp surface, so surface voxels
fall below its 0.99 "inside" threshold and lose the Lorentz term
(1e6·tr χ/9 ≈ 920 ppm for χ ≈ 3e-3). Moving one particle by half a voxel
changes its mean inside shift from +25 to −8 ppm there (2026-11-07);
here it does not move beyond the discretisation (~1 %).

Boundary conditions and the macroscopic sample
----------------------------------------------
* ``boundary="periodic"`` (default): the box is one cell of a packing
  that fills space -- no particle sees an edge of the "sample". The
  k = 0 term of the kernel is then the macroscopic demagnetising tensor
  D of the whole sample, set by ``sample_shape``: the mean field is
  −D·⟨χ⟩. The Lorentz sphere is then −χ/3 inside particles, so a
  spherical sample (D = 1/3) has no bulk shift, a long cylinder along B₀
  (D_zz = 0) shifts everything by +⟨χ_zz⟩/3, and so on. Under MAS the
  isotropic part of any D is 1/3 (tr D = 1), so ``sample_shape`` only
  matters for static spectra.
* ``boundary="isolated"``: the particles in vacuum, as the original
  run_bms does (its per-particle box plus point-dipole far field), by
  zero padding to twice the box. Eight times the voxels -- use it to
  compare with the original code, or for a single cluster.

Spectrum
--------
* ``"static"``: averaged over the packing's orientation by sampling
  (:mod:`~opennmr.notebook.bms.interactions`). :func:`bms_spectrum`'s
  own static route is δ_zz with B₀ along z for the one orientation the
  packing has -- not a powder average for a packing of tens of
  particles, so :func:`fast_bms_spectrum` does not use it.
* ``"mas_infinite"``: the isotropic shift only (infinitely fast MAS).
  With isotropic χ the isotropic BMS shift is zero everywhere except
  for the bulk term -- MAS removes it completely. Only an anisotropic χ
  (ASB, Alla & Lippmaa / VanderHart) leaves an isotropic broadening.
* ``"mas"``: δ_iso plus spinning sidebands of each voxel's (δ, η),
  powder-averaged over the packing's orientation in the rotor, from the
  same numerically simulated sideband manifold as ``bms.spectrum``.

Which voxels give signal (``observe``): ``"inside"`` -- nuclei in the
particles (weight s, the shape function); ``"outside"`` -- nuclei
between them (electrolyte, binder, a matrix; weight 1 − s); ``"all"``.

Sign: δ = −1e6 · sym(χN − L), the same sign as ``tensor2haeberlen``
gives the original code's d_iso (positive = deshielded, higher ppm).
Outside a sphere magnetised along z, δ_zz > 0 above the poles.
"""

from __future__ import annotations

import os
import time

import numpy as np
import scipy.fft as sfft

from .geometry import quat_to_rotmat
from .io_utils import inp_part
from .packing import _bounding_radius
from .shift import tensor2haeberlen

# The unique components of a symmetric 3×3 tensor, in this order.
_PAIRS = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))

SAMPLE_SHAPES = {
    "sphere": ("Sphere (no bulk shift)", np.eye(3) / 3.0),
    "cylinder_parallel": ("Long cylinder along B₀", np.diag([0.5, 0.5, 0.0])),
    "cylinder_perpendicular": ("Long cylinder perpendicular to B₀ (along x)",
                               np.diag([0.0, 0.5, 0.5])),
    "plate_perpendicular": ("Flat plate perpendicular to B₀", np.diag([0.0, 0.0, 1.0])),
}

SPECTRUM_MODES = {
    "static": "Static (B₀ along z)",
    "mas_infinite": "MAS, infinitely fast (isotropic shift only)",
    "mas": "MAS with spinning sidebands",
}

OBSERVE = {
    "inside": "Inside the particles",
    "outside": "Between the particles",
    "all": "Everywhere (every voxel equally)",
}


def _workers(workers):
    if workers is None or workers == -1:
        return os.cpu_count() or 1
    return max(1, int(workers))


def _sample_demag(sample_shape):
    if isinstance(sample_shape, str):
        if sample_shape not in SAMPLE_SHAPES:
            raise ValueError(f"Unknown sample shape {sample_shape!r}; choose one of "
                             f"{', '.join(SAMPLE_SHAPES)} or give a 3×3 demagnetising tensor.")
        return SAMPLE_SHAPES[sample_shape][1]
    D = np.asarray(sample_shape, dtype=float)
    if D.shape != (3, 3):
        raise ValueError("sample_shape must be a name or a 3×3 demagnetising tensor.")
    if abs(np.trace(D) - 1.0) > 1e-6:
        raise ValueError(f"A demagnetising tensor has trace 1 (this one: {np.trace(D):.4g}).")
    return D


def prepare_particles(particles, magnetic_susceptibility, equation="ellipsoid", deformation="A",
                      r=2, t=2, slope_filter=2000, threshold=0.99):
    """The particle records of the original code (``io_utils.inp_part``):
    each with its laboratory-frame ``Chi_tensor``, quaternion, sizes and
    shape parameters. ``particles`` is a list of :func:`make_particle`
    dicts, an (N, 10/11) array in the LAMMPS dump layout, or a dump file
    path."""
    if isinstance(particles, (str, os.PathLike)):
        props = {"filename": str(particles), "data": None}
    else:
        props = {"data": particles}
    props["Magnetic_Susceptibility"] = list(np.asarray(magnetic_susceptibility, float).ravel())
    if len(props["Magnetic_Susceptibility"]) != 3:
        raise ValueError("magnetic_susceptibility: give the three principal values (χ1, χ2, χ3).")
    shape = {"r": r, "t": t, "slope_filter": slope_filter, "threshold": threshold,
             "Equation": equation, "Deformation": deformation}
    return inp_part(props, shape)


def _box_lengths(box_size):
    if np.isscalar(box_size):
        return np.array([float(box_size)] * 3)
    L = np.asarray(box_size, dtype=float).ravel()
    if L.shape != (3,):
        raise ValueError("box_size: one length, or (Lx, Ly, Lz).")
    return L


def _steep(u, sl):
    """1/(1 + u**sl), evaluated only where it is neither 1 nor 0 to double
    precision (the thin shell at the particle surface)."""
    lo, hi = 10.0 ** (-17.0 / sl), 10.0 ** (17.0 / sl)
    out = (u < lo).astype(float)
    band = (u >= lo) & (u <= hi)
    if band.any():
        out[band] = 1.0 / (1.0 + u[band] ** sl)
    return out


def block_shape(p, xs, ys, zs):
    """The shape function of particle ``p`` (an ``inp_part`` record) on
    the block ``xs × ys × zs`` (1-d coordinates relative to the particle
    centre): equal to :func:`geometry.shape_function` on the same points
    (to ~1e-16), with the rotation done by broadcasting and the steep
    edge only evaluated in the surface shell."""
    from .geometry import quat_to_rotmat
    R = quat_to_rotmat(p["q"])
    x, y, z = xs[:, None, None], ys[None, :, None], zs[None, None, :]
    # geometry._to_local_frame: pts @ R
    xt = x * R[0, 0] + y * R[1, 0] + z * R[2, 0]
    yt = x * R[0, 1] + y * R[1, 1] + z * R[2, 1]
    zt = x * R[0, 2] + y * R[1, 2] + z * R[2, 2]
    eq, sl = p["Equation"], float(p["sl"])
    v1, v2, v3 = p["v1"], p["v2"], p["v3"]
    if eq == "hexagonal":
        Rhex, h = v1, v3
        deform = p["Deformation"]
        a, b, c = {"C": (xt, yt, zt), "B": (xt, zt, yt)}.get(deform, (yt, zt, xt))
        r3 = np.sqrt(3) / 2
        d_hex = np.maximum(np.maximum(np.abs(a), np.abs(0.5 * a + r3 * b)),
                           np.abs(0.5 * a - r3 * b)) / Rhex
        d_z = np.abs(c) / (h / 2)
        m = np.maximum(d_hex, d_z)
        lo, hi = 10.0 ** (-17.0 / sl), 10.0 ** (17.0 / sl)
        out = (m < lo).astype(float)
        band = (m >= lo) & (m <= hi)
        if band.any():
            with np.errstate(over="ignore"):
                out[band] = 1.0 / (1.0 + (d_hex[band] ** sl + d_z[band] ** sl))
        return out
    if eq == "cuboid":
        # geometry.cuboid_mask: 1/(1 + Σ |x_i / v_i|^sl), the steep edge
        # evaluated only in the surface shell (max_i |x_i / v_i| near 1).
        dx, dy, dz = np.abs(xt) / v1, np.abs(yt) / v2, np.abs(zt) / v3
        m = np.maximum(np.maximum(dx, dy), dz)
        lo, hi = 10.0 ** (-17.0 / sl), 10.0 ** (17.0 / sl)
        out = (m < lo).astype(float)
        band = (m >= lo) & (m <= hi)
        if band.any():
            with np.errstate(over="ignore"):
                out[band] = 1.0 / (1.0 + (dx[band] ** sl + dy[band] ** sl + dz[band] ** sl))
        return out
    if eq == "donut":
        R0, r0 = v1, v2
        deform = p["Deformation"]
        if deform == "C":
            rho, w = np.sqrt(xt ** 2 + yt ** 2), zt
        elif deform == "B":
            rho, w = np.sqrt(zt ** 2 + xt ** 2), yt
        else:
            rho, w = np.sqrt(zt ** 2 + yt ** 2), xt
        d = np.sqrt((rho - R0) ** 2 + w ** 2) / r0
        return _steep(d, sl)
    r, t = p["r"], p["t"]
    if r == 2 and t == 2:
        a, b, c = xt / v1, yt / v2, zt / v3
        u = a * a + b * b + c * c
    else:
        u = ((xt / v1) ** r + (yt / v2) ** r) ** (t / r) + (zt / v3) ** t
    return _steep(u, sl)


def _reach(p, margin):
    """How far, in bounding radii, the shape function is above ~1e-6: the
    edge is soft for a small slope filter (u**sl reaches 1e6 only at
    u = 10**(6/sl)), and a superellipsoid (r or t > 2) reaches its box
    corners, √3 times its largest semi-axis."""
    f = max(margin, 10.0 ** (6.0 / float(p["sl"])))
    if p["Equation"] not in ("hexagonal", "donut", "cuboid") and (p["r"] > 2 or p["t"] > 2):
        f *= np.sqrt(3.0)
    return f


def _rasterise(parts, L, n, n_fft, periodic, dtype, need_tensor, margin=1.15,
               threshold_owner=False):
    """X (one scalar field, or the six components of the symmetric χ
    field) on the FFT grid, the occupancy Σ s_p, and the owner index of
    each voxel: with ``lorentz="threshold"`` the particle whose core
    (s_p > threshold) holds it, otherwise any particle with s_p > 1e-9; −1
    outside every core). Each particle is evaluated on its own block."""
    step = L / n
    shape_fft = tuple(n_fft)
    occ = np.zeros(shape_fft, dtype=dtype)
    owner = np.full(shape_fft, -1, dtype=np.int32)
    X = [np.zeros(shape_fft, dtype=dtype) for _ in range(6 if need_tensor else 1)]
    vols = np.zeros(len(parts))                 # Σ s per particle (voxels), for ⟨χ⟩
    grid0 = -L / 2.0
    for p_index, p in enumerate(parts):
        R = float(_bounding_radius(p["Equation"], p["v1"], p["v2"], p["v3"])) * _reach(p, margin)
        pos = np.asarray(p["posc"], float)
        idx = []
        for a in range(3):
            lo = int(np.floor((pos[a] - R - grid0[a]) / step[a]))
            hi = int(np.ceil((pos[a] + R - grid0[a]) / step[a])) + 1
            if hi - lo > n[a]:                       # bigger than the box: every row once
                lo, hi = 0, n[a]
            k = np.arange(lo, hi)
            if periodic:
                coords = grid0[a] + step[a] * k - pos[a]   # unwrapped distance to the centre
                k = np.mod(k, n[a])
            else:
                keep = (k >= 0) & (k < n[a])
                k = k[keep]
                coords = grid0[a] + step[a] * k - pos[a]
            idx.append((k, coords))
        if any(len(k) == 0 for k, _ in idx):
            continue
        s = block_shape(p, idx[0][1], idx[1][1], idx[2][1]).astype(dtype, copy=False)
        sel = np.ix_(idx[0][0], idx[1][0], idx[2][0])
        occ[sel] += s
        vols[p_index] = float(s.sum(dtype=np.float64))
        core = s > (p["thresh"] if threshold_owner else 1e-9)
        if core.any():
            o = owner[sel]
            o[core] = p_index
            owner[sel] = o
        chi = p["Chi_tensor"]
        if need_tensor:
            for c, (i, j) in enumerate(_PAIRS):
                if chi[i, j] != 0.0:
                    X[c][sel] += dtype(chi[i, j]) * s
        else:
            X[0][sel] += dtype(chi[0, 0]) * s
    return X, occ, owner, vols


def _kvectors(L_fft, n_fft, dtype):
    """k-vectors of the rfftn grid (broadcastable), and 1/|k|² (0 at k = 0)."""
    ks = []
    for a in range(3):
        d = L_fft[a] / n_fft[a]
        f = sfft.rfftfreq(n_fft[a], d) if a == 2 else sfft.fftfreq(n_fft[a], d)
        shape = [1, 1, 1]
        shape[a] = len(f)
        ks.append(f.reshape(shape).astype(dtype))
    k2 = ks[0] ** 2 + ks[1] ** 2 + ks[2] ** 2
    with np.errstate(divide="ignore"):
        inv = np.where(k2 > 0, 1.0 / k2, 0.0).astype(dtype)
    return ks, inv


def bms_field(particles, magnetic_susceptibility, box_size=300.0, gridpoints=128,
              equation="ellipsoid", deformation="A", r=2, t=2, slope_filter=2000,
              threshold=0.99, boundary="periodic", sample_shape="sphere",
              components=("zz", "iso", "tensor"), dtype="float32", workers=-1,
              lorentz="proportional"):
    """The BMS shift field of a packing on the grid, by superposition.

    Parameters as :func:`opennmr.notebook.bms.main.run_bms` (sizes and
    box in the same length unit; χ the dimensionless SI volume
    susceptibility principal values), plus ``boundary``,
    ``sample_shape`` (see the module docstring), ``components`` -- any of
    ``"zz"`` (static δ_zz), ``"iso"`` (δ_iso) and ``"tensor"`` (the six
    unique components of the symmetric shift tensor, for MAS sidebands)
    -- ``dtype``, ``workers`` (FFT threads, −1 = all cores) and
    ``lorentz``: ``"proportional"`` (default) applies the Lorentz-sphere
    term −χ/3 in proportion to each voxel's occupancy s, the volume
    average over a voxel that is partly inside a particle -- with an
    isotropic χ this makes δ_iso exactly zero everywhere, as it must be;
    ``"threshold"`` applies it in full where s > ``threshold`` and not at
    all elsewhere, as the original code does (surface voxels with
    0 < s < threshold then lose it, up to 1e6·tr χ/9 ppm off).

    Returns a dict: ``x``, ``y``, ``z`` (grid axes), ``occupancy`` (Σ s_p,
    the fraction of each voxel inside a particle), ``owner`` (particle
    index of each voxel a particle reaches, −1 elsewhere), the requested ``delta_zz``,
    ``delta_iso`` (ppm) and ``delta_tensor`` (dict of (i, j) -> field,
    ppm), the ``volume_fraction``, ``n_particles``, ``timings`` (s) and
    ``parameters``.
    """
    t_start = time.perf_counter()
    dtype = np.dtype(dtype).type
    if dtype not in (np.float32, np.float64):
        raise ValueError("dtype: 'float32' or 'float64'.")
    if boundary not in ("periodic", "isolated"):
        raise ValueError("boundary: 'periodic' or 'isolated'.")
    components = tuple(components)
    bad = set(components) - {"zz", "iso", "tensor"}
    if bad:
        raise ValueError(f"Unknown components {sorted(bad)}: use 'zz', 'iso', 'tensor'.")
    n = np.array([int(gridpoints)] * 3) if np.isscalar(gridpoints) else \
        np.asarray(gridpoints, int).ravel()
    if np.any(n < 8):
        raise ValueError("gridpoints: at least 8 per axis.")
    L = _box_lengths(box_size)
    periodic = boundary == "periodic"
    D = _sample_demag(sample_shape) if periodic else np.eye(3) / 3.0
    parts = prepare_particles(particles, magnetic_susceptibility, equation, deformation,
                              r, t, slope_filter, threshold)
    chis = np.array([p["Chi_tensor"] for p in parts])
    iso_chi = bool(np.allclose(chis - np.trace(chis, axis1=1, axis2=2)[:, None, None] / 3
                               * np.eye(3), 0.0, atol=1e-15 + 1e-12 * np.abs(chis).max()))
    n_fft = n if periodic else 2 * n
    L_fft = L if periodic else 2 * L
    w = _workers(workers)

    timings = {}
    t0 = time.perf_counter()
    if lorentz not in ("proportional", "threshold"):
        raise ValueError("lorentz: 'proportional' or 'threshold'.")
    X, occ, owner, vols = _rasterise(parts, L, n, n_fft, periodic, dtype, need_tensor=not iso_chi,
                               threshold_owner=lorentz == "threshold")
    timings["shapes"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    ks, inv_k2 = _kvectors(L_fft, n_fft, dtype)
    volume = float(np.prod(n_fft))
    # G_i(k) = Σ_j FFT(X_ij)(k) k_j / |k|²  and the k = 0 (mean) term.
    if iso_chi:
        F = sfft.rfftn(X[0], workers=w)
        mean_X = np.eye(3) * (F[0, 0, 0].real / volume)
        G = [F * ks[i] * inv_k2 for i in range(3)]
        del F
    else:
        G = [None, None, None]
        mean_X = np.zeros((3, 3))
        for c, (i, j) in enumerate(_PAIRS):
            if not X[c].any():
                continue
            F = sfft.rfftn(X[c], workers=w)
            mean_X[i, j] = mean_X[j, i] = F[0, 0, 0].real / volume
            for a, b in ((i, j), (j, i)) if i != j else ((i, j),):
                term = F * ks[b]
                G[a] = term if G[a] is None else G[a] + term
            del F
        G = [(g * inv_k2 if g is not None else None) for g in G]
    del X
    timings["forward_fft"] = time.perf_counter() - t0

    # Lorentz sphere: χ_owner·s/3 (proportional) or χ_owner/3 on core voxels
    # (threshold, as the original code). A lookup table per component, its
    # last entry (index −1) zero.
    def lorentz_term(i, j):
        table = np.append(chis[:, i, j] / 3.0, 0.0).astype(dtype)
        out = table[owner]
        if lorentz == "proportional":
            out *= np.minimum(occ, 1.0)
        return out

    bulk = mean_X @ D       # the k = 0 term of χN: ⟨X⟩·D

    def chiN(i, k):
        """sym(χN)_ik on the grid (without the Lorentz term)."""
        parts_ = []
        if G[i] is not None:
            parts_.append(G[i] * ks[k])
        if G[k] is not None:
            parts_.append(G[k] * ks[i])
        if not parts_:
            field = np.zeros(tuple(n_fft), dtype=dtype)
        else:
            spec = parts_[0] if len(parts_) == 1 else parts_[0] + parts_[1]
            field = sfft.irfftn(0.5 * spec, s=tuple(n_fft), workers=w).astype(dtype, copy=False)
        field += dtype(0.5 * (bulk[i, k] + bulk[k, i]))
        return field

    def crop(a):
        return a if periodic else a[:n[0], :n[1], :n[2]]

    t0 = time.perf_counter()
    out = {}
    if "tensor" in components:
        tensor = {}
        for (i, j) in _PAIRS:
            tensor[(i, j)] = -1e6 * (crop(chiN(i, j)) - crop(lorentz_term(i, j)))
        out["delta_tensor"] = tensor
        if "zz" in components:
            out["delta_zz"] = tensor[(2, 2)]
        if "iso" in components:
            out["delta_iso"] = (tensor[(0, 0)] + tensor[(1, 1)] + tensor[(2, 2)]) / 3.0
    else:
        if "zz" in components:
            out["delta_zz"] = -1e6 * (crop(chiN(2, 2)) - crop(lorentz_term(2, 2)))
        if "iso" in components:
            # trace(χN) = IFFT(Σ_i G_i k_i) + tr(bulk): one inverse transform.
            spec = None
            for i in range(3):
                if G[i] is not None:
                    term = G[i] * ks[i]
                    spec = term if spec is None else spec + term
            tr = (sfft.irfftn(spec, s=tuple(n_fft), workers=w).astype(dtype, copy=False)
                  if spec is not None else np.zeros(tuple(n_fft), dtype=dtype))
            tr += dtype(np.trace(bulk))
            trL = lorentz_term(0, 0) + lorentz_term(1, 1) + lorentz_term(2, 2)
            out["delta_iso"] = -1e6 * (crop(tr) - crop(trL)) / 3.0
    timings["inverse_fft"] = time.perf_counter() - t0

    occ = crop(occ)
    owner = crop(owner)
    axes = [-L[a] / 2 + (L[a] / n[a]) * np.arange(n[a]) for a in range(3)]
    timings["total"] = time.perf_counter() - t_start
    out.update({
        "x": axes[0], "y": axes[1], "z": axes[2],
        "occupancy": np.clip(occ, 0.0, 1.0),
        "owner": owner,
        "volume_fraction": float(np.clip(occ, 0.0, 1.0).mean()),
        "n_particles": len(parts),
        # particle frame -> packing frame is Rᵀ (χ_lab = Rᵀ diag(χ) R, io_utils.inp_part)
        "rotations": np.array([quat_to_rotmat(p["q"]) for p in parts]),
        # the packing's mean χ tensor (volume-weighted over the box): χ_v = tr/3
        "mean_chi": np.einsum("p,pij->ij", vols, chis) / float(np.prod(n)),
        "isotropic_chi": iso_chi,
        "timings": timings,
        "parameters": {
            "magnetic_susceptibility": [float(v) for v in np.ravel(magnetic_susceptibility)],
            "box_size": [float(v) for v in L], "gridpoints": [int(v) for v in n],
            "equation": equation, "deformation": deformation, "r": r, "t": t,
            "slope_filter": slope_filter, "threshold": threshold, "boundary": boundary,
            "sample_shape": (sample_shape if isinstance(sample_shape, str)
                             else np.asarray(sample_shape, float).tolist()),
            "dtype": np.dtype(dtype).name, "lorentz": lorentz,
        },
    })
    return out


def observed_weights(field, observe="inside"):
    """Signal weight of each voxel (see ``OBSERVE``)."""
    occ = field["occupancy"]
    if observe == "inside":
        return occ
    if observe == "outside":
        return 1.0 - occ
    if observe == "all":
        return np.ones_like(occ)
    raise ValueError(f"observe: one of {', '.join(OBSERVE)}.")


MAGIC = np.arccos(1.0 / np.sqrt(3.0))


def _phase_basis(n_beta, n_gamma, n_t):
    """Per powder orientation, the phase (cycles, per unit μ) of the
    anisotropic tensors (−½, −½, 1) and (−½, ½, 0): the phase of (δ, η)
    is μ·(A + ηB). Under MAS the lab zz element is a Fourier series in
    the rotor angle φ (harmonics 1 and 2), so it integrates analytically."""
    cb, wb = np.polynomial.legendre.leggauss(n_beta)
    beta = np.arccos(cb)
    gamma = np.linspace(0, 2 * np.pi, n_gamma, endpoint=False)
    B, G = np.meshgrid(beta, gamma, indexing="ij")
    B, G = B.ravel(), G.ravel()
    w = np.repeat(wb, n_gamma)
    w = w / w.sum()
    # rotor-frame tensor σ = R diag(pas) Rᵀ, R = Ry(β) Rz(γ) (as mas_sidebands)
    cbt, sbt, cg, sg = np.cos(B), np.sin(B), np.cos(G), np.sin(G)
    R = np.empty((len(B), 3, 3))
    R[:, 0] = np.stack([cbt * cg, -cbt * sg, sbt], -1)
    R[:, 1] = np.stack([sg, cg, np.zeros_like(cg)], -1)
    R[:, 2] = np.stack([-sbt * cg, sbt * sg, cbt], -1)
    phi = 2 * np.pi * np.arange(n_t) / n_t
    s, c = np.sin(MAGIC), np.cos(MAGIC)
    # ∫₀^φ v σ vᵀ dφ' for v = (−s cos φ', s sin φ', c):
    I_xx = phi / 2 + np.sin(2 * phi) / 4
    I_yy = phi / 2 - np.sin(2 * phi) / 4
    I_xy = np.sin(phi) ** 2 / 2
    I_x = np.sin(phi)
    I_y = 1 - np.cos(phi)

    def phase(pas):
        sig = np.einsum("oij,j,okj->oik", R, np.asarray(pas, float), R)
        return (s * s * (sig[:, 0, 0, None] * I_xx + sig[:, 1, 1, None] * I_yy
                         - 2 * sig[:, 0, 1, None] * I_xy)
                + c * c * sig[:, 2, 2, None] * phi
                + 2 * s * c * (-sig[:, 0, 2, None] * I_x + sig[:, 1, 2, None] * I_y)) / (2 * np.pi)
    return w, phase((-0.5, -0.5, 1.0)), phase((-0.5, 0.5, 0.0))


def mas_sideband_intensities(mu, eta, n_beta=20, n_gamma=20, n_t=256):
    """Powder-averaged MAS sideband intensities of a traceless shift
    tensor with anisotropy δ (Haeberlen, as ``tensor2haeberlen``) and
    asymmetry η, for ``mu`` = δ / ν_r (signed: the pattern of −δ is the
    mirror image of +δ). Returns ``(orders, intensities)``, intensities of
    shape ``broadcast(mu, eta).shape + (n_t,)``, each summing to 1.

    The same physics as :func:`mas_sidebands.sideband_manifold` (rotor
    frame β, γ powder average of |c_n|², Gauss–Legendre in cos β), with
    the phase integrated analytically and every orientation at once.
    ``n_t`` points per rotor period hold orders up to ±n_t/2."""
    mu, eta = np.broadcast_arrays(np.asarray(mu, float), np.asarray(eta, float))
    w, A, B = _phase_basis(n_beta, n_gamma, n_t)
    out = np.empty(mu.shape + (n_t,))
    flat_out = out.reshape(-1, n_t)
    for k, (m, e) in enumerate(zip(mu.ravel(), eta.ravel())):
        if m == 0.0:
            flat_out[k] = 0.0
            flat_out[k, 0] = 1.0
            continue
        amp = np.fft.fft(np.exp(2j * np.pi * m * (A + e * B)), axis=1)
        flat_out[k] = w @ (amp.real ** 2 + amp.imag ** 2)
    out /= out.sum(axis=-1, keepdims=True)
    orders = np.fft.fftfreq(n_t, d=1.0 / n_t).astype(int)
    return orders, out


MU_GRID_POS = np.geomspace(0.02, 128.0, 44)
MU_GRID = np.concatenate([-MU_GRID_POS[::-1], [0.0], MU_GRID_POS])
ETA_GRID = np.linspace(0.0, 1.0, 11)
N_T = 512                                     # orders up to ±256
_ROWS: dict = {}


def _orientations(mu):
    """Powder points per angle for a converged pattern: the pattern gets
    finer with |μ| (checked against 64×64: < 1e-4 absolute)."""
    return int(min(192, max(20, np.ceil(2.2 * abs(mu)) + 16)))


_CACHE_TAG = "bms_sidebands_v1"   # change when the grids or the method change


def sideband_cache_dir():
    """This computer's folder for the computed sideband rows (next to
    openNMR's spectrum cache). ``OPENNMR_BMS_CACHE`` overrides it; set to
    an empty string, nothing is written."""
    env = os.environ.get("OPENNMR_BMS_CACHE")
    if env is not None:
        return env or None
    from ...io.load_cache import cache_dir
    return os.path.join(os.path.dirname(cache_dir()), _CACHE_TAG)


def _row(i):
    """Sideband intensities of MU_GRID[i] at every ETA_GRID value: in
    memory, else from this computer's cache, else computed (and saved)."""
    if i in _ROWS:
        return _ROWS[i]
    folder = sideband_cache_dir()
    path = os.path.join(folder, f"row_{i:03d}.npy") if folder else None
    if path and os.path.isfile(path):
        try:
            row = np.load(path)
            if row.shape == (len(ETA_GRID), N_T):
                _ROWS[i] = row
                return row
        except (OSError, ValueError):
            pass
    if i not in _ROWS:
        m = float(MU_GRID[i])
        n = _orientations(m)
        nt = int(min(N_T, max(64, 2 ** int(np.ceil(np.log2(4 * abs(m) + 48))))))
        o, I = mas_sideband_intensities(np.full(len(ETA_GRID), m), ETA_GRID,
                                        n_beta=n, n_gamma=n, n_t=nt)
        full = np.zeros((len(ETA_GRID), N_T))
        full[:, np.mod(o, N_T)] = I                 # orders beyond ±nt/2 are ~0 here
        _ROWS[i] = full
        if path:
            try:
                os.makedirs(folder, exist_ok=True)
                tmp = f"{path}.{os.getpid()}.tmp"
                with open(tmp, "wb") as fh:
                    np.save(fh, full)
                os.replace(tmp, path)
            except OSError:
                pass                                # a cache, never an error
    return _ROWS[i]


def _manifold(max_order, mu_needed=None):
    """The sideband table on the signed μ grid (rows built on demand, only
    as far as ``mu_needed`` = max |μ| of the data) and the η grid, orders
    ±max_order -- whatever lies beyond is reported by :func:`bms_spectrum`."""
    mu_needed = MU_GRID_POS[-1] if mu_needed is None else min(abs(mu_needed), MU_GRID_POS[-1])
    k = int(np.searchsorted(MU_GRID_POS, mu_needed))      # first grid point ≥ μ needed
    hi = min(len(MU_GRID_POS) - 1, k)
    c = len(MU_GRID_POS)                                   # MU_GRID[c] = 0, MU_GRID[c ± (k+1)] = ±POS[k]
    idx = np.arange(c - 1 - hi, c + 2 + hi)
    table = np.stack([_row(i) for i in idx])
    orders = np.fft.fftfreq(N_T, d=1.0 / N_T).astype(int)
    keep = np.abs(orders) <= max_order
    sel = np.argsort(orders[keep])
    return MU_GRID[idx], ETA_GRID, orders[keep][sel], table[..., keep][..., sel]


def _lookup(mu, eta, mu_grid, eta_grid, table):
    """Bilinear interpolation of the table at signed μ and |η|."""
    m = np.clip(mu, mu_grid[0], mu_grid[-1])
    e = np.clip(np.abs(eta), eta_grid[0], eta_grid[-1])
    i = np.clip(np.searchsorted(mu_grid, m) - 1, 0, len(mu_grid) - 2)
    j = np.clip(np.searchsorted(eta_grid, e) - 1, 0, len(eta_grid) - 2)
    fi = ((m - mu_grid[i]) / (mu_grid[i + 1] - mu_grid[i]))[:, None]
    fj = ((e - eta_grid[j]) / (eta_grid[j + 1] - eta_grid[j]))[:, None]
    return (table[i, j] * (1 - fi) * (1 - fj) + table[i + 1, j] * fi * (1 - fj)
            + table[i, j + 1] * (1 - fi) * fj + table[i + 1, j + 1] * fi * fj)


def _broaden(hist, sw_hz, linewidth_hz, lineshape):
    """Histogram -> spectrum convolved with a Lorentzian or Gaussian of
    FWHM ``linewidth_hz`` (by multiplying its FID)."""
    npts = len(hist)
    fid = np.fft.ifft(np.fft.ifftshift(hist.astype(complex)))
    tt = np.fft.fftfreq(npts, d=sw_hz / npts)      # signed times: a symmetric (real) lineshape
    if linewidth_hz > 0:
        if lineshape == "gaussian":
            fid *= np.exp(-(np.pi * linewidth_hz * tt) ** 2 / (4 * np.log(2)))
        else:
            fid *= np.exp(-np.pi * linewidth_hz * np.abs(tt))
    return np.fft.fftshift(np.fft.fft(fid)).real


def bms_spectrum(field, nuclide="7Li", B0_tesla=9.4, mode="static", observe="inside",
                 spin_rate_hz=10000.0, linewidth_hz=50.0, lineshape="lorentzian",
                 centre_ppm=None, sw_ppm=None, n_points=None, max_order=8,
                 offset_ppm=0.0, gamma_MHz_per_T=None, min_weight=1e-3):
    """The spectrum of a :func:`bms_field` result, as openNMR's standard
    spectrum dict (``x_ppm``, ``real_spectrum``, ``BF1``, ...), plus
    ``"stats"``: the weighted mean and standard deviation of the shift
    distribution (ppm) and the fraction of signal outside the window.

    ``centre_ppm``/``sw_ppm``: the window; by default centred on the
    weighted mean, wide enough for the whole distribution (and the
    sidebands) plus ten linewidths. ``min_weight``: voxels with less
    weight are left out (their sum is reported)."""
    from ..utils import compute_BF1_SFO1
    if mode not in SPECTRUM_MODES:
        raise ValueError(f"mode: one of {', '.join(SPECTRUM_MODES)}.")
    BF1, SFO1, nuclide = compute_BF1_SFO1(nuclide, B0_tesla, offset_ppm=offset_ppm,
                                          gamma_MHz_per_T=gamma_MHz_per_T)
    w_all = observed_weights(field, observe).ravel().astype(float)
    keep = w_all > min_weight
    w = w_all[keep]
    if w.sum() <= 0:
        raise ValueError("No voxel carries signal: is the packing empty, or is every voxel "
                         "inside a particle?")
    key = "delta_zz" if mode == "static" else "delta_iso"
    if key not in field:
        raise ValueError(f"This field has no {key}: calculate it with components including "
                         f"{'zz' if mode == 'static' else 'iso'}.")
    pos = field[key].ravel()[keep].astype(float)
    sticks_ppm, sticks_w = pos, w
    spin_ppm = spin_rate_hz / BF1
    if mode == "mas":
        if "delta_tensor" not in field:
            raise ValueError("MAS sidebands need the full tensor: components including 'tensor'.")
        T = np.zeros((len(pos), 3, 3))
        for (i, j), f in field["delta_tensor"].items():
            T[:, i, j] = T[:, j, i] = f.ravel()[keep]
        # tensor2haeberlen expects χN-like (it negates); δ = −that, so pass −δ.
        haeb = tensor2haeberlen(-T)
        if spin_rate_hz <= 0:
            raise ValueError("Spin rate must be positive for MAS with sidebands.")
        mu = haeb[:, 1] / spin_ppm
        mu_g, eta_g, orders, table = _manifold(int(max_order), float(np.max(np.abs(mu))))
        intens = _lookup(mu, haeb[:, 2], mu_g, eta_g, table)
        beyond = float(np.sum(w[np.abs(mu) > mu_g[-1]]) / w.sum())
        kept = intens.sum(axis=1)                       # < 1 where orders were cut off
        cut = float(1.0 - np.sum(kept * w) / w.sum())
        sticks_ppm = (haeb[:, 0][:, None] + orders[None, :] * spin_ppm).ravel()
        sticks_w = (intens * w[:, None]).ravel()
        pos = haeb[:, 0]
    else:
        beyond = cut = 0.0
    mean = float(np.average(pos, weights=w))
    std = float(np.sqrt(np.average((pos - mean) ** 2, weights=w)))
    lw_ppm = linewidth_hz / BF1
    if centre_ppm is None:
        centre_ppm = mean
    if sw_ppm is None:
        lo, hi = np.percentile(sticks_ppm, [0.05, 99.95]) if len(sticks_ppm) else (mean, mean)
        if mode == "mas":
            lo = min(lo, mean - (max_order + 0.5) * spin_ppm)
            hi = max(hi, mean + (max_order + 0.5) * spin_ppm)
        half = max(abs(hi - centre_ppm), abs(centre_ppm - lo)) + 10 * lw_ppm
        sw_ppm = 2 * max(half, 20 * lw_ppm, 1e-3)
    from ..simulations import _bin_sticks
    n_points, under = resolve_points(sw_ppm, lw_ppm, n_points)
    sw_hz = sw_ppm * BF1
    edges_lo = centre_ppm - sw_ppm / 2
    x_ppm = edges_lo + sw_ppm * (np.arange(n_points) + 0.5) / n_points
    hist = _bin_sticks(x_ppm, sticks_ppm, sticks_w)          # linear, as the sampled route
    total = float(sticks_w.sum())
    outside_window = 1.0 - (float(hist.sum()) / total if total else 1.0)
    spec = _broaden(hist, sw_hz, linewidth_hz, lineshape)
    peak = np.max(np.abs(spec)) or 1.0
    return {
        "x_ppm": x_ppm, "x_Hz": x_ppm * BF1,
        "real_spectrum": spec, "normalised_real_spectrum": spec / peak,
        "imag_spectrum": np.zeros_like(spec), "normalised_imag_spectrum": np.zeros_like(spec),
        "nuclide": nuclide, "BF1": float(BF1), "SFO1": float(SFO1), "path": None,
        "stats": {"mean_ppm": mean, "std_ppm": std, "fwhm_gauss_ppm": 2.3548 * std,
                  "outside_window": outside_window, "weight": float(w.sum()),
                  "left_out_weight": float(w_all[~keep].sum()),
                  "sidebands_beyond_max_order": cut, "beyond_mu_grid": beyond,
                  "n_points": n_points, "under_resolved": under},
        "parameters": {"nuclide": nuclide, "B0_tesla": float(B0_tesla), "mode": mode,
                       "observe": observe, "spin_rate_hz": float(spin_rate_hz),
                       "linewidth_hz": float(linewidth_hz), "lineshape": lineshape},
    }


#: Points per FWHM the automatic axis gives the narrowest line, and the
#: largest automatic axis.
POINTS_PER_FWHM = 5
MAX_POINTS = 2 ** 20


def resolve_points(sw_ppm, linewidth_ppm, n_points=None):
    """``(n, under_resolved)``: the points of a spectrum window of
    ``sw_ppm`` -- automatic (``n_points=None``: a power of two with at least
    :data:`POINTS_PER_FWHM` points across the line width, at least 4096,
    at most :data:`MAX_POINTS`), or as given -- and whether the line is
    then sampled by fewer than 3 points per FWHM (its height and shape are
    then set by the grid, not by the physics: found 2026-11-07, a 2048-point
    axis put 1.6 points across a 1.9 ppm line and misplaced peak heights by
    5 %)."""
    lw = float(linewidth_ppm)
    if n_points is None:
        need = 4096 if lw <= 0 else POINTS_PER_FWHM * sw_ppm / lw
        n = int(min(MAX_POINTS, 2 ** int(np.ceil(np.log2(max(need, 4096))))))
    else:
        n = int(n_points)
    under = lw > 0 and sw_ppm / n > lw / 3.0
    return n, bool(under)


def default_samples(mode) -> int:
    """Samples for a sampled spectrum: static ones are cheap (one B₀
    direction each), MAS ones carry a rotor cycle and its sidebands."""
    return 1_000_000 if mode == "static" else 200_000


def needs_sampling(sample=None, local=None, mode="mas") -> bool:
    """Whether a spectrum goes through :func:`interactions.sample_spectrum`:
    every static spectrum (averaged over the packing's orientation), and
    under MAS a rotor or plate around the packing, or a local anisotropy
    or quadrupolar coupling. The exact grid routes cover MAS of a
    spherical sample with at most an isotropic local shift.

    Why static is always sampled: one orientation of a finite packing
    (B₀ along z) is not a powder average -- a 30-particle packing's δ_zz
    had a mean of +8.1 ppm where the orientation average (= the mean
    isotropic shift) is −3.8 ppm (2026-11-07)."""
    from .interactions import has_local
    from .sample import sample_from_dict
    if mode == "static":
        return True
    return (sample is not None and not sample_from_dict(sample).is_uniform()) or has_local(local)


def route(mode, sample=None, local=None):
    """``(sampled, components)``: whether the spectrum is sampled
    (:func:`needs_sampling`) and the field components it needs (plus δ_zz
    and δ_iso for the field slice when sampled)."""
    if mode not in SPECTRUM_MODES:
        raise ValueError(f"mode: one of {', '.join(SPECTRUM_MODES)}.")
    sampled = needs_sampling(sample, local, mode)
    if sampled:
        return True, ("tensor", "zz", "iso")
    return False, {"mas_infinite": ("iso",), "mas": ("tensor", "iso")}[mode]


def spectrum_from_field(field, *, nuclide="7Li", B0_tesla=9.4, mode="static", observe="inside",
                        spin_rate_hz=10000.0, linewidth_hz=50.0, lineshape="lorentzian",
                        centre_ppm=None, sw_ppm=None, n_points=None, max_order=8, sample=None,
                        local=None, transitions="ct", quad_order="second", n_samples=None,
                        seed=0, progress=None) -> dict:
    """The spectrum of a field from :func:`bms_field` (with the components
    :func:`route` asks for), by the route :func:`route` picks: sampled
    (:func:`interactions.sample_spectrum`) or the exact grid route
    (:func:`bms_spectrum`). A field can be reused for any number of
    spectra (nucleus, field, MAS rate, sample, local interactions …)."""
    sampled, comps = route(mode, sample, local)
    keys = {"tensor": "delta_tensor", "iso": "delta_iso", "zz": "delta_zz"}
    missing = [c for c in (("tensor",) if sampled else comps) if keys[c] not in field]
    if missing:
        raise ValueError(f"The field lacks {', '.join(missing)}: calculate it with "
                         f"components={comps!r}.")
    if sampled:
        from .interactions import sample_spectrum
        spec = sample_spectrum(field, nuclide=nuclide, B0_tesla=B0_tesla,
                               mode="static" if mode == "static" else "mas",
                               spin_rate_hz=1e12 if mode == "mas_infinite" else spin_rate_hz,
                               sample=sample, local=local, observe=observe,
                               transitions=transitions, quad_order=quad_order,
                               n_samples=default_samples(mode) if not n_samples else n_samples,
                               seed=seed, linewidth_hz=linewidth_hz, lineshape=lineshape,
                               centre_ppm=centre_ppm, sw_ppm=sw_ppm, n_points=n_points,
                               progress=progress)
        spec["parameters"]["mode"] = mode
        spec["route"] = "sampled"
        return spec
    contact = float((local or {}).get("contact_ppm", 0.0))
    spec = bms_spectrum(field, nuclide=nuclide, B0_tesla=B0_tesla, mode=mode, observe=observe,
                        spin_rate_hz=spin_rate_hz, linewidth_hz=linewidth_hz,
                        lineshape=lineshape,
                        centre_ppm=None if centre_ppm is None else centre_ppm - contact,
                        sw_ppm=sw_ppm, n_points=n_points, max_order=max_order)
    if contact:
        spec["x_ppm"] = spec["x_ppm"] + contact
        spec["x_Hz"] = spec["x_ppm"] * spec["BF1"]
        spec["stats"]["mean_ppm"] += contact
    spec["route"] = "grid"
    if local:
        spec["parameters"]["local"] = dict(local)
    return spec


def fast_bms_spectrum(particles, magnetic_susceptibility, nuclide="7Li", B0_tesla=9.4,
                      mode="static", observe="inside", spin_rate_hz=10000.0, box_size=300.0,
                      gridpoints=128, equation="ellipsoid", deformation="A", r=2, t=2,
                      slope_filter=2000, threshold=0.99, boundary="periodic",
                      sample_shape="sphere", linewidth_hz=50.0, lineshape="lorentzian",
                      centre_ppm=None, sw_ppm=None, n_points=None, max_order=8,
                      dtype="float32", workers=-1, sample=None, local=None,
                      transitions="ct", quad_order="second", n_samples=None, seed=0,
                      progress=None):
    """Packing -> BMS spectrum in one call: :func:`bms_field` (only the
    components the route needs) then :func:`spectrum_from_field`. Returns
    ``(spectrum_dict, field)``.

    ``sample`` (a :class:`~opennmr.notebook.bms.sample.Sample` or its dict:
    a rotor, tube or plate around the packing) and ``local``
    (:func:`~opennmr.notebook.bms.interactions.local_interactions`: contact
    shift, spin-dipolar anisotropy, quadrupolar coupling) go through
    :func:`~opennmr.notebook.bms.interactions.sample_spectrum`
    (``transitions``, ``quad_order``, ``n_samples``, ``seed``), as does
    every static spectrum (see :func:`needs_sampling`). Otherwise -- MAS of
    a spherical sample, at most an isotropic local shift -- the exact grid
    routes. With ``sample`` the field's own k = 0 term is a sphere's, the
    sample's shape coming from ``sample`` instead."""
    _sampled, comps = route(mode, sample, local)
    if _sampled and sample_shape != "sphere":
        raise ValueError("sample_shape is the grid routes' uniform bulk term; for this spectrum "
                         "give the sample as sample= (opennmr.notebook.bms.sample.Sample), e.g. "
                         "Sample('cylinder', diameter=…, length=…, axis_angle=0).")
    if sample is not None:
        sample_shape = "sphere"
    field = bms_field(particles, magnetic_susceptibility, box_size=box_size,
                      gridpoints=gridpoints, equation=equation, deformation=deformation, r=r,
                      t=t, slope_filter=slope_filter, threshold=threshold, boundary=boundary,
                      sample_shape=sample_shape, components=comps, dtype=dtype, workers=workers)
    spec = spectrum_from_field(field, nuclide=nuclide, B0_tesla=B0_tesla, mode=mode,
                               observe=observe, spin_rate_hz=spin_rate_hz,
                               linewidth_hz=linewidth_hz, lineshape=lineshape,
                               centre_ppm=centre_ppm, sw_ppm=sw_ppm, n_points=n_points,
                               max_order=max_order, sample=sample, local=local,
                               transitions=transitions, quad_order=quad_order,
                               n_samples=n_samples, seed=seed, progress=progress)
    return spec, field


def plot_bms_slice(field, figure=None, quantity="delta_zz", axis="y", index=None,
                   fontsize=11.0, cmap="RdBu_r", limit=None):
    """A slice of the shift field through the box (default: the xz plane
    through the centre, B₀ vertical), with the particle outlines."""
    import matplotlib.pyplot as plt
    if quantity not in field:
        raise ValueError(f"This field has no {quantity}.")
    fig = figure if figure is not None else plt.figure(figsize=(5.5, 4.5))
    fig.clear()
    ax = fig.add_subplot(111)
    a = "xyz".index(axis)
    data, occ = field[quantity], field["occupancy"]
    n = data.shape[a]
    i = n // 2 if index is None else int(index)
    sl = [slice(None)] * 3
    sl[a] = i
    img, o = data[tuple(sl)], occ[tuple(sl)]
    others = [c for c in "xyz" if c != axis]
    h, v = field[others[0]], field[others[1]]
    lim = limit if limit is not None else float(np.percentile(np.abs(img), 99)) or 1.0
    m = ax.pcolormesh(h, v, img.T, cmap=cmap, vmin=-lim, vmax=lim, shading="nearest",
                      rasterized=True)
    ax.contour(h, v, o.T, levels=[0.5], colors="k", linewidths=0.6)
    ax.set_aspect("equal")
    ax.set_xlabel(f"${others[0]}$", fontsize=fontsize)
    ax.set_ylabel(f"${others[1]}$", fontsize=fontsize)
    label = {"delta_zz": r"$\delta_{zz}$ (ppm)", "delta_iso": r"$\delta_\mathrm{iso}$ (ppm)"}
    cb = fig.colorbar(m, ax=ax)
    cb.set_label(label.get(quantity, quantity), fontsize=fontsize)
    ax.set_title(f"${axis}$ = {field[axis][i]:.4g}", fontsize=fontsize)
    ax.tick_params(labelsize=fontsize * 0.9)
    fig.tight_layout()
    return fig
