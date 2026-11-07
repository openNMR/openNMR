"""
BMS spectra with the macroscopic sample and local interactions (explicit
user request, 2026-11-07: a rotor or a plate; "add on the local
contact / spin dipolar / quadrupolar interactions").

What each observed nucleus sees, as a frequency offset for a B₀ direction:

* the BMS shift tensor of its voxel, from the packing
  (:func:`opennmr.notebook.bms.fast.bms_field`, packing frame);
* the macroscopic shift of the sample at the cell's position,
  δ_M = −1e6 χ_v (N(r) − I/3) (:mod:`opennmr.notebook.bms.sample`, sample
  frame);
* a local shift tensor: an isotropic shift (Fermi contact, plus any
  orbital shift) and a traceless anisotropy (the spin-dipolar shift;
  Haeberlen δ, η), fixed in its particle's frame (Euler angles from the
  particle's χ principal frame);
* a quadrupolar coupling (C_Q, η_Q), likewise fixed in the particle,
  at first or second order or exact, for the central transition or every
  transition -- openNMR's own quadrupolar functions
  (:mod:`opennmr.notebook.simulations`).

These are correlated through orientations (the local tensors turn with
their particle; the macroscopic field is fixed in the rotor while the
packing is a powder in it), so the spectrum is not a product of separate
averages. It is sampled: each sample is a voxel (drawn in proportion to
its signal), an orientation of the packing in the sample (quasi-uniform
directions, a random turn about each) and a cell of the sample (cells of
equal volume, a random azimuth). Under MAS each sample's frequencies are
taken at 16 rotor phases and turned into a centreband and sidebands by
γ-COMPUTE (the machinery of the quadrupolar and combined simulators), so
the third Euler angle is averaged exactly. Static: one B₀ direction per
sample.

The Monte Carlo noise is reported: the samples are split into two halves,
each binned on its own, and the rms difference of the halves (relative to
the peak) is √2 times the noise of either -- so the noise of the full set
is about half of it. Raise the sample count until it is small next to the
features you look at.
"""

from __future__ import annotations

import time

import numpy as np

from .sample import MAGIC_ANGLE_DEG, sample_from_dict

GOLDEN = np.pi * (3.0 - np.sqrt(5.0))
N_GAMMA = 16                              # rotor phases per period (rank 2 and 4: exact)


def local_interactions(contact_ppm=0.0, aniso_ppm=0.0, eta=0.0, euler=(0.0, 0.0, 0.0),
                       C_Q_MHz=0.0, eta_Q=0.0, euler_Q=(0.0, 0.0, 0.0)) -> dict:
    """The local interactions of an observed nucleus, in its particle's
    frame (the principal frame of the particle's χ): ``contact_ppm`` the
    isotropic shift (Fermi contact and any orbital shift), ``aniso_ppm``
    and ``eta`` the traceless anisotropy (spin dipolar; Haeberlen δ, η),
    ``euler`` its principal frame (ZYZ, degrees, active, as the combined
    simulator); ``C_Q_MHz``, ``eta_Q``, ``euler_Q`` the quadrupolar
    coupling's (EFG frame)."""
    if not 0.0 <= eta <= 1.0 or not 0.0 <= eta_Q <= 1.0:
        raise ValueError("η and η_Q lie between 0 and 1.")
    return {"contact_ppm": float(contact_ppm), "aniso_ppm": float(aniso_ppm),
            "eta": float(eta), "euler": tuple(float(a) for a in euler),
            "C_Q_MHz": float(C_Q_MHz), "eta_Q": float(eta_Q),
            "euler_Q": tuple(float(a) for a in euler_Q)}


def has_local(local) -> bool:
    return bool(local) and (local.get("aniso_ppm", 0.0) != 0.0 or local.get("C_Q_MHz", 0.0) != 0.0)


def _rotation_zyz(euler):
    from ..combined_hamiltonian import _rotation_zyz
    return _rotation_zyz(*euler)


def _sphere_points(n, rng):
    """n quasi-uniform unit vectors (Fibonacci sphere), randomly ordered."""
    k = np.arange(n) + 0.5
    z = 1.0 - 2.0 * k / n
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    phi = GOLDEN * np.arange(n) + rng.uniform(0, 2 * np.pi)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return pts[rng.permutation(n)]


def _frames(axes, rng):
    """Unit vectors e1, e2 completing each axis, turned by a random angle."""
    ref = np.zeros_like(axes)
    ref[:, 2] = 1.0
    ref[np.abs(axes[:, 2]) > 0.9] = [1.0, 0.0, 0.0]
    e1 = np.cross(axes, ref)
    e1 /= np.linalg.norm(e1, axis=1)[:, None]
    e2 = np.cross(axes, e1)
    psi = rng.uniform(0, 2 * np.pi, len(axes))[:, None]
    return np.cos(psi) * e1 + np.sin(psi) * e2, -np.sin(psi) * e1 + np.cos(psi) * e2


def _weighted_quantiles(values, weights, qs):
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cw = np.cumsum(w)
    if cw[-1] <= 0:
        return [float(v[0]), float(v[-1])]
    return [float(np.interp(q * cw[-1], cw, v)) for q in qs]


def _random_rotations(n, rng):
    from scipy.spatial.transform import Rotation
    return Rotation.random(n, random_state=rng).as_matrix()


def _spin_rows(nuclide, transitions, C_Q):
    from ..simulations import _select_transitions, quadrupolar_transition_table
    from ..combined_hamiltonian import _spin_of
    spin = _spin_of(nuclide)
    if spin <= 0.5 or C_Q == 0.0:
        return spin, [{"m": 0.5, "weight": 1.0, "group": "CT"}], 1.0
    rows = _select_transitions(spin, transitions)
    total = sum(r["weight"] for r in quadrupolar_transition_table(spin))
    return spin, rows, total


def _quad_offsets(spin, local, larmor_Hz, b_part, rows, order):
    """Quadrupolar offsets (Hz) and intensities for B₀ directions b_part
    (…, 3) in the particle frame."""
    from ..simulations import _perturbative_offsets, _exact_offsets
    shape = b_part.shape[:-1] + (len(rows),)
    w = np.array([r["weight"] for r in rows], dtype=float)
    if local is None or local.get("C_Q_MHz", 0.0) == 0.0 or spin <= 0.5:
        return np.zeros(shape), np.broadcast_to(w, shape)
    RQ = _rotation_zyz(local["euler_Q"])
    b_efg = b_part @ RQ                       # R_Qᵀ b for each b
    m = np.array([r["m"] for r in rows])
    C_Q = local["C_Q_MHz"] * 1e6
    if order == "exact":
        nu, inten = _exact_offsets(spin, C_Q, local["eta_Q"], larmor_Hz, b_efg, m)
        return nu, inten
    nu = _perturbative_offsets(spin, C_Q, local["eta_Q"], larmor_Hz, b_efg, m, order)
    return nu, np.broadcast_to(w, shape)


def _local_tensor(local):
    """The local shift tensor (ppm) in the particle frame."""
    if not local:
        return np.zeros((3, 3))
    from ..simulations import csa_definition
    pv = np.diag(np.asarray(csa_definition(local["contact_ppm"], local["aniso_ppm"], local["eta"],
                                           convention="Haeberlen"), dtype=float))
    R = _rotation_zyz(local["euler"])
    return R @ np.diag(pv) @ R.T


def sample_spectrum(field, *, nuclide="7Li", B0_tesla=9.4, mode="static", spin_rate_hz=10000.0,
                    sample=None, local=None, observe="inside", transitions="ct",
                    quad_order="second", n_samples=200000, seed=0, linewidth_hz=50.0,
                    lineshape="lorentzian", centre_ppm=None, sw_ppm=None, n_points=None,
                    chunk=20000, progress=None, offset_ppm=0.0, gamma_MHz_per_T=None) -> dict:
    """The spectrum of a :func:`bms_field` result (it needs the full tensor:
    ``components`` including ``"tensor"``, and the periodic k = 0 term of
    a sphere, which is what ``sample_shape="sphere"`` gives) with the
    macroscopic ``sample`` (:class:`Sample` or its dict; None: a sphere)
    and ``local`` interactions (:func:`local_interactions`), by sampling
    (see the module docstring). ``mode``: ``"static"`` or ``"mas"``.
    ``transitions``: ``"ct"`` or ``"all"`` (a quadrupolar nucleus);
    ``quad_order``: ``"first"``, ``"second"`` or ``"exact"``.

    Returns the standard spectrum dict with ``stats``: the weighted mean
    and standard deviation of the shift, ``mc_noise`` (rms difference of
    two halves over the peak), ``outside_window``, ``n_samples``,
    ``seconds``."""
    from ..utils import compute_BF1_SFO1
    from ..simulations import _mas_prepare, _mas_sticks, _bin_sticks
    from .fast import observed_weights, _broaden
    t_start = time.perf_counter()
    if mode not in ("static", "mas"):
        raise ValueError("mode: 'static' or 'mas' (infinitely fast MAS: use 'mas' at a high rate).")
    if "delta_tensor" not in field:
        raise ValueError("This field has no full tensor: calculate it with components "
                         "including 'tensor'.")
    sample = sample_from_dict(sample)
    if field["parameters"].get("boundary") == "periodic" and \
            field["parameters"].get("sample_shape") not in ("sphere", None):
        raise ValueError("Give the sample here, not to bms_field (whose sample_shape must be "
                         "'sphere'), or its bulk shift would be counted twice.")
    if quad_order not in ("first", "second", "exact"):
        raise ValueError("quad_order: 'first', 'second' or 'exact'.")
    n_samples = int(n_samples)
    if n_samples < 100:
        raise ValueError("At least 100 samples.")
    if mode == "mas" and spin_rate_hz <= 0:
        raise ValueError("The MAS rate must be positive.")
    BF1, SFO1, nuclide = compute_BF1_SFO1(nuclide, B0_tesla, offset_ppm=offset_ppm,
                                          gamma_MHz_per_T=gamma_MHz_per_T)
    larmor = BF1 * 1e6
    rng = np.random.default_rng(seed)
    C_Q = (local or {}).get("C_Q_MHz", 0.0)
    spin, rows, total_w = _spin_rows(nuclide, transitions, C_Q)
    n_tr = len(rows)

    # voxels: systematic sampling in proportion to the signal
    w_all = observed_weights(field, observe).ravel().astype(float)
    cum = np.cumsum(w_all)
    if cum[-1] <= 0:
        raise ValueError("No voxel carries signal.")
    u = (np.arange(n_samples) + rng.uniform()) / n_samples * cum[-1]
    vox = np.searchsorted(cum, u)
    vox = vox[rng.permutation(n_samples)]
    T6 = {ij: f.ravel() for ij, f in field["delta_tensor"].items()}
    owner = field["owner"].ravel()
    rot = field["rotations"]
    chi_v = float(np.trace(field["mean_chi"]) / 3.0)
    Tloc = _local_tensor(local)
    use_local = bool(local) and (np.any(Tloc != 0) or C_Q != 0.0)

    # orientations of the packing in the sample, and cells of the sample
    axes = _sphere_points(n_samples, rng)                 # sample z in packing frame
    e1, e2 = _frames(axes, rng)
    N_cells, _wc = sample.cells()
    cell = np.resize(rng.permutation(len(N_cells)), n_samples)
    rng.shuffle(cell)
    psi = rng.uniform(0, 2 * np.pi, n_samples) if sample.axisymmetric else None

    beta = np.radians(MAGIC_ANGLE_DEG if mode == "mas" else sample.axis_angle)
    gam = 2 * np.pi * np.arange(N_GAMMA) / N_GAMMA if mode == "mas" else np.zeros(1)
    b_s = np.stack([np.sin(beta) * np.cos(gam), np.sin(beta) * np.sin(gam),
                    np.full_like(gam, np.cos(beta))], -1)          # (G, 3) sample frame

    lw_ppm = linewidth_hz / BF1
    axis = None
    if centre_ppm is not None and sw_ppm is not None:
        axis = (float(centre_ppm), float(sw_ppm))
    h = None
    under = False
    x = None
    total = 0.0
    acc = [0.0, 0.0, 0.0]
    for start in range(0, n_samples, chunk):
        stop = min(start + chunk, n_samples)
        n = stop - start
        v = vox[start:stop]
        T = np.empty((n, 3, 3))
        for (i, j), f in T6.items():
            T[:, i, j] = T[:, j, i] = f[v]
        # B0 in the packing frame along the rotor cycle: (n, G, 3)
        bp = (b_s[None, :, 0, None] * e1[start:stop, None, :]
              + b_s[None, :, 1, None] * e2[start:stop, None, :]
              + b_s[None, :, 2, None] * axes[start:stop, None, :])
        shift = np.einsum("ngi,nij,ngj->ng", bp, T, bp)
        M = sample.macro_tensors(chi_v, cell[start:stop], None if psi is None else psi[start:stop])
        shift += np.einsum("gi,nij,gj->ng", b_s, M, b_s)
        if use_local:
            own = owner[v]
            Rp = np.empty((n, 3, 3))
            inside = own >= 0
            if inside.any():
                Rp[inside] = rot[own[inside]]
            if (~inside).any():
                Rp[~inside] = _random_rotations(int((~inside).sum()), rng)
            b_part = np.einsum("nij,ngj->ngi", Rp, bp)
            shift += np.einsum("ngi,ij,ngj->ng", b_part, Tloc, b_part)
            nu_q, inten = _quad_offsets(spin, local, larmor, b_part, rows, quad_order)
        else:
            nu_q, inten = _quad_offsets(spin, None, larmor, bp, rows, quad_order)
        centre_shift = shift.mean(axis=1)                 # the isotropic shift (MAS) / δ (static)
        acc[0] += n
        acc[1] += float(centre_shift.sum())
        acc[2] += float((centre_shift ** 2).sum())
        nu = larmor * 1e-6 * shift[..., None] + nu_q                    # (n, G, T) Hz
        inten = np.asarray(inten, dtype=float) / total_w
        half = (np.arange(start, stop) % 2).astype(np.int8)
        if mode == "static":
            pos = nu[:, 0, :] / larmor * 1e6
            wt = inten[:, 0, :] if inten.ndim == 3 else np.broadcast_to(inten, pos.shape)
            parts = [(pos, np.ascontiguousarray(wt), np.repeat(half[:, None], n_tr, axis=1))]
        else:
            prep = _mas_prepare(nu, spin_rate_hz, N_GAMMA)
            it = inten.mean(axis=1) if inten.ndim == 3 else np.broadcast_to(inten, (n, n_tr))
            parts = []
            for a, b, centre, sb, weight in _mas_sticks(prep, it, np.ones(n)):
                pos = (centre[:, None, :] + sb[None, :, None]) / larmor * 1e6
                hh = np.broadcast_to(half[a:b, None, None], weight.shape)
                parts.append((pos, weight, hh))
        if x is None:
            if axis is None:
                P = np.concatenate([p_.ravel() for p_, _w, _h in parts])
                W = np.concatenate([w_.ravel() for _p, w_, _h in parts])
                lo, hi = _weighted_quantiles(P, W, (1e-4, 1 - 1e-4))
                pad = 0.1 * (hi - lo) + 20 * lw_ppm
                lo, hi = lo - pad, hi + pad
                if centre_ppm is not None:
                    half_w = max(centre_ppm - lo, hi - centre_ppm)
                    lo, hi = centre_ppm - half_w, centre_ppm + half_w
                axis = (0.5 * (lo + hi), max(hi - lo, 1e-3) if sw_ppm is None else sw_ppm)
            c, sw = axis
            from .fast import resolve_points
            n_points, under = resolve_points(sw, lw_ppm, n_points)
            x = c - sw / 2 + sw * (np.arange(n_points) + 0.5) / n_points
            h = [np.zeros(n_points), np.zeros(n_points)]
        for p_, w_, hh in parts:
            total += float(w_.sum())
            for k in (0, 1):
                sel = hh == k
                h[k] += _bin_sticks(x, p_[sel], w_[sel])
        if progress is not None:
            progress(stop / n_samples)

    mean = acc[1] / acc[0]
    std = float(np.sqrt(max(acc[2] / acc[0] - mean * mean, 0.0)))
    c, sw = axis
    hist = h[0] + h[1]
    outside = 1.0 - float(hist.sum()) / total if total else 0.0
    sw_hz = sw * BF1
    spec = _broaden(hist, sw_hz, linewidth_hz, lineshape)
    s0, s1 = (_broaden(hk, sw_hz, linewidth_hz, lineshape) for hk in h)
    peak = np.max(np.abs(spec)) or 1.0
    noise = float(np.sqrt(np.mean((s0 - s1) ** 2)) / peak) if peak else 0.0
    return {
        "x_ppm": x, "x_Hz": x * BF1, "real_spectrum": spec, "normalised_real_spectrum": spec / peak,
        "imag_spectrum": np.zeros_like(spec), "normalised_imag_spectrum": np.zeros_like(spec),
        "nuclide": nuclide, "BF1": float(BF1), "SFO1": float(SFO1), "path": None,
        "stats": {"mean_ppm": float(mean), "std_ppm": std, "fwhm_gauss_ppm": 2.3548 * std,
                  "outside_window": outside, "mc_noise": noise, "n_samples": n_samples,
                  "weight": float(total), "n_points": int(n_points), "under_resolved": under,
                  "seconds": time.perf_counter() - t_start, "spin": spin,
                  "transitions": [r.get("group", "CT") for r in rows], "chi_v": chi_v},
        "parameters": {"nuclide": nuclide, "B0_tesla": float(B0_tesla), "mode": mode,
                       "spin_rate_hz": float(spin_rate_hz), "sample": sample.to_dict(),
                       "local": dict(local) if local else None, "observe": observe,
                       "transitions": transitions, "quad_order": quad_order,
                       "n_samples": n_samples, "seed": seed,
                       "linewidth_hz": float(linewidth_hz), "lineshape": lineshape},
    }
