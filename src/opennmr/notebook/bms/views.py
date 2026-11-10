"""
Pictures and run-time estimates for the BMS simulator (explicit user
request, 2026-11-07: "a 3-d render of the materials packed"; "timing
warnings for simulations, as there are for other dense calculations").

* :func:`plot_packing_3d` -- every particle drawn as its own surface
  (ellipsoid, hexagonal prism or torus, with the shape and orientation
  the simulation uses), in the box, B₀ along z.
* :func:`estimate_field_time` / :func:`estimate_sampling_time` -- how long
  a run will take before it starts: the field from the grid and the
  particles' blocks with per-session calibrated costs, the sampled
  spectrum from a small trial run of the same calculation (as
  :mod:`opennmr.notebook.run_time` does for the Solid-state simulator).
"""

from __future__ import annotations

import time

import numpy as np

from .geometry import quat_to_rotmat

_COSTS: dict = {}


# ---------------------------------------------------------------------------
# 3-D packing
# ---------------------------------------------------------------------------

def particle_volume(p) -> float:
    """The volume of a particle record's shape (the sharp-edged limit):
    superellipsoid, hexagonal prism (apothem v1, height v3) or torus
    (radii v1, v2)."""
    from scipy.special import beta as B
    eq = p["Equation"]
    if eq == "cuboid":
        return float(8.0 * p["v1"] * p["v2"] * p["v3"])
    if eq == "hexagonal":
        return float(2.0 * np.sqrt(3.0) * p["v1"] ** 2 * p["v3"])
    if eq == "donut":
        return float(2.0 * np.pi ** 2 * p["v1"] * p["v2"] ** 2)
    r, t = float(p["r"]), float(p["t"])
    # superellipsoid ((x/a)^r + (y/b)^r)^(t/r) + (z/c)^t <= 1, ε1 = 2/t, ε2 = 2/r (Jaklič)
    e1, e2 = 2.0 / t, 2.0 / r
    return float(2.0 * p["v1"] * p["v2"] * p["v3"] * e1 * e2 * B(e1 / 2 + 1, e1)
                 * B(e2 / 2, e2 / 2))


def _surface_points(p, n_u, n_v):
    """Points (n_u, n_v, 3) of particle ``p``'s surface, particle frame."""
    eq = p["Equation"]
    v1, v2, v3 = p["v1"], p["v2"], p["v3"]
    u = np.linspace(0, 2 * np.pi, n_u)
    if eq == "donut":
        R0, r0 = v1, v2
        w = np.linspace(0, 2 * np.pi, n_v)
        U, W = np.meshgrid(u, w, indexing="ij")
        ring = R0 + r0 * np.cos(W)
        a, b, c = ring * np.cos(U), ring * np.sin(U), r0 * np.sin(W)
        return _place_axis(a, b, c, p["Deformation"])
    v = np.linspace(0, np.pi, n_v)
    U, V = np.meshgrid(u, v, indexing="ij")
    r, t = float(p["r"]), float(p["t"])
    # superellipsoid |x/v1|^r … : signed powers keep the shape the mask uses
    def sp(x, e):
        return np.sign(x) * np.abs(x) ** e
    cv, sv = np.cos(V), np.sin(V)
    x = v1 * sp(sv, 2.0 / t) * sp(np.cos(U), 2.0 / r)
    y = v2 * sp(sv, 2.0 / t) * sp(np.sin(U), 2.0 / r)
    z = v3 * sp(cv, 2.0 / t)
    return np.stack([x, y, z], -1)


def _place_axis(a, b, c, deform):
    """(a, b, c) of a shape whose axis is c, into the particle frame as
    geometry's masks lay it out (C: z, B: y, A: x)."""
    if deform == "C":
        return np.stack([a, b, c], -1)
    if deform == "B":
        return np.stack([a, c, b], -1)
    return np.stack([c, a, b], -1)


def _hex_prism(p):
    """Faces (list of (k, 3) arrays) of a hexagonal prism, particle frame:
    apothem v1 in the hexagon's plane, height v3 (geometry.hexagonal_mask)."""
    Rv = p["v1"] / np.cos(np.pi / 6)                 # circumradius from the apothem
    h = p["v3"] / 2
    ang = np.pi / 6 + np.arange(6) * np.pi / 3
    a, b = Rv * np.cos(ang), Rv * np.sin(ang)
    top = _place_axis(a, b, np.full(6, h), p["Deformation"])
    bot = _place_axis(a, b, np.full(6, -h), p["Deformation"])
    faces = [top, bot]
    for k in range(6):
        j = (k + 1) % 6
        faces.append(np.array([bot[k], bot[j], top[j], top[k]]))
    return faces


def _cuboid(p):
    """The six faces of a block, particle frame (half sides v1, v2, v3)."""
    a, b, c = p["v1"], p["v2"], p["v3"]
    corners = np.array([[sx * a, sy * b, sz * c] for sx in (-1, 1) for sy in (-1, 1)
                        for sz in (-1, 1)])
    index = ((0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4), (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5))
    return [corners[list(f)] for f in index]


def plot_packing_3d(particles, magnetic_susceptibility=(1e-3, 1e-3, 1e-3), *, box_size=None,
                    equation="ellipsoid", deformation="A", r=2, t=2, slope_filter=2000,
                    figure=None, fontsize=11.0, colour_by="size", cmap="viridis",
                    tensor_style=None, unit="", alpha=0.85, max_mesh=4000):
    """The packing in 3-D: each particle's surface at its position and
    orientation, the box, and B₀ (z). ``particles`` as for
    :func:`~opennmr.notebook.bms.fast.bms_field` (with the same shape
    arguments). ``colour_by``: ``"size"`` (volume), ``"orientation"``
    (the angle of the particle's z axis to B₀) or ``"none"``."""
    import matplotlib as mpl
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from ..tensor_views import _new_3d_axes, _finish_3d
    from .fast import prepare_particles, _box_lengths
    parts = prepare_particles(particles, magnetic_susceptibility, equation, deformation, r, t,
                              slope_filter)
    figure, ax = _new_3d_axes(figure)
    n = len(parts)
    if n == 0:
        raise ValueError("No particles to draw.")
    # mesh density per particle so a large packing stays quick to draw
    per = max(6, int(np.sqrt(max_mesh * 50 / max(n, 1))))
    n_u, n_v = min(per, 28), min(max(per // 2, 5), 16)
    if colour_by == "orientation":
        vals = np.array([abs(quat_to_rotmat(p["q"])[2, 2]) for p in parts])   # |cos| of z to B0
        label = r"|cos $\theta$|"
    elif colour_by == "size":
        vals = np.array([particle_volume(p) for p in parts])
        label = f"volume ({unit or 'box units'}³)"
    else:
        vals = np.zeros(n)
        label = None
    lo, hi = float(vals.min()), float(vals.max())
    norm = mpl.colors.Normalize(lo, hi if hi > lo else lo + 1.0)
    cm = mpl.colormaps[cmap]
    for p, val in zip(parts, vals):
        R = quat_to_rotmat(p["q"])
        pos = np.asarray(p["posc"], float)
        colour = cm(norm(val)) if colour_by != "none" else "0.6"
        if p["Equation"] in ("hexagonal", "cuboid"):
            shape_faces = _hex_prism(p) if p["Equation"] == "hexagonal" else _cuboid(p)
            faces = [f @ R + pos for f in shape_faces]           # Rᵀ f: particle -> packing
            ax.add_collection3d(Poly3DCollection(faces, facecolor=colour, edgecolor="0.25",
                                                 linewidths=0.2, alpha=alpha))
        else:
            pts = _surface_points(p, n_u, n_v) @ R + pos
            ax.plot_surface(pts[..., 0], pts[..., 1], pts[..., 2], color=colour, alpha=alpha,
                            linewidth=0, antialiased=False, shade=True)
    if box_size is not None:
        L = _box_lengths(box_size) / 2
        for sx in (-1, 1):
            for sy in (-1, 1):
                ax.plot([sx * L[0]] * 2, [sy * L[1]] * 2, [-L[2], L[2]], color="0.4", lw=0.6)
                ax.plot([sx * L[0]] * 2, [-L[1], L[1]], [sy * L[2]] * 2, color="0.4", lw=0.6)
                ax.plot([-L[0], L[0]], [sx * L[1]] * 2, [sy * L[2]] * 2, color="0.4", lw=0.6)
        extent = float(L.max())
    else:
        extent = float(max(np.abs(np.asarray([p["posc"] for p in parts])).max()
                           + max(max(p["v1"], p["v2"], p["v3"]) for p in parts), 1.0))
    ax.quiver(extent * 0.9, -extent * 0.9, -extent * 0.6, 0, 0, extent * 0.9, color="k",
              lw=1.2, arrow_length_ratio=0.15)
    ax.text(extent * 0.9, -extent * 0.9, extent * 0.4, r"$B_0$", fontsize=fontsize)
    _finish_3d(ax, extent, fontsize, unit or "box units", tensor_style)
    bar = label is not None and hi > lo
    # room for the z label on the left and the colour bar's labels on the right
    figure.subplots_adjust(left=0.13, right=0.76 if bar else 0.9)
    if bar:
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cm)
        cax = figure.add_axes([0.8, 0.25, 0.022, 0.5])
        cb = figure.colorbar(sm, cax=cax)
        cb.set_label(label, fontsize=fontsize * 0.85)
        cb.ax.tick_params(labelsize=fontsize * 0.8)
    ax.set_title(f"{n} particles", fontsize=fontsize)
    return figure


# ---------------------------------------------------------------------------
# Run-time estimates
# ---------------------------------------------------------------------------

def _calibrate():
    """Per-session costs: s per (N log₂ N) of a float32 real FFT pair, and
    s per voxel of a particle's shape."""
    if _COSTS:
        return _COSTS
    import scipy.fft as sfft
    from .fast import block_shape, prepare_particles
    from .io_utils import make_particle
    n = 64
    a = np.random.default_rng(0).random((n, n, n), dtype=np.float32)
    t0 = time.perf_counter()
    for _ in range(2):
        sfft.irfftn(sfft.rfftn(a, workers=-1), s=a.shape, workers=-1)
    t_fft = (time.perf_counter() - t0) / 2 / (2 * n ** 3 * np.log2(n ** 3))
    p = prepare_particles([make_particle(0, 0, 0, (10, 8, 6), quaternion=(0.9, 0.2, 0.3, 0.1))],
                          (1e-3,) * 3)[0]
    xs = np.linspace(-12, 12, 48)
    t0 = time.perf_counter()
    for _ in range(3):
        block_shape(p, xs, xs, xs)
    t_shape = (time.perf_counter() - t0) / 3 / 48 ** 3
    _COSTS.update(fft=t_fft, shape=t_shape)
    return _COSTS


def estimate_field_time(particles, magnetic_susceptibility, *, box_size=300.0, gridpoints=128,
                        boundary="periodic", components=("tensor",), equation="ellipsoid",
                        deformation="A", r=2, t=2, slope_filter=2000, dtype="float32") -> dict:
    """Seconds and memory a :func:`bms_field` run will need:
    ``{"seconds", "memory_GB", "text"}``. The shapes are costed from each
    particle's block, the FFTs from their size, both with costs measured
    on this computer once per session."""
    from .fast import prepare_particles, _box_lengths, _reach
    from .packing import _bounding_radius
    from ..run_time import format_duration as duration
    c = _calibrate()
    parts = prepare_particles(particles, magnetic_susceptibility, equation, deformation, r, t,
                              slope_filter)
    n = int(gridpoints)
    L = _box_lengths(box_size)
    step = L / n
    voxels = 0.0
    for p in parts:
        R = float(_bounding_radius(p["Equation"], p["v1"], p["v2"], p["v3"])) * _reach(p, 1.15)
        voxels += float(np.prod(np.minimum(2 * R / step + 2, n)))
    n_fft = n * (2 if boundary == "isolated" else 1)
    N = float(n_fft) ** 3
    chi = np.asarray(magnetic_susceptibility, float)
    iso = bool(np.allclose(chi, chi.mean()))
    n_forward = 1 if iso else 6
    n_inverse = 6 if "tensor" in components else len(components)
    seconds = (voxels * c["shape"] + (n_forward + n_inverse) / 2 * N * np.log2(N) * c["fft"] * 2
               + 2e-8 * N * (n_inverse + 4))          # element-wise work around the FFTs
    item = 4 if dtype == "float32" else 8
    fields = (1 if iso else 6) + 2 + (6 if "tensor" in components else len(components))
    memory = (fields * N * item + 4 * N * item + 4 * N) / 1e9      # + complex work arrays, owner
    text = f"about {duration(seconds)}"
    if memory > 1.0:
        text += f", {memory:.1f} GB of memory"
    return {"seconds": float(seconds), "memory_GB": float(memory), "text": text}


def estimate_sampling_time(field, n_samples, **kwargs) -> dict:
    """Seconds :func:`interactions.sample_spectrum` will take for
    ``n_samples`` (the same arguments otherwise): two short trial runs of
    the same calculation, ``time = a + b·samples``."""
    from .interactions import sample_spectrum
    from ..run_time import format_duration as duration
    kwargs = dict(kwargs)
    kwargs.pop("progress", None)
    times = []
    for n in (1000, 4000):
        t0 = time.perf_counter()
        sample_spectrum(field, n_samples=n, **{k: v for k, v in kwargs.items()
                                                if k != "n_samples"})
        times.append(time.perf_counter() - t0)
    b = max((times[1] - times[0]) / 3000.0, times[1] / 4000.0 * 0.5)
    a = max(times[0] - 1000 * b, 0.0)
    seconds = a + b * float(n_samples)
    return {"seconds": float(seconds), "text": f"about {duration(seconds)}",
            "per_sample_s": float(b)}
