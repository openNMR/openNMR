"""
The macroscopic sample around a BMS packing (explicit user request,
2026-11-07: "simulate their sample in a rotor (0.4 ... 7.0 mm, corrected
for the wall thickness) as well as in a static plate or other format").

The packing of :mod:`opennmr.notebook.bms.fast` is one cell of a powder
that fills the sample. On the scale of the sample the powder is a
continuum with the volume susceptibility χ_v (the cell's mean χ: the
particles' χ times their volume fraction), and its own demagnetising
field depends on where in the sample a cell sits. With the Lorentz sphere
of each cell (the k = 0 term of the cell taken as a sphere, D = 1/3), a
cell at r adds the shift tensor

    δ_M(r) = −1e6 · χ_v · (N(r) − I/3),

N(r) the demagnetising tensor field of the uniformly magnetised sample
(H = −N M inside it; tr N = 1 everywhere inside). N(r) is uniform only
in an ellipsoid (a sphere: N = I/3, no shift); in a rotor or a plate it
varies, most near the ends and edges.

N(r) is computed from the surface "magnetic charges" σ = M·n:

    N_ij(r) = −(1/4π) ∮ (r − r')_i n_j(r') / |r − r'|³ dS'.

* Cylinder (rotor, tube) of radius R and length L, axis z: on the side
  the z' integral and on the end discs the ρ' integral are done
  analytically, the azimuth φ' numerically (periodic trapezoid, which
  converges exponentially). The field is axially symmetric, so it is
  computed at φ = 0 on a (ρ, z) grid and rotated.
* Rectangular plate (cuboid) of side lengths (a, b, c): every face
  integral is closed form (arctan for the normal component, logarithms
  for the in-plane ones).

Checks (tests): tr N = 1; N_zz at the centre of a cylinder,
1 − (L/2)/√((L/2)² + R²); N_zz at the centre of a cuboid,
(2/π) arctan(ab / (c √(a² + b² + c²))) for half-sides a, b, c; a long
cylinder → diag(½, ½, 0); a cube's centre → I/3.

Rotor dimensions
----------------
:data:`ROTORS` gives, for each outer diameter, a typical inner diameter
(the outer one less the wall) and a typical length of the packed sample.
They are typical values, not one manufacturer's specification: walls
and inserts differ between makers and rotor types (thin-walled rotors,
spacers, inserts), so check yours. What matters for the spectrum is the
shape of the packed volume -- its length over its inner diameter -- and
how full the rotor is.
"""

from __future__ import annotations

import numpy as np

MAGIC_ANGLE_DEG = float(np.degrees(np.arccos(1.0 / np.sqrt(3.0))))

#: Outer diameter (mm) -> (typical inner diameter, typical packed length), mm.
#: Typical values (see the module docstring): set your own rotor's.
ROTORS = {
    0.4: (0.20, 2.0),
    0.7: (0.40, 4.6),
    1.3: (0.80, 6.0),
    1.9: (1.30, 9.0),
    2.5: (1.60, 6.0),
    3.2: (2.20, 10.0),
    4.0: (3.00, 10.0),
    7.0: (5.00, 15.0),
}

#: Volume susceptibilities (SI, dimensionless) of rotor and cap materials:
#: key -> (χ, description and source). Rotors vary (stabilisers, impurities):
#: give your own value when you know it.
MATERIALS = {
    "zirconia": (-8.5e-6, "Zirconia (ZrO₂): ≈ −8.5 ppm, from pure ZrO₂'s molar susceptibility "
                          "(−13.8 × 10⁻⁶ cm³/mol, CRC) at 6.05 g/cm³; stabilisers and impurities "
                          "change it"),
    "vespel": (-9.02e-6, "Vespel: −9.02 ppm (Wapler et al., Magn. Reson. 2014, Table 7)"),
    "ptfe": (-10.28e-6, "PTFE: −10.28 ppm (Wapler et al. 2014, measured)"),
    "peek": (-9.33e-6, "PEEK: −9.33 ppm (Wapler et al. 2014, measured)"),
    "kapton": (-8.92e-6, "Kapton (polyimide): −8.92 ppm (Wapler et al. 2014, measured)"),
    "none": (0.0, "Nothing (air)"),
}

#: Sample shapes: key -> description.
SHAPES = {
    "sphere": "Sphere (no macroscopic shift)",
    "cylinder": "Rotor or tube (cylinder)",
    "plate": "Plate (rectangular block)",
}


# ---------------------------------------------------------------------------
# Demagnetising tensor fields
# ---------------------------------------------------------------------------

def cylinder_demag(rho, z, radius, length, n_phi=4096):
    """N(r) (…, 3, 3) inside a uniformly magnetised cylinder of ``radius``
    and ``length`` (axis z, centred at the origin), at the points
    (ρ, 0, z). Rotate by the azimuth for other points
    (:func:`rotate_about_z`)."""
    x = np.atleast_1d(np.asarray(rho, dtype=float))
    zz = np.atleast_1d(np.asarray(z, dtype=float))
    x, zz = np.broadcast_arrays(x, zz)
    shape = x.shape
    x, zz = x.ravel()[:, None], zz.ravel()[:, None]
    R, Lh = float(radius), 0.5 * float(length)
    phi = 2.0 * np.pi * (np.arange(n_phi) + 0.5) / n_phi
    dphi = 2.0 * np.pi / n_phi
    c, s = np.cos(phi)[None, :], np.sin(phi)[None, :]
    G = np.zeros((x.shape[0], 3, 3))
    # side: ∫ dz' analytically
    a2 = x * x + R * R - 2.0 * x * R * c
    a2 = np.maximum(a2, 1e-300)
    u1, u2 = -Lh - zz, Lh - zz
    r1, r2 = np.sqrt(a2 + u1 * u1), np.sqrt(a2 + u2 * u2)
    I0 = (u2 / r2 - u1 / r1) / a2
    I1 = 1.0 / r2 - 1.0 / r1
    vec = (( x - R * c) * I0, (-R * s) * I0, I1)
    n_side = (c, s)
    for i in range(3):
        for j in range(2):
            G[:, i, j] += R * dphi * np.sum(vec[i] * n_side[j], axis=1)
    # end discs: ∫ ρ' dρ' analytically
    for zd, sign in ((Lh, 1.0), (-Lh, -1.0)):
        h = zz - zd
        b2 = x * x * s * s + h * h
        b = np.sqrt(np.maximum(b2, 1e-300))
        xc = x * c

        def A(t):
            return t / (b2 * np.sqrt(t * t + b2))

        def B(t):
            return -1.0 / np.sqrt(t * t + b2)

        def C(t):
            return np.arcsinh(t / b) - t / np.sqrt(t * t + b2)
        t0, t1 = -xc, R - xc
        J1 = (B(t1) + xc * A(t1)) - (B(t0) + xc * A(t0))
        J2 = (C(t1) + 2 * xc * B(t1) + xc * xc * A(t1)) - (C(t0) + 2 * xc * B(t0) + xc * xc * A(t0))
        vx, vy, vz = x * J1 - c * J2, -s * J2, h * J1
        for i, v in enumerate((vx, vy, vz)):
            G[:, i, 2] += sign * dphi * np.sum(v, axis=1)
    N = -G / (4.0 * np.pi)
    return N.reshape(shape + (3, 3))


def cuboid_demag(points, sides):
    """N(r) (…, 3, 3) inside a uniformly magnetised rectangular block of
    side lengths ``sides`` (a, b, c) along x, y, z, centred at the origin,
    at ``points`` (…, 3). Closed form."""
    P = np.asarray(points, dtype=float)
    shape = P.shape[:-1]
    P = P.reshape(-1, 3)
    h = 0.5 * np.asarray(sides, dtype=float)
    G = np.zeros((P.shape[0], 3, 3))
    for k in range(3):
        p, q = [m for m in range(3) if m != k]
        for face, sign in ((h[k], 1.0), (-h[k], -1.0)):
            d = P[:, k] - face
            lo_p, hi_p = P[:, p] - h[p], P[:, p] + h[p]
            lo_q, hi_q = P[:, q] - h[q], P[:, q] + h[q]
            tot = {k: 0.0, p: 0.0, q: 0.0}
            for Pv, sp in ((hi_p, 1.0), (lo_p, -1.0)):
                for Qv, sq in ((hi_q, 1.0), (lo_q, -1.0)):
                    Rr = np.sqrt(Pv * Pv + Qv * Qv + d * d)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        Fn = np.arctan(Pv * Qv / (d * Rr))       # d ≠ 0 inside the block
                    Fp = -np.log(Qv + Rr)
                    Fq = -np.log(Pv + Rr)
                    tot[k] = tot[k] + sp * sq * Fn
                    tot[p] = tot[p] + sp * sq * Fp
                    tot[q] = tot[q] + sp * sq * Fq
            for i in range(3):
                G[:, i, k] += sign * tot[i]
    N = -G / (4.0 * np.pi)
    return N.reshape(shape + (3, 3))


def rotate_about_z(N, psi):
    """R_z(ψ) N R_z(ψ)ᵀ for each ψ (N (…, 3, 3), ψ broadcastable to …)."""
    c, s = np.cos(psi), np.sin(psi)
    R = np.zeros(np.shape(psi) + (3, 3))
    R[..., 0, 0], R[..., 0, 1], R[..., 1, 0], R[..., 1, 1], R[..., 2, 2] = c, -s, s, c, 1.0
    return R @ N @ np.swapaxes(R, -1, -2)


_CYL_CACHE: dict = {}


def _n_phi(rho, radius, h_min=None):
    """Azimuth points for a converged φ' integral at ρ: the integrand
    peaks over an angle ~ |R − ρ|/R near the side, and ~ |h|/R near an end
    face at distance h (checked: tr N = 1 to 1e-9 inside, 0 outside)."""
    gap = max(abs(radius - rho) / radius, 1e-6)
    if h_min is not None:
        gap = min(gap, max(abs(h_min) / radius, 1e-6))
    return int(min(16384, max(128, 2 ** int(np.ceil(np.log2(96.0 / gap))))))


def _cylinder_cells(R, L, n_r, n_z):
    """N on the equal-volume (ρ², z) cells of a cylinder, ring by ring with
    as many azimuth points as each ring needs; z → −z by symmetry."""
    key = (round(L / R, 9), n_r, n_z)
    if key not in _CYL_CACHE:
        rho = R * np.sqrt((np.arange(n_r) + 0.5) / n_r)
        z = L * ((np.arange(n_z) + 0.5) / n_z - 0.5)
        half = z[n_z // 2:]                         # z ≥ 0 (n_z even) or ≥ centre
        out = np.empty((n_r, n_z, 3, 3))
        M = np.diag([1.0, 1.0, -1.0])
        for i, r in enumerate(rho):
            Nh = cylinder_demag(np.full(len(half), r), half, R, L, n_phi=_n_phi(r, R))
            out[i, n_z // 2:] = Nh
            out[i, :n_z - n_z // 2] = (M @ Nh @ M)[::-1][:n_z - n_z // 2]
        _CYL_CACHE[key] = out.reshape(-1, 3, 3)
    return _CYL_CACHE[key]


def _bodies_on_cells(parts, R, L, n_r, n_z):
    """Σ w N of uniformly magnetised cylinders ``parts`` [(Rc, Lc, zc, w)]
    on the sample's axis -- a set symmetric under z → −z (a tube; a pair
    of caps) -- at the sample's (ρ², z) cells, inside or outside each body
    (outside, tr N = 0 and there is no Lorentz term). Computed for z ≥ 0
    and mirrored."""
    key = ("bodies", tuple((round(Rc / R, 9), round(Lc / R, 9), round(zc / R, 9), w)
                           for Rc, Lc, zc, w in parts), round(L / R, 9), n_r, n_z)
    if key not in _CYL_CACHE:
        rho = R * np.sqrt((np.arange(n_r) + 0.5) / n_r)
        z = L * ((np.arange(n_z) + 0.5) / n_z - 0.5)
        half = z[n_z // 2:]
        out = np.zeros((n_r, n_z, 3, 3))
        Mz = np.diag([1.0, 1.0, -1.0])
        for Rc, Lc, zc, w in parts:
            dz = half - zc
            h_min = float(np.min(np.abs(np.concatenate([dz - 0.5 * Lc, dz + 0.5 * Lc]))))
            for i, r in enumerate(rho):
                near_face = h_min if abs(r - Rc) < 0.5 * Rc else None
                out[i, n_z // 2:] += w * cylinder_demag(np.full(len(half), r), dz, Rc, Lc,
                                                        n_phi=_n_phi(r, Rc, near_face))
        for i in range(n_r):
            out[i, :n_z - n_z // 2] = (Mz @ out[i, n_z // 2:] @ Mz)[::-1][:n_z - n_z // 2]
        _CYL_CACHE[key] = out.reshape(-1, 3, 3)
    return _CYL_CACHE[key]


# ---------------------------------------------------------------------------
# The sample: cells of equal volume and their N
# ---------------------------------------------------------------------------

class Sample:
    """The macroscopic sample: its shape, its axis's angle to B₀ when
    static, and N(r) on cells of equal volume.

    ``shape``: ``"sphere"``, ``"cylinder"`` (``diameter``, ``length``: the
    packed volume, i.e. the rotor's inner diameter and the fill length) or
    ``"plate"`` (``sides`` (a, b, c), c the thickness along the plate
    normal). The sample frame has z along the cylinder axis or the plate
    normal. ``axis_angle``: the angle (degrees) between that axis and B₀
    in a static experiment (a tube in a solution probe: 0; a rotor at
    rest in a MAS probe: the magic angle; a plate lying in a horizontal
    coil: 90). Under MAS the rotor axis is at the magic angle and turns."""

    def __init__(self, shape="sphere", *, diameter=None, length=None, sides=None,
                 axis_angle=0.0, n_cells=None, rotor=None):
        if shape not in SHAPES:
            raise ValueError(f"Sample shape: one of {', '.join(SHAPES)}.")
        self.shape = shape
        self.axis_angle = float(axis_angle)
        if shape == "cylinder":
            if not diameter or not length or diameter <= 0 or length <= 0:
                raise ValueError("A cylinder needs a positive diameter and length (the packed "
                                 "volume: the rotor's inner diameter and the fill length).")
            self.diameter, self.length = float(diameter), float(length)
        elif shape == "plate":
            if sides is None or len(sides) != 3 or min(sides) <= 0:
                raise ValueError("A plate needs three positive side lengths (a, b, thickness).")
            self.sides = tuple(float(v) for v in sides)
        self.rotor = None
        if rotor is not None:
            if shape != "cylinder":
                raise ValueError("A rotor holds a cylindrical sample.")
            r = dict(rotor)
            od, Lr = float(r.get("outer_diameter", 0)), float(r.get("length", 0))
            if od <= self.diameter:
                raise ValueError(f"The rotor's outer diameter ({od:g}) must be larger than the "
                                 f"inner one ({self.diameter:g}).")
            if Lr < self.length:
                raise ValueError(f"The rotor ({Lr:g} long) must be at least as long as the "
                                 f"packed sample ({self.length:g}).")
            self.rotor = {"outer_diameter": od, "length": Lr,
                          "chi_wall": float(r.get("chi_wall", 0.0)),
                          "chi_caps": float(r.get("chi_caps", 0.0))}
        self._cells = None
        self._rotor_cells = None
        self._n_cells = n_cells

    def describe(self) -> str:
        if self.shape == "sphere":
            return "sphere"
        if self.shape == "cylinder":
            text = (f"cylinder {self.diameter:g} × {self.length:g} (inner diameter × length, "
                    f"length/diameter {self.length / self.diameter:.3g})")
            if self.rotor:
                r = self.rotor
                text += (f" in a rotor {r['outer_diameter']:g} × {r['length']:g} (wall χ "
                         f"{r['chi_wall'] * 1e6:.3g} ppm, caps χ {r['chi_caps'] * 1e6:.3g} ppm)")
            return text
        a, b, c = self.sides
        return f"plate {a:g} × {b:g} × {c:g} (thickness {c:g})"

    def to_dict(self) -> dict:
        d = {"shape": self.shape, "axis_angle": self.axis_angle}
        if self.shape == "cylinder":
            d.update(diameter=self.diameter, length=self.length)
            if self.rotor:
                d["rotor"] = dict(self.rotor)
        elif self.shape == "plate":
            d.update(sides=list(self.sides))
        return d

    def cells(self):
        """``(N, weight)``: N (n, 3, 3) at cells of equal volume in the
        sample frame (for a cylinder at azimuth 0: rotate by a random
        azimuth, :func:`rotate_about_z`), weights summing to 1."""
        if self._cells is not None:
            return self._cells
        if self.shape == "sphere":
            N = (np.eye(3) / 3.0)[None]
        elif self.shape == "cylinder":
            n_r, n_z = self._n_cells or (48, 96)
            N = _cylinder_cells(0.5 * self.diameter, self.length, n_r, n_z)
        else:
            n = self._n_cells or (24, 24, 24)
            axes = [s * ((np.arange(k) + 0.5) / k - 0.5) for s, k in zip(self.sides, n)]
            pts = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
            N = cuboid_demag(pts, self.sides)
        N = 0.5 * (N + np.swapaxes(N, -1, -2))
        w = np.full(len(N), 1.0 / len(N))
        self._cells = (N, w)
        return self._cells

    @property
    def axisymmetric(self) -> bool:
        return self.shape == "cylinder"

    def rotor_cells(self):
        """χ_wall (N_outer − N_inner) + χ_caps N_caps (n, 3, 3) at the cells:
        what the rotor's own magnetisation adds (the field of the wall, a
        tube over the rotor's length, and of the caps or spacers filling the
        bore beyond the packed sample), or None. A long tube has no field
        in its bore, so the wall acts through its ends."""
        if self.rotor is None:
            return None
        if self._rotor_cells is None:
            n_r, n_z = self._n_cells or (48, 96)
            R, L = 0.5 * self.diameter, self.length
            Ro, Lr = 0.5 * self.rotor["outer_diameter"], self.rotor["length"]
            M = np.zeros((n_r * n_z, 3, 3))
            if self.rotor["chi_wall"] != 0.0:
                M = M + self.rotor["chi_wall"] * _bodies_on_cells(
                    [(Ro, Lr, 0.0, 1.0), (R, Lr, 0.0, -1.0)], R, L, n_r, n_z)
            Lc = 0.5 * (Lr - L)
            if Lc > 0 and self.rotor["chi_caps"] != 0.0:
                zc = 0.5 * (L + Lc)
                M = M + self.rotor["chi_caps"] * _bodies_on_cells(
                    [(R, Lc, zc, 1.0), (R, Lc, -zc, 1.0)], R, L, n_r, n_z)
            self._rotor_cells = 0.5 * (M + np.swapaxes(M, -1, -2))
        return self._rotor_cells

    def macro_tensors(self, chi_v, index=None, psi=None):
        """The macroscopic shift tensors (ppm) at cells ``index`` (all when
        None), turned by ``psi`` about the axis for a cylinder: the
        sample's own, −1e6 χ_v (N − I/3), and the rotor's (wall and caps,
        :meth:`rotor_cells`), −1e6 Σ χ_k N_k."""
        N, _w = self.cells()
        Mr = self.rotor_cells()
        if index is not None:
            N = N[index]
            Mr = None if Mr is None else Mr[index]
        T = float(chi_v) * (N - np.eye(3) / 3.0)
        if Mr is not None:
            T = T + Mr
        if self.axisymmetric and psi is not None:
            T = rotate_about_z(T, psi)
        return -1e6 * T

    def is_uniform(self) -> bool:
        return self.shape == "sphere"


def rotor_sample(outer_diameter_mm, *, inner_diameter=None, length=None, axis_angle=None,
                 rotor_length=None, wall="zirconia", caps="vespel"):
    """A :class:`Sample` for a MAS rotor of ``outer_diameter_mm`` (one of
    :data:`ROTORS`, or any with ``inner_diameter`` and ``length`` given),
    the inner diameter and packed length from :data:`ROTORS` unless
    given. Static: the rotor axis at ``axis_angle`` (default the magic
    angle, a rotor at rest in a MAS probe).

    The rotor's own contribution (explicit user request, 2026-11-07): its
    wall, a tube of the outer and inner diameters over ``rotor_length``
    (default: the packed length plus one outer diameter at each end), and
    the caps or spacers filling the bore beyond the packed sample. ``wall``
    and ``caps``: a :data:`MATERIALS` key or a volume susceptibility (SI);
    None leaves the rotor out."""
    key = min(ROTORS, key=lambda k: abs(k - float(outer_diameter_mm)))
    typical = ROTORS[key] if abs(key - float(outer_diameter_mm)) < 1e-6 else (None, None)
    inner = inner_diameter if inner_diameter is not None else typical[0]
    fill = length if length is not None else typical[1]
    if inner is None or fill is None:
        raise ValueError(f"No typical dimensions for a {outer_diameter_mm} mm rotor: give "
                         "inner_diameter and length.")
    if inner >= float(outer_diameter_mm):
        raise ValueError("The inner diameter must be smaller than the outer one (the wall).")
    rotor = None
    if wall is not None or caps is not None:
        rotor = {"outer_diameter": float(outer_diameter_mm),
                 "length": float(rotor_length) if rotor_length is not None
                 else fill + 2.0 * float(outer_diameter_mm),
                 "chi_wall": material_chi(wall), "chi_caps": material_chi(caps)}
    return Sample("cylinder", diameter=inner, length=fill,
                  axis_angle=MAGIC_ANGLE_DEG if axis_angle is None else axis_angle, rotor=rotor)


def material_chi(material) -> float:
    """χ (SI) of a :data:`MATERIALS` key, a number, or None (0)."""
    if material is None:
        return 0.0
    if isinstance(material, str):
        if material not in MATERIALS:
            raise ValueError(f"Unknown material {material!r}: one of {', '.join(MATERIALS)}, "
                             "or give its volume susceptibility (SI).")
        return MATERIALS[material][0]
    return float(material)


def sample_from_dict(d):
    if d is None:
        return Sample("sphere")
    if isinstance(d, Sample):
        return d
    d = dict(d)
    shape = d.pop("shape", "sphere")
    return Sample(shape, **d)
