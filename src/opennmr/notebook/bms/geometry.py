"""
Particle-shape functions, ported from ellipsoid_mask.m / hexagonal_mask.m / donut_mask.m.

The MATLAB originals each re-derived the same quaternion->rotation-matrix
code inline; that's pulled out once here (quat_to_rotmat) since there's no
reason for three copies to drift independently.

NOTE on convention -- READ BEFORE TRUSTING ORIENTATION-DEPENDENT RESULTS:
The MATLAB code normalises the quaternion and then NEGATES the vector
part (qi,qj,qk) before building R (i.e. conjugates it, which for a unit
quaternion is the inverse rotation). Separately, _to_local_frame below
computes `pts @ R`, not `R @ pts`. I checked numerically
(scripts/verify_rotation_convention.py) what these two steps do *together*:

    pts @ R  ==  R_std(q) @ pt   for each point pt

i.e. applying R.T (from `pts @ R`) to a matrix that is itself already
R_std(q)^-1 = R_std(q).T exactly cancels the conjugation, and the net
effect is identical to simply building the rotation matrix from the
PLAIN (non-conjugated) quaternion and applying it directly, with no
transpose anywhere. That is NOT the same thing as the "R maps world ->
particle-local frame directly" claim the previous version of this
docstring made -- that claim was wrong about the mechanism, even though
(as far as I can tell) it does not change any numerical output, since the
conjugate-then-transpose cancellation was already happening before this
comment was corrected.

This module has deliberately NOT been rewritten to remove the
conjugate+transpose dance, because I have no independent ground truth
(e.g. a validated MATLAB reference run) to confirm the net convention is
the physically-intended one -- only that it's self-consistent between
quat_to_rotmat() and _to_local_frame() as they stand. Whether "R_std(q)
applied directly to world points" is the orientation convention you
actually want depends on how the quaternion in your particle dump is
defined (body->world vs world->body), which is a property of your
upstream data generation, not of this code. Recommended before trusting
any orientation-sensitive result: run scripts/verify_rotation_convention.py
against a particle with a known, non-trivial orientation (e.g. a
needle-shaped ellipsoid tilted 90 deg about a known axis) and confirm the
mask's long axis ends up where you expect in world coordinates.
"""
import numpy as np


def quat_to_rotmat(q):
    """q = [qr, qi, qj, qk] (w,x,y,z). Returns the SAME R as the MATLAB
    code (built from the conjugated/negated vector part), such that
    R @ pts_world gives particle-local coordinates directly (matching
    `(R' * pts')'` in the MATLAB, since here R already IS that transpose)."""
    q = np.asarray(q, dtype=float)
    q = q / np.linalg.norm(q)
    qr, qi, qj, qk = q[0], -q[1], -q[2], -q[3]
    R = np.array([
        [1 - 2*(qj**2 + qk**2),   2*(qi*qj - qk*qr),     2*(qi*qk + qj*qr)],
        [2*(qi*qj + qk*qr),       1 - 2*(qi**2 + qk**2), 2*(qj*qk - qi*qr)],
        [2*(qi*qk - qj*qr),       2*(qj*qk + qi*qr),     1 - 2*(qi**2 + qj**2)],
    ])
    return R


def _to_local_frame(sim_part, x, y, z):
    """x,y,z: flat arrays of world coords -> (xt,yt,zt) in particle frame."""
    pts = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    R = quat_to_rotmat(sim_part["q"])
    # MATLAB: pts_rot = (R' * pts')' -> pts @ R
    pts_rot = pts @ R
    return pts_rot[:, 0], pts_rot[:, 1], pts_rot[:, 2]


def ellipsoid_mask(sim_part, x, y, z):
    shape = x.shape
    xt, yt, zt = _to_local_frame(sim_part, x, y, z)
    r, t, sl = sim_part["r"], sim_part["t"], sim_part["sl"]
    v1, v2, v3 = sim_part["v1"], sim_part["v2"], sim_part["v3"]
    # A steep slope_filter (sl, typically ~2000) intentionally sends the
    # base to +inf for points even slightly outside the particle, which
    # numpy reports as an "overflow in power" RuntimeWarning -- harmless,
    # since 1/(1+inf) correctly evaluates to 0, but silenced deliberately
    # here so it doesn't look like an error to someone running this fresh.
    with np.errstate(over="ignore"):
        c0 = 1.0 / (1.0 + (((xt / v1) ** r + (yt / v2) ** r) ** (t / r)
                           + (zt / v3) ** t) ** sl)
    return c0.reshape(shape)


def hexagonal_mask(sim_part, x, y, z):
    shape = x.shape
    xt, yt, zt = _to_local_frame(sim_part, x, y, z)
    Rhex, h, p = sim_part["v1"], sim_part["v3"], sim_part["sl"]
    deform = sim_part["Deformation"]

    def hex_dist(a, b):
        return np.maximum.reduce([np.abs(a),
                                   np.abs(0.5 * a + np.sqrt(3) / 2 * b),
                                   np.abs(0.5 * a - np.sqrt(3) / 2 * b)]) / Rhex

    if deform == "C":
        d_hex = hex_dist(xt, yt)
        d_z = np.abs(zt) / (h / 2)
    elif deform == "B":
        d_hex = hex_dist(xt, zt)
        d_z = np.abs(yt) / (h / 2)
    else:
        d_hex = hex_dist(yt, zt)
        d_z = np.abs(xt) / (h / 2)

    with np.errstate(over="ignore"):  # see note in ellipsoid_mask
        c0 = 1.0 / (1.0 + (d_hex ** p + d_z ** p))
    return c0.reshape(shape)


def donut_mask(sim_part, x, y, z):
    shape = x.shape
    xt, yt, zt = _to_local_frame(sim_part, x, y, z)
    R0, r0, p = sim_part["v1"], sim_part["v2"], sim_part["sl"]
    deform = sim_part["Deformation"]

    if deform == "C":
        rho = np.sqrt(xt ** 2 + yt ** 2)
        d_torus = np.sqrt((rho - R0) ** 2 + zt ** 2) / r0
    elif deform == "B":
        rho = np.sqrt(zt ** 2 + xt ** 2)
        d_torus = np.sqrt((rho - R0) ** 2 + yt ** 2) / r0
    else:
        rho = np.sqrt(zt ** 2 + yt ** 2)
        d_torus = np.sqrt((rho - R0) ** 2 + xt ** 2) / r0

    with np.errstate(over="ignore"):  # see note in ellipsoid_mask
        c0 = 1.0 / (1.0 + d_torus ** p)
    return c0.reshape(shape)


def cuboid_mask(sim_part, x, y, z):
    """A rectangular block (cube when v1 = v2 = v3): half side lengths
    v1, v2, v3 along the particle's x, y, z, with the same soft edge (the
    slope filter) as the other shapes. Explicit user request, 2026-11-10:
    cubic and cuboidal particles."""
    shape = x.shape
    xt, yt, zt = _to_local_frame(sim_part, x, y, z)
    p = sim_part["sl"]
    dx = np.abs(xt) / sim_part["v1"]
    dy = np.abs(yt) / sim_part["v2"]
    dz = np.abs(zt) / sim_part["v3"]
    with np.errstate(over="ignore"):  # see note in ellipsoid_mask
        c0 = 1.0 / (1.0 + (dx ** p + dy ** p + dz ** p))
    return c0.reshape(shape)


def shape_function(sim_part, x, y, z):
    eq = sim_part["Equation"]
    if eq == "cuboid":
        return cuboid_mask(sim_part, x, y, z)
    if eq == "hexagonal":
        return hexagonal_mask(sim_part, x, y, z)
    elif eq == "donut":
        return donut_mask(sim_part, x, y, z)
    else:
        return ellipsoid_mask(sim_part, x, y, z)
