"""Ported from point_dipole.m, shielding_bms.m, tensor2haeberlen.m."""
import numpy as np


def point_dipole(x, y, z):
    """Vectorized: x,y,z can be arrays. Returns array of shape (...,3,3)."""
    x, y, z = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float), np.asarray(z, float))
    rsq = x**2 + y**2 + z**2
    pref = 1.0 / (4 * np.pi * rsq**2.5)
    T = np.zeros(x.shape + (3, 3))
    T[..., 0, 0] = -(3 * x * x - rsq) * pref
    T[..., 0, 1] = -(3 * x * y) * pref
    T[..., 0, 2] = -(3 * x * z) * pref
    T[..., 1, 1] = -(3 * y * y - rsq) * pref
    T[..., 1, 2] = -(3 * y * z) * pref
    T[..., 2, 2] = -(3 * z * z - rsq) * pref
    T[..., 1, 0] = T[..., 0, 1]
    T[..., 2, 0] = T[..., 0, 2]
    T[..., 2, 1] = T[..., 1, 2]
    return T


def shielding_bms(sim_part, demag_tensor):
    """demag_tensor: (...,3,3) real array (already real-part-extracted).
    Vectorized version of shielding_bms.m: applies the -1/3 Lorentz-sphere
    correction only where trace(demag) > sim_part['thresh'] (i.e. "inside"
    the particle core, per the shape function's own value -- see note in
    demag.py: trace(N) reconstructs the shape function c0 itself)."""
    chi = sim_part["Chi_tensor"]
    trace = np.trace(demag_tensor, axis1=-2, axis2=-1)
    inside = trace > sim_part["thresh"]

    eye3 = np.eye(3)
    # demag_tensor is already built symmetric (demag.py fills the lower
    # triangle from the upper one), so the MATLAB `demag_tensor'` transpose
    # this used to reproduce is a no-op here -- removed rather than paying
    # for a swapaxes on every particle for nothing.
    shift_inside = 1e6 * np.einsum("ij,...jk->...ik", chi, demag_tensor - eye3 / 3)
    shift_outside = 1e6 * np.einsum("ij,...jk->...ik", chi, demag_tensor)

    shift = np.where(inside[..., None, None], shift_inside, shift_outside)
    return shift


def tensor2haeberlen(tensor, eta_rel_tol=1e-8):
    """tensor: (...,3,3) symmetric. Returns (..., 3) = [iso, delta, eta].
    NOTE: reproduces the MATLAB's exact sign/ordering convention, which is
    a specific (non-canonical-Haeberlen-textbook) choice -- see caveat in
    the accompanying writeup. Kept as-is so numbers match the original tool.

    eta_rel_tol: for (near-)isotropic voxels, delta -> 0 and eta =
    (e2-e1)/delta is a 0/0 (or numerically-unstable-nonzero/near-zero)
    division; this previously produced NaN/inf/huge-garbage values
    silently (confirmed via `RuntimeWarning: invalid value encountered in
    divide` when profiling far-field/vacuum voxels). eta is set to 0
    wherever |delta| is smaller than eta_rel_tol times that voxel's own
    eigenvalue scale, rather than propagating whatever an unstable
    division happens to produce. eta is conventionally undefined (not
    just zero) in this limit -- if your downstream analysis needs to
    distinguish "isotropic" from "eta undefined", filter on delta directly
    rather than trusting eta at very small delta."""
    tensor2 = -tensor
    e = np.linalg.eigvalsh(tensor2)  # ascending, unlike MATLAB's unordered eig
    iso = e.mean(axis=-1)

    dev = np.abs(iso[..., None] - e)
    order = np.argsort(-dev, axis=-1)  # descending |e - iso|, ties broken like MATLAB's stable sort as best effort
    e_sorted = np.take_along_axis(e, order, axis=-1)

    delta = e_sorted[..., 0] - iso
    scale = np.max(np.abs(e), axis=-1)
    tol = eta_rel_tol * np.maximum(scale, np.finfo(float).tiny)
    near_isotropic = np.abs(delta) < tol

    with np.errstate(invalid="ignore", divide="ignore"):
        eta = (e_sorted[..., 2] - e_sorted[..., 1]) / delta
    eta = np.where(near_isotropic, 0.0, eta)
    return np.stack([iso, delta, eta], axis=-1)
