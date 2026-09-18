"""
Ported from evaluate.m and main.m.

evaluate.m used MATLAB's `parfor` over every evaluation point, branching
per-point between (a) trilinear interpolation into the FFT-computed demag
grid, for points inside the box, or (b) a point-dipole far-field formula,
for points outside it (evaluate.m's `condition`/`vec2`/`vec3` split).

Point-dipole is only ever used for genuinely out-of-box points; if you
run demag box and evalpoints over the same domain (as BMS_input.json
does), that branch never fires. Because of that, `evaluate` here is
`fully vectorized` (RegularGridInterpolator handles the whole point
cloud in one call) with a fallback to point_dipole for any points that
land outside the interpolation grid -- no explicit per-point Python loop,
which is both simpler and much faster than a translated parfor would be.

Also fixes evaluate.m's bug where yvec/zvec were built with `stepx`
instead of `stepy`/`stepz` (silently harmless only because the box was
always a cube in the original setup).
"""
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from .shift import point_dipole, shielding_bms, tensor2haeberlen


def evaluate(sim_str, sim_part, demag_tensor, grid_vecs, compute_haeberlen=False):
    """compute_haeberlen: main.run()'s per-particle loop only ever needs
    demag_tensor/shift_tensor (it sums those across particles and does the
    Haeberlen decomposition once, at the end, on the summed tensor) --
    haeb_shift_tensor for an individual particle's own contribution was
    computed here unconditionally and then simply discarded by every
    caller. That's a full eigendecomposition over every evaluation point
    (profiled at ~2.3s/particle at a 128x128x128 grid) for a result nobody
    used. Default is now off; pass compute_haeberlen=True if you're calling
    evaluate() directly and actually want a single particle's own Haeberlen
    values."""
    xvec, yvec, zvec = grid_vecs
    evalpoints = sim_str["evalpoints"]
    if evalpoints.shape[1] != 3:
        raise ValueError("evalpoints must be an (N,3) array")

    vec = evalpoints - sim_part["posc"]

    sizex, sizey, sizez = sim_str["sizex"], sim_str["sizey"], sim_str["sizez"]
    n = sim_str["gridpoints"]
    stepx, stepy, stepz = sizex / n, sizey / n, sizez / n

    outside = ((np.abs(vec[:, 0]) > sizex / 2 - stepx) |
               (np.abs(vec[:, 1]) > sizey / 2 - stepy) |
               (np.abs(vec[:, 2]) > sizez / 2 - stepz))
    inside_idx = np.where(~outside)[0]
    outside_idx = np.where(outside)[0]

    # flatten the 6 independent tensor components for one interpolator call
    values = np.stack([
        demag_tensor[..., 0, 0], demag_tensor[..., 0, 1], demag_tensor[..., 0, 2],
        demag_tensor[..., 1, 1], demag_tensor[..., 1, 2], demag_tensor[..., 2, 2],
    ], axis=-1).real

    interp = RegularGridInterpolator((xvec, yvec, zvec), values,
                                      method="linear", bounds_error=False, fill_value=None)

    corner = np.array([[-sizex / 2, -sizey / 2, -sizez / 2]])
    background = interp(corner)[0]  # (6,) -- box-artifact correction, per evaluate.m

    demag_full = np.zeros((evalpoints.shape[0], 3, 3))

    if len(inside_idx):
        vals_in = interp(vec[inside_idx]) - background
        demag_full[inside_idx, 0, 0] = vals_in[:, 0]
        demag_full[inside_idx, 0, 1] = demag_full[inside_idx, 1, 0] = vals_in[:, 1]
        demag_full[inside_idx, 0, 2] = demag_full[inside_idx, 2, 0] = vals_in[:, 2]
        demag_full[inside_idx, 1, 1] = vals_in[:, 3]
        demag_full[inside_idx, 1, 2] = demag_full[inside_idx, 2, 1] = vals_in[:, 4]
        demag_full[inside_idx, 2, 2] = vals_in[:, 5]

    if len(outside_idx):
        v = vec[outside_idx]
        demag_full[outside_idx] = sim_part["vol"] * point_dipole(v[:, 0], v[:, 1], v[:, 2])

    shift_tensor = shielding_bms(sim_part, demag_full)

    result = {
        "pos": evalpoints,
        "demag_tensor": demag_full,
        "shift_tensor": shift_tensor,
    }
    if compute_haeberlen:
        result["haeb_shift_tensor"] = tensor2haeberlen(shift_tensor)
    return result
