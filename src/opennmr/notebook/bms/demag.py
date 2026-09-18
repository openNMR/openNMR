"""
Ported from demag_calc.m.

Computes N_ij(r) = IFFT[ (k_i k_j / |k|^2) * FFT(shape(r)) ] via the
standard Fourier/dipolar-kernel trick. Note this kernel has NO "-1/3"
correction baked in (that correction is applied later, conditionally,
in shift.shielding_bms) -- this function returns the raw demagnetizing
tensor field on the same box used for the shape function.

The k-space grid/kernel (KX, KY, KZ, knsq below) depends only on the
simulation box (sim_str), not on any individual particle, so it is
built once via build_kspace_kernel() and reused for every particle --
previously it was silently rebuilt from scratch inside demag_calc() on
every single call, which is wasted work when looping over many
particles on the same box.
"""
import numpy as np
from .geometry import shape_function


def build_kspace_kernel(sim_str):
    """Build the box-dependent (not particle-dependent) real-space and
    k-space grids once. Returns a dict to pass into demag_calc()."""
    sizex, sizey, sizez = sim_str["sizex"], sim_str["sizey"], sim_str["sizez"]
    n = sim_str["gridpoints"]

    stepx, stepy, stepz = sizex / n, sizey / n, sizez / n

    # Built as n evenly spaced points starting at -size/2 (matches MATLAB's
    # `-(size/2):step:(size/2-step)`, which is exactly n points when
    # step = size/n). Using arange for this is fragile to float round-off
    # (can silently drop/add a point), so we build it explicitly by index.
    xvec = -sizex / 2 + stepx * np.arange(n)
    yvec = -sizey / 2 + stepy * np.arange(n)
    zvec = -sizez / 2 + stepz * np.arange(n)

    X, Y, Z = np.meshgrid(xvec, yvec, zvec, indexing="ij")

    kvecx = np.fft.fftshift(np.fft.fftfreq(len(xvec), d=stepx))
    kvecy = np.fft.fftshift(np.fft.fftfreq(len(yvec), d=stepy))
    kvecz = np.fft.fftshift(np.fft.fftfreq(len(zvec), d=stepz))
    KX, KY, KZ = np.meshgrid(kvecx, kvecy, kvecz, indexing="ij")

    knsq = KX**2 + KY**2 + KZ**2
    knsq[knsq == 0] = np.finfo(float).eps

    return {
        "xvec": xvec, "yvec": yvec, "zvec": zvec,
        "X": X, "Y": Y, "Z": Z,
        "KX": KX, "KY": KY, "KZ": KZ, "knsq": knsq,
    }


def demag_calc(sim_str, sim_part, kernel=None):
    """kernel: optional, from build_kspace_kernel(sim_str) -- pass it in
    when calling this repeatedly for many particles on the same box, to
    avoid rebuilding the k-space grid every time. Built automatically
    (at a small one-off cost) if not supplied, so this remains backwards
    compatible with a single-particle call."""
    if kernel is None:
        kernel = build_kspace_kernel(sim_str)

    X, Y, Z = kernel["X"], kernel["Y"], kernel["Z"]
    KX, KY, KZ, knsq = kernel["KX"], kernel["KY"], kernel["KZ"], kernel["knsq"]
    xvec, yvec, zvec = kernel["xvec"], kernel["yvec"], kernel["zvec"]

    mat = shape_function(sim_part, X, Y, Z)

    ft = np.fft.fftshift(np.fft.fftn(mat))

    def comp(ka, kb):
        # Only the real part is ever used downstream (evaluate() takes
        # .real of these components before interpolating), so drop the
        # imaginary part immediately rather than carrying complex128
        # through the rest of the pipeline -- halves the memory/bandwidth
        # cost of everything from here on.
        return np.fft.ifftn(np.fft.ifftshift(ft * (ka * kb) / knsq)).real

    demag = np.zeros(KX.shape + (3, 3), dtype=float)
    demag[..., 0, 0] = comp(KX, KX)
    demag[..., 0, 1] = comp(KX, KY)
    demag[..., 0, 2] = comp(KX, KZ)
    demag[..., 1, 1] = comp(KY, KY)
    demag[..., 1, 2] = comp(KY, KZ)
    demag[..., 2, 2] = comp(KZ, KZ)
    demag[..., 1, 0] = demag[..., 0, 1]
    demag[..., 2, 0] = demag[..., 0, 2]
    demag[..., 2, 1] = demag[..., 1, 2]

    return demag, (xvec, yvec, zvec)
