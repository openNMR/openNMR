"""
Turns the per-voxel (d_iso, delta, eta) table from bms.main.run() into the
requested output dict:
    x_ppm, x_Hz, real_spectrum, normalised_real_spectrum,
    imag_spectrum, normalised_imag_spectrum, nuclide, BF1, SFO1, path

ASSUMPTIONS baked in here (confirmed with you, but flagging again since
none of this exists in the original MATLAB):
  - MAS with sidebands: each voxel's (delta,eta) is treated as if it were
    a powder-averaged CSA-like site (see mas_sidebands.py for why that's
    the right treatment even though we only have eigenvalues, not
    eigenvectors, per voxel).
  - ppm/Hz referencing: Hz = ppm * BF1 (BF1 given in MHz, Bruker-style),
    i.e. ppm = Hz / BF1 -- exactly the "standard ppm=freq/BF1" you asked
    for. SFO1 is carried through as metadata only, NOT used to offset
    the axis (tell me if you actually want SFO1 as the ppm-zero reference
    instead of a bare BF1 scaling -- that's a one-line change here).
  - Every voxel is weighted equally (one unit of "spin population" per
    grid point). If you want voxels weighted by e.g. (1 - shape function)
    (matrix material only, excluding solid particle interior) rather than
    uniformly, that's a filter to add before histogramming -- currently
    ALL evalpoints (inside and outside particles) contribute equally,
    which is almost certainly not what you want physically and is the
    first thing I'd revisit with you.
  - Lineshape: Lorentzian, fixed linewidth in Hz, built via the standard
    histogram -> IFFT -> apply T2 decay -> FFT trick (efficient: avoids
    an explicit Lorentzian-per-line sum over potentially millions of
    voxels x sideband orders).
"""
import numpy as np
from pathlib import Path
from .mas_sidebands import sideband_manifold


def build_manifold_table(mu_values, eta_values, max_order, n_t=64, n_beta=14, n_gamma=14):
    """Precompute sideband intensities on a (mu, eta) grid, once, so
    per-voxel lookups are cheap. Returns (mu_values, eta_values,
    intensity_table) where intensity_table has shape
    (len(mu_values), len(eta_values), 2*max_order+1)."""
    orders_ref = np.arange(-max_order, max_order + 1)
    table = np.zeros((len(mu_values), len(eta_values), len(orders_ref)))
    for i, mu in enumerate(mu_values):
        for j, eta in enumerate(eta_values):
            orders, intens = sideband_manifold(delta=mu, eta=eta, spin_rate=1.0,
                                                n_orient_beta=n_beta, n_orient_gamma=n_gamma,
                                                n_t=n_t, max_order=max_order)
            for o, val in zip(orders, intens):
                table[i, j, o + max_order] = val
    return mu_values, eta_values, orders_ref, table


def _lookup_bilinear(mu, eta, mu_grid, eta_grid, table):
    """table: (Nmu, Neta, Norders). mu, eta: (N,) arrays of per-voxel values,
    clipped to the precomputed grid range. Returns (N, Norders)."""
    mu_c = np.clip(mu, mu_grid[0], mu_grid[-1])
    eta_c = np.clip(np.abs(eta), eta_grid[0], eta_grid[-1])  # manifold symmetric in sign of eta by construction
    i = np.clip(np.searchsorted(mu_grid, mu_c) - 1, 0, len(mu_grid) - 2)
    j = np.clip(np.searchsorted(eta_grid, eta_c) - 1, 0, len(eta_grid) - 2)
    fi = (mu_c - mu_grid[i]) / (mu_grid[i + 1] - mu_grid[i])
    fj = (eta_c - eta_grid[j]) / (eta_grid[j + 1] - eta_grid[j])
    fi, fj = fi[:, None], fj[:, None]
    out = (table[i, j] * (1 - fi) * (1 - fj) + table[i + 1, j] * fi * (1 - fj)
           + table[i, j + 1] * (1 - fi) * fj + table[i + 1, j + 1] * fi * fj)
    return out


def assemble_spectrum(df, spin_rate_hz, BF1, SFO1, nuclide, path,
                       n_points=16384, sw_ppm=200.0, linewidth_hz=50.0,
                       max_order=8, mu_grid=None, eta_grid=None):
    """
    df: DataFrame with columns d_iso, delta, eta (ppm), as returned by
        bms.main.run().
    spin_rate_hz: MAS rate in Hz.
    BF1: reference frequency in MHz (Bruker convention).
    SFO1: carried through as metadata only (see module docstring).
    sw_ppm: total ppm window of the output spectrum, centered on 0 ppm --
        change the centering here if your d_iso values aren't already
        referenced the way you want the spectrum centered.
    """
    if mu_grid is None:
        mu_grid = np.concatenate([[0.0], np.geomspace(0.02, 50, 24)])
    if eta_grid is None:
        eta_grid = np.linspace(0, 1, 11)

    mu_grid, eta_grid, orders_ref, table = build_manifold_table(mu_grid, eta_grid, max_order)

    spin_rate_ppm = spin_rate_hz / BF1  # Hz / MHz = ppm

    d_iso = df["d_iso"].to_numpy()
    delta = df["delta"].to_numpy()
    eta = df["eta"].to_numpy()
    mu = np.abs(delta) / spin_rate_ppm

    intens = _lookup_bilinear(mu, eta, mu_grid, eta_grid, table)  # (N, Norders)

    # every (voxel, sideband order) pair becomes one stick at this ppm position
    stick_ppm = d_iso[:, None] + orders_ref[None, :] * spin_rate_ppm  # (N, Norders)
    stick_weight = intens  # equal per-voxel population weight of 1, folded in here

    sw_hz = sw_ppm * BF1
    freq_axis = np.linspace(-sw_hz / 2, sw_hz / 2, n_points, endpoint=False)
    stick_hz = stick_ppm.ravel() * BF1

    hist, edges = np.histogram(stick_hz, bins=n_points, range=(-sw_hz / 2, sw_hz / 2),
                                weights=stick_weight.ravel())

    # histogram -> time domain -> apply T2 decay (Lorentzian broadening) -> back to frequency domain
    fid = np.fft.ifft(np.fft.ifftshift(hist.astype(complex)))
    dt = 1.0 / sw_hz
    t = np.arange(n_points) * dt
    fid *= np.exp(-np.pi * linewidth_hz * t)
    spectrum = np.fft.fftshift(np.fft.fft(fid))

    x_hz = freq_axis
    x_ppm = x_hz / BF1
    real_spectrum = spectrum.real
    imag_spectrum = spectrum.imag
    real_max = np.max(np.abs(real_spectrum)) or 1.0
    imag_max = np.max(np.abs(imag_spectrum)) or 1.0

    return {
        "x_ppm": x_ppm,
        "x_Hz": x_hz,
        "real_spectrum": real_spectrum,
        "normalised_real_spectrum": real_spectrum / real_max,
        "imag_spectrum": imag_spectrum,
        "normalised_imag_spectrum": imag_spectrum / imag_max,
        "nuclide": nuclide,
        "BF1": float(BF1),
        "SFO1": float(SFO1),
        "path": Path(path),
    }
