"""
Simulate static and finite-speed-MAS powder NMR lineshapes for the
anisotropic (CSA) and quadrupolar interactions.

Workflow
--------

    CSA, static     -> csa_definition -> make_static_csa_spectrum
    CSA, spinning    -> csa_definition -> finite_mas_csa_spectrum
    Quadrupolar, static  -> quadrupolar_static_spectrum
    Quadrupolar, MAS     -> quadrupolar_mas_spectrum
        (either: order='first' | 'second' | 'exact',
         transitions='ct' | 'st' | 'all'; see the QUADRUPOLAR NMR
         section, and quadrupolar_order_comparison to see what each
         order contributes)

All of the ``*_spectrum*`` functions return a dict containing at least
``x_ppm``, ``real_spec`` and ``normalised_real_spec``, so they can be
plotted directly, e.g.::

    spec = make_static_csa_spectrum(csa_definition(100, 500, 0.5,
                                                     convention='Herzfeld-Berger'),
                                     np.linspace(-1000, 1000, 5001))
    plt.plot(spec['x_ppm'], spec['real_spec'])
"""

from math import comb

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import fftconvolve

from .utils import find_isotope_info


# ============================================================
# CHANGELOG (relative to the version you posted)
# ------------------------------------------------------------
# BUG FIXES
#   1. quadrupolar_hamiltonian: prefactor denominator was 6*I*(2I-1);
#      correct value is 4*I*(2I-1). (mas_quadrupolar_fid already had
#      the correct 4.0 -- the two were inconsistent with each other.)
#   2. finite_mas_csa_spectrum: the mean-frequency de-trending line
#      subtracted a term in radians from a phase that is in cycles
#      (an extra stray factor of 2*pi). Fixed. NOTE: for rotor_angle
#      != 'magic' this term is not just numerical hygiene -- off-MAS
#      spinning leaves a genuine residual CSA shift, and subtracting
#      it will erase real physics. Left as an explicit toggle.
#   3. herzfeld_berger_csa_tensor: skew term had the opposite sign to
#      the standard Herzfeld-Berger definition
#      kappa = 3*(d_iso - d_22)/Omega. Fixed.
#   4. haeberlen_csa_tensor: eta was implemented as (d_xx-d_yy)/Delta,
#      i.e. the negative of the standard eta = (d_yy-d_xx)/Delta. This
#      does not change any spectrum computed by full powder averaging
#      here (swapping which axis is called x vs y is invisible to a
#      quadratic form integrated over the full sphere), but it will
#      matter the moment axis identity is used elsewhere (dipolar/CSA
#      cross terms, MQMAS crystal-frame relations, etc.), so fixed for
#      correctness.
#
# EFFICIENCY
#   - Quadrupolar spin operators (Ix,Iy,Iz) and the (orientation-
#     independent) quadrupolar Hamiltonian H_Q are now built once per
#     (I, C_Q, eta_Q) and passed into the per-crystallite loop, instead
#     of being rebuilt from scratch (matrix products + eye()) at every
#     one of up to ~45,000 orientations.
#   - static_csa / make_static_csa_spectrum: the CSA resonance
#     frequency is a quadratic form n^T D n with diagonal D, so it is
#     invariant under independent sign flips of any direction-cosine
#     component. That means the powder average only needs one octant
#     of the sphere (exact, not approximate) -- an 8x reduction in
#     points for identical accuracy. NOT applied to the quadrupolar
#     Hamiltonian (H_Z is linear, not quadratic, in the direction
#     cosines) or to finite_mas_csa_spectrum (gamma-dependent rotation
#     makes the symmetry argument non-trivial and I have not verified
#     it, so I left that grid as a plain rectangular grid, just much
#     smaller by default -- see note there).
#   - mas_quadrupolar_fid: replaced per-timestep scipy.linalg.expm
#     (Pade approximant, general matrix) with an eigendecomposition-
#     based propagator (H is Hermitian at every step), which is both
#     cheaper and removes the import-inside-loop overhead.
#   - intensity_threshold defaults unified to 1e-10 across functions.
#
# EFFICIENCY (continued)
#   - ZCW (Zaremba-Conroy-Wolfsberg) powder averaging is now available
#     as an alternative to the rectangular (theta,phi) product grid,
#     which oversamples near the poles and needs far more points than
#     an optimised angle set for the same convergence. See zcw_angles()
#     and get_powder_grid(); every *_spectrum* function below now takes
#     powder_scheme='rectangular' (default, unchanged behaviour) or
#     'zcw' (+ n_zcw). Biggest win is finite_mas_csa_spectrum, where
#     each orientation costs a full FFT and is called ~100x per fit via
#     curve_fit's Jacobian -- try n_zcw in the low thousands there and
#     compare against a converged rectangular grid. use_octant_symmetry
#     is a rectangular-grid-only trick and is not combined with ZCW
#     here; SOPHE/REPULSION sets are still not implemented.
#
# QUADRUPOLAR (2026-09-25): the per-crystallite loops and the
# single-crystallite MAS propagator were replaced by one vectorised
# engine (static + MAS, first/second order/exact, CT and/or STs) --
# see the QUADRUPOLAR NMR section's own header.
# ============================================================


# ============================================================
# CSA NMR
# ============================================================

def haeberlen_csa_tensor(d_iso, Delta, eta):
    """
    Principal values of the CSA tensor in the Haeberlen convention.

    Delta = d_zz - d_iso   (by definition/ordering)
    eta   = (d_yy - d_xx) / Delta
    """
    d_zz = Delta + d_iso
    # d_xx + d_yy = 3*d_iso - d_zz
    # d_yy - d_xx = eta * Delta
    d_yy = ((3.0 * d_iso - d_zz) + eta * Delta) / 2.0
    d_xx = ((3.0 * d_iso - d_zz) - eta * Delta) / 2.0
    csa_principal = np.array([d_xx, d_yy, d_zz])
    return np.diag(csa_principal)


def herzfeld_berger_csa_tensor(d_iso, span, skew):
    """
    Principal values of the CSA tensor in the Herzfeld-Berger convention.

    span (Omega) = d_zz - d_yy      (d_zz >= d_xx >= d_yy)
    skew (kappa) = 3*(d_iso - d_xx) / span
    """
    # d_xx (the "middle" element, d_22) from kappa:
    d_xx = d_iso - (span * skew) / 3.0
    # d_zz + d_yy = 3*d_iso - d_xx ; d_zz - d_yy = span
    d_zz = ((2.0 * d_iso) + ((span * skew) / 3.0) + span) / 2.0
    d_yy = d_zz - span
    csa_principal = np.array([d_xx, d_yy, d_zz])
    return np.diag(csa_principal)


def csa_axis_limits(d_iso, span_or_Delta, factor=1.5, fwhm=0.0):
    """ppm axis limits ``(low, high)`` that frame a CSA powder pattern:
    ``d_iso +/- factor * |span_or_Delta|``.

    Works for either convention: a Haeberlen pattern runs from
    ``d_iso + Delta`` to ``d_iso - Delta(1 + eta)/2`` and a Herzfeld-Berger
    one lies within ``d_iso +/- 2*span/3``, so 1.5 x either leaves margin
    on both sides. The half-width never drops below five broadening
    FWHMs (or 1 ppm), so a near-zero anisotropy still frames its line.
    """
    half = max(factor * abs(float(span_or_Delta)), 5.0 * abs(float(fwhm)), 1.0)
    return float(d_iso) - half, float(d_iso) + half


def csa_definition(d_iso, span_or_Delta, skew_or_eta, convention='Haeberlen'):
    if convention == 'Haeberlen':
        return haeberlen_csa_tensor(d_iso, span_or_Delta, skew_or_eta)
    elif convention == 'Herzfeld-Berger':
        return herzfeld_berger_csa_tensor(d_iso, span_or_Delta, skew_or_eta)
    else:
        raise ValueError("Please define convention as either 'Haeberlen' or 'Herzfeld-Berger'")


def csa_frequency(csa, theta, phi):
    """
    Resonance chemical shift for crystallite orientations.
    theta, phi are arrays in radians.
    Returns an array of chemical shifts in ppm.
    """
    d = csa
    n = np.column_stack([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta),
    ])
    return np.einsum("ni,ij,nj->n", n, d, n)


def powder_orientations(n_theta=1500, n_phi=3000):
    """
    Uniform spherical powder grid over the full sphere.
    Uniform in cos(theta), so every point carries equal weight.
    """
    u_extrema = np.linspace(-1.0, 1.0, n_theta + 1)
    u_mid = 0.5 * (u_extrema[:-1] + u_extrema[1:])
    theta = np.arccos(u_mid)
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    theta, phi = np.meshgrid(theta, phi, indexing="ij")
    theta = theta.ravel()
    phi = phi.ravel()
    weights = np.full(theta.shape, 1.0 / theta.size)
    return theta, phi, weights


def powder_orientations_octant(n_theta=1500, n_phi=3000):
    """
    Uniform powder grid restricted to one octant: theta in [0, pi/2],
    phi in [0, pi/2].

    VALID ONLY for observables depending on the crystallite orientation
    through squares of the direction-cosine components (n_i^2), e.g. a
    quadratic form n^T D n with diagonal D -- which is exactly the
    static CSA resonance frequency. Such an observable is invariant
    under an independent sign flip of any single n_i, so averaging
    over one octant with the same per-point weight as the full-sphere
    grid gives the EXACT full-sphere average, using 1/8 the points.

    Do NOT use this for the quadrupolar Hamiltonian: its Zeeman term
    is linear in the direction cosines (H_Z ~ n . I), not quadratic,
    so this symmetry argument does not apply.
    """
    u_extrema = np.linspace(0.0, 1.0, n_theta + 1)
    u_mid = 0.5 * (u_extrema[:-1] + u_extrema[1:])
    theta = np.arccos(u_mid)
    phi = np.linspace(0.0, np.pi / 2.0, n_phi, endpoint=False)
    theta, phi = np.meshgrid(theta, phi, indexing="ij")
    theta = theta.ravel()
    phi = phi.ravel()
    weights = np.full(theta.shape, 1.0 / theta.size)
    return theta, phi, weights


def zcw_angles(n_min):
    """
    Zaremba-Conroy-Wolfsberg (ZCW) powder-orientation set, covering the
    full sphere with a single well-distributed angle set rather than a
    rectangular (theta, phi) product grid.

    Use this in place of ``powder_orientations`` for any observable
    that depends on crystallite orientation only through (theta, phi)
    -- i.e. does not depend on the third (gamma) Euler angle, or where
    gamma-averaging is handled separately (as in
    ``finite_mas_csa_spectrum``'s analytic gamma-COMPUTE approach, or
    the static quadrupolar functions, which have no spinning gamma to
    average at all). It is NOT a drop-in replacement for anything that
    needs a genuine 3-angle (alpha, beta, gamma) set.

    Unlike the rectangular grid, ZCW orientations are not wasted
    oversampling near the poles, so a given accuracy is reached with
    far fewer crystallites -- the more expensive the per-orientation
    cost (a diagonalization, an FFT), the bigger the win. The cost is
    that ZCW sets only exist at a discrete family of sizes (Fibonacci
    numbers), so you get *at least* ``n_min`` orientations, not exactly
    ``n_min``.

    Parameters
    ----------
    n_min : int
        Minimum number of crystallite orientations requested. The
        actual count returned, N, is the smallest Fibonacci number
        >= n_min (with N >= 2).

    Returns
    -------
    theta, phi : ndarray, shape (N,)
        Polar and azimuthal angles in radians.
    weights : ndarray, shape (N,)
        Uniform weights, each 1/N (they sum to 1).

    Notes
    -----
    Formula (Cheng, Suzukawa & Wolfsberg, J. Chem. Phys. 59, 3992
    (1973); as summarised e.g. in Eden, Concepts Magn. Reson. A 18, 24
    (2003)):

        phi_k   = 2*pi * ((k * M) mod N) / N
        theta_k = arccos(1 - (2k+1)/N)
        weight_k = 1/N,   k = 0, ..., N-1

    where M and N are Fibonacci numbers two places apart in the
    sequence (N = F(j), M = F(j-2)) -- e.g. (M, N) = (55, 144),
    (89, 233), (144, 377), .... This choice of M makes the sequence of
    phi_k values a low-discrepancy (quasi-random, golden-ratio-like)
    sweep over [0, 2*pi) that, combined with the theta_k spacing
    (uniform in cos(theta), as required for a solid-angle-correct
    average), gives near-uniform coverage of the sphere for any N in
    the Fibonacci family.
    """
    if n_min < 1:
        raise ValueError("n_min must be >= 1.")
    fib = [1, 1]
    while fib[-1] < n_min or len(fib) < 3:
        fib.append(fib[-1] + fib[-2])
    N = fib[-1]
    M = fib[-3]
    k = np.arange(N)
    phi = 2.0 * np.pi * np.mod(k * M, N) / N
    theta = np.arccos(1.0 - (2.0 * k + 1.0) / N)
    weights = np.full(N, 1.0 / N)
    return theta, phi, weights


def get_powder_grid(n_theta, n_phi, powder_scheme='rectangular', n_zcw=None,
                     use_octant_symmetry=False):
    """
    Dispatch to the requested powder-orientation grid, so every
    ``*_spectrum*`` function below can offer the same
    ``powder_scheme``/``n_zcw`` choice without duplicating the
    if/else.

    Parameters
    ----------
    n_theta, n_phi : int
        Rectangular-grid size, used when ``powder_scheme='rectangular'``
        (and as the default ``n_zcw = n_theta * n_phi`` when
        ``powder_scheme='zcw'`` and ``n_zcw`` is not given explicitly,
        so swapping schemes at fixed n_theta/n_phi is a fair-ish
        like-for-like comparison rather than a silent count change).
    powder_scheme : {'rectangular', 'zcw'}, optional
        'rectangular' (default -- unchanged behaviour): the (theta,
        phi) product grid via ``powder_orientations`` /
        ``powder_orientations_octant``.
        'zcw' : a single ZCW angle set via ``zcw_angles``, sized by
        ``n_zcw``. Typically needs far fewer orientations than the
        rectangular grid for the same convergence -- most worthwhile
        where each orientation is expensive (an eigh call, an FFT):
        ``finite_mas_csa_spectrum``, the quadrupolar exact/decomposed
        functions. For plain static CSA, the rectangular grid with
        ``use_octant_symmetry=True`` is already an exact 8x reduction
        at negligible per-point cost, so ZCW's advantage there is
        smaller (though still real at high resolution).
    n_zcw : int, optional
        Minimum ZCW orientation count (see ``zcw_angles``). Only used
        when ``powder_scheme='zcw'``. Defaults to ``n_theta * n_phi``.
    use_octant_symmetry : bool, optional
        Only consulted when ``powder_scheme='rectangular'``. ZCW
        already spans the full sphere efficiently -- the octant-folding
        trick is specific to the rectangular grid's redundant
        structure and is NOT applied when ``powder_scheme='zcw'``.

    Returns
    -------
    theta, phi, weights : ndarray
    """
    if powder_scheme == 'rectangular':
        if use_octant_symmetry:
            return powder_orientations_octant(n_theta=n_theta, n_phi=n_phi)
        return powder_orientations(n_theta=n_theta, n_phi=n_phi)
    elif powder_scheme == 'zcw':
        if n_zcw is None:
            n_zcw = n_theta * n_phi
        return zcw_angles(n_zcw)
    else:
        raise ValueError("powder_scheme must be 'rectangular' or 'zcw'.")


def static_csa(csa, ppm_axis, n_theta=1500, n_phi=3000, use_octant_symmetry=True,
                powder_scheme='rectangular', n_zcw=None):
    theta, phi, wts = get_powder_grid(
        n_theta, n_phi, powder_scheme=powder_scheme, n_zcw=n_zcw,
        use_octant_symmetry=use_octant_symmetry,
    )
    shifts = csa_frequency(csa, theta, phi)
    spectrum = np.zeros_like(ppm_axis, dtype=float)
    dx = ppm_axis[1] - ppm_axis[0]
    idx = (shifts - ppm_axis[0]) / dx
    i0 = np.floor(idx).astype(int)
    frac = idx - i0
    valid = ((i0 >= 0) & (i0 < len(ppm_axis) - 1))
    np.add.at(spectrum, i0[valid], wts[valid] * (1.0 - frac[valid]))
    np.add.at(spectrum, i0[valid] + 1, wts[valid] * frac[valid])
    return spectrum


def gaussian_broadening(spectrum, ppm_axis, fwhm=2.0):
    if fwhm <= 0:
        print("fwhm set to 0 or less; proceeding without broadening...")
        return spectrum
    dx = abs(ppm_axis[1] - ppm_axis[0])
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    sigma_pts = sigma / dx
    return gaussian_filter1d(spectrum, sigma_pts)


def lorentzian_broadening(spectrum, ppm_axis, fwhm=2.0):
    """
    Convolve `spectrum` with a Lorentzian kernel of the given FWHM (in
    ppm). There is no closed-form recursive filter for a Lorentzian the
    way `scipy.ndimage.gaussian_filter1d` gives one for a Gaussian, so
    this builds an explicit kernel on the same grid as `ppm_axis` (as
    wide as the data itself, to avoid truncating a Lorentzian's slowly-
    decaying tails at realistic fwhm) and convolves via FFT.
    """
    if fwhm <= 0:
        print("fwhm set to 0 or less; proceeding without broadening...")
        return spectrum
    dx = abs(ppm_axis[1] - ppm_axis[0])
    gamma = fwhm / 2.0  # HWHM
    # An odd-length kernel, centred on its middle sample: with an
    # even length, fftconvolve(mode="same") aligns sample (n-1)//2, one
    # before the kernel's peak at n//2, so every line came out one grid
    # point off -- on every 2^n-point axis, i.e. almost all real data.
    # (Found 2026-09-24: a J-coupling fit's centres were all off by
    # exactly one point.)
    n_kernel = len(ppm_axis) if len(ppm_axis) % 2 else len(ppm_axis) + 1
    x_kernel = (np.arange(n_kernel) - n_kernel // 2) * dx
    kernel = gamma / (np.pi * (x_kernel ** 2 + gamma ** 2))
    kernel = kernel / kernel.sum()
    return fftconvolve(spectrum, kernel, mode="same")


def pseudo_voigt_broadening(spectrum, ppm_axis, fwhm=2.0, eta=0.5):
    """
    Broaden `spectrum` with a pseudo-Voigt kernel of the given FWHM
    (shared between the Gaussian and Lorentzian components -- the same
    convention already used for closed-form peaks in
    `nmrtools.fitting.pseudo_voigt`): ``eta * Lorentzian + (1 - eta) *
    Gaussian``, ``eta`` in [0, 1] (0 = pure Gaussian, 1 = pure
    Lorentzian). NOTE: this `eta` is the lineshape-mixing fraction, an
    entirely different quantity from a CSA tensor's Haeberlen
    asymmetry (also conventionally called eta) -- see
    `nmrtools.csa_fitting`'s module docstring.
    """
    eta = float(np.clip(eta, 0.0, 1.0))
    if eta == 0.0:
        return gaussian_broadening(spectrum, ppm_axis, fwhm=fwhm)
    if eta == 1.0:
        return lorentzian_broadening(spectrum, ppm_axis, fwhm=fwhm)
    return (eta * lorentzian_broadening(spectrum, ppm_axis, fwhm=fwhm)
            + (1.0 - eta) * gaussian_broadening(spectrum, ppm_axis, fwhm=fwhm))


def _apply_broadening(spectrum, ppm_axis, fwhm=2.0, lineshape="gaussian", eta=0.0):
    """Dispatch to gaussian_broadening / lorentzian_broadening /
    pseudo_voigt_broadening by name -- shared by
    `make_static_csa_spectrum` and `finite_mas_csa_spectrum`."""
    key = str(lineshape).lower().replace("-", "_")
    if key == "gaussian":
        return gaussian_broadening(spectrum, ppm_axis, fwhm=fwhm)
    if key == "lorentzian":
        return lorentzian_broadening(spectrum, ppm_axis, fwhm=fwhm)
    if key in ("pseudo_voigt", "pv"):
        return pseudo_voigt_broadening(spectrum, ppm_axis, fwhm=fwhm, eta=eta)
    raise ValueError(
        f"Unknown lineshape {lineshape!r}; use 'gaussian', 'lorentzian', "
        f"or 'pseudo_voigt'."
    )


def make_static_csa_spectrum(csa, ppm_axis, fwhm=2.0, n_theta=1500, n_phi=3000,
                              use_octant_symmetry=True, powder_scheme='rectangular',
                              n_zcw=None, lineshape="gaussian", eta=0.0):
    """
    lineshape : {'gaussian', 'lorentzian', 'pseudo_voigt'}, optional
        Broadening kernel convolved onto the simulated powder pattern.
        Default 'gaussian' (unchanged behaviour). 'pseudo_voigt' uses
        `eta` as the Gaussian/Lorentzian mixing fraction -- see
        `pseudo_voigt_broadening`.
    eta : float, optional
        Only used when ``lineshape='pseudo_voigt'``.
    """
    spec_before_broad = static_csa(csa, ppm_axis, n_theta=n_theta, n_phi=n_phi,
                                    use_octant_symmetry=use_octant_symmetry,
                                    powder_scheme=powder_scheme, n_zcw=n_zcw)
    spec_after_broad = _apply_broadening(spec_before_broad, ppm_axis, fwhm=fwhm,
                                          lineshape=lineshape, eta=eta)
    maximum = np.max(spec_after_broad)
    normalised = spec_after_broad / maximum if maximum > 0 else spec_after_broad
    return {
        "x_ppm": ppm_axis,
        "real_spec": spec_after_broad,
        "normalised_real_spec": normalised,
    }


##############

def finite_mas_csa_spectrum(csa, ppm_axis, larmor_Hz, mas_Hz,
                             rotor_angle='magic', n_theta=60, n_phi=120,
                             n_time=4096, n_sidebands=30, fwhm=2.0,
                             subtract_residual_mean=True, orientation_chunk_size=None,
                             powder_scheme='rectangular', n_zcw=None,
                             lineshape="gaussian", eta=0.0):
    """
    lineshape, eta : see `make_static_csa_spectrum` -- same broadening
        options, applied to the simulated sideband pattern below.

    NOTE ON GRID SIZE: each crystallite orientation needs its own
    length-n_time trajectory/FFT, so this cannot use the same
    rectangular-grid point counts as the static CSA case (the original
    defaults of n_theta=1500, n_phi=3000 imply ~4.5e6 FFTs). Defaults
    have been reduced to n_theta=60, n_phi=120 (7200 orientations);
    check convergence for your specific tensor/spinning-rate
    combination by increasing these and confirming sideband
    intensities are stable.

    POWDER SCHEME: this is exactly the function the module-level
    changelog flagged as most in need of ZCW (each orientation costs a
    full FFT, and this gets called ~100x per fit via curve_fit's
    Jacobian). Pass ``powder_scheme='zcw'`` (with ``n_zcw`` -- defaults
    to n_theta*n_phi) to use a ZCW angle set instead of the rectangular
    grid; gamma is still handled the same way either way (analytically,
    via the modulation/FFT below), since ZCW here only replaces the
    (theta, phi) sampling, not the gamma treatment. Expect convergence
    at a noticeably smaller orientation count than the rectangular grid
    -- e.g. try n_zcw=987 or 1597 as a starting point and compare
    against a larger rectangular grid before trusting it for
    production sideband intensities.

    PERFORMANCE: all orientations are evaluated together as batched
    array operations (one call to np.fft.fft over all orientations at
    once, rather than looping in Python and calling np.fft.fft once
    per orientation) -- ~50-100x faster at the default grid size, since
    the earlier per-orientation Python loop paid full numpy call
    overhead 7200 times (and a fit calls this function on the order of
    a hundred times via curve_fit's finite-difference Jacobian, so that
    overhead compounds into the multi-minute fits this was rewritten
    to fix). This does mean an (n_orientations, n_time) complex array
    is held in memory rather than one length-n_time array at a time;
    orientation_chunk_size caps that (processing that many orientations'
    full (chunk, n_time) arrays per batch, looping only over chunks
    -- a handful of iterations, not thousands) if the default grid size
    is increased enough that memory becomes a concern. None (default)
    picks a chunk size that keeps each batch under ~150M array elements
    (a few hundred MB), which covers the default grid in one pass.

    subtract_residual_mean: at the exact magic angle the gamma-average
    of the CSA modulation over one rotor period is (by construction)
    already zero after removing d_iso, so this term is normally just
    numerical hygiene. If rotor_angle != 'magic', the residual mean is
    a genuine physical shift from incomplete averaging; set this False
    if you need that effect preserved.
    """
    if mas_Hz <= 0:
        raise ValueError("mas_hz must be > 0.")
    theta, phi, wts = get_powder_grid(
        n_theta, n_phi, powder_scheme=powder_scheme, n_zcw=n_zcw,
    )
    if rotor_angle == 'magic':
        beta_rotor = np.arccos(1.0 / np.sqrt(3.0))
    else:
        beta_rotor = np.deg2rad(rotor_angle)

    rotor_axis = np.column_stack([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta),
    ])
    ref_vec = np.array([0.0, 0.0, 1.0])
    cross_with_ref = np.cross(rotor_axis, ref_vec)
    cross_norm = np.linalg.norm(cross_with_ref, axis=1)
    close = cross_norm < 1e-10
    if np.any(close):
        ref2 = np.array([1.0, 0.0, 0.0])
        cross_with_ref[close] = np.cross(rotor_axis[close], ref2)
        cross_norm[close] = np.linalg.norm(cross_with_ref[close], axis=1)
    e1 = cross_with_ref / cross_norm[:, None]
    b0 = (np.cos(beta_rotor) * rotor_axis + np.sin(beta_rotor) * e1)
    gamma = (2.0 * np.pi * np.arange(n_time)) / n_time
    cos_gamma = np.cos(gamma)
    sin_gamma = np.sin(gamma)
    tensor = csa
    csa_iso = (csa[0, 0] + csa[1, 1] + csa[2, 2]) / 3.0
    sideband_int = np.zeros(2 * n_sidebands + 1, dtype=float)
    dt = 1.0 / (mas_Hz * n_time)
    t_grid = np.arange(n_time) * dt
    centre = n_time // 2

    n_orientations = len(wts)
    if orientation_chunk_size is None:
        # Cache-blocking, not just a memory cap: every array below is
        # shape (chunk, n_time), and even though this whole function is
        # otherwise fully vectorised (no more per-orientation Python
        # loop), processing all orientations as a single giant array
        # measured SLOWER than a reasonably-sized chunk loop -- e.g. at
        # n_theta=100, n_phi=200, n_time=2048 (20,000 orientations),
        # one single pass took ~5.2s, chunks of 100 orientations ~2.2s,
        # because a (20000, 2048) array (~650MB for the complex128
        # arrays below) blows well past CPU cache, so every elementwise/
        # cumsum/FFT pass over it is memory-bandwidth-bound rather than
        # compute-bound. Targeting ~200,000 elements per chunk keeps
        # each chunk's working set in the low single-digit MB range
        # (comfortably inside L2/L3 on essentially any machine) while
        # still being large enough that the reduced number of chunk-
        # loop iterations doesn't reintroduce Python-overhead as the
        # bottleneck (the failure mode of the original per-orientation
        # loop this whole function replaces).
        orientation_chunk_size = max(1, int(2.0e5 // n_time))

    # `vectors[k,n,:]` (the Rodrigues-rotated orientation at gamma[n] for
    # crystallite k) is a linear combination of THREE fixed 3-vectors
    # per k -- u1=b0[k], u2=cross_ab[k], u3=dot_ab[k]*rotor_axis[k] --
    # with coefficients cos_gamma[n], sin_gamma[n], (1-cos_gamma[n]).
    # delta[k,n] = vectors[k,n,:] @ tensor @ vectors[k,n,:] is therefore
    # a quadratic form in those three coefficients, i.e. a fixed sum of
    # 6 outer products of a length-n_orientations array with a
    # length-n_time array -- computed below via the 6 scalar bilinear
    # forms (Auu, Avv, Aww, Auv, Auw, Avw), each O(n_orientations) to
    # get (batched (n,3)@(3,3) matmuls). This never has to build the
    # (n_orientations, n_time, 3) array a direct translation of the
    # original per-orientation loop would need -- that intermediate
    # was tried first and made things SLOWER than the loop it replaced
    # (memory-bandwidth bound: 100+MB of temporaries, several passes
    # over all of it), even though the orientation loop itself is
    # replaced by cheap array ops either way.
    cos2 = cos_gamma ** 2
    sin2 = sin_gamma ** 2
    one_minus_cos = 1.0 - cos_gamma
    omc2 = one_minus_cos ** 2
    cs = cos_gamma * sin_gamma
    comc = cos_gamma * one_minus_cos
    somc = sin_gamma * one_minus_cos

    for start in range(0, n_orientations, orientation_chunk_size):
        stop = min(start + orientation_chunk_size, n_orientations)
        axis = rotor_axis[start:stop]                          # (c, 3)
        b = b0[start:stop]                                      # (c, 3)
        wts_chunk = wts[start:stop]                              # (c,)

        u1 = b
        u2 = np.cross(axis, b)
        u3 = np.einsum("ki,ki->k", axis, b)[:, None] * axis      # dot_ab[:,None]*axis

        Tu1 = u1 @ tensor
        Tu2 = u2 @ tensor
        Tu3 = u3 @ tensor
        Auu = np.sum(Tu1 * u1, axis=1)
        Avv = np.sum(Tu2 * u2, axis=1)
        Aww = np.sum(Tu3 * u3, axis=1)
        Auv = np.sum(Tu1 * u2, axis=1)
        Auw = np.sum(Tu1 * u3, axis=1)
        Avw = np.sum(Tu2 * u3, axis=1)

        delta = (Auu[:, None] * cos2[None, :] + Avv[:, None] * sin2[None, :]
                 + Aww[:, None] * omc2[None, :] + 2.0 * Auv[:, None] * cs[None, :]
                 + 2.0 * Auw[:, None] * comc[None, :] + 2.0 * Avw[:, None] * somc[None, :])   # (c, n_time)

        modulation_Hz = (delta - csa_iso) * larmor_Hz / (1e+06)

        phase = np.zeros_like(modulation_Hz)
        # phase (cycles) = integral of modulation_Hz dt (trapezoidal)
        phase[:, 1:] = np.cumsum(0.5 * (modulation_Hz[:, :-1] + modulation_Hz[:, 1:]) * dt, axis=1)
        if subtract_residual_mean:
            phase -= np.mean(modulation_Hz, axis=1, keepdims=True) * t_grid[None, :]

        signal = np.exp(-1j * 2.0 * np.pi * phase)              # (c, n_time)
        coeff = np.fft.fft(signal, axis=1) / n_time              # one batched FFT call
        coeff = np.fft.fftshift(coeff, axes=1)
        selected_ssb = coeff[:, centre - n_sidebands: centre + n_sidebands + 1]
        intensities = np.abs(selected_ssb ** 2)                  # (c, 2*n_sidebands+1)
        sideband_int += np.sum(wts_chunk[:, None] * intensities, axis=0)

    orders = np.arange(-n_sidebands, n_sidebands + 1)
    sideband_Hz = (orders * mas_Hz)
    sideband_ppm = (csa_iso - sideband_Hz * (1.0e+06) / larmor_Hz)
    spectrum = np.zeros_like(ppm_axis, dtype=float)
    for shift, ints in zip(sideband_ppm, sideband_int):
        i = np.searchsorted(ppm_axis, shift)
        if i <= 0 or i >= len(ppm_axis):
            continue
        x1 = ppm_axis[i - 1]
        x2 = ppm_axis[i]
        frac = (shift - x1) / (x2 - x1)
        spectrum[i - 1] += (ints * (1.0 - frac))
        spectrum[i] += (ints * frac)
    spectrum = _apply_broadening(spectrum, ppm_axis, fwhm=fwhm, lineshape=lineshape, eta=eta)
    maximum = np.max(spectrum)
    normalised = spectrum / maximum if maximum > 0 else spectrum
    return {
        "x_ppm": ppm_axis,
        "real_spec": spectrum,
        "normalised_real_spec": normalised,
    }


# ============================================================
# QUADRUPOLAR NMR
# ============================================================
#
# One engine for static and MAS powder patterns of a quadrupolar
# nucleus (I >= 1), at three levels of theory, for any choice of
# transitions:
#
#     order='first'   Zeeman + first-order quadrupolar shift
#     order='second'  + second-order shift (perturbation theory)
#     order='exact'   exact diagonalisation of H_Z + H_Q
#
#     transitions='ct'   central transition (-1/2 <-> +1/2) only
#                 'st'   satellite transitions only
#                 'all'  CT and STs together
#                 'ST1', ['CT', 'ST2'], ...  named groups
#
# Workflow
# --------
#
#     larmor = larmor_frequency_Hz('27Al', 11.7467)
#     axis = np.linspace(*quadrupolar_axis_limits(2.5, 3e6, 0.3, larmor,
#                                                 transitions='ct'), 4096)
#     static = quadrupolar_static_spectrum(2.5, 3e6, 0.3, larmor, axis,
#                                          order='exact', transitions='ct')
#     mas = quadrupolar_mas_spectrum(2.5, 3e6, 0.3, larmor, 12.5e3, axis,
#                                    order='second', transitions='all')
#     orders = quadrupolar_order_comparison(2.5, 3e6, 0.3, larmor, axis,
#                                           mas_Hz=12.5e3)
#
# Why this is fast (and why the per-time-step version was slow)
# -------------------------------------------------------------
# In the high-field limit every transition frequency depends only on
# the direction of B0 in the EFG principal-axis frame (PAS), b. The
# first-order shift is a rank-2 function of b and the second-order
# shift is built from three scalars of b (see _quad_b_invariants), so
# both are closed-form array expressions -- no matrices at all. Under
# MAS, b rotates about the rotor axis, so the frequency of each
# crystallite is a Fourier series in the rotor phase with at most 4
# harmonics (2 at first order). It is therefore sampled exactly at a
# handful of rotor phases, integrated to a phase analytically in the
# Fourier domain, and the gamma-averaged sideband intensities are
# |a_k|^2 of the one-period signal (gamma-COMPUTE; the same idea as
# finite_mas_csa_spectrum). Every crystallite is processed at once as
# arrays. 'exact' does the same with batched np.linalg.eigh calls on
# the (2I+1)x(2I+1) Hamiltonian at each sampled rotor phase.
#
# Conventions
# -----------
# H_Q/h = C_Q / [4I(2I-1)] * [3Iz^2 - I(I+1) + eta_Q (Ix^2 - Iy^2)] in
# the PAS, |V_zz| >= |V_yy| >= |V_xx|, C_Q = e^2 q Q / h in Hz. The
# Zeeman term is -nu0 (b . I) with nu0 = |gamma B0|, so ppm offsets do
# not depend on the sign of gamma. Transition "m-1 <-> m" is labelled
# by its upper m. Relative intensities are |<m-1|I+|m>|^2 =
# I(I+1) - m(m-1) (perturbative orders) or the exact eigenvector
# matrix elements ('exact'), normalised so that ALL 2I transitions of
# a crystallite sum to 1: a CT-only spectrum therefore integrates to
# the CT's true share of the total (9/35 for I=5/2).
# ============================================================

QUADRUPOLAR_ORDERS = ("first", "second", "exact")


def spin_operators(I):
    """
    Return spin operators Ix, Iy, Iz for arbitrary spin I.
    Operators are dimensionless, i.e. eigenvalues of Iz are m
    rather than hbar*m. Basis order is m = I, I-1, ..., -I.
    """
    n = int(round(2 * I + 1))
    m = np.arange(I, -I - 1, -1, dtype=float)
    Iz = np.diag(m)
    Ip = np.zeros((n, n), dtype=complex)
    for j, mj in enumerate(m):
        mp = mj + 1.0
        if mp <= I:
            i = np.where(np.isclose(m, mp))[0]
            if len(i):
                Ip[i[0], j] = np.sqrt(I * (I + 1.0) - mj * mp)
    Im = Ip.T.conj()
    Ix = 0.5 * (Ip + Im)
    Iy = (Ip - Im) / (2.0j)
    return Ix, Iy, Iz


def larmor_frequency_Hz(nucleus, B0_T):
    """|gamma| * B0 in Hz for a nuclide label such as '27Al'."""
    return abs(float(find_isotope_info(nucleus)["gamma"])) * float(B0_T)


def quadrupolar_nu_Q(I, C_Q):
    """nu_Q = 3 C_Q / [2I(2I-1)] in Hz -- the first-order splitting
    between adjacent satellite transitions for eta_Q = 0, theta = 0."""
    I = float(I)
    return 3.0 * C_Q / (2.0 * I * (2.0 * I - 1.0))


def _check_quad_inputs(I, eta_Q, larmor_Hz):
    I = float(I)
    if I < 1.0:
        raise ValueError("Quadrupolar interaction requires I >= 1.")
    if not np.isclose(2.0 * I, round(2.0 * I)):
        raise ValueError("I must be an integer or half-integer.")
    if not (0.0 <= float(eta_Q) <= 1.0):
        raise ValueError("eta_Q must lie between 0 and 1.")
    if not larmor_Hz or larmor_Hz <= 0:
        raise ValueError("larmor_Hz must be > 0.")
    return I


def _format_m(m):
    """'+3/2', '-1/2', '0', '+1' -- quantum numbers as usually written."""
    if np.isclose(m, 0.0):
        return "0"
    sign = "+" if m > 0 else "−"
    twice = int(round(2 * abs(m)))
    return f"{sign}{twice}/2" if twice % 2 else f"{sign}{twice // 2}"


def quadrupolar_transition_table(I):
    """
    Every single-quantum transition of spin I, as a list of dicts with
    ``m`` (upper m of the pair m-1 <-> m), ``label`` (e.g.
    '+1/2↔+3/2') and ``group``: 'CT' for -1/2 <-> +1/2, 'ST1' for
    the pair of transitions next to it (+-1/2 <-> +-3/2), 'ST2' for the
    next, and so on. For integer spins every transition is a satellite
    (the two lines of a 2H Pake doublet are 'ST1').
    """
    I = float(I)
    half_integer = not float(I).is_integer()
    table = []
    for m in np.arange(I, -I, -1.0):
        if half_integer and np.isclose(m, 0.5):
            group = "CT"
        else:
            group = f"ST{int(np.ceil(abs(m - 0.5) - 1e-9))}"
        table.append({
            "m": float(m),
            "label": f"{_format_m(m - 1.0)}↔{_format_m(m)}",
            "group": group,
            "weight": I * (I + 1.0) - m * (m - 1.0),
        })
    return table


def _select_transitions(I, transitions):
    """Rows of quadrupolar_transition_table picked by ``transitions``."""
    table = quadrupolar_transition_table(I)
    groups = {row["group"] for row in table}
    if isinstance(transitions, str):
        key = transitions.strip().lower()
        if key in ("all", "ct+st", "both"):
            return table
        if key == "ct":
            if "CT" not in groups:
                raise ValueError(
                    f"I = {float(I):g} is an integer spin: it has no central "
                    "transition. Use transitions='all' (or 'st').")
            return [row for row in table if row["group"] == "CT"]
        if key in ("st", "sts", "satellites"):
            return [row for row in table if row["group"] != "CT"]
        wanted = {transitions.strip().upper()}
    else:
        wanted = {str(t).strip().upper() for t in transitions}
    unknown = wanted - groups
    if unknown:
        raise ValueError(
            f"Unknown transition group(s) {sorted(unknown)} for I = {float(I):g}; "
            f"available: {sorted(groups)}.")
    return [row for row in table if row["group"] in wanted]


def _efg_principal(eta_Q):
    """Principal values of V / eq (traceless, V_zz = 1)."""
    return np.array([-(1.0 - eta_Q) / 2.0, -(1.0 + eta_Q) / 2.0, 1.0])


def _quad_b_invariants(eta_Q, b):
    """
    The three scalars every perturbative quadrupolar frequency is
    built from, for unit vectors ``b`` (B0 direction in the PAS,
    shape (..., 3)), with V in units of eq and x', y', z' any lab frame
    with z' along B0:

        V0 = V_z'z'                                   (first order)
        P1 = V_x'z'^2 + V_y'z'^2    = |V b|^2 - V0^2
        P2 = (V_x'x' - V_y'y')^2 + 4 V_x'y'^2
           = 2 (tr V^2 - V0^2 - 2 P1) - V0^2

    All three are invariant to rotations about B0, which is why no
    third Euler angle is needed.
    """
    v = _efg_principal(eta_Q)
    b2 = b * b
    V0 = b2 @ v
    P1 = np.maximum(b2 @ (v * v) - V0 * V0, 0.0)
    P2 = np.maximum(2.0 * (np.sum(v * v) - V0 * V0 - 2.0 * P1) - V0 * V0, 0.0)
    return V0, P1, P2


def _second_order_energy(I, m, P1, P2):
    """Second-order energy of Zeeman level m, in units of K^2 / nu0
    (K = C_Q / [2I(2I-1)]). Standard non-degenerate perturbation
    theory with E_Z(m) = -nu0 m; |<m+-1|H_Q|m>|^2 = K^2/4 P1
    (2m+-1)^2 c+-(m)^2 and |<m+-2|H_Q|m>|^2 = K^2/16 P2 c+-(m)^2
    c+-(m+-1)^2, c+-(m)^2 = I(I+1) - m(m+-1)."""
    s = I * (I + 1.0)
    cp = lambda mm: s - mm * (mm + 1.0)
    cm = lambda mm: s - mm * (mm - 1.0)
    first = cp(m) * (2.0 * m + 1.0) ** 2 - cm(m) * (2.0 * m - 1.0) ** 2
    second = cp(m) * cp(m + 1.0) - cm(m) * cm(m - 1.0)
    return P1 * first / 4.0 + P2 * second / 32.0


def _perturbative_offsets(I, C_Q, eta_Q, larmor_Hz, b, m, order):
    """
    Transition frequencies minus nu0 (Hz), shape b.shape[:-1] + (len(m),),
    for the transitions m-1 <-> m at first or second order.
    """
    m = np.asarray(m, dtype=float)
    K = C_Q / (2.0 * I * (2.0 * I - 1.0))
    V0, P1, P2 = _quad_b_invariants(eta_Q, b)
    nu = -1.5 * K * V0[..., None] * (2.0 * m - 1.0)
    if order == "second":
        P1 = P1[..., None]
        P2 = P2[..., None]
        nu = nu + (K * K / larmor_Hz) * (
            _second_order_energy(I, m - 1.0, P1, P2) - _second_order_energy(I, m, P1, P2))
    return nu


def _exact_offsets(I, C_Q, eta_Q, larmor_Hz, b, m, chunk_size=50000):
    """
    Exact transition frequencies minus nu0 (Hz) and intensities for the
    transitions m-1 <-> m, by batched diagonalisation of
    H = -nu0 (b . I) + H_Q over every direction in ``b``.

    Levels are assigned by energy order (level m is the (I-m)-th
    lowest), which is unambiguous while nu0 >> nu_Q -- the regime in
    which "central" and "satellite" transitions mean anything.
    Intensities are 2 x the transverse (perpendicular to B0) squared
    matrix element, i.e. I(I+1) - m(m-1) when H_Q -> 0.
    """
    m = np.asarray(m, dtype=float)
    Ix, Iy, Iz = spin_operators(I)
    v = _efg_principal(eta_Q)
    K = C_Q / (2.0 * I * (2.0 * I - 1.0))
    H_Q = K * (v[0] * Ix @ Ix + v[1] * Iy @ Iy + v[2] * Iz @ Iz)
    ops = np.stack([Ix, Iy, Iz])                       # (3, d, d)
    lead = b.shape[:-1]
    flat = b.reshape(-1, 3)
    j = np.round(I - m).astype(int)                    # index of level m
    nu = np.empty((flat.shape[0], len(m)))
    inten = np.empty((flat.shape[0], len(m)))
    for start in range(0, flat.shape[0], chunk_size):
        bb = flat[start:start + chunk_size]
        H = H_Q[None] - larmor_Hz * np.einsum("nk,kij->nij", bb, ops)
        E, U = np.linalg.eigh(H)
        nu[start:start + chunk_size] = E[:, j + 1] - E[:, j] - larmor_Hz
        upper = U[:, :, j]                             # level m   (n, d, T)
        lower = U[:, :, j + 1]                         # level m-1
        mel = np.einsum("ndt,kde,net->nkt", lower.conj(), ops, upper)  # (n, 3, T)
        along_b = np.einsum("nk,nkt->nt", bb, mel)
        inten[start:start + chunk_size] = 2.0 * (
            np.sum(np.abs(mel) ** 2, axis=1) - np.abs(along_b) ** 2)
    return nu.reshape(lead + (len(m),)), inten.reshape(lead + (len(m),))


def quadrupolar_transition_frequencies(I, C_Q, eta_Q, larmor_Hz, theta, phi,
                                       order="exact", transitions="all"):
    """
    Transition frequencies (as offsets from nu0, in Hz) and intensities
    for crystallites with B0 at polar angles (theta, phi) in the EFG
    PAS -- a single orientation or arrays of them.

    Returns a dict with ``offset_Hz`` and ``intensity`` (shape
    (n_orientations, n_transitions)), plus ``m``, ``label``, ``group``
    describing each column. Useful for inspecting one crystallite, or
    for comparing orders directly: ``exact - second`` is the third-
    and-higher-order remainder.
    """
    I = _check_quad_inputs(I, eta_Q, larmor_Hz)
    if order not in QUADRUPOLAR_ORDERS:
        raise ValueError(f"order must be one of {QUADRUPOLAR_ORDERS}.")
    rows = _select_transitions(I, transitions)
    m = np.array([row["m"] for row in rows])
    theta = np.atleast_1d(np.asarray(theta, dtype=float))
    phi = np.atleast_1d(np.asarray(phi, dtype=float))
    b = np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi),
                  np.cos(theta)], axis=-1)
    total_weight = sum(row["weight"] for row in quadrupolar_transition_table(I))
    if order == "exact":
        nu, inten = _exact_offsets(I, C_Q, eta_Q, larmor_Hz, b, m)
    else:
        nu = _perturbative_offsets(I, C_Q, eta_Q, larmor_Hz, b, m, order)
        inten = np.broadcast_to(np.array([row["weight"] for row in rows]), nu.shape).copy()
    return {
        "offset_Hz": nu,
        "intensity": inten / total_weight,
        "m": m,
        "label": [row["label"] for row in rows],
        "group": [row["group"] for row in rows],
    }


def quadrupolar_ct_isotropic_shift_Hz(I, C_Q, eta_Q, larmor_Hz):
    """
    Second-order quadrupolar-induced isotropic shift of the central
    transition (Hz; always negative), i.e. the centre of gravity of
    the CT under MAS relative to delta_iso:

        -(3/40) C_Q^2 / nu0 * [I(I+1) - 3/4] / [I^2 (2I-1)^2] * (1 + eta^2/3)
    """
    I = float(I)
    return (-(3.0 / 40.0) * C_Q ** 2 / larmor_Hz
            * (I * (I + 1.0) - 0.75) / (I ** 2 * (2.0 * I - 1.0) ** 2)
            * (1.0 + eta_Q ** 2 / 3.0))


def _bin_sticks(ppm_axis, positions_ppm, weights):
    """Linear-interpolation histogram of sticks onto ``ppm_axis``
    (ascending or descending, uniformly spaced)."""
    out = np.zeros(len(ppm_axis))
    positions_ppm = np.ravel(positions_ppm)
    weights = np.ravel(weights)
    dx = ppm_axis[1] - ppm_axis[0]
    idx = (positions_ppm - ppm_axis[0]) / dx
    i0 = np.floor(idx).astype(np.int64)
    frac = idx - i0
    ok = (i0 >= 0) & (i0 < len(ppm_axis) - 1) & (weights != 0)
    i0, frac, w = i0[ok], frac[ok], weights[ok]
    out += np.bincount(i0, w * (1.0 - frac), minlength=len(ppm_axis))
    out += np.bincount(i0 + 1, w * frac, minlength=len(ppm_axis))
    return out


def _quad_result(ppm_axis, components, fwhm, lineshape, eta, info):
    """Broaden each transition group, sum, and pack the result dict."""
    broadened = {}
    for group, stick in components.items():
        broadened[group] = (_apply_broadening(stick, ppm_axis, fwhm=fwhm,
                                              lineshape=lineshape, eta=eta)
                            if fwhm > 0 else stick)
    total = np.sum(list(broadened.values()), axis=0) if broadened else np.zeros(len(ppm_axis))
    maximum = np.max(total) if total.size else 0.0
    return {
        "x_ppm": ppm_axis,
        "real_spec": total,
        "normalised_real_spec": total / maximum if maximum > 0 else total,
        "components": broadened,
        **info,
    }


def _quad_log(regime, I, C_Q, eta_Q, larmor_Hz, order, rows, delta_iso, extra=""):
    groups = sorted({row["group"] for row in rows}, key=lambda g: (g != "CT", g))
    order_text = {"first": "first-order perturbation theory",
                  "second": "second-order perturbation theory",
                  "exact": "exact diagonalisation of H_Z + H_Q"}[order]
    return (f"{regime} quadrupolar powder pattern: I = {I:g}, C_Q = {C_Q / 1e6:g} MHz, "
            f"eta_Q = {eta_Q:g}, nu0 = {larmor_Hz / 1e6:.6g} MHz, "
            f"nu_Q = {quadrupolar_nu_Q(I, C_Q) / 1e3:.4g} kHz, delta_iso = {delta_iso:g} ppm; "
            f"{order_text}; transitions {', '.join(groups)}{extra}.")


def quadrupolar_static_spectrum(I, C_Q, eta_Q, larmor_Hz, ppm_axis, order="exact",
                                transitions="all", delta_iso=0.0, fwhm=1.0,
                                lineshape="gaussian", eta=0.0, n_zcw=17711,
                                powder_scheme="zcw", n_theta=150, n_phi=300):
    """
    Static powder pattern of a quadrupolar nucleus.

    Parameters
    ----------
    I : float
        Spin (>= 1), e.g. 2.5 for 27Al.
    C_Q : float
        Quadrupolar coupling constant e^2qQ/h, in Hz.
    eta_Q : float
        Asymmetry parameter, 0..1.
    larmor_Hz : float
        |gamma B0| in Hz (see larmor_frequency_Hz).
    ppm_axis : ndarray
        Uniformly spaced ppm axis (either direction).
    order : {'first', 'second', 'exact'}
    transitions : {'all', 'ct', 'st'} or group name(s) such as 'ST1'
    delta_iso : float
        Isotropic chemical shift, ppm.
    fwhm, lineshape, eta
        Broadening, as make_static_csa_spectrum (fwhm in ppm; 0 = none).
    n_zcw / powder_scheme / n_theta, n_phi
        Powder average, as get_powder_grid. The default ZCW set is
        large because each orientation is cheap here.

    Returns
    -------
    dict with ``x_ppm``, ``real_spec`` (sum of the selected
    transitions), ``normalised_real_spec``, ``components`` (one
    broadened spectrum per transition group: 'CT', 'ST1', ...),
    ``order``, ``transitions`` (labels), ``n_orientations`` and
    ``log`` (a one-line description of what was simulated).
    """
    I = _check_quad_inputs(I, eta_Q, larmor_Hz)
    ppm_axis = np.asarray(ppm_axis, dtype=float)
    theta, phi, wts = get_powder_grid(n_theta, n_phi, powder_scheme=powder_scheme,
                                      n_zcw=n_zcw)
    freqs = quadrupolar_transition_frequencies(I, C_Q, eta_Q, larmor_Hz, theta, phi,
                                               order=order, transitions=transitions)
    ppm = delta_iso + freqs["offset_Hz"] / larmor_Hz * 1e6
    weight = wts[:, None] * freqs["intensity"]
    components = {}
    for col, group in enumerate(freqs["group"]):
        stick = _bin_sticks(ppm_axis, ppm[:, col], weight[:, col])
        components[group] = components.get(group, 0.0) + stick
    rows = _select_transitions(I, transitions)
    info = {
        "order": order, "regime": "static", "transitions": freqs["label"],
        "n_orientations": len(wts),
        "log": _quad_log("Static", I, C_Q, eta_Q, larmor_Hz, order, rows, delta_iso,
                         f"; {len(wts)} orientations ({powder_scheme})"),
    }
    return _quad_result(ppm_axis, components, fwhm, lineshape, eta, info)


def _rotor_frame(theta, phi):
    """Rotor-axis unit vectors r (from polar angles in the PAS) and
    two unit vectors e1, e2 completing a right-handed frame."""
    r = np.column_stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi),
                         np.cos(theta)])
    ref = np.zeros_like(r)
    ref[:, 2] = 1.0
    near_z = np.abs(r[:, 2]) > 0.9
    ref[near_z] = [1.0, 0.0, 0.0]
    e1 = np.cross(r, ref)
    e1 /= np.linalg.norm(e1, axis=1)[:, None]
    e2 = np.cross(r, e1)
    return r, e1, e2


def quadrupolar_mas_spectrum(I, C_Q, eta_Q, larmor_Hz, mas_Hz, ppm_axis, order="second",
                             transitions="all", delta_iso=0.0, rotor_angle="magic",
                             fwhm=1.0, lineshape="gaussian", eta=0.0, n_zcw=4181,
                             powder_scheme="zcw", n_theta=40, n_phi=80,
                             n_gamma=None, n_time=None, max_n_time=65536):
    """
    Finite-speed MAS powder pattern of a quadrupolar nucleus: the
    centreband lineshape and every spinning sideband, for the CT, the
    STs, or both, at first order, second order or exactly.

    Method (gamma-COMPUTE): for each crystallite, the transition
    frequency is sampled at ``n_gamma`` rotor phases (exact for the
    perturbative orders -- they contain no harmonic above the 4th --
    and converged for 'exact'), its mean nu_c is the centreband
    position, the rest is integrated to a phase in the Fourier
    domain, and the gamma-averaged intensity of sideband k (at
    nu_c + k * mas_Hz) is |a_k|^2, a_k being the k-th Fourier
    coefficient of exp(2 pi i * phase) over one rotor period.

    Parameters (beyond quadrupolar_static_spectrum's)
    ----------
    mas_Hz : float
        Spinning rate, Hz (> 0).
    rotor_angle : 'magic' or float (degrees)
    n_zcw : int
        Rotor-axis orientations (ZCW). The rotor phase supplies the
        third angle, so a 2-angle set is the complete powder average.
    n_gamma : int, optional
        Rotor-phase samples per period: default 16 for the perturbative
        orders (exact), 32 for 'exact'.
    n_time : int, optional
        Time points per rotor period for the sideband FFT (a power of
        two). Default: chosen from the largest frequency excursion so
        that no sideband aliases. Raises if that would exceed
        ``max_n_time`` (a static simulation is then the better tool).

    Notes
    -----
    The CT's centreband carries the second-order MAS lineshape and the
    isotropic quadrupolar shift (quadrupolar_ct_isotropic_shift_Hz);
    at first order the CT is a single line at delta_iso. 'exact'
    assumes the eigenstates follow the rotation adiabatically (true for
    nu0 >> nu_Q) and uses the rotor-period-averaged intensity of each
    crystallite.
    """
    I = _check_quad_inputs(I, eta_Q, larmor_Hz)
    if order not in QUADRUPOLAR_ORDERS:
        raise ValueError(f"order must be one of {QUADRUPOLAR_ORDERS}.")
    if not mas_Hz or mas_Hz <= 0:
        raise ValueError("mas_Hz must be > 0 (use quadrupolar_static_spectrum for static).")
    ppm_axis = np.asarray(ppm_axis, dtype=float)
    rows = _select_transitions(I, transitions)
    m = np.array([row["m"] for row in rows])
    total_weight = sum(row["weight"] for row in quadrupolar_transition_table(I))
    beta = (np.arccos(1.0 / np.sqrt(3.0)) if rotor_angle == "magic"
            else np.deg2rad(float(rotor_angle)))
    if n_gamma is None:
        n_gamma = 32 if order == "exact" else 16
    n_gamma = int(n_gamma)

    theta, phi, wts = get_powder_grid(n_theta, n_phi, powder_scheme=powder_scheme,
                                      n_zcw=n_zcw)
    r, e1, e2 = _rotor_frame(theta, phi)
    gam = 2.0 * np.pi * np.arange(n_gamma) / n_gamma
    b = (np.cos(beta) * r[:, None, :]
         + np.sin(beta) * (np.cos(gam)[None, :, None] * e1[:, None, :]
                           + np.sin(gam)[None, :, None] * e2[:, None, :]))  # (N, G, 3)

    if order == "exact":
        nu, inten = _exact_offsets(I, C_Q, eta_Q, larmor_Hz, b, m)
        inten = inten.mean(axis=1)                                          # (N, T)
    else:
        nu = _perturbative_offsets(I, C_Q, eta_Q, larmor_Hz, b, m, order)   # (N, G, T)
        inten = np.broadcast_to(np.array([row["weight"] for row in rows]),
                                (len(wts), len(m)))
    inten = inten / total_weight

    # Fourier series of the frequency in the rotor phase.
    coeff = np.fft.fft(nu, axis=1) / n_gamma                                # (N, G, T)
    centre = coeff[:, 0, :].real                                            # (N, T)
    k_g = np.fft.fftfreq(n_gamma, d=1.0 / n_gamma).round().astype(int)
    keep = (k_g != 0) & (np.abs(k_g) < n_gamma / 2)
    excursion = float(np.max(np.abs(nu - centre[:, None, :]))) if nu.size else 0.0

    if n_time is None:
        need = 2.0 * (1.3 * excursion / mas_Hz + 4 * n_gamma) + 16
        n_time = int(2 ** np.ceil(np.log2(max(need, 64))))
    n_time = int(n_time)
    if n_time > max_n_time:
        raise ValueError(
            f"The spinning sidebands span about {2 * excursion / mas_Hz:.0f} x the "
            f"MAS rate, needing {n_time} points per rotor period (max_n_time = "
            f"{max_n_time}). Spin faster, simulate fewer transitions (e.g. "
            "transitions='ct'), or use the static simulation.")
    if n_time < n_gamma:
        raise ValueError("n_time must be >= n_gamma.")

    k_t = np.fft.fftfreq(n_time, d=1.0 / n_time).round().astype(int)
    # phase (cycles) Fourier coefficients: c_k / (2 pi i k nu_r) * (2 pi) -> c_k / (i k nu_r) / (2 pi)
    phase_coeff = np.zeros_like(coeff)
    phase_coeff[:, keep, :] = coeff[:, keep, :] / (2j * np.pi * k_g[keep, None] * mas_Hz)
    slots = np.mod(k_g[keep], n_time)

    components = {row["group"]: np.zeros(len(ppm_axis)) for row in rows}
    groups = [row["group"] for row in rows]
    n_trans = len(m)
    chunk = max(1, int(4.0e6 // (n_time * n_trans)))
    sideband_Hz = k_t * mas_Hz
    for start in range(0, len(wts), chunk):
        stop = min(start + chunk, len(wts))
        spectrum_coeff = np.zeros((stop - start, n_time, n_trans), dtype=complex)
        spectrum_coeff[:, slots, :] = phase_coeff[start:stop][:, keep, :]
        phase = np.fft.ifft(spectrum_coeff, axis=1).real * n_time           # cycles
        signal = np.exp(2j * np.pi * phase)
        a = np.fft.fft(signal, axis=1) / n_time
        power = (a.real ** 2 + a.imag ** 2)                                 # (c, n_time, T)
        weight = wts[start:stop, None, None] * inten[start:stop, None, :] * power
        position = delta_iso + (centre[start:stop, None, :] + sideband_Hz[None, :, None]) \
            / larmor_Hz * 1e6
        for col, group in enumerate(groups):
            components[group] += _bin_sticks(ppm_axis, position[:, :, col], weight[:, :, col])

    rotor_text = "magic angle" if rotor_angle == "magic" else f"{float(rotor_angle):g} deg"
    info = {
        "order": order, "regime": "mas", "mas_Hz": float(mas_Hz),
        "transitions": [row["label"] for row in rows],
        "n_orientations": len(wts), "n_gamma": n_gamma, "n_time": n_time,
        "log": _quad_log(f"MAS ({mas_Hz / 1e3:g} kHz, {rotor_text})", I, C_Q, eta_Q,
                         larmor_Hz, order, rows, delta_iso,
                         f"; {len(wts)} rotor orientations x {n_gamma} rotor phases, "
                         f"{n_time} points per rotor period"),
    }
    return _quad_result(ppm_axis, components, fwhm, lineshape, eta, info)


def quadrupolar_spectrum(I, C_Q, eta_Q, larmor_Hz, ppm_axis, mas_Hz=0.0, **kwargs):
    """Static (``mas_Hz`` = 0) or MAS quadrupolar powder pattern --
    dispatches to quadrupolar_static_spectrum / quadrupolar_mas_spectrum
    with the same keyword arguments."""
    if mas_Hz and mas_Hz > 0:
        return quadrupolar_mas_spectrum(I, C_Q, eta_Q, larmor_Hz, mas_Hz, ppm_axis, **kwargs)
    kwargs.pop("rotor_angle", None)
    for key in ("n_gamma", "n_time", "max_n_time"):
        kwargs.pop(key, None)
    return quadrupolar_static_spectrum(I, C_Q, eta_Q, larmor_Hz, ppm_axis, **kwargs)


def quadrupolar_order_comparison(I, C_Q, eta_Q, larmor_Hz, ppm_axis, mas_Hz=0.0,
                                 orders=QUADRUPOLAR_ORDERS, **kwargs):
    """
    The same pattern at several levels of theory, for seeing what each
    order contributes: returns ``{order: result}`` (results as
    quadrupolar_spectrum), plus ``'second_minus_first'`` and
    ``'exact_minus_second'`` difference spectra (on the shared
    normalisation, not renormalised) where both orders were computed.
    """
    out = {order: quadrupolar_spectrum(I, C_Q, eta_Q, larmor_Hz, ppm_axis, mas_Hz=mas_Hz,
                                       order=order, **kwargs)
           for order in orders}
    if "first" in out and "second" in out:
        out["second_minus_first"] = out["second"]["real_spec"] - out["first"]["real_spec"]
    if "second" in out and "exact" in out:
        out["exact_minus_second"] = out["exact"]["real_spec"] - out["second"]["real_spec"]
    return out


def quadrupolar_axis_limits(I, C_Q, eta_Q, larmor_Hz, transitions="all", delta_iso=0.0,
                            fwhm=0.0, margin=0.15):
    """
    ppm axis limits ``(low, high)`` that frame a quadrupolar pattern:
    the full static second-order range of the selected transitions
    (which also contains all their MAS sidebands of significant
    intensity), widened by ``margin`` of that range on each side and
    by at least five broadening FWHMs.
    """
    I = _check_quad_inputs(I, eta_Q, larmor_Hz)
    theta, phi, _ = zcw_angles(2584)
    freqs = quadrupolar_transition_frequencies(I, C_Q, eta_Q, larmor_Hz, theta, phi,
                                               order="second", transitions=transitions)
    ppm = delta_iso + freqs["offset_Hz"] / larmor_Hz * 1e6
    low, high = float(np.min(ppm)), float(np.max(ppm))
    pad = max(margin * (high - low), 5.0 * abs(float(fwhm)), 1.0)
    return low - pad, high + pad


# ============================================================
# J-COUPLING (SCALAR-COUPLED) MULTIPLETS
# ============================================================
#
# First-order (weak-coupling) multiplet simulation: the observed
# nucleus is coupled to one or more *groups* of equivalent nuclei,
# each group defined by (nuclide, how many equivalent nuclei n, the
# J coupling in Hz to that group). Several independent groups can be
# combined (e.g. a 13C coupled to 1H, 31P and 19F simultaneously,
# each with its own n and J) -- their splitting patterns simply
# convolve.
#
# Natural isotopic abundance is folded in per group: for a coupling
# partner whose NMR-active isotope is not ~100% abundant (13C, 2H,
# ...), only a fraction of molecules actually have an active spin at
# each of the n equivalent sites -- the rest have a spin-0 isotope
# there (e.g. 12C) that does not split the line at all. This is
# handled as a binomial mixture over "how many of the n sites are
# NMR-active" (see `jcoupling_group_pattern`), which is exactly what
# produces the familiar small J-coupling *satellites* either side of
# a large uncoupled centreband for a low-abundance partner (13C
# satellites in a 1H spectrum, for example), rather than a full
# multiplet with no centreband (the 100%-abundance case, e.g. 1H,
# 19F, 31P partners).
#
# This is first-order/weak-coupling only (no roofing/second-order
# strong-coupling distortion -- that needs full density-matrix
# simulation and is a different scope to the existing CSA/
# quadrupolar machinery in this module).
#
# Workflow
# --------
#
#   couplings = [
#       {'nuclide': '1H',  'n': 2, 'J_Hz': 7.0},
#       {'nuclide': '31P', 'n': 1, 'J_Hz': 140.0},
#   ]
#   spec = jcoupled_spectrum(100.0, larmor_Hz, couplings, ppm_axis, fwhm=2.0)
#   plt.plot(spec['x_ppm'], spec['real_spec'])
#
# or, for several distinct chemical sites at once:
#
#   sites = [
#       {'ppm': 100.0, 'relative_intensity': 1.0, 'couplings': couplings_A},
#       {'ppm': 40.0,  'relative_intensity': 2.0, 'couplings': couplings_B},
#   ]
#   spec = multi_site_jcoupled_spectrum(sites, larmor_Hz, ppm_axis, fwhm=2.0)
#
# ============================================================


def spin_multiplet_pattern(n, I, J_Hz):
    """
    First-order stick multiplet from `n` magnetically-equivalent spin-I
    nuclei, each coupled with the same J (Hz) to the observed spin.

    Works for any spin I, not just I=1/2: reduces to the familiar
    Pascal's-triangle binomial pattern (n+1 lines, intensities
    C(n,k)) for I=1/2, and generalises correctly to higher-spin
    coupling partners (e.g. n equivalent 14N (I=1), 27Al (I=5/2), ...)
    via the multinomial expansion of (sum_{m=-I}^{I} x^m)^n.

    Parameters
    ----------
    n : int
        Number of equivalent coupled nuclei (n=0 -> single line, no
        splitting).
    I : float
        Spin quantum number of the coupled nuclei (e.g. 0.5, 1, 1.5).
    J_Hz : float
        Scalar coupling constant, in Hz.

    Returns
    -------
    freqs_Hz : ndarray
        Frequency offsets (Hz) of each line, relative to the
        uncoupled position, sorted ascending. Spans
        -n*I*J_Hz to +n*I*J_Hz in steps of J_Hz.
    weights : ndarray
        Relative intensity of each line, normalised to sum to 1.
    """
    if n <= 0:
        return np.array([0.0]), np.array([1.0])

    m_single = np.round(np.arange(-I, I + 1e-9, 1.0), 6)

    # Multinomial expansion via repeated convolution: total_M is the
    # sum of n i.i.d. m values, each drawn (with equal weight) from
    # m_single -- exactly analogous to summing n dice.
    totals = {0.0: 1}
    for _ in range(n):
        new_totals = {}
        for m_tot, w_tot in totals.items():
            for m in m_single:
                key = round(m_tot + m, 6)
                new_totals[key] = new_totals.get(key, 0) + w_tot
        totals = new_totals

    m_sorted = np.array(sorted(totals.keys()))
    weights = np.array([totals[m] for m in m_sorted], dtype=float)
    weights /= weights.sum()
    freqs_Hz = m_sorted * J_Hz
    return freqs_Hz, weights


def element_isotopes(nuclide):
    """Every NMR-active isotope of ``nuclide``'s element in the table, as
    a list of dicts (``nuclide``, ``spin``, ``gamma`` in Hz/T,
    ``abundance`` in %), plus the spin-0 remainder of the element,
    ``100 - sum(abundances)`` %, as the second return value.

    >>> element_isotopes('11B')   # ([10B 19.9 %, 11B 80.1 %], 0.0)
    """
    from .utils import _GAMMA_DF
    element = str(nuclide).lstrip("0123456789")
    out = []
    for iso in _GAMMA_DF["Nuclide"]:
        iso = str(iso)
        if iso.lstrip("0123456789") != element:
            continue
        info = find_isotope_info(iso)
        if info["spin"] > 0 and info["abundance"] > 0:
            out.append(info)
    if not any(i["nuclide"] == nuclide for i in out):
        raise KeyError(f"'{nuclide}' is not an NMR-active isotope with a natural abundance "
                       f"in the table.")
    spin_zero = max(0.0, 100.0 - sum(i["abundance"] for i in out))
    return out, spin_zero


def _convolve_patterns(a, b):
    """Convolve two stick patterns given as {freq_Hz (rounded): weight}."""
    out = {}
    for fa, wa in a.items():
        for fb, wb in b.items():
            key = round(fa + fb, 6)
            out[key] = out.get(key, 0.0) + wa * wb
    return out


def jcoupling_group_pattern(n, J_Hz, nuclide=None, spin=None, abundance=None,
                            all_isotopes=False):
    """
    Stick pattern for one coupling *group* -- n equivalent coupling-
    partner sites of a given nuclide, each coupled with the same J --
    folding in natural isotopic abundance.

    Each of the n sites is independently NMR-active with probability
    p = abundance/100 (binomial), and an inactive site (the other,
    spin-0 isotope, e.g. 12C when nuclide='13C') contributes no
    splitting at all. The result is a mixture over k = 0..n active
    sites: weight C(n,k) p^k (1-p)^(n-k) for the k-site multiplet
    from :func:`spin_multiplet_pattern`. For an ~100%-abundant
    partner (1H, 19F, 31P, ...) this collapses to the plain n-site
    multiplet with no separate centreband; for a low-abundance
    partner (13C, 2H, ...) most of the weight sits in the k=0
    (uncoupled centreband) term, with the k>=1 terms forming small
    satellites either side of it.

    Parameters
    ----------
    n : int
        Number of equivalent coupling-partner sites of this nuclide.
    J_Hz : float
        Scalar coupling constant to this group, in Hz.
    nuclide : str, optional
        Coupling-partner nuclide, e.g. '1H', '31P', '19F', '13C'.
        Used to look up `spin`/`abundance` via
        :func:`nmrtools.utils.find_isotope_info` when those are not
        given explicitly. Required unless both `spin` and
        `abundance` are given directly.
    spin : float, optional
        Overrides the table lookup for the coupling-partner spin.
    abundance : float, optional
        Overrides the table lookup for the coupling-partner natural
        abundance (%, 0-100). Pass 100.0 to force the "always
        active, no satellites" treatment regardless of the real
        isotope (e.g. for an isotopically-enriched sample).
    all_isotopes : bool, optional
        Include *every* NMR-active isotope of the partner's element, each
        at its natural abundance, instead of treating all but `nuclide`
        as spin-0. `J_Hz` is the coupling to `nuclide`; each other
        isotope's J is scaled by the ratio of gyromagnetic ratios
        (J is proportional to gamma of the partner), and uses that
        isotope's own spin. Each of the n positions independently holds
        one isotope (or a spin-0 one), so e.g. 19F in BF4- (n=1 boron)
        gives the 11B 1:1:1:1 quartet (80.1 %) plus the 10B 1:1:1:1:1:1:1
        septet at J x gamma(10B)/gamma(11B) (19.9 %). Isotope shifts
        and quadrupolar relaxation (which often collapses couplings to
        Cl, Br, 14N...) are not modelled. Cannot be combined with
        `spin`/`abundance` overrides.

    Returns
    -------
    freqs_Hz : ndarray
        Frequency offsets (Hz), sorted ascending.
    weights : ndarray
        Relative intensities, normalised to sum to 1.
    """
    if all_isotopes:
        if nuclide is None:
            raise ValueError("all_isotopes needs the partner 'nuclide'.")
        if spin is not None or abundance is not None:
            raise ValueError("all_isotopes uses each isotope's own spin and natural "
                             "abundance; do not also give 'spin' or 'abundance'.")
        isotopes, spin_zero = element_isotopes(nuclide)
        gamma_ref = find_isotope_info(nuclide)["gamma"]
        single = {0.0: spin_zero / 100.0} if spin_zero > 0 else {}
        for iso in isotopes:
            J_iso = J_Hz * iso["gamma"] / gamma_ref
            f1, w1 = spin_multiplet_pattern(1, iso["spin"], J_iso)
            for f, w in zip(f1, w1):
                key = round(float(f), 6)
                single[key] = single.get(key, 0.0) + iso["abundance"] / 100.0 * float(w)
        totals = {0.0: 1.0}
        for _ in range(int(n)):
            totals = _convolve_patterns(totals, single)
        freqs_Hz = np.array(sorted(totals.keys()))
        weights = np.array([totals[f] for f in freqs_Hz])
        return freqs_Hz, weights / weights.sum()

    if spin is None or abundance is None:
        if nuclide is None:
            raise ValueError(
                "Provide 'nuclide' (to look up spin/abundance), or pass "
                "both 'spin' and 'abundance' explicitly."
            )
        info = find_isotope_info(nuclide)
        if spin is None:
            spin = info['spin']
        if abundance is None:
            abundance = info['abundance']

    p = abundance / 100.0

    totals = {}
    for k in range(0, n + 1):
        binom_weight = comb(n, k) * (p ** k) * ((1.0 - p) ** (n - k))
        if binom_weight <= 0.0:
            continue
        freqs_k, weights_k = spin_multiplet_pattern(k, spin, J_Hz)
        for f, w in zip(freqs_k, weights_k):
            key = round(float(f), 6)
            totals[key] = totals.get(key, 0.0) + binom_weight * float(w)

    freqs_Hz = np.array(sorted(totals.keys()))
    weights = np.array([totals[f] for f in freqs_Hz])
    weights /= weights.sum()
    return freqs_Hz, weights


def combine_jcouplings(couplings, intensity_threshold=1e-8):
    """
    Combine several independent J-coupling groups (e.g. a nucleus
    coupled to 1H *and* 31P *and* 19F simultaneously, each with its
    own n and J) into one overall stick pattern by convolving their
    individual patterns.

    Parameters
    ----------
    couplings : list of dict
        One entry per coupling group, each with keys:

        ``nuclide`` : str
            Coupling-partner nuclide, e.g. ``'1H'``, ``'31P'``,
            ``'19F'``, ``'13C'``.
        ``n`` : int
            Number of equivalent nuclei of this nuclide coupled (with
            the same J) to the observed spin.
        ``J_Hz`` : float
            Scalar coupling constant to this group, in Hz.
        ``spin`` : float, optional
            Overrides the table spin for this nuclide.
        ``abundance`` : float, optional
            Overrides the table natural abundance (%) for this
            nuclide -- see :func:`jcoupling_group_pattern`.
        ``all_isotopes`` : bool, optional
            Include every magnetic isotope of the partner's element,
            with gamma-scaled J -- see :func:`jcoupling_group_pattern`.

    intensity_threshold : float, optional
        Lines with combined relative weight below this (after
        normalising the total to 1) are dropped -- keeps the stick
        list from growing enormous when several low-abundance groups
        (each spawning many small satellite combinations) are
        combined.

    Returns
    -------
    freqs_Hz : ndarray
        Combined frequency offsets (Hz), relative to the fully
        uncoupled position, sorted ascending.
    weights : ndarray
        Combined relative intensities, normalised to sum to 1.
    """
    total_freqs = np.array([0.0])
    total_weights = np.array([1.0])

    for group in couplings:
        freqs_g, weights_g = jcoupling_group_pattern(
            n=group['n'], J_Hz=group['J_Hz'],
            nuclide=group.get('nuclide'),
            spin=group.get('spin'), abundance=group.get('abundance'),
            all_isotopes=bool(group.get('all_isotopes', False)),
        )
        new_freqs = (total_freqs[:, None] + freqs_g[None, :]).ravel()
        new_weights = (total_weights[:, None] * weights_g[None, :]).ravel()

        merged = {}
        for f, w in zip(new_freqs, new_weights):
            key = round(float(f), 6)
            merged[key] = merged.get(key, 0.0) + float(w)

        total_freqs = np.array(sorted(merged.keys()))
        total_weights = np.array([merged[f] for f in total_freqs])

    total_weights /= total_weights.sum()

    if intensity_threshold is not None:
        keep = total_weights >= intensity_threshold
        total_freqs = total_freqs[keep]
        total_weights = total_weights[keep]
        total_weights /= total_weights.sum()

    order = np.argsort(total_freqs)
    return total_freqs[order], total_weights[order]


#: Sticks evaluated per block in multiplet_lineshape, bounding its
#: temporary (block x points) array.
_MULTIPLET_BLOCK_ELEMENTS = 4_000_000


def _line_profile(dx_ppm, fwhm, lineshape, eta, normalise):
    """One line's profile at offsets ``dx_ppm`` from its centre.
    normalise='peak': height 1 at the centre; 'area': unit integral
    (per ppm). Pseudo-Voigt shares one FWHM between its parts, as
    :func:`pseudo_voigt_broadening` does: ``eta*L + (1-eta)*G``."""
    key = str(lineshape).lower().replace("-", "_")
    gamma = 0.5 * fwhm
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    lor = gamma * gamma / (dx_ppm * dx_ppm + gamma * gamma)
    gau = np.exp(-0.5 * (dx_ppm / sigma) ** 2)
    if normalise == "area":
        lor = lor / (np.pi * gamma)
        gau = gau / (sigma * np.sqrt(2.0 * np.pi))
    if key == "lorentzian":
        return lor
    if key == "gaussian":
        return gau
    if key in ("pseudo_voigt", "pv"):
        e = float(np.clip(eta, 0.0, 1.0))
        return e * lor + (1.0 - e) * gau
    raise ValueError(f"Unknown lineshape {lineshape!r}; use 'gaussian', 'lorentzian', "
                     f"or 'pseudo_voigt'.")


def multiplet_lineshape(x_ppm, stick_ppm, weights, fwhm, lineshape='lorentzian', eta=0.0,
                        normalise='area'):
    """
    Exact sum of closed-form lines, one per stick: ``sum_k w_k *
    profile(x - x_k)``, evaluated at the points ``x_ppm`` (any order, any
    spacing -- no grid needed).

    Unlike placing sticks on a grid and convolving (the ``method='grid'``
    path of :func:`jcoupled_spectrum`), every line sits exactly at its
    own position with exactly its own width, so a fitted shift or
    linewidth does not depend on where the lines fall between data
    points.

    normalise : 'area' (each line integrates to its weight, per ppm) or
        'peak' (each line's height is its weight).
    """
    x = np.asarray(x_ppm, dtype=float)
    sticks = np.asarray(stick_ppm, dtype=float)
    w = np.asarray(weights, dtype=float)
    out = np.zeros(x.shape, dtype=float)
    if sticks.size == 0 or x.size == 0:
        return out
    if not fwhm > 0:
        raise ValueError(f"fwhm must be > 0, got {fwhm}.")
    block = max(1, _MULTIPLET_BLOCK_ELEMENTS // max(x.size, 1))
    flat = x.ravel()
    acc = np.zeros(flat.size, dtype=float)
    for start in range(0, sticks.size, block):
        sl = slice(start, start + block)
        prof = _line_profile(flat[None, :] - sticks[sl, None], fwhm, lineshape, eta, normalise)
        acc += w[sl] @ prof
    return acc.reshape(x.shape)


def jcoupled_spectrum(center_ppm, larmor_Hz, couplings, ppm_axis, fwhm=2.0,
                       lineshape='lorentzian', eta=0.0, intensity_threshold=1e-8,
                       method='analytic'):
    """
    Simulate a first-order J-coupled multiplet spectrum for one
    chemical site.

    Parameters
    ----------
    center_ppm : float
        Uncoupled chemical shift of the observed site, in ppm.
    larmor_Hz : float
        Observe (Larmor) frequency of the *observed* nucleus, in Hz
        (e.g. from :func:`nmrtools.utils.compute_BF1_SFO1`) -- used
        only to convert each line's J-coupling offset (Hz) into ppm.
    couplings : list of dict
        See :func:`combine_jcouplings`.
    ppm_axis : ndarray
        Output ppm axis.
    fwhm : float, optional
        Linewidth (ppm) of each individual multiplet line. Default 2.0.
    lineshape : {'gaussian', 'lorentzian', 'pseudo_voigt'}, optional
        Broadening kernel. Default 'lorentzian' (more physically
        typical for scalar-coupled multiplets than the CSA/
        quadrupolar powder patterns elsewhere in this module, which
        default to 'gaussian').
    eta : float, optional
        Gaussian/Lorentzian mixing fraction, only used when
        ``lineshape='pseudo_voigt'``.
    intensity_threshold : float, optional
        See :func:`combine_jcouplings`.
    method : {'analytic', 'grid'}, optional
        'analytic' (default): every line is an exact closed-form profile
        at its own position (:func:`multiplet_lineshape`), scaled so the
        spectrum sums to ~1 over the axis points, as the grid method's
        does. 'grid': the original method -- sticks shared between the
        two nearest axis points, then convolved with the broadening
        kernel. The grid method needs a uniformly spaced axis, drops
        sticks outside it, and widens each line slightly (by up to the
        point spacing) in a way that depends on where it falls between
        points; fits built on it came out 2-3% narrow on typical 1H data.

    Returns
    -------
    dict with keys:

    ``x_ppm``, ``real_spec``, ``normalised_real_spec``
        As for the other ``*_spectrum`` functions in this module --
        plottable directly.
    ``stick_ppm``, ``stick_intensity``
        The underlying (unbroadened) multiplet stick positions (ppm)
        and relative intensities, useful for annotating a plot or
        checking the pattern independently of the chosen linewidth.
    """
    freqs_Hz, weights = combine_jcouplings(
        couplings, intensity_threshold=intensity_threshold,
    )
    stick_ppm = center_ppm + freqs_Hz / larmor_Hz * 1.0e6

    if method == 'analytic':
        ppm_axis = np.asarray(ppm_axis, dtype=float)
        spacing = abs(ppm_axis[1] - ppm_axis[0]) if ppm_axis.size > 1 else 1.0
        spectrum = spacing * multiplet_lineshape(
            ppm_axis, stick_ppm, weights, fwhm, lineshape=lineshape, eta=eta,
            normalise='area')
    elif method == 'grid':
        spectrum = np.zeros_like(ppm_axis, dtype=float)
        dx = ppm_axis[1] - ppm_axis[0]
        idx = (stick_ppm - ppm_axis[0]) / dx
        i0 = np.floor(idx).astype(int)
        frac = idx - i0
        valid = (i0 >= 0) & (i0 < len(ppm_axis) - 1)
        np.add.at(spectrum, i0[valid], weights[valid] * (1.0 - frac[valid]))
        np.add.at(spectrum, i0[valid] + 1, weights[valid] * frac[valid])

        spectrum = _apply_broadening(spectrum, ppm_axis, fwhm=fwhm,
                                      lineshape=lineshape, eta=eta)
    else:
        raise ValueError(f"method must be 'analytic' or 'grid', got {method!r}.")
    maximum = np.max(spectrum)
    normalised = spectrum / maximum if maximum > 0 else spectrum

    return {
        "x_ppm": ppm_axis,
        "real_spec": spectrum,
        "normalised_real_spec": normalised,
        "stick_ppm": stick_ppm,
        "stick_intensity": weights,
    }


def multi_site_jcoupled_spectrum(sites, larmor_Hz, ppm_axis, fwhm=2.0,
                                  lineshape='lorentzian', eta=0.0,
                                  intensity_threshold=1e-8, method='analytic'):
    """
    Sum :func:`jcoupled_spectrum` over several distinct chemical
    sites (e.g. a molecule with more than one resolved resonance,
    each with its own coupling network) onto one shared ppm axis.

    Parameters
    ----------
    sites : list of dict
        One entry per site, with keys:

        ``ppm`` : float
            Uncoupled chemical shift of this site.
        ``couplings`` : list of dict
            This site's coupling groups -- see
            :func:`combine_jcouplings`. Pass ``[]`` for an
            uncoupled singlet.
        ``relative_intensity`` : float, optional
            Relative weight of this site before summing (e.g.
            proportional to the number of equivalent observed nuclei
            at that site). Default 1.0.
        ``fwhm`` : float, optional
            Per-site linewidth override (ppm); falls back to the
            function's `fwhm` argument if omitted.

    larmor_Hz, ppm_axis, lineshape, eta, intensity_threshold, method
        As for :func:`jcoupled_spectrum`, shared across all sites.

    Returns
    -------
    dict with keys ``x_ppm``, ``real_spec``, ``normalised_real_spec``
    (summed over all sites) and ``sites``, a list of the individual
    per-site :func:`jcoupled_spectrum` result dicts (each still
    carrying its own ``stick_ppm``/``stick_intensity``), in case you
    want to plot or inspect them separately.
    """
    total_spectrum = np.zeros_like(ppm_axis, dtype=float)
    site_results = []

    for site in sites:
        rel_intensity = site.get('relative_intensity', 1.0)
        site_fwhm = site.get('fwhm', fwhm)
        result = jcoupled_spectrum(
            center_ppm=site['ppm'], larmor_Hz=larmor_Hz,
            couplings=site.get('couplings', []), ppm_axis=ppm_axis,
            fwhm=site_fwhm, lineshape=lineshape, eta=eta,
            intensity_threshold=intensity_threshold, method=method,
        )
        total_spectrum += rel_intensity * result['real_spec']
        site_results.append(result)

    maximum = np.max(total_spectrum)
    normalised = total_spectrum / maximum if maximum > 0 else total_spectrum

    return {
        "x_ppm": ppm_axis,
        "real_spec": total_spectrum,
        "normalised_real_spec": normalised,
        "sites": site_results,
    }
