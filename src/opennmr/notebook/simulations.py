"""
Simulate static and finite-speed-MAS powder NMR lineshapes for the
anisotropic (CSA) and quadrupolar interactions.

Workflow
--------

    CSA, static     -> csa_definition -> make_static_csa_spectrum
    CSA, spinning    -> csa_definition -> finite_mas_csa_spectrum
    Quadrupolar, static (first-order only)
                     -> static_quadrupolar_spectrum
    Quadrupolar, static (exact, all orders)
                     -> static_quadrupolar_spectrum_exact
    Quadrupolar, static (exact, decomposed into
    central/satellite and first-/second-order contributions)
                     -> static_quadrupolar_spectrum_decomposed
    Quadrupolar, spinning (explicit density-matrix propagation,
    single crystallite)
                     -> mas_quadrupolar_fid

All of the ``*_spectrum*`` functions return a dict containing at least
``x_ppm``, ``real_spec`` and ``normalised_real_spec``, so they can be
plotted directly, e.g.::

    spec = make_static_csa_spectrum(csa_definition(100, 500, 0.5,
                                                     convention='Herzfeld-Berger'),
                                     np.linspace(-1000, 1000, 5001))
    plt.plot(spec['x_ppm'], spec['real_spec'])
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d


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
# NOT DONE (flagged, not fixed)
#   - True powder-averaging efficiency (ZCW / SOPHE / REPULSION angle
#     sets) is not implemented. The rectangular (theta,phi) grid used
#     throughout is wasteful -- it over-samples near the poles and
#     needs far more points than an optimized angle set for the same
#     convergence. This matters most for finite_mas_csa_spectrum,
#     where each orientation costs a full FFT: the original defaults
#     (n_theta=1500, n_phi=3000 -> 4.5M orientations) are not tractable
#     with a per-orientation FFT and have been reduced substantially
#     below. If you want production-quality sideband simulations,
#     porting a ZCW angle set (as ssNake/SIMPSON do) is the right next
#     step, not further brute-force grid tuning.
#   - The exact-diagonalization quadrupolar functions still do one
#     np.linalg.eigh call per crystallite in a Python loop rather than
#     a single batched eigh over a stacked array of Hamiltonians.
#     eigh does support batching (input shape (N,n,n) -> eigenvalues
#     (N,n)), which would remove the Python-loop overhead; not done
#     here because the variable number of allowed transitions per
#     orientation makes the transition-selection step awkward to
#     vectorize cleanly, and I did not want to introduce a subtle
#     indexing bug under time pressure. Flagged as the next efficiency
#     step if the exact quadrupolar functions turn out to be a
#     bottleneck.
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


def static_csa(csa, ppm_axis, n_theta=1500, n_phi=3000, use_octant_symmetry=True):
    if use_octant_symmetry:
        theta, phi, wts = powder_orientations_octant(n_theta=n_theta, n_phi=n_phi)
    else:
        theta, phi, wts = powder_orientations(n_theta=n_theta, n_phi=n_phi)
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


def make_static_csa_spectrum(csa, ppm_axis, fwhm=2.0, n_theta=1500, n_phi=3000,
                              use_octant_symmetry=True):
    spec_before_broad = static_csa(csa, ppm_axis, n_theta=n_theta, n_phi=n_phi,
                                    use_octant_symmetry=use_octant_symmetry)
    spec_after_broad = gaussian_broadening(spec_before_broad, ppm_axis, fwhm=fwhm)
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
                             subtract_residual_mean=True):
    """
    NOTE ON GRID SIZE: each crystallite orientation here costs a full
    length-n_time FFT, so this cannot use the same rectangular-grid
    point counts as the static CSA case (the original defaults of
    n_theta=1500, n_phi=3000 imply ~4.5e6 FFTs). Defaults have been
    reduced to n_theta=60, n_phi=120 (7200 orientations); check
    convergence for your specific tensor/spinning-rate combination by
    increasing these and confirming sideband intensities are stable.
    A proper ZCW/SOPHE angle set would converge with far fewer points
    than either the old or new rectangular grid -- see module-level
    changelog.

    subtract_residual_mean: at the exact magic angle the gamma-average
    of the CSA modulation over one rotor period is (by construction)
    already zero after removing d_iso, so this term is normally just
    numerical hygiene. If rotor_angle != 'magic', the residual mean is
    a genuine physical shift from incomplete averaging; set this False
    if you need that effect preserved.
    """
    if mas_Hz <= 0:
        raise ValueError("mas_hz must be > 0.")
    theta, phi, wts = powder_orientations(n_theta=n_theta, n_phi=n_phi)
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
    e2 = np.cross(rotor_axis, e1)
    b0 = (np.cos(beta_rotor) * rotor_axis + np.sin(beta_rotor) * e1)
    gamma = (2.0 * np.pi * np.arange(n_time)) / n_time
    tensor = csa
    csa_iso = (csa[0, 0] + csa[1, 1] + csa[2, 2]) / 3.0
    sideband_int = np.zeros(2 * n_sidebands + 1, dtype=float)

    for k in range(len(wts)):
        axis = rotor_axis[k]
        b = b0[k]
        cross_ab = np.cross(axis, b)
        dot_ab = np.dot(axis, b)
        cos_gamma = np.cos(gamma)
        sin_gamma = np.sin(gamma)
        vectors = (b[None, :] * cos_gamma[:, None]
                   + cross_ab[None, :] * sin_gamma[:, None]
                   + axis[None, :] * dot_ab * (1.0 - cos_gamma[:, None]))
        delta = np.einsum("ni,ij,nj->n", vectors, tensor, vectors)
        modulation_Hz = (delta - csa_iso) * larmor_Hz / (1e+06)
        dt = 1.0 / (mas_Hz * n_time)
        phase = np.zeros(n_time, dtype=float)
        # phase (cycles) = integral of modulation_Hz dt (trapezoidal)
        phase[1:] = np.cumsum(0.5 * (modulation_Hz[:-1] + modulation_Hz[1:]) * dt)
        if subtract_residual_mean:
            # FIX: this term must be in the same units as `phase` above
            # (cycles), NOT radians -- previously had a stray 2*pi here.
            phase -= np.mean(modulation_Hz) * np.arange(n_time) * dt
        signal = np.exp(-1j * 2.0 * np.pi * phase)
        coeff = np.fft.fft(signal) / n_time
        coeff = np.fft.fftshift(coeff)
        centre = n_time // 2
        selected_ssb = coeff[centre - n_sidebands: centre + n_sidebands + 1]
        intensities = np.abs(selected_ssb ** 2)
        sideband_int += (wts[k] * intensities)

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
    spectrum = gaussian_broadening(spectrum, ppm_axis, fwhm=fwhm)
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

def spin_operators(I):
    """
    Return spin operators Ix, Iy, Iz for arbitrary spin I.
    Operators are dimensionless, i.e. eigenvalues of Iz are m
    rather than hbar*m.
    """
    n = int(2 * I + 1)
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


def _quad_operators(I, C_Q, eta_Q):
    """
    Build the orientation-INDEPENDENT parts of the quadrupolar problem
    once: Ix, Iy, Iz and H_Q. These do not depend on (theta, phi), so
    hoisting them out of the per-crystallite powder loop is the single
    biggest efficiency win available here (the original code rebuilt
    Ix@Ix, Iy@Iy, Iz@Iz and re-assembled H_Q from scratch at every one
    of up to ~45,000 orientations).

    H_Q/h = C_Q / [4 I (2I-1)] * [3 Iz^2 - I(I+1) + eta (Ix^2 - Iy^2)]

    FIX: denominator is 4*I*(2I-1), not 6*I*(2I-1) as in the original
    quadrupolar_hamiltonian (mas_quadrupolar_fid already had this
    right; the two were inconsistent).
    """
    if I < 1:
        raise ValueError("Quadrupolar interaction requires I >= 1.")
    if not np.isclose(2 * I, round(2 * I)):
        raise ValueError("I must be integer or half-integer.")
    if not (0.0 <= eta_Q <= 1.0):
        raise ValueError("eta_Q should normally be between 0 and 1.")
    Ix, Iy, Iz = spin_operators(I)
    n = int(2 * I + 1)
    prefactor = C_Q / (4.0 * I * (2.0 * I - 1.0))
    H_Q = prefactor * (
        3.0 * Iz @ Iz
        - I * (I + 1.0) * np.eye(n)
        + eta_Q * (Ix @ Ix - Iy @ Iy)
    )
    return Ix, Iy, Iz, H_Q


def quadrupolar_hamiltonian(I, C_Q, eta_Q, theta, phi, larmor_Hz, _ops=None):
    """
    Return the total spin Hamiltonian in Hz for one crystallite.

    _ops: optional pre-built (Ix, Iy, Iz, H_Q) tuple from
    _quad_operators, to avoid rebuilding it when called repeatedly
    over a powder grid at fixed (I, C_Q, eta_Q).
    """
    if _ops is None:
        Ix, Iy, Iz, H_Q = _quad_operators(I, C_Q, eta_Q)
    else:
        Ix, Iy, Iz, H_Q = _ops

    nx = np.sin(theta) * np.cos(phi)
    ny = np.sin(theta) * np.sin(phi)
    nz = np.cos(theta)
    Iz_lab = nx * Ix + ny * Iy + nz * Iz
    H_Z = -larmor_Hz * Iz_lab
    return H_Z + H_Q


def _transverse_basis(n):
    """Two unit vectors perpendicular to n (handles n along +/- z)."""
    ref = np.array([0.0, 0.0, 1.0])
    e1 = np.cross(n, ref)
    if np.linalg.norm(e1) < 1e-10:
        ref = np.array([1.0, 0.0, 0.0])
        e1 = np.cross(n, ref)
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    return e1, e2


def quadrupolar_transitions(I, C_Q, eta_Q, theta, phi, larmor_Hz,
                             intensity_threshold=1e-10, _ops=None):
    """
    Calculate all observable transitions for one crystallite.
    """
    if _ops is None:
        _ops = _quad_operators(I, C_Q, eta_Q)
    Ix, Iy, Iz, H_Q = _ops
    H = quadrupolar_hamiltonian(I, C_Q, eta_Q, theta, phi, larmor_Hz, _ops=_ops)
    eigenvalues, eigenvectors = np.linalg.eigh(H)

    n = np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta),
    ])
    e1, e2 = _transverse_basis(n)
    I1 = e1[0] * Ix + e1[1] * Iy + e1[2] * Iz
    I2 = e2[0] * Ix + e2[1] * Iy + e2[2] * Iz
    I1_eig = eigenvectors.conj().T @ I1 @ eigenvectors
    I2_eig = eigenvectors.conj().T @ I2 @ eigenvectors

    n_states = len(eigenvalues)
    frequencies = []
    intensities = []
    for a in range(n_states):
        for b in range(a + 1, n_states):
            frequency = eigenvalues[b] - eigenvalues[a]
            if frequency <= 0:
                continue
            matrix_element = abs(I1_eig[a, b]) ** 2 + abs(I2_eig[a, b]) ** 2
            if matrix_element < intensity_threshold:
                continue
            frequencies.append(frequency)
            intensities.append(matrix_element)
    return np.asarray(frequencies), np.asarray(intensities)


def static_quadrupolar_spectrum(I, C_Q, eta_Q, ppm_axis, larmor_Hz,
                                 n_theta=150, n_phi=300, fwhm=2.0,
                                 intensity_threshold=1e-10):
    """
    Simulate a static powder quadrupolar spectrum (first-order-only
    transition energies from the eigenbasis, but computed via the same
    exact diagonalization as static_quadrupolar_spectrum_exact -- see
    that function's docstring for the distinction from the perturbative
    decomposition used in static_quadrupolar_spectrum_decomposed).
    """
    if I < 1:
        raise ValueError("Quadrupolar interaction requires I >= 1.")
    ops = _quad_operators(I, C_Q, eta_Q)  # built once, reused every orientation
    theta, phi, wts = powder_orientations(n_theta=n_theta, n_phi=n_phi)
    spectrum = np.zeros_like(ppm_axis, dtype=float)
    for k in range(len(wts)):
        frequencies_Hz, intensities = quadrupolar_transitions(
            I=I, C_Q=C_Q, eta_Q=eta_Q,
            theta=theta[k], phi=phi[k], larmor_Hz=larmor_Hz,
            intensity_threshold=intensity_threshold, _ops=ops,
        )
        shifts_ppm = (frequencies_Hz - larmor_Hz) / larmor_Hz * 1.0e6
        for shift, intensity in zip(shifts_ppm, intensities):
            i = np.searchsorted(ppm_axis, shift)
            if i <= 0 or i >= len(ppm_axis):
                continue
            x1 = ppm_axis[i - 1]
            x2 = ppm_axis[i]
            frac = (shift - x1) / (x2 - x1)
            weight = wts[k] * intensity
            spectrum[i - 1] += weight * (1.0 - frac)
            spectrum[i] += weight * frac
    spectrum = gaussian_broadening(spectrum, ppm_axis, fwhm=fwhm)
    maximum = np.max(spectrum)
    normalised = spectrum / maximum if maximum > 0 else spectrum
    return {"x_ppm": ppm_axis, "real_spec": spectrum, "normalised_real_spec": normalised}


def quadrupolar_transitions_exact(I, C_Q, eta_Q, theta, phi, larmor_Hz,
                                   intensity_threshold=1e-10, _ops=None):
    """
    Calculate quadrupolar transitions by exact diagonalization of
    H = H_Z + H_Q, so the returned frequencies contain first-, second-,
    and higher-order quadrupolar effects (no perturbative truncation).
    """
    if _ops is None:
        _ops = _quad_operators(I, C_Q, eta_Q)
    Ix, Iy, Iz, H_Q = _ops
    H = quadrupolar_hamiltonian(I, C_Q, eta_Q, theta, phi, larmor_Hz, _ops=_ops)
    eigenvalues, eigenvectors = np.linalg.eigh(H)

    n = np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta),
    ])
    e1, e2 = _transverse_basis(n)
    I1 = e1[0] * Ix + e1[1] * Iy + e1[2] * Iz
    I2 = e2[0] * Ix + e2[1] * Iy + e2[2] * Iz
    I1_eig = eigenvectors.conj().T @ I1 @ eigenvectors
    I2_eig = eigenvectors.conj().T @ I2 @ eigenvectors

    frequencies = []
    intensities = []
    transition_indices = []
    n_states = len(eigenvalues)
    for a in range(n_states):
        for b in range(a + 1, n_states):
            frequency = eigenvalues[b] - eigenvalues[a]
            if frequency <= 0:
                continue
            intensity = np.abs(I1_eig[a, b]) ** 2 + np.abs(I2_eig[a, b]) ** 2
            if intensity < intensity_threshold:
                continue
            frequencies.append(frequency)
            intensities.append(intensity)
            transition_indices.append((a, b))
    return (np.asarray(frequencies), np.asarray(intensities),
            np.asarray(transition_indices))


def static_quadrupolar_spectrum_exact(I, C_Q, eta_Q, ppm_axis, larmor_Hz,
                                       n_theta=150, n_phi=300, fwhm=2.0,
                                       intensity_threshold=1e-10):
    """
    Static powder quadrupolar spectrum including exact first-, second-,
    and higher-order quadrupolar effects via exact diagonalization of
    H_Z + H_Q for every crystallite orientation.
    """
    if I < 1:
        raise ValueError("Quadrupolar interaction requires I >= 1.")
    if larmor_Hz <= 0:
        raise ValueError("larmor_Hz must be > 0.")
    if C_Q < 0:
        print("Warning: C_Q is negative. This is allowed, but changes the "
              "ordering of the quadrupolar principal values.")

    ops = _quad_operators(I, C_Q, eta_Q)
    theta, phi, wts = powder_orientations(n_theta=n_theta, n_phi=n_phi)
    spectrum = np.zeros_like(ppm_axis, dtype=float)

    for k in range(len(wts)):
        frequencies_Hz, intensities, transition_indices = quadrupolar_transitions_exact(
            I=I, C_Q=C_Q, eta_Q=eta_Q,
            theta=theta[k], phi=phi[k], larmor_Hz=larmor_Hz,
            intensity_threshold=intensity_threshold, _ops=ops,
        )
        shifts_ppm = (frequencies_Hz - larmor_Hz) / larmor_Hz * 1.0e6
        for shift, intensity in zip(shifts_ppm, intensities):
            i = np.searchsorted(ppm_axis, shift)
            if i <= 0 or i >= len(ppm_axis):
                continue
            x1 = ppm_axis[i - 1]
            x2 = ppm_axis[i]
            frac = (shift - x1) / (x2 - x1)
            weight = wts[k] * intensity
            spectrum[i - 1] += weight * (1.0 - frac)
            spectrum[i] += weight * frac

    spectrum = gaussian_broadening(spectrum, ppm_axis, fwhm=fwhm)
    maximum = np.max(spectrum)
    normalised = spectrum / maximum if maximum > 0 else spectrum
    return {"x_ppm": ppm_axis, "real_spec": spectrum, "normalised_real_spec": normalised}


def quadrupolar_transition_decomposition(I, C_Q, eta_Q, theta, phi, larmor_Hz,
                                          intensity_threshold=1e-10, _ops=None):
    """
    Calculate exact quadrupolar transitions and classify them by their
    dominant Zeeman quantum numbers, separating central vs satellite
    transitions and first- vs second-order(+) frequency contributions.
    """
    if _ops is None:
        _ops = _quad_operators(I, C_Q, eta_Q)
    Ix, Iy, Iz, H_Q = _ops

    n = np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta),
    ])
    I_B0 = n[0] * Ix + n[1] * Iy + n[2] * Iz

    H_Z = -larmor_Hz * I_B0
    H = H_Z + H_Q
    eigenvalues, eigenvectors = np.linalg.eigh(H)

    m_values, zeeman_vectors = np.linalg.eigh(I_B0)
    order_m = np.argsort(m_values)
    m_values = m_values[order_m]
    zeeman_vectors = zeeman_vectors[:, order_m]

    overlaps = np.abs(zeeman_vectors.conj().T @ eigenvectors) ** 2
    dominant_m_index = np.argmax(overlaps, axis=0)
    dominant_m = m_values[dominant_m_index]

    e1, e2 = _transverse_basis(n)
    I1 = e1[0] * Ix + e1[1] * Iy + e1[2] * Iz
    I2 = e2[0] * Ix + e2[1] * Iy + e2[2] * Iz
    I1_eig = eigenvectors.conj().T @ I1 @ eigenvectors
    I2_eig = eigenvectors.conj().T @ I2 @ eigenvectors

    transitions = []
    n_states = len(eigenvalues)
    for a in range(n_states):
        for b in range(a + 1, n_states):
            frequency = eigenvalues[b] - eigenvalues[a]
            if frequency <= 0:
                continue
            intensity = np.abs(I1_eig[a, b]) ** 2 + np.abs(I2_eig[a, b]) ** 2
            if intensity < intensity_threshold:
                continue
            m_lower = dominant_m[a]
            m_upper = dominant_m[b]
            delta_m = m_upper - m_lower
            if not np.isclose(abs(delta_m), 1.0):
                continue

            is_central = np.isclose(m_lower, -0.5) and np.isclose(m_upper, +0.5)
            if is_central:
                transition_type = "central"
                perturbative_order = 2
            else:
                transition_type = "satellite"
                perturbative_order = 1

            zeeman_frequency = larmor_Hz
            idx_lower = dominant_m_index[a]
            idx_upper = dominant_m_index[b]
            E1_lower = np.real(
                zeeman_vectors[:, idx_lower].conj() @ H_Q @ zeeman_vectors[:, idx_lower]
            )
            E1_upper = np.real(
                zeeman_vectors[:, idx_upper].conj() @ H_Q @ zeeman_vectors[:, idx_upper]
            )
            first_order_shift = E1_upper - E1_lower
            total_shift = frequency - zeeman_frequency
            second_order_and_higher = total_shift - first_order_shift

            transitions.append({
                "frequency_Hz": frequency,
                "intensity": intensity,
                "m_lower": m_lower,
                "m_upper": m_upper,
                "delta_m": delta_m,
                "type": transition_type,
                "order": perturbative_order,
                "frequency_zeeman_Hz": zeeman_frequency,
                "first_order_shift_Hz": first_order_shift,
                "second_order_shift_Hz": second_order_and_higher,
            })
    return transitions


def static_quadrupolar_spectrum_decomposed(I, C_Q, eta_Q, ppm_axis, larmor_Hz,
                                            n_theta=150, n_phi=300, fwhm=2.0,
                                            intensity_threshold=1e-10):
    """
    Static powder quadrupolar spectrum separated into central and
    satellite transitions, each further split into a first-order
    representation and the (exact - first-order) residual.
    """
    ops = _quad_operators(I, C_Q, eta_Q)
    theta, phi, wts = powder_orientations(n_theta=n_theta, n_phi=n_phi)

    central = np.zeros_like(ppm_axis, dtype=float)
    satellites = np.zeros_like(ppm_axis, dtype=float)
    central_first_order = np.zeros_like(ppm_axis, dtype=float)
    satellites_first_order = np.zeros_like(ppm_axis, dtype=float)

    for k in range(len(wts)):
        transitions = quadrupolar_transition_decomposition(
            I=I, C_Q=C_Q, eta_Q=eta_Q,
            theta=theta[k], phi=phi[k], larmor_Hz=larmor_Hz,
            intensity_threshold=intensity_threshold, _ops=ops,
        )
        for tr in transitions:
            frequency = tr["frequency_Hz"]
            intensity = tr["intensity"]
            first_order_shift = tr["first_order_shift_Hz"]

            shift_ppm = (frequency - larmor_Hz) / larmor_Hz * 1.0e6
            first_order_frequency = larmor_Hz + first_order_shift
            first_order_ppm = (first_order_frequency - larmor_Hz) / larmor_Hz * 1.0e6

            if tr["type"] == "central":
                target_exact = central
                target_first = central_first_order
            else:
                target_exact = satellites
                target_first = satellites_first_order

            i = np.searchsorted(ppm_axis, shift_ppm)
            if 0 < i < len(ppm_axis):
                x1 = ppm_axis[i - 1]
                x2 = ppm_axis[i]
                frac = (shift_ppm - x1) / (x2 - x1)
                weight = wts[k] * intensity
                target_exact[i - 1] += weight * (1.0 - frac)
                target_exact[i] += weight * frac

            i = np.searchsorted(ppm_axis, first_order_ppm)
            if 0 < i < len(ppm_axis):
                x1 = ppm_axis[i - 1]
                x2 = ppm_axis[i]
                frac = (first_order_ppm - x1) / (x2 - x1)
                weight = wts[k] * intensity
                target_first[i - 1] += weight * (1.0 - frac)
                target_first[i] += weight * frac

    central = gaussian_broadening(central, ppm_axis, fwhm=fwhm)
    satellites = gaussian_broadening(satellites, ppm_axis, fwhm=fwhm)
    central_first_order = gaussian_broadening(central_first_order, ppm_axis, fwhm=fwhm)
    satellites_first_order = gaussian_broadening(satellites_first_order, ppm_axis, fwhm=fwhm)

    central_second_order = central - central_first_order
    satellites_second_order = satellites - satellites_first_order
    total = central + satellites
    maximum = np.max(total)
    normalised = total / maximum if maximum > 0 else total

    return {
        "x_ppm": ppm_axis,
        "central": central,
        "satellites": satellites,
        "central_first_order": central_first_order,
        "satellites_first_order": satellites_first_order,
        "central_second_order": central_second_order,
        "satellites_second_order": satellites_second_order,
        "total": total,
        "real_spec": total,
        "normalised_real_spec": normalised,
    }


def quadrupolar_second_order_map(I, C_Q, eta_Q, larmor_Hz, n_theta=100, n_phi=200):
    """
    Calculate the exact and first-order frequencies of each transition
    over a powder grid -- useful for visualising the second-order
    central-transition shift as a function of crystallite orientation.
    """
    ops = _quad_operators(I, C_Q, eta_Q)
    theta, phi, wts = powder_orientations(n_theta=n_theta, n_phi=n_phi)
    records = []
    for k in range(len(wts)):
        transitions = quadrupolar_transition_decomposition(
            I=I, C_Q=C_Q, eta_Q=eta_Q,
            theta=theta[k], phi=phi[k], larmor_Hz=larmor_Hz, _ops=ops,
        )
        for tr in transitions:
            records.append({
                "theta": theta[k],
                "phi": phi[k],
                "weight": wts[k],
                "type": tr["type"],
                "m_lower": tr["m_lower"],
                "m_upper": tr["m_upper"],
                "exact_frequency_Hz": tr["frequency_Hz"],
                "first_order_frequency_Hz": larmor_Hz + tr["first_order_shift_Hz"],
                "second_order_shift_Hz": tr["second_order_shift_Hz"],
                "intensity": tr["intensity"],
            })
    return records


def rotation_matrix(axis, angle):
    """Rodrigues rotation matrix."""
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    K = np.array([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0],
    ])
    I3 = np.eye(3)
    return I3 + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def efg_orientation(theta, phi):
    """Rotation taking EFG PAS vectors into the laboratory frame."""
    Rz = rotation_matrix(np.array([0., 0., 1.]), phi)
    Ry = rotation_matrix(np.array([0., 1., 0.]), theta)
    return Rz @ Ry


def mas_quadrupolar_fid(I, C_Q, eta_Q, larmor_Hz, mas_Hz, theta, phi,
                         rotor_angle='magic', n_rotor_periods=32,
                         n_steps_per_rotor=256):
    """
    Explicit stepwise density-matrix propagation under MAS for a single
    crystallite. NOTE: this is the most expensive way to get a MAS FID
    -- the gamma-COMPUTE approach already used in finite_mas_csa_spectrum
    (compute the modulation analytically over one rotor period and FFT)
    is applicable to the quadrupolar case too and would avoid explicit
    propagation altogether. Flagged as a bigger structural improvement;
    not implemented here, only the propagator itself was fixed/sped up.
    """
    Ix, Iy, Iz, H_Q = _quad_operators(I, C_Q, eta_Q)
    n_states = int(2 * I + 1)

    if rotor_angle == 'magic':
        beta = np.arccos(1.0 / np.sqrt(3.0))
    else:
        beta = np.deg2rad(rotor_angle)

    rotor_axis = np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta),
    ])
    e1, e2 = _transverse_basis(rotor_axis)

    n0 = np.cos(beta) * rotor_axis + np.sin(beta) * e1
    Iz_lab0 = n0[0] * Ix + n0[1] * Iy + n0[2] * Iz
    rho = Iz_lab0.astype(complex)

    Iplus = 2.0 * (0.5 * (Ix + 1j * Iy))

    dt = 1.0 / (mas_Hz * n_steps_per_rotor)
    n_total = n_rotor_periods * n_steps_per_rotor
    signal = np.zeros(n_total, dtype=complex)

    for j in range(n_total):
        gamma = 2.0 * np.pi * j / n_steps_per_rotor
        b = (np.cos(beta) * rotor_axis
             + np.sin(beta) * (np.cos(gamma) * e1 + np.sin(gamma) * e2))
        Iz_lab = b[0] * Ix + b[1] * Iy + b[2] * Iz
        H_Z = -larmor_Hz * Iz_lab
        H = H_Z + H_Q

        # FIX/EFFICIENCY: H is Hermitian at every step, so build the
        # propagator from its eigendecomposition rather than a general
        # Pade-approximant matrix exponential (scipy.linalg.expm),
        # which was also being imported inside the innermost loop.
        evals, evecs = np.linalg.eigh(H)
        U = evecs @ np.diag(np.exp(-1j * 2.0 * np.pi * evals * dt)) @ evecs.conj().T

        rho = U @ rho @ U.conj().T
        signal[j] = np.trace(rho @ Iplus)

    return signal
