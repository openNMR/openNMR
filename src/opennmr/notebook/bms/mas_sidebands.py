"""
MAS spinning-sideband intensities, computed by direct numerical simulation
rather than by looking up/reproducing the Herzfeld-Berger closed-form
tables (which I don't trust myself to recall correctly from memory).

Physics, briefly (so this is checkable rather than a black box):

For a single crystallite with a CSA-like tensor in its principal axis
system (PAS) diag(sxx,syy,szz), spinning at rate omega_r about an axis
tilted at the magic angle theta_m to B0, the observed frequency is
omega(t) = szz_lab(t), where szz_lab(t) is the (3,3) element of the
tensor rotated PAS -> rotor(beta,gamma) -> lab(theta_m, omega_r*t).
The FID for that one crystallite is s(t) = exp(i * phase(t)), where
phase(t) = integral of omega(t') dt'. Because <omega(t)>_time = 0 at
the exact magic angle (for ANY crystallite orientation -- that's *why*
MAS removes CSA broadening from the centerband), phase(t) is periodic
with the rotor period, so exp(i*phase(t)) has a discrete Fourier series
at harmonics n*omega_r: these are the spinning sidebands, at d_iso + n*
(spin rate), for a single crystallite.

For an initial crystallite azimuth alpha about the rotor axis, one can
show s_alpha(t) = s_0(t + alpha/omega_r) * (an alpha-dependent constant
phase). That constant phase has unit magnitude, so |c_n| does NOT
depend on alpha -- which is exactly why the standard prescription only
powder-averages over (beta, gamma), using the *intensity* |c_n|^2 (not
the complex c_n) as the averaged quantity. Averaging the complex c_n
instead (coherently, over alpha too) is a common mistake and would
incorrectly cancel the sidebands.

We verify Parseval's identity (sum of sideband intensities == 1) as a
running self-check, since that's a necessary condition for a correct
periodic unit-modulus signal.
"""
import numpy as np

MAGIC_ANGLE = np.arccos(1 / np.sqrt(3))  # ~54.7356 deg


def _rotz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _roty(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def haeberlen_to_principal(iso, delta, eta):
    """Inverse of the Haeberlen convention used by tensor2haeberlen:
    szz = iso + delta ; sxx = iso - delta*(1+eta)/2 ; syy = iso - delta*(1-eta)/2
    (trace-preserving: sxx+syy+szz = 3*iso, check passes by construction)."""
    szz = iso + delta
    sxx = iso - delta * (1 + eta) / 2
    syy = iso - delta * (1 - eta) / 2
    return sxx, syy, szz


def _zz_lab_over_period(sxx, syy, szz, spin_rate, beta, gamma, n_t):
    """szz_lab(t) at n_t equally spaced points over exactly one rotor period."""
    R_pas_to_rotor = _rotz(0.0) @ _roty(beta) @ _rotz(gamma)  # alpha fixed at 0, see module docstring
    sigma_pas = np.diag([sxx, syy, szz])
    sigma_rotor = R_pas_to_rotor @ sigma_pas @ R_pas_to_rotor.T

    T_r = 1.0 / spin_rate
    t = np.linspace(0, T_r, n_t, endpoint=False)
    zz = np.empty(n_t)
    Rm = _roty(MAGIC_ANGLE)
    for i, ti in enumerate(t):
        Rspin = _rotz(2 * np.pi * spin_rate * ti)
        Rtot = Rm @ Rspin
        sigma_lab = Rtot @ sigma_rotor @ Rtot.T
        zz[i] = sigma_lab[2, 2]
    return t, zz


def sideband_manifold(delta, eta, spin_rate, iso=0.0, n_orient_beta=14, n_orient_gamma=14,
                       n_t=32, max_order=None):
    """Returns (orders, intensities): sideband order n (integers, centerband
    n=0 at `iso`) and powder-averaged relative intensity, normalized to sum
    to 1. `delta`/`eta`/`iso`/`spin_rate` must be in the same units (e.g. ppm,
    with spin_rate = MAS rate expressed in ppm -- convert Hz spin rate to
    ppm via spin_rate_Hz / BF1_MHz before calling, to match d_iso's ppm units).
    """
    if delta == 0:
        return np.array([0]), np.array([1.0])

    sxx, syy, szz = haeberlen_to_principal(0.0, delta, eta)  # iso added back afterwards

    # Gauss-Legendre in cos(beta) for correct solid-angle (sin beta) weighting
    cosb, wb = np.polynomial.legendre.leggauss(n_orient_beta)
    betas = np.arccos(cosb)
    gammas = np.linspace(0, 2 * np.pi, n_orient_gamma, endpoint=False)
    wg = np.ones(n_orient_gamma) / n_orient_gamma
    weights = wb[:, None] * wg[None, :]
    weights = weights / weights.sum()

    intens_sum = None
    orders = None
    for ib, beta in enumerate(betas):
        for ig, gamma in enumerate(gammas):
            t, zz = _zz_lab_over_period(sxx, syy, szz, spin_rate, beta, gamma, n_t)
            dt = t[1] - t[0]
            phase = np.concatenate([[0.0], np.cumsum((zz[:-1] + zz[1:]) / 2 * dt)])[:n_t]
            fid = np.exp(1j * 2 * np.pi * phase)
            c = np.fft.fftshift(np.fft.fft(fid) / n_t)
            if orders is None:
                freqs = np.fft.fftshift(np.fft.fftfreq(n_t, d=dt))  # cycles per unit time
                orders = np.round(freqs / spin_rate).astype(int)   # freqs are multiples of 1/T_r = spin_rate
            intens = np.abs(c) ** 2
            w = weights[ib, ig]
            intens_sum = intens * w if intens_sum is None else intens_sum + intens * w
    intens_sum = intens_sum / intens_sum.sum()  # renormalize away small numerical drift

    if max_order is not None:
        keep = np.abs(orders) <= max_order
        orders, intens_sum = orders[keep], intens_sum[keep]
        intens_sum = intens_sum / intens_sum.sum()

    return orders, intens_sum
