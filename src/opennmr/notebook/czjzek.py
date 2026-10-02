"""
Czjzek distributions of quadrupolar couplings: simulate the powder
lineshape of a disordered site, fit it to a spectrum, and map the
distribution of environments (C_Q across, eta_Q up, as a heat map) beside
the spectrum, fit or simulation.

The distribution (the Gaussian isotropic model; d = 5 is Czjzek's)::

    P(C_Q, eta_Q) = C_Q^(d-1) eta_Q (1 - eta_Q^2/9) / N(sigma, d)
                    * exp(-C_Q^2 (1 + eta_Q^2/3) / (2 sigma^2))

on C_Q >= 0, 0 <= eta_Q <= 1 (G. Czjzek et al., Phys. Rev. B 23, 2513
(1981); in this C_Q form, J.-B. d'Espinose de Lacaillerie, C. Fretigny,
D. Massiot, J. Magn. Reson. 192, 244 (2008)). ``sigma`` is in the units of
C_Q (MHz here). ``N`` is exact for any d:
``N = Gamma(d/2)/2 * (2 sigma^2)^(d/2) * J(d)``, with
``J(d) = int_0^1 eta (1 - eta^2/9) (1 + eta^2/3)^(-d/2) d eta`` (J(5) = 1/3,
which gives the published 2 / (sqrt(2 pi) sigma^5) for d = 5).

This distribution is centred on a *zero* EFG: it has no "central" C_Q.
Its ⟨C_Q⟩ (≈ 1.995 σ for d = 5), mode and ⟨η_Q⟩ = 0.61 all follow from σ
alone. For a site with a well-defined local structure spread by disorder,
use the extended Czjzek model (Le Caër & Brand 1998): the EFG is a mean
tensor (C_Q0, eta_Q0) plus a random Czjzek part of width σ -- give a site
``CQ0`` (and ``eta0``) and it is fitted; ``CQ0 = 0`` is plain Czjzek. The
extended model has no closed form, so it is sampled (fixed quasi-random
samples, deposited on the grid; see :func:`czjzek_samples`).

The spectrum is sum over a (C_Q, eta_Q) grid of P x cell area x the
powder lineshape of that (C_Q, eta_Q), then broadened (the isotropic
chemical-shift distribution; a Gaussian ``fwhm`` in ppm) and shifted to
``d_iso``. Two ways to get the lineshapes, chosen automatically and
named in every log:

* ``"scaling"`` -- the central transition at second order, static or
  under MAS fast enough that its spinning sidebands carry negligible
  intensity. Every CT frequency is then (C_Q)^2 x a function of eta_Q and
  orientation, so one powder set per eta_Q serves every C_Q, exactly.
  Under MAS each crystallite sits at its rotor-averaged (centreband)
  frequency.
* ``"simulated"`` -- anything else (satellites, 'exact' order, slow MAS):
  :func:`opennmr.notebook.simulations.quadrupolar_spectrum` for every
  grid point. Correct but slower (seconds static, minutes under MAS).

Workflow (notebook)::

    sim = czjzek_spectrum(2.5, sigma_MHz=3.0, larmor_Hz=130.3e6, ppm_axis=x,
                          delta_iso=60, fwhm=5)
    plot_czjzek_distribution({"sigma": 3.0}, spectrum={"x": x, "y": sim["real_spec"]})

    popt, pcov, is_free, scale, info = fit_czjzek_spectrum(
        results, [{"d_iso": 60, "sigma": 3, "fwhm": 5}])
    fig = plot_czjzek_fit(results, popt, pcov, info, layout="right")

From the GUI: Fit ▸ Quadrupolar (Czjzek) and Simulate ▸ Czjzek, which
call these functions.
"""

from __future__ import annotations

import math
import time

import numpy as np
from scipy.integrate import quad as _integrate
from scipy.optimize import curve_fit
from scipy.special import gammaln

from .fitting import build_region_mask, estimate_baseline_robust, expand_pcov, fit_quality
from .simulations import (
    QUADRUPOLAR_ORDERS, _apply_broadening, _bin_sticks, _perturbative_offsets, _quad_powder,
    _rotor_frame, _select_transitions, quadrupolar_ct_isotropic_shift_Hz,
    quadrupolar_spectrum, quadrupolar_transition_table,
)
from .utils import find_isotope_info
from ..core import fit_progress as _fit_progress

#: Per-site parameters, in ``popt`` order. ``CQ0`` and ``eta0`` (the mean
#: EFG of the extended model) come last so the first six keep their
#: places; both are 0 -- and held -- for a plain Czjzek site.
CZJZEK_PARAM_NAMES = ["d_iso", "sigma", "d", "fwhm", "amplitude", "eta", "CQ0", "eta0"]
_N = len(CZJZEK_PARAM_NAMES)
CZJZEK_PARAM_LABELS = {"d_iso": "δiso (ppm)", "sigma": "σ (MHz)", "d": "d",
                       "fwhm": "FWHM (ppm)", "amplitude": "Amplitude", "eta": "L fraction",
                       "CQ0": "C_Q0 (MHz)", "eta0": "η_Q0"}


# ============================================================
# The distribution
# ============================================================

_J_CACHE: dict = {}


def _eta_integral(d: float) -> float:
    key = round(float(d), 12)
    if key not in _J_CACHE:
        _J_CACHE[key] = _integrate(
            lambda e: e * (1 - e * e / 9) * (1 + e * e / 3) ** (-d / 2), 0.0, 1.0)[0]
    return _J_CACHE[key]


def czjzek_pdf(C_Q, eta_Q, sigma, d=5.0):
    """The normalised Czjzek probability density P(C_Q, eta_Q) (per unit
    C_Q per unit eta_Q), ``C_Q`` and ``sigma`` in the same units. Broadcasts.
    Zero outside C_Q >= 0, 0 <= eta_Q <= 1."""
    C = np.asarray(C_Q, dtype=float)
    e = np.asarray(eta_Q, dtype=float)
    sigma, d = float(sigma), float(d)
    if sigma <= 0:
        raise ValueError("sigma must be > 0.")
    if d <= 0:
        raise ValueError("d must be > 0.")
    log_norm = (gammaln(d / 2) - math.log(2.0) + (d / 2) * math.log(2 * sigma * sigma)
                + math.log(_eta_integral(d)))
    with np.errstate(divide="ignore", invalid="ignore"):
        log_p = ((d - 1) * np.log(np.where(C > 0, C, 1.0)) - C * C * (1 + e * e / 3) / (2 * sigma ** 2)
                 - log_norm)
        p = np.exp(log_p) * e * (1 - e * e / 9)
    inside = (C > 0) & (e >= 0) & (e <= 1)
    return np.where(inside, p, 0.0)


def czjzek_cq_quantile(q: float, sigma: float, d: float = 5.0, cq0: float = 0.0,
                       eta0: float = 0.0) -> float:
    """The C_Q below which a fraction ``q`` of the distribution lies
    (sampled for the extended model, ``cq0 > 0``)."""
    if cq0 and cq0 > 0:
        cq, _eta = czjzek_samples(sigma, cq0, eta0)
        return float(np.quantile(cq, q))
    c = np.linspace(0, 12 * sigma * max(1.0, math.sqrt(d / 5)), 6001)
    e = np.linspace(0, 1, 201)
    integrate = getattr(np, "trapezoid", None) or np.trapz
    marginal = integrate(czjzek_pdf(c[:, None], e[None, :], sigma, d), e, axis=1)
    cdf = np.concatenate([[0], np.cumsum(0.5 * (marginal[1:] + marginal[:-1]) * np.diff(c))])
    cdf /= cdf[-1]
    return float(np.interp(q, cdf, c))


def czjzek_grid(cq_max_MHz: float, n_cq: int = 64, n_eta: int = 20):
    """Cell edges and centres of the (C_Q, eta_Q) grid:
    ``(cq_edges, cq_centres, eta_edges, eta_centres)`` (MHz, -)."""
    cq_edges = np.linspace(0.0, float(cq_max_MHz), int(n_cq) + 1)
    eta_edges = np.linspace(0.0, 1.0, int(n_eta) + 1)
    return (cq_edges, 0.5 * (cq_edges[1:] + cq_edges[:-1]),
            eta_edges, 0.5 * (eta_edges[1:] + eta_edges[:-1]))


def czjzek_weights(sigma, d, cq_edges, eta_edges, subdivide: int = 4):
    """Probability in each grid cell, shape ``(n_eta, n_cq)``, integrated
    over the cell (``subdivide`` x ``subdivide`` midpoints), *not*
    renormalised -- the sum is the fraction of the distribution the grid
    captures."""
    s = int(subdivide)
    fc = (np.arange(s) + 0.5) / s
    dc, de = np.diff(cq_edges), np.diff(eta_edges)
    c = (cq_edges[:-1, None] + dc[:, None] * fc[None, :]).ravel()          # (n_cq*s)
    e = (eta_edges[:-1, None] + de[:, None] * fc[None, :]).ravel()         # (n_eta*s)
    p = czjzek_pdf(c[None, :], e[:, None], sigma, d)                        # (n_eta*s, n_cq*s)
    p = p.reshape(len(de), s, len(dc), s).mean(axis=(1, 3))
    return p * de[:, None] * dc[None, :]


def czjzek_statistics(sigma, d=5.0, I=None, larmor_Hz=None, n_cq=600, n_eta=200, cq0=0.0,
                      eta0=0.0, extended=None):
    """Averages over the distribution, on a fine grid: mean and most
    probable C_Q (MHz), mean eta_Q, mean P_Q = C_Q sqrt(1 + eta^2/3) (MHz),
    and -- given ``I`` and ``larmor_Hz`` -- the mean second-order
    isotropic quadrupolar shift of the CT (ppm). With a mean EFG
    (``cq0 > 0``: the extended model, d = 5) they are sample averages."""
    if _is_extended(cq0, extended):
        return _stats_from_samples(sigma, cq0, eta0, I=I, larmor_Hz=larmor_Hz)
    cq_max = czjzek_cq_quantile(0.99999, sigma, d)
    cq_edges, cq, eta_edges, eta = czjzek_grid(cq_max, n_cq, n_eta)
    w = czjzek_weights(sigma, d, cq_edges, eta_edges, subdivide=2)
    w = w / w.sum()
    C, E = np.meshgrid(cq, eta)
    marginal = w.sum(axis=0)
    out = {"mean_CQ": float(np.sum(w * C)), "mode_CQ": float(cq[int(np.argmax(marginal))]),
           "mean_eta": float(np.sum(w * E)),
           "mean_PQ": float(np.sum(w * C * np.sqrt(1 + E * E / 3))),
           "rms_PQ": float(np.sqrt(np.sum(w * C * C * (1 + E * E / 3))))}
    if I is not None and larmor_Hz and not float(I).is_integer():
        shift = quadrupolar_ct_isotropic_shift_Hz(float(I), C * 1e6, E, float(larmor_Hz))
        out["mean_shift_ppm"] = float(np.sum(w * shift) / float(larmor_Hz) * 1e6)
    return out


# ============================================================
# The extended Czjzek model: a mean EFG plus a random one
# ============================================================
#
# Le Caer & Brand (J. Phys.: Condens. Matter 10, 10715 (1998)); in NMR,
# d'Espinose de Lacaillerie et al. (2008) and Vasconcelos et al.
# (J. Phys.: Condens. Matter 25, 255402 (2013) -- not J. Chem. Phys., as
# this note said until 2026-10-01). The EFG tensor is
#
#     V = V0 + sum_i U_i S_i,   U_i ~ N(0, 3 sigma^2 / 2), i = 1..5,
#
# with V0 = diag(-C_Q0 (1 - eta0)/2, -C_Q0 (1 + eta0)/2, C_Q0) (the local,
# ordered part) and S_i an orthonormal basis of traceless symmetric
# matrices; the variance makes <C_Q^2 (1 + eta^2/3)> = 5 sigma^2 + that of
# V0, so V0 = 0 is exactly Czjzek's d = 5 distribution with the same sigma.
# The distribution has no closed form in (C_Q, eta_Q), so it is sampled:
# fixed, scrambled Sobol normal deviates (common to every call, so a fit
# sees a smooth, deterministic model), eigenvalues sorted by magnitude,
# deposited bilinearly onto the grid.

_EFG_BASIS = np.array([
    np.diag([-1.0, -1.0, 2.0]) / math.sqrt(6.0),
    np.diag([1.0, -1.0, 0.0]) / math.sqrt(2.0),
    np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=float) / math.sqrt(2.0),
    np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]], dtype=float) / math.sqrt(2.0),
    np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=float) / math.sqrt(2.0),
])

#: Samples of the extended distribution (2^16: a spectrum's weights).
N_SAMPLES = 2 ** 16

#: The papers behind the two models, as the GUI and every log cite them
#: (explicit user request, 2026-10-01: cite the original papers).
CITATIONS = {
    "czjzek": ("G. Czjzek, J. Fink, F. Götz, H. Schmidt, J. M. D. Coey, J.-P. Rebouillat and "
               "A. Liénard, Phys. Rev. B 23, 2513 (1981), doi:10.1103/PhysRevB.23.2513"),
    "czjzek_nmr": ("J.-B. d'Espinose de Lacaillerie, C. Fretigny and D. Massiot, J. Magn. Reson. "
                   "192, 244 (2008), doi:10.1016/j.jmr.2008.03.001"),
    "extended": ("G. Le Caër and R. A. Brand, J. Phys.: Condens. Matter 10, 10715 (1998), "
                 "doi:10.1088/0953-8984/10/47/020"),
    "extended_nmr": ("F. Vasconcelos, S. Cristol, J.-F. Paul, L. Delevoye, F. Mauri, T. Charpentier "
                     "and G. Le Caër, J. Phys.: Condens. Matter 25, 255402 (2013), "
                     "doi:10.1088/0953-8984/25/25/255402"),
}


def citations(extended: bool = False) -> list[str]:
    """The references for a Czjzek (``extended=False``) or extended
    Czjzek simulation or fit: the model's origin, then its NMR form."""
    keys = ("czjzek", "czjzek_nmr") + (("extended", "extended_nmr") if extended else ())
    return [CITATIONS[k] for k in keys]


class SimulationCancelled(RuntimeError):
    """Raised when a ``progress`` callback asks to stop building a library."""
_NORMAL_CACHE: dict = {}
_SAMPLE_CACHE: dict = {}


def _gaussian_samples(n: int = N_SAMPLES) -> np.ndarray:
    """``(n, 5)`` standard normal deviates, quasi-random and fixed."""
    n = int(n)
    if n not in _NORMAL_CACHE:
        from scipy.stats import norm, qmc
        u = qmc.Sobol(5, scramble=True, seed=20260929).random(n)
        _NORMAL_CACHE[n] = norm.ppf(np.clip(u, 1e-12, 1 - 1e-12))
    return _NORMAL_CACHE[n]


def czjzek_samples(sigma, cq0=0.0, eta0=0.0, n: int = N_SAMPLES):
    """``(C_Q, eta_Q)`` samples (MHz, -) of the extended Czjzek
    distribution: a mean EFG (``cq0``, ``eta0``) plus a random, Czjzek
    (d = 5) EFG of width ``sigma``. ``cq0 = 0`` samples plain Czjzek."""
    sigma, cq0, eta0 = float(sigma), float(cq0), float(eta0)
    if sigma <= 0:
        raise ValueError("sigma must be > 0.")
    if cq0 < 0 or not 0.0 <= eta0 <= 1.0:
        raise ValueError("C_Q0 must be >= 0 and eta_Q0 in [0, 1].")
    key = (round(sigma, 12), round(cq0, 12), round(eta0, 12), int(n))
    hit = _SAMPLE_CACHE.get(key)
    if hit is not None:
        return hit
    u = _gaussian_samples(n) * (math.sqrt(1.5) * sigma)
    v = np.einsum("ni,ijk->njk", u, _EFG_BASIS)
    v += np.diag([-cq0 * (1 - eta0) / 2, -cq0 * (1 + eta0) / 2, cq0])
    lam = np.linalg.eigvalsh(v)
    lam = np.take_along_axis(lam, np.argsort(np.abs(lam), axis=1), axis=1)
    vxx, vyy, vzz = lam[:, 0], lam[:, 1], lam[:, 2]
    cq = np.abs(vzz)
    with np.errstate(divide="ignore", invalid="ignore"):
        eta = np.where(cq > 0, np.abs((vxx - vyy) / np.where(cq > 0, vzz, 1.0)), 0.0)
    out = (cq, np.clip(eta, 0.0, 1.0))
    if len(_SAMPLE_CACHE) > 64:
        _SAMPLE_CACHE.clear()
    _SAMPLE_CACHE[key] = out
    return out


def _deposit(cq, eta, cq_edges, eta_edges):
    """Samples shared bilinearly among the four nearest cell centres
    (continuous in the parameters, unlike a histogram); samples beyond
    the last C_Q centre are dropped (they are the uncaptured tail).
    Returns counts / n, shape (n_eta, n_cq)."""
    n_cq, n_eta = len(cq_edges) - 1, len(eta_edges) - 1
    dc = float(cq_edges[1] - cq_edges[0])
    de = float(eta_edges[1] - eta_edges[0])
    fc = (cq - cq_edges[0]) / dc - 0.5          # in units of cells, from the first centre
    fe = (eta - eta_edges[0]) / de - 0.5
    keep = fc <= n_cq - 0.5                     # inside the last cell's edge
    fc, fe = np.clip(fc[keep], 0.0, n_cq - 1), np.clip(fe[keep], 0.0, n_eta - 1)
    i0 = np.minimum(np.floor(fc).astype(int), max(n_cq - 2, 0))
    j0 = np.minimum(np.floor(fe).astype(int), max(n_eta - 2, 0))
    tc, te = fc - i0, fe - j0
    i1, j1 = np.minimum(i0 + 1, n_cq - 1), np.minimum(j0 + 1, n_eta - 1)
    w = np.zeros(n_eta * n_cq)
    for jj, ii, ww in ((j0, i0, (1 - te) * (1 - tc)), (j0, i1, (1 - te) * tc),
                       (j1, i0, te * (1 - tc)), (j1, i1, te * tc)):
        w += np.bincount(jj * n_cq + ii, weights=ww, minlength=n_eta * n_cq)
    return w.reshape(n_eta, n_cq) / float(len(cq))


def extended_czjzek_weights(sigma, cq0, eta0, cq_edges, eta_edges, n: int = N_SAMPLES):
    """As :func:`czjzek_weights`, for the extended model (sampled): the
    probability at each grid cell, shape ``(n_eta, n_cq)``, summing to the
    fraction the grid captures."""
    cq, eta = czjzek_samples(sigma, cq0, eta0, n)
    return _deposit(cq, eta, cq_edges, eta_edges)


def _stats_from_samples(sigma, cq0, eta0, I=None, larmor_Hz=None):
    cq, eta = czjzek_samples(sigma, cq0, eta0)
    pq = cq * np.sqrt(1 + eta * eta / 3)
    hist, edges = np.histogram(cq, bins=200, range=(0.0, float(np.quantile(cq, 0.999))))
    kernel = np.array([1, 2, 3, 2, 1], dtype=float)
    smooth = np.convolve(hist, kernel / kernel.sum(), mode="same")
    k = int(np.argmax(smooth))
    out = {"mean_CQ": float(cq.mean()), "mode_CQ": float(0.5 * (edges[k] + edges[k + 1])),
           "mean_eta": float(eta.mean()), "mean_PQ": float(pq.mean()),
           "rms_PQ": float(np.sqrt(np.mean(pq * pq)))}
    if I is not None and larmor_Hz and not float(I).is_integer():
        shift = quadrupolar_ct_isotropic_shift_Hz(float(I), cq * 1e6, eta, float(larmor_Hz))
        out["mean_shift_ppm"] = float(np.mean(shift) / float(larmor_Hz) * 1e6)
    return out


def _is_extended(cq0, extended=None) -> bool:
    return bool(extended) if extended is not None else bool(cq0 and cq0 > 0)


# ============================================================
# Lineshape library
# ============================================================

def _ct_unit_positions(I, eta, larmor_Hz, mas_Hz, n_zcw, rotor_angle="magic", n_gamma=16):
    """CT offsets (Hz) of every crystallite at C_Q = 1 MHz (they scale as
    C_Q^2), and the powder weights. Under MAS, the rotor-averaged
    (centreband) offset."""
    rows = _select_transitions(I, "ct")
    m = np.array([row["m"] for row in rows])
    theta, phi, wts = _quad_powder(int(n_zcw), "zcw", 0, 0, True)
    if mas_Hz:
        beta = (np.arccos(1.0 / np.sqrt(3.0)) if rotor_angle == "magic"
                else np.deg2rad(float(rotor_angle)))
        r, e1, e2 = _rotor_frame(theta, phi)
        gam = 2.0 * np.pi * np.arange(n_gamma) / n_gamma
        b = (np.cos(beta) * r[:, None, :]
             + np.sin(beta) * (np.cos(gam)[None, :, None] * e1[:, None, :]
                               + np.sin(gam)[None, :, None] * e2[:, None, :]))
        nu = _perturbative_offsets(I, 1e6, eta, larmor_Hz, b, m, "second")[..., 0].mean(axis=1)
    else:
        b = np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi),
                      np.cos(theta)], axis=-1)
        nu = _perturbative_offsets(I, 1e6, eta, larmor_Hz, b, m, "second")[:, 0]
    return nu, wts


#: Largest share of a crystallite set's CT intensity that may sit in
#: spinning sidebands (at the distribution's 99th-percentile C_Q) for the
#: centreband C_Q^2 scaling to be used under MAS.
MAX_SIDEBAND_FRACTION = 0.02


def ct_sideband_fraction(I, C_Q_MHz, eta_Q, larmor_Hz, mas_Hz, n_zcw=610):
    """Fraction of a (C_Q, eta_Q) site's CT intensity outside its
    centreband under MAS (full finite-rate simulation, coarse powder)."""
    width = float(np.ptp(_ct_unit_positions(I, eta_Q, larmor_Hz, 0.0, 377)[0])) * C_Q_MHz ** 2
    iso = quadrupolar_ct_isotropic_shift_Hz(I, C_Q_MHz * 1e6, eta_Q, larmor_Hz)
    reach = width + 8 * mas_Hz
    x = np.linspace((iso - reach) / larmor_Hz * 1e6, (iso + reach) / larmor_Hz * 1e6, 20001)
    spec = quadrupolar_spectrum(I, C_Q_MHz * 1e6, eta_Q, larmor_Hz, x, mas_Hz=mas_Hz,
                                order="second", transitions="ct", fwhm=0.0, n_zcw=n_zcw)["real_spec"]
    inband = np.abs(x * larmor_Hz / 1e6 - iso) < 0.5 * mas_Hz
    total = float(spec.sum())
    return 1.0 - float(spec[inband].sum()) / total if total else 0.0


def choose_library_method(I, order, transitions, mas_Hz, larmor_Hz, cq_check_MHz):
    """``("scaling" | "simulated", reason)`` -- see the module notes.
    Under MAS, scaling (centreband only) is used when the CT sidebands of
    a site at ``cq_check_MHz`` (the distribution's 99th-percentile C_Q;
    eta_Q = 0.6, its mean) hold under :data:`MAX_SIDEBAND_FRACTION` of the
    intensity."""
    I = float(I)
    if I.is_integer():
        return "simulated", "an integer spin has no central transition"
    if str(transitions).lower() != "ct":
        return "simulated", "satellite transitions do not scale with C_Q^2"
    if order != "second":
        return "simulated", f"the {order} model does not scale with C_Q^2"
    if not mas_Hz:
        return "scaling", "static central transition at second order: exact C_Q^2 scaling"
    leak = ct_sideband_fraction(I, float(cq_check_MHz), 0.6, larmor_Hz, float(mas_Hz))
    if leak < MAX_SIDEBAND_FRACTION:
        return "scaling", (f"centreband C_Q^2 scaling: at C_Q = {cq_check_MHz:.3g} MHz "
                           f"(99th percentile) the CT sidebands hold {100 * leak:.2f} % at "
                           f"{mas_Hz / 1e3:g} kHz")
    return "simulated", (f"CT sidebands hold {100 * leak:.1f} % at C_Q = {cq_check_MHz:.3g} MHz "
                         f"and {mas_Hz / 1e3:g} kHz, so every grid point is simulated at the "
                         "finite spinning rate")


class CzjzekLibrary:
    """Unbroadened, unit-area powder lineshapes at delta_iso = 0 on the
    ascending, evenly spaced axis ``x_lib`` (ppm), one per grid cell, as
    ``L`` of shape ``(n_eta * n_cq, len(x_lib))`` (eta-major). A row
    holds less than unit area only where its pattern runs off ``x_lib``."""

    def __init__(self, I, larmor_Hz, x_lib, cq_max_MHz, n_cq=64, n_eta=20, mas_Hz=0.0,
                 order="second", transitions="ct", n_zcw=4181, rotor_angle="magic",
                 method="auto", cq_check_MHz=None, progress=None):
        """``progress(done, total)`` : called after every simulated
        lineshape (or every η_Q row when scaling); returning False stops
        the build with :class:`SimulationCancelled`."""
        started = time.perf_counter()
        self.I, self.larmor_Hz = float(I), float(larmor_Hz)
        self.x = np.asarray(x_lib, dtype=float)
        if self.x.size < 3 or np.any(np.diff(self.x) <= 0):
            raise ValueError("x_lib must be ascending and evenly spaced.")
        self.dx = float(self.x[1] - self.x[0])
        self.mas_Hz, self.order, self.transitions = float(mas_Hz or 0.0), order, transitions
        if order not in QUADRUPOLAR_ORDERS:
            raise ValueError(f"order must be one of {QUADRUPOLAR_ORDERS}.")
        (self.cq_edges, self.cq, self.eta_edges, self.eta) = czjzek_grid(cq_max_MHz, n_cq, n_eta)
        if method == "auto":
            method, self.reason = choose_library_method(
                self.I, order, transitions, self.mas_Hz, self.larmor_Hz,
                float(cq_check_MHz or cq_max_MHz))
        else:
            self.reason = "chosen explicitly"
        if method not in ("scaling", "simulated"):
            raise ValueError("method must be 'auto', 'scaling' or 'simulated'.")
        self.method = method
        L = np.zeros((len(self.eta), len(self.cq), self.x.size))
        def report(done, total):
            if progress is not None and progress(done, total) is False:
                raise SimulationCancelled("Stopped before every lineshape was simulated.")

        if method == "scaling":
            for i, eta in enumerate(self.eta):
                nu, wts = _ct_unit_positions(self.I, eta, self.larmor_Hz, self.mas_Hz, n_zcw,
                                             rotor_angle)
                ppm_unit = nu / self.larmor_Hz * 1e6
                norm = float(wts.sum()) * self.dx
                for j, cq in enumerate(self.cq):
                    L[i, j] = _bin_sticks(self.x, ppm_unit * cq * cq, wts) / norm
                report((i + 1) * len(self.cq), len(self.eta) * len(self.cq))
        else:
            rows = _select_transitions(self.I, transitions)
            table = quadrupolar_transition_table(self.I)
            expected = (sum(r["weight"] for r in rows) / sum(r["weight"] for r in table))
            for i, eta in enumerate(self.eta):
                for j, cq in enumerate(self.cq):
                    spec = quadrupolar_spectrum(
                        self.I, cq * 1e6, float(eta), self.larmor_Hz, self.x,
                        mas_Hz=self.mas_Hz, order=order, transitions=transitions,
                        delta_iso=0.0, fwhm=0.0, n_zcw=int(n_zcw), rotor_angle=rotor_angle)
                    L[i, j] = spec["real_spec"] / (expected * self.dx)
                    report(i * len(self.cq) + j + 1, len(self.eta) * len(self.cq))
        self.L = L.reshape(len(self.eta) * len(self.cq), self.x.size)
        self.captured = self.L.sum(axis=1).reshape(len(self.eta), len(self.cq)) * self.dx
        self.n_zcw = int(n_zcw)
        self.build_seconds = time.perf_counter() - started

    def weights(self, sigma, d=5.0, cq0=0.0, eta0=0.0, extended=None):
        """Grid-cell probabilities: the analytic Czjzek density, or -- with
        a mean EFG (``cq0 > 0``, or ``extended``) -- the sampled extended
        model (d = 5 only)."""
        if _is_extended(cq0, extended):
            if abs(float(d) - 5.0) > 1e-9:
                raise ValueError("The extended Czjzek model is defined for d = 5 only.")
            return extended_czjzek_weights(sigma, max(float(cq0), 0.0),
                                           float(np.clip(eta0, 0.0, 1.0)),
                                           self.cq_edges, self.eta_edges)
        return czjzek_weights(sigma, d, self.cq_edges, self.eta_edges)

    def spectrum(self, sigma, d=5.0, fwhm=0.0, lineshape="gaussian", eta=0.0, cq0=0.0,
                 eta0=0.0, extended=None):
        """The distribution's lineshape on ``x`` (delta_iso = 0): unit area
        for a distribution the grid and axis capture whole."""
        w = self.weights(sigma, d, cq0, eta0, extended).ravel()
        s = w @ self.L
        if fwhm and fwhm > 0:
            s = _apply_broadening(s, self.x, fwhm=float(fwhm), lineshape=lineshape,
                                  eta=float(eta))
        return s

    def captured_fraction(self, sigma, d=5.0, cq0=0.0, eta0=0.0, extended=None) -> float:
        """Of the whole distribution, the fraction this grid and axis draw."""
        return float(np.sum(self.weights(sigma, d, cq0, eta0, extended) * self.captured))

    def describe(self) -> str:
        regime = "static" if not self.mas_Hz else f"MAS {self.mas_Hz / 1e3:g} kHz"
        return (f"{len(self.cq)} × {len(self.eta)} (C_Q × η_Q) grid to C_Q = "
                f"{self.cq_edges[-1]:.3g} MHz, {regime}, {self.order} order, transitions "
                f"{self.transitions}, {self.n_zcw} orientations; lineshapes by {self.method} "
                f"({self.reason}); built in {self.build_seconds:.2f} s")


def estimate_library_seconds(I, larmor_Hz, ppm_axis, cq_max_MHz, cq_check_MHz, n_cq=64,
                             n_eta=20, mas_Hz=0.0, order="second", transitions="ct",
                             n_zcw=4181, rotor_angle="magic", method="auto"):
    """How long building a :class:`CzjzekLibrary` will take, *measured
    on this computer*: one lineshape is simulated at a representative
    C_Q and timed, then multiplied by the grid. Replaces a fixed
    seconds-per-cell table that was out by 10x and more (user report,
    2026-10-01: "told 1 minute ... crashed after about 10 mins").

    Returns ``{"method", "reason", "per_cell", "seconds", "cells"}``;
    ``seconds`` is ~0 for the "scaling" method, which simulates nothing
    per cell."""
    I = float(I)
    cells = int(n_cq) * int(n_eta)
    if method == "auto":
        method, reason = choose_library_method(I, order, transitions, mas_Hz, larmor_Hz,
                                               float(cq_check_MHz or cq_max_MHz))
    else:
        reason = "chosen explicitly"
    if method == "scaling":
        return {"method": method, "reason": reason, "per_cell": 0.0, "seconds": 0.0,
                "cells": cells}
    x = np.sort(np.asarray(ppm_axis, dtype=float))
    started = time.perf_counter()
    quadrupolar_spectrum(I, 0.6 * float(cq_max_MHz) * 1e6, 0.5, float(larmor_Hz), x,
                         mas_Hz=float(mas_Hz or 0.0), order=order, transitions=transitions,
                         delta_iso=0.0, fwhm=0.0, n_zcw=int(n_zcw), rotor_angle=rotor_angle)
    per = time.perf_counter() - started
    return {"method": method, "reason": reason, "per_cell": per, "seconds": per * cells,
            "cells": cells}


# ============================================================
# Simulation
# ============================================================

def _resolve_spin(I, nucleus):
    if I is None:
        if not nucleus:
            raise ValueError("Give I, or a nucleus.")
        I = float(find_isotope_info(nucleus)["spin"])
    I = float(I)
    if I < 1:
        raise ValueError(f"I = {I:g} has no quadrupole moment.")
    return I


def czjzek_spectrum(I, sigma_MHz, larmor_Hz, ppm_axis, d=5.0, delta_iso=0.0, fwhm=1.0,
                    lineshape="gaussian", eta=0.0, mas_Hz=0.0, order="second", transitions=None,
                    n_cq=64, n_eta=20, cq_max_MHz=None, n_zcw=4181, rotor_angle="magic",
                    method="auto", nucleus=None, cq0_MHz=0.0, eta0=0.0, progress=None):
    """
    The powder spectrum of a Czjzek-distributed site.

    ``cq0_MHz``, ``eta0`` : the mean EFG of the extended Czjzek model
    (a local, ordered coupling C_Q0, eta_Q0, plus a random Czjzek part of
    width ``sigma``; d must be 5). The plain Czjzek model (``cq0_MHz = 0``)
    is centred on a zero EFG: it has no central C_Q, and its ⟨C_Q⟩ ≈ 2σ
    comes from the width alone.

    Parameters
    ----------
    I : spin (or None with ``nucleus``). sigma_MHz : the distribution width
    (C_Q units). larmor_Hz : nu0. ppm_axis : evenly spaced, either direction.
    progress : ``progress(done, total)`` while lineshapes are built
    (see :class:`CzjzekLibrary`); return False to stop.
    d : 5 for Czjzek. delta_iso, fwhm (ppm) : isotropic shift and its
    distribution (broadening; Gaussian unless ``lineshape`` says otherwise).
    mas_Hz : 0 for static. order, transitions : as quadrupolar_spectrum
    (default: the CT for a half-integer spin). n_cq, n_eta, cq_max_MHz :
    the grid (default C_Q range: 99.99 % of the distribution).

    Returns
    -------
    dict with ``x_ppm``, ``real_spec`` (unit area over the captured
    distribution), ``normalised_real_spec``, ``sigma``, ``d``, ``cq``,
    ``eta`` (grid centres), ``weights`` ((n_eta, n_cq), sum 1),
    ``captured`` (fraction of the distribution drawn), ``stats``
    (:func:`czjzek_statistics`), ``method`` and ``log``.
    """
    I = _resolve_spin(I, nucleus)
    if transitions is None:
        transitions = "all" if I.is_integer() else "ct"
    x = np.asarray(ppm_axis, dtype=float)
    order_idx = np.argsort(x)
    xs = x[order_idx]
    cq0, eta0 = float(cq0_MHz or 0.0), float(eta0 or 0.0)
    extended = cq0 > 0
    if extended and abs(float(d) - 5.0) > 1e-9:
        raise ValueError("The extended Czjzek model (C_Q0 > 0) is defined for d = 5 only.")
    if cq_max_MHz is None:
        cq_max_MHz = czjzek_cq_quantile(0.9999, sigma_MHz, d, cq0, eta0)
    lib = CzjzekLibrary(I, larmor_Hz, xs - float(delta_iso), cq_max_MHz, n_cq=n_cq, n_eta=n_eta,
                        mas_Hz=mas_Hz, order=order, transitions=transitions, n_zcw=n_zcw,
                        rotor_angle=rotor_angle, method=method,
                        cq_check_MHz=czjzek_cq_quantile(0.99, sigma_MHz, d, cq0, eta0),
                        progress=progress)
    spec_sorted = lib.spectrum(sigma_MHz, d, fwhm, lineshape, eta, cq0, eta0)
    spec = np.empty_like(spec_sorted)
    spec[order_idx] = spec_sorted
    w = lib.weights(sigma_MHz, d, cq0, eta0)
    captured = lib.captured_fraction(sigma_MHz, d, cq0, eta0)
    peak = float(np.max(spec)) if spec.size else 0.0
    stats = czjzek_statistics(sigma_MHz, d, I=I, larmor_Hz=larmor_Hz, cq0=cq0, eta0=eta0)
    model = (f"extended Czjzek (mean EFG C_Q0 = {cq0:g} MHz, η_Q0 = {eta0:g}, plus a random part "
             f"of σ = {sigma_MHz:g} MHz; {N_SAMPLES} samples)" if extended else
             f"Czjzek (random EFG centred on zero), σ = {sigma_MHz:g} MHz, d = {d:g}")
    log = (f"Czjzek simulation: I = {I:g}, ν0 = {larmor_Hz / 1e6:.6g} MHz, {model}, "
           f"δiso = {delta_iso:g} ppm, FWHM {fwhm:g} ppm ({lineshape}); "
           f"⟨C_Q⟩ = {stats['mean_CQ']:.3g} MHz, ⟨η_Q⟩ = {stats['mean_eta']:.3f}; "
           + lib.describe() + f"; {100 * captured:.2f} % of the distribution drawn. "
           + "Model: " + "; ".join(citations(extended)) + ".")
    return {"x_ppm": x, "real_spec": spec, "normalised_real_spec": spec / peak if peak else spec,
            "sigma": float(sigma_MHz), "d": float(d), "delta_iso": float(delta_iso),
            "cq0": cq0, "eta0": eta0, "extended": extended,
            "cq": lib.cq, "eta": lib.eta, "cq_edges": lib.cq_edges, "eta_edges": lib.eta_edges,
            "weights": w / w.sum() if w.sum() else w, "captured": captured, "stats": stats,
            "method": lib.method, "log": log, "citations": citations(extended)}


# ============================================================
# Fitting
# ============================================================

_STAT_KEYS = ("mean_CQ", "mode_CQ", "mean_eta", "mean_PQ", "rms_PQ", "mean_shift_ppm")


def _extended_stats_with_errors(sigma, cq0, eta0, pcov, idx, I, larmor_Hz):
    """Sample averages of the extended model, with 1σ errors propagated
    from (σ, C_Q0, η_Q0)'s covariance by a central-difference Jacobian
    (on the same fixed samples, so the differences are not noise)."""
    st = _stats_from_samples(sigma, cq0, eta0, I=I, larmor_Hz=larmor_Hz)
    keys = [k for k in _STAT_KEYS if k in st and k != "mode_CQ"]
    point = np.array([sigma, cq0, eta0], dtype=float)
    lo = np.array([1e-6, 0.0, 0.0])
    hi = np.array([np.inf, np.inf, 1.0])
    J = np.zeros((len(keys), 3))
    for j in range(3):
        h = 1e-3 * max(abs(point[j]), 1e-2 * max(sigma, cq0, 1e-3))
        up, down = point.copy(), point.copy()
        up[j] = min(point[j] + h, hi[j])
        down[j] = max(point[j] - h, lo[j])
        if up[j] == down[j]:
            continue
        a = _stats_from_samples(*up, I=I, larmor_Hz=larmor_Hz)
        b = _stats_from_samples(*down, I=I, larmor_Hz=larmor_Hz)
        J[:, j] = [(a[k] - b[k]) / (up[j] - down[j]) for k in keys]
    sub = np.asarray(pcov)[np.ix_(idx, idx)]
    var = np.einsum("ki,ij,kj->k", J, sub, J)
    for k, v in zip(keys, var):
        st[k + "_err"] = float(np.sqrt(max(v, 0.0)))
    # The mode is a histogram peak: scale its error as the mean's.
    st["mode_CQ_err"] = (st["mean_CQ_err"] * st["mode_CQ"] / st["mean_CQ"]
                         if st["mean_CQ"] else float("nan"))
    return st

def _normalise_sites(p0):
    sites = [dict(p0)] if isinstance(p0, dict) else [dict(s) for s in p0]
    if not sites:
        raise ValueError("p0 needs at least one site.")
    labels = []
    for k, site in enumerate(sites):
        for key in ("d_iso", "sigma", "fwhm"):
            if key not in site:
                raise ValueError(f"Site {k + 1} needs {key!r}.")
        labels.append(str(site.get("label") or f"Site {k + 1}"))
    return sites, labels


def _site_fixed(site, fixed, index, label):
    """Names fixed for one site: d is fixed at its start (5) unless freed."""
    out = {"d"}
    for source in (fixed if isinstance(fixed, (list, tuple, set)) else None,
                   (fixed or {}).get(index) if isinstance(fixed, dict) else None,
                   (fixed or {}).get(label) if isinstance(fixed, dict) else None,
                   site.get("fixed")):
        if not source:
            continue
        items = source.items() if isinstance(source, dict) else ((n, True) for n in source)
        for name, on in items:
            if name not in CZJZEK_PARAM_NAMES:
                raise ValueError(f"Unknown parameter {name!r} in fixed; use {CZJZEK_PARAM_NAMES}.")
            (out.add if on else out.discard)(name)
    return out


def _legacy_params(params, n_sites):
    """A six-per-site ``popt`` (before C_Q0, eta_Q0) as eight-per-site."""
    params = np.asarray(params, dtype=float)
    if params.size == 6 * n_sites + 2:
        sites = np.concatenate([params[:6 * n_sites].reshape(n_sites, 6),
                                np.zeros((n_sites, 2))], axis=1).ravel()
        params = np.concatenate([sites, params[6 * n_sites:]])
    return params


def czjzek_model(x_sorted, params, n_sites, library, x_ref, lineshape="gaussian",
                 extended=None):
    """``sum_i amplitude_i * S_i(x - d_iso_i) / max(S_i) + b0 + b1 (x - x_ref)``,
    and the per-site curves. ``params`` as :data:`CZJZEK_PARAM_NAMES` per
    site, then (b0, b1). ``extended`` : per site, whether it is the
    extended model (default: where C_Q0 > 0)."""
    params = _legacy_params(params, n_sites)
    total = params[_N * n_sites] + params[_N * n_sites + 1] * (x_sorted - x_ref)
    sites = []
    ext = list(extended) if extended is not None else [None] * n_sites
    for k, (d_iso, sigma, d, fwhm, amp, eta, cq0, eta0) in enumerate(
            params[:_N * n_sites].reshape(n_sites, _N)):
        s = library.spectrum(max(sigma, 1e-6), max(d, 0.5), max(fwhm, 0.0), lineshape,
                             float(np.clip(eta, 0, 1)), max(cq0, 0.0), float(np.clip(eta0, 0, 1)),
                             ext[k])
        peak = float(np.max(s)) or 1.0
        curve = amp * np.interp(x_sorted - d_iso, library.x, s, left=0.0, right=0.0) / peak
        sites.append(curve)
        total = total + curve
    return total, sites


def fit_czjzek_spectrum(results, p0, I=None, nucleus=None, larmor_Hz=None, mas_Hz=0.0,
                        order="second", transitions=None, rotor_angle="magic", n_zcw=4181,
                        lineshape="gaussian", fit_regions=None, baseline_regions=None,
                        bounds=None, fixed=None, fit_baseline=False, n_cq=64, n_eta=20,
                        cq_max_MHz=None, method="auto", refine_grid=True, progress=None):
    """
    Fit one or more Czjzek-distributed quadrupolar sites to a 1D spectrum.
    ``progress(done, total)`` is called while each lineshape library is
    built (see :class:`CzjzekLibrary`); returning False stops the fit.

    ``p0`` : dict or list of dicts ``{"d_iso": ppm, "sigma": MHz, "fwhm": ppm,
    ["d"]: 5, ["amplitude"], ["eta"] (L fraction), ["label"], ["fixed"],
    ["bounds"], ["CQ0"], ["eta0"], ["model"]}``. ``d`` is held at its start
    (5, Czjzek's value) unless a site's ``fixed`` says ``{"d": False}``.

    A site with ``CQ0`` > 0 (MHz), or ``"model": "extended"``, is the
    extended Czjzek model: a mean EFG (C_Q0, eta_Q0, both fitted) plus a
    random Czjzek part of width sigma; d is then 5. Without it, C_Q0 and
    eta_Q0 are 0 and held: plain Czjzek, a random EFG centred on zero. ``fixed``/``bounds`` may also be
    global: a list of names, or ``{site index or label: ...}`` /
    ``{name: (lo, hi)}``.

    The (C_Q, eta_Q) grid runs to the C_Q holding 99.99 % of the widest
    distribution the bounds allow; with ``refine_grid`` the fit is then
    repeated once on a grid sized to the fitted sigma, so the result does
    not depend on how wide the bounds were.

    Returns ``popt, pcov_free, is_free, scale, fit_info``, as
    :func:`opennmr.notebook.quadrupolar_fitting.fit_quadrupolar_spectrum`
    (``popt``: :data:`CZJZEK_PARAM_NAMES` per site, then b0, b1; amplitudes
    in raw units). ``fit_info`` also holds ``library`` (the last
    :class:`CzjzekLibrary`), ``stats`` per site and ``components``-ready
    settings.
    """
    started = time.perf_counter()
    nucleus = nucleus or results.get("nuclide")
    I = _resolve_spin(I, nucleus)
    if larmor_Hz is None:
        larmor_Hz = float(results["SFO1"]) * 1e6
    if transitions is None:
        transitions = "all" if I.is_integer() else "ct"
    lineshape = str(lineshape).lower().replace("-", "_")
    if lineshape == "pv":
        lineshape = "pseudo_voigt"
    mas_Hz = float(mas_Hz or 0.0)
    sites, labels = _normalise_sites(p0)
    n_sites = len(sites)

    x_full = np.asarray(results["x_ppm"], dtype=float)
    y_raw = np.asarray(results["real_spectrum"], dtype=float)
    scale = float(np.max(np.abs(y_raw))) or 1.0
    idx = np.argsort(x_full)
    x_sorted, y_sorted = x_full[idx], y_raw[idx] / scale
    dx = float(np.median(np.diff(x_sorted)))
    mask = build_region_mask(x_sorted, fit_regions)
    x_fit, y_fit = x_sorted[mask], y_sorted[mask]
    if x_fit.size < 8:
        raise ValueError("The fit window holds fewer than 8 points.")
    x_ref = float(np.mean(x_fit))
    # Everything the fit needs lies inside the fit window: the default
    # d_iso/FWHM bounds and the lineshape library are sized to it, not to
    # the whole spectrum. (Sized to the whole spectrum, a 9 ppm window on a
    # 3700 ppm-wide 23Na spectrum built a 2^19-point x 1280-lineshape
    # library -- ~5 GB -- and was killed.)
    span = float(x_fit[-1] - x_fit[0])
    b0, b1, _ = estimate_baseline_robust(x_sorted, y_sorted, baseline_regions=baseline_regions,
                                         x_ref=x_ref)

    default_eta = 0.5 if lineshape == "pseudo_voigt" else (1.0 if lineshape == "lorentzian" else 0.0)
    values, is_free, lower, upper = [], [], [], []
    global_bounds = bounds if isinstance(bounds, dict) and all(
        k in CZJZEK_PARAM_NAMES for k in bounds) else {}
    extended = []
    for k, (site, label) in enumerate(zip(sites, labels)):
        sigma0 = float(site["sigma"])
        cq00 = float(site.get("CQ0", 0.0) or 0.0)
        ext = str(site.get("model", "")).lower() == "extended" or cq00 > 0
        extended.append(ext)
        amp = (float(site["amplitude"]) / scale if "amplitude" in site
               else max(float(np.max(y_fit) - b0), 1e-3) / n_sites)
        start = {"d_iso": float(site["d_iso"]), "sigma": sigma0, "d": float(site.get("d", 5.0)),
                 "fwhm": float(site["fwhm"]), "amplitude": amp,
                 "eta": float(site.get("eta", default_eta)), "CQ0": cq00 if ext else 0.0,
                 "eta0": float(site.get("eta0", 0.0) or 0.0) if ext else 0.0}
        default_bounds = {"d_iso": (start["d_iso"] - span, start["d_iso"] + span),
                          "sigma": (1e-3, max(3.0 * sigma0, 0.5)), "d": (1.0, 20.0),
                          "fwhm": (0.0, span), "amplitude": (0.0, np.inf), "eta": (0.0, 1.0),
                          "CQ0": (0.0, max(3.0 * cq00, 5.0 * sigma0, 1.0)), "eta0": (0.0, 1.0)}
        per_site = {}
        if isinstance(bounds, dict):
            per_site = bounds.get(k) or bounds.get(label) or {}
        fixed_names = _site_fixed(site, fixed, k, label)
        if ext:
            if "d" not in fixed_names or abs(start["d"] - 5.0) > 1e-9:
                raise ValueError(f"{label}: the extended Czjzek model (a mean EFG C_Q0 plus a "
                                 "random part) is defined for d = 5 only; hold d at 5.")
        else:
            fixed_names |= {"CQ0", "eta0"}
        if lineshape != "pseudo_voigt":
            fixed_names.add("eta")
            start["eta"] = default_eta
        for name in CZJZEK_PARAM_NAMES:
            lo, hi = (site.get("bounds", {}).get(name) or per_site.get(name)
                      or global_bounds.get(name) or default_bounds[name])
            lo, hi = float(lo), float(hi)
            if name == "amplitude" and "amplitude" in (site.get("bounds") or per_site or
                                                       global_bounds):
                lo, hi = lo / scale, hi / scale
            values.append(start[name])
            is_free.append(name not in fixed_names)
            lower.append(lo)
            upper.append(hi)
    values += [b0, b1]
    is_free += [bool(fit_baseline)] * 2
    lower += [-np.inf, -np.inf]
    upper += [np.inf, np.inf]
    p0_arr, is_free = np.array(values), np.array(is_free)
    lower, upper = np.array(lower), np.array(upper)
    notes = []
    names = [f"{lab}: {n}" for lab in labels for n in CZJZEK_PARAM_NAMES] + ["b0", "b1"]
    for i in np.flatnonzero(is_free):
        if not lower[i] <= p0_arr[i] <= upper[i]:
            moved = float(np.clip(p0_arr[i], lower[i], upper[i]))
            notes.append(f"{names[i]}: starting value {p0_arr[i]:g} is outside its bounds; "
                         f"started at {moved:g}")
            p0_arr[i] = moved

    diso_idx = np.arange(n_sites) * _N
    sigma_idx = diso_idx + 1
    d_idx = diso_idx + 2
    cq0_idx = diso_idx + 6
    eta0_idx = diso_idx + 7

    def library_for(sigmas, ds, check_sigmas, cq0s, eta0s):
        cq_max = cq_max_MHz or max(
            czjzek_cq_quantile(0.9999, s, dd, c0, e0) * (1.1 if c0 > 0 else 1.0)
            for s, dd, c0, e0 in zip(sigmas, ds, cq0s, eta0s))
        cq_check = max(czjzek_cq_quantile(0.99, s, dd, c0, e0)
                       for s, dd, c0, e0 in zip(check_sigmas, ds, cq0s, eta0s))
        lo_d = float(np.min(np.where(is_free[diso_idx], lower[diso_idx], p0_arr[diso_idx])))
        hi_d = float(np.max(np.where(is_free[diso_idx], upper[diso_idx], p0_arr[diso_idx])))
        pad = 0.1 * span + 5 * dx
        lo, hi = x_fit[0] - hi_d - pad, x_fit[-1] - lo_d + pad
        n = int(round((hi - lo) / dx)) + 1
        if n > 2 ** 19:
            n = 2 ** 19
        x_lib = np.linspace(lo, hi, n)
        return CzjzekLibrary(I, larmor_Hz, x_lib, cq_max, n_cq=n_cq, n_eta=n_eta, mas_Hz=mas_Hz,
                             order=order, transitions=transitions, n_zcw=n_zcw,
                             rotor_angle=rotor_angle, method=method, cq_check_MHz=cq_check,
                             progress=progress)

    def run(library, start):
        def model(_x, *free):
            full = start.copy()
            full[is_free] = free
            return czjzek_model(x_fit, full, n_sites, library, x_ref, lineshape,
                                extended)[0]
        popt_free, pcov_free = curve_fit(_fit_progress.tracked(model, y_fit, n_free=int(is_free.sum())),
                                         x_fit, y_fit, p0=start[is_free],
                                         bounds=(lower[is_free], upper[is_free]),
                                         diff_step=1e-4, maxfev=5000)
        full = start.copy()
        full[is_free] = popt_free
        return full, np.asarray(pcov_free)

    # Stage 1: a grid wide enough for every sigma the bounds allow.
    sig_hi = np.where(is_free[sigma_idx], upper[sigma_idx], p0_arr[sigma_idx])
    d_hi = np.where(is_free[d_idx], upper[d_idx], p0_arr[d_idx])
    cq0_hi = np.where(is_free[cq0_idx], upper[cq0_idx], p0_arr[cq0_idx])
    library = library_for(sig_hi, np.maximum(d_hi, p0_arr[d_idx]), 1.5 * p0_arr[sigma_idx],
                          cq0_hi, p0_arr[eta0_idx])
    _fit_progress.set_stage("fitting on the first grid")
    params, pcov_free = run(library, p0_arr)
    grid_log = library.describe()
    if refine_grid and cq_max_MHz is None:
        # Stage 2: a grid sized to the answer, started from it.
        library = library_for(1.5 * params[sigma_idx], params[d_idx], params[sigma_idx],
                              params[cq0_idx], params[eta0_idx])
        _fit_progress.set_stage("fitting on a grid refined to the fitted σ")
        params, pcov_free = run(library, params)
        grid_log = library.describe() + " (refined to the fitted σ)"

    unit = np.ones(len(params))
    amp_idx = diso_idx + 4
    unit[amp_idx] = scale
    unit[-2:] = scale
    popt = params * unit
    pcov_free = pcov_free * np.outer(unit[is_free], unit[is_free])
    pcov = expand_pcov(pcov_free, is_free, len(popt))

    warnings = list(notes)
    stats = []
    for s, label in enumerate(labels):
        sigma, d = float(popt[sigma_idx[s]]), float(popt[d_idx[s]])
        cq0, eta0 = float(popt[cq0_idx[s]]), float(popt[eta0_idx[s]])
        if extended[s]:
            st = _extended_stats_with_errors(sigma, cq0, eta0, pcov,
                                             (sigma_idx[s], cq0_idx[s], eta0_idx[s]), I, larmor_Hz)
        else:
            st = czjzek_statistics(sigma, d, I=I, larmor_Hz=larmor_Hz)
            # Every average scales with sigma at fixed d: propagate exactly.
            err_sigma = float(np.sqrt(max(pcov[sigma_idx[s], sigma_idx[s]], 0.0)))
            for key in ("mean_CQ", "mode_CQ", "mean_PQ", "rms_PQ"):
                st[key + "_err"] = st[key] / sigma * err_sigma if sigma else float("nan")
            st["mean_eta_err"] = 0.0 if not is_free[d_idx[s]] else float("nan")
        st["extended"] = bool(extended[s])
        stats.append(st)
        # Of the site's whole (unit-area) lineshape, what lies inside the
        # spectrum's own range once shifted to its d_iso.
        shape = library.spectrum(sigma, d, float(popt[diso_idx[s] + 3]), lineshape,
                                 float(np.clip(popt[diso_idx[s] + 5], 0, 1)), cq0, eta0,
                                 extended[s])
        shift = float(popt[diso_idx[s]])
        inside = (library.x + shift >= x_fit[0]) & (library.x + shift <= x_fit[-1])
        captured = float(shape[inside].sum() * library.dx)
        if captured < 0.99:
            where = "spectrum" if fit_regions is None else "fit window"
            warnings.append(f"{label}: only {100 * captured:.1f} % of the distribution falls "
                            f"inside the {where} -- its large-C_Q tail runs off it, so the "
                            f"wings are not fitted; widen the {where} or check σ.")
        for i, name in ((sigma_idx[s], "σ"), (diso_idx[s], "δiso"), (d_idx[s], "d")):
            if is_free[i] and (np.isclose(popt[i] / (unit[i]), lower[i], rtol=1e-3, atol=1e-9)
                               or np.isclose(popt[i] / unit[i], upper[i], rtol=1e-3, atol=1e-9)):
                warnings.append(f"{label}: {name} ended on a bound -- not a free minimum.")
        if extended[s] and is_free[cq0_idx[s]]:
            if cq0 <= 1e-3 * max(sigma, 1e-9):
                warnings.append(f"{label}: C_Q0 went to 0 -- the data need no mean EFG, and "
                                "plain Czjzek (fewer parameters) describes them as well.")
            elif np.isclose(cq0, upper[cq0_idx[s]], rtol=1e-3):
                warnings.append(f"{label}: C_Q0 ended on its upper bound -- not a free minimum.")
            if sigma < 0.1 * cq0:
                warnings.append(f"{label}: σ is under a tenth of C_Q0 -- the site is nearly "
                                "ordered; a single (C_Q, η_Q) site with a C_Q spread may be the "
                                "simpler description.")
        lo_q = czjzek_cq_quantile(0.01, sigma, d, cq0, eta0) if extended[s] else 0.0
        hi_q = czjzek_cq_quantile(0.99, sigma, d, cq0, eta0)
        n_cells = int(np.sum((library.cq_edges[1:] <= hi_q) & (library.cq_edges[:-1] >= lo_q)))
        if n_cells < 15:
            warnings.append(f"{label}: only {n_cells} C_Q cells cover the distribution; "
                            "raise n_cq for a smoother model.")
    if not fit_baseline:
        # The held baseline came from the spectrum's outer 5 % at each end (or
        # baseline_regions). If the fitted pattern still has intensity there,
        # part of its tail was taken as baseline, and sigma, FWHM and the
        # amplitude absorb the error -- silently, with small error bars (a
        # static 27Al site with 1.5 % of its height at one end of the axis gave
        # sigma 3 % low and FWHM 12 % high, each tens of sigma off). The
        # quadrupolar and combined fitters already warn about this.
        if baseline_regions is not None:
            regions = ([baseline_regions] if np.isscalar(baseline_regions[0])
                       else baseline_regions)
            zone = build_region_mask(x_sorted, regions)
        else:
            k = max(1, int(x_sorted.size * 0.05))
            zone = np.zeros(x_sorted.size, dtype=bool)
            zone[:k] = zone[-k:] = True
        _total, site_curves = czjzek_model(x_sorted, params, n_sites, library, x_ref, lineshape,
                                           extended)
        pattern = np.sum(site_curves, axis=0)
        peak = float(np.max(np.abs(pattern[mask]))) if mask.any() else 0.0
        if peak > 0 and zone.any():
            inside = float(np.max(np.abs(pattern[zone]))) / peak
            if inside > 0.01:
                warnings.append(
                    f"The baseline was estimated where the fitted pattern still has {inside:.1%} "
                    "of its height (the spectrum's outer 5% at each end, or baseline_regions): "
                    "it is partly the pattern's tail, which biases σ, FWHM and the amplitude. "
                    "Give baseline_regions clear of the pattern, widen the spectrum, or fit the "
                    "baseline (fit_baseline=True).")
    regime = "static" if not mas_Hz else f"MAS {mas_Hz / 1e3:g} kHz"
    log = (f"Czjzek fit: I = {I:g}, ν0 = {larmor_Hz / 1e6:.6g} MHz, {regime}, {n_sites} site(s), "
           f"{lineshape} CS-distribution broadening, window "
           + ("whole spectrum" if fit_regions is None else
              ", ".join(f"{max(a, b):g} to {min(a, b):g} ppm" for a, b in fit_regions))
           + f"; {grid_log}.")
    if any(extended):
        log += (f" Extended Czjzek for {', '.join(l for l, e in zip(labels, extended) if e)}: "
                f"mean EFG (C_Q0, η_Q0) plus a random part of width σ, sampled "
                f"({N_SAMPLES} fixed quasi-random samples).")
    log += " Model: " + "; ".join(citations(any(extended))) + "."
    fit_info = {"labels": labels, "n_sites": n_sites, "x_ref": x_ref, "library": library,
                "extended": [bool(e) for e in extended],
                "citations": citations(any(extended)),
                "lineshape": lineshape, "stats": stats, "warnings": warnings, "log": log,
                "fit_regions": (None if fit_regions is None else
                                [(float(min(a, b)), float(max(a, b))) for a, b in fit_regions]),
                "settings": dict(I=I, larmor_Hz=float(larmor_Hz), mas_Hz=mas_Hz, order=order,
                                 transitions=transitions, rotor_angle=rotor_angle,
                                 n_zcw=int(n_zcw), lineshape=lineshape, n_cq=int(n_cq),
                                 n_eta=int(n_eta), method=library.method),
                "elapsed": time.perf_counter() - started,
                "lower": lower * unit, "upper": upper * unit}
    return popt, pcov_free, is_free, scale, fit_info


def evaluate_czjzek_fit(x_ppm, popt, fit_info):
    """``(total, components)`` on ``x_ppm`` (any order, returned in that
    order); ``components`` is one curve per site then the baseline, as
    the other fitters' ``return_components``."""
    x = np.asarray(x_ppm, dtype=float)
    order = np.argsort(x)
    xs = x[order]
    n = fit_info["n_sites"]
    params = np.asarray(popt, dtype=float).copy()
    params = _legacy_params(params, n)
    total, sites = czjzek_model(xs, params, n, fit_info["library"], fit_info["x_ref"],
                                fit_info["lineshape"], fit_info.get("extended"))
    baseline = params[_N * n] + params[_N * n + 1] * (xs - fit_info["x_ref"])
    back = np.empty_like(order)
    back[order] = np.arange(order.size)
    return total[back], [c[back] for c in sites] + [baseline[back]]


def czjzek_fit_report(popt, pcov, is_free, fit_info) -> list[dict]:
    """One dict per site: every parameter with its 1σ error and whether it
    was fixed, plus the distribution's averages (⟨C_Q⟩ etc.)."""
    n = fit_info["n_sites"]
    popt = _legacy_params(popt, n)
    if np.asarray(pcov).shape[0] == 6 * n + 2:
        keep = [i for i in range(_N * n + 2) if i >= _N * n or i % _N < 6]
        full = np.zeros((_N * n + 2, _N * n + 2))
        full[np.ix_(keep, keep)] = pcov
        pcov = full
        is_free = np.array([bool(is_free[keep.index(i)]) if i in keep else False
                            for i in range(_N * n + 2)])
    errors = np.sqrt(np.clip(np.diag(pcov), 0.0, None))
    rows = []
    for s, label in enumerate(fit_info["labels"]):
        row = {"label": label, "extended": bool((fit_info.get("extended") or [False] * n)[s])}
        for k, name in enumerate(CZJZEK_PARAM_NAMES):
            i = s * _N + k
            row[name] = float(popt[i])
            row[name + "_err"] = float(errors[i])
            row[name + "_fixed"] = not bool(is_free[i])
        row.update(fit_info["stats"][s])
        if "mean_shift_ppm" in row:
            row["delta_cg"] = row["d_iso"] + row["mean_shift_ppm"]
        rows.append(row)
    return rows


# ============================================================
# The distribution map
# ============================================================

#: Samples behind an extended site's map (a finer grid than a spectrum's).
N_MAP_SAMPLES = 2 ** 19


def _site_cq0(site):
    return float(site.get("CQ0", site.get("cq0", 0.0)) or 0.0)


def _site_density(site, cq_edges, eta_edges):
    """One site's density on the grid's cell centres, shape (n_eta, n_cq),
    per unit C_Q per unit eta_Q: analytic for plain Czjzek; for the
    extended model, sampled, deposited and lightly smoothed (one cell)."""
    cq = 0.5 * (cq_edges[1:] + cq_edges[:-1])
    eta = 0.5 * (eta_edges[1:] + eta_edges[:-1])
    cq0 = _site_cq0(site)
    if cq0 <= 0:
        return czjzek_pdf(cq[None, :], eta[:, None], float(site["sigma"]), float(site.get("d", 5)))
    from scipy.ndimage import gaussian_filter
    c, e = czjzek_samples(float(site["sigma"]), cq0,
                          float(site.get("eta0", site.get("eta_Q0", 0.0)) or 0.0),
                          N_MAP_SAMPLES)
    w = gaussian_filter(_deposit(c, e, cq_edges, eta_edges), 1.0, mode="nearest")
    return w / (np.diff(cq_edges)[None, :] * np.diff(eta_edges)[:, None])


def czjzek_mixture_map(sites, cq_max_MHz=None, n_cq=200, n_eta=100):
    """The (C_Q, eta_Q) density of one or several Czjzek sites, each
    ``{"sigma", ["d"], ["CQ0"], ["eta0"], ["weight"], ["label"]}``
    (``CQ0`` > 0: the extended model, sampled; weights: e.g. site areas,
    normalised here). Returns ``(cq_edges, eta_edges, density)`` with
    ``density`` shape (n_eta, n_cq), per unit C_Q per unit eta_Q."""
    sites = [sites] if isinstance(sites, dict) else list(sites)
    if not sites:
        raise ValueError("No sites to map.")
    weights = np.array([float(s.get("weight", 1.0)) for s in sites])
    if not np.all(weights >= 0) or weights.sum() <= 0:
        raise ValueError("Site weights must be non-negative and not all zero.")
    weights = weights / weights.sum()
    if cq_max_MHz is None:
        cq_max_MHz = max(czjzek_cq_quantile(0.999, float(s["sigma"]), float(s.get("d", 5)),
                                            _site_cq0(s), float(s.get("eta0", 0.0) or 0.0))
                         * (1.05 if _site_cq0(s) > 0 else 1.0) for s in sites)
    cq_edges, cq, eta_edges, eta = czjzek_grid(cq_max_MHz, n_cq, n_eta)
    density = np.zeros((len(eta), len(cq)))
    for w, s in zip(weights, sites):
        density += w * _site_density(s, cq_edges, eta_edges)
    return cq_edges, eta_edges, density


#: Axis-label symbols for the Czjzek map: C_Q, eta_Q and P italic, the
#: "Q" subscript upright (IUPAC: symbols for quantities in italic, a
#: descriptive subscript in roman). User request, 2026-10-01.
CQ_LABEL = r"$\mathit{C}_\mathrm{Q}$"
ETAQ_LABEL = r"$\mathit{\eta}_\mathrm{Q}$"
P_LABEL = r"$\mathit{P}(\mathit{C}_\mathrm{Q},\,\mathit{\eta}_\mathrm{Q})$"


def plot_czjzek_distribution(sites, spectrum=None, layout="right", figure=None, cmap="viridis",
                             cq_max_MHz=None, n_cq=200, n_eta=100, colorbar=True,
                             fontsize=12.0, component_style=None, xlabel=None,
                             show_mean=True, heatmap_ratio=1.0, site_contours=True):
    """
    The distribution of environments as a heat map -- C_Q (MHz) across,
    eta_Q up -- with the spectrum beside it.

    ``sites`` : as :func:`czjzek_mixture_map` (a fit's sites weighted by
    area give the sample's overall distribution).
    ``spectrum`` : optional ``{"x", "y", ["y_fit"], ["components"],
    ["labels"], ["xlabel"]}`` -- data and fit (drawn as the Fit view draws
    them, with ``component_style``), or ``y`` alone for a simulation.
    ``layout`` : ``"right"`` (heat map, then the spectrum in a column to
    its right) or ``"top"`` (the spectrum in a row above it).
    ``figure`` : draw into this figure (cleared); otherwise a new one.
    ``show_mean`` : mark ⟨C_Q⟩ and ⟨η_Q⟩ of the (mixture) distribution.
    ``site_contours`` : with several sites, outline each one's own
    distribution at half its maximum, in its ``"colour"`` (the colour of
    its component in the spectrum, when drawn from a fit -- which is what
    names it; without a spectrum, a legend does).

    Returns the figure; its axes are ``figure.czjzek_axes`` =
    ``{"map", "spectrum", "colorbar"}``.
    """
    import matplotlib.pyplot as plt
    from ..plotting.style import apply_spine_style, apply_tick_style
    from .fit_display import draw_fit_components

    if layout not in ("right", "top"):
        raise ValueError("layout must be 'right' or 'top'.")
    if figure is None:
        figure = plt.figure(figsize=(7.5, 3.6) if layout == "right" else (4.5, 6.5))
    else:
        figure.clear()
    cq_edges, eta_edges, density = czjzek_mixture_map(sites, cq_max_MHz, n_cq, n_eta)
    has_spec = spectrum is not None
    # The colour bar gets its own grid slot (not figure.colorbar(ax=...)),
    # so tight_layout -- and the GUI's label-fitting pass -- can lay the
    # figure out instead of giving up on it.
    cb = [0.06] if colorbar else []
    ax_spec = cbar_ax = None
    if layout == "right":
        ratios = [heatmap_ratio] + cb + ([1.3] if has_spec else [])
        grid = figure.add_gridspec(1, len(ratios), width_ratios=ratios)
        ax_map = figure.add_subplot(grid[0, 0])
        if colorbar:
            cbar_ax = figure.add_subplot(grid[0, 1])
        if has_spec:
            ax_spec = figure.add_subplot(grid[0, -1])
    else:
        rows = 2 if has_spec else 1
        grid = figure.add_gridspec(rows, 1 + len(cb),
                                   height_ratios=[1.0, heatmap_ratio] if has_spec else None,
                                   width_ratios=[1.0] + cb)
        if has_spec:
            ax_spec = figure.add_subplot(grid[0, 0])
        ax_map = figure.add_subplot(grid[-1, 0])
        if colorbar:
            cbar_ax = figure.add_subplot(grid[-1, 1])

    mesh = ax_map.pcolormesh(cq_edges, eta_edges, density, cmap=cmap, shading="flat",
                             rasterized=True)
    # Symbols italic, subscripts upright (IUPAC). \mathit is needed: the
    # package's mathtext.default is "regular", which sets letters upright.
    ax_map.set_xlabel(CQ_LABEL + " (MHz)", fontsize=fontsize)
    ax_map.set_ylabel(ETAQ_LABEL, fontsize=fontsize)
    ax_map.set_xlim(cq_edges[0], cq_edges[-1])
    ax_map.set_ylim(0, 1)
    apply_tick_style(ax_map, labelsize=fontsize * 0.9)
    if colorbar:
        cbar = figure.colorbar(mesh, cax=cbar_ax)
        cbar.set_label(P_LABEL + r" (MHz$^{-1}$)", fontsize=fontsize * 0.9)
        cbar.ax.tick_params(labelsize=fontsize * 0.8)
    site_list = [sites] if isinstance(sites, dict) else list(sites)
    if site_contours and len(site_list) > 1:
        cq_c = 0.5 * (cq_edges[1:] + cq_edges[:-1])
        eta_c = 0.5 * (eta_edges[1:] + eta_edges[:-1])
        from ..plotting import qualitative_colors
        fallback = qualitative_colors(len(site_list))
        for k, site in enumerate(site_list):
            own = _site_density(site, cq_edges, eta_edges)
            colour = site.get("colour") or fallback[k]
            ax_map.contour(cq_c, eta_c, own, levels=[0.5 * float(own.max())], colors=[colour],
                           linewidths=1.6)
            ax_map.plot([], [], color=colour, linewidth=1.6,
                        label=str(site.get("label", f"Site {k + 1}")))
        if not has_spec:
            # With a spectrum beside it, its components (same colours)
            # already name the outlines.
            ax_map.legend(frameon=True, fontsize=fontsize * 0.7, loc="lower right")
    if show_mean:
        cq_c = 0.5 * (cq_edges[1:] + cq_edges[:-1])
        eta_c = 0.5 * (eta_edges[1:] + eta_edges[:-1])
        w = density / density.sum()
        mean_cq = float(np.sum(w * cq_c[None, :]))
        mean_eta = float(np.sum(w * eta_c[:, None]))
        ax_map.plot([mean_cq], [mean_eta], marker="+", color="white", markersize=12,
                    markeredgewidth=2.0, label=r"$\langle\mathit{C}_\mathrm{Q}\rangle$, "
                                               r"$\langle\mathit{\eta}_\mathrm{Q}\rangle$")
    for site in site_list:
        if _site_cq0(site) > 0:
            # The extended model's mean EFG (C_Q0, eta_Q0): the centre the
            # random part spreads around (not the distribution's mean).
            ax_map.plot([_site_cq0(site)], [float(site.get("eta0", 0.0) or 0.0)], marker="x",
                        color="white", markersize=9, markeredgewidth=1.8, clip_on=False,
                        label=r"$\mathit{C}_\mathrm{Q0}$, $\mathit{\eta}_\mathrm{Q0}$")
    if any(_site_cq0(site) > 0 for site in site_list) and ax_map.get_legend() is None:
        # With a mean EFG there are two marks -- the centre the disorder
        # spreads around, and the mean it produces: say which is which.
        marks = [l for l in ax_map.get_lines() if l.get_marker() in ("+", "x")]
        if marks:
            legend = ax_map.legend(handles=marks, fontsize=fontsize * 0.65, loc="upper right",
                                   frameon=True, facecolor="0.25", labelcolor="white",
                                   edgecolor="none", framealpha=0.7)
            legend.set_zorder(10)

    if has_spec:
        x = np.asarray(spectrum["x"], dtype=float)
        y = np.asarray(spectrum["y"], dtype=float)
        if spectrum.get("y_fit") is not None:
            draw_fit_components(ax_spec, x, y, np.asarray(spectrum["y_fit"], dtype=float),
                                components=spectrum.get("components"),
                                labels=spectrum.get("labels"), style=component_style,
                                auto_hide_single=True)
            ax_spec.legend(frameon=False, fontsize=fontsize * 0.75)
        else:
            ax_spec.plot(x, y, color=(component_style or {}).get("data_colour", "black"),
                         linewidth=1.5)
        ax_spec.set_xlim(float(np.max(x)), float(np.min(x)))
        ax_spec.set_yticks([])
        ax_spec.set_xlabel(xlabel or spectrum.get("xlabel") or "Shift (ppm)", fontsize=fontsize)
        apply_spine_style(ax_spec, top=False, right=False, bottom=True, left=False)
        apply_tick_style(ax_spec, axis="x", labelsize=fontsize * 0.9)
    figure.czjzek_axes = {"map": ax_map, "spectrum": ax_spec, "colorbar": cbar_ax}
    try:
        figure.tight_layout()
    except Exception:  # noqa: BLE001 -- layout is cosmetic; never lose the figure over it
        pass
    return figure


def plot_czjzek_fit(results, popt, pcov, fit_info, layout="right", site=None, figure=None,
                    component_style=None, **kwargs):
    """A fit's distribution map (all sites weighted by area, or ``site``,
    index or label) beside the data, fit and sites. Other keywords go to
    :func:`plot_czjzek_distribution`."""
    from .utils import NUC_label
    n = fit_info["n_sites"]
    x = np.sort(np.asarray(results["x_ppm"], dtype=float))
    y = np.asarray(results["real_spectrum"], dtype=float)[np.argsort(results["x_ppm"])]
    y_fit, comps = evaluate_czjzek_fit(x, popt, fit_info)
    mask = build_region_mask(x, fit_info.get("fit_regions"))
    integrate = getattr(np, "trapezoid", None) or np.trapz
    areas = [abs(float(integrate(c, x))) for c in comps[:n]]
    from .fit_display import component_style as _style, default_component_colours
    style = _style(component_style)
    defaults = default_component_colours([c[mask] for c in comps[:n]], list(fit_info["labels"]))
    popt = _legacy_params(popt, n)
    sites = [{"sigma": popt[s * _N + 1], "d": popt[s * _N + 2], "CQ0": popt[s * _N + 6],
              "eta0": popt[s * _N + 7], "weight": areas[s],
              "label": fit_info["labels"][s],
              "colour": style["line_colours"].get(fit_info["labels"][s],
                                                  defaults[fit_info["labels"][s]])}
             for s in range(n)]
    if site is not None:
        k = fit_info["labels"].index(site) if isinstance(site, str) else int(site)
        sites = [sites[k]]
    try:
        xlabel = NUC_label(results.get("nuclide", ""), x_unit="ppm") if results.get("nuclide") \
            else None
    except Exception:  # noqa: BLE001 -- a label must not sink a plot
        xlabel = None
    return plot_czjzek_distribution(
        sites, spectrum={"x": x[mask], "y": y[mask], "y_fit": y_fit[mask],
                         "components": [c[mask] for c in comps], "labels": fit_info["labels"],
                         "xlabel": xlabel},
        layout=layout, figure=figure, component_style=component_style, **kwargs)
