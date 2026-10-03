"""
Chemical-exchange (ion-hopping) lineshapes from the Bloch-McConnell
equations.

The successor to ``XReSpecs`` (SCRIPT_XReSpecsV17, 2023): 1D spectra of
nuclei hopping between sites at a temperature ``T``, with Arrhenius hop
rates, optional Gaussian distributions of shifts and barriers, and any
number of sites joined by any network of hops.

What it computes
----------------
For ``N`` sites the transverse magnetisation obeys

    dM/dt = L M,   L = i*Omega - R2 + K

with ``Omega`` the site frequencies (rad/s), ``R2`` the intrinsic
(exchange-free) decay rates, and ``K`` the rate matrix
(``K[j, i] = k_{i->j}``, ``K[i, i] = -sum_j k_{i->j}``). Starting from
the *equilibrium* populations ``p`` (the null vector of ``K``, i.e.
detailed balance), the FID is ``1^T exp(L t) p``. Diagonalising
``L = V diag(lambda) V^-1`` gives the spectrum in closed form,

    S(omega) = sum_k c_k / (i*omega - lambda_k),
    c_k = (1^T V)_k (V^-1 p)_k,

a sum of complex Lorentzians -- exact at every frequency, with no FID
truncation or FFT resolution limit. A Gaussian broadening (an FID
weighting ``exp(-sigma^2 t^2 / 2)``) turns each term into a complex
Voigt, evaluated exactly with the Faddeeva function.

Differences from XReSpecs, all deliberate
-----------------------------------------
* Populations come from detailed balance. XReSpecs computed Boltzmann
  factors but never used them, and a sign inconsistency in its
  eigenvalue discriminant (``+i*dw`` against ``-i*w`` in the trace)
  swapped the site weights whenever the forward and reverse barriers
  differed: its 20/0 ppm example (0.35/0.37 eV) narrowed to 13.2 ppm
  instead of the population-weighted 6.8 ppm.
* Barrier distributions are weighted by their Gaussian. XReSpecs built
  the weights but never applied them (a uniform +-3 sigma distribution).
* Widths are FWHM: ``R2 = pi * FWHM``. XReSpecs used the FWHM in Hz as a
  rate, giving lines a factor pi too narrow, and applied an
  ``exp(-(T_fast - T) t)`` factor mixing kelvin with s^-1.
* Sites are a network: a site shared by several hops is one site, not
  one copy per pair.

Everything here runs identically in a notebook and in the GUI
(Dynamics > Exchange lineshape), which calls these functions.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
from scipy.special import wofz

__all__ = [
    "K_B_EV", "arrhenius_rate", "rate_matrix", "equilibrium_populations",
    "bloch_mcconnell_spectrum", "exchange_spectrum", "exchange_spectra_vs_temperature",
    "coalescence_rate", "coalescence_temperature", "exchange_regime",
    "fit_exchange_spectra", "exchange_report", "plot_exchange_stack", "plot_exchange_fit",
    "exchange_notebook_code",
]

#: Boltzmann constant in eV/K (CODATA 2018, exact).
K_B_EV = 8.617333262e-5


# ----------------------------------------------------------------------
# Rates and populations
# ----------------------------------------------------------------------

def arrhenius_rate(nu0_Hz, Ea_eV, T_K):
    """Hop rate ``k = nu0 * exp(-Ea / kB T)`` in s^-1.

    ``nu0_Hz`` is the attempt frequency (typically 1e12-1e13 s^-1 for
    ion hopping), ``Ea_eV`` the barrier, ``T_K`` the temperature."""
    T = np.asarray(T_K, dtype=float)
    if np.any(T <= 0):
        raise ValueError("Temperatures must be above 0 K.")
    return np.asarray(nu0_Hz, dtype=float) * np.exp(-np.asarray(Ea_eV, dtype=float) / (K_B_EV * T))


def rate_matrix(n_sites: int, hops, T_K: float, overrides=None) -> np.ndarray:
    """The Bloch-McConnell exchange matrix ``K`` (s^-1) for ``hops`` at ``T_K``.

    Each hop is a dict with ``from``/``to`` (site indices), ``Ea_eV``
    (forward barrier), optional ``Ea_rev_eV`` (reverse barrier; equal to
    ``Ea_eV`` when omitted, i.e. equal site energies) and ``nu0_Hz``
    (attempt frequency, shared by both directions unless
    ``nu0_rev_Hz`` is given). Alternatively ``k_Hz`` (and ``k_rev_Hz``)
    give the rates directly, independent of temperature.

    ``overrides`` : optional list, one dict per hop, of values replacing
    the hop's own (used when sampling barrier distributions).
    """
    K = np.zeros((n_sites, n_sites))
    for h, hop in enumerate(hops):
        hop = dict(hop)
        if overrides is not None:
            hop.update(overrides[h])
        i, j = int(hop["from"]), int(hop["to"])
        if i == j or not (0 <= i < n_sites and 0 <= j < n_sites):
            raise ValueError(f"Hop {h + 1} joins sites {i + 1} and {j + 1}; it must join "
                             f"two different sites among the {n_sites} defined.")
        if hop.get("k_Hz") is not None:
            k_f = float(hop["k_Hz"])
            k_r = float(hop.get("k_rev_Hz", k_f) if hop.get("k_rev_Hz") is not None else k_f)
        else:
            Ea = float(hop["Ea_eV"])
            Ea_r = float(hop["Ea_rev_eV"]) if hop.get("Ea_rev_eV") is not None else Ea
            nu0 = float(hop.get("nu0_Hz", 1e12))
            nu0_r = float(hop["nu0_rev_Hz"]) if hop.get("nu0_rev_Hz") is not None else nu0
            k_f = float(arrhenius_rate(nu0, Ea, T_K))
            k_r = float(arrhenius_rate(nu0_r, Ea_r, T_K))
        if k_f < 0 or k_r < 0:
            raise ValueError(f"Hop {h + 1} has a negative rate.")
        # Column = source site: dM_j/dt gains k_{i->j} M_i.
        K[j, i] += k_f
        K[i, i] -= k_f
        K[i, j] += k_r
        K[j, j] -= k_r
    return K


def equilibrium_populations(K: np.ndarray) -> np.ndarray:
    """Steady-state populations ``p`` with ``K p = 0`` and ``sum(p) = 1``.

    For a network whose barriers satisfy detailed balance this is the
    Boltzmann distribution over site energies. Sites not connected to
    anything (all-zero rows) share the population equally with their
    disconnected partners; the null space is then resolved by least
    squares with the normalisation constraint."""
    n = K.shape[0]
    if not np.any(K):
        return np.full(n, 1.0 / n)
    A = np.vstack([K, np.ones((1, n))])
    b = np.zeros(n + 1)
    b[-1] = 1.0
    # Disconnected components make K's null space more than 1D; adding
    # a vanishing coupling to every site picks the "equal weight per
    # component" solution rather than an arbitrary one.
    components = _components(K)
    if len(components) > 1:
        p = np.zeros(n)
        for comp in components:
            sub = K[np.ix_(comp, comp)]
            p_sub = equilibrium_populations(sub) if len(comp) > 1 else np.ones(1)
            p[comp] = p_sub * len(comp) / n
        return p
    p, *_ = np.linalg.lstsq(A, b, rcond=None)
    p = np.clip(p, 0.0, None)
    return p / p.sum()


def _components(K):
    n = K.shape[0]
    adj = (np.abs(K) + np.abs(K.T)) > 0
    seen, comps = set(), []
    for s in range(n):
        if s in seen:
            continue
        stack, comp = [s], []
        seen.add(s)
        while stack:
            v = stack.pop()
            comp.append(v)
            for w in np.nonzero(adj[v])[0]:
                if w not in seen and w != v:
                    seen.add(int(w))
                    stack.append(int(w))
        comps.append(sorted(comp))
    return comps


# ----------------------------------------------------------------------
# The lineshape
# ----------------------------------------------------------------------

def _gauss_sigma(fwhm_Hz):
    """rad/s standard deviation of a Gaussian of FWHM ``fwhm_Hz``."""
    return 2.0 * np.pi * float(fwhm_Hz) / (2.0 * math.sqrt(2.0 * math.log(2.0)))


def bloch_mcconnell_spectrum(freqs_Hz, R2_s, K, axis_Hz, populations=None,
                             gauss_fwhm_Hz: float = 0.0, return_sites: bool = False):
    """Exact exchange spectrum on ``axis_Hz``.

    Parameters
    ----------
    freqs_Hz : (N,) site offsets in Hz.
    R2_s : (N,) intrinsic transverse decay rates in s^-1 (``pi * FWHM``).
    K : (N, N) exchange matrix from :func:`rate_matrix`.
    axis_Hz : frequency axis in Hz (any order).
    populations : (N,) starting magnetisation; equilibrium when omitted.
    gauss_fwhm_Hz : extra Gaussian broadening (FWHM, Hz) applied to the
        whole spectrum, exactly (complex Voigt via the Faddeeva function).
    return_sites : also return each site's contribution (columns of the
        propagator weighted by its starting magnetisation), which sum to
        the total.

    Returns
    -------
    complex ndarray, same shape as ``axis_Hz`` (absorption = real part),
    and, with ``return_sites``, an (N, len(axis)) complex array.
    """
    freqs = np.asarray(freqs_Hz, dtype=float)
    R2 = np.broadcast_to(np.asarray(R2_s, dtype=float), freqs.shape)
    n = freqs.size
    K = np.asarray(K, dtype=float)
    p = equilibrium_populations(K) if populations is None else np.asarray(populations, float)
    L = np.diag(2j * np.pi * freqs - R2) + K
    lam, V = np.linalg.eig(L)
    Vinv = np.linalg.inv(V)
    left = np.ones(n) @ V                         # 1^T V
    w = 2.0 * np.pi * np.asarray(axis_Hz, dtype=float)
    sigma = _gauss_sigma(gauss_fwhm_Hz) if gauss_fwhm_Hz and gauss_fwhm_Hz > 0 else 0.0

    def _line(lk):
        a = 1j * w - lk                           # FID exp(lk t) e^{-i w t} -> 1/a
        if sigma <= 0:
            return 1.0 / a
        # int_0^inf exp(-a t - sigma^2 t^2/2) dt = sqrt(pi/2)/sigma * w(i a/(sigma sqrt 2))
        return math.sqrt(math.pi / 2.0) / sigma * wofz(1j * a / (sigma * math.sqrt(2.0)))

    lines = [_line(lk) for lk in lam]
    if not return_sites:
        right = Vinv @ p
        return sum(left[k] * right[k] * lines[k] for k in range(n))
    sites = np.zeros((n, w.size), dtype=complex)
    for s in range(n):
        right = Vinv[:, s] * p[s]
        sites[s] = sum(left[k] * right[k] * lines[k] for k in range(n))
    return sites.sum(axis=0), sites


# ----------------------------------------------------------------------
# High-level: sites + hops + temperature
# ----------------------------------------------------------------------

def _larmor_MHz(nucleus=None, B0_T=None, larmor_MHz=None):
    if larmor_MHz:
        return float(larmor_MHz)
    if nucleus and B0_T:
        from .simulations import larmor_frequency_Hz
        return larmor_frequency_Hz(nucleus, B0_T) / 1e6
    raise ValueError("Give larmor_MHz, or nucleus and B0_T, to convert ppm to Hz.")


def _gauss_nodes(n):
    """``n`` equally spaced points over +-4 standard deviations of N(0, 1)
    with Gaussian weights (normalised). Equal spacing rather than
    Gauss-Hermite: the quantity averaged -- a line at the sampled
    shift -- is sharply peaked, and Hermite nodes (spaced ~1 s.d.
    apart) leave a narrow line's distribution as separate lumps (15
    Hermite nodes gave 2.0 ppm where the exact FWHM is 2.69 ppm)."""
    n = int(n)
    if n <= 1:
        return np.zeros(1), np.ones(1)
    x = np.linspace(-4.0, 4.0, n)
    w = np.exp(-0.5 * x ** 2)
    return x, w / w.sum()


def _site_fwhm_Hz(site, larmor):
    if site.get("fwhm_Hz") is not None:
        return float(site["fwhm_Hz"])
    return float(site.get("fwhm_ppm", 0.0)) * larmor


#: Most grid points along one distributed shift.
_MAX_SHIFT_NODES = 161


def _shift_nodes(sd_ppm, fwhm_ppm, n_dist):
    """Grid points for a shift distribution: enough that neighbours are
    at most 0.4 of the line's width apart (so the distribution is
    smooth), never fewer than ``n_dist``, at most
    :data:`_MAX_SHIFT_NODES`; odd, so the mean is sampled."""
    if fwhm_ppm <= 0:
        n = _MAX_SHIFT_NODES
    else:
        n = int(math.ceil(8.0 * sd_ppm / (0.4 * fwhm_ppm))) + 1
    n = min(max(n, int(n_dist)), _MAX_SHIFT_NODES)
    return n + 1 - n % 2


def _distribution_samples(sites, hops, n_dist, max_combos, seed=0, larmor=1.0, gauss_fwhm_ppm=0.0):
    """(weights, list of (shift_offsets_ppm (N,), hop_overrides), warnings)
    sampling every Gaussian distribution given (``shift_sd_ppm`` per
    site, ``Ea_sd_eV`` per hop, applied to both directions together so a
    hop's site-energy difference is kept)."""
    dims = []
    warnings = []
    for s, site in enumerate(sites):
        sd = float(site.get("shift_sd_ppm") or 0.0)
        if sd > 0:
            width = _site_fwhm_Hz(site, larmor) / larmor + float(gauss_fwhm_ppm)
            dims.append(("shift", s, sd, _shift_nodes(sd, width, n_dist), width))
    for h, hop in enumerate(hops):
        sd = float(hop.get("Ea_sd_eV") or 0.0)
        if sd > 0 and hop.get("k_Hz") is None:
            dims.append(("Ea", h, sd, max(int(n_dist), 1), None))
    if not dims:
        return np.ones(1), [(np.zeros(len(sites)), None)], warnings
    grids = [_gauss_nodes(d[3]) for d in dims]
    total = int(np.prod([len(g[0]) for g in grids], dtype=float))
    if total <= max_combos:
        points = []
        for idx in itertools.product(*[range(len(g[0])) for g in grids]):
            z = np.array([grids[k][0][i] for k, i in enumerate(idx)])
            wt = float(np.prod([grids[k][1][i] for k, i in enumerate(idx)]))
            points.append((z, wt))
        for (kind, idx, sd, n, width), (x, _w) in zip(dims, grids):
            if kind == "shift" and len(x) > 1 and (x[1] - x[0]) * sd > 0.5 * width:
                warnings.append(
                    f"Site {idx + 1}'s shift distribution ({sd:g} ppm s.d.) is wider than "
                    f"{_MAX_SHIFT_NODES} points can sample smoothly for its "
                    f"{width:.3g} ppm line: expect ripples. Add width or Gaussian broadening.")
    else:
        # Too many dimensions for a product grid: a scrambled Sobol set
        # (fixed seed, so the same inputs give the same spectrum).
        from scipy.stats import norm, qmc
        m = int(math.ceil(math.log2(max_combos)))
        u = qmc.Sobol(len(dims), scramble=True, seed=seed).random_base2(m)[:max_combos]
        zs = norm.ppf(np.clip(u, 1e-12, 1 - 1e-12))
        points = [(row, 1.0 / len(zs)) for row in zs]
        warnings.append(
            f"{len(dims)} distributions would need {total} grid points; sampled at "
            f"{len(zs)} quasi-random points instead, which can leave a narrow line "
            "slightly uneven. Raise max_combos for a smoother result.")
    samples, weights = [], []
    for z, wt in points:
        offsets = np.zeros(len(sites))
        overrides = [dict() for _ in hops]
        for (kind, idx, sd, _n, _w), zi in zip(dims, z):
            if kind == "shift":
                offsets[idx] = zi * sd
            else:
                hop = hops[idx]
                Ea = float(hop["Ea_eV"])
                Ea_r = float(hop["Ea_rev_eV"]) if hop.get("Ea_rev_eV") is not None else Ea
                overrides[idx] = {"Ea_eV": Ea + zi * sd, "Ea_rev_eV": Ea_r + zi * sd}
        samples.append((offsets, overrides))
        weights.append(wt)
    w = np.asarray(weights)
    return w / w.sum(), samples, warnings


def default_ppm_axis(sites, larmor_MHz, gauss_fwhm_ppm=0.0, n_points=4096):
    """A ppm axis (descending) covering every site with room for its line."""
    shifts = [float(s["shift_ppm"]) for s in sites]
    widths = [_site_fwhm_Hz(s, larmor_MHz) / larmor_MHz + 3 * float(s.get("shift_sd_ppm") or 0)
              for s in sites]
    span = max(shifts) - min(shifts)
    pad = max(10 * max(widths + [gauss_fwhm_ppm, 1e-3]), 0.3 * span, 1.0)
    return np.linspace(max(shifts) + pad, min(shifts) - pad, int(n_points))


def exchange_spectrum(sites, hops, T_K, nucleus=None, B0_T=None, larmor_MHz=None,
                      ppm_axis=None, gauss_fwhm_ppm: float = 0.0, n_dist: int = 7,
                      max_combos: int = 4096, normalise: str | None = "max",
                      site_components: bool = True) -> dict:
    """Exchange-broadened spectrum of ``sites`` hopping via ``hops`` at ``T_K``.

    Parameters
    ----------
    sites : list of dict
        ``shift_ppm`` (required); ``fwhm_ppm`` or ``fwhm_Hz`` (intrinsic
        Lorentzian width without exchange, default 0); ``shift_sd_ppm``
        (Gaussian distribution of the shift, default 0); ``label``.
    hops : list of dict
        ``from``, ``to`` (0-based site indices); ``Ea_eV`` and optional
        ``Ea_rev_eV`` (reverse barrier -- a different value makes the
        sites' energies differ by ``Ea_eV - Ea_rev_eV``); ``nu0_Hz``
        (attempt frequency, default 1e12); ``Ea_sd_eV`` (Gaussian spread
        of both barriers together); or ``k_Hz``/``k_rev_Hz`` for fixed
        rates. Each hop is reversible: list a pair once.
    T_K : temperature in kelvin.
    nucleus, B0_T, larmor_MHz : the ppm-to-Hz conversion (give
        ``larmor_MHz``, or a nucleus such as '7Li' and the field in T).
    ppm_axis : axis to evaluate on (descending ppm, as everywhere in
        openNMR); chosen from the sites when omitted.
    gauss_fwhm_ppm : Gaussian broadening of the whole spectrum (ppm).
    n_dist : minimum grid points per distributed parameter (a shift
        distribution gets as many more as its line needs to be smooth).
    max_combos : above this many grid points, a Sobol sample of that
        size replaces the product grid.
    normalise : 'max' (peak = 1), 'area' (integral = 1) or None.
    site_components : also return each site's contribution.

    Returns
    -------
    dict with ``ppm``, ``spectrum`` (real absorption), ``components``
    (per-site, same scaling) when asked, ``labels``, ``populations``
    (at the mean barriers), ``rates`` (hop list with k forward/back,
    s^-1), ``fast_limit_ppm`` (population-weighted mean shift: where a
    fully narrowed line sits), ``T_K``, ``larmor_MHz`` and ``log`` (a
    human-readable account of how the spectrum was made).
    """
    if not sites:
        raise ValueError("Define at least one site.")
    larmor = _larmor_MHz(nucleus, B0_T, larmor_MHz)
    n = len(sites)
    ppm = (default_ppm_axis(sites, larmor, gauss_fwhm_ppm)
           if ppm_axis is None else np.asarray(ppm_axis, dtype=float))
    axis_Hz = ppm * larmor
    R2 = np.array([math.pi * _site_fwhm_Hz(s, larmor) for s in sites])
    # A floor on R2 keeps a zero-width, exchange-free line finite.
    R2 = np.maximum(R2, 1e-9)
    base_shifts = np.array([float(s["shift_ppm"]) for s in sites])
    weights, samples, warnings = _distribution_samples(sites, hops, n_dist, max_combos,
                                                       larmor=larmor, gauss_fwhm_ppm=gauss_fwhm_ppm)

    total = np.zeros(ppm.size)
    comps = np.zeros((n, ppm.size))
    for wt, (offsets, overrides) in zip(weights, samples):
        K = rate_matrix(n, hops, T_K, overrides)
        spec, per_site = bloch_mcconnell_spectrum(
            (base_shifts + offsets) * larmor, R2, K, axis_Hz,
            gauss_fwhm_Hz=gauss_fwhm_ppm * larmor, return_sites=True)
        total += wt * spec.real
        comps += wt * per_site.real

    K0 = rate_matrix(n, hops, T_K)
    pops = equilibrium_populations(K0)
    scale = 1.0
    if normalise == "max":
        peak = float(np.max(np.abs(total)))
        scale = 1.0 / peak if peak > 0 else 1.0
    elif normalise == "area":
        area = abs(float(np.trapezoid(total, ppm))) if hasattr(np, "trapezoid") else \
            abs(float(np.trapz(total, ppm)))
        scale = 1.0 / area if area > 0 else 1.0
    labels = [s.get("label") or f"Site {i + 1}" for i, s in enumerate(sites)]
    rates = []
    for h, hop in enumerate(hops):
        i, j = int(hop["from"]), int(hop["to"])
        rates.append({"from": i, "to": j, "k_fwd": float(K0[j, i]), "k_rev": float(K0[i, j]),
                      "label": f"{labels[i]} ⇄ {labels[j]}"})
    fast = float(np.dot(pops, base_shifts))
    result = {
        "ppm": ppm, "spectrum": total * scale, "labels": labels,
        "populations": pops, "rates": rates, "fast_limit_ppm": fast,
        "T_K": float(T_K), "larmor_MHz": larmor, "n_samples": len(samples),
        "warnings": warnings,
        "regimes": [exchange_regime(r["k_fwd"] + r["k_rev"],
                                    abs(base_shifts[r["from"]] - base_shifts[r["to"]]) * larmor)
                    for r in rates],
    }
    if site_components:
        result["components"] = comps * scale
    result["log"] = exchange_report(result, sites, hops)
    return result


def exchange_spectra_vs_temperature(sites, hops, temperatures, **kwargs) -> list:
    """:func:`exchange_spectrum` at each temperature (same axis for all)."""
    temperatures = [float(t) for t in temperatures]
    if not temperatures:
        raise ValueError("Give at least one temperature.")
    if kwargs.get("ppm_axis") is None:
        larmor = _larmor_MHz(kwargs.get("nucleus"), kwargs.get("B0_T"), kwargs.get("larmor_MHz"))
        kwargs["ppm_axis"] = default_ppm_axis(sites, larmor, kwargs.get("gauss_fwhm_ppm", 0.0))
    return [exchange_spectrum(sites, hops, T, **kwargs) for T in temperatures]


# ----------------------------------------------------------------------
# Regimes and coalescence
# ----------------------------------------------------------------------

def coalescence_rate(delta_nu_Hz: float) -> float:
    """Exchange rate ``k`` (each direction, s^-1) at which two equally
    populated lines coalesce: ``k_c = pi * dnu / sqrt(2)`` (Gutowsky-Holm)."""
    return math.pi * abs(float(delta_nu_Hz)) / math.sqrt(2.0)


def coalescence_temperature(delta_ppm: float, larmor_MHz: float, Ea_eV: float,
                            nu0_Hz: float = 1e12) -> float:
    """Temperature (K) at which an Arrhenius hop ``nu0 exp(-Ea/kB T)``
    reaches the coalescence rate for two equally populated sites
    ``delta_ppm`` apart. Raises if ``nu0`` is below the coalescence rate
    (the lines never coalesce)."""
    kc = coalescence_rate(delta_ppm * larmor_MHz)
    if kc <= 0:
        raise ValueError("The two sites have the same shift: there is nothing to coalesce.")
    if nu0_Hz <= kc:
        raise ValueError("The attempt frequency is below the coalescence rate; "
                         "the lines never coalesce.")
    return float(Ea_eV) / (K_B_EV * math.log(nu0_Hz / kc))


def exchange_regime(k_total_s: float, delta_nu_Hz: float) -> str:
    """'slow', 'intermediate' or 'fast' from the exchange rate (sum of
    both directions, s^-1) against the angular frequency difference."""
    dw = 2.0 * math.pi * abs(float(delta_nu_Hz))
    if dw == 0:
        return "fast"
    ratio = float(k_total_s) / dw
    if ratio < 0.1:
        return "slow"
    if ratio > 10.0:
        return "fast"
    return "intermediate"


# ----------------------------------------------------------------------
# Fitting measured spectra
# ----------------------------------------------------------------------

#: Parameters :func:`fit_exchange_spectra` can vary, with their kind.
#: ``hop{h}.log10_nu0`` is fitted on a log scale (the attempt frequency
#: spans decades); everything else is linear.
_FIT_PARAM_DOC = ("site{i}.shift_ppm", "site{i}.fwhm_ppm", "site{i}.shift_sd_ppm",
                  "hop{h}.Ea_eV", "hop{h}.Ea_rev_eV", "hop{h}.log10_nu0",
                  "hop{h}.log10_k", "gauss_fwhm_ppm")


def _apply_param(name, value, sites, hops, extra):
    if name == "gauss_fwhm_ppm":
        extra["gauss_fwhm_ppm"] = max(float(value), 0.0)
        return
    head, key = name.split(".", 1)
    if head.startswith("site"):
        site = sites[int(head[4:]) - 1]
        if key == "fwhm_ppm":
            site.pop("fwhm_Hz", None)
            value = max(float(value), 0.0)
        elif key == "shift_sd_ppm":
            value = max(float(value), 0.0)
        site[key] = float(value)
    elif head.startswith("hop"):
        hop = hops[int(head[3:]) - 1]
        if key == "log10_nu0":
            hop["nu0_Hz"] = 10.0 ** float(value)
        elif key == "log10_k":
            hop["k_Hz"] = hop["k_rev_Hz"] = 10.0 ** float(value)
        else:
            hop[key] = float(value)
    else:
        raise ValueError(f"Unknown parameter {name!r}; use one of {_FIT_PARAM_DOC}.")


def _read_param(name, sites, hops, gauss):
    if name == "gauss_fwhm_ppm":
        return float(gauss)
    head, key = name.split(".", 1)
    if head.startswith("site"):
        site = sites[int(head[4:]) - 1]
        if key == "fwhm_ppm" and site.get("fwhm_ppm") is None:
            return 0.0
        return float(site.get(key) or 0.0)
    hop = hops[int(head[3:]) - 1]
    if key == "log10_nu0":
        return math.log10(float(hop.get("nu0_Hz", 1e12)))
    if key == "log10_k":
        return math.log10(float(hop.get("k_Hz") or 1.0))
    if key == "Ea_rev_eV" and hop.get("Ea_rev_eV") is None:
        return float(hop["Ea_eV"])
    return float(hop[key])


def fit_exchange_spectra(spectra, sites, hops, vary, larmor_MHz=None, nucleus=None, B0_T=None,
                         gauss_fwhm_ppm: float = 0.0, bounds=None, n_dist: int = 5,
                         baseline: bool = True, max_nfev: int = 400) -> dict:
    """Fit one or more measured spectra (e.g. a VT series) with one
    exchange model.

    ``spectra`` : list of dicts ``{"ppm": axis, "y": intensities,
    "T_K": temperature}`` (optionally ``"window": (lo, hi)`` ppm).
    ``vary`` : names of the parameters to fit, from
    ``site{i}.shift_ppm``, ``site{i}.fwhm_ppm``, ``site{i}.shift_sd_ppm``,
    ``hop{h}.Ea_eV``, ``hop{h}.Ea_rev_eV``, ``hop{h}.log10_nu0``,
    ``hop{h}.log10_k`` (a temperature-independent rate, for a single
    spectrum) and ``gauss_fwhm_ppm`` (1-based site/hop numbers).
    Everything else is held at the value in ``sites``/``hops``.

    Each spectrum's scale (and a constant baseline, if ``baseline``) is
    solved linearly at every step, so it never needs a guess.

    Note that ``Ea`` and ``log10_nu0`` of the same hop are strongly
    correlated: they are only separable from spectra spanning a
    temperature range. Fitting both to one spectrum is refused.

    Returns a dict with ``params`` and ``errors`` (1-sigma, from the
    Jacobian, scaled by the reduced chi-square), ``sites``/``hops``
    (fitted copies), ``fits`` (per spectrum: ppm, y, model), ``redchi``
    and ``log``.
    """
    import copy
    from scipy.optimize import least_squares

    if not spectra:
        raise ValueError("Give at least one spectrum to fit.")
    vary = list(vary)
    if not vary:
        raise ValueError("Choose at least one parameter to fit.")
    temps = {round(float(s["T_K"]), 6) for s in spectra}
    for h in range(1, len(hops) + 1):
        if {f"hop{h}.Ea_eV", f"hop{h}.log10_nu0"} <= set(vary) and len(temps) < 2:
            raise ValueError(f"Hop {h}: Ea and the attempt frequency cannot both be fitted to "
                             "spectra at a single temperature -- only their combination, the "
                             "rate, is defined. Fit hop{h}.log10_k instead, or add temperatures.")
    larmor = _larmor_MHz(nucleus, B0_T, larmor_MHz)
    sites0 = copy.deepcopy(list(sites))
    hops0 = copy.deepcopy(list(hops))
    x0 = np.array([_read_param(v, sites0, hops0, gauss_fwhm_ppm) for v in vary])
    lo = np.full(x0.size, -np.inf)
    hi = np.full(x0.size, np.inf)
    for k, name in enumerate(vary):
        if name.endswith(("fwhm_ppm", "shift_sd_ppm")):
            lo[k] = 0.0
        if name.endswith(("Ea_eV", "Ea_rev_eV")):
            lo[k] = 0.0
    if bounds:
        for name, (b_lo, b_hi) in bounds.items():
            k = vary.index(name)
            lo[k], hi[k] = b_lo, b_hi
    x0 = np.clip(x0, lo + 1e-12 * np.isfinite(lo), hi)

    prepared = []
    for s in spectra:
        ppm = np.asarray(s["ppm"], dtype=float)
        y = np.asarray(s["y"], dtype=float)
        mask = np.ones(ppm.size, bool)
        if s.get("window"):
            a, b = sorted(map(float, s["window"]))
            mask = (ppm >= a) & (ppm <= b)
        if mask.sum() < 5:
            raise ValueError(f"The spectrum at {s['T_K']} K has fewer than 5 points in its window.")
        prepared.append((ppm[mask], y[mask], float(s["T_K"])))
    noise = np.concatenate([p[1] for p in prepared])
    norm = float(np.max(np.abs(noise))) or 1.0

    def models(x):
        st, hp, extra = copy.deepcopy(sites0), copy.deepcopy(hops0), {"gauss_fwhm_ppm": gauss_fwhm_ppm}
        for name, value in zip(vary, x):
            _apply_param(name, value, st, hp, extra)
        out = []
        for ppm, y, T in prepared:
            shape = exchange_spectrum(st, hp, T, larmor_MHz=larmor, ppm_axis=ppm,
                                      gauss_fwhm_ppm=extra["gauss_fwhm_ppm"], n_dist=n_dist,
                                      normalise=None, site_components=False)["spectrum"]
            cols = [shape] + ([np.ones_like(shape)] if baseline else [])
            A = np.vstack(cols).T
            coef, *_ = np.linalg.lstsq(A, y, rcond=None)
            out.append((A @ coef, coef))
        return out, st, hp, extra

    def residuals(x):
        out, *_ = models(x)
        return np.concatenate([(p[1] - m) / norm for p, (m, _c) in zip(prepared, out)])

    sol = least_squares(residuals, x0, bounds=(lo, hi), x_scale="jac", max_nfev=max_nfev)
    out, st, hp, extra = models(sol.x)
    n_pts = sum(p[1].size for p in prepared)
    n_free = len(vary) + len(prepared) * (2 if baseline else 1)
    dof = max(n_pts - n_free, 1)
    chi2 = float(np.sum(sol.fun ** 2))
    redchi = chi2 / dof
    errors = np.full(len(vary), np.nan)
    try:
        J = sol.jac
        cov = np.linalg.pinv(J.T @ J) * redchi
        errors = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        pass
    params = dict(zip(vary, map(float, sol.x)))
    errs = dict(zip(vary, map(float, errors)))
    fits = [{"ppm": p[0], "y": p[1], "model": m, "scale": float(c[0]),
             "baseline": float(c[1]) if baseline else 0.0, "T_K": p[2]}
            for p, (m, c) in zip(prepared, out)]
    lines = [f"Exchange lineshape fit: {len(prepared)} spectrum(s) "
             f"({', '.join(f'{p[2]:g} K' for p in prepared)}), {n_pts} points, "
             f"{len(vary)} model parameter(s) + {len(prepared)} scale(s)"
             + (" and baseline(s)" if baseline else "") + ".",
             f"Converged: {bool(sol.success)} ({sol.message}); reduced chi-square "
             f"(normalised intensities) = {redchi:.4g}."]
    for name in vary:
        v, e = params[name], errs[name]
        extra_txt = ""
        if name.endswith("log10_nu0"):
            extra_txt = f"  (nu0 = {10 ** v:.3g} s^-1)"
        if name.endswith("log10_k"):
            extra_txt = f"  (k = {10 ** v:.3g} s^-1)"
        lines.append(f"  {name} = {v:.6g} ± {e:.2g}{extra_txt}")
    return {"params": params, "errors": errs, "sites": st, "hops": hp,
            "gauss_fwhm_ppm": extra["gauss_fwhm_ppm"], "fits": fits, "redchi": redchi,
            "success": bool(sol.success), "larmor_MHz": larmor, "log": "\n".join(lines)}


# ----------------------------------------------------------------------
# Report, plot, notebook code
# ----------------------------------------------------------------------

def exchange_report(result, sites, hops) -> str:
    """How a spectrum from :func:`exchange_spectrum` was arrived at."""
    labels = result["labels"]
    lines = [f"Exchange lineshape at {result['T_K']:g} K, Larmor frequency "
             f"{result['larmor_MHz']:.4g} MHz (Bloch-McConnell, exact frequency-domain solution)."]
    for i, s in enumerate(sites):
        width = _site_fwhm_Hz(s, result["larmor_MHz"])
        sd = float(s.get("shift_sd_ppm") or 0.0)
        lines.append(f"  {labels[i]}: {float(s['shift_ppm']):g} ppm, intrinsic FWHM {width:.4g} Hz"
                     + (f", shift s.d. {sd:g} ppm" if sd else "")
                     + f"; equilibrium population {result['populations'][i]:.4f}")
    for r, hop, regime in zip(result["rates"], hops, result["regimes"]):
        desc = (f"  {r['label']}: k = {r['k_fwd']:.4g} s^-1 forward, {r['k_rev']:.4g} s^-1 back"
                f" ({regime} exchange)")
        if hop.get("k_Hz") is None:
            Ea_r = hop.get("Ea_rev_eV", hop["Ea_eV"])
            Ea_r = hop["Ea_eV"] if Ea_r is None else Ea_r
            desc += (f"; Ea = {float(hop['Ea_eV']):g} / {float(Ea_r):g} eV, "
                     f"nu0 = {float(hop.get('nu0_Hz', 1e12)):.3g} s^-1")
            if hop.get("Ea_sd_eV"):
                desc += f", barrier s.d. {float(hop['Ea_sd_eV']):g} eV"
        lines.append(desc)
    lines.append(f"  Fully narrowed (fast-limit) line: {result['fast_limit_ppm']:.4g} ppm.")
    if result["n_samples"] > 1:
        lines.append(f"  Distributions sampled at {result['n_samples']} points.")
    for w in result.get("warnings", []):
        lines.append(f"  Warning: {w}")
    return "\n".join(lines)


def plot_exchange_stack(results, ax=None, offset: float = 1.1, colours=None,
                        label_fmt: str = "{T:g} K", show_labels: bool = True,
                        nucleus: str | None = None, xlim=None, linewidth: float = 1.0,
                        show_fast_limit: bool = False, text_size="small"):
    """Spectra from :func:`exchange_spectra_vs_temperature`, stacked
    (lowest temperature at the bottom) with their temperatures beside
    them, ppm decreasing to the right and no intensity axis -- the usual
    VT-stack figure.

    ``colours`` : a list, or a palette name; the colour-blind-friendly
    ordered default otherwise. Returns ``(fig, ax)``."""
    import matplotlib.pyplot as plt

    from ..plotting.accessibility import default_series_colours, series_colours
    from .utils import NUC_label

    results = sorted(results, key=lambda r: r["T_K"])
    if ax is None:
        fig, ax = plt.subplots(figsize=(3.375, 4.0))
    else:
        fig = ax.figure
    n = len(results)
    if colours is None:
        colours = default_series_colours(n, ordered=True)
    elif isinstance(colours, str):
        colours = series_colours(n, colours)
    for i, r in enumerate(results):
        y = r["spectrum"] / (np.max(np.abs(r["spectrum"])) or 1.0) + i * offset
        ax.plot(r["ppm"], y, "-", color=colours[i % len(colours)], lw=linewidth,
                label=label_fmt.format(T=r["T_K"]))
        if show_fast_limit:
            ax.plot([r["fast_limit_ppm"]] * 2, [i * offset, i * offset + 0.15], color="0.4", lw=0.8)
        if show_labels:
            ax.annotate(label_fmt.format(T=r["T_K"]), xy=(1.0, i * offset + 0.05),
                        xycoords=("axes fraction", "data"), ha="right", va="bottom",
                        fontsize=text_size, color=colours[i % len(colours)])
    ppm = results[0]["ppm"]
    ax.set_xlim(xlim if xlim is not None else (float(np.max(ppm)), float(np.min(ppm))))
    if nucleus:
        try:
            ax.set_xlabel(NUC_label(nucleus))
        except Exception:  # noqa: BLE001 -- a label is not worth failing a plot over
            ax.set_xlabel(f"δ {nucleus} (ppm)")
    else:
        ax.set_xlabel(r"$\delta$ (ppm)")
    ax.yaxis.set_visible(False)
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="x", direction="out")
    fig.tight_layout()
    return fig, ax


def plot_exchange_fit(fit, ax=None, offset: float = 1.1, nucleus: str | None = None,
                      data_colour="0.15", colours=None, linewidth: float = 1.2,
                      text_size="small"):
    """Measured spectra (thin, dark) with the fitted exchange model
    (coloured) from :func:`fit_exchange_spectra`, stacked by temperature,
    each normalised to its own maximum. Returns ``(fig, ax)``."""
    import matplotlib.pyplot as plt

    from ..plotting.accessibility import default_series_colours
    from .utils import NUC_label

    fits = sorted(fit["fits"], key=lambda f: f["T_K"])
    if ax is None:
        fig, ax = plt.subplots(figsize=(3.375, 4.0))
    else:
        fig = ax.figure
    colours = colours or default_series_colours(len(fits), ordered=True)
    lo, hi = np.inf, -np.inf
    for i, f in enumerate(fits):
        norm = float(np.max(np.abs(f["y"]))) or 1.0
        base = i * offset
        ax.plot(f["ppm"], f["y"] / norm + base, "-", color=data_colour, lw=0.8 * linewidth,
                label="data" if i == 0 else None)
        ax.plot(f["ppm"], f["model"] / norm + base, "-", color=colours[i % len(colours)],
                lw=linewidth, label=f"{f['T_K']:g} K fit")
        ax.annotate(f"{f['T_K']:g} K", xy=(1.0, base + 0.05), xycoords=("axes fraction", "data"),
                    ha="right", va="bottom", fontsize=text_size, color=colours[i % len(colours)])
        lo, hi = min(lo, float(np.min(f["ppm"]))), max(hi, float(np.max(f["ppm"])))
    ax.set_xlim(hi, lo)
    if nucleus:
        try:
            ax.set_xlabel(NUC_label(nucleus))
        except Exception:  # noqa: BLE001
            ax.set_xlabel(f"δ {nucleus} (ppm)")
    else:
        ax.set_xlabel(r"$\delta$ (ppm)")
    ax.yaxis.set_visible(False)
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    return fig, ax


def exchange_notebook_code(sites, hops, temperatures, larmor_MHz, gauss_fwhm_ppm=0.0,
                           nucleus=None, n_dist=7) -> str:
    """Python that reproduces a GUI simulation in a notebook."""
    def fmt(obj):
        return repr(obj).replace("'", '"')
    return "\n".join([
        "from opennmr.notebook.exchange import exchange_spectra_vs_temperature, plot_exchange_stack",
        "",
        f"sites = {fmt([dict(s) for s in sites])}",
        f"hops = {fmt([dict(h) for h in hops])}",
        f"temperatures = {fmt([float(t) for t in temperatures])}",
        "",
        "results = exchange_spectra_vs_temperature(",
        f"    sites, hops, temperatures, larmor_MHz={float(larmor_MHz)!r},",
        f"    gauss_fwhm_ppm={float(gauss_fwhm_ppm)!r}, n_dist={int(n_dist)})",
        "for r in results:",
        "    print(r['log'])",
        f"fig, ax = plot_exchange_stack(results, nucleus={nucleus!r})",
    ])
