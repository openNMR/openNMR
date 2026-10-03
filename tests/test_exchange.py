"""notebook/exchange.py: Bloch-McConnell exchange lineshapes (the
XReSpecs successor). Every expected value here is analytic: the
fast-exchange limit is the population-weighted shift, populations obey
detailed balance, two equally populated lines coalesce at
k = pi*dnu/sqrt(2) (Gutowsky-Holm), and the widths are exact FWHM."""
import math

import numpy as np
import pytest
from scipy.linalg import expm

from opennmr.notebook import exchange as ex

TWO_SITES = [{"label": "A", "shift_ppm": 20.0, "fwhm_Hz": 50.0},
             {"label": "B", "shift_ppm": 0.0, "fwhm_Hz": 50.0}]
#: The XReSpecs example's barriers: A->B 0.35 eV, B->A 0.37 eV.
XRESPECS_HOP = [{"from": 0, "to": 1, "Ea_eV": 0.35, "Ea_rev_eV": 0.37, "nu0_Hz": 1e11}]


def _fwhm(x, y):
    half = x[y >= 0.5 * y.max()]
    return float(half.max() - half.min())


def test_arrhenius_rate():
    assert ex.arrhenius_rate(1e12, 0.0, 300) == pytest.approx(1e12)
    assert ex.arrhenius_rate(1e12, 0.5, 300) == pytest.approx(1e12 * math.exp(-0.5 / (ex.K_B_EV * 300)))
    with pytest.raises(ValueError):
        ex.arrhenius_rate(1e12, 0.3, 0)


def test_rate_matrix_conserves_magnetisation_and_detailed_balance():
    K = ex.rate_matrix(2, XRESPECS_HOP, 300)
    assert np.allclose(K.sum(axis=0), 0)          # columns sum to zero
    p = ex.equilibrium_populations(K)
    kf, kr = K[1, 0], K[0, 1]
    assert p[0] * kf == pytest.approx(p[1] * kr)  # detailed balance
    # pA/pB = exp(-(Ea_rev - Ea_fwd)/kT): A is the higher-energy site
    assert p[0] / p[1] == pytest.approx(math.exp(-0.02 / (ex.K_B_EV * 300)))


def test_rate_matrix_rejects_bad_hops():
    with pytest.raises(ValueError):
        ex.rate_matrix(2, [{"from": 0, "to": 0, "Ea_eV": 0.3}], 300)
    with pytest.raises(ValueError):
        ex.rate_matrix(2, [{"from": 0, "to": 5, "Ea_eV": 0.3}], 300)


def test_disconnected_sites_share_population_equally():
    K = np.zeros((3, 3))
    assert np.allclose(ex.equilibrium_populations(K), 1 / 3)


def test_fast_limit_is_population_weighted_not_xrespecs():
    """The XReSpecs example narrows to pA*20 ppm = 6.8 ppm at 350 K (A is
    the minority site); XReSpecs put it at 13.2 ppm, i.e. weighted by the
    wrong population."""
    r = ex.exchange_spectrum(TWO_SITES, XRESPECS_HOP, 350, nucleus="7Li", B0_T=18.79,
                             ppm_axis=np.linspace(40, -20, 30001))
    peak = r["ppm"][np.argmax(r["spectrum"])]
    pA = r["populations"][0]
    assert pA == pytest.approx(0.340, abs=1e-3)
    assert r["fast_limit_ppm"] == pytest.approx(20 * pA)
    assert peak == pytest.approx(20 * pA, abs=0.02)
    assert abs(peak - 13.2) > 5


def test_slow_limit_two_lines_with_population_areas():
    r = ex.exchange_spectrum(TWO_SITES, XRESPECS_HOP, 150, larmor_MHz=300,
                             ppm_axis=np.linspace(40, -20, 60001), normalise=None)
    x, y = r["ppm"][::-1], r["spectrum"][::-1]
    a_area = np.trapezoid(y[x > 10], x[x > 10])
    b_area = np.trapezoid(y[x < 10], x[x < 10])
    pA, pB = r["populations"]
    assert a_area / b_area == pytest.approx(pA / pB, rel=0.02)
    assert r["regimes"] == ["slow"]


def test_matches_time_domain_propagation():
    """The closed form against a direct expm propagation + Fourier sum."""
    larmor = 100.0
    sites = [{"shift_ppm": 1.0, "fwhm_Hz": 20.0}, {"shift_ppm": -1.0, "fwhm_Hz": 40.0}]
    hops = [{"from": 0, "to": 1, "k_Hz": 300.0, "k_rev_Hz": 150.0}]
    axis = np.linspace(4, -4, 801)
    r = ex.exchange_spectrum(sites, hops, 300, larmor_MHz=larmor, ppm_axis=axis, normalise=None)
    K = ex.rate_matrix(2, hops, 300)
    p = ex.equilibrium_populations(K)
    L = np.diag(2j * np.pi * np.array([100.0, -100.0]) - np.pi * np.array([20.0, 40.0])) + K
    dt = 2e-5
    t = np.arange(0, 0.5, dt)
    step = expm(L * dt)
    M = np.empty((t.size, 2), complex)
    M[0] = p
    for i in range(1, t.size):
        M[i] = step @ M[i - 1]
    fid = M.sum(axis=1)
    w = 2 * np.pi * axis * larmor
    # trapezoid in time of fid * exp(-i w t)
    spec = np.array([np.trapezoid(fid * np.exp(-1j * wi * t), t) for wi in w]).real
    assert np.allclose(r["spectrum"], spec, atol=2e-3 * spec.max())


def test_lorentzian_and_gaussian_fwhm_exact():
    axis = np.linspace(5, -5, 20001)
    lor = ex.exchange_spectrum([{"shift_ppm": 0, "fwhm_Hz": 100}], [], 300, larmor_MHz=100,
                               ppm_axis=axis)
    assert _fwhm(lor["ppm"], lor["spectrum"]) == pytest.approx(1.0, abs=1e-3)
    gau = ex.exchange_spectrum([{"shift_ppm": 0, "fwhm_Hz": 0}], [], 300, larmor_MHz=100,
                               ppm_axis=axis, gauss_fwhm_ppm=1.0)
    assert _fwhm(gau["ppm"], gau["spectrum"]) == pytest.approx(1.0, abs=1e-3)
    # Voigt: both widths, broader than either
    voigt = ex.exchange_spectrum([{"shift_ppm": 0, "fwhm_ppm": 1.0}], [], 300, larmor_MHz=100,
                                 ppm_axis=axis, gauss_fwhm_ppm=1.0)
    w = _fwhm(voigt["ppm"], voigt["spectrum"])
    assert 1.6 < w < 2.0          # Olivero-Longbothum: ~1.64


def test_area_is_conserved_through_exchange():
    axis = np.linspace(60, -40, 40001)
    areas = []
    for T in (150, 275, 600):
        r = ex.exchange_spectrum(TWO_SITES, XRESPECS_HOP, T, larmor_MHz=300, ppm_axis=axis,
                                 normalise=None)
        areas.append(np.trapezoid(r["spectrum"][::-1], r["ppm"][::-1]))
    assert np.allclose(areas, areas[0], rtol=0.01)


@pytest.mark.parametrize("factor, flat", [(0.9, False), (1.0, True), (1.1, True)])
def test_gutowsky_holm_coalescence(factor, flat):
    dnu = 1000.0
    kc = ex.coalescence_rate(dnu)
    assert kc == pytest.approx(math.pi * dnu / math.sqrt(2))
    r = ex.exchange_spectrum([{"shift_ppm": 5, "fwhm_Hz": 1e-3}, {"shift_ppm": -5, "fwhm_Hz": 1e-3}],
                             [{"from": 0, "to": 1, "k_Hz": kc * factor}], 300, larmor_MHz=100,
                             ppm_axis=np.linspace(15, -15, 6001))
    centre = r["spectrum"][3000] / r["spectrum"].max()
    assert (centre > 0.999) == flat


def test_coalescence_temperature_inverts_arrhenius():
    Tc = ex.coalescence_temperature(10.0, 100.0, 0.4, 1e12)
    k = ex.arrhenius_rate(1e12, 0.4, Tc)
    assert k == pytest.approx(ex.coalescence_rate(1000.0), rel=1e-9)
    with pytest.raises(ValueError):
        ex.coalescence_temperature(0.0, 100.0, 0.4)
    with pytest.raises(ValueError):
        ex.coalescence_temperature(10.0, 100.0, 0.4, nu0_Hz=10.0)


def test_shared_site_is_one_site_in_a_network():
    """Three sites, the first exchanging with both others (the XReSpecs
    docstring's 'three pairs sharing a site' case). In the fast limit all
    three narrow to one line at the population-weighted shift, with the
    shared site counted once."""
    sites = [{"shift_ppm": 30, "fwhm_Hz": 10}, {"shift_ppm": 0, "fwhm_Hz": 10},
             {"shift_ppm": -30, "fwhm_Hz": 10}]
    hops = [{"from": 0, "to": 1, "k_Hz": 1e8}, {"from": 0, "to": 2, "k_Hz": 1e8}]
    r = ex.exchange_spectrum(sites, hops, 300, larmor_MHz=100, ppm_axis=np.linspace(40, -40, 16001))
    assert np.allclose(r["populations"], 1 / 3)
    assert r["ppm"][np.argmax(r["spectrum"])] == pytest.approx(0.0, abs=0.01)
    assert np.allclose(r["components"].sum(axis=0), r["spectrum"])


def test_shift_distribution_converges_to_the_voigt():
    """A Gaussian shift distribution with no exchange is a Voigt: the
    sampled sum must match the exact convolution (Hermite nodes did not:
    a narrow line's distribution came out as lumps)."""
    from scipy.signal import fftconvolve
    x = np.linspace(8, -8, 4001)
    sites = [{"shift_ppm": 0, "fwhm_Hz": 60.0, "shift_sd_ppm": 1.0}]
    r = ex.exchange_spectrum(sites, [], 300, larmor_MHz=100, ppm_axis=x)
    lor = 0.3 ** 2 / (x ** 2 + 0.3 ** 2)
    exact = fftconvolve(lor, np.exp(-0.5 * x ** 2), mode="same")
    exact /= exact.max()
    y = r["spectrum"]
    assert np.allclose(y, y[::-1], atol=1e-9)
    assert np.max(np.abs(y - exact)) < 5e-3
    assert _fwhm(x, y) == pytest.approx(2.688, abs=0.02)
    assert not r["warnings"]


def test_too_narrow_a_line_for_its_distribution_warns():
    sites = [{"shift_ppm": 0, "fwhm_Hz": 0.01, "shift_sd_ppm": 5.0}]
    r = ex.exchange_spectrum(sites, [], 300, larmor_MHz=100)
    assert r["warnings"] and "ripples" in r["warnings"][0]
    assert "Warning" in r["log"]


def test_barrier_distribution_is_sampled():
    hops = [dict(XRESPECS_HOP[0], Ea_sd_eV=0.01)]
    r2 = ex.exchange_spectrum(TWO_SITES, hops, 275, larmor_MHz=300, n_dist=5)
    assert r2["n_samples"] == 5
    assert "barrier s.d." in r2["log"]
    # A spread of barriers broadens the intermediate-exchange line
    # relative to the single barrier at its mean.
    r1 = ex.exchange_spectrum(TWO_SITES, XRESPECS_HOP, 275, larmor_MHz=300, n_dist=5)
    assert not np.allclose(r1["spectrum"], r2["spectrum"])


def test_sobol_fallback_beyond_max_combos():
    sites = [dict(s, shift_sd_ppm=0.2) for s in TWO_SITES]
    hops = [dict(XRESPECS_HOP[0], Ea_sd_eV=0.005)]
    r = ex.exchange_spectrum(sites, hops, 275, larmor_MHz=300, n_dist=7, max_combos=64)
    assert r["n_samples"] == 64
    r_again = ex.exchange_spectrum(sites, hops, 275, larmor_MHz=300, n_dist=7, max_combos=64)
    assert np.array_equal(r["spectrum"], r_again["spectrum"])   # deterministic


def test_needs_a_larmor_frequency():
    with pytest.raises(ValueError):
        ex.exchange_spectrum(TWO_SITES, XRESPECS_HOP, 300)


def test_vs_temperature_shares_one_axis_and_plots():
    import matplotlib
    matplotlib.use("Agg")
    res = ex.exchange_spectra_vs_temperature(TWO_SITES, XRESPECS_HOP, [200, 300, 250],
                                             nucleus="7Li", B0_T=18.79)
    assert all(np.array_equal(res[0]["ppm"], r["ppm"]) for r in res)
    fig, ax = ex.plot_exchange_stack(res, nucleus="7Li", show_fast_limit=True)
    assert ax.get_xlim()[0] > ax.get_xlim()[1]          # ppm decreasing to the right
    assert not ax.yaxis.get_visible()
    assert "Li" in ax.get_xlabel()
    assert [t.get_text() for t in ax.texts] == ["200 K", "250 K", "300 K"]


def test_fit_recovers_a_rate_from_one_spectrum():
    larmor = 300.0
    axis = np.linspace(40, -20, 3001)
    hops = [{"from": 0, "to": 1, "k_Hz": 2.0e4}]
    truth = ex.exchange_spectrum(TWO_SITES, hops, 300, larmor_MHz=larmor, ppm_axis=axis,
                                 normalise=None)["spectrum"]
    rng = np.random.default_rng(0)
    y = 3.0 * truth + 0.1 + 0.002 * truth.max() * rng.standard_normal(axis.size)
    start = [{"from": 0, "to": 1, "k_Hz": 5.0e3}]
    fit = ex.fit_exchange_spectra([{"ppm": axis, "y": y, "T_K": 300}], TWO_SITES, start,
                                  ["hop1.log10_k"], larmor_MHz=larmor)
    assert 10 ** fit["params"]["hop1.log10_k"] == pytest.approx(2.0e4, rel=0.02)
    assert fit["fits"][0]["scale"] == pytest.approx(3.0, rel=0.02)
    assert fit["fits"][0]["baseline"] == pytest.approx(0.1, abs=0.01)


def test_fit_recovers_barrier_and_attempt_frequency_from_a_vt_series():
    larmor = 300.0
    axis = np.linspace(40, -20, 1201)
    hops = [{"from": 0, "to": 1, "Ea_eV": 0.30, "nu0_Hz": 1e10}]
    spectra = []
    for T in (220, 250, 280, 320):
        y = ex.exchange_spectrum(TWO_SITES, hops, T, larmor_MHz=larmor, ppm_axis=axis)["spectrum"]
        spectra.append({"ppm": axis, "y": y, "T_K": T})
    start = [{"from": 0, "to": 1, "Ea_eV": 0.25, "nu0_Hz": 1e9}]
    fit = ex.fit_exchange_spectra(spectra, TWO_SITES, start, ["hop1.Ea_eV", "hop1.log10_nu0"],
                                  larmor_MHz=larmor)
    assert fit["params"]["hop1.Ea_eV"] == pytest.approx(0.30, abs=0.003)
    assert fit["params"]["hop1.log10_nu0"] == pytest.approx(10.0, abs=0.1)
    assert "Exchange lineshape fit" in fit["log"]


def test_fit_refuses_barrier_and_prefactor_at_one_temperature():
    axis = np.linspace(40, -20, 301)
    y = ex.exchange_spectrum(TWO_SITES, XRESPECS_HOP, 275, larmor_MHz=300, ppm_axis=axis)["spectrum"]
    with pytest.raises(ValueError, match="single temperature"):
        ex.fit_exchange_spectra([{"ppm": axis, "y": y, "T_K": 275}], TWO_SITES, XRESPECS_HOP,
                                ["hop1.Ea_eV", "hop1.log10_nu0"], larmor_MHz=300)


def test_notebook_code_runs():
    import matplotlib
    matplotlib.use("Agg")
    code = ex.exchange_notebook_code(TWO_SITES, XRESPECS_HOP, [200, 300], 310.9, 0.5, "7Li")
    namespace = {}
    exec(compile(code, "<notebook>", "exec"), namespace)
    assert len(namespace["results"]) == 2
