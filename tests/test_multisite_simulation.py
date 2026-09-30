"""Several CSA or quadrupolar sites in one simulated spectrum
(notebook.simulations.csa_multisite_spectrum / quadrupolar_multisite_spectrum),
and the fit-amplitude -> amount conversion that lets a fit's sites be
simulated whole (csa_fitting.csa_site_amounts, quadrupolar_fitting.quad_site_amounts).

Explicit user request, 2026-09-30: fit several sites, simulate the full
spectrum in Simulate, adjust, send every site back to Fit."""
import numpy as np
import pytest

from opennmr.notebook import csa_fitting as cf
from opennmr.notebook import quadrupolar_fitting as qf
from opennmr.notebook import simulations as s

X = np.linspace(-150.0, 250.0, 2048)
GRID = dict(powder_scheme="zcw", n_zcw=3000)
CSA_SITES = [
    dict(label="A", d_iso=20.0, span_or_Delta=60.0, skew_or_eta=0.3, fwhm=3.0, amount=1.0),
    dict(label="B", d_iso=90.0, span_or_Delta=-40.0, skew_or_eta=0.7, fwhm=6.0, amount=2.5),
]
QUAD_SITES = [
    dict(label="Al1", C_Q_MHz=3.0, eta_Q=0.2, d_iso=10.0, fwhm=1.5, amount=1.0),
    dict(label="Al2", C_Q_MHz=6.0, eta_Q=0.6, d_iso=60.0, fwhm=3.0, amount=3.0),
]
XQ = np.linspace(-150.0, 150.0, 2048)


def test_one_csa_site_is_exactly_the_single_site_simulator():
    site = CSA_SITES[0]
    csa = s.csa_definition(site["d_iso"], site["span_or_Delta"], site["skew_or_eta"])
    single = s.make_static_csa_spectrum(csa, X, fwhm=site["fwhm"], **GRID)
    multi = s.csa_multisite_spectrum([{**site, "amount": 1.0}], X, **GRID)
    np.testing.assert_array_equal(multi["real_spec"], single["real_spec"])
    mas = s.finite_mas_csa_spectrum(csa, X, larmor_Hz=100e6, mas_Hz=5000, fwhm=site["fwhm"],
                                    **GRID)
    multi_mas = s.csa_multisite_spectrum([site | {"amount": 1.0}], X, static=False,
                                         larmor_Hz=100e6, mas_Hz=5000, **GRID)
    np.testing.assert_array_equal(multi_mas["real_spec"], mas["real_spec"])


def test_csa_sites_sum_weighted_by_amount():
    r = s.csa_multisite_spectrum(CSA_SITES, X, **GRID)
    assert r["labels"] == ["A", "B"]
    np.testing.assert_allclose(r["real_spec"], r["components"][0] + r["components"][1])
    # Amount is the number of nuclei: integrated intensity in the ratio 1 : 2.5.
    areas = [c.sum() for c in r["components"]]
    assert areas[1] / areas[0] == pytest.approx(2.5, rel=1e-6)
    np.testing.assert_allclose(r["shares"], [1 / 3.5, 2.5 / 3.5])
    assert set(r["sites"]) == {"A", "B"}
    assert "2 site(s)" in r["log"] and "amount 2.5" in r["log"]


def test_a_site_clipped_by_the_axis_is_not_inflated():
    # Only part of site B lies on this axis: its contribution must be that
    # part at its true scale, not rescaled to make up the lost intensity.
    narrow = np.linspace(-150.0, 80.0, 1024)
    alone = s.csa_multisite_spectrum([CSA_SITES[1] | {"amount": 1.0}], narrow, **GRID)
    assert alone["real_spec"].sum() < 0.9


def test_quadrupolar_single_site_matches_and_sites_sum():
    site = QUAD_SITES[0]
    single = s.quadrupolar_spectrum(2.5, site["C_Q_MHz"] * 1e6, site["eta_Q"], 130.3e6, XQ,
                                    order="second", transitions="ct", delta_iso=site["d_iso"],
                                    fwhm=site["fwhm"], n_zcw=4181)
    multi = s.quadrupolar_multisite_spectrum([site], 2.5, 130.3e6, XQ, order="second",
                                             transitions="ct", n_zcw=4181)
    np.testing.assert_array_equal(multi["real_spec"], single["real_spec"])
    both = s.quadrupolar_multisite_spectrum(QUAD_SITES, 2.5, 130.3e6, XQ, order="second",
                                            transitions="ct", n_zcw=4181)
    areas = [c.sum() for c in both["components"]]
    assert areas[1] / areas[0] == pytest.approx(3.0, rel=1e-3)
    np.testing.assert_allclose(both["transition_components"]["CT"], both["real_spec"])


def test_quadrupolar_order_comparison_of_several_sites():
    out = s.quadrupolar_multisite_order_comparison(QUAD_SITES, 2.5, 130.3e6, XQ,
                                                   transitions="ct", n_zcw=1000,
                                                   orders=("first", "second"))
    assert set(out) >= {"first", "second", "second_minus_first"}
    np.testing.assert_allclose(out["second_minus_first"],
                               out["second"]["real_spec"] - out["first"]["real_spec"])
    assert "CT" in out["second"]["components"]          # keyed by transition for plotting


def test_axis_limits_union_and_invalid_input():
    assert s.multisite_axis_limits([(-10, 5), (0, 40), (-3, 2)]) == (-10, 40)
    with pytest.raises(ValueError):
        s.csa_multisite_spectrum([], X)
    with pytest.raises(ValueError, match="amount"):
        s.csa_multisite_spectrum([CSA_SITES[0] | {"amount": -1}], X, **GRID)
    with pytest.raises(ValueError, match="larmor_Hz"):
        s.csa_multisite_spectrum(CSA_SITES, X, static=False)


def test_amounts_from_amplitudes():
    np.testing.assert_allclose(s.site_amounts_from_amplitudes([2.0, 6.0], [4.0, 3.0]),
                               [0.25, 1.0])
    np.testing.assert_allclose(
        s.site_amounts_from_amplitudes([2.0, 6.0], [4.0, 3.0], normalise=False), [0.5, 2.0])


def test_csa_fit_amounts_recover_the_simulated_ones():
    """Fit two simulated sites; the amounts come back in the ratio of the
    nuclei (the amplitudes, peak heights, need not be)."""
    y = s.csa_multisite_spectrum(CSA_SITES, X, **GRID)["real_spec"] * 1000
    results = {"x_ppm": X[::-1], "real_spectrum": y[::-1], "SFO1": 100.0}
    p0 = [dict(label="A", d_iso=22, span_or_Delta=55, skew_or_eta=0.4, fwhm=3),
          dict(label="B", d_iso=88, span_or_Delta=-45, skew_or_eta=0.6, fwhm=5)]
    popt, _pcov, _free, _scale, info = cf.fit_csa_spectrum(results, p0, static=True, **GRID)
    amounts = cf.csa_site_amounts(X, popt, info)
    np.testing.assert_allclose(amounts, [1 / 2.5, 1.0], rtol=5e-3)


def test_quad_fit_amounts_recover_the_simulated_ones():
    y = s.quadrupolar_multisite_spectrum(QUAD_SITES, 2.5, 130.3e6, XQ, order="second",
                                         transitions="ct", n_zcw=4181)["real_spec"] * 1e4
    results = {"x_ppm": XQ, "real_spectrum": y, "SFO1": 130.3, "nuclide": "27Al"}
    p0 = [dict(label="Al1", C_Q=3.1, eta_Q=0.25, d_iso=11, fwhm=1.5),
          dict(label="Al2", C_Q=5.8, eta_Q=0.55, d_iso=58, fwhm=3)]
    popt, _pcov, _free, _scale, info = qf.fit_quadrupolar_spectrum(
        results, p0, I=2.5, larmor_Hz=130.3e6, transitions="ct", order="second")
    # Amplitudes (peak heights) are far from the 1 : 3 of the nuclei ...
    assert popt[4] / popt[10] > 1.0
    # ... amounts are not.
    np.testing.assert_allclose(qf.quad_site_amounts(XQ, popt, info), [1 / 3, 1.0], rtol=5e-3)


def test_normalised_false_is_the_raw_simulator_output():
    shape = cf.csa_model_full(np.sort(X), 20, 60, 0.3, 3.0, normalised=False, **GRID)
    norm = cf.csa_model_full(np.sort(X), 20, 60, 0.3, 3.0, **GRID)
    np.testing.assert_allclose(norm, shape / shape.max())
    assert shape.sum() == pytest.approx(1.0, abs=1e-6)
