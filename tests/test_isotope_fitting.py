"""Several nuclei on one spectrum (notebook.isotopes, Site.link_to /
isotope_partner, simulate_combined / fit_combined with sites of other
nuclei): 47Ti and 49Ti fitted as one chemistry, and a 63Cu background
line on a 23Na spectrum fitted with its own parameters.

Expected values are worked out here from the IUPAC 2001 frequency
ratios and the nuclear data table, not read back from the code."""

import json

import numpy as np
import pytest

from opennmr.notebook import combined_fitting as cf
from opennmr.notebook import fit_advice, isotopes
from opennmr.notebook.combined_hamiltonian import (Site, combined_axis_limits, combined_to_python,
                                                   resolve_links, simulate_combined)
from opennmr.notebook.utils import _GAMMA_DF

NU0_TI = 22.55e6        # 47Ti at 9.4 T
NU0_NA = 105.84e6       # 23Na at 9.4 T
N_ZCW = 987


# ---- nuclear data and placement ---------------------------------------------------------------

def test_iupac_frequency_ratios_are_a_column_of_the_one_table():
    col = _GAMMA_DF.set_index("Nuclide")["Frequency Ratio Xi (%)"]
    assert col["47Ti"] == pytest.approx(5.637534)
    assert col["49Ti"] == pytest.approx(5.639037)
    assert col["23Na"] == pytest.approx(26.451900)
    assert col["63Cu"] == pytest.approx(26.515473)
    assert col["1H"] == pytest.approx(100.0)


def test_49ti_sits_266_ppm_above_47ti_and_63cu_2403_ppm_above_23na():
    # (Xi_49 / Xi_47 - 1) 1e6 = (5.639037 / 5.637534 - 1) 1e6
    assert isotopes.to_observed_ppm(0.0, "49Ti", "47Ti") == pytest.approx(266.606, abs=1e-3)
    assert isotopes.to_observed_ppm(0.0, "63Cu", "23Na") == pytest.approx(2403.34, abs=0.01)
    R, src = isotopes.frequency_ratio("49Ti", "47Ti")
    assert src == "Xi"
    # a shift on 49Ti's own scale, carried over exactly: R (1 + d) - 1
    assert isotopes.to_observed_ppm(-800.0, "49Ti", "47Ti") == pytest.approx(
        (R * (1 - 800e-6) - 1) * 1e6, rel=1e-12)


def test_the_gamma_ratio_would_misplace_them():
    # the reason for Xi: the gyromagnetic ratios leave out the reference shielding
    g = _GAMMA_DF.set_index("Nuclide")["Gyromagnetic Ratio (MHz/T)"].astype(float)
    gamma_ppm = (g["49Ti"] / g["47Ti"] - 1) * 1e6
    assert abs(gamma_ppm - 266.606) > 20


def test_observed_scale_and_own_scale_are_inverse():
    x = np.linspace(-2000, 500, 11)
    back = isotopes.to_observed_ppm(isotopes.from_observed_ppm(x, "49Ti", "47Ti"), "49Ti", "47Ti")
    assert np.allclose(back, x, atol=1e-9)


def test_a_nucleus_without_xi_falls_back_to_gamma_and_says_so():
    missing = [str(n) for n, xi, s in zip(_GAMMA_DF["Nuclide"], _GAMMA_DF["Frequency Ratio Xi (%)"],
                                          _GAMMA_DF["Spin"])
               if not np.isfinite(xi) and float(s) > 0]
    assert missing
    _R, src = isotopes.frequency_ratio(missing[0], "1H")
    assert src == "gamma"
    assert "approximate" in isotopes.describe_placement(missing[0], "1H")


def test_partners_and_factors():
    assert isotopes.isotope_partners("47Ti") == ["49Ti"]
    assert isotopes.isotope_partners("63Cu") == ["65Cu"]
    assert isotopes.isotope_partners("23Na") == []
    f = isotopes.partner_factors("47Ti", "49Ti")
    assert f["C_Q"] == pytest.approx(24.7 / 30.2)
    # abundance x |gamma|^3 I(I+1): 49Ti (5.41 %, I 7/2) over 47Ti (7.44 %, I 5/2)
    g47, g49 = 2.40404000000000, 2.40475161264699
    expected = (5.41 * g49 ** 3 * 3.5 * 4.5) / (7.44 * g47 ** 3 * 2.5 * 3.5)
    assert f["amount"] == pytest.approx(expected, rel=1e-9)
    assert f["J"] == pytest.approx(g49 / g47)


# ---- sites ---------------------------------------------------------------------------------------

def _ti(delta=-800.0, cq=4.0, eta=0.2, fwhm=8.0):
    return Site("47Ti", delta, label="Ti1", fwhm=fwhm).add_quadrupolar(cq, eta)


def test_isotope_partner_takes_the_chemistry_and_round_trips():
    ti = _ti().add_csa(60.0, 0.3, euler=(0, 30, 0))
    p = ti.isotope_partner()
    assert p.nucleus == "49Ti" and p.spin == 3.5 and p.label == "Ti1 (49Ti)"
    assert p.delta_iso == -800.0 and p.fwhm == 8.0
    assert p.get("quadrupolar")[0]["C_Q"] == pytest.approx(4.0 * 24.7 / 30.2)
    assert p.get("csa")[0]["aniso"] == 60.0 and p.get("csa")[0]["euler"] == (0, 30, 0)
    assert p.link == {"to": "Ti1", "free": []}
    again = Site.from_dict(json.loads(json.dumps(p.to_dict())))
    assert again.link == p.link
    # an unlinked site's dict is unchanged in shape
    assert "link" not in ti.to_dict()


def test_a_linked_partner_follows_its_parent_except_what_it_keeps():
    ti = _ti()
    p = Site("49Ti", 0.0, label="P", amount=5.0, fwhm=30.0).link_to(ti, free=["fwhm"])
    _t, q = resolve_links([ti, p])
    assert q.delta_iso == -800.0                      # followed
    assert q.fwhm == 30.0                             # its own
    assert q.amount == pytest.approx(isotopes.partner_factors("47Ti", "49Ti")["amount"])


@pytest.mark.parametrize("bad, match", [
    (lambda ti: Site("49Ti", label="P").link_to("nope"), "not another site"),
    (lambda ti: Site("23Na", label="P").link_to(ti), "only isotopes of one element"),
    (lambda ti: Site("47Ti", label="P").link_to(ti), "same nucleus"),
])
def test_links_that_cannot_hold_are_named(bad, match):
    ti = _ti()
    with pytest.raises(ValueError, match=match):
        resolve_links([ti, bad(ti)])


def test_a_chain_of_links_is_refused():
    ti = _ti()
    p = ti.isotope_partner()
    q = Site("47Ti", label="Q").link_to(p)
    with pytest.raises(ValueError, match="itself linked"):
        resolve_links([ti, p, q])


# ---- simulation ------------------------------------------------------------------------------------

def test_another_nucleus_is_placed_and_scaled_in_the_simulation():
    # isotropic lines: the 49Ti line must sit at Xi-ratio ppm on the 47Ti axis
    ti = Site("47Ti", -800, label="Ti1", fwhm=2.0)
    p = ti.isotope_partner()
    x = np.linspace(-1000, -300, 14001)
    r = simulate_combined([ti, p], x, NU0_TI)
    assert x[np.argmax(r["sites"]["Ti1"])] == pytest.approx(-800, abs=0.06)
    assert x[np.argmax(r["sites"]["Ti1 (49Ti)"])] == pytest.approx(
        float(isotopes.to_observed_ppm(-800, "49Ti", "47Ti")), abs=0.06)
    ratio = r["sites"]["Ti1 (49Ti)"].sum() / r["sites"]["Ti1"].sum()
    assert ratio == pytest.approx(isotopes.partner_factors("47Ti", "49Ti")["amount"], rel=1e-3)
    assert "IUPAC frequency ratios" in r["log"]
    assert r["settings"]["observe"] == "47Ti"


def test_central_transition_shares_follow_each_isotopes_spin():
    # CT share of all transitions: 9/35 for I = 5/2, 16/84 for I = 7/2
    ti = _ti()
    p = ti.isotope_partner()
    lo, hi = combined_axis_limits([ti, p], NU0_TI, transitions="ct")
    x = np.linspace(lo, hi, 8192)
    r = simulate_combined([ti, p], x, NU0_TI, transitions="ct", n_zcw=N_ZCW)
    amount = isotopes.partner_factors("47Ti", "49Ti")["amount"]
    assert r["sites"]["Ti1"].sum() == pytest.approx(9 / 35, rel=1e-3)
    assert r["sites"]["Ti1 (49Ti)"].sum() == pytest.approx(amount * 16 / 84, rel=1e-3)


def test_a_site_off_the_axis_is_noted_not_an_error():
    al = Site("27Al", 10, label="A").add_quadrupolar(3, 0.2)
    r = simulate_combined([al, Site("13C", label="C")], np.linspace(-100, 80, 2048), 130.3e6)
    assert "outside the axis" in r["log"]
    assert r["sites"]["C"].sum() == 0


def test_python_export_keeps_the_link():
    ti = _ti()
    code = combined_to_python([ti, ti.isotope_partner()], np.linspace(-1500, -200, 512),
                              larmor_Hz=NU0_TI, transitions="ct", n_zcw=N_ZCW, observe="47Ti")
    assert ".link_to('Ti1')" in code
    scope = {}
    exec(code, scope)
    assert set(scope["spec"]["sites"]) == {"Ti1", "Ti1 (49Ti)"}


# ---- fitting ----------------------------------------------------------------------------------------

def _ti_spectrum(sites, noise=0.005, seed=0):
    lo, hi = combined_axis_limits(sites, NU0_TI, transitions="ct")
    x = np.linspace(lo, hi, 2048)
    y = simulate_combined(sites, x, NU0_TI, transitions="ct", n_zcw=N_ZCW)["real_spec"]
    y = y / y.max() + np.random.default_rng(seed).normal(0, noise, x.size)
    return {"x_ppm": x, "real_spectrum": y * 1000, "SFO1": NU0_TI / 1e6, "nuclide": "47Ti"}


def test_47ti_and_49ti_are_fitted_as_one_chemistry():
    truth = _ti()
    res = _ti_spectrum([truth, truth.isotope_partner()])
    start = _ti(-780, 3.6, 0.4, 12.0)
    fit = cf.fit_combined(res, [start, start.isotope_partner()], transitions="ct", n_zcw=N_ZCW)
    v = {(p["label"], p["name"]): p["value"] for p in fit["parameters"]}
    assert fit["quality"]["r2"] > 0.995
    assert v[("Ti1", "delta_iso")] == pytest.approx(-800, abs=0.5)
    assert v[("Ti1", "C_Q")] == pytest.approx(4.0, abs=0.01)
    assert v[("Ti1", "eta_Q")] == pytest.approx(0.2, abs=0.01)
    # the partner has no parameters of its own, and its fitted site follows
    assert not [p for p in fit["parameters"] if p["label"] == "Ti1 (49Ti)"]
    partner = next(s for s in fit["sites"] if s.label == "Ti1 (49Ti)")
    assert partner.get("quadrupolar")[0]["C_Q"] == pytest.approx(v[("Ti1", "C_Q")] * 24.7 / 30.2)
    assert partner.amount / v[("Ti1", "amount")] == pytest.approx(
        isotopes.partner_factors("47Ti", "49Ti")["amount"])
    assert fit["derived"]["Ti1 (49Ti)"]["linked_to"] == "Ti1"
    assert "P_Q" in fit["derived"]["Ti1 (49Ti)"]
    assert fit["settings"]["observe"] == "47Ti"
    report = cf.print_combined_fit_report(fit, advice=False)
    assert "linked to Ti1" in report and "(from Ti1)" in report


def test_leaving_out_the_49ti_partner_gives_a_wrong_fit():
    truth = _ti()
    res = _ti_spectrum([truth, truth.isotope_partner()])
    fit = cf.fit_combined(res, [_ti(-780, 3.6, 0.4, 12.0)], transitions="ct", n_zcw=N_ZCW)
    assert fit["quality"]["r2"] < 0.9


def test_the_partner_can_keep_its_own_amount():
    # a selective pulse: 49Ti at 0.6 of what natural abundance and receptivity predict
    truth = _ti()
    p = truth.isotope_partner()
    p.amount *= 0.6
    p.link["free"] = ["amount"]
    res = _ti_spectrum([truth, p])
    start = _ti(-790, 3.8, 0.3, 10.0)
    fit = cf.fit_combined(res, [start, start.isotope_partner(free=["amount"])], transitions="ct",
                          n_zcw=N_ZCW)
    v = {(p["label"], p["name"]): p["value"] for p in fit["parameters"]}
    ratio = v[("Ti1 (49Ti)", "amount")] / v[("Ti1", "amount")]
    assert ratio == pytest.approx(0.6 * isotopes.partner_factors("47Ti", "49Ti")["amount"],
                                  rel=0.02)
    # shares are per nucleus: each is all of its own nucleus's signal
    assert fit["derived"]["Ti1"]["share_%"] == pytest.approx(100)
    assert fit["derived"]["Ti1 (49Ti)"]["share_%"] == pytest.approx(100)


def test_varying_a_linked_parameter_is_refused_by_name():
    ti = _ti()
    with pytest.raises(ValueError, match="follow Ti1"):
        cf.fit_combined(_ti_spectrum([ti, ti.isotope_partner()]), [ti, ti.isotope_partner()],
                        vary={"Ti1 (49Ti)": ["C_Q"]}, transitions="ct", n_zcw=N_ZCW)


def test_the_starting_model_is_the_fits_first_step_with_partners():
    ti = _ti(-790, 3.9, 0.25, 9.0)
    sites = [ti, ti.isotope_partner()]
    res = _ti_spectrum([_ti(), _ti().isotope_partner()])
    guess = cf.starting_model(res, sites, transitions="ct", n_zcw=N_ZCW)
    held = cf.fit_combined(res, sites, vary=[], transitions="ct", n_zcw=N_ZCW)
    assert np.allclose(guess["start"], held["y_fit"], rtol=1e-9, atol=1e-9)
    assert guess["amounts"]["Ti1 (49Ti)"] / guess["amounts"]["Ti1"] == pytest.approx(
        isotopes.partner_factors("47Ti", "49Ti")["amount"])


def test_a_63cu_coil_line_on_a_23na_spectrum_has_its_own_parameters():
    na = Site("23Na", 5, label="Na1", fwhm=20).add_quadrupolar(2.5, 0.3)
    cu = Site("63Cu", 2380, label="Cu", fwhm=60, amount=0.3)
    x = np.linspace(-1500, 6000, 4096)
    y = simulate_combined([na, cu], x, NU0_NA, transitions="ct", n_zcw=N_ZCW)["real_spec"]
    y = y / y.max() + np.random.default_rng(1).normal(0, 0.003, x.size)
    res = {"x_ppm": x, "real_spectrum": y, "SFO1": NU0_NA / 1e6, "nuclide": "23Na"}
    fit = cf.fit_combined(res, [Site("23Na", 0, label="Na1", fwhm=30).add_quadrupolar(2.0, 0.5),
                                Site("63Cu", 2300, label="Cu", fwhm=100)],
                          transitions="ct", n_zcw=N_ZCW)
    v = {(p["label"], p["name"]): p["value"] for p in fit["parameters"]}
    assert v[("Cu", "delta_iso")] == pytest.approx(2380, abs=1)     # on 63Cu's own scale
    assert v[("Cu", "fwhm")] == pytest.approx(60, rel=0.02)
    assert v[("Na1", "C_Q")] == pytest.approx(2.5, abs=0.01)
    assert fit["settings"]["observe"] == "23Na"
    assert "63Cu on the 23Na scale" in fit["log"]


def test_advice_about_a_partner_goes_to_its_parent():
    # data with a CSA the model lacks: any suggestion naming the partner is re-aimed at Ti1
    truth = _ti().add_csa(150.0, 0.0)
    res = _ti_spectrum([truth, truth.isotope_partner()], noise=0.002)
    start = _ti(-790, 3.9, 0.25, 9.0)
    fit = cf.fit_combined(res, [start, start.isotope_partner()], transitions="ct", n_zcw=N_ZCW)
    items = fit_advice.advise_combined_fit(fit)
    nexts = [it for it in items if it["section"] == "next"]
    assert nexts
    for it in nexts:
        assert (it["action"] or {}).get("site") != "Ti1 (49Ti)"
        assert "Ti1 (49Ti)" not in it["text"]
