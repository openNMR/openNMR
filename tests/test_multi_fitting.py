"""Fitting one set of sites to several spectra (notebook/multi_fitting.py):
a series fitted in a given order and seeded forward, isotopes linked
through shielding, EFG (Q ratio) and J (γ ratio), fields linked with a
Hz + ppm linewidth split -- each against synthetic spectra with known
answers."""

import numpy as np
import pytest

from opennmr.notebook import multi_fitting as MF
from opennmr.notebook.combined_hamiltonian import Site, simulate_combined


def _spectrum(sites, nucleus, larmor_Hz, lo, hi, n=4096, mas=0.0, transitions="ct",
              noise=2.0, seed=0, n_zcw=6765):
    x = np.linspace(hi, lo, n)
    y = simulate_combined(sites, x[::-1], larmor_Hz, mas_Hz=mas, transitions=transitions,
                          n_zcw=n_zcw)["real_spec"][::-1]
    y = y / np.max(y) * 1000 + np.random.default_rng(seed).standard_normal(n) * noise
    return {"x_ppm": x, "real_spectrum": y, "SFO1": larmor_Hz / 1e6, "nuclide": nucleus}


def _values(fit):
    return {f"{p['label']}.{p['name']}": p for p in fit["parameters"]}


def _al(delta=12.0, cq=4.2, eta=0.35, fwhm=0.6, amount=1.0, label="Al1"):
    return Site("27Al", delta, label=label, fwhm=fwhm, amount=amount).add_quadrupolar(cq, eta)


# ------------------------------------------------------------------
# Scopes and links
# ------------------------------------------------------------------

def test_presets():
    scopes, amounts = MF.default_scopes("series")
    assert set(scopes.values()) == {"experiment"} and amounts == "experiment"
    scopes, amounts = MF.default_scopes("isotopes")
    assert scopes["delta_iso"] == "global" and scopes["C_Q"] == "global" and scopes["J"] == "global"
    assert scopes["fwhm"] == "experiment" and amounts == "nucleus"
    scopes, amounts = MF.default_scopes("fields")
    assert scopes["fwhm"] == "nucleus" and amounts == "nucleus"


def test_efg_is_shared_through_the_quadrupole_moment_and_j_through_gamma():
    # 6Li/7Li: C_Q scales by Q(6Li)/Q(7Li); 1H/2H: J to the same partner
    # scales by gamma(2H)/gamma(1H).
    res = {"x_ppm": np.linspace(10, -10, 64), "real_spectrum": np.zeros(64), "SFO1": 100.0}
    exps = [MF.make_experiment(res, "7Li", "7Li"), MF.make_experiment(res, "6Li", "6Li"),
            MF.make_experiment(res, "1H", "1H"), MF.make_experiment(res, "2H", "2H")]
    sites = {"7Li": [Site("7Li", 0, label="Li").add_quadrupolar(0.1, 0.2)],
             "6Li": [Site("6Li", 0, label="Li").add_quadrupolar(0.1, 0.2)],
             "1H": [Site("1H", 5, label="PH").add_jcoupling("31P", 1, 600)],
             "2H": [Site("2H", 5, label="PH").add_quadrupolar(0.1, 0.0).add_jcoupling("31P", 1, 90)]}
    pmap = MF._Map(exps, MF._templates(sites), MF.default_scopes("isotopes")[0], "nucleus",
                   None, None, None)
    k_li = pmap.links[1]["Li"]["C_Q"]
    assert k_li[0] == "scaled" and k_li[2] == pytest.approx(0.0808 / 4.01, rel=1e-3)
    assert pmap.links[1]["Li"]["eta_Q"][2] == 1.0
    k_j = pmap.links[3]["PH"]["J1"][2]
    assert k_j == pytest.approx(MF.gyromagnetic_ratio("2H") / MF.gyromagnetic_ratio("1H"), rel=1e-6)
    # delta_iso shared, with a per-isotope offset that starts fixed at 0.
    link = pmap.links[3]["PH"]["delta_iso"]
    assert link[0] == "offset" and pmap.keys[link[2]]["free"] is False
    # 2H's C_Q has nothing on 1H to share with: its own key.
    assert pmap.links[3]["PH"]["C_Q"][2] == 1.0


# ------------------------------------------------------------------
# 1. A series, in the order given, seeded forward
# ------------------------------------------------------------------

def _series(cqs, labels):
    exps = []
    for k, (cq, label) in enumerate(zip(cqs, labels)):
        res = _spectrum([_al(cq=cq)], "27Al", 130.3e6, -150, 100, mas=20e3, seed=k)
        exps.append(MF.make_experiment(res, label, mas_Hz=20e3, transitions="ct", n_zcw=987,
                                       x_value=300 + 20 * k))
    return exps


def test_a_series_is_fitted_in_the_given_order_each_seeded_from_the_last():
    labels = ["5", "302", "12", "2"]
    cqs = [4.0, 4.1, 4.2, 4.3]
    exps = _series(cqs, labels)
    start = {"27Al": [_al(delta=10, cq=3.6, eta=0.5, fwhm=1.0)]}
    seen = []
    result = MF.fit_multi(exps, start, preset="series",
                          progress=lambda d, t, m: seen.append(m) or True)
    assert result["mode"] == "sequential" and result["shared"] == []
    assert [f["label"] for f in result["experiments"]] == labels
    for fit, cq in zip(result["experiments"], cqs):
        assert _values(fit)["Al1.C_Q"]["value"] == pytest.approx(cq, abs=0.02)
    assert seen[0].startswith("Fitting 5 (1 of 4)") and seen[1].startswith("Fitting 302")
    x, v, e, labs = MF.parameter_trend(result, "Al1", "C_Q")
    assert x == [300, 320, 340, 360] and labs == labels and np.all(e > 0)
    assert np.all(np.diff(v) > 0)


def test_a_shared_parameter_in_a_series_is_fitted_once_jointly():
    exps = _series([4.2, 4.2, 4.2], ["a", "b", "c"])
    start = {"27Al": [_al(delta=10, cq=3.9, eta=0.5, fwhm=1.0)]}
    result = MF.fit_multi(exps, start, preset="series", scopes={"C_Q": "global", "eta_Q": "global"})
    assert result["mode"] == "joint"
    shared = {r["name"]: r for r in result["shared"]}
    assert shared["C_Q"]["value"] == pytest.approx(4.2, abs=0.01)
    assert shared["C_Q"]["group"] == "all"
    per = [_values(f)["Al1.C_Q"] for f in result["experiments"]]
    assert all(p["value"] == pytest.approx(shared["C_Q"]["value"]) and p["scope"] == "global"
               for p in per)
    # δiso stayed per experiment.
    assert all(_values(f)["Al1.delta_iso"]["scope"] == "experiment" for f in result["experiments"])
    assert "Joint refinement" in result["log"]


def test_cancel_stops_the_fit():
    exps = _series([4.2, 4.2], ["a", "b"])
    with pytest.raises(MF.MultiFitCancelled):
        MF.fit_multi(exps, {"27Al": [_al()]}, progress=lambda d, t, m: d < 1)


# ------------------------------------------------------------------
# 3. One isotope at two fields and two spinning rates
# ------------------------------------------------------------------

def test_two_fields_share_the_chemistry_and_split_the_linewidth():
    # Truth: FWHM(ppm) = sqrt((a/nu0)^2 + b^2), a = 150 Hz, b = 0.4 ppm.
    a, b = 150.0, 0.4
    exps = []
    for k, (nu0, mas) in enumerate(((104.3e6, 14e3), (156.4e6, 22e3))):
        fwhm = float(np.hypot(a / (nu0 / 1e6), b))
        sites = [_al(fwhm=fwhm, amount=2.0), _al(delta=-8, cq=6.0, eta=0.7, fwhm=fwhm, label="Al2")]
        res = _spectrum(sites, "27Al", nu0, -200, 120, n=6000, mas=mas, seed=k, noise=1.0)
        exps.append(MF.make_experiment(res, f"{nu0 / 1e6:.0f} MHz", mas_Hz=mas, transitions="ct",
                                       n_zcw=987))
    start = {"27Al": [_al(delta=10.5, cq=4.0, eta=0.5, fwhm=1.0),
                      _al(delta=-6, cq=5.7, eta=0.5, fwhm=1.0, label="Al2")]}
    result = MF.fit_multi(exps, start, preset="fields")
    shared = {(r["label"], r["name"]): r for r in result["shared"]}
    assert shared[("Al1", "delta_iso")]["value"] == pytest.approx(12.0, abs=0.03)
    assert shared[("Al1", "C_Q")]["value"] == pytest.approx(4.2, abs=0.01)
    assert shared[("Al2", "C_Q")]["value"] == pytest.approx(6.0, abs=0.02)
    assert shared[("Al1", "width_ppm")]["value"] == pytest.approx(b, abs=0.1)
    assert shared[("Al1", "width_Hz")]["value"] == pytest.approx(a, rel=0.3)
    # The site ratio is shared; one scale is held, the other fitted.
    for fit in result["experiments"]:
        assert fit["derived"]["Al1"]["share_%"] == pytest.approx(200 / 3, abs=1.5)
    scales = [r for r in result["shared"] if r["name"] == "scale"]
    assert [r["free"] for r in scales] == [False, True]
    assert all(f["quality"]["r2"] > 0.99 for f in result["experiments"])


def test_one_field_cannot_split_a_linewidth_and_says_so():
    exps = _series([4.2, 4.2], ["a", "b"])
    result = MF.fit_multi(exps, {"27Al": [_al(delta=11, cq=4.0)]}, preset="fields")
    assert any("width_Hz held at 0" in w for w in result["warnings"])
    width = [r for r in result["shared"] if r["name"] == "width_Hz"][0]
    assert width["value"] == 0.0 and not width["free"]


# ------------------------------------------------------------------
# 2. Two isotopes of one material
# ------------------------------------------------------------------

def test_isotopes_share_the_shift_and_scale_j_by_gamma():
    k = MF.gyromagnetic_ratio("2H") / MF.gyromagnetic_ratio("1H")
    h = [Site("1H", 5.0, label="PH", fwhm=0.3).add_jcoupling("31P", 1, 600.0)]
    d = [Site("2H", 5.0, label="PH", fwhm=0.3).add_jcoupling("31P", 1, 600.0 * k)]
    exps = [MF.make_experiment(_spectrum(h, "1H", 400e6, 0, 10, n=4000, transitions="all",
                                         n_zcw=10), "1H", transitions="all", n_zcw=10),
            MF.make_experiment(_spectrum(d, "2H", 61.4e6, -10, 20, n=4000, transitions="all",
                                         n_zcw=10, seed=1), "2H", transitions="all", n_zcw=10)]
    start = {"1H": [Site("1H", 5.1, label="PH", fwhm=0.5).add_jcoupling("31P", 1, 550.0)],
             "2H": [Site("2H", 4.9, label="PH", fwhm=0.5).add_jcoupling("31P", 1, 80.0)]}
    result = MF.fit_multi(exps, start, preset="isotopes")
    shared = {(r["label"], r["name"]): r for r in result["shared"]}
    assert shared[("PH", "delta_iso")]["value"] == pytest.approx(5.0, abs=0.005)
    assert shared[("PH", "J1")]["value"] == pytest.approx(600.0, abs=2.0)
    j2h = _values(result["experiments"][1])["PH.J1"]
    assert j2h["value"] == pytest.approx(shared[("PH", "J1")]["value"] * k)
    assert j2h["error"] == pytest.approx(shared[("PH", "J1")]["error"] * k, rel=1e-3)


def test_a_freed_isotope_offset_is_fitted():
    h = [Site("1H", 5.00, label="OH", fwhm=0.2)]
    d = [Site("2H", 5.25, label="OH", fwhm=0.2)]
    exps = [MF.make_experiment(_spectrum(h, "1H", 400e6, 0, 10, transitions="all", n_zcw=10),
                               "1H", transitions="all", n_zcw=10),
            MF.make_experiment(_spectrum(d, "2H", 61.4e6, 0, 10, transitions="all", n_zcw=10,
                                         seed=1), "2H", transitions="all", n_zcw=10)]
    start = {"1H": [Site("1H", 5.1, label="OH", fwhm=0.3)],
             "2H": [Site("2H", 5.1, label="OH", fwhm=0.3)]}
    held = MF.fit_multi(exps, start, preset="isotopes")
    shared = {(r["label"], r["name"]): r for r in held["shared"]}
    assert shared[("OH", "delta_offset")]["value"] == 0.0
    freed = MF.fit_multi(exps, start, preset="isotopes",
                         vary={"OH": ["delta_iso", "delta_offset", "amount", "fwhm"]})
    shared = {(r["label"], r["name"]): r for r in freed["shared"]}
    assert shared[("OH", "delta_iso")]["value"] == pytest.approx(5.00, abs=0.01)
    assert shared[("OH", "delta_offset")]["value"] == pytest.approx(0.25, abs=0.01)
    assert _values(freed["experiments"][1])["OH.delta_iso"]["value"] == pytest.approx(5.25, abs=0.01)


# ------------------------------------------------------------------
# Output
# ------------------------------------------------------------------

def test_tables_csv_report_and_plots(tmp_path):
    exps = _series([4.1, 4.3], ["5", "302"])
    result = MF.fit_multi(exps, {"27Al": [_al(delta=11, cq=4.0)]}, preset="series")
    rows = MF.multi_fit_table(result)
    assert {r["experiment"] for r in rows} == {"5", "302"}
    text = MF.multi_fit_to_csv(result, path=str(tmp_path / "t.csv"))
    header = text.splitlines()[0].split(",")
    assert header[:2] == ["experiment", "x_value"] and "Al1: C_Q" in header and "Al1: C_Q error" in header
    assert text.splitlines()[1].startswith("5,300,")
    report = MF.print_multi_fit_report(result)
    assert "MULTI-SPECTRUM FIT REPORT" in report and "302" in report
    fig = MF.plot_multi_fit(result)
    assert len(fig.axes) == 2
    ax = MF.plot_parameter_trend(result, "Al1", "C_Q", xlabel="T (K)")
    assert ax.get_xlabel() == "T (K)"


def test_bad_inputs_are_named():
    res = {"x_ppm": np.linspace(10, -10, 64), "real_spectrum": np.zeros(64), "SFO1": 100.0,
           "nuclide": "1H"}
    with pytest.raises(ValueError, match="unique"):
        MF.fit_multi([MF.make_experiment(res, "a"), MF.make_experiment(res, "a")],
                     {"1H": [Site("1H", 0)]})
    with pytest.raises(ValueError, match="no sites are defined for 1H"):
        MF.fit_multi([MF.make_experiment(res, "a")], {"13C": [Site("13C", 0)]})
    with pytest.raises(ValueError, match="scope"):
        MF.fit_multi([MF.make_experiment(res, "a")], {"1H": [Site("1H", 0)]},
                     scopes={"delta_iso": "everywhere"})
