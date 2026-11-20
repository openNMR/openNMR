"""Fitting motional lineshapes (notebook/motion_fit.py): parameters, a global VT fit, a rate per
spectrum with the Arrhenius line, and what the fits refuse or flag."""

import math

import numpy as np
import pytest

from opennmr.notebook import motion as M
from opennmr.notebook import motion_fit as F
from opennmr.notebook.combined_hamiltonian import Site
from opennmr.notebook.exchange import K_B_EV
from opennmr.notebook.vt_relaxation import fit_arrhenius

LARMOR = 61.4e6


def _model(beta=40.0, Ea=0.30, nu0=1e12, k=None, fwhm=20.0):
    site = Site("2H", 0.0, label="D", fwhm=fwhm, lineshape="lorentzian").add_quadrupolar(0.17, 0.0)
    if k is not None:
        return M.cone_jump(site, beta, 3, k_Hz=k)
    return M.cone_jump(site, beta, 3, Ea_eV=Ea, nu0_Hz=nu0)


def _series(model, temps, n=256, noise=0.01, seed=0):
    x = np.linspace(2500, -2500, n)
    rng = np.random.default_rng(seed)
    out = []
    for T in temps:
        y = M.motion_spectrum(model, x, larmor_Hz=LARMOR, T_K=T)["real_spec"]
        item = {"ppm": x, "y": y / y.max() + rng.normal(0, noise, n)}
        if T is not None:
            item["T_K"] = float(T)
        out.append(item)
    return out


@pytest.fixture(scope="module")
def vt():
    return _series(_model(), (230.0, 290.0, 350.0))


# ------------------------------------------------------------ parameter names

def test_names_are_parsed_and_unknown_ones_refused():
    assert F._parse_name("mode1.Ea_eV") == ("mode", 0, None, "Ea_eV")
    assert F._parse_name("hop2.log10_k_ref") == ("hop", 1, None, "log10_k_ref")
    assert F._parse_name("site3.dipolar2.D_Hz") == ("site", 2, 1, "D_Hz")
    assert F._parse_name("gauss_fwhm_ppm")[0] == "gauss"
    for bad in ("mode0.Ea_eV", "mode1.beta", "site1.Ea_eV", "nonsense", "mode1", "hop1.alpha"):
        with pytest.raises(ValueError):
            F._parse_name(bad)


def test_the_parameter_list_marks_what_is_usually_fitted():
    rows = {p["name"]: p for p in F.fit_parameter_list(_model(), T_ref_K=300.0)}
    assert rows["mode1.Ea_eV"]["default_free"] and rows["mode1.log10_k_ref"]["default_free"]
    assert not rows["mode1.log10_nu0"]["default_free"] and not rows["site1.beta"]["default_free"]
    assert rows["mode1.Ea_eV"]["value"] == 0.30
    assert rows["mode1.log10_k_ref"]["value"] == pytest.approx(
        12.0 - 0.30 / (K_B_EV * 300.0 * math.log(10)), rel=1e-9)
    assert rows["site1.beta"]["value"] == 40.0 and rows["site1.C_Q"]["value"] == 0.17
    assert (rows["site1.beta"]["min"], rows["site1.beta"]["max"]) == (0.0, 180.0)
    fixed = {p["name"]: p for p in F.fit_parameter_list(_model(k=1e5))}
    assert fixed["mode1.log10_k"]["default_free"] and "mode1.Ea_eV" not in fixed


def test_the_reference_rate_keeps_the_rate_at_the_reference_temperature():
    d = _model().to_dict()
    F._apply(d, {"mode1.Ea_eV": 0.45, "mode1.log10_k_ref": 6.0}, 300.0)
    m = M.MotionModel.from_dict(d)
    assert m.modes[0].rate_value(300.0) == pytest.approx(1e6, rel=1e-9)
    assert m.modes[0].rate["Ea_eV"] == 0.45
    d = _model().to_dict()
    F._apply(d, {"mode1.log10_k": 3.0, "site1.beta": 55.0, "site1.fwhm": 10.0}, None)
    m = M.MotionModel.from_dict(d)
    assert m.modes[0].rate == {"k_Hz": 1000.0} and m.sites[0].euler[1] == 55.0
    assert m.sites[0].site.fwhm == 10.0


# ---------------------------------------------------------------- global fit

def test_a_vt_series_gives_back_its_barrier_and_rate(vt):
    fit = F.fit_motion(_model(Ea=0.22, nu0=1e11), vt, ["mode1.Ea_eV", "mode1.log10_k_ref"],
                       larmor_Hz=LARMOR, n_points=256)
    assert fit["success"]
    assert fit["params"]["mode1.Ea_eV"] == pytest.approx(0.30, abs=0.01)
    truth = math.log10(_model().modes[0].rate_value(fit["T_ref_K"]))
    assert fit["params"]["mode1.log10_k_ref"] == pytest.approx(truth, abs=0.05)
    nu0, e_nu0 = fit["derived"]["mode1.log10_nu0"]
    assert nu0 == pytest.approx(12.0, abs=0.3) and e_nu0 > fit["errors"]["mode1.log10_k_ref"]
    # the reference temperature sits inside the data, in 1/T
    assert 230.0 < fit["T_ref_K"] < 350.0
    assert fit["model"].modes[0].rate["Ea_eV"] == fit["params"]["mode1.Ea_eV"]
    assert len(fit["overlay"]["overlays"]) == 3 and fit["overlay"]["overlays"][0]["rms"] < 0.03
    assert "Errors are formal" in fit["log"] and "mode1.log10_nu0" in fit["log"]
    assert fit["redchi"] > 0 and np.isfinite(fit["correlation"]).all()


def test_a_geometry_parameter_and_a_rate_are_fitted_together_on_one_spectrum():
    truth = _model(k=2e5)
    spec = _series(truth, (None,), n=256, seed=3)
    fit = F.fit_motion(_model(beta=47.0, k=5e4), spec, ["site1.beta", "mode1.log10_k"],
                       larmor_Hz=LARMOR, n_points=256)
    assert fit["params"]["site1.beta"] == pytest.approx(40.0, abs=0.5)
    assert fit["params"]["mode1.log10_k"] == pytest.approx(math.log10(2e5), abs=0.05)


def test_a_fixed_parameter_is_held_and_bounds_are_flagged(vt):
    fit = F.fit_motion(_model(Ea=0.22, nu0=1e11), vt,
                       {"mode1.Ea_eV": {"value": 0.35, "min": 0.31, "max": 0.5},
                        "mode1.log10_k_ref": {}, "site1.beta": {"value": 41.0, "vary": False}},
                       larmor_Hz=LARMOR, n_points=256)
    assert fit["fixed"] == {"site1.beta": 41.0} and fit["model"].sites[0].euler[1] == 41.0
    assert fit["params"]["mode1.Ea_eV"] == pytest.approx(0.31, abs=1e-3)
    assert "mode1.Ea_eV" in fit["at_bound"] and "AT A BOUND" in fit["log"]


def test_search_starts_are_reported(vt):
    fit = F.fit_motion(_model(Ea=0.22, nu0=1e11), vt,
                       {"mode1.Ea_eV": (0.1, 0.5), "mode1.log10_k_ref": (4.0, 8.0)},
                       larmor_Hz=LARMOR, n_points=256, n_starts=2, search_nfev=4)
    assert len(fit["starts"]) == 3 and "Search: 3 coarse fits" in fit["log"]
    assert fit["params"]["mode1.Ea_eV"] == pytest.approx(0.30, abs=0.02)


def test_the_fit_refuses_what_cannot_be_determined(vt):
    start = _model()
    one_T = [vt[0]]
    with pytest.raises(ValueError, match="cannot both be fitted"):
        F.fit_motion(start, one_T, ["mode1.Ea_eV", "mode1.log10_nu0"], larmor_Hz=LARMOR)
    with pytest.raises(ValueError, match="same freedom"):
        F.fit_motion(start, vt, ["mode1.log10_nu0", "mode1.log10_k_ref"], larmor_Hz=LARMOR)
    with pytest.raises(ValueError, match="does not depend on temperature"):
        F.fit_motion(_model(k=1e5), vt, ["mode1.log10_k"], larmor_Hz=LARMOR)
    with pytest.raises(ValueError, match="fixed rate"):
        F.fit_motion(_model(k=1e5), vt, ["mode1.Ea_eV"], larmor_Hz=LARMOR)
    no_T = [{"ppm": s["ppm"], "y": s["y"]} for s in vt]
    with pytest.raises(ValueError, match="temperature"):
        F.fit_motion(start, no_T, ["mode1.Ea_eV"], larmor_Hz=LARMOR)
    with pytest.raises(ValueError, match="Every parameter is held fixed"):
        F.fit_motion(start, vt, {"site1.beta": {"vary": False}}, larmor_Hz=LARMOR)
    with pytest.raises(ValueError, match="outside its bounds"):
        F.fit_motion(start, vt, {"site1.beta": {"value": 200.0}}, larmor_Hz=LARMOR)
    with pytest.raises(ValueError, match="larmor"):
        F.fit_motion(start, vt, ["mode1.Ea_eV"])
    with pytest.raises(ValueError, match="Choose at least one"):
        F.fit_motion(start, vt, [], larmor_Hz=LARMOR)
    with pytest.raises(ValueError, match="no mode 2|no mode"):
        F.fit_motion(start, vt, ["mode2.Ea_eV"], larmor_Hz=LARMOR)


def test_a_coarse_simulation_grid_is_flagged():
    model = _model(fwhm=3.0, k=1e5)
    spec = _series(model, (None,), n=256)
    fit = F.fit_motion(_model(fwhm=3.0, k=1e5), spec, ["mode1.log10_k"], larmor_Hz=LARMOR,
                       n_points=256, max_nfev=3)
    assert "simulation grid" in fit["log"] and "n_points" in fit["log"]


# ------------------------------------------------------------ rate per spectrum

def test_a_rate_per_spectrum_follows_the_truth_and_flags_the_limit(vt):
    temps = (200.0, 260.0, 320.0, 700.0)
    spectra = _series(_model(), temps, n=256, seed=1)
    out = F.fit_motion_rates(_model(Ea=0.2, nu0=1e11), spectra, mode=1, larmor_Hz=LARMOR,
                             n_points=256)
    truth = [_model().modes[0].rate_value(T) for T in temps]
    for k_fit, k_true, ok in list(zip(out["k_Hz"], truth, out["determined"]))[:3]:
        assert ok and abs(math.log10(k_fit / k_true)) < 0.1
    assert not out["determined"][3]                      # the fast limit: no rate to read
    ar = out["arrhenius"]
    assert ar["n"] == 3 and ar["Ea_eV"] == pytest.approx(0.30, abs=0.03)
    assert "NOT determined" in out["log"] and "Arrhenius line through 3 rates" in out["log"]
    assert len(out["overlay"]["overlays"]) == 4


def test_the_rate_scan_range_is_checked_and_temperatures_are_optional():
    spec = _series(_model(k=1e5), (None,), n=256)
    out = F.fit_motion_rates(_model(k=3e4), spec, mode=1, larmor_Hz=LARMOR, n_points=256)
    assert out["determined"][0] and abs(math.log10(out["k_Hz"][0] / 1e5)) < 0.1
    assert out["arrhenius"] is None
    with pytest.raises(ValueError):
        F.fit_motion_rates(_model(k=3e4), spec, mode=2, larmor_Hz=LARMOR)
    with pytest.raises(ValueError):
        F.fit_motion_rates(_model(k=3e4), spec, mode=1, k_range=(1e6, 1e2), larmor_Hz=LARMOR)


def test_the_arrhenius_regression_recovers_a_line_and_agrees_with_the_t1_line_fit():
    T = np.array([200.0, 250.0, 300.0, 350.0])
    k = 1e12 * np.exp(-0.35 / (K_B_EV * T))
    ar = F.arrhenius_regression(T, k)
    assert ar["Ea_eV"] == pytest.approx(0.35, rel=1e-9) and ar["log10_nu0"] == pytest.approx(12.0)
    assert ar["redchi"] == pytest.approx(0.0, abs=1e-12)
    old = fit_arrhenius(T, k, is_rate=True)
    assert ar["Ea_eV"] == pytest.approx(old["Ea_eV"], rel=1e-9)
    noisy = F.arrhenius_regression(T, k * np.array([1.1, 0.95, 1.05, 0.9]),
                                   sigma_log10k=[0.03] * 4)
    assert noisy["Ea_err_eV"] > 0 and noisy["log10_nu0_err"] > noisy["Ea_err_eV"]
    two = F.arrhenius_regression(T[:2], k[:2])
    assert two["redchi"] is None and np.isnan(two["Ea_err_eV"])
    with pytest.raises(ValueError):
        F.arrhenius_regression([300.0], [1e5])


# --------------------------------------------------------------------- output

def test_figures_and_notebook_code(vt):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fit = F.fit_motion(_model(Ea=0.25, nu0=1e11), vt, ["mode1.Ea_eV", "mode1.log10_k_ref"],
                       larmor_Hz=LARMOR, n_points=256, max_nfev=6)
    fig, (ax1, ax2) = F.plot_motion_fit(fit, nucleus="2H")
    assert ax2 is not None and len(ax1.lines) >= 6 and ax2.get_legend() is None
    assert ax1.get_xlim()[0] > ax1.get_xlim()[1]
    plt.close(fig)
    out = F.fit_motion_rates(_model(k=3e4), _series(_model(k=1e5), (250.0,), n=256),
                             mode=1, larmor_Hz=LARMOR, n_points=256)
    fig, _axes = F.plot_motion_fit(out, nucleus="2H")
    plt.close(fig)
    with pytest.raises(ValueError, match="temperatures"):
        no_T = dict(out, T_K=np.array([np.nan]))
        F.plot_motion_rates(no_T)
    code = F.motion_fit_code(_model(), {"mode1.Ea_eV": {"value": 0.3, "min": 0.0, "max": np.inf}},
                             T_ref_K=280.0, kwargs={"larmor_Hz": LARMOR}, n_starts=3,
                             nucleus="2H")
    compile(code, "code", "exec")
    assert "motion_fit.fit_motion(model, spectra, parameters, T_ref_K=280.0, n_starts=3" in code
    rates_code = F.motion_fit_code(_model(), {}, route="rates", mode=1, kwargs={"larmor_Hz": LARMOR})
    compile(rates_code, "code", "exec")
    assert "fit_motion_rates(model, spectra, mode=1" in rates_code


def test_a_dipolar_coupling_can_be_fitted_in_the_notebook():
    def model(D):
        site = Site("13C", 0.0, label="C", fwhm=15.0, lineshape="lorentzian").add_csa(40.0, 0.0)
        ms = M.MotionSite(site, euler=(0.0, 60.0, 0.0)).add_dipolar("1H", D_Hz=D)
        return M.model_from_spec([ms], [{"kind": "ring_jump", "n": 3, "axis": "z", "k_Hz": 2e4}])

    assert [p["name"] for p in F.fit_parameter_list(model(23300.0)) if "dipolar" in p["name"]] == [
        "site1.dipolar1.D_Hz"]
    x = np.linspace(400, -400, 256)
    y = M.motion_spectrum(model(23300.0), x, larmor_Hz=100.6e6)["real_spec"]
    y = y / y.max() + np.random.default_rng(0).normal(0, 0.01, x.size)
    fit = F.fit_motion(model(18000.0), [{"ppm": x, "y": y}],
                       ["site1.dipolar1.D_Hz", "mode1.log10_k"], larmor_Hz=100.6e6, n_points=256)
    err = fit["errors"]["site1.dipolar1.D_Hz"]
    assert abs(fit["params"]["site1.dipolar1.D_Hz"] - 23300.0) < max(4 * err, 300.0)
    assert fit["params"]["mode1.log10_k"] == pytest.approx(math.log10(2e4), abs=0.05)
    assert fit["model"].sites[0].dipolar[0]["D_Hz"] == fit["params"]["site1.dipolar1.D_Hz"]
    with pytest.raises(ValueError, match="no dipolar coupling"):
        F.fit_motion(model(23300.0), [{"ppm": x, "y": y}], ["site1.dipolar2.D_Hz"],
                     larmor_Hz=100.6e6)
