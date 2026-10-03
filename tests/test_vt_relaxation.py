"""notebook/vt_relaxation.py: relaxation against temperature (BPP /
Arrhenius), Hendrickson-Bray linewidths, and their plots. Synthetic data
from the model itself with known parameters, plus the textbook limits
(BPP R1 maximum at w0*tau = 0.616, extreme narrowing R1 = R2, the
quadrupolar extreme-narrowing constant)."""
import math

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from opennmr.notebook import vt_relaxation as vt  # noqa: E402

T = np.linspace(200, 450, 18)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    import matplotlib.pyplot as plt
    plt.close("all")
TRUTH = {"C": 3e9, "tau0_s": 2e-13, "Ea_eV": 0.32}


def _dataset(larmor, noise=0.02, seed=1, quantity="T1", **truth):
    p = dict(TRUTH, **truth)
    rng = np.random.default_rng(seed)
    R = vt.relaxation_rate(T, larmor, p["tau0_s"], p["Ea_eV"], p["C"], quantity=quantity)
    value = (1 / R) * (1 + noise * rng.standard_normal(T.size))
    return vt.vt_dataset(T, value, noise * value, larmor_MHz=larmor, quantity=quantity,
                         label=f"{larmor:g} MHz")


# -------------------------------------------------------------- physics

def test_bpp_t1_minimum_at_omega_tau_0616():
    w0 = 2 * math.pi * 100e6
    taus = np.logspace(-11, -7, 40001)
    F = vt.spectral_density(w0, taus) + 4 * vt.spectral_density(2 * w0, taus)
    assert w0 * taus[np.argmax(F)] == pytest.approx(0.6158, abs=1e-3)
    assert F.max() * w0 == pytest.approx(1.4252, abs=1e-3)


def test_extreme_narrowing_r1_equals_r2_equals_r1rho():
    rates = [vt.relaxation_rate(2000.0, 100.0, 1e-13, 0.1, 1e9, q, spinlock_kHz=50)
             for q in ("T1", "T2", "T1rho")]
    assert rates[0] == pytest.approx(rates[1], rel=1e-6)
    assert rates[0] == pytest.approx(rates[2], rel=1e-6)


def test_r2_exceeds_r1_in_slow_motion():
    assert vt.relaxation_rate(150.0, 100.0, 1e-13, 0.4, 1e9, "T2") > \
        vt.relaxation_rate(150.0, 100.0, 1e-13, 0.4, 1e9, "T1")


def test_t1rho_needs_a_spinlock():
    with pytest.raises(ValueError):
        vt.relaxation_rate(300.0, 100.0, 1e-13, 0.3, 1e9, "T1rho")
    with pytest.raises(ValueError):
        vt.vt_dataset([300], [0.1], larmor_MHz=100, quantity="T1rho")


def test_quadrupolar_prefactor_extreme_narrowing():
    """In extreme narrowing R1 = (3 pi^2/10) (2I+3)/(I^2(2I-1)) (1+eta^2/3) CQ^2 tau_c."""
    I, eta, CQ, tau = 1.5, 0.3, 2e6, 1e-12
    A = vt.quadrupolar_prefactor(I, eta) * CQ ** 2
    R = A * 5 * tau                         # J(w) + 4 J(2w) -> 5 tau
    textbook = (3 * math.pi ** 2 / 10) * (2 * I + 3) / (I ** 2 * (2 * I - 1)) * (1 + eta ** 2 / 3) \
        * CQ ** 2 * tau
    assert R == pytest.approx(textbook)
    with pytest.raises(ValueError):
        vt.quadrupolar_prefactor(0.5)


def test_asymmetric_low_temperature_slope_is_beta_minus_one_times_ea():
    Ts = np.array([120.0, 130.0])            # deep in the slow-motion flank
    R = vt.relaxation_rate(Ts, 100.0, 1e-13, 0.4, 1e9, beta=1.5)
    x = 1 / (vt.K_B_EV * Ts)
    slope = (np.log(R[1]) - np.log(R[0])) / (x[1] - x[0])
    assert -slope == pytest.approx(0.5 * 0.4, rel=0.01)


def test_rate_maximum_temperature():
    Tm = vt.rate_maximum_temperature(100.0, 1e-13, 0.3)
    tau = 1e-13 * math.exp(0.3 / (vt.K_B_EV * Tm))
    assert 2 * math.pi * 100e6 * tau == pytest.approx(0.6158, abs=1e-3)


# ----------------------------------------------------------------- data

def test_vt_dataset_units_sorting_and_dropping():
    d = vt.vt_dataset([50, 0, 25], [200, 100, -1], [2, 1, 1], larmor_MHz=100,
                      temperature_unit="C", value_unit="ms")
    assert np.allclose(d["T_K"], [273.15, 323.15])
    assert np.allclose(d["value"], [0.1, 0.2])
    assert np.allclose(d["error"], [0.001, 0.002])
    assert d["dropped"] == 1


def test_vt_dataset_from_rates():
    d = vt.vt_dataset([300], [10.0], [1.0], larmor_MHz=100, value_is_rate=True)
    assert d["value"][0] == pytest.approx(0.1)
    assert d["error"][0] / d["value"][0] == pytest.approx(0.1)     # relative error kept


def test_load_vt_csv_groups_by_series_and_field(tmp_path):
    path = tmp_path / "t1.csv"
    path.write_text("T,T1,error,larmor_MHz,label\n300,0.1,0.01,100,a\n320,0.2,0.01,100,a\n"
                    "300,0.3,0.01,200,a\n")
    data = vt.load_vt_csv(str(path))
    assert [d["larmor_MHz"] for d in data] == [100.0, 200.0]
    assert data[0]["T_K"].size == 2
    bad = tmp_path / "bad.csv"
    bad.write_text("x,y\n1,2\n")
    with pytest.raises(ValueError):
        vt.load_vt_csv(str(bad))


# -------------------------------------------------------------- fitting

def test_joint_two_field_fit_recovers_the_motion():
    fit = vt.fit_vt_relaxation([_dataset(116.6), _dataset(194.4, seed=2)])
    p = fit["params"]
    assert p["Ea_eV"] == pytest.approx(0.32, abs=0.003)
    assert p["tau0_s"] == pytest.approx(2e-13, rel=0.08)
    assert p["C"] == pytest.approx(3e9, rel=0.04)
    assert fit["redchi"] == pytest.approx(1.0, abs=0.6)
    assert fit["known_sigma"]
    assert len(fit["T_max"]) == 2 and all(np.isfinite(t[0]) for t in fit["T_max"])
    assert "Ea_eV" in fit["log"] and "kJ/mol" in fit["log"]


def test_fit_without_errors_is_unweighted_and_says_so():
    d = _dataset(116.6)
    d["error"] = np.full(d["value"].size, np.nan)
    fit = vt.fit_vt_relaxation([d])
    assert not fit["known_sigma"]
    assert "unweighted" in fit["log"]
    assert fit["params"]["Ea_eV"] == pytest.approx(0.32, abs=0.01)


def test_fixed_parameters_are_held_and_reported():
    fit = vt.fit_vt_relaxation([_dataset(116.6)], fixed={"Ea_eV": 0.32})
    assert fit["params"]["Ea_eV"] == 0.32
    assert "Ea_eV" not in fit["free"]
    assert "(fixed)" in fit["log"]
    with pytest.raises(ValueError, match="Cannot fix"):
        vt.fit_vt_relaxation([_dataset(116.6)], fixed={"beta": 1.5})   # not a parameter here


def test_quadrupolar_fit_returns_cq_in_mhz():
    A = vt.quadrupolar_prefactor(1.5) * (0.12e6) ** 2
    d = _dataset(155.5, C=A)
    fit = vt.fit_vt_relaxation([d], mechanism="quadrupolar", spin=1.5)
    assert fit["params"]["CQ_MHz"] == pytest.approx(0.12, rel=0.03)
    with pytest.raises(ValueError):
        vt.fit_vt_relaxation([d], mechanism="quadrupolar")          # no spin


def test_two_processes_ordered_by_ea_without_relabelled_rivals():
    R = (vt.relaxation_rate(T, 155.5, 1e-13, 0.25, 4e9)
         + vt.relaxation_rate(T, 155.5, 1e-15, 0.45, 2e9))
    rng = np.random.default_rng(3)
    value = (1 / R) * (1 + 0.01 * rng.standard_normal(T.size))
    d = vt.vt_dataset(T, value, 0.01 * value, larmor_MHz=155.5)
    fit = vt.fit_vt_relaxation([d], n_processes=2, n_starts=32)
    assert fit["params"]["Ea_eV_1"] < fit["params"]["Ea_eV_2"]
    assert fit["params"]["Ea_eV_1"] == pytest.approx(0.25, abs=0.01)
    # a copy of the best solution with the processes swapped is not a rival
    for rival in fit["rivals"]:
        assert not (rival["params"]["Ea_eV_1"] == pytest.approx(fit["params"]["Ea_eV_1"], rel=0.02)
                    and rival["params"]["Ea_eV_2"] == pytest.approx(fit["params"]["Ea_eV_2"],
                                                                    rel=0.02))


def test_model_comparison_prefers_the_true_model():
    R = vt.relaxation_rate(T, 116.6, 2e-13, 0.32, 3e9, beta=1.5)
    rng = np.random.default_rng(4)
    value = (1 / R) * (1 + 0.01 * rng.standard_normal(T.size))
    d = vt.vt_dataset(T, value, 0.01 * value, larmor_MHz=116.6)
    fits = vt.compare_vt_models([d], [{}, {"asymmetric": True}], n_starts=12)
    assert fits[0]["model"]["asymmetric"]
    assert fits[0]["delta_aicc"] == 0
    assert fits[0]["params"]["beta"] == pytest.approx(1.5, abs=0.05)
    assert sum(f["weight"] for f in fits) == pytest.approx(1.0)


def test_t_range_and_too_few_points():
    fit = vt.fit_vt_relaxation([_dataset(116.6)], T_range=(200, 330))
    assert fit["datasets"][0]["T_K"].max() <= 330
    d = vt.vt_dataset([300, 310], [0.1, 0.2], larmor_MHz=100)
    with pytest.raises(ValueError, match="cannot determine"):
        vt.fit_vt_relaxation([d])


def test_refuses_mixed_quantities_missing_field_and_linewidths():
    with pytest.raises(ValueError, match="same quantity"):
        vt.fit_vt_relaxation([_dataset(100.0), _dataset(100.0, quantity="T2")])
    d = _dataset(100.0)
    d["larmor_MHz"] = None
    with pytest.raises(ValueError, match="Larmor"):
        vt.fit_vt_relaxation([d])
    lw = vt.vt_dataset(T, np.linspace(1000, 100, T.size), quantity="linewidth", value_unit="Hz")
    with pytest.raises(ValueError, match="hendrickson"):
        vt.fit_vt_relaxation([lw])


def test_bootstrap_intervals_bracket_the_estimate():
    fit = vt.fit_vt_relaxation([_dataset(116.6)], n_bootstrap=40, n_starts=4)
    lo, hi = fit["ci"]["Ea_eV"]
    assert lo <= fit["params"]["Ea_eV"] <= hi
    assert "bootstrap" in fit["log"]


def test_evaluate_fit_matches_the_model():
    fit = vt.fit_vt_relaxation([_dataset(116.6)])
    T_c, R = vt.evaluate_vt_fit(fit, T_K=[300.0])
    p = fit["params"]
    assert R[0] == pytest.approx(vt.relaxation_rate(300.0, 116.6, p["tau0_s"], p["Ea_eV"], p["C"]))


def test_arrhenius_line_on_each_flank():
    d = _dataset(116.6, noise=0.0)
    low = vt.fit_arrhenius(d["T_K"][:4], d["value"][:4])
    assert low["flank"] == "low-temperature"
    assert low["Ea_eV"] == pytest.approx(0.32, rel=0.03)
    high = vt.fit_arrhenius([900, 1000, 1100], 1 / vt.relaxation_rate(
        np.array([900, 1000, 1100.0]), 116.6, 2e-13, 0.32, 3e9))
    assert high["flank"] == "high-temperature"
    assert high["Ea_eV"] == pytest.approx(0.32, rel=0.05)
    with pytest.raises(ValueError):
        vt.fit_arrhenius([300], [0.1])


# ------------------------------------------------------- Hendrickson-Bray

def test_hendrickson_bray_limits_and_fit():
    A, B, Ea, D = 8000.0, 1.5e-3, 0.4, 300.0
    assert vt.hendrickson_bray(10.0, A, B, Ea, D) == pytest.approx(A + D)
    assert vt.hendrickson_bray(1e7, A, B, Ea, D) == pytest.approx(B + D, rel=1e-3)
    Tw = np.linspace(150, 450, 25)
    rng = np.random.default_rng(5)
    w = vt.hendrickson_bray(Tw, A, B, Ea, D)
    w = w * (1 + 0.01 * rng.standard_normal(Tw.size))
    fit = vt.fit_hendrickson_bray(Tw, w, 0.01 * w)
    assert fit["params"]["Ea_eV"] == pytest.approx(0.4, abs=0.01)
    assert fit["params"]["D"] == pytest.approx(300, rel=0.03)
    assert fit["T_inflection_K"] == pytest.approx(Ea / (vt.K_B_EV * math.log(A / B - 1)), rel=0.03)
    assert fit["T_onset_K"] < fit["T_inflection_K"]
    assert fit["waugh_fedin_Ea_eV"] == pytest.approx(1.617e-3 * fit["T_onset_K"])
    assert "Waugh-Fedin" in fit["log"]


def test_hendrickson_bray_needs_enough_points():
    with pytest.raises(ValueError):
        vt.fit_hendrickson_bray([200, 300, 400], [1000, 500, 100])


# ---------------------------------------------------------------- plots

def test_arrhenius_plot_has_temperature_on_top():
    fit = vt.fit_vt_relaxation([_dataset(116.6), _dataset(194.4, seed=2)])
    fig, ax = vt.plot_arrhenius(fit)
    assert "1000" in ax.get_xlabel()
    assert "ln" in ax.get_ylabel()
    tops = [c for c in ax.child_axes if type(c).__name__ == "SecondaryAxis"]
    assert len(tops) == 1
    # the top axis maps 1000/T back to T: 1000/T = 4 is 250 K
    top = tops[0]
    fwd = top._functions[0]
    assert float(fwd(np.array([4.0]))[0]) == pytest.approx(250.0)
    ticks = top.get_xticks()
    assert len(ticks) >= 3 and np.all(np.diff(ticks) > 0)
    assert ax.get_legend() is not None             # two datasets
    assert any("E" in t.get_text() for t in ax.texts)


@pytest.mark.parametrize("x_axis, top", [("1/kBT", "C"), ("T", "K"), ("1000/T", None)])
@pytest.mark.parametrize("y_axis", ["ln_time", "ln_rate", "log10_time", "time", "rate"])
def test_arrhenius_plot_axis_choices(x_axis, top, y_axis):
    fig, ax = vt.plot_arrhenius(datasets=[_dataset(116.6)], x_axis=x_axis, y_axis=y_axis,
                                top_axis=top)
    n_top = len([c for c in ax.child_axes if type(c).__name__ == "SecondaryAxis"])
    assert n_top == (0 if (top is None or x_axis == "T") else 1)
    if y_axis in ("time", "rate"):
        assert ax.get_yscale() == "log"


def test_components_do_not_blow_up_the_y_range():
    R = (vt.relaxation_rate(T, 155.5, 1e-13, 0.25, 4e9)
         + vt.relaxation_rate(T, 155.5, 1e-16, 0.6, 1e9))
    d = vt.vt_dataset(T, 1 / R, 0.01 / R, larmor_MHz=155.5)
    fit = vt.fit_vt_relaxation([d], n_processes=2, n_starts=16)
    fig, ax = vt.plot_arrhenius(fit, show_components=True)
    lo, hi = ax.get_ylim()
    data = np.log(d["value"])
    assert lo > data.min() - 1.0 and hi < data.max() + 1.0


def test_vs_temperature_and_hb_plots():
    fig, ax = vt.plot_vs_temperature(fit=vt.fit_vt_relaxation([_dataset(116.6)]),
                                     temperature_unit="C")
    assert "°C" in ax.get_xlabel() and ax.get_yscale() == "log"
    Tw = np.linspace(150, 450, 20)
    fit = vt.fit_hendrickson_bray(Tw, vt.hendrickson_bray(Tw, 8000, 1.5e-3, 0.4, 300))
    fig, ax = vt.plot_hendrickson_bray(fit, unit="kHz")
    assert "kHz" in ax.get_ylabel()


def test_notebook_code_runs_and_reproduces_the_fit():
    data = [_dataset(116.6), _dataset(194.4, seed=2)]
    gui_fit = vt.fit_vt_relaxation(data, n_starts=8)
    code = vt.vt_notebook_code(data, {"n_starts": 8}, {"top_axis": "K"})
    ns = {}
    exec(compile(code, "<notebook>", "exec"), ns)
    assert ns["fit"]["params"]["Ea_eV"] == pytest.approx(gui_fit["params"]["Ea_eV"], rel=1e-6)
    hb_data = [vt.vt_dataset(np.linspace(150, 450, 20),
                             vt.hendrickson_bray(np.linspace(150, 450, 20), 8000, 1.5e-3, 0.4, 300),
                             quantity="linewidth", value_unit="Hz")]
    ns = {}
    exec(compile(vt.vt_notebook_code(hb_data, hendrickson=True), "<nb>", "exec"), ns)
    assert ns["fit"]["params"]["Ea_eV"] == pytest.approx(0.4, abs=0.01)


def test_exported_from_the_notebook_package():
    import opennmr.notebook as nb
    for name in ("fit_vt_relaxation", "plot_arrhenius", "fit_hendrickson_bray",
                 "exchange_spectrum", "plot_exchange_stack", "fit_exchange_spectra"):
        assert name in nb.__all__ and callable(getattr(nb, name))
