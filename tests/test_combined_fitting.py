"""Combined-Hamiltonian fitting (notebook/combined_fitting.py).

Synthetic spectra are simulated with the exact quadrupolar model, a
finer powder set and noise, so the (second-order, coarser) fit model is
not just reproducing itself; the fit must recover the known values.
"""
import numpy as np
import pytest

import opennmr.notebook as nmr
from opennmr.notebook import combined_fitting as CF
from opennmr.notebook.combined_hamiltonian import Site, simulate_combined

NU0 = 130.3e6


def _truth():
    return [Site("27Al", 12, label="Al1", fwhm=0.5, amount=2.0).add_quadrupolar(4.2, 0.35)
            .add_csa(60, 0.2).add_jcoupling("31P", 1, 400),
            Site("27Al", -8, label="Al2", fwhm=0.8, amount=1.0).add_quadrupolar(6.0, 0.7)]


def _data(mas, lo, hi, n, noise=4.0, seed=0, sites=None):
    rng = np.random.default_rng(seed)
    x = np.linspace(hi, lo, n)
    y = simulate_combined(sites or _truth(), x[::-1], NU0, mas_Hz=mas, transitions="ct",
                          n_zcw=17711, order="exact")["real_spec"][::-1]
    return {"x_ppm": x, "real_spectrum": y / y.max() * 1000 + noise * rng.standard_normal(n),
            "SFO1": NU0 / 1e6, "nuclide": "27Al"}


def _values(fit):
    return {f"{p['label']}.{p['name']}": p["value"] for p in fit["parameters"]}


def test_parameter_names_follow_the_interactions():
    a, b = _truth()
    assert CF.site_parameter_names(a) == ["delta_iso", "amount", "fwhm", "aniso", "asym", "alpha",
                                          "beta", "gamma", "C_Q", "eta_Q", "J1"]
    assert CF.site_parameter_names(Site("13C").add_csa(100).add_jcoupling("1H", 1, 150)
                                   .add_jcoupling("31P", 1, 20)) == \
        ["delta_iso", "amount", "fwhm", "aniso", "asym", "J1", "J2"]


def test_mas_fit_recovers_quadrupolar_j_shifts_and_the_site_ratio():
    res = _data(20e3, -450, 450, 16384)
    start = [Site("27Al", 10, label="Al1", fwhm=1).add_quadrupolar(4.0, 0.5).add_csa(60, 0.2)
             .add_jcoupling("31P", 1, 350),
             Site("27Al", -5, label="Al2", fwhm=1).add_quadrupolar(5.5, 0.5)]
    fit = CF.fit_combined(res, start, mas_Hz=20e3, transitions="ct", n_zcw=1597,
                          fixed={"Al1": ["aniso", "asym"]})
    v = _values(fit)
    assert v["Al1.C_Q"] == pytest.approx(4.2, abs=0.01)
    assert v["Al1.eta_Q"] == pytest.approx(0.35, abs=0.01)
    assert v["Al1.delta_iso"] == pytest.approx(12.0, abs=0.02)
    assert v["Al1.J1"] == pytest.approx(400, abs=2)
    assert v["Al2.C_Q"] == pytest.approx(6.0, abs=0.02)
    assert v["Al2.eta_Q"] == pytest.approx(0.7, abs=0.01)
    assert fit["derived"]["Al1"]["share_%"] == pytest.approx(200 / 3, abs=0.5)   # 2 : 1
    assert fit["quality"]["r2"] > 0.99
    assert fit["sites"][0].get("quadrupolar")[0]["C_Q"] == pytest.approx(v["Al1.C_Q"])
    assert "MAS 20 kHz" in fit["log"]


def test_static_csa_with_mas_values_fixed_and_the_multistart():
    res = _data(0.0, -150, 150, 8192)
    fixed = {"Al1": ["delta_iso", "C_Q", "eta_Q", "J1"], "Al2": ["delta_iso", "C_Q", "eta_Q"]}
    start = [Site("27Al", 12, label="Al1", fwhm=1).add_quadrupolar(4.2, 0.35).add_csa(-40, 0.5)
             .add_jcoupling("31P", 1, 400),
             Site("27Al", -8, label="Al2", fwhm=1).add_quadrupolar(6.0, 0.7)]
    # Which starts fall into a wrong minimum depends on the details (the
    # powder set, the noise); a scan over both signs of Delta must find
    # the right one regardless.
    best = CF.fit_combined_multistart(res, start, starts={"Al1": {"aniso": [-80, -40, 40, 80]}},
                                      fixed=fixed, transitions="ct", n_zcw=2584, verbose=False)
    v = _values(best)
    assert v["Al1.aniso"] == pytest.approx(60.0, abs=0.3)
    assert v["Al1.asym"] == pytest.approx(0.2, abs=0.01)
    assert best["quality"]["r2"] > 0.99
    assert [r["chi2"] for r in best["starts"]] == sorted(r["chi2"] for r in best["starts"])
    assert "Multi-start: 4 starts" in best["log"] and len(best["starts"]) == 4


def test_vary_fixed_bounds_and_errors():
    res = _data(0.0, -150, 150, 4096)
    s = [Site("27Al", 12, label="Al1", fwhm=0.5).add_quadrupolar(4.2, 0.35).add_csa(60, 0.2)
         .add_jcoupling("31P", 1, 400),
         Site("27Al", -8, label="Al2", fwhm=0.8).add_quadrupolar(6.0, 0.7)]
    fit = CF.fit_combined(res, s, vary=["amount", "fwhm"], transitions="ct", n_zcw=2584)
    free = {f"{p['label']}.{p['name']}" for p in fit["parameters"] if p["free"]}
    assert free == {"Al1.amount", "Al1.fwhm", "Al2.amount", "Al2.fwhm"}
    fit = CF.fit_combined(res, s, vary={"Al1": ["amount", "beta"], "Al2": ["amount"]},
                          bounds={"Al1": {"beta": (0.0, 5.0)}}, transitions="ct", n_zcw=2584)
    v = _values(fit)
    assert 0.0 <= v["Al1.beta"] <= 5.0                     # Euler angles free only on request
    with pytest.raises(ValueError, match="no parameter"):
        CF.fit_combined(res, s, vary={"Al2": ["aniso"]}, transitions="ct")
    with pytest.raises(KeyError, match="not site labels"):
        CF.fit_combined(res, s, fixed={"Al9": ["fwhm"]}, transitions="ct")


def test_amounts_are_started_from_the_data():
    res = _data(20e3, -450, 450, 8192)
    s = [Site("27Al", 12, label="Al1", fwhm=0.5, amount=1e-9).add_quadrupolar(4.2, 0.35)
         .add_csa(60, 0.2).add_jcoupling("31P", 1, 400),
         Site("27Al", -8, label="Al2", fwhm=0.8, amount=1e9).add_quadrupolar(6.0, 0.7)]
    fit = CF.fit_combined(res, s, vary=["amount"], mas_Hz=20e3, transitions="ct", n_zcw=1597)
    assert fit["derived"]["Al1"]["share_%"] == pytest.approx(200 / 3, abs=0.5)


def test_correlation_and_baseline_warnings():
    # A broad pattern filling the spectrum: the baseline zone sits on it.
    res = _data(0.0, -48, 28, 4096, sites=[Site("27Al", 0, label="A", fwhm=1)
                                            .add_quadrupolar(6.0, 0.2)])
    fit = CF.fit_combined(res, [Site("27Al", 0, label="A", fwhm=1).add_quadrupolar(6.0, 0.2)],
                          transitions="ct", n_zcw=2584)
    assert any("baseline was estimated" in w for w in fit["warnings"])
    fit = CF.fit_combined(res, [Site("27Al", 0, label="A", fwhm=1).add_quadrupolar(6.0, 0.2)],
                          transitions="ct", n_zcw=2584, fit_baseline=True)
    assert not any("baseline was estimated" in w for w in fit["warnings"])
    assert any(p["name"] == "b0" and p["free"] for p in fit["parameters"])


def test_report_plot_and_workflow(monkeypatch, tmp_path):
    res = _data(20e3, -450, 450, 8192)
    import opennmr.notebook.one_d as one_d
    monkeypatch.setattr(one_d, "NMR1dimport", lambda path, procno=1: res)
    fit = nmr.NMRCombinedFitWorkflow("unused", _truth(), mas_Hz=20e3, transitions="ct",
                                     n_zcw=1597, fixed={"Al1": ["aniso", "asym"]},
                                     save_report=str(tmp_path / "r.txt"),
                                     save_csv=str(tmp_path / "f.csv"),
                                     fname=str(tmp_path / "f.png"))
    text = (tmp_path / "r.txt").read_text(encoding="utf-8")
    for key in ("COMBINED-HAMILTONIAN FIT REPORT", "C_Q (MHz)", "J1 (Hz)", "share (%)",
                "P_Q (MHz)", "CT c.o.g. (ppm)", "R^2"):
        assert key in text
    assert (tmp_path / "f.csv").read_text().startswith("x_ppm,y,y_fit,Al1,Al2,baseline")
    assert (tmp_path / "f.png").exists() and len(fit["figure"].axes) == 2
