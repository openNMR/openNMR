"""Quadrupolar lineshape fitting (notebook/quadrupolar_fitting.py).

Fits run against synthetic spectra simulated with known parameters --
on a finer powder grid, with the exact model and with noise, so the
fit's own (coarser, second-order) model is not simply reproducing
itself -- and must recover them.
"""
import numpy as np
import pytest

from opennmr.notebook import quadrupolar_fitting as QF
from opennmr.notebook import simulations as S


def _close(got, want, tol):
    got, want, tol = (np.asarray(v, dtype=float) for v in (got, want, tol))
    assert np.all(np.abs(got - want) <= tol), f"{got} vs {want} (tol {tol})"
from opennmr.notebook.fitting import expand_pcov

NU0 = 130.3e6


def _results(truth, mas=0.0, lo=-80, hi=60, n=4096, noise=0.005, seed=0, I=2.5, nu0=NU0,
             nuclide="27Al", transitions="ct"):
    """truth: [(d_iso, C_Q MHz, eta_Q, fwhm, height)]"""
    rng = np.random.default_rng(seed)
    x = np.linspace(hi, lo, n)
    y = np.zeros(n)
    for d, cq, eta, fw, h in truth:
        y += h * S.quadrupolar_spectrum(I, cq * 1e6, eta, nu0, x[::-1], mas_Hz=mas,
                                        order="exact", transitions=transitions, delta_iso=d,
                                        fwhm=fw, n_zcw=46368)["normalised_real_spec"][::-1]
    y += noise * y.max() * rng.standard_normal(n) + 3.0
    return {"x_ppm": x, "real_spectrum": y, "SFO1": nu0 / 1e6, "nuclide": nuclide}


def test_static_ct_fit_recovers_the_parameters_from_an_estimated_start():
    r = _results([(12.0, 4.2, 0.35, 0.8, 1000.0)])
    start = QF.estimate_quadrupolar_start(r, 2.5)
    assert start["C_Q"] == pytest.approx(4.2, rel=0.15)
    popt, pcov, free, _s, info = QF.fit_quadrupolar_spectrum(r, start)
    _close(popt[:3], [12.0, 4.2, 0.35], [0.03, 0.01, 0.01])
    assert popt[3] == pytest.approx(0.8, abs=0.1)
    assert popt[4] == pytest.approx(1000.0, rel=0.03)          # raw units
    assert info["sim_kwargs"]["I"] == 2.5                      # read from results['nuclide']
    assert "static" in info["log"] and "second-order" in info["log"]


def test_mas_ct_fit_recovers_the_parameters():
    r = _results([(12.0, 4.2, 0.35, 0.5, 1000.0)], mas=15e3)
    start = QF.estimate_quadrupolar_start(r, 2.5, mas_Hz=15e3)
    popt, pcov, free, _s, info = QF.fit_quadrupolar_spectrum(r, start, mas_Hz=15e3, n_zcw=2584)
    _close(popt[:4], [12.0, 4.2, 0.35, 0.5], [0.03, 0.01, 0.01, 0.05])
    assert "MAS 15 kHz" in info["log"]


def test_two_static_sites_and_a_fixed_parameter():
    truth = [(10.0, 3.0, 0.2, 0.6, 1000.0), (-5.0, 6.0, 0.7, 0.8, 500.0)]
    r = _results(truth, noise=0.002, lo=-120)
    p0 = [dict(label="A", d_iso=9, C_Q=2.8, eta_Q=0.2, fwhm=1, fixed=["eta_Q"]),
          dict(label="B", d_iso=-3, C_Q=5.6, eta_Q=0.5, fwhm=1)]
    popt, pcov, free, _s, info = QF.fit_quadrupolar_spectrum(r, p0)
    a, b = popt[:6], popt[6:12]
    assert a[2] == 0.2 and not free[2]                         # held
    _close(a[:2], [10.0, 3.0], [0.05, 0.02])
    _close(b[:3], [-5.0, 6.0, 0.7], [0.1, 0.03, 0.03])
    true_cg = -5.0 + S.quadrupolar_ct_isotropic_shift_Hz(2.5, 6e6, 0.7, NU0) / NU0 * 1e6
    derived = QF.quadrupolar_derived(popt, expand_pcov(pcov, free, len(popt)), info)
    assert derived[1]["delta_cg"] == pytest.approx(true_cg, abs=0.05)
    assert info["labels"] == ["A", "B"]
    y, comps = QF.evaluate_quad_fit(r["x_ppm"], popt, info, return_components=True)
    assert np.allclose(y, np.sum(comps, axis=0))


def test_bounds_are_respected_and_c_q_sign_is_irrelevant():
    r = _results([(12.0, 4.2, 0.35, 0.8, 1000.0)])
    popt, *_rest, info = QF.fit_quadrupolar_spectrum(
        r, dict(d_iso=12, C_Q=-4.0, eta_Q=0.3, fwhm=1), bounds={"eta_Q": (0.0, 0.2)})
    assert popt[1] > 0                                         # |C_Q|
    assert popt[2] <= 0.2 + 1e-12
    pcov = np.zeros((len(popt), len(popt)))
    hits = QF.check_quad_bound_hits(popt, pcov, np.ones(len(popt), bool), info)
    assert any("eta_Q at its upper bound" in h for h in hits)


def test_integer_spin_uses_satellites_and_refuses_ct():
    nu0 = 61.4e6
    r = _results([(0.0, 0.165, 0.05, 30.0, 100.0)], I=1.0, nu0=nu0, nuclide="2H",
                 lo=-3000, hi=3000, transitions="all")
    popt, *_rest, info = QF.fit_quadrupolar_spectrum(r, dict(d_iso=0, C_Q=0.15, eta_Q=0.2, fwhm=40))
    assert info["sim_kwargs"]["transitions"] == "all"
    assert popt[1] == pytest.approx(0.165, rel=0.01)
    with pytest.raises(ValueError, match="no central"):
        QF.fit_quadrupolar_spectrum(r, dict(d_iso=0, C_Q=0.15, eta_Q=0.2, fwhm=40),
                                    transitions="ct")


def test_derived_quantities_and_their_errors():
    r = _results([(12.0, 4.2, 0.35, 0.8, 1000.0)])
    popt, pcov_f, free, _s, info = QF.fit_quadrupolar_spectrum(
        r, dict(d_iso=12, C_Q=4.1, eta_Q=0.3, fwhm=1))
    pcov = expand_pcov(pcov_f, free, len(popt))
    d = QF.quadrupolar_derived(popt, pcov, info)[0]
    assert d["P_Q"] == pytest.approx(popt[1] * np.sqrt(1 + popt[2] ** 2 / 3))
    iso = S.quadrupolar_ct_isotropic_shift_Hz(2.5, popt[1] * 1e6, popt[2], NU0) / NU0 * 1e6
    assert d["delta_cg"] == pytest.approx(popt[0] + iso)
    # Monte-Carlo check of the propagated P_Q error
    rng = np.random.default_rng(0)
    draws = rng.multivariate_normal(popt[:3], pcov[:3, :3], 20000)
    assert d["P_Q_err"] == pytest.approx(np.std(draws[:, 1] * np.sqrt(1 + draws[:, 2] ** 2 / 3)),
                                         rel=0.05)
    k = iso / (popt[1] ** 2 * (1 + popt[2] ** 2 / 3))
    cg = draws[:, 0] + k * draws[:, 1] ** 2 * (1 + draws[:, 2] ** 2 / 3)
    assert d["delta_cg_err"] == pytest.approx(np.std(cg), rel=0.05)


def test_report_plot_and_workflow(monkeypatch, tmp_path):
    r = _results([(12.0, 4.2, 0.35, 0.8, 1000.0)])
    monkeypatch.setattr(QF, "NMR1dimport", lambda path, procno=1: r)
    out = QF.NMRQuadFitWorkflow("unused", p0=None, show_plot=True, save_csv=True,
                                csv_filename=str(tmp_path / "q.csv"), save_report=True,
                                report_filename=str(tmp_path / "q.txt"), font="DejaVu Sans")
    _close(out["popt"][:3], [12.0, 4.2, 0.35], [0.05, 0.02, 0.02])
    report = (tmp_path / "q.txt").read_text(encoding="utf-8")
    for key in ("QUADRUPOLAR FIT REPORT", "C_Q (MHz)", "P_Q (MHz)", "CT c.o.g.", "R^2"):
        assert key in report
    assert (tmp_path / "q.csv").read_text().startswith("x_ppm,y,y_fit")
    assert out["figure"] is not None and len(out["figure"].axes) == 2   # fit + residual


def test_perturbation_warning_at_low_field():
    popt = np.array([0, 20.0, 0.0, 1, 1, 0, 0, 0])
    w = QF._physics_warnings(popt, np.zeros((5, 5)), np.array([1, 1, 1, 1, 1, 0, 0, 0], bool),
                             ["S"], 1.5, 20e6, 0.0, "ct", "second")
    assert any("exact model" in x for x in w)


def test_a_baseline_estimated_on_the_pattern_is_flagged_and_fitting_it_rescues_the_fit():
    """This pattern's wing runs to -77 ppm; on a -80 ppm axis the
    baseline, estimated from the outer 5% of the spectrum, sits on the
    wing, and every parameter comes out biased (d_iso -4.86, C_Q 5.97,
    amplitude -18%, measured). The fit must say so -- and fitting the
    baseline fixes it."""
    r = _results([(-5.0, 6.0, 0.7, 0.8, 500.0)], noise=0.0, lo=-80)
    p0 = dict(d_iso=-5, C_Q=6, eta_Q=0.7, fwhm=0.8)
    popt, pcov, free, _s, info = QF.fit_quadrupolar_spectrum(r, p0)
    assert any("baseline was estimated where" in w for w in info["warnings"])
    popt, pcov, free, _s, info = QF.fit_quadrupolar_spectrum(r, p0, fit_baseline=True)
    _close(popt[:3], [-5.0, 6.0, 0.7], [0.02, 0.005, 0.005])
    assert free[-2:].all()


def test_well_framed_fit_has_no_edge_warning():
    r = _results([(12.0, 4.2, 0.35, 0.8, 1000.0)])
    *_rest, info = QF.fit_quadrupolar_spectrum(r, dict(d_iso=12, C_Q=4.2, eta_Q=0.35, fwhm=0.8))
    assert not any("clipped" in w or "baseline" in w for w in info["warnings"])


def test_a_pattern_running_past_the_window_is_flagged():
    r = _results([(12.0, 4.2, 0.35, 0.8, 1000.0)])
    *_rest, info = QF.fit_quadrupolar_spectrum(r, dict(d_iso=12, C_Q=4.2, eta_Q=0.35, fwhm=0.8),
                                               fit_regions=[(-10, 30)])
    assert any("clipped" in w for w in info["warnings"])
