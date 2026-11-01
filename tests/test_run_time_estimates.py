"""Run-time estimates before long Solid-state NMR simulations and fits
(opennmr.notebook.run_time, gui/long_run.py; explicit user request,
2026-11-01)."""

import time

import numpy as np
import pytest
from PySide6.QtWidgets import QMessageBox

from opennmr.notebook import run_time as rt
from opennmr.notebook.combined_hamiltonian import Site, simulate_combined

from test_gui import qapp  # noqa: F401 -- fixture

NU0 = 104.26e6


def _quad(fwhm=1.0):
    return Site("27Al", 12, fwhm=fwhm).add_quadrupolar(5.0, 0.3)


def test_format_duration():
    assert rt.format_duration(0.42) == "0.4 s"
    assert rt.format_duration(12.4) == "12 s"
    assert rt.format_duration(200) == "3 min 20 s"
    assert rt.format_duration(3900) == "1 h 05 min"


def test_simulation_estimate_is_close_to_the_real_time():
    """Exact-order MAS, the case that scales worst (measured 2026-11-01:
    within 10-30% of the real time)."""
    ax = np.linspace(-400, 400, 4096)
    kw = dict(mas_Hz=2e4, order="exact", n_zcw=3000)
    est = rt.estimate_simulation_time([_quad()], ax, larmor_Hz=NU0, **kw)
    t = time.perf_counter()
    simulate_combined([_quad()], ax, NU0, **kw)
    real = time.perf_counter() - t
    assert not est["bound"]
    assert 0.4 < est["seconds"] / real < 2.5, (est["seconds"], real)
    assert est["probe_s"] < 0.6 * real + 0.5
    assert est["total_orientations"] == next(iter(
        simulate_combined([_quad()], ax, NU0, **kw)["powder"].values()))["orientations"]
    assert "orientations" in est["text"]


def test_orientation_counts_match_the_simulation_for_every_scheme():
    ax = np.linspace(-300, 300, 2048)
    sites = [_quad(), Site("27Al", 40, label="C").add_csa(-100, 0.3)]
    for scheme in (None, "zcw", "lebedev", "rectangular"):
        for mas in (0.0, 1e4):
            est = rt.estimate_simulation_time(sites, ax, larmor_Hz=NU0, mas_Hz=mas,
                                              n_zcw=500, powder_scheme=scheme, stop_below=1e9)
            run = simulate_combined(sites, ax, NU0, mas_Hz=mas, n_zcw=500, powder_scheme=scheme)
            assert est["orientations"] == {k: v["orientations"] for k, v in run["powder"].items()}


def test_a_quick_simulation_is_checked_with_one_small_trial():
    ax = np.linspace(-400, 400, 2048)
    est = rt.estimate_simulation_time([_quad()], ax, larmor_Hz=NU0, n_zcw=2000, stop_below=60)
    assert est["bound"] and est["text"].startswith("at most")
    assert est["probe_s"] < 1.0


def test_free_shape_parameters_follow_fit_combined_rules():
    a = Site("27Al", 0, label="A").add_quadrupolar(3, 0.2).add_csa(50, 0.3)
    b = Site("27Al", 10, label="B")
    assert rt.free_shape_parameters([a, b]) == {"A": 5, "B": 1}       # Euler angles held
    assert rt.free_shape_parameters([a, b], vary={"A": ["C_Q", "amount"], "B": []}) == \
        {"A": 1, "B": 0}
    assert rt.free_shape_parameters([a, b], fixed={"A": ["delta_iso"]}) == {"A": 4, "B": 1}


def _results(sim):
    noise = np.random.default_rng(3).normal(0, 3e-3 * sim["real_spec"].max(), sim["x_ppm"].size)
    return {"x_ppm": sim["x_ppm"], "real_spectrum": sim["real_spec"] + noise,
            "SFO1": NU0 / 1e6, "nuclide": "27Al"}


def test_fit_estimate_brackets_a_real_fit():
    from opennmr.notebook import combined_fitting as cfit
    ax = np.linspace(-200, 150, 2048)
    truth = simulate_combined(Site("27Al", 12, fwhm=1.5).add_quadrupolar(4.0, 0.4), ax, NU0,
                              mas_Hz=15e3, transitions="ct", n_zcw=4181)
    res = _results(truth)
    start = [Site("27Al", 14, fwhm=2.0).add_quadrupolar(3.6, 0.3)]
    est = rt.estimate_fit_time(res, start, mas_Hz=15e3, n_zcw=6000, transitions="ct")
    t = time.perf_counter()
    cfit.fit_combined(res, start, mas_Hz=15e3, n_zcw=6000, transitions="ct")
    real = time.perf_counter() - t
    assert est["low"] < est["high"]
    assert est["low"] / 2 < real < est["high"] * 2, (est["low"], real, est["high"])
    assert est["n_free"] == 5             # δiso, amount, FWHM, C_Q, η_Q
    assert "free parameters" in est["text"]


def test_more_starts_and_a_mas_scan_cost_more():
    ax = np.linspace(-200, 150, 1024)
    res = _results(simulate_combined(_quad(), ax, NU0, mas_Hz=15e3, n_zcw=500))
    one = rt.estimate_fit_time(res, [_quad()], mas_Hz=15e3, n_zcw=500)
    many = rt.estimate_fit_time(res, [_quad()], mas_Hz=15e3, n_zcw=500, starts=6)
    scan = rt.estimate_fit_time(res, [_quad()], mas_Hz=15e3, n_zcw=500, fit_mas=True)
    assert many["high"] > 4 * one["high"]
    assert scan["high"] > one["high"] and "scan of the MAS rate" in scan["text"]


# ---- the GUI: ask before a long run ---------------------------------------------------

@pytest.fixture
def asking(monkeypatch):
    """Estimates on, with a threshold every run exceeds; records questions."""
    from opennmr.gui import long_run
    monkeypatch.setenv(long_run.ENV, "1")
    monkeypatch.setattr(rt, "SIMULATION_WARN_S", 1e-6)
    monkeypatch.setattr(rt, "FIT_WARN_S", 1e-6)
    asked = {"answer": True, "calls": []}

    def ask(parent, title, what, text, tips):
        asked["calls"].append((what, text))
        return asked["answer"]
    monkeypatch.setattr(long_run, "ask", ask)
    return asked


def _sim_panel(qapp):  # noqa: F811
    from opennmr.gui.main_window import MainWindow
    window = MainWindow()
    p = window.combined_sim_panel
    p.nucleus_combo.setCurrentText("27Al")
    p.larmor_spin.setValue(104.26)
    p.regime_combo.setCurrentIndex(p.regime_combo.findData("mas"))
    p.mas_spin.setValue(15000)
    p.transitions_combo.setCurrentIndex(p.transitions_combo.findData("ct"))
    p.orientations_spin.setValue(800)
    p.add_interaction("quadrupolar")
    return window, p


def test_simulate_asks_and_cancel_runs_nothing(qapp, asking, monkeypatch):  # noqa: F811
    window, p = _sim_panel(qapp)
    asking["answer"] = False
    assert p.simulate() is False
    assert asking["calls"] and asking["calls"][0][0] == "simulation"
    assert p.result is None or "result" not in (p.result or {})
    asking["answer"] = True
    assert p.simulate() is True
    assert "Took" in p.result["result"]["log"] and "estimated" in p.result["result"]["log"]
    window.close()


def test_no_question_when_estimates_are_off(qapp, monkeypatch):  # noqa: F811
    from opennmr.gui import long_run
    monkeypatch.setenv(long_run.ENV, "0")
    monkeypatch.setattr(long_run, "ask", lambda *a: pytest.fail("asked"))
    window, p = _sim_panel(qapp)
    assert p.simulate() is True
    assert "Took" not in p.result["result"]["log"]
    window.close()


def test_quick_runs_are_not_asked_about(qapp, monkeypatch):  # noqa: F811
    from opennmr.gui import long_run
    monkeypatch.setenv(long_run.ENV, "1")
    monkeypatch.setattr(long_run, "ask", lambda *a: pytest.fail("asked about a quick run"))
    window, p = _sim_panel(qapp)
    assert p.simulate() is True
    window.close()


def test_combined_fit_asks_first(qapp, asking, monkeypatch):  # noqa: F811
    from opennmr.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
    from opennmr.gui.main_window import MainWindow
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: None)
    x = np.linspace(150, -200, 2048)
    y = simulate_combined(Site("27Al", 12, fwhm=1.5).add_quadrupolar(4.0, 0.4), x[::-1], NU0,
                          mas_Hz=15e3, transitions="ct", n_zcw=800)["real_spec"][::-1]
    spectrum = NMRSpectrum(data=y / y.max() * 1000, axes=[NMRAxis(kind="frequency", ppm=x)],
                           metadata=NMRMetadata(vendor="simulated", nucleus="27Al",
                                                sfo1=NU0 / 1e6, raw={"acqus": {"MASR": 15e3}}))
    window = MainWindow()
    window._add_derived_spectrum(spectrum, "target")
    p = window.combined_fit_panel
    p.transitions_combo.setCurrentIndex(p.transitions_combo.findData("ct"))
    p.orientations_spin.setValue(600)
    p.set_state({"widgets": {}, "current": 0, "sites": [
        {"label": "Al1", "delta_iso": 10.0, "amount": 1.0, "fwhm": 1.0, "lineshape": "gaussian",
         "eta": 0.5, "interactions": [{"type": "quadrupolar", "C_Q": 3.8, "eta_Q": 0.5}]}]})
    asking["answer"] = False
    assert p.fit() is False
    assert asking["calls"][-1][0] == "fit"
    assert p.result is None
    asking["answer"] = True
    assert p.fit() is True
    assert "Took" in p.result["log"]
    window.close()


def test_long_run_text_helpers():
    from opennmr.gui import long_run
    assert long_run.n_starts(None) == 1
    assert long_run.n_starts({"A": {"C_Q": [1, 2, 3]}, "B": {"aniso": [1, 2]}}) == 6
    assert long_run.report_line(None, 3) == ""
    assert long_run.report_line({"seconds": 10, "bound": False}, 12) == "Took 12 s (estimated 10 s)."
    assert long_run.report_line({"low": 5, "high": 70, "bound": False}, 30) == \
        "Took 30 s (estimated 5 s to 1 min 10 s)."
