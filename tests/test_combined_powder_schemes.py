"""Powder-averaging schemes in the Solid-state NMR (combined) engine and its
panels: ZCW, Lebedev, rectangular, and the automatic per-site choice
(explicit user request, 2026-11-01: ZCW static, Lebedev MAS -- refined by
measurement to Lebedev only for MAS sites without a quadrupolar
interaction, see combined_hamiltonian.default_powder_scheme)."""

import numpy as np
import pytest

from opennmr.notebook import combined_fitting as cfit
from opennmr.notebook import combined_hamiltonian as ch
from opennmr.notebook.combined_hamiltonian import Site, simulate_combined

NU0_AL = 104.26e6


def _rel(a, b):
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


def _csa_site(fwhm=0.5):
    return Site("27Al", 50, fwhm=fwhm).add_csa(-150, 0.4)


def _quad_site(fwhm=0.5):
    return Site("27Al", 12, fwhm=fwhm).add_quadrupolar(4.2, 0.35)


# ---- the automatic choice --------------------------------------------------------------

def test_default_scheme_static_is_zcw_for_every_site():
    assert ch.default_powder_scheme(0.0) == "zcw"
    assert ch.default_powder_scheme(0.0, _quad_site()) == "zcw"
    assert ch.default_powder_scheme(0.0, _csa_site()) == "zcw"


def test_default_scheme_mas_is_lebedev_without_quadrupolar_and_zcw_with():
    assert ch.default_powder_scheme(10e3, _csa_site()) == "lebedev"
    assert ch.default_powder_scheme(10e3, Site("13C", 10)) == "lebedev"
    assert ch.default_powder_scheme(10e3, _quad_site()) == "zcw"


def test_auto_records_each_sites_scheme_in_powder_info_and_log():
    ax = np.linspace(-300, 400, 4096)
    r = simulate_combined([_csa_site(),
                           Site("27Al", 0, label="Q", fwhm=0.5).add_quadrupolar(3.0, 0.2)],
                          ax, NU0_AL, mas_Hz=12e3, transitions="ct")
    schemes = {label: info["scheme"] for label, info in r["powder"].items()}
    assert schemes["Q"] == "zcw"
    assert set(schemes.values()) == {"zcw", "lebedev"}
    assert "Lebedev" in r["log"] and "ZCW" in r["log"]
    assert r["settings"]["powder_scheme"] == "auto"


def test_unknown_scheme_is_refused():
    with pytest.raises(ValueError, match="powder_scheme"):
        simulate_combined(_csa_site(), np.linspace(-200, 200, 512), NU0_AL, powder_scheme="hex")


# ---- each scheme converges to the same answer -----------------------------------------

@pytest.mark.parametrize("scheme", ["zcw", "lebedev", "rectangular"])
def test_every_scheme_converges_on_mas_csa_sidebands(scheme):
    ax = np.linspace(-400, 500, 8192)
    ref = simulate_combined(_csa_site(), ax, NU0_AL, mas_Hz=5e3, n_zcw=46368,
                            powder_scheme="zcw")["real_spec"]
    got = simulate_combined(_csa_site(), ax, NU0_AL, mas_Hz=5e3, n_zcw=4181,
                            powder_scheme=scheme)["real_spec"]
    assert _rel(got, ref) < 5e-3


def test_lebedev_beats_zcw_on_mas_csa_sidebands_with_fewer_orientations():
    ax = np.linspace(-400, 500, 8192)
    ref = simulate_combined(_csa_site(), ax, NU0_AL, mas_Hz=5e3, n_zcw=46368,
                            powder_scheme="zcw")["real_spec"]
    leb = simulate_combined(_csa_site(), ax, NU0_AL, mas_Hz=5e3, n_zcw=4181,
                            powder_scheme="lebedev")
    zcw = simulate_combined(_csa_site(), ax, NU0_AL, mas_Hz=5e3, n_zcw=4181,
                            powder_scheme="zcw")
    assert _rel(leb["real_spec"], ref) < 0.2 * _rel(zcw["real_spec"], ref)
    n_leb = next(iter(leb["powder"].values()))["orientations"]
    n_zcw = next(iter(zcw["powder"].values()))["orientations"]
    assert n_leb < n_zcw / 4


def test_auto_keeps_zcw_accuracy_for_a_narrow_quadrupolar_mas_lineshape():
    """The reason the automatic choice is per site: Lebedev's largest rule is
    too coarse for a second-order MAS central transition with narrow lines."""
    ax = np.linspace(-80, 60, 4096)
    ref = simulate_combined(_quad_site(), ax, NU0_AL, mas_Hz=15e3, transitions="ct",
                            n_zcw=46368, powder_scheme="zcw")["real_spec"]
    auto = simulate_combined(_quad_site(), ax, NU0_AL, mas_Hz=15e3, transitions="ct",
                             n_zcw=4181)["real_spec"]
    leb = simulate_combined(_quad_site(), ax, NU0_AL, mas_Hz=15e3, transitions="ct",
                            n_zcw=4181, powder_scheme="lebedev")["real_spec"]
    assert _rel(auto, ref) < 1e-2
    assert _rel(leb, ref) > 5 * _rel(auto, ref)


def test_static_csa_zcw_converges_and_lebedev_is_capped():
    """Why the automatic choice is ZCW for a static pattern: a static
    lineshape is a histogram, and the largest Lebedev rule (5810 points,
    760 after folding into the octant) is far too few for it."""
    ax = np.linspace(-200, 300, 4096)
    site = _csa_site(fwhm=3.0)
    ref = simulate_combined(site, ax, NU0_AL, n_zcw=75025, powder_scheme="zcw")["real_spec"]
    err = {s: _rel(simulate_combined(site, ax, NU0_AL, n_zcw=20000, powder_scheme=s)["real_spec"],
                   ref) for s in ("zcw", "lebedev", "rectangular")}
    assert err["zcw"] < 3e-3
    assert err["rectangular"] < 0.1
    assert err["lebedev"] > 10 * err["zcw"]
    auto = simulate_combined(site, ax, NU0_AL, n_zcw=20000)
    assert next(iter(auto["powder"].values()))["scheme"] == "zcw"


# ---- the octant fold ---------------------------------------------------------------------

def test_fold_to_octant_conserves_weight_and_quadratic_averages():
    from opennmr.notebook.simulations import lebedev_angles
    theta, phi, w = lebedev_angles(590)
    t, p, wf = ch._fold_to_octant(theta, phi, w)
    assert np.isclose(wf.sum(), 1.0)
    assert np.all(t <= np.pi / 2 + 1e-12) and np.all((p >= -1e-12) & (p <= np.pi / 2 + 1e-12))
    assert len(t) < len(theta) / 4

    def avg(tt, pp, ww):            # <x^2 y^2>, <z^4>: even in every component
        x, y, z = np.sin(tt) * np.cos(pp), np.sin(tt) * np.sin(pp), np.cos(tt)
        return np.sum(ww * x ** 2 * y ** 2), np.sum(ww * z ** 4)

    np.testing.assert_allclose(avg(t, p, wf), avg(theta, phi, w), rtol=1e-10)
    np.testing.assert_allclose(avg(theta, phi, w), (1 / 15, 1 / 5), rtol=1e-10)


def test_lebedev_cap_is_reported_in_the_log():
    r = simulate_combined(_csa_site(), np.linspace(-400, 500, 2048), NU0_AL, mas_Hz=5e3,
                          n_zcw=100000, powder_scheme="lebedev")
    assert "largest rule" in r["log"] and "5810" in r["log"]
    r = simulate_combined(_csa_site(), np.linspace(-400, 500, 2048), NU0_AL, mas_Hz=5e3,
                          n_zcw=300, powder_scheme="lebedev")
    assert "largest rule" not in r["log"]


# ---- fits and code export carry the scheme ----------------------------------------------

def _results_from(sim, nu0):
    return {"x_ppm": sim["x_ppm"], "real_spectrum": sim["real_spec"], "SFO1": nu0 / 1e6,
            "nuclide": "27Al"}


@pytest.mark.parametrize("scheme", [None, "zcw", "lebedev", "rectangular"])
def test_fit_combined_accepts_every_scheme_and_records_it(scheme):
    ax = np.linspace(-400, 500, 2048)
    truth = simulate_combined(_csa_site(fwhm=2.0), ax, NU0_AL, mas_Hz=5e3, n_zcw=2000)
    fit = cfit.fit_combined(_results_from(truth, NU0_AL), [_csa_site(fwhm=2.0)],
                            mas_Hz=5e3, n_zcw=2000, powder_scheme=scheme)
    assert fit["settings"]["powder_scheme"] == (scheme or "auto")
    assert fit["quality"]["r2"] > 0.99
    assert ("automatic scheme" if scheme is None else
            {"zcw": "ZCW", "lebedev": "Lebedev", "rectangular": "θ/φ grid"}[scheme]) in fit["log"]


def test_starting_model_and_code_export_take_the_scheme():
    ax = np.linspace(-400, 500, 2048)
    truth = simulate_combined(_csa_site(fwhm=2.0), ax, NU0_AL, mas_Hz=5e3, n_zcw=2000)
    guess = cfit.starting_model(_results_from(truth, NU0_AL), [_csa_site(fwhm=2.0)],
                                mas_Hz=5e3, n_zcw=2000, powder_scheme="rectangular")
    assert guess["powder_scheme"] == "rectangular"
    code = ch.combined_to_python([_csa_site()], ax, larmor_Hz=NU0_AL, mas_Hz=5e3,
                                 powder_scheme="lebedev")
    assert "powder_scheme='lebedev'" in code
    namespace = {}
    exec(code, namespace)                                       # the export runs as written
    assert np.isfinite(namespace["spec"]["real_spec"]).all()


def test_multi_fit_experiment_carries_the_scheme():
    from opennmr.notebook import multi_fitting as mf
    ax = np.linspace(-400, 500, 1024)
    truth = simulate_combined(_csa_site(fwhm=2.0), ax, NU0_AL, mas_Hz=5e3, n_zcw=1000)
    e = mf.make_experiment(_results_from(truth, NU0_AL), "A", mas_Hz=5e3, powder_scheme="zcw")
    assert e["powder_scheme"] == "zcw"
    assert mf.make_experiment(_results_from(truth, NU0_AL), "B")["powder_scheme"] == "auto"


# ---- GUI ---------------------------------------------------------------------------------

from test_gui import qapp  # noqa: E402,F401 -- fixture


@pytest.fixture
def window(qapp):  # noqa: F811
    from opennmr.gui.main_window import MainWindow
    w = MainWindow()
    yield w
    w.close()
    w.deleteLater()


def test_simulate_and_fit_panels_offer_the_schemes(window):
    for panel in (window.combined_sim_panel, window.combined_fit_panel):
        combo = panel.powder_combo
        assert [combo.itemData(i) for i in range(combo.count())] == [
            "auto", "zcw", "lebedev", "rectangular"]
        assert combo.currentData() == "auto"
        assert combo.isVisibleTo(panel)
        assert "Lebedev" in combo.toolTip()


def test_panel_settings_pass_the_scheme_and_state_restores_it(window):
    panel = window.combined_sim_panel
    panel.larmor_spin.setValue(104.26)
    panel.powder_combo.setCurrentIndex(panel.powder_combo.findData("rectangular"))
    assert panel.settings()["powder_scheme"] == "rectangular"
    assert panel.transfer_settings()["powder_scheme"] == "rectangular"
    state = panel.get_state()
    panel.powder_combo.setCurrentIndex(0)
    panel.set_state(state)
    assert panel.powder_combo.currentData() == "rectangular"


def test_simulation_settings_go_to_the_fit_panel(window):
    sim, fit = window.combined_sim_panel, window.combined_fit_panel
    sim.powder_combo.setCurrentIndex(sim.powder_combo.findData("lebedev"))
    fit.set_start_from_simulation([{"label": "A", "delta_iso": 0.0, "amount": 1.0,
                                    "fwhm": 1.0, "lineshape": "gaussian", "eta": 0.5,
                                    "interactions": []}], sim.transfer_settings(), "")
    assert fit.powder_combo.currentData() == "lebedev"


def test_multi_fit_panel_offers_the_schemes(window):
    panel = window.multi_fit_panel
    assert panel.powder_combo.currentData() == "auto"
    assert panel.fit_arguments.__func__ is not None
