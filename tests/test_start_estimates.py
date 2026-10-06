"""Starting values for a fit, estimated from the spectrum for one kind of
interaction (user request, 2026-10-06) -- notebook.start_estimates and
Fit ▸ Combined Hamiltonian's "Estimate Starting Values"."""

import numpy as np
import pytest

from opennmr.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
from opennmr.gui.main_window import MainWindow
from opennmr.notebook import start_estimates as se
from opennmr.notebook.combined_hamiltonian import Site, simulate_combined

from test_gui import qapp  # noqa: F401 -- fixture


def _data(truth, x, nu0, mas=0, transitions="all", noise=1.0, seed=0, n=2584):
    y = simulate_combined(truth, x, nu0, mas_Hz=mas, n_zcw=n, transitions=transitions)["real_spec"]
    y = y / y.max() * 1000 + np.random.default_rng(seed).normal(0, noise, x.size)
    # Descending ppm, as spectra are stored.
    return {"x_ppm": x[::-1], "real_spectrum": y[::-1], "SFO1": nu0 / 1e6}


def _interaction(est, kind):
    (site,) = est["sites"]
    return site, next(i for i in site["interactions"] if i["type"] == kind)


X13 = np.linspace(-150, 350, 8192)


@pytest.mark.parametrize("aniso", [-150.0, 120.0])
def test_static_csa_size_sign_and_asymmetry(aniso):
    data = _data([Site("13C", 120, fwhm=1.0).add_csa(aniso, 0.4)], X13, 100.6e6, n=6765)
    est = se.estimate_start(data, "csa", nucleus="13C")
    site, csa = _interaction(est, "csa")
    assert csa["aniso"] == pytest.approx(aniso, abs=3)
    assert csa["asym"] == pytest.approx(0.4, abs=0.1)
    assert site["delta_iso"] == pytest.approx(120, abs=1)
    assert est["start_r2"] > 0.98 and est["notes"]


def test_mas_csa_from_the_sideband_manifold():
    data = _data([Site("13C", 120, fwhm=0.6).add_csa(-150, 0.4)], X13, 100.6e6, mas=8000)
    est = se.estimate_start(data, "csa", nucleus="13C", mas_Hz=8000)
    site, csa = _interaction(est, "csa")
    assert csa["aniso"] == pytest.approx(-150, abs=3)
    assert csa["asym"] == pytest.approx(0.4, abs=0.1)
    assert site["delta_iso"] == pytest.approx(120, abs=0.2)     # the centreband


def test_noisy_csa_still_lands_near_the_truth():
    data = _data([Site("13C", 120, fwhm=1.0).add_csa(-150, 0.4)], X13, 100.6e6, noise=20.0,
                 n=6765)
    _site, csa = _interaction(se.estimate_start(data, "csa", nucleus="13C"), "csa")
    assert csa["aniso"] == pytest.approx(-150, rel=0.08)


XQ = np.linspace(-150, 100, 8192)


@pytest.mark.parametrize("mas", [0, 15000])
def test_quadrupolar_central_transition(mas):
    truth = [Site("27Al", 10, fwhm=1.0 if not mas else 0.5).add_quadrupolar(5, 0.3)]
    data = _data(truth, XQ, 130.3e6, mas=mas, transitions="ct", n=6765 if not mas else 2584)
    est = se.estimate_start(data, "quadrupolar", nucleus="27Al", transitions="ct", mas_Hz=mas)
    site, quad = _interaction(est, "quadrupolar")
    assert quad["C_Q"] == pytest.approx(5, rel=0.05)
    assert site["delta_iso"] == pytest.approx(10, abs=1.0)


def test_a_spin_half_has_no_quadrupolar_coupling():
    data = _data([Site("13C", 120).add_csa(-50, 0.2)], X13, 100.6e6, n=987)
    with pytest.raises(ValueError, match="spin-1/2"):
        se.estimate_start(data, "quadrupolar", nucleus="13C")


def test_isotropic_peaks():
    x = np.linspace(0, 10, 8192)
    truth = [Site("1H", 2, fwhm=0.05, label="a"), Site("1H", 5, fwhm=0.1, label="b", amount=2),
             Site("1H", 7.5, fwhm=0.03, label="c", amount=0.5)]
    est = se.estimate_start(_data(truth, x, 400e6), "isotropic", nucleus="1H")
    shifts = sorted(s["delta_iso"] for s in est["sites"])
    assert shifts == pytest.approx([2, 5, 7.5], abs=0.01)
    widths = {round(s["delta_iso"]): s["fwhm"] for s in est["sites"]}
    assert widths[2] == pytest.approx(0.05, rel=0.2) and widths[5] == pytest.approx(0.1, rel=0.2)
    assert est["start_r2"] > 0.99


def test_j_multiplet_spacing_and_partner_count():
    x = np.linspace(0, 10, 8192)
    truth = [Site("1H", 1.2, fwhm=0.003, lineshape="lorentzian").add_jcoupling("1H", 3, 7.0)]
    est = se.estimate_start(_data(truth, x, 400e6), "jcoupling", nucleus="1H",
                            fit_regions=[(1.0, 1.4)])
    site, j = _interaction(est, "jcoupling")
    assert j["J_Hz"] == pytest.approx(7.0, abs=0.1) and j["n"] == 3
    assert site["delta_iso"] == pytest.approx(1.2, abs=0.002)


def test_unknown_kind_and_missing_nucleus():
    data = {"x_ppm": np.linspace(0, 1, 50), "real_spectrum": np.zeros(50), "SFO1": 400.0}
    with pytest.raises(ValueError, match="kind must be"):
        se.estimate_start(data, "dipolar", nucleus="1H")
    with pytest.raises(ValueError, match="nucleus"):
        se.estimate_start(data, "csa")


# ---- GUI -----------------------------------------------------------------------------

def _window(qapp):
    window = MainWindow()
    r = _data([Site("13C", 120, fwhm=1.0).add_csa(-150, 0.4)], X13, 100.6e6, n=6765)
    spectrum = NMRSpectrum(data=np.asarray(r["real_spectrum"]),
                           axes=[NMRAxis(kind="frequency", ppm=np.asarray(r["x_ppm"]))],
                           metadata=NMRMetadata(vendor="simulated", nucleus="13C",
                                                sfo1=100.6))
    window._add_derived_spectrum(spectrum, "C")
    panel = window.combined_fit_panel
    panel.seed_from_spectrum(spectrum)
    panel.regime_combo.setCurrentIndex(panel.regime_combo.findData("static"))
    panel.diso_spin.setValue(100.0)
    panel.add_interaction("csa", aniso=-60.0, asym=0.9)
    return window, panel


def test_gui_estimate_replaces_the_selected_site_and_undo_restores_it(qapp):
    window, panel = _window(qapp)
    label = panel.site_dicts[0]["label"]
    before = [dict(s) for s in panel.site_dicts]
    panel.estimate_kind_combo.setCurrentIndex(panel.estimate_kind_combo.findData("csa"))
    est = panel.estimate_starting_values(quiet=True)
    assert est is not None
    site = panel.site_dicts[0]
    assert site["label"] == label                   # kept
    csa = next(i for i in site["interactions"] if i["type"] == "csa")
    assert csa["aniso"] == pytest.approx(-150, abs=3)
    assert site["delta_iso"] == pytest.approx(120, abs=1)
    assert panel.undo_estimate_button.isEnabled()
    assert "•" in panel.estimate_label.text()
    panel.undo_estimate()
    assert panel.site_dicts[0]["delta_iso"] == before[0]["delta_iso"]
    assert next(i for i in panel.site_dicts[0]["interactions"]
                if i["type"] == "csa")["aniso"] == pytest.approx(-60)


def test_gui_quadrupolar_on_a_spin_half_says_why(qapp):
    window, panel = _window(qapp)
    panel.estimate_kind_combo.setCurrentIndex(panel.estimate_kind_combo.findData("quadrupolar"))
    assert panel.estimate_starting_values(quiet=True) is None
    assert "spin-1/2" in panel.estimate_label.text()
    assert not panel.undo_estimate_button.isEnabled()
