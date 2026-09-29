"""notebook.czjzek: the distribution, its lineshape library, simulation,
fitting and map. Expected values are independent of the code: the
published normalisation, known moments of the Czjzek distribution, the
full quadrupolar simulator, and the CT centre-of-gravity theorem."""
import numpy as np
import pytest
from scipy.integrate import dblquad

from opennmr.notebook import czjzek as cz
from opennmr.notebook.fitting import expand_pcov
from opennmr.notebook.simulations import quadrupolar_spectrum

I, NU0 = 2.5, 130.3e6


@pytest.mark.parametrize("d", [3.0, 5.0, 7.5])
def test_pdf_is_normalised_for_any_d(d):
    total = dblquad(lambda c, e: cz.czjzek_pdf(c, e, 2.0, d), 0, 1, 0, 80)[0]
    assert total == pytest.approx(1.0, abs=1e-8)


def test_pdf_matches_the_published_d5_form():
    # d'Espinose de Lacaillerie, Fretigny, Massiot, JMR 192 (2008) 244,
    # normalised on 0 <= eta <= 1 (a factor 2 over the whole-sphere form).
    c, e, s = np.array([0.5, 3.1, 7.0]), np.array([0.1, 0.4, 0.95]), 2.0
    published = 2 * c ** 4 * e * (1 - e * e / 9) * np.exp(-c * c * (1 + e * e / 3) / (2 * s * s)) \
        / (np.sqrt(2 * np.pi) * s ** 5)
    np.testing.assert_allclose(cz.czjzek_pdf(c, e, s, 5), published, rtol=1e-12)


def test_known_moments():
    st = cz.czjzek_statistics(2.0, 5.0)
    # <C_Q^2 (1 + eta^2/3)> = d sigma^2 exactly; <eta_Q> = 0.61 for Czjzek.
    assert st["rms_PQ"] == pytest.approx(np.sqrt(5) * 2.0, rel=1e-3)
    assert st["mean_eta"] == pytest.approx(0.610, abs=2e-3)
    # Everything scales with sigma at fixed d.
    assert cz.czjzek_statistics(4.0, 5.0)["mean_CQ"] == pytest.approx(2 * st["mean_CQ"], rel=1e-3)


def test_pdf_zero_outside_its_domain_and_rejects_bad_width():
    assert cz.czjzek_pdf(-1.0, 0.5, 1.0) == 0.0
    assert cz.czjzek_pdf(1.0, 1.2, 1.0) == 0.0
    with pytest.raises(ValueError):
        cz.czjzek_pdf(1.0, 0.5, 0.0)


def test_scaling_library_equals_the_full_static_simulator():
    x = np.linspace(-150, 50, 4001)
    lib = cz.CzjzekLibrary(I, NU0, x, 8.0, n_cq=8, n_eta=4, n_zcw=4181)
    assert lib.method == "scaling"
    i, j = 2, 5
    ref = quadrupolar_spectrum(I, lib.cq[j] * 1e6, lib.eta[i], NU0, x, order="second",
                               transitions="ct", fwhm=0, n_zcw=4181)["real_spec"]
    ref = ref / (ref.sum() * lib.dx)
    np.testing.assert_allclose(lib.L[i * 8 + j], ref, atol=1e-10 * ref.max())


def test_mas_scaling_is_the_centreband_of_the_full_simulator():
    x = np.linspace(-150, 50, 4001)
    lib = cz.CzjzekLibrary(I, NU0, x, 8.0, n_cq=8, n_eta=4, mas_Hz=60000, n_zcw=1597,
                           method="scaling")
    i, j = 2, 5
    ref = quadrupolar_spectrum(I, lib.cq[j] * 1e6, lib.eta[i], NU0, x, mas_Hz=60000,
                               order="second", transitions="ct", fwhm=0, n_zcw=1597)["real_spec"]
    ref = ref / (ref.sum() * lib.dx)
    assert np.max(np.abs(lib.L[i * 8 + j] - ref)) < 2e-3 * ref.max()


def test_method_choice_is_explained_and_conservative():
    method, why = cz.choose_library_method(I, "second", "ct", 0.0, NU0, 8.0)
    assert method == "scaling" and "static" in why
    assert cz.choose_library_method(I, "second", "all", 0.0, NU0, 8.0)[0] == "simulated"
    assert cz.choose_library_method(I, "exact", "ct", 0.0, NU0, 8.0)[0] == "simulated"
    assert cz.choose_library_method(1.0, "second", "all", 0.0, NU0, 8.0)[0] == "simulated"
    # Fast spinning: centreband; slow: sidebands too big, so simulated.
    assert cz.choose_library_method(I, "second", "ct", 60000, NU0, 7.3)[0] == "scaling"
    method, why = cz.choose_library_method(I, "second", "ct", 5000, NU0, 7.3)
    assert method == "simulated" and "sidebands" in why


@pytest.mark.parametrize("mas", [0.0, 60000.0])
def test_simulated_spectrum_area_and_centre_of_gravity(mas):
    # The CT centre of gravity is delta_iso + the isotropic second-order
    # shift, averaged over the distribution (static and MAS alike).
    x = np.linspace(-400, 200, 6001)
    sim = cz.czjzek_spectrum(I, 2.0, NU0, x, delta_iso=10, fwhm=2, mas_Hz=mas)
    dx = x[1] - x[0]
    area = sim["real_spec"].sum() * dx
    assert area == pytest.approx(sim["captured"], rel=1e-6)
    assert sim["captured"] > 0.999
    cog = (x * sim["real_spec"]).sum() * dx / area
    assert cog == pytest.approx(10 + sim["stats"]["mean_shift_ppm"], abs=0.02)
    assert sim["method"] == "scaling"
    assert "Czjzek simulation" in sim["log"] and "scaling" in sim["log"]


def _two_site(mas=0.0, noise=5.0, seed=1):
    x = np.linspace(200, -200, 4096)
    a = cz.czjzek_spectrum(I, 3.0, NU0, x, delta_iso=60, fwhm=4, mas_Hz=mas)["real_spec"]
    b = cz.czjzek_spectrum(I, 1.5, NU0, x, delta_iso=10, fwhm=6, mas_Hz=mas)["real_spec"]
    y = 1000 * a / a.max() + 400 * b / b.max() + np.random.default_rng(seed).normal(0, noise, x.size)
    return {"x_ppm": x, "real_spectrum": y, "SFO1": NU0 / 1e6, "nuclide": "27Al"}


def test_fit_recovers_two_sites():
    res = _two_site()
    popt, pcov, free, _s, info = cz.fit_czjzek_spectrum(
        res, [{"d_iso": 50, "sigma": 2, "fwhm": 3, "label": "AlO4"},
              {"d_iso": 15, "sigma": 1, "fwhm": 3, "label": "AlO6"}])
    assert popt[0] == pytest.approx(60, abs=0.05)
    assert popt[1] == pytest.approx(3.0, rel=5e-3)
    assert popt[2] == 5.0 and not free[2]           # d held by default
    assert popt[6] == pytest.approx(10, abs=0.1)
    assert popt[7] == pytest.approx(1.5, rel=1e-2)
    assert popt[4] == pytest.approx(1000, rel=1e-2)
    assert info["warnings"] == []
    assert "refined to the fitted σ" in info["log"]
    report = cz.czjzek_fit_report(popt, expand_pcov(pcov, free, len(popt)), free, info)
    assert report[0]["label"] == "AlO4"
    assert report[0]["mean_CQ"] == pytest.approx(cz.czjzek_statistics(popt[1])["mean_CQ"])
    assert report[0]["mean_CQ_err"] > 0
    total, comps = cz.evaluate_czjzek_fit(res["x_ppm"], popt, info)
    assert len(comps) == 3
    np.testing.assert_allclose(total, comps[0] + comps[1] + comps[2], atol=1e-9)


def test_fit_can_free_d():
    res = _two_site(noise=1.0)
    popt, _pcov, free, _s, _info = cz.fit_czjzek_spectrum(
        res, [{"d_iso": 55, "sigma": 2.5, "fwhm": 3, "fixed": {"d": False}},
              {"d_iso": 12, "sigma": 1.2, "fwhm": 4}])
    assert free[2] and not free[8]
    assert popt[2] == pytest.approx(5.0, abs=0.3)


def test_fit_warns_when_the_tail_runs_off_the_axis():
    res = _two_site()
    keep = (res["x_ppm"] > -10)                   # cut the low-shift tail off
    res = {**res, "x_ppm": res["x_ppm"][keep], "real_spectrum": res["real_spectrum"][keep]}
    _p, _c, _f, _s, info = cz.fit_czjzek_spectrum(res, [{"d_iso": 55, "sigma": 2.5, "fwhm": 3}])
    assert any("falls inside the spectrum" in w for w in info["warnings"])


def test_mixture_map_and_plots():
    import matplotlib
    matplotlib.use("Agg")
    cq_edges, eta_edges, density = cz.czjzek_mixture_map(
        [{"sigma": 3.0, "weight": 3}, {"sigma": 1.0, "weight": 1}], n_cq=150, n_eta=60)
    dc, de = np.diff(cq_edges)[0], np.diff(eta_edges)[0]
    assert density.sum() * dc * de == pytest.approx(1.0, abs=5e-3)
    res = _two_site()
    popt, pcov, free, _s, info = cz.fit_czjzek_spectrum(
        res, [{"d_iso": 50, "sigma": 2, "fwhm": 3}, {"d_iso": 15, "sigma": 1, "fwhm": 3}])
    for layout in ("right", "top"):
        fig = cz.plot_czjzek_fit(res, popt, pcov, info, layout=layout)
        axes = fig.czjzek_axes
        assert axes["map"].get_xlabel().startswith("$C")
        assert axes["spectrum"] is not None and axes["colorbar"] is not None
        box_map, box_spec = axes["map"].get_position(), axes["spectrum"].get_position()
        if layout == "right":
            assert box_spec.x0 > box_map.x1            # spectrum to the right
        else:
            assert box_spec.y0 > box_map.y1            # spectrum above
    with pytest.raises(ValueError):
        cz.plot_czjzek_distribution({"sigma": 1.0}, layout="left")
