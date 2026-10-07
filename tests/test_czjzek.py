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
    N = len(cz.CZJZEK_PARAM_NAMES)
    assert popt[N] == pytest.approx(10, abs=0.1)
    assert popt[N + 1] == pytest.approx(1.5, rel=1e-2)
    assert popt[6] == 0 and popt[7] == 0 and not free[6] and not free[7]   # plain: no mean EFG
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
    assert free[2] and not free[len(cz.CZJZEK_PARAM_NAMES) + 2]
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
        assert axes["map"].get_xlabel().startswith(r"$\mathit{C}_\mathrm{Q}")
        assert axes["spectrum"] is not None and axes["colorbar"] is not None
        box_map, box_spec = axes["map"].get_position(), axes["spectrum"].get_position()
        if layout == "right":
            assert box_spec.x0 > box_map.x1            # spectrum to the right
        else:
            assert box_spec.y0 > box_map.y1            # spectrum above
    with pytest.raises(ValueError):
        cz.plot_czjzek_distribution({"sigma": 1.0}, layout="left")


# ---- the extended Czjzek model (a mean EFG plus a random part) ---------------

def test_extended_samples_reduce_to_czjzek_at_zero_mean_efg():
    st = cz.czjzek_statistics(2.0, 5.0)
    cq, eta = cz.czjzek_samples(2.0, 0.0, 0.0)
    assert cq.mean() == pytest.approx(st["mean_CQ"], rel=2e-3)
    assert eta.mean() == pytest.approx(st["mean_eta"], rel=2e-3)
    ce, _c, ee, _e = cz.czjzek_grid(12, 64, 20)
    analytic = cz.czjzek_weights(2.0, 5, ce, ee)
    sampled = cz.extended_czjzek_weights(2.0, 0.0, 0.0, ce, ee)
    assert np.abs(analytic - sampled).sum() < 0.08          # L1 over the whole grid


def test_extended_samples_collapse_to_the_mean_efg_and_keep_the_second_moment():
    cq, eta = cz.czjzek_samples(1e-5, 5.0, 0.4)
    assert cq.mean() == pytest.approx(5.0, rel=1e-5) and eta.mean() == pytest.approx(0.4, abs=1e-5)
    # <C_Q^2 (1 + eta^2/3)> = 5 sigma^2 + C_Q0^2 (1 + eta0^2/3): V0 and the
    # random part add in quadrature (the random part has zero mean).
    cq, eta = cz.czjzek_samples(1.0, 4.0, 0.5)
    assert np.mean(cq ** 2 * (1 + eta ** 2 / 3)) == pytest.approx(5 + 16 * (1 + 0.25 / 3), rel=2e-3)
    with pytest.raises(ValueError):
        cz.czjzek_samples(1.0, -1.0)


def test_extended_spectrum_matches_plain_at_zero_and_needs_d5():
    x = np.linspace(200, -200, 2048)
    a = cz.czjzek_spectrum(I, 3.0, NU0, x, delta_iso=60, fwhm=4)["real_spec"]
    b = cz.czjzek_spectrum(I, 3.0, NU0, x, delta_iso=60, fwhm=4, cq0_MHz=1e-9)
    assert b["extended"]
    assert np.max(np.abs(a - b["real_spec"])) < 5e-3 * a.max()
    with pytest.raises(ValueError):
        cz.czjzek_spectrum(I, 3.0, NU0, x, d=4.0, cq0_MHz=2.0)


def test_extended_spectrum_approaches_a_single_site_as_sigma_shrinks():
    x = np.linspace(100, -150, 4096)
    ext = cz.czjzek_spectrum(I, 0.02, NU0, x, fwhm=2, cq0_MHz=4.0, eta0=0.3, n_cq=200)
    single = quadrupolar_spectrum(I, 4e6, 0.3, NU0, x, fwhm=2, transitions="ct")["real_spec"]
    a, b = ext["real_spec"] / ext["real_spec"].max(), single / single.max()
    assert np.max(np.abs(a - b)) < 0.06
    assert ext["stats"]["mean_CQ"] == pytest.approx(4.0, rel=2e-3)


def test_fit_recovers_the_mean_efg():
    x = np.linspace(200, -200, 4096)
    sim = cz.czjzek_spectrum(I, 0.8, NU0, x, delta_iso=60, fwhm=3, cq0_MHz=5.0, eta0=0.3)
    y = 1000 * sim["real_spec"] / sim["real_spec"].max()
    y = y + np.random.default_rng(3).normal(0, 5, x.size)
    res = {"x_ppm": x, "real_spectrum": y, "SFO1": NU0 / 1e6, "nuclide": "27Al"}
    popt, pcov, free, _s, info = cz.fit_czjzek_spectrum(
        res, [{"d_iso": 55, "sigma": 1.2, "fwhm": 4, "CQ0": 4.0, "eta0": 0.5}])
    assert info["extended"] == [True] and free[6] and free[7] and not free[2]
    assert popt[0] == pytest.approx(60, abs=0.1)
    assert popt[1] == pytest.approx(0.8, rel=0.03)
    assert popt[6] == pytest.approx(5.0, rel=0.01)
    assert popt[7] == pytest.approx(0.3, abs=0.03)
    rep = cz.czjzek_fit_report(popt, expand_pcov(pcov, free, len(popt)), free, info)[0]
    assert rep["extended"] and rep["CQ0_err"] > 0 and rep["mean_CQ_err"] > 0
    assert rep["mean_CQ"] > popt[6]              # the random part adds to |V|
    assert "Extended Czjzek" in info["log"]


def test_extended_fit_rejects_a_free_d():
    res = _two_site()
    with pytest.raises(ValueError, match="d = 5"):
        cz.fit_czjzek_spectrum(res, [{"d_iso": 55, "sigma": 2.5, "fwhm": 3, "CQ0": 2.0,
                                      "fixed": {"d": False}}])


def test_fit_warns_when_the_mean_efg_is_not_needed():
    res = _two_site(noise=2.0)
    _p, _c, _f, _s, info = cz.fit_czjzek_spectrum(
        res, [{"d_iso": 55, "sigma": 2.5, "fwhm": 3, "model": "extended", "CQ0": 0.0},
              {"d_iso": 12, "sigma": 1.2, "fwhm": 4}])
    assert info["extended"] == [True, False]
    # Either C_Q0 reaches 0 (within a grid cell), or -- on the default
    # 64-cell grid -- it settles below sigma, traded against it, which is
    # the same verdict (TODO 2026-10-07 round 2): both say so.
    assert any("need no mean EFG" in w for w in info["warnings"])


def test_a_real_mean_efg_is_not_called_unneeded():
    x = np.linspace(200, -200, 4096)
    a = cz.czjzek_spectrum(I, 1.0, NU0, x, delta_iso=60, fwhm=4, cq0_MHz=5.0, eta0=0.3)["real_spec"]
    y = 1000 * a / a.max() + np.random.default_rng(3).normal(0, 2.0, x.size)
    res = {"x_ppm": x, "real_spectrum": y, "SFO1": NU0 / 1e6, "nuclide": "27Al"}
    popt, _c, _f, _s, info = cz.fit_czjzek_spectrum(
        res, [{"d_iso": 55, "sigma": 1.5, "fwhm": 3, "model": "extended", "CQ0": 4.0}])
    assert popt[6] == pytest.approx(5.0, rel=0.05)
    assert not any("need no mean EFG" in w for w in info["warnings"])


def test_old_six_parameter_popt_still_evaluates():
    res = _two_site()
    popt, pcov, free, _s, info = cz.fit_czjzek_spectrum(res, [{"d_iso": 55, "sigma": 2.5,
                                                                "fwhm": 3}])
    old = np.concatenate([popt[:6], popt[8:]])
    old_free = np.concatenate([free[:6], free[8:]])
    old_pcov = np.delete(np.delete(expand_pcov(pcov, free, len(popt)), [6, 7], 0), [6, 7], 1)
    np.testing.assert_allclose(cz.evaluate_czjzek_fit(res["x_ppm"], old, info)[0],
                               cz.evaluate_czjzek_fit(res["x_ppm"], popt, info)[0])
    rep = cz.czjzek_fit_report(old, old_pcov, old_free, info)[0]
    assert rep["CQ0"] == 0 and rep["sigma"] == pytest.approx(popt[1])


def test_extended_map_is_normalised_and_centred():
    import matplotlib
    matplotlib.use("Agg")
    cq_edges, eta_edges, density = cz.czjzek_mixture_map(
        [{"sigma": 0.5, "CQ0": 4.0, "eta0": 0.2}], n_cq=150, n_eta=60)
    dc, de = np.diff(cq_edges)[0], np.diff(eta_edges)[0]
    assert density.sum() * dc * de == pytest.approx(1.0, abs=1e-2)
    cq_c = 0.5 * (cq_edges[1:] + cq_edges[:-1])
    peak_cq = cq_c[np.argmax(density.sum(axis=0))]
    assert peak_cq == pytest.approx(4.0, abs=0.3)
    fig = cz.plot_czjzek_distribution([{"sigma": 0.5, "CQ0": 4.0, "eta0": 0.2, "label": "A"},
                                       {"sigma": 2.0, "label": "B"}])
    labels = [l.get_label() for l in fig.czjzek_axes["map"].get_lines()]
    assert any("Q0" in l for l in labels)
