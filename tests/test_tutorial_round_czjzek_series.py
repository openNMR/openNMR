"""Tutorial round (2026-10-02): a Czjzek fit applied to every slice of a
pseudo-2D series (series_fitting.model_components("czjzek", ...)), and
the bound warnings of the quadrupolar and combined fitters when a
parameter is undetermined rather than on a bound.

Known-answer tests: a synthetic T1 series built from the Czjzek
simulator itself, with a known T1; and fitted values whose 1-sigma error
is set by hand to cover the whole allowed range."""
import numpy as np
import pytest

from opennmr.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
from opennmr.notebook import bridge, series_fitting as SF
from opennmr.notebook import czjzek as cz
from opennmr.notebook import quadrupolar_fitting as QF
from opennmr.notebook.combined_fitting import _warnings as combined_warnings
from opennmr.processing.relaxation import NMR2D_fit_relaxation

NU0 = 130.3e6        # 27Al at 11.7 T


def _pseudo2d(rows, ppm_ascending, nucleus="27Al", sfo1=NU0 / 1e6):
    ppm = ppm_ascending[::-1]
    data = np.asarray(rows)[:, ::-1]
    return NMRSpectrum(
        data=data,
        axes=[NMRAxis(kind="index", index=np.arange(1, len(rows) + 1)),
              NMRAxis(kind="frequency", ppm=ppm, hz=ppm * sfo1)],
        metadata=NMRMetadata(vendor="simulated", nucleus=nucleus, sfo1=sfo1, pseudo_2d=True))


@pytest.fixture(scope="module")
def czjzek_series():
    """A static Czjzek CT relaxing by saturation recovery (T1 = 0.4 s)."""
    x = np.linspace(-250.0, 120.0, 1536)
    shape = cz.czjzek_spectrum(2.5, 3.0, NU0, x, delta_iso=40.0, fwhm=4.0)["normalised_real_spec"]
    delays = np.array([0.01, 0.03, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2])
    rng = np.random.default_rng(3)
    rows = [(1 - np.exp(-t / 0.4)) * shape + rng.normal(0, 0.004, x.size) for t in delays]
    return x, delays, _pseudo2d(rows, x)


def test_czjzek_is_a_series_model_kind():
    assert "czjzek" in SF.MODEL_KINDS
    with pytest.raises(ValueError):
        SF.model_components("not-a-model", {})


def test_czjzek_components_are_the_fitted_curves(czjzek_series):
    x, _delays, spectrum = czjzek_series
    results = bridge.results_from_spectrum(spectrum, row=len(_delays) - 1)
    popt, pcov, free, scale, info = cz.fit_czjzek_spectrum(
        results, [{"d_iso": 38.0, "sigma": 2.6, "fwhm": 5.0, "label": "AlO6"}], I=2.5)
    labels, xc, comps = SF.model_components("czjzek", {"popt": popt, "info": info,
                                                       "x": results["x_ppm"]})
    assert labels == ["AlO6"]
    assert np.all(np.diff(xc) > 0)                       # ascending, as the series fit expects
    total, parts = cz.evaluate_czjzek_fit(xc, popt, info)
    np.testing.assert_allclose(comps[0], parts[0])       # the site, not the baseline
    assert len(comps) == 1


def test_czjzek_saturation_recovery_recovers_t1(czjzek_series):
    x, delays, spectrum = czjzek_series
    results = bridge.results_from_spectrum(spectrum, row=len(delays) - 1)
    popt, _p, _f, _s, info = cz.fit_czjzek_spectrum(
        results, [{"d_iso": 38.0, "sigma": 2.6, "fwhm": 5.0}], I=2.5)
    labels, xc, comps = SF.model_components("czjzek", {"popt": popt, "info": info,
                                                       "x": results["x_ppm"]})
    series = SF.fit_series_with_lineshapes(spectrum, xc, comps, labels, kind="czjzek",
                                           reference_row=len(delays))
    assert [r["model"] for r in series] == ["czjzek"] * len(delays)
    assert series[-1]["amplitudes"][0] == pytest.approx(1.0, abs=0.02)
    fit = NMR2D_fit_relaxation(series, model="saturation_recovery", vdlist=delays,
                               value_source="area", show_report=False)[0]
    assert fit["params"]["T1"] == pytest.approx(0.4, rel=0.05)
    assert "Czjzek lineshapes applied to 8 slice(s)" in SF.series_fit_report(series)


def test_quadrupolar_undetermined_parameter_is_not_reported_on_both_bounds():
    names = QF.QUAD_PARAM_NAMES
    n = len(names)
    k = names.index("eta_Q")
    lower, upper = np.full(n + 2, -np.inf), np.full(n + 2, np.inf)
    lower[k], upper[k] = 0.0, 1.0
    popt = np.zeros(n + 2)
    popt[names.index("C_Q")] = 1.4
    popt[k] = 0.4
    free = np.zeros(n + 2, bool)
    free[k] = True
    info = {"labels": ["Na"], "lower": lower, "upper": upper, "n_sites": 1}

    pcov = np.zeros((n + 2, n + 2))
    pcov[k, k] = 50.0 ** 2                                # 1 sigma = 50 >> the range 0..1
    hits = QF.check_quad_bound_hits(popt, pcov, free, info)
    assert len(hits) == 1
    assert "eta_Q is not determined by the data" in hits[0]
    assert "upper bound" not in hits[0] and "lower bound" not in hits[0]

    pcov[k, k] = 0.0                                       # well determined, mid-range
    assert QF.check_quad_bound_hits(popt, pcov, free, info) == []

    popt[k] = 1.0                                          # genuinely on the upper bound
    hits = QF.check_quad_bound_hits(popt, pcov, free, info)
    assert hits == ["Na: eta_Q at its upper bound 1"]


def test_combined_undetermined_parameter_is_not_reported_on_both_bounds():
    row = {"label": "Na", "name": "asym", "value": 1.0, "error": 1.1, "free": True,
           "lower": 0.0, "upper": 1.0}
    x = np.linspace(-5, 5, 50)
    out = combined_warnings([row], np.zeros((1, 1)), ["Na: asym"], [np.zeros(x.size)], x,
                            np.ones(x.size, bool), None, False)
    bound_lines = [w for w in out if "asym" in w]
    assert len(bound_lines) == 1
    assert "not determined by the data" in bound_lines[0]

    row = dict(row, error=0.01)                            # on the upper bound, well determined
    out = combined_warnings([row], np.zeros((1, 1)), ["Na: asym"], [np.zeros(x.size)], x,
                            np.ones(x.size, bool), None, False)
    assert [w for w in out if "asym" in w] == ["Na: asym at its upper bound 1"]


def test_czjzek_library_is_sized_to_the_fit_window_not_the_whole_spectrum():
    """A narrow window on a very wide spectrum (as a real 23Na echo at
    800 MHz: 3700 ppm sampled every 0.03 ppm) must build a library over
    the window, not over the spectrum -- sized to the spectrum it was
    several GB and the fit was killed. The answer must not change."""
    nu0 = 211.65e6                                        # 23Na at 18.8 T
    x = np.linspace(1800.0, -1800.0, 2 ** 17)             # descending, as imported
    sim = cz.czjzek_spectrum(1.5, 0.6, nu0, x, delta_iso=-0.5, fwhm=0.3, mas_Hz=20e3)
    rng = np.random.default_rng(0)
    results = {"x_ppm": x, "real_spectrum": 1000 * sim["normalised_real_spec"]
               + rng.normal(0, 2.0, x.size), "SFO1": nu0 / 1e6, "nuclide": "23Na"}
    popt, pcov, free, scale, info = cz.fit_czjzek_spectrum(
        results, [{"d_iso": -0.3, "sigma": 0.5, "fwhm": 0.25}], mas_Hz=20e3,
        fit_regions=[(-8.0, 4.0)])
    assert info["library"].x.size < 5000                  # was capped at 2**19
    assert popt[0] == pytest.approx(-0.5, abs=0.02)       # d_iso
    assert popt[1] == pytest.approx(0.6, rel=0.02)        # sigma
    # The default d_iso bounds follow the window too.
    assert info["upper"][0] - info["lower"][0] == pytest.approx(2 * 12.0, rel=0.01)
    assert not any("falls inside" in w for w in info["warnings"])


def test_round_off_sized_errors_are_not_printed_to_seventeen_decimals():
    from opennmr.notebook.combined_fitting import _fmt
    # A single site's share: exactly 100 %, with a covariance error of 1e-16.
    assert _fmt(100.00000000000001, 1.4e-16) == "100"
    assert _fmt(1.40149, 0.00105) == "1.4015 ± 0.0010"
    assert _fmt(12.3, None) == "12.3 (fixed)"


def test_plot_multi_fit_zooms_to_the_fit_window():
    import matplotlib
    matplotlib.use("Agg")
    from opennmr.notebook import multi_fitting as MF
    x = np.linspace(1000.0, -1000.0, 4001)                # wide axis, narrow window
    mask = (x > -10) & (x < 5)
    y = np.exp(-0.5 * ((x + 2) / 1.0) ** 2)
    fit = {"x": x, "y": y, "y_fit": y, "mask": mask, "site_spectra": {"A": y},
           "baseline": np.zeros_like(x), "label": "250 K", "nucleus": "23Na"}
    fig = MF.plot_multi_fit({"experiments": [fit, dict(fit, label="260 K")]})
    left, right = fig.axes[0].get_xlim()
    lo, hi = x[mask].min(), x[mask].max()                # -9.5 and 4.5 on this grid
    pad = 0.05 * (hi - lo)
    assert left == pytest.approx(hi + pad) and right == pytest.approx(lo - pad)
    fig = MF.plot_multi_fit({"experiments": [fit]}, xlims=None)
    assert fig.axes[0].get_xlim() == pytest.approx((1000.0, -1000.0))
    fig = MF.plot_multi_fit({"experiments": [fit]}, xlims=(-4, 2))
    assert fig.axes[0].get_xlim() == pytest.approx((2.0, -4.0))


def test_czjzek_fit_warns_when_the_held_baseline_sits_on_the_pattern():
    """A static Czjzek CT on an axis that ends where the pattern still has
    ~1.5 % of its height: the held baseline (outer 5 %) is partly the tail,
    and sigma/FWHM come out biased far beyond their errors. The fit must
    say so; fitting the baseline (or a wide axis) recovers the truth."""
    x = np.linspace(110.0, -120.0, 3000)
    y = cz.czjzek_spectrum(2.5, 2.5, NU0, x, delta_iso=60.0, fwhm=6.0)["normalised_real_spec"]
    assert y[0] > 0.01                                        # the premise: a tail at the edge
    results = {"x_ppm": x, "real_spectrum": 1000 * y, "SFO1": NU0 / 1e6, "nuclide": "27Al"}
    start = [{"d_iso": 55.0, "sigma": 2.0, "fwhm": 4.0}]
    popt, _p, _f, _s, info = cz.fit_czjzek_spectrum(results, start)
    assert any("baseline was estimated where the fitted pattern" in w for w in info["warnings"])
    assert abs(popt[1] - 2.5) > 0.05                          # biased, as the warning says
    popt, _p, _f, _s, info = cz.fit_czjzek_spectrum(results, start, fit_baseline=True)
    assert not any("baseline was estimated" in w for w in info["warnings"])
    assert popt[1] == pytest.approx(2.5, abs=0.005)
    assert popt[3] == pytest.approx(6.0, abs=0.1)

    # Clear baseline regions: no warning either.
    x = np.linspace(200.0, -400.0, 6000)
    y = cz.czjzek_spectrum(2.5, 2.5, NU0, x, delta_iso=60.0, fwhm=6.0)["normalised_real_spec"]
    results = {"x_ppm": x, "real_spectrum": 1000 * y, "SFO1": NU0 / 1e6, "nuclide": "27Al"}
    popt, _p, _f, _s, info = cz.fit_czjzek_spectrum(results, start)
    assert not any("baseline was estimated" in w for w in info["warnings"])
    assert popt[1] == pytest.approx(2.5, abs=0.005)
