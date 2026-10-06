"""Relaxometry extras (user request, 2026-10-06): three-exponential models,
threshold-found integration regions, trapezoid or Simpson integration, and
the correlation matrix of a fit -- processing.relaxometry,
notebook.relaxometry, both relaxation fitters and the advice."""

import os
import sys

import matplotlib
import numpy as np
import pytest
from scipy.integrate import simpson

matplotlib.use("Agg")

from opennmr.notebook import fit_advice  # noqa: E402
from opennmr.notebook import relaxation as nb_relaxation  # noqa: E402
from opennmr.notebook import relaxometry as nb_rx  # noqa: E402
from opennmr.processing import relaxation as pr  # noqa: E402
from opennmr.processing import relaxometry as rx  # noqa: E402

from test_gui import _write_params  # noqa: E402

T = np.geomspace(2e-4, 10, 30)
TRUE_T = (0.002, 0.02, 0.2)
TRUE_A = (0.5, 0.3, 0.2)


def _three(t=T, noise=0.003, seed=3, recovery=False, C=1.9):
    rng = np.random.default_rng(seed)
    if recovery:
        y = sum(a * (1 - C * np.exp(-t / tt)) for a, tt in zip(TRUE_A, TRUE_T))
    else:
        y = sum(a * np.exp(-t / tt) for a, tt in zip(TRUE_A, TRUE_T))
    return y + rng.normal(0, noise, t.size)


# ---- integration rules ---------------------------------------------------------------

@pytest.mark.parametrize("n", [3, 4, 5, 10, 11, 200, 201])
def test_simpson_weights_match_scipy_on_uneven_spacing(n):
    rng = np.random.default_rng(n)
    x = np.sort(rng.uniform(0, 5, n))
    y = np.sin(x) + x ** 2
    assert rx.integrate(x, y, "simpson")[0] == pytest.approx(simpson(y, x=x), abs=1e-12)
    assert rx.integrate(x, y, "trapezoid")[0] == pytest.approx(np.trapezoid(y, x), abs=1e-12)


def test_simpson_is_exact_on_a_cubic_and_the_sign_follows_the_data():
    x = np.linspace(0, 2, 11)
    assert rx.integrate(x, x ** 3, "simpson")[0] == pytest.approx(4.0, abs=1e-12)
    # A descending axis (ppm) gives the same, positive, integral.
    assert rx.integrate(x[::-1], (x ** 3)[::-1], "simpson")[0] == pytest.approx(4.0)


def test_the_propagated_error_is_the_scatter_of_the_integral():
    rng = np.random.default_rng(0)
    x = np.linspace(0, 1, 41)
    for method in rx.INTEGRATION_METHODS:
        _v, predicted = rx.integrate(x, np.zeros_like(x), method, noise=1.0)
        draws = [rx.integrate(x, rng.normal(0, 1, x.size), method)[0] for _ in range(4000)]
        assert np.std(draws) == pytest.approx(predicted, rel=0.05)
    # Simpson's weights carry about 5% more noise than the trapezoid's.
    ratio = rx.integrate(x, 0 * x, "simpson", 1.0)[1] / rx.integrate(x, 0 * x, "trapezoid", 1.0)[1]
    assert 1.03 < ratio < 1.08


def test_integration_arguments_are_checked():
    with pytest.raises(ValueError, match="method"):
        rx.integration_weights([0, 1, 2], "midpoint")
    with pytest.raises(ValueError, match="two points"):
        rx.integration_weights([0.0], "trapezoid")
    with pytest.raises(ValueError, match="ascending"):
        rx.integration_weights([0, 1, 1], "simpson")


def _slices(n=3):
    """Pseudo-2D slices as get_slices tags them, a peak ~3 points per FWHM."""
    from opennmr.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
    ppm = np.linspace(6, 4, 51) + 0.0123
    out = []
    for i in range(n):
        y = 100 * np.exp(-0.5 * ((ppm - 5) / 0.05) ** 2) * np.exp(-i / 2)
        sp = NMRSpectrum(data=y, axes=[NMRAxis(kind="frequency", ppm=ppm, hz=ppm * 400)],
                         metadata=NMRMetadata(vendor="bruker", nucleus="1H", sfo1=400.0,
                                              raw={"slice": {"index": i + 1, "shift": None}}))
        out.append(sp)
    return {"rows": out}


def _exact(lo, hi, w=0.05, a=100.0):
    from scipy.special import erf
    s = w * np.sqrt(2)
    return a * w * np.sqrt(np.pi / 2) * (erf((hi - 5) / s) - erf((lo - 5) / s))


def _integral(region, method):
    rows = pr.NMR2D_integrate_slices(_slices(), [region], method=method,
                                     subtract_baseline=False)["rows"]
    assert rows[0]["integration_method"] == method
    x = rows[0]["x"]
    inside = x[(x >= region[0]) & (x <= region[1])]
    return rows[0]["integrals"][0], (inside.min(), inside.max())


def test_which_rule_is_better_depends_on_the_region_edges():
    """Measured, and the reason both are offered: over a whole peak whose
    region reaches the baseline the trapezoid rule is already all but
    exact (its error terms are the derivatives at the ends, which vanish),
    and Simpson's is *worse* on a coarsely sampled peak; where a region
    edge cuts through signal, Simpson's is the better one."""
    full = {m: _integral((4.6, 5.4), m) for m in rx.INTEGRATION_METHODS}
    exact_full = _exact(*full["trapezoid"][1])
    assert abs(full["trapezoid"][0] - exact_full) <= abs(full["simpson"][0] - exact_full)
    cut = {m: _integral((4.6, 5.03), m) for m in rx.INTEGRATION_METHODS}
    exact_cut = _exact(*cut["trapezoid"][1])
    assert abs(cut["simpson"][0] - exact_cut) < abs(cut["trapezoid"][0] - exact_cut) / 3
    with pytest.raises(ValueError):
        pr.NMR2D_integrate_slices(_slices(), [(4.6, 5.4)], method="boole")


# ---- automatic regions ---------------------------------------------------------------

X = np.linspace(10, 0, 4000)              # descending, as ppm is stored


def _g(c, w, a, x=X):
    return a * np.exp(-0.5 * ((x - c) / w) ** 2)


def test_regions_are_found_around_every_peak_and_hold_their_wings():
    rng = np.random.default_rng(0)
    y = _g(2, 0.02, 100) + _g(2.3, 0.02, 60) + _g(7, 0.05, 20) + _g(5, 0.02, 1.5) \
        + rng.normal(0, 0.2, X.size) + 0.5
    found = rx.find_integration_regions(X, y, threshold=5)
    regions = found["regions"]
    assert len(regions) == 4
    for centre in (2, 2.3, 5, 7):
        assert sum(lo < centre < hi for lo, hi in regions) == 1
    assert all(lo < hi for lo, hi in regions) and regions == sorted(regions)
    assert found["noise"] == pytest.approx(0.2, rel=0.1)
    lo, hi = next(r for r in regions if r[0] < 7 < r[1])
    m = (X >= lo) & (X <= hi)
    assert rx.integrate(X[m], _g(7, 0.05, 20)[m])[0] / (20 * 0.05 * np.sqrt(2 * np.pi)) > 0.99


def test_a_series_counts_signal_in_any_slice_including_inverted_ones():
    rng = np.random.default_rng(1)
    t = np.geomspace(0.001, 5, 10)
    Y = np.array([(1 - 2 * np.exp(-tt / 0.5)) * _g(3, 0.03, 50)
                  + (1 - np.exp(-tt / 2)) * _g(8, 0.03, 10)      # only in the late slices
                  + rng.normal(0, 0.2, X.size) for tt in t])
    found = rx.find_integration_regions(X, Y, threshold=5)
    assert len(found["regions"]) == 2
    assert "any of the 10 slices" in found["notes"][0]
    # The late peak alone, in the first slice, would not have been found.
    assert len(rx.find_integration_regions(X, Y[0], threshold=5)["regions"]) == 1


def test_fraction_is_of_the_series_maximum_so_decayed_slices_add_no_noise_regions():
    rng = np.random.default_rng(5)
    Y = np.array([np.exp(-tt / 0.1) * _g(3, 0.03, 50) + rng.normal(0, 0.2, X.size)
                  for tt in np.geomspace(0.001, 10, 12)])      # the last slices are noise
    found = rx.find_integration_regions(X, Y, threshold=0.05, mode="fraction")
    assert len(found["regions"]) == 1


def test_noise_alone_gives_no_region_and_says_so():
    rng = np.random.default_rng(2)
    found = rx.find_integration_regions(X, rng.normal(0, 1, X.size), threshold=5)
    assert found["regions"] == [] and "Lower the threshold" in found["notes"][0]


def test_fraction_mode_and_bad_arguments():
    rng = np.random.default_rng(3)
    y = _g(2, 0.02, 100) + _g(7, 0.02, 3) + rng.normal(0, 0.2, X.size)
    big_only = rx.find_integration_regions(X, y, threshold=0.1, mode="fraction")
    assert len(big_only["regions"]) == 1
    both = rx.find_integration_regions(X, y, threshold=0.01, mode="fraction")
    assert len(both["regions"]) == 2
    with pytest.raises(ValueError, match="mode"):
        rx.find_integration_regions(X, y, mode="snr")
    with pytest.raises(ValueError, match="positive"):
        rx.find_integration_regions(X, y, threshold=0)
    with pytest.raises(ValueError, match="one value per"):
        rx.find_integration_regions(X[:10], y)


# ---- three exponentials ----------------------------------------------------------------

def _within(fit, names, truth, k=3.0):
    return all(abs(fit["params"][n] - v) <= k * fit["errors"][n] for n, v in zip(names, truth))


def test_triexponential_decay_recovers_three_components_with_honest_errors():
    good = 0
    for seed in range(10):
        fit = pr.fit_relaxation(T, _three(seed=seed), "triexponential_decay",
                                y_err=np.full(T.size, 0.003))
        good += _within(fit, ("T2a", "T2b", "T2c"), TRUE_T)
        assert fit["params"]["T2a"] < fit["params"]["T2b"] < fit["params"]["T2c"]
    assert good >= 9


def test_triexponential_recovery_and_the_notebook_fitter():
    y = _three(recovery=True)
    for fitter in (pr.fit_relaxation, nb_relaxation.fit_relaxation):
        fit = fitter(T, y, "triexponential_recovery", y_err=np.full(T.size, 0.003))
        assert _within(fit, ("T1a", "T1b", "T1c"), TRUE_T)
        assert fit["params"]["C"] == pytest.approx(1.9, abs=0.05)


def test_components_are_sorted_with_their_covariance():
    y = _three()
    start = [0.2, 1.0, 0.3, 0.05, 0.5, 0.002]           # slowest first
    fit = pr.fit_relaxation(T, y, "triexponential_decay", p0=start)
    p = fit["params"]
    assert p["T2a"] < p["T2b"] < p["T2c"] and fit["components_sorted"]
    assert p["A1"] == pytest.approx(0.5, abs=0.02)      # A1 goes with T2a
    sd = np.sqrt(np.diag(fit["pcov"]))
    for name, s in zip(fit["free_names"], sd):
        assert fit["errors"][name] == pytest.approx(s)
    assert list(fit["popt"]) == pytest.approx([p[n] for n in fit["free_names"]])


def test_a_fixed_component_is_not_relabelled():
    fit = pr.fit_relaxation(T, _three(), "triexponential_decay",
                            p0=[0.2, 0.2, 0.3, 0.02, 0.5, 0.002], fixed={"T2a": 0.2})
    assert fit["params"]["T2a"] == 0.2 and not fit.get("components_sorted")


def test_too_few_delays_is_said_plainly_but_p0_still_answers():
    t = np.geomspace(1e-3, 1, 8)
    with pytest.raises(ValueError, match="at least 10 delays"):
        pr.fit_relaxation(t, np.exp(-t / 0.1), "triexponential_decay")
    assert len(rx.p0_triexp_decay(t, np.exp(-t / 0.1))) == 6     # guess table / SCORE probe


def test_multistart_is_never_worse_than_a_median_delay_start():
    tm = np.median(T)
    for seed in range(8):
        y = _three(seed=seed, noise=0.01)
        best = pr.fit_relaxation(T, y, "triexponential_decay")
        median = pr.fit_relaxation(T, y, "triexponential_decay",
                                   p0=[1 / 3, tm * 0.1, 1 / 3, tm, 1 / 3, tm * 10])
        assert best["fit_quality"]["chi2"] <= median["fit_quality"]["chi2"] * (1 + 1e-6)


def test_report_units_and_shares_for_three_components():
    fit = pr.fit_relaxation(T, _three(), "triexponential_decay")
    shares = pr.amplitude_fractions(fit["params"])
    assert set(shares) == {"A1", "A2", "A3"} and sum(shares.values()) == pytest.approx(1)
    assert pr._paired_rate_param("A3", fit["param_names"]) == "T2c"
    assert "ms" in pr.format_param_value("T2c", 0.2) or " s" in pr.format_param_value("T2c", 0.2)


# ---- correlation matrix ----------------------------------------------------------------

def test_correlation_matrix_from_the_covariance():
    fit = pr.fit_relaxation(T, _three(), "triexponential_decay")
    names, R = rx.correlation_matrix(fit)
    assert names == fit["free_names"]
    pcov = fit["pcov"]
    i, j = 1, 3
    assert R[i, j] == pytest.approx(pcov[i, j] / np.sqrt(pcov[i, i] * pcov[j, j]))
    assert np.allclose(np.diag(R), 1) and np.allclose(R, R.T)
    assert np.all(np.abs(R) <= 1)


def test_fixed_parameters_have_no_row_and_bad_input_is_refused():
    t = np.geomspace(1e-3, 3, 16)
    fit = pr.fit_relaxation(t, 1 - 2 * np.exp(-t / 0.3), "inversion_recovery", fixed={"C": 2.0})
    names, R = rx.correlation_matrix(fit)
    assert names == ["M0", "T1"] and R.shape == (2, 2)
    with pytest.raises(ValueError, match="no covariance"):
        rx.correlation_matrix({"free_names": ["a"], "pcov": None})
    with pytest.raises(ValueError, match="does not match"):
        rx.correlation_matrix({"free_names": ["a", "b"], "pcov": np.eye(3)})
    bad = {"free_names": ["a", "b"], "pcov": np.array([[1.0, 0.0], [0.0, np.inf]])}
    _n, R = rx.correlation_matrix(bad)
    assert R[0, 0] == 1 and np.isnan(R[1, 1]) and np.isnan(R[0, 1])


def test_strong_correlations_are_listed_strongest_first():
    fit = {"free_names": ["a", "b", "c"],
           "pcov": np.array([[1, 0.97, 0.2], [0.97, 1, -0.99], [0.2, -0.99, 1.0]])}
    pairs = rx.strong_correlations(fit)
    assert [(a, b) for a, b, _r in pairs] == [("b", "c"), ("a", "b")]


def test_the_figure_is_labelled_and_lower_triangular_by_default():
    fit = pr.fit_relaxation(T, _three(), "triexponential_decay")
    fit["peak_label"] = "Peak 1 (4.21 ppm)"
    fig = nb_rx.plot_correlation_matrix(fit)
    ax = fig.correlation_axes["matrix"]
    n = len(fit["free_names"])
    assert len(ax.texts) == n * (n + 1) // 2
    assert ax.get_title() == "Peak 1 (4.21 ppm)"
    labels = [t.get_text() for t in ax.get_xticklabels()]
    assert labels[1] == r"$\mathit{T}_{\mathrm{2a}}$"
    assert "Correlation" in fig.correlation_axes["colorbar"].get_ylabel()
    assert any("−" in t.get_text() for t in ax.texts)      # a real minus sign
    full = nb_rx.plot_correlation_matrix(fit, triangle="full", annotate=False)
    assert not full.correlation_axes["matrix"].texts
    with pytest.raises(ValueError):
        nb_rx.plot_correlation_matrix(fit, triangle="upper")


def test_parameter_labels():
    assert nb_rx.parameter_label("A1") == r"$\mathit{A}_{\mathrm{1}}$"
    assert nb_rx.parameter_label("C") == r"$\mathit{C}$"
    assert nb_rx.parameter_label("beta") == r"$\beta$"
    assert nb_rx.parameter_label("T1rho") == r"$\mathit{T}_{1\rho}$"
    assert nb_rx.parameter_label("my param") == "my param"


# ---- advice ------------------------------------------------------------------------

def _texts(items):
    return " | ".join(i["text"] for i in items)


def test_advice_supports_three_components_only_when_the_data_do():
    err = np.full(T.size, 0.003)
    three = pr.fit_relaxation(T, _three(), "triexponential_decay", y_err=err)
    assert "clearly better than two" in _texts(fit_advice.advise_relaxation(three))
    rng = np.random.default_rng(3)
    y2 = 0.6 * np.exp(-T / 0.002) + 0.4 * np.exp(-T / 0.2) + rng.normal(0, 0.003, T.size)
    over = pr.fit_relaxation(T, y2, "triexponential_decay", y_err=err)
    items = fit_advice.advise_relaxation(over)
    text = _texts(items)
    assert "third component is not supported" in text
    assert {"type": "set_relaxation_model", "model": "biexponential_decay"} in \
        [i.get("action") for i in items]
    assert "Strongly correlated parameters" in text


def test_a_biexponential_misfit_suggests_three():
    rng = np.random.default_rng(4)
    y = _three(noise=0.0005) + rng.normal(0, 0.0005, T.size)
    two = pr.fit_relaxation(T, y, "biexponential_decay")
    actions = [i.get("action") for i in fit_advice.advise_relaxation(two)]
    assert {"type": "set_relaxation_model", "model": "triexponential_decay"} in actions


def test_limits_are_reported_at_the_bound_not_for_a_loose_value():
    t = np.geomspace(1e-3, 3, 16)
    y = 1 - 4.5 * np.exp(-t / 0.3)                       # needs C > 4: stops at the bound
    fit = pr.fit_relaxation(t, y, "inversion_recovery")
    assert "C stopped at a limit" in _texts(fit_advice.advise_relaxation(fit))
    loose = {"t": t, "y": y, "y_fit": y, "param_names": ["M0", "T1"], "free_names": ["M0", "T1"],
             "params": {"M0": 1.0, "T1": 0.1}, "errors": {"M0": 0.01, "T1": 0.3},
             "rate_param": "T1", "model": "saturation_recovery",
             "bounds": ([-np.inf, 1e-9], [np.inf, np.inf]), "fixed": {}}
    assert "stopped at a limit" not in _texts(fit_advice.advise_relaxation(loose))


# ---- notebook workflow -------------------------------------------------------------------

def _bruker_pseudo2d(root, n1=12, n2=1024, T2=0.1, seed=0):
    pd = root / "pdata" / "1"
    pd.mkdir(parents=True)
    _write_params(root / "acqus", {"NUC1": "1H", "BF1": 400.13, "SFO1": 400.13, "O1": 2000.0,
                                   "SW_h": 4000.0, "NS": 1, "TE": 300})
    _write_params(root / "acqu2s", {"NUC1": "off", "BF1": 400.13, "O1": 0.0, "SW_h": 1.0})
    _write_params(pd / "procs", {"SW_p": 4000.0, "SI": n2, "SF": 400.13, "NC_proc": 0,
                                 "XDIM": n2, "BYTORDP": 0})
    _write_params(pd / "proc2s", {"SW_p": 1.0, "SI": n1, "SF": 400.13, "NC_proc": 0, "XDIM": n1})
    sw, centre = 4000 / 400.13, 2000 / 400.13
    ppm = np.linspace(centre + sw / 2, centre - sw / 2, n2)
    t = np.geomspace(0.001, 2, n1)
    rng = np.random.default_rng(seed)
    data = np.array([1e6 * np.exp(-tt / T2) * np.exp(-0.5 * ((ppm - 5) / 0.02) ** 2)
                     + rng.normal(0, 1e3, n2) for tt in t])
    data.astype("<i4").tofile(pd / "2rr")
    return str(root), t


def test_workflow_with_automatic_regions_and_simpson(tmp_path, capsys):
    path, t = _bruker_pseudo2d(tmp_path / "e1")
    out = nb_relaxation.NMR2D_relaxation_workflow(
        path, rows=list(range(1, 13)), integration="regions", regions="auto",
        model="exponential_decay", vdlist=t, show_fit=False, show_report=True,
        integration_method="simpson")
    (lo, hi), = out["auto_regions"]["regions"]
    assert lo < 5 < hi and hi - lo < 0.5
    assert out["series"][0]["integration_method"] == "simpson"
    assert out["relaxation_fits"][0]["params"]["T2"] == pytest.approx(0.1, rel=0.01)
    assert "Integration regions: 1 region" in capsys.readouterr().out
    with pytest.raises(ValueError, match="'auto'"):
        nb_relaxation.NMR2D_relaxation_workflow(path, rows=[1, 2], integration="regions",
                                                regions="everything", show_fit=False,
                                                show_report=False)
