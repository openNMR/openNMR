"""Manual baselines (notebook.manual_baseline, processing.ft's
"manual_polynomial"/"manual_sine" methods) and the Herzfeld-Berger skew
definition -- explicit user requests, 2026-09-29."""
import numpy as np
import pytest

from opennmr.notebook import (
    MANUAL_BASELINE_PARAMS, describe_manual_baseline, evaluate_manual_baseline,
    fit_manual_baseline, manual_baseline_correct, plot_manual_baseline,
)
from opennmr.notebook.manual_baseline import region_mask
from opennmr.processing import ft

X = np.linspace(-100.0, 300.0, 4001)
PEAK = 100.0 * np.exp(-((X - 100.0) / 5.0) ** 2)
REGIONS = [(-100.0, 50.0), (150.0, 300.0)]
POLY = {"A": 2e-8, "B": -5e-6, "C": 3e-4, "D": 0.05, "E": 3.0}
SINE = {"A": 2.0, "B": 0.03, "C": 0.7, "D": -1.0}


def _noisy(baseline, sigma=0.01, seed=0):
    return baseline + PEAK + np.random.default_rng(seed).normal(0.0, sigma, X.size)


# ---- the models, written out by hand -------------------------------------

def test_polynomial_is_the_quartic_written_out():
    x = np.array([-2.0, 0.0, 1.5, 10.0])
    expected = 1.0 * x ** 4 + 2.0 * x ** 3 + 3.0 * x ** 2 + 4.0 * x + 5.0
    np.testing.assert_allclose(evaluate_manual_baseline(x, "polynomial", [1, 2, 3, 4, 5]), expected)
    np.testing.assert_allclose(
        evaluate_manual_baseline(x, "polynomial", {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5}), expected)


def test_sine_is_a_sin_bx_plus_c_plus_d():
    x = np.array([-3.0, 0.0, 0.5, 7.0])
    expected = 2.5 * np.sin(0.4 * x + 1.1) - 0.3
    np.testing.assert_allclose(evaluate_manual_baseline(x, "sine", [2.5, 0.4, 1.1, -0.3]), expected)


def test_wrong_parameter_count_is_refused():
    with pytest.raises(ValueError, match="5 parameters"):
        evaluate_manual_baseline(X, "polynomial", [1, 2, 3])
    with pytest.raises(ValueError, match="polynomial' or 'sine"):
        evaluate_manual_baseline(X, "cubic-spline", [1])


# ---- fitting ---------------------------------------------------------------

def test_polynomial_fit_recovers_every_coefficient():
    y = _noisy(evaluate_manual_baseline(X, "polynomial", POLY))
    fit = fit_manual_baseline(X, y, "polynomial", regions=REGIONS)
    for name, true in POLY.items():
        assert fit["params"][name] == pytest.approx(true, rel=5e-3, abs=5 * fit["errors"][name])
        # the reported error is an honest 1-sigma: within 5 of the truth
        assert abs(fit["params"][name] - true) < 5 * fit["errors"][name]
    assert fit["rms"] == pytest.approx(0.01, rel=0.1)          # the noise
    assert fit["n_points"] == int(region_mask(X, REGIONS).sum())


def test_polynomial_fit_is_exact_on_noiseless_data():
    y = evaluate_manual_baseline(X, "polynomial", POLY) + PEAK
    fit = fit_manual_baseline(X, y, "polynomial", regions=REGIONS)
    np.testing.assert_allclose(fit["baseline"], evaluate_manual_baseline(X, "polynomial", POLY),
                               atol=1e-9)


def test_fixed_coefficients_are_held():
    # A = B = 0 fixed: a quadratic, fitted exactly.
    truth = {"A": 0.0, "B": 0.0, "C": 1e-4, "D": -0.02, "E": 1.0}
    y = evaluate_manual_baseline(X, "polynomial", truth) + PEAK
    fit = fit_manual_baseline(X, y, "polynomial", regions=REGIONS, fixed={"A": 0, "B": 0})
    assert fit["params"]["A"] == 0.0 and fit["params"]["B"] == 0.0
    assert fit["errors"]["A"] == 0.0 and fit["fixed"] == ["A", "B"]
    for name in "CDE":
        assert fit["params"][name] == pytest.approx(truth[name], rel=1e-8, abs=1e-12)


def test_fixed_at_a_nonzero_value():
    y = evaluate_manual_baseline(X, "polynomial", POLY) + PEAK
    fit = fit_manual_baseline(X, y, "polynomial", regions=REGIONS, fixed={"E": 3.0})
    assert fit["params"]["E"] == 3.0
    assert fit["params"]["D"] == pytest.approx(POLY["D"], rel=1e-8)


def test_sine_fit_recovers_amplitude_frequency_phase_offset():
    y = _noisy(evaluate_manual_baseline(X, "sine", SINE))
    fit = fit_manual_baseline(X, y, "sine", regions=REGIONS)
    for name, true in SINE.items():
        assert fit["params"][name] == pytest.approx(true, rel=2e-3, abs=2e-3)
    np.testing.assert_allclose(y - fit["baseline"], PEAK, atol=0.06)


def test_sine_fit_reports_conventional_signs():
    # -A sin(-Bx - C) + D is the same curve; reported with A, B >= 0.
    truth = {"A": -1.5, "B": -0.02, "C": -2.0, "D": 0.5}
    y = evaluate_manual_baseline(X, "sine", truth)
    fit = fit_manual_baseline(X, y, "sine", regions=REGIONS)
    assert fit["params"]["A"] > 0 and fit["params"]["B"] > 0
    assert -np.pi < fit["params"]["C"] <= np.pi
    np.testing.assert_allclose(fit["baseline"], y, atol=1e-6)


def test_sine_with_frequency_fixed():
    y = evaluate_manual_baseline(X, "sine", SINE) + PEAK
    fit = fit_manual_baseline(X, y, "sine", regions=REGIONS, fixed={"B": 0.03})
    assert fit["params"]["B"] == 0.03
    assert fit["params"]["A"] == pytest.approx(2.0, rel=1e-6)


def test_regions_are_required_to_hold_points():
    with pytest.raises(ValueError, match="no points"):
        fit_manual_baseline(X, PEAK, "polynomial", regions=[(1000, 2000)])
    with pytest.raises(ValueError, match="cannot determine"):
        fit_manual_baseline(X, PEAK, "polynomial", regions=[(0.0, 0.25)])


def test_no_regions_falls_back_to_the_edges():
    y = evaluate_manual_baseline(X, "polynomial", {"D": 0.01, "E": 2.0}) + PEAK
    fit = fit_manual_baseline(X, y, "polynomial", fixed={"A": 0, "B": 0, "C": 0})
    assert fit["n_points"] == 2 * 200
    assert fit["params"]["D"] == pytest.approx(0.01, rel=1e-6)


def test_complex_data_is_fitted_on_its_real_part():
    y = evaluate_manual_baseline(X, "polynomial", POLY) + PEAK
    fit = fit_manual_baseline(X, y + 1j * 50.0, "polynomial", regions=REGIONS)
    assert fit["params"]["E"] == pytest.approx(POLY["E"], rel=1e-8)


# ---- notebook result dicts, plotting, logging --------------------------------

def test_manual_baseline_correct_on_a_result_dict():
    result = {"x_ppm": X, "real_spectrum": evaluate_manual_baseline(X, "sine", SINE) + PEAK,
              "label": "kept"}
    out = manual_baseline_correct(result, "sine", regions=REGIONS)
    np.testing.assert_allclose(out["real_spectrum"], PEAK, atol=1e-5)
    assert out["label"] == "kept" and "baseline_fit" in out
    # a copy: the input is untouched
    assert not np.allclose(result["real_spectrum"], out["real_spectrum"])
    # hand-set parameters are used as given
    hand = manual_baseline_correct(result, "sine", params=[2.0, 0.03, 0.7, -1.0])
    np.testing.assert_allclose(hand["real_spectrum"], PEAK, atol=1e-12)


def test_plot_manual_baseline_draws_three_traces_and_the_regions():
    import matplotlib
    matplotlib.use("Agg")
    y = evaluate_manual_baseline(X, "polynomial", POLY) + PEAK
    fit = fit_manual_baseline(X, y, "polynomial", regions=REGIONS)
    ax = plot_manual_baseline(X, y, fit)
    assert len(ax.lines) == 3 and len(ax.patches) == 2
    assert ax.get_xlim()[0] > ax.get_xlim()[1]          # ppm runs right to left


def test_description_states_the_model_and_numbers():
    text = describe_manual_baseline("sine", [2, 0.03, 0.7, -1], 3)
    assert text.startswith("A·sin(B·x + C) + D (x in ppm)")
    assert "B = 0.03" in text and "D = -1" in text


# ---- through processing.ft (what the GUI calls) ------------------------------

def test_compute_baseline_evaluates_fixed_manual_params():
    params = [POLY[n] for n in MANUAL_BASELINE_PARAMS["polynomial"]]
    out = ft.compute_baseline(X, PEAK, method="manual_polynomial", manual_params=params)
    np.testing.assert_allclose(out, evaluate_manual_baseline(X, "polynomial", POLY))


def test_compute_baseline_refits_to_regions_on_each_trace():
    y1 = evaluate_manual_baseline(X, "sine", SINE) + PEAK
    y2 = evaluate_manual_baseline(X, "sine", {**SINE, "A": 5.0}) + PEAK
    stale = [1.0, 0.01, 0.0, 0.0]
    for y in (y1, y2):
        corrected = ft.baseline_correct(X, y, REGIONS, method="manual_sine",
                                        manual_params=stale, manual_refit=True)
        np.testing.assert_allclose(corrected, PEAK, atol=1e-5)


def test_compute_baseline_refit_honours_fixed_flags():
    y = evaluate_manual_baseline(X, "polynomial", {"C": 1e-4, "D": 0.02, "E": 1.0}) + PEAK
    out = ft.compute_baseline(X, y, REGIONS, method="manual_polynomial",
                              manual_params=[0, 0, 0, 0, 0], manual_fixed=[True, True, False, False, False],
                              manual_refit=True)
    np.testing.assert_allclose(y - out, PEAK, atol=1e-8)


def test_unknown_method_message_lists_the_manual_ones():
    with pytest.raises(ValueError, match="manual_sine"):
        ft.compute_baseline(X, PEAK, method="nope")


# ---- Herzfeld-Berger skew: the standard definition ---------------------------

from opennmr.notebook.simulations import (  # noqa: E402
    convert_csa_parameters, herzfeld_berger_csa_tensor,
)


@pytest.mark.parametrize("d11, d22, d33", [
    (100.0, 20.0, -50.0), (10.0, 10.0, -80.0), (60.0, -30.0, -30.0), (5.0, 0.0, -5.0)])
def test_skew_is_three_d22_minus_diso_over_span(d11, d22, d33):
    iso = (d11 + d22 + d33) / 3.0
    span = d11 - d33
    kappa = 3.0 * (d22 - iso) / span
    principal = np.sort(np.diag(herzfeld_berger_csa_tensor(iso, span, kappa)))[::-1]
    np.testing.assert_allclose(principal, [d11, d22, d33], atol=1e-12)
    assert convert_csa_parameters(iso, span, kappa, "Herzfeld-Berger", "Herzfeld-Berger")[2] == kappa


def test_skew_plus_one_and_minus_one():
    # kappa = +1: d22 = d11 (unique component d33, lowest); -1: d22 = d33.
    plus = np.sort(np.diag(herzfeld_berger_csa_tensor(0.0, 90.0, 1.0)))[::-1]
    assert plus[0] == pytest.approx(plus[1]) and plus[0] == pytest.approx(30.0)
    minus = np.sort(np.diag(herzfeld_berger_csa_tensor(0.0, 90.0, -1.0)))[::-1]
    assert minus[1] == pytest.approx(minus[2]) and minus[0] == pytest.approx(60.0)


def test_haeberlen_to_herzfeld_berger_uses_the_standard_skew():
    # Haeberlen (10, -80, 0.4): principal 66, 34, -70 ppm (worked by hand)
    iso, span, skew = convert_csa_parameters(10.0, -80.0, 0.4, "Haeberlen", "Herzfeld-Berger")
    assert (iso, span) == pytest.approx((10.0, 136.0))
    assert skew == pytest.approx(3.0 * (34.0 - 10.0) / 136.0)   # +0.529
