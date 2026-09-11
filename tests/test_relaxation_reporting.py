"""What a fit result *says*: the type of its uncertainty, its units,
its quality numbers, and the joint (multivariate) fit."""

import numpy as np
import pytest

from opennmr.processing import diffusion as dif
from opennmr.processing.relaxation import (
    UNCERTAINTY_KINDS, describe_uncertainty, fit_relaxation, fit_relaxation_global,
    format_global_report, format_param_value, format_relaxation_report,
)


def _decay_series(t2=0.05, n=16, noise=0.0, seed=0):
    t = np.linspace(0.0, 0.3, n)
    y = 10.0 * np.exp(-t / t2)
    if noise:
        y = y + np.random.default_rng(seed).normal(0.0, noise, n)
    return t, y


# ============================================================
# Which kind of uncertainty
# ============================================================

def test_supplying_point_errors_gives_a_measured_uncertainty():
    t, y = _decay_series(noise=0.05, seed=1)
    fit = fit_relaxation(t, y, model="exponential_decay", y_err=np.full(t.size, 0.05))
    assert fit["uncertainty"]["kind"] == "measured"
    assert fit["uncertainty"]["from_measurement_errors"] is True
    assert "supplied" in fit["uncertainty"]["description"]


def test_omitting_point_errors_gives_a_scatter_uncertainty():
    t, y = _decay_series(noise=0.05, seed=1)
    fit = fit_relaxation(t, y, model="exponential_decay")
    assert fit["uncertainty"]["kind"] == "scatter"
    assert fit["uncertainty"]["from_measurement_errors"] is False
    # The claim that makes this kind different, stated in the result
    # rather than left for the reader to know.
    assert "Assumes the model is correct" in fit["uncertainty"]["description"]


def test_the_two_uncertainty_kinds_give_genuinely_different_numbers():
    # The whole reason for labelling them. Same data, same model; the
    # only difference is whether an independent noise estimate was
    # given, and the reported error changes with it.
    t, y = _decay_series(noise=0.05, seed=3)
    scatter = fit_relaxation(t, y, model="exponential_decay")
    measured = fit_relaxation(t, y, model="exponential_decay", y_err=np.full(t.size, 0.5))
    assert measured["errors"]["T2"] != pytest.approx(scatter["errors"]["T2"], rel=0.05)


def test_uncertainty_block_carries_its_degrees_of_freedom():
    t, y = _decay_series(n=20)
    fit = fit_relaxation(t, y, model="exponential_decay")
    assert fit["uncertainty"]["dof"] == 18  # 20 points, 2 free parameters
    assert fit["uncertainty"]["sigma_level"] == 1.0


def test_describe_uncertainty_matches_the_documented_kinds():
    for from_errors, kind in ((True, "measured"), (False, "scatter")):
        block = describe_uncertainty(from_errors, dof=10)
        assert block["kind"] == kind
        assert block["short"] == UNCERTAINTY_KINDS[kind]["short"]
        assert block["description"] == UNCERTAINTY_KINDS[kind]["long"]


# ============================================================
# Fit quality
# ============================================================

def test_fit_quality_reports_r_squared_and_the_point_count():
    t, y = _decay_series()
    quality = fit_relaxation(t, y, model="exponential_decay")["fit_quality"]
    assert quality["r_squared"] == pytest.approx(1.0, abs=1e-9)
    assert quality["n_points"] == 16
    assert quality["dof"] == 14


def test_r_squared_falls_for_a_model_that_does_not_fit():
    t = np.linspace(0.0, 0.3, 20)
    y = 3.0 + 2.0 * t  # a rising line is not an exponential decay
    fit = fit_relaxation(t, y, model="exponential_decay")
    assert fit["fit_quality"]["r_squared"] < 0.99


# ============================================================
# Formatting
# ============================================================

def test_a_relaxation_time_is_formatted_in_its_own_unit():
    assert format_param_value("T2", 0.00152, 0.00003) == "1.52 ± 0.03 ms"
    assert format_param_value("T1", 2.5) == "2.50 s"


def test_a_diffusion_coefficient_is_formatted_in_m2_per_s():
    text = format_param_value("D", 2.03e-9, 0.05e-9)
    assert "m^2/s" in text and "e-09" in text
    # Numbered components from build_diffusion_model get the same
    # treatment as the single "D" of the named models.
    assert "m^2/s" in format_param_value("D2", 4.5e-12)


def test_a_dimensionless_parameter_gets_no_invented_unit():
    text = format_param_value("beta", 0.812, 0.014)
    assert "m^2/s" not in text and "ms" not in text
    assert "0.812" in text


def test_report_states_the_uncertainty_kind_and_the_quality():
    t, y = _decay_series(noise=0.05, seed=4)
    report = format_relaxation_report(fit_relaxation(t, y, model="exponential_decay"))
    assert "Uncertainty" in report
    assert "residual scatter" in report
    assert "R²" in report and "reduced χ²" in report
    assert "points / d.o.f" in report


def test_report_headlines_the_rate_constant_in_its_own_unit():
    t, y = _decay_series(t2=0.05)
    report = format_relaxation_report(fit_relaxation(t, y, model="exponential_decay"))
    assert "T2 = " in report and "ms" in report


def test_report_marks_a_fixed_parameter_as_fixed():
    t = np.linspace(0.0, 2.0, 16)
    y = 5.0 * (1.0 - 2.0 * np.exp(-t / 0.4))
    report = format_relaxation_report(
        fit_relaxation(t, y, model="inversion_recovery", fixed={"C": 2.0}))
    assert "(fixed)" in report


def test_report_shows_population_weights_for_a_multi_component_model():
    b = np.linspace(0, 4e9, 24)
    y = 0.7 * np.exp(-3e-9 * b) + 0.3 * np.exp(-2e-10 * b)
    spec = dif.build_diffusion_model(2, expected_D=[3e-9, 2e-10])
    report = format_relaxation_report(
        fit_relaxation(b, y, model="diffusion_n", custom_model=spec))
    assert "Population weights" in report
    assert "70.00%" in report and "30.00%" in report


def test_report_accepts_a_custom_title():
    t, y = _decay_series()
    report = format_relaxation_report(fit_relaxation(t, y, model="exponential_decay"),
                                       title="Peak 3 (4.21 ppm)")
    assert "Peak 3 (4.21 ppm)" in report


# ============================================================
# Guesses, bounds, and the log search space
# ============================================================

def test_the_fit_result_carries_back_the_guesses_and_bounds_it_used():
    t, y = _decay_series()
    fit = fit_relaxation(t, y, model="exponential_decay", p0=[9.0, 0.04],
                         bounds=([0.0, 1e-4], [100.0, 10.0]))
    assert fit["p0"] == [9.0, 0.04]
    assert fit["bounds"] == ([0.0, 1e-4], [100.0, 10.0])


def test_a_supplied_guess_is_honoured():
    # A deliberately poor but legal guess still converges, and the
    # result records what it started from.
    t, y = _decay_series(t2=0.05)
    fit = fit_relaxation(t, y, model="exponential_decay", p0=[1.0, 1.0])
    assert fit["params"]["T2"] == pytest.approx(0.05, rel=1e-3)


def test_bounds_are_actually_enforced():
    # A bound below the true value keeps the fit off it. p0 is given
    # explicitly because the model's own automatic guess (the median
    # delay, ~0.15 s) falls outside these deliberately-tight bounds --
    # scipy rejects that outright, which is why the GUI's guess table
    # checks guess-within-bounds itself and says so in plain words.
    t, y = _decay_series(t2=0.05)
    fit = fit_relaxation(t, y, model="exponential_decay", p0=[10.0, 0.3],
                         bounds=([0.0, 0.2], [1000.0, 10.0]))
    assert fit["params"]["T2"] >= 0.2 - 1e-9


def test_a_log_fitted_parameter_is_reported_in_natural_units():
    # D is searched in log10 space but must never *surface* that way --
    # the value and its error both come back in m^2/s.
    b = np.linspace(0, 4e9, 24)
    y = np.exp(-1.5e-9 * b)
    fit = fit_relaxation(b, y, model="stejskal_tanner")
    assert fit["params"]["D"] == pytest.approx(1.5e-9, rel=1e-4)
    assert 0 < fit["errors"]["D"] < 1e-9


def test_log_space_bounds_may_be_given_as_zero():
    # "No lower bound" is naturally written as 0 for a positive
    # quantity; it has no logarithm, so it is floored rather than
    # rejected.
    b = np.linspace(0, 4e9, 24)
    y = np.exp(-1.5e-9 * b)
    fit = fit_relaxation(b, y, model="stejskal_tanner",
                         bounds=([0.0, 0.0], [np.inf, np.inf]))
    assert fit["params"]["D"] == pytest.approx(1.5e-9, rel=1e-3)


def test_the_spec_travels_with_the_result_for_a_built_model():
    # So anything re-evaluating the curve later -- the uncertainty
    # band, the report's weights, a refit -- can, even though the model
    # exists nowhere in RELAXATION_MODELS to look up by name.
    b = np.linspace(0, 4e9, 24)
    y = np.exp(-1e-9 * b)
    spec = dif.build_diffusion_model(1)
    fit = fit_relaxation(b, y, model="diffusion_n", custom_model=spec)
    assert fit["spec"] is spec


# ============================================================
# Global (multivariate) fitting
# ============================================================

def _shared_d_datasets(d=8.5e-10, noise=0.01, seed=1):
    b = np.linspace(0, 3e9, 18)
    rng = np.random.default_rng(seed)
    return b, [
        {"t": b, "y": amp * np.exp(-d * b) + rng.normal(0, noise * amp, b.size), "label": label}
        for amp, label in [(10.0, "peak A"), (4.0, "peak B"), (1.5, "peak C")]
    ]


def test_global_fit_recovers_one_shared_diffusion_coefficient():
    _b, datasets = _shared_d_datasets(d=8.5e-10)
    result = fit_relaxation_global(datasets, model="stejskal_tanner", shared=["D"])
    assert result["shared"]["D"] == pytest.approx(8.5e-10, rel=0.02)
    assert result["shared_errors"]["D"] > 0


def test_every_curve_reports_the_same_shared_value():
    # The point of a joint fit: the peaks were never allowed to
    # disagree, and the per-curve results say so.
    _b, datasets = _shared_d_datasets()
    result = fit_relaxation_global(datasets, model="stejskal_tanner", shared=["D"])
    values = {entry["params"]["D"] for entry in result["per_dataset"]}
    assert len(values) == 1
    assert all(entry["shared"] == ["D"] for entry in result["per_dataset"])


def test_unshared_parameters_stay_free_per_curve():
    _b, datasets = _shared_d_datasets()
    result = fit_relaxation_global(datasets, model="stejskal_tanner", shared=["D"])
    amplitudes = [entry["params"]["I0"] for entry in result["per_dataset"]]
    assert amplitudes[0] == pytest.approx(10.0, rel=0.05)
    assert amplitudes[1] == pytest.approx(4.0, rel=0.05)
    assert amplitudes[2] == pytest.approx(1.5, rel=0.1)


def test_per_dataset_results_are_shaped_like_ordinary_fit_results():
    # So the Fit tab's plot, the peak spinner and the CSV export work
    # on a joint fit unchanged.
    _b, datasets = _shared_d_datasets()
    result = fit_relaxation_global(datasets, model="stejskal_tanner", shared=["D"])
    entry = result["per_dataset"][0]
    for key in ("model", "param_names", "params", "errors", "t", "y", "y_fit",
                 "fit_quality", "uncertainty", "peak_label", "rate_param"):
        assert key in entry
    assert entry["y_fit"].shape == entry["y"].shape


def test_global_fit_beats_averaging_separate_fits_on_precision():
    # Fitting separately and averaging throws away the constraint that
    # the peaks share a D, and reports a looser number for it.
    _b, datasets = _shared_d_datasets(noise=0.03, seed=9)
    joint = fit_relaxation_global(datasets, model="stejskal_tanner", shared=["D"])
    separate = [fit_relaxation(d["t"], d["y"], model="stejskal_tanner") for d in datasets]
    worst_separate = max(fit["errors"]["D"] for fit in separate)
    assert joint["shared_errors"]["D"] < worst_separate


def test_global_fit_can_share_something_other_than_the_rate_constant():
    # One imperfect inversion pulse across every peak, each keeping its
    # own M0 and T1.
    t = np.linspace(0.0, 2.0, 20)
    datasets = []
    for amp, t1 in [(5.0, 0.3), (2.0, 0.8)]:
        datasets.append({"t": t, "y": amp * (1.0 - 1.7 * np.exp(-t / t1))})
    result = fit_relaxation_global(datasets, model="inversion_recovery", shared=["C"])
    assert result["shared"]["C"] == pytest.approx(1.7, rel=0.05)
    t1_values = [entry["params"]["T1"] for entry in result["per_dataset"]]
    assert t1_values[0] == pytest.approx(0.3, rel=0.05)
    assert t1_values[1] == pytest.approx(0.8, rel=0.05)


def test_global_fit_with_nothing_shared_matches_independent_fits():
    _b, datasets = _shared_d_datasets(noise=0.0)
    joint = fit_relaxation_global(datasets, model="stejskal_tanner", shared=[])
    for entry, data in zip(joint["per_dataset"], datasets):
        alone = fit_relaxation(data["t"], data["y"], model="stejskal_tanner")
        assert entry["params"]["D"] == pytest.approx(alone["params"]["D"], rel=1e-3)


def test_global_fit_honours_fixed_parameters():
    _b, datasets = _shared_d_datasets()
    result = fit_relaxation_global(datasets, model="stejskal_tanner", shared=["D"],
                                    fixed={"I0": 10.0})
    assert all(entry["params"]["I0"] == 10.0 for entry in result["per_dataset"])
    assert all(np.isnan(entry["errors"]["I0"]) for entry in result["per_dataset"])


def test_global_fit_accepts_tuples_and_uneven_series():
    # The datasets need not share an x-grid or even a point count --
    # separately-loaded 1D spectra rarely do.
    a = (np.linspace(0, 3e9, 12), 5.0 * np.exp(-1e-9 * np.linspace(0, 3e9, 12)))
    b = (np.linspace(0, 2e9, 20), 2.0 * np.exp(-1e-9 * np.linspace(0, 2e9, 20)))
    result = fit_relaxation_global([a, b], model="stejskal_tanner", shared=["D"])
    assert result["shared"]["D"] == pytest.approx(1e-9, rel=1e-3)
    assert result["labels"] == ["1", "2"]


def test_global_fit_uses_point_errors_when_every_curve_has_them():
    _b, datasets = _shared_d_datasets(noise=0.02, seed=6)
    for data in datasets:
        data["y_err"] = np.full(data["t"].size, 0.05)
    result = fit_relaxation_global(datasets, model="stejskal_tanner", shared=["D"])
    assert result["uncertainty"]["kind"] == "measured"


def test_global_fit_rejects_an_unknown_shared_name_and_empty_input():
    _b, datasets = _shared_d_datasets()
    with pytest.raises(ValueError, match="not parameters of model"):
        fit_relaxation_global(datasets, model="stejskal_tanner", shared=["T1"])
    with pytest.raises(ValueError, match="at least one dataset"):
        fit_relaxation_global([], model="stejskal_tanner", shared=["D"])


def test_global_fit_rejects_a_fully_fixed_model():
    _b, datasets = _shared_d_datasets()
    with pytest.raises(ValueError, match="nothing to fit"):
        fit_relaxation_global(datasets, model="stejskal_tanner", shared=["D"],
                               fixed={"I0": 1.0, "D": 1e-9})


def test_global_report_leads_with_the_shared_parameters():
    _b, datasets = _shared_d_datasets()
    result = fit_relaxation_global(datasets, model="stejskal_tanner", shared=["D"])
    report = format_global_report(result)
    assert "Shared across every curve" in report
    assert "3 curve(s)" in report
    assert "peak A" in report and "peak C" in report
    assert "Uncertainty" in report and "R²" in report
    # The shared value is above the per-curve block, since it is the
    # answer the joint fit exists to produce.
    assert report.index("Shared across every curve") < report.index("Per curve")


def test_global_report_says_so_when_nothing_is_shared():
    _b, datasets = _shared_d_datasets()
    result = fit_relaxation_global(datasets, model="stejskal_tanner", shared=[])
    assert "No shared parameters" in format_global_report(result)
