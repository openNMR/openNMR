"""SCORE -- multivariate resolution of a series into component spectra."""

import numpy as np
import pytest

from opennmr.processing import score as score_mod
from opennmr.processing.fitting import lorentzian


PPM = np.linspace(0.0, 10.0, 1200)


def _species(*peaks):
    """A pure component spectrum from ``(shift, amplitude)`` pairs."""
    return sum(lorentzian(PPM, shift, 0.05, amplitude) for shift, amplitude in peaks)


def _mixture(b, components, noise=2e-3, seed=0):
    """``[(spectrum, decay_constant), ...]`` -> the (levels x points)
    data matrix a diffusion series would produce."""
    matrix = sum(np.outer(np.exp(-b * constant), spectrum) for spectrum, constant in components)
    if noise:
        matrix = matrix + np.random.default_rng(seed).normal(0, noise, matrix.shape)
    return matrix


#: Two species sharing a peak at 5.0 ppm -- the case the whole method
#: exists for, since a per-peak fit of that peak measures the sum of two
#: decays and cannot separate them.
SLOW = _species((5.0, 0.9), (6.5, 1.1))
FAST = _species((2.0, 1.0), (5.0, 0.8), (8.2, 0.5))
D_SLOW, D_FAST = 2.5e-10, 1.8e-09


def _overlapping_series(n_levels=20, noise=2e-3, seed=0):
    b = np.linspace(0.0, 3e9, n_levels)
    return b, _mixture(b, [(FAST, D_FAST), (SLOW, D_SLOW)], noise=noise, seed=seed)


def _height(spectrum, shift):
    return spectrum[int(np.argmin(np.abs(PPM - shift)))]


# ============================================================
# Rank suggestion
# ============================================================

def test_svd_suggests_the_true_component_count():
    _b, matrix = _overlapping_series()
    assert score_mod.estimate_components(matrix)["n_components"] == 2


def test_svd_suggestion_is_one_for_a_single_component():
    b = np.linspace(0.0, 3e9, 16)
    assert score_mod.estimate_components(_mixture(b, [(FAST, D_FAST)]))["n_components"] == 1


def test_svd_suggestion_survives_a_change_in_total_signal():
    # A fixed fraction of the largest singular value fails here: the
    # same two components scaled differently cross a relative threshold
    # differently. The noise-floor rule does not.
    b, matrix = _overlapping_series()
    assert score_mod.estimate_components(matrix)["n_components"] == 2
    assert score_mod.estimate_components(matrix * 50.0)["n_components"] == 2


def test_singular_values_are_normalised_and_ordered():
    _b, matrix = _overlapping_series()
    values = score_mod.estimate_components(matrix)["singular_values"]
    assert values[0] == pytest.approx(1.0)
    assert np.all(np.diff(values) <= 1e-12)


def test_estimate_components_rejects_a_non_matrix():
    with pytest.raises(ValueError, match="must be 2-D"):
        score_mod.estimate_components(np.ones(10))


# ============================================================
# The resolution itself
# ============================================================

def test_score_recovers_both_diffusion_coefficients():
    b, matrix = _overlapping_series()
    result = score_mod.score(b, matrix, n_components=2)
    assert result["D"][0] == pytest.approx(D_SLOW, rel=0.05)
    assert result["D"][1] == pytest.approx(D_FAST, rel=0.05)
    assert result["r_squared"] > 0.99


def test_score_separates_spectra_of_overlapping_species():
    # The claim that distinguishes SCORE from every per-peak fit: the
    # 5.0 ppm peak belongs to *both* species, and each resolved
    # component must contain its own share of it while containing none
    # of the other species' exclusive peaks.
    b, matrix = _overlapping_series()
    result = score_mod.score(b, matrix, n_components=2)
    slow_spectrum, fast_spectrum = result["spectra"]

    slow_spectrum = slow_spectrum / slow_spectrum.max()
    fast_spectrum = fast_spectrum / fast_spectrum.max()

    # 6.5 ppm is the slow species alone; 2.0 and 8.2 the fast one alone.
    assert _height(slow_spectrum, 6.5) > 0.8
    assert _height(slow_spectrum, 2.0) < 0.05
    assert _height(slow_spectrum, 8.2) < 0.05

    assert _height(fast_spectrum, 2.0) > 0.8
    assert _height(fast_spectrum, 6.5) < 0.05

    # And the shared peak is present in both.
    assert _height(slow_spectrum, 5.0) > 0.3
    assert _height(fast_spectrum, 5.0) > 0.3


def test_resolved_spectra_correlate_with_the_true_pure_spectra():
    b, matrix = _overlapping_series()
    result = score_mod.score(b, matrix, n_components=2)
    assert np.corrcoef(result["spectra"][0], SLOW)[0, 1] > 0.99
    assert np.corrcoef(result["spectra"][1], FAST)[0, 1] > 0.99


def test_components_come_back_slowest_first():
    # Stable ordering regardless of where the optimiser placed each
    # one, so component 1 always means the same thing across runs.
    b, matrix = _overlapping_series()
    for seed in (0, 1, 2):
        result = score_mod.score(b, matrix, n_components=2,
                                  initial=[5e-9, 1e-11] if seed else None)
        assert np.all(np.diff(result["constants"]) > 0)


def test_score_is_insensitive_to_the_starting_guess():
    # The headline claim of the method over CORE. Decades away in
    # either direction still converges to the same answer.
    b, matrix = _overlapping_series()
    baseline = score_mod.score(b, matrix, n_components=2)
    for guess in ([1e-8, 1e-9], [1e-9, 1e-12], [3e-10, 2e-9]):
        result = score_mod.score(b, matrix, n_components=2, initial=guess)
        np.testing.assert_allclose(result["constants"], baseline["constants"], rtol=0.05)


def test_score_is_speedy():
    # "Speedy" is the S: the nonlinear search runs over only the k
    # decay constants, whatever the point count, so it converges in a
    # handful of evaluations rather than the hundreds a full nonlinear
    # fit of the spectra would need.
    b, matrix = _overlapping_series()
    result = score_mod.score(b, matrix, n_components=2)
    assert result["success"]
    assert result["n_evaluations"] < 100


def test_fractions_sum_to_one_and_reflect_the_mixture():
    b, matrix = _overlapping_series()
    result = score_mod.score(b, matrix, n_components=2)
    assert float(np.sum(result["fractions"])) == pytest.approx(1.0)
    assert np.all(result["fractions"] > 0.2)


def test_non_negativity_keeps_the_spectra_non_negative():
    b, matrix = _overlapping_series(noise=8e-3, seed=5)
    constrained = score_mod.score(b, matrix, n_components=2, non_negative=True)
    assert np.all(constrained["spectra"] >= 0)
    # Unconstrained, noise pushes some baseline points negative -- which
    # is exactly the instability the constraint exists to remove.
    unconstrained = score_mod.score(b, matrix, n_components=2, non_negative=False)
    assert np.any(unconstrained["spectra"] < 0)


def test_enforcing_non_negativity_throughout_agrees_with_doing_it_once():
    # The documented reason the fast path is the default.
    b, matrix = _overlapping_series()
    quick = score_mod.score(b, matrix, n_components=2)
    strict = score_mod.score(b, matrix, n_components=2, non_negative_during_fit=True)
    np.testing.assert_allclose(strict["constants"], quick["constants"], rtol=0.02)


def test_three_components_are_resolved():
    third = _species((1.0, 0.7), (9.0, 0.6))
    b = np.linspace(0.0, 4e9, 28)
    matrix = _mixture(b, [(FAST, D_FAST), (SLOW, D_SLOW), (third, 1.0e-10)], noise=1e-3)
    result = score_mod.score(b, matrix, n_components=3)
    for got, true in zip(result["constants"], [1.0e-10, D_SLOW, D_FAST]):
        assert got == pytest.approx(true, rel=0.1)


def test_a_barely_decaying_component_is_only_approximately_recovered():
    # A property of the experiment, not of the algorithm, and worth
    # pinning down: at b_max = 4e9 a component with D = 2e-11 loses
    # only ~8% of its intensity across the entire series, so it is
    # nearly indistinguishable from a constant and its D is
    # correspondingly poorly determined. The fix is a longer diffusion
    # time or stronger gradients, not a better fit.
    third = _species((1.0, 0.7), (9.0, 0.6))
    b = np.linspace(0.0, 4e9, 28)
    assert 1.0 - np.exp(-2.0e-11 * b.max()) < 0.1   # under 10% attenuation
    matrix = _mixture(b, [(FAST, D_FAST), (SLOW, D_SLOW), (third, 2.0e-11)], noise=1e-3)
    result = score_mod.score(b, matrix, n_components=3)
    # Right order of magnitude, but nothing like the precision of the
    # two well-sampled components alongside it.
    assert result["constants"][0] == pytest.approx(2.0e-11, rel=0.5)
    assert result["constants"][1] == pytest.approx(D_SLOW, rel=0.1)
    assert result["constants"][2] == pytest.approx(D_FAST, rel=0.1)


def test_relaxation_kind_reports_a_time_constant():
    t = np.linspace(0.0, 0.5, 20)
    matrix = sum(np.outer(np.exp(-t / tau), spectrum)
                 for spectrum, tau in [(FAST, 0.05), (SLOW, 0.30)])
    result = score_mod.score(t, matrix, n_components=2, kind="relaxation")
    # Sorted slowest-decaying first: the smallest *rate* is the longest T.
    assert result["T"][0] == pytest.approx(0.30, rel=0.05)
    assert result["T"][1] == pytest.approx(0.05, rel=0.05)


def test_the_full_point_grid_is_reconstructed_even_though_fewer_were_fitted():
    # The threshold protects the decay constants; a component spectrum
    # with holes wherever signal was weak would be useless to look at.
    b, matrix = _overlapping_series()
    result = score_mod.score(b, matrix, n_components=2, threshold=0.2)
    assert len(result["columns"]) < PPM.size
    assert result["spectra"].shape == (2, PPM.size)
    assert result["fit"].shape == matrix.shape


# ============================================================
# Input validation
# ============================================================

def test_score_rejects_mismatched_rows_and_x_values():
    b, matrix = _overlapping_series(n_levels=12)
    with pytest.raises(ValueError, match="one per row"):
        score_mod.score(b[:5], matrix, n_components=2)


def test_score_rejects_an_underdetermined_model():
    # More components than gradient levels is not a hard fit, it is an
    # ill-posed one.
    b, matrix = _overlapping_series(n_levels=3)
    with pytest.raises(ValueError, match="underdetermined"):
        score_mod.score(b, matrix, n_components=4)


def test_score_rejects_an_unknown_kind_and_a_bad_component_count():
    b, matrix = _overlapping_series()
    with pytest.raises(ValueError, match="Unknown kind"):
        score_mod.score(b, matrix, n_components=2, kind="magic")
    with pytest.raises(ValueError, match="n_components must be"):
        score_mod.score(b, matrix, n_components=0)


def test_score_rejects_an_all_zero_matrix():
    b = np.linspace(0.0, 3e9, 10)
    with pytest.raises(ValueError, match="entirely zero"):
        score_mod.score(b, np.zeros((10, 50)), n_components=2)


def test_initial_guesses_must_match_the_component_count_and_be_positive():
    b, matrix = _overlapping_series()
    with pytest.raises(ValueError, match="entries for 2 component"):
        score_mod.score(b, matrix, n_components=2, initial=[1e-9])
    with pytest.raises(ValueError, match="must be positive"):
        score_mod.score(b, matrix, n_components=2, initial=[1e-9, 0.0])


def test_a_threshold_that_keeps_nothing_says_so():
    b, matrix = _overlapping_series()
    with pytest.raises(ValueError, match="lower the threshold"):
        score_mod.score(b, matrix, n_components=2, threshold=1.5)


# ============================================================
# Reporting
# ============================================================

def test_report_states_the_components_and_the_caveat():
    b, matrix = _overlapping_series()
    report = score_mod.format_score_report(score_mod.score(b, matrix, n_components=2))
    assert "SCORE — 2 components" in report
    assert "m^2/s" in report and "R²" in report
    # The caveat is the point: k was an input.
    assert "input, not a result" in report


def test_report_uses_time_constants_for_a_relaxation_resolution():
    t = np.linspace(0.0, 0.5, 20)
    matrix = sum(np.outer(np.exp(-t / tau), spectrum)
                 for spectrum, tau in [(FAST, 0.05), (SLOW, 0.30)])
    report = score_mod.format_score_report(
        score_mod.score(t, matrix, n_components=2, kind="relaxation"))
    assert "ms" in report and "m^2/s" not in report


# ============================================================
# OUTSCORE -- cross-talk minimisation
# ============================================================

#: Two species with **distinct** peaks -- no shared resonances -- which
#: is the case OUTSCORE is for: similar in *size*, not in spectrum.
DISTINCT_A = _species((1.2, 1.0), (3.4, 0.7), (7.1, 0.5))
DISTINCT_B = _species((2.3, 0.8), (5.6, 1.0), (8.4, 0.6))


def _similar_sized_series(ratio, n_levels=24, noise=1.5e-3, seed=0):
    """Two distinct-peaked species whose diffusion coefficients differ
    by ``ratio`` -- the regime where SCORE's error surface goes flat."""
    slow = 1.0e-9
    fast = slow * ratio
    b = np.linspace(0.0, 3e9, n_levels)
    matrix = _mixture(b, [(DISTINCT_A, fast), (DISTINCT_B, slow)], noise=noise, seed=seed)
    return b, matrix, slow, fast


def _worst_spectral_correlation(result):
    return min(abs(np.corrcoef(result["spectra"][0], DISTINCT_B)[0, 1]),
               abs(np.corrcoef(result["spectra"][1], DISTINCT_A)[0, 1]))


def test_crosstalk_is_zero_for_non_overlapping_spectra():
    apart = np.array([_species((1.0, 1.0)), _species((9.0, 1.0))])
    assert score_mod.crosstalk(apart) < 0.01
    # And maximal for a spectrum against itself.
    same = np.array([DISTINCT_A, DISTINCT_A])
    assert score_mod.crosstalk(same) == pytest.approx(1.0)


def test_crosstalk_normalisation_is_scale_invariant():
    # The bare dot product is degenerate: a component shrinking toward
    # zero would score perfectly for having resolved nothing.
    pair = np.array([DISTINCT_A, DISTINCT_B])
    scaled = np.array([DISTINCT_A * 1e-6, DISTINCT_B * 1e3])
    assert score_mod.crosstalk(pair) == pytest.approx(score_mod.crosstalk(scaled))


def test_crosstalk_of_a_vanished_component_is_the_worst_value():
    pair = np.array([DISTINCT_A, np.zeros_like(DISTINCT_B)])
    assert score_mod.crosstalk(pair) == pytest.approx(1.0)


def test_outscore_beats_score_on_similar_sized_species():
    # The headline claim: separation down to a few percent difference in
    # D, where SCORE needs ~30%.
    b, matrix, slow, fast = _similar_sized_series(1.03)
    by_residual = score_mod.score(b, matrix, n_components=2, method="score")
    by_crosstalk = score_mod.score(b, matrix, n_components=2, method="outscore")

    assert _worst_spectral_correlation(by_crosstalk) > _worst_spectral_correlation(by_residual)
    assert _worst_spectral_correlation(by_crosstalk) > 0.95
    assert by_crosstalk["constants"][0] == pytest.approx(slow, rel=0.02)
    assert by_crosstalk["constants"][1] == pytest.approx(fast, rel=0.02)


@pytest.mark.parametrize("ratio", [1.10, 1.05, 1.03])
def test_outscore_recovers_similar_diffusion_coefficients(ratio):
    b, matrix, slow, fast = _similar_sized_series(ratio)
    result = score_mod.score(b, matrix, n_components=2, method="outscore")
    assert result["constants"][0] == pytest.approx(slow, rel=0.03)
    assert result["constants"][1] == pytest.approx(fast, rel=0.03)


def test_outscore_achieves_lower_crosstalk_than_score():
    b, matrix, _slow, _fast = _similar_sized_series(1.05)
    by_residual = score_mod.score(b, matrix, n_components=2, method="score")
    by_crosstalk = score_mod.score(b, matrix, n_components=2, method="outscore")
    assert by_crosstalk["crosstalk"] < by_residual["crosstalk"]


def test_outscore_seeds_itself_from_a_score_run():
    # Required, not an optimisation: OUTSCORE's sharp minimum is what
    # gives it its resolving power and also what makes it unfindable
    # from a cold start.
    b, matrix, _slow, _fast = _similar_sized_series(1.05)
    result = score_mod.score(b, matrix, n_components=2, method="outscore")
    assert result["seed"] is not None
    assert result["seed"]["method"] == "score"
    # An explicit starting point skips the seeding.
    supplied = score_mod.score(b, matrix, n_components=2, method="outscore",
                                initial=[1.0e-9, 1.05e-9])
    assert supplied["seed"] is None


def test_score_does_not_seed_itself():
    b, matrix = _overlapping_series()
    assert score_mod.score(b, matrix, n_components=2)["seed"] is None


def test_both_methods_report_both_numbers():
    b, matrix = _overlapping_series()
    for method in ("score", "outscore"):
        result = score_mod.score(b, matrix, n_components=2, method=method)
        assert np.isfinite(result["r_squared"])
        assert np.isfinite(result["crosstalk"])
        assert result["method"] == method


def test_outscore_needs_at_least_two_components():
    b, matrix = _overlapping_series()
    with pytest.raises(ValueError, match="at least 2 components"):
        score_mod.score(b, matrix, n_components=1, method="outscore")


def test_unknown_method_is_rejected():
    b, matrix = _overlapping_series()
    with pytest.raises(ValueError, match="Unknown method"):
        score_mod.score(b, matrix, n_components=2, method="bestscore")


# ============================================================
# Diagnostics -- what R² cannot tell you
# ============================================================

def test_outscore_warns_when_components_genuinely_share_resonances():
    # SLOW and FAST both have a peak at 5.0 ppm, so their true
    # cross-talk is high and minimising it is chasing something
    # impossible. The run still happens; it says so.
    b, matrix = _overlapping_series()
    result = score_mod.score(b, matrix, n_components=2, method="outscore")
    assert result["warnings"]
    assert any("share resonances" in w for w in result["warnings"])
    assert any("Prefer SCORE" in w for w in result["warnings"])


def test_a_clean_separation_raises_no_warnings():
    b, matrix, _slow, _fast = _similar_sized_series(1.05)
    assert score_mod.score(b, matrix, n_components=2, method="outscore")["warnings"] == []


def test_r_squared_alone_cannot_distinguish_a_wrong_answer():
    # The measurement the whole diagnostic block exists for, pinned
    # down: on the shared-resonance case the two methods land on
    # materially different answers while their fit statistics are
    # indistinguishable.
    b, matrix = _overlapping_series()
    by_residual = score_mod.score(b, matrix, n_components=2, method="score")
    by_crosstalk = score_mod.score(b, matrix, n_components=2, method="outscore")
    assert abs(by_crosstalk["r_squared"] - by_residual["r_squared"]) < 0.001
    # Yet cross-talk separates them clearly, which is why it is reported.
    assert abs(by_crosstalk["crosstalk"] - by_residual["crosstalk"]) > 0.0


def test_report_names_which_objective_the_run_optimised():
    b, matrix, _slow, _fast = _similar_sized_series(1.05)
    report = score_mod.format_score_report(
        score_mod.score(b, matrix, n_components=2, method="outscore"))
    assert "OUTSCORE" in report
    assert "cross-talk" in report
    assert "not as independent corroboration" in report


def test_report_includes_any_warnings():
    b, matrix = _overlapping_series()
    report = score_mod.format_score_report(
        score_mod.score(b, matrix, n_components=2, method="outscore"))
    assert "⚠" in report


# ============================================================
# Export
# ============================================================

def test_components_csv_carries_constants_shares_and_spectra():
    import csv as csv_module
    from io import StringIO

    b, matrix = _overlapping_series()
    result = score_mod.score(b, matrix, n_components=2)
    rows = list(csv_module.reader(StringIO(score_mod.components_to_csv(result, PPM))))

    assert "SCORE" in rows[0][0]
    header = next(r for r in rows if r and r[0] == "Component")
    assert header[1] == "Decay constant" and "Share" in header

    # One data row per component, carrying its own constant.
    component_rows = [r for r in rows if r and r[0] in ("1", "2")]
    assert len(component_rows) == 2
    assert float(component_rows[0][1]) == pytest.approx(result["constants"][0])
    assert component_rows[0][2] == "m^2/s"

    # And the spectra themselves, one column per component.
    spectra_header = rows.index(["ppm", "component_1", "component_2"])
    first_point = rows[spectra_header + 1]
    assert float(first_point[0]) == pytest.approx(PPM[0])
    assert len(rows) - spectra_header - 1 == PPM.size


def test_components_csv_names_the_method_and_its_reference():
    b, matrix, _slow, _fast = _similar_sized_series(1.05)
    text = score_mod.components_to_csv(
        score_mod.score(b, matrix, n_components=2, method="outscore"), PPM)
    assert "OUTSCORE" in text
    assert "Colbourne" in text  # the reference travels with the data


def test_components_csv_embeds_any_warning():
    b, matrix = _overlapping_series()
    text = score_mod.components_to_csv(
        score_mod.score(b, matrix, n_components=2, method="outscore"), PPM)
    assert "WARNING" in text


def test_components_csv_rejects_a_mismatched_axis():
    b, matrix = _overlapping_series()
    result = score_mod.score(b, matrix, n_components=2)
    with pytest.raises(ValueError, match="shifts against"):
        score_mod.components_to_csv(result, PPM[:10])
