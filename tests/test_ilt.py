"""Inverse Laplace transform and the DOSY map built from it."""

import numpy as np
import pytest

from opennmr.processing import ilt as ilt_mod


def _decay(b, components):
    """``[(weight, D), ...]`` -> the sampled decay."""
    return sum(weight * np.exp(-d * b) for weight, d in components)


def _peak_clusters(result, threshold=0.02, ratio=1.5):
    """Distinct populations in a distribution, as (centre, weight)
    pairs -- adjacent grid points closer than ``ratio`` apart in D
    belong to the same peak, since the grid is far finer than the ILT
    can actually resolve."""
    mask = result["P"] > threshold * result["P"].max()
    d_values, weights = result["D"][mask], result["P"][mask]
    clusters, current_d, current_w = [], [d_values[0]], [weights[0]]
    for d, w in zip(d_values[1:], weights[1:]):
        if d / current_d[-1] < ratio:
            current_d.append(d)
            current_w.append(w)
        else:
            clusters.append((float(np.average(current_d, weights=current_w)), float(sum(current_w))))
            current_d, current_w = [d], [w]
    clusters.append((float(np.average(current_d, weights=current_w)), float(sum(current_w))))
    return clusters


# ============================================================
# The grid and the kernel
# ============================================================

def test_log_grid_spans_the_requested_decades():
    grid = ilt_mod.log_grid(1e-12, 1e-9, n=4)
    np.testing.assert_allclose(grid, [1e-12, 1e-11, 1e-10, 1e-9])


def test_log_grid_rejects_a_non_positive_or_inverted_range():
    with pytest.raises(ValueError, match="0 < low < high"):
        ilt_mod.log_grid(0.0, 1e-9)
    with pytest.raises(ValueError, match="0 < low < high"):
        ilt_mod.log_grid(1e-9, 1e-12)


def test_kernel_is_the_exponential_decay_matrix():
    b = np.array([0.0, 1e9])
    d = np.array([1e-9, 2e-9])
    k = ilt_mod.kernel(b, d)
    np.testing.assert_allclose(k[0], [1.0, 1.0])
    np.testing.assert_allclose(k[1], [np.exp(-1.0), np.exp(-2.0)])


# ============================================================
# One distribution
# ============================================================

def test_ilt_recovers_a_single_diffusion_coefficient():
    b = np.linspace(0, 4e9, 32)
    result = ilt_mod.ilt(b, _decay(b, [(1.0, 1e-9)]), alpha=1e-3)
    assert result["D_mode"] == pytest.approx(1e-9, rel=0.1)
    assert result["D_mean"] == pytest.approx(1e-9, rel=0.1)
    # A monodisperse sample should come back narrow, not smeared over
    # a decade -- the width here is the regularisation's own floor.
    assert result["log10_width"] < 0.25


def test_ilt_separates_two_populations_and_gets_their_weights():
    b = np.linspace(0, 4e9, 32)
    result = ilt_mod.ilt(b, _decay(b, [(0.6, 2e-9), (0.4, 5e-11)]), alpha=1e-4, n_grid=192)
    clusters = _peak_clusters(result)
    assert len(clusters) == 2
    slow, fast = clusters  # the grid ascends in D
    assert slow[0] == pytest.approx(5e-11, rel=0.15)
    assert fast[0] == pytest.approx(2e-9, rel=0.15)
    assert slow[1] == pytest.approx(0.4, abs=0.05)
    assert fast[1] == pytest.approx(0.6, abs=0.05)


def test_ilt_distribution_is_non_negative_everywhere():
    # Non-negativity is the single most stabilising constraint here,
    # and it is imposed rather than hoped for -- a negative population
    # is not a physical answer.
    b = np.linspace(0, 3e9, 24)
    rng = np.random.default_rng(11)
    y = _decay(b, [(1.0, 8e-10)]) + rng.normal(0, 0.02, b.size)
    result = ilt_mod.ilt(b, y, alpha=1e-2)
    assert np.all(result["P"] >= 0)


def test_ilt_distribution_sums_to_one_when_normalised():
    b = np.linspace(0, 3e9, 24)
    result = ilt_mod.ilt(b, _decay(b, [(1.0, 1e-9)]), alpha=1e-3)
    assert float(np.sum(result["P"])) == pytest.approx(1.0)


def test_ilt_is_insensitive_to_the_intensity_scale():
    b = np.linspace(0, 3e9, 24)
    y = _decay(b, [(1.0, 1e-9)])
    a = ilt_mod.ilt(b, y, alpha=1e-3)
    scaled = ilt_mod.ilt(b, y * 5e6, alpha=1e-3)
    np.testing.assert_allclose(a["P"], scaled["P"], atol=1e-12)


def test_more_regularisation_broadens_the_distribution():
    # The behaviour the module docstring warns about, asserted rather
    # than just described: peak width is set by alpha, so a narrow ILT
    # peak is not by itself evidence of a narrow distribution.
    b = np.linspace(0, 4e9, 32)
    y = _decay(b, [(1.0, 1e-9)])
    sharp = ilt_mod.ilt(b, y, alpha=1e-5, n_grid=192)
    smooth = ilt_mod.ilt(b, y, alpha=1e-1, n_grid=192)
    assert smooth["log10_width"] > sharp["log10_width"]


def test_ilt_rejects_mismatched_or_too_short_input():
    with pytest.raises(ValueError, match="against"):
        ilt_mod.ilt([1.0, 2.0, 3.0], [1.0, 0.5])
    with pytest.raises(ValueError, match="at least 3 points"):
        ilt_mod.ilt([1.0, 2.0], [1.0, 0.5])
    with pytest.raises(ValueError, match="alpha must be"):
        ilt_mod.ilt([1.0, 2.0, 3.0], [1.0, 0.5, 0.2], alpha=-1.0)


def test_distribution_moments_uses_a_geometric_mean():
    # Equal weight at 1e-11 and 1e-9 has geometric mean 1e-10. An
    # arithmetic mean would report 5.05e-10 -- dominated by the larger
    # value and describing neither population.
    d = np.array([1e-11, 1e-9])
    moments = ilt_mod.distribution_moments(d, np.array([0.5, 0.5]))
    assert moments["D_mean"] == pytest.approx(1e-10, rel=1e-9)
    assert moments["log10_width"] == pytest.approx(1.0)


def test_distribution_moments_of_an_empty_solution_is_not_a_crash():
    moments = ilt_mod.distribution_moments(np.array([1e-10, 1e-9]), np.zeros(2))
    assert np.isnan(moments["D_mean"])


# ============================================================
# Choosing alpha
# ============================================================

def test_l_curve_returns_an_alpha_from_the_range_it_searched():
    b = np.linspace(0, 4e9, 32)
    rng = np.random.default_rng(5)
    y = _decay(b, [(0.6, 2e-9), (0.4, 5e-11)]) + rng.normal(0, 0.005, b.size)
    result = ilt_mod.l_curve_alpha(b, y)
    assert result["alpha"] in set(result["alphas"])
    assert result["residual_norms"].size == result["alphas"].size
    assert result["solution_norms"].size == result["alphas"].size


def test_l_curve_residual_grows_and_solution_norm_shrinks_with_alpha():
    # The trade-off the corner sits on: more regularisation always
    # means a smoother solution and a worse fit.
    b = np.linspace(0, 4e9, 32)
    result = ilt_mod.l_curve_alpha(b, _decay(b, [(1.0, 1e-9)]),
                                   alphas=np.logspace(-5, 0, 8))
    assert result["residual_norms"][-1] > result["residual_norms"][0]
    assert result["solution_norms"][-1] <= result["solution_norms"][0]


# ============================================================
# The DOSY map
# ============================================================

def _dosy_series(b, ppm, components, noise=1e-3, seed=2):
    """A synthetic PFG series: each component is a Lorentzian at its
    own shift, decaying with its own D."""
    data = np.zeros((b.size, ppm.size))
    for shift, d, amplitude in components:
        line = amplitude / (1.0 + ((ppm - shift) / 0.06) ** 2)
        data += np.outer(np.exp(-d * b), line)
    if noise:
        data = data + np.random.default_rng(seed).normal(0, noise, data.shape)
    return data


def test_dosy_map_places_each_peak_at_its_own_diffusion_coefficient():
    b = np.linspace(0, 4e9, 32)
    ppm = np.linspace(0, 10, 400)
    components = [(2.0, 1.5e-9, 1.0), (5.0, 1.5e-9, 0.8), (7.5, 1.2e-10, 1.2)]
    dosy = ilt_mod.dosy_map(b, _dosy_series(b, ppm, components), ppm,
                            alpha=1e-3, threshold=0.05)
    for shift, true_d, _amplitude in components:
        column = int(np.argmin(np.abs(ppm - shift)))
        found = dosy["D"][int(np.argmax(dosy["Z"][:, column]))]
        assert found == pytest.approx(true_d, rel=0.2)


def test_dosy_map_leaves_baseline_columns_empty():
    # An ILT of a noise-only column returns whatever the regularisation
    # likes best, which would paint a band of meaningless "population"
    # clean across the map -- the threshold is what stops that.
    b = np.linspace(0, 4e9, 24)
    ppm = np.linspace(0, 10, 200)
    dosy = ilt_mod.dosy_map(b, _dosy_series(b, ppm, [(5.0, 1e-9, 1.0)]), ppm,
                            alpha=1e-3, threshold=0.1)
    baseline = int(np.argmin(np.abs(ppm - 1.0)))
    assert np.all(dosy["Z"][:, baseline] == 0)
    assert dosy["columns"].size < ppm.size


def test_dosy_map_shape_and_projection():
    b = np.linspace(0, 4e9, 16)
    ppm = np.linspace(0, 10, 128)
    data = _dosy_series(b, ppm, [(5.0, 1e-9, 1.0)], noise=0.0)
    dosy = ilt_mod.dosy_map(b, data, ppm, alpha=1e-3, n_grid=64)
    assert dosy["Z"].shape == (64, 128)
    # The projection is the lowest-gradient row, i.e. the ordinary
    # spectrum -- what the strip along the top of a DOSY plot shows.
    np.testing.assert_allclose(dosy["projection"], data[0])


def test_dosy_map_sorts_rows_by_b_value():
    # A series handed over in a scrambled gradient order must not
    # silently produce a scrambled decay.
    b = np.linspace(0, 4e9, 16)
    ppm = np.linspace(0, 10, 64)
    data = _dosy_series(b, ppm, [(5.0, 1e-9, 1.0)], noise=0.0)
    shuffle = np.argsort(np.random.default_rng(0).random(b.size))
    ordered = ilt_mod.dosy_map(b, data, ppm, alpha=1e-3, n_grid=48)
    scrambled = ilt_mod.dosy_map(b[shuffle], data[shuffle], ppm, alpha=1e-3, n_grid=48)
    np.testing.assert_allclose(ordered["Z"], scrambled["Z"])


def test_dosy_map_auto_alpha_picks_one_value_for_the_whole_map():
    b = np.linspace(0, 4e9, 24)
    ppm = np.linspace(0, 10, 96)
    dosy = ilt_mod.dosy_map(b, _dosy_series(b, ppm, [(5.0, 1e-9, 1.0)]), ppm,
                            n_grid=48, auto_alpha=True)
    # One alpha, reported -- a per-column alpha would make neighbouring
    # shifts incomparable, since peak width down D would then vary with
    # the regularisation rather than with the sample.
    assert dosy["alpha"] > 0


def test_dosy_map_rejects_shapes_that_do_not_line_up():
    b = np.linspace(0, 1e9, 8)
    ppm = np.linspace(0, 10, 32)
    with pytest.raises(ValueError, match="one b per row"):
        ilt_mod.dosy_map(b, np.ones((4, 32)), ppm)
    with pytest.raises(ValueError, match="points per row"):
        ilt_mod.dosy_map(b, np.ones((8, 16)), ppm)
    with pytest.raises(ValueError, match="must be 2-D"):
        ilt_mod.dosy_map(b, np.ones(8), ppm)


def test_dosy_map_rejects_an_empty_reference_row():
    b = np.linspace(0, 1e9, 8)
    ppm = np.linspace(0, 10, 32)
    with pytest.raises(ValueError, match="entirely zero"):
        ilt_mod.dosy_map(b, np.zeros((8, 32)), ppm)
