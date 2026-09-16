"""PFG diffusion: b-values, gradient calibration, N-component models."""

import numpy as np
import pytest

from opennmr.processing import diffusion as dif
from opennmr.processing.relaxation import fit_relaxation


GAMMA_1H = dif.gamma_for("1H")


# ============================================================
# b-values
# ============================================================

def test_gamma_for_1h_is_in_hz_per_tesla():
    # Every formula here multiplies by 2*pi itself, so a gamma already
    # in rad/s/T would be out by 2*pi -- a 40x error in b.
    assert GAMMA_1H == pytest.approx(42.577e6, rel=1e-3)


def test_b_value_matches_the_stejskal_tanner_expression():
    b = dif.b_value(1.0, 2e-3, 100e-3, gamma=GAMMA_1H)
    expected = (2 * np.pi * GAMMA_1H * 1.0 * 2e-3) ** 2 * (100e-3 - 2e-3 / 3)
    assert b == pytest.approx(expected, rel=1e-12)


def test_b_value_scales_with_the_square_of_the_shape_factor():
    # delta_eff = delta * shape_factor, and b goes as delta_eff^2 -- so
    # assuming a perfect rectangle for Bruker's SMSQ10 overestimates b
    # by 1/0.81, i.e. 23%, landing directly on D.
    rect = dif.b_value(1.0, 2e-3, 100e-3, gamma=GAMMA_1H, shape="rect")
    smooth = dif.b_value(1.0, 2e-3, 100e-3, gamma=GAMMA_1H, shape="smsq10")
    assert smooth / rect == pytest.approx(0.9 ** 2)


def test_b_value_bipolar_subtracts_half_the_recovery_delay():
    plain = dif.b_value(1.0, 2e-3, 100e-3, gamma=GAMMA_1H, sequence="pgse")
    bipolar = dif.b_value(1.0, 2e-3, 100e-3, gamma=GAMMA_1H, sequence="bipolar", tau=4e-3)
    assert bipolar / plain == pytest.approx((100e-3 - 2e-3 / 3 - 2e-3) / (100e-3 - 2e-3 / 3))


def test_b_value_is_quadratic_in_gradient():
    gradients = np.array([0.1, 0.2, 0.4])
    b = dif.b_value(gradients, 2e-3, 100e-3, gamma=GAMMA_1H)
    assert b[1] / b[0] == pytest.approx(4.0)
    assert b[2] / b[0] == pytest.approx(16.0)


def test_b_value_accepts_a_nucleus_instead_of_a_gamma():
    assert dif.b_value(1.0, 2e-3, 0.1, nucleus="1H") == pytest.approx(
        dif.b_value(1.0, 2e-3, 0.1, gamma=GAMMA_1H))


def test_b_value_rejects_timings_that_cannot_contain_the_gradients():
    # DELTA shorter than the pulses it is meant to contain is an input
    # error, not a small b -- silently returning a negative b would put
    # a *negative* diffusion coefficient downstream.
    with pytest.raises(ValueError, match="DELTA must be long enough"):
        dif.b_value(1.0, 20e-3, 1e-3, gamma=GAMMA_1H)


def test_b_value_rejects_an_unknown_shape_or_sequence():
    with pytest.raises(ValueError, match="Unknown gradient shape"):
        dif.b_value(1.0, 2e-3, 0.1, gamma=GAMMA_1H, shape="triangular")
    with pytest.raises(ValueError, match="Unknown sequence"):
        dif.b_value(1.0, 2e-3, 0.1, gamma=GAMMA_1H, sequence="magic")


def test_b_value_needs_a_gamma_or_a_nucleus():
    with pytest.raises(ValueError, match="either gamma"):
        dif.b_value(1.0, 2e-3, 0.1)


def test_gradient_unit_conversion_round_trips():
    assert dif.g_per_cm_to_t_per_m(100.0) == pytest.approx(1.0)
    assert dif.t_per_m_to_g_per_cm(dif.g_per_cm_to_t_per_m(37.5)) == pytest.approx(37.5)


def test_b_values_from_relative_reads_percentages_as_well_as_fractions():
    as_fraction = dif.b_values_from_relative([0.1, 0.5, 1.0], 0.5, 2e-3, 0.1, nucleus="1H")
    as_percent = dif.b_values_from_relative([10.0, 50.0, 100.0], 0.5, 2e-3, 0.1, nucleus="1H")
    np.testing.assert_allclose(as_fraction, as_percent)


def test_b_values_from_relative_accepts_g_per_cm():
    in_t_per_m = dif.b_values_from_relative([0.5], 0.5, 2e-3, 0.1, nucleus="1H")
    in_g_per_cm = dif.b_values_from_relative([0.5], 50.0, 2e-3, 0.1, nucleus="1H",
                                             max_gradient_unit="G/cm")
    np.testing.assert_allclose(in_t_per_m, in_g_per_cm)


# ============================================================
# Gradient calibration
# ============================================================

def _calibration_series(g_max_t_per_m, known_d, n=16, shape="smsq10", noise=0.0, seed=0):
    relative = np.linspace(0.02, 0.95, n)
    b = dif.b_value(relative * g_max_t_per_m, 2e-3, 100e-3, gamma=GAMMA_1H, shape=shape)
    y = 1000.0 * np.exp(-known_d * b)
    if noise:
        y = y + np.random.default_rng(seed).normal(0.0, noise * 1000.0, y.size)
    return relative, y


def test_calibrate_max_gradient_recovers_a_known_maximum():
    known_d = dif.CALIBRATION_LIQUIDS["H2O_25C"]["D"]
    relative, y = _calibration_series(0.55, known_d)
    result = dif.calibrate_max_gradient(relative, y, known_d, 2e-3, 100e-3,
                                        nucleus="1H", shape="smsq10")
    assert result["max_gradient_t_per_m"] == pytest.approx(0.55, rel=1e-6)
    assert result["max_gradient_g_per_cm"] == pytest.approx(55.0, rel=1e-6)
    assert result["r_squared"] == pytest.approx(1.0, abs=1e-9)


def test_calibrate_max_gradient_survives_realistic_noise():
    known_d = 2.299e-9
    relative, y = _calibration_series(1.7, known_d, n=24, noise=0.01, seed=7)
    result = dif.calibrate_max_gradient(relative, y, known_d, 2e-3, 100e-3,
                                        nucleus="1H", shape="smsq10",
                                        g_max_guess_t_per_m=1.0)
    assert result["max_gradient_t_per_m"] == pytest.approx(1.7, rel=0.03)
    assert result["max_gradient_t_per_m_error"] > 0


def test_calibration_is_insensitive_to_the_intensity_scale():
    known_d = 2.299e-9
    relative, y = _calibration_series(0.8, known_d)
    a = dif.calibrate_max_gradient(relative, y, known_d, 2e-3, 100e-3, nucleus="1H")
    b = dif.calibrate_max_gradient(relative, y * 1e6, known_d, 2e-3, 100e-3, nucleus="1H")
    assert a["max_gradient_t_per_m"] == pytest.approx(b["max_gradient_t_per_m"], rel=1e-6)


def test_calibration_shape_factor_changes_the_answer_it_should():
    # The shape factor is not cosmetic: calibrating a smoothed-square
    # series as if it were rectangular reports G_max too small by
    # exactly 1/0.9, because b and G_max^2 trade off directly.
    known_d = 2.299e-9
    relative, y = _calibration_series(0.55, known_d, shape="smsq10")
    as_rect = dif.calibrate_max_gradient(relative, y, known_d, 2e-3, 100e-3,
                                         nucleus="1H", shape="rect")
    assert as_rect["max_gradient_t_per_m"] == pytest.approx(0.55 * 0.9, rel=1e-4)


def test_calibration_rejects_mismatched_lengths_and_too_few_points():
    with pytest.raises(ValueError, match="one intensity per gradient step"):
        dif.calibrate_max_gradient([0.1, 0.2, 0.3], [1.0, 0.5], 2.3e-9, 2e-3, 0.1)
    with pytest.raises(ValueError, match="at least 3 points"):
        dif.calibrate_max_gradient([0.1, 0.2], [1.0, 0.5], 2.3e-9, 2e-3, 0.1)


def test_known_probe_maximum_gradients_are_available():
    assert dif.PROBE_MAX_GRADIENT_G_PER_CM["diffBB"] == pytest.approx(1702.41, abs=0.01)
    assert dif.PROBE_MAX_GRADIENT_G_PER_CM["diff50"] == pytest.approx(2897.61, abs=0.01)


# ============================================================
# N-component models
# ============================================================

def test_one_component_model_names_match_the_long_standing_ones():
    # So a generalised single-component fit is a drop-in for the named
    # stejskal_tanner model, rather than renaming D out from under
    # anything that reads it.
    spec = dif.build_diffusion_model(1)
    assert spec["params"] == ["I0", "D"]
    assert spec["rate_param"] == "D"


def test_component_names_are_numbered_beyond_one_component():
    assert dif.build_diffusion_model(2)["params"] == ["I1", "D1", "I2", "D2"]
    assert dif.build_diffusion_model(2, stretched=True)["params"] == [
        "I1", "D1", "beta1", "I2", "D2", "beta2"]
    # Per-component stretching: one distribution, one discrete value.
    assert dif.build_diffusion_model(2, stretched=[False, True])["params"] == [
        "I1", "D1", "I2", "D2", "beta2"]


def test_stretched_flags_must_match_the_component_count():
    with pytest.raises(ValueError, match="one flag per component"):
        dif.build_diffusion_model(3, stretched=[True, False])


def test_multi_component_model_has_no_single_rate_param():
    # Nothing should headline one of three diffusion coefficients as
    # "the" answer.
    assert dif.build_diffusion_model(3)["rate_param"] is None


def test_every_d_is_registered_for_log_space_fitting():
    assert dif.build_diffusion_model(1)["log_params"] == ("D",)
    assert dif.build_diffusion_model(3)["log_params"] == ("D1", "D2", "D3")


@pytest.mark.parametrize("expected_D", [[3e-9, 2e-10], 1e-9, None])
def test_two_component_fit_recovers_synthetic_truth(expected_D):
    # The headline check for the generalisation, and the reason
    # LOG_FITTED_PARAMS exists: fitted linearly, this same case
    # converged to 5.5e-09/6.2e-10 against a truth of 3e-09/2e-10 even
    # when started exactly at the answer.
    b = np.linspace(0, 4e9, 24)
    y = 0.7 * np.exp(-3e-9 * b) + 0.3 * np.exp(-2e-10 * b)
    spec = dif.build_diffusion_model(2, expected_D=expected_D)
    fit = fit_relaxation(b, y, model="diffusion_n", custom_model=spec)
    fast, slow = sorted([fit["params"]["D1"], fit["params"]["D2"]], reverse=True)
    assert fast == pytest.approx(3e-9, rel=1e-3)
    assert slow == pytest.approx(2e-10, rel=1e-3)
    assert fit["fit_quality"]["r_squared"] == pytest.approx(1.0, abs=1e-9)


def test_three_component_fit_recovers_synthetic_truth():
    b = np.linspace(0, 4e9, 40)
    y = (0.5 * np.exp(-4e-9 * b) + 0.3 * np.exp(-4e-10 * b) + 0.2 * np.exp(-4e-11 * b))
    spec = dif.build_diffusion_model(3, expected_D=[5e-9, 5e-10, 5e-11])
    fit = fit_relaxation(b, y, model="diffusion_n", custom_model=spec)
    recovered = sorted([fit["params"][f"D{i}"] for i in (1, 2, 3)], reverse=True)
    for got, true in zip(recovered, [4e-9, 4e-10, 4e-11]):
        assert got == pytest.approx(true, rel=1e-2)


def test_stretched_single_component_recovers_its_exponent():
    b = np.linspace(0, 3e9, 30)
    y = np.exp(-((1e-9 * b) ** 0.7))
    spec = dif.build_diffusion_model(1, stretched=True, expected_D=1e-9)
    fit = fit_relaxation(b, y, model="diffusion_n", custom_model=spec)
    assert fit["params"]["D"] == pytest.approx(1e-9, rel=1e-2)
    assert fit["params"]["beta"] == pytest.approx(0.7, rel=1e-2)


def test_component_weights_are_fractions_of_the_total():
    weights = dif.component_weights({"I1": 3.0, "I2": 1.0}, 2)
    assert weights == pytest.approx([0.75, 0.25])
    # A single component is trivially all of it, whatever I0 happens to
    # be on the data's own arbitrary intensity scale.
    assert dif.component_weights({"I0": 1234.0}, 1) == [1.0]


def test_automatic_d_guesses_are_spread_across_decades():
    # Two components started at similar values give curve_fit nothing to
    # tell them apart, and it converges with both on the same D and the
    # amplitudes split arbitrarily.
    guesses = dif._component_d_guesses(np.linspace(0, 1e9, 10), None, 3)
    assert guesses[0] / guesses[1] == pytest.approx(10.0)
    assert guesses[1] / guesses[2] == pytest.approx(10.0)


def test_expected_d_of_the_wrong_length_is_rejected():
    with pytest.raises(ValueError, match="one per component"):
        dif.build_diffusion_model(3, expected_D=[1e-9, 2e-9])


# ============================================================
# Trapezoidal (ramped) gradients
# ============================================================

def test_trapezoid_with_no_ramp_is_exactly_the_rectangle():
    # The correction must vanish as ramp -> 0, or every shape choice
    # would silently shift b even where no ramp exists.
    rect = dif.b_value(1.0, 2e-3, 100e-3, gamma=GAMMA_1H, shape="rect")
    trap = dif.b_value(1.0, 2e-3, 100e-3, gamma=GAMMA_1H, shape="trapezoid", ramp=0.0)
    assert trap == pytest.approx(rect, rel=1e-12)


def test_trapezoid_matches_the_published_expression():
    # b = gamma^2 G^2 [delta^2 (DELTA - delta/3) + ramp^3/30 - delta*ramp^2/6]
    delta, big_delta, ramp, gradient = 2e-3, 100e-3, 0.5e-3, 0.8
    expected = (2 * np.pi * GAMMA_1H * gradient) ** 2 * (
        delta ** 2 * (big_delta - delta / 3) + ramp ** 3 / 30 - delta * ramp ** 2 / 6
    )
    got = dif.b_value(gradient, delta, big_delta, gamma=GAMMA_1H,
                      shape="trapezoid", ramp=ramp)
    assert got == pytest.approx(expected, rel=1e-12)


def test_a_ramp_reduces_b():
    # Ramping means less dephasing than a perfect rectangle of the same
    # nominal duration, so b must come down, never up.
    plain = dif.b_value(1.0, 2e-3, 100e-3, gamma=GAMMA_1H, shape="trapezoid", ramp=0.0)
    ramped = dif.b_value(1.0, 2e-3, 100e-3, gamma=GAMMA_1H, shape="trapezoid", ramp=0.4e-3)
    assert ramped < plain


def test_trapezoid_composes_with_the_bipolar_correction():
    plain = dif.b_value(1.0, 2e-3, 100e-3, gamma=GAMMA_1H, shape="trapezoid",
                        ramp=0.3e-3, sequence="pgse")
    bipolar = dif.b_value(1.0, 2e-3, 100e-3, gamma=GAMMA_1H, shape="trapezoid",
                          ramp=0.3e-3, sequence="bipolar", tau=4e-3)
    assert bipolar < plain


def test_a_ramp_longer_than_delta_is_rejected():
    # delta is the flat top plus one ramp, so a longer ramp describes no
    # real pulse.
    with pytest.raises(ValueError, match="cannot exceed delta"):
        dif.b_value(1.0, 2e-3, 100e-3, gamma=GAMMA_1H, shape="trapezoid", ramp=3e-3)
    with pytest.raises(ValueError, match="ramp must be"):
        dif.b_value(1.0, 2e-3, 100e-3, gamma=GAMMA_1H, ramp=-1.0)


def test_ramp_is_ignored_by_the_effective_duration_shapes():
    # Only the trapezoid has a ramp term; passing one elsewhere must
    # not silently change the answer.
    for shape in ("rect", "sine", "smsq10", "gauss"):
        assert dif.b_value(1.0, 2e-3, 0.1, gamma=GAMMA_1H, shape=shape, ramp=0.4e-3) == \
            pytest.approx(dif.b_value(1.0, 2e-3, 0.1, gamma=GAMMA_1H, shape=shape))
    assert "trapezoid" in dif.SHAPES_WITH_RAMP
    assert "rect" not in dif.SHAPES_WITH_RAMP


def test_relative_gradients_and_calibration_forward_the_ramp():
    b_plain = dif.b_values_from_relative([0.5], 0.5, 2e-3, 0.1, nucleus="1H",
                                          shape="trapezoid", ramp=0.0)
    b_ramped = dif.b_values_from_relative([0.5], 0.5, 2e-3, 0.1, nucleus="1H",
                                           shape="trapezoid", ramp=0.4e-3)
    assert b_ramped[0] < b_plain[0]

    # And a calibration run with a ramp recovers the gradient that
    # produced data *with* that ramp.
    relative = np.linspace(0.05, 0.95, 14)
    b_true = dif.b_value(relative * 0.6, 2e-3, 0.1, gamma=GAMMA_1H,
                         shape="trapezoid", ramp=0.4e-3)
    y = 1000.0 * np.exp(-2.299e-9 * b_true)
    result = dif.calibrate_max_gradient(relative, y, 2.299e-9, 2e-3, 0.1, nucleus="1H",
                                        shape="trapezoid", ramp=0.4e-3)
    assert result["max_gradient_t_per_m"] == pytest.approx(0.6, rel=1e-5)


# ============================================================
# Reading the timings out of an acquisition file
# ============================================================

def test_pfg_params_read_from_a_real_xpulse_experiment():
    import os

    from opennmr.io import xpulse

    path = os.path.join("test_data", "xpulse_PFG", "4")
    if not os.path.isdir(path):
        pytest.skip(f"{path} not available")
    spectrum = xpulse.read_array(path)
    found = dif.pfg_params_from_metadata(spectrum)

    assert found["delta"] == pytest.approx(4.0e-3)     # 4000 us, plateau
    assert found["DELTA"] == pytest.approx(50.0e-3)    # 50000 us
    assert found["ramp"] == pytest.approx(0.5e-3)      # D70, 500 us
    assert found["delta_is_plateau_only"] is True


def test_effective_delta_adds_the_ramp_only_when_the_file_excludes_it():
    # The trap this exists to close: X-Pulse records the plateau alone,
    # while the trapezoid expression wants plateau plus one ramp.
    assert dif.effective_delta(4e-3, 0.5e-3, plateau_only=True) == pytest.approx(4.5e-3)
    # A file recording the total already must not have it added twice.
    assert dif.effective_delta(4.5e-3, 0.5e-3, plateau_only=False) == pytest.approx(4.5e-3)


def test_pfg_params_are_empty_for_a_vendor_that_records_none():
    from opennmr.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum

    spectrum = NMRSpectrum(
        data=np.zeros(4), axes=[NMRAxis(kind="frequency", ppm=np.arange(4.0), hz=np.arange(4.0))],
        metadata=NMRMetadata(vendor="bruker", nucleus="1H", sfo1=400.0),
    )
    # Nothing invented: a caller can tell "the file says 4 ms" from
    # "the file does not say".
    assert dif.pfg_params_from_metadata(spectrum) == {}


def test_stejskal_tanner_cannot_fit_a_negative_intensity():
    """Explicit user request: "for PFG there should be no possibility
    of negative intensities". A PFG amplitude is a peak intensity, and
    an unbounded I0 let a noisy or over-attenuated series settle on a
    mirror-image decay -- a negative amplitude with a perfectly
    plausible-looking D."""
    from opennmr.processing import relaxation as processing_relaxation

    b = np.linspace(0.0, 4.0e9, 12)
    # Deliberately upside down: the best unconstrained fit to this is a
    # negative I0.
    y = -1.0 * np.exp(-1.0e-9 * b)

    result = processing_relaxation.fit_relaxation(b, y, model="stejskal_tanner")
    assert result["params"]["I0"] >= 0.0
    assert result["bounds"][0][0] == 0.0

    biexp = processing_relaxation.fit_relaxation(
        b, y, model="stejskal_tanner_biexponential")
    assert biexp["params"]["I0a"] >= 0.0 and biexp["params"]["I0b"] >= 0.0

    stretched = processing_relaxation.fit_relaxation(
        b, y, model="stejskal_tanner_stretched")
    assert stretched["params"]["I0"] >= 0.0


def test_stejskal_tanner_still_fits_an_ordinary_decay():
    """The bound must not cost anything for real data."""
    from opennmr.processing import relaxation as processing_relaxation

    b = np.linspace(0.0, 4.0e9, 16)
    y = 3.0 * np.exp(-1.5e-9 * b)
    result = processing_relaxation.fit_relaxation(b, y, model="stejskal_tanner")
    assert result["params"]["I0"] == pytest.approx(3.0, rel=1e-4)
    assert result["params"]["D"] == pytest.approx(1.5e-9, rel=1e-4)
