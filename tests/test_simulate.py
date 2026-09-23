import numpy as np
import pytest

from opennmr.core.registry import find_gamma
from opennmr.simulate import bloch, diffusion


# ============================================================
# PFG attenuation (port of leonmr.sim_diffusion)
# ============================================================

def test_defaults_reproduce_leonmr_sim_diffusion_exactly():
    # leonmr: B = (2*pi*gamma*delta*G)**2 * (DELTA - delta/3), I = exp(-D*B),
    # over G = arange(0, g + g/100, g/99), D swept 1e-8..1e-15.
    result = diffusion.pfg_attenuation("1H")
    gamma = find_gamma("1H")["gamma"]
    delta, DELTA, maxgrad = 1 / 1000, 20 / 1000, 17
    G = np.arange(0, maxgrad + (maxgrad / 100.0), maxgrad / 99.0)
    B = np.array([(2 * np.pi * gamma * delta * g) ** 2 * (DELTA - (delta / 3)) for g in G])
    D = np.logspace(-8, -15, 8)
    np.testing.assert_allclose(result["gradient"], G)
    np.testing.assert_allclose(result["b"], B, rtol=1e-12)
    np.testing.assert_allclose(result["D"], D)
    np.testing.assert_allclose(result["intensity"], np.exp(-np.outer(D, B)), rtol=1e-12)


def test_a_single_d_gives_one_curve():
    result = diffusion.pfg_attenuation("7Li", D=2e-10, max_gradient=0.5)
    assert result["intensity"].shape == (1, len(result["gradient"]))
    assert result["intensity"][0, 0] == 1.0
    assert np.all(np.diff(result["intensity"][0]) <= 0)


def test_a_half_sine_gradient_attenuates_less_than_a_rectangle():
    # Same nominal gradient and timings: the sine lobe's area is 2/pi of
    # the rectangle's, so b is (2/pi)^2 of it.
    rect = diffusion.pfg_attenuation("1H", D=1e-9, shape="rect")
    sine = diffusion.pfg_attenuation("1H", D=1e-9, shape="sine")
    np.testing.assert_allclose(sine["b"][1:] / rect["b"][1:], (2 / np.pi) ** 2)
    assert np.all(sine["intensity"][0, 1:] > rect["intensity"][0, 1:])


def test_a_bipolar_recovery_delay_shortens_the_diffusion_time():
    pgse = diffusion.pfg_attenuation("1H", D=1e-9, sequence="pgse")
    bipolar = diffusion.pfg_attenuation("1H", D=1e-9, sequence="bipolar", tau_ms=2.0)
    ratio = bipolar["b"][1:] / pgse["b"][1:]
    expected = (20e-3 - 1e-3 / 3 - 2e-3 / 2) / (20e-3 - 1e-3 / 3)
    np.testing.assert_allclose(ratio, expected)


def test_compare_draws_every_shape_and_every_sequence():
    shapes = diffusion.compare_attenuation("1H", D=1e-9, vary="shape", ramp_ms=0.1)
    assert [c["key"] for c in shapes] == list(diffusion._diffusion.GRADIENT_SHAPE_LABELS)
    assert all(c["intensity"] is not None for c in shapes)
    sequences = diffusion.compare_attenuation("1H", D=1e-9, vary="sequence", tau_ms=1.0)
    assert [c["key"] for c in sequences] == list(diffusion._diffusion.DIFFUSION_SEQUENCES)


def test_compare_keeps_an_impossible_option_with_its_reason():
    # A recovery delay longer than the diffusion time leaves no time:
    # the bipolar options cannot be drawn, the plain ones still can.
    curves = diffusion.compare_attenuation("1H", D=1e-9, vary="sequence", DELTA_ms=5.0, tau_ms=12.0)
    by_key = {c["key"]: c for c in curves}
    assert by_key["pgse"]["intensity"] is not None
    assert by_key["bipolar"]["intensity"] is None
    assert "DELTA" in by_key["bipolar"]["error"]


def test_compare_rejects_an_unknown_axis():
    with pytest.raises(ValueError):
        diffusion.compare_attenuation("1H", D=1e-9, vary="colour")


# ============================================================
# Bloch simulation (port of leonmr/bloch_sim.py)
# ============================================================

def _reference_simulate(shape, duration_us, peak_b1_hz, offsets_hz, power_scales):
    """The original simulate_pulse, one isochromat at a time."""
    def rotate(M, axis, angle):
        if angle == 0.0:
            return M
        k = axis
        return M * np.cos(angle) + np.cross(k, M) * np.sin(angle) + k * np.dot(k, M) * (1 - np.cos(angle))

    n_steps = int(round(duration_us))
    amp, phase = bloch.generate_shape(shape, n_steps)
    phase = np.deg2rad(phase)
    out = np.zeros((3, len(power_scales), len(offsets_hz)))
    for pi, p in enumerate(power_scales):
        bx = p * peak_b1_hz * amp * np.cos(phase)
        by = p * peak_b1_hz * amp * np.sin(phase)
        for oi, off in enumerate(offsets_hz):
            M = np.array([0.0, 0.0, 1.0])
            for i in range(n_steps):
                B = np.array([bx[i], by[i], off])
                mag = np.linalg.norm(B)
                if mag > 0:
                    M = rotate(M, B / mag, 2 * np.pi * mag * 1e-6)
            out[:, pi, oi] = M
    return out


@pytest.mark.parametrize("shape", ["GAUSSIAN", "SINC", "EBURP2", "REBURP"])
def test_vectorised_simulation_matches_the_original_loop(shape):
    offsets = np.linspace(-3000, 3000, 13)
    scales = (0.8, 1.0)
    fast = bloch.simulate_pulse(shape, 300, 2500.0, offsets, power_scales=scales)
    slow = _reference_simulate(shape, 300, 2500.0, offsets, scales)
    np.testing.assert_allclose(fast["Mx"], slow[0], atol=1e-10)
    np.testing.assert_allclose(fast["My"], slow[1], atol=1e-10)
    np.testing.assert_allclose(fast["Mz"], slow[2], atol=1e-10)


def test_magnetisation_stays_unit_length():
    sim = bloch.simulate_pulse("SINC", 1000, 3000.0, np.linspace(-5000, 5000, 51))
    norm = np.sqrt(sim["Mx"] ** 2 + sim["My"] ** 2 + sim["Mz"] ** 2)
    np.testing.assert_allclose(norm, 1.0, atol=1e-9)


def test_burp_integration_factors_match_the_bruker_shape_files():
    # SHAPE_INTEGFAC quoted in bloch.py's table comments.
    assert bloch.integration_factor("EBURP2", 1000) == pytest.approx(6.102960e-02, rel=1e-5)
    assert bloch.integration_factor("REBURP", 1000) == pytest.approx(7.981016e-02, rel=1e-5)


@pytest.mark.parametrize("flip, mz", [(90.0, 0.0), (180.0, -1.0)])
def test_flip_angle_power_gives_that_flip_on_resonance(flip, mz):
    b1 = bloch.peak_b1_for_flip_angle("GAUSSIAN", 1000, flip)
    sim = bloch.simulate_pulse("GAUSSIAN", 1000, b1, [0.0])
    assert sim["Mz"][0, 0] == pytest.approx(mz, abs=1e-3)


def test_hard_pulse_conversion_is_the_original():
    # 10 us hard 90 at amplitude 1.0 -> 25 kHz nutation; half the TX
    # amplitude gives half of it.
    assert bloch.hard_pulse_to_b1_hz(0.5, 10.0, 1.0) == pytest.approx(12500.0)


def test_plotting_draws_onto_a_supplied_figure():
    from matplotlib.figure import Figure

    fig = Figure()
    sim = bloch.simulate_pulse("GAUSSIAN", 200, 1000.0, np.linspace(-2000, 2000, 11),
                               power_scales=(0.9, 1.1))
    ax_shape, ax_prof = bloch.plot_pulse_simulation(sim, fig, title="t")
    assert len(ax_prof.lines) == 6 + 1  # Mx/My/Mz per power scale, plus the zero line

    fig2 = Figure()
    ax = fig2.add_subplot(111)
    diffusion.plot_attenuation([diffusion.pfg_attenuation("1H")], ax)
    assert len(ax.lines) == 8
