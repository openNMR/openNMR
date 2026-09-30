"""Adiabatic (frequency-swept) shapes in the Bloch simulator:
hyperbolic secant (HSn), WURST-n and tanh/tan (opennmr.simulate.bloch).

The property that makes a pulse adiabatic -- inversion that does not
depend on B1 once it is high enough -- is what these check, against a
Gaussian 180 that is ruined by the same change in power.
"""
import numpy as np
import pytest

from opennmr.simulate import bloch

T_US, SW = 500.0, 100e3


def _sim(shape, b1, power_scales=(1.0,), offsets=None, **kw):
    offsets = np.linspace(-SW, SW, 401) if offsets is None else offsets
    return bloch.simulate_pulse(shape, T_US, b1, offsets, power_scales=power_scales,
                                sweep_width_hz=SW, **kw)


def test_shapes_sweep_from_minus_to_plus_half_the_sweep_width():
    for shape in bloch.ADIABATIC_SHAPES:
        amp, phase, freq = bloch.adiabatic_shape(shape, 500, T_US, SW)
        assert amp.max() == pytest.approx(1.0)
        assert freq[0] == pytest.approx(-SW / 2, rel=0.08)
        assert freq[-1] == pytest.approx(SW / 2, rel=0.08)
        assert np.all(np.diff(freq) > 0)
        # The phase is the running integral of the frequency.
        dt = T_US * 1e-6 / 500
        assert np.diff(np.deg2rad(phase)) == pytest.approx(2 * np.pi * freq[1:] * dt, abs=1e-9)


def test_hs1_frequency_is_the_classic_tanh_sweep():
    amp, _phase, freq = bloch.adiabatic_shape("HS", 2000, T_US, SW, beta=5.3)
    tau = (np.arange(2000) + 0.5) / 2000 * 2 - 1
    assert amp == pytest.approx(1 / np.cosh(5.3 * tau) / (1 / np.cosh(5.3 * tau)).max())
    assert freq == pytest.approx(SW / 2 * np.tanh(5.3 * tau) / np.tanh(5.3), abs=0.002 * SW)


def test_a_phase_ramp_acts_at_the_offset_it_sweeps_to():
    # Sign convention: RF at +f from the carrier is on resonance for the
    # isochromat at offset +f (checked with a rectangular 180 whose phase
    # ramps at 20 kHz -- the same rotation the adiabatic shapes rely on).
    n, f = 200, 20e3
    b1 = 1 / (2 * n * 1e-6)
    offsets = np.linspace(-40e3, 40e3, 801)
    M = np.zeros((1, len(offsets), 3))
    M[..., 2] = 1
    for k in range(n):
        ph = 2 * np.pi * f * (k + 0.5) * 1e-6
        M = bloch._rotate_many(M, b1 * np.cos(ph), b1 * np.sin(ph), offsets[None, :], 1e-6)
    assert offsets[np.argmin(M[0, :, 2])] == pytest.approx(f)


@pytest.mark.parametrize("shape", ["HS", "WURST"])
def test_inversion_does_not_depend_on_power_above_the_adiabatic_threshold(shape):
    b1 = bloch.peak_b1_for_adiabaticity(shape, T_US, SW, q=5.0)
    sim = _sim(shape, b1, power_scales=(1.0, 2.0, 4.0))
    bands = [bloch.inverted_band_hz(sim, power_index=i) for i in range(3)]
    assert all(band is not None for band in bands)
    widths = [hi - lo for lo, hi in bands]
    assert min(widths) > 0.55 * SW                           # a broad band inverted...
    assert max(widths) - min(widths) < 0.15 * SW             # ...at 1x, 2x and 4x power
    centre = np.abs(sim["offsets_hz"]) < 0.25 * SW
    assert np.all(sim["Mz"][:, centre] < -0.97)


def test_a_gaussian_180_is_not_power_insensitive():
    # The contrast that makes the test above mean something.
    b1 = bloch.peak_b1_for_flip_angle("GAUSSIAN", T_US, 180.0)
    sim = bloch.simulate_pulse("GAUSSIAN", T_US, b1, [0.0], power_scales=(1.0, 2.0))
    assert sim["Mz"][0, 0] == pytest.approx(-1.0, abs=1e-3)
    assert sim["Mz"][1, 0] == pytest.approx(1.0, abs=1e-3)


def test_hs_inverts_nearly_the_whole_sweep_with_uniform_adiabaticity():
    b1 = bloch.peak_b1_for_adiabaticity("HS", T_US, SW, q=5.0)
    sim = _sim("HS", b1)
    lo, hi = bloch.inverted_band_hz(sim)
    assert hi - lo > 0.9 * SW
    q = sim["adiabaticity"]
    assert q["q0"] == pytest.approx(5.0, rel=1e-3)
    assert q["q_band"] == pytest.approx(q["q0"], rel=0.02)   # offset-independent


def test_q0_for_a_linear_sweep_matches_the_formula():
    # WURST sweeps linearly: d nu/dt = SW/T, amplitude 1 at the centre,
    # so Q0 = 2 pi nu1^2 T / SW.
    b1 = 20e3
    q = bloch.adiabaticity("WURST", T_US, SW, b1)
    assert q["q0"] == pytest.approx(2 * np.pi * b1 ** 2 * T_US * 1e-6 / SW, rel=1e-3)
    assert q["q_band"] < q["q0"]                              # falls towards the edges
    assert bloch.peak_b1_for_adiabaticity("WURST", T_US, SW, q=q["q0"]) == pytest.approx(b1, rel=1e-6)


def test_tanh_tan_inverts_on_resonance_and_widens_with_a_longer_pulse():
    b1 = bloch.peak_b1_for_adiabaticity("TANH_TAN", T_US, SW, q=5.0)
    sim = _sim("TANH_TAN", b1, offsets=np.array([0.0]))
    assert sim["Mz"][0, 0] < -0.95
    short = bloch.simulate_pulse("TANH_TAN", 500.0, 4 * b1, np.linspace(-SW, SW, 401),
                                 sweep_width_hz=SW)
    long_ = bloch.simulate_pulse("TANH_TAN", 5000.0, 4 * b1, np.linspace(-SW, SW, 401),
                                 sweep_width_hz=SW)
    w_short = np.subtract(*bloch.inverted_band_hz(short)[::-1]) if bloch.inverted_band_hz(short) else 0
    w_long = np.subtract(*bloch.inverted_band_hz(long_)[::-1])
    assert w_long > w_short


def test_simulation_is_converged_in_the_time_step():
    # One point per us is the hardware's resolution; the physics must not
    # depend on it (tanh/tan sweeps fastest, so it is the hardest case).
    b1, n_fine = 20e3, 5000
    coarse = bloch.simulate_pulse("TANH_TAN", T_US, b1, [0.0, 10e3], sweep_width_hz=SW)
    amp, ph, _f = bloch.adiabatic_shape("TANH_TAN", n_fine, T_US, SW)
    dt = T_US * 1e-6 / n_fine
    M = np.zeros((1, 2, 3))
    M[..., 2] = 1
    for k in range(n_fine):
        p = np.deg2rad(ph[k])
        M = bloch._rotate_many(M, b1 * amp[k] * np.cos(p), b1 * amp[k] * np.sin(p),
                               np.array([[0.0, 10e3]]), dt)
    assert coarse["Mz"][0] == pytest.approx(M[0, :, 2], abs=0.05)


def test_parameters_can_be_changed():
    a = bloch.adiabatic_shape("WURST", 500, T_US, SW, n=2)[0]
    b = bloch.adiabatic_shape("WURST", 500, T_US, SW, n=40)[0]
    assert a.mean() < b.mean()                                # WURST-2 is rounder
    with pytest.raises(ValueError):
        bloch.adiabatic_shape("WURST", 500, T_US, SW, beta=3)  # not a WURST parameter


def test_needs_a_sweep_width_and_refuses_one_it_cannot_synthesise():
    with pytest.raises(ValueError, match="sweep width"):
        bloch.simulate_pulse("HS", T_US, 1e4, [0.0])
    # 1 point per us: the RF cannot be more than 500 kHz off the carrier.
    with pytest.raises(ValueError, match="cannot be synthesised"):
        bloch.adiabatic_shape("HS", 500, T_US, 1.2e6)


def test_adiabatic_shapes_have_no_flip_angle():
    with pytest.raises(ValueError, match="adiabatic"):
        bloch.peak_b1_for_flip_angle("HS", T_US, 180.0)


def test_default_offset_range_covers_the_sweep():
    assert bloch.default_offset_range_hz(T_US, sweep_width_hz=SW) == (-SW, SW)
    assert bloch.default_offset_range_hz(T_US) == (-16000.0, 16000.0)


def test_plot_shows_the_frequency_sweep_not_the_phase():
    from matplotlib.figure import Figure
    sim = _sim("WURST", 15e3)
    fig = Figure()
    ax_shape, _prof = bloch.plot_pulse_simulation(sim, fig)
    twin = [ax for ax in fig.axes if ax is not ax_shape and ax is not _prof][0]
    assert "kHz" in twin.get_ylabel()


# -- time step (2026-09-30: sweeps of several MHz) --------------------------------

def test_auto_time_step_is_the_hardware_microsecond_until_a_sweep_needs_finer():
    assert bloch.auto_time_step_us(None) == 1.0
    assert bloch.auto_time_step_us(100e3) == 1.0          # 0.05 turn per step exactly
    assert bloch.auto_time_step_us(200e3) == 0.5
    assert bloch.auto_time_step_us(2e6) == 0.05
    assert bloch.auto_time_step_us(5e6) == 0.02
    for sw in (150e3, 1e6, 3.3e6, 20e6):
        step = bloch.auto_time_step_us(sw)
        assert 0.5 * sw * step * 1e-6 <= bloch.MAX_TURNS_PER_STEP + 1e-12


def test_unchanged_below_100_khz():
    # Every sweep the old 1 us simulator handled well gives the same answer.
    old = _sim("WURST", 2e3)
    assert old["time_step_us"] == 1.0
    assert len(old["time_us"]) == int(T_US)
    np.testing.assert_allclose(old["time_us"][:2], [0.5, 1.5])


@pytest.mark.parametrize("shape", ["WURST", "HS", "TANH_TAN"])
def test_a_several_mhz_sweep_is_converged(shape):
    # 2 MHz in 200 us, power from Q0 = 5: auto against a 4x finer step.
    sw, t_us = 2e6, 200.0
    offsets = np.linspace(-sw, sw, 81)
    b1 = bloch.peak_b1_for_adiabaticity(shape, t_us, sw, q=5.0)
    auto = bloch.simulate_pulse(shape, t_us, b1, offsets, sweep_width_hz=sw)
    fine = bloch.simulate_pulse(shape, t_us, b1, offsets, sweep_width_hz=sw,
                                time_step_us=auto["time_step_us"] / 4)
    assert auto["time_step_us"] == pytest.approx(0.05)
    assert auto["max_turns_per_step"] <= bloch.MAX_TURNS_PER_STEP + 1e-9
    assert np.max(np.abs(auto["Mz"] - fine["Mz"])) < 0.02
    # ... and WURST and HS invert most of the sweep, as they should (tanh/tan
    # at Q0 = 5 does not: its Q falls far below Q0 away from the centre --
    # see bloch.adiabaticity).
    if shape != "TANH_TAN":
        band = bloch.inverted_band_hz(auto)
        assert band is not None and band[1] - band[0] > 0.6 * sw


def test_the_old_one_microsecond_limit_was_wrong_near_it():
    # A 900 kHz sweep at 1 us per step (allowed by the old GUI) is 0.45
    # turn per step: its profile is far from the converged one.
    sw, t_us = 900e3, 500.0
    offsets = np.linspace(-sw, sw, 61)
    b1 = bloch.peak_b1_for_adiabaticity("WURST", t_us, sw, q=5.0)
    coarse = bloch.simulate_pulse("WURST", t_us, b1, offsets, sweep_width_hz=sw,
                                  time_step_us=1.0)
    auto = bloch.simulate_pulse("WURST", t_us, b1, offsets, sweep_width_hz=sw)
    assert coarse["max_turns_per_step"] > 0.4
    assert np.max(np.abs(coarse["Mz"] - auto["Mz"])) > 0.3


def test_too_many_steps_is_refused():
    with pytest.raises(ValueError, match="steps"):
        bloch.simulate_pulse("WURST", 1e5, 1e4, [0.0], sweep_width_hz=2e7)


def test_offsets_axis_switches_to_mhz():
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    sim = bloch.simulate_pulse("HS", 100.0, 1e5, np.linspace(-3e7, 3e7, 11),
                               sweep_width_hz=4e6)
    fig = Figure()
    _shape, prof = bloch.plot_pulse_simulation(sim, fig)
    assert prof.get_xlabel() == "Offset / MHz"
    assert bloch.frequency_unit(5e3) == (1.0, "Hz")
    assert bloch.frequency_unit(5e5) == (1e3, "kHz")
