"""Shaped-pulse simulator at any duration (it used to stop at 10 µs).

Auto time step: 1 µs from 100 µs up (unchanged), at least
``MIN_AUTO_STEPS`` steps below, and the flip-angle power is integrated on
the same steps the simulation uses.
"""
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from opennmr.simulate import bloch


@pytest.fixture
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.mark.parametrize("duration", [100.0, 500.0, 1000.0, 100000.0])
def test_pulses_of_100us_and_longer_keep_the_hardware_1us_step(duration):
    assert bloch.auto_time_step_us(None, duration) == 1.0


@pytest.mark.parametrize("duration", [0.01, 0.25, 1.0, 2.0, 10.0, 50.0])
def test_short_pulses_get_at_least_min_auto_steps(duration):
    step = bloch.auto_time_step_us(None, duration)
    assert step <= 1.0
    assert bloch.n_steps_for(duration, step) >= bloch.MIN_AUTO_STEPS


def test_the_sweep_rule_still_applies_and_the_finer_rule_wins():
    assert bloch.auto_time_step_us(2e6, 1000.0) == 0.05     # sweep sets it
    assert bloch.auto_time_step_us(2e6, 2.0) == 0.02        # duration sets it
    assert bloch.auto_time_step_us(None) == 1.0             # old call unchanged


def test_very_long_amplitude_pulses_are_capped_not_refused():
    step = bloch.auto_time_step_us(None, 1e7)               # 10 s
    assert bloch.n_steps_for(1e7, step) == bloch.MAX_AUTO_STEPS


@pytest.mark.parametrize("shape", ["GAUSSIAN", "SINC", "EBURP2", "REBURP"])
@pytest.mark.parametrize("duration", [0.2, 2.0, 7.5])
def test_short_pulse_gives_the_requested_flip_on_resonance(shape, duration):
    flip = 180.0 if shape == "REBURP" else 90.0
    b1 = bloch.peak_b1_for_flip_angle(shape, duration, flip)
    sim = bloch.simulate_pulse(shape, duration, b1, [0.0])
    assert sim["Mz"][0, 0] == pytest.approx(np.cos(np.deg2rad(flip)), abs=1e-6)


def test_1us_steps_would_be_wrong_for_a_2us_pulse():
    # Why the minimum exists: at the old fixed 1 us a 2 us Gaussian is two
    # equal rectangles -- a hard pulse -- and its profile is off by more
    # than half of M0 somewhere in the offsets it excites.
    duration = 2.0
    offsets = np.linspace(*bloch.default_offset_range_hz(duration), 201)

    def run(step):
        b1 = bloch.peak_b1_for_flip_angle("GAUSSIAN", duration, 90.0, time_step_us=step)
        return bloch.simulate_pulse("GAUSSIAN", duration, b1, offsets, time_step_us=step)

    reference, coarse, auto = run(0.002), run(1.0), run(None)
    error = lambda sim: max(np.abs(sim[k] - reference[k]).max() for k in ("Mx", "My", "Mz"))
    assert error(coarse) > 0.5
    # 100 steps: the sampling a 100 us pulse gets at 1 us, within 0.01 of M0.
    assert error(auto) < 0.01


def test_a_non_positive_duration_is_refused():
    with pytest.raises(ValueError):
        bloch.simulate_pulse("GAUSSIAN", 0.0, 1000.0, [0.0])


@pytest.mark.parametrize("duration,unit", [(0.25, "ns"), (5.0, "μs"), (1000.0, "μs"),
                                           (5e4, "ms"), (2e7, "s")])
def test_plot_time_axis_unit_follows_the_duration(duration, unit):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure

    sim = bloch.simulate_pulse("GAUSSIAN", duration, 1000.0, [0.0], time_step_us=duration / 50)
    ax_shape, _ = bloch.plot_pulse_simulation(sim, Figure())
    assert ax_shape.get_xlabel() == f"Time / {unit}"
    scale, _ = bloch.time_unit(duration)
    assert ax_shape.lines[0].get_xdata()[-1] == pytest.approx(sim["time_us"][-1] / scale)


def test_gui_duration_accepts_sub_10us_and_long_pulses(qapp):
    from opennmr.gui.main_window import MainWindow

    window = MainWindow()
    spin = window.bloch_duration_spin
    assert spin.minimum() <= 0.001 and spin.maximum() >= 1e7
    spin.setValue(0.25)
    assert spin.value() == pytest.approx(0.25)


def test_gui_simulates_a_250ns_pulse(qapp):
    from opennmr.gui.main_window import MainWindow

    window = MainWindow()
    window.bloch_shape_combo.setCurrentIndex(window.bloch_shape_combo.findData("GAUSSIAN"))
    window.bloch_power_mode_combo.setCurrentIndex(window.bloch_power_mode_combo.findData("flip"))
    window.bloch_flip_spin.setValue(90.0)
    window.bloch_duration_spin.setValue(0.25)
    window._run_bloch_simulation()
    text = window.bloch_summary_label.text()
    assert "Cannot simulate" not in text
    assert "100 steps" in text
    assert "Mz = 0.000" in text or "Mz = -0.000" in text
