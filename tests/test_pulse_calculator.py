"""Pulse calculator (2026-11-13): pulse lengths, powers in W / dB / scaled
units, nutation frequencies and bandwidths -- notebook.pulse_calculator
and Simulate ▸ Pulse calculator."""

import math

import numpy as np
import pytest

from opennmr.notebook import pulse_calculator as pc


# ------------------------------------------------------------------
# Units
# ------------------------------------------------------------------

def test_db_is_topspins_higher_db_is_less_power():
    assert pc.power_to_watts(0.0, "dB") == pytest.approx(1.0)
    assert pc.power_to_watts(10.0, "dB") == pytest.approx(0.1)
    assert pc.power_to_watts(-20.0, "dB") == pytest.approx(100.0)
    assert pc.watts_to_power(100.0, "dB") == pytest.approx(-20.0)
    # What 0 dB is can be changed.
    assert pc.power_to_watts(3.0, "dB", db_ref_w=50.0) == pytest.approx(50.0 * 10 ** -0.3)


def test_scaled_units_are_watts_times_the_factor_or_relative_without_it():
    assert pc.power_to_watts(40.0, "scaled", scale=2.5) == pytest.approx(100.0)
    assert pc.power_to_watts(40.0, "scaled") is None
    assert pc.watts_to_power(100.0, "scaled", scale=2.5) == pytest.approx(40.0)
    # Two scaled powers compare without the factor; it cancels.
    assert pc.power_ratio(10.0, "scaled", 40.0, "scaled") == pytest.approx(0.25)
    # Against W or dB the factor is needed.
    with pytest.raises(ValueError, match="scaling factor"):
        pc.power_ratio(10.0, "scaled", 40.0, "W")
    assert pc.power_ratio(10.0, "scaled", 100.0, "W", scale=2.5) == pytest.approx(0.25)


def test_power_ratio_between_db_values_and_mixed_units():
    assert pc.power_ratio(23.0103, "dB", 20.0, "dB") == pytest.approx(0.5, rel=1e-5)
    assert pc.power_ratio(-20.0, "dB", 50.0, "W") == pytest.approx(2.0)
    with pytest.raises(ValueError):
        pc.power_to_watts(-1.0, "W")
    with pytest.raises(ValueError):
        pc.power_to_watts(1.0, "mW")


# ------------------------------------------------------------------
# Nutation
# ------------------------------------------------------------------

def test_nutation_frequency_and_pulse_length_are_inverse():
    assert pc.nutation_frequency(2.5) == pytest.approx(100e3)
    assert pc.nutation_frequency(5.0, 180.0) == pytest.approx(100e3)
    assert pc.pulse_length(100e3, 90.0) == pytest.approx(2.5)
    assert pc.flip_angle(100e3, 2.5) == pytest.approx(90.0)


def test_nu1_goes_as_the_square_root_of_the_power():
    out = pc.pulse_calculator(2.5, 100.0, "W", power=25.0, unit="W")
    assert out["power_ratio"] == pytest.approx(0.25)
    assert out["nu1_hz"] == pytest.approx(50e3)
    assert out["pulse_us"][90.0] == pytest.approx(5.0)
    assert out["pulse_us"][180.0] == pytest.approx(10.0)
    assert out["power"]["W"] == pytest.approx(25.0)
    assert out["power"]["dB"] == pytest.approx(-10 * math.log10(25.0))
    assert "√(P / P_ref)" in out["report"]


def test_the_same_conversion_in_db():
    # 3.0103 dB more attenuation = half the power = ν₁ / √2.
    out = pc.pulse_calculator(2.5, 0.0, "dB", power=3.0103, unit="dB")
    assert out["nu1_hz"] == pytest.approx(100e3 / math.sqrt(2), rel=1e-5)
    assert out["power"]["dB"] == pytest.approx(3.0103)


def test_scaled_units_without_a_factor_still_convert_among_themselves():
    out = pc.pulse_calculator(2.5, 80.0, "scaled", power=20.0, unit="scaled")
    assert out["nu1_hz"] == pytest.approx(50e3)
    assert out["power"]["scaled"] == pytest.approx(20.0)
    assert out["power"]["W"] is None            # no factor: no watts


def test_power_for_a_nutation_frequency_and_for_a_pulse_length():
    out = pc.pulse_calculator(2.5, 100.0, "W", nu1_hz=50e3)
    assert out["power"]["W"] == pytest.approx(25.0)
    out = pc.pulse_calculator(2.5, 100.0, "W", target_pulse_us=10.0, target_flip_deg=180.0)
    assert out["nu1_hz"] == pytest.approx(50e3)
    assert out["power"]["W"] == pytest.approx(25.0)
    assert out["pulse_us"][180.0] == pytest.approx(10.0)
    with pytest.raises(ValueError, match="at most one"):
        pc.pulse_calculator(2.5, 100.0, "W", power=1.0, nu1_hz=1.0)


def test_central_transition_and_b1():
    out = pc.pulse_calculator(3.0, 100.0, "W", spin=2.5, nucleus="1H")
    assert out["ct"]["factor"] == pytest.approx(3.0)
    assert out["ct"]["pulse_us"][90.0] == pytest.approx(1.0)
    # 1H: γ/2π = 42.577 MHz/T, so ν₁ = 83.3 kHz is about 1.957 mT.
    assert out["b1_mT"] == pytest.approx(83333.3 / 42.577e6 * 1e3, rel=1e-3)
    with pytest.raises(ValueError, match="half-integer"):
        pc.pulse_calculator(3.0, 100.0, "W", spin=1.0)


# ------------------------------------------------------------------
# Excitation profile and bandwidth
# ------------------------------------------------------------------

def test_profile_on_resonance_is_the_flip_angle():
    for flip in (30.0, 90.0, 180.0):
        tau = pc.pulse_length(50e3, flip)
        prof = pc.excitation_profile([0.0], 50e3, tau)
        assert prof["mz"][0] == pytest.approx(math.cos(math.radians(flip)), abs=1e-12)
        assert prof["mxy"][0] == pytest.approx(abs(math.sin(math.radians(flip))), abs=1e-12)


def test_profile_keeps_the_magnetisation_length():
    offsets = np.linspace(-500e3, 500e3, 101)
    prof = pc.excitation_profile(offsets, 50e3, 5.0)
    norm = prof["mx"] ** 2 + prof["my"] ** 2 + prof["mz"] ** 2
    np.testing.assert_allclose(norm, 1.0, atol=1e-12)


def test_low_flip_bandwidth_approaches_the_sinc_width():
    # A 5° pulse is in the linear regime: |Mxy| ∝ |sinc(Δ τ)|, whose FWHM
    # is 1.2067 / τ.
    nu1, tau = 1e3, pc.pulse_length(1e3, 5.0)
    out = pc.pulse_bandwidth(nu1, tau)
    assert out["kind"] == "excitation"
    assert out["fwhm_hz"] == pytest.approx(1.2067 / (tau * 1e-6), rel=5e-3)
    assert out["first_null_hz"] == pytest.approx(1 / (tau * 1e-6))


def test_a_180_reports_the_inversion_bandwidth():
    out = pc.pulse_bandwidth(50e3, 10.0)
    assert out["kind"] == "inversion" and out["flip_deg"] == pytest.approx(180.0)
    assert 0 < out["fwhm_hz"] < 1 / 10e-6 * 2


# ------------------------------------------------------------------
# The notebook exports and the Simulate panel
# ------------------------------------------------------------------

def test_notebook_exports():
    from opennmr import notebook as nmr
    assert nmr.calculate_pulse is pc.pulse_calculator
    assert nmr.nutation_frequency(2.5) == pytest.approx(100e3)
    assert nmr.pulse_calculator is pc


def test_simulate_panel_calculates_and_plots(qapp):
    from opennmr.gui.main_window import MainWindow
    from opennmr.gui.widgets.collapsible_section import CollapsibleSection
    window = MainWindow()
    panel = window.pulse_calculator_panel
    assert isinstance(window._pulse_calculator_section, CollapsibleSection)
    assert window._pulse_calculator_section.isAncestorOf(panel)
    panel.pc_ref_pulse_spin.setValue(2.5)
    panel.pc_ref_power_spin.setValue(100.0)
    panel.pc_ref_unit_combo.setCurrentIndex(panel.pc_ref_unit_combo.findData("W"))
    panel.pc_mode_combo.setCurrentIndex(panel.pc_mode_combo.findData("power"))
    panel.pc_power_spin.setValue(25.0)
    panel.pc_unit_combo.setCurrentIndex(panel.pc_unit_combo.findData("W"))
    result = panel.calculate()
    assert result["nu1_hz"] == pytest.approx(50e3)
    assert "50 kHz" in panel.pc_results_label.text()
    panel.show_profile()
    assert window.simulation_canvas.figure.axes

    # A mixed comparison with no factor is explained, not raised.
    panel.pc_unit_combo.setCurrentIndex(panel.pc_unit_combo.findData("scaled"))
    assert panel.calculate() is None
    assert "scaling factor" in panel.pc_results_label.text()
    panel.pc_scale_edit.setText("4")
    assert panel.calculate()["power_ratio"] == pytest.approx(1.0)   # 25 scaled = 100 W

    # Only the target row for the chosen calculation is shown.
    panel.pc_mode_combo.setCurrentIndex(panel.pc_mode_combo.findData("nu1"))
    assert panel._target_form.isRowVisible(panel.pc_nu1_spin)
    assert not panel._target_form.isRowVisible(panel._power_row)


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])
