"""Simulate panel: adiabatic shapes in Shaped pulse (Bloch), and the
Tensor picture boxes of CSA, Quadrupolar and Combined Hamiltonian."""
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from opennmr.gui.main_window import MainWindow


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(qapp):
    w = MainWindow()
    w._show_side_tab("Simulate")
    return w


def _choose(combo, data):
    combo.setCurrentIndex(combo.findData(data))


# -- Bloch -----------------------------------------------------------------------

def test_adiabatic_shapes_are_offered_with_their_own_rows(window):
    shapes = [window.bloch_shape_combo.itemData(i) for i in range(window.bloch_shape_combo.count())]
    assert {"HS", "WURST", "TANH_TAN"} <= set(shapes)
    form = window._bloch_form
    _choose(window.bloch_shape_combo, "WURST")
    assert form.isRowVisible(window.bloch_sweep_spin)
    assert form.isRowVisible(window.bloch_wurst_n_spin)
    assert not form.isRowVisible(window.bloch_hs_beta_spin)
    _choose(window.bloch_shape_combo, "HS")
    assert form.isRowVisible(window.bloch_hs_beta_spin) and form.isRowVisible(window.bloch_hs_n_spin)
    _choose(window.bloch_shape_combo, "GAUSSIAN")
    assert not form.isRowVisible(window.bloch_sweep_spin)


def test_power_mode_follows_the_kind_of_pulse(window):
    # No flip angle for an adiabatic pulse; no Q for an amplitude-modulated one.
    _choose(window.bloch_shape_combo, "HS")
    assert window.bloch_power_mode_combo.currentData() == "q"
    model = window.bloch_power_mode_combo.model()
    assert not model.item(window.bloch_power_mode_combo.findData("flip")).isEnabled()
    _choose(window.bloch_shape_combo, "GAUSSIAN")
    assert window.bloch_power_mode_combo.currentData() == "flip"
    assert not model.item(window.bloch_power_mode_combo.findData("q")).isEnabled()


def test_hs_simulation_reports_q_and_a_power_independent_band(window):
    _choose(window.bloch_shape_combo, "HS")
    window.bloch_duration_spin.setValue(500)
    window.bloch_sweep_spin.setValue(100.0)
    window.bloch_q_spin.setValue(5.0)
    window.bloch_power_scales_edit.setText("1, 2")
    window._run_bloch_simulation()
    text = window.bloch_summary_label.text()
    assert "Q₀ = 5" in text
    assert text.count("inverted (Mz < −0.9)") == 2
    assert "At 2× power" in text
    # The frequency sweep, not the phase, is drawn beside the amplitude.
    labels = [ax.get_ylabel() for ax in window.simulation_canvas.figure.axes]
    assert any("kHz" in label for label in labels)


def test_shape_parameters_reach_the_simulation(window, monkeypatch):
    from opennmr.simulate import bloch
    seen = {}
    real = bloch.simulate_pulse

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return real(*args, **kwargs)
    monkeypatch.setattr(bloch, "simulate_pulse", spy)
    _choose(window.bloch_shape_combo, "TANH_TAN")
    window.bloch_tanh_xi_spin.setValue(8.0)
    window.bloch_tanh_kappa_spin.setValue(15.0)
    window.bloch_sweep_spin.setValue(60.0)
    window.bloch_duration_spin.setValue(300)
    window._run_bloch_simulation()
    assert seen["sweep_width_hz"] == pytest.approx(60e3)
    assert seen["adiabatic_params"] == {"xi": 8.0, "tan_kappa": 15.0}


def test_a_several_mhz_sweep_is_simulated_at_a_finer_time_step(window):
    # Explicit user request (2026-09-30): adiabatic sweeps of several MHz.
    # The box used to stop below 1 MHz, the most a 1 us step can carry at
    # all; now the time step follows the sweep (auto), so the box allows it.
    assert window.bloch_sweep_spin.maximum() >= 20000.0            # kHz: 20 MHz
    _choose(window.bloch_shape_combo, "WURST")
    window.bloch_duration_spin.setValue(200)
    window.bloch_sweep_spin.setValue(4000.0)                       # 4 MHz
    window.bloch_q_spin.setValue(5.0)
    window.bloch_n_offsets_spin.setValue(101)
    window.bloch_power_scales_edit.setText("1")
    window._run_bloch_simulation()
    text = window.bloch_summary_label.text()
    assert "Cannot simulate" not in text
    assert "Time step 0.025 µs (auto" in text                      # 0.05 turn at 2 MHz
    assert "MHz wide" in text and "⚠" not in text
    labels = [ax.get_ylabel() for ax in window.simulation_canvas.figure.axes]
    assert any("MHz" in label for label in labels)
    assert "MHz" in window.simulation_canvas.figure.axes[1].get_xlabel()


def test_a_coarse_time_step_by_hand_is_warned_about(window):
    _choose(window.bloch_shape_combo, "WURST")
    window.bloch_duration_spin.setValue(200)
    window.bloch_sweep_spin.setValue(800.0)                        # 0.4 turn per 1 us step
    window.bloch_time_step_spin.setValue(1.0)
    window.bloch_n_offsets_spin.setValue(51)
    window._run_bloch_simulation()
    text = window.bloch_summary_label.text()
    assert "Time step 1 µs (as set" in text
    assert "too coarse" in text


# -- tensor pictures ------------------------------------------------------------

def test_csa_tensor_surface_on_the_simulation_plot(window):
    window.sim_csa_nucleus_edit.setText("13C")
    window.sim_csa_diso_spin.setValue(120.0)
    controls = window.sim_csa_tensor_controls
    assert controls.show_tensors()
    assert window.canvas_tabs.currentWidget() is window.simulation_canvas
    ax = window.simulation_canvas.figure.axes[0]
    assert ax.name == "3d"                      # rotatable with the mouse
    assert controls.notes and "Radius" in controls.notes[0]


def test_csa_needs_a_nucleus(window):
    window.sim_csa_nucleus_edit.setText("")
    assert not window.sim_csa_tensor_controls.show_tensors()
    assert "nucleus" in window.sim_csa_tensor_controls.notes_label.text()


def test_quadrupolar_tensor_picture_uses_the_panels_field_and_order(window, monkeypatch):
    from opennmr.notebook import tensor_views
    window.sim_quad_nucleus_combo.setCurrentText("27Al")
    window.sim_quad_spin_spin.setValue(2.5)
    window.sim_quad_larmor_spin.setValue(104.3)
    _choose(window.sim_quad_method_combo, "first_order")
    seen = {}
    real = tensor_views.plot_tensors
    monkeypatch.setattr(tensor_views, "plot_tensors",
                        lambda *a, **k: (seen.update(k, larmor=a[1]), real(*a, **k))[1])
    controls = window.sim_quad_tensor_controls
    assert controls.show_tensors()
    assert seen["larmor"] == pytest.approx(104.3e6)
    assert seen["order"] == "first"
    labels = [controls.transition_combo.itemText(i) for i in range(controls.transition_combo.count())]
    assert len(labels) == 5 and controls.transition_combo.currentText() == "−1/2↔+1/2"


def test_combined_tensor_picture_shows_both_tensors_in_one_frame(window):
    panel = window.combined_sim_panel
    panel.nucleus_combo.setCurrentText("27Al")
    panel.larmor_spin.setValue(104.3)
    panel.set_state({"widgets": {}, "current": 0, "sites": [{
        "label": "Al(1)", "delta_iso": 10.0, "amount": 1.0, "fwhm": 1.0,
        "lineshape": "gaussian", "eta": 0.5, "interactions": [
            {"type": "quadrupolar", "C_Q": 4.0, "eta_Q": 0.3},
            {"type": "csa", "aniso": 60.0, "asym": 0.2, "convention": "Haeberlen",
             "euler": [0.0, 40.0, 0.0]},
            {"type": "jcoupling", "nuclide": "31P", "n": 1, "J_Hz": 15.0}]}]})
    controls = window.combined_tensor_controls
    _choose(controls.style_combo, "ellipsoid")
    assert not controls._form.isRowVisible(controls.transition_combo)
    assert controls.show_tensors()
    text = " ".join(controls.notes)
    assert "traceless" in text and "Euler" in text and "31P" in text
    _choose(controls.style_combo, "surface")
    _choose(controls.relative_combo, "iso")
    assert controls.show_tensors()
    assert "Al(1)" in window.simulation_canvas.figure._suptitle.get_text()
