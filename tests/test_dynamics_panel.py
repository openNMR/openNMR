"""Dynamics panel: step multiplier, ILT button, workflow order, collapsible settings."""
import numpy as np
import pytest

from opennmr.processing.relaxation import start_step_values

from test_gui import qapp, MainWindow, _make_pseudo2d_relaxation_spectrum, _set_delay_unit  # noqa: F401


def test_start_step_values_equal_steps():
    np.testing.assert_allclose(start_step_values(1.0, 0.5, 1.0, 4), [1.0, 1.5, 2.0, 2.5])


def test_start_step_values_doubling_and_fractional():
    np.testing.assert_allclose(start_step_values(0.1, 0.1, 2.0, 5), [0.1, 0.2, 0.4, 0.8, 1.6])
    np.testing.assert_allclose(start_step_values(0.0, 1.0, 1.5, 3), [0.0, 1.0, 2.5])


def test_start_step_values_rejects_nonpositive():
    with pytest.raises(ValueError):
        start_step_values(0.0, 1.0, 0.0, 3)


def test_window_start_step_multiplier(qapp):
    window = MainWindow()
    spectrum = _make_pseudo2d_relaxation_spectrum([0.001, 0.01, 0.1, 1.0])
    window.delay_source_combo.setCurrentIndex(2)
    _set_delay_unit(window, "s")
    window.delay_start_spin.setValue(0.1)
    window.delay_step_spin.setValue(0.1)
    window.delay_multiplier_spin.setValue(1.0)
    np.testing.assert_allclose(
        window._resolve_independent_values(spectrum, is_diffusion=False), [0.1, 0.2, 0.3, 0.4])
    window.delay_multiplier_spin.setValue(2.0)
    np.testing.assert_allclose(
        window._resolve_independent_values(spectrum, is_diffusion=False), [0.1, 0.2, 0.4, 0.8])
    window.delay_multiplier_spin.setValue(1.5)
    np.testing.assert_allclose(
        window._resolve_independent_values(spectrum, is_diffusion=False),
        [0.1, 0.2, 0.35, 0.575])


def test_multiplier_only_visible_for_start_step(qapp):
    window = MainWindow()
    window.delay_source_combo.setCurrentIndex(2)
    assert not window.delay_multiplier_spin.isHidden()
    window.delay_source_combo.setCurrentIndex(1)
    assert window.delay_multiplier_spin.isHidden()


def test_multiplier_is_remembered_with_the_state(qapp):
    window = MainWindow()
    window.delay_source_combo.setCurrentIndex(2)
    window.delay_multiplier_spin.setValue(1.7)
    state = window._capture_analysis_state()
    assert state["step_multiplier"] == pytest.approx(1.7)


def test_ilt_button_and_sections(qapp):
    window = MainWindow()
    assert window.dosy_ilt_button.text().startswith("DOSY Map")
    # Every panel opens with its sections folded.
    for section in (window._curve_settings_section, window._kept_curve_fits_section,
                    window._score_section):
        assert not section.isExpanded()
    window._curve_settings_section.setExpanded(True)
    assert window._curve_settings_section.isExpanded()


def test_results_follow_the_fit_button_and_kept_fits_follow_results(qapp):
    window = MainWindow()
    layout = window.curve_fit_summary_label.parentWidget().layout()

    def row(widget):
        for i in range(layout.count()):
            if layout.itemAt(i).widget() is widget:
                return i
        raise AssertionError("not in layout")

    run = row(window.dosy_ilt_button.parentWidget())
    assert run < row(window.curve_fit_summary_label) < row(window.relaxation_fit_params_label)
    assert row(window.relaxation_fit_params_label) < row(window._kept_curve_fits_section)
    assert row(window._curve_settings_section) < run
    assert row(window._kept_curve_fits_section) < row(window._score_section)
