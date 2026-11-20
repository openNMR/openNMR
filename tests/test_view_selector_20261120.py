"""Notes 8.2: one slice selector at the top right of the plot area."""
import numpy as np
from PySide6.QtWidgets import QSpinBox

from opennmr.gui.widgets.view_selector import MirroredSpinBar
from opennmr.gui.main_window import MainWindow
from test_gui import (_fit_a_t2_curve, _load_spectrum_directly,  # noqa: F401
                      _make_pseudo2d_relaxation_spectrum, qapp)


# ---- the widget ----------------------------------------------------------------------

def test_the_bar_mirrors_its_target_both_ways(qapp):
    target = QSpinBox()
    target.setRange(1, 8)
    target.setValue(3)
    bar = MirroredSpinBar()
    bar.bind(target, "Slice:")
    assert (bar.spin.minimum(), bar.spin.maximum(), bar.spin.value()) == (1, 8, 3)
    target.setValue(5)
    assert bar.spin.value() == 5
    bar.spin.setValue(7)
    assert target.value() == 7
    bar.next_button.click()
    assert target.value() == 8
    bar.next_button.click()                      # clamped at the end, not wrapped or errored
    assert target.value() == 8


def test_rebinding_stops_following_the_old_target(qapp):
    a, b = QSpinBox(), QSpinBox()
    for spin, top in ((a, 4), (b, 9)):
        spin.setRange(1, top)
    bar = MirroredSpinBar()
    bar.bind(a)
    bar.bind(b)
    a.setValue(3)
    assert bar.spin.value() == 1                 # b's, untouched by a
    assert bar.spin.maximum() == 9
    bar.spin.setValue(6)
    assert b.value() == 6 and a.value() == 3


def test_a_disabled_target_disables_the_bar(qapp):
    target = QSpinBox()
    target.setRange(1, 4)
    target.setEnabled(False)
    bar = MirroredSpinBar()
    bar.bind(target)
    assert not bar.isEnabled()


# ---- in the window -------------------------------------------------------------------

def _window_with_series(n=6):
    window = MainWindow()
    window.show()
    _load_spectrum_directly(window, _make_pseudo2d_relaxation_spectrum(np.geomspace(0.001, 0.3, n)))
    return window


def test_the_process_views_get_the_slice_selector_at_the_top_right(qapp):
    window = _window_with_series(6)
    window.canvas_tabs.setCurrentWidget(window.processing_preview_canvas)
    qapp.processEvents()
    bar = window.view_slice_bar
    assert bar.isVisibleTo(window.view_selector_bar)
    assert bar.target is window.slice_spin
    assert bar.spin.maximum() == 6
    bar.spin.setValue(4)
    assert window.slice_spin.value() == 4        # drives the panel's own control
    window.slice_spin.setValue(2)
    assert bar.spin.value() == 2                 # and follows it


def test_the_spectral_fit_view_gets_the_same_selector(qapp):
    window = _window_with_series(5)
    window.view_slice_fit_spin.blockSignals(True)
    window.view_slice_fit_spin.setEnabled(True)
    window.view_slice_fit_spin.setMaximum(5)
    window.view_slice_fit_spin.blockSignals(False)
    window.canvas_tabs.setCurrentWidget(window.spectral_fit_canvas)
    qapp.processEvents()
    bar = window.view_slice_bar
    assert bar.isVisibleTo(window.view_selector_bar)
    assert bar.target is window.view_slice_fit_spin and bar.spin.maximum() == 5


def test_the_curve_fit_peak_selector_shares_the_corner(qapp):
    window = _window_with_series(8)
    assert window.canvas_tabs.cornerWidget() is window.view_selector_bar
    assert window.curve_peak_bar.parentWidget() is window.view_selector_bar
    assert window.view_slice_bar.parentWidget() is window.view_selector_bar


def test_no_selector_for_a_1d_spectrum_or_an_unrelated_view(qapp):
    window = _window_with_series(6)
    window.canvas_tabs.setCurrentWidget(window.compare_canvas)
    qapp.processEvents()
    assert not window.view_slice_bar.isVisibleTo(window.view_selector_bar)
