"""Notes section 8: selection changes must not leave another dataset's
fit read-outs on screen (8.3)."""
import numpy as np

from opennmr.gui.main_window import MainWindow

from test_gui import (_fit_a_t2_curve, _load_spectrum_directly,  # noqa: F401
                      _make_pseudo2d_relaxation_spectrum, qapp)


def test_selecting_a_dataset_without_a_fit_clears_the_fit_readouts(qapp):
    delays = np.geomspace(0.001, 0.3, 10)
    window = MainWindow()
    _fit_a_t2_curve(window, delays, true_t2=0.05)
    assert window._relaxation_fit_results  # a fit exists for dataset 0

    # Stale read-outs that belong to dataset 0.
    window.shared_shape_status_label.setText("shared shape fit: 3 peaks")
    window.diffusion_select_label.setText("2 components preferred")
    window._diffusion_selection = {"n": 2, "offset": False}
    window.diffusion_apply_selection_button.setEnabled(True)

    # A second, never-fitted dataset.
    other = _make_pseudo2d_relaxation_spectrum(delays, T2=0.1)
    _load_spectrum_directly(window, other)

    assert not window._relaxation_fit_results
    assert window.curve_fit_summary_label.text() == ""
    assert window.shared_shape_status_label.text() == ""
    assert window.diffusion_select_label.text() == ""
    assert window._diffusion_selection is None
    assert not window.diffusion_apply_selection_button.isEnabled()

    # Going back restores dataset 0's own fit.
    window._select_single_row(0)
    assert window._relaxation_fit_results
