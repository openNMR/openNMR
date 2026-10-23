"""Integral labels under the peaks (Integrate tile) and the notebook helper."""
import numpy as np
import pytest

from opennmr.notebook import peak_integral_labels as il

from test_gui import qapp, _load_spectrum_directly  # noqa: F401
from test_integrate_tile_plot_options_regularisation import _one_d


def test_smallest_is_one():
    assert il.normalised_integrals([2.0, 4.0, 6.0]) == pytest.approx([1.0, 2.0, 3.0])
    assert il.normalised_integrals([-2.0, 4.0]) == pytest.approx([1.0, 2.0])


def test_all_zero_is_nan():
    assert all(np.isnan(il.normalised_integrals([0.0, 0.0])))


def test_label_items_text_and_position():
    items = il.label_items([(1.0, 3.0), (5.0, 7.0)], [1.0, 2.5], decimals=1)
    assert items == [(2.0, "1.0"), (6.0, "2.5")]


def test_integral_labels_of_a_spectrum():
    spec = _one_d(height=1.0)  # peaks of height 1.0 and 0.5, equal widths
    items = il.integral_labels(spec, [(4.7, 5.3), (1.7, 2.3)], decimals=2)
    assert [t for _x, t in items] == ["2.00", "1.00"]


def test_add_and_remove_on_axes():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    il.add_integral_labels(ax, [(1.0, "1.00"), (2.0, "2.00")])
    assert len(ax.texts) == 2
    il.add_integral_labels(ax, [(1.0, "1.00")])  # replaces
    assert len(ax.texts) == 1
    il.remove_integral_labels(ax)
    assert not ax.texts
    plt.close(fig)


def _texts(window):
    return [t for t in window.canvas.ax.texts if t.get_gid() == il.GID]


def test_checkbox_labels_the_plot_and_they_persist(qapp):
    from opennmr.gui.main_window import MainWindow
    window = MainWindow()
    _load_spectrum_directly(window, _one_d())
    window._add_region_row(4.7, 5.3)
    window._add_region_row(1.7, 2.3)
    assert not _texts(window)
    window.integral_labels_check.setChecked(True)
    assert [t.get_text() for t in _texts(window)] == ["2.00", "1.00"]
    # Still there after leaving the Integrate tab and redrawing.
    window.side_tabs.setCurrentIndex(0)
    window._draw_selected_spectrum(window.session.spectra[window._selected_index()])
    assert len(_texts(window)) == 2
    window.integral_labels_decimals_spin.setValue(0)
    assert [t.get_text() for t in _texts(window)] == ["2", "1"]
    window.integral_labels_check.setChecked(False)
    assert not _texts(window)
    window.close()
