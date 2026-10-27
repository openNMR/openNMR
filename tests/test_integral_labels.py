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
    assert items == [(2.0, "\u222b = 1.0"), (6.0, "\u222b = 2.5")]
    # The bare number, or another symbol, on request.
    assert il.label_items([(1.0, 3.0)], [1.0], decimals=1, symbol=False) == [(2.0, "1.0")]
    assert il.label_items([(1.0, 3.0)], [1.0], decimals=1, symbol="I") == [(2.0, "I = 1.0")]
    assert il.label_text(float("nan")) == "\u222b = \u2013"


def test_integral_labels_of_a_spectrum():
    spec = _one_d(height=1.0)  # peaks of height 1.0 and 0.5, equal widths
    items = il.integral_labels(spec, [(4.7, 5.3), (1.7, 2.3)], decimals=2)
    assert [t for _x, t in items] == ["\u222b = 2.00", "\u222b = 1.00"]
    bare = il.integral_labels(spec, [(4.7, 5.3), (1.7, 2.3)], decimals=2, symbol=False)
    assert [t for _x, t in bare] == ["2.00", "1.00"]


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


def test_label_text_style_in_the_notebook():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    artists = il.add_integral_labels(ax, [(1.0, "\u222b = 1.00")], fontsize=9, color="tab:red",
                                     family="Times New Roman", bold=True, italic=True,
                                     offset_axes=0.05, rotation=90)
    t = artists[0]
    assert t.get_fontsize() == 9 and t.get_color() == "tab:red"
    assert t.get_fontweight() == "bold" and t.get_fontstyle() == "italic"
    assert t.get_rotation() == 90 and t.get_position()[1] == 0.05
    # A named typeface falls back to DejaVu Sans for glyphs it lacks (the integral sign).
    assert t.get_fontfamily() == ["Times New Roman", "DejaVu Sans"]
    # Left alone, the text is ordinary: not bold, not italic, horizontal.
    plain = il.add_integral_labels(ax, [(1.0, "x")])[0]
    # "regular" and "normal" are the same weight (400); setup_plotting()
    # sets rcParams' font.weight to "regular", so which one is reported
    # depends on whether an earlier test in this process called it.
    assert plain.get_fontweight() in ("normal", "regular", 400)
    assert plain.get_fontstyle() == "normal"
    assert plain.get_rotation() == 0
    fig.canvas.draw()          # renders without a missing-glyph failure
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
    assert [t.get_text() for t in _texts(window)] == ["\u222b = 2.00", "\u222b = 1.00"]
    # Still there after leaving the Integrate tab and redrawing.
    window.side_tabs.setCurrentIndex(0)
    window._draw_selected_spectrum(window.session.spectra[window._selected_index()])
    assert len(_texts(window)) == 2
    window.integral_labels_decimals_spin.setValue(0)
    assert [t.get_text() for t in _texts(window)] == ["\u222b = 2", "\u222b = 1"]
    window.integral_labels_symbol_check.setChecked(False)
    assert [t.get_text() for t in _texts(window)] == ["2", "1"]
    window.integral_labels_check.setChecked(False)
    assert not _texts(window)
    window.close()


def test_label_style_controls_restyle_the_text(qapp):
    from opennmr.gui.main_window import MainWindow
    window = MainWindow()
    _load_spectrum_directly(window, _one_d())
    window._add_region_row(4.7, 5.3)
    window.integral_labels_check.setChecked(True)
    tick_size = window.canvas.ax.xaxis.get_ticklabels()[0].get_fontsize()
    assert _texts(window)[0].get_fontsize() == tick_size          # "Match axes"
    assert _texts(window)[0].get_fontweight() in ("normal", "regular", 400)   # the same weight
    window.integral_labels_size_spin.setValue(14)
    window._integral_label_colour = "#cc0000"
    window.integral_labels_bold_check.setChecked(True)
    window.integral_labels_italic_check.setChecked(True)
    window.integral_labels_angle_spin.setValue(90)
    window.integral_labels_height_spin.setValue(5.0)
    family = "serif"
    window.integral_labels_font_combo.setCurrentIndex(window.integral_labels_font_combo.findData(family))
    text = _texts(window)[0]
    assert text.get_fontsize() == 14 and text.get_fontweight() == "bold"
    assert text.get_fontstyle() == "italic" and text.get_rotation() == 90
    assert text.get_position()[1] == pytest.approx(0.05)
    assert text.get_fontfamily()[0] == "serif"
    # The colour button picks it too.
    window._sync_integral_labels()
    from matplotlib.colors import to_hex
    assert to_hex(_texts(window)[0].get_color()) == "#cc0000"
    # And the style survives a redraw of the plot.
    window._draw_selected_spectrum(window.session.spectra[window._selected_index()])
    again = _texts(window)[0]
    assert again.get_fontsize() == 14 and again.get_fontweight() == "bold"
    window.close()
