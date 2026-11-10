"""Insets on any plot: the matplotlib helper, the toolbar tool, the dialog."""
from types import SimpleNamespace

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from opennmr.plotting import inset as ins

from test_gui import qapp, _load_spectrum_directly  # noqa: F401
from test_integrate_tile_plot_options_regularisation import _one_d, _series


def _plot():
    fig, ax = plt.subplots()
    x = np.linspace(10, 0, 2000)
    y = np.exp(-0.5 * ((x - 5) / 0.05) ** 2) + 0.3 * np.exp(-0.5 * ((x - 4.8) / 0.03) ** 2)
    ax.plot(x, y, color="k")
    ax.fill_between(x, 0, y, alpha=0.3)
    ax.scatter(x[::100], y[::100])
    ax.set_xlim(10, 0)
    return fig, ax


# ---- the matplotlib helper ---------------------------------------------------

def test_inset_copies_the_data_and_keeps_the_axis_direction():
    fig, ax = _plot()
    inset = ins.add_inset(ax, (5.2, 4.6), (0, 1.1))
    assert inset.get_xlim() == (5.2, 4.6)
    assert len(inset.get_lines()) == 1 and len(inset.collections) == 2
    assert ins.insets_of(ax) == [inset]
    plt.close(fig)


def test_axes_off_leaves_only_the_peaks():
    fig, ax = _plot()
    inset = ins.add_inset(ax, (5.2, 4.6), None, show_axes=False)
    assert len(inset.get_xticks()) == 0 and len(inset.get_yticks()) == 0
    shown = ins.add_inset(ax, (5.2, 4.6), None, show_axes=True)
    assert len(shown.get_xticks()) > 0
    plt.close(fig)


def test_connectors_can_be_switched_off():
    fig, ax = _plot()
    ins.add_inset(ax, (5.2, 4.6), None, connectors=False)
    assert ax._opennmr_insets[0]["indicator"] is None
    ins.remove_insets(ax)
    ins.add_inset(ax, (5.2, 4.6), None, connectors=True)
    assert ax._opennmr_insets[0]["indicator"] is not None
    plt.close(fig)


def test_typeface_and_style_are_applied():
    fig, ax = _plot()
    inset = ins.add_inset(ax, (5.2, 4.6), None, family="serif", bold=True, italic=True,
                          fontsize=7)
    fig.canvas.draw()
    label = inset.get_xticklabels()[0]
    assert label.get_fontweight() == "bold" and label.get_fontstyle() == "italic"
    assert label.get_fontfamily() == ["serif"]
    assert label.get_fontsize() == 7
    plt.close(fig)


def test_position_corner_and_custom():
    assert ins.inset_bounds(ins.normalise(corner="upper left", width=0.3, height=0.2,
                                          margin=0.05)) == pytest.approx((0.05, 0.75, 0.3, 0.2))
    assert ins.inset_bounds(ins.normalise(corner="custom", x=0.1, y=0.2, width=0.5,
                                          height=0.4)) == pytest.approx((0.1, 0.2, 0.5, 0.4))


def test_auto_vertical_range_fits_the_data():
    fig, ax = _plot()
    inset = ins.add_inset(ax, (5.3, 4.9), None, auto_ylim=True)
    low, high = inset.get_ylim()
    assert low < 0.01 and 0.95 < high < 1.2
    plt.close(fig)


def test_bad_options_are_errors():
    fig, ax = _plot()
    with pytest.raises(ValueError):
        ins.add_inset(ax, (5.2, 4.6), None, colour="red")       # misspelt key
    with pytest.raises(ValueError):
        ins.add_inset(ax, (5.2, 4.6), None, width=2.0)
    with pytest.raises(ValueError):
        ins.add_inset(ax, None, None)
    with pytest.raises(ValueError):
        ins.add_inset(ax, (5.0, 5.0), None)
    plt.close(fig)


def test_remove_and_replace():
    fig, ax = _plot()
    ins.draw_insets(ax, [{"xlim": (5.2, 4.6), "ylim": (0, 1)},
                         {"xlim": (3, 2), "ylim": (0, 1), "corner": "lower left"}])
    assert len(ax.child_axes) == 2
    ins.draw_insets(ax, [{"xlim": (5.2, 4.6), "ylim": (0, 1)}])
    assert len(ax.child_axes) == 1
    assert ins.remove_insets(ax) == 1 and not ax.child_axes
    plt.close(fig)


def test_vector_export_has_the_inset(tmp_path):
    fig, ax = _plot()
    ins.add_inset(ax, (5.2, 4.6), None)
    out = tmp_path / "inset.pdf"
    fig.savefig(out)
    assert out.stat().st_size > 1000
    svg = tmp_path / "inset.svg"
    fig.savefig(svg)
    assert svg.read_text(encoding="utf-8").count("<path") > 5
    plt.close(fig)


# ---- the toolbar tool ----------------------------------------------------------

def _drag(canvas, ax, x0, x1, y0, y1):
    """A box dragged on ``ax`` between two data points."""
    p0 = ax.transData.transform((x0, y0))
    p1 = ax.transData.transform((x1, y1))
    canvas._inset_start = (p0[0], p0[1], ax)
    canvas._finish_inset_box(SimpleNamespace(x=p1[0], y=p1[1]))


def test_inset_button_draws_a_box_and_adds_an_inset(qapp):
    from opennmr.gui.main_window import MainWindow
    window = MainWindow()
    _load_spectrum_directly(window, _one_d())
    canvas = window.canvas
    assert canvas._inset_button.text() == "Inset"
    added = []
    canvas.insetAdded.connect(added.append)
    canvas.set_inset_mode(True)
    assert canvas._inset_button.isChecked()
    _drag(canvas, canvas.ax, 5.3, 4.7, 0.0, 1.0)
    assert added == [0]
    assert not canvas._inset_mode                         # one box, then the tool lets go
    canvas.canvas.draw()
    assert len(canvas.ax.child_axes) == 1
    spec = canvas.inset_specs()[0]
    assert spec["xlim"][0] > spec["xlim"][1]              # still an NMR axis
    window.close()


def test_a_click_is_not_a_box(qapp):
    from opennmr.gui.main_window import MainWindow
    window = MainWindow()
    _load_spectrum_directly(window, _one_d())
    canvas = window.canvas
    canvas.set_inset_mode(True)
    _drag(canvas, canvas.ax, 5.0, 5.0, 0.5, 0.5)
    assert canvas.inset_specs() == [] and canvas._inset_mode
    canvas.set_inset_mode(False)
    window.close()


def test_inset_mode_is_exclusive_with_the_other_tools(qapp):
    from opennmr.gui.main_window import MainWindow
    window = MainWindow()
    _load_spectrum_directly(window, _one_d())
    canvas = window.canvas
    canvas.set_inset_mode(True)
    canvas.set_width_mode(True)
    assert not canvas._inset_mode and not canvas._inset_button.isChecked()
    canvas.set_inset_mode(True)
    assert not canvas._width_mode
    canvas.set_inset_mode(False)
    window.close()


def test_insets_come_back_after_a_redraw_and_follow_the_spectrum(qapp):
    from opennmr.gui.main_window import MainWindow
    window = MainWindow()
    _load_spectrum_directly(window, _one_d())
    canvas = window.canvas
    canvas.add_inset_region((5.3, 4.7), (0.0, 1.0))
    canvas.canvas.draw()
    spectrum = window.session.spectra[window._selected_index()]
    window._draw_selected_spectrum(spectrum)               # a fresh set of axes
    canvas.canvas.draw()
    assert len(canvas.ax.child_axes) == 1
    # Another spectrum with the same axes does not inherit it.
    _load_spectrum_directly(window, _one_d(height=2.0))
    canvas.canvas.draw()
    assert len(canvas.ax.child_axes) == 0
    window._select_single_row(0)
    canvas.canvas.draw()
    assert len(canvas.ax.child_axes) == 1
    window.close()


def test_inset_follows_changed_data(qapp):
    from opennmr.gui.main_window import MainWindow
    window = MainWindow()
    _load_spectrum_directly(window, _one_d())
    canvas = window.canvas
    canvas.add_inset_region((5.3, 4.7), (0.0, 1.0))
    canvas.canvas.draw()
    inset = canvas.ax.child_axes[0]
    before = float(np.max(inset.get_lines()[0].get_ydata()))
    line = canvas.ax.get_lines()[0]
    line.set_ydata(np.asarray(line.get_ydata()) * 2.0)     # re-processed in place
    canvas.canvas.draw()
    canvas.canvas.draw()
    after = float(np.max(canvas.ax.child_axes[0].get_lines()[0].get_ydata()))
    assert after == pytest.approx(2.0 * before)
    window.close()


def test_pseudo_2d_slice_and_curve_plots_take_insets(qapp):
    from opennmr.gui.main_window import MainWindow
    window = MainWindow()
    canvas = window.fit_canvas

    def draw(figure):
        ax = figure.add_subplot(111)
        t = np.linspace(0, 1, 50)
        ax.plot(t, np.exp(-t / 0.2), "o-")
        ax.set_xlabel("delay / s")

    canvas.show_external_plot(draw, kind="curve")
    canvas.canvas.draw()
    assert canvas.insets_supported()
    canvas.add_inset_region((0.0, 0.3), (0.2, 1.1))
    canvas.canvas.draw()
    assert len(canvas._plain_axes()[0].child_axes) == 1
    window.close()


def test_2d_contour_inset_uses_the_dialog_settings(qapp, tmp_path):
    from test_compare_2d import _bruker_2d
    from opennmr.gui.main_window import MainWindow
    window = MainWindow()
    window._load(_bruker_2d(tmp_path, "a"), vendor="bruker")
    canvas = window.canvas
    (x0, x1), (y0, y1) = canvas.contour_view_region()
    spec = canvas.add_inset_region((x0 - 0.3 * (x0 - x1), x1 + 0.3 * (x0 - x1)),
                                    (y0 + 0.3 * (y1 - y0), y1 - 0.3 * (y1 - y0)),
                                    show_axes=False, connectors=False)
    assert spec["show_axes"] is False
    canvas.canvas.draw()
    assert len(canvas.contour_insets()) == 1
    assert canvas.contour_insets()[0]["mark_inset"] is False
    window.close()


# ---- the dialog -------------------------------------------------------------

def test_dialog_edits_the_inset(qapp):
    from opennmr.gui.main_window import MainWindow
    window = MainWindow()
    _load_spectrum_directly(window, _one_d())
    canvas = window.canvas
    canvas.add_inset_region((5.3, 4.7), (0.0, 1.0))
    dialog = window._on_open_inset_dialog(canvas)
    assert dialog.inset_list.count() == 1
    dialog.show_axes_check.setChecked(False)
    dialog.connectors_check.setChecked(False)
    dialog.corner_combo.setCurrentText("custom")
    dialog.pos_x_spin.setValue(20)
    dialog.bold_check.setChecked(True)
    dialog.apply()
    spec = canvas.inset_specs()[0]
    assert spec["show_axes"] is False and spec["connectors"] is False
    assert spec["corner"] == "custom" and spec["x"] == pytest.approx(0.2)
    canvas.canvas.draw()
    assert len(canvas.ax.child_axes[0].get_xticks()) == 0
    dialog.clear_insets()
    assert canvas.inset_specs() == []
    canvas.canvas.draw()
    assert not canvas.ax.child_axes
    window.close()


def test_dialog_add_from_zoom_and_reachable_from_the_plot(qapp):
    from opennmr.gui.main_window import MainWindow
    window = MainWindow()
    _load_spectrum_directly(window, _one_d())
    canvas = window.canvas
    dialog = window._on_open_inset_dialog(canvas)
    assert dialog.add_from_view() is None                  # not zoomed
    canvas.ax.set_xlim(5.4, 4.6)
    assert dialog.add_from_view() is not None
    assert len(canvas.inset_specs()) == 1
    assert "inset" in canvas.current_plot_owners()
    # Insets are a section of Plot Style now (2026-11-10), not a separate window.
    assert "Insets" in window.style_sections and hasattr(window, "style_inset_panel")
    window.close()


# ---- notebook --------------------------------------------------------------

def test_notebook_add_inset_on_figure_or_axes():
    import opennmr.notebook as nb
    fig, ax = _plot()
    inset = nb.add_inset(fig, (5.3, 4.9))              # a figure: its first axes; y fitted
    assert inset.get_ylim()[1] > 0.9
    assert len(nb.insets_of(ax)) == 1
    nb.remove_insets(ax)
    nb.add_inset(ax, (5.3, 4.9), (0, 1), corner="lower left", show_axes=False)
    assert len(nb.insets_of(ax)) == 1
    assert "show_axes" in nb.INSET_DEFAULTS
    plt.close(fig)


def test_export_draws_the_inset_from_full_resolution_data(qapp):
    from opennmr.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
    from opennmr.gui.main_window import MainWindow
    n = 400_000
    ppm = np.linspace(10.0, 0.0, n)
    y = np.exp(-0.5 * ((ppm - 5.0) / 0.002) ** 2)
    spectrum = NMRSpectrum(data=y, axes=[NMRAxis(kind="frequency", ppm=ppm, hz=ppm * 400.0)],
                           metadata=NMRMetadata(vendor="bruker", nucleus="1H", sfo1=400.0))
    window = MainWindow()
    _load_spectrum_directly(window, spectrum)
    canvas = window.canvas
    canvas.set_full_resolution(False)                        # the live view thins long lines
    canvas.canvas.draw()
    live_points = len(canvas.ax.get_lines()[0].get_xdata())
    assert live_points < n                                  # the live view is thinned
    canvas.add_inset_region((5.01, 4.99), (0.0, 1.0))
    canvas.canvas.draw()
    seen = {}

    def save():
        inset = canvas.ax.child_axes[0]
        seen["points"] = max(len(line.get_xdata()) for line in inset.get_lines())

    canvas.export_full_resolution(save)
    assert seen["points"] == n
    canvas.canvas.draw()
    canvas.canvas.draw()
    assert len(canvas.ax.child_axes) == 1                    # and back to the live view
    window.close()


# ---- y-values, axis labels and Ctrl/Cmd+I (explicit user request, 2026-10-24) ------

def _labelled_plot():
    fig, ax = _plot()
    ax.set_xlabel("δ(¹H) / ppm")
    ax.set_ylabel("Intensity")
    return fig, ax


def test_line_inset_never_shows_y_values():
    fig, ax = _labelled_plot()
    shown = ins.add_inset(ax, (5.2, 4.6), None, show_axes=True)
    assert len(shown.get_xticks()) > 0                 # the x axis keeps its numbers
    assert len(shown.get_yticks()) == 0
    assert not any(t.get_text() for t in shown.get_yticklabels())
    plt.close(fig)


def test_axis_labels_are_off_by_default_and_optional():
    fig, ax = _labelled_plot()
    plain = ins.add_inset(ax, (5.2, 4.6), None)
    assert plain.get_xlabel() == "" and plain.get_ylabel() == ""
    both = ins.add_inset(ax, (5.2, 4.6), None, corner="lower left",
                         show_xlabel=True, show_ylabel=True)
    assert both.get_xlabel() == "δ(¹H) / ppm" and both.get_ylabel() == "Intensity"
    own = ins.add_inset(ax, (5.2, 4.6), None, show_xlabel=True, xlabel="shift", show_ylabel=True,
                        ylabel="signal", family="serif", bold=True, italic=True, fontsize=8)
    assert own.get_xlabel() == "shift" and own.get_ylabel() == "signal"
    label = own.xaxis.get_label()
    assert label.get_fontsize() == 8 and label.get_fontweight() == "bold"
    assert label.get_fontstyle() == "italic" and label.get_fontfamily() == ["serif"]
    # Only the x label, only the y label.
    only_x = ins.add_inset(ax, (5.2, 4.6), None, show_xlabel=True)
    assert only_x.get_xlabel() and not only_x.get_ylabel()
    # The labels are there with the axes off too.
    bare = ins.add_inset(ax, (5.2, 4.6), None, show_axes=False, show_xlabel=True)
    assert bare.get_xlabel() and len(bare.get_xticks()) == 0
    # They sit on the side facing into the plot (an inset in a lower corner: on top).
    assert both.xaxis.get_label_position() == "top"
    assert own.xaxis.get_label_position() == "bottom"
    fig.canvas.draw()
    plt.close(fig)


def test_2d_inset_keeps_its_y_values_and_takes_labels():
    from opennmr.plotting import twod
    fig, ax = plt.subplots()
    # restyle is what a 2D contour inset goes through: the F1 numbers stay.
    inset = ax.inset_axes((0.5, 0.5, 0.4, 0.4))
    inset.plot([0, 1, 2], [0, 1, 4])
    inset.set_ylim(0, 4)
    ax.set_xlabel("F2 / ppm")
    ax.set_ylabel("F1 / ppm")
    ins.restyle(inset, {"show_axes": True, "show_xlabel": True, "show_ylabel": True},
                ax, (0.5, 0.5, 0.4, 0.4))
    fig.canvas.draw()
    assert len(inset.get_yticks()) > 0 and any(t.get_text() for t in inset.get_yticklabels())
    assert inset.get_xlabel() == "F2 / ppm" and inset.get_ylabel() == "F1 / ppm"
    off = ax.inset_axes((0.1, 0.1, 0.3, 0.3))
    ins.restyle(off, {"show_axes": True}, ax, (0.1, 0.1, 0.3, 0.3))
    assert off.get_xlabel() == "" and off.get_ylabel() == ""
    assert twod  # the contour inset calls restyle with the parent and its bounds
    plt.close(fig)


def test_notebook_inset_takes_label_options():
    import opennmr.notebook as nb
    fig, ax = _labelled_plot()
    inset = nb.add_inset(ax, xlim=(5.2, 4.6), show_xlabel=True, show_ylabel=True)
    assert inset.get_xlabel() and inset.get_ylabel()
    assert not inset.get_yticks().size
    assert {"show_xlabel", "show_ylabel", "xlabel", "ylabel"} <= set(nb.INSET_DEFAULTS)
    plt.close(fig)


def test_dialog_controls_axis_labels(qapp):
    from opennmr.gui.main_window import MainWindow
    window = MainWindow()
    _load_spectrum_directly(window, _one_d())
    canvas = window.canvas
    canvas.ax.set_xlabel("δ / ppm")
    canvas.add_inset_region((5.3, 4.7), (0.0, 1.0))
    dialog = window._on_open_inset_dialog(canvas)
    assert not dialog.show_xlabel_check.isChecked() and not dialog.show_ylabel_check.isChecked()
    assert not dialog.xlabel_edit.isEnabled()
    dialog.show_xlabel_check.setChecked(True)
    assert dialog.xlabel_edit.isEnabled()
    dialog.show_ylabel_check.setChecked(True)
    dialog.ylabel_edit.setText("signal")
    dialog.ylabel_edit.textEdited.emit("signal")
    dialog.apply()
    spec = canvas.inset_specs()[0]
    assert spec["show_xlabel"] is True and spec["xlabel"] is None
    assert spec["show_ylabel"] is True and spec["ylabel"] == "signal"
    canvas.canvas.draw()
    inset = canvas.ax.child_axes[0]
    assert inset.get_xlabel() == canvas.ax.get_xlabel() and inset.get_ylabel() == "signal"
    assert len(inset.get_yticks()) == 0
    dialog.show_xlabel_check.setChecked(False)
    dialog.apply()
    canvas.canvas.draw()
    assert canvas.ax.child_axes[0].get_xlabel() == ""
    window.close()


def test_cmd_i_starts_the_inset_tool(qapp):
    from opennmr.gui.main_window import MainWindow
    window = MainWindow()
    assert window.inset_menu_action.shortcut().toString() in ("Ctrl+I", "⌘I")
    # Region selection moved to Ctrl+Shift+I, so the two do not collide.
    assert window.select_region_menu_action.shortcut() != window.inset_menu_action.shortcut()
    # Each of the two keys belongs to one action only.
    keys = [a.shortcut().toString() for a in window.findChildren(type(window.inset_menu_action))
            if not a.shortcut().isEmpty()]
    for key in (window.inset_menu_action.shortcut().toString(),
                window.select_region_menu_action.shortcut().toString()):
        assert keys.count(key) == 1
    _load_spectrum_directly(window, _one_d())
    canvas = window.canvas
    window.canvas_tabs.setCurrentWidget(canvas)
    assert not canvas._inset_mode
    window.inset_menu_action.trigger()
    assert canvas._inset_mode
    window.inset_menu_action.trigger()                  # again: cancels
    assert not canvas._inset_mode
    window.close()
