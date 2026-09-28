"""Legend key line width, names written on the traces instead of a
legend, and DNP spectra drawn as 1D spectra (explicit user requests,
2026-09-28)."""
from types import SimpleNamespace

import matplotlib
import numpy as np
import pytest
from matplotlib.figure import Figure
from PySide6.QtWidgets import QApplication

from opennmr.plotting import label_traces, set_legend_linewidth
from test_dnp import ERRORS, FIELDS, _sweep


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _noisy(baseline=5.0, sigma=0.5, seed=0, peak_at=None):
    rng = np.random.default_rng(seed)
    x = np.linspace(100, 0, 4000)
    y = baseline + rng.normal(0, sigma, x.size)
    if peak_at is not None:
        y += 100 * np.exp(-((x - peak_at) / 0.5) ** 2)
    return x, y


def _agg_axes():
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig = Figure(figsize=(6, 4), dpi=100)
    FigureCanvasAgg(fig)
    return fig, fig.add_subplot()


# -- notebook-level helpers -----------------------------------------------------------

def test_label_sits_just_above_the_noise():
    fig, ax = _agg_axes()
    x, y = _noisy(baseline=5.0, sigma=0.5, peak_at=80)
    ax.plot(x, y, color="#123456", label="sample")
    ax.set_xlim(100, 0)
    (text,) = label_traces(ax, position=0.5, n_sigma=3.0)
    assert text.get_text() == "sample"
    assert text.get_color() == "#123456"
    # baseline + 3 sigma, not pulled up by the peak far away.
    assert text.get_position()[1] == pytest.approx(5.0 + 1.5, abs=0.3)


@pytest.mark.parametrize("position", [0.0, 0.5, 1.0])
def test_label_position_runs_left_edge_to_right_edge(position):
    fig, ax = _agg_axes()
    x, y = _noisy()
    ax.plot(x, y, label="a fairly long sample name")
    ax.set_xlim(100, 0)
    (text,) = label_traces(ax, position=position)
    fig.canvas.draw()
    box = text.get_window_extent()
    axes_box = ax.get_window_extent()
    if position == 0.0:
        assert box.x0 == pytest.approx(axes_box.x0, abs=2)
    elif position == 1.0:
        assert box.x1 == pytest.approx(axes_box.x1, abs=2)
    else:
        assert (box.x0 + box.x1) / 2 == pytest.approx((axes_box.x0 + axes_box.x1) / 2, abs=2)


def test_overlaid_labels_do_not_collide():
    fig, ax = _agg_axes()
    for k in range(3):
        x, y = _noisy(seed=k)
        ax.plot(x, y, label=f"trace {k}")
    texts = label_traces(ax, position=0.1)
    fig.canvas.draw()
    boxes = sorted((t.get_window_extent() for t in texts), key=lambda b: b.y0)
    for lower, upper in zip(boxes, boxes[1:]):
        assert upper.y0 >= lower.y1 - 1


def test_label_names_may_be_latex_and_bad_latex_does_not_break_the_figure():
    fig, ax = _agg_axes()
    x, y = _noisy()
    ax.plot(x, y)
    ax.plot(x, y + 20)
    texts = label_traces(ax, ax.get_lines(), [r"$^{7}$Li", r"$^{7$Li"])
    fig.canvas.draw()
    assert texts[0].get_text() == r"$^{7}$Li"


def test_set_legend_linewidth():
    fig, ax = _agg_axes()
    ax.plot([0, 1], linewidth=0.5, label="a")
    legend = ax.legend()
    set_legend_linewidth(legend, 4.0)
    assert [ln.get_linewidth() for ln in legend.get_lines()] == [4.0]
    set_legend_linewidth(legend, None)          # no-op
    assert legend.get_lines()[0].get_linewidth() == 4.0


# -- the GUI -----------------------------------------------------------------------

@pytest.fixture
def window(qapp):
    from opennmr.gui.main_window import MainWindow
    w = MainWindow()
    w.resize(1400, 900)
    w.show()
    for k, s in enumerate(_sweep(FIELDS, ERRORS)[:3]):
        w._add_derived_spectrum(s, f"s{k}")
    w._rebuild_compare_list([(0, None), (1, None), (2, None)], preserve_previous=False)
    return w


def _compare(window, mode_index=0):
    window.layout_mode_combo.setCurrentIndex(mode_index)
    window._refresh_compare_view(switch_tab=True, force_render=True)
    QApplication.processEvents()
    return window.compare_canvas


def _legends(canvas):
    return [ax.get_legend() for ax in canvas.figure.get_axes() if ax.get_legend()]


def test_legend_line_width_is_independent_of_the_traces(window):
    canvas = _compare(window, 1)
    canvas.set_style(linewidth=1.0, legend_linewidth=4.5)
    (legend,) = _legends(canvas)
    assert {ln.get_linewidth() for ln in legend.get_lines()} == {4.5}
    assert all(ln.get_linewidth() != 4.5 for ln in canvas.ax.lines)
    # Survives a restyle and a redraw.
    canvas.set_style(labelsize=14)
    window._refresh_compare_view(switch_tab=True, force_render=True)
    assert {ln.get_linewidth() for ln in _legends(canvas)[0].get_lines()} == {4.5}
    canvas.set_style(legend_linewidth=0)          # back to matching the traces
    assert canvas.style_settings()["legend_linewidth"] is None


@pytest.mark.parametrize("mode_index", [0, 1])     # stacked, overlaid
def test_names_on_the_traces_instead_of_a_legend(window, mode_index):
    canvas = _compare(window, mode_index)
    canvas.set_style(legend_style="labels", trace_label_position=0.9)
    assert _legends(canvas) == []
    texts = canvas.trace_labels()
    assert [t.get_text() for t in texts] and len(texts) == 3
    series_colours = {matplotlib.colors.to_hex(e.colour) for e in canvas.series()}
    assert {matplotlib.colors.to_hex(t.get_color()) for t in texts} == series_colours
    canvas.canvas.draw()
    # Back to a legend.
    canvas.set_style(legend_style="legend")
    assert canvas.trace_labels() == [] and len(_legends(canvas)) == 1


def test_trace_names_survive_a_redraw_and_follow_a_zoom(window):
    canvas = _compare(window, 1)
    canvas.set_style(legend_style="labels", trace_label_position=0.0)
    window._refresh_compare_view(switch_tab=True, force_render=True)
    assert len(canvas.trace_labels()) == 3
    before = canvas.trace_labels()[0]
    canvas.ax.set_xlim(*reversed(sorted(canvas.ax.get_xlim())[:1] + [0.0]))
    assert canvas.trace_labels() and canvas.trace_labels()[0] is not before   # re-placed


def test_hidden_names_are_hidden_in_either_style(window):
    canvas = _compare(window)
    canvas.set_style(legend_style="labels")
    canvas.set_legend_visible(False)
    assert canvas.trace_labels() == [] and _legends(canvas) == []


def test_plot_style_controls_reach_the_canvas_and_read_back(window):
    canvas = _compare(window, 1)
    window.output_legend_linewidth_spin.setValue(3.0)
    window.output_legend_style_combo.setCurrentIndex(
        window.output_legend_style_combo.findData("labels"))
    window.output_trace_label_slider.setValue(60)
    settings = canvas.style_settings()
    assert settings["legend_linewidth"] == 3.0
    assert settings["legend_style"] == "labels"
    assert settings["trace_label_position"] == pytest.approx(0.6)
    assert window.output_trace_label_slider.isEnabled()
    assert not window.output_legend_linewidth_spin.isEnabled()
    # Another plot's own settings are shown when it is the target.
    window.canvas_tabs.setCurrentWidget(window.canvas)
    window._load_style_controls_from(window.canvas)
    assert window.output_legend_style_combo.currentData() == "legend"
    assert window.output_legend_linewidth_spin.value() == 0.0
    # And they are among the keys a saved per-plot default keeps.
    for key in ("legend_linewidth", "legend_style", "trace_label_position"):
        assert key in canvas._STYLE_KEYS


# -- DNP --------------------------------------------------------------------------

def _dnp(window):
    window.spectrum_list.selectAll()
    window._show_side_tab("Compare")
    window._dnp_section.setExpanded(True)
    return window.dnp_panel


def _is_1d_style(ax):
    spines = {k: v.get_visible() for k, v in ax.spines.items()}
    ticks_y = any(t.tick1line.get_visible() or t.tick2line.get_visible()
                  for t in ax.yaxis.get_major_ticks() + ax.yaxis.get_minor_ticks())
    return spines == {"left": False, "right": False, "top": False, "bottom": True} \
        and not ticks_y and ax.get_ylabel() == ""


def test_dnp_spectra_are_drawn_as_1d_spectra(window):
    panel = _dnp(window)
    panel.sweep_pick_button.setChecked(True)          # sweep spectra view
    ax = window.compare_canvas.figure.axes[0]
    window.compare_canvas.canvas.draw()
    assert _is_1d_style(ax)
    panel.sweep_pick_button.setChecked(False)


def test_notebook_dnp_spectrum_plots_are_1d_style():
    from opennmr.notebook import dnp
    spectra = _sweep(FIELDS, ERRORS)
    fig = dnp.plot_sweep_spectra(spectra)
    assert _is_1d_style(fig.axes[0])
    fig = dnp.plot_enhancement_spectra(spectra[0], spectra[1])
    assert _is_1d_style(fig.axes[0])
    # The field profile is a plot of numbers, not a spectrum: its axis stays.
    result = dnp.field_sweep(spectra, peak=0.0, region=(-3000.0, 3000.0))
    fig = dnp.plot_field_profile(result)
    assert fig.axes[0].spines["left"].get_visible() and fig.axes[0].get_ylabel()


def test_a_dnp_view_does_not_switch_the_compare_legend_off(window):
    panel = _dnp(window)
    panel.sweep_pick_button.setChecked(True)
    panel.sweep_pick_button.setChecked(False)
    canvas = _compare(window, 1)
    assert len(_legends(canvas)) == 1


def test_home_restores_the_dnp_field_profile(window):
    panel = _dnp(window)
    panel.sweep_peak, panel.sweep_region = 0.0, (-3000.0, 3000.0)
    panel.run_sweep()
    canvas = window.fit_canvas
    canvas.canvas.draw()
    ax = canvas.figure.axes[0]
    start = (ax.get_xlim(), ax.get_ylim())
    event = SimpleNamespace(inaxes=ax, xdata=sum(ax.get_xlim()) / 2,
                            ydata=sum(ax.get_ylim()) / 2, step=2, button="up", x=400, y=300,
                            key=None, guiEvent=None, canvas=canvas.canvas)
    canvas._on_scroll(event)
    assert ax.get_ylim() != start[1]
    canvas.toolbar.home()
    assert ax.get_xlim() == pytest.approx(start[0]) and ax.get_ylim() == pytest.approx(start[1])


def test_crosshair_does_not_pull_the_view_out_to_zero(window):
    """User report: with crosshairs on, entering the DNP field profile
    (fields near 9396 mT) zoomed out to show (0, 0), and Home could not
    undo it. The crosshair's lines were data at x = 0 and y = 0."""
    from PySide6.QtCore import QPoint
    from PySide6.QtTest import QTest
    panel = _dnp(window)
    panel.sweep_peak, panel.sweep_region = 0.0, (-3000.0, 3000.0)
    panel.run_sweep()
    canvas = window.fit_canvas
    canvas.set_crosshair_enabled(True)
    canvas._ensure_crosshair()
    canvas.canvas.draw()
    ax = canvas.figure.axes[0]
    start = (ax.get_xlim(), ax.get_ylim())
    assert canvas._crosshair is not None
    # A move into the axes shows the crosshair; then a full redraw.
    x_px, y_px = ax.transData.transform((sum(start[0]) / 2, sum(start[1]) / 2))
    ratio = canvas.canvas.devicePixelRatioF()
    QTest.mouseMove(canvas.canvas, QPoint(int(x_px / ratio),
                                          int(canvas.canvas.height() - y_px / ratio)))
    canvas.canvas.draw()
    assert ax.get_xlim() == pytest.approx(start[0])
    assert ax.get_ylim() == pytest.approx(start[1])
    assert min(ax.get_xlim()) > 1000                       # nowhere near 0
    canvas.toolbar.home()
    assert ax.get_xlim() == pytest.approx(start[0]) and ax.get_ylim() == pytest.approx(start[1])
    # The crosshair lines are not data.
    assert not any(line in ax.lines for line in canvas._crosshair._vlines)
