from types import SimpleNamespace

import types

import matplotlib.colors
import numpy as np
from PySide6.QtCore import QPoint
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QComboBox, QDialog, QSpinBox

from opennmr.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
from opennmr import plotting
from opennmr.gui.widgets.canvas import MplCanvas


@pytest.fixture(scope="module")
def qapp():
    yield QApplication.instance() or QApplication([])


def _make_1d():
    ppm = np.linspace(0, 10, 100)
    metadata = NMRMetadata(vendor="bruker", nucleus="1H", sfo1=400.0)
    axis = NMRAxis(kind="frequency", ppm=ppm, hz=ppm * 400.0)
    return NMRSpectrum(data=np.exp(-((ppm - 5) ** 2)), axes=[axis], metadata=metadata)


def _make_2d(n_f1=8, n_f2=64):
    x_ppm = np.linspace(0, 10, n_f2)
    y_ppm = np.linspace(90, 100, n_f1)
    data = np.exp(-((x_ppm - 5) ** 2)) * np.ones((n_f1, 1))
    metadata = NMRMetadata(vendor="bruker", nucleus="1H", nucleus_indirect="13C", sfo1=400.0, sfo1_indirect=100.0)
    f1_axis = NMRAxis(kind="frequency", ppm=y_ppm, hz=y_ppm * 100.0)
    f2_axis = NMRAxis(kind="frequency", ppm=x_ppm, hz=x_ppm * 400.0)
    return NMRSpectrum(data=data, axes=[f1_axis, f2_axis], metadata=metadata)


# _on_scroll now distinguishes "hovering the x-axis tick-label strip"
# from "hovering the plot body" using real pixel coordinates (see
# MplCanvas._hovering_x_axis) -- these three helpers build fake scroll
# events with realistic pixel positions for each case, rather than the
# old bare inaxes/xdata/ydata SimpleNamespace (which had no pixel
# position at all, since the old zoom logic never needed one).

def _body_scroll_event(ax, xdata, ydata, button="up"):
    """A scroll event over the plot body, at data position
    (xdata, ydata) -- matches what a real matplotlib event looks like
    while the cursor is inside the axes' own data rectangle."""
    x, y = ax.transData.transform((xdata, ydata))
    return SimpleNamespace(inaxes=ax, xdata=xdata, ydata=ydata, x=x, y=y, button=button)


def _x_strip_scroll_event(ax, xdata, button="up"):
    """A scroll event over ``ax``'s x-tick-label strip (below the
    plot body, at data-x ``xdata``) -- matplotlib itself would report
    ``inaxes=None``/``xdata=None`` there, since the cursor isn't over
    the axes' own data rectangle."""
    bbox = ax.get_window_extent()
    x, _ = ax.transData.transform((xdata, ax.get_ylim()[0]))
    return SimpleNamespace(inaxes=None, xdata=None, ydata=None, x=x, y=bbox.y0 - 5, button=button)


def _y_strip_scroll_event(ax, ydata, button="up"):
    """A scroll event over ``ax``'s y-tick-label strip -- on the
    *right* of the plot body, matching ``plot_2d_contour``'s own
    ``tick_right()`` for the main contour panel (see
    ``MplCanvas._hovering_y_axis``)."""
    bbox = ax.get_window_extent()
    _, y = ax.transData.transform((ax.get_xlim()[0], ydata))
    return SimpleNamespace(inaxes=None, xdata=None, ydata=None, x=bbox.x1 + 5, y=y, button=button)


def _outside_scroll_event(ax, button="up"):
    """A scroll event well outside both ``ax``'s plot body and its
    x-axis strip -- a true no-op position."""
    bbox = ax.get_window_extent()
    return SimpleNamespace(inaxes=None, xdata=None, ydata=None, x=bbox.x0 - 50, y=bbox.y1 + 50, button=button)


def test_scroll_up_over_x_axis_zooms_x_in_around_cursor(qapp):
    canvas = MplCanvas()
    canvas.show_spectrum(_make_1d())
    canvas.canvas.draw()
    ax = canvas.ax
    ax.set_xlim(0, 10)
    canvas._on_scroll(_x_strip_scroll_event(ax, xdata=3.0, button="up"))
    xmin, xmax = ax.get_xlim()
    assert (xmax - xmin) < 10  # narrower than before
    assert xmin < 3.0 < xmax  # cursor position still inside view


def test_scroll_down_over_x_axis_zooms_x_out(qapp):
    canvas = MplCanvas()
    canvas.show_spectrum(_make_1d())
    canvas.canvas.draw()
    ax = canvas.ax
    ax.set_xlim(4, 6)
    canvas._on_scroll(_x_strip_scroll_event(ax, xdata=5.0, button="down"))
    xmin, xmax = ax.get_xlim()
    assert (xmax - xmin) > 2


def test_scroll_over_x_axis_leaves_y_untouched(qapp):
    canvas = MplCanvas()
    canvas.show_spectrum(_make_1d())
    canvas.canvas.draw()
    ax = canvas.ax
    ax.set_xlim(0, 10)
    ax.set_ylim(-1, 1)
    canvas._on_scroll(_x_strip_scroll_event(ax, xdata=3.0, button="up"))
    assert ax.get_ylim() == pytest.approx((-1, 1))


def test_scroll_over_plot_body_zooms_y_intensity_only(qapp):
    canvas = MplCanvas()
    canvas.show_spectrum(_make_1d())
    canvas.canvas.draw()
    ax = canvas.ax
    ax.set_xlim(0, 10)
    ax.set_ylim(-1, 1)
    canvas._on_scroll(_body_scroll_event(ax, xdata=5.0, ydata=0.0, button="up"))
    ymin, ymax = ax.get_ylim()
    assert (ymax - ymin) < 2  # narrower than before
    assert ax.get_xlim() == pytest.approx((0, 10))  # x untouched


def test_scroll_over_plot_body_pins_the_bottom(qapp):
    canvas = MplCanvas()
    canvas.show_spectrum(_make_1d())
    canvas.canvas.draw()
    ax = canvas.ax
    ax.set_ylim(-1, 1)
    canvas._on_scroll(_body_scroll_event(ax, xdata=5.0, ydata=0.8, button="up"))
    ymin, ymax = ax.get_ylim()
    assert ymin == pytest.approx(-1.0)  # bottom untouched regardless of cursor position
    assert ymax < 1.0  # only the top moved

    canvas._on_scroll(_body_scroll_event(ax, xdata=5.0, ydata=0.1, button="down"))
    ymin2, ymax2 = ax.get_ylim()
    assert ymin2 == pytest.approx(-1.0)
    assert ymax2 > ymax


def test_scroll_outside_axes_is_noop(qapp):
    canvas = MplCanvas()
    canvas.show_spectrum(_make_1d())
    canvas.canvas.draw()
    ax = canvas.ax
    ax.set_xlim(0, 10)
    canvas._on_scroll(_outside_scroll_event(ax))
    assert ax.get_xlim() == (0, 10)


def test_drag_pans_view(qapp):
    canvas = MplCanvas()
    canvas.show_spectrum(_make_1d())
    ax = canvas.ax
    ax.set_xlim(0, 10)
    canvas.toolbar.mode = ""

    press = SimpleNamespace(inaxes=ax, button=1, x=100, y=100)
    canvas._on_press(press)
    assert canvas._pan_start is not None

    # Simulate a drag: move 50 pixels right in display space.
    x0_data, _ = ax.transData.inverted().transform((100, 100))
    x1_data, _ = ax.transData.inverted().transform((150, 100))
    expected_dx = x1_data - x0_data

    motion = SimpleNamespace(x=150, y=100)
    canvas._on_motion(motion)
    xmin, xmax = ax.get_xlim()
    assert xmin == pytest.approx(0 - expected_dx, abs=1e-6)
    assert xmax == pytest.approx(10 - expected_dx, abs=1e-6)

    release = SimpleNamespace()
    canvas._on_release(release)
    assert canvas._pan_start is None


def test_drag_ignored_when_toolbar_tool_active(qapp):
    canvas = MplCanvas()
    canvas.show_spectrum(_make_1d())
    ax = canvas.ax
    canvas.toolbar.mode = "zoom rect"
    press = SimpleNamespace(inaxes=ax, button=1, x=100, y=100)
    canvas._on_press(press)
    assert canvas._pan_start is None


def test_drag_ignored_for_right_click(qapp):
    canvas = MplCanvas()
    canvas.show_spectrum(_make_1d())
    ax = canvas.ax
    canvas.toolbar.mode = ""
    press = SimpleNamespace(inaxes=ax, button=3, x=100, y=100)
    canvas._on_press(press)
    assert canvas._pan_start is None


def test_show_spectrum_labels_the_line(qapp):
    canvas = MplCanvas()
    canvas.show_spectrum(_make_1d())
    assert canvas.ax.lines[0].get_label() == "1H"


def test_legend_hidden_by_default(qapp):
    canvas = MplCanvas()
    canvas.show_spectrum(_make_1d())
    assert canvas.ax.get_legend() is None


def test_set_legend_visible_shows_and_hides(qapp):
    canvas = MplCanvas()
    canvas.show_spectrum(_make_1d())

    canvas.set_legend_visible(True)
    assert canvas.ax.get_legend() is not None

    canvas.set_legend_visible(False)
    assert canvas.ax.get_legend() is None


def test_legend_choice_persists_across_show_spectrum(qapp):
    canvas = MplCanvas()
    canvas.show_spectrum(_make_1d())
    canvas.set_legend_visible(True)

    # A fresh spectrum rebuilds the axes entirely -- the legend
    # choice must survive that rebuild.
    canvas.show_spectrum(_make_1d())
    assert canvas.ax.get_legend() is not None


def test_aspect_ratio_unlocked_by_default(qapp):
    canvas = MplCanvas()
    canvas.show_spectrum(_make_1d())
    assert canvas.ax.get_box_aspect() is None


def test_set_aspect_ratio_locks_and_unlocks(qapp):
    canvas = MplCanvas()
    canvas.show_spectrum(_make_1d())

    canvas.set_aspect_ratio(2.0)
    assert canvas.ax.get_box_aspect() == pytest.approx(2.0)

    canvas.set_aspect_ratio(None)
    assert canvas.ax.get_box_aspect() is None


def test_aspect_ratio_persists_across_show_spectrum(qapp):
    canvas = MplCanvas()
    canvas.show_spectrum(_make_1d())
    canvas.set_aspect_ratio(1.5)

    canvas.show_spectrum(_make_1d())
    assert canvas.ax.get_box_aspect() == pytest.approx(1.5)


def _make_2d_asymmetric_range(n_f1=8, n_f2=64):
    """Like _make_2d, but with a deliberately very different F1 vs. F2
    ppm *span* (10 ppm vs. 200 ppm, a real-world-scale mismatch -- e.g.
    a real 1H/13C HSQC's own F2/F1 spans differ by roughly this much).
    _make_2d's own equal 10-ppm/10-ppm spans are a real blind spot for
    testing square_aspect: mpl_toolkits.axes_grid1's default divider
    reference size (Size.AxesX/AxesY) is the axes' own *data range*,
    not its physical size -- with equal spans, "equalise the data-range
    reference" and "make the physical box square" coincidentally agree,
    which is exactly what let a real square_aspect+projections bug slip
    through undetected (found directly against real HSQC data, not
    this fixture -- fixed by using Size.Scaled(1) references instead,
    see plotting.twod.plot_2d_contour's own square_aspect docstring)."""
    x_ppm = np.linspace(0, 10, n_f2)
    y_ppm = np.linspace(0, 200, n_f1)
    data = np.exp(-((x_ppm - 5) ** 2)) * np.ones((n_f1, 1))
    metadata = NMRMetadata(vendor="bruker", nucleus="1H", nucleus_indirect="13C", sfo1=400.0, sfo1_indirect=100.0)
    f1_axis = NMRAxis(kind="frequency", ppm=y_ppm, hz=y_ppm * 100.0)
    f2_axis = NMRAxis(kind="frequency", ppm=x_ppm, hz=x_ppm * 400.0)
    return NMRSpectrum(data=data, axes=[f1_axis, f2_axis], metadata=metadata)


def test_2d_contour_defaults_to_a_square_physical_panel(qapp):
    # get_box_aspect() alone doesn't discriminate here -- square_aspect
    # is implemented via the AxesDivider (divider.set_aspect(True)), not
    # ax.set_box_aspect, so box_aspect itself stays None. The physical
    # position (in figure-fraction, converted to inches) is what
    # actually needs checking. A real synchronous draw() is required --
    # the divider's own locator resolves lazily at draw time, not at
    # show_spectrum() (which only calls the async draw_idle()).
    canvas = MplCanvas()
    canvas.figure.set_size_inches(10, 6)  # deliberately non-square figure
    canvas.show_spectrum(_make_2d_asymmetric_range())
    canvas.canvas.draw()
    pos = canvas.ax.get_position()
    fig_w, fig_h = canvas.figure.get_size_inches()
    assert pos.width * fig_w == pytest.approx(pos.height * fig_h, rel=1e-6)


def test_2d_square_default_keeps_f1_f2_projections_matched_to_the_panel(qapp):
    # An invariant, not the specific bug (see test_2d_contour_defaults_
    # to_a_square_physical_panel for that -- confirmed directly by
    # temporarily reverting to a plain make_axes_locatable() divider
    # that this "matched" property alone does NOT discriminate the real
    # bug: the appended F1/F2 axes are sized via that exact same
    # divider as the main panel, so they track *whatever* the main
    # panel's own cell resolves to, square or not). Kept anyway as a
    # genuine, separately-worth-having regression check: if a future
    # change ever gave the projections their own independent sizing
    # instead of going through the same coordinated divider, this is
    # what would catch it going out of sync with the main panel.
    canvas = MplCanvas()
    canvas.figure.set_size_inches(10, 6)
    canvas.show_spectrum(_make_2d_asymmetric_range())
    canvas.canvas.draw()
    ax_pos = canvas.ax.get_position()
    f1_pos = canvas._proj_axes["f1"].get_position()
    f2_pos = canvas._proj_axes["f2"].get_position()
    assert ax_pos.height == pytest.approx(f1_pos.height, rel=1e-6)
    assert ax_pos.width == pytest.approx(f2_pos.width, rel=1e-6)
    assert ax_pos.x0 - f1_pos.x1 == pytest.approx(0.1 / 10, abs=1e-3)  # projection_pad, in fig-fraction
    assert f2_pos.y0 - ax_pos.y1 == pytest.approx(0.1 / 6, abs=1e-3)


def test_2d_square_default_survives_a_contour_redraw(qapp):
    # _redraw_contour (contour-fraction/level changes, a live phase
    # drag's debounced refresh, ...) is a separate code path from
    # show_spectrum that also rebuilds the axes from scratch -- confirm
    # the square default isn't only applied on the very first render.
    canvas = MplCanvas()
    canvas.figure.set_size_inches(10, 6)
    canvas.show_spectrum(_make_2d_asymmetric_range())
    canvas._redraw_contour(preserve_view=False)
    canvas.canvas.draw()
    pos = canvas.ax.get_position()
    fig_w, fig_h = canvas.figure.get_size_inches()
    assert pos.width * fig_w == pytest.approx(pos.height * fig_h, rel=1e-6)


def test_explicit_aspect_ratio_overrides_the_2d_square_default(qapp):
    canvas = MplCanvas()
    canvas.set_aspect_ratio(2.0)
    canvas.show_spectrum(_make_2d())
    assert canvas.ax.get_box_aspect() == pytest.approx(2.0)


def test_1d_still_defaults_to_unlocked_aspect_after_a_2d_view(qapp):
    # The square default is contour-specific (gated on _contour_spectrum,
    # set by show_spectrum's own ndim==2 branch) -- switching back to a
    # 1D spectrum must not leave the box aspect stuck at 1.0.
    canvas = MplCanvas()
    canvas.show_spectrum(_make_2d())
    canvas.show_spectrum(_make_1d())
    assert canvas.ax.get_box_aspect() is None


def test_toolbar_has_no_popup_dialog_buttons():
    from opennmr.gui.widgets.canvas import _Toolbar

    names = [item[0] for item in _Toolbar.toolitems if item[0]]
    assert "Subplots" not in names
    assert "Customize" not in names
    assert "Home" in names  # the rest of the stock toolbar is untouched


# ============================================================
# Contour-level scroll-scaling (2D/pseudo-2D only)
# ============================================================

def test_show_spectrum_2d_sets_contour_state(qapp):
    canvas = MplCanvas()
    canvas.show_spectrum(_make_2d())
    assert canvas._contour_spectrum is not None
    assert canvas._contour_factor == pytest.approx(canvas._CONTOUR_FACTOR_DEFAULT)


def test_show_spectrum_1d_clears_contour_state(qapp):
    canvas = MplCanvas()
    canvas.show_spectrum(_make_2d())
    canvas.show_spectrum(_make_1d())
    assert canvas._contour_spectrum is None
    assert canvas._contour_factor is None


def test_scroll_over_contour_adjusts_factor_not_view(qapp):
    canvas = MplCanvas()
    canvas.show_spectrum(_make_2d())
    canvas.canvas.draw()
    ax = canvas.ax
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    original_factor = canvas._contour_factor

    canvas._on_scroll(_body_scroll_event(ax, xdata=5.0, ydata=95.0, button="up"))

    assert canvas._contour_factor != pytest.approx(original_factor)
    # The pan/zoom view itself must be unaffected -- scrolling over a
    # contour changes the levels, not the axis range.
    assert canvas.ax.get_xlim() == pytest.approx(xlim)
    assert canvas.ax.get_ylim() == pytest.approx(ylim)


def test_scroll_up_over_contour_raises_factor(qapp):
    canvas = MplCanvas()
    canvas.show_spectrum(_make_2d())
    canvas.canvas.draw()
    before = canvas._contour_factor
    canvas._on_scroll(_body_scroll_event(canvas.ax, xdata=5.0, ydata=95.0, button="up"))
    assert canvas._contour_factor > before


def test_scroll_down_over_contour_lowers_factor(qapp):
    canvas = MplCanvas()
    canvas.show_spectrum(_make_2d())
    canvas.canvas.draw()
    before = canvas._contour_factor
    canvas._on_scroll(_body_scroll_event(canvas.ax, xdata=5.0, ydata=95.0, button="down"))
    assert canvas._contour_factor < before


def test_scroll_over_contour_x_axis_zooms_x_instead(qapp):
    canvas = MplCanvas()
    canvas.show_spectrum(_make_2d())
    canvas.canvas.draw()
    ax = canvas.ax
    xlim = ax.get_xlim()
    original_factor = canvas._contour_factor

    canvas._on_scroll(_x_strip_scroll_event(ax, xdata=5.0, button="up"))

    assert canvas._contour_factor == pytest.approx(original_factor)  # levels untouched
    assert canvas.ax.get_xlim() != pytest.approx(xlim)  # x zoomed instead


def test_scroll_over_contour_y_axis_zooms_y_instead(qapp):
    # Explicit user request: y-axis scroll-zoom (like x already had)
    # for the 2D/pseudo-2D contour view, not just the F1/F2 projection
    # panels or the contour-threshold body scroll.
    canvas = MplCanvas()
    canvas.show_spectrum(_make_2d())
    canvas.canvas.draw()
    ax = canvas.ax
    ylim = ax.get_ylim()
    original_factor = canvas._contour_factor

    canvas._on_scroll(_y_strip_scroll_event(ax, ydata=95.0, button="up"))

    assert canvas._contour_factor == pytest.approx(original_factor)  # levels untouched
    assert canvas.ax.get_ylim() != pytest.approx(ylim)  # y zoomed instead


def test_scroll_up_over_contour_y_axis_zooms_in_around_cursor(qapp):
    canvas = MplCanvas()
    canvas.show_spectrum(_make_2d())
    canvas.canvas.draw()
    ax = canvas.ax
    ax.set_ylim(90, 100)
    canvas._on_scroll(_y_strip_scroll_event(ax, ydata=93.0, button="up"))
    ymin, ymax = ax.get_ylim()
    assert (ymax - ymin) < 10  # narrower than before
    assert min(ymin, ymax) < 93.0 < max(ymin, ymax)  # cursor position still inside view


def test_scroll_over_contour_y_axis_leaves_x_untouched(qapp):
    canvas = MplCanvas()
    canvas.show_spectrum(_make_2d())
    canvas.canvas.draw()
    ax = canvas.ax
    ax.set_xlim(0, 10)
    ax.set_ylim(90, 100)
    canvas._on_scroll(_y_strip_scroll_event(ax, ydata=95.0, button="up"))
    assert ax.get_xlim() == pytest.approx((0, 10))


def test_scroll_over_1d_right_margin_does_not_zoom_y(qapp):
    # The new y-strip zoom is gated to 2D/pseudo-2D contours only
    # (_contour_factor is not None) -- a plain 1D view has no y-ticks
    # shown at all (intensity already zooms via hovering the body, see
    # test_scroll_over_plot_body_zooms_y_intensity_only), so its right
    # margin must stay a no-op rather than start zooming y too.
    canvas = MplCanvas()
    canvas.show_spectrum(_make_1d())
    canvas.canvas.draw()
    ax = canvas.ax
    ax.set_xlim(0, 10)
    ax.set_ylim(-1, 1)
    canvas._on_scroll(_y_strip_scroll_event(ax, ydata=0.0, button="up"))
    assert ax.get_ylim() == pytest.approx((-1, 1))


def test_contour_factor_clamped_to_bounds(qapp):
    canvas = MplCanvas()
    canvas.show_spectrum(_make_2d())
    canvas.canvas.draw()
    for _ in range(60):  # far more notches than needed to hit either bound
        canvas._on_scroll(_body_scroll_event(canvas.ax, xdata=5.0, ydata=95.0, button="up"))
        canvas.canvas.draw()  # _adjust_contour_levels rebuilds canvas.ax each time
    assert canvas._contour_factor <= canvas._CONTOUR_FACTOR_MAX
    for _ in range(60):
        canvas._on_scroll(_body_scroll_event(canvas.ax, xdata=5.0, ydata=95.0, button="down"))
        canvas.canvas.draw()
    assert canvas._contour_factor >= canvas._CONTOUR_FACTOR_MIN


def test_scroll_over_1d_still_zooms_normally(qapp):
    canvas = MplCanvas()
    canvas.show_spectrum(_make_2d())
    canvas.show_spectrum(_make_1d())  # _contour_factor must reset to None
    canvas.canvas.draw()
    ax = canvas.ax
    ax.set_xlim(0, 10)
    canvas._on_scroll(_x_strip_scroll_event(ax, xdata=3.0, button="up"))
    xmin, xmax = ax.get_xlim()
    assert (xmax - xmin) < 10


# ============================================================
# Contour levels, slice extraction, projection scroll-zoom
# ============================================================

def test_set_contour_levels_updates_state_and_redraws(qapp):
    canvas = MplCanvas()
    canvas.show_spectrum(_make_2d())
    canvas.set_contour_levels(8, "linear", 2.0)
    assert canvas._contour_clevels == 8
    assert canvas._contour_level_mode == "linear"
    assert canvas._contour_power == 2.0
    assert canvas._contour_spectrum is not None  # still a contour view
    assert canvas.ax is not None


def test_set_contour_levels_can_also_set_start_factor(qapp):
    canvas = MplCanvas()
    canvas.show_spectrum(_make_2d())
    canvas.set_contour_levels(8, "power", 1.25, factor=0.05)
    assert canvas._contour_factor == pytest.approx(0.05)


def test_set_contour_levels_preserves_current_zoom(qapp):
    canvas = MplCanvas()
    canvas.show_spectrum(_make_2d())
    canvas.canvas.draw()
    canvas.ax.set_xlim(2, 8)
    canvas.ax.set_ylim(92, 98)
    canvas.set_contour_levels(10, "power", 1.0)
    assert canvas.ax.get_xlim() == pytest.approx((2, 8))
    assert canvas.ax.get_ylim() == pytest.approx((92, 98))


def test_extract_slice_shows_1d_row(qapp):
    canvas = MplCanvas()
    spec = _make_2d(n_f1=8, n_f2=64)
    canvas.show_spectrum(spec)
    canvas.extract_slice("f1", 3)
    assert canvas._contour_spectrum is None  # now a 1D view, not the contour
    assert canvas._current_1d_spectrum is not None
    assert canvas._current_1d_spectrum.ndim == 1
    assert len(canvas._current_1d_spectrum.data) == 64  # traced along F2


def test_extract_slice_remembers_its_own_source_contour(qapp):
    # Real gap, found and fixed: extract_slice had no way back to the
    # contour at all once used, short of reselecting the spectrum in
    # the list -- confirmed there was no "show contour" mechanism
    # anywhere in this file before this fix.
    canvas = MplCanvas()
    spec = _make_2d(n_f1=8, n_f2=64)
    canvas.show_spectrum(spec)
    canvas.extract_slice("f1", 3)
    assert canvas._extracted_from_spectrum is spec


def test_show_contour_after_extract_slice_restores_the_contour(qapp):
    canvas = MplCanvas()
    spec = _make_2d(n_f1=8, n_f2=64)
    canvas.show_spectrum(spec)
    canvas.extract_slice("f1", 3)

    # Mirrors the right-click "Show Contour" handler in _on_context_menu.
    saved = canvas._extracted_from_spectrum
    canvas.show_spectrum(saved)

    assert canvas._contour_spectrum is spec
    assert canvas._current_1d_spectrum is None
    assert canvas._extracted_from_spectrum is None  # consumed, not left stale


def test_extracted_from_spectrum_cleared_by_an_unrelated_show_spectrum_call(qapp):
    # Staleness guard: after extracting a slice, showing a *different*
    # spectrum must not leave a "Show Contour" action pointing at the
    # old one.
    canvas = MplCanvas()
    spec = _make_2d(n_f1=8, n_f2=64)
    canvas.show_spectrum(spec)
    canvas.extract_slice("f1", 3)
    assert canvas._extracted_from_spectrum is not None

    canvas.show_spectrum(_make_1d())
    assert canvas._extracted_from_spectrum is None


def test_show_2d_overlay_renders_and_clears_contour_spectrum(qapp):
    canvas = MplCanvas()
    canvas.show_spectrum(_make_2d(n_f1=8, n_f2=64))
    assert canvas._contour_spectrum is not None

    spec_a = _make_2d(n_f1=8, n_f2=64)
    spec_b = _make_2d(n_f1=8, n_f2=64)
    canvas.show_2d_overlay([spec_a, spec_b], labels=["A", "B"])

    # Not a single-spectrum contour view -- see show_2d_overlay's own
    # docstring for why (scroll-to-threshold/Extract Slice/etc. all
    # assume one specific spectrum's own data).
    assert canvas._contour_spectrum is None
    assert len(canvas.ax.collections) >= 2
    assert canvas.ax.get_legend() is not None


def test_show_2d_overlay_raises_for_incompatible_spectra(qapp):
    canvas = MplCanvas()
    spec_a = _make_2d(n_f1=8, n_f2=64)
    spec_b = _make_2d(n_f1=8, n_f2=64)
    spec_b.metadata.nucleus_indirect = "31P"  # deliberately mismatched
    with pytest.raises(ValueError):
        canvas.show_2d_overlay([spec_a, spec_b])


def test_extract_slice_shows_1d_column(qapp):
    canvas = MplCanvas()
    spec = _make_2d(n_f1=8, n_f2=64)
    canvas.show_spectrum(spec)
    canvas.extract_slice("f2", 10)
    assert canvas._current_1d_spectrum.ndim == 1
    assert len(canvas._current_1d_spectrum.data) == 8  # traced along F1



def _slice_index_of(spectrum, axis, cursor_value):
    """_nearest_slice_index is an instance method now -- it resolves the
    cursor against the axis *as displayed*, which needs the canvas's own
    unit. These tests all mean "showing this spectrum in ppm"."""
    canvas = MplCanvas()
    canvas.show_spectrum(spectrum)
    return canvas._nearest_slice_index(spectrum, axis, cursor_value)


def test_nearest_slice_index_matches_cursor_ppm_on_a_frequency_axis(qapp):
    spec = _make_2d(n_f1=8, n_f2=64)  # F1 ppm: linspace(90, 100, 8); F2 ppm: linspace(0, 10, 64)
    row = _slice_index_of(spec, "f1", 95.0)
    assert spec.axes[0].ppm[row - 1] == pytest.approx(95.0, abs=0.75)
    col = _slice_index_of(spec, "f2", 5.0)
    assert spec.axes[1].ppm[col - 1] == pytest.approx(5.0, abs=0.2)


def test_nearest_slice_index_clips_to_valid_range(qapp):
    spec = _make_2d(n_f1=8, n_f2=64)
    assert _slice_index_of(spec, "f1", 1000.0) == 8  # far past the axis's own max ppm
    assert _slice_index_of(spec, "f1", -1000.0) == 1


def test_nearest_slice_index_handles_pseudo_2d_index_axis(qapp):
    # A pseudo-2D's own F1 axis has no ppm at all (kind="index") --
    # the plotted Y value already *is* approximately the 1-based slice
    # number, so this should just round/clip it rather than trying
    # (and failing) to look up a ppm value.
    n_f1, n_f2 = 6, 32
    x_ppm = np.linspace(0, 10, n_f2)
    data = np.exp(-((x_ppm - 5) ** 2)) * np.ones((n_f1, 1))
    metadata = NMRMetadata(vendor="bruker", nucleus="1H", sfo1=400.0, pseudo_2d=True)
    f1_axis = NMRAxis(kind="index", index=np.arange(1, n_f1 + 1))
    f2_axis = NMRAxis(kind="frequency", ppm=x_ppm, hz=x_ppm * 400.0)
    spec = NMRSpectrum(data=data, axes=[f1_axis, f2_axis], metadata=metadata)

    assert _slice_index_of(spec, "f1", 3.4) == 3
    assert _slice_index_of(spec, "f1", 100.0) == n_f1  # clipped


def test_extract_slice_dialog_defaults_index_to_the_right_clicked_position(qapp):
    # Real roadmap ask: "row/slice extraction should default to the
    # cursor's current position rather than a fixed index."
    canvas = MplCanvas()
    spec = _make_2d(n_f1=8, n_f2=64)
    canvas.show_spectrum(spec)
    canvas.canvas.draw()

    # Simulate a real right-click at F2=5ppm (x), F1=95ppm (y) -- same
    # capture path _on_press already uses (button 3, xdata/ydata set).
    from types import SimpleNamespace
    canvas._on_press(SimpleNamespace(button=3, xdata=5.0, ydata=95.0, inaxes=canvas.ax))

    seen = {}

    def fake_exec(dialog):
        seen["axis_combo"] = dialog.findChild(QComboBox)
        seen["index_spin"] = dialog.findChild(QSpinBox)
        return QDialog.DialogCode.Rejected  # don't actually extract -- just inspect the pre-filled state

    orig_exec = QDialog.exec
    QDialog.exec = fake_exec
    try:
        canvas._prompt_extract_slice()
    finally:
        QDialog.exec = orig_exec

    # Default axis is "Row (fix F1)" -- its index should default from
    # the cursor's own Y position (95 ppm), not a fixed 1.
    expected_row = _slice_index_of(spec, "f1", 95.0)
    assert seen["index_spin"].value() == expected_row
    assert expected_row != 1  # confirms this is a real, non-trivial default, not a coincidence

    # Switching to "Column (fix F2)" should re-derive from the cursor's
    # X position (5 ppm) instead.
    seen["axis_combo"].setCurrentIndex(1)
    expected_col = _slice_index_of(spec, "f2", 5.0)
    assert seen["index_spin"].value() == expected_col


def test_extract_slice_dialog_falls_back_to_index_1_without_a_tracked_cursor(qapp):
    canvas = MplCanvas()
    canvas.show_spectrum(_make_2d(n_f1=8, n_f2=64))
    canvas.canvas.draw()
    assert canvas._last_right_click_xy is None  # never right-clicked in this test

    seen = {}

    def fake_exec(dialog):
        seen["index_spin"] = dialog.findChild(QSpinBox)
        return QDialog.DialogCode.Rejected

    orig_exec = QDialog.exec
    QDialog.exec = fake_exec
    try:
        canvas._prompt_extract_slice()
    finally:
        QDialog.exec = orig_exec

    assert seen["index_spin"].value() == 1


def test_scroll_over_f2_projection_zooms_its_intensity_only(qapp):
    canvas = MplCanvas()
    canvas.show_spectrum(_make_2d())
    canvas.canvas.draw()
    ax_f2 = canvas._proj_axes["f2"]
    assert ax_f2 is not None
    ylim_before = ax_f2.get_ylim()
    xdata = sum(ax_f2.get_xlim()) / 2
    ydata = sum(ylim_before) / 2
    x, y = ax_f2.transData.transform((xdata, ydata))
    event = SimpleNamespace(inaxes=ax_f2, xdata=xdata, ydata=ydata, x=x, y=y, button="up")
    canvas._on_scroll(event)
    ylim_after = ax_f2.get_ylim()
    assert ylim_after != pytest.approx(ylim_before)
    assert ylim_after[0] == pytest.approx(ylim_before[0])  # baseline (bottom) end pinned
    # The main contour's own factor/view must be untouched.
    assert canvas._contour_factor == pytest.approx(MplCanvas._CONTOUR_FACTOR_DEFAULT)


def test_scroll_over_f1_projection_zooms_its_intensity_only(qapp):
    canvas = MplCanvas()
    canvas.show_spectrum(_make_2d())
    canvas.canvas.draw()
    ax_f1 = canvas._proj_axes["f1"]
    assert ax_f1 is not None
    xlim_before = ax_f1.get_xlim()
    xdata = sum(xlim_before) / 2
    ydata = sum(ax_f1.get_ylim()) / 2
    x, y = ax_f1.transData.transform((xdata, ydata))
    event = SimpleNamespace(inaxes=ax_f1, xdata=xdata, ydata=ydata, x=x, y=y, button="up")
    canvas._on_scroll(event)
    xlim_after = ax_f1.get_xlim()
    assert xlim_after != pytest.approx(xlim_before)
    assert xlim_after[1] == pytest.approx(xlim_before[1])  # baseline (xmax) end pinned


# ============================================================
# FID window-function overlay
# ============================================================

def test_show_fid_without_envelope_has_one_line(qapp):
    canvas = MplCanvas()
    fid = np.exp(-np.arange(50) / 10.0) * np.exp(1j * np.arange(50) * 0.3)
    canvas.show_fid(fid, dt=1e-4)
    assert len(canvas.ax.lines) == 1  # magnitude only, not real+imag together
    assert canvas.ax.lines[0].get_label() == "Magnitude"


def test_show_fid_real_mode_plots_real_part_only(qapp):
    canvas = MplCanvas()
    fid = np.exp(-np.arange(50) / 10.0) * np.exp(1j * np.arange(50) * 0.3)
    canvas.show_fid(fid, dt=1e-4, mode="real")
    assert len(canvas.ax.lines) == 1
    assert canvas.ax.lines[0].get_label() == "Real"
    np.testing.assert_allclose(canvas.ax.lines[0].get_ydata(), np.real(fid))


def test_show_fid_with_envelope_adds_second_line_and_legend(qapp):
    canvas = MplCanvas()
    fid = np.exp(-np.arange(50) / 10.0) * np.exp(1j * np.arange(50) * 0.3)
    envelope = np.exp(-np.arange(50) / 20.0)
    canvas.show_fid(fid, dt=1e-4, window_envelope=envelope)
    assert len(canvas.ax.lines) == 2
    labels = [line.get_label() for line in canvas.ax.lines]
    assert "Window" in labels
    assert canvas.ax.get_legend() is not None


def test_show_fid_envelope_scaled_to_fid_peak(qapp):
    canvas = MplCanvas()
    fid = np.full(20, 4.0 + 0j)
    envelope = np.full(20, 0.5)
    canvas.show_fid(fid, dt=1e-4, window_envelope=envelope)
    window_line = next(line for line in canvas.ax.lines if line.get_label() == "Window")
    np.testing.assert_allclose(window_line.get_ydata(), 2.0)  # 0.5 * peak(4.0)


def test_show_fid_clears_contour_state(qapp):
    canvas = MplCanvas()
    canvas.show_spectrum(_make_2d())
    canvas.show_fid(np.ones(10, dtype=complex), dt=1e-4)
    assert canvas._contour_spectrum is None
    assert canvas._contour_factor is None


def test_show_fid_scroll_zoom_is_freeform_not_pinned(qapp):
    # Explicit user request: an FID oscillates around y=0 rather than
    # rising from a baseline the way a phased spectrum does, so
    # pinning "the bottom" on scroll-zoom doesn't mean anything
    # sensible for it -- should be a plain cursor-centred zoom instead
    # (both ends move), like the x-axis already does.
    canvas = MplCanvas()
    fid = np.exp(-np.arange(50) / 10.0) * np.exp(1j * np.arange(50) * 0.3)
    canvas.show_fid(fid, dt=1e-4)
    canvas.canvas.draw()
    ax = canvas.ax
    assert ax in canvas._freeform_zoom_axes
    ax.set_ylim(-1, 1)
    canvas._on_scroll(_body_scroll_event(ax, xdata=0.001, ydata=0.8, button="up"))
    ymin, ymax = ax.get_ylim()
    assert ymin > -1.0  # bottom *did* move -- not pinned
    assert ymax < 1.0


def test_show_spectrum_scroll_zoom_still_pins_the_bottom(qapp):
    # The freeform behaviour above is specific to FID views -- a plain
    # phased-spectrum view keeps the existing pin-bottom zoom.
    canvas = MplCanvas()
    canvas.show_spectrum(_make_1d())
    canvas.canvas.draw()
    assert canvas.ax not in canvas._freeform_zoom_axes


# ============================================================
# Measurement tool (Cmd/Ctrl+D)
# ============================================================

def test_measurement_mode_toggle_changes_cursor(qapp):
    canvas = MplCanvas()
    canvas.show_spectrum(_make_1d())
    canvas.canvas.draw()
    assert canvas._measurement_mode is False
    canvas.set_measurement_mode(True)
    assert canvas._measurement_mode is True
    canvas.set_measurement_mode(False)
    assert canvas._measurement_mode is False


def test_measurement_records_two_points_and_emits_signal(qapp):
    from unittest.mock import MagicMock
    canvas = MplCanvas()
    canvas.show_spectrum(_make_1d())
    canvas.canvas.draw()
    canvas.set_measurement_mode(True)
    assert canvas._measurement_points == []
    # Record first point
    canvas._record_measurement_point(canvas.ax, 1.0, 0.5)
    assert canvas._measurement_points == [(canvas.ax, 1.0, 0.5)]
    # Record second point and emit signal
    signal_mock = MagicMock()
    canvas.measurementCompleted.connect(signal_mock)
    canvas._record_measurement_point(canvas.ax, 3.0, 0.8)
    assert canvas._measurement_points == []  # Points cleared after emit
    signal_mock.assert_called_once()
    result = signal_mock.call_args[0][0]
    assert result["x1"] == 1.0
    assert result["x2"] == 3.0
    # 2 ppm apart on a 400 MHz axis is 800 Hz. This used to report the
    # raw display-unit difference (2.0) labelled as Hz -- the exact
    # unit confusion the measuring plan called the one real risk.
    assert result["dx_display"] == 2.0
    assert result["dx"]["ppm"] == pytest.approx(2.0)
    assert result["dx"]["hz"] == pytest.approx(800.0)


# ============================================================
# Axis unit switching (ppm / Hz / kHz)
# ============================================================

def test_show_spectrum_1d_tracks_current_spectrum_and_defaults_to_ppm(qapp):
    canvas = MplCanvas()
    spectrum = _make_1d()
    canvas.show_spectrum(spectrum)
    assert canvas._current_1d_spectrum is spectrum
    assert canvas._axis_unit == "ppm"


def test_show_spectrum_2d_does_not_track_1d_spectrum(qapp):
    canvas = MplCanvas()
    canvas.show_spectrum(_make_2d())
    assert canvas._current_1d_spectrum is None


def test_set_axis_unit_hz_updates_xdata_and_label(qapp):
    canvas = MplCanvas()
    spectrum = _make_1d()
    canvas.show_spectrum(spectrum)

    canvas.set_axis_unit("hz")

    np.testing.assert_allclose(canvas.ax.lines[0].get_xdata(), spectrum.axes[0].hz)
    assert canvas.ax.get_xlabel() == "Frequency / Hz"
    assert canvas._axis_unit == "hz"


def test_set_axis_unit_khz_scales_hz_by_1000(qapp):
    canvas = MplCanvas()
    spectrum = _make_1d()
    canvas.show_spectrum(spectrum)

    canvas.set_axis_unit("khz")

    np.testing.assert_allclose(canvas.ax.lines[0].get_xdata(), spectrum.axes[0].hz / 1000.0)
    assert canvas.ax.get_xlabel() == "Frequency / kHz"


def test_set_axis_unit_back_to_ppm_restores_original_xdata_and_label(qapp):
    canvas = MplCanvas()
    spectrum = _make_1d()
    canvas.show_spectrum(spectrum)
    original_label = canvas.ax.get_xlabel()

    canvas.set_axis_unit("hz")
    canvas.set_axis_unit("ppm")

    np.testing.assert_allclose(canvas.ax.lines[0].get_xdata(), spectrum.axes[0].ppm)
    assert canvas.ax.get_xlabel() == original_label


def test_set_axis_unit_switches_a_contour(qapp):
    # Real user report: "I am not able to right-click and change the
    # axis on a 2D (for imaging)". The Axis Units menu was gated on a 1D
    # view, and set_axis_unit only ever rewrote line data -- a contour
    # takes its axes from the spectrum, so the unit change is a rebuild.
    canvas = MplCanvas()
    spectrum = _make_2d(n_f1=8, n_f2=64)
    canvas.show_spectrum(spectrum)
    ppm_limits = canvas.ax.get_xlim()

    canvas.set_axis_unit("hz")
    assert canvas._axis_unit == "hz"
    hz = spectrum.axes[1].values("hz")
    assert canvas.ax.get_xlim() != ppm_limits
    # Compared as a span, not as limits: plot_2d_contour pads the view
    # past the data, so the endpoints are not the data's own.
    span = abs(canvas.ax.get_xlim()[1] - canvas.ax.get_xlim()[0])
    assert span == pytest.approx(float(hz.max() - hz.min()), rel=0.25)
    assert canvas.ax.get_xlabel() == "Frequency / Hz"

    # The underlying spectrum is untouched -- slicing and referencing
    # still reason in ppm.
    np.testing.assert_allclose(canvas._contour_spectrum.axes[1].ppm, spectrum.axes[1].ppm)

    canvas.set_axis_unit("ppm")
    assert canvas.ax.get_xlim() == pytest.approx(ppm_limits)


def test_set_axis_unit_mm_on_a_contour_needs_a_gradient(qapp):
    canvas = MplCanvas()
    canvas.show_spectrum(_make_2d(n_f1=8, n_f2=64))
    canvas.set_axis_unit("mm")
    assert canvas._axis_unit == "ppm"        # no gradient -> nothing to convert with

    canvas.set_imaging_gradient_params(gamma_hz_per_t=42.577e6, max_gradient_g_per_cm=50.0)
    canvas.set_axis_unit("mm")
    assert canvas._axis_unit == "mm"
    assert canvas.ax.get_xlabel() == "Position / mm"


def test_nearest_slice_index_follows_the_displayed_unit(qapp):
    # The cursor value comes off the displayed axis, so resolving it
    # against ppm would pick the wrong row once the contour shows Hz.
    canvas = MplCanvas()
    spectrum = _make_2d(n_f1=8, n_f2=64)
    canvas.show_spectrum(spectrum)
    in_ppm = canvas._nearest_slice_index(spectrum, "f1", 95.0)

    canvas.set_axis_unit("hz")
    hz_at_that_row = float(spectrum.axes[0].values("hz")[in_ppm - 1])
    assert canvas._nearest_slice_index(spectrum, "f1", hz_at_that_row) == in_ppm
    # ...and the raw ppm number now resolves somewhere else entirely.
    assert canvas._nearest_slice_index(spectrum, "f1", 95.0) != in_ppm


def test_canvas_offers_axis_units_on_a_contour_but_not_ppm_referencing_off_ppm(qapp, monkeypatch):
    canvas = MplCanvas()
    canvas.show_spectrum(_make_2d(n_f1=8, n_f2=64))
    canvas._last_right_click_xy = (5.0, 95.0)

    labels = _menu_labels(canvas, monkeypatch)
    assert "Axis Units" in labels
    assert "Set Reference from Cursor (ppm)…" in labels

    # Off ppm the click is in another unit, so asking for a shift in ppm
    # would be nonsense -- the action stands down, the units stay.
    canvas.set_axis_unit("hz")
    labels = _menu_labels(canvas, monkeypatch)
    assert "Axis Units" in labels
    assert "Set Reference from Cursor (ppm)…" not in labels


def test_set_axis_unit_rejects_unknown_unit(qapp):
    canvas = MplCanvas()
    spectrum = _make_1d()
    canvas.show_spectrum(spectrum)
    canvas.set_axis_unit("bogus_unit")
    assert canvas._axis_unit == "ppm"


def test_set_axis_unit_mm_without_gradient_params_is_a_noop(qapp):
    # "mm" is a recognised unit, but needs set_imaging_gradient_params
    # to have real values first -- silently ignored otherwise, same
    # "can't do this yet" behaviour as every other precondition here.
    canvas = MplCanvas()
    spectrum = _make_1d()
    canvas.show_spectrum(spectrum)
    canvas.set_axis_unit("mm")
    assert canvas._axis_unit == "ppm"


def test_set_axis_unit_mm_converts_hz_offset_to_position(qapp):
    canvas = MplCanvas()
    spectrum = _make_1d()
    canvas.show_spectrum(spectrum)
    gamma_hz_per_t = 42.577e6  # 1H
    max_gradient_g_per_cm = 50.0
    canvas.set_imaging_gradient_params(gamma_hz_per_t, max_gradient_g_per_cm)

    canvas.set_axis_unit("mm")

    hz_per_mm = gamma_hz_per_t * max_gradient_g_per_cm * 1e-5
    expected = spectrum.axes[0].hz / hz_per_mm
    np.testing.assert_allclose(canvas.ax.lines[0].get_xdata(), expected)
    assert canvas.ax.get_xlabel() == "Position / mm"
    assert canvas._axis_unit == "mm"


def test_set_axis_unit_mm_sanity_check_known_offset(qapp):
    # A concrete, hand-checkable case: at 50 G/cm for 1H, a 1 mm
    # displacement should correspond to ~21.29 kHz (gamma * gradient
    # in T/m = 42.577e6 Hz/T * 0.5 T/m = 21.29 MHz/m = 21.29 kHz/mm).
    canvas = MplCanvas()
    ppm = np.array([0.0, 1.0])
    axis = NMRAxis(kind="frequency", ppm=ppm, hz=np.array([0.0, 21288.7]))
    metadata = NMRMetadata(vendor="bruker", nucleus="1H", sfo1=400.0)
    spectrum = NMRSpectrum(data=np.array([1.0, 1.0]), axes=[axis], metadata=metadata)
    canvas.show_spectrum(spectrum)
    canvas.set_imaging_gradient_params(42.577e6, 50.0)

    canvas.set_axis_unit("mm")

    x = canvas.ax.lines[0].get_xdata()
    assert x[1] == pytest.approx(1.0, rel=1e-3)


def test_update_data_matches_downsampled_length_for_high_point_count(qapp):
    # A phase-drag (update_data, called on every tick) must re-derive
    # the exact same downsampling plot_1d already applied to the
    # x-data on screen -- a length mismatch would raise inside
    # matplotlib's own set_ydata.
    from opennmr.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
    from opennmr.plotting.style import DOWNSAMPLE_1D_MAX_POINTS

    n = DOWNSAMPLE_1D_MAX_POINTS * 3
    ppm = np.linspace(0, 10, n)
    metadata = NMRMetadata(vendor="bruker", nucleus="1H", sfo1=400.0)
    axis = NMRAxis(kind="frequency", ppm=ppm, hz=ppm * 400.0)
    spectrum = NMRSpectrum(data=np.exp(-((ppm - 5) ** 2)), axes=[axis], metadata=metadata)

    canvas = MplCanvas()
    canvas.show_spectrum(spectrum)
    assert len(canvas.ax.lines[0].get_xdata()) == DOWNSAMPLE_1D_MAX_POINTS

    spectrum.data = spectrum.data * 2.0  # simulate a phase/data change
    canvas.update_data(spectrum)  # must not raise (length mismatch)
    assert len(canvas.ax.lines[0].get_ydata()) == DOWNSAMPLE_1D_MAX_POINTS


# ============================================================
# Full-resolution toggle and export safety
# ============================================================

def _make_1d_n(n):
    from opennmr.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
    ppm = np.linspace(0, 10, n)
    metadata = NMRMetadata(vendor="bruker", nucleus="1H", sfo1=400.0)
    axis = NMRAxis(kind="frequency", ppm=ppm, hz=ppm * 400.0)
    return NMRSpectrum(data=np.exp(-((ppm - 5) ** 2)), axes=[axis], metadata=metadata)


def test_has_downsampled_data_false_for_small_spectrum(qapp):
    canvas = MplCanvas()
    canvas.show_spectrum(_make_1d())  # 100 points, well under the cap
    assert canvas.has_downsampled_data() is False


def test_has_downsampled_data_true_for_large_spectrum(qapp):
    from opennmr.plotting.style import DOWNSAMPLE_1D_MAX_POINTS
    canvas = MplCanvas()
    canvas.show_spectrum(_make_1d_n(DOWNSAMPLE_1D_MAX_POINTS * 3))
    assert canvas.has_downsampled_data() is True


def test_set_full_resolution_shows_every_point(qapp):
    from opennmr.plotting.style import DOWNSAMPLE_1D_MAX_POINTS
    n = DOWNSAMPLE_1D_MAX_POINTS * 3
    canvas = MplCanvas()
    canvas.show_spectrum(_make_1d_n(n))
    assert len(canvas.ax.lines[0].get_xdata()) == DOWNSAMPLE_1D_MAX_POINTS

    canvas.set_full_resolution(True)

    assert canvas._full_resolution is True
    assert len(canvas.ax.lines[0].get_xdata()) == n


def test_set_full_resolution_toggle_off_restores_downsampled(qapp):
    from opennmr.plotting.style import DOWNSAMPLE_1D_MAX_POINTS
    n = DOWNSAMPLE_1D_MAX_POINTS * 3
    canvas = MplCanvas()
    canvas.show_spectrum(_make_1d_n(n))
    canvas.set_full_resolution(True)
    assert len(canvas.ax.lines[0].get_xdata()) == n

    canvas.set_full_resolution(False)

    assert len(canvas.ax.lines[0].get_xdata()) == DOWNSAMPLE_1D_MAX_POINTS


def test_full_resolution_persists_across_a_fresh_plot(qapp):
    # Matches how the legend-visible/aspect-lock choices already
    # persist across a rebuild (see show_spectrum's own docstring) --
    # this is the same kind of canvas-level preference, not reset
    # every time a new spectrum is shown.
    from opennmr.plotting.style import DOWNSAMPLE_1D_MAX_POINTS
    n = DOWNSAMPLE_1D_MAX_POINTS * 3
    canvas = MplCanvas()
    canvas.show_spectrum(_make_1d_n(n))
    canvas.set_full_resolution(True)

    canvas.show_spectrum(_make_1d_n(n))  # a fresh plot_1d call

    assert len(canvas.ax.lines[0].get_xdata()) == n


def test_export_full_resolution_uses_full_data_then_restores_view(qapp):
    # The actual explicit user request: an export must always be
    # fully sampled, even while the live view (deliberately) isn't.
    from opennmr.plotting.style import DOWNSAMPLE_1D_MAX_POINTS
    n = DOWNSAMPLE_1D_MAX_POINTS * 3
    canvas = MplCanvas()
    canvas.show_spectrum(_make_1d_n(n))
    assert len(canvas.ax.lines[0].get_xdata()) == DOWNSAMPLE_1D_MAX_POINTS  # still downsampled on screen

    seen_lengths = []

    def fake_save():
        seen_lengths.append(len(canvas.ax.lines[0].get_xdata()))

    canvas.export_full_resolution(fake_save)

    assert seen_lengths == [n]  # save_fn saw the true full-resolution data
    # ...but the live view goes right back to downsampled afterward.
    assert len(canvas.ax.lines[0].get_xdata()) == DOWNSAMPLE_1D_MAX_POINTS


def test_export_full_resolution_restores_view_even_if_save_fn_raises(qapp):
    from opennmr.plotting.style import DOWNSAMPLE_1D_MAX_POINTS
    n = DOWNSAMPLE_1D_MAX_POINTS * 3
    canvas = MplCanvas()
    canvas.show_spectrum(_make_1d_n(n))

    def raising_save():
        raise RuntimeError("save cancelled")

    with pytest.raises(RuntimeError):
        canvas.export_full_resolution(raising_save)

    assert len(canvas.ax.lines[0].get_xdata()) == DOWNSAMPLE_1D_MAX_POINTS


def test_export_full_resolution_noop_when_already_full_resolution(qapp):
    # No swap needed (nothing to restore afterward either) when the
    # live view is already full resolution.
    from opennmr.plotting.style import DOWNSAMPLE_1D_MAX_POINTS
    n = DOWNSAMPLE_1D_MAX_POINTS * 3
    canvas = MplCanvas()
    canvas.show_spectrum(_make_1d_n(n))
    canvas.set_full_resolution(True)

    seen_lengths = []
    canvas.export_full_resolution(lambda: seen_lengths.append(len(canvas.ax.lines[0].get_xdata())))

    assert seen_lengths == [n]
    assert len(canvas.ax.lines[0].get_xdata()) == n  # still full resolution after


# ============================================================
# Full resolution -- 2D contour (a QuadContourSet has no in-place
# swap the way a Line2D does, so this is a full show_spectrum rebuild,
# not a data-swap -- see set_full_resolution's own docstring)
# ============================================================

def _contour_shape(canvas):
    return canvas._contour_spectrum.data.shape if canvas._contour_spectrum is not None else None


def test_2d_default_is_downsampled(qapp):
    canvas = MplCanvas()  # default: full_resolution=False, matching every canvas but Spectrum
    canvas.show_spectrum(_make_2d(n_f1=20, n_f2=200))
    fraction_kwargs = canvas._contour_fraction_kwargs()
    # Explicit now (canvas._contour_fraction), not just relying on
    # plot_2d_contour's own hardcoded default -- same effective value
    # (0.25) but user-adjustable via set_contour_fraction/MainWindow's
    # Output tab.
    assert fraction_kwargs == {"contour_fraction_x": 0.25, "contour_fraction_y": 0.25}


def test_full_resolution_true_at_construction_skips_2d_downsampling(qapp):
    # Explicit user feedback: the main Spectrum canvas should show a
    # fully sampled 2D contour by default, unlike every other canvas
    # (e.g. the FID-processing live preview).
    canvas = MplCanvas(full_resolution=True)
    assert canvas._contour_fraction_kwargs() == {"contour_fraction_x": None, "contour_fraction_y": None}


def test_set_full_resolution_rebuilds_2d_contour(qapp):
    canvas = MplCanvas()
    spectrum = _make_2d(n_f1=20, n_f2=200)
    canvas.show_spectrum(spectrum)
    downsampled_shape = _contour_shape(canvas)

    canvas.set_full_resolution(True)

    # _contour_spectrum is the original (never-downsampled) spectrum
    # object either way -- what actually changed is the *rendered*
    # contour, checked via _prepare_contour with the same kwargs
    # show_spectrum itself would now use.
    from opennmr.plotting.twod import _prepare_contour
    _, _, full_res_data, _, _, _ = _prepare_contour(spectrum, **canvas._contour_fraction_kwargs())
    _, _, downsampled_data, _, _, _ = _prepare_contour(spectrum)
    assert full_res_data.shape == spectrum.data.shape
    assert downsampled_data.shape != full_res_data.shape
    assert downsampled_shape == spectrum.data.shape  # _contour_spectrum itself is never downsampled


def test_set_contour_fraction_changes_the_default_downsample_kwargs(qapp):
    canvas = MplCanvas()
    assert canvas._contour_fraction_kwargs() == {"contour_fraction_x": 0.25, "contour_fraction_y": 0.25}

    canvas.set_contour_fraction(0.5)

    assert canvas._contour_fraction_kwargs() == {"contour_fraction_x": 0.5, "contour_fraction_y": 0.5}


def test_set_contour_fraction_rebuilds_a_currently_shown_2d_contour(qapp):
    # _contour_spectrum itself is never downsampled (see
    # test_set_full_resolution_rebuilds_2d_contour's own comment) --
    # check the actually-*rendered* contour data via _prepare_contour,
    # the same way that test does, not _contour_shape.
    from opennmr.plotting.twod import _prepare_contour

    canvas = MplCanvas()
    spectrum = _make_2d(n_f1=20, n_f2=200)
    canvas.show_spectrum(spectrum)
    _, _, quarter_data, _, _, _ = _prepare_contour(spectrum, **canvas._contour_fraction_kwargs())

    canvas.set_contour_fraction(0.5)

    _, _, half_data, _, _, _ = _prepare_contour(spectrum, **canvas._contour_fraction_kwargs())
    assert half_data.shape != quarter_data.shape
    assert half_data.shape[0] > quarter_data.shape[0]
    assert half_data.shape[1] > quarter_data.shape[1]


def test_nq_line_kwargs_default_off(qapp):
    # plotting.plot_2d_contour's own show_nQ_line/nQ params already
    # existed in the backend -- this is the canvas-level plumbing that
    # actually passes them through, off by default matching
    # show_nQ_line's own default.
    canvas = MplCanvas()
    assert canvas._nq_line_kwargs() == {"show_nQ_line": False, "nQ": [1]}


def test_set_nq_line_changes_the_kwargs(qapp):
    canvas = MplCanvas()
    canvas.set_nq_line(True, [2])
    assert canvas._nq_line_kwargs() == {"show_nQ_line": True, "nQ": [2]}


def test_set_nq_line_rebuilds_a_currently_shown_2d_contour(qapp):
    canvas = MplCanvas()
    spectrum = _make_2d(n_f1=20, n_f2=200)
    canvas.show_spectrum(spectrum)
    assert len(canvas.ax.lines) == 0

    canvas.set_nq_line(True)

    assert len(canvas.ax.lines) == 1

    canvas.set_nq_line(False)

    assert len(canvas.ax.lines) == 0


def test_set_contour_fraction_is_not_overridden_by_full_resolution(qapp):
    # Recorded either way (see set_contour_fraction's own docstring) so
    # it's already in effect the moment full resolution is turned back
    # off -- but has no visible effect while full_resolution is on,
    # since that already forces no downsampling regardless.
    from opennmr.plotting.twod import _prepare_contour

    canvas = MplCanvas(full_resolution=True)
    spectrum = _make_2d(n_f1=20, n_f2=200)
    canvas.show_spectrum(spectrum)
    _, _, full_data, _, _, _ = _prepare_contour(spectrum, **canvas._contour_fraction_kwargs())
    assert full_data.shape == spectrum.data.shape

    canvas.set_contour_fraction(0.1)
    _, _, still_full_data, _, _, _ = _prepare_contour(spectrum, **canvas._contour_fraction_kwargs())
    assert still_full_data.shape == spectrum.data.shape  # still full resolution, unaffected

    canvas.set_full_resolution(False)
    _, _, downsampled_data, _, _, _ = _prepare_contour(spectrum, **canvas._contour_fraction_kwargs())
    assert downsampled_data.shape != spectrum.data.shape  # now the 0.1 fraction takes effect


def test_has_downsampled_data_true_for_a_2d_contour(qapp):
    canvas = MplCanvas()
    assert canvas.has_downsampled_data() is False  # nothing plotted yet
    canvas.show_spectrum(_make_2d(n_f1=20, n_f2=200))
    assert canvas.has_downsampled_data() is True


def test_export_full_resolution_2d_uses_full_data_then_restores_view(qapp):
    # The actual explicit user request, for 2D: an export (and,
    # per this same-session fix, the live Spectrum view too) must
    # never be silently downsampled.
    from opennmr.plotting.twod import _prepare_contour

    canvas = MplCanvas()  # default False -- exercises the swap-then-restore path
    spectrum = _make_2d(n_f1=20, n_f2=200)
    canvas.show_spectrum(spectrum)
    canvas.ax.set_xlim(3.0, 7.0)
    canvas.ax.set_ylim(92.0, 98.0)

    seen_shapes = []

    def fake_save():
        _, _, data, _, _, _ = _prepare_contour(spectrum, **canvas._contour_fraction_kwargs())
        seen_shapes.append(data.shape)

    canvas.export_full_resolution(fake_save)

    assert seen_shapes == [spectrum.data.shape]  # save_fn saw the true full-resolution data
    assert canvas._full_resolution is False  # live view setting restored
    assert canvas.ax.get_xlim() == pytest.approx((3.0, 7.0))  # zoom preserved across the rebuild
    assert canvas.ax.get_ylim() == pytest.approx((92.0, 98.0))


def test_export_full_resolution_2d_noop_when_already_full_resolution(qapp):
    from opennmr.plotting.twod import _prepare_contour

    canvas = MplCanvas(full_resolution=True)
    spectrum = _make_2d(n_f1=20, n_f2=200)
    canvas.show_spectrum(spectrum)

    seen_shapes = []

    def fake_save():
        _, _, data, _, _, _ = _prepare_contour(spectrum, **canvas._contour_fraction_kwargs())
        seen_shapes.append(data.shape)

    canvas.export_full_resolution(fake_save)

    assert seen_shapes == [spectrum.data.shape]
    assert canvas._full_resolution is True  # untouched -- nothing needed swapping


# ============================================================
# show_compare "Stacked" mode: one subplot per spectrum
# ============================================================

def _make_1d_at(center_ppm, amplitude=1.0):
    ppm = np.linspace(0, 10, 100)
    metadata = NMRMetadata(vendor="bruker", nucleus="1H", sfo1=400.0)
    axis = NMRAxis(kind="frequency", ppm=ppm, hz=ppm * 400.0)
    return NMRSpectrum(data=amplitude * np.exp(-((ppm - center_ppm) ** 2)), axes=[axis], metadata=metadata)


def test_show_compare_stack_creates_one_subplot_per_spectrum(qapp):
    canvas = MplCanvas()
    specs = [_make_1d_at(c) for c in (3.0, 5.0, 7.0)]
    canvas.show_compare(specs, mode="stack")
    canvas.canvas.draw()
    assert canvas._compare_axes is not None
    assert len(canvas._compare_axes) == 3
    assert all(len(ax.lines) == 1 for ax in canvas._compare_axes)
    assert canvas.ax is canvas._compare_axes[-1]  # bottom row, where x-ticks are shown


def test_show_compare_stack_shares_x_axis(qapp):
    canvas = MplCanvas()
    specs = [_make_1d_at(c) for c in (3.0, 5.0, 7.0)]
    canvas.show_compare(specs, mode="stack")
    canvas.canvas.draw()
    canvas._compare_axes[0].set_xlim(1, 2)
    # sharex=True: changing one row's xlim moves every other row's too.
    assert canvas._compare_axes[1].get_xlim() == pytest.approx((1, 2))
    assert canvas._compare_axes[2].get_xlim() == pytest.approx((1, 2))


def test_show_compare_stack_lines_are_not_clipped_to_their_own_row(qapp):
    # Explicit user request: a tall (e.g. zoomed-in) peak should be
    # able to visually overlap a neighbouring row, matching a classic
    # NMR stacked plot, instead of being hard-clipped at its own row's
    # axes edge.
    canvas = MplCanvas()
    specs = [_make_1d_at(c) for c in (3.0, 5.0, 7.0)]
    canvas.show_compare(specs, mode="stack")
    canvas.canvas.draw()
    for ax in canvas._compare_axes:
        assert ax.lines[0].get_clip_on() is False
        # Also not opaque -- otherwise a row drawn *after* (below) one
        # bleeding into it would just paint over the bleed with its
        # own background.
        assert ax.patch.get_alpha() == 0.0


def test_show_compare_stack_scroll_zooms_every_row_together_by_default(qapp):
    # Explicit user request: scroll-zoom on a "Stacked" Compare view
    # should be LINKED across every row by default (a quick "zoom the
    # whole stack in/out" gesture) -- each row is its own independent
    # axes (so each keeps its own baseline pinned, unaffected by the
    # others' own heights), but scrolling over any one of them zooms
    # all of them together unless a row has been selected first (see
    # test_show_compare_stack_click_row_then_scroll_zooms_only_that_row).
    canvas = MplCanvas()
    specs = [_make_1d_at(c) for c in (3.0, 5.0, 7.0)]
    canvas.show_compare(specs, mode="stack")
    canvas.canvas.draw()
    top_ax, middle_ax, bottom_ax = canvas._compare_axes
    before = {ax: ax.get_ylim() for ax in (top_ax, middle_ax, bottom_ax)}

    canvas._on_scroll(_body_scroll_event(top_ax, xdata=3.0, ydata=before[top_ax][0], button="up"))

    for ax in (top_ax, middle_ax, bottom_ax):
        ylim_before = before[ax]
        range_before = ylim_before[1] - ylim_before[0]
        range_after = ax.get_ylim()[1] - ax.get_ylim()[0]
        assert range_after < range_before  # every row zoomed, not just the hovered one
        assert ax.get_ylim()[0] == pytest.approx(ylim_before[0])  # each row's own baseline stays pinned


def test_show_compare_stack_click_row_then_scroll_zooms_only_that_row(qapp):
    # Clicking a row first decouples it from the linked-by-default
    # scroll-zoom above -- the explicit "some kind of clickable thing
    # per spectrum" request, recovering the original "keep every OTHER
    # baseline fixed" guarantee behind an opt-in instead of it being
    # unconditional.
    canvas = MplCanvas()
    specs = [_make_1d_at(c) for c in (3.0, 5.0, 7.0)]
    canvas.show_compare(specs, mode="stack")
    canvas.canvas.draw()
    canvas.toolbar.mode = ""
    top_ax, middle_ax, bottom_ax = canvas._compare_axes

    # A plain click (press+release with no real movement) on the top
    # row selects it.
    canvas._on_press(SimpleNamespace(inaxes=top_ax, button=1, x=100, y=100))
    canvas._on_release(SimpleNamespace(x=100, y=100))
    assert canvas._selected_compare_index == 0
    assert top_ax.get_facecolor() != middle_ax.get_facecolor()  # visibly highlighted

    top_ylim_before = top_ax.get_ylim()
    middle_ylim_before = middle_ax.get_ylim()
    bottom_ylim_before = bottom_ax.get_ylim()

    # Scrolling while hovering a DIFFERENT (unselected) row still
    # targets the selected one, not whichever row the cursor is
    # actually over.
    canvas._on_scroll(_body_scroll_event(bottom_ax, xdata=7.0, ydata=bottom_ylim_before[0], button="up"))

    assert top_ax.get_ylim() != pytest.approx(top_ylim_before)
    assert middle_ax.get_ylim() == pytest.approx(middle_ylim_before)
    assert bottom_ax.get_ylim() == pytest.approx(bottom_ylim_before)

    # Clicking the same row again de-selects it, back to linked zoom.
    canvas._on_press(SimpleNamespace(inaxes=top_ax, button=1, x=100, y=100))
    canvas._on_release(SimpleNamespace(x=100, y=100))
    assert canvas._selected_compare_index is None


def test_show_compare_stack_click_in_a_rows_margin_selects_it(qapp):
    # Explicit user request: the clickable area should "extend all the
    # way to the next stacked spectrum... to include both the axis and
    # the plotting spaces". Only the data rectangle is `inaxes`, so a
    # click on a row's own y-tick strip used to do nothing at all --
    # there was no `inaxes`, so no pan started and no row was selected.
    canvas = MplCanvas()
    canvas.figure.set_size_inches(8, 6)
    specs = [_make_1d_at(c) for c in (3.0, 5.0, 7.0)]
    canvas.show_compare(specs, mode="stack")
    canvas.canvas.draw()
    canvas.toolbar.mode = ""
    middle_ax = canvas._compare_axes[1]

    # Left of the middle row's data rectangle -- its tick strip -- at
    # that row's own height. inaxes is None, as it is for a real click
    # there.
    bbox = middle_ax.get_window_extent()
    x = bbox.x0 - 5
    y = 0.5 * (bbox.y0 + bbox.y1)
    canvas._on_press(SimpleNamespace(inaxes=None, button=1, x=x, y=y))
    canvas._on_release(SimpleNamespace(x=x, y=y))

    assert canvas._selected_compare_index == 1


def test_compare_row_index_at_y_ignores_a_click_outside_the_stack(qapp):
    # A click well below the bottom row (the shared x-axis label area)
    # should select nothing rather than snapping onto the nearest row.
    canvas = MplCanvas()
    canvas.figure.set_size_inches(8, 6)
    specs = [_make_1d_at(c) for c in (3.0, 5.0, 7.0)]
    canvas.show_compare(specs, mode="stack")
    canvas.canvas.draw()

    assert canvas._compare_row_index_at_y(canvas.figure.bbox.y1 + 50) is None
    # ...and every row's own centre still resolves to itself.
    for i, ax in enumerate(canvas._compare_axes):
        bbox = ax.get_window_extent()
        assert canvas._compare_row_index_at_y(0.5 * (bbox.y0 + bbox.y1)) == i


def test_show_compare_stack_drag_does_not_select_a_row(qapp):
    # A real pan drag (press then release somewhere else) must not
    # also be treated as a row-select click.
    canvas = MplCanvas()
    specs = [_make_1d_at(c) for c in (3.0, 5.0, 7.0)]
    canvas.show_compare(specs, mode="stack")
    canvas.canvas.draw()
    canvas.toolbar.mode = ""
    top_ax = canvas._compare_axes[0]

    canvas._on_press(SimpleNamespace(inaxes=top_ax, button=1, x=100, y=100))
    canvas._on_release(SimpleNamespace(x=140, y=100))
    assert canvas._selected_compare_index is None


def test_show_compare_overlay_still_uses_one_shared_axes(qapp):
    # Overlay mode has only one baseline (y=0) shared by every trace,
    # so the single-shared-axes problem this whole redesign is about
    # doesn't apply there -- it's unchanged, still one axes.
    canvas = MplCanvas()
    specs = [_make_1d_at(c) for c in (3.0, 5.0, 7.0)]
    canvas.show_compare(specs, mode="overlay")
    canvas.canvas.draw()
    assert canvas._compare_axes is None
    assert len(canvas.ax.lines) == 3


def test_show_compare_stack_single_spectrum_uses_one_shared_axes(qapp):
    # No "other traces" to protect a baseline from -- one shared axes
    # (matching every other single-spectrum view) rather than a
    # pointless one-row subplot grid.
    canvas = MplCanvas()
    canvas.show_compare([_make_1d_at(5.0)], mode="stack")
    canvas.canvas.draw()
    assert canvas._compare_axes is None
    assert len(canvas.ax.lines) == 1


def test_show_spectrum_after_compare_clears_compare_axes(qapp):
    canvas = MplCanvas()
    canvas.show_compare([_make_1d_at(c) for c in (3.0, 5.0)], mode="stack")
    canvas.canvas.draw()
    assert canvas._compare_axes is not None

    canvas.show_spectrum(_make_1d())
    assert canvas._compare_axes is None


def _region_drag_events(ax, x_start, x_mid, x_end):
    """Press/motion/release SimpleNamespaces for a region-select drag,
    with real pixel positions derived from ax's own transform (like
    the scroll-event helpers above) -- _on_press/_on_motion/_on_release
    read pixel coordinates, not just xdata, so a realistic drag needs
    both."""
    def event(xdata, button=1):
        x, y = ax.transData.transform((xdata, 0.5))
        return SimpleNamespace(inaxes=ax, xdata=xdata, ydata=0.5, x=x, y=y, button=button)
    return event(x_start), event(x_mid), event(x_end)


def test_region_select_mode_sets_crosshair_cursor(qapp):
    canvas = MplCanvas()
    canvas.show_spectrum(_make_1d())
    canvas.set_region_select_mode(True)
    assert canvas.canvas.cursor().shape() == Qt.CursorShape.CrossCursor
    canvas.set_region_select_mode(False)
    assert canvas.canvas.cursor().shape() == Qt.CursorShape.ArrowCursor


def test_region_select_drag_emits_sorted_region_and_does_not_pan(qapp):
    canvas = MplCanvas()
    canvas.show_spectrum(_make_1d())
    canvas.set_region_select_mode(True)
    original_xlim = canvas.ax.get_xlim()

    emitted = []
    canvas.regionSelected.connect(lambda lo, hi: emitted.append((lo, hi)))

    press, motion, release = _region_drag_events(canvas.ax, 7.0, 5.0, 3.0)  # dragged right-to-left
    canvas._on_press(press)
    assert canvas._region_select_patch is not None  # live rubber-band shown mid-drag
    canvas._on_motion(motion)
    canvas._on_release(release)

    assert emitted == [(3.0, 7.0)]  # sorted low-to-high regardless of drag direction
    assert canvas._region_select_patch is None  # cleared after release
    assert canvas.ax.get_xlim() == original_xlim  # never panned while in this mode


def test_region_select_mode_off_falls_back_to_pan(qapp):
    canvas = MplCanvas()
    canvas.show_spectrum(_make_1d())
    press = SimpleNamespace(inaxes=canvas.ax, button=1, x=100, y=100)
    canvas._on_press(press)
    assert canvas._pan_start is not None
    assert canvas._region_select_start is None


def test_set_regions_shades_and_replaces_previous_spans(qapp):
    canvas = MplCanvas()
    canvas.show_spectrum(_make_1d())

    canvas.set_regions([(2.0, 4.0), (6.0, 8.0)])
    tagged = [p for p in canvas.ax.patches if p.get_gid() == canvas._REGION_PATCH_GID]
    assert len(tagged) == 2

    canvas.set_regions([(1.0, 2.0)])
    tagged = [p for p in canvas.ax.patches if p.get_gid() == canvas._REGION_PATCH_GID]
    assert len(tagged) == 1

    canvas.set_regions([])
    tagged = [p for p in canvas.ax.patches if p.get_gid() == canvas._REGION_PATCH_GID]
    assert len(tagged) == 0


# ============================================================
# set_style (Output tab live style panel)
# ============================================================

def test_set_style_updates_instance_attributes(qapp):
    canvas = MplCanvas()
    canvas.set_style(linewidth=3.0, labelsize=20, tick_length=9.0, tick_width=2.5, grid=True)
    assert canvas._AXIS_LINEWIDTH == 3.0
    assert canvas._AXIS_LABELSIZE == 20
    assert canvas._TICK_LENGTH == 9.0
    assert canvas._TICK_WIDTH == 2.5
    assert canvas._grid_visible is True


def test_set_style_future_draw_picks_up_new_values(qapp):
    canvas = MplCanvas()
    canvas.set_style(linewidth=3.0)
    canvas.show_spectrum(_make_1d())
    assert canvas.ax.spines["bottom"].get_linewidth() == pytest.approx(3.0)


def test_set_style_minor_ticks_toggle(qapp):
    canvas = MplCanvas()
    canvas.show_spectrum(_make_1d())
    assert canvas._minor_ticks_visible is True  # default

    canvas.set_style(minor_ticks=False)
    assert canvas._minor_ticks_visible is False
    assert canvas.ax.xaxis.get_minor_locator().__class__.__name__ == "NullLocator"

    canvas.set_style(minor_ticks=True)
    assert canvas._minor_ticks_visible is True
    assert canvas.ax.xaxis.get_minor_locator().__class__.__name__ != "NullLocator"


def test_set_style_updates_already_drawn_axes_live(qapp):
    canvas = MplCanvas()
    canvas.show_spectrum(_make_1d())
    canvas.set_style(linewidth=5.0, grid=True)
    assert canvas.ax.spines["bottom"].get_linewidth() == pytest.approx(5.0)
    assert any(line.get_visible() for line in canvas.ax.get_xgridlines())


def test_set_style_grid_toggle_does_not_warn(qapp, recwarn):
    canvas = MplCanvas()
    canvas.show_spectrum(_make_1d())
    canvas.set_style(grid=True)
    canvas.set_style(grid=False)
    grid_warnings = [w for w in recwarn.list if "grid()" in str(w.message)]
    assert grid_warnings == []


def test_set_export_settings_does_not_touch_live_figure(qapp):
    # Regression test: DPI/figure size used to apply straight to the
    # live embedded figure, which visibly fought Qt's own automatic
    # canvas-widget resizing. Now export-only, applied (and restored)
    # only around an actual save -- see export_full_resolution.
    canvas = MplCanvas()
    original_dpi = canvas.figure.get_dpi()
    original_size = canvas.figure.get_size_inches()

    canvas.set_export_settings(dpi=150, fig_width=8.0, fig_height=5.0)

    assert canvas.figure.get_dpi() == pytest.approx(original_dpi)
    assert canvas.figure.get_size_inches() == pytest.approx(original_size)
    assert canvas._export_dpi == pytest.approx(150)
    assert canvas._export_fig_width == pytest.approx(8.0)
    assert canvas._export_fig_height == pytest.approx(5.0)


def test_export_full_resolution_applies_export_settings_around_save_only(qapp):
    canvas = MplCanvas()
    original_dpi = canvas.figure.get_dpi()
    original_size = canvas.figure.get_size_inches()
    canvas.set_export_settings(dpi=150, fig_width=8.0, fig_height=5.0)

    seen = {}

    def fake_save():
        seen["dpi"] = canvas.figure.get_dpi()
        seen["size"] = canvas.figure.get_size_inches()

    canvas.export_full_resolution(fake_save)

    assert seen["dpi"] == pytest.approx(150)
    assert tuple(seen["size"]) == pytest.approx((8.0, 5.0))
    # Restored afterward -- the live view must not stay at export size.
    assert canvas.figure.get_dpi() == pytest.approx(original_dpi)
    assert canvas.figure.get_size_inches() == pytest.approx(original_size)


def test_export_full_resolution_restores_settings_even_if_save_raises(qapp):
    canvas = MplCanvas()
    original_dpi = canvas.figure.get_dpi()
    canvas.set_export_settings(dpi=150)

    def failing_save():
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        canvas.export_full_resolution(failing_save)

    assert canvas.figure.get_dpi() == pytest.approx(original_dpi)


def test_set_style_compare_colormap_used_on_next_redraw(qapp):
    canvas = MplCanvas()
    canvas.set_style(compare_colormap="plasma")
    assert canvas._compare_colormap == "plasma"


def _2d_for_slicing(n_f1=8, n_f2=32):
    y = np.linspace(150, 100, n_f1)
    x = np.linspace(10, 0, n_f2)
    yy, xx = np.meshgrid(np.arange(n_f1), np.arange(n_f2), indexing="ij")
    data = np.exp(-(((xx - n_f2 / 2) ** 2 + (yy - n_f1 / 2) ** 2)) / 8.0) * 100
    return NMRSpectrum(
        data=data,
        axes=[
            NMRAxis(kind="frequency", ppm=y, hz=y * 100.0),
            NMRAxis(kind="frequency", ppm=x, hz=x * 400.0),
        ],
        metadata=NMRMetadata(vendor="bruker", nucleus="1H", nucleus_indirect="13C",
                             sfo1=400.0, sfo1_indirect=100.0),
        data_imag=np.zeros_like(data),
    )


def test_canvas_slice_bar_appears_only_while_a_slice_is_viewed(qapp):
    # Real user report: "There is no way to actually phase the extracted
    # slice as it stands. maybe the extracted view should have new
    # options to do something with the slice". The capability existed in
    # the right-click menu and worked -- but a context menu on a plot is
    # not somewhere anyone looks, so a visible bar is what makes it real.
    canvas = MplCanvas()
    canvas.show_spectrum(_2d_for_slicing())
    # isHidden(), not isVisible(): this canvas is never shown in a
    # window here, and Qt reports every child of an unshown widget as
    # not visible regardless of its own state.
    assert canvas._slice_bar.isHidden() is True

    canvas.extract_slice("f1", 2)
    assert canvas._slice_bar.isHidden() is False
    assert "F1 slice 2" in canvas._slice_label.text()

    # Returning to the contour hides it again -- the bar must never be
    # left showing over an unrelated view.
    canvas._back_to_contour_button.click()
    assert canvas._slice_bar.isHidden() is True
    assert canvas._contour_spectrum is not None


def test_canvas_keep_slice_button_emits_the_displayed_trace(qapp):
    canvas = MplCanvas()
    source = _2d_for_slicing()
    canvas.show_spectrum(source)
    canvas.extract_slice("f1", 3)

    received = []
    canvas.sliceKeepRequested.connect(received.append)
    canvas._keep_slice_button.click()

    assert len(received) == 1
    # The payload carries provenance too, so a kept slice can push its
    # phase back into the 2D it came from.
    trace, label, parent, axis, index = received[0]
    assert label == "F1 slice 3"
    assert parent is source
    assert (axis, index) == ("f1", 3)
    assert trace.ndim == 1
    # The trace must carry its imaginary part, or the whole point --
    # phasing it -- does not work.
    assert trace.data_imag is not None
    np.testing.assert_allclose(trace.data, source.data[2])


def test_canvas_keep_slice_helper_does_not_recurse(qapp):
    # Regression test for a real self-inflicted bug: a global text
    # replace rewrote the emit *inside* the helper into a call to the
    # helper itself, so clicking Keep recursed until RecursionError.
    canvas = MplCanvas()
    canvas.show_spectrum(_2d_for_slicing())
    canvas.extract_slice("f1", 1)
    received = []
    canvas.sliceKeepRequested.connect(received.append)
    canvas._emit_slice_keep_requested()
    assert len(received) == 1


def test_canvas_keep_slice_is_a_noop_without_a_slice(qapp):
    canvas = MplCanvas()
    canvas.show_spectrum(_2d_for_slicing())
    received = []
    canvas.sliceKeepRequested.connect(received.append)
    canvas._emit_slice_keep_requested()
    assert received == []


def test_canvas_extract_slice_can_re_slice_while_showing_a_slice(qapp):
    # show_spectrum(trace) clears _contour_spectrum (the trace is 1D),
    # so extract_slice's source was None on any second call and
    # get_trace raised "expects a 2D NMRSpectrum". That made the
    # right-click "Extract Slice..." a one-shot, and silently pinned
    # peak-picking to whichever slice it started on.
    canvas = MplCanvas()
    canvas.show_spectrum(_make_2d(n_f1=6))

    canvas.extract_slice("f1", 1)
    assert canvas._extracted_index == 1
    source = canvas._extracted_from_spectrum
    assert source is not None

    canvas.extract_slice("f1", 4)
    assert canvas._extracted_index == 4
    assert canvas._extracted_from_spectrum is source     # same 2D, not lost


def test_canvas_adding_a_slice_preserves_the_view(qapp):
    # "The scale/view shouldn't reset when a slice is extracted" --
    # adding one is an annotation, not a change of subject.
    canvas = MplCanvas()
    canvas.show_spectrum(_make_2d(n_f1=8))
    canvas.ax.set_xlim(6.0, 2.0)
    canvas.ax.set_ylim(40.0, 10.0)
    before = (canvas.ax.get_xlim(), canvas.ax.get_ylim())

    canvas._add_contour_slice("f1", 3, "A")
    assert (canvas.ax.get_xlim(), canvas.ax.get_ylim()) == before

    canvas._clear_contour_slices()
    assert (canvas.ax.get_xlim(), canvas.ax.get_ylim()) == before


def _motion_at(canvas, ax, xfrac=0.5, yfrac=0.5):
    bbox = ax.get_window_extent()
    x, y = bbox.x0 + xfrac * bbox.width, bbox.y0 + yfrac * bbox.height
    event = types.SimpleNamespace(name="motion_notify_event", canvas=canvas.canvas,
                                  inaxes=ax, x=x, y=y)
    event.xdata, event.ydata = ax.transData.inverted().transform((x, y))
    return event


def test_canvas_crosshair_is_not_created_until_the_mouse_moves(qapp):
    # Creating it the moment plot_2d_contour returns would add its
    # (invisible-until-hovered) Line2D artists to ax.lines for a view
    # nobody has looked at yet -- breaking anything that counts lines
    # on a freshly built contour (see the nQ-diagonal toggle test).
    canvas = MplCanvas()
    canvas.show_spectrum(_make_2d(n_f1=8))
    assert canvas._crosshair is None
    assert len(canvas.ax.lines) == 0

    canvas._on_motion(_motion_at(canvas, canvas.ax))
    assert canvas._crosshair is not None


def test_canvas_crosshair_spans_the_contour_and_both_projections(qapp):
    # "When the input isn't panning or zooming, the cursor should be a
    # cross with lines covering the entire contour space AND the
    # projections."
    canvas = MplCanvas()
    canvas.show_spectrum(_make_2d(n_f1=8))
    canvas._on_motion(_motion_at(canvas, canvas.ax))

    crosshair = canvas._crosshair
    assert set(crosshair.axes) == {canvas.ax, canvas._proj_axes["f1"], canvas._proj_axes["f2"]}
    assert set(crosshair._vline_axes) == {canvas.ax, canvas._proj_axes["f2"]}
    assert set(crosshair._hline_axes) == {canvas.ax, canvas._proj_axes["f1"]}


def test_canvas_crosshair_establishes_its_own_background_on_first_move(qapp):
    # Real, live user report: "when I extract or clear a slice, until I
    # move the cursor away from the plot it does the same 'solitaire
    # effect', covering the plot." A freshly created instance starts
    # with no cached background -- extracting/clearing a slice rebuilds
    # the figure and only lazily recreates the crosshair on the next
    # mouse move, so nothing has forced a real draw() (the only thing
    # that fires draw_event -> _on_draw) since creation. Without a
    # background, onmove can't restore_region before drawing, so every
    # move painted directly onto whatever the Agg buffer already held
    # -- accumulating lines that only ever got wiped by leaving the
    # axes (_hide's own draw_idle). Fixed by onmove forcing a
    # synchronous draw() itself the first time it finds no background.
    canvas = MplCanvas()
    canvas.figure.set_size_inches(8, 6)
    canvas.show_spectrum(_make_2d(n_f1=8))
    canvas.canvas.draw()
    canvas._add_contour_slice("f1", 3, "A")  # rebuilds; no motion in between
    canvas._on_motion(_motion_at(canvas, canvas.ax))
    crosshair = canvas._crosshair
    assert crosshair._background is None

    draw_calls = []
    real_draw = canvas.canvas.draw
    canvas.canvas.draw = lambda *a, **k: (draw_calls.append(1), real_draw(*a, **k))[1]
    try:
        crosshair._onmove(_motion_at(canvas, canvas.ax, xfrac=0.3, yfrac=0.3))
        assert draw_calls == [1]
        assert crosshair._background is not None

        crosshair._onmove(_motion_at(canvas, canvas.ax, xfrac=0.6, yfrac=0.6))
        assert draw_calls == [1]  # not called again -- the background is reused
    finally:
        canvas.canvas.draw = real_draw


def test_canvas_crosshair_draws_both_lines_through_one_shared_background(qapp):
    # Real user report: "crosshair is just filling the contour and not
    # going away". Traced to two independent MultiCursor instances
    # (the first two attempts at this) each keeping their own blit
    # background for the same overlapping Axes -- matplotlib's own
    # MultiCursor.onmove unconditionally restore_regions its own
    # background before drawing its own line, so whichever cursor's
    # motion callback ran second always wiped out the first one's line
    # before that frame was ever shown. One widget, one background,
    # one restore + both lines drawn + one blit per move -- checked
    # here by call count, since the wipe-out itself is a screen-pixel
    # effect a headless test can't see directly.
    canvas = MplCanvas()
    canvas.figure.set_size_inches(8, 6)
    canvas.show_spectrum(_make_2d(n_f1=8))
    canvas.canvas.draw()
    canvas._on_motion(_motion_at(canvas, canvas.ax))
    crosshair = canvas._crosshair
    crosshair._background = canvas.canvas.copy_from_bbox(canvas.figure.bbox)

    restore_calls = []
    blit_calls = []
    real_restore = canvas.canvas.restore_region
    real_blit = canvas.canvas.blit
    canvas.canvas.restore_region = lambda *a, **k: (restore_calls.append(1), real_restore(*a, **k))[1]
    canvas.canvas.blit = lambda *a, **k: (blit_calls.append(1), real_blit(*a, **k))[1]
    try:
        crosshair._onmove(_motion_at(canvas, canvas.ax, xfrac=0.3, yfrac=0.7))
    finally:
        canvas.canvas.restore_region = real_restore
        canvas.canvas.blit = real_blit

    assert restore_calls == [1]
    assert blit_calls == [1]
    assert all(line.get_visible() for line in (*crosshair._vlines, *crosshair._hlines))


def test_canvas_crosshair_stands_down_during_pan_and_zoom(qapp):
    canvas = MplCanvas()
    canvas.figure.set_size_inches(8, 6)
    canvas.show_spectrum(_make_2d(n_f1=8))
    canvas.canvas.draw()
    canvas._on_motion(_motion_at(canvas, canvas.ax))
    crosshair = canvas._crosshair
    event = _motion_at(canvas, canvas.ax)

    canvas.toolbar.mode = ""
    crosshair._onmove(event)
    assert crosshair._vlines[0].get_visible()

    for mode in ("pan/zoom", "zoom rect"):
        canvas.toolbar.mode = mode
        crosshair._onmove(event)
        assert not crosshair._vlines[0].get_visible(), mode
    canvas.toolbar.mode = ""


def test_canvas_crosshair_is_torn_down_synchronously_by_a_rebuild(qapp):
    # The actual bug behind "filling the contour and not going away",
    # once traced through matplotlib's own MultiCursor: the previous
    # design tore a stale crosshair down only lazily, the next time
    # _on_motion ran -- and a contour can rebuild several times with
    # no mouse movement in between (e.g. repeated contour-level-adjust
    # scrolls). A crosshair still connected across that gap keeps
    # receiving real callbacks and drawing through its old Axes' own
    # (by-then meaningless) transform, straight onto the live renderer.
    # Teardown must happen in the same call that destroys the Axes --
    # never wait for the next motion event to notice.
    canvas = MplCanvas()
    canvas.figure.set_size_inches(8, 6)
    canvas.show_spectrum(_make_2d(n_f1=8))
    canvas._on_motion(_motion_at(canvas, canvas.ax))
    stale = canvas._crosshair
    assert stale is not None
    motion_cid = stale._cid_motion

    def connected() -> bool:
        return motion_cid in canvas.canvas.callbacks.callbacks.get("motion_notify_event", {})

    assert connected()

    # Two rebuilds in a row, exactly the "no intervening mouse move"
    # case a scroll-wheel contour-level adjust produces.
    canvas._redraw_contour(preserve_view=True)
    assert canvas._crosshair is None
    assert not connected()
    canvas._redraw_contour(preserve_view=True)
    assert canvas._crosshair is None
    assert not connected()

    # A later motion event gets a fresh crosshair bound to the axes
    # actually on screen now, not the ones from before the rebuilds.
    canvas._on_motion(_motion_at(canvas, canvas.ax))
    assert canvas._crosshair is not None
    assert canvas._crosshair is not stale
    assert all(a in canvas.figure.axes for a in canvas._crosshair.axes)


def test_canvas_rebuild_cancels_an_interrupted_zoom_rubberband(qapp):
    # Real user report: "the zoom dashed line box stays sometimes when
    # switching around". NavigationToolbar2.release_zoom returns early
    # when _zoom_info is None and is the only caller of
    # remove_rubberband -- so a zoom drag whose release never arrives
    # (the figure is rebuilt under it by a spectrum/tab switch) leaves
    # the Qt backend's _draw_rect_callback installed, and every later
    # repaint draws that same dashed box over unrelated content.
    canvas = MplCanvas()
    canvas.show_spectrum(_make_2d(n_f1=8))
    canvas.canvas.draw()

    # Stand in for a zoom drag that got interrupted mid-flight.
    canvas.toolbar._zoom_info = SimpleNamespace(axes=canvas.ax, cid=None, start_xy=(0, 0), cbar=None)
    canvas.canvas.drawRectangle((10, 10, 40, 40))
    assert canvas.canvas._draw_rect_callback is not None

    rect_calls = []
    canvas.canvas._draw_rect_callback = lambda painter: rect_calls.append(1)

    canvas.show_spectrum(_make_2d(n_f1=8))  # the switch that used to strand it

    assert canvas.toolbar._zoom_info is None
    # The callback was replaced with the backend's own no-op, so a
    # later repaint draws nothing: calling it must not reach ours.
    canvas.canvas._draw_rect_callback(None)
    assert rect_calls == []


def test_canvas_crosshair_is_torn_down_when_the_view_changes_away_from_a_contour(qapp):
    canvas = MplCanvas()
    canvas.show_spectrum(_make_2d(n_f1=8))
    canvas._on_motion(_motion_at(canvas, canvas.ax))
    assert canvas._crosshair is not None

    canvas.show_spectrum(_make_1d())
    assert canvas._crosshair is None


def _scroll_at(canvas, ax, xfrac=0.5, yfrac=0.5, button="up", inaxes=True):
    bbox = ax.get_window_extent()
    x, y = bbox.x0 + xfrac * bbox.width, bbox.y0 + yfrac * bbox.height
    event = types.SimpleNamespace(button=button, step=1, name="scroll_event",
                                  canvas=canvas.canvas, inaxes=ax if inaxes else None, x=x, y=y)
    event.xdata, event.ydata = ax.transData.inverted().transform((x, y))
    return event


def test_canvas_scroll_over_the_slice_panel_is_a_no_op(qapp):
    # The slice panel lives in _proj_axes alongside the F1/F2 strips,
    # but it is not a projection: its y axis is a normalised 0..1 stack
    # of evenly-spread traces, so an intensity zoom would silently
    # rescale it and break the layout. It also sits exactly where the
    # contour's right-hand "y strip" zoom region used to be empty.
    canvas = MplCanvas()
    canvas.figure.set_size_inches(9, 6)
    canvas.show_spectrum(_make_2d(n_f1=16, n_f2=128))
    canvas._add_contour_slice("f1", 3, "A")
    canvas.canvas.draw()

    panel = canvas._proj_axes["slices"]
    before = (panel.get_xlim(), panel.get_ylim())
    canvas._on_scroll(_scroll_at(canvas, panel))
    assert (panel.get_xlim(), panel.get_ylim()) == before


def test_canvas_projection_zoom_still_works_with_a_slice_panel(qapp):
    # ...and the strips it sits beside must keep their own behaviour --
    # the projection loop is named f1/f2 explicitly now, so nothing else
    # appearing in _proj_axes can be mistaken for a strip.
    canvas = MplCanvas()
    canvas.figure.set_size_inches(9, 6)
    canvas.show_spectrum(_make_2d(n_f1=16, n_f2=128))
    canvas._add_contour_slice("f1", 3, "A")
    canvas.canvas.draw()

    for kind in ("f1", "f2"):
        ax = canvas._proj_axes[kind]
        before = (ax.get_xlim(), ax.get_ylim())
        canvas._on_scroll(_scroll_at(canvas, ax))
        assert (ax.get_xlim(), ax.get_ylim()) != before, kind

    # The contour's own x zoom keeps working, and the panel follows it.
    panel = canvas._proj_axes["slices"]
    before_x = canvas.ax.get_xlim()
    canvas._on_scroll(_scroll_at(canvas, canvas.ax, yfrac=-0.07, inaxes=False))
    assert canvas.ax.get_xlim() != before_x
    assert panel.get_xlim() == canvas.ax.get_xlim()


def test_canvas_contour_redraw_keeps_the_output_tab_style(qapp):
    # Real user report: "scrolling the contours redraws the axis with
    # defaults, not the ones set in output". Every contour rebuild --
    # including the one a contour-level scroll triggers -- went through
    # plot_2d_contour without the style, so it fell back to that
    # function's own manuscript defaults and silently undid the Output
    # tab. The 1D views never had this problem because they read
    # self._AXIS_LABELSIZE and friends directly.
    canvas = MplCanvas()
    canvas.figure.set_size_inches(9, 6)
    canvas.show_spectrum(_make_2d(n_f1=16, n_f2=128))
    canvas.set_style(linewidth=3.0, labelsize=22, tick_width=3.0, tick_length=9.0)

    def styled(ax):
        return (ax.spines["bottom"].get_linewidth(),
                ax.xaxis.label.get_fontsize(),
                ax.xaxis.get_major_ticks()[0].tick1line.get_markeredgewidth())

    expected = styled(canvas.ax)
    assert expected == (3.0, 22.0, 3.0)

    canvas._on_scroll(_scroll_at(canvas, canvas.ax))      # contour-level adjust -> rebuild
    assert styled(canvas.ax) == expected
    assert styled(canvas._proj_axes["f2"]) == expected    # projection strips too

    canvas._add_contour_slice("f1", 3, "A")               # another rebuild path
    assert styled(canvas.ax) == expected


def test_canvas_export_lays_out_at_the_size_it_writes(qapp):
    # Reported directly: "saved png/pdf cuts off bottom of x-axis label
    # below like 10x10 in". Subplot margins are stored as *fractions*
    # of the figure and were computed against whatever size Qt gave the
    # live widget -- but the text keeps its absolute point size, so a
    # layout done while the widget was large leaves too little room for
    # the same label at a smaller export size.
    from opennmr.plotting.style import clipped_labels

    canvas = MplCanvas()
    canvas.figure.set_size_inches(16, 11)          # a large live widget
    canvas.show_spectrum(_make_1d())
    canvas.set_export_settings(fig_width=6.0, fig_height=4.5)

    captured = {}

    def fake_save():
        canvas.figure.draw_without_rendering()
        captured["size"] = tuple(canvas.figure.get_size_inches())
        captured["clipped"] = [t.get_text() for t in clipped_labels(canvas.figure)]

    canvas.export_full_resolution(fake_save)

    assert captured["size"] == pytest.approx((6.0, 4.5))
    assert captured["clipped"] == []          # nothing written off the edge


def test_canvas_export_restores_the_on_screen_layout(qapp):
    # The export's own margins must not be left behind on the live view.
    canvas = MplCanvas()
    canvas.figure.set_size_inches(16, 11)
    canvas.show_spectrum(_make_1d())
    before = canvas.figure.subplotpars.bottom
    canvas.set_export_settings(fig_width=5.0, fig_height=4.0)

    canvas.export_full_resolution(lambda: None)

    assert tuple(canvas.figure.get_size_inches()) == pytest.approx((16.0, 11.0))
    assert canvas.figure.subplotpars.bottom == pytest.approx(before, abs=0.02)


def test_canvas_text_comes_back_when_the_widget_grows(qapp):
    # Reported directly: "DOSY axes are still reverting to the small
    # fonts sometimes". fit_labels_within_figure only ever shrinks, so
    # a figure laid out once while its tab was hidden (and therefore
    # tiny) kept that shrunken text however large the widget later got.
    canvas = MplCanvas()
    # Below ~1.4 x 0.9 in, a 1D canvas genuinely cannot fit its x label
    # -- the measured threshold in fit_labels_within_figure's own
    # docstring, and the size a not-yet-shown tab behaves like.
    canvas.figure.set_size_inches(1.2, 0.8)
    canvas.show_spectrum(_make_1d())
    shrunk = canvas.ax.xaxis.label.get_fontsize()
    assert shrunk < canvas._AXIS_LABELSIZE        # it really did shrink

    canvas.figure.set_size_inches(10, 7)
    canvas._run_debounced_resize_layout()         # what the resize event triggers

    assert canvas.ax.xaxis.label.get_fontsize() == pytest.approx(canvas._AXIS_LABELSIZE)
    assert all(t.get_fontsize() == pytest.approx(canvas._AXIS_LABELSIZE)
               for t in canvas.ax.get_xticklabels())


def test_canvas_resize_relayout_is_debounced(qapp):
    # A window drag fires a resize per pixel, and each layout pass is a
    # real cost on a large contour.
    canvas = MplCanvas()
    canvas.show_spectrum(_make_1d())
    assert canvas._resize_layout_timer.isActive() is False
    canvas._on_canvas_resized()
    assert canvas._resize_layout_timer.isActive() is True
    canvas._on_canvas_resized()                   # restarts, does not stack
    assert canvas._resize_layout_timer.isActive() is True


def test_canvas_default_export_size_switches_for_a_sliced_2d(qapp):
    canvas = MplCanvas()
    canvas.show_spectrum(_make_2d(n_f1=8))
    assert canvas.default_export_size() == (8.0, 8.0)

    # Landscape: the sliced layout is contour + trace panel side by
    # side, so it is wider than tall (the earlier 7.5 x 10 portrait
    # predated the panel actually sitting beside the contour).
    canvas._add_contour_slice("f1", 3, "A")
    assert canvas.default_export_size() == (10.0, 7.5)

    canvas._clear_contour_slices()
    assert canvas.default_export_size() == (8.0, 8.0)


def test_canvas_export_applies_the_default_size_without_an_override(qapp):
    # The size used to be left alone entirely unless the Output tab had
    # been edited, so an exported figure inherited whatever size Qt's
    # layout happened to give the live widget.
    canvas = MplCanvas()
    canvas.figure.set_size_inches(9, 6)
    canvas.show_spectrum(_make_2d(n_f1=8))

    seen = []
    canvas.export_full_resolution(lambda: seen.append(tuple(canvas.figure.get_size_inches())))
    assert seen == [(8.0, 8.0)]
    # ...and restored afterwards, never left on the live figure.
    assert tuple(canvas.figure.get_size_inches()) == (9.0, 6.0)

    seen.clear()
    canvas.set_export_settings(fig_width=12.0)
    canvas.export_full_resolution(lambda: seen.append(tuple(canvas.figure.get_size_inches())))
    assert seen == [(12.0, 8.0)]     # explicit width, defaulted height


def _xlabel_is_inside_the_figure(canvas, ax):
    canvas.canvas.draw()
    label_box = ax.xaxis.label.get_window_extent()
    return label_box.y0 >= 0 and label_box.x0 >= 0


def test_canvas_shrinks_labels_rather_than_clipping_them(qapp):
    # User request for "better safeguards to ensure that the axis
    # labels don't get cut off... maybe some kind of warning or push to
    # resize the font". tight_layout alone reserves room only when the
    # figure is big enough and quietly gives up when it is not -- and a
    # GUI canvas is sized by the window, so that is an ordinary state.
    from opennmr.plotting import clipped_labels

    canvas = MplCanvas()
    canvas.figure.set_size_inches(1.4, 0.9)
    canvas.show_spectrum(_make_1d())

    assert clipped_labels(canvas.figure) == []
    # Shrunk to achieve it, rather than the label being left hanging
    # off the bottom.
    assert canvas.ax.xaxis.label.get_fontsize() < canvas._AXIS_LABELSIZE


def test_canvas_reports_labels_it_cannot_fit_at_any_size(qapp):
    # Below a certain size no font helps; that is a real "make this
    # bigger" situation only the user can fix, so it is announced
    # rather than silently rendered with labels off the edge.
    canvas = MplCanvas()
    canvas.figure.set_size_inches(0.7, 0.45)
    seen = []
    canvas.labelsClipped.connect(seen.append)

    canvas.show_spectrum(_make_1d())

    assert seen and seen[-1]  # non-empty: named the labels that don't fit


def test_canvas_labels_clipped_only_fires_on_a_change(qapp):
    # A phase drag relayouts continuously; re-announcing the same
    # problem every frame would be noise. And when the canvas has room
    # again, an empty emission clears the message rather than leaving a
    # stale warning up.
    canvas = MplCanvas()
    canvas.figure.set_size_inches(0.7, 0.45)
    canvas.show_spectrum(_make_1d())

    seen = []
    canvas.labelsClipped.connect(seen.append)

    canvas._apply_layout()
    canvas._apply_layout()
    assert seen == []  # already clipped, unchanged -- says nothing again

    canvas.figure.set_size_inches(8, 6)
    canvas._apply_layout()
    assert seen == [[]]  # one all-clear, exactly once


def test_canvas_contour_rebuild_keeps_the_x_label_on_the_figure(qapp):
    # Real user report: "the x-axis label got cut off". _redraw_contour
    # never re-ran the layout pass show_spectrum ends with, so a rebuild
    # kept whatever subplot margins were last in force -- and once the
    # slice panel (which brings its own x-axis label) changed what the
    # figure needs, the contour's own label fell off the bottom.
    canvas = MplCanvas()
    canvas.figure.set_size_inches(9, 6)
    canvas.show_spectrum(_make_2d(n_f1=16, n_f2=128))
    assert _xlabel_is_inside_the_figure(canvas, canvas.ax)

    canvas._add_contour_slice("f1", 3, "A")
    assert _xlabel_is_inside_the_figure(canvas, canvas.ax)
    assert _xlabel_is_inside_the_figure(canvas, canvas._proj_axes["slices"])

    # Asserted as "the rebuild runs the layout pass", not as an exact
    # axes position: tight_layout is not idempotent on an AxesDivider
    # figure, so comparing a rebuild's geometry against a fresh
    # show_spectrum's differs by a hair either way and discriminates
    # nothing. Whether the label is *visibly* clipped depends on the
    # figure size and on how much the slice panel's own x label shifts
    # the margins, which is exactly why the layout has to re-run.
    calls = []
    original = canvas.figure.tight_layout
    canvas.figure.tight_layout = lambda *a, **k: (calls.append(1), original(*a, **k))[1]
    try:
        canvas._redraw_contour(preserve_view=True)
    finally:
        canvas.figure.tight_layout = original
    assert calls, "a contour rebuild must re-run the layout, as show_spectrum does"


def test_canvas_reset_view_restores_limits_and_layout(qapp):
    # "Be sure that the contour plot position resets back to the
    # original load position." toolbar.home() only restores axis limits
    # from the navigation stack -- and that stack still holds Axes from
    # before this canvas's own figure rebuilds, so it cannot restore a
    # layout at all.
    canvas = MplCanvas()
    canvas.figure.set_size_inches(9, 6)
    canvas.show_spectrum(_make_2d(n_f1=16, n_f2=128))
    original = (canvas.ax.get_xlim(), canvas.ax.get_ylim())

    canvas._add_contour_slice("f1", 3, "A")
    canvas.ax.set_xlim(6.0, 2.0)
    canvas.ax.set_ylim(96.0, 92.0)
    assert (canvas.ax.get_xlim(), canvas.ax.get_ylim()) != original

    canvas.reset_view()
    assert (canvas.ax.get_xlim(), canvas.ax.get_ylim()) == original
    assert _xlabel_is_inside_the_figure(canvas, canvas.ax)
    # The slices themselves survive a view reset -- it resets the view,
    # not the annotations.
    assert len(canvas._contour_slices) == 1


def test_canvas_reset_view_on_a_1d_still_uses_the_toolbar(qapp):
    canvas = MplCanvas()
    canvas.show_spectrum(_make_1d())
    canvas.reset_view()          # must not raise without a contour
    assert canvas._contour_spectrum is None


def _menu_labels(canvas, monkeypatch, choose=None):
    """Open the context menu, capture what it offers, and optionally
    pick one entry.

    Goes through MplCanvas._exec_context_menu rather than patching
    QMenu.exec: PySide6 resolves exec through the C++ binding, so a
    patch on the Qt class does not take and the real modal menu opens
    -- which hangs the headless suite instead of failing it."""
    captured = {}

    def fake_exec(self, menu, pos):
        captured["labels"] = [a.text() for a in menu.actions() if a.text()]
        if choose is None:
            return None
        return next((a for a in menu.actions() if a.text() == choose), None)

    monkeypatch.setattr(MplCanvas, "_exec_context_menu", fake_exec)
    canvas._on_context_menu(QPoint(5, 5))
    return captured.get("labels", [])


def test_canvas_offers_ppm_referencing_on_a_2d_only(qapp, monkeypatch):
    # User request: "I need a right click option on the 2D spectra to be
    # able to set the sr (by setting ppm values)." The existing Hz entry
    # is gated on a 1D view, so a contour had no referencing action at
    # all -- and an Hz offset is the wrong thing to ask for on one
    # anyway, since what a user knows is where a peak should be.
    label = "Set Reference from Cursor (ppm)…"

    canvas = MplCanvas()
    canvas.show_spectrum(_make_1d())
    assert label not in _menu_labels(canvas, monkeypatch)

    canvas = MplCanvas()
    canvas.show_spectrum(_make_2d(n_f1=8))
    canvas._last_right_click_xy = (5.0, 95.0)
    assert label in _menu_labels(canvas, monkeypatch)


def test_canvas_ppm_referencing_emits_the_clicked_position(qapp, monkeypatch):
    # The position travels with the signal rather than being read back
    # out of _last_right_click_xy, so the handler cannot act on a
    # different click than the one that opened the menu.
    canvas = MplCanvas()
    canvas.show_spectrum(_make_2d(n_f1=8))
    canvas._last_right_click_xy = (5.0, 95.0)

    seen = []
    canvas.referenceCalibrationRequested.connect(seen.append)
    _menu_labels(canvas, monkeypatch, choose="Set Reference from Cursor (ppm)…")
    assert seen == [(5.0, 95.0)]


def test_canvas_projection_zoom_survives_a_contour_level_change(qapp):
    # Real user report: "the 2D projections shouldn't reset when I scale
    # the contours". Scrolling the contour body adjusts the levels,
    # which rebuilds the figure -- and the projection strips are their
    # own Axes with their own intensity zoom, which the rebuild threw
    # away.
    canvas = MplCanvas()
    canvas.figure.set_size_inches(9, 6)
    canvas.show_spectrum(_make_2d(n_f1=16, n_f2=128))
    canvas.canvas.draw()

    canvas._on_scroll(_scroll_at(canvas, canvas._proj_axes["f2"]))
    canvas._on_scroll(_scroll_at(canvas, canvas._proj_axes["f1"]))
    zoomed = (canvas._proj_axes["f2"].get_ylim(), canvas._proj_axes["f1"].get_xlim())

    canvas._on_scroll(_scroll_at(canvas, canvas.ax))     # contour body -> levels -> rebuild
    assert (canvas._proj_axes["f2"].get_ylim(), canvas._proj_axes["f1"].get_xlim()) == zoomed


def test_canvas_crosshair_on_a_1d_spectrum(qapp):
    """1D gets a crosshair too -- one vertical and one horizontal line
    across the single Axes ("ideally I'd also like crosshairs on the 1D
    spectra").

    The same class as the 2D one, with no projection strips to span, so
    there is nothing separate to keep in step."""
    canvas = MplCanvas()
    canvas.show_spectrum(_make_1d())
    assert canvas._crosshair is None

    canvas._on_motion(_motion_at(canvas, canvas.ax))

    crosshair = canvas._crosshair
    assert crosshair is not None
    assert set(crosshair.axes) == {canvas.ax}
    assert crosshair._vline_axes == [canvas.ax]
    assert crosshair._hline_axes == [canvas.ax]


def test_canvas_crosshair_1d_tracks_and_hides(qapp):
    """It follows the cursor while over the plot and disappears when
    the cursor leaves -- the same contract the 2D one has."""
    canvas = MplCanvas()
    canvas.show_spectrum(_make_1d())
    event = _motion_at(canvas, canvas.ax)
    canvas._on_motion(event)
    crosshair = canvas._crosshair
    crosshair._onmove(event)

    assert crosshair._vlines[0].get_visible()
    assert crosshair._vlines[0].get_xdata() == (event.xdata, event.xdata)
    assert crosshair._hlines[0].get_ydata() == (event.ydata, event.ydata)

    crosshair._onmove(types.SimpleNamespace(inaxes=None, xdata=None, ydata=None, x=0, y=0))

    assert not crosshair._vlines[0].get_visible()
    assert not crosshair._hlines[0].get_visible()


def test_canvas_crosshair_absent_on_views_with_no_spectrum(qapp):
    """The FID view is not a spectrum view -- a crosshair there would
    be a measurement cue on axes that measure nothing the user asked
    about."""
    canvas = MplCanvas()
    canvas.show_fid(np.exp(-np.arange(256) / 50.0) + 0j, dt=1e-5)

    canvas._on_motion(_motion_at(canvas, canvas.ax))

    assert canvas._crosshair is None


def _press_at(canvas, ax, xfrac=0.5, yfrac=0.5, button=1):
    bbox = ax.get_window_extent()
    x, y = bbox.x0 + xfrac * bbox.width, bbox.y0 + yfrac * bbox.height
    event = types.SimpleNamespace(name="button_press_event", canvas=canvas.canvas,
                                  inaxes=ax, x=x, y=y, button=button)
    event.xdata, event.ydata = ax.transData.inverted().transform((x, y))
    return event


def test_switching_axis_unit_does_not_leave_a_broken_crosshair(qapp):
    # Real user report: "The crosshair still disappears when changing
    # between axes", with a ValueError: shape mismatch out of
    # Line2D.recache on every mouse move afterwards. set_axis_unit
    # rewrites the x data of every line on the Axes -- and the
    # crosshair's own axvline/axhline are lines on that Axes, so they
    # came out of it holding a full-length x array against their own
    # two-point y, which cannot broadcast.
    canvas = MplCanvas()
    canvas.figure.set_size_inches(8, 6)
    canvas.show_spectrum(_make_1d())
    canvas.canvas.draw()
    canvas._on_motion(_motion_at(canvas, canvas.ax))
    assert canvas._crosshair is not None

    canvas.set_axis_unit("hz")
    assert canvas._crosshair is None                       # removed, not corrupted
    assert len(canvas.ax.lines) == 1                       # and taken back out of the Axes

    # The next move rebuilds it against the new unit, and moving again
    # is what used to raise.
    canvas._on_motion(_motion_at(canvas, canvas.ax, xfrac=0.3, yfrac=0.3))
    assert canvas._crosshair is not None
    canvas._crosshair._onmove(_motion_at(canvas, canvas.ax, xfrac=0.6, yfrac=0.4))
    for line in canvas._crosshair._vlines:
        assert len(line.get_xdata()) == 2
    for line in canvas._crosshair._hlines:
        assert len(line.get_ydata()) == 2


def test_crosshair_can_be_switched_off_and_back_on(qapp):
    # The Crosshair toggle on the canvas's own toolbar -- reported
    # missing entirely.
    canvas = MplCanvas()
    canvas.figure.set_size_inches(8, 6)
    canvas.show_spectrum(_make_1d())
    canvas.canvas.draw()
    canvas._on_motion(_motion_at(canvas, canvas.ax))
    assert canvas._crosshair is not None

    canvas._crosshair_button.setChecked(False)
    assert canvas._crosshair_enabled is False
    assert canvas._crosshair is None
    canvas._on_motion(_motion_at(canvas, canvas.ax, xfrac=0.3))
    assert canvas._crosshair is None                       # stays off

    canvas._crosshair_button.setChecked(True)
    canvas._on_motion(_motion_at(canvas, canvas.ax, xfrac=0.4))
    assert canvas._crosshair is not None


def test_measurement_reports_ppm_and_hz_together_whatever_unit_is_displayed(qapp):
    # The core requirement: a separation is quoted in ppm *and* Hz at
    # once, and switching the displayed unit to read one must not
    # change the other. The raw x difference is in display units, so a
    # measurement taken in Hz and the same one taken in ppm have to
    # agree after conversion -- that is the whole risk in this feature.
    results = []

    def measure(unit):
        canvas = MplCanvas()
        canvas.figure.set_size_inches(8, 6)
        canvas.show_spectrum(_make_1d())
        canvas.canvas.draw()
        canvas.set_axis_unit(unit)
        canvas.set_measurement_mode(True)
        got = []
        canvas.measurementCompleted.connect(got.append)
        canvas._on_press(_press_at(canvas, canvas.ax, xfrac=0.3, yfrac=0.5))
        assert not got                                     # one click is not a measurement
        canvas._on_press(_press_at(canvas, canvas.ax, xfrac=0.7, yfrac=0.5))
        assert len(got) == 1
        return got[0]

    for unit in ("ppm", "hz"):
        results.append(measure(unit))

    in_ppm, in_hz = results
    assert in_ppm["display_unit"] == "ppm" and in_hz["display_unit"] == "hz"
    # sfo1 = 400 MHz in _make_1d, so the axis is 400 Hz per ppm.
    assert in_ppm["dx"]["hz"] == pytest.approx(in_ppm["dx"]["ppm"] * 400.0, rel=1e-6)
    assert in_hz["dx"]["ppm"] == pytest.approx(in_hz["dx"]["hz"] / 400.0, rel=1e-6)
    assert in_ppm["dx"]["ppm"] == pytest.approx(in_hz["dx"]["ppm"], rel=1e-6)
    # And every number reported carries its unit in the same string.
    assert "ppm" in in_hz["summary"] and "Hz" in in_hz["summary"]


def test_measurement_draws_itself_on_the_plot(qapp):
    # "Measurement isn't coming up" -- two clicks used to leave no mark
    # on the plot at all, and printed to stdout.
    canvas = MplCanvas()
    canvas.figure.set_size_inches(8, 6)
    canvas.show_spectrum(_make_1d())
    canvas.canvas.draw()
    canvas.set_measurement_mode(True)

    canvas._on_press(_press_at(canvas, canvas.ax, xfrac=0.3, yfrac=0.5))
    assert len(canvas._measurement_artists) == 1           # the pending first-point marker

    canvas._on_press(_press_at(canvas, canvas.ax, xfrac=0.7, yfrac=0.5))
    labels = [a for a in canvas._measurement_artists if hasattr(a, "get_text") and a.get_text()]
    assert labels and "ppm" in labels[0].get_text()
    assert not any(a.get_gid() == canvas._MEASUREMENT_PENDING_GID
                   for a in canvas._measurement_artists)

    # Several can stand at once -- a J-coupling analysis is rarely one.
    canvas._on_press(_press_at(canvas, canvas.ax, xfrac=0.4, yfrac=0.6))
    canvas._on_press(_press_at(canvas, canvas.ax, xfrac=0.5, yfrac=0.6))
    labelled = [a for a in canvas._measurement_artists
                if hasattr(a, "get_text") and a.get_text()]
    assert len(labelled) == 2

    canvas.set_measurement_mode(False)
    assert canvas._measurement_artists == []


def test_measurement_on_a_2d_reports_both_dimensions(qapp):
    canvas = MplCanvas()
    canvas.figure.set_size_inches(8, 6)
    canvas.show_spectrum(_make_2d(n_f1=8))
    canvas.canvas.draw()
    canvas.set_measurement_mode(True)
    got = []
    canvas.measurementCompleted.connect(got.append)
    canvas._on_press(_press_at(canvas, canvas.ax, xfrac=0.3, yfrac=0.3))
    canvas._on_press(_press_at(canvas, canvas.ax, xfrac=0.7, yfrac=0.7))

    result = got[0]
    # F2 is 400 Hz/ppm, F1 is 100 Hz/ppm -- each converted through its
    # own axis, not through one shared frequency.
    assert result["dx"]["hz"] == pytest.approx(result["dx"]["ppm"] * 400.0, rel=1e-6)
    assert result["dy"]["hz"] == pytest.approx(result["dy"]["ppm"] * 100.0, rel=1e-6)
    assert "Δx" in result["summary"] and "Δy" in result["summary"]


def test_measurement_restarts_when_the_second_click_is_in_another_axes(qapp):
    # A separation across two unrelated coordinate systems is not a
    # measurement of anything.
    canvas = MplCanvas()
    canvas.figure.set_size_inches(8, 6)
    canvas.show_spectrum(_make_2d(n_f1=8))
    canvas.canvas.draw()
    canvas.set_measurement_mode(True)
    got = []
    canvas.measurementCompleted.connect(got.append)

    canvas._on_press(_press_at(canvas, canvas.ax, xfrac=0.3, yfrac=0.3))
    canvas._on_press(_press_at(canvas, canvas._proj_axes["f2"], xfrac=0.7, yfrac=0.5))
    assert not got
    assert len(canvas._measurement_points) == 1
    assert canvas._measurement_points[0][0] is canvas._proj_axes["f2"]


def test_measurement_in_mm_when_the_axis_is_displayed_in_mm(qapp):
    canvas = MplCanvas()
    canvas.figure.set_size_inches(8, 6)
    canvas.show_spectrum(_make_1d())
    canvas.canvas.draw()
    canvas.set_imaging_gradient_params(gamma_hz_per_t=42.577e6, max_gradient_g_per_cm=50.0)
    canvas.set_axis_unit("mm")
    canvas.canvas.draw()
    canvas.set_measurement_mode(True)
    got = []
    canvas.measurementCompleted.connect(got.append)
    canvas._on_press(_press_at(canvas, canvas.ax, xfrac=0.3, yfrac=0.5))
    canvas._on_press(_press_at(canvas, canvas.ax, xfrac=0.7, yfrac=0.5))

    result = got[0]
    assert result["dx"]["mm"] == pytest.approx(result["dx"]["hz"] / canvas._imaging_hz_per_mm())
    assert result["summary"].startswith("Δx = ") and "mm" in result["summary"]


def test_measurement_omits_mm_while_the_axis_is_in_ppm(qapp):
    """Imaging gradient parameters being set is not enough on its own.

    They are a property of the acquisition, so once entered they stay
    set for the whole session -- which had every ordinary chemical-shift
    measurement quoting a distance beside the ppm value. Explicit user
    request: "don't give a mm value unless the axis is set to mm"."""
    canvas = MplCanvas()
    canvas.figure.set_size_inches(8, 6)
    canvas.show_spectrum(_make_1d())
    canvas.canvas.draw()
    canvas.set_imaging_gradient_params(gamma_hz_per_t=42.577e6, max_gradient_g_per_cm=50.0)
    assert canvas._axis_unit == "ppm"
    canvas.set_measurement_mode(True)
    got = []
    canvas.measurementCompleted.connect(got.append)
    canvas._on_press(_press_at(canvas, canvas.ax, xfrac=0.3, yfrac=0.5))
    canvas._on_press(_press_at(canvas, canvas.ax, xfrac=0.7, yfrac=0.5))

    result = got[0]
    assert "mm" not in result["dx"]
    assert "mm" not in result["summary"]
    # The pairing the measurement exists for is untouched.
    assert "ppm" in result["summary"] and "Hz" in result["summary"]


def test_measurement_omits_mm_in_hz_view_even_with_a_gradient(qapp):
    """Hz is not mm either -- the gate is the displayed unit, not
    "is this a frequency axis"."""
    canvas = MplCanvas()
    canvas.figure.set_size_inches(8, 6)
    canvas.show_spectrum(_make_1d())
    canvas.canvas.draw()
    canvas.set_imaging_gradient_params(gamma_hz_per_t=42.577e6, max_gradient_g_per_cm=50.0)
    canvas.set_axis_unit("hz")
    canvas.canvas.draw()
    canvas.set_measurement_mode(True)
    got = []
    canvas.measurementCompleted.connect(got.append)
    canvas._on_press(_press_at(canvas, canvas.ax, xfrac=0.3, yfrac=0.5))
    canvas._on_press(_press_at(canvas, canvas.ax, xfrac=0.7, yfrac=0.5))

    assert "mm" not in got[0]["dx"]


# ============================================================
# Compare stack: row labels, shared scale, unit switching
# ============================================================

def _stack_pair(qapp, big=1.0, small=0.1):
    """Two 1D spectra of deliberately different intensity, for checking
    whether a stack shows their relative heights or hides them."""
    ppm = np.linspace(0, 10, 200)
    metadata = NMRMetadata(vendor="bruker", nucleus="1H", sfo1=400.0)
    axis = NMRAxis(kind="frequency", ppm=ppm, hz=ppm * 400.0)
    return [
        NMRSpectrum(data=amp * np.exp(-((ppm - 5) ** 2)), axes=[axis], metadata=metadata)
        for amp in (big, small)
    ]


def test_compare_stack_autoscales_each_row_by_default(qapp):
    canvas = MplCanvas()
    canvas.show_compare(_stack_pair(qapp), mode="stack")
    ranges = [abs(ax.get_ylim()[1] - ax.get_ylim()[0]) for ax in canvas._compare_axes]
    # A tenth-height trace still fills its own row, so the two ranges
    # differ by about the same factor -- which is exactly what hides
    # the relative amplitudes.
    assert ranges[0] / ranges[1] > 5


def test_compare_stack_shared_scale_shows_relative_heights(qapp):
    canvas = MplCanvas()
    canvas.show_compare(_stack_pair(qapp), mode="stack", shared_scale=True)
    ranges = [abs(ax.get_ylim()[1] - ax.get_ylim()[0]) for ax in canvas._compare_axes]
    assert ranges[0] == pytest.approx(ranges[1])


def test_shared_scale_is_a_no_op_when_normalising(qapp):
    # Normalising has already put every trace at the same peak, so
    # there are no relative heights left to show.
    canvas = MplCanvas()
    canvas.show_compare(_stack_pair(qapp), mode="stack", normalize=True, shared_scale=True)
    ranges = [abs(ax.get_ylim()[1] - ax.get_ylim()[0]) for ax in canvas._compare_axes]
    assert ranges[0] == pytest.approx(ranges[1], rel=0.2)


def test_compare_stack_row_labels_are_drawn_on_their_own_rows(qapp):
    canvas = MplCanvas()
    canvas.show_compare(_stack_pair(qapp), mode="stack", row_labels=["first", "second"])
    for axes, expected in zip(canvas._compare_axes, ("first", "second")):
        assert any(t.get_text() == expected for t in axes.texts)


def test_compare_stack_without_labels_draws_none(qapp):
    canvas = MplCanvas()
    canvas.show_compare(_stack_pair(qapp), mode="stack")
    assert all(not axes.texts for axes in canvas._compare_axes)


def test_compare_stack_unit_switch_converts_every_row(qapp):
    # self.ax is only the last row -- converting that alone would leave
    # every other row silently in the old unit.
    canvas = MplCanvas()
    spectra = _stack_pair(qapp)
    canvas.show_compare(spectra, mode="stack")
    canvas.set_axis_unit("hz")

    expected = spectra[0].axes[0].values("hz")
    for axes in canvas._compare_axes:
        np.testing.assert_allclose(axes.lines[0].get_xdata(), expected)
    assert canvas.axis_unit == "hz"


def test_compare_stack_unit_switch_round_trips(qapp):
    canvas = MplCanvas()
    spectra = _stack_pair(qapp)
    canvas.show_compare(spectra, mode="stack")
    canvas.set_axis_unit("khz")
    canvas.set_axis_unit("ppm")
    np.testing.assert_allclose(canvas._compare_axes[0].lines[0].get_xdata(),
                               spectra[0].axes[0].values("ppm"))


def test_compare_stack_mm_needs_a_gradient(qapp):
    canvas = MplCanvas()
    spectra = _stack_pair(qapp)
    canvas.show_compare(spectra, mode="stack")
    canvas.set_axis_unit("mm")
    # No gradient set, so nothing converts and the unit does not change.
    assert canvas.axis_unit == "ppm"

    canvas.set_imaging_gradient_params(gamma_hz_per_t=42.577e6, max_gradient_g_per_cm=50.0)
    canvas.set_axis_unit("mm")
    assert canvas.axis_unit == "mm"
    expected = spectra[0].axes[0].values("hz") / canvas._imaging_hz_per_mm()
    np.testing.assert_allclose(canvas._compare_axes[0].lines[0].get_xdata(), expected)


def test_compare_spectra_do_not_outlive_their_figure(qapp):
    # A stale list would have set_axis_unit rewriting rows that no
    # longer exist.
    canvas = MplCanvas()
    canvas.show_compare(_stack_pair(qapp), mode="stack")
    assert canvas._compare_spectra
    canvas.show_spectrum(_make_1d())
    assert canvas._compare_spectra == []


# ============================================================
# x-axis scroll-zoom on a stacked multi-panel view
# ============================================================

def _margin_scroll(canvas, ax, below=6.0):
    """A scroll event in ``ax``'s own x-tick-label strip -- below its
    plot area, and in no axes' data rectangle (``inaxes is None``),
    which is what matplotlib reports for a margin."""
    bbox = ax.get_window_extent()
    return types.SimpleNamespace(
        name="scroll_event", canvas=canvas.canvas, inaxes=None,
        x=bbox.x0 + 0.5 * bbox.width, y=bbox.y0 - below, step=1, button="up",
    )


def _dual_panel_canvas(qapp):
    """A canvas showing two stacked panels with their own x axes --
    the shape of the Processing tab's FID + spectrum preview."""
    canvas = MplCanvas()
    canvas.figure.set_size_inches(9, 7)
    # Clear first, as every real view does: MplCanvas.__init__ leaves a
    # full-figure Axes behind, and building panels on top of it gives
    # the figure three Axes rather than two.
    canvas._clear_figure()
    top, bottom = canvas.figure.subplots(2, 1)
    top.plot(np.linspace(0, 1, 50), np.random.default_rng(0).normal(size=50))
    bottom.plot(np.linspace(0, 10, 50), np.random.default_rng(1).normal(size=50))
    canvas.ax = bottom          # what every multi-panel view sets
    canvas.canvas.draw()
    return canvas, top, bottom


def test_each_panel_scroll_zooms_its_own_x_axis(qapp):
    # The gap this closes: the strip check only ever consulted
    # self.ax -- the *last* Axes a multi-panel view creates -- so a
    # panel above it could never have its own x-tick strip hovered,
    # and its x axis could not be scroll-zoomed at all.
    canvas, top, bottom = _dual_panel_canvas(qapp)

    top_before, bottom_before = top.get_xlim(), bottom.get_xlim()
    canvas._on_scroll(_margin_scroll(canvas, top))
    assert top.get_xlim() != top_before
    assert bottom.get_xlim() == bottom_before      # the other panel untouched


def test_the_bottom_panel_still_zooms_its_own_x_axis(qapp):
    # A point below the bottom panel is also below every panel above
    # it, so taking the *first* strip match zoomed the top panel
    # whenever the bottom one's own strip was aimed at. The nearest
    # panel above the cursor wins.
    canvas, top, bottom = _dual_panel_canvas(qapp)

    top_before, bottom_before = top.get_xlim(), bottom.get_xlim()
    canvas._on_scroll(_margin_scroll(canvas, bottom))
    assert bottom.get_xlim() != bottom_before
    assert top.get_xlim() == top_before


def test_scrolling_a_panel_body_still_zooms_intensity_not_x(qapp):
    # The strip search must not steal the gesture from the intensity
    # zoom -- it only runs when the cursor is in a margin.
    canvas, top, _bottom = _dual_panel_canvas(qapp)
    bbox = top.get_window_extent()
    event = types.SimpleNamespace(
        name="scroll_event", canvas=canvas.canvas, inaxes=top,
        x=bbox.x0 + 0.5 * bbox.width, y=bbox.y0 + 0.5 * bbox.height, step=1, button="up",
    )
    event.xdata, event.ydata = top.transData.inverted().transform((event.x, event.y))

    x_before, y_before = top.get_xlim(), top.get_ylim()
    canvas._on_scroll(event)
    assert top.get_xlim() == x_before
    assert top.get_ylim() != y_before


def test_a_single_panel_view_is_unaffected(qapp):
    canvas = MplCanvas()
    canvas.figure.set_size_inches(8, 6)
    canvas.show_spectrum(_make_1d())
    canvas.canvas.draw()
    before = canvas.ax.get_xlim()
    canvas._on_scroll(_margin_scroll(canvas, canvas.ax))
    assert canvas.ax.get_xlim() != before


# ============================================================
# Crosshairs on every plot worth reading a value off
# ============================================================
# "It would be good to have crosshairs available on all plots including
# compare, spectral fit, and fit." The gate used to require a contour
# or a 1D spectrum, which kept it off exactly those three.

def _crosshair_after_move(canvas):
    canvas.figure.set_size_inches(8, 6)
    canvas._on_motion(_motion_at(canvas, canvas.ax))
    return canvas._crosshair


def test_canvas_crosshair_on_the_compare_view(qapp):
    canvas = MplCanvas()
    canvas.show_compare([_make_1d(), _make_1d()], mode="stack")
    assert _crosshair_after_move(canvas) is not None


def test_canvas_crosshair_on_the_spectral_fit_view(qapp):
    canvas = MplCanvas()
    x = np.linspace(0.0, 10.0, 200)
    y = np.exp(-0.5 * ((x - 5.0) / 0.2) ** 2)
    canvas.show_slice_fit(x, y, y)
    assert _crosshair_after_move(canvas) is not None


def test_canvas_crosshair_on_the_curve_fit_view(qapp):
    canvas = MplCanvas()
    t = np.geomspace(0.01, 1.0, 8)
    y = np.exp(-t / 0.1)
    canvas.show_relaxation_fit(
        {"t": t, "y": y, "y_fit": y, "peak_label": "Peak 1", "model": "exponential_decay"})
    assert _crosshair_after_move(canvas) is not None


def test_canvas_crosshair_stays_off_the_fid_view(qapp):
    # Not every plot: the FID and the processing preview are deliberate
    # exclusions, and a view that never opts in must not inherit the
    # flag from whatever was on this canvas before.
    canvas = MplCanvas()
    canvas.show_spectrum(_make_1d())
    assert _crosshair_after_move(canvas) is not None

    canvas.show_fid(np.exp(-np.arange(64) * 0.1), dt=1e-4)
    assert canvas._crosshair is None
    assert _crosshair_after_move(canvas) is None


def test_canvas_crosshair_toggle_still_governs_every_view(qapp):
    canvas = MplCanvas()
    canvas.set_crosshair_enabled(False)
    canvas.show_compare([_make_1d(), _make_1d()], mode="stack")
    assert _crosshair_after_move(canvas) is None

    canvas.set_crosshair_enabled(True)
    assert _crosshair_after_move(canvas) is not None


# ============================================================
# Extracted slices: colour, and clearance from the y-axis label
# ============================================================

def test_slice_colours_are_stored_not_derived_at_draw_time(qapp):
    # Reported directly: "when I changed the units, the colours reset".
    # The colour used to be assigned by position inside the draw loop,
    # so every rebuild re-derived it.
    canvas = MplCanvas()
    canvas.show_spectrum(_make_2d(n_f1=16, n_f2=128))
    canvas._add_contour_slice("f1", 3, "a")
    canvas._add_contour_slice("f1", 9, "b")

    stored = [dict(s) for s in canvas._contour_slices]
    assert all(s["color"] for s in stored)
    assert stored[0]["color"] != stored[1]["color"]

    canvas.set_axis_unit("hz")

    assert [s["color"] for s in canvas._contour_slices] == [s["color"] for s in stored]


def test_slice_colours_come_from_the_compare_colour_map(qapp):
    # Explicit user request: they "should be able to either take a
    # colour from the standard compare map, or be independently
    # clickable".
    import matplotlib.colors
    from opennmr import plotting

    canvas = MplCanvas()
    canvas.set_style(compare_colormap="plasma")
    canvas.show_spectrum(_make_2d(n_f1=16, n_f2=128))
    canvas._add_contour_slice("f1", 3, "a")

    expected = matplotlib.colors.to_hex(
        plotting.stack_colors(canvas._SLICE_PALETTE_SIZE, "plasma")[0])
    assert canvas._contour_slices[0]["color"] == expected


def test_a_slice_colour_can_be_set_on_its_own(qapp):
    canvas = MplCanvas()
    canvas.show_spectrum(_make_2d(n_f1=16, n_f2=128))
    canvas._add_contour_slice("f1", 3, "a")
    canvas._add_contour_slice("f1", 9, "b")
    other = canvas._contour_slices[1]["color"]

    canvas.set_slice_colour(0, "#123456")

    assert canvas._contour_slices[0]["color"] == "#123456"
    assert canvas._contour_slices[1]["color"] == other     # its neighbour is untouched


def test_adding_a_slice_does_not_recolour_the_existing_ones(qapp):
    canvas = MplCanvas()
    canvas.show_spectrum(_make_2d(n_f1=16, n_f2=128))
    canvas._add_contour_slice("f1", 3, "a")
    first = canvas._contour_slices[0]["color"]
    canvas._add_contour_slice("f1", 9, "b")
    assert canvas._contour_slices[0]["color"] == first


def test_slice_panel_clears_the_contour_y_axis_labels(qapp):
    # Reported directly: "I still see the extracted slices overlapping
    # with the y-axis label of the contour, probably because the data
    # I'm looking at has units in the 1000s because if I change to kHz
    # (100s scale) it is less cut off".
    #
    # The contour's y ticks and label are on the right and the panel is
    # appended to that same side, so a fixed pad is eaten by whatever
    # the tick labels take -- which grows with the digit count.
    canvas = MplCanvas()
    canvas.figure.set_size_inches(9, 6)
    canvas.show_spectrum(_make_2d(n_f1=16, n_f2=128))
    canvas._add_contour_slice("f1", 3, "a")
    canvas.canvas.draw()
    renderer = canvas.figure.canvas.get_renderer()

    panel = canvas._proj_axes.get("slices")
    assert panel is not None
    right_edge = max(
        [canvas.ax.yaxis.label.get_window_extent(renderer).x1]
        + [t.get_window_extent(renderer).x1
           for t in canvas.ax.get_yticklabels() if t.get_text()])
    assert panel.get_window_extent(renderer).x0 > right_edge


# ============================================================
# One series interface across every multi-series plot
# ============================================================

def _two_1d():
    return [_make_1d(), _make_1d()]


def test_series_is_empty_where_there_is_no_notion_of_several(qapp):
    """So a caller can offer per-series controls unconditionally."""
    canvas = MplCanvas()
    canvas.show_spectrum(_make_1d())
    assert canvas.series() == []


def test_every_multi_series_view_answers_the_same_interface(qapp):
    # The point of the interface: one dialog section drives all of
    # them, instead of one bespoke control per plot type.
    canvas = MplCanvas()

    canvas.show_compare(_two_1d(), mode="overlay", row_labels=["a", "b"])
    assert [e.key for e in canvas.series()] == ["0", "1"]

    canvas.show_spectrum(_make_2d(n_f1=16, n_f2=128))
    canvas._add_contour_slice("f1", 3, "slice a")
    assert [e.label for e in canvas.series()] == ["slice a"]

    canvas.show_2d_overlay([_make_2d(n_f1=8), _make_2d(n_f1=8)], labels=["x", "y"])
    assert [e.label for e in canvas.series()] == ["x", "y"]


def test_every_series_gets_its_own_colour_from_the_shared_palette(qapp):
    canvas = MplCanvas()
    canvas.set_style(compare_colormap="plasma")
    canvas.show_compare(_two_1d(), mode="overlay", row_labels=["a", "b"])

    colours = [e.colour for e in canvas.series()]
    assert len(set(colours)) == 2
    expected = matplotlib.colors.to_hex(plotting.stack_colors(2, "plasma")[0])
    assert colours[0] == expected


def test_a_series_colour_survives_a_redraw(qapp):
    # The rule the interface exists to enforce: colours are stored when
    # a series is created and only read at draw time.
    canvas = MplCanvas()
    canvas.show_compare(_two_1d(), mode="overlay", row_labels=["a", "b"])
    canvas.set_series_style("0", colour="#ff00ff")
    assert canvas.series()[0].colour == "#ff00ff"

    canvas.show_compare(_two_1d(), mode="overlay", row_labels=["a", "b"])
    assert canvas.series()[0].colour == "#ff00ff"


def test_renaming_a_series_reaches_the_plot(qapp):
    canvas = MplCanvas()
    canvas.show_2d_overlay([_make_2d(n_f1=8), _make_2d(n_f1=8)], labels=["x", "y"])

    canvas.set_series_style("1", label="my sample")

    assert [e.label for e in canvas.series()] == ["x", "my sample"]
    assert "my sample" in [t.get_text() for t in canvas.ax.get_legend().get_texts()]


def test_a_compare_trace_is_named_by_its_spectrum_not_the_plot(qapp):
    # Its name comes from the Data tab's Rename, which reaches every
    # other view too -- so the plot does not offer a competing one.
    canvas = MplCanvas()
    canvas.show_compare(_two_1d(), mode="overlay", row_labels=["a", "b"])
    assert all(e.can_rename is False for e in canvas.series())

    canvas.set_series_style("0", label="ignored")
    assert canvas.series()[0].label == "a"


def test_setting_a_slice_colour_through_the_interface_reaches_its_spec(qapp):
    # Slices keep their own copy on the spec, which is what persists
    # with the spectrum -- the two must not drift apart.
    canvas = MplCanvas()
    canvas.show_spectrum(_make_2d(n_f1=16, n_f2=128))
    canvas._add_contour_slice("f1", 3, "a")

    canvas.set_series_style("0", colour="#010203", label="renamed")

    assert canvas._contour_slices[0]["color"] == "#010203"
    assert canvas._contour_slices[0]["label"] == "renamed"
    assert canvas.series()[0].colour == "#010203"


def test_changing_the_palette_recolours_what_is_already_drawn(qapp):
    # Reported directly: "when I change the colour map from a plot, it
    # should update the existing colours from that new series rather
    # than me having to redraw the plot in some way". The reuse that
    # stops colours resetting on a redraw was also stopping this.
    canvas = MplCanvas()
    canvas.set_style(compare_colormap="viridis")
    canvas.show_compare(_two_1d(), mode="overlay", row_labels=["a", "b"])
    before = [e.colour for e in canvas.series()]

    canvas.set_style(compare_colormap="plasma")
    canvas.restyle_series()

    after = [e.colour for e in canvas.series()]
    assert after != before
    assert after[0] == matplotlib.colors.to_hex(plotting.stack_colors(2, "plasma")[0])


def test_a_deliberate_colour_survives_a_palette_change(qapp):
    canvas = MplCanvas()
    canvas.show_compare(_two_1d(), mode="overlay", row_labels=["a", "b"])
    canvas.set_series_style("0", colour="#ff00ff")
    untouched_before = canvas.series()[1].colour

    canvas.set_style(compare_colormap="plasma")
    canvas.restyle_series()

    assert canvas.series()[0].colour == "#ff00ff"      # set on purpose, left alone
    assert canvas.series()[1].colour != untouched_before  # derived, follows the map


def test_resetting_colours_puts_everything_back_on_the_palette(qapp):
    canvas = MplCanvas()
    canvas.show_compare(_two_1d(), mode="overlay", row_labels=["a", "b"])
    canvas.set_series_style("0", colour="#ff00ff")

    canvas.reset_series_colours()

    entries = canvas.series()
    assert entries[0].colour != "#ff00ff"
    assert all(not e.colour_overridden for e in entries)
    assert entries[0].colour == matplotlib.colors.to_hex(
        plotting.stack_colors(2, canvas._series_colormap)[0])


def test_a_compare_trace_takes_a_renamed_spectrum_name(qapp):
    # can_rename=False must not mean "frozen at the first name" -- a
    # rename made in the Data tab has to reach the plot.
    canvas = MplCanvas()
    canvas.show_compare(_two_1d(), mode="overlay", row_labels=["a", "b"])
    assert canvas.series()[0].label == "a"

    canvas.show_compare(_two_1d(), mode="overlay", row_labels=["renamed", "b"])
    assert canvas.series()[0].label == "renamed"


def test_score_components_are_renameable_unlike_compare_traces(qapp):
    # Resolved components are not session spectra; their labels are
    # generated annotations and theirs to rename.
    canvas = MplCanvas()
    canvas.show_compare(_two_1d(), mode="stack", row_labels=["c1", "c2"],
                        series_renameable=True)
    assert all(e.can_rename for e in canvas.series())

    canvas.set_series_style("0", label="free water")
    assert canvas.series()[0].label == "free water"
    # And it survives the redraw, unlike a Compare trace's name.
    canvas.show_compare(_two_1d(), mode="stack", row_labels=["c1", "c2"],
                        series_renameable=True)
    assert canvas.series()[0].label == "free water"


def test_a_slice_colour_override_is_marked_on_its_spec(qapp):
    # The spec persists with the spectrum, so a palette change in a
    # later session has to be able to tell deliberate from derived.
    canvas = MplCanvas()
    canvas.show_spectrum(_make_2d(n_f1=16, n_f2=128))
    canvas._add_contour_slice("f1", 3, "a")
    assert not canvas._contour_slices[0].get("color_override")

    canvas.set_series_style("0", colour="#010203")
    assert canvas._contour_slices[0]["color_override"] is True
    assert canvas._contour_slices[0]["color"] == "#010203"

    canvas.set_style(compare_colormap="plasma")
    canvas.restyle_series()
    assert canvas._contour_slices[0]["color"] == "#010203"


def test_contour_line_width_is_editable_on_a_2d(qapp):
    # Explicit user request: "DOSY (all 2D?) line width should be
    # editable on contour". Deliberately separate from _AXIS_LINEWIDTH,
    # which styles axis furniture -- at its default the rings merge.
    canvas = MplCanvas()
    canvas.show_spectrum(_make_2d(n_f1=16, n_f2=128))
    assert canvas.style_settings()["contour_linewidth"] == pytest.approx(0.8)

    canvas.set_style(contour_linewidth=2.5)

    assert canvas.style_settings()["contour_linewidth"] == pytest.approx(2.5)
    widths = [c.get_linewidth() for c in canvas.ax.collections]
    assert widths and all(np.allclose(w, 2.5) for w in widths)


def test_contour_line_width_is_independent_of_the_axis_line_width(qapp):
    canvas = MplCanvas()
    canvas.show_spectrum(_make_2d(n_f1=16, n_f2=128))
    canvas.set_style(contour_linewidth=2.5, linewidth=4.0)

    assert canvas._contour_linewidth == pytest.approx(2.5)
    assert canvas._AXIS_LINEWIDTH == pytest.approx(4.0)
    assert all(np.allclose(c.get_linewidth(), 2.5) for c in canvas.ax.collections)


# ============================================================
# Context cursors: a double-headed arrow on the scrollable strips
# ============================================================
#
# The rule being tested throughout: the crosshair stays the cursor over
# anything you read values off -- spectra, projections, contours, DOSY
# maps, all of which are scrollable -- and only the tick-label *strips*,
# where a scroll zooms the axis itself, swap it for a double-headed
# arrow. The scroll-event helpers above are reused verbatim: they carry
# exactly the x/y/inaxes a motion event does, and _scroll_strip_at
# deliberately never looks at `button`.

def _shape(canvas):
    return canvas.canvas.cursor().shape()


def _hover(canvas, event):
    canvas._on_motion(event)
    return _shape(canvas)


def test_x_strip_shows_a_horizontal_double_arrow(qapp):
    canvas = MplCanvas()
    canvas.figure.set_size_inches(8, 6)
    canvas.show_spectrum(_make_1d())
    canvas.canvas.draw()
    assert _hover(canvas, _x_strip_scroll_event(canvas.ax, xdata=3.0)) == Qt.CursorShape.SizeHorCursor


def test_contour_y_strip_shows_a_vertical_double_arrow(qapp):
    canvas = MplCanvas()
    canvas.figure.set_size_inches(8, 6)
    canvas.show_spectrum(_make_2d())
    canvas.canvas.draw()
    assert _hover(canvas, _y_strip_scroll_event(canvas.ax, ydata=95.0)) == Qt.CursorShape.SizeVerCursor


def test_a_1d_right_margin_gets_no_arrow(qapp):
    """A plain 1D shows no y ticks and has no y-strip zoom -- so there
    must be no arrow there promising one. The gate is the same
    `has_levels` the scroll itself uses."""
    canvas = MplCanvas()
    canvas.figure.set_size_inches(8, 6)
    canvas.show_spectrum(_make_1d())
    canvas.canvas.draw()
    assert _hover(canvas, _y_strip_scroll_event(canvas.ax, ydata=0.5)) == Qt.CursorShape.ArrowCursor


def test_contour_body_keeps_the_crosshair_not_an_arrow(qapp):
    """Scrolling a contour body adjusts the threshold, and the
    crosshair is what belongs over it: "the projections, spectra, and
    contours (inc DOSY contours) ... should stay as a (toggleable)
    crosshair"."""
    canvas = MplCanvas()
    canvas.figure.set_size_inches(8, 6)
    canvas.show_spectrum(_make_2d())
    canvas.canvas.draw()
    assert _hover(canvas, _body_scroll_event(canvas.ax, xdata=5.0, ydata=95.0)) == Qt.CursorShape.ArrowCursor
    assert canvas._crosshair is not None      # the crosshair is the cursor here


def test_projection_body_keeps_the_crosshair_not_an_arrow(qapp):
    canvas = MplCanvas()
    canvas.figure.set_size_inches(8, 6)
    canvas.show_spectrum(_make_2d())
    canvas.canvas.draw()
    f2 = canvas._proj_axes["f2"]
    bbox = f2.get_window_extent()
    event = SimpleNamespace(
        inaxes=f2, x=bbox.x0 + 0.5 * bbox.width, y=bbox.y0 + 0.5 * bbox.height,
        button="up",
    )
    event.xdata, event.ydata = f2.transData.inverted().transform((event.x, event.y))
    assert _hover(canvas, event) == Qt.CursorShape.ArrowCursor


def test_1d_body_keeps_the_crosshair_though_scrolling_zooms_intensity(qapp):
    canvas = MplCanvas()
    canvas.figure.set_size_inches(8, 6)
    canvas.show_spectrum(_make_1d())
    canvas.canvas.draw()
    assert _hover(canvas, _body_scroll_event(canvas.ax, xdata=5.0, ydata=0.5)) == Qt.CursorShape.ArrowCursor


def test_each_panel_strip_gets_its_own_arrow(qapp):
    """The arrow follows the same nearest-panel-above resolution the
    zoom does, so it appears over a panel that is not self.ax."""
    canvas, top, _bottom = _dual_panel_canvas(qapp)
    assert _hover(canvas, _margin_scroll(canvas, top)) == Qt.CursorShape.SizeHorCursor


def test_strip_arrows_survive_the_crosshair_being_switched_off(qapp):
    """They answer different questions: the crosshair is a reading aid
    on the data, the arrow is an affordance on the axis. Turning one
    off must not remove the other."""
    canvas = MplCanvas()
    canvas.figure.set_size_inches(8, 6)
    canvas.show_spectrum(_make_1d())
    canvas.canvas.draw()
    canvas.set_crosshair_enabled(False)
    assert _hover(canvas, _x_strip_scroll_event(canvas.ax, xdata=3.0)) == Qt.CursorShape.SizeHorCursor
    assert canvas._crosshair is None


def test_measurement_mode_wins_over_a_strip_arrow(qapp):
    """A click mode arms a click, and the cross is what says so -- it
    outranks an affordance for a gesture you are not making."""
    canvas = MplCanvas()
    canvas.figure.set_size_inches(8, 6)
    canvas.show_spectrum(_make_1d())
    canvas.canvas.draw()
    canvas.set_measurement_mode(True)
    assert _hover(canvas, _x_strip_scroll_event(canvas.ax, xdata=3.0)) == Qt.CursorShape.CrossCursor
    canvas.set_measurement_mode(False)
    assert _hover(canvas, _x_strip_scroll_event(canvas.ax, xdata=3.0)) == Qt.CursorShape.SizeHorCursor


def test_pick_mode_still_sets_the_cross_through_the_resolver(qapp):
    """The three click-mode setters no longer call setCursor
    themselves; they route through _refresh_cursor so the four writers
    cannot disagree. Their visible behaviour must be unchanged."""
    canvas = MplCanvas()
    canvas.show_spectrum(_make_1d())
    canvas.set_pick_mode(True)
    assert _shape(canvas) == Qt.CursorShape.CrossCursor
    canvas.set_pick_mode(False)
    assert _shape(canvas) == Qt.CursorShape.ArrowCursor


def test_region_select_mode_clears_its_drag_before_refreshing(qapp):
    """_refresh_cursor leaves the shape alone mid-drag, so switching the
    mode off has to drop the half-finished drag *first* or the cross
    would stay behind."""
    canvas = MplCanvas()
    canvas.show_spectrum(_make_1d())
    canvas.set_region_select_mode(True)
    canvas._region_select_start = 5.0
    canvas.set_region_select_mode(False)
    assert _shape(canvas) == Qt.CursorShape.ArrowCursor


def test_toolbar_pan_mode_keeps_matplotlibs_own_cursor(qapp):
    """matplotlib sets MOVE/SELECT_REGION there and caches what it set.
    Ours stands down, and forgets its own shape so the context cursor
    is re-applied once the mode ends."""
    canvas = MplCanvas()
    canvas.figure.set_size_inches(8, 6)
    canvas.show_spectrum(_make_1d())
    canvas.canvas.draw()
    _hover(canvas, _x_strip_scroll_event(canvas.ax, xdata=3.0))
    canvas.toolbar.mode = "pan/zoom"
    canvas._refresh_cursor(_body_scroll_event(canvas.ax, xdata=5.0, ydata=0.5))
    assert canvas._last_cursor_shape is None

    canvas.toolbar.mode = ""
    assert _hover(canvas, _x_strip_scroll_event(canvas.ax, xdata=3.0)) == Qt.CursorShape.SizeHorCursor


def test_the_cursor_is_left_alone_mid_drag(qapp):
    canvas = MplCanvas()
    canvas.figure.set_size_inches(8, 6)
    canvas.show_spectrum(_make_1d())
    canvas.canvas.draw()
    _hover(canvas, _x_strip_scroll_event(canvas.ax, xdata=3.0))
    canvas._pan_start = (0, 0, canvas.ax.get_xlim(), canvas.ax.get_ylim(), canvas.ax)
    canvas._refresh_cursor(_body_scroll_event(canvas.ax, xdata=5.0, ydata=0.5))
    assert _shape(canvas) == Qt.CursorShape.SizeHorCursor


def test_clear_figure_and_leaving_the_canvas_forget_the_shape(qapp):
    """The cached shape was resolved against bboxes of Axes a rebuild
    destroys, and against a position the cursor has left."""
    canvas = MplCanvas()
    canvas.figure.set_size_inches(8, 6)
    canvas.show_spectrum(_make_1d())
    canvas.canvas.draw()
    _hover(canvas, _x_strip_scroll_event(canvas.ax, xdata=3.0))
    assert canvas._last_cursor_shape == Qt.CursorShape.SizeHorCursor
    canvas._clear_figure()
    assert canvas._last_cursor_shape is None

    canvas.ax = canvas.figure.add_subplot(111)
    canvas.show_spectrum(_make_1d())
    canvas.canvas.draw()
    _hover(canvas, _x_strip_scroll_event(canvas.ax, xdata=3.0))
    canvas._on_figure_leave(None)
    assert canvas._last_cursor_shape is None


def test_the_strip_classifier_agrees_with_what_scrolling_does(qapp):
    """The point of extracting _scroll_strip_at: one hit-test behind
    both, so the cursor cannot promise a zoom that does not happen (or
    stay silent about one that does)."""
    canvas = MplCanvas()
    canvas.figure.set_size_inches(8, 6)
    canvas.show_spectrum(_make_2d())
    canvas.canvas.draw()
    ax = canvas.ax

    x_event = _x_strip_scroll_event(ax, xdata=5.0, button="up")
    assert canvas._scroll_strip_at(x_event) == ("x", ax)
    before_x, before_y = ax.get_xlim(), ax.get_ylim()
    canvas._on_scroll(x_event)
    assert ax.get_xlim() != before_x and ax.get_ylim() == before_y

    y_event = _y_strip_scroll_event(ax, ydata=95.0, button="up")
    assert canvas._scroll_strip_at(y_event) == ("y", ax)
    before_x, before_y = ax.get_xlim(), ax.get_ylim()
    canvas._on_scroll(y_event)
    assert ax.get_ylim() != before_y and ax.get_xlim() == before_x

    # A body is not a strip -- the scroll there is a threshold change,
    # which is exactly why it gets no arrow.
    body = _body_scroll_event(ax, xdata=5.0, ydata=95.0, button="up")
    assert canvas._scroll_strip_at(body) is None
    before_factor = canvas._contour_factor
    canvas._on_scroll(body)
    assert canvas._contour_factor != before_factor


def test_the_classifier_ignores_the_slice_and_projection_panels(qapp):
    """Both are early returns in _on_scroll proper; the classifier
    repeats those guards so it stays answerable in isolation from a
    mouse move."""
    canvas = MplCanvas()
    canvas.figure.set_size_inches(9, 7)
    canvas.show_spectrum(_make_2d())
    canvas.canvas.draw()
    f1 = canvas._proj_axes["f1"]
    bbox = f1.get_window_extent()
    event = SimpleNamespace(
        inaxes=f1, x=bbox.x0 + 0.5 * bbox.width, y=bbox.y0 + 0.5 * bbox.height, button="up",
    )
    event.xdata, event.ydata = f1.transData.inverted().transform((event.x, event.y))
    assert canvas._scroll_strip_at(event) is None


def _processing_preview_canvas(qapp):
    """The Processing tab's own dual view: FID on top, a preview
    spectrum below, each with its own x axis."""
    canvas = MplCanvas()
    canvas.figure.set_size_inches(9, 7)
    spectrum = _make_1d()
    fid = np.exp(-np.linspace(0, 5, 128)) * np.exp(2j * np.pi * 10 * np.linspace(0, 1, 128))
    canvas.show_processing_preview(fid, dt=1e-4, spectrum=spectrum)
    canvas.canvas.draw()
    return canvas


def test_processing_preview_gives_both_panels_their_own_strip_arrow(qapp):
    """Explicit user request: the scroll cursors have to be on the
    Processing window's axes too, on the FID panel as well as the
    spectrum one. The FID panel is not self.ax (the spectrum panel
    below it is), so this is the nearest-panel-above resolution doing
    its job for the cursor as well as for the zoom."""
    canvas = _processing_preview_canvas(qapp)
    fid_ax = canvas._processing_preview_fid_ax
    spec_ax = canvas._processing_preview_spec_ax
    assert fid_ax is not None and spec_ax is not None and fid_ax is not spec_ax

    assert canvas._scroll_strip_at(_margin_scroll(canvas, fid_ax)) == ("x", fid_ax)
    assert _hover(canvas, _margin_scroll(canvas, fid_ax)) == Qt.CursorShape.SizeHorCursor

    canvas._on_figure_leave(None)
    assert canvas._scroll_strip_at(_margin_scroll(canvas, spec_ax)) == ("x", spec_ax)
    assert _hover(canvas, _margin_scroll(canvas, spec_ax)) == Qt.CursorShape.SizeHorCursor


def test_processing_preview_panel_bodies_keep_the_plain_cursor(qapp):
    """Their bodies scroll-zoom intensity (the FID freeform, the
    spectrum pin-bottom), which is a body gesture, not an axis one."""
    canvas = _processing_preview_canvas(qapp)
    for ax in (canvas._processing_preview_fid_ax, canvas._processing_preview_spec_ax):
        bbox = ax.get_window_extent()
        event = SimpleNamespace(
            inaxes=ax, x=bbox.x0 + 0.5 * bbox.width, y=bbox.y0 + 0.5 * bbox.height,
            button="up",
        )
        event.xdata, event.ydata = ax.transData.inverted().transform((event.x, event.y))
        canvas._on_figure_leave(None)
        assert _hover(canvas, event) == Qt.CursorShape.ArrowCursor


# ============================================================
# Plot style: title, fonts, contour levels, legend appearance
# ============================================================

def _fitted_canvas(qapp):
    """A view whose legend is built by opennmr.plotting.fit, not by the
    canvas -- the case the legend sweep exists for."""
    canvas = MplCanvas()
    canvas.figure.set_size_inches(8, 6)
    x = np.linspace(0, 10, 100)
    y = np.exp(-((x - 5) ** 2))
    canvas.show_slice_fit(x, y, y * 0.98, components=[y * 0.5, y * 0.48],
                          peak_labels=["a", "b"], title="Fit")
    canvas.canvas.draw()
    return canvas


def test_plot_title_is_hidden_by_default_and_toggles_back_on(qapp):
    """Off by default on request. Hidden rather than blanked, so the
    views that set a title keep setting it and turning the option on
    needs no redraw."""
    canvas = _fitted_canvas(qapp)
    assert canvas.ax.get_title() == "Fit"       # still set...
    assert canvas.ax.title.get_visible() is False   # ...just not shown

    canvas.set_style(title_visible=True)
    assert canvas.ax.title.get_visible() is True
    canvas.set_style(title_visible=False)
    assert canvas.ax.title.get_visible() is False


def test_a_title_set_by_a_later_rebuild_is_still_hidden(qapp):
    """The setting has to survive a redraw, or it reads as a toggle
    that keeps coming undone."""
    canvas = _fitted_canvas(qapp)
    canvas.show_spectrum(_make_2d())
    canvas.canvas.draw()
    canvas.ax.set_title("Slice 3")
    canvas._apply_current_axis_style()
    assert canvas.ax.title.get_visible() is False


def test_legend_defaults_to_frameless_at_the_axis_font_size(qapp):
    """Explicit user request. Checked on a legend the *plotting layer*
    builds, since that is the one a per-call-site kwarg would have
    missed."""
    canvas = _fitted_canvas(qapp)
    legend = canvas.ax.get_legend()
    assert legend is not None
    assert legend.get_frame_on() is False
    assert legend.get_texts()[0].get_fontsize() == canvas._AXIS_LABELSIZE


def test_legend_font_size_follows_the_axis_font_size_then_can_be_pinned(qapp):
    canvas = _fitted_canvas(qapp)
    canvas.set_style(labelsize=24)
    assert canvas.ax.get_legend().get_texts()[0].get_fontsize() == 24

    canvas.set_style(legend_fontsize=9)
    assert canvas.ax.get_legend().get_texts()[0].get_fontsize() == 9
    canvas.set_style(labelsize=14)
    assert canvas.ax.get_legend().get_texts()[0].get_fontsize() == 9   # pinned, not following


def test_legend_frame_alpha_and_colour_are_options_on_top_of_the_frame(qapp):
    canvas = _fitted_canvas(qapp)
    canvas.set_style(legend_frameon=True, legend_framealpha=0.5, legend_facecolor="#ff0000")
    frame = canvas.ax.get_legend().get_frame()
    assert canvas.ax.get_legend().get_frame_on() is True
    assert frame.get_alpha() == pytest.approx(0.5)
    assert matplotlib.colors.to_hex(frame.get_facecolor()) == "#ff0000"


def test_legend_frame_colour_is_not_passed_while_the_frame_is_off(qapp):
    """matplotlib treats any framealpha/facecolor as an implicit "there
    is a frame", which would quietly undo the setting just turned off."""
    canvas = _fitted_canvas(qapp)
    canvas.set_style(legend_frameon=True, legend_facecolor="#ff0000")
    canvas.set_style(legend_frameon=False)
    assert canvas.ax.get_legend().get_frame_on() is False
    assert "facecolor" not in canvas._legend_kwargs()
    assert "framealpha" not in canvas._legend_kwargs()


def test_font_family_reaches_every_piece_of_text(qapp):
    canvas = _fitted_canvas(qapp)
    canvas.set_style(font_family="monospace")
    assert canvas.ax.xaxis.label.get_fontfamily() == ["monospace"]
    assert canvas.ax.title.get_fontfamily() == ["monospace"]
    assert canvas.ax.get_legend().get_texts()[0].get_fontfamily() == ["monospace"]

    canvas.set_style(font_family="")     # the panel's "Default" entry
    assert canvas._font_family is None


def test_contour_levels_set_through_the_style_panel_drive_both_maps(qapp):
    """One number behind the Plot Style box, the right-click dialog and
    the Analysis tab's DOSY levels spin -- a canvas shows a 2D or a
    DOSY, never both, so both are stored and neither is lost by a
    switch between the two views."""
    canvas = MplCanvas()
    canvas.figure.set_size_inches(8, 6)
    canvas.show_spectrum(_make_2d())
    canvas.canvas.draw()
    canvas.set_style(contour_clevels=15)
    assert canvas._contour_clevels == 15
    assert canvas._dosy_clevels == 15
    assert canvas.style_settings()["contour_clevels"] == 15


def test_contour_levels_rebuild_the_map_that_is_showing(qapp):
    """Levels live on the QuadContourSet matplotlib builds, so unlike a
    trace colour they cannot be set on what is already drawn."""
    canvas = MplCanvas()
    canvas.figure.set_size_inches(8, 6)
    canvas.show_spectrum(_make_2d())
    canvas.canvas.draw()
    canvas.set_style(contour_clevels=20)
    canvas.canvas.draw()
    assert canvas._contour_clevels == 20
    assert canvas._contour_spectrum is not None     # still a contour, not cleared


def test_measurement_on_a_compare_row_reports_ppm_and_hz(qapp):
    """A Compare row is a spectrum like any other, but its axis was not
    consulted -- so a measurement there fell through to the "no spectral
    axis" branch and came back as a bare difference. Reported directly:
    "when measuring in compare, the value should show ppm / Hz like in
    the spectrum view"."""
    canvas = MplCanvas()
    canvas.figure.set_size_inches(8, 6)
    canvas.show_compare([_make_1d(), _make_1d()], mode="stack")
    canvas.canvas.draw()
    canvas.set_measurement_mode(True)
    got = []
    canvas.measurementCompleted.connect(got.append)
    row = canvas._compare_axes[0]
    canvas._on_press(_press_at(canvas, row, xfrac=0.3, yfrac=0.5))
    canvas._on_press(_press_at(canvas, row, xfrac=0.7, yfrac=0.5))

    summary = got[0]["summary"]
    assert "ppm" in summary and "Hz" in summary
    assert got[0]["dx"]["ppm"] == pytest.approx(4.0, rel=1e-3)


def test_measurement_in_mm_reports_mm_and_hz_not_ppm(qapp):
    """On an imaging axis the distance is the number wanted, and ppm
    beside it is the one that stops meaning anything. Explicit user
    request: "If the axis is mm, then show mm / Hz"."""
    canvas = MplCanvas()
    canvas.figure.set_size_inches(8, 6)
    canvas.show_spectrum(_make_1d())
    canvas.canvas.draw()
    canvas.set_imaging_gradient_params(gamma_hz_per_t=42.577e6, max_gradient_g_per_cm=50.0)
    canvas.set_axis_unit("mm")
    canvas.canvas.draw()
    canvas.set_measurement_mode(True)
    got = []
    canvas.measurementCompleted.connect(got.append)
    canvas._on_press(_press_at(canvas, canvas.ax, xfrac=0.3, yfrac=0.5))
    canvas._on_press(_press_at(canvas, canvas.ax, xfrac=0.7, yfrac=0.5))

    summary = got[0]["summary"]
    assert "mm" in summary and "Hz" in summary
    assert "ppm" not in summary          # the whole point


def test_measurement_is_always_a_pair_of_units(qapp):
    """Two units, never one and never three -- the pairing is what lets
    you read a position and a coupling off the same gesture."""
    canvas = MplCanvas()
    canvas.show_spectrum(_make_1d())
    assert len(canvas._measurement_unit_order()) == 2
    assert [u for u, _f in canvas._measurement_unit_order()] == ["ppm", "hz"]

    canvas.set_imaging_gradient_params(gamma_hz_per_t=42.577e6, max_gradient_g_per_cm=50.0)
    canvas.set_axis_unit("mm")
    assert [u for u, _f in canvas._measurement_unit_order()] == ["mm", "hz"]


def test_hiding_the_crosshair_blits_instead_of_redrawing(qapp):
    """``draw_idle`` here was a full contour repaint on every exit --
    295 ms for a real HSQC, 1.34 s for the EXSY -- which is exactly the
    cost this class is blitted to avoid. It also ate the strip cursor:
    the app is busy for that whole repaint, so the platform's own wait
    cursor covers the canvas until it finishes."""
    canvas = MplCanvas()
    canvas.figure.set_size_inches(9, 7)
    canvas.show_spectrum(_make_2d())
    canvas.canvas.draw()
    canvas._on_motion(_body_scroll_event(canvas.ax, xdata=5.0, ydata=95.0))
    crosshair = canvas._crosshair
    assert crosshair is not None
    crosshair._background = canvas.canvas.copy_from_bbox(canvas.figure.bbox)
    for line in (*crosshair._vlines, *crosshair._hlines):
        line.set_visible(True)

    draws = []
    canvas.canvas.draw_idle = lambda *a, **k: draws.append(1)
    crosshair._hide()

    assert draws == []                                    # no repaint
    assert not any(line.get_visible()
                   for line in (*crosshair._vlines, *crosshair._hlines))


def test_hiding_falls_back_to_a_redraw_with_no_cached_background(qapp):
    """The one case blitting would leave whatever the buffer holds."""
    canvas = MplCanvas()
    canvas.figure.set_size_inches(9, 7)
    canvas.show_spectrum(_make_2d())
    canvas.canvas.draw()
    canvas._on_motion(_body_scroll_event(canvas.ax, xdata=5.0, ydata=95.0))
    crosshair = canvas._crosshair
    crosshair._background = None
    for line in (*crosshair._vlines, *crosshair._hlines):
        line.set_visible(True)

    draws = []
    canvas.canvas.draw_idle = lambda *a, **k: draws.append(1)
    crosshair._hide()
    assert draws == [1]


def test_the_cursor_is_re_asserted_after_a_scroll(qapp):
    """A contour-level scroll rebuilds the figure through
    _clear_figure, which drops the cached shape -- and no mouse
    movement follows a scroll, so nothing put one back. The canvas sat
    on whatever the platform left until the pointer moved. Reported:
    "the scroll cursor... only briefly appears again, and also it
    should remain while scrolling too"."""
    canvas = MplCanvas()
    canvas.figure.set_size_inches(9, 7)
    canvas.show_spectrum(_make_2d())
    canvas.canvas.draw()

    body = _body_scroll_event(canvas.ax, xdata=5.0, ydata=95.0, button="up")
    canvas._on_motion(body)
    canvas._on_scroll(body)
    # Not None: the cache is re-established by the scroll itself, so a
    # later move into a strip is applied rather than skipped as "no
    # change".
    assert canvas._last_cursor_shape is not None
    assert _shape(canvas) == Qt.CursorShape.ArrowCursor


def test_a_strip_scroll_keeps_the_double_arrow(qapp):
    canvas = MplCanvas()
    canvas.figure.set_size_inches(9, 7)
    canvas.show_spectrum(_make_2d())
    canvas.canvas.draw()

    strip = _x_strip_scroll_event(canvas.ax, xdata=5.0, button="up")
    canvas._on_motion(strip)
    assert _shape(canvas) == Qt.CursorShape.SizeHorCursor
    for _ in range(3):                       # a real scroll is many notches
        canvas._on_scroll(strip)
        assert _shape(canvas) == Qt.CursorShape.SizeHorCursor


def test_the_scroll_refresh_survives_a_dispatch_that_raises(qapp):
    """The refresh is in a `finally` so it covers every one of the
    dispatch's eleven early returns -- and anything that goes wrong in
    one of them."""
    canvas = MplCanvas()
    canvas.figure.set_size_inches(8, 6)
    canvas.show_spectrum(_make_1d())
    canvas.canvas.draw()
    canvas._dispatch_scroll = lambda _event: (_ for _ in ()).throw(RuntimeError("boom"))

    strip = _x_strip_scroll_event(canvas.ax, xdata=3.0, button="up")
    with pytest.raises(RuntimeError):
        canvas._on_scroll(strip)
    assert _shape(canvas) == Qt.CursorShape.SizeHorCursor
