from types import SimpleNamespace

import types

import numpy as np
from matplotlib.backend_bases import MouseEvent
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QComboBox, QDialog, QSpinBox

from opennmr.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
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


def test_nearest_slice_index_matches_cursor_ppm_on_a_frequency_axis(qapp):
    spec = _make_2d(n_f1=8, n_f2=64)  # F1 ppm: linspace(90, 100, 8); F2 ppm: linspace(0, 10, 64)
    row = MplCanvas._nearest_slice_index(spec, "f1", 95.0)
    assert spec.axes[0].ppm[row - 1] == pytest.approx(95.0, abs=0.75)
    col = MplCanvas._nearest_slice_index(spec, "f2", 5.0)
    assert spec.axes[1].ppm[col - 1] == pytest.approx(5.0, abs=0.2)


def test_nearest_slice_index_clips_to_valid_range(qapp):
    spec = _make_2d(n_f1=8, n_f2=64)
    assert MplCanvas._nearest_slice_index(spec, "f1", 1000.0) == 8  # far past the axis's own max ppm
    assert MplCanvas._nearest_slice_index(spec, "f1", -1000.0) == 1


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

    assert MplCanvas._nearest_slice_index(spec, "f1", 3.4) == 3
    assert MplCanvas._nearest_slice_index(spec, "f1", 100.0) == n_f1  # clipped


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
    expected_row = MplCanvas._nearest_slice_index(spec, "f1", 95.0)
    assert seen["index_spin"].value() == expected_row
    assert expected_row != 1  # confirms this is a real, non-trivial default, not a coincidence

    # Switching to "Column (fix F2)" should re-derive from the cursor's
    # X position (5 ppm) instead.
    seen["axis_combo"].setCurrentIndex(1)
    expected_col = MplCanvas._nearest_slice_index(spec, "f2", 5.0)
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


def test_set_axis_unit_noop_without_a_1d_spectrum(qapp):
    canvas = MplCanvas()
    canvas.show_spectrum(_make_2d())
    canvas.set_axis_unit("hz")  # must not raise, and must not touch the contour view
    assert canvas._axis_unit == "ppm"


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


def test_canvas_crosshair_spans_the_contour_and_both_projections(qapp):
    # "When the input isn't panning or zooming, the cursor should be a
    # cross with lines covering the entire contour space AND the
    # projections." Two cursors, not one: a single MultiCursor over all
    # three axes would also draw a horizontal line across the top strip
    # and a vertical one down the left strip, in those panels' own
    # unrelated intensity coordinates.
    canvas = MplCanvas()
    canvas.show_spectrum(_make_2d(n_f1=8))
    canvas._sync_crosshair()

    assert len(canvas._crosshairs) == 2
    vertical, horizontal = canvas._crosshairs
    assert vertical.vertOn and not vertical.horizOn
    assert horizontal.horizOn and not horizontal.vertOn
    assert set(vertical.axes) == {canvas.ax, canvas._proj_axes["f2"]}
    assert set(horizontal.axes) == {canvas.ax, canvas._proj_axes["f1"]}


def _motion_at(canvas, ax, xfrac=0.5, yfrac=0.5):
    bbox = ax.get_window_extent()
    return MouseEvent("motion_notify_event", canvas.canvas,
                      bbox.x0 + xfrac * bbox.width, bbox.y0 + yfrac * bbox.height)


def test_canvas_crosshair_stands_down_during_pan_and_zoom(qapp):
    canvas = MplCanvas()
    canvas.figure.set_size_inches(8, 6)
    canvas.show_spectrum(_make_2d(n_f1=8))
    canvas.canvas.draw()
    canvas._sync_crosshair()
    cursor = canvas._crosshairs[0]
    event = _motion_at(canvas, canvas.ax)

    canvas.toolbar.mode = ""
    cursor.onmove(event)
    assert cursor.vlines[0].get_visible()

    for mode in ("pan/zoom", "zoom rect"):
        canvas.toolbar.mode = mode
        cursor.onmove(event)
        assert not cursor.vlines[0].get_visible(), mode
    canvas.toolbar.mode = ""


def test_canvas_crosshair_refreshes_its_blit_background_in_every_mode(qapp):
    # Real user report: "zooming on the 2D no longer works. Something
    # has been broken."
    #
    # MultiCursor.clear -- the draw_event handler that *recaptures the
    # blit background* -- goes through ignore() as well as onmove does.
    # Gating ignore() on the toolbar mode therefore froze that
    # background for as long as pan or zoom was selected, and the next
    # mouse move blitted the stale image back over the canvas: the zoom
    # had happened, but the display reverted to the frame captured
    # before it. Pan being the default mode on load made this the normal
    # case rather than an edge one. Standing down belongs in onmove, not
    # ignore.
    canvas = MplCanvas()
    canvas.figure.set_size_inches(8, 6)
    canvas.show_spectrum(_make_2d(n_f1=8))
    canvas.canvas.draw()
    canvas._sync_crosshair()
    cursor = canvas._crosshairs[0]

    for mode in ("", "pan/zoom", "zoom rect"):
        canvas.toolbar.mode = mode
        for info in cursor._canvas_infos.values():
            info["background"] = None
        canvas.canvas.draw()
        assert all(info["background"] is not None
                   for info in cursor._canvas_infos.values()), mode
    canvas.toolbar.mode = ""


def test_canvas_crosshair_is_rebuilt_when_the_figure_changes(qapp):
    # This canvas clears and rebuilds its figure from a dozen methods; a
    # cursor still holding removed Axes would draw into them. The sync
    # is driven from motion rather than from every redraw path, so it
    # has to notice for itself.
    canvas = MplCanvas()
    canvas.show_spectrum(_make_2d(n_f1=8))
    canvas._sync_crosshair()
    first = canvas._crosshairs[0]
    stale_axes = canvas._crosshair_axes

    canvas.show_spectrum(_make_2d(n_f1=8))          # new figure, new Axes
    canvas._sync_crosshair()
    assert canvas._crosshairs[0] is not first
    assert canvas._crosshair_axes != stale_axes
    assert all(a in canvas.figure.axes for a in canvas._crosshair_axes)

    # ...and torn down entirely once there is no contour to hover over.
    canvas.show_spectrum(_make_1d())
    canvas._sync_crosshair()
    assert canvas._crosshairs == []
