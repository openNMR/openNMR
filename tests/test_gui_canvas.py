from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

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


def test_extract_slice_shows_1d_column(qapp):
    canvas = MplCanvas()
    spec = _make_2d(n_f1=8, n_f2=64)
    canvas.show_spectrum(spec)
    canvas.extract_slice("f2", 10)
    assert canvas._current_1d_spectrum.ndim == 1
    assert len(canvas._current_1d_spectrum.data) == 8  # traced along F1


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
    canvas.set_axis_unit("mm")  # not yet supported -- silently ignored, not an error
    assert canvas._axis_unit == "ppm"


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


def test_set_style_dpi_and_figsize(qapp):
    canvas = MplCanvas()
    canvas.set_style(dpi=150, fig_width=8.0, fig_height=5.0)
    assert canvas.figure.get_dpi() == pytest.approx(150)
    assert canvas.figure.get_size_inches() == pytest.approx([8.0, 5.0])


def test_set_style_compare_colormap_used_on_next_redraw(qapp):
    canvas = MplCanvas()
    canvas.set_style(compare_colormap="plasma")
    assert canvas._compare_colormap == "plasma"
