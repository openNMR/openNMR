from types import SimpleNamespace

import numpy as np
import pytest
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
