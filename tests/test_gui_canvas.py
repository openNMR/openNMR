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


def test_scroll_up_zooms_in_around_cursor(qapp):
    canvas = MplCanvas()
    canvas.show_spectrum(_make_1d())
    ax = canvas.ax
    ax.set_xlim(0, 10)
    event = SimpleNamespace(inaxes=ax, xdata=3.0, ydata=None, button="up")
    canvas._on_scroll(event)
    xmin, xmax = ax.get_xlim()
    assert (xmax - xmin) < 10  # narrower than before
    assert xmin < 3.0 < xmax  # cursor position still inside view


def test_scroll_down_zooms_out(qapp):
    canvas = MplCanvas()
    canvas.show_spectrum(_make_1d())
    ax = canvas.ax
    ax.set_xlim(4, 6)
    event = SimpleNamespace(inaxes=ax, xdata=5.0, ydata=None, button="down")
    canvas._on_scroll(event)
    xmin, xmax = ax.get_xlim()
    assert (xmax - xmin) > 2


def test_scroll_outside_axes_is_noop(qapp):
    canvas = MplCanvas()
    canvas.show_spectrum(_make_1d())
    ax = canvas.ax
    ax.set_xlim(0, 10)
    event = SimpleNamespace(inaxes=None, xdata=None, ydata=None, button="up")
    canvas._on_scroll(event)
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
    ax = canvas.ax
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    original_factor = canvas._contour_factor

    event = SimpleNamespace(inaxes=ax, xdata=5.0, ydata=95.0, button="up")
    canvas._on_scroll(event)

    assert canvas._contour_factor != pytest.approx(original_factor)
    # The pan/zoom view itself must be unaffected -- scrolling over a
    # contour changes the levels, not the axis range.
    assert canvas.ax.get_xlim() == pytest.approx(xlim)
    assert canvas.ax.get_ylim() == pytest.approx(ylim)


def test_scroll_up_over_contour_raises_factor(qapp):
    canvas = MplCanvas()
    canvas.show_spectrum(_make_2d())
    before = canvas._contour_factor
    canvas._on_scroll(SimpleNamespace(inaxes=canvas.ax, xdata=5.0, ydata=95.0, button="up"))
    assert canvas._contour_factor > before


def test_scroll_down_over_contour_lowers_factor(qapp):
    canvas = MplCanvas()
    canvas.show_spectrum(_make_2d())
    before = canvas._contour_factor
    canvas._on_scroll(SimpleNamespace(inaxes=canvas.ax, xdata=5.0, ydata=95.0, button="down"))
    assert canvas._contour_factor < before


def test_contour_factor_clamped_to_bounds(qapp):
    canvas = MplCanvas()
    canvas.show_spectrum(_make_2d())
    for _ in range(60):  # far more notches than needed to hit either bound
        canvas._on_scroll(SimpleNamespace(inaxes=canvas.ax, xdata=5.0, ydata=95.0, button="up"))
    assert canvas._contour_factor <= canvas._CONTOUR_FACTOR_MAX
    for _ in range(60):
        canvas._on_scroll(SimpleNamespace(inaxes=canvas.ax, xdata=5.0, ydata=95.0, button="down"))
    assert canvas._contour_factor >= canvas._CONTOUR_FACTOR_MIN


def test_scroll_over_1d_still_zooms_normally(qapp):
    canvas = MplCanvas()
    canvas.show_spectrum(_make_1d())
    ax = canvas.ax
    ax.set_xlim(0, 10)
    canvas._on_scroll(SimpleNamespace(inaxes=ax, xdata=3.0, ydata=None, button="up"))
    xmin, xmax = ax.get_xlim()
    assert (xmax - xmin) < 10


# ============================================================
# FID window-function overlay
# ============================================================

def test_show_fid_without_envelope_has_two_lines(qapp):
    canvas = MplCanvas()
    fid = np.exp(-np.arange(50) / 10.0) * np.exp(1j * np.arange(50) * 0.3)
    canvas.show_fid(fid, dt=1e-4)
    assert len(canvas.ax.lines) == 2  # real, imag


def test_show_fid_with_envelope_adds_third_line_and_legend(qapp):
    canvas = MplCanvas()
    fid = np.exp(-np.arange(50) / 10.0) * np.exp(1j * np.arange(50) * 0.3)
    envelope = np.exp(-np.arange(50) / 20.0)
    canvas.show_fid(fid, dt=1e-4, window_envelope=envelope)
    assert len(canvas.ax.lines) == 3
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
