"""2026-10-22: (1) the dashed zoom box's thickness is a setting; (2) an exported
figure is written at the DPI the Output tab says, and is never left with
something touching or crossing its edge.

The DPI bug: matplotlib's ``savefig`` defaults to ``savefig.dpi = "figure"``,
which for a Qt canvas means the dpi the figure was created with (screen
resolution) -- ``figure.set_dpi`` does not change it. So an "800 dpi" export
came out at ~100 dpi.
"""
import io
import json

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from opennmr import plotting
from opennmr.gui.widgets.canvas import MplCanvas
from opennmr.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _make_1d():
    ppm = np.linspace(0, 10, 200)
    metadata = NMRMetadata(vendor="bruker", nucleus="1H", sfo1=400.0)
    axis = NMRAxis(kind="frequency", ppm=ppm, hz=ppm * 400.0)
    return NMRSpectrum(data=np.exp(-((ppm - 5) ** 2)), axes=[axis], metadata=metadata)


def _png_size(data: bytes):
    from PIL import Image
    return Image.open(io.BytesIO(data)).size


# ---------------------------------------------------------------------------
# zoom box
# ---------------------------------------------------------------------------

def test_zoom_box_default_is_thicker_than_matplotlibs_hairline(qapp):
    canvas = MplCanvas()
    assert canvas.zoom_box_width() == MplCanvas.DEFAULT_ZOOM_BOX_WIDTH
    assert MplCanvas.DEFAULT_ZOOM_BOX_WIDTH > 1.0


def test_zoom_box_width_is_set_and_bad_values_ignored(qapp):
    canvas = MplCanvas()
    assert canvas.set_zoom_box_width(4.5) is True
    assert canvas.zoom_box_width() == pytest.approx(4.5)
    for bad in ("thick", None, 0.0, -1, 99):
        assert canvas.set_zoom_box_width(bad) is False
    assert canvas.zoom_box_width() == pytest.approx(4.5)


def test_zoom_box_is_drawn_with_the_chosen_width(qapp):
    """The paint callback the canvas installs uses a pen of the set width."""
    from PySide6.QtGui import QImage, QPainter
    canvas = MplCanvas()
    canvas.set_zoom_box_width(6.0)
    seen = []

    class Recorder(QPainter):
        def setPen(self, pen):                              # noqa: N802 (Qt name)
            seen.append(pen.widthF())
            super().setPen(pen)

    canvas.canvas.drawRectangle((10, 10, 50, 40))
    image = QImage(100, 100, QImage.Format.Format_ARGB32)
    painter = Recorder(image)
    canvas.canvas._draw_rect_callback(painter)
    painter.end()
    assert seen and all(w == pytest.approx(6.0) for w in seen)
    # and None clears it
    canvas.canvas.drawRectangle(None)
    painter = Recorder(image)
    seen.clear()
    canvas.canvas._draw_rect_callback(painter)
    painter.end()
    assert seen == []


def test_toolbar_rubberband_goes_through_the_adjustable_box(qapp):
    canvas = MplCanvas()
    canvas.show_spectrum(_make_1d())
    canvas.set_zoom_box_width(3.0)
    canvas.toolbar.draw_rubberband(None, 10, 10, 60, 50)
    assert canvas.canvas._draw_rect_callback is not None
    # the callback is ours, not matplotlib's (ours closes over the width)
    assert canvas.canvas.drawRectangle == canvas._draw_zoom_rectangle
    canvas.toolbar.remove_rubberband()


@pytest.fixture
def window(qapp):
    from opennmr.gui.main_window import MainWindow
    w = MainWindow()
    w.resize(1200, 800)
    yield w
    w.close()


def test_settings_box_changes_every_plot(window):
    window.zoom_box_width_spin.setValue(5.0)
    assert all(c.zoom_box_width() == pytest.approx(5.0) for c in window._all_canvases())


def test_settings_reset_button(window):
    window._set_zoom_box_width(7.0)
    window._set_zoom_box_width(MplCanvas.DEFAULT_ZOOM_BOX_WIDTH)
    assert window.zoom_box_width_spin.value() == pytest.approx(MplCanvas.DEFAULT_ZOOM_BOX_WIDTH)


def test_zoom_box_width_is_remembered(qapp):
    from opennmr.gui.main_window import MainWindow
    first = MainWindow()
    first._set_zoom_box_width(3.5)
    with open(MainWindow._UI_SETTINGS_PATH, encoding="utf-8") as handle:
        assert json.load(handle)["zoom_box_width"] == pytest.approx(3.5)
    first.close()
    second = MainWindow()
    try:
        assert all(c.zoom_box_width() == pytest.approx(3.5) for c in second._all_canvases())
    finally:
        second.close()


# ---------------------------------------------------------------------------
# export DPI and layout
# ---------------------------------------------------------------------------

def _export_png(canvas) -> bytes:
    """What the toolbar's Save does: ``figure.savefig(fname)`` with NO dpi."""
    buffer = io.BytesIO()
    canvas.export_full_resolution(lambda: canvas.figure.savefig(buffer, format="png"))
    return buffer.getvalue()


@pytest.mark.parametrize("dpi", [100, 300, 800])
def test_export_is_written_at_the_requested_dpi(qapp, dpi):
    canvas = MplCanvas()
    canvas.show_spectrum(_make_1d())
    canvas.set_export_settings(dpi=dpi, fig_width=3.0, fig_height=2.0)
    width, height = _png_size(_export_png(canvas))
    assert (width, height) == (3 * dpi, 2 * dpi)


def test_export_dpi_is_not_the_screen_dpi_regression(qapp):
    """The bug itself: the figure's creation dpi (what ``savefig.dpi='figure'``
    means) must not decide the export."""
    canvas = MplCanvas()
    canvas.show_spectrum(_make_1d())
    assert canvas.figure._original_dpi != 800
    canvas.set_export_settings(dpi=800, fig_width=2.0, fig_height=1.5)
    assert _png_size(_export_png(canvas))[0] == 1600


def test_live_figure_is_restored_and_repaints_resume(qapp):
    canvas = MplCanvas()
    canvas.show_spectrum(_make_1d())
    dpi, size = canvas.figure.get_dpi(), tuple(canvas.figure.get_size_inches())
    canvas.set_export_settings(dpi=600, fig_width=4.0, fig_height=3.0)
    seen = {}
    canvas.export_full_resolution(
        lambda: seen.update(updates=canvas.canvas.updatesEnabled(), dpi=canvas.figure.get_dpi()))
    assert seen["updates"] is False and seen["dpi"] == 600     # no live repaint at 600 dpi
    assert canvas.canvas.updatesEnabled() is True
    assert canvas.figure.get_dpi() == pytest.approx(dpi)
    assert tuple(canvas.figure.get_size_inches()) == pytest.approx(size)


def test_repaints_resume_if_the_save_raises(qapp):
    canvas = MplCanvas()
    canvas.show_spectrum(_make_1d())

    def boom():
        raise RuntimeError("disk full")

    with pytest.raises(RuntimeError):
        canvas.export_full_resolution(boom)
    assert canvas.canvas.updatesEnabled() is True


def test_a_normal_figure_is_not_cropped(qapp):
    """A laid-out figure keeps its requested size exactly (no tight crop)."""
    canvas = MplCanvas()
    canvas.show_spectrum(_make_1d())
    canvas.set_export_settings(dpi=200, fig_width=5.0, fig_height=3.0)
    seen = {}
    canvas.export_full_resolution(lambda: seen.update(bbox=matplotlib_rc("savefig.bbox")))
    assert seen["bbox"] != "tight"


def matplotlib_rc(key):
    import matplotlib
    return matplotlib.rcParams[key]


def test_something_hanging_off_the_edge_is_kept_not_clipped(qapp):
    """An annotation outside the axes that the layout pass does not know about:
    the export falls back to cropping to the content (with a margin)."""
    canvas = MplCanvas()
    canvas.show_spectrum(_make_1d())
    canvas.ax.annotate("a note at the very edge", xy=(0.5, 0.5), xycoords="axes fraction",
                       xytext=(1.0, 1.0), textcoords="figure fraction", ha="right", va="top",
                       annotation_clip=False)
    canvas.set_export_settings(dpi=200, fig_width=5.0, fig_height=3.0)
    seen = {}
    canvas.export_full_resolution(lambda: seen.update(
        bbox=matplotlib_rc("savefig.bbox"), pad=matplotlib_rc("savefig.pad_inches")))
    assert seen["bbox"] == "tight"
    assert seen["pad"] == pytest.approx(MplCanvas._EXPORT_PAD_INCHES)
    # ... and the rc is back to normal afterwards
    assert matplotlib_rc("savefig.bbox") != "tight"


def test_nothing_in_the_saved_image_touches_the_edge(qapp):
    from PIL import Image
    canvas = MplCanvas()
    canvas.show_spectrum(_make_1d())
    canvas.ax.annotate("edge", xy=(0.5, 0.5), xycoords="axes fraction", xytext=(1.0, 0.5),
                       textcoords="figure fraction", ha="left", va="center", fontsize=14,
                       annotation_clip=False)
    canvas.set_export_settings(dpi=150, fig_width=4.0, fig_height=3.0)
    image = np.asarray(Image.open(io.BytesIO(_export_png(canvas))).convert("L"))
    for edge in (image[0, :], image[-1, :], image[:, 0], image[:, -1]):
        assert (edge > 250).all()          # white border on all four sides


def test_edge_crowding_helper(qapp):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot([0, 1], [0, 1])
    fig.tight_layout()
    assert plotting.edge_crowding(fig) == []
    ax.text(1.0, 1.0, "over the top", transform=fig.transFigure, ha="right", va="top")
    assert "right" in plotting.edge_crowding(fig) and "top" in plotting.edge_crowding(fig)
    plt.close(fig)
