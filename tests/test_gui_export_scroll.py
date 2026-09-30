"""Exports never contain the crosshair; intensity scroll keeps the
baseline still; notebook-figure legends follow Show Legend; Compare is
drawn at full resolution (explicit user reports, 2026-09-28)."""
import io
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

import opennmr.gui.widgets.canvas as canvas_module
from opennmr.gui.widgets.canvas import MplCanvas
from test_dnp import ERRORS, FIELDS, _sweep
from test_gui_canvas import _make_1d, _make_1d_n


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _grey_pixels(png: bytes) -> int:
    from PIL import Image
    im = np.asarray(Image.open(io.BytesIO(png)).convert("RGB")).astype(int)
    # The crosshair is drawn in 0.35 grey (89, 89, 89).
    return int(np.all(np.abs(im - 89) <= 6, axis=-1).sum())


def _hover(canvas):
    ax = canvas.ax
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    canvas._ensure_crosshair()
    canvas._crosshair._onmove(SimpleNamespace(inaxes=ax, xdata=(x0 + x1) / 2,
                                              ydata=(y0 + y1) / 2))


def _export(canvas) -> bytes:
    buffer = io.BytesIO()
    canvas.export_full_resolution(lambda: canvas.figure.savefig(buffer, format="png", dpi=100))
    return buffer.getvalue()


@pytest.mark.parametrize("old_style_lines", [False, True])
def test_an_export_never_contains_the_crosshair(qapp, monkeypatch, old_style_lines):
    if old_style_lines:
        # Whatever the lines are drawn as -- here the axvline/axhline
        # the crosshair used to use, which a save did draw.
        monkeypatch.setattr(canvas_module._ContourCrosshair, "_guide", staticmethod(
            lambda ax, o, p: (ax.axvline(0.5, visible=False, animated=True, **p) if o == "v"
                              else ax.axhline(0.5, visible=False, animated=True, **p))))
    canvas = MplCanvas()
    canvas.resize(800, 600)
    canvas.show()
    canvas.show_spectrum(_make_1d())
    canvas.canvas.draw()
    clean = _grey_pixels(_export(canvas))
    _hover(canvas)
    assert any(line.get_visible() for line in canvas._crosshair._vlines)
    assert _grey_pixels(_export(canvas)) == clean


def _baseline_spectrum(offset=1.0, n=4096):
    s = _make_1d_n(n)
    x = np.arange(s.data.size)
    rng = np.random.default_rng(0)
    s.data = offset + 0.01 * rng.normal(size=x.size) + np.exp(-((x - x.size / 2) / 5.0) ** 2)
    return s


def _scroll(canvas, button):
    ax = canvas.ax
    canvas._on_scroll(SimpleNamespace(
        inaxes=ax, xdata=sum(ax.get_xlim()) / 2, ydata=sum(ax.get_ylim()) / 2, step=1,
        button=button, x=400, y=300, key=None, guiEvent=None, canvas=canvas.canvas))


def _baseline_screen_fraction(ax, level):
    y0, y1 = ax.get_ylim()
    return (level - y0) / (y1 - y0)


@pytest.mark.parametrize("button", ["up", "down"])
def test_intensity_scroll_keeps_a_raised_baseline_still(qapp, button):
    canvas = MplCanvas(full_resolution=True)
    canvas.resize(800, 600)
    canvas.show_spectrum(_baseline_spectrum(offset=1.0))
    ax = canvas.ax
    level = float(np.median(ax.lines[0].get_ydata()))   # the plotted (normalised) baseline
    assert level > 0.3
    ax.set_ylim(0.0, 3.0 * level)      # the reported case: view from 0, baseline well above it
    before = _baseline_screen_fraction(ax, level)
    for _ in range(3):
        _scroll(canvas, button)
    assert ax.get_ylim()[1] != pytest.approx(3.0 * level)
    assert _baseline_screen_fraction(ax, level) == pytest.approx(before, abs=0.01)


def test_baseline_anchor_is_the_midpoint_of_the_two_ends(qapp):
    from matplotlib.figure import Figure
    ax = Figure().add_subplot()
    x = np.linspace(0, 100, 1000)
    y = 2.0 + 0.02 * x                 # 2 at one end, 4 at the other
    ax.plot(x, y)
    ax.set_xlim(0, 100)
    assert MplCanvas.baseline_anchor(ax) == pytest.approx(3.0, abs=0.02)
    ax.set_xlim(0, 50)                 # only what is visible counts
    assert MplCanvas.baseline_anchor(ax) == pytest.approx(2.5, abs=0.02)


def test_curve_plots_keep_the_pinned_bottom_zoom(qapp):
    canvas = MplCanvas()
    canvas.resize(800, 600)

    def draw(figure):
        ax = figure.add_subplot()
        ax.plot(np.linspace(0, 1, 50), np.linspace(5, 10, 50))
    canvas.show_external_plot(draw, kind="curve")
    ax = canvas.ax
    bottom = ax.get_ylim()[0]
    _scroll(canvas, "up")
    assert ax.get_ylim()[0] == pytest.approx(bottom)


# -- the main window -------------------------------------------------------------

@pytest.fixture
def window(qapp):
    from opennmr.gui.main_window import MainWindow
    w = MainWindow()
    w.resize(1400, 900)
    w.show()
    for k, s in enumerate(_sweep(FIELDS, ERRORS)):
        w._add_derived_spectrum(s, f"s{k}")
    return w


def test_dnp_profile_draws_as_a_curve_plot(window):
    # Apply ▸ DNP (2026-09-30): the profile is a "curve" plot on the Fit view.
    panel = window.dnp_sweep_panel
    panel.refresh()
    panel._tick_all(True)
    import numpy as np
    ref = panel._spectra_list()[0]
    panel.set_line(float(ref.axes[0].ppm[np.argmax(ref.data)]))
    panel.align()
    panel.confirm()
    pk = panel.alignment["peak_ppm"]
    panel.set_region(pk - 20, pk + 20)
    assert panel.profile is not None
    assert window.fit_canvas.plot_kind() == "curve"
    assert window.canvas_tabs.currentWidget() is window.fit_canvas


def test_compare_draws_every_point(window):
    window._rebuild_compare_list([(0, None), (1, None)], preserve_previous=False)
    window.layout_mode_combo.setCurrentIndex(1)
    window._refresh_compare_view(switch_tab=True, force_render=True)
    n = window.session.spectra[0].data.size
    assert window.compare_canvas._full_resolution
    assert all(line.get_xdata().size == n for line in window.compare_canvas.ax.lines[:2])
