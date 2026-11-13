"""2026-11-12, explicit user requests:

* Measure "sometimes, like in the Superimposed mode of Compare, gives a
  delta x value but without units (and only one number)" -- it must
  always give a distance in ppm and Hz, for x and y on a 2D, on every plot.
  Where an axis is not a chemical shift (a time, a delay, a D), the
  separation takes the unit its label states, never a bare number.
* On Simulate's SIMPSON shape and Bloch plots the crosshair was "either
  just not present or only appears in one of the plots": it now follows
  the cursor into every plot of the figure, with the vertical line through
  every plot sharing that x axis, as in a stacked Compare.
"""

import re

import numpy as np
import pytest
from PySide6.QtWidgets import QMessageBox

from opennmr.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
from opennmr.gui.widgets.canvas import MplCanvas

from test_gui_canvas import qapp, _make_1d, _make_1d_at, _make_2d, _motion_at, _press_at  # noqa: F401

# A number with nothing after it but the gap before the next part, or the
# end: "Δx = 4.4   Δy = …" is bare; "Δx = 4.4 µs" is not.
BARE = re.compile(r"Δ[xy] = [-+0-9.eE]+(?:\s{3}|$)")


def test_bare_pattern():
    assert BARE.search("Δx = 4.4   Δy = 2 Hz") and BARE.search("Δy = 3")
    assert not BARE.search("Δx = 4.0000 ppm / 1600.00 Hz   Δy = 0 deg")


def _measure(canvas, ax, a=(0.3, 0.5), b=(0.7, 0.5), ax2=None):
    canvas.canvas.draw()
    canvas.set_measurement_mode(True)
    got = []
    canvas.measurementCompleted.connect(got.append)
    canvas._on_press(_press_at(canvas, ax, xfrac=a[0], yfrac=a[1]))
    canvas._on_press(_press_at(canvas, ax2 or ax, xfrac=b[0], yfrac=b[1]))
    assert len(got) == 1, "the second click completed a measurement"
    return got[0]


def _canvas():
    canvas = MplCanvas()
    canvas.figure.set_size_inches(8, 6)
    return canvas


def _assert_ppm_hz(deltas, hz_per_ppm, ppm=None):
    assert deltas["ppm"] > 0
    assert deltas["hz"] == pytest.approx(deltas["ppm"] * hz_per_ppm, rel=1e-6)
    if ppm is not None:
        assert deltas["ppm"] == pytest.approx(ppm, rel=2e-3)


# ---- Measure: ppm and Hz everywhere ---------------------------------------------------------

def test_compare_superimposed_gives_ppm_and_hz(qapp):
    canvas = _canvas()
    canvas.show_compare([_make_1d_at(4.0), _make_1d_at(6.0)], mode="overlay")
    result = _measure(canvas, canvas.ax)
    _assert_ppm_hz(result["dx"], 400.0, ppm=4.0)
    assert "ppm" in result["summary"] and "Hz" in result["summary"]
    assert not BARE.search(result["summary"])


def test_compare_superimposed_in_hz_still_gives_both(qapp):
    canvas = _canvas()
    canvas.show_compare([_make_1d_at(4.0), _make_1d_at(6.0)], mode="overlay")
    canvas.set_axis_unit("hz")
    result = _measure(canvas, canvas.ax)
    _assert_ppm_hz(result["dx"], 400.0, ppm=4.0)


@pytest.mark.parametrize("mode", ["overlay", "stack"])
def test_compare_after_a_1d_in_hz_still_measures_right(qapp, mode):
    # Compare is drawn in ppm; the display unit left over from a 1D in Hz
    # made Measure read those ppm as Hz. Both layouts are converted back to
    # Hz (the overlay too since 2026-11-13, when Compare gained units).
    canvas = _canvas()
    canvas.show_spectrum(_make_1d())
    canvas.set_axis_unit("hz")
    canvas.show_compare([_make_1d_at(4.0), _make_1d_at(6.0)], mode=mode)
    ax = canvas.ax if mode == "overlay" else canvas._compare_axes[0]
    assert canvas._axis_unit == "hz"
    assert max(ax.get_xlim()) > 1000                    # really in Hz now
    result = _measure(canvas, ax)
    _assert_ppm_hz(result["dx"], 400.0, ppm=4.0)


def test_compare_panels_give_ppm_and_hz_on_both_axes_of_a_2d(qapp):
    canvas = _canvas()
    canvas.show_compare_panels([_make_1d(), _make_2d(n_f1=16)], [[0], [1]])
    one_d, two_d = canvas._spectrum_panels
    result = _measure(canvas, one_d)
    _assert_ppm_hz(result["dx"], 400.0)
    result = _measure(canvas, two_d, a=(0.3, 0.3), b=(0.7, 0.7))
    _assert_ppm_hz(result["dx"], 400.0)
    _assert_ppm_hz(result["dy"], 100.0)
    assert "Δy" in result["summary"] and not BARE.search(result["summary"])


def test_two_d_overlay_and_panels_give_both_axes(qapp):
    for show in ("show_2d_overlay", "show_2d_panels"):
        canvas = _canvas()
        getattr(canvas, show)([_make_2d(n_f1=16), _make_2d(n_f1=16)])
        result = _measure(canvas, canvas._panel_axes[0], a=(0.3, 0.3), b=(0.7, 0.7))
        _assert_ppm_hz(result["dx"], 400.0)
        _assert_ppm_hz(result["dy"], 100.0)


def test_heatmap_gives_ppm_and_hz(qapp):
    canvas = _canvas()
    canvas.show_1d_heatmap([_make_1d_at(4.0), _make_1d_at(6.0)])
    result = _measure(canvas, canvas.ax)
    _assert_ppm_hz(result["dx"], 400.0)


def test_contour_both_axes_in_ppm_and_hz(qapp):
    canvas = _canvas()
    canvas.show_spectrum(_make_2d(n_f1=16))
    result = _measure(canvas, canvas.ax, a=(0.3, 0.3), b=(0.7, 0.7))
    _assert_ppm_hz(result["dx"], 400.0)
    _assert_ppm_hz(result["dy"], 100.0)


def test_processing_preview_spectrum_in_ppm_and_hz_and_fid_in_seconds(qapp):
    canvas = _canvas()
    spectrum = _make_1d()
    fid = np.exp(-np.arange(256) / 50.0).astype(complex)
    canvas.show_processing_preview(fid, 1e-4, spectrum)
    result = _measure(canvas, canvas._processing_preview_spec_ax)
    _assert_ppm_hz(result["dx"], 400.0)
    fid_ax = canvas._processing_preview_fid_ax
    if fid_ax.xaxis.label.get_text():
        result = _measure(canvas, fid_ax)
        assert result["dx"]["label_unit"] == "s"
        assert result["summary"].startswith("Δx = ") and " s" in result["summary"]


def _dosy():
    from opennmr.processing import ilt
    b = np.linspace(0.0, 4.0e9, 16)
    ppm = np.linspace(0.0, 10.0, 120)
    data = np.array([np.exp(-((ppm - 3.0) / 0.1) ** 2) * np.exp(-1.5e-9 * x)
                     + np.exp(-((ppm - 7.0) / 0.1) ** 2) * np.exp(-2e-10 * x) for x in b])
    dosy = ilt.dosy_map(b, data, ppm)
    dosy["hz_per_ppm"] = 400.0
    return dosy


def test_dosy_map_gives_ppm_and_hz_and_d_in_its_unit(qapp):
    canvas = _canvas()
    canvas.show_dosy(_dosy(), title="DOSY")
    result = _measure(canvas, canvas.ax, a=(0.3, 0.3), b=(0.7, 0.7))
    _assert_ppm_hz(result["dx"], 400.0)
    assert not BARE.search(result["summary"])
    # The D axis, in its label's unit with the ratio (a log axis), and no
    # mathtext left in it (it was "m²\\,s⁻¹").
    assert result["dy"]["label_unit"] == "m² s⁻¹" and "×" in result["summary"]
    assert "\\" not in result["summary"]


def test_the_dosy_run_records_the_field(qapp, monkeypatch):
    # MainWindow stores Hz per ppm with the map, so Measure can use it.
    from PySide6.QtWidgets import QDialog
    from test_gui import _load_spectrum_directly
    from opennmr.gui.main_window import MainWindow, _DosySettingsDialog
    from opennmr.processing.fitting import gaussian

    b_values = np.linspace(0.0, 4.0e9, 16)
    ppm = np.linspace(0.0, 10.0, 200)
    data = np.array([gaussian(ppm, 3.0, 0.1, np.exp(-1.5e-9 * b)) for b in b_values])
    spectrum = NMRSpectrum(
        data=data,
        axes=[NMRAxis(kind="index", index=np.arange(1, len(b_values) + 1)),
              NMRAxis(kind="frequency", ppm=ppm, hz=ppm * 400.0)],
        metadata=NMRMetadata(vendor="bruker", nucleus="1H", sfo1=400.0, pseudo_2d=True))
    window = MainWindow()
    _load_spectrum_directly(window, spectrum)
    window.fit_type_combo.setCurrentIndex(1)
    window.delay_source_combo.setCurrentIndex(1)
    window.manual_delays_edit.setText(", ".join(f"{b:.8g}" for b in b_values))
    monkeypatch.setattr(_DosySettingsDialog, "exec", lambda self: QDialog.DialogCode.Accepted)
    window._on_dosy_clicked()
    assert window._dosy_result["hz_per_ppm"] == pytest.approx(400.0)


def test_relaxation_style_plot_takes_its_label_unit(qapp):
    canvas = _canvas()
    canvas.show_parameter_trend(np.linspace(0.01, 1.0, 8), np.linspace(1, 2, 8),
                                xlabel="Delay (s)", ylabel="T1 (ms)")
    result = _measure(canvas, canvas.ax, a=(0.3, 0.3), b=(0.7, 0.7))
    assert result["dx"]["label_unit"] == "s" and result["dy"]["label_unit"] == "ms"
    assert re.search(r"Δx = [0-9.]+ s", result["summary"])
    assert re.search(r"Δy = [0-9.]+ ms", result["summary"])


def test_one_d_intensity_is_still_not_quoted(qapp):
    canvas = _canvas()
    canvas.show_spectrum(_make_1d())
    canvas.ax.set_ylabel("Intensity (a.u.)")
    result = _measure(canvas, canvas.ax, a=(0.3, 0.3), b=(0.7, 0.7))
    _assert_ppm_hz(result["dx"], 400.0, ppm=4.0)
    assert "Δy" not in result["summary"]


@pytest.mark.parametrize("label, unit", [
    ("Delay (s)", "s"), ("$\\mathit{t}$ (µs)", "µs"), ("Time / ms", "ms"),
    ("$\\mathit{D}$ / m$^2$ s$^{-1}$", "m² s⁻¹"), ("Field (mT)", "mT"),
    ("$\\mathit{D}$ / m$^{2}$\\,s$^{-1}$", "m² s⁻¹"),
    ("Intensity (a.u.)", ""), ("Slice number", ""),
])
def test_label_units(label, unit):
    assert MplCanvas._label_unit(label) == unit


def _external(canvas, draw, hz_per_ppm=None):
    canvas.show_external_plot(draw, kind="curve", hz_per_ppm=hz_per_ppm)
    canvas.canvas.draw()


def _shape_like(fig):
    """SIMPSON's shape plot: three rows sharing a time axis."""
    axes = fig.subplots(3, 1, sharex=True)
    t = np.linspace(0, 10, 50)
    for ax, y in zip(axes, (np.sin(t), np.cos(t), t)):
        ax.plot(t, y, color="black")
    axes[0].set_ylabel(r"$\mathit{\nu}_1$ (kHz)")
    axes[1].set_ylabel(r"$\mathit{\phi}$ (°)")
    axes[2].set_ylabel("Offset (kHz)")
    axes[2].set_xlabel(r"$\mathit{t}$ (µs)")
    return axes[0]


def _bloch_like(fig):
    """The Bloch plot: the pulse against time, the profile against offset."""
    shape, profile = fig.subplots(2, 1)
    t = np.linspace(0, 100, 50)
    shape.plot(t, np.exp(-((t - 50) / 20) ** 2))
    shape.set_xlabel("Time / µs")
    offset = np.linspace(-50, 50, 80)
    profile.plot(offset, np.tanh(offset / 10))
    profile.set_xlabel("Offset / kHz")
    profile.set_ylabel("Mz")
    return shape


def test_measure_on_a_stacked_external_figure_uses_the_shared_time_label(qapp):
    canvas = _canvas()
    _external(canvas, _shape_like)
    top = canvas.figure.axes[0]              # no x label of its own
    result = _measure(canvas, top, a=(0.3, 0.3), b=(0.7, 0.7))
    assert result["dx"]["label_unit"] == "µs"
    assert result["dy"]["hz"] > 0            # ν1 in kHz -> Hz
    assert not BARE.search(result["summary"])


def test_measure_on_an_offset_axis_in_khz_gives_hz_and_ppm_when_the_field_is_known(qapp):
    canvas = _canvas()
    _external(canvas, _bloch_like, hz_per_ppm=400.0)
    profile = canvas.figure.axes[1]
    result = _measure(canvas, profile)
    dx = result["dx"]
    assert dx["hz"] == pytest.approx(dx["khz"] * 1000.0)
    assert dx["ppm"] == pytest.approx(dx["hz"] / 400.0)
    assert "ppm" in result["summary"] and "Hz" in result["summary"]


# ---- Crosshair on figures drawn elsewhere --------------------------------------------------

def _hover(canvas, ax):
    canvas._on_motion(_motion_at(canvas, ax))
    crosshair = canvas._crosshair
    assert crosshair is not None
    crosshair._onmove(_motion_at(canvas, ax))
    vis_v = {a: line.get_visible() for a, line in zip(crosshair._vline_axes, crosshair._vlines)}
    vis_h = {a: line.get_visible() for a, line in zip(crosshair._hline_axes, crosshair._hlines)}
    return crosshair, vis_v, vis_h


def test_crosshair_runs_through_every_row_sharing_x(qapp):
    canvas = _canvas()
    _external(canvas, _shape_like)
    rows = canvas.figure.axes
    crosshair, vis_v, vis_h = _hover(canvas, rows[1])
    assert set(rows) <= set(crosshair.axes)
    assert all(vis_v[ax] for ax in rows)                 # the x position in every row
    assert vis_h[rows[1]] and not vis_h[rows[0]] and not vis_h[rows[2]]


def test_crosshair_stays_in_its_own_plot_when_the_x_axes_differ(qapp):
    canvas = _canvas()
    _external(canvas, _bloch_like)
    shape, profile = canvas.figure.axes
    _crosshair, vis_v, vis_h = _hover(canvas, profile)
    assert vis_v[profile] and vis_h[profile]             # present in the second plot
    assert not vis_v[shape] and not vis_h[shape]
    _crosshair, vis_v, vis_h = _hover(canvas, shape)
    assert vis_v[shape] and not vis_v[profile]


def test_crosshair_skips_a_colour_bar(qapp):
    canvas = _canvas()

    def draw(fig):
        ax = fig.add_subplot(111)
        image = ax.imshow(np.random.default_rng(0).random((8, 8)))
        fig.colorbar(image, ax=ax)
        return ax
    _external(canvas, draw)
    canvas._on_motion(_motion_at(canvas, canvas.figure.axes[0]))
    crosshair = canvas._crosshair
    assert len(crosshair.axes) == 1


def test_x_linked():
    from matplotlib.figure import Figure
    from opennmr.gui.widgets.canvas import _ContourCrosshair
    fig = Figure()
    a, b = fig.subplots(2, 1, sharex=True)
    c = fig.add_subplot(3, 1, 3)
    assert _ContourCrosshair.x_linked(a, b) and not _ContourCrosshair.x_linked(a, c)
    assert not _ContourCrosshair.x_linked(a, None)


@pytest.fixture
def _no_modal_dialogs(monkeypatch):
    for name in ("warning", "information", "critical"):
        monkeypatch.setattr(QMessageBox, name, lambda *a, **k: None)


def test_simpson_shape_in_the_gui_has_a_crosshair_in_every_row(qapp, _no_modal_dialogs):
    import os
    from opennmr.gui.main_window import MainWindow
    wurst = os.path.join(os.path.dirname(__file__), "data", "simpson", "inv_pow.shape")
    window = MainWindow()
    panel = window.simpson_panel
    panel.add_paths([wurst])
    panel.shape_duration_spin.setValue(10.0)
    panel.plot_button.click()
    canvas = window.fit_canvas
    canvas.set_crosshair_enabled(True)
    canvas.figure.set_size_inches(8, 6)
    canvas.canvas.draw()
    rows = canvas.figure.axes
    assert len(rows) == 3
    _crosshair, vis_v, vis_h = _hover(canvas, rows[2])
    assert all(vis_v[ax] for ax in rows) and vis_h[rows[2]]
    result = _measure(canvas, rows[0])
    assert result["dx"]["label_unit"] == "µs"


def test_bloch_in_the_gui_has_a_crosshair_in_both_plots(qapp):
    from opennmr.gui.main_window import MainWindow
    window = MainWindow()
    window.bloch_shape_combo.setCurrentIndex(window.bloch_shape_combo.findData("GAUSSIAN"))
    window._run_bloch_simulation()
    canvas = window.simulation_canvas
    canvas.set_crosshair_enabled(True)
    canvas.figure.set_size_inches(8, 6)
    canvas.canvas.draw()
    plots = [ax for ax in canvas.figure.axes if ax.axison]
    assert len(plots) >= 2
    for ax in plots:
        _crosshair, vis_v, vis_h = _hover(canvas, ax)
        assert vis_v[ax] and vis_h[ax]
    result = _measure(canvas, plots[-1])
    assert not BARE.search(result["summary"])
