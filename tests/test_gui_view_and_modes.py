"""Home, one-interaction-mode-at-a-time, the Compare heatmap, and
drag-and-drop into Compare (explicit user requests, 2026-09-23)."""
import numpy as np
import pytest
from matplotlib.backend_bases import MouseEvent
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QDropEvent

from opennmr.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
from opennmr.gui.main_window import MainWindow, _ExperimentDropList
from opennmr import plotting

from test_gui import (qapp, _load_spectrum_directly, _make_1d_dataset, _make_xpulse_dataset,  # noqa: F401 -- qapp is a fixture
                      _select_rows)


def _limits(canvas):
    return [(ax.get_xlim(), ax.get_ylim()) for ax in canvas.figure.axes]


def _scroll(canvas, qapp, where, clicks=3):
    """Scroll as the user does: over the x tick strip (x zoom) or the
    plot body (intensity zoom)."""
    ax = canvas.figure.axes[0]
    bb = ax.bbox
    x = bb.x0 + bb.width / 2
    y = bb.y0 - 8 if where == "x" else bb.y0 + bb.height / 2
    for _ in range(clicks):
        canvas.canvas.callbacks.process(
            "scroll_event", MouseEvent("scroll_event", canvas.canvas, x, y, button="up", step=1))
    qapp.processEvents()


@pytest.fixture
def xpulse_window(qapp, tmp_path):
    window = MainWindow()
    window.resize(1300, 850)
    window.show()
    window._load(_make_xpulse_dataset(tmp_path), vendor="xpulse")
    yield window
    window.close()


# ============================================================
# Home
# ============================================================

@pytest.mark.parametrize("view", ["canvas", "fid_canvas", "processing_preview_canvas"])
@pytest.mark.parametrize("via", ["house icon", "View > Home"])
def test_home_restores_the_view_after_a_scroll_zoom(qapp, xpulse_window, view, via):
    # Reported on the FID and Processing views: after zooming, Home did
    # nothing. The canvas's own scroll-zoom never wrote matplotlib's
    # navigation stack, which is all Home replayed -- true of every plot.
    window = xpulse_window
    canvas = getattr(window, view)
    if view == "processing_preview_canvas":
        window.side_tabs.setCurrentIndex(window._processing_tab_index)
    window.canvas_tabs.setCurrentWidget(canvas)
    qapp.processEvents()
    full = _limits(canvas)
    _scroll(canvas, qapp, "x")
    _scroll(canvas, qapp, "y")
    assert _limits(canvas) != full
    if via == "house icon":
        canvas.toolbar.home()
    else:
        window._reset_navigation_view()
    assert _limits(canvas) == full


def test_home_after_a_drag_pan(qapp, xpulse_window):
    canvas = xpulse_window.fid_canvas
    xpulse_window.canvas_tabs.setCurrentWidget(canvas)
    qapp.processEvents()
    full = _limits(canvas)
    ax = canvas.figure.axes[0]
    bb = ax.bbox
    x0, y0 = bb.x0 + bb.width / 2, bb.y0 + bb.height / 2
    for name, x in (("button_press_event", x0), ("motion_notify_event", x0 + 60),
                    ("button_release_event", x0 + 60)):
        canvas.canvas.callbacks.process(name, MouseEvent(name, canvas.canvas, x, y0, button=1))
    assert _limits(canvas) != full
    canvas.toolbar.home()
    # To 1e-12, not exactly: for an autoscaled axis Home re-derives the
    # limits from the (display-downsampled) data rather than replaying
    # stored numbers, and the last bit can differ with the canvas's
    # pixel width (it did, by 3e-16 relative, when the Title / notes
    # strip narrowed the plot by 22 px).
    assert np.asarray(_limits(canvas)) == pytest.approx(np.asarray(full), rel=1e-12, abs=1e-15)


def test_home_is_the_new_plots_view_not_the_last_ones(qapp, xpulse_window, tmp_path):
    # The record is per plot: a stale one would send Home to the view of
    # whatever was drawn before.
    window = xpulse_window
    canvas = window.canvas
    window.canvas_tabs.setCurrentWidget(canvas)
    _scroll(canvas, qapp, "x")
    window._load(_make_1d_dataset(tmp_path, name="other"), vendor="bruker")
    qapp.processEvents()
    full = _limits(canvas)
    _scroll(canvas, qapp, "x")
    canvas.toolbar.home()
    assert _limits(canvas) == full


def test_home_survives_a_redraw_that_keeps_the_zoom(qapp, xpulse_window):
    # The processing preview keeps a zoom across live redraws; Home must
    # still reach the full view afterwards, not the kept zoom.
    window = xpulse_window
    canvas = window.processing_preview_canvas
    window.side_tabs.setCurrentIndex(window._processing_tab_index)
    window.canvas_tabs.setCurrentWidget(canvas)
    qapp.processEvents()
    full = _limits(canvas)
    _scroll(canvas, qapp, "x")
    kept = canvas.processing_preview_limits()
    window._refresh_processing_preview_if_active(preserve_view=True)
    canvas.restore_processing_preview_limits(kept)
    assert _limits(canvas) != full
    canvas.toolbar.home()
    home = _limits(canvas)
    # Home is the full view of the plot *as now drawn*: exactly what a
    # fresh redraw without the kept zoom gives. (Its autoscaled
    # intensity range may differ slightly from the very first draw's.)
    window._refresh_processing_preview_if_active(preserve_view=False)
    assert home == _limits(canvas)
    assert home[0][0] == full[0][0]                    # the time axis is back in full


# ============================================================
# One interaction mode at a time
# ============================================================

def test_pick_peaks_releases_the_toolbar_pan(qapp, xpulse_window):
    # Reported: with pan (or zoom) highlighted, Pick peaks could not
    # pick -- the toolbar consumed every click.
    window = xpulse_window
    canvas = window.canvas
    canvas.toolbar.pan()
    assert canvas.toolbar.mode.name == "PAN"
    window.pick_peak_button.setChecked(True)
    assert not canvas.toolbar.mode
    assert canvas._pick_mode


def test_choosing_pan_or_zoom_releases_pick_and_unchecks_its_button(qapp, xpulse_window):
    window = xpulse_window
    canvas = window.canvas
    window.pick_peak_button.setChecked(True)
    canvas.toolbar.zoom()
    assert canvas.toolbar.mode.name == "ZOOM"
    assert not canvas._pick_mode
    assert not window.pick_peak_button.isChecked()     # the button tells the truth


def test_pick_and_select_region_are_exclusive(qapp, xpulse_window):
    window = xpulse_window
    window.pick_peak_button.setChecked(True)
    window.select_region_button.setChecked(True)
    assert window.canvas._region_select_mode and not window.canvas._pick_mode
    assert not window.pick_peak_button.isChecked()
    window.pick_peak_button.setChecked(True)
    assert not window.select_region_button.isChecked()


def test_measure_releases_pick_and_crosshair_everywhere(qapp, xpulse_window):
    window = xpulse_window
    canvas = window.canvas
    window.pick_peak_button.setChecked(True)
    assert canvas._crosshair_enabled
    canvas._measure_button.click()
    assert canvas._measurement_mode
    assert not canvas._pick_mode and not window.pick_peak_button.isChecked()
    assert not canvas._crosshair_enabled
    # Measure applies to every view, and the menu agrees.
    for other in window._measurement_canvases():
        assert other._measurement_mode and not other._crosshair_enabled
    assert window.measurement_menu_action.isChecked()


def test_crosshair_releases_measure_everywhere(qapp, xpulse_window):
    # "on clicking Crosshair, Measure (if selected) should be deselected;
    # likewise for the opposite".
    window = xpulse_window
    window.fid_canvas._measure_button.click()
    window.fid_canvas._crosshair_button.click()
    for canvas in window._measurement_canvases():
        assert canvas._crosshair_enabled and not canvas._measurement_mode
    assert not window.measurement_menu_action.isChecked()
    assert not window.fid_canvas._measure_button.isChecked()


def test_crosshair_stays_on_through_pan_and_zoom(qapp, xpulse_window):
    # A display overlay, not a click mode: deliberately not released.
    canvas = xpulse_window.canvas
    canvas.toolbar.zoom()
    assert canvas._crosshair_enabled


# ============================================================
# Compare heatmap
# ============================================================

def _peak_spectrum(shift, amplitude, low=0.0, high=10.0):
    x = np.linspace(high, low, 2000)
    y = amplitude / (1 + ((x - shift) / 0.05) ** 2)
    return NMRSpectrum(data=y, axes=[NMRAxis(kind="frequency", ppm=x, hz=x * 400.0)],
                       metadata=NMRMetadata(vendor="bruker", nucleus="1H", sfo1=400.0))


@pytest.fixture
def heatmap_window(qapp):
    window = MainWindow()
    window.show()
    for i, (shift, amplitude) in enumerate([(2.0, 1.0), (3.0, 0.5), (4.0, 0.25)]):
        _load_spectrum_directly(window, _peak_spectrum(shift, amplitude, low=-1.0 if i else 0.5))
    window.side_tabs.setCurrentIndex(window._compare_tab_index)
    _select_rows(window, [0, 1, 2])
    window.layout_mode_combo.setCurrentIndex(window.layout_mode_combo.findText("Heatmap"))
    qapp.processEvents()
    yield window
    window.close()


def test_compare_offers_a_heatmap_with_rows_in_list_order(heatmap_window):
    window = heatmap_window
    ax = window.compare_canvas.ax
    labels = [t.get_text() for t in ax.get_yticklabels()]
    expected = [window.compare_list.item(i).text().split(" · ")[0] for i in range(3)]
    # First list row at the top: the y-axis is inverted, so tick order
    # bottom-to-top is the list order top-to-bottom.
    assert ax.yaxis_inverted()
    assert labels == expected
    mesh = ax.collections[0]
    matrix = np.asarray(mesh.get_array()).reshape(3, -1)
    peaks = [ax.get_xlim()]  # noqa: F841 -- documents that x is ppm, high to low
    assert np.argmax(matrix.max(axis=1)) == 0         # the strongest spectrum is row 0


def test_heatmap_reports_the_common_ppm_range(heatmap_window):
    # Rows are interpolated onto the range every spectrum covers; data
    # outside it is not shown, so the status line says which range.
    text = heatmap_window.compare_status_label.text()
    assert "Heatmap over 10 to 0.5 ppm" in text


@pytest.mark.parametrize("norm_index, label", [
    (0, "Intensity (a.u.)"), (1, "Intensity / own maximum")])
def test_heatmap_colour_bar_says_what_the_scale_means(heatmap_window, qapp, norm_index, label):
    window = heatmap_window
    window.compare_normalize_combo.setCurrentIndex(norm_index)
    qapp.processEvents()
    colorbar_axes = [a for a in window.compare_canvas.figure.axes if a is not window.compare_canvas.ax]
    assert colorbar_axes and colorbar_axes[0].get_ylabel() == label
    mesh = window.compare_canvas.ax.collections[0]
    if norm_index == 1:
        assert np.nanmax(mesh.get_array()) == pytest.approx(1.0, abs=0.02)


def test_heatmap_uses_the_whole_colour_map():
    # plot_1d_heatmap borrowed the contour-line map, whose light end is
    # trimmed for line contrast on white -- the strongest signal came
    # out mid-scale. A filled heatmap needs the full map.
    spectra = [_peak_spectrum(2.0, 1.0), _peak_spectrum(3.0, 1.0)]
    _fig, _ax, mesh = plotting.plot_1d_heatmap(spectra, cmap="viridis")
    top = mesh.cmap(1.0)
    import matplotlib.pyplot as plt
    assert np.allclose(top, plt.get_cmap("viridis")(1.0))


def test_stacked_and_superimposed_still_work(heatmap_window, qapp):
    window = heatmap_window
    window.layout_mode_combo.setCurrentIndex(0)
    qapp.processEvents()
    assert window.compare_canvas._compare_axes
    window.layout_mode_combo.setCurrentIndex(1)
    qapp.processEvents()
    assert len(window.compare_canvas.figure.axes) == 1


# ============================================================
# Drag experiments into Compare
# ============================================================

def _entries(window):
    return [window.compare_list.item(i).data(Qt.ItemDataRole.UserRole)
            for i in range(window.compare_list.count())]


def _drag_rows_onto_compare(window, rows, onto_row=None):
    model = window.spectrum_list.model()
    mime = model.mimeData([model.index(r, 0) for r in rows])
    if onto_row is None:
        pos = QPointF(5, window.compare_list.viewport().height() - 2)
    else:
        pos = QPointF(window.compare_list.visualItemRect(window.compare_list.item(onto_row)).center())
    event = QDropEvent(pos, Qt.DropAction.CopyAction, mime, Qt.MouseButton.LeftButton,
                       Qt.KeyboardModifier.NoModifier)
    window.compare_list.dropEvent(event)


@pytest.fixture
def four_spectra(qapp, tmp_path):
    window = MainWindow()
    window.show()
    for name in "abcd":
        window._load(_make_1d_dataset(tmp_path, name=name), vendor="bruker")
    window.side_tabs.setCurrentIndex(window._compare_tab_index)
    yield window
    window.close()


def test_drag_payload_decodes_to_the_dragged_rows(four_spectra):
    model = four_spectra.spectrum_list.model()
    mime = model.mimeData([model.index(r, 0) for r in (3, 1)])
    assert _ExperimentDropList.rows_from_mime(mime) == [3, 1]


def test_the_open_list_is_a_drag_source_and_compare_accepts_drops(four_spectra):
    assert four_spectra.spectrum_list.dragEnabled()
    assert four_spectra.compare_list.acceptDrops()


def test_dropped_experiments_go_where_they_are_dropped(four_spectra):
    window = four_spectra
    _select_rows(window, [0, 1])
    _drag_rows_onto_compare(window, [3, 2], onto_row=1)
    assert _entries(window) == [(0, None), (3, None), (2, None), (1, None)]
    assert all(window.compare_list.item(i).checkState() == Qt.CheckState.Checked for i in range(4))


def test_dropping_on_empty_space_appends_and_duplicates_are_ignored(four_spectra):
    window = four_spectra
    _select_rows(window, [0, 1])
    _drag_rows_onto_compare(window, [1, 2])           # 1 is already there
    assert _entries(window) == [(0, None), (1, None), (2, None)]


def test_dropped_entries_survive_a_selection_change(four_spectra):
    # The list is rebuilt from the selection, which would drop anything
    # not selected -- dragged-in entries are the user's own, so they stay.
    window = four_spectra
    _select_rows(window, [0, 1])
    _drag_rows_onto_compare(window, [3])
    _select_rows(window, [1, 2])
    assert (3, None) in _entries(window)
    assert (0, None) not in _entries(window)          # selection-driven, now deselected


def test_remove_from_comparison_unpins(four_spectra):
    window = four_spectra
    _select_rows(window, [0, 1])
    _drag_rows_onto_compare(window, [3])
    window._remove_compare_entry((3, None))
    assert (3, None) not in _entries(window)
    _select_rows(window, [1, 2])
    assert (3, None) not in _entries(window)


def test_dropping_a_pseudo_2d_adds_its_slices(qapp, tmp_path, four_spectra):
    window = four_spectra
    series = NMRSpectrum(
        data=np.ones((3, 16)),
        axes=[NMRAxis(kind="index", index=np.arange(1, 4)),
              NMRAxis(kind="frequency", ppm=np.linspace(10, 0, 16), hz=np.linspace(10, 0, 16) * 400)],
        metadata=NMRMetadata(vendor="bruker", nucleus="1H", sfo1=400.0, pseudo_2d=True))
    _load_spectrum_directly(window, series)
    window.compare_list.clear()
    window._on_experiments_dropped_on_compare([4], -1)
    assert _entries(window) == [(4, 1), (4, 2), (4, 3)]


@pytest.mark.parametrize("route", ["Remove button", "right-click Remove"])
def test_removing_an_experiment_renumbers_compare(four_spectra, route, monkeypatch):
    # Removing shifts every later row down by one. The right-click
    # Remove did not touch Compare at all, so its entries silently
    # pointed at the wrong spectra.
    window = four_spectra
    _select_rows(window, [0, 1])
    _drag_rows_onto_compare(window, [3])
    name_of_3 = window.session.label(3)
    if route == "Remove button":
        window.spectrum_list.setCurrentRow(0)
        window._on_remove_clicked()
    else:
        window.session.remove(0)
        window.spectrum_list.takeItem(0)
        window._after_session_row_removed(0)
    assert (2, None) in window._compare_pinned_entries
    labels = [window.compare_list.item(i).text() for i in range(window.compare_list.count())]
    assert name_of_3 in labels


def test_clear_all_drops_the_pins(four_spectra):
    window = four_spectra
    _select_rows(window, [0, 1])
    _drag_rows_onto_compare(window, [3])
    window._on_clear_clicked()
    assert window._compare_pinned_entries == []
