"""Compare with several 2Ds: the legend names the checked spectra (not the
first n of the session), an overlay can carry projections, scrolling over
the contours moves the lowest contour and keeps the zoom, and Lock Aspect
Ratio applies to every row of a stacked Compare (user report, 2026-11-23)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from opennmr.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum  # noqa: E402
from opennmr.gui.widgets.canvas import MplCanvas  # noqa: E402
from opennmr.plotting import twod  # noqa: E402

from test_compare_2d import _spectrum, _window_with  # noqa: E402
from test_gui import qapp, _select_rows  # noqa: F401,E402 -- fixture


def test_overlay_plot_draws_one_projection_per_spectrum_in_its_colour():
    spectra = [_spectrum(0.0), _spectrum(0.3)]
    fig, ax = twod.overlay_plot(spectra, show_projections=True, colours=["tab:red", "tab:blue"])
    strips = ax.opennmr_projection_axes
    assert strips["f1"] is not None and strips["f2"] is not None
    assert len(strips["f2"].lines) == 2 and len(strips["f1"].lines) == 2
    assert strips["f2"].lines[0].get_color() == "tab:red"
    # each trace scaled to its own maximum
    assert max(strips["f2"].lines[1].get_ydata()) == pytest.approx(1.0)
    plt.close(fig)


def test_overlay_plot_without_projections_is_unchanged():
    fig, ax = twod.overlay_plot([_spectrum(0.0), _spectrum(0.3)])
    assert ax.opennmr_projection_axes == {"f1": None, "f2": None}
    assert len(fig.axes) == 1
    plt.close(fig)


def test_legend_names_the_checked_spectra_not_the_first_ones(qapp, tmp_path):  # noqa: F811
    window = _window_with(tmp_path, ("a", 0, "13C"), ("b", 6, "13C"), ("c", 3, "13C"))
    _select_rows(window, [1, 2])
    legend = window.compare_canvas.ax.get_legend()
    assert [t.get_text() for t in legend.get_texts()] == [window.session.label(1), window.session.label(2)]


def test_projections_survive_a_layout_change_and_scroll_moves_the_lowest_contour(qapp, tmp_path):  # noqa: F811
    window = _window_with(tmp_path, ("a", 0, "13C"), ("b", 6, "13C"))
    _select_rows(window, [0, 1])
    controls = window.compare_2d_controls
    canvas = window.compare_canvas
    controls.projections_check.setChecked(True)
    controls._debounce.timeout.emit()
    assert canvas._proj_axes["f1"] is not None and canvas._proj_axes["f2"] is not None

    window.layout_mode_combo.setCurrentIndex(1)       # the Compare layout menu
    assert canvas._proj_axes["f2"] is not None

    canvas.ax.set_xlim(7, 3)
    before = controls.floor_spin.value()
    canvas.compareContourScrolled.emit(1.5)
    assert controls.floor_spin.value() == pytest.approx(before * 1.5)
    assert canvas.ax.get_xlim() == pytest.approx((7, 3))      # the zoom is kept
    assert canvas._proj_axes["f2"] is not None
    canvas.compareContourScrolled.emit(1e-6)
    assert controls.floor_spin.value() == pytest.approx(controls.floor_spin.minimum())


def test_scrolling_the_contours_of_side_by_side_panels_moves_the_floor_too(qapp, tmp_path):  # noqa: F811
    window = _window_with(tmp_path, ("a", 0, "13C"), ("b", 6, "13C"))
    _select_rows(window, [0, 1])
    controls = window.compare_2d_controls
    controls.layout_combo.setCurrentIndex(1)
    controls._debounce.timeout.emit()
    before = controls.floor_spin.value()
    window.compare_canvas.compareContourScrolled.emit(0.5)
    assert controls.floor_spin.value() == pytest.approx(before * 0.5)
    assert len(window.compare_canvas._panel_axes) == 2


def test_lock_aspect_ratio_applies_to_every_row_of_a_stacked_compare(qapp):  # noqa: F811
    def one(i):
        x = np.linspace(10, 0, 400)
        return NMRSpectrum(np.exp(-(x - 3 - i) ** 2 / 0.1), [NMRAxis("frequency", ppm=x, hz=x * 400)],
                           NMRMetadata(vendor="bruker", nucleus="1H", sfo1=400.0))
    canvas = MplCanvas()
    canvas.show_compare([one(i) for i in range(3)], mode="stack")
    canvas.set_aspect_ratio(0.4)
    assert len(canvas._compare_axes) == 3
    assert [ax.get_box_aspect() for ax in canvas._compare_axes] == [0.4, 0.4, 0.4]
    canvas.show_compare([one(i) for i in range(4)], mode="stack")     # a new set keeps the lock on all rows
    assert [ax.get_box_aspect() for ax in canvas._compare_axes] == [0.4] * 4
