"""2D spectra in Compare: overlay (2-6), side-by-side panels, and zoomed
insets -- the GUI side of the notebook's NMR2D_overlay_plot /
NMR2D_multi_plot / inset=, drawn by opennmr.plotting.twod."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from opennmr.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum  # noqa: E402
from opennmr.gui.main_window import MainWindow  # noqa: E402
from opennmr.notebook import two_d as td  # noqa: E402
from opennmr.plotting import twod  # noqa: E402

from test_gui import qapp, _write_params, _select_rows  # noqa: F401,E402 -- fixture


def _spectrum(shift=0.0, weak=0.04, nuc_indirect="13C"):
    x = np.linspace(10, 0, 256)
    y = np.linspace(150, 50, 128)
    xx, yy = np.meshgrid(x, y)
    data = (1e6 * np.exp(-((xx - 5 - shift) ** 2 / 0.02 + (yy - 100) ** 2 / 4))
            + weak * 1e6 * np.exp(-((xx - 2 - shift) ** 2 / 0.01 + (yy - 70) ** 2 / 2)))
    return NMRSpectrum(data, [NMRAxis("frequency", ppm=y, hz=y * 100), NMRAxis("frequency", ppm=x, hz=x * 400)],
                       NMRMetadata(vendor="bruker", nucleus="1H", nucleus_indirect=nuc_indirect,
                                   sfo1=400.0, sfo1_indirect=100.0))


def _bruker_2d(tmp_path, name, shift=0, f1_nuc="13C"):
    expt = tmp_path / name
    pdata = expt / "pdata" / "1"
    pdata.mkdir(parents=True)
    n_f1, n_f2 = 128, 256
    _write_params(expt / "acqus", {"NUC1": "1H", "BF1": 400.13, "O1": 2000.0, "NS": 1})
    _write_params(expt / "acqu2s", {"NUC1": f1_nuc, "BF1": 100.6, "O1": 10000.0})
    _write_params(pdata / "procs", {"SW_p": 4000.0, "SI": n_f2, "SF": 400.13, "OFFSET": 10.0})
    _write_params(pdata / "proc2s", {"SW_p": 10060.0, "SI": n_f1, "SF": 100.6, "OFFSET": 150.0, "FT_mod": 4})
    xx, yy = np.meshgrid(np.arange(n_f2), np.arange(n_f1))
    data = (1e6 * np.exp(-((xx - 120 - shift) ** 2 / 20 + (yy - 60) ** 2 / 10))
            + 4e4 * np.exp(-((xx - 200 - shift) ** 2 / 10 + (yy - 100) ** 2 / 6)))
    data.astype("int32").tofile(pdata / "2rr")
    return str(expt)


# --- plotting ------------------------------------------------------------
def test_crop_keeps_a_point_past_each_edge_and_refuses_empty_ranges():
    s = _spectrum()
    region = twod.crop_2d(s, (3.0, 1.0), (60.0, 80.0))
    x, y = region.axes[1].ppm, region.axes[0].ppm
    assert x.max() > 3.0 and x.min() < 1.0 and y.min() < 60.0 and y.max() > 80.0
    assert region.data.shape == (len(y), len(x))
    with pytest.raises(ValueError, match="no data points"):
        twod.crop_2d(s, (20.0, 30.0), (60.0, 80.0))


def test_inset_levels_come_from_the_whole_spectrum():
    s = _spectrum()
    fig, ax, _ = twod.plot_2d_contour(s, show_projections=False, factor=0.02, clevels=5,
                                      insets=[{"xlims": (2.5, 1.5), "ylims": (65, 75)}])
    inset = ax.child_axes[0]
    peak = s.data.max()
    expected = twod._contour_levels(0.02 * peak, peak, 5, "power", 1.25)
    assert np.allclose(inset.collections[0].levels, expected, rtol=1e-6)
    # The weak (4 %) peak is in the region and above 2 %: drawn.
    assert len(inset.collections[0].allsegs[0]) > 0
    plt.close(fig)


def test_inset_placement_and_labels_face_inwards():
    assert twod.inset_bounds("upper right", 0.4) == pytest.approx((0.57, 0.57, 0.4, 0.4))
    assert twod.inset_bounds("lower left", 0.3) == pytest.approx((0.03, 0.03, 0.3, 0.3))
    with pytest.raises(ValueError):
        twod.inset_bounds("middle")
    fig, ax = plt.subplots()
    [upper_right] = twod.draw_insets(ax, _spectrum(), [{"xlims": (3, 1), "ylims": (60, 80),
                                                         "corner": "upper right"}])
    [lower_left] = twod.draw_insets(ax, _spectrum(), [{"xlims": (3, 1), "ylims": (60, 80),
                                                        "corner": "lower left"}])
    assert upper_right.yaxis.get_ticks_position() == "left"
    assert lower_left.yaxis.get_ticks_position() == "right"
    assert lower_left.xaxis.get_ticks_position() == "top"
    # Same orientation as the parent: high ppm left, and (F1) bottom.
    assert upper_right.get_xlim()[0] > upper_right.get_xlim()[1]
    assert upper_right.get_ylim()[0] > upper_right.get_ylim()[1]
    plt.close(fig)


def test_overlay_takes_two_to_six():
    fig, ax = twod.overlay_plot([_spectrum(0.1 * i) for i in range(6)])
    assert len(ax.collections) == 6
    plt.close(fig)
    with pytest.raises(ValueError, match="2 to 6"):
        twod.overlay_plot([_spectrum()] * 7)


def test_multi_plot_draws_into_a_given_figure():
    fig = plt.figure()
    out, axes = twod.multi_plot([_spectrum(), _spectrum(0.2)], fig=fig, show_projections=False,
                                panel_kwargs=[{"insets": [{"xlims": (3, 1), "ylims": (60, 80)}]}, {}])
    assert out is fig and len(axes) == 2
    assert len(axes[0].child_axes) == 1 and len(axes[1].child_axes) == 0
    plt.close(fig)


def test_notebook_overlay_limit_raised_to_six():
    def dataset(shift):
        s = _spectrum(shift)
        return {"spectrum": s.data, "x_ppm": s.axes[1].ppm, "y_ppm": s.axes[0].ppm,
                "y_index": np.arange(1, 129), "pseudo_2d": False, "homonuclear": False,
                "SI1": 256, "SI2": 128, "NUC1": "1H", "NUC2": "13C", "SF1": 400.0, "SF2": 100.0}
    fig, ax = td.NMR2D_overlay_plot([dataset(0.1 * i) for i in range(4)])
    plt.close(fig)
    with pytest.raises(ValueError, match="2 to 6"):
        td.NMR2D_overlay_plot([dataset(0)] * 7)


# --- GUI -----------------------------------------------------------------
def _window_with(tmp_path, *specs):
    window = MainWindow()
    for name, shift, nuc in specs:
        window._load(_bruker_2d(tmp_path, name, shift, nuc), vendor="bruker")
    return window


def test_single_2d_is_drawn_and_controls_appear(qapp, tmp_path):  # noqa: F811
    window = _window_with(tmp_path, ("a", 0, "13C"))
    window._populate_compare_list([0])
    window._refresh_compare_view()
    assert not window.compare_2d_controls.isHidden()
    assert "Showing 1 2D spectrum side by side" in window.compare_status_label.text()
    assert len(window.compare_canvas._panel_axes) == 1


def test_add_inset_from_zoom_on_an_overlay(qapp, tmp_path):  # noqa: F811
    window = _window_with(tmp_path, ("a", 0, "13C"), ("b", 6, "13C"))
    _select_rows(window, [0, 1])
    assert "Overlaying 2 2D spectra" in window.compare_status_label.text()
    controls = window.compare_2d_controls

    controls.add_button.click()                       # not zoomed: explained, nothing added
    assert controls.insets() == [] and "Zoom into a region" in window.compare_status_label.text()

    ax = window.compare_canvas.ax
    ax.set_xlim(2.55, 1.69)
    ax.set_ylim(77, 64)
    controls.add_button.click()
    [inset] = controls.insets()
    assert inset["panel"] is None
    assert inset["xlims"] == pytest.approx((2.55, 1.69)) and inset["ylims"] == pytest.approx((64, 77))
    assert inset["corner"] == "lower left"            # farthest from the region (upper right)
    assert inset["factor"] == pytest.approx(0.02)     # weak peak (4 %) would be under the 7 % floor
    controls._debounce.timeout.emit()
    ax = window.compare_canvas.ax
    assert len(ax.child_axes) == 1
    assert ax.get_xlim() == pytest.approx(window.compare_canvas._panel_home_limits[0][0])  # full view again


def test_side_by_side_linked_and_per_panel_insets(qapp, tmp_path):  # noqa: F811
    window = _window_with(tmp_path, ("a", 0, "13C"), ("b", 6, "13C"))
    _select_rows(window, [0, 1])
    controls = window.compare_2d_controls
    controls.layout_combo.setCurrentIndex(1)          # side by side: row
    controls._debounce.timeout.emit()
    first, second = window.compare_canvas._panel_axes
    first.set_xlim(6, 4)
    assert second.get_xlim() == pytest.approx((6, 4))  # linked

    controls.link_check.setChecked(False)
    controls._debounce.timeout.emit()
    first, second = window.compare_canvas._panel_axes
    first.set_xlim(6, 4)
    assert second.get_xlim() != pytest.approx((6, 4))
    first.set_ylim(77, 64)
    first.set_xlim(2.55, 1.69)
    controls.add_button.click()
    assert controls.insets()[0]["panel"] == 0
    controls._debounce.timeout.emit()
    first, second = window.compare_canvas._panel_axes
    assert len(first.child_axes) == 1 and len(second.child_axes) == 0


def test_impossible_overlay_is_shown_side_by_side_with_the_reason(qapp, tmp_path):  # noqa: F811
    window = _window_with(tmp_path, ("het", 0, "13C"), ("n15", 0, "15N"))
    _select_rows(window, [0, 1])
    text = window.compare_status_label.text()
    assert "Can't overlay" in text and "nuclide" in text and "side by side" in text
    assert len(window.compare_canvas._panel_axes) == 2


def test_a_bad_inset_does_not_lose_the_plot(qapp, tmp_path):  # noqa: F811
    window = _window_with(tmp_path, ("a", 0, "13C"), ("b", 6, "13C"))
    _select_rows(window, [0, 1])
    controls = window.compare_2d_controls
    controls.add_inset(None, (40.0, 30.0), (64.0, 77.0))   # nothing there
    controls._debounce.timeout.emit()
    assert "Insets not drawn" in window.compare_status_label.text()
    assert len(window.compare_canvas.ax.collections) >= 2
