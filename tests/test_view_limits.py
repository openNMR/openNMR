"""Plot Style ▸ Axis limits: exact x (1D, Compare) and x/y (2D contour) limits typed
by the user (2026-10-29). Synthetic data; offscreen Qt."""
import numpy as np
import pytest

from opennmr.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
from opennmr.gui.widgets.canvas import MplCanvas


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@pytest.fixture
def canvas(qapp):
    return MplCanvas()


def spec1d(centre=5.0):
    ppm = np.linspace(10, 0, 400)
    return NMRSpectrum(data=np.exp(-((ppm - centre) ** 2) * 8),
                       axes=[NMRAxis(kind="frequency", ppm=ppm, hz=ppm * 400)],
                       metadata=NMRMetadata(vendor="bruker", nucleus="1H", sfo1=400.0, ns=1))


def spec2d():
    x = np.linspace(10, 0, 96)
    y = np.linspace(150, 50, 48)
    xx, yy = np.meshgrid(x, y)
    data = 1e6 * np.exp(-((xx - 5) ** 2 / 0.02 + (yy - 100) ** 2 / 4))
    return NMRSpectrum(data, [NMRAxis("frequency", ppm=y, hz=y * 100), NMRAxis("frequency", ppm=x, hz=x * 400)],
                       NMRMetadata(vendor="bruker", nucleus="1H", nucleus_indirect="13C",
                                   sfo1=400.0, sfo1_indirect=100.0))


def test_1d_x_limits_keep_the_axis_direction_whatever_the_order(canvas):
    canvas.show_spectrum(spec1d())
    canvas.set_view_limits(x=(2, 8))
    assert canvas.ax.get_xlim() == (8, 2)
    canvas.set_view_limits(x=(8, 2))
    assert canvas.ax.get_xlim() == (8, 2)


def test_1d_intensity_axis_fits_the_window_not_the_whole_spectrum(canvas):
    canvas.show_spectrum(spec1d())
    full_top = canvas.ax.get_ylim()[1]
    canvas.set_view_limits(x=(7, 9))           # the peak (at 5) is outside
    assert canvas.ax.get_ylim()[1] < 0.2 * full_top


def test_one_sided_limit_leaves_the_other_end_at_the_data(canvas):
    canvas.show_spectrum(spec1d())
    canvas.set_view_limits(x=(None, 4))
    left, right = canvas.ax.get_xlim()
    assert left > right and right == pytest.approx(0, abs=0.1) and left == pytest.approx(4)


def test_limits_survive_a_redraw_and_a_new_spectrum(canvas):
    canvas.show_spectrum(spec1d())
    canvas.set_view_limits(x=(2, 8))
    canvas.show_spectrum(spec1d())
    assert canvas.ax.get_xlim() == (8, 2)
    canvas.show_spectrum(spec1d(centre=3.0))
    assert canvas.ax.get_xlim() == (8, 2)


def test_home_returns_to_the_typed_range_and_clearing_to_the_full_view(canvas):
    canvas.show_spectrum(spec1d())
    canvas.set_view_limits(x=(2, 8))
    canvas.ax.set_xlim(6, 5)                    # a zoom
    canvas.reset_view()
    assert canvas.ax.get_xlim() == (8, 2)
    canvas.clear_view_limits()
    left, right = canvas.ax.get_xlim()
    assert left > right and left >= 9.9 and right <= 0.1
    assert canvas.view_limits() == {"x": None, "y": None}


def test_invalid_limits_are_refused(canvas):
    canvas.show_spectrum(spec1d())
    for bad in ((1, 1), (float("nan"), 3), (float("inf"), 3), (1, 2, 3)):
        with pytest.raises(ValueError):
            canvas.set_view_limits(x=bad)
    assert canvas.view_limits()["x"] is None


def test_a_range_outside_the_data_is_said_so(canvas):
    canvas.show_spectrum(spec1d())
    canvas.set_view_limits(x=(50, 60))
    assert "outside the data" in canvas.view_limits_note()


def test_only_supported_views_take_limits(canvas):
    assert canvas.view_limit_support() == (False, False)
    canvas.show_spectrum(spec1d())
    assert canvas.view_limit_support() == (True, False)
    canvas.show_spectrum(spec2d())
    assert canvas.view_limit_support() == (True, True)


def test_2d_x_and_y_limits_survive_a_rebuild(canvas):
    canvas.show_spectrum(spec2d())
    x0, y0 = canvas.ax.get_xlim(), canvas.ax.get_ylim()
    canvas.set_view_limits(x=(2, 8), y=(70, 120))
    assert sorted(canvas.ax.get_xlim()) == [2, 8] and sorted(canvas.ax.get_ylim()) == [70, 120]
    # the axes keep the direction they were drawn in
    assert (canvas.ax.get_xlim()[0] > canvas.ax.get_xlim()[1]) == (x0[0] > x0[1])
    assert (canvas.ax.get_ylim()[0] > canvas.ax.get_ylim()[1]) == (y0[0] > y0[1])
    canvas._redraw_contour(preserve_view=False)
    assert sorted(canvas.ax.get_xlim()) == [2, 8] and sorted(canvas.ax.get_ylim()) == [70, 120]
    canvas.clear_view_limits()
    assert canvas.ax.get_xlim() == pytest.approx(x0) and canvas.ax.get_ylim() == pytest.approx(y0)


def test_limits_are_not_part_of_the_style_that_defaults_and_apply_to_all_carry(canvas):
    canvas.show_spectrum(spec1d())
    canvas.set_view_limits(x=(2, 8))
    assert not any("lim" in key for key in canvas.style_settings())


def test_compare_stack_rows_all_take_the_x_range(canvas):
    canvas.show_compare([spec1d(4.0), spec1d(6.0)], mode="stack")
    canvas.set_view_limits(x=(3, 7))
    assert len(canvas._compare_axes) == 2
    for ax in canvas._compare_axes:
        assert ax.get_xlim() == (7, 3)


def test_limits_are_in_the_saved_figure_and_vector_export_writes(canvas, tmp_path):
    canvas.show_spectrum(spec1d())
    canvas.set_view_limits(x=(2, 8))
    seen = []
    path = tmp_path / "figure.svg"
    canvas.export_full_resolution(lambda: (seen.append(canvas.ax.get_xlim()),
                                           canvas.figure.savefig(path)))
    assert seen == [(8, 2)] and path.stat().st_size > 500
    assert canvas.ax.get_xlim() == (8, 2)


def test_typed_limits_wait_for_the_unit_they_were_typed_in(canvas, qapp):
    canvas.show_spectrum(spec1d())
    canvas.set_view_limits(x=(2, 8))
    canvas.set_axis_unit("hz")
    qapp.processEvents()
    assert max(canvas.ax.get_xlim()) > 100                  # Hz range, not 2-8
    assert "not applied" in canvas.view_limits_note()
    canvas.set_axis_unit("ppm")
    qapp.processEvents()
    assert canvas.ax.get_xlim() == (8, 2) and canvas.view_limits_note() == ""


# --- the Plot Style panel ------------------------------------------------------

@pytest.fixture
def window(qapp):
    from opennmr.gui.main_window import MainWindow
    return MainWindow()


def test_panel_boxes_set_the_current_plot_and_mirror_it(window):
    window._add_derived_spectrum(spec1d(), "S1")
    window.spectrum_list.setCurrentRow(0)
    canvas = window._series_target_canvas()
    canvas.show_spectrum(spec1d())
    window._load_style_controls_from(canvas)
    assert window.output_xlim_low_edit.isEnabled() and not window.output_ylim_low_edit.isEnabled()
    window.output_xlim_low_edit.setText("8")
    window.output_xlim_high_edit.setText("2,5")            # a decimal comma
    window._on_view_limits_edited()
    assert canvas.view_limits()["x"] == (2.5, 8.0) and canvas.ax.get_xlim() == (8.0, 2.5)
    assert window.output_xlim_low_edit.text() == "2.5"      # shown in order
    window.output_xlim_low_edit.setText("abc")
    window._on_view_limits_edited()
    assert window.output_limits_note_label.text() and canvas.view_limits()["x"] == (2.5, 8.0)
    window._on_reset_view_limits_clicked()
    assert canvas.view_limits()["x"] is None and window.output_xlim_low_edit.text() == ""
