"""2026-10-19:

* Compare: a click on a spectrum in the plot makes that spectrum the list's
  current row, so Move Up / Move Down work after it; Up / Down arrow keys
  move the highlighted spectrum (from the plot or from the list);
* 2D contours: the colour is an Output ▸ Plot Style choice (black, one
  colour, or a colour map coloured by height) -- the notebook's
  ``colour_spectra`` -- and ``colour_spectra`` also takes a plain colour;
* Offset Tracker: several regions (typed or dragged), integrated together;
  a region (or several) selected in the list is plotted and kept; every
  region on one plot, stacked or superimposed.
"""
import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QPushButton
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure

from opennmr.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
from opennmr.gui.main_window import MainWindow
from opennmr.gui.offset_tracker_panel import OffsetTrackerPanel
from opennmr.gui.widgets.canvas import MplCanvas
from opennmr.notebook import offset_tracker as ot
from opennmr.plotting import style as pstyle

from test_gui import (  # noqa: F401 -- qapp is a fixture
    _make_1d_dataset, _make_2d_dataset, _select_rows, qapp,
)


# ======================================================================
# Compare: click a spectrum, then Move Up / Move Down / arrow keys
# ======================================================================

def _compare_three(window, tmp_path):
    for name in ("a", "b", "c"):
        window._load(_make_1d_dataset(tmp_path, name=name), vendor="bruker")
    _select_rows(window, [0, 1, 2])
    window.compare_list.resize(300, 400)
    window._refresh_compare_view(switch_tab=True, force_render=True)
    assert window.compare_list.count() == 3
    return window.compare_canvas


def _order(window):
    return [window.compare_list.item(i).text() for i in range(window.compare_list.count())]


def _button(window, text):
    return next(b for b in window.findChildren(QPushButton) if b.text() == text)


def test_a_click_on_a_spectrum_makes_it_the_lists_current_row(qapp, tmp_path):
    window = MainWindow()
    canvas = _compare_three(window, tmp_path)
    window.compare_list.setCurrentRow(-1)
    canvas._toggle_compare_selection(canvas._compare_axes[1])      # what a click does
    assert window.compare_list.currentRow() == 1
    assert canvas._selected_compare_index == 1


def test_move_up_and_down_work_after_clicking_the_plot(qapp, tmp_path):
    window = MainWindow()
    canvas = _compare_three(window, tmp_path)
    before = _order(window)
    window.compare_list.setCurrentRow(-1)
    canvas._toggle_compare_selection(canvas._compare_axes[1])
    _button(window, "Move Up").click()
    assert _order(window) == [before[1], before[0], before[2]]
    # the same spectrum is still the highlighted one, now first
    assert window.compare_list.currentRow() == 0
    assert canvas._selected_compare_index == 0
    _button(window, "Move Down").click()
    assert _order(window) == before


def test_clicking_the_same_spectrum_again_lets_go_of_the_list_row(qapp, tmp_path):
    window = MainWindow()
    canvas = _compare_three(window, tmp_path)
    canvas._toggle_compare_selection(canvas._compare_axes[2])
    assert window.compare_list.currentRow() == 2
    canvas._toggle_compare_selection(canvas._compare_axes[2])
    assert window.compare_list.currentRow() == -1


def test_arrow_keys_move_the_spectrum_highlighted_in_the_plot(qapp, tmp_path):
    window = MainWindow()
    canvas = _compare_three(window, tmp_path)
    before = _order(window)
    canvas._toggle_compare_selection(canvas._compare_axes[1])
    QTest.keyClick(canvas.canvas, Qt.Key.Key_Up)
    assert _order(window) == [before[1], before[0], before[2]]
    QTest.keyClick(canvas.canvas, Qt.Key.Key_Down)
    QTest.keyClick(canvas.canvas, Qt.Key.Key_Down)
    assert _order(window) == [before[0], before[2], before[1]]
    assert window.compare_list.currentRow() == 2


def test_arrow_keys_move_the_spectrum_highlighted_in_the_list(qapp, tmp_path):
    window = MainWindow()
    _compare_three(window, tmp_path)
    before = _order(window)
    window.compare_list.setCurrentRow(2)
    QTest.keyClick(window.compare_list, Qt.Key.Key_Up)
    assert _order(window) == [before[0], before[2], before[1]]
    assert window.compare_list.currentRow() == 1


def test_arrow_keys_do_nothing_past_the_ends_and_never_crash(qapp, tmp_path):
    window = MainWindow()
    _compare_three(window, tmp_path)
    before = _order(window)
    window.compare_list.setCurrentRow(0)
    QTest.keyClick(window.compare_list, Qt.Key.Key_Up)
    assert _order(window) == before and window.compare_list.currentRow() == 0
    window.compare_list.setCurrentRow(2)
    QTest.keyClick(window.compare_list, Qt.Key.Key_Down)
    assert _order(window) == before and window.compare_list.currentRow() == 2


def test_arrow_keys_with_nothing_highlighted_leave_the_order_alone(qapp, tmp_path):
    window = MainWindow()
    canvas = _compare_three(window, tmp_path)
    before = _order(window)
    window.compare_list.setCurrentRow(-1)
    QTest.keyClick(canvas.canvas, Qt.Key.Key_Up)
    QTest.keyClick(window.compare_list, Qt.Key.Key_Down)
    assert _order(window) == before


def test_arrow_keys_with_a_modifier_are_not_taken(qapp, tmp_path):
    window = MainWindow()
    _compare_three(window, tmp_path)
    before = _order(window)
    window.compare_list.setCurrentRow(1)
    QTest.keyClick(window.compare_list, Qt.Key.Key_Up, Qt.KeyboardModifier.ShiftModifier)
    assert _order(window) == before


# ======================================================================
# 2D contour colour
# ======================================================================

def _grid_peak():
    x = np.linspace(-3, 3, 61)
    xx, yy = np.meshgrid(x, x)
    z = 5.0 * np.exp(-(xx ** 2 + yy ** 2)) + 1.0 * np.exp(-((xx - 1.5) ** 2 + (yy + 1.0) ** 2))
    return x, x, z, np.array([0.1, 0.3, 0.6, 1.0, 2.0, 4.0])


def _edge_colours(cs):
    return {tuple(np.round(c, 6)) for c in np.atleast_2d(cs.get_edgecolor())}


def test_resolve_contour_colouring_accepts_defaults_colours_lists_and_maps():
    assert pstyle.resolve_contour_colouring(None) == ("default", None)
    assert pstyle.resolve_contour_colouring("") == ("default", None)
    assert pstyle.resolve_contour_colouring("crimson") == ("colors", ["crimson"])
    assert pstyle.resolve_contour_colouring("#1b6ca8") == ("colors", ["#1b6ca8"])
    assert pstyle.resolve_contour_colouring(["red", "blue"]) == ("colors", ["red", "blue"])
    kind, cmap = pstyle.resolve_contour_colouring("viridis")
    assert kind == "cmap" and cmap is not None


def test_resolve_contour_colouring_names_what_is_accepted_on_a_typo():
    with pytest.raises(ValueError, match="Unknown contour colour"):
        pstyle.resolve_contour_colouring("not-a-colour")
    with pytest.raises(ValueError, match="viridis"):
        pstyle.resolve_contour_colouring(["red", "not-a-colour"])


def test_draw_positive_contours_default_is_black_one_colour_and_a_map():
    import matplotlib.pyplot as plt
    x, y, z, levels = _grid_peak()
    fig, ax = plt.subplots()
    black = pstyle.draw_positive_contours(ax, x, y, z, levels, None)
    assert _edge_colours(black) == {tuple(np.round(to_rgba("black"), 6))}
    one = pstyle.draw_positive_contours(ax, x, y, z, levels, "crimson")
    assert _edge_colours(one) == {tuple(np.round(to_rgba("crimson"), 6))}
    height = pstyle.draw_positive_contours(ax, x, y, z, levels, "viridis")
    assert len(_edge_colours(height)) > 1                      # colour follows the height
    two = pstyle.draw_positive_contours(ax, x, y, z, levels, ["red", "blue"])
    assert {tuple(np.round(to_rgba(c), 6)) for c in ("red", "blue")} >= _edge_colours(two)
    plt.close(fig)


def test_plot_2d_contour_takes_a_plain_colour_as_colour_spectra():
    import matplotlib.pyplot as plt
    from opennmr.plotting import plot_2d_contour
    from test_gui_canvas import _make_2d
    spectrum = _make_2d(n_f1=32, n_f2=64)
    fig, ax = plt.subplots()
    plot_2d_contour(spectrum, ax=ax, fig=fig, colour_spectra="crimson")
    sets = [c for c in ax.collections if hasattr(c, "levels")]
    assert sets and _edge_colours(sets[0]) == {tuple(np.round(to_rgba("crimson"), 6))}
    plt.close(fig)


def _contour_sets(canvas):
    return [c for c in canvas.ax.collections if hasattr(c, "levels")]


def test_canvas_contour_colour_single_colour_map_and_back_to_black(qapp):
    from test_gui_canvas import _make_2d
    canvas = MplCanvas()
    canvas.show_spectrum(_make_2d(n_f1=32, n_f2=64))
    assert canvas.style_settings()["contour_colour"] == ""
    assert _edge_colours(_contour_sets(canvas)[0]) == {tuple(np.round(to_rgba("black"), 6))}
    canvas.set_style(contour_colour="#cc0000")
    assert _edge_colours(_contour_sets(canvas)[0]) == {tuple(np.round(to_rgba("#cc0000"), 6))}
    assert canvas.style_settings()["contour_colour"] == "#cc0000"
    canvas.set_style(contour_colour="viridis")
    assert len(_edge_colours(_contour_sets(canvas)[0])) > 1
    canvas.set_style(contour_colour="")
    assert _edge_colours(_contour_sets(canvas)[0]) == {tuple(np.round(to_rgba("black"), 6))}


def test_canvas_contour_colour_survives_a_level_change_and_rejects_typos(qapp):
    from test_gui_canvas import _make_2d
    canvas = MplCanvas()
    canvas.show_spectrum(_make_2d(n_f1=32, n_f2=64))
    canvas.set_style(contour_colour="#cc0000")
    canvas.set_style(contour_clevels=9)                          # rebuilds the contour
    assert _edge_colours(_contour_sets(canvas)[0]) == {tuple(np.round(to_rgba("#cc0000"), 6))}
    with pytest.raises(ValueError, match="Unknown contour colour"):
        canvas.set_style(contour_colour="nonsense")
    assert canvas.style_settings()["contour_colour"] == "#cc0000"     # unchanged by the failure


def test_canvas_saved_style_carries_the_contour_colour(qapp):
    canvas = MplCanvas()
    canvas.apply_style_defaults({"contour_colour": "plasma"})
    assert canvas.style_settings()["contour_colour"] == "plasma"
    canvas.apply_style_defaults({"contour_colour": ""})
    assert canvas.style_settings()["contour_colour"] == ""


def test_output_tab_has_a_contour_colour_choice_that_reaches_the_canvases(qapp):
    window = MainWindow()
    combo = window.output_contour_colour_combo
    assert combo.itemData(0) == "" and combo.findData("viridis") >= 0
    assert "colorbar" not in combo.toolTip().lower().replace("no colour bar", "")
    combo.setCurrentIndex(combo.findData("viridis"))
    assert window._current_output_style_kwargs()["contour_colour"] == "viridis"
    assert all(c.style_settings()["contour_colour"] == "viridis"
               for c in window._style_target_canvases())
    window._set_custom_contour_colour("#00aa00")
    assert combo.currentData() == "#00aa00"
    assert all(c.style_settings()["contour_colour"] == "#00aa00"
               for c in window._style_target_canvases())
    # a second custom colour replaces the first rather than piling up
    window._set_custom_contour_colour("#aa0000")
    assert combo.findData("#00aa00") < 0 and combo.currentData() == "#aa0000"
    combo.setCurrentIndex(0)
    assert all(c.style_settings()["contour_colour"] == "" for c in window._style_target_canvases())


def test_output_tab_follows_the_canvas_it_shows(qapp):
    window = MainWindow()
    window.canvas.set_style(contour_colour="#123456")
    window._load_style_controls_from(window.canvas)
    assert window.output_contour_colour_combo.currentData() == "#123456"
    window.canvas.set_style(contour_colour="")
    window._load_style_controls_from(window.canvas)
    assert window.output_contour_colour_combo.currentData() == ""


# ======================================================================
# Offset Tracker: several regions
# ======================================================================

def _two_peak_spectra():
    """Peak A at 0 ppm follows the offset around 0 ppm; peak B at 20 ppm
    follows it around +5 ppm (2000 Hz of 400 MHz)."""
    x = np.linspace(50, -50, 4001)
    out = []
    for o2 in np.arange(-4000.0, 4001.0, 500.0):
        a = np.exp(-(o2 / 1000.0) ** 2) * np.exp(-x ** 2 / 2)
        b = 3.0 * np.exp(-((o2 - 2000.0) / 1000.0) ** 2) * np.exp(-(x - 20.0) ** 2 / 2)
        meta = NMRMetadata(vendor="bruker", nucleus="13C", sfo1=100.0, bf1=100.0, ns=1.0,
                           raw={"acqus": {"O2": o2, "BF2": 400.0, "NUC2": "<1H>", "NS": 1}})
        out.append(NMRSpectrum(data=a + b, axes=[NMRAxis(kind="frequency", ppm=x, hz=x * 100.0)],
                               metadata=meta, data_imag=np.zeros_like(x)))
    return out


A, B = (5.0, -5.0), (25.0, 15.0)


def test_offset_series_regions_one_series_per_region_on_the_same_offsets():
    series = ot.offset_series_regions(_two_peak_spectra(), [A, B], channel="O2",
                                      baseline="none", divide_rg=False)
    assert len(series) == 2
    np.testing.assert_allclose(series[0]["offset_ppm"], series[1]["offset_ppm"])
    assert series[0]["region"] == (-5.0, 5.0) and series[1]["region"] == (15.0, 25.0)
    assert series[0]["name"] == "5 to -5 ppm" and series[1]["name"] == "25 to 15 ppm"
    peak = lambda s: float(s["offset_ppm"][int(np.argmax(s["y"]))])     # noqa: E731
    assert peak(series[0]) == pytest.approx(0.0)
    assert peak(series[1]) == pytest.approx(5.0)
    named = ot.offset_series_regions(_two_peak_spectra(), [A, B], names=["CH", "CH3"])
    assert [s["name"] for s in named] == ["CH", "CH3"]


def test_offset_series_regions_checks_its_input():
    with pytest.raises(ValueError, match="No regions"):
        ot.offset_series_regions(_two_peak_spectra(), [])
    with pytest.raises(ValueError, match="names"):
        ot.offset_series_regions(_two_peak_spectra(), [A, B], names=["only one"])


@pytest.mark.parametrize("style", ["bars", "stems", "lineshape"])
def test_stacked_regions_each_have_a_baseline_a_name_and_no_y_numbers(style):
    series = ot.offset_series_regions(_two_peak_spectra(), [A, B], names=["CH", "CH3"])
    fig, ax = ot.plot_offset_series_regions(series, style=style, arrangement="stacked",
                                            lineshape="gaussian", fwhm=1.0)
    assert sorted(t.get_text() for t in ax.texts) == ["CH", "CH3"]
    assert list(ax.get_yticks()) == []
    assert "shifted" in ax.get_ylabel()
    # the first region is drawn above the second
    texts = {t.get_text(): t.xy[1] for t in ax.texts}
    assert texts["CH"] > texts["CH3"]


def test_superimposed_regions_share_a_baseline_with_a_legend():
    series = ot.offset_series_regions(_two_peak_spectra(), [A, B], names=["CH", "CH3"])
    fig, ax = ot.plot_offset_series_regions(series, style="stems", arrangement="superimposed")
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["CH", "CH3"]
    assert not ax.texts
    zero_lines = [ln for ln in ax.lines if list(ln.get_ydata()) == [0.0, 0.0]]
    assert len(zero_lines) == 2 and len({tuple(ln.get_ydata()) for ln in zero_lines}) == 1
    # x runs high to low ppm as a spectrum does
    left, right = ax.get_xlim()
    assert left > right


def test_scale_each_puts_every_region_on_a_largest_of_one():
    series = ot.offset_series_regions(_two_peak_spectra(), [A, B], baseline="none", divide_rg=False)
    assert max(series[1]["y"]) > 2 * max(series[0]["y"])         # B is the bigger one
    fig, ax = ot.plot_offset_series_regions(series, style="stems", arrangement="superimposed",
                                            scale_each=True)
    tops = [max(ln.get_ydata()) for ln in ax.lines
            if len(ln.get_ydata()) == len(series[0]["y"]) and ln.get_linestyle() == "None"]
    assert tops and all(t == pytest.approx(1.0) for t in tops)
    assert "each region" in ax.get_ylabel()


def test_regions_plot_rejects_a_bad_arrangement_style_or_nothing():
    series = ot.offset_series_regions(_two_peak_spectra(), [A, B])
    with pytest.raises(ValueError, match="arrangement"):
        ot.plot_offset_series_regions(series, arrangement="sideways")
    with pytest.raises(ValueError, match="style"):
        ot.plot_offset_series_regions(series, style="triangles")
    with pytest.raises(ValueError, match="No series"):
        ot.plot_offset_series_regions([])


def test_single_series_plot_is_unchanged_by_the_refactor():
    series = ot.offset_series(_two_peak_spectra(), A, channel="O2")
    fig, ax = ot.plot_offset_series(series, style="bars")
    assert len(ax.patches) == len(series["y"])
    assert ax.get_xlim()[0] > ax.get_xlim()[1]


def test_one_import_has_the_region_functions():
    import opennmr.notebook as nmr
    for name in ("offset_series_regions", "plot_offset_series_regions"):
        assert name in nmr.__all__ and hasattr(nmr, name)


# ---- the panel ---------------------------------------------------------------

class _Rig:
    """A standalone Offset Tracker panel over two-peak spectra, with a
    recording draw and keep."""

    def __init__(self):
        self.spectra = _two_peak_spectra()
        self.figures: list[Figure] = []
        self.kept: list[tuple] = []
        self.highlights: list = []
        entries = [{"label": f"s{i}", "spectrum": s, "index": i}
                   for i, s in enumerate(self.spectra)]

        def draw(fn, view):
            fig = Figure()
            fn(fig)
            self.figures.append(fig)

        self.panel = OffsetTrackerPanel(
            spectra=lambda: entries, draw=draw, arm=lambda mode: None,
            highlight=lambda regions: self.highlights.append(list(regions)),
            keep=lambda spectrum, label, description: self.kept.append((spectrum, label)))
        self.panel.refresh()
        self.panel.tick_all_button.click()
        self.panel.read_offsets()

    @property
    def ax(self):
        return self.figures[-1].axes[0]


def test_dragging_regions_adds_them_and_integrates_each(qapp):
    rig = _Rig()
    panel = rig.panel
    panel.armed = True
    panel.on_canvas_region(5, -5)
    panel.on_canvas_region(25, 15)
    assert panel.regions() == [(-5.0, 5.0), (15.0, 25.0)]
    assert panel.regions_list.count() == 2
    assert all("integrated" in panel.regions_list.item(i).text() for i in range(2))
    assert len(panel._results) == 2
    # a repeat of the same region is not added twice
    panel.on_canvas_region(-5, 5)
    assert panel.regions_list.count() == 2
    # every region is shaded on Compare, named
    names = [r[0] for r in rig.highlights[-1]]
    assert len(names) == 2 and names[0].startswith("R1") and names[1].startswith("R2")
    assert names[1].endswith("(selected)")             # the last one added is the selected one


def test_adding_a_region_without_enough_spectra_ticked_waits(qapp):
    rig = _Rig()
    panel = rig.panel
    panel.untick_all_button.click()
    panel.add_region(5, -5)
    assert panel.regions() == [(-5.0, 5.0)]
    assert panel._results == {}
    assert "integrated" not in panel.regions_list.item(0).text()
    assert "two spectra" in panel.spectra_status.text()


def test_selecting_a_region_and_plotting_it_shows_that_regions_integrals(qapp):
    rig = _Rig()
    panel = rig.panel
    panel.add_region(*A)
    panel.add_region(*B)
    panel.regions_list.clearSelection()
    panel.regions_list.item(1).setSelected(True)
    series = panel.plot_regions()
    assert series["region"] == (15.0, 25.0)
    assert panel._plotted == [(15.0, 25.0)]
    peak = float(series["offset_ppm"][int(np.argmax(series["y"]))])
    assert peak == pytest.approx(5.0)
    # one region: the ordinary single plot, no legend or stacked names
    assert not rig.ax.texts and rig.ax.get_legend() is None
    panel.regions_list.clearSelection()
    panel.regions_list.item(0).setSelected(True)
    series = panel.plot_regions()
    assert float(series["offset_ppm"][int(np.argmax(series["y"]))]) == pytest.approx(0.0)


def test_plot_all_regions_stacked_and_superimposed(qapp):
    rig = _Rig()
    panel = rig.panel
    panel.add_region(*A)
    panel.add_region(*B)
    panel.arrangement_combo.setCurrentIndex(panel.arrangement_combo.findData("stacked"))
    panel.plot_regions(all_regions=True)
    assert panel._plotted == [(-5.0, 5.0), (15.0, 25.0)]
    assert sorted(t.get_text() for t in rig.ax.texts) == ["25 to 15 ppm", "5 to -5 ppm"]
    panel.arrangement_combo.setCurrentIndex(panel.arrangement_combo.findData("superimposed"))
    assert [t.get_text() for t in rig.ax.get_legend().get_texts()] == ["5 to -5 ppm", "25 to 15 ppm"]
    panel.scale_each_check.setChecked(True)
    assert "each region" in rig.ax.get_ylabel()


def test_selecting_several_regions_plots_just_those(qapp):
    rig = _Rig()
    panel = rig.panel
    for region in (A, B, (35.0, 30.0)):
        panel.add_region(*region)
    panel.regions_list.clearSelection()
    panel.regions_list.item(0).setSelected(True)
    panel.regions_list.item(2).setSelected(True)
    panel.plot_regions()
    assert panel._plotted == [(-5.0, 5.0), (30.0, 35.0)]


def test_keep_keeps_one_spectrum_per_plotted_region(qapp):
    rig = _Rig()
    panel = rig.panel
    panel.add_region(*A)
    panel.add_region(*B)
    panel.plot_regions(all_regions=True)
    first = panel.keep_pattern()
    assert first is not None and len(rig.kept) == 2
    labels = [label for _s, label in rig.kept]
    assert len(set(labels)) == 2 and all(" ppm" in label for label in labels)
    assert all(s.metadata.nucleus == "1H" for s, _l in rig.kept)
    # a single plotted region keeps a single spectrum
    rig.kept.clear()
    panel.regions_list.clearSelection()
    panel.regions_list.item(1).setSelected(True)
    panel.plot_regions()
    panel.keep_pattern()
    assert len(rig.kept) == 1 and "25 to 15" in rig.kept[0][1]


def test_remove_selected_and_clear_regions(qapp):
    rig = _Rig()
    panel = rig.panel
    panel.add_region(*A)
    panel.add_region(*B)
    panel.plot_regions(all_regions=True)
    panel.regions_list.clearSelection()
    panel.regions_list.item(0).setSelected(True)
    panel.remove_selected_regions()
    assert panel.regions() == [(15.0, 25.0)]
    assert list(panel._results) == [(15.0, 25.0)] and panel._plotted == [(15.0, 25.0)]
    panel.clear_regions()
    assert panel.regions() == [] and panel.series is None and panel.regions_list.count() == 0


def test_typed_region_flow_still_works_and_joins_the_list(qapp):
    rig = _Rig()
    panel = rig.panel
    panel.set_region(5, -5)
    series = panel.integrate()
    assert series is not None and len(series["y"]) == len(rig.spectra)
    assert panel.regions() == [(-5.0, 5.0)]
    # integrating again does not add it twice
    panel.integrate()
    assert panel.regions() == [(-5.0, 5.0)]
    panel.set_region(25, 15)
    assert panel.add_typed_region() is True
    assert len(panel.regions()) == 2


def test_integrate_without_any_region_says_so(qapp, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    shown = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: shown.append(a[2]))
    rig = _Rig()
    assert rig.panel.integrate() is None
    assert shown and "region" in shown[0]


def test_window_drag_on_compare_adds_regions_to_the_list(qapp):
    from test_gui import _load_spectrum_directly
    window = MainWindow()
    for s in _two_peak_spectra():
        _load_spectrum_directly(window, s)
    panel = window.offset_tracker_panel
    panel.refresh()
    panel.tick_all_button.click()
    panel.read_offsets()
    panel.armed = True
    window.compare_canvas.regionSelected.emit(5.0, -5.0)
    window.compare_canvas.regionSelected.emit(25.0, 15.0)
    assert panel.regions() == [(-5.0, 5.0), (15.0, 25.0)]
    n = len(window.session)
    panel.plot_regions(all_regions=True)
    panel.keep_pattern()
    assert len(window.session) == n + 2
