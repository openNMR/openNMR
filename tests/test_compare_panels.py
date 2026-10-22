"""2026-10-22: Compare 1D spectra as a row / column / grid of separate panels
(notebook ``compare_panels`` + the canvas + Compare ▸ Layout ▸ Panels).
Synthetic data only."""
import numpy as np
import pytest
import matplotlib

matplotlib.use("Agg")

from PySide6.QtWidgets import QApplication

from opennmr.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
from opennmr.notebook import compare_panels as cp
from opennmr.notebook import fit_models as fm


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _spec(centre, n=400):
    ppm = np.linspace(0, 10, n)
    data = np.exp(-((ppm - centre) ** 2) * 5)
    return NMRSpectrum(data=data, axes=[NMRAxis(kind="frequency", ppm=ppm, hz=ppm * 400.0)],
                       metadata=NMRMetadata(vendor="bruker", nucleus="1H", sfo1=400.0))


def _kept_fit(centre=5.0):
    s = _spec(centre)
    ppm = s.axes[0].ppm
    comp = [0.6 * np.exp(-((ppm - centre + 0.3) ** 2) * 5), 0.4 * np.exp(-((ppm - centre - 0.3) ** 2) * 5)]
    model = fm.make_fit_model("2 sites", "isotropic", ppm, s.data, comp[0] + comp[1],
                              components=comp, labels=["A", "B"], n_free=6)
    s.metadata.raw["fit_model"] = fm.encode_fit_model(model)
    return s


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    import matplotlib.pyplot as plt
    plt.close("all")


# --- geometry ---------------------------------------------------------------

@pytest.mark.parametrize("n,layout,ncols,expected", [
    (3, "row", None, (1, 3)), (3, "column", None, (3, 1)), (4, "grid", None, (2, 2)),
    (5, "grid", None, (2, 3)), (5, "grid", 2, (3, 2)), (2, "grid", 9, (1, 2)), (1, "grid", None, (1, 1)),
])
def test_grid_shape(n, layout, ncols, expected):
    assert cp.grid_shape(n, layout, ncols) == expected


def test_bad_layout_is_refused():
    with pytest.raises(ValueError):
        cp.grid_shape(3, "diagonal")


def test_panel_groups_follow_panel_numbers():
    assert cp.panel_groups([1, 1, 2, 3]) == [[0, 1], [2], [3]]
    assert cp.panel_groups([3, 1, 3]) == [[1], [0, 2]]


# --- the plot ---------------------------------------------------------------

def test_row_has_one_axes_per_panel():
    fig, axes, link = cp.plot_compare_panels([_spec(2), _spec(5), _spec(8)], layout="row")
    assert len(axes) == 3
    tops = {round(a.get_position().y0, 3) for a in axes}
    assert len(tops) == 1


def test_grid_hides_unused_cells():
    fig, axes, _ = cp.plot_compare_panels([_spec(c) for c in (1, 2, 3)], layout="grid", ncols=2)
    assert len(axes) == 3 and len(fig.axes) == 4
    assert sum(a.get_visible() for a in fig.axes) == 3


def test_a_panel_can_hold_many_spectra():
    fig, axes, _ = cp.plot_compare_panels([[_spec(2), _spec(5)], _spec(8)], layout="row")
    assert len(axes[0].lines) == 2 and len(axes[1].lines) == 1
    assert axes[0].get_legend() is not None and axes[1].get_legend() is None


def test_x_axis_is_in_nmr_direction_and_shared_range():
    _, axes, _ = cp.plot_compare_panels([_spec(2), _spec(8)], layout="row")
    for ax in axes:
        lo, hi = ax.get_xlim()
        assert lo > hi


def test_each_panel_autoscales_its_own_intensity():
    weak, strong = _spec(3), _spec(7)
    strong.data = strong.data * 100
    _, axes, _ = cp.plot_compare_panels([weak, strong], layout="row")
    assert axes[1].get_ylim()[1] > 20 * axes[0].get_ylim()[1]


def test_linked_intensity_shares_one_range():
    weak, strong = _spec(3), _spec(7)
    strong.data = strong.data * 100
    _, axes, _ = cp.plot_compare_panels([weak, strong], layout="row", link_y=True)
    assert axes[0].get_ylim() == pytest.approx(axes[1].get_ylim())


def test_normalize_scales_every_spectrum_to_one():
    big = _spec(5)
    big.data = big.data * 50
    _, axes, _ = cp.plot_compare_panels([big, _spec(2)], layout="row", normalize=True)
    for ax in axes:
        assert np.max(ax.lines[0].get_ydata()) == pytest.approx(1.0, rel=0.05)


def test_empty_and_non_1d_input_are_refused():
    with pytest.raises(ValueError):
        cp.plot_compare_panels([])
    with pytest.raises(ValueError):
        cp.plot_compare_panels([[]])
    with pytest.raises(ValueError):
        cp.plot_compare_panels(["not a spectrum"])


def test_titles_and_labels_are_used():
    _, axes, _ = cp.plot_compare_panels([_spec(2), [_spec(3), _spec(4)]], titles=["one", "two"],
                                        labels=[["x"], ["p", "q"]])
    assert axes[0].get_title() == "one" and axes[1].get_title() == "two"
    assert [t.get_text() for t in axes[1].get_legend().get_texts()] == ["p", "q"]


def test_shift_axis_label_only_on_the_bottom_of_a_column():
    _, axes, _ = cp.plot_compare_panels([_spec(c) for c in (1, 2, 3)], layout="column")
    assert axes[0].get_xlabel() == "" and axes[1].get_xlabel() == "" and axes[2].get_xlabel() != ""


# --- linking ----------------------------------------------------------------

def test_linked_panels_zoom_together():
    _, axes, link = cp.plot_compare_panels([_spec(2), _spec(5), _spec(8)], layout="row")
    axes[0].set_xlim(7, 3)
    for ax in axes:
        assert ax.get_xlim() == pytest.approx((7, 3))


def test_unlinked_panels_zoom_alone():
    _, axes, link = cp.plot_compare_panels([_spec(2), _spec(5)], layout="row", link=False)
    before = axes[1].get_xlim()
    axes[0].set_xlim(7, 3)
    assert axes[1].get_xlim() == before


def test_link_can_be_switched_after_drawing_and_resyncs():
    _, axes, link = cp.plot_compare_panels([_spec(2), _spec(5)], layout="row", link=False)
    axes[0].set_xlim(6, 4)
    link.set(x=True)                       # joins the first panel's view
    assert axes[1].get_xlim() == pytest.approx((6, 4))
    axes[1].set_xlim(9, 1)
    assert axes[0].get_xlim() == pytest.approx((9, 1))
    link.set(x=False)
    axes[1].set_xlim(5, 4)
    assert axes[0].get_xlim() == pytest.approx((9, 1))


def test_y_link_is_independent_of_x_link():
    _, axes, link = cp.plot_compare_panels([_spec(2), _spec(5)], layout="row", link_y=True)
    axes[0].set_ylim(-1, 3)
    assert axes[1].get_ylim() == pytest.approx((-1, 3))
    link.set(y=False)
    axes[0].set_ylim(0, 1)
    assert axes[1].get_ylim() == pytest.approx((-1, 3))


def test_a_single_panel_never_links():
    _, axes, link = cp.plot_compare_panels([_spec(2)], layout="row")
    assert link.x is False


# --- kept fits --------------------------------------------------------------

def test_a_kept_fit_draws_its_fit_and_components():
    plain = _spec(2)
    _, axes, _ = cp.plot_compare_panels([_kept_fit(5.0), plain], layout="row")
    assert len(axes[0].lines) >= 2         # the data and the total fit
    assert len(axes[1].lines) == 1 and not axes[1].collections


def test_components_are_drawn_when_the_view_shows_them():
    from opennmr.notebook.fit_models import fit_view
    shown = _kept_fit()
    shown.metadata.raw["fit_model_view"] = fit_view({"style": {"show": "all", "fill": False}})
    hidden = _kept_fit()
    hidden.metadata.raw["fit_model_view"] = fit_view({"style": {"show": "none"}})
    _, a, _ = cp.plot_compare_panels([shown], layout="row")
    _, b, _ = cp.plot_compare_panels([hidden], layout="row")
    assert len(a[0].lines) == len(b[0].lines) + 2


def test_fit_view_residual_is_drawn_when_asked():
    from opennmr.notebook.fit_models import fit_view
    without = _kept_fit()
    with_res = _kept_fit()
    with_res.metadata.raw["fit_model_view"] = fit_view({"show_residual": True})
    without.metadata.raw["fit_model_view"] = fit_view({"show_residual": False})
    _, a, _ = cp.plot_compare_panels([without], layout="row")
    _, b, _ = cp.plot_compare_panels([with_res], layout="row")
    assert len(b[0].lines) > len(a[0].lines)


def test_fits_and_plain_spectra_share_one_panel():
    _, axes, _ = cp.plot_compare_panels([[_kept_fit(), _spec(7)]], layout="row")
    assert len(axes[0].lines) >= 3          # fit data, total fit and the plain spectrum


# --- canvas -----------------------------------------------------------------

def test_canvas_draws_panels_and_links(qapp):
    from opennmr.gui.widgets.canvas import MplCanvas
    canvas = MplCanvas()
    spectra = [_spec(2), _spec(5), _spec(8)]
    canvas.show_compare_panels(spectra, [[0, 1], [2]], layout="row", legend_labels=["a", "b", "c"])
    assert len(canvas._spectrum_panels) == 2 and canvas.spectrum_panels_linked() == (True, False)
    a, b = canvas._spectrum_panels
    a.set_xlim(8, 2)
    assert b.get_xlim() == pytest.approx((8, 2))
    canvas.set_spectrum_panels_linked(False)
    a.set_xlim(6, 4)
    assert b.get_xlim() == pytest.approx((8, 2))
    assert canvas.spectrum_panels_linked() == (False, False)


def test_canvas_names_traces_and_follows_the_legend_switch(qapp):
    from opennmr.gui.widgets.canvas import MplCanvas
    canvas = MplCanvas()
    canvas.show_compare_panels([_spec(2), _spec(5)], [[0, 1]], layout="row", legend_labels=["a", "b"])
    ax = canvas._spectrum_panels[0]
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["a", "b"]
    canvas.set_legend_visible(False)
    assert ax.get_legend() is None


def test_canvas_redraw_keeps_panels_and_link_state(qapp):
    from opennmr.gui.widgets.canvas import MplCanvas
    canvas = MplCanvas()
    canvas.show_compare_panels([_spec(2), _spec(5)], [[0], [1]], layout="column")
    canvas.set_spectrum_panels_linked(False)
    canvas._redraw_active_series_view()
    assert len(canvas._spectrum_panels) == 2 and canvas.spectrum_panels_linked() == (False, False)


def test_canvas_leaving_panels_forgets_them(qapp):
    from opennmr.gui.widgets.canvas import MplCanvas
    canvas = MplCanvas()
    canvas.show_compare_panels([_spec(2), _spec(5)], [[0], [1]])
    canvas.show_compare([_spec(2), _spec(5)], mode="overlay")
    assert canvas._spectrum_panels == [] and canvas._spectrum_panel_link is None
    canvas._redraw_active_series_view()                    # redraws as a plain compare
    assert canvas._spectrum_panels == []


def test_canvas_crosshair_spans_the_panels(qapp):
    from opennmr.gui.widgets.canvas import MplCanvas
    canvas = MplCanvas()
    canvas.show_compare_panels([_spec(2), _spec(5)], [[0], [1]])
    canvas._ensure_crosshair()
    assert canvas._crosshair is not None


def test_panels_export_at_the_requested_dpi(qapp):
    import io
    from PIL import Image
    from opennmr.gui.widgets.canvas import MplCanvas
    canvas = MplCanvas()
    canvas.show_compare_panels([_spec(2), _spec(5)], [[0], [1]])
    canvas.set_export_settings(dpi=300, fig_width=4.0, fig_height=2.0)
    buffer = io.BytesIO()
    canvas.export_full_resolution(lambda: canvas.figure.savefig(buffer, format="png"))
    assert Image.open(io.BytesIO(buffer.getvalue())).size == (1200, 600)


# --- GUI --------------------------------------------------------------------

@pytest.fixture
def controls(qapp):
    from opennmr.gui.compare_panels_controls import ComparePanelsControls
    c = ComparePanelsControls()
    c.set_spectra(["a", "b", "c"], ["A", "B", "C"])
    return c


def test_controls_default_to_one_panel_per_spectrum(controls):
    assert controls.assignment() == [1, 2, 3] and controls.groups() == [[0], [1], [2]]
    assert controls.linked() and not controls.linked_y() and controls.columns() is None


def test_controls_group_spectra_into_a_panel(controls):
    seen = []
    controls.changed.connect(lambda: seen.append(1))
    controls.table.cellWidget(1, 1).setValue(1)
    assert controls.groups() == [[0, 1], [2]] and seen


def test_controls_buttons(controls):
    controls.all_in_one()
    assert controls.groups() == [[0, 1, 2]]
    controls.one_per_panel()
    assert controls.groups() == [[0], [1], [2]]


def test_controls_remember_assignments_when_rows_change(controls):
    controls.table.cellWidget(2, 1).setValue(1)
    controls.set_spectra(["a", "c"], ["A", "C"])
    assert controls.groups() == [[0, 1]]
    controls.set_spectra(["a", "b", "c"], ["A", "B", "C"])
    assert controls.assignment()[0] == controls.assignment()[2]


def test_controls_link_signal_does_not_redraw(controls):
    changes, links = [], []
    controls.changed.connect(lambda: changes.append(1))
    controls.link_toggled.connect(lambda x, y: links.append((x, y)))
    controls.link_check.setChecked(False)
    controls.link_y_check.setChecked(True)
    assert links == [(False, False), (False, True)] and not changes


@pytest.fixture
def window(qapp):
    from opennmr.gui.main_window import MainWindow
    w = MainWindow()
    w.resize(1300, 850)
    yield w
    w.close()


def _window_with_three_compared(window, tmp_path):
    from test_gui import _make_1d_dataset, _select_rows
    for name in "abc":
        window._load(_make_1d_dataset(tmp_path, name=name), vendor="bruker")
    _select_rows(window, [0, 1, 2])
    assert window.compare_list.count() >= 3


def test_layout_menu_offers_the_panel_layouts(window):
    items = [window.layout_mode_combo.itemText(i) for i in range(window.layout_mode_combo.count())]
    assert items[:3] == ["Stacked", "Superimposed", "Heatmap"]
    assert {"Panels: row", "Panels: column", "Panels: grid"} <= set(items)


def test_choosing_panels_draws_them_and_shows_the_controls(window, tmp_path):
    _window_with_three_compared(window, tmp_path)
    window.layout_mode_combo.setCurrentText("Panels: row")
    window._refresh_compare_view(force_render=True)
    assert len(window.compare_canvas._spectrum_panels) == 3
    assert not window.compare_panels_controls.isHidden()


def test_grouping_in_the_table_reaches_the_plot(window, tmp_path):
    _window_with_three_compared(window, tmp_path)
    window.layout_mode_combo.setCurrentText("Panels: grid")
    window._refresh_compare_view(force_render=True)
    window.compare_panels_controls.all_in_one()
    assert len(window.compare_canvas._spectrum_panels) == 1
    assert len(window.compare_canvas._spectrum_panels[0].lines) == 3


def test_the_link_switch_reaches_the_plot_without_redrawing(window, tmp_path):
    _window_with_three_compared(window, tmp_path)
    window.layout_mode_combo.setCurrentText("Panels: row")
    window._refresh_compare_view(force_render=True)
    axes = list(window.compare_canvas._spectrum_panels)
    window.compare_panels_controls.link_check.setChecked(False)
    assert window.compare_canvas.spectrum_panels_linked()[0] is False
    assert list(window.compare_canvas._spectrum_panels) == axes       # same Axes: no redraw


def test_going_back_to_stacked_hides_the_panel_controls(window, tmp_path):
    _window_with_three_compared(window, tmp_path)
    window.layout_mode_combo.setCurrentText("Panels: row")
    window._refresh_compare_view(force_render=True)
    window.layout_mode_combo.setCurrentText("Stacked")
    window._refresh_compare_view(force_render=True)
    assert window.compare_panels_controls.isHidden()
    assert window.compare_canvas._spectrum_panels == []


# ======================================================================
# 2026-10-22: an axis label on every panel for different nuclei, and per-panel
# axis options (label text, tick direction, sizes ...)
# ======================================================================

def _nuc_spec(nucleus, centre=5.0):
    s = _spec(centre)
    s.metadata.nucleus = nucleus
    return s


def _labels(axes):
    return [ax.get_xlabel() for ax in axes]


def test_different_nuclei_stacked_each_keep_their_own_axis_label():
    fig, axes, _ = cp.plot_compare_panels([_nuc_spec("1H"), _nuc_spec("23Na"), _nuc_spec("13C")],
                                          layout="column")
    labels = _labels(axes)
    assert all(labels) and len(set(labels)) == 3


def test_same_nucleus_stacked_keeps_one_shared_label():
    fig, axes, _ = cp.plot_compare_panels([_nuc_spec("1H"), _nuc_spec("1H")], layout="column")
    assert _labels(axes)[0] == "" and _labels(axes)[1] != ""


def test_different_nuclei_in_a_grid_all_labelled():
    fig, axes, _ = cp.plot_compare_panels([_nuc_spec("1H"), _nuc_spec("23Na"), _nuc_spec("13C")],
                                          layout="grid", ncols=2)
    assert all(_labels(axes))


@pytest.mark.parametrize("mode,expected", [("all", [True, True]), ("bottom", [False, True])])
def test_axis_labels_mode_overrides_the_automatic_choice(mode, expected):
    fig, axes, _ = cp.plot_compare_panels([_nuc_spec("1H"), _nuc_spec("1H")], layout="column",
                                          axis_labels=mode)
    assert [bool(t) for t in _labels(axes)] == expected
    fig, axes, _ = cp.plot_compare_panels([_nuc_spec("1H"), _nuc_spec("23Na")], layout="column",
                                          axis_labels="bottom")
    assert [bool(t) for t in _labels(axes)] == [False, True]


def test_bad_axis_labels_mode_is_refused():
    with pytest.raises(ValueError):
        cp.plot_compare_panels([_spec(3)], axis_labels="sometimes")


def _tick_dirs(ax):
    return ax.xaxis.get_major_ticks()[0].tick1line.get_markeredgewidth(), ax.xaxis._major_tick_kw["tickdir"]


def test_axis_styles_change_one_panel_and_all_panels():
    fig, axes, _ = cp.plot_compare_panels(
        [_spec(3), _spec(5), _spec(7)], layout="column",
        axis_styles={"all": {"tick_direction": "out", "tick_labelsize": 9},
                     1: {"xlabel": r"$\delta$ custom", "tick_direction": "inout", "tick_length": 9,
                         "xlabel_fontsize": 15}})
    assert [ax.xaxis._major_tick_kw["tickdir"] for ax in axes] == ["out", "inout", "out"]
    assert axes[1].get_xlabel() == r"$\delta$ custom"
    assert axes[1].xaxis.label.get_fontsize() == 15
    assert axes[1].xaxis._major_tick_kw["size"] == 9
    assert axes[0].xaxis._major_tick_kw["labelsize"] == 9


def test_axis_styles_as_a_list_hide_labels_ticks_and_minor_ticks():
    fig, axes, _ = cp.plot_compare_panels(
        [_spec(3), _spec(5)], layout="column", axis_labels="all",
        axis_styles=[{"show_xlabel": False, "show_tick_labels": False, "minor_ticks": False}, None])
    assert axes[0].get_xlabel() == "" and axes[1].get_xlabel() != ""
    assert not axes[0].xaxis._major_tick_kw.get("label1On", True)
    assert not axes[0].xaxis.get_minor_locator().__class__.__name__ == "AutoMinorLocator"
    assert axes[1].xaxis.get_minor_locator().__class__.__name__ == "AutoMinorLocator"


def test_axis_styles_num_ticks_and_spine_width():
    fig, axes, _ = cp.plot_compare_panels([_spec(3)], axis_styles={0: {"num_ticks": 3, "spine_width": 2.5}})
    assert axes[0].spines["bottom"].get_linewidth() == 2.5
    assert len(axes[0].get_xticks()) <= 6


def test_unknown_axis_option_or_direction_is_refused():
    with pytest.raises(ValueError):
        cp.plot_compare_panels([_spec(3)], axis_styles={0: {"colour": "red"}})
    with pytest.raises(ValueError):
        cp.plot_compare_panels([_spec(3)], axis_styles={0: {"tick_direction": "sideways"}})


def test_gui_each_nucleus_panel_labelled_and_styles_survive_canvas_defaults(window, tmp_path):
    _window_with_three_compared(window, tmp_path)
    for i, nuc in enumerate(("1H", "23Na", "13C")):
        window.session.spectra[i].metadata.nucleus = nuc
    window.layout_mode_combo.setCurrentText("Panels: column")
    window._refresh_compare_view(force_render=True)
    axes = window.compare_canvas._spectrum_panels
    assert all(ax.get_xlabel() for ax in axes)
    c = window.compare_panels_controls
    c.axis_target_combo.setCurrentIndex(c.axis_target_combo.findData(1))
    c.axis_direction_combo.setCurrentIndex(c.axis_direction_combo.findData("out"))
    c.axis_label_edit.setText("middle")
    c.axis_label_edit.editingFinished.emit()
    window._refresh_compare_view(force_render=True)
    axes = window.compare_canvas._spectrum_panels
    assert axes[1].xaxis._major_tick_kw["tickdir"] == "out" and axes[0].xaxis._major_tick_kw["tickdir"] != "out"
    assert axes[1].get_xlabel() == "middle"
    c.axis_labels_combo.setCurrentIndex(c.axis_labels_combo.findData("bottom"))
    window._refresh_compare_view(force_render=True)
    assert [bool(ax.get_xlabel()) for ax in window.compare_canvas._spectrum_panels][-1]
    c.reset_axes()
    window._refresh_compare_view(force_render=True)
    assert window.compare_canvas._spectrum_panels[1].xaxis._major_tick_kw["tickdir"] != "out"
