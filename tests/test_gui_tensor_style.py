"""2026-10-01 (round 3): a tensor plot style of its own (right-click ▸
Tensor Plot Style…, a "Tensor pictures" per-type default), Plot Style's
axis-label choice on Simulate's figures, and colour-blind-friendly
default colours in Simulate."""
import json

import matplotlib.colors as mcolors
import numpy as np
import pytest
from PySide6.QtWidgets import QMessageBox

from opennmr.gui.main_window import MainWindow
from opennmr.gui.widgets.canvas import MplCanvas
from opennmr.notebook import tensor_views as tv
from opennmr.plotting import accessibility

from test_gui import qapp  # noqa: F401 -- fixture
from test_gui_simulation_plots import _csa_two_sites, _menu


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    for name in ("warning", "information", "critical"):
        monkeypatch.setattr(QMessageBox, name, lambda *a, **k: None)
    monkeypatch.setattr(MainWindow, "_FIGURE_STYLES_PATH", str(tmp_path / "figure_styles.json"))


def _tensor_window():
    window = MainWindow()
    window.sim_csa_nucleus_edit.setText("13C")
    window.sim_csa_aniso_spin.setValue(-80.0)
    assert window.sim_csa_tensor_controls.show_tensors()
    return window


def _surface_cmap(canvas):
    return canvas.figure.tensor_mappable.cmap.name


def _line_colours(canvas):
    colours = []
    for ax in canvas.figure.axes:
        for line in ax.lines:
            if not line.get_label().startswith("_"):
                colours.append(mcolors.to_hex(line.get_color()))
    return list(dict.fromkeys(colours))


# ---------------------------------------------------------------------------
# Tensor plot style
# ---------------------------------------------------------------------------

def test_tensor_pictures_are_their_own_plot_kind_with_a_style_menu(qapp, monkeypatch):  # noqa: F811
    window = _tensor_window()
    canvas = window.simulation_canvas
    assert canvas._plot_kind == "tensor"
    assert ("Tensor pictures", "tensor") in MainWindow._FIGURE_STYLE_TYPES
    assert MainWindow._FIGURE_STYLE_FALLBACK["tensor"] == "2d"
    actions = _menu(canvas, monkeypatch)
    assert "Tensor Plot Style…" in actions and "Plot Style…" in actions
    # Not offered on other plots.
    _csa_two_sites(window)
    assert "Tensor Plot Style…" not in _menu(canvas, monkeypatch)


def test_tensor_style_dialog_changes_the_picture_live(qapp, monkeypatch):  # noqa: F811
    window = _tensor_window()
    canvas = window.simulation_canvas
    window._open_tensor_style()
    dialog = window._tensor_style_dialog
    dialog.colormap_combo.setCurrentIndex(dialog.colormap_combo.findData("cividis"))
    assert _surface_cmap(canvas) == "cividis"
    dialog.reverse_check.setChecked(True)
    assert _surface_cmap(canvas) == "cividis_r"
    n_axes = len(canvas.figure.axes)
    dialog.colorbar_check.setChecked(False)
    assert len(canvas.figure.axes) == n_axes - 1          # the colour bar's Axes went
    dialog.elev_spin.setValue(60.0)
    ax3d = [ax for ax in canvas.figure.axes if ax.name == "3d"][0]
    assert ax3d.elev == pytest.approx(60.0)
    # A view turned to by hand survives a colour change.
    ax3d.view_init(elev=10.0, azim=100.0)
    dialog.colormap_combo.setCurrentIndex(dialog.colormap_combo.findData("magma"))
    ax3d = [ax for ax in canvas.figure.axes if ax.name == "3d"][0]
    assert (ax3d.elev, ax3d.azim) == pytest.approx((10.0, 100.0))
    # Colour-blind-friendly maps are marked.
    assert "✓" in dialog.colormap_combo.itemText(dialog.colormap_combo.findData("cividis"))


def test_tensor_style_colours_reach_the_ellipsoid_picture(qapp):  # noqa: F811
    window = _tensor_window()
    controls = window.sim_csa_tensor_controls
    controls.style_combo.setCurrentIndex(controls.style_combo.findData("ellipsoid"))
    window.simulation_canvas.set_style(tensor_style={"positive_colour": "#009e73",
                                                     "negative_colour": "#cc79a7"})
    assert controls.show_tensors()
    colours = _line_colours(window.simulation_canvas) + [
        mcolors.to_hex(line.get_color()) for ax in window.simulation_canvas.figure.axes
        for line in ax.lines]
    assert "#009e73" in colours or "#cc79a7" in colours


def test_tensor_style_saved_as_the_tensor_default(qapp, tmp_path):  # noqa: F811
    window = _tensor_window()
    window._save_tensor_style_default({"colormap": "PuOr_r", "colorbar": False})
    saved = json.loads((tmp_path / "figure_styles.json").read_text())
    assert saved["tensor"]["tensor_style"] == {"colormap": "PuOr_r", "colorbar": False}
    # A new window opening a tensor picture takes it.
    fresh = _tensor_window()
    assert fresh.simulation_canvas.style_settings()["tensor_style"]["colormap"] == "PuOr_r"
    assert _surface_cmap(fresh.simulation_canvas) == "PuOr_r"


def test_notebook_tensor_style_matches_the_gui_keys():
    site = tv.ch.Site("13C", delta_iso=50.0)
    site.add_csa(aniso=-60.0, asym=0.3)
    fig, _notes = tv.plot_tensors(site, style="surface", tensor_style={"colormap": "magma",
                                                                       "colorbar": False})
    assert fig.tensor_mappable.cmap.name == "magma"
    assert len(fig.axes) == 1
    assert tv.tensor_style({"bogus": 1})["colormap"] is None


# ---------------------------------------------------------------------------
# Axis-label style on Simulate's figures
# ---------------------------------------------------------------------------

def test_axis_label_style_reaches_tensor_and_simulation_figures(qapp):  # noqa: F811
    window = _tensor_window()
    canvas = window.simulation_canvas
    canvas.set_style(axis_label_style="slash")
    ax3d = [ax for ax in canvas.figure.axes if ax.name == "3d"][0]
    # Coordinates are quantity symbols, so italic (2026-10-17).
    assert ax3d.get_xlabel() == "$x$ / ppm" and ax3d.get_zlabel() == "$z$ / ppm"
    bar = [ax for ax in canvas.figure.axes if ax.name != "3d"][0]
    assert bar.get_ylabel().endswith("/ ppm")
    canvas.set_style(axis_label_style="brackets")
    ax3d = [ax for ax in canvas.figure.axes if ax.name == "3d"][0]
    assert ax3d.get_xlabel() == "$x$ (ppm)"
    # The Bloch plot writes "Time / μs": brackets puts it in brackets,
    # and a qualifier that is not a unit is left alone.
    window._run_bloch_simulation()
    labels = [ax.get_xlabel() for ax in canvas.figure.axes] + [ax.get_ylabel() for ax in canvas.figure.axes]
    assert "Time (μs)" in labels and "Amplitude (normalised)" in labels
    canvas.set_style(axis_label_style="slash")
    labels = [ax.get_ylabel() for ax in canvas.figure.axes]
    assert "Amplitude (normalised)" in labels


# ---------------------------------------------------------------------------
# Colour-blind-friendly defaults in Simulate
# ---------------------------------------------------------------------------

def test_simulate_default_colours_are_colourblind_friendly(qapp):  # noqa: F811
    window = MainWindow()
    canvas = window.simulation_canvas
    window._run_diffusion_simulation()                          # eight D values
    colours = _line_colours(canvas)
    assert len(colours) >= 6 and accessibility.is_colourblind_safe(colours)
    window._run_bloch_simulation()
    colours = _line_colours(canvas)
    assert "#d55e00" in colours                                  # the phase trace
    assert accessibility.is_colourblind_safe(colours)
    window.sim_csa_nucleus_edit.setText("13C")
    window.sim_csa_orientations_spin.setValue(300)
    window.sim_csa_points_spin.setValue(512)
    for shift in (20.0, 60.0, 100.0, 140.0, 180.0):
        window.sim_csa_diso_spin.setValue(shift)
        if shift != 180.0:
            window.sim_csa_sites.add_site()
    window._on_simulate_csa_clicked()
    colours = _line_colours(canvas)
    assert len(colours) >= 6 and accessibility.is_colourblind_safe(colours)


def test_default_palette_choice_is_measured():
    for n in range(2, 10):
        name = accessibility.default_palette(n)
        assert accessibility.max_safe_series(name) >= n
    assert accessibility.max_safe_series(accessibility.default_palette(5, ordered=True)) >= 5


# ---------------------------------------------------------------------------
# Round 4 (2026-10-01): Plot Style's colour maps and grid on tensor pictures
# ---------------------------------------------------------------------------

def _ax3d(canvas):
    return [ax for ax in canvas.figure.axes if ax.name == "3d"][0]


def _plot_style_on_tensor(window):
    window.canvas_tabs.setCurrentWidget(window.simulation_canvas)
    return window.simulation_canvas


def test_label_style_change_keeps_the_tensor_gridlines(qapp):  # noqa: F811
    # User report: switching "value (units)" to "value / units" switched
    # the 3D gridlines off (the panel re-sent grid=False with every edit).
    window = _tensor_window()
    canvas = _plot_style_on_tensor(window)
    assert canvas._tensor_style.get("grid", True) is True
    assert _ax3d(canvas)._draw_grid
    combo = window.output_axis_label_style_combo
    combo.setCurrentIndex(combo.findData("slash"))
    ax = _ax3d(canvas)
    assert ax._draw_grid
    assert ax.get_xlabel() == "$x$ / ppm"


def test_tensor_gridlines_are_optional(qapp):  # noqa: F811
    window = _tensor_window()
    canvas = _plot_style_on_tensor(window)
    window._open_tensor_style()
    dialog = window._tensor_style_dialog
    dialog.grid_check.setChecked(False)
    assert not _ax3d(canvas)._draw_grid
    dialog.grid_check.setChecked(True)
    assert _ax3d(canvas)._draw_grid
    # Plot Style's "Show grid" box drives it too, and the dialog follows.
    window.output_grid_checkbox.setChecked(not window.output_grid_checkbox.isChecked())
    expected = window.output_grid_checkbox.isChecked()
    assert _ax3d(canvas)._draw_grid is expected
    assert dialog.grid_check.isChecked() is expected
    # Notebook: the same key.
    import matplotlib.pyplot as plt
    fig = plt.figure()
    tv.plot_tensor_ellipsoids(window._csa_tensor_site()[0], figure=fig, tensor_style={"grid": False})
    assert not [a for a in fig.axes if a.name == "3d"][0]._draw_grid
    plt.close(fig)


def test_series_colour_map_reaches_the_tensor_surface(qapp):  # noqa: F811
    # User report: 'Changing "series colour map" did nothing for the tensor.'
    window = _tensor_window()
    canvas = _plot_style_on_tensor(window)
    window._open_tensor_style()
    dialog = window._tensor_style_dialog
    window._select_series_colormap("magma")
    window._on_series_colormap_changed()
    assert _surface_cmap(canvas) == "magma"
    assert dialog.colormap_combo.currentData() == "magma"
    # A scale the dialog's list lacks is added to it, not dropped on the next edit.
    window._select_series_colormap("YlGnBu")
    window._on_series_colormap_changed()
    assert _surface_cmap(canvas) == "YlGnBu"
    assert dialog.colormap_combo.currentData() == "YlGnBu"
    dialog.reverse_check.setChecked(True)
    assert _surface_cmap(canvas) == "YlGnBu_r"
    dialog.reverse_check.setChecked(False)
    # A qualitative palette is categories, not a scale: the surface keeps its map.
    window._select_series_colormap("okabe_ito")
    window._on_series_colormap_changed()
    assert _surface_cmap(canvas) == "YlGnBu"
    # The dialog's own choice is the last change, so it wins.
    dialog.colormap_combo.setCurrentIndex(dialog.colormap_combo.findData("cividis"))
    assert _surface_cmap(canvas) == "cividis"
    # Map colour scale changed while the tensor is shown: the last change wins again.
    combo = window.output_map_colormap_combo
    combo.setCurrentIndex(combo.findData("plasma"))
    assert _surface_cmap(canvas) == "plasma"


def test_unchanged_plot_style_settings_leave_the_tensor_style_alone(qapp):  # noqa: F811
    window = _tensor_window()
    canvas = _plot_style_on_tensor(window)
    canvas.set_style(tensor_style={"colormap": "cividis", "grid": False})
    style = canvas.style_settings()
    # The panel re-sends everything on any edit: nothing changes.
    canvas.set_style(compare_colormap=style["compare_colormap"], map_colormap=style["map_colormap"] or "",
                     grid=style["grid"], axis_label_style="slash")
    assert _surface_cmap(canvas) == "cividis"
    assert not _ax3d(canvas)._draw_grid


def _text_sizes(fig):
    """(label sizes, tick-label sizes) over every axis of ``fig``."""
    labels, ticks = set(), set()
    for ax in fig.axes:
        axes = [ax.xaxis, ax.yaxis] + ([ax.zaxis] if ax.name == "3d" else [])
        for axis in axes:
            shown = axis.get_ticklabels()
            if axis.label.get_text():
                labels.add(round(axis.label.get_fontsize(), 3))
                ticks.update(round(t.get_fontsize(), 3) for t in shown)
    return labels, ticks


@pytest.mark.parametrize("size", [9.0, 11.0, 16.0])
def test_tensor_tick_labels_match_the_axis_labels(qapp, size):  # noqa: F811
    # User request: "The axis ticklabels and the axis labels need to have
    # the same size for the tensor plot".
    window = _tensor_window()
    canvas = window.simulation_canvas
    canvas.set_style(labelsize=size)
    for picture in ("surface", "ellipsoid"):
        if picture == "ellipsoid":
            window.sim_csa_tensor_controls.style_combo.setCurrentIndex(
                window.sim_csa_tensor_controls.style_combo.findData("ellipsoid"))
            assert window.sim_csa_tensor_controls.show_tensors()
        labels, ticks = _text_sizes(canvas.figure)
        assert labels == {size} and ticks == {size}, picture
    canvas.figure.canvas.draw()          # and it still draws


def test_notebook_tensor_tick_labels_match_the_axis_labels():
    import matplotlib.pyplot as plt
    from opennmr.notebook import combined_hamiltonian as ch
    site = ch.Site("13C", delta_iso=0.0, label="C")
    site.add_csa(aniso=-80.0, asym=0.3)
    for style in ("surface", "ellipsoid"):
        fig = plt.figure()
        tv.plot_tensors(site, None, style=style, figure=fig, fontsize=12.0)
        labels, ticks = _text_sizes(fig)
        assert len(labels) == 1 and labels == ticks, style
        plt.close(fig)


# ---------------------------------------------------------------------------
# Round 6 (2026-10-01): Home after a zoom; ticks thinned so labels never overlap
# ---------------------------------------------------------------------------

def test_home_after_a_zoom_restores_the_undistorted_tensor(qapp):  # noqa: F811
    # User report: zooming in on the tensor plot and then hitting Home left
    # it "distorted (e.g., elongated along z)" -- Home restored x and y only.
    window = _tensor_window()
    canvas = window.simulation_canvas
    ax = _ax3d(canvas)
    limits = (ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d())
    canvas.note_natural_view()                   # what a toolbar zoom press does
    ax.set_xlim3d(-20, 20)
    ax.set_ylim3d(-20, 20)
    ax.set_zlim3d(-5, 5)
    ax.view_init(elev=40.0, azim=60.0)
    canvas.toolbar.home()
    ax = _ax3d(canvas)
    for now, then in zip((ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()), limits):
        assert tuple(now) == pytest.approx(tuple(then))
    aspect = np.asarray(ax._box_aspect)
    assert aspect == pytest.approx(np.full(3, aspect[0]))          # still a cube
    assert (ax.elev, ax.azim) == pytest.approx((40.0, 60.0))       # rotation kept


def _overlaps(fig, ax):
    """Tick labels of the 3D ``ax`` whose boxes overlap, any axis with any
    other. Measured while drawing: mplot3d places a tick label at its
    projected position only for the draw, so a box read afterwards is
    meaningless."""
    import matplotlib.text as mtext
    owner = {id(t): axis.axis_name for axis in (ax.xaxis, ax.yaxis, ax.zaxis)
             for t in axis.get_ticklabels()}
    boxes, draw = [], mtext.Text.draw

    def record(self, renderer):
        draw(self, renderer)
        if id(self) in owner and self.get_text() and self.get_visible():
            boxes.append((owner[id(self)], self.get_text(), self.get_window_extent(renderer)))
    mtext.Text.draw = record
    try:
        fig.canvas.draw()
    finally:
        mtext.Text.draw = draw
    assert boxes, "no tick labels drawn"
    return [(a[:2], b[:2]) for i, a in enumerate(boxes) for b in boxes[i + 1:]
            if a[2].overlaps(b[2])]


@pytest.mark.parametrize("inches", [(6.4, 4.8), (10.0, 7.0)])
def test_tensor_tick_labels_never_overlap_and_thin_with_font_size(qapp, inches):  # noqa: F811
    # User request: "the axis tick labels can end up overlapping each other
    # when the font is large ... decreasing the number of ticks".
    window = _tensor_window()
    canvas = window.simulation_canvas
    canvas.figure.set_size_inches(*inches)
    previous = None
    for size in (8.0, 11.0, 16.0, 24.0, 32.0):
        canvas.set_style(labelsize=size)
        canvas.canvas.draw()
        ax = _ax3d(canvas)
        assert _overlaps(canvas.figure, ax) == [], size
        counts = {n: len(getattr(ax, n + "axis").get_majorticklocs()) for n in "xyz"}
        assert all(v >= 1 for v in counts.values())
        if previous:
            assert all(counts[n] <= previous[n] for n in "xyz"), (size, counts, previous)
        previous = counts
    assert sum(previous.values()) < 21          # thinned from 7 per axis at 8 pt


def test_tensor_ticks_refit_after_rotating(qapp):  # noqa: F811
    window = _tensor_window()
    canvas = window.simulation_canvas
    canvas.set_style(labelsize=14.0)
    ax = _ax3d(canvas)
    side = len(ax.zaxis.get_majorticklocs())
    ax.view_init(elev=85.0, azim=35.0)           # looking down: z almost end-on
    canvas._refit_3d_ticks_after_release(type("E", (), {"inaxes": ax})())
    assert len(ax.zaxis.get_majorticklocs()) < side
    canvas.canvas.draw()
    assert _overlaps(canvas.figure, ax) == []


def test_notebook_fit_3d_ticks():
    import matplotlib.pyplot as plt
    from opennmr.notebook import combined_hamiltonian as ch
    site = ch.Site("13C", delta_iso=0.0, label="C")
    site.add_csa(aniso=-80.0, asym=0.3)
    counts = {}
    for size in (8.0, 30.0):
        fig = plt.figure()
        tv.plot_tensors(site, None, style="surface", figure=fig, fontsize=size)
        ax = [a for a in fig.axes if a.name == "3d"][0]
        counts[size] = tv.fit_3d_ticks(ax, size * 0.85)
        fig.canvas.draw()
        assert _overlaps(fig, ax) == []
        plt.close(fig)
    assert sum(counts[30.0].values()) < sum(counts[8.0].values())
