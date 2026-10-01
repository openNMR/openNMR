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
    assert ax3d.get_xlabel() == "x / ppm" and ax3d.get_zlabel() == "z / ppm"
    bar = [ax for ax in canvas.figure.axes if ax.name != "3d"][0]
    assert bar.get_ylabel().endswith("/ ppm")
    canvas.set_style(axis_label_style="brackets")
    ax3d = [ax for ax in canvas.figure.axes if ax.name == "3d"][0]
    assert ax3d.get_xlabel() == "x (ppm)"
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
