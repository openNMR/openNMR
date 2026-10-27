"""2026-10-27 (explicit user requests):

* "plot style options to be able to move the legend, or to make the
  legend multi-column" -- plotting.legend_placement (more positions,
  outside right, a custom place, columns), Plot Style ▸ Legend, and room
  made for a legend outside a 3D picture (make_room_for_legend).
* "options to control the colours and rendering of the balls for the
  atoms" -- tensor_style's atom/bond keys, Tensor Plot Style ▸ Atoms &
  Bonds.
* "show supercells (or change the limits in terms of fractional cell
  coords - e.g., x from -0.2 to +1.6)" -- structure_atoms(region="range"),
  Simulate ▸ DFT tensors ▸ Atoms shown: Fractional range / supercell.
* "the tensor view x, y, z tick labels can get in the way - padding
  options and the ability to just remove the labels entirely" --
  tensor_style's tick_labels / axis_labels / axes_box / tick_pad /
  label_pad, Tensor Plot Style ▸ Axes & Labels.

All data synthetic."""

import itertools

import matplotlib
import matplotlib.colors as mcolors
import numpy as np
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Path3DCollection, Poly3DCollection  # noqa: E402

from opennmr.notebook import dft_nmr as D  # noqa: E402
from opennmr.notebook import dft_structure as S  # noqa: E402
from opennmr.notebook import tensor_views as tv  # noqa: E402
from opennmr.plotting import LEGEND_POSITIONS, legend_placement, make_room_for_legend  # noqa: E402

from test_dft_nmr import EFG1, SIG_NA1, _rotation, _tensor, write_magres  # noqa: E402

LATTICE = np.array(((5.0, 0, 0), (0, 6.0, 0), (-1.0, 0, 7.0)))
ATOMS = [("Na", 1, (0.0, 0.0, 0.0)), ("Na", 2, (2.0, 3.0, 3.5)), ("Cl", 1, (2.6, 0.0, 0.0))]


def _write(tmp_path, name="s.magres"):
    R1, R2 = _rotation(1), _rotation(2)
    return write_magres(tmp_path / name, [
        ("Na", 1, ATOMS[0][2], _tensor(SIG_NA1, R1), _tensor(EFG1, R2)),
        ("Na", 2, ATOMS[1][2], _tensor(SIG_NA1, R2), _tensor(EFG1, R1)),
        ("Cl", 1, ATOMS[2][2], np.eye(3) * 980.0, np.zeros((3, 3))),
    ], lattice=LATTICE)


@pytest.fixture
def sample(tmp_path):
    return D.read_calculation(_write(tmp_path))


@pytest.fixture
def model(sample):
    return D.dft_sites(sample, element="Na", delta_ref=7.2, sigma_ref=550.0)


def _brute_force(span):
    """Every (label, fractional position) with the fractions in span."""
    inv = np.linalg.inv(LATTICE)
    found = []
    for sp, i, pos in ATOMS:
        f0 = np.mod(np.asarray(pos) @ inv, 1.0)
        for n in itertools.product(range(-8, 9), repeat=3):
            f = f0 + n
            if np.all(f >= span[:, 0] - 1e-4) and np.all(f <= span[:, 1] + 1e-4):
                found.append((f"{sp}{i}", tuple(np.round(f, 6))))
    return sorted(found)


def _drawn(atoms):
    inv = np.linalg.inv(LATTICE)
    return sorted((a["label"], tuple(np.round(np.asarray(a["position"]) @ inv, 6)))
                  for a in atoms)


# ======================================================================
# Fractional range / supercell
# ======================================================================

@pytest.mark.parametrize("span", [((0, 2), (0, 2), (0, 1)),
                                  ((-0.2, 1.6), (0, 1), (0, 2)),
                                  ((0.3, 0.9), (-0.5, 0.5), (0.25, 1.75))])
def test_range_region_is_every_atom_in_the_range(sample, span):
    atoms = S.structure_atoms(sample, region="range", frac_range=span)
    assert _drawn(atoms) == _brute_force(np.asarray(span, float))


def test_supercell_counts_and_the_unit_cell_is_the_same_as_region_cell(sample):
    atoms = S.structure_atoms(sample, region="range", frac_range=((0, 2), (0, 2), (0, 1)))
    # Na1 on a corner: 3 x 3 x 2 corner points; Na2 inside: 4; Cl1 on an edge (b = c = 0): 2 x 3 x 2.
    counts = {lab: sum(a["label"] == lab for a in atoms) for lab in ("Na1", "Na2", "Cl1")}
    assert counts == {"Na1": 18, "Na2": 4, "Cl1": 12}
    one = S.structure_atoms(sample, region="range", frac_range=((0, 1),) * 3)
    assert _drawn(one) == _drawn(S.structure_atoms(sample, region="cell"))


def test_range_checks(sample, tmp_path):
    with pytest.raises(ValueError, match="smaller to a larger"):
        S.structure_atoms(sample, region="range", frac_range=((1, 0), (0, 1), (0, 1)))
    with pytest.raises(ValueError, match="at most 12"):
        S.structure_atoms(sample, region="range", frac_range=((0, 13), (0, 1), (0, 1)))
    with pytest.raises(ValueError, match="three"):
        S.structure_atoms(sample, region="range", frac_range=(0, 1))
    with pytest.raises(ValueError, match="More than"):
        S.structure_atoms(sample, region="range", frac_range=((0, 12), (0, 12), (0, 12)))
    from test_dft_molecular import SIG_H, SIG_N, write_orca
    calc = D.read_calculation(write_orca(tmp_path / "nh3.out",
                                         [("N", (0, 0, 0), SIG_N), ("H", (1.0, 0, 0), SIG_H)]))
    with pytest.raises(ValueError, match="needs a cell"):
        S.structure_atoms(calc, region="range", frac_range=((0, 1),) * 3)


def test_tensor_on_every_copy_or_the_middle_one(sample, model):
    na2 = next(r for r in model["atoms"] if r["label"] == "Na2")
    span = ((-0.2, 1.6), (0, 1), (0, 2))
    # Na2 (fractions 0.5, 0.5, 0.5) has copies at a = 0.5, 1.5 and c = 0.5, 1.5 (a = -0.5
    # and 2.5 are out).
    fig, notes = S.plot_dft_structure(sample, na2, region="range", frac_range=span,
                                      tensor_copies="all")
    glyphs = [c for c in fig.axes[0].collections if type(c) is Poly3DCollection]
    copies = sum(lab == "Na2" for lab, _f in _brute_force(np.asarray(span, float)))
    assert len(glyphs) == copies == 4
    assert "4 drawn (every copy in the range)" in notes[0]
    # One label, however many copies.
    assert sum(t.get_text() == "Na2" for t in fig.axes[0].texts) == 1
    plt.close(fig)
    fig, _notes = S.plot_dft_structure(sample, na2, region="range", frac_range=span)
    assert sum(type(c) is Poly3DCollection for c in fig.axes[0].collections) == 1
    plt.close(fig)


def test_tensor_with_no_copy_in_the_range_is_said(sample, model):
    na2 = next(r for r in model["atoms"] if r["label"] == "Na2")
    fig, notes = S.plot_dft_structure(sample, na2, region="range",
                                      frac_range=((0.6, 0.9), (0, 1), (0, 1)))
    assert any("no copy inside the range" in n for n in notes)
    assert any("No atom lies inside this range" in n for n in notes)
    plt.close(fig)


def _cell_segments(ax, colour):
    n = 0
    for c in ax.collections:
        if isinstance(c, Line3DCollection) and len(c.get_colors()) and \
                mcolors.to_hex(c.get_colors()[0]) == colour:
            n += len(c._segments3d)
    return n


@pytest.mark.parametrize("edges, expected", [("cells", 12 * 4), ("origin", 12),
                                             ("range", 12), ("none", 0)])
def test_cell_edges(sample, model, edges, expected):
    fig, _notes = S.plot_dft_structure(sample, model["atoms"], region="range",
                                       frac_range=((0, 2), (0, 2), (0, 1)), cell_edges=edges,
                                       bonds=False, tensor_style={"cell_colour": "#123456"})
    assert _cell_segments(fig.axes[0], "#123456") == expected
    labels = {t.get_text() for t in fig.axes[0].texts}
    assert ({"a", "b", "c"} <= labels) == (expected > 0)
    plt.close(fig)


def test_supercell_title(sample, model):
    fig, notes = S.plot_dft_structure(sample, model["atoms"], region="range",
                                      frac_range=((0, 2), (0, 2), (0, 1)))
    assert "2 × 2 × 1 supercell" in fig._suptitle.get_text()
    assert "2 × 2 × 1 supercell" in notes[0]
    plt.close(fig)


# ======================================================================
# Atoms and bonds
# ======================================================================

def _atoms_scatter(ax):
    return next(c for c in ax.collections if isinstance(c, Path3DCollection))


def test_atom_style(sample, model):
    style = {"atom_colours": {"Cl": "#00aa00"}, "atom_size_by": "uniform", "atom_scale": 0.5,
             "atom_alpha": 0.4, "atom_edge_colour": "#ff00ff", "atom_edge_width": 1.0,
             "atom_depthshade": True}
    fig, notes = S.plot_dft_structure(sample, model["atoms"], tensor_style=style, bonds=False)
    ax = fig.axes[0]
    sc = _atoms_scatter(ax)
    atoms = S.structure_atoms(sample, region="cell")
    # (Read before the figure is drawn: drawing sorts them by depth.)
    np.testing.assert_allclose(sc.get_sizes(), (1.2 * 9.0 * 0.5) ** 2)
    assert sc.get_alpha() == pytest.approx(0.4)
    assert sc._depthshade
    assert mcolors.to_hex(sc._edgecolors[0]) == "#ff00ff"
    faces = [mcolors.to_hex(c) for c in sc._facecolors]
    want = ["#00aa00" if a["species"] == "Cl" else S.ELEMENT_COLOURS["Na"].lower()
            for a in atoms]
    assert faces == want
    # Every atom of interest (Na1, Na2) gets the heavier outline.
    widths = list(sc.get_linewidths())
    heavy = [a["label"] in ("Na1", "Na2") for a in atoms]
    assert all(w == pytest.approx(2.5) for w, h in zip(widths, heavy) if h)
    assert all(w == pytest.approx(1.0) for w, h in zip(widths, heavy) if not h)
    # The legend's key follows the element colours.
    legend = {t.get_text(): h for t, h in zip(ax.get_legend().get_texts(),
                                              ax.get_legend().legend_handles)}
    assert mcolors.to_hex(legend["Cl"].get_markerfacecolor()) == "#00aa00"
    assert any("all the same size" in n for n in notes)
    plt.close(fig)
    # The defaults are untouched by a style with element colours.
    assert tv.TENSOR_STYLE_DEFAULTS["atom_colours"] == {}


def test_atom_size_zero_and_by_radius(sample, model):
    fig, _n = S.plot_dft_structure(sample, model["atoms"], tensor_style={"atom_scale": 0})
    assert not any(isinstance(c, Path3DCollection) for c in fig.axes[0].collections)
    plt.close(fig)
    fig, _n = S.plot_dft_structure(sample, model["atoms"], atom_scale=2.0)
    sizes = _atoms_scatter(fig.axes[0]).get_sizes()
    atoms = S.structure_atoms(sample, region="cell")
    np.testing.assert_allclose(sizes, [(S.radius_of(a["species"]) * 18.0) ** 2 for a in atoms])
    plt.close(fig)
    with pytest.raises(ValueError, match="atom_size_by"):
        S.plot_dft_structure(sample, model["atoms"], tensor_style={"atom_size_by": "bogus"})


def test_bond_style(sample, model):
    fig, _n = S.plot_dft_structure(sample, model["atoms"],
                                   tensor_style={"bond_colour": "#abcdef", "bond_width": 5.0})
    bonds = [c for c in fig.axes[0].collections if isinstance(c, Line3DCollection)
             and mcolors.to_hex(c.get_colors()[0]) == "#abcdef"]
    assert len(bonds) == 1 and bonds[0].get_linewidths()[0] == pytest.approx(5.0)
    plt.close(fig)
    fig, _n = S.plot_dft_structure(sample, model["atoms"], tensor_style={"bond_width": 0})
    # Only the cell's edges are left.
    assert sum(isinstance(c, Line3DCollection) for c in fig.axes[0].collections) == 1
    plt.close(fig)


# ======================================================================
# The 3D box's text
# ======================================================================

def _render(fig):
    fig.canvas.draw()


def _tick_texts(axis):
    return [t.get_text() for t in axis.get_ticklabels()]


def test_tick_and_axis_labels_can_be_removed_and_stay_removed(sample, model):
    fig, _n = S.plot_dft_structure(sample, model["atoms"],
                                   tensor_style={"tick_labels": False, "axis_labels": False})
    ax = fig.axes[0]
    _render(fig)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        assert all(text == "" for text in _tick_texts(axis))
        assert len(axis.get_major_locator()()) >= 2          # the ticks themselves stay
        assert not axis.label.get_visible()
    # A refit (the GUI's, after a rotation) keeps them off ...
    ax.view_init(elev=50, azim=-20)
    tv.fit_3d_ticks(ax, 10.0)
    _render(fig)
    assert all(text == "" for text in _tick_texts(ax.xaxis))
    assert not ax.xaxis.label.get_visible()
    # ... and the style puts them back.
    tv.apply_3d_axes_style(ax, {})
    tv.fit_3d_ticks(ax, 10.0)
    _render(fig)
    assert any(text for text in _tick_texts(ax.xaxis)) and ax.xaxis.label.get_visible()
    plt.close(fig)


def test_box_off(sample, model):
    fig, _n = S.plot_dft_structure(sample, model["atoms"], tensor_style={"axes_box": False})
    ax = fig.axes[0]
    assert not ax._axis3don
    assert not ax.xaxis.label.get_visible()
    plt.close(fig)


def test_padding(sample, model):
    fig0, _n = S.plot_dft_structure(sample, model["atoms"])
    fig1, _n = S.plot_dft_structure(sample, model["atoms"],
                                    tensor_style={"tick_pad": 6.0, "label_pad": 9.0})
    a0, a1 = fig0.axes[0], fig1.axes[0]
    assert a1.xaxis.get_major_ticks()[0].get_pad() == pytest.approx(6.0)
    for axis0, axis1 in zip((a0.xaxis, a0.yaxis, a0.zaxis), (a1.xaxis, a1.yaxis, a1.zaxis)):
        # The same view and ticks: the label moves out by both paddings.
        assert axis1.labelpad - axis0.labelpad == pytest.approx(15.0, abs=0.5)
    plt.close(fig0)
    plt.close(fig1)


def test_ellipsoid_and_surface_views_take_the_axes_style(model, sample):
    rec = model["atoms"][0]
    for style in ("ellipsoid", "surface"):
        fig, _notes = D.plot_dft_tensors(rec, sample, frame="cell", style=style,
                                         larmor_Hz=105.8e6, nucleus="23Na",
                                         tensor_style={"tick_labels": False,
                                                       "axis_labels": False})
        ax = [a for a in fig.axes if a.name == "3d"][0]
        _render(fig)
        assert all(text == "" for text in _tick_texts(ax.zaxis))
        assert not ax.zaxis.label.get_visible()
        plt.close(fig)


# ======================================================================
# Legend: placement, columns, room outside a 3D picture
# ======================================================================

def test_legend_placement():
    assert legend_placement("best") == {"loc": "best"}
    assert legend_placement("upper center", columns=3) == {"loc": "upper center", "ncols": 3}
    out = legend_placement("outside right")
    assert out["loc"] == "center left" and out["bbox_to_anchor"][0] > 1.0
    assert legend_placement("custom", 2, 0.1, 1.15) == {
        "loc": "upper left", "bbox_to_anchor": (0.1, 1.15), "ncols": 2}
    assert "ncols" not in legend_placement("best", 0)
    with pytest.raises(ValueError):
        legend_placement("somewhere")
    with pytest.raises(ValueError):
        legend_placement("best", -1)
    for position in LEGEND_POSITIONS:
        fig, ax = plt.subplots()
        ax.plot([0, 1], label="a")
        ax.legend(**legend_placement(position, 2))
        fig.canvas.draw()
        plt.close(fig)


def _legend_inside_figure(fig, ax):
    fig.canvas.draw()
    box = ax.get_legend().get_window_extent().transformed(fig.transFigure.inverted())
    return box.x0 >= -1e-3 and box.y0 >= -1e-3 and box.x1 <= 1 + 1e-3 and box.y1 <= 1 + 1e-3


def test_room_is_made_for_a_legend_outside_a_3d_picture(sample, model):
    fig, _n = S.plot_dft_structure(sample, model["atoms"], legend_loc="outside right")
    ax = fig.axes[0]
    assert _legend_inside_figure(fig, ax)
    narrowed = ax.get_position().x1
    # A legend put back inside gives the room back ...
    ax.legend(handles=ax.get_legend().legend_handles, **legend_placement("upper left"))
    make_room_for_legend(ax)
    assert ax.get_position().x1 > narrowed
    # ... and one above the picture lowers it.
    ax.legend(handles=ax.get_legend().legend_handles, **legend_placement("custom", 4, 0.0, 1.2))
    make_room_for_legend(ax)
    assert _legend_inside_figure(fig, ax)
    plt.close(fig)


def test_structure_legend_columns(sample, model):
    fig, _n = S.plot_dft_structure(sample, model["atoms"], legend_columns=3,
                                   legend_loc="upper left")
    assert fig.axes[0].get_legend()._ncols == 3
    plt.close(fig)


# ======================================================================
# GUI
# ======================================================================

@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@pytest.fixture
def gui(qapp, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    from opennmr.gui.main_window import MainWindow
    for name in ("critical", "information", "warning"):
        monkeypatch.setattr(QMessageBox, name, lambda *a, **k: None)
    monkeypatch.setattr(MainWindow, "_FIGURE_STYLES_PATH", str(tmp_path / "figure_styles.json"))
    monkeypatch.setattr(MainWindow, "_PLOT_STYLE_CONFIG_PATH", str(tmp_path / "plot_style.json"))
    window = MainWindow()
    panel = window.dft_sim_panel
    panel.dft_sample_edit.setText(_write(tmp_path))
    assert panel.read_files(quiet=True)
    panel.dft_sigma_override_check.setChecked(True)
    panel.dft_sigma_ref_spin.setValue(550.0)
    assert panel.calculate(simulate=False)
    panel.dft_tensor_style_combo.setCurrentIndex(
        panel.dft_tensor_style_combo.findData("structure"))
    panel.dft_structure_sites_combo.setCurrentIndex(
        panel.dft_structure_sites_combo.findData("all"))
    return window, panel


def _ax3d(window):
    return next(ax for ax in window.simulation_canvas.figure.axes if ax.name == "3d")


def test_gui_fractional_range_and_supercell(gui):
    window, panel = gui
    spins = panel.dft_structure_range_spins
    assert spins["a"][0].isHidden() and panel.dft_structure_copies_combo.isHidden()
    panel.dft_structure_region_combo.setCurrentIndex(
        panel.dft_structure_region_combo.findData("range"))
    assert not spins["a"][0].isHidden() and not panel.dft_structure_edges_combo.isHidden()
    assert panel.dft_structure_radius_spin.isHidden()
    for spin, n in zip(panel.dft_structure_supercell_spins, (2, 2, 1)):
        spin.setValue(n)
    panel.dft_structure_supercell_button.click()
    assert panel.frac_range() == ((0.0, 2.0), (0.0, 2.0), (0.0, 1.0))
    panel.dft_structure_copies_combo.setCurrentIndex(
        panel.dft_structure_copies_combo.findData("all"))
    assert panel.show_tensors(), panel.dft_tensor_notes_label.text()
    assert "2 × 2 × 1 supercell" in panel.dft_tensor_notes_label.text()
    ax = _ax3d(window)
    # Na1 on 18 corner points, Na2 on 4: every copy carries its tensor.
    assert sum(type(c) is Poly3DCollection for c in ax.collections) == 22
    args = panel.structure_arguments()
    assert args["region"] == "range" and args["tensor_copies"] == "all"
    # A backwards range is a message, not a crash.
    spins["a"][0].setValue(1.5)
    spins["a"][1].setValue(0.5)
    assert not panel.show_tensors()
    assert "smaller to a larger" in panel.dft_tensor_notes_label.text()


def test_gui_legend_position_and_columns(gui):
    window, panel = gui
    assert panel.show_tensors()
    combo = window.output_legend_loc_combo
    assert [combo.itemText(i) for i in range(combo.count())] == list(LEGEND_POSITIONS)
    assert not window.output_legend_x_spin.isEnabled()
    window.output_legend_checkbox.setChecked(True)      # Show Legend
    canvas = window.simulation_canvas
    window.output_legend_columns_spin.setValue(3)
    assert canvas.style_settings()["legend_columns"] == 3
    assert _ax3d(window).get_legend()._ncols == 3
    # Outside right: the picture is narrowed to make room for it.
    window.output_legend_columns_spin.setValue(1)
    wide = _ax3d(window).get_position().x1
    combo.setCurrentText("outside right")
    ax = _ax3d(window)
    assert ax.get_position().x1 < wide
    canvas.canvas.draw()
    assert _legend_inside_figure(canvas.figure, ax)
    combo.setCurrentText("custom")
    assert window.output_legend_x_spin.isEnabled()
    window.output_legend_x_spin.setValue(0.1)
    window.output_legend_y_spin.setValue(0.9)
    anchor = _ax3d(window).get_legend().get_bbox_to_anchor()
    box = _ax3d(window).get_position()
    fig_w, fig_h = canvas.figure.bbox.width, canvas.figure.bbox.height
    assert anchor.x0 / fig_w == pytest.approx(box.x0 + 0.1 * box.width, abs=1e-3)
    assert anchor.y0 / fig_h == pytest.approx(box.y0 + 0.9 * box.height, abs=1e-3)
    # Auto columns: the picture's own choice comes back.
    window.output_legend_columns_spin.setValue(0)
    assert _ax3d(window).get_legend()._ncols == 1
    # Kept as the per-type default and the saved default.
    window._on_save_style_default_clicked()
    import json
    saved = json.load(open(MainWindowPath(window)))
    assert saved["legend_loc"] == "custom" and saved["legend_x"] == pytest.approx(0.1)


def MainWindowPath(window):  # noqa: N802
    return type(window)._PLOT_STYLE_CONFIG_PATH


def test_gui_tensor_style_axes_and_atoms(gui):
    window, panel = gui
    assert panel.show_tensors()
    window._open_tensor_style()
    dialog = window._tensor_style_dialog
    assert [dialog.tabs.tabText(i) for i in range(dialog.tabs.count())] == [
        "Tensors", "Axes && Labels", "Atoms && Bonds"]
    dialog.tick_labels_check.setChecked(False)
    dialog.axis_labels_check.setChecked(False)
    canvas = window.simulation_canvas
    canvas.canvas.draw()
    ax = _ax3d(window)
    assert all(t.get_text() == "" for t in ax.xaxis.get_ticklabels())
    assert not ax.zaxis.label.get_visible()
    dialog.tick_pad_spin.setValue(5.0)
    assert _ax3d(window).xaxis.get_major_ticks()[0].get_pad() == pytest.approx(5.0)
    dialog.axes_box_check.setChecked(False)
    assert not _ax3d(window)._axis3don
    # The elements of the structure on show, and their colours.
    names = [dialog.element_combo.itemText(i) for i in range(dialog.element_combo.count())]
    assert names == ["Na", "Cl"]
    dialog.set_element_colour("Cl", "#00aa00")
    sc = _atoms_scatter(_ax3d(window))
    assert "#00aa00" in {mcolors.to_hex(c) for c in sc.get_facecolor()}
    dialog.atom_size_combo.setCurrentIndex(dialog.atom_size_combo.findData("uniform"))
    dialog.atom_scale_spin.setValue(0.5)
    np.testing.assert_allclose(_atoms_scatter(_ax3d(window)).get_sizes(), (1.2 * 9.0 * 0.5) ** 2)
    dialog.bond_own_colour_check.setChecked(True)
    dialog.set_colour("bond_colour", "#abcdef")
    assert canvas.style_settings()["tensor_style"]["bond_colour"] == "#abcdef"
    # Round trip: what the dialog shows is what it writes.
    style = dialog.style()
    dialog.load(style)
    assert dialog.style() == style
    dialog._reset_element_colours()
    assert canvas.style_settings()["tensor_style"]["atom_colours"] == {}
    dialog._reset()
    assert _ax3d(window)._axis3don and _ax3d(window).zaxis.label.get_visible()
