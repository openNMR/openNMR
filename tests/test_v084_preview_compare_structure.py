"""0.8.4 (explicit user requests, 2026-10-26):

* File ▸ Save Figure As… shows a preview of the figure exactly as it will
  be written (legend position, text size relative to the figure size)
  before the file is chosen -- gui.figure_preview_dialog.
* Compare: entries can be taken out of the list entirely (Remove, Clear,
  the Delete key), not only unticked, and stay out when the Open
  experiments selection changes; everything above Spectrum Arithmetic is
  one collapsible "Compare Controls" section.
* DFT tensors drawn on the structure, centred on their atoms --
  notebook.dft_structure, and Simulate ▸ DFT tensors ▸ Show: On the
  structure.

All data synthetic."""

import io

import matplotlib
import numpy as np
import pytest
from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QKeyEvent

matplotlib.use("Agg")

from opennmr.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum  # noqa: E402
from opennmr.gui import figure_preview_dialog as fpd  # noqa: E402
from opennmr.gui.widgets.canvas import MplCanvas  # noqa: E402
from opennmr.gui.widgets.collapsible_section import CollapsibleSection  # noqa: E402
from opennmr.notebook import dft_nmr as D  # noqa: E402
from opennmr.notebook import dft_structure as S  # noqa: E402

from test_dft_nmr import EFG1, SIG_NA1, _rotation, _tensor, write_magres  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(qapp):
    from opennmr.gui.main_window import MainWindow
    return MainWindow()


def _peak(x0, lo=0.0, hi=10.0, n=501, sfo1=400.0):
    ppm = np.linspace(hi, lo, n)
    data = np.exp(-0.5 * ((ppm - x0) / 0.1) ** 2)
    axis = NMRAxis(kind="frequency", ppm=ppm, hz=ppm * sfo1)
    return NMRSpectrum(data=data, axes=[axis],
                       metadata=NMRMetadata(vendor="bruker", nucleus="1H", sfo1=sfo1))


def _png_array(data: bytes):
    import matplotlib.image as mpimg
    return mpimg.imread(io.BytesIO(data), format="png")


# ======================================================================
# Save Figure preview
# ======================================================================

def test_preview_has_the_export_size_and_leaves_the_canvas_as_it_was(qapp):
    canvas = MplCanvas()
    canvas.show_spectrum(_peak(5.0))
    live_size = tuple(canvas.figure.get_size_inches())
    canvas.set_export_settings(dpi=300, fig_width=6.0, fig_height=4.0)
    preview = fpd.render_export_preview(canvas, preview_dpi=50)
    assert preview.dpi == 300
    assert preview.width_in == pytest.approx(6.0, abs=0.03)
    assert preview.height_in == pytest.approx(4.0, abs=0.03)
    assert preview.pixels == (round(preview.width_in * 300), round(preview.height_in * 300))
    image = _png_array(preview.png)
    assert image.shape[1] == pytest.approx(6.0 * 50, abs=1)
    assert image.shape[0] == pytest.approx(4.0 * 50, abs=1)
    # Nothing about the live view, or the stored export settings, changed.
    assert tuple(canvas.figure.get_size_inches()) == pytest.approx(live_size)
    assert (canvas._export_dpi, canvas._export_fig_width, canvas._export_fig_height) \
        == (300, 6.0, 4.0)


def test_preview_overrides_are_for_that_preview_only(qapp):
    canvas = MplCanvas()
    canvas.show_spectrum(_peak(5.0))
    preview = fpd.render_export_preview(canvas, width=3.0, height=2.0, dpi=150, preview_dpi=40)
    assert preview.width_in == pytest.approx(3.0, abs=0.05)
    assert preview.dpi == 150
    assert canvas._export_fig_width is None and canvas._export_dpi is None


def test_the_preview_is_the_saved_figure(qapp, tmp_path):
    # The point of it: what the preview shows is what is written -- same
    # pixels when rendered at the same dpi, legend and labels included.
    canvas = MplCanvas()
    canvas.show_compare([_peak(3.0), _peak(6.0)], mode="overlay",
                        legend_labels=["first", "second"])
    canvas.set_legend_visible(True)
    assert any(ax.get_legend() is not None for ax in canvas.figure.axes) or canvas.figure.legends
    canvas.set_export_settings(dpi=60, fig_width=5.0, fig_height=3.5)
    preview = fpd.render_export_preview(canvas, preview_dpi=60)
    out = tmp_path / "fig.png"
    fpd.save_with_overrides(canvas, str(out))
    saved = _png_array(out.read_bytes())
    shown = _png_array(preview.png)
    assert saved.shape == shown.shape
    assert np.max(np.abs(saved - shown)) < 1e-6


def test_the_preview_follows_the_size_and_fits_the_window(qapp):
    canvas = MplCanvas()
    canvas.show_spectrum(_peak(5.0))
    wide = fpd.render_export_preview(canvas, width=8, height=3, max_side_px=400)
    tall = _png_array(fpd.render_export_preview(canvas, width=3, height=8, max_side_px=400).png)
    assert _png_array(wide.png).shape[:2] == (150, 400)      # longest side at max_side_px
    assert tall.shape[:2] == (400, 150)
    assert wide.preview_dpi == pytest.approx(50.0)


def test_toolbar_save_opens_the_preview(qapp, monkeypatch):
    canvas = MplCanvas()
    canvas.show_spectrum(_peak(5.0))
    seen = []
    monkeypatch.setattr(fpd.FigurePreviewDialog, "exec", lambda self: seen.append(self) or 0)
    canvas.toolbar.save_figure()
    assert len(seen) == 1 and seen[0].canvas is canvas
    assert seen[0].preview is not None and seen[0].image_label.pixmap() is not None


def test_dialog_saves_with_its_size_and_keeps_it(qapp, tmp_path):
    canvas = MplCanvas()
    canvas.show_spectrum(_peak(5.0))
    dialog = fpd.FigurePreviewDialog(canvas)
    dialog.width_spin.setValue(4.0)
    dialog.height_spin.setValue(3.0)
    dialog.dpi_spin.setValue(100)
    dialog.refresh()
    assert "4.00 × 3.00 in" in dialog.info_label.text()
    kept = []
    dialog.settingsKept.connect(lambda *v: kept.append(v))
    path = dialog.save_to(str(tmp_path / "figure"), chosen_filter="PDF (vector) (*.pdf)")
    assert path.endswith(".pdf") and (tmp_path / "figure.pdf").read_bytes()[:4] == b"%PDF"
    assert kept == [(100.0, 4.0, 3.0)]
    assert (canvas._export_dpi, canvas._export_fig_width, canvas._export_fig_height) \
        == (100.0, 4.0, 3.0)
    png = dialog.save_to(str(tmp_path / "figure.png"))
    image = _png_array((tmp_path / "figure.png").read_bytes())
    assert png and image.shape[1] == pytest.approx(400, abs=2)


def test_dialog_in_centimetres(qapp):
    canvas = MplCanvas()
    canvas.show_spectrum(_peak(5.0))
    dialog = fpd.FigurePreviewDialog(canvas, unit="cm")
    assert dialog.width_spin.unit() == "cm"
    assert " cm at " in dialog.info_label.text()


def test_kept_settings_reach_plot_style_export(window):
    window.canvas.exportSettingsKept.emit(450.0, 5.5, 4.25)
    assert window.output_dpi_spin.value() == 450
    assert window.output_fig_width_spin.value() == pytest.approx(5.5)
    assert window.output_fig_height_spin.value() == pytest.approx(4.25)


def test_export_unit_setting_reaches_the_preview(window):
    window._set_export_unit("cm", save=False)
    assert window.canvas.export_length_unit == "cm"
    window._set_export_unit("in", save=False)
    assert window.compare_canvas.export_length_unit == "in"


# ======================================================================
# Compare: remove, and the Compare Controls section
# ======================================================================

def _three(window):
    for x0, label in ((2.0, "A"), (5.0, "B"), (8.0, "C")):
        window._add_derived_spectrum(_peak(x0), label)
    return window


def _select(window, rows):
    window.spectrum_list.clearSelection()
    for r in rows:
        window.spectrum_list.item(r).setSelected(True)


def _listed(window):
    return [window.compare_list.item(i).data(Qt.ItemDataRole.UserRole)
            for i in range(window.compare_list.count())]


def test_remove_takes_an_entry_out_and_it_stays_out(window):
    _three(window)
    _select(window, [0, 1])
    assert _listed(window) == [(0, None), (1, None)]
    window.compare_list.setCurrentRow(1)
    window.compare_remove_button.click()
    assert _listed(window) == [(0, None)]
    # Selecting one more spectrum must not bring the removed one back...
    _select(window, [0, 1, 2])
    assert _listed(window) == [(0, None), (2, None)]
    # ...but deselecting it and selecting it again does.
    _select(window, [0, 2])
    _select(window, [0, 1, 2])
    assert (1, None) in _listed(window)


def test_delete_key_removes_the_highlighted_entry(window):
    _three(window)
    _select(window, [0, 1, 2])
    window.compare_list.setCurrentRow(0)
    event = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Delete, Qt.KeyboardModifier.NoModifier)
    assert window._compare_arrow_keys.eventFilter(window.compare_list, event)
    assert _listed(window) == [(1, None), (2, None)]
    # The next row is highlighted, so Delete again removes that one.
    assert window.compare_list.currentRow() == 0
    window._compare_arrow_keys.eventFilter(window.compare_list, event)
    assert _listed(window) == [(2, None)]


def test_clear_empties_the_list_and_a_drag_brings_one_back(window):
    _three(window)
    _select(window, [0, 1])
    window.compare_clear_button.click()
    assert _listed(window) == []
    window._on_experiments_dropped_on_compare([1], -1)
    assert _listed(window) == [(1, None)]


def test_right_click_remove_also_sticks(window):
    _three(window)
    _select(window, [0, 1])
    window._remove_compare_entry((0, None))
    _select(window, [0, 1, 2])
    assert (0, None) not in _listed(window)


def test_compare_controls_section_holds_the_plot_options(window):
    page = window.side_tabs.widget(window._compare_tab_index)
    sections = {s.title(): s for s in page.findChildren(CollapsibleSection)}
    controls = sections["Compare Controls"]
    assert not controls.isExpanded()           # every section starts collapsed
    content = controls.contentWidget()
    for widget in (window.layout_mode_combo, window.compare_normalize_combo,
                   window.compare_2d_controls, window.compare_panels_controls,
                   window.compare_plot_button):
        assert content.isAncestorOf(widget)
    # The list, and the buttons that edit it, stay on show.
    for widget in (window.compare_list, window.compare_remove_button,
                   window.compare_clear_button):
        assert not content.isAncestorOf(widget) and page.isAncestorOf(widget)
    assert not content.isAncestorOf(window.arithmetic_panel)
    controls.setExpanded(True)
    assert controls.isExpanded() and not content.parentWidget().isHidden()


# ======================================================================
# DFT tensors on the structure
# ======================================================================

LATTICE = ((5.0, 0, 0), (0, 6.0, 0), (-1.0, 0, 7.0))


@pytest.fixture
def sample(tmp_path):
    R1, R2 = _rotation(1), _rotation(2)
    path = write_magres(tmp_path / "s.magres", [
        ("Na", 1, (0.0, 0.0, 0.0), _tensor(SIG_NA1, R1), _tensor(EFG1, R2)),
        ("Na", 2, (2.0, 3.0, 3.5), _tensor(SIG_NA1, R2), _tensor(EFG1, R1)),
        ("Cl", 1, (2.6, 0.0, 0.0), np.eye(3) * 980.0, np.zeros((3, 3))),
    ], lattice=LATTICE)
    return D.read_calculation(path)


def test_cell_region_wraps_and_adds_the_face_copies(sample):
    atoms = S.structure_atoms(sample, region="cell")
    lattice = np.array(LATTICE)
    frac = np.array([a["position"] for a in atoms]) @ np.linalg.inv(lattice)
    assert np.all(frac > -1e-9) and np.all(frac < 1 + 1e-9)
    # Na1 sits on a corner: all eight corners are drawn.
    assert sum(a["label"] == "Na1" for a in atoms) == 8
    # Na2 is inside: once. Cl1 has y = z = 0: four copies (an edge).
    assert sum(a["label"] == "Na2" for a in atoms) == 1
    assert sum(a["label"] == "Cl1" for a in atoms) == 4


def test_radius_region_includes_periodic_images_by_distance(sample):
    centre = np.zeros(3)
    atoms = S.structure_atoms(sample, centres=[centre], region="radius", radius=5.5)
    d = [np.linalg.norm(a["position"] - centre) for a in atoms]
    assert max(d) <= 5.5 + 1e-9
    # Every Na1 image within 5.5 Å: the origin and ±a (5 Å); |b| = 6 is out.
    na1 = sorted(round(float(np.linalg.norm(a["position"])), 6)
                 for a in atoms if a["label"] == "Na1")
    assert na1 == [0.0, 5.0, 5.0]
    # Brute force agrees on the count.
    lattice = np.array(LATTICE)
    count = 0
    for a in sample["atoms"]:
        for n in np.ndindex(7, 7, 7):
            p = np.asarray(a["position"]) + (np.array(n) - 3) @ lattice
            count += np.linalg.norm(p - centre) <= 5.5
    assert len(atoms) == count


def test_bonds_by_covalent_radii_and_no_metal_metal(sample):
    atoms = [{"species": "Na", "position": np.zeros(3)},
             {"species": "Cl", "position": np.array([2.8, 0, 0])},
             {"species": "Na", "position": np.array([3.0, 0.1, 0])},
             {"species": "O", "position": np.array([0, 0, 2.0])}]
    pairs = S.find_bonds(atoms)
    assert (0, 1) in pairs and (1, 2) in pairs and (0, 3) in pairs
    assert (0, 2) not in pairs               # two metals: never bonded
    assert (0, 2) in S.find_bonds(atoms, metal_metal=True)
    assert (1, 3) not in S.find_bonds(atoms)  # 3.4 Å > 1.15 × (1.02 + 0.66)


def test_glyphs_are_the_tensor(sample):
    R = _rotation(5)
    T = _tensor((10.0, -4.0, -6.0), R)               # traceless
    x, y, z, sign = S.tensor_glyph(T, "efg", 0.1, (1.0, 2.0, 3.0))
    pts = np.stack([x - 1.0, y - 2.0, z - 3.0], -1)
    r = np.linalg.norm(pts, axis=-1)
    n = pts / np.where(r > 0, r, 1)[..., None]
    q = np.einsum("...i,ij,...j->...", n, T, n)
    ok = r > 1e-6
    np.testing.assert_allclose(r[ok], 0.1 * np.abs(q[ok]), rtol=1e-9, atol=1e-10)
    assert np.all(sign[ok][np.abs(q[ok]) > 1e-6] == np.sign(q[ok][np.abs(q[ok]) > 1e-6]))
    # Largest extent along the largest principal axis, |10| × scale (to
    # the resolution of the drawn grid).
    assert r.max() == pytest.approx(1.0, rel=5e-3) and r.max() <= 1.0 + 1e-12
    # The ellipsoid: semi-axes |λ| along the eigenvectors.
    P = _tensor((30.0, 20.0, 10.0), R)
    x, y, z, _s = S.tensor_glyph(P, "shift", 0.01, (0, 0, 0))
    pts = np.stack([x, y, z], -1) @ R                   # into the principal frame
    np.testing.assert_allclose(np.sum((pts / np.array([0.3, 0.2, 0.1])) ** 2, -1), 1.0,
                               atol=1e-8)


def test_tensor_of_kinds(sample):
    model = D.dft_sites(sample, element="Na", delta_ref=7.2, sigma_ref=550.0)
    rec = model["atoms"][0]
    full = S.tensor_of(rec, "shift")
    aniso = S.tensor_of(rec, "shift_aniso")
    np.testing.assert_allclose(full - aniso, np.trace(full) / 3 * np.eye(3), atol=1e-9)
    efg = S.tensor_of(rec, "efg")
    assert abs(np.trace(aniso)) < 1e-9
    assert abs(np.trace(efg)) < 1e-12 * np.max(np.abs(efg))
    with pytest.raises(ValueError):
        S.tensor_of(rec, "bogus")


@pytest.mark.parametrize("tensor", ["shift", "shift_aniso", "efg"])
@pytest.mark.parametrize("region", ["cell", "radius"])
def test_plot_on_the_structure(sample, tensor, region):
    model = D.dft_sites(sample, element="Na", delta_ref=7.2, sigma_ref=550.0)
    fig, notes = S.plot_dft_structure(sample, model["atoms"], tensor=tensor, region=region,
                                      radius=4.0, tensor_size=0.8)
    ax = fig.axes[0]
    assert ax.name == "3d"
    assert any("Scale: 0.8 Å" in n for n in notes)
    # Equal scaling on all three axes.
    spans = [np.ptp(lim) for lim in (ax.get_xlim(), ax.get_ylim(), ax.get_zlim())]
    assert spans == pytest.approx([spans[0]] * 3)
    # One glyph per site drawn, centred on the site.
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    assert sum(isinstance(c, Poly3DCollection) for c in ax.collections) == 2
    labels = {t.get_text() for t in ax.texts}
    assert {"Na1", "Na2"} <= labels
    if region == "cell":
        assert {"a", "b", "c"} <= labels


def test_plot_tensor_centres_in_the_cell_frame(sample):
    # The tensor of an atom inside the cell is centred on it, after the
    # same rotation as the structure (a along x).
    model = D.dft_sites(sample, element="Na", delta_ref=7.2, sigma_ref=550.0)
    rec = next(r for r in model["atoms"] if r["label"] == "Na2")
    fig, _notes = S.plot_dft_structure(sample, rec, frame="cell", tensor="shift_aniso",
                                       region="cell")
    Q = D.standard_cell_rotation(np.array(LATTICE))
    expected = Q @ np.array([2.0, 3.0, 3.5])
    # The site's label sits 1.1 × tensor_size above the tensor's centre.
    label = next(t for t in fig.axes[0].texts if t.get_text() == "Na2")
    x, y = label.get_position()
    np.testing.assert_allclose([x, y, label._z], expected + [0, 0, 1.1], atol=1e-9)
    # And the glyph is the tensor about that centre: its extent along
    # the largest principal axis equals tensor_size.
    td = Q @ S.tensor_of(rec, "shift_aniso") @ Q.T
    gx, gy, gz, _sign = S.tensor_glyph(td, "shift_aniso",
                                       1.0 / np.max(np.abs(np.linalg.eigvalsh(td))), expected)
    r = np.linalg.norm(np.stack([gx, gy, gz], -1) - expected, axis=-1)
    assert r.max() == pytest.approx(1.0, rel=1e-3)


def test_molecule_without_cell(tmp_path):
    from test_dft_molecular import SIG_H, SIG_N, write_orca
    path = write_orca(tmp_path / "nh3.out", [("N", (0, 0, 0), SIG_N), ("H", (1.0, 0, 0), SIG_H)])
    calc = D.read_calculation(path)
    model = D.dft_sites(calc, element="N", sigma_ref=-130.0)
    with pytest.raises(ValueError, match="no cell"):
        S.plot_dft_structure(calc, model["atoms"], frame="cell")
    fig, notes = S.plot_dft_structure(calc, model["atoms"], frame="cartesian", tensor="shift")
    assert "molecule" in notes[0]
    assert S.find_bonds(S.structure_atoms(calc)) == [(0, 1)]


def test_gui_shows_the_structure(qapp, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    for name in ("critical", "information", "warning"):
        monkeypatch.setattr(QMessageBox, name, lambda *a, **k: None)
    from opennmr.gui.main_window import MainWindow
    R1, R2 = _rotation(1), _rotation(2)
    path = write_magres(tmp_path / "s.magres", [
        ("Na", 1, (0.0, 0.0, 0.0), _tensor(SIG_NA1, R1), _tensor(EFG1, R2)),
        ("Na", 2, (2.0, 3.0, 3.5), _tensor(SIG_NA1, R2), _tensor(EFG1, R1)),
        ("Cl", 1, (2.6, 0.0, 0.0), np.eye(3) * 980.0, np.zeros((3, 3)))])
    window = MainWindow()
    panel = window.dft_sim_panel
    panel.dft_sample_edit.setText(path)
    assert panel.read_files(quiet=True)
    panel.dft_sigma_override_check.setChecked(True)
    panel.dft_sigma_ref_spin.setValue(550.0)
    assert panel.calculate(simulate=False)
    panel.dft_tensor_style_combo.setCurrentIndex(
        panel.dft_tensor_style_combo.findData("structure"))
    assert not panel.dft_structure_tensor_combo.isHidden()
    assert panel.dft_relative_combo.isHidden()
    # Radius only for the radius region.
    assert panel.dft_structure_radius_spin.isHidden()
    panel.dft_structure_region_combo.setCurrentIndex(
        panel.dft_structure_region_combo.findData("radius"))
    assert not panel.dft_structure_radius_spin.isHidden()
    panel.dft_structure_sites_combo.setCurrentIndex(
        panel.dft_structure_sites_combo.findData("all"))
    assert panel.show_tensors(), panel.dft_tensor_notes_label.text()
    texts = {t.get_text() for ax in window.simulation_canvas.figure.axes for t in ax.texts}
    assert {"Na1", "Na2"} <= texts
    assert "Scale:" in panel.dft_tensor_notes_label.text()
    args = panel.structure_arguments()
    assert args["region"] == "radius" and args["tensor"] == "shift_aniso"
    # Back to the old views still works.
    panel.dft_tensor_style_combo.setCurrentIndex(
        panel.dft_tensor_style_combo.findData("ellipsoid"))
    assert panel.dft_structure_tensor_combo.isHidden()
    assert panel.show_tensors()
