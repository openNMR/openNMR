"""2026-11-10 round 2: hide removed points, insets as a Plot Style section,
and BMS cuboid particles, length units, the simulated region vs the sample
container, diamond and sapphire rotors, a nucleus list and packing tooltips."""
import numpy as np
import pytest
from PySide6.QtCore import Qt

from opennmr.gui.inset_dialog import InsetDialog, InsetPanel
from opennmr.gui.main_window import MainWindow
from opennmr.gui.paramagnetic_panel import ParamagneticToolsPanel
from opennmr.notebook.bms import fast as bmsf
from opennmr.notebook.bms import geometry, packing, sample, views

from test_gui import _fit_a_t2_curve, _load_spectrum_directly, qapp  # noqa: F401
from test_inset import _one_d


@pytest.fixture
def bms_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENNMR_BMS_CACHE", str(tmp_path / "sidebands"))


# ---- 1. hide removed points --------------------------------------------------------

def _crosses(window):
    return sum(1 for line in window.fit_canvas.ax.lines if line.get_marker() == "x")


def test_removed_points_can_be_hidden(qapp):
    window = MainWindow()
    _fit_a_t2_curve(window, np.geomspace(0.001, 0.3, 10), true_t2=0.05)
    original = window._relaxation_fit_original[0]
    window._on_fit_point_exclude_requested(float(original["t"][2]), float(original["y"][2]))
    box = window.curve_fit_plot_box
    assert box.show_removed() and _crosses(window) == 1
    box.show_removed_check.setChecked(False)
    assert _crosses(window) == 0
    assert window._excluded_relaxation_indices == {0: {2}}       # still out of the fit
    assert box.settings()["show_removed"] is False
    # All peaks together too.
    box.view_combo.setCurrentIndex(box.view_combo.findData("all"))
    assert _crosses(window) == 0
    box.show_removed_check.setChecked(True)
    assert _crosses(window) == 1


def test_show_removed_setting_round_trips(qapp):
    from opennmr.gui.curve_fit_plot_box import CurveFitPlotBox
    box = CurveFitPlotBox()
    box.apply_settings({"show_removed": False})
    assert not box.show_removed()
    box.apply_settings({})                                       # older settings: shown
    assert box.show_removed()


# ---- 2. insets in Plot Style ---------------------------------------------------------

def test_inset_controls_are_a_plot_style_section(qapp):
    window = MainWindow()
    _load_spectrum_directly(window, _one_d())
    canvas = window.canvas
    canvas.add_inset_region((5.3, 4.7), (0.0, 1.0))
    panel = window._on_open_inset_dialog(canvas)
    assert isinstance(panel, InsetPanel) and panel is window.style_inset_panel
    assert window.style_sections["Insets"].isExpanded()
    assert panel.canvas is canvas and panel.inset_list.count() == 1
    assert window._style_dialog.scroll.verticalScrollBarPolicy() == \
        Qt.ScrollBarPolicy.ScrollBarAsNeeded
    window.close()


def test_large_intensity_limits_are_not_clamped(qapp):
    """A 1.6e10 intensity axis was clamped to ±1e9 by the old spin boxes,
    and the inset went flat on the first edit."""
    window = MainWindow()
    _load_spectrum_directly(window, _one_d())
    canvas = window.canvas
    canvas.add_inset_region((5.3, 4.7), (-2.0e10, 1.6e10))
    panel = window._on_open_inset_dialog(canvas)
    assert panel.y_to.value() == pytest.approx(1.6e10)
    panel.bold_check.setChecked(True)
    assert panel.apply()
    assert canvas.inset_specs()[0]["ylim"] == pytest.approx((-2.0e10, 1.6e10))
    panel.y_to.setText("3.5e10")
    panel.y_to.editingFinished.emit()
    assert panel.apply()
    assert canvas.inset_specs()[0]["ylim"][1] == pytest.approx(3.5e10)
    window.close()


def test_inset_list_wraps_and_names_the_range(qapp):
    window = MainWindow()
    _load_spectrum_directly(window, _one_d())
    window.canvas.add_inset_region((5.3, 4.7), (0.0, 1.0))
    panel = window._on_open_inset_dialog(window.canvas)
    assert panel.inset_list.wordWrap()
    assert panel.inset_list.horizontalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAlwaysOff
    item = panel.inset_list.item(0)
    assert item.text() == "Inset 1: 5.3 to 4.7"
    assert "vertical" in item.toolTip()
    window.close()


def test_the_inset_window_still_works_and_scrolls(qapp):
    window = MainWindow()
    _load_spectrum_directly(window, _one_d())
    canvas = window.canvas
    canvas.add_inset_region((5.3, 4.7), (0.0, 1.0))
    dialog = InsetDialog(canvas)
    assert dialog.inset_list.count() == 1                 # passed to its panel
    dialog.clear_insets()
    assert canvas.inset_specs() == []
    window.close()


# ---- 3. BMS: cuboid particles ------------------------------------------------------

def _cuboid(v=(10.0, 5.0, 4.0)):
    return bmsf.prepare_particles([{"x": 0, "y": 0, "z": 0, "v1": v[0], "v2": v[1], "v3": v[2],
                                    "qi": 0, "qj": 0, "qk": 0, "qr": 1}],
                                  (1e-3, 1e-3, 1e-3), equation="cuboid")[0]


def test_cuboid_shape_function_and_volume():
    p = _cuboid()
    xs = np.linspace(-15.0, 15.0, 61)
    block = bmsf.block_shape(p, xs, xs, xs)
    X, Y, Z = np.meshgrid(xs, xs, xs, indexing="ij")
    np.testing.assert_allclose(block, geometry.shape_function(p, X, Y, Z), atol=1e-12)
    step = xs[1] - xs[0]
    assert block.sum() * step ** 3 == pytest.approx(8 * 10 * 5 * 4, rel=0.02)
    assert block[30, 30, 30] == pytest.approx(1.0)            # centre
    assert block[30 + 2 * 9, 30, 30] > 0.99                   # x = 9: inside
    assert block[30 + 2 * 11, 30, 30] < 1e-6                  # x = 11: outside
    assert block[30 + 2 * 9, 30 + 2 * 4, 30 + 2 * 3] > 0.99   # near a corner: inside


def test_cuboid_volumes_and_reach():
    p = _cuboid()
    assert p["vol"] == pytest.approx(1600.0)
    assert views.particle_volume(p) == pytest.approx(1600.0)
    assert packing._particle_volume("cuboid", 10, 5, 4) == pytest.approx(1600.0)
    assert packing._bounding_radius("cuboid", 10, 5, 4) == pytest.approx(np.sqrt(141.0))


def test_cuboid_packing_does_not_overlap():
    particles, stats = packing.random_close_pack(
        12, 120.0, (8.0, 8.0, 8.0), (0.0, 0.0, 0.0), equation="cuboid", seed=3, verbose=False)
    assert stats["n_placed"] >= 1
    centres = np.array([[p["x"], p["y"], p["z"]] for p in particles])
    reach = np.sqrt(3) * 8.0
    for i in range(len(centres)):
        for j in range(i + 1, len(centres)):
            assert np.linalg.norm(centres[i] - centres[j]) >= 2 * reach - 1e-9


def test_isotropic_cubes_have_no_isotropic_bms_shift(bms_cache):
    particles = [{"x": -20, "y": 0, "z": 0, "v1": 10, "v2": 10, "v3": 10,
                  "qi": 0, "qj": 0, "qk": 0, "qr": 1},
                 {"x": 20, "y": 5, "z": 0, "v1": 8, "v2": 12, "v3": 6,
                  "qi": 0.3, "qj": 0.1, "qk": 0.0, "qr": 0.95}]
    field = bmsf.bms_field(particles, (1e-3, 1e-3, 1e-3), box_size=100.0, gridpoints=64,
                           equation="cuboid", components=("zz", "iso"), dtype="float64")
    # Exactly zero up to rounding: the Lorentz term is applied in
    # proportion to each voxel's occupancy (lorentz="proportional").
    scale = np.max(np.abs(field["delta_zz"]))
    assert scale > 1.0                                   # the anisotropic part is there
    assert np.max(np.abs(field["delta_iso"])) < 1e-6 * scale


def test_cuboid_packing_draws_in_3d():
    import matplotlib.pyplot as plt
    particles, _stats = packing.random_close_pack(
        4, 120.0, (8.0, 6.0, 4.0), (0.0, 0.0, 0.0), equation="cuboid", seed=1, verbose=False)
    fig = views.plot_packing_3d(particles, box_size=120.0, equation="cuboid", unit="µm")
    assert fig.axes
    plt.close(fig)


# ---- 4. BMS: rotor materials --------------------------------------------------------

def test_diamond_and_sapphire_rotor_materials():
    assert sample.MATERIALS["diamond"][0] == pytest.approx(-21.7e-6)
    assert sample.MATERIALS["sapphire"][0] == pytest.approx(-18.1e-6)
    assert sample.material_chi("diamond") == pytest.approx(-21.7e-6)
    panel = ParamagneticToolsPanel()
    for key in ("diamond", "sapphire"):
        assert panel.pm_bms_wall_combo.findData(key) >= 0
    panel.pm_bms_wall_combo.setCurrentIndex(panel.pm_bms_wall_combo.findData("diamond"))
    panel.pm_bms_wall_combo.activated.emit(panel.pm_bms_wall_combo.currentIndex())
    assert panel.pm_bms_wall_chi_spin.value() == pytest.approx(-21.7)
    assert panel.bms_sample().rotor["chi_wall"] == pytest.approx(-21.7e-6)


# ---- 5. BMS GUI: units, region, cuboid sizes, nucleus, tooltips ------------------------

def test_length_unit_converts_and_labels(qapp):
    t = ParamagneticToolsPanel()
    assert t.bms_length_unit() == "µm"
    t.pm_bms_region_mode_combo.setCurrentIndex(t.pm_bms_region_mode_combo.findData("side"))
    t.pm_bms_box_spin.setValue(300.0)
    t.pm_bms_size_edit.setText("25")
    t.pm_bms_length_unit_combo.setCurrentIndex(t.pm_bms_length_unit_combo.findData("nm"))
    assert t.pm_bms_size_edit.text() == "25000"
    assert t.pm_bms_box_spin.value() == pytest.approx(300000.0)
    assert t.pm_bms_box_spin.suffix() == " nm"
    label = t._bms_particle_form.labelForField(t.pm_bms_size_edit).text()
    assert "(nm)" in label and "semi-axes" in label


def test_region_side_follows_the_packing_fraction(qapp):
    t = ParamagneticToolsPanel()
    assert t.pm_bms_region_mode_combo.currentData() == "fraction"
    t.pm_bms_n_spin.setValue(20)
    t.pm_bms_size_edit.setText("10")
    t.pm_bms_fraction_spin.setValue(0.4)
    expected = (20 * 4.0 / 3.0 * np.pi * 1000.0 / 0.4) ** (1.0 / 3.0)
    assert t.pm_bms_box_spin.value() == pytest.approx(expected, rel=1e-3)
    assert not t.pm_bms_box_spin.isEnabled()
    assert "cube of the powder" in t.pm_bms_region_note.text()
    # Typing a side switches to it.
    t.pm_bms_region_mode_combo.setCurrentIndex(t.pm_bms_region_mode_combo.findData("side"))
    t.pm_bms_box_spin.setValue(500.0)
    t.pm_bms_n_spin.setValue(30)
    assert t.pm_bms_box_spin.value() == pytest.approx(500.0)


def test_typing_a_region_side_switches_mode(qapp):
    t = ParamagneticToolsPanel()
    t.pm_bms_box_spin.setValue(222.0)
    assert t.pm_bms_region_mode_combo.currentData() == "side"


def test_region_bigger_than_the_container_is_flagged(qapp):
    t = ParamagneticToolsPanel()
    t.pm_bms_length_unit_combo.setCurrentIndex(t.pm_bms_length_unit_combo.findData("mm"))
    t.pm_bms_box_spin.setValue(50.0)                     # 50 mm in a 2.2 mm rotor
    assert "larger than the container" in t.pm_bms_region_note.text()


def test_cuboid_sizes_are_edge_lengths(qapp):
    t = ParamagneticToolsPanel()
    t.pm_bms_equation_combo.setCurrentIndex(t.pm_bms_equation_combo.findData("cuboid"))
    t.pm_bms_size_edit.setText("20, 10, 8")
    t.pm_bms_spread_edit.setText("2")
    kw = t.bms_packing_arguments()
    assert kw["mean_size"] == (10.0, 5.0, 4.0) and kw["spread_size"] == (1.0, 1.0, 1.0)
    assert "edge lengths" in t._bms_particle_form.labelForField(t.pm_bms_size_edit).text()
    assert not t._bms_particle_form.isRowVisible(t.pm_bms_deform_combo)
    assert "half its edge" in t.bms_python()


def test_cuboid_simulates(qapp, bms_cache):
    spectra = []
    t = ParamagneticToolsPanel(plot_spectrum=spectra.append)
    t.pm_bms_equation_combo.setCurrentIndex(t.pm_bms_equation_combo.findData("cuboid"))
    t.pm_bms_n_spin.setValue(8)
    t.pm_bms_size_edit.setText("20")
    t.pm_bms_grid_combo.setCurrentIndex(t.pm_bms_grid_combo.findData(64))
    t.pm_bms_mode_combo.setCurrentIndex(t.pm_bms_mode_combo.findData("mas_infinite"))
    t.pm_bms_sample_combo.setCurrentIndex(t.pm_bms_sample_combo.findData("sphere"))
    assert t.simulate_bms()
    assert t.bms_field["parameters"]["equation"] == "cuboid"
    assert spectra and "µm region" in t._packing_text()


def test_nucleus_is_a_list(qapp):
    from PySide6.QtWidgets import QComboBox
    t = ParamagneticToolsPanel()
    combo = t.pm_bms_nucleus_combo
    assert isinstance(combo, QComboBox) and not combo.isEditable()
    assert combo.currentText() == "7Li"
    assert combo.findText("23Na") >= 0 and combo.findText("1H") >= 0
    combo.setCurrentIndex(combo.findText("23Na"))
    assert t.bms_spectrum_arguments()["nuclide"] == "23Na"


def test_packing_methods_are_described(qapp):
    t = ParamagneticToolsPanel()
    combo = t.pm_bms_packer_combo
    for i in range(combo.count()):
        tip = combo.itemData(i, Qt.ItemDataRole.ToolTipRole)
        assert tip and len(tip) > 80
    assert "Grow" in combo.toolTip() and "bounding sphere" in combo.toolTip()
    combo.setCurrentIndex(combo.findData("rsa"))
    assert "sequential" in combo.toolTip()


def test_container_dimensions_are_named(qapp):
    t = ParamagneticToolsPanel()
    form = t._bms_sample_form
    assert form.labelForField(t.pm_bms_inner_spin).text() == "Rotor inner diameter:"
    t.pm_bms_sample_combo.setCurrentIndex(t.pm_bms_sample_combo.findData("cylinder"))
    assert form.labelForField(t.pm_bms_inner_spin).text() == "Tube inner diameter:"
    assert form.labelForField(t.pm_bms_length_spin).text() == "Filled height:"


def test_bms_unit_and_sizes_survive_state(qapp):
    t = ParamagneticToolsPanel()
    t.pm_bms_length_unit_combo.setCurrentIndex(t.pm_bms_length_unit_combo.findData("nm"))
    t.pm_bms_equation_combo.setCurrentIndex(t.pm_bms_equation_combo.findData("cuboid"))
    t.pm_bms_size_edit.setText("80")
    t.pm_bms_box_spin.setValue(900.0)
    t.pm_bms_nucleus_combo.setCurrentIndex(t.pm_bms_nucleus_combo.findText("6Li"))
    u = ParamagneticToolsPanel()
    u.set_state(t.get_state())
    assert u.pm_bms_length_unit_combo.currentData() == "nm"
    assert u.pm_bms_size_edit.text() == "80"                    # not converted again
    assert u.pm_bms_box_spin.value() == pytest.approx(900.0)
    assert u.pm_bms_region_mode_combo.currentData() == "side"
    assert u.pm_bms_nucleus_combo.currentText() == "6Li"
