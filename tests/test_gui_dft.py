"""Simulate ▸ DFT tensors (user requests, 2026-10-06 and 2026-10-07):
import, reference compounds and per-element calibrations, G=0, sites,
simulation, tensor pictures, saved calibrations, state."""

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMessageBox

from opennmr.gui.main_window import MainWindow
from opennmr.notebook import dft_calibration as C
from opennmr.notebook import dft_nmr as D

from test_gui import qapp  # noqa: F401
from test_dft_nmr import (EFG1, SIG_NA1, _rotation, _tensor, write_magres,  # noqa: F401
                          write_outcar)


@pytest.fixture
def said(monkeypatch):
    out = []
    for name in ("critical", "information", "warning"):
        monkeypatch.setattr(QMessageBox, name,
                            lambda *a, _n=name, **k: out.append((_n, a[2] if len(a) > 2 else "")))
    return out


@pytest.fixture
def paths(tmp_path):
    R1, R2 = _rotation(1), _rotation(2)
    sample = write_magres(tmp_path / "sample.magres", [
        ("Na", 1, (0, 0, 0), _tensor(SIG_NA1, R1), _tensor(EFG1, R2)),
        ("Na", 2, (0, 3.0, 0), _tensor(SIG_NA1, R1), _tensor(EFG1, R2)),
        ("Cd", 1, (2.5, 0, 0), _tensor((3300.0, 3400.0, 3500.0), R2), np.zeros((3, 3))),
    ])
    ref = write_magres(tmp_path / "ref.magres", [
        ("Na", 1, (0, 0, 0), np.eye(3) * 550.0, np.zeros((3, 3))),
        ("Cl", 1, (2.8, 0, 0), np.eye(3) * 980.0, np.zeros((3, 3)))])
    ref2 = write_magres(tmp_path / "ref2.magres", [
        ("Na", 1, (0, 0, 0), np.eye(3) * 561.0, np.zeros((3, 3)))])
    ref3 = write_magres(tmp_path / "ref3.magres", [
        ("Na", 1, (0, 0, 0), np.eye(3) * 530.5, np.zeros((3, 3)))])
    cd_ref = write_magres(tmp_path / "cdref.magres", [
        ("Cd", 1, (0, 0, 0), np.eye(3) * 3500.0, None)])
    return {"sample": sample, "ref": ref, "ref2": ref2, "ref3": ref3, "cd": cd_ref,
            "lib": str(tmp_path / "lib.json")}


def _panel(paths, delta=7.2):
    window = MainWindow()
    panel = window.dft_sim_panel
    panel.library_path = paths["lib"]
    panel.refresh_saved_calibrations()
    panel.dft_sample_edit.setText(paths["sample"])
    assert panel.read_files(quiet=True), panel.dft_files_label.text()
    if paths.get("ref"):
        assert panel.add_reference_file(paths["ref"], element="Na") == 1
        panel.set_experimental_shift(0, delta)
    return window, panel


def test_the_section_is_in_simulate_and_starts_empty(qapp):
    window = MainWindow()
    assert window._dft_sim_section.title() == "DFT tensors (CASTEP, VASP, ORCA, Gaussian)"
    panel = window.dft_sim_panel
    assert panel.site_dicts == [] and not panel.dft_calculate_button.isEnabled()
    assert panel.ref_points == [] and panel.dft_reference_table.rowCount() == 0
    assert "dft" in window._simulation_panels()


def test_import_calculate_and_simulate(qapp, said, paths):
    window, panel = _panel(paths)
    assert [panel.dft_element_combo.itemText(i)
            for i in range(panel.dft_element_combo.count())] == ["Na", "Cd"]
    assert panel.dft_isotope_combo.currentText() == "23Na"
    assert panel.checked_labels() == ["Na1", "Na2"]
    assert panel.dft_reference_table.item(0, 4).text() == "550.0000"
    assert "σ_calc = 557.200" in panel.dft_calibration_label.text()
    panel.regime_combo.setCurrentIndex(panel.regime_combo.findData("mas"))
    panel.mas_spin.setValue(12500)
    assert panel.calculate(), said
    assert panel.nucleus_combo.currentText() == "23Na"
    assert len(panel.site_dicts) == 1 and panel.site_dicts[0]["amount"] == 2.0  # merged
    assert panel.dft_result_table.rowCount() == 2
    iso = np.mean(SIG_NA1)
    assert panel.model["atoms"][0]["iso"] == pytest.approx(557.2 - iso)
    assert panel.spectrum is not None and np.max(panel.spectrum.data) > 0
    assert any("Shift scale" in line for line in panel.dft_notes_label.text().splitlines())
    assert panel.diso_spin.value() == pytest.approx(panel.model["atoms"][0]["iso"], abs=1e-3)


def test_several_references_fit_the_slope_and_give_error_bars(qapp, said, paths):
    window, panel = _panel(paths)
    panel.add_reference_file(paths["ref2"], element="Na")
    panel.add_reference_file(paths["ref3"], element="Na")
    panel.set_experimental_shift(1, -3.0)
    panel.set_experimental_shift(2, 26.1)
    cal = panel.calibrations()["Na"]
    expected = C.fit_calibration([C.typed_point("Na", s, d) for s, d in
                                  ((550.0, 7.2), (561.0, -3.0), (530.5, 26.1))])
    assert cal["n"] == 3 and cal["free_slope"]
    assert cal["slope"] == pytest.approx(expected["slope"])
    assert panel.dft_reference_table.item(0, 6).text() not in ("", "—")      # residual shown
    assert panel.calculate(simulate=False)
    rec = panel.model["atoms"][0]
    assert rec["iso"] == pytest.approx((np.mean(SIG_NA1) - cal["intercept"]) / cal["slope"])
    assert rec["iso_err"] is not None and rec["iso_err"] > 0
    assert panel.dft_result_table.item(0, 2).text() not in ("—", "")          # ± δiso
    # untick one row: two references, exact line, no error bars
    panel.dft_reference_table.item(2, 0).setCheckState(Qt.CheckState.Unchecked)
    assert panel.calibrations()["Na"]["n"] == 2
    panel.dft_slope_mode_combo.setCurrentIndex(panel.dft_slope_mode_combo.findData("fixed"))
    assert panel.calibrations()["Na"]["slope"] == -1.0


def test_each_element_uses_its_own_references(qapp, said, paths):
    window, panel = _panel(paths)
    assert panel.add_reference_file(paths["cd"], element="*") == 1
    panel.dft_element_combo.setCurrentText("Cd")
    assert panel.dft_isotope_combo.currentText() == "113Cd"
    assert not panel.calculate(simulate=False) and "No shift scale for Cd" in said[-1][1]
    panel.set_experimental_shift(1, 0.0)
    assert set(panel.calibrations()) == {"Na", "Cd"}
    assert panel.calculate(simulate=False)
    assert panel.model["atoms"][0]["iso"] == pytest.approx(3500.0 - 3400.0)
    panel.dft_element_combo.setCurrentText("Na")
    assert panel.calculate(simulate=False)
    assert panel.model["atoms"][0]["iso"] == pytest.approx(557.2 - np.mean(SIG_NA1))


def test_a_typed_scale_replaces_the_references(qapp, said, paths):
    window, panel = _panel(dict(paths, ref=None))
    panel.dft_element_combo.setCurrentText("Cd")
    assert not panel.calculate(simulate=False)
    panel.dft_sigma_override_check.setChecked(True)
    panel.dft_sigma_ref_spin.setValue(3400.0)
    panel.dft_delta_ref_spin.setValue(10.0)
    assert panel.calculate(simulate=False)
    assert panel.model["atoms"][0]["iso"] == pytest.approx(10.0)
    assert panel.site_dicts[0]["interactions"][0]["type"] == "csa"


def test_rows_merge_for_a_methyl_group_and_can_be_removed(qapp, said, paths, tmp_path):
    R = _rotation(4)
    ala = write_magres(tmp_path / "ala.magres", [
        ("H", 1, (0, 0, 0), _tensor((28.0, 30.0, 34.0), R), None),
        ("H", 2, (1, 0, 0), _tensor((27.0, 30.5, 34.5), R), None),
        ("H", 3, (0, 1, 0), _tensor((28.5, 29.5, 33.0), R), None)])
    window, panel = _panel(paths)
    assert panel.add_reference_file(ala, element="H") == 3
    assert panel.dft_reference_table.rowCount() == 4
    assert panel.merge_selected_rows([1, 2, 3])
    assert panel.dft_reference_table.rowCount() == 2
    merged = panel.ref_points[1]
    assert sorted(merged["labels"]) == ["H1", "H2", "H3"]
    assert merged["sigma"] == pytest.approx(np.mean([np.mean(v) for v in
                                                     ((28, 30, 34), (27, 30.5, 34.5),
                                                      (28.5, 29.5, 33))]))
    panel.remove_selected_rows([1])
    assert [p["element"] for p in panel.ref_points] == ["Na"]
    assert not panel.merge_selected_rows([0]) and "at least two" in said[-1][1]


def test_g0_applies_to_sample_and_references_at_once(qapp, said, tmp_path):
    """The bug the user found: excluding G=0 changed the sample only. Now
    toggling re-reads every VASP file, and the shifts follow."""
    R = _rotation(7)

    def outcar(name, valence_iso, g0_iso):
        valence = [-_tensor((valence_iso - 5, valence_iso, valence_iso + 5), R)]
        g0 = np.eye(3) * g0_iso
        return write_outcar(tmp_path / name, ["Na"], [1], [(0, 0, 0)], valence,
                            [np.diag([1.0, 2.0, -3.0])], g0, g0 / 0.08, -10.0, {"Na": -400.0})
    sample = outcar("OUTCAR_s", 140.0, -9.0)
    ref = outcar("OUTCAR_r", 150.0, -11.0)
    window, panel = _panel({"sample": sample, "ref": None, "lib": str(tmp_path / "l.json")})
    panel.add_reference_file(ref, element="Na")
    panel.set_experimental_shift(0, 7.2)
    assert panel.calculate(simulate=False)
    with_g0 = panel.model["atoms"][0]["iso"]
    sigma_ref_in = panel.ref_points[0]["sigma"]
    panel.dft_g0_check.setChecked(False)            # no Read Files needed
    assert panel.calcs["sample"]["g0"] == "excluded"
    assert panel.ref_points[0]["g0"] == "excluded"
    sigma_ref_out = panel.ref_points[0]["sigma"]
    k_core = -10.0 * 0.08                           # k χ_core, also in VASP's G=0 total
    assert sigma_ref_in - sigma_ref_out == pytest.approx(-(-11.0 + k_core), abs=1e-4)
    without = panel.model["atoms"][0]["iso"]
    # σ = −(valence + G0 + core) and δ = δ_ref + σ_ref − σ, so the G=0
    # terms enter δ as G0_sample − G0_ref
    assert with_g0 - without == pytest.approx((-9.0 + k_core) - (-11.0 + k_core), abs=1e-4)
    # a CASTEP reference cannot follow: say so
    castep = write_magres(tmp_path / "c.magres", [("Na", 1, (0, 0, 0), np.eye(3) * 550, None)])
    panel.add_reference_file(castep, element="Na")
    panel.read_files(quiet=True)
    assert "keep theirs" in panel.dft_files_label.text()


def test_calibration_plot_save_load_delete(qapp, said, paths):
    window, panel = _panel(paths)
    panel.add_reference_file(paths["ref2"], element="Na")
    panel.set_experimental_shift(1, -3.0)
    assert panel.plot_calibration()
    axes = window.simulation_canvas.figure.axes
    assert axes and "sigma" in axes[0].get_ylabel()
    assert panel.save_calibration("Na test")
    assert panel.dft_saved_combo.count() == 1 and "Na test (Na, n = 2)" in \
        panel.dft_saved_combo.itemText(0)
    other = MainWindow().dft_sim_panel
    other.library_path = paths["lib"]
    other.refresh_saved_calibrations()
    other.dft_sample_edit.setText(paths["sample"])
    assert other.read_files(quiet=True)
    assert other.load_saved_calibration()
    assert other.calibrations()["Na"]["slope"] == pytest.approx(
        panel.calibrations()["Na"]["slope"])
    assert other.calculate(simulate=False)
    assert other.delete_saved_calibration() and other.dft_saved_combo.count() == 0


def test_state_round_trip_and_an_old_state(qapp, said, paths):
    window, panel = _panel(paths)
    panel.dft_site_list.item(1).setCheckState(Qt.CheckState.Unchecked)
    assert panel.calculate(simulate=False)
    state = panel.get_state()
    assert state["dft"]["checked"] == ["Na1"]
    assert state["dft"]["references"][0]["delta_exp"] == 7.2
    assert "557.200" in state["extra"]["DFT shift scale"]
    other = MainWindow().dft_sim_panel
    other.set_state(state)
    assert other.dft_sample_edit.text() == paths["sample"]
    assert other.checked_labels() == ["Na1"]
    assert [s["label"] for s in other.site_dicts] == ["Na1"]
    assert other.calibrations()["Na"]["intercept"] == pytest.approx(557.2)
    # a state saved by round 6: one reference file and δ_ref
    old = dict(state)
    old["dft"] = {k: v for k, v in state["dft"].items() if k != "references"}
    old["widgets"] = dict(state["widgets"],
                          dft_reference_edit={"kind": "line", "value": paths["ref"]},
                          dft_delta_ref_spin={"kind": "spin", "value": 7.2})
    third = MainWindow().dft_sim_panel
    third.set_state(old)
    assert third.calibrations()["Na"]["intercept"] == pytest.approx(557.2)


def test_unreadable_file_is_reported(qapp, said, tmp_path):
    bad = tmp_path / "x.txt"
    bad.write_text("not dft\n")
    window = MainWindow()
    panel = window.dft_sim_panel
    panel.dft_sample_edit.setText(str(bad))
    assert not panel.read_files(quiet=True)
    assert "not a calculation openNMR reads" in panel.dft_files_label.text()
    assert panel.add_reference_file(str(bad)) == 0
    assert "Could not add" in panel.dft_calibration_label.text()


def test_copy_buttons_give_code_that_reproduces_the_import(qapp, said, paths):
    from PySide6.QtGui import QGuiApplication
    window, panel = _panel(paths)
    assert panel.calculate(simulate=False)
    panel.dft_csv_button.click()
    assert QGuiApplication.clipboard().text().startswith("Site,δiso (ppm),± δiso (ppm)")
    panel.dft_python_button.click()
    code = QGuiApplication.clipboard().text()
    assert "dft_nmr.dft_sites(sample, calibration=cal" in code
    ns = {}
    exec(code.replace("print(", "(lambda *a: None)("), ns)
    assert ns["model"]["atoms"][0]["iso"] == pytest.approx(panel.model["atoms"][0]["iso"])


def test_tensor_picture_in_cell_and_cartesian_frames(qapp, said, paths):
    window, panel = _panel(paths)
    assert panel.calculate(simulate=False)
    assert panel.dft_show_tensor_button.isEnabled()
    assert panel.show_tensors(), panel.dft_tensor_notes_label.text()
    texts = {t.get_text() for ax in window.simulation_canvas.figure.axes for t in ax.texts}
    assert {"a", "b", "c"} <= texts
    panel.dft_frame_combo.setCurrentIndex(panel.dft_frame_combo.findData("cartesian"))
    panel.dft_tensor_style_combo.setCurrentIndex(panel.dft_tensor_style_combo.findData("surface"))
    assert panel.show_tensors()


def test_a_molecular_file_forces_the_cartesian_frame(qapp, said, tmp_path):
    from test_dft_molecular import SIG_H, SIG_N, write_orca
    path = write_orca(tmp_path / "nh3.out", [("N", (0, 0, 0), SIG_N), ("H", (1, 0, 0), SIG_H)])
    window = MainWindow()
    panel = window.dft_sim_panel
    panel.dft_sample_edit.setText(path)
    assert panel.read_files(quiet=True)
    assert "ORCA" in panel.dft_files_label.text()
    assert panel.dft_frame_combo.currentData() == "cartesian"
    assert not panel.dft_frame_combo.model().item(0).isEnabled()
    panel.dft_sigma_override_check.setChecked(True)
    panel.dft_sigma_ref_spin.setValue(-130.0)
    assert panel.calculate(simulate=False)
    assert panel.show_tensors()
    _ = D


# ---- solution / solid and motional averaging (2026-10-24) --------------------------------

def _methyl_paths(tmp_path):
    def rz(deg):
        t = np.radians(deg)
        return np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1.0]])
    atoms = [("H", i + 1, (np.cos(np.radians(120 * i)), np.sin(np.radians(120 * i)), 0.0),
              _tensor((24.0, 28.0, 36.0), rz(120 * i)), None) for i in range(3)]
    atoms.append(("H", 4, (5.0, 5.0, 5.0), np.diag([20.0, 20.0, 20.0]), None))
    sample = write_magres(tmp_path / "ch3.magres", atoms, lattice=None)
    ref = write_magres(tmp_path / "href.magres", [("H", 1, (0, 0, 0), np.eye(3) * 30.0, None)])
    return {"sample": sample, "ref": ref, "lib": str(tmp_path / "lib.json")}


def test_state_and_averaging_controls(qapp, said, tmp_path):  # noqa: F811
    paths = _methyl_paths(tmp_path)
    window = MainWindow()
    panel = window.dft_sim_panel
    panel.library_path = paths["lib"]
    panel.dft_sample_edit.setText(paths["sample"])
    assert panel.read_files(quiet=True)
    panel.dft_element_combo.setCurrentText("H")
    assert panel.add_reference_file(paths["ref"], element="H") == 1
    panel.set_experimental_shift(0, 0.0)
    assert panel.dft_state_combo.currentData() == "solid"
    assert panel.average_selected(["H1"]) is False                      # one site: told so
    assert panel.average_selected(["H1", "H2", "H3"]) is True
    assert panel.dft_average_list.count() == 1
    assert panel.average_selected(["H3", "H4"]) is False                # H3 already grouped
    assert panel.calculate(simulate=False)
    labels = [r["label"] for r in panel.model["atoms"]]
    assert labels == ["H1–H3", "H4"]
    assert panel.model["atoms"][0]["eta"] == pytest.approx(0.0, abs=1e-9)
    assert "python" in panel.dft_python().lower() or "average=" in panel.dft_python()
    assert "average=[['H1', 'H2', 'H3']]" in panel.dft_python()
    # solution: isotropic only, CSA switches disabled
    panel.dft_state_combo.setCurrentIndex(panel.dft_state_combo.findData("solution"))
    assert not panel.dft_include_csa_check.isEnabled()
    assert panel.calculate(simulate=False)
    assert panel.model["state"] == "solution"
    assert all(r["aniso"] is None for r in panel.model["atoms"])
    assert "state='solution'" in panel.dft_python()
    # state survives get/set
    saved = panel.get_state()
    assert saved["dft"]["state"] == "solution" and saved["dft"]["average"] == [["H1", "H2", "H3"]]
    panel.remove_average_group(0)
    assert panel.dft_average_list.count() == 0
    panel.set_state(saved)
    assert panel.dft_state_combo.currentData() == "solution"
    assert panel.average_groups == [["H1", "H2", "H3"]]
    assert said == [("information", "Select at least two sites in the site list to average."),
                    ("information", "H3 is already in a group; remove that group first.")]


def test_average_from_the_results_table_recalculates_at_once(qapp, said, tmp_path):  # noqa: F811
    paths = _methyl_paths(tmp_path)
    window = MainWindow()
    panel = window.dft_sim_panel
    panel.library_path = paths["lib"]
    panel.dft_sample_edit.setText(paths["sample"])
    assert panel.read_files(quiet=True)
    panel.dft_element_combo.setCurrentText("H")
    panel.dft_sigma_override_check.setChecked(True)          # a typed scale, no reference file
    panel.dft_sigma_ref_spin.setValue(30.0)
    assert panel.calculate(simulate=False)
    assert [r["label"] for r in panel.model["atoms"]] == ["H1", "H2", "H3", "H4"]
    panel.dft_result_table.selectRow(0)
    panel.dft_result_table.selectRow(1)                       # replaces; extend with Ctrl below
    from PySide6.QtCore import QItemSelectionModel
    sel = panel.dft_result_table.selectionModel()
    for r in (0, 1, 2):
        sel.select(panel.dft_result_table.model().index(r, 0),
                   QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows)
    assert panel.average_selected()
    assert [r["label"] for r in panel.model["atoms"]] == ["H1–H3", "H4"]   # recalculated
    assert panel.dft_result_table.rowCount() == 2
    panel.dft_average_list.setCurrentRow(0)
    panel.remove_average_group()
    assert [r["label"] for r in panel.model["atoms"]] == ["H1", "H2", "H3", "H4"]


# ---------------------------------------------------------------------------
# CRYSTAL: paramagnetic scaling (2026-10-24)
# ---------------------------------------------------------------------------

def test_crystal_file_shows_the_scaling_controls_and_gives_shifts_and_cq(qapp, said, tmp_path):  # noqa: F811
    from test_crystal_nmr import crystal_text
    path = tmp_path / "properties.outp"
    path.write_text(crystal_text())
    window = MainWindow()
    panel = window.dft_sim_panel
    panel.dft_sample_edit.setText(str(path))
    assert panel.read_files(quiet=True), panel.dft_files_label.text()
    assert "CRYSTAL" in panel.dft_files_label.text()
    assert panel.dft_pm_box.isVisibleTo(panel)
    panel.dft_element_combo.setCurrentText("Li")
    assert panel.dft_site_list.count() == 2                  # the raw calculation has no σ yet
    # three routes, each showing only its own rows
    panel.dft_pm_route_combo.setCurrentIndex(panel.dft_pm_route_combo.findData("curie_weiss"))
    panel.dft_pm_theta_spin.setValue(-300.0)
    panel.dft_pm_mueff_spin.setValue(3.87)
    panel.dft_pm_spin_spin.setValue(1.5)
    panel.dft_pm_T_spin.setValue(318.0)
    panel.dft_pm_B0_spin.setValue(11.7)
    panel._on_pm_changed()
    assert "0.0211" in panel.dft_pm_phi_label.text()
    assert panel.calculate(simulate=False), said
    model = panel.model
    assert model["state"] == "solid" and model["atoms"][0]["C_Q"] is not None
    assert model["atoms"][0]["iso"] > 0
    first = model["atoms"][0]["iso"]
    # the orbital shift and the contact-only choice recalculate at once
    panel.dft_pm_orbital_spin.setValue(10.0)
    panel._on_pm_changed()
    assert panel.model["atoms"][0]["iso"] == pytest.approx(first + 10.0)
    panel.dft_pm_components_combo.setCurrentIndex(panel.dft_pm_components_combo.findData("contact"))
    assert panel.model["atoms"][0]["span"] == pytest.approx(0.0, abs=1e-9)
    # the susceptibility route and a typed φ give a sensible, different message
    panel.dft_pm_route_combo.setCurrentIndex(panel.dft_pm_route_combo.findData("susceptibility"))
    panel.dft_pm_chi_spin.setValue(0.02)
    panel._on_pm_changed()
    assert "χ_M" in panel.dft_pm_phi_label.text()
    panel.dft_pm_route_combo.setCurrentIndex(panel.dft_pm_route_combo.findData("phi"))
    panel.dft_pm_phi_spin.setValue(0.0211622)
    panel._on_pm_changed()
    assert "given" in panel.dft_pm_phi_label.text()
    # a Weiss constant above the temperature cannot give φ: said, not raised
    panel.dft_pm_route_combo.setCurrentIndex(panel.dft_pm_route_combo.findData("curie_weiss"))
    panel.dft_pm_theta_spin.setValue(400.0)
    panel._on_pm_changed()
    assert "cannot be found" in panel.dft_pm_phi_label.text()
    # the import code reproduces the import
    panel.dft_pm_theta_spin.setValue(-300.0)
    panel._on_pm_changed()
    code = panel.dft_python()
    assert "paramagnetic=" in code and "orbital_shift=" in code
    namespace = {}
    exec(code, namespace)                                   # runs on the user's file
    assert namespace["model"]["atoms"][0]["iso"] == pytest.approx(panel.model["atoms"][0]["iso"])
    window.close()


def test_crystal_scaling_survives_the_saved_state(qapp, said, tmp_path):  # noqa: F811
    from test_crystal_nmr import crystal_text
    path = tmp_path / "properties.out"
    path.write_text(crystal_text())
    window = MainWindow()
    panel = window.dft_sim_panel
    panel.dft_sample_edit.setText(str(path))
    assert panel.read_files(quiet=True)
    panel.dft_pm_route_combo.setCurrentIndex(panel.dft_pm_route_combo.findData("phi"))
    panel.dft_pm_phi_spin.setValue(0.0123)
    panel.dft_pm_orbital_spin.setValue(-4.0)
    state = panel.get_state()
    other = MainWindow().dft_sim_panel
    other.set_state(state)
    assert other.dft_pm_route_combo.currentData() == "phi"
    assert other.dft_pm_phi_spin.value() == pytest.approx(0.0123)
    assert other.dft_pm_orbital_spin.value() == pytest.approx(-4.0)
    assert other.dft_pm_box.isVisibleTo(other)
    window.close()
