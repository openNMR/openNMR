"""Simulate ▸ DFT tensors (CASTEP / VASP) (user request, 2026-10-06)."""

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMessageBox

from opennmr.gui.main_window import MainWindow

from test_gui import qapp  # noqa: F401
from test_dft_nmr import EFG1, SIG_NA1, _rotation, _tensor, write_magres  # noqa: F401


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
        ("Na", 1, (0, 0, 0), np.eye(3) * 550.0, np.zeros((3, 3)))])
    return sample, ref


def _panel(qapp, paths):
    window = MainWindow()
    panel = window.dft_sim_panel
    panel.dft_sample_edit.setText(paths[0])
    panel.dft_reference_edit.setText(paths[1])
    assert panel.read_files(quiet=True), panel.dft_files_label.text()
    return window, panel


def test_the_section_is_in_simulate_and_starts_empty(qapp):
    window = MainWindow()
    assert window._dft_sim_section.title().startswith("DFT tensors")
    panel = window.dft_sim_panel
    assert panel.site_dicts == [] and not panel.dft_calculate_button.isEnabled()
    assert "dft" in window._simulation_panels()


def test_import_calculate_and_simulate(qapp, said, paths):
    window, panel = _panel(qapp, paths)
    assert [panel.dft_element_combo.itemText(i)
            for i in range(panel.dft_element_combo.count())] == ["Na", "Cd"]
    assert panel.dft_isotope_combo.currentText() == "23Na"
    assert panel.checked_labels() == ["Na1", "Na2"]
    assert "550.0000 ppm" in panel.dft_sigma_ref_label.text()
    panel.dft_delta_ref_spin.setValue(7.2)
    panel.regime_combo.setCurrentIndex(panel.regime_combo.findData("mas"))
    panel.mas_spin.setValue(12500)
    assert panel.calculate(), said
    assert panel.nucleus_combo.currentText() == "23Na"
    assert len(panel.site_dicts) == 1 and panel.site_dicts[0]["amount"] == 2.0  # merged
    assert panel.site_dicts[0]["label"] == "Na1–Na2"
    assert panel.dft_result_table.rowCount() == 2
    assert panel.spectrum is not None and np.max(panel.spectrum.data) > 0
    assert "Simulated" in panel.results_label.text()
    assert any("σ_ref" in line for line in panel.dft_notes_label.text().splitlines())
    # The sites are ordinary Combined-Hamiltonian sites: editable here.
    assert panel.diso_spin.value() == pytest.approx(panel.model["atoms"][0]["iso"], abs=1e-3)


def test_unticked_sites_are_left_out_and_merge_can_be_off(qapp, said, paths):
    window, panel = _panel(qapp, paths)
    panel.dft_site_list.item(1).setCheckState(Qt.CheckState.Unchecked)
    assert panel.calculate(simulate=False)
    assert [a["label"] for a in panel.model["atoms"]] == ["Na1"]
    panel._check_all(True)
    panel.dft_merge_check.setChecked(False)
    assert panel.calculate(simulate=False)
    assert len(panel.site_dicts) == 2
    panel._check_all(False)
    assert not panel.calculate(simulate=False) and "Tick at least one" in said[-1][1]


def test_an_element_missing_from_the_reference_needs_a_typed_sigma_ref(qapp, said, paths):
    window, panel = _panel(qapp, paths)
    panel.dft_element_combo.setCurrentText("Cd")
    assert panel.dft_isotope_combo.currentText() == "113Cd"
    assert "no Cd" in panel.dft_sigma_ref_label.text()
    assert not panel.calculate(simulate=False)
    assert "Give σ_ref" in said[-1][1]
    panel.dft_sigma_override_check.setChecked(True)
    panel.dft_sigma_ref_spin.setValue(3400.0)
    assert panel.calculate(simulate=False)
    assert panel.model["atoms"][0]["iso"] == pytest.approx(0.0, abs=1e-6)
    assert panel.site_dicts[0]["interactions"][0]["type"] == "csa"


def test_tensor_picture_in_cell_and_cartesian_frames(qapp, said, paths):
    window, panel = _panel(qapp, paths)
    panel.dft_delta_ref_spin.setValue(7.2)
    assert panel.calculate(simulate=False)
    assert panel.dft_show_tensor_button.isEnabled()
    assert panel.show_tensors(), panel.dft_tensor_notes_label.text()
    texts = {t.get_text() for ax in window.simulation_canvas.figure.axes for t in ax.texts}
    assert {"a", "b", "c"} <= texts
    assert "to a" in panel.dft_tensor_notes_label.text()
    panel.dft_frame_combo.setCurrentIndex(panel.dft_frame_combo.findData("cartesian"))
    panel.dft_tensor_style_combo.setCurrentIndex(panel.dft_tensor_style_combo.findData("surface"))
    assert panel.show_tensors()
    texts = {t.get_text() for ax in window.simulation_canvas.figure.axes for t in ax.texts}
    assert "a" not in texts


def test_no_cell_forces_the_cartesian_frame(qapp, said, tmp_path):
    path = write_magres(tmp_path / "mol.magres", [
        ("C", 1, (0, 0, 0), _tensor((40.0, 60.0, 170.0), _rotation(9)), None)], lattice=None)
    window = MainWindow()
    panel = window.dft_sim_panel
    panel.dft_sample_edit.setText(path)
    assert panel.read_files(quiet=True)
    assert panel.dft_frame_combo.currentData() == "cartesian"
    assert not panel.dft_frame_combo.model().item(0).isEnabled()
    panel.dft_sigma_override_check.setChecked(True)
    panel.dft_sigma_ref_spin.setValue(170.0)
    assert panel.calculate(simulate=False)
    assert panel.show_tensors()


def test_state_round_trip_and_journal(qapp, said, paths):
    window, panel = _panel(qapp, paths)
    panel.dft_site_list.item(1).setCheckState(Qt.CheckState.Unchecked)
    panel.dft_delta_ref_spin.setValue(7.2)
    assert panel.calculate(simulate=False)
    state = panel.get_state()
    assert state["dft"]["checked"] == ["Na1"]
    other = MainWindow().dft_sim_panel
    other.set_state(state)
    assert other.dft_sample_edit.text() == paths[0]
    assert other.checked_labels() == ["Na1"]
    assert other.dft_delta_ref_spin.value() == pytest.approx(7.2)
    assert [s["label"] for s in other.site_dicts] == ["Na1"]


def test_unreadable_file_is_reported(qapp, said, tmp_path):
    bad = tmp_path / "x.txt"
    bad.write_text("not dft\n")
    window = MainWindow()
    panel = window.dft_sim_panel
    panel.dft_sample_edit.setText(str(bad))
    assert not panel.read_files(quiet=True)
    assert "neither" in panel.dft_files_label.text()


def test_copy_buttons(qapp, said, paths):
    from PySide6.QtGui import QGuiApplication
    window, panel = _panel(qapp, paths)
    assert panel.calculate(simulate=False)
    panel.dft_csv_button.click()
    assert QGuiApplication.clipboard().text().startswith("Site,δiso (ppm)")
    panel.dft_python_button.click()
    assert "dft_nmr.dft_sites(sample, ref" in QGuiApplication.clipboard().text()
