"""Simulate ▸ DFT tensors with orca_pnmr files and .xyz structures (explicit
user request, 2026-11-13): the Structure row, the structure found next to
the file, the check against what orca_pnmr printed, the paramagnetic
glyphs on the structure, and the notebook code."""

import pytest
from PySide6.QtWidgets import QFileDialog, QMessageBox

from opennmr.gui.dft_panel import DftSimulationPanel
from test_gui import qapp  # noqa: F401
from test_orca_pnmr_structure import write_pnmr_output, write_xyz


@pytest.fixture
def no_dialogs(monkeypatch):
    said = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: said.append(a[2]))
    return said


class _Tensors:
    def __init__(self):
        self.drawn = []

    def __call__(self, draw):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig = plt.figure()
        draw(fig)
        self.drawn.append(fig)


def _panel(tmp_path, with_structure=True, tensors=None):
    if with_structure:
        write_xyz(tmp_path / "PrFeCN6_hfc.xyz")
    out = write_pnmr_output(tmp_path / "PrFeCN6_pnmr.txt", inp="PrFeCN6_hfc.pnmr.inp")
    p = DftSimulationPanel(plot_spectrum=lambda s: None, add_to_open=lambda s: None,
                           draw_tensors=tensors)
    p.dft_sample_edit.setText(out)
    assert p.read_files(quiet=True), p.dft_files_label.text()
    p.dft_element_combo.setCurrentIndex(p.dft_element_combo.findText("N"))
    p.larmor_spin.setValue(0)
    return p


def test_structure_row_shows_the_structure_found(qapp, tmp_path, no_dialogs):
    p = _panel(tmp_path)
    assert p._is_orca_pnmr() and p._is_orca_epr()
    assert p._import_form.isRowVisible(p._structure_file_row)
    assert p.dft_structure_edit.text().endswith("PrFeCN6_hfc.xyz")
    assert not p.dft_opm_box.isHidden()
    assert p.dft_opm_S_spin.value() == 1.0
    assert p.dft_site_list.count() == 2


def test_structure_row_is_hidden_for_other_files(qapp, tmp_path, no_dialogs):
    from test_paramagnetic_nmr import _mn_file
    p = DftSimulationPanel(plot_spectrum=lambda s: None, add_to_open=lambda s: None)
    p.dft_sample_edit.setText(_mn_file(tmp_path))
    assert p.read_files(quiet=True)
    assert not p._import_form.isRowVisible(p._structure_file_row)


def test_check_against_its_own_printout_needs_no_file(qapp, tmp_path, no_dialogs, monkeypatch):
    p = _panel(tmp_path)
    monkeypatch.setattr(QFileDialog, "getOpenFileName",
                        lambda *a, **k: pytest.fail("no file should be asked for"))
    assert p.calculate(simulate=False)
    text = p.check_against_orca()
    assert text and "3 nuclei" in text and "6 decimals" in text


def test_paramagnetic_parts_on_the_structure(qapp, tmp_path, no_dialogs):
    tensors = _Tensors()
    p = _panel(tmp_path, tensors=tensors)
    assert p.calculate(simulate=False)
    p.dft_tensor_style_combo.setCurrentIndex(p.dft_tensor_style_combo.findData("structure"))
    p.dft_structure_sites_combo.setCurrentIndex(p.dft_structure_sites_combo.findData("all"))
    for kind in ("para_contact", "para_dipolar"):
        index = p.dft_structure_tensor_combo.findData(kind)
        assert index >= 0
        p.dft_structure_tensor_combo.setCurrentIndex(index)
        assert p.show_tensors(), p.dft_tensor_notes_label.text()
    assert len(tensors.drawn) == 2
    # a molecule: the Cartesian frame is chosen for it
    assert p.dft_frame_combo.currentData() == "cartesian"


def test_without_a_structure_the_picture_says_what_is_missing(qapp, tmp_path, no_dialogs):
    p = _panel(tmp_path, with_structure=False, tensors=_Tensors())
    assert "no atomic positions" in p.dft_files_label.text()
    assert p.calculate(simulate=False)
    p.dft_tensor_style_combo.setCurrentIndex(p.dft_tensor_style_combo.findData("structure"))
    assert not p.show_tensors()
    assert "no atomic positions" in p.dft_tensor_notes_label.text()


def test_a_found_structure_is_not_kept_for_another_sample(qapp, tmp_path, no_dialogs):
    p = _panel(tmp_path)
    found = p.dft_structure_edit.text()
    other = tmp_path / "other"
    other.mkdir()
    p.dft_sample_edit.setText(write_pnmr_output(other / "lonely.txt"))
    assert p.read_files(quiet=True), p.dft_files_label.text()
    assert p.dft_structure_edit.text() != found


def test_a_chosen_structure_that_does_not_fit_is_reported(qapp, tmp_path, no_dialogs):
    p = _panel(tmp_path)
    p.dft_structure_edit.setText(write_xyz(tmp_path / "wrong.xyz",
                                                [("C", float(i), 0.0, 0.0) for i in range(9)]))
    assert not p.read_files(quiet=True)
    assert "does not belong" in p.dft_files_label.text()


def test_python_code_names_the_structure(qapp, tmp_path, no_dialogs):
    p = _panel(tmp_path)
    assert p.calculate(simulate=False)
    code = p.dft_python()
    assert "structure=" in code and "PrFeCN6_hfc.xyz" in code
    scope = {}
    exec(code.replace("print(", "(lambda *a, **k: None)("), scope)
    assert [r["iso"] for r in scope["model"]["atoms"]] == pytest.approx(
        [r["iso"] for r in p.model["atoms"]])


def test_paramagnetic_source_carries_the_structure(qapp, tmp_path, no_dialogs):
    p = _panel(tmp_path)
    src = p.paramagnetic_source()
    assert src["kind"] == "orca" and src["structure"].endswith("PrFeCN6_hfc.xyz")
