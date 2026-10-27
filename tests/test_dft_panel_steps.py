"""2026-10-27 (explicit user request): Simulate ▸ DFT tensors laid out as
collapsible numbered steps -- 1 read the file and choose the sites, 2 the
shift scale or paramagnetic scaling, 3 motional averaging, 4 the
experiment (static/MAS, field; the nucleus is not typed again), 5
simulate, 6 view the tensors."""

import pytest

from opennmr.gui.widgets.collapsible_section import CollapsibleSection

from test_legend_supercell_atom_style import _write, qapp  # noqa: F401 -- fixture


@pytest.fixture
def window(qapp, tmp_path, monkeypatch):  # noqa: F811
    from PySide6.QtWidgets import QMessageBox
    from opennmr.gui.main_window import MainWindow
    for name in ("critical", "information", "warning"):
        monkeypatch.setattr(QMessageBox, name, lambda *a, **k: None)
    return MainWindow()


def _step_of(panel, widget):
    node = widget.parentWidget()
    while node is not None:
        if node in panel.dft_steps:
            return panel.dft_steps.index(node) + 1
        node = node.parentWidget()
    return None


def test_six_steps_in_order_and_collapsed(window):
    panel = window.dft_sim_panel
    assert [s.title() for s in panel.dft_steps] == list(panel.STEP_TITLES)
    assert [s.title()[0] for s in panel.dft_steps] == list("123456")
    # Directly in the DFT section, one after another, then nothing else.
    sections = [s for s in panel.findChildren(CollapsibleSection) if s.parentWidget() is panel]
    assert sections == panel.dft_steps
    assert all(not s.isExpanded() for s in panel.findChildren(CollapsibleSection))


def test_every_control_is_in_its_step(window):
    panel = window.dft_sim_panel
    where = {
        1: (panel.dft_sample_edit, panel.dft_element_combo, panel.dft_isotope_combo,
            panel.dft_site_list, panel.dft_fwhm_spin),
        2: (panel.dft_add_reference_button, panel.dft_reference_table,
            panel.dft_sigma_override_check, panel.dft_pm_box),
        3: (panel.dft_state_combo, panel.dft_average_button, panel.dft_average_list),
        4: (panel.larmor_spin, panel.regime_combo, panel.mas_spin, panel.transitions_combo,
            panel.order_combo, panel.orientations_spin, panel.dft_nucleus_label,
            window.dft_sim_field_link.field_spin),
        5: (panel.dft_calculate_button, panel.dft_result_table, panel.site_list,
            panel.site_box, panel.auto_axis_check, panel.simulate_button,
            panel.results_label, panel.export_button, panel.python_button),
        6: (panel.dft_tensor_site_combo, panel.dft_tensor_style_combo,
            panel.dft_show_tensor_button, panel.dft_structure_region_combo),
    }
    for step, widgets in where.items():
        for w in widgets:
            assert _step_of(panel, w) == step, (step, w.objectName() or type(w).__name__)
    # Inside step 5: the site editor and the axis options are sub-sections.
    assert panel.dft_sites_section.isAncestorOf(panel.site_box)
    assert panel.dft_axis_section.isAncestorOf(panel.auto_axis_check)
    assert panel.simulate_button.text() == "Simulate the Sites as Edited"


def test_the_nucleus_is_not_typed_again(window, tmp_path):
    panel = window.dft_sim_panel
    # Step 4 shows the nucleus, read-only; the typed box is out of sight.
    form = panel._setup_form
    assert not form.isRowVisible(panel.nucleus_combo)
    assert "step 1" in panel.dft_nucleus_label.text()
    assert not panel.dft_g0_check.isHidden()          # before a file: it may be VASP
    panel.dft_sample_edit.setText(_write(tmp_path))
    assert panel.read_files(quiet=True)
    assert panel.dft_g0_check.isHidden() and panel.dft_g0_help.isHidden()   # a .magres
    assert panel.dft_element_combo.currentText() == "Na"
    assert panel.dft_isotope_combo.currentText() == "23Na"
    assert panel.dft_nucleus_label.text() == "23Na"
    assert panel.nucleus_combo.currentText() == "23Na"
    # The field gives the Larmor frequency of that nucleus: 23Na's is
    # 26.45 % of 1H's (IUPAC Ξ).
    window.dft_sim_field_link.field_spin.setValue(9.4)
    proton = window.dft_sim_field_link.proton_spin.value()
    assert panel.larmor_spin.value() / proton == pytest.approx(0.264519, rel=2e-3)
    # The whole workflow: references typed, calculate and simulate.
    panel.dft_sigma_override_check.setChecked(True)
    panel.dft_sigma_ref_spin.setValue(550.0)
    panel.regime_combo.setCurrentIndex(panel.regime_combo.findData("mas"))
    panel.mas_spin.setValue(12000.0)
    assert panel.calculate(simulate=True)
    assert panel.spectrum is not None
    assert panel.nucleus_combo.currentText() == "23Na"
