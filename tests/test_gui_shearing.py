"""Advanced Processing > Shearing and Scaling (2D): the panel
(gui/shear_panel.py) and its wiring into the main window."""

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from opennmr.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
from opennmr.gui.shear_panel import ShearPanel, shear_source_problem
from opennmr.notebook import shearing as sh

from test_shearing import make_2d, two_site_mqmas


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _mqmas(f1_sign=1, nucleus="23Na"):
    cryst, _ = two_site_mqmas(f1_sign=f1_sign)
    spec = make_2d(cryst)
    spec.metadata.nucleus = spec.metadata.nucleus_indirect = nucleus
    return spec


def test_source_problems(qapp):
    assert "Select" in shear_source_problem(None)
    one_d = NMRSpectrum(data=np.zeros(8), axes=[NMRAxis(kind="frequency", ppm=np.arange(8.0))],
                        metadata=NMRMetadata(vendor="bruker"))
    assert "1D" in shear_source_problem(one_d)
    pseudo = _mqmas()
    pseudo.metadata.pseudo_2d = True
    assert "pseudo-2D" in shear_source_problem(pseudo)
    panel = ShearPanel()
    panel.set_source(one_d, "x")
    assert not panel.shear_button.isEnabled()
    assert panel.shear() is None


def test_plan_readout_follows_settings_and_nucleus(qapp):
    panel = ShearPanel()
    panel.set_source(_mqmas(nucleus="27Al"), "exp 5")
    assert panel.spin_combo.currentData() == 2.5           # from the nucleus
    assert panel.mas_spin.value() == 12500                 # from acqus MASR
    text = panel.plan_edit.toPlainText()
    assert "-19/12" in text and "10/17" in text
    panel.order_combo.setCurrentIndex(panel.order_combo.findData(5))
    assert "-25/12" in panel.plan_edit.toPlainText()
    panel.experiment_combo.setCurrentIndex(panel.experiment_combo.findData("stmas"))
    assert panel.satellite_combo.count() == 2              # I = 5/2: ST1, ST2
    assert "STMAS" in panel.plan_edit.toPlainText()
    panel.experiment_combo.setCurrentIndex(panel.experiment_combo.findData("jres"))
    assert "no chemical shift" in panel.plan_edit.toPlainText()
    panel.experiment_combo.setCurrentIndex(panel.experiment_combo.findData("custom_coefficients"))
    panel.use_a_check.setChecked(True)
    panel.a1_spin.setValue(-7)
    panel.a2_spin.setValue(9)
    assert "-7/9" in panel.plan_edit.toPlainText()
    panel.a2_spin.setValue(0)
    assert "Not possible" in panel.plan_edit.toPlainText()
    assert not panel.shear_button.isEnabled()


def test_check_against_data_sets_f1_direction(qapp):
    panel = ShearPanel()
    panel.set_source(_mqmas(f1_sign=-1), "reversed")
    panel.spin_combo.setCurrentIndex(panel.spin_combo.findData(1.5))
    scan = panel.check_against_data()
    assert scan["kappa"] == pytest.approx(7 / 9, abs=0.02)
    assert panel.f1_sign_combo.currentData() == -1
    assert "f1_sign = -1" in panel.check_label.text()
    assert panel.plan().kappa_t2 == pytest.approx(7 / 9)
    assert panel.figure.axes                                  # the scan is drawn


def test_shear_emits_the_notebook_result(qapp):
    panel = ShearPanel()
    spec = _mqmas()
    panel.set_source(spec, "exp 7")
    panel.spin_combo.setCurrentIndex(panel.spin_combo.findData(1.5))
    got = []
    panel.result_ready.connect(lambda s, label, log: got.append((s, label, log)))
    preview = panel.preview()
    assert preview is not None and not got                    # preview adds nothing
    out = panel.shear()
    assert got and got[0][0] is out
    assert got[0][1].startswith("Sheared (MQMAS)") and "exp 7" in got[0][1]
    assert "-7/9" in got[0][2]
    ref = sh.shear_2d(spec, sh.make_shear_plan("mqmas", spin=1.5))
    np.testing.assert_allclose(out.data, ref.data)
    np.testing.assert_allclose(out.axes[0].ppm, ref.axes[0].ppm)
    # selecting the result keeps the parent as the source
    panel.set_source(out, "Sheared (MQMAS) exp 7")
    assert panel._source is spec


def test_unified_and_custom_scaling(qapp):
    panel = ShearPanel()
    spec = _mqmas()
    panel.set_source(spec, "e")
    panel.spin_combo.setCurrentIndex(panel.spin_combo.findData(1.5))
    panel.convention_combo.setCurrentIndex(panel.convention_combo.findData("unified"))
    assert "17/8" in panel.plan_edit.toPlainText()
    panel.convention_combo.setCurrentIndex(panel.convention_combo.findData("custom"))
    assert panel.custom_scale_spin.isEnabled()
    panel.custom_scale_spin.setValue(2.0)
    assert panel.plan().scale_f1 == pytest.approx(2.0)


def test_main_window_adv_proc_section(qapp):
    from opennmr.gui.main_window import MainWindow
    from test_gui import _side_tab_names
    window = MainWindow()
    side = _side_tab_names(window)
    adv = window.side_tabs.widget(side.index("Advanced Processing"))
    assert adv.isAncestorOf(window.shear_panel)
    assert window.shear_section.title() == "Shearing and Scaling (2D)"
    assert not window.shear_section.isExpanded()          # collapsed, like every section
    spec = _mqmas()
    window._add_derived_spectrum(spec, "mqmas test")
    assert window.shear_panel._source is window.session.spectra[0]
    window.shear_panel.spin_combo.setCurrentIndex(window.shear_panel.spin_combo.findData(1.5))
    n_before = len(window.session)
    out = window.shear_panel.shear()
    assert out is not None
    assert len(window.session) == n_before + 1
    new_index = len(window.session) - 1
    assert window.session.spectra[new_index].metadata.raw["shear"]["plan"]["kappa_t2"] == pytest.approx(-7 / 9)
    history = window._ensure_history(new_index)
    assert any(e.get("kind") == "shear" and "-7/9" in e["text"] for e in history["entries"])
    # removing everything leaves the panel without a source
    window.shear_panel.set_source(None)
    assert not window.shear_panel.shear_button.isEnabled()


def test_split_t1_option(qapp):
    from test_shearing import _split_t1_crystallites
    cryst, _ = _split_t1_crystallites()
    spec = make_2d(cryst, sw1=10000.0)
    spec.metadata.nucleus = spec.metadata.nucleus_indirect = "23Na"
    panel = ShearPanel()
    panel.set_source(spec, "split")
    assert panel.split_t1_check.isVisibleTo(panel)
    panel.check_against_data()
    assert panel.split_t1_check.isChecked()                  # recognised as already sheared
    assert "Split-t1 ticked" in panel.check_label.text()
    text = panel.plan_edit.toPlainText()
    assert "No shear" in text and "9/16" in text and "10/17" in text
    out = panel.shear()
    assert out.metadata.raw["shear"]["plan"]["kappa_t2"] == 0.0
    panel.experiment_combo.setCurrentIndex(panel.experiment_combo.findData("jres"))
    assert not panel.split_t1_check.isVisibleTo(panel)
    assert panel.plan().kappa_t1 == 1.0                       # split-t1 ignored off MQMAS/STMAS
