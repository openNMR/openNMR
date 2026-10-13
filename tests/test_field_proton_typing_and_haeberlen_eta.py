"""2026-10-13 (explicit user requests):

* Simulate: the field can be given as the magnet's 1H frequency (MHz),
  as in the NMR Periodic Table -- typed in either box, the other follows.
* NMR Periodic Table (and every field box): values above 20 T can be
  typed. Qt refused keystrokes on the way to them ("3" then "0" in front
  of "9.394641 T" is 309 T, over the limit; a digit at the end is one
  decimal too many), so 30 T could only be reached by clicking the up
  arrow. ``widgets/typed_spinbox.py`` accepts any number while typing.
* A negative Haeberlen η is simulated and fitted, with the warning
  "Negative values of η in the Haeberlen convention are physically
  incorrect ... Please check the value entered"; a Herzfeld-Berger skew
  of either sign is fine.
"""

import warnings

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QLineEdit, QVBoxLayout, QWidget

from opennmr.gui.main_window import MainWindow
from opennmr.gui.periodic_table_panel import LarmorCalculatorPanel
from opennmr.gui.widgets.typed_spinbox import TypedDoubleSpinBox
from opennmr.notebook import combined_fitting as cfit
from opennmr.notebook.combined_hamiltonian import (
    Site, csa_asymmetry_warning, csa_warnings, simulate_combined)
from opennmr.notebook.utils import find_gamma

from test_gui import qapp  # noqa: F401 -- fixture

GAMMA_1H = abs(find_gamma("1H")["gamma"])          # Hz/T


def _type(box, text, at=None, select_all=False):
    le = box.lineEdit()
    box.setFocus()
    if select_all:
        le.selectAll()
    elif at is not None:
        le.setCursorPosition(at if at >= 0 else len(le.text()) + at)
    QTest.keyClicks(le, text)
    QTest.keyClick(le, Qt.Key.Key_Return)


# ---- typing freely ------------------------------------------------------------------------------

def _box(qapp):
    host = QWidget()
    layout = QVBoxLayout(host)
    box = TypedDoubleSpinBox()
    box.setRange(0.0, 50.0)
    box.setDecimals(6)
    box.setSuffix(" T")
    box.setValue(9.394641)
    layout.addWidget(box)
    layout.addWidget(QLineEdit())
    host.show()
    return host, box


def test_keystrokes_on_the_way_to_a_number_are_not_refused(qapp):
    host, box = _box(qapp)
    _type(box, "30", at=0)                 # "309.394641" on the way: kept, then clamped
    assert box.value() == pytest.approx(50.0)
    _type(box, "28.2", select_all=True)
    assert box.value() == pytest.approx(28.2)
    _type(box, "77", at=-2)                # past the decimals shown: rounded on Enter
    assert box.value() == pytest.approx(28.200001)
    box.lineEdit().selectAll()
    QTest.keyClicks(box.lineEdit(), "30")          # no Enter: leaving the box takes it
    host.findChild(QLineEdit).setFocus()
    QTest.qWait(20)
    box.interpretText()
    assert box.value() == pytest.approx(30.0)


def test_arrows_and_set_value_are_unchanged(qapp):
    _host, box = _box(qapp)
    box.setValue(12.0)
    box.stepBy(2)
    assert box.value() == pytest.approx(12.0 + 2 * box.singleStep())
    box.setValue(99.0)
    assert box.value() == pytest.approx(50.0)


def test_periodic_table_takes_a_typed_30_tesla(qapp):
    panel = LarmorCalculatorPanel()
    panel.show()
    _type(panel.field_spin, "30", select_all=True)
    assert panel.field() == pytest.approx(30.0)
    assert panel.proton_spin.value() == pytest.approx(GAMMA_1H * 30.0 / 1e6, abs=1e-3)
    # Typed in front of the old value, as a user clicking at the start would.
    panel.set_field(2.0)
    _type(panel.field_spin, "3", at=0)
    assert panel.field() == pytest.approx(32.0)
    _type(panel.proton_spin, "1200", select_all=True)
    assert panel.field() == pytest.approx(1200e6 / GAMMA_1H, abs=1e-6)


# ---- 1H frequency in Simulate ---------------------------------------------------------------------

def test_simulate_field_can_be_given_as_the_proton_frequency(qapp):
    window = MainWindow()
    link = window.combined_sim_field_link
    panel = window.combined_sim_panel
    panel.nucleus_combo.setCurrentText("27Al")
    link.proton_spin.setValue(600.0)
    assert link.field_spin.value() == pytest.approx(600e6 / GAMMA_1H, abs=1e-6)
    gamma_al = abs(find_gamma("27Al")["gamma"])
    assert panel.larmor_spin.value() == pytest.approx(gamma_al * 600e6 / GAMMA_1H / 1e6, rel=1e-6)
    assert link.user_set                   # a spectrum no longer re-seeds it
    link.field_spin.setValue(18.8)
    assert link.proton_spin.value() == pytest.approx(GAMMA_1H * 18.8 / 1e6, abs=1e-3)
    assert link.field_spin.maximum() >= 45.0
    # Every Simulate field built by link_field has one.
    for other in (window.sim_quad_field_link, window.sim_csa_field_link):
        assert other.proton_spin.value() == pytest.approx(
            GAMMA_1H * other.field_spin.value() / 1e6, abs=1e-3)


def test_typed_proton_frequency_reads_back_exactly(qapp):
    window = MainWindow()
    window.show()
    link = window.combined_sim_field_link
    window._show_side_tab("Simulate")
    _type(link.proton_spin, "400", select_all=True)
    assert link.proton_spin.value() == pytest.approx(400.0, abs=5e-4)   # 6-decimal field


def test_pnmr_and_j_panels_have_a_proton_box_that_is_not_saved_twice(qapp):
    window = MainWindow()
    for panel in (window.pnmr_panel, window.jcoupling_sim_panel):
        box = panel._field_proton_spin
        box.setValue(500.0)
        assert panel.field_spin.value() == pytest.approx(500e6 / GAMMA_1H, abs=1e-6)
        from opennmr.gui import panel_state
        assert "_field_proton_spin" not in panel_state.capture_widgets(panel)


# ---- Haeberlen η out of range: said, not refused -------------------------------------------------

def test_the_warning_words_and_ranges():
    message = csa_asymmetry_warning(-0.2, "Haeberlen")
    assert message.startswith("Negative values of η in the Haeberlen convention are "
                              "physically incorrect")
    assert "Please check the value entered." in message
    assert csa_asymmetry_warning(1.2, "Haeberlen") is not None
    assert csa_asymmetry_warning(0.0, "Haeberlen") is None
    assert csa_asymmetry_warning(1.0, "Haeberlen") is None
    for skew in (-1.0, -0.4, 0.0, 0.6, 1.0):
        assert csa_asymmetry_warning(skew, "Herzfeld-Berger") is None
    assert csa_asymmetry_warning(-1.3, "Herzfeld-Berger") is not None


def test_a_negative_haeberlen_eta_is_simulated_with_a_warning():
    x = np.linspace(-300, 300, 1024)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        site = Site("13C", 0, label="C1", fwhm=3).add_csa(100, -0.4)
    assert any("Negative values of η" in str(w.message) for w in caught)
    assert site.get("csa")[0]["asym"] == -0.4                     # kept as given
    result = simulate_combined([site], x, 100e6, n_zcw=987)
    assert np.all(np.isfinite(result["real_spec"]))
    assert any(n.startswith("⚠ C1: Negative values of η") for n in result["notes"])
    assert "⚠ C1: Negative values of η" in result["log"]
    hb = Site("13C", 0, label="C2", fwhm=3).add_csa(100, -0.4, convention="Herzfeld-Berger")
    assert csa_warnings([hb]) == []


def test_a_fit_from_a_negative_eta_says_so_and_starts_from_its_magnitude():
    x = np.linspace(-300, 300, 2048)
    truth = [Site("13C", 0, label="A", fwhm=3).add_csa(100, 0.4)]
    y = simulate_combined(truth, x, 100e6, n_zcw=2584)["real_spec"] * 1e3
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        start = [Site("13C", 0, label="A", fwhm=3).add_csa(90, -0.3)]
    fit = cfit.fit_combined({"x_ppm": x, "real_spectrum": y, "SFO1": 100.0}, start,
                            n_zcw=2584)
    said = [w for w in fit["warnings"] if "Negative values of η" in w]
    assert said and "started from η = 0.3" in said[0]
    asym = next(p["value"] for p in fit["parameters"] if p["name"] == "asym")
    assert asym == pytest.approx(0.4, abs=0.01)


def test_the_site_editor_warns_for_haeberlen_only(qapp):
    window = MainWindow()
    panel = window.combined_sim_panel
    panel.nucleus_combo.setCurrentText("13C")
    panel.add_interaction("csa", aniso=100.0, asym=0.5)
    editor = next(e for e in panel.interaction_editors if e.data["type"] == "csa")
    assert not editor.asym_warning.isVisibleTo(editor)
    editor.asym.setValue(-0.3)
    assert editor.data["asym"] == pytest.approx(-0.3)              # not refused
    assert editor.asym_warning.isVisibleTo(editor)
    assert "Negative values of η in the Haeberlen convention" in editor.asym_warning.text()
    editor.convention.setCurrentIndex(editor.convention.findData("Herzfeld-Berger"))
    assert not editor.asym_warning.isVisibleTo(editor)             # a negative skew is fine
    editor.convention.setCurrentIndex(editor.convention.findData("Haeberlen"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert panel.simulate()
    assert "Negative values of η" in panel.results_label.text()
    assert "Negative values of η" in panel.results_label.html()
