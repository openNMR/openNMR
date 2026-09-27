"""Two usability fixes (explicit user requests, 2026-09-27):

* the mouse wheel never changes a value box -- it scrolls the panel;
* Simulate takes the nucleus and the field in tesla; the Larmor frequency
  follows (nu0 = gamma/2pi B0), read-only.
"""
import numpy as np
import pytest
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtWidgets import (
    QApplication, QComboBox, QDoubleSpinBox, QFormLayout, QScrollArea, QSpinBox, QVBoxLayout,
    QWidget,
)

from opennmr.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
from opennmr.gui.main_window import MainWindow
from opennmr.gui.widgets import wheel_guard
from opennmr.gui.widgets.field_input import gamma_hz_per_t, link_field
from opennmr.notebook.utils import find_gamma


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _wheel(widget, steps=1):
    event = QWheelEvent(QPointF(widget.rect().center()), QPointF(widget.mapToGlobal(
        widget.rect().center())), QPoint(0, 0), QPoint(0, 120 * steps), Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase, False)
    QApplication.sendEvent(widget, event)


def _panel_in_scroll_area(qapp):
    """A tall panel of value boxes in a scroll area, as the side panels are."""
    wheel_guard.install(qapp)
    area = QScrollArea()
    inner = QWidget()
    v = QVBoxLayout(inner)
    spin, combo = QDoubleSpinBox(), QComboBox()
    spin.setRange(0, 1000)
    spin.setValue(100.0)
    combo.addItems(["a", "b", "c"])
    v.addWidget(spin)
    v.addWidget(combo)
    for _ in range(40):
        v.addWidget(QSpinBox())
    area.setWidget(inner)
    area.resize(300, 200)
    area.show()
    qapp.processEvents()
    return area, spin, combo


def test_wheel_over_a_spin_box_scrolls_the_panel_instead(qapp):
    area, spin, _combo = _panel_in_scroll_area(qapp)
    before = area.verticalScrollBar().value()
    _wheel(spin, steps=-3)
    _wheel(spin.lineEdit(), steps=-3)            # the box's own text field
    assert spin.value() == 100.0
    assert area.verticalScrollBar().value() != before


def test_wheel_does_not_change_a_focused_box_either(qapp):
    area, spin, combo = _panel_in_scroll_area(qapp)
    spin.setFocus()
    _wheel(spin, steps=2)
    combo.setFocus()
    _wheel(combo, steps=-1)
    assert spin.value() == 100.0 and combo.currentIndex() == 0


def test_typing_and_arrow_keys_still_work(qapp):
    from PySide6.QtTest import QTest
    _area, spin, combo = _panel_in_scroll_area(qapp)
    spin.setFocus()
    QTest.keyClick(spin, Qt.Key.Key_Up)
    assert spin.value() == pytest.approx(101.0)
    combo.setFocus()
    QTest.keyClick(combo, Qt.Key.Key_Down)
    assert combo.currentIndex() == 1


def test_the_main_window_installs_the_guard(qapp):
    MainWindow()
    assert getattr(qapp, "_opennmr_wheel_guard", None) is not None


def test_scroll_bars_and_lists_keep_the_wheel(qapp):
    from PySide6.QtWidgets import QListWidget
    wheel_guard.install(qapp)
    lst = QListWidget()
    lst.addItems([str(k) for k in range(200)])
    lst.resize(150, 100)
    lst.show()
    qapp.processEvents()
    _wheel(lst.viewport(), steps=-3)
    assert lst.verticalScrollBar().value() > 0


# -- field and Larmor frequency ------------------------------------------------

def _larmor(nucleus, field):
    # The Larmor frequency is |gamma|/2pi B0: 17O's gamma is negative.
    return abs(find_gamma(nucleus)["gamma"]) * field / 1e6


def test_link_computes_larmor_from_field_and_nucleus(qapp):
    form_host = QWidget()
    form = QFormLayout(form_host)
    nucleus = QComboBox()
    nucleus.setEditable(True)
    nucleus.setCurrentText("27Al")
    larmor = QDoubleSpinBox()
    larmor.setRange(0, 5000)
    larmor.setDecimals(4)
    form.addRow("Nucleus:", nucleus)
    form.addRow("Larmor frequency:", larmor)
    link = link_field(form, larmor, nucleus.currentText, nucleus.currentTextChanged)
    assert form.getWidgetPosition(link.field_spin.parentWidget())[0] == 1   # before Larmor
    assert larmor.isReadOnly()
    link.field_spin.setValue(9.4)
    assert larmor.value() == pytest.approx(_larmor("27Al", 9.4), abs=1e-4)
    # Same magnet, different nucleus.
    nucleus.setCurrentText("23Na")
    assert link.field_spin.value() == pytest.approx(9.4)
    assert larmor.value() == pytest.approx(_larmor("23Na", 9.4), abs=1e-4)
    # Setting the Larmor frequency from outside moves the field.
    larmor.setValue(_larmor("23Na", 14.1))
    assert link.field_spin.value() == pytest.approx(14.1, abs=1e-4)


def test_unknown_nucleus_leaves_the_larmor_frequency_alone(qapp):
    assert gamma_hz_per_t("Xx99") is None
    assert gamma_hz_per_t("") is None


def test_simulate_panels_take_a_field(qapp):
    window = MainWindow()
    # Quadrupolar
    window.sim_quad_nucleus_combo.setCurrentText("27Al")
    window.sim_quad_field_spin.setValue(18.8)
    assert window.sim_quad_larmor_spin.value() == pytest.approx(_larmor("27Al", 18.8), abs=1e-4)
    assert window.sim_quad_larmor_spin.isReadOnly()
    # CSA
    window.sim_csa_nucleus_edit.setText("13C")
    window.sim_csa_field_spin.setValue(9.4)
    assert window.sim_csa_larmor_spin.value() == pytest.approx(_larmor("13C", 9.4), abs=1e-4)
    # Combined Hamiltonian
    panel = window.combined_sim_panel
    panel.nucleus_combo.setCurrentText("17O")
    window.combined_sim_field_link.field_spin.setValue(11.7)
    assert panel.larmor_spin.value() == pytest.approx(_larmor("17O", 11.7), abs=1e-4)
    assert panel.settings()["larmor_Hz"] == pytest.approx(_larmor("17O", 11.7) * 1e6, rel=1e-6)


def test_field_changed_by_the_arrows_counts_as_the_users(qapp):
    window = MainWindow()
    window.sim_quad_nucleus_combo.setCurrentText("27Al")
    window.sim_quad_field_spin.stepBy(3)
    window._refresh_simulation_controls(_spectrum("1H", 400.13))
    assert window.sim_quad_field_link.user_set


def test_combined_default_is_kept_as_its_field(qapp):
    # The panel's own default (130.3 MHz 27Al) becomes the equivalent field.
    window = MainWindow()
    field = window.combined_sim_field_link.field_spin.value()
    assert field == pytest.approx(130.3e6 / find_gamma("27Al")["gamma"], abs=1e-4)


def test_csa_field_is_greyed_out_for_a_static_pattern(qapp):
    window = MainWindow()
    window.sim_csa_regime_combo.setCurrentIndex(window.sim_csa_regime_combo.findData(True))
    assert not window.sim_csa_field_spin.isEnabled()
    window.sim_csa_regime_combo.setCurrentIndex(window.sim_csa_regime_combo.findData(False))
    assert window.sim_csa_field_spin.isEnabled()


def _spectrum(nucleus, sfo1):
    ppm = np.linspace(0, 10, 64)
    return NMRSpectrum(data=np.zeros(64), axes=[NMRAxis(kind="frequency", ppm=ppm)],
                       metadata=NMRMetadata(vendor="bruker", nucleus=nucleus, sfo1=sfo1))


def test_a_selected_spectrum_seeds_the_field_from_its_own_nucleus(qapp):
    # A 1H spectrum at 400.13 MHz: the quadrupolar panel simulating 27Al
    # must get 9.3977 T -- i.e. 104.26 MHz for 27Al, not 400.13 MHz.
    window = MainWindow()
    window.sim_quad_nucleus_combo.setCurrentText("27Al")
    window._refresh_simulation_controls(_spectrum("1H", 400.13))
    b0 = 400.13e6 / find_gamma("1H")["gamma"]
    assert window.sim_quad_field_spin.value() == pytest.approx(b0, abs=1e-4)
    assert window.sim_quad_larmor_spin.value() == pytest.approx(_larmor("27Al", b0), abs=1e-3)


def test_a_field_the_user_set_is_not_overwritten_by_a_spectrum(qapp):
    window = MainWindow()
    window.sim_quad_nucleus_combo.setCurrentText("27Al")
    from PySide6.QtTest import QTest
    field = window.sim_quad_field_spin
    field.lineEdit().selectAll()
    QTest.keyClicks(field.lineEdit(), "21.1")        # typed, as a user does
    assert field.value() == pytest.approx(21.1)
    window._refresh_simulation_controls(_spectrum("1H", 400.13))
    assert window.sim_quad_field_spin.value() == pytest.approx(21.1)
