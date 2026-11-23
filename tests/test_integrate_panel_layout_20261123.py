"""Integrate: the Select and Integrate button is the first, most obvious
control of the Integration Regions box, and the label checkbox stays one
line (user request, 2026-11-23)."""
from PySide6.QtWidgets import QSizePolicy

from opennmr.gui.main_window import MainWindow

from test_gui import qapp  # noqa: F401 -- fixture


def test_select_and_integrate_is_first_and_prominent(qapp):  # noqa: F811
    window = MainWindow()
    button = window.select_region_button
    layout = button.parentWidget().layout()
    assert layout.itemAt(0).widget() is button
    assert button.minimumHeight() >= 36
    assert "font-weight" in button.styleSheet() and ":checked" in button.styleSheet()
    assert button.text() == "Select and Integrate Regions from Plot"


def test_label_checkbox_is_one_line_with_a_short_tooltip(qapp):  # noqa: F811
    window = MainWindow()
    check = window.integral_labels_check
    assert check.sizePolicy().verticalPolicy() == QSizePolicy.Policy.Fixed
    assert all(len(line) <= 70 for line in check.toolTip().splitlines())
