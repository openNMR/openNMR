"""The left-hand column's contents reflow to its width (explicit user
request, 2026-09-30: dragging the column narrower or wider must re-wrap
the text; on Simulate's and Fit's collapsible sections it did not move).

Cause: a panel's rows of buttons, wide combos, long check boxes and group
titles set its minimum width (to ~590 px) above the column's own 280 px,
and with horizontal scrolling off a panel narrower than its minimum is
clipped, not re-laid out. widgets/reflow.py lowers every minimum."""
import pytest
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QGroupBox, QHBoxLayout, QLabel, QPushButton, QScrollArea,
    QVBoxLayout, QWidget,
)

from opennmr.gui.main_window import MainWindow
from opennmr.gui.widgets import reflow
from opennmr.gui.widgets.collapsible_section import CollapsibleSection

from test_gui import qapp  # noqa: F401 -- fixture


def _settle(app, n=4):
    for _ in range(n):
        app.processEvents()


@pytest.fixture
def window(qapp):  # noqa: F811
    w = MainWindow()
    w.resize(1500, 1000)
    w.show()
    _settle(qapp)
    return w


def test_every_side_panel_fits_the_narrowest_column(qapp, window):  # noqa: F811
    """With every section open, no panel needs more than the column's own
    minimum (280 px, less its scroll bar) -- so nothing is clipped and
    everything reflows at any width the column can be dragged to."""
    for index in range(window.side_tabs.count()):
        window.side_tabs.setCurrentIndex(index)
        page = window.side_tabs.currentWidget()
        for section in page.findChildren(CollapsibleSection):
            section.setExpanded(True)
        _settle(qapp)
        content = page.widget() if isinstance(page, QScrollArea) else page
        name = window.side_tabs.tabText(index)
        assert content.minimumSizeHint().width() <= 266, name


@pytest.mark.parametrize("tile", ["Simulate", "Fit"])
def test_text_rewraps_when_the_column_is_resized(qapp, window, tile):  # noqa: F811
    window._show_side_tab(tile)
    page = window.side_tabs.currentWidget()
    for section in page.findChildren(CollapsibleSection):
        section.setExpanded(True)
    labels = [label for label in page.widget().findChildren(QLabel)
              if label.wordWrap() and len(label.text()) > 80 and label.isVisibleTo(page)]
    assert labels
    widths = {}
    for column in (300, 450, 700):
        window.main_splitter.setSizes([column, 1500 - column])
        _settle(qapp, 6)
        widths[column] = [label.width() for label in labels]
        # Nothing runs past the visible width.
        assert page.widget().width() <= page.viewport().width()
    for k in range(len(labels)):
        assert widths[300][k] < widths[450][k] < widths[700][k], labels[k].text()[:40]


def test_wrap_row_flows_and_fills(qapp):  # noqa: F811
    host = QWidget()
    v = QVBoxLayout(host)
    row = QHBoxLayout()
    buttons = [QPushButton(f"Button number {k}") for k in range(3)]
    for b in buttons:
        row.addWidget(b)
    v.addLayout(row)
    reflow.make_reflowable(host, limit=150)
    container = buttons[0].parentWidget()
    assert isinstance(container.layout(), reflow.WrapRowLayout)
    host.resize(900, 200)
    host.show()
    _settle(qapp)
    assert len({b.y() for b in buttons}) == 1                   # one line ...
    assert buttons[2].geometry().right() >= container.width() - 2   # ... filling it
    one_line = container.height()
    host.resize(160, 400)
    _settle(qapp)
    assert len({b.y() for b in buttons}) == 3                   # one per line
    assert container.height() > 2 * one_line
    assert host.minimumSizeHint().width() < 200


def test_label_stays_with_its_control_and_hidden_items_take_no_room(qapp):  # noqa: F811
    host = QWidget()
    v = QVBoxLayout(host)
    row = QHBoxLayout()
    label, check, hidden = QLabel("Mode:"), QCheckBox("x"), QPushButton("A very wide hidden button")
    hidden.setVisible(False)
    for w in (label, check, QPushButton("Another quite long button"), hidden):
        row.addWidget(w)
    v.addLayout(row)
    reflow.make_reflowable(host, limit=120)
    assert label.parentWidget() is check.parentWidget()         # kept as a pair
    host.resize(600, 100)
    host.show()
    _settle(qapp)
    assert label.y() == check.y()


def test_long_button_and_check_box_text_breaks_and_comes_back(qapp):  # noqa: F811
    host = QWidget()
    v = QVBoxLayout(host)
    button = QPushButton("Fit Every Slice with this Combined-Hamiltonian Model")
    check = QCheckBox("Bound peak amplitudes by region integrals always")
    v.addWidget(button)
    v.addWidget(check)
    reflow.make_reflowable(host, limit=150)
    host.resize(170, 200)
    host.show()
    _settle(qapp)
    assert "\n" in button.text() and "\n" in check.text()
    assert reflow.button_full_text(button) == "Fit Every Slice with this Combined-Hamiltonian Model"
    host.resize(900, 200)
    _settle(qapp)
    assert button.text() == "Fit Every Slice with this Combined-Hamiltonian Model"
    # Text set by code later is honoured.
    button.setText("Short")
    host.resize(880, 200)
    _settle(qapp)
    assert button.text() == "Short"


def test_long_group_title_is_shortened_not_a_minimum(qapp):  # noqa: F811
    host = QWidget()
    v = QVBoxLayout(host)
    box = QGroupBox("Enhancement factor  ε = spectrum 1 / spectrum 2, a long title")
    QVBoxLayout(box).addWidget(QLabel("x"))
    v.addWidget(box)
    reflow.make_reflowable(host, limit=150)
    assert host.minimumSizeHint().width() < 150
    host.resize(160, 100)
    host.show()
    _settle(qapp)
    assert box.title().endswith("…")
    assert reflow.group_full_title(box).endswith("a long title")
    assert "a long title" in box.toolTip()
    host.resize(900, 100)
    _settle(qapp)
    assert box.title().endswith("a long title")


def test_section_title_wraps_when_narrow(qapp):  # noqa: F811
    section = CollapsibleSection("Quadrupolar (Czjzek distribution) with a long name",
                                 content=QLabel("body"))
    section.resize(140, 100)
    section.show()
    _settle(qapp)
    assert "\n" in section._header.text()
    assert section.minimumSizeHint().width() < 140
    section.resize(900, 100)
    _settle(qapp)
    assert "\n" not in section._header.text()
    assert section.title() == "Quadrupolar (Czjzek distribution) with a long name"
