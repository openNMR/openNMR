import pytest
from PySide6.QtWidgets import QApplication, QLabel, QPushButton

from opennmr.gui.widgets.collapsible_section import CollapsibleSection


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


# Widgets never shown in a real window report isVisible() == False
# regardless of state (Qt requires the whole ancestor chain to be
# shown for that), so these tests check isHidden() instead -- it
# reflects whether setVisible(False)/hide() was explicitly called on
# the widget itself, which is exactly what expand/collapse toggles.


def test_starts_collapsed_by_default(qapp):
    # Explicit user request: every collapsible box starts collapsed, so a
    # panel opens as a short list of headings.
    content = QLabel("body")
    section = CollapsibleSection("FID Processing", content=content)
    assert section.isExpanded() is False
    assert section._body.isHidden() is True
    assert section.contentWidget() is content


def test_can_start_expanded(qapp):
    content = QLabel("body")
    section = CollapsibleSection("Phase", content=content, expanded=True)
    assert section.isExpanded() is True
    assert section._body.isHidden() is False


def test_clicking_header_toggles_body_visibility(qapp):
    content = QLabel("body")
    section = CollapsibleSection("Reference", content=content)
    assert section._body.isHidden() is True

    section._header.click()
    assert section.isExpanded() is True
    assert section._body.isHidden() is False

    section._header.click()
    assert section.isExpanded() is False
    assert section._body.isHidden() is True


def test_set_expanded_matches_click_behaviour(qapp):
    section = CollapsibleSection("VOCS", content=QLabel("x"))
    section.setExpanded(True)
    assert section.isExpanded() is True
    assert section._body.isHidden() is False
    section.setExpanded(False)
    assert section.isExpanded() is False
    assert section._body.isHidden() is True


def test_set_expanded_to_current_state_is_a_noop(qapp):
    section = CollapsibleSection("Imaging", content=QLabel("x"))
    seen = []
    section.toggled.connect(seen.append)
    section.setExpanded(False)  # already collapsed
    assert seen == []


def test_toggled_signal_emits_new_state(qapp):
    section = CollapsibleSection("eNMR", content=QLabel("x"))
    seen = []
    section.toggled.connect(seen.append)
    section._header.click()
    section._header.click()
    assert seen == [True, False]


def test_independent_sections_do_not_affect_each_other(qapp):
    # Explicit user choice: not an accordion -- opening/closing one
    # section must never touch a sibling's own state.
    a = CollapsibleSection("Curve Fitting", content=QLabel("a"))
    b = CollapsibleSection("Imaging", content=QLabel("b"))

    a.setExpanded(True)

    assert a.isExpanded() is True
    assert b.isExpanded() is False


def test_header_shows_the_title_text(qapp):
    section = CollapsibleSection("Integration", content=QLabel("x"))
    assert "Integration" in section._header.text()


def test_set_content_widget_replaces_the_body(qapp):
    first = QLabel("first")
    second = QPushButton("second")
    section = CollapsibleSection("CSA/Dipolar", content=first)

    section.setContentWidget(second)

    assert section.contentWidget() is second
    assert first.parent() is None  # detached, not left dangling in the layout
    assert section._body_layout.count() == 1


def test_title_is_bold_and_scaled_up_from_the_section_font(qapp):
    from opennmr.gui.widgets.collapsible_section import HEADER_FONT_SCALE

    section = CollapsibleSection("Phase", content=QLabel("body"))
    header = section.headerFont()
    assert header.bold()
    assert header.pointSizeF() == pytest.approx(section.font().pointSizeF() * HEADER_FONT_SCALE)
    assert HEADER_FONT_SCALE > 1.0


def test_title_follows_a_change_of_the_section_font(qapp):
    from PySide6.QtGui import QFont

    section = CollapsibleSection("Reference", content=QLabel("body"))
    font = QFont(section.font())
    font.setPointSizeF(20.0)
    section.setFont(font)
    assert section.headerFont().pointSizeF() == pytest.approx(25.0)
    assert section.headerFont().bold()


def test_title_follows_the_application_font(qapp):
    from PySide6.QtGui import QFont

    section = CollapsibleSection("FID Processing", content=QLabel("body"))
    original = QFont(qapp.font())
    try:
        bigger = QFont(original)
        bigger.setPointSizeF(16.0)
        qapp.setFont(bigger)
        qapp.processEvents()
        assert section.headerFont().pointSizeF() == pytest.approx(20.0)
    finally:
        qapp.setFont(original)
