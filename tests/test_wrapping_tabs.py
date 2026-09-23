import pytest
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget

from opennmr.gui.widgets.wrapping_tabs import WrappingTabWidget


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _tabs(qapp, labels=("One", "Two", "Three")):
    widget = WrappingTabWidget()
    pages = []
    for label in labels:
        page = QLabel(label)
        pages.append(page)
        widget.addTab(page, label)
    return widget, pages


def test_wrapping_tabs_match_the_qtabwidget_api_used(qapp):
    widget, pages = _tabs(qapp)

    assert widget.count() == 3
    assert widget.widget(1) is pages[1]
    assert widget.indexOf(pages[2]) == 2
    assert widget.tabText(0) == "One"
    assert widget.currentIndex() == 0
    assert widget.currentWidget() is pages[0]

    widget.setCurrentWidget(pages[2])
    assert widget.currentIndex() == 2
    widget.setCurrentIndex(1)
    assert widget.currentWidget() is pages[1]


def test_wrapping_tabs_emit_current_changed_like_qtabwidget(qapp):
    widget, pages = _tabs(qapp)
    seen = []
    widget.currentChanged.connect(seen.append)

    widget.setCurrentWidget(pages[2])
    widget.setCurrentIndex(0)

    assert seen == [2, 0]


def test_wrapping_tabs_button_follows_the_page_however_it_changed(qapp):
    # The strip must stay in step whether the page was changed by a
    # click or programmatically -- MainWindow does both.
    widget, pages = _tabs(qapp)

    widget.setCurrentWidget(pages[1])

    assert widget._buttons[1].isChecked()
    assert not widget._buttons[0].isChecked()


def test_hiding_a_tab_does_not_renumber_the_others(qapp):
    # setTabVisible hides the button but leaves the page in the stack:
    # callers hold indices from indexOf(), and MainWindow really does
    # hide a canvas tab (Processing Preview) while keeping the rest
    # addressable.
    widget, pages = _tabs(qapp)

    widget.setTabVisible(1, False)

    assert widget.isTabVisible(1) is False
    assert widget.count() == 3
    assert widget.indexOf(pages[2]) == 2
    assert widget.tabText(2) == "Three"

    widget.setTabVisible(1, True)
    assert widget.isTabVisible(1) is True


def test_tab_strip_grows_taller_as_it_narrows(qapp):
    # The whole point: a strip too narrow for one row takes two (or
    # more) instead of scrolling its tabs out of reach. QTabWidget
    # cannot do this -- see the module docstring for why subclassing
    # QTabBar does not work either.
    widget, _pages = _tabs(
        qapp, labels=("Peak Picking", "Integration", "Curve Fitting", "VOCS", "Imaging"),
    )
    layout = widget._bar_layout

    wide = layout.heightForWidth(1000)
    narrow = layout.heightForWidth(240)

    assert narrow > wide          # wrapped onto more rows
    assert wide > 0


def test_every_tab_stays_reachable_when_narrow(qapp):
    # Wrapping is only worth anything if the tabs remain clickable --
    # a scrolled-away tab is what this replaces.
    widget, pages = _tabs(
        qapp, labels=("FID", "Processing Preview", "Spectrum", "Spectral Fit",
                      "Fit", "Compare"),
    )
    widget.resize(200, 400)
    widget.show()
    try:
        for index, page in enumerate(pages):
            widget._buttons[index].click()
            assert widget.currentWidget() is page
    finally:
        widget.hide()


def test_wrapping_tabs_strip_actually_wraps_when_narrow(qapp):
    """The point of the widget: a strip too narrow for one row uses
    more rows rather than clipping or scrolling.

    Asserted through the buttons' own y positions, not the layout's
    arithmetic -- a `heightForWidth` that Qt never consults produces
    exactly the right number and still clips on screen, which is the
    bug this covers (the strip's size policy has to opt in with
    `setHeightForWidth(True)` before a parent layout will ask)."""
    host = QWidget()
    layout = QVBoxLayout(host)
    widget, _pages = _tabs(qapp, ("Peak Picking", "Integration", "Curve Fitting", "VOCS", "Imaging"))
    layout.addWidget(widget)
    host.show()

    def rows(width):
        host.resize(width, 500)
        qapp.processEvents()
        return len({button.y() for button in widget._buttons if not button.isHidden()})

    wide_rows = rows(900)
    narrow_rows = rows(240)

    assert wide_rows == 1
    assert narrow_rows > wide_rows
    # And the strip was actually given the height those rows need,
    # rather than being left one row tall with the rest cut off.
    assert widget._bar.height() >= widget._bar.heightForWidth(widget._bar.width())


def test_wrapping_tabs_strip_reports_height_for_width(qapp):
    """The strip must opt in to height-for-width, or a parent layout
    never asks and the wrapping is invisible."""
    widget, _pages = _tabs(qapp, ("Peak Picking", "Integration", "Curve Fitting", "VOCS", "Imaging"))

    assert widget._bar.sizePolicy().hasHeightForWidth()
    assert widget._bar.heightForWidth(200) > widget._bar.heightForWidth(2000)


# =====================================================# addTab(icon=...) -- the top-level launcher tiles (Load/Save/Compare,
# Processing, Advanced Processing, Fitting, Relaxation, Applications,
# Simulations) need an icon-above-text look; every existing sub-tab
# caller keeps its plain text-only button by simply not passing one.
# ============================================================

def _solid_icon(qapp):
    pixmap = QPixmap(16, 16)
    pixmap.fill()
    return QIcon(pixmap)


def test_addtab_without_icon_stays_text_only(qapp):
    widget = WrappingTabWidget()
    widget.addTab(QLabel("Peak Picking"), "Peak Picking")
    button = widget._buttons[0]
    assert button.icon().isNull()
    # No icon set at all -- explicitly not switched to icon-with-text.
    from PySide6.QtCore import Qt as _Qt
    assert button.toolButtonStyle() != _Qt.ToolButtonStyle.ToolButtonTextUnderIcon


def test_addtab_with_icon_sets_icon_and_text_under_icon_style(qapp):
    from PySide6.QtCore import Qt as _Qt
    widget = WrappingTabWidget()
    icon = _solid_icon(qapp)
    widget.addTab(QLabel("Processing"), "Processing", icon=icon)
    button = widget._buttons[0]
    assert not button.icon().isNull()
    assert button.toolButtonStyle() == _Qt.ToolButtonStyle.ToolButtonTextUnderIcon
    assert button.text() == "Processing"
    assert button.iconSize() == WrappingTabWidget._ICON_SIZE


def test_launcher_icon_size_is_large_enough_for_detailed_artwork(qapp):
    """Real user report: at the previous 28px, the launcher icons'
    fine linework (thin multi-colour illustrations, originally
    ~354x354) was lost -- e.g. the Processing FID icon's individual
    oscillation lines merged into a single muddy wedge. Not a
    rendering-quality assertion (that needs eyes on it -- see the
    session's own before/after size comparison), just a floor that
    stops this specific regression from creeping back in silently."""
    assert WrappingTabWidget._ICON_SIZE.width() >= 40
    assert WrappingTabWidget._ICON_SIZE.height() >= 40


def test_addtab_icon_and_no_icon_can_be_mixed_in_one_strip(qapp):
    # The launcher and an ordinary sub-tab strip use the same class --
    # confirms passing icon= on some tabs and not others doesn't
    # confuse indexing/labels/current-page tracking.
    widget = WrappingTabWidget()
    icon = _solid_icon(qapp)
    page_a, page_b = QLabel("A"), QLabel("B")
    widget.addTab(page_a, "A", icon=icon)
    widget.addTab(page_b, "B")

    assert widget.tabText(0) == "A"
    assert widget.tabText(1) == "B"
    assert not widget._buttons[0].icon().isNull()
    assert widget._buttons[1].icon().isNull()

    widget.setCurrentWidget(page_b)
    assert widget._buttons[1].isChecked()
    assert not widget._buttons[0].isChecked()
=======
def test_hidden_strip_can_show_the_current_tab_as_a_heading(qapp):
    # The main toolbar switches pages once the strips are hidden, so a
    # nested strip shows the page's name instead -- otherwise nothing
    # on screen says which sub-panel is open.
    widget, pages = _tabs(qapp)
    assert widget.isTabBarVisible()
    assert widget.headingText() == ""

    widget.setTabBarVisible(False, show_heading=True)
    assert not widget.isTabBarVisible()
    assert widget.headingText() == "One"
    widget.setCurrentWidget(pages[2])
    assert widget.headingText() == "Three"

    widget.setTabBarVisible(False)
    assert widget.headingText() == ""
    widget.setTabBarVisible(True)
    assert widget.isTabBarVisible()
    assert widget.headingText() == ""
