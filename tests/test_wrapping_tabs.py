import pytest
from PySide6.QtWidgets import QApplication, QLabel, QWidget

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
