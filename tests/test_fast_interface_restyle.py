"""Settings ▸ Interface: font size and colour scheme change fast (explicit user
request, 2026-11-01). Re-setting the application stylesheet re-polished all
~7,900 widgets, 5-6 s a step; style.restyle re-polishes those on screen and
the rest when shown. These tests hold it to the slow path's result."""

import time

import pytest
from PySide6.QtWidgets import (QAbstractButton, QApplication, QComboBox, QLineEdit,
                               QStackedWidget, QTabWidget)

from opennmr.gui import style
from opennmr.gui.main_window import MainWindow

from test_gui import qapp  # noqa: F401 -- fixture


@pytest.fixture
def window(qapp):  # noqa: F811
    w = MainWindow()
    w.resize(1500, 950)
    w.show()
    for _ in range(5):
        qapp.processEvents()
    yield w
    # Back to the defaults for the tests that follow (one QApplication).
    w.ui_theme_combo.setCurrentIndex(max(w.ui_theme_combo.findData("system"), 0))
    w._on_ui_font_reset_clicked()
    w.close()
    w.deleteLater()
    qapp.processEvents()


def _pages(window):
    return window.findChildren(QStackedWidget) + window.findChildren(QTabWidget)


def _drawn_colour(widget):
    image = widget.grab().toImage()
    if image.isNull() or image.width() < 3 or image.height() < 3:
        return None
    return image.pixelColor(image.width() // 2, image.height() // 2).name()


def _record_every_page(app, window):
    """{widget id: (font size, bold, drawn colour or None)} for every widget
    that was on screen while each page of every stack and tab was shown."""
    seen = {}
    for stack in _pages(window):
        start = stack.currentIndex()
        for i in range(stack.count()):
            stack.setCurrentIndex(i)
            app.processEvents()
            for w in app.allWidgets():
                if w.isVisible() and w.window() is window and id(w) not in seen:
                    colour = (_drawn_colour(w) if isinstance(w, (QAbstractButton, QLineEdit,
                                                                 QComboBox)) else None)
                    seen[id(w)] = (round(w.font().pointSizeF(), 2), w.font().bold(), colour)
        stack.setCurrentIndex(start)
        app.processEvents()
    return seen


def _set_theme(window, mode):
    window.ui_theme_combo.setCurrentIndex(window.ui_theme_combo.findData(mode))


def test_font_and_theme_changes_match_a_full_restyle_on_every_page(qapp, window):  # noqa: F811
    # Several changes in a row, so pages hidden through more than one
    # generation are covered too.
    window.ui_font_size_spin.setValue(14)
    _set_theme(window, "dark")
    window.ui_font_size_spin.setValue(12)
    _set_theme(window, "nord")
    qapp.processEvents()
    assert style.restyle_generation() >= 4
    fast = _record_every_page(qapp, window)

    # The slow path: re-set the whole sheet, every widget polished afresh.
    qapp.setStyleSheet(qapp.styleSheet() + " ")
    qapp.processEvents()
    slow = _record_every_page(qapp, window)

    assert len(fast) > 300
    shared = set(fast) & set(slow)
    differ = [(k, fast[k], slow[k]) for k in shared if fast[k] != slow[k]]
    assert not differ, differ[:10]
    assert {v[0] for v in fast.values() if not v[1]} >= {12.0}


def test_changes_are_fast(qapp, window):  # noqa: F811
    times = []
    for value in (13, 9, 16):
        t = time.perf_counter()
        window.ui_font_size_spin.setValue(value)
        qapp.processEvents()
        times.append(time.perf_counter() - t)
    for mode in ("dark", "light", "solarized_dark"):
        t = time.perf_counter()
        _set_theme(window, mode)
        qapp.processEvents()
        times.append(time.perf_counter() - t)
    # Was 5-6 s per change on a slow CI machine; now well under a second.
    assert max(times) < 1.5, times


def test_theme_change_does_not_reset_the_stylesheet(qapp, window, monkeypatch):  # noqa: F811
    calls = []
    real = QApplication.setStyleSheet
    monkeypatch.setattr(QApplication, "setStyleSheet",
                        lambda self, s: (calls.append(1), real(self, s))[1])
    _set_theme(window, "dark")
    window.ui_font_size_spin.setValue(15)
    assert calls == []


def test_a_widget_built_after_the_change_takes_it(qapp, window):  # noqa: F811
    from PySide6.QtWidgets import QPushButton
    window.ui_font_size_spin.setValue(16)
    _set_theme(window, "dark")
    button = QPushButton("new", window)
    button.resize(120, 40)
    button.show()
    qapp.processEvents()
    assert button.font().pointSizeF() == 16
    image = button.grab().toImage()            # background, clear of the text
    assert image.pixelColor(8, 20).name() == style._palette(True).button().color().name()


def test_hidden_widget_is_restyled_when_shown(qapp, window):  # noqa: F811
    # A combo on a page of a stack that is not the current page, and that
    # will be visible once that page is.
    found = None
    for stack in window.findChildren(QStackedWidget):
        if not stack.isVisible():
            continue
        for i in range(stack.count()):
            page = stack.widget(i)
            if i == stack.currentIndex():
                continue
            combo = next((c for c in page.findChildren(QComboBox) if c.isVisibleTo(page)), None)
            if combo is not None:
                found = (stack, i, combo)
                break
        if found:
            break
    assert found, "no hidden page with a combo box"
    stack, index, combo = found
    window.ui_font_size_spin.setValue(17)
    assert not combo.isVisible()
    assert combo.property("_opennmr_restyle_generation") != style.restyle_generation()
    stack.setCurrentIndex(index)
    qapp.processEvents()
    assert combo.isVisible()
    assert combo.font().pointSizeF() == 17
    assert combo.property("_opennmr_restyle_generation") == style.restyle_generation()


def test_settings_are_still_saved_at_once(qapp, window):  # noqa: F811
    import json
    window.ui_font_size_spin.setValue(11)
    _set_theme(window, "dark")
    with open(MainWindow._UI_SETTINGS_PATH, encoding="utf-8") as handle:
        saved = json.load(handle)
    assert saved["ui_font_size"] == 11 and saved["ui_theme"] == "dark"


def test_section_titles_stay_proportionally_larger(qapp, window):  # noqa: F811
    from opennmr.gui.widgets.collapsible_section import HEADER_FONT_SCALE, CollapsibleSection
    window.ui_font_size_spin.setValue(14)
    qapp.processEvents()
    section = next(s for s in window.findChildren(CollapsibleSection) if s.isVisible())
    assert section._header.font().pointSizeF() == pytest.approx(14 * HEADER_FONT_SCALE, rel=0.05)
