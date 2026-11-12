"""The search box at the right of the tool row (explicit user request,
2026-11-12): the Ctrl+H "Where is…?" search made visible, shown or hidden
from View ▸ Show Search Box (remembered), folding to a button in a narrow
window. Typing shows the palette's results below the box without taking
the focus from it; ↑/↓, Enter and Esc work from the box."""

import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from opennmr.gui.widgets import search_box as sb

from test_gui import qapp  # noqa: F401 -- fixture


@pytest.fixture
def main(qapp):
    from opennmr.gui.main_window import MainWindow
    window = MainWindow()
    window.resize(1500, 900)
    window.show()
    QApplication.processEvents()
    return window


def test_the_box_is_in_the_tool_row_and_shown_by_default(main):
    box = main.search_box
    assert not box.isHidden() and not box.is_compact()
    assert main.search_box_action.isChecked()
    assert box.edit.placeholderText().startswith("Search")


def test_typing_shows_results_below_the_box(main):
    box = main.search_box
    box.edit.setFocus()
    QTest.keyClicks(box.edit, "baseline")
    palette = box.palette()
    assert palette is not None and palette.shown()
    assert not palette.edit.isVisible()                 # the box is the field
    assert palette.y() >= box.mapToGlobal(box.rect().bottomLeft()).y()
    from opennmr.gui.command_palette import rank
    assert [e.key for e in palette.shown()] == [e.key for e in rank("baseline", palette.entries())]


def test_arrows_enter_and_escape_work_from_the_box(main, monkeypatch):
    box = main.search_box
    box.edit.setFocus()
    QTest.keyClicks(box.edit, "export")
    palette = box.palette()
    assert palette.results.currentRow() == 0
    QTest.keyClick(box.edit, Qt.Key.Key_Down)
    assert palette.results.currentRow() == (1 if palette.results.count() > 1 else 0)
    chosen = []
    monkeypatch.setattr(type(palette), "activate",
                        lambda self, row=None: chosen.append(self.results.currentRow()) or None)
    QTest.keyClick(box.edit, Qt.Key.Key_Return)
    assert chosen
    box.edit.setText("x")
    box._results()
    QTest.keyClick(box.edit, Qt.Key.Key_Escape)
    assert box.edit.text() == "" and box.palette() is None


def test_ctrl_h_goes_to_the_box_when_shown_else_the_popup(main):
    main._on_search_shortcut()
    assert main.search_box.edit.hasFocus() or main.search_box.palette() is not None
    assert getattr(main, "_command_palette", None) is None
    main._set_search_box_visible(False)
    main._on_search_shortcut()
    assert main._command_palette is not None and main._command_palette.isVisible()


def test_show_hide_is_remembered(main):
    from opennmr.gui.main_window import MainWindow
    main.search_box_action.setChecked(False)
    assert main.search_box.isHidden()
    assert main._ui_settings["search_box_visible"] is False
    later = MainWindow()                          # same settings file (conftest)
    assert later.search_box.isHidden() and not later.search_box_action.isChecked()
    later.search_box_action.setChecked(True)
    assert not later.search_box.isHidden()


def test_a_narrow_window_folds_it_to_a_button(main):
    box = main.search_box
    main.resize(sb.COMPACT_BELOW - 200, 800)
    QApplication.processEvents()
    assert box.is_compact() and box.button.isVisibleTo(box) and not box.edit.isVisibleTo(box)
    main._on_search_shortcut()                    # the pop-up, as before
    assert main._command_palette is not None
    main.resize(sb.COMPACT_BELOW + 200, 800)
    QApplication.processEvents()
    assert not box.is_compact()


def test_the_box_is_not_a_search_result_of_its_own(main):
    from opennmr.gui.command_palette import build_index
    titles = [e.title for e in build_index(main)]
    assert "Show Search Box" in titles
    assert not any(t.startswith("Search…") for t in titles)
