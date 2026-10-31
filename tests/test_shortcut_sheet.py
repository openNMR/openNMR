"""Help ▸ Keyboard Shortcuts… (F1, opennmr.gui.shortcut_sheet).

The sheet is generated from the window's menus; what is tested is that it
lists them, filters, finds conflicts, and exports Markdown.
"""
import pytest
from PySide6.QtGui import QGuiApplication, QKeySequence
from PySide6.QtWidgets import QMainWindow

from opennmr.gui.shortcut_sheet import (
    EXTRA_SHORTCUTS, ShortcutRow, ShortcutSheet, clean_text, collect_shortcuts, find_conflicts,
    iter_menu_actions, matches, to_markdown,
)

from test_gui import qapp  # noqa: F401 -- fixture


def test_clean_text_removes_mnemonics_but_keeps_a_literal_ampersand():
    assert clean_text("&Open…") == "Open…"
    assert clean_text("Save && Close") == "Save & Close"
    assert clean_text("") == ""


def _window():
    win = QMainWindow()
    file_menu = win.menuBar().addMenu("&File")
    file_menu.addAction("&Open…").setShortcut(QKeySequence("Ctrl+O"))
    file_menu.addSeparator()
    file_menu.addAction("Plain, no shortcut")
    window_menu = win.menuBar().addMenu("&Window")
    panel = window_menu.addMenu("Panel")
    panel.addAction("Processing").setShortcut(QKeySequence("Ctrl+2"))
    window_menu.addAction("Spectrum View").setShortcut(QKeySequence("Ctrl+2"))     # a conflict
    return win


def test_it_lists_every_menu_action_that_has_a_shortcut_with_its_menu_path(qapp):
    rows = collect_shortcuts(_window())
    assert [(r.group, r.action) for r in rows] == [
        ("File", "Open…"), ("Window ▸ Panel", "Processing"), ("Window", "Spectrum View")]
    assert rows[0].portable == "Ctrl+O"


def test_separators_and_unlabelled_actions_are_skipped(qapp):
    window = _window()
    paths = [(path, a.text()) for path, a in iter_menu_actions(window.menuBar().actions())]
    assert all(text for _, text in paths) and len(paths) == 4


def test_a_key_given_to_two_actions_is_reported_as_a_conflict(qapp):
    conflicts = find_conflicts(collect_shortcuts(_window()))
    assert list(conflicts) == ["Ctrl+2"]
    assert {r.action for r in conflicts["Ctrl+2"]} == {"Processing", "Spectrum View"}


def test_the_same_action_listed_twice_is_not_a_conflict():
    rows = [ShortcutRow("File", "Save", "Ctrl+S", "Ctrl+S"), ShortcutRow("File", "Save", "Ctrl+S", "Ctrl+S")]
    assert find_conflicts(rows) == {}


def test_filter_matches_every_word_in_menu_action_or_keys():
    row = ShortcutRow("View", "Show crosshair", "Ctrl+.", "Ctrl+.")
    assert matches(row, "view cross") and matches(row, "CTRL") and matches(row, "")
    assert not matches(row, "crosshair zebra")


def test_markdown_has_a_table_per_menu_and_the_extra_shortcuts():
    rows = [ShortcutRow("File", "Open | Load", "Ctrl+O", "Ctrl+O")]
    text = to_markdown(rows)
    assert "## File" in text and "| Open \\| Load | `Ctrl+O` |" in text
    for group, what, keys in EXTRA_SHORTCUTS:
        assert f"## {group}" in text and what in text


def test_the_dialog_shows_every_row_filters_and_warns_about_conflicts(qapp):
    window = _window()
    sheet = ShortcutSheet(window)
    assert sheet.table.rowCount() == 3 + len(EXTRA_SHORTCUTS)
    assert sheet.conflict_label.isHidden() is False and "Ctrl+2" in sheet.conflict_label.text()
    sheet.filter_edit.setText("open")
    assert sheet.visible_row_count() == 1
    sheet.filter_edit.setText("")
    assert sheet.visible_row_count() == sheet.table.rowCount()


def test_the_dialog_without_conflicts_shows_no_warning(qapp):
    window = QMainWindow()
    sheet = ShortcutSheet(window, rows=[ShortcutRow("File", "Open", "Ctrl+O", "Ctrl+O")])
    assert sheet.conflict_label.isHidden()


def test_copy_puts_markdown_on_the_clipboard(qapp):
    window = _window()
    sheet = ShortcutSheet(window)
    text = sheet.copy_markdown()
    assert QGuiApplication.clipboard().text() == text and "# openNMR keyboard shortcuts" in text


def test_the_real_window_lists_shortcuts_and_reports_its_conflicts_honestly(qapp):
    from opennmr.gui.main_window import MainWindow
    window = MainWindow()
    sheet = window.show_shortcut_sheet()
    rows = sheet.rows()
    assert any(r.action == "Open…" for r in rows)
    # Print any conflicts so a collision is visible in the test log; the
    # assertion is that they are *reported*, not that there are none.
    conflicts = find_conflicts(rows)
    if conflicts:
        print("Conflicting shortcuts:", {k: [(r.group, r.action) for r in v] for k, v in conflicts.items()})
        assert sheet.conflict_label.isHidden() is False
    window.close()
