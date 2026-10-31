"""
Help ▸ Keyboard Shortcuts… (F1): every keyboard shortcut in one searchable list.

**Generated, not written.** The rows are read from the window's own menu
actions, so a shortcut added to a menu later appears here without
anyone remembering to list it, and the list cannot say something the
menus do not. The few shortcuts that live outside the menus (arrow keys
in Compare, a double-click) are the short, hand-kept ``EXTRA_SHORTCUTS``.

The sheet also reports **conflicts** -- one key sequence assigned to
more than one action. Qt resolves those as "ambiguous" and fires
neither, so the user just sees a shortcut that does nothing. Reading
the menus makes them visible.

The pure parts (:func:`clean_text`, :func:`find_conflicts`,
:func:`to_markdown`) need no Qt objects.
"""

from __future__ import annotations

import dataclasses
import re

from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication, QKeySequence
from PySide6.QtWidgets import (
    QAbstractItemView, QDialog, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QPushButton,
    QTableWidget, QTableWidgetItem, QVBoxLayout,
)

#: Opens the sheet. F1 is the platform's "help" key.
SHEET_KEY = "F1"

#: Shortcuts that are not menu actions: (group, what it does, keys).
EXTRA_SHORTCUTS = (
    ("Compare", "Move the highlighted spectrum up / down the stack", "Up / Down"),
    ("Compare", "Reset the Spacing slider to its default", "Double-click the slider"),
)


@dataclasses.dataclass(frozen=True)
class ShortcutRow:
    group: str                  # the menu it is in, e.g. "View" or "Window ▸ Panel"
    action: str                 # what it does
    keys: str                   # as shown on this platform (⌘ on a Mac)
    portable: str = ""          # platform-independent text, for conflict detection
    enabled: bool = True


def clean_text(text: str) -> str:
    """A menu label without its ``&`` mnemonic marker (``&&`` is a literal &)."""
    return re.sub(r"&(?!&)", "", text or "").replace("&&", "&").strip()


def iter_menu_actions(actions, path=()):
    """``(menu_path, QAction)`` for every leaf action under ``actions``
    (a menu bar's or a menu's), separators and empty labels skipped."""
    for action in actions:
        if action.isSeparator():
            continue
        text = clean_text(action.text())
        menu = action.menu()
        if menu is not None:
            yield from iter_menu_actions(menu.actions(), path + (text,))
        elif text:
            yield path, action


def collect_shortcuts(window) -> list[ShortcutRow]:
    """Every menu action of ``window`` that has a shortcut, in menu order."""
    rows = []
    for path, action in iter_menu_actions(window.menuBar().actions()):
        for sequence in action.shortcuts() or ([action.shortcut()] if not action.shortcut().isEmpty() else []):
            if sequence.isEmpty():
                continue
            rows.append(ShortcutRow(
                group=" ▸ ".join(path) or "Other", action=clean_text(action.text()),
                keys=sequence.toString(QKeySequence.SequenceFormat.NativeText),
                portable=sequence.toString(QKeySequence.SequenceFormat.PortableText),
                enabled=action.isEnabled()))
    return rows


def find_conflicts(rows) -> dict[str, list[ShortcutRow]]:
    """Key sequences assigned to more than one *different* action:
    ``{portable text: [rows]}``."""
    by_keys: dict[str, list[ShortcutRow]] = {}
    for row in rows:
        if row.portable:
            by_keys.setdefault(row.portable, []).append(row)
    return {keys: group for keys, group in by_keys.items()
            if len({(r.group, r.action) for r in group}) > 1}


def to_markdown(rows, extras=EXTRA_SHORTCUTS) -> str:
    """The sheet as Markdown, one table per menu."""
    groups: dict[str, list[tuple[str, str]]] = {}
    for row in rows:
        groups.setdefault(row.group, []).append((row.action, row.keys))
    for group, what, keys in extras:
        groups.setdefault(group, []).append((what, keys))
    out = ["# openNMR keyboard shortcuts", ""]
    for group, items in groups.items():
        out += [f"## {group}", "", "| Action | Shortcut |", "| --- | --- |"]
        for what, keys in items:
            escaped = what.replace("|", "\\|")
            out.append(f"| {escaped} | `{keys}` |")
        out.append("")
    return "\n".join(out)


def matches(row: ShortcutRow, query: str) -> bool:
    """Whether every word of ``query`` is in the row's group, action or keys."""
    text = f"{row.group} {row.action} {row.keys}".lower()
    return all(word in text for word in query.lower().split())


class ShortcutSheet(QDialog):
    """The searchable list. ``rows`` defaults to ``collect_shortcuts(window)``."""

    def __init__(self, window, rows=None, parent=None) -> None:
        super().__init__(parent or window)
        self.setWindowTitle("Keyboard shortcuts")
        self.setModal(False)
        self.resize(620, 560)
        self._rows = list(rows if rows is not None else collect_shortcuts(window))
        for group, what, keys in EXTRA_SHORTCUTS:
            self._rows.append(ShortcutRow(group, what, keys, ""))
        layout = QVBoxLayout(self)

        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("Filter, e.g. 'phase', 'ctrl+s', 'view'…")
        self.filter_edit.setClearButtonEnabled(True)
        self.filter_edit.textChanged.connect(self._apply_filter)
        layout.addWidget(self.filter_edit)

        conflicts = find_conflicts(self._rows)
        self.conflict_label = QLabel()
        self.conflict_label.setWordWrap(True)
        self.conflict_label.setVisible(bool(conflicts))
        if conflicts:
            parts = []
            for keys, group in conflicts.items():
                shown = group[0].keys
                names = "; ".join(f"{r.group} ▸ {r.action}" for r in group)
                parts.append(f"<b>{shown}</b> is assigned to {len(group)} actions ({names})")
            self.conflict_label.setText(
                "⚠ Conflicting shortcuts: Qt treats these as ambiguous, so none of the "
                "actions below fires from the keyboard.<br>" + "<br>".join(parts))
        layout.addWidget(self.conflict_label)

        self.table = QTableWidget(len(self._rows), 3)
        self.table.setHorizontalHeaderLabels(["Menu", "Action", "Shortcut"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setAlternatingRowColors(True)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        for r, row in enumerate(self._rows):
            for c, text in enumerate((row.group, row.action, row.keys)):
                item = QTableWidgetItem(text)
                if not row.enabled:
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEnabled)
                self.table.setItem(r, c, item)
        layout.addWidget(self.table, 1)

        buttons = QHBoxLayout()
        self.copy_button = QPushButton("Copy as Markdown")
        self.copy_button.clicked.connect(self.copy_markdown)
        close = QPushButton("Close")
        close.clicked.connect(self.close)
        buttons.addWidget(self.copy_button)
        buttons.addStretch(1)
        buttons.addWidget(close)
        layout.addLayout(buttons)

    def rows(self) -> list[ShortcutRow]:
        return list(self._rows)

    def visible_row_count(self) -> int:
        return sum(1 for r in range(self.table.rowCount()) if not self.table.isRowHidden(r))

    def _apply_filter(self, text: str) -> None:
        for r, row in enumerate(self._rows):
            self.table.setRowHidden(r, not matches(row, text))

    def copy_markdown(self) -> str:
        text = to_markdown([r for r in self._rows if r.portable], EXTRA_SHORTCUTS)
        QGuiApplication.clipboard().setText(text)
        return text
