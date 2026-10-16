"""Cmd/Ctrl+E: a random fact, from a list the user keeps themselves.

Explicit user request (2026-10-16), reachable **only** by the shortcut --
no menu item, button or tooltip mentions it. The facts are not shipped with
openNMR: the user's choice, since the lists they come from (QI, No Such
Thing As A Fish) are their authors' own wording. They live in a plain text
file, one fact per line, in ``~/.opennmr/facts.txt`` (``OPENNMR_FACTS_FILE``
overrides the path); blank lines and lines starting with ``#`` are skipped.
"""

from __future__ import annotations

import os
import random

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QLabel, QPushButton, QVBoxLayout

_TEMPLATE = """\
# openNMR facts -- Cmd/Ctrl+E shows one of these at random.
# One fact per line. Lines starting with # and blank lines are skipped.
"""


def facts_path() -> str:
    return os.environ.get("OPENNMR_FACTS_FILE") or os.path.expanduser("~/.opennmr/facts.txt")


def read_facts(path: str | None = None) -> list[str]:
    """The facts in ``path`` (default :func:`facts_path`); [] when the file
    is missing or holds none."""
    path = path or facts_path()
    try:
        with open(path, encoding="utf-8-sig") as handle:
            lines = handle.read().splitlines()
    except (OSError, UnicodeDecodeError):
        return []
    return [line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")]


def ensure_facts_file(path: str | None = None) -> str:
    """Create the file with a short header if it does not exist; its path."""
    path = path or facts_path()
    if not os.path.exists(path):
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(_TEMPLATE)
        except OSError:
            pass
    return path


def pick_fact(facts: list[str], previous: str | None = None, rng=random) -> str | None:
    """A random fact, not the one just shown when there is a choice."""
    if not facts:
        return None
    choices = [f for f in facts if f != previous] or facts
    return rng.choice(choices)


class FactDialog(QDialog):
    """"Did you know?": one fact, Another, Close. Non-modal."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Did you know?")
        self.setMinimumWidth(420)
        v = QVBoxLayout(self)
        self.fact_label = QLabel("")
        self.fact_label.setWordWrap(True)
        self.fact_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        font = self.fact_label.font()
        font.setPointSizeF(font.pointSizeF() * 1.2)
        self.fact_label.setFont(font)
        v.addWidget(self.fact_label)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        self.another_button = QPushButton("Another")
        buttons.addButton(self.another_button, QDialogButtonBox.ButtonRole.ActionRole)
        self.another_button.clicked.connect(self.show_fact)
        buttons.rejected.connect(self.close)
        v.addWidget(buttons)
        self.current: str | None = None

    def show_fact(self) -> str | None:
        """Read the file afresh (so edits show at once) and show a fact, or
        say where the file is when it holds none."""
        facts = read_facts()
        self.current = pick_fact(facts, self.current)
        if self.current is None:
            path = ensure_facts_file()
            self.fact_label.setText(
                f"No facts yet. Add some to\n{path}\none per line, then press Another.")
        else:
            self.fact_label.setText(self.current)
        return self.current
