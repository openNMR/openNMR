"""The advice under a fit's results: what the fit tells you, and what to try
next, in plain language (:mod:`opennmr.notebook.fit_advice`), with a
button beside each suggestion the panel can carry out in one click."""

from __future__ import annotations

import html

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QGroupBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

#: Button text for each action type.
ACTION_TEXT = {"add_peak": "Add this peak", "add_interaction": "Add it",
               "set_lineshape": "Use pseudo-Voigt", "free_mas": "Free the MAS rate",
               "free_parameter": "Free it", "fix_parameter": "Fix it",
               "widen_bound": "Widen the limit", "remove_site": "Remove it",
               "set_relaxation_model": "Switch model"}
_MARK = {"good": ("✓", "#1b7f3b"), "note": ("•", "#555555"), "warn": ("⚠", "#a15c00")}


def action_text(action: dict) -> str:
    if action.get("type") == "add_interaction":
        return {"csa": "Add a CSA", "quadrupolar": "Add quadrupolar",
                "jcoupling": "Add a J-coupling"}.get(action.get("kind"), "Add it")
    return ACTION_TEXT.get(action.get("type"), "Do it")


class AdviceBox(QGroupBox):
    """``set_items(items)`` shows advice; ``on_action(action) -> str | None``
    (optional) carries out a suggestion and returns a line saying what was
    done. Hidden while there is nothing to say."""

    def __init__(self, on_action=None, title="What the fit tells you, and what to try next",
                 parent=None) -> None:
        super().__init__(title, parent)
        self._on_action = on_action
        self.items: list[dict] = []
        self.action_buttons: list[QPushButton] = []
        self._layout = QVBoxLayout(self)
        self.done_label = QLabel("")
        self.done_label.setWordWrap(True)
        self.setVisible(False)

    def _clear(self) -> None:
        while self._layout.count():
            item = self._layout.takeAt(0)
            widget = item.widget()
            if widget is not None and widget is not self.done_label:
                widget.hide()
                widget.setParent(None)
                widget.deleteLater()
        self.action_buttons = []

    def set_items(self, items) -> None:
        self.items = list(items or [])
        self._clear()
        self.done_label.setText("")
        if not self.items:
            self.setVisible(False)
            return
        tells = [it for it in self.items if it["section"] == "tells"]
        nexts = [it for it in self.items if it["section"] == "next"]
        if tells:
            self._layout.addWidget(self._heading("What the fit tells you"))
            for it in tells:
                mark, colour = _MARK.get(it["tone"], _MARK["note"])
                self._layout.addWidget(self._line(
                    f"<span style='color:{colour}'>{mark}</span>&nbsp;{html.escape(it['text'])}"))
        if nexts:
            self._layout.addWidget(self._heading("What to try next"))
            for it in nexts:
                row = QWidget()
                h = QHBoxLayout(row)
                h.setContentsMargins(0, 0, 0, 0)
                h.addWidget(self._line(f"→&nbsp;{html.escape(it['text'])}"), 1)
                action = it.get("action")
                if action and self._on_action is not None:
                    button = QPushButton(action_text(action))
                    button.setToolTip("Make this change to the model now; fit again to see "
                                      "the effect.")
                    button.clicked.connect(lambda _c=False, a=action, b=button: self._run(a, b))
                    h.addWidget(button, 0, Qt.AlignmentFlag.AlignTop)
                    self.action_buttons.append(button)
                self._layout.addWidget(row)
        self._layout.addWidget(self.done_label)
        self.setVisible(True)

    def _run(self, action, button) -> None:
        done = self._on_action(action) if self._on_action is not None else None
        button.setEnabled(False)
        if done:
            self.done_label.setText(f"Done: {done} Fit again to see the effect.")

    @staticmethod
    def _heading(text):
        label = QLabel(f"<b>{html.escape(text)}</b>")
        label.setTextFormat(Qt.TextFormat.RichText)
        return label

    @staticmethod
    def _line(rich):
        label = QLabel(rich)
        label.setTextFormat(Qt.TextFormat.RichText)
        label.setWordWrap(True)
        label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        return label

    def text(self) -> str:
        from ...notebook.fit_advice import advice_text
        return advice_text(self.items)
