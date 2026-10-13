"""A number box you can simply type into.

Qt's QDoubleSpinBox rejects any keystroke that would make the text
invalid *on the way* to the number you want: with six decimals shown,
typing a digit at the end is refused (one decimal too many); typing "3"
then "0" in front of "9.394641 T" is refused at the "0" (309 T is over
the limit). In the NMR Periodic Table this meant 30 T could only be
reached by clicking the up arrow -- explicit user request, 2026-10-13.

:class:`TypedDoubleSpinBox` accepts any number while it is being typed:
out of range or with more decimals than shown, the text is kept as
typed, and on Enter or leaving the box the value is rounded to the
decimals shown and brought inside the range. Arrow keys, the buttons
and ``setValue`` behave exactly as before.
"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtGui import QValidator
from PySide6.QtWidgets import QDoubleSpinBox

_PARTIAL = {"", "-", "+", ".", "-.", "+.", ","}


class TypedDoubleSpinBox(QDoubleSpinBox):
    """See the module notes. ``userEdited`` fires when the *user* changes
    the value (typing, arrow keys, buttons), not when code does."""

    userEdited = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.lineEdit().textEdited.connect(lambda _t: self.userEdited.emit())

    def stepBy(self, steps: int) -> None:  # noqa: N802 -- Qt override
        super().stepBy(steps)
        self.userEdited.emit()

    def _number_text(self, text: str) -> str:
        t = str(text)
        if self.prefix() and t.startswith(self.prefix()):
            t = t[len(self.prefix()):]
        if self.suffix() and t.endswith(self.suffix()):
            t = t[: -len(self.suffix())]
        elif self.suffix().strip() and t.rstrip().endswith(self.suffix().strip()):
            t = t.rstrip()[: -len(self.suffix().strip())]
        return t.strip().replace(",", ".")

    def _parse(self, text: str):
        try:
            return float(self._number_text(text))
        except ValueError:
            return None

    def validate(self, text: str, pos: int):  # noqa: D401 -- Qt override
        t = self._number_text(text)
        if t in _PARTIAL or t.lower().endswith(("e", "e-", "e+")):
            return QValidator.State.Intermediate, text, pos
        value = self._parse(text)
        if value is None:
            return QValidator.State.Invalid, text, pos
        if self.minimum() <= value <= self.maximum():
            return QValidator.State.Acceptable, text, pos
        return QValidator.State.Intermediate, text, pos        # typed past a limit: keep it

    def valueFromText(self, text: str) -> float:  # noqa: N802 -- Qt override
        value = self._parse(text)
        if value is None:
            return self.value()
        return round(min(max(value, self.minimum()), self.maximum()), self.decimals())

    def fixup(self, text: str) -> str:
        value = self._parse(text)
        if value is None:
            value = self.value()
        value = min(max(value, self.minimum()), self.maximum())
        return self.prefix() + self.textFromValue(value) + self.suffix()
