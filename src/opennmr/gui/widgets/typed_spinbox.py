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

:class:`TypedSpinBox` is the whole-number version. Qt's QSpinBox has the
same flaw: with "999" in Process ▸ Truncate to, typing a fourth digit
makes 9,99x -- refused when over the FID's length -- so the box seemed
capped at three figures; the export DPI box likewise (explicit user
report, 2026-10-14).

**"k" shorthand** (explicit user request, 2026-10-20), as in TopSpin's
SI/TD boxes: in any whole-number box, ``2k`` is read as 2048, ``8k`` as
8192, ``16k`` as 16384 -- the power of two nearest the number of
thousands typed (:func:`expand_k`). It is read when the entry is
committed (Enter, Tab, leaving the box), like any other typed number.
The decimal boxes ignore it: a power of two means nothing for a
frequency or a shift.
"""

from __future__ import annotations

import math
import re

from PySide6.QtCore import Signal
from PySide6.QtGui import QValidator
from PySide6.QtWidgets import QDoubleSpinBox, QSpinBox

_PARTIAL = {"", "-", "+", ".", "-.", "+.", ","}

_K_RE = re.compile(r"^\s*(\d+(?:[.,]\d+)?|[.,]\d+)\s*[kK]\s*$")


def expand_k(text: str) -> int | None:
    """``"2k"`` -> 2048, ``"8K"`` -> 8192, ``"16k"`` -> 16384: the power
    of two nearest ``number x 1000``, nearest on a logarithmic scale
    (so ``3k`` -- 3000, between 2048 and 4096 -- gives 4096, and ``2.5k``
    gives 2048). ``None`` if ``text`` is not a number followed by ``k``
    (or is zero)."""
    match = _K_RE.match(str(text))
    if not match:
        return None
    thousands = float(match.group(1).replace(",", "."))
    if thousands <= 0:
        return None
    return 1 << max(0, int(round(math.log2(thousands * 1000.0))))


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


class TypedSpinBox(QSpinBox):
    """Whole numbers, typed freely; see the module notes."""

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
        return t.strip().replace(",", "").replace(" ", "")

    def _parse(self, text: str):
        t = self._number_text(text)
        try:
            return int(t)
        except ValueError:
            return expand_k(t)           # "2k" -> 2048 (None if not that either)

    def _is_special(self, text: str) -> bool:
        special = self.specialValueText()
        return bool(special) and str(text).strip() == special.strip()

    def validate(self, text: str, pos: int):  # noqa: D401 -- Qt override
        # A box with special text ("Auto", "all", ...) shows it for its
        # minimum: it must stay valid, and typing towards it must not be
        # refused (the override would otherwise hide Qt's own handling).
        special = self.specialValueText()
        if special:
            if self._is_special(text):
                return QValidator.State.Acceptable, text, pos
            if text and special.lower().startswith(str(text).strip().lower()):
                return QValidator.State.Intermediate, text, pos
        t = self._number_text(text)
        if t in ("", "-", "+"):
            return QValidator.State.Intermediate, text, pos
        if re.fullmatch(r"\d*[.,]\d*", t):                # "2." / "2.5": heading for "2.5k"
            return QValidator.State.Intermediate, text, pos
        value = self._parse(text)
        if value is None:
            return QValidator.State.Invalid, text, pos
        if self.minimum() <= value <= self.maximum():
            return QValidator.State.Acceptable, text, pos
        return QValidator.State.Intermediate, text, pos        # typed past a limit: keep it

    def valueFromText(self, text: str) -> int:  # noqa: N802 -- Qt override
        if self._is_special(text):
            return self.minimum()
        value = self._parse(text)
        if value is None:
            try:                            # "12.6" typed in a whole-number box
                value = int(round(float(self._number_text(text).replace(",", "."))))
            except ValueError:
                return self.value()
        return min(max(value, self.minimum()), self.maximum())

    def fixup(self, text: str) -> str:
        if self._is_special(text):
            return self.specialValueText()
        value = self._parse(text)
        if value is None:
            try:                            # "12.6" typed in a whole-number box
                value = int(round(float(self._number_text(text).replace(",", "."))))
            except ValueError:
                value = self.value()
        value = min(max(value, self.minimum()), self.maximum())
        if value == self.minimum() and self.specialValueText():
            return self.specialValueText()
        return self.prefix() + self.textFromValue(value) + self.suffix()


class LengthSpinBox(TypedDoubleSpinBox):
    """A length kept in **inches** (``value()``, ``setValue``, signals --
    what matplotlib's ``figsize`` takes and what saved styles hold),
    shown and typed in inches or centimetres (:meth:`set_unit`). Explicit
    user request, 2026-10-14: export size in cm, inches by default.

    Only the display converts, so every caller, saved style and test that
    deals in inches is unaffected by the choice."""

    UNITS = {"in": 1.0, "cm": 2.54}

    def __init__(self, parent=None, shown_decimals: int = 2) -> None:
        super().__init__(parent)
        self._unit = "in"
        self._shown_decimals = shown_decimals
        # Inches held finely enough that a typed centimetre value comes
        # back as typed (8.5 cm -> 3.34646 in -> 8.50 cm).
        self.setDecimals(6)
        self.set_unit("in")

    def unit(self) -> str:
        return self._unit

    def set_unit(self, unit: str) -> None:
        if unit not in self.UNITS:
            raise ValueError(f"unit must be one of {tuple(self.UNITS)}, not {unit!r}")
        self._unit = unit
        self.setSuffix(f" {unit}")
        # One step = 0.1 in, or 0.5 cm.
        self.setSingleStep(0.1 if unit == "in" else 0.5 / 2.54)
        blocked = self.blockSignals(True)
        value = self.value()
        self.setValue(value)           # redraws the text in the new unit
        self.lineEdit().setText(self.textFromValue(value) + self.suffix())
        self.blockSignals(blocked)

    def textFromValue(self, value: float) -> str:  # noqa: N802 -- Qt override
        return f"{value * self.UNITS[self._unit]:.{self._shown_decimals}f}"

    def _parse(self, text: str):
        shown = super()._parse(text)
        return None if shown is None else shown / self.UNITS[self._unit]
