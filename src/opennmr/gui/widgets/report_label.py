"""A result read-out label: plain text in, a readable layout on screen.

``setText(plain)`` keeps the plain report (what ``text()`` returns, and
what is logged, copied and saved) and shows it laid out by
:func:`opennmr.gui.report_format.report_html`. ``setReport(plain, html)``
shows a purpose-made layout for the same plain report. Explicit user
request, 2026-10-13: outputs read as "a wall of text".
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QLabel

from .. import report_format


def _mix(a: QColor, b: QColor, t: float) -> str:
    """``a`` moved a fraction ``t`` of the way towards ``b``, as #rrggbb."""
    return QColor(round(a.red() + (b.red() - a.red()) * t),
                  round(a.green() + (b.green() - a.green()) * t),
                  round(a.blue() + (b.blue() - a.blue()) * t)).name()


def palette_colours(widget) -> tuple[str, str]:
    """(muted, warning) colours for rich text shown in ``widget``: the
    muted one mixed from its own palette, so light and dark themes read."""
    pal = widget.palette()
    text, window = pal.color(QPalette.ColorRole.WindowText), pal.color(QPalette.ColorRole.Window)
    return _mix(text, window, 0.42), report_format.WARN


class ReportLabel(QLabel):
    """A word-wrapped, selectable QLabel whose ``text()`` is the plain
    report and whose display is rich text (see the module notes)."""

    def __init__(self, text: str = "", parent=None) -> None:
        super().__init__(parent)
        self._plain = ""
        self._html_override: str | None = None
        self.setTextFormat(Qt.TextFormat.RichText)
        self.setWordWrap(True)
        self.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.setText(text)

    # -- colours from the palette, so light and dark themes both read --
    def colours(self) -> tuple[str, str]:
        return palette_colours(self)

    def setText(self, text: str) -> None:  # noqa: N802 -- Qt's name
        self._plain = "" if text is None else str(text)
        self._html_override = None
        self._render()

    def setReport(self, plain: str, html: str) -> None:  # noqa: N802 -- Qt style
        """Show ``html`` for the report whose plain form is ``plain``."""
        self._plain = "" if plain is None else str(plain)
        self._html_override = html or ""
        self._render()

    def _render(self) -> None:
        if self._html_override is not None:
            shown = self._html_override
        else:
            muted, warn = self.colours()
            shown = report_format.report_html(self._plain, muted=muted, warn=warn)
        super().setText(shown)

    def text(self) -> str:
        """The plain report (not the rich text on screen)."""
        return self._plain

    def html(self) -> str:
        """The rich text on screen."""
        return super().text()

    def changeEvent(self, event) -> None:  # noqa: N802 -- Qt's name
        super().changeEvent(event)
        if event.type() == event.Type.PaletteChange and self._html_override is None:
            self._render()             # a theme switch: new muted colour
