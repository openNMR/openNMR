"""A click-and-drag numeric control: finer, effectively unbounded
alternative to a slider for a value like a phase correction, where the
useful range isn't known ahead of time and a slider's fixed
pixel-to-range mapping makes precise adjustment hard.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPainter, QPalette
from PySide6.QtWidgets import QWidget


class DragHandle(QWidget):
    """Click and drag vertically to change a value -- up increases,
    down decreases, with no fixed range (the drag can continue past
    the widget's own edges). Hold Shift while dragging for finer
    (0.1x) control. Scrolling the wheel over it nudges by one
    ``step``. Emits ``valueChanged(float)`` as the value changes.
    """

    valueChanged = Signal(float)

    def __init__(self, sensitivity: float = 0.5, step: float = 1.0, parent=None) -> None:
        super().__init__(parent)
        self.sensitivity = sensitivity
        self.step = step
        self._value = 0.0
        self._drag_start_y: float | None = None
        self._drag_start_value: float = 0.0
        self._hovered = False

        self.setFixedSize(44, 40)
        self.setCursor(Qt.CursorShape.SizeVerCursor)
        self.setToolTip("Click and drag up/down to adjust (hold Shift to fine-tune)")

    def value(self) -> float:
        return self._value

    def setValue(self, value: float, emit: bool = True) -> None:
        if value == self._value:
            return
        self._value = value
        self.update()
        if emit:
            self.valueChanged.emit(value)

    #: Corner radius, matching the rest of the interface's controls
    #: (``gui.style.RADIUS``). Not imported from there: this widget has
    #: no other reason to depend on the theme module, and a handle a
    #: pixel off the buttons beside it is not worth the coupling.
    _RADIUS = 6

    def paintEvent(self, event) -> None:  # noqa: N802 -- Qt override
        """Painted from the **palette**, not from fixed colours.

        This used to be a hardcoded dark grey (fill 80,80,80; glyph
        220,220,220), which made it the one control in the window that
        ignored the theme completely -- it stayed dark in light mode
        however the palette or style was set, because nothing about a
        literal QColor consults either. Reported as "the phasing
        buttons are still dark in light mode".

        It is drawn in the highlight colour, so it reads as the
        accent-coloured control it is ("maybe make them blue like the
        headings") and follows both themes for free.
        """
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect().adjusted(1, 1, -1, -1)

        palette = self.palette()
        fill = palette.color(QPalette.ColorRole.Highlight)
        if not self.isEnabled():
            fill = palette.color(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Button)
        elif self._drag_start_y is not None:
            fill = fill.darker(115)          # pressed, while a drag is live
        elif self._hovered:
            fill = fill.lighter(112)

        painter.setBrush(fill)
        painter.setPen(fill.darker(125))
        painter.drawRoundedRect(rect, self._RADIUS, self._RADIUS)
        painter.setPen(palette.color(QPalette.ColorRole.HighlightedText))
        font = painter.font()
        font.setPointSize(16)
        painter.setFont(font)
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "↕")

    def enterEvent(self, event) -> None:  # noqa: N802 -- Qt override
        self._hovered = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802 -- Qt override
        self._hovered = False
        self.update()
        super().leaveEvent(event)

    def mousePressEvent(self, event) -> None:  # noqa: N802 -- Qt override
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start_y = event.position().y()
            self._drag_start_value = self._value
            self.grabMouse()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 -- Qt override
        if self._drag_start_y is None:
            return
        dy = self._drag_start_y - event.position().y()
        sensitivity = self.sensitivity
        if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            sensitivity *= 0.1
        self.setValue(self._drag_start_value + dy * sensitivity)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 -- Qt override
        if self._drag_start_y is not None:
            self._drag_start_y = None
            self.releaseMouse()

    def wheelEvent(self, event) -> None:  # noqa: N802 -- Qt override
        direction = 1 if event.angleDelta().y() > 0 else -1
        self.setValue(self._value + direction * self.step)
        event.accept()
