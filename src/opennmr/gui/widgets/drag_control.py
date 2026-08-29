"""A click-and-drag numeric control: finer, effectively unbounded
alternative to a slider for a value like a phase correction, where the
useful range isn't known ahead of time and a slider's fixed
pixel-to-range mapping makes precise adjustment hard.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import QWidget


class DragHandle(QWidget):
    """Click and drag vertically to change a value -- up increases,
    down decreases, with no fixed range (the drag can continue past
    the widget's own edges). Hold Shift while dragging for finer
    (0.1x) control. Scrolling the wheel over it nudges by one
    ``step``. Emits ``valueChanged(float)`` as the value changes.
    """

    valueChanged = Signal(float)

    #: Emitted (no value) when a drag gesture ends -- lets a caller do
    #: something only worth doing once per gesture (not on every
    #: intermediate tick a continuous drag already fires via
    #: valueChanged), e.g. a slower live-preview refresh.
    dragFinished = Signal()

    def __init__(self, sensitivity: float = 0.5, step: float = 1.0, parent=None) -> None:
        super().__init__(parent)
        self.sensitivity = sensitivity
        self.step = step
        self._value = 0.0
        self._drag_start_y: float | None = None
        self._drag_start_value: float = 0.0

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

    def paintEvent(self, event) -> None:  # noqa: N802 -- Qt override
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect().adjusted(1, 1, -1, -1)
        painter.setBrush(QColor(80, 80, 80))
        painter.setPen(QColor(130, 130, 130))
        painter.drawRoundedRect(rect, 6, 6)
        painter.setPen(QColor(220, 220, 220))
        font = painter.font()
        font.setPointSize(16)
        painter.setFont(font)
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "↕")

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
            self.dragFinished.emit()

    def wheelEvent(self, event) -> None:  # noqa: N802 -- Qt override
        direction = 1 if event.angleDelta().y() > 0 else -1
        self.setValue(self._value + direction * self.step)
        event.accept()
