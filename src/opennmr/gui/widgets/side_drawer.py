"""
A panel that opens from the right-hand edge of the window.

Built for the experiment's Title / notes (explicit user request,
2026-09-27): under Open experiments the notes box took height from the
experiment list and was itself too small to use. Here the plot keeps
the whole width by default; a long, thin strip at the far right edge
opens the panel beside it. Clicking the strip again -- it stays at the
panel's outer border -- closes it. The window also binds Ctrl+T (Cmd+T
on macOS) to :meth:`SideDrawer.toggle`.

The panel sits in a splitter with the plot, so its width can be dragged;
the width is remembered while the window is open.
"""

from __future__ import annotations

from PySide6.QtCore import QRectF, QSize, Qt, Signal
from PySide6.QtGui import QPainter, QPalette
from PySide6.QtWidgets import (
    QAbstractButton, QHBoxLayout, QSizePolicy, QSplitter, QWidget,
)

#: Width of the closed strip, in pixels.
STRIP_WIDTH = 22


class DrawerStrip(QAbstractButton):
    """The thin vertical button at the window edge: its label reads
    bottom-to-top, with an arrow showing which way a click will move
    the panel."""

    def __init__(self, text: str, parent=None) -> None:
        super().__init__(parent)
        self.setText(text)
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        self.setFixedWidth(STRIP_WIDTH)
        self.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        self._hover = False

    def sizeHint(self) -> QSize:  # noqa: N802 -- Qt override
        return QSize(STRIP_WIDTH, 160)

    def enterEvent(self, event) -> None:  # noqa: N802 -- Qt override
        self._hover = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802 -- Qt override
        self._hover = False
        self.update()
        super().leaveEvent(event)

    def paintEvent(self, _event) -> None:  # noqa: N802 -- Qt override
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        pal = self.palette()
        active = self.isChecked() or self._hover
        painter.fillRect(self.rect(), pal.color(QPalette.ColorRole.Highlight if active
                                                else QPalette.ColorRole.Button))
        painter.setPen(pal.color(QPalette.ColorRole.Mid))
        painter.drawLine(0, 0, 0, self.height())
        painter.setPen(pal.color(QPalette.ColorRole.HighlightedText if active
                                 else QPalette.ColorRole.ButtonText))
        # The label is drawn rotated a quarter-turn anticlockwise, which
        # turns an up-pointing glyph into a left-pointing one: "▴" reads
        # as "opens to the left" (closed), "▾" as "closes to the right".
        arrow = "▾" if self.isChecked() else "▴"
        label = f"{arrow}  {self.text()}  {arrow}"
        painter.translate(self.width() / 2, self.height() / 2)
        painter.rotate(-90)
        metrics = painter.fontMetrics()
        w = metrics.horizontalAdvance(label)
        h = metrics.height()
        painter.drawText(QRectF(-w / 2, -h / 2, w, h), Qt.AlignmentFlag.AlignCenter, label)
        painter.end()


class SideDrawer(QWidget):
    """``main`` (the plot) with ``panel`` opening beside it on the right,
    and the strip on the far right edge. The panel starts closed.

    ``toggled(bool)`` is emitted with the new state."""

    toggled = Signal(bool)

    #: Width the panel opens at the first time, pixels.
    DEFAULT_PANEL_WIDTH = 320

    def __init__(self, main: QWidget, panel: QWidget, label: str, parent=None) -> None:
        super().__init__(parent)
        self.main = main
        self.panel = panel
        self._panel_width = self.DEFAULT_PANEL_WIDTH

        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.addWidget(main)
        self.splitter.addWidget(panel)
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 0)
        panel.setMinimumWidth(180)
        panel.setVisible(False)

        self.strip = DrawerStrip(label)
        self.strip.setToolTip(f"Show or hide the {label} panel (Ctrl+T; ⌘T on macOS).")
        self.strip.toggled.connect(self._on_strip_toggled)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.splitter, 1)
        layout.addWidget(self.strip)

    def isOpen(self) -> bool:  # noqa: N802 -- matches Qt's naming
        return self.strip.isChecked()

    def setOpen(self, open_: bool) -> None:  # noqa: N802
        self.strip.setChecked(bool(open_))

    def toggle(self) -> None:
        self.setOpen(not self.isOpen())

    def _on_strip_toggled(self, open_: bool) -> None:
        if open_:
            self.panel.setVisible(True)
            total = sum(self.splitter.sizes()) or self.width()
            width = min(self._panel_width, max(180, total // 2))
            self.splitter.setSizes([max(1, total - width), width])
        else:
            sizes = self.splitter.sizes()
            if len(sizes) == 2 and sizes[1] > 0:
                self._panel_width = sizes[1]
            self.panel.setVisible(False)
        self.strip.update()
        self.toggled.emit(open_)
