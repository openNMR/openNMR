"""Keep pop-up message boxes on the screen (notes 8.6).

A ``QMessageBox`` sizes itself to its text, so a long message - an update's
release notes, an install log, a pip error tail - makes a box taller than a
small laptop screen and pushes its buttons off the bottom, where they cannot
be reached. :func:`fit_message_box_to_screen` puts the long text in a scroll
area limited to the screen's available height, so the buttons always stay in
view. :class:`ScreenFitFilter` applies it to every message box the application
shows, so no dialog has to remember to ask for it.
"""
from __future__ import annotations

from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QApplication, QFrame, QGridLayout, QLabel, QMessageBox, QScrollArea

#: Share of the screen's available height a message box may take.
MAX_SCREEN_FRACTION = 0.9
#: Height kept free inside the box for the title bar, buttons and margins.
CHROME_ALLOWANCE = 150
_LABELS = ("qt_msgbox_label", "qt_msgbox_informativelabel")


def _available_height(widget) -> int:
    screen = widget.screen() if widget.windowHandle() is not None else None
    screen = screen or QGuiApplication.primaryScreen()
    return screen.availableGeometry().height() if screen is not None else 600


def fit_message_box_to_screen(box: QMessageBox) -> bool:
    """Limit ``box`` to the screen and make its text scroll. True when it had to."""
    if box.property("_screen_fitted"):
        return False
    box.setProperty("_screen_fitted", True)
    limit = int(_available_height(box) * MAX_SCREEN_FRACTION)
    box.adjustSize()
    if box.sizeHint().height() <= limit:
        return False
    layout = box.layout()
    if not isinstance(layout, QGridLayout):
        return False
    text_room = max(120, limit - CHROME_ALLOWANCE)
    for name in _LABELS:
        label = box.findChild(QLabel, name)
        if label is None or not label.text():
            continue
        index = layout.indexOf(label)
        if index < 0:
            continue
        row, col, row_span, col_span = layout.getItemPosition(index)
        scroll = QScrollArea(box)
        scroll.setObjectName(name + "_scroll")
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        layout.removeWidget(label)
        label.setWordWrap(True)
        label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        scroll.setWidget(label)
        # Wide enough to read without a horizontal scroll bar, never taller than the room.
        scroll.setMinimumWidth(min(label.sizeHint().width(), 560) + 24)
        scroll.setMaximumHeight(text_room)
        layout.addWidget(scroll, row, col, row_span, col_span)
    box.setMaximumHeight(limit)
    box.adjustSize()
    return True


class ScreenFitFilter(QObject):
    """Application-wide event filter: every message box is fitted as it is shown."""

    def eventFilter(self, obj, event):  # noqa: N802 (Qt API)
        if event.type() == QEvent.Type.Show and isinstance(obj, QMessageBox):
            fit_message_box_to_screen(obj)
        return False


def install_screen_fit(app: QApplication | None = None) -> ScreenFitFilter | None:
    """Install the filter once per application (safe to call repeatedly)."""
    app = app or QApplication.instance()
    if app is None:
        return None
    existing = getattr(app, "_opennmr_screen_fit", None)
    if existing is not None:
        return existing
    flt = ScreenFitFilter(app)
    app.installEventFilter(flt)
    app._opennmr_screen_fit = flt
    return flt
