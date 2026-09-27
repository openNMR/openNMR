"""
The mouse wheel never changes a value box.

Qt's spin boxes, combo boxes and sliders take the wheel whenever the
pointer is over them, focused or not. In a long scrolling panel that
means scrolling *past* a box changes it: reported as the Larmor frequency
changing under the pointer while the user was only scrolling the panel
("very annoying, and may cause unwanted problems" -- explicit user
request, 2026-09-27, to remove it).

:class:`WheelGuard` is installed once on the application. A wheel event
aimed at a spin box, combo box or slider (or at the line edit inside a
spin box) is not delivered to it; it is passed to the nearest scroll
area instead, so the panel scrolls as the user meant. Values still change
by typing, the arrow keys, the spin buttons and the drop-down.

Deliberately untouched: scroll bars (their job *is* the wheel), the
plots (the wheel zooms them) and the phase dials (``DragHandle``, whose
wheel nudge is a documented part of the control).
"""

from __future__ import annotations

from PySide6.QtCore import QEvent, QObject
from PySide6.QtWidgets import (
    QAbstractItemView, QAbstractScrollArea, QAbstractSpinBox, QApplication, QComboBox,
    QScrollBar, QSlider, QWidget,
)

#: The widgets whose values the wheel must not change.
GUARDED = (QAbstractSpinBox, QComboBox, QSlider)


def _guarded_owner(obj):
    """The value box ``obj`` is, or belongs to (a spin box's own line
    edit receives its wheel events), or None. An open drop-down list or a
    scroll bar on the way up means the wheel is meant for that list, not
    for the box it belongs to (a combo box's pop-up is its child)."""
    widget = obj if isinstance(obj, QWidget) else None
    depth = 0
    while widget is not None and depth < 4:
        if isinstance(widget, GUARDED):
            return widget
        if isinstance(widget, (QAbstractItemView, QScrollBar)):
            return None
        widget = widget.parentWidget()
        depth += 1
    return None


class WheelGuard(QObject):
    """Application event filter: see the module docstring."""

    def eventFilter(self, obj, event) -> bool:  # noqa: N802 -- Qt override
        if event.type() != QEvent.Type.Wheel:
            return False
        box = _guarded_owner(obj)
        if box is None:
            return False
        area = box.parentWidget()
        while area is not None and not isinstance(area, QAbstractScrollArea):
            area = area.parentWidget()
        if area is not None:
            QApplication.sendEvent(area.viewport(), event)
        return True                      # never to the box itself


def install(app=None) -> WheelGuard | None:
    """Install the guard on ``app`` (default: the running application),
    once. Returns it."""
    app = app or QApplication.instance()
    if app is None:
        return None
    guard = getattr(app, "_opennmr_wheel_guard", None)
    if guard is None:
        guard = WheelGuard(app)
        app.installEventFilter(guard)
        app._opennmr_wheel_guard = guard
    return guard
