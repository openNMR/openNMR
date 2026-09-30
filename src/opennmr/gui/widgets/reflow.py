"""
Make a side panel's contents reflow to the width the user drags the
left-hand column to (explicit user request, 2026-09-30: "when someone
re-sizes the width of the left-hand menu, the text comments [should]
rearrange to match the width" -- on the Simulate and Fit tiles they did
not move).

Why they did not move: the side panels sit in a scroll area whose
horizontal scroll bar is off on purpose (``MainWindow._scrollable``), so
a panel is always laid out at least as wide as its *minimum* width. Rows
of two or three buttons side by side, a combo box sized to its longest
item, or a form row of a label beside a wide field each set that minimum
-- up to ~590 px on Fit and Simulate, against a 280 px column. Below it,
nothing reflows: word-wrapped text keeps the width it has and the right
of the panel is cut off.

:func:`make_reflowable` lowers every panel's minimum width without
rebuilding it, so it also covers panels added later:

* **Form layouts** wrap a field under its label when the two do not fit
  side by side (``QFormLayout.WrapLongRows``).
* **Rows** (``QHBoxLayout``) wider than the limit become a
  :class:`WrapRowLayout`, which flows onto a second line when narrow and
  fills the width when not (a label stays with the control after it).
* **Combo boxes** wider than the limit stop sizing themselves to their
  longest item (the popup still shows it whole).
* **Plain-text labels** wider than the limit word-wrap.
* **Push buttons** with long text break it onto two lines when narrower
  than it (:class:`_WrapButtonText`).

Collapsible-section titles wrap in the same way
(``CollapsibleSection``).
"""

from __future__ import annotations

from PySide6.QtCore import QEvent, QObject, QRect, QSize, Qt
from PySide6.QtWidgets import (
    QAbstractButton, QBoxLayout, QCheckBox, QComboBox, QFormLayout, QGridLayout, QGroupBox,
    QHBoxLayout, QLabel, QLayout, QPushButton, QRadioButton, QSizePolicy, QToolButton, QWidget,
)

#: A control wider than this (px) is made narrower-capable (group boxes
#: nest up to three deep, each adding ~20 px of margin). A side
#: panel's usable width at the column's 280 px minimum, less scroll bar,
#: margins and a section's indent, is a little over 200 px.
DEFAULT_LIMIT = 160


class WrapRowLayout(QLayout):
    """A row that wraps: items left to right, onto the next line when the
    next would not fit. With ``justify`` each line's spare width is shared
    among its items, so an unwrapped row looks like the ``QHBoxLayout``
    it replaced; hidden items take no room."""

    def __init__(self, parent=None, spacing: int = 6, justify: bool = True) -> None:
        super().__init__(parent)
        self._items: list = []
        self._justify = justify
        self.setContentsMargins(0, 0, 0, 0)
        self.setSpacing(spacing)

    # -- QLayout plumbing --------------------------------------------------
    def addItem(self, item) -> None:  # noqa: N802
        self._items.append(item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index):  # noqa: N802
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index):  # noqa: N802
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self):  # noqa: N802
        return Qt.Orientation.Horizontal if self._justify else Qt.Orientation(0)

    def hasHeightForWidth(self) -> bool:  # noqa: N802
        return True

    def heightForWidth(self, width: int) -> int:  # noqa: N802
        return self._arrange(QRect(0, 0, width, 0), apply=False)

    def setGeometry(self, rect) -> None:  # noqa: N802
        super().setGeometry(rect)
        self._arrange(rect, apply=True)

    def _visible(self):
        return [item for item in self._items if not item.isEmpty()]

    @staticmethod
    def _width(item) -> int:
        return max(item.sizeHint().width(), item.minimumSize().width())

    def sizeHint(self) -> QSize:  # noqa: N802
        items = self._visible()
        width = sum(self._width(i) for i in items) + self.spacing() * max(0, len(items) - 1)
        height = max((i.sizeHint().height() for i in items), default=0)
        m = self.contentsMargins()
        return QSize(width + m.left() + m.right(), height + m.top() + m.bottom())

    def minimumSize(self) -> QSize:  # noqa: N802
        size = QSize(0, 0)
        for item in self._visible():
            size = size.expandedTo(item.minimumSize())
        m = self.contentsMargins()
        return size + QSize(m.left() + m.right(), m.top() + m.bottom())

    def _arrange(self, rect: QRect, apply: bool) -> int:
        m = self.contentsMargins()
        left, top = rect.x() + m.left(), rect.y() + m.top()
        avail = max(1, rect.width() - m.left() - m.right())
        gap = self.spacing()
        lines, line, used = [], [], 0
        for item in self._visible():
            w = min(self._width(item), avail)
            if line and used + gap + w > avail:
                lines.append((line, used))
                line, used = [], 0
            used += (gap if line else 0) + w
            line.append((item, w))
        if line:
            lines.append((line, used))
        y = top
        for line, used in lines:
            height = max(item.sizeHint().height() for item, _w in line)
            growable = [k for k, (item, _w) in enumerate(line) if self._can_grow(item)]
            spare = max(0, avail - used) if self._justify and growable else 0
            x = left
            for k, (item, w) in enumerate(line):
                if spare and k in growable:
                    share = spare // len(growable) + (1 if growable.index(k) < spare % len(growable)
                                                      else 0)
                    w += share
                if apply:
                    h = min(height, max(item.sizeHint().height(), item.minimumSize().height()))
                    item.setGeometry(QRect(x, y + (height - h) // 2, w, h))
                x += w + gap
            y += height + gap
        return (y - gap if lines else y) - rect.y() + m.bottom()

    @staticmethod
    def _can_grow(item) -> bool:
        widget = item.widget()
        if widget is None:
            return True
        if isinstance(widget, QLabel):
            return False
        policy = widget.sizePolicy().horizontalPolicy()
        return policy not in (QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Maximum)


class _WrapRow(QWidget):
    """The widget a converted row lives in. Its height depends on its
    width, which the size policy must say for a parent layout to ask."""

    def __init__(self, justify: bool) -> None:
        super().__init__()
        self.flow = WrapRowLayout(self, justify=justify)
        policy = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        policy.setHeightForWidth(True)
        self.setSizePolicy(policy)

    def hasHeightForWidth(self) -> bool:  # noqa: N802
        return True

    def heightForWidth(self, width: int) -> int:  # noqa: N802
        return self.flow.heightForWidth(width)


class _WrapButtonText(QObject):
    """Lets a button (push button, check box, radio button, tool button)
    with long text be narrower than it, breaking the text onto two (or
    more) lines at spaces when it has to."""

    def __init__(self, button: QAbstractButton, minimum: int) -> None:
        super().__init__(button)
        self._button = button
        self._full = button.text()
        self._busy = False
        button.setMinimumWidth(minimum)
        button.installEventFilter(self)

    def full_text(self) -> str:
        return self._full

    def eventFilter(self, obj, event) -> bool:  # noqa: N802
        if obj is self._button and not self._busy:
            if event.type() == QEvent.Type.Resize:
                self._fit(event.size().width())
            elif event.type() == QEvent.Type.DynamicPropertyChange:
                pass
        return False

    def _fit(self, width: int) -> None:
        button = self._button
        shown = button.text()
        if "\n" not in shown and shown != self._full:
            self._full = shown                       # the text was changed by code
        metrics = button.fontMetrics()
        # Text room: a check box or radio button also holds its indicator.
        extra = 34 if isinstance(button, (QCheckBox, QRadioButton)) else 24
        room = max(20, width - extra)
        if metrics.horizontalAdvance(self._full) <= room:
            text = self._full
        else:
            lines, line = [], ""
            for word in self._full.split(" "):
                trial = f"{line} {word}".strip()
                if line and metrics.horizontalAdvance(trial) > room:
                    lines.append(line)
                    line = word
                else:
                    line = trial
            lines.append(line)
            text = "\n".join(lines)
        if text != shown:
            self._busy = True
            try:
                button.setText(text)
            finally:
                self._busy = False


class _FitGroupTitle(QObject):
    """A group box's title sets its minimum width and cannot wrap. This
    pins the box's minimum to what its *contents* need instead, and
    shortens the title with "…" (full title in the tooltip) while the box
    is narrower than it."""

    def __init__(self, box: QGroupBox) -> None:
        super().__init__(box)
        self._box = box
        self._full = box.title()
        self._busy = False
        if not box.toolTip():
            box.setToolTip(self._full)
        box.installEventFilter(self)
        self._update(box.width())

    def full_title(self) -> str:
        return self._full

    def eventFilter(self, obj, event) -> bool:  # noqa: N802
        if obj is self._box and not self._busy and event.type() in (
                QEvent.Type.Resize, QEvent.Type.LayoutRequest, QEvent.Type.Show):
            self._update(self._box.width())
        return False

    def _update(self, width: int) -> None:
        box = self._box
        self._busy = True
        try:
            shown = box.title()
            if "…" not in shown and shown != self._full:
                self._full = shown                   # retitled by code
            layout = box.layout()
            need = layout.totalMinimumSize().width() if layout is not None else 0
            if box.minimumWidth() != max(need, 40):
                box.setMinimumWidth(max(need, 40))
            metrics = box.fontMetrics()
            room = max(20, width - 24)
            title = (self._full if width <= 0 or metrics.horizontalAdvance(self._full) <= room
                     else metrics.elidedText(self._full, Qt.TextElideMode.ElideRight, room))
            if title != shown:
                box.setTitle(title)
        finally:
            self._busy = False


def group_full_title(box: QGroupBox) -> str:
    """A group box's title without the "…" shortening it may be shown with."""
    for child in box.children():
        if isinstance(child, _FitGroupTitle):
            return child.full_title()
    return box.title()


def button_full_text(button: QAbstractButton) -> str:
    """A button's text without the line breaks :func:`make_reflowable`
    may have put in it."""
    for child in button.children():
        if isinstance(child, _WrapButtonText):
            return child.full_text()
    return button.text()


def _replace_in_parent(layout: QLayout, container: QWidget) -> bool:
    """Put ``container`` where ``layout`` sits in its parent layout."""
    parent = layout.parent()
    if isinstance(parent, QBoxLayout):
        index = parent.indexOf(layout)
        if index < 0:
            return False
        stretch = parent.stretch(index)
        parent.removeItem(layout)
        parent.insertWidget(index, container, stretch)
    elif isinstance(parent, QFormLayout):
        row, role = parent.getLayoutPosition(layout)
        if row < 0:
            return False
        parent.removeItem(layout)
        parent.setWidget(row, role, container)
    elif isinstance(parent, QGridLayout):
        index = parent.indexOf(layout)
        if index < 0:
            return False
        r, c, rs, cs = parent.getItemPosition(index)
        parent.removeItem(layout)
        parent.addWidget(container, r, c, rs, cs)
    else:
        return False
    layout.setParent(None)
    return True


def _holder(layout: QLayout) -> QWidget:
    holder = QWidget()
    layout.setParent(None)
    holder.setLayout(layout)
    return holder


def convert_row(row: QHBoxLayout, limit: int = DEFAULT_LIMIT) -> QWidget | None:
    """Turn ``row`` into a wrapping row in place; returns the widget it
    now lives in (for a widget's own top-level row, that widget), or None
    where it cannot be moved."""
    owner = row.parent()
    top_level = isinstance(owner, QWidget) and owner.layout() is row
    if not top_level and not isinstance(owner, (QBoxLayout, QFormLayout, QGridLayout)):
        return None
    items = []
    stretch = False
    parent_layout = row.parent()
    for index in range(row.count()):
        item = row.itemAt(index)
        if item.widget() is not None:
            items.append(item.widget())
        elif item.layout() is not None:
            items.append(item.layout())
        else:
            stretch = True
    for entry in list(items):
        if isinstance(entry, QLayout):
            row.removeItem(entry)
        else:
            row.removeWidget(entry)
    items = [_holder(e) if isinstance(e, QLayout) else e for e in items]
    del parent_layout
    if top_level:
        margins = row.contentsMargins()
        spare = QWidget()
        spare.setLayout(row)                       # takes it off the owner
        owner._replaced_layout = spare
        container = owner
        flow = WrapRowLayout(owner, justify=not stretch)
        flow.setContentsMargins(margins)
        policy = owner.sizePolicy()
        policy.setHeightForWidth(True)
        owner.setSizePolicy(policy)
    else:
        container = _WrapRow(justify=not stretch)
        flow = container.flow
    k = 0
    while k < len(items):
        widget = items[k]
        # "Label: [control]" stays together on one line.
        if (isinstance(widget, QLabel) and not widget.wordWrap() and k + 1 < len(items)
                and not isinstance(items[k + 1], QLabel)
                and widget.minimumSizeHint().width()
                + items[k + 1].minimumSizeHint().width() + 6 <= limit):
            pair = QHBoxLayout()
            pair.setContentsMargins(0, 0, 0, 0)
            pair.addWidget(widget)
            pair.addWidget(items[k + 1], 1)
            pair_widget = QWidget()
            pair_widget.setLayout(pair)
            flow.addWidget(pair_widget)
            k += 2
            continue
        flow.addWidget(widget)
        k += 1
    if top_level:
        return container
    if not _replace_in_parent(row, container):
        return None
    # The emptied row is out of every layout now; the container keeps the
    # only reference, so Python never deletes it while Qt might still.
    container._replaced_layout = row
    return container


def _layouts(root: QWidget):
    """Every layout under ``root`` (its own and its descendants')."""
    seen = []
    widgets = [root] + root.findChildren(QWidget)
    for widget in widgets:
        top = widget.layout()
        if top is None:
            continue
        stack = [top]
        while stack:
            layout = stack.pop()
            if layout in seen:
                continue
            seen.append(layout)
            for i in range(layout.count()):
                child = layout.itemAt(i).layout() if layout.itemAt(i) else None
                if child is not None:
                    stack.append(child)
    return seen


def make_reflowable(root: QWidget, limit: int = DEFAULT_LIMIT) -> None:
    """Lower ``root``'s minimum width so its contents reflow down to about
    ``limit`` px per control (see the module notes). Safe to call again,
    e.g. after more controls were added."""
    for layout in _layouts(root):
        if isinstance(layout, QFormLayout):
            layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
    for combo in root.findChildren(QComboBox):
        if combo.minimumSizeHint().width() > limit or combo.minimumWidth() > limit:
            combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(min(combo.minimumContentsLength() or 12, 12))
            # QComboBox keeps the minimum size hint it computed first; an
            # explicit minimum width is what the layout then goes by.
            combo.setMinimumWidth(min(limit, 120))
    for label in root.findChildren(QLabel):
        if (not label.wordWrap() and label.minimumSizeHint().width() > limit
                and label.textFormat() != Qt.TextFormat.RichText and label.pixmap().isNull()):
            label.setWordWrap(True)
    for button in root.findChildren(QAbstractButton):
        if isinstance(button, QToolButton) and button.parent() is not None and \
                type(button.parent()).__name__ == "CollapsibleSection":
            continue                                   # wraps itself
        if (button.minimumSizeHint().width() > limit and " " in button.text()
                and not any(isinstance(c, _WrapButtonText) for c in button.children())):
            _WrapButtonText(button, minimum=max(80, limit // 2))
    for box in root.findChildren(QGroupBox):
        metrics = box.fontMetrics()
        if (box.title() and metrics.horizontalAdvance(box.title()) + 30 > limit
                and not any(isinstance(c, _FitGroupTitle) for c in box.children())):
            _FitGroupTitle(box)
    # Rows last: their minimum is only known once what is in them has
    # been made narrower-capable. Innermost first, so a row inside a
    # row is judged on its own.
    rows = [layout for layout in _layouts(root) if type(layout) is QHBoxLayout]
    rows.sort(key=_depth, reverse=True)
    for row in rows:
        if row.minimumSize().width() > limit:
            convert_row(row, limit)


def _depth(layout: QLayout) -> int:
    depth, parent = 0, layout.parent()
    while parent is not None:
        depth += 1
        parent = parent.parent()
    return depth
