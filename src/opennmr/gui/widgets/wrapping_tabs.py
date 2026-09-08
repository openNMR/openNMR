"""
A tab widget whose tab strip wraps onto as many rows as it needs.

``QTabWidget`` cannot do this. Its bar lays out in a single row and,
when that overflows, either scrolls (hiding tabs behind arrows) or
elides the labels to nothing; ``setUsesScrollButtons(False)`` only
swaps one for the other. Subclassing ``QTabBar`` does not help either:
``tabRect``/``tabAt`` are not virtual, so painting tabs at wrapped
positions would leave hit-testing still laid out in one row and clicks
would select the wrong tab.

So the strip here is a row of checkable buttons in a flow layout over
a ``QStackedWidget``, exposing the slice of ``QTabWidget``'s API this
project actually uses. Real user request: "maybe the subtabs for more
dense tabs like analysis need to go over multiple lines so they are
always visible when the window is smaller" -- worst case is the
Analysis sub-tabs (5) inside an already-narrow side panel, but the
canvas strip has 6, so this is applied to all of them rather than
special-cased.
"""

from __future__ import annotations

from PySide6.QtCore import QPoint, QRect, QSize, Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup, QLayout, QSizePolicy, QStackedWidget, QToolButton,
    QVBoxLayout, QWidget,
)


class FlowLayout(QLayout):
    """Lays items left to right, wrapping to a new row when the next
    item would not fit -- Qt's own documented flow-layout pattern.

    ``heightForWidth`` is what makes the wrapping real: the layout
    reports the height it needs *for a given width*, so a parent
    shrinking horizontally is told to give it more vertical space
    rather than clipping the overflow."""

    def __init__(self, parent=None, margin: int = 0, spacing: int = 4) -> None:
        super().__init__(parent)
        self._items: list = []
        self.setContentsMargins(margin, margin, margin, margin)
        self.setSpacing(spacing)

    def addItem(self, item) -> None:  # noqa: N802 -- Qt's own casing
        self._items.append(item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index):  # noqa: N802
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index):  # noqa: N802
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self):  # noqa: N802
        return Qt.Orientation(0)

    def hasHeightForWidth(self) -> bool:  # noqa: N802
        return True

    def heightForWidth(self, width: int) -> int:  # noqa: N802
        return self._layout(QRect(0, 0, width, 0), apply=False)

    def setGeometry(self, rect) -> None:  # noqa: N802
        super().setGeometry(rect)
        self._layout(rect, apply=True)

    def sizeHint(self) -> QSize:  # noqa: N802
        return self.minimumSize()

    def minimumSize(self) -> QSize:  # noqa: N802
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        margins = self.contentsMargins()
        return size + QSize(margins.left() + margins.right(),
                            margins.top() + margins.bottom())

    def _layout(self, rect, apply: bool) -> int:
        """Place (or just measure) the items inside ``rect``. Returns
        the total height used."""
        margins = self.contentsMargins()
        x = rect.x() + margins.left()
        y = rect.y() + margins.top()
        right = rect.right() - margins.right()
        row_height = 0
        spacing = self.spacing()
        for item in self._items:
            hint = item.sizeHint()
            if row_height and x + hint.width() > right:
                x = rect.x() + margins.left()
                y += row_height + spacing
                row_height = 0
            if apply:
                item.setGeometry(QRect(QPoint(x, y), hint))
            x += hint.width() + spacing
            row_height = max(row_height, hint.height())
        return y + row_height + margins.bottom() - rect.y()


class WrappingTabWidget(QWidget):
    """``QTabWidget``'s API, with a tab strip that wraps.

    Only the parts this project uses are implemented -- see the module
    docstring. ``currentChanged`` carries the new index, matching
    ``QTabWidget``'s own signal, so existing connections are unchanged.
    """

    currentChanged = Signal(int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._bar = QWidget(self)
        self._bar_layout = FlowLayout(self._bar)
        self._bar.setSizePolicy(QSizePolicy.Policy.Preferred,
                                 QSizePolicy.Policy.Minimum)
        self._buttons: list[QToolButton] = []
        self._labels: list[str] = []
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self._stack = QStackedWidget(self)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(self._bar)
        layout.addWidget(self._stack, 1)

        self._stack.currentChanged.connect(self._on_stack_changed)

    # -- QTabWidget API ------------------------------------------------

    def addTab(self, widget: QWidget, label: str) -> int:  # noqa: N802
        index = self._stack.addWidget(widget)
        button = QToolButton(self._bar)
        button.setText(label)
        button.setCheckable(True)
        button.setToolTip(label)
        button.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        button.clicked.connect(lambda _checked, w=widget: self.setCurrentWidget(w))
        self._group.addButton(button)
        self._bar_layout.addWidget(button)
        self._buttons.append(button)
        self._labels.append(label)
        if self._stack.count() == 1:
            button.setChecked(True)
        return index

    def count(self) -> int:
        return self._stack.count()

    def widget(self, index: int) -> QWidget | None:
        return self._stack.widget(index)

    def indexOf(self, widget: QWidget) -> int:  # noqa: N802
        return self._stack.indexOf(widget)

    def tabText(self, index: int) -> str:  # noqa: N802
        return self._labels[index] if 0 <= index < len(self._labels) else ""

    def currentIndex(self) -> int:  # noqa: N802
        return self._stack.currentIndex()

    def setCurrentIndex(self, index: int) -> None:  # noqa: N802
        self._stack.setCurrentIndex(index)

    def currentWidget(self) -> QWidget | None:  # noqa: N802
        return self._stack.currentWidget()

    def setCurrentWidget(self, widget: QWidget) -> None:  # noqa: N802
        self._stack.setCurrentWidget(widget)

    def setTabVisible(self, index: int, visible: bool) -> None:  # noqa: N802
        """Hide a tab's button, matching ``QTabWidget.setTabVisible``.

        The page stays in the stack -- hiding a tab must not renumber
        the others, since callers hold indices from ``indexOf``."""
        if 0 <= index < len(self._buttons):
            self._buttons[index].setVisible(visible)

    def isTabVisible(self, index: int) -> bool:  # noqa: N802
        """Whether this tab is *meant* to be shown.

        ``not isHidden()``, not ``isVisible()``: the latter is False
        for every child of a window that has not been shown yet, so it
        would report a perfectly visible tab as hidden simply because
        the widget tree is not on screen. ``QTabWidget.isTabVisible``
        reports the requested state, and so does this."""
        if 0 <= index < len(self._buttons):
            return not self._buttons[index].isHidden()
        return False

    # -- internals -----------------------------------------------------

    def _on_stack_changed(self, index: int) -> None:
        """Keep the buttons in step with the stack however the page was
        changed -- a click, ``setCurrentWidget``, or a caller setting
        the index directly."""
        if 0 <= index < len(self._buttons):
            self._buttons[index].setChecked(True)
        self.currentChanged.emit(index)
