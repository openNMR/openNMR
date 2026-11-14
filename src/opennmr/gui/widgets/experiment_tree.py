"""The Open experiments tree: what is open, each entry with its type, the
fits kept with a spectrum beneath it and a multi-spectrum fit as a group
(explicit user request, 2026-11-13). The nodes are
:func:`opennmr.notebook.experiment_tree.build_experiment_tree`'s; this only
draws them and reports what the user does.

The session and the (now hidden) Open experiments list remain the model:
every node names the list row it stands for (``row``), selecting nodes
asks for those rows to be selected (``rowsChosen``), and the list's
selection is mirrored back with :meth:`select_rows`.

As the list did: a plain press on a row does not select it until the
button is released, so a row can be dragged (to Compare, onto a 2D's
projection) without replacing what is on show.
"""

from __future__ import annotations

from PySide6.QtCore import QItemSelectionModel, Qt, Signal
from PySide6.QtGui import QBrush, QFont
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QHeaderView, QTreeWidget, QTreeWidgetItem,
)

_NODE_ROLE = Qt.ItemDataRole.UserRole


class ExperimentTree(QTreeWidget):
    #: Rows the user selected (sorted, without repeats), and the current one.
    rowsChosen = Signal(list, int)
    #: A node opened (double-click or Return): the node dict.
    nodeActivated = Signal(object)
    #: Right-click: the node dict and the global position.
    menuRequested = Signal(object, object)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setColumnCount(2)
        self.setHeaderLabels(["Experiment", "Type"])
        self.header().setStretchLastSection(False)
        self.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.header().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setUniformRowHeights(True)
        self.setExpandsOnDoubleClick(False)       # double-click opens the entry
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._on_menu)
        self.itemSelectionChanged.connect(self._on_selection)
        self.itemActivated.connect(self._on_activated)
        self.itemDoubleClicked.connect(self._on_activated)
        self.itemCollapsed.connect(lambda item: self._collapsed.add(self.node(item)["key"]))
        self.itemExpanded.connect(lambda item: self._collapsed.discard(self.node(item)["key"]))
        self._collapsed: set[str] = set()
        self._syncing = False
        self._pending_press = None
        #: ``mime_for_rows(rows)``: the drag data for those rows (the
        #: list's own, so every drop target takes it as before).
        self.mime_for_rows = None
        self._last_activation = None

    # ---- contents -----------------------------------------------------

    @staticmethod
    def node(item) -> dict:
        return item.data(0, _NODE_ROLE)

    def set_nodes(self, nodes) -> None:
        """Show ``nodes`` (the selection is cleared: follow with
        :meth:`select_rows`)."""
        self._syncing = True
        try:
            self.clear()
            for node in nodes:
                self._add(self.invisibleRootItem(), node)
        finally:
            self._syncing = False

    def _add(self, parent, node) -> None:
        item = QTreeWidgetItem(parent, [node["label"], node["type_label"]])
        item.setData(0, _NODE_ROLE, node)
        tip = node["label"] + (f"\n{node['tooltip']}" if node["tooltip"] else "")
        if node["action"] is not None:
            tip += "\nDouble-click to show it."
        item.setToolTip(0, tip)
        item.setToolTip(1, node["type_label"])
        muted = QBrush(self.palette().color(self.palette().ColorGroup.Disabled,
                                            self.palette().ColorRole.Text))
        item.setForeground(1, muted)
        if node["kind"] == "multi_fit":
            font = QFont(item.font(0))
            font.setBold(True)
            item.setFont(0, font)
        elif node["virtual"]:
            font = QFont(item.font(0))
            font.setItalic(True)
            item.setFont(0, font)
        for child in node["children"]:
            self._add(item, child)
        item.setExpanded(node["key"] not in self._collapsed)

    def items(self):
        """Every item, in display order."""
        ordered = []

        def walk(parent):
            for i in range(parent.childCount()):
                ordered.append(parent.child(i))
                walk(parent.child(i))
        walk(self.invisibleRootItem())
        return ordered

    def item_for_row(self, row: int):
        """The item that *is* row ``row`` (not a fit stored with it)."""
        for item in self.items():
            node = self.node(item)
            if node["row"] == row and not node["virtual"]:
                return item
        return None

    def find(self, key: str):
        for item in self.items():
            if self.node(item)["key"] == key:
                return item
        return None

    # ---- selection ----------------------------------------------------

    def selected_rows(self) -> list[int]:
        return sorted({self.node(i)["row"] for i in self.selectedItems()
                       if self.node(i)["row"] is not None})

    def select_rows(self, rows, current: int | None = None) -> None:
        """Mirror the list's selection (no ``rowsChosen``). A fit stored
        with a row stays selected while its row is the only one chosen."""
        rows = sorted(set(int(r) for r in rows))
        if rows == self.selected_rows():
            return
        self._syncing = True
        try:
            self.clearSelection()
            first = None
            for row in rows:
                item = self.item_for_row(row)
                if item is None:
                    continue
                item.setSelected(True)
                parent = item.parent()
                while parent is not None:          # make it visible
                    parent.setExpanded(True)
                    parent = parent.parent()
                if first is None or row == current:
                    first = item
            if first is not None:
                self.setCurrentItem(first, 0, QItemSelectionModel.SelectionFlag.NoUpdate)
                self.scrollToItem(first)
        finally:
            self._syncing = False

    def _on_selection(self) -> None:
        if self._syncing:
            return
        rows = self.selected_rows()
        current = self.currentItem()
        row = self.node(current)["row"] if current is not None else None
        if row is None or row not in rows:
            row = rows[0] if rows else -1
        self.rowsChosen.emit(rows, int(row))

    def _on_activated(self, item, _column=0) -> None:
        if item is None:
            return
        # Double-click fires both itemDoubleClicked and (on some styles)
        # itemActivated: one opening per gesture.
        import time
        now = time.monotonic()
        last = self._last_activation
        if last is not None and last[0] is item and now - last[1] < 0.3:
            return
        self._last_activation = (item, now)
        self.nodeActivated.emit(self.node(item))

    def _on_menu(self, pos) -> None:
        item = self.itemAt(pos)
        if item is not None:
            self.menuRequested.emit(self.node(item), self.viewport().mapToGlobal(pos))

    # ---- a press that can become a drag -------------------------------

    def mousePressEvent(self, event) -> None:  # noqa: N802 -- Qt override
        self._pending_press = None
        if (event.button() == Qt.MouseButton.LeftButton
                and event.modifiers() == Qt.KeyboardModifier.NoModifier):
            item = self.itemAt(event.position().toPoint())
            # Not on the expand arrow, and only rows can be dragged.
            if item is not None and self.node(item)["row"] is not None \
                    and not self.node(item)["virtual"] \
                    and event.position().x() >= self.visualItemRect(item).left():
                self._pending_press = (event.clone(), self.node(item)["row"])
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 -- Qt override
        if self._pending_press is not None:
            press, row = self._pending_press
            if not (event.buttons() & Qt.MouseButton.LeftButton):
                self._pending_press = None
            elif ((event.position() - press.position()).manhattanLength()
                  >= QApplication.startDragDistance()):
                self._pending_press = None
                selected = self.selected_rows()
                self.start_drag(selected if row in selected else [row])
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 -- Qt override
        if self._pending_press is not None:
            press, _row = self._pending_press
            self._pending_press = None
            super().mousePressEvent(press)        # it was a click after all
        super().mouseReleaseEvent(event)

    def start_drag(self, rows) -> None:
        """Drag ``rows`` (a seam: a real ``QDrag`` blocks in its own event
        loop, which a headless test cannot finish)."""
        if self.mime_for_rows is None or not rows:
            return
        from PySide6.QtGui import QDrag
        drag = QDrag(self)
        drag.setMimeData(self.mime_for_rows(list(rows)))
        drag.exec(Qt.DropAction.CopyAction)
