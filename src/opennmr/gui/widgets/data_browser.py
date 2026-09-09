"""
A collapsible browser for experiments on disk.

MNova/TopSpin-style: pick a folder once, then click between the
experiments under it instead of going back through Open… every time.

Two design decisions worth stating, both from the request that prompted
it -- "make sure that the bruker experiment folders are the appropriate
destinations because people generally won't know to click the fid/ser":

* **Experiments are the leaves, not files.** A Bruker experiment is one
  node; its ``fid``/``ser``/``pdata`` never appear, because none of
  them is a thing a person should have to know to click. The same holds
  for X-Pulse, where the leaf is the ``.jdx`` that is actually loadable
  and the per-scan ``..._raw.jdx`` siblings are not offered.
* **Discovery is not reimplemented here.** The tree is built from
  :meth:`MainWindow._scan_for_spectra`, the same scan the Open… dialog
  and drag-and-drop already use -- which is what guarantees the browser
  cannot offer something Open would refuse, or hide something Open
  would find. It already knows to stop at a recognised experiment
  rather than descending into it, and to skip redundant per-scan and
  per-slice files.

What each experiment *is* comes from :func:`opennmr.io.peek`, which
reads only the small parameter files -- about a millisecond, against
seconds for a real load -- so the tree can say "1H, 2D, 1024x4096"
without opening anything large. Peeking is deferred until a folder is
expanded, so pointing this at a big archive costs nothing up front.
"""

from __future__ import annotations

import os

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView, QHBoxLayout, QLabel, QPushButton, QTreeWidget,
    QTreeWidgetItem, QVBoxLayout, QWidget,
)

from ... import io

#: Item data roles: the loadable path and its vendor, set only on leaves.
_PATH_ROLE = Qt.ItemDataRole.UserRole
_VENDOR_ROLE = Qt.ItemDataRole.UserRole + 1
#: Set on a folder once its children have been described, so expanding
#: it a second time doesn't re-peek.
_DESCRIBED_ROLE = Qt.ItemDataRole.UserRole + 2


class DataBrowser(QWidget):
    """A tree of loadable experiments under a chosen folder."""

    #: ``(path, vendor)`` for an experiment the user chose to open.
    spectrumActivated = Signal(str, str)
    #: Emitted when the user asks for a different root folder.
    browseRequested = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._root: str | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        button_row = QHBoxLayout()
        self._choose_button = QPushButton("Choose Folder…")
        self._choose_button.setToolTip(
            "Pick a folder to browse. Its experiments stay listed here so you "
            "can click between them without reopening the dialog."
        )
        self._choose_button.clicked.connect(self.browseRequested)
        button_row.addWidget(self._choose_button)
        button_row.addStretch(1)
        layout.addLayout(button_row)

        self._root_label = QLabel("No folder chosen.")
        self._root_label.setWordWrap(True)
        layout.addWidget(self._root_label)

        self._tree = QTreeWidget()
        self._tree.setHeaderHidden(True)
        self._tree.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._tree.setToolTip(
            "Double-click an experiment to load it. Folders group experiments; "
            "only things this app can actually open are listed."
        )
        self._tree.itemActivated.connect(self._on_item_activated)
        self._tree.itemDoubleClicked.connect(self._on_item_activated)
        self._tree.itemExpanded.connect(self._describe_children)
        layout.addWidget(self._tree, 1)

        self._status = QLabel("")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

    # -- population ----------------------------------------------------

    def set_matches(self, root: str, matches: list[tuple[str, str]]) -> None:
        """Show ``matches`` -- ``(path, vendor)`` pairs, as returned by
        the same scan Open… uses -- grouped by their folders under
        ``root``."""
        self._root = root
        self._tree.clear()
        self._root_label.setText(root)

        if not matches:
            self._status.setText("No loadable experiments found in this folder.")
            return
        self._status.setText(
            f"{len(matches)} experiment{'s' if len(matches) != 1 else ''}. "
            f"Double-click to load."
        )

        folders: dict[str, QTreeWidgetItem] = {}

        def folder_item(relative: str) -> QTreeWidgetItem | None:
            """The node for a folder, creating its ancestors as needed.
            ``None`` means the tree's own invisible root."""
            if not relative or relative == ".":
                return None
            if relative in folders:
                return folders[relative]
            parent = folder_item(os.path.dirname(relative))
            item = QTreeWidgetItem([os.path.basename(relative)])
            if parent is None:
                self._tree.addTopLevelItem(item)
            else:
                parent.addChild(item)
            folders[relative] = item
            return item

        for path, vendor in matches:
            relative = os.path.relpath(path, root)
            parent = folder_item(os.path.dirname(relative))
            leaf = QTreeWidgetItem([os.path.basename(path)])
            leaf.setData(0, _PATH_ROLE, path)
            leaf.setData(0, _VENDOR_ROLE, vendor)
            leaf.setToolTip(0, path)
            if parent is None:
                self._tree.addTopLevelItem(leaf)
            else:
                parent.addChild(leaf)

        # Describe whatever is visible without expanding anything --
        # top-level leaves are on screen immediately.
        self._describe_children(self._tree.invisibleRootItem())

    def _describe_children(self, parent: QTreeWidgetItem) -> None:
        """Fill in "1H, 2D, 1024x4096" for a folder's own leaves.

        Deferred to expansion, and done once: :func:`opennmr.io.peek` is
        cheap (~1 ms, parameter files only) but a large archive holds
        hundreds of experiments, and there is no reason to describe a
        folder nobody has opened."""
        if parent is None:
            return
        if parent is not self._tree.invisibleRootItem():
            if parent.data(0, _DESCRIBED_ROLE):
                return
            parent.setData(0, _DESCRIBED_ROLE, True)

        for index in range(parent.childCount()):
            child = parent.child(index)
            path = child.data(0, _PATH_ROLE)
            if not path:
                continue
            summary = self._describe(path, child.data(0, _VENDOR_ROLE))
            if summary:
                child.setText(0, f"{os.path.basename(path)}   {summary}")

    @staticmethod
    def _describe(path: str, vendor: str) -> str:
        """A one-line "what is this" for an experiment, or ``""`` if it
        cannot be determined cheaply.

        Never raises: a folder that cannot be described is still
        loadable, and a browser that refused to list it would be worse
        than one that lists it without a summary."""
        try:
            info = io.peek(path, vendor=vendor)
        except Exception:  # noqa: BLE001 -- an undescribable experiment still lists
            return ""
        bits = []
        if info.get("nucleus"):
            bits.append(str(info["nucleus"]))
        ndim = info.get("ndim")
        if ndim:
            bits.append("pseudo-2D" if info.get("pseudo_2d") else f"{ndim}D")
        shape = info.get("shape")
        if shape:
            bits.append("x".join(str(int(n)) for n in shape))
        return "· " + ", ".join(bits) if bits else ""

    # -- interaction ---------------------------------------------------

    def _on_item_activated(self, item: QTreeWidgetItem, _column: int = 0) -> None:
        path = item.data(0, _PATH_ROLE)
        if path:
            self.spectrumActivated.emit(path, item.data(0, _VENDOR_ROLE))
        else:
            item.setExpanded(not item.isExpanded())

    def selected_path(self) -> tuple[str, str] | None:
        """The selected experiment, or ``None`` if a folder (or
        nothing) is selected."""
        item = self._tree.currentItem()
        if item is None:
            return None
        path = item.data(0, _PATH_ROLE)
        return (path, item.data(0, _VENDOR_ROLE)) if path else None
