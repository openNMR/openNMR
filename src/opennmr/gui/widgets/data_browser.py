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

**That was only ever true of Bruker.** ``io.peek`` falls back to a
full ``load()`` for any vendor that has not registered its own, and
until every vendor did, expanding a folder here read *and processed*
every experiment in it -- measured at 1.8 s per X-Pulse arrayed
experiment and 0.7 s per JEOL file, so a five-experiment PFG folder
blocked the GUI for about nine seconds to print five labels. All four
vendors now have a real ``peek``, and each is pinned by a test
asserting it reports the same shape a real load produces; describing
the whole of ``test_data`` (238 experiments, all four vendors) takes
about 250 ms.

**Several experiments can be opened at once.** The tree uses extended
selection: Ctrl-click (Cmd-click on macOS) adds or removes one
experiment, Shift-click takes the whole run between two. "Open
Selected" (or Return) then opens every selected experiment, top to
bottom in tree order -- not click order, so a Shift range of 10..15
lands in the list as 10, 11, ... 15. Folders in a selection are
skipped rather than opened wholesale: a Shift range that happens to
cross a folder heading should not quietly load everything under it.
A plain double-click still opens just the experiment under the cursor,
because its first click has already reduced the selection to that one.
Explicit user request.
"""

from __future__ import annotations

import os

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView, QHBoxLayout, QLabel, QPushButton, QTreeWidget,
    QTreeWidgetItem, QTreeWidgetItemIterator, QVBoxLayout, QWidget,
)

from ... import io
from ...io import cloud

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
        self._open_selected_button = QPushButton("Open Selected")
        self._open_selected_button.setToolTip(
            "Open every selected experiment. Ctrl-click (Cmd-click on macOS) "
            "to pick separate experiments, Shift-click to pick a run of them."
        )
        self._open_selected_button.setEnabled(False)
        self._open_selected_button.clicked.connect(self._open_selected)
        button_row.addWidget(self._open_selected_button)
        button_row.addStretch(1)
        layout.addLayout(button_row)

        self._root_label = QLabel("No folder chosen.")
        self._root_label.setWordWrap(True)
        layout.addWidget(self._root_label)

        # **No per-selection details panel here**, deliberately. One
        # briefly sat between this label and the tree, restating the
        # selected experiment's nucleus, size and pulse programme. But
        # the Data tab's own "Selected spectrum" box already says all
        # of that (see MainWindow._refresh_selection_display), and each
        # tree row already carries its own one-line summary from
        # `_describe` -- so the panel made it the third statement of
        # the same facts on one tab. User report: it was "double
        # noting" what the selected-spectrum panel notes already.
        self._tree = QTreeWidget()
        self._tree.setHeaderHidden(True)
        self._tree.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._tree.setToolTip(
            "Double-click an experiment to load it. Ctrl-click (Cmd-click on "
            "macOS) or Shift-click to select several, then Open Selected or "
            "press Return. Folders group experiments; only things this app "
            "can actually open are listed."
        )
        self._tree.itemSelectionChanged.connect(self._update_open_selected_button)
        # `itemActivated` only -- **not** also `itemDoubleClicked`.
        # A double-click emits both, so connecting both ran this
        # handler twice for one gesture and loaded the same experiment
        # twice: two identical entries in the list, and for a large 2D,
        # the whole multi-second read done twice over. Real user
        # report: "the folder browser shows each data set twice, I hope
        # this isn't a double load."
        #
        # `itemActivated` is also the correct single signal to keep: it
        # is the platform's own notion of activation, so it fires on
        # double-click *and* on Return, and follows the setting on
        # desktops configured for single-click activation --
        # `itemDoubleClicked` does none of that.
        self._tree.itemActivated.connect(self._on_item_activated)
        self._tree.itemExpanded.connect(self._describe_children)
        layout.addWidget(self._tree, 1)

        self._status = QLabel("")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

    # -- population ----------------------------------------------------

    def set_matches(self, root: str, matches: list[tuple[str, str]],
                    truncated_at: int | None = None) -> None:
        """Show ``matches`` -- ``(path, vendor)`` pairs, as returned by
        the same scan Open… uses -- grouped by their folders under
        ``root``. ``truncated_at``: the scan stopped at this many, so
        the folder holds more than are listed -- said in the status line."""
        self._root = root
        self._tree.clear()
        self._root_label.setText(root)
        self._update_open_selected_button()

        if not matches:
            self._status.setText("No loadable experiments found in this folder.")
            return
        self._status.setText(
            f"{len(matches):,} experiment{'s' if len(matches) != 1 else ''}. "
            f"Double-click to load, or Ctrl/Shift-click to select several "
            f"and Open Selected."
            + (f" The scan stops at {truncated_at:,}: this folder holds more -- browse a "
               f"subfolder to see them." if truncated_at else "")
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
            # Say up front what opening this will cost, rather than
            # letting the first click block on a silent download.
            pending = cloud.pending_download_bytes(path)
            if pending:
                leaf.setToolTip(
                    0, f"{path}\n\nNot downloaded yet -- opening this will "
                       f"fetch {cloud.format_bytes(pending)} from the cloud.",
                )
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
        folder nobody has opened.

        Still synchronous, and that is now fine: at ~1 ms per
        experiment a folder would have to hold thousands before this
        is noticeable. It was *not* fine while three of the four
        vendors fell back to a full load -- see this module's own
        docstring."""
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
        # Cloud-backed data is described from its name alone. Peeking
        # reads the parameter files, and on a OneDrive/iCloud folder
        # that *downloads* them -- so describing a folder of a hundred
        # experiments would silently become a hundred sequential
        # downloads on expand. See opennmr.io.cloud: the stat this uses
        # is free and never materialises anything.
        if not cloud.parameters_are_local(path):
            pending = cloud.pending_download_bytes(path)
            size = f", {cloud.format_bytes(pending)}" if pending else ""
            return f"· in cloud{size} -- double-click to download"
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
        # **Last, and it is the most useful thing here.** "PGSTE" or
        # "hsqcetgp" identifies an experiment in a way no shape can --
        # a folder of 1H 1D 65536s is indistinguishable without it.
        # All four vendors record one and all four peeks now report it
        # (Bruker PULPROG, Varian seqfil, JEOL `experiment`, X-Pulse
        # .PULSESEQUENCE), so this costs nothing beyond the peek that
        # already ran. Explicit user request.
        pulse_program = info.get("pulse_program")
        if pulse_program:
            bits.append(str(pulse_program))
        return "· " + ", ".join(bits) if bits else ""

    # -- interaction ---------------------------------------------------

    def _on_item_activated(self, item: QTreeWidgetItem, _column: int = 0) -> None:
        path = item.data(0, _PATH_ROLE)
        if not path:
            item.setExpanded(not item.isExpanded())
            return
        # Return on a multi-selection opens all of it -- the activated
        # item is just the one holding focus. A double-click never gets
        # here with several selected: its first (unmodified) click has
        # already cut the selection down to the item under the cursor.
        selected = self.selected_paths()
        if item.isSelected() and len(selected) > 1:
            self._emit_all(selected)
        else:
            self.spectrumActivated.emit(path, item.data(0, _VENDOR_ROLE))

    def selected_paths(self) -> list[tuple[str, str]]:
        """Every selected experiment as ``(path, vendor)``, in tree
        order (top to bottom), folders excluded. Tree order rather than
        :meth:`QTreeWidget.selectedItems`, whose order is not defined --
        a Shift range should open in the order it is listed."""
        found = []
        iterator = QTreeWidgetItemIterator(
            self._tree, QTreeWidgetItemIterator.IteratorFlag.Selected)
        while iterator.value() is not None:
            item = iterator.value()
            path = item.data(0, _PATH_ROLE)
            if path:
                found.append((path, item.data(0, _VENDOR_ROLE)))
            iterator += 1
        return found

    def _open_selected(self) -> None:
        self._emit_all(self.selected_paths())

    def _emit_all(self, pairs: list[tuple[str, str]]) -> None:
        # One emission per experiment, through the same signal a
        # double-click uses, so each goes through MainWindow._open_one
        # and its already-open / already-loading check individually.
        for path, vendor in pairs:
            self.spectrumActivated.emit(path, vendor)

    def _update_open_selected_button(self) -> None:
        count = len(self.selected_paths())
        self._open_selected_button.setEnabled(count > 0)
        self._open_selected_button.setText(
            f"Open {count} Selected" if count > 1 else "Open Selected")

    def selected_path(self) -> tuple[str, str] | None:
        """The selected experiment, or ``None`` if a folder (or
        nothing) is selected."""
        item = self._tree.currentItem()
        if item is None:
            return None
        path = item.data(0, _PATH_ROLE)
        return (path, item.data(0, _VENDOR_ROLE)) if path else None
