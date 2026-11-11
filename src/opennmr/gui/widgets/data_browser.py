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

**The list can be sorted** (right-click ▸ Sort by; explicit user request,
2026-11-11): by name, A to Z (the default, and the order the scan
already produces) or Z to A, or by date, newest or oldest first. Names
sort *naturally* -- "2" before "10", as Bruker's numbered experiment
folders expect. An experiment's date is when its data were written:
the mtime of its ``fid``/``ser`` (or ``acqus``) for a Bruker-style
folder, of the file itself otherwise -- not of the folder, whose mtime
moves whenever anything is reprocessed into it. A folder sorts by the
newest experiment in it for "newest first" and by the oldest for
"oldest first", so the folder holding the experiment you are after is
where you would look for it. Sorting reorders each level in place:
folders and experiments stay where they are in the hierarchy, and what
was expanded or selected stays so. Reading the dates is a stat per
experiment, which never downloads a cloud placeholder.
"""

from __future__ import annotations

import os
import re

from PySide6.QtCore import QItemSelectionModel, Qt, Signal
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
#: An experiment's date (seconds since the epoch), read once when first needed.
_DATE_ROLE = Qt.ItemDataRole.UserRole + 3

#: The orders the tree can be sorted in: ``{key: menu text}``.
SORT_MODES = {
    "name_asc": "Name (A to Z)",
    "name_desc": "Name (Z to A)",
    "date_new": "Date (newest first)",
    "date_old": "Date (oldest first)",
}
DEFAULT_SORT = "name_asc"

_NATURAL_SPLIT = re.compile(r"(\d+)")
#: Files whose mtime is an experiment's acquisition date, best first.
_DATE_FILES = ("fid", "ser", "acqus", "acqu", "data.1d", "data.2d", "acqu.par", "procpar")


def natural_key(name: str) -> list:
    """"2" before "10": digit runs compare as numbers, the rest
    case-insensitively. Each chunk is tagged so a number never meets a
    string in a comparison."""
    return [(0, int(chunk), "") if chunk.isdigit() else (1, 0, chunk.lower())
            for chunk in _NATURAL_SPLIT.split(str(name)) if chunk != ""]


def experiment_date(path: str) -> float:
    """When an experiment's data were written (see the module notes);
    0.0 if it cannot be read, so an unreadable one sorts as oldest."""
    try:
        if os.path.isdir(path):
            for name in _DATE_FILES:
                candidate = os.path.join(path, name)
                if os.path.isfile(candidate):
                    return os.stat(candidate).st_mtime
        return os.stat(path).st_mtime
    except OSError:
        return 0.0


class DataBrowser(QWidget):
    """A tree of loadable experiments under a chosen folder."""

    #: ``(path, vendor)`` for an experiment the user chose to open.
    spectrumActivated = Signal(str, str)
    #: Emitted when the user asks for a different root folder.
    browseRequested = Signal()
    #: ``(path, vendor)``: right-click ▸ Proc Numbers… on a Bruker
    #: experiment (explicit user request, 2026-10-14).
    procnosRequested = Signal(str, str)
    #: The sort order chosen from the right-click menu (a SORT_MODES key),
    #: so the window can remember it.
    sortModeChanged = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._root: str | None = None
        self._sort_mode = DEFAULT_SORT

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
            "press Return. Right-click to sort by name or date. Folders group "
            "experiments; only things this app can actually open are listed."
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
        self._tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._tree.customContextMenuRequested.connect(self._on_context_menu)
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
            + (f" The scan stops at {truncated_at:,}: this folder holds more – browse a "
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
                    0, f"{path}\n\nNot downloaded yet – opening this will "
                       f"fetch {cloud.format_bytes(pending)} from the cloud.",
                )
            if parent is None:
                self._tree.addTopLevelItem(leaf)
            else:
                parent.addChild(leaf)

        # The scan already lists by name, A to Z; any other order is
        # applied here, before anything is described or shown.
        if self._sort_mode != DEFAULT_SORT:
            self._sort_tree()

        # Describe whatever is visible without expanding anything --
        # top-level leaves are on screen immediately.
        self._describe_children(self._tree.invisibleRootItem())

    # -- sorting -------------------------------------------------------

    def sort_mode(self) -> str:
        return self._sort_mode

    def set_sort_mode(self, mode: str, emit: bool = False) -> None:
        """Sort the tree by ``mode`` (a :data:`SORT_MODES` key; anything
        else means the default). ``emit``: say so through
        :attr:`sortModeChanged` (the menu does; restoring a saved choice
        does not)."""
        mode = mode if mode in SORT_MODES else DEFAULT_SORT
        self._sort_mode = mode
        self._sort_tree()
        if emit:
            self.sortModeChanged.emit(mode)

    def _item_date(self, item: QTreeWidgetItem, newest: bool) -> float:
        path = item.data(0, _PATH_ROLE)
        if path:
            date = item.data(0, _DATE_ROLE)
            if date is None:
                date = experiment_date(path)
                item.setData(0, _DATE_ROLE, date)
            return float(date)
        dates = [self._item_date(item.child(i), newest) for i in range(item.childCount())]
        if not dates:
            return 0.0
        return max(dates) if newest else min(dates)

    def _item_name(self, item: QTreeWidgetItem) -> str:
        path = item.data(0, _PATH_ROLE)
        # The leaf's text gains a "· 1H, 1D, ..." summary once described;
        # its name is the path's own.
        return os.path.basename(path) if path else item.text(0)

    def _sort_tree(self) -> None:
        mode = self._sort_mode
        tree = self._tree
        current = tree.currentItem()
        selected = list(tree.selectedItems())
        expanded = []
        iterator = QTreeWidgetItemIterator(tree)
        while iterator.value() is not None:
            if iterator.value().isExpanded():
                expanded.append(iterator.value())
            iterator += 1

        if mode.startswith("date"):
            newest = mode == "date_new"

            def key(item):
                # Ties (equal dates, unreadable ones) fall back to the name.
                return (self._item_date(item, newest), natural_key(self._item_name(item)))
            reverse = newest
        else:
            def key(item):
                return natural_key(self._item_name(item))
            reverse = mode == "name_desc"

        def sort_children(parent: QTreeWidgetItem) -> None:
            children = parent.takeChildren()
            children.sort(key=key, reverse=reverse)
            parent.addChildren(children)
            for child in children:
                if child.childCount():
                    sort_children(child)

        tree.setUpdatesEnabled(False)
        try:
            sort_children(tree.invisibleRootItem())
            for item in expanded:
                item.setExpanded(True)
            for item in selected:
                item.setSelected(True)
            if current is not None:
                tree.setCurrentItem(current, 0,
                                    QItemSelectionModel.SelectionFlag.NoUpdate)
                tree.scrollToItem(current)
        finally:
            tree.setUpdatesEnabled(True)
        self._update_open_selected_button()

    def ordered_names(self, parent: QTreeWidgetItem | None = None) -> list[str]:
        """The names at one level of the tree, top to bottom (the top
        level by default) -- what a sort is checked against."""
        parent = parent or self._tree.invisibleRootItem()
        return [self._item_name(parent.child(i)) for i in range(parent.childCount())]

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
            return f"· in cloud{size} – double-click to download"
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

    def context_actions(self, item) -> list[tuple[str, object]]:
        """What right-clicking ``item`` offers: ``[(text, callback)]``."""
        path = item.data(0, _PATH_ROLE) if item is not None else None
        if not path:
            return []
        vendor = item.data(0, _VENDOR_ROLE)
        actions = [("Open", lambda: self.spectrumActivated.emit(path, vendor))]
        if vendor == "bruker":
            actions.append(("Proc Numbers…", lambda: self.procnosRequested.emit(path, vendor)))
        return actions

    def _on_context_menu(self, pos) -> None:
        """Right-click: the item's own actions (Open, Proc Numbers…),
        then Sort by -- offered anywhere in the tree, on folders and
        empty space too."""
        from PySide6.QtGui import QActionGroup
        from PySide6.QtWidgets import QMenu
        actions = self.context_actions(self._tree.itemAt(pos))
        menu = QMenu(self)
        chosen = {menu.addAction(text): callback for text, callback in actions}
        if self._tree.topLevelItemCount():
            if actions:
                menu.addSeparator()
            sort_menu = menu.addMenu("Sort by")
            group = QActionGroup(sort_menu)
            for mode, text in SORT_MODES.items():
                action = sort_menu.addAction(text)
                action.setCheckable(True)
                action.setChecked(mode == self._sort_mode)
                group.addAction(action)
                chosen[action] = (lambda m=mode: self.set_sort_mode(m, emit=True))
        if not chosen:
            return
        picked = menu.exec(self._tree.viewport().mapToGlobal(pos))
        if picked in chosen:
            chosen[picked]()

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
