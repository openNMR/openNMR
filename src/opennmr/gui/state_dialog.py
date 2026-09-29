"""File ▸ Save openNMR State: choose which open spectra to save together,
under what name, and which slices of any 2D to save as spectra of their
own. The saving itself is :func:`opennmr.io.state.save_state` (via
``MainWindow._save_state``)."""

from __future__ import annotations

import os

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout, QHBoxLayout, QHeaderView,
    QLabel, QLineEdit, QMessageBox, QPlainTextEdit, QPushButton, QTreeWidget, QTreeWidgetItem,
    QVBoxLayout, QWidget,
)

from ..io import state as io_state


def parse_rows(text: str, n_rows: int) -> list[int]:
    """``"1, 4-6"`` -> ``[1, 4, 5, 6]`` (1-based), checked against
    ``n_rows``. Empty text -> []. Raises ValueError with a readable
    message."""
    rows = []
    for part in (p.strip() for p in text.replace(";", ",").split(",")):
        if not part:
            continue
        if "-" in part[1:]:
            lo, hi = part.split("-", 1) if not part.startswith("-") else ("", part)
            try:
                lo, hi = int(lo), int(hi)
            except ValueError as exc:
                raise ValueError(f"{part!r} is not a slice range like 4-6.") from exc
            if lo > hi:
                lo, hi = hi, lo
            rows += list(range(lo, hi + 1))
        else:
            try:
                rows.append(int(part))
            except ValueError as exc:
                raise ValueError(f"{part!r} is not a slice number.") from exc
    bad = [r for r in rows if not 1 <= r <= n_rows]
    if bad:
        raise ValueError(f"Slice {bad[0]} is outside 1-{n_rows}.")
    return sorted(set(rows))


class SaveStateDialog(QDialog):
    """``items``: ``[{"index", "label", "kind", "detail", "rows"}]``
    (``rows``: slice count of a 2D, else 0); ``checked``: indices ticked
    at the start."""

    def __init__(self, parent, items, checked=(), default_name="") -> None:
        super().__init__(parent)
        self.setWindowTitle("Save openNMR State")
        self.resize(760, 560)
        self._items = list(items)
        self._folder_chosen = False
        v = QVBoxLayout(self)
        intro = QLabel(
            "Save a group of open spectra together -- with their processing, fits and kept fit "
            "models -- under one name, to reopen later with File ▸ Open openNMR State. "
            "Experiments are reopened from their own data and reprocessed exactly as now; "
            "anything without raw data of its own (slices, simulations, VOCS, arithmetic) is "
            "saved as its data.")
        intro.setWordWrap(True)
        v.addWidget(intro)
        form = QFormLayout()
        self.name_edit = QLineEdit(default_name)
        self.name_edit.textChanged.connect(self._name_changed)
        form.addRow("Name:", self.name_edit)
        folder_row = QWidget()
        h = QHBoxLayout(folder_row)
        h.setContentsMargins(0, 0, 0, 0)
        self.folder_edit = QLineEdit(io_state.default_state_folder(default_name or "state"))
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        h.addWidget(self.folder_edit, 1)
        h.addWidget(browse)
        form.addRow("Save in:", folder_row)
        self.notes_edit = QPlainTextEdit()
        self.notes_edit.setPlaceholderText("Optional: what this state is, for whoever opens it.")
        self.notes_edit.setMaximumHeight(60)
        form.addRow("Notes:", self.notes_edit)
        v.addLayout(form)

        self.tree = QTreeWidget()
        self.tree.setColumnCount(4)
        self.tree.setHeaderLabels(["Spectrum", "Type", "Contains", "Also save slices (e.g. 1, 4-6)"])
        header = self.tree.header()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for c in (1, 2):
            header.setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
        self.tree.setRootIsDecorated(False)
        self.slice_edits = {}
        for item in self._items:
            row = QTreeWidgetItem([item["label"], item["kind"], item["detail"], ""])
            row.setFlags(row.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            row.setCheckState(0, Qt.CheckState.Checked if item["index"] in set(checked)
                              else Qt.CheckState.Unchecked)
            row.setData(0, Qt.ItemDataRole.UserRole, item["index"])
            row.setToolTip(0, item["label"])
            self.tree.addTopLevelItem(row)
            if item.get("rows"):
                edit = QLineEdit()
                edit.setPlaceholderText(f"1-{item['rows']}")
                edit.setToolTip(
                    "Slices of this 2D to save as 1D spectra of their own, as they are processed "
                    "now (phase, baseline...). Saved even if the 2D itself is not ticked.")
                self.tree.setItemWidget(row, 3, edit)
                self.slice_edits[item["index"]] = (edit, item["rows"])
        v.addWidget(self.tree, 1)
        buttons_row = QHBoxLayout()
        for text, state in (("Tick All", Qt.CheckState.Checked), ("Untick All", Qt.CheckState.Unchecked)):
            b = QPushButton(text)
            b.clicked.connect(lambda _c=False, s=state: self._tick_all(s))
            buttons_row.addWidget(b)
        buttons_row.addStretch(1)
        v.addLayout(buttons_row)
        self.copy_2d_check = QCheckBox("Also keep a copy of each 2D experiment's processed data "
                                       "(can be large; used if its raw data move)")
        self.copy_2d_check.setToolTip(
            "1D experiments always get a copy, so the state still opens if the raw data are moved "
            "or deleted. A 2D is reopened from its own folder unless this is ticked.")
        v.addWidget(self.copy_2d_check)
        box = QDialogButtonBox(QDialogButtonBox.StandardButton.Save
                               | QDialogButtonBox.StandardButton.Cancel)
        box.accepted.connect(self._accept)
        box.rejected.connect(self.reject)
        v.addWidget(box)

    def _name_changed(self, text) -> None:
        if not self._folder_chosen:
            self.folder_edit.setText(io_state.default_state_folder(text.strip() or "state"))

    def _browse(self) -> None:
        parent = QFileDialog.getExistingDirectory(
            self, "Folder to save the state in", os.path.dirname(self.folder_edit.text()),
            options=QFileDialog.Option.DontUseNativeDialog)
        if parent:
            name = os.path.basename(self.folder_edit.text()) or "state" + io_state.STATE_SUFFIX
            self.folder_edit.setText(os.path.join(parent, name))
            self._folder_chosen = True

    def _tick_all(self, state) -> None:
        for i in range(self.tree.topLevelItemCount()):
            self.tree.topLevelItem(i).setCheckState(0, state)

    def values(self):
        """``(name, folder, notes, indices, slices, copy_2d)``; raises
        ValueError for a bad slice list or nothing to save."""
        indices = []
        for i in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(i)
            if item.checkState(0) == Qt.CheckState.Checked:
                indices.append(item.data(0, Qt.ItemDataRole.UserRole))
        slices = {}
        for index, (edit, n_rows) in self.slice_edits.items():
            rows = parse_rows(edit.text(), n_rows)
            if rows:
                slices[index] = rows
        if not indices and not slices:
            raise ValueError("Tick at least one spectrum (or give slices of a 2D) to save.")
        name = self.name_edit.text().strip() or "openNMR state"
        return (name, self.folder_edit.text().strip(), self.notes_edit.toPlainText().strip(),
                indices, slices, self.copy_2d_check.isChecked())

    def _accept(self) -> None:
        try:
            self.values()
        except ValueError as exc:
            QMessageBox.warning(self, "Save openNMR State", str(exc))
            return
        if not self.folder_edit.text().strip():
            QMessageBox.warning(self, "Save openNMR State", "Choose where to save it.")
            return
        self.accept()
