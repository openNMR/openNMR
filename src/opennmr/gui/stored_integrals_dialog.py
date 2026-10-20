"""Integrate ▸ Integration Regions ▸ View Stored Integral Table: the table of
integrals kept with *Store integral in "Stored integrals"*
(:mod:`opennmr.notebook.stored_integrals`), with notes you can edit, rows
you can delete, and Save as CSV / JSON / text."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView, QDialog, QFileDialog, QHBoxLayout, QHeaderView, QLabel, QMessageBox,
    QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout,
)

from ..notebook import stored_integrals as nb_stored

_NOTES_COLUMN = [key for key, _ in nb_stored.COLUMNS].index("notes")
_FILTERS = {"CSV (*.csv)": "csv", "JSON (*.json)": "json", "Text (*.txt)": "txt"}


class StoredIntegralsDialog(QDialog):
    """A non-modal window on a :class:`~opennmr.notebook.stored_integrals.StoredIntegrals`."""

    def __init__(self, store, parent=None) -> None:
        super().__init__(parent)
        self.store = store
        self.setWindowTitle("Stored integrals")
        self.resize(900, 420)
        self._filling = False
        v = QVBoxLayout(self)
        self.intro = QLabel(
            "Integrals kept from Integrate ▸ Integration Regions. Double-click a note to edit it. "
            f"Kept in {store.path or 'memory'}.")
        self.intro.setWordWrap(True)
        v.addWidget(self.intro)

        self.table = QTableWidget(0, len(nb_stored.COLUMNS))
        self.table.setHorizontalHeaderLabels([heading for _, heading in nb_stored.COLUMNS])
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.itemChanged.connect(self._on_item_changed)
        v.addWidget(self.table)

        row = QHBoxLayout()
        self.remove_button = QPushButton("Remove Selected")
        self.remove_button.clicked.connect(self.remove_selected)
        self.clear_button = QPushButton("Clear All")
        self.clear_button.clicked.connect(self.clear_all)
        self.save_button = QPushButton("Save Table As…")
        self.save_button.setToolTip("Save the table as a CSV, JSON or plain-text file.")
        self.save_button.clicked.connect(self.save_as)
        for button in (self.remove_button, self.clear_button, self.save_button):
            row.addWidget(button)
        row.addStretch(1)
        v.addLayout(row)
        self.refresh()

    # ------------------------------------------------------------------
    def refresh(self) -> None:
        """Re-read the table from the store."""
        self._filling = True
        try:
            _, rows = self.store.table()
            self.table.setRowCount(len(rows))
            for r, cells in enumerate(rows):
                for c, text in enumerate(cells):
                    item = QTableWidgetItem(text)
                    if c != _NOTES_COLUMN:
                        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    self.table.setItem(r, c, item)
        finally:
            self._filling = False
        empty = not len(self.store)
        for button in (self.remove_button, self.clear_button, self.save_button):
            button.setEnabled(not empty)

    def _on_item_changed(self, item) -> None:
        if self._filling or item.column() != _NOTES_COLUMN:
            return
        if 0 <= item.row() < len(self.store):
            self.store.set_notes(item.row(), item.text())

    def remove_selected(self) -> None:
        rows = sorted({index.row() for index in self.table.selectedIndexes()})
        if rows:
            self.store.remove(rows)
            self.refresh()

    def clear_all(self) -> None:
        if not len(self.store):
            return
        answer = QMessageBox.question(
            self, "Clear stored integrals", f"Delete all {len(self.store)} stored integrals?")
        if answer == QMessageBox.StandardButton.Yes:
            self.store.clear()
            self.refresh()

    def save_as(self) -> str | None:
        """Ask for a file and write the table to it (CSV, JSON or text by
        the file type chosen). Returns the path, or None if cancelled."""
        path, chosen = QFileDialog.getSaveFileName(
            self, "Save stored integrals", "stored_integrals.csv", ";;".join(_FILTERS))
        if not path:
            return None
        fmt = _FILTERS.get(chosen)
        suffix = path.rsplit(".", 1)[-1].lower() if "." in path else ""
        if suffix in nb_stored.EXPORT_FORMATS:
            fmt = suffix                       # the name typed wins over the filter
        elif fmt:
            path += "." + fmt
        try:
            self.store.export(path, fmt)
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, "Save stored integrals", str(exc))
            return None
        return path
