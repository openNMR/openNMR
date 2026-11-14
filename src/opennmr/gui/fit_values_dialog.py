"""A fit's parameters and their uncertainties as a table (explicit user
request, 2026-11-13: fits in Open experiments "store tables of parameters
and errors that can be restored, copied and exported").

The record is :mod:`opennmr.notebook.fit_values`'s -- the same one Save the
Fitted Values writes -- shown one parameter per row, in full precision:
Site, Parameter, Value, Error (1σ), Fixed. Copy puts it on the clipboard
(tab-separated: it pastes into a spreadsheet), Export writes CSV or the
JSON record, Show draws the fit again and Restore puts it back in the
panel that made it. Not modal: several can be open side by side.
"""

from __future__ import annotations

import os

from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QAbstractItemView, QDialog, QDialogButtonBox, QFileDialog, QHBoxLayout, QHeaderView, QLabel,
    QMessageBox, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout,
)

from ..notebook import fit_values


class FitValuesDialog(QDialog):
    def __init__(self, record: dict, parent=None, *, on_show=None, on_restore=None,
                 folder: str = "", on_saved=None) -> None:
        super().__init__(parent)
        self.record = record
        self._folder = folder
        self._on_saved = on_saved
        title = " — ".join(t for t in (record.get("fit"), record.get("spectrum")) if t)
        self.setWindowTitle(f"Fit parameters: {title}" if title else "Fit parameters")
        self.setMinimumSize(520, 360)
        layout = QVBoxLayout(self)

        self.header_label = QLabel(self._header())
        self.header_label.setWordWrap(True)
        self.header_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.header_label)

        cells = fit_values.parameter_cells(record)
        self.table = QTableWidget(len(cells), len(fit_values.PARAMETER_ROW_COLUMNS))
        self.table.setHorizontalHeaderLabels(list(fit_values.PARAMETER_ROW_COLUMNS))
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.verticalHeader().setVisible(False)
        for r, row in enumerate(cells):
            for c, text in enumerate(row):
                item = QTableWidgetItem(text)
                if c in (2, 3):
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight
                                          | Qt.AlignmentFlag.AlignVCenter)
                self.table.setItem(r, c, item)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setStretchLastSection(True)
        layout.addWidget(self.table, 1)

        if cells:
            note = ""
        elif record.get("report"):
            note = "This fit stored no table of values; its report is below."
        else:
            note = "This fit stored no values."
        self.status_label = QLabel(note)
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        if not cells and record.get("report"):
            report = QLabel(str(record["report"]))
            report.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            report.setWordWrap(True)
            layout.addWidget(report)

        row = QHBoxLayout()
        self.copy_button = QPushButton("Copy Table")
        self.copy_button.setToolTip("Tab-separated, with its header: pastes into a spreadsheet.")
        self.copy_button.clicked.connect(self.copy_table)
        self.copy_button.setEnabled(bool(cells))
        row.addWidget(self.copy_button)
        self.csv_button = QPushButton("Export CSV…")
        self.csv_button.clicked.connect(lambda: self.export_csv())
        self.csv_button.setEnabled(bool(cells))
        row.addWidget(self.csv_button)
        self.json_button = QPushButton("Save as JSON…")
        self.json_button.setToolTip("The whole record: values, uncertainties, quality, report.")
        self.json_button.clicked.connect(lambda: self.save_json())
        row.addWidget(self.json_button)
        row.addStretch(1)
        self.show_button = QPushButton("Show Fit")
        self.show_button.setToolTip("Draw this fit on the Fit view again.")
        self.show_button.setVisible(on_show is not None)
        if on_show is not None:
            self.show_button.clicked.connect(lambda: on_show())
        row.addWidget(self.show_button)
        self.restore_button = QPushButton("Restore")
        self.restore_button.setToolTip(
            "Put these values back in the panel that made the fit, as its current fit.")
        self.restore_button.setVisible(on_restore is not None)
        if on_restore is not None:
            self.restore_button.clicked.connect(lambda: on_restore())
        row.addWidget(self.restore_button)
        layout.addLayout(row)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.close)
        layout.addWidget(buttons)

    def _header(self) -> str:
        record = self.record
        lines = [" — ".join(t for t in (record.get("fit"), record.get("spectrum")) if t)]
        when = record.get("fitted") or record.get("kept")
        if when:
            lines.append(f"Fitted {when}")
        quality = fit_values.quality_text(record)
        if quality:
            lines.append(quality)
        lines.append("Errors are 1σ; a fixed value has none. Values are "
                     + str(record.get("values are") or "as stored") + ".")
        return "\n".join(line for line in lines if line)

    # ---- actions --------------------------------------------------------

    def table_text(self) -> str:
        return fit_values.values_to_tsv(self.record)

    def copy_table(self) -> None:
        QGuiApplication.clipboard().setText(self.table_text())
        self.status_label.setText("Table copied.")

    def _suggested(self, extension: str) -> str:
        stem = fit_values.safe_file_stem(
            " - ".join(t for t in (self.record.get("spectrum"), self.record.get("fit")) if t)
            or "fit") + " parameters"
        return os.path.join(self._folder or "", stem + extension)

    def _written(self, path: str) -> str:
        self._folder = os.path.dirname(path)
        self.status_label.setText(f"Written to {path}")
        if self._on_saved is not None:
            self._on_saved(path)
        return path

    def export_csv(self, path: str | None = None) -> str | None:
        if path is None:
            path, _ = QFileDialog.getSaveFileName(self, "Export fit parameters",
                                                  self._suggested(".csv"), "CSV (*.csv)")
        if not path:
            return None
        try:
            return self._written(fit_values.values_to_csv(self.record, path))
        except OSError as exc:
            QMessageBox.warning(self, "Export fit parameters", f"Couldn't write {path}:\n{exc}")
            return None

    def save_json(self, path: str | None = None) -> str | None:
        if path is None:
            path, _ = QFileDialog.getSaveFileName(self, "Save fit parameters",
                                                  self._suggested(".json"), "JSON (*.json)")
        if not path:
            return None
        try:
            return self._written(fit_values.save_fitted_values(self.record, path))
        except OSError as exc:
            QMessageBox.warning(self, "Save fit parameters", f"Couldn't write {path}:\n{exc}")
            return None
