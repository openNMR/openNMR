"""Fit ▸ Kept fit models: several fits of one spectrum, kept with it.

Keep the fit on the plot (button here, or right-click the plot ▸ Keep
This Fit…) under a name; it is saved with the spectrum (its processing
state, so its sidecar, save points and openNMR states) and listed here
whenever the spectrum is selected. From the list: show one, put one back
into the panel that made it, or tick several and compare them on one
plot -- data, each fit, residuals -- with R², reduced χ², ΔAIC and ΔBIC.

The records and statistics are :mod:`opennmr.notebook.fit_models`, so a
notebook reads and compares the same models.
"""

from __future__ import annotations

import math

from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QHBoxLayout, QHeaderView, QInputDialog, QLabel,
    QMenu, QMessageBox, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from ..notebook import fit_models as fm

_COLUMNS = ("Compare", "Name", "Kind", "k", "R²", "Red. χ²", "ΔAIC", "ΔBIC")


def _fmt(value, spec):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    return format(value, spec)


class FitModelsPanel(QWidget):
    """Callbacks (MainWindow): ``models()`` -> decoded models of the
    selected spectrum, each with an ``id``; ``keep()``; ``show(id)``;
    ``compare(ids, residuals, components_of)``; ``restore(id)``;
    ``rename(id, name)``; ``delete(id)``; ``can_restore(model)``."""

    def __init__(self, models, keep, show, compare, restore, rename, delete, can_restore,
                 parent=None) -> None:
        super().__init__(parent)
        self._models, self._keep, self._show = models, keep, show
        self._compare, self._restore, self._rename, self._delete = compare, restore, rename, delete
        self._can_restore = can_restore
        self._rows: list[dict] = []
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        intro = QLabel(
            "Keep several fits of the selected spectrum -- a 2-, 3- and 4-site model, say -- "
            "and compare them. Kept fits are saved with the spectrum. (Or right-click any fit "
            "plot ▸ Keep This Fit…)")
        intro.setWordWrap(True)
        v.addWidget(intro)
        self.keep_button = QPushButton("Keep the Fit on the Plot…")
        self.keep_button.setToolTip(
            "Save the fit on the Fit (or Spectral Fit) view with this spectrum, under a name: "
            "the fitted curve and components, its report, and the fit panel's settings and "
            "result so it can be put back.")
        self.keep_button.clicked.connect(lambda: self._keep())
        v.addWidget(self.keep_button)

        self.table = QTableWidget(0, len(_COLUMNS))
        self.table.setHorizontalHeaderLabels(list(_COLUMNS))
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setToolTip(
            "k: free parameters. ΔAIC / ΔBIC: from the best of the ticked models, when they were "
            "fitted to the same points -- lower is better; a ΔBIC over 6 is strong, over 10 very "
            "strong evidence against a model (Kass & Raftery 1995). R² always favours more sites; "
            "the criteria charge for them.\n\nDouble-click to show a fit; right-click for more.")
        self.table.cellDoubleClicked.connect(lambda row, _c: self._on_show(row))
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._on_menu)
        v.addWidget(self.table)
        self.empty_label = QLabel("<i>No kept fits for the selected spectrum.</i>")
        v.addWidget(self.empty_label)

        row = QHBoxLayout()
        self.show_button = QPushButton("Show")
        self.show_button.clicked.connect(lambda: self._on_show(self.table.currentRow()))
        self.restore_button = QPushButton("Put Back in Its Panel")
        self.restore_button.setToolTip("Restore the fit panel's settings and result as they were "
                                       "when this fit was kept, and make it the current fit.")
        self.restore_button.clicked.connect(lambda: self._on_restore(self.table.currentRow()))
        for b in (self.show_button, self.restore_button):
            row.addWidget(b)
        v.addLayout(row)

        compare_row = QHBoxLayout()
        self.compare_button = QPushButton("Compare Ticked")
        self.compare_button.setToolTip("The data, each ticked fit and their residuals on the Fit "
                                       "view, and ΔAIC/ΔBIC in the table.")
        self.compare_button.clicked.connect(self.compare)
        self.residuals_check = QCheckBox("Residuals")
        self.residuals_check.setChecked(True)
        self.components_combo = QComboBox()
        self.components_combo.setToolTip("Shade the components of one of the compared fits.")
        compare_row.addWidget(self.compare_button)
        compare_row.addWidget(self.residuals_check)
        compare_row.addWidget(QLabel("Components of:"))
        compare_row.addWidget(self.components_combo, 1)
        v.addLayout(compare_row)
        self.note_label = QLabel("")
        self.note_label.setWordWrap(True)
        v.addWidget(self.note_label)
        self.refresh()

    # ------------------------------------------------------------------

    def ticked_ids(self) -> list:
        ids = []
        for row, model in enumerate(self._rows):
            check = self.table.cellWidget(row, 0)
            if check is not None and check.isChecked():
                ids.append(model["id"])
        return ids

    def set_ticked(self, ids) -> None:
        for row, model in enumerate(self._rows):
            check = self.table.cellWidget(row, 0)
            if check is not None:
                check.setChecked(model["id"] in set(ids))

    def refresh(self) -> None:
        """Re-read the selected spectrum's kept fits, keeping the ticks."""
        ticked = set(self.ticked_ids())
        self._rows = list(self._models() or [])
        self.table.setRowCount(len(self._rows))
        for row, model in enumerate(self._rows):
            check = QCheckBox()
            check.setChecked(model["id"] in ticked)
            check.setAccessibleName(f"Compare {model['name']}")
            check.toggled.connect(lambda _on: self._update_statistics())
            self.table.setCellWidget(row, 0, check)
            for col, text in ((1, model["name"]), (2, model.get("kind_label", model["kind"]))):
                item = QTableWidgetItem(text)
                item.setToolTip(f"{model['name']}\n{model.get('target', '')}\nkept {model['time']}")
                self.table.setItem(row, col, item)
        self.empty_label.setVisible(not self._rows)
        self.table.setVisible(bool(self._rows))
        self._update_statistics()

    def _update_statistics(self) -> None:
        ticked = set(self.ticked_ids())
        chosen = [m for m in self._rows if m["id"] in ticked]
        comparison = fm.compare_fit_models(chosen) if len(chosen) > 1 else None
        by_id = {}
        if comparison:
            for m, r in zip(chosen, comparison["rows"]):
                by_id[m["id"]] = r
        for row, model in enumerate(self._rows):
            stats = by_id.get(model["id"]) or {**fm.fit_statistics(model), "delta_aic": None,
                                                "delta_bic": None, "evidence": ""}
            cells = (_fmt(stats["k"], "d"), _fmt(stats["r2"], ".5f"),
                     _fmt(stats["reduced_chi2"], ".4g"), _fmt(stats["delta_aic"], ".1f"),
                     _fmt(stats["delta_bic"], ".1f"))
            for col, text in zip(range(3, 8), cells):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                if col == 7 and stats.get("evidence"):
                    item.setToolTip(stats["evidence"])
                self.table.setItem(row, col, item)
        blocked = self.components_combo.blockSignals(True)
        keep = self.components_combo.currentData()
        self.components_combo.clear()
        self.components_combo.addItem("none", None)
        for m in chosen:
            if len(m.get("components") or []) > 1:
                self.components_combo.addItem(m["name"], m["name"])
        self.components_combo.setCurrentIndex(max(0, self.components_combo.findData(keep)))
        self.components_combo.blockSignals(blocked)
        self.compare_button.setEnabled(len(chosen) >= 2)
        if comparison is None:
            self.note_label.setText("Tick two or more fits to compare them." if self._rows else "")
        else:
            best = [r["name"] for r in comparison["rows"] if r["delta_bic"] == 0]
            text = comparison["note"] or (
                f"Lowest BIC: {best[0]}." if best else "")
            self.note_label.setText(text)

    def comparison_text(self) -> str:
        chosen = [m for m in self._rows if m["id"] in set(self.ticked_ids())]
        return fm.format_comparison(fm.compare_fit_models(chosen)) if chosen else ""

    # ------------------------------------------------------------------

    def _model_at(self, row):
        return self._rows[row] if 0 <= row < len(self._rows) else None

    def _on_show(self, row) -> None:
        model = self._model_at(row)
        if model is not None:
            self._show(model["id"])

    def _on_restore(self, row) -> None:
        model = self._model_at(row)
        if model is None:
            return
        if not self._can_restore(model):
            QMessageBox.information(
                self, "Put back", f"“{model['name']}” did not come from a fit panel that can "
                "take it back (an isotropic peak fit or a curve fit is kept as its curves and "
                "report). Show it instead.")
            return
        self._restore(model["id"])

    def compare(self) -> None:
        ids = self.ticked_ids()
        if len(ids) < 2:
            QMessageBox.information(self, "Compare", "Tick at least two kept fits.")
            return
        self._compare(ids, self.residuals_check.isChecked(), self.components_combo.currentData())

    def _on_menu(self, pos) -> None:
        row = self.table.rowAt(pos.y())
        model = self._model_at(row)
        menu = QMenu(self)
        show = restore = rename = delete = copy_report = None
        if model is not None:
            show = menu.addAction("Show")
            restore = menu.addAction("Put Back in Its Panel")
            restore.setEnabled(bool(self._can_restore(model)))
            rename = menu.addAction("Rename…")
            copy_report = menu.addAction("Copy Report")
            delete = menu.addAction("Delete")
            menu.addSeparator()
        copy_table = menu.addAction("Copy Comparison Table")
        chosen = menu.exec(self.table.viewport().mapToGlobal(pos))
        if chosen is None:
            return
        if chosen is show:
            self._show(model["id"])
        elif chosen is restore:
            self._on_restore(row)
        elif chosen is rename:
            name, ok = QInputDialog.getText(self, "Rename kept fit", "Name:", text=model["name"])
            if ok and name.strip():
                self._rename(model["id"], name.strip())
        elif chosen is delete:
            answer = QMessageBox.question(self, "Delete kept fit",
                                          f"Delete “{model['name']}”? This cannot be undone.")
            if answer == QMessageBox.StandardButton.Yes:
                self._delete(model["id"])
        elif chosen is copy_report:
            QGuiApplication.clipboard().setText(model.get("report") or "")
        elif chosen is copy_table:
            QGuiApplication.clipboard().setText(self.comparison_text())
