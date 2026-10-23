"""Fit ▸ Kept fit models: several fits of one spectrum, kept with it.

Keep the fit on the plot (button here, or right-click the plot ▸ Keep
This Fit…) under a name; it is saved with the spectrum (its processing
state, so its sidecar, save points and openNMR states) and listed here
whenever the spectrum is selected. From the list: show one, put one back
into the panel that made it, or tick several and compare them -- with
R², reduced χ², ΔAIC and ΔBIC in the table.

Compare Ticked puts each ticked fit in Open experiments (once; the same
entry is reused) and shows them in Compare, stacked (a row per fit: its
data, fit, components and residual) or superimposed, where Compare's own
tools apply: Plot Style colours, reordering, export, and each entry's
right-click ▸ Fit Display… for its components, colours and residual. The
older one-figure summary (data once, every fit, residual strips) is kept
as the third choice. Curve fits can only use the summary.

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

#: The kept fit's "put it back" action (explicit user request, 2026-10-08:
#: re-labelled from "Put Back in Its Panel").
RESTORE_TEXT = "Re-plot and Show “What's Next?”"

_COLUMNS = ("Compare", "Name", "Kind", "k", "R²", "Red. χ²", "ΔAIC", "ΔBIC")


def _fmt(value, spec):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    return format(value, spec)


class FitModelsPanel(QWidget):
    """Callbacks (MainWindow): ``models()`` -> decoded models of the
    selected spectrum, each with an ``id``; ``keep()``; ``show(id)``;
    ``compare(ids, residuals, components_of, layout)``; ``restore(id)``;
    ``rename(id, name)``; ``delete(id)``; ``can_restore(model)``;
    ``to_open(id)`` (optional): list it in Open experiments; ``recolour(id,
    colour)`` (optional); ``save_values(ids)`` (optional): save their
    fitted values; ``owner()`` (optional): the label of the spectrum whose
    kept fits these are."""

    #: Compare Ticked's choices (explicit user request: stacked or
    #: superimposed, as Compare offers).
    LAYOUTS = (("Stacked in Compare", "stack"), ("Superimposed in Compare", "overlay"),
               ("Summary figure (Fit view)", "summary"))

    def __init__(self, models, keep, show, compare, restore, rename, delete, can_restore,
                 parent=None, to_open=None, recolour=None, save_values=None,
                 owner=None) -> None:
        super().__init__(parent)
        self._to_open, self._recolour = to_open, recolour
        self._save_values, self._owner = save_values, owner
        #: The last Compare Ticked (ids, layout): Residuals and Components
        #: of: redraw it at once rather than needing another compare
        #: (explicit user request, 2026-10-08).
        self._last_compare: tuple | None = None
        self._models, self._keep, self._show = models, keep, show
        self._compare, self._restore, self._rename, self._delete = compare, restore, rename, delete
        self._can_restore = can_restore
        self._rows: list[dict] = []
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        intro = QLabel(
            "Keep several fits of the selected spectrum (e.g., 2-, 3- and 4-site models) "
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
        self.owner_label = QLabel("")
        self.owner_label.setWordWrap(True)
        v.addWidget(self.owner_label)

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
            "fitted to the same points – lower is better; a ΔBIC over 6 is strong, over 10 very "
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
        self.restore_button = QPushButton(RESTORE_TEXT)
        self.restore_button.setToolTip(
            "Re-plot this fit and put its settings and result back in the fit panel that made "
            "it (making it the current fit), then open that panel at “What's next?” – what "
            "the fit tells you and what to try next.")
        self.restore_button.clicked.connect(lambda: self._on_restore(self.table.currentRow()))
        for b in (self.show_button, self.restore_button):
            row.addWidget(b)
        v.addLayout(row)
        row2 = QHBoxLayout()
        self.to_open_button = QPushButton("Send to Open Experiments")
        self.to_open_button.setToolTip(
            "List the ticked fits (or, with none ticked, the selected one) in Open experiments: "
            "the data with the full fit drawn over it, to view, style and export like any "
            "spectrum. Each is listed once and reused after. The kept fits stay here.")
        self.to_open_button.clicked.connect(self.send_to_open)
        self.to_open_button.setVisible(to_open is not None)
        self.save_values_button = QPushButton("Save the Fitted Values…")
        self.save_values_button.setToolTip(
            "Write the fitted values and their 1σ errors of the ticked fits (or the selected "
            "one) to a JSON file anyone can read: a table of site/peak × parameter, with "
            "the fit's quality and report.")
        self.save_values_button.clicked.connect(self.save_values)
        self.save_values_button.setVisible(save_values is not None)
        row2.addWidget(self.to_open_button)
        row2.addWidget(self.save_values_button)
        v.addLayout(row2)

        compare_row = QHBoxLayout()
        self.compare_button = QPushButton("Compare Ticked")
        self.compare_button.setToolTip(
            "Each ticked fit into Open experiments (reused if already there) and onto the "
            "Compare view – stacked, a row per fit, or superimposed – with ΔAIC/ΔBIC in the "
            "table. Right-click an entry in Compare ▸ Fit Display… for its components, colours "
            "and residual; Plot Style recolours each fit.")
        self.compare_button.clicked.connect(self.compare)
        self.layout_combo = QComboBox()
        for text, data in self.LAYOUTS:
            self.layout_combo.addItem(text, data)
        self.layout_combo.setToolTip("Stacked: every row the data with one fit (the data "
                                     "repeated). Superimposed: the data once, every fit over it. "
                                     "Summary: one figure on the Fit view (curve fits too).")
        self.residuals_check = QCheckBox("Residuals")
        self.residuals_check.setChecked(True)
        self.residuals_check.setToolTip("Show each fit's residual. Switching it redraws the "
                                        "comparison straight away.")
        self.residuals_check.toggled.connect(lambda _on: self._redraw_comparison())
        self.components_combo = QComboBox()
        self.components_combo.setToolTip("Shade the components of one of the compared fits. "
                                         "Changing it redraws the comparison straight away.")
        self.components_combo.currentIndexChanged.connect(lambda _i: self._redraw_comparison())
        compare_row.addWidget(self.compare_button)
        compare_row.addWidget(self.layout_combo)
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
        owner = self._owner() if self._owner is not None else ""
        self.owner_label.setText(f"Kept fits of <b>{owner}</b>" if owner and self._rows else "")
        self.owner_label.setVisible(bool(owner and self._rows))
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
                self, "Re-plot", f"“{model['name']}” did not come from a fit panel that can "
                "take it back (an isotropic peak fit or a curve fit is kept as its curves and "
                "report), so it has no “What's next?”. Show it instead.")
            return
        self._restore(model["id"])

    def compare(self) -> None:
        ids = self.ticked_ids()
        if len(ids) < 2:
            QMessageBox.information(self, "Compare", "Tick at least two kept fits.")
            return
        layout = self.layout_combo.currentData()
        self._compare(ids, self.residuals_check.isChecked(), self.components_combo.currentData(),
                      layout)
        self._last_compare = (list(ids), layout)

    def _redraw_comparison(self) -> None:
        """Residuals / Components of: redraw the comparison on show, if
        its fits are all still here."""
        if self._last_compare is None:
            return
        ids, layout = self._last_compare
        present = {m["id"] for m in self._rows}
        if len(ids) < 2 or not set(ids) <= present:
            self._last_compare = None
            return
        self._compare(ids, self.residuals_check.isChecked(), self.components_combo.currentData(),
                      layout)

    def _chosen_ids(self) -> list:
        """The ticked fits, or with none ticked the selected one."""
        ids = self.ticked_ids()
        if not ids:
            model = self._model_at(self.table.currentRow())
            ids = [model["id"]] if model is not None else []
        return ids

    def send_to_open(self) -> None:
        if self._to_open is None:
            return
        ids = [i for i in self._chosen_ids()
               if next((m for m in self._rows if m["id"] == i), {}).get("kind") != "curve"]
        if not ids:
            QMessageBox.information(self, "Send to Open Experiments",
                                    "Tick (or select) a kept spectrum fit first – a curve fit "
                                    "has no spectrum to list.")
            return
        for model_id in ids:
            self._to_open(model_id)

    def save_values(self) -> None:
        if self._save_values is None:
            return
        ids = self._chosen_ids()
        if not ids:
            QMessageBox.information(self, "Save the fitted values",
                                    "Tick (or select) a kept fit first.")
            return
        self._save_values(ids)

    def _on_menu(self, pos) -> None:
        row = self.table.rowAt(pos.y())
        model = self._model_at(row)
        menu = QMenu(self)
        show = restore = rename = delete = copy_report = to_open = recolour = save = None
        if model is not None:
            show = menu.addAction("Show")
            if self._to_open is not None and model.get("kind") != "curve":
                to_open = menu.addAction("Send to Open Experiments")
            if self._recolour is not None:
                recolour = menu.addAction("Colour…")
            restore = menu.addAction(RESTORE_TEXT)
            restore.setEnabled(bool(self._can_restore(model)))
            if self._save_values is not None:
                save = menu.addAction("Save the Fitted Values…")
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
        elif chosen is to_open:
            self._to_open(model["id"])
        elif chosen is save:
            self._save_values([model["id"]])
        elif chosen is recolour:
            from PySide6.QtGui import QColor
            from PySide6.QtWidgets import QColorDialog
            colour = QColorDialog.getColor(QColor(model.get("colour") or "#d62728"), self,
                                           f"Colour of {model['name']}")
            if colour.isValid():
                self._recolour(model["id"], colour.name())
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
