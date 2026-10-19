"""Dynamics ▸ Batch: many pseudo-2D series -- every series processed,
measured and fitted the same way as one you have set up, then each
peak's T1/T2 with its temperature and field sent to Relaxation vs
Temperature for the BPP/Arrhenius fit (explicit user request, 2026-10-06).

The panel holds the list and the options; the main window does each
series (``run_one``), since processing, measuring and Fit Curve are its
own. Notebook equivalent: :func:`opennmr.notebook.batch_relaxation.NMR2D_relaxation_batch`.
"""

from __future__ import annotations

import numpy as np
from .widgets.report_label import ReportLabel
from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QCheckBox, QComboBox, QFormLayout, QHBoxLayout,
    QHeaderView, QLabel, QMessageBox, QProgressDialog, QPushButton, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

from ..notebook import batch_relaxation as nb_batch

_C_NAME, _C_T, _C_FIELD, _C_STATUS, _C_RESULT = range(5)
_HEADERS = ("Experiment", "T (K)", "Field (MHz)", "Status", "Result")


class BatchRelaxationPanel(QWidget):
    """Callbacks (all from the main window):

    ``open_items()`` -> [(key, label)] of open experiments;
    ``selected_keys()`` -> keys selected in Open experiments;
    ``open_expnos(folder, expnos)`` -> (keys, problems);
    ``describe(key)`` -> {"label", "T_K", "larmor_MHz", "pseudo_2d"} or None;
    ``reference()`` -> (key, label) of the series set up by hand, or None;
    ``run_one(key, options)`` -> {"rows": [...], "status": str, "error": str|None};
    ``send_rows(rows)`` -> number sent (to Relaxation vs Temperature).
    """

    def __init__(self, open_items, selected_keys, open_expnos, describe, reference, run_one,
                 send_rows, parent=None) -> None:
        super().__init__(parent)
        self._open_items = open_items
        self._selected_keys = selected_keys
        self._open_expnos = open_expnos
        self._describe = describe
        self._reference = reference
        self._run_one = run_one
        self._send_rows = send_rows
        self.entries: list[dict] = []
        self.rows: list[dict] = []
        self._filling = False

        v = QVBoxLayout(self)
        blurb = QLabel(
            "Fit a whole set of pseudo-2D series in one go (e.g. one T₁ measurement per "
            "temperature, at one field or several). Set up one series by hand first (process it, "
            "add its peaks or regions, choose the model and its delays), select it, then add "
            "the others here: each is processed and measured the same way and fitted, and "
            "every peak's T₁/T₂ goes to Relaxation vs Temperature with its temperature and "
            "field, grouped by field.")
        blurb.setWordWrap(True)
        v.addWidget(blurb)
        self.reference_label = QLabel("")
        self.reference_label.setWordWrap(True)
        v.addWidget(self.reference_label)

        self.table = QTableWidget(0, len(_HEADERS))
        self.table.setHorizontalHeaderLabels(list(_HEADERS))
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setMinimumHeight(140)
        self.table.setToolTip("T and field come from each experiment's own parameters (TE, "
                              "SFO1); type over them where they are missing or wrong.")
        self.table.itemChanged.connect(self._on_edited)
        v.addWidget(self.table)
        row = QHBoxLayout()
        self.add_selected_button = QPushButton("Add selected")
        self.add_selected_button.setToolTip("Add the pseudo-2D series selected in Open "
                                            "experiments.")
        self.add_selected_button.clicked.connect(self.add_selected)
        self.add_expnos_button = QPushButton("Add by number…")
        self.add_expnos_button.setToolTip("Open experiments of one dataset folder by number "
                                          "(e.g., 10-25).")
        self.add_expnos_button.clicked.connect(self._on_add_expnos)
        self.remove_button = QPushButton("Remove")
        self.remove_button.clicked.connect(self.remove_selected)
        for b in (self.add_selected_button, self.add_expnos_button, self.remove_button):
            row.addWidget(b)
        v.addLayout(row)

        form = QFormLayout()
        self.processing_check = QCheckBox("Process every series as the reference is "
                                          "(phase, apodisation, zero filling, baseline…)")
        self.processing_check.setChecked(True)
        form.addRow("", self.processing_check)
        self.delays_combo = QComboBox()
        self.delays_combo.addItem("Each series reads its own (as the reference's Values "
                                  "setting does)", "own")
        self.delays_combo.addItem("The reference's values, for every series", "reference")
        self.delays_combo.setToolTip(
            "Own: with Values on Auto or a list file: each series' delays come from its own "
            "folder (vdlist/vclist). The reference's: every "
            "series gets the reference's delays, written out.")
        form.addRow("Delays:", self.delays_combo)
        self.send_check = QCheckBox("Send the results to Relaxation vs Temperature when done")
        self.send_check.setChecked(True)
        form.addRow("", self.send_check)
        v.addLayout(form)

        run_row = QHBoxLayout()
        self.run_button = QPushButton("Process and Fit Every Series")
        self.run_button.clicked.connect(lambda: self.run())
        self.send_button = QPushButton("Send to Relaxation vs Temperature")
        self.send_button.setEnabled(False)
        self.send_button.clicked.connect(lambda: self.send())
        self.csv_button = QPushButton("Copy Results (CSV)")
        self.csv_button.setEnabled(False)
        self.csv_button.clicked.connect(
            lambda: QGuiApplication.clipboard().setText(nb_batch.batch_table(self.rows)))
        run_row.addWidget(self.run_button, 1)
        run_row.addWidget(self.send_button)
        run_row.addWidget(self.csv_button)
        v.addLayout(run_row)
        self.summary_label = ReportLabel("")
        self.summary_label.setWordWrap(True)
        self.summary_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        v.addWidget(self.summary_label)
        self.refresh_reference()

    # ---- the list ------------------------------------------------------------

    def refresh_reference(self) -> None:
        ref = self._reference()
        self.reference_label.setText(
            f"Reference (set up by hand): <b>{ref[1]}</b>, the series selected in Open "
            "experiments." if ref else
            "Reference: select the pseudo-2D series you have set up in Open experiments.")

    def add_keys(self, keys) -> list[str]:
        problems = []
        have = {e["key"] for e in self.entries}
        for key in keys:
            if key in have:
                continue
            info = self._describe(key)
            if info is None:
                problems.append(f"{key}: not open")
                continue
            if not info.get("pseudo_2d"):
                problems.append(f"{info['label']}: not a pseudo-2D series")
                continue
            self.entries.append({"key": key, "label": info["label"], "T_K": info.get("T_K"),
                                 "larmor_MHz": info.get("larmor_MHz"), "status": "",
                                 "result": "", "rows": []})
            have.add(key)
        self._fill()
        return problems

    def add_selected(self) -> None:
        problems = self.add_keys(self._selected_keys())
        if problems:
            QMessageBox.information(self, "Batch", "Not added:\n" + "\n".join(problems))

    def _on_add_expnos(self) -> None:
        from .multi_fit_panel import ExpnoDialog
        dialog = ExpnoDialog(self)
        if dialog.exec() != dialog.DialogCode.Accepted:
            return
        try:
            folder, expnos = dialog.values()
        except ValueError as exc:
            QMessageBox.warning(self, "Batch", str(exc))
            return
        keys, problems = self._open_expnos(folder, expnos)
        problems += self.add_keys(keys)
        if problems:
            QMessageBox.information(self, "Batch", "Not added:\n" + "\n".join(problems))

    def remove_selected(self) -> None:
        rows = sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True)
        for r in rows:
            del self.entries[r]
        self._fill()

    def _fill(self) -> None:
        self._filling = True
        t = self.table
        t.setRowCount(len(self.entries))
        for r, e in enumerate(self.entries):
            cells = [e["label"], "" if e["T_K"] is None else f"{e['T_K']:g}",
                     "" if e["larmor_MHz"] is None else f"{e['larmor_MHz']:.6g}",
                     e["status"], e["result"]]
            for c, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if c not in (_C_T, _C_FIELD):
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                t.setItem(r, c, item)
        self._filling = False

    def _on_edited(self, item) -> None:
        if self._filling:
            return
        e = self.entries[item.row()]
        text = item.text().strip()
        key = "T_K" if item.column() == _C_T else "larmor_MHz" if item.column() == _C_FIELD \
            else None
        if key is None:
            return
        try:
            e[key] = float(text) if text else None
        except ValueError:
            self._fill()

    # ---- running ---------------------------------------------------------------

    def options(self) -> dict:
        return {"apply_processing": self.processing_check.isChecked(),
                "delays": self.delays_combo.currentData()}

    def run(self, show_progress: bool = True) -> bool:
        self.refresh_reference()
        if self._reference() is None:
            QMessageBox.information(self, "Batch", "Select the series you have set up by hand "
                                    "(its processing, peaks or regions, model and delays) in "
                                    "Open experiments first: it is the reference the others "
                                    "follow.")
            return False
        if not self.entries:
            QMessageBox.information(self, "Batch", "Add the series to fit first.")
            return False
        options = self.options()
        # The reference as it is now: running the batch selects each series
        # in turn, so "the selected series" changes under it.
        options["reference"] = self._reference()[0]
        dialog = None
        if show_progress:
            dialog = QProgressDialog("Fitting…", "Stop", 0, len(self.entries), self)
            dialog.setWindowTitle("Batch fit")
            dialog.setWindowModality(Qt.WindowModality.WindowModal)
            dialog.setMinimumDuration(0)
        self.rows = []
        done = failed = 0
        for k, e in enumerate(self.entries):
            if dialog is not None:
                dialog.setLabelText(f"{e['label']} ({k + 1} of {len(self.entries)})")
                dialog.setValue(k)
                QApplication.processEvents()
                if dialog.wasCanceled():
                    e["status"] = "not run (stopped)"
                    continue
            out = self._run_one(e["key"], options)
            rows = out.get("rows") or []
            for r in rows:
                r["experiment"] = e["label"]
                if e["T_K"] is not None:
                    r["T_K"] = e["T_K"]          # a typed temperature wins
                if e["larmor_MHz"] is not None:
                    r["larmor_MHz"] = e["larmor_MHz"]
            e["rows"] = rows
            e["status"] = out.get("status") or ("fitted" if rows else "failed")
            e["result"] = "; ".join(
                f"{r['series']}: {r['quantity']} {r['value']:.4g} s"
                + ("" if not np.isfinite(r["error"]) else f" ± {r['error']:.2g}")
                for r in rows) or (out.get("error") or "")
            if rows:
                done += 1
                self.rows += rows
            else:
                failed += 1
            self._fill()
        if dialog is not None:
            dialog.setValue(len(self.entries))
        groups = nb_batch.group_rows(self.rows)
        fields = sorted({f for (_s, f) in groups if f is not None})
        missing_T = sum(1 for r in self.rows if r["T_K"] is None)
        lines = [f"Fitted {done} of {len(self.entries)} series"
                 + (f"; {failed} failed (see Status)" if failed else "") + "."]
        if self.rows:
            lines.append(f"{len(self.rows)} value(s) in {len(groups)} group(s), each peak at "
                         "each field: " + (", ".join(f"{f:g} MHz" for f in fields) or
                                           "field unknown") + ".")
        if missing_T:
            lines.append(f"{missing_T} value(s) have no temperature: type it in the T column, "
                         "then send again.")
        self.summary_label.setText(" ".join(lines))
        self.send_button.setEnabled(bool(self.rows))
        self.csv_button.setEnabled(bool(self.rows))
        if self.rows and self.send_check.isChecked():
            self.send()
        return done > 0

    def send(self) -> int:
        if not self.rows:
            return 0
        # Typed temperatures/fields edited after the run still count.
        by_label = {e["label"]: e for e in self.entries}
        for r in self.rows:
            e = by_label.get(r.get("experiment"))
            if e is not None:
                if e["T_K"] is not None:
                    r["T_K"] = e["T_K"]
                if e["larmor_MHz"] is not None:
                    r["larmor_MHz"] = e["larmor_MHz"]
        return self._send_rows(list(self.rows))

    # ---- saved state ----------------------------------------------------------

    def get_state(self) -> dict:
        return {"entries": [{k: e[k] for k in ("key", "label", "T_K", "larmor_MHz")}
                            for e in self.entries], "options": self.options()}
