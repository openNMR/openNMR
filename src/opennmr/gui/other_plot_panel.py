"""
Apply ▸ Plot other data: any numbers as a 2-D or 3-D scatter, line or bar chart.

Add one or more data files (or pasted columns), then add as many *datasets* as you like: each is a
choice of x, y (and z) from one file, from a list typed here, or the row number, with its own
name, way of drawing and colour. Dynamics curve fits can be added too. The look of the plot
(marker size, line width, spines, colour map ...) is set in Plot Style, like every other plot.
The drawing is :func:`opennmr.notebook.other_plot.plot_datasets`; *Copy notebook code*
reproduces the figure.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QGuiApplication
from PySide6.QtWidgets import (
    QColorDialog, QComboBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QInputDialog, QLabel,
    QLineEdit, QListWidget, QPlainTextEdit, QPushButton, QSpinBox, QTreeWidget, QTreeWidgetItem,
    QVBoxLayout, QWidget,
)

import numpy as np

from ..notebook import other_plot as op
from .widgets.report_label import ReportLabel
from .widgets.typed_spinbox import TypedSpinBox

# Combo item data: plain strings (a tuple would come back from Qt as a list).
_ROWS = "@rows"
_LIST = "@list"
_NONE = "@none"
_COL = "col:"
_SELECTED = object()          # add_dataset: "the file selected in the list"


class OtherPlotPanel(QWidget):
    """``draw(draw)`` puts a figure-drawing function on the canvas (from MainWindow);
    ``fontsize()`` -> the figure font size set in Plot Style; ``style()`` -> the Plot Style
    "Data plots" options (marker size, line width ...); ``curve_fits()`` ->
    ``[(experiment label, [fit result per peak], row), ...]`` for *Add curve fit*."""

    DEFAULT_STYLE = dict(marker="o", marker_size=6.0, linewidth=1.5, colormap="viridis",
                         bar_width=0.8)

    def __init__(self, draw, fontsize=lambda: 12.0, curve_fits=lambda: [],
                 style=lambda: {}, parent=None) -> None:
        super().__init__(parent)
        self._draw, self._fontsize, self._curve_fits, self._style = draw, fontsize, curve_fits, style
        self.sources: list[dict] = []        # {"name", "path", "text", "skip", "table"}
        self.datasets: list[dict] = []       # see _new_record
        self._loading = False                # the editor is being filled: its edits are not user edits
        self._drawn = False
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        intro = QLabel("Plot any numbers as a scatter, line or bar chart, in two or three dimensions. "
                       "Add a data file, then add datasets: each takes its x and y from a file's "
                       "columns, from a list you type, or from the row number. Put as many "
                       "datasets on the plot as you like. Marker size, line width, spines and the "
                       "like are in Plot Style.")
        intro.setWordWrap(True)
        v.addWidget(intro)

        # -------------------------------------------------------------- 1. files
        box = QGroupBox("1. Data files")
        form = QVBoxLayout(box)
        self.source_list = QListWidget()
        self.source_list.setMaximumHeight(80)
        self.source_list.setToolTip("The files and pasted tables the datasets can draw from.")
        self.source_list.currentRowChanged.connect(self._source_selected)
        form.addWidget(self.source_list)
        row = QHBoxLayout()
        self.add_file_button = QPushButton("Add file…")
        self.add_file_button.clicked.connect(self._browse)
        self.remove_file_button = QPushButton("Remove file")
        self.remove_file_button.clicked.connect(self.remove_file)
        row.addWidget(self.add_file_button)
        row.addWidget(self.remove_file_button)
        row.addStretch(1)
        form.addLayout(row)
        skip_row = QHBoxLayout()
        skip_row.addWidget(QLabel("Skip header lines:"))
        self.skip_spin = TypedSpinBox()
        self.skip_spin.setRange(0, 100000)
        self.skip_spin.setToolTip("Lines at the top of a file (or pasted text) to leave out before the "
                                  "header row or the numbers: a title, instrument settings, comments. "
                                  "Blank lines count. Used for the next file you add; with a file "
                                  "selected in the list, changing it re-reads that file.")
        self.skip_spin.editingFinished.connect(self._skip_edited)
        skip_row.addWidget(self.skip_spin)
        skip_row.addStretch(1)
        form.addLayout(skip_row)
        self.paste_edit = QPlainTextEdit()
        self.paste_edit.setPlaceholderText("or paste columns here (with a header row), then Add pasted data")
        self.paste_edit.setMaximumHeight(70)
        form.addWidget(self.paste_edit)
        self.add_pasted_button = QPushButton("Add pasted data")
        self.add_pasted_button.clicked.connect(lambda: self.add_pasted())
        form.addWidget(self.add_pasted_button)
        v.addWidget(box)

        # -------------------------------------------------------------- 2. datasets
        box = QGroupBox("2. Datasets on the plot")
        dv = QVBoxLayout(box)
        top = QFormLayout()
        self.dim_combo = QComboBox()
        self.dim_combo.addItem("2-D (y against x)", "2d")
        self.dim_combo.addItem("3-D (z against x and y)", "3d")
        self.dim_combo.currentIndexChanged.connect(self._sync)
        self.default_kind_combo = QComboBox()
        for key in op.KINDS:
            self.default_kind_combo.addItem(op.KIND_NAMES[key], key)
        self.default_kind_combo.setToolTip("How datasets are drawn unless one is set to something else.")
        self.default_kind_combo.currentIndexChanged.connect(self._refresh_rows)
        top.addRow("Dimensions:", self.dim_combo)
        top.addRow("Draw datasets as:", self.default_kind_combo)
        dv.addLayout(top)
        self.dataset_tree = QTreeWidget()
        self.dataset_tree.setColumnCount(4)
        self.dataset_tree.setHeaderLabels(["Dataset", "Data", "Draw as", "Colour"])
        self.dataset_tree.setRootIsDecorated(False)
        self.dataset_tree.setMinimumHeight(110)
        self.dataset_tree.setMaximumHeight(170)
        self.dataset_tree.setToolTip("Every dataset on the plot and what it is made from. Select one to "
                                     "change it below.")
        self.dataset_tree.currentItemChanged.connect(lambda *_: self._dataset_selected())
        dv.addWidget(self.dataset_tree)
        buttons = QHBoxLayout()
        self.add_dataset_button = QPushButton("Add dataset")
        self.add_dataset_button.clicked.connect(lambda: self.add_dataset())
        self.add_fit_button = QPushButton("Add curve fit…")
        self.add_fit_button.setToolTip("The points and fitted curve of a Dynamics curve fit in an open experiment.")
        self.add_fit_button.clicked.connect(self.add_curve_fit)
        self.duplicate_button = QPushButton("Duplicate")
        self.duplicate_button.clicked.connect(self.duplicate_dataset)
        self.remove_dataset_button = QPushButton("Remove")
        self.remove_dataset_button.clicked.connect(self.remove_dataset)
        self.clear_datasets_button = QPushButton("Clear")
        self.clear_datasets_button.clicked.connect(self.clear_datasets)
        for b in (self.add_dataset_button, self.add_fit_button, self.duplicate_button,
                  self.remove_dataset_button, self.clear_datasets_button):
            buttons.addWidget(b)
        dv.addLayout(buttons)

        self.editor = QGroupBox("Selected dataset")
        ef = QFormLayout(self.editor)
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("legend name (blank = automatic)")
        self.name_edit.editingFinished.connect(self._name_edited)
        ef.addRow("Name:", self.name_edit)
        self.source_combo = QComboBox()
        self.source_combo.currentIndexChanged.connect(self._source_choice_changed)
        ef.addRow("From file:", self.source_combo)
        self.x_combo, self.y_combo, self.z_combo, self.yerr_combo, self.colour_by_combo = (
            QComboBox(), QComboBox(), QComboBox(), QComboBox(), QComboBox())
        self.x_list_edit, self.y_list_edit, self.z_list_edit = QLineEdit(), QLineEdit(), QLineEdit()
        for edit, axis in ((self.x_list_edit, "x"), (self.y_list_edit, "y"), (self.z_list_edit, "z")):
            edit.setPlaceholderText(f"numbers for {axis}: 1, 2, 3  or  0:10:2 (start:stop:step)")
            edit.setToolTip("A list of numbers typed here instead of a column: separated by commas, "
                            "spaces or new lines; start:stop:step makes a range that includes stop.")
            edit.editingFinished.connect(self._editor_changed)
        self._axis_rows = {}
        for key, label, combo, edit in (("x", "x:", self.x_combo, self.x_list_edit),
                                        ("y", "y:", self.y_combo, self.y_list_edit),
                                        ("z", "z (3-D):", self.z_combo, self.z_list_edit)):
            combo.currentIndexChanged.connect(self._editor_changed)
            holder = QWidget()
            hv = QVBoxLayout(holder)
            hv.setContentsMargins(0, 0, 0, 0)
            hv.addWidget(combo)
            hv.addWidget(edit)
            ef.addRow(label, holder)
            self._axis_rows[key] = (holder, label)
        self.yerr_combo.currentIndexChanged.connect(self._editor_changed)
        self.yerr_combo.setToolTip("A column of error bar sizes for y (2-D scatter and line).")
        ef.addRow("y error bars:", self.yerr_combo)
        self.colour_by_combo.currentIndexChanged.connect(self._editor_changed)
        self.colour_by_combo.setToolTip("A column that colours the points of a scatter (with a colour bar). "
                                        "The colour map is in Plot Style.")
        ef.addRow("Colour points by:", self.colour_by_combo)
        self.kind_combo = QComboBox()
        self.kind_combo.addItem("(as above)", None)
        for key in op.KINDS:
            self.kind_combo.addItem(op.KIND_NAMES[key], key)
        self.kind_combo.currentIndexChanged.connect(self._editor_changed)
        ef.addRow("Draw as:", self.kind_combo)
        self.colour_button = QPushButton("automatic")
        self.colour_button.clicked.connect(self._pick_colour)
        ef.addRow("Colour:", self.colour_button)
        # Multiply an axis by a constant (units, or matching another series' scale).
        self.scale_edits = {}
        scale_row = QWidget()
        sh = QHBoxLayout(scale_row)
        sh.setContentsMargins(0, 0, 0, 0)
        for axis in ("x", "y", "z"):
            sh.addWidget(QLabel(f"{axis} ×"))
            edit = QLineEdit("1")
            edit.setMaximumWidth(90)
            edit.setToolTip(f"Multiply every {axis} value of this dataset by this number "
                            "(1e-3 for a thousandth, for example). Error bars scale with y.")
            edit.editingFinished.connect(self._editor_changed)
            self.scale_edits[axis] = edit
            sh.addWidget(edit)
        sh.addStretch(1)
        ef.addRow("Multiply by:", scale_row)
        self._editor_form = ef
        dv.addWidget(self.editor)
        v.addWidget(box)

        # -------------------------------------------------------------- 3. labels
        box = QGroupBox("3. Labels")
        form = QFormLayout(box)
        self.title_edit, self.xlabel_edit, self.ylabel_edit, self.zlabel_edit = (
            QLineEdit(), QLineEdit(), QLineEdit(), QLineEdit())
        for w in (self.title_edit, self.xlabel_edit, self.ylabel_edit, self.zlabel_edit):
            w.setPlaceholderText("blank = none" if w is self.title_edit else "blank = the column name")
        form.addRow("Title:", self.title_edit)
        form.addRow("x label:", self.xlabel_edit)
        form.addRow("y label:", self.ylabel_edit)
        self.zlabel_row_label = QLabel("z label:")
        form.addRow(self.zlabel_row_label, self.zlabel_edit)
        v.addWidget(box)

        row = QHBoxLayout()
        self.plot_button = QPushButton("Plot")
        self.plot_button.clicked.connect(self.plot)
        self.copy_button = QPushButton("Copy Notebook Code")
        self.copy_button.clicked.connect(self.copy_code)
        row.addWidget(self.plot_button)
        row.addWidget(self.copy_button)
        row.addStretch(1)
        v.addLayout(row)
        self.report_label = ReportLabel("")
        v.addWidget(self.report_label)
        self._sync()
        self._dataset_selected()

    # ------------------------------------------------------------------ files
    def _browse(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "Data files", "",
                                                "Tables (*.csv *.tsv *.txt *.dat);;All files (*)")
        for path in paths:
            self.add_file(path)

    def add_file(self, path: str, skip_lines: int | None = None) -> bool:
        """Read ``path`` and add it to the files; the first one also gets a first dataset."""
        skip = self.skip_spin.value() if skip_lines is None else int(skip_lines)
        path = str(path).strip()
        try:
            table = op.read_table(path, skip_lines=skip)
        except ValueError as exc:
            self._fail(str(exc))
            return False
        self._append_source({"name": Path(path).stem, "path": path, "text": None, "skip": skip,
                             "table": table})
        return True

    def add_pasted(self, text: str | None = None, skip_lines: int | None = None) -> bool:
        text = self.paste_edit.toPlainText() if text is None else text
        skip = self.skip_spin.value() if skip_lines is None else int(skip_lines)
        if not text.strip():
            self._fail("Paste some columns first.")
            return False
        try:
            table = op.read_table(text.strip("\n") + "\n", skip_lines=skip)
        except ValueError as exc:
            self._fail(str(exc))
            return False
        n = 1 + sum(1 for s in self.sources if s["path"] is None)
        self._append_source({"name": "pasted" if n == 1 else f"pasted {n}", "path": None,
                             "text": text.strip("\n") + "\n", "skip": skip, "table": table})
        return True

    def _append_source(self, source: dict) -> None:
        taken = {s["name"] for s in self.sources}
        base, k = source["name"], 2
        while source["name"] in taken:               # two files of the same name stay distinguishable
            source["name"] = f"{base} ({k})"
            k += 1
        self.sources.append(source)
        self._refresh_sources(select=len(self.sources) - 1)
        shape = source["table"].shape
        self._fail(f"Read {source['name']}: {shape[0]} rows, {shape[1]} columns.")
        if not self.datasets:
            self.add_dataset(source=len(self.sources) - 1)

    def remove_file(self) -> None:
        row = self.source_list.currentRow()
        if not 0 <= row < len(self.sources):
            return
        del self.sources[row]
        for record in self.datasets:                 # datasets that drew from it, or from a later one
            if record["source"] == row:
                record["source"] = None
                for key in ("x", "y", "z", "yerr", "colour_by"):
                    if isinstance(record[key], dict) and "col" in record[key]:
                        record[key] = None
            elif record["source"] is not None and record["source"] > row:
                record["source"] -= 1
        self._refresh_sources(select=min(row, len(self.sources) - 1))
        self._refresh_rows()
        self._dataset_selected()
        self._fail("File removed; datasets that used it now need their own numbers.")

    def _refresh_sources(self, select: int | None = None) -> None:
        self.source_list.blockSignals(True)
        self.source_list.clear()
        for source in self.sources:
            shape = source["table"].shape
            self.source_list.addItem(f"{source['name']}  ({shape[0]} × {shape[1]})")
        self.source_list.blockSignals(False)
        if self.sources:
            self.source_list.setCurrentRow(len(self.sources) - 1 if select is None else select)
        self.remove_file_button.setEnabled(bool(self.sources))

    def _source_selected(self, row: int) -> None:
        if 0 <= row < len(self.sources):
            self.skip_spin.blockSignals(True)
            self.skip_spin.setValue(self.sources[row]["skip"])
            self.skip_spin.blockSignals(False)

    def _skip_edited(self) -> None:
        row = self.source_list.currentRow()
        if not 0 <= row < len(self.sources):
            return
        source = self.sources[row]
        if source["skip"] == self.skip_spin.value():
            return
        try:
            table = op.read_table(source["path"] if source["path"] else source["text"],
                                  skip_lines=self.skip_spin.value())
        except ValueError as exc:
            self._fail(str(exc))
            return
        source["skip"], source["table"] = self.skip_spin.value(), table
        self._refresh_sources(select=row)
        self._dataset_selected()
        self._refresh_rows()
        self._fail(f"Re-read {source['name']}: {table.shape[0]} rows, {table.shape[1]} columns.")

    # ------------------------------------------------------------------ dataset records
    @staticmethod
    def _new_record(**kw) -> dict:
        record = {"label": "", "auto_label": True, "source": None, "x": None, "y": None, "z": None,
                  "yerr": None, "colour_by": None, "kind": None, "colour": None, "fit": None,
                  "fit_code": None, "xscale": "1", "yscale": "1", "zscale": "1"}
        record.update(kw)
        return record

    def _columns(self, source) -> list[str]:
        return list(self.sources[source]["table"].columns) if source is not None else []

    def _auto_label(self, record: dict) -> str:
        """The name a dataset gets unless it was renamed: its y column, with the file's name when
        there is more than one file; "dataset N" when y is not a column."""
        if record["fit"] is not None:
            return record["fit"]["label"]
        ref = record["y"]
        if not (isinstance(ref, dict) and "col" in ref):
            number = self.datasets.index(record) + 1 if record in self.datasets else len(self.datasets) + 1
            return f"dataset {number}"
        name = self.sources[record["source"]]["name"] if record["source"] is not None else ""
        return f"{name}: {ref['col']}" if name and len(self.sources) > 1 else str(ref["col"])

    def _label(self, record: dict) -> str:
        return record["label"] if not record["auto_label"] and record["label"] else self._auto_label(record)

    @staticmethod
    def _ref_text(ref) -> str:
        if ref is None:
            return ""
        if "col" in ref:
            return str(ref["col"])
        if "list" in ref:
            return "custom list"
        return "row number"

    def _data_text(self, record: dict) -> str:
        if record["fit"] is not None:
            return f"curve fit of {record['fit']['experiment']}"
        name = self.sources[record["source"]]["name"] if record["source"] is not None else "own numbers"
        if self.dim_combo.currentData() == "3d":
            return (f"{name}: {self._ref_text(record['y'])} vs {self._ref_text(record['x'])}, "
                    f"z = {self._ref_text(record['z']) or '—'}")
        return f"{name}: {self._ref_text(record['y']) or '—'} vs {self._ref_text(record['x']) or '—'}"

    def add_dataset(self, source=_SELECTED, x=None, y=None, z=None) -> dict:
        """A new dataset: by default the first two columns of the selected file (the next unused
        column for y when there are already datasets from that file). ``x``/``y``/``z``: a column
        name, a list of numbers, or ``"rows"``. ``source``: the file's number, ``None`` for no file
        (own numbers), by default the file selected in the list."""
        if source is _SELECTED:
            source = self.source_list.currentRow() if self.sources else None
            if source is not None and not 0 <= source < len(self.sources):
                source = None
        cols = self._columns(source)

        def ref(value):
            if value is None:
                return None
            if isinstance(value, str) and value == "rows":
                return {"rows": True}
            if isinstance(value, str):
                return {"col": value}
            return {"list": [float(v) for v in value]}
        if source is not None and cols:
            used = {r["y"]["col"] for r in self.datasets
                    if r["source"] == source and isinstance(r["y"], dict) and "col" in r["y"]}
            free = [c for c in cols[1:] if c not in used] or cols[1:] or cols
            x = x if x is not None else cols[0]
            y = y if y is not None else free[0]
            if z is None and self.dim_combo.currentData() == "3d":
                z = next((c for c in cols[2:] if c not in (x, y)), cols[-1])
        record = self._new_record(source=source, x=ref(x), y=ref(y), z=ref(z))
        self.datasets.append(record)
        self._refresh_rows(select=len(self.datasets) - 1)
        self._fail(f"{len(self.datasets)} dataset(s) on the plot.")
        return record

    def duplicate_dataset(self) -> None:
        row = self._current_row()
        if row is None:
            return
        import copy
        record = copy.deepcopy(self.datasets[row])
        record["label"] = self._label(self.datasets[row]) + " (copy)"
        record["auto_label"] = False
        self.datasets.insert(row + 1, record)
        self._refresh_rows(select=row + 1)

    def remove_dataset(self) -> None:
        row = self._current_row()
        if row is None:
            return
        del self.datasets[row]
        self._refresh_rows(select=min(row, len(self.datasets) - 1))

    def clear_datasets(self) -> None:
        self.datasets = []
        self._refresh_rows()

    def add_curve_fit(self) -> None:
        """Add the points and the fitted curve of a Dynamics curve fit from an open experiment."""
        sources = [s_ for s_ in self._curve_fits() if s_[1]]
        if not sources:
            self._fail("No open experiment has a Dynamics curve fit yet.")
            return
        labels = [f"{label}  ({len(results)} peak{'s' if len(results) != 1 else ''})"
                  for label, results, _ in sources]
        chosen, ok = QInputDialog.getItem(self, "Add curve fit", "Experiment:", labels, 0, False)
        if not ok:
            return
        label, results, number = sources[labels.index(chosen)]
        try:
            new = op.curve_fit_datasets(results, label=label)
        except ValueError as exc:
            self._fail(str(exc))
            return
        for i, ds in enumerate(new):
            self.datasets.append(self._new_record(
                fit={"ds": ds, "label": ds["label"], "experiment": label, "row": number,
                     "first": i == 0, "n": len(new)},
                kind=None, colour=None))
        self._refresh_rows(select=len(self.datasets) - 1)
        self._fail(f"Added the curve fit of {label}: {len(new)} dataset(s).")

    # ------------------------------------------------------------------ the list and the editor
    def _current_row(self) -> int | None:
        item = self.dataset_tree.currentItem()
        row = self.dataset_tree.indexOfTopLevelItem(item) if item is not None else -1
        return row if 0 <= row < len(self.datasets) else None

    def _kind_text(self, record: dict) -> str:
        kind = record["kind"] or (record["fit"]["ds"]["kind"] if record["fit"] is not None else None) \
            or self.default_kind_combo.currentData()
        return op.KIND_NAMES.get(kind, str(kind))

    def _row_texts(self, record: dict) -> list[str]:
        return [self._label(record), self._data_text(record),
                self._kind_text(record), self._colour_text(record["colour"])]

    def _refresh_rows(self, *_a, select: int | None = None) -> None:
        keep = self._current_row() if select is None else select
        self.dataset_tree.blockSignals(True)
        self.dataset_tree.clear()
        for record in self.datasets:
            self.dataset_tree.addTopLevelItem(QTreeWidgetItem(self._row_texts(record)))
        self.dataset_tree.blockSignals(False)
        for c in range(4):
            self.dataset_tree.resizeColumnToContents(c)
        if self.datasets:
            self.dataset_tree.setCurrentItem(self.dataset_tree.topLevelItem(
                min(max(keep if keep is not None else 0, 0), len(self.datasets) - 1)))
        self._dataset_selected()

    def _refresh_row_text(self, row: int) -> None:
        item = self.dataset_tree.topLevelItem(row)
        if item is not None:
            for c, text in enumerate(self._row_texts(self.datasets[row])):
                item.setText(c, text)

    @staticmethod
    def _colour_text(colour) -> str:
        if colour is None or (isinstance(colour, str) and not colour):
            return "automatic"
        from matplotlib.colors import to_hex
        try:
            return to_hex(colour)
        except ValueError:
            return str(colour)

    def _fill_axis_combo(self, combo, cols, *, extras, current) -> None:
        combo.blockSignals(True)
        combo.clear()
        for text, data in extras:
            combo.addItem(text, data)
        for name in cols:
            combo.addItem(name, _COL + str(name))
        want = (_COL + str(current["col"])) if isinstance(current, dict) and "col" in current else (
            _LIST if isinstance(current, dict) and "list" in current else
            _ROWS if isinstance(current, dict) and "rows" in current else _NONE)
        index = combo.findData(want)
        combo.setCurrentIndex(max(0, index))
        combo.blockSignals(False)

    def _dataset_selected(self) -> None:
        row = self._current_row()
        on = row is not None
        self.editor.setEnabled(on)
        self.duplicate_button.setEnabled(on)
        self.remove_dataset_button.setEnabled(on)
        self.clear_datasets_button.setEnabled(bool(self.datasets))
        self._loading = True
        try:
            if not on:
                self.name_edit.setText("")
                return
            record = self.datasets[row]
            fit = record["fit"] is not None
            for w in (self.source_combo, self.x_combo, self.y_combo, self.z_combo, self.yerr_combo,
                      self.colour_by_combo, self.x_list_edit, self.y_list_edit, self.z_list_edit):
                w.setEnabled(not fit)
            self.name_edit.setText(self._label(record))
            self.source_combo.blockSignals(True)
            self.source_combo.clear()
            self.source_combo.addItem("(none: my own numbers)", None)
            for i, source in enumerate(self.sources):
                self.source_combo.addItem(source["name"], i)
            self.source_combo.setCurrentIndex(max(0, self.source_combo.findData(record["source"])))
            self.source_combo.blockSignals(False)
            cols = self._columns(record["source"])
            have = [("(custom list…)", _LIST)] + ([("row number (1, 2, 3 …)", _ROWS)] if cols else [])
            for key, combo, edit in (("x", self.x_combo, self.x_list_edit),
                                     ("y", self.y_combo, self.y_list_edit),
                                     ("z", self.z_combo, self.z_list_edit)):
                self._fill_axis_combo(combo, cols, extras=have, current=record[key])
                shown = isinstance(record[key], dict) and "list" in record[key]
                edit.setVisible(combo.currentData() == _LIST)
                edit.setText(", ".join(f"{v:g}" for v in record[key]["list"]) if shown else edit.text()
                             if combo.currentData() == _LIST else "")
            self._fill_axis_combo(self.yerr_combo, cols, extras=[("(none)", _NONE)], current=record["yerr"])
            self._fill_axis_combo(self.colour_by_combo, cols, extras=[("(one colour)", _NONE)],
                                  current=record["colour_by"])
            self.kind_combo.setCurrentIndex(max(0, self.kind_combo.findData(record["kind"])))
            self.colour_button.setText(self._colour_text(record["colour"]))
            for axis, edit in self.scale_edits.items():
                edit.setText(str(record.get(f"{axis}scale", "1")))
            self._sync()
        finally:
            self._loading = False

    def _ref_from_combo(self, combo, edit, name: str):
        data = combo.currentData()
        if data == _NONE:
            return None
        if data == _ROWS:
            return {"rows": True}
        if data == _LIST:
            try:
                values = op.number_list(edit.text())
            except ValueError as exc:
                self._fail(f"{name}: {exc}")
                return None
            return {"list": values} if values else None
        return {"col": str(data)[len(_COL):]}

    def _editor_changed(self, *_a) -> None:
        if self._loading:
            return
        row = self._current_row()
        if row is None:
            return
        record = self.datasets[row]
        if record["fit"] is None:
            for combo, edit in ((self.x_combo, self.x_list_edit), (self.y_combo, self.y_list_edit),
                                (self.z_combo, self.z_list_edit)):
                edit.setVisible(combo.currentData() == _LIST)
            record["x"] = self._ref_from_combo(self.x_combo, self.x_list_edit, "x")
            record["y"] = self._ref_from_combo(self.y_combo, self.y_list_edit, "y")
            record["z"] = self._ref_from_combo(self.z_combo, self.z_list_edit, "z")
            record["yerr"] = self._ref_from_combo(self.yerr_combo, QLineEdit(), "y error")
            record["colour_by"] = self._ref_from_combo(self.colour_by_combo, QLineEdit(), "colour")
        record["kind"] = self.kind_combo.currentData()
        for axis, edit in self.scale_edits.items():
            record[f"{axis}scale"] = edit.text().strip() or "1"
        if record["auto_label"] and record["fit"] is None:
            self.name_edit.blockSignals(True)
            self.name_edit.setText(self._label(record))
            self.name_edit.blockSignals(False)
        self._refresh_rows_text_only()

    def _source_choice_changed(self, *_a) -> None:
        if self._loading:
            return
        row = self._current_row()
        if row is None:
            return
        record = self.datasets[row]
        record["source"] = self.source_combo.currentData()
        cols = self._columns(record["source"])
        for key in ("x", "y", "z", "yerr", "colour_by"):          # columns of the old file do not carry over
            if isinstance(record[key], dict) and "col" in record[key] and record[key]["col"] not in cols:
                record[key] = None
        if cols:
            record["x"] = record["x"] or {"col": cols[0]}
            record["y"] = record["y"] or {"col": cols[1] if len(cols) > 1 else cols[0]}
        self._dataset_selected()
        self._refresh_rows_text_only()

    def _name_edited(self) -> None:
        if self._loading:
            return
        row = self._current_row()
        if row is None:
            return
        record = self.datasets[row]
        text = self.name_edit.text().strip()
        record["auto_label"] = not text or text == self._auto_label(record)
        record["label"] = text
        self.name_edit.setText(self._label(record))
        self._refresh_rows_text_only()

    def _pick_colour(self) -> None:
        row = self._current_row()
        if row is None:
            return
        record = self.datasets[row]
        chosen = QColorDialog.getColor(QColor(self._colour_text(record["colour"]) if record["colour"]
                                              else "#1f77b4"), self, "Dataset colour")
        if chosen.isValid():
            record["colour"] = chosen.name()
            self.colour_button.setText(record["colour"])
            self._refresh_row_text(row)

    def _sync(self, *_a) -> None:
        three = self.dim_combo.currentData() == "3d"
        holder, label = self._axis_rows["z"]
        holder.setVisible(three)
        self._editor_form.labelForField(holder).setVisible(three)
        self.zlabel_edit.setVisible(three)
        self.zlabel_row_label.setVisible(three)
        for w in (self.yerr_combo,):
            w.setEnabled(not three and self._current_row() is not None)
            self._editor_form.labelForField(w).setEnabled(not three)
        if hasattr(self, "dataset_tree"):
            self._refresh_rows_text_only()

    def _refresh_rows_text_only(self) -> None:
        for row in range(len(self.datasets)):
            self._refresh_row_text(row)

    # ------------------------------------------------------------------ resolving to numbers
    def _resolve(self, record: dict, key: str, index: int):
        """The numbers of one axis of a dataset, or ``None`` when it has none (z of a 2-D)."""
        ref = record[key]
        if ref is None:
            return None
        if "col" in ref:
            table = self.sources[record["source"]]["table"]
            return table[ref["col"]].to_numpy(dtype=float)
        if "list" in ref:
            return ref["list"]
        n = len(self.sources[record["source"]]["table"]) if record["source"] is not None else 0
        return list(range(1, n + 1))

    @staticmethod
    def _scales(record: dict, who: str) -> dict:
        """``{"x": .., "y": .., "z": ..}`` from the Multiply by boxes; ValueError saying which."""
        out = {}
        for axis in ("x", "y", "z"):
            text = str(record.get(f"{axis}scale", "1")).strip() or "1"
            try:
                out[axis] = float(text.replace(",", "."))
            except ValueError:
                raise ValueError(f"{who}: the {axis} multiplier {text!r} is not a number.") from None
            if not np.isfinite(out[axis]):
                raise ValueError(f"{who}: the {axis} multiplier must be a finite number.")
        return out

    def _safe_scales(self, record: dict) -> dict:
        """The multipliers for the notebook code: 1 for one that is not a number (the plot says so)."""
        try:
            return self._scales(record, "dataset")
        except ValueError:
            return {"x": 1.0, "y": 1.0, "z": 1.0}

    def build_datasets(self) -> list[dict]:
        """The datasets as :func:`opennmr.notebook.other_plot.dataset` dicts. Raises ValueError,
        saying which dataset and what is missing or mismatched."""
        three = self.dim_combo.currentData() == "3d"
        out = []
        for i, record in enumerate(self.datasets):
            who = f"Dataset {i + 1} ({self._label(record)})"
            if record["fit"] is not None:
                if three:
                    raise ValueError(f"{who} is a curve fit, which is 2-D: switch the dimensions to 2-D "
                                     "or remove it.")
                s = self._scales(record, who)
                ds = op.rescaled(record["fit"]["ds"], x=s["x"], y=s["y"])
                ds["label"] = self._label(record)
                if record["kind"]:
                    ds["kind"] = record["kind"]
                if record["colour"]:
                    ds["colour"] = record["colour"]
                out.append(ds)
                continue
            if record["x"] is None or record["y"] is None:
                raise ValueError(f"{who} needs an x and a y: choose a column, the row number, or type a list.")
            if three and record["z"] is None:
                raise ValueError(f"{who} needs a z for a 3-D plot.")
            scales = self._scales(record, who)
            try:
                x = self._resolve(record, "x", i)
                y = self._resolve(record, "y", i)
                z = self._resolve(record, "z", i) if three else None
                err = self._resolve(record, "yerr", i) if not three else None
                cby = self._resolve(record, "colour_by", i)
                if "rows" in (record["x"] or {}) and record["source"] is None:
                    raise ValueError("the row number needs a file to count the rows of")
                names = {k: self._ref_text(record[k]) if record[k] and "col" in record[k] else ""
                         for k in ("x", "y", "z")}
                out.append(op.dataset(x, y, z, label=self._label(record), kind=record["kind"],
                                      colour=record["colour"], yerr=err, xname=names["x"],
                                      yname=names["y"], zname=names["z"], colour_by=cby,
                                      xscale=scales["x"], yscale=scales["y"], zscale=scales["z"],
                                      colour_name=self._ref_text(record["colour_by"])
                                      if record["colour_by"] and "col" in record["colour_by"] else ""))
            except ValueError as exc:
                raise ValueError(f"{who}: {exc}") from None
        return out

    # ------------------------------------------------------------------ plot
    def _kwargs(self) -> dict:
        """The drawing options: Plot Style's data-plot settings plus this panel's labels."""
        args = dict(self.DEFAULT_STYLE)
        args.update({k: v for k, v in (self._style() or {}).items() if v is not None})
        args["kind"] = self.default_kind_combo.currentData()
        for key, edit in (("title", self.title_edit), ("xlabel", self.xlabel_edit),
                          ("ylabel", self.ylabel_edit), ("zlabel", self.zlabel_edit)):
            if edit.text().strip():
                args[key] = edit.text().strip()
        return args

    def plot(self) -> None:
        try:
            if not self.datasets and self.sources:
                self.add_dataset()
            if not self.datasets:
                raise ValueError("There is nothing to plot yet: add a data file, then a dataset.")
            sets = self.build_datasets()
            kwargs = self._kwargs()
            from matplotlib.figure import Figure
            op.plot_datasets(sets, figure=Figure(), fontsize=self._fontsize(), **kwargs)
        except ValueError as exc:       # said here; the plot on screen is kept
            self._fail(str(exc))
            return
        self._show(sets, kwargs)
        self._drawn = True
        self._fail(f"Plotted {len(sets)} dataset(s).")

    def _show(self, sets, kwargs) -> None:
        def draw(figure, kwargs=dict(kwargs), sets=sets):
            op.plot_datasets([dict(d) for d in sets], figure=figure, fontsize=self._fontsize(), **kwargs)
        self._draw(draw)

    def replot(self) -> None:
        """Plot Style changed a data-plot option: redraw what was last plotted, if anything."""
        if self._drawn and self.datasets:
            try:
                self._show(self.build_datasets(), self._kwargs())
            except ValueError:
                pass

    # ------------------------------------------------------------------ notebook code
    def _expr(self, record: dict, key: str, var: dict) -> str:
        ref = record[key]
        if ref is None:
            return "None"
        if "col" in ref:
            return f"{var[record['source']]}[{ref['col']!r}]"
        if "list" in ref:
            return "[" + ", ".join(f"{v:g}" for v in ref["list"]) + "]"
        return f"list(range(1, len({var[record['source']]}) + 1))"

    def notebook_code(self) -> str:
        lines = ["from opennmr.notebook import other_plot as op", ""]
        var = {}
        for i, source in enumerate(self.sources):
            if not any(r["source"] == i for r in self.datasets):
                continue
            var[i] = f"table{len(var) + 1}"
            skip = f", skip_lines={source['skip']}" if source["skip"] else ""
            if source["path"]:
                lines.append(f"{var[i]} = op.read_table({source['path']!r}{skip})")
            else:
                lines.append(f"{var[i]} = op.read_table({source['text']!r}{skip})  # the text you pasted")
        three = self.dim_combo.currentData() == "3d"
        lines.append("datasets = [")
        for r in self.datasets:
            if r["fit"] is not None:
                if r["fit"]["first"]:
                    lines.append(f"    # the curve fit (points and curve) of the open experiment "
                                 f"{r['fit']['experiment']!r}")
                    lines.append(f"    *op.curve_fit_datasets(spectra[{r['fit']['row']}], "
                                 f"label={r['fit']['experiment']!r}),")
                continue
            args = [self._expr(r, "x", var), self._expr(r, "y", var)]
            if three:
                args.append(self._expr(r, "z", var))
            extra = [f"label={self._label(r)!r}"]
            if r["kind"]:
                extra.append(f"kind={r['kind']!r}")
            if r["colour"]:
                extra.append(f"colour={r['colour']!r}")
            if r["yerr"] and not three:
                extra.append(f"yerr={self._expr(r, 'yerr', var)}")
            if r["colour_by"]:
                extra.append(f"colour_by={self._expr(r, 'colour_by', var)}")
            for k, ref in (("xname", r["x"]), ("yname", r["y"])):
                if ref and "col" in ref:
                    extra.append(f"{k}={ref['col']!r}")
            for axis in ("x", "y", "z") if three else ("x", "y"):
                value = self._safe_scales(r)[axis]
                if value != 1.0:
                    extra.append(f"{axis}scale={value:g}")
            lines.append(f"    op.dataset({', '.join(args + extra)}),")
        lines.append("]")
        for k, r in enumerate(self.datasets):
            if r["fit"] is not None:
                s = self._safe_scales(r)
                if s["x"] != 1.0 or s["y"] != 1.0:
                    lines.append(f"datasets[{k}] = op.rescaled(datasets[{k}], x={s['x']:g}, y={s['y']:g})")
        try:
            kwargs = self._kwargs()
        except ValueError:
            kwargs = {}
        shown = ", ".join(f"{k}={v!r}" for k, v in kwargs.items())
        lines.append(f"fig = op.plot_datasets(datasets{', ' + shown if shown else ''})")
        if any(r["fit"] is not None for r in self.datasets):
            lines.append("# the curve fits come from open experiments: spectra = your list of loaded spectra")
        return "\n".join(lines)

    def copy_code(self) -> None:
        QGuiApplication.clipboard().setText(self.notebook_code())
        self._fail("Notebook code copied.")

    def _fail(self, text: str) -> None:
        self.report_label.setText(text)
