"""
Apply ▸ Plot other data: any table of numbers as a 2-D or 3-D scatter, line or bar chart.

Load a csv/tsv/txt file, or paste columns copied from a spreadsheet, choose the columns for x, y
(and z for a 3-D figure) and how to draw them. The maths-free drawing is
:func:`opennmr.notebook.other_plot.plot_other_data`; *Copy notebook code* reproduces the figure.
"""

from __future__ import annotations

import pandas as pd
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QListWidget, QPlainTextEdit, QPushButton, QVBoxLayout, QWidget,
)

from ..notebook import other_plot as op
from .widgets.report_label import ReportLabel


class OtherPlotPanel(QWidget):
    """``draw(draw)`` puts a figure-drawing function on the Compare canvas (from MainWindow);
    ``fontsize()`` -> the figure font size set in Plot Style."""

    def __init__(self, draw, fontsize=lambda: 12.0, parent=None) -> None:
        super().__init__(parent)
        self._draw, self._fontsize = draw, fontsize
        self.table: pd.DataFrame | None = None
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        intro = QLabel("Plot any columns of numbers as a scatter, line or bar chart, in two or three "
                       "dimensions. Load a csv/tsv/txt file with a header row, or paste columns "
                       "copied from a spreadsheet.")
        intro.setWordWrap(True)
        v.addWidget(intro)

        box = QGroupBox("1. The data")
        form = QFormLayout(box)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("csv / tsv / txt file")
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        row.addWidget(self.path_edit, 1)
        row.addWidget(browse)
        holder = QWidget()
        holder.setLayout(row)
        form.addRow("File:", holder)
        self.paste_edit = QPlainTextEdit()
        self.paste_edit.setPlaceholderText("or paste columns here (with a header row)")
        self.paste_edit.setMaximumHeight(90)
        form.addRow("Paste:", self.paste_edit)
        load = QPushButton("Read data")
        load.clicked.connect(self.load)
        form.addRow(load)
        self.path_edit.editingFinished.connect(lambda: self.path_edit.text().strip() and self.load())
        v.addWidget(box)

        box = QGroupBox("2. What to draw")
        form = QFormLayout(box)
        self.dim_combo = QComboBox()
        self.dim_combo.addItem("2-D (y against x)", "2d")
        self.dim_combo.addItem("3-D (z against x and y)", "3d")
        self.dim_combo.currentIndexChanged.connect(self._sync)
        form.addRow("Dimensions:", self.dim_combo)
        self.kind_combo = QComboBox()
        for key in op.KINDS:
            self.kind_combo.addItem(op.KIND_NAMES[key], key)
        form.addRow("Draw as:", self.kind_combo)
        self.x_combo, self.y_combo, self.z_combo, self.colour_combo = (
            QComboBox(), QComboBox(), QComboBox(), QComboBox())
        form.addRow("x:", self.x_combo)
        self.y_list = QListWidget()
        self.y_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.y_list.setMaximumHeight(90)
        self.y_list.setToolTip("One or more columns; each is its own series.")
        self.y_label = QLabel("y:")
        form.addRow(self.y_label, self.y_list)
        form.addRow("y (3-D):", self.y_combo)
        form.addRow("z (3-D):", self.z_combo)
        form.addRow("Colour points by:", self.colour_combo)
        self.colour_combo.setToolTip("A column that colours the points of a scatter (with a colour bar).")
        self._form = form
        v.addWidget(box)

        box = QGroupBox("3. Labels and look")
        form = QFormLayout(box)
        self.title_edit, self.xlabel_edit, self.ylabel_edit, self.zlabel_edit = (
            QLineEdit(), QLineEdit(), QLineEdit(), QLineEdit())
        for w in (self.title_edit, self.xlabel_edit, self.ylabel_edit, self.zlabel_edit):
            w.setPlaceholderText("blank = the column name" if w is not self.title_edit else "blank = none")
        form.addRow("Title:", self.title_edit)
        form.addRow("x label:", self.xlabel_edit)
        form.addRow("y label:", self.ylabel_edit)
        form.addRow("z label:", self.zlabel_edit)
        self.marker_size = QDoubleSpinBox()
        self.marker_size.setRange(1.0, 40.0)
        self.marker_size.setValue(6.0)
        self.line_width = QDoubleSpinBox()
        self.line_width.setRange(0.2, 10.0)
        self.line_width.setValue(1.5)
        self.colormap_combo = QComboBox()
        for name in ("viridis", "plasma", "inferno", "magma", "cividis", "coolwarm", "turbo"):
            self.colormap_combo.addItem(name, name)
        self.grid_check = QCheckBox("Grid")
        form.addRow("Marker size:", self.marker_size)
        form.addRow("Line width:", self.line_width)
        form.addRow("Colour map:", self.colormap_combo)
        form.addRow(self.grid_check)
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

    # ------------------------------------------------------------------ data
    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Data file", self.path_edit.text(),
                                              "Tables (*.csv *.tsv *.txt *.dat);;All files (*)")
        if path:
            self.path_edit.setText(path)
            self.load()

    def load(self) -> bool:
        """Read the pasted text if there is any, otherwise the file; fill the column choices."""
        pasted = self.paste_edit.toPlainText().strip()
        try:
            self.table = op.read_table(pasted + "\n" if pasted else self.path_edit.text().strip())
        except ValueError as exc:
            self.table = None
            self._fail(str(exc))
            return False
        self._fill_columns()
        self._fail(f"Read {len(self.table)} rows, {self.table.shape[1]} columns.")
        return True

    def _fill_columns(self) -> None:
        names = list(self.table.columns)
        for combo in (self.x_combo, self.y_combo, self.z_combo, self.colour_combo):
            combo.clear()
        self.colour_combo.addItem("(one colour)", None)
        for n in names:
            for combo in (self.x_combo, self.y_combo, self.z_combo, self.colour_combo):
                combo.addItem(n, n)
        self.y_list.clear()
        for n in names:
            self.y_list.addItem(n)
        if len(names) > 1:
            self.y_combo.setCurrentIndex(1)
            self.y_list.item(1).setSelected(True)
        if len(names) > 2:
            self.z_combo.setCurrentIndex(2)

    def _sync(self, *_a) -> None:
        three = self.dim_combo.currentData() == "3d"
        for w in (self.y_list, self.y_label):
            w.setVisible(not three)
        for w in (self.y_combo, self.z_combo, self.zlabel_edit):
            w.setVisible(three)
            label = self._form.labelForField(w)
            if label is not None:
                label.setVisible(three)

    # ------------------------------------------------------------------ plot
    def _kwargs(self) -> dict:
        if self.table is None and not self.load():
            raise ValueError("There is no data to plot yet.")
        three = self.dim_combo.currentData() == "3d"
        if three:
            args = dict(x=self.x_combo.currentData(), y=self.y_combo.currentData(),
                        z=self.z_combo.currentData())
        else:
            ys = [i.text() for i in self.y_list.selectedItems()]
            if not ys:
                raise ValueError("Choose at least one column for y.")
            args = dict(x=self.x_combo.currentData(), y=ys if len(ys) > 1 else ys[0])
        if None in args.values():
            raise ValueError("Choose the columns to plot.")
        args.update(kind=self.kind_combo.currentData(), colour_by=self.colour_combo.currentData(),
                    colormap=self.colormap_combo.currentData(), marker_size=self.marker_size.value(),
                    linewidth=self.line_width.value(), grid=self.grid_check.isChecked())
        for key, edit in (("title", self.title_edit), ("xlabel", self.xlabel_edit),
                          ("ylabel", self.ylabel_edit), ("zlabel", self.zlabel_edit)):
            if edit.text().strip() and (key != "zlabel" or three):
                args[key] = edit.text().strip()
        return args

    def plot(self) -> None:
        try:
            kwargs = self._kwargs()
            from matplotlib.figure import Figure
            op.plot_other_data(self.table, figure=Figure(), fontsize=self._fontsize(), **kwargs)
        except ValueError as exc:       # said here; the plot on screen is kept
            self._fail(str(exc))
            return
        table = self.table

        def draw(figure, kwargs=dict(kwargs)):
            op.plot_other_data(table, figure=figure, fontsize=self._fontsize(), **kwargs)
        self._draw(draw)
        self._fail("Plotted.")

    def notebook_code(self) -> str:
        src = self.path_edit.text().strip()
        try:
            kwargs = self._kwargs()
        except ValueError:
            kwargs = {}
        args = ", ".join(f"{k}={v!r}" for k, v in kwargs.items())
        return "\n".join(["from opennmr.notebook import other_plot as op", "",
                          f"table = op.read_table({src!r})" if src and not self.paste_edit.toPlainText().strip()
                          else "table = op.read_table(pasted_text)  # the text you pasted",
                          f"fig = op.plot_other_data(table, {args})"])

    def copy_code(self) -> None:
        QGuiApplication.clipboard().setText(self.notebook_code())
        self._fail("Notebook code copied.")

    def _fail(self, text: str) -> None:
        self.report_label.setText(text)
