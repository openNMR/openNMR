"""
Apply ▸ Electrochemistry: the cell beside the spectra.

Load a cycler file (Biologic .mpr/.mpt, or a plain csv/txt table), enter the sample mass, area
and formula, and put the cell curve beside either

* **operando** spectra (a series recorded while the cell cycled): choose the experiments
  (the rows selected in Open experiments, or a start, stop and step of experiment
  numbers), and say when the first began on the cycler's clock and how long each took.
  The two instruments' clocks are not compared, so the order and timing are the user's;
* **ex situ** spectra: one value per sample (usually its voltage), the cycle and the
  direction it was stopped in, so the point on the curve is found for you; or
* **fitted values** from a csv/txt (a shift, an area, a linewidth per spectrum or time).

The cell can sit left, right, above or below and be drawn against any column of the file.
Spectra are stacked, superimposed or a heat map, optionally as differences. Style (tick
labels, axis names, colours, markers) is in Plot Style ▸ Electrochemistry. The maths and
the figure are :mod:`opennmr.notebook.echem` and :mod:`opennmr.notebook.echem_plot`;
*Copy notebook code* reproduces the figure.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor, QGuiApplication
from PySide6.QtWidgets import (
    QCheckBox, QColorDialog, QComboBox, QDoubleSpinBox, QMenu, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from ..notebook import echem, echem_plot as ep
from .widgets.report_label import ReportLabel
from .widgets.typed_spinbox import TypedSpinBox

_MODES = (("operando", "Operando (a series recorded during cycling)"),
          ("ex_situ", "Ex situ (one value per sample)"),
          ("values", "Fitted values (from a csv/txt file)"))
_COLUMNS = ("Sample", "Value", "Cycle", "Half", "Which crossing", "Mass (mg)", "× factor")
_UNITS = (("s", 1.0), ("min", 60.0), ("h", 3600.0))
_DIFFS = (("", "No differences"), ("successive", "Each minus the one before"),
          ("from_first", "Each minus the first"), ("from_last", "Each minus the last"))


def _plain(label: str) -> str:
    """A mathtext axis label as plain text for a list: ``g$^{\\mathregular{-1}}$`` -> ``g⁻¹``."""
    import re
    sup = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")
    label = re.sub(r"\$\^\{\\mathregular\{([^}]*)\}\}\$", lambda m: m.group(1).translate(sup), label)
    return label.replace("$", "")


def _number(text):
    text = str(text).strip().replace("−", "-")
    if not text:
        return None
    try:
        v = float(text)
    except ValueError:
        return None
    return v if math.isfinite(v) else None


class EchemPanel(QWidget):
    """Callbacks (from MainWindow): ``spectra()`` -> ``[{"label", "spectrum", "index", "path",
    "selected"}]`` (the 1D spectra); ``draw(draw, hz_per_ppm=None)`` puts a figure-drawing function
    on a canvas (``hz_per_ppm``: the spectrometer frequency, MHz, so Measure can quote Hz);
    ``style()`` -> the keyword arguments of the Plot Style ▸ Electrochemistry section."""

    def __init__(self, spectra, draw, style, parent=None) -> None:
        super().__init__(parent)
        self._spectra, self._draw, self._style = spectra, draw, style
        self.cell: dict | None = None
        self.values: pd.DataFrame | None = None
        self.link: pd.DataFrame | None = None
        self.plotted = False
        self._entries: list[dict] = []

        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        intro = QLabel(
            "Put an electrochemical curve beside spectra recorded during cycling (operando), "
            "stopped at chosen voltages (ex situ) or beside fitted values. Positive current is "
            "charge. Fonts, colours, tick labels and markers are in Plot Style ▸ Electrochemistry.")
        intro.setWordWrap(True)
        v.addWidget(intro)
        v.addWidget(self._build_file())
        v.addWidget(self._build_source())
        v.addWidget(self._build_layout())
        row = QHBoxLayout()
        self.plot_button = QPushButton("Plot")
        self.plot_button.clicked.connect(self.plot)
        self.export_button = QPushButton("Export Link Table…")
        self.export_button.clicked.connect(self.export_csv)
        self.copy_button = QPushButton("Copy Notebook Code")
        self.copy_button.clicked.connect(self.copy_code)
        for b in (self.plot_button, self.export_button, self.copy_button):
            row.addWidget(b)
        row.addStretch(1)
        v.addLayout(row)
        self.report_label = ReportLabel("")
        v.addWidget(self.report_label)
        self.refresh()
        self._sync()

    # ------------------------------------------------------------------ build
    def _build_file(self) -> QGroupBox:
        box = QGroupBox("1. Electrochemistry file")
        form = QFormLayout(box)
        row = QHBoxLayout()
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText(".mpr, .mpt, .res, .nda(x), .xlsx, .csv … (.mpr/.mpt and tables need nothing extra)")
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        row.addWidget(self.path_edit, 1)
        row.addWidget(browse)
        form.addRow("File:", row)
        self.mass_edit = QLineEdit()
        self.mass_edit.setPlaceholderText("active mass (mg)")
        self.area_edit = QLineEdit()
        self.area_edit.setPlaceholderText("electrode area (cm²), optional")
        self.formula_edit = QLineEdit()
        self.formula_edit.setPlaceholderText("e.g. NaMnO2 (for the composition)")
        self.molar_edit = QLineEdit()
        self.molar_edit.setPlaceholderText("g/mol, or leave blank to use the formula")
        self.x0_edit = QLineEdit("1")
        self.x0_edit.setToolTip("The composition x at the start of the file.")
        self.removes_check = QCheckBox("Charging removes the ion (x falls on charge)")
        self.removes_check.setChecked(True)
        form.addRow("Mass (mg):", self.mass_edit)
        form.addRow("Area (cm²):", self.area_edit)
        form.addRow("Formula:", self.formula_edit)
        form.addRow("Molar mass:", self.molar_edit)
        form.addRow("Starting x:", self.x0_edit)
        form.addRow("", self.removes_check)
        self.load_button = QPushButton("Load")
        self.load_button.clicked.connect(self.load)
        form.addRow("", self.load_button)
        self.info_label = QLabel("No file loaded.")
        self.info_label.setWordWrap(True)
        form.addRow(self.info_label)
        return box

    def _build_source(self) -> QGroupBox:
        box = QGroupBox("2. What goes beside it")
        form = QFormLayout(box)
        self._source_form = form
        self.mode_combo = QComboBox()
        for key, text in _MODES:
            self.mode_combo.addItem(text, key)
        self.mode_combo.currentIndexChanged.connect(self._sync)
        form.addRow("Beside the cell:", self.mode_combo)

        self.select_combo = QComboBox()
        self.select_combo.addItem("The rows selected in Open experiments", "selected")
        self.select_combo.addItem("Experiment numbers: start, stop, step", "range")
        self.select_combo.currentIndexChanged.connect(self._sync)
        self.select_combo.setToolTip(
            "A long series is a range of experiment numbers (matched to the number of each "
            "experiment's folder) rather than a long list. The order is yours to get right.")
        form.addRow("Spectra:", self.select_combo)
        self.exp_start, self.exp_stop, self.exp_step = TypedSpinBox(), TypedSpinBox(), TypedSpinBox()
        for s, val in ((self.exp_start, 1), (self.exp_stop, 10), (self.exp_step, 1)):
            s.setRange(-99999, 99999)
            s.setValue(val)
            s.valueChanged.connect(self._sync)
        range_row = QHBoxLayout()
        for label, w in (("start", self.exp_start), ("stop", self.exp_stop), ("step", self.exp_step)):
            range_row.addWidget(QLabel(label))
            range_row.addWidget(w)
        self.range_widget = QWidget()
        self.range_widget.setLayout(range_row)
        range_row.setContentsMargins(0, 0, 0, 0)
        form.addRow("Experiments:", self.range_widget)
        self.chosen_label = QLabel("")
        self.chosen_label.setWordWrap(True)
        form.addRow(self.chosen_label)

        # operando timing
        self.time_unit_combo = QComboBox()
        for text, _ in _UNITS:
            self.time_unit_combo.addItem(text, text)
        self.time_unit_combo.setCurrentText("h")
        self.start_edit = QLineEdit("0")
        self.start_edit.setToolTip("When the first spectrum began, on the cycler's time axis.")
        self.per_edit = QLineEdit()
        self.per_edit.setPlaceholderText("time per spectrum")
        self.gap_edit = QLineEdit("0")
        self.starts_edit = QLineEdit()
        self.starts_edit.setPlaceholderText("or: the start time of every spectrum, e.g. 0.5, 1.5, 2.5")
        form.addRow("Time unit:", self.time_unit_combo)
        form.addRow("First starts at:", self.start_edit)
        form.addRow("Each takes:", self.per_edit)
        form.addRow("Dead time between:", self.gap_edit)
        form.addRow("Start times:", self.starts_edit)
        self._timing_rows = ["Time unit:", "First starts at:", "Each takes:", "Dead time between:",
                             "Start times:"]

        # ex situ
        self.quantity_combo = QComboBox()
        self.quantity_combo.setToolTip("The quantity each sample's value is of.")
        form.addRow("Values are:", self.quantity_combo)
        self.ex_table = QTableWidget(0, len(_COLUMNS))
        self.ex_table.setHorizontalHeaderLabels(list(_COLUMNS))
        self.ex_table.setMinimumHeight(150)
        self.ex_table.setToolTip(
            "One row per spectrum. Value, cycle, half and crossing place an ex situ sample on the "
            "curve. Mass (mg) divides the spectrum by it when 'by sample mass' is ticked. "
            "× multiplies the spectrum by a factor. The colour of each spectrum and point is in "
            "Plot Style, in the series list.")
        form.addRow(self.ex_table)

        # values
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        self.values_edit = QLineEdit()
        self.values_edit.setPlaceholderText("csv/txt with a header row")
        vb = QPushButton("Browse…")
        vb.clicked.connect(self._browse_values)
        row.addWidget(self.values_edit, 1)
        row.addWidget(vb)
        self.values_row = QWidget()
        self.values_row.setLayout(row)
        form.addRow("Values file:", self.values_row)
        self.value_x_combo, self.value_x_type = QComboBox(), QComboBox()
        self.value_x_type.addItem("Time (h)", "time_h")
        self.value_x_type.addItem("Time (s)", "time_s")
        self.value_x_type.addItem("Spectrum number (uses the timing above)", "index")
        self.value_cols_edit = QLineEdit()
        self.value_cols_edit.setPlaceholderText("columns to plot, comma separated (blank = the first)")
        form.addRow("x column:", self.value_x_combo)
        form.addRow("x is:", self.value_x_type)
        form.addRow("Plot:", self.value_cols_edit)
        self.values_edit.editingFinished.connect(self._read_values)
        return box

    def _build_layout(self) -> QGroupBox:
        box = QGroupBox("3. Layout")
        form = QFormLayout(box)
        self.side_combo = QComboBox()
        for s in ep.SIDES:
            self.side_combo.addItem(f"Cell on the {s}" if s in ("left", "right") else
                                    f"Cell {'above' if s == 'top' else 'below'} the spectra", s)
        self.side_combo.setCurrentIndex(1)
        self.display_combo = QComboBox()
        for key, text in (("stacked", "Stacked"), ("superimposed", "Superimposed"),
                          ("heatmap", "Heat map")):
            self.display_combo.addItem(text, key)
        self.diff_combo = QComboBox()
        for key, text in _DIFFS:
            self.diff_combo.addItem(text, key)
        self.x_combo, self.y_combo, self.position_combo = QComboBox(), QComboBox(), QComboBox()
        self.position_combo.setToolTip(
            "What places each spectrum: its number, or a cell quantity (time, voltage, "
            "composition). When it is also the cell's other axis the two share it.")
        self.cycles_edit = QLineEdit()
        self.cycles_edit.setPlaceholderText("blank = all, or e.g. 1 or 1, 2, 5")
        self.half_combo = QComboBox()
        for key in ("both", "charge", "discharge"):
            self.half_combo.addItem(key.capitalize(), key)
        self.distribute_combo = QComboBox()
        self.distribute_combo.addItem("By its position on the cell", "position")
        self.distribute_combo.addItem("Evenly spaced, in the order listed", "even")
        self.distribute_combo.addItem("Spaced by each one's maximum intensity", "intensity")
        self.distribute_combo.setToolTip(
            "Stacked and heat map: spectra at their cell position (they then share that axis with "
            "the cell), or each given the same room, labelled by name. Evenly spaced suits ex situ "
            "samples, whose positions are not evenly spread.")
        self.per_scan_check = QCheckBox("Divide by the number of scans")
        self.divide_rg_check = QCheckBox("Divide by the receiver gain")
        self.by_mass_check = QCheckBox("Divide by sample mass (the Mass column of the table)")
        self.auto_colour_check = QCheckBox("Automatic colours from the colour map")
        self.auto_colour_check.setChecked(True)
        self.link_colour_check = QCheckBox("A spectrum and its point share one colour setting (Plot Style series list)")
        self.link_colour_check.setChecked(True)
        self.link_colour_check.toggled.connect(self._sync)
        self.points_check = QCheckBox("Mark each spectrum on the curve")
        self.points_check.setChecked(True)
        self.label_check = QCheckBox("Number the marks")
        self.cycle_colour_check = QCheckBox("Colour the curve by cycle")
        for label, w in (("Placement:", self.side_combo), ("Spectra shown:", self.display_combo),
                         ("Differences:", self.diff_combo), ("Cell x:", self.x_combo),
                         ("Cell y:", self.y_combo), ("Spectra placed by:", self.position_combo),
                         ("Cycles shown:", self.cycles_edit), ("Half cycle:", self.half_combo),
                         ("Spacing:", self.distribute_combo)):
            form.addRow(label, w)
        for option in (self.per_scan_check, self.divide_rg_check, self.by_mass_check,
                       self.auto_colour_check, self.link_colour_check):
            form.addRow("", option)
        form.addRow("", self.points_check)
        form.addRow("", self.label_check)
        form.addRow("", self.cycle_colour_check)
        return box

    # ------------------------------------------------------------- state
    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Electrochemistry file", self.path_edit.text())
        if path:
            self.path_edit.setText(path)
            self.load()

    def _browse_values(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Values file", self.values_edit.text(),
                                              "Tables (*.csv *.tsv *.txt *.dat)")
        if path:
            self.values_edit.setText(path)
            self._read_values()

    def refresh(self) -> None:
        """Re-read the spectra of Open experiments."""
        self._entries = list(self._spectra())
        self._sync()

    def _mode(self) -> str:
        return self.mode_combo.currentData()

    def _sync(self, *_args) -> None:
        mode = self._mode()
        form = self._source_form
        spectra = mode in ("operando", "ex_situ")
        ranged = spectra and self.select_combo.currentData() == "range"
        def show(widget, on):
            widget.setVisible(bool(on))
            label = form.labelForField(widget)
            if label is not None:
                label.setVisible(bool(on))
        show(self.select_combo, spectra)
        show(self.range_widget, ranged)
        self.chosen_label.setVisible(spectra)
        for w in (self.time_unit_combo, self.start_edit, self.per_edit, self.gap_edit, self.starts_edit):
            show(w, mode == "operando" or mode == "values")
        show(self.quantity_combo, mode == "ex_situ")
        self.ex_table.setVisible(spectra)
        for c in range(1, 5):
            self.ex_table.setColumnHidden(c, mode != "ex_situ")
        for w in (self.values_row, self.value_x_combo, self.value_x_type, self.value_cols_edit):
            show(w, mode == "values")
        chosen, notes = self._chosen() if spectra else ([], [])
        if spectra:
            names = ", ".join(e["label"] for e in chosen[:4]) + (" …" if len(chosen) > 4 else "")
            self.chosen_label.setText(f"{len(chosen)} spectra: {names}" if chosen else
                                      "No spectra chosen." + (" " + " ".join(notes) if notes else ""))
        if spectra:
            self._sync_table(chosen)
        self.position_combo.setEnabled(True)

    def _chosen(self) -> tuple[list[dict], list[str]]:
        entries, notes = self._entries, []
        if self.select_combo.currentData() == "range":
            try:
                numbers = ep.experiment_numbers(self.exp_start.value(), self.exp_stop.value(),
                                                self.exp_step.value())
            except ValueError as exc:
                return [], [str(exc)]
            by_number = {}
            for e in entries:
                n = ep.experiment_number(e.get("path"))
                if n is not None:
                    by_number.setdefault(n, e)
            chosen = [by_number[n] for n in numbers if n in by_number]
            missing = [n for n in numbers if n not in by_number]
            if missing:
                notes.append("Not open: " + ", ".join(map(str, missing[:8])) + (" …" if len(missing) > 8 else "") + ".")
            return chosen, notes
        return [e for e in entries if e.get("selected")], notes

    def _sync_table(self, chosen) -> None:
        """One row per chosen spectrum; what was typed stays with its spectrum (by name). The mass
        column starts as the sample mass set by right-clicking the experiment (a value typed here
        is kept and wins)."""
        t = self.ex_table
        keep = {}
        for r in range(t.rowCount()):
            texts = [t.item(r, c).text() if t.item(r, c) else "" for c in range(len(_COLUMNS))]
            keep[texts[0]] = texts
        auto = getattr(self, "_auto_mass", {})
        t.blockSignals(True)
        t.setRowCount(len(chosen))
        for r, entry in enumerate(chosen):
            old = keep.get(entry["label"], [""] * len(_COLUMNS))
            mass = getattr(entry["spectrum"].metadata, "mass", None)
            from_menu = f"{mass * 1000.0:g}" if mass else ""
            mass_text = from_menu if old[5] in ("", auto.get(entry["label"], "")) else old[5]
            auto[entry["label"]] = from_menu
            texts = [entry["label"], old[1], old[2] or "1", old[3] or "charge", old[4] or "first",
                     mass_text, old[6]]
            for c, text in enumerate(texts):
                item = QTableWidgetItem(text)
                if c == 0:
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                t.setItem(r, c, item)
        self._auto_mass = auto
        t.blockSignals(False)

    def _fill_columns(self) -> None:
        data = self.cell["data"]
        columns = echem.plottable_columns(data)
        for combo, default in ((self.x_combo, "voltage_V"), (self.y_combo, "time_h")):
            keep = combo.currentData()
            combo.blockSignals(True)
            combo.clear()
            for c in columns:
                combo.addItem(_plain(echem.axis_label(c)), c)
            combo.setCurrentIndex(max(0, combo.findData(keep if keep in columns else default)))
            combo.blockSignals(False)
        keep = self.position_combo.currentData()
        self.position_combo.clear()
        self.position_combo.addItem("Spectrum number", "index")
        for c in columns:
            self.position_combo.addItem(_plain(echem.axis_label(c)), c)
        self.position_combo.setCurrentIndex(max(0, self.position_combo.findData(keep or "time_h")))
        self.quantity_combo.clear()
        for c in columns:
            self.quantity_combo.addItem(_plain(echem.axis_label(c)), c)
        self.quantity_combo.setCurrentIndex(max(0, self.quantity_combo.findData("voltage_V")))

    # ------------------------------------------------------------- actions
    def _fail(self, text: str) -> None:
        self.report_label.setText(text)

    def load(self) -> bool:
        path = self.path_edit.text().strip().strip('"')
        if not path:
            self._fail("Choose an electrochemistry file first.")
            return False
        try:
            self.cell = echem.load_echem(
                path, mass_mg=_number(self.mass_edit.text()), area_cm2=_number(self.area_edit.text()),
                formula=self.formula_edit.text().strip() or None,
                molar_mass=_number(self.molar_edit.text()), x0=_number(self.x0_edit.text()) or 1.0,
                charge_removes_ion=self.removes_check.isChecked())
        except Exception as exc:    # noqa: BLE001 -- shown to the user
            self.cell = None
            exc = str(exc) or type(exc).__name__
            self.info_label.setText(f"Not loaded: {exc}")
            self._fail(f"Could not read the file: {exc}")
            return False
        data = self.cell["data"]
        cycles = int(data["cycle"].max()) if "cycle" in data else 0
        self.info_label.setText(
            f"{len(data)} points, {cycles} cycle(s), read with {self.cell['reader']}. "
            f"Columns: {', '.join(echem.plottable_columns(data))}.")
        self._fill_columns()
        self._fail("\n".join(self.cell.get("notes", [])))
        return True

    def _read_values(self) -> None:
        path = self.values_edit.text().strip().strip('"')
        if not path:
            return
        try:
            self.values = echem.read_value_table(path)
        except Exception as exc:    # noqa: BLE001
            self.values = None
            self._fail(f"Could not read the values file: {exc}")
            return
        self.value_x_combo.clear()
        for c in self.values.columns:
            self.value_x_combo.addItem(str(c), c)

    def _timing(self, n: int):
        unit = dict(_UNITS)[self.time_unit_combo.currentData()]
        text = self.starts_edit.text().replace(",", " ").split()
        if text:
            times = [_number(t) for t in text]
            if None in times:
                raise ValueError("The start times must be numbers.")
            return echem.operando_times(n, times_s=np.asarray(times) * unit)
        per = _number(self.per_edit.text())
        if per is None or per <= 0:
            raise ValueError("Say how long each spectrum took, or give every start time.")
        return echem.operando_times(n, start_s=(_number(self.start_edit.text()) or 0.0) * unit,
                                    per_spectrum_s=per * unit, gaps_s=(_number(self.gap_edit.text()) or 0.0) * unit)

    def _ex_situ_points(self) -> list[dict]:
        points = []
        for r in range(self.ex_table.rowCount()):
            text = lambda c: self.ex_table.item(r, c).text().strip()
            value = _number(text(1))
            if value is None:
                raise ValueError(f"Sample {r + 1} ({text(0)}) needs a value.")
            cycle = _number(text(2))
            points.append({"value": value, "cycle": int(cycle) if cycle else 1,
                           "half": text(3) or "charge", "choose": text(4) or "first", "label": text(0)})
        return points

    def _cycles(self):
        text = self.cycles_edit.text().replace(",", " ").split()
        if not text:
            return None
        numbers = [_number(t) for t in text]
        if None in numbers:
            raise ValueError("Cycles are whole numbers, e.g. 1, 2, 5.")
        return [int(n) for n in numbers]

    def compute(self) -> dict:
        """Everything the figure needs, or raises ``ValueError`` saying what is missing."""
        self.refresh()          # a mass set by right-clicking since last time is picked up
        if self.cell is None and not self.load():
            raise ValueError(self.report_label.text() or "Load an electrochemistry file first.")
        data, mode = self.cell["data"], self._mode()
        out = {"data": data, "mode": mode, "values": None, "value_columns": None}
        if mode == "values":
            if self.values is None:
                self._read_values()
            if self.values is None:
                raise ValueError("Choose a values file first.")
            table = self.values.dropna(subset=[self.value_x_combo.currentData()]).reset_index(drop=True)
            xs = pd.to_numeric(table[self.value_x_combo.currentData()], errors="coerce").to_numpy(float)
            kind = self.value_x_type.currentData()
            if kind == "index":
                starts, ends = self._timing(len(table))
            else:
                scale = 3600.0 if kind == "time_h" else 1.0
                starts = ends = xs * scale
            link = ep.link_table(data, "operando", starts_s=starts, ends_s=ends)
            cols = [c.strip() for c in self.value_cols_edit.text().split(",") if c.strip()]
            cols = cols or [c for c in table.columns if c != self.value_x_combo.currentData()][:1]
            missing = [c for c in cols if c not in table.columns]
            if missing:
                raise ValueError(f"The values file has no column {missing[0]!r}.")
            out.update(link=link, labels=[str(i + 1) for i in range(len(table))], ppm=None,
                       matrix=None, values=table, value_columns=cols, display="values")
            return out
        chosen, notes = self._chosen()
        if not chosen:
            raise ValueError("No spectra chosen. " + " ".join(notes))
        ppm, matrix, interp_notes = ep.common_axis([e["spectrum"] for e in chosen])
        t = self.ex_table
        cell = lambda r, c: (t.item(r, c).text().strip() if t.item(r, c) else "")
        rows = range(len(chosen))
        masses = [_number(cell(r, 5)) for r in rows]
        factors = [_number(cell(r, 6)) for r in rows]
        mults, scale_notes = ep.scaling_factors(
            [e["spectrum"] for e in chosen], per_scan=self.per_scan_check.isChecked(),
            divide_rg=self.divide_rg_check.isChecked(),
            masses_mg=[np.nan if m is None else m for m in masses] if self.by_mass_check.isChecked() else None,
            factors=[1.0 if f is None else f for f in factors])
        matrix = matrix * mults[:, None]
        interp_notes = interp_notes + scale_notes
        unit = ("a.u." if not (self.per_scan_check.isChecked() or self.by_mass_check.isChecked()
                               or self.divide_rg_check.isChecked()) else "a.u., " + ", ".join(
            name for name, on in (("per scan", self.per_scan_check.isChecked()),
                                  ("per RG", self.divide_rg_check.isChecked()),
                                  ("per mg", self.by_mass_check.isChecked())) if on))
        nuclei = {str(e["spectrum"].metadata.nucleus) for e in chosen if e["spectrum"].metadata.nucleus}
        if len(nuclei) > 1:
            interp_notes = interp_notes + ["The spectra are of different nuclei: the shift axis is not named."]
        out.update(intensity_label=f"Intensity ({unit})",
                   nucleus=next(iter(nuclei)) if len(nuclei) == 1 else None,
                   mhz=next((float(e["spectrum"].metadata.sfo1) for e in chosen
                             if getattr(e["spectrum"].metadata, "sfo1", None)), None))
        if mode == "operando":
            starts, ends = self._timing(len(chosen))
            link = ep.link_table(data, "operando", starts_s=starts, ends_s=ends)
        else:
            link = ep.link_table(data, "ex_situ", points=self._ex_situ_points(),
                                 quantity=self.quantity_combo.currentData())
        link.attrs["notes"] = list(link.attrs.get("notes", [])) + notes + interp_notes
        out.update(link=link, labels=[e["label"] for e in chosen], ppm=ppm, matrix=matrix,
                   display=self.display_combo.currentData())
        return out

    def _kwargs(self, spec: dict) -> dict:
        kw = dict(x=self.x_combo.currentData(), y=self.y_combo.currentData(),
                  side=self.side_combo.currentData(), display=spec["display"],
                  position=self.position_combo.currentData(),
                  differences=self.diff_combo.currentData() or None,
                  cycles=self._cycles(), half=self.half_combo.currentData(),
                  colour_by_cycle=self.cycle_colour_check.isChecked(),
                  show_points=self.points_check.isChecked(), label_points=self.label_check.isChecked())
        if spec["display"] == "values":
            kw["differences"] = None
        style = self._style()
        kw.update(style)
        kw["distribute"] = self.distribute_combo.currentData()
        kw["link_colours"] = self.link_colour_check.isChecked()
        if spec["display"] != "values":
            kw["intensity_label"] = spec["intensity_label"]
            import matplotlib.colors as _mc
            n = len(spec["labels"])
            auto = [_mc.to_hex(c) for c in ep._auto_colours(n, style.get("colormap", "viridis"))]
            # Each spectrum and point is its own series in Plot Style, where its colour is changed
            kw["point_colours"] = auto if self.auto_colour_check.isChecked() else None
            kw["spectrum_colours"] = None
            kw["nucleus"] = spec.get("nucleus")
        return kw

    def plot(self) -> None:
        try:
            spec = self.compute()
            kwargs = self._kwargs(spec)
        except ValueError as exc:
            self._fail(str(exc))
            return
        self.link = spec["link"]
        try:        # draw once off screen, so a problem is said here and the plot on screen is kept
            from matplotlib.figure import Figure
            ep.plot_echem_nmr(spec["data"], spec["ppm"], spec["matrix"], spec["link"], spec["labels"],
                              values=spec["values"], value_columns=spec["value_columns"],
                              figure=Figure(), **{**kwargs, "crosshair": False})
        except ValueError as exc:
            self._fail(str(exc))
            return
        def draw(figure, spec=spec, kwargs=dict(kwargs)):
            kwargs = {**kwargs, **self._style()}
            ep.plot_echem_nmr(spec["data"], spec["ppm"], spec["matrix"], spec["link"], spec["labels"],
                              values=spec["values"], value_columns=spec["value_columns"],
                              figure=figure, **kwargs)
        self._last_spec = spec
        self._draw(draw, hz_per_ppm=spec.get("mhz"))
        self.plotted = True
        notes = list(self.link.attrs.get("notes", []))
        self._fail("Plotted " + f"{len(self.link)} points. " + " ".join(notes))

    def replot(self) -> None:
        """A Plot Style ▸ Electrochemistry control changed: draw again, if there is a plot."""
        if self.plotted:
            self.plot()

    def export_csv(self) -> None:
        try:
            spec = self.compute()
        except ValueError as exc:
            self._fail(str(exc))
            return
        path, _ = QFileDialog.getSaveFileName(self, "Link table", "echem_link.csv", "CSV (*.csv)")
        if not path:
            return
        table = spec["link"].copy()
        table.insert(1, "label", spec["labels"])
        table.to_csv(path, index=False)
        self._fail(f"Saved {len(table)} rows to {path}.")

    def notebook_code(self) -> str:
        mode = self._mode()
        lines = ["from opennmr.notebook import echem, echem_plot as ep", "",
                 f"cell = echem.load_echem({self.path_edit.text().strip()!r}, "
                 f"mass_mg={_number(self.mass_edit.text())!r}, area_cm2={_number(self.area_edit.text())!r}, "
                 f"formula={(self.formula_edit.text().strip() or None)!r}, "
                 f"molar_mass={_number(self.molar_edit.text())!r}, x0={_number(self.x0_edit.text()) or 1.0!r}, "
                 f"charge_removes_ion={self.removes_check.isChecked()!r})",
                 "data = cell[\"data\"]"]
        if mode == "operando":
            unit = dict(_UNITS)[self.time_unit_combo.currentData()]
            lines += ["# spectra: a list of 1D opennmr spectra, in order",
                      "ppm, matrix, _ = ep.common_axis(spectra)",
                      f"starts, ends = echem.operando_times(len(spectra), start_s="
                      f"{(_number(self.start_edit.text()) or 0.0) * unit!r}, per_spectrum_s="
                      f"{(_number(self.per_edit.text()) or 0.0) * unit!r}, gaps_s="
                      f"{(_number(self.gap_edit.text()) or 0.0) * unit!r})",
                      "link = ep.link_table(data, \"operando\", starts_s=starts, ends_s=ends)"]
        elif mode == "ex_situ":
            try:
                points = self._ex_situ_points()
            except ValueError:
                points = []
            lines += ["ppm, matrix, _ = ep.common_axis(spectra)", f"points = {points!r}",
                      f"link = ep.link_table(data, \"ex_situ\", points=points, "
                      f"quantity={self.quantity_combo.currentData()!r})"]
        else:
            lines += [f"values = echem.read_value_table({self.values_edit.text().strip()!r})",
                      "# link: positions of the values on the cell, e.g. from their times in seconds",
                      "link = ep.link_table(data, \"operando\", starts_s=times_s, ends_s=times_s)",
                      "ppm = matrix = None"]
        if mode != "values":
            t = self.ex_table
            cell = lambda r, c: (t.item(r, c).text().strip() if t.item(r, c) else "")
            masses = [_number(cell(r, 5)) for r in range(t.rowCount())]
            factors = [_number(cell(r, 6)) or 1.0 for r in range(t.rowCount())]
            by_mass = self.by_mass_check.isChecked()
            if (self.per_scan_check.isChecked() or self.divide_rg_check.isChecked() or by_mass
                    or any(f != 1.0 for f in factors)):
                lines += [f"mult, _ = ep.scaling_factors(spectra, per_scan={self.per_scan_check.isChecked()!r}, "
                          f"divide_rg={self.divide_rg_check.isChecked()!r}, "
                          f"masses_mg={masses if by_mass else None!r}, factors={factors!r})",
                          "matrix = matrix * mult[:, None]"]
        kw = (f"x={self.x_combo.currentData()!r}, y={self.y_combo.currentData()!r}, "
              f"side={self.side_combo.currentData()!r}, "
              f"display={('values' if mode == 'values' else self.display_combo.currentData())!r}, "
              f"position={self.position_combo.currentData()!r}")
        if self.distribute_combo.currentData() != "position" and mode != "values":
            kw += f", distribute={self.distribute_combo.currentData()!r}"
        if self.diff_combo.currentData() and mode != "values":
            kw += f", differences={self.diff_combo.currentData()!r}"
        if mode == "values":
            kw += f", values=values, value_columns={[c.strip() for c in self.value_cols_edit.text().split(',') if c.strip()] or None!r}"
        lines += ["", f"fig = ep.plot_echem_nmr(data, ppm, matrix, link, {kw})"]
        return "\n".join(lines)

    def copy_code(self) -> None:
        QGuiApplication.clipboard().setText(self.notebook_code())
        self._fail("Notebook code copied.")
