"""Simulate > SIMPSON: bring SIMPSON output into openNMR.

Three things SIMPSON writes, three ways to use them:

* **Spectra** (``fsave`` of a spectrum, ``.spe``) -- plotted, or added to
  *Open experiments* where everything that works on a 1D spectrum works.
* **FIDs** (``.fid``) -- read as a time-domain signal, so Process (window,
  zero-fill, phase ...) acts on them live exactly as on an experimental FID.
* **Points from FIDs** (efficiencies) -- the first few points of a series of
  FIDs, plotted against a quantity whose name and label you choose, or
  combined (``A + B - 0.3*C``).
* **A curve from one file** (REDOR-like) -- every point of one file as y,
  against x values from the file (k / SW or MAS rate), a typed list, a list
  file or start/step -- the same choices as the vdlist in Dynamics.

* **A heatmap of two swept quantities** -- a table with one row per
  combination (columns such as ``O1 O2 Re``, header optional), or one FID per
  combination with the two values in each file name.

Pulse shapes (``save_shape``) are plotted as nu1, phase and the frequency
sweep against time, with the pulse length from the file name or typed.

A spectrum, or one made by transforming a SIMPSON FID, can be sent to
Fit > SIMPSON spectrum to fit experimental data against it.

All the reading and maths are ``opennmr.io.simpson`` and
``opennmr.notebook.simpson_series``; *Copy notebook code* gives the calls.
"""

from __future__ import annotations

import os
import re
from typing import Callable

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit, QListWidget, QMessageBox, QPlainTextEdit,
    QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from ..io import simpson as simpson_io
from ..notebook import simpson_series
from . import panel_state
from .widgets.typed_spinbox import TypedSpinBox
from .widgets.field_input import MAX_FIELD_T, FieldSpinBox, proton_frequency_box

FILE_FILTER = ("SIMPSON files (*.spe *.fid *.csdf *.dat *.shape *.tcl *.txt *.xreim *.out);;"
               "All files (*)")
COMMON_NUCLEI = ("", "1H", "2H", "6Li", "7Li", "11B", "13C", "14N", "15N", "17O", "19F", "23Na",
                 "27Al", "29Si", "31P", "35Cl", "51V", "89Y", "119Sn", "133Cs", "207Pb")
KINDS = (("auto", "Detect from the file"), ("spe", "Spectrum (frequency domain)"),
         ("fid", "FID (time domain)"))
X_SOURCES = (("filenames", "Number in each file name"),
             ("steps", "Start + step"),
             ("typed", "Typed into the table"))
CURVE_X_SOURCES = (("file", "From the file (k / rate, or its x column)"),
                   ("index", "Point number (0, 1, 2 …)"),
                   ("typed", "Typed list"),
                   ("list", "List file × factor"),
                   ("steps", "Start + step"))


class SimpsonPanel(QWidget):
    """The SIMPSON section of the Simulate tile.

    Callbacks (all supplied by the main window):

    ``open_files(paths, kwargs)``  read each file into *Open experiments*;
    ``plot_spectrum(spectrum)``    draw a spectrum without adding it;
    ``send_to_fit(spectrum, label)`` give it to Fit > SIMPSON spectrum;
    ``selected_spectrum()``        ``(spectrum, label)`` selected in Open
                                   experiments, or ``None``;
    ``draw_plot(draw)``            show ``draw(fig)`` on the plot area.
    """

    def __init__(self, open_files: Callable, plot_spectrum: Callable, send_to_fit: Callable,
                 selected_spectrum: Callable, draw_plot: Callable, parent=None) -> None:
        super().__init__(parent)
        self._open_files = open_files
        self._plot_spectrum = plot_spectrum
        self._send_to_fit = send_to_fit
        self._selected_spectrum = selected_spectrum
        self._draw_plot = draw_plot
        #: The last spectrum read here (for Send to Fit without adding).
        self.spectrum = None
        self.series: dict | None = None
        self.curve: dict | None = None
        self.grid: dict | None = None

        layout = QVBoxLayout(self)
        layout.addWidget(self._build_files_box())
        layout.addWidget(self._build_axis_box())
        layout.addWidget(self._build_spectra_box())
        layout.addWidget(self._build_points_box())
        layout.addWidget(self._build_curve_box())
        layout.addWidget(self._build_heatmap_box())
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        self._refresh_state()

    # ------------------------------------------------------------------
    # 1. Files
    # ------------------------------------------------------------------
    def _build_files_box(self) -> QGroupBox:
        box = QGroupBox("1. SIMPSON files")
        v = QVBoxLayout(box)
        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.file_list.setMinimumHeight(90)
        self.file_list.setToolTip("The SIMPSON output files (fsave: .spe, .fid, or -xreim/-re "
                                  "text). Spectra and FIDs are opened one by one; for points, "
                                  "the whole list is one series.")
        v.addWidget(self.file_list)
        row = QHBoxLayout()
        self.add_files_button = QPushButton("Add files…")
        self.add_files_button.clicked.connect(self._on_add_files)
        self.remove_files_button = QPushButton("Remove")
        self.remove_files_button.clicked.connect(self._on_remove_files)
        self.clear_files_button = QPushButton("Clear")
        self.clear_files_button.clicked.connect(self._on_clear_files)
        for b in (self.add_files_button, self.remove_files_button, self.clear_files_button):
            row.addWidget(b)
        v.addLayout(row)
        return box

    def paths(self) -> list[str]:
        return [self.file_list.item(i).data(Qt.UserRole) for i in range(self.file_list.count())]

    def add_paths(self, paths) -> None:
        have = set(self.paths())
        for path in sorted(str(p) for p in paths):
            if path in have:
                continue
            if path.lower().endswith(".tcl"):
                self.apply_script(path)
                continue
            item = self.file_list.addItem(os.path.basename(path))
            self.file_list.item(self.file_list.count() - 1).setData(Qt.UserRole, path)
            self.file_list.item(self.file_list.count() - 1).setToolTip(path)
            have.add(path)
        self._sync_x_table()
        self._refresh_state()

    def apply_script(self, path: str) -> dict | None:
        """Fill in the nuclide and field from a SIMPSON input script (.tcl).
        The script is only read, never run."""
        try:
            settings = simpson_io.read_tcl_settings(path)
        except Exception as exc:  # noqa: BLE001 -- shown to the user
            self._say(f"Could not read {os.path.basename(path)}: {exc}")
            return None
        done = []
        if settings.get("nucleus"):
            self.nuclide_combo.setCurrentText(settings["nucleus"])
            done.append(f"nuclide {settings['nucleus']}")
        if settings.get("proton_mhz"):
            self._proton_spin.setValue(float(settings["proton_mhz"]))
            done.append(f"{settings['proton_mhz']:g} MHz ¹H")
        self._say(f"{os.path.basename(path)}: set " + (", ".join(done) if done else "nothing")
                  + ". Check the nuclide and field; the carrier is not in the script.")
        return settings

    def _on_add_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "SIMPSON files", "", FILE_FILTER)
        if paths:
            self.add_paths(paths)

    def _on_remove_files(self) -> None:
        for item in self.file_list.selectedItems():
            self.file_list.takeItem(self.file_list.row(item))
        self._sync_x_table()
        self._refresh_state()

    def _on_clear_files(self) -> None:
        self.file_list.clear()
        self._sync_x_table()
        self._refresh_state()

    # ------------------------------------------------------------------
    # 2. Frequency axis
    # ------------------------------------------------------------------
    def _build_axis_box(self) -> QGroupBox:
        box = QGroupBox("2. Spectrometer (for the ppm axis)")
        form = QFormLayout(box)
        self.kind_combo = QComboBox()
        for key, text in KINDS:
            self.kind_combo.addItem(text, key)
        self.kind_combo.setToolTip("What the file holds. Headerless -xreim text files cannot be "
                                   "recognised; say which it is.")
        form.addRow("File holds:", self.kind_combo)
        self.nuclide_combo = QComboBox()
        self.nuclide_combo.setEditable(True)
        self.nuclide_combo.addItems(COMMON_NUCLEI)
        self.nuclide_combo.setToolTip("Isotope, for the ppm axis and its label. Leave empty to "
                                      "keep the axis in Hz.")
        form.addRow("Nuclide:", self.nuclide_combo)
        self.field_spin = FieldSpinBox()
        self.field_spin.setRange(0.0, MAX_FIELD_T)
        self.field_spin.setDecimals(6)
        self.field_spin.setValue(0.0)
        self.field_spin.setSuffix(" T")
        self.field_spin.setSpecialValueText("not set")
        self.field_spin.setToolTip("Field the simulation was run at – or type the ¹H frequency "
                                   "beside it. Needed for a ppm axis.")
        self._proton_spin = proton_frequency_box(self.field_spin, parent=self)
        row = QHBoxLayout()
        row.addWidget(self.field_spin, 1)
        row.addWidget(QLabel("or"))
        row.addWidget(self._proton_spin, 1)
        form.addRow("Field:", row)
        self.reference_spin = QDoubleSpinBox()
        self.reference_spin.setRange(-1e6, 1e6)
        self.reference_spin.setDecimals(3)
        self.reference_spin.setSuffix(" ppm")
        self.reference_spin.setToolTip("Shift of the transmitter (the carrier) – 0 for a "
                                       "simulation done on resonance.")
        form.addRow("Carrier at:", self.reference_spin)
        self.time_unit_combo = QComboBox()
        for key, text in (("s", "seconds"), ("ms", "milliseconds"), ("us", "microseconds")):
            self.time_unit_combo.addItem(text, key)
        self.time_unit_combo.setToolTip("Unit of the time axis of a text FID that carries no sweep "
                                        "width in its header.")
        form.addRow("FID time unit:", self.time_unit_combo)
        self.sw_spin = QDoubleSpinBox()
        self.sw_spin.setRange(0.0, 1e9)
        self.sw_spin.setDecimals(3)
        self.sw_spin.setSuffix(" Hz")
        self.sw_spin.setSpecialValueText("from file")
        self.sw_spin.setToolTip("Spectral width, when the file does not give it.")
        form.addRow("Spectral width:", self.sw_spin)
        return box

    def read_kwargs(self) -> dict:
        """The ``opennmr.io.simpson.read_1d`` arguments the controls describe."""
        kwargs: dict = {}
        kind = self.kind_combo.currentData()
        if kind != "auto":
            kwargs["kind"] = kind
        nucleus = self.nuclide_combo.currentText().strip()
        if nucleus:
            kwargs["nucleus"] = nucleus
        if self.field_spin.value() > 0:
            kwargs["field"] = float(self.field_spin.value())
        if self.reference_spin.value() != 0:
            kwargs["reference_ppm"] = float(self.reference_spin.value())
        if self.time_unit_combo.currentData() != "s":
            kwargs["time_unit"] = self.time_unit_combo.currentData()
        if self.sw_spin.value() > 0:
            kwargs["sw"] = float(self.sw_spin.value())
        return kwargs

    # ------------------------------------------------------------------
    # 3. Spectra and FIDs
    # ------------------------------------------------------------------
    def _build_spectra_box(self) -> QGroupBox:
        box = QGroupBox("3. Spectra and FIDs")
        v = QVBoxLayout(box)
        note = QLabel("An FID is transformed here and stays an FID: window, zero-fill, phase "
                      "and the rest are in Process, as for any experiment.")
        note.setWordWrap(True)
        v.addWidget(note)
        row = QHBoxLayout()
        row.addWidget(QLabel("Element to plot:"))
        self.element_spin = TypedSpinBox()
        self.element_spin.setRange(1, 9999)
        self.element_spin.setValue(1)
        self.element_spin.setToolTip("A SIMPSON file with NELEM > 1 holds several FIDs or spectra "
                                     "(e.g. one per detect operator). Plot shows this one; Add to "
                                     "Open experiments adds all of them, as name [k/n].")
        row.addWidget(self.element_spin)
        row.addStretch(1)
        v.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(QLabel("Pulse length (shapes):"))
        self.shape_duration_spin = QDoubleSpinBox()
        self.shape_duration_spin.setRange(0.0, 1e7)
        self.shape_duration_spin.setDecimals(4)
        self.shape_duration_spin.setSuffix(" µs")
        self.shape_duration_spin.setSpecialValueText("from file name")
        self.shape_duration_spin.setToolTip(
            "Length of a pulse shape (.shape, from save_shape), for its time axis, frequency "
            "sweep and adiabaticity. \"from file name\" reads it from names such as "
            "inv_pow_10.0us_…shape. Each of the N points lasts length / N, as in SIMPSON's "
            "pulse_shaped.")
        row.addWidget(self.shape_duration_spin)
        row.addStretch(1)
        v.addLayout(row)
        row = QHBoxLayout()
        self.plot_button = QPushButton("Plot")
        self.plot_button.setToolTip("Read the selected file (or the first) and show it, without "
                                    "adding it to Open experiments.")
        self.plot_button.clicked.connect(self._on_plot)
        self.open_button = QPushButton("Add to Open experiments")
        self.open_button.setToolTip("Read every file in the list (or the selected ones) into Open "
                                    "experiments. Everything that works on a 1D spectrum then "
                                    "works on them.")
        self.open_button.clicked.connect(self._on_open)
        row.addWidget(self.plot_button)
        row.addWidget(self.open_button)
        v.addLayout(row)
        row = QHBoxLayout()
        self.send_button = QPushButton("Send to Fit")
        self.send_button.setToolTip("Use this simulated spectrum – the one just plotted, or the "
                                    "SIMPSON spectrum selected in Open experiments (e.g. a "
                                    "transformed FID) – as a lineshape in Fit > SIMPSON spectrum.")
        self.send_button.clicked.connect(self._on_send_to_fit)
        self.send_selected_button = QPushButton("Send selected experiment to Fit")
        self.send_selected_button.setToolTip("The spectrum selected in Open experiments, "
                                             "whatever its origin.")
        self.send_selected_button.clicked.connect(lambda: self._on_send_to_fit(selected=True))
        row.addWidget(self.send_button)
        row.addWidget(self.send_selected_button)
        v.addLayout(row)
        return box

    def _chosen_paths(self) -> list[str]:
        selected = [i.data(Qt.UserRole) for i in self.file_list.selectedItems()]
        return selected or self.paths()

    def read_spectrum(self, path: str, element: int | None = None):
        kwargs = self.read_kwargs()
        kwargs["element"] = (self.element_spin.value() - 1) if element is None else element
        return simpson_io.read_1d(path, **kwargs)

    def _plot_table(self, path: str) -> None:
        from ..notebook import simpson as simpson_nb
        table = simpson_nb.SIMPSONread_table(path)

        def draw(fig):
            fig.clear()
            ax = fig.add_subplot(111)
            simpson_nb.SIMPSONplot_table(table, ax=ax)
            return ax

        self._draw_plot(draw)

    def shape_profile(self, path: str) -> dict:
        """The pulse shape at ``path``, with this panel's pulse length."""
        from ..notebook import simpson as simpson_nb
        duration = self.shape_duration_spin.value() or None
        return simpson_nb.SIMPSONshape_profile(path, duration)

    def _plot_shape(self, path: str) -> str:
        from ..notebook import simpson as simpson_nb
        profile = self.shape_profile(path)  # raises with a clear message if malformed
        duration = profile["duration_us"]

        def draw(fig):
            _fig, axes = simpson_nb.SIMPSONplot_shape(path, duration_us=duration, fig=fig,
                                                      show_offset=True)
            return axes[0]

        self._draw_plot(draw)
        return simpson_nb.SIMPSONdescribe_shape(profile)

    def _on_plot(self) -> None:
        paths = self._chosen_paths()
        if not paths:
            self._say("Add a SIMPSON file first.")
            return
        name = os.path.basename(paths[0])
        role = simpson_io.describe(paths[0])["role"]
        try:
            if role == "table":
                self._plot_table(paths[0])
                self._say(f"Plotted {name} as a table of x, y, z (a parameter scan). It is not "
                          "a spectrum, so it cannot be added to Open experiments.")
                return
            if role == "shape":
                summary = self._plot_shape(paths[0])
                self._say(f"Plotted the pulse shape {name}: {summary}")
                return
            n_elements = simpson_io.describe(paths[0]).get("elements", 1)
            self.element_spin.setMaximum(max(n_elements, 1))
            self.spectrum = self.read_spectrum(paths[0])
        except Exception as exc:  # noqa: BLE001 -- shown to the user
            QMessageBox.warning(self, "SIMPSON", f"Could not read {name}:\n{exc}")
            return
        self._plot_spectrum(self.spectrum)
        self._say(f"Plotted {name}"
                  + (f" (element {self.element_spin.value()} of {n_elements})"
                     if n_elements > 1 else "") + ".")

    def _on_open(self) -> None:
        paths = self._chosen_paths()
        if not paths:
            self._say("Add a SIMPSON file first.")
            return
        added, skipped, count = [], [], 0
        for path in paths:
            info = simpson_io.describe(path)
            if info["role"] != "data":
                skipped.append(f"{os.path.basename(path)} ({info['detail']})")
                continue
            elements = info.get("elements", 1)
            for k in range(elements):
                kwargs = self.read_kwargs()
                if elements > 1:
                    kwargs["element"] = k
                self._open_files([path], kwargs)
                count += 1
            added.append(path)
        text = f"Added {count} spectrum/FID(s) from {len(added)} file(s) to Open experiments."
        if skipped:
            text += " Not spectra or FIDs, so not added: " + "; ".join(skipped) + "."
        self._say(text)

    def _on_send_to_fit(self, selected: bool = False) -> None:
        spectrum, label = None, ""
        if not selected and self.spectrum is not None:
            spectrum = self.spectrum
            label = os.path.basename(getattr(spectrum.metadata, "source_path", "") or "SIMPSON")
        if spectrum is None:
            chosen = self._selected_spectrum()
            if chosen is None:
                self._say("Plot a SIMPSON file here, or select one in Open experiments, first.")
                return
            spectrum, label = chosen
        if spectrum.ndim != 1:
            self._say("Only a 1D spectrum can be sent to Fit.")
            return
        self._send_to_fit(spectrum, label)
        self._say(f"Sent {label} to Fit ▸ SIMPSON spectrum.")

    # ------------------------------------------------------------------
    # 4. Points
    # ------------------------------------------------------------------
    def _build_points_box(self) -> QGroupBox:
        box = QGroupBox("4. Points from FIDs (efficiencies)")
        v = QVBoxLayout(box)
        note = QLabel("Reads the first points of every file in the list (A = first point, "
                      "B = second ...) and plots them, or combinations of them, against a "
                      "quantity you name.")
        note.setWordWrap(True)
        v.addWidget(note)
        form = QFormLayout()
        self.n_points_spin = TypedSpinBox()
        self.n_points_spin.setRange(1, 26)
        self.n_points_spin.setValue(3)
        self.n_points_spin.setToolTip("Points kept from each file (A, B, C ... up to Z).")
        form.addRow("Points per file:", self.n_points_spin)
        self.x_source_combo = QComboBox()
        for key, text in X_SOURCES:
            self.x_source_combo.addItem(text, key)
        self.x_source_combo.currentIndexChanged.connect(lambda _i: self._sync_x_table())
        form.addRow("Swept values from:", self.x_source_combo)
        self.x_start_spin = QDoubleSpinBox()
        self.x_start_spin.setRange(-1e9, 1e9)
        self.x_start_spin.setDecimals(6)
        self.x_start_spin.valueChanged.connect(lambda _v: self._sync_x_table())
        self.x_step_spin = QDoubleSpinBox()
        self.x_step_spin.setRange(-1e9, 1e9)
        self.x_step_spin.setDecimals(6)
        self.x_step_spin.setValue(1.0)
        self.x_step_spin.valueChanged.connect(lambda _v: self._sync_x_table())
        step_row = QHBoxLayout()
        step_row.addWidget(self.x_start_spin)
        step_row.addWidget(QLabel("step"))
        step_row.addWidget(self.x_step_spin)
        form.addRow("Start / step:", step_row)
        self.x_name_edit = QLineEdit("x")
        self.x_name_edit.setToolTip("Short name for the quantity: a column heading in the saved "
                                    "table (e.g. B1, tp).")
        form.addRow("Quantity name:", self.x_name_edit)
        self.x_label_edit = QLineEdit()
        self.x_label_edit.setPlaceholderText("axis text, e.g. RF amplitude (kHz)")
        self.x_label_edit.setToolTip("The x-axis text. Symbols such as B1 or t are italicised.")
        form.addRow("Axis label:", self.x_label_edit)
        v.addLayout(form)

        self.x_table = QTableWidget(0, 2)
        self.x_table.setHorizontalHeaderLabels(["File", "Value"])
        self.x_table.horizontalHeader().setStretchLastSection(True)
        self.x_table.setMinimumHeight(110)
        v.addWidget(self.x_table)

        self.expressions_edit = QPlainTextEdit("A")
        self.expressions_edit.setFixedHeight(64)
        self.expressions_edit.setToolTip("What to plot, one curve per line: A, or A + B - 0.3*C. "
                                         "Letters name the points; + - * / ( ) and numbers only.")
        v.addWidget(QLabel("Curves (one per line):"))
        v.addWidget(self.expressions_edit)
        form2 = QFormLayout()
        self.part_combo = QComboBox()
        for part in simpson_series.PARTS:
            self.part_combo.addItem(part, part)
        self.part_combo.setToolTip("Which part of the complex points is combined and plotted.")
        form2.addRow("Part:", self.part_combo)
        self.ylabel_edit = QLineEdit()
        self.ylabel_edit.setPlaceholderText("default: the expression, or Intensity")
        form2.addRow("y label:", self.ylabel_edit)
        self.normalise_check = QCheckBox("Normalise to the largest value")
        form2.addRow("", self.normalise_check)
        v.addLayout(form2)
        row = QHBoxLayout()
        self.plot_points_button = QPushButton("Plot points")
        self.plot_points_button.clicked.connect(self._on_plot_points)
        self.save_points_button = QPushButton("Save table (CSV)…")
        self.save_points_button.clicked.connect(self._on_save_points)
        self.copy_code_button = QPushButton("Copy notebook code")
        self.copy_code_button.clicked.connect(self._on_copy_code)
        for b in (self.plot_points_button, self.save_points_button, self.copy_code_button):
            row.addWidget(b)
        v.addLayout(row)
        return box

    def _sync_x_table(self) -> None:
        paths = self.paths()
        mode = self.x_source_combo.currentData()
        old = {}
        for r in range(self.x_table.rowCount()):
            item = self.x_table.item(r, 1)
            if item is not None:
                old[self.x_table.item(r, 0).data(Qt.UserRole)] = item.text()
        self.x_table.blockSignals(True)
        self.x_table.setRowCount(len(paths))
        if mode == "filenames":
            try:
                values = [f"{v:g}" for v in simpson_series.x_from_filenames(paths)]
            except Exception:  # noqa: BLE001 -- shown in the cells
                values = ["?" for _ in paths]
        elif mode == "steps":
            values = [f"{self.x_start_spin.value() + i * self.x_step_spin.value():g}"
                      for i in range(len(paths))]
        else:
            values = [old.get(p, str(i + 1)) for i, p in enumerate(paths)]
        for r, (path, value) in enumerate(zip(paths, values)):
            name = QTableWidgetItem(os.path.basename(path))
            name.setData(Qt.UserRole, path)
            name.setFlags(name.flags() & ~Qt.ItemIsEditable)
            self.x_table.setItem(r, 0, name)
            cell = QTableWidgetItem(value)
            if mode != "typed":
                cell.setFlags(cell.flags() & ~Qt.ItemIsEditable)
            self.x_table.setItem(r, 1, cell)
        self.x_table.blockSignals(False)
        self.x_start_spin.setEnabled(mode == "steps")
        self.x_step_spin.setEnabled(mode == "steps")

    def x_values(self) -> list[float]:
        values = []
        for r in range(self.x_table.rowCount()):
            text = self.x_table.item(r, 1).text().strip().replace(",", ".")
            try:
                values.append(float(text))
            except ValueError:
                raise ValueError(f"Row {r + 1} ({self.x_table.item(r, 0).text()}): "
                                 f"{text!r} is not a number.") from None
        return values

    def expressions(self) -> list[str]:
        return [ln.strip() for ln in self.expressions_edit.toPlainText().splitlines() if ln.strip()]

    def read_series(self) -> dict:
        paths = self.paths()
        if not paths:
            raise ValueError("Add the SIMPSON FID files first.")
        kwargs = self.read_kwargs()
        series = simpson_series.SIMPSONpoint_series(
            paths, x=self.x_values(), xname=self.x_name_edit.text().strip() or "x",
            xlabel=self.x_label_edit.text().strip() or None,
            n_points=self.n_points_spin.value(), kind=kwargs.get("kind"))
        self.series = series
        return series

    def _checked_series(self):
        try:
            series = self.read_series()
            expressions = self.expressions() or ["A"]
            for expression in expressions:
                problem = simpson_series.check_expression(series, expression)
                if problem:
                    raise ValueError(f"{expression!r}: {problem}")
        except Exception as exc:  # noqa: BLE001 -- shown to the user
            QMessageBox.warning(self, "SIMPSON points", str(exc))
            return None, None
        return series, expressions

    def _on_plot_points(self) -> None:
        series, expressions = self._checked_series()
        if series is None:
            return
        part = self.part_combo.currentData()
        ylabel = self.ylabel_edit.text().strip() or None
        normalise = self.normalise_check.isChecked()

        def draw(fig):
            fig.clear()
            ax = fig.add_subplot(111)
            simpson_series.SIMPSONplot_point_series(series, expressions, part=part, ax=ax,
                                                    ylabel=ylabel, normalise=normalise)
            return ax

        self._draw_plot(draw)
        self._say(f"Plotted {len(expressions)} curve(s) from {len(series['paths'])} files "
                  f"(points {', '.join(series['letters'])}).")

    def _on_save_points(self) -> None:
        series, expressions = self._checked_series()
        if series is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save point table", "points.csv",
                                              "CSV (*.csv)")
        if path:
            simpson_series.SIMPSONsave_point_table(series, expressions, path,
                                                   part=self.part_combo.currentData())
            self._say(f"Saved {path}.")

    # ------------------------------------------------------------------
    # 5. One file as a curve (REDOR-like)
    # ------------------------------------------------------------------
    def _build_curve_box(self) -> QGroupBox:
        box = QGroupBox("5. Curve from one file (REDOR-like)")
        v = QVBoxLayout(box)
        note = QLabel("Every point of the selected file (or the first) as y – the real "
                      "column, by default – against x values you choose: e.g. a REDOR "
                      "curve saved as real, imaginary (zeros) and the MAS rate.")
        note.setWordWrap(True)
        v.addWidget(note)
        form = QFormLayout()
        self.curve_x_combo = QComboBox()
        for key, text in CURVE_X_SOURCES:
            self.curve_x_combo.addItem(text, key)
        self.curve_x_combo.setToolTip(
            "From the file: point k at k / rate – the rate is the SIMP header's SW, or the "
            "MAS rate written in the file (a third column, or a 'mas = …' line); or the file's "
            "own x column.\nPoint number: 0, 1, 2 …\nTyped list / List file / Start + step: as "
            "for the vdlist in Dynamics. A list longer than the data is fine (the extra values "
            "are not used); a shorter one is refused.")
        self.curve_x_combo.currentIndexChanged.connect(lambda _i: self._sync_curve_controls())
        form.addRow("x values:", self.curve_x_combo)

        self.curve_time_unit_combo = QComboBox()
        for key, text in (("ms", "ms"), ("us", "µs"), ("s", "s")):
            self.curve_time_unit_combo.addItem(text, key)
        self.curve_time_unit_label = QLabel("Time unit:")
        form.addRow(self.curve_time_unit_label, self.curve_time_unit_combo)

        self.curve_typed_edit = QLineEdit()
        self.curve_typed_edit.setPlaceholderText("e.g. 0, 0.4, 0.8, 1.2  (10m = 0.01)")
        self.curve_typed_edit.setToolTip("One value per point, comma- or space-separated. A "
                                         "Bruker suffix (10m, 500u) is honoured.")
        self.curve_typed_label = QLabel("List:")
        form.addRow(self.curve_typed_label, self.curve_typed_edit)

        file_row = QHBoxLayout()
        self.curve_list_edit = QLineEdit()
        self.curve_list_edit.setPlaceholderText("a one-column file of numbers")
        self.curve_list_browse = QPushButton("Browse…")
        self.curve_list_browse.clicked.connect(self._on_curve_browse_list)
        file_row.addWidget(self.curve_list_edit, 1)
        file_row.addWidget(self.curve_list_browse)
        self.curve_list_label = QLabel("List file:")
        form.addRow(self.curve_list_label, file_row)
        self.curve_factor_spin = QDoubleSpinBox()
        self.curve_factor_spin.setRange(-1e12, 1e12)
        self.curve_factor_spin.setDecimals(6)
        self.curve_factor_spin.setValue(1.0)
        self.curve_factor_spin.setToolTip("Every value in the list file is multiplied by this.")
        self.curve_factor_label = QLabel("Factor:")
        form.addRow(self.curve_factor_label, self.curve_factor_spin)

        self.curve_start_spin = QDoubleSpinBox()
        self.curve_step_spin = QDoubleSpinBox()
        for spin in (self.curve_start_spin, self.curve_step_spin):
            spin.setRange(-1e12, 1e12)
            spin.setDecimals(6)
        self.curve_step_spin.setValue(1.0)
        self.curve_stop_edit = QLineEdit()
        self.curve_stop_edit.setPlaceholderText("optional")
        self.curve_points_edit = QLineEdit()
        self.curve_points_edit.setPlaceholderText("optional (default: one per point)")
        self.curve_multiplier_spin = QDoubleSpinBox()
        self.curve_multiplier_spin.setRange(0.0001, 10000.0)
        self.curve_multiplier_spin.setDecimals(4)
        self.curve_multiplier_spin.setValue(1.0)
        self.curve_multiplier_spin.setToolTip(
            "Each step is this many times the one before: 1 gives equal steps; 2 with start "
            "0.1, step 0.1 gives 0.1, 0.2, 0.4, 0.8 …")
        self.curve_stop_edit.setToolTip(
            "Stop alone: values from Start up to and including Stop. Stop and Points: that many "
            "values from Start to exactly Stop (the step is worked out). Neither: one value per "
            "point of the file.")
        self.curve_points_edit.setToolTip(self.curve_stop_edit.toolTip())
        self._curve_step_rows = []
        for text, widget in (("Start:", self.curve_start_spin), ("Step:", self.curve_step_spin),
                             ("Stop:", self.curve_stop_edit), ("Points:", self.curve_points_edit),
                             ("Step multiplier:", self.curve_multiplier_spin)):
            label = QLabel(text)
            form.addRow(label, widget)
            self._curve_step_rows.append((label, widget))

        self.curve_xname_edit = QLineEdit("t")
        self.curve_xname_edit.setToolTip("Short name for x: the column heading in the saved CSV.")
        form.addRow("Quantity name:", self.curve_xname_edit)
        self.curve_xlabel_edit = QLineEdit()
        self.curve_xlabel_edit.setPlaceholderText("default: t (ms) from the file, else the name")
        self.curve_xlabel_edit.setToolTip("The x-axis text. Symbols such as t or NτR are "
                                          "italicised.")
        form.addRow("Axis label:", self.curve_xlabel_edit)
        self.curve_part_combo = QComboBox()
        for part in ("real", "imag", "magnitude"):
            self.curve_part_combo.addItem(part, part)
        form.addRow("Part:", self.curve_part_combo)
        self.curve_normalise_combo = QComboBox()
        for key, text in (("", "As stored"), ("first", "Divide by the first point (S/S₀)"),
                          ("max", "Divide by the largest value")):
            self.curve_normalise_combo.addItem(text, key)
        form.addRow("Scale:", self.curve_normalise_combo)
        self.curve_ylabel_edit = QLineEdit()
        self.curve_ylabel_edit.setPlaceholderText("default: Intensity, or S/S₀")
        form.addRow("y label:", self.curve_ylabel_edit)
        v.addLayout(form)
        self.curve_preview_label = QLabel("")
        self.curve_preview_label.setWordWrap(True)
        v.addWidget(self.curve_preview_label)
        row = QHBoxLayout()
        self.plot_curve_button = QPushButton("Plot curve")
        self.plot_curve_button.clicked.connect(self._on_plot_curve)
        self.save_curve_button = QPushButton("Save curve (CSV)…")
        self.save_curve_button.clicked.connect(self._on_save_curve)
        self.copy_curve_code_button = QPushButton("Copy notebook code")
        self.copy_curve_code_button.clicked.connect(
            lambda: (QGuiApplication.clipboard().setText(self.curve_notebook_code()),
                     self._say("Notebook code copied.")))
        for b in (self.plot_curve_button, self.save_curve_button, self.copy_curve_code_button):
            row.addWidget(b)
        v.addLayout(row)
        self._sync_curve_controls()
        return box

    def _sync_curve_controls(self) -> None:
        mode = self.curve_x_combo.currentData()
        for widget, label, shown in (
                (self.curve_time_unit_combo, self.curve_time_unit_label, mode == "file"),
                (self.curve_typed_edit, self.curve_typed_label, mode == "typed"),
                (self.curve_factor_spin, self.curve_factor_label, mode == "list")):
            widget.setVisible(shown)
            label.setVisible(shown)
        for w in (self.curve_list_edit, self.curve_list_browse):
            w.setVisible(mode == "list")
        self.curve_list_label.setVisible(mode == "list")
        for label, widget in self._curve_step_rows:
            label.setVisible(mode == "steps")
            widget.setVisible(mode == "steps")

    def _on_curve_browse_list(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "List file", "", "All files (*)")
        if path:
            self.curve_list_edit.setText(path)

    @staticmethod
    def _optional_number(edit: QLineEdit, name: str, integer: bool = False):
        text = edit.text().strip().replace(",", ".")
        if not text:
            return None
        try:
            return int(text) if integer else float(text)
        except ValueError:
            raise ValueError(f"{name}: {text!r} is not {'a whole number' if integer else 'a number'}.") from None

    def curve_kwargs(self) -> dict:
        """The :func:`opennmr.notebook.simpson_series.SIMPSONcurve` arguments
        (beside the path) these controls describe."""
        mode = self.curve_x_combo.currentData()
        kwargs: dict = {"xname": self.curve_xname_edit.text().strip() or "x",
                        "xlabel": self.curve_xlabel_edit.text().strip() or None}
        if mode in ("file", "index"):
            kwargs["x"] = mode
            if mode == "file":
                kwargs["time_unit"] = self.curve_time_unit_combo.currentData()
        elif mode == "typed":
            kwargs["x"] = self.curve_typed_edit.text()
        elif mode == "list":
            path = self.curve_list_edit.text().strip()
            if not path or not os.path.isfile(path):
                raise ValueError("Choose the list file (Browse…).")
            kwargs["x"] = path
            kwargs["factor"] = float(self.curve_factor_spin.value())
        else:
            kwargs.update(x="steps", start=float(self.curve_start_spin.value()),
                          step=float(self.curve_step_spin.value()),
                          multiplier=float(self.curve_multiplier_spin.value()),
                          stop=self._optional_number(self.curve_stop_edit, "Stop"),
                          points=self._optional_number(self.curve_points_edit, "Points",
                                                       integer=True))
        if kwargs["xlabel"] is None and mode != "file":
            kwargs["xlabel"] = kwargs["xname"]
        return kwargs

    def read_curve(self) -> dict:
        paths = self._chosen_paths()
        if not paths:
            raise ValueError("Add the file first.")
        curve = simpson_series.SIMPSONcurve(paths[0], **self.curve_kwargs())
        self.curve = curve
        return curve

    def _checked_curve(self):
        try:
            curve = self.read_curve()
            normalise = self.curve_normalise_combo.currentData() or None
            simpson_series.curve_y(curve, self.curve_part_combo.currentData(), normalise)
        except Exception as exc:  # noqa: BLE001 -- shown to the user
            QMessageBox.warning(self, "SIMPSON curve", str(exc))
            self.curve_preview_label.setText(str(exc))
            return None
        x = curve["x"]
        shown = ", ".join(f"{v:g}" for v in x[:6]) + (" …" if x.size > 6 else "")
        text = (f"{curve['n']} points; x from {curve['x_source']}: {shown}"
                + (f" ({curve['x_unused']} extra value(s) not used)" if curve["x_unused"] else ""))
        if curve["rate_Hz"] and curve["x_source"] != "point number":
            text += f". Rate in the file: {curve['rate_Hz']:g} Hz."
        self.curve_preview_label.setText(text)
        return curve

    def _on_plot_curve(self) -> None:
        curve = self._checked_curve()
        if curve is None:
            return
        part = self.curve_part_combo.currentData()
        normalise = self.curve_normalise_combo.currentData() or None
        ylabel = self.curve_ylabel_edit.text().strip() or None

        def draw(fig):
            fig.clear()
            ax = fig.add_subplot(111)
            simpson_series.SIMPSONplot_curve(curve, part=part, normalise=normalise, ax=ax,
                                             ylabel=ylabel)
            return ax

        self._draw_plot(draw)
        self._say(f"Plotted {os.path.basename(curve['path'])} as a curve of {curve['n']} "
                  "points.")

    def _on_save_curve(self) -> None:
        curve = self._checked_curve()
        if curve is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save curve", "curve.csv", "CSV (*.csv)")
        if path:
            simpson_series.SIMPSONsave_curve(curve, path, part=self.curve_part_combo.currentData(),
                                             normalise=self.curve_normalise_combo.currentData()
                                             or None)
            self._say(f"Saved {path}.")

    def curve_notebook_code(self) -> str:
        paths = self._chosen_paths()
        try:
            kwargs = self.curve_kwargs()
        except ValueError:
            kwargs = {"x": "file"}
        args = ", ".join(f"{k}={v!r}" for k, v in kwargs.items() if v is not None)
        normalise = self.curve_normalise_combo.currentData() or None
        return "\n".join([
            "import opennmr.notebook as nmr", "",
            f"curve = nmr.SIMPSONcurve({(paths[0] if paths else 'file.fid')!r}, {args})",
            f"fig, ax = nmr.SIMPSONplot_curve(curve, part={self.curve_part_combo.currentData()!r}, "
            f"normalise={normalise!r})"])

    # ------------------------------------------------------------------
    # 6. Heatmap of two swept quantities
    # ------------------------------------------------------------------
    HEATMAP_SOURCES = (("table", "Table file (one row per combination)"),
                       ("files", "FID files (two numbers in each name)"))
    HEATMAP_CMAPS = ("viridis", "cividis", "magma", "plasma", "inferno", "RdBu_r", "coolwarm",
                     "Greys")

    def _build_heatmap_box(self) -> QGroupBox:
        box = QGroupBox("6. Heatmap (two swept quantities)")
        v = QVBoxLayout(box)
        note = QLabel("A value for every combination of two swept quantities, as colour. From a "
                      "table (the selected file, or the first: columns such as O1 O2 Re, with or "
                      "without a header line) or from the FIDs in the list, one per "
                      "combination, with the two values in each file name.")
        note.setWordWrap(True)
        v.addWidget(note)
        form = QFormLayout()
        self.heatmap_source_combo = QComboBox()
        for key, text in self.HEATMAP_SOURCES:
            self.heatmap_source_combo.addItem(text, key)
        self.heatmap_source_combo.currentIndexChanged.connect(
            lambda _i: self._sync_heatmap_controls())
        form.addRow("Data:", self.heatmap_source_combo)

        self.heatmap_x_combo, self.heatmap_y_combo, self.heatmap_z_combo = (
            QComboBox(), QComboBox(), QComboBox())
        self._heatmap_table_rows = []
        self.heatmap_skip_spin = TypedSpinBox()
        self.heatmap_skip_spin.setRange(-1, 100000)
        self.heatmap_skip_spin.setValue(-1)
        self.heatmap_skip_spin.setSpecialValueText("auto")
        self.heatmap_skip_spin.setToolTip(
            "Lines at the top of the file to pass over (a title, notes, settings). auto: the "
            "data are the rows of numbers running to the end of the file, and everything above "
            "them is passed over. The line under the plot says what was found.")
        self.heatmap_header_combo = QComboBox()
        for key, text in (("auto", "Auto-detect"), ("names", "Names the columns"),
                          ("data", "Is data (no names)")):
            self.heatmap_header_combo.addItem(text, key)
        self.heatmap_header_combo.setToolTip(
            "Whether the first row (after any skipped lines – with auto, the line just above the "
            "numbers) names the columns. Auto-detect: yes when it holds one name per column, "
            "e.g. O1 O2 Re Im, # O1 O2 Re Im, or tab-separated names with spaces in them. "
            "Without names the columns are c1, c2 …")
        for widget in (self.heatmap_skip_spin, self.heatmap_header_combo):
            signal = (widget.valueChanged if widget is self.heatmap_skip_spin
                      else widget.currentIndexChanged)
            signal.connect(lambda *_a: self.paths() and self._on_heatmap_read_columns())
        for text, widget in (("Lines to skip:", self.heatmap_skip_spin),
                             ("First row:", self.heatmap_header_combo)):
            label = QLabel(text)
            form.addRow(label, widget)
            self._heatmap_table_rows.append((label, widget))
        for text, combo in (("x column:", self.heatmap_x_combo),
                            ("y column:", self.heatmap_y_combo),
                            ("Colour column:", self.heatmap_z_combo)):
            combo.setToolTip("Filled from the table's header (or c1, c2 … without one) – press "
                             "Read columns after choosing the file.")
            label = QLabel(text)
            form.addRow(label, combo)
            self._heatmap_table_rows.append((label, combo))
        self.heatmap_read_button = QPushButton("Read columns")
        self.heatmap_read_button.clicked.connect(self._on_heatmap_read_columns)
        label = QLabel("")
        form.addRow(label, self.heatmap_read_button)
        self._heatmap_table_rows.append((label, self.heatmap_read_button))

        self.heatmap_x_number_spin = TypedSpinBox()
        self.heatmap_y_number_spin = TypedSpinBox()
        for spin, value in ((self.heatmap_x_number_spin, -2), (self.heatmap_y_number_spin, -1)):
            spin.setRange(-20, 20)
            spin.setValue(value)
            spin.setToolTip("Which number in each file name: 0 is the first, 1 the second …; "
                            "-1 is the last, -2 the one before. In O1_10_O2_-5.fid the numbers "
                            "are 1, 10, 2, -5, so x = -3 (10) and y = -1 (-5).")
        self.heatmap_expression_edit = QLineEdit("A")
        self.heatmap_expression_edit.setToolTip("What each FID contributes: A = its first "
                                                "point, or A + B - 0.3*C as in section 4.")
        self.heatmap_part_combo = QComboBox()
        for part in simpson_series.PARTS:
            self.heatmap_part_combo.addItem(part, part)
        self._heatmap_file_rows = []
        for text, widget in (("x = number:", self.heatmap_x_number_spin),
                             ("y = number:", self.heatmap_y_number_spin),
                             ("Value:", self.heatmap_expression_edit),
                             ("Part:", self.heatmap_part_combo)):
            label = QLabel(text)
            form.addRow(label, widget)
            self._heatmap_file_rows.append((label, widget))

        self.heatmap_xlabel_edit = QLineEdit()
        self.heatmap_xlabel_edit.setPlaceholderText("default: the column / x")
        self.heatmap_ylabel_edit = QLineEdit()
        self.heatmap_ylabel_edit.setPlaceholderText("default: the column / y")
        self.heatmap_zlabel_edit = QLineEdit()
        self.heatmap_zlabel_edit.setPlaceholderText("default: the column / value")
        form.addRow("x label:", self.heatmap_xlabel_edit)
        form.addRow("y label:", self.heatmap_ylabel_edit)
        form.addRow("Colour-bar label:", self.heatmap_zlabel_edit)
        self.heatmap_cmap_combo = QComboBox()
        for name in self.HEATMAP_CMAPS:
            self.heatmap_cmap_combo.addItem(name, name)
        form.addRow("Colour map:", self.heatmap_cmap_combo)
        range_row = QHBoxLayout()
        self.heatmap_vmin_edit = QLineEdit()
        self.heatmap_vmin_edit.setPlaceholderText("auto")
        self.heatmap_vmax_edit = QLineEdit()
        self.heatmap_vmax_edit.setPlaceholderText("auto")
        range_row.addWidget(self.heatmap_vmin_edit)
        range_row.addWidget(QLabel("to"))
        range_row.addWidget(self.heatmap_vmax_edit)
        for edit in (self.heatmap_vmin_edit, self.heatmap_vmax_edit):
            edit.setToolTip("Fix the colour range (e.g. the same for several maps); empty: "
                            "the data's own range.")
        form.addRow("Colour range:", range_row)
        v.addLayout(form)
        self.heatmap_preview_label = QLabel("")
        self.heatmap_preview_label.setWordWrap(True)
        v.addWidget(self.heatmap_preview_label)
        row = QHBoxLayout()
        self.plot_heatmap_button = QPushButton("Plot heatmap")
        self.plot_heatmap_button.clicked.connect(self._on_plot_heatmap)
        self.save_heatmap_button = QPushButton("Save grid (CSV)…")
        self.save_heatmap_button.clicked.connect(self._on_save_heatmap)
        self.copy_heatmap_code_button = QPushButton("Copy notebook code")
        self.copy_heatmap_code_button.clicked.connect(
            lambda: (QGuiApplication.clipboard().setText(self.heatmap_notebook_code()),
                     self._say("Notebook code copied.")))
        for b in (self.plot_heatmap_button, self.save_heatmap_button,
                  self.copy_heatmap_code_button):
            row.addWidget(b)
        v.addLayout(row)
        self._sync_heatmap_controls()
        return box

    def _sync_heatmap_controls(self) -> None:
        table = self.heatmap_source_combo.currentData() == "table"
        for label, widget in self._heatmap_table_rows:
            label.setVisible(table)
            widget.setVisible(table)
        for label, widget in self._heatmap_file_rows:
            label.setVisible(not table)
            widget.setVisible(not table)

    def heatmap_table_kwargs(self) -> dict:
        """``skip_rows`` and ``header`` for the table readers."""
        skip = self.heatmap_skip_spin.value()
        header = {"auto": "auto", "names": True, "data": False}[
            self.heatmap_header_combo.currentData()]
        return {"skip_rows": None if skip < 0 else skip, "header": header}

    def _on_heatmap_read_columns(self) -> list[str] | None:
        paths = self._chosen_paths()
        if not paths:
            self._say("Add the table file first.")
            return None
        try:
            table = simpson_series.SIMPSONread_scan(paths[0], **self.heatmap_table_kwargs())
        except Exception as exc:  # noqa: BLE001 -- shown to the user
            self.heatmap_preview_label.setText(str(exc))
            for combo in (self.heatmap_x_combo, self.heatmap_y_combo, self.heatmap_z_combo):
                combo.clear()
            return None
        names = table["names"]
        if len(names) < 3:
            self.heatmap_preview_label.setText(
                f"{os.path.basename(paths[0])} has {len(names)} column(s); a heatmap needs x, y "
                "and a value.")
            return None
        for k, combo in enumerate((self.heatmap_x_combo, self.heatmap_y_combo,
                                   self.heatmap_z_combo)):
            keep = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(names)
            combo.setCurrentIndex(names.index(keep) if keep in names else k)
            combo.blockSignals(False)
        found = []
        if table["skipped"]:
            found.append(f"passed over {table['skipped']} line(s) at the top")
        found.append(f"names from line {table['header_line']}" if table["header_line"]
                     else "no names in the file")
        found.append(f"data from line {table['first_data_line']}")
        self.heatmap_preview_label.setText(
            f"Columns: {', '.join(names)} ({'; '.join(found)}; "
            f"{len(table['columns'][names[0]])} rows).")
        return names

    @staticmethod
    def _optional_float(edit: QLineEdit, name: str):
        text = edit.text().strip().replace(",", ".")
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            raise ValueError(f"{name}: {text!r} is not a number.") from None

    def read_heatmap(self) -> dict:
        paths = self._chosen_paths()
        if not paths:
            raise ValueError("Add the file(s) first.")
        if self.heatmap_source_combo.currentData() == "table":
            if self.heatmap_x_combo.count() == 0 and self._on_heatmap_read_columns() is None:
                raise ValueError(self.heatmap_preview_label.text() or "Could not read the columns.")
            grid = simpson_series.SIMPSONscan_from_table(
                paths[0], self.heatmap_x_combo.currentText(), self.heatmap_y_combo.currentText(),
                self.heatmap_z_combo.currentText(), **self.heatmap_table_kwargs())
        else:
            grid = simpson_series.SIMPSONscan_from_files(
                paths, x_number=self.heatmap_x_number_spin.value(),
                y_number=self.heatmap_y_number_spin.value(),
                expression=self.heatmap_expression_edit.text().strip() or "A",
                part=self.heatmap_part_combo.currentData(), kind=self.read_kwargs().get("kind"))
        self.grid = grid
        return grid

    def _heatmap_style(self) -> dict:
        return {"xlabel": self.heatmap_xlabel_edit.text().strip() or None,
                "ylabel": self.heatmap_ylabel_edit.text().strip() or None,
                "zlabel": self.heatmap_zlabel_edit.text().strip() or None,
                "cmap": self.heatmap_cmap_combo.currentData(),
                "vmin": self._optional_float(self.heatmap_vmin_edit, "Colour range minimum"),
                "vmax": self._optional_float(self.heatmap_vmax_edit, "Colour range maximum")}

    def _checked_heatmap(self):
        try:
            grid = self.read_heatmap()
            style = self._heatmap_style()
            if grid["xs"].size < 2 or grid["ys"].size < 2:
                raise ValueError("Both swept quantities need at least two values for a heatmap.")
        except Exception as exc:  # noqa: BLE001 -- shown to the user
            QMessageBox.warning(self, "SIMPSON heatmap", str(exc))
            self.heatmap_preview_label.setText(str(exc))
            return None, None
        text = (f"{grid['xs'].size} {grid['xname']} × {grid['ys'].size} {grid['yname']} values"
                f" ({grid['xs'].min():g} to {grid['xs'].max():g}; "
                f"{grid['ys'].min():g} to {grid['ys'].max():g}).")
        if grid["missing"]:
            text += (f" {grid['missing']} combination(s) not in the data: left blank, not "
                     "filled in.")
        self.heatmap_preview_label.setText(text)
        return grid, style

    def _on_plot_heatmap(self) -> None:
        grid, style = self._checked_heatmap()
        if grid is None:
            return

        def draw(fig):
            fig.clear()
            ax = fig.add_subplot(111)
            simpson_series.SIMPSONplot_heatmap(grid, ax=ax, **style)
            return ax

        self._draw_plot(draw)
        self._say(f"Plotted a heatmap of {grid['zname']} over {grid['xname']} and "
                  f"{grid['yname']}.")

    def _on_save_heatmap(self) -> None:
        grid, _style = self._checked_heatmap()
        if grid is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save grid", "heatmap.csv", "CSV (*.csv)")
        if path:
            simpson_series.SIMPSONscan_table(grid).to_csv(path, index=False)
            self._say(f"Saved {path}.")

    def heatmap_notebook_code(self) -> str:
        paths = self._chosen_paths()
        try:
            style = self._heatmap_style()
        except ValueError:
            style = {}
        style_args = ", ".join(f"{k}={v!r}" for k, v in style.items() if v is not None)
        if self.heatmap_source_combo.currentData() == "table":
            read = (f"grid = nmr.SIMPSONscan_from_table({(paths[0] if paths else 'scan.txt')!r}, "
                    f"{self.heatmap_x_combo.currentText() or 0!r}, "
                    f"{self.heatmap_y_combo.currentText() or 1!r}, "
                    f"{self.heatmap_z_combo.currentText() or 2!r}"
                    + "".join(f", {k}={v!r}" for k, v in self.heatmap_table_kwargs().items()
                              if v not in (None, "auto")) + ")")
            lines = ["import opennmr.notebook as nmr", "", read]
        else:
            lines = ["import opennmr.notebook as nmr", "", f"files = {paths!r}",
                     "grid = nmr.SIMPSONscan_from_files(files, "
                     f"x_number={self.heatmap_x_number_spin.value()}, "
                     f"y_number={self.heatmap_y_number_spin.value()}, "
                     f"expression={self.heatmap_expression_edit.text().strip() or 'A'!r}, "
                     f"part={self.heatmap_part_combo.currentData()!r})"]
        lines.append(f"fig, ax = nmr.SIMPSONplot_heatmap(grid{', ' + style_args if style_args else ''})")
        return "\n".join(lines)

    def notebook_code(self) -> str:
        paths = self.paths()
        kwargs = self.read_kwargs()
        try:
            xs = self.x_values()
        except ValueError:
            xs = []
        lines = ["import opennmr.notebook as nmr", "", f"files = {paths!r}"]
        lines.append(f"x = {xs!r}")
        extra = f", kind={kwargs['kind']!r}" if "kind" in kwargs else ""
        lines.append("series = nmr.SIMPSONpoint_series(files, x=x, "
                     f"xname={self.x_name_edit.text().strip() or 'x'!r}, "
                     f"xlabel={(self.x_label_edit.text().strip() or None)!r}, "
                     f"n_points={self.n_points_spin.value()}{extra})")
        lines.append(f"fig, ax = nmr.SIMPSONplot_point_series(series, {self.expressions()!r}, "
                     f"part={self.part_combo.currentData()!r}, "
                     f"normalise={self.normalise_check.isChecked()})")
        return "\n".join(lines)

    def _on_copy_code(self) -> None:
        QGuiApplication.clipboard().setText(self.notebook_code())
        self._say("Notebook code copied.")

    # ------------------------------------------------------------------
    def _say(self, text: str) -> None:
        self.status_label.setText(text)

    def _refresh_state(self) -> None:
        has = self.file_list.count() > 0
        for b in (self.plot_button, self.open_button, self.plot_points_button,
                  self.save_points_button, self.copy_code_button, self.plot_curve_button,
                  self.save_curve_button, self.copy_curve_code_button, self.plot_heatmap_button,
                  self.save_heatmap_button, self.copy_heatmap_code_button,
                  self.heatmap_read_button):
            b.setEnabled(has)

    # ------------------------------------------------------------------
    # Saved state
    # ------------------------------------------------------------------
    def get_state(self) -> dict:
        return {"widgets": panel_state.capture_widgets(self), "paths": self.paths(),
                "x": [self.x_table.item(r, 1).text() for r in range(self.x_table.rowCount())]}

    def set_state(self, state: dict) -> bool:
        state = state or {}
        self.file_list.clear()
        self.add_paths([p for p in state.get("paths", []) if os.path.exists(p)])
        panel_state.restore_widgets(self, state.get("widgets") or {})
        self._sync_x_table()
        self._sync_curve_controls()
        self._sync_heatmap_controls()
        if (self.heatmap_source_combo.currentData() == "table" and self.paths()
                and self._on_heatmap_read_columns() is not None):
            # The column choices can only be put back once the columns are listed.
            panel_state.restore_widgets(self, state.get("widgets") or {})
        if self.x_source_combo.currentData() == "typed":
            for r, text in enumerate(state.get("x") or []):
                if r < self.x_table.rowCount():
                    self.x_table.item(r, 1).setText(text)
        return True
