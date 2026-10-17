"""Integrate ▸ Offset Tracker: a series of 1D spectra integrated over one
region, plotted against the offset (O2 or O3, in ppm) of the channel
pulsed while they were recorded -- 1D sideband-selective HMQC or
J-resolved data (explicit user request, 2026-10-17). The maths is
:mod:`opennmr.notebook.offset_tracker`, so a notebook gets the same
numbers.

Laid out like CP Build-up: 1. spectra and their offsets (read from O2/O3
or typed), 2. the region (typed, or dragged on any of the spectra), 3.
integrate and plot (bars, stems or triangles), and keep the bars'
skyline as a spectrum in Open experiments, to fit in Fit like any other.
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QFormLayout, QGroupBox, QHBoxLayout, QHeaderView,
    QLabel, QLineEdit, QMessageBox, QPushButton, QSpinBox, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)

from ..notebook import offset_tracker as ot
from .widgets.report_label import ReportLabel

_C_USE, _C_NAME, _C_OFFSET = 0, 1, 2


def _number(text: str):
    try:
        return float(str(text).strip().replace("−", "-"))
    except (TypeError, ValueError):
        return None


class OffsetTrackerPanel(QWidget):
    """``spectra()`` -> [{"label", "spectrum", "index"}] (Open experiments'
    1D spectra); ``draw(draw_fn, view)`` puts a figure on the Fit view
    ("profile") or Compare; ``arm(mode)`` turns region dragging on Compare
    on ("region") or off (None); ``show_spectrum(spectrum)`` puts one
    spectrum on Compare; ``highlight(regions)`` shades regions there;
    ``keep(spectrum, label, description)`` adds a spectrum to Open
    experiments."""

    def __init__(self, spectra, draw, arm, show_spectrum=None, highlight=None, keep=None,
                 parent=None) -> None:
        super().__init__(parent)
        self._spectra, self._draw, self._arm = spectra, draw, arm
        self._show_spectrum, self._highlight, self._keep = show_spectrum, highlight, keep
        self._entries: list[dict] = []
        self._checked: set[int] = set()
        self._offsets: dict[int, str] = {}
        self.armed = False
        self.series: dict | None = None

        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        intro = QLabel(
            "For 1D spectra of one nucleus recorded while pulsing on another at a stepped "
            "offset (sideband-selective HMQC, J-resolved): integrate the same region of every "
            "spectrum and plot it against the pulsed channel's offset (O2 or O3) in ppm. The "
            "bars' outline can be kept as a spectrum of the pulsed nucleus and fitted.")
        intro.setWordWrap(True)
        v.addWidget(intro)
        v.addWidget(self._build_spectra())
        v.addWidget(self._build_region())
        v.addWidget(self._build_plot())
        self.refresh()

    # ------------------------------------------------------------------
    def _build_spectra(self) -> QGroupBox:
        box = QGroupBox("1. Spectra and offsets")
        v = QVBoxLayout(box)
        self.spectra_table = QTableWidget(0, 3)
        self.spectra_table.setHorizontalHeaderLabels(["Use", "Spectrum", "Offset (ppm)"])
        header = self.spectra_table.horizontalHeader()
        header.setSectionResizeMode(_C_USE, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(_C_NAME, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(_C_OFFSET, QHeaderView.ResizeMode.ResizeToContents)
        self.spectra_table.verticalHeader().setVisible(False)
        self.spectra_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.spectra_table.setMinimumHeight(140)
        self.spectra_table.setToolTip("Tick the spectra of the series; Read Offsets fills the "
                                      "offsets from O2/O3, or type them (ppm of the pulsed "
                                      "nucleus).")
        self.spectra_table.itemChanged.connect(self._on_item_changed)
        v.addWidget(self.spectra_table)
        ticks = QHBoxLayout()
        self.tick_all_button = QPushButton("Tick All")
        self.tick_all_button.clicked.connect(
            lambda: self.set_checked(range(self.spectra_table.rowCount())))
        self.untick_all_button = QPushButton("Untick All")
        self.untick_all_button.clicked.connect(lambda: self.set_checked([]))
        ticks.addWidget(self.tick_all_button)
        ticks.addWidget(self.untick_all_button)
        ticks.addStretch(1)
        v.addLayout(ticks)
        row = QHBoxLayout()
        row.addWidget(QLabel("Offset from:"))
        self.channel_combo = QComboBox()
        for key in ot.CHANNELS:
            self.channel_combo.addItem(key, key)
        self.channel_combo.setToolTip(
            "The channel whose offset was stepped: O2 (f2) or O3 (f3) -- whichever the pulse "
            "programme pulses the second nucleus on. Read as O/BF in ppm of that channel's "
            "nucleus.")
        row.addWidget(self.channel_combo)
        self.read_button = QPushButton("Read Offsets")
        self.read_button.clicked.connect(lambda _c=False: self.read_offsets())
        row.addWidget(self.read_button)
        row.addStretch(1)
        v.addLayout(row)
        shift_row = QHBoxLayout()
        shift_row.addWidget(QLabel("Add to every offset:"))
        self.shift_edit = QLineEdit("0")
        self.shift_edit.setMaximumWidth(80)
        self.shift_edit.setToolTip(
            "ppm added to every offset -- when the pulsed nucleus's 0 ppm is not at its "
            "channel's basic frequency (BF), e.g. a non-standard reference.")
        shift_row.addWidget(self.shift_edit)
        shift_row.addWidget(QLabel("ppm"))
        shift_row.addStretch(1)
        v.addLayout(shift_row)
        self.spectra_status = QLabel("")
        self.spectra_status.setWordWrap(True)
        v.addWidget(self.spectra_status)
        return box

    def _build_region(self) -> QGroupBox:
        box = QGroupBox("2. Region (observed nucleus)")
        v = QVBoxLayout(box)
        form = QFormLayout()
        region = QHBoxLayout()
        self.from_edit = QLineEdit()
        self.to_edit = QLineEdit()
        for edit in (self.from_edit, self.to_edit):
            edit.setPlaceholderText("ppm")
            edit.setMaximumWidth(90)
            edit.editingFinished.connect(self._show_region)
        region.addWidget(self.from_edit)
        region.addWidget(QLabel("to"))
        region.addWidget(self.to_edit)
        region.addWidget(QLabel("ppm"))
        region.addStretch(1)
        form.addRow("Integrate:", region)
        self.drag_on_combo = QComboBox()
        self.drag_on_combo.setToolTip("The spectrum shown to drag the region on -- any of them.")
        form.addRow("Drag on:", self.drag_on_combo)
        v.addLayout(form)
        buttons = QHBoxLayout()
        self.drag_button = QPushButton("Drag on Plot")
        self.drag_button.setCheckable(True)
        self.drag_button.setToolTip("Shows the spectrum chosen in Drag on; drag across it to "
                                    "set the region.")
        self.drag_button.toggled.connect(self._on_drag_toggled)
        buttons.addWidget(self.drag_button)
        buttons.addStretch(1)
        v.addLayout(buttons)
        opts = QFormLayout()
        self.baseline_combo = QComboBox()
        self.baseline_combo.addItem("Local (fitted just outside the region)", "local")
        self.baseline_combo.addItem("None (spectrum as it is)", "none")
        self.baseline_combo.addItem("Global (robust, whole spectrum)", "global")
        opts.addRow("Baseline:", self.baseline_combo)
        self.rg_check = QCheckBox("Divide by receiver gain (RG)")
        self.rg_check.setChecked(True)
        opts.addRow("", self.rg_check)
        self.normalise_check = QCheckBox("Relative to the largest")
        opts.addRow("", self.normalise_check)
        v.addLayout(opts)
        return box

    def _build_plot(self) -> QGroupBox:
        box = QGroupBox("3. Plot and keep")
        v = QVBoxLayout(box)
        form = QFormLayout()
        self.style_combo = QComboBox()
        for text, key in (("Bars", "bars"), ("Vertical lines", "stems"),
                          ("Vertical triangles", "triangles")):
            self.style_combo.addItem(text, key)
        self.style_combo.currentIndexChanged.connect(lambda _i: self.redraw())
        form.addRow("Draw as:", self.style_combo)
        self.skyline_check = QCheckBox("Show the skyline (the bars' outline)")
        self.skyline_check.toggled.connect(lambda _c: self.redraw())
        form.addRow("", self.skyline_check)
        self.points_spin = QSpinBox()
        self.points_spin.setRange(64, 65536)
        self.points_spin.setValue(2048)
        self.points_spin.setToolTip("Points of the kept spectrum's ppm grid.")
        self.shape_combo = QComboBox()
        self.shape_combo.addItem("Steps (the skyline)", "steps")
        self.shape_combo.addItem("Straight lines between offsets", "linear")
        keep_row = QHBoxLayout()
        keep_row.addWidget(self.shape_combo)
        keep_row.addWidget(self.points_spin)
        form.addRow("Kept spectrum:", keep_row)
        v.addLayout(form)
        buttons = QHBoxLayout()
        self.integrate_button = QPushButton("Integrate and Plot")
        self.integrate_button.clicked.connect(lambda _c=False: self.integrate())
        self.keep_button = QPushButton("Keep Skyline in Open Experiments")
        self.keep_button.setToolTip(
            "Add the skyline as a 1D spectrum of the pulsed nucleus to Open experiments, to "
            "fit in Fit (CSA, quadrupolar, ...) like any other spectrum.")
        self.keep_button.clicked.connect(lambda _c=False: self.keep_skyline())
        buttons.addWidget(self.integrate_button)
        buttons.addWidget(self.keep_button)
        buttons.addStretch(1)
        v.addLayout(buttons)
        self.result_label = ReportLabel("")
        self.result_label.setWordWrap(True)
        v.addWidget(self.result_label)
        return box

    # ------------------------------------------------------------------
    def refresh(self) -> None:
        """Re-read Open experiments, keeping ticks and typed offsets."""
        self._entries = list(self._spectra() or [])
        t = self.spectra_table
        t.blockSignals(True)
        try:
            t.setRowCount(len(self._entries))
            for r, e in enumerate(self._entries):
                key = id(e["spectrum"])
                use = QTableWidgetItem("")
                use.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
                use.setCheckState(Qt.CheckState.Checked if key in self._checked
                                  else Qt.CheckState.Unchecked)
                t.setItem(r, _C_USE, use)
                name = QTableWidgetItem(str(e["label"]))
                name.setFlags(Qt.ItemFlag.ItemIsEnabled)
                t.setItem(r, _C_NAME, name)
                t.setItem(r, _C_OFFSET, QTableWidgetItem(self._offsets.get(key, "")))
        finally:
            t.blockSignals(False)
        chosen = self.drag_on_combo.currentData()
        self.drag_on_combo.blockSignals(True)
        self.drag_on_combo.clear()
        for e in self._entries:
            self.drag_on_combo.addItem(str(e["label"]), id(e["spectrum"]))
        i = self.drag_on_combo.findData(chosen)
        self.drag_on_combo.setCurrentIndex(i if i >= 0 else 0)
        self.drag_on_combo.blockSignals(False)

    def _on_item_changed(self, item) -> None:
        r = item.row()
        if not (0 <= r < len(self._entries)):
            return
        key = id(self._entries[r]["spectrum"])
        if item.column() == _C_USE:
            if item.checkState() == Qt.CheckState.Checked:
                self._checked.add(key)
            else:
                self._checked.discard(key)
        elif item.column() == _C_OFFSET:
            self._offsets[key] = item.text().strip()

    def set_checked(self, rows) -> None:
        rows = set(rows)
        for r in range(self.spectra_table.rowCount()):
            self.spectra_table.item(r, _C_USE).setCheckState(
                Qt.CheckState.Checked if r in rows else Qt.CheckState.Unchecked)

    def set_offset(self, row: int, ppm) -> None:
        self.spectra_table.item(row, _C_OFFSET).setText(f"{float(ppm):.6g}")

    def set_region(self, lo: float, hi: float) -> None:
        self.from_edit.setText(f"{max(lo, hi):.6g}")
        self.to_edit.setText(f"{min(lo, hi):.6g}")
        self._show_region()

    def checked_rows(self) -> list[int]:
        return [r for r in range(self.spectra_table.rowCount())
                if self.spectra_table.item(r, _C_USE).checkState() == Qt.CheckState.Checked]

    def region(self):
        a, b = _number(self.from_edit.text()), _number(self.to_edit.text())
        if a is None or b is None or a == b:
            return None
        return (min(a, b), max(a, b))

    def read_offsets(self) -> int:
        """Fill the ticked rows' offsets from the chosen channel."""
        rows = self.checked_rows() or list(range(self.spectra_table.rowCount()))
        channel = self.channel_combo.currentData()
        n, problems = 0, []
        for r in rows:
            try:
                info = ot.channel_offset(self._entries[r]["spectrum"], channel)
            except ValueError as exc:
                problems.append(f"{self._entries[r]['label']}: {exc}")
                continue
            self.set_offset(r, info["ppm"])
            n += 1
        text = f"Read {n} offset(s) from {channel}."
        if problems:
            text += " " + " ".join(problems[:4]) + (" …" if len(problems) > 4 else "")
        self.spectra_status.setText(text)
        return n

    def _drag_entry(self):
        key = self.drag_on_combo.currentData()
        return next((e for e in self._entries if id(e["spectrum"]) == key), None)

    def _show_region(self) -> None:
        if self._highlight is not None:
            region = self.region()
            self._highlight([("region", region[0], region[1])] if region else [])

    def _on_drag_toggled(self, on: bool) -> None:
        if on:
            entry = self._drag_entry()
            if entry is None:
                self.drag_button.blockSignals(True)
                self.drag_button.setChecked(False)
                self.drag_button.blockSignals(False)
                QMessageBox.information(self, "Offset Tracker", "Open a 1D spectrum first.")
                return
            if self._show_spectrum is not None:
                self._show_spectrum(entry["spectrum"])
            self._show_region()
        self.armed = bool(on)
        self._arm("region" if on else None)

    def on_canvas_region(self, lo: float, hi: float) -> None:
        if lo == hi:
            return
        self.set_region(float(lo), float(hi))
        if self.armed:
            self._arm("region")

    def canvas_mode_released(self) -> None:
        if self.drag_button.isChecked():
            self.drag_button.blockSignals(True)
            self.drag_button.setChecked(False)
            self.drag_button.blockSignals(False)
        self.armed = False

    # ------------------------------------------------------------------
    def integrate(self) -> dict | None:
        rows = self.checked_rows()
        if len(rows) < 2:
            QMessageBox.warning(self, "Offset Tracker", "Tick at least two spectra.")
            return None
        region = self.region()
        if region is None:
            QMessageBox.warning(self, "Offset Tracker", "Give the region to integrate (two "
                                "different ppm values), or drag it on the plot.")
            return None
        offsets = []
        for r in rows:
            item = self.spectra_table.item(r, _C_OFFSET)
            offsets.append(_number(item.text()) if item and item.text().strip() else None)
        shift = _number(self.shift_edit.text()) or 0.0
        try:
            series = ot.offset_series(
                [self._entries[r]["spectrum"] for r in rows], region,
                channel=self.channel_combo.currentData(), offsets_ppm=offsets,
                labels=[self._entries[r]["label"] for r in rows],
                baseline=self.baseline_combo.currentData(), divide_rg=self.rg_check.isChecked(),
                ppm_shift=shift, normalise=self.normalise_check.isChecked())
        except ValueError as exc:
            QMessageBox.warning(self, "Offset Tracker", str(exc))
            return None
        self.series = series
        self.result_label.setText(ot.offset_report(series))
        self.redraw()
        return series

    def redraw(self) -> None:
        series = self.series
        if series is None:
            return
        style = self.style_combo.currentData()
        sky = self.skyline_check.isChecked()

        def draw(figure):
            ax = figure.add_subplot(111)
            ot.plot_offset_series(series, style=style, ax=ax, show_skyline=sky)
        self._draw(draw, "profile")

    def skyline(self):
        if self.series is None:
            return None
        return ot.skyline_spectrum(self.series, n_points=self.points_spin.value(),
                                   shape=self.shape_combo.currentData())

    def keep_skyline(self):
        if self.series is None and self.integrate() is None:
            return None
        try:
            spectrum = self.skyline()
        except ValueError as exc:
            QMessageBox.warning(self, "Offset Tracker", str(exc))
            return None
        if not spectrum.metadata.nucleus:
            # Offsets typed for data that do not record the channel: ask.
            from PySide6.QtWidgets import QInputDialog
            nucleus, ok = QInputDialog.getText(
                self, "Offset Tracker", "Nucleus of the pulsed channel (e.g. 1H, 27Al):")
            if not ok or not nucleus.strip():
                return None
            spectrum.metadata.nucleus = nucleus.strip()
        lo, hi = self.series["region"]
        label = (f"Skyline: {self.series.get('nucleus') or self.series['channel']} offsets, "
                 f"{hi:g} to {lo:g} ppm")
        description = (f"Offset Tracker skyline: integrals of {lo:g} to {hi:g} ppm in "
                       f"{len(self.series['y'])} spectra against the {self.series['channel']} "
                       f"offset ({self.shape_combo.currentText().lower()}, "
                       f"{self.points_spin.value()} points).")
        if self._keep is not None:
            self._keep(spectrum, label, description)
        self.result_label.setText(self.result_label.text()
                                  + f"\nKept as “{label}” in Open experiments.")
        return spectrum
