"""
Apply ▸ CP Build-up: integrate a region across a variable-contact-time
series and fit the CP kinetics (explicit user request, 2026-10-16). The
maths is :mod:`opennmr.notebook.cp_buildup` (models from Kolodziejski &
Klinowski, Chem. Rev. 2002, 102, 613), which a notebook calls the same way.

1. **Spectra** -- tick the series in Open experiments' 1D spectra and give
   each its contact time: typed in the table (µs), or read from a pulse
   programme parameter (``P15``, as VOCS reads ``P1``). Choose a
   reference (ideally direct excitation; any spectrum will do).
2. **Region** -- typed, or dragged on the plot (Show Reference first).
   Integrate draws the build-up curve on the Fit view.
3. **Fit** -- a model, starting values (read off the data, editable),
   any parameter fixed; the result with errors, the contact time of
   maximum signal, warnings and advice. Kept with the spectra (their log
   and sidecar) like any fit.
"""

from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox,
    QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMessageBox, QPushButton, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

from ..notebook import cp_buildup as cpb
from ..notebook import fit_advice
from .widgets.advice_box import AdviceBox
from .widgets.report_label import ReportLabel

_C_USE, _C_NAME, _C_TIME = range(3)


def _number(text: str):
    text = str(text).strip()
    if not text:
        return None
    try:
        v = float(text)
    except ValueError:
        return None
    return v if math.isfinite(v) else None


class CpBuildupPanel(QWidget):
    """Callbacks (from MainWindow), as the S/N panel's: ``spectra()`` ->
    ``[{"label", "spectrum", "index"}]``; ``draw(draw, view)`` with view
    ``"spectrum"`` (Compare) or ``"profile"`` (the Fit view, a curve);
    ``arm(mode)``: ``"region"`` drag on Compare, or None; ``store(kind,
    record, indices)``."""

    calculated = Signal()
    fitted = Signal()

    def __init__(self, spectra, draw, arm, store, style=None, parent=None) -> None:
        super().__init__(parent)
        self._spectra, self._draw, self._arm, self._store = spectra, draw, arm, store
        self._style = style or (lambda _view: {})
        self._entries: list[dict] = []
        #: id(spectrum) -> typed contact time text, kept across refreshes.
        self._times: dict[int, str] = {}
        self._checked: set[int] = set()
        self.armed = False
        self.series: dict | None = None
        self.fit: dict | None = None
        self._series_indices: list[int] = []

        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        intro = QLabel(
            "Cross-polarisation build-up: integrate one region in every spectrum of a "
            "variable-contact-time series, plot the integral against the contact time, and "
            "fit the CP kinetics (Kolodziejski & Klinowski, Chem. Rev. 2002, 102, 613).")
        intro.setWordWrap(True)
        v.addWidget(intro)
        v.addWidget(self._build_spectra())
        v.addWidget(self._build_region())
        v.addWidget(self._build_fit())
        self.refresh()

    # ------------------------------------------------------------------
    def _build_spectra(self) -> QGroupBox:
        box = QGroupBox("1. Spectra and contact times")
        v = QVBoxLayout(box)
        self.spectra_table = QTableWidget(0, 3)
        self.spectra_table.setHorizontalHeaderLabels(["Use", "Spectrum", "Contact time (µs)"])
        header = self.spectra_table.horizontalHeader()
        header.setSectionResizeMode(_C_USE, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(_C_NAME, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(_C_TIME, QHeaderView.ResizeMode.ResizeToContents)
        self.spectra_table.verticalHeader().setVisible(False)
        self.spectra_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.spectra_table.setMinimumHeight(140)
        self.spectra_table.setToolTip("Tick the spectra of the series and type each one's "
                                      "contact time in µs -- or read them from a parameter below.")
        self.spectra_table.itemChanged.connect(self._on_item_changed)
        v.addWidget(self.spectra_table)
        row = QHBoxLayout()
        row.addWidget(QLabel("Contact time from parameter:"))
        self.time_parameter_edit = QLineEdit("P15")
        self.time_parameter_edit.setMaximumWidth(90)
        self.time_parameter_edit.setToolTip(
            "The pulse-programme parameter holding the contact time, as written in the "
            "sequence (P15 on most Bruker CP programmes; pulses are in µs, D delays in s and "
            "converted). Or type a number here to give every ticked spectrum that time.")
        row.addWidget(self.time_parameter_edit)
        self.read_times_button = QPushButton("Read Contact Times")
        self.read_times_button.clicked.connect(self.read_contact_times)
        row.addWidget(self.read_times_button)
        row.addStretch(1)
        v.addLayout(row)
        form = QFormLayout()
        self.reference_combo = QComboBox()
        self.reference_combo.setToolTip(
            "Integrated over the same region and reported beside the curve; the curve can be "
            "expressed relative to it. Ideally a direct-excitation (non-CP) spectrum, so the "
            "CP gain is visible, but any spectrum will do.")
        form.addRow("Reference:", self.reference_combo)
        v.addLayout(form)
        self.spectra_status = QLabel("")
        self.spectra_status.setWordWrap(True)
        v.addWidget(self.spectra_status)
        return box

    def _build_region(self) -> QGroupBox:
        box = QGroupBox("2. Region and integration")
        v = QVBoxLayout(box)
        form = QFormLayout()
        row = QHBoxLayout()
        self.region_hi_spin, self.region_lo_spin = QDoubleSpinBox(), QDoubleSpinBox()
        for spin, value in ((self.region_hi_spin, 10.0), (self.region_lo_spin, 0.0)):
            spin.setRange(-1e5, 1e5)
            spin.setDecimals(3)
            spin.setSuffix(" ppm")
            spin.setValue(value)
            spin.setKeyboardTracking(False)
        row.addWidget(self.region_hi_spin)
        row.addWidget(QLabel("to"))
        row.addWidget(self.region_lo_spin)
        form.addRow("Region:", row)
        buttons = QHBoxLayout()
        self.show_reference_button = QPushButton("Show Reference")
        self.show_reference_button.setToolTip("Draw the reference (or the first ticked spectrum) "
                                              "with the region on the Compare view.")
        self.show_reference_button.clicked.connect(self.show_reference)
        self.drag_button = QPushButton("Drag on Plot")
        self.drag_button.setCheckable(True)
        self.drag_button.setToolTip("Drag across the region on the Compare view.")
        self.drag_button.toggled.connect(self._on_drag_toggled)
        buttons.addWidget(self.show_reference_button)
        buttons.addWidget(self.drag_button)
        buttons.addStretch(1)
        form.addRow("", buttons)
        self.baseline_combo = QComboBox()
        self.baseline_combo.addItem("Local (fitted just outside the region)", "local")
        self.baseline_combo.addItem("None (spectrum as it is)", "none")
        self.baseline_combo.addItem("Global (robust, whole spectrum)", "global")
        form.addRow("Baseline:", self.baseline_combo)
        self.rg_check = QCheckBox("Divide by receiver gain (RG)")
        self.rg_check.setChecked(True)
        self.rg_check.setToolTip("Integrals per scan and per unit RG, so experiments recorded "
                                 "with different receiver gains can be compared.")
        form.addRow("", self.rg_check)
        self.relative_combo = QComboBox()
        self.relative_combo.addItem("Integral per scan", "none")
        self.relative_combo.addItem("Relative to the reference", "reference")
        self.relative_combo.addItem("Relative to the largest", "max")
        form.addRow("Plot as:", self.relative_combo)
        v.addLayout(form)
        self.integrate_button = QPushButton("Integrate and Plot")
        self.integrate_button.clicked.connect(self.integrate)
        v.addWidget(self.integrate_button)
        self.series_label = ReportLabel("")
        self.series_label.setWordWrap(True)
        v.addWidget(self.series_label)
        return box

    def _build_fit(self) -> QGroupBox:
        box = QGroupBox("3. Fit the build-up")
        v = QVBoxLayout(box)
        form = QFormLayout()
        self.model_combo = QComboBox()
        for key, model in cpb.MODELS.items():
            self.model_combo.addItem(model["name"], key)
        self.model_combo.setToolTip(
            "Classical I–S (eq 10): rise with T_IS, decay with the protons' T1ρ -- the usual "
            "choice. Simple rise (eq 11): no decay seen. Eq 9: the S spins' T1ρ too. Two "
            "components: two sites (or rates) in one region. I–I*–S (eq 28): protonated "
            "carbons under MAS (fast transfer within C–H, then spin diffusion; λ = ½ for CH, "
            "⅓ for CH₂). Equation numbers from Kolodziejski & Klinowski, Chem. Rev. 2002.")
        self.model_combo.currentIndexChanged.connect(lambda _i: self._rebuild_parameters())
        form.addRow("Model:", self.model_combo)
        v.addLayout(form)
        self.parameter_table = QTableWidget(0, 5)
        self.parameter_table.setHorizontalHeaderLabels(
            ["Parameter", "Start", "Fixed", "Lower", "Upper"])
        self.parameter_table.verticalHeader().setVisible(False)
        self.parameter_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents)
        self.parameter_table.setToolTip("Starting values (times in µs). Tick Fixed to hold a "
                                        "value; Lower/Upper are optional limits.")
        v.addWidget(self.parameter_table)
        row = QHBoxLayout()
        self.estimate_button = QPushButton("Starting Values from Data")
        self.estimate_button.clicked.connect(self.estimate_start)
        self.fit_button = QPushButton("Fit")
        self.fit_button.clicked.connect(self.run_fit)
        row.addWidget(self.estimate_button)
        row.addWidget(self.fit_button)
        row.addStretch(1)
        v.addLayout(row)
        self.fit_results_label = ReportLabel("")
        self.fit_results_label.setWordWrap(True)
        self.fit_results_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        v.addWidget(self.fit_results_label)
        self.advice_box = AdviceBox()
        v.addWidget(self.advice_box)
        self._rebuild_parameters()
        return box

    # ------------------------------------------------------------------
    # spectra
    # ------------------------------------------------------------------
    def refresh(self) -> None:
        """Re-read Open experiments, keeping ticks and typed times."""
        self._entries = list(self._spectra() or [])
        table = self.spectra_table
        table.blockSignals(True)
        try:
            table.setRowCount(len(self._entries))
            for r, e in enumerate(self._entries):
                key = id(e["spectrum"])
                use = QTableWidgetItem("")
                use.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
                use.setCheckState(Qt.CheckState.Checked if key in self._checked
                                  else Qt.CheckState.Unchecked)
                table.setItem(r, _C_USE, use)
                name = QTableWidgetItem(str(e["label"]))
                name.setFlags(Qt.ItemFlag.ItemIsEnabled)
                name.setToolTip(str(e["label"]))
                table.setItem(r, _C_NAME, name)
                table.setItem(r, _C_TIME, QTableWidgetItem(self._times.get(key, "")))
        finally:
            table.blockSignals(False)
        current = self.reference_combo.currentData()
        self.reference_combo.blockSignals(True)
        self.reference_combo.clear()
        self.reference_combo.addItem("(none)", None)
        for r, e in enumerate(self._entries):
            self.reference_combo.addItem(str(e["label"]), id(e["spectrum"]))
        i = self.reference_combo.findData(current)
        self.reference_combo.setCurrentIndex(i if i >= 0 else 0)
        self.reference_combo.blockSignals(False)

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
        elif item.column() == _C_TIME:
            self._times[key] = item.text().strip()

    def set_checked(self, rows) -> None:
        """Tick these table rows (and untick the rest)."""
        rows = set(rows)
        for r in range(self.spectra_table.rowCount()):
            self.spectra_table.item(r, _C_USE).setCheckState(
                Qt.CheckState.Checked if r in rows else Qt.CheckState.Unchecked)

    def set_contact_time(self, row: int, value) -> None:
        self.spectra_table.item(row, _C_TIME).setText(f"{value:g}" if isinstance(value, float)
                                                      else str(value))

    def set_reference(self, row: int | None) -> None:
        key = None if row is None else id(self._entries[row]["spectrum"])
        i = self.reference_combo.findData(key)
        self.reference_combo.setCurrentIndex(max(i, 0))

    def _checked_rows(self) -> list[int]:
        return [r for r in range(self.spectra_table.rowCount())
                if self.spectra_table.item(r, _C_USE).checkState() == Qt.CheckState.Checked]

    def _reference_entry(self):
        key = self.reference_combo.currentData()
        return next((e for e in self._entries if id(e["spectrum"]) == key), None)

    def read_contact_times(self) -> int:
        """Fill the ticked rows' contact times from the parameter (or the
        number) typed. Returns how many were filled; problems are listed."""
        source = self.time_parameter_edit.text().strip()
        if not source:
            QMessageBox.warning(self, "CP Build-up", "Type a parameter name (e.g. P15) or a "
                                "number of µs.")
            return 0
        rows = self._checked_rows()
        if not rows:
            QMessageBox.warning(self, "CP Build-up", "Tick the spectra of the series first.")
            return 0
        problems, filled = [], 0
        for r in rows:
            e = self._entries[r]
            try:
                t = cpb.contact_time_us(e["spectrum"], source)
            except ValueError as exc:
                problems.append(f"{e['label']}: {exc}")
                continue
            self.set_contact_time(r, float(t))
            filled += 1
        text = f"Contact times read from {source} for {filled} of {len(rows)} spectra."
        if problems:
            text += " Not read: " + " ".join(problems)
        self.spectra_status.setText(text)
        return filled

    # ------------------------------------------------------------------
    # region
    # ------------------------------------------------------------------
    def region(self) -> tuple[float, float]:
        lo, hi = sorted((self.region_lo_spin.value(), self.region_hi_spin.value()))
        if lo == hi:
            raise ValueError("The region has no width.")
        return lo, hi

    def show_reference(self) -> bool:
        entry = self._reference_entry()
        if entry is None:
            rows = self._checked_rows()
            entry = self._entries[rows[0]] if rows else None
        if entry is None:
            QMessageBox.warning(self, "CP Build-up", "Choose a reference or tick a spectrum.")
            return False
        spectrum, label = entry["spectrum"], entry["label"]
        lo, hi = self.region()

        def draw(figure):
            ax = figure.add_subplot(111)
            x = spectrum.axes[0].ppm
            ax.plot(x, spectrum.data, color=self._style("spectrum").get("trace_color") or "black",
                    lw=1.0, label=str(label))
            ax.axvspan(lo, hi, color="#0072b2", alpha=0.15, lw=0, label="CP region")
            ax.set_xlim(float(np.max(x)), float(np.min(x)))
            ax.set_xlabel(f"{spectrum.metadata.nucleus or ''} shift (ppm)".strip())
            ax.set_yticks([])
            ax.legend(frameon=False)
        self._draw(draw, "spectrum")
        return True

    def _on_drag_toggled(self, on: bool) -> None:
        self.armed = bool(on)
        self._arm("region" if on else None)

    def on_canvas_region(self, lo: float, hi: float) -> None:
        lo, hi = sorted((float(lo), float(hi)))
        self.region_lo_spin.setValue(lo)
        self.region_hi_spin.setValue(hi)
        self.drag_button.setChecked(False)
        self.show_reference()

    def canvas_mode_released(self) -> None:
        if self.drag_button.isChecked():
            self.drag_button.blockSignals(True)
            self.drag_button.setChecked(False)
            self.drag_button.blockSignals(False)
        self.armed = False

    # ------------------------------------------------------------------
    # integrate
    # ------------------------------------------------------------------
    def integrate(self) -> bool:
        rows = self._checked_rows()
        if len(rows) < 2:
            QMessageBox.warning(self, "CP Build-up", "Tick at least two spectra of the series.")
            return False
        times, missing = [], []
        for r in rows:
            v = _number(self.spectra_table.item(r, _C_TIME).text())
            if v is None or v < 0:
                missing.append(str(self._entries[r]["label"]))
            times.append(v)
        if missing:
            QMessageBox.warning(self, "CP Build-up", "No contact time for: " + ", ".join(missing)
                                + ". Type them (µs) or use Read Contact Times.")
            return False
        ref = self._reference_entry()
        try:
            self.series = cpb.cp_series(
                [self._entries[r]["spectrum"] for r in rows], self.region(), times,
                reference=ref["spectrum"] if ref else None,
                labels=[self._entries[r]["label"] for r in rows],
                baseline=self.baseline_combo.currentData(), divide_rg=self.rg_check.isChecked(),
                relative_to=self.relative_combo.currentData())
        except ValueError as exc:
            QMessageBox.warning(self, "CP Build-up", str(exc))
            return False
        self._series_indices = [self._entries[r].get("index") for r in rows]
        if ref is not None:
            self._series_indices.append(ref.get("index"))
        self.fit = None
        self.fit_results_label.setText("")
        self.advice_box.set_items([])
        lines = [f"{len(rows)} spectra, contact times {self.series['t_us'].min():g} to "
                 f"{self.series['t_us'].max():g} µs; y = {self.series['unit']}."]
        if self.series["reference"] is not None:
            lines.append(f"Reference integral: {self.series['reference']['integral']:.4g} ± "
                         f"{self.series['reference']['error']:.2g}.")
        lines += [f"Note: {n}" for n in self.series["notes"]]
        self.series_label.setText("\n".join(lines))
        self._draw_curve()
        self.estimate_start()
        self.calculated.emit()
        return True

    def _draw_curve(self) -> None:
        series, fit = self.series, self.fit

        def draw(figure):
            style = self._style("profile")
            cpb.plot_cp_buildup(series, fit, figure=figure,
                                colour=style.get("trace_color") or "black")
        self._draw(draw, "profile")

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------
    def model(self) -> str:
        return self.model_combo.currentData()

    def _rebuild_parameters(self, start: dict | None = None) -> None:
        names = cpb.MODELS[self.model()]["params"]
        old = self._parameter_values()
        table = self.parameter_table
        table.setRowCount(len(names))
        for r, n in enumerate(names):
            label, unit = cpb.PARAMETERS[n]
            item = QTableWidgetItem(f"{label}" + (f" ({unit})" if unit else ""))
            item.setFlags(Qt.ItemFlag.ItemIsEnabled)
            item.setData(Qt.ItemDataRole.UserRole, n)
            table.setItem(r, 0, item)
            value = (start or {}).get(n, old.get(n, {}).get("start"))
            table.setItem(r, 1, QTableWidgetItem("" if value is None else f"{value:.6g}"))
            fixed = QTableWidgetItem("")
            fixed.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
            fixed.setCheckState(Qt.CheckState.Checked if old.get(n, {}).get("fixed")
                                else Qt.CheckState.Unchecked)
            table.setItem(r, 2, fixed)
            for c, key in ((3, "lower"), (4, "upper")):
                v = old.get(n, {}).get(key)
                table.setItem(r, c, QTableWidgetItem("" if v is None else f"{v:g}"))

    def _parameter_values(self) -> dict:
        out = {}
        table = self.parameter_table
        for r in range(table.rowCount()):
            item = table.item(r, 0)
            if item is None:
                continue
            n = item.data(Qt.ItemDataRole.UserRole)
            get = lambda c: table.item(r, c).text() if table.item(r, c) else ""  # noqa: E731
            out[n] = {"start": _number(get(1)), "lower": _number(get(3)), "upper": _number(get(4)),
                      "fixed": table.item(r, 2) is not None
                      and table.item(r, 2).checkState() == Qt.CheckState.Checked}
        return out

    def set_parameter(self, name: str, start=None, fixed=None) -> None:
        for r in range(self.parameter_table.rowCount()):
            if self.parameter_table.item(r, 0).data(Qt.ItemDataRole.UserRole) == name:
                if start is not None:
                    self.parameter_table.item(r, 1).setText(f"{start:g}")
                if fixed is not None:
                    self.parameter_table.item(r, 2).setCheckState(
                        Qt.CheckState.Checked if fixed else Qt.CheckState.Unchecked)
                return
        raise KeyError(name)

    def estimate_start(self) -> None:
        if self.series is None:
            return
        start = cpb.default_start(self.series["t_us"], self.series["y"], self.model())
        values = self._parameter_values()
        for n, v in values.items():         # keep values the user fixed
            if v["fixed"] and v["start"] is not None:
                start[n] = v["start"]
        self._rebuild_parameters(start)

    def run_fit(self) -> bool:
        if self.series is None and not self.integrate():
            return False
        values = self._parameter_values()
        start = {n: v["start"] for n, v in values.items() if v["start"] is not None}
        fixed = {}
        for n, v in values.items():
            if v["fixed"]:
                if v["start"] is None:
                    QMessageBox.warning(self, "CP Build-up", f"Give a value for the fixed "
                                        f"{cpb.PARAMETERS[n][0]}.")
                    return False
                fixed[n] = v["start"]
        bounds = {}
        for n, v in values.items():
            if v["lower"] is not None or v["upper"] is not None:
                bounds[n] = (v["lower"] if v["lower"] is not None else -np.inf,
                             v["upper"] if v["upper"] is not None else np.inf)
        try:
            self.fit = cpb.fit_cp_buildup(self.series["t_us"], self.series["y"], self.model(),
                                          start=start, fixed=fixed, bounds=bounds,
                                          y_err=self.series["y_err"])
        except ValueError as exc:
            QMessageBox.warning(self, "CP Build-up", str(exc))
            return False
        report = cpb.cp_report(self.fit, self.series)
        self.fit_results_label.setText(report)
        self.advice_box.set_items(fit_advice.advise_cp_buildup(self.fit))
        self._draw_curve()
        record = {"label": "CP build-up", "target": ", ".join(map(str, self.series["labels"])),
                  "inputs_text": (f"region {self.series['region'][1]:g} to "
                                  f"{self.series['region'][0]:g} ppm; model {self.model()}; "
                                  f"baseline {self.series['baseline']}; RG divided: "
                                  f"{'yes' if self.series['divide_rg'] else 'no'}"),
                  "report": report,
                  "result": {"model": self.model(), "params": self.fit["params"],
                             "errors": self.fit["errors"], "fixed": self.fit["fixed"],
                             "t_us": self.series["t_us"].tolist(), "y": self.series["y"].tolist(),
                             "y_err": self.series["y_err"].tolist(),
                             "r_squared": self.fit["r_squared"],
                             "t_max_us": self.fit["t_max_us"]}}
        indices = [i for i in self._series_indices if i is not None]
        if indices:
            self._store("cp_buildup", record, indices)
        self.fitted.emit()
        return True
