"""
Dynamics ▸ DNP Polarisation: the difference between two pseudo-2D series, against a delay.

Pick two experiments in Open experiments (microwaves on and off, or two
polarisation conditions), give the delay list the way the Fit Curve panel
does (the experiment's own list, a typed list, start/stop/step, or start +
step with a multiplier), choose what to measure in every slice, and plot the
difference (or ratio, enhancement, sum) against the delay. Nothing is fitted.
The two series can be drawn with it, and the points joined by lines.

The maths is :mod:`opennmr.notebook.dnp_polarisation`; *Copy notebook code*
gives the same numbers in a notebook. The figure is drawn on the Fit view,
so Plot Style (fonts, ticks, colours, axis names) applies to it.
"""

from __future__ import annotations

import math

import numpy as np
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QMessageBox, QPushButton, QVBoxLayout, QWidget,
)

from ..notebook import dnp_polarisation as dp
from .widgets.report_label import ReportLabel
from .widgets.typed_spinbox import TypedSpinBox

_SOURCES = (
    ("experiment", "From the experiment (vdlist / recorded delays)"),
    ("slice", "Slice number (1, 2, 3, …)"),
    ("list", "Typed list"),
    ("start_stop_step", "Start, stop and step"),
    ("start_step", "Start + step (× multiplier)"),
)
_UNITS = (("s", 1.0), ("ms", 1e-3), ("µs", 1e-6))
_QUANTITIES = (
    ("integral", "Region integral"),
    ("height", "Peak height"),
    ("fit_area", "Stored per-slice fit: area"),
    ("fit_height", "Stored per-slice fit: height"),
)


def _number(text):
    text = str(text).strip().replace("−", "-")
    if not text:
        return None
    try:
        v = float(text)
    except ValueError:
        return None
    return v if math.isfinite(v) else None


class DnpPolarisationPanel(QWidget):
    """Callbacks (from MainWindow): ``spectra()`` -> ``[{"label", "spectrum",
    "index"}]`` (the pseudo-2D and 2D experiments); ``draw(draw)`` puts a
    figure-drawing function on the Fit view."""

    def __init__(self, spectra, draw, parent=None) -> None:
        super().__init__(parent)
        self._spectra, self._draw = spectra, draw
        self.result: dict | None = None
        self._entries: list[dict] = []

        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        intro = QLabel(
            "Polarisation against a delay: measure every slice of two pseudo-2D experiments "
            "(for example microwaves on and off), subtract them and plot the result against "
            "the delay. Nothing is fitted.")
        intro.setWordWrap(True)
        v.addWidget(intro)
        v.addWidget(self._build_datasets())
        v.addWidget(self._build_delays())
        v.addWidget(self._build_measure())
        v.addWidget(self._build_plot())
        row = QHBoxLayout()
        self.plot_button = QPushButton("Plot Polarisation")
        self.plot_button.clicked.connect(self.plot)
        self.export_button = QPushButton("Export CSV…")
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
    def _build_datasets(self) -> QGroupBox:
        box = QGroupBox("1. Experiments")
        form = QFormLayout(box)
        self.a_combo, self.b_combo = QComboBox(), QComboBox()
        self.a_combo.setToolTip("Experiment A: for example microwaves on. The result is A − B.")
        self.b_combo.setToolTip("Experiment B: for example microwaves off.")
        form.addRow("A (e.g. µw on):", self.a_combo)
        form.addRow("B (e.g. µw off):", self.b_combo)
        self.a_name_edit, self.b_name_edit = QLineEdit(), QLineEdit()
        self.a_name_edit.setPlaceholderText("optional, e.g. µw on")
        self.b_name_edit.setPlaceholderText("optional, e.g. µw off")
        form.addRow("A label:", self.a_name_edit)
        form.addRow("B label:", self.b_name_edit)
        return box

    def _build_delays(self) -> QGroupBox:
        box = QGroupBox("2. Delays")
        form = QFormLayout(box)
        self.source_combo = QComboBox()
        for key, text in _SOURCES:
            self.source_combo.addItem(text, key)
        self.source_combo.setToolTip(
            "Where the delay of each slice comes from. The experiment's own list is read for "
            "each of A and B separately; the other choices apply to both.")
        self.source_combo.currentIndexChanged.connect(self._sync)
        form.addRow("Delays:", self.source_combo)
        self.unit_combo = QComboBox()
        for text, _scale in _UNITS:
            self.unit_combo.addItem(text, text)
        self.unit_combo.setToolTip("The unit of the numbers typed here (not of the plot).")
        form.addRow("Unit:", self.unit_combo)
        self.list_edit = QLineEdit()
        self.list_edit.setPlaceholderText("0.5, 1, 2, 4, 8 … (commas or spaces)")
        self.list_row = form.rowCount()
        form.addRow("List:", self.list_edit)
        self.start_edit, self.stop_edit, self.step_edit = QLineEdit(), QLineEdit(), QLineEdit()
        self.multiplier_edit = QLineEdit("1")
        self.multiplier_edit.setToolTip("Each step is this many times the last (1 = equal steps).")
        form.addRow("Start:", self.start_edit)
        form.addRow("Stop:", self.stop_edit)
        form.addRow("Step:", self.step_edit)
        form.addRow("Step multiplier:", self.multiplier_edit)
        self._delay_form = form
        self.delays_label = QLabel("")
        self.delays_label.setWordWrap(True)
        form.addRow(self.delays_label)
        for w in (self.list_edit, self.start_edit, self.stop_edit, self.step_edit,
                  self.multiplier_edit):
            w.textChanged.connect(self._sync)
        self.unit_combo.currentIndexChanged.connect(self._sync)
        self.a_combo.currentIndexChanged.connect(self._sync)
        return box

    def _build_measure(self) -> QGroupBox:
        box = QGroupBox("3. What to measure in each slice")
        form = QFormLayout(box)
        self._measure_form = form
        self.quantity_combo = QComboBox()
        for key, text in _QUANTITIES:
            self.quantity_combo.addItem(text, key)
        self.quantity_combo.setToolTip(
            "Region integral and peak height are measured here, slice by slice. The stored "
            "per-slice fit options take values from the peak fit or region integration "
            "already made on each experiment (Fit or Integrate tab).")
        self.quantity_combo.currentIndexChanged.connect(self._sync)
        form.addRow("Quantity:", self.quantity_combo)
        self.region_edit = QLineEdit()
        self.region_edit.setPlaceholderText("from, to (ppm)")
        form.addRow("Region (ppm):", self.region_edit)
        self.peak_edit = QLineEdit()
        self.peak_edit.setPlaceholderText("ppm")
        form.addRow("Peak position (ppm):", self.peak_edit)
        self.peak_index_spin = TypedSpinBox()
        self.peak_index_spin.setRange(1, 99)
        self.peak_index_spin.setToolTip("Which fitted peak (or integrated region), counting from 1.")
        form.addRow("Fitted peak / region #:", self.peak_index_spin)
        self.baseline_combo = QComboBox()
        for key, text in (("local", "Local (fitted beside the peak)"), ("global", "Global"),
                          ("none", "None")):
            self.baseline_combo.addItem(text, key)
        form.addRow("Baseline:", self.baseline_combo)
        self.per_scan_check = QCheckBox("Per scan")
        self.per_scan_check.setChecked(True)
        self.divide_rg_check = QCheckBox("Per unit receiver gain")
        self.divide_rg_check.setChecked(True)
        self.per_scan_check.setToolTip(
            "Scale to one scan where the data are summed (JEOL); Bruker data already are per scan.")
        self.divide_rg_check.setToolTip(
            "Divide by the receiver gain where it is recorded, so experiments recorded with "
            "different gains are comparable.")
        form.addRow(self.per_scan_check)
        form.addRow(self.divide_rg_check)
        return box

    def _build_plot(self) -> QGroupBox:
        box = QGroupBox("4. Plot")
        form = QFormLayout(box)
        self.mode_combo = QComboBox()
        for key, text in (("difference", "Difference, A − B"), ("ratio", "Ratio, A / B"),
                          ("enhancement", "Enhancement, (A − B) / B"), ("sum", "Sum, A + B")):
            self.mode_combo.addItem(text, key)
        form.addRow("Plot:", self.mode_combo)
        self.individual_check = QCheckBox("Also show the two experiments separately")
        self.individual_check.setToolTip(
            "Draws A and B in a panel above the result, on the same delay axis.")
        self.connect_check = QCheckBox("Join the points with lines")
        self.connect_check.setChecked(True)
        self.error_check = QCheckBox("Error bars")
        self.error_check.setChecked(True)
        self.interpolate_check = QCheckBox("Interpolate B onto A's delays if they differ")
        self.time_combo = QComboBox()
        for text in ("auto", "s", "ms", "µs"):
            self.time_combo.addItem(text, text)
        form.addRow(self.individual_check)
        form.addRow(self.connect_check)
        form.addRow(self.error_check)
        form.addRow(self.interpolate_check)
        form.addRow("Delay axis unit:", self.time_combo)
        return box

    # --------------------------------------------------------------- behaviour
    def refresh(self) -> None:
        """Rebuild the experiment lists, keeping each choice where it still exists."""
        keep_a, keep_b = self.a_combo.currentData(), self.b_combo.currentData()
        self._entries = list(self._spectra())
        for combo, keep in ((self.a_combo, keep_a), (self.b_combo, keep_b)):
            combo.blockSignals(True)
            combo.clear()
            for entry in self._entries:
                combo.addItem(entry["label"], entry["index"])
            position = combo.findData(keep) if keep is not None else -1
            if position >= 0:
                combo.setCurrentIndex(position)
            combo.blockSignals(False)
        # A and B are never the same experiment unless there is only one to choose: B moves
        # to the next one along.
        if self.b_combo.count() > 1 and self.a_combo.currentData() == self.b_combo.currentData():
            self.b_combo.setCurrentIndex((self.b_combo.currentIndex() + 1) % self.b_combo.count())
        self._sync()

    def _entry(self, combo) -> dict | None:
        index = combo.currentData()
        return next((e for e in self._entries if e["index"] == index), None)

    def _sync(self, *_args) -> None:
        """Show only the boxes the current choices use, and say what the delays are."""
        source = self.source_combo.currentData()
        quantity = self.quantity_combo.currentData()
        form = self._delay_form
        shown = {self.list_edit: source == "list",
                 self.start_edit: source in ("start_stop_step", "start_step"),
                 self.stop_edit: source == "start_stop_step",
                 self.step_edit: source in ("start_stop_step", "start_step"),
                 self.multiplier_edit: source == "start_step",
                 self.unit_combo: source in ("list", "start_stop_step", "start_step")}
        for widget, visible in shown.items():
            widget.setVisible(visible)
            label = form.labelForField(widget)
            if label is not None:
                label.setVisible(visible)
        measure = self._measure_form
        for widget, visible in ((self.region_edit, quantity == "integral"),
                                (self.peak_edit, quantity == "height"),
                                (self.peak_index_spin, quantity in ("fit_area", "fit_height")),
                                (self.baseline_combo, quantity in ("integral", "height")),
                                (self.per_scan_check, quantity in ("integral", "height")),
                                (self.divide_rg_check, quantity in ("integral", "height"))):
            widget.setVisible(visible)
            label = measure.labelForField(widget)
            if label is not None:
                label.setVisible(visible)
        self.delays_label.setText(self._delay_summary())

    def _delay_summary(self) -> str:
        entry = self._entry(self.a_combo)
        if entry is None:
            return ""
        try:
            values, note = self._delays_for(entry)
        except ValueError as exc:
            return f"⚠ {exc}"
        shown = (", ".join(f"{v:.4g}" for v in values) if values.size <= 6
                 else ", ".join(f"{v:.4g}" for v in values[:5]) + f", …, {values[-1]:.4g}")
        text = f"A: {values.size} delays [s]: {shown}"
        if note:
            text += f"\n{note}"
        return text

    def _scale(self) -> float:
        return dict(_UNITS)[self.unit_combo.currentData()]

    def _delays_for(self, entry: dict) -> tuple[np.ndarray, str | None]:
        """The delays (seconds) of ``entry``'s slices, and a note on where they came from."""
        spectrum = entry["spectrum"]
        n = spectrum.data.shape[0]
        source = self.source_combo.currentData()
        if source == "slice":
            return np.arange(1, n + 1, dtype=float), "Slice numbers, not times: the axis is the slice."
        if source == "experiment":
            values, provenance = dp.experiment_delays(spectrum)
            if values is None:
                raise ValueError(
                    f"{entry['label']} records no delay list. Choose a typed list, start/stop/step "
                    "or start + step instead.")
            return values, f"From {provenance}."
        scale = self._scale()
        if source == "list":
            tokens = [t for t in self.list_edit.text().replace(",", " ").split() if t]
            if not tokens:
                raise ValueError("Type the list of delays.")
            numbers = [_number(t) for t in tokens]
            if any(v is None for v in numbers):
                raise ValueError("The list must be numbers separated by commas or spaces.")
            return dp.delay_values(n, values=numbers, scale=scale), None
        start, stop = _number(self.start_edit.text()), _number(self.stop_edit.text())
        step, mult = _number(self.step_edit.text()), _number(self.multiplier_edit.text())
        if start is None:
            raise ValueError("Type a start value.")
        if source == "start_stop_step":
            if stop is None or step is None:
                raise ValueError("Type a stop and a step.")
            return dp.delay_values(n, start=start, stop=stop, step=step, scale=scale), None
        if step is None:
            raise ValueError("Type a step.")
        return dp.delay_values(n, start=start, step=step, multiplier=mult or 1.0, scale=scale), None

    def _series(self, entry: dict, label: str) -> dict:
        spectrum = entry["spectrum"]
        values, _note = self._delays_for(entry)
        quantity = self.quantity_combo.currentData()
        if quantity in ("integral", "height"):
            region = peak = None
            if quantity == "integral":
                parts = self.region_edit.text().replace(",", " ").replace(";", " ").split()
                numbers = [_number(p) for p in parts]
                if len(numbers) != 2 or any(v is None for v in numbers):
                    raise ValueError("Type the region as two numbers, from and to, in ppm.")
                region = tuple(numbers)
            else:
                peak = _number(self.peak_edit.text())
                if peak is None:
                    raise ValueError("Type the peak position in ppm.")
            return dp.measure_series(
                spectrum, values, quantity, region=region, peak=peak,
                baseline=self.baseline_combo.currentData(), label=label,
                per_scan=self.per_scan_check.isChecked(), divide_rg=self.divide_rg_check.isChecked())
        stored = (spectrum.metadata.raw or {}).get("slice_fit_results")
        if not stored:
            raise ValueError(
                f"{entry['label']} has no stored per-slice fit. Fit its peaks (Fit tab) or "
                "integrate its regions (Integrate tab) first, or measure here instead.")
        source = "area" if quantity == "fit_area" else "amplitude"
        return dp.series_from_fits(stored, values, source,
                                   peak_index=self.peak_index_spin.value() - 1, label=label)

    def compute(self) -> dict:
        """Measure both experiments and combine them (raises ValueError with the reason)."""
        a, b = self._entry(self.a_combo), self._entry(self.b_combo)
        if a is None or b is None:
            raise ValueError("Open two pseudo-2D experiments first (Open tab), then choose them here.")
        if a["index"] == b["index"]:
            raise ValueError("A and B are the same experiment; choose two different ones.")
        series_a = self._series(a, self.a_name_edit.text().strip() or a["label"])
        series_b = self._series(b, self.b_name_edit.text().strip() or b["label"])
        return dp.polarisation(series_a, series_b, self.mode_combo.currentData(),
                               interpolate=self.interpolate_check.isChecked())

    def plot(self) -> None:
        try:
            result = self.compute()
        except ValueError as exc:
            QMessageBox.warning(self, "DNP polarisation", str(exc))
            return
        self.result = result
        self.report_label.setText(dp.polarisation_report(result))
        options = dict(show_individual=self.individual_check.isChecked(),
                       connect=self.connect_check.isChecked(),
                       error_bars=self.error_check.isChecked(),
                       time_unit=self.time_combo.currentData())
        self._draw(lambda figure: dp.plot_polarisation(result, figure=figure, **options))

    def export_csv(self) -> None:
        if self.result is None:
            try:
                self.result = self.compute()
            except ValueError as exc:
                QMessageBox.warning(self, "DNP polarisation", str(exc))
                return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export polarisation", "polarisation.csv", "CSV (*.csv)")
        if path:
            dp.polarisation_to_csv(self.result, path)

    def notebook_code(self) -> str:
        """The same measurement as notebook code."""
        a, b = self._entry(self.a_combo), self._entry(self.b_combo)
        quantity = self.quantity_combo.currentData()
        source = self.source_combo.currentData()
        scale = self._scale()
        if source == "list":
            delays = f"dp.delay_values(len(on.data), values=[{self.list_edit.text().strip()}], scale={scale:g})"
        elif source == "start_stop_step":
            delays = (f"dp.delay_values(len(on.data), start={self.start_edit.text() or 0}, "
                      f"stop={self.stop_edit.text() or 0}, step={self.step_edit.text() or 1}, scale={scale:g})")
        elif source == "start_step":
            delays = (f"dp.delay_values(len(on.data), start={self.start_edit.text() or 0}, "
                      f"step={self.step_edit.text() or 1}, multiplier={self.multiplier_edit.text() or 1}, "
                      f"scale={scale:g})")
        elif source == "slice":
            delays = "np.arange(1, len(on.data) + 1)"
        else:
            delays = "dp.experiment_delays(on)[0]"
        lines = ["import numpy as np", "from opennmr.notebook import dnp_polarisation as dp", "",
                 f"# on = {a['label'] if a else 'experiment A'};  off = {b['label'] if b else 'experiment B'}",
                 f"delays = {delays}"]
        if quantity == "integral":
            region = tuple(_number(p) for p in self.region_edit.text().replace(",", " ").split()[:2])
            kw = f'"integral", region={region}'
        elif quantity == "height":
            kw = f'"height", peak={_number(self.peak_edit.text())}'
        else:
            kw = None
        if kw is not None:
            opts = (f", baseline={self.baseline_combo.currentData()!r}, "
                    f"per_scan={self.per_scan_check.isChecked()}, divide_rg={self.divide_rg_check.isChecked()}")
            lines += [f"series_on = dp.measure_series(on, delays, {kw}{opts})",
                      f"series_off = dp.measure_series(off, delays, {kw}{opts})"]
        else:
            src = "area" if quantity == "fit_area" else "amplitude"
            n = self.peak_index_spin.value() - 1
            lines += [f'series_on = dp.series_from_fits(on.metadata.raw["slice_fit_results"], delays, "{src}", peak_index={n})',
                      f'series_off = dp.series_from_fits(off.metadata.raw["slice_fit_results"], delays, "{src}", peak_index={n})']
        lines += [f"result = dp.polarisation(series_on, series_off, {self.mode_combo.currentData()!r}, "
                  f"interpolate={self.interpolate_check.isChecked()})",
                  f"dp.plot_polarisation(result, show_individual={self.individual_check.isChecked()}, "
                  f"connect={self.connect_check.isChecked()}, error_bars={self.error_check.isChecked()}, "
                  f"time_unit={self.time_combo.currentData()!r})",
                  'dp.polarisation_to_csv(result, "polarisation.csv")']
        return "\n".join(lines) + "\n"

    def copy_code(self) -> None:
        QGuiApplication.clipboard().setText(self.notebook_code())
