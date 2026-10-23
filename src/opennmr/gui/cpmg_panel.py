"""
Advanced Processing > CPMG: the notebook's :func:`CPMGprocess` (split a
raw Bruker CPMG echo train into echoes, co-add them, apodise, zero-fill,
FT) as a panel.

Every step that is normally judged by eye is shown before it is used:
*Check echo split* reads the parameters and draws the raw echo train
with the windows that will be co-added, so a wrong first point or a
wrong parameter index is visible before anything is processed. The
result goes to Open experiments as a new spectrum, with a log of what
was used.

The processing itself is entirely :mod:`opennmr.notebook.one_d`
(:func:`cpmg_echo_layout`, :func:`CPMGprocess`, :func:`cpmg_to_spectrum`)
-- this module only collects settings and shows results.
"""

from __future__ import annotations

import os

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from .widgets.typed_spinbox import TypedSpinBox
from .widgets.report_label import ReportLabel
from PySide6.QtCore import Signal
from PySide6.QtGui import QPalette
from PySide6.QtWidgets import (
    QCheckBox, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
    QPlainTextEdit, QPushButton, QSpinBox, QVBoxLayout, QWidget,
)

from ..notebook import one_d as nb_one_d


def cpmg_source_problem(spectrum) -> str | None:
    """Why ``spectrum`` cannot be CPMG-processed, or ``None`` if it can:
    it must be a Bruker experiment whose folder still has ``acqus`` and
    the raw ``fid``."""
    if spectrum is None:
        return "Select a Bruker CPMG experiment in Open experiments."
    meta = spectrum.metadata
    if meta.vendor != "bruker" or not meta.source_path or not os.path.isdir(meta.source_path):
        return ("CPMG processing reads the raw echo train from a Bruker experiment folder; "
                "the selected spectrum is not one.")
    for name in ("acqus", "fid"):
        if not os.path.isfile(os.path.join(meta.source_path, name)):
            return f"The experiment folder has no {name} file."
    return None


class CpmgProcessingPanel(QWidget):
    """Settings, echo-split preview, and Process for one CPMG experiment.

    ``set_source(spectrum, label)`` points it at the selected experiment;
    ``spectrum_ready(NMRSpectrum, label)`` is emitted with the result."""

    spectrum_ready = Signal(object, str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._source = None
        self._source_label = ""
        self._layout_result = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        intro = QLabel(
            "Co-adds the echoes of a CPMG echo train into one echo, then Fourier "
            "transforms it (the notebook's CPMGprocess). Check the echo split first.")
        intro.setWordWrap(True)
        layout.addWidget(intro)
        self.source_label = QLabel("")
        self.source_label.setWordWrap(True)
        layout.addWidget(self.source_label)

        # --- where the pulse programme keeps its timings ----------------
        params_box = QGroupBox("Pulse-programme parameters")
        params_form = QFormLayout(params_box)
        self.echo_delay_spin = self._index_spin(6, "D", "Delay holding the echo length (d6 in the usual qcpmg).")
        self.blanking_delay_spin = self._index_spin(3, "D", "Delay used for blanking: 2 x this is part of the blanking time.")
        self.blanking_cnst_spin = self._index_spin(20, "CNST", "Constant (us) added to the blanking time.")
        self.num_echoes_spin = self._index_spin(22, "L", "Loop counter holding the number of acquired echoes.")
        params_form.addRow("Echo length:", self.echo_delay_spin)
        params_form.addRow("Blanking delay:", self.blanking_delay_spin)
        params_form.addRow("Blanking constant:", self.blanking_cnst_spin)
        params_form.addRow("Number of echoes:", self.num_echoes_spin)
        layout.addWidget(params_box)

        # --- the echo split ---------------------------------------------
        split_box = QGroupBox("Echo split")
        split_form = QFormLayout(split_box)
        first_row = QHBoxLayout()
        self.first_auto_check = QCheckBox("Auto")
        self.first_auto_check.setChecked(True)
        self.first_auto_check.setToolTip(
            "Find the first acquired (non-zero) point of the raw FID and add the offset. "
            "Untick to give the first point yourself if the split lands in the wrong place.")
        self.first_point_spin = TypedSpinBox()
        self.first_point_spin.setRange(0, 100_000_000)
        self.first_point_spin.setEnabled(False)
        self.first_point_spin.setToolTip("Index of the first point of the echo train in the raw FID.")
        self.first_auto_check.toggled.connect(lambda auto: self.first_point_spin.setEnabled(not auto))
        first_row.addWidget(self.first_auto_check)
        first_row.addWidget(self.first_point_spin, 1)
        split_form.addRow("First point:", first_row)
        self.first_offset_spin = TypedSpinBox()
        self.first_offset_spin.setRange(0, 100_000)
        self.first_offset_spin.setValue(20)
        self.first_offset_spin.setToolTip("Auto only: points added past the first non-zero point.")
        split_form.addRow("Auto offset (points):", self.first_offset_spin)
        self.safety_spin = TypedSpinBox()
        self.safety_spin.setRange(0, 100_000)
        self.safety_spin.setValue(10)
        self.safety_spin.setToolTip(
            "Points kept back from the blanking at each end of every echo – guards "
            "against blanking artefacts at the echo edges.")
        split_form.addRow("Safety (points):", self.safety_spin)
        self.cutout_spin = TypedSpinBox()
        self.cutout_spin.setRange(0, 1_000_000)
        self.cutout_spin.setToolTip("Leave out this many echoes at the start of the train.")
        split_form.addRow("Skip first echoes:", self.cutout_spin)
        self.echoes_added_spin = TypedSpinBox()
        self.echoes_added_spin.setRange(0, 1_000_000)
        self.echoes_added_spin.setSpecialValueText("All")
        self.echoes_added_spin.setToolTip(
            "How many echoes to co-add, counted after the skipped ones. "
            "Late echoes add mostly noise once the signal has decayed.")
        split_form.addRow("Echoes to co-add:", self.echoes_added_spin)
        check_button = QPushButton("Check echo split")
        check_button.setToolTip("Read the parameters and draw where the echoes will be cut.")
        check_button.clicked.connect(self.check_split)
        split_form.addRow(check_button)
        self.summary_label = ReportLabel("")
        self.summary_label.setWordWrap(True)
        split_form.addRow(self.summary_label)
        layout.addWidget(split_box)

        self.figure = Figure(figsize=(4.0, 2.2), layout="constrained")
        self.preview_canvas = FigureCanvasQTAgg(self.figure)
        self.preview_canvas.setMinimumHeight(200)
        # Zoom/pan, to look at individual echo cuts in a long train.
        self.preview_toolbar = NavigationToolbar2QT(self.preview_canvas, self)
        layout.addWidget(self.preview_toolbar)
        layout.addWidget(self.preview_canvas)

        # --- processing -------------------------------------------------
        proc_box = QGroupBox("Processing")
        proc_form = QFormLayout(proc_box)
        self.apodize_check = QCheckBox("Apodise the co-added echo")
        self.apodize_check.setChecked(True)
        proc_form.addRow(self.apodize_check)
        self.apod_zero_spin = TypedSpinBox()
        self.apod_zero_spin.setRange(0, 100_000)
        self.apod_zero_spin.setValue(20)
        self.apod_zero_spin.setToolTip("Exact zeros at each end of the window.")
        proc_form.addRow("Zeroed edge points:", self.apod_zero_spin)
        self.apod_shape_spin = QDoubleSpinBox()
        self.apod_shape_spin.setRange(0.1, 20.0)
        self.apod_shape_spin.setValue(2.0)
        self.apod_shape_spin.setToolTip("Exponent of the |sin| taper; 2 is Hamming-like.")
        proc_form.addRow("Window shape:", self.apod_shape_spin)
        self.apod_alpha_spin = QDoubleSpinBox()
        self.apod_alpha_spin.setRange(0.0, 1.0)
        self.apod_alpha_spin.setDecimals(3)
        self.apod_alpha_spin.setSingleStep(0.01)
        self.apod_alpha_spin.setValue(0.54)
        self.apod_alpha_spin.setToolTip("Hamming coefficient (0.54 = Hamming).")
        proc_form.addRow("Window alpha:", self.apod_alpha_spin)
        for widget in (self.apod_zero_spin, self.apod_shape_spin, self.apod_alpha_spin):
            self.apodize_check.toggled.connect(widget.setEnabled)
        self.zero_fill_spin = TypedSpinBox()
        self.zero_fill_spin.setRange(0, 1 << 24)
        self.zero_fill_spin.setSpecialValueText("Auto")
        self.zero_fill_spin.setToolTip(
            "Points to zero-fill the co-added echo to. Auto: the next power of two "
            "above its length, times four.")
        proc_form.addRow("Zero-fill to:", self.zero_fill_spin)
        self.magnitude_check = QCheckBox("Magnitude mode")
        self.magnitude_check.setChecked(True)
        self.magnitude_check.setToolTip(
            "Untick to keep the real and imaginary parts, then phase the result with "
            "the usual Phase controls.")
        proc_form.addRow(self.magnitude_check)
        self.process_button = QPushButton("Process CPMG")
        self.process_button.clicked.connect(self.process)
        proc_form.addRow(self.process_button)
        layout.addWidget(proc_box)

        self.log_edit = QPlainTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setMaximumBlockCount(500)
        self.log_edit.setPlaceholderText("What was used appears here.")
        self.log_edit.setMinimumHeight(90)
        layout.addWidget(self.log_edit)

        self.set_source(None)

    # --- public -------------------------------------------------------
    def set_source(self, spectrum, label: str = "") -> None:
        """Point the panel at ``spectrum`` (the selected experiment).

        Selecting a CPMG *result* keeps the experiment it came from as
        the source, so settings can be tweaked and processed again
        without reselecting the raw experiment each time."""
        if (spectrum is not None and "cpmg" in (spectrum.metadata.raw or {})
                and self._source is not None and cpmg_source_problem(self._source) is None):
            return
        changed = spectrum is not self._source
        self._source = spectrum
        self._source_label = label
        problem = cpmg_source_problem(spectrum)
        self.process_button.setEnabled(problem is None)
        self.source_label.setText(problem or f"Experiment: {label or spectrum.metadata.source_path}")
        if changed:
            self._layout_result = None
            self.summary_label.setText("")
            self.figure.clear()
            self.preview_canvas.draw_idle()

    def settings(self) -> dict:
        """The keyword arguments for :func:`CPMGprocess`."""
        return {
            "echo_delay_index": self.echo_delay_spin.value(),
            "blanking_delay_index": self.blanking_delay_spin.value(),
            "blanking_cnst_index": self.blanking_cnst_spin.value(),
            "num_echoes_index": self.num_echoes_spin.value(),
            "first_point": "auto" if self.first_auto_check.isChecked() else self.first_point_spin.value(),
            "first_point_offset": self.first_offset_spin.value(),
            "safety": self.safety_spin.value(),
            "cutout": self.cutout_spin.value(),
            "num_echoes_added": self.echoes_added_spin.value() or None,
            "apodize": self.apodize_check.isChecked(),
            "apod_offset_zero": self.apod_zero_spin.value(),
            "apod_shape": self.apod_shape_spin.value(),
            "apod_alpha": self.apod_alpha_spin.value(),
            "zero_fill": self.zero_fill_spin.value() or "auto",
            "magnitude_mode": self.magnitude_check.isChecked(),
        }

    def check_split(self) -> bool:
        """Resolve the echo split and draw it. Returns False on a problem
        (which is shown, not raised)."""
        problem = cpmg_source_problem(self._source)
        if problem:
            self._log(problem)
            return False
        s = self.settings()
        try:
            result = nb_one_d.cpmg_echo_layout(
                self._source.metadata.source_path,
                echo_delay_index=s["echo_delay_index"], blanking_delay_index=s["blanking_delay_index"],
                blanking_cnst_index=s["blanking_cnst_index"], num_echoes_index=s["num_echoes_index"],
                first_point=s["first_point"], first_point_offset=s["first_point_offset"],
                safety=s["safety"])
        except (ValueError, OSError) as exc:
            self.summary_label.setText(f"Can't split the echo train: {exc}")
            self._log(f"Echo split failed: {exc}")
            return False
        self._layout_result = result
        if self.first_auto_check.isChecked():
            self.first_point_spin.setValue(int(result["first_point"]))
        self.summary_label.setText(self._describe(result, s))
        self._draw(result, s)
        return True

    def process(self) -> bool:
        """Run :func:`CPMGprocess` and emit the result. Returns False on a
        problem (shown, not raised)."""
        if not self.check_split():
            return False
        s = self.settings()
        path = self._source.metadata.source_path
        procno = int(self._source.metadata.procno or 1)
        try:
            results = nb_one_d.CPMGprocess(path, procno=procno, **s)
            spectrum = nb_one_d.cpmg_to_spectrum(results, procno=procno)
        except (ValueError, OSError) as exc:
            self._log(f"CPMG processing failed: {exc}")
            return False
        label = f"CPMG {os.path.basename(os.path.normpath(path))}"
        self._log(self._report(results, s, label))
        self.spectrum_ready.emit(spectrum, label)
        return True

    # --- internals ----------------------------------------------------
    @staticmethod
    def _index_spin(value: int, array: str, tip: str) -> QSpinBox:
        spin = TypedSpinBox()
        spin.setRange(0, 63)
        spin.setPrefix(array.lower())
        spin.setValue(value)
        spin.setToolTip(tip + " Change it if your pulse programme keeps it elsewhere.")
        return spin

    def _used_range(self, result, s) -> tuple[int, int]:
        n = result["num_echoes"]
        start = min(s["cutout"], n)
        stop = n if s["num_echoes_added"] is None else min(n, start + s["num_echoes_added"])
        return start, stop

    def _describe(self, result, s) -> str:
        start, stop = self._used_range(result, s)
        idx = result["indices"]
        text = (
            f"Echo length d{idx['echo_delay']} = {result['echo_length_us']:g} µs; "
            f"blanking 2·d{idx['blanking_delay']} + cnst{idx['blanking_cnst']} = "
            f"{result['blanking_us']:g} µs; dwell {result['dwell_us']:g} µs.\n"
            f"Period {result['period_exact']:.3f} points; {result['num_echoes']} echoes "
            f"(l{idx['num_echoes']}); first point {result['first_point']}; "
            f"{result['echo_points']} points kept per echo.\n"
            f"Co-adding echoes {start + 1}–{stop} ({stop - start}).")
        if abs(result["period_exact"] - round(result["period_exact"])) > 1e-6:
            text += ("\nThe period is not a whole number of points; each echo is cut at its "
                     "own rounded position so the split does not drift along the train.")
        return text

    def _report(self, results, s, label) -> str:
        zero_fill = s["zero_fill"] if s["zero_fill"] != "auto" else f"auto ({len(results['x_ppm'])})"
        window = ("off" if not s["apodize"] else
                  f"zeroed edges {s['apod_offset_zero']}, shape {s['apod_shape']:g}, alpha {s['apod_alpha']:g}")
        return (
            f"{label}: co-added {results['num_echoes_used']} of {results['num_echoes']} echoes "
            f"(skipped {results['cutout']}), first point {results['first_point']}, "
            f"period {results['period_exact']:.3f} points, {len(results['fid_coadded'])} points per echo; "
            f"apodisation {window}; zero-filled to {zero_fill}; "
            f"{'magnitude' if s['magnitude_mode'] else 'phase-sensitive'}; "
            f"referenced to SF {results['SF']:.6f} MHz (SR {results['SR']:g} Hz).")

    def _draw(self, result, s) -> None:
        fig = self.figure
        fig.clear()
        palette = self.palette()
        fg = palette.color(QPalette.ColorRole.WindowText).name()
        bg = palette.color(QPalette.ColorRole.Base).name()
        fig.set_facecolor(bg)
        ax = fig.add_subplot(111)
        ax.set_facecolor(bg)
        fid = np.abs(result["fid"])
        start, stop = self._used_range(result, s)
        period, trim = result["period"], result["trim"]
        last = int(result["starts"][-1]) + period
        lo = max(0, int(result["first_point"]) - period)
        hi = min(len(fid), last + period)
        x = np.arange(lo, hi)
        ax.plot(x, fid[lo:hi], color=fg, lw=0.6)
        for k, s0 in enumerate(result["starts"]):
            used = start <= k < stop
            # Alternate the shade echo to echo, so each cut is visible
            # even when the windows touch (no trim).
            shade = (0.30 if k % 2 else 0.16) if used else (0.14 if k % 2 else 0.07)
            ax.axvspan(s0 + trim, s0 + period - trim, color="#0072B2" if used else "#999999",
                       alpha=shade, lw=0)
        ax.axvline(result["first_point"], color="#D55E00", lw=0.8, ls="--")
        ax.set_xlim(lo, hi)
        ax.set_xlabel("Raw FID point", color=fg)
        ax.set_ylabel("|FID|", color=fg)
        ax.tick_params(colors=fg, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(fg)
        ax.set_title("Blue: echoes co-added; grey: left out; dashed: first point",
                     fontsize=8, color=fg)
        self.preview_canvas.draw_idle()

    def _log(self, text: str) -> None:
        self.log_edit.appendPlainText(text)
