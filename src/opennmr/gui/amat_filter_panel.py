"""
Advanced Processing > aMAT Spectral Filter (2D): the notebook's
:mod:`opennmr.notebook.amat_filter` as a panel (explicit user request,
2026-11-13; Angew. Chem. Int. Ed. 2024, e202408704, SI section S7).

Each F1 row of an aMAT 2D can only carry signal at its own isotropic
position plus whole multiples of the spinning rate; the filter keeps those
positions (a train of pulses of a chosen width) and removes the rest -- the
artefacts of intense signals that hide weak ones. *Show Filter* draws the
2D filter (its centre-pulse ridge on the diagonal), *Show Trace* a row
before and after with its pulse train, and *Filter → New Spectrum* adds the
filtered 2D to Open experiments with the log in its history.

All the maths is in :mod:`opennmr.notebook.amat_filter`; this module only
collects settings and shows results.
"""

from __future__ import annotations

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox, QDoubleSpinBox, QFormLayout, QHBoxLayout, QLabel, QPlainTextEdit, QPushButton,
    QVBoxLayout, QWidget,
)

from ..notebook import amat_filter as nb_amat


def amat_source_problem(spectrum) -> str | None:
    """Why ``spectrum`` cannot be filtered, or ``None`` if it can."""
    if spectrum is None:
        return "Select an aMAT 2D spectrum in Open experiments."
    if spectrum.ndim != 2:
        return "The aMAT filter needs a 2D spectrum; the selected one is 1D."
    if spectrum.metadata.pseudo_2d or spectrum.axes[0].kind != "frequency":
        return "The selected spectrum is a pseudo-2D series, whose F1 is not a frequency."
    return None


def _spin(minimum, maximum, decimals, value, step, suffix="", tip=""):
    w = QDoubleSpinBox()
    w.setRange(minimum, maximum)
    w.setDecimals(decimals)
    w.setSingleStep(step)
    w.setValue(value)
    if suffix:
        w.setSuffix(suffix)
    if tip:
        w.setToolTip(tip)
    return w


class AmatFilterPanel(QWidget):
    """Settings, the filter and trace views, and Filter → New Spectrum.

    ``set_source(spectrum, label)`` follows the Open experiments
    selection; ``result_ready(NMRSpectrum, label, log)`` is emitted with
    the filtered spectrum."""

    result_ready = Signal(object, str, str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._source = None
        self._source_label = ""
        self._last = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        intro = QLabel(
            "In an aMAT 2D each F1 row can carry real signal only at its own isotropic "
            "position plus whole multiples of the spinning rate. This filter multiplies each "
            "row by a train of pulses at those positions and removes the rest: the artefacts "
            "of intense signals that hide weak ones at fast MAS (Angew. Chem. Int. Ed. 2024, "
            "e202408704, SI S7).")
        intro.setWordWrap(True)
        layout.addWidget(intro)
        self.source_label = QLabel("")
        self.source_label.setWordWrap(True)
        layout.addWidget(self.source_label)

        form = QFormLayout()
        self.mas_spin = _spin(0.0, 1e4, 3, 0.0, 1.0, " kHz",
                              "Spinning rate ν_rot: the period of the pulse train (read from the "
                              "experiment when recorded).")
        form.addRow("MAS rate:", self.mas_spin)
        width_row = QHBoxLayout()
        self.width_mode_combo = QComboBox()
        self.width_mode_combo.addItem("Fraction of ν_rot", "fraction")
        self.width_mode_combo.addItem("kHz", "khz")
        self.width_spin = _spin(0.001, 1e4, 4, 1.0 / 3.0, 0.05, "",
                                "Width of each pulse. The SI uses ν_rot/3. Narrower removes more "
                                "artefacts and noise, but the pulses must stay broader than the "
                                "homogeneous linewidth 1/(πT2′).")
        width_row.addWidget(self.width_spin, 1)
        width_row.addWidget(self.width_mode_combo)
        form.addRow("Pulse width:", width_row)
        self.shape_combo = QComboBox()
        for key, text in nb_amat.SHAPES.items():
            self.shape_combo.addItem(text, key)
        self.shape_combo.setToolTip(
            "Hann: each pulse tapers to zero at its edges, emphasising the signal at its "
            "centre (the SI's choice); its half-height width is half the pulse width. Flat: "
            "everything inside the pulse is kept unchanged.")
        form.addRow("Pulse shape:", self.shape_combo)
        self.t2_spin = _spin(0.0, 1e6, 3, 0.0, 0.1, " ms",
                             "Homogeneous T2′, to check the pulse width against 1/(πT2′) "
                             "(0: not checked).")
        form.addRow("T2′ (check):", self.t2_spin)
        self.offset_spin = _spin(-1e4, 1e4, 3, 0.0, 0.1, " ppm",
                                 "Moves every pulse train relative to its F1 row: for an F1 axis "
                                 "referenced or scaled a little differently from F2. The centre "
                                 "pulses should lie on the diagonal ν1 = ν2.")
        form.addRow("F1 offset:", self.offset_spin)
        self.trace_spin = _spin(-1e5, 1e5, 2, 0.0, 1.0, " ppm",
                                "The F1 position (nearest row) of the trace to show.")
        form.addRow("Trace at F1:", self.trace_spin)
        layout.addLayout(form)
        self.width_mode_combo.currentIndexChanged.connect(self._on_width_mode)

        buttons = QHBoxLayout()
        self.filter_view_button = QPushButton("Show Filter")
        self.filter_view_button.setToolTip("The 2D filter, with the centre pulses' ridge in red.")
        self.filter_view_button.clicked.connect(lambda: self.show_filter())
        self.trace_button = QPushButton("Show Trace")
        self.trace_button.setToolTip("The row at 'Trace at F1' before and after, with its pulse "
                                     "train above it.")
        self.trace_button.clicked.connect(lambda: self.show_trace())
        self.apply_button = QPushButton("Filter → New Spectrum")
        self.apply_button.setToolTip("Add the filtered 2D to Open experiments, with its log.")
        self.apply_button.clicked.connect(lambda: self.apply())
        self.python_button = QPushButton("Copy as Python")
        self.python_button.setToolTip("Notebook code that filters the same file the same way.")
        self.python_button.clicked.connect(self._copy_python)
        for b in (self.filter_view_button, self.trace_button, self.apply_button,
                  self.python_button):
            buttons.addWidget(b)
        layout.addLayout(buttons)

        # No layout engine of its own: the notebook plots lay themselves out
        # with tight_layout (a constrained figure would warn and switch).
        self.figure = Figure(figsize=(4.0, 3.0))
        self.preview_canvas = FigureCanvasQTAgg(self.figure)
        self.preview_canvas.setMinimumHeight(240)
        self.preview_toolbar = NavigationToolbar2QT(self.preview_canvas, self)
        layout.addWidget(self.preview_toolbar)
        layout.addWidget(self.preview_canvas)

        self.log_edit = QPlainTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setMaximumBlockCount(500)
        self.log_edit.setMinimumHeight(80)
        self.log_edit.setPlaceholderText("What was done, and any warning, appears here.")
        layout.addWidget(self.log_edit)
        self.set_source(None)

    # --- public -----------------------------------------------------------
    def set_source(self, spectrum, label: str = "") -> None:
        """Point the panel at the selected spectrum; a recorded MAS rate
        fills the MAS field. A filtered result keeps its parent as the
        source."""
        if (spectrum is not None and "amat_filter" in (spectrum.metadata.raw or {})
                and self._source is not None and amat_source_problem(self._source) is None):
            return
        changed = spectrum is not self._source
        self._source, self._source_label = spectrum, label
        problem = amat_source_problem(spectrum)
        for b in (self.filter_view_button, self.trace_button, self.apply_button,
                  self.python_button):
            b.setEnabled(problem is None)
        if problem:
            self.source_label.setText(problem)
            return
        nuc = spectrum.metadata.nucleus or "?"
        self.source_label.setText(f"Spectrum: {label or spectrum.metadata.source_path} ({nuc})")
        rate = nb_amat.recorded_mas_rate(spectrum)
        if changed and rate:
            self.mas_spin.setValue(rate / 1e3)
        if changed:
            self._last = None
            self.figure.clear()
            self.preview_canvas.draw_idle()

    def arguments(self) -> dict:
        """The keyword arguments of :func:`nb_amat.apply_amat_filter`."""
        rate = self.mas_spin.value() * 1e3
        if rate <= 0:
            raise ValueError("Give the MAS rate.")
        args = {"nu_rot_hz": rate, "shape": self.shape_combo.currentData(),
                "f1_offset_ppm": self.offset_spin.value()}
        if self.width_mode_combo.currentData() == "fraction":
            args["width_fraction"] = self.width_spin.value()
        else:
            args["width_hz"] = self.width_spin.value() * 1e3
        if self.t2_spin.value() > 0:
            args["t2_prime_s"] = self.t2_spin.value() * 1e-3
        return args

    def run(self):
        """Filter the source (nothing is added); the result or None."""
        if amat_source_problem(self._source):
            return None
        try:
            result = nb_amat.apply_amat_filter(self._source, source_label=self._source_label,
                                               **self.arguments())
        except ValueError as exc:
            self._log(f"Not possible: {exc}")
            return None
        self._last = result
        return result

    def show_filter(self) -> bool:
        result = self.run()
        if result is None:
            return False
        self.figure.clear()
        nb_amat.plot_amat_filter_map(self._source, result, figure=self.figure, fontsize=8)
        self.preview_canvas.draw_idle()
        return True

    def show_trace(self) -> bool:
        result = self.run()
        if result is None:
            return False
        self.figure.clear()
        nb_amat.plot_amat_trace(self._source, result, self.trace_spin.value(),
                                figure=self.figure, fontsize=8)
        self.preview_canvas.draw_idle()
        return True

    def apply(self):
        """Filter and emit the result. Returns it, or None."""
        result = self.run()
        if result is None:
            return None
        log = result.metadata.raw["amat_filter"]["log"]
        self._log(log)
        label = f"aMAT-filtered {self._source_label}".strip()
        self.result_ready.emit(result, label, log)
        return result

    def python_code(self) -> str:
        """Notebook code repeating the filter on the same file."""
        args = ", ".join(f"{k}={v!r}" for k, v in self.arguments().items())
        path = getattr(self._source.metadata, "source_path", None) if self._source else None
        return ("from opennmr import io\nfrom opennmr.notebook import amat_filter as af\n"
                f"spectrum = io.load({path!r})\n"
                f"filtered = af.apply_amat_filter(spectrum, {args})\n"
                "print(filtered.metadata.raw['amat_filter']['log'])\n")

    # --- internals ----------------------------------------------------------
    def _copy_python(self) -> None:
        from PySide6.QtGui import QGuiApplication
        try:
            QGuiApplication.clipboard().setText(self.python_code())
        except ValueError as exc:
            self._log(f"Not possible: {exc}")

    def _on_width_mode(self, _i) -> None:
        rate_khz = self.mas_spin.value()
        if self.width_mode_combo.currentData() == "fraction":
            self.width_spin.setSuffix("")
            if rate_khz > 0:
                self.width_spin.setValue(self.width_spin.value() / rate_khz)
        else:
            self.width_spin.setSuffix(" kHz")
            if rate_khz > 0:
                self.width_spin.setValue(self.width_spin.value() * rate_khz)

    def _log(self, text: str) -> None:
        self.log_edit.appendPlainText(text)
