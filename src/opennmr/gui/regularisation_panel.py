"""Dynamics ▸ Regularisation (Tikhonov): distributions of T1, T2 or D, and
T1-T2, T2-T2 and D-T2 maps -- the MATLAB_RELAX_ALGORITHM scripts, in the GUI
(explicit user request, 2026-10-16).

All the maths is :mod:`opennmr.notebook.regularisation`; "Copy notebook
code" gives the same call. Data come from Fit Curve's last result (one
peak's series, already in seconds or s/m²), or from text files: two
columns for 1D (the MATLAB scripts' input), a matrix and two value lists
for 2D. The result is drawn on the Fit view; its display options are on
the plot's right-click menu (Regularisation Plot…).
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPlainTextEdit, QPushButton,
    QVBoxLayout, QWidget,
)

from .widgets.typed_spinbox import TypedSpinBox
from ..notebook import regularisation as reg
from .plot_options_dialog import PlotOptionsDialog
from .widgets.report_label import ReportLabel

#: Units a time is typed in, to seconds.
TIME_UNITS = (("ms", 1e-3), ("s", 1.0), ("µs", 1e-6))
#: Units of a value list read from a file: times, or b-values.
VALUE_UNITS = (("ms", 1e-3), ("s", 1.0), ("µs", 1e-6), ("b (s/m²)", 1.0))


def load_columns(path: str) -> np.ndarray:
    """Numbers from a text file -- comma, tab or space separated, ``#``
    comments and any header lines that are not numbers skipped -- as a 2D
    array (one row per line)."""
    rows = []
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            parts = line.replace(",", " ").replace(";", " ").replace("\t", " ").split()
            try:
                rows.append([float(p) for p in parts])
            except ValueError:
                continue          # a header line
    if not rows:
        raise ValueError(f"No numbers found in {path}.")
    width = min(len(r) for r in rows)
    return np.asarray([r[:width] for r in rows], dtype=float)


def _kernel_combo(default: str) -> QComboBox:
    combo = QComboBox()
    for key, (label, _axis, _k) in reg.KERNELS.items():
        combo.addItem(label, key)
    combo.setCurrentIndex(combo.findData(default))
    return combo


def _value_spin(value: float) -> QDoubleSpinBox:
    spin = QDoubleSpinBox()
    spin.setDecimals(6)
    spin.setRange(1e-15, 1e15)
    spin.setValue(value)
    return spin


class _GridRow(QWidget):
    """From / to / points / unit for one axis of the grid."""

    def __init__(self, kernel: str, n: int, parent=None) -> None:
        super().__init__(parent)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        self.low = QLineEdit()
        self.high = QLineEdit()
        for edit in (self.low, self.high):
            edit.setMaximumWidth(90)
        self.unit = QComboBox()
        self.points = TypedSpinBox()
        self.points.setRange(4, 1000)
        self.points.setValue(n)
        self.points.setSuffix(" pts")
        row.addWidget(self.low)
        row.addWidget(QLabel("to"))
        row.addWidget(self.high)
        row.addWidget(self.unit)
        row.addWidget(self.points)
        self.kernel = None
        self.set_kernel(kernel)

    def set_kernel(self, kernel: str) -> None:
        if kernel == self.kernel:
            return
        self.kernel = kernel
        self.unit.clear()
        if kernel == "diffusion":
            self.unit.addItem("m²/s", 1.0)
        else:
            for label, scale in TIME_UNITS:
                self.unit.addItem(label, scale)
        self.set_range(reg.DEFAULT_RANGES[kernel])

    def set_range(self, low_high) -> None:
        scale = float(self.unit.currentData())
        self.low.setText(f"{low_high[0] / scale:.4g}")
        self.high.setText(f"{low_high[1] / scale:.4g}")

    def grid_range(self) -> tuple:
        scale = float(self.unit.currentData())
        try:
            lo, hi = float(self.low.text()) * scale, float(self.high.text()) * scale
        except ValueError as exc:
            raise ValueError("Type the grid's two ends as numbers.") from exc
        if not (0 < lo < hi):
            raise ValueError("The grid needs 0 < from < to.")
        return lo, hi


class RegularisationPanel(QWidget):
    """``canvas``/``show_view`` as the other Dynamics panels.
    ``curve_series()``: Fit Curve's last result as ``[{"label", "x" (s or
    s/m²), "y", "kernel" (a guess: 't2', 't1_ir', 't1_sr' or
    'diffusion')}]``, or []."""

    def __init__(self, canvas=None, show_view=None, curve_series=None, parent=None) -> None:
        super().__init__(parent)
        self.canvas = canvas
        self.show_view = show_view or (lambda: None)
        self.curve_series = curve_series or (lambda: [])
        self.result = None
        self._last_call = None
        self._file_1d = None
        self._files_2d = {"data": None, "x1": None, "x2": None}
        self._build()
        if canvas is not None and hasattr(canvas, "register_plot_options"):
            canvas.register_plot_options(
                "regularisation", "Regularisation Plot…", self.plot_options_dialog.open_options,
                "Panels shown, the NNLS comparison, units, and the map's colours.")

    # ---------------------------------------------------------------- UI
    def _build(self) -> None:
        layout = QVBoxLayout(self)
        blurb = QLabel(
            "A distribution of T₁, T₂ or D (or a T₁–T₂, T₂–T₂ or D–T₂ map) from a decay, by "
            "Tikhonov regularisation — the MATLAB_RELAX_ALGORITHM scripts. Ill-posed: trust "
            "peak positions and log means, not peak widths. Where one to three components fit, "
            "Fit Curve's values (with errors) are the better answer.")
        blurb.setWordWrap(True)
        layout.addWidget(blurb)

        top = QFormLayout()
        self.dimension_combo = QComboBox()
        self.dimension_combo.addItem("1D distribution", "1d")
        self.dimension_combo.addItem("2D map (T₁–T₂, T₂–T₂, D–T₂)", "2d")
        self.dimension_combo.currentIndexChanged.connect(self._on_dimension_changed)
        top.addRow("Kind:", self.dimension_combo)
        layout.addLayout(top)

        # Two pages, one shown (a QStackedWidget keeps the taller page's
        # height and left a gap under the 1D one).
        self.pages = [self._build_1d(), self._build_2d()]
        for page in self.pages:
            layout.addWidget(page)
        self._on_dimension_changed(0)

        solve = QGroupBox("Regularisation")
        form = QFormLayout(solve)
        self.baseline_spin = QDoubleSpinBox()
        self.baseline_spin.setRange(-1e15, 1e15)
        self.baseline_spin.setDecimals(6)
        self.baseline_spin.setToolTip("Subtracted from every point first (the scripts' "
                                      "\"Manual Baseline\"). A constant offset is also fitted.")
        form.addRow("Baseline:", self.baseline_spin)
        lam_row = QHBoxLayout()
        self.lambda_auto_check = QCheckBox("Automatic")
        self.lambda_auto_check.setChecked(True)
        self.lambda_auto_check.setToolTip(
            "MATLAB scheme: λ minimising the Brezinski–Rodriguez–Seatzu error estimate (η₃), "
            "by bisection (errestik). Non-negative solver: the corner of its L-curve.")
        self.lambda_edit = QLineEdit("1")
        self.lambda_edit.setMaximumWidth(90)
        self.lambda_edit.setEnabled(False)
        self.lambda_auto_check.toggled.connect(lambda on: self.lambda_edit.setEnabled(not on))
        lam_row.addWidget(self.lambda_auto_check)
        lam_row.addWidget(QLabel("or λ ="))
        lam_row.addWidget(self.lambda_edit)
        lam_row.addStretch(1)
        form.addRow("λ:", lam_row)
        self.solver_combo = QComboBox()
        for key, label in reg.SOLVERS.items():
            self.solver_combo.addItem(label, key)
        self.solver_combo.setToolTip(
            "MATLAB scheme: Tikhonov, then n times with |x| as the prior — each pass weakens "
            "the smoothing, so the iterations act as a second regularisation parameter, and a "
            "little negative amplitude can remain.\n\n"
            "Non-negative: the same functional with x ≥ 0 imposed exactly.")
        self.solver_combo.currentIndexChanged.connect(self._sync)
        form.addRow("Solver:", self.solver_combo)
        self.iterations_spin = TypedSpinBox()
        self.iterations_spin.setRange(0, 100000)
        self.iterations_spin.setValue(100)
        self.iterations_spin.setToolTip("The scripts' \"Number of Fitting iterations\". Keep it "
                                        "the same when comparing samples.")
        self.iterations_label = QLabel("Iterations:")
        form.addRow(self.iterations_label, self.iterations_spin)
        layout.addWidget(solve)

        actions = QHBoxLayout()
        self.run_button = QPushButton("Regularise")
        self.run_button.clicked.connect(self.run)
        self.export_button = QPushButton("Export CSV…")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.export_csv)
        self.code_button = QPushButton("Copy notebook code")
        self.code_button.setEnabled(False)
        self.code_button.clicked.connect(self.copy_notebook_code)
        for b in (self.run_button, self.export_button, self.code_button):
            actions.addWidget(b)
        actions.addStretch(1)
        layout.addLayout(actions)

        self.results_label = ReportLabel("")
        self.results_label.setWordWrap(True)
        self.results_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.results_label)
        hint = QLabel("Plot panels, units and colours: right-click the plot ▸ "
                      "Regularisation Plot…")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: palette(mid);")
        layout.addWidget(hint)
        self.status_log = QPlainTextEdit()
        self.status_log.setReadOnly(True)
        self.status_log.setMaximumHeight(140)
        self.status_log.setPlaceholderText("What was done, and how each number was arrived at.")
        layout.addWidget(self.status_log)

        self._build_plot_options()
        self._sync()

    def _build_1d(self) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)
        form.setContentsMargins(0, 0, 0, 0)
        self.source_combo = QComboBox()
        self.source_combo.addItem("Fit Curve's last result", "curve")
        self.source_combo.addItem("Text file (x and y columns)", "file")
        self.source_combo.currentIndexChanged.connect(self._sync)
        form.addRow("Data:", self.source_combo)

        self.series_combo = QComboBox()
        self.series_combo.setToolTip("Which peak's series, from the last Fit Curve "
                                     "(its excluded points stay excluded).")
        self.series_combo.currentIndexChanged.connect(self._on_series_changed)
        refresh = QPushButton("Refresh")
        refresh.setToolTip("Re-read Fit Curve's last result.")
        refresh.clicked.connect(self.refresh)
        series_row = QHBoxLayout()
        series_row.addWidget(self.series_combo, 1)
        series_row.addWidget(refresh)
        self.series_label = QLabel("Peak:")
        form.addRow(self.series_label, series_row)

        self.file_row = QWidget()
        file_layout = QHBoxLayout(self.file_row)
        file_layout.setContentsMargins(0, 0, 0, 0)
        self.file_button = QPushButton("Choose…")
        self.file_button.clicked.connect(self._choose_file_1d)
        self.file_name_label = QLabel("no file")
        self.x_column_spin = TypedSpinBox()
        self.y_column_spin = TypedSpinBox()
        for spin, value in ((self.x_column_spin, 1), (self.y_column_spin, 2)):
            spin.setRange(1, 99)
            spin.setValue(value)
        self.x_unit_combo = QComboBox()
        for label, scale in VALUE_UNITS:
            self.x_unit_combo.addItem(label, scale)
        file_layout.addWidget(self.file_button)
        file_layout.addWidget(self.file_name_label, 1)
        self.file_label = QLabel("File:")
        form.addRow(self.file_label, self.file_row)
        self.columns_row = QWidget()
        cols = QHBoxLayout(self.columns_row)
        cols.setContentsMargins(0, 0, 0, 0)
        cols.addWidget(QLabel("x col."))
        cols.addWidget(self.x_column_spin)
        cols.addWidget(QLabel("in"))
        cols.addWidget(self.x_unit_combo)
        cols.addWidget(QLabel("y col."))
        cols.addWidget(self.y_column_spin)
        cols.addStretch(1)
        self.columns_label = QLabel("Columns:")
        form.addRow(self.columns_label, self.columns_row)

        self.kernel_combo = _kernel_combo("t2")
        self.kernel_combo.currentIndexChanged.connect(
            lambda _i: self.grid_row.set_kernel(self.kernel_combo.currentData()))
        form.addRow("Kernel:", self.kernel_combo)
        self.grid_row = _GridRow("t2", 100)
        self.grid_row.setToolTip("The log-spaced grid the distribution is found on. Make it "
                                 "reach past the shortest and longest values in the sample.")
        auto = QPushButton("From data")
        auto.setToolTip("A third of the shortest delay to three times the longest "
                        "(for D: from the b range).")
        auto.clicked.connect(self._range_from_data)
        grid = QHBoxLayout()
        grid.addWidget(self.grid_row, 1)
        grid.addWidget(auto)
        form.addRow("Grid:", grid)
        return page

    def _build_2d(self) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)
        form.setContentsMargins(0, 0, 0, 0)
        note = QLabel("A data matrix — one row per indirect value (recovery delay, first T₂ "
                      "or b), one column per point of the echo train — and the two value "
                      "lists, each a one-column text file.")
        note.setWordWrap(True)
        form.addRow(note)
        self.file_labels_2d = {}
        for key, label in (("data", "Data matrix:"), ("x1", "Direct values (x1):"),
                           ("x2", "Indirect values (x2):")):
            row = QHBoxLayout()
            button = QPushButton("Choose…")
            button.clicked.connect(lambda _c=False, k=key: self._choose_file_2d(k))
            name = QLabel("no file")
            self.file_labels_2d[key] = name
            row.addWidget(button)
            row.addWidget(name, 1)
            if key != "data":
                unit = QComboBox()
                for text, scale in VALUE_UNITS:
                    unit.addItem(text, scale)
                setattr(self, f"{key}_unit_combo", unit)
                row.addWidget(unit)
            form.addRow(label, row)
        self.transpose_check = QCheckBox("Rows are the direct values (transpose)")
        form.addRow(self.transpose_check)
        self.kernel1_combo = _kernel_combo("t2")
        self.kernel2_combo = _kernel_combo("t1_ir")
        self.grid1_row = _GridRow("t2", 40)
        self.grid2_row = _GridRow("t1_ir", 40)
        self.kernel1_combo.currentIndexChanged.connect(
            lambda _i: self.grid1_row.set_kernel(self.kernel1_combo.currentData()))
        self.kernel2_combo.currentIndexChanged.connect(
            lambda _i: self.grid2_row.set_kernel(self.kernel2_combo.currentData()))
        form.addRow("Direct kernel:", self.kernel1_combo)
        form.addRow("Direct grid:", self.grid1_row)
        form.addRow("Indirect kernel:", self.kernel2_combo)
        form.addRow("Indirect grid:", self.grid2_row)
        thin = QHBoxLayout()
        self.keep_first_spin = TypedSpinBox()
        self.keep_first_spin.setRange(0, 100000)
        self.keep_first_spin.setValue(0)
        self.step_spin = TypedSpinBox()
        self.step_spin.setRange(1, 1000)
        self.step_spin.setValue(1)
        thin.addWidget(QLabel("all of the first"))
        thin.addWidget(self.keep_first_spin)
        thin.addWidget(QLabel("then every"))
        thin.addWidget(self.step_spin)
        thin.addStretch(1)
        thin_widget = QWidget()
        thin_widget.setLayout(thin)
        thin_widget.setToolTip("Thin a long echo train (the scripts' \"Step after\" and "
                               "\"Step\"), so the problem stays a manageable size.")
        form.addRow("Echo points:", thin_widget)
        return page

    def _build_plot_options(self) -> None:
        options = QWidget()
        form = QFormLayout(options)
        self.time_unit_combo = QComboBox()
        for label, _scale in TIME_UNITS:
            self.time_unit_combo.addItem(label, label)
        self.show_cumulative_check = QCheckBox("Cumulative distribution")
        self.show_cumulative_check.setChecked(True)
        self.show_residuals_check = QCheckBox("Residuals")
        self.show_residuals_check.setChecked(True)
        self.show_nnls_check = QCheckBox("NNLS (no regularisation) for comparison")
        self.cmap_combo = QComboBox()
        for name in ("viridis", "magma", "cividis", "Blues", "Greys"):
            self.cmap_combo.addItem(name, name)
        self.levels_spin = TypedSpinBox()
        self.levels_spin.setRange(2, 40)
        self.levels_spin.setValue(6)
        self.projections_check = QCheckBox("Projections on the map")
        self.projections_check.setChecked(True)
        form.addRow("Time unit:", self.time_unit_combo)
        form.addRow("1D:", self.show_cumulative_check)
        form.addRow("", self.show_residuals_check)
        form.addRow("", self.show_nnls_check)
        form.addRow("2D colours:", self.cmap_combo)
        form.addRow("Contour levels:", self.levels_spin)
        form.addRow("", self.projections_check)
        for w in (self.time_unit_combo, self.cmap_combo):
            w.currentIndexChanged.connect(lambda _i: self.redraw())
        for w in (self.show_cumulative_check, self.show_residuals_check, self.show_nnls_check,
                  self.projections_check):
            w.toggled.connect(lambda _on: self.redraw())
        self.levels_spin.valueChanged.connect(lambda _v: self.redraw())
        self.plot_options_dialog = PlotOptionsDialog("Regularisation: plot", options, self)

    # ------------------------------------------------------------ state
    def _sync(self, *_args) -> None:
        curve = self.source_combo.currentData() == "curve"
        for w in (self.series_label, self.series_combo):
            w.setVisible(curve)
        for w in (self.file_label, self.file_row, self.columns_label, self.columns_row):
            w.setVisible(not curve)
        matlab = self.solver_combo.currentData() == "matlab"
        self.iterations_spin.setVisible(matlab)
        self.iterations_label.setVisible(matlab)

    def _on_dimension_changed(self, index: int) -> None:
        for i, page in enumerate(self.pages):
            page.setVisible(i == index)

    def refresh(self) -> None:
        """Re-read Fit Curve's series into the peak list."""
        current = self.series_combo.currentText()
        self._series = list(self.curve_series() or [])
        self.series_combo.blockSignals(True)
        self.series_combo.clear()
        for k, entry in enumerate(self._series):
            self.series_combo.addItem(entry.get("label") or f"Peak {k + 1}", k)
        index = self.series_combo.findText(current)
        self.series_combo.setCurrentIndex(max(index, 0))
        self.series_combo.blockSignals(False)
        if not self._series:
            self.series_combo.addItem("(run Fit Curve first)", None)
        self._on_series_changed()

    def _on_series_changed(self, *_args) -> None:
        entry = self._current_series()
        if entry is None:
            return
        kernel = entry.get("kernel")
        if kernel in reg.KERNELS and self.kernel_combo.currentData() != kernel:
            self.kernel_combo.setCurrentIndex(self.kernel_combo.findData(kernel))
        self.grid_row.set_range(reg.suggest_range(self.kernel_combo.currentData(), entry["x"]))

    def _current_series(self):
        series = getattr(self, "_series", None) or []
        k = self.series_combo.currentData()
        if k is None or not (0 <= k < len(series)):
            return None
        return series[k]

    def _range_from_data(self) -> None:
        try:
            x, _y, _label = self._data_1d()
        except ValueError as exc:
            QMessageBox.information(self, "No data", str(exc))
            return
        self.grid_row.set_range(reg.suggest_range(self.kernel_combo.currentData(), x))

    def _choose_file_1d(self) -> None:
        from .data_directory import start_directory
        path, _ = QFileDialog.getOpenFileName(self, "Decay data (x and y columns)",
                                              start_directory(),
                                              "Text (*.txt *.dat *.csv *.asc);;All files (*)")
        if path:
            self.set_file_1d(path)

    def set_file_1d(self, path: str) -> None:
        self._file_1d = path
        self.file_name_label.setText(path.replace("\\", "/").rsplit("/", 1)[-1])
        self.source_combo.setCurrentIndex(self.source_combo.findData("file"))

    def _choose_file_2d(self, key: str) -> None:
        from .data_directory import start_directory
        path, _ = QFileDialog.getOpenFileName(self, "Text file", start_directory(),
                                              "Text (*.txt *.dat *.csv *.asc);;All files (*)")
        if path:
            self.set_file_2d(key, path)

    def set_file_2d(self, key: str, path: str) -> None:
        self._files_2d[key] = path
        self.file_labels_2d[key].setText(path.replace("\\", "/").rsplit("/", 1)[-1])

    # ------------------------------------------------------------- data
    def _data_1d(self):
        if self.source_combo.currentData() == "curve":
            entry = self._current_series()
            if entry is None:
                raise ValueError("Run Fit Curve first (Dynamics ▸ Fit Curve): its series is "
                                 "what is regularised. Or choose a text file.")
            return (np.asarray(entry["x"], float), np.asarray(entry["y"], float),
                    entry.get("label"))
        if not self._file_1d:
            raise ValueError("Choose a text file with the x and y columns.")
        table = load_columns(self._file_1d)
        cx, cy = self.x_column_spin.value() - 1, self.y_column_spin.value() - 1
        if max(cx, cy) >= table.shape[1]:
            raise ValueError(f"The file has {table.shape[1]} column(s).")
        x = table[:, cx] * float(self.x_unit_combo.currentData())
        return x, table[:, cy], self.file_name_label.text()

    def _data_2d(self):
        missing = [k for k, v in self._files_2d.items() if not v]
        if missing:
            raise ValueError("Choose the data matrix and both value lists.")
        data = load_columns(self._files_2d["data"])
        if self.transpose_check.isChecked():
            data = data.T
        x1 = load_columns(self._files_2d["x1"])[:, 0] * float(self.x1_unit_combo.currentData())
        x2 = load_columns(self._files_2d["x2"])[:, 0] * float(self.x2_unit_combo.currentData())
        x1 = x1[:data.shape[1]] if x1.size > data.shape[1] else x1
        x2 = x2[:data.shape[0]] if x2.size > data.shape[0] else x2
        if data.shape != (x2.size, x1.size):
            raise ValueError(f"The matrix is {data.shape[0]} × {data.shape[1]} but the lists "
                             f"have {x2.size} (indirect) and {x1.size} (direct) values.")
        x1, data = reg.subsample(x1, data, self.keep_first_spin.value(), self.step_spin.value(),
                                 axis=1)
        return x1, x2, data

    def _lambda(self):
        if self.lambda_auto_check.isChecked():
            return None
        try:
            lam = float(self.lambda_edit.text())
        except ValueError as exc:
            raise ValueError("Type λ as a number, or tick Automatic.") from exc
        if lam <= 0:
            raise ValueError("λ must be positive.")
        return lam

    # ------------------------------------------------------------ actions
    def run(self) -> dict | None:
        try:
            common = {"baseline": self.baseline_spin.value(), "lam": self._lambda(),
                      "n_iter": self.iterations_spin.value(),
                      "solver": self.solver_combo.currentData()}
            if self.dimension_combo.currentData() == "1d":
                x, y, label = self._data_1d()
                kwargs = {"kernel": self.kernel_combo.currentData(),
                          "grid_range": self.grid_row.grid_range(),
                          "n_grid": self.grid_row.points.value(), **common}
                call = ("regularise", (x, y), kwargs)
            else:
                x1, x2, data = self._data_2d()
                label = self.file_labels_2d["data"].text()
                kwargs = {"kernel1": self.kernel1_combo.currentData(),
                          "kernel2": self.kernel2_combo.currentData(),
                          "range1": self.grid1_row.grid_range(),
                          "range2": self.grid2_row.grid_range(),
                          "n1": self.grid1_row.points.value(),
                          "n2": self.grid2_row.points.value(), **common}
                call = ("regularise_2d", (x1, x2, data), kwargs)
        except (ValueError, OSError) as exc:
            QMessageBox.information(self, "Regularisation", str(exc))
            return None
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            result = getattr(reg, call[0])(*call[1], label=label, **call[2])
        except (ValueError, np.linalg.LinAlgError, MemoryError) as exc:
            QMessageBox.warning(self, "Regularisation failed", str(exc))
            return None
        finally:
            QApplication.restoreOverrideCursor()
        self.result = result
        self._last_call = call
        self.export_button.setEnabled(True)
        self.code_button.setEnabled(True)
        text = reg.report(result, self.time_unit_combo.currentData())
        self.results_label.setText(text)
        self.status_log.appendPlainText(f"— {label or 'data'} —\n{text}\n")
        self.redraw(raise_view=True)
        return result

    def redraw(self, raise_view: bool = False) -> None:
        if self.canvas is None or self.result is None:
            return
        if not raise_view and getattr(self.canvas, "plot_owner", None) != "regularisation":
            return
        result = self.result
        unit = self.time_unit_combo.currentData()
        if result["kind"] == "1d":
            opts = {"show_cumulative": self.show_cumulative_check.isChecked(),
                    "show_residuals": self.show_residuals_check.isChecked(),
                    "show_nnls": self.show_nnls_check.isChecked()}
            draw = lambda fig: reg.plot_regularisation(result, fig=fig, time_unit=unit, **opts)  # noqa: E731
        else:
            opts = {"cmap": self.cmap_combo.currentData(), "levels": self.levels_spin.value(),
                    "show_projections": self.projections_check.isChecked()}
            draw = lambda fig: reg.plot_regularisation_2d(result, fig=fig, time_unit=unit, **opts)  # noqa: E731
        self.canvas.show_external_plot(draw, kind="curve")
        self.canvas.plot_owner = "regularisation"
        if raise_view:
            self.show_view()

    def export_csv(self) -> None:
        if self.result is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export regularisation", "regularisation.csv",
                                              "CSV (*.csv)")
        if path:
            reg.to_csv(self.result, path, time_unit=self.time_unit_combo.currentData())
            self.status_log.appendPlainText(f"Exported to {path}.")

    def notebook_code(self) -> str:
        if self._last_call is None:
            return ""
        name, arrays, kwargs = self._last_call

        def arr(a):
            a = np.asarray(a, dtype=float)
            if a.ndim == 1:
                return "np.array([" + ", ".join(f"{v:.8g}" for v in a) + "])"
            return "np.array([" + ",\n    ".join(
                "[" + ", ".join(f"{v:.8g}" for v in row) + "]" for row in a) + "])"

        names = ("x", "y") if name == "regularise" else ("x1", "x2", "data")
        lines = ["import numpy as np", "import opennmr.notebook as nmr", ""]
        lines += [f"{n} = {arr(a)}" for n, a in zip(names, arrays)]
        args = ", ".join(f"{k}={v!r}" for k, v in kwargs.items())
        plot = "plot_regularisation" if name == "regularise" else "plot_regularisation_2d"
        lines += ["", f"result = nmr.{name}({', '.join(names)}, {args})",
                  "print(nmr.regularisation_report(result))",
                  f"nmr.{plot}(result, time_unit={self.time_unit_combo.currentData()!r})"]
        return "\n".join(lines)

    def copy_notebook_code(self) -> None:
        QGuiApplication.clipboard().setText(self.notebook_code())
        self.status_log.appendPlainText("Notebook code copied to the clipboard.")
