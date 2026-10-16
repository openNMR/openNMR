"""Dynamics tile: relaxation against temperature and exchange lineshapes.

Two sections under Dynamics, below Fit Curve:

* :class:`VTRelaxationPanel` -- "Relaxation vs temperature (BPP /
  Arrhenius)". A table of T1/T2/T1rho values (or linewidths) against
  temperature, at one or more Larmor frequencies -- typed, pasted,
  imported from CSV, or taken from the Fit Curve result -- plotted as
  ln(T1) against 1000/T with the temperature on a second axis, and
  fitted with a BPP model (dipolar or quadrupolar, symmetric or
  asymmetric, one to three processes, optional background, Arrhenius or
  VFT; several fields jointly) or, for linewidths, the Hendrickson-Bray
  equation. Models are compared by AICc/BIC; distinct minima are
  reported.
* :class:`ExchangeLineshapePanel` -- "Exchange lineshape (ion
  hopping)". Sites and hops with Arrhenius rates, simulated over a set
  of temperatures (exact Bloch-McConnell) and stacked; or fitted to the
  spectra selected in Open experiments (a VT series fits Ea and the
  attempt frequency).

All the maths is in :mod:`opennmr.notebook.vt_relaxation` and
:mod:`opennmr.notebook.exchange`; Copy notebook code gives the same
numbers in a notebook. Plots are drawn by those modules' plot functions
onto the Fit view (``MplCanvas.show_external_plot``), so Plot Style
applies to them like any other figure.
"""

from __future__ import annotations

import csv
import io
import math
import re
from typing import Callable

import numpy as np
from .plot_options_dialog import PlotOptionsDialog
from .widgets.report_label import ReportLabel
from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QListWidget, QListWidgetItem,
    QMessageBox, QPlainTextEdit, QPushButton, QSpinBox, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)

from ..notebook import exchange as ex
from ..notebook import vt_relaxation as vt


def _float(text, default=None):
    try:
        text = str(text).strip().replace("−", "-")
        return float(text) if text else default
    except (TypeError, ValueError):
        return default


def parse_temperatures(text: str) -> list[float]:
    """'200, 225 250' or '200:350:25' (inclusive) -> temperatures."""
    text = text.strip()
    if not text:
        raise ValueError("Enter at least one temperature.")
    out = []
    for token in re.split(r"[,;\s]+", text):
        if not token:
            continue
        if ":" in token:
            parts = [_float(p) for p in token.split(":")]
            if len(parts) != 3 or any(p is None for p in parts) or parts[2] == 0:
                raise ValueError(f"'{token}' is not start:stop:step.")
            a, b, s = parts
            n = int(math.floor((b - a) / s + 1e-9)) + 1
            if n <= 0 or n > 500:
                raise ValueError(f"'{token}' gives {n} temperatures.")
            out.extend(a + i * s for i in range(n))
        else:
            v = _float(token)
            if v is None:
                raise ValueError(f"'{token}' is not a temperature.")
            out.append(v)
    if any(t <= 0 for t in out):
        raise ValueError("Temperatures are in kelvin and must be positive.")
    return out


def parse_fixed(text: str) -> dict:
    """'Ea_eV=0.3, beta = 1.8' -> {'Ea_eV': 0.3, 'beta': 1.8}."""
    out = {}
    for token in re.split(r"[,;]+", text or ""):
        token = token.strip()
        if not token:
            continue
        if "=" not in token:
            raise ValueError(f"'{token}' is not name=value.")
        name, value = (s.strip() for s in token.split("=", 1))
        v = _float(value)
        if not name or v is None:
            raise ValueError(f"'{token}' is not name=value.")
        out[name] = v
    return out


def _read_only_table(columns) -> QTableWidget:
    table = QTableWidget(0, len(columns))
    table.setHorizontalHeaderLabels(list(columns))
    table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
    table.horizontalHeader().setStretchLastSection(True)
    table.setMaximumHeight(200)
    return table


def _log_box() -> QPlainTextEdit:
    box = QPlainTextEdit()
    box.setReadOnly(True)
    box.setMaximumHeight(160)
    box.setPlaceholderText("What was done, and how each number was arrived at, appears here.")
    return box


def _copy(text: str) -> None:
    QGuiApplication.clipboard().setText(text)


def _text_size(canvas) -> float:
    """Annotation size on ``canvas``: a little under its axis-label size,
    so in-plot text follows Plot Style rather than staying at
    matplotlib's 'small'."""
    try:
        return 0.85 * float(canvas.style_settings()["labelsize"])
    except Exception:  # noqa: BLE001 -- no canvas (notebook use, tests)
        return 9.0


def _show(canvas, draw, kind, hz_per_ppm=None) -> None:
    """``draw(fig)`` on ``canvas``; the legend is switched on when the
    figure has one (several series), as the canvas's own plots do."""
    canvas.show_external_plot(draw, kind=kind, hz_per_ppm=hz_per_ppm)
    if any(ax.get_legend() is not None for ax in canvas.figure.axes):
        canvas.set_legend_visible(True)


# ======================================================================
# Relaxation vs temperature
# ======================================================================

class VTRelaxationPanel(QWidget):
    """Dynamics ▸ Relaxation vs temperature (BPP / Arrhenius).

    ``canvas`` : the MplCanvas to draw on; ``show_view()`` raises it.
    ``curve_fit_rows()`` : rows from the current Fit Curve result, each
    ``{"series", "T_K", "value", "error", "larmor_MHz", "quantity"}``
    (``T_K``/``larmor_MHz`` None when the spectrum does not say), or an
    empty list.
    """

    COLUMNS = ("Series", "T", "Value", "±", "Larmor (MHz)")
    _QUANTITIES = (("T1", "T₁"), ("T2", "T₂"), ("T1rho", "T₁ρ"),
                   ("linewidth", "Linewidth (Hendrickson–Bray)"))

    def __init__(self, canvas=None, show_view: Callable[[], None] | None = None,
                 curve_fit_rows: Callable[[], list] | None = None, parent=None):
        super().__init__(parent)
        self.canvas = canvas
        self.show_view = show_view or (lambda: None)
        self.curve_fit_rows = curve_fit_rows or (lambda: [])
        self.fit = None
        self.hb_fit = None
        self.arrhenius = None
        self.comparison = None
        self._build()
        self._sync()
        if canvas is not None and hasattr(canvas, "register_plot_options"):
            canvas.register_plot_options(
                "vt", "Relaxation vs Temperature Plot…", self.plot_options_dialog.open_options,
                "Which axes (ln T or ln R against 1000/T, 1/k_BT or T), the temperature axis "
                "on top, and each process shown.")

    # ---------------------------------------------------------------- UI
    def _build(self) -> None:
        layout = QVBoxLayout(self)
        blurb = QLabel(
            "Relaxation times (or linewidths) against temperature: plot ln(T₁) against "
            "1000/T with the temperature on top, and fit a BPP model for the activation "
            "energy and correlation time. Several Larmor frequencies are fitted jointly "
            "(same motion, shared τ₀ and Ea).")
        blurb.setWordWrap(True)
        layout.addWidget(blurb)

        form = QFormLayout()
        self.quantity_combo = QComboBox()
        for key, text in self._QUANTITIES:
            self.quantity_combo.addItem(text, key)
        self.quantity_combo.setToolTip(
            "What the Value column holds. Linewidths are fitted with the "
            "Hendrickson–Bray motional-narrowing equation rather than BPP, which "
            "does not describe the rigid-lattice plateau.")
        self.quantity_combo.currentIndexChanged.connect(lambda _i: self._sync())
        form.addRow("Quantity:", self.quantity_combo)

        self.spinlock_spin = QDoubleSpinBox()
        self.spinlock_spin.setRange(0.1, 1000.0)
        self.spinlock_spin.setValue(20.0)
        self.spinlock_spin.setSuffix(" kHz")
        self.spinlock_spin.setToolTip("The spin-lock nutation frequency ν₁ of the T₁ρ experiment.")
        self.spinlock_label = QLabel("Spin-lock ν₁:")
        form.addRow(self.spinlock_label, self.spinlock_spin)

        units = QHBoxLayout()
        self.value_unit_combo = QComboBox()
        self.value_unit_combo.currentIndexChanged.connect(lambda _i: self._sync_headers())
        self.temperature_unit_combo = QComboBox()
        self.temperature_unit_combo.addItem("K", "K")
        self.temperature_unit_combo.addItem("°C", "C")
        self.temperature_unit_combo.currentIndexChanged.connect(lambda _i: self._sync_headers())
        self.rate_check = QCheckBox("Values are rates")
        self.rate_check.setToolTip("The Value column holds R = 1/T (s⁻¹) rather than times.")
        self.rate_check.toggled.connect(lambda _c: self._sync_headers())
        units.addWidget(QLabel("Value in"))
        units.addWidget(self.value_unit_combo)
        units.addWidget(QLabel("T in"))
        units.addWidget(self.temperature_unit_combo)
        units.addWidget(self.rate_check)
        units.addStretch(1)
        form.addRow("Units:", units)
        layout.addLayout(form)

        self.table = QTableWidget(0, len(self.COLUMNS))
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setMinimumHeight(150)
        self.table.setToolTip(
            "One row per measurement. Rows with the same Series name and Larmor frequency "
            "form one dataset. ± is the 1σ uncertainty of the value (optional; with it "
            "the fit is weighted). Paste a block copied from a spreadsheet with Paste.")
        layout.addWidget(self.table)

        buttons = QHBoxLayout()
        for text, slot, tip in (
                ("Add row", self.add_row, "Add an empty row."),
                ("Remove", self.remove_selected_rows, "Remove the selected rows."),
                ("Paste", self.paste_from_clipboard,
                 "Append rows from the clipboard: columns T, value[, ±[, Larmor MHz[, series]]], "
                 "tab- or comma-separated."),
                ("Import CSV…", self.import_csv,
                 "Columns T, value and optionally error, larmor_MHz, label (header row needed)."),
                ("From Fit Curve", self.add_from_curve_fit,
                 "Append each peak's fitted T₁/T₂ from the current Dynamics ▸ Fit Curve result, "
                 "with the temperature (TE) and Larmor frequency of its spectrum."),
                ("Clear", self.clear_rows, "Remove every row.")):
            b = QPushButton(text)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            buttons.addWidget(b)
        layout.addLayout(buttons)

        # Model
        self.model_box = QGroupBox("BPP model")
        mform = QFormLayout(self.model_box)
        self.mechanism_combo = QComboBox()
        self.mechanism_combo.addItem("Dipolar (amplitude C)", "dipolar")
        self.mechanism_combo.addItem("Quadrupolar (amplitude C_Q)", "quadrupolar")
        self.mechanism_combo.setToolTip(
            "Dipolar: the amplitude C (s⁻²) is free. Quadrupolar: the amplitude is "
            "(3π²/50)(2I+3)/(I²(2I−1))(1+η²/3)·C_Q², so the fit returns C_Q in MHz.")
        self.mechanism_combo.currentIndexChanged.connect(lambda _i: self._sync())
        mform.addRow("Mechanism:", self.mechanism_combo)
        self.quad_row = QWidget()
        quad = QHBoxLayout(self.quad_row)
        quad.setContentsMargins(0, 0, 0, 0)
        self.spin_spin = QDoubleSpinBox()
        self.spin_spin.setRange(1.0, 9.0)
        self.spin_spin.setSingleStep(0.5)
        self.spin_spin.setValue(1.5)
        self.eta_spin = QDoubleSpinBox()
        self.eta_spin.setRange(0.0, 1.0)
        self.eta_spin.setSingleStep(0.1)
        quad.addWidget(QLabel("I ="))
        quad.addWidget(self.spin_spin)
        quad.addWidget(QLabel("η_Q ="))
        quad.addWidget(self.eta_spin)
        quad.addStretch(1)
        self.quad_label = QLabel("Spin, asymmetry:")
        mform.addRow(self.quad_label, self.quad_row)
        self.processes_spin = QSpinBox()
        self.processes_spin.setRange(1, 3)
        self.processes_spin.setToolTip(
            "Independent motions, each with its own τ₀ and Ea; their rates add. Use "
            "two when ln(T₁) against 1000/T shows two minima or a shoulder.")
        mform.addRow("Processes:", self.processes_spin)
        opts = QHBoxLayout()
        self.asymmetric_check = QCheckBox("Asymmetric (free β)")
        self.asymmetric_check.setToolTip(
            "J(ω) = τ/(1+(ωτ)^β), 1 < β ≤ 2: the low-temperature flank has slope (β−1)Ea, "
            "the high-temperature flank Ea -- the usual signature of correlated hopping "
            "in ion conductors. β = 2 is BPP.")
        self.background_check = QCheckBox("Background rate")
        self.background_check.setToolTip(
            "A temperature-independent rate R_bg added to the total (paramagnetic "
            "impurities, a spin-diffusion floor) -- it flattens the low-temperature end.")
        self.per_dataset_check = QCheckBox("Amplitude per dataset")
        self.per_dataset_check.setToolTip(
            "In a joint fit of several fields, give each dataset its own amplitude "
            "(τ₀, Ea, β stay shared). Off: one amplitude, the stricter test of BPP.")
        for w in (self.asymmetric_check, self.background_check, self.per_dataset_check):
            opts.addWidget(w)
        opts.addStretch(1)
        mform.addRow(opts)
        self.law_combo = QComboBox()
        self.law_combo.addItem("Arrhenius  τ = τ₀ exp(Ea/k_BT)", "arrhenius")
        self.law_combo.addItem("VFT  τ = τ₀ exp(B/k_B(T − T₀))", "vft")
        self.law_combo.setToolTip("VFT for polymers and glass-formers; its 'Ea' is B, in eV.")
        mform.addRow("Correlation time:", self.law_combo)
        self.fixed_edit = QLineEdit()
        self.fixed_edit.setPlaceholderText("e.g. Ea_eV=0.3, beta=1.8")
        self.fixed_edit.setToolTip("Parameters to hold fixed, name=value. Names are listed "
                                   "by 'Parameter names'.")
        names_button = QPushButton("Parameter names")
        names_button.clicked.connect(self.show_parameter_names)
        fixed_row = QHBoxLayout()
        fixed_row.addWidget(self.fixed_edit)
        fixed_row.addWidget(names_button)
        mform.addRow("Fixed:", fixed_row)
        search = QHBoxLayout()
        self.starts_spin = QSpinBox()
        self.starts_spin.setRange(1, 256)
        self.starts_spin.setValue(24)
        self.starts_spin.setToolTip("Deterministic starting points: BPP surfaces have several "
                                    "minima, especially with more than one process.")
        self.bootstrap_spin = QSpinBox()
        self.bootstrap_spin.setRange(0, 2000)
        self.bootstrap_spin.setSingleStep(100)
        self.bootstrap_spin.setToolTip("Residual-bootstrap refits for 16–84 % intervals that do "
                                       "not assume a linear model (0 = off).")
        search.addWidget(QLabel("Starts"))
        search.addWidget(self.starts_spin)
        search.addWidget(QLabel("Bootstrap"))
        search.addWidget(self.bootstrap_spin)
        search.addStretch(1)
        mform.addRow("Search:", search)
        layout.addWidget(self.model_box)

        trange = QHBoxLayout()
        self.range_check = QCheckBox("Only")
        self.range_check.setToolTip("Fit (and draw an Arrhenius line through) only this "
                                    "temperature range, in kelvin.")
        self.t_lo_spin = QDoubleSpinBox()
        self.t_hi_spin = QDoubleSpinBox()
        for s, v in ((self.t_lo_spin, 200.0), (self.t_hi_spin, 400.0)):
            s.setRange(0.0, 5000.0)
            s.setDecimals(1)
            s.setValue(v)
            s.setSuffix(" K")
        trange.addWidget(self.range_check)
        trange.addWidget(self.t_lo_spin)
        trange.addWidget(QLabel("to"))
        trange.addWidget(self.t_hi_spin)
        trange.addStretch(1)
        layout.addLayout(trange)

        # Plot
        plot_box = QGroupBox("Plot")
        pform = QFormLayout(plot_box)
        self.y_axis_combo = QComboBox()
        for key, text in (("ln_time", "ln(T)"), ("ln_rate", "ln(R = 1/T)"),
                          ("log10_time", "log₁₀(T)"), ("log10_rate", "log₁₀(R)"),
                          ("time", "T (log scale)"), ("rate", "R (log scale)")):
            self.y_axis_combo.addItem(text, key)
        self.x_axis_combo = QComboBox()
        for key, text in (("1000/T", "1000/T"), ("1/kBT", "1/k_BT (eV⁻¹)"), ("T", "T")):
            self.x_axis_combo.addItem(text, key)
        self.top_axis_combo = QComboBox()
        for key, text in (("K", "Temperature (K)"), ("C", "Temperature (°C)"), ("none", "None")):
            self.top_axis_combo.addItem(text, key)
        self.components_check = QCheckBox("Show each process")
        self.components_check.setChecked(True)
        for w in (self.y_axis_combo, self.x_axis_combo, self.top_axis_combo):
            w.currentIndexChanged.connect(lambda _i: self.redraw())
        self.components_check.toggled.connect(lambda _c: self.redraw())
        pform.addRow("y axis:", self.y_axis_combo)
        pform.addRow("x axis:", self.x_axis_combo)
        pform.addRow("Top axis:", self.top_axis_combo)
        pform.addRow(self.components_check)
        # On the plot's right-click menu, not in the panel (explicit user
        # request, 2026-10-16).
        self.plot_box = plot_box
        self.plot_options_dialog = PlotOptionsDialog(
            "Relaxation vs Temperature: plot", plot_box, self)
        hint = QLabel("Plot axes and processes: right-click the plot ▸ "
                      "Relaxation vs Temperature Plot…")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: palette(mid);")
        layout.addWidget(hint)

        actions = QHBoxLayout()
        self.plot_button = QPushButton("Plot Data")
        self.plot_button.clicked.connect(self.plot_data)
        self.fit_button = QPushButton("Fit")
        self.fit_button.setToolTip("Fit the model above to every dataset at once.")
        self.fit_button.clicked.connect(self.run_fit)
        self.compare_button = QPushButton("Compare Models")
        self.compare_button.setToolTip(
            "Fit BPP, asymmetric BPP, BPP + background and two processes, and rank them by "
            "AICc (lower is better; a difference under ~2 is not decisive).")
        self.compare_button.clicked.connect(self.run_comparison)
        self.arrhenius_button = QPushButton("Arrhenius Line")
        self.arrhenius_button.setToolTip(
            "A straight line through ln(rate) against 1/k_BT for the first dataset, over the "
            "temperature range above when 'Only' is ticked: Ea from one flank alone.")
        self.arrhenius_button.clicked.connect(self.run_arrhenius)
        for b in (self.plot_button, self.fit_button, self.compare_button, self.arrhenius_button):
            actions.addWidget(b)
        layout.addLayout(actions)

        self.results_label = ReportLabel("")
        self.results_label.setWordWrap(True)
        self.results_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.results_label)
        self.results_table = _read_only_table(("Parameter", "Value", "± 1σ", "Note"))
        layout.addWidget(self.results_table)

        more = QHBoxLayout()
        self.code_button = QPushButton("Copy notebook code")
        self.code_button.clicked.connect(self.copy_notebook_code)
        self.export_button = QPushButton("Export Results…")
        self.export_button.clicked.connect(self.export_results)
        more.addWidget(self.code_button)
        more.addWidget(self.export_button)
        more.addStretch(1)
        layout.addLayout(more)
        self.status_log = _log_box()
        layout.addWidget(self.status_log)

    # -------------------------------------------------------------- state
    @property
    def quantity(self) -> str:
        return self.quantity_combo.currentData()

    def _sync(self) -> None:
        q = self.quantity
        lw = q == "linewidth"
        self.spinlock_label.setVisible(q == "T1rho")
        self.spinlock_spin.setVisible(q == "T1rho")
        self.model_box.setVisible(not lw)
        self.compare_button.setVisible(not lw)
        self.rate_check.setVisible(not lw)
        for w in (self.y_axis_combo, self.x_axis_combo, self.top_axis_combo,
                  self.components_check):
            w.setEnabled(not lw)
        quad = self.mechanism_combo.currentData() == "quadrupolar"
        self.quad_label.setVisible(quad)
        self.quad_row.setVisible(quad)
        current = self.value_unit_combo.currentData()
        self.value_unit_combo.blockSignals(True)
        self.value_unit_combo.clear()
        units = (("Hz", "Hz"), ("kHz", "kHz")) if lw else (("s", "s"), ("ms", "ms"))
        for key, text in units:
            self.value_unit_combo.addItem(text, key)
        idx = self.value_unit_combo.findData(current)
        self.value_unit_combo.setCurrentIndex(max(idx, 0))
        self.value_unit_combo.blockSignals(False)
        self._sync_headers()

    def _sync_headers(self) -> None:
        q = self.quantity
        unit = self.value_unit_combo.currentData() or ""
        t_unit = "°C" if self.temperature_unit_combo.currentData() == "C" else "K"
        if q == "linewidth":
            vname = f"FWHM ({unit})"
        elif self.rate_check.isChecked():
            vname = {"T1": "R₁", "T2": "R₂", "T1rho": "R₁ρ"}[q] + f" (1/{unit})"
        else:
            vname = {"T1": "T₁", "T2": "T₂", "T1rho": "T₁ρ"}[q] + f" ({unit})"
        self.table.setHorizontalHeaderLabels(["Series", f"T ({t_unit})", vname, "±",
                                              "Larmor (MHz)"])

    def add_row(self, values=None) -> None:
        r = self.table.rowCount()
        self.table.insertRow(r)
        values = list(values or [])
        for c in range(len(self.COLUMNS)):
            text = "" if c >= len(values) or values[c] is None else str(values[c])
            self.table.setItem(r, c, QTableWidgetItem(text))

    def remove_selected_rows(self) -> None:
        rows = sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True)
        for r in rows:
            self.table.removeRow(r)

    def clear_rows(self) -> None:
        self.table.setRowCount(0)

    def set_rows(self, rows) -> None:
        """Replace the table with ``rows`` of (series, T, value, error,
        larmor) -- used by tests and by Import."""
        self.clear_rows()
        for row in rows:
            self.add_row(row)

    def paste_from_clipboard(self, text: str | None = None) -> int:
        text = QGuiApplication.clipboard().text() if text is None else text
        n = 0
        for line in (text or "").splitlines():
            if not line.strip():
                continue
            parts = [p.strip() for p in re.split(r"\t|,|;", line)]
            nums = [_float(p) for p in parts]
            if len(parts) < 2 or nums[0] is None or nums[1] is None:
                continue  # a header or a stray line
            T, v = parts[0], parts[1]
            e = parts[2] if len(parts) > 2 else ""
            f = parts[3] if len(parts) > 3 else ""
            s = parts[4] if len(parts) > 4 else "data"
            self.add_row([s, T, v, e, f])
            n += 1
        self._log(f"Pasted {n} row(s).")
        return n

    def import_csv(self, path: str | None = None) -> int:
        if path is None:
            path, _ = QFileDialog.getOpenFileName(self, "Import temperature series", "",
                                                  "CSV (*.csv *.txt);;All files (*)")
            if not path:
                return 0
        try:
            datasets = vt.load_vt_csv(path, quantity=self.quantity if self.quantity else "T1",
                                      value_unit=self.value_unit_combo.currentData() or "s")
        except Exception as exc:  # noqa: BLE001 -- shown to the user
            QMessageBox.warning(self, "Import failed", str(exc))
            return 0
        n = 0
        unit_scale = {"s": 1.0, "ms": 1e3, "Hz": 1.0, "kHz": 1e-3}[self.value_unit_combo.currentData()]
        for d in datasets:
            for T, v, e in zip(d["T_K"], d["value"], d["error"]):
                Tshow = T - 273.15 if self.temperature_unit_combo.currentData() == "C" else T
                self.add_row([d["label"], f"{Tshow:g}", f"{v * unit_scale:.6g}",
                              "" if not np.isfinite(e) else f"{e * unit_scale:.3g}",
                              "" if d["larmor_MHz"] is None else f"{d['larmor_MHz']:g}"])
                n += 1
        self._log(f"Imported {n} row(s) from {path}.")
        return n

    def add_from_curve_fit(self) -> int:
        rows = self.curve_fit_rows() or []
        if not rows:
            QMessageBox.information(
                self, "No Fit Curve result",
                "Fit a relaxation curve first (Dynamics ▸ Fit Curve); each peak's T₁/T₂ "
                "is then added here with its spectrum's temperature.")
            return 0
        return self.add_rows(rows, source="the Fit Curve result")

    def add_rows(self, rows, source="the batch fit") -> int:
        """Rows ``{"series", "T_K", "value", "error", "larmor_MHz",
        "quantity"}`` into the table (values in s); the table groups them
        by (series, Larmor) when fitting -- one dataset per peak per field."""
        rows = list(rows or [])
        if not rows:
            return 0
        q = rows[0].get("quantity")
        if q in ("T1", "T2") and q != self.quantity:
            self.quantity_combo.setCurrentIndex(self.quantity_combo.findData(q))
        scale = {"s": 1.0, "ms": 1e3}.get(self.value_unit_combo.currentData(), 1.0)
        missing = 0
        for r in rows:
            T = r.get("T_K")
            if T is not None and self.temperature_unit_combo.currentData() == "C":
                T = T - 273.15
            if T is None:
                missing += 1
            self.add_row([r.get("series", "data"), "" if T is None else f"{T:g}",
                          f"{r['value'] * scale:.6g}",
                          "" if r.get("error") is None or not np.isfinite(r["error"])
                          else f"{r['error'] * scale:.3g}",
                          "" if not r.get("larmor_MHz") else f"{r['larmor_MHz']:.6g}"])
        self._log(f"Added {len(rows)} value(s) from {source}"
                  + (f"; {missing} without a recorded temperature -- type it in." if missing else "."))
        return len(rows)

    def datasets(self) -> list:
        """The table as :func:`vt_relaxation.vt_dataset` dicts, grouped by
        (series, Larmor). Raises ValueError naming a bad row."""
        groups: dict = {}
        q = self.quantity
        for r in range(self.table.rowCount()):
            cells = [(self.table.item(r, c).text() if self.table.item(r, c) else "").strip()
                     for c in range(len(self.COLUMNS))]
            if not any(cells):
                continue
            T, v = _float(cells[1]), _float(cells[2])
            if T is None or v is None:
                raise ValueError(f"Row {r + 1}: needs a temperature and a value.")
            e = _float(cells[3])
            f = _float(cells[4])
            key = (cells[0] or "data", f)
            g = groups.setdefault(key, {"T": [], "v": [], "e": []})
            g["T"].append(T)
            g["v"].append(v)
            g["e"].append(np.nan if e is None else e)
        if not groups:
            raise ValueError("The table is empty.")
        out = []
        names = [series for series, _f in groups]
        for (series, f), g in groups.items():
            if q != "linewidth" and f is None:
                raise ValueError(f"Series '{series}' has no Larmor frequency; BPP needs it.")
            errs = g["e"] if not all(np.isnan(g["e"])) else None
            if series == "data" and f is not None:
                label = f"{f:g} MHz"
            elif names.count(series) > 1 and f is not None:
                label = f"{series}, {f:g} MHz"     # one series at several fields
            else:
                label = series
            out.append(vt.vt_dataset(
                g["T"], g["v"], errs, larmor_MHz=f, quantity=q, label=label,
                spinlock_kHz=self.spinlock_spin.value() if q == "T1rho" else None,
                value_is_rate=self.rate_check.isChecked() and q != "linewidth",
                temperature_unit=self.temperature_unit_combo.currentData(),
                value_unit=self.value_unit_combo.currentData()))
        return out

    def fit_kwargs(self) -> dict:
        kw = {"mechanism": self.mechanism_combo.currentData(),
              "n_processes": self.processes_spin.value(),
              "asymmetric": self.asymmetric_check.isChecked(),
              "background": self.background_check.isChecked(),
              "law": self.law_combo.currentData(),
              "per_dataset_amplitude": self.per_dataset_check.isChecked(),
              "n_starts": self.starts_spin.value()}
        if kw["mechanism"] == "quadrupolar":
            kw["spin"] = self.spin_spin.value()
            kw["eta"] = self.eta_spin.value()
        fixed = parse_fixed(self.fixed_edit.text())
        if fixed:
            kw["fixed"] = fixed
        if self.bootstrap_spin.value():
            kw["n_bootstrap"] = self.bootstrap_spin.value()
        if self.range_check.isChecked():
            kw["T_range"] = (self.t_lo_spin.value(), self.t_hi_spin.value())
        return kw

    def plot_kwargs(self) -> dict:
        return {"y_axis": self.y_axis_combo.currentData(),
                "text_size": _text_size(self.canvas),
                "x_axis": self.x_axis_combo.currentData(),
                "top_axis": None if self.top_axis_combo.currentData() == "none"
                else self.top_axis_combo.currentData(),
                "show_components": self.components_check.isChecked()}

    # ------------------------------------------------------------ actions
    def _datasets_or_warn(self):
        try:
            return self.datasets()
        except ValueError as exc:
            QMessageBox.warning(self, "Check the table", str(exc))
            return None

    def show_parameter_names(self) -> None:
        try:
            kw = self.fit_kwargs()
        except ValueError:
            kw = {}
        try:
            n = len(self.datasets()) if self.table.rowCount() else 1
        except ValueError:
            n = 1
        try:
            model = vt.VTModel(kw.get("mechanism", "dipolar"), kw.get("n_processes", 1),
                               kw.get("asymmetric", False), kw.get("background", False),
                               kw.get("law", "arrhenius"), kw.get("spin"), kw.get("eta", 0.0),
                               kw.get("per_dataset_amplitude", False))
            text = ", ".join(model.param_names(n))
        except ValueError as exc:
            text = str(exc)
        QMessageBox.information(self, "Parameter names", text)

    def _draw(self, draw) -> None:
        if self.canvas is None:
            return
        _show(self.canvas, draw, "curve")
        self.canvas.plot_owner = "vt"
        self.show_view()

    def plot_data(self) -> None:
        data = self._datasets_or_warn()
        if data is None:
            return
        self.fit = self.hb_fit = None
        if self.quantity == "linewidth":
            def draw(fig):
                vt.plot_vs_temperature(data, ax=fig.add_subplot(111), log_y=False,
                                       temperature_unit=self.temperature_unit_combo.currentData())
        else:
            pk = self.plot_kwargs()

            def draw(fig):
                vt.plot_arrhenius(datasets=data, ax=fig.add_subplot(111), **pk)
        self._last_draw = ("data", data)
        self._draw(draw)
        self._log(f"Plotted {sum(d['T_K'].size for d in data)} point(s) in {len(data)} dataset(s).")

    def redraw(self) -> None:
        if self.fit is not None:
            pk = self.plot_kwargs()
            fit = self.fit
            self._draw(lambda fig: vt.plot_arrhenius(fit, ax=fig.add_subplot(111), **pk))
        elif getattr(self, "_last_draw", None) and self._last_draw[0] == "data":
            if self.quantity != "linewidth":
                pk = self.plot_kwargs()
                data = self._last_draw[1]
                self._draw(lambda fig: vt.plot_arrhenius(datasets=data, ax=fig.add_subplot(111),
                                                         **pk))

    def run_fit(self) -> dict | None:
        data = self._datasets_or_warn()
        if data is None:
            return None
        if self.quantity == "linewidth":
            return self._run_hb(data)
        try:
            kw = self.fit_kwargs()
            fit = vt.fit_vt_relaxation(data, **kw)
        except (ValueError, RuntimeError) as exc:
            QMessageBox.warning(self, "Fit failed", str(exc))
            return None
        self.fit, self.hb_fit = fit, None
        self._fill_results(fit)
        self.results_label.setText(
            f"{fit['model_text']}: reduced χ² = {fit['redchi']:.3g}, AICc = {fit['aicc']:.2f}"
            + (f"; {len(fit['rivals'])} near-equivalent solution(s), see the log."
               if fit["rivals"] else "."))
        self._log(fit["log"])
        self.redraw()
        return fit

    def _run_hb(self, data):
        if len(data) > 1:
            QMessageBox.information(self, "One series at a time",
                                    "Hendrickson–Bray is fitted to one series; using the first.")
        d = data[0]
        try:
            fixed = parse_fixed(self.fixed_edit.text())
            fixed = {k: v for k, v in fixed.items() if k in ("A", "B", "Ea_eV", "D")}
            hb = vt.fit_hendrickson_bray(d["T_K"], d["value"],
                                         None if np.all(np.isnan(d["error"])) else d["error"],
                                         fixed=fixed or None)
        except (ValueError, RuntimeError) as exc:
            QMessageBox.warning(self, "Fit failed", str(exc))
            return None
        self.hb_fit, self.fit = hb, None
        unit = self.value_unit_combo.currentData()
        tunit = self.temperature_unit_combo.currentData()
        size = _text_size(self.canvas)
        self._draw(lambda fig: vt.plot_hendrickson_bray(hb, ax=fig.add_subplot(111), unit=unit,
                                                        temperature_unit=tunit, text_size=size))
        self.results_table.setRowCount(0)
        for name in ("A", "B", "Ea_eV", "D"):
            self._result_row(name, hb["params"][name], hb["errors"][name],
                             "fixed" if name in hb["fixed"] else "")
        self._result_row("T_onset_K", hb["T_onset_K"], None, "10 % narrowed")
        self._result_row("waugh_fedin_Ea_eV", hb["waugh_fedin_Ea_eV"], None,
                         "rule of thumb, for comparison")
        self.results_label.setText(f"Hendrickson–Bray: Ea = {hb['params']['Ea_eV']:.3f} eV, "
                                   f"reduced χ² = {hb['redchi']:.3g}.")
        self._log(hb["log"])
        return hb

    def run_comparison(self) -> list | None:
        data = self._datasets_or_warn()
        if data is None:
            return None
        try:
            kw = self.fit_kwargs()
        except ValueError as exc:
            QMessageBox.warning(self, "Check the settings", str(exc))
            return None
        base = {k: kw[k] for k in ("mechanism", "spin", "eta", "law", "per_dataset_amplitude",
                                   "n_starts", "T_range") if k in kw}
        candidates = [{"n_processes": 1}, {"n_processes": 1, "asymmetric": True},
                      {"n_processes": 1, "background": True}, {"n_processes": 2}]
        fits = vt.compare_vt_models(data, candidates, **base)
        self.comparison = fits
        self.results_table.setRowCount(0)
        lines = ["Model comparison (AICc; Akaike weight = relative support):"]
        for f in fits:
            if "error" in f:
                self._result_row(f["model_text"], float("nan"), None, f["error"])
                lines.append(f"  {f['model_text']}: not fitted ({f['error']})")
                continue
            Ea = ", ".join(f"{v:.3f}" for n, v in f["params"].items() if n.startswith("Ea"))
            self._result_row(f["model_text"], f["delta_aicc"], None,
                             f"weight {f['weight']:.2f}; Ea = {Ea} eV")
            lines.append(f"  ΔAICc = {f['delta_aicc']:6.2f}  weight {f['weight']:.2f}  "
                         f"{f['model_text']}  (Ea = {Ea} eV, reduced χ² {f['redchi']:.3g})")
        self.results_label.setText("ΔAICc per model (0 = best). Differences under ~2 do not "
                                   "distinguish the models; above ~10 they do.")
        self._log("\n".join(lines))
        best = next((f for f in fits if "error" not in f), None)
        if best is not None:
            self.fit = best
            self.redraw()
        return fits

    def run_arrhenius(self) -> dict | None:
        data = self._datasets_or_warn()
        if data is None:
            return None
        d = data[0]
        T_range = (self.t_lo_spin.value(), self.t_hi_spin.value()) if self.range_check.isChecked() \
            else None
        try:
            res = vt.fit_arrhenius(d["T_K"], d["value"],
                                   None if np.all(np.isnan(d["error"])) else d["error"],
                                   is_rate=d["quantity"] == "linewidth", T_range=T_range)
        except ValueError as exc:
            QMessageBox.warning(self, "Arrhenius line", str(exc))
            return None
        self.arrhenius = res
        self._log(f"{d['label']}: " + res["log"])
        self.results_table.setRowCount(0)
        self._result_row("Ea_eV", res["Ea_eV"], res["Ea_err_eV"], res["flank"] + " flank")
        self._result_row("ln_A", res["ln_A"], None, "")
        self.results_label.setText(f"Arrhenius line: Ea = {res['Ea_eV']:.3f} eV "
                                   f"({res['flank']} flank).")
        if d["quantity"] != "linewidth":
            pk = self.plot_kwargs()

            def draw(fig):
                ax = fig.add_subplot(111)
                vt.plot_arrhenius(datasets=data, ax=ax, **pk)
                fx, _i, _l = vt._x_transform(pk["x_axis"])
                fy, _yl, _log = vt._y_transform(pk["y_axis"], d["quantity"])
                Tl = np.linspace(res["T_K"].min(), res["T_K"].max(), 50)
                rate = np.exp(res["ln_A"] + res["slope"] / (vt.K_B_EV * Tl))
                ax.plot(fx(Tl), fy(1.0 / rate), "-", color="0.25", lw=1.0,
                        label=f"Arrhenius, Ea = {res['Ea_eV']:.3f} eV")
            self._draw(draw)
        return res

    def _result_row(self, name, value, error, note) -> None:
        r = self.results_table.rowCount()
        self.results_table.insertRow(r)
        val = "" if value is None or not np.isfinite(value) else f"{value:.5g}"
        err = "" if error is None or not np.isfinite(error) else f"{error:.2g}"
        for c, text in enumerate((name, val, err, note)):
            self.results_table.setItem(r, c, QTableWidgetItem(text))

    def _fill_results(self, fit) -> None:
        self.results_table.setRowCount(0)
        for name, v in fit["params"].items():
            note = "fixed" if name in fit["fixed"] else ""
            if name in fit["ci"]:
                lo, hi = fit["ci"][name]
                note = (note + "; " if note else "") + f"bootstrap {lo:.4g}–{hi:.4g}"
            if name.startswith("Ea_eV"):
                note = (note + "; " if note else "") + f"{v * 96.485:.1f} kJ/mol"
            self._result_row(name, v, fit["errors"].get(name), note)
        for d, row in zip(fit["datasets"], fit["T_max"]):
            for k, t in enumerate(row):
                if np.isfinite(t):
                    self._result_row(f"T at R max ({d['label']})" + (f", process {k + 1}"
                                                                    if len(row) > 1 else ""),
                                     t, None, "K")

    def notebook_code(self) -> str:
        data = self.datasets()
        if self.quantity == "linewidth":
            return vt.vt_notebook_code(data, {"fixed": parse_fixed(self.fixed_edit.text())},
                                       hendrickson=True)
        plot_kwargs = self.plot_kwargs()
        plot_kwargs.pop("text_size", None)      # the GUI's Plot Style, not a choice
        return vt.vt_notebook_code(data, self.fit_kwargs(), plot_kwargs)

    def copy_notebook_code(self) -> None:
        try:
            code = self.notebook_code()
        except ValueError as exc:
            QMessageBox.warning(self, "Copy notebook code", str(exc))
            return
        _copy(code)
        self._log("Copied notebook code to the clipboard.")

    def results_csv(self) -> str:
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["parameter", "value", "error", "note"])
        for r in range(self.results_table.rowCount()):
            w.writerow([self.results_table.item(r, c).text() if self.results_table.item(r, c)
                        else "" for c in range(4)])
        return buf.getvalue()

    def export_results(self, path: str | None = None) -> None:
        if path is None:
            path, _ = QFileDialog.getSaveFileName(self, "Export results", "vt_fit.csv",
                                                  "CSV (*.csv)")
            if not path:
                return
        with open(path, "w", newline="", encoding="utf-8") as fh:
            fh.write(self.results_csv())
            log = self.status_log.toPlainText()
            if log:
                fh.write("\n# log\n" + "\n".join("# " + line for line in log.splitlines()) + "\n")
        self._log(f"Exported results to {path}.")

    def _log(self, text: str) -> None:
        self.status_log.appendPlainText(text)


# ======================================================================
# Exchange lineshape
# ======================================================================

class ExchangeLineshapePanel(QWidget):
    """Dynamics ▸ Exchange lineshape (ion hopping).

    ``canvas``/``show_view`` as :class:`VTRelaxationPanel`.
    ``get_spectra()`` returns the spectra selected in Open experiments
    as ``[{"label", "ppm", "y", "T_K" (or None), "larmor_MHz", "nucleus"}]``.
    ``default_nucleus()`` -> (nucleus, field in T) of the selected
    spectrum, to prefill the spectrometer row."""

    SITE_COLUMNS = ("Label", "δ (ppm)", "FWHM (ppm)", "Shift s.d. (ppm)")
    HOP_COLUMNS = ("From", "To", "Ea → (eV)", "Ea ← (eV)", "ν₀ (s⁻¹)", "Ea s.d. (eV)")

    def __init__(self, canvas=None, show_view=None, get_spectra=None, parent=None):
        super().__init__(parent)
        self.canvas = canvas
        self.show_view = show_view or (lambda: None)
        self.get_spectra = get_spectra or (lambda: [])
        self.results = None
        self.fit = None
        self._build()
        self.load_example()

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        blurb = QLabel(
            "Spectra of nuclei hopping between sites, from the Bloch–McConnell equations "
            "(exact; equilibrium populations from detailed balance). Each hop has a forward "
            "and reverse barrier and an attempt frequency, k = ν₀ exp(−Ea/k_BT); unequal "
            "barriers make the sites' energies, and so their populations, differ.")
        blurb.setWordWrap(True)
        layout.addWidget(blurb)

        form = QFormLayout()
        spec = QHBoxLayout()
        self.nucleus_edit = QLineEdit("7Li")
        self.nucleus_edit.setMaximumWidth(70)
        self.field_spin = QDoubleSpinBox()
        self.field_spin.setRange(0.1, 40.0)
        self.field_spin.setDecimals(3)
        self.field_spin.setValue(18.79)
        self.field_spin.setSuffix(" T")
        self.larmor_label = QLabel("")
        self.nucleus_edit.textChanged.connect(lambda _t: self._update_larmor())
        self.field_spin.valueChanged.connect(lambda _v: self._update_larmor())
        spec.addWidget(self.nucleus_edit)
        spec.addWidget(QLabel("at"))
        spec.addWidget(self.field_spin)
        spec.addWidget(self.larmor_label)
        spec.addStretch(1)
        form.addRow("Nucleus:", spec)
        layout.addLayout(form)

        layout.addWidget(QLabel("Sites"))
        self.sites_table = QTableWidget(0, len(self.SITE_COLUMNS))
        self.sites_table.setHorizontalHeaderLabels(list(self.SITE_COLUMNS))
        self.sites_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.sites_table.setToolTip(
            "δ: the site's shift without exchange. FWHM: its own Lorentzian width without "
            "exchange. Shift s.d.: a Gaussian distribution of the shift (disorder).")
        self.sites_table.setMaximumHeight(140)
        self.sites_table.itemChanged.connect(lambda _i: self._refresh_vary_list())
        layout.addWidget(self.sites_table)
        row = QHBoxLayout()
        for text, slot in (("Add site", lambda: self.add_site()),
                           ("Remove site", self.remove_site)):
            b = QPushButton(text)
            b.clicked.connect(slot)
            row.addWidget(b)
        row.addStretch(1)
        layout.addLayout(row)

        layout.addWidget(QLabel("Hops (each reversible; site numbers from 1)"))
        self.hops_table = QTableWidget(0, len(self.HOP_COLUMNS))
        self.hops_table.setHorizontalHeaderLabels(list(self.HOP_COLUMNS))
        self.hops_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.hops_table.setToolTip(
            "Ea →: barrier from the 'From' site to the 'To' site; Ea ←: back (blank = the "
            "same). ν₀: attempt frequency, typically 10¹²–10¹³ s⁻¹. Ea s.d.: a Gaussian "
            "distribution of both barriers together.")
        self.hops_table.setMaximumHeight(120)
        self.hops_table.itemChanged.connect(lambda _i: self._refresh_vary_list())
        layout.addWidget(self.hops_table)
        row = QHBoxLayout()
        for text, slot in (("Add hop", lambda: self.add_hop()), ("Remove hop", self.remove_hop),
                           ("Example", self.load_example)):
            b = QPushButton(text)
            b.clicked.connect(slot)
            row.addWidget(b)
        row.addStretch(1)
        layout.addLayout(row)

        sim = QFormLayout()
        self.temperatures_edit = QLineEdit("200:350:25")
        self.temperatures_edit.setToolTip("Kelvin: a list (200, 250, 300) or start:stop:step.")
        sim.addRow("Temperatures (K):", self.temperatures_edit)
        self.gauss_spin = QDoubleSpinBox()
        self.gauss_spin.setRange(0.0, 1000.0)
        self.gauss_spin.setDecimals(3)
        self.gauss_spin.setValue(1.0)
        self.gauss_spin.setSuffix(" ppm")
        self.gauss_spin.setToolTip("Gaussian broadening of the whole spectrum (FWHM), applied "
                                   "exactly (Voigt lines).")
        sim.addRow("Gaussian broadening:", self.gauss_spin)
        self.nodes_spin = QSpinBox()
        self.nodes_spin.setRange(1, 15)
        self.nodes_spin.setValue(7)
        self.nodes_spin.setToolTip("Minimum grid points per distributed shift or barrier (over ±4 s.d.); "
                                   "a shift distribution gets as many more as its line needs to be smooth.")
        sim.addRow("Distribution points:", self.nodes_spin)
        layout.addLayout(sim)
        # The plot's own option, on its right-click menu (2026-10-16).
        self.fast_limit_check = QCheckBox("Mark the fully narrowed position")
        self.fast_limit_check.setToolTip("A line at the shift every site's line merges to "
                                         "in the fast-exchange limit, on the simulated stack.")
        self.fast_limit_check.toggled.connect(lambda _on: self._redraw_stack())
        options = QWidget()
        QVBoxLayout(options).addWidget(self.fast_limit_check)
        self.plot_options_dialog = PlotOptionsDialog("Exchange Lineshape: plot", options, self)
        if self.canvas is not None and hasattr(self.canvas, "register_plot_options"):
            self.canvas.register_plot_options(
                "exchange_stack", "Exchange Lineshape Plot…",
                self.plot_options_dialog.open_options,
                "Mark the fully narrowed position on the simulated stack.")

        actions = QHBoxLayout()
        self.simulate_button = QPushButton("Simulate")
        self.simulate_button.clicked.connect(self.simulate)
        self.code_button = QPushButton("Copy notebook code")
        self.code_button.clicked.connect(self.copy_notebook_code)
        actions.addWidget(self.simulate_button)
        actions.addWidget(self.code_button)
        actions.addStretch(1)
        layout.addLayout(actions)

        fit_box = QGroupBox("Fit to the spectra selected in Open experiments")
        fl = QVBoxLayout(fit_box)
        note = QLabel("Tick what to fit; everything else is held at the tables' values. "
                      "Each spectrum's scale and baseline are solved exactly. Ea and ν₀ of a "
                      "hop can only be separated by spectra at several temperatures (their TE "
                      "is used, or the Temperatures list in order).")
        note.setWordWrap(True)
        fl.addWidget(note)
        self.vary_list = QListWidget()
        self.vary_list.setMaximumHeight(130)
        fl.addWidget(self.vary_list)
        win = QHBoxLayout()
        self.window_check = QCheckBox("Window")
        self.win_lo = QDoubleSpinBox()
        self.win_hi = QDoubleSpinBox()
        for s, v in ((self.win_lo, -50.0), (self.win_hi, 80.0)):
            s.setRange(-1e5, 1e5)
            s.setValue(v)
            s.setSuffix(" ppm")
        win.addWidget(self.window_check)
        win.addWidget(self.win_lo)
        win.addWidget(QLabel("to"))
        win.addWidget(self.win_hi)
        win.addStretch(1)
        fl.addLayout(win)
        self.fit_button = QPushButton("Fit Selected Spectra")
        self.fit_button.clicked.connect(self.fit_selected)
        fl.addWidget(self.fit_button)
        layout.addWidget(fit_box)

        self.results_label = ReportLabel("")
        self.results_label.setWordWrap(True)
        self.results_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.results_label)
        self.status_log = _log_box()
        layout.addWidget(self.status_log)
        self._update_larmor()

    # ------------------------------------------------------------ tables
    def _set_row(self, table, r, values) -> None:
        table.blockSignals(True)
        for c, v in enumerate(values):
            table.setItem(r, c, QTableWidgetItem("" if v is None else str(v)))
        table.blockSignals(False)

    def add_site(self, values=None) -> None:
        r = self.sites_table.rowCount()
        self.sites_table.insertRow(r)
        self._set_row(self.sites_table, r, values or [f"S{r + 1}", 0, 0.5, 0])
        self._refresh_vary_list()

    def remove_site(self) -> None:
        r = self.sites_table.currentRow()
        if r < 0:
            r = self.sites_table.rowCount() - 1
        if r >= 0:
            self.sites_table.removeRow(r)
        self._refresh_vary_list()

    def add_hop(self, values=None) -> None:
        r = self.hops_table.rowCount()
        self.hops_table.insertRow(r)
        self._set_row(self.hops_table, r, values or [1, 2, 0.35, "", "1e12", 0])
        self._refresh_vary_list()

    def remove_hop(self) -> None:
        r = self.hops_table.currentRow()
        if r < 0:
            r = self.hops_table.rowCount() - 1
        if r >= 0:
            self.hops_table.removeRow(r)
        self._refresh_vary_list()

    def load_example(self) -> None:
        """The XReSpecs example: 7Li, 20 and 0 ppm, Ea 0.35 / 0.37 eV."""
        self.sites_table.setRowCount(0)
        self.hops_table.setRowCount(0)
        self.add_site(["A", 20, 0.5, 0.1])
        self.add_site(["B", 0, 0.5, 0.1])
        self.add_hop([1, 2, 0.35, 0.37, "1e11", 0.0005])
        self.nucleus_edit.setText("7Li")
        self.field_spin.setValue(18.79)
        self.temperatures_edit.setText("200:350:25")

    def _cell(self, table, r, c):
        item = table.item(r, c)
        return item.text().strip() if item else ""

    def sites(self) -> list:
        out = []
        for r in range(self.sites_table.rowCount()):
            shift = _float(self._cell(self.sites_table, r, 1))
            if shift is None:
                raise ValueError(f"Site {r + 1} needs a shift.")
            fwhm = _float(self._cell(self.sites_table, r, 2), 0.0)
            sd = _float(self._cell(self.sites_table, r, 3), 0.0)
            if fwhm < 0 or sd < 0:
                raise ValueError(f"Site {r + 1}: widths cannot be negative.")
            out.append({"label": self._cell(self.sites_table, r, 0) or f"S{r + 1}",
                        "shift_ppm": shift, "fwhm_ppm": fwhm, "shift_sd_ppm": sd})
        if not out:
            raise ValueError("Add at least one site.")
        return out

    def hops(self) -> list:
        n = self.sites_table.rowCount()
        out = []
        for r in range(self.hops_table.rowCount()):
            a = _float(self._cell(self.hops_table, r, 0))
            b = _float(self._cell(self.hops_table, r, 1))
            Ea = _float(self._cell(self.hops_table, r, 2))
            if a is None or b is None or Ea is None:
                raise ValueError(f"Hop {r + 1} needs From, To and Ea →.")
            a, b = int(a), int(b)
            if not (1 <= a <= n and 1 <= b <= n) or a == b:
                raise ValueError(f"Hop {r + 1} must join two different sites among 1–{n}.")
            hop = {"from": a - 1, "to": b - 1, "Ea_eV": Ea,
                   "nu0_Hz": _float(self._cell(self.hops_table, r, 4), 1e12)}
            Ea_r = _float(self._cell(self.hops_table, r, 3))
            if Ea_r is not None:
                hop["Ea_rev_eV"] = Ea_r
            sd = _float(self._cell(self.hops_table, r, 5), 0.0)
            if sd:
                hop["Ea_sd_eV"] = sd
            out.append(hop)
        return out

    def _update_larmor(self) -> None:
        try:
            self.larmor_label.setText(f"= {self.larmor_MHz():.4g} MHz")
        except Exception:  # noqa: BLE001 -- unknown nucleus while typing
            self.larmor_label.setText("(unknown nucleus)")

    def larmor_MHz(self) -> float:
        from ..notebook.simulations import larmor_frequency_Hz
        return larmor_frequency_Hz(self.nucleus_edit.text().strip(), self.field_spin.value()) / 1e6

    def _refresh_vary_list(self) -> None:
        if not hasattr(self, "vary_list"):
            return
        ticked = {self.vary_list.item(i).data(Qt.ItemDataRole.UserRole)
                  for i in range(self.vary_list.count())
                  if self.vary_list.item(i).checkState() == Qt.CheckState.Checked}
        self.vary_list.clear()
        names = []
        for r in range(self.sites_table.rowCount()):
            label = self._cell(self.sites_table, r, 0) or f"S{r + 1}"
            names += [(f"site{r + 1}.shift_ppm", f"{label}: shift"),
                      (f"site{r + 1}.fwhm_ppm", f"{label}: FWHM"),
                      (f"site{r + 1}.shift_sd_ppm", f"{label}: shift s.d.")]
        for r in range(self.hops_table.rowCount()):
            names += [(f"hop{r + 1}.Ea_eV", f"Hop {r + 1}: Ea →"),
                      (f"hop{r + 1}.Ea_rev_eV", f"Hop {r + 1}: Ea ←"),
                      (f"hop{r + 1}.log10_nu0", f"Hop {r + 1}: log₁₀ ν₀"),
                      (f"hop{r + 1}.log10_k", f"Hop {r + 1}: log₁₀ k (fixed rate, one T)")]
        names.append(("gauss_fwhm_ppm", "Gaussian broadening"))
        for key, text in names:
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, key)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if key in ticked else Qt.CheckState.Unchecked)
            self.vary_list.addItem(item)

    def set_vary(self, keys) -> None:
        keys = set(keys)
        for i in range(self.vary_list.count()):
            item = self.vary_list.item(i)
            item.setCheckState(Qt.CheckState.Checked if item.data(Qt.ItemDataRole.UserRole) in keys
                               else Qt.CheckState.Unchecked)

    def vary(self) -> list:
        return [self.vary_list.item(i).data(Qt.ItemDataRole.UserRole)
                for i in range(self.vary_list.count())
                if self.vary_list.item(i).checkState() == Qt.CheckState.Checked]

    # ----------------------------------------------------------- actions
    def simulate(self) -> list | None:
        try:
            sites, hops = self.sites(), self.hops()
            temps = parse_temperatures(self.temperatures_edit.text())
            larmor = self.larmor_MHz()
            results = ex.exchange_spectra_vs_temperature(
                sites, hops, temps, larmor_MHz=larmor, gauss_fwhm_ppm=self.gauss_spin.value(),
                n_dist=self.nodes_spin.value())
        except (ValueError, KeyError) as exc:
            QMessageBox.warning(self, "Cannot simulate", str(exc))
            return None
        self.results = results
        self._stack_drawn = (results, self.nucleus_edit.text().strip(), larmor)
        self._redraw_stack(raise_view=True)
        self._log(results[0]["log"].splitlines()[0].split(" at ")[0]
                  + f" at {len(results)} temperature(s): "
                  + ", ".join(f"{r['T_K']:g}" for r in results) + " K.")
        for r in results:
            self._log(r["log"])
        info = []
        for h, hop in enumerate(hops):
            s = sites
            same_energy = hop.get("Ea_rev_eV") in (None, hop["Ea_eV"])
            delta = abs(s[hop["from"]]["shift_ppm"] - s[hop["to"]]["shift_ppm"])
            if same_energy and delta > 0:
                try:
                    Tc = ex.coalescence_temperature(delta, larmor, hop["Ea_eV"], hop["nu0_Hz"])
                    info.append(f"hop {h + 1} coalesces near {Tc:.0f} K")
                except ValueError as exc:
                    info.append(f"hop {h + 1}: {exc}")
        self.results_label.setText(
            f"Fully narrowed line at {results[-1]['fast_limit_ppm']:.2f} ppm at "
            f"{results[-1]['T_K']:g} K." + (" " + "; ".join(info) + "." if info else ""))
        return results

    def _redraw_stack(self, raise_view: bool = False) -> None:
        """The simulated stack, as last simulated, with the current plot
        options -- only while it is the plot on show (or on a new one)."""
        drawn = getattr(self, "_stack_drawn", None)
        if self.canvas is None or drawn is None:
            return
        if not raise_view and getattr(self.canvas, "plot_owner", None) != "exchange_stack":
            return
        results, nucleus, larmor = drawn
        mark = self.fast_limit_check.isChecked()
        size = _text_size(self.canvas)
        _show(self.canvas,
              lambda fig: ex.plot_exchange_stack(results, ax=fig.add_subplot(111),
                                                 nucleus=nucleus, show_fast_limit=mark,
                                                 text_size=size),
              "compare", hz_per_ppm=larmor)
        self.canvas.plot_owner = "exchange_stack"
        if raise_view:
            self.show_view()

    def fit_selected(self) -> dict | None:
        spectra = self.get_spectra() or []
        if not spectra:
            QMessageBox.information(self, "No spectra",
                                    "Select one or more 1D spectra in Open experiments.")
            return None
        vary = self.vary()
        if not vary:
            QMessageBox.information(self, "Nothing to fit", "Tick at least one parameter.")
            return None
        try:
            temps_typed = parse_temperatures(self.temperatures_edit.text())
        except ValueError:
            temps_typed = []
        prepared = []
        for i, s in enumerate(spectra):
            T = s.get("T_K")
            if T is None:
                if i < len(temps_typed):
                    T = temps_typed[i]
                else:
                    QMessageBox.warning(self, "Temperature missing",
                                        f"{s['label']} has no recorded temperature; list the "
                                        "temperatures, in order, under Temperatures.")
                    return None
            entry = {"ppm": s["ppm"], "y": s["y"], "T_K": float(T)}
            if self.window_check.isChecked():
                entry["window"] = (self.win_lo.value(), self.win_hi.value())
            prepared.append(entry)
        larmor = spectra[0].get("larmor_MHz") or self.larmor_MHz()
        try:
            fit = ex.fit_exchange_spectra(prepared, self.sites(), self.hops(), vary,
                                          larmor_MHz=larmor, gauss_fwhm_ppm=self.gauss_spin.value(),
                                          n_dist=min(self.nodes_spin.value(), 5))
        except (ValueError, RuntimeError) as exc:
            QMessageBox.warning(self, "Fit failed", str(exc))
            return None
        self.fit = fit
        self._write_back(fit)
        nucleus = spectra[0].get("nucleus") or self.nucleus_edit.text().strip()
        if self.canvas is not None:
            size = _text_size(self.canvas)
            _show(self.canvas,
                  lambda fig: ex.plot_exchange_fit(fit, ax=fig.add_subplot(111), nucleus=nucleus,
                                                   text_size=size),
                  "compare", hz_per_ppm=larmor)
            self.show_view()
        self.results_label.setText(
            "Fitted: " + "; ".join(f"{k} = {v:.4g} ± {fit['errors'][k]:.2g}"
                                   for k, v in fit["params"].items()))
        self._log(fit["log"])
        return fit

    def _write_back(self, fit) -> None:
        """Fitted values into the tables, so Simulate and the next fit
        start from them."""
        for r, site in enumerate(fit["sites"]):
            self._set_row(self.sites_table, r, [site.get("label", f"S{r + 1}"),
                                                f"{site['shift_ppm']:.6g}",
                                                f"{site.get('fwhm_ppm', 0.0):.6g}",
                                                f"{site.get('shift_sd_ppm', 0.0):.6g}"])
        for r, hop in enumerate(fit["hops"]):
            Ea_r = hop.get("Ea_rev_eV")
            self._set_row(self.hops_table, r, [hop["from"] + 1, hop["to"] + 1,
                                               f"{hop['Ea_eV']:.6g}",
                                               "" if Ea_r is None else f"{Ea_r:.6g}",
                                               f"{hop.get('nu0_Hz', 1e12):.4g}",
                                               f"{hop.get('Ea_sd_eV', 0.0):.4g}"])
        self.gauss_spin.setValue(fit["gauss_fwhm_ppm"])

    def notebook_code(self) -> str:
        return ex.exchange_notebook_code(self.sites(), self.hops(),
                                         parse_temperatures(self.temperatures_edit.text()),
                                         self.larmor_MHz(), self.gauss_spin.value(),
                                         self.nucleus_edit.text().strip(),
                                         self.nodes_spin.value())

    def copy_notebook_code(self) -> None:
        try:
            code = self.notebook_code()
        except (ValueError, KeyError) as exc:
            QMessageBox.warning(self, "Copy notebook code", str(exc))
            return
        _copy(code)
        self._log("Copied notebook code to the clipboard.")

    def _log(self, text: str) -> None:
        self.status_log.appendPlainText(text)
