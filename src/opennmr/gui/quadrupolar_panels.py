"""Fit > Quadrupolar: fit static or MAS quadrupolar powder lineshapes.

The maths is :mod:`opennmr.notebook.quadrupolar_fitting` (the same
functions a notebook calls); this panel collects the settings, runs the
fit on the selected spectrum (via ``notebook.bridge``), and reports the
result as a table plus a setup/quality/warnings line.
"""

from __future__ import annotations

import time

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox,
    QDoubleSpinBox, QFormLayout, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMenu,
    QMessageBox, QPushButton, QSpinBox, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from .fit_progress_dialog import FitProgress
from ..core import registry as core_registry
from ..io import serialize
from . import panel_state
from .widgets.report_label import ReportLabel
from .widgets.field_input import link_field
from .widgets.sideband_hint import sideband_hint_label
from .widgets.ranges_edit import PpmRangesEdit
from ..notebook import bridge as notebook_bridge
from ..notebook import fitting as notebook_fitting
from ..notebook import quadrupolar_fitting as qfit

#: Sites table columns, and which fit parameter each holds.
COLUMNS = ("Label", "δiso / ppm", "C_Q / MHz", "η_Q", "FWHM / ppm", "L fraction")
COLUMN_PARAMS = {1: "d_iso", 2: "C_Q", 3: "eta_Q", 4: "fwhm", 5: "eta"}
PARAM_LABELS = {"d_iso": "δiso (ppm)", "C_Q": "C_Q (MHz)", "eta_Q": "η_Q",
                "fwhm": "FWHM (ppm)", "amplitude": "Amplitude", "eta": "L fraction"}
_BOUNDS_ROLE = Qt.ItemDataRole.UserRole
_FIXED_ROLE = Qt.ItemDataRole.UserRole + 1


class QuadrupolarFitPanel(QWidget):
    """Callbacks: ``get_target()`` -> (spectrum, row, index) or raises
    ValueError; ``get_regions()`` -> Integration regions; ``draw_fit(result)``;
    ``format_value_error(value, error, fixed)``; ``send_to_simulation(settings)``
    with a dict of the first site's values and the fit's setup."""

    #: The sites table: its columns, which fit parameter each holds, and
    #: readable parameter names. Class attributes so a panel for another
    #: model of the same experiment (Czjzek) can reuse everything else.
    COLUMNS = COLUMNS
    COLUMN_PARAMS = COLUMN_PARAMS
    PARAM_LABELS = PARAM_LABELS
    #: ``popt``'s per-site parameter order.
    PARAM_NAMES = tuple(qfit.QUAD_PARAM_NAMES)
    BLURB = ("Fit quadrupolar powder lineshapes (static or MAS) to the selected 1D spectrum "
             "(or the displayed slice of a 2D). One row per site; Estimate reads a starting "
             "C_Q and δiso off the data. The fitted model is the Simulate > Quadrupolar one.")
    FIT_TEXT = "Fit Quadrupolar Lineshape"
    SEND_TEXT = "Send to Simulate > Quadrupolar"
    DEFAULT_SITE = ("0", "3", "0.5", "1", "0.5")

    def __init__(self, get_target, get_regions, draw_fit, format_value_error,
                 send_to_simulation=None, on_fitted=None, parent=None) -> None:
        super().__init__(parent)
        # Called after every successful fit (the processing log, saved
        # with the spectrum). Optional, so the panel works on its own.
        self._on_fitted = on_fitted or (lambda: None)
        self._get_target = get_target
        self._get_regions = get_regions
        self._draw_fit = draw_fit
        self._fmt = format_value_error
        self._send_to_simulation = send_to_simulation
        self._seeded_for = None
        self.result: dict | None = None

        v = QVBoxLayout(self)
        blurb = QLabel(self.BLURB)
        blurb.setWordWrap(True)
        self.blurb_label = blurb
        v.addWidget(blurb)

        self.sites_table = QTableWidget(0, len(self.COLUMNS))
        self.sites_table.setHorizontalHeaderLabels(list(self.COLUMNS))
        self.sites_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.sites_table.setMaximumHeight(150)
        self.sites_table.setToolTip(
            "Starting values, one row per site. δiso is the isotropic chemical shift (the CT "
            "appears below it by the second-order quadrupolar shift). C_Q's sign cannot be "
            "fitted from a lineshape, so |C_Q| is used.\n\nRight-click a value to fix it, or "
            "for bounds (fixed values are shown in italics).")
        self.sites_table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.sites_table.customContextMenuRequested.connect(self._on_sites_menu)
        v.addWidget(self.sites_table)

        buttons = QHBoxLayout()
        self.add_button = QPushButton("Add site")
        self.add_button.clicked.connect(lambda: self.add_site())
        self.remove_button = QPushButton("Remove site")
        self.remove_button.clicked.connect(self.remove_site)
        self.estimate_button = QPushButton("Estimate from data")
        self.estimate_button.setToolTip(
            "Set the first site's C_Q and δiso from the pattern's width and centre of gravity "
            "over the fit window (η_Q is kept as typed). A start for the fit, not a result.")
        self.estimate_button.clicked.connect(self.estimate)
        for b in (self.add_button, self.remove_button, self.estimate_button):
            buttons.addWidget(b)
        buttons.addStretch(1)
        v.addLayout(buttons)

        form = QFormLayout()
        self.nucleus_combo = QComboBox()
        self.nucleus_combo.setEditable(True)
        self.nucleus_combo.addItems(core_registry.quadrupolar_nuclides(40))
        self.nucleus_combo.setCurrentIndex(-1)
        self.nucleus_combo.lineEdit().setPlaceholderText("from the spectrum, e.g. 27Al")
        self.nucleus_combo.setToolTip("The observed nucleus; sets the spin I. Seeded from the "
                                      "selected spectrum.")
        self.nucleus_combo.currentTextChanged.connect(lambda _t: self._sync_spin())
        form.addRow("Nucleus:", self.nucleus_combo)
        self.spin_label = QLabel("—")
        form.addRow("Spin I:", self.spin_label)

        self.larmor_spin = QDoubleSpinBox()
        self.larmor_spin.setRange(0.0, 5000.0)
        self.larmor_spin.setDecimals(4)
        self.larmor_spin.setSuffix(" MHz")
        self.larmor_spin.setToolTip("The spectrum's SFO1 (seeded from it). Second-order "
                                    "effects scale as C_Q²/ν0, so it must be right.")
        form.addRow("Larmor frequency:", self.larmor_spin)

        self.regime_combo = QComboBox()
        self.regime_combo.addItem("Static powder", "static")
        self.regime_combo.addItem("MAS (centreband + sidebands)", "mas")
        self.regime_combo.currentIndexChanged.connect(lambda _i: self._sync_controls())
        form.addRow("Regime:", self.regime_combo)
        self.mas_spin = QDoubleSpinBox()
        self.mas_spin.setRange(0.0, 200000.0)
        self.mas_spin.setDecimals(0)
        self.mas_spin.setSingleStep(1000.0)
        self.mas_spin.setSuffix(" Hz")
        self.mas_spin.setToolTip("Spinning rate at the magic angle; read from the file (MASR) "
                                 "where recorded.")
        form.addRow("MAS rate:", self.mas_spin)
        self.sideband_hint = sideband_hint_label()
        form.addRow(self.sideband_hint)

        self.transitions_combo = QComboBox()
        self.transitions_combo.addItem("Central transition only", "ct")
        self.transitions_combo.addItem("CT + satellites", "all")
        self.transitions_combo.addItem("Satellites only", "st")
        self.transitions_combo.setToolTip(
            "Which transitions the model contains. CT only is the usual choice for a "
            "half-integer nucleus; an integer spin (2H, 14N) has satellites only.")
        form.addRow("Transitions:", self.transitions_combo)

        self.order_combo = QComboBox()
        self.order_combo.addItem("Second-order perturbation", "second")
        self.order_combo.addItem("Exact (diagonalization)", "exact")
        self.order_combo.addItem("First-order only", "first")
        self.order_combo.setToolTip(
            "Second order is fast and accurate while ν_Q ≪ ν0 (the report warns when it is "
            "not). Exact includes every order but is slower, especially under MAS. First "
            "order cannot describe a central transition at all.")
        form.addRow("Model:", self.order_combo)

        self.broadening_combo = QComboBox()
        self.broadening_combo.addItem("Gaussian", "gaussian")
        self.broadening_combo.addItem("Lorentzian", "lorentzian")
        self.broadening_combo.addItem("Pseudo-Voigt", "pseudo_voigt")
        self.broadening_combo.currentIndexChanged.connect(lambda _i: self._sync_controls())
        form.addRow("Broadening:", self.broadening_combo)

        self.window_combo = QComboBox()
        self.window_combo.addItem("Whole spectrum", "all")
        self.window_combo.addItem("Range (ppm)", "range")
        self.window_combo.addItem("Integration regions", "regions")
        self.window_combo.addItem("Several ppm ranges", "ranges")
        self.window_combo.setToolTip("The part of the spectrum the fit sees. Keep the whole "
                                     "pattern (and, under MAS, its sidebands) inside it.")
        self.window_combo.currentIndexChanged.connect(lambda _i: self._sync_controls())
        form.addRow("Fit window:", self.window_combo)
        self._window_box = QWidget()
        row = QHBoxLayout(self._window_box)
        row.setContentsMargins(0, 0, 0, 0)
        self.window_lo_spin, self.window_hi_spin = QDoubleSpinBox(), QDoubleSpinBox()
        for spin, value in ((self.window_lo_spin, -200.0), (self.window_hi_spin, 200.0)):
            spin.setRange(-1e6, 1e6)
            spin.setDecimals(2)
            spin.setSuffix(" ppm")
            spin.setValue(value)
        row.addWidget(self.window_lo_spin)
        row.addWidget(QLabel("to"))
        row.addWidget(self.window_hi_spin)
        form.addRow("Range:", self._window_box)
        self.window_ranges_edit = PpmRangesEdit()
        form.addRow("Ranges (ppm):", self.window_ranges_edit)
        self._form = form

        self.fit_baseline_check = QCheckBox("Fit the baseline too")
        self.fit_baseline_check.setToolTip(
            "Fit the linear baseline along with the sites, instead of estimating it from the "
            "ends of the spectrum and holding it. Use when a broad pattern leaves little "
            "baseline at the ends (the report warns when the pattern reaches the window edge).")
        form.addRow("", self.fit_baseline_check)

        self.orientations_spin = QSpinBox()
        self.orientations_spin.setRange(0, 200000)
        self.orientations_spin.setSingleStep(1000)
        self.orientations_spin.setSpecialValueText("auto")
        self.orientations_spin.setValue(0)
        self.orientations_spin.setToolTip(
            "Powder orientations while fitting (in one octant, which is exact by symmetry, so "
            "8× denser than a whole-sphere set). Auto: 28,657 static, 4,181 MAS. Too few leaves "
            "powder ripple that the fit absorbs into a larger FWHM.")
        form.addRow("Orientations (fitting):", self.orientations_spin)
        self.render_orientations_spin = QSpinBox()
        self.render_orientations_spin.setRange(100, 500000)
        self.render_orientations_spin.setSingleStep(5000)
        self.render_orientations_spin.setValue(4181)
        self.render_orientations_spin.setToolTip("Orientations for the drawn curve, once fitted.")
        form.addRow("Orientations (final):", self.render_orientations_spin)
        v.addLayout(form)
        # The magnet's field in front of the Larmor box, as in Simulate ▸
        # Quadrupolar (explicit user request for the Czjzek panel, which
        # shares this code): the Larmor frequency follows from the field
        # and the nucleus, and a selected spectrum's SFO1 moves the field.
        self.field_link = link_field(form, self.larmor_spin,
                                     lambda: self.nucleus_combo.currentText(),
                                     self.nucleus_combo.currentTextChanged)
        self.field_spin = self.field_link.field_spin

        self.fit_button = QPushButton(self.FIT_TEXT)
        self.fit_button.clicked.connect(self.fit)
        v.addWidget(self.fit_button)

        self.results_label = ReportLabel("")
        self.results_label.setWordWrap(True)
        self.results_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        v.addWidget(self.results_label)
        self.results_table = QTableWidget(0, 0)
        self.results_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.results_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents)
        self.results_table.horizontalHeader().setStretchLastSection(True)
        self.results_table.setMaximumHeight(200)
        self.results_table.setToolTip(
            "Fitted values with 1σ uncertainties. P_Q = C_Q·√(1+η_Q²/3); CT c.o.g. is the "
            "central transition's centre of gravity (δiso plus its isotropic quadrupolar "
            "shift) -- both are well determined even when C_Q and δiso are correlated.\n\n"
            "Right-click to copy the table or the full report.")
        self.results_table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.results_table.customContextMenuRequested.connect(self._on_results_menu)
        self.results_table.setVisible(False)
        v.addWidget(self.results_table)

        transfer = QHBoxLayout()
        self.use_fitted_button = QPushButton("Use Fitted Values as Start")
        self.use_fitted_button.setToolTip(
            "Copy every fitted site's values into the sites table, so the next fit starts from "
            "this best fit (fixes and bounds are kept).")
        self.use_fitted_button.setEnabled(False)
        self.use_fitted_button.clicked.connect(self.use_fitted_as_start)
        self.send_button = QPushButton(self.SEND_TEXT)
        self.send_button.setToolTip("Open a fitted site -- the one selected in the results "
                                    "table, else the first -- in the simulator, with the fit's "
                                    "field, spinning, transitions and level of theory.")
        self.send_button.setEnabled(False)
        self.send_button.clicked.connect(self._on_send)
        transfer.addWidget(self.use_fitted_button)
        transfer.addWidget(self.send_button)
        v.addLayout(transfer)
        v.addStretch(1)

        self.add_site()
        self._sync_controls()

    # ---- saved state (processing log / Load Progress) ------------------

    def get_state(self) -> dict:
        """Every input, including the sites and their bounds/fixes."""
        return panel_state.default_state(self)

    def set_state(self, state: dict) -> None:
        panel_state.restore_default_state(self, state)
        for row in range(self.sites_table.rowCount()):
            bounds, fixed = self._constraints(row)
            self.set_constraints(row, bounds, fixed)  # the italics and tooltips
        self._sync_controls()
        self.spin_label.setText("—" if self.spin() is None else f"{self.spin():g}")

    def result_state(self):
        """The last fit, encoded exactly (without the spectrum object,
        which the caller re-attaches)."""
        if not self.result:
            return None
        return serialize.encode({k: v for k, v in self.result.items() if k != "spectrum"})

    def restore_result(self, encoded, spectrum=None) -> None:
        """Show a saved fit's table and report as they were -- nothing
        is re-fitted and nothing is redrawn."""
        result = serialize.decode(encoded)
        result["spectrum"] = spectrum
        self.result = result
        self._report()
        self.send_button.setEnabled(self._send_to_simulation is not None)
        self.use_fitted_button.setEnabled(True)

    def clear_result(self) -> None:
        self.result = None
        self.results_label.setText("")
        self.results_table.clear()
        self.results_table.setRowCount(0)
        self.results_table.setVisible(False)
        self.send_button.setEnabled(False)
        self.use_fitted_button.setEnabled(False)

    def report_text(self) -> str:
        """What the log records of the fit: the summary line(s) and table."""
        if not self.result:
            return ""
        return self.results_label.text() + "\n" + panel_state.table_text(self.results_table)

    # ---- controls ------------------------------------------------------

    def _sync_controls(self) -> None:
        self.mas_spin.setEnabled(self.regime_combo.currentData() == "mas")
        self.sideband_hint.setVisible(self.regime_combo.currentData() == "mas")
        self.sites_table.setColumnHidden(5, self.broadening_combo.currentData() != "pseudo_voigt")
        self._window_box.setVisible(self.window_combo.currentData() == "range")
        label = self._form.labelForField(self._window_box)
        if label is not None:
            label.setVisible(self.window_combo.currentData() == "range")
        self._form.setRowVisible(self.window_ranges_edit,
                                 self.window_combo.currentData() == "ranges")

    def spin(self) -> float | None:
        return core_registry.find_spin(self.nucleus_combo.currentText().strip())

    def _sync_spin(self) -> None:
        spin = self.spin()
        self.spin_label.setText("—" if spin is None else f"{spin:g}")
        if spin is not None and spin >= 1 and float(spin).is_integer():
            # No central transition: the only sensible default.
            self.transitions_combo.setCurrentIndex(self.transitions_combo.findData("all"))

    def seed_from_spectrum(self, spectrum) -> None:
        """Nucleus, SFO1 and MAS rate from the selected spectrum -- re-seeded
        when the experiment changes, never over a hand edit of the same one."""
        if spectrum is None or spectrum is self._seeded_for:
            return
        self._seeded_for = spectrum
        meta = getattr(spectrum, "metadata", None)
        nucleus = getattr(meta, "nucleus", None)
        spin = core_registry.find_spin(nucleus) if nucleus else None
        if spin is not None and spin >= 1:
            self.nucleus_combo.setCurrentText(nucleus)
        sfo1 = getattr(meta, "sfo1", None)
        if sfo1:
            self.larmor_spin.setValue(float(sfo1))
        try:
            masr = float(meta.raw.get("acqus", {}).get("MASR", 0) or 0)
        except (AttributeError, TypeError, ValueError):
            masr = 0.0
        if masr > 0:
            self.mas_spin.setValue(masr)
            self.regime_combo.setCurrentIndex(self.regime_combo.findData("mas"))

    # ---- sites table ---------------------------------------------------

    def add_site(self, values=None) -> None:
        row = self.sites_table.rowCount()
        values = values or (f"Site {row + 1}",) + tuple(self.DEFAULT_SITE)
        self.sites_table.insertRow(row)
        for column, value in enumerate(values):
            self.sites_table.setItem(row, column, QTableWidgetItem(str(value)))

    def remove_site(self) -> None:
        row = self.sites_table.currentRow()
        row = row if row >= 0 else self.sites_table.rowCount() - 1
        if row >= 0:
            self.sites_table.removeRow(row)

    def _constraints(self, row):
        item = self.sites_table.item(row, 0)
        if item is None:
            return {}, {}
        return dict(item.data(_BOUNDS_ROLE) or {}), dict(item.data(_FIXED_ROLE) or {})

    def set_constraints(self, row, bounds, fixed) -> None:
        item = self.sites_table.item(row, 0)
        item.setData(_BOUNDS_ROLE, bounds or None)
        item.setData(_FIXED_ROLE, fixed or None)
        for column, name in self.COLUMN_PARAMS.items():
            cell = self.sites_table.item(row, column)
            if cell is None:
                continue
            font = cell.font()
            font.setItalic(bool(fixed.get(name)))
            cell.setFont(font)
        notes = [f"{self.PARAM_LABELS[n]} fixed" for n, on in fixed.items() if on]
        notes += [f"{self.PARAM_LABELS[n]} in [{lo:g}, {hi:g}]" for n, (lo, hi) in bounds.items()]
        item.setToolTip("\n".join(notes))

    def _on_sites_menu(self, pos) -> None:
        index = self.sites_table.indexAt(pos)
        row = index.row() if index.isValid() else self.sites_table.currentRow()
        if row < 0:
            return
        bounds, fixed = self._constraints(row)
        menu = QMenu(self)
        name = self.COLUMN_PARAMS.get(index.column()) if index.isValid() else None
        toggle = (menu.addAction(("Unfix " if fixed.get(name) else "Fix ") + self.PARAM_LABELS[name])
                  if name else None)
        dialog = menu.addAction("Set Bounds / Fixed Parameters…")
        clear = menu.addAction("Clear Bounds and Fixed") if (bounds or fixed) else None
        chosen = menu.exec(self.sites_table.viewport().mapToGlobal(pos))
        if chosen is None:
            return
        if chosen is toggle:
            fixed[name] = not fixed.get(name, False)
            self.set_constraints(row, bounds, {k: v for k, v in fixed.items() if v})
        elif chosen is dialog:
            self._bounds_dialog(row)
        elif chosen is clear:
            self.set_constraints(row, {}, {})

    def _bounds_dialog(self, row) -> None:
        bounds, fixed = self._constraints(row)
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Bounds / fixed parameters -- site {row + 1}")
        form = QFormLayout(dialog)
        edits, checks = {}, {}
        for name, label in self.PARAM_LABELS.items():
            edit = QLineEdit(f"{bounds[name][0]:g}, {bounds[name][1]:g}" if name in bounds else "")
            edit.setPlaceholderText("optional, 'low, high'")
            check = QCheckBox("Fixed")
            check.setChecked(bool(fixed.get(name)))
            holder = QWidget()
            h = QHBoxLayout(holder)
            h.setContentsMargins(0, 0, 0, 0)
            h.addWidget(edit)
            h.addWidget(check)
            form.addRow(f"{label}:", holder)
            edits[name], checks[name] = edit, check
        box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        box.accepted.connect(dialog.accept)
        box.rejected.connect(dialog.reject)
        form.addRow(box)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        new_bounds = {}
        for name, edit in edits.items():
            text = edit.text().strip()
            if not text:
                continue
            try:
                lo, hi = (float(t) for t in text.split(","))
            except ValueError:
                QMessageBox.critical(self, "Invalid bounds",
                                     f"{self.PARAM_LABELS[name]}: expected 'low, high', got {text!r}.")
                return
            new_bounds[name] = (min(lo, hi), max(lo, hi))
        self.set_constraints(row, new_bounds,
                             {n: True for n, c in checks.items() if c.isChecked()})

    def read_sites(self) -> list[dict]:
        sites = []
        for row in range(self.sites_table.rowCount()):
            def cell(column, name):
                item = self.sites_table.item(row, column)
                text = item.text().strip() if item is not None else ""
                try:
                    return float(text)
                except ValueError as exc:
                    raise ValueError(f"Site row {row + 1}: {name} must be a number, "
                                     f"got {text!r}.") from exc
            label_item = self.sites_table.item(row, 0)
            site = {"label": (label_item.text().strip() if label_item else "") or f"Site {row + 1}",
                    "d_iso": cell(1, "δiso"), "C_Q": cell(2, "C_Q"), "eta_Q": cell(3, "η_Q"),
                    "fwhm": cell(4, "FWHM")}
            item = self.sites_table.item(row, 5)
            if item is not None and item.text().strip():
                site["eta"] = cell(5, "L fraction")
            bounds, fixed = self._constraints(row)
            if fixed:
                site["fixed"] = {n: True for n, on in fixed.items() if on}
            if bounds:
                site["bounds"] = {n: tuple(b) for n, b in bounds.items()}
            sites.append(site)
        if not sites:
            raise ValueError("Add at least one site before fitting.")
        return sites

    # ---- settings --------------------------------------------------------

    def fit_regions(self):
        mode = self.window_combo.currentData()
        if mode == "all":
            return None
        if mode == "range":
            lo, hi = self.window_lo_spin.value(), self.window_hi_spin.value()
            return [(min(lo, hi), max(lo, hi))]
        if mode == "ranges":
            return self.window_ranges_edit.ranges()
        return self._get_regions() or None

    def _settings(self):
        spin = self.spin()
        if spin is None:
            raise ValueError("Choose the observed nucleus (e.g. 27Al) -- it sets the spin I.")
        if spin < 1:
            raise ValueError(f"{self.nucleus_combo.currentText()} (I = {spin:g}) has no "
                             "quadrupole moment.")
        larmor = self.larmor_spin.value()
        if larmor <= 0:
            raise ValueError("Set the field B₀ (or select the spectrum, whose SFO1 sets it).")
        mas = self.mas_spin.value() if self.regime_combo.currentData() == "mas" else 0.0
        if self.regime_combo.currentData() == "mas" and mas <= 0:
            raise ValueError("The MAS lineshape needs a spinning rate: set MAS rate, or "
                             "switch Regime to Static powder.")
        transitions = self.transitions_combo.currentData()
        if transitions == "ct" and float(spin).is_integer():
            raise ValueError(f"I = {spin:g} is an integer spin: it has no central transition. "
                             "Choose CT + satellites.")
        return dict(I=float(spin), larmor_Hz=larmor * 1e6, mas_Hz=mas,
                    order=self.order_combo.currentData(), transitions=transitions,
                    lineshape=self.broadening_combo.currentData(),
                    n_zcw=self.orientations_spin.value() or None,
                    fit_baseline=self.fit_baseline_check.isChecked())

    def _target_results(self):
        spectrum, row, index = self._get_target()
        return spectrum, index, notebook_bridge.results_from_spectrum(spectrum, row=row)

    # ---- actions -----------------------------------------------------------

    def estimate(self) -> None:
        try:
            settings = self._settings()
            _spectrum, _index, results = self._target_results()
            sites = self.read_sites()
        except (ValueError, IndexError) as exc:
            QMessageBox.warning(self, "Estimate", str(exc))
            return
        start = qfit.estimate_quadrupolar_start(
            results, settings["I"], larmor_Hz=settings["larmor_Hz"], mas_Hz=settings["mas_Hz"],
            eta_Q=float(np.clip(sites[0]["eta_Q"], 0, 1)), transitions=settings["transitions"],
            fit_regions=self.fit_regions())
        for column, key in ((1, "d_iso"), (2, "C_Q"), (4, "fwhm")):
            self.sites_table.item(0, column).setText(f"{start[key]:.4g}")

    def fit(self) -> None:
        try:
            settings = self._settings()
            sites = self.read_sites()
            spectrum, index, results = self._target_results()
        except (ValueError, IndexError) as exc:
            QMessageBox.warning(self, "Quadrupolar fit", str(exc))
            return
        regions = self.fit_regions()
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        started = time.perf_counter()
        with FitProgress(self, "Quadrupolar fit") as fit_progress_box:
            try:
                popt, pcov_free, is_free, _scale, info = qfit.fit_quadrupolar_spectrum(
                    results, sites, fit_regions=regions, **settings)
            except Exception as exc:  # noqa: BLE001 -- a fit that fails is an outcome, not a crash
                QApplication.restoreOverrideCursor()
                QMessageBox.warning(self, "Quadrupolar fit", f"The fit did not succeed:\n\n{exc}")
                return
        if fit_progress_box.stopped:
            return
        elapsed = time.perf_counter() - started
        x = np.sort(np.asarray(results["x_ppm"], dtype=float))
        try:
            y_fit, components = qfit.evaluate_quad_fit(
                x, popt, info, return_components=True,
                n_zcw=self.render_orientations_spin.value())
        finally:
            QApplication.restoreOverrideCursor()
        order = np.argsort(results["x_ppm"])
        y = np.asarray(results["real_spectrum"], dtype=float)[order]
        pcov = notebook_fitting.expand_pcov(pcov_free, is_free, len(popt))
        _target, row, _index = self._get_target()
        self.result = {"popt": popt, "pcov": pcov, "is_free": is_free, "info": info,
                       "x": x, "y": y, "y_fit": y_fit, "components": components,
                       "regions": regions, "spectrum": spectrum, "index": index,
                       # Which slice of a pseudo-2D this was fitted to -- the
                       # reference for applying the model to every slice.
                       "row": row,
                       "settings": settings, "elapsed": elapsed,
                       "nucleus": self.nucleus_combo.currentText().strip()}
        self._draw_fit(self.result)
        self._report()
        self.send_button.setEnabled(self._send_to_simulation is not None)
        self.use_fitted_button.setEnabled(True)
        self._on_fitted()

    # ---- report --------------------------------------------------------------

    def _report(self) -> None:
        r = self.result
        popt, pcov, is_free, info = r["popt"], r["pcov"], r["is_free"], r["info"]
        n = info["n_sites"]
        errors = np.sqrt(np.clip(np.diag(pcov), 0.0, None))
        derived = qfit.quadrupolar_derived(popt, pcov, info)
        half_integer = not float(info["sim_kwargs"]["I"]).is_integer()
        pv = info["sim_kwargs"]["lineshape"] == "pseudo_voigt"
        cols = [("δiso\n(ppm)", 0), ("C_Q\n(MHz)", 1), ("η_Q", 2), ("P_Q\n(MHz)", "P_Q")]
        if half_integer:
            cols.append(("CT c.o.g.\n(ppm)", "delta_cg"))
        cols.append(("FWHM\n(ppm)", 3))
        if pv:
            cols.append(("L fraction", 5))
        cols.append(("Amplitude", 4))
        comps = r["components"][:n]
        mask = notebook_fitting.build_region_mask(r["x"], r["regions"]) if r["regions"] else \
            np.ones(r["x"].size, dtype=bool)
        integrate = getattr(np, "trapezoid", None) or np.trapz
        areas = np.array([abs(integrate(c, r["x"])) for c in comps])
        shares = 100 * areas / areas.sum() if areas.sum() else np.full(n, np.nan)

        table = self.results_table
        table.clear()
        headers = [c[0] for c in cols] + ["Area\n(%)"]
        table.setColumnCount(len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.setRowCount(n)
        table.setVerticalHeaderLabels(list(info["labels"]))
        for s in range(n):
            for c, (_h, key) in enumerate(cols):
                if isinstance(key, str):
                    text = self._fmt(float(derived[s][key]), float(derived[s][key + "_err"]))
                else:
                    i = s * len(qfit.QUAD_PARAM_NAMES) + key
                    text = self._fmt(float(popt[i]), float(errors[i]), not is_free[i])
                item = QTableWidgetItem(text.replace(" ± ", "\n± ").replace(" (fixed)", "\n(fixed)"))
                item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                table.setItem(s, c, item)
            share = QTableWidgetItem("—" if not np.isfinite(shares[s]) else f"{shares[s]:.1f}")
            share.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            table.setItem(s, len(cols), share)
        table.resizeRowsToContents()
        table.setVisible(True)

        y, y_fit = r["y"][mask], r["y_fit"][mask]
        quality = notebook_fitting.fit_quality(y, y_fit, int(is_free.sum()))
        total = float(np.sum((y - y.mean()) ** 2))
        r2 = 1 - float(np.sum((y - y_fit) ** 2)) / total if total else float("nan")
        warnings = list(info.get("warnings", [])) + qfit.check_quad_bound_hits(popt, pcov, is_free, info)
        lines = [info["log"].replace("Quadrupolar fit: ", "Quadrupolar fit — "),
                 f"R² {r2:.4f}  ·  RMSE {quality['rmse']:.3g}  ·  reduced χ² "
                 f"{quality['reduced_chi2']:.3g}  ·  {r['elapsed']:.1f} s. |C_Q| reported "
                 "(its sign is not in a lineshape)."]
        lines += [f"⚠ {w}" for w in warnings]
        self.results_label.setText("\n".join(lines))

    def text_report(self) -> str:
        r = self.result
        if not r:
            return ""
        mask = notebook_fitting.build_region_mask(r["x"], r["regions"]) if r["regions"] else \
            np.ones(r["x"].size, dtype=bool)
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            return qfit.print_quadrupolar_fit_report(r["x"][mask], r["y"][mask], r["y_fit"][mask],
                                                     r["popt"], r["pcov"], r["is_free"], r["info"])

    def table_tsv(self) -> str:
        t = self.results_table
        rows = ["\t".join(["Site"] + [t.horizontalHeaderItem(c).text().replace("\n", " ")
                                      for c in range(t.columnCount())])]
        for row in range(t.rowCount()):
            head = t.verticalHeaderItem(row)
            rows.append("\t".join([head.text() if head else str(row + 1)] +
                                  [(t.item(row, c).text().replace("\n", " ") if t.item(row, c) else "")
                                   for c in range(t.columnCount())]))
        return "\n".join(rows)

    def _on_results_menu(self, pos) -> None:
        menu = QMenu(self.results_table)
        copy_table = menu.addAction("Copy Table")
        copy_report = menu.addAction("Copy Full Report")
        chosen = menu.exec(self.results_table.viewport().mapToGlobal(pos))
        if chosen is copy_table:
            QGuiApplication.clipboard().setText(self.table_tsv())
        elif chosen is copy_report:
            QGuiApplication.clipboard().setText(self.text_report())

    # ---- moving values between fits and simulations -------------------------

    def send_site_index(self) -> int:
        """The site Send to Simulate sends: the one selected in the results
        table, else the first."""
        n = int((self.result or {}).get("info", {}).get("n_sites", 1) or 1)
        row = self.results_table.currentRow()
        return row if 0 <= row < n else 0

    def fitted_site_values(self) -> list[dict]:
        """Each fitted site's parameters, ``{name: value}`` by
        :attr:`PARAM_NAMES`."""
        r = self.result
        if not r:
            return []
        popt = np.asarray(r["popt"], dtype=float)
        n, k = int(r["info"]["n_sites"]), len(self.PARAM_NAMES)
        if popt.size < n * k:                  # an older record, fewer per-site values
            k = (popt.size - 2) // n
        return [{name: float(popt[s * k + i]) for i, name in enumerate(self.PARAM_NAMES[:k])}
                for s in range(n)]

    def _write_site(self, row: int, values: dict) -> None:
        for column, name in self.COLUMN_PARAMS.items():
            if name not in values or values[name] is None:
                continue
            value = float(values[name])
            if name == "C_Q":
                value = abs(value)
            item = self.sites_table.item(row, column)
            if item is None:
                item = QTableWidgetItem("")
                self.sites_table.setItem(row, column, item)
            item.setText(f"{value:.6g}")

    def use_fitted_as_start(self) -> None:
        """The last fit's values become the starting values (explicit
        user request: start again from the old best fit). Rows are added
        if the fit had more sites than the table now has; labels, fixes
        and bounds stay."""
        r = self.result
        if not r:
            return
        labels = list(r["info"].get("labels") or [])
        for s, values in enumerate(self.fitted_site_values()):
            if s >= self.sites_table.rowCount():
                self.add_site()
                if s < len(labels):
                    self.sites_table.item(s, 0).setText(labels[s])
            self._write_site(s, values)

    def set_sites_from_simulation(self, sites: list[dict], settings: dict | None = None) -> str:
        """Several simulated sites as the starting values (explicit user
        request): one row per site, labels included. Rows that remain
        keep their bounds and fixed flags; extra rows go. Settings (and
        the nucleus/field checks) as :meth:`set_start_from_simulation`."""
        table = self.sites_table
        while table.rowCount() > len(sites):
            table.removeRow(table.rowCount() - 1)
        for row, site in enumerate(sites):
            if row >= table.rowCount():
                self.add_site()
            label = table.item(row, 0)
            if label is None:
                label = QTableWidgetItem("")
                table.setItem(row, 0, label)
            label.setText(str(site.get("label") or f"Site {row + 1}"))
            self._write_site(row, {"d_iso": site["d_iso"], "C_Q": site["C_Q_MHz"],
                                   "eta_Q": site["eta_Q"], "fwhm": site["fwhm"],
                                   "eta": site.get("eta")})
        return self._apply_simulation_settings(settings or {})

    def set_start_from_simulation(self, values: dict, settings: dict | None = None) -> str:
        """A simulated site as the starting values of the selected row
        (else the first; one is added if the table is empty), with the
        simulation's spinning, transitions and level of theory. The
        nucleus and field stay the spectrum's: a simulation at another
        field would give the wrong start for these data, so a mismatch is
        reported (returned) rather than copied."""
        if self.sites_table.rowCount() == 0:
            self.add_site()
        row = self.sites_table.currentRow()
        row = row if 0 <= row < self.sites_table.rowCount() else 0
        self._write_site(row, values)
        return self._apply_simulation_settings(settings or {})

    def _apply_simulation_settings(self, settings: dict) -> str:
        notes = []
        if settings.get("regime") in ("static", "mas"):
            self.regime_combo.setCurrentIndex(self.regime_combo.findData(settings["regime"]))
            if settings["regime"] == "mas" and settings.get("mas_Hz"):
                self.mas_spin.setValue(float(settings["mas_Hz"]))
        for combo, key in ((self.transitions_combo, "transitions"), (self.order_combo, "order"),
                           (self.broadening_combo, "lineshape")):
            index = combo.findData(settings.get(key))
            if index >= 0:
                combo.setCurrentIndex(index)
        nucleus = str(settings.get("nucleus") or "").strip()
        mine = self.nucleus_combo.currentText().strip()
        if nucleus and mine and nucleus != mine:
            notes.append(f"the simulation is of {nucleus}, the fit of {mine}")
        larmor = settings.get("larmor_MHz")
        if larmor and self.larmor_spin.value() > 0 and \
                abs(float(larmor) - self.larmor_spin.value()) > 1e-3 * self.larmor_spin.value():
            notes.append(f"the simulation used ν0 = {float(larmor):.6g} MHz, the fit uses "
                         f"{self.larmor_spin.value():.6g} MHz (the spectrum's)")
        elif larmor and self.larmor_spin.value() <= 0:
            self.larmor_spin.setValue(float(larmor))
        if nucleus and not mine:
            self.nucleus_combo.setCurrentText(nucleus)
        self._sync_controls()
        return "; ".join(notes)

    def _on_send(self) -> None:
        """Every fitted site to Simulate ▸ Quadrupolar (explicit user
        request: the whole fitted spectrum), each with its fitted share
        of the nuclei as its amount (qfit.quad_site_amounts, largest =
        1); the site selected in the results table (else the first) is
        the one selected there. The single-site keys are that site's, for
        callers that take one."""
        r = self.result
        if not r or self._send_to_simulation is None:
            return
        k = self.send_site_index()
        n, m = int(r["info"]["n_sites"]), len(self.PARAM_NAMES)
        popt = np.asarray(r["popt"], dtype=float)
        p = popt[k * m:]
        s = r["settings"]
        try:
            amounts = qfit.quad_site_amounts(r["x"], popt, r["info"])
        except Exception:  # noqa: BLE001 -- an old record: equal amounts, still every site
            amounts = np.ones(n)
        labels = list(r["info"].get("labels") or [])
        sites = [{"label": labels[i] if i < len(labels) else f"Site {i + 1}",
                  "C_Q_MHz": float(abs(v["C_Q"])), "eta_Q": float(v["eta_Q"]),
                  "d_iso": float(v["d_iso"]), "fwhm": float(v["fwhm"]),
                  "eta": float(np.clip(v.get("eta", 0.5), 0, 1)),
                  "amount": float(amounts[i]) if i < len(amounts) else 1.0}
                 for i, v in enumerate(self.fitted_site_values())]
        self._send_to_simulation({
            "nucleus": r["nucleus"], "C_Q_MHz": float(abs(p[1])), "eta_Q": float(p[2]),
            "delta_iso": float(p[0]), "fwhm": float(p[3]), "larmor_MHz": s["larmor_Hz"] / 1e6,
            "mas_Hz": s["mas_Hz"], "transitions": s["transitions"], "order": s["order"],
            "lineshape": s.get("lineshape", "gaussian"),
            "ppm_range": (float(np.min(r["x"])), float(np.max(r["x"]))),
            "sites": sites, "current": k,
        })
