"""Fit > SIMPSON spectrum: fit a measured spectrum with SIMPSON spectra.

Each SIMPSON spectrum (read in Simulate > SIMPSON, or any 1D experiment you
add here) is a **lineshape template**: the fit finds how much of each is
present (a linear, always-solved-exactly amplitude), and -- optionally --
how far each is shifted and how much extra broadening it needs (Lorentzian
and Gaussian, as a Voigt kernel). A flat or sloping baseline can be added.

The maths is :func:`opennmr.notebook.simpson_fit.fit_simpson_templates`;
*Copy notebook code* gives the equivalent notebook call.

Note on the broadening: a Lorentzian and a Gaussian width are only separable
when the line is well resolved and has good signal-to-noise. For a broad or
noisy line they trade off against each other; the errors shown say when.
"""

from __future__ import annotations

import os
from typing import Callable

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QCursor, QGuiApplication
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
    QFormLayout, QGroupBox, QHBoxLayout, QLabel, QListWidget, QMessageBox, QPushButton,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from ..io import serialize
from ..notebook import simpson_fit
from . import panel_state

BASELINES = (("none", "None"), ("offset", "Constant offset"), ("slope", "Offset and slope"))
COLUMNS = ("Template", "Amplitude", "Shift (ppm)", "Lorentzian lb (Hz)", "Gaussian gb (Hz)",
           "Share of signal")


class SimpsonFitPanel(QWidget):
    """``get_target()`` -> ``(spectrum, row, index)`` of the experiment to
    fit; ``get_regions()`` -> the Integration regions; ``selected_spectrum()``
    -> ``(spectrum, label)`` of the experiment selected in Open experiments;
    ``draw_fit(result)`` draws it; ``on_fitted()`` is called after a fit."""

    def __init__(self, get_target: Callable, get_regions: Callable, selected_spectrum: Callable,
                 draw_fit: Callable, show_view: Callable, on_fitted: Callable | None = None,
                 read_file: Callable | None = None, parent=None) -> None:
        super().__init__(parent)
        self._get_target = get_target
        self._get_regions = get_regions
        self._selected_spectrum = selected_spectrum
        self._draw_fit = draw_fit
        self._show_view = show_view
        self._on_fitted = on_fitted
        self._read_file = read_file
        self.templates: list[tuple[str, object]] = []   # (name, NMRSpectrum)
        self.result: dict | None = None

        layout = QVBoxLayout(self)
        intro = QLabel("Fit the selected experiment with one or more SIMPSON spectra. Send a "
                       "spectrum here from Simulate ▸ SIMPSON, or add one below.")
        intro.setWordWrap(True)
        layout.addWidget(intro)
        layout.addWidget(self._build_templates_box())
        layout.addWidget(self._build_options_box())
        layout.addWidget(self._build_results_box())
        self._sync_controls()

    # -- templates ------------------------------------------------------
    def _build_templates_box(self) -> QGroupBox:
        box = QGroupBox("1. SIMPSON spectra (lineshape templates)")
        v = QVBoxLayout(box)
        self.template_list = QListWidget()
        self.template_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.template_list.setMinimumHeight(70)
        v.addWidget(self.template_list)
        row = QHBoxLayout()
        self.add_selected_button = QPushButton("Add selected experiment")
        self.add_selected_button.setToolTip("Use the spectrum selected in Open experiments (a "
                                            "SIMPSON spectrum, or an FID transformed here).")
        self.add_selected_button.clicked.connect(self._on_add_selected)
        self.add_file_button = QPushButton("Add SIMPSON file…")
        self.add_file_button.clicked.connect(self._on_add_file)
        self.remove_button = QPushButton("Remove")
        self.remove_button.clicked.connect(self._on_remove)
        for b in (self.add_selected_button, self.add_file_button, self.remove_button):
            row.addWidget(b)
        v.addLayout(row)
        return box

    def add_template(self, spectrum, name: str | None = None) -> None:
        """Add a 1D spectrum (what Send to Fit calls)."""
        if spectrum.ndim != 1:
            raise ValueError("A template must be a 1D spectrum.")
        base = name or os.path.basename(getattr(spectrum.metadata, "source_path", "") or "") \
            or f"Simulation {len(self.templates) + 1}"
        unique, k = base, 2
        while unique in [n for n, _ in self.templates]:
            unique, k = f"{base} ({k})", k + 1
        self.templates.append((unique, spectrum))
        self.template_list.addItem(unique)
        self.clear_result()
        self._sync_controls()

    def _on_add_selected(self) -> None:
        chosen = self._selected_spectrum()
        if chosen is None:
            QMessageBox.information(self, "SIMPSON fit", "Select a spectrum in Open experiments.")
            return
        try:
            self.add_template(*chosen)
        except ValueError as exc:
            QMessageBox.information(self, "SIMPSON fit", str(exc))

    def _on_add_file(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "SIMPSON spectra or FIDs", "", "SIMPSON files (*.spe *.fid *.dat *.txt);;All (*)")
        for path in paths:
            try:
                spectrum = self._read_file(path) if self._read_file else None
                if spectrum is None:
                    from ..io import simpson
                    spectrum = simpson.read_1d(path)
                self.add_template(spectrum)
            except Exception as exc:  # noqa: BLE001 -- shown
                QMessageBox.warning(self, "SIMPSON fit", f"{os.path.basename(path)}: {exc}")

    def _on_remove(self) -> None:
        rows = sorted((self.template_list.row(i) for i in self.template_list.selectedItems()),
                      reverse=True)
        for row in rows:
            self.template_list.takeItem(row)
            del self.templates[row]
        self.clear_result()
        self._sync_controls()

    # -- options --------------------------------------------------------
    def _build_options_box(self) -> QGroupBox:
        box = QGroupBox("2. Fit")
        form = QFormLayout(box)
        self.float_shift_check = QCheckBox("Each template may shift")
        self.float_shift_check.setChecked(True)
        self.float_shift_check.setToolTip("Allow a rigid shift of each template (referencing, "
                                          "isotropic-shift differences).")
        form.addRow("", self.float_shift_check)
        self.shift_window_spin = QDoubleSpinBox()
        self.shift_window_spin.setRange(0.001, 1e6)
        self.shift_window_spin.setDecimals(3)
        self.shift_window_spin.setValue(5.0)
        self.shift_window_spin.setSuffix(" ppm")
        self.shift_window_spin.setToolTip("How far each template may move either way.")
        form.addRow("Shift window ±:", self.shift_window_spin)
        self.float_broadening_check = QCheckBox("Each template may broaden")
        self.float_broadening_check.setToolTip("Add Lorentzian and Gaussian broadening to each "
                                               "template (they are partly degenerate).")
        form.addRow("", self.float_broadening_check)
        self.max_broadening_spin = QDoubleSpinBox()
        self.max_broadening_spin.setRange(1.0, 1e6)
        self.max_broadening_spin.setValue(2000.0)
        self.max_broadening_spin.setSuffix(" Hz")
        form.addRow("Largest broadening:", self.max_broadening_spin)
        self.baseline_combo = QComboBox()
        for key, text in BASELINES:
            self.baseline_combo.addItem(text, key)
        self.baseline_combo.setCurrentIndex(1)
        form.addRow("Baseline:", self.baseline_combo)
        self.non_negative_check = QCheckBox("Amplitudes cannot be negative")
        form.addRow("", self.non_negative_check)
        self.use_regions_check = QCheckBox("Only inside the Integration regions")
        self.use_regions_check.setToolTip("Fit just these windows (Integrate ▸ regions).")
        form.addRow("", self.use_regions_check)
        row = QHBoxLayout()
        self.fit_button = QPushButton("Fit")
        self.fit_button.clicked.connect(self.fit)
        self.copy_code_button = QPushButton("Copy notebook code")
        self.copy_code_button.clicked.connect(self._on_copy_code)
        row.addWidget(self.fit_button)
        row.addWidget(self.copy_code_button)
        form.addRow(row)
        return box

    def fit_kwargs(self) -> dict:
        kwargs = dict(float_shift=self.float_shift_check.isChecked(),
                      shift_window=self.shift_window_spin.value(),
                      float_broadening=self.float_broadening_check.isChecked(),
                      max_broadening_hz=self.max_broadening_spin.value(),
                      baseline=self.baseline_combo.currentData(),
                      non_negative=self.non_negative_check.isChecked())
        if self.use_regions_check.isChecked():
            regions = self._get_regions()
            if regions:
                kwargs["fit_regions"] = [list(r) for r in regions]
        return kwargs

    def fit(self) -> dict | None:
        if not self.templates:
            QMessageBox.information(self, "SIMPSON fit", "Add at least one SIMPSON spectrum.")
            return None
        try:
            spectrum, row, index = self._get_target()
        except ValueError as exc:
            QMessageBox.information(self, "SIMPSON fit", str(exc))
            return None
        QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
        try:
            experiment = spectrum
            result = simpson_fit.SIMPSONfit_spectrum(
                experiment, [s for _n, s in self.templates],
                names=[n for n, _s in self.templates], **self.fit_kwargs())
        except Exception as exc:  # noqa: BLE001 -- shown
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self, "SIMPSON fit", str(exc))
            return None
        QApplication.restoreOverrideCursor()
        result["spectrum"] = spectrum
        result["spectrum_index"] = index
        self.result = result
        self._report()
        self._draw_fit(result)
        self._show_view()
        if self._on_fitted:
            self._on_fitted()
        self._sync_controls()
        return result

    # -- results --------------------------------------------------------
    def _build_results_box(self) -> QGroupBox:
        box = QGroupBox("3. Result")
        v = QVBoxLayout(box)
        self.results_label = QLabel("")
        self.results_label.setWordWrap(True)
        self.results_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        v.addWidget(self.results_label)
        self.results_table = QTableWidget(0, len(COLUMNS))
        self.results_table.setHorizontalHeaderLabels(list(COLUMNS))
        self.results_table.horizontalHeader().setStretchLastSection(True)
        self.results_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.results_table.setMinimumHeight(90)
        v.addWidget(self.results_table)
        self.results_table.setVisible(False)
        return box

    @staticmethod
    def _pm(value: float, error: float, digits: int = 4) -> str:
        if error is None or not np.isfinite(error) or error == 0:
            return f"{value:.{digits}g}"
        return f"{value:.{digits}g} ± {error:.2g}"

    def _report(self) -> None:
        result = self.result
        if not result:
            return
        self.results_table.setRowCount(len(result["params"]))
        for r, p in enumerate(result["params"]):
            cells = (p["name"], self._pm(p["amplitude"], p["amplitude_err"]),
                     self._pm(p["shift"], p["shift_err"]), self._pm(p["lb"], p["lb_err"]),
                     self._pm(p["gb"], p["gb_err"]), f"{100 * p['fraction']:.1f} %")
            for c, text in enumerate(cells):
                self.results_table.setItem(r, c, QTableWidgetItem(text))
        self.results_table.setVisible(True)
        base = "; ".join(f"{b['name']} = {self._pm(b['value'], b['error'])}"
                         for b in result.get("baseline_params", []))
        self.results_label.setText(
            f"RMSE {result['rmse']:.4g}, R² {result['r_squared']:.5f} "
            f"({result['n_points']} points, {result['n_free']} free parameters)"
            + (f"\nBaseline: {base}" if base else ""))

    def clear_result(self) -> None:
        self.result = None
        self.results_label.setText("")
        self.results_table.setRowCount(0)
        self.results_table.setVisible(False)

    def result_state(self):
        if not self.result:
            return None
        return serialize.encode({k: v for k, v in self.result.items() if k != "spectrum"})

    def restore_result(self, encoded, spectrum=None) -> None:
        result = serialize.decode(encoded)
        result["spectrum"] = spectrum
        self.result = result
        self._report()

    def report_text(self) -> str:
        if not self.result:
            return ""
        from .panel_state import table_text
        return self.results_label.text() + "\n" + table_text(self.results_table)

    # -- state ----------------------------------------------------------
    def get_state(self) -> dict:
        return {"widgets": panel_state.capture_widgets(self),
                "templates": [{"name": n, "path": getattr(s.metadata, "source_path", "") or ""}
                              for n, s in self.templates]}

    def set_state(self, state: dict) -> None:
        state = state or {}
        panel_state.restore_widgets(self, state.get("widgets") or {})
        if state.get("templates") and not self.templates:
            from ..io import simpson
            for entry in state["templates"]:
                path = entry.get("path")
                if path and os.path.exists(path):
                    try:
                        self.add_template(simpson.read_1d(path), entry.get("name"))
                    except Exception:  # noqa: BLE001 -- the template is just not restored
                        pass

    def notebook_code(self) -> str:
        names = [n for n, _s in self.templates]
        kw = ", ".join(f"{k}={v!r}" for k, v in self.fit_kwargs().items())
        return "\n".join([
            "import opennmr.notebook as nmr", "",
            "# experiment: an NMR1dimport result or spectrum; sims: SIMPSONload(...) results",
            f"result = nmr.SIMPSONfit_spectrum(experiment, sims, names={names!r}, {kw})",
            "fig, ax = nmr.SIMPSONplot_fit(result)"])

    def _on_copy_code(self) -> None:
        QGuiApplication.clipboard().setText(self.notebook_code())

    def _sync_controls(self) -> None:
        self.fit_button.setEnabled(bool(self.templates))
        self.remove_button.setEnabled(bool(self.templates))
