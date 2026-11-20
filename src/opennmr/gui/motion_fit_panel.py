"""Dynamics ▸ Fit anisotropic motion to spectra (the maths is notebook.motion_fit).

Reads the model, the field, MAS, echo and order from the Anisotropic motion panel above it and
the spectra selected in Open experiments (their temperatures come from the files). Two routes:
a rate for every spectrum followed by the Arrhenius line, and one global fit of chosen
parameters to all spectra at their own temperatures.
"""

from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMessageBox,
    QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from ..notebook import motion_fit as F
from .widgets.typed_spinbox import TypedSpinBox
from .dynamics_panels import _copy, _float, _log_box, _show, _text_size
from .widgets.report_label import ReportLabel

COLUMNS = ("Fit", "Parameter", "Start", "Min", "Max", "Fitted", "±")
_FIT, _NAME, _START, _MIN, _MAX, _FITTED, _ERR = range(7)


def _num(value) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return ""
    return f"{float(value):.6g}"


class MotionFitPanel(QWidget):
    """``motion``: the :class:`AnisotropicMotionPanel` that holds the model and the experiment.
    ``on_run()`` is called after each fit (the window journals it)."""

    def __init__(self, motion, canvas=None, show_view=None, on_run=None, parent=None):
        super().__init__(parent)
        self.motion = motion
        self.canvas = canvas
        self.show_view = show_view or (lambda: None)
        self.on_run = on_run or (lambda: None)
        self.fit = None
        self._names: list[str] = []
        self._build()

    # ------------------------------------------------------------------ UI
    def _build(self) -> None:
        layout = QVBoxLayout(self)
        note = QLabel(
            "Fit the model above to the spectra selected in Open experiments (each at its own "
            "temperature). <b>Rate per spectrum</b> finds the rate of one mode in every spectrum "
            "and draws the Arrhenius line through the ones the lineshape can determine – start "
            "here. <b>Global fit</b> then refines chosen parameters (barriers, geometry, "
            "tensors) against all spectra together. Scale and offset of every spectrum are "
            "solved automatically.")
        note.setWordWrap(True)
        note.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(note)

        route = QHBoxLayout()
        route.addWidget(QLabel("Method:"))
        self.route_combo = QComboBox()
        self.route_combo.addItem("Rate per spectrum, then Arrhenius line", "rates")
        self.route_combo.addItem("Global fit of the checked parameters", "global")
        self.route_combo.currentIndexChanged.connect(self._route_changed)
        route.addWidget(self.route_combo)
        route.addStretch(1)
        layout.addLayout(route)

        self.rates_box = QGroupBox("Rate per spectrum")
        rl = QHBoxLayout(self.rates_box)
        self.mode_combo = QComboBox()
        self.mode_combo.setToolTip("The mode whose rate is fitted in each spectrum; the other "
                                   "modes keep the rates of the table above.")
        self.kmin_edit = QLineEdit("1e-1")
        self.kmax_edit = QLineEdit("1e10")
        for e in (self.kmin_edit, self.kmax_edit):
            e.setMaximumWidth(80)
        rl.addWidget(QLabel("Mode:"))
        rl.addWidget(self.mode_combo)
        rl.addWidget(QLabel("scan k from"))
        rl.addWidget(self.kmin_edit)
        rl.addWidget(QLabel("to"))
        rl.addWidget(self.kmax_edit)
        rl.addWidget(QLabel("s⁻¹"))
        rl.addStretch(1)
        layout.addWidget(self.rates_box)

        self.global_box = QGroupBox("Global fit")
        gl = QVBoxLayout(self.global_box)
        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(list(COLUMNS))
        self.table.horizontalHeader().setSectionResizeMode(_NAME, QHeaderView.ResizeMode.Stretch)
        self.table.setMinimumHeight(150)
        self.table.setToolTip(
            "Check the parameters to fit; the start, minimum and maximum are editable. An "
            "Arrhenius rate is fitted as Ea and log10 k at a reference temperature (T_ref): Ea "
            "and ν₀ alone are almost perfectly correlated. Parameters left unchecked keep the "
            "values of the tables above.")
        gl.addWidget(self.table)
        row = QHBoxLayout()
        self.refresh_button = QPushButton("Refresh from the model above")
        self.refresh_button.clicked.connect(self.refresh_parameters)
        self.tref_edit = QLineEdit("")
        self.tref_edit.setPlaceholderText("auto")
        self.tref_edit.setMaximumWidth(70)
        self.tref_edit.setToolTip("Reference temperature (K) of log10 k. Blank: the middle of "
                                  "the data in 1/T.")
        self.starts_spin = TypedSpinBox()
        self.starts_spin.setRange(0, 50)
        self.starts_spin.setToolTip("Extra coarse fits from random starting points inside the "
                                    "bounds before the final fit (needs finite min and max). "
                                    "They show whether the answer depends on where you start.")
        row.addWidget(self.refresh_button)
        row.addWidget(QLabel("T_ref (K):"))
        row.addWidget(self.tref_edit)
        row.addWidget(QLabel("Search starts:"))
        row.addWidget(self.starts_spin)
        row.addStretch(1)
        gl.addLayout(row)
        layout.addWidget(self.global_box)

        opt = QHBoxLayout()
        self.points_spin = TypedSpinBox()
        self.points_spin.setRange(128, 16384)
        self.points_spin.setValue(1024)
        self.points_spin.setToolTip("Points of each simulation (interpolated onto the data). "
                                    "Keep the spacing below half the narrowest line.")
        self.offset_check = QCheckBox("Fit an offset per spectrum")
        self.offset_check.setChecked(True)
        self.window_check = QCheckBox("Only inside the axis range above")
        opt.addWidget(QLabel("Simulation points:"))
        opt.addWidget(self.points_spin)
        opt.addWidget(self.offset_check)
        opt.addWidget(self.window_check)
        opt.addStretch(1)
        layout.addLayout(opt)

        act = QHBoxLayout()
        self.fit_button = QPushButton("Fit")
        self.fit_button.clicked.connect(self.run_fit)
        self.use_button = QPushButton("Use fitted values")
        self.use_button.setToolTip("Write the fitted values into the tables above.")
        self.use_button.setEnabled(False)
        self.use_button.clicked.connect(self.use_fitted_values)
        self.code_button = QPushButton("Copy notebook code")
        self.code_button.clicked.connect(self.copy_notebook_code)
        for b in (self.fit_button, self.use_button, self.code_button):
            act.addWidget(b)
        act.addStretch(1)
        layout.addLayout(act)

        self.results_label = ReportLabel("")
        self.results_label.setWordWrap(True)
        self.results_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.results_label)
        self.status_log = _log_box()
        self.status_log.setMaximumHeight(220)
        layout.addWidget(self.status_log)
        self._route_changed()

    def _route_changed(self) -> None:
        rates = self.route_combo.currentData() == "rates"
        self.rates_box.setVisible(rates)
        self.global_box.setVisible(not rates)
        if rates:
            self._refresh_modes()

    def _refresh_modes(self) -> None:
        current = self.mode_combo.currentData()
        self.mode_combo.clear()
        for i, row in enumerate(self.motion._mode_rows(), 1):
            self.mode_combo.addItem(f"{i}: {row['kind'].replace('_', ' ')}", i)
        j = self.mode_combo.findData(current)
        self.mode_combo.setCurrentIndex(max(j, 0))

    # ----------------------------------------------------------- parameters
    def _t_ref(self):
        v = _float(self.tref_edit.text())
        if v is not None and v <= 0:
            raise ValueError("T_ref is a temperature in kelvin and must be positive.")
        return v

    def refresh_parameters(self) -> bool:
        """Rebuild the parameter table from the model above (keeping the choices made for
        parameters that are still there)."""
        try:
            model = self.motion.build_model()
            T_ref = self._t_ref()
        except (ValueError, KeyError) as exc:
            QMessageBox.warning(self, "Fit anisotropic motion", str(exc))
            return False
        kept = {}
        for r in range(self.table.rowCount()):
            name = self.table.item(r, _NAME).data(Qt.ItemDataRole.UserRole)
            kept[name] = (self.table.item(r, _FIT).checkState() == Qt.CheckState.Checked,
                          self.table.item(r, _MIN).text(), self.table.item(r, _MAX).text())
        rows = F.fit_parameter_list(model, T_ref)
        self.table.blockSignals(True)
        self.table.setRowCount(0)
        for p in rows:
            r = self.table.rowCount()
            self.table.insertRow(r)
            check = QTableWidgetItem()
            check.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable)
            on, lo, hi = kept.get(p["name"], (p["default_free"], _num(p["min"]), _num(p["max"])))
            check.setCheckState(Qt.CheckState.Checked if on else Qt.CheckState.Unchecked)
            label = QTableWidgetItem(p["label"])
            label.setFlags(Qt.ItemFlag.ItemIsEnabled)
            label.setData(Qt.ItemDataRole.UserRole, p["name"])
            self.table.setItem(r, _FIT, check)
            self.table.setItem(r, _NAME, label)
            self.table.setItem(r, _START, QTableWidgetItem(_num(p["value"])))
            self.table.setItem(r, _MIN, QTableWidgetItem(lo))
            self.table.setItem(r, _MAX, QTableWidgetItem(hi))
            for c in (_FITTED, _ERR):
                item = QTableWidgetItem("")
                item.setFlags(Qt.ItemFlag.ItemIsEnabled)
                self.table.setItem(r, c, item)
        self.table.blockSignals(False)
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setSectionResizeMode(_NAME, QHeaderView.ResizeMode.Stretch)
        return True

    def _parameters(self) -> dict:
        if not self.table.rowCount():
            self.refresh_parameters()
        out = {}
        for r in range(self.table.rowCount()):
            if self.table.item(r, _FIT).checkState() != Qt.CheckState.Checked:
                continue
            name = self.table.item(r, _NAME).data(Qt.ItemDataRole.UserRole)
            value = _float(self.table.item(r, _START).text())
            if value is None:
                raise ValueError(f"{name}: the start value must be a number.")
            lo = _float(self.table.item(r, _MIN).text(), -math.inf)
            hi = _float(self.table.item(r, _MAX).text(), math.inf)
            out[name] = {"value": value, "min": lo, "max": hi}
        if not out:
            raise ValueError("Check at least one parameter to fit.")
        return out

    # --------------------------------------------------------------- the fit
    def _inputs(self):
        """Everything a fit needs, read now (the job must not touch widgets)."""
        spectra = self.motion.get_spectra() or []
        if not spectra:
            raise ValueError("Select one or more 1D spectra in Open experiments.")
        model = self.motion.build_model()
        kw = self.motion._kwargs(model, "T")
        kw.pop("T_K", None)
        if spectra[0].get("larmor_MHz"):
            kw["larmor_Hz"] = float(spectra[0]["larmor_MHz"]) * 1e6
        window = None
        if self.window_check.isChecked() and not self.motion.auto_axis_check.isChecked():
            window = (self.motion.axis_lo.value(), self.motion.axis_hi.value())
        items = [{"ppm": s["ppm"], "y": s["y"], "T_K": s.get("T_K"), "label": s.get("label")}
                 for s in spectra]
        return model, items, kw, window, (spectra[0].get("nucleus") or self.motion.nucleus())

    def run_fit(self) -> bool:
        route = self.route_combo.currentData()
        try:
            model, spectra, kw, window, nucleus = self._inputs()
            T_ref = self._t_ref()
            if route == "global":
                parameters = self._parameters()
                n_starts = self.starts_spin.value()
            else:
                mode = self.mode_combo.currentData()
                if mode is None:
                    raise ValueError("Add a rotational mode to fit its rate.")
                k_lo, k_hi = _float(self.kmin_edit.text()), _float(self.kmax_edit.text())
                if not (k_lo and k_hi and 0 < k_lo < k_hi):
                    raise ValueError("The scan range must be two positive rates, lowest first.")
        except (ValueError, KeyError) as exc:
            QMessageBox.warning(self, "Fit anisotropic motion", str(exc))
            return False
        n_points = self.points_spin.value()
        fit_offset = self.offset_check.isChecked()
        larmor_MHz = kw["larmor_Hz"] / 1e6

        def compute():
            if route == "global":
                return F.fit_motion(model, spectra, parameters, T_ref_K=T_ref,
                                    fit_offset=fit_offset, n_points=n_points, n_starts=n_starts,
                                    window=window, **kw)
            return F.fit_motion_rates(model, spectra, mode=mode, k_range=(k_lo, k_hi),
                                      fit_offset=fit_offset, n_points=n_points, window=window,
                                      **kw)

        def apply(fit, _stopped):
            self.fit = fit
            self._show_fit(fit, route, nucleus, larmor_MHz)

        from . import jobs

        def fail(exc):
            QMessageBox.warning(self, "Fit anisotropic motion", str(exc))

        title = "Anisotropic motion fit" if route == "global" else "Anisotropic motion rates"
        return jobs.submit(self, jobs.Job(title, compute, apply, fail=fail, watched=True,
                                          keepable=(route == "global")))

    def _show_fit(self, fit, route, nucleus, larmor_MHz) -> None:
        self.status_log.appendPlainText(fit["log"])
        if self.canvas is not None:
            size = _text_size(self.canvas)
            _show(self.canvas,
                  lambda fig: F.plot_motion_fit(fit, fig=fig, nucleus=nucleus, text_size=size),
                  "compare", hz_per_ppm=larmor_MHz)
            self.canvas.plot_owner = "motion_fit"
            self.show_view()
        if route == "global":
            self._fill_table(fit)
            bits = [f"{n} = {fit['params'][n]:.5g} ± {fit['errors'][n]:.2g}" for n in fit["free"]]
            for key, (v, e) in fit["derived"].items():
                bits.append(f"{key} = {v:.4g} ± {e:.2g}")
            self.results_label.setText(
                ("Fitted: " if fit["success"] else "Not converged: ") + "; ".join(bits)
                + f". Reduced χ² {fit['redchi']:.3g}.")
            self.use_button.setEnabled(True)
        else:
            ar = fit["arrhenius"]
            n = int(np.sum(fit["determined"]))
            self.results_label.setText(
                f"{n} of {len(fit['k_Hz'])} rates determined. "
                + (f"Arrhenius: Ea = {ar['Ea_eV']:.4g} ± {ar['Ea_err_eV']:.2g} eV, "
                   f"log10 ν₀ = {ar['log10_nu0']:.3g} ± {ar['log10_nu0_err']:.2g}."
                   if ar else "No Arrhenius line (fewer than two determined rates, or no "
                   "temperatures)."))
            self.use_button.setEnabled(ar is not None and fit["mode"] <= self.mode_combo.count())
        try:
            self.on_run()
        except Exception:  # noqa: BLE001 -- the journal never breaks a fit
            pass

    def _fill_table(self, fit) -> None:
        for r in range(self.table.rowCount()):
            name = self.table.item(r, _NAME).data(Qt.ItemDataRole.UserRole)
            if name in fit["params"]:
                self.table.item(r, _FITTED).setText(_num(fit["params"][name]))
                self.table.item(r, _ERR).setText(f"{fit['errors'][name]:.2g}")
            else:
                self.table.item(r, _FITTED).setText("")
                self.table.item(r, _ERR).setText("")

    # -------------------------------------------------------- use the values
    def use_fitted_values(self) -> bool:
        fit = self.fit
        if fit is None:
            return False
        if "arrhenius" in fit:                                  # the rate route
            ar = fit["arrhenius"]
            if ar is None:
                return False
            i = fit["mode"]
            row = self.motion._mode_rows()[i - 1]
            self.motion._set_row_from(self.motion.modes_table, i - 1, 3, [
                "", f"{ar['Ea_eV']:.6g}", f"{10 ** ar['log10_nu0']:.6g}"])
            self.status_log.appendPlainText(
                f"Mode {i} ({row['kind'].replace('_', ' ')}) now has Ea = {ar['Ea_eV']:.5g} eV and "
                f"ν₀ = {10 ** ar['log10_nu0']:.4g} s⁻¹ from the Arrhenius line.")
            self.motion._refresh_sweep_choices()
            return True
        done = self.motion.apply_fitted_model(fit["model"].to_dict(), fit["free"])
        self.status_log.appendPlainText("Written to the tables above: " + ", ".join(done) + ".")
        self.refresh_parameters()
        return True

    # ---------------------------------------------------------- notebook code
    def notebook_code(self) -> str:
        route = self.route_combo.currentData()
        model, _spectra, kw, _window, nucleus = self._inputs_for_code()
        parameters = self._parameters() if route == "global" else {}
        kwargs = {k: v for k, v in kw.items()}
        return F.motion_fit_code(model, parameters, route=route,
                                 mode=self.mode_combo.currentData() or 1,
                                 T_ref_K=self._t_ref(), kwargs=kwargs,
                                 n_starts=self.starts_spin.value(), nucleus=nucleus)

    def _inputs_for_code(self):
        model = self.motion.build_model()
        kw = self.motion._kwargs(model, "T")
        kw.pop("T_K", None)
        return model, [], kw, None, self.motion.nucleus()

    def copy_notebook_code(self) -> None:
        try:
            code = self.notebook_code()
        except (ValueError, KeyError) as exc:
            QMessageBox.warning(self, "Copy notebook code", str(exc))
            return
        _copy(code)
        self.status_log.appendPlainText("Copied notebook code to the clipboard.")

    # ------------------------------------------------------------------ state
    def get_state(self) -> dict:
        rows = []
        for r in range(self.table.rowCount()):
            rows.append({"name": self.table.item(r, _NAME).data(Qt.ItemDataRole.UserRole),
                         "fit": self.table.item(r, _FIT).checkState() == Qt.CheckState.Checked,
                         "start": self.table.item(r, _START).text(),
                         "min": self.table.item(r, _MIN).text(),
                         "max": self.table.item(r, _MAX).text()})
        checked = [f"{x['name']}: start {x['start']}, bounds {x['min'] or '-∞'} to "
                   f"{x['max'] or '∞'}" for x in rows if x["fit"]]
        return {"route": self.route_combo.currentData(), "mode": self.mode_combo.currentData(),
                "k_range": [self.kmin_edit.text(), self.kmax_edit.text()],
                "t_ref": self.tref_edit.text(), "starts": self.starts_spin.value(),
                "n_points": self.points_spin.value(), "offset": self.offset_check.isChecked(),
                "window": self.window_check.isChecked(), "parameters": rows,
                "extra": {f"Fitted parameter {k + 1}": text for k, text in enumerate(checked)}}

    def set_state(self, state: dict) -> None:
        state = state or {}
        if "route" in state:
            self.route_combo.setCurrentIndex(max(self.route_combo.findData(state["route"]), 0))
        self._refresh_modes()
        if state.get("mode") is not None:
            self.mode_combo.setCurrentIndex(max(self.mode_combo.findData(state["mode"]), 0))
        if "k_range" in state:
            self.kmin_edit.setText(state["k_range"][0])
            self.kmax_edit.setText(state["k_range"][1])
        self.tref_edit.setText(state.get("t_ref", ""))
        self.starts_spin.setValue(int(state.get("starts", 0)))
        self.points_spin.setValue(int(state.get("n_points", 1024)))
        self.offset_check.setChecked(bool(state.get("offset", True)))
        self.window_check.setChecked(bool(state.get("window", False)))
        if state.get("parameters"):
            if not self.table.rowCount():
                self.refresh_parameters()
            by_name = {x["name"]: x for x in state["parameters"]}
            self.table.blockSignals(True)
            for r in range(self.table.rowCount()):
                x = by_name.get(self.table.item(r, _NAME).data(Qt.ItemDataRole.UserRole))
                if x is None:
                    continue
                self.table.item(r, _FIT).setCheckState(
                    Qt.CheckState.Checked if x["fit"] else Qt.CheckState.Unchecked)
                self.table.item(r, _START).setText(x["start"])
                self.table.item(r, _MIN).setText(x["min"])
                self.table.item(r, _MAX).setText(x["max"])
            self.table.blockSignals(False)
