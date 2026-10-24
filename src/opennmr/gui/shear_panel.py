"""
Advanced Processing > Shearing and Scaling (2D): the notebook's
:mod:`opennmr.notebook.shearing` as a panel.

The user picks the experiment (MQMAS, STMAS, DAS, J-resolved, PASS/MAT
or custom numbers); the panel shows, before anything is done, the shear
ratios it will use and what each axis of the result will report
(``ShearPlan.describe``). Because the sign of the ratio depends on how
F1 was detected, *Check against the data* measures the ratio from the
spectrum itself and sets the F1 direction from it. *Preview* draws the
result here, with the expected spinning-sideband positions marked;
*Shear → New Spectrum* adds it to Open experiments with the full log
in its history.

All the maths is in :mod:`opennmr.notebook.shearing`; this module only
collects settings and shows results.
"""

from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
    QPlainTextEdit, QPushButton, QVBoxLayout, QWidget,
)

from ..notebook import shearing as nb_shear

#: (key, combo text) in display order.
EXPERIMENT_CHOICES = [
    ("mqmas", "MQMAS (quadrupolar, MAS)"),
    ("stmas", "STMAS (quadrupolar, MAS)"),
    ("das", "DAS (quadrupolar, two angles)"),
    ("jres", "2D J-resolved, homonuclear (solution)"),
    ("pass", "2D PASS sideband separation (MAS)"),
    ("mat", "Magic-angle turning / hopping (MAT)"),
    ("shifting_p", "Shifting-p echo (static, shift / quadrupole)"),
    ("shifting_d", "Shifting-d echo, incl. adiabatic (static)"),
    ("custom_kappa", "Custom: shear ratios"),
    ("custom_coefficients", "Custom: pathway coefficients a, b"),
]

CONVENTION_CHOICES = [
    ("cs", "Chemical-shift scale (δF1 = δCS − 10/17 δQIS)"),
    ("unified", "Unified, 1 + |κ| (Grandinetti)"),
    ("none", "None (sheared Hz)"),
    ("custom", "Custom factor"),
]

SPINS = [1.5, 2.5, 3.5, 4.5]


def shear_source_problem(spectrum) -> str | None:
    """Why ``spectrum`` cannot be sheared, or ``None`` if it can."""
    if spectrum is None:
        return "Select a 2D spectrum in Open experiments."
    if spectrum.ndim != 2:
        return "Shearing needs a 2D spectrum; the selected one is 1D."
    if spectrum.metadata.pseudo_2d or spectrum.axes[0].kind != "frequency":
        return "The selected spectrum is a pseudo-2D series, whose F1 is not a frequency."
    return None


def _spin(minimum=-1e4, maximum=1e4, decimals=6, value=0.0, step=0.01, tip=""):
    w = QDoubleSpinBox()
    w.setRange(minimum, maximum)
    w.setDecimals(decimals)
    w.setSingleStep(step)
    w.setValue(value)
    if tip:
        w.setToolTip(tip)
    return w


class ShearPanel(QWidget):
    """Settings, live plan readout, data check, preview and Shear.

    ``set_source(spectrum, label)`` follows the Open experiments
    selection; ``result_ready(NMRSpectrum, label, log)`` is emitted with
    the sheared spectrum."""

    result_ready = Signal(object, str, str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._source = None
        self._source_label = ""
        self._scan = None
        self._plan = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        intro = QLabel(
            "Shears and scales a 2D so that an anisotropic ridge lies parallel to an axis "
            "(MQMAS, STMAS, DAS), or tilts a J-resolved / sideband spectrum by 45°. "
            "Choose the experiment; what the result's axes will mean is shown before "
            "anything is done. Check the ratio against the data if unsure of its sign.")
        intro.setWordWrap(True)
        layout.addWidget(intro)
        self.source_label = QLabel("")
        self.source_label.setWordWrap(True)
        layout.addWidget(self.source_label)

        # --- experiment ----------------------------------------------------
        exp_box = QGroupBox("Experiment")
        exp_layout = QVBoxLayout(exp_box)
        form = QFormLayout()
        self.experiment_combo = QComboBox()
        for key, text in EXPERIMENT_CHOICES:
            self.experiment_combo.addItem(text, key)
            self.experiment_combo.setItemData(self.experiment_combo.count() - 1,
                                              nb_shear.EXPERIMENTS[key], 3)   # tooltip role
        form.addRow("Experiment:", self.experiment_combo)
        exp_layout.addLayout(form)

        # quadrupolar spin (MQMAS / STMAS / DAS)
        self.quad_box = QWidget()
        quad_form = QFormLayout(self.quad_box)
        quad_form.setContentsMargins(0, 0, 0, 0)
        self.spin_combo = QComboBox()
        for s in SPINS:
            self.spin_combo.addItem(f"I = {nb_shear._frac(s)}", s)
        self.spin_combo.setToolTip("Set from the selected spectrum's nucleus when it is quadrupolar.")
        quad_form.addRow("Spin:", self.spin_combo)
        self.order_combo = QComboBox()
        self.order_combo.setToolTip("|p1|, the multiple-quantum order evolving in t1.")
        quad_form.addRow("MQ order:", self.order_combo)
        self.pathway_combo = QComboBox()
        self.pathway_combo.addItem("Echo (refocuses the anisotropy)", "echo")
        self.pathway_combo.addItem("Antiecho", "antiecho")
        quad_form.addRow("Pathway:", self.pathway_combo)
        self.satellite_combo = QComboBox()
        quad_form.addRow("Satellite:", self.satellite_combo)
        self.das_theta1 = _spin(0, 90, 3, 37.38, 0.01, "Rotor angle during t1 (degrees).")
        self.das_theta2 = _spin(0, 90, 3, 79.19, 0.01, "Rotor angle during t2 (degrees).")
        das_row = QHBoxLayout()
        das_row.addWidget(self.das_theta1)
        das_row.addWidget(QLabel("/"))
        das_row.addWidget(self.das_theta2)
        quad_form.addRow("DAS angles (°):", das_row)
        self.split_t1_check = QCheckBox("Acquired with split-t1 (already sheared)")
        self.split_t1_check.setToolTip(
            "Split-t1 sequences share t1 between the MQ (or satellite) coherence and the CT "
            "so the anisotropy refocuses within t1: the data arrive sheared, in the unified "
            "scaling. Ticked, nothing is sheared; F1 is only scaled and referenced. Assumes "
            "the F1 dwell (IN0) is the whole t1 increment.")
        quad_form.addRow(self.split_t1_check)
        self.second_shear_check = QCheckBox("Second shear: centre every F2 pattern on zero")
        self.second_shear_check.setToolTip(
            "Also shear parallel to t1, removing the isotropic part from F2, so each "
            "site's anisotropic pattern is centred at zero in F2.")
        quad_form.addRow(self.second_shear_check)
        exp_layout.addWidget(self.quad_box)
        self._quad_form = quad_form

        # custom ratios
        self.kappa_box = QWidget()
        kappa_form = QFormLayout(self.kappa_box)
        kappa_form.setContentsMargins(0, 0, 0, 0)
        self.kappa_t2_spin = _spin(tip="Shear parallel to t2: (ν1, ν2) → (ν1 − κ ν2, ν2).")
        self.kappa_t1_spin = _spin(tip="Shear parallel to t1: (ν1, ν2) → (ν1, ν2 − κ ν1).")
        self.cs1_spin = _spin(value=1.0, tip=(
            "Chemical-shift coefficient of the t1 coherence relative to t2 (p1/p2 for a "
            "chemical shift). Used only to reference F1; 0 for a J or sideband F1."))
        kappa_form.addRow("κ (shear ∥ t2):", self.kappa_t2_spin)
        kappa_form.addRow("κ (shear ∥ t1):", self.kappa_t1_spin)
        kappa_form.addRow("CS coefficient in t1:", self.cs1_spin)
        exp_layout.addWidget(self.kappa_box)

        # custom coefficients
        self.coef_box = QWidget()
        coef_form = QFormLayout(self.coef_box)
        coef_form.setContentsMargins(0, 0, 0, 0)
        self.use_a_check = QCheckBox("Remove a from F1 (shear ∥ t2)")
        self.use_a_check.setChecked(True)
        self.a1_spin, self.a2_spin = _spin(value=-7.0), _spin(value=9.0)
        self.use_b_check = QCheckBox("Remove b from F2 (shear ∥ t1)")
        self.b1_spin, self.b2_spin = _spin(value=1.0), _spin(value=1.0)
        self.c1_spin, self.c2_spin = _spin(value=1.0), _spin(value=1.0)
        for check, s1, s2, name in ((self.use_a_check, self.a1_spin, self.a2_spin, "a"),
                                    (self.use_b_check, self.b1_spin, self.b2_spin, "b")):
            row = QHBoxLayout()
            row.addWidget(QLabel(f"{name}[1]"))
            row.addWidget(s1)
            row.addWidget(QLabel(f"{name}[2]"))
            row.addWidget(s2)
            coef_form.addRow(check)
            coef_form.addRow(row)
        crow = QHBoxLayout()
        crow.addWidget(QLabel("c[1]"))
        crow.addWidget(self.c1_spin)
        crow.addWidget(QLabel("c[2]"))
        crow.addWidget(self.c2_spin)
        coef_form.addRow(QLabel("Chemical-shift coefficients (referencing only):"))
        coef_form.addRow(crow)
        self.coef_box.setToolTip(
            "Frequency in each period = a·Ω_a + b·Ω_b + c·δ_CS + ...; the echo line of "
            "component a is a[1] t1 + a[2] t2 = 0, so κ(t2) = a[1]/a[2] (Grandinetti 2011, "
            "Eq. 196). Only ratios matter.")
        exp_layout.addWidget(self.coef_box)

        # F1 direction and the data check
        dir_form = QFormLayout()
        self.f1_sign_combo = QComboBox()
        self.f1_sign_combo.addItem("Same sense as the t1 coherence (+1)", 1)
        self.f1_sign_combo.addItem("Reversed (−1)", -1)
        self.f1_sign_combo.setToolTip(
            "Whether F1 is displayed with the same sense as the coherence evolving in t1. "
            "It depends on the quadrature scheme and receiver phases, which is why the sign "
            "of κ is often in doubt. Check against the data sets it.")
        dir_form.addRow("F1 direction:", self.f1_sign_combo)
        exp_layout.addLayout(dir_form)
        check_row = QHBoxLayout()
        self.check_button = QPushButton("Check against the data")
        self.check_button.setToolTip(
            "Measure the shear ratio from the selected spectrum (the κ giving the sharpest "
            "projection) and compare it with the standard values; sets the F1 direction "
            "when one matches.")
        self.check_button.clicked.connect(self.check_against_data)
        check_row.addWidget(self.check_button)
        check_row.addStretch(1)
        exp_layout.addLayout(check_row)
        self.check_label = QLabel("")
        self.check_label.setWordWrap(True)
        exp_layout.addWidget(self.check_label)
        layout.addWidget(exp_box)

        # --- scaling, timing, sidebands -------------------------------------
        opt_box = QGroupBox("Scaling and options")
        opt_form = QFormLayout(opt_box)
        self.convention_combo = QComboBox()
        for key, text in CONVENTION_CHOICES:
            self.convention_combo.addItem(text, key)
            if key in nb_shear.SCALE_CONVENTIONS:
                self.convention_combo.setItemData(self.convention_combo.count() - 1,
                                                  nb_shear.SCALE_CONVENTIONS[key], 3)
        opt_form.addRow("F1 scaling:", self.convention_combo)
        self.custom_scale_spin = _spin(0.001, 1000, 6, 1.0, 0.1, "Divide F1 by this.")
        opt_form.addRow("Custom F1 factor:", self.custom_scale_spin)
        self.t1_0_spin = _spin(-1e6, 1e6, 3, 0.0, 1.0,
                               "Time of the first t1 point (µs), if the first increment is not "
                               "zero. Ignoring it leaves an F1 phase that changes across F2.")
        opt_form.addRow("First t1 point (µs):", self.t1_0_spin)
        self.mas_spin = _spin(0, 1e6, 1, 0.0, 100.0,
                              "MAS rate (Hz) for marking where spinning sidebands will land "
                              "(read from the experiment when recorded).")
        opt_form.addRow("MAS rate (Hz):", self.mas_spin)
        self.sideband_check = QCheckBox("Mark expected sideband positions in the preview")
        self.sideband_check.setChecked(True)
        opt_form.addRow(self.sideband_check)
        layout.addWidget(opt_box)

        self.plan_edit = QPlainTextEdit()
        self.plan_edit.setReadOnly(True)
        self.plan_edit.setMinimumHeight(120)
        self.plan_edit.setToolTip("What the shear will do and what the result's axes report.")
        layout.addWidget(self.plan_edit)

        buttons = QHBoxLayout()
        self.preview_button = QPushButton("Preview")
        self.preview_button.clicked.connect(self.preview)
        self.shear_button = QPushButton("Shear → New Spectrum")
        self.shear_button.setToolTip("Add the sheared spectrum to Open experiments, with its log.")
        self.shear_button.clicked.connect(self.shear)
        buttons.addWidget(self.preview_button)
        buttons.addWidget(self.shear_button)
        layout.addLayout(buttons)

        self.figure = Figure(figsize=(4.0, 3.0), layout="constrained")
        self.preview_canvas = FigureCanvasQTAgg(self.figure)
        self.preview_canvas.setMinimumHeight(240)
        self.preview_toolbar = NavigationToolbar2QT(self.preview_canvas, self)
        layout.addWidget(self.preview_toolbar)
        layout.addWidget(self.preview_canvas)

        self.log_edit = QPlainTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setMaximumBlockCount(500)
        self.log_edit.setMinimumHeight(80)
        self.log_edit.setPlaceholderText("Warnings and what was done appear here.")
        layout.addWidget(self.log_edit)

        for w in (self.experiment_combo, self.spin_combo, self.order_combo, self.pathway_combo,
                  self.satellite_combo, self.f1_sign_combo, self.convention_combo):
            w.currentIndexChanged.connect(self._on_settings_changed)
        for w in (self.das_theta1, self.das_theta2, self.kappa_t2_spin, self.kappa_t1_spin,
                  self.cs1_spin, self.a1_spin, self.a2_spin, self.b1_spin, self.b2_spin,
                  self.c1_spin, self.c2_spin, self.custom_scale_spin):
            w.valueChanged.connect(self._on_settings_changed)
        for w in (self.second_shear_check, self.split_t1_check, self.use_a_check, self.use_b_check):
            w.toggled.connect(self._on_settings_changed)
        self.spin_combo.currentIndexChanged.connect(self._refill_orders)
        self._refill_orders()
        self.set_source(None)

    # --- public -----------------------------------------------------------
    def set_source(self, spectrum, label: str = "") -> None:
        """Point the panel at the selected spectrum. Its nucleus sets the
        spin (when quadrupolar) and a recorded MAS rate fills the MAS
        field; a sheared result keeps its parent as the source."""
        if (spectrum is not None and "shear" in (spectrum.metadata.raw or {})
                and self._source is not None and shear_source_problem(self._source) is None):
            return
        changed = spectrum is not self._source
        self._source, self._source_label = spectrum, label
        problem = shear_source_problem(spectrum)
        for b in (self.check_button, self.preview_button, self.shear_button):
            b.setEnabled(problem is None)
        if problem:
            self.source_label.setText(problem)
        else:
            nuc = spectrum.metadata.nucleus or "?"
            spin = nb_shear.spin_of(nuc)
            spin_txt = f", I = {nb_shear._frac(spin)}" if spin else ""
            self.source_label.setText(f"Spectrum: {label or spectrum.metadata.source_path} "
                                      f"({nuc}{spin_txt})")
            if changed and spin and spin > 0.5:
                i = self.spin_combo.findData(spin)
                if i >= 0:
                    self.spin_combo.setCurrentIndex(i)
            masr = (spectrum.metadata.raw or {}).get("acqus", {}) or {}
            try:
                rate = float(masr.get("MASR", 0) or 0)
            except (TypeError, ValueError):
                rate = 0.0
            if changed and rate > 0:
                self.mas_spin.setValue(rate)
        if changed:
            self._scan = None
            self.check_label.setText("")
            self.figure.clear()
            self.preview_canvas.draw_idle()
        self._on_settings_changed()

    def experiment(self) -> str:
        return self.experiment_combo.currentData()

    def plan(self) -> nb_shear.ShearPlan:
        """The :class:`ShearPlan` for the current settings (raises
        ``ValueError`` for impossible ones)."""
        exp = self.experiment()
        conv = self.convention_combo.currentData()
        convention = self.custom_scale_spin.value() if conv == "custom" else conv
        if exp in ("jres", "pass", "shifting_d") and conv == "cs":
            convention = "none"
        kwargs = dict(f1_sign=int(self.f1_sign_combo.currentData()), convention=convention)
        if exp in ("mqmas", "stmas", "das"):
            kwargs.update(spin=float(self.spin_combo.currentData()),
                          mq_order=int(self.order_combo.currentData() or 3),
                          pathway=self.pathway_combo.currentData(),
                          satellite=int(self.satellite_combo.currentData() or 1),
                          das_angles=(self.das_theta1.value(), self.das_theta2.value()),
                          second_shear=self.second_shear_check.isChecked(),
                          split_t1=self.split_t1_check.isChecked() and exp in ("mqmas", "stmas"))
        elif exp == "custom_kappa":
            kwargs.update(kappa_t2=self.kappa_t2_spin.value(), kappa_t1=self.kappa_t1_spin.value(),
                          cs=(self.cs1_spin.value(), 1.0))
        elif exp == "custom_coefficients":
            kwargs.update(a=(self.a1_spin.value(), self.a2_spin.value()) if self.use_a_check.isChecked() else None,
                          b=(self.b1_spin.value(), self.b2_spin.value()) if self.use_b_check.isChecked() else None,
                          cs=(self.c1_spin.value(), self.c2_spin.value()))
        return nb_shear.make_shear_plan(exp, **kwargs)

    def check_against_data(self) -> dict | None:
        """Measure the ratio from the data, show the scan, and set the
        F1 direction when a standard value matches."""
        if shear_source_problem(self._source):
            return None
        direction = "t1" if self.experiment() in ("jres", "pass", "mat", "shifting_p", "shifting_d") else "t2"
        spin = float(self.spin_combo.currentData()) if self.experiment() in ("mqmas", "stmas", "das") else None
        try:
            scan = nb_shear.estimate_shear(self._source, direction=direction, spin=spin)
        except ValueError as exc:
            self.check_label.setText(f"Can't measure a ratio: {exc}")
            return None
        self._scan = scan
        text = scan["summary"]
        top = scan["matches"][0] if scan["matches"] else None
        already = (direction == "t2" and abs(scan["kappa"]) < 0.06 and scan["prominence"] >= 0.2
                   and self.experiment() in ("mqmas", "stmas"))
        if already:
            self.split_t1_check.setChecked(True)
            text += "\nSplit-t1 ticked: the data are already sheared."
            top = None
        elif direction == "t2" and self.split_t1_check.isChecked() and abs(scan["kappa"]) > 0.1:
            self.split_t1_check.setChecked(False)
            text += "\nSplit-t1 unticked: the data still need shearing."
        if top is not None and (direction == "t1" or top["relative"] < 0.05):
            self.f1_sign_combo.setCurrentIndex(self.f1_sign_combo.findData(top["f1_sign"]))
            text += f"\nF1 direction set to {'+1' if top['f1_sign'] > 0 else '−1'}."
        try:
            planned = self.plan()
            if direction == "t2" and planned.kappa_t2 and not already and abs(scan["kappa"] - planned.kappa_t2) > 0.05 * abs(planned.kappa_t2):
                text += (f"\nThe chosen experiment's κ = {planned.kappa_t2:+.4f} differs from the "
                         f"measured {scan['kappa']:+.4f}: check the experiment, spin and MQ order.")
        except ValueError:
            pass
        self.check_label.setText(text)
        self._log(text)
        self.figure.clear()
        nb_shear.plot_shear_scan(scan, self.figure.add_subplot(111), fontsize=8)
        self.preview_canvas.draw_idle()
        return scan

    def preview(self):
        """Shear and draw the result here; nothing is added. Returns the
        sheared spectrum, or None."""
        result = self._run()
        if result is None:
            return None
        self._draw(result)
        return result

    def shear(self):
        """Shear and emit the result. Returns it, or None."""
        result = self._run()
        if result is None:
            return None
        plan = self._plan
        name = {"mqmas": "MQMAS", "stmas": "STMAS", "das": "DAS", "jres": "tilted",
                "pass": "PASS", "mat": "MAT", "shifting_p": "shifting-p",
                "shifting_d": "shifting-d"}.get(plan.experiment, "sheared")
        label = f"Sheared ({name}) {self._source_label}".strip()
        log = result.metadata.raw["shear"]["log"]
        self._log(log)
        self._draw(result)
        self.result_ready.emit(result, label, log)
        return result

    # --- internals ----------------------------------------------------------
    def _refill_orders(self) -> None:
        spin = float(self.spin_combo.currentData() or 1.5)
        cur_o = self.order_combo.currentData()
        cur_s = self.satellite_combo.currentData()
        for combo, values, fmt in (
                (self.order_combo, range(3, int(round(2 * spin)) + 1, 2), "{}Q"),
                (self.satellite_combo, range(1, int(round(spin - 0.5)) + 1),
                 "ST{} (±{}/2 ↔ ±{}/2)")):
            combo.blockSignals(True)
            combo.clear()
            for v in values:
                text = fmt.format(v) if "ST" not in fmt else fmt.format(v, 2 * v + 1, 2 * v - 1)
                combo.addItem(text, v)
            cur = cur_o if combo is self.order_combo else cur_s
            i = combo.findData(cur)
            combo.setCurrentIndex(i if i >= 0 else 0)
            combo.blockSignals(False)

    def _on_settings_changed(self, *_args) -> None:
        exp = self.experiment()
        quad = exp in ("mqmas", "stmas", "das")
        self.quad_box.setVisible(quad)
        self.kappa_box.setVisible(exp == "custom_kappa")
        self.coef_box.setVisible(exp == "custom_coefficients")
        rows = {"MQ order:": exp == "mqmas", "Pathway:": exp in ("mqmas", "stmas"),
                "Satellite:": exp == "stmas", "DAS angles (°):": exp == "das"}
        for row in range(self._quad_form.rowCount()):
            label = self._quad_form.itemAt(row, QFormLayout.LabelRole)
            field = self._quad_form.itemAt(row, QFormLayout.FieldRole) or \
                self._quad_form.itemAt(row, QFormLayout.SpanningRole)
            text = label.widget().text() if label is not None and label.widget() else None
            visible = rows.get(text, True)
            if field is not None and field.widget() is self.second_shear_check:
                visible = exp == "das"
            if field is not None and field.widget() is self.split_t1_check:
                visible = exp in ("mqmas", "stmas")
            for item in (label, field):
                if item is None:
                    continue
                if item.widget() is not None:
                    item.widget().setVisible(visible)
                elif item.layout() is not None:
                    for k in range(item.layout().count()):
                        w = item.layout().itemAt(k).widget()
                        if w is not None:
                            w.setVisible(visible)
        self.custom_scale_spin.setEnabled(self.convention_combo.currentData() == "custom")
        try:
            self._plan = self.plan()
            self.plan_edit.setPlainText(self._plan.describe())
            ok = shear_source_problem(self._source) is None
            self.preview_button.setEnabled(ok)
            self.shear_button.setEnabled(ok)
        except ValueError as exc:
            self._plan = None
            self.plan_edit.setPlainText(f"Not possible with these settings: {exc}")
            self.preview_button.setEnabled(False)
            self.shear_button.setEnabled(False)

    def _run(self):
        if shear_source_problem(self._source):
            return None
        try:
            plan = self.plan()
        except ValueError as exc:
            self._log(f"Not possible with these settings: {exc}")
            return None
        self._plan = plan
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")      # they are in the log and shown below
            try:
                result = nb_shear.shear_2d(self._source, plan, t1_0=self.t1_0_spin.value() * 1e-6,
                                           source_label=self._source_label)
            except ValueError as exc:
                self._log(f"Shear failed: {exc}")
                return None
        for w in result.metadata.raw["shear"]["warnings"]:
            self._log(f"Warning: {w}")
        return result

    def _draw(self, result) -> None:
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        data = result.data
        x, y = result.axes[1].ppm, result.axes[0].ppm
        top = float(np.max(data)) or 1.0
        levels = top * np.geomspace(0.03, 0.95, 12)
        ax.contour(x, y, data, levels=levels, linewidths=0.6, cmap="viridis")
        ax.set_xlim(x.max(), x.min())
        ax.set_ylim(y.max(), y.min())
        nuc = result.metadata.nucleus or ""
        ax.set_xlabel(f"δ F2 ({nuc}) / ppm", fontsize=8)
        ax.set_ylabel("δ F1 / ppm" if self._plan and abs(self._plan.cs_after[0]) > 1e-12
                      else "F1 / ppm", fontsize=8)
        ax.tick_params(labelsize=7)
        rate = self.mas_spin.value()
        if self.sideband_check.isChecked() and rate > 0 and self._plan is not None:
            i, j = np.unravel_index(int(np.argmax(data)), data.shape)
            sw = result.metadata.raw["shear"]["sw_after"]
            pos = nb_shear.sideband_positions(self._plan, rate, orders_f1=range(-2, 3),
                                              orders_f2=range(-2, 3), sw=sw)
            sf1 = result.metadata.sfo1_indirect or result.metadata.sfo1
            # one point's width beyond the ends, so the window is SW wide
            step = abs(y[1] - y[0]) if len(y) > 1 else 0.0
            nb_shear.plot_sideband_markers(ax, pos, y[i], x[j], sf1=sf1, sf2=result.metadata.sfo1,
                                           f1_window=(y.min() - step / 2, y.max() + step / 2),
                                           f2_window=(x.min(), x.max()), fontsize=6)
            ax.set_title("+ expected sidebands (k, j) of the tallest peak", fontsize=7)
        self.preview_canvas.draw_idle()

    def _log(self, text: str) -> None:
        self.log_edit.appendPlainText(text)
