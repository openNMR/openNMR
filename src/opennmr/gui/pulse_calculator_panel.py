"""Simulate ▸ Pulse calculator: pulse lengths, RF powers, nutation
frequencies and bandwidths (explicit user request, 2026-11-13).

A calibrated reference pulse (its length, flip angle and power), then one
of: the pulse lengths at another power; the power for a nutation
frequency; the power for a pulse length. Powers in W, dB (TopSpin's: 0 dB
= 1 W unless changed, higher dB = less power) or "scaled" units (W times a
factor typed in, or left blank for relative values). All the arithmetic is
:mod:`opennmr.notebook.pulse_calculator`, so a notebook gets the same
numbers; the working is shown step by step and can be copied.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout, QLineEdit, QPushButton,
    QVBoxLayout, QWidget,
)

from ..notebook import pulse_calculator as pc
from .widgets.report_label import ReportLabel

#: What to work out: (label, key).
MODES = (("Pulse lengths at another power", "power"),
         ("Power for a nutation frequency", "nu1"),
         ("Power for a pulse length", "pulse"))

_UNIT_ITEMS = (("W", "W"), ("dB", "dB"), ("scaled units", "scaled"))


def _spin(minimum, maximum, value, decimals, suffix, tip) -> QDoubleSpinBox:
    spin = QDoubleSpinBox()
    spin.setRange(minimum, maximum)
    spin.setDecimals(decimals)
    spin.setValue(value)
    spin.setSuffix(suffix)
    spin.setToolTip(tip)
    return spin


def _unit_combo(tip: str) -> QComboBox:
    combo = QComboBox()
    for label, key in _UNIT_ITEMS:
        combo.addItem(label, key)
    combo.setToolTip(tip)
    return combo


def _optional_float(text: str, what: str) -> float | None:
    text = (text or "").strip().replace("−", "-").replace(",", ".")
    if not text:
        return None
    try:
        return float(text)
    except ValueError as exc:
        raise ValueError(f"{what} must be a number, or left blank.") from exc


class PulseCalculatorPanel(QWidget):
    """The calculator's controls and read-out. ``draw_plot(draw, kind)``
    shows the excitation profile on the Simulation plot."""

    POWER_TIP = ("W; dB as TopSpin uses it (PLdB = −10·log10(PLW / W): higher dB is less "
                 "power, 0 dB is 1 W unless set below); or scaled units – W times the "
                 "scaling factor below (leave the factor blank and scaled values are only "
                 "compared with each other).")

    def __init__(self, draw_plot: Callable | None = None, parent=None) -> None:
        super().__init__(parent)
        self._draw_plot = draw_plot
        self.result: dict | None = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        ref_box = QGroupBox("Calibrated pulse (reference)")
        ref_form = QFormLayout(ref_box)
        self.pc_ref_pulse_spin = _spin(0.001, 1e6, 2.5, 4, " µs",
                                       "The length of a pulse you have calibrated.")
        ref_form.addRow("Pulse length:", self.pc_ref_pulse_spin)
        self.pc_ref_flip_spin = _spin(0.1, 3600.0, 90.0, 2, " °",
                                      "Its flip angle on resonance (90° for a calibrated 90°).")
        ref_form.addRow("Flip angle:", self.pc_ref_flip_spin)
        power_row = QWidget()
        power_layout = QHBoxLayout(power_row)
        power_layout.setContentsMargins(0, 0, 0, 0)
        self.pc_ref_power_spin = _spin(-200.0, 1e7, 100.0, 4, "",
                                       "The power the pulse was calibrated at.")
        self.pc_ref_unit_combo = _unit_combo(self.POWER_TIP)
        power_layout.addWidget(self.pc_ref_power_spin, 1)
        power_layout.addWidget(self.pc_ref_unit_combo)
        ref_form.addRow("Power:", power_row)
        layout.addWidget(ref_box)

        units_box = QGroupBox("Power units")
        units_form = QFormLayout(units_box)
        self.pc_scale_edit = QLineEdit()
        self.pc_scale_edit.setPlaceholderText("blank: scaled values are relative")
        self.pc_scale_edit.setToolTip(
            "Watts per scaled unit: a scaled power P means P × this many W. Leave blank when "
            "it is not known – scaled powers then convert among themselves (only their ratio "
            "matters), but not to W or dB.")
        units_form.addRow("Scaling factor (W per unit):", self.pc_scale_edit)
        self.pc_db_ref_spin = _spin(1e-9, 1e7, 1.0, 6, " W",
                                    "What 0 dB is, in W. TopSpin: 1 W (PLdB = −10·log10(PLW / W)).")
        units_form.addRow("0 dB is:", self.pc_db_ref_spin)
        layout.addWidget(units_box)

        target_box = QGroupBox("Calculate")
        target_form = QFormLayout(target_box)
        self.pc_mode_combo = QComboBox()
        for label, key in MODES:
            self.pc_mode_combo.addItem(label, key)
        self.pc_mode_combo.setToolTip(
            "At another power: ν₁, pulse lengths and bandwidth there.\n"
            "For a nutation frequency: the power that gives it.\n"
            "For a pulse length: the power at which a pulse of the flip angle below is that "
            "long.")
        target_form.addRow("What:", self.pc_mode_combo)
        self._power_row = QWidget()
        target_power_layout = QHBoxLayout(self._power_row)
        target_power_layout.setContentsMargins(0, 0, 0, 0)
        self.pc_power_spin = _spin(-200.0, 1e7, 25.0, 4, "", "The power to convert the pulse to.")
        self.pc_unit_combo = _unit_combo(self.POWER_TIP)
        target_power_layout.addWidget(self.pc_power_spin, 1)
        target_power_layout.addWidget(self.pc_unit_combo)
        target_form.addRow("Power:", self._power_row)
        self.pc_nu1_spin = _spin(0.001, 1e6, 100.0, 4, " kHz", "The nutation frequency ν₁ wanted.")
        target_form.addRow("ν₁:", self.pc_nu1_spin)
        self.pc_target_pulse_spin = _spin(0.001, 1e6, 5.0, 4, " µs", "The pulse length wanted.")
        target_form.addRow("Pulse length:", self.pc_target_pulse_spin)
        self.pc_target_flip_spin = _spin(0.1, 3600.0, 90.0, 2, " °",
                                         "The flip angle of the target pulse; its length is "
                                         "listed alongside the 90° and 180°.")
        target_form.addRow("Flip angle:", self.pc_target_flip_spin)
        self.pc_nucleus_edit = QLineEdit()
        self.pc_nucleus_edit.setPlaceholderText("optional, e.g. 1H")
        self.pc_nucleus_edit.setToolTip("Give the nucleus to have the RF field B₁ in mT.")
        target_form.addRow("Nucleus:", self.pc_nucleus_edit)
        self.pc_spin_edit = QLineEdit()
        self.pc_spin_edit.setPlaceholderText("optional, e.g. 2.5 for 27Al")
        self.pc_spin_edit.setToolTip(
            "A half-integer quadrupolar nucleus's spin I: adds the central-transition-"
            "selective pulse lengths, (I + ½) times shorter (for ν_Q ≫ ν₁).")
        target_form.addRow("Spin I (CT-selective):", self.pc_spin_edit)
        layout.addWidget(target_box)
        self._target_form = target_form

        buttons = QHBoxLayout()
        self.pc_calculate_button = QPushButton("Calculate")
        self.pc_calculate_button.clicked.connect(lambda: self.calculate())
        buttons.addWidget(self.pc_calculate_button)
        self.pc_profile_button = QPushButton("Show Excitation Profile")
        self.pc_profile_button.setToolTip(
            "Plot the exact excitation profile of the 90° pulse (and the target flip) against "
            "resonance offset on the Simulation plot.")
        self.pc_profile_button.clicked.connect(lambda: self.show_profile())
        self.pc_profile_button.setEnabled(draw_plot is not None)
        buttons.addWidget(self.pc_profile_button)
        self.pc_copy_button = QPushButton("Copy")
        self.pc_copy_button.setToolTip("Copy the working and results.")
        self.pc_copy_button.clicked.connect(self.copy_report)
        buttons.addWidget(self.pc_copy_button)
        buttons.addStretch(1)
        layout.addLayout(buttons)
        self.pc_results_label = ReportLabel("")
        layout.addWidget(self.pc_results_label)

        self.pc_mode_combo.currentIndexChanged.connect(lambda _i: self._sync_mode())
        self._sync_mode()

    # ---- inputs -----------------------------------------------------

    def mode(self) -> str:
        return self.pc_mode_combo.currentData()

    def _sync_mode(self) -> None:
        mode = self.mode()
        self._target_form.setRowVisible(self._power_row, mode == "power")
        self._target_form.setRowVisible(self.pc_nu1_spin, mode == "nu1")
        self._target_form.setRowVisible(self.pc_target_pulse_spin, mode == "pulse")

    def kwargs(self) -> dict:
        """The :func:`opennmr.notebook.pulse_calculator.pulse_calculator`
        arguments the controls describe."""
        kwargs = {
            "ref_pulse_us": self.pc_ref_pulse_spin.value(),
            "ref_flip_deg": self.pc_ref_flip_spin.value(),
            "ref_power": self.pc_ref_power_spin.value(),
            "ref_unit": self.pc_ref_unit_combo.currentData(),
            "target_flip_deg": self.pc_target_flip_spin.value(),
            "scale": _optional_float(self.pc_scale_edit.text(), "The scaling factor"),
            "db_ref_w": self.pc_db_ref_spin.value(),
            "nucleus": self.pc_nucleus_edit.text().strip() or None,
            "spin": _optional_float(self.pc_spin_edit.text(), "Spin I"),
        }
        mode = self.mode()
        if mode == "power":
            kwargs["power"] = self.pc_power_spin.value()
            kwargs["unit"] = self.pc_unit_combo.currentData()
        elif mode == "nu1":
            kwargs["nu1_hz"] = self.pc_nu1_spin.value() * 1e3
        else:
            kwargs["target_pulse_us"] = self.pc_target_pulse_spin.value()
        return kwargs

    # ---- actions ----------------------------------------------------

    def calculate(self) -> dict | None:
        try:
            self.result = pc.pulse_calculator(**self.kwargs())
        except ValueError as exc:
            self.result = None
            self.pc_results_label.setText(f"⚠ {exc}")
            return None
        self.pc_results_label.setText(self.result["report"])
        return self.result

    def copy_report(self) -> None:
        text = self.pc_results_label.text()
        if text:
            QGuiApplication.clipboard().setText(text)

    def show_profile(self) -> None:
        result = self.result or self.calculate()
        if result is None or self._draw_plot is None:
            return
        nu1 = result["nu1_hz"]
        flips = sorted({90.0, float(self.pc_target_flip_spin.value())})
        tau90 = result["pulse_us"][90.0]
        span = 4.0 / (tau90 * 1e-6) + 3.0 * nu1
        offsets = np.linspace(-span, span, 2001)

        def draw(figure):
            ax = figure.add_subplot(111)
            for flip in flips:
                tau = result["pulse_us"][flip]
                profile = pc.excitation_profile(offsets, nu1, tau)
                if flip <= 135.0:
                    ax.plot(offsets / 1e3, profile["mxy"], label=f"{flip:g}°: $|M_{{xy}}|$")
                else:
                    ax.plot(offsets / 1e3, (1.0 - profile["mz"]) / 2.0,
                            label=f"{flip:g}°: $(1 - M_z)/2$")
            ax.set_xlabel(r"Offset, $\Delta\nu$ / kHz")
            ax.set_ylabel("Fraction of $M_0$")
            ax.set_title(rf"Rectangular pulse, $\nu_1$ = {nu1 / 1e3:.4g} kHz")
            ax.set_ylim(-0.02, 1.05)
            ax.legend(frameon=False)
            figure.tight_layout()

        self._draw_plot(draw, "curve")
