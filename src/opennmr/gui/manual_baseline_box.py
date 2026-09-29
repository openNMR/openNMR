"""Process ▸ Baseline ▸ Manual: the controls for a user-defined baseline.

Shown under the Baseline method row when the method is "Manual:
A·x⁴ + B·x³ + C·x² + D·x + E" or "Manual: A·sin(B·x + C) + D". The
widget only holds and reports state; MainWindow stores it in the
spectrum's ``metadata.raw`` and the maths is
:mod:`opennmr.notebook.manual_baseline`, reached through
:func:`opennmr.processing.ft.compute_baseline` -- so the GUI and a
notebook compute the same baseline from the same numbers.

Workflow the layout follows, top to bottom:

1. choose baseline regions -- typed, or dragged on the Processing
   preview with "Pick on plot" -- shaded green there;
2. "Fit to regions" fills in the coefficients (any can be ticked
   "fix" to hold it, e.g. A = B = 0 for a quadratic);
3. edit any coefficient by hand; the dashed overlay follows at once;
4. "Baseline Correct" (above) subtracts it.
"""
from __future__ import annotations

import re

from PySide6.QtCore import Signal
from PySide6.QtGui import QDoubleValidator
from PySide6.QtWidgets import (
    QCheckBox, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget,
)

from ..notebook.manual_baseline import MANUAL_BASELINE_EQUATIONS, MANUAL_BASELINE_PARAMS

#: Longest parameter list (the polynomial's); the sine hides the last row.
_ALL_NAMES = MANUAL_BASELINE_PARAMS["polynomial"]

_NUMBER = re.compile(r"(?<![\d.eE])[-+−]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")

_PARAM_TIPS = {
    "polynomial": {
        "A": "x⁴ coefficient (intensity / ppm⁴).",
        "B": "x³ coefficient (intensity / ppm³).",
        "C": "x² coefficient (intensity / ppm²).",
        "D": "x coefficient: the slope (intensity / ppm).",
        "E": "Constant offset (intensity).",
    },
    "sine": {
        "A": "Amplitude (intensity).",
        "B": "Angular frequency, rad/ppm: the roll's period is 2π/B ppm.",
        "C": "Phase, rad.",
        "D": "Constant offset (intensity).",
    },
}


def parse_regions(text: str) -> list[tuple[float, float]]:
    """``"-50 – -20; 180 to 220"`` -> ``[(-50.0, -20.0), (180.0, 220.0)]``.

    Regions are separated by ``;`` or new lines; each needs exactly two
    numbers, in either order, with any separator between them (``–``,
    ``-``, ``to``, ``,``, space). A minus sign directly after a digit is
    read as a separator, so ``10-20`` is 10 to 20. Raises ``ValueError``
    naming the part it could not read."""
    regions = []
    for part in re.split(r"[;\n]+", text or ""):
        if not part.strip():
            continue
        numbers = [float(m.replace("−", "-")) for m in _NUMBER.findall(part)]
        if len(numbers) != 2:
            raise ValueError(f"could not read “{part.strip()}” as a range: "
                             f"give two numbers, e.g. “180 – 220”")
        lo, hi = sorted(numbers)
        if hi == lo:
            raise ValueError(f"“{part.strip()}” has zero width")
        regions.append((lo, hi))
    return regions


def format_regions(regions) -> str:
    return "; ".join(f"{_short(lo)} – {_short(hi)}" for lo, hi in regions or [])


def _short(value: float) -> str:
    return f"{float(value):.6g}"


class ManualBaselineBox(QWidget):
    """Regions, coefficients and actions for a manual baseline."""

    #: The regions line was edited (already validated).
    regionsEdited = Signal(list)
    #: A coefficient box was edited by hand.
    paramsEdited = Signal(list)
    #: A "fix" box changed.
    fixedChanged = Signal(list)
    refitToggled = Signal(bool)
    fitRequested = Signal()
    pickToggled = Signal(bool)
    clearRequested = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._model = "polynomial"
        self._params_dirty = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.equation_label = QLabel()
        self.equation_label.setWordWrap(True)
        layout.addWidget(self.equation_label)

        layout.addWidget(QLabel("Baseline regions (ppm):"))
        self.regions_edit = QLineEdit()
        self.regions_edit.setPlaceholderText("e.g. −50 – −20; 180 – 220")
        self.regions_edit.setToolTip(
            "Ranges that contain only baseline, no signal. The model is fitted "
            "to the points inside them (no automatic point rejection -- these "
            "regions are your statement of where the baseline is).\n\n"
            "Separate ranges with “;”. Or use “Pick on plot” and drag across "
            "the Processing preview. With no regions, the first and last 5 % of "
            "the spectrum are used.")
        self.regions_edit.editingFinished.connect(self._on_regions_edited)
        layout.addWidget(self.regions_edit)

        region_buttons = QHBoxLayout()
        self.pick_button = QPushButton("Pick on plot")
        self.pick_button.setCheckable(True)
        self.pick_button.setToolTip(
            "While on, each drag across the Processing preview adds a baseline region.")
        self.pick_button.toggled.connect(self.pickToggled)
        self.clear_button = QPushButton("Clear regions")
        self.clear_button.clicked.connect(self.clearRequested)
        region_buttons.addWidget(self.pick_button)
        region_buttons.addWidget(self.clear_button)
        region_buttons.addStretch(1)
        layout.addLayout(region_buttons)

        grid = QGridLayout()
        grid.addWidget(QLabel("Coefficient"), 0, 0)
        grid.addWidget(QLabel("Value"), 0, 1)
        grid.addWidget(QLabel("± (fit)"), 0, 2)
        grid.addWidget(QLabel("Fix"), 0, 3)
        self.param_labels: dict[str, QLabel] = {}
        self.param_edits: dict[str, QLineEdit] = {}
        self.error_labels: dict[str, QLabel] = {}
        self.fix_checks: dict[str, QCheckBox] = {}
        validator = QDoubleValidator(self)
        validator.setNotation(QDoubleValidator.Notation.ScientificNotation)
        for row, name in enumerate(_ALL_NAMES, start=1):
            label = QLabel(f"{name}:")
            edit = QLineEdit("0")
            edit.setValidator(validator)
            # Only a real edit counts: editingFinished also fires on a
            # plain focus change, which must not switch "Refit" off.
            edit.textEdited.connect(self._mark_params_dirty)
            edit.editingFinished.connect(self._on_param_edited)
            error = QLabel("")
            fix = QCheckBox()
            fix.setToolTip(f"Hold {name} at the value in its box when fitting.")
            fix.toggled.connect(lambda _c: self.fixedChanged.emit(self.fixed()))
            grid.addWidget(label, row, 0)
            grid.addWidget(edit, row, 1)
            grid.addWidget(error, row, 2)
            grid.addWidget(fix, row, 3)
            self.param_labels[name] = label
            self.param_edits[name] = edit
            self.error_labels[name] = error
            self.fix_checks[name] = fix
        layout.addLayout(grid)

        actions = QHBoxLayout()
        self.fit_button = QPushButton("Fit to regions")
        self.fit_button.setToolTip(
            "Least-squares fit of the model to the points in the baseline regions "
            "of the trace currently shown (the selected slice, for a series), "
            "holding any ticked “Fix” coefficients. Fills in the boxes above.")
        self.fit_button.clicked.connect(self.fitRequested)
        actions.addWidget(self.fit_button)
        actions.addStretch(1)
        layout.addLayout(actions)

        self.refit_check = QCheckBox("Refit on every trace (each slice, after re-phasing)")
        self.refit_check.setToolTip(
            "On: the model is fitted afresh to the regions whenever the baseline is "
            "applied -- every slice of a series gets its own coefficients, and a "
            "re-phased spectrum is refitted. Off: exactly the coefficients in the "
            "boxes are subtracted, everywhere. Editing a coefficient by hand turns "
            "this off.")
        self.refit_check.toggled.connect(self.refitToggled)
        layout.addWidget(self.refit_check)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        self.set_model("polynomial")

    # ---- state ----------------------------------------------------------
    @property
    def model(self) -> str:
        return self._model

    def set_model(self, model: str) -> None:
        self._model = model
        names = MANUAL_BASELINE_PARAMS[model]
        self.equation_label.setText(
            f"<b>baseline = {MANUAL_BASELINE_EQUATIONS[model]}</b>, x = δ in ppm, "
            f"in the spectrum's intensity units.")
        for name in _ALL_NAMES:
            shown = name in names
            for widget in (self.param_labels[name], self.param_edits[name],
                           self.error_labels[name], self.fix_checks[name]):
                widget.setVisible(shown)
            if shown:
                self.param_edits[name].setToolTip(_PARAM_TIPS[model][name])

    def regions(self) -> list[tuple[float, float]]:
        return parse_regions(self.regions_edit.text())

    def set_regions(self, regions) -> None:
        self.regions_edit.setText(format_regions(regions))

    def params(self) -> list[float]:
        values = []
        for name in MANUAL_BASELINE_PARAMS[self._model]:
            text = self.param_edits[name].text().strip().replace("−", "-")
            try:
                values.append(float(text))
            except ValueError:
                values.append(0.0)
        return values

    def set_params(self, values, errors=None) -> None:
        names = MANUAL_BASELINE_PARAMS[self._model]
        for i, name in enumerate(names):
            value = values[i] if values is not None and i < len(values) else 0.0
            self.param_edits[name].setText(_short(value))
            err = None if errors is None else errors.get(name)
            self.error_labels[name].setText("" if not err else f"± {float(err):.2g}")

    def clear_errors(self) -> None:
        for label in self.error_labels.values():
            label.setText("")

    def fixed(self) -> list[bool]:
        return [self.fix_checks[n].isChecked() for n in MANUAL_BASELINE_PARAMS[self._model]]

    def set_fixed(self, flags) -> None:
        names = MANUAL_BASELINE_PARAMS[self._model]
        for i, name in enumerate(names):
            check = self.fix_checks[name]
            check.blockSignals(True)
            check.setChecked(bool(flags[i]) if flags is not None and i < len(flags) else False)
            check.blockSignals(False)

    # ---- handlers -------------------------------------------------------
    def _on_regions_edited(self) -> None:
        try:
            regions = self.regions()
        except ValueError as exc:
            self.status_label.setText(f"Regions: {exc}.")
            return
        self.regionsEdited.emit(regions)

    def _mark_params_dirty(self, *_args) -> None:
        self._params_dirty = True

    def _on_param_edited(self) -> None:
        if not self._params_dirty:
            return
        self._params_dirty = False
        self.clear_errors()
        self.paramsEdited.emit(self.params())
