"""
Compare > Spectrum Arithmetic: signed, scaled sums of the checked spectra.

The maths is :func:`opennmr.notebook.arithmetic.combine_spectra`, called
exactly as a notebook calls it (on :func:`..notebook.bridge.results_from_spectrum`
dicts), so a difference spectrum made here and one made in a notebook
are the same numbers. This module only turns the controls into that
call's arguments and its result back into an :class:`NMRSpectrum`.

Every spectrum gets its own sign and factor, so any combination the
notebook function can make can be made here -- ``A - 0.5 B``, an average,
a scaled difference. The formula being computed is always written out
under the table, and is recorded in the new spectrum's history.
"""

from __future__ import annotations

import dataclasses

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView, QComboBox, QDoubleSpinBox, QFormLayout, QHBoxLayout, QHeaderView,
    QLabel, QPushButton, QSpinBox, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from ..core.spectrum import NMRAxis, NMRSpectrum
from ..notebook import arithmetic as nb_arithmetic
from ..notebook.bridge import results_from_spectrum

#: (label, data) of the preset combo. "custom" means "as the table says".
PRESETS = (("Sum", "sum"), ("Average", "average"), ("Subtract from first", "subtract"),
           ("Custom (signs and factors below)", "custom"))

_RANGES = (("Overlap only (recommended)", "intersection",
            "Only the range every spectrum covers. Right for a difference spectrum: "
            "nothing is invented where one spectrum has no data."),
           ("Union, zero-filled", "union",
            "The range any spectrum covers, each taken as zero outside its own range "
            "(as a VOCS sum does). A difference will show false 'signal' at the edges."))

_INTERP = (("Linear (recommended)", "linear"), ("Cubic", "cubic"), ("Nearest point", "nearest"))


def preset_terms(preset: str, n: int) -> tuple[list[str], list[float]]:
    """``(ops, factors)`` for ``n`` spectra under a preset."""
    if preset == "average":
        return ["+"] * n, [1.0 / n] * n
    if preset == "subtract":
        return ["+"] + ["-"] * (n - 1), [1.0] * n
    return ["+"] * n, [1.0] * n


def formula(labels, ops, factors) -> str:
    """The combination written out, e.g. ``A − 0.5 × B``."""
    parts = []
    for k, (label, op, factor) in enumerate(zip(labels, ops, factors)):
        term = label if factor == 1 else f"{factor:g} × {label}"
        if k == 0:
            parts.append(("−" if op == "-" else "") + term)
        else:
            parts.append(("− " if op == "-" else "+ ") + term)
    return " ".join(parts)


def combine_to_spectrum(spectra, ops, factors, x_unit: str = "ppm",
                        common_range: str = "intersection", npoints: int | None = None,
                        interp_kind: str = "linear") -> NMRSpectrum:
    """:func:`notebook.arithmetic.combine_spectra` on 1D spectra, returned
    as an :class:`NMRSpectrum` on the common axis. Hz <-> ppm use the
    first spectrum's SFO1 (exact for the unit interpolated on; for the
    other, an approximation when the spectra's SFO1s differ).

    ``metadata.raw`` records what was done: ``"ops"``, ``"scaling_factors"``,
    ``"arithmetic"`` (the options) and ``"interpolated_spectra"`` (each
    input on the common axis, before signs/factors), as
    :func:`opennmr.processing.arithmetic.combine_spectra` does."""
    unit = "Hz" if str(x_unit).lower() == "hz" else "ppm"
    results = [results_from_spectrum(s) for s in spectra]
    out = nb_arithmetic.combine_spectra(results, ops=list(ops), scaling_factors=list(factors),
                                        x_unit=unit, common_range=common_range,
                                        npoints=npoints, interp_kind=interp_kind)
    ref = spectra[0]
    sfo1 = float(ref.metadata.sfo1) if ref.metadata.sfo1 else None
    if unit == "ppm":
        x_ppm = np.asarray(out["x_ppm"], dtype=float)
        x_hz = x_ppm * sfo1 if sfo1 else None
    else:
        x_hz = np.asarray(out["x_Hz"], dtype=float)
        x_ppm = x_hz / sfo1 if sfo1 else None
    axis = NMRAxis(kind="frequency", ppm=x_ppm, hz=x_hz)
    metadata = dataclasses.replace(
        ref.metadata, source_path=None,
        raw={"ops": list(out["ops"]), "scaling_factors": [float(f) for f in factors],
             "arithmetic": {"x_unit": unit, "common_range": common_range,
                            "npoints": int(len(out["combined_spectrum"])),
                            "interp_kind": interp_kind},
             "interpolated_spectra": out["interpolated_spectra"]})
    return NMRSpectrum(data=np.asarray(out["combined_spectrum"], dtype=float), axes=[axis],
                       metadata=metadata)


class ArithmeticPanel(QWidget):
    """``entries()`` -> ``[{"label", "spectrum"}]``: the 1D spectra
    checked in the Compare list, in list order.
    ``add_result(spectrum, label, description)``: put the combination in
    the session (MainWindow logs ``description`` in its history)."""

    combined = Signal(object)

    def __init__(self, entries, add_result, parent=None) -> None:
        super().__init__(parent)
        self._entries = entries
        self._add_result = add_result
        #: Sign/factor chosen per spectrum label, kept across refreshes so
        #: checking another spectrum does not reset the others.
        self._terms: dict[str, tuple[str, float]] = {}
        self._labels: list[str] = []

        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        intro = QLabel(
            "Add, subtract, average or scale the spectra checked in the 'Compare' list. The "
            "experiments are first interpolated onto one common axis (different experiments never share "
            "an identical axis). The result is added to 'Open experiments' (bottom left) as a new spectrum.")
        intro.setWordWrap(True)
        v.addWidget(intro)

        form = QFormLayout()
        self.op_combo = QComboBox()
        for text, data in PRESETS:
            self.op_combo.addItem(text, data)
        self.op_combo.setToolTip(
            "Fills in the signs and factors below. Edit any of them to make your own "
            "combination (the choice becomes Custom).")
        self.op_combo.currentIndexChanged.connect(lambda _i: self._on_preset_changed())
        form.addRow("Operation:", self.op_combo)
        v.addLayout(form)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Spectrum", "Sign", "× Factor"])
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.table.setToolTip("One row per checked spectrum, in the Compare list's order.")
        v.addWidget(self.table)

        self.formula_label = QLabel("")
        self.formula_label.setWordWrap(True)
        self.formula_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        v.addWidget(self.formula_label)

        options = QFormLayout()
        self.range_combo = QComboBox()
        for text, data, tip in _RANGES:
            self.range_combo.addItem(text, data)
            self.range_combo.setItemData(self.range_combo.count() - 1, tip,
                                         Qt.ItemDataRole.ToolTipRole)
        self.range_combo.setToolTip("\n\n".join(f"{t}: {tip}" for t, _d, tip in _RANGES))
        options.addRow("Range:", self.range_combo)
        self.unit_combo = QComboBox()
        self.unit_combo.addItem("ppm (recommended)", "ppm")
        self.unit_combo.addItem("Hz", "Hz")
        self.unit_combo.setToolTip(
            "The axis the spectra are lined up on. ppm is field-independent, so it is right "
            "for spectra recorded at different fields or carriers; Hz assumes one SFO1.")
        options.addRow("Align on:", self.unit_combo)
        self.points_spin = QSpinBox()
        self.points_spin.setRange(0, 4_194_304)
        self.points_spin.setSingleStep(1024)
        self.points_spin.setSpecialValueText("Auto (finest input)")
        self.points_spin.setToolTip(
            "Points on the common axis. Auto uses the largest point count among the inputs, "
            "so the result is never coarser than the finest spectrum.")
        options.addRow("Points:", self.points_spin)
        self.interp_combo = QComboBox()
        for text, data in _INTERP:
            self.interp_combo.addItem(text, data)
        self.interp_combo.setToolTip(
            "How each spectrum is resampled onto the common axis. Linear is safest on noisy "
            "data; cubic can ring beside sharp peaks.")
        options.addRow("Interpolation:", self.interp_combo)
        v.addLayout(options)

        self.combine_button = QPushButton("Combine → New Spectrum")
        self.combine_button.setToolTip(
            "Compute the formula above and add the result to Open experiments.")
        self.combine_button.clicked.connect(self.combine)
        v.addWidget(self.combine_button)
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        v.addWidget(self.status_label)
        self.refresh()

    # ------------------------------------------------------------------

    def _current_entries(self) -> list[dict]:
        return list(self._entries() or [])

    def refresh(self) -> None:
        """Rebuild the table from the checked spectra, keeping each
        spectrum's sign/factor where it is still checked."""
        entries = self._current_entries()
        self._labels = [e["label"] for e in entries]
        preset = self.op_combo.currentData()
        if preset != "custom":
            ops, factors = preset_terms(preset, max(len(entries), 1))
        self.table.setRowCount(len(entries))
        for row, label in enumerate(self._labels):
            if preset != "custom":
                op, factor = ops[row], factors[row]
            else:
                op, factor = self._terms.get(label, ("+", 1.0))
            self._terms[label] = (op, factor)
            name = QTableWidgetItem(label)
            name.setToolTip(label)
            self.table.setItem(row, 0, name)
            sign = QComboBox()
            sign.addItem("+", "+")
            sign.addItem("−", "-")
            sign.setCurrentIndex(1 if op == "-" else 0)
            sign.setAccessibleName(f"Sign for {label}")
            sign.currentIndexChanged.connect(lambda _i: self._on_term_edited())
            self.table.setCellWidget(row, 1, sign)
            spin = QDoubleSpinBox()
            spin.setRange(0.0, 1e9)
            spin.setDecimals(4)
            spin.setSingleStep(0.1)
            spin.setValue(float(factor))
            spin.setAccessibleName(f"Factor for {label}")
            spin.valueChanged.connect(lambda _v: self._on_term_edited())
            self.table.setCellWidget(row, 2, spin)
        # As tall as its rows (up to six), not a fixed block of blank space.
        rows_h = sum(self.table.rowHeight(r) for r in range(min(len(entries), 6)))
        self.table.setFixedHeight(self.table.horizontalHeader().height() + max(rows_h, 24) + 4)
        self._refresh_formula()

    def terms(self) -> tuple[list[str], list[float]]:
        """``(ops, factors)`` as the table shows them, in row order."""
        ops, factors = [], []
        for row in range(self.table.rowCount()):
            ops.append(self.table.cellWidget(row, 1).currentData())
            factors.append(float(self.table.cellWidget(row, 2).value()))
        return ops, factors

    def _on_preset_changed(self) -> None:
        self.refresh()

    def _on_term_edited(self) -> None:
        ops, factors = self.terms()
        for label, op, factor in zip(self._labels, ops, factors):
            self._terms[label] = (op, factor)
        if self.op_combo.currentData() != "custom":
            blocked = self.op_combo.blockSignals(True)
            self.op_combo.setCurrentIndex(self.op_combo.findData("custom"))
            self.op_combo.blockSignals(blocked)
        self._refresh_formula()

    def _refresh_formula(self) -> None:
        n = len(self._labels)
        if n < 2:
            self.formula_label.setText(
                "<i>Select at least two 1D spectra in the Compare list.</i>")
            self.combine_button.setEnabled(False)
            return
        ops, factors = self.terms()
        self.formula_label.setText("<b>Result</b> = " + formula(self._labels, ops, factors))
        self.combine_button.setEnabled(True)

    # ------------------------------------------------------------------

    def combine(self):
        """Compute the combination and hand it to ``add_result``; returns
        the new spectrum, or None (with the reason in the status line)."""
        self.refresh()
        entries = self._current_entries()
        if len(entries) < 2:
            self.status_label.setText("Needs at least two 1D spectra checked in the Compare list.")
            return None
        ops, factors = self.terms()
        if not any(factors):
            self.status_label.setText("Every factor is zero: the result would be empty.")
            return None
        common_range = self.range_combo.currentData()
        x_unit = self.unit_combo.currentData()
        npoints = self.points_spin.value() or None
        interp = self.interp_combo.currentData()
        try:
            spectrum = combine_to_spectrum([e["spectrum"] for e in entries], ops, factors,
                                           x_unit=x_unit, common_range=common_range,
                                           npoints=npoints, interp_kind=interp)
        except ValueError as exc:
            self.status_label.setText(f"Not combined: {exc}")
            return None
        text = formula(self._labels, ops, factors)
        label = self._result_label(ops, factors)
        description = (
            f"Spectrum arithmetic (notebook.arithmetic.combine_spectra): {text}\n"
            f"Aligned on {x_unit}, range {common_range}, "
            f"{spectrum.data.size} points ({'auto' if npoints is None else 'set'}), "
            f"{interp} interpolation.")
        self._add_result(spectrum, label, description)
        self.status_label.setText(f"Added “{label}”: {text}")
        self.combined.emit(spectrum)
        return spectrum

    def _result_label(self, ops, factors) -> str:
        n = len(ops)
        preset = self.op_combo.currentData()
        if preset == "sum" or (all(o == "+" for o in ops) and all(f == 1 for f in factors)):
            return f"sum({n} spectra)"
        if preset == "average":
            return f"avg({n} spectra)"
        if preset == "subtract":
            return f"diff({n} spectra)"
        return f"combination({n} spectra)"
