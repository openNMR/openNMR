"""
Apply ▸ DNP: a field-sweep profile and enhancement factors, step by step
(explicit user request, 2026-09-30: the old Compare ▸ DNP boxes did not
make clear what to do).

Both sections are one :class:`DnpWorkflowPanel`, in ``"sweep"`` or
``"enhancement"`` mode, and follow the same numbered steps:

1. **Spectra** -- tick the spectra to use (any 1D spectra in Open
   experiments). For a sweep, each one's B0 from its nucleus and SFO1
   is listed straight away.
2. **Reference** -- choose the reference spectrum (shown on the Compare
   view), click its line, and align every other spectrum to it.
3. **Check** -- the aligned spectra, stacked or superimposed, each
   normalised; a table of the shifts found and how well each matched.
   Correct any by typing a shift, or drag the traces in the alignment
   window; then **confirm**. Every spectrum is then on its own ppm axis
   (its own SR), its line at the reference line's shift.
4. **Measure** -- pick a shift (height) and/or integrate a region on the
   aligned spectra; both are encouraged, and compared.
5. **Result** -- the sweep: height and integral against B0 (from the
   line, or from SFO1), absolute or as an enhancement (ON/OFF or
   1 − ON/OFF) against a chosen OFF spectrum. The enhancement: every
   ticked ON spectrum against the OFF one, by height and by integral.

The maths is :mod:`opennmr.notebook.dnp_workflow`, called as a notebook
calls it; its plots are drawn onto the GUI's canvases.
"""

from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFileDialog, QFormLayout,
    QGroupBox, QHBoxLayout, QHeaderView, QLabel, QMessageBox, QPushButton, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

from ..notebook import dnp
from ..notebook import dnp_workflow as dw

#: What an armed canvas gesture is for.
PICK_LINE, PICK_SHIFT, PICK_REGION = "line", "shift", "region"

_BASELINE_ITEMS = (
    ("Local (recommended)", "local",
     "A quadratic fitted just outside each peak (±5 FWHM) or region: follows a rolling "
     "baseline."),
    ("Global (as Quantify)", "global",
     "One robust straight line fitted to the spectrum's edges, as Quantify's integration uses."),
    ("None", "none", "The data as they are."))

_BOTH_HINT = ("Do both if you can: the height and the integral agree only if the line shape is "
              "the same in every spectrum, so measuring both checks the result.")


def _baseline_combo() -> QComboBox:
    combo = QComboBox()
    for text, data, tip in _BASELINE_ITEMS:
        combo.addItem(text, data)
        combo.setItemData(combo.count() - 1, tip, Qt.ItemDataRole.ToolTipRole)
    combo.setToolTip("\n".join(f"{t}: {tip}" for t, _d, tip in _BASELINE_ITEMS))
    return combo


def _item(text, editable=False, align_right=False) -> QTableWidgetItem:
    item = QTableWidgetItem(text)
    if not editable:
        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
    if align_right:
        item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    return item


def _fmt(value, digits=4) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "—"
    return f"{value:.{digits}g}"


def _pm(value, error) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{value:.4g} ± {error:.2g}"


class DnpWorkflowPanel(QWidget):
    """One DNP measurement, in five steps (see the module docstring).

    Callbacks (from MainWindow): ``spectra()`` -> ``[{"label",
    "spectrum", "index"}]`` for every 1D spectrum in Open experiments;
    ``selected()`` -> the session rows selected there; ``draw(draw,
    view)``: show ``draw(figure)`` -- ``"spectrum"`` (one spectrum) or
    ``"spectra"`` on the Compare view, ``"profile"`` on the Fit view; ``arm(mode)``: the Compare view's
    ``"peak"`` pick, ``"region"`` drag, or neither (None); ``store(kind,
    record, indices)``: keep a result with those spectra; ``style(view)``
    the view's plot style."""

    #: A step finished (for tests and the main window).
    changed = Signal()

    def __init__(self, mode, spectra, selected, draw, arm, store, style=None,
                 parent=None) -> None:
        super().__init__(parent)
        if mode not in ("sweep", "enhancement"):
            raise ValueError(f"mode must be 'sweep' or 'enhancement', not {mode!r}")
        self.mode = mode
        self._spectra, self._selected = spectra, selected
        self._draw, self._arm, self._store = draw, arm, store
        self._style = style or (lambda _view: {})
        self._armed: str | None = None
        self._checked_ids: set[int] | None = None          # id(spectrum) of ticked rows
        self._entries: list[dict] = []                     # the ticked spectra, in order
        self.line_ppm: float | None = None                 # picked on the reference
        self.alignment: dict | None = None
        self.aligned: list | None = None
        self.peak_ppm: float | None = None                 # step 4
        self.region: tuple[float, float] | None = None
        self.measurement: dict | None = None
        self.profile: dict | None = None
        self.enhancement_rows: list | None = None
        self._align_dialog = None

        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        intro = QLabel(
            "A DNP field profile: signal against B₀, from spectra each recorded at a different "
            "field. Work down the steps; each needs the one before." if mode == "sweep" else
            "DNP enhancement: microwaves-on spectra against a microwaves-off one. Work down the "
            "steps; each needs the one before.")
        intro.setWordWrap(True)
        v.addWidget(intro)
        v.addWidget(self._build_spectra())
        v.addWidget(self._build_reference())
        v.addWidget(self._build_check())
        v.addWidget(self._build_measure())
        v.addWidget(self._build_sweep_result() if mode == "sweep" else self._build_enh_result())
        self.refresh()
        self._sync()

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------

    def _build_spectra(self) -> QGroupBox:
        box = QGroupBox("1. Spectra")
        v = QVBoxLayout(box)
        hint = QLabel("Tick the spectra to use (1D spectra in Open experiments)."
                      + (" B₀ from each one's nucleus and SFO1 is shown." if self.mode == "sweep"
                         else " One of them will be the microwaves-off spectrum."))
        hint.setWordWrap(True)
        v.addWidget(hint)
        columns = ["Spectrum", "SFO1 (MHz)"] + (["B₀ from SFO1 (T)"] if self.mode == "sweep"
                                               else [])
        self.spectra_table = QTableWidget(0, len(columns))
        self.spectra_table.setHorizontalHeaderLabels(columns)
        self.spectra_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch)
        self.spectra_table.verticalHeader().setVisible(False)
        self.spectra_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.spectra_table.setMaximumHeight(170)
        self.spectra_table.itemChanged.connect(self._on_spectrum_ticked)
        v.addWidget(self.spectra_table)
        row = QHBoxLayout()
        self.use_selected_button = QPushButton("Use Selected")
        self.use_selected_button.setToolTip("Tick exactly the spectra selected in Open "
                                            "experiments (Ctrl/Shift-click to select several).")
        self.use_selected_button.clicked.connect(self.use_selected)
        self.all_button = QPushButton("All")
        self.all_button.clicked.connect(lambda: self._tick_all(True))
        self.none_button = QPushButton("None")
        self.none_button.clicked.connect(lambda: self._tick_all(False))
        for b in (self.use_selected_button, self.all_button, self.none_button):
            row.addWidget(b)
        v.addLayout(row)
        self.spectra_status = QLabel("")
        self.spectra_status.setWordWrap(True)
        v.addWidget(self.spectra_status)
        return box

    def _build_reference(self) -> QGroupBox:
        box = QGroupBox("2. Reference and alignment")
        v = QVBoxLayout(box)
        hint = QLabel("Choose the reference spectrum and click its line: every other spectrum is "
                      "searched for the same line shape and aligned to it (the receiver offset "
                      "may differ between experiments; positions are not trusted).")
        hint.setWordWrap(True)
        v.addWidget(hint)
        form = QFormLayout()
        self.reference_combo = QComboBox()
        self.reference_combo.setToolTip("The spectrum the others are aligned to (and whose "
                                        "receiver gain the others are scaled to).")
        self.reference_combo.currentIndexChanged.connect(self._on_reference_changed)
        form.addRow("Reference:", self.reference_combo)
        self.window_spin = QDoubleSpinBox()
        self.window_spin.setRange(0.1, 1000.0)
        self.window_spin.setDecimals(1)
        self.window_spin.setValue(dnp.DEFAULT_WINDOW_HZ / 1e3)
        self.window_spin.setSuffix(" kHz")
        self.window_spin.setToolTip("How far from the reference line's offset from its carrier "
                                    "to search each spectrum (± this).")
        form.addRow("Search ±:", self.window_spin)
        self.match_spin = QDoubleSpinBox()
        self.match_spin.setRange(0.0, 1e6)
        self.match_spin.setDecimals(0)
        self.match_spin.setSuffix(" Hz")
        self.match_spin.setSpecialValueText("auto (3 × FWHM)")
        self.match_spin.setToolTip("Half-width of the line shape matched (the template), about "
                                   "the picked line. Auto: 3 × the reference line's FWHM.")
        form.addRow("Match ±:", self.match_spin)
        v.addLayout(form)
        row = QHBoxLayout()
        self.show_reference_button = QPushButton("Show Reference")
        self.show_reference_button.clicked.connect(self.show_reference)
        self.pick_line_button = QPushButton("Pick the Line…")
        self.pick_line_button.setCheckable(True)
        self.pick_line_button.setToolTip("Then click the line on the reference spectrum "
                                         "(Compare view).")
        self.pick_line_button.toggled.connect(lambda on: self._arm_for(PICK_LINE, on))
        row.addWidget(self.show_reference_button)
        row.addWidget(self.pick_line_button)
        v.addLayout(row)
        self.line_label = QLabel("No line picked yet.")
        v.addWidget(self.line_label)
        self.align_button = QPushButton("Align All to This Line")
        self.align_button.clicked.connect(self.align)
        v.addWidget(self.align_button)
        return box

    def _build_check(self) -> QGroupBox:
        box = QGroupBox("3. Check the alignment")
        v = QVBoxLayout(box)
        hint = QLabel("Each spectrum normalised, on the aligned axis. Correct a shift by typing "
                      "it (By hand) or by dragging traces in the alignment window, then confirm.")
        hint.setWordWrap(True)
        v.addWidget(hint)
        self.align_table = QTableWidget(0, 5)
        self.align_table.setHorizontalHeaderLabels(
            ["Spectrum", "Shift found (Hz)", "By hand (Hz)", "Match |r|", "Note"])
        self.align_table.horizontalHeader().setSectionResizeMode(
            4, QHeaderView.ResizeMode.Stretch)
        self.align_table.verticalHeader().setVisible(False)
        self.align_table.setMaximumHeight(180)
        self.align_table.setToolTip(
            "Shift found: where each spectrum's line sits relative to the reference's, as "
            "offsets from their own carriers. Match |r|: how well its line's shape matched the "
            "reference's (1 = identical). By hand: an extra shift you set (edit it here).")
        self.align_table.itemChanged.connect(self._on_align_table_edited)
        v.addWidget(self.align_table)
        form = QFormLayout()
        self.view_combo = QComboBox()
        self.view_combo.addItem("Stacked", "stack")
        self.view_combo.addItem("Superimposed", "overlay")
        self.view_combo.currentIndexChanged.connect(lambda _i: self.show_aligned())
        form.addRow("View:", self.view_combo)
        self.normalise_check = QCheckBox("Normalise each to height 1")
        self.normalise_check.setChecked(True)
        self.normalise_check.toggled.connect(lambda _on: self.show_aligned())
        form.addRow("", self.normalise_check)
        v.addLayout(form)
        row = QHBoxLayout()
        self.show_aligned_button = QPushButton("Show Aligned")
        self.show_aligned_button.clicked.connect(lambda: self.show_aligned())
        self.drag_button = QPushButton("Drag to Adjust…")
        self.drag_button.setToolTip("A window where each trace can be dragged left/right onto "
                                    "the others (stacked or superimposed).")
        self.drag_button.clicked.connect(self.open_align_dialog)
        row.addWidget(self.show_aligned_button)
        row.addWidget(self.drag_button)
        v.addLayout(row)
        self.confirm_button = QPushButton("Confirm Alignment")
        self.confirm_button.clicked.connect(self.confirm)
        v.addWidget(self.confirm_button)
        self.align_status = QLabel("")
        self.align_status.setWordWrap(True)
        v.addWidget(self.align_status)
        return box

    def _build_measure(self) -> QGroupBox:
        box = QGroupBox("4. Measure")
        v = QVBoxLayout(box)
        hint = QLabel("On the aligned spectra: pick a shift for the height, and/or drag a region "
                      "to integrate. " + _BOTH_HINT)
        hint.setWordWrap(True)
        v.addWidget(hint)
        row = QHBoxLayout()
        self.pick_shift_button = QPushButton("Pick Shift…")
        self.pick_shift_button.setCheckable(True)
        self.pick_shift_button.toggled.connect(lambda on: self._arm_for(PICK_SHIFT, on))
        self.region_button = QPushButton("Integrate Region…")
        self.region_button.setCheckable(True)
        self.region_button.toggled.connect(lambda on: self._arm_for(PICK_REGION, on))
        row.addWidget(self.pick_shift_button)
        row.addWidget(self.region_button)
        v.addLayout(row)
        form = QFormLayout()
        self.baseline_combo = _baseline_combo()
        self.baseline_combo.currentIndexChanged.connect(lambda _i: self.remeasure())
        form.addRow("Baseline:", self.baseline_combo)
        self.rg_check = QCheckBox("Scale to the reference's receiver gain")
        self.rg_check.setChecked(True)
        self.rg_check.setToolTip("Intensities are already per scan (divided by NS on import); "
                                 "this also removes differences in receiver gain.")
        self.rg_check.toggled.connect(lambda _on: self.remeasure())
        form.addRow("", self.rg_check)
        v.addLayout(form)
        self.measure_label = QLabel("")
        self.measure_label.setWordWrap(True)
        self.measure_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        v.addWidget(self.measure_label)
        return box

    def _build_sweep_result(self) -> QGroupBox:
        box = QGroupBox("5. Profile against B₀")
        v = QVBoxLayout(box)
        form = QFormLayout()
        self.field_source_combo = QComboBox()
        self.field_source_combo.addItem("At the aligned line (recommended)", "line")
        self.field_source_combo.addItem("From SFO1 (line assumed on the carrier)", "sfo1")
        self.field_source_combo.setToolTip(
            "From SFO1: B₀ = SFO1 / γ, which assumes the line sat exactly on the carrier. At the "
            "aligned line: B₀ from the line's own frequency in each spectrum (carrier + its "
            "offset, as the alignment found it) -- the direct measurement; the carrier is set by "
            "hand and can sit several kHz (~1 mT for ²³Na) off the line.")
        self.field_source_combo.currentIndexChanged.connect(lambda _i: self.plot_profile())
        form.addRow("B₀:", self.field_source_combo)
        self.profile_mode_combo = QComboBox()
        self.profile_mode_combo.addItem("Absolute (per scan)", "absolute")
        self.profile_mode_combo.addItem("Enhancement ε = ON / OFF", "ratio")
        self.profile_mode_combo.addItem("1 − ON / OFF", "one_minus")
        self.profile_mode_combo.currentIndexChanged.connect(lambda _i: self._sync())
        self.profile_mode_combo.currentIndexChanged.connect(lambda _i: self.plot_profile())
        form.addRow("Plot:", self.profile_mode_combo)
        self.off_combo = QComboBox()
        self.off_combo.setToolTip("The microwaves-off spectrum the others are divided by.")
        self.off_combo.currentIndexChanged.connect(lambda _i: self.plot_profile())
        form.addRow("OFF spectrum:", self.off_combo)
        self.exclude_off_check = QCheckBox("Leave the OFF spectrum out of the profile")
        self.exclude_off_check.setChecked(True)
        self.exclude_off_check.toggled.connect(lambda _on: self.plot_profile())
        form.addRow("", self.exclude_off_check)
        self.quantity_combo = QComboBox()
        self.quantity_combo.addItem("Height and integral", "both")
        self.quantity_combo.addItem("Height", "height")
        self.quantity_combo.addItem("Integral", "area")
        self.quantity_combo.currentIndexChanged.connect(lambda _i: self.plot_profile())
        form.addRow("Show:", self.quantity_combo)
        self.field_axis_combo = QComboBox()
        self.field_axis_combo.addItem("B₀ (T)", False)
        self.field_axis_combo.addItem("ΔB₀ from the lowest field (mT)", True)
        self.field_axis_combo.currentIndexChanged.connect(lambda _i: self.plot_profile())
        form.addRow("Field axis:", self.field_axis_combo)
        v.addLayout(form)
        row = QHBoxLayout()
        self.plot_button = QPushButton("Plot Profile")
        self.plot_button.clicked.connect(lambda: self.plot_profile(explicit=True))
        self.export_button = QPushButton("Export CSV…")
        self.export_button.clicked.connect(self.export_csv)
        row.addWidget(self.plot_button)
        row.addWidget(self.export_button)
        v.addLayout(row)
        self.result_label = QLabel("")
        self.result_label.setWordWrap(True)
        self.result_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        v.addWidget(self.result_label)
        return box

    def _build_enh_result(self) -> QGroupBox:
        box = QGroupBox("5. Enhancement")
        v = QVBoxLayout(box)
        form = QFormLayout()
        self.off_combo = QComboBox()
        self.off_combo.setToolTip("The microwaves-off spectrum; every other ticked spectrum is "
                                  "an ON spectrum.")
        self.off_combo.currentIndexChanged.connect(lambda _i: self.calculate())
        form.addRow("OFF spectrum:", self.off_combo)
        self.definition_combo = QComboBox()
        self.definition_combo.addItem("ε = ON / OFF", "ratio")
        self.definition_combo.addItem("1 − ON / OFF", "one_minus")
        self.definition_combo.setToolTip("Which the table leads with; the report gives both.")
        self.definition_combo.currentIndexChanged.connect(lambda _i: self.calculate())
        form.addRow("Show as:", self.definition_combo)
        v.addLayout(form)
        self.calculate_button = QPushButton("Calculate Enhancement")
        self.calculate_button.clicked.connect(lambda: self.calculate(explicit=True))
        v.addWidget(self.calculate_button)
        self.enh_table = QTableWidget(0, 4)
        self.enh_table.setHorizontalHeaderLabels(["ON spectrum", "By height", "By integral",
                                                  "Agree?"])
        self.enh_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.enh_table.verticalHeader().setVisible(False)
        self.enh_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.enh_table.setMaximumHeight(160)
        v.addWidget(self.enh_table)
        self.result_label = QLabel("")
        self.result_label.setWordWrap(True)
        self.result_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        v.addWidget(self.result_label)
        return box

    # ------------------------------------------------------------------
    # step 1: spectra
    # ------------------------------------------------------------------

    def refresh(self) -> None:
        """Re-read Open experiments' 1D spectra, keeping what was ticked."""
        rows = self._spectra()
        ticked = self._checked_ids
        table = self.spectra_table
        blocked = table.blockSignals(True)
        table.setRowCount(len(rows))
        self._rows = rows
        for r, entry in enumerate(rows):
            name = _item(entry["label"])
            name.setFlags(name.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            on = ticked is not None and id(entry["spectrum"]) in ticked
            name.setCheckState(Qt.CheckState.Checked if on else Qt.CheckState.Unchecked)
            table.setItem(r, 0, name)
            try:
                sfo1, _src = dnp.carrier_frequency_mhz(entry["spectrum"])
                table.setItem(r, 1, _item(f"{sfo1:.6f}", align_right=True))
                if self.mode == "sweep":
                    field = dw.spectrum_fields([entry["spectrum"]])["field_T"][0]
                    table.setItem(r, 2, _item(f"{field:.6f}", align_right=True))
            except ValueError:
                table.setItem(r, 1, _item("—"))
                if self.mode == "sweep":
                    table.setItem(r, 2, _item("—"))
        table.blockSignals(blocked)
        self._update_entries()

    def _update_entries(self) -> None:
        rows = getattr(self, "_rows", [])
        ticked = [rows[r] for r in range(self.spectra_table.rowCount())
                  if self.spectra_table.item(r, 0) is not None
                  and self.spectra_table.item(r, 0).checkState() == Qt.CheckState.Checked]
        before = [id(e["spectrum"]) for e in self._entries]
        self._entries = ticked
        self._checked_ids = {id(e["spectrum"]) for e in ticked}
        labels = [e["label"] for e in ticked]
        for combo in (self.reference_combo, self.off_combo):
            keep = combo.currentText()
            blocked = combo.blockSignals(True)
            combo.clear()
            combo.addItems(labels)
            if keep in labels:
                combo.setCurrentIndex(labels.index(keep))
            elif combo is self.off_combo and len(labels) > 1 and self.mode == "enhancement":
                combo.setCurrentIndex(len(labels) - 1)
            combo.blockSignals(blocked)
        if [id(e["spectrum"]) for e in ticked] != before:
            self._reset_from(2)
        nuclei = set()
        for e in ticked:
            try:
                nuclei.add(dnp._nucleus(e["spectrum"]))
            except ValueError:
                nuclei.add("?")
        need = 2
        if not ticked:
            text = "Tick at least two spectra."
        elif len(nuclei) > 1:
            text = f"{len(ticked)} ticked, but of different nuclei ({', '.join(sorted(nuclei))})."
        else:
            text = f"{len(ticked)} ticked, {', '.join(sorted(nuclei))}." + (
                "" if len(ticked) >= need else " Tick at least two.")
        self.spectra_status.setText(text)
        self._sync()

    def _on_spectrum_ticked(self, item) -> None:
        if item.column() == 0:
            self._update_entries()

    def _tick_all(self, on: bool) -> None:
        table = self.spectra_table
        blocked = table.blockSignals(True)
        for r in range(table.rowCount()):
            table.item(r, 0).setCheckState(Qt.CheckState.Checked if on else Qt.CheckState.Unchecked)
        table.blockSignals(blocked)
        self._update_entries()

    def use_selected(self) -> None:
        selected = set(self._selected())
        table = self.spectra_table
        blocked = table.blockSignals(True)
        for r, entry in enumerate(getattr(self, "_rows", [])):
            on = entry.get("index") in selected
            table.item(r, 0).setCheckState(Qt.CheckState.Checked if on else Qt.CheckState.Unchecked)
        table.blockSignals(blocked)
        self._update_entries()

    def set_checked(self, labels) -> None:
        """Tick exactly the spectra named ``labels`` (for code and tests)."""
        wanted = set(labels)
        table = self.spectra_table
        blocked = table.blockSignals(True)
        for r, entry in enumerate(getattr(self, "_rows", [])):
            table.item(r, 0).setCheckState(Qt.CheckState.Checked if entry["label"] in wanted
                                           else Qt.CheckState.Unchecked)
        table.blockSignals(blocked)
        self._update_entries()

    def _spectra_list(self):
        return [e["spectrum"] for e in self._entries]

    def _labels(self):
        return [e["label"] for e in self._entries]

    # ------------------------------------------------------------------
    # state
    # ------------------------------------------------------------------

    def _reset_from(self, step: int) -> None:
        """Forget everything from ``step`` on (a change upstream makes it
        stale)."""
        if step <= 2:
            self.line_ppm = None
            self.line_label.setText("No line picked yet.")
        if step <= 3:
            self.alignment = None
            blocked = self.align_table.blockSignals(True)
            self.align_table.setRowCount(0)
            self.align_table.blockSignals(blocked)
            self.align_status.setText("")
        if step <= 4:
            self.aligned = None
            self.measurement = None
            self.peak_ppm = self.region = None
            self.measure_label.setText("")
        self.profile = None
        self.enhancement_rows = None
        self.result_label.setText("")
        if self.mode == "enhancement":
            self.enh_table.setRowCount(0)

    def _sync(self) -> None:
        n = len(self._entries)
        self.show_reference_button.setEnabled(n >= 1)
        self.pick_line_button.setEnabled(n >= 1)
        self.align_button.setEnabled(n >= 2 and self.line_ppm is not None)
        have_alignment = self.alignment is not None
        for w in (self.show_aligned_button, self.drag_button, self.confirm_button,
                  self.view_combo, self.normalise_check):
            w.setEnabled(have_alignment)
        confirmed = bool(have_alignment and self.alignment.get("confirmed"))
        self.confirm_button.setText("Alignment Confirmed ✓" if confirmed else "Confirm Alignment")
        for w in (self.pick_shift_button, self.region_button):
            w.setEnabled(confirmed)
        measured = self.measurement is not None
        if self.mode == "sweep":
            self.plot_button.setEnabled(measured)
            self.export_button.setEnabled(self.profile is not None)
            relative = self.profile_mode_combo.currentData() != "absolute"
            self.off_combo.setEnabled(relative)
            self.exclude_off_check.setEnabled(relative)
        else:
            self.calculate_button.setEnabled(measured and n >= 2)

    # ------------------------------------------------------------------
    # canvas gestures
    # ------------------------------------------------------------------

    def _buttons(self):
        return {PICK_LINE: self.pick_line_button, PICK_SHIFT: self.pick_shift_button,
                PICK_REGION: self.region_button}

    def _arm_for(self, mode: str, on: bool) -> None:
        if on:
            for other, button in self._buttons().items():
                if other != mode and button.isChecked():
                    blocked = button.blockSignals(True)
                    button.setChecked(False)
                    button.blockSignals(blocked)
            shown = self.show_reference() if mode == PICK_LINE else self.show_aligned()
            if not shown:
                self._disarm()
                return
            self._armed = mode
            self._arm("region" if mode == PICK_REGION else "peak")
        elif self._armed == mode:
            self._armed = None
            self._arm(None)

    def _disarm(self) -> None:
        self._armed = None
        for button in self._buttons().values():
            blocked = button.blockSignals(True)
            button.setChecked(False)
            button.blockSignals(blocked)
        self._arm(None)

    @property
    def armed(self) -> str | None:
        return self._armed

    def canvas_mode_released(self) -> None:
        """The Compare view's pick/region mode was turned off elsewhere."""
        if self._armed is not None:
            self._armed = None
            for button in self._buttons().values():
                blocked = button.blockSignals(True)
                button.setChecked(False)
                button.blockSignals(blocked)

    def on_canvas_pick(self, x: float) -> None:
        mode = self._armed
        if mode == PICK_LINE:
            self._disarm()
            self.set_line(float(x))
        elif mode == PICK_SHIFT:
            self._disarm()
            self.set_peak(float(x))

    def on_canvas_region(self, lo: float, hi: float) -> None:
        if self._armed == PICK_REGION:
            self._disarm()
            self.set_region(float(lo), float(hi))

    # ------------------------------------------------------------------
    # step 2: reference and alignment
    # ------------------------------------------------------------------

    def _reference_index(self) -> int:
        return max(0, self.reference_combo.currentIndex())

    def _on_reference_changed(self, _i) -> None:
        self._reset_from(2)
        self._sync()

    def show_reference(self) -> bool:
        spectra = self._spectra_list()
        if not spectra:
            QMessageBox.information(self, "DNP", "Tick the spectra first (step 1).")
            return False
        k = self._reference_index()
        ref, label = spectra[k], self._labels()[k]

        def draw(figure):
            dw.plot_aligned_spectra([ref], labels=[label], mode="overlay", normalise=False,
                                    peak_ppm=self.line_ppm, figure=figure, aligned_axis=False,
                                    cmap="black")
        self._draw(draw, "spectrum")
        return True

    def set_line(self, ppm: float) -> None:
        """The line to align on, on the reference's own ppm axis."""
        self._reset_from(3)
        self.line_ppm = float(ppm)
        self.line_label.setText(f"Line at {ppm:.4g} ppm on the reference. Now align.")
        self.show_reference()
        self._sync()

    def align(self) -> dict | None:
        spectra = self._spectra_list()
        if len(spectra) < 2 or self.line_ppm is None:
            QMessageBox.information(self, "DNP", "Tick at least two spectra and pick the line on "
                                    "the reference first.")
            return None
        try:
            dw.spectrum_fields(spectra)                      # one nucleus
            self.alignment = dw.align_spectra(
                spectra, reference=self._reference_index(), peak_ppm=self.line_ppm,
                window_hz=self.window_spin.value() * 1e3,
                match_halfwidth_hz=self.match_spin.value() or None, labels=self._labels())
        except ValueError as exc:
            QMessageBox.warning(self, "DNP alignment", str(exc))
            return None
        self._reset_from(4)
        self._fill_align_table()
        self.show_aligned()
        self._sync()
        self.changed.emit()
        return self.alignment

    # ------------------------------------------------------------------
    # step 3: check, adjust, confirm
    # ------------------------------------------------------------------

    def _fill_align_table(self) -> None:
        al = self.alignment
        table = self.align_table
        blocked = table.blockSignals(True)
        table.setRowCount(len(al["entries"]) if al else 0)
        for r, e in enumerate(al["entries"] if al else []):
            table.setItem(r, 0, _item(e["label"]))
            table.setItem(r, 1, _item(f"{e['auto_shift_hz']:.1f}", align_right=True))
            table.setItem(r, 2, _item(f"{e['manual_shift_hz']:.1f}", editable=True,
                                      align_right=True))
            score = _item(f"{e['score']:.3f}", align_right=True)
            if not e["found"]:
                score.setForeground(Qt.GlobalColor.red)
            table.setItem(r, 3, score)
            table.setItem(r, 4, _item(e["note"] if e["note"] != "reference" else "(reference)"))
        table.blockSignals(blocked)
        n_missed = sum(1 for e in al["entries"] if not e["found"]) if al else 0
        self.align_status.setText(
            ("Confirmed: every spectrum is on its own ppm axis, its line at "
             f"{al['peak_ppm']:.4g} ppm." if al and al.get("confirmed") else
             (f"{n_missed} spectrum(s) not matched (red): check them and move them by hand. "
              if n_missed else "") + "Check the aligned spectra, then confirm.")
            if al else "")

    def _on_align_table_edited(self, item) -> None:
        if item.column() != 2 or self.alignment is None:
            return
        try:
            value = float(item.text())
        except ValueError:
            self._fill_align_table()
            return
        self.set_manual_shift(item.row(), value)

    def set_manual_shift(self, index: int, shift_hz: float) -> None:
        """A hand-set shift (Hz) for spectrum ``index``; un-confirms."""
        dw.set_manual_shift(self.alignment, index, shift_hz)
        self._reset_from(4)
        self._fill_align_table()
        self.show_aligned()
        self._sync()

    def nudge_ppm(self, index: int, delta_ppm: float) -> None:
        """Move spectrum ``index``'s trace by ``delta_ppm`` on the aligned
        axis (a drag in the alignment window)."""
        e = self.alignment["entries"][index]
        self.set_manual_shift(index, e["manual_shift_hz"]
                              + dw.shift_for_display_ppm(self.alignment, index, delta_ppm))

    def reset_manual(self) -> None:
        if self.alignment is None:
            return
        for i in range(len(self.alignment["entries"])):
            dw.set_manual_shift(self.alignment, i, 0.0)
        self._reset_from(4)
        self._fill_align_table()
        self.show_aligned()
        self._sync()

    def current_aligned(self):
        if self.alignment is None:
            return None
        return dw.aligned_spectra(self._spectra_list(), self.alignment)

    def _xlims(self):
        """A window about the line: +-12 FWHM, or the region / pick."""
        al = self.alignment
        width = al["ref_fwhm_hz"] / (dw.aligned_reference_mhz(al, al["reference"]))
        lo, hi = al["peak_ppm"] - 12 * width, al["peak_ppm"] + 12 * width
        if self.region is not None:
            lo, hi = min(lo, *self.region), max(hi, *self.region)
        return lo, hi

    def show_aligned(self) -> bool:
        if self.alignment is None:
            return False
        aligned = self.aligned if self.aligned is not None else self.current_aligned()
        fields = None
        if self.mode == "sweep":
            try:
                fields = dw.alignment_fields(self._spectra_list(), self.alignment)["field_T"]
            except ValueError:
                fields = None
        mode, normalise = self.view_combo.currentData(), self.normalise_check.isChecked()

        def draw(figure):
            cmap = self._style("spectra").get("compare_colormap") or "viridis"
            dw.plot_aligned_spectra(aligned, labels=self._labels(), mode=mode,
                                    normalise=normalise, peak_ppm=self.peak_ppm or
                                    self.alignment["peak_ppm"], region=self.region,
                                    xlims=self._xlims(), figure=figure, fields=fields, cmap=cmap)
        self._draw(draw, "spectra")
        if self._align_dialog is not None and self._align_dialog.isVisible():
            self._align_dialog.redraw()
        return True

    def open_align_dialog(self) -> None:
        if self.alignment is None:
            return
        if self._align_dialog is None:
            self._align_dialog = DnpAlignDialog(self)
        self._align_dialog.redraw()
        self._align_dialog.show()
        self._align_dialog.raise_()

    def confirm(self) -> None:
        """Step 3 done: the spectra, each on its own ppm axis, are what
        step 4 measures."""
        if self.alignment is None:
            return
        self.alignment["confirmed"] = True
        self.aligned = self.current_aligned()
        self._fill_align_table()
        self.show_aligned()
        self._sync()
        self.changed.emit()

    # ------------------------------------------------------------------
    # step 4: measure
    # ------------------------------------------------------------------

    def set_peak(self, ppm: float) -> None:
        self.peak_ppm = float(ppm)
        self.remeasure()

    def set_region(self, lo: float, hi: float) -> None:
        self.region = (min(lo, hi), max(lo, hi))
        self.remeasure()

    def remeasure(self) -> dict | None:
        if self.aligned is None or (self.peak_ppm is None and self.region is None):
            return None
        try:
            self.measurement = dw.measure(
                self.aligned, peak_ppm=self.peak_ppm, region=self.region,
                baseline=self.baseline_combo.currentData(), correct_rg=self.rg_check.isChecked(),
                reference=self.alignment["reference"], labels=self._labels())
        except ValueError as exc:
            QMessageBox.warning(self, "DNP measurement", str(exc))
            return None
        m = self.measurement
        what = []
        if m["height"] is not None:
            what.append(f"height at {self.peak_ppm:.4g} ppm")
        if m["area"] is not None:
            what.append(f"integral over {self.region[0]:.4g} to {self.region[1]:.4g} ppm")
        text = "Measured " + " and ".join(what) + "."
        if m["height"] is None or m["area"] is None:
            text += " " + _BOTH_HINT
        self.measure_label.setText(text)
        self.show_aligned()
        self._sync()
        if self.mode == "sweep":
            self.plot_profile()
        else:
            self.calculate()
        self.changed.emit()
        return self.measurement

    # ------------------------------------------------------------------
    # step 5
    # ------------------------------------------------------------------

    def _off_index(self) -> int | None:
        i = self.off_combo.currentIndex()
        return i if 0 <= i < len(self._entries) else None

    def plot_profile(self, explicit: bool = False) -> dict | None:
        if self.mode != "sweep" or self.measurement is None:
            if explicit:
                QMessageBox.information(self, "DNP", "Measure first (step 4).")
            return None
        mode = self.profile_mode_combo.currentData()
        off = self._off_index() if mode != "absolute" else None
        try:
            fields = dw.alignment_fields(self._spectra_list(), self.alignment,
                                         source=self.field_source_combo.currentData())
            exclude = [off] if (off is not None and self.exclude_off_check.isChecked()) else []
            self.profile = dw.field_profile(self.measurement, fields, mode=mode, off=off,
                                            exclude=exclude)
        except ValueError as exc:
            if explicit:
                QMessageBox.warning(self, "DNP profile", str(exc))
            return None
        quantity = self.quantity_combo.currentData()
        if quantity != "both" and self.profile.get(quantity) is None:
            quantity = "height" if self.profile.get("height") is not None else "area"
        elif quantity == "both" and (self.profile["height"] is None or self.profile["area"] is None):
            quantity = "height" if self.profile["height"] is not None else "area"
        relative = self.field_axis_combo.currentData()
        nucleus = fields["nucleus"]

        def draw(figure):
            # Read while drawing: the canvas has just taken the "curve"
            # default style, which the first profile must be drawn in.
            colours = (self._style("profile").get("trace_color") or "black", "#d55e00")
            dw.plot_field_profile(self.profile, quantity=quantity, relative=relative,
                                  figure=figure, nucleus=nucleus, colors=colours)
        self._draw(draw, "profile")
        report = dw.profile_report(self.profile)
        extra = ""
        if fields["difference_mT"] is not None:
            worst = float(np.nanmax(np.abs(fields["difference_mT"])))
            extra = (f"\nB₀ at the line vs from SFO1: up to {worst:.3f} mT apart"
                     + (" -- worth using the line." if worst > 0.05 else "."))
        # The panel says what was plotted; the per-spectrum numbers are in
        # the tooltip, the CSV and the kept record (too long for the side).
        n = len(self.profile["labels"])
        first = report.splitlines()[0]
        self.result_label.setText(f"{first} {n} spectra plotted on the Fit view."
                                  + extra + "\n(Hover for every value; Export CSV for a table.)")
        self.result_label.setToolTip(report + extra)
        self._sync()
        self._keep("dnp_sweep", "DNP field profile", report + extra, self.profile)
        return self.profile

    def export_csv(self) -> None:
        if self.profile is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export DNP profile", "dnp_profile.csv",
                                              "CSV files (*.csv)")
        if path:
            dw.profile_to_csv(self.profile, path)

    def calculate(self, explicit: bool = False) -> list | None:
        if self.mode != "enhancement" or self.measurement is None:
            if explicit:
                QMessageBox.information(self, "DNP", "Measure first (step 4).")
            return None
        off = self._off_index()
        if off is None or len(self._entries) < 2:
            return None
        self.enhancement_rows = dw.enhancement_table(self.measurement, off=off)
        key = self.definition_combo.currentData()
        table = self.enh_table
        table.setRowCount(len(self.enhancement_rows))
        for r, row in enumerate(self.enhancement_rows):
            table.setItem(r, 0, _item(row["label"]))
            for c, q in ((1, row["height"]), (2, row["area"])):
                table.setItem(r, c, _item(_pm(q[key], q["error"]) if q else "—", align_right=True))
            ag = row["agreement"]
            table.setItem(r, 3, _item("—" if ag is None else ("yes" if ag["consistent"] else "NO")))
        report = dw.enhancement_report(self.enhancement_rows)
        self.result_label.setText(report)
        self._keep("dnp_enhancement", "DNP enhancement", report, self.enhancement_rows)
        self._sync()
        return self.enhancement_rows

    # ------------------------------------------------------------------
    # keeping results
    # ------------------------------------------------------------------

    def inputs_text(self) -> list[str]:
        lines = [f"Spectra: {', '.join(self._labels())}"]
        if self.alignment is not None:
            lines.append(dw.alignment_report(self.alignment))
        if self.measurement is not None:
            lines.append(dw.measurement_report(self.measurement))
        return lines

    def _keep(self, kind, label, report, result) -> None:
        indices = [e["index"] for e in self._entries if e.get("index") is not None]
        if not indices:
            return
        record = {"label": label, "target": ", ".join(self._labels()),
                  "inputs_text": self.inputs_text(), "report": report,
                  "result": {"result": result, "alignment": self.alignment}}
        try:
            self._store(kind, record, indices)
        except Exception:  # noqa: BLE001 -- keeping is best-effort; the result is on screen
            pass


class DnpAlignDialog(QDialog):
    """Drag the aligned traces left/right (explicit user request): press
    on a trace, drag, release -- the move becomes that spectrum's hand
    shift. Stacked or superimposed, like Compare."""

    def __init__(self, panel: DnpWorkflowPanel) -> None:
        super().__init__(panel)
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
        from matplotlib.figure import Figure

        self.panel = panel
        self.setWindowTitle("DNP: adjust the alignment")
        self.resize(820, 620)
        v = QVBoxLayout(self)
        hint = QLabel("Press on a trace and drag it left or right onto the others; release to "
                      "keep the move. Zoom with the toolbar (switch it off again to drag).")
        hint.setWordWrap(True)
        v.addWidget(hint)
        row = QHBoxLayout()
        self.view_combo = QComboBox()
        self.view_combo.addItem("Stacked", "stack")
        self.view_combo.addItem("Superimposed", "overlay")
        self.view_combo.setCurrentIndex(self.view_combo.findData(panel.view_combo.currentData()))
        self.view_combo.currentIndexChanged.connect(lambda _i: self.redraw())
        row.addWidget(QLabel("View:"))
        row.addWidget(self.view_combo)
        self.reset_button = QPushButton("Reset Hand Shifts")
        self.reset_button.clicked.connect(panel.reset_manual)
        self.confirm_button = QPushButton("Confirm Alignment")
        self.confirm_button.clicked.connect(self._confirm)
        row.addStretch(1)
        row.addWidget(self.reset_button)
        row.addWidget(self.confirm_button)
        v.addLayout(row)
        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        v.addWidget(self.toolbar)
        v.addWidget(self.canvas, 1)
        self.status = QLabel("")
        v.addWidget(self.status)
        self.lines = []
        self._drag = None               # (index, x0 in data, original xdata)
        self._selected: int | None = None
        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.canvas.mpl_connect("button_release_event", self._on_release)

    def redraw(self) -> None:
        panel = self.panel
        if panel.alignment is None:
            return
        aligned = panel.current_aligned()
        xlim = None
        if self.figure.axes and self.lines:
            xlim = self.figure.axes[0].get_xlim()
        _fig, self.lines = dw.plot_aligned_spectra(
            aligned, labels=panel._labels(), mode=self.view_combo.currentData(), normalise=True,
            peak_ppm=panel.alignment["peak_ppm"], xlims=panel._xlims(), figure=self.figure,
            highlight=self._selected)
        if xlim is not None:
            self.figure.axes[0].set_xlim(xlim)
        self.canvas.draw_idle()
        self.status.setText("Confirmed." if panel.alignment.get("confirmed") else
                            "Not confirmed yet.")

    def _confirm(self) -> None:
        self.panel.confirm()
        self.redraw()

    # -- dragging (the handlers take data coordinates, so tests can drive them)

    def nearest_trace(self, x: float, y: float) -> int | None:
        """The trace under (x, y): nearest in y at that x."""
        best, best_d = None, math.inf
        for line in self.lines:
            xs, ys = np.asarray(line.get_xdata(), float), np.asarray(line.get_ydata(), float)
            order = np.argsort(xs)
            yy = float(np.interp(x, xs[order], ys[order]))
            d = abs(yy - y)
            if d < best_d:
                best, best_d = line.dnp_index, d
        return best

    def press(self, x: float, y: float) -> None:
        index = self.nearest_trace(x, y)
        if index is None:
            return
        line = self.lines[index]
        self._drag = (index, float(x), np.asarray(line.get_xdata(), dtype=float).copy())
        self._selected = index
        line.set_linewidth(1.8)
        self.canvas.draw_idle()

    def motion(self, x: float) -> None:
        if self._drag is None:
            return
        index, x0, original = self._drag
        self.lines[index].set_xdata(original + (float(x) - x0))
        self.canvas.draw_idle()

    def release(self, x: float) -> None:
        if self._drag is None:
            return
        index, x0, _original = self._drag
        self._drag = None
        delta = float(x) - x0
        if delta:
            self.panel.nudge_ppm(index, delta)
            self.status.setText(f"{self.panel._labels()[index]} moved {delta:+.3g} ppm.")
        self.redraw()

    def _on_press(self, event) -> None:
        if event.inaxes is None or event.button != 1 or self.toolbar.mode:
            return
        self.press(event.xdata, event.ydata)

    def _on_motion(self, event) -> None:
        if event.inaxes is not None and event.xdata is not None:
            self.motion(event.xdata)

    def _on_release(self, event) -> None:
        if self._drag is not None:
            x = event.xdata if event.xdata is not None else self._drag[1]
            self.release(x)
