"""
Compare > DNP: enhancement factors and field-sweep profiles.

The maths is :mod:`opennmr.notebook.dnp`, called exactly as a notebook
calls it; the plots are that module's own plot functions drawn onto the
GUI's canvases (``MplCanvas.show_external_plot``), so a figure made here
and one made in a notebook are the same figure.

**Enhancement** -- spectrum 1 / spectrum 2 (microwaves on / off) by the
height of a picked peak *and* by the integral of a dragged region. Both
are encouraged, neither is forced: until both are measured the panel
says which is missing and why it matters; when both are, it says
whether they agree.

**Field sweep** -- every spectrum checked in the Compare list, shown
against its offset from its own carrier (where the line stays near
zero while SFO1 follows the field). Pick the line or drag a region once;
the line is found in every spectrum, and its intensity/area plotted
against B0 on the Fit view, with the table, notes and CSV export.

Picks and drags are made on the Compare view. The panel arms the
canvas's own pick / region-select mode for one gesture at a time.
"""

from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QGroupBox, QHBoxLayout, QHeaderView, QLabel, QMessageBox, QPushButton, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

from ..notebook import dnp

#: What an armed canvas gesture is for.
ENH_PEAK, ENH_REGION, SWEEP_PEAK, SWEEP_REGION = "enh_peak", "enh_region", "sweep_peak", "sweep_region"

_TABLE = (("label", "Spectrum"), ("field_T", "B₀ (T)"), ("delta_B_mT", "ΔB₀ (mT)"),
          ("field_source", "Field from"), ("found", "Line found"), ("snr", "S/N"),
          ("sfo1_error_mT", "SFO1 − line (mT)"), ("intensity", "Intensity"), ("area", "Area"))

_BASELINE_ITEMS = (
    ("Local (recommended)", "local",
     "A quadratic fitted just outside each peak (±5 FWHM) or region: follows a rolling "
     "baseline."),
    ("Global (as Quantify)", "global",
     "One robust straight line fitted to the spectrum's edges, as Quantify's integration uses."),
    ("None", "none", "The data as they are."))


def _baseline_combo() -> QComboBox:
    combo = QComboBox()
    for text, data, tip in _BASELINE_ITEMS:
        combo.addItem(text, data)
        combo.setItemData(combo.count() - 1, tip, Qt.ItemDataRole.ToolTipRole)
    combo.setToolTip("\n".join(f"{t}: {tip}" for t, _d, tip in _BASELINE_ITEMS))
    return combo


_BOTH_HINT = ("Measure both: pick the peak <i>and</i> drag a region. The height and area ratios "
              "agree only if the line shape is unchanged, so doing both checks the result.")


def _fmt_cell(key, value) -> str:
    if value is None:
        return ""
    if key == "found":
        return "yes" if value else "NO"
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(value):
            return "—"
        if key == "field_T":
            return f"{value:.6f}"
        if key in ("delta_B_mT", "sfo1_error_mT"):
            return f"{value:.3f}"
        if key == "snr":
            return f"{value:.0f}"
        return f"{value:.5g}"
    return str(value)


class DnpPanel(QWidget):
    """Callbacks (all from MainWindow):

    ``entries()`` -> ``[{"label", "spectrum", "index"}]``: the 1D spectra
    checked in the Compare list, in list order (``index`` is the session
    row, or None for a slice of a pseudo-2D, which cannot hold a record);
    ``draw_spectra(draw)`` / ``draw_profile(draw)``: show a
    ``draw(figure)`` plot on the Compare / Fit view;
    ``arm(mode)``: turn the Compare view's pick (``"peak"``), region
    (``"region"``) or neither (``None``) mode on;
    ``store(kind, record, indices)``: keep a result with those spectra,
    so it is logged and saved;
    ``style(which)``: the current plot style (``MplCanvas.style_settings``)
    of the "spectra" or "profile" view, read while drawing so the
    colours follow Plot Style and the saved defaults."""

    def __init__(self, entries, draw_spectra, draw_profile, arm, store, style=None,
                 parent=None) -> None:
        super().__init__(parent)
        self._entries = entries
        self._style = style or (lambda _which: {})
        self._draw_spectra = draw_spectra
        self._draw_profile = draw_profile
        self._arm = arm
        self._store = store
        self._armed: str | None = None
        self.enh_peak: float | None = None
        self.enh_region: tuple[float, float] | None = None
        self.enh_result: dict | None = None
        self.sweep_peak: float | None = None          # Hz from the carrier
        self.sweep_region: tuple[float, float] | None = None
        self.sweep_result: dict | None = None
        self._sweep_entries: list[dict] = []

        # Two independent pages, one per measurement, so MainWindow can
        # give each its own collapsible section ("DNP Enhancement",
        # "DNP Field Sweep Profile" -- explicit user request). Until
        # something takes them, they sit in this panel as before, so
        # the panel still works standalone.
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        self.enhancement_page = self._page(self._build_enhancement())
        self.sweep_page = self._page(self._build_sweep())
        v.addWidget(self.enhancement_page)
        v.addWidget(self.sweep_page)
        self.refresh()

    @staticmethod
    def _page(box: QWidget) -> QWidget:
        page = QWidget()
        v = QVBoxLayout(page)
        v.setContentsMargins(0, 0, 0, 0)
        blurb = QLabel("Uses the spectra selected in the 'Compare' list above. Peak picking and region selection/integration "
                       "are made on the 'Compare' view.")
        blurb.setWordWrap(True)
        v.addWidget(blurb)
        v.addWidget(box)
        return page

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------

    def _build_enhancement(self) -> QGroupBox:
        box = QGroupBox("Enhancement factor  ε = spectrum 1 / spectrum 2")
        v = QVBoxLayout(box)
        form = QFormLayout()
        self.enh_combo_1 = QComboBox()
        self.enh_combo_1.setToolTip("Normally the microwaves-on spectrum.")
        self.enh_combo_2 = QComboBox()
        self.enh_combo_2.setToolTip("Normally the microwaves-off spectrum.")
        for combo in (self.enh_combo_1, self.enh_combo_2):
            combo.currentIndexChanged.connect(lambda _i: self._enh_inputs_changed())
        form.addRow("Spectrum 1 (µw on):", self.enh_combo_1)
        form.addRow("Spectrum 2 (µw off):", self.enh_combo_2)
        self.enh_snap_spin = QDoubleSpinBox()
        self.enh_snap_spin.setRange(0.0, 1e4)
        self.enh_snap_spin.setDecimals(2)
        self.enh_snap_spin.setSuffix(" ppm")
        self.enh_snap_spin.setSpecialValueText("auto (1% of width)")
        self.enh_snap_spin.setToolTip(
            "The height is the largest signal within ± this of the click, in each spectrum -- "
            "a click need not land exactly on the maximum.")
        self.enh_snap_spin.valueChanged.connect(lambda _v: self._enh_inputs_changed())
        form.addRow("Snap to maximum within:", self.enh_snap_spin)
        self.enh_baseline_combo = _baseline_combo()
        self.enh_baseline_combo.currentIndexChanged.connect(lambda _i: self._enh_inputs_changed())
        form.addRow("Baseline:", self.enh_baseline_combo)
        v.addLayout(form)

        row = QHBoxLayout()
        self.enh_show_button = QPushButton("Show Both")
        self.enh_show_button.setToolTip("Draw both spectra on the Compare view, to pick on.")
        self.enh_show_button.clicked.connect(self.show_enhancement_spectra)
        self.enh_pick_button = QPushButton("Pick Peak")
        self.enh_pick_button.setCheckable(True)
        self.enh_pick_button.setToolTip("Then click the peak on the Compare view.")
        self.enh_pick_button.toggled.connect(lambda on: self._arm_for(ENH_PEAK, on))
        self.enh_region_button = QPushButton("Select Region")
        self.enh_region_button.setCheckable(True)
        self.enh_region_button.setToolTip("Then drag across the peak on the Compare view.")
        self.enh_region_button.toggled.connect(lambda on: self._arm_for(ENH_REGION, on))
        for b in (self.enh_show_button, self.enh_pick_button, self.enh_region_button):
            row.addWidget(b)
        v.addLayout(row)

        self.enh_picks_label = QLabel("")
        self.enh_picks_label.setWordWrap(True)
        v.addWidget(self.enh_picks_label)
        opts = QHBoxLayout()
        self.enh_rg_check = QCheckBox("Correct receiver gain")
        self.enh_rg_check.setChecked(True)
        self.enh_rg_check.setToolTip("Data are already per scan; this also removes RG differences.")
        self.enh_rg_check.toggled.connect(lambda _on: self._enh_inputs_changed())
        opts.addWidget(self.enh_rg_check)
        clear = QPushButton("Clear")
        clear.setToolTip("Forget the picked peak and region.")
        clear.clicked.connect(self.clear_enhancement)
        opts.addWidget(clear)
        v.addLayout(opts)

        self.enh_result_label = QLabel(_BOTH_HINT)
        self.enh_result_label.setWordWrap(True)
        self.enh_result_label.setTextFormat(Qt.TextFormat.RichText)
        self.enh_result_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        v.addWidget(self.enh_result_label)
        return box

    def _build_sweep(self) -> QGroupBox:
        box = QGroupBox("Field sweep: signal against B₀")
        v = QVBoxLayout(box)
        intro = QLabel(
            "Each spectrum is shown against its offset from its own carrier (SFO1), where the "
            "line stays near zero as the field and SFO1 change. Pick the line or drag a region "
            "once: the line is found in every spectrum, and B₀ is taken from its frequency and "
            "the nucleus's gyromagnetic ratio.")
        intro.setWordWrap(True)
        v.addWidget(intro)
        self.sweep_status_label = QLabel("")
        self.sweep_status_label.setWordWrap(True)
        v.addWidget(self.sweep_status_label)

        form = QFormLayout()
        self.sweep_reference_combo = QComboBox()
        self.sweep_reference_combo.setToolTip(
            "Sets the line's chemical shift (read off this spectrum's ppm axis), the receiver "
            "gain everything is scaled to, and ΔB0 = 0.")
        form.addRow("Reference spectrum:", self.sweep_reference_combo)
        self.sweep_window_spin = QDoubleSpinBox()
        self.sweep_window_spin.setRange(0.1, 1000.0)
        self.sweep_window_spin.setDecimals(1)
        self.sweep_window_spin.setValue(dnp.DEFAULT_WINDOW_HZ / 1e3)
        self.sweep_window_spin.setSuffix(" kHz")
        self.sweep_window_spin.setToolTip(
            "Look for the line within ± this of the pick in every spectrum. It must cover how "
            "far SFO1 missed the line (up to 14 kHz in the first real sweep), and exclude other "
            "resonances.")
        form.addRow("Search window ±:", self.sweep_window_spin)
        self.sweep_snr_spin = QDoubleSpinBox()
        self.sweep_snr_spin.setRange(1.0, 1000.0)
        self.sweep_snr_spin.setValue(dnp.DEFAULT_MIN_SNR)
        self.sweep_snr_spin.setToolTip(
            "A line weaker than this (noise standard deviations) is 'not found': its value is "
            "read at the pick and its field comes from SFO1, rather than trusting a noise "
            "maximum.")
        form.addRow("Minimum S/N:", self.sweep_snr_spin)
        self.sweep_field_combo = QComboBox()
        self.sweep_field_combo.addItem("Line position (recommended)", "peak")
        self.sweep_field_combo.addItem("SFO1 (assumes the line on the carrier)", "sfo1")
        self.sweep_field_combo.setToolTip(
            "Both are always in the table. SFO1 is set by hand and misses the line; the line's "
            "own frequency is the direct measurement of B0.")
        form.addRow("B₀ from:", self.sweep_field_combo)
        shift_row = QHBoxLayout()
        self.sweep_shift_auto = QCheckBox("from reference")
        self.sweep_shift_auto.setChecked(True)
        self.sweep_shift_spin = QDoubleSpinBox()
        self.sweep_shift_spin.setRange(-1e5, 1e5)
        self.sweep_shift_spin.setDecimals(3)
        self.sweep_shift_spin.setSuffix(" ppm")
        self.sweep_shift_spin.setEnabled(False)
        self.sweep_shift_auto.toggled.connect(lambda on: self.sweep_shift_spin.setEnabled(not on))
        shift_row.addWidget(self.sweep_shift_auto)
        shift_row.addWidget(self.sweep_shift_spin)
        shift_widget = QWidget()
        shift_widget.setLayout(shift_row)
        shift_row.setContentsMargins(0, 0, 0, 0)
        shift_widget.setToolTip(
            "ν = γ B0 (1 + δ): the absolute field needs the line's shift. 1 ppm moves every field "
            "by 1e-6 B0; the profile's shape and field differences are unaffected.")
        form.addRow("Line shift δ:", shift_widget)
        self.sweep_baseline_combo = _baseline_combo()
        form.addRow("Baseline:", self.sweep_baseline_combo)
        v.addLayout(form)

        opts = QHBoxLayout()
        self.sweep_track_check = QCheckBox("Track the line")
        self.sweep_track_check.setChecked(True)
        self.sweep_track_check.setToolTip(
            "Find the line in every spectrum and move the region with it. Off: read every "
            "spectrum at the same carrier offset.")
        self.sweep_rg_check = QCheckBox("Correct receiver gain")
        self.sweep_rg_check.setChecked(True)
        for c in (self.sweep_track_check, self.sweep_rg_check):
            opts.addWidget(c)
        v.addLayout(opts)

        row = QHBoxLayout()
        self.sweep_show_button = QPushButton("Show Spectra")
        self.sweep_show_button.setToolTip(
            "Draw every checked spectrum against its carrier offset on the Compare view.")
        self.sweep_show_button.clicked.connect(self.show_sweep_spectra)
        self.sweep_pick_button = QPushButton("Pick Line")
        self.sweep_pick_button.setCheckable(True)
        self.sweep_pick_button.toggled.connect(lambda on: self._arm_for(SWEEP_PEAK, on))
        self.sweep_region_button = QPushButton("Select Region")
        self.sweep_region_button.setCheckable(True)
        self.sweep_region_button.toggled.connect(lambda on: self._arm_for(SWEEP_REGION, on))
        for b in (self.sweep_show_button, self.sweep_pick_button, self.sweep_region_button):
            row.addWidget(b)
        v.addLayout(row)
        self.sweep_picks_label = QLabel("")
        self.sweep_picks_label.setWordWrap(True)
        v.addWidget(self.sweep_picks_label)

        plot_form = QFormLayout()
        self.sweep_quantity_combo = QComboBox()
        self.sweep_quantity_combo.addItem("Intensity", "intensity")
        self.sweep_quantity_combo.addItem("Area", "area")
        self.sweep_quantity_combo.addItem("Both", "both")
        plot_form.addRow("Plot:", self.sweep_quantity_combo)
        self.sweep_normalise_combo = QComboBox()
        self.sweep_normalise_combo.setToolTip(
            "Divide by one spectrum (e.g. microwaves off) for an enhancement profile, or by "
            "the largest value.")
        plot_form.addRow("Divide by:", self.sweep_normalise_combo)
        self.sweep_axis_combo = QComboBox()
        self.sweep_axis_combo.addItem("B₀ (mT)", ("mT", False))
        self.sweep_axis_combo.addItem("B₀ (T)", ("T", False))
        self.sweep_axis_combo.addItem("ΔB₀ from reference (mT)", ("mT", True))
        plot_form.addRow("Field axis:", self.sweep_axis_combo)
        for combo in (self.sweep_quantity_combo, self.sweep_normalise_combo, self.sweep_axis_combo):
            combo.currentIndexChanged.connect(lambda _i: self.draw_profile())
        v.addLayout(plot_form)

        actions = QHBoxLayout()
        self.sweep_run_button = QPushButton("Track && Plot")
        self.sweep_run_button.setToolTip("Measure every checked spectrum and plot the profile on "
                                         "the Fit view.")
        self.sweep_run_button.clicked.connect(self.run_sweep)
        self.sweep_tracked_button = QPushButton("Show Tracked Lines")
        self.sweep_tracked_button.setToolTip(
            "The spectra with where the line was found in each (Compare view) -- check the "
            "tracking before trusting the profile.")
        self.sweep_tracked_button.clicked.connect(lambda: self.show_sweep_spectra(tracked=True))
        self.sweep_aligned_check = QCheckBox("aligned")
        self.sweep_aligned_check.setToolTip("Shift each spectrum so the tracked lines coincide.")
        self.sweep_csv_button = QPushButton("Export CSV…")
        self.sweep_csv_button.clicked.connect(self.export_csv)
        for w in (self.sweep_run_button, self.sweep_tracked_button, self.sweep_aligned_check,
                  self.sweep_csv_button):
            actions.addWidget(w)
        v.addLayout(actions)

        self.sweep_table = QTableWidget(0, len(_TABLE))
        self.sweep_table.setHorizontalHeaderLabels([t for _k, t in _TABLE])
        self.sweep_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents)
        self.sweep_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.sweep_table.setMinimumHeight(140)
        v.addWidget(self.sweep_table)
        self.sweep_report_label = QLabel("")
        self.sweep_report_label.setWordWrap(True)
        self.sweep_report_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        v.addWidget(self.sweep_report_label)
        self._update_sweep_buttons()
        return box

    # ------------------------------------------------------------------
    # spectra
    # ------------------------------------------------------------------

    def refresh(self) -> None:
        """Re-read the Compare list's checked spectra into the combos,
        keeping each combo's choice where that spectrum is still there."""
        entries = self._entries()
        labels = [e["label"] for e in entries]
        for combo, default in ((self.enh_combo_1, 0), (self.enh_combo_2, 1),
                               (self.sweep_reference_combo, 0)):
            keep = combo.currentText()
            blocked = combo.blockSignals(True)
            combo.clear()
            combo.addItems(labels)
            if keep in labels:
                combo.setCurrentIndex(labels.index(keep))
            elif len(labels) > default:
                combo.setCurrentIndex(default)
            combo.blockSignals(blocked)
        keep = self.sweep_normalise_combo.currentData()
        blocked = self.sweep_normalise_combo.blockSignals(True)
        self.sweep_normalise_combo.clear()
        self.sweep_normalise_combo.addItem("Nothing", None)
        self.sweep_normalise_combo.addItem("Largest value", "max")
        for label in labels:
            self.sweep_normalise_combo.addItem(label, ("label", label))
        found = self.sweep_normalise_combo.findData(keep)
        self.sweep_normalise_combo.setCurrentIndex(max(found, 0))
        self.sweep_normalise_combo.blockSignals(blocked)
        nuclei = set()
        for e in entries:
            try:
                nuclei.add(dnp._nucleus(e["spectrum"]))
            except ValueError:
                nuclei.add("?")
        if not entries:
            self.sweep_status_label.setText("Check the spectra of the sweep in the Compare list.")
        else:
            self.sweep_status_label.setText(
                f"{len(entries)} spectra checked" + (f", {', '.join(sorted(nuclei))}" if nuclei
                                                     else "")
                + (" -- a sweep must be one nucleus." if len(nuclei) > 1 else "."))

    def _entry(self, combo: QComboBox) -> dict | None:
        entries = self._entries()
        i = combo.currentIndex()
        label = combo.currentText()
        for e in entries:
            if e["label"] == label:
                return e
        return entries[i] if 0 <= i < len(entries) else None

    # ------------------------------------------------------------------
    # canvas gestures
    # ------------------------------------------------------------------

    def _buttons(self):
        return {ENH_PEAK: self.enh_pick_button, ENH_REGION: self.enh_region_button,
                SWEEP_PEAK: self.sweep_pick_button, SWEEP_REGION: self.sweep_region_button}

    def _arm_for(self, mode: str, on: bool) -> None:
        if on:
            for other, button in self._buttons().items():
                if other != mode and button.isChecked():
                    blocked = button.blockSignals(True)
                    button.setChecked(False)
                    button.blockSignals(blocked)
            # Draw the view the gesture belongs to, so the click lands on
            # the right axis (ppm for enhancement, carrier offset for a sweep).
            if mode in (ENH_PEAK, ENH_REGION):
                if not self.show_enhancement_spectra():
                    self._disarm()
                    return
            elif not self.show_sweep_spectra(tracked=self.sweep_result is not None):
                self._disarm()
                return
            self._armed = mode
            self._arm("peak" if mode in (ENH_PEAK, SWEEP_PEAK) else "region")
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

    def canvas_mode_released(self) -> None:
        """The Compare view's pick/region mode was turned off from
        elsewhere (another mode took over): un-press our buttons."""
        if self._armed is not None:
            self._armed = None
            for button in self._buttons().values():
                blocked = button.blockSignals(True)
                button.setChecked(False)
                button.blockSignals(blocked)

    def on_canvas_pick(self, x: float) -> None:
        mode = self._armed
        if mode == ENH_PEAK:
            self._disarm()
            self.enh_peak = float(x)
            self.compute_enhancement()
        elif mode == SWEEP_PEAK:
            self._disarm()
            self.sweep_peak = float(x) * 1e3           # the view is in kHz
            self._update_sweep_buttons()
            self.show_sweep_spectra(tracked=self.sweep_result is not None)

    def on_canvas_region(self, lo: float, hi: float) -> None:
        mode = self._armed
        if mode == ENH_REGION:
            self._disarm()
            self.enh_region = (float(lo), float(hi))
            self.compute_enhancement()
        elif mode == SWEEP_REGION:
            self._disarm()
            self.sweep_region = (float(lo) * 1e3, float(hi) * 1e3)
            self._update_sweep_buttons()
            self.show_sweep_spectra(tracked=self.sweep_result is not None)

    # ------------------------------------------------------------------
    # enhancement
    # ------------------------------------------------------------------

    def _enh_pair(self):
        e1, e2 = self._entry(self.enh_combo_1), self._entry(self.enh_combo_2)
        if e1 is None or e2 is None:
            return None
        return e1, e2

    def show_enhancement_spectra(self) -> bool:
        pair = self._enh_pair()
        if pair is None:
            QMessageBox.information(self, "DNP enhancement",
                                    "Check at least two 1D spectra in the Compare list first.")
            return False
        e1, e2 = pair
        result = self.enh_result

        def draw(figure):
            dnp.plot_enhancement_spectra(e1["spectrum"], e2["spectrum"], result=result,
                                         labels=(e1["label"], e2["label"]), figure=figure)
        self._draw_spectra(draw)
        return True

    def _enh_inputs_changed(self) -> None:
        if self.enh_peak is not None or self.enh_region is not None:
            self.compute_enhancement()

    def clear_enhancement(self) -> None:
        self.enh_peak = self.enh_region = self.enh_result = None
        self.enh_picks_label.setText("")
        self.enh_result_label.setText(_BOTH_HINT)

    def compute_enhancement(self) -> dict | None:
        pair = self._enh_pair()
        picks = []
        if self.enh_peak is not None:
            picks.append(f"peak at {self.enh_peak:.4g} ppm")
        if self.enh_region is not None:
            picks.append(f"region {self.enh_region[0]:.4g} to {self.enh_region[1]:.4g} ppm")
        self.enh_picks_label.setText("Measured: " + ", ".join(picks) if picks else "")
        if pair is None or (self.enh_peak is None and self.enh_region is None):
            return None
        e1, e2 = pair
        if e1["label"] == e2["label"]:
            self.enh_result_label.setText("Choose two different spectra.")
            return None
        snap = self.enh_snap_spin.value() or None
        try:
            result = dnp.enhancement(
                e1["spectrum"], e2["spectrum"], peak=self.enh_peak, region=self.enh_region,
                unit="ppm", snap=snap, baseline=self.enh_baseline_combo.currentData(),
                correct_rg=self.enh_rg_check.isChecked(), labels=(e1["label"], e2["label"]))
        except Exception as exc:  # noqa: BLE001 -- surfaced to the user
            self.enh_result_label.setText(f"<b>Could not measure:</b> {exc}")
            return None
        self.enh_result = result
        self.enh_result_label.setText(self._enhancement_html(result))
        self.show_enhancement_spectra()
        indices = [e["index"] for e in (e1, e2) if e["index"] is not None]
        self._store("dnp_enhancement", {
            "label": "DNP enhancement", "target": f"{e1['label']} / {e2['label']}",
            "inputs_text": self._enhancement_inputs(), "report": dnp.enhancement_report(result),
            "result": result}, indices)
        return result

    def _enhancement_inputs(self) -> list[str]:
        return [f"Spectrum 1: {self.enh_combo_1.currentText()}",
                f"Spectrum 2: {self.enh_combo_2.currentText()}",
                f"Peak: {self.enh_peak} ppm", f"Region: {self.enh_region} ppm",
                f"Snap: {self.enh_snap_spin.text()}",
                f"Baseline: {self.enh_baseline_combo.currentText()}",
                f"Correct receiver gain: {self.enh_rg_check.isChecked()}"]

    @staticmethod
    def _enhancement_html(result: dict) -> str:
        parts = []
        i, a = result["intensity"], result["area"]
        if i:
            parts.append(f"<b>ε (height) = {dnp._fmt_sys(i['ratio'], i['ratio_error'], i['ratio_systematic'])}</b>")
        if a:
            parts.append(f"<b>ε (area) = {dnp._fmt_sys(a['ratio'], a['ratio_error'], a['ratio_systematic'])}</b>")
        ag = result["agreement"]
        if ag:
            parts.append(("Height and area agree" if ag["consistent"] else
                          "<span style='color:#b03a2e'>Height and area disagree</span>")
                         + f" ({ag['percent']:.1f}% apart).")
        parts.append("<span style='color:gray'>± noise; 'sys': change when the peak's baseline "
                     "window or the region is varied (see the report)</span>")
        for note in result["notes"]:
            parts.append(f"<i>{note}</i>")
        return "<br>".join(parts)

    # ------------------------------------------------------------------
    # field sweep
    # ------------------------------------------------------------------

    def _update_sweep_buttons(self) -> None:
        picks = []
        if self.sweep_peak is not None:
            picks.append(f"line at {self.sweep_peak / 1e3:.3f} kHz")
        if self.sweep_region is not None:
            picks.append(f"region {self.sweep_region[0] / 1e3:.3f} to "
                         f"{self.sweep_region[1] / 1e3:.3f} kHz")
        self.sweep_picks_label.setText(
            ("Picked: " + ", ".join(picks) + " from the carrier") if picks else
            "Pick the line and/or drag a region on the carrier-offset view.")
        ready = bool(picks)
        self.sweep_run_button.setEnabled(ready)
        has = self.sweep_result is not None
        self.sweep_tracked_button.setEnabled(has)
        self.sweep_csv_button.setEnabled(has)

    def show_sweep_spectra(self, tracked: bool = False) -> bool:
        entries = self._entries()
        if not entries:
            QMessageBox.information(self, "DNP field sweep",
                                    "Check the spectra of the sweep in the Compare list first.")
            return False
        result = self.sweep_result if tracked else None
        if result is not None and [e["label"] for e in self._sweep_entries] != [
                e["label"] for e in entries]:
            result = None                   # the checked spectra changed since the sweep ran
        spectra = [e["spectrum"] for e in entries]
        window = self.sweep_window_spin.value()
        centre = (self.sweep_peak if self.sweep_peak is not None else
                  (0.5 * sum(self.sweep_region) if self.sweep_region else 0.0)) / 1e3
        aligned = self.sweep_aligned_check.isChecked() and result is not None
        try:
            fields = [dnp.field_from_sfo1(s) for s in spectra]
        except ValueError:
            fields = None

        def draw(figure):
            cmap = self._style("spectra").get("compare_colormap") or "viridis"
            if cmap == "black":
                cmap = "viridis"        # one colour cannot show which spectrum is which field
            dnp.plot_sweep_spectra(spectra, result=result, aligned=aligned, figure=figure,
                                   cmap=cmap,
                                   xlim_khz=(centre - 1.5 * window, centre + 1.5 * window))
            ax = figure.axes[0]
            if result is None:
                if self.sweep_peak is not None:
                    ax.axvline(self.sweep_peak / 1e3, color="0.4", linewidth=0.8, linestyle="--")
                if self.sweep_region is not None:
                    ax.axvspan(self.sweep_region[0] / 1e3, self.sweep_region[1] / 1e3,
                               color="0.88", zorder=0)
        if fields is None:
            QMessageBox.information(self, "DNP field sweep",
                                    "A checked spectrum records no nucleus or carrier frequency.")
            return False
        self._draw_spectra(draw)
        return True

    def _normalise_choice(self, labels):
        data = self.sweep_normalise_combo.currentData()
        if isinstance(data, tuple) and data[0] == "label":
            return labels.index(data[1]) if data[1] in labels else None
        return data

    def run_sweep(self) -> dict | None:
        entries = self._entries()
        if len(entries) < 2:
            QMessageBox.information(self, "DNP field sweep",
                                    "Check at least two spectra of the sweep in the Compare list.")
            return None
        if self.sweep_peak is None and self.sweep_region is None:
            QMessageBox.information(self, "DNP field sweep",
                                    "Pick the line or drag a region on the carrier-offset view "
                                    "first (Show Spectra, then Pick Line / Select Region).")
            return None
        labels = [e["label"] for e in entries]
        ref_label = self.sweep_reference_combo.currentText()
        reference = labels.index(ref_label) if ref_label in labels else 0
        try:
            result = dnp.field_sweep(
                [e["spectrum"] for e in entries], peak=self.sweep_peak, region=self.sweep_region,
                reference=reference, window=self.sweep_window_spin.value() * 1e3,
                track=self.sweep_track_check.isChecked(), min_snr=self.sweep_snr_spin.value(),
                field_from=self.sweep_field_combo.currentData(),
                shift_ppm=None if self.sweep_shift_auto.isChecked() else self.sweep_shift_spin.value(),
                baseline=self.sweep_baseline_combo.currentData(),
                correct_rg=self.sweep_rg_check.isChecked(), labels=labels)
        except Exception as exc:  # noqa: BLE001 -- surfaced to the user
            QMessageBox.critical(self, "DNP field sweep", str(exc))
            return None
        self.sweep_result = result
        self._sweep_entries = entries
        if self.sweep_shift_auto.isChecked():
            blocked = self.sweep_shift_spin.blockSignals(True)
            self.sweep_shift_spin.setValue(result["shift_ppm"])
            self.sweep_shift_spin.blockSignals(blocked)
        if self.sweep_region is None and self.sweep_quantity_combo.currentData() != "intensity":
            self.sweep_quantity_combo.setCurrentIndex(0)
        elif self.sweep_peak is None and self.sweep_quantity_combo.currentData() != "area":
            self.sweep_quantity_combo.setCurrentIndex(1)
        self._fill_table(result)
        self.sweep_report_label.setText(dnp.field_sweep_report(result))
        self._update_sweep_buttons()
        self.draw_profile()
        self._store("dnp_sweep", {
            "label": "DNP field sweep", "target": f"{len(entries)} spectra: " + ", ".join(labels),
            "inputs_text": self._sweep_inputs(), "report": dnp.field_sweep_report(result),
            "result": result}, [e["index"] for e in entries if e["index"] is not None])
        return result

    def _sweep_inputs(self) -> list[str]:
        return [f"Line: {None if self.sweep_peak is None else round(self.sweep_peak, 1)} Hz from carrier",
                f"Region: {self.sweep_region} Hz from carrier",
                f"Reference: {self.sweep_reference_combo.currentText()}",
                f"Search window: ±{self.sweep_window_spin.value()} kHz",
                f"Minimum S/N: {self.sweep_snr_spin.value()}",
                f"B0 from: {self.sweep_field_combo.currentText()}",
                "Line shift: " + ("from reference" if self.sweep_shift_auto.isChecked()
                                  else f"{self.sweep_shift_spin.value()} ppm"),
                f"Track: {self.sweep_track_check.isChecked()}",
                f"Baseline: {self.sweep_baseline_combo.currentText()}",
                f"Correct receiver gain: {self.sweep_rg_check.isChecked()}"]

    def _fill_table(self, result: dict) -> None:
        rows = result["rows"]
        self.sweep_table.setRowCount(len(rows))
        italic = QFont()
        italic.setItalic(True)
        for r, row in enumerate(rows):
            for c, (key, _title) in enumerate(_TABLE):
                item = QTableWidgetItem(_fmt_cell(key, row.get(key)))
                if not row["found"]:
                    item.setFont(italic)
                    item.setToolTip("No line found here: value read at the pick, field from SFO1.")
                self.sweep_table.setItem(r, c, item)

    def draw_profile(self) -> None:
        result = self.sweep_result
        if result is None:
            return
        quantity = self.sweep_quantity_combo.currentData()
        labels = [row["label"] for row in result["rows"]]
        normalise = self._normalise_choice(labels)
        field_unit, relative = self.sweep_axis_combo.currentData()
        if quantity in ("intensity", "both") and result["rows"][0]["intensity"] is None:
            quantity = "area"
        if quantity in ("area", "both") and result["rows"][0]["area"] is None:
            quantity = "intensity"

        def draw(figure):
            style = self._style("profile")
            dnp.plot_field_profile(result, quantity=quantity, normalise=normalise,
                                   field_unit=field_unit, relative=relative, figure=figure,
                                   color=style.get("trace_color") or "black",
                                   fontsize=style.get("labelsize") or 12.0,
                                   show_top_spine=bool(style.get("top_spine")),
                                   show_right_spine=bool(style.get("right_spine")))
        self._draw_profile(draw)

    def export_csv(self) -> None:
        if self.sweep_result is None:
            return
        path, _filter = QFileDialog.getSaveFileName(self, "Export DNP field sweep",
                                                    "dnp_field_sweep.csv", "CSV (*.csv)")
        if path:
            dnp.field_sweep_to_csv(self.sweep_result, path)
