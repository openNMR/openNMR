"""Integrate ▸ Offset Tracker: a series of 1D spectra integrated over one
region, plotted against the offset (O2 or O3, in ppm) of the channel
pulsed while they were recorded -- 1D sideband-selective HMQC or
J-resolved data (explicit user request, 2026-10-17). The maths is
:mod:`opennmr.notebook.offset_tracker`, so a notebook gets the same
numbers.

Laid out like CP Build-up: 1. spectra and their offsets (read from O2/O3
or typed), 2. the region (typed, or dragged on any of the spectra), 3.
integrate and plot (bars, vertical lines, or peaks -- a Gaussian,
Lorentzian or pseudo-Voigt of a chosen width centred on every offset,
which with the offsets on the sidebands is the pulsed nucleus's MAS
spectrum; 2026-10-18, replacing triangles), and keep what is drawn as a
spectrum in Open experiments, to fit in Fit like any other.
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout,
    QHeaderView, QLabel, QLineEdit, QMessageBox, QPushButton, QSpinBox, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

from ..notebook import offset_tracker as ot
from .widgets.report_label import ReportLabel
from .widgets.typed_spinbox import TypedDoubleSpinBox

_C_USE, _C_NAME, _C_OFFSET = 0, 1, 2


def _number(text: str):
    try:
        return float(str(text).strip().replace("−", "-"))
    except (TypeError, ValueError):
        return None


class OffsetTrackerPanel(QWidget):
    """``spectra()`` -> [{"label", "spectrum", "index"}] (Open experiments'
    1D spectra); ``draw(draw_fn, view)`` puts a figure on the Fit view
    ("profile") or Compare; ``arm(mode)`` turns region dragging on Compare
    on ("region") or off (None); ``show_spectrum(spectrum)`` puts one
    spectrum on Compare; ``highlight(regions)`` shades regions there;
    ``keep(spectrum, label, description)`` adds a spectrum to Open
    experiments."""

    def __init__(self, spectra, draw, arm, show_spectrum=None, highlight=None, keep=None,
                 parent=None) -> None:
        super().__init__(parent)
        self._spectra, self._draw, self._arm = spectra, draw, arm
        self._show_spectrum, self._highlight, self._keep = show_spectrum, highlight, keep
        self._entries: list[dict] = []
        self._checked: set[int] = set()
        self._offsets: dict[int, str] = {}
        self.armed = False
        self.series: dict | None = None
        self.pattern: dict | None = None       # the last lineshape pattern drawn
        self._fwhm_user_set = False            # the width was chosen, not defaulted

        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        intro = QLabel(
            "For 1D spectra of one nucleus recorded while pulsing on another at a stepped "
            "offset (sideband-selective HMQC, J-resolved): integrate the same region of every "
            "spectrum and plot it against the pulsed channel's offset (O2 or O3) in ppm, as bars, "
            "vertical lines or peaks centred on the offsets (with the offsets on the sidebands, "
            "the pulsed nucleus's MAS spectrum). What is drawn can be kept as a spectrum of the "
            "pulsed nucleus and fitted.")
        intro.setWordWrap(True)
        v.addWidget(intro)
        v.addWidget(self._build_spectra())
        v.addWidget(self._build_region())
        v.addWidget(self._build_plot())
        self.refresh()

    # ------------------------------------------------------------------
    def _build_spectra(self) -> QGroupBox:
        box = QGroupBox("1. Spectra and offsets")
        v = QVBoxLayout(box)
        self.spectra_table = QTableWidget(0, 3)
        self.spectra_table.setHorizontalHeaderLabels(["Use", "Spectrum", "Offset (ppm)"])
        header = self.spectra_table.horizontalHeader()
        header.setSectionResizeMode(_C_USE, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(_C_NAME, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(_C_OFFSET, QHeaderView.ResizeMode.ResizeToContents)
        self.spectra_table.verticalHeader().setVisible(False)
        self.spectra_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.spectra_table.setMinimumHeight(140)
        self.spectra_table.setToolTip("Tick the spectra of the series; Read Offsets fills the "
                                      "offsets from O2/O3, or type them (ppm of the pulsed "
                                      "nucleus).")
        self.spectra_table.itemChanged.connect(self._on_item_changed)
        v.addWidget(self.spectra_table)
        ticks = QHBoxLayout()
        self.tick_all_button = QPushButton("Tick All")
        self.tick_all_button.clicked.connect(
            lambda: self.set_checked(range(self.spectra_table.rowCount())))
        self.untick_all_button = QPushButton("Untick All")
        self.untick_all_button.clicked.connect(lambda: self.set_checked([]))
        ticks.addWidget(self.tick_all_button)
        ticks.addWidget(self.untick_all_button)
        ticks.addStretch(1)
        v.addLayout(ticks)
        row = QHBoxLayout()
        row.addWidget(QLabel("Offset from:"))
        self.channel_combo = QComboBox()
        for key in ot.CHANNELS:
            self.channel_combo.addItem(key, key)
        self.channel_combo.setToolTip(
            "The channel whose offset was stepped: O2 (f2) or O3 (f3) -- whichever the pulse "
            "programme pulses the second nucleus on. Read as O/BF in ppm of that channel's "
            "nucleus.")
        row.addWidget(self.channel_combo)
        self.read_button = QPushButton("Read Offsets")
        self.read_button.clicked.connect(lambda _c=False: self.read_offsets())
        row.addWidget(self.read_button)
        row.addStretch(1)
        v.addLayout(row)
        shift_row = QHBoxLayout()
        shift_row.addWidget(QLabel("Add to every offset:"))
        self.shift_edit = QLineEdit("0")
        self.shift_edit.setMaximumWidth(80)
        self.shift_edit.setToolTip(
            "ppm added to every offset -- when the pulsed nucleus's 0 ppm is not at its "
            "channel's basic frequency (BF), e.g. a non-standard reference.")
        shift_row.addWidget(self.shift_edit)
        shift_row.addWidget(QLabel("ppm"))
        shift_row.addStretch(1)
        v.addLayout(shift_row)
        self.spectra_status = QLabel("")
        self.spectra_status.setWordWrap(True)
        v.addWidget(self.spectra_status)
        return box

    def _build_region(self) -> QGroupBox:
        box = QGroupBox("2. Region (observed nucleus)")
        v = QVBoxLayout(box)
        form = QFormLayout()
        region = QHBoxLayout()
        self.from_edit = QLineEdit()
        self.to_edit = QLineEdit()
        for edit in (self.from_edit, self.to_edit):
            edit.setPlaceholderText("ppm")
            edit.setMaximumWidth(90)
            edit.editingFinished.connect(self._show_region)
        region.addWidget(self.from_edit)
        region.addWidget(QLabel("to"))
        region.addWidget(self.to_edit)
        region.addWidget(QLabel("ppm"))
        region.addStretch(1)
        form.addRow("Integrate:", region)
        self.drag_on_combo = QComboBox()
        self.drag_on_combo.setToolTip("The spectrum shown to drag the region on -- any of them.")
        form.addRow("Drag on:", self.drag_on_combo)
        v.addLayout(form)
        buttons = QHBoxLayout()
        self.drag_button = QPushButton("Drag on Plot")
        self.drag_button.setCheckable(True)
        self.drag_button.setToolTip("Shows the spectrum chosen in Drag on; drag across it to "
                                    "set the region.")
        self.drag_button.toggled.connect(self._on_drag_toggled)
        buttons.addWidget(self.drag_button)
        buttons.addStretch(1)
        v.addLayout(buttons)
        opts = QFormLayout()
        self.baseline_combo = QComboBox()
        self.baseline_combo.addItem("Local (fitted just outside the region)", "local")
        self.baseline_combo.addItem("None (spectrum as it is)", "none")
        self.baseline_combo.addItem("Global (robust, whole spectrum)", "global")
        opts.addRow("Baseline:", self.baseline_combo)
        self.rg_check = QCheckBox("Divide by receiver gain (RG)")
        self.rg_check.setChecked(True)
        opts.addRow("", self.rg_check)
        self.normalise_check = QCheckBox("Relative to the largest")
        opts.addRow("", self.normalise_check)
        v.addLayout(opts)
        return box

    def _build_plot(self) -> QGroupBox:
        box = QGroupBox("3. Plot and keep")
        v = QVBoxLayout(box)
        form = QFormLayout()
        self.style_combo = QComboBox()
        for text, key in (("Bars", "bars"), ("Vertical lines", "stems"),
                          ("Peaks at the offsets (lineshape)", "lineshape")):
            self.style_combo.addItem(text, key)
        self.style_combo.setToolTip(
            "Bars (the skyline), vertical lines, or a peak of the chosen lineshape and width "
            "centred on every offset, its height the integral there. Keep in Open Experiments "
            "keeps what is drawn.")
        self.style_combo.currentIndexChanged.connect(lambda _i: self._on_style_changed())
        form.addRow("Draw as:", self.style_combo)

        # Lineshape controls: type and width only -- the positions are the offsets.
        self.lineshape_box = QWidget()
        lf = QFormLayout(self.lineshape_box)
        lf.setContentsMargins(0, 0, 0, 0)
        self.lineshape_combo = QComboBox()
        for key, text in ot.LINESHAPES.items():
            self.lineshape_combo.addItem(text, key)
        self.lineshape_combo.setToolTip("The shape of every peak (notebook.fitting's "
                                        "definitions, so a fit with the same shape matches).")
        self.lineshape_combo.currentIndexChanged.connect(lambda _i: self._on_lineshape_changed())
        lf.addRow("Lineshape:", self.lineshape_combo)
        width_row = QHBoxLayout()
        self.fwhm_spin = TypedDoubleSpinBox()
        self.fwhm_spin.setDecimals(4)
        self.fwhm_spin.setRange(0.0001, 1.0e7)
        self.fwhm_spin.setValue(1.0)
        self.fwhm_spin.setToolTip(
            "Full width at half maximum of every peak (the same for all). Starts at a quarter "
            "of the smallest offset spacing. It is chosen here, not measured: fix it in a fit.")
        self.fwhm_spin.userEdited.connect(self._on_fwhm_edited)
        self.fwhm_spin.valueChanged.connect(lambda _v: self._redraw_lineshape())
        self.width_unit_combo = QComboBox()
        for unit in ot.WIDTH_UNITS:
            self.width_unit_combo.addItem(unit, unit)
        self.width_unit_combo.setToolTip("Hz and kHz use the pulsed channel's BF.")
        self._width_unit = "ppm"
        self.width_unit_combo.currentIndexChanged.connect(lambda _i: self._on_width_unit_changed())
        width_row.addWidget(self.fwhm_spin)
        width_row.addWidget(self.width_unit_combo)
        width_row.addStretch(1)
        lf.addRow("Width (FWHM):", width_row)
        self.eta_spin = QDoubleSpinBox()
        self.eta_spin.setRange(0.0, 1.0)
        self.eta_spin.setSingleStep(0.05)
        self.eta_spin.setDecimals(3)
        self.eta_spin.setValue(0.5)
        self.eta_spin.setToolTip("Lorentzian fraction of the pseudo-Voigt: 0 Gaussian, "
                                 "1 Lorentzian.")
        self.eta_spin.valueChanged.connect(lambda _v: self._redraw_lineshape())
        self._eta_label = QLabel("η (Lorentzian fraction):")
        lf.addRow(self._eta_label, self.eta_spin)
        self.components_check = QCheckBox("Show each peak (dashed)")
        self.components_check.toggled.connect(lambda _c: self._redraw_lineshape())
        lf.addRow("", self.components_check)
        form.addRow(self.lineshape_box)

        self.skyline_check = QCheckBox("Show the skyline (the bars' outline)")
        self.skyline_check.toggled.connect(lambda _c: self.redraw())
        form.addRow("", self.skyline_check)
        self.points_spin = QSpinBox()
        self.points_spin.setRange(64, 65536)
        self.points_spin.setValue(4096)
        self.points_spin.setToolTip(
            "Points of the kept spectrum's ppm grid. For peaks it is enlarged if needed, so "
            f"one FWHM spans at least {ot.MIN_POINTS_PER_FWHM} points.")
        self.shape_combo = QComboBox()
        self.shape_combo.addItem("Steps (the skyline)", "steps")
        self.shape_combo.addItem("Straight lines between offsets", "linear")
        self.shape_combo.setToolTip("How bars are kept as a spectrum.")
        keep_row = QHBoxLayout()
        keep_row.addWidget(self.shape_combo)
        keep_row.addWidget(self.points_spin)
        form.addRow("Kept spectrum:", keep_row)
        v.addLayout(form)
        buttons = QHBoxLayout()
        self.integrate_button = QPushButton("Integrate and Plot")
        self.integrate_button.clicked.connect(lambda _c=False: self.integrate())
        self.keep_button = QPushButton("Keep in Open Experiments")
        self.keep_button.setToolTip(
            "Add what is drawn (bars, vertical lines or peaks) as a 1D spectrum of the pulsed "
            "nucleus to Open experiments, to fit in Fit (CSA, quadrupolar, ...) like any other "
            "spectrum.")
        self.keep_button.clicked.connect(lambda _c=False: self.keep_pattern())
        buttons.addWidget(self.integrate_button)
        buttons.addWidget(self.keep_button)
        buttons.addStretch(1)
        v.addLayout(buttons)
        self.pattern_label = QLabel("")
        self.pattern_label.setWordWrap(True)
        v.addWidget(self.pattern_label)
        self.result_label = ReportLabel("")
        self.result_label.setWordWrap(True)
        v.addWidget(self.result_label)
        self._on_style_changed(redraw=False)
        pv = self.lineshape() == "pseudo_voigt"
        self.eta_spin.setVisible(pv)
        self._eta_label.setVisible(pv)
        return box

    # -- lineshape controls ----------------------------------------------
    def style(self) -> str:
        return self.style_combo.currentData()

    def lineshape(self) -> str:
        return self.lineshape_combo.currentData()

    def width_unit(self) -> str:
        return self.width_unit_combo.currentData()

    def _on_style_changed(self, redraw: bool = True) -> None:
        style = self.style()
        self.lineshape_box.setVisible(style == "lineshape")
        self.shape_combo.setVisible(style == "bars")
        if style != "lineshape":
            self.pattern_label.setText("")
        if redraw:
            self.redraw()

    def _on_lineshape_changed(self) -> None:
        pv = self.lineshape() == "pseudo_voigt"
        self.eta_spin.setVisible(pv)
        self._eta_label.setVisible(pv)
        self._redraw_lineshape()

    def _on_fwhm_edited(self) -> None:
        self._fwhm_user_set = True

    def _bf(self):
        return (self.series or {}).get("bf_MHz")

    def _ppm_per_unit(self, unit: str):
        """ppm in one ``unit`` (None when Hz cannot be converted)."""
        if unit == "ppm":
            return 1.0
        bf = self._bf()
        if not bf:
            return None
        return (1000.0 if unit == "kHz" else 1.0) / float(bf)

    def _on_width_unit_changed(self) -> None:
        """Keep the same physical width when the unit changes."""
        old, new = self._width_unit, self.width_unit()
        self._width_unit = new
        a, b = self._ppm_per_unit(old), self._ppm_per_unit(new)
        if a is not None and b is not None and old != new:
            self.fwhm_spin.blockSignals(True)
            self.fwhm_spin.setValue(self.fwhm_spin.value() * a / b)
            self.fwhm_spin.blockSignals(False)
        self._redraw_lineshape()

    def set_fwhm(self, value: float, unit: str | None = None) -> None:
        """Set the width (in ``unit``, default the one shown), as a user would."""
        if unit is not None:
            self.width_unit_combo.blockSignals(True)
            self.width_unit_combo.setCurrentIndex(self.width_unit_combo.findData(unit))
            self.width_unit_combo.blockSignals(False)
            self._width_unit = unit
        self._fwhm_user_set = True
        self.fwhm_spin.setValue(float(value))
        self._redraw_lineshape()

    def fwhm_ppm(self) -> float:
        return ot.fwhm_in_ppm(self.series or {}, self.fwhm_spin.value(), self.width_unit())

    def _default_width(self) -> None:
        """A quarter of the smallest spacing, in the unit shown -- until the
        user chooses a width."""
        if self._fwhm_user_set or self.series is None or len(self.series["y"]) < 1:
            return
        per = self._ppm_per_unit(self.width_unit())
        if per is None:                     # Hz shown but no BF: fall back to ppm
            self.width_unit_combo.blockSignals(True)
            self.width_unit_combo.setCurrentIndex(self.width_unit_combo.findData("ppm"))
            self.width_unit_combo.blockSignals(False)
            self._width_unit, per = "ppm", 1.0
        self.fwhm_spin.blockSignals(True)
        self.fwhm_spin.setValue(ot.default_fwhm_ppm(self.series) / per)
        self.fwhm_spin.blockSignals(False)

    def _lineshape_kwargs(self) -> dict:
        return {"lineshape": self.lineshape(), "fwhm": self.fwhm_spin.value(),
                "width_unit": self.width_unit(), "eta": self.eta_spin.value()}

    def _redraw_lineshape(self) -> None:
        if self.style() == "lineshape":
            self.redraw()

    # ------------------------------------------------------------------
    def refresh(self) -> None:
        """Re-read Open experiments, keeping ticks and typed offsets."""
        self._entries = list(self._spectra() or [])
        t = self.spectra_table
        t.blockSignals(True)
        try:
            t.setRowCount(len(self._entries))
            for r, e in enumerate(self._entries):
                key = id(e["spectrum"])
                use = QTableWidgetItem("")
                use.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
                use.setCheckState(Qt.CheckState.Checked if key in self._checked
                                  else Qt.CheckState.Unchecked)
                t.setItem(r, _C_USE, use)
                name = QTableWidgetItem(str(e["label"]))
                name.setFlags(Qt.ItemFlag.ItemIsEnabled)
                t.setItem(r, _C_NAME, name)
                t.setItem(r, _C_OFFSET, QTableWidgetItem(self._offsets.get(key, "")))
        finally:
            t.blockSignals(False)
        chosen = self.drag_on_combo.currentData()
        self.drag_on_combo.blockSignals(True)
        self.drag_on_combo.clear()
        for e in self._entries:
            self.drag_on_combo.addItem(str(e["label"]), id(e["spectrum"]))
        i = self.drag_on_combo.findData(chosen)
        self.drag_on_combo.setCurrentIndex(i if i >= 0 else 0)
        self.drag_on_combo.blockSignals(False)

    def _on_item_changed(self, item) -> None:
        r = item.row()
        if not (0 <= r < len(self._entries)):
            return
        key = id(self._entries[r]["spectrum"])
        if item.column() == _C_USE:
            if item.checkState() == Qt.CheckState.Checked:
                self._checked.add(key)
            else:
                self._checked.discard(key)
        elif item.column() == _C_OFFSET:
            self._offsets[key] = item.text().strip()

    def set_checked(self, rows) -> None:
        rows = set(rows)
        for r in range(self.spectra_table.rowCount()):
            self.spectra_table.item(r, _C_USE).setCheckState(
                Qt.CheckState.Checked if r in rows else Qt.CheckState.Unchecked)

    def set_offset(self, row: int, ppm) -> None:
        self.spectra_table.item(row, _C_OFFSET).setText(f"{float(ppm):.6g}")

    def set_region(self, lo: float, hi: float) -> None:
        self.from_edit.setText(f"{max(lo, hi):.6g}")
        self.to_edit.setText(f"{min(lo, hi):.6g}")
        self._show_region()

    def checked_rows(self) -> list[int]:
        return [r for r in range(self.spectra_table.rowCount())
                if self.spectra_table.item(r, _C_USE).checkState() == Qt.CheckState.Checked]

    def region(self):
        a, b = _number(self.from_edit.text()), _number(self.to_edit.text())
        if a is None or b is None or a == b:
            return None
        return (min(a, b), max(a, b))

    def read_offsets(self) -> int:
        """Fill the ticked rows' offsets from the chosen channel."""
        rows = self.checked_rows() or list(range(self.spectra_table.rowCount()))
        channel = self.channel_combo.currentData()
        n, problems = 0, []
        for r in rows:
            try:
                info = ot.channel_offset(self._entries[r]["spectrum"], channel)
            except ValueError as exc:
                problems.append(f"{self._entries[r]['label']}: {exc}")
                continue
            self.set_offset(r, info["ppm"])
            n += 1
        text = f"Read {n} offset(s) from {channel}."
        if problems:
            text += " " + " ".join(problems[:4]) + (" …" if len(problems) > 4 else "")
        self.spectra_status.setText(text)
        return n

    def _drag_entry(self):
        key = self.drag_on_combo.currentData()
        return next((e for e in self._entries if id(e["spectrum"]) == key), None)

    def _show_region(self) -> None:
        if self._highlight is not None:
            region = self.region()
            self._highlight([("region", region[0], region[1])] if region else [])

    def _on_drag_toggled(self, on: bool) -> None:
        if on:
            entry = self._drag_entry()
            if entry is None:
                self.drag_button.blockSignals(True)
                self.drag_button.setChecked(False)
                self.drag_button.blockSignals(False)
                QMessageBox.information(self, "Offset Tracker", "Open a 1D spectrum first.")
                return
            if self._show_spectrum is not None:
                self._show_spectrum(entry["spectrum"])
            self._show_region()
        self.armed = bool(on)
        self._arm("region" if on else None)

    def on_canvas_region(self, lo: float, hi: float) -> None:
        if lo == hi:
            return
        self.set_region(float(lo), float(hi))
        if self.armed:
            self._arm("region")

    def canvas_mode_released(self) -> None:
        if self.drag_button.isChecked():
            self.drag_button.blockSignals(True)
            self.drag_button.setChecked(False)
            self.drag_button.blockSignals(False)
        self.armed = False

    # ------------------------------------------------------------------
    def integrate(self) -> dict | None:
        rows = self.checked_rows()
        if len(rows) < 2:
            QMessageBox.warning(self, "Offset Tracker", "Tick at least two spectra.")
            return None
        region = self.region()
        if region is None:
            QMessageBox.warning(self, "Offset Tracker", "Give the region to integrate (two "
                                "different ppm values), or drag it on the plot.")
            return None
        offsets = []
        for r in rows:
            item = self.spectra_table.item(r, _C_OFFSET)
            offsets.append(_number(item.text()) if item and item.text().strip() else None)
        shift = _number(self.shift_edit.text()) or 0.0
        try:
            series = ot.offset_series(
                [self._entries[r]["spectrum"] for r in rows], region,
                channel=self.channel_combo.currentData(), offsets_ppm=offsets,
                labels=[self._entries[r]["label"] for r in rows],
                baseline=self.baseline_combo.currentData(), divide_rg=self.rg_check.isChecked(),
                ppm_shift=shift, normalise=self.normalise_check.isChecked())
        except ValueError as exc:
            QMessageBox.warning(self, "Offset Tracker", str(exc))
            return None
        self.series = series
        self._default_width()
        self.result_label.setText(ot.offset_report(series))
        self.redraw()
        return series

    def redraw(self) -> None:
        series = self.series
        if series is None:
            return
        style = self.style()
        sky = self.skyline_check.isChecked()
        kwargs = {}
        if style == "lineshape":
            kwargs = dict(self._lineshape_kwargs(), show_components=self.components_check.isChecked())
            try:
                self.pattern = ot.sideband_pattern(series, **self._lineshape_kwargs())
            except ValueError as exc:
                self.pattern = None
                self.pattern_label.setText(str(exc))
                return
            p = self.pattern
            text = (f"{ot.LINESHAPES[p['lineshape']]} peaks, FWHM {p['fwhm_ppm']:.4g} ppm"
                    + (f" ({p['fwhm_hz']:.4g} Hz)" if p["fwhm_hz"] else "")
                    + (f", η {p['eta']:.3g}" if p["eta"] is not None else "") + ".")
            self.pattern_label.setText(" ".join([text] + list(p["notes"])))

        def draw(figure):
            ax = figure.add_subplot(111)
            ot.plot_offset_series(series, style=style, ax=ax, show_skyline=sky, **kwargs)
        self._draw(draw, "profile")

    def skyline(self):
        """The bars' skyline as a spectrum (whatever is drawn)."""
        if self.series is None:
            return None
        return ot.skyline_spectrum(self.series, n_points=self.points_spin.value(),
                                   shape=self.shape_combo.currentData())

    def pattern_spectrum(self):
        """What is drawn, as a spectrum of the pulsed nucleus."""
        if self.series is None:
            return None
        style = self.style()
        kwargs = self._lineshape_kwargs() if style == "lineshape" else {}
        return ot.pattern_spectrum(self.series, style=style, n_points=self.points_spin.value(),
                                   shape=self.shape_combo.currentData(), **kwargs)

    def _kept_description(self, spectrum) -> tuple[str, str]:
        lo, hi = self.series["region"]
        who = self.series.get("nucleus") or self.series["channel"]
        n = len(self.series["y"])
        style = self.style()
        rec = spectrum.metadata.raw.get("offset_pattern", {})
        if style == "lineshape":
            shape = ot.LINESHAPES[rec["lineshape"]]
            width = f"FWHM {rec['fwhm_ppm']:.4g} ppm" + (
                f" = {rec['fwhm_hz']:.4g} Hz" if rec.get("fwhm_hz") else "")
            if rec.get("eta") is not None:
                width += f", η {rec['eta']:.3g}"
            label = f"{shape} peaks at {who} offsets, {hi:g} to {lo:g} ppm"
            how = f"a {shape} peak at every offset ({width}), heights the integrals, summed"
        elif style == "stems":
            label = f"Lines at {who} offsets, {hi:g} to {lo:g} ppm"
            how = "each integral as a line one point wide at its offset"
        else:
            label = f"Skyline: {who} offsets, {hi:g} to {lo:g} ppm"
            how = f"the bars' skyline ({self.shape_combo.currentText().lower()})"
        description = (f"Offset Tracker: integrals of {lo:g} to {hi:g} ppm in {n} spectra "
                       f"against the {self.series['channel']} offset, kept as {how}; "
                       f"{spectrum.data.size} points.")
        return label, description

    def keep_skyline(self):
        """Kept for callers of the 2026-10-17 name: keeps what is drawn."""
        return self.keep_pattern()

    def keep_pattern(self):
        """Keep what is drawn (bars, vertical lines or peaks) in Open
        experiments as a 1D spectrum of the pulsed nucleus."""
        if self.series is None and self.integrate() is None:
            return None
        try:
            spectrum = self.pattern_spectrum()
        except ValueError as exc:
            QMessageBox.warning(self, "Offset Tracker", str(exc))
            return None
        if not spectrum.metadata.nucleus:
            # Offsets typed for data that do not record the channel: ask.
            from PySide6.QtWidgets import QInputDialog
            nucleus, ok = QInputDialog.getText(
                self, "Offset Tracker", "Nucleus of the pulsed channel (e.g. 1H, 27Al):")
            if not ok or not nucleus.strip():
                return None
            spectrum.metadata.nucleus = nucleus.strip()
        label, description = self._kept_description(spectrum)
        if self._keep is not None:
            self._keep(spectrum, label, description)
        self.result_label.setText(self.result_label.text()
                                  + f"\nKept as “{label}” in Open experiments.")
        return spectrum
