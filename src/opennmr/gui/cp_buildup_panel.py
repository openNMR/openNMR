"""
Integrate ▸ CP Build-up: integrate a region across a variable-contact-time
series and fit the CP kinetics (explicit user request, 2026-10-16). The
maths is :mod:`opennmr.notebook.cp_buildup` (models from Kolodziejski &
Klinowski, Chem. Rev. 2002, 102, 613), which a notebook calls the same way.

1. **Spectra** -- tick the series in Open experiments' 1D spectra and give
   each its contact time: typed in the table (µs), or read from a pulse
   programme parameter (``P15``, as VOCS reads ``P1``). Choose a
   reference (ideally direct excitation; any spectrum will do).
2. **Regions** -- as many as wanted (explicit user request, 2026-10-16):
   typed in the table, or dragged on the reference spectrum (Drag on Plot
   shows it, and every drag adds a region until it is switched off).
   Integrate draws one build-up curve per region on the Fit view.
3. **Fit** -- a model, starting values (read off the data, editable),
   any parameter fixed; the result with errors, the contact time of
   maximum signal, warnings and advice. Kept with the spectra (their log
   and sidecar) like any fit.
"""

from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QFormLayout, QGroupBox,
    QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMessageBox, QPushButton, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

from ..notebook import cp_buildup as cpb
from ..notebook import fit_advice
from .widgets.advice_box import AdviceBox
from .widgets.report_label import ReportLabel

_C_USE, _C_NAME, _C_TIME = range(3)


def _number(text: str):
    text = str(text).strip()
    if not text:
        return None
    try:
        v = float(text)
    except ValueError:
        return None
    return v if math.isfinite(v) else None


class CpBuildupPanel(QWidget):
    """Callbacks (from MainWindow), as the S/N panel's: ``spectra()`` ->
    ``[{"label", "spectrum", "index"}]``; ``draw(draw, view)`` with view
    ``"spectrum"`` (Compare) or ``"profile"`` (the Fit view, a curve);
    ``arm(mode)``: ``"region"`` drag on Compare, or None; ``store(kind,
    record, indices)``."""

    calculated = Signal()
    fitted = Signal()

    def __init__(self, spectra, draw, arm, store, style=None, parent=None,
                 show_spectrum=None, highlight=None, has_plot=None) -> None:
        super().__init__(parent)
        self._spectra, self._draw, self._arm, self._store = spectra, draw, arm, store
        #: ``show_spectrum(spectrum)``: put one spectrum on the Compare view,
        #: in its own plot style; ``highlight(regions)``: shade ``[(name, lo,
        #: hi)]`` on what Compare shows, without redrawing it; ``has_plot()``:
        #: whether Compare shows a spectrum to drag on (switching to it).
        self._show_spectrum, self._highlight, self._has_plot = show_spectrum, highlight, has_plot
        self._style = style or (lambda _view: {})
        self._entries: list[dict] = []
        #: id(spectrum) -> typed contact time text, kept across refreshes.
        self._times: dict[int, str] = {}
        self._checked: set[int] = set()
        self.armed = False
        #: One series per region (cp_series_regions), and the fits by name.
        self.series_list: list[dict] = []
        self.fits: dict[str, dict] = {}
        self._series_indices: list[int] = []

        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        intro = QLabel(
            "Cross-polarisation build-up: integrate one region in every spectrum of a "
            "variable-contact-time series, plot the integral against the contact time, and "
            "fit the CP kinetics (Kolodziejski & Klinowski, Chem. Rev. 2002, 102, 613).")
        intro.setWordWrap(True)
        v.addWidget(intro)
        v.addWidget(self._build_spectra())
        v.addWidget(self._build_region())
        v.addWidget(self._build_fit())
        self.refresh()

    # ------------------------------------------------------------------
    def _build_spectra(self) -> QGroupBox:
        box = QGroupBox("1. Spectra and contact times")
        v = QVBoxLayout(box)
        self.spectra_table = QTableWidget(0, 3)
        self.spectra_table.setHorizontalHeaderLabels(["Use", "Spectrum", "Contact time (µs)"])
        header = self.spectra_table.horizontalHeader()
        header.setSectionResizeMode(_C_USE, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(_C_NAME, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(_C_TIME, QHeaderView.ResizeMode.ResizeToContents)
        self.spectra_table.verticalHeader().setVisible(False)
        self.spectra_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.spectra_table.setMinimumHeight(140)
        self.spectra_table.setToolTip("Tick the spectra of the series and type each one's "
                                      "contact time in µs -- or read them from a parameter below.")
        self.spectra_table.itemChanged.connect(self._on_item_changed)
        v.addWidget(self.spectra_table)
        # Explicit user request, 2026-10-17.
        tick_row = QHBoxLayout()
        self.tick_all_button = QPushButton("Tick All")
        self.tick_all_button.setToolTip("Use every spectrum listed.")
        self.tick_all_button.clicked.connect(
            lambda: self.set_checked(range(self.spectra_table.rowCount())))
        self.untick_all_button = QPushButton("Untick All")
        self.untick_all_button.setToolTip("Use none of them (then tick the ones you want).")
        self.untick_all_button.clicked.connect(lambda: self.set_checked([]))
        tick_row.addWidget(self.tick_all_button)
        tick_row.addWidget(self.untick_all_button)
        tick_row.addStretch(1)
        v.addLayout(tick_row)
        row = QHBoxLayout()
        row.addWidget(QLabel("Contact time from parameter:"))
        self.time_parameter_edit = QLineEdit("P15")
        self.time_parameter_edit.setMaximumWidth(90)
        self.time_parameter_edit.setToolTip(
            "The pulse-programme parameter holding the contact time, as written in the "
            "sequence (P15 on most Bruker CP programmes; pulses are in µs, D delays in s and "
            "converted). Or type a number here to give every ticked spectrum that time.")
        row.addWidget(self.time_parameter_edit)
        self.read_times_button = QPushButton("Read Contact Times")
        self.read_times_button.clicked.connect(self.read_contact_times)
        row.addWidget(self.read_times_button)
        row.addStretch(1)
        v.addLayout(row)
        form = QFormLayout()
        self.reference_combo = QComboBox()
        self.reference_combo.setToolTip(
            "Integrated over the same region and reported beside the curve; the curve can be "
            "expressed relative to it. Ideally a direct-excitation (non-CP) spectrum, so the "
            "CP gain is visible, but any spectrum will do.")
        form.addRow("Reference:", self.reference_combo)
        v.addLayout(form)
        self.spectra_status = QLabel("")
        self.spectra_status.setWordWrap(True)
        v.addWidget(self.spectra_status)
        return box

    def _build_region(self) -> QGroupBox:
        box = QGroupBox("2. Region and integration")
        v = QVBoxLayout(box)
        self.region_table = QTableWidget(0, 3)
        self.region_table.setHorizontalHeaderLabels(["Name", "From (ppm)", "To (ppm)"])
        self.region_table.verticalHeader().setVisible(False)
        rh = self.region_table.horizontalHeader()
        rh.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        rh.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        rh.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.region_table.setMinimumHeight(90)
        self.region_table.setToolTip("Each region gets its own build-up curve and fit. Type the "
                                     "limits, or drag them on the plot; rename freely.")
        self.region_table.itemChanged.connect(lambda _i: self._on_regions_edited())
        v.addWidget(self.region_table)
        buttons = QHBoxLayout()
        self.add_region_button = QPushButton("Add Region")
        self.add_region_button.clicked.connect(lambda: self.add_region())
        self.remove_region_button = QPushButton("Remove")
        self.remove_region_button.setToolTip("Remove the selected region(s).")
        self.remove_region_button.clicked.connect(self.remove_selected_regions)
        self.show_reference_button = QPushButton("Show Reference")
        self.show_reference_button.setToolTip("Draw the reference (or the first ticked spectrum) "
                                              "with the regions on the Compare view.")
        self.show_reference_button.clicked.connect(self.show_reference)
        self.drag_button = QPushButton("Drag on Plot")
        self.drag_button.setCheckable(True)
        self.drag_button.setToolTip("Shows the reference spectrum; each drag across it adds a "
                                    "region. Click again to stop.")
        self.drag_button.toggled.connect(self._on_drag_toggled)
        for b in (self.add_region_button, self.remove_region_button, self.show_reference_button,
                  self.drag_button):
            buttons.addWidget(b)
        buttons.addStretch(1)
        v.addLayout(buttons)
        # Which spectrum the regions are dragged on (user report,
        # 2026-10-17: only the first experiment could be used).
        drag_row = QHBoxLayout()
        drag_row.addWidget(QLabel("Drag on:"))
        self.drag_on_combo = QComboBox()
        self.drag_on_combo.setToolTip(
            "The spectrum shown for Drag on Plot: the Compare view as it is, the "
            "reference, or any spectrum listed above. Choosing one shows it at once.")
        self.drag_on_combo.activated.connect(lambda _i: self._on_drag_on_chosen())
        drag_row.addWidget(self.drag_on_combo, 1)
        v.addLayout(drag_row)
        form = QFormLayout()
        self.baseline_combo = QComboBox()
        self.baseline_combo.addItem("Local (fitted just outside the region)", "local")
        self.baseline_combo.addItem("None (spectrum as it is)", "none")
        self.baseline_combo.addItem("Global (robust, whole spectrum)", "global")
        form.addRow("Baseline:", self.baseline_combo)
        self.rg_check = QCheckBox("Divide by receiver gain (RG)")
        self.rg_check.setChecked(True)
        self.rg_check.setToolTip("Integrals per scan and per unit RG, so experiments recorded "
                                 "with different receiver gains can be compared.")
        form.addRow("", self.rg_check)
        self.relative_combo = QComboBox()
        self.relative_combo.addItem("Integral per scan", "none")
        self.relative_combo.addItem("Relative to the reference", "reference")
        self.relative_combo.addItem("Relative to the largest", "max")
        form.addRow("Plot as:", self.relative_combo)
        v.addLayout(form)
        self.integrate_button = QPushButton("Integrate and Plot")
        self.integrate_button.clicked.connect(self.integrate)
        v.addWidget(self.integrate_button)
        self.series_label = ReportLabel("")
        self.series_label.setWordWrap(True)
        v.addWidget(self.series_label)
        return box

    def _build_fit(self) -> QGroupBox:
        box = QGroupBox("3. Fit the build-up")
        v = QVBoxLayout(box)
        form = QFormLayout()
        self.model_combo = QComboBox()
        for key, model in cpb.MODELS.items():
            self.model_combo.addItem(model["name"], key)
        self.model_combo.setToolTip(
            "Classical I–S (eq 10): rise with T_IS, decay with the protons' T1ρ -- the usual "
            "choice. Simple rise (eq 11): no decay seen. Eq 9: the S spins' T1ρ too. Two "
            "components: two sites (or rates) in one region. I–I*–S (eq 28): protonated "
            "carbons under MAS (fast transfer within C–H, then spin diffusion; λ = ½ for CH, "
            "⅓ for CH₂). Equation numbers from Kolodziejski & Klinowski, Chem. Rev. 2002.")
        self.model_combo.currentIndexChanged.connect(lambda _i: self._rebuild_parameters())
        form.addRow("Model:", self.model_combo)
        self.fit_region_combo = QComboBox()
        self.fit_region_combo.setToolTip("The region whose curve the starting values and Fit "
                                         "refer to. Fit All Regions fits every one.")
        self.fit_region_combo.currentIndexChanged.connect(lambda _i: self._on_fit_region_changed())
        form.addRow("Region:", self.fit_region_combo)
        v.addLayout(form)
        self.parameter_table = QTableWidget(0, 5)
        self.parameter_table.setHorizontalHeaderLabels(
            ["Parameter", "Start", "Fixed", "Lower", "Upper"])
        self.parameter_table.verticalHeader().setVisible(False)
        self.parameter_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents)
        self.parameter_table.setToolTip("Starting values (times in µs). Tick Fixed to hold a "
                                        "value; Lower/Upper are optional limits.")
        v.addWidget(self.parameter_table)
        row = QHBoxLayout()
        self.estimate_button = QPushButton("Starting Values from Data")
        self.estimate_button.clicked.connect(self.estimate_start)
        self.fit_button = QPushButton("Fit")
        self.fit_button.clicked.connect(self.run_fit)
        self.fit_all_button = QPushButton("Fit All Regions")
        self.fit_all_button.setToolTip("Every region with this model: starting values read off "
                                       "each region's own data; Fixed values held for all.")
        self.fit_all_button.clicked.connect(self.fit_all)
        row.addWidget(self.estimate_button)
        row.addWidget(self.fit_button)
        row.addWidget(self.fit_all_button)
        row.addStretch(1)
        v.addLayout(row)
        self.fit_results_label = ReportLabel("")
        self.fit_results_label.setWordWrap(True)
        self.fit_results_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        v.addWidget(self.fit_results_label)
        self.advice_box = AdviceBox()
        v.addWidget(self.advice_box)
        self._rebuild_parameters()
        return box

    # ------------------------------------------------------------------
    # spectra
    # ------------------------------------------------------------------
    def refresh(self) -> None:
        """Re-read Open experiments, keeping ticks and typed times."""
        self._entries = list(self._spectra() or [])
        table = self.spectra_table
        table.blockSignals(True)
        try:
            table.setRowCount(len(self._entries))
            for r, e in enumerate(self._entries):
                key = id(e["spectrum"])
                use = QTableWidgetItem("")
                use.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
                use.setCheckState(Qt.CheckState.Checked if key in self._checked
                                  else Qt.CheckState.Unchecked)
                table.setItem(r, _C_USE, use)
                name = QTableWidgetItem(str(e["label"]))
                name.setFlags(Qt.ItemFlag.ItemIsEnabled)
                name.setToolTip(str(e["label"]))
                table.setItem(r, _C_NAME, name)
                table.setItem(r, _C_TIME, QTableWidgetItem(self._times.get(key, "")))
        finally:
            table.blockSignals(False)
        current = self.reference_combo.currentData()
        self.reference_combo.blockSignals(True)
        self.reference_combo.clear()
        self.reference_combo.addItem("(none)", None)
        for r, e in enumerate(self._entries):
            self.reference_combo.addItem(str(e["label"]), id(e["spectrum"]))
        i = self.reference_combo.findData(current)
        self.reference_combo.setCurrentIndex(i if i >= 0 else 0)
        self.reference_combo.blockSignals(False)
        chosen = self.drag_on_combo.currentData()
        self.drag_on_combo.blockSignals(True)
        self.drag_on_combo.clear()
        self.drag_on_combo.addItem("The plot as it is (else the reference)", None)
        for e in self._entries:
            self.drag_on_combo.addItem(str(e["label"]), id(e["spectrum"]))
        i = self.drag_on_combo.findData(chosen)
        self.drag_on_combo.setCurrentIndex(i if i >= 0 else 0)
        self.drag_on_combo.blockSignals(False)

    def _drag_entry(self):
        key = self.drag_on_combo.currentData()
        return next((e for e in self._entries if id(e["spectrum"]) == key), None)

    def show_spectrum_for_drag(self, entry) -> bool:
        """Put ``entry``'s spectrum on the plot (its own style) with the
        regions shaded, to drag regions on."""
        if entry is None:
            return False
        if self._show_spectrum is not None:
            self._show_spectrum(entry["spectrum"])
            self._show_regions()
            return True
        return False

    def _on_drag_on_chosen(self) -> None:
        entry = self._drag_entry()
        if entry is None:
            return
        self.show_spectrum_for_drag(entry)
        if self.armed:
            self._arm("region")     # the redraw must not drop the drag mode

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
        elif item.column() == _C_TIME:
            self._times[key] = item.text().strip()

    def set_checked(self, rows) -> None:
        """Tick these table rows (and untick the rest)."""
        rows = set(rows)
        for r in range(self.spectra_table.rowCount()):
            self.spectra_table.item(r, _C_USE).setCheckState(
                Qt.CheckState.Checked if r in rows else Qt.CheckState.Unchecked)

    def set_contact_time(self, row: int, value) -> None:
        self.spectra_table.item(row, _C_TIME).setText(f"{value:g}" if isinstance(value, float)
                                                      else str(value))

    def set_reference(self, row: int | None) -> None:
        key = None if row is None else id(self._entries[row]["spectrum"])
        i = self.reference_combo.findData(key)
        self.reference_combo.setCurrentIndex(max(i, 0))

    def _checked_rows(self) -> list[int]:
        return [r for r in range(self.spectra_table.rowCount())
                if self.spectra_table.item(r, _C_USE).checkState() == Qt.CheckState.Checked]

    def _reference_entry(self):
        key = self.reference_combo.currentData()
        return next((e for e in self._entries if id(e["spectrum"]) == key), None)

    def read_contact_times(self) -> int:
        """Fill the ticked rows' contact times from the parameter (or the
        number) typed. Returns how many were filled; problems are listed."""
        source = self.time_parameter_edit.text().strip()
        if not source:
            QMessageBox.warning(self, "CP Build-up", "Type a parameter name (e.g. P15) or a "
                                "number of µs.")
            return 0
        rows = self._checked_rows()
        if not rows:
            QMessageBox.warning(self, "CP Build-up", "Tick the spectra of the series first.")
            return 0
        problems, filled = [], 0
        for r in rows:
            e = self._entries[r]
            try:
                t = cpb.contact_time_us(e["spectrum"], source)
            except ValueError as exc:
                problems.append(f"{e['label']}: {exc}")
                continue
            self.set_contact_time(r, float(t))
            filled += 1
        text = f"Contact times read from {source} for {filled} of {len(rows)} spectra."
        if problems:
            text += " Not read: " + " ".join(problems)
        self.spectra_status.setText(text)
        return filled

    # ------------------------------------------------------------------
    # region
    # ------------------------------------------------------------------
    def add_region(self, hi: float | None = None, lo: float | None = None,
                   name: str | None = None) -> int:
        """Append a region (``hi``/``lo`` in ppm; blank to type them).
        Returns its row."""
        t = self.region_table
        r = t.rowCount()
        used = {t.item(i, 0).text() for i in range(r) if t.item(i, 0)}
        if name is None:
            k = r + 1
            while f"Region {k}" in used:
                k += 1
            name = f"Region {k}"
        t.blockSignals(True)
        try:
            t.insertRow(r)
            t.setItem(r, 0, QTableWidgetItem(name))
            t.setItem(r, 1, QTableWidgetItem("" if hi is None else f"{hi:.4g}"))
            t.setItem(r, 2, QTableWidgetItem("" if lo is None else f"{lo:.4g}"))
        finally:
            t.blockSignals(False)
        self._on_regions_edited()
        return r

    def remove_selected_regions(self) -> None:
        rows = sorted({i.row() for i in self.region_table.selectedIndexes()}, reverse=True)
        for r in rows:
            self.region_table.removeRow(r)
        self._on_regions_edited()

    def _on_regions_edited(self) -> None:
        """Regions changed: the curves and fits no longer match them."""
        self._show_regions()
        if self.series_list:
            self.series_list, self.fits = [], {}
            self._fill_fit_regions()
            self.series_label.setText("Regions changed: Integrate and Plot again.")

    def regions(self) -> list[tuple[str, float, float]]:
        """``[(name, lo, hi)]`` in table order; ValueError naming a bad row."""
        out = []
        t = self.region_table
        for r in range(t.rowCount()):
            name = (t.item(r, 0).text().strip() if t.item(r, 0) else "") or f"Region {r + 1}"
            a = _number(t.item(r, 1).text() if t.item(r, 1) else "")
            b = _number(t.item(r, 2).text() if t.item(r, 2) else "")
            if a is None or b is None:
                raise ValueError(f"{name}: give both limits in ppm.")
            if a == b:
                raise ValueError(f"{name}: the region has no width.")
            out.append((name, min(a, b), max(a, b)))
        if not out:
            raise ValueError("Add at least one region (Add Region, or Drag on Plot).")
        names = [n for n, _lo, _hi in out]
        if len(set(names)) != len(names):
            raise ValueError("Give every region its own name.")
        return out

    def region(self) -> tuple[float, float]:
        """The first region's (lo, hi)."""
        _name, lo, hi = self.regions()[0]
        return lo, hi

    def _valid_regions(self) -> list:
        try:
            return self.regions()
        except ValueError:
            return [(n, lo, hi) for n, lo, hi in self._partial_regions()]

    def _partial_regions(self):
        t = self.region_table
        for r in range(t.rowCount()):
            a = _number(t.item(r, 1).text() if t.item(r, 1) else "")
            b = _number(t.item(r, 2).text() if t.item(r, 2) else "")
            if a is not None and b is not None and a != b:
                name = (t.item(r, 0).text().strip() if t.item(r, 0) else "") or f"Region {r + 1}"
                yield name, min(a, b), max(a, b)

    def _show_regions(self) -> None:
        """Shade the regions on the plot as it is -- no redraw."""
        if self._highlight is not None:
            self._highlight(self._valid_regions())

    def show_reference(self) -> bool:
        entry = self._reference_entry()
        if entry is None:
            rows = self._checked_rows()
            entry = self._entries[rows[0]] if rows else None
        if entry is None:
            QMessageBox.warning(self, "CP Build-up", "Choose a reference or tick a spectrum.")
            return False
        spectrum, label = entry["spectrum"], entry["label"]
        if self._show_spectrum is not None:
            # The ordinary spectrum plot, in the Compare view's own style.
            self._show_spectrum(spectrum)
            self._show_regions()
            return True
        regions = self._valid_regions()

        def draw(figure):
            from ..plotting.accessibility import ordered_qualitative_colours
            colours = ordered_qualitative_colours("okabe_ito")
            ax = figure.add_subplot(111)
            x = spectrum.axes[0].ppm
            ax.plot(x, spectrum.data, color=self._style("spectrum").get("trace_color") or "black",
                    lw=1.0, label=str(label))
            for k, (name, lo, hi) in enumerate(regions):
                ax.axvspan(lo, hi, color=colours[k % len(colours)], alpha=0.2, lw=0, label=name)
            ax.set_xlim(float(np.max(x)), float(np.min(x)))
            ax.set_xlabel(f"{spectrum.metadata.nucleus or ''} shift (ppm)".strip())
            ax.set_yticks([])
            ax.legend(frameon=False)
        self._draw(draw, "spectrum")
        return True

    def _on_drag_toggled(self, on: bool) -> None:
        # The reference first, so there is a spectrum to drag on -- the
        # drag used to land on whatever Compare held, often nothing (user
        # report, 2026-10-16).
        # Drag on the plot as it is -- its style, view and traces kept
        # (user report, 2026-10-16); the reference only when Compare shows
        # no spectrum at all.
        if on:
            entry = self._drag_entry()
            if entry is not None and self.show_spectrum_for_drag(entry):
                self._show_regions()
                self.armed = True
                self._arm("region")
                return
            has_plot = self._has_plot() if self._has_plot is not None else False
            if not has_plot and not self.show_reference():
                self.drag_button.blockSignals(True)
                self.drag_button.setChecked(False)
                self.drag_button.blockSignals(False)
                return
            self._show_regions()
        self.armed = bool(on)
        self._arm("region" if on else None)

    def on_canvas_region(self, lo: float, hi: float) -> None:
        """A drag on the plot: a new region. Stays armed for the next one."""
        lo, hi = sorted((float(lo), float(hi)))
        if lo == hi:
            return
        self.add_region(hi=hi, lo=lo)      # shades it (no redraw)
        if self.armed:
            self._arm("region")     # redrawing must not drop the drag mode

    def canvas_mode_released(self) -> None:
        if self.drag_button.isChecked():
            self.drag_button.blockSignals(True)
            self.drag_button.setChecked(False)
            self.drag_button.blockSignals(False)
        self.armed = False

    # ------------------------------------------------------------------
    # integrate
    # ------------------------------------------------------------------
    def integrate(self) -> bool:
        rows = self._checked_rows()
        if len(rows) < 2:
            QMessageBox.warning(self, "CP Build-up", "Tick at least two spectra of the series.")
            return False
        times, missing = [], []
        for r in rows:
            v = _number(self.spectra_table.item(r, _C_TIME).text())
            if v is None or v < 0:
                missing.append(str(self._entries[r]["label"]))
            times.append(v)
        if missing:
            QMessageBox.warning(self, "CP Build-up", "No contact time for: " + ", ".join(missing)
                                + ". Type them (µs) or use Read Contact Times.")
            return False
        ref = self._reference_entry()
        try:
            regions = self.regions()
            series_list = cpb.cp_series_regions(
                [self._entries[r]["spectrum"] for r in rows], [(lo, hi) for _n, lo, hi in regions],
                times, names=[n for n, _lo, _hi in regions],
                reference=ref["spectrum"] if ref else None,
                labels=[self._entries[r]["label"] for r in rows],
                baseline=self.baseline_combo.currentData(), divide_rg=self.rg_check.isChecked(),
                relative_to=self.relative_combo.currentData())
        except ValueError as exc:
            QMessageBox.warning(self, "CP Build-up", str(exc))
            return False
        self.series_list, self.fits = series_list, {}
        self._series_indices = [self._entries[r].get("index") for r in rows]
        if ref is not None:
            self._series_indices.append(ref.get("index"))
        self.fit_results_label.setText("")
        self.advice_box.set_items([])
        first = series_list[0]
        lines = [f"{len(rows)} spectra, {len(series_list)} region(s), contact times "
                 f"{first['t_us'].min():g} to {first['t_us'].max():g} µs; y = {first['unit']}."]
        for series in series_list:
            if series["reference"] is not None:
                lines.append(f"{series['name']}: reference integral "
                             f"{series['reference']['integral']:.4g} ± "
                             f"{series['reference']['error']:.2g}.")
        notes = []
        for series in series_list:
            notes += [n for n in series["notes"] if n not in notes]
        lines += [f"Note: {n}" for n in notes]
        self.series_label.setText("\n".join(lines))
        self._fill_fit_regions()
        self._draw_curve()
        self.estimate_start()
        self.calculated.emit()
        return True

    # the region the fit controls act on --------------------------------
    def _fill_fit_regions(self) -> None:
        current = self.fit_region_combo.currentText()
        self.fit_region_combo.blockSignals(True)
        self.fit_region_combo.clear()
        for series in self.series_list:
            self.fit_region_combo.addItem(series["name"])
        i = self.fit_region_combo.findText(current)
        self.fit_region_combo.setCurrentIndex(i if i >= 0 else 0)
        self.fit_region_combo.blockSignals(False)

    def _on_fit_region_changed(self) -> None:
        self.estimate_start()
        fit = self.fit
        if fit is not None:
            self.fit_results_label.setText(cpb.cp_report(fit, self.series))
            self.advice_box.set_items(fit_advice.advise_cp_buildup(fit))

    def set_fit_region(self, name: str) -> None:
        i = self.fit_region_combo.findText(name)
        if i < 0:
            raise KeyError(name)
        self.fit_region_combo.setCurrentIndex(i)

    @property
    def series(self) -> dict | None:
        """The series of the region chosen under Fit (the first by default)."""
        if not self.series_list:
            return None
        name = self.fit_region_combo.currentText()
        return next((s for s in self.series_list if s["name"] == name), self.series_list[0])

    @property
    def fit(self) -> dict | None:
        series = self.series
        return self.fits.get(series["name"]) if series is not None else None

    def _draw_curve(self) -> None:
        series_list, fits = list(self.series_list), dict(self.fits)

        def draw(figure):
            style = self._style("profile")
            if len(series_list) == 1:
                cpb.plot_cp_buildup(series_list[0], fits.get(series_list[0]["name"]),
                                    figure=figure, colour=style.get("trace_color") or "black")
            else:
                cpb.plot_cp_buildups(series_list, fits, figure=figure)
        self._draw(draw, "profile")

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------
    def model(self) -> str:
        return self.model_combo.currentData()

    def _rebuild_parameters(self, start: dict | None = None) -> None:
        names = cpb.MODELS[self.model()]["params"]
        old = self._parameter_values()
        table = self.parameter_table
        table.setRowCount(len(names))
        for r, n in enumerate(names):
            label, unit = cpb.PARAMETERS[n]
            item = QTableWidgetItem(f"{label}" + (f" ({unit})" if unit else ""))
            item.setFlags(Qt.ItemFlag.ItemIsEnabled)
            item.setData(Qt.ItemDataRole.UserRole, n)
            table.setItem(r, 0, item)
            value = (start or {}).get(n, old.get(n, {}).get("start"))
            table.setItem(r, 1, QTableWidgetItem("" if value is None else f"{value:.6g}"))
            fixed = QTableWidgetItem("")
            fixed.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
            fixed.setCheckState(Qt.CheckState.Checked if old.get(n, {}).get("fixed")
                                else Qt.CheckState.Unchecked)
            table.setItem(r, 2, fixed)
            for c, key in ((3, "lower"), (4, "upper")):
                v = old.get(n, {}).get(key)
                table.setItem(r, c, QTableWidgetItem("" if v is None else f"{v:g}"))

    def _parameter_values(self) -> dict:
        out = {}
        table = self.parameter_table
        for r in range(table.rowCount()):
            item = table.item(r, 0)
            if item is None:
                continue
            n = item.data(Qt.ItemDataRole.UserRole)
            get = lambda c: table.item(r, c).text() if table.item(r, c) else ""  # noqa: E731
            out[n] = {"start": _number(get(1)), "lower": _number(get(3)), "upper": _number(get(4)),
                      "fixed": table.item(r, 2) is not None
                      and table.item(r, 2).checkState() == Qt.CheckState.Checked}
        return out

    def set_parameter(self, name: str, start=None, fixed=None) -> None:
        for r in range(self.parameter_table.rowCount()):
            if self.parameter_table.item(r, 0).data(Qt.ItemDataRole.UserRole) == name:
                if start is not None:
                    self.parameter_table.item(r, 1).setText(f"{start:g}")
                if fixed is not None:
                    self.parameter_table.item(r, 2).setCheckState(
                        Qt.CheckState.Checked if fixed else Qt.CheckState.Unchecked)
                return
        raise KeyError(name)

    def estimate_start(self) -> None:
        if self.series is None:
            return
        start = cpb.default_start(self.series["t_us"], self.series["y"], self.model())
        values = self._parameter_values()
        for n, v in values.items():         # keep values the user fixed
            if v["fixed"] and v["start"] is not None:
                start[n] = v["start"]
        self._rebuild_parameters(start)

    def run_fit(self) -> bool:
        """Fit the region chosen under Fit, with the table's values."""
        if self.series is None and not self.integrate():
            return False
        return self._fit_region(self.series, use_table_starts=True)

    def fit_all(self) -> bool:
        """Fit every region: starting values from each region's own data,
        the table's Fixed values and limits held for all."""
        if not self.series_list and not self.integrate():
            return False
        ok = True
        for series in list(self.series_list):
            ok = self._fit_region(series, use_table_starts=False, quiet=True) and ok
        self._on_fit_region_changed()
        if not ok:
            QMessageBox.warning(self, "CP Build-up", "Some regions could not be fitted: see the "
                                "report of each (choose it under Region).")
        return ok

    def _fit_region(self, series: dict, use_table_starts: bool, quiet: bool = False) -> bool:
        values = self._parameter_values()
        start = {n: v["start"] for n, v in values.items()
                 if v["start"] is not None and (use_table_starts or v["fixed"])}
        fixed = {}
        for n, v in values.items():
            if v["fixed"]:
                if v["start"] is None:
                    QMessageBox.warning(self, "CP Build-up", f"Give a value for the fixed "
                                        f"{cpb.PARAMETERS[n][0]}.")
                    return False
                fixed[n] = v["start"]
        bounds = {}
        for n, v in values.items():
            if v["lower"] is not None or v["upper"] is not None:
                bounds[n] = (v["lower"] if v["lower"] is not None else -np.inf,
                             v["upper"] if v["upper"] is not None else np.inf)
        try:
            fit = cpb.fit_cp_buildup(series["t_us"], series["y"], self.model(),
                                     start=start, fixed=fixed, bounds=bounds,
                                     y_err=series["y_err"])
        except ValueError as exc:
            if not quiet:
                QMessageBox.warning(self, "CP Build-up", f"{series['name']}: {exc}")
            return False
        self.fits[series["name"]] = fit
        if series is self.series:
            self.fit_results_label.setText(cpb.cp_report(fit, series))
            self.advice_box.set_items(fit_advice.advise_cp_buildup(fit))
        self._draw_curve()
        self._store_fits()
        self.fitted.emit()
        return True

    def _store_fits(self) -> None:
        """Keep every fitted region with the spectra (their log and sidecar)."""
        indices = [i for i in self._series_indices if i is not None]
        if not indices or not self.fits:
            return
        by_name = {s["name"]: s for s in self.series_list}
        reports, results, regions = [], {}, []
        for name, fit in self.fits.items():
            series = by_name.get(name)
            if series is None:
                continue
            lo, hi = series["region"]
            regions.append(f"{name} {hi:g} to {lo:g} ppm")
            reports.append(f"[{name}]\n" + cpb.cp_report(fit, series))
            results[name] = {"model": fit["model"], "region_ppm": [lo, hi],
                             "params": fit["params"], "errors": fit["errors"],
                             "fixed": fit["fixed"], "t_us": series["t_us"].tolist(),
                             "y": series["y"].tolist(), "y_err": series["y_err"].tolist(),
                             "r_squared": fit["r_squared"], "t_max_us": fit["t_max_us"]}
        first = self.series_list[0]
        record = {"label": "CP build-up", "target": ", ".join(map(str, first["labels"])),
                  "inputs_text": ("; ".join(regions) + f"; baseline {first['baseline']}; "
                                  f"RG divided: {'yes' if first['divide_rg'] else 'no'}"),
                  "report": "\n\n".join(reports), "result": results}
        self._store("cp_buildup", record, indices)
