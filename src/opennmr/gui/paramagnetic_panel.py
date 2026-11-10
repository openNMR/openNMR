"""
Simulate ▸ Paramagnetic NMR tools: each tool a collapsible section,
drawing on the Simulation plot and copying its notebook code.

1. Shift against temperature (2026-11-06; any source 2026-11-07) --
   calculated from Simulate ▸ DFT tensors (an ORCA g/ZFS/hyperfine
   calculation with its magnetic model, or a CRYSTAL calculation with its
   φ / χ_M / Curie–Weiss scaling) or from shifts typed in from any code;
   measured shifts fitted to a Curie, Curie + Bleaney (1/T²) or
   Curie–Weiss law (:mod:`opennmr.notebook.paramagnetic_nmr`).
2. Bulk magnetic susceptibility (BMS) shifts (2026-11-07) -- the spectrum
   of a particle packing (one set of FFTs for every particle,
   :mod:`opennmr.notebook.bms.fast`) in its sample (a rotor, tube, plate
   or sphere, :mod:`~opennmr.notebook.bms.sample`), with local contact,
   spin-dipolar and quadrupolar interactions
   (:mod:`~opennmr.notebook.bms.interactions`); the spectrum, a field
   slice or the packing in 3-D (:mod:`~opennmr.notebook.bms.views`); a
   time check before long runs.
3. Bond pathways (2026-11-06; 2026-11-07: which ion each file flips, and
   the neighbour distances to set the shells by) -- per-shell, per-species
   contributions from spin-flip or single-ion DFT calculations, sent to
   Simulate ▸ pNMR (CIF) (:mod:`opennmr.notebook.bond_pathways`).

The lanthanide-series and pseudocontact-shift sections were removed from
the GUI (explicit user request, 2026-11-07); their notebook modules,
:mod:`opennmr.notebook.lanthanides` and :mod:`opennmr.notebook.pseudocontact`,
are unchanged.
"""

from __future__ import annotations

import os
import re
from contextlib import contextmanager

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QCheckBox, QComboBox, QFileDialog, QFormLayout, QGroupBox,
    QHBoxLayout, QHeaderView, QLabel, QLineEdit, QListWidget, QMessageBox, QPushButton,
    QSlider, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from ..notebook import bond_pathways as bp
from ..notebook import paramagnetic_nmr as pmr
from ..notebook.bms import fast as bmsf
from ..notebook.bms import interactions as bms_inter
from ..notebook.bms import sample as bms_sample
from ..notebook.bms import views as bms_views
from ..notebook import susceptibility as chi_units
from .combined_panels import _nuclides, _spin_box
from .widgets.collapsible_section import CollapsibleSection

_CALC_FILTER = "Calculation output (*.out *.outp *.log *.magres OUTCAR*);;All files (*)"


def _num(text, what):
    try:
        return float(str(text).strip())
    except ValueError:
        raise ValueError(f"{what}: '{text}' is not a number.") from None


def _table(columns, rows=0, height=160):
    t = QTableWidget(rows, len(columns))
    t.setHorizontalHeaderLabels(columns)
    t.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
    t.verticalHeader().setVisible(False)
    t.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    t.setMinimumHeight(min(height, 90))
    t.setMaximumHeight(height)
    return t


def _set(t, r, c, text, editable=False):
    item = QTableWidgetItem(str(text))
    if not editable:
        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
    t.setItem(r, c, item)
    return item


def _cell(t, r, c):
    item = t.item(r, c)
    return item.text().strip() if item is not None else ""


@contextmanager
def _busy():
    """A wait cursor while a calculation runs."""
    QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
    try:
        yield
    finally:
        QApplication.restoreOverrideCursor()


def _label():
    lab = QLabel("")
    lab.setWordWrap(True)
    lab.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    return lab


class ParamagneticToolsPanel(QWidget):
    """``draw_plot(draw, kind)`` shows ``draw(figure)`` on the Simulation
    plot; ``plot_spectrum(spectrum)`` shows a simulated spectrum there (and
    logs the run); ``style()`` its style; ``dft_source()`` the DFT
    section's ORCA EPR calculation and model
    (:meth:`DftSimulationPanel.paramagnetic_source`);
    ``send_pathways(shell_bins, shift_dict)`` fills Simulate ▸ pNMR (CIF)."""

    BLURB = ("Paramagnetic NMR: the pNMR (CIF) shift simulation, the temperature dependence of "
             "shifts, the bulk magnetic susceptibility (BMS) shifts of a particle packing, and "
             "bond-pathway contributions from DFT for pNMR (CIF).")

    def __init__(self, draw_plot=None, style=None, dft_source=None, send_pathways=None,
                 pnmr_panel=None,
                 plot_spectrum=None, show_spectrum=None, keep_spectrum=None,
                 parent=None) -> None:
        super().__init__(parent)
        self._draw_plot = draw_plot
        self._plot_spectrum = plot_spectrum
        # show a spectrum again without logging a new run (Show ▸ Spectrum)
        self._show_spectrum = show_spectrum or plot_spectrum
        # into Open experiments (MainWindow._keep_simulation)
        self._keep_spectrum = keep_spectrum
        self._style = style or (lambda: {})
        self._dft_source = dft_source or (lambda: None)
        self._send_pathways = send_pathways
        self.temperature_curves = None
        self.temperature_fit = None
        self.bms_particles = None
        self.bms_packing_stats = None
        self._bms_packed_key = None
        self.bms_field = None
        self.bms_spectrum = None
        self.bms_nmr_spectrum = None
        self._bms_field_key = None
        self._bms_field_new = False
        self.pathway_files: list[str] = []
        self.pathway_result = None
        self.pathway_ions = None
        self.neighbour_rows = None
        layout = QVBoxLayout(self)
        blurb = QLabel(self.BLURB)
        blurb.setWordWrap(True)
        layout.addWidget(blurb)
        # Simulate ▸ pNMR (CIF) lives here since 2026-11-07 (explicit user request: the
        # paramagnetic sections together, shown only when chosen).
        self.pnmr_panel = pnmr_panel
        self.pm_pnmr_section = (CollapsibleSection("pNMR (CIF)", content=pnmr_panel)
                                if pnmr_panel is not None else None)
        self.pm_temperature_section = CollapsibleSection("Shift against temperature",
                                                         content=self._build_temperature())
        self.pm_bms_section = CollapsibleSection(
            "Bulk magnetic susceptibility (BMS) shifts", content=self._build_bms())
        self.pm_pathway_section = CollapsibleSection("Bond pathways from DFT (for pNMR (CIF))",
                                                     content=self._build_pathways())
        for sec in (self.pm_pnmr_section, self.pm_temperature_section, self.pm_bms_section,
                    self.pm_pathway_section):
            if sec is not None:
                layout.addWidget(sec)
        layout.addStretch(1)

    def _fontsize(self) -> float:
        return float((self._style() or {}).get("labelsize") or 11.0)

    def _show(self, draw, kind="curve") -> bool:
        """``kind``: "curve", "2d" (a map) or "tensor" (a 3-D picture), so
        the Simulation plot gives it the matching style and tools."""
        if self._draw_plot is None:
            return False
        self._draw_plot(draw, kind)
        return True

    @staticmethod
    def _copy(text: str) -> None:
        QGuiApplication.clipboard().setText(text)

    # ==== 1. temperature ==========================================================

    def _build_temperature(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        calc_box = QGroupBox("Calculated shifts")
        cv = QVBoxLayout(calc_box)
        cf = QFormLayout()
        self._T_calc_form = cf
        self.pm_T_source_combo = QComboBox()
        self.pm_T_source_combo.addItem("The calculation in Simulate ▸ DFT tensors (ORCA or CRYSTAL)",
                                       "dft")
        self.pm_T_source_combo.addItem("Shifts at one temperature, typed in (any code)", "typed")
        self.pm_T_source_combo.setToolTip(
            "ORCA: its magnetic model (calculated spin Hamiltonian, Curie–Weiss, μ_eff). CRYSTAL: "
            "its scaling (φ ∝ 1/(T − θ)). Typed in: shifts from any code (VASP, CASTEP, "
            "Gaussian, …) at one temperature, carried to others by a Curie, Curie–Weiss or "
            "Curie + Bleaney law.")
        cf.addRow("Source:", self.pm_T_source_combo)
        self.pm_T_ref_spin = _spin_box(1.0, 3000.0, 298.0, 2, " K", step=1.0,
                                       tip="The temperature the shifts (or a CRYSTAL φ or χ_M) "
                                           "belong to.")
        cf.addRow("Shifts at T:", self.pm_T_ref_spin)
        self.pm_T_scaling_combo = QComboBox()
        for key, text in pmr.SCALING_MODELS.items():
            self.pm_T_scaling_combo.addItem(text, key)
        cf.addRow("Temperature law:", self.pm_T_scaling_combo)
        self.pm_T_theta_spin = _spin_box(-1e5, 1e5, 0.0, 2, " K", step=1.0,
                                         tip="Weiss constant θ (negative: antiferromagnetic). "
                                             "For a CRYSTAL φ or χ_M, 0 is a Curie law.")
        cf.addRow("Weiss constant θ:", self.pm_T_theta_spin)
        cv.addLayout(cf)
        self._T_typed_box = QWidget()
        tv = QVBoxLayout(self._T_typed_box)
        tv.setContentsMargins(0, 0, 0, 0)
        hint = QLabel("One row per site: the diamagnetic (orbital) shift, and the contact and "
                      "isotropic dipolar (pseudocontact) shifts at the temperature above. Paste "
                      "(Ctrl+V) four columns: site, δ_dia, δ_contact, δ_dipolar.")
        hint.setWordWrap(True)
        tv.addWidget(hint)
        self.pm_T_calc_table = _table(["Site", "δ_dia (ppm)", "δ_contact (ppm)",
                                       "δ_dipolar (ppm)"], 0, 120)
        tv.addWidget(self.pm_T_calc_table)
        row = QHBoxLayout()
        for text, fn in (("Add Site", lambda: self.pm_T_calc_table.insertRow(
                             self.pm_T_calc_table.rowCount())),
                         ("Remove Sites", self._remove_calc_rows),
                         ("Paste", self.paste_calculated)):
            b = QPushButton(text)
            b.clicked.connect(lambda _c=False, f=fn: f())
            row.addWidget(b)
        row.addStretch(1)
        tv.addLayout(row)
        cv.addWidget(self._T_typed_box)
        rf = QFormLayout()
        self.pm_T_min_spin = _spin_box(1.0, 3000.0, 200.0, 1, " K", step=10.0)
        self.pm_T_max_spin = _spin_box(1.0, 3000.0, 400.0, 1, " K", step=10.0)
        self.pm_T_points_spin = _spin_box(2, 2000, 41, 0, step=1)
        rf.addRow("From:", self.pm_T_min_spin)
        rf.addRow("To:", self.pm_T_max_spin)
        rf.addRow("Points:", self.pm_T_points_spin)
        self.pm_T_components_check = QCheckBox("Also show the contact and dipolar parts")
        rf.addRow("", self.pm_T_components_check)
        self.pm_T_calc_button = QPushButton("Calculate δ(T)")
        self.pm_T_calc_button.setToolTip(
            "The isotropic shift of each site over this range: the sites ticked in Simulate ▸ "
            "DFT tensors with its magnetic model or scaling, or the typed-in sites.")
        self.pm_T_calc_button.clicked.connect(lambda: self.calculate_temperature())
        rf.addRow("", self.pm_T_calc_button)
        self.pm_T_source_label = _label()
        rf.addRow(self.pm_T_source_label)
        cv.addLayout(rf)
        self.pm_T_source_combo.currentIndexChanged.connect(lambda _i: self._sync_T_rows())
        self.pm_T_scaling_combo.currentIndexChanged.connect(lambda _i: self._sync_T_rows())
        v.addWidget(calc_box)
        data_box = QGroupBox("Measured shifts")
        dv = QVBoxLayout(data_box)
        hint = QLabel("Type or paste (Ctrl+V: two columns, T and δ) the measured shifts of one "
                      "site, then fit.")
        hint.setWordWrap(True)
        dv.addWidget(hint)
        self.pm_T_table = _table(["T (K)", "δ (ppm)"], 0, 180)
        dv.addWidget(self.pm_T_table)
        row = QHBoxLayout()
        for text, fn in (("Add Row", lambda: self.pm_T_table.insertRow(self.pm_T_table.rowCount())),
                         ("Remove Rows", self._remove_T_rows), ("Paste", self.paste_temperature)):
            b = QPushButton(text)
            b.clicked.connect(lambda _c=False, f=fn: f())
            row.addWidget(b)
        row.addStretch(1)
        dv.addLayout(row)
        form = QFormLayout()
        self.pm_T_model_combo = QComboBox()
        for key, (eq, tip) in pmr.TEMPERATURE_MODELS.items():
            self.pm_T_model_combo.addItem(eq, key)
            self.pm_T_model_combo.setItemData(self.pm_T_model_combo.count() - 1, tip,
                                              Qt.ItemDataRole.ToolTipRole)
        form.addRow("Model:", self.pm_T_model_combo)
        self.pm_T_fix_dia_check = QCheckBox("Fix δ_dia at")
        self.pm_T_dia_spin = _spin_box(-1e5, 1e5, 0.0, 3, " ppm", step=1.0,
                                       tip="The diamagnetic shift, e.g. of a diamagnetic analogue.")
        hrow = QHBoxLayout()
        hrow.addWidget(self.pm_T_fix_dia_check)
        hrow.addWidget(self.pm_T_dia_spin)
        hrow.addStretch(1)
        form.addRow("", hrow)
        self.pm_T_name_edit = QLineEdit("measured")
        form.addRow("Name:", self.pm_T_name_edit)
        dv.addLayout(form)
        self.pm_T_fit_button = QPushButton("Fit")
        self.pm_T_fit_button.clicked.connect(lambda: self.fit_temperature())
        dv.addWidget(self.pm_T_fit_button)
        v.addWidget(data_box)
        form2 = QFormLayout()
        self.pm_T_axis_combo = QComboBox()
        self.pm_T_axis_combo.addItem("T (K)", "T")
        self.pm_T_axis_combo.addItem("1000/T (K⁻¹): a Curie law is straight", "1/T")
        form2.addRow("x axis:", self.pm_T_axis_combo)
        v.addLayout(form2)
        row = QHBoxLayout()
        self.pm_T_plot_button = QPushButton("Plot")
        self.pm_T_plot_button.clicked.connect(lambda: self.plot_temperature())
        row.addWidget(self.pm_T_plot_button)
        self.pm_T_python_button = QPushButton("Copy as Python")
        self.pm_T_python_button.clicked.connect(lambda: self._copy_safe(self.temperature_python))
        row.addWidget(self.pm_T_python_button)
        row.addStretch(1)
        v.addLayout(row)
        self.pm_T_result_label = _label()
        v.addWidget(self.pm_T_result_label)
        self._sync_T_rows()
        return w

    def _sync_T_rows(self) -> None:
        typed = self.pm_T_source_combo.currentData() == "typed"
        self._T_typed_box.setVisible(typed)
        self._T_calc_form.setRowVisible(self.pm_T_scaling_combo, typed)
        # θ: the typed-in Curie–Weiss law, or carrying a CRYSTAL φ / χ_M to other T
        self._T_calc_form.setRowVisible(
            self.pm_T_theta_spin,
            not typed or self.pm_T_scaling_combo.currentData() == "curie_weiss")
        self.pm_T_ref_spin.setToolTip(
            "The temperature of the typed-in shifts." if typed else
            "Only for a CRYSTAL φ or χ_M (they belong to one temperature): the temperature "
            "they were measured at. A Curie–Weiss scaling carries its own T and θ.")

    def _remove_calc_rows(self) -> None:
        rows = sorted({i.row() for i in self.pm_T_calc_table.selectedIndexes()}, reverse=True)
        if not rows and self.pm_T_calc_table.rowCount():
            rows = [self.pm_T_calc_table.rowCount() - 1]
        for r in rows:
            self.pm_T_calc_table.removeRow(r)

    def paste_calculated(self, text=None) -> int:
        """Rows of site, δ_dia, δ_contact, δ_dipolar (tab, comma or space
        separated; a missing dipolar column is 0)."""
        text = QGuiApplication.clipboard().text() if text is None else text
        rows = []
        for line in (text or "").splitlines():
            line = line.strip()
            parts = [x.strip() for x in (re.split(r"[\t,;]+", line) if re.search(r"[\t,;]", line)
                                         else line.split()) if x.strip()]
            if len(parts) < 3:
                continue
            try:
                vals = [float(x) for x in parts[1:4]]
            except ValueError:
                continue
            rows.append([parts[0]] + vals + [0.0] * (3 - len(vals)))
        if rows:
            self.pm_T_calc_table.setRowCount(0)
            for site, dia, c, d in rows:
                r = self.pm_T_calc_table.rowCount()
                self.pm_T_calc_table.insertRow(r)
                for col, t in enumerate((site, f"{dia:g}", f"{c:g}", f"{d:g}")):
                    _set(self.pm_T_calc_table, r, col, t, True)
        return len(rows)

    def calculated_sites(self) -> dict:
        sites = {}
        for r in range(self.pm_T_calc_table.rowCount()):
            name = _cell(self.pm_T_calc_table, r, 0)
            vals = [_cell(self.pm_T_calc_table, r, c) for c in (1, 2, 3)]
            if not name and not any(vals):
                continue
            name = name or f"site {r + 1}"
            sites[name] = {k: (_num(v, f"{name}, {k}") if v else 0.0)
                           for k, v in zip(("dia", "contact", "dipolar"), vals)}
        if not sites:
            raise ValueError("Type at least one site: its δ_dia, δ_contact and δ_dipolar.")
        return sites

    def _remove_T_rows(self) -> None:
        rows = sorted({i.row() for i in self.pm_T_table.selectedIndexes()}, reverse=True)
        if not rows and self.pm_T_table.rowCount():
            rows = [self.pm_T_table.rowCount() - 1]
        for r in rows:
            self.pm_T_table.removeRow(r)

    def set_temperature_data(self, T, delta) -> None:
        self.pm_T_table.setRowCount(0)
        for t, d in zip(T, delta):
            r = self.pm_T_table.rowCount()
            self.pm_T_table.insertRow(r)
            _set(self.pm_T_table, r, 0, f"{float(t):g}", True)
            _set(self.pm_T_table, r, 1, f"{float(d):g}", True)

    def paste_temperature(self, text=None) -> int:
        text = QGuiApplication.clipboard().text() if text is None else text
        T, d = [], []
        for line in (text or "").splitlines():
            vals = re.findall(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?", line)
            if len(vals) >= 2:
                T.append(float(vals[0]))
                d.append(float(vals[1]))
        if T:
            self.set_temperature_data(T, d)
        return len(T)

    def temperature_data(self):
        T, d = [], []
        for r in range(self.pm_T_table.rowCount()):
            a, b = _cell(self.pm_T_table, r, 0), _cell(self.pm_T_table, r, 1)
            if not a and not b:
                continue
            T.append(_num(a, f"Row {r + 1}, T"))
            d.append(_num(b, f"Row {r + 1}, δ"))
        return np.array(T), np.array(d)

    def _temperature_grid(self):
        T0, T1 = self.pm_T_min_spin.value(), self.pm_T_max_spin.value()
        if T1 <= T0:
            raise ValueError("The range must go up: 'To' above 'From'.")
        return np.linspace(T0, T1, int(self.pm_T_points_spin.value()))

    def calculate_temperature(self, quiet=False) -> bool:
        try:
            T = self._temperature_grid()
            if self.pm_T_source_combo.currentData() == "typed":
                curves = pmr.scaled_shift_vs_temperature(
                    self.calculated_sites(), T, T_ref=self.pm_T_ref_spin.value(),
                    model=self.pm_T_scaling_combo.currentData(),
                    theta=self.pm_T_theta_spin.value())
                where = (f"{len(curves['sites'])} typed-in sites at "
                         f"{self.pm_T_ref_spin.value():g} K, "
                         f"{pmr.SCALING_MODELS[curves['model']].split(':')[0]}")
            else:
                src = self._dft_source()
                if src is None:
                    raise ValueError("Read an ORCA output with hyperfine tensors or a CRYSTAL "
                                     "calculation in Simulate ▸ DFT tensors first (step 1), and "
                                     "set its magnetic model or scaling (step 2), or type the "
                                     "shifts in (Source).")
                if not src["labels"]:
                    raise ValueError("Tick at least one site in Simulate ▸ DFT tensors.")
                if src.get("kind") == "crystal":
                    args = dict(src["args"])
                    comps = args.pop("components", "both")
                    curves = pmr.crystal_shift_vs_temperature(
                        src["calc"], T, scaling=args, labels=src["labels"],
                        orbital_shift=src["orbital"], components=comps,
                        T_ref=self.pm_T_ref_spin.value(), theta=self.pm_T_theta_spin.value())
                    where = (f"CRYSTAL, {os.path.basename(src['path'])}: δ ∝ φ(T) ∝ "
                             f"1/(T − θ), θ = {curves['theta']:g} K, from "
                             f"{curves['T_ref']:g} K")
                else:
                    args = dict(src["args"])
                    args.pop("components", None)
                    args.pop("T", None)
                    curves = pmr.shift_vs_temperature(
                        src["calc"], T, labels=src["labels"], orbital_shift=src["orbital"], **args)
                    where = (f"ORCA, {os.path.basename(src['path'])}: model "
                             f"{src['args'].get('model')}")
        except (ValueError, KeyError) as exc:
            if not quiet:
                QMessageBox.warning(self, "Shift against temperature", str(exc))
            return False
        self.temperature_curves = curves
        self.pm_T_source_label.setText(f"Calculated: {where}.")
        self.plot_temperature()
        return True

    def fit_temperature(self, quiet=False) -> bool:
        try:
            T, d = self.temperature_data()
            dia = self.pm_T_dia_spin.value() if self.pm_T_fix_dia_check.isChecked() else None
            fit = pmr.fit_temperature(T, d, self.pm_T_model_combo.currentData(), delta_dia=dia)
        except (ValueError, RuntimeError) as exc:
            if not quiet:
                QMessageBox.warning(self, "Fit δ(T)", str(exc))
            return False
        self.temperature_fit = fit
        names = {"delta_dia": "δ_dia (ppm)", "a": "a (ppm K)", "b": "b (ppm K²)",
                 "C": "C (ppm K)", "theta": "θ (K)"}
        bits = [f"{names[k]} = {v:.6g}" + (f" ± {fit['errors'][k]:.2g}"
                                           if fit["errors"].get(k) else "")
                for k, v in fit["params"].items()]
        text = f"{fit['equation']}: " + "; ".join(bits) + f"; R² = {fit['r2']:.5f}"
        if fit["model"] == "curie_bleaney":
            Tm = float(np.mean(T))
            text += (f". At T = {Tm:.0f} K: a/T = {fit['params']['a'] / Tm:.4g} ppm (contact-like), "
                     f"b/T² = {fit['params']['b'] / Tm ** 2:.4g} ppm (pseudocontact-like)")
        self.pm_T_result_label.setText(text + "." + "".join(f"\n• {n}" for n in fit["notes"]))
        self.plot_temperature()
        return True

    def plot_temperature(self) -> bool:
        data, fits = {}, {}
        try:
            T, d = self.temperature_data()
        except ValueError:
            T = np.zeros(0)
        name = self.pm_T_name_edit.text().strip() or "measured"
        if len(T):
            data[name] = (T, d)
            if self.temperature_fit is not None:
                fits[name] = self.temperature_fit
        if not data and self.temperature_curves is None:
            return False
        curves = self.temperature_curves
        axis = self.pm_T_axis_combo.currentData()
        comps = self.pm_T_components_check.isChecked()

        def draw(figure):
            pmr.plot_shift_temperature(curves, data, fits, figure=figure, x_axis=axis,
                                       fontsize=self._fontsize(), components=comps)
        return self._show(draw)

    def temperature_python(self) -> str:
        lines = ["from opennmr.notebook import paramagnetic_nmr as pm", "import numpy as np"]
        src = self._dft_source()
        grid = (f"T = np.linspace({self.pm_T_min_spin.value()!r}, "
                f"{self.pm_T_max_spin.value()!r}, {int(self.pm_T_points_spin.value())})")
        typed = self.pm_T_source_combo.currentData() == "typed"
        if self.temperature_curves is not None and typed:
            lines += [grid, f"sites = {self.calculated_sites()!r}",
                      f"curves = pm.scaled_shift_vs_temperature(sites, T, "
                      f"T_ref={self.pm_T_ref_spin.value()!r}, "
                      f"model={self.pm_T_scaling_combo.currentData()!r}, "
                      f"theta={self.pm_T_theta_spin.value()!r})"]
        elif self.temperature_curves is not None and src is not None:
            lines += ["from opennmr.notebook import dft_nmr",
                      f"calc = dft_nmr.read_calculation({src['path']!r})", grid]
            if src.get("kind") == "crystal":
                args = dict(src["args"])
                comps = args.pop("components", "both")
                lines.append(f"curves = pm.crystal_shift_vs_temperature(calc, T, scaling={args!r}, "
                             f"labels={src['labels']!r}, orbital_shift={src['orbital']!r}, "
                             f"components={comps!r}, T_ref={self.pm_T_ref_spin.value()!r}, "
                             f"theta={self.pm_T_theta_spin.value()!r})")
            else:
                args = {k: v for k, v in src["args"].items() if k not in ("components", "T")}
                lines.append(f"curves = pm.shift_vs_temperature(calc, T, labels={src['labels']!r}, "
                             f"orbital_shift={src['orbital']!r}, "
                             + ", ".join(f"{k}={v!r}" for k, v in args.items()) + ")")
        else:
            lines.append("curves = None")
        T, d = self.temperature_data()
        dia = self.pm_T_dia_spin.value() if self.pm_T_fix_dia_check.isChecked() else None
        name = self.pm_T_name_edit.text().strip() or "measured"
        if len(T):
            lines += [f"T_exp = np.array({T.tolist()!r})", f"d_exp = np.array({d.tolist()!r})",
                      f"fit = pm.fit_temperature(T_exp, d_exp, "
                      f"{self.pm_T_model_combo.currentData()!r}, delta_dia={dia!r})",
                      "print(fit['equation'], fit['params'], fit['errors'])",
                      f"fig = pm.plot_shift_temperature(curves, {{{name!r}: (T_exp, d_exp)}}, "
                      f"{{{name!r}: fit}}, x_axis={self.pm_T_axis_combo.currentData()!r})"]
        else:
            lines.append(f"fig = pm.plot_shift_temperature(curves, x_axis="
                         f"{self.pm_T_axis_combo.currentData()!r})")
        return "\n".join(lines) + "\n"

    def _copy_safe(self, make) -> None:
        try:
            self._copy(make())
        except (ValueError, KeyError, TypeError) as exc:
            QMessageBox.warning(self, "Copy as Python", str(exc))

    def _browse(self, edit, title, filt=_CALC_FILTER) -> None:
        path, _ = QFileDialog.getOpenFileName(self, title, os.path.dirname(edit.text()), filt)
        if path:
            edit.setText(path)

    # ==== 2. bulk magnetic susceptibility (BMS) shifts ===========================

    BMS_GRIDS = (64, 96, 128, 160, 192, 256)

    #: Length units for the particle sizes and the simulated region:
    #: key -> (text, metres).
    LENGTH_UNITS = {"nm": ("nm", 1e-9), "um": ("µm", 1e-6), "mm": ("mm", 1e-3)}

    #: What each Packing method does (its tooltip), and what both share.
    PACKER_TIPS = {
        "grow": ("Grow and separate (force-biased, Jodrey–Tory style): every particle starts "
                 "small and grows towards its drawn size together, overlapping ones being "
                 "pushed apart. Reaches dense random packings (about 55–64 % for spheres). A "
                 "particle that runs out of room stops growing, so some can end up smaller than "
                 "drawn; the result says how many."),
        "rsa": ("Random sequential addition: particles are placed one at a time at random, at "
                "their full drawn size, and a placement that overlaps an earlier one is "
                "rejected. Never shrinks a particle, but jams at about 38 % for spheres, so not "
                "every particle may fit; the result says how many were placed."),
        "both": ("Both test overlaps with each particle's bounding sphere, so elongated "
                 "particles and cubes (whose corners reach √3 × half an edge) pack less densely "
                 "than they really could."),
    }

    #: What the three sizes are for each shape (the Mean size row's label).
    SIZE_MEANINGS = {"ellipsoid": "semi-axes", "cuboid": "edge lengths",
                     "hexagonal": "apothem, –, height", "donut": "radii R, r, –"}

    def _build_bms(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        hint = QLabel(
            "The spectrum of the bulk magnetic susceptibility (BMS) shifts in a packing of "
            "magnetised particles. Particles fill a small simulated region of the powder, "
            "repeated periodically; the sample container (rotor, tube or plate) sets the "
            "macroscopic shift around it. Every particle shares one set of FFTs, so the time "
            "depends on the grid, not on the number of particles.")
        hint.setWordWrap(True)
        v.addWidget(hint)

        pk = QGroupBox("Particles")
        pf = QFormLayout(pk)
        self._bms_particle_form = pf
        self.pm_bms_source_combo = QComboBox()
        self.pm_bms_source_combo.addItem("Generate a random packing", "pack")
        self.pm_bms_source_combo.addItem("LAMMPS dump file", "dump")
        pf.addRow("Particles:", self.pm_bms_source_combo)
        self.pm_bms_dump_edit = QLineEdit()
        self.pm_bms_dump_edit.setPlaceholderText("LAMMPS dump (ITEM: ATOMS id x y z v1 v2 v3 qi qj qk qr)")
        b = QPushButton("Browse…")
        b.clicked.connect(lambda: self._browse(self.pm_bms_dump_edit, "LAMMPS dump",
                                               "LAMMPS dump (*.dump *.txt *.lammpstrj);;All files (*)"))
        self._bms_dump_row = QWidget()
        h = QHBoxLayout(self._bms_dump_row)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(self.pm_bms_dump_edit, 1)
        h.addWidget(b)
        pf.addRow("File:", self._bms_dump_row)
        self.pm_bms_n_spin = _spin_box(1, 20000, 150, 0, step=10,
                                       tip="Number of particles in the simulated region.")
        pf.addRow("Number:", self.pm_bms_n_spin)
        self.pm_bms_equation_combo = QComboBox()
        for text, key, tip in (
                ("Ellipsoid (sphere when all equal)", "ellipsoid",
                 "Sizes are the three semi-axes; a sphere of 10 µm diameter is 5, 5, 5."),
                ("Cuboid (cube when all equal)", "cuboid",
                 "Sizes are the three edge lengths; a cube of 10 µm edge is 10, 10, 10."),
                ("Hexagonal prism", "hexagonal",
                 "Sizes: the hexagon's apothem (centre to flat side), unused, the height."),
                ("Torus", "donut", "Sizes: the major radius R, the tube radius r, unused.")):
            self.pm_bms_equation_combo.addItem(text, key)
            self.pm_bms_equation_combo.setItemData(self.pm_bms_equation_combo.count() - 1, tip,
                                                   Qt.ItemDataRole.ToolTipRole)
        self.pm_bms_equation_combo.setToolTip("The particles' shape; hover over each for what "
                                              "its three sizes mean.")
        pf.addRow("Shape:", self.pm_bms_equation_combo)
        # Sizes in a named length unit (explicit user request, 2026-11-10:
        # "most people know (or can guess) their particle sizes as nm, µm").
        self.pm_bms_length_unit_combo = QComboBox()
        for key, (text, _metres) in self.LENGTH_UNITS.items():
            self.pm_bms_length_unit_combo.addItem(text, key)
        self.pm_bms_length_unit_combo.setCurrentIndex(
            self.pm_bms_length_unit_combo.findData("um"))
        self.pm_bms_length_unit_combo.setToolTip(
            "The unit of the particle sizes and of the simulated region. Changing it keeps the "
            "physical sizes (the numbers are converted). The BMS field depends only on shapes "
            "and proportions, so the unit does not change the shifts; it names the sizes and "
            "lets them be checked against the sample container's.")
        self._bms_length_unit = "um"
        pf.addRow("Length unit:", self.pm_bms_length_unit_combo)
        self.pm_bms_size_edit = QLineEdit("25, 25, 25")
        self.pm_bms_size_edit.setToolTip(
            "Mean size per axis (v1, v2, v3), in the length unit above: semi-axes of an "
            "ellipsoid; edge lengths of a cuboid; apothem, unused, height of a hexagonal prism; "
            "major and minor radii of a torus. One value for all three.")
        pf.addRow("Mean size:", self.pm_bms_size_edit)
        self.pm_bms_spread_edit = QLineEdit("4, 4, 4")
        self.pm_bms_spread_edit.setToolTip(
            "Spread per axis, in the same unit and the same sense as the sizes: the standard "
            "deviation (Gaussian) or the half width at half maximum (Lorentzian, pseudo-Voigt, "
            "which have no standard deviation).")
        pf.addRow("Spread:", self.pm_bms_spread_edit)
        self.pm_bms_dist_combo = QComboBox()
        for text, key in (("Gaussian", "gaussian"), ("Lorentzian", "lorentzian"),
                          ("Pseudo-Voigt", "pseudo_voigt")):
            self.pm_bms_dist_combo.addItem(text, key)
        pf.addRow("Size distribution:", self.pm_bms_dist_combo)
        self.pm_bms_packer_combo = QComboBox()
        for text, key, tip in (("Grow and separate (dense)", "grow", self.PACKER_TIPS["grow"]),
                               ("Random sequential addition (≤ ~38 %)", "rsa",
                                self.PACKER_TIPS["rsa"])):
            self.pm_bms_packer_combo.addItem(text, key)
            self.pm_bms_packer_combo.setItemData(self.pm_bms_packer_combo.count() - 1, tip,
                                                 Qt.ItemDataRole.ToolTipRole)
        self.pm_bms_packer_combo.currentIndexChanged.connect(
            lambda _i: self.pm_bms_packer_combo.setToolTip(
                self.PACKER_TIPS.get(self.pm_bms_packer_combo.currentData(), "")
                + "\n\n" + self.PACKER_TIPS["both"]))
        self.pm_bms_packer_combo.setToolTip(self.PACKER_TIPS["grow"] + "\n\n"
                                            + self.PACKER_TIPS["both"])
        pf.addRow("Packing:", self.pm_bms_packer_combo)
        self.pm_bms_seed_spin = _spin_box(0, 2 ** 31 - 1, 1, 0, step=1,
                                          tip="Random seed: the same seed gives the same packing.")
        pf.addRow("Seed:", self.pm_bms_seed_spin)
        # The simulated region, said for what it is: a small periodic piece
        # of the powder, not a container (explicit user request, 2026-11-10:
        # "what is the size of the box" beside "what sample container do we
        # have" read as two containers). Its side follows from the packing
        # fraction unless set.
        self.pm_bms_region_mode_combo = QComboBox()
        self.pm_bms_region_mode_combo.addItem("From the packing fraction", "fraction")
        self.pm_bms_region_mode_combo.addItem("Set its side", "side")
        self.pm_bms_region_mode_combo.setToolTip(
            "The particles fill a small cubic region of the powder, which repeats "
            "periodically (with Boundary: Periodic) to stand for the powder inside the sample "
            "container. It is not the container: that is set under Sample container, and only "
            "its shape matters. From the packing fraction: the side is worked out from the "
            "number, size and shape of the particles. Set its side: type it.")
        pf.addRow("Simulated region:", self.pm_bms_region_mode_combo)
        self.pm_bms_fraction_spin = _spin_box(0.01, 0.74, 0.30, 3, step=0.05,
                                              tip="The volume fraction of the powder that is "
                                                  "particles (a loose powder about 0.3, a "
                                                  "pressed one 0.5–0.6). Sets the region's "
                                                  "side; the packing reports what it reached.")
        pf.addRow("Packing fraction:", self.pm_bms_fraction_spin)
        self.pm_bms_box_spin = _spin_box(1e-3, 1e9, 300.0, 3, step=10.0,
                                         tip="Side of the cubic simulated region, in the length "
                                             "unit above. Typing a side switches to Set its "
                                             "side.")
        self.pm_bms_box_spin.setSuffix(" µm")
        pf.addRow("Region side:", self.pm_bms_box_spin)
        self._bms_setting_box = False
        self.pm_bms_deform_combo = QComboBox()
        for key in "ABC":
            self.pm_bms_deform_combo.addItem(key, key)
        self.pm_bms_deform_combo.setToolTip("Which particle axis the prism or torus axis lies "
                                            "along (A: x, B: y, C: z of the particle).")
        self.pm_bms_region_note = _label()
        pf.addRow("Orientation (A/B/C):", self.pm_bms_deform_combo)
        self.pm_bms_slope_spin = _spin_box(2.0, 1e5, 2000.0, 0, step=100.0,
                                           tip="Edge steepness of the shape function (large = sharp).")
        pf.addRow("Slope filter:", self.pm_bms_slope_spin)
        self.pm_bms_pack_button = QPushButton("Generate Packing")
        self.pm_bms_pack_button.clicked.connect(lambda: self.generate_bms_packing())
        pf.addRow("", self.pm_bms_pack_button)
        pf.addRow(self.pm_bms_region_note)
        v.addWidget(pk)

        gb = QGroupBox("Susceptibility and grid")
        gf = QFormLayout(gb)
        self._bms_grid_form = gf
        self.pm_bms_chi_edit = QLineEdit("1e-3, 1e-3, 1e-3")
        self.pm_bms_chi_edit.setToolTip(
            "Principal values χ1, χ2, χ3 of the particles' susceptibility, in the units chosen "
            "next to it, in each particle's own frame. One value: isotropic.")
        self.pm_bms_chi_unit_combo = QComboBox()
        for key, text in chi_units.UNITS.items():
            self.pm_bms_chi_unit_combo.addItem(text, key)
        self.pm_bms_chi_unit_combo.setToolTip(
            "SI volume (dimensionless) is what the simulation uses; the others are converted "
            "(cgs volume × 4π; mass × density; molar ÷ molar volume; μ_eff by the Curie–Weiss "
            "law).")
        # Each on its own row: a combo whose text changes must not share a row that
        # the narrow-panel reflow wraps (it was then drawn under the next row and
        # could not be clicked again, 2026-11-07).
        gf.addRow("χ:", self.pm_bms_chi_edit)
        gf.addRow("Units:", self.pm_bms_chi_unit_combo)
        self.pm_bms_density_spin = _spin_box(0.0, 100.0, 0.0, 4, " g cm⁻³", step=0.1,
                                             tip="Density of the solid itself (the crystal "
                                                 "density), not a tap or packing density: the "
                                                 "packing is simulated.")
        gf.addRow("Density:", self.pm_bms_density_spin)
        self.pm_bms_vm_combo = QComboBox()
        self.pm_bms_vm_combo.addItem("Molar mass and density", "mass")
        self.pm_bms_vm_combo.addItem("Unit cell volume and Z", "cell")
        self.pm_bms_vm_combo.setToolTip("The volume of one mole of formula units, which a molar "
                                        "susceptibility is divided by.")
        gf.addRow("Molar volume from:", self.pm_bms_vm_combo)
        self.pm_bms_molar_mass_spin = _spin_box(0.0, 1e5, 0.0, 4, " g mol⁻¹", step=1.0,
                                                tip="Molar mass of one formula unit.")
        gf.addRow("Molar mass:", self.pm_bms_molar_mass_spin)
        self.pm_bms_cell_spin = _spin_box(0.0, 1e7, 0.0, 3, " Å³", step=10.0,
                                          tip="Unit cell volume.")
        gf.addRow("Unit cell volume:", self.pm_bms_cell_spin)
        self.pm_bms_Z_spin = _spin_box(0.0, 1000.0, 0.0, 0, step=1,
                                       tip="Formula units per unit cell (the formula the molar "
                                           "susceptibility refers to).")
        gf.addRow("Z:", self.pm_bms_Z_spin)
        self.pm_bms_chi_T_spin = _spin_box(0.1, 3000.0, 298.0, 2, " K", step=1.0,
                                           tip="The temperature of the experiment.")
        gf.addRow("Temperature:", self.pm_bms_chi_T_spin)
        self.pm_bms_chi_theta_spin = _spin_box(-1e5, 1e5, 0.0, 2, " K", step=1.0,
                                               tip="Weiss constant θ (negative: "
                                                   "antiferromagnetic); 0: a Curie law.")
        gf.addRow("Weiss constant θ:", self.pm_bms_chi_theta_spin)
        self.pm_bms_nions_spin = _spin_box(0.0, 1000.0, 1.0, 3, step=1.0,
                                           tip="Paramagnetic ions per formula unit (e.g. 1 for "
                                               "LiMnPO₄, 0.5 for LiMn₀.₅Fe₀.₅PO₄ if only Mn is "
                                               "counted).")
        gf.addRow("Ions per formula unit:", self.pm_bms_nions_spin)
        self.pm_bms_chidia_spin = _spin_box(-1e3, 1e3, 0.0, 8, " emu mol⁻¹", step=1e-6,
                                            tip="Optional diamagnetic (core) susceptibility of "
                                                "the formula unit, added to the Curie–Weiss "
                                                "term.")
        gf.addRow("χ_dia:", self.pm_bms_chidia_spin)
        self.pm_bms_chi_si_label = _label()
        gf.addRow(self.pm_bms_chi_si_label)
        self.pm_bms_grid_combo = QComboBox()
        for n in self.BMS_GRIDS:
            self.pm_bms_grid_combo.addItem(f"{n}³", n)
        self.pm_bms_grid_combo.setCurrentIndex(self.BMS_GRIDS.index(128))
        self.pm_bms_grid_combo.setToolTip("Grid points per axis. Converged when a finer grid no "
                                          "longer changes the spectrum; aim for ≥ 8 points across "
                                          "the smallest particle.")
        gf.addRow("Grid:", self.pm_bms_grid_combo)
        self.pm_bms_boundary_combo = QComboBox()
        self.pm_bms_boundary_combo.addItem("Periodic (the packing fills the sample)", "periodic")
        self.pm_bms_boundary_combo.addItem("Isolated cluster in vacuum (8× the voxels)", "isolated")
        gf.addRow("Boundary:", self.pm_bms_boundary_combo)
        self.pm_bms_double_check = QCheckBox("Double precision (slower; single is ample)")
        gf.addRow("", self.pm_bms_double_check)
        v.addWidget(gb)

        smp = QGroupBox("Sample container")
        smf = QFormLayout(smp)
        self._bms_sample_form = smf
        self.pm_bms_sample_combo = QComboBox()
        for text, key in (("MAS rotor", "rotor"), ("Tube or other cylinder", "cylinder"),
                          ("Plate (rectangular block)", "plate"),
                          ("Sphere (no macroscopic shift)", "sphere")):
            self.pm_bms_sample_combo.addItem(text, key)
        self.pm_bms_sample_combo.setToolTip(
            "What holds the powder, and so the shape of the packed powder. Its own "
            "demagnetising field shifts each part of the sample differently: −χ_v (N(r) − 1/3), "
            "χ_v the powder's volume susceptibility. Only the shape matters (its proportions), "
            "not its size. Its dimensions are in mm.")
        smf.addRow("Container:", self.pm_bms_sample_combo)
        self.pm_bms_rotor_combo = QComboBox()
        for od in bms_sample.ROTORS:
            self.pm_bms_rotor_combo.addItem(f"{od:g} mm", od)
        self.pm_bms_rotor_combo.setCurrentIndex(list(bms_sample.ROTORS).index(3.2))
        self.pm_bms_rotor_combo.setToolTip("Outer diameter; choosing one fills in a typical inner "
                                           "diameter (less the wall) and packed length.")
        smf.addRow("Rotor:", self.pm_bms_rotor_combo)
        inner0, len0 = bms_sample.ROTORS[3.2]
        self.pm_bms_inner_spin = _spin_box(0.01, 100.0, inner0, 3, " mm", step=0.1,
                                           tip="Inner diameter of the packed volume: the rotor's "
                                               "outer diameter less twice the wall.")
        smf.addRow("Inner diameter:", self.pm_bms_inner_spin)
        self.pm_bms_length_spin = _spin_box(0.01, 500.0, len0, 3, " mm", step=0.5,
                                            tip="Length of the packed powder (between the caps "
                                                "or spacers).")
        smf.addRow("Packed length:", self.pm_bms_length_spin)
        self.pm_bms_plate_edit = QLineEdit("10, 10, 1")
        self.pm_bms_plate_edit.setToolTip("Side lengths a, b and the thickness c (along the plate "
                                          "normal), mm.")
        smf.addRow("Plate a, b, thickness (mm):", self.pm_bms_plate_edit)
        self.pm_bms_angle_spin = _spin_box(0.0, 90.0, bms_sample.MAGIC_ANGLE_DEG, 3, "°",
                                           step=1.0,
                                           tip="Static only: the angle between B₀ and the "
                                               "cylinder axis or plate normal. A rotor at rest "
                                               "in a MAS probe: 54.74°; a tube in a solution or "
                                               "static probe: 0°; a plate lying in a horizontal "
                                               "coil: 90°. Under MAS the rotor is at the magic "
                                               "angle and turns.")
        smf.addRow("Axis angle to B₀ (static):", self.pm_bms_angle_spin)
        self.pm_bms_rotor_check = QCheckBox("Include the rotor's own field (wall and caps)")
        self.pm_bms_rotor_check.setChecked(True)
        self.pm_bms_rotor_check.setToolTip(
            "The rotor is magnetised too: its wall (a tube over the rotor's length) and the caps "
            "or spacers filling the bore beyond the powder shift the sample, most near its ends. "
            "A long tube has no field in its bore, so the wall acts through its ends. Shifts are "
            "relative to the bare sample; an external reference in the same rotor cancels the "
            "uniform part.")
        smf.addRow("", self.pm_bms_rotor_check)
        self.pm_bms_rotor_length_spin = _spin_box(0.01, 500.0, len0 + 2 * 3.2, 3, " mm",
                                                  step=0.5,
                                                  tip="Length of the rotor tube (the wall). "
                                                      "Typical: the packed length plus about one "
                                                      "outer diameter at each end.")
        smf.addRow("Rotor length:", self.pm_bms_rotor_length_spin)
        self.pm_bms_wall_combo, self.pm_bms_wall_chi_spin = self._material_row(
            smf, "Rotor wall:", "zirconia",
            exclude=("none",), tip="The wall's volume susceptibility (SI, ppm).")
        self.pm_bms_caps_combo, self.pm_bms_caps_chi_spin = self._material_row(
            smf, "Caps and spacers:", "vespel",
            tip="What fills the bore above and below the powder (caps, spacers, inserts); "
                "'Nothing' for an empty bore.")
        self.pm_bms_sample_note = _label()
        smf.addRow(self.pm_bms_sample_note)
        v.addWidget(smp)

        lc = QGroupBox("Local interactions of the observed nuclei")
        lc.setCheckable(True)
        lc.setChecked(False)
        lc.setToolTip("Interactions every observed nucleus has besides the BMS shift, fixed in "
                      "its particle's frame (the principal frame of the particle's χ): the "
                      "Fermi contact (isotropic) shift, the spin-dipolar anisotropy, a "
                      "quadrupolar coupling. Nuclei between the particles take them in random "
                      "orientations.")
        self.pm_bms_local_group = lc
        lf = QFormLayout(lc)
        self._bms_local_form = lf
        self.pm_bms_contact_spin = _spin_box(-1e6, 1e6, 0.0, 3, " ppm", step=10.0,
                                             tip="Isotropic shift: Fermi contact plus any "
                                                 "orbital (diamagnetic) shift.")
        lf.addRow("Contact shift δ_iso:", self.pm_bms_contact_spin)
        self.pm_bms_aniso_spin = _spin_box(-1e6, 1e6, 0.0, 3, " ppm", step=10.0,
                                           tip="Spin-dipolar shift anisotropy δ (Haeberlen: "
                                               "δ_zz − δ_iso).")
        lf.addRow("Spin-dipolar anisotropy δ:", self.pm_bms_aniso_spin)
        self.pm_bms_eta_spin = _spin_box(0.0, 1.0, 0.0, 3, step=0.05)
        lf.addRow("Asymmetry η:", self.pm_bms_eta_spin)
        self.pm_bms_euler_edit = QLineEdit("0, 0, 0")
        self.pm_bms_euler_edit.setToolTip("Euler angles α, β, γ (ZYZ, degrees) of the shift "
                                          "tensor's principal frame in the particle's frame.")
        lf.addRow("Its Euler angles (°):", self.pm_bms_euler_edit)
        self.pm_bms_cq_spin = _spin_box(-100.0, 100.0, 0.0, 4, " MHz", step=0.1,
                                        tip="Quadrupolar coupling constant C_Q (0: none).")
        lf.addRow("Quadrupolar C_Q:", self.pm_bms_cq_spin)
        self.pm_bms_etaq_spin = _spin_box(0.0, 1.0, 0.0, 3, step=0.05)
        lf.addRow("Asymmetry η_Q:", self.pm_bms_etaq_spin)
        self.pm_bms_eulerq_edit = QLineEdit("0, 0, 0")
        self.pm_bms_eulerq_edit.setToolTip("Euler angles α, β, γ (ZYZ, degrees) of the EFG "
                                           "principal frame in the particle's frame.")
        lf.addRow("Its Euler angles (°):", self.pm_bms_eulerq_edit)
        self.pm_bms_transitions_combo = QComboBox()
        self.pm_bms_transitions_combo.addItem("Central transition", "ct")
        self.pm_bms_transitions_combo.addItem("All transitions", "all")
        lf.addRow("Transitions:", self.pm_bms_transitions_combo)
        self.pm_bms_qorder_combo = QComboBox()
        for text, key in (("Second order", "second"), ("First order", "first"),
                          ("Exact", "exact")):
            self.pm_bms_qorder_combo.addItem(text, key)
        lf.addRow("Quadrupolar theory:", self.pm_bms_qorder_combo)
        v.addWidget(lc)

        sb = QGroupBox("Spectrum")
        sf = QFormLayout(sb)
        self._bms_spec_form = sf
        # Chosen, not typed (explicit user request, 2026-11-10): every
        # NMR-active isotope, as in Solid-state NMR's own nucleus list.
        self.pm_bms_nucleus_combo = QComboBox()
        self.pm_bms_nucleus_combo.addItems(_nuclides())
        index = self.pm_bms_nucleus_combo.findText("7Li")
        self.pm_bms_nucleus_combo.setCurrentIndex(max(index, 0))
        self.pm_bms_nucleus_combo.setMaxVisibleItems(20)
        self.pm_bms_nucleus_combo.setToolTip("The observed nucleus: sets the Larmor frequency, "
                                             "and with a quadrupolar coupling its spin.")
        sf.addRow("Nucleus:", self.pm_bms_nucleus_combo)
        self.pm_bms_B0_spin = _spin_box(0.01, 40.0, 9.4, 4, " T", step=0.1)
        sf.addRow("B₀:", self.pm_bms_B0_spin)
        self.pm_bms_mode_combo = QComboBox()
        for key, text in bmsf.SPECTRUM_MODES.items():
            self.pm_bms_mode_combo.addItem(text, key)
        sf.addRow("Experiment:", self.pm_bms_mode_combo)
        self.pm_bms_rate_spin = _spin_box(1.0, 2e5, 10000.0, 0, " Hz", step=1000.0)
        sf.addRow("MAS rate:", self.pm_bms_rate_spin)
        self.pm_bms_order_spin = _spin_box(1, 200, 16, 0, step=1,
                                           tip="Sidebands kept on each side; what lies beyond is "
                                               "reported.")
        sf.addRow("Sideband orders:", self.pm_bms_order_spin)
        self.pm_bms_samples_spin = _spin_box(0, 1e8, 0, 0, step=50000,
                                             tip="Samples of the sampled spectrum (a static "
                                                 "spectrum, or a rotor, plate or local "
                                                 "interactions under MAS). Automatic: 1,000,000 "
                                                 "static, 200,000 MAS. The result reports the "
                                                 "Monte Carlo noise: raise it if that is not "
                                                 "small.")
        self.pm_bms_samples_spin.setSpecialValueText("Automatic")
        sf.addRow("Samples:", self.pm_bms_samples_spin)
        self.pm_bms_observe_combo = QComboBox()
        for key, text in bmsf.OBSERVE.items():
            self.pm_bms_observe_combo.addItem(text, key)
        self.pm_bms_observe_combo.setToolTip("Where the observed nuclei are: in the particles, or "
                                             "in what is between them (electrolyte, binder).")
        sf.addRow("Observed nuclei:", self.pm_bms_observe_combo)
        self.pm_bms_lw_spin = _spin_box(0.0, 1e6, 50.0, 1, " Hz", step=10.0,
                                        tip="Full width at half maximum of the line every voxel "
                                            "gives (other broadening).")
        sf.addRow("Line width:", self.pm_bms_lw_spin)
        self.pm_bms_shape_combo = QComboBox()
        self.pm_bms_shape_combo.addItem("Lorentzian", "lorentzian")
        self.pm_bms_shape_combo.addItem("Gaussian", "gaussian")
        sf.addRow("Line shape:", self.pm_bms_shape_combo)
        v.addWidget(sb)

        row = QHBoxLayout()
        self.pm_bms_run_button = QPushButton("Simulate")
        self.pm_bms_run_button.setToolTip("The field is kept: changing only the sample, the "
                                          "local interactions or the spectrum settings does "
                                          "not recalculate it.")
        self.pm_bms_run_button.clicked.connect(lambda: self.simulate_bms())
        row.addWidget(self.pm_bms_run_button)
        row.addWidget(QLabel("Show:"))
        self.pm_bms_view_combo = QComboBox()
        for text, key in (("Spectrum", "spectrum"), ("Field slice", "slice"),
                          ("Packing (3-D)", "packing")):
            self.pm_bms_view_combo.addItem(text, key)
        self.pm_bms_view_combo.setToolTip("What the Simulation plot shows, from the last run: "
                                          "nothing is simulated again.")
        self.pm_bms_view_combo.activated.connect(lambda _i: self.show_bms_view())
        row.addWidget(self.pm_bms_view_combo)
        self.pm_bms_keep_button = QPushButton("Keep in Open Experiments")
        self.pm_bms_keep_button.setToolTip(
            "Add the simulated spectrum to Open experiments (with its simulation settings in "
            "its history), to compare with measured spectra or process it like any other.")
        self.pm_bms_keep_button.setEnabled(False)
        self.pm_bms_keep_button.clicked.connect(lambda: self.keep_bms_spectrum())
        row.addWidget(self.pm_bms_keep_button)
        self.pm_bms_python_button = QPushButton("Copy as Python")
        self.pm_bms_python_button.clicked.connect(lambda: self._copy_safe(self.bms_python))
        row.addWidget(self.pm_bms_python_button)
        row.addStretch(1)
        v.addLayout(row)
        self._bms_slice_row = QWidget()
        sr = QHBoxLayout(self._bms_slice_row)
        sr.setContentsMargins(0, 0, 0, 0)
        sr.addWidget(QLabel("Slice:"))
        self.pm_bms_plane_combo = QComboBox()
        for text, key in (("xz (B₀ vertical)", "y"), ("yz (B₀ vertical)", "x"),
                          ("xy (B₀ out of the page)", "z")):
            self.pm_bms_plane_combo.addItem(text, key)
        self.pm_bms_plane_combo.currentIndexChanged.connect(lambda _i: self._on_slice_changed())
        sr.addWidget(self.pm_bms_plane_combo)
        self.pm_bms_quantity_combo = QComboBox()
        self.pm_bms_quantity_combo.addItem("δ_zz (B₀ along z)", "delta_zz")
        self.pm_bms_quantity_combo.addItem("δ_iso", "delta_iso")
        self.pm_bms_quantity_combo.currentIndexChanged.connect(lambda _i: self._on_slice_changed())
        sr.addWidget(self.pm_bms_quantity_combo)
        self.pm_bms_slice_slider = QSlider(Qt.Orientation.Horizontal)
        self.pm_bms_slice_slider.setRange(0, 127)
        self.pm_bms_slice_slider.setValue(64)
        self.pm_bms_slice_slider.setToolTip("Where the slice cuts the box.")
        self.pm_bms_slice_slider.valueChanged.connect(lambda _v: self._on_slice_changed())
        sr.addWidget(self.pm_bms_slice_slider, 1)
        v.addWidget(self._bms_slice_row)
        self.pm_bms_result_label = _label()
        v.addWidget(self.pm_bms_result_label)
        for combo in (self.pm_bms_source_combo, self.pm_bms_mode_combo,
                      self.pm_bms_boundary_combo, self.pm_bms_equation_combo,
                      self.pm_bms_sample_combo, self.pm_bms_view_combo,
                      self.pm_bms_region_mode_combo):
            combo.currentIndexChanged.connect(lambda _i: self._sync_bms_rows())
        self.pm_bms_length_unit_combo.currentIndexChanged.connect(
            lambda _i: self._on_length_unit_changed())
        # The region's side follows the packing fraction (until one is typed).
        self.pm_bms_box_spin.valueChanged.connect(lambda _v: self._on_region_side_typed())
        for wdg in (self.pm_bms_n_spin, self.pm_bms_fraction_spin):
            wdg.valueChanged.connect(lambda _v: self._update_region())
        for edit in (self.pm_bms_size_edit,):
            edit.textChanged.connect(lambda _t: self._update_region())
        for combo in (self.pm_bms_equation_combo, self.pm_bms_region_mode_combo,
                      self.pm_bms_sample_combo):
            combo.currentIndexChanged.connect(lambda _i: self._update_region())
        self.pm_bms_local_group.toggled.connect(lambda _c: self._sync_bms_rows())
        self.pm_bms_rotor_check.toggled.connect(lambda _c: (self._sync_bms_rows(), self._update_sample_note()))
        self.pm_bms_chi_unit_combo.currentIndexChanged.connect(lambda _i: self._sync_chi_rows())
        self.pm_bms_vm_combo.currentIndexChanged.connect(lambda _i: self._sync_chi_rows())
        self.pm_bms_chi_edit.textChanged.connect(lambda _t: self._update_chi_label())
        for wdg in (self.pm_bms_density_spin, self.pm_bms_molar_mass_spin, self.pm_bms_cell_spin,
                    self.pm_bms_Z_spin, self.pm_bms_chi_T_spin, self.pm_bms_chi_theta_spin,
                    self.pm_bms_nions_spin, self.pm_bms_chidia_spin):
            wdg.valueChanged.connect(lambda _v: self._update_chi_label())
        self._sync_chi_rows()
        self.pm_bms_aniso_spin.valueChanged.connect(lambda _v: self._sync_bms_rows())
        self.pm_bms_cq_spin.valueChanged.connect(lambda _v: self._sync_bms_rows())
        self.pm_bms_rotor_combo.activated.connect(lambda _i: self._rotor_chosen())
        self.pm_bms_sample_combo.activated.connect(lambda _i: self._sample_shape_chosen())
        for wdg in (self.pm_bms_inner_spin, self.pm_bms_length_spin):
            wdg.valueChanged.connect(lambda _v: (self._update_sample_note(),
                                                 self._update_region_note()))
        self.pm_bms_plate_edit.textChanged.connect(lambda _t: self._update_region_note())
        self._sync_bms_rows()
        self._update_sample_note()
        self._update_region()
        return w

    # ---- the simulated region and the length unit -------------------------------

    def bms_length_unit(self) -> str:
        """The length unit's text ("µm")."""
        return self.LENGTH_UNITS[self.pm_bms_length_unit_combo.currentData()][0]

    def _on_length_unit_changed(self) -> None:
        """Re-express the sizes and the region side in the new unit, so the
        particles stay the same physical size."""
        old = self._bms_length_unit
        new = self.pm_bms_length_unit_combo.currentData()
        self._bms_length_unit = new
        if getattr(self, "_bms_restoring", False) or old == new:
            self._sync_bms_rows()
            return
        factor = self.LENGTH_UNITS[old][1] / self.LENGTH_UNITS[new][1]
        for edit in (self.pm_bms_size_edit, self.pm_bms_spread_edit):
            try:
                values = [_num(x, "size") for x in re.split(r"[,;\s]+", edit.text().strip()) if x]
            except ValueError:
                continue
            edit.setText(", ".join(f"{v * factor:.6g}" for v in values))
        self._bms_setting_box = True
        try:
            self.pm_bms_box_spin.setValue(self.pm_bms_box_spin.value() * factor)
        finally:
            self._bms_setting_box = False
        self._sync_bms_rows()
        self._update_region()

    def _on_region_side_typed(self) -> None:
        """A typed region side (not one worked out here) means Set its side."""
        if self._bms_setting_box:
            return
        index = self.pm_bms_region_mode_combo.findData("side")
        if self.pm_bms_region_mode_combo.currentIndex() != index:
            self.pm_bms_region_mode_combo.blockSignals(True)
            self.pm_bms_region_mode_combo.setCurrentIndex(index)
            self.pm_bms_region_mode_combo.blockSignals(False)
            self._sync_bms_rows()
        self._update_region_note()

    def _mean_half_sizes(self):
        """The mean sizes as the packer takes them (half sides for a cuboid)."""
        size = self._triplet(self.pm_bms_size_edit.text(), "Mean size")
        if self.pm_bms_equation_combo.currentData() == "cuboid":
            size = tuple(v / 2.0 for v in size)
        return size

    def _update_region(self) -> None:
        """Work out the region's side from the packing fraction, when that
        is how it is set: side = (N × mean particle volume / fraction)^(1/3)."""
        if not hasattr(self, "pm_bms_region_note"):
            return
        if (self.pm_bms_region_mode_combo.currentData() == "fraction"
                and self.pm_bms_source_combo.currentData() == "pack"):
            from ..notebook.bms import packing
            try:
                size = self._mean_half_sizes()
                volume = float(packing._particle_volume(
                    self.pm_bms_equation_combo.currentData(), *size))
                side = (self.pm_bms_n_spin.value() * volume
                        / self.pm_bms_fraction_spin.value()) ** (1.0 / 3.0)
            except (ValueError, ZeroDivisionError):
                side = None
            if side is not None and np.isfinite(side) and side > 0:
                self._bms_setting_box = True
                try:
                    self.pm_bms_box_spin.setValue(side)
                finally:
                    self._bms_setting_box = False
        self._update_region_note()

    def _container_min_mm(self):
        """The container's smallest inside dimension, mm (None for a sphere)."""
        shape = self.pm_bms_sample_combo.currentData()
        if shape in ("rotor", "cylinder"):
            return min(self.pm_bms_inner_spin.value(), self.pm_bms_length_spin.value())
        if shape == "plate":
            try:
                return min(self._triplet(self.pm_bms_plate_edit.text(), "Plate sides"))
            except ValueError:
                return None
        return None

    def _update_region_note(self) -> None:
        if not hasattr(self, "pm_bms_region_note"):
            return
        unit = self.bms_length_unit()
        side = self.pm_bms_box_spin.value()
        text = (f"Simulated region: a {side:.4g} {unit} cube of the powder, repeated "
                "periodically; it sits inside the sample container below, which sets the "
                "macroscopic shift.")
        metres = self.LENGTH_UNITS[self.pm_bms_length_unit_combo.currentData()][1]
        side_mm = side * metres / 1e-3
        container = self._container_min_mm()
        if container is not None and side_mm > container:
            text += (f" ⚠ It is larger than the container ({container:g} mm across): check the "
                     "length unit.")
        if self.pm_bms_equation_combo.currentData() == "cuboid":
            text += (" Cubes are kept apart by their corner-to-corner spheres, so they reach a "
                     "lower packing fraction than they could.")
        self.pm_bms_region_note.setText(text)

    def _sync_chi_rows(self) -> None:
        unit = self.pm_bms_chi_unit_combo.currentData()
        needs = chi_units.NEEDS[unit]
        gf = self._bms_grid_form
        molar = "molar_volume" in needs
        by_cell = molar and self.pm_bms_vm_combo.currentData() == "cell"
        gf.setRowVisible(self.pm_bms_density_spin, "density" in needs or (molar and not by_cell))
        gf.setRowVisible(self.pm_bms_vm_combo, molar)
        gf.setRowVisible(self.pm_bms_molar_mass_spin, molar and not by_cell)
        gf.setRowVisible(self.pm_bms_cell_spin, by_cell)
        gf.setRowVisible(self.pm_bms_Z_spin, by_cell)
        mu = unit == "mu_eff"
        for wdg in (self.pm_bms_chi_T_spin, self.pm_bms_chi_theta_spin, self.pm_bms_nions_spin,
                    self.pm_bms_chidia_spin):
            gf.setRowVisible(wdg, mu)
        self._update_chi_label()

    def chi_conversion(self) -> dict:
        """The keyword arguments of :func:`susceptibility.to_si_volume` the
        chosen unit needs (only those)."""
        unit = self.pm_bms_chi_unit_combo.currentData()
        needs = chi_units.NEEDS[unit]
        kw = {}
        if "density" in needs:
            kw["density"] = self.pm_bms_density_spin.value()
        if "molar_volume" in needs:
            if self.pm_bms_vm_combo.currentData() == "cell":
                kw.update(cell_volume=self.pm_bms_cell_spin.value(),
                          Z=self.pm_bms_Z_spin.value())
            else:
                kw.update(molar_mass=self.pm_bms_molar_mass_spin.value(),
                          density=self.pm_bms_density_spin.value())
        if unit == "mu_eff":
            kw.update(T=self.pm_bms_chi_T_spin.value(), theta=self.pm_bms_chi_theta_spin.value(),
                      n_ions=self.pm_bms_nions_spin.value(),
                      chi_dia=self.pm_bms_chidia_spin.value())
        return kw

    def bms_chi_si(self):
        """χ1, χ2, χ3 as SI volume susceptibilities (the simulation's)."""
        values = self._triplet(self.pm_bms_chi_edit.text(), "χ")
        unit = self.pm_bms_chi_unit_combo.currentData()
        si = chi_units.to_si_volume(values, unit, **self.chi_conversion())
        return tuple(float(v) for v in np.atleast_1d(si))

    def _update_chi_label(self) -> None:
        unit = self.pm_bms_chi_unit_combo.currentData()
        if unit == "si_volume":
            self.pm_bms_chi_si_label.setText("")
            return
        try:
            si = self.bms_chi_si()
            self.pm_bms_chi_si_label.setText(
                "→ χ_v (SI volume) = " + ", ".join(f"{v:.4g}" for v in si))
        except ValueError as exc:
            self.pm_bms_chi_si_label.setText(f"→ {exc}")

    def _sync_bms_rows(self) -> None:
        pack = self.pm_bms_source_combo.currentData() == "pack"
        pf = self._bms_particle_form
        pf.setRowVisible(self._bms_dump_row, not pack)
        for wdg in (self.pm_bms_n_spin, self.pm_bms_size_edit, self.pm_bms_spread_edit,
                    self.pm_bms_dist_combo, self.pm_bms_packer_combo, self.pm_bms_seed_spin,
                    self.pm_bms_pack_button):
            pf.setRowVisible(wdg, pack)
        equation = self.pm_bms_equation_combo.currentData()
        pf.setRowVisible(self.pm_bms_deform_combo, equation not in ("ellipsoid", "cuboid"))
        pf.setRowVisible(self.pm_bms_region_mode_combo, pack)
        pf.setRowVisible(self.pm_bms_fraction_spin,
                         pack and self.pm_bms_region_mode_combo.currentData() == "fraction")
        unit = self.bms_length_unit()
        meaning = self.SIZE_MEANINGS.get(equation, "sizes")
        for wdg, text in ((self.pm_bms_size_edit, f"Mean {meaning} ({unit}):"),
                          (self.pm_bms_spread_edit, f"Spread ({unit}):")):
            label = pf.labelForField(wdg)
            if label is not None:
                label.setText(text)
        self.pm_bms_box_spin.setSuffix(f" {unit}")
        self.pm_bms_box_spin.setEnabled(
            not pack or self.pm_bms_region_mode_combo.currentData() == "side")
        shape = self.pm_bms_sample_combo.currentData()
        smf = self._bms_sample_form
        smf.setRowVisible(self.pm_bms_rotor_combo, shape == "rotor")
        for wdg in (self.pm_bms_inner_spin, self.pm_bms_length_spin):
            smf.setRowVisible(wdg, shape in ("rotor", "cylinder"))
        # The receptacle's own dimensions, named for what it is.
        for wdg, (rotor_text, tube_text) in (
                (self.pm_bms_inner_spin, ("Rotor inner diameter:", "Tube inner diameter:")),
                (self.pm_bms_length_spin, ("Packed length:", "Filled height:"))):
            label = smf.labelForField(wdg)
            if label is not None:
                label.setText(rotor_text if shape == "rotor" else tube_text)
        smf.setRowVisible(self.pm_bms_plate_edit, shape == "plate")
        rotor = shape == "rotor"
        smf.setRowVisible(self.pm_bms_rotor_check, rotor)
        with_rotor = rotor and self.pm_bms_rotor_check.isChecked()
        for wdg in (self.pm_bms_rotor_length_spin, self.pm_bms_wall_combo,
                    self.pm_bms_wall_chi_spin, self.pm_bms_caps_combo, self.pm_bms_caps_chi_spin):
            smf.setRowVisible(wdg, with_rotor)
        mode = self.pm_bms_mode_combo.currentData()
        smf.setRowVisible(self.pm_bms_angle_spin, shape != "sphere" and mode == "static")
        lf = self._bms_local_form
        quad = self.pm_bms_cq_spin.value() != 0.0
        for wdg in (self.pm_bms_etaq_spin, self.pm_bms_eulerq_edit, self.pm_bms_transitions_combo,
                    self.pm_bms_qorder_combo):
            lf.setRowVisible(wdg, quad)
        lf.setRowVisible(self.pm_bms_euler_edit,
                         self.pm_bms_aniso_spin.value() != 0.0 or quad)
        try:
            sampled = bmsf.needs_sampling(self.bms_sample(), self.bms_local(), mode)
        except ValueError:
            sampled = True
        self._bms_spec_form.setRowVisible(self.pm_bms_rate_spin, mode == "mas")
        self._bms_spec_form.setRowVisible(self.pm_bms_order_spin, mode == "mas" and not sampled)
        self._bms_spec_form.setRowVisible(self.pm_bms_samples_spin, sampled)
        self._bms_slice_row.setVisible(self.pm_bms_view_combo.currentData() == "slice")

    def _material_row(self, form, label, default, exclude=(), tip=""):
        """A material combo (with Custom) and its χ (ppm) box, kept in step."""
        combo = QComboBox()
        for key, (chi, text) in bms_sample.MATERIALS.items():
            if key in exclude:
                continue
            combo.addItem(text.split(":")[0], key)
            combo.setItemData(combo.count() - 1, text, Qt.ItemDataRole.ToolTipRole)
        combo.addItem("Custom", "custom")
        combo.setCurrentIndex(combo.findData(default))
        spin = _spin_box(-1e5, 1e5, bms_sample.MATERIALS[default][0] * 1e6, 4, " ppm", step=0.1,
                         tip=tip + " Volume susceptibility, SI, in ppm (10⁻⁶).")
        # two rows, not one wrapped row (see the χ units row)
        form.addRow(label, combo)
        form.addRow("    its χ (SI):", spin)
        combo.setToolTip(tip)

        def chosen(_i):
            key = combo.currentData()
            if key in bms_sample.MATERIALS:
                spin.blockSignals(True)
                spin.setValue(bms_sample.MATERIALS[key][0] * 1e6)
                spin.blockSignals(False)

        def typed(value):
            key = combo.currentData()
            if key in bms_sample.MATERIALS and abs(bms_sample.MATERIALS[key][0] * 1e6 - value) > 1e-9:
                combo.blockSignals(True)
                combo.setCurrentIndex(combo.findData("custom"))
                combo.blockSignals(False)
        combo.activated.connect(chosen)
        spin.valueChanged.connect(typed)
        return combo, spin

    def _rotor_chosen(self) -> None:
        od = self.pm_bms_rotor_combo.currentData()
        inner, length = bms_sample.ROTORS[od]
        self.pm_bms_inner_spin.setValue(inner)
        self.pm_bms_length_spin.setValue(length)
        self.pm_bms_rotor_length_spin.setValue(length + 2.0 * od)
        self._update_sample_note()

    def _sample_shape_chosen(self) -> None:
        """A sensible static axis angle when the user picks a shape."""
        shape = self.pm_bms_sample_combo.currentData()
        self.pm_bms_angle_spin.setValue(bms_sample.MAGIC_ANGLE_DEG if shape == "rotor" else 0.0)
        self._update_sample_note()

    def _update_sample_note(self) -> None:
        shape = self.pm_bms_sample_combo.currentData()
        text = ""
        if shape in ("rotor", "cylinder"):
            d, L = self.pm_bms_inner_spin.value(), self.pm_bms_length_spin.value()
            vol = np.pi * (d / 2) ** 2 * L
            text = f"Packed volume {vol:.3g} µL (mm³), length/diameter {L / d:.3g}."
            if shape == "rotor":
                od = self.pm_bms_rotor_combo.currentData()
                typical = bms_sample.ROTORS[od]
                text += (f" Typical {od:g} mm values: {typical[0]:g} mm inner diameter, "
                         f"{typical[1]:g} mm packed length; walls, spacers and inserts differ "
                         "between rotors, so set yours.")
                if self.pm_bms_rotor_check.isChecked():
                    text += (" Shifts are relative to the bare sample; a reference in the same "
                             "rotor cancels the rotor's uniform part.")
        self.pm_bms_sample_note.setText(text)

    def bms_sample(self):
        """The macroscopic sample (None: a sphere)."""
        shape = self.pm_bms_sample_combo.currentData()
        angle = self.pm_bms_angle_spin.value()
        if shape == "sphere":
            return None
        if shape == "plate":
            sides = self._triplet(self.pm_bms_plate_edit.text(), "Plate sides")
            return bms_sample.Sample("plate", sides=sides, axis_angle=angle)
        d, L = self.pm_bms_inner_spin.value(), self.pm_bms_length_spin.value()
        rotor = None
        if shape == "rotor":
            od = self.pm_bms_rotor_combo.currentData()
            if d >= od:
                raise ValueError(f"The inner diameter ({d:g} mm) must be below the rotor's "
                                 f"outer diameter ({od:g} mm).")
            if self.pm_bms_rotor_check.isChecked():
                rotor = {"outer_diameter": od, "length": self.pm_bms_rotor_length_spin.value(),
                         "chi_wall": self.pm_bms_wall_chi_spin.value() * 1e-6,
                         "chi_caps": self.pm_bms_caps_chi_spin.value() * 1e-6}
        return bms_sample.Sample("cylinder", diameter=d, length=L, axis_angle=angle, rotor=rotor)

    def bms_local(self):
        """The local interactions (None when the group is off)."""
        if not self.pm_bms_local_group.isChecked():
            return None
        return bms_inter.local_interactions(
            contact_ppm=self.pm_bms_contact_spin.value(), aniso_ppm=self.pm_bms_aniso_spin.value(),
            eta=self.pm_bms_eta_spin.value(),
            euler=self._triplet(self.pm_bms_euler_edit.text(), "Euler angles"),
            C_Q_MHz=self.pm_bms_cq_spin.value(), eta_Q=self.pm_bms_etaq_spin.value(),
            euler_Q=self._triplet(self.pm_bms_eulerq_edit.text(), "Quadrupolar Euler angles"))

    @staticmethod
    def _triplet(text, what):
        vals = [_num(x, what) for x in re.split(r"[,;\s]+", str(text).strip()) if x]
        if len(vals) == 1:
            vals *= 3
        if len(vals) != 3:
            raise ValueError(f"{what}: give one value or three.")
        return tuple(vals)

    def bms_packing_arguments(self) -> dict:
        size = self._triplet(self.pm_bms_size_edit.text(), "Mean size")
        spread = self._triplet(self.pm_bms_spread_edit.text(), "Spread")
        if min(size) <= 0:
            raise ValueError("Mean size: every value must be positive.")
        if min(spread) < 0:
            raise ValueError("Spread: no negative values.")
        if self.pm_bms_equation_combo.currentData() == "cuboid":
            # Edge lengths here; the packer and the field take half sides
            # (v1, v2, v3), as every other shape's radii.
            size = tuple(v / 2.0 for v in size)
            spread = tuple(v / 2.0 for v in spread)
        return {"n": int(self.pm_bms_n_spin.value()), "box_size": self.pm_bms_box_spin.value(),
                "mean_size": size, "spread_size": spread,
                "size_dist": self.pm_bms_dist_combo.currentData(),
                "equation": self.pm_bms_equation_combo.currentData(),
                "deformation": self.pm_bms_deform_combo.currentData(),
                "seed": int(self.pm_bms_seed_spin.value())}

    def _packing_key(self) -> tuple:
        return (self.pm_bms_packer_combo.currentData(),
                tuple(sorted(self.bms_packing_arguments().items())))

    def generate_bms_packing(self, quiet=False) -> bool:
        from ..notebook.bms import packing
        try:
            kw = self.bms_packing_arguments()
            packer = self.pm_bms_packer_combo.currentData()
            fn = packing.grow_and_separate_pack if packer == "grow" else packing.random_close_pack
            with _busy():
                particles, stats = fn(verbose=False, **kw)
        except (ValueError, RuntimeError) as exc:
            if not quiet:
                QMessageBox.warning(self, "Packing", str(exc))
            return False
        if not particles:
            if not quiet:
                QMessageBox.warning(self, "Packing", "No particle could be placed: make the box "
                                                     "bigger or the particles smaller.")
            return False
        self.bms_particles = particles
        self.bms_packing_stats = stats
        self._bms_packed_key = self._packing_key()
        self.pm_bms_result_label.setText(self._packing_text())
        if not quiet and self.pm_bms_view_combo.currentData() == "packing":
            self.show_bms_view()
        return True

    def _packing_text(self) -> str:
        st = self.bms_packing_stats or {}
        text = (f"Packing: {st.get('n_placed', 0)} of {st.get('n_requested', 0)} particles in a "
                f"{self.pm_bms_box_spin.value():.4g} {self.bms_length_unit()} region, "
                f"volume fraction {st.get('achieved_volume_fraction', 0):.3f}")
        if self.pm_bms_region_mode_combo.currentData() == "fraction":
            text += f" (asked {self.pm_bms_fraction_spin.value():.2f})"
        if "n_at_full_size" in st and st["n_at_full_size"] < st.get("n_placed", 0):
            text += (f"; {st['n_placed'] - st['n_at_full_size']} shrunk to fit (mean size "
                     f"{st.get('mean_achieved_size_fraction', 1):.2f} of nominal)")
        return text + "."

    def bms_particles_for_run(self):
        if self.pm_bms_source_combo.currentData() == "dump":
            path = self.pm_bms_dump_edit.text().strip()
            if not path:
                raise ValueError("Choose a LAMMPS dump file.")
            if not os.path.isfile(path):
                raise ValueError(f"No such file: {path}")
            return path
        if self.bms_particles is None or self._bms_packed_key != self._packing_key():
            if not self.generate_bms_packing(quiet=True):
                raise ValueError("The packing could not be generated: check its settings.")
        return self.bms_particles

    def bms_field_arguments(self) -> dict:
        chi = self.bms_chi_si()
        return {"magnetic_susceptibility": chi, "box_size": self.pm_bms_box_spin.value(),
                "gridpoints": int(self.pm_bms_grid_combo.currentData()),
                "equation": self.pm_bms_equation_combo.currentData(),
                "deformation": self.pm_bms_deform_combo.currentData(),
                "slope_filter": self.pm_bms_slope_spin.value(),
                "boundary": self.pm_bms_boundary_combo.currentData(),
                "dtype": "float64" if self.pm_bms_double_check.isChecked() else "float32"}

    def bms_spectrum_arguments(self) -> dict:
        nucleus = self.pm_bms_nucleus_combo.currentText().strip()
        if not nucleus:
            raise ValueError("Choose the nucleus, e.g. 7Li.")
        sample = self.bms_sample()
        return {"nuclide": nucleus, "B0_tesla": self.pm_bms_B0_spin.value(),
                "mode": self.pm_bms_mode_combo.currentData(),
                "observe": self.pm_bms_observe_combo.currentData(),
                "spin_rate_hz": self.pm_bms_rate_spin.value(),
                "max_order": int(self.pm_bms_order_spin.value()),
                "linewidth_hz": self.pm_bms_lw_spin.value(),
                "lineshape": self.pm_bms_shape_combo.currentData(),
                "sample": None if sample is None else sample.to_dict(),
                "local": self.bms_local(),
                "transitions": self.pm_bms_transitions_combo.currentData(),
                "quad_order": self.pm_bms_qorder_combo.currentData(),
                "n_samples": int(self.pm_bms_samples_spin.value()) or None}

    def _field_key(self, particles, fkw, comps):
        src = (os.path.abspath(particles), os.path.getmtime(particles)) \
            if isinstance(particles, str) else id(particles)
        return (src, tuple((k, tuple(v) if isinstance(v, (list, tuple)) else v)
                           for k, v in sorted(fkw.items())))

    def _bms_field_for(self, particles, fkw, comps, quiet):
        """The field: the kept one when its inputs are the same and it has
        the components, else a new one (after a time check)."""
        from . import long_run
        from ..notebook import run_time
        key = self._field_key(particles, fkw, comps)
        keys = {"tensor": "delta_tensor", "iso": "delta_iso", "zz": "delta_zz"}
        if self._bms_field_key == key and self.bms_field is not None and \
                all(keys[c] in self.bms_field for c in comps):
            return self.bms_field, False
        if long_run.enabled() and not quiet:
            est = bms_views.estimate_field_time(particles, components=comps, **fkw)
            if est["seconds"] > run_time.SIMULATION_WARN_S or est["memory_GB"] > 4.0:
                if not long_run.ask(self, "BMS field", "field calculation", est["text"],
                                    "To make it faster: a coarser grid (each step down halves "
                                    "the time or better); single precision; a periodic "
                                    "boundary (isolated has 8× the voxels)."):
                    return None, False
        with _busy():
            field = bmsf.bms_field(particles, components=comps, sample_shape="sphere", **fkw)
        self._bms_field_key = key
        return field, True

    def simulate_bms(self, quiet=False) -> bool:
        from . import long_run
        from ..notebook import run_time
        try:
            particles = self.bms_particles_for_run()
            fkw = self.bms_field_arguments()
            skw = self.bms_spectrum_arguments()
            sampled, comps = bmsf.route(skw["mode"], skw["sample"], skw["local"])
            field, new = self._bms_field_for(particles, fkw, comps, quiet)
            if field is None:
                return False
            self.bms_field = field
            if sampled and long_run.enabled() and not quiet:
                n = skw["n_samples"] or bmsf.default_samples(skw["mode"])
                est = bms_views.estimate_sampling_time(
                    field, n, **self._sampling_kwargs(skw))
                if est["seconds"] > run_time.SIMULATION_WARN_S:
                    if not long_run.ask(self, "BMS spectrum", "spectrum", est["text"],
                                        "To make it faster: fewer samples (more Monte Carlo "
                                        "noise); a faster MAS rate (fewer sidebands); the "
                                        "central transition only; second rather than exact "
                                        "quadrupolar theory."):
                        return False
            with _busy():
                spectrum = bmsf.spectrum_from_field(field, **skw)
        except (ValueError, KeyError, OSError, MemoryError) as exc:
            if not quiet:
                msg = str(exc).strip("'\"")
                if isinstance(exc, MemoryError):
                    msg = "Not enough memory for this grid: choose a smaller one."
                QMessageBox.warning(self, "BMS simulation", msg)
            return False
        self.bms_spectrum = spectrum
        self._bms_field_new = new
        self.bms_nmr_spectrum = self._bms_nmr_spectrum(spectrum, fkw, skw)
        self.pm_bms_keep_button.setEnabled(self._keep_spectrum is not None)
        self.pm_bms_result_label.setText(self.bms_summary())
        self._sync_slice_controls()
        self.pm_bms_view_combo.setCurrentIndex(self.pm_bms_view_combo.findData("spectrum"))
        if self._plot_spectrum is not None:
            self._plot_spectrum(self.bms_nmr_spectrum)
        return True

    @staticmethod
    def _sampling_kwargs(skw):
        keep = ("nuclide", "B0_tesla", "observe", "linewidth_hz", "lineshape", "sample", "local",
                "transitions", "quad_order")
        kw = {k: skw[k] for k in keep}
        kw["mode"] = "static" if skw["mode"] == "static" else "mas"
        kw["spin_rate_hz"] = 1e12 if skw["mode"] == "mas_infinite" else skw["spin_rate_hz"]
        return kw

    def _bms_nmr_spectrum(self, spectrum, fkw, skw):
        from ..core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
        axis = NMRAxis(kind="frequency", ppm=np.asarray(spectrum["x_ppm"], dtype=float))
        params = {"kind": "bms", "model": "BMS",
                  **{k: (list(v) if isinstance(v, tuple) else v) for k, v in fkw.items()},
                  **{k: v for k, v in skw.items()}, "route": spectrum.get("route"),
                  "chi_input": list(self._triplet(self.pm_bms_chi_edit.text(), "χ")),
                  "chi_unit": self.pm_bms_chi_unit_combo.currentData(),
                  "chi_conversion": self.chi_conversion(),
                  "n_particles": self.bms_field["n_particles"],
                  "volume_fraction": round(self.bms_field["volume_fraction"], 4)}
        metadata = NMRMetadata(vendor="simulated", nucleus=spectrum["nuclide"],
                               sfo1=float(spectrum["SFO1"]), bf1=float(spectrum["BF1"]),
                               source_path=None, raw={"simulation": params})
        return NMRSpectrum(data=np.asarray(spectrum["real_spectrum"], dtype=float),
                           axes=[axis], metadata=metadata)

    def bms_summary(self) -> str:
        if self.bms_field is None or self.bms_spectrum is None:
            return ""
        f, sp = self.bms_field, self.bms_spectrum
        st, par = sp["stats"], sp.get("parameters", {})
        tm = f["timings"]
        lines = [f"{f['n_particles']} particles, volume fraction {f['volume_fraction']:.3f} "
                 f"(on the grid), {'×'.join(str(n) for n in f['parameters']['gridpoints'])} grid, "
                 f"{f['parameters']['boundary']}."]
        unit = self.pm_bms_chi_unit_combo.currentData()
        if unit != "si_volume":
            try:
                lines.append(chi_units.describe(self._triplet(self.pm_bms_chi_edit.text(), "χ"),
                                                unit, **self.chi_conversion()) + ".")
            except ValueError:
                pass
        sample = par.get("sample")
        if sample:
            lines.append(f"Sample: {bms_sample.sample_from_dict(sample).describe()}, χ_v = "
                         f"{st.get('chi_v', 0):.3g}.")
        if par.get("local"):
            lo = par["local"]
            bits = [f"δ_iso {lo['contact_ppm']:g} ppm"]
            if lo["aniso_ppm"]:
                bits.append(f"δ {lo['aniso_ppm']:g} ppm, η {lo['eta']:g}")
            if lo["C_Q_MHz"]:
                bits.append(f"C_Q {lo['C_Q_MHz']:g} MHz, η_Q {lo['eta_Q']:g} "
                            f"({par.get('quad_order', 'second')} order, "
                            f"{'all transitions' if par.get('transitions') == 'all' else 'CT'})")
            lines.append("Local: " + "; ".join(bits) + ".")
        lines.append(f"Shift: mean {st['mean_ppm']:.4g} ppm, standard deviation "
                     f"{st['std_ppm']:.4g} ppm (Gaussian FWHM equivalent "
                     f"{st['fwhm_gauss_ppm']:.4g} ppm)"
                     + (" (isotropic shifts)" if par.get("mode", "").startswith("mas") else "")
                     + ".")
        timing = (f"Time: field {tm['total']:.2f} s" if self._bms_field_new
                  else "Time: field kept from the last run")
        if sp.get("route") == "sampled":
            timing += (f", spectrum {st['seconds']:.2f} s ({st['n_samples']:,} samples, Monte "
                       f"Carlo noise {100 * st['mc_noise'] / 2:.2g} % of the peak).")
        else:
            timing += ", spectrum on the grid (exact)."
        lines.append(timing)
        warn = []
        if st.get("sidebands_beyond_max_order", 0) > 1e-3:
            warn.append(f"{100 * st['sidebands_beyond_max_order']:.1f} % of the intensity lies in "
                        f"sidebands beyond the orders kept: keep more orders")
        if st.get("beyond_mu_grid", 0) > 0:
            warn.append(f"{100 * st['beyond_mu_grid']:.2f} % of the signal has an anisotropy above "
                        f"128 × the MAS rate: those sidebands are approximate")
        if st.get("outside_window", 0) > 1e-3:
            warn.append(f"{100 * st['outside_window']:.1f} % of the intensity fell outside the window")
        if st.get("under_resolved"):
            warn.append("the line width spans fewer than 3 points of the axis: peak heights are "
                        "set by the grid")
        if st.get("mc_noise", 0) / 2 > 0.02:
            warn.append("the Monte Carlo noise is not small: raise Samples")
        if f["parameters"]["boundary"] == "periodic" and \
                self.pm_bms_source_combo.currentData() == "pack":
            warn.append("the packer keeps particles inside the box walls, so the layer at the box "
                        "faces is a little less dense than the rest")
        lines += [f"• {w_}." for w_ in warn]
        return "\n".join(lines)

    def _sync_slice_controls(self) -> None:
        if self.bms_field is None:
            return
        avail = {"delta_zz": "delta_zz" in self.bms_field, "delta_iso": "delta_iso" in self.bms_field}
        model = self.pm_bms_quantity_combo.model()
        for i in range(self.pm_bms_quantity_combo.count()):
            model.item(i).setEnabled(bool(avail.get(self.pm_bms_quantity_combo.itemData(i))))
        if not avail.get(self.pm_bms_quantity_combo.currentData()):
            for i in range(self.pm_bms_quantity_combo.count()):
                if avail.get(self.pm_bms_quantity_combo.itemData(i)):
                    self.pm_bms_quantity_combo.blockSignals(True)
                    self.pm_bms_quantity_combo.setCurrentIndex(i)
                    self.pm_bms_quantity_combo.blockSignals(False)
                    break
        n = len(self.bms_field[self.pm_bms_plane_combo.currentData()])
        if self.pm_bms_slice_slider.maximum() != n - 1:
            self.pm_bms_slice_slider.blockSignals(True)
            self.pm_bms_slice_slider.setRange(0, n - 1)
            self.pm_bms_slice_slider.setValue(n // 2)
            self.pm_bms_slice_slider.blockSignals(False)

    def _on_slice_changed(self) -> None:
        if self.bms_field is not None and self.pm_bms_view_combo.currentData() == "slice":
            self._sync_slice_controls()
            self.plot_bms_slice()

    def keep_bms_spectrum(self):
        """The last simulated spectrum into Open experiments; its index there."""
        if self.bms_nmr_spectrum is None or self._keep_spectrum is None:
            return None
        return self._keep_spectrum(self.bms_nmr_spectrum)

    def show_bms_view(self, view=None) -> bool:
        """Show the spectrum, the field slice or the packing of the last run
        on the Simulation plot -- nothing is simulated again (the packing
        needs only the particles)."""
        if view is not None:
            self.pm_bms_view_combo.setCurrentIndex(self.pm_bms_view_combo.findData(view))
        view = self.pm_bms_view_combo.currentData()
        self._sync_bms_rows()
        if view == "packing":
            return self.plot_bms_packing()
        if view == "slice":
            return self.plot_bms_slice()
        if self.bms_nmr_spectrum is None:
            return self.simulate_bms()
        if self._show_spectrum is not None:
            self._show_spectrum(self.bms_nmr_spectrum)
            return True
        return False

    def plot_bms_slice(self) -> bool:
        if self.bms_field is None:
            if not self.simulate_bms():
                return False
        self._sync_slice_controls()
        field = self.bms_field
        quantity = self.pm_bms_quantity_combo.currentData()
        axis = self.pm_bms_plane_combo.currentData()
        index = self.pm_bms_slice_slider.value()

        def draw(figure):
            bmsf.plot_bms_slice(field, figure=figure, quantity=quantity, axis=axis, index=index,
                                fontsize=self._fontsize())
        return self._show(draw, "2d")

    def plot_bms_packing(self, quiet=False) -> bool:
        try:
            particles = self.bms_particles_for_run()
            fkw = self.bms_field_arguments()
        except (ValueError, OSError) as exc:
            if not quiet:
                QMessageBox.warning(self, "Packing", str(exc))
            return False
        style = self._style() or {}

        def draw(figure):
            bms_views.plot_packing_3d(particles, fkw["magnetic_susceptibility"],
                                      box_size=fkw["box_size"], equation=fkw["equation"],
                                      deformation=fkw["deformation"],
                                      slope_filter=fkw["slope_filter"], figure=figure,
                                      unit=self.bms_length_unit(),
                                      fontsize=self._fontsize(),
                                      tensor_style=style.get("tensor_style"))
        return self._show(draw, "tensor")

    def bms_python(self) -> str:
        fkw = self.bms_field_arguments()
        skw = self.bms_spectrum_arguments()
        fkw.pop("magnetic_susceptibility")
        unit = self.pm_bms_chi_unit_combo.currentData()
        values = self._triplet(self.pm_bms_chi_edit.text(), "χ")
        if unit == "si_volume":
            chi_lines = [f"chi = {values!r}  # SI volume susceptibility (dimensionless)"]
        else:
            kw = self.chi_conversion()
            chi_lines = ["from opennmr.notebook import susceptibility",
                         f"chi = tuple(susceptibility.to_si_volume({values!r}, {unit!r}"
                         + "".join(f", {k}={v!r}" for k, v in kw.items()) + "))",
                         "print(susceptibility.describe(" + f"{values!r}, {unit!r}"
                         + "".join(f", {k}={v!r}" for k, v in kw.items()) + "))"]
        if self.pm_bms_source_combo.currentData() == "dump":
            particles = f"particles = {self.pm_bms_dump_edit.text().strip()!r}  # LAMMPS dump"
        else:
            kw = self.bms_packing_arguments()
            fn = ("grow_and_separate_pack" if self.pm_bms_packer_combo.currentData() == "grow"
                  else "random_close_pack")
            particles = ("from opennmr.notebook.bms import packing\n"
                         f"particles, stats = packing.{fn}("
                         + ", ".join(f"{k}={v!r}" for k, v in kw.items()) + ", verbose=False)")
        quantity = "delta_zz" if bmsf.route(skw["mode"], skw["sample"], skw["local"])[0] or \
            skw["mode"] == "static" else "delta_iso"
        sizes_note = (f"# Lengths (sizes, box_size) in {self.bms_length_unit()}"
                      + ("; a cuboid's sizes are half its edge lengths"
                         if fkw["equation"] == "cuboid" else "")
                      + ". Only proportions change the shifts.")
        lines = [sizes_note, "from opennmr.notebook.bms import fast as bmsf", *chi_lines,
                 "from opennmr.notebook.bms import views", particles,
                 "spectrum, field = bmsf.fast_bms_spectrum(particles, chi, "
                 + ", ".join(f"{k}={v!r}" for k, v in {**fkw, **skw}.items()) + ")",
                 "print(spectrum['stats'])",
                 f"fig = bmsf.plot_bms_slice(field, quantity={quantity!r})",
                 "fig3d = views.plot_packing_3d(particles, chi, "
                 f"box_size={fkw['box_size']!r}, equation={fkw['equation']!r}, "
                 f"deformation={fkw['deformation']!r}, unit={self.bms_length_unit()!r})"]
        return "\n".join(lines) + "\n"

    # ==== 3. bond pathways ========================================================

    def _build_pathways(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        hint = QLabel(
            "Additive contributions P (ppm) of each paramagnetic neighbour, per shell and "
            "species. Spin flip: a ferromagnetic reference and one calculation per reversed "
            "ion, P = (δ_FM − δ_flip)/2. Single ion: one paramagnetic ion per calculation, the "
            "rest replaced by diamagnetic ions, P = δ. The CRYSTAL scaling (or the ORCA model) "
            "below applies to every file.")
        hint.setWordWrap(True)
        v.addWidget(hint)
        form = QFormLayout()
        self.pm_bp_mode_combo = QComboBox()
        for key, text in bp.MODES.items():
            self.pm_bp_mode_combo.addItem(text, key)
        form.addRow("Method:", self.pm_bp_mode_combo)
        self.pm_bp_reference_edit = QLineEdit()
        self.pm_bp_reference_edit.setPlaceholderText("Ferromagnetic reference calculation")
        b = QPushButton("Browse…")
        b.clicked.connect(lambda: self._browse(self.pm_bp_reference_edit, "Reference"))
        hr = QHBoxLayout()
        hr.setContentsMargins(0, 0, 0, 0)
        hr.addWidget(self.pm_bp_reference_edit)
        hr.addWidget(b)
        self._bp_ref_row = QWidget()
        self._bp_ref_row.setLayout(hr)
        form.addRow("Reference:", self._bp_ref_row)
        v.addLayout(form)
        self._bp_form = form
        self.pm_bp_file_list = QListWidget()
        self.pm_bp_file_list.setMaximumHeight(110)
        self.pm_bp_file_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        v.addWidget(QLabel("Calculations (one flipped or one paramagnetic ion each):"))
        v.addWidget(self.pm_bp_file_list)
        row = QHBoxLayout()
        add = QPushButton("Add Files…")
        add.clicked.connect(lambda: self._add_pathway_files())
        rem = QPushButton("Remove")
        rem.clicked.connect(lambda: self._remove_pathway_files())
        row.addWidget(add)
        row.addWidget(rem)
        row.addStretch(1)
        v.addLayout(row)
        f2 = QFormLayout()
        self.pm_bp_observed_edit = QLineEdit("Na")
        f2.addRow("Observed element:", self.pm_bp_observed_edit)
        self.pm_bp_magnetic_edit = QLineEdit("Mn, Fe")
        self.pm_bp_magnetic_edit.setToolTip("Paramagnetic species, comma separated.")
        f2.addRow("Paramagnetic species:", self.pm_bp_magnetic_edit)
        v.addLayout(f2)
        ion_hint = QLabel("The ion each file flips (or holds), found from the files: check it, "
                          "and type another label (e.g. Mn5, as in the output) to override it.")
        ion_hint.setWordWrap(True)
        v.addWidget(ion_hint)
        self.pm_bp_ion_table = _table(["File", "Ion", "How it was found", "Evidence"], 0, 130)
        self.pm_bp_ion_table.setToolTip(
            "Spin flip: the ion whose own spin density changed sign; otherwise the ion the "
            "changed spin densities sit around. Single ion: the only paramagnetic ion, or the "
            "one carrying spin. ⚠ marks a guess to check. Only the Ion column can be edited.")
        v.addWidget(self.pm_bp_ion_table)
        row = QHBoxLayout()
        self.pm_bp_identify_button = QPushButton("Identify Ions")
        self.pm_bp_identify_button.clicked.connect(lambda: self.identify_pathway_ions())
        row.addWidget(self.pm_bp_identify_button)
        row.addStretch(1)
        v.addLayout(row)
        self.pm_bp_shell_table = _table(["Shell", "r min (Å)", "r max (Å)"], 0, 140)
        for name, lo, hi in (("NN", "0", "3.6"), ("NNN", "3.6", "5.5")):
            r = self.pm_bp_shell_table.rowCount()
            self.pm_bp_shell_table.insertRow(r)
            for c, t in enumerate((name, lo, hi)):
                _set(self.pm_bp_shell_table, r, c, t, True)
        v.addWidget(QLabel("Shells (as in Simulate ▸ pNMR (CIF)):"))
        v.addWidget(self.pm_bp_shell_table)
        nrow = QHBoxLayout()
        self.pm_bp_maxr_spin = _spin_box(1.0, 50.0, 8.0, 2, " Å", step=0.5,
                                         tip="How far out to list the paramagnetic neighbours.")
        nrow.addWidget(QLabel("Neighbours up to:"))
        nrow.addWidget(self.pm_bp_maxr_spin)
        self.pm_bp_distances_button = QPushButton("Neighbour Distances")
        self.pm_bp_distances_button.setToolTip(
            "The paramagnetic neighbours of the observed nuclei against distance (every "
            "periodic image), stacked by species, with the shells shaded: set the shells "
            "where the gaps are. From the reference calculation (spin flip) or the first file.")
        self.pm_bp_distances_button.clicked.connect(lambda: self.plot_neighbour_distances())
        nrow.addWidget(self.pm_bp_distances_button)
        self.pm_bp_suggest_button = QPushButton("Suggest Shells")
        self.pm_bp_suggest_button.setToolTip("Shells with their boundaries halfway across the "
                                             "gaps between clusters of distances: check them "
                                             "against the plot.")
        self.pm_bp_suggest_button.clicked.connect(lambda: self.suggest_pathway_shells())
        nrow.addWidget(self.pm_bp_suggest_button)
        nrow.addStretch(1)
        v.addLayout(nrow)
        row = QHBoxLayout()
        a2 = QPushButton("Add Shell")
        a2.clicked.connect(lambda: self.pm_bp_shell_table.insertRow(self.pm_bp_shell_table.rowCount()))
        r2 = QPushButton("Remove Shell")
        r2.clicked.connect(lambda: self.pm_bp_shell_table.removeRow(
            max(self.pm_bp_shell_table.currentRow(), self.pm_bp_shell_table.rowCount() - 1)
            if self.pm_bp_shell_table.currentRow() < 0 else self.pm_bp_shell_table.currentRow()))
        row.addWidget(a2)
        row.addWidget(r2)
        row.addStretch(1)
        v.addLayout(row)
        sc = QGroupBox("Magnetic scaling (every file)")
        sf = QFormLayout(sc)
        self._bp_scale_form = sf
        self.pm_bp_route_combo = QComboBox()
        for key, text in (("curie_weiss", "CRYSTAL: Weiss constant θ, μ_eff and spin S"),
                          ("susceptibility", "CRYSTAL: susceptibility χ_M and spin S"),
                          ("phi", "CRYSTAL: scaling factor φ"),
                          ("orca", "ORCA: calculated spin Hamiltonian at T")):
            self.pm_bp_route_combo.addItem(text, key)
        sf.addRow("Scaling:", self.pm_bp_route_combo)
        self.pm_bp_phi_spin = _spin_box(-100.0, 100.0, 0.02, 6, step=0.001)
        self.pm_bp_chi_spin = _spin_box(0.0, 1e6, 0.01, 6, step=0.001,
                                        tip="Molar susceptibility, cm³/mol (cgs).")
        self.pm_bp_theta_spin = _spin_box(-1e5, 1e5, -50.0, 3, " K", step=1.0)
        self.pm_bp_mueff_spin = _spin_box(0.0, 100.0, 5.92, 4, " μB", step=0.01)
        self.pm_bp_spin_spin = _spin_box(0.0, 20.0, 2.5, 3, step=0.5)
        self.pm_bp_T_spin = _spin_box(1.0, 2000.0, 298.0, 2, " K", step=1.0)
        self.pm_bp_B0_spin = _spin_box(0.001, 100.0, 11.7, 4, " T", step=0.1)
        for label, wdg in (("φ:", self.pm_bp_phi_spin), ("χ_M (cm³/mol):", self.pm_bp_chi_spin),
                           ("Weiss constant θ:", self.pm_bp_theta_spin),
                           ("μ_eff:", self.pm_bp_mueff_spin), ("Spin S:", self.pm_bp_spin_spin),
                           ("Temperature T:", self.pm_bp_T_spin), ("Field B0:", self.pm_bp_B0_spin)):
            sf.addRow(label, wdg)
        self._bp_rows = {
            "phi": [self.pm_bp_phi_spin, self.pm_bp_B0_spin],
            "susceptibility": [self.pm_bp_chi_spin, self.pm_bp_spin_spin, self.pm_bp_B0_spin],
            "curie_weiss": [self.pm_bp_theta_spin, self.pm_bp_mueff_spin, self.pm_bp_spin_spin,
                            self.pm_bp_T_spin, self.pm_bp_B0_spin],
            "orca": [self.pm_bp_T_spin]}
        v.addWidget(sc)
        row = QHBoxLayout()
        for text, fn in (("Calculate", lambda: self.calculate_pathways()),
                         ("Plot", lambda: self.plot_pathways()),
                         ("Send to pNMR (CIF)", lambda: self.send_pathways()),
                         ("Copy as Python", lambda: self._copy_safe(self.pathways_python))):
            b = QPushButton(text)
            b.clicked.connect(lambda _c=False, f=fn: f())
            row.addWidget(b)
        row.addStretch(1)
        v.addLayout(row)
        self.pm_bp_result_table = _table(["Shell", "Species", "Mean P (ppm)", "SD (ppm)", "n",
                                          "Min", "Max"], 0, 180)
        v.addWidget(self.pm_bp_result_table)
        self.pm_bp_result_label = _label()
        v.addWidget(self.pm_bp_result_label)
        self.pm_bp_mode_combo.currentIndexChanged.connect(lambda _i: self._sync_bp_rows())
        self.pm_bp_route_combo.currentIndexChanged.connect(lambda _i: self._sync_bp_rows())
        self._sync_bp_rows()
        return w

    def _sync_bp_rows(self) -> None:
        self._bp_form.setRowVisible(self._bp_ref_row,
                                    self.pm_bp_mode_combo.currentData() == "spin_flip")
        route = self.pm_bp_route_combo.currentData()
        shown = {id(x) for x in self._bp_rows[route]}
        for ws in self._bp_rows.values():
            for x in ws:
                self._bp_scale_form.setRowVisible(x, id(x) in shown)

    def _add_pathway_files(self, paths=None) -> None:
        if paths is None:
            paths, _ = QFileDialog.getOpenFileNames(self, "Calculations", "", _CALC_FILTER)
        for p in paths or []:
            if p not in self.pathway_files:
                self.pathway_files.append(p)
                self.pm_bp_file_list.addItem(os.path.basename(p))
                self.pm_bp_file_list.item(self.pm_bp_file_list.count() - 1).setToolTip(p)

    def _remove_pathway_files(self) -> None:
        rows = sorted({i.row() for i in self.pm_bp_file_list.selectedIndexes()}, reverse=True)
        for r in rows:
            self.pm_bp_file_list.takeItem(r)
            del self.pathway_files[r]

    def pathway_scaling(self) -> dict:
        route = self.pm_bp_route_combo.currentData()
        if route == "orca":
            return {"model": "curie", "T": self.pm_bp_T_spin.value()}
        args = {"B0": self.pm_bp_B0_spin.value()}
        if route == "phi":
            args["phi"] = self.pm_bp_phi_spin.value()
        elif route == "susceptibility":
            args.update(chi=self.pm_bp_chi_spin.value(), spin=self.pm_bp_spin_spin.value())
        else:
            args.update(theta=self.pm_bp_theta_spin.value(), mu_eff=self.pm_bp_mueff_spin.value(),
                        spin=self.pm_bp_spin_spin.value(), T=self.pm_bp_T_spin.value())
        return args

    def pathway_shells(self) -> list:
        bins = []
        for r in range(self.pm_bp_shell_table.rowCount()):
            name = _cell(self.pm_bp_shell_table, r, 0)
            if not name:
                continue
            bins.append((name, _num(_cell(self.pm_bp_shell_table, r, 1), f"{name}, r min"),
                         _num(_cell(self.pm_bp_shell_table, r, 2), f"{name}, r max")))
        return bins

    def _structure_calc(self):
        """The calculation the neighbour distances come from."""
        mode = self.pm_bp_mode_combo.currentData()
        ref = self.pm_bp_reference_edit.text().strip()
        if mode == "spin_flip" and ref:
            return ref
        if self.pathway_files:
            return self.pathway_files[0]
        raise ValueError("Add the calculations (and, for spin flip, the reference) first.")

    def plot_neighbour_distances(self, quiet=False) -> bool:
        try:
            calc = self._structure_calc()
            observed = self.pm_bp_observed_edit.text().strip()
            magnetic = [x.strip() for x in self.pm_bp_magnetic_edit.text().split(",") if x.strip()]
            max_r = self.pm_bp_maxr_spin.value()
            rows = bp.neighbour_distances(calc, observed, magnetic, max_r=max_r)
            try:
                bins = self.pathway_shells()
            except ValueError:
                bins = []
        except (OSError, ValueError, KeyError) as exc:
            if not quiet:
                QMessageBox.warning(self, "Neighbour distances", str(exc))
            return False
        self.neighbour_rows = rows
        title = f"{os.path.basename(str(calc))}: {'/'.join(magnetic)} around {observed}"

        def draw(figure):
            bp.plot_neighbour_distances(rows, bins, figure=figure, fontsize=self._fontsize(),
                                        max_r=max_r, title=title)
        return self._show(draw)

    def suggest_pathway_shells(self, quiet=False) -> bool:
        if self.neighbour_rows is None and not self.plot_neighbour_distances(quiet):
            return False
        bins = bp.suggest_shells(self.neighbour_rows, max_r=self.pm_bp_maxr_spin.value())
        if not bins:
            if not quiet:
                QMessageBox.warning(self, "Suggest shells", "No gap between the distances to put "
                                                            "a shell boundary in: widen the range.")
            return False
        t = self.pm_bp_shell_table
        t.setRowCount(0)
        for name, lo, hi in bins:
            r = t.rowCount()
            t.insertRow(r)
            for c, text in enumerate((name, f"{lo:g}", f"{hi:g}")):
                _set(t, r, c, text, True)
        return self.plot_neighbour_distances(quiet)

    def _ion_table_matches_files(self) -> bool:
        t = self.pm_bp_ion_table
        return t.rowCount() == len(self.pathway_files) and all(
            (t.item(r, 0) is not None and t.item(r, 0).toolTip() == path)
            for r, path in enumerate(self.pathway_files))

    def identify_pathway_ions(self, quiet=False) -> bool:
        """Find the ion of each file (keeping any label typed over a
        found one) and show how it was found."""
        try:
            mode = self.pm_bp_mode_combo.currentData()
            ref = self.pm_bp_reference_edit.text().strip() or None
            magnetic = [x.strip() for x in self.pm_bp_magnetic_edit.text().split(",") if x.strip()]
            typed = self._typed_ions() if self._ion_table_matches_files() else None
            found = bp.identify_ions(list(self.pathway_files), magnetic=magnetic, mode=mode,
                                     reference=ref if mode == "spin_flip" else None,
                                     flipped=typed)
        except (OSError, ValueError, KeyError) as exc:
            if not quiet:
                QMessageBox.warning(self, "Identify ions", str(exc))
            return False
        t = self.pm_bp_ion_table
        t.setRowCount(0)
        for path, f in zip(self.pathway_files, found):
            r = t.rowCount()
            t.insertRow(r)
            _set(t, r, 0, os.path.basename(path))
            t.item(r, 0).setToolTip(path)
            _set(t, r, 1, f["ion"] or "", True)
            t.item(r, 1).setData(Qt.ItemDataRole.UserRole, f["ion"] or "")
            _set(t, r, 2, ("" if f["certain"] else "⚠ ") + f["how"])
            _set(t, r, 3, f["evidence"])
        self.pathway_ions = found
        return all(f["ion"] for f in found)

    def _typed_ions(self):
        """Ion labels typed over the found ones (None where it is the found one)."""
        out = []
        for r in range(self.pm_bp_ion_table.rowCount()):
            item = self.pm_bp_ion_table.item(r, 1)
            text = item.text().strip() if item is not None else ""
            found = item.data(Qt.ItemDataRole.UserRole) if item is not None else ""
            out.append(text if text and text != found else None)
        return out

    def pathway_flipped(self):
        """The ion labels to use: the table's, when it lists every file."""
        if not self._ion_table_matches_files():
            return None
        ions = [(_cell(self.pm_bp_ion_table, r, 1) or "").strip()
                for r in range(self.pm_bp_ion_table.rowCount())]
        return ions if all(ions) else None

    def pathway_arguments(self) -> dict:
        flipped = self.pathway_flipped()
        mode = self.pm_bp_mode_combo.currentData()
        return {"observed": self.pm_bp_observed_edit.text().strip(),
                "magnetic": [s.strip() for s in self.pm_bp_magnetic_edit.text().split(",")
                             if s.strip()],
                "shell_bins": self.pathway_shells(), "mode": mode,
                "flipped": flipped or None, "paramagnetic": self.pathway_scaling()}

    def calculate_pathways(self, quiet=False) -> bool:
        if not self._ion_table_matches_files() or self.pathway_flipped() is None:
            self.pathway_ions = None
            if not self.identify_pathway_ions(quiet):
                missing = [f["file"] for f in (self.pathway_ions or []) if not f["ion"]]
                if missing and not quiet:
                    QMessageBox.warning(
                        self, "Bond pathways",
                        "Could not tell which ion is " + ("flipped" if
                        self.pm_bp_mode_combo.currentData() == "spin_flip" else "paramagnetic")
                        + f" in: {', '.join(missing)}. Type its label (e.g. Mn5) in the Ion "
                          "column, then Calculate.")
                return False
        try:
            kw = self.pathway_arguments()
            ref = self.pm_bp_reference_edit.text().strip() or None
            if kw["mode"] == "single":
                ref = None
            res = bp.pathway_contributions(list(self.pathway_files), reference=ref, **kw)
        except (OSError, ValueError, KeyError) as exc:
            if not quiet:
                QMessageBox.warning(self, "Bond pathways", str(exc))
            return False
        self.pathway_result = res
        stats = [(k, s) for k, s in res["summary"].items() if s["n"]]
        t = self.pm_bp_result_table
        t.setRowCount(len(stats))

        def f(x):
            return "—" if x is None else f"{x:.4g}"
        for r, ((shell, sp), s) in enumerate(stats):
            for c, text in enumerate((shell, sp, f(s["mean"]), f(s["sd"]), str(s["n"]),
                                      f(s["min"]), f(s["max"]))):
                _set(t, r, c, text)
        self.pm_bp_result_label.setText(f"{len(res['rows'])} ion–nucleus pairs." + "".join(
            f"\n• {n}" for n in res["notes"]))
        self.plot_pathways()
        return True

    def plot_pathways(self) -> bool:
        if self.pathway_result is None:
            return False
        res = self.pathway_result

        def draw(figure):
            bp.plot_pathways(res, figure=figure, fontsize=self._fontsize())
        return self._show(draw)

    def send_pathways(self) -> bool:
        if self.pathway_result is None or self._send_pathways is None:
            return False
        try:
            notes = self._send_pathways(self.pathway_result["shell_bins"],
                                        self.pathway_result["shift_dict"])
        except ValueError as exc:
            QMessageBox.warning(self, "Send to pNMR (CIF)", str(exc))
            return False
        if self.pm_pnmr_section is not None:
            self.pm_pnmr_section.setExpanded(True)
        text = "Sent to pNMR (CIF) above." + "".join(f"\n• {n}" for n in notes or [])
        self.pm_bp_result_label.setText(self.pm_bp_result_label.text() + "\n" + text)
        return True

    def pathways_python(self) -> str:
        kw = self.pathway_arguments()
        ref = self.pm_bp_reference_edit.text().strip() or None
        return bp.pathways_to_python(self.pathway_files,
                                     reference=ref if kw["mode"] == "spin_flip" else None, **kw)

    # ==== state ===================================================================

    def get_state(self) -> dict:
        from . import panel_state
        state = {"widgets": panel_state.capture_widgets(
            self, exclude=("pm_bp_result_table", "pm_bp_file_list", "pm_bp_ion_table"))}
        state["pathway_files"] = list(self.pathway_files)
        state["pathway_ions"] = (self.pathway_flipped() or []) if self.pathway_files else []
        return state

    def set_state(self, state: dict) -> None:
        from . import panel_state
        state = state or {}
        # Restored values are taken as they were saved: a restored unit does
        # not convert sizes restored before it, and a restored region side
        # does not switch how the region is set.
        self._bms_restoring = True
        self._bms_setting_box = True
        try:
            panel_state.restore_widgets(self, state.get("widgets") or {})
        finally:
            self._bms_restoring = False
            self._bms_setting_box = False
        self._bms_length_unit = self.pm_bms_length_unit_combo.currentData()
        self.pathway_files = []
        self.pm_bp_file_list.clear()
        self._add_pathway_files(state.get("pathway_files") or [])
        ions = state.get("pathway_ions") or []
        self.pm_bp_ion_table.setRowCount(0)
        if len(ions) == len(self.pathway_files):
            for path, ion in zip(self.pathway_files, ions):
                r = self.pm_bp_ion_table.rowCount()
                self.pm_bp_ion_table.insertRow(r)
                _set(self.pm_bp_ion_table, r, 0, os.path.basename(path))
                self.pm_bp_ion_table.item(r, 0).setToolTip(path)
                _set(self.pm_bp_ion_table, r, 1, ion, True)
                _set(self.pm_bp_ion_table, r, 2, "restored")
                _set(self.pm_bp_ion_table, r, 3, "")
        self._sync_bp_rows()
        self._sync_bms_rows()
        self._sync_chi_rows()
        self._sync_T_rows()
        self._update_region()
