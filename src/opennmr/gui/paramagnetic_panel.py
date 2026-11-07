"""
Simulate ▸ Paramagnetic NMR tools: each tool a collapsible section,
drawing on the Simulation plot and copying its notebook code.

1. Shift against temperature (2026-11-06) -- calculated from the ORCA
   g/ZFS/hyperfine calculation of Simulate ▸ DFT tensors (its magnetic
   model), and/or measured shifts fitted to a Curie, Curie + Bleaney (1/T²)
   or Curie–Weiss law (:mod:`opennmr.notebook.paramagnetic_nmr`).
2. Bulk magnetic susceptibility, fast (2026-11-07) -- the BMS shift
   spectrum of a particle packing by superposition: one set of FFTs for
   every particle (:mod:`opennmr.notebook.bms.fast`).
3. Bond pathways (2026-11-06) -- per-shell, per-species contributions from
   spin-flip or single-ion DFT calculations, sent to Simulate ▸ pNMR (CIF)
   (:mod:`opennmr.notebook.bond_pathways`).

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
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from ..notebook import bond_pathways as bp
from ..notebook import paramagnetic_nmr as pmr
from ..notebook.bms import fast as bmsf
from .combined_panels import _spin_box
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

    BLURB = ("Tools for paramagnetic shifts: their temperature dependence, the bulk magnetic "
             "susceptibility (BMS) shifts of a particle packing, and bond-pathway contributions "
             "from DFT for the pNMR (CIF) simulation.")

    def __init__(self, draw_plot=None, style=None, dft_source=None, send_pathways=None,
                 plot_spectrum=None, parent=None) -> None:
        super().__init__(parent)
        self._draw_plot = draw_plot
        self._plot_spectrum = plot_spectrum
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
        self.pathway_files: list[str] = []
        self.pathway_result = None
        layout = QVBoxLayout(self)
        blurb = QLabel(self.BLURB)
        blurb.setWordWrap(True)
        layout.addWidget(blurb)
        self.pm_temperature_section = CollapsibleSection("Shift against temperature",
                                                         content=self._build_temperature())
        self.pm_bms_section = CollapsibleSection(
            "Bulk magnetic susceptibility shifts (fast)", content=self._build_bms())
        self.pm_pathway_section = CollapsibleSection("Bond pathways from DFT (for pNMR (CIF))",
                                                     content=self._build_pathways())
        for sec in (self.pm_temperature_section, self.pm_bms_section, self.pm_pathway_section):
            layout.addWidget(sec)
        layout.addStretch(1)

    def _fontsize(self) -> float:
        return float((self._style() or {}).get("labelsize") or 11.0)

    def _show(self, draw) -> bool:
        if self._draw_plot is None:
            return False
        self._draw_plot(draw, "curve")
        return True

    @staticmethod
    def _copy(text: str) -> None:
        QGuiApplication.clipboard().setText(text)

    # ==== 1. temperature ==========================================================

    def _build_temperature(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        calc_box = QGroupBox("Calculated (the ORCA calculation in Simulate ▸ DFT tensors)")
        cf = QFormLayout(calc_box)
        self.pm_T_min_spin = _spin_box(1.0, 3000.0, 200.0, 1, " K", step=10.0)
        self.pm_T_max_spin = _spin_box(1.0, 3000.0, 400.0, 1, " K", step=10.0)
        self.pm_T_points_spin = _spin_box(2, 2000, 41, 0, step=1)
        cf.addRow("From:", self.pm_T_min_spin)
        cf.addRow("To:", self.pm_T_max_spin)
        cf.addRow("Points:", self.pm_T_points_spin)
        self.pm_T_components_check = QCheckBox("Also show contact and pseudocontact parts")
        cf.addRow("", self.pm_T_components_check)
        self.pm_T_calc_button = QPushButton("Calculate δ(T)")
        self.pm_T_calc_button.setToolTip(
            "The isotropic shift of the sites ticked in Simulate ▸ DFT tensors, with its magnetic "
            "model (calculated spin Hamiltonian, Curie–Weiss or μ_eff), over this range.")
        self.pm_T_calc_button.clicked.connect(lambda: self.calculate_temperature())
        cf.addRow("", self.pm_T_calc_button)
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
        return w

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

    def calculate_temperature(self, quiet=False) -> bool:
        src = self._dft_source()
        try:
            if src is None:
                raise ValueError("Read an ORCA output with hyperfine tensors in Simulate ▸ DFT "
                                 "tensors first (step 1), and choose its magnetic model (step 2).")
            if not src["labels"]:
                raise ValueError("Tick at least one site in Simulate ▸ DFT tensors.")
            args = dict(src["args"])
            args.pop("components", None)
            T = np.linspace(self.pm_T_min_spin.value(), self.pm_T_max_spin.value(),
                            int(self.pm_T_points_spin.value()))
            args.pop("T", None)
            self.temperature_curves = pmr.shift_vs_temperature(
                src["calc"], T, labels=src["labels"], orbital_shift=src["orbital"], **args)
        except (ValueError, KeyError) as exc:
            if not quiet:
                QMessageBox.warning(self, "Shift against temperature", str(exc))
            return False
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
        if self.temperature_curves is not None and src is not None:
            args = {k: v for k, v in src["args"].items() if k not in ("components", "T")}
            lines += ["from opennmr.notebook import dft_nmr",
                      f"calc = dft_nmr.read_calculation({src['path']!r})",
                      f"T = np.linspace({self.pm_T_min_spin.value()!r}, "
                      f"{self.pm_T_max_spin.value()!r}, {int(self.pm_T_points_spin.value())})",
                      f"curves = pm.shift_vs_temperature(calc, T, labels={src['labels']!r}, "
                      f"orbital_shift={src['orbital']!r}, "
                      + ", ".join(f"{k}={v!r}" for k, v in args.items()) + ")"]
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

    # ==== 2. bulk magnetic susceptibility (fast) =================================

    BMS_GRIDS = (64, 96, 128, 160, 192, 256)

    def _build_bms(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        hint = QLabel(
            "The spectrum of the bulk magnetic susceptibility (BMS) shifts in a packing of "
            "magnetised particles. Every particle shares one set of FFTs, so the time depends "
            "on the grid, not on the number of particles. Sizes and box share one length unit.")
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
                                       tip="Number of particles to place in the box.")
        pf.addRow("Number:", self.pm_bms_n_spin)
        self.pm_bms_size_edit = QLineEdit("25, 25, 25")
        self.pm_bms_size_edit.setToolTip(
            "Mean size per axis (v1, v2, v3): semi-axes of an ellipsoid; radius, unused, height of "
            "a hexagonal prism; major and minor radii of a torus. One value for all three.")
        pf.addRow("Mean size:", self.pm_bms_size_edit)
        self.pm_bms_spread_edit = QLineEdit("4, 4, 4")
        self.pm_bms_spread_edit.setToolTip(
            "Spread per axis: the standard deviation (Gaussian) or the half width at half maximum "
            "(Lorentzian, pseudo-Voigt, which have no standard deviation).")
        pf.addRow("Spread:", self.pm_bms_spread_edit)
        self.pm_bms_dist_combo = QComboBox()
        for text, key in (("Gaussian", "gaussian"), ("Lorentzian", "lorentzian"),
                          ("Pseudo-Voigt", "pseudo_voigt")):
            self.pm_bms_dist_combo.addItem(text, key)
        pf.addRow("Size distribution:", self.pm_bms_dist_combo)
        self.pm_bms_packer_combo = QComboBox()
        self.pm_bms_packer_combo.addItem("Grow and separate (dense)", "grow")
        self.pm_bms_packer_combo.addItem("Random sequential addition (≤ ~38 %)", "rsa")
        pf.addRow("Packing:", self.pm_bms_packer_combo)
        self.pm_bms_seed_spin = _spin_box(0, 2 ** 31 - 1, 1, 0, step=1,
                                          tip="Random seed: the same seed gives the same packing.")
        pf.addRow("Seed:", self.pm_bms_seed_spin)
        self.pm_bms_box_spin = _spin_box(1.0, 1e7, 300.0, 3, step=10.0,
                                         tip="Side of the cubic box (same unit as the sizes).")
        pf.addRow("Box:", self.pm_bms_box_spin)
        self.pm_bms_equation_combo = QComboBox()
        for text, key in (("Ellipsoid", "ellipsoid"), ("Hexagonal prism", "hexagonal"),
                          ("Torus", "donut")):
            self.pm_bms_equation_combo.addItem(text, key)
        pf.addRow("Shape:", self.pm_bms_equation_combo)
        self.pm_bms_deform_combo = QComboBox()
        for key in "ABC":
            self.pm_bms_deform_combo.addItem(key, key)
        self.pm_bms_deform_combo.setToolTip("Which particle axis the prism or torus axis lies "
                                            "along (A: x, B: y, C: z of the particle).")
        pf.addRow("Orientation (A/B/C):", self.pm_bms_deform_combo)
        self.pm_bms_slope_spin = _spin_box(2.0, 1e5, 2000.0, 0, step=100.0,
                                           tip="Edge steepness of the shape function (large = sharp).")
        pf.addRow("Slope filter:", self.pm_bms_slope_spin)
        self.pm_bms_pack_button = QPushButton("Generate Packing")
        self.pm_bms_pack_button.clicked.connect(lambda: self.generate_bms_packing())
        pf.addRow("", self.pm_bms_pack_button)
        v.addWidget(pk)

        gb = QGroupBox("Susceptibility and grid")
        gf = QFormLayout(gb)
        self._bms_grid_form = gf
        self.pm_bms_chi_edit = QLineEdit("1e-3, 1e-3, 1e-3")
        self.pm_bms_chi_edit.setToolTip(
            "Principal values χ1, χ2, χ3 of the particles' volume susceptibility (SI, "
            "dimensionless), in each particle's own frame. One value: isotropic.")
        gf.addRow("χ (SI volume):", self.pm_bms_chi_edit)
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
        self.pm_bms_sample_combo = QComboBox()
        for key, (text, _D) in bmsf.SAMPLE_SHAPES.items():
            self.pm_bms_sample_combo.addItem(text, key)
        self.pm_bms_sample_combo.setToolTip(
            "The macroscopic sample's demagnetising tensor (the k = 0 term). It only changes "
            "static spectra: under MAS its isotropic part is always 1/3.")
        gf.addRow("Sample shape:", self.pm_bms_sample_combo)
        self.pm_bms_double_check = QCheckBox("Double precision (slower; single is ample)")
        gf.addRow("", self.pm_bms_double_check)
        v.addWidget(gb)

        sb = QGroupBox("Spectrum")
        sf = QFormLayout(sb)
        self._bms_spec_form = sf
        self.pm_bms_nucleus_edit = QLineEdit("7Li")
        sf.addRow("Nucleus:", self.pm_bms_nucleus_edit)
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
        self.pm_bms_run_button.clicked.connect(lambda: self.simulate_bms())
        row.addWidget(self.pm_bms_run_button)
        self.pm_bms_slice_button = QPushButton("Show Field Slice")
        self.pm_bms_slice_button.setToolTip("The shift field through the middle of the box (xz "
                                            "plane, B₀ vertical), with the particle outlines.")
        self.pm_bms_slice_button.clicked.connect(lambda: self.plot_bms_slice())
        row.addWidget(self.pm_bms_slice_button)
        self.pm_bms_python_button = QPushButton("Copy as Python")
        self.pm_bms_python_button.clicked.connect(lambda: self._copy_safe(self.bms_python))
        row.addWidget(self.pm_bms_python_button)
        row.addStretch(1)
        v.addLayout(row)
        self.pm_bms_result_label = _label()
        v.addWidget(self.pm_bms_result_label)
        for combo in (self.pm_bms_source_combo, self.pm_bms_mode_combo,
                      self.pm_bms_boundary_combo, self.pm_bms_equation_combo):
            combo.currentIndexChanged.connect(lambda _i: self._sync_bms_rows())
        self._sync_bms_rows()
        return w

    def _sync_bms_rows(self) -> None:
        pack = self.pm_bms_source_combo.currentData() == "pack"
        pf = self._bms_particle_form
        pf.setRowVisible(self._bms_dump_row, not pack)
        for wdg in (self.pm_bms_n_spin, self.pm_bms_size_edit, self.pm_bms_spread_edit,
                    self.pm_bms_dist_combo, self.pm_bms_packer_combo, self.pm_bms_seed_spin,
                    self.pm_bms_pack_button):
            pf.setRowVisible(wdg, pack)
        pf.setRowVisible(self.pm_bms_deform_combo,
                         self.pm_bms_equation_combo.currentData() != "ellipsoid")
        self._bms_grid_form.setRowVisible(
            self.pm_bms_sample_combo, self.pm_bms_boundary_combo.currentData() == "periodic")
        mode = self.pm_bms_mode_combo.currentData()
        self._bms_spec_form.setRowVisible(self.pm_bms_rate_spin, mode == "mas")
        self._bms_spec_form.setRowVisible(self.pm_bms_order_spin, mode == "mas")

    @staticmethod
    def _triplet(text, what):
        vals = [_num(x, what) for x in re.split(r"[,;\s]+", str(text).strip()) if x]
        if len(vals) == 1:
            vals *= 3
        if len(vals) != 3:
            raise ValueError(f"{what}: give one value or three (x, y, z).")
        return tuple(vals)

    def bms_packing_arguments(self) -> dict:
        size = self._triplet(self.pm_bms_size_edit.text(), "Mean size")
        spread = self._triplet(self.pm_bms_spread_edit.text(), "Spread")
        if min(size) <= 0:
            raise ValueError("Mean size: every value must be positive.")
        if min(spread) < 0:
            raise ValueError("Spread: no negative values.")
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
        return True

    def _packing_text(self) -> str:
        st = self.bms_packing_stats or {}
        text = (f"Packing: {st.get('n_placed', 0)} of {st.get('n_requested', 0)} particles, "
                f"volume fraction {st.get('achieved_volume_fraction', 0):.3f}")
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
        chi = self._triplet(self.pm_bms_chi_edit.text(), "χ")
        return {"magnetic_susceptibility": chi, "box_size": self.pm_bms_box_spin.value(),
                "gridpoints": int(self.pm_bms_grid_combo.currentData()),
                "equation": self.pm_bms_equation_combo.currentData(),
                "deformation": self.pm_bms_deform_combo.currentData(),
                "slope_filter": self.pm_bms_slope_spin.value(),
                "boundary": self.pm_bms_boundary_combo.currentData(),
                "sample_shape": self.pm_bms_sample_combo.currentData(),
                "dtype": "float64" if self.pm_bms_double_check.isChecked() else "float32"}

    def bms_spectrum_arguments(self) -> dict:
        nucleus = self.pm_bms_nucleus_edit.text().strip()
        if not nucleus:
            raise ValueError("Give the nucleus, e.g. 7Li.")
        return {"nuclide": nucleus, "B0_tesla": self.pm_bms_B0_spin.value(),
                "mode": self.pm_bms_mode_combo.currentData(),
                "observe": self.pm_bms_observe_combo.currentData(),
                "spin_rate_hz": self.pm_bms_rate_spin.value(),
                "max_order": int(self.pm_bms_order_spin.value()),
                "linewidth_hz": self.pm_bms_lw_spin.value(),
                "lineshape": self.pm_bms_shape_combo.currentData()}

    def simulate_bms(self, quiet=False) -> bool:
        try:
            particles = self.bms_particles_for_run()
            fkw = self.bms_field_arguments()
            skw = self.bms_spectrum_arguments()
            with _busy():
                spectrum, field = bmsf.fast_bms_spectrum(particles, **fkw, **skw)
        except (ValueError, KeyError, OSError, MemoryError) as exc:
            if not quiet:
                msg = str(exc).strip("'\"")
                if isinstance(exc, MemoryError):
                    msg = "Not enough memory for this grid: choose a smaller one."
                QMessageBox.warning(self, "BMS simulation", msg)
            return False
        self.bms_spectrum = spectrum
        self.bms_field = field
        self.pm_bms_result_label.setText(self.bms_summary())
        if self._plot_spectrum is not None:
            self._plot_spectrum(self._bms_nmr_spectrum(spectrum, fkw, skw))
        return True

    def _bms_nmr_spectrum(self, spectrum, fkw, skw):
        from ..core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
        axis = NMRAxis(kind="frequency", ppm=np.asarray(spectrum["x_ppm"], dtype=float))
        params = {"kind": "bms", "model": "BMS (fast)",
                  **{k: (list(v) if isinstance(v, tuple) else v) for k, v in fkw.items()},
                  **skw, "n_particles": self.bms_field["n_particles"],
                  "volume_fraction": round(self.bms_field["volume_fraction"], 4)}
        metadata = NMRMetadata(vendor="simulated", nucleus=spectrum["nuclide"],
                               sfo1=float(spectrum["SFO1"]), bf1=float(spectrum["BF1"]),
                               source_path=None, raw={"simulation": params})
        return NMRSpectrum(data=np.asarray(spectrum["real_spectrum"], dtype=float),
                           axes=[axis], metadata=metadata)

    def bms_summary(self) -> str:
        if self.bms_field is None or self.bms_spectrum is None:
            return ""
        f, st = self.bms_field, self.bms_spectrum["stats"]
        tm = f["timings"]
        lines = [f"{f['n_particles']} particles, volume fraction {f['volume_fraction']:.3f} "
                 f"(on the grid), {'×'.join(str(n) for n in f['parameters']['gridpoints'])} grid, "
                 f"{f['parameters']['boundary']}.",
                 f"Shift: mean {st['mean_ppm']:.4g} ppm, standard deviation {st['std_ppm']:.4g} ppm "
                 f"(Gaussian FWHM equivalent {st['fwhm_gauss_ppm']:.4g} ppm).",
                 f"Time: shapes {tm['shapes']:.2f} s, FFTs "
                 f"{tm['forward_fft'] + tm['inverse_fft']:.2f} s."]
        warn = []
        if st.get("sidebands_beyond_max_order", 0) > 1e-3:
            warn.append(f"{100 * st['sidebands_beyond_max_order']:.1f} % of the intensity lies in "
                        f"sidebands beyond the orders kept: keep more orders")
        if st.get("beyond_mu_grid", 0) > 0:
            warn.append(f"{100 * st['beyond_mu_grid']:.2f} % of the signal has an anisotropy above "
                        f"128 × the MAS rate: those sidebands are approximate")
        if st.get("outside_window", 0) > 1e-3:
            warn.append(f"{100 * st['outside_window']:.1f} % of the intensity fell outside the window")
        if f["parameters"]["boundary"] == "periodic" and \
                self.pm_bms_source_combo.currentData() == "pack":
            warn.append("the packer keeps particles inside the box walls, so the layer at the box "
                        "faces is a little less dense than the rest")
        lines += [f"• {w_}." for w_ in warn]
        return "\n".join(lines)

    def plot_bms_slice(self) -> bool:
        if self.bms_field is None:
            if not self.simulate_bms():
                return False
        field = self.bms_field
        quantity = "delta_zz" if "delta_zz" in field else "delta_iso"

        def draw(figure):
            bmsf.plot_bms_slice(field, figure=figure, quantity=quantity,
                                fontsize=self._fontsize())
        return self._show(draw)

    def bms_python(self) -> str:
        fkw = self.bms_field_arguments()
        skw = self.bms_spectrum_arguments()
        chi = fkw.pop("magnetic_susceptibility")
        if self.pm_bms_source_combo.currentData() == "dump":
            particles = f"particles = {self.pm_bms_dump_edit.text().strip()!r}  # LAMMPS dump"
        else:
            kw = self.bms_packing_arguments()
            fn = ("grow_and_separate_pack" if self.pm_bms_packer_combo.currentData() == "grow"
                  else "random_close_pack")
            particles = ("from opennmr.notebook.bms import packing\n"
                         f"particles, stats = packing.{fn}("
                         + ", ".join(f"{k}={v!r}" for k, v in kw.items()) + ", verbose=False)")
        lines = ["from opennmr.notebook.bms import fast as bmsf", particles,
                 f"spectrum, field = bmsf.fast_bms_spectrum(particles, {chi!r}, "
                 + ", ".join(f"{k}={v!r}" for k, v in {**fkw, **skw}.items()) + ")",
                 "print(spectrum['stats'])",
                 "fig = bmsf.plot_bms_slice(field, quantity="
                 f"{'delta_zz' if skw['mode'] == 'static' else 'delta_iso'!r})"]
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
        self.pm_bp_flipped_edit = QLineEdit()
        self.pm_bp_flipped_edit.setPlaceholderText("optional: Mn3, Mn5, … (one per file, in order)")
        self.pm_bp_flipped_edit.setToolTip(
            "Which ion each calculation flips (or holds). Empty: found from the spin densities "
            "at the nuclei (CRYSTAL), or as the only ion of a paramagnetic species.")
        f2.addRow("Ion of each file:", self.pm_bp_flipped_edit)
        v.addLayout(f2)
        self.pm_bp_shell_table = _table(["Shell", "r min (Å)", "r max (Å)"], 0, 140)
        for name, lo, hi in (("NN", "0", "3.6"), ("NNN", "3.6", "5.5")):
            r = self.pm_bp_shell_table.rowCount()
            self.pm_bp_shell_table.insertRow(r)
            for c, t in enumerate((name, lo, hi)):
                _set(self.pm_bp_shell_table, r, c, t, True)
        v.addWidget(QLabel("Shells (as in Simulate ▸ pNMR (CIF)):"))
        v.addWidget(self.pm_bp_shell_table)
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

    def pathway_arguments(self) -> dict:
        flipped = [s.strip() for s in self.pm_bp_flipped_edit.text().split(",") if s.strip()]
        mode = self.pm_bp_mode_combo.currentData()
        return {"observed": self.pm_bp_observed_edit.text().strip(),
                "magnetic": [s.strip() for s in self.pm_bp_magnetic_edit.text().split(",")
                             if s.strip()],
                "shell_bins": self.pathway_shells(), "mode": mode,
                "flipped": flipped or None, "paramagnetic": self.pathway_scaling()}

    def calculate_pathways(self, quiet=False) -> bool:
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
        text = "Sent to Simulate ▸ pNMR (CIF)." + "".join(f"\n• {n}" for n in notes or [])
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
            self, exclude=("pm_bp_result_table", "pm_bp_file_list"))}
        state["pathway_files"] = list(self.pathway_files)
        return state

    def set_state(self, state: dict) -> None:
        from . import panel_state
        state = state or {}
        panel_state.restore_widgets(self, state.get("widgets") or {})
        self.pathway_files = []
        self.pm_bp_file_list.clear()
        self._add_pathway_files(state.get("pathway_files") or [])
        self._sync_bp_rows()
        self._sync_bms_rows()
