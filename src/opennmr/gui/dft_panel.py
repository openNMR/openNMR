"""
Simulate ▸ DFT tensors (CASTEP / VASP) (explicit user request,
2026-10-06): import the shielding and EFG tensors of a sample and a
reference calculation, reference the shifts, work out the CSA and
quadrupolar parameters (and their relative orientation) of the chosen
sites, simulate, and look at each site's tensors in the cell or Cartesian
frame.

It *is* a Simulate ▸ Combined Hamiltonian panel with an import box on top:
the imported sites land in its site list, where they can be edited, and its
own field, regime, MAS, transitions, level of theory and axis -- and its
Simulate, plot, Keep and Send to Fit -- do the rest. The maths is
:mod:`opennmr.notebook.dft_nmr`, as a notebook calls it.
"""

from __future__ import annotations

import copy
import os

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout,
    QHeaderView, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMessageBox, QPushButton,
    QTableWidget, QTableWidgetItem, QVBoxLayout,
)

from ..notebook import dft_nmr
from .combined_panels import CombinedHamiltonianPanel, _spin_box


class DftSimulationPanel(CombinedHamiltonianPanel):
    """``draw_tensors(draw)``: show ``draw(figure)`` on the Simulation plot
    as a tensor picture; ``style()``: that plot's style settings."""

    BLURB = ("Simulate from DFT: give the sample's calculation and a reference compound's "
             "(CASTEP .magres or VASP OUTCAR), choose the sites, and enter the reference's "
             "experimental shift. Shifts, CSA (span/skew and Δ/η), C_Q and η_Q, and the CSA's "
             "orientation relative to the EFG come from the tensors; the sites then appear "
             "below, editable, for Simulate.")

    def __init__(self, plot_spectrum, add_to_open, draw_sites=None, draw_tensors=None,
                 style=None, parent=None, send_to_fit=None, draw_orders=None) -> None:
        super().__init__(plot_spectrum=plot_spectrum, add_to_open=add_to_open,
                         draw_sites=draw_sites, parent=parent, send_to_fit=send_to_fit,
                         draw_orders=draw_orders)
        self._draw_tensors = draw_tensors
        self._style = style or (lambda: {})
        self.calcs = {"sample": None, "reference": None}
        self.model = None
        self.tensor_notes = []
        layout = self.layout()
        layout.insertWidget(1, self._build_import_box())
        layout.insertWidget(layout.count() - 1, self._build_tensor_box())
        # The base panel starts with one empty site; a DFT panel starts
        # with none until something is imported.
        self.site_list.clear()
        self.site_dicts = []
        self._current = -1
        self._on_site_selected(-1)
        self._sync_dft()

    # ---- building -------------------------------------------------------------------

    def _file_row(self, attr, title):
        row = QHBoxLayout()
        edit = QLineEdit()
        edit.setPlaceholderText("CASTEP .magres or VASP OUTCAR")
        setattr(self, attr, edit)
        row.addWidget(edit, 1)
        button = QPushButton("Browse…")
        button.clicked.connect(lambda: self._browse(edit, title))
        row.addWidget(button)
        return row

    def _build_import_box(self) -> QGroupBox:
        box = QGroupBox("DFT calculation")
        v = QVBoxLayout(box)
        form = QFormLayout()
        form.addRow("Sample:", self._file_row("dft_sample_edit", "Sample calculation"))
        form.addRow("Reference:", self._file_row("dft_reference_edit", "Reference calculation"))
        self.dft_g0_check = QCheckBox("VASP: include the G=0 (susceptibility) term")
        self.dft_g0_check.setChecked(True)
        self.dft_g0_check.setToolTip(
            "VASP prints the shielding without the macroscopic (G=0) susceptibility term and "
            "adds it, for a spherical sample, in its summary. Included, the shieldings match "
            "VASP's 'INCLUDING G=0' column. It does not cancel between sample and reference "
            "unless their susceptibilities are alike, so treat both the same way.")
        form.addRow("", self.dft_g0_check)
        self.dft_read_button = QPushButton("Read Files")
        self.dft_read_button.clicked.connect(lambda: self.read_files())
        form.addRow("", self.dft_read_button)
        self.dft_files_label = QLabel("")
        self.dft_files_label.setWordWrap(True)
        form.addRow(self.dft_files_label)

        self.dft_element_combo = QComboBox()
        self.dft_element_combo.setToolTip("The element to simulate (one spectrum is one nucleus).")
        self.dft_element_combo.currentIndexChanged.connect(lambda _i: self._on_element())
        form.addRow("Element:", self.dft_element_combo)
        self.dft_isotope_combo = QComboBox()
        self.dft_isotope_combo.setToolTip(
            "The observed isotope: sets the spin, the quadrupole moment Q (signed, openNMR's "
            "table) and so C_Q, and the Larmor frequency at your field.")
        form.addRow("Observed isotope:", self.dft_isotope_combo)
        site_row = QHBoxLayout()
        self.dft_site_list = QListWidget()
        self.dft_site_list.setMaximumHeight(110)
        self.dft_site_list.setToolTip("Tick the sites to simulate (all of them: 'all Na').")
        site_row.addWidget(self.dft_site_list, 1)
        buttons = QVBoxLayout()
        for text, state in (("All", True), ("None", False)):
            b = QPushButton(text)
            b.clicked.connect(lambda _c=False, s=state: self._check_all(s))
            buttons.addWidget(b)
        buttons.addStretch(1)
        site_row.addLayout(buttons)
        form.addRow("Sites:", site_row)

        self.dft_sigma_ref_label = QLabel("—")
        self.dft_sigma_ref_label.setWordWrap(True)
        form.addRow("σ_ref (reference):", self.dft_sigma_ref_label)
        override = QHBoxLayout()
        self.dft_sigma_override_check = QCheckBox("Use instead:")
        self.dft_sigma_override_check.setToolTip(
            "Type the reference shielding yourself -- for an element the reference "
            "calculation does not contain, or a value from elsewhere.")
        self.dft_sigma_override_check.toggled.connect(lambda _c: self._sync_dft())
        override.addWidget(self.dft_sigma_override_check)
        self.dft_sigma_ref_spin = _spin_box(-1e5, 1e5, 0.0, 4, " ppm")
        override.addWidget(self.dft_sigma_ref_spin, 1)
        form.addRow("", override)
        self.dft_delta_ref_spin = _spin_box(-1e5, 1e5, 0.0, 3, " ppm",
                                            tip="The reference compound's experimental shift on "
                                                "your scale, e.g. solid NaCl for ²³Na.")
        form.addRow("δ_ref (experimental):", self.dft_delta_ref_spin)
        self.dft_slope_spin = _spin_box(0.01, 10.0, 1.0, 4, step=0.01,
                                        tip="m in δ = δ_ref + m(σ_ref − σ). 1 unless you have a "
                                            "calibration against several compounds.")
        form.addRow("Slope m:", self.dft_slope_spin)
        self.dft_include_csa_check = QCheckBox("CSA")
        self.dft_include_csa_check.setChecked(True)
        self.dft_include_quad_check = QCheckBox("Quadrupolar (I > ½)")
        self.dft_include_quad_check.setChecked(True)
        self.dft_merge_check = QCheckBox("Merge equivalent sites")
        self.dft_merge_check.setChecked(True)
        self.dft_merge_check.setToolTip(
            "Sites whose every parameter agrees (0.01 ppm, 0.001 in η, 0.1 kHz, 0.1°) become one "
            "site with their count as its amount: the same spectrum, simulated once.")
        opts = QHBoxLayout()
        for w in (self.dft_include_csa_check, self.dft_include_quad_check, self.dft_merge_check):
            opts.addWidget(w)
        form.addRow("Include:", opts)
        self.dft_fwhm_spin = _spin_box(0.0, 1e4, 1.0, 3, " ppm",
                                       tip="Broadening given to every imported site.")
        form.addRow("Broadening FWHM:", self.dft_fwhm_spin)
        v.addLayout(form)
        self.dft_calculate_button = QPushButton("Calculate Sites and Simulate")
        self.dft_calculate_button.setToolTip(
            "Work out every chosen site's parameters, put them in the site list below (replacing "
            "what is there) and simulate with the settings below.")
        self.dft_calculate_button.clicked.connect(lambda: self.calculate())
        v.addWidget(self.dft_calculate_button)
        self.dft_result_table = QTableWidget(0, len(dft_nmr.TABLE_COLUMNS))
        self.dft_result_table.setHorizontalHeaderLabels(
            [name for _k, name, _f in dft_nmr.TABLE_COLUMNS])
        self.dft_result_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents)
        self.dft_result_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.dft_result_table.verticalHeader().setVisible(False)
        self.dft_result_table.setMaximumHeight(200)
        self.dft_result_table.setToolTip(
            "Every chosen atom. Ω, κ: span and skew (Herzfeld–Berger: δ11 ≥ δ22 ≥ δ33). "
            "δaniso, ηCSA: Haeberlen (δaniso = δzz − δiso; Δδ = 1.5 δaniso). C_Q = eQV_zz/h "
            "(signed), ηQ, P_Q = |C_Q|√(1 + ηQ²/3). α, β, γ: the CSA axes in the EFG frame.")
        self.dft_result_table.setVisible(False)
        v.addWidget(self.dft_result_table)
        row = QHBoxLayout()
        self.dft_csv_button = QPushButton("Copy Table (CSV)")
        self.dft_csv_button.clicked.connect(self._copy_csv)
        self.dft_python_button = QPushButton("Copy Import as Python")
        self.dft_python_button.clicked.connect(self._copy_dft_python)
        for b in (self.dft_csv_button, self.dft_python_button):
            b.setEnabled(False)
            row.addWidget(b)
        v.addLayout(row)
        self.dft_notes_label = QLabel("")
        self.dft_notes_label.setWordWrap(True)
        self.dft_notes_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        v.addWidget(self.dft_notes_label)
        return box

    def _build_tensor_box(self) -> QGroupBox:
        box = QGroupBox("Tensors of a site (3D)")
        form = QFormLayout(box)
        self.dft_tensor_site_combo = QComboBox()
        self.dft_tensor_site_combo.setToolTip("Any calculated atom (not only merged sites).")
        form.addRow("Site:", self.dft_tensor_site_combo)
        self.dft_frame_combo = QComboBox()
        self.dft_frame_combo.addItem("Cell axes (a, b, c)", "cell")
        self.dft_frame_combo.addItem("Cartesian (x, y, z of the file)", "cartesian")
        self.dft_frame_combo.setToolTip(
            "Cell: the standard setting (a along x, b in the xy plane), with a, b, c drawn. "
            "Cartesian: the calculation's own x, y, z -- the only choice for a calculation "
            "without a cell.")
        form.addRow("Frame:", self.dft_frame_combo)
        self.dft_tensor_style_combo = QComboBox()
        self.dft_tensor_style_combo.addItem("Ellipsoid and axes", "ellipsoid")
        self.dft_tensor_style_combo.addItem("Frequency surface", "surface")
        self.dft_tensor_style_combo.currentIndexChanged.connect(lambda _i: self._sync_dft())
        form.addRow("Show:", self.dft_tensor_style_combo)
        self.dft_relative_combo = QComboBox()
        self.dft_relative_combo.addItem("Anisotropic part (δ − δiso)", "iso")
        self.dft_relative_combo.addItem("Whole frequency (δ)", "zero")
        form.addRow("Frequency:", self.dft_relative_combo)
        self._tensor_form = form
        self.dft_show_tensor_button = QPushButton("Show Tensors")
        self.dft_show_tensor_button.setToolTip(
            "Draw on the Simulation plot; drag to turn it. Right-click ▸ Tensor Plot Style… for "
            "its colours, axes and view; Plot Style for its text.")
        self.dft_show_tensor_button.clicked.connect(lambda: self.show_tensors())
        form.addRow(self.dft_show_tensor_button)
        self.dft_tensor_notes_label = QLabel("")
        self.dft_tensor_notes_label.setWordWrap(True)
        form.addRow(self.dft_tensor_notes_label)
        return box

    # ---- files ------------------------------------------------------------------------

    def _browse(self, edit, title):
        path, _ = QFileDialog.getOpenFileName(self, title, os.path.dirname(edit.text() or ""),
                                              "DFT output (*.magres OUTCAR* *);;All files (*)")
        if path:
            edit.setText(path)
            self.read_files(quiet=False)

    def read_files(self, quiet: bool = False) -> bool:
        """Read the sample (required) and reference (optional) files."""
        g0 = self.dft_g0_check.isChecked()
        try:
            sample = self.dft_sample_edit.text().strip()
            if not sample:
                raise ValueError("Choose the sample's calculation file.")
            self.calcs["sample"] = dft_nmr.read_calculation(sample, include_g0=g0)
            ref = self.dft_reference_edit.text().strip()
            self.calcs["reference"] = (dft_nmr.read_calculation(ref, include_g0=g0)
                                       if ref else None)
        except (OSError, ValueError, KeyError) as exc:
            self.dft_files_label.setText(f"Could not read: {exc}")
            if not quiet:
                QMessageBox.warning(self, "DFT import", str(exc))
            return False
        lines = []
        for key in ("sample", "reference"):
            c = self.calcs[key]
            if c is None:
                continue
            n_sigma = sum(a["sigma"] is not None for a in c["atoms"])
            n_efg = sum(a["efg"] is not None for a in c["atoms"])
            lines.append(f"{key.title()}: {c['code']}, {len(c['atoms'])} atoms ("
                         + ", ".join(f"{s} {sum(a['species'] == s for a in c['atoms'])}"
                                     for s in dft_nmr.species_of(c))
                         + f"); shielding for {n_sigma}, EFG for {n_efg}"
                         + ("" if c.get("lattice") is not None else "; no cell") + ".")
        self.dft_files_label.setText("\n".join(lines))
        keep = self.dft_element_combo.currentText()
        self.dft_element_combo.blockSignals(True)
        self.dft_element_combo.clear()
        self.dft_element_combo.addItems(dft_nmr.species_of(self.calcs["sample"]))
        if keep:
            index = self.dft_element_combo.findText(keep)
            if index >= 0:
                self.dft_element_combo.setCurrentIndex(index)
        self.dft_element_combo.blockSignals(False)
        self._on_element()
        self._sync_dft()
        return True

    def _on_element(self, checked_labels=None) -> None:
        element = self.dft_element_combo.currentText()
        self.dft_isotope_combo.clear()
        self.dft_site_list.clear()
        if not element or self.calcs["sample"] is None:
            self._sync_dft()
            return
        isotopes = dft_nmr.nmr_isotopes(element)
        self.dft_isotope_combo.addItems(isotopes)
        for atom in self.calcs["sample"]["atoms"]:
            if atom["species"] != element:
                continue
            item = QListWidgetItem(atom["label"])
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            on = checked_labels is None or atom["label"] in checked_labels
            item.setCheckState(Qt.CheckState.Checked if on else Qt.CheckState.Unchecked)
            self.dft_site_list.addItem(item)
        self._update_reference_label()
        self._sync_dft()

    def _check_all(self, state: bool) -> None:
        for i in range(self.dft_site_list.count()):
            self.dft_site_list.item(i).setCheckState(
                Qt.CheckState.Checked if state else Qt.CheckState.Unchecked)

    def checked_labels(self) -> list[str]:
        return [self.dft_site_list.item(i).text() for i in range(self.dft_site_list.count())
                if self.dft_site_list.item(i).checkState() == Qt.CheckState.Checked]

    def _update_reference_label(self) -> None:
        element = self.dft_element_combo.currentText()
        ref = self.calcs["reference"]
        if ref is None:
            self.dft_sigma_ref_label.setText("no reference file: type σ_ref below")
            return
        try:
            info = dft_nmr.reference_shielding(ref, element)
        except ValueError:
            self.dft_sigma_ref_label.setText(f"the reference has no {element}: type σ_ref below")
            return
        spread = f", spread {info['spread']:.3g} ppm" if info["spread"] > 0.005 else ""
        self.dft_sigma_ref_label.setText(
            f"{info['sigma_ref']:.4f} ppm (mean of {len(info['labels'])} {element} site(s) of "
            f"{os.path.basename(ref['path'])}{spread})")
        if not self.dft_sigma_override_check.isChecked():
            self.dft_sigma_ref_spin.setValue(info["sigma_ref"])

    def _sync_dft(self) -> None:
        self.dft_sigma_ref_spin.setEnabled(self.dft_sigma_override_check.isChecked())
        have = self.calcs["sample"] is not None
        self.dft_calculate_button.setEnabled(have)
        self._tensor_form.setRowVisible(self.dft_relative_combo,
                                        self.dft_tensor_style_combo.currentData() == "surface")
        sample = self.calcs["sample"]
        periodic = sample is not None and sample.get("lattice") is not None
        cell_item = self.dft_frame_combo.model().item(0)
        cell_item.setEnabled(periodic or sample is None)
        if sample is not None and not periodic:
            self.dft_frame_combo.setCurrentIndex(1)
        self.dft_show_tensor_button.setEnabled(self.model is not None)

    # ---- calculating ------------------------------------------------------------------

    def import_arguments(self) -> dict:
        labels = self.checked_labels()
        if not labels:
            raise ValueError("Tick at least one site.")
        override = self.dft_sigma_override_check.isChecked()
        return {"element": self.dft_element_combo.currentText(), "labels": labels,
                "nucleus": self.dft_isotope_combo.currentText() or None,
                "delta_ref": self.dft_delta_ref_spin.value(),
                "sigma_ref": self.dft_sigma_ref_spin.value() if override else None,
                "slope": self.dft_slope_spin.value(), "fwhm": self.dft_fwhm_spin.value(),
                "include_csa": self.dft_include_csa_check.isChecked(),
                "include_quadrupolar": self.dft_include_quad_check.isChecked(),
                "merge": self.dft_merge_check.isChecked()}

    def calculate(self, simulate: bool = True) -> bool:
        if self.calcs["sample"] is None and not self.read_files(quiet=True):
            QMessageBox.warning(self, "DFT import", self.dft_files_label.text())
            return False
        try:
            args = self.import_arguments()
            if args["sigma_ref"] is None and self.calcs["reference"] is None:
                raise ValueError("Give a reference calculation, or tick 'Use instead' and type "
                                 "σ_ref.")
            model = dft_nmr.dft_sites(self.calcs["sample"], self.calcs["reference"], **args)
        except (ValueError, KeyError) as exc:
            QMessageBox.warning(self, "DFT import", str(exc))
            return False
        self.model = model
        self._load_sites(model)
        self._fill_table(model)
        self.dft_notes_label.setText("\n".join(f"• {n}" for n in model["notes"]))
        self.dft_tensor_site_combo.clear()
        self.dft_tensor_site_combo.addItems([rec["label"] for rec in model["atoms"]])
        self.dft_csv_button.setEnabled(True)
        self.dft_python_button.setEnabled(True)
        self._sync_dft()
        if simulate and self.larmor_spin.value() > 0:
            return self.simulate()
        return True

    def _load_sites(self, model) -> None:
        self.site_list.blockSignals(True)
        self.site_list.clear()
        self.site_list.blockSignals(False)
        self.site_dicts = []
        self._current = -1
        self.nucleus_combo.setCurrentText(model["nucleus"])
        for d in copy.deepcopy(model["site_dicts"]):
            CombinedHamiltonianPanel.add_site(self, d)
        if self.site_dicts:
            self.site_list.setCurrentRow(0)
        self._sync_spin()
        self._sync_controls()

    def _fill_table(self, model) -> None:
        rows = dft_nmr.table_rows(model)
        t = self.dft_result_table
        t.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, text in enumerate(row):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                t.setItem(r, c, item)
        t.setVisible(True)

    def _copy_csv(self) -> None:
        if self.model is not None:
            QGuiApplication.clipboard().setText(dft_nmr.table_csv(self.model))

    def dft_python(self) -> str:
        args = self.import_arguments()
        return dft_nmr.dft_to_python(self.dft_sample_edit.text().strip(),
                                     self.dft_reference_edit.text().strip() or None, **args)

    def _copy_dft_python(self) -> None:
        try:
            QGuiApplication.clipboard().setText(self.dft_python())
        except ValueError as exc:
            QMessageBox.warning(self, "Copy as Python", str(exc))

    # ---- tensors ----------------------------------------------------------------------

    def tensor_record(self):
        if self.model is None:
            raise ValueError("Calculate the sites first.")
        label = self.dft_tensor_site_combo.currentText()
        for rec in self.model["atoms"]:
            if rec["label"] == label:
                return rec
        return self.model["atoms"][0]

    def show_tensors(self) -> bool:
        if self._draw_tensors is None:
            return False
        try:
            rec = self.tensor_record()
            frame = self.dft_frame_combo.currentData()
            style = self.dft_tensor_style_combo.currentData()
            sample = self.calcs["sample"]
            dft_nmr.display_frame(sample, rec, frame)      # raises if it cannot be drawn
            larmor = self.larmor_spin.value() * 1e6 or None
            order = self.order_combo.currentData()
            holder = {}

            def draw(figure):
                look = self._style() or {}
                extra = {"tensor_style": dict(look.get("tensor_style") or {}),
                         "fontsize": float(look.get("labelsize") or 11.0)}
                if style == "surface":
                    extra.update(relative_to=self.dft_relative_combo.currentData(),
                                 order=order or "second")
                    if look.get("map_colormap") and not extra["tensor_style"].get("colormap"):
                        extra["cmap"] = look["map_colormap"]
                holder["notes"] = dft_nmr.plot_dft_tensors(
                    rec, sample, frame=frame, style=style, larmor_Hz=larmor,
                    nucleus=self.model["nucleus"], figure=figure, **extra)[1]
            self._draw_tensors(draw)
        except (ValueError, KeyError) as exc:
            self.dft_tensor_notes_label.setText(f"Cannot draw: {exc}")
            return False
        self.tensor_notes = holder.get("notes", [])
        self.dft_tensor_notes_label.setText("\n".join(self.tensor_notes))
        return True

    # ---- state ------------------------------------------------------------------------

    def get_state(self) -> dict:
        state = super().get_state()
        state["dft"] = {"checked": self.checked_labels(),
                        "element": self.dft_element_combo.currentText(),
                        "isotope": self.dft_isotope_combo.currentText()}
        state.setdefault("extra", {})["DFT files"] = (
            f"sample {self.dft_sample_edit.text()}; reference {self.dft_reference_edit.text()}")
        return state

    def set_state(self, state: dict) -> None:
        super().set_state(state)
        dft = (state or {}).get("dft") or {}
        if self.dft_sample_edit.text().strip() and os.path.exists(
                self.dft_sample_edit.text().strip()):
            if self.read_files(quiet=True):
                index = self.dft_element_combo.findText(dft.get("element", ""))
                if index >= 0:
                    self.dft_element_combo.blockSignals(True)
                    self.dft_element_combo.setCurrentIndex(index)
                    self.dft_element_combo.blockSignals(False)
                self._on_element(checked_labels=set(dft.get("checked") or []) or None)
                index = self.dft_isotope_combo.findText(dft.get("isotope", ""))
                if index >= 0:
                    self.dft_isotope_combo.setCurrentIndex(index)
        self._sync_dft()
