"""
Simulate ▸ DFT tensors (explicit user requests, 2026-10-06 and
2026-10-07): import the shielding and EFG tensors of a sample calculation
(CASTEP .magres, VASP OUTCAR, ORCA output, Gaussian log), put its shieldings
on your experimental scale with one or more reference compounds, work out
the CSA and quadrupolar parameters (and their relative orientation) of the
chosen sites, simulate, and look at each site's tensors in the cell or
Cartesian frame.

The shift scale is one calibration per element: the references' calculated
shielding against the experimental shifts you type, a straight line fitted
by :mod:`opennmr.notebook.dft_calibration` (one reference: slope −1). The
sample's element picks its calibration, so ¹H can use L-alanine while
²³Na uses NaCl in the same sample. Calibrations can be saved and reused.

It *is* a Simulate ▸ Solid-state NMR panel with an import box on top:
the imported sites land in its site list, where they can be edited, and its
own field, regime, MAS, transitions, level of theory and axis -- and its
Simulate, plot, Keep and Send to Fit -- do the rest. The maths is
:mod:`opennmr.notebook.dft_nmr`, as a notebook calls it.
"""

from __future__ import annotations

import os

from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout,
    QHeaderView, QInputDialog, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMessageBox,
    QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout,
)

from ..notebook import dft_calibration as dcal
from ..notebook import dft_nmr
from .combined_panels import CombinedHamiltonianPanel, _spin_box

_FILE_FILTER = ("DFT output (*.magres OUTCAR* *.out *.log *.LOG);;All files (*)")

#: The help under the G=0 option (wording agreed with the user, 2026-10-07).
G0_HELP = (
    "Included: the long-range (macroscopic) contribution to the shielding, from the bulk "
    "magnetic susceptibility of a spherical sample, as for a crystalline material. Excluded: "
    "it is left out, which is right for a molecule in a vacuum box (the term then reflects "
    "the box, not the molecule). Applies to every VASP file here, sample and references. "
    "A CASTEP .magres always includes it (the file holds only the total σ); ORCA and "
    "Gaussian are molecular and have no such term.")

#: The line shown under it (the user's wording); the rest is its tooltip.
G0_SHORT = ("Included: the long-range contribution to the shielding (bulk susceptibility), as "
            "in crystalline materials. Excluded: omitted – for molecules. Every VASP file; a "
            "CASTEP .magres always includes it.")

# Reference table columns.
_C_USE, _C_NAME, _C_EL, _C_ATOMS, _C_SIGMA, _C_DELTA, _C_RESID = range(7)


class DftSimulationPanel(CombinedHamiltonianPanel):
    """``draw_tensors(draw)``: show ``draw(figure)`` on the Simulation plot
    as a tensor picture; ``draw_plot(draw, kind)``: show it as a plain
    plot (the calibration); ``style()``: that plot's style settings."""

    BLURB = ("Simulate from DFT: give the sample's calculation (CASTEP .magres, VASP OUTCAR, "
             "ORCA output or Gaussian log) and one or more reference compounds with their "
             "experimental shifts. Shifts, CSA (span/skew and Δ/η), C_Q and η_Q, and the CSA's "
             "orientation relative to the EFG come from the tensors; the sites then appear "
             "below, editable, for Simulate.")

    #: Not saved as plain widgets: the references are saved as points, the
    #: tables are rebuilt.
    _SITE_FIELDS = CombinedHamiltonianPanel._SITE_FIELDS + (
        "dft_reference_table", "dft_result_table", "dft_saved_combo", "dft_add_element_combo",
        "dft_average_list", "dft_state_combo")

    def __init__(self, plot_spectrum, add_to_open, draw_sites=None, draw_tensors=None,
                 style=None, parent=None, send_to_fit=None, draw_orders=None,
                 draw_plot=None, library_path=None) -> None:
        super().__init__(plot_spectrum=plot_spectrum, add_to_open=add_to_open,
                         draw_sites=draw_sites, parent=parent, send_to_fit=send_to_fit,
                         draw_orders=draw_orders)
        self._draw_tensors = draw_tensors
        self._draw_plot = draw_plot
        self._style = style or (lambda: {})
        self.library_path = library_path          # None: ~/.opennmr/dft_calibrations.json
        self.calcs = {"sample": None}
        self.ref_points: list[dict] = []
        self.average_groups: list[list[str]] = []
        self.calibration_errors: dict[str, str] = {}
        self.model = None
        self.tensor_notes = []
        self._filling = False
        layout = self.layout()
        layout.insertWidget(1, self._build_import_box())
        layout.insertWidget(2, self._build_reference_box())
        layout.insertWidget(layout.count() - 1, self._build_tensor_box())
        # The base panel starts with one empty site; a DFT panel starts
        # with none until something is imported.
        self.site_list.clear()
        self.site_dicts = []
        self._current = -1
        self._on_site_selected(-1)
        self.refresh_saved_calibrations()
        self._sync_dft()

    # ---- building -------------------------------------------------------------------

    def _file_row(self, attr, title):
        row = QHBoxLayout()
        edit = QLineEdit()
        edit.setPlaceholderText("CASTEP .magres, VASP OUTCAR, ORCA .out or Gaussian .log")
        setattr(self, attr, edit)
        row.addWidget(edit, 1)
        button = QPushButton("Browse…")
        button.clicked.connect(lambda: self._browse(edit, title))
        row.addWidget(button)
        return row

    def _build_import_box(self) -> QGroupBox:
        box = QGroupBox("DFT calculation (sample)")
        v = QVBoxLayout(box)
        form = QFormLayout()
        form.addRow("Sample:", self._file_row("dft_sample_edit", "Sample calculation"))
        self.dft_g0_check = QCheckBox("VASP: include the G=0 term")
        self.dft_g0_check.setChecked(True)
        self.dft_g0_check.setToolTip(G0_HELP)
        self.dft_g0_check.toggled.connect(self._on_g0_toggled)
        form.addRow("", self.dft_g0_check)
        self.dft_g0_help = QLabel(G0_SHORT)
        self.dft_g0_help.setToolTip(G0_HELP)
        self.dft_g0_help.setWordWrap(True)
        self.dft_g0_help.setStyleSheet("color: palette(mid);")
        form.addRow("", self.dft_g0_help)
        self.dft_read_button = QPushButton("Read Files")
        self.dft_read_button.setToolTip("Read the sample again, and every reference file.")
        self.dft_read_button.clicked.connect(lambda: self.read_files())
        form.addRow("", self.dft_read_button)
        self.dft_files_label = QLabel("")
        self.dft_files_label.setWordWrap(True)
        form.addRow(self.dft_files_label)

        self.dft_element_combo = QComboBox()
        self.dft_element_combo.setToolTip(
            "The element to simulate (one spectrum is one nucleus). Its shift scale is the "
            "calibration of the same element below.")
        self.dft_element_combo.currentIndexChanged.connect(lambda _i: self._on_element())
        form.addRow("Element:", self.dft_element_combo)
        self.dft_isotope_combo = QComboBox()
        self.dft_isotope_combo.setToolTip(
            "The observed isotope: sets the spin, the quadrupole moment Q (signed, openNMR's "
            "table) and so C_Q, and the Larmor frequency at your field.")
        form.addRow("Observed isotope:", self.dft_isotope_combo)
        site_row = QHBoxLayout()
        self.dft_site_list = QListWidget()
        self.dft_site_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
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
        return box

    def _build_state_box(self) -> QGroupBox:
        """Solid or solution, and which sites a fast motion averages."""
        box = QGroupBox("Sample state and motion")
        v = QVBoxLayout(box)
        form = QFormLayout()
        self.dft_state_combo = QComboBox()
        self.dft_state_combo.addItem("Solid – use the tensors in full", "solid")
        self.dft_state_combo.addItem("Solution – average the tensors (isotropic shift only)",
                                    "solution")
        self.dft_state_combo.setToolTip(
            "Solid: every site keeps its CSA and quadrupolar tensors, for a powder or single "
            "crystal lineshape.\nSolution: fast tumbling averages each tensor to its trace, so "
            "only the isotropic shift of each site remains (no anisotropy, no quadrupolar "
            "coupling).")
        self.dft_state_combo.currentIndexChanged.connect(lambda _i: self._on_state_changed())
        form.addRow("Sample state:", self.dft_state_combo)
        v.addLayout(form)
        v.addWidget(QLabel(
            "Fast motion (e.g. a rotating CH₃ or NH₃⁺): select the sites that exchange – "
            "rows of the table of sites below (click, Ctrl-click or Shift-click), or names in "
            "the Sites list at the top – and press Average Selected Sites. Their tensors are "
            "averaged in the calculation's frame, so a solid gives the motionally averaged "
            "lineshape and a solution gives the mean shift. The sites are recalculated at once."))
        self.dft_average_hint = QLabel("No sites averaged.")
        self.dft_average_hint.setStyleSheet("color: palette(mid);")
        v.addWidget(self.dft_average_hint)
        self.dft_average_list = QListWidget()
        self.dft_average_list.setMaximumHeight(70)
        self.dft_average_list.setToolTip("Groups of sites averaged by a fast motion. Each group "
                                         "is simulated as one site with the group's size as its "
                                         "amount.")
        v.addWidget(self.dft_average_list)
        row = QHBoxLayout()
        self.dft_average_button = QPushButton("Average Selected Sites")
        self.dft_average_button.setToolTip("Average the sites selected (highlighted) in the "
                                           "site list above into one group.")
        self.dft_average_button.clicked.connect(lambda: self.average_selected())
        self.dft_unaverage_button = QPushButton("Remove Group")
        self.dft_unaverage_button.clicked.connect(lambda: self.remove_average_group())
        row.addWidget(self.dft_average_button)
        row.addWidget(self.dft_unaverage_button)
        v.addLayout(row)
        return box

    # ---- motional averaging -----------------------------------------------------------

    def _on_state_changed(self) -> None:
        solution = self.dft_state_combo.currentData() == "solution"
        for w in (self.dft_include_csa_check, self.dft_include_quad_check):
            w.setEnabled(not solution)
        self._sync_dft()
        if self.model is not None:
            self.calculate()

    def _refresh_average_list(self) -> None:
        self.dft_average_list.clear()
        for group in self.average_groups:
            self.dft_average_list.addItem(", ".join(group))
        self.dft_average_hint.setVisible(not self.average_groups)

    def average_selected(self, labels=None) -> bool:
        """Group ``labels`` (default: the highlighted sites) for averaging."""
        if labels is None:
            labels = [item.text() for item in self.dft_site_list.selectedItems()]
            rows = sorted({i.row() for i in self.dft_result_table.selectedIndexes()})
            if self.model is not None:
                labels += [self.model["atoms"][r]["label"] for r in rows
                           if r < len(self.model["atoms"])]
            labels = list(dict.fromkeys(labels))
        labels = [str(x) for x in labels]
        if len(labels) < 2:
            QMessageBox.information(self, "Average sites",
                                    "Select at least two sites in the site list to average.")
            return False
        taken = {lab for g in self.average_groups for lab in g}
        clash = sorted(set(labels) & taken)
        if clash:
            QMessageBox.information(self, "Average sites",
                                    f"{', '.join(clash)} is already in a group; remove that group first.")
            return False
        for i in range(self.dft_site_list.count()):
            item = self.dft_site_list.item(i)
            if item.text() in labels:
                item.setCheckState(Qt.CheckState.Checked)
        self.average_groups.append(labels)
        self._refresh_average_list()
        self._sync_dft()
        self._recalculate_if_shown()
        return True

    def _recalculate_if_shown(self) -> None:
        """Sites already on screen are recalculated so a change to the
        averaging shows at once."""
        if self.model is not None:
            self.calculate()

    def remove_average_group(self, row: int | None = None) -> None:
        row = self.dft_average_list.currentRow() if row is None else row
        if 0 <= row < len(self.average_groups):
            del self.average_groups[row]
            self._refresh_average_list()
            self._recalculate_if_shown()
        self._sync_dft()

    def _build_reference_box(self) -> QGroupBox:
        box = QGroupBox("Shift scale: reference compounds")
        v = QVBoxLayout(box)
        intro = QLabel(
            "Add the reference calculations and type each one's experimental shift (δ_exp, on "
            "your scale). For each element, openNMR fits calculated shielding against δ_exp "
            "(σ_calc = a + b·δ_exp) and reads the sample's shifts off the line. One reference: "
            "the slope is −1. Rows without δ_exp are not used.")
        intro.setWordWrap(True)
        v.addWidget(intro)
        add_row = QHBoxLayout()
        self.dft_add_reference_button = QPushButton("Add Reference File…")
        self.dft_add_reference_button.setToolTip(
            "Read a reference calculation and add one row per group of equivalent sites of the "
            "chosen element (merge rows for a CH₃ or NH₃⁺, whose atoms exchange fast).")
        self.dft_add_reference_button.clicked.connect(self._on_add_reference)
        add_row.addWidget(self.dft_add_reference_button)
        add_row.addWidget(QLabel("rows for:"))
        self.dft_add_element_combo = QComboBox()
        self.dft_add_element_combo.setToolTip(
            "Which element's sites to add: the sample's element (default), or every element in "
            "the file.")
        add_row.addWidget(self.dft_add_element_combo, 1)
        v.addLayout(add_row)
        t = self.dft_reference_table = QTableWidget(0, 7)
        t.setHorizontalHeaderLabels(["Use", "Reference", "Element", "Atoms", "σ_calc (ppm)",
                                     "δ_exp (ppm)", "δ_calc − δ_exp"])
        t.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        t.horizontalHeader().setSectionResizeMode(_C_NAME, QHeaderView.ResizeMode.Stretch)
        t.verticalHeader().setVisible(False)
        t.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        t.setMinimumHeight(120)
        t.setToolTip("σ_calc: the mean isotropic shielding of the row's atoms. Type δ_exp; edit "
                     "the name. Select rows to merge or remove them.")
        t.itemChanged.connect(self._on_reference_item_changed)
        v.addWidget(t)
        row = QHBoxLayout()
        self.dft_merge_rows_button = QPushButton("Merge Rows")
        self.dft_merge_rows_button.setToolTip(
            "One row from the selected ones (same file and element): the mean over all their "
            "atoms, e.g. the three H of a methyl group.")
        self.dft_merge_rows_button.clicked.connect(lambda: self.merge_selected_rows())
        self.dft_remove_rows_button = QPushButton("Remove Rows")
        self.dft_remove_rows_button.clicked.connect(lambda: self.remove_selected_rows())
        for b in (self.dft_merge_rows_button, self.dft_remove_rows_button):
            row.addWidget(b)
        v.addLayout(row)

        form = QFormLayout()
        slope_row = QHBoxLayout()
        self.dft_slope_mode_combo = QComboBox()
        self.dft_slope_mode_combo.addItem("Fitted (2 or more references)", "fit")
        self.dft_slope_mode_combo.addItem("Fixed at", "fixed")
        self.dft_slope_mode_combo.setToolTip(
            "Fitted: the slope b of σ_calc = a + b·δ_exp is fitted. Fixed: only a is fitted. "
            "With one reference the slope is always fixed.")
        self.dft_slope_mode_combo.currentIndexChanged.connect(lambda _i: self._refit())
        slope_row.addWidget(self.dft_slope_mode_combo, 1)
        self.dft_fixed_slope_spin = _spin_box(-10.0, 10.0, -1.0, 4, step=0.01,
                                              tip="b, when fixed. −1 is the ideal scale.")
        self.dft_fixed_slope_spin.valueChanged.connect(lambda _v: self._refit())
        slope_row.addWidget(self.dft_fixed_slope_spin)
        form.addRow("Slope b:", slope_row)
        self.dft_calibration_label = QLabel("No references yet.")
        self.dft_calibration_label.setWordWrap(True)
        self.dft_calibration_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        form.addRow("Calibrations:", self.dft_calibration_label)
        v.addLayout(form)
        row = QHBoxLayout()
        self.dft_plot_calibration_button = QPushButton("Plot Calibration")
        self.dft_plot_calibration_button.setToolTip(
            "σ_calc against δ_exp for the sample's element, with the line, its equation and "
            "the residuals, on the Simulation plot (Plot Style applies).")
        self.dft_plot_calibration_button.clicked.connect(lambda: self.plot_calibration())
        self.dft_save_calibration_button = QPushButton("Save Calibration…")
        self.dft_save_calibration_button.setToolTip(
            "Save this element's calibration (with its points) to reuse with other samples.")
        self.dft_save_calibration_button.clicked.connect(lambda: self.save_calibration())
        for b in (self.dft_plot_calibration_button, self.dft_save_calibration_button):
            row.addWidget(b)
        v.addLayout(row)
        row = QHBoxLayout()
        self.dft_saved_combo = QComboBox()
        self.dft_saved_combo.setToolTip("Saved calibrations (~/.opennmr/dft_calibrations.json).")
        row.addWidget(self.dft_saved_combo, 1)
        self.dft_load_calibration_button = QPushButton("Load")
        self.dft_load_calibration_button.setToolTip(
            "Add the saved calibration's points as rows (replacing rows of that element).")
        self.dft_load_calibration_button.clicked.connect(lambda: self.load_saved_calibration())
        self.dft_delete_calibration_button = QPushButton("Delete")
        self.dft_delete_calibration_button.clicked.connect(lambda: self.delete_saved_calibration())
        row.addWidget(self.dft_load_calibration_button)
        row.addWidget(self.dft_delete_calibration_button)
        form2 = QFormLayout()
        form2.addRow("Saved:", row)
        v.addLayout(form2)

        typed = QFormLayout()
        override = QHBoxLayout()
        self.dft_sigma_override_check = QCheckBox("Type the scale instead:")
        self.dft_sigma_override_check.setToolTip(
            "Use δ = δ_ref + m(σ_ref − σ) with numbers you type, for the element being "
            "calculated, instead of the references.")
        self.dft_sigma_override_check.toggled.connect(lambda _c: self._sync_dft())
        override.addWidget(self.dft_sigma_override_check)
        self.dft_sigma_ref_spin = _spin_box(-1e5, 1e5, 0.0, 4, " ppm", tip="σ_ref")
        override.addWidget(QLabel("σ_ref"))
        override.addWidget(self.dft_sigma_ref_spin, 1)
        typed.addRow(override)
        row = QHBoxLayout()
        self.dft_delta_ref_spin = _spin_box(-1e5, 1e5, 0.0, 3, " ppm",
                                            tip="δ_ref: the reference's experimental shift.")
        row.addWidget(QLabel("δ_ref"))
        row.addWidget(self.dft_delta_ref_spin, 1)
        self.dft_slope_spin = _spin_box(0.01, 10.0, 1.0, 4, step=0.01,
                                        tip="m in δ = δ_ref + m(σ_ref − σ).")
        row.addWidget(QLabel("m"))
        row.addWidget(self.dft_slope_spin, 1)
        typed.addRow(row)
        v.addLayout(typed)

        v.addWidget(self._build_state_box())
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
        self.dft_result_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.dft_result_table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.dft_result_table.verticalHeader().setVisible(False)
        self.dft_result_table.setMaximumHeight(200)
        self.dft_result_table.setToolTip(
            "Every chosen atom. ± δiso: the standard uncertainty of the shift from the "
            "calibration's scatter (none with one reference). Ω, κ: span and skew "
            "(Herzfeld–Berger: δ11 ≥ δ22 ≥ δ33). δaniso, ηCSA: Haeberlen (δaniso = δzz − δiso; "
            "Δδ = 1.5 δaniso). C_Q = eQV_zz/h (signed), ηQ, P_Q = |C_Q|√(1 + ηQ²/3). α, β, γ: "
            "the CSA axes in the EFG frame.")
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
            "Cartesian: the calculation's own x, y, z – the only choice for a calculation "
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
                                              _FILE_FILTER)
        if path:
            edit.setText(path)
            self.read_files(quiet=False)

    def _read(self, path):
        return dft_nmr.read_calculation(path, include_g0=self.dft_g0_check.isChecked())

    @staticmethod
    def _describe_calc(key, c) -> str:
        n_sigma = sum(a["sigma"] is not None for a in c["atoms"])
        n_efg = sum(a["efg"] is not None for a in c["atoms"])
        g0 = {"included": "; G=0 included", "excluded": "; G=0 excluded",
              "fixed": "; G=0 included (fixed in a .magres)"}.get(c.get("g0"), "")
        return (f"{key}: {c['code']}, {len(c['atoms'])} atoms ("
                + ", ".join(f"{s} {sum(a['species'] == s for a in c['atoms'])}"
                            for s in dft_nmr.species_of(c))
                + f"); shielding for {n_sigma}, EFG for {n_efg}"
                + ("" if c.get("lattice") is not None else "; no cell") + g0 + ". "
                + dft_nmr.describe_method(c.get("method")) + ".")

    def read_files(self, quiet: bool = False) -> bool:
        """Read the sample (required), and every reference file again (with
        the current G=0 choice)."""
        try:
            sample = self.dft_sample_edit.text().strip()
            if not sample:
                raise ValueError("Choose the sample's calculation file.")
            self.calcs["sample"] = self._read(sample)
        except (OSError, ValueError, KeyError) as exc:
            self.dft_files_label.setText(f"Could not read: {exc}")
            if not quiet:
                QMessageBox.warning(self, "DFT import", str(exc))
            return False
        problems = self._reread_reference_points()
        lines = [self._describe_calc("Sample", self.calcs["sample"])]
        lines += problems
        lines += self._g0_notes()
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

    def _reread_reference_points(self) -> list[str]:
        """Re-read each reference file once and refresh its points'
        shieldings. Points whose file is gone keep their stored σ."""
        problems = []
        by_file = {}
        for p in self.ref_points:
            if p.get("file"):
                by_file.setdefault(p["file"], []).append(p)
        fresh = {}
        for path in by_file:
            if not os.path.exists(path):
                problems.append(f"Reference {os.path.basename(path)} not found: its rows keep "
                                "their stored σ.")
                continue
            try:
                fresh[path] = self._read(path)
            except (OSError, ValueError, KeyError) as exc:
                problems.append(f"Reference {os.path.basename(path)}: {exc}")
        new = []
        for p in self.ref_points:
            calc = fresh.get(p.get("file"))
            if calc is not None and p.get("labels"):
                try:
                    p = dcal.refresh_point(p, calc)
                except ValueError as exc:
                    problems.append(f"{p['name']}: {exc}")
            new.append(p)
        self.ref_points = new
        self._fill_reference_table()
        return problems

    def _g0_notes(self) -> list[str]:
        if self.dft_g0_check.isChecked():
            return []
        codes = [self.calcs["sample"]] if self.calcs["sample"] else []
        fixed = sorted({os.path.basename(p["file"]) for p in self.ref_points
                        if p.get("g0") == "fixed"}
                       | {os.path.basename(c["path"]) for c in codes if c.get("g0") == "fixed"})
        if fixed:
            return ["G=0 is excluded for VASP files only: " + ", ".join(fixed)
                    + " (CASTEP) keep theirs – a .magres holds only the total σ – so they are "
                    "not on the same scale as the VASP files."]
        return []

    def _on_g0_toggled(self, _checked) -> None:
        """Re-read everything at once, so the sample and every reference
        change together (they used to change only on Read Files)."""
        if not self.dft_sample_edit.text().strip() and not self.ref_points:
            return
        had_model = self.model is not None
        had_spectrum = self.spectrum is not None
        if self.dft_sample_edit.text().strip():
            if not self.read_files(quiet=True):
                return
        else:
            self._reread_reference_points()
        if had_model:
            self.calculate(simulate=had_spectrum)

    def _on_element(self, checked_labels=None) -> None:
        element = self.dft_element_combo.currentText()
        self.dft_isotope_combo.clear()
        self.dft_site_list.clear()
        if checked_labels is None:
            self.average_groups = []
            self._refresh_average_list()
        self._fill_add_element_combo()
        if not element or self.calcs["sample"] is None:
            self._sync_dft()
            return
        isotopes = dft_nmr.nmr_isotopes(element)
        self.dft_isotope_combo.addItems(isotopes)
        for atom in self.calcs["sample"]["atoms"]:
            if atom["species"] != element or atom["sigma"] is None:
                continue
            item = QListWidgetItem(atom["label"])
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            on = checked_labels is None or atom["label"] in checked_labels
            item.setCheckState(Qt.CheckState.Checked if on else Qt.CheckState.Unchecked)
            self.dft_site_list.addItem(item)
        self._refit()
        self._sync_dft()

    def _fill_add_element_combo(self) -> None:
        keep = self.dft_add_element_combo.currentData()
        self.dft_add_element_combo.clear()
        element = self.dft_element_combo.currentText()
        if element:
            self.dft_add_element_combo.addItem(f"{element} (the sample's element)", element)
        self.dft_add_element_combo.addItem("Every element in the file", "*")
        index = self.dft_add_element_combo.findData(keep)
        self.dft_add_element_combo.setCurrentIndex(max(index, 0))

    def _check_all(self, state: bool) -> None:
        for i in range(self.dft_site_list.count()):
            self.dft_site_list.item(i).setCheckState(
                Qt.CheckState.Checked if state else Qt.CheckState.Unchecked)

    def checked_labels(self) -> list[str]:
        return [self.dft_site_list.item(i).text() for i in range(self.dft_site_list.count())
                if self.dft_site_list.item(i).checkState() == Qt.CheckState.Checked]

    # ---- references -------------------------------------------------------------------

    def _on_add_reference(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Reference calculation", "", _FILE_FILTER)
        if path:
            self.add_reference_file(path, quiet=False)

    def add_reference_file(self, path, element=None, quiet: bool = True) -> int:
        """Add one row per group of equivalent sites of ``element`` (default:
        the choice beside the button) from a reference file. Returns the
        number of rows added."""
        try:
            calc = self._read(path)
            if element is None:
                element = self.dft_add_element_combo.currentData() or "*"
            species = dft_nmr.species_of(calc)
            wanted = species if element == "*" else [element]
            missing = [e for e in wanted if e not in species]
            if missing and element != "*":
                raise ValueError(f"{os.path.basename(path)} has no {element} (it has "
                                 + ", ".join(species) + ").")
            added = []
            for e in wanted:
                added += dcal.suggest_points(calc, e)
            if not added:
                raise ValueError(f"{os.path.basename(path)} has no shieldings for "
                                 + ", ".join(wanted) + ".")
        except (OSError, ValueError, KeyError) as exc:
            if not quiet:
                QMessageBox.warning(self, "Add reference", str(exc))
            self.dft_calibration_label.setText(f"Could not add: {exc}")
            return 0
        self.ref_points += added
        self._fill_reference_table()
        self._refit()
        return len(added)

    def _fill_reference_table(self) -> None:
        t = self.dft_reference_table
        self._filling = True
        t.setRowCount(len(self.ref_points))
        for r, p in enumerate(self.ref_points):
            use = QTableWidgetItem("")
            use.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled
                         | Qt.ItemFlag.ItemIsSelectable)
            use.setCheckState(Qt.CheckState.Checked if p.get("use", True)
                              else Qt.CheckState.Unchecked)
            t.setItem(r, _C_USE, use)
            items = [(p["name"], True), (p["element"], False),
                     (dft_nmr._group_label(p["labels"]) if p["labels"] else "typed", False),
                     (f"{p['sigma']:.4f}", False),
                     ("" if p.get("delta_exp") is None else f"{p['delta_exp']:g}", True),
                     ("—", False)]
            for c, (text, editable) in enumerate(items, start=1):
                item = QTableWidgetItem(text)
                flags = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
                if editable:
                    flags |= Qt.ItemFlag.ItemIsEditable
                item.setFlags(flags)
                if c in (_C_SIGMA, _C_DELTA, _C_RESID):
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight
                                          | Qt.AlignmentFlag.AlignVCenter)
                if c == _C_ATOMS:
                    item.setToolTip(", ".join(p["labels"]) or "a typed shielding")
                t.setItem(r, c, item)
        self._filling = False

    def _on_reference_item_changed(self, item) -> None:
        if self._filling:
            return
        r, c = item.row(), item.column()
        if not 0 <= r < len(self.ref_points):
            return
        p = self.ref_points[r]
        if c == _C_USE:
            p["use"] = item.checkState() == Qt.CheckState.Checked
        elif c == _C_NAME:
            p["name"] = item.text().strip() or p["name"]
        elif c == _C_DELTA:
            text = item.text().strip().replace("−", "-")
            if not text:
                p["delta_exp"] = None
            else:
                try:
                    p["delta_exp"] = float(text)
                except ValueError:
                    self._filling = True
                    item.setText("" if p.get("delta_exp") is None else f"{p['delta_exp']:g}")
                    self._filling = False
                    return
        else:
            return
        self._refit()

    def set_experimental_shift(self, row: int, value) -> None:
        """Type δ_exp into a row (as the user would)."""
        self.dft_reference_table.item(row, _C_DELTA).setText("" if value is None else str(value))

    def _selected_rows(self) -> list[int]:
        return sorted({i.row() for i in self.dft_reference_table.selectedIndexes()})

    def merge_selected_rows(self, rows=None) -> bool:
        rows = self._selected_rows() if rows is None else sorted(rows)
        try:
            merged = dcal.merge_points([self.ref_points[r] for r in rows])
        except (ValueError, IndexError) as exc:
            QMessageBox.warning(self, "Merge rows", str(exc))
            return False
        keep = [p for i, p in enumerate(self.ref_points) if i not in rows]
        keep.insert(rows[0], merged)
        self.ref_points = keep
        self._fill_reference_table()
        self._refit()
        return True

    def remove_selected_rows(self, rows=None) -> None:
        rows = set(self._selected_rows() if rows is None else rows)
        self.ref_points = [p for i, p in enumerate(self.ref_points) if i not in rows]
        self._fill_reference_table()
        self._refit()

    def slope_choice(self):
        return (None if self.dft_slope_mode_combo.currentData() == "fit"
                else self.dft_fixed_slope_spin.value())

    def calibrations(self) -> dict:
        """Each element's calibration from the usable rows (errors are in
        ``calibration_errors``)."""
        out, self.calibration_errors = {}, {}
        elements = dict.fromkeys(p["element"] for p in self.ref_points)
        for e in elements:
            pts = [p for p in self.ref_points if p["element"] == e]
            if not dcal._usable(pts):
                continue
            try:
                out[e] = dcal.fit_calibration(pts, element=e, slope=self.slope_choice())
            except ValueError as exc:
                self.calibration_errors[e] = str(exc)
        return out

    def _refit(self) -> None:
        cals = self.calibrations()
        t = self.dft_reference_table
        self._filling = True
        for r, p in enumerate(self.ref_points):
            text = "—"
            cal = cals.get(p["element"])
            if cal is not None:
                for q in cal["points"]:
                    if q["name"] == p["name"] and q["labels"] == p["labels"] \
                            and q["file"] == p["file"]:
                        text = f"{q['residual']:+.3f}"
            if t.item(r, _C_RESID) is not None:
                t.item(r, _C_RESID).setText(text)
        self._filling = False
        lines = []
        current = self.dft_element_combo.currentText()
        for e, cal in cals.items():
            line = dcal.summary_text(cal)
            if e == current:
                line = "▶ " + line
            lines.append(line)
            lines += [f"   • {n}" for n in cal["notes"]]
        lines += [f"{e}: {msg}" for e, msg in self.calibration_errors.items()]
        if not lines:
            lines = ["No usable references yet: add a reference file and type its δ_exp."]
        elif current and current not in cals and not self.dft_sigma_override_check.isChecked():
            lines.append(f"No calibration for {current} yet: add a {current} reference, or type "
                         "the scale.")
        self.dft_calibration_label.setText("\n".join(lines))
        self._sync_dft()

    # ---- calibration plot and library -------------------------------------------------

    def current_calibration(self):
        element = self.dft_element_combo.currentText()
        if self.dft_sigma_override_check.isChecked():
            return dcal.manual_calibration(element, self.dft_sigma_ref_spin.value(),
                                           self.dft_delta_ref_spin.value(),
                                           self.dft_slope_spin.value())
        cals = self.calibrations()
        if element in cals:
            return cals[element]
        if element in self.calibration_errors:
            raise ValueError(f"{element}: {self.calibration_errors[element]}")
        raise ValueError(f"No shift scale for {element}: add a reference calculation that "
                         f"contains {element} and type its experimental shift, or tick "
                         "'Type the scale instead'.")

    def plot_calibration(self) -> bool:
        if self._draw_plot is None:
            return False
        try:
            element = self.dft_element_combo.currentText()
            cal = self.calibrations().get(element)
            if cal is None:
                raise ValueError(f"No {element} references with δ_exp to plot.")
        except ValueError as exc:
            QMessageBox.warning(self, "Plot calibration", str(exc))
            return False

        def draw(figure):
            look = self._style() or {}
            dcal.plot_calibration(cal, figure=figure,
                                  fontsize=float(look.get("labelsize") or 11.0))
        self._draw_plot(draw, "curve")
        return True

    def refresh_saved_calibrations(self) -> None:
        self.dft_saved_combo.clear()
        try:
            cals = dcal.load_library(self.library_path)
        except (OSError, ValueError) as exc:
            self.dft_saved_combo.addItem(f"(could not read: {exc})", None)
            return
        for c in cals:
            self.dft_saved_combo.addItem(f"{c['name']} ({c['element']}, n = {c.get('n', 0)})",
                                         (c["name"], c["element"]))
        self._sync_dft()

    def save_calibration(self, name=None) -> bool:
        element = self.dft_element_combo.currentText()
        cal = self.calibrations().get(element)
        if cal is None:
            QMessageBox.warning(self, "Save calibration",
                                f"No {element} calibration to save: add references with δ_exp.")
            return False
        if name is None:
            default = f"{element}, {dft_nmr.describe_method(cal.get('method'))}"
            name, ok = QInputDialog.getText(self, "Save calibration", "Name:", text=default)
            if not ok:
                return False
        try:
            dcal.save_calibration(cal, name, self.library_path)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Save calibration", str(exc))
            return False
        self.refresh_saved_calibrations()
        return True

    def load_saved_calibration(self, name=None, element=None) -> bool:
        if name is None:
            data = self.dft_saved_combo.currentData()
            if not data:
                return False
            name, element = data
        try:
            cal = dcal.find_calibration(name, element, self.library_path)
        except KeyError as exc:
            QMessageBox.warning(self, "Load calibration", str(exc))
            return False
        self.ref_points = [p for p in self.ref_points if p["element"] != cal["element"]]
        for p in cal["points"]:
            q = {k: v for k, v in p.items() if k not in ("delta_calc", "residual")}
            self.ref_points.append(q)
        if cal.get("free_slope"):
            self.dft_slope_mode_combo.setCurrentIndex(self.dft_slope_mode_combo.findData("fit"))
        else:
            self.dft_slope_mode_combo.setCurrentIndex(
                self.dft_slope_mode_combo.findData("fixed"))
            self.dft_fixed_slope_spin.setValue(cal["slope"])
        self._reread_reference_points()
        self._refit()
        return True

    def delete_saved_calibration(self) -> bool:
        data = self.dft_saved_combo.currentData()
        if not data:
            return False
        done = dcal.delete_calibration(data[0], data[1], self.library_path)
        self.refresh_saved_calibrations()
        return done

    # ---- syncing ----------------------------------------------------------------------

    def _sync_dft(self) -> None:
        typed = self.dft_sigma_override_check.isChecked()
        for w in (self.dft_sigma_ref_spin, self.dft_delta_ref_spin, self.dft_slope_spin):
            w.setEnabled(typed)
        self.dft_fixed_slope_spin.setEnabled(self.dft_slope_mode_combo.currentData() == "fixed")
        have = self.calcs["sample"] is not None
        self.dft_calculate_button.setEnabled(have)
        self.dft_merge_rows_button.setEnabled(bool(self.ref_points))
        self.dft_remove_rows_button.setEnabled(bool(self.ref_points))
        has_saved = self.dft_saved_combo.currentData() is not None
        self.dft_load_calibration_button.setEnabled(has_saved)
        self.dft_delete_calibration_button.setEnabled(has_saved)
        self._tensor_form.setRowVisible(self.dft_relative_combo,
                                        self.dft_tensor_style_combo.currentData() == "surface")
        sample = self.calcs["sample"]
        periodic = sample is not None and sample.get("lattice") is not None
        cell_item = self.dft_frame_combo.model().item(0)
        cell_item.setEnabled(periodic or sample is None)
        if sample is not None and not periodic:
            self.dft_frame_combo.setCurrentIndex(1)
        self.dft_show_tensor_button.setEnabled(self.model is not None)
        self.dft_average_button.setEnabled(have or self.model is not None)
        self.dft_unaverage_button.setEnabled(bool(self.average_groups))

    # ---- calculating ------------------------------------------------------------------

    def import_arguments(self) -> dict:
        labels = self.checked_labels()
        if not labels:
            raise ValueError("Tick at least one site.")
        return {"element": self.dft_element_combo.currentText(), "labels": labels,
                "nucleus": self.dft_isotope_combo.currentText() or None,
                "fwhm": self.dft_fwhm_spin.value(),
                "include_csa": self.dft_include_csa_check.isChecked(),
                "include_quadrupolar": self.dft_include_quad_check.isChecked(),
                "merge": self.dft_merge_check.isChecked(),
                "state": self.dft_state_combo.currentData(),
                "average": [[lab for lab in g if lab in labels] for g in self.average_groups
                            if sum(lab in labels for lab in g) >= 2]}

    def calculate(self, simulate: bool = True) -> bool:
        if self.calcs["sample"] is None and not self.read_files(quiet=True):
            QMessageBox.warning(self, "DFT import", self.dft_files_label.text())
            return False
        try:
            args = self.import_arguments()
            cal = self.current_calibration()
            model = dft_nmr.dft_sites(self.calcs["sample"], calibration=cal, **args)
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
        with self.bulk():
            CombinedHamiltonianPanel.add_sites(self, model["site_dicts"])
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
        """Notebook code reproducing this import: the calibration (from its
        reference files) and the sites."""
        args = self.import_arguments()
        element = args["element"]
        g0 = self.dft_g0_check.isChecked()
        head = "from opennmr.notebook import dft_nmr, dft_calibration as dc\n"
        if self.dft_sigma_override_check.isChecked():
            cal_code = (f"cal = dc.manual_calibration({element!r}, "
                        f"{self.dft_sigma_ref_spin.value()!r}, {self.dft_delta_ref_spin.value()!r},"
                        f" {self.dft_slope_spin.value()!r})\n")
        else:
            self.current_calibration()          # raises if there is none
            cal_code = dcal.calibration_to_python(self.ref_points, element, self.slope_choice(),
                                                  include_g0=g0, header=False)
        kwargs = ", ".join(f"{k}={v!r}" for k, v in args.items() if v is not None)
        return (head + cal_code
                + f"sample = dft_nmr.read_calculation({self.dft_sample_edit.text().strip()!r}, "
                  f"include_g0={g0})\n"
                + f"model = dft_nmr.dft_sites(sample, calibration=cal, {kwargs})\n"
                + "print(dft_nmr.format_table(model))\n")

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
                        "isotope": self.dft_isotope_combo.currentText(),
                        "state": self.dft_state_combo.currentData(),
                        "average": [list(g) for g in self.average_groups],
                        "references": dcal._jsonable(self.ref_points)}
        refs = "; ".join(f"{p['name']} ({p['element']}, σ {p['sigma']:.3f}, δ_exp "
                         f"{p.get('delta_exp')})" for p in self.ref_points
                         if p.get("use", True) and p.get("delta_exp") is not None)
        state.setdefault("extra", {})["DFT files"] = (
            f"sample {self.dft_sample_edit.text()}; references: {refs or 'none'}")
        try:
            cal = self.current_calibration()
            state["extra"]["DFT shift scale"] = dcal.summary_text(cal)
        except ValueError:
            pass
        return state

    def set_state(self, state: dict) -> None:
        super().set_state(state)
        state = state or {}
        dft = state.get("dft") or {}
        self.ref_points = [dict(p) for p in dft.get("references") or []]
        # A round-6 state (2026-10-06) had one reference file and δ_ref.
        widgets = state.get("widgets") or {}
        old_ref = (widgets.get("dft_reference_edit") or {}).get("value")
        if not self.ref_points and old_ref and os.path.exists(old_ref) \
                and not self.dft_sigma_override_check.isChecked():
            element = dft.get("element") or None
            try:
                calc = self._read(old_ref)
                if element in dft_nmr.species_of(calc):
                    pt = dcal.reference_point(calc, element=element,
                                              delta_exp=self.dft_delta_ref_spin.value(),
                                              name=os.path.basename(old_ref))
                    self.ref_points.append(pt)
            except (OSError, ValueError, KeyError):
                pass
        self._fill_reference_table()
        if self.dft_sample_edit.text().strip() and os.path.exists(
                self.dft_sample_edit.text().strip()):
            if self.read_files(quiet=True):
                index = self.dft_element_combo.findText(dft.get("element", ""))
                if index >= 0:
                    self.dft_element_combo.blockSignals(True)
                    self.dft_element_combo.setCurrentIndex(index)
                    self.dft_element_combo.blockSignals(False)
                self._on_element(checked_labels=set(dft.get("checked") or []) or None)
                self.average_groups = [list(g) for g in dft.get("average") or []]
                self._refresh_average_list()
                index = self.dft_isotope_combo.findText(dft.get("isotope", ""))
                if index >= 0:
                    self.dft_isotope_combo.setCurrentIndex(index)
        self._refit()
        state_index = self.dft_state_combo.findData(dft.get("state", "solid"))
        if state_index >= 0:
            self.dft_state_combo.setCurrentIndex(state_index)
        self._sync_dft()

