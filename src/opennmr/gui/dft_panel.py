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

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout,
    QHeaderView, QInputDialog, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMessageBox,
    QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from ..notebook import dft_calibration as dcal
from ..notebook import crystal_nmr, dft_nmr, dft_structure
from ..notebook import paramagnetic_nmr as pmr
from .combined_panels import CombinedHamiltonianPanel, _spin_box
from .widgets.collapsible_section import CollapsibleSection
from .widgets.typed_spinbox import TypedSpinBox

_FILE_FILTER = ("DFT output (*.magres OUTCAR* *.out *.outp *.log *.LOG *.txt *.inp *.xyz);;"
                "All files (*)")

#: The structure an orca_pnmr file is drawn on (it has no coordinates).
_STRUCTURE_FILTER = "Structure (*.xyz *.out *.log);;All files (*)"

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

    BLURB = ("Simulate a spectrum from a DFT calculation (CASTEP, VASP, ORCA, Gaussian or "
             "CRYSTAL): work through the numbered steps in order.")

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
        self._build_steps()
        # The base panel starts with one empty site; a DFT panel starts
        # with none until something is imported.
        self.site_list.clear()
        self.site_dicts = []
        self._current = -1
        self._on_site_selected(-1)
        self.refresh_saved_calibrations()
        self._sync_dft()

    # ---- building -------------------------------------------------------------------

    #: The workflow's numbered sections (explicit user request, 2026-10-27:
    #: "sub-menus that are collapsible and make the layout clearer").
    STEP_TITLES = ("1 · Calculation and sites",
                   "2 · Shift scale (references) or paramagnetic scaling",
                   "3 · Solid or solution, and motional averaging",
                   "4 · Experiment: field, static or MAS",
                   "5 · Simulate",
                   "6 · View tensors (3D)")

    def _build_steps(self) -> None:
        """Rearrange the Solid-state NMR panel this extends into the six
        steps of :attr:`STEP_TITLES`, each a collapsible section, with
        the DFT parts between them. Its own parts are moved, not rebuilt,
        so everything it does (and every test of it) still holds."""
        layout = self.layout()
        taken = []
        while layout.count():
            taken.append(layout.takeAt(0))
        blurb = taken[0].widget()
        sites_box = taken[2].widget()
        actions = [item for item in taken[4:] if item.spacerItem() is None]
        # The base's experiment form: the nucleus comes from step 1.
        setup = self._setup_form
        setup.setParent(None)
        self.dft_nucleus_label = QLabel("— (choose the sites in step 1)")
        self.dft_nucleus_label.setToolTip("The observed isotope chosen in step 1.")
        setup.insertRow(0, "Nucleus:", self.dft_nucleus_label)
        setup.setRowVisible(self.nucleus_combo, False)

        def part(item):
            return item.widget() if item.widget() is not None else item.layout()

        def section(title, *parts):
            body = QWidget()
            v = QVBoxLayout(body)
            v.setContentsMargins(0, 0, 0, 0)
            for p in parts:
                if isinstance(p, QWidget):
                    v.addWidget(p)
                else:
                    p.setParent(None)
                    v.addLayout(p)
            return CollapsibleSection(title, content=body)

        self.dft_sites_section = section(
            "Edit the sites (peaks)", sites_box, self.site_box)
        self.dft_axis_section = section("Axis and plot options", part(actions[0]))
        self.dft_steps = [
            section(self.STEP_TITLES[0], self._build_import_box()),
            section(self.STEP_TITLES[1], self._build_paramagnetic_box(),
                    self._build_orca_pm_box(), self._build_reference_box()),
            section(self.STEP_TITLES[2], self._build_state_box()),
            section(self.STEP_TITLES[3], setup),
            section(self.STEP_TITLES[4], self._build_calculate_box(), self.dft_sites_section,
                    self.dft_axis_section, *[part(item) for item in actions[1:]]),
            section(self.STEP_TITLES[5], self._build_tensor_box()),
        ]
        self.dft_isotope_combo.currentTextChanged.connect(self._on_isotope)
        self.simulate_button.setText("Simulate the Sites as Edited")
        self.simulate_button.setToolTip(
            "Simulate the sites as they are now (after editing them above), without "
            "recalculating them from the DFT file.")
        layout.addWidget(blurb)
        for step in self.dft_steps:
            layout.addWidget(step)
        layout.addStretch(1)

    def _on_isotope(self, isotope: str) -> None:
        """Step 1's isotope is the observed nucleus (step 4 shows it)."""
        self.dft_nucleus_label.setText(isotope or "— (choose the sites in step 1)")
        if isotope:
            self.nucleus_combo.setCurrentText(isotope)

    def _file_row(self, attr, title):
        row = QHBoxLayout()
        edit = QLineEdit()
        edit.setPlaceholderText("CASTEP .magres, VASP OUTCAR, ORCA .out or orca_pnmr file, "
                                "Gaussian .log or CRYSTAL properties output")
        setattr(self, attr, edit)
        row.addWidget(edit, 1)
        button = QPushButton("Browse…")
        button.clicked.connect(lambda: self._browse(edit, title))
        row.addWidget(button)
        return row

    def _build_import_box(self) -> QWidget:
        box = QWidget()
        v = QVBoxLayout(box)
        v.setContentsMargins(0, 0, 0, 0)
        form = QFormLayout()
        form.addRow("Sample:", self._file_row("dft_sample_edit", "Sample calculation"))
        # An orca_pnmr input/output has no coordinates (explicit user request,
        # 2026-11-13): the geometry comes from the job's .xyz or ORCA output.
        structure_row = QHBoxLayout()
        self.dft_structure_edit = QLineEdit()
        self.dft_structure_edit.setPlaceholderText(
            "orca_pnmr files only: the .xyz or ORCA .out of the same molecule")
        self.dft_structure_edit.setToolTip(
            "An orca_pnmr input (.pnmr.inp) or output has no atomic positions. Give the "
            "geometry it was calculated for: an .xyz (the job's own, orca_plot's, or a "
            "g-orientation .gori.xyz) or the ORCA output, to draw the molecule and number its "
            "atoms as ORCA does. Left empty, a file with the same job name next to it is used "
            "when its elements match.")
        structure_row.addWidget(self.dft_structure_edit, 1)
        self.dft_structure_browse_button = QPushButton("Browse…")
        self.dft_structure_browse_button.clicked.connect(self._browse_structure)
        structure_row.addWidget(self.dft_structure_browse_button)
        self._structure_file_row = structure_row
        form.addRow("Structure:", structure_row)
        self._import_form = form
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

    def _build_paramagnetic_box(self) -> QGroupBox:
        """The scaling of a CRYSTAL calculation's hyperfine interactions."""
        box = QGroupBox("Paramagnetic shift scaling (CRYSTAL)")
        self.dft_pm_box = box
        v = QVBoxLayout(box)
        intro = QLabel(
            "A CRYSTAL calculation is the fully spin-polarised state; the NMR shift is its "
            "Fermi contact and spin dipolar hyperfine interaction scaled by φ = ⟨S_z⟩/S at your "
            "temperature and field: δ = (φ/2) A/ν₀. Give φ, or a susceptibility, or a Weiss "
            "constant with the effective moment and spin. The orbital (diamagnetic) shift is "
            "not in the calculation: add it below. Reference compounds are not used.")
        intro.setWordWrap(True)
        v.addWidget(intro)
        form = QFormLayout()
        self._pm_form = form
        self.dft_pm_route_combo = QComboBox()
        for key, text in (("curie_weiss", "Weiss constant θ, μ_eff and spin S"),
                          ("susceptibility", "Magnetic susceptibility χ_M and spin S"),
                          ("phi", "Scaling factor φ")):
            self.dft_pm_route_combo.addItem(text, key)
        self.dft_pm_route_combo.setToolTip("How φ is found; each needs the others on the "
                                           "rows below it.")
        form.addRow("Scaling from:", self.dft_pm_route_combo)
        self.dft_pm_phi_spin = _spin_box(-100.0, 100.0, 0.02, 6, step=0.001,
                                         tip="φ = ⟨S_z⟩/S at the field B0 below (φ ∝ B0).")
        self.dft_pm_chi_spin = _spin_box(0.0, 1e6, 0.01, 6, step=0.001,
                                         tip="Molar magnetic susceptibility of the sample at the "
                                             "experiment's temperature (paramagnetic, positive).")
        self.dft_pm_chi_units_combo = QComboBox()
        self.dft_pm_chi_units_combo.addItem("cm³/mol (cgs, emu/mol)", "cm3/mol")
        self.dft_pm_chi_units_combo.addItem("m³/mol (SI)", "m3/mol")
        self.dft_pm_theta_spin = _spin_box(-1e5, 1e5, -50.0, 3, " K", step=1.0,
                                           tip="Weiss constant θ: negative for antiferromagnets. "
                                               "T − θ must be positive.")
        self.dft_pm_mueff_spin = _spin_box(0.0, 100.0, 3.87, 4, " μB", step=0.01,
                                           tip="Effective magnetic moment per formula unit.")
        self.dft_pm_spin_spin = _spin_box(0.0, 20.0, 1.5, 3, step=0.5,
                                          tip="Total spin quantum number S per formula unit.")
        self.dft_pm_T_spin = _spin_box(1.0, 2000.0, 298.0, 2, " K", step=1.0,
                                       tip="Sample temperature in the experiment.")
        self.dft_pm_B0_spin = _spin_box(0.001, 100.0, 11.7, 4, " T", step=0.1,
                                        tip="Magnetic field. The Curie–Weiss shift does not depend "
                                            "on it; a typed φ or a susceptibility belongs to it.")
        self.dft_pm_use_field_button = QPushButton("From Larmor frequency")
        self.dft_pm_use_field_button.setToolTip(
            "Set B0 from the Larmor frequency and nucleus of the simulation settings.")
        self.dft_pm_use_field_button.clicked.connect(lambda: self._pm_field_from_larmor())
        self.dft_pm_g_spin = _spin_box(1.0, 3.0, 2.0, 5, step=0.001,
                                       tip="Electron g factor used in φ (2.0 unless you know "
                                           "better).")
        self.dft_pm_components_combo = QComboBox()
        for key, text in (("both", "Fermi contact + spin dipolar"), ("contact", "Fermi contact only"),
                          ("dipolar", "Spin dipolar only")):
            self.dft_pm_components_combo.addItem(text, key)
        self.dft_pm_orbital_spin = _spin_box(-1e5, 1e5, 0.0, 3, " ppm", step=1.0,
                                             tip="Orbital (diamagnetic) shift added to every "
                                                 "site, e.g. the shift of a diamagnetic analogue.")
        self._pm_rows = {
            "phi": [self.dft_pm_phi_spin],
            "susceptibility": [self.dft_pm_chi_spin, self.dft_pm_chi_units_combo,
                               self.dft_pm_spin_spin, self.dft_pm_g_spin],
            "curie_weiss": [self.dft_pm_theta_spin, self.dft_pm_mueff_spin,
                            self.dft_pm_spin_spin, self.dft_pm_T_spin, self.dft_pm_g_spin]}
        form.addRow("φ:", self.dft_pm_phi_spin)
        form.addRow("χ_M:", self.dft_pm_chi_spin)
        form.addRow("χ_M units:", self.dft_pm_chi_units_combo)
        form.addRow("Weiss constant θ:", self.dft_pm_theta_spin)
        form.addRow("Effective moment μ_eff:", self.dft_pm_mueff_spin)
        form.addRow("Spin S:", self.dft_pm_spin_spin)
        form.addRow("Temperature T:", self.dft_pm_T_spin)
        form.addRow("Field B0:", self.dft_pm_B0_spin)
        form.addRow("", self.dft_pm_use_field_button)
        form.addRow("Electron g:", self.dft_pm_g_spin)
        form.addRow("Include:", self.dft_pm_components_combo)
        form.addRow("Orbital shift:", self.dft_pm_orbital_spin)
        v.addLayout(form)
        self.dft_pm_phi_label = QLabel("")
        self.dft_pm_phi_label.setWordWrap(True)
        self.dft_pm_phi_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        v.addWidget(self.dft_pm_phi_label)
        self.dft_pm_route_combo.currentIndexChanged.connect(lambda _i: self._on_pm_changed())
        for w in (self.dft_pm_phi_spin, self.dft_pm_chi_spin, self.dft_pm_theta_spin,
                  self.dft_pm_mueff_spin, self.dft_pm_spin_spin, self.dft_pm_T_spin,
                  self.dft_pm_B0_spin, self.dft_pm_g_spin, self.dft_pm_orbital_spin):
            w.editingFinished.connect(self._on_pm_changed)
        for w in (self.dft_pm_chi_units_combo, self.dft_pm_components_combo):
            w.currentIndexChanged.connect(lambda _i: self._on_pm_changed())
        self._sync_pm_rows()
        box.setVisible(False)
        return box

    # ---- ORCA g / ZFS / hyperfine (explicit user request, 2026-11-06) ----------------

    def _build_orca_pm_box(self) -> QGroupBox:
        """The magnetic model for an ORCA output with hyperfine tensors:
        σ = −(μ_B/g_Nμ_N) g·Z·A (notebook.paramagnetic_nmr)."""
        box = QGroupBox("Paramagnetic shift (ORCA g, zero-field splitting, hyperfine)")
        self.dft_opm_box = box
        v = QVBoxLayout(box)
        intro = QLabel(
            "This ORCA output has hyperfine tensors. The paramagnetic shielding is "
            "σ = −(μ_B/g_Nμ_N) g·Z·A, where Z is the electron spin's susceptibility. The "
            "calculated spin Hamiltonian (S, g, and D if there is one) gives what ORCA's "
            "orca_pnmr gives. A measured moment or susceptibility describes a coupled solid "
            "better than a single ion does. The orbital shift is added below, unless the "
            "file also has shieldings; then the shift scale (references) applies to them.")
        intro.setWordWrap(True)
        v.addWidget(intro)
        form = QFormLayout()
        self._opm_form = form
        self.dft_opm_model_combo = QComboBox()
        for key, text in (("curie", "Calculated spin Hamiltonian (S, g, D): as orca_pnmr"),
                          ("curie_weiss", "Curie–Weiss law (S, θ)"),
                          ("mu_eff", "Measured effective moment μ_eff (and θ)"),
                          ("susceptibility", "Measured susceptibility χ_M at T"),
                          ("phi", "Scaling factor φ = ⟨S_z⟩/S at B0")):
            self.dft_opm_model_combo.addItem(text, key)
            self.dft_opm_model_combo.setItemData(self.dft_opm_model_combo.count() - 1,
                                                 pmr.MODELS[key], Qt.ItemDataRole.ToolTipRole)
        form.addRow("Magnetic model:", self.dft_opm_model_combo)
        self.dft_opm_T_spin = _spin_box(0.1, 3000.0, 298.0, 2, " K", step=1.0,
                                        tip="Sample temperature.")
        self.dft_opm_S_spin = _spin_box(0.5, 10.0, 0.5, 1, step=0.5,
                                        tip="Electron spin S (from the multiplicity in the file; "
                                            "a pseudo-spin for a lanthanide).")
        self.dft_opm_zfs_check = QCheckBox("Use the zero-field splitting D from the file")
        self.dft_opm_zfs_check.setChecked(True)
        self.dft_opm_zfs_check.setToolTip(
            "With D, Z is anisotropic and has a temperature-independent part: a pseudocontact "
            "shift even with an isotropic g. Without D: the Curie law S(S+1)/3kT.")
        self.dft_opm_theta_spin = _spin_box(-1e4, 1e4, 0.0, 2, " K", step=1.0,
                                            tip="Weiss constant θ (negative: antiferromagnetic).")
        self.dft_opm_mueff_spin = _spin_box(0.0, 50.0, 5.92, 3, " μB", step=0.01,
                                            tip="Effective moment per paramagnetic centre.")
        self.dft_opm_chi_spin = _spin_box(0.0, 1e3, 0.0146, 6, step=0.0001,
                                          tip="Molar susceptibility per paramagnetic centre at T "
                                              "(paramagnetic part).")
        self.dft_opm_chi_units_combo = QComboBox()
        self.dft_opm_chi_units_combo.addItem("cm³/mol (cgs, emu/mol)", "cm3/mol")
        self.dft_opm_chi_units_combo.addItem("m³/mol (SI)", "m3/mol")
        self.dft_opm_phi_spin = _spin_box(-100.0, 100.0, 0.02, 6, step=0.001,
                                          tip="φ = ⟨S_z⟩/S at the field B0.")
        self.dft_opm_B0_spin = _spin_box(0.001, 100.0, 11.7, 4, " T", step=0.1,
                                         tip="The field φ belongs to.")
        self.dft_opm_components_combo = QComboBox()
        for key, text in (("both", "Fermi contact + spin dipolar"),
                          ("contact", "Fermi contact only"), ("dipolar", "Spin dipolar only")):
            self.dft_opm_components_combo.addItem(text, key)
        self.dft_opm_orbital_spin = _spin_box(-1e5, 1e5, 0.0, 3, " ppm", step=1.0,
                                              tip="Orbital (diamagnetic) shift added to every "
                                                  "site, e.g. that of a diamagnetic analogue.")
        self._opm_rows = {
            "curie": [self.dft_opm_T_spin, self.dft_opm_S_spin, self.dft_opm_zfs_check],
            "curie_weiss": [self.dft_opm_T_spin, self.dft_opm_S_spin, self.dft_opm_theta_spin],
            "mu_eff": [self.dft_opm_T_spin, self.dft_opm_S_spin, self.dft_opm_mueff_spin,
                       self.dft_opm_theta_spin],
            "susceptibility": [self.dft_opm_T_spin, self.dft_opm_S_spin, self.dft_opm_chi_spin,
                               self.dft_opm_chi_units_combo],
            "phi": [self.dft_opm_S_spin, self.dft_opm_phi_spin, self.dft_opm_B0_spin]}
        form.addRow("Temperature T:", self.dft_opm_T_spin)
        form.addRow("Spin S:", self.dft_opm_S_spin)
        form.addRow("", self.dft_opm_zfs_check)
        form.addRow("Weiss constant θ:", self.dft_opm_theta_spin)
        form.addRow("Effective moment μ_eff:", self.dft_opm_mueff_spin)
        form.addRow("χ_M:", self.dft_opm_chi_spin)
        form.addRow("χ_M units:", self.dft_opm_chi_units_combo)
        form.addRow("φ:", self.dft_opm_phi_spin)
        form.addRow("Field B0:", self.dft_opm_B0_spin)
        form.addRow("Include:", self.dft_opm_components_combo)
        form.addRow("Orbital shift:", self.dft_opm_orbital_spin)
        v.addLayout(form)
        self.dft_opm_label = QLabel("")
        self.dft_opm_label.setWordWrap(True)
        self.dft_opm_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        v.addWidget(self.dft_opm_label)
        row = QHBoxLayout()
        self.dft_opm_plot_button = QPushButton("Plot Contact / Pseudocontact")
        self.dft_opm_plot_button.setToolTip(
            "Bars of each site's Fermi contact and pseudocontact (spin dipolar) isotropic shift, "
            "on the Simulation plot.")
        self.dft_opm_plot_button.clicked.connect(lambda: self.plot_contributions())
        row.addWidget(self.dft_opm_plot_button)
        self.dft_opm_check_button = QPushButton("Check Against orca_pnmr Output…")
        self.dft_opm_check_button.setToolTip(
            "Compare openNMR's paramagnetic shielding tensors with those orca_pnmr printed "
            "for the same g, D and A tensors.")
        self.dft_opm_check_button.clicked.connect(lambda: self.check_against_orca())
        row.addWidget(self.dft_opm_check_button)
        row.addStretch(1)
        v.addLayout(row)
        self.dft_opm_model_combo.currentIndexChanged.connect(lambda _i: self._on_opm_changed())
        for w in (self.dft_opm_T_spin, self.dft_opm_S_spin, self.dft_opm_theta_spin,
                  self.dft_opm_mueff_spin, self.dft_opm_chi_spin, self.dft_opm_phi_spin,
                  self.dft_opm_B0_spin, self.dft_opm_orbital_spin):
            w.editingFinished.connect(self._on_opm_changed)
        for w in (self.dft_opm_chi_units_combo, self.dft_opm_components_combo):
            w.currentIndexChanged.connect(lambda _i: self._on_opm_changed())
        self.dft_opm_zfs_check.toggled.connect(lambda _c: self._on_opm_changed())
        self._sync_opm_rows()
        box.setVisible(False)
        return box

    def paramagnetic_source(self) -> dict | None:
        """The paramagnetic calculation and its magnetic model as set here, for
        Simulate ▸ Paramagnetic NMR tools (shift vs T): ``{"kind", "calc",
        "path", "args", "element", "labels", "orbital"}`` -- ``kind`` "orca"
        (an ORCA output with hyperfine tensors; ``args`` its model) or
        "crystal" (a CRYSTAL calculation; ``args`` its φ / χ_M / Curie–Weiss
        scaling, explicit user request 2026-11-07) -- or None for any other
        sample."""
        if not (self._is_orca_epr() or self._is_crystal()):
            return None
        try:
            labels = self.checked_labels()
        except (ValueError, AttributeError):
            labels = []
        common = {"calc": self.calcs["sample"], "path": self.dft_sample_edit.text().strip(),
                  "structure": self.calcs["sample"].get("structure_path"),
                  "element": self.dft_element_combo.currentText(), "labels": labels}
        if self._is_crystal():
            return {**common, "kind": "crystal", "args": self.paramagnetic_arguments(),
                    "orbital": self.dft_pm_orbital_spin.value()}
        return {**common, "kind": "orca", "args": self.orca_paramagnetic_arguments(),
                "orbital": 0.0 if self._orca_orbital_in_file()
                else self.dft_opm_orbital_spin.value()}

    def _is_orca_epr(self) -> bool:
        sample = self.calcs.get("sample")
        return sample is not None and bool(sample.get("epr"))

    def _sync_opm_rows(self) -> None:
        model = self.dft_opm_model_combo.currentData()
        visible = {id(w) for w in self._opm_rows[model]}
        for ws in self._opm_rows.values():
            for w in ws:
                self._opm_form.setRowVisible(w, id(w) in visible)
        sample = self.calcs.get("sample") if hasattr(self, "calcs") else None
        has_D = bool(sample and (sample.get("epr") or {}).get("D_cm") is not None)
        self.dft_opm_zfs_check.setEnabled(has_D)
        orbital = self._orca_orbital_in_file()
        self._opm_form.setRowVisible(self.dft_opm_orbital_spin, not orbital)

    def _orca_orbital_in_file(self) -> bool:
        sample = self.calcs.get("sample") if hasattr(self, "calcs") else None
        return bool(sample and sample.get("epr") and any(
            a.get("sigma") is not None for a in sample["atoms"] if a.get("hfc")))

    def orca_paramagnetic_arguments(self) -> dict:
        """``paramagnetic=`` for :func:`dft_nmr.dft_sites` (an ORCA EPR file)."""
        model = self.dft_opm_model_combo.currentData()
        args = {"model": model, "S": self.dft_opm_S_spin.value(),
                "components": self.dft_opm_components_combo.currentData()}
        if model != "phi":
            args["T"] = self.dft_opm_T_spin.value()
        if model == "curie" and not (self.dft_opm_zfs_check.isChecked()
                                     and self.dft_opm_zfs_check.isEnabled()):
            args["D_cm"] = None
        if model in ("curie_weiss", "mu_eff"):
            args["theta"] = self.dft_opm_theta_spin.value()
        if model == "mu_eff":
            args["mu_eff"] = self.dft_opm_mueff_spin.value()
        if model == "susceptibility":
            args.update(chi=self.dft_opm_chi_spin.value(),
                        chi_units=self.dft_opm_chi_units_combo.currentData())
        if model == "phi":
            args.update(phi=self.dft_opm_phi_spin.value(), B0=self.dft_opm_B0_spin.value())
        return args

    def _update_opm_label(self) -> None:
        sample = self.calcs.get("sample")
        if not sample or not sample.get("epr"):
            self.dft_opm_label.setText("")
            return
        epr = sample["epr"]
        bits = []
        if epr.get("g") is not None:
            w = np.linalg.svd(np.asarray(epr["g"]), compute_uv=False)
            bits.append("g principal values " + ", ".join(f"{x:.5f}" for x in w))
        if epr.get("D_cm") is not None:
            de = pmr.D_E_from_tensor(epr["D_cm"])
            bits.append(f"D = {de['D']:.4g} cm⁻¹, E/D = {de['E_over_D']:.3f}")
        try:
            args = {k: v for k, v in self.orca_paramagnetic_arguments().items()
                    if k not in ("components", "model", "S")}
            g = epr.get("g")
            D = args.pop("D_cm", "file")
            D = epr.get("D_cm") if isinstance(D, str) else D
            model = self.dft_opm_model_combo.currentData()
            resp = pmr.spin_response(self.dft_opm_S_spin.value(), args.pop("T", 298.0), model,
                                     g=g, D_cm=D, **args)
            bits.append(resp["text"])
        except ValueError as exc:
            bits.append(f"Z cannot be found: {exc}")
        self.dft_opm_label.setText("; ".join(bits) + ".")

    def _on_opm_changed(self) -> None:
        self._sync_opm_rows()
        self._update_opm_label()
        if self.model is not None and self._is_orca_epr():
            self.calculate()

    def plot_contributions(self) -> bool:
        if self._draw_plot is None:
            return False
        rows = pmr.contribution_rows(self.model) if self.model is not None else []
        if not rows:
            QMessageBox.warning(self, "Contributions", "Calculate the sites of a paramagnetic "
                                "(ORCA hyperfine) calculation first.")
            return False

        def draw(figure):
            look = self._style() or {}
            pmr.plot_contributions(rows, figure=figure,
                                   fontsize=float(look.get("labelsize") or 11.0))
        self._draw_plot(draw, "curve")
        return True

    def check_against_orca(self, path=None) -> str | None:
        """Compare with an orca_pnmr output; the result goes in the notes."""
        if not self._is_orca_epr():
            return None
        printed = self.calcs["sample"].get("orca_pnmr")
        if path is None and printed:
            # An orca_pnmr output as the sample: compare with what it printed.
            path = self.dft_sample_edit.text().strip()
        elif path is None:
            path, _ = QFileDialog.getOpenFileName(self, "orca_pnmr output", "",
                                                  "Text (*.txt *.out *.log);;All files (*)")
            if not path:
                return None
        try:
            args = {k: v for k, v in self.orca_paramagnetic_arguments().items()
                    if k != "components"}
            if args.get("model") != "curie":
                raise ValueError("orca_pnmr uses the calculated spin Hamiltonian: choose that "
                                 "model to compare.")
            calc = pmr.apply_paramagnetic(self.calcs["sample"], **args)
            res = pmr.compare_with_orca(
                calc, printed if printed and path == self.dft_sample_edit.text().strip()
                else path)
        except (OSError, ValueError, KeyError) as exc:
            QMessageBox.warning(self, "Check against orca_pnmr", str(exc))
            return None
        limit = ("orca_pnmr prints its input's A to 6 decimals" if printed else
                 "ORCA's .out prints A to 4 decimals")
        text = (f"Against orca_pnmr ({os.path.basename(path)}, {res['T']:g} K, {res['n']} "
                f"nuclei): isotropic shielding within {res['max_iso_ppm']:.3g} ppm, every "
                f"tensor element within {res['max_element_ppm']:.3g} ppm ({limit}, which "
                "limits the agreement).")
        self.dft_opm_label.setText(self.dft_opm_label.text() + "\n" + text)
        return text

    def _sync_pm_rows(self) -> None:
        route = self.dft_pm_route_combo.currentData()
        visible = {id(w) for w in self._pm_rows[route]}
        for ws in self._pm_rows.values():
            for w in ws:
                self._pm_form.setRowVisible(w, id(w) in visible)
        self._pm_form.setRowVisible(self.dft_pm_use_field_button, True)

    def paramagnetic_arguments(self) -> dict:
        """``paramagnetic=`` for :func:`dft_nmr.dft_sites` from the controls."""
        route = self.dft_pm_route_combo.currentData()
        args = {"B0": self.dft_pm_B0_spin.value(),
                "components": self.dft_pm_components_combo.currentData()}
        if route == "phi":
            args["phi"] = self.dft_pm_phi_spin.value()
        elif route == "susceptibility":
            args.update(chi=self.dft_pm_chi_spin.value(),
                        chi_units=self.dft_pm_chi_units_combo.currentData(),
                        spin=self.dft_pm_spin_spin.value(), g=self.dft_pm_g_spin.value())
        else:
            args.update(theta=self.dft_pm_theta_spin.value(), mu_eff=self.dft_pm_mueff_spin.value(),
                        spin=self.dft_pm_spin_spin.value(), T=self.dft_pm_T_spin.value(),
                        g=self.dft_pm_g_spin.value())
        return args

    def _on_pm_changed(self) -> None:
        self._sync_pm_rows()
        self._update_pm_label()
        if self.model is not None and self._is_crystal():
            self.calculate()

    def _update_pm_label(self) -> None:
        try:
            args = {k: v for k, v in self.paramagnetic_arguments().items() if k != "components"}
            self.dft_pm_phi_label.setText(crystal_nmr.magnetic_scaling(**args)["text"])
        except ValueError as exc:
            self.dft_pm_phi_label.setText(f"φ cannot be found: {exc}")

    def _pm_field_from_larmor(self) -> None:
        from ..notebook import dnp
        try:
            nucleus = self.nucleus_combo.currentText().strip()
            self.dft_pm_B0_spin.setValue(float(dnp.field_from_frequency(
                self.larmor_spin.value() * 1e6, nucleus)))
        except (ValueError, KeyError, TypeError) as exc:
            QMessageBox.warning(self, "Field", f"Could not get B0 from the Larmor frequency: {exc}")
            return
        self._on_pm_changed()

    def _is_crystal(self) -> bool:
        sample = self.calcs.get("sample")
        return sample is not None and sample.get("code") == "CRYSTAL"

    def _build_state_box(self) -> QGroupBox:
        """Solid or solution, and which sites a fast motion averages."""
        box = QWidget()
        v = QVBoxLayout(box)
        v.setContentsMargins(0, 0, 0, 0)
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
            "names in the Sites list of step 1, or rows of the table of sites in step 5 (click, "
            "Ctrl-click or Shift-click) – and press Average Selected Sites. Their tensors are "
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
                                           "Sites list of step 1 (or the table of step 5) into "
                                           "one group.")
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
        self.dft_reference_box = box
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
        return box

    def _build_calculate_box(self) -> QWidget:
        """Step 5's top: calculate the sites, simulate, and the table."""
        box = QWidget()
        v = QVBoxLayout(box)
        v.setContentsMargins(0, 0, 0, 0)
        self.dft_calculate_button = QPushButton("Calculate Sites and Simulate")
        self.dft_calculate_button.setToolTip(
            "Work out every chosen site's parameters (steps 1–3), put them in the sites below "
            "(replacing what is there) and simulate with the experiment of step 4.")
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

    def _build_tensor_box(self) -> QWidget:
        box = QWidget()
        form = QFormLayout(box)
        form.setContentsMargins(0, 0, 0, 0)
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
        # 0.8.4 (explicit user request): the tensors on the structure,
        # centred on their atoms -- notebook.dft_structure.
        self.dft_tensor_style_combo.addItem("On the structure", "structure")
        self.dft_tensor_style_combo.setToolTip(
            "On the structure: the cell (or molecule) with the tensors of the chosen site(s) "
            "centred on their atoms, in the same frame, so their orientation relative to the "
            "bonds is seen.")
        self.dft_tensor_style_combo.currentIndexChanged.connect(lambda _i: self._sync_dft())
        form.addRow("Show:", self.dft_tensor_style_combo)
        self.dft_relative_combo = QComboBox()
        self.dft_relative_combo.addItem("Anisotropic part (δ − δiso)", "iso")
        self.dft_relative_combo.addItem("Whole frequency (δ)", "zero")
        form.addRow("Frequency:", self.dft_relative_combo)
        self.dft_structure_sites_combo = QComboBox()
        self.dft_structure_sites_combo.addItem("This site", "one")
        self.dft_structure_sites_combo.addItem("Every site in the table", "all")
        self.dft_structure_sites_combo.setToolTip(
            "Whose tensors to draw. All of them share one length scale, so their sizes "
            "compare directly.")
        form.addRow("Tensors of:", self.dft_structure_sites_combo)
        self.dft_structure_tensor_combo = QComboBox()
        for key, text in (("shift_aniso", "Shift anisotropy δ − δiso (ovaloid)"),
                          ("shift", "Shift tensor δ (ellipsoid, |δii|)"),
                          ("efg", "EFG (ovaloid)"),
                          ("para_contact", "Paramagnetic: Fermi contact part (ovaloid)"),
                          ("para_dipolar", "Paramagnetic: spin-dipolar (pseudocontact) part "
                                           "(ovaloid)")):
            self.dft_structure_tensor_combo.addItem(text, key)
        self.dft_structure_tensor_combo.setToolTip(
            "Ovaloid: in each direction n the distance |nᵀTn|, red where positive and blue "
            "where negative – the only faithful glyph of a traceless tensor (the anisotropy, "
            "the EFG). Ellipsoid: semi-axes |δ11|, |δ22|, |δ33|. The paramagnetic parts (ORCA "
            "hyperfine or orca_pnmr files) split δ_para into its Fermi contact part (anisotropic "
            "when g or the susceptibility is) and its spin-dipolar part, whose trace is the "
            "pseudocontact shift.")
        form.addRow("Tensor:", self.dft_structure_tensor_combo)
        self.dft_structure_region_combo = QComboBox()
        self.dft_structure_region_combo.addItem("Unit cell (or whole molecule)", "cell")
        self.dft_structure_region_combo.addItem("Atoms within a radius of the site(s)", "radius")
        # 2026-10-27 (explicit user request): "show supercells (or change
        # the limits in terms of fractional cell coords -- e.g., x from
        # -0.2 to +1.6)".
        self.dft_structure_region_combo.addItem("Fractional range / supercell", "range")
        self.dft_structure_region_combo.currentIndexChanged.connect(lambda _i: self._sync_dft())
        form.addRow("Atoms shown:", self.dft_structure_region_combo)
        self.dft_structure_range_spins = {}
        range_rows = []
        for axis in "abc":
            row = QHBoxLayout()
            spins = []
            for end, value in (("from", 0.0), ("to", 1.0)):
                spin = _spin_box(-6.0, 7.0, value, 2, step=0.1,
                                 tip=f"Fractional coordinate along {axis}, {end}. Atoms on the "
                                     "ends are included (0 to 2 along a is two cells, with "
                                     "the copies on both faces).")
                row.addWidget(QLabel(end))
                row.addWidget(spin, 1)
                spins.append(spin)
            self.dft_structure_range_spins[axis] = tuple(spins)
            form.addRow(f"{axis} range:", row)
            range_rows.append(row)
        super_row = QHBoxLayout()
        super_row.addWidget(QLabel("Supercell"))
        self.dft_structure_supercell_spins = []
        for axis in "abc":
            spin = TypedSpinBox()
            spin.setRange(1, 12)
            spin.setValue(1)
            spin.setToolTip(f"Cells along {axis}: sets the {axis} range to 0 – n.")
            super_row.addWidget(spin)
            self.dft_structure_supercell_spins.append(spin)
        self.dft_structure_supercell_button = QPushButton("Set")
        self.dft_structure_supercell_button.setToolTip("Set the three ranges to 0 – n.")
        self.dft_structure_supercell_button.clicked.connect(self._set_supercell)
        super_row.addWidget(self.dft_structure_supercell_button)
        form.addRow("", super_row)
        self.dft_structure_copies_combo = QComboBox()
        self.dft_structure_copies_combo.addItem("Nearest the middle of the range", "one")
        self.dft_structure_copies_combo.addItem("Every copy in the range", "all")
        self.dft_structure_copies_combo.setToolTip(
            "Which periodic copies of a site carry its tensor. A lattice translation does not "
            "turn a tensor, so every copy's is the same.")
        form.addRow("Tensors on:", self.dft_structure_copies_combo)
        self.dft_structure_edges_combo = QComboBox()
        self.dft_structure_edges_combo.addItem("Every whole cell", "cells")
        self.dft_structure_edges_combo.addItem("The cell at the origin", "origin")
        self.dft_structure_edges_combo.addItem("Outline of the range", "range")
        self.dft_structure_edges_combo.addItem("None", "none")
        form.addRow("Cell edges:", self.dft_structure_edges_combo)
        self._range_rows = (*range_rows, super_row, self.dft_structure_copies_combo,
                            self.dft_structure_edges_combo)
        self.dft_structure_radius_spin = _spin_box(
            1.0, 30.0, 5.0, 2, " Å", step=0.5,
            tip="Every atom, periodic images included, within this distance of a chosen site.")
        form.addRow("Radius:", self.dft_structure_radius_spin)
        self.dft_structure_size_spin = _spin_box(
            0.1, 10.0, 1.0, 2, " Å", step=0.1,
            tip="Size of the largest tensor drawn (its largest extent); the rest to the same "
                "scale.")
        form.addRow("Tensor size:", self.dft_structure_size_spin)
        self.dft_structure_bonds_check = QCheckBox("Draw bonds")
        self.dft_structure_bonds_check.setChecked(True)
        self.dft_structure_bonds_check.setToolTip(
            "Between atoms closer than the tolerance × the sum of their covalent radii; never "
            "between two metals. A guide to the eye only.")
        self.dft_structure_bond_spin = _spin_box(
            0.8, 1.6, 1.15, 2, " ×", step=0.05, tip="Bond tolerance (× sum of covalent radii).")
        bond_row = QHBoxLayout()
        bond_row.addWidget(self.dft_structure_bonds_check)
        bond_row.addWidget(self.dft_structure_bond_spin)
        form.addRow("Bonds:", bond_row)
        self._structure_rows = (self.dft_structure_sites_combo, self.dft_structure_tensor_combo,
                                self.dft_structure_region_combo, self.dft_structure_radius_spin,
                                self.dft_structure_size_spin, bond_row)
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

    def _browse_structure(self) -> None:
        start = self.dft_structure_edit.text() or self.dft_sample_edit.text() or ""
        path, _ = QFileDialog.getOpenFileName(self, "Structure of the orca_pnmr file",
                                              os.path.dirname(start), _STRUCTURE_FILTER)
        if path:
            self.dft_structure_edit.setText(path)
            if self.dft_sample_edit.text().strip():
                self.read_files(quiet=False)

    def _is_orca_pnmr(self) -> bool:
        sample = self.calcs.get("sample") if hasattr(self, "calcs") else None
        return bool(sample and str(sample.get("code", "")).endswith("(orca_pnmr)"))

    def _read_sample(self, path):
        """The sample, with its structure when it is an orca_pnmr file."""
        structure = self.dft_structure_edit.text().strip() or None
        found = getattr(self, "_structure_found", None)      # (sample, structure)
        if structure is not None and found and found[1] == structure and found[0] != path:
            # Found next to another sample, not chosen: look again for this one.
            structure = None
            self.dft_structure_edit.setText("")
        if structure is not None and pmr.is_orca_pnmr(path) is None:
            structure = None                # only orca_pnmr files take one
        calc = dft_nmr.read_calculation(path, include_g0=self.dft_g0_check.isChecked(),
                                        structure=structure)
        # Show the structure that was found next to the file, so the choice is
        # visible (and saved with the state).
        if calc.get("structure_path") and not structure:
            self.dft_structure_edit.setText(calc["structure_path"])
            self._structure_found = (path, calc["structure_path"])
        return calc

    @staticmethod
    def _describe_calc(key, c) -> str:
        if c.get("code") == "CRYSTAL":
            return (f"{key}: {crystal_nmr.crystal_summary(c)} "
                    + dft_nmr.describe_method(c.get("method")) + ".")
        n_sigma = sum(a["sigma"] is not None for a in c["atoms"])
        n_efg = sum(a["efg"] is not None for a in c["atoms"])
        g0 = {"included": "; G=0 included", "excluded": "; G=0 excluded",
              "fixed": "; G=0 included (fixed in a .magres)"}.get(c.get("g0"), "")
        return (f"{key}: {c['code']}, {len(c['atoms'])} atoms ("
                + ", ".join(f"{s} {sum(a['species'] == s for a in c['atoms'])}"
                            for s in dft_nmr.species_of(c))
                + f"); shielding for {n_sigma}, EFG for {n_efg}"
                + ("" if c.get("lattice") is not None else "; no cell") + g0
                + ("" if pmr.has_positions(c) else "; no atomic positions (give its structure)")
                + ". "
                + dft_nmr.describe_method(c.get("method")) + ".")

    def read_files(self, quiet: bool = False) -> bool:
        """Read the sample (required), and every reference file again (with
        the current G=0 choice)."""
        try:
            sample = self.dft_sample_edit.text().strip()
            if not sample:
                raise ValueError("Choose the sample's calculation file.")
            self.calcs["sample"] = self._read_sample(sample)
            epr = self.calcs["sample"].get("epr") or {}
            # A new ORCA EPR file sets S from its multiplicity; the same file
            # read again keeps whatever S was chosen (a pseudo-spin, say).
            if epr.get("S") and sample != getattr(self, "_opm_S_from", None):
                self.dft_opm_S_spin.setValue(float(epr["S"]))
            self._opm_S_from = sample
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
            if atom["species"] != element or (atom["sigma"] is None and not self._is_crystal()
                                              and not atom.get("hfc")):
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
        on_structure = self.dft_tensor_style_combo.currentData() == "structure"
        for row in self._structure_rows:
            self._tensor_form.setRowVisible(row, on_structure)
        region = self.dft_structure_region_combo.currentData()
        for row in self._range_rows:
            self._tensor_form.setRowVisible(row, on_structure and region == "range")
        if on_structure:
            self._tensor_form.setRowVisible(
                self.dft_structure_radius_spin, region == "radius")
        sample = self.calcs["sample"]
        # The G=0 choice only concerns VASP: shown before a file is read
        # and for a VASP file, not for the others (a tidier step 1).
        vasp = sample is None or str(sample.get("code", "")).upper().startswith("VASP")
        self.dft_g0_check.setVisible(vasp)
        self.dft_g0_help.setVisible(vasp)
        self._import_form.setRowVisible(
            self._structure_file_row,
            self._is_orca_pnmr() or bool(self.dft_structure_edit.text().strip()))
        periodic = sample is not None and sample.get("lattice") is not None
        cell_item = self.dft_frame_combo.model().item(0)
        cell_item.setEnabled(periodic or sample is None)
        if sample is not None and not periodic:
            self.dft_frame_combo.setCurrentIndex(1)
        self.dft_show_tensor_button.setEnabled(self.model is not None)
        self.dft_pm_box.setVisible(self._is_crystal())
        self.dft_opm_box.setVisible(self._is_orca_epr())
        if self._is_orca_epr():
            self._sync_opm_rows()
            self._update_opm_label()
        if self._is_crystal():
            self._update_pm_label()
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
            if self._is_crystal():
                model = dft_nmr.dft_sites(self.calcs["sample"],
                                          paramagnetic=self.paramagnetic_arguments(),
                                          orbital_shift=self.dft_pm_orbital_spin.value(), **args)
            elif self._is_orca_epr() and not self._orca_orbital_in_file():
                model = dft_nmr.dft_sites(self.calcs["sample"],
                                          paramagnetic=self.orca_paramagnetic_arguments(),
                                          orbital_shift=self.dft_opm_orbital_spin.value(), **args)
            elif self._is_orca_epr():
                model = dft_nmr.dft_sites(self.calcs["sample"],
                                          paramagnetic=self.orca_paramagnetic_arguments(),
                                          calibration=self.current_calibration(), **args)
            else:
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
        if self._is_orca_epr() and not self._orca_orbital_in_file():
            return pmr.paramagnetic_to_python(
                self.dft_sample_edit.text().strip(), paramagnetic=self.orca_paramagnetic_arguments(),
                structure=(self.dft_structure_edit.text().strip() or None)
                if self._is_orca_pnmr() else None,
                orbital_shift=self.dft_opm_orbital_spin.value(),
                **{k: v for k, v in args.items() if k not in ("element",)}, element=args["element"])
        if self._is_crystal():
            kwargs = ", ".join(f"{k}={v!r}" for k, v in args.items() if v is not None)
            return ("from opennmr.notebook import dft_nmr\n"
                    f"sample = dft_nmr.read_calculation({self.dft_sample_edit.text().strip()!r})\n"
                    f"model = dft_nmr.dft_sites(sample, paramagnetic={self.paramagnetic_arguments()!r}, "
                    f"orbital_shift={self.dft_pm_orbital_spin.value()!r}, {kwargs})\n"
                    "print(dft_nmr.format_table(model))\n")
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

    def structure_arguments(self) -> dict:
        """The "On the structure" options, as
        :func:`opennmr.notebook.dft_structure.plot_dft_structure` takes them."""
        return {"frame": self.dft_frame_combo.currentData(),
                "tensor": self.dft_structure_tensor_combo.currentData(),
                "region": self.dft_structure_region_combo.currentData(),
                "radius": self.dft_structure_radius_spin.value(),
                "tensor_size": self.dft_structure_size_spin.value(),
                "bonds": self.dft_structure_bonds_check.isChecked(),
                "bond_tolerance": self.dft_structure_bond_spin.value(),
                "frac_range": self.frac_range(),
                "tensor_copies": self.dft_structure_copies_combo.currentData(),
                "cell_edges": self.dft_structure_edges_combo.currentData()}

    def frac_range(self) -> tuple:
        """The fractional range ((a0, a1), (b0, b1), (c0, c1)) typed."""
        return tuple((lo.value(), hi.value())
                     for lo, hi in (self.dft_structure_range_spins[k] for k in "abc"))

    def _set_supercell(self) -> None:
        for (lo, hi), n in zip((self.dft_structure_range_spins[k] for k in "abc"),
                               self.dft_structure_supercell_spins):
            lo.setValue(0.0)
            hi.setValue(float(n.value()))

    def show_structure(self) -> bool:
        """The structure with the chosen site's (or every site's) tensors."""
        try:
            records = ([self.tensor_record()]
                       if self.dft_structure_sites_combo.currentData() == "one"
                       else list(self.model["atoms"] if self.model else []))
            if not records:
                raise ValueError("Calculate the sites first.")
            sample = self.calcs["sample"]
            args = self.structure_arguments()
            holder = {}

            def draw(figure):
                look = self._style() or {}
                holder["notes"] = dft_structure.plot_dft_structure(
                    sample, records, figure=figure,
                    fontsize=float(look.get("labelsize") or 11.0),
                    tensor_style=dict(look.get("tensor_style") or {}), **args)[1]
            self._draw_tensors(draw)
        except (ValueError, KeyError, np.linalg.LinAlgError) as exc:
            self.dft_tensor_notes_label.setText(f"Cannot draw: {exc}")
            return False
        self.tensor_notes = holder.get("notes", [])
        self.dft_tensor_notes_label.setText("\n".join(self.tensor_notes))
        return True

    def show_tensors(self) -> bool:
        if self._draw_tensors is None:
            return False
        if self.dft_tensor_style_combo.currentData() == "structure":
            return self.show_structure()
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
        self._sync_pm_rows()
        self._opm_S_from = self.dft_sample_edit.text().strip()   # keep the saved S
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

