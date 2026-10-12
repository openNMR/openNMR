"""Simulate > CSA, Quad & J-coupling: sites with any mix of CSA,
quadrupolar and J interactions, static or MAS.

A list of sites on the left; the selected site's shift, amount and
broadening beneath it, and its interactions -- each added with "Add
interaction" and removed with its own button, so a combination is
built by adding, never chosen from a list of combinations. The physics
is :mod:`opennmr.notebook.combined_hamiltonian`; "Copy as Python" gives
the notebook code for the same simulation.
"""

from __future__ import annotations

import contextlib
import copy
import time

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout,
    QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QListWidget, QMenu, QMessageBox,
    QPushButton, QSpinBox, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from .fit_progress_dialog import FitProgress
from .widgets.advice_box import AdviceBox
from .widgets.sideband_hint import sideband_hint_label
from ..core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
from . import panel_state
from ..notebook.simulations import CSA_ANISOTROPY_TIP, CSA_ASYMMETRY_TIP
from ..io import serialize
from ..notebook import combined_fitting as cfit
from ..notebook import fit_advice
from ..notebook import start_estimates
from ..notebook import combined_hamiltonian as ch
from ..notebook.utils import _GAMMA_DF, find_isotope_info
from ..notebook import isotopes

_TITLES = {"csa": "Chemical-shift anisotropy (CSA)", "quadrupolar": "Quadrupolar",
           "jcoupling": "J-coupling"}


def _spin_box(lo, hi, value, decimals, suffix="", step=None, tip=""):
    box = QDoubleSpinBox()
    box.setRange(lo, hi)
    box.setDecimals(decimals)
    box.setValue(value)
    if suffix:
        box.setSuffix(suffix)
    if step:
        box.setSingleStep(step)
    if tip:
        box.setToolTip(tip)
    return box


def _nuclides():
    names = []
    for nuc, spin, ab in zip(_GAMMA_DF["Nuclide"], _GAMMA_DF["Spin"],
                             _GAMMA_DF["Natural Abundance (%)"]):
        try:
            if float(spin) > 0 and float(ab) > 0:
                names.append(str(nuc))
        except (TypeError, ValueError):
            continue
    return names


class InteractionEditor(QGroupBox):
    """One interaction of one site. Edits write straight into ``data``
    (the interaction's dict in the site's list)."""

    def __init__(self, data: dict, on_remove, has_quadrupolar: bool, parent=None,
                 panel=None) -> None:
        super().__init__(_TITLES[data["type"]], parent)
        self.data = data
        self.theory_combo = None
        self.treatment_combo = None
        outer = QVBoxLayout(self)
        head = QHBoxLayout()
        head.addStretch(1)
        self.remove_button = QPushButton("Remove")
        self.remove_button.setToolTip("Remove this interaction from the site.")
        self.remove_button.clicked.connect(lambda: on_remove(self.data))
        head.addWidget(self.remove_button)
        outer.addLayout(head)
        form = QFormLayout()
        outer.addLayout(form)
        kind = data["type"]
        if kind == "csa":
            self.convention = QComboBox()
            self.convention.addItem("Haeberlen (Δ, η)", "Haeberlen")
            self.convention.addItem("Herzfeld-Berger (span, skew)", "Herzfeld-Berger")
            self.convention.setCurrentIndex(self.convention.findData(data["convention"]))
            self.convention.currentIndexChanged.connect(self._write_csa)
            form.addRow("Convention:", self.convention)
            self.aniso = _spin_box(-1e5, 1e5, data["aniso"], 3, " ppm",
                                   tip=CSA_ANISOTROPY_TIP)
            self.asym = _spin_box(-1.0, 1.0, data["asym"], 3, step=0.1,
                                  tip=CSA_ASYMMETRY_TIP)
            form.addRow("Δ or span:", self.aniso)
            form.addRow("η or skew:", self.asym)
            euler_box = QWidget()
            row = QHBoxLayout(euler_box)
            row.setContentsMargins(0, 0, 0, 0)
            self.euler = [_spin_box(-360.0, 360.0, v, 2, "°") for v in data["euler"]]
            for name, box in zip("αβγ", self.euler):
                row.addWidget(QLabel(name))
                row.addWidget(box)
                box.valueChanged.connect(self._write_csa)
            euler_box.setEnabled(has_quadrupolar)
            euler_box.setToolTip(
                "Orientation of the CSA tensor in the EFG frame (ZYZ, degrees): β and α are the "
                "polar angles of the CSA's unique axis (δzz) in the EFG frame; γ turns the CSA "
                "about it. All zero: coincident tensors. Only meaningful with a quadrupolar "
                "interaction on the same site.")
            form.addRow("Euler angles:", euler_box)
            for w in (self.aniso, self.asym):
                w.valueChanged.connect(self._write_csa)
        elif kind == "quadrupolar":
            self.cq = _spin_box(-1e4, 1e4, data["C_Q"], 4, " MHz",
                                tip="C_Q = e²qQ/h. Its sign does not change a lineshape.")
            self.eta_q = _spin_box(0.0, 1.0, data["eta_Q"], 3, step=0.1)
            form.addRow("C_Q:", self.cq)
            form.addRow("η_Q:", self.eta_q)
            if panel is not None:
                # Every level of theory, where the interaction is (explicit
                # user request: the full suite of each interaction reachable
                # from Combined). One setting for the whole spectrum: it
                # mirrors the panel's "Quadrupolar theory".
                self.theory_combo = QComboBox()
                for i in range(panel.order_combo.count()):
                    self.theory_combo.addItem(panel.order_combo.itemText(i),
                                              panel.order_combo.itemData(i))
                self.theory_combo.setCurrentIndex(panel.order_combo.currentIndex())
                self.theory_combo.setToolTip(
                    "First order, second-order perturbation, or exact diagonalisation of the "
                    "Zeeman + quadrupolar Hamiltonian. Applies to every quadrupolar peak (the "
                    "same setting as Quadrupolar theory above).")
                self.theory_combo.currentIndexChanged.connect(
                    lambda i: panel.order_combo.setCurrentIndex(i))
                form.addRow("Theory:", self.theory_combo)
            self.cq.valueChanged.connect(lambda v: data.__setitem__("C_Q", v))
            self.eta_q.valueChanged.connect(lambda v: data.__setitem__("eta_Q", v))
        else:
            self.partner = QComboBox()
            self.partner.setEditable(True)
            self.partner.addItems(_nuclides())
            self.partner.setCurrentText(data["nuclide"])
            self.partner.setToolTip("Coupling partner. Its spin and natural abundance come from "
                                    "the nuclide table (a low-abundance partner gives satellites).")
            self.partner.currentTextChanged.connect(self._write_partner)
            self.n = QSpinBox()
            self.n.setRange(1, 20)
            self.n.setValue(int(data["n"]))
            self.n.setToolTip("Number of equivalent partner nuclei with this J.")
            self.n.valueChanged.connect(lambda v: data.__setitem__("n", int(v)))
            self.j = _spin_box(-1e5, 1e5, data["J_Hz"], 3, " Hz")
            self.j.valueChanged.connect(lambda v: data.__setitem__("J_Hz", v))
            form.addRow("Partner:", self.partner)
            form.addRow("Number n:", self.n)
            form.addRow("J:", self.j)
            if panel is not None and panel._strong_coupling is not None:
                # First order here; strongly coupled peaks (J comparable to
                # their shift difference) go to the exact spin-system
                # solver, which this site model cannot express.
                self.treatment_combo = QComboBox()
                self.treatment_combo.addItem("First order (weak coupling)", "first")
                self.treatment_combo.addItem("Strong coupling: exact, between peaks…", "strong")
                self.treatment_combo.setToolTip(
                    "First order: this peak split by n partner nuclei (any isotope, with its "
                    "natural abundance) -- valid when J is much smaller than the shift "
                    "difference to the partner.\n\nStrong coupling: J between observed peaks, "
                    "solved exactly (roofing, AB, ABX, AA'BB'). Opens the spin-system model "
                    "with these peaks as its spins.")
                self.treatment_combo.activated.connect(
                    lambda i: self._on_treatment(i, panel))
                form.addRow("Treatment:", self.treatment_combo)

    def _write_csa(self, *_):
        self.data.update(convention=self.convention.currentData(), aniso=self.aniso.value(),
                         asym=self.asym.value(), euler=tuple(b.value() for b in self.euler))

    def _on_treatment(self, index, panel):
        if self.treatment_combo.itemData(index) == "strong":
            self.treatment_combo.setCurrentIndex(0)   # this interaction stays first order
            panel.request_strong_coupling()

    def _write_partner(self, text):
        try:
            find_isotope_info(text.strip())
        except KeyError:
            return  # mid-typing; keep the last valid partner
        self.data["nuclide"] = text.strip()


class CombinedHamiltonianPanel(QWidget):
    """``plot_spectrum(spectrum)`` draws the total; ``add_to_open(spectrum,
    name)`` exports it; ``draw_sites(result, nucleus)`` (optional) draws
    each site separately on the Simulation plot."""

    BLURB = ("Build each site from its interactions: add a CSA, a quadrupolar coupling and any "
             "number of J-couplings to the same site, in any combination. Static or MAS; every "
             "site is simulated with all of its interactions together and the sites summed.")

    #: Sites may be of another nucleus than the panel's, and isotope
    #: partners may be linked (the multi-spectrum editor sets False: there
    #: the nucleus belongs to the spectrum).
    OTHER_NUCLEI = True

    DEFAULT_INTERACTIONS = {
        "csa": {"type": "csa", "aniso": 100.0, "asym": 0.5, "convention": "Haeberlen",
                "euler": (0.0, 0.0, 0.0)},
        "quadrupolar": {"type": "quadrupolar", "C_Q": 3.0, "eta_Q": 0.5},
        "jcoupling": {"type": "jcoupling", "nuclide": "1H", "n": 1, "J_Hz": 100.0},
    }

    def __init__(self, plot_spectrum, add_to_open, draw_sites=None, parent=None,
                 send_to_fit=None, strong_coupling=None, draw_orders=None) -> None:
        super().__init__(parent)
        #: ``strong_coupling(site_dicts, nucleus, larmor_MHz)``: open the
        #: exact spin-system model seeded with these peaks (None: not offered).
        self._strong_coupling = strong_coupling
        #: ``draw_orders(comparison, nucleus, larmor_MHz)``: first vs second
        #: order vs exact, for "Compare levels of theory".
        self._draw_orders = draw_orders
        self._plot_spectrum = plot_spectrum
        self._add_to_open = add_to_open
        self._send_to_fit = send_to_fit
        self._draw_sites = draw_sites
        self._seeded_for = None
        #: > 0 while code changes many sites at once (see bulk()).
        self._bulk = 0
        self.site_dicts: list[dict] = []
        self._current = -1
        self.result: dict | None = None
        self.spectrum: NMRSpectrum | None = None

        v = QVBoxLayout(self)
        blurb = QLabel(self.BLURB)
        blurb.setWordWrap(True)
        v.addWidget(blurb)

        form = QFormLayout()
        self.nucleus_combo = QComboBox()
        self.nucleus_combo.setEditable(True)
        self.nucleus_combo.addItems(_nuclides())
        self.nucleus_combo.setCurrentText("27Al")
        self.nucleus_combo.setToolTip("The observed nucleus, shared by every site (one spectrum). "
                                      "Seeded from the selected spectrum.")
        self.nucleus_combo.currentTextChanged.connect(lambda _t: self._on_panel_nucleus())
        form.addRow("Nucleus:", self.nucleus_combo)
        self.spin_label = QLabel("—")
        form.addRow("Spin I:", self.spin_label)
        self.larmor_spin = _spin_box(0.0, 5000.0, 130.3, 4, " MHz",
                                     tip="Observe frequency (the spectrum's SFO1).")
        form.addRow("Larmor frequency:", self.larmor_spin)
        self.regime_combo = QComboBox()
        self.regime_combo.addItem("Static powder", "static")
        self.regime_combo.addItem("MAS (centreband + sidebands)", "mas")
        self.regime_combo.currentIndexChanged.connect(lambda _i: self._sync_controls())
        form.addRow("Regime:", self.regime_combo)
        self.mas_spin = _spin_box(0.0, 200000.0, 10000.0, 0, " Hz", step=1000.0,
                                  tip="Spinning rate at the magic angle.")
        form.addRow("MAS rate:", self.mas_spin)
        self.sideband_hint = sideband_hint_label()
        form.addRow(self.sideband_hint)
        self.transitions_combo = QComboBox()
        self.transitions_combo.addItem("All transitions", "all")
        self.transitions_combo.addItem("Central transition only", "ct")
        self.transitions_combo.addItem("Satellites only", "st")
        self.transitions_combo.setToolTip(
            "Applied to every site. A spin-1/2 has one transition; for a quadrupolar nucleus the "
            "intensities keep their true shares (CT = 9/35 for I = 5/2).")
        form.addRow("Transitions:", self.transitions_combo)
        self.order_combo = QComboBox()
        self.order_combo.addItem("Second-order perturbation", "second")
        self.order_combo.addItem("Exact (diagonalization)", "exact")
        self.order_combo.addItem("First-order only", "first")
        self.order_combo.setToolTip("Level of theory for the quadrupolar part. CSA and J are "
                                    "first order, which is exact to ~1e-6 of their size.")
        self.order_combo.currentIndexChanged.connect(self._sync_theory_combos)
        form.addRow("Quadrupolar theory:", self.order_combo)
        self.orientations_spin = QSpinBox()
        self.orientations_spin.setRange(0, 500000)
        self.orientations_spin.setSingleStep(2000)
        self.orientations_spin.setSpecialValueText("auto")
        self.orientations_spin.setValue(0)
        self.orientations_spin.setToolTip(
            "Powder orientation density (auto: 75,025 static, 4,181 MAS). One octant is used "
            "when that is exact; a CSA not aligned with the EFG needs the full sphere.")
        form.addRow("Orientations:", self.orientations_spin)
        v.addLayout(form)
        self._setup_form = form

        # ---- sites ----
        sites_box = QGroupBox("Peaks (sites)")
        sites_layout = QHBoxLayout(sites_box)
        self.site_list = QListWidget()
        self.site_list.setMaximumHeight(110)
        # Several peaks can be selected (Ctrl/Shift-click, Ctrl+A) to give
        # them the current peak's interactions at once (explicit user
        # request, 2026-10-12: 1200 enumerated pNMR sites, one CSA).
        self.site_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.site_list.currentRowChanged.connect(self._on_site_selected)
        self.site_list.itemSelectionChanged.connect(self._sync_copy_controls)
        left = QVBoxLayout()
        left.addWidget(self.site_list, 1)
        self.selection_label = QLabel("")
        self.selection_label.setWordWrap(True)
        left.addWidget(self.selection_label)
        sites_layout.addLayout(left, 1)
        buttons = QVBoxLayout()
        self.add_site_button = QPushButton("Add peak")
        self.add_site_button.setToolTip("Add a peak (a site): an isotropic line, to which "
                                        "interactions are added below.")
        self.add_site_button.clicked.connect(lambda: self.add_site())
        self.duplicate_button = QPushButton("Duplicate")
        self.duplicate_button.clicked.connect(self.duplicate_site)
        self.remove_site_button = QPushButton("Remove")
        self.remove_site_button.clicked.connect(self.remove_site)
        self.add_partner_button = QPushButton("Add isotope partner")
        self.add_partner_button.setToolTip(
            "Add the same site as observed through another isotope of its element (49Ti for "
            "47Ti, 65Cu for 63Cu, ...), linked to it: same shift, CSA, η_Q and broadening; C_Q "
            "scaled by the quadrupole moments; amount by natural abundance and receptivity. "
            "It is placed on this spectrum by the IUPAC frequency ratios.")
        self.add_partner_button.clicked.connect(lambda: self.add_isotope_partner())
        self.add_partner_button.setVisible(self.OTHER_NUCLEI)
        self.copy_button = QPushButton("Copy to Peaks ▾")
        self.copy_button.setToolTip(
            "Give other peaks this peak's interactions (CSA, quadrupolar, J) -- the selected "
            "peaks (Ctrl/Shift-click, Ctrl+A for all) or every peak. Their own interactions are "
            "replaced; δiso and amount are kept.")
        self._copy_menu = QMenu(self)
        for text, targets, broadening in (
                ("Interactions → selected peaks", "selected", False),
                ("Interactions → all peaks", "all", False),
                ("Interactions and broadening → selected peaks", "selected", True),
                ("Interactions and broadening → all peaks", "all", True)):
            action = self._copy_menu.addAction(text)
            action.triggered.connect(lambda _c=False, t=targets, br=broadening:
                                     self.copy_interactions(t, broadening=br))
        self._copy_menu.addSeparator()
        clear = self._copy_menu.addAction("Remove all interactions from the selected peaks")
        clear.triggered.connect(lambda _c=False: self.clear_interactions("selected"))
        self._copy_menu.aboutToShow.connect(self._name_copy_actions)
        self._copy_actions = [a for a in self._copy_menu.actions() if not a.isSeparator()]
        self.copy_button.setMenu(self._copy_menu)
        for b in (self.add_site_button, self.duplicate_button, self.remove_site_button,
                  self.add_partner_button, self.copy_button):
            buttons.addWidget(b)
        buttons.addStretch(1)
        sites_layout.addLayout(buttons)
        v.addWidget(sites_box)

        # ---- the selected site ----
        self.site_box = QGroupBox("Selected peak")
        site_layout = QVBoxLayout(self.site_box)
        sform = QFormLayout()
        self.label_edit = QLineEdit()
        self.label_edit.textEdited.connect(self._on_label_edited)
        sform.addRow("Label:", self.label_edit)
        self.site_nucleus_combo = QComboBox()
        self.site_nucleus_combo.setToolTip(
            "The nucleus this peak belongs to. Normally the observed one; another nucleus (an "
            "isotope of the same element, or a background signal such as 63Cu from the coil on "
            "a 23Na spectrum) is simulated on its own ppm scale and placed on this spectrum by "
            "the IUPAC frequency ratios. Its δiso is then on its own nucleus's scale.")
        self.site_nucleus_combo.currentIndexChanged.connect(self._on_site_nucleus)
        sform.addRow("Nucleus:", self.site_nucleus_combo)
        self.link_combo = QComboBox()
        self.link_combo.setToolTip(
            "Share another isotope's site's chemistry: this peak then has no parameters of its "
            "own (except those ticked under Not shared) -- they follow that site, so the two "
            "isotopes are fitted as one. '(own parameters)' keeps them independent.")
        self.link_combo.currentIndexChanged.connect(self._on_link)
        sform.addRow("Same chemistry as:", self.link_combo)
        own = QWidget()
        own_row = QHBoxLayout(own)
        own_row.setContentsMargins(0, 0, 0, 0)
        self.own_amount_check = QCheckBox("Amount")
        self.own_amount_check.setToolTip(
            "Fit this isotope's amount separately -- for a selective (long or weak) pulse, which "
            "excites the two isotopes' central transitions differently, or an isotopically "
            "enriched sample. Otherwise it is the parent's × abundance × receptivity.")
        self.own_fwhm_check = QCheckBox("Broadening")
        self.own_fwhm_check.setToolTip(
            "Fit this isotope's broadening separately -- when a distribution of quadrupolar "
            "couplings, rather than of shifts, dominates the width.")
        for box, name in ((self.own_amount_check, "amount"), (self.own_fwhm_check, "fwhm")):
            box.toggled.connect(lambda on, n=name: self._on_own(n, on))
            own_row.addWidget(box)
        own_row.addStretch(1)
        self.own_widget = own
        sform.addRow("Not shared:", own)
        self.diso_spin = _spin_box(-1e5, 1e5, 0.0, 3, " ppm", tip="Isotropic chemical shift.")
        self.diso_spin.valueChanged.connect(lambda x: self._set("delta_iso", x))
        sform.addRow("δiso:", self.diso_spin)
        self.amount_spin = _spin_box(0.0, 1e6, 1.0, 3, step=0.1,
                                     tip="Relative number of nuclei (integrated intensity).")
        self.amount_spin.valueChanged.connect(lambda x: self._set("amount", x))
        sform.addRow("Amount:", self.amount_spin)
        self.fwhm_spin = _spin_box(0.0, 1e4, 1.0, 3, " ppm", tip="Broadening FWHM.")
        self.fwhm_spin.valueChanged.connect(lambda x: self._set("fwhm", x))
        sform.addRow("Broadening FWHM:", self.fwhm_spin)
        self.lineshape_combo = QComboBox()
        self.lineshape_combo.addItem("Gaussian", "gaussian")
        self.lineshape_combo.addItem("Lorentzian", "lorentzian")
        self.lineshape_combo.addItem("Pseudo-Voigt", "pseudo_voigt")
        self.lineshape_combo.currentIndexChanged.connect(self._on_lineshape)
        sform.addRow("Broadening:", self.lineshape_combo)
        self.pv_eta_spin = _spin_box(0.0, 1.0, 0.5, 2, step=0.1,
                                     tip="Lorentzian fraction of the pseudo-Voigt broadening.")
        self.pv_eta_spin.valueChanged.connect(lambda x: self._set("eta", x))
        sform.addRow("L fraction:", self.pv_eta_spin)
        self._sform = sform
        site_layout.addLayout(sform)
        self.link_note = QLabel("")
        self.link_note.setWordWrap(True)
        self.link_note.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        site_layout.addWidget(self.link_note)
        self.interactions_layout = QVBoxLayout()
        site_layout.addLayout(self.interactions_layout)
        self.no_interactions_label = QLabel("No interactions yet: an isotropic peak. Add one "
                                            "below.")
        self.no_interactions_label.setWordWrap(True)
        site_layout.addWidget(self.no_interactions_label)
        self.add_interaction_button = QPushButton("Add interaction ▾")
        self.add_interaction_button.setToolTip(
            "Add a CSA, a quadrupolar coupling (first order, second order or exact) or a "
            "J-coupling (first order, or strong coupling between peaks, exact) to this peak.")
        self._interaction_menu = QMenu(self)
        self._actions = {}
        for kind in ch.INTERACTION_TYPES:
            action = self._interaction_menu.addAction(_TITLES[kind])
            action.triggered.connect(lambda _c=False, k=kind: self.add_interaction(k))
            self._actions[kind] = action
        if self._strong_coupling is not None:
            action = self._interaction_menu.addAction(
                "J-coupling between peaks (strong coupling, exact)…")
            action.setToolTip("Opens the exact spin-system model, with these peaks as its spins.")
            action.triggered.connect(lambda _c=False: self.request_strong_coupling())
            self._actions["strong"] = action
        self._interaction_menu.aboutToShow.connect(self._sync_menu)
        self.add_interaction_button.setMenu(self._interaction_menu)
        site_layout.addWidget(self.add_interaction_button)
        v.addWidget(self.site_box)

        self._build_actions(v)
        v.addStretch(1)

        self._sync_spin()
        self._sync_controls()
        self.add_site()

    def _build_actions(self, v) -> None:
        """The part below the site editor: for Simulate, the axis and
        Simulate/Export; Fit > CSA, Quad & J-coupling replaces it."""
        aform = QFormLayout()
        self.auto_axis_check = QCheckBox("Auto: frame every site")
        self.auto_axis_check.setChecked(True)
        self.auto_axis_check.setToolTip("Set the axis from every site's full range when "
                                        "Simulate is pressed. Typing a limit turns this off.")
        aform.addRow("Axis:", self.auto_axis_check)
        self.ppm_min_spin = _spin_box(-1e6, 1e6, -200.0, 2, " ppm")
        self.ppm_max_spin = _spin_box(-1e6, 1e6, 200.0, 2, " ppm")
        for box in (self.ppm_min_spin, self.ppm_max_spin):
            box.valueChanged.connect(lambda _v: self.auto_axis_check.setChecked(False))
        aform.addRow("Axis min:", self.ppm_min_spin)
        aform.addRow("Axis max:", self.ppm_max_spin)
        self.points_spin = QSpinBox()
        self.points_spin.setRange(64, 1 << 20)
        self.points_spin.setValue(8192)
        aform.addRow("Axis points:", self.points_spin)
        self.show_sites_check = QCheckBox("Also plot each site (Simulation plot)")
        self.show_sites_check.setChecked(True)
        aform.addRow("", self.show_sites_check)
        self.compare_orders_check = QCheckBox("Compare levels of theory (first, second, exact)")
        self.compare_orders_check.setToolTip(
            "With a quadrupolar peak: simulate first order, second order and exact together "
            "and overlay them, with their differences, on the Simulation plot. The spectrum "
            "kept is the one Quadrupolar theory selects.")
        self.compare_orders_check.setEnabled(self._draw_orders is not None)
        aform.addRow("", self.compare_orders_check)
        v.addLayout(aform)

        self.simulate_button = QPushButton("Simulate")
        self.simulate_button.clicked.connect(self.simulate)
        v.addWidget(self.simulate_button)
        self.results_label = QLabel("")
        self.results_label.setWordWrap(True)
        self.results_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        v.addWidget(self.results_label)
        row = QHBoxLayout()
        self.export_button = QPushButton("Keep in Open Experiments")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self._on_export)
        self.python_button = QPushButton("Copy as Python")
        self.python_button.setToolTip("Copy notebook code that reproduces this simulation.")
        self.python_button.clicked.connect(self.copy_python)
        self.to_fit_button = QPushButton("Send to Fit as Starting Values")
        self.to_fit_button.setToolTip("Put these sites into Fit ▸ CSA, Quad & J-coupling as the "
                                      "starting model, with this regime, transitions and level "
                                      "of theory.")
        self.to_fit_button.setEnabled(self._send_to_fit is not None)
        self.to_fit_button.clicked.connect(self.send_to_fit)
        row.addWidget(self.export_button)
        row.addWidget(self.to_fit_button)
        row.addWidget(self.python_button)
        v.addLayout(row)

    # ---- helpers ----------------------------------------------------------------

    def spin(self):
        try:
            return float(find_isotope_info(self.nucleus_combo.currentText().strip())["spin"])
        except KeyError:
            return None

    def _sync_spin(self):
        spin = self.spin()
        self.spin_label.setText("—" if spin is None else f"{spin:g}")
        self.transitions_combo.setEnabled(bool(spin and spin >= 1))
        if spin is not None and spin >= 1 and float(spin).is_integer() \
                and self.transitions_combo.currentData() == "ct":
            self.transitions_combo.setCurrentIndex(self.transitions_combo.findData("all"))

    def _sync_controls(self):
        self.mas_spin.setEnabled(self.regime_combo.currentData() == "mas")
        self.sideband_hint.setVisible(self.regime_combo.currentData() == "mas")

    def _on_panel_nucleus(self):
        self._sync_spin()
        if hasattr(self, "site_nucleus_combo"):
            self._show_site_nucleus(self.current_site())

    # ---- several nuclei --------------------------------------------------------

    def observed(self) -> str:
        return self.nucleus_combo.currentText().strip()

    def site_nucleus(self, site) -> str:
        """A site's nucleus: its own ("isotope") or the observed one."""
        return (site or {}).get("isotope") or self.observed()

    def _site_spin(self, site):
        try:
            return float(find_isotope_info(self.site_nucleus(site))["spin"])
        except KeyError:
            return None

    def _resolved(self, site):
        """The site as it is simulated: a linked partner's values derived
        from its parent's (None if the link cannot hold right now)."""
        if not (site or {}).get("link"):
            return site
        dicts = [{**s, "nucleus": self.site_nucleus(s)} for s in self.site_dicts]
        try:
            resolved = ch.resolve_link_dicts(dicts)
        except (ValueError, KeyError):
            return None
        return resolved[self.site_dicts.index(site)]

    def _link_candidates(self, site):
        """Labels of the sites ``site`` may be linked to: unlinked sites
        of another isotope of its element."""
        if site is None:
            return []
        mine = self.site_nucleus(site)
        try:
            element = isotopes.element_of(mine)
        except KeyError:
            return []
        out = []
        for other in self.site_dicts:
            if other is site or other.get("link"):
                continue
            theirs = self.site_nucleus(other)
            try:
                if theirs != mine and isotopes.element_of(theirs) == element:
                    out.append(other["label"])
            except KeyError:
                continue
        return out

    def _show_site_nucleus(self, site) -> None:
        """Fill the Nucleus / Same chemistry as / Not shared rows for ``site``."""
        rows = (self.site_nucleus_combo, self.link_combo, self.own_widget)
        for w in rows:
            self._sform.setRowVisible(w, self.OTHER_NUCLEI)
        if not self.OTHER_NUCLEI:
            self.link_note.setVisible(False)
            return
        observed = self.observed()
        combo = self.site_nucleus_combo
        combo.blockSignals(True)
        combo.clear()
        combo.addItem(f"Observed ({observed or '?'})", "")
        try:
            partners = isotopes.isotope_partners(observed)
        except KeyError:
            partners = []
        for nuc in partners:
            combo.addItem(f"{nuc} (isotope of {isotopes.element_of(nuc)})", nuc)
        combo.insertSeparator(combo.count())
        for nuc in _nuclides():
            if nuc != observed and nuc not in partners:
                combo.addItem(nuc, nuc)
        own = (site or {}).get("isotope") or ""
        if own == observed:
            own = ""
        index = combo.findData(own)
        if index < 0 and own:
            combo.addItem(own, own)
            index = combo.count() - 1
        combo.setCurrentIndex(max(index, 0))
        combo.blockSignals(False)

        link = (site or {}).get("link")
        lcombo = self.link_combo
        lcombo.blockSignals(True)
        lcombo.clear()
        lcombo.addItem("— (own parameters)", "")
        for label in self._link_candidates(site):
            lcombo.addItem(label, label)
        if link and lcombo.findData(link["to"]) < 0:
            lcombo.addItem(f"{link['to']} (cannot link)", link["to"])
        lcombo.setCurrentIndex(max(lcombo.findData(link["to"]) if link else 0, 0))
        lcombo.setEnabled(lcombo.count() > 1)
        lcombo.blockSignals(False)
        free = set((link or {}).get("free") or [])
        for box, name in ((self.own_amount_check, "amount"), (self.own_fwhm_check, "fwhm")):
            box.blockSignals(True)
            box.setChecked(name in free)
            box.blockSignals(False)
        self._sform.setRowVisible(self.own_widget, bool(link))
        self.link_note.setText(self.link_text(site))
        self.link_note.setVisible(bool(self.link_note.text()))

    def link_text(self, site) -> str:
        """What a site of another nucleus is, in words: where it sits on the
        spectrum and, when linked, what it takes from its parent."""
        if not site:
            return ""
        nucleus, observed = self.site_nucleus(site), self.observed()
        lines = []
        try:
            if nucleus != observed:
                lines.append(isotopes.describe_placement(nucleus, observed))
            link = site.get("link")
            if link:
                parent = next((s for s in self.site_dicts if s["label"] == link["to"]), None)
                if parent is None:
                    return f"Linked to {link['to']}, which is not a peak any more."
                f = isotopes.partner_factors(self.site_nucleus(parent), nucleus)
                own = set(link.get("free") or [])
                shared = ["δiso", "CSA"] + ([] if "fwhm" in own else ["broadening"])
                text = f"Follows {link['to']}: " + ", ".join(shared) + ", η_Q and Euler angles"
                if f["C_Q"]:
                    text += f"; C_Q × {f['C_Q']:.4g} (quadrupole moments)"
                if any(i.get("type") == "jcoupling" for i in parent.get("interactions") or []):
                    text += f"; J × {f['J']:.6g} (γ)"
                if "amount" not in own and f["amount"]:
                    text += (f"; amount × {f['amount']:.4g} (natural abundance × receptivity, "
                             "for uniform excitation)")
                lines.append(text + ". Edit the parent to change them.")
                if self._resolved(site) is None:
                    lines.append("⚠ This link cannot hold as the peaks stand -- see Simulate "
                                 "or Fit for the reason.")
        except KeyError:
            return ""
        return "\n".join(lines)

    def _on_site_nucleus(self, _index) -> None:
        site = self.current_site()
        if site is None:
            return
        nucleus = self.site_nucleus_combo.currentData() or ""
        if nucleus:
            site["isotope"] = nucleus
        else:
            site.pop("isotope", None)
        if site.get("link") and site["link"]["to"] not in self._link_candidates(site):
            site.pop("link", None)
        spin = self._site_spin(site) or 0.0
        if spin < 1:
            site["interactions"] = [i for i in site["interactions"] if i["type"] != "quadrupolar"]
        self._on_site_selected(self._current)

    def _on_link(self, _index) -> None:
        site = self.current_site()
        if site is None:
            return
        to = self.link_combo.currentData() or ""
        if to:
            site["link"] = {"to": to, "free": list((site.get("link") or {}).get("free") or [])}
        else:
            resolved = self._resolved(site)
            if resolved is not None and site.get("link"):
                # Keep what it showed as the start of its own parameters.
                for key in ("delta_iso", "amount", "fwhm", "lineshape", "eta"):
                    site[key] = resolved[key]
                site["interactions"] = copy.deepcopy(resolved["interactions"])
            site.pop("link", None)
        self._on_site_selected(self._current)

    def _on_own(self, name, on) -> None:
        site = self.current_site()
        if site is None or not site.get("link"):
            return
        free = [n for n in site["link"].get("free") or [] if n != name]
        if on:
            resolved = self._resolved(site)
            if resolved is not None:
                site[name] = resolved[name]      # start from the derived value
            free.append(name)
        site["link"]["free"] = free
        self._on_site_selected(self._current)

    def add_isotope_partner(self, nucleus=None):
        """Add the selected peak's isotope partner, linked to it. With
        several isotopes to choose from and no ``nucleus``, ask with a menu."""
        site = self.current_site()
        if site is None:
            return None
        if site.get("link"):
            QMessageBox.information(self, "Add isotope partner",
                                    f"{site['label']} is itself a partner of "
                                    f"{site['link']['to']}: add partners to that peak.")
            return None
        mine = self.site_nucleus(site)
        try:
            options = isotopes.isotope_partners(mine)
        except KeyError:
            options = []
        if nucleus is None:
            if not options:
                QMessageBox.information(self, "Add isotope partner",
                                        f"{mine} has no other NMR-active isotope.")
                return None
            if len(options) == 1:
                nucleus = options[0]
            else:
                menu = QMenu(self)
                for nuc in options:
                    menu.addAction(nuc).setData(nuc)
                chosen = menu.exec(self.add_partner_button.mapToGlobal(
                    self.add_partner_button.rect().bottomLeft()))
                if chosen is None:
                    return None
                nucleus = chosen.data()
        parent_label = site["label"]
        new = copy.deepcopy(site)
        new["label"] = self._unique_label(f"{parent_label} ({nucleus})")
        new["link"] = {"to": parent_label, "free": []}
        if nucleus == self.observed():
            new.pop("isotope", None)
        else:
            new["isotope"] = nucleus
        dicts = [{**s, "nucleus": self.site_nucleus(s)} for s in self.site_dicts + [new]]
        try:
            resolved = ch.resolve_link_dicts(dicts)[-1]
        except (ValueError, KeyError) as exc:
            QMessageBox.warning(self, "Add isotope partner", str(exc))
            return None
        for key in ("delta_iso", "amount", "fwhm", "lineshape", "eta"):
            new[key] = resolved[key]
        new["interactions"] = copy.deepcopy(resolved["interactions"])
        return self.add_site(new)

    def _sync_menu(self):
        site = self.current_site()
        spin = (self._site_spin(site) if site else self.spin()) or 0.0
        has = {k: bool(site and any(i["type"] == k for i in site["interactions"]))
               for k in ch.INTERACTION_TYPES}
        self._actions["csa"].setEnabled(site is not None and not has["csa"])
        self._actions["quadrupolar"].setEnabled(site is not None and not has["quadrupolar"]
                                                and spin >= 1)
        self._actions["quadrupolar"].setToolTip(
            "" if spin >= 1 else "The observed nucleus is spin-1/2: no quadrupole moment.")
        self._actions["jcoupling"].setEnabled(site is not None)
        if "strong" in self._actions:
            self._actions["strong"].setEnabled(bool(self.site_dicts))

    def _sync_theory_combos(self, index=None):
        """Every quadrupolar editor's Theory box follows the panel's."""
        index = self.order_combo.currentIndex() if index is None else index
        for editor in getattr(self, "interaction_editors", []):
            combo = editor.theory_combo
            if combo is not None and combo.currentIndex() != index:
                combo.blockSignals(True)
                combo.setCurrentIndex(index)
                combo.blockSignals(False)

    def request_strong_coupling(self) -> None:
        """Hand these peaks to the exact spin-system model (strongly
        coupled J between peaks), when the panel offers it."""
        if self._strong_coupling is None:
            return
        self._strong_coupling(copy.deepcopy(self.site_dicts),
                              self.nucleus_combo.currentText().strip(), self.larmor_spin.value())

    # ---- saved state (simulation history) -----------------------------

    #: The "Selected site" fields show one site of site_dicts; the model
    #: itself is saved, not these.
    _SITE_FIELDS = ("label_edit", "diso_spin", "amount_spin", "fwhm_spin", "lineshape_combo",
                    "pv_eta_spin", "site_nucleus_combo", "link_combo", "own_amount_check",
                    "own_fwhm_check")

    # ---- many sites at once ----------------------------------------------------------

    @contextlib.contextmanager
    def bulk(self):
        """Changes to many sites, made by code: whatever is redrawn per
        change (the fit's parameter table, the preview) is redrawn once,
        at the end -- 1200 sites would otherwise take minutes."""
        self._bulk += 1
        try:
            yield
        finally:
            self._bulk -= 1
            if not self._bulk:
                self._after_bulk()

    def _after_bulk(self) -> None:
        """Called once when the outermost bulk() ends (subclasses refresh)."""

    def add_sites(self, sites) -> None:
        """Append many sites in one go (one list update, one selection)."""
        new = [copy.deepcopy(s) for s in sites]
        if not new:
            return
        with self.bulk():
            self.site_list.blockSignals(True)
            for site in new:
                self.site_dicts.append(site)
                self.site_list.addItem(site["label"])
            self.site_list.blockSignals(False)
            self.site_list.setCurrentRow(len(self.site_dicts) - len(new))
        self._sync_copy_controls()

    def selected_rows(self) -> list[int]:
        return sorted({self.site_list.row(i) for i in self.site_list.selectedItems()})

    def _target_rows(self, targets) -> list[int]:
        if targets == "all":
            return list(range(len(self.site_dicts)))
        if targets == "selected":
            return self.selected_rows()
        return sorted({int(r) for r in targets})

    def copy_interactions(self, targets="selected", broadening: bool = False,
                          source: int | None = None) -> dict:
        """The source peak's (default: the current one's) interactions --
        and, with ``broadening``, its FWHM, lineshape and L fraction -- onto
        ``targets`` ('selected', 'all', or row numbers), replacing theirs.
        δiso, amount and label are each peak's own. A peak linked to another
        isotope's site follows that site and is skipped; a quadrupolar
        coupling is not given to a spin-1/2 peak, nor to a peak of another
        nucleus (C_Q depends on the isotope's quadrupole moment). Returns a
        summary, also shown under the list."""
        source = self._current if source is None else int(source)
        if not 0 <= source < len(self.site_dicts):
            return {"copied": 0, "text": "Select the peak to copy from first."}
        rows = [r for r in self._target_rows(targets) if r != source]
        if not rows:
            text = ("Select the peaks to copy to (Ctrl/Shift-click; Ctrl+A for all), or use "
                    "'→ all peaks'.")
            self.selection_label.setText(text)
            return {"copied": 0, "linked": [], "no_quadrupolar": [], "text": text}
        # The notebook's copy_interactions, on the panel's dicts with their
        # nuclei filled in; a linked source gives what it shows (its parent's).
        dicts = [{**d, "nucleus": self.site_nucleus(d)} for d in self.site_dicts]
        dicts[source] = {**(self._resolved(self.site_dicts[source]) or self.site_dicts[source]),
                         "nucleus": self.site_nucleus(self.site_dicts[source])}
        dicts[source].pop("link", None)
        new, report = ch.copy_interactions(dicts, source, rows, broadening=broadening)
        copied = set(report["copied"])
        with self.bulk():
            for r in rows:
                site = self.site_dicts[r]
                if site.get("label") not in copied:
                    continue
                site["interactions"] = new[r]["interactions"]
                if broadening:
                    for key in ("fwhm", "lineshape", "eta"):
                        if key in new[r]:
                            site[key] = new[r][key]
            self._on_site_selected(self._current)
        self.copy_status = report["text"]
        self.selection_label.setText(report["text"])
        return {**report, "copied": len(report["copied"])}

    def clear_interactions(self, targets="selected") -> int:
        """Make ``targets`` isotropic again (linked peaks skipped)."""
        rows = self._target_rows(targets)
        n = 0
        with self.bulk():
            for r in rows:
                site = self.site_dicts[r]
                if not site.get("link") and site["interactions"]:
                    site["interactions"] = []
                    n += 1
            self._on_site_selected(self._current)
        self.selection_label.setText(f"Interactions removed from {n} peak(s).")
        return n

    def _name_copy_actions(self) -> None:
        """Name the source in the menu: the peak shown under Selected peak
        (the last one clicked)."""
        site = self.current_site()
        name = site["label"] if site else "this peak"
        texts = (f"{name}'s interactions → selected peaks",
                 f"{name}'s interactions → all peaks",
                 f"{name}'s interactions and broadening → selected peaks",
                 f"{name}'s interactions and broadening → all peaks",
                 "Remove all interactions from the selected peaks")
        for action, text in zip(self._copy_actions, texts):
            action.setText(text.replace("&", "&&"))

    def _sync_copy_controls(self) -> None:
        n = len(self.selected_rows())
        total = len(self.site_dicts)
        if total > 1:
            self.selection_label.setText(
                f"{n} of {total} peaks selected" + (" -- Copy to Peaks gives them the current "
                                                    "peak's interactions." if n > 1 else "."))
        else:
            self.selection_label.setText("")
        self.copy_button.setEnabled(total > 1)

    def get_state(self) -> dict:
        state = {"widgets": panel_state.capture_widgets(self, exclude=self._SITE_FIELDS),
                 "sites": copy.deepcopy(self.site_dicts), "current": self._current}
        state["extra"] = {f"Site {s.get('label')}": _describe_site(s) for s in self.site_dicts}
        return state

    def set_state(self, state: dict) -> None:
        state = state or {}
        panel_state.restore_widgets(self, state.get("widgets") or {})
        self.site_list.blockSignals(True)
        self.site_list.clear()
        self.site_list.blockSignals(False)
        self.site_dicts = []
        self._current = -1
        self.add_sites(state.get("sites") or [])
        current = int(state.get("current", 0) or 0)
        if self.site_dicts:
            self.site_list.setCurrentRow(min(max(current, 0), len(self.site_dicts) - 1))
        else:
            self._on_site_selected(-1)
        self._sync_spin()
        self._sync_controls()

    def current_site(self):
        return self.site_dicts[self._current] if 0 <= self._current < len(self.site_dicts) else None

    def _set(self, key, value):
        site = self.current_site()
        if site is not None:
            site[key] = value

    # ---- sites ------------------------------------------------------------------------

    def _unique_label(self, base):
        labels = {s["label"] for s in self.site_dicts}
        if base not in labels:
            return base
        k = 2
        while f"{base} ({k})" in labels:
            k += 1
        return f"{base} ({k})"

    def add_site(self, site: dict | None = None):
        site = copy.deepcopy(site) if site else {
            "label": self._unique_label(f"Site {len(self.site_dicts) + 1}"), "delta_iso": 0.0,
            "amount": 1.0, "fwhm": 1.0, "lineshape": "gaussian", "eta": 0.5, "interactions": []}
        self.site_dicts.append(site)
        self.site_list.addItem(site["label"])
        self.site_list.setCurrentRow(len(self.site_dicts) - 1)
        return site

    def duplicate_site(self):
        site = self.current_site()
        if site is not None:
            dup = copy.deepcopy(site)
            dup["label"] = self._unique_label(site["label"])
            self.add_site(dup)

    def remove_site(self):
        row = self._current
        if row < 0:
            return
        removed = self.site_dicts.pop(row)
        for other in self.site_dicts:
            if (other.get("link") or {}).get("to") == removed["label"]:
                # Its partner keeps what it showed, as its own parameters.
                resolved = self._resolved(other)
                if resolved is not None:
                    for key in ("delta_iso", "amount", "fwhm", "lineshape", "eta"):
                        other[key] = resolved[key]
                    other["interactions"] = copy.deepcopy(resolved["interactions"])
                other.pop("link", None)
        self.site_list.blockSignals(True)
        self.site_list.takeItem(row)
        self.site_list.blockSignals(False)
        self._current = -1
        self.site_list.setCurrentRow(min(row, len(self.site_dicts) - 1))
        if not self.site_dicts:
            self._on_site_selected(-1)

    def _on_label_edited(self, text):
        site = self.current_site()
        if site is not None:
            old = site["label"]
            site["label"] = text
            self.site_list.item(self._current).setText(text)
            for other in self.site_dicts:
                if (other.get("link") or {}).get("to") == old:
                    other["link"]["to"] = text

    def _on_lineshape(self, _i):
        data = self.lineshape_combo.currentData()
        self._set("lineshape", data)
        self.pv_eta_spin.setVisible(data == "pseudo_voigt")
        label = self._sform.labelForField(self.pv_eta_spin)
        if label is not None:
            label.setVisible(data == "pseudo_voigt")

    def _on_site_selected(self, row):
        self._current = row
        site = self.current_site()
        self.site_box.setEnabled(site is not None)
        if site is None:
            self._rebuild_interactions()
            return
        widgets = (self.label_edit, self.diso_spin, self.amount_spin, self.fwhm_spin,
                   self.lineshape_combo, self.pv_eta_spin)
        for w in widgets:
            w.blockSignals(True)
        shown = self._resolved(site) or site       # a linked partner shows its derived values
        self.label_edit.setText(site["label"])
        self.diso_spin.setValue(shown["delta_iso"])
        self.amount_spin.setValue(shown["amount"])
        self.fwhm_spin.setValue(shown["fwhm"])
        self.lineshape_combo.setCurrentIndex(self.lineshape_combo.findData(shown["lineshape"]))
        self.pv_eta_spin.setValue(shown.get("eta", 0.5))
        for w in widgets:
            w.blockSignals(False)
        free = set((site.get("link") or {}).get("free") or []) if site.get("link") else None
        linked = free is not None
        self.diso_spin.setEnabled(not linked)
        self.amount_spin.setEnabled(not linked or "amount" in free)
        self.fwhm_spin.setEnabled(not linked or "fwhm" in free)
        self.lineshape_combo.setEnabled(not linked)
        self.pv_eta_spin.setEnabled(not linked)
        self.add_interaction_button.setEnabled(not linked)
        self.add_partner_button.setEnabled(not linked)
        self._show_site_nucleus(site)
        if not linked:
            self._on_lineshape(0)
        self._rebuild_interactions()

    # ---- interactions ----------------------------------------------------------------------

    def add_interaction(self, kind, **values):
        site = self.current_site()
        if site is None:
            return None
        if kind in ("csa", "quadrupolar") and any(i["type"] == kind for i in site["interactions"]):
            QMessageBox.information(self, "Add interaction",
                                    f"{site['label']} already has a {_TITLES[kind]} -- a site "
                                    "has one; edit it instead.")
            return None
        if kind == "quadrupolar" and (self._site_spin(site) or 0) < 1:
            QMessageBox.information(self, "Add interaction",
                                    f"{self.site_nucleus(site)} is spin-1/2: it has no "
                                    "quadrupole moment.")
            return None
        if site.get("link"):
            QMessageBox.information(self, "Add interaction",
                                    f"{site['label']} shares {site['link']['to']}'s chemistry: "
                                    f"add the interaction to {site['link']['to']}.")
            return None
        data = copy.deepcopy(self.DEFAULT_INTERACTIONS[kind])
        data.update(values)
        site["interactions"].append(data)
        self._rebuild_interactions()
        return data

    def _remove_interaction(self, data):
        site = self.current_site()
        if site is not None:
            site["interactions"] = [i for i in site["interactions"] if i is not data]
            self._rebuild_interactions()

    def _rebuild_interactions(self):
        while self.interactions_layout.count():
            item = self.interactions_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                # Detach now: deleteLater alone leaves the old box drawn
                # (overlapping the new ones) until the event loop runs.
                widget.hide()
                widget.setParent(None)
                widget.deleteLater()
        site = self.current_site()
        interactions = site["interactions"] if site else []
        linked = bool(site and site.get("link"))
        if linked:
            # Shown, not edited: copies of the parent's, as this isotope sees them.
            interactions = copy.deepcopy((self._resolved(site) or site)["interactions"])
        has_quad = any(i["type"] == "quadrupolar" for i in interactions)
        order = {"quadrupolar": 0, "csa": 1, "jcoupling": 2}
        self.interaction_editors = []
        for data in sorted(interactions, key=lambda i: order[i["type"]]):
            editor = InteractionEditor(data, self._remove_interaction, has_quad, panel=self)
            editor.setEnabled(not linked)
            self.interactions_layout.addWidget(editor)
            self.interaction_editors.append(editor)
        self.no_interactions_label.setVisible(site is not None and not interactions)

    # ---- seeding --------------------------------------------------------------------------------

    def seed_from_spectrum(self, spectrum) -> None:
        if spectrum is None or spectrum is self._seeded_for:
            return
        self._seeded_for = spectrum
        meta = getattr(spectrum, "metadata", None)
        nucleus = getattr(meta, "nucleus", None)
        if nucleus:
            try:
                find_isotope_info(nucleus)
                self.nucleus_combo.setCurrentText(nucleus)
            except KeyError:
                pass
        if getattr(meta, "sfo1", None):
            self.larmor_spin.setValue(float(meta.sfo1))
        try:
            masr = float(meta.raw.get("acqus", {}).get("MASR", 0) or 0)
        except (AttributeError, TypeError, ValueError):
            masr = 0.0
        if masr > 0:
            self.mas_spin.setValue(masr)
            self.regime_combo.setCurrentIndex(self.regime_combo.findData("mas"))

    # ---- simulate -----------------------------------------------------------------------------------

    def build_sites(self):
        nucleus = self.nucleus_combo.currentText().strip()
        try:
            find_isotope_info(nucleus)
        except KeyError:
            raise ValueError(f"Unknown nucleus {nucleus!r}; enter e.g. '27Al'.") from None
        if not self.site_dicts:
            raise ValueError("Add at least one site.")
        sites = [ch.Site.from_dict({**s, "nucleus": s.get("isotope") or nucleus})
                 for s in self.site_dicts]
        ch.resolve_links(sites)          # a link that cannot hold is reported now, by name
        return sites

    def settings(self):
        larmor = self.larmor_spin.value()
        if larmor <= 0:
            raise ValueError("Set the field (or the Larmor frequency).")
        mas = self.mas_spin.value() if self.regime_combo.currentData() == "mas" else 0.0
        if self.regime_combo.currentData() == "mas" and mas <= 0:
            raise ValueError("The MAS lineshape needs a spinning rate: set MAS rate, or switch "
                             "Regime to Static powder.")
        spin = self.spin() or 0.5
        transitions = self.transitions_combo.currentData() if spin >= 1 else "all"
        out = {"larmor_Hz": larmor * 1e6, "mas_Hz": mas,
               "order": self.order_combo.currentData(), "transitions": transitions,
               "n_zcw": self.orientations_spin.value() or None}
        observed = self.observed()
        if any(self.site_nucleus(s) != observed for s in self.site_dicts):
            out["observe"] = observed          # sites of other nuclei are placed on its scale
        return out

    def simulate(self) -> bool:
        try:
            sites = self.build_sites()
            settings = self.settings()
            if self.auto_axis_check.isChecked():
                lo, hi = ch.combined_axis_limits(sites, settings["larmor_Hz"],
                                                 transitions=settings["transitions"],
                                                 mas_Hz=settings["mas_Hz"],
                                                 order=settings["order"],
                                                 observe=self.observed())
                for box, value in ((self.ppm_min_spin, lo), (self.ppm_max_spin, hi)):
                    box.blockSignals(True)
                    box.setValue(value)
                    box.blockSignals(False)
            if self.ppm_max_spin.value() <= self.ppm_min_spin.value():
                raise ValueError("Axis max must be greater than axis min.")
            axis = np.linspace(self.ppm_min_spin.value(), self.ppm_max_spin.value(),
                               self.points_spin.value())
            started = time.perf_counter()
            comparison = None
            if self._compare_orders_wanted(sites):
                comparison = {}
                for order in ("first", "second", "exact"):
                    run = ch.simulate_combined(sites, axis, **{**settings, "order": order})
                    comparison[order] = {"x_ppm": axis, "real_spec": run["real_spec"],
                                         "components": {}}
                    if order == settings["order"]:
                        result = run
            else:
                result = ch.simulate_combined(sites, axis, **settings)
            elapsed = time.perf_counter() - started
        except (ValueError, KeyError) as exc:
            QMessageBox.warning(self, "CSA, Quad & J-coupling", str(exc))
            return False
        nucleus = self.observed() or sites[0].nucleus
        self.result = {"result": result, "sites": sites, "settings": settings, "axis": axis}
        data = np.asarray(result["real_spec"], dtype=float)
        metadata = NMRMetadata(
            vendor="simulated", nucleus=nucleus, sfo1=settings["larmor_Hz"] / 1e6,
            raw={"simulation": {"kind": "combined", "sites": result["site_dicts"],
                                "settings": result["settings"], "log": result["log"]}})
        self.spectrum = NMRSpectrum(data=data, axes=[NMRAxis(kind="frequency", ppm=axis)],
                                    metadata=metadata)
        self._plot_spectrum(self.spectrum)
        if comparison is not None:
            self._draw_orders(comparison, nucleus, settings["larmor_Hz"] / 1e6)
        elif self._draw_sites is not None and self.show_sites_check.isChecked() and len(sites) > 1:
            self._draw_sites(result, nucleus)
        edge = max(abs(data[0]), abs(data[-1])) / (np.max(np.abs(data)) or 1.0)
        warn = ("\n⚠ Intensity at the axis edge is {:.0%} of the peak -- the pattern is "
                "clipped; widen the axis.".format(edge)) if edge > 0.01 else ""
        self.results_label.setText(f"Simulated in {elapsed:.2f} s.\n{result['log']}{warn}")
        self.export_button.setEnabled(True)
        return True

    def _compare_orders_wanted(self, sites) -> bool:
        check = getattr(self, "compare_orders_check", None)
        return (check is not None and check.isChecked() and self._draw_orders is not None
                and any(any(i.get("type") == "quadrupolar" for i in s.to_dict()["interactions"])
                        for s in sites))

    def transfer_settings(self) -> dict:
        """Regime, spinning, level of theory, transitions and field, as
        another Combined panel takes them (no validation)."""
        return {"regime": self.regime_combo.currentData(), "mas_Hz": self.mas_spin.value(),
                "order": self.order_combo.currentData(),
                "transitions": self.transitions_combo.currentData(),
                "larmor_MHz": self.larmor_spin.value()}

    def send_to_fit(self) -> None:
        if self._send_to_fit is None or not self.site_dicts:
            return
        self._send_to_fit(copy.deepcopy(self.site_dicts), self.transfer_settings(),
                          self.nucleus_combo.currentText().strip())

    def _on_export(self):
        if self.spectrum is not None:
            self._add_to_open(self.spectrum, "Simulated (combined)")

    def python_code(self) -> str:
        sites = self.build_sites()
        settings = self.settings()
        axis = np.linspace(self.ppm_min_spin.value(), self.ppm_max_spin.value(),
                           self.points_spin.value())
        return ch.combined_to_python(sites, axis, **settings)

    def copy_python(self):
        try:
            QGuiApplication.clipboard().setText(self.python_code())
        except (ValueError, KeyError) as exc:
            QMessageBox.warning(self, "Copy as Python", str(exc))


def cfit_n_zcw(settings: dict) -> int:
    """The orientation count a fit with these settings uses (its default
    when "auto")."""
    return int(settings.get("n_zcw") or (2584 if settings.get("mas_Hz") else 28657))


def _describe_site(site: dict) -> str:
    """A site as one readable line, for the simulation log."""
    bits = [f"δiso {site.get('delta_iso', 0):g} ppm", f"amount {site.get('amount', 1):g}",
            f"FWHM {site.get('fwhm', 0):g} ppm ({site.get('lineshape', '')})"]
    if site.get("isotope"):
        bits.insert(0, f"nucleus {site['isotope']}")
    if site.get("link"):
        own = site["link"].get("free") or []
        bits.append(f"linked to {site['link'].get('to')}"
                    + (f" (own {', '.join(own)})" if own else "") + " -- its other values follow")
    for i in site.get("interactions") or []:
        kind = i.get("type")
        if kind == "csa":
            bits.append(f"CSA Δ {i.get('aniso', 0):g} ppm, η {i.get('asym', 0):g} "
                        f"({i.get('convention', '')}), Euler {tuple(i.get('euler', ()))}")
        elif kind == "quadrupolar":
            bits.append(f"C_Q {i.get('C_Q', 0):g} MHz, η_Q {i.get('eta_Q', 0):g}")
        elif kind == "jcoupling":
            bits.append(f"J to {i.get('n', 1)}×{i.get('nuclide', '?')} {i.get('J_Hz', 0):g} Hz")
    return "; ".join(bits)


# ============================================================================
# Fit > CSA, Quad & J-coupling
# ============================================================================

#: Parameter table columns.
_P_SITE, _P_NAME, _P_START, _P_FREE, _P_MIN, _P_MAX, _P_TRY = range(7)
_P_HEADERS = ("Site", "Parameter", "Start", "Free", "Min", "Max", "Try starts")
_PARAM_LABELS = {"delta_iso": "δiso (ppm)", "amount": "Amount", "fwhm": "FWHM (ppm)",
                 "eta": "L fraction", "aniso": "Δ or span (ppm)", "asym": "η or skew",
                 "alpha": "α (°)", "beta": "β (°)", "gamma": "γ (°)", "C_Q": "C_Q (MHz)",
                 "eta_Q": "η_Q", "mas_Hz": "MAS rate (Hz)"}

#: The parameter table's row for what belongs to the whole spectrum,
#: not to a peak: the MAS rate (explicit user request, 2026-10-06 --
#: "add the MAS rate as a fitting parameter; by default fixed").
SPECTRUM_ROW = "(spectrum)"
#: How far a freed MAS rate may move when Min/Max are blank.
MAS_WINDOW_HZ = 500.0


def _param_label(name: str) -> str:
    return _PARAM_LABELS.get(name, f"{name} (Hz)" if name.startswith("J") else name)


def _parse_float(text: str):
    text = (text or "").strip()
    if not text:
        return None
    return float(text)


class CombinedFitPanel(CombinedHamiltonianPanel):
    """Fit CSA, Quad & J-coupling sites to the selected spectrum: the same
    site editor as Simulate > CSA, Quad & J-coupling, plus a table of every
    parameter -- start, free/fixed, bounds, and optional extra starting
    values for a multi-start fit -- and the result as a table.

    The maths is :func:`opennmr.notebook.combined_fitting.fit_combined`
    (and ``fit_combined_multistart``), exactly as a notebook calls it.

    Callbacks: ``get_target()`` -> (spectrum, row, index) or raises
    ValueError; ``get_regions()`` -> the Integration regions;
    ``draw_fit(result)``; ``format_value_error(value, error, fixed)``;
    ``send_to_simulation(settings, site_dicts, nucleus)``; ``on_fitted()``.
    """

    BLURB = ("Fit sites built from any mix of CSA, quadrupolar and J interactions to the "
             "selected 1D spectrum (or the displayed slice of a 2D), static or MAS. Build the "
             "starting model as in Simulate, then choose below which parameters are free and "
             "their bounds. Euler angles start fixed: one 1D lineshape rarely determines them.")

    #: Live-preview powder orientations when "auto": static, MAS. The
    #: shape is within a few % of the fit's (measured: 233 under MAS is
    #: 4% of the peak off the converged pattern, in ~0.1 s per site).
    PREVIEW_ZCW = (4181, 233)
    #: ms of quiet before the preview recomputes (typing, spinning a box).
    PREVIEW_DELAY_MS = 200

    def __init__(self, get_target, get_regions, draw_fit, format_value_error,
                 send_to_simulation=None, on_fitted=None, parent=None,
                 strong_coupling=None, get_alternatives=None,
                 draw_correlations=None) -> None:
        #: ``draw_correlations(fit) -> bool``: the fit's parameter
        #: correlation matrix on the Fit view (False: it has none).
        self._draw_correlations = draw_correlations
        #: ``get_alternatives(result) -> [(name, fit), ...]``: other fits of
        #: the same spectrum (kept fit models) for the advice to compare with.
        self._get_alternatives = get_alternatives
        #: Live preview of the starting model (explicit user request,
        #: 2026-10-06: "add a peak and see it appear, and change with each
        #: change in parameter"). Debounced; suspended while the panel is
        #: filled by code (restores, transfers) so it never overdraws a fit.
        self._preview_timer = QTimer()
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(self.PREVIEW_DELAY_MS)
        self._preview_timer.timeout.connect(lambda: self.preview_now(quiet=True))
        self._preview_suspended = 0
        self.preview: dict | None = None
        self._get_target = get_target
        self._get_regions = get_regions
        self._draw_fit = draw_fit
        self._fmt = format_value_error
        self._send_to_simulation = send_to_simulation
        self._on_fitted = on_fitted or (lambda: None)
        #: {site label: {parameter: {"free": bool, "min": str, "max": str, "try": str}}}
        self.constraints: dict[str, dict[str, dict]] = {}
        self._refreshing = False
        super().__init__(plot_spectrum=lambda _s: None, add_to_open=lambda *_a: None,
                         draw_sites=None, parent=parent, strong_coupling=strong_coupling)
        self.refresh_parameter_table()
        for widget in (self.nucleus_combo, self.regime_combo, self.transitions_combo,
                       self.order_combo, self.window_combo, self.estimate_amounts_check,
                       self.preview_check):
            signal = (widget.toggled if isinstance(widget, QCheckBox)
                      else widget.currentIndexChanged)
            signal.connect(lambda *_a: self.schedule_preview())
        self.regime_combo.currentIndexChanged.connect(lambda _i: self.refresh_parameter_table())
        self.mas_spin.valueChanged.connect(lambda _v: self.refresh_parameter_table())
        for box in (self.larmor_spin, self.mas_spin, self.orientations_spin, self.window_lo_spin,
                    self.window_hi_spin, self.preview_orientations_spin):
            box.valueChanged.connect(lambda *_a: self.schedule_preview())

    # ---- the part below the site editor ----------------------------------

    def _build_actions(self, v) -> None:
        v.addWidget(self._build_estimate_box())
        box = QGroupBox("Parameters")
        box_layout = QVBoxLayout(box)
        note = QLabel("Start values come from the sites above (and can be typed here). Untick "
                      "Free to hold a value. Min/Max blank: the natural limits. Try starts: extra "
                      "starting values, comma-separated -- every combination is fitted and the "
                      "best kept (a static CSA is often multimodal).")
        note.setWordWrap(True)
        box_layout.addWidget(note)
        self.parameter_table = QTableWidget(0, len(_P_HEADERS))
        self.parameter_table.setHorizontalHeaderLabels(list(_P_HEADERS))
        self.parameter_table.verticalHeader().setVisible(False)
        self.parameter_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents)
        self.parameter_table.horizontalHeader().setStretchLastSection(True)
        self.parameter_table.setMinimumHeight(150)
        self.parameter_table.itemChanged.connect(self._on_parameter_edited)
        box_layout.addWidget(self.parameter_table)
        v.addWidget(box)

        form = QFormLayout()
        self.window_combo = QComboBox()
        self.window_combo.addItem("Whole spectrum", "all")
        self.window_combo.addItem("Integration regions", "regions")
        self.window_combo.addItem("A ppm range", "range")
        self.window_combo.currentIndexChanged.connect(lambda _i: self._sync_fit_controls())
        form.addRow("Fit window:", self.window_combo)
        self.window_lo_spin = _spin_box(-1e6, 1e6, -200.0, 2, " ppm")
        self.window_hi_spin = _spin_box(-1e6, 1e6, 200.0, 2, " ppm")
        form.addRow("Window low:", self.window_lo_spin)
        form.addRow("Window high:", self.window_hi_spin)
        self._fit_form = form
        self.fit_baseline_check = QCheckBox("Fit the baseline too")
        self.fit_baseline_check.setToolTip("Otherwise a linear baseline is estimated from the "
                                           "outer 5% of the spectrum and held.")
        form.addRow("", self.fit_baseline_check)
        self.estimate_amounts_check = QCheckBox("Estimate starting amounts from the data")
        self.estimate_amounts_check.setChecked(True)
        self.estimate_amounts_check.setToolTip(
            "Start every peak's amount from a non-negative least-squares fit of the starting "
            "shapes to the data (recommended): the amounts typed above are then not used as "
            "starts. Untick to start from the typed amounts (1 = the spectrum's tallest point).")
        form.addRow("", self.estimate_amounts_check)
        self.preview_check = QCheckBox("Live preview of the starting model (Fit view)")
        self.preview_check.setChecked(True)
        self.preview_check.setToolTip(
            "Draw the model the fit will start from over the data, redrawn as peaks, "
            "interactions and settings change -- the fit's own first step: same window, "
            "held baseline and amounts.")
        form.addRow("", self.preview_check)
        self.preview_orientations_spin = QSpinBox()
        self.preview_orientations_spin.setRange(0, 500000)
        self.preview_orientations_spin.setSingleStep(500)
        self.preview_orientations_spin.setSpecialValueText("auto")
        self.preview_orientations_spin.setValue(0)
        self.preview_orientations_spin.setToolTip(
            "Powder orientations for the preview (auto: 4,181 static, 233 MAS -- a few % of "
            "the converged shape, fast enough to follow each change). The fit itself uses "
            "Orientations above.")
        form.addRow("Preview orientations:", self.preview_orientations_spin)
        self.screen_orientations_spin = QSpinBox()
        self.screen_orientations_spin.setRange(0, 500000)
        self.screen_orientations_spin.setSingleStep(500)
        self.screen_orientations_spin.setSpecialValueText("off")
        self.screen_orientations_spin.setValue(0)
        self.screen_orientations_spin.setToolTip(
            "Multi-start only: fit every start with this coarser powder set, then refit only "
            "the best with the full Orientations. Off: every start at full quality.")
        form.addRow("Screen starts at:", self.screen_orientations_spin)
        v.addLayout(form)

        preview_row = QHBoxLayout()
        self.preview_button = QPushButton("Preview Starting Model")
        self.preview_button.setToolTip("Draw the starting model over the data now.")
        self.preview_button.clicked.connect(lambda: self.preview_now(quiet=False))
        preview_row.addWidget(self.preview_button)
        self.fit_button = QPushButton("Fit CSA, Quad && J-coupling")
        self.fit_button.clicked.connect(self.fit)
        preview_row.addWidget(self.fit_button, 1)
        v.addLayout(preview_row)
        self.preview_label = QLabel("")
        self.preview_label.setWordWrap(True)
        v.addWidget(self.preview_label)
        self.results_label = QLabel("")
        self.results_label.setWordWrap(True)
        self.results_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        v.addWidget(self.results_label)
        self.results_table = QTableWidget(0, 1)
        self.results_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.results_table.horizontalHeader().setStretchLastSection(True)
        self.results_table.setMaximumHeight(260)
        self.results_table.setToolTip("Fitted values with 1σ uncertainties; share = number of "
                                      "nuclei (every transition counted). Right-click to copy.")
        self.results_table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.results_table.customContextMenuRequested.connect(self._on_results_menu)
        self.results_table.setVisible(False)
        v.addWidget(self.results_table)
        # What the fit says and what to try next, in plain words (explicit
        # user request, 2026-10-06); notebook.fit_advice.
        self.advice_box = AdviceBox(on_action=self.apply_advice_action)
        v.addWidget(self.advice_box)
        row = QHBoxLayout()
        self.use_fitted_button = QPushButton("Use fitted values as start")
        self.use_fitted_button.clicked.connect(self.use_fitted_as_start)
        self.send_button = QPushButton("Open in Simulate")
        self.send_button.setToolTip("Put the fitted sites into Simulate > CSA, Quad & J-coupling.")
        self.send_button.clicked.connect(self._on_send)
        for b in (self.use_fitted_button, self.send_button):
            b.setEnabled(False)
            row.addWidget(b)
        v.addLayout(row)
        # The fit's parameter correlations (explicit user request,
        # 2026-10-06), in place of the fit on the Fit view while on.
        self.correlation_button = QPushButton("Show Parameter Correlations")
        self.correlation_button.setCheckable(True)
        self.correlation_button.setEnabled(False)
        self.correlation_button.setToolTip(
            "Show the correlation matrix of the fitted parameters instead of the fit (untick "
            "to go back).\n\n"
            "r near ±1: two parameters trade off -- typically Δ and η of a CSA, C_Q and "
            "δiso of a quadrupolar site, or the amounts and widths of overlapping sites. The "
            "data then fix a combination of them, not each one, and their ± understate how "
            "loosely each is pinned. Fixed parameters have no row.")
        self.correlation_button.toggled.connect(self._on_correlation_toggled)
        v.addWidget(self.correlation_button)
        self._sync_fit_controls()

    def _on_correlation_toggled(self, checked: bool) -> None:
        if not self.result:
            return
        if checked:
            if self._draw_correlations is None or not self._draw_correlations(self.result):
                self.correlation_button.blockSignals(True)
                self.correlation_button.setChecked(False)
                self.correlation_button.blockSignals(False)
            return
        if self.result.get("x") is not None:
            self._draw_fit(self.result)

    def _sync_fit_controls(self) -> None:
        manual = self.window_combo.currentData() == "range"
        self._fit_form.setRowVisible(self.window_lo_spin, manual)
        self._fit_form.setRowVisible(self.window_hi_spin, manual)

    # ---- keeping the parameter table in step with the sites --------------

    def _after_bulk(self) -> None:
        self.refresh_parameter_table()

    def add_site(self, site=None):
        out = super().add_site(site)
        self.refresh_parameter_table()
        return out

    def remove_site(self):
        super().remove_site()
        self.refresh_parameter_table()

    def _on_label_edited(self, text):
        site = self.current_site()
        old = site["label"] if site is not None else None
        super()._on_label_edited(text)
        if old is not None and old in self.constraints and text not in self.constraints:
            self.constraints[text] = self.constraints.pop(old)
        self.refresh_parameter_table()

    def _set(self, key, value):
        super()._set(key, value)
        self.refresh_parameter_table()

    def _on_lineshape(self, _i):
        super()._on_lineshape(_i)
        self.refresh_parameter_table()

    def _rebuild_interactions(self):
        super()._rebuild_interactions()
        for editor in getattr(self, "interaction_editors", []):
            for box in editor.findChildren(QDoubleSpinBox) + editor.findChildren(QSpinBox):
                box.valueChanged.connect(lambda _v: self.refresh_parameter_table())
            for combo in editor.findChildren(QComboBox):
                combo.currentIndexChanged.connect(lambda _i: self.refresh_parameter_table())
        self.refresh_parameter_table()

    def _parameter_rows(self):
        for site in self.site_dicts:
            link = site.get("link")
            own = set((link or {}).get("free") or [])
            for name in cfit.site_parameter_names(site):
                if link and name not in own:
                    continue           # follows its parent: no row of its own
                yield site, name

    def _mas_row_shown(self) -> bool:
        return self.regime_combo.currentData() == "mas"

    def mas_constraint(self) -> dict:
        """The MAS rate's row: fixed unless Free is ticked; Min/Max blank =
        the start ± 500 Hz."""
        per = self.constraints.setdefault(SPECTRUM_ROW, {})
        if "mas_Hz" not in per:
            per["mas_Hz"] = {"free": False, "min": "", "max": "", "try": ""}
        return per["mas_Hz"]

    def mas_fit_arguments(self) -> dict:
        """``fit_mas`` (and ``mas_bounds_Hz``) for fit_combined, from the
        MAS rate's row. Raises ValueError naming the cell."""
        c = self.mas_constraint()
        if not (self._mas_row_shown() and c["free"]):
            return {}
        start = self.mas_spin.value()
        try:
            lo, hi = _parse_float(c["min"]), _parse_float(c["max"])
        except ValueError:
            raise ValueError("MAS rate: Min and Max must be numbers in Hz (or blank: ±"
                             f"{MAS_WINDOW_HZ:g} Hz).") from None
        lo = start - MAS_WINDOW_HZ if lo is None else lo
        hi = start + MAS_WINDOW_HZ if hi is None else hi
        if not lo <= start <= hi or lo >= hi:
            raise ValueError(f"MAS rate: Min ({lo:g}) and Max ({hi:g}) must enclose the start "
                             f"({start:g} Hz).")
        return {"fit_mas": True, "mas_bounds_Hz": (lo, hi)}

    def refresh_parameter_table(self) -> None:
        """Rebuild the table from the sites, keeping each parameter's
        free/bounds/try choices by (site label, parameter)."""
        if not hasattr(self, "parameter_table") or self._refreshing or self._bulk:
            return
        self.schedule_preview()
        self._refreshing = True
        t = self.parameter_table
        t.blockSignals(True)
        try:
            rows = list(self._parameter_rows())
            t.setRowCount(len(rows) + (1 if self._mas_row_shown() else 0))
            if self._mas_row_shown():
                c = self.mas_constraint()
                start = self.mas_spin.value()
                cells = [SPECTRUM_ROW, _param_label("mas_Hz"), f"{start:.6g}", "", c["min"],
                         c["max"], ""]
                for col, text in enumerate(cells):
                    item = QTableWidgetItem(text)
                    if col in (_P_SITE, _P_NAME, _P_FREE, _P_TRY):
                        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    if col == _P_FREE:
                        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                        item.setCheckState(Qt.CheckState.Checked if c["free"]
                                           else Qt.CheckState.Unchecked)
                        item.setToolTip("Tick to let the spinning rate vary: the recorded rate "
                                        "is often off by a little, and the sidebands are very "
                                        "sensitive to it. Fixed by default.")
                    if col in (_P_MIN, _P_MAX):
                        item.setToolTip(f"Hz. Blank: the start ± {MAS_WINDOW_HZ:g} Hz "
                                        f"({start - MAS_WINDOW_HZ:g} to {start + MAS_WINDOW_HZ:g}).")
                    if col == _P_NAME:
                        item.setData(Qt.ItemDataRole.UserRole, "mas_Hz")
                    t.setItem(len(rows), col, item)
            for r, (site, name) in enumerate(rows):
                c = self._constraint(site["label"], name)
                lo, hi = cfit._default_bounds(site, name)
                cells = [site["label"], _param_label(name), f"{cfit._get(site, name):.6g}", "",
                         c["min"], c["max"], c["try"]]
                for col, text in enumerate(cells):
                    item = QTableWidgetItem(text)
                    if col in (_P_SITE, _P_NAME, _P_FREE):
                        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    if col == _P_FREE:
                        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                        item.setCheckState(Qt.CheckState.Checked if c["free"] else Qt.CheckState.Unchecked)
                    if col in (_P_MIN, _P_MAX):
                        item.setToolTip(f"Blank: {lo:g} to {hi:g}.")
                    if col == _P_NAME:
                        item.setData(Qt.ItemDataRole.UserRole, name)
                    t.setItem(r, col, item)
        finally:
            t.blockSignals(False)
            self._refreshing = False

    def _constraint(self, label: str, name: str) -> dict:
        per = self.constraints.setdefault(label, {})
        if name not in per:
            per[name] = {"free": name not in ("alpha", "beta", "gamma"), "min": "", "max": "",
                         "try": ""}
        return per[name]

    def _on_parameter_edited(self, item) -> None:
        r, col = item.row(), item.column()
        label_item, name_item = self.parameter_table.item(r, _P_SITE), self.parameter_table.item(r, _P_NAME)
        if label_item is None or name_item is None:
            return
        label, name = label_item.text(), name_item.data(Qt.ItemDataRole.UserRole)
        if label == SPECTRUM_ROW:
            c = self.mas_constraint()
            if col == _P_START:
                try:
                    self.mas_spin.setValue(float(item.text()))
                except ValueError:
                    self.refresh_parameter_table()
                return
        else:
            c = self._constraint(label, name)
        if col == _P_FREE:
            c["free"] = item.checkState() == Qt.CheckState.Checked
        elif col == _P_MIN:
            c["min"] = item.text().strip()
        elif col == _P_MAX:
            c["max"] = item.text().strip()
        elif col == _P_TRY:
            c["try"] = item.text().strip()
        elif col == _P_START:
            site = next((s for s in self.site_dicts if s["label"] == label), None)
            try:
                value = float(item.text())
            except ValueError:
                self.refresh_parameter_table()
                return
            if site is not None:
                cfit._set(site, name, value)
                if site is self.current_site():
                    self._on_site_selected(self._current)  # show it in the site editor
                self.schedule_preview()

    # ---- fitting --------------------------------------------------------------

    def fit_regions(self):
        mode = self.window_combo.currentData()
        if mode == "all":
            return None
        if mode == "range":
            lo, hi = self.window_lo_spin.value(), self.window_hi_spin.value()
            return [(min(lo, hi), max(lo, hi))]
        regions = self._get_regions() or None
        if not regions:
            raise ValueError("Fit window is 'Integration regions', but there are none -- draw "
                             "them under Fit > Integration, or choose another window.")
        return regions

    def fit_arguments(self) -> dict:
        """What the fit is called with: sites, vary, bounds, starts and the
        settings. Raises ValueError, naming the cell, for anything unreadable."""
        self.refresh_parameter_table()
        sites = self.build_sites()
        vary, bounds, starts = {}, {}, {}
        for site, name in self._parameter_rows():
            label = site["label"]
            c = self._constraint(label, name)
            if c["free"]:
                vary.setdefault(label, []).append(name)
            else:
                vary.setdefault(label, [])
            lo_default, hi_default = cfit._default_bounds(site, name)
            try:
                lo, hi = _parse_float(c["min"]), _parse_float(c["max"])
            except ValueError:
                raise ValueError(f"{label}, {_param_label(name)}: Min and Max must be numbers "
                                 "(or blank).") from None
            if lo is not None or hi is not None:
                lo = lo_default if lo is None else lo
                hi = hi_default if hi is None else hi
                if lo >= hi:
                    raise ValueError(f"{label}, {_param_label(name)}: Min must be below Max.")
                bounds.setdefault(label, {})[name] = (lo, hi)
            if c["try"]:
                try:
                    values = [float(v) for v in c["try"].replace(";", ",").split(",") if v.strip()]
                except ValueError:
                    raise ValueError(f"{label}, {_param_label(name)}: Try starts must be numbers "
                                     "separated by commas.") from None
                if values:
                    starts.setdefault(label, {})[name] = [cfit._get(site, name)] + [
                        v for v in values if v != cfit._get(site, name)]
        settings = self.settings()
        settings.update(self.mas_fit_arguments())
        return {"sites": sites, "vary": vary, "bounds": bounds, "starts": starts,
                "settings": settings, "fit_regions": self.fit_regions(),
                "fit_baseline": self.fit_baseline_check.isChecked(),
                "estimate_amounts": self.estimate_amounts_check.isChecked()}

    def fit(self) -> bool:
        try:
            spectrum, row, index = self._get_target()
            from ..notebook import bridge
            results = bridge.results_from_spectrum(spectrum, row=row)
            args = self.fit_arguments()
        except (ValueError, IndexError, KeyError) as exc:
            QMessageBox.warning(self, "CSA, Quad & J-coupling fit", str(exc))
            return False
        self._preview_timer.stop()
        kwargs = dict(vary=args["vary"], bounds=args["bounds"] or None,
                      fit_regions=args["fit_regions"], fit_baseline=args["fit_baseline"],
                      estimate_amounts=args["estimate_amounts"], **args["settings"])
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        started = time.perf_counter()
        with FitProgress(self, "CSA, Quad & J-coupling fit") as fit_progress_box:
            try:
                if args["starts"]:
                    fit = cfit.fit_combined_multistart(
                        results, args["sites"], args["starts"], verbose=False,
                        screen_n_zcw=self.screen_orientations_spin.value() or None, **kwargs)
                else:
                    fit = cfit.fit_combined(results, args["sites"], **kwargs)
            except Exception as exc:  # noqa: BLE001 -- a fit that fails is an outcome, not a crash
                QApplication.restoreOverrideCursor()
                QMessageBox.warning(self, "CSA, Quad & J-coupling fit", f"The fit did not succeed:\n\n{exc}")
                return False
        if fit_progress_box.stopped:
            return False
        QApplication.restoreOverrideCursor()
        fit["elapsed"] = time.perf_counter() - started
        fit["spectrum"] = spectrum
        fit["spectrum_index"] = index
        fit["site_labels"] = [s.label for s in args["sites"]]
        # The slice of a pseudo-2D this was fitted to, and the window --
        # what applying the model to every slice starts from.
        fit["row"] = row
        fit["fit_regions"] = args["fit_regions"]
        fit["advice"] = self.make_advice(fit)
        self.result = fit
        self._report()
        self._draw_fit(fit)
        self._sync_result_buttons()
        self._on_fitted()
        return True

    # ---- starting values estimated from the spectrum -------------------------

    _ESTIMATE_KINDS = (("isotropic", "Isotropic peaks (replaces every peak)"),
                       ("csa", "CSA (the selected peak)"),
                       ("quadrupolar", "Quadrupolar (the selected peak)"),
                       ("jcoupling", "J-coupling multiplet (the selected peak)"))

    def _build_estimate_box(self) -> QGroupBox:
        """Explicit user request, 2026-10-06: estimate good starting values
        for one type of interaction. notebook.start_estimates."""
        box = QGroupBox("Estimate starting values from the spectrum")
        form = QFormLayout(box)
        note = QLabel("Reads the starting values off the data in the fit window: peaks by "
                      "peak picking; a CSA or quadrupolar site from the pattern's centre of "
                      "mass and spread (matched by quick simulations); a multiplet from its "
                      "line spacing. For one anisotropic site among several, set the fit "
                      "window around it first.")
        note.setWordWrap(True)
        form.addRow(note)
        self.estimate_kind_combo = QComboBox()
        for key, text in self._ESTIMATE_KINDS:
            self.estimate_kind_combo.addItem(text, key)
        self.estimate_kind_combo.currentIndexChanged.connect(lambda _i: self._sync_estimate())
        form.addRow("Model:", self.estimate_kind_combo)
        self.estimate_peaks_spin = QSpinBox()
        self.estimate_peaks_spin.setRange(1, 50)
        self.estimate_peaks_spin.setValue(8)
        self.estimate_peaks_spin.setToolTip("At most this many peaks, tallest first.")
        form.addRow("Up to peaks:", self.estimate_peaks_spin)
        self.estimate_partner_combo = QComboBox()
        self.estimate_partner_combo.setEditable(True)
        self.estimate_partner_combo.addItems(_nuclides())
        self.estimate_partner_combo.setCurrentText("1H")
        form.addRow("Coupled to:", self.estimate_partner_combo)
        self._estimate_form = form
        row = QHBoxLayout()
        self.estimate_button = QPushButton("Estimate Starting Values")
        self.estimate_button.clicked.connect(lambda: self.estimate_starting_values())
        self.undo_estimate_button = QPushButton("Undo")
        self.undo_estimate_button.setToolTip("Put back the peaks as they were before the "
                                             "estimate.")
        self.undo_estimate_button.setEnabled(False)
        self.undo_estimate_button.clicked.connect(self.undo_estimate)
        row.addWidget(self.estimate_button, 1)
        row.addWidget(self.undo_estimate_button)
        form.addRow(row)
        self.estimate_label = QLabel("")
        self.estimate_label.setWordWrap(True)
        self.estimate_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        form.addRow(self.estimate_label)
        self._before_estimate = None
        self._sync_estimate()
        return box

    def _sync_estimate(self) -> None:
        kind = self.estimate_kind_combo.currentData()
        self._estimate_form.setRowVisible(self.estimate_peaks_spin, kind == "isotropic")
        self._estimate_form.setRowVisible(self.estimate_partner_combo, kind == "jcoupling")

    def estimate_starting_values(self, quiet: bool = False) -> dict | None:
        """Estimate the chosen model's starting values from the selected
        spectrum, inside the fit window, and put them into the peaks: every
        peak for isotropic peaks, otherwise the selected peak (keeping its
        label). Returns the estimate (notes included), or None."""
        kind = self.estimate_kind_combo.currentData()
        try:
            spectrum, row, _index = self._get_target()
            from ..notebook import bridge
            results = bridge.results_from_spectrum(spectrum, row=row)
            settings = self.settings()
            est = start_estimates.estimate_start(
                results, kind, nucleus=self.nucleus_combo.currentText().strip(),
                larmor_Hz=settings["larmor_Hz"], mas_Hz=settings["mas_Hz"],
                order=settings["order"], transitions=settings["transitions"],
                fit_regions=self.fit_regions(), max_peaks=self.estimate_peaks_spin.value(),
                partner=self.estimate_partner_combo.currentText().strip() or "1H",
                lineshape=(self.current_site() or {}).get("lineshape", "gaussian"))
        except (ValueError, KeyError, IndexError) as exc:
            self.estimate_label.setText(f"Could not estimate: {exc}")
            if not quiet:
                QMessageBox.information(self, "Estimate starting values", str(exc))
            return None
        self._before_estimate = (copy.deepcopy(self.site_dicts), max(self._current, 0))
        with self._suspend_preview():
            if kind == "isotropic" or not self.site_dicts:
                self._replace_sites(est["sites"])
            else:
                current = self._current if self._current >= 0 else 0
                new = dict(est["sites"][0])
                new["label"] = self.site_dicts[current]["label"]
                self.site_dicts[current] = new
                self.site_list.setCurrentRow(current)
                self._on_site_selected(current)
        self.refresh_parameter_table()
        self.undo_estimate_button.setEnabled(True)
        self.estimate_label.setText("\n".join(f"• {n}" for n in est["notes"]))
        self.schedule_preview()
        return est

    def _replace_sites(self, dicts) -> None:
        self.site_list.blockSignals(True)
        self.site_list.clear()
        self.site_list.blockSignals(False)
        self.site_dicts = []
        self._current = -1
        with self.bulk():
            CombinedHamiltonianPanel.add_sites(self, list(dicts))
            if self.site_dicts:
                self.site_list.setCurrentRow(0)

    def undo_estimate(self) -> None:
        if self._before_estimate is None:
            return
        dicts, current = self._before_estimate
        self._before_estimate = None
        with self._suspend_preview():
            self._replace_sites(dicts)
            if self.site_dicts:
                self.site_list.setCurrentRow(min(current, len(self.site_dicts) - 1))
        self.refresh_parameter_table()
        self.undo_estimate_button.setEnabled(False)
        self.estimate_label.setText("Estimate undone: the peaks are as they were.")

    # ---- live preview of the starting model --------------------------------

    def schedule_preview(self) -> None:
        """Recompute the preview after a short quiet spell -- only while
        the panel is on screen, previewing, and not being filled by code."""
        if (not hasattr(self, "preview_check") or self._preview_suspended
                or not self.preview_check.isChecked() or not self.isVisible()):
            return
        self._preview_timer.start()

    def preview_orientations(self) -> int:
        n = self.preview_orientations_spin.value()
        if n:
            return n
        return self.PREVIEW_ZCW[1 if self.regime_combo.currentData() == "mas" else 0]

    def preview_now(self, quiet: bool = True) -> bool:
        """Compute and draw the starting model over the selected spectrum
        (:func:`~opennmr.notebook.combined_fitting.starting_model`, the
        fit's own first step). ``quiet``: a problem (no spectrum yet, a
        half-typed value) goes to the preview line, not a dialog."""
        try:
            spectrum, row, index = self._get_target()
            from ..notebook import bridge
            results = bridge.results_from_spectrum(spectrum, row=row)
            sites = self.build_sites()
            settings = dict(self.settings())
            settings["n_zcw"] = self.preview_orientations()
            regions = self.fit_regions()
            guess = cfit.starting_model(
                results, sites, fit_regions=regions,
                estimate_amounts=self.estimate_amounts_check.isChecked(), **settings)
        except (ValueError, IndexError, KeyError) as exc:
            self.preview_label.setText(f"Preview: {exc}")
            if not quiet:
                QMessageBox.warning(self, "Preview starting model", str(exc))
            return False
        guess.update(settings={"fit_regions": regions}, spectrum=spectrum,
                     y_fit=guess["start"], title="Starting model (preview)",
                     fit_label="Starting model")
        self.preview = guess
        self._draw_fit(guess)
        amounts = ", ".join(f"{k} {v:.3g}" for k, v in guess["amounts"].items())
        self.preview_label.setText(
            f"Preview: the model the fit starts from ({guess['n_zcw']:,} orientations; the "
            f"fit uses {cfit_n_zcw(self.settings()):,}). Amounts "
            + ("estimated from the data" if self.estimate_amounts_check.isChecked()
               else "as typed") + f": {amounts}.")
        return True

    def _suspend_preview(self):
        """Context: changes made by code do not redraw the preview."""
        panel = self

        class _Suspend:
            def __enter__(self_inner):
                panel._preview_suspended += 1
                panel._preview_timer.stop()

            def __exit__(self_inner, *exc):
                panel._preview_suspended -= 1
                panel._preview_timer.stop()
        return _Suspend()

    # ---- advice -----------------------------------------------------------

    def make_advice(self, fit) -> list:
        alternatives = []
        if self._get_alternatives is not None:
            try:
                alternatives = list(self._get_alternatives(fit) or [])
            except Exception:  # noqa: BLE001 -- a comparison must never sink a fit
                alternatives = []
        return fit_advice.advise_combined_fit(fit, alternatives=alternatives,
                                              nucleus_spin=self.spin())

    def _select_site(self, label) -> bool:
        for k, site in enumerate(self.site_dicts):
            if site["label"] == label:
                self.site_list.setCurrentRow(k)
                return True
        return False

    def apply_advice_action(self, action: dict) -> str | None:
        """Carry out one of the advice's suggestions on the model; returns
        what was done, in words (None if it could not be)."""
        kind = action.get("type")
        site = action.get("site")
        if kind == "add_peak":
            lineshape = self.site_dicts[0]["lineshape"] if self.site_dicts else "gaussian"
            new = self.add_site({"label": self._unique_label(f"Site {len(self.site_dicts) + 1}"),
                                 "delta_iso": float(action["delta_iso"]), "amount": 1.0,
                                 "fwhm": float(action.get("fwhm") or 1.0),
                                 "lineshape": lineshape, "eta": 0.5, "interactions": []})
            return f"added {new['label']} at {new['delta_iso']:g} ppm."
        if kind == "free_mas":
            if self.regime_combo.currentData() != "mas":
                return None
            self.mas_constraint()["free"] = True
            self.refresh_parameter_table()
            return "the MAS rate is free to vary (±500 Hz unless you set Min/Max)."
        if kind == "widen_bound" and action.get("name") == "mas_Hz":
            self.mas_constraint()["min" if action["side"] == "lower" else "max"] = \
                f"{action['value']:.6g}"
            self.refresh_parameter_table()
            return f"the MAS rate's {'Min' if action['side'] == 'lower' else 'Max'} is now " \
                f"{action['value']:.6g} Hz."
        if site is None or not self._select_site(site):
            return None
        if kind == "add_interaction":
            added = self.add_interaction(action["kind"])
            return f"added a {_TITLES[action['kind']]} to {site}." if added else None
        if kind == "set_lineshape":
            self.lineshape_combo.setCurrentIndex(self.lineshape_combo.findData(action["lineshape"]))
            return f"{site} now has a pseudo-Voigt line shape."
        if kind == "remove_site":
            self.remove_site()
            return f"removed {site}."
        name = action.get("name")
        if kind in ("free_parameter", "fix_parameter"):
            names = ["alpha", "beta", "gamma"] if name in ("alpha", "beta", "gamma") else [name]
            for n in names:
                self._constraint(site, n)["free"] = kind == "free_parameter"
            self.refresh_parameter_table()
            what = "the Euler angles" if len(names) > 1 else _param_label(name)
            return f"{what} of {site} {'free' if kind == 'free_parameter' else 'fixed'}."
        if kind == "widen_bound":
            key = "min" if action["side"] == "lower" else "max"
            self._constraint(site, name)[key] = f"{action['value']:.6g}"
            self.refresh_parameter_table()
            return f"{_param_label(name)} of {site}: {key.title()} is now {action['value']:.6g}."
        return None

    def _sync_result_buttons(self) -> None:
        on = self.result is not None
        self.use_fitted_button.setEnabled(on)
        self.send_button.setEnabled(on and self._send_to_simulation is not None)
        if hasattr(self, "correlation_button"):
            # A new (or no) result is drawn as its fit first.
            self.correlation_button.blockSignals(True)
            self.correlation_button.setChecked(False)
            self.correlation_button.blockSignals(False)
            self.correlation_button.setEnabled(
                on and self._draw_correlations is not None
                and self.result.get("pcov") is not None)

    # ---- results -----------------------------------------------------------

    def _report(self) -> None:
        """The fitted values as a table -- a row per site (and one for
        the spectrum's own MAS rate and baseline), a column per
        parameter -- not a single column of "site: parameter" rows
        (explicit user request, 2026-10-08: "Fitted values should be a
        table (like they were previously), not a list")."""
        r = self.result
        cells: dict = {}                    # row -> {column: text}
        columns: list = []

        def put(row, column, text):
            if column not in columns:
                columns.append(column)
            cells.setdefault(row, {})[column] = text

        for p in r["parameters"]:
            if p["name"] == "mas_Hz" or not p["label"]:
                row = SPECTRUM_ROW
                column = (_param_label("mas_Hz") if p["name"] == "mas_Hz"
                          else f"baseline {p['name']}")
            else:
                row, column = p["label"], _param_label(p["name"])
            put(row, column, self._fmt(float(p["value"]),
                                       float(p["error"]) if p["error"] is not None else 0.0,
                                       not p["free"]))
        for site_label, d in r["derived"].items():
            put(site_label, "Share (%)", self._fmt(d["share_%"], d["share_%_err"], False))
            if "P_Q" in d:
                put(site_label, "P_Q (MHz)", self._fmt(d["P_Q"], d["P_Q_err"], False))
            if "delta_cg" in d:
                put(site_label, "CT c.o.g. (ppm)",
                    self._fmt(d["delta_cg"], d["delta_cg_err"], False))
        # Sites first, in the fit's order; the spectrum's own row last.
        row_names = [k for k in cells if k != SPECTRUM_ROW]
        if SPECTRUM_ROW in cells:
            row_names.append(SPECTRUM_ROW)
        t = self.results_table
        t.clear()
        t.setColumnCount(len(columns))
        t.setHorizontalHeaderLabels(columns)
        t.setRowCount(len(row_names))
        t.setVerticalHeaderLabels(row_names)
        for i, row in enumerate(row_names):
            for j, column in enumerate(columns):
                text = cells[row].get(column)
                item = QTableWidgetItem("—" if text is None else text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                if text is None:
                    item.setToolTip(f"{column} does not apply to {row}.")
                t.setItem(i, j, item)
        t.resizeColumnsToContents()
        t.setVisible(True)
        q = r["quality"]
        head = r["log"].splitlines()
        lines = [head[0].replace("CSA, Quad & J-coupling fit: ", "CSA, Quad & J-coupling fit — ")]
        lines += [line for line in head[1:] if line.startswith("Multi-start")]
        lines.append(f"R² {q['r2']:.5f}  ·  RMSE {q['rmse']:.3g}  ·  reduced χ² "
                     f"{q['reduced_chi2']:.3g}  ·  {r.get('elapsed', 0.0):.1f} s. |C_Q| reported "
                     "(its sign is not in a lineshape).")
        lines += [f"⚠ {w}" for w in r["warnings"]]
        self.results_label.setText("\n".join(lines))
        self.advice_box.set_items(r.get("advice") or [])

    def text_report(self) -> str:
        if not self.result:
            return ""
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            return cfit.print_combined_fit_report(self.result)

    def _on_results_menu(self, pos) -> None:
        menu = QMenu(self.results_table)
        copy_table = menu.addAction("Copy Table")
        copy_report = menu.addAction("Copy Full Report")
        chosen = menu.exec(self.results_table.viewport().mapToGlobal(pos))
        if chosen is copy_table:
            QGuiApplication.clipboard().setText(panel_state.table_text(self.results_table, "\t"))
        elif chosen is copy_report:
            QGuiApplication.clipboard().setText(self.text_report())

    def fitted_site_dicts(self) -> list[dict]:
        """The fitted sites in the panel's own format (no nucleus)."""
        out = []
        observed = self.observed()
        for s in self.result["sites"]:
            d = s.to_dict() if hasattr(s, "to_dict") else dict(s)
            own = {k: v for k, v in d.items() if k not in ("nucleus", "spin")}
            if d.get("nucleus") and d["nucleus"] != observed:
                own["isotope"] = d["nucleus"]
            out.append(own)
        return out

    def use_fitted_as_start(self) -> None:
        if not self.result:
            return
        with self._suspend_preview():
            self._use_fitted_as_start()

    def _use_fitted_as_start(self) -> None:
        if (self.result.get("settings") or {}).get("fit_mas"):
            self.mas_spin.setValue(float(self.result["settings"]["mas_Hz"]))
        current = max(self._current, 0)
        fitted = self.fitted_site_dicts()
        self.site_list.blockSignals(True)
        self.site_list.clear()
        self.site_list.blockSignals(False)
        self.site_dicts = []
        self._current = -1
        with self.bulk():
            CombinedHamiltonianPanel.add_sites(self, fitted)
            self.site_list.setCurrentRow(min(current, len(self.site_dicts) - 1))

    def set_start_from_simulation(self, site_dicts, settings: dict, nucleus: str) -> str:
        """Simulated sites as the starting model, with the simulation's
        regime, spinning, level of theory and transitions. The nucleus and
        field stay the spectrum's; a mismatch is returned as a note."""
        with self._suspend_preview():
            return self._set_start_from_simulation(site_dicts, settings, nucleus)

    def _set_start_from_simulation(self, site_dicts, settings: dict, nucleus: str) -> str:
        notes = []
        self.site_list.blockSignals(True)
        self.site_list.clear()
        self.site_list.blockSignals(False)
        self.site_dicts = []
        self._current = -1
        with self.bulk():
            CombinedHamiltonianPanel.add_sites(self, list(site_dicts))
            if self.site_dicts:
                self.site_list.setCurrentRow(0)
        for combo, key in ((self.regime_combo, "regime"), (self.order_combo, "order"),
                           (self.transitions_combo, "transitions")):
            index = combo.findData(settings.get(key))
            if index >= 0:
                combo.setCurrentIndex(index)
        if settings.get("regime") == "mas" and settings.get("mas_Hz"):
            self.mas_spin.setValue(float(settings["mas_Hz"]))
        mine = self.nucleus_combo.currentText().strip()
        if nucleus and mine and nucleus != mine:
            notes.append(f"the simulation is of {nucleus}, the fit of {mine}")
        elif nucleus and not mine:
            self.nucleus_combo.setCurrentText(nucleus)
        larmor = float(settings.get("larmor_MHz") or 0)
        if larmor and self.larmor_spin.value() > 0 and \
                abs(larmor - self.larmor_spin.value()) > 1e-3 * self.larmor_spin.value():
            notes.append(f"the simulation used ν0 = {larmor:.6g} MHz, the fit uses "
                         f"{self.larmor_spin.value():.6g} MHz")
        elif larmor and self.larmor_spin.value() <= 0:
            self.larmor_spin.setValue(larmor)
        self.refresh_parameter_table()
        return "; ".join(notes)

    def _on_send(self) -> None:
        if self.result and self._send_to_simulation is not None:
            self._send_to_simulation(dict(self.result["settings"]), self.fitted_site_dicts(),
                                     self.nucleus_combo.currentText().strip())

    # ---- saved state (processing log / Load Progress) -----------------------

    def get_state(self) -> dict:
        state = super().get_state()
        state["widgets"].pop("parameter_table", None)  # rebuilt from the sites + constraints
        state["constraints"] = copy.deepcopy(self.constraints)
        for site in self.site_dicts:
            bits = []
            for name in cfit.site_parameter_names(site):
                c = self._constraint(site["label"], name)
                bit = f"{_param_label(name)} {'free' if c['free'] else 'fixed'}"
                if c["min"] or c["max"]:
                    bit += f" in [{c['min'] or '…'}, {c['max'] or '…'}]"
                if c["try"]:
                    bit += f", also tried from {c['try']}"
                bits.append(bit)
            state["extra"][f"{site['label']} parameters"] = "; ".join(bits)
        if self._mas_row_shown():
            c = self.mas_constraint()
            state["extra"]["MAS rate"] = (
                f"free, {c['min'] or f'start − {MAS_WINDOW_HZ:g}'} to "
                f"{c['max'] or f'start + {MAS_WINDOW_HZ:g}'} Hz" if c["free"]
                else "fixed at the value given")
        return state

    def set_state(self, state: dict) -> None:
        with self._suspend_preview():
            self.constraints = copy.deepcopy((state or {}).get("constraints") or {})
            super().set_state(state)
            self._sync_fit_controls()
            self.refresh_parameter_table()

    def result_state(self):
        if not self.result:
            return None
        r = {k: v for k, v in self.result.items() if k not in ("spectrum", "sites")}
        r["site_dicts"] = [s.to_dict() for s in self.result["sites"]]
        return serialize.encode(r)

    def restore_result(self, encoded, spectrum=None) -> None:
        r = serialize.decode(encoded)
        r["sites"] = [ch.Site.from_dict(d) for d in r.pop("site_dicts")]
        r["spectrum"] = spectrum
        self.result = r
        self._report()
        self._sync_result_buttons()

    def clear_result(self) -> None:
        self.result = None
        self.results_label.setText("")
        self.results_table.clear()
        self.results_table.setRowCount(0)
        self.results_table.setVisible(False)
        self.advice_box.set_items([])
        self._sync_result_buttons()

    def report_text(self) -> str:
        if not self.result:
            return ""
        advice = fit_advice.advice_text(self.result.get("advice") or [])
        return (self.results_label.text() + "\n" + panel_state.table_text(self.results_table)
                + ("\n" + advice if advice else ""))
