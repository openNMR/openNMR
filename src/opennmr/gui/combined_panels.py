"""Simulate > Combined Hamiltonian: sites with any mix of CSA,
quadrupolar and J interactions, static or MAS.

A list of sites on the left; the selected site's shift, amount and
broadening beneath it, and its interactions -- each added with "Add
interaction" and removed with its own button, so a combination is
built by adding, never chosen from a list of combinations. The physics
is :mod:`opennmr.notebook.combined_hamiltonian`; "Copy as Python" gives
the notebook code for the same simulation.
"""

from __future__ import annotations

import copy
import time

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout,
    QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QListWidget, QMenu, QMessageBox,
    QPushButton, QSpinBox, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from ..core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
from . import panel_state
from ..notebook.simulations import CSA_ANISOTROPY_TIP, CSA_ASYMMETRY_TIP
from ..io import serialize
from ..notebook import combined_fitting as cfit
from ..notebook import combined_hamiltonian as ch
from ..notebook.utils import _GAMMA_DF, find_isotope_info

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

    def __init__(self, data: dict, on_remove, has_quadrupolar: bool, parent=None) -> None:
        super().__init__(_TITLES[data["type"]], parent)
        self.data = data
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

    def _write_csa(self, *_):
        self.data.update(convention=self.convention.currentData(), aniso=self.aniso.value(),
                         asym=self.asym.value(), euler=tuple(b.value() for b in self.euler))

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

    DEFAULT_INTERACTIONS = {
        "csa": {"type": "csa", "aniso": 100.0, "asym": 0.5, "convention": "Haeberlen",
                "euler": (0.0, 0.0, 0.0)},
        "quadrupolar": {"type": "quadrupolar", "C_Q": 3.0, "eta_Q": 0.5},
        "jcoupling": {"type": "jcoupling", "nuclide": "1H", "n": 1, "J_Hz": 100.0},
    }

    def __init__(self, plot_spectrum, add_to_open, draw_sites=None, parent=None,
                 send_to_fit=None) -> None:
        super().__init__(parent)
        self._plot_spectrum = plot_spectrum
        self._add_to_open = add_to_open
        self._send_to_fit = send_to_fit
        self._draw_sites = draw_sites
        self._seeded_for = None
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
        self.nucleus_combo.currentTextChanged.connect(lambda _t: self._sync_spin())
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
        sites_box = QGroupBox("Sites")
        sites_layout = QHBoxLayout(sites_box)
        self.site_list = QListWidget()
        self.site_list.setMaximumHeight(110)
        self.site_list.currentRowChanged.connect(self._on_site_selected)
        sites_layout.addWidget(self.site_list, 1)
        buttons = QVBoxLayout()
        self.add_site_button = QPushButton("Add site")
        self.add_site_button.clicked.connect(lambda: self.add_site())
        self.duplicate_button = QPushButton("Duplicate")
        self.duplicate_button.clicked.connect(self.duplicate_site)
        self.remove_site_button = QPushButton("Remove")
        self.remove_site_button.clicked.connect(self.remove_site)
        for b in (self.add_site_button, self.duplicate_button, self.remove_site_button):
            buttons.addWidget(b)
        buttons.addStretch(1)
        sites_layout.addLayout(buttons)
        v.addWidget(sites_box)

        # ---- the selected site ----
        self.site_box = QGroupBox("Selected site")
        site_layout = QVBoxLayout(self.site_box)
        sform = QFormLayout()
        self.label_edit = QLineEdit()
        self.label_edit.textEdited.connect(self._on_label_edited)
        sform.addRow("Label:", self.label_edit)
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
        self.interactions_layout = QVBoxLayout()
        site_layout.addLayout(self.interactions_layout)
        self.no_interactions_label = QLabel("No interactions yet: an isotropic line. Add one below.")
        self.no_interactions_label.setWordWrap(True)
        site_layout.addWidget(self.no_interactions_label)
        self.add_interaction_button = QPushButton("Add interaction ▾")
        self.add_interaction_button.setToolTip("Add a CSA, a quadrupolar coupling or a J-coupling "
                                               "to this site.")
        self._interaction_menu = QMenu(self)
        self._actions = {}
        for kind in ch.INTERACTION_TYPES:
            action = self._interaction_menu.addAction(_TITLES[kind])
            action.triggered.connect(lambda _c=False, k=kind: self.add_interaction(k))
            self._actions[kind] = action
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
        Simulate/Export; Fit > Combined Hamiltonian replaces it."""
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
        self.to_fit_button.setToolTip("Put these sites into Fit ▸ Combined Hamiltonian as the "
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

    def _sync_menu(self):
        site = self.current_site()
        spin = self.spin() or 0.0
        has = {k: bool(site and any(i["type"] == k for i in site["interactions"]))
               for k in ch.INTERACTION_TYPES}
        self._actions["csa"].setEnabled(site is not None and not has["csa"])
        self._actions["quadrupolar"].setEnabled(site is not None and not has["quadrupolar"]
                                                and spin >= 1)
        self._actions["quadrupolar"].setToolTip(
            "" if spin >= 1 else "The observed nucleus is spin-1/2: no quadrupole moment.")
        self._actions["jcoupling"].setEnabled(site is not None)

    # ---- saved state (simulation history) -----------------------------

    #: The "Selected site" fields show one site of site_dicts; the model
    #: itself is saved, not these.
    _SITE_FIELDS = ("label_edit", "diso_spin", "amount_spin", "fwhm_spin", "lineshape_combo",
                    "pv_eta_spin")

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
        for site in state.get("sites") or []:
            self.add_site(site)
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
        del self.site_dicts[row]
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
            site["label"] = text
            self.site_list.item(self._current).setText(text)

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
        self.label_edit.setText(site["label"])
        self.diso_spin.setValue(site["delta_iso"])
        self.amount_spin.setValue(site["amount"])
        self.fwhm_spin.setValue(site["fwhm"])
        self.lineshape_combo.setCurrentIndex(self.lineshape_combo.findData(site["lineshape"]))
        self.pv_eta_spin.setValue(site.get("eta", 0.5))
        for w in widgets:
            w.blockSignals(False)
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
        if kind == "quadrupolar" and (self.spin() or 0) < 1:
            QMessageBox.information(self, "Add interaction",
                                    "The observed nucleus is spin-1/2: it has no quadrupole moment.")
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
        has_quad = any(i["type"] == "quadrupolar" for i in interactions)
        order = {"quadrupolar": 0, "csa": 1, "jcoupling": 2}
        self.interaction_editors = []
        for data in sorted(interactions, key=lambda i: order[i["type"]]):
            editor = InteractionEditor(data, self._remove_interaction, has_quad)
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
        return [ch.Site.from_dict({**s, "nucleus": nucleus}) for s in self.site_dicts]

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
        return {"larmor_Hz": larmor * 1e6, "mas_Hz": mas,
                "order": self.order_combo.currentData(), "transitions": transitions,
                "n_zcw": self.orientations_spin.value() or None}

    def simulate(self) -> bool:
        try:
            sites = self.build_sites()
            settings = self.settings()
            if self.auto_axis_check.isChecked():
                lo, hi = ch.combined_axis_limits(sites, settings["larmor_Hz"],
                                                 transitions=settings["transitions"],
                                                 mas_Hz=settings["mas_Hz"],
                                                 order=settings["order"])
                for box, value in ((self.ppm_min_spin, lo), (self.ppm_max_spin, hi)):
                    box.blockSignals(True)
                    box.setValue(value)
                    box.blockSignals(False)
            if self.ppm_max_spin.value() <= self.ppm_min_spin.value():
                raise ValueError("Axis max must be greater than axis min.")
            axis = np.linspace(self.ppm_min_spin.value(), self.ppm_max_spin.value(),
                               self.points_spin.value())
            started = time.perf_counter()
            result = ch.simulate_combined(sites, axis, **settings)
            elapsed = time.perf_counter() - started
        except (ValueError, KeyError) as exc:
            QMessageBox.warning(self, "Combined Hamiltonian", str(exc))
            return False
        nucleus = sites[0].nucleus
        self.result = {"result": result, "sites": sites, "settings": settings, "axis": axis}
        data = np.asarray(result["real_spec"], dtype=float)
        metadata = NMRMetadata(
            vendor="simulated", nucleus=nucleus, sfo1=settings["larmor_Hz"] / 1e6,
            raw={"simulation": {"kind": "combined", "sites": result["site_dicts"],
                                "settings": result["settings"], "log": result["log"]}})
        self.spectrum = NMRSpectrum(data=data, axes=[NMRAxis(kind="frequency", ppm=axis)],
                                    metadata=metadata)
        self._plot_spectrum(self.spectrum)
        if self._draw_sites is not None and self.show_sites_check.isChecked() and len(sites) > 1:
            self._draw_sites(result, nucleus)
        edge = max(abs(data[0]), abs(data[-1])) / (np.max(np.abs(data)) or 1.0)
        warn = ("\n⚠ Intensity at the axis edge is {:.0%} of the peak -- the pattern is "
                "clipped; widen the axis.".format(edge)) if edge > 0.01 else ""
        self.results_label.setText(f"Simulated in {elapsed:.2f} s.\n{result['log']}{warn}")
        self.export_button.setEnabled(True)
        return True

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


def _describe_site(site: dict) -> str:
    """A site as one readable line, for the simulation log."""
    bits = [f"δiso {site.get('delta_iso', 0):g} ppm", f"amount {site.get('amount', 1):g}",
            f"FWHM {site.get('fwhm', 0):g} ppm ({site.get('lineshape', '')})"]
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
# Fit > Combined Hamiltonian
# ============================================================================

#: Parameter table columns.
_P_SITE, _P_NAME, _P_START, _P_FREE, _P_MIN, _P_MAX, _P_TRY = range(7)
_P_HEADERS = ("Site", "Parameter", "Start", "Free", "Min", "Max", "Try starts")
_PARAM_LABELS = {"delta_iso": "δiso (ppm)", "amount": "Amount", "fwhm": "FWHM (ppm)",
                 "eta": "L fraction", "aniso": "Δ or span (ppm)", "asym": "η or skew",
                 "alpha": "α (°)", "beta": "β (°)", "gamma": "γ (°)", "C_Q": "C_Q (MHz)",
                 "eta_Q": "η_Q"}


def _param_label(name: str) -> str:
    return _PARAM_LABELS.get(name, f"{name} (Hz)" if name.startswith("J") else name)


def _parse_float(text: str):
    text = (text or "").strip()
    if not text:
        return None
    return float(text)


class CombinedFitPanel(CombinedHamiltonianPanel):
    """Fit Combined-Hamiltonian sites to the selected spectrum: the same
    site editor as Simulate > Combined Hamiltonian, plus a table of every
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

    def __init__(self, get_target, get_regions, draw_fit, format_value_error,
                 send_to_simulation=None, on_fitted=None, parent=None) -> None:
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
                         draw_sites=None, parent=parent)
        self.refresh_parameter_table()

    # ---- the part below the site editor ----------------------------------

    def _build_actions(self, v) -> None:
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

        self.fit_button = QPushButton("Fit Combined Hamiltonian")
        self.fit_button.clicked.connect(self.fit)
        v.addWidget(self.fit_button)
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
        row = QHBoxLayout()
        self.use_fitted_button = QPushButton("Use fitted values as start")
        self.use_fitted_button.clicked.connect(self.use_fitted_as_start)
        self.send_button = QPushButton("Open in Simulate")
        self.send_button.setToolTip("Put the fitted sites into Simulate > Combined Hamiltonian.")
        self.send_button.clicked.connect(self._on_send)
        for b in (self.use_fitted_button, self.send_button):
            b.setEnabled(False)
            row.addWidget(b)
        v.addLayout(row)
        self._sync_fit_controls()

    def _sync_fit_controls(self) -> None:
        manual = self.window_combo.currentData() == "range"
        self._fit_form.setRowVisible(self.window_lo_spin, manual)
        self._fit_form.setRowVisible(self.window_hi_spin, manual)

    # ---- keeping the parameter table in step with the sites --------------

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
            for name in cfit.site_parameter_names(site):
                yield site, name

    def refresh_parameter_table(self) -> None:
        """Rebuild the table from the sites, keeping each parameter's
        free/bounds/try choices by (site label, parameter)."""
        if not hasattr(self, "parameter_table") or self._refreshing:
            return
        self._refreshing = True
        t = self.parameter_table
        t.blockSignals(True)
        try:
            rows = list(self._parameter_rows())
            t.setRowCount(len(rows))
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
        return {"sites": sites, "vary": vary, "bounds": bounds, "starts": starts,
                "settings": settings, "fit_regions": self.fit_regions(),
                "fit_baseline": self.fit_baseline_check.isChecked()}

    def fit(self) -> bool:
        try:
            spectrum, row, index = self._get_target()
            from ..notebook import bridge
            results = bridge.results_from_spectrum(spectrum, row=row)
            args = self.fit_arguments()
        except (ValueError, IndexError, KeyError) as exc:
            QMessageBox.warning(self, "Combined Hamiltonian fit", str(exc))
            return False
        kwargs = dict(vary=args["vary"], bounds=args["bounds"] or None,
                      fit_regions=args["fit_regions"], fit_baseline=args["fit_baseline"],
                      **args["settings"])
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        started = time.perf_counter()
        try:
            if args["starts"]:
                fit = cfit.fit_combined_multistart(
                    results, args["sites"], args["starts"], verbose=False,
                    screen_n_zcw=self.screen_orientations_spin.value() or None, **kwargs)
            else:
                fit = cfit.fit_combined(results, args["sites"], **kwargs)
        except Exception as exc:  # noqa: BLE001 -- a fit that fails is an outcome, not a crash
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self, "Combined Hamiltonian fit", f"The fit did not succeed:\n\n{exc}")
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
        self.result = fit
        self._report()
        self._draw_fit(fit)
        self._sync_result_buttons()
        self._on_fitted()
        return True

    def _sync_result_buttons(self) -> None:
        on = self.result is not None
        self.use_fitted_button.setEnabled(on)
        self.send_button.setEnabled(on and self._send_to_simulation is not None)

    # ---- results -----------------------------------------------------------

    def _report(self) -> None:
        r = self.result
        rows = []
        for p in r["parameters"]:
            if not p["label"]:
                label = f"baseline {p['name']}"
            else:
                label = f"{p['label']}: {_param_label(p['name'])}"
            rows.append((label, self._fmt(float(p["value"]),
                                          float(p["error"]) if p["error"] is not None else 0.0,
                                          not p["free"])))
        for site_label, d in r["derived"].items():
            rows.append((f"{site_label}: share (%)", self._fmt(d["share_%"], d["share_%_err"], False)))
            if "P_Q" in d:
                rows.append((f"{site_label}: P_Q (MHz)", self._fmt(d["P_Q"], d["P_Q_err"], False)))
            if "delta_cg" in d:
                rows.append((f"{site_label}: CT c.o.g. (ppm)",
                             self._fmt(d["delta_cg"], d["delta_cg_err"], False)))
        t = self.results_table
        t.clear()
        t.setColumnCount(1)
        t.setHorizontalHeaderLabels(["Value"])
        t.setRowCount(len(rows))
        t.setVerticalHeaderLabels([label for label, _v in rows])
        for i, (_label, text) in enumerate(rows):
            item = QTableWidgetItem(text)
            item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            t.setItem(i, 0, item)
        t.setVisible(True)
        q = r["quality"]
        head = r["log"].splitlines()
        lines = [head[0].replace("Combined-Hamiltonian fit: ", "Combined-Hamiltonian fit — ")]
        lines += [line for line in head[1:] if line.startswith("Multi-start")]
        lines.append(f"R² {q['r2']:.5f}  ·  RMSE {q['rmse']:.3g}  ·  reduced χ² "
                     f"{q['reduced_chi2']:.3g}  ·  {r.get('elapsed', 0.0):.1f} s. |C_Q| reported "
                     "(its sign is not in a lineshape).")
        lines += [f"⚠ {w}" for w in r["warnings"]]
        self.results_label.setText("\n".join(lines))

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
        for s in self.result["sites"]:
            d = s.to_dict() if hasattr(s, "to_dict") else dict(s)
            out.append({k: v for k, v in d.items() if k not in ("nucleus", "spin")})
        return out

    def use_fitted_as_start(self) -> None:
        if not self.result:
            return
        current = max(self._current, 0)
        fitted = self.fitted_site_dicts()
        self.site_list.blockSignals(True)
        self.site_list.clear()
        self.site_list.blockSignals(False)
        self.site_dicts = []
        self._current = -1
        for site in fitted:
            CombinedHamiltonianPanel.add_site(self, site)
        self.site_list.setCurrentRow(min(current, len(self.site_dicts) - 1))
        self.refresh_parameter_table()

    def set_start_from_simulation(self, site_dicts, settings: dict, nucleus: str) -> str:
        """Simulated sites as the starting model, with the simulation's
        regime, spinning, level of theory and transitions. The nucleus and
        field stay the spectrum's; a mismatch is returned as a note."""
        notes = []
        self.site_list.blockSignals(True)
        self.site_list.clear()
        self.site_list.blockSignals(False)
        self.site_dicts = []
        self._current = -1
        for site in copy.deepcopy(list(site_dicts)):
            CombinedHamiltonianPanel.add_site(self, site)
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
        return state

    def set_state(self, state: dict) -> None:
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
        self._sync_result_buttons()

    def report_text(self) -> str:
        if not self.result:
            return ""
        return self.results_label.text() + "\n" + panel_state.table_text(self.results_table)
