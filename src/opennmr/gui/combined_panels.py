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
    QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
    QLineEdit, QListWidget, QMenu, QMessageBox, QPushButton, QSpinBox, QVBoxLayout, QWidget,
)

from ..core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
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
                                   tip="Δ (Haeberlen) or span Ω (Herzfeld-Berger).")
            self.asym = _spin_box(-1.0, 1.0, data["asym"], 3, step=0.1,
                                  tip="η (0..1, Haeberlen) or skew κ (−1..1, Herzfeld-Berger).")
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

    DEFAULT_INTERACTIONS = {
        "csa": {"type": "csa", "aniso": 100.0, "asym": 0.5, "convention": "Haeberlen",
                "euler": (0.0, 0.0, 0.0)},
        "quadrupolar": {"type": "quadrupolar", "C_Q": 3.0, "eta_Q": 0.5},
        "jcoupling": {"type": "jcoupling", "nuclide": "1H", "n": 1, "J_Hz": 100.0},
    }

    def __init__(self, plot_spectrum, add_to_open, draw_sites=None, parent=None) -> None:
        super().__init__(parent)
        self._plot_spectrum = plot_spectrum
        self._add_to_open = add_to_open
        self._draw_sites = draw_sites
        self._seeded_for = None
        self.site_dicts: list[dict] = []
        self._current = -1
        self.result: dict | None = None
        self.spectrum: NMRSpectrum | None = None

        v = QVBoxLayout(self)
        blurb = QLabel(
            "Build each site from its interactions: add a CSA, a quadrupolar coupling and any "
            "number of J-couplings to the same site, in any combination. Static or MAS; every "
            "site is simulated with all of its interactions together and the sites summed.")
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
            "Powder orientation density (auto: 10,946 static, 4,181 MAS). One octant is used "
            "when that is exact; a CSA not aligned with the EFG needs the full sphere.")
        form.addRow("Orientations:", self.orientations_spin)
        v.addLayout(form)

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

        # ---- axis, actions ----
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
        self.export_button = QPushButton("Export to Open Experiments…")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self._on_export)
        self.python_button = QPushButton("Copy as Python")
        self.python_button.setToolTip("Copy notebook code that reproduces this simulation.")
        self.python_button.clicked.connect(self.copy_python)
        row.addWidget(self.export_button)
        row.addWidget(self.python_button)
        v.addLayout(row)
        v.addStretch(1)

        self._sync_spin()
        self._sync_controls()
        self.add_site()

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
            raise ValueError("Set the Larmor frequency.")
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
