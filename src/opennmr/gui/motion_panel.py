"""Dynamics ▸ Anisotropic motion: tensors that rotate about one or several axes, as well as hop.

Three steps, top to bottom:

1. **Interactions** -- the shift, CSA, quadrupolar and dipolar tensors that rotate. Type them in
   (*Add interaction*) or import them from Simulate. Each one is placed relative to rotation
   axis 1 by Euler angles; beta is the angle between axis 1 and the interaction's principal
   z axis.
2. **Rotation axes** -- one row per axis: how many positions, how far apart, which way round
   (both ways, anticlockwise, clockwise or to any position), and its *own* rate or Arrhenius
   barrier. Axis 1 is the reference direction; each further axis is given by its angle from
   axis 1 and its azimuth about it. The first axis listed is the innermost (a methyl turning on
   a group that itself turns: methyl first).
3. **Simulate or fit** (the sections below).

Hops between environments stay under *Advanced*. The lineshape is the exact Bloch-McConnell
solution, static, under MAS and after a Hahn or quadrupolar echo, at rates from slow to fast:
a rate (or temperature) sweep is stacked on the Fit view with the slow and fast limits; one
spectrum can be overlaid on the spectrum selected in Open experiments, scaled onto it, with the
residual. All the maths is :mod:`opennmr.notebook.motion` (:func:`~opennmr.notebook.motion.
rotation_model` builds the same model in a notebook); *Copy notebook code* gives the same
numbers there.
"""

from __future__ import annotations

import math
import re
import warnings

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
    QFormLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMessageBox,
    QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from ..notebook import motion as M
from ..notebook.motion_fit import _parse_name
from ..notebook.combined_hamiltonian import Site
from .dynamics_panels import _copy, _float, _log_box, _show, _text_size
from .widgets.report_label import ReportLabel
from .widgets.typed_spinbox import TypedSpinBox

#: Mode kinds: (menu text, key, what the parameter column means, its default).
MODE_KINDS = (
    ("Ring jump (n-fold)", "ring_jump", "sites n", "3"),
    ("Flip (angle)", "flip", "angle (°)", "180"),
    ("Ring diffusion", "ring_diffusion", "states n", "24"),
    ("Random reorientation", "random_reorientation", "octahedral or tetrahedral", "octahedral"),
)
#: Plain-language motions for the quick set-up: (menu text, mode kind, explanation).
QUICK_MOTIONS = (
    ("Jumps between equivalent positions about an axis", "ring_jump",
     "The group hops in steps of 360°/n about one axis: a CH₃ or CD₃ group (3 positions), a "
     "phenyl ring flipping about its axis (2 positions), a NH₃⁺ group, ClO₄⁻ …"),
    ("Two-position flip", "flip",
     "Swaps between two orientations that differ by a fixed angle about an axis (a 180° flip "
     "of a ring, or a hop between two tilted positions)."),
    ("Continuous rotation about an axis", "ring_diffusion",
     "Smooth rotation about one axis (a fast-spinning rotor, or many tiny steps), modelled as "
     "24 small jumps. The rate is the rotational diffusion rate."),
    ("Tumbling in every direction (liquid-like, isotropic)", "random_reorientation",
     "Reorientation in all directions: anisotropic interactions average to zero when it is fast."),
)
#: Sense of a ring jump: (menu text, Mode model). 'forward' is anticlockwise looking down the axis.
SENSES = (
    ("Both ways (to either neighbour)", "nearest"),
    ("Anticlockwise only", "forward"),
    ("Clockwise only", "backward"),
    ("To any other position", "random"),
)
_PARAM_KEY = {"ring_jump": "n", "flip": "angle_deg", "ring_diffusion": "n",
              "random_reorientation": "group"}


def parse_values(text: str) -> list[float]:
    """'1e2, 1e4 1e6', 'lo:hi:n' (n points, linear) or 'lo:hi:n log' (log-spaced)."""
    text = (text or "").strip().replace("−", "-")
    if not text:
        raise ValueError("Enter at least one value.")
    log = bool(re.search(r"\blog\b", text, re.I))
    text = re.sub(r"\blog\b", " ", text, flags=re.I)
    out: list[float] = []
    for token in re.split(r"[,;\s]+", text):
        if not token:
            continue
        if ":" in token:
            parts = [_float(p) for p in token.split(":")]
            if len(parts) != 3 or any(p is None for p in parts) or int(parts[2]) < 2:
                raise ValueError(f"'{token}' is not lo:hi:n (n ≥ 2).")
            lo, hi, n = parts[0], parts[1], int(parts[2])
            if n > 200:
                raise ValueError(f"'{token}' gives {n} values; the limit is 200.")
            if log:
                if lo <= 0 or hi <= 0:
                    raise ValueError("A log range needs positive limits.")
                out.extend(float(v) for v in np.logspace(math.log10(lo), math.log10(hi), n))
            else:
                out.extend(float(v) for v in np.linspace(lo, hi, n))
        else:
            v = _float(token)
            if v is None:
                raise ValueError(f"'{token}' is not a number.")
            out.append(v)
    if len(out) > 200:
        raise ValueError("At most 200 values.")
    return out


def parse_dipolar(text: str) -> list[dict]:
    """'1H:23.3, 1H@1.09' -> [{'partner': '1H', 'D_Hz': 23300.0}, {'partner': '1H',
    'r_A': 1.09}]: 'partner:D in kHz' or 'partner@distance in Å'."""
    out = []
    for token in re.split(r"[,;]+", (text or "").strip()):
        token = token.strip()
        if not token:
            continue
        m = re.fullmatch(r"(\w+)\s*([:@])\s*([-+0-9.eE]+)", token)
        if not m:
            raise ValueError(f"'{token}' is not partner:D(kHz) or partner@r(Å).")
        value = float(m.group(3))
        out.append({"partner": m.group(1), "D_Hz": value * 1e3} if m.group(2) == ":"
                   else {"partner": m.group(1), "r_A": value})
    return out


def _interaction_text(site) -> str:
    bits = []
    for i in site.interactions:
        if i["type"] == "csa":
            names = ("Δ", "η") if i["convention"] == "Haeberlen" else ("span", "skew")
            bits.append(f"CSA {names[0]} {i['aniso']:g}, {names[1]} {i['asym']:g}")
        elif i["type"] == "quadrupolar":
            bits.append(f"C_Q {i['C_Q']:g} MHz, η_Q {i['eta_Q']:g}")
        else:
            bits.append("J (ignored)")
    return "; ".join(bits)


class InteractionDialog(QDialog):
    """Type in one environment's tensors: shift, CSA, quadrupolar coupling, dipolar partners and
    the orientation of its principal axes relative to rotation axis 1. :meth:`values` gives the
    keyword arguments of :meth:`AnisotropicMotionPanel.add_interaction`."""

    def __init__(self, parent=None, site=None, euler=(0.0, 0.0, 0.0), dipolar="", nucleus="2H"):
        super().__init__(parent)
        self.setWindowTitle("Interaction tensors")
        outer = QVBoxLayout(self)
        form = QFormLayout()
        outer.addLayout(form)
        self.label_edit = QLineEdit("" if site is None else site.label)
        self.label_edit.setPlaceholderText("optional")
        self.nucleus_edit = QLineEdit(nucleus if site is None else site.nucleus)
        self.nucleus_edit.setToolTip("The observed nuclide, e.g. 2H, 13C, 23Na.")
        self.diso_spin = self._spin(-1e5, 1e5, 3, " ppm", 0.0 if site is None else site.delta_iso)
        self.fwhm_spin = self._spin(0.0, 1e4, 3, " ppm", 2.0 if site is None else site.fwhm)
        self.shape_combo = QComboBox()
        self.shape_combo.addItems(["lorentzian", "gaussian"])
        if site is not None and site.lineshape in ("lorentzian", "gaussian"):
            self.shape_combo.setCurrentText(site.lineshape)
        form.addRow("Label:", self.label_edit)
        form.addRow("Nucleus:", self.nucleus_edit)
        form.addRow("Isotropic shift δiso:", self.diso_spin)
        form.addRow("Line width (FWHM):", self.fwhm_spin)
        form.addRow("Line shape:", self.shape_combo)

        csa = None if site is None else site.get("csa")
        quad = None if site is None else site.get("quadrupolar")
        self.csa_box = QGroupBox("Chemical shift anisotropy (Haeberlen)")
        self.csa_box.setCheckable(True)
        self.csa_box.setChecked(bool(csa))
        cf = QFormLayout(self.csa_box)
        aniso, asym, ceuler = 0.0, 0.0, (0.0, 0.0, 0.0)
        if csa:
            c = csa[0]
            aniso, asym = float(c["aniso"]), float(c["asym"])
            if c.get("convention", "Haeberlen") != "Haeberlen":
                from ..notebook.simulations import convert_csa_parameters
                _iso, aniso, asym = convert_csa_parameters(0.0, aniso, asym, c["convention"],
                                                           "Haeberlen")
            ceuler = tuple(c.get("euler", ceuler))
        self.aniso_spin = self._spin(-1e5, 1e5, 3, " ppm", aniso)
        self.asym_spin = self._spin(0.0, 1.0, 3, "", asym)
        cf.addRow("Anisotropy Δ:", self.aniso_spin)
        cf.addRow("Asymmetry η:", self.asym_spin)
        self.csa_euler = [self._spin(-360, 360, 2, "°", v) for v in ceuler]
        self.csa_euler_row = self._row(self.csa_euler)
        cf.addRow("CSA axes in the EFG frame (α, β, γ):", self.csa_euler_row)
        outer.addWidget(self.csa_box)

        self.quad_box = QGroupBox("Quadrupolar coupling")
        self.quad_box.setCheckable(True)
        self.quad_box.setChecked(bool(quad))
        qf = QFormLayout(self.quad_box)
        self.cq_spin = self._spin(-1e3, 1e3, 4, " MHz", float(quad[0]["C_Q"]) if quad else 0.17)
        self.etaq_spin = self._spin(0.0, 1.0, 3, "", float(quad[0]["eta_Q"]) if quad else 0.0)
        qf.addRow("C_Q:", self.cq_spin)
        qf.addRow("η_Q:", self.etaq_spin)
        outer.addWidget(self.quad_box)

        self.dipolar_edit = QLineEdit(dipolar)
        self.dipolar_edit.setPlaceholderText("1H@1.09   (partner@distance in Å)   or   1H:23.3 (kHz)")
        self.dipolar_edit.setToolTip("Dipolar couplings to spin-1/2 partners, comma separated: "
                                     "'partner@r(Å)' or 'partner:D(kHz)'. The bond lies along the "
                                     "interaction's principal z axis.")
        outer.addWidget(QLabel("Dipolar partner(s) (optional):"))
        outer.addWidget(self.dipolar_edit)

        orient = QGroupBox("Orientation relative to rotation axis 1")
        of = QFormLayout(orient)
        self.euler = [self._spin(-360, 360, 2, "°", v) for v in euler]
        of.addRow("Euler angles α, β, γ:", self._row(self.euler))
        note = QLabel("β is the angle between rotation axis 1 and the principal z axis of the "
                      "interaction (the EFG frame, or the CSA frame without a quadrupolar "
                      "coupling; for a C–D bond, between the axis and the bond). With a single "
                      "axis only β matters.")
        note.setWordWrap(True)
        of.addRow(note)
        outer.addWidget(orient)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                   | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._check)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)

    @staticmethod
    def _spin(lo, hi, decimals, suffix, value):
        s = QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setDecimals(decimals)
        s.setSuffix(suffix)
        s.setValue(float(value))
        return s

    @staticmethod
    def _row(widgets):
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        for x in widgets:
            h.addWidget(x)
        return w

    def values(self) -> dict:
        return {
            "nucleus": self.nucleus_edit.text().strip(),
            "delta_iso": self.diso_spin.value(),
            "label": self.label_edit.text().strip() or None,
            "fwhm": self.fwhm_spin.value(), "lineshape": self.shape_combo.currentText(),
            "csa": ((self.aniso_spin.value(), self.asym_spin.value())
                    if self.csa_box.isChecked() else None),
            "csa_euler": tuple(s.value() for s in self.csa_euler),
            "quad": ((self.cq_spin.value(), self.etaq_spin.value())
                     if self.quad_box.isChecked() else None),
            "dipolar": self.dipolar_edit.text().strip(),
            "euler": tuple(s.value() for s in self.euler),
        }

    def _check(self) -> None:
        try:
            v = self.values()
            Site(v["nucleus"], 0.0)                       # an unknown nuclide fails here
            parse_dipolar(v["dipolar"])
            if v["quad"] is not None:
                Site(v["nucleus"], 0.0).add_quadrupolar(*v["quad"])
        except (ValueError, KeyError) as exc:
            QMessageBox.warning(self, "Interaction tensors", str(exc))
            return
        self.accept()


class AnisotropicMotionPanel(QWidget):
    """Dynamics ▸ Anisotropic motion.

    ``canvas`` / ``show_view``: the Fit view, as the other Dynamics panels.
    ``get_spectra()``: the 1D spectra selected in Open experiments, as
    ``[{"label", "ppm", "y", "T_K" (or None), "larmor_MHz", "nucleus"}]``.
    ``get_simulate_sites()``: the Simulate environments (:class:`Site` list) for the observed
    nucleus; ``get_simulate_settings()``: Simulate's ``{"larmor_Hz", "mas_Hz", "order",
    "transitions"}``."""

    SITE_COLUMNS = ("Label", "δiso (ppm)", "Interactions", "α (°)", "β (°)", "γ (°)",
                    "Dipolar partner(s)")
    MODE_COLUMNS = ("Motion", "Positions / angle", "Axis: angle from axis 1, azimuth (°)",
                    "k (s⁻¹)", "Ea (eV)", "ν₀ (s⁻¹)", "Sense", "Step (°)")
    HOP_COLUMNS = ("From", "To", "k → (s⁻¹)", "Ea → (eV)", "Ea ← (eV)", "ν₀ (s⁻¹)")

    def __init__(self, canvas=None, show_view=None, get_spectra=None, get_simulate_sites=None,
                 get_simulate_settings=None, on_run=None, parent=None):
        super().__init__(parent)
        self.canvas = canvas
        self.show_view = show_view or (lambda: None)
        self.get_spectra = get_spectra or (lambda: [])
        self.get_simulate_sites = get_simulate_sites or (lambda: [])
        self.get_simulate_settings = get_simulate_settings or (lambda: None)
        self.on_run = on_run or (lambda: None)     # the window journals each run
        self._last_run = "sweep"
        self.series = None
        self.overlay = None
        self._build()
        self.load_example()

    # ------------------------------------------------------------------ UI
    def _build(self) -> None:
        layout = QVBoxLayout(self)
        main_layout = layout
        intro = QLabel(
            "<b>1. Interactions</b> (type them in or import from Simulate) &nbsp;→&nbsp; "
            "<b>2. Rotation axes</b> (each with its own positions, sense and rate) &nbsp;→&nbsp; "
            "<b>3. Simulate or fit</b> (next sections). Hops between environments are under "
            "<i>Advanced</i>.")
        intro.setWordWrap(True)
        intro.setTextFormat(Qt.TextFormat.RichText)
        main_layout.addWidget(intro)

        step1 = QGroupBox("1. Interactions: the tensors that rotate")
        v1 = QVBoxLayout(step1)
        hint1 = QLabel(
            "One row per environment. Use <i>Add interaction…</i> to type its shift, CSA, "
            "quadrupolar and dipolar tensors, or import them from Simulate. α, β, γ place its "
            "principal axes relative to rotation axis 1; β is the angle between axis 1 and the "
            "principal z axis (for a C–D bond, between the axis and the bond).")
        hint1.setWordWrap(True)
        hint1.setTextFormat(Qt.TextFormat.RichText)
        v1.addWidget(hint1)
        head = QHBoxLayout()
        self.quick_nucleus_label = QLabel("")
        head.addWidget(QLabel("Nucleus:"))
        head.addWidget(self.quick_nucleus_label)
        head.addStretch(1)
        v1.addLayout(head)
        self.sites_table = QTableWidget(0, len(self.SITE_COLUMNS))
        self.sites_table.setHorizontalHeaderLabels(list(self.SITE_COLUMNS))
        self.sites_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.sites_table.horizontalHeader().setStretchLastSection(True)
        self.sites_table.setMaximumHeight(150)
        self.sites_table.setToolTip(
            "Double-click a label to edit the tensors. α, β, γ: the Euler angles (ZYZ, degrees) "
            "that place the interaction's own frame – the EFG frame with a quadrupolar coupling, "
            "else the CSA frame – relative to rotation axis 1 (the z axis of the reference frame). "
            "Dipolar: 'partner:D(kHz)' or 'partner@r(Å)', e.g. 1H:23.3 or 1H@1.09 (the bond along "
            "the interaction's z axis).")
        for c, tip in ((3, "Euler α (°): rotation about axis 1; only matters with several "
                           "interactions or axes."),
                       (4, "Euler β (°): the angle between rotation axis 1 and the principal z "
                           "axis of the interaction."),
                       (5, "Euler γ (°): rotation about the interaction's own z axis.")):
            self.sites_table.horizontalHeaderItem(c).setToolTip(tip)
        self.sites_table.cellDoubleClicked.connect(
            lambda r, c: self.edit_interaction(r) if c < 3 else None)
        self.sites_table.itemChanged.connect(lambda _i: self._refresh_summary())
        v1.addWidget(self.sites_table)
        row = QHBoxLayout()
        self.add_site_button = QPushButton("Add interaction…")
        self.add_site_button.setToolTip("Type in the shift, CSA, quadrupolar and dipolar tensors "
                                        "of one environment.")
        self.add_site_button.clicked.connect(self.add_interaction_dialog)
        self.edit_site_button = QPushButton("Edit…")
        self.edit_site_button.clicked.connect(lambda: self.edit_interaction())
        self.take_button = QPushButton("Import from Simulate")
        self.take_button.setToolTip(
            "Replace the environments by the sites of Simulate ▸ Solid-state NMR (shift, CSA, "
            "quadrupolar coupling; the observed nucleus). Set them up there first.")
        self.take_button.clicked.connect(self.take_from_simulate)
        self.quick_take_button = self.take_button
        self.remove_site_button = QPushButton("Remove")
        self.remove_site_button.clicked.connect(self.remove_site)
        self.settings_button = QPushButton("Use Simulate's field, MAS and order")
        self.settings_button.clicked.connect(self.use_simulate_settings)
        self.quick_example_button = QPushButton("Example: ²H C–D bond")
        self.quick_example_button.clicked.connect(self.load_example)
        for b in (self.add_site_button, self.edit_site_button, self.take_button,
                  self.remove_site_button, self.settings_button, self.quick_example_button):
            row.addWidget(b)
        row.addStretch(1)
        v1.addLayout(row)
        main_layout.addWidget(step1)

        step2 = QGroupBox("2. Rotation axes")
        v2 = QVBoxLayout(step2)
        hint2 = QLabel(
            "Describe one axis, then <i>Add as another axis</i>. <b>Axis 1 is the reference "
            "direction (z)</b>; every further axis is given by its angle from axis 1 and its "
            "azimuth about it. Every axis has its own positions, sense and rate. List the "
            "innermost axis first (a methyl turning on a group that itself turns: methyl first).")
        hint2.setWordWrap(True)
        hint2.setTextFormat(Qt.TextFormat.RichText)
        v2.addWidget(hint2)
        f2 = QFormLayout()
        v2.addLayout(f2)
        self.quick_kind_combo = QComboBox()
        for text, key, tip in QUICK_MOTIONS:
            self.quick_kind_combo.addItem(text, key)
            self.quick_kind_combo.setItemData(self.quick_kind_combo.count() - 1, tip,
                                              Qt.ItemDataRole.ToolTipRole)
        self.quick_kind_combo.currentIndexChanged.connect(lambda _i: self._quick_kind_changed())
        f2.addRow("Type of motion:", self.quick_kind_combo)
        self.quick_kind_help = QLabel("")
        self.quick_kind_help.setWordWrap(True)
        f2.addRow("", self.quick_kind_help)
        self.quick_n_spin = QDoubleSpinBox()
        self.quick_n_spin.setDecimals(0)
        self.quick_n_spin.setRange(2, 360)
        self.quick_n_spin.setValue(3)
        self.quick_n_label = QLabel("Number of positions:")
        f2.addRow(self.quick_n_label, self.quick_n_spin)
        axis_row = QHBoxLayout()
        self.quick_axis_combo = QComboBox()
        for text, key in (("Axis 1 / reference (z)", "z"), ("x of the reference frame", "x"),
                          ("y of the reference frame", "y"),
                          ("At an angle to axis 1…", "custom")):
            self.quick_axis_combo.addItem(text, key)
        self.quick_axis_edit = QLineEdit("")
        self.quick_axis_edit.setPlaceholderText("angle from axis 1, azimuth (°)  e.g. 55, 0")
        self.quick_axis_edit.setToolTip("The direction of this axis in the reference frame (axis 1 "
                                        "is z): 'polar,azimuth' degrees – the angle from axis 1 "
                                        "and the azimuth about it – or 'x,y,z' components.")
        self.quick_axis_edit.setEnabled(False)
        self.quick_axis_combo.currentIndexChanged.connect(
            lambda _i: self.quick_axis_edit.setEnabled(self.quick_axis_combo.currentData() == "custom"))
        axis_row.addWidget(self.quick_axis_combo)
        axis_row.addWidget(self.quick_axis_edit)
        self.quick_axis_label = QLabel("Rotation axis:")
        f2.addRow(self.quick_axis_label, axis_row)
        self.quick_sense_combo = QComboBox()
        for text, key in (("Both ways (to either neighbour)", "nearest"),
                          ("One way: anticlockwise (looking down the axis)", "forward"),
                          ("One way: clockwise (looking down the axis)", "backward"),
                          ("To any other position", "random")):
            self.quick_sense_combo.addItem(text, key)
        self.quick_sense_combo.setToolTip(
            "'Back and forth': each jump goes to either neighbour with equal probability. 'One way': "
            "every jump goes the same way round. For a single tensor the two one-way senses give "
            "the same powder lineshape (the mirror image is the same pattern); the choice matters "
            "when several modes or hops are combined.")
        self.quick_sense_label = QLabel("Sense of rotation:")
        f2.addRow(self.quick_sense_label, self.quick_sense_combo)
        self.quick_step_edit = QLineEdit("")
        self.quick_step_edit.setMaximumWidth(90)
        self.quick_step_edit.setPlaceholderText("360/n")
        self.quick_step_edit.setToolTip(
            "Angle between neighbouring positions. Blank: 360°/n, a closed ring (a methyl's three "
            "positions). A smaller step gives an open arc (e.g. 3 positions 20° apart: a wobble "
            "that does not wrap round).")
        self.quick_step_label = QLabel("Step between positions (°):")
        f2.addRow(self.quick_step_label, self.quick_step_edit)
        rate_row = QHBoxLayout()
        self.quick_rate_combo = QComboBox()
        self.quick_rate_combo.addItem("Rate k (s⁻¹)", "k")
        self.quick_rate_combo.addItem("Activation barrier Ea (eV) at a temperature", "Ea")
        self.quick_rate_edit = QLineEdit("1e5")
        self.quick_nu0_edit = QLineEdit("1e12")
        self.quick_nu0_edit.setMaximumWidth(80)
        self.quick_nu0_edit.setToolTip("Attempt frequency ν₀ (s⁻¹): k = ν₀ exp(−Ea / k_B T).")
        self.quick_nu0_label = QLabel("ν₀ (s⁻¹):")
        self.quick_T_edit = QLineEdit("")
        self.quick_T_edit.setMaximumWidth(70)
        self.quick_T_edit.setPlaceholderText("T (K)")
        self.quick_rate_combo.currentIndexChanged.connect(lambda _i: self._quick_rate_changed())
        for w in (self.quick_rate_combo, self.quick_rate_edit, self.quick_nu0_label,
                  self.quick_nu0_edit, self.quick_T_edit):
            rate_row.addWidget(w)
        rate_row.addStretch(1)
        f2.addRow("How fast:", rate_row)
        arow = QHBoxLayout()
        self.quick_add_button = QPushButton("Add as another axis")
        self.quick_add_button.setToolTip("Appends this axis (outer to the ones already listed).")
        self.quick_add_button.clicked.connect(lambda: self.apply_quick_motion(replace=False))
        self.quick_apply_button = QPushButton("Use as the only axis")
        self.quick_apply_button.setToolTip("Replaces every listed axis by this one.")
        self.quick_apply_button.clicked.connect(lambda: self.apply_quick_motion())
        arow.addWidget(self.quick_add_button)
        arow.addWidget(self.quick_apply_button)
        arow.addStretch(1)
        v2.addLayout(arow)

        v2.addWidget(QLabel("Axes in the model (innermost first; edit any cell, each axis has "
                            "its own rate)"))
        self.modes_table = QTableWidget(0, len(self.MODE_COLUMNS))
        self.modes_table.setHorizontalHeaderLabels(list(self.MODE_COLUMNS))
        self.modes_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.modes_table.setMaximumHeight(170)
        self.modes_table.setToolTip(
            "Axis: z, x, y, 'x,y,z' or 'angle from axis 1, azimuth' in degrees. Positions: how "
            "many; Step: the angle between neighbours (blank = 360/n, a closed ring). k: the rate "
            "of each allowed jump (for ring diffusion the diffusion rate D). Give Ea and ν₀ "
            "instead for an Arrhenius rate (needs a temperature). Axes are independent processes "
            "with their own rates; the first listed is the innermost (applied first).")
        self.modes_table.itemChanged.connect(lambda _i: self._refresh_summary())
        v2.addWidget(self.modes_table)
        row = QHBoxLayout()
        for text, slot in (("Remove axis", self.remove_mode), ("Up", lambda: self.move_mode(-1)),
                           ("Down", lambda: self.move_mode(1))):
            b = QPushButton(text)
            b.clicked.connect(slot)
            row.addWidget(b)
        row.addStretch(1)
        v2.addLayout(row)
        self.summary_label = QLabel("")
        self.summary_label.setWordWrap(True)
        self.summary_label.setTextFormat(Qt.TextFormat.RichText)
        v2.addWidget(self.summary_label)
        main_layout.addWidget(step2)

        self.advanced_box = QGroupBox("Advanced: hops between environments")
        self.advanced_box.setCheckable(True)
        self.advanced_box.setChecked(False)
        advanced_inner = QWidget()
        adv_layout = QVBoxLayout(advanced_inner)
        adv_layout.setContentsMargins(0, 0, 0, 0)
        outer_adv = QVBoxLayout(self.advanced_box)
        outer_adv.addWidget(advanced_inner)
        self.advanced_box.toggled.connect(advanced_inner.setVisible)
        advanced_inner.setVisible(False)
        layout = adv_layout

        layout.addWidget(QLabel("Hops between environments (each reversible; numbers from 1)"))
        self.hops_table = QTableWidget(0, len(self.HOP_COLUMNS))
        self.hops_table.setHorizontalHeaderLabels(list(self.HOP_COLUMNS))
        self.hops_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.hops_table.setMaximumHeight(100)
        self.hops_table.setToolTip("k →: a fixed rate (s⁻¹) – or leave it blank and give Ea → "
                                   "(and Ea ← for the reverse barrier, ν₀) for an Arrhenius "
                                   "rate. Different reverse barriers set the populations.")
        layout.addWidget(self.hops_table)
        row = QHBoxLayout()
        for text, slot in (("Add hop", lambda: self.add_hop()), ("Remove hop", self.remove_hop)):
            b = QPushButton(text)
            b.clicked.connect(slot)
            row.addWidget(b)
        row.addStretch(1)
        layout.addLayout(row)

        layout = main_layout
        layout.addWidget(self.advanced_box)
        box = QGroupBox("Experiment")
        form = QFormLayout(box)
        field = QHBoxLayout()
        self.nucleus_label = QLabel("")
        self.field_spin = QDoubleSpinBox()
        self.field_spin.setRange(0.1, 40.0)
        self.field_spin.setDecimals(3)
        self.field_spin.setValue(9.4)
        self.field_spin.setSuffix(" T")
        self.larmor_label = QLabel("")
        self.field_spin.valueChanged.connect(lambda _v: self._update_larmor())
        field.addWidget(self.nucleus_label)
        field.addWidget(QLabel("at"))
        field.addWidget(self.field_spin)
        field.addWidget(self.larmor_label)
        field.addStretch(1)
        form.addRow("Nucleus:", field)
        self.order_combo = QComboBox()
        self.order_combo.addItem("First order", "first")
        self.order_combo.addItem("Second order (quadrupolar)", "second")
        self.order_combo.setToolTip("Second order is state by state: valid for motion slower than "
                                    "the quadrupolar frequency (the log says when it is not).")
        form.addRow("Quadrupolar order:", self.order_combo)
        self.transitions_combo = QComboBox()
        for text, key in (("All transitions", "all"), ("Central transition", "ct"),
                          ("Satellite transitions", "st")):
            self.transitions_combo.addItem(text, key)
        form.addRow("Transitions:", self.transitions_combo)
        self.mas_spin = QDoubleSpinBox()
        self.mas_spin.setRange(0.0, 200.0)
        self.mas_spin.setDecimals(3)
        self.mas_spin.setSuffix(" kHz")
        self.mas_spin.setToolTip("0 = static. Under MAS rates comparable to the spinning speed are "
                                 "included (time-domain engine).")
        form.addRow("MAS rate:", self.mas_spin)
        echo = QHBoxLayout()
        self.echo_combo = QComboBox()
        for text, key in (("No echo (FID)", None), ("Hahn echo (π pulse)", "hahn"),
                          ("Quadrupolar (solid) echo", "solid")):
            self.echo_combo.addItem(text, key)
        self.tau_spin = QDoubleSpinBox()
        self.tau_spin.setRange(0.1, 100000.0)
        self.tau_spin.setDecimals(2)
        self.tau_spin.setValue(30.0)
        self.tau_spin.setSuffix(" µs")
        self.tau_spin.setToolTip("Delay τ between the pulses (the echo top is at 2τ). Under MAS "
                                 "use whole rotor periods.")
        echo.addWidget(self.echo_combo)
        echo.addWidget(QLabel("τ ="))
        echo.addWidget(self.tau_spin)
        echo.addStretch(1)
        form.addRow("Sequence:", echo)
        self.gauss_spin = QDoubleSpinBox()
        self.gauss_spin.setRange(0.0, 1000.0)
        self.gauss_spin.setDecimals(3)
        self.gauss_spin.setSuffix(" ppm")
        form.addRow("Gaussian broadening:", self.gauss_spin)
        axis = QHBoxLayout()
        self.auto_axis_check = QCheckBox("Auto")
        self.auto_axis_check.setChecked(True)
        self.axis_hi = QDoubleSpinBox()
        self.axis_lo = QDoubleSpinBox()
        for s, v in ((self.axis_hi, 200.0), (self.axis_lo, -200.0)):
            s.setRange(-1e6, 1e6)
            s.setDecimals(2)
            s.setValue(v)
            s.setSuffix(" ppm")
        self.points_spin = TypedSpinBox()
        self.points_spin.setRange(256, 16384)
        self.points_spin.setValue(2048)
        self.auto_axis_check.toggled.connect(self._axis_enabled)
        for w in (self.auto_axis_check, self.axis_hi, QLabel("to"), self.axis_lo,
                  QLabel("points"), self.points_spin):
            axis.addWidget(w)
        axis.addStretch(1)
        form.addRow("Axis:", axis)
        self.temperature_edit = QLineEdit("")
        self.temperature_edit.setPlaceholderText("only for Arrhenius rates")
        self.temperature_edit.setToolTip("K. Needed when a mode or hop is given as Ea and ν₀ "
                                         "(and the sweep is not over temperature).")
        form.addRow("Temperature (K):", self.temperature_edit)
        layout.addWidget(box)

        sweep = QGroupBox("Optional: simulate a series (vary the rate or the temperature)")
        sform = QFormLayout(sweep)
        self.sweep_combo = QComboBox()
        self._picked = None                       # what the user chose, kept as the list changes
        self.sweep_combo.activated.connect(
            lambda _i: setattr(self, "_picked", self.sweep_combo.currentData()))
        sform.addRow("Vary:", self.sweep_combo)
        self.values_edit = QLineEdit("1e2:1e8:7 log")
        self.values_edit.setToolTip("A list (100, 1e4, 1e6), lo:hi:n (linear) or lo:hi:n log. Rates "
                                    "in s⁻¹; hop scale a multiplier; temperature in K.")
        sform.addRow("Values:", self.values_edit)
        self.limits_check = QCheckBox("Show the slow and fast limits")
        self.limits_check.setChecked(True)
        sform.addRow("", self.limits_check)
        layout.addWidget(sweep)

        actions = QHBoxLayout()
        self.simulate_button = QPushButton("Simulate series")
        self.simulate_button.clicked.connect(self.simulate)
        self.limits_button = QPushButton("Limits and averaged tensor")
        self.limits_button.setToolTip("The slow limit, the two fast limits and the motionally "
                                      "averaged tensor (δiso, CSA, quadrupolar, dipolar), static "
                                      "or under the MAS rate above.")
        self.limits_button.clicked.connect(self.show_limits)
        self.code_button = QPushButton("Copy notebook code")
        self.code_button.clicked.connect(self.copy_notebook_code)
        for b in (self.simulate_button, self.limits_button, self.code_button):
            actions.addWidget(b)
        actions.addStretch(1)
        layout.addLayout(actions)

        over = QGroupBox("Compare with the spectrum selected in Open experiments")
        ol = QVBoxLayout(over)
        note = QLabel("The simulation is scaled onto the spectrum (scale and offset by least "
                      "squares) and the residual shown. With the value blank the tables' rates are "
                      "used; a value replaces the varied quantity above. Several selected spectra "
                      "are each simulated at their own temperature when they have one.")
        note.setWordWrap(True)
        ol.addWidget(note)
        orow = QHBoxLayout()
        self.overlay_value_edit = QLineEdit("")
        self.overlay_value_edit.setPlaceholderText("rate / scale / T")
        self.overlay_value_edit.setMaximumWidth(120)
        self.window_check = QCheckBox("Only inside the axis range above")
        self.overlay_button = QPushButton("Overlay")
        self.overlay_button.clicked.connect(self.overlay_selected)
        orow.addWidget(QLabel("Value:"))
        orow.addWidget(self.overlay_value_edit)
        orow.addWidget(self.window_check)
        orow.addWidget(self.overlay_button)
        orow.addStretch(1)
        ol.addLayout(orow)
        layout.addWidget(over)

        self.results_label = ReportLabel("")
        self.results_label.setWordWrap(True)
        self.results_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.results_label)
        self.status_log = _log_box()
        layout.addWidget(self.status_log)
        self._axis_enabled()
        self._update_larmor()
        self._quick_kind_changed()
        self._quick_rate_changed()

    # ------------------------------------------------------ guided set-up
    def _quick_kind_changed(self) -> None:
        key = self.quick_kind_combo.currentData()
        self.quick_kind_help.setText(next(t for _x, k, t in QUICK_MOTIONS if k == key))
        has_axis = key != "random_reorientation"
        for w in (self.quick_axis_combo, self.quick_axis_edit, self.quick_axis_label):
            w.setVisible(has_axis)
        show_n = key == "ring_jump"
        self.quick_n_spin.setVisible(key in ("ring_jump", "flip"))
        self.quick_n_label.setVisible(key in ("ring_jump", "flip"))
        self.quick_n_label.setText("Number of positions:" if show_n else "Angle between positions (°):")
        if key == "flip":
            self.quick_n_spin.setRange(1, 359)
            self.quick_n_spin.setValue(180)
        elif key == "ring_jump":
            self.quick_n_spin.setRange(2, 360)
            self.quick_n_spin.setValue(3)
        self.quick_sense_combo.setVisible(show_n)
        self.quick_sense_label.setVisible(show_n)
        self.quick_step_edit.setVisible(show_n)
        self.quick_step_label.setVisible(show_n)

    def _quick_rate_changed(self) -> None:
        arrh = self.quick_rate_combo.currentData() == "Ea"
        self.quick_rate_edit.setPlaceholderText("Ea (eV)" if arrh else "k (s⁻¹)")
        self.quick_rate_edit.setText("0.3" if arrh else "1e5")
        self.quick_nu0_label.setVisible(arrh)
        self.quick_nu0_edit.setVisible(arrh)
        self.quick_T_edit.setVisible(arrh)

    def quick_mode_spec(self) -> dict:
        """The mode dict the quick set-up describes (ValueError for a bad entry)."""
        key = self.quick_kind_combo.currentData()
        spec = {"kind": key}
        if key == "random_reorientation":
            spec["param"] = "octahedral"
            spec["axis"] = ""
        else:
            axis = self.quick_axis_combo.currentData()
            if axis == "custom":
                axis = self.quick_axis_edit.text().strip()
                if not axis:
                    raise ValueError("Type the axis as angle from axis 1, azimuth (degrees) "
                                     "or as x,y,z.")
            spec["axis"] = axis
            spec["param"] = {"ring_jump": f"{int(self.quick_n_spin.value())}",
                             "flip": f"{self.quick_n_spin.value():g}",
                             "ring_diffusion": "24"}[key]
        if key == "ring_jump":
            spec["sense"] = self.quick_sense_combo.currentData()
            step_text = self.quick_step_edit.text().strip()
            if step_text:
                step = _float(step_text)
                n = int(self.quick_n_spin.value())
                if step is None or step <= 0 or n * step > 360.0 + 1e-9:
                    raise ValueError(f"The step must be a positive angle with n × step ≤ 360° "
                                     f"(n = {n}); leave it blank for 360°/n.")
                if (abs(n * step - 360.0) > 1e-9
                        and self.quick_sense_combo.currentData() in ("forward", "backward")):
                    raise ValueError("One-way hops need a closed ring (n × step = 360°). Use "
                                     "'both ways' for an open arc of positions.")
                spec["step"] = f"{step:g}"
        value = _float(self.quick_rate_edit.text())
        if value is None or value <= 0:
            raise ValueError("Enter the rate (or barrier) as a positive number.")
        if self.quick_rate_combo.currentData() == "Ea":
            spec["Ea"] = f"{value:g}"
            spec["nu0"] = f"{_float(self.quick_nu0_edit.text(), 1e12):g}"
        else:
            spec["k"] = f"{value:g}"
        return spec

    def apply_quick_motion(self, replace: bool = True) -> bool:
        """Write the axis described in the form into the axes table: replacing every axis
        (``replace=True``) or appended after the ones listed."""
        try:
            spec = self.quick_mode_spec()
            T = None
            if "Ea" in spec:
                T = _float(self.quick_T_edit.text())
                if T is None or T <= 0:
                    raise ValueError("An activation barrier needs the temperature (K).")
        except ValueError as exc:
            QMessageBox.warning(self, "Anisotropic motion", str(exc))
            return False
        if replace:
            self._load_modes([spec])
        else:
            self.add_mode(spec)
        if T is not None:
            self.temperature_edit.setText(f"{T:g}")
        n_axes = self.modes_table.rowCount()
        self._log(("Rotation set" if replace else f"Axis {n_axes} added") + ": "
                  + ", ".join(f"{k} {v}" for k, v in spec.items() if v != ""))
        return True

    def _axis_enabled(self) -> None:
        on = not self.auto_axis_check.isChecked()
        self.axis_hi.setEnabled(on)
        self.axis_lo.setEnabled(on)

    # ------------------------------------------------------------- tables
    @staticmethod
    def _cell(table, r, c) -> str:
        item = table.item(r, c)
        return item.text().strip() if item else ""

    @staticmethod
    def _set_row(table, r, values, read_only=()) -> None:
        table.blockSignals(True)
        for c, v in enumerate(values):
            item = QTableWidgetItem("" if v is None else str(v))
            if c in read_only:
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            table.setItem(r, c, item)
        table.blockSignals(False)

    def _fill_site_row(self, r, site: Site, euler, dipolar) -> None:
        self._set_row(self.sites_table, r,
                      [site.label, f"{site.delta_iso:g}", _interaction_text(site) or "isotropic",
                       euler[0], euler[1], euler[2], dipolar], read_only=(0, 1, 2))
        self.sites_table.item(r, 0).setData(Qt.ItemDataRole.UserRole, site.to_dict())

    def add_site(self, site: Site, euler=(0.0, 0.0, 0.0), dipolar="") -> None:
        r = self.sites_table.rowCount()
        self.sites_table.insertRow(r)
        self._fill_site_row(r, site, euler, dipolar)
        self._update_nucleus()

    def add_interaction(self, nucleus, delta_iso=0.0, label=None, csa=None, csa_euler=(0, 0, 0),
                        quad=None, dipolar="", euler=(0.0, 0.0, 0.0), fwhm=2.0,
                        lineshape="lorentzian", row=None) -> int:
        """Add an environment from tensors given directly (or replace row ``row``).

        ``csa`` = (Δ ppm, η) Haeberlen; ``quad`` = (C_Q MHz, η_Q); ``csa_euler``: the CSA axes in
        the EFG frame (only with a quadrupolar coupling); ``dipolar``: 'partner@r(Å)' or
        'partner:D(kHz)', comma separated; ``euler``: α, β, γ placing the interaction's
        principal axes relative to rotation axis 1. Returns the row. ValueError for anything
        that is not a valid site."""
        parse_dipolar(dipolar)                                   # validate before touching rows
        n = self.sites_table.rowCount()
        site = Site(str(nucleus), float(delta_iso),
                    label=label or (f"{nucleus} site {n + 1 if row is None else row + 1}"),
                    fwhm=float(fwhm), lineshape=lineshape)
        if quad is not None:
            site.add_quadrupolar(float(quad[0]), float(quad[1]) if len(quad) > 1 else 0.0)
        if csa is not None:
            site.add_csa(float(csa[0]), float(csa[1]) if len(csa) > 1 else 0.0,
                         euler=tuple(float(a) for a in csa_euler))
        euler = tuple(float(a) for a in euler)
        if row is None:
            self.add_site(site, euler=euler, dipolar=dipolar)
            return self.sites_table.rowCount() - 1
        self._fill_site_row(row, site, euler, dipolar)
        self._update_nucleus()
        return row

    def add_interaction_dialog(self) -> bool:
        dlg = InteractionDialog(self, nucleus=self.nucleus() or "2H")
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return False
        try:
            self.add_interaction(**dlg.values())
        except (ValueError, KeyError) as exc:
            QMessageBox.warning(self, "Anisotropic motion", str(exc))
            return False
        self._log("Interaction added.")
        return True

    def edit_interaction(self, row=None) -> bool:
        r = self.sites_table.currentRow() if row is None else row
        if r < 0 or r >= self.sites_table.rowCount():
            QMessageBox.information(self, "Anisotropic motion", "Select an environment to edit.")
            return False
        site = Site.from_dict(self.sites_table.item(r, 0).data(Qt.ItemDataRole.UserRole))
        euler = tuple(_float(self._cell(self.sites_table, r, c), 0.0) for c in (3, 4, 5))
        dlg = InteractionDialog(self, site=site, euler=euler,
                                dipolar=self._cell(self.sites_table, r, 6))
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return False
        try:
            self.add_interaction(row=r, **dlg.values())
        except (ValueError, KeyError) as exc:
            QMessageBox.warning(self, "Anisotropic motion", str(exc))
            return False
        return True

    def remove_site(self) -> None:
        r = self.sites_table.currentRow()
        if r < 0:
            r = self.sites_table.rowCount() - 1
        if r >= 0:
            self.sites_table.removeRow(r)
        self._update_nucleus()

    def add_mode(self, spec=None) -> None:
        r = self.modes_table.rowCount()
        self.modes_table.blockSignals(True)
        self.modes_table.insertRow(r)
        self.modes_table.blockSignals(False)
        spec = spec or {"kind": "ring_jump", "param": "3", "axis": "z", "k": "1e4"}
        combo = QComboBox()
        for text, key, _what, _default in MODE_KINDS:
            combo.addItem(text, key)
        combo.setCurrentIndex(max(0, combo.findData(spec.get("kind", "ring_jump"))))
        combo.currentIndexChanged.connect(lambda _i, c=combo: self._kind_changed(c))
        self.modes_table.setCellWidget(r, 0, combo)
        sense = QComboBox()
        for text, key in SENSES:
            sense.addItem(text, key)
        sense.setCurrentIndex(max(0, sense.findData(spec.get("sense") or "nearest")))
        sense.currentIndexChanged.connect(lambda _i: self._refresh_summary())
        self.modes_table.setCellWidget(r, 6, sense)
        self._set_row_from(self.modes_table, r, 1, [spec.get("param", ""), spec.get("axis", "z"),
                                                    spec.get("k", ""), spec.get("Ea", ""),
                                                    spec.get("nu0", "1e12" if spec.get("Ea") else "")])
        self._set_row_from(self.modes_table, r, 7, [spec.get("step", "")])
        self._sync_mode_row(r)
        self._refresh_sweep_choices()

    def _sync_mode_row(self, r) -> None:
        """Sense and step only mean something for ring jumps."""
        combo = self.modes_table.cellWidget(r, 0)
        ring = combo is not None and combo.currentData() == "ring_jump"
        self.modes_table.cellWidget(r, 6).setEnabled(ring)
        item = self.modes_table.item(r, 7)
        if item is not None:
            flags = item.flags() | Qt.ItemFlag.ItemIsEditable
            item.setFlags(flags if ring else flags & ~Qt.ItemFlag.ItemIsEditable)

    def _set_row_from(self, table, r, first_col, values) -> None:
        table.blockSignals(True)
        for i, v in enumerate(values):
            table.setItem(r, first_col + i, QTableWidgetItem("" if v is None else str(v)))
        table.blockSignals(False)

    def _kind_changed(self, combo) -> None:
        for r in range(self.modes_table.rowCount()):
            if self.modes_table.cellWidget(r, 0) is combo:
                key = combo.currentData()
                default = next(d for _t, k, _w, d in MODE_KINDS if k == key)
                self._set_row_from(self.modes_table, r, 1, [default])
                if key != "ring_jump":
                    self._set_row_from(self.modes_table, r, 7, [""])
                self._sync_mode_row(r)
        self._refresh_sweep_choices()

    def remove_mode(self) -> None:
        r = self.modes_table.currentRow()
        if r < 0:
            r = self.modes_table.rowCount() - 1
        if r >= 0:
            self.modes_table.removeRow(r)
        self._refresh_sweep_choices()

    def move_mode(self, step: int) -> None:
        r = self.modes_table.currentRow()
        specs = self._mode_rows()
        j = r + step
        if r < 0 or not 0 <= j < len(specs):
            return
        specs[r], specs[j] = specs[j], specs[r]
        self._load_modes(specs)
        self.modes_table.setCurrentCell(j, 1)

    def _mode_rows(self) -> list[dict]:
        out = []
        for r in range(self.modes_table.rowCount()):
            combo = self.modes_table.cellWidget(r, 0)
            sense = self.modes_table.cellWidget(r, 6)
            out.append({"kind": combo.currentData(), "param": self._cell(self.modes_table, r, 1),
                        "axis": self._cell(self.modes_table, r, 2),
                        "k": self._cell(self.modes_table, r, 3),
                        "Ea": self._cell(self.modes_table, r, 4),
                        "nu0": self._cell(self.modes_table, r, 5),
                        "sense": sense.currentData() if sense is not None else "nearest",
                        "step": self._cell(self.modes_table, r, 7)})
        return out

    def _load_modes(self, specs) -> None:
        self.modes_table.setRowCount(0)
        for s in specs:
            self.add_mode(s)

    def add_hop(self, values=None) -> None:
        r = self.hops_table.rowCount()
        self.hops_table.insertRow(r)
        self._set_row(self.hops_table, r, values or [1, 2, "1e4", "", "", ""])
        self._refresh_sweep_choices()

    def remove_hop(self) -> None:
        r = self.hops_table.currentRow()
        if r < 0:
            r = self.hops_table.rowCount() - 1
        if r >= 0:
            self.hops_table.removeRow(r)
        self._refresh_sweep_choices()

    def load_example(self) -> None:
        """A deuterated methyl-like group: C–D along a 3-fold axis (the cone jump of Weblab)."""
        self.sites_table.setRowCount(0)
        site = Site("2H", 0.0, label="C–D", fwhm=2.0, lineshape="lorentzian").add_quadrupolar(0.17, 0.0)
        self.add_site(site, euler=(0.0, 70.5, 0.0))
        self._load_modes([{"kind": "ring_jump", "param": "3", "axis": "z", "k": "1e5"}])
        self.hops_table.setRowCount(0)
        self.field_spin.setValue(9.4)
        self.mas_spin.setValue(0.0)
        self.values_edit.setText("1e3:1e8:6 log")
        self._picked = None
        self._refresh_sweep_choices()

    # -------------------------------------------------- Simulate as source
    def take_from_simulate(self) -> bool:
        try:
            sites = list(self.get_simulate_sites() or [])
        except (ValueError, KeyError) as exc:
            QMessageBox.warning(self, "Anisotropic motion", str(exc))
            return False
        if not sites:
            QMessageBox.information(self, "Anisotropic motion",
                                    "Simulate ▸ Solid-state NMR has no sites yet.")
            return False
        self.sites_table.setRowCount(0)
        dropped = []
        for s in sites:
            if s.get("jcoupling"):
                dropped.append(s.label)
            self.add_site(s)
        self._log(f"Took {len(sites)} environment(s) from Simulate: "
                  + ", ".join(s.label for s in sites) + "."
                  + (f" J couplings of {', '.join(dropped)} are not part of the motional model "
                     "and are ignored." if dropped else ""))
        self.use_simulate_settings(quiet=True)
        return True

    def use_simulate_settings(self, quiet: bool = False) -> None:
        try:
            s = self.get_simulate_settings()
        except (ValueError, KeyError) as exc:
            if not quiet:
                QMessageBox.warning(self, "Anisotropic motion", str(exc))
            return
        if not s:
            return
        if s.get("larmor_Hz"):
            self._set_larmor_MHz(float(s["larmor_Hz"]) / 1e6)
        self.mas_spin.setValue(float(s.get("mas_Hz") or 0.0) / 1e3)
        i = self.order_combo.findData("second" if s.get("order") in ("second", "exact") else "first")
        self.order_combo.setCurrentIndex(max(i, 0))
        i = self.transitions_combo.findData(s.get("transitions", "all"))
        self.transitions_combo.setCurrentIndex(max(i, 0))
        self._log("Field, MAS rate, order and transitions taken from Simulate.")

    # ------------------------------------------------------ field/nucleus
    def nucleus(self) -> str:
        for r in range(self.sites_table.rowCount()):
            d = self.sites_table.item(r, 0).data(Qt.ItemDataRole.UserRole)
            if d:
                return str(d.get("nucleus") or d.get("isotope") or "")
        return ""

    def _update_nucleus(self) -> None:
        self.nucleus_label.setText(self.nucleus() or "(no environments)")
        if hasattr(self, "quick_nucleus_label"):
            from ..notebook.motion import _nucleus_math
            self.quick_nucleus_label.setText(_nucleus_math(self.nucleus()) or "(none yet)")
        self._update_larmor()
        self._refresh_sweep_choices()

    def larmor_MHz(self) -> float:
        from ..notebook.simulations import larmor_frequency_Hz
        return larmor_frequency_Hz(self.nucleus(), self.field_spin.value()) / 1e6

    def _set_larmor_MHz(self, mhz: float) -> None:
        from ..notebook.simulations import larmor_frequency_Hz
        per_tesla = larmor_frequency_Hz(self.nucleus(), 1.0) / 1e6
        if per_tesla > 0:
            self.field_spin.setValue(mhz / per_tesla)

    def _update_larmor(self) -> None:
        try:
            self.larmor_label.setText(f"= {self.larmor_MHz():.4g} MHz")
        except Exception:  # noqa: BLE001 -- no or unknown nucleus yet
            self.larmor_label.setText("")

    # ------------------------------------------------------------ the model
    def build_model(self):
        """The :class:`MotionModel` the tables describe (ValueError for anything the tables
        cannot make into one)."""
        if self.sites_table.rowCount() == 0:
            raise ValueError("Add at least one environment (Take sites from Simulate).")
        sites = []
        for r in range(self.sites_table.rowCount()):
            d = self.sites_table.item(r, 0).data(Qt.ItemDataRole.UserRole)
            euler = []
            for c, name in ((3, "α"), (4, "β"), (5, "γ")):
                v = _float(self._cell(self.sites_table, r, c), 0.0)
                euler.append(v)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)         # J couplings: said in the log
                ms = M.MotionSite(Site.from_dict(d), euler=tuple(euler))
            for dip in parse_dipolar(self._cell(self.sites_table, r, 6)):
                ms.add_dipolar(**dip)
            sites.append(ms)
        modes = []
        for i, row in enumerate(self._mode_rows(), start=1):
            spec = {"kind": row["kind"], "axis": row["axis"] or "z"}
            key = _PARAM_KEY[row["kind"]]
            if row["kind"] == "random_reorientation":
                spec[key] = row["param"] or "octahedral"
            else:
                v = _float(row["param"])
                if v is None:
                    raise ValueError(f"Mode {i}: the parameter must be a number.")
                spec[key] = v
            if row["kind"] == "ring_jump":
                if row.get("sense") in ("forward", "backward", "random"):
                    spec["model"] = row["sense"]
                if row.get("step"):
                    step = _float(row["step"])
                    if step is None:
                        raise ValueError(f"Axis {i}: the step must be a number of degrees "
                                         "(or blank for 360°/n).")
                    spec["step_deg"] = step
            k, Ea = _float(row["k"]), _float(row["Ea"])
            if (k is None) == (Ea is None):
                raise ValueError(f"Mode {i}: give either k or Ea (with ν₀).")
            if k is not None:
                spec["k_Hz"] = k
            else:
                spec["Ea_eV"] = Ea
                spec["nu0_Hz"] = _float(row["nu0"], 1e12)
            modes.append(spec)
        hops = []
        n = self.sites_table.rowCount()
        for r in range(self.hops_table.rowCount()):
            a, b = _float(self._cell(self.hops_table, r, 0)), _float(self._cell(self.hops_table, r, 1))
            if a is None or b is None or not (1 <= int(a) <= n and 1 <= int(b) <= n) or int(a) == int(b):
                raise ValueError(f"Hop {r + 1} must join two different environments among 1–{n}.")
            hop = {"from": int(a) - 1, "to": int(b) - 1}
            k = _float(self._cell(self.hops_table, r, 2))
            Ea = _float(self._cell(self.hops_table, r, 3))
            if (k is None) == (Ea is None):
                raise ValueError(f"Hop {r + 1}: give either k → or Ea →.")
            if k is not None:
                hop["k_Hz"] = k
            else:
                hop["Ea_eV"] = Ea
                hop["nu0_Hz"] = _float(self._cell(self.hops_table, r, 5), 1e12)
                Ea_r = _float(self._cell(self.hops_table, r, 4))
                if Ea_r is not None:
                    hop["Ea_rev_eV"] = Ea_r
            hops.append(hop)
        return M.model_from_spec(sites, modes, hops)

    # ------------------------------------------------------------- sweeps
    def _refresh_sweep_choices(self) -> None:
        if not hasattr(self, "sweep_combo"):
            return
        current = self._picked
        self.sweep_combo.blockSignals(True)
        self.sweep_combo.clear()
        for i, row in enumerate(self._mode_rows() if self.modes_table.rowCount() else []):
            name = next(t for t, k, _w, _d in MODE_KINDS if k == row["kind"])
            self.sweep_combo.addItem(f"Rate of axis {i + 1} ({name}), s⁻¹", f"mode:{i}")
        if self.modes_table.rowCount() > 1:
            self.sweep_combo.addItem("Rate of all axes together, s⁻¹", "all")
        if self.hops_table.rowCount():
            self.sweep_combo.addItem("Hop rates × scale", "hops")
        self.sweep_combo.addItem("Temperature, K (Arrhenius rates)", "T")
        i = self.sweep_combo.findData(current)
        self.sweep_combo.setCurrentIndex(max(i, 0))
        self.sweep_combo.blockSignals(False)
        self._refresh_summary()

    def _refresh_summary(self) -> None:
        """One line under the axes: how many states the model has, or what is missing."""
        if not all(hasattr(self, a) for a in ("summary_label", "hops_table", "modes_table",
                                              "sites_table")):
            return
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                model = self.build_model()
        except Exception as exc:  # noqa: BLE001 -- an unfinished table: say what is missing
            self.summary_label.setText(f"<i>Not complete yet: {exc}</i>")
            return
        parts = [f"{m.n} position{'s' if m.n != 1 else ''}" for m in model.modes]
        text = f"<b>{model.n_states} states</b>"
        if model.modes:
            text += (f" = {model.n_env} environment{'s' if model.n_env != 1 else ''} × "
                     + " × ".join(parts) + " (one factor per axis, each with its own rate)")
        if model.n_states > model.MAX_STATES:
            text += f" – <b>more than {model.MAX_STATES}: slow</b>; use fewer positions"
        self.summary_label.setText(text)

    def _temperature(self):
        v = _float(self.temperature_edit.text())
        if v is not None and v <= 0:
            raise ValueError("The temperature is in kelvin and must be positive.")
        return v

    def _kwargs(self, model, sweep_key=None) -> dict:
        kw = {"larmor_Hz": self.larmor_MHz() * 1e6,
              "gauss_fwhm_ppm": self.gauss_spin.value(),
              "transitions": self.transitions_combo.currentData(),
              "order": self.order_combo.currentData()}
        if self.mas_spin.value() > 0:
            kw["mas_Hz"] = self.mas_spin.value() * 1e3
        if self.echo_combo.currentData():
            kw["echo"] = {"kind": self.echo_combo.currentData(),
                          "tau_s": self.tau_spin.value() * 1e-6}
        if sweep_key != "T":
            T = self._temperature()
            if T is not None:
                kw["T_K"] = T
        return kw

    def _extra_for_code(self, kw: dict) -> dict:
        return {k: v for k, v in kw.items()
                if k not in ("larmor_Hz", "mas_Hz", "echo", "T_K")}

    def _axis(self, model, kw) -> np.ndarray:
        if self.auto_axis_check.isChecked():
            hi, lo = M.motion_axis_limits(model, larmor_Hz=kw["larmor_Hz"], order=kw["order"],
                                          transitions=kw["transitions"],
                                          mas_Hz=kw.get("mas_Hz", 0.0),
                                          gauss_fwhm_ppm=kw["gauss_fwhm_ppm"])
        else:
            hi, lo = self.axis_hi.value(), self.axis_lo.value()
            if hi <= lo:
                raise ValueError("The axis maximum must be above its minimum.")
        return np.linspace(hi, lo, self.points_spin.value())

    def simulate(self) -> bool:
        try:
            model = self.build_model()
            key = self.sweep_combo.currentData()
            rates = parse_values(self.values_edit.text())
            kw = self._kwargs(model, key)
            axis = self._axis(model, kw)
            if key == "T" and not any("Ea_eV" in m.rate for m in model.modes) and not any(
                    "Ea_eV" in h for h in model.hops):
                raise ValueError("A temperature sweep needs an Arrhenius rate (Ea and ν₀) on at "
                                 "least one mode or hop.")
            if key != "T" and self._needs_temperature(model) and "T_K" not in kw:
                raise ValueError("A rate is given as Ea and ν₀: enter the temperature, or sweep "
                                 "over temperature.")
        except (ValueError, KeyError) as exc:
            QMessageBox.warning(self, "Anisotropic motion", str(exc))
            return False
        include = self.limits_check.isChecked()
        nucleus, larmor_MHz = self.nucleus(), kw["larmor_Hz"] / 1e6
        code_kw = dict(kw)

        def compute():
            return M.motion_rate_series(model, rates, which=key, ppm_axis=axis,
                                        include_limits=include, **kw)

        def apply(series, _stopped):
            self.series = series
            self._code = (model, rates, key, code_kw, (float(axis[0]), float(axis[-1])))
            size = _text_size(self.canvas) if self.canvas is not None else 9.0
            if self.canvas is not None:
                _show(self.canvas,
                      lambda fig: M.plot_motion_series(series, ax=fig.add_subplot(111),
                                                       nucleus=nucleus, text_size=size),
                      "compare", hz_per_ppm=larmor_MHz)
                self.canvas.plot_owner = "motion_series"
                self.show_view()
            self._log_series(series)
            self.results_label.setText(
                f"{len(series['spectra'])} spectra, {self.sweep_combo.currentText()}: "
                + ", ".join(f"{r:.3g}" for r in series["rates"]) + ".")
            self._ran("sweep")

        return self._submit("Anisotropic motion sweep", compute, apply)

    def _ran(self, what: str) -> None:
        self._last_run = what
        try:
            self.on_run()
        except Exception:  # noqa: BLE001 -- the journal never breaks a run
            pass

    def rerun(self) -> bool:
        """Run again what was run last (restoring a journalled run): the limits, the overlay
        (when spectra are selected) or the sweep."""
        if self._last_run == "limits":
            return self.show_limits()
        if self._last_run == "overlay" and (self.get_spectra() or []):
            return self.overlay_selected()
        return self.simulate()

    @staticmethod
    def _needs_temperature(model) -> bool:
        return (any("Ea_eV" in m.rate for m in model.modes)
                or any("Ea_eV" in h for h in model.hops))

    def _submit(self, title, compute, apply) -> bool:
        from . import jobs

        def fail(exc):
            QMessageBox.warning(self, "Anisotropic motion", str(exc))

        return jobs.submit(self, jobs.Job(title, compute, apply, fail=fail, watched=False))

    def _log_series(self, series) -> None:
        first = series["results"][0]
        self._log(first["log"])
        seen = set(first["log"].splitlines())
        for r in series["results"][1:]:
            for line in r["log"].splitlines():
                if line.startswith("⚠") and line not in seen:
                    seen.add(line)
                    self._log(f"(at {r['rate']:.3g}) {line}")

    # ------------------------------------------------- limits and tensor
    def show_limits(self) -> bool:
        try:
            model = self.build_model()
            kw = self._kwargs(model)
            if self._needs_temperature(model) and "T_K" not in kw:
                raise ValueError("A rate is given as Ea and ν₀: enter the temperature.")
            axis = self._axis(model, kw)
        except (ValueError, KeyError) as exc:
            QMessageBox.warning(self, "Anisotropic motion", str(exc))
            return False
        lim_kw = {k: kw[k] for k in ("larmor_Hz", "gauss_fwhm_ppm", "transitions", "order", "mas_Hz",
                                     "T_K") if k in kw}
        nucleus = self.nucleus()

        def compute():
            return (M.limit_spectra(model, axis, **lim_kw), M.averaged_tensors(
                model, T_K=lim_kw.get("T_K")))

        def apply(out, _stopped):
            limits, eff = out
            self.limits = limits
            if self.canvas is not None:
                size = _text_size(self.canvas)
                _show(self.canvas,
                      lambda fig: M.plot_motion_limits(limits, ax=fig.add_subplot(111),
                                                       nucleus=nucleus, text_size=size),
                      "compare", hz_per_ppm=kw["larmor_Hz"] / 1e6)
                self.canvas.plot_owner = "motion_limits"
                self.show_view()
            self._log(eff["log"])
            self.results_label.setText("Slow limit, fast limits (frequencies averaged; tensors "
                                       "averaged) and the averaged tensor: see the log.")
            self._ran("limits")

        return self._submit("Anisotropic motion limits", compute, apply)

    # ------------------------------------------------------------- overlay
    def overlay_selected(self) -> bool:
        spectra = self.get_spectra() or []
        if not spectra:
            QMessageBox.information(self, "No spectra",
                                    "Select one or more 1D spectra in Open experiments.")
            return False
        try:
            model = self.build_model()
            key = self.sweep_combo.currentData()
            kw = self._kwargs(model, key if len(spectra) > 1 else key)
            value = _float(self.overlay_value_edit.text())
            if self.overlay_value_edit.text().strip() and value is None:
                raise ValueError("The overlay value must be a number.")
            if value is not None:
                if key == "T":
                    kw["T_K"] = value
                elif key == "hops":
                    kw["hop_scale"] = value
                elif key == "all":
                    kw["k_modes"] = {i: value for i in range(len(model.modes))}
                elif key and key.startswith("mode:"):
                    kw["k_modes"] = {int(key[5:]): value}
            multi = len(spectra) > 1
            kw["n_points"] = self.points_spin.value()
            window = ((self.axis_lo.value(), self.axis_hi.value())
                      if self.window_check.isChecked() and not self.auto_axis_check.isChecked()
                      else None)
            if window is not None:
                kw["window"] = window
            if spectra[0].get("larmor_MHz"):
                kw["larmor_Hz"] = float(spectra[0]["larmor_MHz"]) * 1e6
            needs_T = self._needs_temperature(model) and "T_K" not in kw
            if needs_T and not (multi and all(s.get("T_K") for s in spectra)) and not (
                    not multi and spectra[0].get("T_K")):
                raise ValueError("A rate is given as Ea and ν₀: enter the temperature (or select "
                                 "spectra that have one).")
        except (ValueError, KeyError) as exc:
            QMessageBox.warning(self, "Anisotropic motion", str(exc))
            return False
        nucleus = spectra[0].get("nucleus") or self.nucleus()
        larmor_MHz = kw["larmor_Hz"] / 1e6

        def compute():
            if multi:
                return M.motion_overlay_series(model, spectra, **kw)
            sp = spectra[0]
            one = dict(kw)
            if needs_T:
                one["T_K"] = float(sp["T_K"])
            return M.motion_overlay(model, sp["ppm"], sp["y"], **one)

        def apply(out, _stopped):
            self.overlay = out
            size = _text_size(self.canvas) if self.canvas is not None else 9.0
            if self.canvas is not None:
                if multi:
                    draw = lambda fig: M.plot_motion_overlay_stack(  # noqa: E731
                        out, ax=fig.add_subplot(111), nucleus=nucleus, text_size=size)
                else:
                    draw = lambda fig: M.plot_motion_overlay(  # noqa: E731
                        out, ax=fig.add_subplot(111), nucleus=nucleus, text_size=size)
                _show(self.canvas, draw, "compare", hz_per_ppm=larmor_MHz)
                self.canvas.plot_owner = "motion_overlay"
                self.show_view()
            if multi:
                self._log(out["overlays"][0]["simulation"]["log"])
                self._log(out["log"])
                self.results_label.setText("Overlaid " + str(len(out["overlays"])) + " spectra.")
            else:
                self._log(out["log"])
                self.results_label.setText(
                    f"Overlay: scale {out['scale']:.4g}, offset {out['offset']:.4g}, rms residual "
                    f"{100 * out['rms']:.2f} % of the maximum.")
            self._ran("overlay")

        return self._submit("Anisotropic motion overlay", compute, apply)

    # ---------------------------------------------------------- notebook
    def notebook_code(self) -> str:
        model = self.build_model()
        key = self.sweep_combo.currentData()
        rates = parse_values(self.values_edit.text())
        kw = self._kwargs(model, key)
        axis = self._axis(model, kw)
        overlay = {}
        value = _float(self.overlay_value_edit.text())
        if value is not None:
            if key == "T":
                overlay["T_K"] = value
            elif key == "hops":
                overlay["hop_scale"] = value
            elif key == "all":
                overlay["k_modes"] = {i: value for i in range(len(model.modes))}
            elif key and key.startswith("mode:"):
                overlay["k_modes"] = {int(key[5:]): value}
        T = kw.get("T_K")
        return M.motion_notebook_code(
            model, rates=rates, which=key, larmor_MHz=kw["larmor_Hz"] / 1e6, T_K=T,
            mas_Hz=kw.get("mas_Hz"), echo=kw.get("echo"),
            ppm_range=(float(axis[0]), float(axis[-1])), extra=self._extra_for_code(kw),
            overlay=overlay)

    def copy_notebook_code(self) -> None:
        try:
            code = self.notebook_code()
        except (ValueError, KeyError) as exc:
            QMessageBox.warning(self, "Copy notebook code", str(exc))
            return
        _copy(code)
        self._log("Copied notebook code to the clipboard.")

    # -------------------------------------------------------------- state
    def get_state(self) -> dict:
        return {
            "sites": [{"site": self.sites_table.item(r, 0).data(Qt.ItemDataRole.UserRole),
                       "euler": [self._cell(self.sites_table, r, c) for c in (3, 4, 5)],
                       "dipolar": self._cell(self.sites_table, r, 6)}
                      for r in range(self.sites_table.rowCount())],
            "modes": self._mode_rows(),
            "hops": [[self._cell(self.hops_table, r, c) for c in range(len(self.HOP_COLUMNS))]
                     for r in range(self.hops_table.rowCount())],
            "field_T": self.field_spin.value(), "order": self.order_combo.currentData(),
            "transitions": self.transitions_combo.currentData(),
            "mas_kHz": self.mas_spin.value(), "echo": self.echo_combo.currentData(),
            "tau_us": self.tau_spin.value(), "gauss_ppm": self.gauss_spin.value(),
            "auto_axis": self.auto_axis_check.isChecked(),
            "axis": [self.axis_hi.value(), self.axis_lo.value(), self.points_spin.value()],
            "temperature": self.temperature_edit.text(), "sweep": self.sweep_combo.currentData(),
            "values": self.values_edit.text(), "limits": self.limits_check.isChecked(),
            "overlay_value": self.overlay_value_edit.text(),
            "extra": self._readable_state(),
        }

    def _readable_state(self) -> dict:
        """What the simulation log prints: the tables, one line per row."""
        out = {}
        for r in range(self.sites_table.rowCount()):
            out[f"Environment {r + 1}"] = " | ".join(
                self._cell(self.sites_table, r, c) for c in range(self.sites_table.columnCount()))
        for r in range(self.modes_table.rowCount()):
            out[f"Mode {r + 1}"] = ", ".join(f"{k} {v}" for k, v in self._mode_rows()[r].items()
                                              if v not in (None, ""))
        for r in range(self.hops_table.rowCount()):
            out[f"Hop {r + 1}"] = " | ".join(
                self._cell(self.hops_table, r, c) for c in range(self.hops_table.columnCount()))
        return out

    def set_state(self, state: dict) -> None:
        state = state or {}
        if "sites" in state:
            self.sites_table.setRowCount(0)
            for s in state["sites"]:
                e = [_float(v, 0.0) for v in s.get("euler", (0, 0, 0))]
                self.add_site(Site.from_dict(s["site"]), euler=e, dipolar=s.get("dipolar", ""))
        if "modes" in state:
            self._load_modes(state["modes"])
        if "hops" in state:
            self.hops_table.setRowCount(0)
            for h in state["hops"]:
                self.add_hop(h)
        self._refresh_sweep_choices()
        for widget, key, setter in (
                (self.field_spin, "field_T", self.field_spin.setValue),
                (self.mas_spin, "mas_kHz", self.mas_spin.setValue),
                (self.tau_spin, "tau_us", self.tau_spin.setValue),
                (self.gauss_spin, "gauss_ppm", self.gauss_spin.setValue)):
            if key in state:
                setter(float(state[key]))
        for combo, key in ((self.order_combo, "order"), (self.transitions_combo, "transitions"),
                           (self.echo_combo, "echo"), (self.sweep_combo, "sweep")):
            if key in state:
                combo.setCurrentIndex(max(combo.findData(state[key]), 0))
        self._picked = state.get("sweep")
        if "auto_axis" in state:
            self.auto_axis_check.setChecked(bool(state["auto_axis"]))
        if "axis" in state:
            self.axis_hi.setValue(state["axis"][0])
            self.axis_lo.setValue(state["axis"][1])
            self.points_spin.setValue(int(state["axis"][2]))
        self.temperature_edit.setText(state.get("temperature", ""))
        self.values_edit.setText(state.get("values", self.values_edit.text()))
        self.limits_check.setChecked(state.get("limits", True))
        self.overlay_value_edit.setText(state.get("overlay_value", ""))

    def apply_fitted_model(self, model_dict: dict, names) -> list[str]:
        """Write the fitted values of ``names`` (motion_fit parameter names, 1-based) from
        ``model_dict`` (a fitted model's ``to_dict()``) back into the tables. Returns the names
        written. Rates are written as the pair that defines them (Ea and ν₀, or k)."""
        done = []
        for name in names:
            kind, i, _j, key = _parse_name(name)
            if kind == "gauss":
                continue
            if kind == "mode" and i < self.modes_table.rowCount():
                rate = model_dict["modes"][i]["rate"]
                if rate.get("k_Hz") is not None:
                    self._set_row_from(self.modes_table, i, 3, [f"{rate['k_Hz']:.6g}", "", ""])
                else:
                    self._set_row_from(self.modes_table, i, 3, [
                        "", f"{rate['Ea_eV']:.6g}", f"{rate.get('nu0_Hz', 1e12):.6g}"])
                done.append(name)
            elif kind == "hop" and i < self.hops_table.rowCount():
                hop = model_dict["hops"][i]
                if hop.get("k_Hz") is not None:
                    self._set_row_from(self.hops_table, i, 2, [f"{hop['k_Hz']:.6g}", "", "", ""])
                else:
                    Ea_r = hop.get("Ea_rev_eV")
                    self._set_row_from(self.hops_table, i, 2, [
                        "", f"{hop['Ea_eV']:.6g}", "" if Ea_r is None else f"{Ea_r:.6g}",
                        f"{hop.get('nu0_Hz', 1e12):.6g}"])
                done.append(name)
            elif kind == "site" and i < self.sites_table.rowCount():
                entry = model_dict["sites"][i]
                if key in ("alpha", "beta", "gamma"):
                    c = 3 + "abg".index(key[0])
                    self._set_row_from(self.sites_table, i, c, [f"{entry['euler'][c - 3]:.6g}"])
                elif key == "D_Hz":          # written as partner:D(kHz), whichever way it was given
                    text = ", ".join(f"{d['partner']}:{d['D_Hz'] / 1e3:.6g}"
                                     for d in entry["dipolar"])
                    self._set_row_from(self.sites_table, i, 6, [text])
                else:
                    site = Site.from_dict(entry["site"])
                    item = self.sites_table.item(i, 0)
                    item.setData(Qt.ItemDataRole.UserRole, site.to_dict())
                    self.sites_table.blockSignals(True)
                    self.sites_table.item(i, 1).setText(f"{site.delta_iso:g}")
                    self.sites_table.item(i, 2).setText(_interaction_text(site) or "isotropic")
                    self.sites_table.blockSignals(False)
                done.append(name)
        return done

    def _log(self, text: str) -> None:
        self.status_log.appendPlainText(text)
