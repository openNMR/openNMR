"""J-coupling multiplets in the GUI: Simulate > J-coupling and
Fit > J-coupling.

Both panels edit the same thing -- a list of chemical sites, each with
its shift, linewidth and a list of coupling *groups* (partner nuclide,
how many equivalent partners, J) -- through one shared editor,
:class:`JCouplingSitesEditor`: a sites table, and under it the couplings
of whichever site is selected. A fit's result can be taken back as the
next starting point, or sent to the simulator.

The physics is all in :mod:`opennmr.notebook.simulations`
(``multi_site_jcoupled_spectrum``) and
:mod:`opennmr.notebook.jcoupling_fitting` (``fit_jcoupling_spectrum``);
this module collects inputs, calls those, and shows the results. As in
those modules the treatment is first order (weak coupling): no roofing
or other strong-coupling distortion.

Linewidths are entered in Hz (as J is) and converted to ppm with the
observe frequency; shifts are in ppm.
"""

from __future__ import annotations

import json
from io import StringIO
from typing import Callable

import numpy as np
from .widgets.typed_spinbox import TypedSpinBox
from .widgets.report_label import ReportLabel
from .widgets.ranges_edit import PpmRangesEdit
from .widgets.field_input import MAX_FIELD_T, FieldSpinBox, proton_frequency_box
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont, QGuiApplication
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout,
    QHBoxLayout, QHeaderView, QInputDialog, QLabel, QMenu, QMessageBox, QPushButton,
    QStackedWidget, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from ..core import thread_io
from . import jobs
from ..core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
from ..io import serialize
from . import panel_state
from ..notebook import fitting as notebook_fitting
from ..notebook import jcoupling_fitting as jfit
from ..notebook import mixed_j as nmixed
from ..notebook import simulations as nsim
from ..notebook import spin_system as nss
from ..notebook.utils import _GAMMA_DF, compute_BF1_SFO1, find_isotope_info

#: Offered in the Partner column's tooltip and the observe-nucleus list;
#: any nuclide in the table can be typed.
COMMON_NUCLIDES = ("1H", "2H", "13C", "15N", "19F", "29Si", "31P", "11B", "14N",
                   "77Se", "103Rh", "117Sn", "119Sn", "183W", "195Pt", "199Hg", "207Pb")

_LINESHAPES = (("Lorentzian", "lorentzian"), ("Gaussian", "gaussian"),
               ("Pseudo-Voigt", "pseudo_voigt"))

_FIXED_FONT = QFont()
_FIXED_FONT.setItalic(True)


def _element(nuclide: str) -> str:
    return nuclide.lstrip("0123456789")


def other_magnetic_isotopes(nuclide: str, min_abundance: float = 1.0) -> list[tuple[str, float]]:
    """Other NMR-active isotopes of ``nuclide``'s element above
    ``min_abundance`` %. The simulator treats the rest of a partner's
    element as spin-0 (the 12C of a 13C satellite pattern) -- right for
    C, Si, Pt, Sn..., but not for B, Cl, Br, Cu, N, where the other
    isotope is magnetic and has its own, differently spaced multiplet."""
    element = _element(nuclide)
    out = []
    for iso, spin, ab in zip(_GAMMA_DF["Nuclide"], _GAMMA_DF["Spin"],
                             _GAMMA_DF["Natural Abundance (%)"]):
        iso = str(iso)
        if iso == nuclide or _element(iso) != element:
            continue
        try:
            if float(spin) > 0 and float(ab) >= min_abundance:
                out.append((iso, float(ab)))
        except (TypeError, ValueError):
            continue
    return out


def _txt(value) -> str:
    """A cell's text: numbers compactly, text (as typed) unchanged,
    None as blank."""
    if value is None:
        return ""
    if isinstance(value, (int, float, np.floating, np.integer)):
        return f"{float(value):g}" if isinstance(value, (float, np.floating)) else str(value)
    return str(value)


def _item(text, editable=True) -> QTableWidgetItem:
    item = QTableWidgetItem(str(text))
    if not editable:
        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
    return item


class JCouplingSitesEditor(QWidget):
    """Sites and their coupling groups.

    ``mode="simulate"`` adds a relative-intensity column; ``mode="fit"``
    adds right-click Fix / Set bounds on shifts, linewidths and Js
    (fixed values shown in italics). Emits :attr:`changed` on any edit.
    """

    changed = Signal()

    SITE_COLUMNS = ("Label", "Shift (ppm)", "FWHM (Hz)")
    COUPLING_COLUMNS = ("Partner", "n", "J (Hz)", "Abund. (%)", "All isotopes", "Label")
    _ALL_ISOTOPES_COL = 4
    #: Role on a site row's label cell: that site's coupling list.
    _COUPLINGS_ROLE = Qt.ItemDataRole.UserRole
    #: Role on a site row's label cell: {"fixed": {...}, "bounds": {...}}.
    _CONSTRAINTS_ROLE = Qt.ItemDataRole.UserRole + 1

    def __init__(self, mode: str = "simulate", parent=None) -> None:
        super().__init__(parent)
        self.mode = mode
        columns = list(self.SITE_COLUMNS) + (["Intensity"] if mode == "simulate" else [])
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)

        self.sites_table = QTableWidget(0, len(columns))
        self.sites_table.setHorizontalHeaderLabels(columns)
        self.sites_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.sites_table.verticalHeader().setVisible(False)
        self.sites_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.sites_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.sites_table.setMinimumHeight(90)
        self.sites_table.setMaximumHeight(160)
        tip = ("One row per chemical site: its uncoupled shift and the width of each "
               "multiplet line. Select a site to edit its couplings below.")
        if mode == "simulate":
            tip += ("\n\nIntensity: the site's relative number of observed nuclei (e.g. 3 for "
                    "a CH3 beside a CH2 at 2).")
        else:
            tip += "\n\nRight-click a shift or width to fix it or bound it."
            self.sites_table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
            self.sites_table.customContextMenuRequested.connect(self._on_sites_menu)
        self.sites_table.setToolTip(tip)
        self.sites_table.itemChanged.connect(lambda _i: self._emit())
        self.sites_table.itemSelectionChanged.connect(self._show_selected_couplings)
        v.addWidget(self.sites_table)

        row = QHBoxLayout()
        self.add_site_button = QPushButton("Add site")
        self.add_site_button.clicked.connect(lambda: self.add_site())
        self.remove_site_button = QPushButton("Remove site")
        self.remove_site_button.clicked.connect(self.remove_site)
        row.addWidget(self.add_site_button)
        row.addWidget(self.remove_site_button)
        row.addStretch(1)
        v.addLayout(row)

        self.couplings_label = QLabel("Couplings of the selected site:")
        v.addWidget(self.couplings_label)
        self.couplings_table = QTableWidget(0, len(self.COUPLING_COLUMNS))
        self.couplings_table.setHorizontalHeaderLabels(list(self.COUPLING_COLUMNS))
        self.couplings_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.couplings_table.verticalHeader().setVisible(False)
        self.couplings_table.setMinimumHeight(80)
        self.couplings_table.setMaximumHeight(150)
        ctip = ("One row per group of equivalent coupling partners.\n"
                "Partner: the coupled nuclide, e.g. " + ", ".join(COMMON_NUCLIDES[:8]) + " …\n"
                "n: how many equivalent partners (2 for a CH2 neighbour).\n"
                "J: the coupling in Hz (its sign cannot be seen in a first-order multiplet).\n"
                "Abundance: blank = the natural abundance of that isotope, so a 13C partner "
                "gives 1.07% satellites; enter 100 for an enriched sample.\n"
                "All isotopes: include every magnetic isotope of the partner's element at its "
                "natural abundance – J is to the named isotope, the others' Js follow from "
                "their gyromagnetic ratios (e.g. 11B ticked also gives the 10B septet).\n"
                "Label: optional, e.g. 3J(H-H), used in the reported notation.")
        if mode == "fit":
            ctip += "\n\nRight-click a J to fix it or bound it."
            self.couplings_table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
            self.couplings_table.customContextMenuRequested.connect(self._on_couplings_menu)
        self.couplings_table.setToolTip(ctip)
        self.couplings_table.itemChanged.connect(self._on_couplings_edited)
        v.addWidget(self.couplings_table)

        row = QHBoxLayout()
        self.add_coupling_button = QPushButton("Add coupling")
        self.add_coupling_button.clicked.connect(lambda: self.add_coupling())
        self.remove_coupling_button = QPushButton("Remove coupling")
        self.remove_coupling_button.clicked.connect(self.remove_coupling)
        row.addWidget(self.add_coupling_button)
        row.addWidget(self.remove_coupling_button)
        row.addStretch(1)
        v.addLayout(row)

        self.pattern_label = QLabel("")
        self.pattern_label.setWordWrap(True)
        v.addWidget(self.pattern_label)
        self._sync_buttons()

    # ---- editing ---------------------------------------------------------
    def _emit(self) -> None:
        self._update_pattern_label()
        self.changed.emit()

    def _selected_row(self) -> int:
        rows = self.sites_table.selectionModel().selectedRows()
        return rows[0].row() if rows else -1

    def add_site(self, label=None, shift=0.0, fwhm_hz=1.0, intensity=1.0, couplings=None,
                 constraints=None) -> int:
        t = self.sites_table
        t.blockSignals(True)
        r = t.rowCount()
        t.insertRow(r)
        name = label or self._unique_label()
        label_item = _item(name)
        label_item.setData(self._COUPLINGS_ROLE,
                           [dict(c) for c in (couplings if couplings is not None
                                              else [{"nuclide": "1H", "n": 1, "J_Hz": 7.0}])])
        label_item.setData(self._CONSTRAINTS_ROLE, constraints or {"fixed": {}, "bounds": {}})
        t.setItem(r, 0, label_item)
        t.setItem(r, 1, _item(f"{shift:g}"))
        t.setItem(r, 2, _item(f"{fwhm_hz:g}"))
        if self.mode == "simulate":
            t.setItem(r, 3, _item(f"{intensity:g}"))
        t.blockSignals(False)
        self._restyle_site(r)
        t.selectRow(r)
        self._sync_buttons()
        self._emit()
        return r

    def get_state(self) -> dict:
        """The sites table, with each site's couplings and constraints
        (held in its items' data)."""
        state = {"widgets": panel_state.capture_widgets(self, names=["sites_table"])}
        lines = []
        for r in range(self.sites_table.rowCount()):
            label = self.sites_table.item(r, 0).text() if self.sites_table.item(r, 0) else f"Site {r + 1}"
            couplings = ", ".join(f"{c.get('n', 1)}×{c.get('nuclide', '?')} {c.get('J_Hz', '?')} Hz"
                                  for c in self._site_couplings(r))
            if couplings:
                lines.append((f"{label} couplings", couplings))
        if lines:
            state["extra"] = dict(lines)
        return state

    def set_state(self, state: dict) -> None:
        panel_state.restore_widgets(self, (state or {}).get("widgets") or {})
        for r in range(self.sites_table.rowCount()):
            if self.sites_table.item(r, 0) is not None:
                self._restyle_site(r)
        if self.sites_table.rowCount():
            self.sites_table.selectRow(0)
        self._show_selected_couplings()
        self._sync_buttons()
        self._emit()

    def _unique_label(self) -> str:
        used = {self.sites_table.item(r, 0).text() for r in range(self.sites_table.rowCount())
                if self.sites_table.item(r, 0)}
        k = self.sites_table.rowCount() + 1
        while f"Site {k}" in used:
            k += 1
        return f"Site {k}"

    def remove_site(self) -> None:
        r = self._selected_row()
        if r < 0:
            r = self.sites_table.rowCount() - 1
        if r >= 0:
            self.sites_table.removeRow(r)
        self._show_selected_couplings()
        self._sync_buttons()
        self._emit()

    def clear(self) -> None:
        self.sites_table.setRowCount(0)
        self._show_selected_couplings()
        self._sync_buttons()
        self._emit()

    def _site_couplings(self, r: int) -> list[dict]:
        item = self.sites_table.item(r, 0)
        return [dict(c) for c in (item.data(self._COUPLINGS_ROLE) or [])] if item else []

    def _set_site_couplings(self, r: int, couplings: list[dict]) -> None:
        item = self.sites_table.item(r, 0)
        self.sites_table.blockSignals(True)
        item.setData(self._COUPLINGS_ROLE, couplings)
        self.sites_table.blockSignals(False)

    def _show_selected_couplings(self) -> None:
        r = self._selected_row()
        t = self.couplings_table
        t.blockSignals(True)
        t.setRowCount(0)
        if r >= 0:
            label = self.sites_table.item(r, 0).text()
            self.couplings_label.setText(f"Couplings of {label}:")
            for c in self._site_couplings(r):
                i = t.rowCount()
                t.insertRow(i)
                for col, text in enumerate((c.get("nuclide", ""), _txt(c.get("n", 1)),
                                            _txt(c.get("J_Hz", 0.0)), _txt(c.get("abundance")),
                                            "", c.get("bond_label") or "")):
                    t.setItem(i, col, _item(text, editable=col != self._ALL_ISOTOPES_COL))
                tick = t.item(i, self._ALL_ISOTOPES_COL)
                tick.setFlags(tick.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                tick.setCheckState(Qt.CheckState.Checked if c.get("all_isotopes")
                                   else Qt.CheckState.Unchecked)
                self._restyle_coupling(i, c)
        else:
            self.couplings_label.setText("Couplings of the selected site:")
        t.blockSignals(False)
        self._sync_buttons()
        self._update_pattern_label()

    def _on_couplings_edited(self, _item=None) -> None:
        """Write the couplings table back into the selected site, keeping
        each row's fixed/bounds flags."""
        r = self._selected_row()
        if r < 0:
            return
        old = self._site_couplings(r)
        new = []
        for i in range(self.couplings_table.rowCount()):
            cell = lambda c: (self.couplings_table.item(i, c).text().strip()  # noqa: E731
                              if self.couplings_table.item(i, c) else "")
            entry = dict(old[i]) if i < len(old) else {}
            entry["nuclide"] = cell(0)
            entry["n"] = cell(1)
            entry["J_Hz"] = cell(2)
            entry["abundance"] = cell(3) or None
            tick = self.couplings_table.item(i, self._ALL_ISOTOPES_COL)
            entry["all_isotopes"] = bool(tick and tick.checkState() == Qt.CheckState.Checked)
            entry["bond_label"] = cell(5) or None
            new.append(entry)
        self._set_site_couplings(r, new)
        self._emit()

    def add_coupling(self, nuclide="1H", n=1, J_Hz=7.0) -> None:
        r = self._selected_row()
        if r < 0:
            return
        couplings = self._site_couplings(r) + [{"nuclide": nuclide, "n": n, "J_Hz": J_Hz}]
        self._set_site_couplings(r, couplings)
        self._show_selected_couplings()
        self._emit()

    def remove_coupling(self) -> None:
        r = self._selected_row()
        if r < 0:
            return
        couplings = self._site_couplings(r)
        rows = {i.row() for i in self.couplings_table.selectedIndexes()}
        i = max(rows) if rows else len(couplings) - 1
        if 0 <= i < len(couplings):
            couplings.pop(i)
        self._set_site_couplings(r, couplings)
        self._show_selected_couplings()
        self._emit()

    def _sync_buttons(self) -> None:
        has_site = self._selected_row() >= 0
        self.remove_site_button.setEnabled(self.sites_table.rowCount() > 0)
        self.add_coupling_button.setEnabled(has_site)
        self.remove_coupling_button.setEnabled(has_site and self.couplings_table.rowCount() > 0)
        self.couplings_table.setEnabled(has_site)

    # ---- reading ---------------------------------------------------------
    @staticmethod
    def _parse_coupling(site_label, c) -> dict:
        nuclide = str(c.get("nuclide", "")).strip()
        try:
            find_isotope_info(nuclide)
        except KeyError:
            raise ValueError(f"{site_label}: '{nuclide}' is not a nuclide in the table "
                             f"(write it like 1H, 31P, 13C).") from None
        try:
            n = int(str(c.get("n")).strip())
            J = float(str(c.get("J_Hz")).strip())
        except (TypeError, ValueError):
            raise ValueError(f"{site_label}, {nuclide}: n must be a whole number and J a "
                             f"number in Hz.") from None
        if n < 1:
            raise ValueError(f"{site_label}, {nuclide}: n must be at least 1.")
        out = {"nuclide": nuclide, "n": n, "J_Hz": J}
        if c.get("all_isotopes"):
            if c.get("abundance") not in (None, ""):
                raise ValueError(f"{site_label}, {nuclide}: 'All isotopes' uses natural "
                                 f"abundances – clear the abundance, or untick it.")
            out["all_isotopes"] = True
        ab = c.get("abundance")
        if ab not in (None, ""):
            try:
                ab = float(ab)
            except (TypeError, ValueError):
                raise ValueError(f"{site_label}, {nuclide}: abundance must be a number (%), "
                                 f"or blank for natural abundance.") from None
            if not 0 < ab <= 100:
                raise ValueError(f"{site_label}, {nuclide}: abundance must be in (0, 100] %.")
            out["abundance"] = ab
        if c.get("bond_label"):
            out["bond_label"] = str(c["bond_label"])
        for key in ("fixed_J", "bounds_J"):
            if c.get(key):
                out[key] = c[key]
        return out

    def sites(self, larmor_Hz: float) -> list[dict]:
        """The sites in the notebook's format (fwhm in ppm, via
        ``larmor_Hz``). Raises ValueError, naming the cell, for anything
        unreadable."""
        out = []
        labels = set()
        t = self.sites_table
        if t.rowCount() == 0:
            raise ValueError("Add at least one site.")
        for r in range(t.rowCount()):
            label = t.item(r, 0).text().strip() or f"Site {r + 1}"
            if label in labels:
                raise ValueError(f"Two sites are called '{label}'.")
            labels.add(label)
            try:
                shift = float(t.item(r, 1).text())
                fwhm_hz = float(t.item(r, 2).text())
            except (AttributeError, ValueError):
                raise ValueError(f"{label}: shift (ppm) and FWHM (Hz) must be numbers.") from None
            if fwhm_hz <= 0:
                raise ValueError(f"{label}: FWHM must be > 0 Hz.")
            site = {"label": label, "center_ppm": shift, "fwhm": fwhm_hz / larmor_Hz * 1e6,
                    "couplings": [self._parse_coupling(label, c) for c in self._site_couplings(r)]}
            if self.mode == "simulate":
                try:
                    site["relative_intensity"] = float(t.item(r, 3).text())
                except (AttributeError, ValueError):
                    raise ValueError(f"{label}: intensity must be a number.") from None
            else:
                cons = t.item(r, 0).data(self._CONSTRAINTS_ROLE) or {}
                fixed, bounds = dict(cons.get("fixed", {})), dict(cons.get("bounds", {}))
                if "fwhm" in bounds:        # entered in Hz, like the column
                    lo, hi = bounds["fwhm"]
                    bounds["fwhm"] = (lo / larmor_Hz * 1e6, hi / larmor_Hz * 1e6)
                site["fixed"], site["bounds"] = fixed, bounds
            out.append(site)
        return out

    def set_sites(self, sites: list[dict], larmor_Hz: float) -> None:
        """Replace the table with ``sites`` (notebook format, fwhm in ppm)."""
        self.sites_table.setRowCount(0)
        for s in sites:
            # To the precision a fit or a spectrum supports: 1e-6 ppm,
            # 1e-4 Hz; not every float digit.
            self.add_site(label=s.get("label"), shift=round(float(s["center_ppm"]), 6),
                          fwhm_hz=round(float(s["fwhm"]) * larmor_Hz / 1e6, 4),
                          intensity=float(f"{float(s.get('relative_intensity', 1.0)):.4g}"),
                          couplings=[{**{k: v for k, v in c.items()
                                         if k not in ("fixed_J", "bounds_J")},
                                      "J_Hz": round(float(c["J_Hz"]), 4)}
                                     for c in s.get("couplings", [])])
        if self.sites_table.rowCount():
            self.sites_table.selectRow(0)

    # ---- the selected site's pattern, stated -----------------------------
    def _update_pattern_label(self) -> None:
        r = self._selected_row()
        if r < 0:
            self.pattern_label.setText("")
            return
        label = self.sites_table.item(r, 0).text()
        try:
            couplings = [self._parse_coupling(label, c) for c in self._site_couplings(r)]
        except ValueError as exc:
            self.pattern_label.setText(f"⚠ {exc}")
            return
        if not couplings:
            self.pattern_label.setText(f"{label}: singlet (no couplings).")
            return
        topology = [{"nuclide": c["nuclide"], "n": c["n"],
                     "spin": find_isotope_info(c["nuclide"])["spin"],
                     "bond_label": c.get("bond_label")} for c in couplings]
        mult, js = jfit.jcoupling_chemist_notation(None, topology, [c["J_Hz"] for c in couplings])
        # (for an 'all isotopes' group the letter is the named isotope's)
        _freqs, w = nsim.combine_jcouplings(couplings)
        major = int(np.sum(w >= 1e-3 * w.max()))
        text = f"{label}: {mult} ({', '.join(js)}); {major} lines"
        if major < w.size:
            text += f" (+{w.size - major} below 0.1% of the tallest, from partners' other isotopes)"
        text += "."
        notes, info_lines = [], []
        for c in couplings:
            ab = c.get("abundance", find_isotope_info(c["nuclide"])["abundance"])
            if c.get("all_isotopes"):
                isos, zero = nsim.element_isotopes(c["nuclide"])
                parts = ", ".join(f"{i['nuclide']} {i['abundance']:g}%" for i in isos)
                info_lines.append(f"{c['nuclide']} group, all isotopes: {parts}"
                                  + (f", spin-0 {zero:.3g}%" if zero > 0.05 else "") + ".")
                continue
            if ab < 99.0:
                notes.append(f"{c['nuclide']} at {ab:g}%: most molecules have a non-magnetic "
                             f"isotope there, so this coupling shows as satellites.")
            others = other_magnetic_isotopes(c["nuclide"])
            if others and "abundance" not in c:
                iso = ", ".join(f"{i} ({a:g}%)" for i, a in others)
                notes.append(f"{iso} is also magnetic but is treated here as non-magnetic: its "
                             f"own multiplet is not simulated. Tick 'All isotopes' to include "
                             f"it, or enter 100% abundance to model pure {c['nuclide']}.")
        self.pattern_label.setText("\n".join([text] + info_lines + [f"⚠ {n}" for n in notes]))

    # ---- fixing and bounding (fit mode) -----------------------------------
    _SITE_PARAMS = {1: ("center_ppm", "Shift (ppm)"), 2: ("fwhm", "FWHM (Hz)")}

    def _restyle_site(self, r: int) -> None:
        if self.mode != "fit":
            return
        cons = self.sites_table.item(r, 0).data(self._CONSTRAINTS_ROLE) or {}
        for col, (name, _label) in self._SITE_PARAMS.items():
            item = self.sites_table.item(r, col)
            if item is None:
                continue
            fixed = cons.get("fixed", {}).get(name, False)
            bounds = cons.get("bounds", {}).get(name)
            item.setFont(_FIXED_FONT if fixed else QFont())
            item.setToolTip("fixed" if fixed else (f"bounded to [{bounds[0]:g}, {bounds[1]:g}]"
                                                   if bounds else ""))

    def _restyle_coupling(self, i: int, c: dict) -> None:
        item = self.couplings_table.item(i, 2)
        if item is None:
            return
        item.setFont(_FIXED_FONT if c.get("fixed_J") else QFont())
        b = c.get("bounds_J")
        item.setToolTip("fixed" if c.get("fixed_J") else
                        (f"bounded to [{b[0]:g}, {b[1]:g}] Hz" if b else ""))

    def set_site_constraint(self, r: int, name: str, fixed: bool | None = None,
                            bounds: tuple | None | bool = False) -> None:
        """fixed: True/False to set, None to leave. bounds: (lo, hi) to
        set, None to clear, False to leave."""
        item = self.sites_table.item(r, 0)
        cons = item.data(self._CONSTRAINTS_ROLE) or {}
        cons = {"fixed": dict(cons.get("fixed", {})), "bounds": dict(cons.get("bounds", {}))}
        if fixed is not None:
            cons["fixed"][name] = fixed
        if bounds is None:
            cons["bounds"].pop(name, None)
        elif bounds is not False:
            cons["bounds"][name] = tuple(bounds)
        self.sites_table.blockSignals(True)
        item.setData(self._CONSTRAINTS_ROLE, cons)
        self.sites_table.blockSignals(False)
        self._restyle_site(r)
        self._emit()

    def set_coupling_constraint(self, r: int, i: int, fixed: bool | None = None,
                                bounds: tuple | None | bool = False) -> None:
        couplings = self._site_couplings(r)
        c = couplings[i]
        if fixed is not None:
            c["fixed_J"] = fixed
        if bounds is None:
            c.pop("bounds_J", None)
        elif bounds is not False:
            c["bounds_J"] = tuple(bounds)
        self._set_site_couplings(r, couplings)
        if r == self._selected_row():
            self._show_selected_couplings()
        self._emit()

    def _ask_bounds(self, what: str, current) -> tuple | None | bool:
        text, ok = QInputDialog.getText(
            self, "Set bounds", f"{what}: low, high (blank to remove the bounds)",
            text="" if not current else f"{current[0]:g}, {current[1]:g}")
        if not ok:
            return False
        if not text.strip():
            return None
        try:
            lo, hi = (float(p) for p in text.replace(";", ",").split(","))
        except ValueError:
            QMessageBox.warning(self, "Set bounds", f"Expected 'low, high', got {text!r}.")
            return False
        if lo >= hi:
            QMessageBox.warning(self, "Set bounds", "Low must be below high.")
            return False
        return (lo, hi)

    def _on_sites_menu(self, pos) -> None:
        index = self.sites_table.indexAt(pos)
        if not index.isValid() or index.column() not in self._SITE_PARAMS:
            return
        r, (name, label) = index.row(), self._SITE_PARAMS[index.column()]
        cons = self.sites_table.item(r, 0).data(self._CONSTRAINTS_ROLE) or {}
        menu = QMenu(self)
        fix = menu.addAction(f"Fix {label}")
        fix.setCheckable(True)
        fix.setChecked(bool(cons.get("fixed", {}).get(name)))
        bounds = menu.addAction(f"Set bounds on {label}…")
        chosen = menu.exec(self.sites_table.viewport().mapToGlobal(pos))
        if chosen is fix:
            self.set_site_constraint(r, name, fixed=fix.isChecked())
        elif chosen is bounds:
            b = self._ask_bounds(label, cons.get("bounds", {}).get(name))
            if b is not False:
                self.set_site_constraint(r, name, bounds=b)

    def _on_couplings_menu(self, pos) -> None:
        index = self.couplings_table.indexAt(pos)
        r = self._selected_row()
        if not index.isValid() or index.column() != 2 or r < 0:
            return
        c = self._site_couplings(r)[index.row()]
        menu = QMenu(self)
        fix = menu.addAction("Fix J")
        fix.setCheckable(True)
        fix.setChecked(bool(c.get("fixed_J")))
        bounds = menu.addAction("Set bounds on J…")
        chosen = menu.exec(self.couplings_table.viewport().mapToGlobal(pos))
        if chosen is fix:
            self.set_coupling_constraint(r, index.row(), fixed=fix.isChecked())
        elif chosen is bounds:
            b = self._ask_bounds("J (Hz)", c.get("bounds_J"))
            if b is not False:
                self.set_coupling_constraint(r, index.row(), bounds=b)


class SpinSystemEditor(QWidget):
    """An explicit spin system for exact (strong-coupling) simulation and
    fitting: a spins table (label, nucleus, shift, n magnetically
    equivalent copies) and the J matrix between them.

    The J matrix is edited in its upper triangle; the lower mirrors it.
    Blank or 0 means uncoupled (and, when fitting, not fitted). In
    ``mode="fit"`` a shift or J can be right-clicked to fix or bound it.
    ``get_B0()`` gives the field in tesla (or None) for the strength
    readout under the tables.
    """

    changed = Signal()
    SPIN_COLUMNS = ("Label", "Nucleus", "Shift (ppm)", "n")

    def __init__(self, mode: str = "simulate", get_B0=None, parent=None) -> None:
        super().__init__(parent)
        self.mode = mode
        self._get_B0 = get_B0 or (lambda: None)
        self._J: list[list[str]] = []           # upper-triangle text, aligned with rows
        self._constraints: dict[str, dict] = {}  # engine parameter name -> fixed/bounds
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        self.spins_table = QTableWidget(0, len(self.SPIN_COLUMNS))
        self.spins_table.setHorizontalHeaderLabels(list(self.SPIN_COLUMNS))
        self.spins_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.spins_table.verticalHeader().setVisible(False)
        self.spins_table.setMinimumHeight(90)
        self.spins_table.setMaximumHeight(170)
        tip = ("One row per spin (or set of magnetically equivalent spins, n > 1: e.g. a CH3 "
               "as one row with n = 3). Spins of the observed nucleus give the spectrum; "
               "others (19F, 31P...) couple to them. Every spin is its NMR-active isotope: "
               "the system is one isotopomer.")
        if mode == "fit":
            tip += "\n\nRight-click a shift to fix it or bound it."
            self.spins_table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
            self.spins_table.customContextMenuRequested.connect(self._on_spins_menu)
        self.spins_table.setToolTip(tip)
        self.spins_table.itemChanged.connect(self._on_spins_edited)
        v.addWidget(self.spins_table)
        row = QHBoxLayout()
        self.add_spin_button = QPushButton("Add spin")
        self.add_spin_button.clicked.connect(lambda: self.add_spin())
        self.remove_spin_button = QPushButton("Remove spin")
        self.remove_spin_button.clicked.connect(self.remove_spin)
        row.addWidget(self.add_spin_button)
        row.addWidget(self.remove_spin_button)
        row.addStretch(1)
        v.addLayout(row)
        v.addWidget(QLabel("J (Hz) between spins:"))
        self.j_table = QTableWidget(0, 0)
        self.j_table.setMinimumHeight(90)
        self.j_table.setMaximumHeight(190)
        jtip = ("Couplings in Hz, upper triangle (the lower mirrors it). Blank or 0 = "
                "uncoupled. Signs matter when spins are strongly coupled (relative signs "
                "change ABX-type patterns); a first-order multiplet cannot show them.")
        if mode == "fit":
            jtip += "\n\nRight-click a J to fix it or bound it."
            self.j_table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
            self.j_table.customContextMenuRequested.connect(self._on_j_menu)
        self.j_table.setToolTip(jtip)
        self.j_table.itemChanged.connect(self._on_j_edited)
        v.addWidget(self.j_table)
        self.info_label = QLabel("")
        self.info_label.setWordWrap(True)
        v.addWidget(self.info_label)

    # ---- editing ---------------------------------------------------------
    def labels(self) -> list[str]:
        return [(self.spins_table.item(r, 0).text().strip() if self.spins_table.item(r, 0)
                 else "") for r in range(self.spins_table.rowCount())]

    def add_spin(self, label=None, nuclide="1H", shift=0.0, n=1) -> None:
        t = self.spins_table
        t.blockSignals(True)
        r = t.rowCount()
        t.insertRow(r)
        used = set(self.labels())
        if label is None:
            k = r
            label = chr(ord("A") + k) if k < 26 else f"S{k + 1}"
            while label in used:
                k += 1
                label = chr(ord("A") + k) if k < 26 else f"S{k + 1}"
        for c, text in enumerate((label, nuclide, _txt(float(shift)), _txt(int(n)))):
            t.setItem(r, c, _item(text))
        t.blockSignals(False)
        for row in self._J:
            row.append("")
        self._J.append([""] * (r + 1))
        self._rebuild_j()
        self._emit()

    def remove_spin(self) -> None:
        rows = sorted({i.row() for i in self.spins_table.selectedIndexes()})
        r = rows[0] if rows else self.spins_table.rowCount() - 1
        if r < 0:
            return
        self.spins_table.removeRow(r)
        del self._J[r]
        for row in self._J:
            del row[r]
        self._rebuild_j()
        self._emit()

    def clear(self) -> None:
        self.spins_table.setRowCount(0)
        self._J = []
        self._constraints = {}
        self._rebuild_j()
        self._emit()

    def get_state(self) -> dict:
        """The spins table, the J matrix and the fixes/bounds -- the last
        two live in Python, not in the widgets."""
        labels = self.labels()
        js = [f"{labels[i]}–{labels[j]} {self._J[i][j]} Hz"
              for i in range(len(self._J)) for j in range(i + 1, len(self._J[i]))
              if i < len(labels) and j < len(labels) and str(self._J[i][j]).strip()]
        state = {"widgets": panel_state.capture_widgets(self, names=["spins_table"]),
                 "model": {"J": serialize.encode(self._J),
                           "constraints": serialize.encode(self._constraints)}}
        if js:
            state["extra"] = {"J couplings": ", ".join(js)}
        return state

    def set_state(self, state: dict) -> None:
        state = state or {}
        panel_state.restore_widgets(self, state.get("widgets") or {})
        model = state.get("model") or {}
        n = self.spins_table.rowCount()
        J = [list(map(str, row)) for row in (serialize.decode(model.get("J")) or [])]
        self._J = [(J[i] if i < len(J) else []) + [""] * max(0, n - (len(J[i]) if i < len(J) else 0))
                   for i in range(n)]
        self._constraints = serialize.decode(model.get("constraints")) or {}
        self._rebuild_j()
        self._restyle_spins()
        self._emit()

    def set_J(self, a: str, b: str, J_Hz) -> None:
        labels = self.labels()
        i, j = sorted((labels.index(a), labels.index(b)))
        self._J[i][j] = "" if J_Hz in (None, "") else _txt(float(J_Hz))
        self._rebuild_j()
        self._emit()

    def _rebuild_j(self) -> None:
        labels = self.labels()
        n = len(labels)
        t = self.j_table
        t.blockSignals(True)
        t.setRowCount(n)
        t.setColumnCount(n)
        t.setHorizontalHeaderLabels(labels)
        t.setVerticalHeaderLabels(labels)
        grey = QColor(128, 128, 128, 50)
        for i in range(n):
            for j in range(n):
                if i == j:
                    item = _item("—", editable=False)
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEnabled)
                elif i < j:
                    item = _item(self._J[i][j])
                    name = self._j_name(labels[i], labels[j])
                    item.setFont(_FIXED_FONT if self._constraints.get(name, {}).get("fixed")
                                 else QFont())
                else:
                    item = _item(self._J[j][i], editable=False)
                    item.setData(Qt.ItemDataRole.BackgroundRole, grey)
                t.setItem(i, j, item)
        t.resizeColumnsToContents()
        t.blockSignals(False)

    def _on_spins_edited(self, item) -> None:
        if item.column() == 0:
            self._rebuild_j()
        self._restyle_spins()
        self._emit()

    def _on_j_edited(self, item) -> None:
        i, j = item.row(), item.column()
        if i < j:
            self._J[i][j] = item.text().strip()
            self._rebuild_j()
            self._emit()

    def _emit(self) -> None:
        self.refresh_info()
        self.changed.emit()

    # ---- reading ---------------------------------------------------------
    @staticmethod
    def _j_name(a, b):
        a, b = sorted((a, b))
        return f"J {a}-{b} (Hz)"

    def spins(self) -> list[dict]:
        out, seen = [], set()
        if self.spins_table.rowCount() == 0:
            raise ValueError("Add at least one spin.")
        for r in range(self.spins_table.rowCount()):
            cell = lambda c: (self.spins_table.item(r, c).text().strip()  # noqa: E731
                              if self.spins_table.item(r, c) else "")
            label = cell(0)
            if not label:
                raise ValueError(f"Spin {r + 1} has no label.")
            if label in seen:
                raise ValueError(f"Two spins are labelled '{label}'.")
            seen.add(label)
            nuclide = cell(1)
            try:
                if find_isotope_info(nuclide)["spin"] <= 0:
                    raise KeyError
            except KeyError:
                raise ValueError(f"{label}: '{nuclide}' is not a magnetic nuclide in the "
                                 f"table.") from None
            try:
                shift = float(cell(2))
                n = int(cell(3) or 1)
            except ValueError:
                raise ValueError(f"{label}: shift must be a number (ppm) and n a whole "
                                 f"number.") from None
            if n < 1:
                raise ValueError(f"{label}: n must be at least 1.")
            out.append({"label": label, "nuclide": nuclide, "shift_ppm": shift, "n": n})
        return out

    def couplings(self) -> list[dict]:
        labels = self.labels()
        out = []
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                text = self._J[i][j].strip()
                if not text:
                    continue
                try:
                    J = float(text)
                except ValueError:
                    raise ValueError(f"J {labels[i]}-{labels[j]}: '{text}' is not a number "
                                     f"(Hz).") from None
                if J != 0.0:
                    out.append({"a": labels[i], "b": labels[j], "J_Hz": J})
        return out

    def constraints(self) -> tuple[dict, dict]:
        """(fixed, bounds) keyed by the engine's parameter names, for
        parameters that still exist."""
        labels = set(self.labels())
        valid = {f"shift {l} (ppm)" for l in labels}
        valid |= {c_name for c_name in self._constraints if c_name.startswith("J ")}
        fixed = {k: True for k, v in self._constraints.items() if v.get("fixed") and k in valid}
        bounds = {k: tuple(v["bounds"]) for k, v in self._constraints.items()
                  if v.get("bounds") and k in valid}
        pairs = {self._j_name(c["a"], c["b"]) for c in self.couplings()}
        fixed = {k: v for k, v in fixed.items() if not k.startswith("J ") or k in pairs}
        bounds = {k: v for k, v in bounds.items() if not k.startswith("J ") or k in pairs}
        return fixed, bounds

    def set_system(self, spins: list[dict], couplings: list[dict]) -> None:
        self.clear()
        for s in spins:
            self.add_spin(label=s["label"], nuclide=s["nuclide"],
                          shift=round(float(s["shift_ppm"]), 6), n=int(s.get("n", 1)))
        for c in couplings:
            self.set_J(c["a"], c["b"], round(float(c["J_Hz"]), 4))

    # ---- readout ---------------------------------------------------------
    def refresh_info(self) -> None:
        try:
            spins, couplings = self.spins(), self.couplings()
        except ValueError as exc:
            self.info_label.setText(f"⚠ {exc}")
            return
        lines = []
        B0 = self._get_B0()
        try:
            biggest = max(nss.block_sizes(spins, couplings).values())
            n_spins = sum(s["n"] for s in spins)
            lines.append(f"{n_spins} spins; largest Hamiltonian block {biggest} × {biggest}"
                         + (" – too large (limit %d)" % nss.MAX_BLOCK_DIM
                            if biggest > nss.MAX_BLOCK_DIM else
                            (" – expect a few seconds per simulation" if biggest > 250 else "")))
            if B0:
                strong = []
                for c in nss.coupling_strengths(spins, couplings, B0):
                    tag = "strong" if c["ratio"] < 3 else ("intermediate" if c["ratio"] < 10
                                                           else "weak")
                    strong.append(f"{c['a']}–{c['b']} {c['ratio']:.2g} ({tag})")
                if strong:
                    lines.append("|Δν/J|: " + ", ".join(strong))
        except (ValueError, KeyError) as exc:
            lines.append(f"⚠ {exc}")
        self.info_label.setText("\n".join(lines))

    # ---- fixing and bounding (fit mode) -----------------------------------
    def _restyle_spins(self) -> None:
        if self.mode != "fit":
            return
        self.spins_table.blockSignals(True)
        for r, label in enumerate(self.labels()):
            item = self.spins_table.item(r, 2)
            if item is not None:
                c = self._constraints.get(f"shift {label} (ppm)", {})
                item.setFont(_FIXED_FONT if c.get("fixed") else QFont())
        self.spins_table.blockSignals(False)

    def set_constraint(self, name: str, fixed: bool | None = None,
                       bounds: tuple | None | bool = False) -> None:
        c = dict(self._constraints.get(name, {}))
        if fixed is not None:
            c["fixed"] = fixed
        if bounds is None:
            c.pop("bounds", None)
        elif bounds is not False:
            c["bounds"] = tuple(bounds)
        self._constraints[name] = c
        self._restyle_spins()
        self._rebuild_j()
        self._emit()

    def _menu(self, widget, pos, name, what) -> None:
        c = self._constraints.get(name, {})
        menu = QMenu(self)
        fix = menu.addAction(f"Fix {what}")
        fix.setCheckable(True)
        fix.setChecked(bool(c.get("fixed")))
        bounds = menu.addAction(f"Set bounds on {what}…")
        chosen = menu.exec(widget.viewport().mapToGlobal(pos))
        if chosen is fix:
            self.set_constraint(name, fixed=fix.isChecked())
        elif chosen is bounds:
            b = JCouplingSitesEditor._ask_bounds(self, what, c.get("bounds"))
            if b is not False:
                self.set_constraint(name, bounds=b)

    def _on_spins_menu(self, pos) -> None:
        index = self.spins_table.indexAt(pos)
        if not index.isValid() or index.column() != 2:
            return
        label = self.labels()[index.row()]
        self._menu(self.spins_table, pos, f"shift {label} (ppm)", f"the shift of {label}")

    def _on_j_menu(self, pos) -> None:
        index = self.j_table.indexAt(pos)
        if not index.isValid() or index.row() == index.column():
            return
        labels = self.labels()
        a, b = labels[index.row()], labels[index.column()]
        self._menu(self.j_table, pos, self._j_name(a, b), f"J {a}-{b}")


def _model_combo(mixed: bool = False) -> QComboBox:
    combo = QComboBox()
    combo.addItem("First order: sites and coupling groups", "first_order")
    combo.addItem("Spin system: exact, with strong coupling", "spin_system")
    if mixed:
        combo.addItem("Mixed: weak and strong sets together", "mixed")
    combo.setToolTip(
        "First order builds each site's multiplet from its coupling groups – fast, handles "
        "natural-abundance satellites and isotopes, but assumes every shift difference is much "
        "larger than J.\n\nSpin system diagonalises the full Hamiltonian of explicit spins: "
        "exact for any shifts and Js (roofing, AB, ABX, AA'BB', virtual coupling), up to about "
        "12 spins."
        + ("\n\nMixed puts both on one spectrum: first-order (weak) sites and exactly solved "
           "(strong) spin sets, added together. Use it when only some of the multiplets are "
           "roofed. The sets must not be coupled to each other." if mixed else ""))
    return combo


#: The blurb of a J panel hosted in Solid-state NMR's strong-coupling
#: mode, where first-order J is an interaction of a peak instead.
STRONG_COUPLING_BLURB = (
    "Strongly coupled peaks: every spin's shift and the J values between them, solved exactly "
    "(roofing, AB, ABX, AA'BB', virtual coupling), up to about 12 spins. Use this when J is "
    "comparable to the shift difference between peaks; otherwise add a first-order J-coupling "
    "to a peak in the peaks-with-interactions model.")


#: The same panel's blurb when hosted in Solution NMR (Fit and Simulate),
#: where strong coupling is the ordinary way to treat a roofed spectrum.
SOLUTION_STRONG_BLURB = (
    "Strongly coupled spin system: every spin's shift and the J values between them, solved "
    "exactly (roofing, AB quartets, ABX, AA'BB', virtual coupling), up to about 12 spins. Use "
    "this when J is comparable to the shift difference between multiplets; when every shift "
    "difference is much larger than J, first-order peaks and multiplets are quicker.")


#: The simulator's blurb in Solution NMR, where weak (first-order), strong
#: (exact) and mixed spectra are all ordinary (explicit user request, 2026-11-02).
SOLUTION_SIM_BLURB = (
    "Solution-state J-coupling. Weak coupling (first order) builds each multiplet from its "
    "coupling groups and is exact when every shift difference is much larger than J. Strong "
    "coupling solves every spin's shift and the J between them exactly (roofing, AB quartets, "
    "ABX, AA'BB', virtual coupling; up to about 12 spins). Mixed puts both on one spectrum: "
    "weak sites and strong spin sets added together, e.g. an ethyl group beside an aromatic "
    "AB pair. Choose the model below.")


def lock_to_spin_system(panel) -> None:
    """Show a J panel as the exact spin-system model only: its first-order
    mode is Solid-state NMR's J interaction now (explicit user request,
    2026-10-06: one Combined entry point for every interaction)."""
    panel.model_combo.setCurrentIndex(panel.model_combo.findData("spin_system"))
    panel.model_combo.setVisible(False)
    # The example spins the editor opens with: replaced by the peaks on
    # the first switch from Combined (see seed_spins_from_peaks).
    panel._example_spin_state = panel.spin_editor.get_state()
    panel.blurb_label.setText(STRONG_COUPLING_BLURB)
    # A first-order fit or simulation restored from before (kept fits,
    # the history) still opens as it was made, with the switch shown.
    panel.model_combo.currentIndexChanged.connect(lambda _i: _sync_model_switch(panel))


def _sync_model_switch(panel) -> None:
    """Show the model switch in Solution NMR (weak / strong / mixed), and in
    Solid-state NMR only for a model restored from before (which opens as it
    was made)."""
    panel.model_combo.setVisible(getattr(panel, "host_key", "solid") == "solution"
                                 or panel.model_combo.currentData() != "spin_system")


def seed_spins_from_peaks(panel, site_dicts, nucleus: str) -> bool:
    """Start the spin system from Combined peaks (label and δiso, one spin
    each), when it has no spins yet or only the example it opens with --
    never over spins someone typed. Returns whether it was seeded."""
    editor = panel.spin_editor
    untouched = editor.get_state() == getattr(panel, "_example_spin_state", None)
    if (editor.spins_table.rowCount() and not untouched) or not site_dicts:
        return False
    spins = [{"label": str(d.get("label") or f"S{k + 1}").replace(" ", ""),
              "nuclide": nucleus or "1H", "shift_ppm": float(d.get("delta_iso", 0.0)), "n": 1}
             for k, d in enumerate(site_dicts)]
    seen = set()
    for k, sp in enumerate(spins):
        if sp["label"] in seen:
            sp["label"] = f"{sp['label']}_{k + 1}"
        seen.add(sp["label"])
    editor.set_system(spins, [])
    return True


def _as_simulator_sites(sites: list[dict]) -> list[dict]:
    """The fitter's site format (center_ppm) as multi_site_jcoupled_spectrum
    takes it (ppm)."""
    return [{"label": s.get("label"), "ppm": s["center_ppm"], "fwhm": s["fwhm"],
             "relative_intensity": s.get("relative_intensity", 1.0),
             "couplings": s["couplings"]} for s in sites]


def _lineshape_controls(form: QFormLayout, owner) -> tuple[QComboBox, QDoubleSpinBox]:
    combo = QComboBox()
    for label, key in _LINESHAPES:
        combo.addItem(label, key)
    combo.setToolTip("Shape of every multiplet line. Lorentzian is the natural shape of a "
                     "line in a well-shimmed solution spectrum.")
    form.addRow("Lineshape:", combo)
    eta = QDoubleSpinBox()
    eta.setRange(0.0, 1.0)
    eta.setSingleStep(0.1)
    eta.setValue(0.5)
    eta.setToolTip("Pseudo-Voigt Lorentzian fraction: 0 = Gaussian, 1 = Lorentzian.")
    form.addRow("L fraction:", eta)

    def sync(_i=None):
        form.setRowVisible(eta, combo.currentData() == "pseudo_voigt")
    combo.currentIndexChanged.connect(sync)
    sync()
    return combo, eta


# ============================================================================
# Simulate > J-coupling
# ============================================================================
class JCouplingSimulationPanel(QWidget):
    """Simulate a first-order J-coupled spectrum of one or more sites.

    ``plot_spectrum(spectrum)`` draws the result; ``add_to_open(spectrum,
    label)`` puts it in Open experiments.
    """

    def __init__(self, plot_spectrum: Callable[[NMRSpectrum], None],
                 add_to_open: Callable[[NMRSpectrum, str], None], parent=None,
                 send_to_fit=None) -> None:
        super().__init__(parent)
        self._plot_spectrum = plot_spectrum
        self._add_to_open = add_to_open
        self._send_to_fit = send_to_fit
        self.spectrum: NMRSpectrum | None = None
        self._last_args: dict | None = None

        v = QVBoxLayout(self)
        blurb = QLabel("Build a J-coupled spectrum. First order: one row per chemical site and "
                       "the groups of nuclei it couples to (quick; valid when shift differences "
                       "are much larger than J). Spin system: every spin's shift and every J value, "
                       "solved exactly (leading to roofing, AB/ABX patterns and other strong-coupling "
                       "effects).")
        blurb.setWordWrap(True)
        v.addWidget(blurb)
        self.blurb_label = blurb

        form = QFormLayout()
        self.nucleus_combo = QComboBox()
        self.nucleus_combo.setEditable(True)
        self.nucleus_combo.addItems(COMMON_NUCLIDES)
        self.nucleus_combo.setToolTip("The observed nucleus – sets the observe frequency that "
                                      "converts J (Hz) to ppm, and the axis label.")
        form.addRow("Observe:", self.nucleus_combo)
        self.field_spin = FieldSpinBox()
        self.field_spin.setRange(0.01, MAX_FIELD_T)
        self.field_spin.setDecimals(6)
        self.field_spin.setValue(9.3950)
        self.field_spin.setSuffix(" T")
        self.field_spin.setToolTip("Magnet field – or type its ¹H frequency beside it. "
                                   "9.395 T is a '400 MHz' spectrometer.")
        # Not saved separately: it is the field, in other units.
        self._field_proton_spin = proton_frequency_box(self.field_spin, parent=self)
        field_row = QHBoxLayout()
        field_row.addWidget(self.field_spin, 1)
        field_row.addWidget(QLabel("or"))
        field_row.addWidget(self._field_proton_spin, 1)
        form.addRow("Field:", field_row)
        self.frequency_label = QLabel("")
        form.addRow("", self.frequency_label)
        self.lineshape_combo, self.eta_spin = _lineshape_controls(form, self)
        v.addLayout(form)

        self.model_combo = _model_combo(mixed=True)
        v.addWidget(self.model_combo)
        self.editor = JCouplingSitesEditor("simulate")
        self.spin_editor = SpinSystemEditor("simulate", get_B0=self.field_spin.value)
        spin_page = QWidget()
        spin_layout = QVBoxLayout(spin_page)
        spin_layout.setContentsMargins(0, 0, 0, 0)
        spin_layout.addWidget(self.spin_editor)
        spin_form = QFormLayout()
        self.spin_fwhm_spin = QDoubleSpinBox()
        self.spin_fwhm_spin.setRange(0.001, 1e5)
        self.spin_fwhm_spin.setDecimals(3)
        self.spin_fwhm_spin.setValue(1.0)
        self.spin_fwhm_spin.setSuffix(" Hz")
        self.spin_fwhm_spin.setToolTip("Width (FWHM) of every line of the spin system.")
        spin_form.addRow("Linewidth:", self.spin_fwhm_spin)
        spin_layout.addLayout(spin_form)
        self.model_stack = QStackedWidget()
        self.model_stack.addWidget(self.editor)
        self.model_stack.addWidget(spin_page)
        # Mixed: weak sites and strong spin sets on one spectrum (its own
        # editors: the two pages above keep theirs, untouched).
        mixed_page = QWidget()
        mixed_layout = QVBoxLayout(mixed_page)
        mixed_layout.setContentsMargins(0, 0, 0, 0)
        mixed_note = QLabel(
            "Weak sites: first-order multiplets, one row per site (valid when shift differences "
            "are much larger than J). Strong spins: solved exactly. Spins joined by a J are one "
            "set; sets with no J between them are solved separately, so any number of strong "
            "sets can sit beside the weak ones. A weak site and a strong set must not be coupled "
            "to each other.")
        mixed_note.setWordWrap(True)
        mixed_layout.addWidget(mixed_note)
        mixed_layout.addWidget(QLabel("<b>Weak (first-order) sites</b>"))
        self.mixed_weak_editor = JCouplingSitesEditor("simulate")
        mixed_layout.addWidget(self.mixed_weak_editor)
        mixed_layout.addWidget(QLabel("<b>Strong (exact) spins and the J between them</b>"))
        self.mixed_strong_editor = SpinSystemEditor("simulate", get_B0=self.field_spin.value)
        mixed_layout.addWidget(self.mixed_strong_editor)
        mixed_form = QFormLayout()
        self.mixed_fwhm_spin = QDoubleSpinBox()
        self.mixed_fwhm_spin.setRange(0.001, 1e5)
        self.mixed_fwhm_spin.setDecimals(3)
        self.mixed_fwhm_spin.setValue(1.0)
        self.mixed_fwhm_spin.setSuffix(" Hz")
        self.mixed_fwhm_spin.setToolTip("Width (FWHM) of every line of the strong sets. Each "
                                        "weak site has its own width in its table row.")
        mixed_form.addRow("Strong linewidth:", self.mixed_fwhm_spin)
        mixed_layout.addLayout(mixed_form)
        self.model_stack.addWidget(mixed_page)
        self.model_combo.currentIndexChanged.connect(self.model_stack.setCurrentIndex)
        v.addWidget(self.model_stack)

        axis = QFormLayout()
        self.auto_axis_check = QCheckBox("Auto (every line ± 10 × FWHM)")
        self.auto_axis_check.setChecked(True)
        axis.addRow("Axis:", self.auto_axis_check)
        self.ppm_min_spin = QDoubleSpinBox()
        self.ppm_max_spin = QDoubleSpinBox()
        for spin, value in ((self.ppm_min_spin, 0.0), (self.ppm_max_spin, 10.0)):
            spin.setRange(-1e5, 1e5)
            spin.setDecimals(4)
            spin.setValue(value)
            spin.setSuffix(" ppm")
        axis.addRow("Axis min:", self.ppm_min_spin)
        axis.addRow("Axis max:", self.ppm_max_spin)
        self.points_spin = TypedSpinBox()
        self.points_spin.setRange(0, 1 << 21)
        self.points_spin.setSpecialValueText("auto")
        self.points_spin.setValue(0)
        self.points_spin.setToolTip("Axis points. Auto: at least 12 points across the narrowest "
                                    "line (so the plotted peaks are smooth), 4096 to 2 M.")
        axis.addRow("Axis points:", self.points_spin)
        self._axis_form = axis
        v.addLayout(axis)

        self.simulate_button = QPushButton("Simulate")
        self.simulate_button.clicked.connect(self.simulate)
        v.addWidget(self.simulate_button)
        self.results_label = ReportLabel("")
        self.results_label.setWordWrap(True)
        self.results_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        v.addWidget(self.results_label)
        row = QHBoxLayout()
        self.export_button = QPushButton("Keep in Open Experiments")
        self.export_button.setToolTip("Add the simulated spectrum to Open experiments, to "
                                      "overlay it on measured data on the Compare tile.")
        self.export_button.clicked.connect(self.export)
        row.addWidget(self.export_button)
        self.to_fit_button = QPushButton("Send to Fit as Starting Values")
        self.to_fit_button.setToolTip("Put these sites (or this spin system) into Fit ▸ "
                                      "J-coupling as the starting model.")
        self.to_fit_button.setEnabled(send_to_fit is not None)
        self.to_fit_button.clicked.connect(self.send_to_fit)
        row.addWidget(self.to_fit_button)
        self.copy_code_button = QPushButton("Copy notebook code")
        self.copy_code_button.setToolTip("The multi_site_jcoupled_spectrum call for the last "
                                         "simulation, to rerun it in a notebook.")
        self.copy_code_button.clicked.connect(self._copy_code)
        row.addWidget(self.copy_code_button)
        v.addLayout(row)
        v.addStretch(1)

        self.editor.add_site(label="CH2", shift=1.25, fwhm_hz=1.0,
                             couplings=[{"nuclide": "1H", "n": 3, "J_Hz": 7.0}])
        # A strongly coupled AB pair as the spin-system example.
        self.spin_editor.set_system(
            [{"label": "A", "nuclide": "1H", "shift_ppm": 7.30},
             {"label": "B", "nuclide": "1H", "shift_ppm": 7.26}],
            [{"a": "A", "b": "B", "J_Hz": 8.0}])
        # An ethyl group (weak: its shifts are far apart) beside an aromatic
        # AB pair (strong), as the mixed example.
        self.mixed_weak_editor.add_site(label="CH3", shift=1.25, fwhm_hz=1.0,
                                        couplings=[{"nuclide": "1H", "n": 2, "J_Hz": 7.0}])
        self.mixed_weak_editor.add_site(label="CH2", shift=2.65, fwhm_hz=1.0,
                                        couplings=[{"nuclide": "1H", "n": 3, "J_Hz": 7.0}])
        self.mixed_strong_editor.set_system(
            [{"label": "A", "nuclide": "1H", "shift_ppm": 7.30},
             {"label": "B", "nuclide": "1H", "shift_ppm": 7.26}],
            [{"a": "A", "b": "B", "J_Hz": 8.0}])
        for signal in (self.nucleus_combo.currentTextChanged, self.field_spin.valueChanged):
            signal.connect(lambda *_: self._update_frequency())
        self.field_spin.valueChanged.connect(lambda _v: self.spin_editor.refresh_info())
        self.field_spin.valueChanged.connect(lambda _v: self.mixed_strong_editor.refresh_info())
        self.auto_axis_check.toggled.connect(lambda _on: self._sync_axis())
        for w in (self.editor.changed, self.nucleus_combo.currentTextChanged,
                  self.field_spin.valueChanged, self.lineshape_combo.currentIndexChanged,
                  self.eta_spin.valueChanged, self.auto_axis_check.toggled,
                  self.ppm_min_spin.valueChanged, self.ppm_max_spin.valueChanged,
                  self.points_spin.valueChanged, self.spin_editor.changed,
                  self.model_combo.currentIndexChanged, self.spin_fwhm_spin.valueChanged,
                  self.mixed_weak_editor.changed, self.mixed_strong_editor.changed,
                  self.mixed_fwhm_spin.valueChanged):
            w.connect(lambda *_: self._mark_stale())
        self._update_frequency()
        self._sync_axis()
        self._mark_stale()

    def get_state(self) -> dict:
        return panel_state.default_state(self)

    def set_state(self, state: dict) -> None:
        panel_state.restore_default_state(self, state)
        self._update_frequency()
        self._sync_axis()

    def on_hosted_in(self, key: str) -> None:
        """Called by the SpinSystemHost when the panel moves: Solution NMR
        shows the weak / strong / mixed switch, Solid-state NMR (where the
        strong model is Combined's) does not."""
        self.host_key = key
        _sync_model_switch(self)

    def _sync_axis(self) -> None:
        manual = not self.auto_axis_check.isChecked()
        self._axis_form.setRowVisible(self.ppm_min_spin, manual)
        self._axis_form.setRowVisible(self.ppm_max_spin, manual)

    def larmor_Hz(self) -> float:
        """Observe frequency, Hz. Raises ValueError for an unknown nucleus."""
        try:
            bf1, _sfo1, _n = compute_BF1_SFO1(self.nucleus_combo.currentText().strip(),
                                              self.field_spin.value())
        except KeyError:
            raise ValueError(f"'{self.nucleus_combo.currentText()}' is not a nuclide in the "
                             f"table.") from None
        return abs(bf1) * 1e6

    def _update_frequency(self) -> None:
        try:
            obs = self.larmor_Hz() / 1e6
            h = abs(compute_BF1_SFO1("1H", self.field_spin.value())[0])
            self.frequency_label.setText(f"{h:.2f} MHz ¹H; observe frequency {obs:.4f} MHz")
        except ValueError as exc:
            self.frequency_label.setText(f"⚠ {exc}")

    def _mark_stale(self) -> None:
        self.export_button.setEnabled(False)
        if self.spectrum is not None:
            self.results_label.setText(self.results_label.text().split("\n(inputs changed")[0]
                                       + "\n(inputs changed since this simulation – "
                                         "Simulate again to update it)")
        self.spectrum = None

    def spin_mode(self) -> bool:
        return self.model_combo.currentData() == "spin_system"

    def mixed_mode(self) -> bool:
        return self.model_combo.currentData() == "mixed"

    def _axis_from(self, lo_list, hi_list, min_width):
        if self.auto_axis_check.isChecked():
            ppm_min, ppm_max = min(lo_list), max(hi_list)
        else:
            ppm_min, ppm_max = self.ppm_min_spin.value(), self.ppm_max_spin.value()
            if ppm_max <= ppm_min:
                raise ValueError("Axis max must be above axis min.")
        n = self.points_spin.value()
        if n == 0:
            n = int(np.clip(12 * (ppm_max - ppm_min) / min_width, 4096, 1 << 21))
        elif n < 3:
            raise ValueError("Use at least 3 axis points.")
        return ppm_min, ppm_max, n

    def _spin_arguments(self) -> dict:
        nucleus = self.nucleus_combo.currentText().strip()
        larmor = self.larmor_Hz()
        spins, couplings = self.spin_editor.spins(), self.spin_editor.couplings()
        if not any(s["nuclide"] == nucleus for s in spins):
            raise ValueError(f"The spin system has no {nucleus} spin to observe – change "
                             f"Observe, or a spin's nucleus.")
        t = nss.spin_system_transitions(spins, couplings, self.field_spin.value(), observe=nucleus)
        fwhm_ppm = self.spin_fwhm_spin.value() / larmor * 1e6
        strong = t["intensity"] > 1e-4 * t["intensity"].max()
        ppm_min, ppm_max, n = self._axis_from([t["stick_ppm"][strong].min() - 10 * fwhm_ppm],
                                              [t["stick_ppm"][strong].max() + 10 * fwhm_ppm],
                                              fwhm_ppm)
        return {"model": "spin_system", "spins": spins, "couplings": couplings,
                "fwhm_Hz": self.spin_fwhm_spin.value(), "larmor_Hz": larmor,
                "ppm_min": ppm_min, "ppm_max": ppm_max, "n_points": n,
                "lineshape": self.lineshape_combo.currentData(), "eta": self.eta_spin.value(),
                "nucleus": nucleus, "B0_tesla": self.field_spin.value(), "transitions": t}

    def _mixed_arguments(self) -> dict:
        nucleus = self.nucleus_combo.currentText().strip()
        larmor = self.larmor_Hz()
        weak_editor, strong_editor = self.mixed_weak_editor, self.mixed_strong_editor
        weak = weak_editor.sites(larmor) if weak_editor.sites_table.rowCount() else []
        spins = strong_editor.spins() if strong_editor.spins_table.rowCount() else []
        couplings = strong_editor.couplings() if spins else []
        if not weak and not spins:
            raise ValueError("Add at least one weak site or one strong spin.")
        fwhm_ppm = self.mixed_fwhm_spin.value() / larmor * 1e6
        lo, hi, widths = [], [], []
        for s in weak:
            freqs, _w = nsim.combine_jcouplings(s["couplings"])
            sticks = s["center_ppm"] + freqs / larmor * 1e6
            lo.append(sticks.min() - 10 * s["fwhm"])
            hi.append(sticks.max() + 10 * s["fwhm"])
            widths.append(s["fwhm"])
        n_sets = 0
        for set_spins, set_couplings in (nmixed.split_spin_system(spins, couplings) if spins
                                         else []):
            if not any(sp["nuclide"] == nucleus for sp in set_spins):
                continue
            n_sets += 1
            t = nss.spin_system_transitions(set_spins, set_couplings, self.field_spin.value(),
                                            observe=nucleus)
            strong = t["intensity"] > 1e-4 * t["intensity"].max()
            lo.append(t["stick_ppm"][strong].min() - 10 * fwhm_ppm)
            hi.append(t["stick_ppm"][strong].max() + 10 * fwhm_ppm)
            widths.append(fwhm_ppm)
        if spins and not n_sets and not weak:
            raise ValueError(f"The strong spins have no {nucleus} spin to observe – change "
                             f"Observe, or a spin's nucleus.")
        ppm_min, ppm_max, n = self._axis_from(lo, hi, min(widths))
        return {"model": "mixed", "sites": weak, "spins": spins, "couplings": couplings,
                "fwhm_Hz": self.mixed_fwhm_spin.value(), "larmor_Hz": larmor,
                "ppm_min": ppm_min, "ppm_max": ppm_max, "n_points": n,
                "lineshape": self.lineshape_combo.currentData(), "eta": self.eta_spin.value(),
                "nucleus": nucleus, "B0_tesla": self.field_spin.value(),
                "min_width_ppm": min(widths)}

    def arguments(self) -> dict:
        """What simulate() passes to the simulator (plus the axis). Raises
        ValueError for anything unreadable."""
        if self.mixed_mode():
            return self._mixed_arguments()
        if self.spin_mode():
            return self._spin_arguments()
        larmor = self.larmor_Hz()
        sites = self.editor.sites(larmor)
        widths = [s["fwhm"] for s in sites]
        if self.auto_axis_check.isChecked():
            lo, hi = [], []
            for s in sites:
                freqs, _w = nsim.combine_jcouplings(s["couplings"])
                sticks = s["center_ppm"] + freqs / larmor * 1e6
                lo.append(sticks.min() - 10 * s["fwhm"])
                hi.append(sticks.max() + 10 * s["fwhm"])
            ppm_min, ppm_max = min(lo), max(hi)
        else:
            ppm_min, ppm_max = self.ppm_min_spin.value(), self.ppm_max_spin.value()
            if ppm_max <= ppm_min:
                raise ValueError("Axis max must be above axis min.")
        n = self.points_spin.value()
        if n == 0:
            n = int(np.clip(12 * (ppm_max - ppm_min) / min(widths), 4096, 1 << 21))
        elif n < 3:
            raise ValueError("Use at least 3 axis points.")
        return {"model": "first_order", "sites": sites, "larmor_Hz": larmor,
                "ppm_min": ppm_min, "ppm_max": ppm_max,
                "n_points": n, "lineshape": self.lineshape_combo.currentData(),
                "eta": self.eta_spin.value(),
                "nucleus": self.nucleus_combo.currentText().strip(),
                "B0_tesla": self.field_spin.value()}

    def simulate(self) -> bool:
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            a = self.arguments()
        except (ValueError, KeyError) as exc:
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self, "Simulate J-coupling", str(exc).strip("'\""))
            return False
        try:
            if a["model"] == "spin_system":
                return self._simulate_spin_system(a)
            if a["model"] == "mixed":
                return self._simulate_mixed(a)
            return self._simulate_first_order(a)
        finally:
            QApplication.restoreOverrideCursor()

    def _finish(self, a, data, axis, raw, lines) -> bool:
        metadata = NMRMetadata(
            vendor="simulated", nucleus=a["nucleus"], sfo1=a["larmor_Hz"] / 1e6,
            bf1=a["larmor_Hz"] / 1e6, raw={"simulation": raw})
        spectrum = NMRSpectrum(data=data, axes=[NMRAxis(kind="frequency", ppm=axis)],
                               metadata=metadata)
        edge = max(abs(data[0]), abs(data[-1])) / (np.max(np.abs(data)) or 1.0)
        if edge > 0.01:
            lines.append(f"⚠ {edge:.0%} of the peak height is at the edge of the axis: part of "
                         f"a multiplet is cut off. Widen the axis or use Auto.")
        width = (a["min_width_ppm"] if a["model"] == "mixed"
                 else a["fwhm_Hz"] / a["larmor_Hz"] * 1e6 if a["model"] == "spin_system"
                 else min(s["fwhm"] for s in a["sites"]))
        spacing = (a["ppm_max"] - a["ppm_min"]) / (a["n_points"] - 1)
        if width / spacing < 3:
            lines.append(f"⚠ Only {width / spacing:.1f} points across the narrowest line: the "
                         f"plotted peaks are under-sampled. Use more axis points.")
        lines.append(f"{a['n_points']:,} points, {a['ppm_min']:.4f} to {a['ppm_max']:.4f} ppm, "
                     f"observe {a['larmor_Hz'] / 1e6:.4f} MHz.")
        self.results_label.setText("\n".join(lines))
        self.spectrum = spectrum
        self._last_args = a
        self.export_button.setEnabled(True)
        self._plot_spectrum(spectrum)
        return True

    def _simulate_spin_system(self, a) -> bool:
        axis = np.linspace(a["ppm_min"], a["ppm_max"], a["n_points"])
        result = nss.spin_system_spectrum(a["spins"], a["couplings"], a["B0_tesla"], axis,
                                          fwhm_Hz=a["fwhm_Hz"], lineshape=a["lineshape"],
                                          eta=a["eta"], observe=a["nucleus"])
        t = a["transitions"]
        w = result["stick_intensity"]
        lines = [f"Spin system: {t['n_observed']} observed {a['nucleus']} spin(s), "
                 f"{int(np.sum(w >= 1e-3 * w.max()))} lines (≥ 0.1% of the tallest), exact "
                 f"diagonalisation (largest block {t['max_block']} × {t['max_block']})."]
        pairs = nss.coupling_strengths(a["spins"], a["couplings"], a["B0_tesla"])
        strong = [f"{c['a']}–{c['b']} ({c['ratio']:.2g})" for c in pairs if c["ratio"] < 10]
        if strong:
            lines.append("Strongly coupled, |Δν/J| < 10: " + ", ".join(strong)
                         + " – a first-order simulation would be visibly wrong here.")
        elif pairs:
            lines.append("Every like-nucleus pair has |Δν/J| ≥ 10: first order would be close.")
        raw = {"kind": "jcoupling_spin_system", "B0_tesla": a["B0_tesla"],
               "lineshape": a["lineshape"], "eta": a["eta"], "fwhm_Hz": a["fwhm_Hz"],
               "spins": a["spins"], "couplings": a["couplings"]}
        return self._finish(a, np.asarray(result["real_spec"], dtype=float), axis, raw, lines)

    @staticmethod
    def _weak_site_lines(nucleus, sites, site_results) -> list[str]:
        lines = []
        for s, r in zip(sites, site_results):
            if s["couplings"]:
                topo = [{"nuclide": c["nuclide"], "n": c["n"],
                         "spin": find_isotope_info(c["nuclide"])["spin"],
                         "bond_label": c.get("bond_label")} for c in s["couplings"]]
                mult, js = jfit.jcoupling_chemist_notation(nucleus, topo,
                                                           [c["J_Hz"] for c in s["couplings"]])
                pattern = f"{mult}, {', '.join(js)}"
            else:
                pattern = "s"
            w = np.asarray(r["stick_intensity"])
            major = int(np.sum(w >= 1e-3 * w.max()))
            lines.append(f"{s['label']}: δ {s['center_ppm']:g} ppm ({pattern}); "
                         f"{major} lines, intensity {s['relative_intensity']:g}.")
        return lines

    def _simulate_mixed(self, a) -> bool:
        axis = np.linspace(a["ppm_min"], a["ppm_max"], a["n_points"])
        result = nmixed.mixed_j_spectrum(
            axis, a["larmor_Hz"], a["B0_tesla"], weak_sites=_as_simulator_sites(a["sites"]),
            strong_spins=a["spins"], strong_couplings=a["couplings"],
            strong_fwhm_Hz=a["fwhm_Hz"], observe=a["nucleus"], lineshape=a["lineshape"],
            eta=a["eta"], weak_fwhm=a["sites"][0]["fwhm"] if a["sites"] else None)
        lines = []
        if result["weak"] is not None:
            lines.append("Weak (first order):")
            lines += ["  " + t for t in self._weak_site_lines(a["nucleus"], a["sites"],
                                                              result["weak"]["sites"])]
        for entry in result["strong"]:
            sub = [s for s in a["spins"] if s["label"] in entry["labels"]]
            sub_j = [c for c in a["couplings"] if c["a"] in entry["labels"]]
            t = entry["result"]["transitions"]
            w = entry["result"]["stick_intensity"]
            lines.append(f"Strong set {', '.join(entry['labels'])}: {t['n_observed']} observed "
                         f"{a['nucleus']} spin(s), {int(np.sum(w >= 1e-3 * w.max()))} lines, "
                         f"exact (largest block {t['max_block']} × {t['max_block']}).")
            weak_pairs = [f"{c['a']}–{c['b']} ({c['ratio']:.2g})" for c in
                          nss.coupling_strengths(sub, sub_j, a["B0_tesla"]) if c["ratio"] < 10]
            if weak_pairs:
                lines.append("  Strongly coupled, |Δν/J| < 10: " + ", ".join(weak_pairs) + ".")
        for labels in result["skipped"]:
            lines.append(f"Set {', '.join(labels)} has no {a['nucleus']} spin: not shown.")
        raw = {"kind": "jcoupling_mixed", "B0_tesla": a["B0_tesla"],
               "lineshape": a["lineshape"], "eta": a["eta"], "fwhm_Hz": a["fwhm_Hz"],
               "sites": json.loads(json.dumps(a["sites"], default=float)),
               "spins": a["spins"], "couplings": a["couplings"]}
        return self._finish(a, np.asarray(result["real_spec"], dtype=float), axis, raw, lines)

    def _simulate_first_order(self, a) -> bool:
        axis = np.linspace(a["ppm_min"], a["ppm_max"], a["n_points"])
        # Each site keeps its own linewidth (the per-site 'fwhm' key).
        result = nsim.multi_site_jcoupled_spectrum(
            _as_simulator_sites(a["sites"]), a["larmor_Hz"], axis, fwhm=a["sites"][0]["fwhm"],
            lineshape=a["lineshape"], eta=a["eta"])
        data = np.asarray(result["real_spec"], dtype=float)
        lines = self._weak_site_lines(a["nucleus"], a["sites"], result["sites"])
        raw = {"kind": "jcoupling", "B0_tesla": a["B0_tesla"], "lineshape": a["lineshape"],
               "eta": a["eta"], "sites": json.loads(json.dumps(a["sites"], default=float))}
        return self._finish(a, data, axis, raw, lines)

    def export(self) -> None:
        if self.spectrum is None:
            return
        a = self._last_args
        items = (a["sites"] + a["spins"] if a["model"] == "mixed"
                 else a["spins"] if a["model"] == "spin_system" else a["sites"])
        names = ", ".join(x["label"] for x in items)
        kind = {"spin_system": "spin system", "mixed": "mixed weak/strong J-coupling"}.get(
            a["model"], "J-coupling")
        self._add_to_open(self.spectrum, f"Simulated {kind} {a['nucleus']} ({names})")

    def send_to_fit(self) -> None:
        if self._send_to_fit is None:
            return
        if self.mixed_mode():
            QMessageBox.information(
                self, "Send to Fit",
                "Fit ▸ J-coupling fits one model at a time. Switch the model to Weak "
                "(first order) or Strong (spin system) to send it as a starting model.")
            return
        try:
            arguments = self.arguments()
        except ValueError as exc:
            QMessageBox.warning(self, "Send to Fit", str(exc))
            return
        self._send_to_fit(arguments)

    def notebook_code(self) -> str:
        a = self._last_args or self.arguments()
        if a["model"] == "mixed":
            return "\n".join([
                "import numpy as np",
                "import opennmr.notebook as nmr",
                "",
                f"larmor_Hz = {a['larmor_Hz']!r}   # {a['nucleus']} at {a['B0_tesla']:g} T",
                f"ppm_axis = np.linspace({a['ppm_min']!r}, {a['ppm_max']!r}, {a['n_points']})",
                "# Weak sites: first-order multiplets (fwhm in ppm).",
                f"weak_sites = {json.dumps(_as_simulator_sites(a['sites']), indent=4, default=float)}",
                "# Strong spins: solved exactly; spins joined by a J form one set.",
                f"spins = {json.dumps(a['spins'], indent=4)}",
                f"couplings = {json.dumps(a['couplings'], indent=4)}",
                "spec = nmr.mixed_j_spectrum(",
                "    ppm_axis, larmor_Hz, B0_tesla=" + repr(a['B0_tesla']) + ",",
                "    weak_sites=weak_sites, strong_spins=spins, strong_couplings=couplings,",
                f"    strong_fwhm_Hz={a['fwhm_Hz']!r}, observe={a['nucleus']!r},",
                f"    lineshape={a['lineshape']!r}, eta={a['eta']!r})",
            ])
        if a["model"] == "spin_system":
            return "\n".join([
                "import numpy as np",
                "import opennmr.notebook as nmr",
                "",
                f"spins = {json.dumps(a['spins'], indent=4)}",
                f"couplings = {json.dumps(a['couplings'], indent=4)}",
                f"ppm_axis = np.linspace({a['ppm_min']!r}, {a['ppm_max']!r}, {a['n_points']})",
                "spec = nmr.spin_system_spectrum(",
                f"    spins, couplings, B0_tesla={a['B0_tesla']!r}, ppm_axis=ppm_axis,",
                f"    fwhm_Hz={a['fwhm_Hz']!r}, lineshape={a['lineshape']!r}, eta={a['eta']!r},",
                f"    observe={a['nucleus']!r})",
            ])
        return "\n".join([
            "import numpy as np",
            "import opennmr.notebook as nmr",
            "",
            f"larmor_Hz = {a['larmor_Hz']!r}   # {a['nucleus']} at {a['B0_tesla']:g} T",
            f"ppm_axis = np.linspace({a['ppm_min']!r}, {a['ppm_max']!r}, {a['n_points']})",
            f"sites = {json.dumps(_as_simulator_sites(a['sites']), indent=4, default=float)}",
            "spec = nmr.multi_site_jcoupled_spectrum(",
            "    sites, larmor_Hz, ppm_axis, fwhm=sites[0]['fwhm'],",
            f"    lineshape={a['lineshape']!r}, eta={a['eta']!r})",
        ])

    def _copy_code(self) -> None:
        try:
            QGuiApplication.clipboard().setText(self.notebook_code())
        except ValueError as exc:
            QMessageBox.information(self, "Simulate J-coupling", str(exc))

    def set_spin_system_from_fit(self, nucleus: str, larmor_Hz: float, spins, couplings,
                                 fwhm_Hz: float, lineshape: str, eta: float) -> None:
        """Take a spin-system fit so it can be explored further."""
        self.nucleus_combo.setCurrentText(nucleus)
        gamma = abs(find_isotope_info(nucleus)["gamma"]) / 1e6        # MHz/T
        self.field_spin.setValue(larmor_Hz / 1e6 / gamma)
        index = self.lineshape_combo.findData(lineshape)
        if index >= 0:
            self.lineshape_combo.setCurrentIndex(index)
        self.eta_spin.setValue(eta)
        self.model_combo.setCurrentIndex(self.model_combo.findData("spin_system"))
        self.spin_fwhm_spin.setValue(round(fwhm_Hz, 3))
        self.spin_editor.set_system(spins, couplings)

    def set_from_fit(self, nucleus: str, larmor_Hz: float, sites: list[dict],
                     lineshape: str, eta: float) -> None:
        """Take a fit's sites (their fitted amplitudes become the
        intensities) so the multiplet can be explored further."""
        self.nucleus_combo.setCurrentText(nucleus)
        gamma = abs(find_isotope_info(nucleus)["gamma"]) / 1e6        # MHz/T
        self.field_spin.setValue(larmor_Hz / 1e6 / gamma)
        index = self.lineshape_combo.findData(lineshape)
        if index >= 0:
            self.lineshape_combo.setCurrentIndex(index)
        self.eta_spin.setValue(eta)
        self.model_combo.setCurrentIndex(self.model_combo.findData("first_order"))
        self.editor.set_sites(sites, larmor_Hz)


# ============================================================================
# Fit > J-coupling
# ============================================================================
class JCouplingFitPanel(QWidget):
    """Fit first-order multiplets to the selected spectrum.

    Callbacks: ``get_target()`` -> (spectrum, row, index) or raises
    ValueError with a message for the user; ``get_regions()`` -> the
    Integration regions [(lo, hi), ...]; ``draw_fit(result)`` shows the
    data and fit; ``format_value_error(value, error, fixed)`` formats a
    result; ``send_to_simulation(nucleus, larmor_Hz, sites, lineshape,
    eta)`` opens the result in Simulate > J-coupling.
    """

    def __init__(self, get_target, get_regions, draw_fit, format_value_error,
                 send_to_simulation, send_spin_system=None, on_fitted=None, parent=None) -> None:
        super().__init__(parent)
        # Called after every successful fit, and when another solution is
        # chosen (the processing log, saved with the spectrum).
        self._on_fitted = on_fitted or (lambda: None)
        self._send_spin_system = send_spin_system or (lambda *a: None)
        self._get_target = get_target
        self._get_regions = get_regions
        self._draw_fit = draw_fit
        self._fmt = format_value_error
        self._send_to_simulation = send_to_simulation
        self.result: dict | None = None

        v = QVBoxLayout(self)
        blurb = QLabel("Fit J-coupled multiplets to the selected 1D spectrum (or the displayed "
                       "slice of a 2D). First order: for each site give the partners it couples "
                       "to and starting J values (e.g., as read off the line spacings). Spin system: give every "
                       "spin's shift and the J values between them. The matrix will then be solved exactly, for strongly "
                       "coupled spectra. The observe frequency comes from the spectrum.")
        blurb.setWordWrap(True)
        v.addWidget(blurb)
        self.blurb_label = blurb

        self.model_combo = _model_combo()
        v.addWidget(self.model_combo)
        self.editor = JCouplingSitesEditor("fit")
        self.spin_editor = SpinSystemEditor("fit", get_B0=self._current_B0)
        spin_page = QWidget()
        spin_layout = QVBoxLayout(spin_page)
        spin_layout.setContentsMargins(0, 0, 0, 0)
        spin_layout.addWidget(self.spin_editor)
        spin_form = QFormLayout()
        self.spin_fwhm_spin = QDoubleSpinBox()
        self.spin_fwhm_spin.setRange(0.001, 1e5)
        self.spin_fwhm_spin.setDecimals(3)
        self.spin_fwhm_spin.setValue(1.0)
        self.spin_fwhm_spin.setSuffix(" Hz")
        self.spin_fwhm_spin.setToolTip("Starting linewidth (FWHM), shared by every line; fitted.")
        spin_form.addRow("Linewidth start:", self.spin_fwhm_spin)
        self.ambiguity_check = QCheckBox("Look for other solutions that fit equally well")
        self.ambiguity_check.setChecked(True)
        self.ambiguity_check.setToolTip(
            "Strongly coupled spectra can be ambiguous: in an ABX system seen on A and B, two "
            "different sets of shifts and J(AX), J(BX) give identical spectra, and a fit "
            "returns one of them. Refits from the exchanged alternative for every such pair "
            "and lists all solutions that fit as well.")
        spin_form.addRow("", self.ambiguity_check)
        spin_layout.addLayout(spin_form)
        self.model_stack = QStackedWidget()
        self.model_stack.addWidget(self.editor)
        self.model_stack.addWidget(spin_page)
        self.model_combo.currentIndexChanged.connect(self.model_stack.setCurrentIndex)
        v.addWidget(self.model_stack)

        form = QFormLayout()
        self.lineshape_combo, self.eta_spin = _lineshape_controls(form, self)
        self.shift_window_spin = QDoubleSpinBox()
        self.shift_window_spin.setRange(0.0, 100.0)
        self.shift_window_spin.setDecimals(3)
        self.shift_window_spin.setValue(0.05)
        self.shift_window_spin.setSuffix(" ppm")
        self.shift_window_spin.setSpecialValueText("unlimited")
        self.shift_window_spin.setToolTip(
            "Keep each site's shift within this distance of its starting value (unless the "
            "site has its own shift bounds). A multiplet that starts from a poor J can "
            "otherwise wander off to fit a different peak.")
        form.addRow("Shift may move ±:", self.shift_window_spin)
        self.window_combo = QComboBox()
        self.window_combo.addItem("Around the sites", "auto")
        self.window_combo.addItem("Range (ppm)", "range")
        self.window_combo.addItem("Integration regions", "regions")
        self.window_combo.addItem("Whole spectrum", "all")
        self.window_combo.addItem("Several ppm ranges", "ranges")
        self.window_combo.setToolTip(
            "Which part of the spectrum the fit sees. Around the sites: each site's starting "
            "multiplet plus a margin – the usual choice, as other peaks in the window would "
            "otherwise be fitted as part of it.")
        form.addRow("Fit window:", self.window_combo)
        window_box = QWidget()
        window_row = QHBoxLayout(window_box)
        window_row.setContentsMargins(0, 0, 0, 0)
        self.window_lo_spin = QDoubleSpinBox()
        self.window_hi_spin = QDoubleSpinBox()
        for spin, value in ((self.window_lo_spin, 0.0), (self.window_hi_spin, 10.0)):
            spin.setRange(-1e5, 1e5)
            spin.setDecimals(4)
            spin.setSuffix(" ppm")
            spin.setValue(value)
        window_row.addWidget(self.window_lo_spin)
        window_row.addWidget(QLabel("to"))
        window_row.addWidget(self.window_hi_spin)
        form.addRow("Range:", window_box)
        self.window_ranges_edit = PpmRangesEdit()
        form.addRow("Ranges (ppm):", self.window_ranges_edit)
        self._window_box, self._window_form = window_box, form
        self.window_combo.currentIndexChanged.connect(lambda _i: self._sync_window())
        v.addLayout(form)

        self.fit_button = QPushButton("Fit J-coupled multiplets")
        self.fit_button.clicked.connect(self.fit)
        v.addWidget(self.fit_button)

        self.results_label = ReportLabel("")
        self.results_label.setWordWrap(True)
        self.results_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        v.addWidget(self.results_label)
        solution_row = QHBoxLayout()
        self.solution_label = QLabel("Solution:")
        self.solution_combo = QComboBox()
        self.solution_combo.setToolTip("Every solution listed fits the data as well as the "
                                       "others; the table, plot and buttons use the one chosen.")
        self.solution_combo.currentIndexChanged.connect(self._on_solution_changed)
        solution_row.addWidget(self.solution_label)
        solution_row.addWidget(self.solution_combo, 1)
        v.addLayout(solution_row)
        self._solution_widgets = (self.solution_label, self.solution_combo)
        for w in self._solution_widgets:
            w.setVisible(False)
        self.results_table = QTableWidget(0, 0)
        self.results_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.results_table.setWordWrap(False)
        self.results_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents)
        self.results_table.horizontalHeader().setStretchLastSection(True)
        self.results_table.setMaximumHeight(240)
        self.results_table.setToolTip(
            "Fitted values with 1-sigma uncertainties (fixed values marked). Height is the "
            "multiplet's height at its tallest line; Area (%) is each site's share of the "
            "fitted intensity in the window.\n\nRight-click to copy the table or the full "
            "report.")
        self.results_table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.results_table.customContextMenuRequested.connect(self._on_results_menu)
        self.results_table.setVisible(False)
        v.addWidget(self.results_table)
        row = QHBoxLayout()
        self.use_fitted_button = QPushButton("Use fitted values as start")
        self.use_fitted_button.setToolTip("Copy the fitted shifts, Js and widths into the table "
                                          "above, to refine from there.")
        self.use_fitted_button.clicked.connect(self.use_fitted_as_start)
        self.to_simulation_button = QPushButton("Open in Simulate")
        self.to_simulation_button.setToolTip("Send the fitted sites to Simulate > J-coupling.")
        self.to_simulation_button.clicked.connect(self.send_to_simulation)
        row.addWidget(self.use_fitted_button)
        row.addWidget(self.to_simulation_button)
        row.addStretch(1)
        v.addLayout(row)
        v.addStretch(1)

        self.editor.add_site(label="Site 1", shift=1.25, fwhm_hz=1.0,
                             couplings=[{"nuclide": "1H", "n": 2, "J_Hz": 7.0}])
        self.spin_editor.set_system(
            [{"label": "A", "nuclide": "1H", "shift_ppm": 7.30},
             {"label": "B", "nuclide": "1H", "shift_ppm": 7.26}],
            [{"a": "A", "b": "B", "J_Hz": 8.0}])
        self._sync_window()
        self._sync_result_buttons()

    def _sync_window(self) -> None:
        self._window_form.setRowVisible(self._window_box,
                                        self.window_combo.currentData() == "range")
        self._window_form.setRowVisible(self.window_ranges_edit,
                                        self.window_combo.currentData() == "ranges")

    # ---- saved state (processing log / Load Progress) ------------------

    def get_state(self) -> dict:
        return panel_state.default_state(self)

    def set_state(self, state: dict) -> None:
        panel_state.restore_default_state(self, state)
        self._sync_window()

    def result_state(self):
        if not self.result:
            return None
        return serialize.encode({k: v for k, v in self.result.items() if k != "spectrum"})

    def restore_result(self, encoded, spectrum=None) -> None:
        """Show a saved fit's table and report as they were, without
        re-fitting or redrawing."""
        r = serialize.decode(encoded)
        r["spectrum"] = spectrum
        self.result = r
        if r.get("model") == "spin_system":
            self.solution_combo.blockSignals(True)
            self.solution_combo.clear()
            for k, sol in enumerate(r["solutions"]):
                self.solution_combo.addItem("1 (as fitted)" if k == 0 else
                                            f"{k + 1}: differs in " + ", ".join(sol["differences"]))
            self.solution_combo.setCurrentIndex(int(r.get("chosen", 0)))
            self.solution_combo.blockSignals(False)
            for w in self._solution_widgets:
                w.setVisible(len(r["solutions"]) > 1)
            self._report_spin_system()
        else:
            for w in self._solution_widgets:
                w.setVisible(False)
            self._report()
        self._sync_result_buttons()

    def clear_result(self) -> None:
        self.result = None
        self.results_label.setText("")
        self.results_table.clear()
        self.results_table.setRowCount(0)
        self.results_table.setVisible(False)
        for w in self._solution_widgets:
            w.setVisible(False)
        self._sync_result_buttons()

    def report_text(self) -> str:
        if not self.result:
            return ""
        return self.results_label.text() + "\n" + panel_state.table_text(self.results_table)

    def _sync_result_buttons(self) -> None:
        self.use_fitted_button.setEnabled(self.result is not None)
        self.to_simulation_button.setEnabled(self.result is not None)

    def spin_mode(self) -> bool:
        return self.model_combo.currentData() == "spin_system"

    def _current_B0(self):
        """Field of the selected spectrum (from its observe frequency), for
        the spin editor's strong-coupling readout; None if unknown."""
        try:
            spectrum, _row, _index = self._get_target()
            md = spectrum.metadata
            return abs(float(md.sfo1)) * 1e6 / abs(find_isotope_info(md.nucleus)["gamma"])
        except Exception:  # noqa: BLE001 -- only a readout
            return None

    @staticmethod
    def _merge_spans(spans):
        spans = sorted([list(sp) for sp in spans])
        merged = [spans[0]]
        for lo, hi in spans[1:]:
            if lo <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], hi)
            else:
                merged.append([lo, hi])
        return [tuple(m) for m in merged]

    def _window_or_none(self):
        """The fit window when it is not 'Around the sites' (None: use the
        automatic one)."""
        mode = self.window_combo.currentData()
        if mode == "all":
            return ("all", None)
        if mode == "range":
            lo, hi = self.window_lo_spin.value(), self.window_hi_spin.value()
            return ("range", [(min(lo, hi), max(lo, hi))])
        if mode == "ranges":
            return ("ranges", self.window_ranges_edit.ranges())
        if mode == "regions":
            regions = self._get_regions()
            if not regions:
                raise ValueError("No Integration regions are defined – add some on the "
                                 "Integration section, or choose another fit window.")
            return ("regions", regions)
        return None

    def fit_regions(self, sites: list[dict], larmor_Hz: float):
        mode = self.window_combo.currentData()
        if mode == "all":
            return None
        if mode == "range":
            lo, hi = self.window_lo_spin.value(), self.window_hi_spin.value()
            return [(min(lo, hi), max(lo, hi))]
        if mode == "ranges":
            return self.window_ranges_edit.ranges()
        if mode == "regions":
            regions = self._get_regions()
            if not regions:
                raise ValueError("No Integration regions are defined – add some on the "
                                 "Integration section, or choose another fit window.")
            return regions
        # auto: each site's starting multiplet +/- a margin, merged
        spans = []
        for s in sites:
            freqs, _w = nsim.combine_jcouplings(s["couplings"])
            sticks = s["center_ppm"] + freqs / larmor_Hz * 1e6
            margin = 8 * s["fwhm"] + self.shift_window_spin.value() + \
                max((abs(c["J_Hz"]) for c in s["couplings"]), default=0.0) / larmor_Hz * 1e6
            spans.append([sticks.min() - margin, sticks.max() + margin])
        spans.sort()
        merged = [spans[0]]
        for lo, hi in spans[1:]:
            if lo <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], hi)
            else:
                merged.append([lo, hi])
        return [tuple(m) for m in merged]

    def fit(self) -> bool:
        if self.spin_mode():
            return self._fit_spin_system()
        try:
            spectrum, row, index = self._get_target()
        except ValueError as exc:
            QMessageBox.information(self, "J-coupling fit", str(exc))
            return False
        from ..notebook import bridge
        try:
            results = bridge.results_from_spectrum(spectrum, row=row)
        except (ValueError, IndexError) as exc:
            QMessageBox.warning(self, "J-coupling fit", str(exc))
            return False
        # From the spectrum itself: results_from_spectrum substitutes
        # 1.0 MHz for a missing SFO1, which would silently scale every J.
        sfo1 = getattr(spectrum.metadata, "sfo1", None)
        if not sfo1:
            QMessageBox.warning(self, "J-coupling fit",
                                "This spectrum has no observe frequency (SFO1), which is needed "
                                "to convert J from Hz to ppm.")
            return False
        larmor = abs(float(sfo1)) * 1e6
        try:
            sites = self.editor.sites(larmor)
            if any(not s["couplings"] for s in sites):
                raise ValueError("Every site needs at least one coupling – fit a singlet with "
                                 "Solution NMR instead.")
            window = self.shift_window_spin.value()
            for s in sites:
                if window > 0 and "center_ppm" not in s["bounds"]:
                    s["bounds"]["center_ppm"] = (s["center_ppm"] - window, s["center_ppm"] + window)
            regions = self.fit_regions(sites, larmor)
        except ValueError as exc:
            QMessageBox.warning(self, "J-coupling fit", str(exc))
            return False

        lineshape, eta = self.lineshape_combo.currentData(), self.eta_spin.value()

        # Inputs read above; the fit touches no widget, so it runs in the
        # background (gui/jobs.py; explicit user request, 2026-11-13).
        def compute():
            with thread_io.quiet_stdout():
                popt, pcov_free, is_free, _scale, info = jfit.fit_jcoupling_spectrum(
                    results, sites, larmor_Hz=larmor, lineshape=lineshape, eta=eta,
                    fit_regions=regions)
            x_sorted = np.sort(np.asarray(results["x_ppm"], dtype=float))
            y_fit, components = jfit.evaluate_jcoupling_fit(x_sorted, popt, info,
                                                             return_components=True)
            return popt, pcov_free, is_free, info, x_sorted, y_fit, components

        def apply(outcome, _stopped_early):
            popt, pcov_free, is_free, info, x_sorted, y_fit, components = outcome
            pcov = notebook_fitting.expand_pcov(pcov_free, is_free, len(popt))
            order = np.argsort(results["x_ppm"])
            y_sorted = np.asarray(results["real_spectrum"], dtype=float)[order]
            mask = (notebook_fitting.build_region_mask(x_sorted, regions) if regions
                    else np.ones(x_sorted.size, dtype=bool))
            self.result = {"popt": popt, "pcov": pcov, "is_free": is_free, "info": info,
                           "sites": sites, "regions": regions, "larmor_Hz": larmor,
                           "x": x_sorted, "y": y_sorted, "y_fit": y_fit,
                           "components": components, "mask": mask, "spectrum": spectrum,
                           "spectrum_index": job.current_index(),
                           "nucleus": results.get("nuclide") or spectrum.metadata.nucleus or "",
                           "lineshape": lineshape, "eta": eta}
            self._report()
            self._draw_fit(self.result)
            self._sync_result_buttons()
            self._on_fitted()

        job = jobs.Job("J-coupling fit", compute, apply, spectrum_index=index,
                       fail=lambda exc: QMessageBox.warning(
                           self, "J-coupling fit", f"The fit did not succeed:\n\n{exc}"),
                       inputs=self.get_state(), current_inputs=self.get_state,
                       restore_inputs=self.set_state)
        return jobs.submit(self, job)

    # ---- results ----------------------------------------------------------
    # ---- spin system -------------------------------------------------------
    def _results_for_fit(self, title):
        """(results, spectrum, index, larmor_Hz) or None after telling the user why."""
        try:
            spectrum, row, index = self._get_target()
        except ValueError as exc:
            QMessageBox.information(self, title, str(exc))
            return None
        from ..notebook import bridge
        try:
            results = bridge.results_from_spectrum(spectrum, row=row)
        except (ValueError, IndexError) as exc:
            QMessageBox.warning(self, title, str(exc))
            return None
        sfo1 = getattr(spectrum.metadata, "sfo1", None)
        if not sfo1:
            QMessageBox.warning(self, title,
                                "This spectrum has no observe frequency (SFO1), which is needed "
                                "to convert J from Hz to ppm.")
            return None
        return results, spectrum, index, abs(float(sfo1)) * 1e6

    def _fit_spin_system(self) -> bool:
        title = "Spin-system fit"
        got = self._results_for_fit(title)
        if got is None:
            return False
        results, spectrum, index, larmor = got
        observe = results.get("nuclide") or spectrum.metadata.nucleus
        try:
            gamma = abs(find_isotope_info(observe)["gamma"])
        except (KeyError, TypeError):
            QMessageBox.warning(self, title, f"The spectrum's nucleus ({observe!r}) is not in "
                                             f"the table, so the field is unknown.")
            return False
        B0 = larmor / gamma
        try:
            spins, couplings = self.spin_editor.spins(), self.spin_editor.couplings()
            if not any(s["nuclide"] == observe for s in spins):
                raise ValueError(f"The spin system has no {observe} spin – the spectrum is "
                                 f"{observe}.")
            fixed, bounds = self.spin_editor.constraints()
            window = self.shift_window_spin.value()
            if window > 0:
                for s in spins:
                    name = f"shift {s['label']} (ppm)"
                    if s["nuclide"] == observe and name not in bounds and not fixed.get(name):
                        bounds[name] = (s["shift_ppm"] - window, s["shift_ppm"] + window)
            chosen = self._window_or_none()
            if chosen is None:
                t = nss.spin_system_transitions(spins, couplings, B0, observe=observe)
                keep = t["intensity"] > 1e-3 * t["intensity"].max()
                margin = (8 * self.spin_fwhm_spin.value() / larmor * 1e6) + window
                regions = self._merge_spans([(p - margin, p + margin)
                                             for p in t["stick_ppm"][keep]])
            else:
                regions = chosen[1]
        except (ValueError, KeyError) as exc:
            QMessageBox.warning(self, title, str(exc).strip("'\""))
            return False

        lineshape, eta = self.lineshape_combo.currentData(), self.eta_spin.value()
        fwhm_hz = self.spin_fwhm_spin.value()
        ambiguity = self.ambiguity_check.isChecked()

        # Inputs read above; the fit touches no widget, so it runs in the
        # background (gui/jobs.py; explicit user request, 2026-11-13).
        def compute():
            popt, pcov, is_free, info = nss.fit_spin_system_spectrum(
                results, spins, couplings, B0_tesla=B0, fwhm_Hz=fwhm_hz,
                lineshape=lineshape, eta=eta, fit_regions=regions, fixed=fixed, bounds=bounds,
                observe=observe)
            if ambiguity:
                solutions = nss.equivalent_solutions(results, popt, info, fixed=fixed,
                                                     bounds=bounds, pcov=pcov, is_free=is_free)
            else:
                solutions = [{"popt": popt, "pcov": pcov, "is_free": is_free, "rmse": None,
                              "differences": []}]
            return solutions, info

        job = jobs.Job(title, compute,
                       lambda outcome, _early: self._show_spin_system_fit(
                           *outcome, regions=regions, larmor=larmor, B0=B0, results=results,
                           spectrum=spectrum, index=job.current_index(), observe=observe,
                           lineshape=lineshape, eta=eta),
                       spectrum_index=index,
                       fail=lambda exc: QMessageBox.warning(
                           self, title, f"The fit did not succeed:\n\n{exc}"),
                       inputs=self.get_state(), current_inputs=self.get_state,
                       restore_inputs=self.set_state)
        return jobs.submit(self, job)

    def _show_spin_system_fit(self, solutions, info, *, regions, larmor, B0, results, spectrum,
                              index, observe, lineshape, eta) -> None:
        """Show a spin-system fit just finished (GUI thread)."""
        x_sorted = np.sort(np.asarray(results["x_ppm"], dtype=float))
        order = np.argsort(results["x_ppm"])
        y_sorted = np.asarray(results["real_spectrum"], dtype=float)[order]
        mask = (notebook_fitting.build_region_mask(x_sorted, regions) if regions
                else np.ones(x_sorted.size, dtype=bool))
        self.result = {"model": "spin_system", "solutions": solutions, "chosen": 0,
                       "info": info, "regions": regions, "larmor_Hz": larmor, "B0_tesla": B0,
                       "x": x_sorted, "y": y_sorted, "mask": mask, "spectrum": spectrum,
                       "spectrum_index": index, "nucleus": observe, "lineshape": lineshape,
                       "eta": eta}
        self.solution_combo.blockSignals(True)
        self.solution_combo.clear()
        for k, sol in enumerate(solutions):
            text = "1 (as fitted)" if k == 0 else (
                f"{k + 1}: differs in " + ", ".join(sol["differences"]))
            self.solution_combo.addItem(text)
        self.solution_combo.blockSignals(False)
        for w in self._solution_widgets:
            w.setVisible(len(solutions) > 1)
        self._select_solution(0)
        self._sync_result_buttons()
        self._on_fitted()

    def _select_solution(self, k: int) -> None:
        r = self.result
        sol = r["solutions"][k]
        r["chosen"] = k
        r["popt"], r["is_free"] = sol["popt"], sol["is_free"]
        n = len(sol["popt"])
        r["pcov"] = sol["pcov"] if sol["pcov"] is not None else np.zeros((n, n))
        r["y_fit"] = nss.evaluate_spin_system_fit(r["x"], r["popt"], r["info"])
        baseline = r["popt"][-2] + r["popt"][-1] * (r["x"] - r["info"]["x_ref"])
        r["components"] = [baseline]
        self._report_spin_system()
        self._draw_fit(r)

    def _on_solution_changed(self, k: int) -> None:
        if self.result and self.result.get("model") == "spin_system" and k >= 0:
            self._select_solution(k)
            self._on_fitted()

    def _report_spin_system(self) -> None:
        r = self.result
        info, popt, is_free = r["info"], r["popt"], r["is_free"]
        errors = np.sqrt(np.clip(np.diag(r["pcov"]), 0.0, None))
        names = [n for n in info["names"] if n != "baseline: b1"]
        t = self.results_table
        t.clear()
        t.setColumnCount(1)
        t.setHorizontalHeaderLabels(["Value"])
        t.setRowCount(len(names))
        t.setVerticalHeaderLabels([n.replace("baseline: b0", "baseline offset") for n in names])
        for row, name in enumerate(names):
            k = info["names"].index(name)
            item = QTableWidgetItem(self._fmt(float(popt[k]), float(errors[k]), not is_free[k]))
            item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            t.setItem(row, 0, item)
        t.resizeRowsToContents()
        t.setVisible(True)

        mask = r["mask"]
        y, y_fit = r["y"][mask], r["y_fit"][mask]
        quality = notebook_fitting.fit_quality(y, y_fit, int(is_free.sum()))
        total = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - float(np.sum((y - y_fit) ** 2)) / total if total else float("nan")
        window = ("whole spectrum" if not r["regions"] else
                  ", ".join(f"{hi:.4f} to {lo:.4f} ppm" for lo, hi in r["regions"]))
        shape = dict((k, l) for l, k in _LINESHAPES)[r["lineshape"]]
        spins, couplings, _fw, _amp = nss.fitted_system(popt, info)
        lines = [f"Spin-system fit (exact) — {shape} lines, observe {r['larmor_Hz'] / 1e6:.4f} "
                 f"MHz ({r['B0_tesla']:.4f} T); window {window}.",
                 f"R² {r2:.5f}  ·  RMSE {quality['rmse']:.3g}  ·  reduced χ² "
                 f"{quality['reduced_chi2']:.3g}"]
        pairs = nss.coupling_strengths(spins, couplings, r["B0_tesla"])
        if pairs:
            lines.append("|Δν/J|: " + ", ".join(f"{c['a']}–{c['b']} {c['ratio']:.2g}"
                                                 for c in pairs))
        warnings = list(info.get("warnings", []))
        if len(r["solutions"]) > 1:
            warnings.append(
                f"{len(r['solutions'])} solutions fit equally well (choose one below). They "
                f"cannot be told apart from this spectrum – the classic ABX ambiguity; the "
                f"spectrum of the other nucleus, or a heteronuclear correlation, can decide.")
        if any(info["names"][k].startswith("J ") and "-" in info["names"][k]
               for k in range(len(popt))) and len(r["solutions"]) == 1 \
                and any(s["nuclide"] != r["nucleus"] for s in spins):
            warnings.append("The sign of every J to a spin of another nucleus can be flipped "
                            "together without changing this spectrum; only relative signs "
                            "are measured.")
        if np.isfinite(r2) and r2 < 0.98:
            warnings.append("The fit is poor (R² < 0.98): the starting values may be too far "
                            "off – start the shifts and Js closer to what the spectrum shows.")
        lines += [f"⚠ {w}" for w in warnings]
        self.results_label.setText("\n".join(lines))

    def fitted_sites(self) -> list[dict]:
        """The fitted sites in the notebook format, with each fitted
        height as ``relative_intensity``."""
        r = self.result
        popt, info = r["popt"], r["info"]
        out = []
        for (start, _end), topo, site in zip(info["site_slices"], info["topologies"], r["sites"]):
            k = len(topo)
            couplings = []
            for i, c in enumerate(site["couplings"]):
                c = {key: val for key, val in c.items() if key not in ("fixed_J", "bounds_J")}
                c["J_Hz"] = float(popt[start + 1 + i])
                couplings.append(c)
            out.append({"label": site["label"], "center_ppm": float(popt[start]),
                        "fwhm": float(popt[start + 1 + k]),
                        "relative_intensity": float(popt[start + 2 + k]),
                        "couplings": couplings})
        return out

    def use_fitted_as_start(self) -> None:
        if self.result is None:
            return
        if self.result.get("model") == "spin_system":
            spins, couplings, fwhm, _amp = nss.fitted_system(self.result["popt"],
                                                             self.result["info"])
            self.spin_editor.set_system(spins, couplings)
            self.spin_fwhm_spin.setValue(round(fwhm, 3))
            self.model_combo.setCurrentIndex(self.model_combo.findData("spin_system"))
            return
        sites = self.fitted_sites()
        for s in sites:
            s.pop("relative_intensity", None)
        self.editor.set_sites(sites, self.result["larmor_Hz"])

    def set_start_from_simulation(self, a: dict) -> str:
        """A simulation's sites (or spin system) as the starting model,
        with its lineshape. Returns notes on anything to check."""
        index = self.lineshape_combo.findData(a.get("lineshape"))
        if index >= 0:
            self.lineshape_combo.setCurrentIndex(index)
        self.eta_spin.setValue(float(a.get("eta", self.eta_spin.value())))
        if a.get("model") == "spin_system":
            self.spin_editor.set_system(a["spins"], a["couplings"])
            self.spin_fwhm_spin.setValue(float(a["fwhm_Hz"]))
            self.model_combo.setCurrentIndex(self.model_combo.findData("spin_system"))
        else:
            sites = [dict(s) for s in a["sites"]]
            for s in sites:
                s.pop("relative_intensity", None)
            self.editor.set_sites(sites, float(a["larmor_Hz"]))
            self.model_combo.setCurrentIndex(self.model_combo.findData("first_order"))
        return ""

    def send_to_simulation(self) -> None:
        if self.result is None:
            return
        r = self.result
        if r.get("model") == "spin_system":
            spins, couplings, fwhm, _amp = nss.fitted_system(r["popt"], r["info"])
            self._send_spin_system(r["nucleus"], r["larmor_Hz"], spins, couplings, fwhm,
                                   r["lineshape"], r["eta"])
            return
        sites = self.fitted_sites()
        # A fitted height is the multiplet's height at its tallest line;
        # the simulator's intensity scales each site's *area*. Divide by
        # the height a unit-area multiplet of that width reaches, so the
        # simulated sites stand in the fitted ratio (exact for Lorentzian
        # and Gaussian lines).
        for s in sites:
            freqs, w = nsim.combine_jcouplings(s["couplings"])
            sticks = s["center_ppm"] + freqs / r["larmor_Hz"] * 1e6
            unit_height = nsim.multiplet_lineshape(
                sticks, sticks, w, s["fwhm"], r["lineshape"], r["eta"], normalise="area").max()
            s["relative_intensity"] /= unit_height if unit_height else 1.0
        biggest = max(s["relative_intensity"] for s in sites) or 1.0
        for s in sites:
            s["relative_intensity"] = round(s["relative_intensity"] / biggest, 6)
        self._send_to_simulation(r["nucleus"], r["larmor_Hz"], sites, r["lineshape"], r["eta"])

    def _report(self) -> None:
        r = self.result
        popt, pcov, is_free, info = r["popt"], r["pcov"], r["is_free"], r["info"]
        errors = np.sqrt(np.clip(np.diag(pcov), 0.0, None))
        larmor = r["larmor_Hz"]
        hz = larmor / 1e6
        x, mask = r["x"], r["mask"]
        integrate = getattr(np, "trapezoid", None) or np.trapz
        areas = np.array([abs(integrate(c[mask], x[mask])) for c in r["components"][:-1]])
        shares = 100 * areas / areas.sum() if areas.sum() else np.full(len(areas), np.nan)

        headers = ["Shift\n(ppm)", "J (Hz)", "FWHM\n(Hz)", "Height", "Area\n(%)", "Notation"]
        t = self.results_table
        t.clear()
        t.setColumnCount(len(headers))
        t.setHorizontalHeaderLabels(headers)
        t.setRowCount(len(info["labels"]))
        t.setVerticalHeaderLabels(info["labels"])
        warnings = list(info.get("warnings", []))
        for row, ((start, _end), topo, label) in enumerate(
                zip(info["site_slices"], info["topologies"], info["labels"])):
            k = len(topo)
            js = []
            for i, g in enumerate(topo):
                j = start + 1 + i
                name = g.get("bond_label") or f"{g['n']}×{g['nuclide']}"
                js.append(f"{name}: {self._fmt(float(popt[j]), float(errors[j]), not is_free[j])}")
                if is_free[j] and popt[j] < 1e-3:
                    warnings.append(f"{label}, {name}: J went to 0 Hz – this coupling is not "
                                    f"resolved, or the starting J was far off.")
            f = start + 1 + k
            cells = [self._fmt(float(popt[start]), float(errors[start]), not is_free[start]),
                     "\n".join(js),
                     self._fmt(float(popt[f]) * hz, float(errors[f]) * hz, not is_free[f]),
                     self._fmt(float(popt[f + 1]), float(errors[f + 1]), not is_free[f + 1]),
                     "—" if not np.isfinite(shares[row]) else f"{shares[row]:.1f}"]
            mult, jstr = jfit.jcoupling_chemist_notation(r["nucleus"] or None, topo,
                                                         popt[start + 1:start + 1 + k])
            cells.append(f"{mult}, " + ", ".join(jstr))
            for col, text in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                                      if col < 5 else Qt.AlignmentFlag.AlignLeft
                                      | Qt.AlignmentFlag.AlignVCenter)
                t.setItem(row, col, item)
            lo, hi = r["sites"][row]["bounds"].get("center_ppm", (-np.inf, np.inf))
            if is_free[start] and np.isfinite(lo) and (
                    min(popt[start] - lo, hi - popt[start]) < 1e-3 * (hi - lo)):
                warnings.append(f"{label}: the shift stopped at the edge of its allowed range "
                                f"({lo:.4f} to {hi:.4f} ppm) – check the starting shift and Js.")
        t.resizeRowsToContents()
        t.setVisible(True)

        y, y_fit = r["y"][mask], r["y_fit"][mask]
        quality = notebook_fitting.fit_quality(y, y_fit, int(is_free.sum()))
        total = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - float(np.sum((y - y_fit) ** 2)) / total if total else float("nan")
        window = ("whole spectrum" if not r["regions"] else
                  ", ".join(f"{hi:.4f} to {lo:.4f} ppm" for lo, hi in r["regions"]))
        shape = dict((k, l) for l, k in _LINESHAPES)[r["lineshape"]]
        lines = [f"J-coupling fit — {shape} lines, observe {larmor / 1e6:.4f} MHz; window "
                 f"{window}.",
                 f"R² {r2:.5f}  ·  RMSE {quality['rmse']:.3g}  ·  reduced χ² "
                 f"{quality['reduced_chi2']:.3g}"]
        lines += [f"⚠ {w}" for w in warnings]
        self.results_label.setText("\n".join(lines))

    def text_report(self) -> str:
        r = self.result
        if not r:
            return ""
        if r.get("model") == "spin_system":
            rows = [f"{self.results_table.verticalHeaderItem(i).text():<22} "
                    f"{self.results_table.item(i, 0).text()}"
                    for i in range(self.results_table.rowCount())]
            return self.results_label.text() + "\n\n" + "\n".join(rows)
        buffer = StringIO()
        try:
            with thread_io.quiet_stdout() as buffer:
                jfit.print_jcoupling_fit_report(
                    r["x"][r["mask"]], r["y"][r["mask"]], r["popt"], r["pcov"], r["is_free"],
                    r["info"], chemist_notation=True)
        except Exception:  # noqa: BLE001 -- the table is still there
            pass
        return self.results_label.text() + "\n\n" + buffer.getvalue().strip()

    def table_as_tsv(self) -> str:
        t = self.results_table
        corner = "Parameter" if (self.result or {}).get("model") == "spin_system" else "Site"
        header = [corner] + [t.horizontalHeaderItem(c).text().replace("\n", " ")
                             for c in range(t.columnCount())]
        lines = ["\t".join(header)]
        for row in range(t.rowCount()):
            cells = [t.verticalHeaderItem(row).text()]
            cells += [(t.item(row, c).text() if t.item(row, c) else "").replace("\n", "; ")
                      for c in range(t.columnCount())]
            lines.append("\t".join(cells))
        return "\n".join(lines)

    def _on_results_menu(self, pos) -> None:
        menu = QMenu(self.results_table)
        copy_table = menu.addAction("Copy Table")
        copy_report = menu.addAction("Copy Full Report")
        chosen = menu.exec(self.results_table.viewport().mapToGlobal(pos))
        if chosen is copy_table:
            QGuiApplication.clipboard().setText(self.table_as_tsv())
        elif chosen is copy_report:
            QGuiApplication.clipboard().setText(self.text_report())
