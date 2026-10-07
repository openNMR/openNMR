"""
Fit > Multi-spectrum: one set of sites fitted consistently across
several spectra -- a series (VT, time), several isotopes of one
material, one isotope at several fields.

The maths is :func:`opennmr.notebook.multi_fitting.fit_multi`, called
exactly as a notebook calls it. This panel collects:

* **Experiments**, in the order to fit them (any order: 5, 302, 12, 2,
  49), each with its own nucleus, field, MAS rate, transitions, fit
  window and x value (temperature from ``TE``, or time from ``DATE``).
* **Sites per nucleus** -- the Combined Hamiltonian site editor, one site
  list per observed isotope; the same label is the same chemical site.
* **Parameters**: each (site, parameter) with its scope (per experiment,
  per nucleus, global), free/fixed and bounds; the site amounts' scope;
  the seeding.

Results: a table of every parameter across the experiments, each fit and
each parameter's trend on the Fit view, CSV export, and the full report.
"""

from __future__ import annotations

import contextlib
import copy
import io
import os
import re

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox,
    QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QHeaderView, QInputDialog, QLabel,
    QLineEdit, QListWidget, QListWidgetItem, QMenu,
    QMessageBox, QProgressDialog, QPushButton, QSpinBox, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)

from ..core.acquisition import acquisition_constants
from ..io import serialize
from ..notebook import combined_fitting as cfit
from ..notebook import combined_hamiltonian as ch
from ..notebook import multi_fitting as mf
from . import panel_state
from .widgets.advice_box import AdviceBox
from ..notebook import fit_advice
from .combined_panels import CombinedHamiltonianPanel, _param_label

#: Experiment table columns.
_E_NAME, _E_NUC, _E_NU0, _E_MAS, _E_TRANS, _E_WINDOW, _E_X = range(7)
_E_HEADERS = ("Experiment", "Nucleus", "ν₀ (MHz)", "MAS (Hz)", "Transitions", "Window (ppm)",
              "x value")
#: Parameter table columns.
_Q_SITE, _Q_NAME, _Q_NUCLEI, _Q_SCOPE, _Q_FREE, _Q_MIN, _Q_MAX = range(7)
_Q_HEADERS = ("Site", "Parameter", "Nuclei", "Scope", "Free", "Min", "Max")

_SCOPE_TEXT = {"experiment": "per experiment", "nucleus": "per nucleus", "global": "global"}
_WIDTH_TEXT = {"experiment": "per experiment", "nucleus": "per nucleus (Hz + ppm)",
               "global": "global (Hz + ppm)"}
_PRESET_TEXT = (("series", "Series (VT, time): each seeded from the previous"),
                ("isotopes", "Isotopes of one material (e.g. ¹H/²H)"),
                ("fields", "One isotope at several fields"),
                ("custom", "Custom"))


def parse_expnos(text: str) -> list[str]:
    """``"5, 302, 12-14, 2"`` -> ``["5", "302", "12", "13", "14", "2"]``,
    in the order written (a range may run downwards)."""
    out = []
    for part in re.split(r"[,\s;]+", text.strip()):
        if not part:
            continue
        m = re.fullmatch(r"(\d+)\s*-\s*(\d+)", part)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            step = 1 if b >= a else -1
            out += [str(k) for k in range(a, b + step, step)]
        elif part.isdigit():
            out.append(part)
        else:
            raise ValueError(f"'{part}' is not an experiment number or a range like 12-14.")
    return out


class _SiteEditor(CombinedHamiltonianPanel):
    """The Combined Hamiltonian site editor without its setup rows or
    Simulate buttons -- the nucleus is set by the panel."""

    BLURB = ("Sites observed as the nucleus chosen above. Use the same label for the same "
             "chemical site in every nucleus; it is how the isotopes are linked.")
    OTHER_NUCLEI = False      # each nucleus here is its own spectrum

    def _build_actions(self, v) -> None:
        pass

    def __init__(self, on_changed, parent=None) -> None:
        self._on_changed = on_changed
        super().__init__(plot_spectrum=lambda _s: None, add_to_open=lambda *_a: None, parent=parent)
        for r in range(self._setup_form.rowCount()):
            self._setup_form.setRowVisible(r, False)

    def _changed(self):
        cb = getattr(self, "_on_changed", None)
        if cb is not None:
            cb()

    def add_site(self, site=None):
        out = super().add_site(site)
        self._changed()
        return out

    def remove_site(self):
        super().remove_site()
        self._changed()

    def _on_label_edited(self, text):
        super()._on_label_edited(text)
        self._changed()

    def _on_lineshape(self, _i):
        super()._on_lineshape(_i)
        self._changed()

    def _rebuild_interactions(self):
        super()._rebuild_interactions()
        self._changed()

    def load(self, nucleus: str, dicts: list[dict]) -> None:
        self.nucleus_combo.setCurrentText(nucleus)
        self.site_list.blockSignals(True)
        self.site_list.clear()
        self.site_list.blockSignals(False)
        self.site_dicts = []
        self._current = -1
        for d in dicts:
            CombinedHamiltonianPanel.add_site(self, d)
        if not dicts:
            self._on_site_selected(-1)


class ExpnoDialog(QDialog):
    """Experiment numbers of one dataset folder, in the order to fit them."""

    def __init__(self, parent=None, folder: str = "") -> None:
        super().__init__(parent)
        self.setWindowTitle("Add experiments by number")
        layout = QVBoxLayout(self)
        form = QFormLayout()
        row = QHBoxLayout()
        self.folder_edit = QLineEdit(folder)
        self.folder_edit.setToolTip("The dataset folder holding the numbered experiments.")
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        row.addWidget(self.folder_edit, 1)
        row.addWidget(browse)
        form.addRow("Dataset folder:", row)
        self.expnos_edit = QLineEdit()
        self.expnos_edit.setPlaceholderText("e.g. 5, 302, 12, 2, 49  or  10-15")
        self.expnos_edit.setToolTip("Fitted in the order written. Ranges run either way.")
        form.addRow("Experiments:", self.expnos_edit)
        layout.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                   | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _browse(self):
        chosen = QFileDialog.getExistingDirectory(self, "Dataset folder", self.folder_edit.text())
        if chosen:
            self.folder_edit.setText(chosen)

    def values(self):
        return self.folder_edit.text().strip(), parse_expnos(self.expnos_edit.text())


class MultiFitPanel(QWidget):
    """Callbacks (all from MainWindow):

    ``open_items()`` -> [(key, label)] of Open experiments;
    ``selected_keys()`` -> keys of the selected ones, in list order;
    ``resolve(key)`` -> (spectrum, label) or None;
    ``open_expnos(folder, expnos)`` -> keys, in the order given;
    ``draw_fit(fit)``, ``draw_trend(x, y, err, labels, xlabel, ylabel, title)``;
    ``format_value_error(value, error, fixed)``; ``on_fitted()``.
    A key is what identifies an experiment across sessions (see
    MainWindow._multi_key)."""

    def __init__(self, open_items, selected_keys, resolve, open_expnos, draw_fit, draw_trend,
                 format_value_error, on_fitted=None, parent=None, draw_quantities=None,
                 draw_correlations=None) -> None:
        super().__init__(parent)
        #: ``draw_correlations(fit) -> bool``: a correlation matrix on the
        #: Fit view (notebook.fit_correlation); None offers none.
        self._draw_correlations = draw_correlations
        #: ``draw_quantities(draw)``: show ``draw(figure)`` on the Fit view
        #: (a curve plot); None draws nothing.
        self._draw_quantities = draw_quantities
        self._open_items = open_items
        self._selected_keys = selected_keys
        self._resolve = resolve
        self._open_expnos = open_expnos
        self._draw_fit = draw_fit
        self._draw_trend = draw_trend
        self._fmt = format_value_error
        self._on_fitted = on_fitted or (lambda: None)
        self.experiments: list[dict] = []
        self.sites_by_nucleus: dict[str, list[dict]] = {}
        self._editing_nucleus: str | None = None
        #: {(label, name): {"scope", "free", "min", "max"}}
        self.choices: dict[tuple, dict] = {}
        self.result: dict | None = None
        self._refreshing = False
        #: True while opening experiments or fitting: selections made on
        #: the way must not restore another saved fit into the panel.
        self._busy = False

        v = QVBoxLayout(self)
        blurb = QLabel(
            "Fit one set of sites to several spectra consistently: a series (each spectrum "
            "seeded from the previous fit), isotopes of one material, or one isotope at several "
            "fields and spinning rates. Parameters can be fitted per experiment, shared by every "
            "spectrum of a nucleus, or shared by all.")
        blurb.setWordWrap(True)
        v.addWidget(blurb)

        # ---- experiments ----
        box = QGroupBox("Experiments, in fitting order")
        bl = QVBoxLayout(box)
        self.experiment_table = QTableWidget(0, len(_E_HEADERS))
        self.experiment_table.setHorizontalHeaderLabels(list(_E_HEADERS))
        self.experiment_table.verticalHeader().setVisible(True)
        self.experiment_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents)
        self.experiment_table.horizontalHeader().setStretchLastSection(True)
        self.experiment_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.experiment_table.setMinimumHeight(130)
        self.experiment_table.setToolTip(
            "Edit MAS, Transitions (all, ct or st), Window ('lo, hi' in ppm; blank = whole "
            "spectrum) and the x value. Nucleus and ν0 come from the data.")
        self.experiment_table.itemChanged.connect(self._on_experiment_edited)
        bl.addWidget(self.experiment_table)
        row = QHBoxLayout()
        self.add_selected_button = QPushButton("Add selected")
        self.add_selected_button.setToolTip("Add the spectra selected in Open experiments, in list order (reorder with Up/Down).")
        self.add_selected_button.clicked.connect(self.add_selected)
        self.add_expnos_button = QPushButton("Add by number…")
        self.add_expnos_button.setToolTip("Open experiments of one dataset folder by number, "
                                          "in the order typed (5, 302, 12…).")
        self.add_expnos_button.clicked.connect(self._on_add_expnos)
        self.up_button = QPushButton("Up")
        self.up_button.clicked.connect(lambda: self.move(-1))
        self.down_button = QPushButton("Down")
        self.down_button.clicked.connect(lambda: self.move(+1))
        self.remove_button = QPushButton("Remove")
        self.remove_button.clicked.connect(self.remove_selected)
        for b in (self.add_selected_button, self.add_expnos_button, self.up_button,
                  self.down_button, self.remove_button):
            row.addWidget(b)
        bl.addLayout(row)
        xrow = QHBoxLayout()
        xrow.addWidget(QLabel("x value from:"))
        self.x_source_combo = QComboBox()
        self.x_source_combo.addItem("—", "none")
        self.x_source_combo.addItem("Temperature (TE, K)", "te")
        self.x_source_combo.addItem("Time since the first (min)", "time")
        self.x_source_combo.addItem("Acquisition parameter…", "param")
        self.x_source_combo.addItem("Experiment number", "expno")
        self.x_source_combo.addItem("Order (1, 2, 3…)", "order")
        self.x_source_combo.setToolTip(
            "Fill the x value column from each experiment's own parameters: temperature (TE), "
            "time from the acquisition dates, any acquisition parameter (d1, p1, cnst2, l4, "
            "…; those that differ between these experiments are listed first), the "
            "experiment number, or the fitting order. Or type any values -- a field, a "
            "concentration, a pressure. Typed values are kept until you choose again.")
        self.x_source_combo.activated.connect(lambda _i: self.fill_x_values())
        xrow.addWidget(self.x_source_combo)
        xrow.addWidget(QLabel("x label:"))
        self.x_label_edit = QLineEdit("x")
        xrow.addWidget(self.x_label_edit, 1)
        bl.addLayout(xrow)
        v.addWidget(box)

        # ---- how ----
        form = QFormLayout()
        self.preset_combo = QComboBox()
        for key, text in _PRESET_TEXT:
            self.preset_combo.addItem(text, key)
        self.preset_combo.setToolTip("Sets every scope below to the usual choice for the "
                                     "situation; edit them afterwards as needed.")
        self.preset_combo.activated.connect(lambda _i: self.apply_preset())
        form.addRow("Situation:", self.preset_combo)
        self.amounts_combo = QComboBox()
        self.amounts_combo.addItem("per experiment", "experiment")
        self.amounts_combo.addItem("site ratio shared per nucleus", "nucleus")
        self.amounts_combo.addItem("site ratio shared by all", "global")
        self.amounts_combo.setToolTip(
            "Shared: one ratio between the sites, times a free overall scale per spectrum. Not "
            "safe across isotopes of a partially or selectively labelled sample.")
        self.amounts_combo.currentIndexChanged.connect(lambda _i: self.refresh_parameter_table())
        form.addRow("Site amounts:", self.amounts_combo)
        self.seed_combo = QComboBox()
        self.seed_combo.addItem("from the previous fit", "previous")
        self.seed_combo.addItem("from the starting model", "start")
        self.seed_combo.setToolTip("Where each spectrum's single-spectrum fit starts. From the "
                                   "previous fit follows a series; a drop in R² along it is "
                                   "flagged, since a bad fit can carry forward.")
        form.addRow("Start each fit:", self.seed_combo)
        self.order_combo = QComboBox()
        self.order_combo.addItem("Second-order perturbation", "second")
        self.order_combo.addItem("Exact (diagonalization)", "exact")
        self.order_combo.addItem("First-order only", "first")
        form.addRow("Quadrupolar theory:", self.order_combo)
        self.orientations_spin = QSpinBox()
        self.orientations_spin.setRange(0, 500000)
        self.orientations_spin.setSingleStep(500)
        self.orientations_spin.setSpecialValueText("auto")
        self.orientations_spin.setValue(0)
        self.orientations_spin.setToolTip("Powder orientations for fitting (auto: 2,584 MAS, "
                                          "28,657 static).")
        form.addRow("Orientations:", self.orientations_spin)
        self.fit_baseline_check = QCheckBox("Fit each spectrum's baseline")
        form.addRow("", self.fit_baseline_check)
        # The MAS rate as a fitting parameter, fixed by default (explicit
        # user request, 2026-10-06).
        mas_row = QHBoxLayout()
        self.fit_mas_check = QCheckBox("Let each spectrum's MAS rate vary, by up to ±")
        self.fit_mas_check.setToolTip(
            "Fitted in each spectrum's own fit, starting from its MAS (Hz) column; a joint "
            "refinement then holds each at its fitted value. The recorded rate is often a "
            "little off, and the sidebands are very sensitive to it. Off by default.")
        self.mas_window_spin = QSpinBox()
        self.mas_window_spin.setRange(1, 20000)
        self.mas_window_spin.setValue(500)
        self.mas_window_spin.setSuffix(" Hz")
        mas_row.addWidget(self.fit_mas_check)
        mas_row.addWidget(self.mas_window_spin)
        mas_row.addStretch(1)
        form.addRow("", mas_row)
        v.addLayout(form)

        # ---- sites per nucleus ----
        sbox = QGroupBox("Sites")
        sl = QVBoxLayout(sbox)
        srow = QHBoxLayout()
        srow.addWidget(QLabel("Nucleus:"))
        self.nucleus_combo = QComboBox()
        self.nucleus_combo.setToolTip("Each observed isotope has its own site list.")
        self.nucleus_combo.currentTextChanged.connect(self._on_nucleus_changed)
        srow.addWidget(self.nucleus_combo, 1)
        self.copy_sites_button = QPushButton("Copy sites from ▾")
        self.copy_sites_button.setToolTip(
            "Start this nucleus's sites from another's: labels, δiso, amounts, widths and CSA "
            "(and C_Q scaled by the quadrupole moments, when both are quadrupolar).")
        self._copy_menu = QMenu(self)
        self._copy_menu.aboutToShow.connect(self._fill_copy_menu)
        self.copy_sites_button.setMenu(self._copy_menu)
        srow.addWidget(self.copy_sites_button)
        sl.addLayout(srow)
        self.site_editor = _SiteEditor(on_changed=self._on_sites_changed)
        sl.addWidget(self.site_editor)
        v.addWidget(sbox)

        # ---- parameters ----
        pbox = QGroupBox("Parameters")
        pl = QVBoxLayout(pbox)
        note = QLabel("Starting values are the sites'. Scope: fitted per experiment, shared by "
                      "every spectrum of a nucleus, or shared by all (C_Q through the quadrupole "
                      "moments, J through γ). A width shared across fields is split into Hz and "
                      "ppm parts. δ offset: a per-isotope shift on a shared δiso, held at 0 "
                      "unless freed.")
        note.setWordWrap(True)
        pl.addWidget(note)
        self.parameter_table = QTableWidget(0, len(_Q_HEADERS))
        self.parameter_table.setHorizontalHeaderLabels(list(_Q_HEADERS))
        self.parameter_table.verticalHeader().setVisible(False)
        self.parameter_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents)
        self.parameter_table.horizontalHeader().setStretchLastSection(True)
        self.parameter_table.setMinimumHeight(160)
        self.parameter_table.itemChanged.connect(self._on_parameter_edited)
        pl.addWidget(self.parameter_table)
        v.addWidget(pbox)

        self.fit_button = QPushButton("Fit All")
        self.fit_button.clicked.connect(self.fit)
        v.addWidget(self.fit_button)

        # ---- results ----
        self.results_label = QLabel("")
        self.results_label.setWordWrap(True)
        self.results_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        v.addWidget(self.results_label)
        show = QHBoxLayout()
        show.addWidget(QLabel("Show:"))
        self.show_combo = QComboBox()
        self.show_combo.setToolTip("Draw one experiment's fit, or a parameter along the "
                                   "experiments, on the Fit view.")
        self.show_combo.activated.connect(lambda _i: self.show_selected())
        show.addWidget(self.show_combo, 1)
        v.addLayout(show)
        v.addWidget(self._build_quantity_box())
        self.results_table = QTableWidget(0, 0)
        self.results_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.results_table.setMaximumHeight(280)
        self.results_table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.results_table.customContextMenuRequested.connect(self._on_results_menu)
        self.results_table.setToolTip("Every parameter across the experiments (± 1σ). "
                                      "Right-click to copy, export CSV or copy the report.")
        self.results_table.setVisible(False)
        v.addWidget(self.results_table)
        # Plain-language advice on each spectrum's fit and the series
        # (explicit user request, 2026-10-06); notebook.fit_advice.
        self.advice_box = AdviceBox(title="What the fits tell you, and what to try next")
        v.addWidget(self.advice_box)
        v.addStretch(1)

        self.apply_preset()
        self._sync()

    # ---- experiments ------------------------------------------------------

    def _experiment_row(self, key, spectrum, label) -> dict:
        meta = spectrum.metadata
        raw = meta.raw or {}
        try:
            mas = float((raw.get("acqus") or {}).get("MASR", 0) or 0)
        except (TypeError, ValueError):
            mas = 0.0
        return {"key": key, "label": label, "nucleus": meta.nucleus or "",
                "larmor_MHz": float(meta.sfo1 or 0.0), "mas_Hz": mas, "transitions": "all",
                "window": "", "x": ""}

    def add_keys(self, keys) -> list[str]:
        """Append experiments by key; returns what could not be added."""
        skipped = []
        present = {e["key"] for e in self.experiments}
        for key in keys:
            if key in present:
                continue
            got = self._resolve(key)
            if got is None:
                skipped.append(str(key))
                continue
            spectrum, label = got
            if spectrum.ndim != 1:
                skipped.append(f"{label} (2D -- keep a slice first)")
                continue
            self.experiments.append(self._experiment_row(key, spectrum, label))
            present.add(key)
        self._after_experiments_changed()
        return skipped

    def add_selected(self) -> None:
        skipped = self.add_keys(self._selected_keys())
        if skipped:
            QMessageBox.information(self, "Multi-spectrum fit", "Not added:\n" + "\n".join(skipped))

    def _on_add_expnos(self) -> None:
        folder = ""
        for e in self.experiments[-1:]:
            if isinstance(e["key"], str) and os.path.isdir(e["key"]):
                folder = os.path.dirname(e["key"].rstrip("/\\"))
        dialog = ExpnoDialog(self, folder)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            folder, expnos = dialog.values()
        except ValueError as exc:
            QMessageBox.warning(self, "Multi-spectrum fit", str(exc))
            return
        self.add_expnos(folder, expnos)

    def add_expnos(self, folder: str, expnos: list[str]) -> list[str]:
        self._busy = True
        try:
            keys, problems = self._open_expnos(folder, expnos)
        finally:
            self._busy = False
        problems = list(problems) + self.add_keys(keys)
        if problems:
            QMessageBox.information(self, "Multi-spectrum fit", "Not added:\n" + "\n".join(problems))
        return problems

    def move(self, step: int) -> None:
        rows = sorted({i.row() for i in self.experiment_table.selectedIndexes()})
        if len(rows) != 1:
            return
        r = rows[0]
        t = r + step
        if 0 <= t < len(self.experiments):
            self.experiments[r], self.experiments[t] = self.experiments[t], self.experiments[r]
            self._after_experiments_changed()
            self.experiment_table.selectRow(t)

    def remove_selected(self) -> None:
        rows = sorted({i.row() for i in self.experiment_table.selectedIndexes()}, reverse=True)
        for r in rows:
            del self.experiments[r]
        self._after_experiments_changed()

    def _after_experiments_changed(self) -> None:
        for n in self.nuclei():
            self.sites_by_nucleus.setdefault(n, [])
        self._fill_experiment_table()
        current = self.nucleus_combo.currentText()
        self.nucleus_combo.blockSignals(True)
        self.nucleus_combo.clear()
        self.nucleus_combo.addItems(self.nuclei())
        self.nucleus_combo.blockSignals(False)
        if current in self.nuclei():
            self.nucleus_combo.setCurrentText(current)
        self._on_nucleus_changed(self.nucleus_combo.currentText())
        self._sync()

    def nuclei(self) -> list[str]:
        return list(dict.fromkeys(e["nucleus"] for e in self.experiments if e["nucleus"]))

    def _fill_experiment_table(self) -> None:
        t = self.experiment_table
        t.blockSignals(True)
        t.setRowCount(len(self.experiments))
        for r, e in enumerate(self.experiments):
            cells = [e["label"], e["nucleus"], f"{e['larmor_MHz']:.4f}", f"{e['mas_Hz']:g}",
                     e["transitions"], e["window"], str(e["x"])]
            for c, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if c in (_E_NAME, _E_NUC, _E_NU0):
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                t.setItem(r, c, item)
        t.setVerticalHeaderLabels([str(r + 1) for r in range(len(self.experiments))])
        t.blockSignals(False)

    def _on_experiment_edited(self, item) -> None:
        r, c = item.row(), item.column()
        if not 0 <= r < len(self.experiments):
            return
        e, text = self.experiments[r], item.text().strip()
        if c == _E_MAS:
            try:
                e["mas_Hz"] = max(float(text or 0), 0.0)
            except ValueError:
                pass
        elif c == _E_TRANS:
            if text.lower() in ("all", "ct", "st"):
                e["transitions"] = text.lower()
        elif c == _E_WINDOW:
            e["window"] = text
        elif c == _E_X:
            e["x"] = text
        self._fill_experiment_table()

    # ---- x values ---------------------------------------------------------------

    def _constants_of(self, e) -> dict:
        got = self._resolve(e["key"])
        return acquisition_constants(got[0]) if got else {}

    def acquisition_parameter_names(self) -> list[str]:
        """Numeric acquisition parameters every experiment has: those whose
        value differs between the experiments first, then the rest."""
        tables = [self._constants_of(e) for e in self.experiments]
        if not tables or not all(tables):
            return sorted(set().union(*tables)) if tables else []
        common = set(tables[0]).intersection(*tables[1:])
        varying = sorted(n for n in common if len({t[n] for t in tables}) > 1)
        return varying + sorted(common - set(varying))

    @staticmethod
    def _expno_of(key) -> str | None:
        """The experiment number in a Bruker path (…/12 or …/12/pdata/1)."""
        found = re.findall(r"[\\/](\d+)(?=[\\/]pdata|[\\/]?$)", str(key or ""))
        return found[0] if found else None

    def fill_x_values(self, source: str | None = None, name: str | None = None) -> None:
        """Fill the x value column. ``source`` defaults to the combo's;
        for ``"param"``, ``name`` is the parameter (asked for when None)."""
        source = source or self.x_source_combo.currentData()
        if source == "none":
            return
        if source == "param":
            if name is None:
                names = self.acquisition_parameter_names()
                if not names:
                    QMessageBox.information(self, "x value from", "These experiments carry no "
                                            "acquisition parameters to choose from.")
                    return
                name, ok = QInputDialog.getItem(
                    self, "x value from", "Acquisition parameter (those that differ between "
                    "the experiments are listed first):", names, 0, True)
                if not ok or not name.strip():
                    return
            name = name.strip()
            missing = []
            for e in self.experiments:
                table = self._constants_of(e)
                value = table.get(name, table.get(name.lower()))
                if value is None:
                    missing.append(e["label"])
                else:
                    e["x"] = f"{value:g}"
            self.x_label_edit.setText(name)
            self._fill_experiment_table()
            if missing:
                QMessageBox.information(self, "x value from", f"No {name} in: "
                                        + ", ".join(missing) + ". Type their x values.")
            return
        if source in ("expno", "order"):
            for k, e in enumerate(self.experiments):
                value = str(k + 1) if source == "order" else self._expno_of(e["key"])
                if value is not None:
                    e["x"] = value
            self.x_label_edit.setText("Experiment" if source == "expno" else "Order")
            self._fill_experiment_table()
            return
        dates = {}
        for e in self.experiments:
            got = self._resolve(e["key"])
            acqus = ((got[0].metadata.raw or {}).get("acqus") or {}) if got else {}
            if source == "te" and acqus.get("TE") not in (None, ""):
                e["x"] = f"{float(acqus['TE']):g}"
                self.x_label_edit.setText("T (K)")
            elif source == "time" and acqus.get("DATE") not in (None, ""):
                dates[id(e)] = float(acqus["DATE"])
        if dates:
            first = min(dates.values())
            for e in self.experiments:
                if id(e) in dates:
                    e["x"] = f"{(dates[id(e)] - first) / 60.0:.4g}"
            self.x_label_edit.setText("t (min)")
        self._fill_experiment_table()

    # ---- sites ------------------------------------------------------------

    def _save_editor(self) -> None:
        if self._editing_nucleus is not None:
            self.sites_by_nucleus[self._editing_nucleus] = copy.deepcopy(self.site_editor.site_dicts)

    def _on_nucleus_changed(self, nucleus: str) -> None:
        self._save_editor()
        self._editing_nucleus = nucleus or None
        if nucleus:
            self.site_editor.load(nucleus, copy.deepcopy(self.sites_by_nucleus.get(nucleus, [])))
        self.site_editor.setEnabled(bool(nucleus))
        self.refresh_parameter_table()

    def _on_sites_changed(self) -> None:
        if self._refreshing or self._editing_nucleus is None:
            return
        self._save_editor()
        self.refresh_parameter_table()

    def _fill_copy_menu(self) -> None:
        self._copy_menu.clear()
        for n in self.nuclei():
            if n == self._editing_nucleus:
                continue
            action = self._copy_menu.addAction(n)
            action.triggered.connect(lambda _c=False, src=n: self.copy_sites_from(src))

    def copy_sites_from(self, source: str) -> None:
        """Start the nucleus being edited from ``source``'s sites."""
        target = self._editing_nucleus
        if not target or source not in self.sites_by_nucleus:
            return
        q_src, q_dst = mf.quadrupole_moment(source), mf.quadrupole_moment(target)
        existing = {d["label"] for d in self.sites_by_nucleus.get(target, [])}
        out = list(self.sites_by_nucleus.get(target, []))
        for d in self.sites_by_nucleus[source]:
            if d["label"] in existing:
                continue
            new = {k: copy.deepcopy(v) for k, v in d.items() if k != "interactions"}
            new["interactions"] = []
            for i in d.get("interactions", []):
                if i["type"] == "csa":
                    new["interactions"].append(copy.deepcopy(i))
                elif i["type"] == "quadrupolar" and q_src and q_dst:
                    new["interactions"].append({**i, "C_Q": i["C_Q"] * q_dst / q_src})
            out.append(new)
        self.sites_by_nucleus[target] = out
        self._refreshing = True
        try:
            self.site_editor.load(target, copy.deepcopy(out))
        finally:
            self._refreshing = False
        self.refresh_parameter_table()

    def set_sites(self, nucleus: str, sites) -> None:
        """Programmatic: the sites for ``nucleus`` (Sites or dicts)."""
        dicts = []
        for s in sites:
            d = s.to_dict() if hasattr(s, "to_dict") else copy.deepcopy(dict(s))
            dicts.append({k: v for k, v in d.items() if k not in ("nucleus", "spin")})
        self.sites_by_nucleus[nucleus] = dicts
        if nucleus == self._editing_nucleus:
            self._refreshing = True
            try:
                self.site_editor.load(nucleus, copy.deepcopy(dicts))
            finally:
                self._refreshing = False
        self.refresh_parameter_table()

    # ---- parameters -------------------------------------------------------

    def apply_preset(self) -> None:
        preset = self.preset_combo.currentData()
        if preset == "custom":
            return
        scopes, amounts = mf.default_scopes(preset)
        for (label, name), c in self.choices.items():
            c["scope"] = mf._scope_of(scopes, label, name) if name != "delta_offset" else "nucleus"
        self._preset_scopes = scopes
        self.amounts_combo.setCurrentIndex(self.amounts_combo.findData(amounts))
        self.seed_combo.setCurrentIndex(self.seed_combo.findData("previous"))
        self.refresh_parameter_table()

    def _rows(self):
        """(label, name, nuclei) for every parameter of every nucleus's
        sites, first appearance first; δ offsets where a site is in more
        than one nucleus."""
        seen = {}
        for n in self.nuclei():
            for d in self.sites_by_nucleus.get(n, []):
                for name in cfit.site_parameter_names({**d, "nucleus": n}):
                    seen.setdefault((d["label"], name), []).append(n)
        rows = [(label, name, nuclei) for (label, name), nuclei in seen.items()]
        out = []
        for label, name, nuclei in rows:
            out.append((label, name, nuclei))
            if name == "delta_iso" and len(nuclei) > 1:
                out.append((label, "delta_offset", nuclei[1:]))
        return out

    def _choice(self, label, name) -> dict:
        key = (label, name)
        if key not in self.choices:
            scopes = getattr(self, "_preset_scopes", None) or mf.default_scopes("series")[0]
            self.choices[key] = {
                "scope": "nucleus" if name == "delta_offset" else mf._scope_of(scopes, label, name),
                "free": name not in ("alpha", "beta", "gamma", "delta_offset"),
                "min": "", "max": ""}
        return self.choices[key]

    def refresh_parameter_table(self) -> None:
        if not hasattr(self, "parameter_table"):
            return
        t = self.parameter_table
        t.blockSignals(True)
        try:
            rows = self._rows()
            t.setRowCount(len(rows))
            amounts = self.amounts_combo.currentData()
            for r, (label, name, nuclei) in enumerate(rows):
                c = self._choice(label, name)
                text = _param_label(name) if name != "delta_offset" else "δ offset (ppm)"
                cells = [label, text, ", ".join(nuclei), "", "", c["min"], c["max"]]
                for col, value in enumerate(cells):
                    item = QTableWidgetItem(value)
                    if col in (_Q_SITE, _Q_NAME, _Q_NUCLEI, _Q_SCOPE, _Q_FREE):
                        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    if col == _Q_NAME:
                        item.setData(Qt.ItemDataRole.UserRole, name)
                    if col == _Q_FREE:
                        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                        item.setCheckState(Qt.CheckState.Checked if c["free"] else Qt.CheckState.Unchecked)
                    t.setItem(r, col, item)
                combo = QComboBox()
                if name == "amount":
                    combo.addItem({"experiment": "per experiment", "nucleus": "ratio per nucleus",
                                   "global": "ratio shared by all"}[amounts], amounts)
                    combo.setEnabled(False)
                    combo.setToolTip("Set by Site amounts, above, for every site.")
                elif name == "delta_offset":
                    combo.addItem("per nucleus", "nucleus")
                    combo.setEnabled(False)
                else:
                    labels = _WIDTH_TEXT if name == "fwhm" else _SCOPE_TEXT
                    for scope in mf.SCOPES:
                        combo.addItem(labels[scope], scope)
                    combo.setCurrentIndex(combo.findData(c["scope"]))
                    combo.currentIndexChanged.connect(
                        lambda _i, cb=combo, lab=label, nm=name: self._on_scope_changed(lab, nm, cb))
                t.setCellWidget(r, _Q_SCOPE, combo)
        finally:
            t.blockSignals(False)

    def _on_scope_changed(self, label, name, combo) -> None:
        self._choice(label, name)["scope"] = combo.currentData()
        self.preset_combo.setCurrentIndex(self.preset_combo.findData("custom"))

    def _on_parameter_edited(self, item) -> None:
        r, col = item.row(), item.column()
        label_item, name_item = self.parameter_table.item(r, _Q_SITE), self.parameter_table.item(r, _Q_NAME)
        if label_item is None or name_item is None:
            return
        c = self._choice(label_item.text(), name_item.data(Qt.ItemDataRole.UserRole))
        if col == _Q_FREE:
            c["free"] = item.checkState() == Qt.CheckState.Checked
        elif col == _Q_MIN:
            c["min"] = item.text().strip()
        elif col == _Q_MAX:
            c["max"] = item.text().strip()

    def _sync(self) -> None:
        self.fit_button.setEnabled(bool(self.experiments))

    # ---- fitting ----------------------------------------------------------

    def fit_arguments(self) -> dict:
        """Everything fit_multi is called with (experiments without data).
        Raises ValueError, naming the cell, for anything unreadable."""
        self._save_editor()
        if not self.experiments:
            raise ValueError("Add the experiments to fit first.")
        rows = self._rows()
        scopes, vary, bounds = {}, {}, {}
        for label, name, _n in rows:
            c = self._choice(label, name)
            if name not in ("amount", "delta_offset"):
                scopes.setdefault(label, {})[name] = c["scope"]
            if c["free"]:
                vary.setdefault(label, []).append(name)
                if name == "fwhm":
                    vary[label] += ["width_Hz", "width_ppm"]
            else:
                vary.setdefault(label, [])
            try:
                lo = float(c["min"]) if c["min"] else None
                hi = float(c["max"]) if c["max"] else None
            except ValueError:
                raise ValueError(f"{label}, {_param_label(name)}: Min and Max must be numbers "
                                 "(or blank).") from None
            if lo is not None or hi is not None:
                if lo is not None and hi is not None and lo >= hi:
                    raise ValueError(f"{label}, {_param_label(name)}: Min must be below Max.")
                bounds.setdefault(label, {})[name] = (-np.inf if lo is None else lo,
                                                      np.inf if hi is None else hi)
        sites = {n: copy.deepcopy(self.sites_by_nucleus.get(n) or []) for n in self.nuclei()}
        for n, dicts in sites.items():
            if not dicts:
                raise ValueError(f"Add at least one {n} site (Sites, nucleus {n}).")
        experiments = []
        for e in self.experiments:
            window = None
            if e["window"]:
                try:
                    lo, hi = (float(v) for v in e["window"].replace(";", ",").split(","))
                except ValueError:
                    raise ValueError(f"{e['label']}: Window must be 'low, high' in ppm "
                                     "(or blank).") from None
                window = [(min(lo, hi), max(lo, hi))]
            x = None
            if str(e["x"]).strip():
                try:
                    x = float(e["x"])
                except ValueError:
                    x = str(e["x"]).strip()
            experiments.append({"key": e["key"], "label": e["label"], "nucleus": e["nucleus"],
                                "larmor_Hz": e["larmor_MHz"] * 1e6, "mas_Hz": e["mas_Hz"],
                                "transitions": e["transitions"], "fit_regions": window,
                                "x_value": x})
        return {"experiments": experiments, "sites": sites, "scopes": scopes, "vary": vary,
                "bounds": bounds or None, "amounts": self.amounts_combo.currentData(),
                "seed": self.seed_combo.currentData(), "order": self.order_combo.currentData(),
                "n_zcw": self.orientations_spin.value() or None,
                "fit_baseline": self.fit_baseline_check.isChecked(),
                "fit_mas": self.fit_mas_check.isChecked(),
                "mas_window_Hz": float(self.mas_window_spin.value())}

    def fit(self, show_progress: bool = True) -> bool:
        from ..notebook import bridge
        try:
            args = self.fit_arguments()
            experiments = []
            for e in args["experiments"]:
                got = self._resolve(e["key"])
                if got is None:
                    raise ValueError(f"{e['label']} is not open any more -- add it again.")
                results = bridge.results_from_spectrum(got[0])
                experiments.append(mf.make_experiment(
                    results, e["label"], nucleus=e["nucleus"], larmor_Hz=e["larmor_Hz"],
                    mas_Hz=e["mas_Hz"], order=args["order"], transitions=e["transitions"],
                    n_zcw=args["n_zcw"], fit_regions=e["fit_regions"],
                    fit_baseline=args["fit_baseline"], x_value=e["x_value"],
                    fit_mas=args["fit_mas"], mas_window_Hz=args["mas_window_Hz"]))
        except (ValueError, KeyError, IndexError) as exc:
            QMessageBox.warning(self, "Multi-spectrum fit", str(exc))
            return False
        dialog = None
        if show_progress:
            dialog = QProgressDialog("Fitting…", "Cancel", 0, len(experiments) + 1, self)
            dialog.setWindowTitle("Multi-spectrum fit")
            dialog.setWindowModality(Qt.WindowModality.WindowModal)
            dialog.setMinimumDuration(300)

        def progress(done, total, message):
            if dialog is None:
                return True
            dialog.setMaximum(total)
            dialog.setValue(done)
            dialog.setLabelText(message)
            QApplication.processEvents()
            return not dialog.wasCanceled()

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        self._busy = True
        try:
            result = mf.fit_multi(experiments, args["sites"], scopes=args["scopes"],
                                  amounts=args["amounts"], vary=args["vary"],
                                  bounds=args["bounds"], seed=args["seed"], progress=progress)
        except mf.MultiFitCancelled:
            QApplication.restoreOverrideCursor()
            return False
        except Exception as exc:  # noqa: BLE001 -- a fit that fails is an outcome, not a crash
            QApplication.restoreOverrideCursor()
            if dialog is not None:
                dialog.close()
            QMessageBox.warning(self, "Multi-spectrum fit", f"The fit did not succeed:\n\n{exc}")
            return False
        finally:
            self._busy = False
            if dialog is not None:
                dialog.close()
        QApplication.restoreOverrideCursor()
        result["keys"] = [e["key"] for e in args["experiments"]]
        result["x_label"] = self.x_label_edit.text().strip() or "x"
        self.result = result
        self._report()
        if self.show_combo.count():
            self.show_combo.setCurrentIndex(0)
            self.show_selected()
        self._on_fitted()
        return True

    # ---- results ----------------------------------------------------------

    def _result_rows(self):
        """[(row label, [(value, error, free) per experiment])]."""
        r = self.result
        rows, index = [], {}
        for j, fit in enumerate(r["experiments"]):
            for p in fit["parameters"]:
                scope = p.get("scope", "experiment")
                label = (f"{p['label']}: " if p["label"] else "") + _param_label(p["name"]) + (
                    "" if scope == "experiment" else f" [{scope}]")
                if label not in index:
                    index[label] = len(rows)
                    rows.append((label, (p["label"], p["name"]), [None] * len(r["experiments"])))
                rows[index[label]][2][j] = (p["value"], p["error"], p["free"])
            for site, d in fit["derived"].items():
                for key, title in (("share_%", "share (%)"), ("P_Q", "P_Q (MHz)"),
                                   ("delta_cg", "CT c.o.g. (ppm)")):
                    if key not in d:
                        continue
                    label = f"{site}: {title}"
                    if label not in index:
                        index[label] = len(rows)
                        rows.append((label, (site, key), [None] * len(r["experiments"])))
                    rows[index[label]][2][j] = (d[key], d.get(key + "_err"), True)
        return rows

    def _report(self) -> None:
        r = self.result
        rows = self._result_rows()
        t = self.results_table
        t.clear()
        names = [f.get("label") for f in r["experiments"]]
        t.setColumnCount(len(names))
        t.setHorizontalHeaderLabels(names)
        t.setRowCount(len(rows))
        t.setVerticalHeaderLabels([row[0] for row in rows])
        for i, (_label, _key, cells) in enumerate(rows):
            for j, cell in enumerate(cells):
                text = "—" if cell is None else self._fmt(float(cell[0]), float(cell[1] or 0.0),
                                                            not cell[2])
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                t.setItem(i, j, item)
        t.resizeColumnsToContents()
        t.setVisible(True)
        lines = [f"Multi-spectrum fit — {r['mode']}, {len(r['experiments'])} spectra."]
        for fit in r["experiments"]:
            lines.append(f"  {fit['label']}: R² {fit['quality']['r2']:.4f}")
        if r.get("joint_quality"):
            lines.append(f"Joint reduced χ² {r['joint_quality']['reduced_chi2']:.3g} "
                         "(each spectrum weighted by its own residual noise).")
        shared = [row for row in r["shared"] if row["name"] in ("width_Hz", "width_ppm", "delta_offset")]
        for row in shared:
            lines.append(f"  {row['label']} {row['name']} [{row['group']}]: "
                         + self._fmt(row["value"], row["error"] or 0.0, not row["free"]))
        lines += [f"⚠ {w}" for w in r["warnings"]]
        self.results_label.setText("\n".join(lines))
        self.show_combo.blockSignals(True)
        self.show_combo.clear()
        for j, fit in enumerate(r["experiments"]):
            self.show_combo.addItem(f"Fit: {fit['label']}", ("fit", j))
        for label, key, _cells in rows:
            self.show_combo.addItem(f"Trend: {label}", ("trend", key))
        # Parameter correlations (explicit user request, 2026-10-06): the
        # joint refinement's across every spectrum, or each spectrum's own
        # fit in sequential mode.
        if self._draw_correlations is not None:
            if r.get("pcov") is not None:
                self.show_combo.addItem("Correlations: joint fit, all spectra", ("corr", None))
            for j, fit in enumerate(r["experiments"]):
                if fit.get("pcov") is not None:
                    self.show_combo.addItem(f"Correlations: {fit['label']}", ("corr", j))
        self.show_combo.blockSignals(False)
        self._fill_quantity_controls()
        try:
            self.advice_box.set_items(fit_advice.advise_multi_fit(r))
        except Exception:  # noqa: BLE001 -- advice must never sink a fit
            self.advice_box.set_items([])

    def show_selected(self) -> None:
        if not self.result:
            return
        what = self.show_combo.currentData()
        if what is None:
            return
        if what[0] == "corr":
            from ..notebook import fit_correlation
            try:
                data = fit_correlation.multi_fit_correlation(self.result, what[1])
            except (ValueError, KeyError, StopIteration) as exc:
                QMessageBox.information(self, "Multi-spectrum fit", str(exc))
                return
            self._draw_correlations(data)
            return
        if what[0] == "fit":
            fit = dict(self.result["experiments"][what[1]])
            if fit.get("x") is None:
                QMessageBox.information(self, "Multi-spectrum fit", "This fit's curves are saved "
                                        "with its own experiment: select it to draw them.")
                return
            key = self.result.get("keys", [None] * (what[1] + 1))[what[1]]
            got = self._resolve(key) if key is not None else None
            fit["spectrum"] = got[0] if got else None
            self._draw_fit(fit)
            return
        label, name = what[1]
        xs, vs, es, labs = mf.parameter_trend(self.result, label, name)
        rows = {row[1]: row for row in self._result_rows()}
        title = rows[(label, name)][0] if (label, name) in rows else f"{label}: {name}"
        self._draw_trend(xs, vs, es, labs, self.result.get("x_label", "x"), title, title)

    # ---- fitted quantities against x ---------------------------------------

    def _build_quantity_box(self) -> QGroupBox:
        box = QGroupBox("Plot a fitted quantity against x")
        box.setToolTip("Any fitted or derived quantity of any peak, along the experiments, "
                       "against the x value column -- a temperature, a time, an acquisition "
                       "parameter or values you type -- on the Fit view.")
        form = QFormLayout(box)
        self.quantity_combo = QComboBox()
        self.quantity_combo.setToolTip(
            "Shift and other fitted parameters with their 1σ; linewidth also in Hz; "
            "integral and height of each peak's fitted curve; centre of mass (Σδ·I/ΣI); the "
            "model amount and share; interaction parameters (CSA, C_Q, P_Q, J…). "
            f"'{mf.WINDOW}' measures the data in the fit window instead, model-free.")
        form.addRow("Quantity:", self.quantity_combo)
        self.quantity_peaks_list = QListWidget()
        self.quantity_peaks_list.setMaximumHeight(96)
        self.quantity_peaks_list.setToolTip("Tick the peaks to plot; each is its own series.")
        form.addRow("Peaks:", self.quantity_peaks_list)
        self.quantity_x_combo = QComboBox()
        for key, (fmt, _f) in mf.X_TRANSFORMS.items():
            self.quantity_x_combo.addItem(fmt.format("x"), key)
        self.quantity_x_combo.setToolTip("1000/x with temperatures in K gives an Arrhenius "
                                         "axis; the x label above names the axis.")
        form.addRow("x axis:", self.quantity_x_combo)
        self.quantity_y_combo = QComboBox()
        for text in mf.Y_TRANSFORMS:
            self.quantity_y_combo.addItem(text, text)
        form.addRow("y axis:", self.quantity_y_combo)
        row = QHBoxLayout()
        self.quantity_plot_button = QPushButton("Plot")
        self.quantity_plot_button.clicked.connect(lambda: self.plot_quantity())
        self.quantity_csv_button = QPushButton("Copy Data (CSV)")
        self.quantity_csv_button.clicked.connect(self._copy_quantity_csv)
        self.quantity_python_button = QPushButton("Copy as Python")
        self.quantity_python_button.setToolTip("Notebook code for this plot, from the saved "
                                               "result.")
        self.quantity_python_button.clicked.connect(
            lambda: QGuiApplication.clipboard().setText(self.quantity_python()))
        for b in (self.quantity_plot_button, self.quantity_csv_button,
                  self.quantity_python_button):
            row.addWidget(b)
        form.addRow(row)
        self.quantity_note_label = QLabel("")
        self.quantity_note_label.setWordWrap(True)
        form.addRow(self.quantity_note_label)
        box.setVisible(False)
        self.quantity_box = box
        return box

    def _fill_quantity_controls(self) -> None:
        """Offer what the result holds; keep the current choices where
        they still exist."""
        if not self.result:
            self.quantity_box.setVisible(False)
            return
        labels, keys = mf.available_quantities(self.result)
        current = self.quantity_combo.currentData()
        self.quantity_combo.blockSignals(True)
        self.quantity_combo.clear()
        for key in keys:
            self.quantity_combo.addItem(mf.quantity_title(key), key)
        index = self.quantity_combo.findData(current)
        self.quantity_combo.setCurrentIndex(index if index >= 0 else 0)
        self.quantity_combo.blockSignals(False)
        ticked = set(self.quantity_labels()) if self.quantity_peaks_list.count() else None
        self.quantity_peaks_list.clear()
        for label in labels:
            item = QListWidgetItem(label)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            on = (label in ticked) if ticked is not None else label != mf.WINDOW
            item.setCheckState(Qt.CheckState.Checked if on else Qt.CheckState.Unchecked)
            self.quantity_peaks_list.addItem(item)
        self.quantity_box.setVisible(True)

    def quantity_labels(self) -> list[str]:
        return [self.quantity_peaks_list.item(i).text()
                for i in range(self.quantity_peaks_list.count())
                if self.quantity_peaks_list.item(i).checkState() == Qt.CheckState.Checked]

    def set_quantity_labels(self, labels) -> None:
        wanted = set(labels)
        for i in range(self.quantity_peaks_list.count()):
            item = self.quantity_peaks_list.item(i)
            item.setCheckState(Qt.CheckState.Checked if item.text() in wanted
                               else Qt.CheckState.Unchecked)

    def plot_x_values(self) -> list:
        """One x per fitted experiment: the x value column as it is now
        (so x can be filled or corrected after fitting), else the value
        the fit was made with."""
        by_key = {e["key"]: e for e in self.experiments}
        keys = self.result.get("keys") or [None] * len(self.result["experiments"])
        out = []
        for fit, key in zip(self.result["experiments"], keys):
            e = by_key.get(key)
            value = None
            if e is not None and str(e.get("x", "")).strip():
                try:
                    value = float(e["x"])
                except ValueError:
                    value = None
            if value is None and fit.get("x_value") is not None:
                try:
                    value = float(fit["x_value"])
                except (TypeError, ValueError):
                    value = None
            out.append(value)
        return out

    def plot_quantity(self, quiet: bool = False) -> bool:
        if not self.result:
            return False
        key = self.quantity_combo.currentData()
        labels = self.quantity_labels()
        x_values = self.plot_x_values()
        problem = None
        if key is None or not labels:
            problem = "Choose a quantity and tick at least one peak."
        elif any(x is None for x in x_values):
            missing = [f["label"] for f, x in zip(self.result["experiments"], x_values) if x is None]
            problem = ("Every experiment needs a numeric x value -- fill the x value column "
                       "(type, or 'x value from'). Missing: " + ", ".join(missing) + ".")
        if problem:
            self.quantity_note_label.setText(problem)
            if not quiet:
                QMessageBox.information(self, "Plot a fitted quantity", problem)
            return False
        xlabel = self.x_label_edit.text().strip() or self.result.get("x_label", "x")
        kwargs = dict(labels=labels, x_values=x_values, xlabel=xlabel,
                      x_transform=self.quantity_x_combo.currentData(),
                      y_transform=self.quantity_y_combo.currentData())

        def draw(figure):
            ax = figure.add_subplot(111)
            mf.plot_quantity_trend(self.result, key, ax=ax, **kwargs)
        try:
            if self._draw_quantities is not None:
                self._draw_quantities(draw)
            else:
                import matplotlib.figure
                draw(matplotlib.figure.Figure())
        except ValueError as exc:
            self.quantity_note_label.setText(str(exc))
            if not quiet:
                QMessageBox.information(self, "Plot a fitted quantity", str(exc))
            return False
        definition = mf._QUANTITY_DEFINITIONS.get(key)
        self.quantity_note_label.setText(
            f"{mf.quantity_title(key)} against {xlabel}"
            + (f" -- {definition}." if definition else "."))
        return True

    def quantity_csv(self) -> str:
        if not self.result or self.quantity_combo.currentData() is None:
            return ""
        return mf.quantity_table(self.result, self.quantity_combo.currentData(),
                                 self.quantity_labels(), x_values=self.plot_x_values(),
                                 xlabel=self.x_label_edit.text().strip() or "x")

    def _copy_quantity_csv(self) -> None:
        QGuiApplication.clipboard().setText(self.quantity_csv())

    def quantity_python(self) -> str:
        key = self.quantity_combo.currentData()
        if not self.result or key is None:
            return ""
        x_values = self.plot_x_values()
        return (
            "# The multi-spectrum fit's result: `result = fit_multi(...)` in a notebook.\n"
            "from opennmr.notebook import multi_fitting as mf\n"
            f"x_values = {[None if x is None else float(x) for x in x_values]!r}\n"
            f"ax = mf.plot_quantity_trend(result, {key!r}, labels={self.quantity_labels()!r},\n"
            f"                            x_values=x_values, "
            f"xlabel={self.x_label_edit.text().strip() or 'x'!r},\n"
            f"                            x_transform={self.quantity_x_combo.currentData()!r}, "
            f"y_transform={self.quantity_y_combo.currentData()!r})\n"
            f"print(mf.quantity_table(result, {key!r}, {self.quantity_labels()!r}, "
            "x_values=x_values))\n")

    def text_report(self) -> str:
        if not self.result:
            return ""
        with contextlib.redirect_stdout(io.StringIO()):
            return mf.print_multi_fit_report(self.result)

    def csv_text(self) -> str:
        return mf.multi_fit_to_csv(self.result) if self.result else ""

    def _on_results_menu(self, pos) -> None:
        menu = QMenu(self.results_table)
        copy_table = menu.addAction("Copy Table")
        export = menu.addAction("Export CSV…")
        copy_report = menu.addAction("Copy Full Report")
        chosen = menu.exec(self.results_table.viewport().mapToGlobal(pos))
        if chosen is copy_table:
            QGuiApplication.clipboard().setText(panel_state.table_text(self.results_table, "\t"))
        elif chosen is copy_report:
            QGuiApplication.clipboard().setText(self.text_report())
        elif chosen is export:
            path, _ = QFileDialog.getSaveFileName(self, "Export multi-spectrum fit", "multi_fit.csv",
                                                  "CSV (*.csv)")
            if path:
                mf.multi_fit_to_csv(self.result, path)

    # ---- saved state (processing log / Load Progress) -----------------------

    def get_state(self) -> dict:
        self._save_editor()
        state = {"widgets": panel_state.capture_widgets(self, exclude=(
                     "experiment_table", "parameter_table", "nucleus_combo")),
                 "experiments": copy.deepcopy(self.experiments),
                 "sites": copy.deepcopy(self.sites_by_nucleus),
                 "choices": [[list(k), v] for k, v in self.choices.items()],
                 "editing": self._editing_nucleus}
        extra = {"Experiments (in order)": ", ".join(
            f"{e['label']} ({e['nucleus']}, {e['larmor_MHz']:.2f} MHz"
            + (f", MAS {e['mas_Hz']:g} Hz" if e["mas_Hz"] else "")
            + (f", x {e['x']}" if str(e['x']).strip() else "") + ")" for e in self.experiments)}
        for n, dicts in self.sites_by_nucleus.items():
            for d in dicts:
                from .combined_panels import _describe_site
                extra[f"{n} site {d['label']}"] = _describe_site(d)
        for label, name, _n in self._rows():
            c = self._choice(label, name)
            bit = (f"{c['scope']}, {'free' if c['free'] else 'fixed'}"
                   + (f", in [{c['min'] or '…'}, {c['max'] or '…'}]" if c["min"] or c["max"] else ""))
            extra[f"{label} {_param_label(name) if name != 'delta_offset' else 'δ offset'}"] = bit
        state["extra"] = extra
        return state

    def set_state(self, state: dict) -> None:
        state = state or {}
        self.experiments = copy.deepcopy(state.get("experiments") or [])
        self.sites_by_nucleus = copy.deepcopy(state.get("sites") or {})
        self.choices = {tuple(k): dict(v) for k, v in state.get("choices") or []}
        self._editing_nucleus = None
        panel_state.restore_widgets(self, state.get("widgets") or {})
        self._fill_experiment_table()
        self.nucleus_combo.blockSignals(True)
        self.nucleus_combo.clear()
        self.nucleus_combo.addItems(self.nuclei())
        self.nucleus_combo.blockSignals(False)
        editing = state.get("editing")
        if editing in self.nuclei():
            self.nucleus_combo.setCurrentText(editing)
        self._editing_nucleus = None
        self._refreshing = True
        try:
            nucleus = self.nucleus_combo.currentText()
            self._editing_nucleus = nucleus or None
            if nucleus:
                self.site_editor.load(nucleus, copy.deepcopy(self.sites_by_nucleus.get(nucleus, [])))
        finally:
            self._refreshing = False
        self.refresh_parameter_table()
        self._sync()

    def result_state_for(self, key):
        """The result, encoded, with curves only for experiment ``key``
        (each experiment keeps its own; the rest are parameters only)."""
        if not self.result:
            return None
        r = {k: v for k, v in self.result.items() if k != "sequential"}
        fits = []
        for fit, k in zip(r["experiments"], r.get("keys", [])):
            sites = fit["sites"]
            fit = {name: value for name, value in fit.items() if name != "sites"}
            fit["site_dicts"] = [s.to_dict() if hasattr(s, "to_dict") else s for s in sites]
            if k != key:
                for name in ("x", "y", "y_fit", "site_spectra", "baseline", "mask"):
                    fit[name] = None
            fits.append(fit)
        r["experiments"] = fits
        return serialize.encode(r)

    def restore_result(self, encoded, spectrum=None) -> None:
        r = serialize.decode(encoded)
        for fit in r["experiments"]:
            fit["sites"] = [ch.Site.from_dict(d) for d in fit.pop("site_dicts", [])]
        r.setdefault("sequential", [])
        self.result = r
        self._report()
        # Point Show at the experiment whose curves this is.
        for j, fit in enumerate(r["experiments"]):
            if fit.get("x") is not None:
                self.show_combo.setCurrentIndex(j)
                break

    def clear_result(self) -> None:
        self.result = None
        self.results_label.setText("")
        self.results_table.clear()
        self.results_table.setRowCount(0)
        self.results_table.setVisible(False)
        self.show_combo.clear()
        self.quantity_box.setVisible(False)
        self.advice_box.set_items([])

    def report_text_for(self, key) -> str:
        """What one experiment's log records: the summary and its column."""
        if not self.result:
            return ""
        keys = self.result.get("keys", [])
        j = keys.index(key) if key in keys else 0
        fit = self.result["experiments"][j]
        lines = [self.results_label.text(), f"This spectrum ({fit['label']}):"]
        for label, _key, cells in self._result_rows():
            cell = cells[j]
            if cell is not None:
                lines.append(f"  {label} = {self._fmt(float(cell[0]), float(cell[1] or 0.0), not cell[2])}")
        return "\n".join(lines)

    def report_text(self) -> str:
        return self.results_label.text() if self.result else ""
