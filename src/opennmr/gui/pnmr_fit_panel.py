"""Fit > pNMR (CIF bond pathways): fit a measured spectrum against the
environments a CIF and its bond-pathway shifts predict, and read off which
environments make up each peak.

The structural model -- CIF, observed species and sites, ignored
species/sites, shells and their pathway shifts -- is the one set up in
Simulate > pNMR (CIF) (:class:`~opennmr.gui.pnmr_panel.PnmrSimulationPanel`),
never a second copy of it. This panel adds the fit:

1. **Structural model** -- a summary of what Simulate > pNMR holds, with
   buttons to go there or load saved settings into it.
2. **Maximal fit** -- one peak per group of predicted environments
   (:func:`opennmr.notebook.pnmr_fit.fit_predicted_spectrum`): shifts fixed
   at the prediction (or allowed a window), widths and areas fitted.
3. **Simplify and refit** (optional) -- merge low-area peaks, or merge onto
   shifts the user names (typed, or clicked on the plot), then refit with
   the shifts free (:func:`~opennmr.notebook.pnmr_fit.merge_low_integral_peaks`,
   :func:`~opennmr.notebook.pnmr_fit.merge_groups_to_picks`,
   :func:`~opennmr.notebook.pnmr_fit.refit_merged_groups`).

The fit is drawn on the Fit view through ``MplCanvas.show_slice_fit``, so
the Fit plot section (components shown, highlighted, filled, colours) and
Plot Style (fonts, ticks, export) style it like any other fit. This panel
adds what is particular to pNMR, as an overlay that survives restyling:
predicted environments as sticks, peaks labelled with their most likely
environment, and the selected peak outlined.

**Hover** over the fit to see the most likely environments under the
cursor (:func:`~opennmr.notebook.pnmr_fit.environments_at_shift`, which
weighs overlapping peaks by their share of the signal there); **click** a
peak to select it and list all of its environments.

All the maths is in :mod:`opennmr.notebook.pnmr_fit`; a notebook gets the
same numbers from the same calls (Copy notebook code).
"""

from __future__ import annotations

import contextlib
import csv
import io
import os
import re
import warnings
from typing import Callable

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtGui import QCursor, QGuiApplication
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMessageBox, QPlainTextEdit,
    QPushButton, QSpinBox, QTableWidget, QTableWidgetItem, QToolTip, QVBoxLayout, QWidget,
)

from ..io import serialize
from ..notebook import bridge
from ..notebook import pnmr_export, pnmr_fit
from ..notebook.pnmr_simulator import species_element
from . import panel_state
from .pnmr_export_row import SendSitesRow
from .fit_progress_dialog import FitProgress

#: The generate_spectrum_dict arguments that define the structural model
#: (the rest -- nuclide, field, broadening, axis -- only shape a simulated
#: spectrum and play no part in a fit).
MODEL_KEYS = ("cif_filename", "atom_of_interest", "shell_bins", "shift_dict",
              "ignore_species", "sites_of_interest", "ignore_sites", "weight_by_occupancy")

#: Environments listed in the hover tooltip.
HOVER_ROWS = 5

_GROUP_METHODS = (("probability", "Most probable first (no chaining)"),
                  ("sequential", "Sequential gaps (can chain)"),
                  ("hierarchical", "Single-linkage clustering (can chain)"))
_PEAK_TYPES = (("gaussian", "Gaussian"), ("lorentzian", "Lorentzian"),
               ("pseudo_voigt", "Pseudo-Voigt"))
_PEAK_NAMES = (("shift", "Fitted shift"), ("environment", "Most likely environment"),
               ("number", "Peak number"))


def _nucleus_element(nucleus) -> str | None:
    """'23Na' -> 'Na'; None if there is no element in it."""
    match = re.search(r"[A-Z][a-z]?", str(nucleus or ""))
    return match.group(0) if match else None


def parse_shifts(text: str) -> list[float]:
    """'120, 45.5 -10' -> [120.0, 45.5, -10.0]. Raises ValueError naming
    the first entry that is not a number."""
    out = []
    for token in re.split(r"[,;\s]+", text.strip()):
        if not token:
            continue
        try:
            out.append(float(token))
        except ValueError:
            raise ValueError(f"'{token}' is not a shift in ppm.") from None
    return out


def _unique(names: list[str]) -> list[str]:
    """Names made unique with ' (2)', ' (3)'... -- the fit display keys
    colours and highlighting on them, so two equal names would be one."""
    seen: dict[str, int] = {}
    out = []
    for name in names:
        seen[name] = seen.get(name, 0) + 1
        out.append(name if seen[name] == 1 else f"{name} ({seen[name]})")
    return out


def environment_name(member: dict) -> str:
    """'Na1: NN:5Br-+1Cl- NNN:12Na+' (the site only when known)."""
    site = member.get("site")
    return f"{site}: {member['label']}" if site else str(member["label"])


class PnmrFitPanel(QWidget):
    """The pNMR (CIF bond pathways) section of the Fit tile.

    ``model()`` returns the structural model (:data:`MODEL_KEYS`) or raises
    ValueError saying what is missing; ``describe_model()`` is a line or
    two about it; ``open_model_setup()`` goes to Simulate > pNMR (CIF);
    ``load_model_settings()`` loads a saved parameters file there.
    ``get_target()`` -> (spectrum, row, index) as the other fit panels;
    ``get_regions()`` -> the Integration regions; ``canvas`` is the Fit
    view's MplCanvas and ``show_view()`` raises it; ``open_fit_display()``
    opens the Fit plot (components and colours) section; ``on_fitted()``
    is called after every successful fit (to log and save it).
    """

    #: The selected peak's outline and hatching. Hatched rather than filled:
    #: a fill colour would always match some component's palette colour,
    #: while hatching reads the same over any colour and in greyscale.
    HIGHLIGHT_COLOUR = "0.1"

    def __init__(self, model: Callable[[], dict], describe_model: Callable[[], str],
                 open_model_setup: Callable[[], None], load_model_settings: Callable[[], None],
                 get_target: Callable, get_regions: Callable[[], list], canvas,
                 show_view: Callable[[], None], open_fit_display: Callable[[], None] | None = None,
                 on_fitted: Callable[[], None] | None = None,
                 export_sites: Callable | None = None, parent=None) -> None:
        super().__init__(parent)
        #: ``export_sites(site_dicts, notes, nucleus, larmor_mhz, destination,
        #: append, source)`` -- puts sites into CSA, Quad & J-coupling.
        self._export_sites = export_sites
        self._model = model
        self._describe_model = describe_model
        self._open_model_setup = open_model_setup
        self._load_model_settings = load_model_settings
        self._get_target = get_target
        self._get_regions = get_regions
        self._canvas = canvas
        self._show_view = show_view
        self._open_fit_display = open_fit_display
        self._on_fitted = on_fitted

        #: The fit on show, as plain data (see _result_from); None before one.
        self.result: dict | None = None
        #: The notebook's own output of the last *maximal* fit, which a
        #: refit merges from. Not saved: a restored fit can be shown and
        #: kept, and is fitted again before it can be simplified.
        self._maximal = None
        self._maximal_results = None      # the spectrum dict it was fitted to
        self.selected_peak: int | None = None
        self.hover_text = ""
        self._hover_index = None
        self._press_xy = None
        self._syncing = False
        self._highlight_artists: list = []

        layout = QVBoxLayout(self)
        intro = QLabel(
            "Fit the selected spectrum with the environments predicted from a CIF and its "
            "bond-pathway shifts (set up in Simulate ▸ pNMR (CIF)). Hover over the fit to see "
            "the most likely environments under the cursor; click a peak to list all of its "
            "environments.")
        intro.setWordWrap(True)
        layout.addWidget(intro)
        layout.addWidget(self._build_model_box())
        layout.addWidget(self._build_fit_box())
        layout.addWidget(self._build_refit_box())
        layout.addWidget(self._build_results_box())
        layout.addWidget(self._build_plot_box())
        layout.addWidget(self._build_log_box())

        mpl = canvas.canvas
        mpl.mpl_connect("motion_notify_event", self._on_hover)
        mpl.mpl_connect("button_press_event", self._on_press)
        mpl.mpl_connect("button_release_event", self._on_release)
        mpl.mpl_connect("figure_leave_event", lambda _e: self._clear_hover())
        canvas.peakPicked.connect(self._on_peak_picked)
        canvas.pickModeChanged.connect(self._on_pick_mode_changed)
        self.refresh_model()
        self._sync_controls()

    # ------------------------------------------------------------------
    # 1. Structural model (held by Simulate > pNMR)
    # ------------------------------------------------------------------
    def _build_model_box(self) -> QGroupBox:
        box = QGroupBox("1. Structural model")
        v = QVBoxLayout(box)
        self.model_summary_label = QLabel()
        self.model_summary_label.setWordWrap(True)
        self.model_summary_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        v.addWidget(self.model_summary_label)
        row = QHBoxLayout()
        self.setup_button = QPushButton("Set up in Simulate ▸ pNMR…")
        self.setup_button.setToolTip(
            "Load the CIF, choose the sites to observe and the shells and their bond-pathway "
            "shifts. The fit uses exactly what is set there.")
        self.setup_button.clicked.connect(self._open_model_setup)
        row.addWidget(self.setup_button)
        self.load_model_button = QPushButton("Load settings…")
        self.load_model_button.setToolTip(
            "Load a saved pNMR parameters file (as Simulate ▸ pNMR ▸ Load settings… does).")
        self.load_model_button.clicked.connect(self._on_load_model)
        row.addWidget(self.load_model_button)
        self.refresh_model_button = QPushButton("Refresh")
        self.refresh_model_button.setToolTip("Read the model from Simulate ▸ pNMR again.")
        self.refresh_model_button.clicked.connect(self.refresh_model)
        row.addWidget(self.refresh_model_button)
        row.addStretch(1)
        v.addLayout(row)
        return box

    def _on_load_model(self) -> None:
        self._load_model_settings()
        self.refresh_model()

    def model_kwargs(self) -> dict:
        """The structural model, as fit_predicted_spectrum takes it."""
        model = self._model()
        return {k: model[k] for k in MODEL_KEYS if k in model}

    def refresh_model(self) -> bool:
        """Show what the model is now; False (and why) if it is not ready."""
        try:
            self.model_kwargs()
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            self.model_summary_label.setText(f"<b>Not ready:</b> {exc}")
            self._model_ready = False
        else:
            self.model_summary_label.setText(self._describe_model())
            self._model_ready = True
        if hasattr(self, "tolerance_auto_check"):
            self._update_auto_tolerance()
            self._sync_controls()
        return self._model_ready

    def showEvent(self, event) -> None:  # noqa: N802 -- Qt override
        # The model lives in another panel: read it again whenever this
        # one comes into view, so the summary is never stale.
        self.refresh_model()
        super().showEvent(event)

    # ------------------------------------------------------------------
    # 2. Maximal fit
    # ------------------------------------------------------------------
    def _build_fit_box(self) -> QGroupBox:
        box = QGroupBox("2. Maximal fit (one peak per group of environments)")
        form = QFormLayout(box)
        self.use_regions_check = QCheckBox("Fit only the Integration regions")
        self.use_regions_check.setToolTip(
            "Fit only inside the regions of Fit ▸ Integration (and keep only the predicted "
            "environments that fall in them). Off: the whole spectrum.")
        form.addRow(self.use_regions_check)
        self.group_method_combo = QComboBox()
        for key, text in _GROUP_METHODS:
            self.group_method_combo.addItem(text, key)
        self.group_method_combo.setToolTip(
            "How predicted shifts closer than the tolerance are merged into one peak. 'Most "
            "probable first' seeds each group at the most probable environment left, so a group "
            "never spans more than the tolerance; the other two can chain neighbours together.")
        form.addRow("Grouping", self.group_method_combo)
        tol_row = QHBoxLayout()
        self.tolerance_auto_check = QCheckBox("Automatic")
        self.tolerance_auto_check.setChecked(True)
        self.tolerance_auto_check.setToolTip(
            "Group shifts closer than the smallest non-zero bond-pathway shift -- the smallest "
            "difference the model itself can make.")
        self.tolerance_spin = QDoubleSpinBox()
        self.tolerance_spin.setRange(0.001, 10000.0)
        self.tolerance_spin.setDecimals(3)
        self.tolerance_spin.setValue(1.0)
        self.tolerance_spin.setSuffix(" ppm")
        self.tolerance_auto_check.toggled.connect(self._sync_controls)
        tol_row.addWidget(self.tolerance_auto_check)
        tol_row.addWidget(self.tolerance_spin, 1)
        form.addRow("Tolerance", tol_row)
        self.snr_spin = QDoubleSpinBox()
        self.snr_spin.setRange(0.0, 1000.0)
        self.snr_spin.setDecimals(1)
        self.snr_spin.setValue(3.0)
        self.snr_spin.setToolTip(
            "Leave out predicted groups where the measured spectrum's local maximum is below "
            "this many times the noise. 0 keeps every group in the fit window.")
        form.addRow("Minimum S/N", self.snr_spin)
        self.peak_type_combo = QComboBox()
        for key, text in _PEAK_TYPES:
            self.peak_type_combo.addItem(text, key)
        form.addRow("Lineshape", self.peak_type_combo)
        fwhm_row = QHBoxLayout()
        self.fwhm_spin = QDoubleSpinBox()
        self.fwhm_spin.setRange(0.001, 10000.0)
        self.fwhm_spin.setDecimals(3)
        self.fwhm_spin.setValue(5.0)
        self.fwhm_spin.setSuffix(" ppm")
        self.fwhm_spin.setToolTip("Starting full width at half maximum of every peak.")
        self.fwhm_max_spin = QDoubleSpinBox()
        self.fwhm_max_spin.setRange(0.01, 100000.0)
        self.fwhm_max_spin.setDecimals(2)
        self.fwhm_max_spin.setValue(100.0)
        self.fwhm_max_spin.setSuffix(" ppm")
        self.fwhm_max_spin.setPrefix("max ")
        self.fwhm_max_spin.setToolTip("Largest FWHM a peak may take.")
        fwhm_row.addWidget(self.fwhm_spin, 1)
        fwhm_row.addWidget(self.fwhm_max_spin, 1)
        form.addRow("FWHM", fwhm_row)
        self.shift_window_spin = QDoubleSpinBox()
        self.shift_window_spin.setRange(0.0, 10000.0)
        self.shift_window_spin.setDecimals(3)
        self.shift_window_spin.setSuffix(" ppm")
        self.shift_window_spin.setSpecialValueText("fixed at the prediction")
        self.shift_window_spin.setToolTip(
            "0: every peak stays at its predicted shift, so the fit tests the model. More: each "
            "may move by up to this much (e.g. to absorb a referencing error).")
        form.addRow("Shift may move ±", self.shift_window_spin)
        self.fit_button = QPushButton("Fit")
        self.fit_button.setToolTip("Predict, group and fit (fit_predicted_spectrum).")
        self.fit_button.clicked.connect(self.run_fit)
        form.addRow(self.fit_button)
        return box

    def _update_auto_tolerance(self) -> None:
        if not self.tolerance_auto_check.isChecked():
            return
        try:
            tol = pnmr_fit.default_tolerance(self.model_kwargs()["shift_dict"])
        except (ValueError, TypeError, KeyError, AttributeError):
            return
        self.tolerance_spin.setValue(tol)

    def fit_settings(self) -> dict:
        """The fit_predicted_spectrum arguments the controls give (the
        structural model and the spectrum aside)."""
        regions = self._get_regions() if self.use_regions_check.isChecked() else None
        if self.use_regions_check.isChecked() and not regions:
            raise ValueError("'Fit only the Integration regions' is on, but Fit ▸ Integration "
                             "has no regions.")
        return {
            "fit_regions": [tuple(map(float, r)) for r in regions] if regions else None,
            "tolerance": None if self.tolerance_auto_check.isChecked()
            else self.tolerance_spin.value(),
            "group_method": self.group_method_combo.currentData(),
            "snr_threshold": self.snr_spin.value(),
            "peak_type": self.peak_type_combo.currentData(),
            "fwhm_init": self.fwhm_spin.value(),
            "fwhm_bounds": (min(0.01, self.fwhm_spin.value()), self.fwhm_max_spin.value()),
            "float_shift_window": self.shift_window_spin.value(),
        }

    def run_fit(self) -> bool:
        """The maximal fit of the selected spectrum (or displayed slice)."""
        try:
            model = self.model_kwargs()
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            self.refresh_model()
            QMessageBox.information(self, "pNMR fit", f"The structural model is not ready: {exc}")
            return False
        try:
            spectrum, row, index = self._get_target()
            settings = self.fit_settings()
            results = bridge.results_from_spectrum(spectrum, row)
        except ValueError as exc:
            QMessageBox.information(self, "pNMR fit", str(exc))
            return False
        if settings["fwhm_init"] > settings["fwhm_bounds"][1]:
            QMessageBox.information(self, "pNMR fit", "The starting FWHM is above its maximum.")
            return False
        self._check_nucleus(spectrum, model["atom_of_interest"])
        out = None
        with FitProgress(self, "pNMR fit") as progress:
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    out = pnmr_fit.fit_predicted_spectrum(
                        results, **model, **settings, show_plot=False, show_report=False)
                for w in caught:
                    if not issubclass(w.category, DeprecationWarning):
                        self._log(f"  Note: {w.message}")
            except Exception as exc:  # noqa: BLE001 -- shown, not raised
                self._log(f"Fit failed: {exc}")
                QMessageBox.warning(self, "pNMR fit", f"The fit failed:\n{exc}")
                return False
        if progress.stopped or out is None:
            return False
        self._maximal, self._maximal_results = out, results
        self.result = self._result_from(out, "maximal", spectrum, index, settings, model)
        if self.refit_window_spin.value() == 0:
            self.refit_window_spin.setValue(float(out["tolerance"]))
        self._after_fit()
        return True

    def _check_nucleus(self, spectrum, atom) -> None:
        nucleus = getattr(spectrum.metadata, "nucleus", None)
        have, want = _nucleus_element(nucleus), species_element(atom)
        if have and want and have != want:
            self._log(f"Warning: the spectrum is {nucleus} but the model observes {atom} -- "
                      f"check the spectrum and the observed species.")

    # ------------------------------------------------------------------
    # 3. Simplify and refit
    # ------------------------------------------------------------------
    def _build_refit_box(self) -> QGroupBox:
        box = QGroupBox("3. Simplify and refit (optional)")
        form = QFormLayout(box)
        self.merge_mode_combo = QComboBox()
        self.merge_mode_combo.addItem("Merge low-area peaks into their neighbours", "auto")
        self.merge_mode_combo.addItem("Merge onto shifts I choose", "picks")
        self.merge_mode_combo.currentIndexChanged.connect(self._sync_controls)
        form.addRow("Simplify by", self.merge_mode_combo)
        self.merge_relative_spin = QDoubleSpinBox()
        self.merge_relative_spin.setRange(0.0, 100.0)
        self.merge_relative_spin.setDecimals(1)
        self.merge_relative_spin.setValue(5.0)
        self.merge_relative_spin.setSuffix(" % of the largest")
        self.merge_relative_spin.setToolTip(
            "A peak with less area than this (or not significant at 2σ) is merged into its "
            "nearest neighbour, if that is within the merge distance.")
        form.addRow("Merge peaks below", self.merge_relative_spin)
        merge_row = QHBoxLayout()
        self.merge_tol_auto_check = QCheckBox("Grouping tolerance")
        self.merge_tol_auto_check.setChecked(True)
        self.merge_tol_auto_check.toggled.connect(self._sync_controls)
        self.merge_tol_spin = QDoubleSpinBox()
        self.merge_tol_spin.setRange(0.001, 100000.0)
        self.merge_tol_spin.setDecimals(3)
        self.merge_tol_spin.setValue(10.0)
        self.merge_tol_spin.setSuffix(" ppm")
        merge_row.addWidget(self.merge_tol_auto_check)
        merge_row.addWidget(self.merge_tol_spin, 1)
        form.addRow("Merge distance", merge_row)
        picks_row = QHBoxLayout()
        self.picks_edit = QLineEdit()
        self.picks_edit.setPlaceholderText("e.g. 120, 45.5, -10")
        self.picks_edit.setToolTip(
            "Peak centres (ppm) to keep: every peak of the maximal fit is merged onto the "
            "nearest one.")
        self.pick_button = QPushButton("Pick on plot")
        self.pick_button.setCheckable(True)
        self.pick_button.setToolTip("Click the Fit plot to add shifts; click this again to stop.")
        self.pick_button.toggled.connect(self._on_pick_toggled)
        picks_row.addWidget(self.picks_edit, 1)
        picks_row.addWidget(self.pick_button)
        form.addRow("Shifts", picks_row)
        self.refit_window_spin = QDoubleSpinBox()
        self.refit_window_spin.setRange(0.0, 100000.0)
        self.refit_window_spin.setDecimals(3)
        self.refit_window_spin.setSuffix(" ppm")
        self.refit_window_spin.setToolTip(
            "How far each merged peak's centre may move from its predicted (probability-"
            "weighted) shift in the refit. Set after a fit to the grouping tolerance.")
        form.addRow("Refit: shift may move ±", self.refit_window_spin)
        self.refit_button = QPushButton("Simplify && refit")
        self.refit_button.clicked.connect(self.run_refit)
        form.addRow(self.refit_button)
        return box

    def _on_pick_toggled(self, checked: bool) -> None:
        self._canvas.set_pick_mode(checked)
        if checked:
            self._show_view()

    def _on_pick_mode_changed(self, enabled: bool) -> None:
        if not enabled and self.pick_button.isChecked():
            self.pick_button.blockSignals(True)
            self.pick_button.setChecked(False)
            self.pick_button.blockSignals(False)

    def _on_peak_picked(self, x: float) -> None:
        if not self.pick_button.isChecked():
            return
        text = self.picks_edit.text().strip()
        self.picks_edit.setText((text + ", " if text else "") + f"{x:.2f}")

    def run_refit(self) -> bool:
        """Simplify the last maximal fit and refit it with shifts free."""
        if self._maximal is None:
            QMessageBox.information(self, "pNMR fit", "Run the maximal fit (step 2) first.")
            return False
        window = self.refit_window_spin.value()
        if window <= 0:
            QMessageBox.information(self, "pNMR fit",
                                    "Give the shift window for the refit (more than 0 ppm).")
            return False
        out = self._maximal
        mode = self.merge_mode_combo.currentData()
        printed = io.StringIO()
        try:
            with contextlib.redirect_stdout(printed):
                if mode == "auto":
                    merge_tol = (float(out["tolerance"]) if self.merge_tol_auto_check.isChecked()
                                 else self.merge_tol_spin.value())
                    merged, merge_log = pnmr_fit.merge_low_integral_peaks(
                        out["kept"], out["integrals"], out["integral_errors"],
                        relative_threshold=self.merge_relative_spin.value() / 100.0,
                        merge_tolerance=merge_tol)
                    how = (f"merged peaks below {self.merge_relative_spin.value():g}% within "
                           f"{merge_tol:g} ppm ({len(merge_log)} merge(s))")
                else:
                    picks = parse_shifts(self.picks_edit.text())
                    if not picks:
                        raise ValueError("Type or pick the shifts to merge onto.")
                    merged, _report = pnmr_fit.merge_groups_to_picks(
                        out["kept"], picks, integrals=out["integrals"])
                    how = "merged onto " + ", ".join(f"{p:g}" for p in picks) + " ppm"
        except ValueError as exc:
            QMessageBox.information(self, "pNMR fit", str(exc))
            return False
        for line in printed.getvalue().splitlines():
            self._log("  " + line.strip())
        settings = dict(self.result["settings"]) if self.result else {}
        spectrum, index = self.result["spectrum"], self.result.get("spectrum_index")
        refit = None
        with FitProgress(self, "pNMR refit") as progress:
            try:
                refit = pnmr_fit.refit_merged_groups(
                    merged, self._maximal_results, shift_float_window=window,
                    fit_regions=settings.get("fit_regions"),
                    snr_threshold=settings.get("snr_threshold", 3.0),
                    peak_type=settings.get("peak_type", "gaussian"),
                    fwhm_init=settings.get("fwhm_init", 5.0),
                    fwhm_bounds=settings.get("fwhm_bounds", (0.1, 100.0)),
                    show_plot=False, show_report=False)
            except Exception as exc:  # noqa: BLE001 -- shown, not raised
                self._log(f"Refit failed: {exc}")
                QMessageBox.warning(self, "pNMR fit", f"The refit failed:\n{exc}")
                return False
        if progress.stopped or refit is None:
            return False
        refit["tolerance"] = out["tolerance"]
        settings.update({"refit_shift_window": window, "simplified": how})
        self.result = self._result_from(refit, "refit", spectrum, index, settings,
                                        self.result.get("model") if self.result else {})
        self._after_fit()
        return True

    # ------------------------------------------------------------------
    # The result, as plain data (saved with the spectrum, kept, restored)
    # ------------------------------------------------------------------
    @staticmethod
    def _result_from(out: dict, stage: str, spectrum, index, settings: dict, model: dict) -> dict:
        dropped = []
        frame = out.get("dropped")
        if isinstance(frame, pd.DataFrame) and not frame.empty:
            for _, r in frame.iterrows():
                snr = r.get("snr")
                dropped.append({"shift": float(r["group_shift"]), "n_members": int(r["n_members"]),
                                "p": float(r["group_p"]), "reason": str(r.get("reason", "")),
                                "snr": float(snr) if snr is not None and np.isfinite(snr)
                                else None})
        y = np.asarray(out["y_data"], dtype=float)
        y_fit = np.asarray(out["y_model"], dtype=float)
        rmse = float(np.sqrt(np.mean((y - y_fit) ** 2))) if y.size else float("nan")
        return {
            "stage": stage,
            "x": np.asarray(out["x_fit"], dtype=float), "y": y, "y_fit": y_fit,
            "components": [np.asarray(c, dtype=float) for c in out["components"]],
            "peaks": pnmr_fit.summarise_peaks(out), "dropped": dropped,
            "tolerance": float(out["tolerance"]), "noise": float(out.get("noise", np.nan)),
            "rmse": rmse, "rel_rmse": rmse / (float(np.max(np.abs(y))) or 1.0),
            "bound_warnings": list(out.get("bound_warnings") or []),
            "regions": settings.get("fit_regions"),
            "settings": settings, "model": dict(model),
            "spectrum": spectrum, "spectrum_index": index,
        }

    def _after_fit(self) -> None:
        self.selected_peak = None
        self._report()
        self._log_fit()
        self._draw_fit(self.result)
        self._sync_controls()
        if self._on_fitted is not None:
            self._on_fitted()

    def result_state(self):
        if not self.result:
            return None
        return serialize.encode({k: v for k, v in self.result.items() if k != "spectrum"})

    def restore_result(self, encoded, spectrum=None) -> None:
        """Show a saved fit's tables as they were -- nothing is refitted or
        redrawn. Simplify needs the fit run again (its working tables are
        not saved)."""
        result = serialize.decode(encoded)
        result["spectrum"] = spectrum
        result["peaks"] = [dict(p) for p in result.get("peaks") or []]
        self.result = result
        self._maximal = self._maximal_results = None
        self.selected_peak = None
        self._report()
        self._sync_controls()

    def clear_result(self) -> None:
        self.result = None
        self._maximal = self._maximal_results = None
        self.selected_peak = None
        self.results_label.setText("")
        self.results_peaks_table.setRowCount(0)
        self.results_env_table.setRowCount(0)
        self.results_env_title.setText("")
        self._sync_controls()

    def get_state(self) -> dict:
        return panel_state.default_state(self)

    def set_state(self, state: dict) -> None:
        panel_state.restore_default_state(self, state)
        self._sync_controls()

    def report_text(self) -> str:
        """What the log records of the fit."""
        if not self.result:
            return ""
        lines = [self.results_label.text(), panel_state.table_text(self.results_peaks_table)]
        for p in self.result["peaks"]:
            top = ", ".join(f"{environment_name(m)} {100 * m['fraction_in_group']:.0f}%"
                            for m in p["members"][:3])
            lines.append(f"Peak {p['index'] + 1} ({p['fitted_shift']:.2f} ppm): {top}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Results: peaks, and the environments of the selected one
    # ------------------------------------------------------------------
    _PEAK_COLUMNS = ("Fitted δ\n(ppm)", "Predicted δ\n(ppm)", "FWHM\n(ppm)", "Area\n(%)",
                     "Envs", "Most likely environment")
    _ENV_COLUMNS = ("Site", "Environment", "δ\n(ppm)", "% of\npeak", "% of\ntotal")

    def _build_results_box(self) -> QGroupBox:
        box = QGroupBox("Results")
        v = QVBoxLayout(box)
        self.results_label = QLabel()
        self.results_label.setWordWrap(True)
        self.results_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        v.addWidget(self.results_label)
        self.results_peaks_table = QTableWidget(0, len(self._PEAK_COLUMNS))
        self.results_peaks_table.setHorizontalHeaderLabels(self._PEAK_COLUMNS)
        self.results_peaks_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.results_peaks_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        # Several rows (Ctrl/Shift-click) for Export Selected; the first
        # selected is the one whose environments are listed and outlined.
        self.results_peaks_table.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection)
        self.results_peaks_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents)
        self.results_peaks_table.horizontalHeader().setStretchLastSection(True)
        self.results_peaks_table.setMinimumHeight(140)
        self.results_peaks_table.itemSelectionChanged.connect(self._on_table_selection)
        v.addWidget(self.results_peaks_table)
        self.results_env_title = QLabel()
        v.addWidget(self.results_env_title)
        self.results_env_table = QTableWidget(0, len(self._ENV_COLUMNS))
        self.results_env_table.setHorizontalHeaderLabels(self._ENV_COLUMNS)
        self.results_env_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.results_env_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents)
        self.results_env_table.horizontalHeader().setStretchLastSection(False)
        self.results_env_table.setMinimumHeight(120)
        v.addWidget(self.results_env_table)
        self.hover_label = QLabel("Under the cursor: hover over the fit.")
        self.hover_label.setWordWrap(True)
        self.hover_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        v.addWidget(self.hover_label)
        row = QHBoxLayout()
        self.save_report_button = QPushButton("Save environments (CSV)…")
        self.save_report_button.setToolTip(
            "Every peak and every environment in it, with its share of the peak and of the "
            "total fitted area.")
        self.save_report_button.clicked.connect(self._on_save_report)
        row.addWidget(self.save_report_button)
        self.copy_code_button = QPushButton("Copy notebook code")
        self.copy_code_button.setToolTip("The notebook calls that give this fit.")
        self.copy_code_button.clicked.connect(self._on_copy_code)
        row.addWidget(self.copy_code_button)
        row.addStretch(1)
        v.addLayout(row)
        #: Where the main window puts "Keep This Fit…" (right after it).
        self.keep_anchor = self.copy_code_button
        # The fit's peaks are isotropic: send them on to have CSA,
        # quadrupolar or J added (explicit user request, 2026-10-12).
        self.send_sites_row = SendSitesRow(default_destination="fit", selected_what="peaks")
        self.send_sites_row.exportRequested.connect(self.export_sites)
        v.addWidget(self.send_sites_row)
        return box

    def selected_peak_indices(self) -> list[int]:
        model = self.results_peaks_table.selectionModel()
        return sorted({index.row() for index in model.selectedRows()}) if model else []

    def export_sites(self, which: str = "all") -> list[dict] | None:
        """The fitted peaks (all, or the selected rows) as sites in CSA,
        Quad & J-coupling (:func:`notebook.pnmr_export.sites_from_pnmr_fit`)."""
        if not self.result:
            return None
        chosen = None if which == "all" else self.selected_peak_indices()
        if which != "all" and not chosen:
            QMessageBox.information(self, "pNMR", "Select one or more peaks in the table first.")
            return None
        spectrum = self.result.get("spectrum")
        meta = getattr(spectrum, "metadata", None)
        nucleus = getattr(meta, "nucleus", None) or self._model_nuclide()
        try:
            sites, notes = pnmr_export.sites_from_pnmr_fit(self.result, nucleus, which=chosen,
                                                           return_notes=True)
        except ValueError as exc:
            QMessageBox.warning(self, "pNMR", str(exc))
            return None
        larmor = getattr(meta, "bf1", None) or getattr(meta, "sfo1", None)
        destination = self.send_sites_row.destination()
        self._log(f"Exported {len(sites)} site(s) to "
                  f"{'Fit' if destination == 'fit' else 'Simulate'} ▸ CSA, Quad & J-coupling"
                  + (" (added)" if self.send_sites_row.append() else "") + ": "
                  + ", ".join(s["label"] for s in sites) + ".\n  " + "\n  ".join(notes))
        if self._export_sites is not None:
            self._export_sites(sites, notes, nucleus, larmor, destination,
                               self.send_sites_row.append(), "pNMR fit")
        return sites

    def _model_nuclide(self) -> str | None:
        settings = (self.result or {}).get("settings") or {}
        return settings.get("nuclide")

    def _report(self) -> None:
        r = self.result
        table = self.results_peaks_table
        self._syncing = True
        table.setRowCount(0)
        self.results_env_table.setRowCount(0)
        self.results_env_title.setText("")
        if not r:
            self.results_label.setText("")
            self._syncing = False
            return
        stage = ("Maximal fit (shifts fixed at the prediction)"
                 if r["stage"] == "maximal" and not r["settings"].get("float_shift_window")
                 else "Maximal fit" if r["stage"] == "maximal"
                 else f"Simplified refit ({r['settings'].get('simplified', '')})")
        n_env = sum(p["n_members"] for p in r["peaks"])
        text = (f"<b>{stage}</b>: {len(r['peaks'])} peak(s) from {n_env} predicted "
                f"environment(s), grouped within {r['tolerance']:.4g} ppm. "
                f"RMS residual {100 * r['rel_rmse']:.2f}% of the largest point.")
        if r["dropped"]:
            reasons = {}
            for d in r["dropped"]:
                reasons[d["reason"]] = reasons.get(d["reason"], 0) + 1
            text += " Left out: " + "; ".join(f"{n} {why}" for why, n in reasons.items()) + "."
        if r["bound_warnings"]:
            text += f" <i>{len(r['bound_warnings'])} parameter(s) at a bound -- see the log.</i>"
        self.results_label.setText(text)
        table.setRowCount(len(r["peaks"]))
        table.setVerticalHeaderLabels([str(p["index"] + 1) for p in r["peaks"]])
        for row, p in enumerate(r["peaks"]):
            top = p["members"][0] if p["members"] else None
            cells = (
                self._value_error(p["fitted_shift"], p.get("shift_err"), 2,
                                  fixed=p.get("shift_fixed")),
                f"{p['predicted_shift']:.2f}",
                self._value_error(p["fwhm"], p.get("fwhm_err"), 2),
                self._value_error(p["integral_pct"], self._pct_err(p), 1),
                str(len(p["members"])),
                (f"{environment_name(top)} ({100 * top['fraction_in_group']:.0f}%)"
                 if top else ""),
            )
            for col, value in enumerate(cells):
                item = QTableWidgetItem(value)
                if col < 5:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                table.setItem(row, col, item)
        self._syncing = False

    @staticmethod
    def _pct_err(p: dict):
        err = p.get("integral_err")
        if err is None or not p.get("integral"):
            return None
        return abs(p["integral_pct"] * err / p["integral"])

    @staticmethod
    def _value_error(value: float, error, decimals: int, fixed: bool = False) -> str:
        if fixed:
            return f"{value:.{decimals}f} (fixed)"
        if error is None or not np.isfinite(error) or error <= 0:
            return f"{value:.{decimals}f}"
        return f"{value:.{decimals}f} ± {error:.{decimals}f}"

    def select_peak(self, index: int | None) -> None:
        """Select peak ``index`` (zero-based; None to clear): list its
        environments, select its table row and outline it on the plot."""
        if self.result is None or (index is not None
                                   and not 0 <= index < len(self.result["peaks"])):
            index = None
        self.selected_peak = index
        if not self._syncing:
            self._syncing = True
            if index is None:
                self.results_peaks_table.clearSelection()
            else:
                self.results_peaks_table.selectRow(index)
            self._syncing = False
        self._fill_environments(index)
        self._draw_highlight()

    def _on_table_selection(self) -> None:
        if self._syncing:
            return
        rows = self.results_peaks_table.selectionModel().selectedRows()
        self._syncing = True
        try:
            self.select_peak(rows[0].row() if rows else None)
        finally:
            self._syncing = False

    def _fill_environments(self, index: int | None) -> None:
        table = self.results_env_table
        table.setRowCount(0)
        if index is None or self.result is None:
            self.results_env_title.setText("")
            return
        p = self.result["peaks"][index]
        self.results_env_title.setText(
            f"<b>Peak {index + 1}</b> at {p['fitted_shift']:.2f} ppm: "
            f"{len(p['members'])} environment(s), most likely first")
        table.setRowCount(len(p["members"]))
        for row, m in enumerate(p["members"]):
            copies = f" ×{m['n_copies']}" if m.get("n_copies", 1) > 1 else ""
            cells = (f"{m.get('site') or ''}{copies}", str(m["label"]), f"{m['shift']:.2f}",
                     f"{100 * m['fraction_in_group']:.1f}",
                     f"{p['integral_pct'] * m['fraction_in_group']:.2f}")
            for col, value in enumerate(cells):
                item = QTableWidgetItem(value)
                if col >= 2:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                if col == 1:
                    item.setToolTip(pnmr_fit.format_environment_label(
                        m["shift"], m.get("composition") or {}, decimals=2))
                table.setItem(row, col, item)

    # ------------------------------------------------------------------
    # Fit plot: drawn by the Fit view, styled there; our overlay on top
    # ------------------------------------------------------------------
    def _build_plot_box(self) -> QGroupBox:
        box = QGroupBox("Fit plot")
        form = QFormLayout(box)
        self.peak_names_combo = QComboBox()
        for key, text in _PEAK_NAMES:
            self.peak_names_combo.addItem(text, key)
        self.peak_names_combo.setToolTip("How each peak is named in the legend and in the "
                                         "Fit plot section's colour list.")
        self.peak_names_combo.currentIndexChanged.connect(self.redraw)
        form.addRow("Name peaks by", self.peak_names_combo)
        self.annotate_check = QCheckBox("Label peaks with their most likely environment")
        self.annotate_check.setToolTip(
            "Write each peak's fitted shift and the full shell composition of its most likely "
            "environment above it (labels are stacked so they never overlap).")
        self.annotate_check.toggled.connect(self.redraw)
        form.addRow(self.annotate_check)
        label_row = QHBoxLayout()
        self.max_labels_spin = QSpinBox()
        self.max_labels_spin.setRange(1, 200)
        self.max_labels_spin.setValue(8)
        self.max_labels_spin.setPrefix("largest ")
        self.max_labels_spin.setSuffix(" peaks")
        self.max_labels_spin.setToolTip("Label only this many peaks, the largest by area.")
        self.max_labels_spin.valueChanged.connect(self.redraw)
        self.label_size_spin = QSpinBox()
        self.label_size_spin.setRange(4, 30)
        self.label_size_spin.setValue(8)
        self.label_size_spin.setSuffix(" pt")
        self.label_size_spin.valueChanged.connect(self.redraw)
        label_row.addWidget(self.max_labels_spin, 1)
        label_row.addWidget(self.label_size_spin)
        form.addRow("Labels", label_row)
        self.sticks_check = QCheckBox("Show predicted environments as sticks")
        self.sticks_check.setToolTip(
            "A stick at each predicted environment's shift along the bottom of the plot, its "
            "height its probability -- shows what each peak was built from.")
        self.sticks_check.toggled.connect(self.redraw)
        form.addRow(self.sticks_check)
        self.hover_check = QCheckBox("Show environments under the cursor")
        self.hover_check.setChecked(True)
        self.hover_check.setToolTip("Hovering over the fit lists the most likely environments "
                                    "there; clicking a peak selects it.")
        form.addRow(self.hover_check)
        row = QHBoxLayout()
        self.components_button = QPushButton("Components and colours…")
        self.components_button.setToolTip("Show, hide or highlight peaks, fill, colours "
                                           "(right-click the fit ▸ Fit Plot: Components and Colours…).")
        self.components_button.clicked.connect(
            lambda: self._open_fit_display() if self._open_fit_display else None)
        row.addWidget(self.components_button)
        row.addStretch(1)
        form.addRow(row)
        hint = QLabel("Fonts, ticks, size and export: right-click the plot ▸ Plot Style.")
        hint.setWordWrap(True)
        form.addRow(hint)
        return box

    def peak_names(self, result: dict | None = None) -> list[str]:
        result = result or self.result
        if not result:
            return []
        mode = self.peak_names_combo.currentData()
        names = []
        for p in result["peaks"]:
            if mode == "environment" and p["members"]:
                names.append(environment_name(p["members"][0]))
            elif mode == "number":
                names.append(f"Peak {p['index'] + 1}")
            else:
                names.append(f"{p['fitted_shift']:.1f} ppm")
        return _unique(names)

    def _draw_fit(self, result: dict) -> None:
        """Draw ``result`` on the Fit view (also how "Keep This Fit…"
        redraws it)."""
        spectrum = result.get("spectrum")
        self._canvas.show_slice_fit(
            result["x"], result["y"], result["y_fit"], components=result["components"],
            peak_labels=self.peak_names(result), title="pNMR bond-pathway fit",
            spectrum=spectrum if spectrum is not None and spectrum.ndim == 1 else None,
            overlay=self._overlay)
        self._show_view()

    def is_showing(self) -> bool:
        """Whether the Fit view shows this panel's fit right now."""
        return self.result is not None and self._canvas.slice_fit_overlay() == self._overlay

    def redraw(self) -> None:
        """Draw the fit again with the current plot options (only if it is
        the one on show -- never take the Fit view over from another fit)."""
        if self.is_showing():
            self._draw_fit(self.result)

    def _overlay(self, ax) -> None:
        r = self.result
        if not r:
            return
        if self.sticks_check.isChecked():
            pnmr_fit.add_predicted_sticks(ax, r, zorder=3)
        if self.annotate_check.isChecked():
            kept = pd.DataFrame({"members": [p["members"] for p in r["peaks"]],
                                 "group_p": [p["group_p"] for p in r["peaks"]]})
            pnmr_fit.add_dominant_environment_labels(
                ax, r["x"], r["components"][:-1], kept, shade=False,
                fontsize=self.label_size_spin.value(), decimals=1,
                integrals_norm=[p["integral_pct"] for p in r["peaks"]],
                max_labels=self.max_labels_spin.value())
        self._highlight_artists = []
        self._draw_highlight(ax, draw=False)

    def _draw_highlight(self, ax=None, draw: bool = True) -> None:
        """Outline the selected peak (replacing any previous outline)."""
        for artist in self._highlight_artists:
            try:
                artist.remove()
            except (ValueError, AttributeError, NotImplementedError):
                pass
        self._highlight_artists = []
        if ax is None:
            if not self.is_showing():
                return
            ax = self._canvas.ax
        k = self.selected_peak
        if ax is not None and k is not None and self.result is not None:
            r = self.result
            y = np.asarray(r["components"][k], dtype=float)
            if self._canvas.fit_component_style().get("on_baseline"):
                y = y + np.asarray(r["components"][-1], dtype=float)
            base = (np.asarray(r["components"][-1], dtype=float)
                     if self._canvas.fit_component_style().get("on_baseline")
                     else np.zeros_like(y))
            fill = ax.fill_between(r["x"], base, y, facecolor="none",
                                   edgecolor=self.HIGHLIGHT_COLOUR, hatch="////",
                                   linewidth=0, zorder=2.6, label="_selected_peak")
            line, = ax.plot(r["x"], y, color=self.HIGHLIGHT_COLOUR, linewidth=1.8, zorder=6,
                            label="_selected_peak")
            self._highlight_artists.extend([fill, line])
        if draw:
            self._canvas.canvas.draw_idle()

    # ---- hover and click -------------------------------------------------

    def _interaction_free(self, event) -> bool:
        c = self._canvas
        return (self.is_showing() and event.inaxes is not None and event.inaxes is c.ax
                and event.xdata is not None and getattr(c.toolbar, "mode", "") == ""
                and not getattr(c, "_pick_mode", False)
                and not getattr(c, "_region_select_mode", False)
                and not getattr(c, "_measurement_mode", False))

    def environments_at(self, x: float, top_n: int | None = HOVER_ROWS) -> pd.DataFrame:
        return pnmr_fit.environments_at_shift(self.result, x, top_n=top_n)

    def hover_at(self, x: float) -> str:
        """The read-out for the cursor at ``x`` ppm (also put in the panel)."""
        everything = self.environments_at(x, top_n=None)
        shown = everything.head(HOVER_ROWS)
        shown.attrs = everything.attrs
        text = pnmr_fit.format_environments(shown, total=len(everything))
        self.hover_text = text
        self.hover_label.setText("Under the cursor: " + text.replace("\n", "\n   "))
        return text

    def _on_hover(self, event) -> None:
        if not self.hover_check.isChecked() or not self._interaction_free(event):
            if self._hover_index is not None:
                self._clear_hover()
            return
        x = np.asarray(self.result["x"])
        k = int(np.argmin(np.abs(x - float(event.xdata))))
        if k == self._hover_index:
            return
        self._hover_index = k
        text = self.hover_at(float(event.xdata))
        QToolTip.showText(QCursor.pos(), text, self._canvas.canvas)

    def _clear_hover(self) -> None:
        self._hover_index = None
        QToolTip.hideText()

    def _on_press(self, event) -> None:
        self._press_xy = (event.x, event.y) if event.button == 1 else None

    def _on_release(self, event) -> None:
        start, self._press_xy = self._press_xy, None
        if (start is None or event.button != 1 or event.x is None
                or not self._interaction_free(event)):
            return
        if abs(event.x - start[0]) > 4 or abs(event.y - start[1]) > 4:
            return   # a pan drag, not a click
        self.click_at(float(event.xdata))

    def click_at(self, x: float) -> int | None:
        """Select the peak contributing most at ``x`` ppm (None: no peak)."""
        index = pnmr_fit.peak_index_at_shift(self.result, x)
        self.select_peak(index)
        return index

    # ------------------------------------------------------------------
    # Export, notebook code, log
    # ------------------------------------------------------------------
    def report_rows(self) -> list[dict]:
        rows = []
        for p in (self.result or {}).get("peaks", []):
            for m in p["members"]:
                rows.append({
                    "peak": p["index"] + 1, "fitted_shift_ppm": p["fitted_shift"],
                    "predicted_shift_ppm": p["predicted_shift"], "fwhm_ppm": p["fwhm"],
                    "peak_area_pct": p["integral_pct"], "site": m.get("site") or "",
                    "environment": m["label"], "environment_shift_ppm": m["shift"],
                    "copies": m.get("n_copies", 1), "probability": m["p"],
                    "pct_of_peak": 100 * m["fraction_in_group"],
                    "pct_of_total": p["integral_pct"] * m["fraction_in_group"],
                })
        return rows

    def save_report(self, path: str) -> None:
        rows = self.report_rows()
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else ["peak"])
            writer.writeheader()
            writer.writerows(rows)
        self._log(f"Saved environments to {path}")

    def _on_save_report(self) -> None:
        if not self.result:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save environments",
                                              "pnmr_fit_environments.csv", "CSV (*.csv)")
        if path:
            self.save_report(path)

    def notebook_code(self) -> str:
        r = self.result
        model = r["model"] if r else self.model_kwargs()
        settings = r["settings"] if r else self.fit_settings()
        spectrum = r.get("spectrum") if r else None
        meta = getattr(spectrum, "metadata", None)
        source = getattr(meta, "source_path", None)
        lines = ["import opennmr.notebook as nmr", ""]
        if source and getattr(meta, "vendor", None) == "bruker" and os.path.isdir(str(source)):
            lines.append(f'results = nmr.NMR1dimport(r"{source}")')
        else:
            lines.append("results = ...  # the spectrum dict the GUI fitted (e.g. nmr.NMR1dimport"
                         "(path), or opennmr.notebook.bridge.results_from_spectrum)")
        lines.append("fit = nmr.fit_predicted_spectrum(")
        lines.append("    results,")
        for key in MODEL_KEYS:
            if key in model:
                value = model[key]
                lines.append(f'    {key}=r"{value}",' if key == "cif_filename"
                             else f"    {key}={value!r},")
        for key in ("fit_regions", "tolerance", "group_method", "snr_threshold", "peak_type",
                    "fwhm_init", "fwhm_bounds", "float_shift_window"):
            if key in settings:
                lines.append(f"    {key}={settings[key]!r},")
        lines.append("    show_composition_labels=True,")
        lines.append(")")
        if r and r["stage"] == "refit":
            lines.append("# then simplified: " + settings.get("simplified", ""))
            lines.append("# merged, _ = nmr.merge_low_integral_peaks(fit['kept'], fit['integrals'], "
                         "fit['integral_errors'], merge_tolerance=fit['tolerance'])  # or "
                         "merge_groups_to_picks")
            lines.append(f"# fit = nmr.refit_merged_groups(merged, results, shift_float_window="
                         f"{settings.get('refit_shift_window')!r})")
        lines.append("")
        lines.append("# The most likely environments at a shift (as hovering in the GUI):")
        shift = r["peaks"][0]["fitted_shift"] if r and r["peaks"] else 0.0
        lines.append(f"print(nmr.format_environments(nmr.environments_at_shift(fit, {shift:.2f})))")
        return "\n".join(lines)

    def _on_copy_code(self) -> None:
        try:
            code = self.notebook_code()
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            QMessageBox.information(self, "pNMR fit", str(exc))
            return
        QGuiApplication.clipboard().setText(code)
        self._log("Copied notebook code to the clipboard.")

    def _build_log_box(self) -> QGroupBox:
        box = QGroupBox("Log")
        v = QVBoxLayout(box)
        self.status_log = QPlainTextEdit()
        self.status_log.setReadOnly(True)
        self.status_log.setMaximumHeight(140)
        v.addWidget(self.status_log)
        return box

    def _log(self, text: str) -> None:
        self.status_log.appendPlainText(text)

    def _log_fit(self) -> None:
        r = self.result
        spectrum = r.get("spectrum")
        name = os.path.basename(str(getattr(getattr(spectrum, "metadata", None),
                                            "source_path", "") or "")) or "spectrum"
        cif = os.path.basename(str(r["model"].get("cif_filename", "")))
        self._log(f"{'Maximal fit' if r['stage'] == 'maximal' else 'Refit'} of {name} with "
                  f"{cif}: {len(r['peaks'])} peak(s), tolerance {r['tolerance']:.4g} ppm.")
        for d in r["dropped"][:20]:
            snr = f", S/N {d['snr']:.1f}" if d.get("snr") is not None else ""
            self._log(f"  Left out {d['shift']:.2f} ppm ({d['n_members']} env.{snr}): "
                      f"{d['reason']}")
        if len(r["dropped"]) > 20:
            self._log(f"  … and {len(r['dropped']) - 20} more left out.")
        for w in r["bound_warnings"]:
            self._log(f"  At a bound: {w}")

    def _sync_controls(self) -> None:
        if not hasattr(self, "refit_button"):
            return
        self.tolerance_spin.setEnabled(not self.tolerance_auto_check.isChecked())
        auto = self.merge_mode_combo.currentData() == "auto"
        for w in (self.merge_relative_spin, self.merge_tol_auto_check):
            w.setEnabled(auto)
        self.merge_tol_spin.setEnabled(auto and not self.merge_tol_auto_check.isChecked())
        self.picks_edit.setEnabled(not auto)
        self.pick_button.setEnabled(not auto)
        self.refit_button.setEnabled(self._maximal is not None)
        has = self.result is not None
        for w in (self.save_report_button, self.copy_code_button):
            w.setEnabled(has)
        self.send_sites_row.set_enabled(has, has)
