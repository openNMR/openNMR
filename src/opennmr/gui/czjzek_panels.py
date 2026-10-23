"""Fit ▸ Quadrupolar (Czjzek) and Simulate ▸ Czjzek distribution.

The maths is :mod:`opennmr.notebook.czjzek` (the functions a notebook
calls). The fit panel reuses every setup control of the quadrupolar fit
panel (nucleus, field, static/MAS, transitions, level of theory,
broadening, window, baseline) -- only the sites differ: a Czjzek site is
δiso, σ and d instead of one (C_Q, η_Q).

Both panels can show the distribution of environments as a heat map
(C_Q across, η_Q up) with the fit or simulation beside it, above or to
the right (:func:`opennmr.notebook.czjzek.plot_czjzek_distribution`).
"""

from __future__ import annotations

import time

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication, QComboBox, QDoubleSpinBox, QFormLayout, QHBoxLayout, QLabel, QMessageBox,
    QProgressDialog, QPushButton, QTableWidgetItem, QVBoxLayout, QWidget,
)

from .widgets.typed_spinbox import TypedSpinBox
from .fit_progress_dialog import FitProgress
from ..core import registry as core_registry
from ..core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
from ..io import serialize
from ..notebook import czjzek as cz
from ..notebook import fitting as notebook_fitting
from ..notebook import quadrupolar_fitting as qfit
from . import panel_state
from .quadrupolar_panels import QuadrupolarFitPanel
from .widgets.report_label import ReportLabel
from .widgets.field_input import link_field

#: Ask before a library build expected to take longer than this (s).
_ASK_ABOVE_SECONDS = 8.0

#: Simulate's starting grid (explicit user request, 2026-10-01: a 12 x 4
#: grid takes seconds where a finite-MAS 32 x 10 took many minutes).
SIM_GRID_DEFAULT = (12, 4)

_LAYOUTS = (("Spectrum to the right", "right"), ("Spectrum above", "top"))

#: The two distribution models (explicit user request: a central C_Q).
_MODELS = (("Czjzek (random EFG centred on zero)", "czjzek"),
           ("Extended Czjzek (mean EFG (C_Q0, η_Q0) + random part)", "extended"))
_MODEL_TIP = (
    "Czjzek's model is a fully random EFG centred on zero: it has no 'central' C_Q. Its "
    "⟨C_Q⟩ ≈ 2σ, its most probable C_Q and ⟨η_Q⟩ = 0.61 all follow from σ alone – shown "
    "beside the sites.\n\n"
    "The extended Czjzek model (Le Caër & Brand 1998) adds a mean EFG: a site with a "
    "definite local structure (C_Q0, η_Q0) spread by disorder of width σ. As σ → 0 it becomes "
    "a single (C_Q0, η_Q0) site; at C_Q0 = 0 it is Czjzek's model. d is 5 in this model.")


def describe_distribution(sigma, d=5.0, cq0=0.0, eta0=0.0) -> str:
    """One line on where a distribution sits: its mean and most probable
    C_Q and its mean η_Q (what σ, or σ with C_Q0, imply)."""
    st = cz.czjzek_statistics(sigma, d, cq0=cq0, eta0=eta0, n_cq=300, n_eta=100)
    text = (f"⟨C_Q⟩ = {st['mean_CQ']:.3g} MHz, most probable C_Q {st['mode_CQ']:.3g} MHz, "
            f"⟨η_Q⟩ = {st['mean_eta']:.2f}")
    if cq0 > 0:
        text += f" (centred on C_Q0 = {cq0:g} MHz, η_Q0 = {eta0:g})"
    else:
        text += " (random EFG centred on zero: set by σ alone)"
    return text


def _format_duration(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 5400:
        return f"{seconds / 60:.0f} min"
    return f"{seconds / 3600:.1f} h"


def library_estimate(settings, sigma, d, n_cq, n_eta, cq0=0.0, eta0=0.0, ppm_axis=None) -> dict:
    """:func:`opennmr.notebook.czjzek.estimate_library_seconds` for these
    settings -- one lineshape timed on this computer, times the grid."""
    cq_max = cz.czjzek_cq_quantile(0.9999, sigma, d, cq0, eta0) * (1.1 if cq0 > 0 else 1.0)
    cq_check = cz.czjzek_cq_quantile(0.99, sigma, d, cq0, eta0)
    if ppm_axis is None:
        lo, hi = sorted(settings.get("ppm", (-150.0, 150.0)))
        ppm_axis = np.linspace(lo, hi, 8192)
    return cz.estimate_library_seconds(
        settings["I"], settings["larmor_Hz"], ppm_axis, cq_max, cq_check, n_cq=n_cq, n_eta=n_eta,
        mas_Hz=settings["mas_Hz"], order=settings["order"], transitions=settings["transitions"],
        n_zcw=settings.get("n_zcw") or 4181)


def _slow_warning(parent, settings, sigma, d, n_cq, n_eta, stages=2, cq0=0.0, eta0=0.0,
                  ppm_axis=None) -> bool:
    """True to go ahead. When every grid lineshape has to be simulated
    (satellites, 'exact', or slow MAS), times one of them on this
    computer and asks first if the whole grid would take more than a few
    seconds -- an estimate that is measured, not assumed (the fixed table
    it replaces said "1 minute" for a build that took ten)."""
    try:
        est = library_estimate(settings, sigma, d, n_cq, n_eta, cq0, eta0, ppm_axis)
    except Exception:  # noqa: BLE001 -- an estimate must never stop a simulation
        return True
    seconds = est["seconds"] * stages
    if est["method"] == "scaling" or seconds < _ASK_ABOVE_SECONDS:
        return True
    faster = max(1, int(round(est["cells"] * _ASK_ABOVE_SECONDS / max(seconds, 1e-9))))
    answer = QMessageBox.question(
        parent, "This will be slow",
        f"Every lineshape of the {n_cq} × {n_eta} grid has to be simulated: {est['reason']}.\n\n"
        f"One lineshape takes {est['per_cell']:.2f} s on this computer, so the {est['cells']} "
        f"lineshapes{' (twice: the fit refines its grid)' if stages > 1 else ''} take about "
        f"{_format_duration(seconds)}. You can stop it part-way.\n\n"
        f"Go ahead? (A grid of about {faster} cells, or a static/central-transition "
        f"simulation, is quicker.)")
    return answer == QMessageBox.StandardButton.Yes


class LibraryProgress:
    """A Stop-able progress dialog for a lineshape-library build, as the
    ``progress(done, total)`` callback the notebook functions take. Keeps
    the window responsive while the lineshapes are simulated (the build
    runs on the GUI thread; a frozen window is what Windows reports as
    "not responding" and offers to close)."""

    def __init__(self, parent, title: str) -> None:
        self._parent, self._title = parent, title
        self._dialog = None
        self._started = time.perf_counter()
        self.cancelled = False

    def __call__(self, done: int, total: int) -> bool:
        if done >= total:
            # Library complete: make way for the fit's own pop-up.
            self.close()
            return True
        if self._dialog is None:
            if time.perf_counter() - self._started < 0.4:
                return True
            self._dialog = QProgressDialog(f"{self._title}: simulating lineshapes…", "Stop", 0,
                                           int(total), self._parent)
            self._dialog.setWindowTitle(self._title)
            self._dialog.setWindowModality(Qt.WindowModality.WindowModal)
            self._dialog.setMinimumDuration(0)
        self._dialog.setMaximum(int(total))
        self._dialog.setValue(int(done))
        elapsed = time.perf_counter() - self._started
        if done:
            left = elapsed / done * (total - done)
            self._dialog.setLabelText(f"{self._title}: lineshape {done} of {total}, about "
                                      f"{_format_duration(left)} left")
        QApplication.processEvents()
        if self._dialog.wasCanceled():
            self.cancelled = True
            return False
        return True

    def close(self) -> None:
        if self._dialog is not None:
            self._dialog.reset()
            self._dialog.close()
            self._dialog.deleteLater()
            self._dialog = None


def citation_html(extended: bool) -> str:
    """The models' references, as a small linked paragraph."""
    refs = []
    for text in cz.citations(extended):
        doi = text.rsplit("doi:", 1)[-1]
        refs.append(text.replace(f"doi:{doi}", f'<a href="https://doi.org/{doi}">doi:{doi}</a>'))
    lead = "Extended Czjzek model" if extended else "Czjzek model"
    return f"<small>{lead}: " + "; ".join(refs) + ".</small>"


def _citation_label() -> QLabel:
    label = QLabel("")
    label.setWordWrap(True)
    label.setTextFormat(Qt.TextFormat.RichText)
    label.setOpenExternalLinks(True)
    label.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
    return label


class CzjzekFitPanel(QuadrupolarFitPanel):
    """As :class:`QuadrupolarFitPanel`, plus ``draw_map(figure_draw)``:
    show a ``draw(figure)`` on the Fit view (the distribution map)."""

    COLUMNS = ("Label", "δiso / ppm", "σ / MHz", "d", "FWHM / ppm", "L fraction",
               "C_Q0 / MHz", "η_Q0")
    COLUMN_PARAMS = {1: "d_iso", 2: "sigma", 3: "d", 4: "fwhm", 5: "eta", 6: "CQ0", 7: "eta0"}
    PARAM_LABELS = dict(cz.CZJZEK_PARAM_LABELS)
    PARAM_NAMES = tuple(cz.CZJZEK_PARAM_NAMES)
    BLURB = ("Fit Czjzek distributions of quadrupolar couplingsto the selected 1D spectrum. Valid for disordered sites of "
             "glasses and amorphous solids. Each site has a shift of δiso, "
             "a width, σ, of its C_Q distribution, and critical exponent, d, (5 for the Gaussian Isotropic Model, held unless "
             "you unfix it; right-click it). FWHM is the full width at half-maximum for the isotropic chemical shift. "
             "Czjzek's model is centred on a zero EFG (no central C_Q; ⟨C_Q⟩ ≈ 2σ); "
             "the extended model adds a mean EFG, C_Q0 and η_Q0. Switch between the original and extended in the 'Distribution' dropdown.")
    FIT_TEXT = "Fit Czjzek Distribution"
    SEND_TEXT = "Send to Simulate ▸ Czjzek"
    DEFAULT_SITE = ("0", "2", "5", "3", "0.5", "0", "0")

    def __init__(self, get_target, get_regions, draw_fit, draw_map, format_value_error,
                 send_to_simulation=None, on_fitted=None, parent=None) -> None:
        super().__init__(get_target, get_regions, draw_fit, format_value_error,
                         send_to_simulation=send_to_simulation, on_fitted=on_fitted,
                         parent=parent)
        self._draw_map = draw_map
        self._library = None
        # Which distribution (explicit user request: give the central C_Q).
        self.model_combo = QComboBox()
        for text, data in _MODELS:
            self.model_combo.addItem(text, data)
        self.model_combo.setToolTip(_MODEL_TIP)
        self.model_combo.currentIndexChanged.connect(lambda _i: self._on_model_changed())
        self._form.insertRow(0, "Distribution:", self.model_combo)
        # What the typed σ (and C_Q0) imply, live: plain Czjzek's C_Q is
        # set by σ alone, which is not obvious from σ.
        self.distribution_label = QLabel("")
        self.distribution_label.setWordWrap(True)
        self.distribution_label.setToolTip(
            "Mean and most probable C_Q, and mean η_Q, of each site's starting distribution.")
        layout0 = self.layout()
        layout0.insertWidget(layout0.indexOf(self.sites_table) + 1, self.distribution_label)
        from PySide6.QtCore import QTimer
        self._describe_timer = QTimer(self)
        self._describe_timer.setSingleShot(True)
        self._describe_timer.setInterval(0)
        self._describe_timer.timeout.connect(self._describe_sites)
        self.sites_table.itemChanged.connect(lambda _i: self._describe_timer.start())
        self.sites_table.setToolTip(
            "Starting values, one row per Czjzek site. σ is the width of the C_Q distribution "
            "(MHz); ⟨C_Q⟩ ≈ 2σ for d = 5. d is fixed at 5 unless you unfix it.\n\n"
            "Right-click a value to fix or free it, or for bounds (fixed values in italics).")
        self.estimate_button.setToolTip(
            "Set the first site's δiso, σ and FWHM from the pattern's width and centre of "
            "gravity over the fit window. A start for the fit, not a result.")
        # One set of orientations serves the whole grid; no second pass.
        self.orientations_spin.setToolTip(
            "Powder orientations for each lineshape of the (C_Q, η_Q) grid (one octant, exact "
            "by symmetry). Auto: 4,181. The distribution averages most powder ripple out.")
        self._form.labelForField(self.render_orientations_spin).setVisible(False)
        self.render_orientations_spin.setVisible(False)

        grid = QWidget()
        g = QHBoxLayout(grid)
        g.setContentsMargins(0, 0, 0, 0)
        self.grid_cq_spin, self.grid_eta_spin = TypedSpinBox(), TypedSpinBox()
        for spin, lo, hi, value, text in ((self.grid_cq_spin, 8, 400, 64, "CQ cells"),
                                          (self.grid_eta_spin, 4, 100, 20, "ηQ cells")):
            spin.setRange(lo, hi)
            spin.setValue(value)
            spin.setSuffix(f" {text}")
            g.addWidget(spin)
        grid.setToolTip(
            "The (C_Q, η_Q) grid the distribution is summed over. C_Q runs to 99.99 % of the "
            "distribution; after a first fit the grid is re-sized to the fitted σ and the fit "
            "repeated, so the answer does not depend on the starting value. The report warns "
            "if too few cells cover the distribution.")
        self._form.addRow("Grid:", grid)
        self._on_model_changed()

        # The distribution map (explicit user request: C_Q across, η_Q
        # up, heat map, with the fit beside it above or to the right).
        map_box = QWidget()
        m = QFormLayout(map_box)
        m.setContentsMargins(0, 0, 0, 0)
        self.map_layout_combo = QComboBox()
        for text, data in _LAYOUTS:
            self.map_layout_combo.addItem(text, data)
        m.addRow("Map layout:", self.map_layout_combo)
        self.map_site_combo = QComboBox()
        self.map_site_combo.setToolTip("All sites together, each weighted by its area (the "
                                       "sample's distribution), or one site.")
        m.addRow("Map of:", self.map_site_combo)
        buttons = QHBoxLayout()
        self.show_fit_button = QPushButton("Show Fit")
        self.show_fit_button.clicked.connect(lambda: self.result and self._draw_fit(self.result))
        self.show_map_button = QPushButton("Show Distribution Map")
        self.show_map_button.setToolTip(
            "The distribution of environments (C_Q across, η_Q up) as a heat map, with the "
            "data, fit and sites beside it – on the Fit view, exportable from there.")
        self.show_map_button.clicked.connect(self.show_map)
        buttons.addWidget(self.show_fit_button)
        buttons.addWidget(self.show_map_button)
        m.addRow(buttons)
        for w in (self.show_fit_button, self.show_map_button):
            w.setEnabled(False)
        layout = self.layout()
        layout.insertWidget(layout.indexOf(self.send_button), map_box)

    # ---- sites ------------------------------------------------------------

    def add_site(self, values=None) -> None:
        if values is not None and len(values) < len(self.COLUMNS):
            values = tuple(values) + tuple(self.DEFAULT_SITE[len(values) - 1:])
        super().add_site(values)
        row = self.sites_table.rowCount() - 1
        bounds, fixed = self._constraints(row)
        if not fixed and not bounds:
            self.set_constraints(row, {}, {"d": True})

    def model(self) -> str:
        combo = getattr(self, "model_combo", None)
        return combo.currentData() if combo is not None else "czjzek"

    def _sync_controls(self) -> None:
        super()._sync_controls()
        extended = self.model() == "extended"
        for column in (6, 7):
            self.sites_table.setColumnHidden(column, not extended)

    def _on_model_changed(self) -> None:
        self._sync_controls()
        if not hasattr(self, "citation_label"):
            self.citation_label = _citation_label()
            self.layout().addWidget(self.citation_label)
        self.citation_label.setText(citation_html(self.model() == "extended"))
        if self.model() == "extended":
            # d is 5 in the extended model: put it back and hold it.
            for row in range(self.sites_table.rowCount()):
                item = self.sites_table.item(row, 3)
                if item is not None:
                    item.setText("5")
                bounds, fixed = self._constraints(row)
                if not fixed.get("d"):
                    self.set_constraints(row, bounds, {**fixed, "d": True})
        self._describe_sites()

    def set_state(self, state: dict) -> None:
        super().set_state(state)
        self._on_model_changed()

    def _describe_sites(self) -> None:
        if not hasattr(self, "distribution_label"):
            return
        lines = []
        extended = self.model() == "extended"
        for row in range(self.sites_table.rowCount()):
            def value(column, default=0.0):
                item = self.sites_table.item(row, column)
                try:
                    return float(item.text()) if item is not None and item.text().strip() \
                        else default
                except ValueError:
                    return None
            sigma, d = value(2), value(3, 5.0)
            cq0, eta0 = (value(6), value(7)) if extended else (0.0, 0.0)
            label_item = self.sites_table.item(row, 0)
            label = (label_item.text().strip() if label_item else "") or f"Site {row + 1}"
            if None in (sigma, d, cq0, eta0) or sigma <= 0 or d <= 0 or cq0 < 0 \
                    or not 0 <= eta0 <= 1:
                lines.append(f"{label}: —")
                continue
            try:
                lines.append(f"{label}: " + describe_distribution(
                    sigma, 5.0 if extended else d, cq0, eta0))
            except Exception:  # noqa: BLE001 -- a hint must never block typing
                lines.append(f"{label}: —")
        self.distribution_label.setText("\n".join(lines))

    def read_sites(self) -> list[dict]:
        sites = []
        for row in range(self.sites_table.rowCount()):
            def cell(column, name):
                item = self.sites_table.item(row, column)
                text = item.text().strip() if item is not None else ""
                try:
                    return float(text)
                except ValueError as exc:
                    raise ValueError(f"Site row {row + 1}: {name} must be a number, "
                                     f"got {text!r}.") from exc
            label_item = self.sites_table.item(row, 0)
            site = {"label": (label_item.text().strip() if label_item else "") or f"Site {row + 1}",
                    "d_iso": cell(1, "δiso"), "sigma": cell(2, "σ"), "d": cell(3, "d"),
                    "fwhm": cell(4, "FWHM")}
            if site["sigma"] <= 0:
                raise ValueError(f"Site row {row + 1}: σ must be > 0.")
            if self.model() == "extended":
                site["model"] = "extended"
                site["CQ0"] = cell(6, "C_Q0") if self._has_text(row, 6) else 0.0
                site["eta0"] = cell(7, "η_Q0") if self._has_text(row, 7) else 0.0
                if site["CQ0"] < 0 or not 0 <= site["eta0"] <= 1:
                    raise ValueError(f"Site row {row + 1}: C_Q0 must be ≥ 0 and η_Q0 in [0, 1].")
            item = self.sites_table.item(row, 5)
            if item is not None and item.text().strip():
                site["eta"] = cell(5, "L fraction")
            bounds, fixed = self._constraints(row)
            site["fixed"] = {name: bool(fixed.get(name)) for name in cz.CZJZEK_PARAM_NAMES}
            if self.model() == "extended" and (not site["fixed"]["d"] or site["d"] != 5):
                raise ValueError(f"Site row {row + 1}: the extended Czjzek model is defined for "
                                 "d = 5 – set d to 5 and keep it fixed.")
            if bounds:
                site["bounds"] = {n: tuple(b) for n, b in bounds.items()}
            sites.append(site)
        if not sites:
            raise ValueError("Add at least one site before fitting.")
        return sites

    def _has_text(self, row, column) -> bool:
        item = self.sites_table.item(row, column)
        return item is not None and bool(item.text().strip())

    def _grid(self):
        return self.grid_cq_spin.value(), self.grid_eta_spin.value()

    # ---- actions -------------------------------------------------------------

    def estimate(self) -> None:
        try:
            settings = self._settings()
            _spectrum, _index, results = self._target_results()
        except (ValueError, IndexError) as exc:
            QMessageBox.warning(self, "Estimate", str(exc))
            return
        start = qfit.estimate_quadrupolar_start(
            results, settings["I"], larmor_Hz=settings["larmor_Hz"], mas_Hz=settings["mas_Hz"],
            eta_Q=0.61, transitions=settings["transitions"], fit_regions=self.fit_regions())
        if self.model() == "extended":
            # A mean EFG near the pattern's C_Q, spread by a quarter of it.
            values = ((1, start["d_iso"]), (2, 0.25 * start["C_Q"]), (4, start["fwhm"]),
                      (6, 0.9 * start["C_Q"]), (7, start.get("eta_Q", 0.5)))
        else:
            # ⟨C_Q⟩ = 1.995 σ for d = 5.
            ratio = cz.czjzek_statistics(1.0, 5.0)["mean_CQ"]
            values = ((1, start["d_iso"]), (2, start["C_Q"] / ratio), (4, start["fwhm"]))
        for column, value in values:
            item = self.sites_table.item(0, column)
            if item is None:
                self.sites_table.setItem(0, column, QTableWidgetItem(""))
                item = self.sites_table.item(0, column)
            item.setText(f"{float(value):.4g}")

    def fit(self) -> None:
        try:
            settings = self._settings()
            sites = self.read_sites()
            spectrum, index, results = self._target_results()
        except (ValueError, IndexError) as exc:
            QMessageBox.warning(self, "Czjzek fit", str(exc))
            return
        n_cq, n_eta = self._grid()
        widest = max(sites, key=lambda s: s["sigma"] + s.get("CQ0", 0.0))
        if not _slow_warning(self, settings, 1.5 * widest["sigma"], widest["d"], n_cq, n_eta,
                             cq0=widest.get("CQ0", 0.0), eta0=widest.get("eta0", 0.0)):
            return
        regions = self.fit_regions()
        progress = LibraryProgress(self, "Czjzek fit")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        started = time.perf_counter()
        with FitProgress(self, "Czjzek fit") as fit_progress_box:
            try:
                popt, pcov_free, is_free, _scale, info = cz.fit_czjzek_spectrum(
                    results, sites, fit_regions=regions, I=settings["I"],
                    larmor_Hz=settings["larmor_Hz"], mas_Hz=settings["mas_Hz"],
                    order=settings["order"], transitions=settings["transitions"],
                    lineshape=settings["lineshape"], n_zcw=settings["n_zcw"] or 4181,
                    fit_baseline=settings["fit_baseline"], n_cq=n_cq, n_eta=n_eta,
                    progress=progress)
                x = np.sort(np.asarray(results["x_ppm"], dtype=float))
                y_fit, components = cz.evaluate_czjzek_fit(x, popt, info)
            except cz.SimulationCancelled:
                QMessageBox.information(self, "Czjzek fit", "Fit stopped. A coarser grid is quicker.")
                return
            except Exception as exc:  # noqa: BLE001 -- a fit that fails is an outcome, not a crash
                QMessageBox.warning(self, "Czjzek fit", f"The fit did not succeed:\n\n{exc}")
                return
            finally:
                progress.close()
                QApplication.restoreOverrideCursor()
        if fit_progress_box.stopped:
            return
        elapsed = time.perf_counter() - started
        self._library = info.pop("library")
        order = np.argsort(results["x_ppm"])
        y = np.asarray(results["real_spectrum"], dtype=float)[order]
        pcov = notebook_fitting.expand_pcov(pcov_free, is_free, len(popt))
        _target, row, _index = self._get_target()
        self.result = {"popt": popt, "pcov": pcov, "is_free": is_free, "info": info,
                       "x": x, "y": y, "y_fit": y_fit, "components": components,
                       "regions": regions, "spectrum": spectrum, "index": index, "row": row,
                       "settings": settings, "elapsed": elapsed, "title": "Czjzek fit",
                       "nucleus": self.nucleus_combo.currentText().strip(),
                       "report": cz.czjzek_fit_report(popt, pcov, is_free, info)}
        self._draw_fit(self.result)
        self._report()
        self.send_button.setEnabled(self._send_to_simulation is not None)
        self.use_fitted_button.setEnabled(True)
        self._on_fitted()

    def _map_sites(self):
        r = self.result
        n = r["info"]["n_sites"]
        integrate = getattr(np, "trapezoid", None) or np.trapz
        areas = [abs(float(integrate(c, r["x"]))) for c in r["components"][:n]]
        rows = r["report"]
        sites = [{"sigma": rows[s]["sigma"], "d": rows[s]["d"], "weight": areas[s],
                  "CQ0": rows[s].get("CQ0", 0.0), "eta0": rows[s].get("eta0", 0.0),
                  "label": rows[s]["label"]} for s in range(n)]
        choice = self.map_site_combo.currentData()
        return sites if choice is None else [sites[int(choice)]]

    def map_draw_function(self, layout=None, component_style=None):
        """``draw(figure)`` for the distribution map of the fit on show."""
        r = self.result
        if not r:
            return None
        sites = self._map_sites()
        layout = layout or self.map_layout_combo.currentData()
        mask = (notebook_fitting.build_region_mask(r["x"], r["regions"]) if r["regions"]
                else np.ones(r["x"].size, dtype=bool))
        from ..notebook.fit_display import default_component_colours, component_style as cs
        style = cs(component_style)
        n = r["info"]["n_sites"]
        labels = list(r["info"]["labels"])
        defaults = default_component_colours([np.asarray(c)[mask] for c in r["components"][:n]],
                                             labels)
        for site in sites:
            site["colour"] = style["line_colours"].get(site["label"], defaults[site["label"]])
        xlabel = None
        spectrum = r.get("spectrum")
        nucleus = getattr(getattr(spectrum, "metadata", None), "nucleus", None) or r.get("nucleus")
        if nucleus:
            try:
                xlabel = core_registry.NUC_label(nucleus, x_unit="ppm")
            except Exception:  # noqa: BLE001 -- a label must not sink a plot
                xlabel = None

        def draw(figure):
            cz.plot_czjzek_distribution(
                sites, spectrum={"x": r["x"][mask], "y": r["y"][mask], "y_fit": r["y_fit"][mask],
                                 "components": [np.asarray(c)[mask] for c in r["components"]],
                                 "labels": labels, "xlabel": xlabel},
                layout=layout, figure=figure, component_style=style)
        return draw

    def show_map(self) -> None:
        draw = self.map_draw_function()
        if draw is not None:
            self._draw_map(draw)

    # ---- report ----------------------------------------------------------------

    def _report(self) -> None:
        r = self.result
        info, rows = r["info"], r["report"]
        n = info["n_sites"]
        pv = r["settings"]["lineshape"] == "pseudo_voigt"
        half_integer = not float(r["settings"]["I"]).is_integer()
        any_ext = any(row.get("extended") for row in rows)
        cols = [("δiso\n(ppm)", "d_iso", True), ("σ\n(MHz)", "sigma", True)]
        cols += ([("C_Q0\n(MHz)", "CQ0", True), ("η_Q0", "eta0", True)] if any_ext
                 else [("d", "d", True)])
        cols += [("⟨C_Q⟩\n(MHz)", "mean_CQ", False), ("C_Q mode\n(MHz)", "mode_CQ", False),
                ("⟨η_Q⟩", "mean_eta", False), ("⟨P_Q⟩\n(MHz)", "mean_PQ", False)]
        if half_integer and "delta_cg" in rows[0]:
            cols.append(("CT c.o.g.\n(ppm)", "delta_cg", False))
        cols.append(("FWHM\n(ppm)", "fwhm", True))
        if pv:
            cols.append(("L fraction", "eta", True))
        cols.append(("Amplitude", "amplitude", True))
        integrate = getattr(np, "trapezoid", None) or np.trapz
        areas = np.array([abs(integrate(c, r["x"])) for c in r["components"][:n]])
        shares = 100 * areas / areas.sum() if areas.sum() else np.full(n, np.nan)
        table = self.results_table
        table.clear()
        headers = [c[0] for c in cols] + ["Area\n(%)"]
        table.setColumnCount(len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.setRowCount(n)
        table.setVerticalHeaderLabels(list(info["labels"]))
        for s, row in enumerate(rows):
            for c, (_h, key, is_param) in enumerate(cols):
                if is_param:
                    text = self._fmt(row[key], row[key + "_err"], row[key + "_fixed"])
                elif key == "delta_cg":
                    text = self._fmt(row[key], row["d_iso_err"])
                elif key == "mean_eta" and not row.get("mean_eta_err"):
                    text = f"{row[key]:.3f}"
                elif key in ("CQ0", "eta0") and not row.get("extended"):
                    text = "—"
                else:
                    text = self._fmt(row[key], row.get(key + "_err", float("nan")))
                item = QTableWidgetItem(text.replace(" ± ", "\n± ").replace(" (fixed)", "\n(fixed)"))
                item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                table.setItem(s, c, item)
            share = QTableWidgetItem("—" if not np.isfinite(shares[s]) else f"{shares[s]:.1f}")
            share.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            table.setItem(s, len(cols), share)
        table.resizeRowsToContents()
        table.setVisible(True)

        mask = (notebook_fitting.build_region_mask(r["x"], r["regions"]) if r["regions"]
                else np.ones(r["x"].size, dtype=bool))
        y, y_fit = r["y"][mask], r["y_fit"][mask]
        quality = notebook_fitting.fit_quality(y, y_fit, int(np.sum(r["is_free"])))
        total = float(np.sum((y - y.mean()) ** 2))
        r2 = 1 - float(np.sum((y - y_fit) ** 2)) / total if total else float("nan")
        lines = [info["log"].replace("Czjzek fit: ", "Czjzek fit — "),
                 f"R² {r2:.4f}  ·  RMSE {quality['rmse']:.3g}  ·  reduced χ² "
                 f"{quality['reduced_chi2']:.3g}  ·  {r['elapsed']:.1f} s. ⟨C_Q⟩, the C_Q mode, "
                 "⟨η_Q⟩ and ⟨P_Q⟩ are averages over each fitted distribution"
                 + (" (for the extended model, sample averages; errors propagated from σ, C_Q0 "
                    "and η_Q0). C_Q0 and η_Q0 are the mean EFG the disorder spreads around – not "
                    "⟨C_Q⟩, which the random part raises." if any_ext else
                    " (their errors follow from σ's; ⟨η_Q⟩ depends on d only). Czjzek's model is "
                    "centred on a zero EFG: there is no central C_Q, and ⟨C_Q⟩ comes from σ.")]
        lines += [f"⚠ {w}" for w in info.get("warnings", [])]
        self.results_label.setText("\n".join(lines))

        blocked = self.map_site_combo.blockSignals(True)
        self.map_site_combo.clear()
        self.map_site_combo.addItem("All sites (weighted by area)", None)
        for s, label in enumerate(info["labels"]):
            self.map_site_combo.addItem(label, s)
        self.map_site_combo.blockSignals(blocked)
        for w in (self.show_fit_button, self.show_map_button):
            w.setEnabled(True)

    def clear_result(self) -> None:
        super().clear_result()
        for w in (self.show_fit_button, self.show_map_button):
            w.setEnabled(False)

    def result_state(self):
        if not self.result:
            return None
        return serialize.encode({k: v for k, v in self.result.items() if k != "spectrum"})

    def text_report(self) -> str:
        if not self.result:
            return ""
        lines = [self.results_label.text(), "", self.table_tsv()]
        return "\n".join(lines)

    def use_fitted_as_start(self) -> None:
        r = self.result
        if r and any(row.get("extended") for row in r.get("report") or []):
            self.model_combo.setCurrentIndex(self.model_combo.findData("extended"))
        super().use_fitted_as_start()
        self._describe_sites()

    def set_start_from_simulation(self, values: dict, settings: dict | None = None) -> str:
        extended = (values.get("CQ0") or 0) > 0 or (settings or {}).get("model") == "extended"
        self.model_combo.setCurrentIndex(self.model_combo.findData(
            "extended" if extended else "czjzek"))
        notes = super().set_start_from_simulation(values, settings)
        self._describe_sites()
        return notes

    def _on_send(self) -> None:
        r = self.result
        if not r or self._send_to_simulation is None:
            return
        row = r["report"][self.send_site_index()]
        s = r["settings"]
        self._send_to_simulation({
            "nucleus": r["nucleus"], "sigma": row["sigma"], "d": row["d"],
            "CQ0": row.get("CQ0", 0.0), "eta0": row.get("eta0", 0.0),
            "model": "extended" if row.get("extended") else "czjzek",
            "delta_iso": row["d_iso"], "fwhm": row["fwhm"], "larmor_MHz": s["larmor_Hz"] / 1e6,
            "mas_Hz": s["mas_Hz"], "transitions": s["transitions"], "order": s["order"],
            "ppm_range": (float(np.min(r["x"])), float(np.max(r["x"]))),
        })


class CzjzekSimPanel(QWidget):
    """Simulate ▸ Czjzek distribution. Callbacks: ``draw(figure_draw)``
    shows a ``draw(figure)`` on the Simulation plot; ``add_to_open(spectrum,
    label)`` adds the simulated spectrum to Open experiments."""

    def __init__(self, draw, add_to_open=None, parent=None, send_to_fit=None, style=None) -> None:
        super().__init__(parent)
        self._draw = draw
        #: The Simulation plot's Plot Style, read as the map is drawn
        #: (text size, the map colour scale).
        self._style = style or (lambda: {})
        self._add_to_open = add_to_open
        self._send_to_fit = send_to_fit
        self.result: dict | None = None
        v = QVBoxLayout(self)
        blurb = QLabel(
            "The powder spectrum of a site whose quadrupolar coupling is spread over a Czjzek "
            "distribution (glasses, disordered solids), with the distribution of environments "
            "(C_Q along x, η_Q along y) drawn as a heat map beside it.")
        blurb.setWordWrap(True)
        v.addWidget(blurb)
        form = QFormLayout()
        self.nucleus_combo = QComboBox()
        self.nucleus_combo.setEditable(True)
        self.nucleus_combo.addItems(core_registry.quadrupolar_nuclides(40))
        self.nucleus_combo.setCurrentText("27Al")
        form.addRow("Nucleus:", self.nucleus_combo)
        self.larmor_spin = self._spin(0.0, 5000.0, 130.3, 4, " MHz", "ν0 (the spectrometer's SFO1).")
        form.addRow("Larmor frequency:", self.larmor_spin)
        # Enter the magnet's field; ν0 follows from it and the nucleus
        # (explicit user request, as in Simulate ▸ Quadrupolar).
        self.field_link = link_field(form, self.larmor_spin,
                                     lambda: self.nucleus_combo.currentText(),
                                     self.nucleus_combo.currentTextChanged)
        self.field_spin = self.field_link.field_spin
        self.regime_combo = QComboBox()
        self.regime_combo.addItem("Static powder", "static")
        self.regime_combo.addItem("MAS", "mas")
        self.regime_combo.currentIndexChanged.connect(lambda _i: self._sync())
        form.addRow("Regime:", self.regime_combo)
        self.mas_spin = self._spin(0.0, 200000.0, 20000.0, 0, " Hz", "Spinning rate.")
        form.addRow("MAS rate:", self.mas_spin)
        self.transitions_combo = QComboBox()
        self.transitions_combo.addItem("Central transition only", "ct")
        self.transitions_combo.addItem("CT + satellites", "all")
        form.addRow("Transitions:", self.transitions_combo)
        self.order_combo = QComboBox()
        self.order_combo.addItem("Second-order perturbation", "second")
        self.order_combo.addItem("Exact (diagonalization)", "exact")
        form.addRow("Model:", self.order_combo)
        self.model_combo = QComboBox()
        for text, data in _MODELS:
            self.model_combo.addItem(text, data)
        self.model_combo.setToolTip(_MODEL_TIP)
        self.model_combo.currentIndexChanged.connect(lambda _i: self._sync())
        form.addRow("Distribution:", self.model_combo)
        self.cq0_spin = self._spin(0.0, 100.0, 3.0, 3, " MHz",
                                   "C_Q0: the mean EFG's coupling – the local, ordered structure "
                                   "the disorder spreads around.")
        form.addRow("C_Q0 (mean EFG):", self.cq0_spin)
        self.eta0_spin = self._spin(0.0, 1.0, 0.0, 3, "", "η_Q0: the mean EFG's asymmetry.")
        self.eta0_spin.setSingleStep(0.05)
        form.addRow("η_Q0 (mean EFG):", self.eta0_spin)
        self.sigma_spin = self._spin(0.001, 100.0, 2.0, 3, " MHz",
                                     "Width of the random part of the EFG. In Czjzek's model "
                                     "this alone sets the C_Q distribution: ⟨C_Q⟩ ≈ 2σ for d = 5.")
        form.addRow("σ:", self.sigma_spin)
        self.d_spin = self._spin(1.0, 20.0, 5.0, 2, "",
                                 "5 is Czjzek's model (a fully random EFG); other values give "
                                 "the Gaussian isotropic model's other members. The extended "
                                 "model is defined for d = 5.")
        form.addRow("d:", self.d_spin)
        # What the inputs imply, live (explicit user request: where the
        # distribution sits was not visible).
        self.distribution_label = QLabel("")
        self.distribution_label.setWordWrap(True)
        form.addRow("Implies:", self.distribution_label)
        self._form = form
        for spin in (self.cq0_spin, self.eta0_spin, self.sigma_spin, self.d_spin):
            spin.valueChanged.connect(lambda _v: self._describe())
        self.diso_spin = self._spin(-1e5, 1e5, 60.0, 2, " ppm", "Mean isotropic chemical shift.")
        form.addRow("δiso:", self.diso_spin)
        self.fwhm_spin = self._spin(0.0, 1e4, 4.0, 2, " ppm",
                                    "Width of the isotropic chemical-shift distribution "
                                    "(Gaussian broadening).")
        form.addRow("FWHM (δiso spread):", self.fwhm_spin)
        axis = QWidget()
        a = QHBoxLayout(axis)
        a.setContentsMargins(0, 0, 0, 0)
        self.ppm_min_spin = self._spin(-1e5, 1e5, -150.0, 1, " ppm", "Axis low end.")
        self.ppm_max_spin = self._spin(-1e5, 1e5, 150.0, 1, " ppm", "Axis high end.")
        a.addWidget(self.ppm_min_spin)
        a.addWidget(QLabel("to"))
        a.addWidget(self.ppm_max_spin)
        form.addRow("Axis:", axis)
        grid = QWidget()
        g = QHBoxLayout(grid)
        g.setContentsMargins(0, 0, 0, 0)
        self.grid_cq_spin, self.grid_eta_spin = TypedSpinBox(), TypedSpinBox()
        for spin, lo, hi, value, text in ((self.grid_cq_spin, 4, 400, SIM_GRID_DEFAULT[0], "CQ cells"),
                                          (self.grid_eta_spin, 2, 100, SIM_GRID_DEFAULT[1], "ηQ cells")):
            spin.setRange(lo, hi)
            spin.setValue(value)
            spin.setSuffix(f" {text}")
            g.addWidget(spin)
        grid.setToolTip(
            "The (C_Q, η_Q) grid the distribution is summed over. Coarse is quick; raise it for "
            "a smoother spectrum. A static (or fast-MAS) central transition is fast at any size; "
            "otherwise every cell is a full simulation, timed before it starts.")
        form.addRow("Grid:", grid)
        self.layout_combo = QComboBox()
        for text, data in _LAYOUTS:
            self.layout_combo.addItem(text, data)
        form.addRow("Map layout:", self.layout_combo)
        v.addLayout(form)
        buttons = QHBoxLayout()
        self.simulate_button = QPushButton("Simulate")
        self.simulate_button.clicked.connect(self.simulate)
        self.export_button = QPushButton("Keep in Open Experiments")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.export)
        self.to_fit_button = QPushButton("Send to Fit as Starting Values")
        self.to_fit_button.setToolTip(
            "Put these values into Fit ▸ Quadrupolar (Czjzek) as the starting values of the "
            "selected site row (else the first), with this model, regime, transitions, level "
            "of theory and grid. The fit keeps the spectrum's nucleus and field.")
        self.to_fit_button.setEnabled(send_to_fit is not None)
        self.to_fit_button.clicked.connect(self.send_to_fit)
        buttons.addWidget(self.simulate_button)
        buttons.addWidget(self.export_button)
        buttons.addWidget(self.to_fit_button)
        v.addLayout(buttons)
        self.results_label = ReportLabel("")
        self.results_label.setWordWrap(True)
        self.results_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        v.addWidget(self.results_label)
        self.citation_label = _citation_label()
        v.addWidget(self.citation_label)
        v.addStretch(1)
        self._sync()

    @staticmethod
    def _spin(lo, hi, value, decimals, suffix, tip):
        spin = QDoubleSpinBox()
        spin.setRange(lo, hi)
        spin.setDecimals(decimals)
        spin.setValue(value)
        spin.setSuffix(suffix)
        spin.setToolTip(tip)
        return spin

    def _sync(self) -> None:
        self.mas_spin.setEnabled(self.regime_combo.currentData() == "mas")
        extended = self.model_combo.currentData() == "extended"
        for spin in (self.cq0_spin, self.eta0_spin):
            spin.setVisible(extended)
            label = self._form.labelForField(spin)
            if label is not None:
                label.setVisible(extended)
        if extended:
            self.d_spin.setValue(5.0)
        self.d_spin.setEnabled(not extended)
        if hasattr(self, "citation_label"):
            self.citation_label.setText(citation_html(extended))
        self._describe()

    def _mean_efg(self):
        if self.model_combo.currentData() == "extended":
            return self.cq0_spin.value(), self.eta0_spin.value()
        return 0.0, 0.0

    def _describe(self) -> None:
        if not hasattr(self, "d_spin"):
            return
        cq0, eta0 = self._mean_efg()
        try:
            text = describe_distribution(self.sigma_spin.value(), self.d_spin.value(), cq0, eta0)
        except Exception:  # noqa: BLE001 -- a hint must never block input
            text = "—"
        self.distribution_label.setText(text)

    def get_state(self) -> dict:
        return panel_state.default_state(self)

    def set_state(self, state: dict) -> None:
        panel_state.restore_default_state(self, state)
        self._sync()

    def set_from_fit(self, fitted: dict) -> None:
        """A fitted site from Fit ▸ Quadrupolar (Czjzek)."""
        self.nucleus_combo.setCurrentText(fitted["nucleus"])
        extended = fitted.get("model") == "extended" or float(fitted.get("CQ0", 0) or 0) > 0
        self.model_combo.setCurrentIndex(self.model_combo.findData(
            "extended" if extended else "czjzek"))
        if extended:
            self.cq0_spin.setValue(float(fitted.get("CQ0", 0.0)))
            self.eta0_spin.setValue(float(fitted.get("eta0", 0.0)))
        self.sigma_spin.setValue(fitted["sigma"])
        self.d_spin.setValue(fitted["d"])
        self.diso_spin.setValue(fitted["delta_iso"])
        self.fwhm_spin.setValue(fitted["fwhm"])
        self.larmor_spin.setValue(fitted["larmor_MHz"])
        self.regime_combo.setCurrentIndex(self.regime_combo.findData(
            "mas" if fitted["mas_Hz"] else "static"))
        if fitted["mas_Hz"]:
            self.mas_spin.setValue(fitted["mas_Hz"])
        self.transitions_combo.setCurrentIndex(max(0, self.transitions_combo.findData(
            fitted["transitions"])))
        self.order_combo.setCurrentIndex(max(0, self.order_combo.findData(fitted["order"])))
        lo, hi = fitted["ppm_range"]
        self.ppm_min_spin.setValue(lo)
        self.ppm_max_spin.setValue(hi)

    def settings(self) -> dict:
        nucleus = self.nucleus_combo.currentText().strip()
        spin = core_registry.find_spin(nucleus)
        if spin is None or spin < 1:
            raise ValueError(f"{nucleus or 'The nucleus'} is not a quadrupolar nucleus.")
        transitions = self.transitions_combo.currentData()
        if transitions == "ct" and float(spin).is_integer():
            transitions = "all"
        mas = self.mas_spin.value() if self.regime_combo.currentData() == "mas" else 0.0
        return dict(I=float(spin), nucleus=nucleus, larmor_Hz=self.larmor_spin.value() * 1e6,
                    mas_Hz=mas, order=self.order_combo.currentData(), transitions=transitions,
                    sigma=self.sigma_spin.value(), d=self.d_spin.value(),
                    model=self.model_combo.currentData(), cq0=self._mean_efg()[0],
                    eta0=self._mean_efg()[1],
                    delta_iso=self.diso_spin.value(), fwhm=self.fwhm_spin.value(),
                    ppm=(self.ppm_min_spin.value(), self.ppm_max_spin.value()),
                    n_cq=self.grid_cq_spin.value(), n_eta=self.grid_eta_spin.value())

    def simulate(self) -> dict | None:
        try:
            s = self.settings()
        except ValueError as exc:
            QMessageBox.warning(self, "Czjzek simulation", str(exc))
            return None
        lo, hi = sorted(s["ppm"])
        if hi - lo <= 0:
            QMessageBox.warning(self, "Czjzek simulation", "The axis range is empty.")
            return None
        x = np.linspace(hi, lo, 8192)
        if not _slow_warning(self, s, s["sigma"], s["d"], s["n_cq"], s["n_eta"], stages=1,
                             cq0=s["cq0"], eta0=s["eta0"], ppm_axis=x - s["delta_iso"]):
            return None
        progress = LibraryProgress(self, "Czjzek simulation")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            sim = cz.czjzek_spectrum(s["I"], s["sigma"], s["larmor_Hz"], x, d=s["d"],
                                     delta_iso=s["delta_iso"], fwhm=s["fwhm"], mas_Hz=s["mas_Hz"],
                                     order=s["order"], transitions=s["transitions"],
                                     n_cq=s["n_cq"], n_eta=s["n_eta"], cq0_MHz=s["cq0"],
                                     eta0=s["eta0"], progress=progress)
        except cz.SimulationCancelled:
            self.results_label.setText("Simulation stopped. A coarser grid is quicker.")
            return None
        except Exception as exc:  # noqa: BLE001 -- surfaced to the user
            QMessageBox.warning(self, "Czjzek simulation", str(exc))
            return None
        finally:
            progress.close()
            QApplication.restoreOverrideCursor()
        sim["settings"] = s
        self.result = sim
        st = sim["stats"]
        text = (f"⟨C_Q⟩ = {st['mean_CQ']:.3g} MHz, C_Q mode {st['mode_CQ']:.3g} MHz, "
                f"⟨η_Q⟩ = {st['mean_eta']:.3f}, ⟨P_Q⟩ = {st['mean_PQ']:.3g} MHz")
        if "mean_shift_ppm" in st:
            text += (f"; CT centre of gravity {s['delta_iso'] + st['mean_shift_ppm']:.3g} ppm "
                     f"(mean quadrupolar shift {st['mean_shift_ppm']:.3g} ppm)")
        self.results_label.setText(text + ".\n" + sim["log"])
        self.export_button.setEnabled(self._add_to_open is not None)
        try:
            xlabel = core_registry.NUC_label(s["nucleus"], x_unit="ppm")
        except Exception:  # noqa: BLE001 -- a label must not sink a plot
            xlabel = None
        layout = self.layout_combo.currentData()

        def draw(figure):
            look = self._style() or {}
            cz.plot_czjzek_distribution(
                {"sigma": s["sigma"], "d": s["d"], "CQ0": s["cq0"], "eta0": s["eta0"]},
                spectrum={"x": sim["x_ppm"], "y": sim["normalised_real_spec"], "xlabel": xlabel},
                layout=layout, figure=figure, cmap=look.get("map_colormap") or "viridis",
                fontsize=float(look.get("labelsize") or 12.0))
        self._draw(draw)
        return sim

    def larmor_mhz(self) -> float | None:
        """The spectrometer frequency the simulation uses (MHz), for
        Measure's Hz on the spectrum panel."""
        value = self.larmor_spin.value()
        return value if value > 0 else None

    def export(self) -> None:
        spectrum = self.make_spectrum()
        if spectrum is None or self._add_to_open is None:
            return
        s = self.result["settings"]
        name = (f"Czjzek C_Q0={s['cq0']:g}, σ={s['sigma']:g} MHz ({s['nucleus']})"
                if s["cq0"] > 0 else f"Czjzek σ={s['sigma']:g} MHz ({s['nucleus']})")
        self._add_to_open(spectrum, name)

    def send_to_fit(self) -> None:
        if self._send_to_fit is None:
            return
        cq0, eta0 = self._mean_efg()
        extended = self.model_combo.currentData() == "extended"
        values = {"d_iso": self.diso_spin.value(), "sigma": self.sigma_spin.value(),
                  "d": 5.0 if extended else self.d_spin.value(), "fwhm": self.fwhm_spin.value(),
                  "CQ0": cq0, "eta0": eta0}
        settings = {"model": self.model_combo.currentData(),
                    "regime": self.regime_combo.currentData(), "mas_Hz": self.mas_spin.value(),
                    "transitions": self.transitions_combo.currentData(),
                    "order": self.order_combo.currentData(),
                    "nucleus": self.nucleus_combo.currentText().strip(),
                    "larmor_MHz": self.larmor_spin.value(),
                    "n_cq": self.grid_cq_spin.value(), "n_eta": self.grid_eta_spin.value()}
        self._send_to_fit(values, settings)

    def make_spectrum(self):
        """The last simulation as an NMRSpectrum (None before one)."""
        sim = self.result
        if sim is None:
            return None
        s = sim["settings"]
        x = sim["x_ppm"]
        nu0 = s["larmor_Hz"]
        spectrum = NMRSpectrum(
            data=np.asarray(sim["normalised_real_spec"], dtype=float),
            axes=[NMRAxis(kind="frequency", ppm=x, hz=x * nu0 / 1e6)],
            metadata=NMRMetadata(vendor="simulated", nucleus=s["nucleus"], sfo1=nu0 / 1e6,
                                 raw={"simulation": {
                                     "model": ("Extended Czjzek" if s["cq0"] > 0 else "Czjzek"),
                                     "sigma_MHz": s["sigma"], "d": s["d"],
                                     "CQ0_MHz": s["cq0"], "eta0": s["eta0"],
                                     "delta_iso_ppm": s["delta_iso"], "fwhm_ppm": s["fwhm"],
                                     "larmor_MHz": nu0 / 1e6, "mas_Hz": s["mas_Hz"],
                                     "transitions": s["transitions"], "order": s["order"],
                                     "grid": f"{s['n_cq']} x {s['n_eta']}",
                                     "method": sim["method"]}}))
        return spectrum
