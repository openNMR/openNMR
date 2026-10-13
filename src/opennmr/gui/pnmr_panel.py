"""Simulate > pNMR (CIF): the notebook's section-5 "bond pathway" workflow,
step by step.

1. **Structure** -- load a CIF (Browse, or drop it on the panel).
2. **Sites** -- the species to observe, optionally only some of its CIF
   sites (e.g. C3, C5, C7), and the species/sites to ignore. *Initialise*
   applies these and finds every neighbour within a search radius, so the
   species listed as shift contributors in step 3 are only ones that are
   actually left.
3. **Shells** -- one row per coordination shell: its radial limits and the
   shift each neighbouring species contributes. The distance histogram on
   the Simulation plot shows where the shells fall.
4. **Spectrum** -- nuclide, field, broadening, axis, output folder, then
   *Enumerate sites*, which runs
   :func:`opennmr.notebook.pnmr_simulator.generate_spectrum_dict` -- the
   same function a notebook calls, with the same arguments (the log shows
   them, as copyable code).
5. **Plot** -- draw the spectrum, and optionally add it to Open
   experiments to compare against measured data.

*Load settings…* fills every step from a saved parameters JSON (the one
Enumerate writes beside the shift table, or *Save settings…*), ready to
Enumerate again. Enumeration runs off the GUI thread with a progress bar
and Cancel; a run whose shift table would exceed
:attr:`PnmrSimulationPanel.MAX_CONFIGURATIONS` rows asks first.

Every result is tied to the inputs that made it: change anything upstream
and the later steps are disabled until they are run again, so what is
plotted is always what the controls say.

All the physics is in :mod:`opennmr.notebook.pnmr_simulator`; this module
only collects inputs and shows results.
"""

from __future__ import annotations

import json
import os
import time
import warnings
from typing import Callable

import numpy as np
from .widgets.report_label import ReportLabel
from .widgets.field_input import MAX_FIELD_T, FieldSpinBox, proton_frequency_box
from PySide6.QtCore import QMetaObject, QObject, Qt, QThread, Signal, Slot
from PySide6.QtGui import QColor, QGuiApplication
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
    QFormLayout, QFrame, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
    QListWidget, QListWidgetItem, QMessageBox, QPlainTextEdit, QProgressBar, QPushButton,
    QSpinBox,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from ..core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
from . import panel_state
from ..notebook import pnmr_export
from ..notebook import pnmr_simulator as pnmr
from ..notebook.utils import compute_BF1_SFO1
from ..plotting.accessibility import default_series_colours
from .loader import background_load_enabled
from .pnmr_export_row import SendSitesRow

_SIGMA_TO_FWHM = 2.0 * np.sqrt(2.0 * np.log(2.0))
#: Rows of the shift table shown in the panel; the saved CSV has them all.
_MAX_TABLE_ROWS = 300
#: Distance histogram bin width, Angstrom.
_HIST_BIN = 0.05


def _fmt_distance(value: float) -> str:
    """Angstrom to 0.001 -- never rounded to significant figures, which
    would turn 3.404 into 3.4 and move a shell boundary."""
    return f"{value:.3f}".rstrip("0").rstrip(".") or "0"


def _is_cif(path: str) -> bool:
    return path.lower().endswith((".cif", ".mcif"))


def _is_json(path: str) -> bool:
    return path.lower().endswith(".json")


class PnmrEnumerator(QObject):
    """Runs one generate_spectrum_dict call, off the GUI thread.

    Same shape as :class:`opennmr.gui.loader.SpectrumReader`: a plain
    QObject moved to a QThread, so a test can also call :meth:`run`
    directly with no thread at all. It only computes -- no widgets, no
    figures -- and reports back by signal. Emits exactly one of
    :attr:`finished`, :attr:`failed` or :attr:`cancelled` per run.

    Cancelling is genuine, not "discard when done": the engine's progress
    callback raises :class:`~opennmr.notebook.pnmr_simulator.
    EnumerationCancelled` at the next site or broadening block.
    """

    #: (stage, done, total), throttled to ~20 per second.
    progress = Signal(str, int, int)
    #: dict: spectrum_dict, table, notes, parser_notes, kwargs, signature.
    finished = Signal(object)
    #: (message, estimate or None) -- estimate is set when the run was
    #: refused for exceeding max_configurations.
    failed = Signal(str, object)
    cancelled = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._cancelled = False
        self._job = None
        self._last_emit = 0.0

    def set_job(self, kwargs: dict, signature) -> None:
        """Set before invoking :meth:`run` -- held here rather than passed
        through the event loop so nothing is converted to a QVariant."""
        self._job = (dict(kwargs), signature)
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def _progress(self, stage: str, done: int, total: int) -> None:
        if self._cancelled:
            raise pnmr.EnumerationCancelled()
        now = time.monotonic()
        if done >= total or now - self._last_emit > 0.05:
            self._last_emit = now
            self.progress.emit(stage, int(done), int(total))

    @Slot()
    def run(self) -> None:
        """Never raises: every outcome is a signal, because an exception
        escaping a slot on a worker thread has nowhere useful to go."""
        if self._job is None:
            return
        kwargs, signature = self._job
        self._job = None
        notes: list[str] = []
        try:
            # Collects pymatgen's CIF-parser notes. catch_warnings is
            # process-wide, so a warning the GUI thread raises meanwhile
            # lands in the log too -- harmless, and rare.
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                spectrum_dict, table = pnmr.generate_spectrum_dict(
                    **kwargs, progress=self._progress, report=notes.append)
        except pnmr.EnumerationCancelled:
            self.cancelled.emit()
            return
        except pnmr.TooManyConfigurations as exc:
            self.failed.emit(str(exc), exc.estimate)
            return
        except Exception as exc:  # noqa: BLE001 -- reported, not raised
            self.failed.emit(str(exc) or type(exc).__name__, None)
            return
        parser_notes = sorted({str(w.message) for w in caught})
        self.finished.emit({"spectrum_dict": spectrum_dict, "table": table, "notes": notes,
                            "parser_notes": parser_notes, "kwargs": kwargs,
                            "signature": signature})


class _CifDropZone(QLabel):
    """A dashed box that takes a CIF dragged in from the file manager."""

    fileDropped = Signal(str)
    #: A saved simulation-parameters .json was dropped.
    settingsDropped = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__("Drop a CIF (or saved settings .json) here", parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setAcceptDrops(True)
        self.setMinimumHeight(44)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet("QLabel { border: 1px dashed palette(mid); border-radius: 4px; }")
        self.setToolTip("Drag a .cif file here from your file manager, or use Browse. A "
                        "*_simulation_parameters.json dropped here loads those settings.")

    @staticmethod
    def _cif_from(event) -> str | None:
        mime = event.mimeData()
        if not mime.hasUrls():
            return None
        for url in mime.urls():
            path = url.toLocalFile()
            if path and (_is_cif(path) or _is_json(path)):
                return path
        return None

    def dragEnterEvent(self, event) -> None:  # noqa: N802 -- Qt override
        if self._cif_from(event):
            event.acceptProposedAction()
        else:
            event.ignore()

    dragMoveEvent = dragEnterEvent  # noqa: N815 -- Qt override

    def dropEvent(self, event) -> None:  # noqa: N802 -- Qt override
        path = self._cif_from(event)
        if path:
            event.acceptProposedAction()
            (self.settingsDropped if _is_json(path) else self.fileDropped).emit(path)
        else:
            event.ignore()


def _checked(list_widget: QListWidget) -> list[str]:
    return [list_widget.item(i).data(Qt.ItemDataRole.UserRole)
            for i in range(list_widget.count())
            if list_widget.item(i).checkState() == Qt.CheckState.Checked]


def _fill_checklist(list_widget: QListWidget, entries, checked: set[str]) -> None:
    """entries: (key, display text) pairs."""
    list_widget.blockSignals(True)
    list_widget.clear()
    for key, text in entries:
        item = QListWidgetItem(text)
        item.setData(Qt.ItemDataRole.UserRole, key)
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        item.setCheckState(Qt.CheckState.Checked if key in checked else Qt.CheckState.Unchecked)
        list_widget.addItem(item)
    list_widget.blockSignals(False)
    rows = max(2, min(list_widget.count(), 6))
    list_widget.setFixedHeight(list_widget.sizeHintForRow(0) * rows + 6
                               if list_widget.count() else 40)


class PnmrSimulationPanel(QWidget):
    """The pNMR (CIF) section of the Simulate tile.

    ``simulation_canvas`` receives the distance histogram;
    ``show_simulation_view`` raises it; ``plot_spectrum(spectrum)`` draws
    the simulated spectrum; ``add_to_open(spectrum, label)`` adds it to
    Open experiments.
    """

    SHELL_FIXED_COLUMNS = ("Shell", "r min (Å)", "r max (Å)")
    #: Shift-table rows above which Enumerate asks before running. At ~10
    #: columns that is of order 1 GB while the table is being built.
    MAX_CONFIGURATIONS = 5_000_000

    def __init__(self, simulation_canvas, show_simulation_view: Callable[[], None],
                 plot_spectrum: Callable[[NMRSpectrum], None],
                 add_to_open: Callable[[NMRSpectrum, str], None],
                 export_sites: Callable | None = None, parent=None) -> None:
        super().__init__(parent)
        #: ``export_sites(site_dicts, notes, nucleus, larmor_mhz, destination,
        #: append, source)`` -- puts sites into CSA, Quad & J-coupling.
        self._export_sites = export_sites
        self._canvas = simulation_canvas
        self._show_simulation_view = show_simulation_view
        self._plot_spectrum = plot_spectrum
        self._add_to_open = add_to_open

        self.cif_path: str | None = None
        self._structure = None
        self._site_data = None
        self._contributors: list[str] = []
        self._distances = None           # neighbour_distance_table at Initialise
        self._n_interest_sites = 0
        self._init_signature = None      # step-2 choices the distances were made with
        self._result = None              # (spectrum_dict, shift_table)
        self._result_signature = None    # every input the result was made with
        self._spectrum: NMRSpectrum | None = None
        self._added_signature = None     # result already added to Open experiments
        self._init_text = ""
        self._thread: QThread | None = None      # created on the first background run
        self._worker: PnmrEnumerator | None = None
        self._running = False

        layout = QVBoxLayout(self)
        layout.addWidget(self._build_structure_box())
        layout.addWidget(self._build_sites_box())
        layout.addWidget(self._build_shells_box())
        layout.addWidget(self._build_spectrum_box())
        layout.addWidget(self._build_plot_box())
        layout.addWidget(self._build_export_box())
        layout.addWidget(self._build_log_box())
        self._refresh_state()

    # ------------------------------------------------------------------
    # 1. Structure
    # ------------------------------------------------------------------
    def _build_structure_box(self) -> QGroupBox:
        box = QGroupBox("1. Structure")
        v = QVBoxLayout(box)
        row = QHBoxLayout()
        self.cif_path_edit = QLineEdit()
        self.cif_path_edit.setReadOnly(True)
        self.cif_path_edit.setPlaceholderText("No CIF loaded")
        row.addWidget(self.cif_path_edit, 1)
        self.cif_browse_button = QPushButton("Browse…")
        self.cif_browse_button.setToolTip("Choose a CIF file to read.")
        self.cif_browse_button.clicked.connect(self._on_browse_cif)
        row.addWidget(self.cif_browse_button)
        v.addLayout(row)
        self.cif_drop = _CifDropZone()
        self.cif_drop.fileDropped.connect(self.load_cif)
        self.cif_drop.settingsDropped.connect(self.load_settings)
        v.addWidget(self.cif_drop)
        settings_row = QHBoxLayout()
        self.load_settings_button = QPushButton("Load settings…")
        self.load_settings_button.setToolTip(
            "Fill every step from a saved *_simulation_parameters.json -- the file "
            "Enumerate writes beside the shift table -- ready to Enumerate again.")
        self.load_settings_button.clicked.connect(self._on_load_settings)
        settings_row.addWidget(self.load_settings_button)
        self.save_settings_button = QPushButton("Save settings…")
        self.save_settings_button.setToolTip(
            "Save the current set-up as a parameters JSON without running it -- to "
            "come back to, or to run from a notebook.")
        self.save_settings_button.clicked.connect(self._on_save_settings)
        settings_row.addWidget(self.save_settings_button)
        settings_row.addStretch(1)
        v.addLayout(settings_row)
        self.structure_label = QLabel("")
        self.structure_label.setWordWrap(True)
        self.structure_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        v.addWidget(self.structure_label)
        return box

    def _on_browse_cif(self) -> None:
        start = os.path.dirname(self.cif_path) if self.cif_path else ""
        path, _ = QFileDialog.getOpenFileName(self, "Open CIF", start,
                                              "Crystallographic files (*.cif *.mcif);;All files (*)")
        if path:
            self.load_cif(path)

    def load_cif(self, path: str) -> bool:
        """Read ``path`` and fill step 2 from it. Returns success."""
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")   # pymatgen's CIF-parsing chatter
                structure, site_data = pnmr.read_cif_with_occupancies(path)
        except ImportError:
            QMessageBox.warning(self, "pNMR simulation",
                                "Reading a CIF needs pymatgen, which is not installed.\n\n"
                                "Install it with:  pip install pymatgen")
            return False
        except Exception as exc:  # noqa: BLE001 -- shown to the user
            QMessageBox.warning(self, "pNMR simulation", f"Could not read {os.path.basename(path)}:\n{exc}")
            self._log(f"Could not read {path}: {exc}")
            return False

        # Native separators: Qt gives a dropped file as C:/Users/..., the
        # file dialog and everything else as C:\\Users\\... on Windows.
        path = os.path.normpath(path)
        self.cif_path = path
        self._structure, self._site_data = structure, site_data
        self.cif_path_edit.setText(path)
        self.cif_path_edit.setToolTip(path)
        self.cif_drop.setText(f"Loaded {os.path.basename(path)} -- drop another CIF to replace it")

        labels = pnmr.site_label_table(site_data)
        species = pnmr.structure_species(site_data)
        self.structure_label.setText(
            f"{structure.composition.reduced_formula}: {len(site_data)} sites in the cell, "
            f"{len(labels)} CIF site labels, species {', '.join(species)}.")
        self._labels = labels

        self.species_combo.blockSignals(True)
        self.species_combo.clear()
        self.species_combo.addItems(species)
        self.species_combo.blockSignals(False)
        _fill_checklist(self.ignore_species_list, [(s, s) for s in species], set())
        _fill_checklist(self.ignore_sites_list,
                        [(row.Label, f"{row.Label}  ({row.Occupancy})")
                         for row in labels.itertuples()], set())
        self._distances = None
        self._init_signature = None
        self._on_species_changed()
        names = list(labels["Label"])
        shown = ", ".join(names[:12]) + (f", … ({len(names)} labels in all)" if len(names) > 12 else "")
        self._log(f"Read {path}: {len(site_data)} sites, labels {shown}.")
        return True

    # ------------------------------------------------------------------
    # 2. Sites
    # ------------------------------------------------------------------
    def _build_sites_box(self) -> QGroupBox:
        box = self._sites_box = QGroupBox("2. Sites")
        form = QFormLayout(box)
        self.species_combo = QComboBox()
        self.species_combo.setToolTip(
            "The species whose NMR spectrum is simulated (atom_of_interest), as the CIF "
            "writes it -- e.g. 'Na+' in a CIF with oxidation states, 'Na' without.")
        self.species_combo.currentIndexChanged.connect(lambda _i: self._on_species_changed())
        form.addRow("Observe:", self.species_combo)

        self.sites_list = QListWidget()
        self.sites_list.setToolTip(
            "Which of that species' crystallographic sites to include. All ticked = every "
            "site (sites_of_interest=None); tick a subset for e.g. only C3, C5 and C7.")
        self.sites_list.itemChanged.connect(lambda _i: self._refresh_state())
        form.addRow("Sites:", self.sites_list)

        self.ignore_species_list = QListWidget()
        self.ignore_species_list.setToolTip(
            "Neighbour species to leave out of the shell search entirely (ignore_species), "
            "e.g. the anion sublattice. A site shared with an ignored species is left out "
            "whole, so its other occupants go too.")
        self.ignore_species_list.itemChanged.connect(lambda _i: self._refresh_state())
        form.addRow("Ignore species:", self.ignore_species_list)

        self.ignore_sites_list = QListWidget()
        self.ignore_sites_list.setToolTip(
            "Individual CIF sites to leave out of the shell search (ignore_sites).")
        self.ignore_sites_list.itemChanged.connect(lambda _i: self._refresh_state())
        form.addRow("Ignore sites:", self.ignore_sites_list)

        self.max_r_spin = QDoubleSpinBox()
        self.max_r_spin.setRange(1.0, 20.0)
        self.max_r_spin.setDecimals(2)
        self.max_r_spin.setValue(6.0)
        self.max_r_spin.setSuffix(" Å")
        self.max_r_spin.setToolTip(
            "How far out to list neighbours for the distance histogram and shell "
            "suggestions. Does not limit the simulation: shells are searched to their "
            "own r max.")
        self.max_r_spin.valueChanged.connect(lambda _v: self._refresh_state())
        form.addRow("Search radius:", self.max_r_spin)

        self.initialise_button = QPushButton("Initialise structure")
        self.initialise_button.setToolTip(
            "Apply the choices above: find every neighbour of the chosen sites, list the "
            "species that can contribute a shift, and draw the distance histogram.")
        self.initialise_button.clicked.connect(self.initialise)
        form.addRow(self.initialise_button)
        self.init_label = QLabel("")
        self.init_label.setWordWrap(True)
        self.init_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        form.addRow(self.init_label)
        return box

    def _on_species_changed(self) -> None:
        species = self.species_combo.currentText()
        if self._site_data is None or not species:
            return
        entries = [(row.Label, f"{row.Label}  ({row.Occupancy}; ×{row.Multiplicity})")
                   for row in self._labels.itertuples() if species in row.Species]
        _fill_checklist(self.sites_list, entries, {key for key, _ in entries})
        current = self.nuclide_combo.currentText()
        self.nuclide_combo.blockSignals(True)
        self.nuclide_combo.clear()
        self.nuclide_combo.addItems(pnmr.candidate_nuclides(species))
        self.nuclide_combo.blockSignals(False)
        if not self.nuclide_combo.count() and current:
            self.nuclide_combo.setEditText(current)
        self._refresh_state()

    def atom_of_interest(self) -> str:
        return self.species_combo.currentText()

    def sites_of_interest(self) -> list[str] | None:
        """None when every site is ticked -- the same simulation as naming
        them all, and what a notebook would write."""
        ticked = _checked(self.sites_list)
        return None if len(ticked) == self.sites_list.count() else ticked

    def ignore_species(self) -> list[str]:
        return _checked(self.ignore_species_list)

    def ignore_sites(self) -> list[str]:
        return _checked(self.ignore_sites_list)

    def _sites_signature(self):
        return (self.cif_path, self.atom_of_interest(), tuple(_checked(self.sites_list)),
                tuple(self.ignore_species()), tuple(self.ignore_sites()),
                round(self.max_r_spin.value(), 4))

    def initialise(self) -> bool:
        if self._site_data is None:
            return False
        species = self.atom_of_interest()
        if not _checked(self.sites_list):
            QMessageBox.information(self, "pNMR simulation", "Tick at least one site to observe.")
            return False
        try:
            pnmr.mark_sites_of_interest(self._site_data, species, self.sites_of_interest())
            self._distances = pnmr.neighbour_distance_table(
                self._structure, self._site_data, species, max_r=self.max_r_spin.value(),
                ignore_species=self.ignore_species(), ignore_sites=self.ignore_sites())
        except ValueError as exc:
            QMessageBox.warning(self, "pNMR simulation", str(exc))
            return False
        key = f"contains {species}"
        self._n_interest_sites = sum(1 for s in self._site_data if s.get(key))
        self._contributors = pnmr.contributing_species(
            self._site_data, self.ignore_species(), self.ignore_sites())
        self._init_signature = self._sites_signature()

        everything = pnmr.structure_species(self._site_data)
        dropped = [s for s in everything
                   if s not in self._contributors and s not in self.ignore_species()]
        chosen = self.sites_of_interest()
        lines = [f"{self._n_interest_sites} {species} site(s) in the cell"
                 + (f" (labels {', '.join(chosen)})." if chosen else " (all of its sites)."),
                 f"Shift contributors: {', '.join(self._contributors) or 'none'}."]
        if dropped:
            lines.append(f"Also left out, sharing a site with an ignored species or on an "
                         f"ignored site: {', '.join(dropped)}.")
        if self._distances.empty:
            lines.append(f"No neighbours within {self.max_r_spin.value():g} Å -- widen the "
                         f"search radius.")
        self._init_text = "\n".join(lines)
        self.init_label.setText(self._init_text)
        self._rebuild_shell_columns()
        if self.shell_table.rowCount() == 0:
            self.add_shell()
        self._log(f"Initialised: observe {species}, sites {chosen or 'all'}, ignore species "
                  f"{self.ignore_species() or 'none'}, ignore sites {self.ignore_sites() or 'none'}; "
                  f"{len(self._distances)} neighbour pairs within {self.max_r_spin.value():g} Å.")
        self._refresh_state()
        self._draw_histogram()
        return True

    # ------------------------------------------------------------------
    # 3. Shells
    # ------------------------------------------------------------------
    def _build_shells_box(self) -> QGroupBox:
        box = self._shells_box = QGroupBox("3. Shells and shift contributions (ppm)")
        v = QVBoxLayout(box)
        hint = QLabel("One row per shell. A neighbour at distance r counts in a shell when "
                      "r min < r ≤ r max. Each species column is the shift (ppm) one such "
                      "neighbour contributes; greyed cells are species with no neighbour "
                      "in that shell.")
        hint.setWordWrap(True)
        v.addWidget(hint)
        self.shell_table = QTableWidget(0, len(self.SHELL_FIXED_COLUMNS))
        self.shell_table.setHorizontalHeaderLabels(self.SHELL_FIXED_COLUMNS)
        self.shell_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.shell_table.verticalHeader().setVisible(False)
        self.shell_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.shell_table.setMinimumHeight(90)
        self.shell_table.setMaximumHeight(170)
        self.shell_table.itemChanged.connect(self._on_shell_table_changed)
        v.addWidget(self.shell_table)
        row = QHBoxLayout()
        self.add_shell_button = QPushButton("Add shell")
        self.add_shell_button.setToolTip(
            "Add the next shell out, its limits suggested from the gaps in the distance "
            "histogram -- check them against the plot.")
        self.add_shell_button.clicked.connect(self.add_shell)
        row.addWidget(self.add_shell_button)
        self.remove_shell_button = QPushButton("Remove shell")
        self.remove_shell_button.setToolTip("Remove the selected shell (or the last one).")
        self.remove_shell_button.clicked.connect(self.remove_shell)
        row.addWidget(self.remove_shell_button)
        row.addStretch(1)
        v.addLayout(row)
        self.shell_error_label = QLabel("")
        self.shell_error_label.setWordWrap(True)
        self.shell_error_label.setStyleSheet("color: #C0392B;")
        v.addWidget(self.shell_error_label)
        self.shell_summary_label = ReportLabel("")
        self.shell_summary_label.setWordWrap(True)
        self.shell_summary_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        v.addWidget(self.shell_summary_label)
        return box

    def _species_columns(self) -> list[str]:
        return [self.shell_table.horizontalHeaderItem(c).text()
                for c in range(len(self.SHELL_FIXED_COLUMNS), self.shell_table.columnCount())]

    def _rebuild_shell_columns(self) -> None:
        """One column per contributor, keeping values already typed for
        species that are still contributors."""
        old_species = self._species_columns()
        old = []
        for r in range(self.shell_table.rowCount()):
            fixed = [self._cell_text(r, c) for c in range(len(self.SHELL_FIXED_COLUMNS))]
            values = {sp: self._cell_text(r, len(self.SHELL_FIXED_COLUMNS) + i)
                      for i, sp in enumerate(old_species)}
            old.append((fixed, values))
        self.shell_table.blockSignals(True)
        self.shell_table.setColumnCount(len(self.SHELL_FIXED_COLUMNS) + len(self._contributors))
        self.shell_table.setHorizontalHeaderLabels(list(self.SHELL_FIXED_COLUMNS) + self._contributors)
        self.shell_table.setRowCount(0)
        self.shell_table.blockSignals(False)
        for fixed, values in old:
            self._append_shell_row(fixed[0], fixed[1], fixed[2], values)

    def _cell_text(self, row: int, col: int) -> str:
        item = self.shell_table.item(row, col)
        return item.text().strip() if item is not None else ""

    def _append_shell_row(self, name, r_min, r_max, values=None) -> None:
        values = values or {}
        self.shell_table.blockSignals(True)
        row = self.shell_table.rowCount()
        self.shell_table.insertRow(row)
        for col, text in enumerate((name, r_min, r_max)):
            self.shell_table.setItem(row, col, QTableWidgetItem(str(text)))
        for i, sp in enumerate(self._species_columns()):
            self.shell_table.setItem(row, len(self.SHELL_FIXED_COLUMNS) + i,
                                     QTableWidgetItem(str(values.get(sp, "") or "0")))
        self.shell_table.blockSignals(False)
        self._on_shell_table_changed()

    @staticmethod
    def _default_shell_name(index: int) -> str:
        return {0: "NN", 1: "NNN"}.get(index, f"N{index + 1}N")

    def add_shell(self) -> None:
        bins = self._parsed_bins(strict=False)
        last = max((b[2] for b in bins), default=0.0)
        suggestion = None
        if self._distances is not None and len(self._distances):
            for lo, hi in pnmr.suggest_shell_bins(self._distances["Distance"],
                                                  max_r=self.max_r_spin.value()):
                if hi > last + 0.01:
                    suggestion = (max(lo, last), hi)
                    break
        if suggestion is None:
            suggestion = (last, last + 1.0)
        names = {b[0] for b in bins}
        index = self.shell_table.rowCount()
        name = self._default_shell_name(index)
        while name in names:
            index += 1
            name = self._default_shell_name(index)
        self._append_shell_row(name, _fmt_distance(suggestion[0]), _fmt_distance(suggestion[1]))

    def remove_shell(self) -> None:
        rows = sorted({i.row() for i in self.shell_table.selectedIndexes()}, reverse=True)
        if not rows and self.shell_table.rowCount():
            rows = [self.shell_table.rowCount() - 1]
        for r in rows:
            self.shell_table.removeRow(r)
        self._on_shell_table_changed()

    def _parsed_bins(self, strict: bool = True):
        bins = []
        for r in range(self.shell_table.rowCount()):
            name = self._cell_text(r, 0)
            try:
                lo, hi = float(self._cell_text(r, 1)), float(self._cell_text(r, 2))
            except ValueError:
                if strict:
                    raise ValueError(f"Shell '{name or r + 1}': r min and r max must be numbers.")
                continue
            bins.append((name, lo, hi))
        return bins

    def shell_bins(self) -> list[tuple[str, float, float]]:
        """The shells as generate_spectrum_dict takes them. Raises
        ValueError for anything it would refuse or misread."""
        bins = self._parsed_bins(strict=True)
        pnmr.validate_shell_bins(bins)
        return bins

    def shift_dict(self) -> dict[str, dict[str, float]]:
        out = {}
        species = self._species_columns()
        for r in range(self.shell_table.rowCount()):
            name = self._cell_text(r, 0)
            out[name] = {}
            for i, sp in enumerate(species):
                text = self._cell_text(r, len(self.SHELL_FIXED_COLUMNS) + i) or "0"
                try:
                    out[name][sp] = float(text)
                except ValueError:
                    raise ValueError(f"Shell '{name}', {sp}: '{text}' is not a number (ppm).")
        return out

    def _on_shell_table_changed(self, _item=None) -> None:
        error = ""
        try:
            bins = self.shell_bins()
            self.shift_dict()
        except ValueError as exc:
            error, bins = str(exc), None
        self.shell_error_label.setText(error)
        self._update_shell_composition(bins)
        self._refresh_state()
        self._draw_histogram()

    def _update_shell_composition(self, bins) -> None:
        """Grey out species absent from each shell, and summarise what each
        shell holds per site of interest."""
        if not bins or self._distances is None:
            self.shell_summary_label.setText("")
            return
        comp = pnmr.shell_composition(self._distances, bins, n_sites=self._n_interest_sites)
        species = self._species_columns()
        max_r = self.max_r_spin.value()
        lines = []
        self.shell_table.blockSignals(True)
        for r, (name, lo, hi) in enumerate(bins):
            rows = comp[comp.Shell == name]
            present = {row.Species: row.Mean_count for row in rows.itertuples() if row.Species}
            for i, sp in enumerate(species):
                item = self.shell_table.item(r, len(self.SHELL_FIXED_COLUMNS) + i)
                if item is None:
                    continue
                if sp in present:
                    item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable | Qt.ItemFlag.ItemIsEnabled)
                    item.setData(Qt.ItemDataRole.BackgroundRole, None)
                    item.setToolTip(f"{present[sp]:.3g} {sp} per site in {name}")
                else:
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    item.setData(Qt.ItemDataRole.BackgroundRole, QColor(128, 128, 128, 60))
                    item.setToolTip(f"No {sp} between {lo:g} and {hi:g} Å of the chosen sites"
                                    f" -- this value is not used.")
            n = rows["Neighbours"].iloc[0] if len(rows) else 0.0
            parts = ", ".join(f"{sp} {c:.3g}" for sp, c in present.items() if sp != "Va")
            vac = present.get("Va")
            line = f"{name} ({lo:g}–{hi:g} Å): {n:.3g} neighbours/site" + (f" — {parts}" if parts else "")
            if vac:
                line += f", vacancy {vac:.3g}"
            if hi > max_r:
                line += f"  [beyond the {max_r:g} Å search radius: counts shown are incomplete]"
            lines.append(line)
        self.shell_table.blockSignals(False)
        self.shell_summary_label.setText("\n".join(lines))

    def _draw_histogram(self) -> None:
        """Neighbours per site of interest against distance, stacked by
        species (occupancy-weighted), with the shells shaded."""
        if self._distances is None:
            return
        self._show_simulation_view()
        # Shown as an external plot on the Simulation canvas: Plot Style,
        # right-click, zoom and Measure, and each species a series whose
        # name and colour Plot Style can change (explicit user request,
        # 2026-10-01 -- only the final spectrum had them).
        if hasattr(self._canvas, "show_external_plot"):
            self._canvas.show_external_plot(self._draw_histogram_into, kind="curve")
        else:
            self._draw_histogram_into(self._canvas.figure)
            self._canvas.draw()

    def _draw_histogram_into(self, fig) -> None:
        fig.clear()
        style = self._canvas.style_settings() if hasattr(self._canvas, "style_settings") else {}
        fontsize = float(style.get("labelsize") or 10.0)
        ax = fig.add_subplot(111)
        table = self._distances
        n_sites = max(self._n_interest_sites, 1)
        max_r = self.max_r_spin.value()
        edges = np.arange(0.0, max_r + _HIST_BIN, _HIST_BIN)
        species = [s for s in pnmr.structure_species(self._site_data, include_vacancies=True)
                   if any(s in occ for occ in table["Occupancy"])]
        colours = default_series_colours(max(len(species), 1))
        dist = table["Distance"].to_numpy(dtype=float)
        data, weights = [], []
        for sp in species:
            w = np.array([occ.get(sp, 0.0) for occ in table["Occupancy"]]) / n_sites
            data.append(dist)
            weights.append(w)
        if data:
            ax.hist(data, bins=edges, weights=weights, stacked=True, label=species,
                    color=colours[:len(species)], edgecolor="none")
        try:
            bins = self.shell_bins()
        except ValueError:
            bins = []
        top = ax.get_ylim()[1] or 1.0
        for k, (name, lo, hi) in enumerate(bins):
            ax.axvspan(lo, hi, color="0.5", alpha=0.10 if k % 2 == 0 else 0.18, lw=0)
            ax.text(0.5 * (lo + min(hi, max_r)), top * 0.97, name, ha="center", va="top",
                    fontsize=fontsize * 0.8)
        ax.set_xlim(0.0, max_r)
        ax.set_xlabel(f"Distance from {self.atom_of_interest()} (Å)")
        ax.set_ylabel(f"Neighbours per site (per {_HIST_BIN:g} Å)")
        ax.set_title(f"{os.path.basename(self.cif_path or '')}: neighbours of "
                     f"{self.atom_of_interest()}", fontsize=fontsize)
        if species:
            ax.legend(frameon=False, fontsize=fontsize * 0.9)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        fig.tight_layout()

    # ------------------------------------------------------------------
    # 4. Spectrum
    # ------------------------------------------------------------------
    def _build_spectrum_box(self) -> QGroupBox:
        box = QGroupBox("4. Spectrum")
        form = QFormLayout(box)
        self.nuclide_combo = QComboBox()
        self.nuclide_combo.setEditable(True)
        self.nuclide_combo.setToolTip("NMR isotope, for referencing (BF1) and the axis label. "
                                      "Type any isotope in the table, e.g. 6Li.")
        self.nuclide_combo.currentTextChanged.connect(lambda _t: self._refresh_state())
        form.addRow("Nuclide:", self.nuclide_combo)

        self.field_spin = FieldSpinBox()
        self.field_spin.setRange(0.01, MAX_FIELD_T)
        self.field_spin.setDecimals(6)
        self.field_spin.setValue(11.7)
        self.field_spin.setSuffix(" T")
        self.field_spin.setToolTip("Field of the simulated spectrometer (B0_tesla) -- or type "
                                   "its ¹H frequency beside it.")
        self.field_spin.valueChanged.connect(lambda _v: self._refresh_state())
        # Not saved separately: it is the field, in other units.
        self._field_proton_spin = proton_frequency_box(self.field_spin, parent=self)
        field_row = QHBoxLayout()
        field_row.addWidget(self.field_spin, 1)
        field_row.addWidget(QLabel("or"))
        field_row.addWidget(self._field_proton_spin, 1)
        form.addRow("Field:", field_row)

        self.broadening_spin = QDoubleSpinBox()
        self.broadening_spin.setRange(0.001, 1.0e5)
        self.broadening_spin.setDecimals(3)
        self.broadening_spin.setValue(1.0)
        self.broadening_spin.setSuffix(" ppm")
        self.broadening_spin.setToolTip(
            "Gaussian broadening of each environment, as sigma (standard deviation) -- "
            "not FWHM. FWHM = 2.355 × sigma.")
        self.broadening_spin.valueChanged.connect(lambda _v: self._refresh_state())
        form.addRow("Broadening σ:", self.broadening_spin)

        self.auto_axis_check = QCheckBox("Auto (shift range ± 5σ)")
        self.auto_axis_check.setChecked(True)
        self.auto_axis_check.toggled.connect(lambda _on: self._refresh_state())
        form.addRow("Axis:", self.auto_axis_check)
        self.ppm_min_spin = QDoubleSpinBox()
        self.ppm_max_spin = QDoubleSpinBox()
        for spin, value in ((self.ppm_min_spin, -1000.0), (self.ppm_max_spin, 9000.0)):
            spin.setRange(-1.0e6, 1.0e6)
            spin.setDecimals(1)
            spin.setValue(value)
            spin.setSuffix(" ppm")
            spin.valueChanged.connect(lambda _v: self._refresh_state())
        form.addRow("Axis min:", self.ppm_min_spin)
        form.addRow("Axis max:", self.ppm_max_spin)
        self._axis_form = form
        self.points_spin = QSpinBox()
        self.points_spin.setRange(256, 1 << 20)
        self.points_spin.setValue(8192)
        self.points_spin.valueChanged.connect(lambda _v: self._refresh_state())
        form.addRow("Axis points:", self.points_spin)

        self.weight_check = QCheckBox("Weight sites by occupancy")
        self.weight_check.setChecked(True)
        self.weight_check.setToolTip(
            "Each site contributes in proportion to its occupancy by the observed species "
            "(a 30%-occupied site gives 0.3 of the signal of a full one). Unticked: every "
            "site counts equally. Identical when all chosen sites have the same occupancy.")
        self.weight_check.toggled.connect(lambda _on: self._refresh_state())
        form.addRow("", self.weight_check)

        out_row = QHBoxLayout()
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setPlaceholderText("<CIF folder>/pnmr_simulation")
        self.output_dir_edit.setToolTip("Where the shift table (CSV) and the parameters used "
                                        "(JSON) are written (output_dir).")
        self.output_dir_edit.textChanged.connect(lambda _t: self._refresh_state())
        out_row.addWidget(self.output_dir_edit, 1)
        self.output_browse_button = QPushButton("…")
        self.output_browse_button.setToolTip("Choose the output folder.")
        self.output_browse_button.clicked.connect(self._on_browse_output)
        out_row.addWidget(self.output_browse_button)
        form.addRow("Output folder:", out_row)
        self.save_check = QCheckBox("Save shift table and parameters there")
        self.save_check.setChecked(True)
        self.save_check.toggled.connect(lambda _on: self._refresh_state())
        form.addRow("", self.save_check)

        self._spectrum_inputs = (
            self.nuclide_combo, self.field_spin, self.broadening_spin, self.auto_axis_check,
            self.points_spin, self.weight_check, self.output_dir_edit, self.output_browse_button,
            self.save_check, self.ppm_min_spin, self.ppm_max_spin)
        self.enumerate_button = QPushButton("Enumerate sites")
        self.enumerate_button.setToolTip(
            "Enumerate every local environment of the chosen sites and its probability, and "
            "build the broadened spectrum (generate_spectrum_dict).")
        self.enumerate_button.clicked.connect(self._on_enumerate_clicked)
        form.addRow(self.enumerate_button)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        form.addRow(self.progress_bar)
        self.progress_label = QLabel("")
        self.progress_label.setVisible(False)
        form.addRow(self.progress_label)
        self.results_label = ReportLabel("")
        self.results_label.setWordWrap(True)
        self.results_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        form.addRow(self.results_label)
        self.shift_table_view = QTableWidget(0, 0)
        self.shift_table_view.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.shift_table_view.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.shift_table_view.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        #: The shift table's index label of each row shown (most probable first).
        self._shown_index: list = []
        self.shift_table_view.verticalHeader().setVisible(False)
        self.shift_table_view.setMinimumHeight(140)
        self.shift_table_view.setVisible(False)
        form.addRow(self.shift_table_view)
        self.save_table_button = QPushButton("Save table as…")
        self.save_table_button.setToolTip("Write the full shift table to a CSV of your choice, "
                                          "with the parameters beside it as JSON.")
        self.save_table_button.clicked.connect(self._on_save_table_as)
        form.addRow(self.save_table_button)
        return box

    def _on_browse_output(self) -> None:
        start = self.output_dir_edit.text().strip() or (os.path.dirname(self.cif_path)
                                                        if self.cif_path else "")
        path = QFileDialog.getExistingDirectory(self, "Output folder", start)
        if path:
            self.output_dir_edit.setText(path)

    def output_dir(self) -> str | None:
        return self.output_dir_edit.text().strip() or None

    def xlims(self) -> list[float] | None:
        if self.auto_axis_check.isChecked():
            return None
        return [self.ppm_max_spin.value(), self.ppm_min_spin.value()]

    def generate_kwargs(self) -> dict:
        """Exactly the arguments Enumerate passes to generate_spectrum_dict.
        Raises ValueError if the shells or shifts are not valid."""
        return dict(
            cif_filename=self.cif_path,
            atom_of_interest=self.atom_of_interest(),
            shell_bins=self.shell_bins(),
            shift_dict=self.shift_dict(),
            ignore_species=self.ignore_species(),
            ignore_sites=self.ignore_sites(),
            sites_of_interest=self.sites_of_interest(),
            nuclide=self.nuclide_combo.currentText().strip(),
            B0_tesla=self.field_spin.value(),
            broadening=self.broadening_spin.value(),
            n_points=self.points_spin.value(),
            xlims=self.xlims(),
            weight_by_occupancy=self.weight_check.isChecked(),
            output_dir=self.output_dir(),
            save_shift_table=self.save_check.isChecked(),
        )

    #: generate_kwargs entries that define the structural model a fit uses
    #: (Fit ▸ pNMR); the others only shape a simulated spectrum.
    FIT_MODEL_KEYS = ("cif_filename", "atom_of_interest", "shell_bins", "shift_dict",
                      "ignore_species", "sites_of_interest", "ignore_sites", "weight_by_occupancy")

    def fit_model_kwargs(self) -> dict:
        """The structural model for Fit ▸ pNMR (CIF bond pathways): the
        same CIF, sites, ignores, shells and shifts Enumerate would use.
        Raises ValueError saying what is still missing."""
        if self._site_data is None or not self.cif_path:
            raise ValueError("no CIF loaded -- load one in Simulate ▸ pNMR (CIF), step 1.")
        if self._init_signature is None or self._init_signature != self._sites_signature():
            raise ValueError("press Initialise structure in Simulate ▸ pNMR (CIF), step 2.")
        if self.shell_table.rowCount() == 0:
            raise ValueError("no shells yet -- add them in Simulate ▸ pNMR (CIF), step 3.")
        kwargs = self.generate_kwargs()
        if not any(v for shell in kwargs["shift_dict"].values() for v in shell.values()):
            raise ValueError("every bond-pathway shift is 0 -- give the shells their shifts in "
                             "Simulate ▸ pNMR (CIF), step 3.")
        return {k: kwargs[k] for k in self.FIT_MODEL_KEYS}

    def model_summary(self) -> str:
        """A line or two describing fit_model_kwargs(), for Fit ▸ pNMR."""
        m = self.fit_model_kwargs()
        sites = ", ".join(m["sites_of_interest"]) if m["sites_of_interest"] else "all sites"
        shells = "; ".join(
            f"{name} {_fmt_distance(lo)}–{_fmt_distance(hi)} Å: "
            + (", ".join(f"{sp} {v:+g}" for sp, v in m["shift_dict"].get(name, {}).items() if v)
               or "no shift")
            for name, lo, hi in m["shell_bins"])
        ignored = m["ignore_species"] + m["ignore_sites"]
        return (f"<b>{os.path.basename(m['cif_filename'])}</b>, observing "
                f"{m['atom_of_interest']} ({sites})"
                + (f", ignoring {', '.join(ignored)}" if ignored else "")
                + f". Shells (ppm per neighbour): {shells}.")

    def _input_signature(self):
        try:
            kwargs = self.generate_kwargs()
        except ValueError:
            return None
        return json.dumps(kwargs, sort_keys=True, default=str)

    def enumerate_sites(self, limit: int | None = -1) -> bool:
        """Start an enumeration with the current inputs. In the background
        (the default) this returns once the run has started; with
        background work switched off (OPENNMR_BACKGROUND_LOAD=0, as under
        test) it returns after the run. Returns False if nothing started.

        limit: the row limit to refuse above; -1 means MAX_CONFIGURATIONS,
        None means no limit (after the user has agreed to a big run)."""
        if self._running:
            return False
        if self._init_signature is None or self._init_signature != self._sites_signature():
            QMessageBox.information(self, "pNMR simulation",
                                    "Initialise the structure with the current site choices first.")
            return False
        try:
            kwargs = self.generate_kwargs()
            compute_BF1_SFO1(kwargs["nuclide"], kwargs["B0_tesla"])
        except (ValueError, KeyError) as exc:
            QMessageBox.warning(self, "pNMR simulation", str(exc).strip("'\""))
            return False
        kwargs["max_configurations"] = self.MAX_CONFIGURATIONS if limit == -1 else limit
        signature = self._input_signature()

        worker = self._ensure_worker()
        worker.set_job(kwargs, signature)
        self._set_running(True)
        self._log("Enumerating…")
        if self._thread is None:
            worker.run()                  # synchronous: signals arrive before this returns
        else:
            QMetaObject.invokeMethod(worker, "run", Qt.ConnectionType.QueuedConnection)
        return True

    def _ensure_worker(self) -> PnmrEnumerator:
        if self._worker is None:
            self._worker = PnmrEnumerator()
            self._worker.progress.connect(self._on_enumeration_progress)
            self._worker.finished.connect(self._on_enumeration_finished)
            self._worker.failed.connect(self._on_enumeration_failed)
            self._worker.cancelled.connect(self._on_enumeration_cancelled)
            if background_load_enabled():
                self._thread = QThread()
                self._worker.moveToThread(self._thread)
                self._thread.start()
        return self._worker

    def cancel_enumeration(self) -> None:
        if self._running and self._worker is not None:
            self._worker.cancel()
            self.progress_label.setText("Cancelling…")

    def is_running(self) -> bool:
        return self._running

    def shutdown(self) -> None:
        """Stop a running enumeration and the worker thread, and wait for
        it. Must run before the window is destroyed (MainWindow.closeEvent
        calls it). Safe to call twice."""
        if self._worker is not None:
            self._worker.cancel()
        if self._thread is not None and self._thread.isRunning():
            self._thread.quit()
            self._thread.wait()

    def _set_running(self, running: bool) -> None:
        self._running = running
        self.enumerate_button.setText("Cancel" if running else "Enumerate sites")
        self.progress_bar.setVisible(running)
        self.progress_label.setVisible(running)
        if running:
            self.progress_bar.setRange(0, 0)              # busy until the first report
            self.progress_label.setText("Starting…")
            QApplication.setOverrideCursor(Qt.CursorShape.BusyCursor)
        else:
            QApplication.restoreOverrideCursor()
        self._refresh_state()

    def _on_enumerate_clicked(self) -> None:
        if self._running:
            self.cancel_enumeration()
        else:
            self.enumerate_sites()

    def _on_enumeration_progress(self, stage: str, done: int, total: int) -> None:
        if total > 0:
            self.progress_bar.setRange(0, total)
            self.progress_bar.setValue(done)
        self.progress_label.setText(f"{stage}: {done:,} of {total:,}" if total > 1 else stage)

    def _on_enumeration_cancelled(self) -> None:
        self._set_running(False)
        self._log("Enumeration cancelled; nothing was saved.")

    def _on_enumeration_failed(self, message: str, estimate) -> None:
        self._set_running(False)
        if estimate is not None:
            answer = QMessageBox.question(
                self, "pNMR simulation",
                f"This run would build {estimate['n_rows']:,} configurations "
                f"({estimate['n_sites']} sites, {estimate['n_distinct_environments']} distinct "
                f"environments), roughly {estimate['est_bytes'] / 1e9:.2g} GB of memory while "
                f"the table is built.\n\nNarrowing the shells or the sites would make it "
                f"smaller. Run it anyway?")
            self._log(f"Large run: {estimate['n_rows']:,} configurations "
                      f"(~{estimate['est_bytes'] / 1e9:.2g} GB).")
            if answer == QMessageBox.StandardButton.Yes:
                self.enumerate_sites(limit=None)
            return
        QMessageBox.warning(self, "pNMR simulation", f"Enumeration failed:\n{message}")
        self._log(f"Enumeration failed: {message}")

    def _on_enumeration_finished(self, payload: dict) -> None:
        self._set_running(False)
        spectrum_dict, table = payload["spectrum_dict"], payload["table"]
        kwargs, notes = payload["kwargs"], payload["notes"]
        self._result = (spectrum_dict, table)
        self._result_signature = payload["signature"]
        self._spectrum = self._make_spectrum(spectrum_dict, table)

        shifts = table["Total_shift"].to_numpy()
        p = table["p"].to_numpy()
        mean = float(np.sum(shifts * p))
        params = table.attrs.get("parameters", {})
        lines = [
            f"{params.get('n_sites', table['Site'].nunique())} site(s), {len(table):,} "
            f"configurations. Shifts {shifts.min():.4g} to {shifts.max():.4g} ppm; "
            f"probability-weighted mean {mean:.4g} ppm.",
            f"{spectrum_dict['nuclide']} at {kwargs['B0_tesla']:g} T: BF1 = "
            f"{spectrum_dict['BF1']:.4f} MHz. σ = {kwargs['broadening']:g} ppm "
            f"(FWHM {kwargs['broadening'] * _SIGMA_TO_FWHM:.3g} ppm).",
        ]
        if kwargs["save_shift_table"]:
            lines.append(f"Saved {spectrum_dict['nuclide']}_shift_table.csv and "
                         f"{spectrum_dict['nuclide']}_simulation_parameters.json to "
                         f"{spectrum_dict['path']}")
        if notes:
            lines.append("Warnings:\n  " + "\n  ".join(notes))
        self.results_label.setText("\n".join(lines))
        self._fill_table_view(table)
        shown = {k: v for k, v in kwargs.items() if k != "max_configurations"}
        parser_notes = payload["parser_notes"]
        self._log("Enumerated sites:\n" + self.notebook_code(shown) + "\n→ " + lines[0]
                  + ("\n" + lines[-1] if notes else "")
                  + ("\nCIF parser notes: " + " | ".join(parser_notes) if parser_notes else ""))
        self._refresh_state()

    def _fill_table_view(self, table) -> None:
        shown = table.sort_values("p", ascending=False).head(_MAX_TABLE_ROWS)
        self._shown_index = list(shown.index)
        view = self.shift_table_view
        view.clear()
        view.setColumnCount(len(shown.columns))
        view.setHorizontalHeaderLabels([str(c) for c in shown.columns])
        view.setRowCount(len(shown))
        for r, row in enumerate(shown.itertuples(index=False)):
            for c, value in enumerate(row):
                if isinstance(value, float):
                    text = f"{value:.4g}"
                else:
                    text = str(value)
                view.setItem(r, c, QTableWidgetItem(text))
        view.resizeColumnsToContents()
        view.setVisible(True)
        if len(table) > _MAX_TABLE_ROWS:
            view.setToolTip(f"The {_MAX_TABLE_ROWS} most probable of {len(table)} configurations; "
                            f"the saved CSV has them all.")
        else:
            view.setToolTip("Every configuration, most probable first.")

    def _on_save_table_as(self) -> None:
        if self._result is None:
            return
        spectrum_dict, table = self._result
        start = os.path.join(spectrum_dict["path"], f"{spectrum_dict['nuclide']}_shift_table.csv")
        path, _ = QFileDialog.getSaveFileName(self, "Save shift table", start, "CSV (*.csv)")
        if path:
            self.save_table(path)

    def save_table(self, path: str) -> None:
        """The full shift table to ``path`` (CSV), and the parameters it was
        made with to ``<path stem>_parameters.json``."""
        _, table = self._result
        table.to_csv(path, index=False)
        stem = os.path.splitext(path)[0]
        with open(stem + "_parameters.json", "w", encoding="utf-8") as fh:
            json.dump(table.attrs.get("parameters", {}), fh, indent=2)
        self._log(f"Saved shift table to {path}")

    def _make_spectrum(self, spectrum_dict, table) -> NMRSpectrum:
        axis = NMRAxis(kind="frequency", ppm=np.asarray(spectrum_dict["x_ppm"], dtype=float))
        metadata = NMRMetadata(
            vendor="simulated", nucleus=spectrum_dict["nuclide"],
            sfo1=float(spectrum_dict["SFO1"]), bf1=float(spectrum_dict["BF1"]),
            source_path=self.cif_path,
            raw={"simulation": {"kind": "pnmr_cif", **table.attrs.get("parameters", {})}},
        )
        return NMRSpectrum(data=np.asarray(spectrum_dict["real_spectrum"], dtype=float),
                           axes=[axis], metadata=metadata)

    # ------------------------------------------------------------------
    # Settings: load a saved parameters JSON into every step, or save one
    # ------------------------------------------------------------------
    def _on_load_settings(self) -> None:
        start = self.output_dir() or (os.path.dirname(self.cif_path) if self.cif_path else "")
        path, _ = QFileDialog.getOpenFileName(self, "Load pNMR settings", start,
                                              "Simulation parameters (*.json);;All files (*)")
        if path:
            self.load_settings(path)

    def _on_save_settings(self) -> None:
        try:
            kwargs = self.generate_kwargs()
        except (ValueError, TypeError) as exc:
            QMessageBox.information(self, "pNMR simulation", f"Nothing complete to save yet: {exc}")
            return
        start = os.path.join(self.output_dir() or os.path.dirname(self.cif_path or ""),
                             f"{kwargs['nuclide'] or 'pnmr'}_simulation_parameters.json")
        path, _ = QFileDialog.getSaveFileName(self, "Save pNMR settings", start,
                                              "Simulation parameters (*.json)")
        if path:
            self.save_settings(path)

    # ------------------------------------------------------------------
    # Saved state (simulation history)
    # ------------------------------------------------------------------
    def get_state(self) -> dict:
        """The run's parameters -- the same ones Save settings writes --
        when every step is filled in, else just the plain controls."""
        try:
            kwargs = self.generate_kwargs()
        except (ValueError, TypeError, AttributeError):
            return {"widgets": panel_state.capture_widgets(self)}
        kwargs = {k: v for k, v in kwargs.items() if k not in ("output_dir", "save_shift_table")}
        extra = {k: (json.dumps(v, default=str) if isinstance(v, (list, dict, tuple)) else str(v))
                 for k, v in kwargs.items()}
        return {"kwargs": json.loads(json.dumps(kwargs, default=str)), "extra": extra}

    def set_state(self, state: dict) -> bool:
        """Put a saved run's parameters back into every step (as Load
        settings does, through a temporary parameters file)."""
        state = state or {}
        if "kwargs" not in state:
            panel_state.restore_widgets(self, state.get("widgets") or {})
            return True
        import tempfile
        kwargs = dict(state["kwargs"])
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "restored_simulation_parameters.json")
            pnmr.save_simulation_parameters(path, output_dir=None, **{
                k: v for k, v in kwargs.items() if k != "output_dir"})
            return self.load_settings(path)

    def save_settings(self, path: str) -> None:
        """The current inputs, as the parameters JSON load_settings reads."""
        pnmr.save_simulation_parameters(path, **self.generate_kwargs())
        self._log(f"Saved settings to {path}")

    def load_settings(self, path: str, cif_filename: str | None = None) -> bool:
        """Fill every step from a saved parameters JSON and Initialise, so
        Enumerate is one click away. Anything the file asks for that this
        CIF or panel cannot honour is listed, not silently dropped.

        cif_filename overrides the CIF the file names; if neither it nor a
        same-named file nearby exists, the user is asked to find it."""
        if self._running:
            return False
        try:
            kwargs = pnmr.load_simulation_parameters(path, cif_filename=cif_filename)
        except FileNotFoundError as exc:
            QMessageBox.information(self, "pNMR simulation", f"{exc}\n\nPlease locate the CIF.")
            cif, _ = QFileDialog.getOpenFileName(
                self, "Locate the CIF", os.path.dirname(path),
                "Crystallographic files (*.cif *.mcif);;All files (*)")
            if not cif:
                return False
            return self.load_settings(path, cif_filename=cif)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            QMessageBox.warning(self, "pNMR simulation",
                                f"Could not read settings from {os.path.basename(path)}:\n{exc}")
            return False

        problems: list[str] = []
        if not self.load_cif(kwargs["cif_filename"]):
            return False
        species = kwargs["atom_of_interest"]
        if self.species_combo.findText(species) < 0:
            QMessageBox.warning(self, "pNMR simulation",
                                f"This CIF has no '{species}', the species the settings observe.")
            return False
        self.species_combo.setCurrentText(species)

        def tick(list_widget, wanted, what):
            wanted = list(wanted)
            keys = {list_widget.item(i).data(Qt.ItemDataRole.UserRole)
                    for i in range(list_widget.count())}
            unknown = [w for w in wanted if w not in keys]
            if unknown:
                problems.append(f"{what} not in this CIF, left out: {', '.join(unknown)}.")
            _fill_checklist(list_widget, [(list_widget.item(i).data(Qt.ItemDataRole.UserRole),
                                           list_widget.item(i).text())
                                          for i in range(list_widget.count())], set(wanted))

        sites = kwargs.get("sites_of_interest")
        if sites:
            tick(self.sites_list, sites, f"{species} sites")
        tick(self.ignore_species_list, kwargs.get("ignore_species") or [], "Ignored species")
        tick(self.ignore_sites_list, kwargs.get("ignore_sites") or [], "Ignored sites")

        # Search far enough to show every shell in the histogram.
        reach = max(b[2] for b in kwargs["shell_bins"]) + 0.5
        if reach > self.max_r_spin.value():
            self.max_r_spin.setValue(min(reach, self.max_r_spin.maximum()))
        if not self.initialise():
            return False

        self.shell_table.blockSignals(True)
        self.shell_table.setRowCount(0)
        self.shell_table.blockSignals(False)
        contributors = set(self._species_columns())
        for name, lo, hi in kwargs["shell_bins"]:
            values = kwargs["shift_dict"].get(name, {})
            dropped = [sp for sp, v in values.items() if sp not in contributors and v != 0]
            if dropped:
                problems.append(f"Shell {name}: shifts for {', '.join(dropped)} dropped -- not a "
                                f"contributor once the ignores are applied.")
            self._append_shell_row(name, _fmt_distance(lo), _fmt_distance(hi),
                                   {sp: f"{v:g}" for sp, v in values.items()})
        for shell in kwargs["shift_dict"]:
            if shell not in {b[0] for b in kwargs["shell_bins"]}:
                problems.append(f"Shifts given for shell '{shell}', which has no limits; ignored.")

        nuclide = kwargs["nuclide"]
        if self.nuclide_combo.findText(nuclide) < 0:
            self.nuclide_combo.addItem(nuclide)
        self.nuclide_combo.setCurrentText(nuclide)
        self.field_spin.setValue(float(kwargs["B0_tesla"]))
        if "broadening" in kwargs:
            self.broadening_spin.setValue(float(kwargs["broadening"]))
        if "n_points" in kwargs:
            self.points_spin.setValue(int(kwargs["n_points"]))
        xlims = kwargs.get("xlims")
        self.auto_axis_check.setChecked(xlims is None)
        if xlims is not None:
            self.ppm_min_spin.setValue(min(xlims))
            self.ppm_max_spin.setValue(max(xlims))
        self.weight_check.setChecked(bool(kwargs.get("weight_by_occupancy", True)))
        if float(kwargs.get("offset_ppm") or 0.0) != 0.0:
            problems.append(f"The file uses a carrier offset of {kwargs['offset_ppm']:g} ppm; the "
                            f"panel always uses 0 (BF1 = SFO1). Shifts are unaffected.")
        if kwargs.get("gamma_MHz_per_T") is not None:
            problems.append(f"The file overrides gamma ({kwargs['gamma_MHz_per_T']:g} MHz/T); the "
                            f"panel uses the table value for {nuclide}.")

        self._log(f"Loaded settings from {path}"
                  + (f" (CIF: {kwargs['cif_filename']})" if cif_filename else "")
                  + ". Output folder left as it was, so a rerun does not overwrite the saved one.")
        if problems:
            self._log("  " + "\n  ".join(problems))
            QMessageBox.information(self, "pNMR simulation",
                                    "Loaded, with these differences:\n\n• " + "\n• ".join(problems))
        self._refresh_state()
        return True

    # ------------------------------------------------------------------
    # 5. Plot
    # ------------------------------------------------------------------
    def _build_plot_box(self) -> QGroupBox:
        box = QGroupBox("5. Plot")
        row = QHBoxLayout(box)
        self.plot_button = QPushButton("Plot")
        self.plot_button.setToolTip("Draw the simulated spectrum.")
        self.plot_button.clicked.connect(self.plot)
        row.addWidget(self.plot_button)
        self.add_to_open_check = QCheckBox("Add to Open experiments")
        self.add_to_open_check.setToolTip(
            "Also add the simulated spectrum to Open experiments, to overlay it on measured "
            "data on the Compare tile.")
        row.addWidget(self.add_to_open_check)
        row.addStretch(1)
        return box

    def spectrum_label(self) -> str:
        params = self._result[1].attrs.get("parameters", {}) if self._result else {}
        cif = os.path.splitext(os.path.basename(self.cif_path or "structure"))[0]
        sites = params.get("sites_of_interest")
        what = ",".join(sites) if sites else params.get("atom_of_interest", "")
        return f"pNMR {params.get('nuclide', '')}: {cif} ({what})"

    def plot(self) -> bool:
        if self._spectrum is None or self._result_signature != self._input_signature():
            return False
        # Add first: selecting the new Open experiments row redraws the
        # Spectrum view, and the plot asked for should be the one left up.
        if self.add_to_open_check.isChecked() and self._added_signature != self._result_signature:
            self._add_to_open(self._spectrum, self.spectrum_label())
            self._added_signature = self._result_signature
            self._log(f"Added '{self.spectrum_label()}' to Open experiments.")
        self._plot_spectrum(self._spectrum)
        return True

    # ------------------------------------------------------------------
    # Log, and keeping later steps tied to the inputs that made them
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # 6. Isotropic peaks -> CSA, Quad & J-coupling (2026-10-12)
    # ------------------------------------------------------------------
    def _build_export_box(self) -> QGroupBox:
        box = QGroupBox("6. Add CSA, quadrupolar or J")
        v = QVBoxLayout(box)
        hint = QLabel("The peaks above are isotropic. Send them on as sites -- environments "
                      "within the grouping tolerance become one site -- then add CSA, "
                      "quadrupolar or J-coupling to each.")
        hint.setWordWrap(True)
        v.addWidget(hint)
        form = QFormLayout()
        self.export_tolerance_spin = QDoubleSpinBox()
        self.export_tolerance_spin.setRange(0.0, 1000.0)
        self.export_tolerance_spin.setDecimals(2)
        self.export_tolerance_spin.setSuffix(" ppm")
        self.export_tolerance_spin.setSpecialValueText("auto (smallest pathway shift)")
        self.export_tolerance_spin.setToolTip(
            "Environments closer than this become one site (as Fit ▸ pNMR groups them).")
        form.addRow("Group within:", self.export_tolerance_spin)
        self.export_min_spin = QDoubleSpinBox()
        self.export_min_spin.setRange(0.0, 50.0)
        self.export_min_spin.setDecimals(2)
        self.export_min_spin.setValue(1.0)
        self.export_min_spin.setSuffix(" %")
        self.export_min_spin.setToolTip(
            "Leave out groups holding less than this share of the intensity (the log says how "
            "much was left out). 0 sends every group.")
        form.addRow("Leave out below:", self.export_min_spin)
        self.export_max_spin = QSpinBox()
        self.export_max_spin.setRange(1, 100000)
        self.export_max_spin.setValue(2000)
        self.export_max_spin.setSuffix(" sites")
        self.export_max_spin.setToolTip(
            "The most probable groups, at most this many. Hundreds are fine: peaks given the "
            "same interactions (Copy to Peaks) are simulated once and shifted.")
        form.addRow("At most:", self.export_max_spin)
        v.addLayout(form)
        self.send_sites_row = SendSitesRow(default_destination="simulate",
                                           selected_what="configurations")
        self.send_sites_row.exportRequested.connect(self.export_sites)
        v.addWidget(self.send_sites_row)
        return box

    def _selected_table_rows(self) -> list[int]:
        """Positions in the shift table of the rows selected in its view."""
        if self._result is None:
            return []
        rows = sorted({i.row() for i in self.shift_table_view.selectionModel().selectedRows()})
        labels = [self._shown_index[r] for r in rows if 0 <= r < len(self._shown_index)]
        table = self._result[1]
        return [int(p) for p in table.index.get_indexer(labels) if p >= 0]

    def export_sites(self, which: str = "all") -> list[dict] | None:
        """The enumerated environments, grouped, as sites in CSA, Quad &
        J-coupling (:func:`notebook.pnmr_export.sites_from_pnmr_table`)."""
        if self._result is None:
            return None
        spectrum_dict, table = self._result
        rows = None
        if which != "all":
            rows = self._selected_table_rows()
            if not rows:
                QMessageBox.information(self, "pNMR", "Select one or more configurations in the "
                                        "shift table first.")
                return None
        labels = None
        if self._site_data is not None:
            labels = [d.get("label") for d in self._site_data]
        tolerance = self.export_tolerance_spin.value() or None
        try:
            sites, notes = pnmr_export.sites_from_pnmr_table(
                table, nucleus=spectrum_dict.get("nuclide"), tolerance=tolerance,
                which_rows=rows, min_fraction=self.export_min_spin.value() / 100.0,
                max_sites=self.export_max_spin.value(),
                site_labels=labels, return_notes=True)
        except ValueError as exc:
            QMessageBox.warning(self, "pNMR", str(exc))
            return None
        destination = self.send_sites_row.destination()
        self._log(f"Exported {len(sites)} site(s) to "
                  f"{'Fit' if destination == 'fit' else 'Simulate'} ▸ CSA, Quad & J-coupling"
                  + (" (added)" if self.send_sites_row.append() else "") + ": "
                  + ", ".join(s["label"] for s in sites) + ".\n  " + "\n  ".join(notes))
        if self._export_sites is not None:
            self._export_sites(sites, notes, spectrum_dict.get("nuclide"),
                               spectrum_dict.get("BF1"), destination,
                               self.send_sites_row.append(), "pNMR simulation")
        return sites

    def _build_log_box(self) -> QGroupBox:
        box = QGroupBox("Log")
        v = QVBoxLayout(box)
        self.log_edit = QPlainTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setMaximumHeight(150)
        v.addWidget(self.log_edit)
        self.copy_code_button = QPushButton("Copy notebook code")
        self.copy_code_button.setToolTip(
            "Copy the generate_spectrum_dict call for the current inputs, to rerun this "
            "simulation in a notebook.")
        self.copy_code_button.clicked.connect(self._on_copy_code)
        v.addWidget(self.copy_code_button)
        return box

    def _log(self, text: str) -> None:
        self.log_edit.appendPlainText(text)

    def notebook_code(self, kwargs: dict | None = None) -> str:
        kwargs = kwargs or self.generate_kwargs()
        lines = ["import opennmr.notebook as nmr", "",
                 "spectrum_dict, shift_table = nmr.generate_spectrum_dict("]
        for key, value in kwargs.items():
            if key == "cif_filename":
                lines.append(f'    cif_filename=r"{value}",')
            elif key == "shift_dict":
                lines.append("    shift_dict={")
                for shell, d in value.items():
                    inner = ", ".join(f"{sp!r}: {v:g}" for sp, v in d.items())
                    lines.append(f"        {shell!r}: {{{inner}}},")
                lines.append("    },")
            else:
                lines.append(f"    {key}={value!r},")
        lines.append(")")
        return "\n".join(lines)

    def _on_copy_code(self) -> None:
        try:
            code = self.notebook_code()
        except ValueError as exc:
            QMessageBox.information(self, "pNMR simulation", str(exc))
            return
        QGuiApplication.clipboard().setText(code)
        self._log("Copied notebook code to the clipboard.")

    def _refresh_state(self) -> None:
        """Enable each step only when everything before it is current, and
        nothing but Cancel while an enumeration runs (its inputs are
        already fixed; editing them would only mislead)."""
        running = self._running
        for w in (self.cif_browse_button, self.cif_drop, self.load_settings_button,
                  self.save_settings_button, self.plot_button, self.save_table_button,
                  self.copy_code_button, self.send_sites_row):
            w.setEnabled(not running)
        for box in (self._sites_box, self._shells_box):
            box.setEnabled(not running)
        for w in self._spectrum_inputs:
            w.setEnabled(not running)
        if running:
            self.enumerate_button.setEnabled(True)
            return
        loaded = self._site_data is not None
        for w in (self.species_combo, self.sites_list, self.ignore_species_list,
                  self.ignore_sites_list, self.max_r_spin, self.initialise_button):
            w.setEnabled(loaded)
        initialised = loaded and self._init_signature is not None
        init_current = initialised and self._init_signature == self._sites_signature()
        if initialised and not init_current:
            self.init_label.setText("Site choices changed -- press Initialise structure again.")
        elif init_current:
            self.init_label.setText(self._init_text)
        for w in (self.shell_table, self.add_shell_button, self.remove_shell_button):
            w.setEnabled(init_current)
        self.ppm_min_spin.setEnabled(not self.auto_axis_check.isChecked())
        self.ppm_max_spin.setEnabled(not self.auto_axis_check.isChecked())
        signature = self._input_signature() if init_current else None
        self.enumerate_button.setEnabled(signature is not None
                                         and bool(self.nuclide_combo.currentText().strip()))
        result_current = self._result is not None and signature == self._result_signature
        self.plot_button.setEnabled(result_current)
        self.save_table_button.setEnabled(self._result is not None)
        self.send_sites_row.set_enabled(result_current, result_current)
        if self._result is not None and not result_current:
            self.plot_button.setToolTip("Inputs changed since the last enumeration -- "
                                        "press Enumerate sites again.")
        else:
            self.plot_button.setToolTip("Draw the simulated spectrum.")
