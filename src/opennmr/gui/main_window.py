"""Main window: a tabbed side panel (spectrum management + live
processing controls) plus an embedded plotting canvas.

The side panel is a persistent, always-visible :class:`QTabWidget`
rather than transient popup dialogs -- opening a file is still a
(necessary) native file dialog, but everything else (list management,
nucleus override, phase adjustment) lives in the docked tabs and
updates the canvas live.
"""

from __future__ import annotations

import dataclasses
import os
import re

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDialog, QDialogButtonBox,
    QDoubleSpinBox, QFileDialog, QFormLayout, QFrame, QGroupBox, QHBoxLayout,
    QHeaderView, QInputDialog, QLabel, QLineEdit, QListWidget, QListWidgetItem,
    QMainWindow, QMenu, QMessageBox, QPushButton, QScrollArea, QSpinBox, QSplitter,
    QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget,
)

from .. import io
from ..io import bruker as io_bruker
from ..processing import fitting as processing_fitting
from ..processing import ft as processing_ft
from ..processing import relaxation as processing_relaxation
from ..processing import slicing as processing_slicing
from ..processing.slicing import get_trace
from .session import Session
from .widgets.canvas import MplCanvas
from .widgets.drag_control import DragHandle

#: Effectively-unbounded spinbox range for phase controls -- a slider's
#: fixed pixel-to-range mapping would force a tradeoff between coarse
#: and fine control; DragHandle (a drag-to-adjust control, not a
#: slider) and this wide spinbox range don't need one.
_PHASE_SPIN_RANGE = 36000.0

#: (display label, ``processing.ft`` window key) shown in the FID
#: processing box's window combo -- "none" isn't a real key in
#: ``ft._WINDOWS``, it's translated to a no-op exponential (lb=0) by
#: ``MainWindow._on_reprocess_fid_clicked``.
#: (display label, ``processing.relaxation.RELAXATION_MODELS`` key)
#: shown in the Analysis tab's model combo.
_RELAXATION_MODEL_ITEMS = [
    ("Inversion recovery (T1)", "inversion_recovery"),
    ("Saturation recovery (T1)", "saturation_recovery"),
    ("Exponential decay (T2)", "exponential_decay"),
    ("Biexponential decay (T2a/T2b)", "biexponential_decay"),
    ("Biexponential recovery (T1a/T1b)", "biexponential_recovery"),
    ("Stretched decay (T2)", "stretched_decay"),
    ("Stretched recovery (T1)", "stretched_recovery"),
]

#: (display label, ``processing.relaxation.RELAXATION_MODELS`` key)
#: shown in the Analysis tab's model combo when "Diffusion (PFG)" is
#: the selected fit type.
_DIFFUSION_MODEL_ITEMS = [
    ("Stejskal-Tanner (D)", "stejskal_tanner"),
    ("Stejskal-Tanner, stretched (D)", "stejskal_tanner_stretched"),
]

_WINDOW_ITEMS = [
    ("None (no apodization)", "none"),
    ("Exponential (Lorentzian)", "exponential"),
    ("Hann", "hann"),
    ("Gaussian", "gaussian"),
    ("Sine", "sine"),
    ("QSine", "qsine"),
]

#: Standard NMR FFT sizes offered by the zero-fill combo -- 1k..256k,
#: the conventional round-number choices (all multiples of 128, and of
#: 1024 specifically), rather than an arbitrary integer spinbox.
_ZERO_FILL_SIZES = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144]


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("openNMR")
        self.resize(1100, 700)

        self.session = Session()
        # What the phase controls currently act on -- one of "1d",
        # "pseudo2d" (a single slice, or every slice via the "apply to
        # all" checkbox), "2d" (the whole dataset along F1 or F2), or
        # None (nothing phaseable selected). ``_phase_reference_full``
        # is always the *untouched* complex data captured when this was
        # last (re)set -- every preview recomputes from it rather than
        # the (possibly already-edited) live spectrum, so repeated drags
        # never compound and "apply to all slices" can be toggled
        # without baking in a partial edit.
        self._phase_mode: str | None = None
        self._phase_reference_full: np.ndarray | None = None
        self._phase_freq_ppm: np.ndarray | None = None
        # Which spectrum's Analysis-tab state (peak list, fit type/
        # model/values source, results) is currently reflected in the
        # tab's widgets -- saved into that spectrum's own
        # metadata.raw before switching to a different one, and
        # restored (or reset to fresh defaults) for the newly
        # selected spectrum, so peak lists/fit results stay siloed per
        # spectrum rather than lingering across a selection change.
        self._current_analysis_index: int | None = None
        # The per-slice pseudo-Voigt fit results from the most recent
        # Analysis-tab fit (NMR2D_fit_slices's own "rows" output) --
        # lets "View slice fit" browse the intermediate spectral fits
        # that fed the T1/T2/D parameter fit, not just its final number.
        self._slice_fit_results: list[dict] | None = None

        # One MplCanvas per kind of plot, each keeping its own figure/
        # zoom/pan state, navigable via canvas_tabs instead of every
        # kind clobbering a single shared canvas. ``self.canvas`` is
        # the main "Spectrum" view (1D / 2D contour / pseudo-2D slice
        # or stack) -- unqualified since it's the default/most-used
        # one and predates the other three.
        self.canvas = MplCanvas()
        self.canvas.peakPicked.connect(self._on_peak_picked)
        self.canvas.referenceShiftRequested.connect(self._on_reference_shift_from_plot)
        self.fid_canvas = MplCanvas()
        self.compare_canvas = MplCanvas()
        self.fit_canvas = MplCanvas()
        # A single-slice pseudo-Voigt spectral fit (data + fit curve,
        # from "View slice fit") gets its own tab rather than sharing
        # the Spectrum canvas -- it used to temporarily overwrite the
        # normal spectrum view there, which meant losing your place.
        self.spectral_fit_canvas = MplCanvas()

        self.canvas_tabs = QTabWidget()
        self.canvas_tabs.addTab(self.canvas, "Spectrum")
        self.canvas_tabs.addTab(self.fid_canvas, "FID")
        self.canvas_tabs.addTab(self.compare_canvas, "Compare")
        self.canvas_tabs.addTab(self.fit_canvas, "Fit")
        self.canvas_tabs.addTab(self.spectral_fit_canvas, "Spectral Fit")
        # No more "Preview as FID" checkbox -- clicking the FID tab
        # itself is what shows/refreshes the current spectrum's FID.
        self.canvas_tabs.currentChanged.connect(self._on_canvas_tab_changed)

        tabs = QTabWidget()
        tabs.addTab(self._scrollable(self._build_spectra_tab()), "Spectra")
        tabs.addTab(self._scrollable(self._build_processing_tab()), "Processing")
        tabs.addTab(self._scrollable(self._build_compare_tab()), "Compare")
        tabs.addTab(self._scrollable(self._build_analysis_tab()), "Analysis")

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(tabs)
        splitter.addWidget(self.canvas_tabs)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([300, 800])

        self.setCentralWidget(splitter)
        self._set_processing_enabled(False)
        self.pseudo_2d_checkbox.setVisible(False)
        self.pseudo2d_view_combo.setVisible(False)
        self.dimension_box.setVisible(False)

    @staticmethod
    def _scrollable(widget: QWidget) -> QScrollArea:
        """Wrap a side-panel tab's content in a scroll area rather
        than adding it to the tab widget directly.

        Several of these tabs show/hide whole groups of controls
        depending on the selected spectrum (the FID-processing box,
        the Dimension box, the delay-source rows, ...). Without this,
        each group becoming visible grows the tab's minimum size hint,
        and Qt only ever grows a top-level window to fit a larger
        minimum size -- it never auto-shrinks it back down once a
        narrower spectrum is selected, so the window would ratchet
        taller over a session and never come back (reported as "the
        window is sometimes becoming taller for no apparent reason").
        Scrolling internally instead means that size-hint growth is
        absorbed here, not propagated up to the window."""
        scroll = QScrollArea()
        scroll.setWidget(widget)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        return scroll

    # ============================================================
    # Spectra tab: open / list / remove / nucleus override
    # ============================================================

    def _build_spectra_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        open_button = QPushButton("Open…")
        open_button.setToolTip(
            "Pick a folder -- either one experiment itself, or a parent "
            "folder holding several. If more than one spectrum is found "
            "inside, you'll get to choose which to open."
        )
        open_button.clicked.connect(self._on_open_clicked)
        layout.addWidget(open_button)

        self.spectrum_list = QListWidget()
        self.spectrum_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.spectrum_list.setToolTip(
            "Ctrl/Cmd-click (or Shift-click) to select several spectra "
            "and compare them in the Compare tab."
        )
        self.spectrum_list.currentRowChanged.connect(self._on_selection_changed)
        self.spectrum_list.itemSelectionChanged.connect(self._on_list_selection_changed)
        self.spectrum_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.spectrum_list.customContextMenuRequested.connect(self._on_spectrum_list_context_menu)
        layout.addWidget(self.spectrum_list)

        remove_button = QPushButton("Remove")
        remove_button.clicked.connect(self._on_remove_clicked)
        clear_button = QPushButton("Clear All")
        clear_button.clicked.connect(self._on_clear_clicked)
        remove_row = QHBoxLayout()
        remove_row.addWidget(remove_button)
        remove_row.addWidget(clear_button)
        layout.addLayout(remove_row)

        info_box = QGroupBox("Selected spectrum")
        info_form = QFormLayout(info_box)
        self.info_label = QLabel("—")
        self.info_label.setWordWrap(True)
        info_form.addRow(self.info_label)

        self.pseudo_2d_checkbox = QCheckBox("Force pseudo-2D (F1 is not a frequency dimension)")
        self.pseudo_2d_checkbox.setToolTip(
            "Override auto-detection of whether F1 is a real chemical-shift "
            "dimension (gets its own FFT) or an index/slice dimension (e.g. "
            "a relaxation delay or diffusion gradient series) -- auto-detection "
            "relies on the vendor having recorded this conventionally, which "
            "isn't guaranteed for every pulse sequence. Only applies to 2D "
            "spectra; reprocesses from the original file when changed."
        )
        self.pseudo_2d_checkbox.stateChanged.connect(self._on_pseudo_2d_toggled)
        info_form.addRow(self.pseudo_2d_checkbox)

        self.pseudo2d_view_combo = QComboBox()
        self.pseudo2d_view_combo.addItems(["Contour", "Stacked slices"])
        self.pseudo2d_view_combo.setToolTip(
            "How to display a pseudo-2D series: as a contour (F2 vs. "
            "slice number), or as every slice plotted as an offset "
            "1D stack."
        )
        self.pseudo2d_view_combo.currentIndexChanged.connect(self._on_pseudo2d_view_changed)
        info_form.addRow("View:", self.pseudo2d_view_combo)

        layout.addWidget(info_box)
        layout.addStretch(1)
        return tab

    def _on_open_clicked(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self, "Open a data folder -- an experiment folder itself, or a "
                  "parent folder containing several",
        )
        if not directory:
            return

        matches = self._scan_for_spectra(directory)
        if not matches:
            QMessageBox.critical(
                self, "No spectra found",
                f"Could not find a Bruker experiment or an X-Pulse spectrum "
                f"under {directory!r}.",
            )
            return

        if len(matches) == 1:
            self._load(*matches[0])
            return

        for path, vendor in self._prompt_select_spectra(matches, directory):
            self._load(path, vendor)

    #: X-Pulse individual-scan-repeat files, e.g.
    #: ``"...H_1_10_raw.jdx"`` -- one per scan within an averaged
    #: acquisition (there's a sibling ``..._avg.jdx`` with the actual
    #: combined spectrum), not independently useful data. Distinct
    #: from ``raw_slices/NNNN_slice.jdx`` (NUS-reconstruction slices,
    #: genuinely independent per-slice data) -- this only matches the
    #: "_raw.jdx" naming, not "_slice.jdx".
    _XPULSE_RAW_SCAN_RE = re.compile(r"_\d+_raw\.(jdx|dx)$", re.IGNORECASE)

    @classmethod
    def _scan_for_spectra(cls, directory: str, max_depth: int = 6, max_matches: int = 500) -> list[tuple[str, str]]:
        """Find every Bruker experiment and X-Pulse spectrum under
        ``directory`` -- so opening works whether the user picks a
        single experiment folder directly, or a parent folder holding
        many (the standard Bruker convention: numbered experiment
        subfolders, or, for X-Pulse, a folder of ``.jdx`` files
        including ones nested under ``raw/sliceN/``).

        A Bruker experiment claims its own subtree once recognised
        (its ``pdata``/``ser`` contents aren't separately re-checked,
        so a big processed dataset doesn't explode into dozens of
        spurious matches); every ``.jdx``/``.dx`` file is its own
        X-Pulse match, since each is independently loadable -- except
        individual per-scan ``..._raw.jdx`` files (see
        ``_XPULSE_RAW_SCAN_RE``), which are skipped so opening a
        folder with N scans doesn't offer N+1 near-duplicate matches
        (the N individual scans, plus the one averaged spectrum that's
        actually worth loading).
        """
        matches: list[tuple[str, str]] = []

        def walk(path: str, depth: int) -> None:
            if len(matches) >= max_matches:
                return
            try:
                vendor = io.detect_vendor(path)
            except ValueError:
                vendor = None
            if vendor is not None:
                matches.append((path, vendor))
                return  # don't descend into a recognised experiment's own subtree

            if depth >= max_depth:
                return
            try:
                entries = sorted(os.scandir(path), key=lambda e: e.name)
            except OSError:
                return
            for entry in entries:
                if len(matches) >= max_matches:
                    return
                if entry.is_dir(follow_symlinks=False):
                    walk(entry.path, depth + 1)
                elif entry.is_file(follow_symlinks=False):
                    if cls._XPULSE_RAW_SCAN_RE.search(entry.name):
                        continue
                    try:
                        file_vendor = io.detect_vendor(entry.path)
                    except ValueError:
                        continue
                    matches.append((entry.path, file_vendor))

        walk(os.path.abspath(directory), 0)
        return matches

    def _prompt_select_spectra(self, matches: list[tuple[str, str]], base_dir: str) -> list[tuple[str, str]]:
        """Let the user pick one or more of several spectra found
        under a folder they opened. Returns the ones chosen (empty if
        cancelled)."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Select spectra to open")
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel(f"Found {len(matches)} spectra under {base_dir}:"))

        list_widget = QListWidget()
        list_widget.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        for path, vendor in matches:
            rel = os.path.relpath(path, base_dir)
            list_widget.addItem(QListWidgetItem(f"{rel}  ({vendor})"))
        if list_widget.count():
            list_widget.setCurrentRow(0)
        layout.addWidget(list_widget)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Open | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return []
        return [matches[list_widget.row(item)] for item in list_widget.selectedItems()]

    def _load(self, path: str, vendor: str) -> None:
        try:
            self.session.add(path, vendor=vendor)
        except Exception as exc:  # noqa: BLE001 -- surfaced to the user, not swallowed
            QMessageBox.critical(self, "Failed to load spectrum", str(exc))
            return

        index = len(self.session) - 1
        self.spectrum_list.addItem(self.session.label(index))
        self._select_single_row(index)

    def _select_single_row(self, row: int) -> None:
        """Select exactly ``row``, clearing any other selection.

        Under ``ExtendedSelection`` mode, ``setCurrentRow`` alone moves
        keyboard/current focus *without* deselecting previously
        selected rows (by design, so ctrl/shift-click gestures compose)
        -- which would otherwise leave a stale multi-selection (and the
        Compare view still showing it) after loading a new spectrum or
        reordering one.
        """
        self.spectrum_list.clearSelection()
        self.spectrum_list.setCurrentRow(row)
        if 0 <= row < self.spectrum_list.count():
            self.spectrum_list.item(row).setSelected(True)

    def _on_remove_clicked(self) -> None:
        row = self.spectrum_list.currentRow()
        if row < 0:
            return
        self.session.remove(row)
        self.spectrum_list.takeItem(row)
        # Removing a spectrum shifts every later row's index, which
        # would otherwise leave the Compare list's stored indices
        # silently pointing at the wrong spectrum (removing one
        # spectrum doesn't necessarily fire itemSelectionChanged, if
        # the set of *selected* items didn't itself change) -- rebuild
        # from scratch rather than try to renumber it in place.
        rows = self._selected_rows()
        if len(rows) > 1:
            self._populate_compare_list(rows)
            self._refresh_compare_view()
        else:
            self.compare_list.clear()

    def _on_clear_clicked(self) -> None:
        self.session.clear()
        self.spectrum_list.clear()
        for canvas in (self.canvas, self.fid_canvas, self.compare_canvas, self.fit_canvas, self.spectral_fit_canvas):
            canvas.clear()
        self._phase_mode = None
        self._set_processing_enabled(False)
        self.info_label.setText("—")
        self.pseudo_2d_checkbox.setVisible(False)
        self.pseudo2d_view_combo.setVisible(False)
        self.dimension_box.setVisible(False)
        self.fid_processing_box.setVisible(False)
        self.compare_status_label.setText("Select at least two 1D spectra to compare.")
        self.compare_list.clear()
        self.peak_table.setRowCount(0)
        self.analysis_results_label.setText("")
        self.analysis_status_label.setText("Select a pseudo-2D spectrum (e.g. a T1/T2 relaxation series) to fit.")
        self._current_analysis_index = None
        self._slice_fit_results = None
        self.view_slice_fit_spin.setEnabled(False)

    def _selected_index(self) -> int | None:
        row = self.spectrum_list.currentRow()
        return row if 0 <= row < len(self.session) else None

    def _on_spectrum_list_context_menu(self, pos) -> None:
        item = self.spectrum_list.itemAt(pos)
        if item is None:
            return
        index = self.spectrum_list.row(item)

        menu = QMenu(self)
        override_action = menu.addAction("Override nucleus…")
        suggest_action = menu.addAction("Suggest nucleus (500 MHz console / 100 MHz magnet)…")
        mass_action = menu.addAction("Set sample mass…")
        procno_action = None
        if self.session.spectra[index].metadata.vendor == "bruker":
            procno_action = menu.addAction("Change proc number…")
        menu.addSeparator()
        remove_action = menu.addAction("Remove")

        chosen = menu.exec(self.spectrum_list.viewport().mapToGlobal(pos))
        if chosen is override_action:
            self._on_override_nucleus_action(index)
        elif chosen is suggest_action:
            self._on_suggest_nucleus_action(index)
        elif chosen is mass_action:
            self._on_set_mass_action(index)
        elif procno_action is not None and chosen is procno_action:
            self._on_change_procno_action(index)
        elif chosen is remove_action:
            self.session.remove(index)
            self.spectrum_list.takeItem(index)

    def _set_nucleus(self, index: int, nucleus: str) -> None:
        """Apply a nucleus override and refresh the UI to match --
        split out from the menu actions that prompt for the value, so
        it can be called directly without a modal dialog in the way
        (e.g. from :func:`_on_pseudo_2d_toggled`'s reload, or tests)."""
        self.session.set_nucleus(index, nucleus)
        self.spectrum_list.item(index).setText(self.session.label(index))
        if index == self._selected_index():
            self._refresh_selection_display(index)

    def _on_override_nucleus_action(self, index: int) -> None:
        current = self.session.spectra[index].metadata.nucleus or ""
        text, ok = QInputDialog.getText(self, "Override nucleus", "Nucleus:", text=current)
        if ok and text.strip():
            self._set_nucleus(index, text.strip())

    def _on_suggest_nucleus_action(self, index: int) -> None:
        suggestion = self.session.suggest_nucleus(index)
        if suggestion is None:
            QMessageBox.information(
                self, "No suggestion available",
                "Need a recorded nucleus and SFO1 to suggest from.",
            )
            return
        reply = QMessageBox.question(
            self, "Suggested nucleus",
            f"For a nuclide picked from an instrument table calibrated for a "
            f"500 MHz console but actually run on a 100 MHz magnet, the real "
            f"nucleus is likely {suggestion} (from the recorded SFO1).\n\n"
            f"Apply this override?",
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._set_nucleus(index, suggestion)

    def _on_set_mass_action(self, index: int) -> None:
        current = self.session.spectra[index].metadata.mass or 0.0
        value, ok = QInputDialog.getDouble(
            self, "Set sample mass", "Mass (g) -- used for mass-normalized comparison:",
            current, 0.0, 1.0e6, 4,
        )
        if ok and value > 0:
            self.session.set_mass(index, value)
            if index == self._selected_index():
                self._refresh_selection_display(index)
            self._refresh_compare_view_if_active()

    def _on_change_procno_action(self, index: int) -> None:
        current = self.session.spectra[index].metadata.procno or 1
        value, ok = QInputDialog.getInt(
            self, "Change proc number",
            "pdata/<n> to use for TopSpin's own recorded processing "
            "parameters (window/phase/zero-fill) -- the raw ser/fid "
            "data is still read by default regardless of proc number:",
            current, 1, 999,
        )
        if not ok or value == current:
            return
        try:
            self.session.reload(index, procno=value)
        except Exception as exc:  # noqa: BLE001 -- surfaced to the user, not swallowed
            QMessageBox.critical(self, "Failed to change proc number", str(exc))
            return
        self.spectrum_list.item(index).setText(self.session.label(index))
        if index == self._selected_index():
            self._on_selection_changed(index)

    def _on_pseudo_2d_toggled(self, _state: int) -> None:
        index = self._selected_index()
        if index is None:
            return
        spectrum = self.session.spectra[index]
        if spectrum.ndim != 2 or spectrum.is_pseudo_2d == self.pseudo_2d_checkbox.isChecked():
            return
        try:
            self.session.reload(index, pseudo_2d=self.pseudo_2d_checkbox.isChecked())
        except Exception as exc:  # noqa: BLE001 -- surfaced to the user, not swallowed
            QMessageBox.critical(self, "Failed to reprocess spectrum", str(exc))
            # Revert the checkbox to match what's actually loaded.
            self.pseudo_2d_checkbox.blockSignals(True)
            self.pseudo_2d_checkbox.setChecked(spectrum.is_pseudo_2d)
            self.pseudo_2d_checkbox.blockSignals(False)
            return
        self._on_selection_changed(index)

    def _on_pseudo2d_view_changed(self, _index: int) -> None:
        index = self._selected_index()
        if index is not None:
            self._draw_selected_spectrum(self.session.spectra[index])

    def _show_pseudo2d_stack(self, spectrum) -> None:
        """Plot every slice of a pseudo-2D series as an offset 1D
        stack -- an alternative to the default contour view, for
        judging a whole relaxation/diffusion series at a glance.
        Reuses :func:`MplCanvas.show_compare` (built for the Compare
        tab): a pseudo-2D stack is exactly "many 1D traces, stacked",
        the same thing that method already does."""
        n_slices = spectrum.data.shape[0]
        slices = processing_slicing.get_slices(spectrum, rows=list(range(1, n_slices + 1)))
        self.canvas.show_compare(slices["rows"], mode="stack", y_offset=0.2, normalize=True)
        self.canvas_tabs.setCurrentWidget(self.canvas)

    # ============================================================
    # Compare tab: multi-spectrum stack/overlay
    # ============================================================

    def _build_compare_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        layout.addWidget(QLabel(
            "Ctrl/Cmd-click (or Shift-click) several spectra in the "
            "Spectra tab's list to start a comparison, then fine-tune "
            "it here -- check/uncheck to include or exclude, Move Up/"
            "Down to reorder the plot."
        ))

        self.compare_status_label = QLabel("Select at least two 1D spectra to compare.")
        self.compare_status_label.setWordWrap(True)
        layout.addWidget(self.compare_status_label)

        self.compare_list = QListWidget()
        self.compare_list.setToolTip(
            "The spectra currently in this comparison. Uncheck one to "
            "drop it from the plot without losing your Spectra-tab "
            "selection; select a row and use Move Up/Down to change "
            "the stack order."
        )
        self.compare_list.itemChanged.connect(lambda _item: self._refresh_compare_view_if_active())
        layout.addWidget(self.compare_list)

        reorder_row = QHBoxLayout()
        move_up_button = QPushButton("Move Up")
        move_up_button.setToolTip("Move the selected row earlier in the comparison's plot order.")
        move_up_button.clicked.connect(lambda: self._on_move_compare_item(-1))
        move_down_button = QPushButton("Move Down")
        move_down_button.setToolTip("Move the selected row later in the comparison's plot order.")
        move_down_button.clicked.connect(lambda: self._on_move_compare_item(1))
        reorder_row.addWidget(move_up_button)
        reorder_row.addWidget(move_down_button)
        layout.addLayout(reorder_row)

        form = QFormLayout()

        self.layout_mode_combo = QComboBox()
        self.layout_mode_combo.addItems(["Stacked", "Superimposed"])
        self.layout_mode_combo.currentIndexChanged.connect(lambda _i: self._on_compare_controls_changed())
        form.addRow("Layout:", self.layout_mode_combo)

        self.stack_gap_spin = QDoubleSpinBox()
        self.stack_gap_spin.setRange(0.0, 5.0)
        self.stack_gap_spin.setSingleStep(0.05)
        self.stack_gap_spin.setValue(0.2)
        self.stack_gap_spin.setToolTip(
            "Vertical gap between stacked spectra, as a fraction of "
            "each spectrum's own peak height."
        )
        self.stack_gap_spin.valueChanged.connect(lambda _v: self._on_compare_controls_changed())
        form.addRow("Stack gap:", self.stack_gap_spin)

        self.compare_normalize_combo = QComboBox()
        self.compare_normalize_combo.addItems(["None", "Max intensity", "By sample mass"])
        self.compare_normalize_combo.setToolTip(
            "None: as loaded (already per-scan -- every importer divides "
            "by NS). Max intensity: each spectrum scaled to its own peak "
            "height, for shape comparison regardless of concentration. "
            "By sample mass: divided by the mass set via the spectrum "
            "list's right-click \"Set sample mass…\" -- spectra with "
            "no mass set are skipped."
        )
        self.compare_normalize_combo.currentIndexChanged.connect(lambda _i: self._on_compare_controls_changed())
        form.addRow("Normalize:", self.compare_normalize_combo)

        layout.addLayout(form)

        layout.addStretch(1)
        return tab

    def _selected_rows(self) -> list[int]:
        return sorted(self.spectrum_list.row(item) for item in self.spectrum_list.selectedItems())

    def _on_list_selection_changed(self) -> None:
        rows = self._selected_rows()
        if len(rows) > 1:
            self._populate_compare_list(rows)
            self._refresh_compare_view()
        else:
            # Deliberately doesn't clear compare_list here -- a
            # multi-select gesture (e.g. clearSelection() followed by
            # several setSelected(True) calls) can pass through a
            # transient <=1-row state along the way, and clearing on
            # every such blip would wipe out check-state/reordering
            # already done in the Compare tab for no reason. It's
            # naturally rebuilt (or intentionally left as-is) next
            # time a *sustained* multi-selection comes in.
            #
            # Multi- -> single-selection doesn't always change
            # currentRow() (the item that was already current can stay
            # current), so currentRowChanged wouldn't fire on its own
            # -- redraw the single-spectrum view explicitly.
            self._on_selection_changed(self.spectrum_list.currentRow())

    def _populate_compare_list(self, rows: list[int]) -> None:
        """Rebuild the Compare tab's own list from the Spectra tab's
        current multi-selection, preserving any check-state/reorder
        already done here for rows still selected -- so ctrl-clicking
        one more spectrum in the Spectra tab appends it rather than
        wiping out prior deselections/reordering, but ctrl-clicking a
        row *out* of the selection still drops it from here too."""
        # Filter out-of-range indices defensively -- a spectrum removal
        # can trigger this (directly, or reentrantly via the list
        # widget's own itemSelectionChanged) at a point where a stale
        # row number briefly doesn't match session.spectra's new,
        # shorter length.
        n = len(self.session)
        rows = [r for r in rows if 0 <= r < n]
        previous_order = [
            r for r in (self.compare_list.item(i).data(Qt.ItemDataRole.UserRole)
                        for i in range(self.compare_list.count()))
            if 0 <= r < n
        ]
        previous_checked = {
            self.compare_list.item(i).data(Qt.ItemDataRole.UserRole): self.compare_list.item(i).checkState()
            for i in range(self.compare_list.count())
        }
        rows_set = set(rows)
        new_order = [r for r in previous_order if r in rows_set] + [r for r in rows if r not in previous_order]

        self.compare_list.blockSignals(True)
        self.compare_list.clear()
        for r in new_order:
            item = QListWidgetItem(self.session.label(r))
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(previous_checked.get(r, Qt.CheckState.Checked))
            item.setData(Qt.ItemDataRole.UserRole, r)
            self.compare_list.addItem(item)
        self.compare_list.blockSignals(False)

    def _compare_checked_rows(self) -> list[int]:
        rows = []
        for i in range(self.compare_list.count()):
            item = self.compare_list.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                rows.append(item.data(Qt.ItemDataRole.UserRole))
        return rows

    def _on_move_compare_item(self, delta: int) -> None:
        """Reorders rows within the Compare tab's own list -- this is
        purely local plot order (which spectrum is on top/bottom of
        the stack), not the master Spectra-tab list order."""
        row = self.compare_list.currentRow()
        new_row = row + delta
        if row < 0 or not (0 <= new_row < self.compare_list.count()):
            return
        item = self.compare_list.takeItem(row)
        self.compare_list.insertItem(new_row, item)
        self.compare_list.setCurrentRow(new_row)
        self._refresh_compare_view_if_active()

    def _on_compare_controls_changed(self) -> None:
        self.stack_gap_spin.setEnabled(self.layout_mode_combo.currentIndex() == 0)
        self._refresh_compare_view_if_active()

    def _refresh_compare_view_if_active(self) -> None:
        if self.compare_list.count() > 0:
            self._refresh_compare_view()

    def _refresh_compare_view(self) -> None:
        rows = self._compare_checked_rows()
        candidates = [self.session.spectra[r] for r in rows]
        spectra = [s for s in candidates if s.ndim == 1]
        skipped_ndim = len(candidates) - len(spectra)

        norm_mode = ["none", "max", "mass"][self.compare_normalize_combo.currentIndex()]
        plot_spectra = spectra
        skipped_mass: list = []
        if norm_mode == "mass":
            # Only `data` (the real part actually plotted here) needs
            # scaling -- these copies are used for this one comparison
            # plot and discarded, never stored back into the session.
            plot_spectra = []
            for s in spectra:
                if not s.metadata.mass:
                    skipped_mass.append(s)
                    continue
                plot_spectra.append(dataclasses.replace(s, data=s.data / s.metadata.mass))

        if len(plot_spectra) < 2:
            bits = ["Select at least two 1D spectra to compare."]
            if skipped_ndim:
                bits.append(f"{skipped_ndim} selected spectrum/spectra skipped (not 1D).")
            if skipped_mass:
                bits.append(f"{len(skipped_mass)} skipped (no sample mass set).")
            self.compare_status_label.setText(" ".join(bits))
            self.compare_canvas.clear()
            return

        bits = [f"Comparing {len(plot_spectra)} spectra."]
        if skipped_ndim:
            bits.append(f"{skipped_ndim} skipped (not 1D).")
        if skipped_mass:
            bits.append(f"{len(skipped_mass)} skipped (no sample mass set).")
        self.compare_status_label.setText(" ".join(bits))

        mode = "stack" if self.layout_mode_combo.currentIndex() == 0 else "overlay"
        try:
            self.compare_canvas.show_compare(
                plot_spectra, mode=mode, y_offset=self.stack_gap_spin.value(),
                normalize=(norm_mode == "max"),
            )
            self.canvas_tabs.setCurrentWidget(self.compare_canvas)
        except Exception as exc:  # noqa: BLE001 -- surfaced to the user, not swallowed
            QMessageBox.critical(self, "Failed to plot comparison", str(exc))

    # ============================================================
    # Processing tab: live phase adjustment
    # ============================================================

    def _build_fid_processing_box(self) -> QGroupBox:
        """A group of pre-FFT (raw-FID) reprocessing controls --
        apodization/window, zero-fill, echo alignment, FID shift,
        linear prediction -- distinct from the Phase group below it,
        which operates on the already-transformed spectrum in place.
        Always visible: each change only takes effect on "Reprocess",
        not live."""
        box = self.fid_processing_box = QGroupBox("FID processing (before FFT)")
        box.setToolTip(
            "Reprocess from the raw time-domain FID with different "
            "apodization, zero-fill, echo alignment, FID shift, or "
            "linear prediction than the default. Only available for "
            "spectra loaded from a raw fid/ser file (or any X-Pulse "
            "spectrum, which is always raw-FID)."
        )
        form = QFormLayout(box)

        self.window_combo = QComboBox()
        self.window_combo.currentIndexChanged.connect(self._on_window_combo_changed)
        form.addRow("Window:", self.window_combo)

        self.lb_spin = QDoubleSpinBox()
        self.lb_spin.setRange(0.0, 1.0e5)
        self.lb_spin.setDecimals(2)
        self.lb_spin.setSuffix(" Hz")
        self.lb_spin.setToolTip("Line broadening, for the Exponential window only.")
        self.lb_spin.valueChanged.connect(lambda _v: self._refresh_fid_preview_if_active())
        form.addRow("Line broadening:", self.lb_spin)

        self.zero_fill_combo = QComboBox()
        for size in _ZERO_FILL_SIZES:
            self.zero_fill_combo.addItem(f"{size // 1024}k ({size} pts)", size)
        self.zero_fill_combo.addItem("Custom…", None)
        self.zero_fill_combo.setToolTip(
            "FFT size -- zero-fills (or truncates) the FID to this many "
            "points. Defaults to the nearest standard size at or above "
            "2x the acquired points."
        )
        self.zero_fill_combo.currentIndexChanged.connect(self._on_zero_fill_combo_changed)
        form.addRow("Zero-fill to:", self.zero_fill_combo)

        self.zero_fill_custom_spin = QSpinBox()
        self.zero_fill_custom_spin.setRange(1, 1 << 22)
        self.zero_fill_custom_spin.setVisible(False)
        self.zero_fill_custom_spin.valueChanged.connect(lambda _v: self._refresh_fid_preview_if_active())
        form.addRow("Custom size:", self.zero_fill_custom_spin)

        self.tdeff_spin = QSpinBox()
        self.tdeff_spin.setRange(0, 1 << 20)
        self.tdeff_spin.setSpecialValueText("Off (full FID)")
        self.tdeff_spin.setSuffix(" pts")
        self.tdeff_spin.setToolTip(
            "Truncate the raw FID to this many points before any other "
            "processing (X-Pulse only) -- e.g. to cut off a recorded "
            "tail past the point the acquisition itself flagged as "
            "effective (TDEFF), prefilled here when the file records "
            "one. 'Off' keeps the full acquired FID. Shown as a dashed "
            "red line on the FID plot."
        )
        self.tdeff_spin.valueChanged.connect(lambda _v: self._refresh_fid_preview_if_active())
        form.addRow("TDeff:", self.tdeff_spin)

        self.align_echo_checkbox = QCheckBox("Align as full echo (centre the echo top)")
        self.align_echo_checkbox.setToolTip(
            "For full-echo acquisitions (rising then falling) rather "
            "than a one-sided decaying FID -- rolls the FID so its "
            "peak is centred, before windowing."
        )
        form.addRow(self.align_echo_checkbox)

        self.fid_shift_spin = QDoubleSpinBox()
        self.fid_shift_spin.setRange(-1.0e6, 1.0e6)
        self.fid_shift_spin.setDecimals(2)
        self.fid_shift_spin.setSuffix(" pts")
        self.fid_shift_spin.setToolTip(
            "Manual circular shift of the FID (samples, fractional "
            "allowed), applied before windowing -- independent of "
            "(and on top of) echo alignment."
        )
        form.addRow("FID shift:", self.fid_shift_spin)

        self.linear_predict_points_spin = QSpinBox()
        self.linear_predict_points_spin.setRange(0, 1 << 16)
        self.linear_predict_points_spin.setToolTip(
            "Extend the FID by this many forward linear-prediction "
            "points before windowing -- reduces truncation artefacts "
            "from a short acquisition. 0 = off."
        )
        form.addRow("Linear-predict points:", self.linear_predict_points_spin)

        self.linear_predict_order_spin = QSpinBox()
        self.linear_predict_order_spin.setRange(1, 1000)
        self.linear_predict_order_spin.setValue(20)
        self.linear_predict_order_spin.setToolTip("Linear-prediction model order (only used when points > 0).")
        form.addRow("LP order:", self.linear_predict_order_spin)

        reprocess_button = QPushButton("Reprocess from raw FID")
        reprocess_button.clicked.connect(self._on_reprocess_fid_clicked)
        form.addRow(reprocess_button)

        return box

    def _populate_window_combo(self, vendor: str) -> None:
        self.window_combo.blockSignals(True)
        self.window_combo.clear()
        if vendor == "bruker":
            self.window_combo.addItem("Recorded (TopSpin's own WDW)", "__recorded__")
        for label, key in _WINDOW_ITEMS:
            self.window_combo.addItem(label, key)
        self.window_combo.setCurrentIndex(0)
        self.window_combo.blockSignals(False)
        self.lb_spin.setEnabled(self.window_combo.currentData() == "exponential")

    def _on_window_combo_changed(self, _index: int) -> None:
        self.lb_spin.setEnabled(self.window_combo.currentData() == "exponential")
        self._refresh_fid_preview_if_active()

    def _set_window_combo_key(self, key: str) -> None:
        idx = self.window_combo.findData(key)
        self.window_combo.blockSignals(True)
        self.window_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.window_combo.blockSignals(False)
        self.lb_spin.setEnabled(self.window_combo.currentData() == "exponential")

    def _on_zero_fill_combo_changed(self, _index: int) -> None:
        self.zero_fill_custom_spin.setVisible(self.zero_fill_combo.currentData() is None)
        self._refresh_fid_preview_if_active()

    def _zero_fill_value(self) -> int:
        data = self.zero_fill_combo.currentData()
        return data if data is not None else self.zero_fill_custom_spin.value()

    def _set_zero_fill_value(self, value: int) -> None:
        idx = self.zero_fill_combo.findData(value)
        self.zero_fill_combo.blockSignals(True)
        self.zero_fill_combo.setCurrentIndex(idx if idx >= 0 else self.zero_fill_combo.count() - 1)
        self.zero_fill_combo.blockSignals(False)
        self.zero_fill_custom_spin.setVisible(idx < 0)
        self.zero_fill_custom_spin.blockSignals(True)
        self.zero_fill_custom_spin.setValue(value)
        self.zero_fill_custom_spin.blockSignals(False)

    @staticmethod
    def _default_zero_fill_size(original_length: int | None) -> int:
        """The nearest standard size at or above 2x the acquired
        points -- conventional NMR zero-filling. Falls back to a
        middling preset when the acquired length isn't known, and to
        the raw doubled value (handled by the caller as "Custom") when
        it exceeds every preset."""
        if not original_length:
            return _ZERO_FILL_SIZES[len(_ZERO_FILL_SIZES) // 2]
        target = 2 * original_length
        for size in _ZERO_FILL_SIZES:
            if size >= target:
                return size
        return target

    @staticmethod
    def _fid_reprocessable(spectrum) -> bool:
        """Whether ``spectrum`` was loaded from a raw FID this GUI can
        re-read and reprocess with different parameters -- true for any
        X-Pulse spectrum (always raw-FID), or a Bruker experiment whose
        raw ``fid``/``ser`` is present on disk (if so, :func:`opennmr.
        io.bruker._read_auto` would have preferred it over the
        already-processed fallback in the first place)."""
        if spectrum.metadata.vendor == "xpulse":
            return True
        if spectrum.metadata.vendor == "bruker" and spectrum.metadata.source_path:
            base = spectrum.metadata.source_path
            return os.path.exists(os.path.join(base, "ser")) or os.path.exists(os.path.join(base, "fid"))
        return False

    @staticmethod
    def _fid_preview_available(spectrum) -> bool:
        """Whether there's a processed FID stashed to preview, and
        showing it unambiguously means one thing -- true for 1D and
        pseudo-2D (a single slice); a conventional 2D's F1 FID isn't
        captured (see ``read_2d``'s docstring), and even F2's FID is a
        whole row-per-F1-position array with no obvious single "the"
        FID to show."""
        fid = spectrum.metadata.raw.get("fid")
        return fid is not None and (spectrum.ndim == 1 or spectrum.is_pseudo_2d)

    def _show_fid_preview(self, spectrum, switch_tab: bool = True) -> None:
        fid = spectrum.metadata.raw.get("fid")
        dt = spectrum.metadata.raw.get("fid_dt")
        title = ""
        if fid.ndim == 2:
            row = min(max(self.slice_spin.value() - 1, 0), fid.shape[0] - 1) if spectrum.is_pseudo_2d else 0
            if spectrum.is_pseudo_2d:
                title = self._slice_label(spectrum, row + 1)
            fid = fid[row]
        envelope = self._compute_window_envelope(dt, len(fid))
        tdeff = self.tdeff_spin.value() if self.tdeff_spin.isEnabled() else 0
        self.fid_canvas.show_fid(fid, dt, title=title, window_envelope=envelope, tdeff_index=tdeff or None)
        if switch_tab:
            self.canvas_tabs.setCurrentWidget(self.fid_canvas)

    def _compute_window_envelope(self, dt: float | None, n: int) -> np.ndarray | None:
        """The currently-selected window's own shape (0..1 envelope),
        for :func:`_show_fid_preview`'s overlay -- ``None`` for
        "Recorded (TopSpin)", which has no explicit user-facing window
        parameters to preview, or if there's no FID length/dt to
        build one against."""
        key = self.window_combo.currentData()
        if key is None or key == "__recorded__" or dt is None:
            return None
        window_kwarg, window_kwargs_kwarg = self._resolve_window_kwargs()
        try:
            envelope = processing_ft.apodize(
                np.ones(n, dtype=np.complex128), dt, window=window_kwarg, **(window_kwargs_kwarg or {}),
            )
        except Exception:  # noqa: BLE001 -- a preview overlay is best-effort, never worth crashing over
            return None
        return np.real(envelope)

    def _refresh_fid_preview_if_active(self) -> None:
        """Live-updates the FID tab (window/TDeff/zero-fill preview
        overlays) as a control changes, but only while that tab is
        actually the one showing -- otherwise this would redraw and
        switch the user to a tab they didn't ask for. Clicking the FID
        tab itself (see :meth:`_on_canvas_tab_changed`) is what first
        shows the preview."""
        if self.canvas_tabs.currentWidget() is not self.fid_canvas:
            return
        index = self._selected_index()
        if index is None:
            return
        spectrum = self.session.spectra[index]
        if self._fid_preview_available(spectrum):
            self._show_fid_preview(spectrum, switch_tab=False)

    def _on_canvas_tab_changed(self, _index: int) -> None:
        if self.canvas_tabs.currentWidget() is not self.fid_canvas:
            return
        index = self._selected_index()
        spectrum = self.session.spectra[index] if index is not None else None
        if spectrum is not None and self._fid_preview_available(spectrum):
            self._show_fid_preview(spectrum, switch_tab=False)
        else:
            # Nothing to preview (no spectrum selected, or this one
            # has no processed FID stashed) -- don't leave a stale
            # plot from whatever was last shown here.
            self.fid_canvas.clear()

    def _refresh_fid_processing_controls(self, spectrum, index: int) -> None:
        reprocessable = self._fid_reprocessable(spectrum)
        self.fid_processing_box.setVisible(reprocessable)

        # Restore this spectrum's own last-used FID-processing settings
        # (Session remembers the kwargs behind every load/reload)
        # rather than always resetting to defaults, so switching away
        # and back doesn't lose an in-progress reprocessing choice.
        saved = self.session.get_load_kwargs(index) if reprocessable else {}

        is_xpulse = spectrum.metadata.vendor == "xpulse"
        self.tdeff_spin.setEnabled(reprocessable and is_xpulse)
        self.tdeff_spin.blockSignals(True)
        saved_truncate = saved.get("truncate")
        if isinstance(saved_truncate, int) and not isinstance(saved_truncate, bool):
            default_tdeff = saved_truncate
        else:
            default_tdeff = int(spectrum.metadata.raw.get("exp_params", {}).get("TDEFF", 0) or 0) if is_xpulse else 0
        self.tdeff_spin.setValue(default_tdeff)
        self.tdeff_spin.blockSignals(False)

        if not reprocessable:
            return

        self._populate_window_combo(spectrum.metadata.vendor)

        if "window" in saved:
            window, window_kwargs = saved["window"], saved.get("window_kwargs") or {}
            key = "__recorded__" if window is None else window
            self._set_window_combo_key(key)
            lb = window_kwargs.get("lb", 0.0) if key == "exponential" else 0.0
        else:
            self.window_combo.setCurrentIndex(0)
            self.lb_spin.setEnabled(self.window_combo.currentData() == "exponential")
            lb = 0.0

        for spin, value in (
            (self.lb_spin, lb),
            (self.fid_shift_spin, saved.get("fid_shift", 0.0)),
            (self.linear_predict_points_spin, saved.get("linear_predict_points", 0)),
        ):
            spin.blockSignals(True)
            spin.setValue(value)
            spin.blockSignals(False)
        self.linear_predict_order_spin.blockSignals(True)
        self.linear_predict_order_spin.setValue(saved.get("linear_predict_order", 20))
        self.linear_predict_order_spin.blockSignals(False)

        self.align_echo_checkbox.blockSignals(True)
        self.align_echo_checkbox.setChecked(saved.get("align_echo", False))
        self.align_echo_checkbox.blockSignals(False)

        original_length = spectrum.metadata.raw.get("fid_original_length")
        default_zero_fill = saved.get("zero_fill", self._default_zero_fill_size(original_length))
        self._set_zero_fill_value(default_zero_fill)

    def _resolve_window_kwargs(self) -> tuple[str | None, dict | None]:
        """Translate the window combo's current selection into the
        ``(window, window_kwargs)`` pair the io readers expect --
        shared by the actual reprocess call and the live FID-preview
        overlay, so the overlay always shows exactly what "Reprocess"
        would apply."""
        key = self.window_combo.currentData()
        if key == "__recorded__":
            return None, None
        if key == "none":
            return "exponential", {"lb": 0.0}
        if key == "exponential":
            return "exponential", {"lb": self.lb_spin.value()}
        return key, {}

    def _on_reprocess_fid_clicked(self) -> None:
        index = self._selected_index()
        if index is None:
            return

        window_kwarg, window_kwargs_kwarg = self._resolve_window_kwargs()
        kwargs = {
            "window": window_kwarg,
            "window_kwargs": window_kwargs_kwarg,
            "zero_fill": self._zero_fill_value(),
            "fid_shift": self.fid_shift_spin.value(),
            "align_echo": self.align_echo_checkbox.isChecked(),
            "linear_predict_points": self.linear_predict_points_spin.value(),
        }
        if self.linear_predict_points_spin.value():
            kwargs["linear_predict_order"] = self.linear_predict_order_spin.value()
        if self.tdeff_spin.isEnabled():  # X-Pulse only -- Bruker's read_1d has no `truncate` kwarg
            kwargs["truncate"] = self.tdeff_spin.value() or None

        try:
            self.session.reload(index, **kwargs)
        except Exception as exc:  # noqa: BLE001 -- surfaced to the user, not swallowed
            QMessageBox.critical(self, "Failed to reprocess spectrum", str(exc))
            return

        self.spectrum_list.item(index).setText(self.session.label(index))
        self._on_selection_changed(index)

    def _build_processing_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        self.processing_status_label = QLabel(
            "Select a spectrum with retained imaginary data to adjust its phase."
        )
        self.processing_status_label.setWordWrap(True)
        layout.addWidget(self.processing_status_label)

        layout.addWidget(self._build_fid_processing_box())

        self.dimension_box = QGroupBox("Dimension")
        dim_form = QFormLayout(self.dimension_box)

        self.slice_spin = QSpinBox()
        self.slice_spin.setMinimum(1)
        self.slice_spin.setToolTip(
            "Which pseudo-2D slice (1-based, matching TopSpin's own "
            "numbering) the phase controls below act on."
        )
        self.slice_spin.valueChanged.connect(lambda _v: self._on_phase_target_changed())
        self.slice_row_label = QLabel("Slice:")
        dim_form.addRow(self.slice_row_label, self.slice_spin)

        self.apply_all_slices_checkbox = QCheckBox("Apply this phase to every slice")
        self.apply_all_slices_checkbox.setToolTip(
            "Off: only the slice above is corrected. On: the same "
            "phase0/phase1 is applied uniformly to every slice at once."
        )
        self.apply_all_slices_checkbox.stateChanged.connect(self._on_apply_all_slices_toggled)
        dim_form.addRow(self.apply_all_slices_checkbox)

        self.dim_combo = QComboBox()
        self.dim_combo.addItems(["Direct (F2)", "Indirect (F1)"])
        self.dim_combo.setToolTip(
            "Which dimension the phase correction is applied along, "
            "uniformly across the whole 2D dataset."
        )
        self.dim_combo.currentIndexChanged.connect(lambda _i: self._on_phase_target_changed())
        self.dim_row_label = QLabel("Dimension:")
        dim_form.addRow(self.dim_row_label, self.dim_combo)

        layout.addWidget(self.dimension_box)

        phase_box = QGroupBox("Phase")
        form = QFormLayout(phase_box)

        self.phase0_handle, self.phase0_spin = self._make_phase_control()
        self.phase1_handle, self.phase1_spin = self._make_phase_control()
        form.addRow("Phase 0 (deg):", self._pair_widget(self.phase0_handle, self.phase0_spin))
        form.addRow("Phase 1 (deg):", self._pair_widget(self.phase1_handle, self.phase1_spin))

        self.pivot_spin = QDoubleSpinBox()
        self.pivot_spin.setRange(-1.0e5, 1.0e5)
        self.pivot_spin.setDecimals(3)
        self.pivot_spin.setSuffix(" ppm")
        self.pivot_spin.setToolTip(
            "Phase 1 pivot -- the point held fixed (phase = Phase 0 exactly) "
            "while Phase 1 varies linearly away from it. Defaults to the "
            "centre of the spectrum on selection."
        )
        self.pivot_spin.valueChanged.connect(lambda _v: self._apply_phase_preview())
        form.addRow("Pivot:", self.pivot_spin)

        reset_button = QPushButton("Reset")
        reset_button.clicked.connect(self._on_reset_phase_clicked)
        form.addRow(reset_button)

        layout.addWidget(phase_box)
        layout.addWidget(self._build_reference_box())
        layout.addStretch(1)
        return tab

    def _build_reference_box(self) -> QGroupBox:
        """Spectral reference (SR) correction -- shift the direct
        (F2) axis by a constant Hz offset so a peak at a known
        position reads correctly, without needing to re-import/
        reprocess. Also available as a one-shot from the Spectrum
        plot's right-click menu (:meth:`_on_reference_shift_from_plot`),
        which is usually quicker than coming to this tab."""
        box = QGroupBox("Reference (SR)")
        box.setToolTip(
            "Shift the direct-dimension ppm/Hz axis by a constant Hz "
            "offset -- e.g. the recorded SR correction from a "
            "calibration standard. Positive moves peaks to higher "
            "frequency/ppm."
        )
        form = QFormLayout(box)

        self.sr_shift_spin = QDoubleSpinBox()
        self.sr_shift_spin.setRange(-1.0e6, 1.0e6)
        self.sr_shift_spin.setDecimals(4)
        self.sr_shift_spin.setSuffix(" Hz")
        form.addRow("Add:", self.sr_shift_spin)

        apply_button = QPushButton("Apply Reference Shift")
        apply_button.clicked.connect(self._on_apply_sr_clicked)
        form.addRow(apply_button)

        return box

    def _make_phase_control(self) -> tuple[DragHandle, QDoubleSpinBox]:
        handle = DragHandle(sensitivity=0.2, step=1.0)

        spin = QDoubleSpinBox()
        spin.setRange(-_PHASE_SPIN_RANGE, _PHASE_SPIN_RANGE)
        spin.setDecimals(2)
        spin.setValue(0.0)

        handle.valueChanged.connect(lambda v: self._on_phase_handle_changed(handle, spin, v))
        spin.valueChanged.connect(lambda v: self._on_phase_spin_changed(handle, spin, v))
        return handle, spin

    @staticmethod
    def _pair_widget(handle: DragHandle, spin: QDoubleSpinBox) -> QWidget:
        widget = QWidget()
        row = QHBoxLayout(widget)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(handle)
        row.addWidget(spin)
        return widget

    def _on_phase_handle_changed(self, handle: DragHandle, spin: QDoubleSpinBox, value: float) -> None:
        if spin.value() != value:
            spin.blockSignals(True)
            spin.setValue(value)
            spin.blockSignals(False)
        self._apply_phase_preview()

    def _on_phase_spin_changed(self, handle: DragHandle, spin: QDoubleSpinBox, value: float) -> None:
        handle.setValue(value, emit=False)
        self._apply_phase_preview()

    def _on_reset_phase_clicked(self) -> None:
        for handle in (self.phase0_handle, self.phase1_handle):
            handle.setValue(0.0, emit=False)
        for spin in (self.phase0_spin, self.phase1_spin):
            spin.blockSignals(True)
            spin.setValue(0.0)
            spin.blockSignals(False)
        self._apply_phase_preview()

    def _on_phase_target_changed(self) -> None:
        """The pseudo-2D slice or the 2D F1/F2 dimension picked in the
        Dimension box changed -- what the phase controls act on is now
        a different view, so re-anchor them to it (see
        ``_capture_phase_reference``) and redraw to show that view at
        its own current (un-previewed) phase."""
        self._capture_phase_reference()
        self._apply_phase_preview()

    def _on_apply_all_slices_toggled(self, _state: int) -> None:
        # Doesn't change *what* the reference is (still the untouched
        # full-2D snapshot), only how the next preview uses it -- no
        # recapture, just re-render with the checkbox's new meaning.
        index = self._selected_index()
        if index is not None:
            spectrum = self.session.spectra[index]
            if spectrum.ndim == 2 and spectrum.is_pseudo_2d:
                n_slices = spectrum.data.shape[0]
                row = self.slice_spin.value()
                status = (
                    f"Adjust phase for all {n_slices} slices at once."
                    if self.apply_all_slices_checkbox.isChecked()
                    else f"Adjust phase for slice {row} of {n_slices}."
                )
                self.processing_status_label.setText(status)
        self._apply_phase_preview()

    def _apply_phase_preview(self) -> None:
        index = self._selected_index()
        if index is None or self._phase_mode is None:
            return

        spectrum = self.session.spectra[index]
        phase0 = self.phase0_spin.value()
        phase1 = self.phase1_spin.value()

        freq_ppm = self._phase_freq_ppm
        span = freq_ppm.max() - freq_ppm.min()
        pivot_frac = (self.pivot_spin.value() - freq_ppm.min()) / span if span else 0.5

        if self._phase_mode == "1d":
            phased = processing_ft.apply_phase(self._phase_reference_full, freq_ppm, phase0, phase1, pivot_frac=pivot_frac)
            spectrum.data = np.real(phased)
            spectrum.data_imag = np.imag(phased)
            self.canvas.update_data(spectrum)
            self.canvas_tabs.setCurrentWidget(self.canvas)

        elif self._phase_mode == "pseudo2d":
            if self.apply_all_slices_checkbox.isChecked():
                phased = processing_ft.apply_phase(
                    self._phase_reference_full, freq_ppm, phase0, phase1, pivot_frac=pivot_frac,
                )
            else:
                row = self.slice_spin.value() - 1
                phased = self._phase_reference_full.copy()
                phased[row, :] = processing_ft.apply_phase(
                    self._phase_reference_full[row, :], freq_ppm, phase0, phase1, pivot_frac=pivot_frac,
                )
            spectrum.data = np.real(phased)
            spectrum.data_imag = np.imag(phased)
            # Phase is a post-FFT operation and never touches the FID --
            # but switching slices (which also routes through here) must
            # still refresh an active FID preview to the newly selected
            # slice, not redraw it with the (irrelevant, for that view)
            # phased spectrum line.
            if self.canvas_tabs.currentWidget() is self.fid_canvas and self._fid_preview_available(spectrum):
                self._show_fid_preview(spectrum, switch_tab=False)
            else:
                self._show_pseudo2d_slice(spectrum, reset_view=False)

        elif self._phase_mode == "2d":
            dim = "f2" if self.dim_combo.currentIndex() == 0 else "f1"
            if dim == "f2":
                phased = processing_ft.apply_phase(
                    self._phase_reference_full, freq_ppm, phase0, phase1, pivot_frac=pivot_frac,
                )
            else:
                phased = processing_ft.apply_phase(
                    self._phase_reference_full.T, freq_ppm, phase0, phase1, pivot_frac=pivot_frac,
                ).T
            spectrum.data = np.real(phased)
            spectrum.data_imag = np.imag(phased)
            self.canvas.show_spectrum(spectrum)
            self.canvas_tabs.setCurrentWidget(self.canvas)

    def _show_pseudo2d_slice(self, spectrum, reset_view: bool) -> None:
        """Preview a pseudo-2D spectrum as the single slice currently
        selected in the Dimension box -- a full contour redraw on every
        drag tick would be far too slow, and a line plot is what you
        actually need to see to judge a phase correction anyway."""
        slice_index = self.slice_spin.value()
        trace = get_trace(spectrum, axis="f1", index=slice_index)
        if reset_view or not self.canvas.ax.lines:
            self.canvas.show_spectrum(trace)
        else:
            self.canvas.update_data(trace)
        self.canvas.ax.set_title(self._slice_label(spectrum, slice_index))
        self.canvas.canvas.draw_idle()
        self.canvas_tabs.setCurrentWidget(self.canvas)

    @staticmethod
    def _slice_label(spectrum, slice_index: int) -> str:
        """A human-readable label for one pseudo-2D slice -- the actual
        indirect-axis value (e.g. a delay or gradient step) when one is
        recorded and differs from the slice's own position, otherwise
        just its 1-based position."""
        axis = spectrum.axes[0]
        if axis.index is not None and 1 <= slice_index <= len(axis.index):
            recorded = axis.index[slice_index - 1]
            if recorded != slice_index:
                return f"Slice {slice_index} (index {recorded:g})"
        return f"Slice {slice_index}"

    def _capture_phase_reference(self) -> None:
        """(Re)anchor the phase controls to whatever they're currently
        meant to act on -- a 1D spectrum, one pseudo-2D slice (or all of
        them), or a whole 2D dataset along F1/F2 -- and zero them, so
        dragging back to 0 always returns to how *that view* looked when
        you started editing it. Called on every spectrum selection and
        every slice/dimension change; never called mid-drag."""
        for handle in (self.phase0_handle, self.phase1_handle):
            handle.setValue(0.0, emit=False)
        for spin in (self.phase0_spin, self.phase1_spin):
            spin.blockSignals(True)
            spin.setValue(0.0)
            spin.blockSignals(False)

        index = self._selected_index()
        if index is None:
            self._phase_mode = None
            self._phase_reference_full = None
            self._phase_freq_ppm = None
            return

        spectrum = self.session.spectra[index]

        if spectrum.data_imag is None:
            self._phase_mode = None
            self._phase_reference_full = None
            self._phase_freq_ppm = None
            self._set_processing_enabled(False, "Phase adjustment needs a spectrum with retained imaginary data.")
            return

        if spectrum.ndim == 1:
            self._phase_mode = "1d"
            self._phase_reference_full = spectrum.complex_data
            self._phase_freq_ppm = spectrum.axis_values(0, "ppm")
            self._set_processing_enabled(True, "Adjust phase for the selected spectrum.")

        elif spectrum.ndim == 2 and spectrum.is_pseudo_2d:
            self._phase_mode = "pseudo2d"
            self._phase_reference_full = spectrum.complex_data
            self._phase_freq_ppm = spectrum.axis_values(-1, "ppm")
            n_slices = spectrum.data.shape[0]
            row = self.slice_spin.value()
            status = (
                f"Adjust phase for all {n_slices} slices at once."
                if self.apply_all_slices_checkbox.isChecked()
                else f"Adjust phase for slice {row} of {n_slices}."
            )
            self._set_processing_enabled(True, status)

        elif spectrum.ndim == 2:
            self._phase_mode = "2d"
            self._phase_reference_full = spectrum.complex_data
            dim = "f2" if self.dim_combo.currentIndex() == 0 else "f1"
            self._phase_freq_ppm = spectrum.axis_values(-1 if dim == "f2" else 0, "ppm")
            dim_label = "F2 (direct)" if dim == "f2" else "F1 (indirect)"
            self._set_processing_enabled(True, f"Adjust phase along {dim_label} for the whole 2D spectrum.")

        else:
            self._phase_mode = None
            self._phase_reference_full = None
            self._phase_freq_ppm = None
            self._set_processing_enabled(False, "Phase adjustment needs a spectrum with retained imaginary data.")
            return

        self.pivot_spin.blockSignals(True)
        self.pivot_spin.setValue(float((self._phase_freq_ppm.min() + self._phase_freq_ppm.max()) / 2))
        self.pivot_spin.blockSignals(False)

    def _set_processing_enabled(self, enabled: bool, status: str | None = None) -> None:
        for widget in (self.phase0_handle, self.phase0_spin, self.phase1_handle, self.phase1_spin, self.pivot_spin):
            widget.setEnabled(enabled)
        if status is not None:
            self.processing_status_label.setText(status)

    # ============================================================
    # Analysis tab: pseudo-Voigt peak fitting -> per-slice relaxation
    # ============================================================
    #
    # Workflow (all backend pieces already existed, ported from
    # nmrtools2, just never wired to the GUI):
    #   processing.slicing.get_slices -> processing.fitting.NMR2D_fit_slices
    #   -> processing.relaxation.NMR2D_fit_relaxation
    # i.e. extract every slice, pseudo-Voigt-fit the peaks the user
    # defined in each one, then fit a relaxation curve to each peak's
    # per-slice amplitude series.

    _PEAK_TABLE_COLUMNS = ("Shift (ppm)", "FWHM (ppm)", "Eta")

    def _build_analysis_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        self.analysis_status_label = QLabel(
            "Select a pseudo-2D spectrum (e.g. a T1/T2 relaxation series) to fit."
        )
        self.analysis_status_label.setWordWrap(True)
        layout.addWidget(self.analysis_status_label)

        peak_box = QGroupBox("Peaks (pseudo-Voigt)")
        peak_layout = QVBoxLayout(peak_box)

        self.peak_table = QTableWidget(0, len(self._PEAK_TABLE_COLUMNS))
        self.peak_table.setHorizontalHeaderLabels(self._PEAK_TABLE_COLUMNS)
        self.peak_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.peak_table.setToolTip(
            "One row per peak to fit -- shift and FWHM are initial "
            "guesses (both, and the amplitude, are refined by the fit); "
            "eta is the Gaussian/Lorentzian mixing fraction (0 = pure "
            "Gaussian, 1 = pure Lorentzian), also refined."
        )
        peak_layout.addWidget(self.peak_table)

        peak_button_row = QHBoxLayout()
        add_peak_button = QPushButton("Add Peak")
        add_peak_button.clicked.connect(lambda: self._add_peak_row())
        remove_peak_button = QPushButton("Remove Peak")
        remove_peak_button.clicked.connect(self._on_remove_peak_clicked)
        self.pick_peak_button = QPushButton("Pick from Plot")
        self.pick_peak_button.setCheckable(True)
        self.pick_peak_button.setToolTip(
            "While active, click a peak's position directly on the "
            "plot to add a row here at that shift, instead of typing "
            "it in."
        )
        self.pick_peak_button.toggled.connect(self._on_pick_peak_toggled)
        peak_button_row.addWidget(add_peak_button)
        peak_button_row.addWidget(remove_peak_button)
        peak_button_row.addWidget(self.pick_peak_button)
        peak_layout.addLayout(peak_button_row)

        tolerance_row = QHBoxLayout()
        tolerance_row.addWidget(QLabel("Position tolerance:"))
        self.position_tolerance_spin = QDoubleSpinBox()
        self.position_tolerance_spin.setRange(0.0, 20.0)
        self.position_tolerance_spin.setDecimals(2)
        self.position_tolerance_spin.setSingleStep(0.1)
        self.position_tolerance_spin.setValue(0.5)
        self.position_tolerance_spin.setSuffix(" × FWHM")
        self.position_tolerance_spin.setToolTip(
            "How far each peak's fitted position is allowed to drift "
            "from the shift entered in the table above, as a multiple "
            "of that peak's own (initial-guess) FWHM -- keeps a fit "
            "from sliding onto a neighbouring peak or drifting across "
            "the whole window. 0 locks the position exactly; a large "
            "value is effectively unconstrained."
        )
        tolerance_row.addWidget(self.position_tolerance_spin)
        peak_layout.addLayout(tolerance_row)

        layout.addWidget(peak_box)

        relax_box = QGroupBox("Fit")
        relax_form = QFormLayout(relax_box)

        self.fit_type_combo = QComboBox()
        self.fit_type_combo.addItems(["Relaxation (T1/T2)", "Diffusion (PFG)"])
        self.fit_type_combo.setToolTip(
            "Relaxation: fits a T1/T2/T1rho model against delay (s), "
            "read from this experiment's vdlist or entered manually. "
            "Diffusion: fits a Stejskal-Tanner model against b-value "
            "(s/m^2), read from a Bruker diff.xml (TopSpin's own "
            "precomputed b-values) or entered manually."
        )
        self.fit_type_combo.currentIndexChanged.connect(self._on_fit_type_changed)
        relax_form.addRow("Fit type:", self.fit_type_combo)

        self.relaxation_model_combo = QComboBox()
        relax_form.addRow("Model:", self.relaxation_model_combo)

        self.delays_row_label = QLabel("Values:")
        self.delay_source_combo = QComboBox()
        self.delay_source_combo.addItems([
            "Auto", "Manual list", "Start + step", "vclist × factor", "Relative gradient (valueslist)",
        ])
        self.delay_source_combo.setToolTip(
            "Auto: this experiment's own vdlist/diff.xml if found, "
            "otherwise plain slice index (1, 2, 3, ...). Manual list: "
            "type values in directly. Start + step: generate an "
            "evenly-spaced series starting at Start and incrementing "
            "by Step, one value per slice (the count is already known "
            "from the data, so there's no separate End to enter). "
            "vclist × factor: Bruker's variable-counter list, scaled "
            "by a constant (e.g. a per-count delay or gradient "
            "increment). Relative gradient: an X-Pulse arrayed PFG "
            "experiment's own relative gradient-strength fractions "
            "(0-1, from valueslist.txt) -- not an absolute b-value, "
            "useful for judging attenuation trends or as a fit x-axis "
            "when no gradient calibration is available; see TODO.md "
            "for converting this to a true b-value via a max-gradient "
            "setting."
        )
        self.delay_source_combo.currentIndexChanged.connect(self._on_delay_source_changed)
        relax_form.addRow(self.delays_row_label, self.delay_source_combo)

        self.manual_delays_edit = QLineEdit()
        self.manual_delays_edit.setPlaceholderText("e.g. 0.001, 0.01, 0.1, 1.0")
        relax_form.addRow("List:", self.manual_delays_edit)

        self.start_end_step_row = QWidget()
        start_end_step_layout = QHBoxLayout(self.start_end_step_row)
        start_end_step_layout.setContentsMargins(0, 0, 0, 0)
        self.delay_start_spin = QDoubleSpinBox()
        self.delay_start_spin.setRange(-1.0e12, 1.0e12)
        self.delay_start_spin.setDecimals(6)
        self.delay_step_spin = QDoubleSpinBox()
        self.delay_step_spin.setRange(-1.0e12, 1.0e12)
        self.delay_step_spin.setDecimals(6)
        self.delay_step_spin.setValue(0.1)
        for widget, label in ((self.delay_start_spin, "Start"), (self.delay_step_spin, "Step")):
            widget.setToolTip(label)
            start_end_step_layout.addWidget(QLabel(f"{label}:"))
            start_end_step_layout.addWidget(widget)
        self.start_end_step_row_label = QLabel("Start / step:")
        relax_form.addRow(self.start_end_step_row_label, self.start_end_step_row)

        self.vclist_factor_spin = QDoubleSpinBox()
        self.vclist_factor_spin.setRange(-1.0e12, 1.0e12)
        self.vclist_factor_spin.setDecimals(6)
        self.vclist_factor_spin.setValue(1.0)
        self.vclist_factor_spin.setToolTip(
            "Each vclist entry (an integer counter) is multiplied by "
            "this factor to get the actual delay (s) or b-value "
            "(s/m^2) -- e.g. a per-count time increment or gradient step."
        )
        self.vclist_factor_row_label = QLabel("Factor:")
        relax_form.addRow(self.vclist_factor_row_label, self.vclist_factor_spin)

        self._populate_fit_model_combo()
        self._on_delay_source_changed(0)

        layout.addWidget(relax_box)

        fit_button = QPushButton("Fit")
        fit_button.clicked.connect(self._on_fit_relaxation_clicked)
        layout.addWidget(fit_button)

        self.analysis_results_label = QLabel("")
        self.analysis_results_label.setWordWrap(True)
        layout.addWidget(self.analysis_results_label)

        slice_fit_row = QHBoxLayout()
        slice_fit_row.addWidget(QLabel("View slice fit:"))
        self.view_slice_fit_spin = QSpinBox()
        self.view_slice_fit_spin.setMinimum(1)
        self.view_slice_fit_spin.setEnabled(False)
        self.view_slice_fit_spin.setToolTip(
            "Inspect the per-slice spectral (pseudo-Voigt) fit that "
            "feeds the parameter fit above, in the Spectrum tab -- "
            "available once a fit has been run."
        )
        self.view_slice_fit_spin.valueChanged.connect(self._on_view_slice_fit_changed)
        slice_fit_row.addWidget(self.view_slice_fit_spin)
        layout.addLayout(slice_fit_row)

        layout.addStretch(1)
        return tab

    def _add_peak_row(self, shift: float = 0.0, fwhm: float = 1.0, eta: float = 0.5) -> None:
        row = self.peak_table.rowCount()
        self.peak_table.insertRow(row)
        for col, value in enumerate((shift, fwhm, eta)):
            self.peak_table.setItem(row, col, QTableWidgetItem(f"{value:.4g}"))

    def _on_remove_peak_clicked(self) -> None:
        row = self.peak_table.currentRow()
        if row >= 0:
            self.peak_table.removeRow(row)

    def _on_peak_picked(self, shift: float) -> None:
        if self.pick_peak_button.isChecked():
            # A reasonable default width: 1% of the currently-visible
            # x-range, since there's no per-peak width information in
            # a single click -- the fit refines it regardless.
            xmin, xmax = self.canvas.ax.get_xlim()
            fwhm_guess = abs(xmax - xmin) * 0.01 or 1.0
            self._add_peak_row(shift=shift, fwhm=fwhm_guess, eta=0.5)

    def _on_pick_peak_toggled(self, checked: bool) -> None:
        self.canvas.set_pick_mode(checked)

    def _on_apply_sr_clicked(self) -> None:
        delta_hz = self.sr_shift_spin.value()
        if delta_hz == 0:
            return
        self._apply_reference_shift_hz(delta_hz)
        self.sr_shift_spin.setValue(0.0)

    def _on_reference_shift_from_plot(self) -> None:
        """Handles :attr:`MplCanvas.referenceShiftRequested`, the
        Spectrum plot's right-click "Set Reference Shift (Hz)…" --
        a one-shot alternative to the Processing tab's Reference box
        for the common case of just entering a known SR correction."""
        index = self._selected_index()
        if index is None:
            return
        spectrum = self.session.spectra[index]
        current_hz = spectrum.metadata.raw.get("sr_shift_hz", 0.0)
        value, ok = QInputDialog.getDouble(
            self, "Reference shift", "Add (Hz):", current_hz, -1.0e6, 1.0e6, 4,
        )
        if ok and value != 0:
            self._apply_reference_shift_hz(value)

    def _apply_reference_shift_hz(self, delta_hz: float) -> None:
        index = self._selected_index()
        if index is None:
            return
        spectrum = self.session.spectra[index]
        axis = spectrum.axes[-1]  # direct (F2) dimension -- always frequency
        if axis.hz is not None:
            axis.hz = axis.hz + delta_hz
        if axis.ppm is not None and spectrum.metadata.sfo1:
            axis.ppm = axis.ppm + delta_hz / spectrum.metadata.sfo1

        spectrum.metadata.raw["sr_shift_hz"] = spectrum.metadata.raw.get("sr_shift_hz", 0.0) + delta_hz
        self._draw_selected_spectrum(spectrum)

    def _read_peak_defs(self) -> list[dict]:
        tolerance = self.position_tolerance_spin.value()
        peak_defs = []
        for row in range(self.peak_table.rowCount()):
            try:
                shift = float(self.peak_table.item(row, 0).text())
                fwhm = float(self.peak_table.item(row, 1).text())
                eta = float(self.peak_table.item(row, 2).text())
            except (AttributeError, ValueError) as exc:
                raise ValueError(f"Peak row {row + 1}: shift/FWHM/eta must all be numbers.") from exc
            peak_def = {"type": "pseudo_voigt", "shift": shift, "fwhm": fwhm, "eta": eta}
            # Default position bounds used to be unconstrained
            # (-inf, inf) -- a fit could slide a peak onto a neighbour
            # or drift across the whole window. Tie it to the peak's
            # own FWHM (rather than a fixed ppm width) so narrow and
            # broad peaks each get a proportionally sensible leash.
            # scipy's curve_fit requires lower < upper strictly, so a
            # tiny floor keeps tolerance=0 ("lock the position") from
            # raising instead of just pinning it very tightly.
            margin = max(tolerance * fwhm, 1e-9)
            peak_def["bounds"] = {"shift": (shift - margin, shift + margin)}
            peak_defs.append(peak_def)
        return peak_defs

    def _is_diffusion_fit(self) -> bool:
        return self.fit_type_combo.currentIndex() == 1

    def _populate_fit_model_combo(self) -> None:
        items = _DIFFUSION_MODEL_ITEMS if self._is_diffusion_fit() else _RELAXATION_MODEL_ITEMS
        self.relaxation_model_combo.blockSignals(True)
        self.relaxation_model_combo.clear()
        for label, key in items:
            self.relaxation_model_combo.addItem(label, key)
        self.relaxation_model_combo.blockSignals(False)

    def _on_fit_type_changed(self, _index: int) -> None:
        self._populate_fit_model_combo()
        self.delays_row_label.setText("b-values:" if self._is_diffusion_fit() else "Delays:")
        unit = "s/m²" if self._is_diffusion_fit() else "s"
        self.manual_delays_edit.setPlaceholderText(f"e.g. 1e8, 5e8, 2e9 ({unit})" if self._is_diffusion_fit()
                                                     else "e.g. 0.001, 0.01, 0.1, 1.0 (s)")

    def _on_delay_source_changed(self, index: int) -> None:
        self.manual_delays_edit.setVisible(index == 1)
        self.start_end_step_row_label.setVisible(index == 2)
        self.start_end_step_row.setVisible(index == 2)
        self.vclist_factor_row_label.setVisible(index == 3)
        self.vclist_factor_spin.setVisible(index == 3)

    def _resolve_independent_values(self, spectrum, is_diffusion: bool) -> np.ndarray | dict:
        """Resolve the x-axis values (delays in seconds, or b-values in
        s/m^2) to fit against, per the Values-row source combo."""
        source = self.delay_source_combo.currentIndex()
        n_slices = spectrum.data.shape[0]

        if source == 1:  # Manual list
            text = self.manual_delays_edit.text().strip()
            if not text:
                raise ValueError("Enter a comma-separated list of values, or switch Values to Auto/Start-End-Step/vclist.")
            try:
                return np.array([float(v) for v in text.split(",")], dtype=float)
            except ValueError as exc:
                raise ValueError("Manual values must be a comma-separated list of numbers.") from exc

        if source == 2:  # Start + step
            start, step = self.delay_start_spin.value(), self.delay_step_spin.value()
            if step == 0:
                raise ValueError("Step must be nonzero.")
            # One value per slice -- the count is already fixed by the
            # data, so there's no separate End to enter (and no risk
            # of an off-by-one/roundoff mismatch against n_slices).
            return start + step * np.arange(n_slices, dtype=float)

        if source == 3:  # vclist x factor
            source_path = spectrum.metadata.source_path
            if spectrum.metadata.vendor != "bruker" or not source_path:
                raise ValueError("vclist is only available for Bruker experiments.")
            vclist = io_bruker.read_vclist(source_path)
            return vclist["seconds"] * self.vclist_factor_spin.value()

        if source == 4:  # Relative gradient (X-Pulse arrayed PFG valueslist.txt)
            gradient_relative = spectrum.metadata.raw.get("gradient_relative")
            if gradient_relative is None:
                raise ValueError(
                    "No relative gradient values found -- this source needs an "
                    "X-Pulse arrayed experiment loaded with its own "
                    "parameters/G1_valueslist.txt (e.g. a PFG series)."
                )
            return np.asarray(gradient_relative, dtype=float)

        # source == 0: Auto -- this experiment's own vdlist/diff.xml,
        # falling back to plain slice index (1, 2, 3, ...) when neither
        # is found, so a fit is always possible without necessarily
        # meaning anything physical (e.g. for quickly checking whether
        # a series decays monotonically at all).
        source_path = spectrum.metadata.source_path
        if spectrum.metadata.vendor == "bruker" and source_path:
            try:
                if is_diffusion:
                    return io_bruker.read_diff_xml(source_path, nucleus=spectrum.metadata.nucleus)["B"]
                return io_bruker.read_vdlist(source_path)
            except FileNotFoundError:
                pass
        return np.arange(1, n_slices + 1, dtype=float)

    def _on_fit_relaxation_clicked(self) -> None:
        index = self._selected_index()
        if index is None:
            return
        spectrum = self.session.spectra[index]
        if spectrum.ndim != 2 or not spectrum.is_pseudo_2d:
            QMessageBox.information(self, "Not a pseudo-2D spectrum", "Select a pseudo-2D relaxation/diffusion series to fit.")
            return

        is_diffusion = self._is_diffusion_fit()
        try:
            peak_defs = self._read_peak_defs()
            if not peak_defs:
                raise ValueError("Add at least one peak to fit.")
            independent_values = self._resolve_independent_values(spectrum, is_diffusion)

            n_slices = spectrum.data.shape[0]
            slices = processing_slicing.get_slices(spectrum, rows=list(range(1, n_slices + 1)))
            fitted = processing_fitting.NMR2D_fit_slices(
                slices, peak_defs, mode="rows", show_report=False, show_warnings=False,
            )
            model_key = self.relaxation_model_combo.currentData()
            results = processing_relaxation.NMR2D_fit_relaxation(
                fitted["rows"], model=model_key, vdlist=independent_values, show_report=False,
            )
        except Exception as exc:  # noqa: BLE001 -- surfaced to the user, not swallowed
            QMessageBox.critical(self, "Fit failed", str(exc))
            return

        lines = []
        for result in results:
            rate_param = result["rate_param"]
            if rate_param is not None:
                formatter = processing_relaxation.format_diffusion_value if is_diffusion else processing_relaxation.format_time_value
                value_str = formatter(result["params"][rate_param], result["errors"].get(rate_param))
                lines.append(f"{result['peak_label']}: {rate_param} = {value_str}")
            else:
                params_str = ", ".join(f"{k}={v:.4g}" for k, v in result["params"].items())
                lines.append(f"{result['peak_label']}: {params_str}")
        xlabel = "b-value (s/m²)" if is_diffusion else "Delay (s)"
        self.analysis_results_label.setText("\n".join(lines))
        self.fit_canvas.show_relaxation_fit(results)
        self.fit_canvas.ax.set_xlabel(xlabel)
        self.fit_canvas.canvas.draw_idle()
        self.canvas_tabs.setCurrentWidget(self.fit_canvas)

        self._slice_fit_results = fitted["rows"]
        # Kept per-spectrum (not just in the live _slice_fit_results
        # attribute) so "View slice fit" and the Fit-tab plot both
        # survive a reselection -- analysis_state alone only remembers
        # the *text* summary, not the plotted curves.
        spectrum.metadata.raw["slice_fit_results"] = fitted["rows"]
        spectrum.metadata.raw["relaxation_fit_results"] = results
        spectrum.metadata.raw["relaxation_fit_xlabel"] = xlabel
        self.view_slice_fit_spin.blockSignals(True)
        self.view_slice_fit_spin.setEnabled(True)
        self.view_slice_fit_spin.setMaximum(max(1, len(self._slice_fit_results)))
        self.view_slice_fit_spin.setValue(1)
        self.view_slice_fit_spin.blockSignals(False)

    def _on_view_slice_fit_changed(self, value: int) -> None:
        if not self._slice_fit_results:
            return
        idx = value - 1
        if not (0 <= idx < len(self._slice_fit_results)):
            return
        result = self._slice_fit_results[idx]
        title = self._slice_label_from_fit_result(result)
        self.spectral_fit_canvas.show_slice_fit(result["x"], result["y"], result["y_fit"], title=title)
        self.canvas_tabs.setCurrentWidget(self.spectral_fit_canvas)

    @staticmethod
    def _slice_label_from_fit_result(result: dict) -> str:
        return f"Slice {result['index']} spectral fit"

    def _refresh_analysis_tab(self, spectrum) -> None:
        is_pseudo2d = spectrum.ndim == 2 and spectrum.is_pseudo_2d
        if is_pseudo2d:
            self.analysis_status_label.setText(
                f"Fit peaks across all {spectrum.data.shape[0]} slices, then a relaxation curve per peak."
            )
        else:
            self.analysis_status_label.setText(
                "Select a pseudo-2D spectrum (e.g. a T1/T2 relaxation series) to fit."
            )
        self._restore_analysis_state(spectrum)

    def _capture_analysis_state(self) -> dict:
        peaks = [
            [self.peak_table.item(row, col).text() if self.peak_table.item(row, col) else ""
             for col in range(len(self._PEAK_TABLE_COLUMNS))]
            for row in range(self.peak_table.rowCount())
        ]
        return {
            "peaks": peaks,
            "fit_type": self.fit_type_combo.currentIndex(),
            "model": self.relaxation_model_combo.currentIndex(),
            "delay_source": self.delay_source_combo.currentIndex(),
            "manual_text": self.manual_delays_edit.text(),
            "start": self.delay_start_spin.value(),
            "step": self.delay_step_spin.value(), "vclist_factor": self.vclist_factor_spin.value(),
            "position_tolerance": self.position_tolerance_spin.value(),
            "results_text": self.analysis_results_label.text(),
        }

    def _save_current_analysis_state(self) -> None:
        index = self._current_analysis_index
        if index is None or not (0 <= index < len(self.session)):
            return
        self.session.spectra[index].metadata.raw["analysis_state"] = self._capture_analysis_state()

    def _restore_analysis_state(self, spectrum) -> None:
        self.peak_table.setRowCount(0)
        self.analysis_results_label.setText("")

        state = spectrum.metadata.raw.get("analysis_state")
        if state is None:
            self.fit_type_combo.setCurrentIndex(0)
            self.delay_source_combo.setCurrentIndex(0)
            self.manual_delays_edit.clear()
            self.delay_start_spin.setValue(0.0)
            self.delay_step_spin.setValue(0.1)
            self.vclist_factor_spin.setValue(1.0)
            self.position_tolerance_spin.setValue(0.5)
        else:
            for shift, fwhm, eta in state["peaks"]:
                row = self.peak_table.rowCount()
                self.peak_table.insertRow(row)
                for col, value in enumerate((shift, fwhm, eta)):
                    self.peak_table.setItem(row, col, QTableWidgetItem(value))
            self.fit_type_combo.setCurrentIndex(state["fit_type"])
            self._populate_fit_model_combo()
            self.relaxation_model_combo.setCurrentIndex(state["model"])
            self.delay_source_combo.setCurrentIndex(state["delay_source"])
            self.manual_delays_edit.setText(state["manual_text"])
            self.delay_start_spin.setValue(state["start"])
            self.delay_step_spin.setValue(state["step"])
            self.vclist_factor_spin.setValue(state["vclist_factor"])
            self.position_tolerance_spin.setValue(state.get("position_tolerance", 0.5))
            self.analysis_results_label.setText(state["results_text"])

        self._slice_fit_results = spectrum.metadata.raw.get("slice_fit_results")
        self.view_slice_fit_spin.blockSignals(True)
        if self._slice_fit_results:
            self.view_slice_fit_spin.setEnabled(True)
            self.view_slice_fit_spin.setMaximum(max(1, len(self._slice_fit_results)))
            self.view_slice_fit_spin.setValue(1)
        else:
            self.view_slice_fit_spin.setEnabled(False)
        self.view_slice_fit_spin.blockSignals(False)

    # ============================================================
    # Selection handling (shared by both tabs)
    # ============================================================

    def _on_selection_changed(self, row: int) -> None:
        if row < 0 or row >= len(self.session):
            return
        self._save_current_analysis_state()
        spectrum = self.session.spectra[row]
        # Refresh widget state (slice index range/default, etc.) before
        # drawing -- the FID-preview branch below reads e.g. slice_spin,
        # which must already reflect *this* spectrum, not whatever was
        # last selected.
        self._refresh_selection_display(row)
        self._draw_selected_spectrum(spectrum)
        self._current_analysis_index = row

    def _draw_selected_spectrum(self, spectrum) -> None:
        """The actual draw step behind a selection change -- also used
        standalone (e.g. by the pseudo-2D view combo) when the
        selection itself hasn't changed and going through
        :func:`_refresh_selection_display` again would wrongly reset
        widgets (like that same combo) back to their per-selection
        defaults."""
        is_pseudo2d_stack = (
            spectrum.ndim == 2 and spectrum.is_pseudo_2d and self.pseudo2d_view_combo.currentIndex() == 1
        )
        # The FID tab is a live view of whatever's selected, not an
        # opt-in overlay (no more checkbox) -- if it's already the
        # active tab, a reselection keeps it active and refreshes it,
        # rather than force-switching away to the Spectrum tab.
        fid_tab_active = self.canvas_tabs.currentWidget() is self.fid_canvas
        try:
            if is_pseudo2d_stack:
                self._show_pseudo2d_stack(spectrum)
            else:
                self.canvas.show_spectrum(spectrum)
                if not fid_tab_active:
                    self.canvas_tabs.setCurrentWidget(self.canvas)
            if fid_tab_active and self._fid_preview_available(spectrum):
                self._show_fid_preview(spectrum, switch_tab=False)
        except Exception as exc:  # noqa: BLE001 -- surfaced to the user, not swallowed
            QMessageBox.critical(self, "Failed to plot spectrum", str(exc))

        self._silo_other_canvas_tabs(spectrum)

    def _silo_other_canvas_tabs(self, spectrum) -> None:
        """Keep the Fit/Spectral Fit/Compare canvas tabs from showing a
        previously-selected spectrum's stale plot -- they aren't
        redrawn by the Spectrum/FID branches above, so without this a
        reselection would leave e.g. the Fit tab showing another
        experiment's relaxation curve."""
        fit_results = spectrum.metadata.raw.get("relaxation_fit_results")
        if fit_results:
            self.fit_canvas.show_relaxation_fit(fit_results)
            self.fit_canvas.ax.set_xlabel(spectrum.metadata.raw.get("relaxation_fit_xlabel", "Delay (s)"))
            self.fit_canvas.canvas.draw_idle()
        else:
            self.fit_canvas.clear()

        slice_fit_results = spectrum.metadata.raw.get("slice_fit_results")
        if slice_fit_results:
            result = slice_fit_results[0]  # matches view_slice_fit_spin resetting to 1 on reselect
            title = self._slice_label_from_fit_result(result)
            self.spectral_fit_canvas.show_slice_fit(result["x"], result["y"], result["y_fit"], title=title)
        else:
            self.spectral_fit_canvas.clear()

        # The Compare tab plots whatever is multi-selected in the
        # Spectra list, not this single spectrum -- it has no
        # per-spectrum state to restore, so just drop the stale plot
        # rather than show a comparison involving a spectrum that may
        # no longer even be part of the current selection.
        self.compare_canvas.clear()

    def _refresh_selection_display(self, row: int) -> None:
        spectrum = self.session.spectra[row]
        meta = spectrum.metadata

        bits = [f"Vendor: {meta.vendor}", f"Nucleus: {meta.nucleus or '—'}"]
        if meta.sfo1:
            bits.append(f"SFO1: {meta.sfo1:.4f} MHz")
        if meta.nucleus_indirect:
            bits.append(f"Indirect nucleus: {meta.nucleus_indirect}")
        if meta.mass:
            bits.append(f"Sample mass: {meta.mass:g} g")
        self.info_label.setText("\n".join(bits))

        self.pseudo_2d_checkbox.setVisible(spectrum.ndim == 2)
        self.pseudo_2d_checkbox.blockSignals(True)
        self.pseudo_2d_checkbox.setChecked(spectrum.is_pseudo_2d)
        self.pseudo_2d_checkbox.blockSignals(False)

        is_pseudo2d_now = spectrum.ndim == 2 and spectrum.is_pseudo_2d
        self.pseudo2d_view_combo.setVisible(is_pseudo2d_now)
        self.pseudo2d_view_combo.blockSignals(True)
        self.pseudo2d_view_combo.setCurrentIndex(0)  # a fresh selection always starts on Contour
        self.pseudo2d_view_combo.blockSignals(False)

        self._refresh_fid_processing_controls(spectrum, row)
        self._refresh_analysis_tab(spectrum)

        # Which Dimension controls apply depends on the spectrum type --
        # a fresh selection always starts from slice 1 / F2, matching
        # the phase controls' own reset-on-selection below.
        has_dimension_controls = spectrum.ndim == 2 and spectrum.data_imag is not None
        is_pseudo2d = has_dimension_controls and spectrum.is_pseudo_2d
        is_2d = has_dimension_controls and not spectrum.is_pseudo_2d
        self.dimension_box.setVisible(is_pseudo2d or is_2d)
        self.slice_row_label.setVisible(is_pseudo2d)
        self.slice_spin.setVisible(is_pseudo2d)
        self.apply_all_slices_checkbox.setVisible(is_pseudo2d)
        self.dim_row_label.setVisible(is_2d)
        self.dim_combo.setVisible(is_2d)

        if is_pseudo2d:
            self.slice_spin.blockSignals(True)
            self.slice_spin.setMaximum(max(1, spectrum.data.shape[0]))
            self.slice_spin.setValue(1)
            self.slice_spin.blockSignals(False)
            self.apply_all_slices_checkbox.blockSignals(True)
            self.apply_all_slices_checkbox.setChecked(False)
            self.apply_all_slices_checkbox.blockSignals(False)
        if is_2d:
            self.dim_combo.blockSignals(True)
            self.dim_combo.setCurrentIndex(0)
            self.dim_combo.blockSignals(False)

        self._capture_phase_reference()
