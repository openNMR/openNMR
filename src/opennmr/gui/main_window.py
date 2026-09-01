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
import json
import os
import re

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QActionGroup, QKeySequence, QPalette
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDialog, QDialogButtonBox,
    QDoubleSpinBox, QFileDialog, QFormLayout, QFrame, QGroupBox, QHBoxLayout,
    QHeaderView, QInputDialog, QLabel, QLineEdit, QListWidget, QListWidgetItem,
    QMainWindow, QMenu, QMessageBox, QPushButton, QScrollArea, QSlider, QSpinBox, QSplitter,
    QTableWidget, QTableWidgetItem, QTabWidget, QToolButton, QVBoxLayout, QWidget,
)

from .. import io
from .. import plotting
from ..core import registry as core_registry
from ..core.spectrum import NMRAxis, NMRSpectrum
from ..io import bruker as io_bruker
from ..io import persistence as io_persistence
from ..processing import arithmetic as processing_arithmetic
from ..processing import fitting as processing_fitting
from ..processing import ft as processing_ft
from ..processing import relaxation as processing_relaxation
from ..processing import slicing as processing_slicing
from ..processing import vocs as processing_vocs
from ..processing.slicing import get_trace
from .session import Session
from .widgets.canvas import MplCanvas
from .widgets.drag_control import DragHandle
from .widgets.relative_spinbox import RelativeDoubleSpinBox

#: Genuinely unbounded spinbox range for phase controls -- a slider's
#: fixed pixel-to-range mapping would force a tradeoff between coarse
#: and fine control; DragHandle (a drag-to-adjust control, not a
#: slider) doesn't need one, so there's no reason to cap the total
#: accumulated phase (a joint-phased or heavily-multi-cycle-wrapped
#: correction can genuinely exceed the previous +/-36000 cap).
_PHASE_SPIN_RANGE = float("inf")

#: Phase 1's drag-sensitivity/step, relative to Phase 0's -- a linear
#: (whole-spectrum) phase correction needs much bigger raw-degree
#: changes than a zero-order one to have a visible effect.
_PHASE1_STEP_MULTIPLIER = 20.0

#: (display label, ``processing.ft`` window key) shown in the FID
#: processing box's window combo -- "none" isn't a real key in
#: ``ft._WINDOWS``, it's translated to a no-op exponential (lb=0) by
#: ``MainWindow._resolve_window_kwargs``.
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
    ("Stejskal-Tanner, biexponential (Da/Db)", "stejskal_tanner_biexponential"),
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
        self.setAcceptDrops(True)  # drag a data folder straight in -- see dropEvent

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
        # Which session index _phase_mode/_phase_reference_full above
        # actually belong to -- needed by _commit_pending_phase_to_metadata,
        # called at the *start* of the next _capture_phase_reference,
        # by which point self._selected_index() already points at the
        # spectrum being switched *to*, not the one whose pending phase
        # delta needs committing.
        self._phase_target_index: int | None = None
        # The pivot (ppm) the last-applied Phase 0/Phase 1 numbers are
        # actually measured from -- see _on_pivot_spin_changed, which
        # compensates Phase 0 whenever the pivot moves so the visible
        # correction stays continuous instead of jumping.
        self._pivot_ppm_reference: float | None = None
        # (phase0, phase1) immediately before the last Autophase click
        # -- lets "Revert Autophase" undo it. None before any
        # Autophase click, or once reselection/reverting invalidates
        # it (see _capture_phase_reference).
        self._last_autophase_revert: tuple[float, float] | None = None
        # Which spectrum's Analysis-tab state (peak list, fit type/
        # model/values source, results) is currently reflected in the
        # tab's widgets -- saved into that spectrum's own
        # metadata.raw before switching to a different one, and
        # restored (or reset to fresh defaults) for the newly
        # selected spectrum, so peak lists/fit results stay siloed per
        # spectrum rather than lingering across a selection change.
        self._current_analysis_index: int | None = None
        # Maps compare_list's own row index -> that row's position among
        # the spectra actually plotted in compare_canvas (checked rows
        # only, further filtered for 1D-ness/sample mass) -- lets
        # compare_list's own current-row selection highlight the
        # matching row directly in the plot too (see
        # _on_compare_list_current_row_changed), not just in the list.
        self._compare_plot_row_positions: dict[int, int] = {}
        # True when compare_list's contents no longer match what
        # compare_canvas last actually rendered (see
        # _refresh_compare_view's deferred-render path) -- checked by
        # _render_compare_view_if_stale, called whenever Compare
        # becomes the visible tab.
        self._compare_view_stale = False
        # Debounces _recompute_multi_row_data (pseudo-2D "apply to
        # every slice" / conventional 2D -- a real per-row FID-domain
        # recompute, genuinely slow for a real multi-hundred-row
        # dataset, ~1-2ms/row) so a rapid run of control changes (a
        # spin box's own repeat-on-hold, or several edits in quick
        # succession) coalesces into one recompute a short idle period
        # after the last one, instead of running the whole expensive
        # recompute again on every single intermediate change and
        # backing up the Qt event queue -- indistinguishable from a
        # hang from the user's own perspective. Explicit user report:
        # spline baseline correction (the single most expensive of
        # these per-row steps) "hangs" on real multi-row data -- this
        # is the fix, not making the per-row work itself faster (see
        # _on_fid_processing_control_changed/_on_baseline_correct_toggled).
        self._live_reprocess_timer = QTimer(self)
        self._live_reprocess_timer.setSingleShot(True)
        self._live_reprocess_timer.timeout.connect(self._run_debounced_multi_row_reprocess)
        self._live_reprocess_pending_index: int | None = None
        # Same debounce reasoning as _live_reprocess_timer just above,
        # for a different job: writing this experiment's own sidecar
        # files (see io.persistence) to disk after any live control/
        # metadata change, so the saved recipe survives an app restart
        # -- explicit user request. A longer interval than the
        # recompute timer's own 200ms since a disk write matters far
        # less to coalesce tightly (nothing on screen depends on it
        # finishing promptly the way a recompute's own visible result
        # does) and a rapid burst of edits (e.g. a spin box held down)
        # would otherwise mean a much higher write rate than the
        # recompute timer already limits itself to.
        self._sidecar_save_timer = QTimer(self)
        self._sidecar_save_timer.setSingleShot(True)
        self._sidecar_save_timer.timeout.connect(self._run_debounced_sidecar_save)
        self._sidecar_save_pending_index: int | None = None
        # Each spectrum's own load kwargs immediately after _load's
        # initial pseudo-2D-default handling, before any live
        # FID-processing/phase/baseline control ever touched it --
        # what the Reset button (see _on_reset_processing_clicked)
        # reverts back to. Keyed by session index (not spectrum
        # identity: reload/reset both replace the spectrum object at
        # that index, so the index is the stable handle here, same as
        # Session's own _load_info).
        self._original_load_kwargs: dict[int, dict] = {}
        # The per-slice pseudo-Voigt fit results from the most recent
        # Analysis-tab fit (NMR2D_fit_slices's own "rows" output) --
        # lets "Spectrum fit (slice #)" browse the intermediate spectral
        # fits that fed the T1/T2/D parameter fit, not just its final number.
        self._slice_fit_results: list[dict] | None = None
        # The per-peak relaxation/diffusion curve-fit results from the
        # most recent Fit Curve run (NMR2D_fit_relaxation's own list
        # output, one entry per peak) -- lets "View curve fit (peak #)" browse
        # them one at a time in the Fit tab, and its own x-axis label
        # (Delay (s) vs. b-value (s/m^2) -- resolved once when Fit
        # Curve runs, not re-derived per peak).
        self._relaxation_fit_results: list[dict] | None = None
        self._relaxation_fit_xlabel: str = "Delay (s)"
        # The *original* (every point included) fit result for each
        # peak, from the same Fit Curve run as _relaxation_fit_results
        # -- kept separately so "Reset Excluded Points" can restore
        # exactly what a point-exclusion re-fit (see
        # _on_fit_point_exclude_requested) started from, without
        # re-running Fit Curve from scratch. Same list positions as
        # _relaxation_fit_results (one entry per peak).
        self._relaxation_fit_original: list[dict] | None = None
        # {peak_index: {excluded array-indices into that peak's own
        # _relaxation_fit_original[peak_index]["t"]/["y"]}} -- which
        # points the user has right-click-excluded from each peak's
        # curve fit, siloed per peak so switching "View curve fit
        # (peak #)" doesn't lose or mix up one peak's exclusions with another's.
        self._excluded_relaxation_indices: dict[int, set[int]] = {}

        # One MplCanvas per kind of plot, each keeping its own figure/
        # zoom/pan state, navigable via canvas_tabs instead of every
        # kind clobbering a single shared canvas. ``self.canvas`` is
        # the main "Spectrum" view (1D / 2D contour / pseudo-2D slice
        # or stack) -- unqualified since it's the default/most-used
        # one and predates the other three.
        # full_resolution=True: explicit user feedback that the main
        # Spectrum view was silently, severely downsampled by default
        # with no obvious way to tell -- every other canvas (e.g. the
        # FID-processing live preview below) keeps the default fast/
        # approximate rendering, which is the right tradeoff there.
        self.canvas = MplCanvas(full_resolution=True)
        self.canvas.peakPicked.connect(self._on_peak_picked)
        self.canvas.regionSelected.connect(self._on_region_selected)
        self.canvas.referenceShiftRequested.connect(self._on_reference_shift_from_plot)
        self.fid_canvas = MplCanvas()
        self.compare_canvas = MplCanvas()
        self.fit_canvas = MplCanvas()
        self.fit_canvas.fitPointExcludeRequested.connect(self._on_fit_point_exclude_requested)
        self.fit_canvas.fitPointsResetRequested.connect(self._on_fit_points_reset_requested)
        # A single-slice pseudo-Voigt spectral fit (data + fit curve,
        # from "Spectrum fit (slice #)") gets its own tab rather than sharing
        # the Spectrum canvas -- it used to temporarily overwrite the
        # normal spectrum view there, which meant losing your place.
        self.spectral_fit_canvas = MplCanvas()
        # The Processing tab's own live dual view (FID top, downsampled
        # spectrum bottom) -- see _show_processing_preview. Distinct
        # from fid_canvas/self.canvas on purpose (explicit user
        # feedback): those two should only ever show the actually-
        # loaded spectrum, not a live preview of a not-yet-applied
        # FID-processing choice.
        self.processing_preview_canvas = MplCanvas()

        self.canvas_tabs = QTabWidget()
        self.canvas_tabs.addTab(self.fid_canvas, "FID")
        self.canvas_tabs.addTab(self.processing_preview_canvas, "Processing Preview")
        self.canvas_tabs.addTab(self.canvas, "Spectrum")
        self.canvas_tabs.addTab(self.spectral_fit_canvas, "Spectral Fit")
        self.canvas_tabs.addTab(self.fit_canvas, "Fit")
        self.canvas_tabs.addTab(self.compare_canvas, "Compare")
        # No more "Preview as FID" checkbox -- clicking the FID tab
        # itself is what shows/refreshes the current spectrum's FID.
        self.canvas_tabs.currentChanged.connect(self._on_canvas_tab_changed)
        # Fit only ever has something to show for a pseudo-2D spectrum
        # (or a Compare-stack fit, which force-shows it directly --
        # see _run_curve_fit); Compare only ever has something once its
        # own list (left-hand Compare tab) has an entry; Processing
        # Preview only has something for a FID-reprocessable spectrum
        # (see _update_processing_preview_tab_visibility) -- all three
        # start hidden and are kept in sync rather than always shown
        # empty.
        self.canvas_tabs.setTabVisible(self.canvas_tabs.indexOf(self.fit_canvas), False)
        self.canvas_tabs.setTabVisible(self.canvas_tabs.indexOf(self.compare_canvas), False)
        self.canvas_tabs.setTabVisible(self.canvas_tabs.indexOf(self.processing_preview_canvas), False)
        # FID was added first (so it appears leftmost in the tab bar),
        # but Spectrum is the actual default/most-used view (see
        # self.canvas's own docstring above) -- without this, a plain
        # QTabWidget starts on whichever tab was added first, so a
        # fresh window (or a first load whose spectrum has no FID to
        # preview) would show an empty FID tab instead of the spectrum
        # that was just drawn to self.canvas.
        self.canvas_tabs.setCurrentWidget(self.canvas)

        self.side_tabs = tabs = QTabWidget()
        tabs.addTab(self._scrollable(self._build_spectra_tab()), "Data")
        self._processing_tab_index = tabs.addTab(self._scrollable(self._build_processing_tab()), "Processing")
        tabs.addTab(self._scrollable(self._build_analysis_tab()), "Analysis")
        tabs.addTab(self._scrollable(self._build_compare_tab()), "Compare")
        tabs.addTab(self._scrollable(self._build_output_tab()), "Output")
        # Entering the Processing tab switches the canvas to its own
        # live dual preview automatically (see _show_processing_preview);
        # leaving it switches back, so that speculative preview state
        # doesn't linger as the shown view once you're no longer
        # editing it.
        tabs.currentChanged.connect(self._on_side_tab_changed)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(tabs)
        splitter.addWidget(self.canvas_tabs)
        # Both panels now get a share of any extra space from resizing
        # the window larger (canvas 3x the left panel's) -- previously
        # only the canvas did (stretch factor 0 is the QSplitter
        # default), which left the left panel pinned at its initial
        # ~300px forever, however much the window grew: proportionally
        # shrinking it down to a sliver on a large/maximized window,
        # where its own content (QFormLayout rows of labels + combo/
        # spin boxes) has nowhere to go but overflow past the
        # scrollbar-disabled QScrollArea's viewport edge (see
        # _scrollable's own docstring for why that scrollbar is off).
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        splitter.setSizes([340, 760])
        # A dragged-too-narrow left panel hits the same overflow --
        # floor it at a usable width. Still fully draggable/resizable
        # by the user (a QSplitter handle's own drag isn't affected by
        # a child widget's minimumWidth -- Qt just won't let the user
        # drag past it), matching "with an option still to change".
        tabs.setMinimumWidth(280)

        self.setCentralWidget(splitter)
        self._set_processing_enabled(False)
        self.process_indirect_button.setVisible(False)
        self.pseudo2d_view_combo.setVisible(False)
        self.dimension_box.setVisible(False)

        self._build_menu_bar()

    def _build_menu_bar(self) -> None:
        """A real top-level menu bar -- native on macOS (the app's own
        menu bar at the screen top, "openNMR" auto-named as the
        Application menu), in-window on Windows/Linux; Qt handles the
        platform difference automatically from one QMenuBar. Every
        action here already exists as a button/control elsewhere --
        this adds menu/keyboard-shortcut access to the ones that make
        sense used from anywhere, not a parallel command set."""
        menu_bar = self.menuBar()

        file_menu = menu_bar.addMenu("&File")
        open_action = QAction("&Open…", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self._on_open_clicked)
        file_menu.addAction(open_action)

        open_files_action = QAction("Open &File(s)…", self)
        open_files_action.setShortcut(QKeySequence("Ctrl+Shift+O"))
        open_files_action.setToolTip(
            "For vendors whose data is a single file rather than a "
            "folder (e.g. X-Pulse, JEOL) -- pick one or more files "
            "directly."
        )
        open_files_action.triggered.connect(self._on_open_files_clicked)
        file_menu.addAction(open_files_action)

        save_figure_action = QAction("&Save Figure As…", self)
        save_figure_action.setShortcut(QKeySequence.StandardKey.Save)
        save_figure_action.setToolTip("Save the currently visible canvas tab as an image.")
        save_figure_action.triggered.connect(self._on_save_figure_clicked)
        file_menu.addAction(save_figure_action)

        file_menu.addSeparator()
        quit_action = QAction("&Quit", self)
        quit_action.setShortcut(QKeySequence.StandardKey.Quit)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        edit_menu = menu_bar.addMenu("&Edit")
        remove_action = QAction("&Remove Selected Spectrum", self)
        remove_action.setShortcut(QKeySequence.StandardKey.Delete)
        # WidgetShortcut (rather than the QAction default, WindowShortcut)
        # confines the Delete/Backspace *keyboard* accelerator to firing
        # only while the Data tab's own list has focus -- previously it
        # fired window-wide, so Backspace pressed anywhere (e.g. while
        # browsing the Analysis tab) deleted the selected spectrum, easy
        # to trigger by accident. The menu item itself still works from
        # anywhere via a click, since that doesn't go through the
        # shortcut-context check at all.
        remove_action.setShortcutContext(Qt.ShortcutContext.WidgetShortcut)
        remove_action.triggered.connect(self._on_remove_clicked)
        self.spectrum_list.addAction(remove_action)
        edit_menu.addAction(remove_action)

        remove_peak_action = QAction("Remove &Peak", self)
        remove_peak_action.setShortcut(QKeySequence.StandardKey.Delete)
        remove_peak_action.setShortcutContext(Qt.ShortcutContext.WidgetShortcut)
        remove_peak_action.triggered.connect(self._on_remove_peak_clicked)
        self.peak_table.addAction(remove_peak_action)

        edit_menu.addSeparator()
        reset_phase_action = QAction("Reset &Phase", self)
        reset_phase_action.triggered.connect(self._on_reset_phase_clicked)
        edit_menu.addAction(reset_phase_action)

        autophase_action = QAction("&Autophase", self)
        autophase_action.triggered.connect(self._on_autophase_clicked)
        edit_menu.addAction(autophase_action)

        window_menu = menu_bar.addMenu("&Window")
        minimize_action = QAction("&Minimize", self)
        minimize_action.setShortcut(QKeySequence("Ctrl+M"))
        minimize_action.triggered.connect(self.showMinimized)
        window_menu.addAction(minimize_action)
        window_menu.addSeparator()

        window_menu.addSection("Panel")
        shortcut_num = 1
        for idx, label in enumerate(("Data", "Processing", "Analysis", "Compare")):
            action = QAction(label, self)
            action.setShortcut(QKeySequence(f"Ctrl+{shortcut_num}"))
            action.triggered.connect(lambda _checked=False, i=idx: self.side_tabs.setCurrentIndex(i))
            window_menu.addAction(action)
            shortcut_num += 1

        # " View" suffix: the canvas tab's own name would otherwise
        # collide with the identically-named Panel entry above it for
        # "Compare" (a left-panel tab *and* a canvas view) -- distinct
        # labels throughout is clearer than relying on the section
        # headers alone to disambiguate.
        window_menu.addSection("View")
        for canvas_widget in (
            self.canvas, self.fid_canvas, self.processing_preview_canvas,
            self.spectral_fit_canvas, self.fit_canvas, self.compare_canvas,
        ):
            label = self.canvas_tabs.tabText(self.canvas_tabs.indexOf(canvas_widget)) + " View"
            action = QAction(label, self)
            action.setShortcut(QKeySequence(f"Ctrl+{shortcut_num}"))
            action.triggered.connect(lambda _checked=False, w=canvas_widget: self.canvas_tabs.setCurrentWidget(w))
            window_menu.addAction(action)
            shortcut_num += 1

        # Quick access to the Analysis tab's own controls from anywhere
        # -- no new capability, every action here already exists as a
        # button/combo in the Spectral Fitting / Curve Fitting sub-tabs;
        # this just saves switching to that side-tab first.
        analysis_menu = menu_bar.addMenu("&Analysis")
        fit_spectrum_menu_action = QAction("Fit &Spectrum", self)
        fit_spectrum_menu_action.triggered.connect(self._on_fit_spectrum_clicked)
        analysis_menu.addAction(fit_spectrum_menu_action)

        auto_pick_menu_action = QAction("&Auto-Pick Peaks", self)
        auto_pick_menu_action.triggered.connect(self._on_auto_pick_peaks_clicked)
        analysis_menu.addAction(auto_pick_menu_action)

        self.pick_peak_menu_action = QAction("Pick Pea&k from Plot", self)
        self.pick_peak_menu_action.setCheckable(True)
        self.pick_peak_menu_action.toggled.connect(self.pick_peak_button.setChecked)
        self.pick_peak_button.toggled.connect(self.pick_peak_menu_action.setChecked)
        analysis_menu.addAction(self.pick_peak_menu_action)

        analysis_menu.addSeparator()
        clear_peaks_menu_action = QAction("&Clear Peak Table", self)
        clear_peaks_menu_action.triggered.connect(self._on_clear_peaks_clicked)
        analysis_menu.addAction(clear_peaks_menu_action)

        analysis_menu.addSeparator()
        integrate_regions_menu_action = QAction("&Integrate Regions", self)
        integrate_regions_menu_action.triggered.connect(self._on_integrate_regions_clicked)
        analysis_menu.addAction(integrate_regions_menu_action)

        self.select_region_menu_action = QAction("Select &Region from Plot", self)
        self.select_region_menu_action.setCheckable(True)
        self.select_region_menu_action.setShortcut(QKeySequence("Ctrl+I"))
        self.select_region_menu_action.toggled.connect(self.select_region_button.setChecked)
        self.select_region_button.toggled.connect(self.select_region_menu_action.setChecked)
        analysis_menu.addAction(self.select_region_menu_action)

        clear_regions_menu_action = QAction("Clear Re&gion Table", self)
        clear_regions_menu_action.triggered.connect(self._on_clear_regions_clicked)
        analysis_menu.addAction(clear_regions_menu_action)

        fitting_menu = menu_bar.addMenu("Fittin&g")
        fit_curve_menu_action = QAction("Fit &Curve…", self)
        fit_curve_menu_action.triggered.connect(self._on_fit_curve_button_clicked)
        fitting_menu.addAction(fit_curve_menu_action)
        fitting_menu.addSeparator()

        # Fit type/model radio-style submenus mirror fit_type_combo/
        # relaxation_model_combo -- picking one here just sets the
        # combo's index and lets its existing handler do the real work,
        # so this menu can never drift out of sync with what those
        # handlers actually do. Checked state instead of being kept
        # live is refreshed on demand (aboutToShow) since the model
        # list itself changes with fit type and with which spectrum is
        # selected (each spectrum silos its own Analysis-tab state).
        self._fit_type_menu_group = QActionGroup(self)
        self._fit_type_menu_group.setExclusive(True)
        fit_type_submenu = fitting_menu.addMenu("Fit &Type")
        for idx in range(self.fit_type_combo.count()):
            action = QAction(self.fit_type_combo.itemText(idx), self)
            action.setCheckable(True)
            action.triggered.connect(lambda _checked=False, i=idx: self.fit_type_combo.setCurrentIndex(i))
            self._fit_type_menu_group.addAction(action)
            fit_type_submenu.addAction(action)

        self._model_menu = fitting_menu.addMenu("&Model")
        self._model_menu_group = QActionGroup(self)
        self._model_menu_group.setExclusive(True)
        self._rebuild_model_menu()

        fitting_menu.aboutToShow.connect(self._sync_fitting_menu_checks)

    def _rebuild_model_menu(self) -> None:
        self._model_menu.clear()
        for action in list(self._model_menu_group.actions()):
            self._model_menu_group.removeAction(action)
        for idx in range(self.relaxation_model_combo.count()):
            action = QAction(self.relaxation_model_combo.itemText(idx), self)
            action.setCheckable(True)
            action.triggered.connect(lambda _checked=False, i=idx: self.relaxation_model_combo.setCurrentIndex(i))
            self._model_menu_group.addAction(action)
            self._model_menu.addAction(action)

    def _sync_fitting_menu_checks(self) -> None:
        self._rebuild_model_menu()
        fit_type_actions = self._fit_type_menu_group.actions()
        if 0 <= self.fit_type_combo.currentIndex() < len(fit_type_actions):
            fit_type_actions[self.fit_type_combo.currentIndex()].setChecked(True)
        model_actions = self._model_menu_group.actions()
        if 0 <= self.relaxation_model_combo.currentIndex() < len(model_actions):
            model_actions[self.relaxation_model_combo.currentIndex()].setChecked(True)

    def _on_save_figure_clicked(self) -> None:
        """File > Save Figure As… -- delegates to whichever canvas tab
        is currently visible's own matplotlib toolbar save dialog
        (format/location chosen there), rather than reimplementing it.
        That toolbar's own save_figure is itself overridden (see
        widgets.canvas._Toolbar) to always export at full resolution
        regardless of whether the live view is currently downsampled,
        so nothing extra is needed here for that."""
        current = self.canvas_tabs.currentWidget()
        if current is not None and hasattr(current, "toolbar"):
            current.toolbar.save_figure()

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
        absorbed here, not propagated up to the window -- but only
        *vertical* growth is meant to be absorbed this way. Horizontal
        scrolling is deliberately turned off: an un-word-wrapped label
        (or anything else with a too-wide size hint) would otherwise
        silently make the tab horizontally scrollable instead of
        wrapping/shrinking to the sidebar's actual width, which is a
        worse failure mode (content quietly cut off to the side,
        easy to miss) than the tall-window ratchet this was built to
        fix in the first place -- forcing content to actually fit
        within whatever width the splitter gives this side panel (see
        __init__'s ``splitter.setSizes``) surfaces a wrapping bug
        immediately instead of hiding it behind a scrollbar."""
        scroll = QScrollArea()
        scroll.setWidget(widget)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        return scroll

    @staticmethod
    def _narrow_combo(combo: QComboBox, chars: int = 14) -> None:
        """Stop a long item's text (e.g. "Biexponential recovery
        (T1a/T1b)", "Relative gradient (valueslist)") from setting this
        combo box's own minimum width -- Qt's default
        ``AdjustToContentsOnFirstShow`` sizes the *closed* box to fit
        whatever the widest item happened to be the first time it was
        shown, and never shrinks it back down again even inside a
        narrower container, which is exactly the kind of fixed-content
        width ``_scrollable``'s disabled horizontal scrollbar (see its
        own docstring) can't route around -- it surfaces as clipped/
        overflowing content instead. The dropdown popup itself is
        unaffected -- still shows every item at its own full width."""
        combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        combo.setMinimumContentsLength(chars)

    @staticmethod
    def _spin_with_step_buttons(spin: QSpinBox) -> tuple[QHBoxLayout, QPushButton, QPushButton]:
        """Wrap ``spin`` with Prev/Next (◀/▶) buttons either side --
        quicker than the spin box's own tiny arrows or typing a number
        for stepping through a series one at a time (originally built
        for the pseudo-2D slice picker while phasing each slice in
        turn; reused wherever else browsing a numbered series one step
        at a time comes up, e.g. "View curve fit (peak #)"). ``setValue`` clamps
        to the spin box's own range on its own, so both buttons are
        safe no-ops at either end. Returns ``(row, prev_button,
        next_button)`` -- the caller stores the buttons as its own
        attributes if anything (e.g. a test) needs to click them
        directly."""
        prev_button = QPushButton("◀")
        prev_button.setFixedWidth(28)
        prev_button.setToolTip("Previous")
        prev_button.clicked.connect(lambda: spin.setValue(spin.value() - 1))
        next_button = QPushButton("▶")
        next_button.setFixedWidth(28)
        next_button.setToolTip("Next")
        next_button.clicked.connect(lambda: spin.setValue(spin.value() + 1))

        row = QHBoxLayout()
        row.addWidget(prev_button)
        row.addWidget(spin)
        row.addWidget(next_button)
        return row, prev_button, next_button

    @staticmethod
    def _collapsible_section(title: str, content: QWidget, expanded: bool = False) -> tuple[QWidget, QToolButton]:
        """A disclosure toggle wrapping ``content`` -- collapsed by
        default, so a side panel's rarely-touched tuning controls
        don't count toward the vertical space a casual user's default
        view needs (explicit user feedback: the old single Spectral
        Fitting sub-tab crammed peak-picking, integration, and per-fit
        tolerance/S-N tuning into one always-visible stack -- "too
        much information at once for casual users"). Qt has no native
        collapsible group box; built from a checkable QToolButton
        (its arrow indicator flips open/closed) driving ``content``'s
        own visibility. Returns ``(container, toggle_button)`` -- the
        caller stores the button itself only if a test needs to click
        it directly, same convention as :meth:`_spin_with_step_buttons`.
        A test checking the collapsed/expanded state should assert on
        ``content.isHidden()`` (this project's own established
        convention, e.g. ``fid_processing_box``), not ``isVisible()``
        -- the latter is always ``False`` in these headless tests
        regardless of this toggle, since the window itself is never
        actually shown; only ``content``'s own explicit flag, not
        composed ancestor visibility, is meaningful here."""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

        toggle = QToolButton()
        toggle.setText(title)
        toggle.setCheckable(True)
        toggle.setChecked(expanded)
        toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        toggle.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)
        toggle.setStyleSheet("QToolButton { border: none; font-weight: bold; }")
        content.setVisible(expanded)

        def _on_toggled(checked: bool) -> None:
            toggle.setArrowType(Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow)
            content.setVisible(checked)

        toggle.toggled.connect(_on_toggled)
        layout.addWidget(toggle)
        layout.addWidget(content)
        return container, toggle

    # ============================================================
    # Data tab: open / list / remove / nucleus override
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

        open_files_button = QPushButton("Open File(s)…")
        open_files_button.setToolTip(
            "For vendors whose data is a single file rather than a "
            "folder (e.g. X-Pulse, JEOL) -- pick one or more files "
            "directly."
        )
        open_files_button.clicked.connect(self._on_open_files_clicked)
        layout.addWidget(open_files_button)

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

        self.process_indirect_button = QPushButton("Process Indirect Dimension")
        self.process_indirect_button.setCheckable(True)
        self.process_indirect_button.setToolTip(
            "Pseudo-2D (F1 is an index/slice dimension -- a relaxation "
            "delay or diffusion gradient series, no FFT along F1) is the "
            "default for every 2D spectrum. Click this only for a "
            "genuine 2D-correlation experiment (COSY/HSQC-style) where "
            "F1 really is a second chemical-shift dimension that needs "
            "its own FFT -- reprocesses from the original file "
            "immediately, like Align Echo does for FID processing."
        )
        self.process_indirect_button.toggled.connect(self._on_process_indirect_toggled)
        info_form.addRow(self.process_indirect_button)

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
        self._open_directory(directory)

    def _on_open_files_clicked(self) -> None:
        paths, _selected_filter = QFileDialog.getOpenFileNames(
            self, "Open one or more spectrum files",
            filter="NMR spectrum files (*.jdx *.dx *.jdf);;All files (*)",
        )
        if not paths:
            return
        self._open_files(paths)

    def _open_files(self, paths: list[str]) -> None:
        """Load each of ``paths`` directly (no folder scan) -- shared
        by the Open File(s)… dialog and drag-and-drop of individual
        files. Unlike a folder drop (which silently skips anything
        unrecognised, since OS clutter like ``.DS_Store`` can easily
        ride along), files picked one-by-one through a dialog are an
        explicit choice -- an unrecognised one is reported, not
        silently dropped."""
        unrecognised: list[str] = []
        for path in paths:
            try:
                vendor = io.detect_vendor(path)
            except ValueError:
                unrecognised.append(path)
                continue
            self._load(path, vendor)

        if unrecognised:
            QMessageBox.critical(
                self, "Some files not recognised",
                "Could not auto-detect a vendor format for:\n\n"
                + "\n".join(unrecognised),
            )

    def _open_directory(self, directory: str) -> None:
        """Scan ``directory`` for spectra and load them -- shared by
        the Open… dialog and drag-and-drop (see ``dropEvent``), so a
        dropped folder behaves exactly like one picked via the dialog."""
        matches = self._scan_for_spectra(directory)
        if not matches:
            QMessageBox.critical(
                self, "No spectra found",
                f"Could not find a Bruker, X-Pulse, or Varian/Agilent "
                f"experiment under {directory!r}.",
            )
            return

        if len(matches) == 1:
            self._load(*matches[0])
            return

        for path, vendor in self._prompt_select_spectra(matches, directory):
            self._load(path, vendor)

    def dragEnterEvent(self, event) -> None:  # noqa: N802 -- Qt override
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:  # noqa: N802 -- Qt override
        paths = [url.toLocalFile() for url in event.mimeData().urls()]
        self._handle_dropped_paths([p for p in paths if p])
        event.acceptProposedAction()

    def _handle_dropped_paths(self, paths: list[str]) -> None:
        """Drag a data folder (an experiment folder itself, or a
        parent folder holding several) straight into the window,
        instead of needing the Open… dialog -- reuses the exact same
        scan/prompt/load path. A dropped single file that's itself a
        recognised spectrum (e.g. one X-Pulse/Varian file, not inside
        a folder) loads directly; anything else unrecognised is
        silently skipped rather than raising, since a drop can easily
        include the OS's own clutter (a `.DS_Store`, etc.) alongside
        the folder actually meant. Split out from ``dropEvent`` itself
        so this is testable without constructing a real ``QDropEvent``."""
        for path in paths:
            if os.path.isdir(path):
                self._open_directory(path)
            elif os.path.isfile(path):
                try:
                    vendor = io.detect_vendor(path)
                except ValueError:
                    continue
                self._load(path, vendor)

    #: X-Pulse individual-scan-repeat files, e.g.
    #: ``"...H_1_10_raw.jdx"`` -- one per scan within an averaged
    #: acquisition (there's a sibling ``..._avg.jdx`` with the actual
    #: combined spectrum), not independently useful data. Distinct
    #: from ``raw_slices/NNNN_slice.jdx`` (NUS-reconstruction slices,
    #: genuinely independent per-slice data) -- this only matches the
    #: "_raw.jdx" naming, not "_slice.jdx".
    _XPULSE_RAW_SCAN_RE = re.compile(r"_\d+_raw\.(jdx|dx)$", re.IGNORECASE)

    _NATURAL_SORT_SPLIT_RE = re.compile(r"(\d+)")

    @classmethod
    def _natural_sort_key(cls, name: str) -> list:
        """Sort key that orders ``"1", "2", ..., "10", "11"`` numerically
        instead of the lexicographic ``"1", "10", "11", "2", ...`` a
        plain string sort produces -- matches Bruker's own numbered
        experiment-folder convention (and any other numbered-folder
        vendor layout), which a parent-folder scan needs to walk in the
        order a person would expect."""
        return [int(chunk) if chunk.isdigit() else chunk.lower()
                for chunk in cls._NATURAL_SORT_SPLIT_RE.split(name)]

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
        actually worth loading) -- and, similarly, a ``raw_slices/``
        folder sitting alongside a genuine NTUPLES 2D file (e.g. an
        HSQC's own real dataset) is skipped entirely rather than
        offering its 100+ individual per-F1-increment files as
        separate matches once the one real 2D dataset they belong to
        is already found.
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
                entries = sorted(os.scandir(path), key=lambda e: cls._natural_sort_key(e.name))
            except OSError:
                return
            # A genuine X-Pulse 2D dataset (e.g. an HSQC's own
            # "..._avg.jdx" NTUPLES file) already contains everything
            # its own sibling raw_slices/ folder holds per-F1-increment
            # -- don't also offer those (128+, for a real dataset)
            # individual slice files as separate matches once the one
            # real dataset they belong to is already found here.
            # raw_slices/ folders that *aren't* siblings of an NTUPLES
            # file (e.g. a genuine per-slice pseudo-2D PFG series) are
            # untouched -- their own contents are the real dataset,
            # not a redundant duplicate of one.
            has_ntuples_sibling = any(
                entry.is_file(follow_symlinks=False)
                and os.path.splitext(entry.name)[1].lower() in (".jdx", ".dx")
                and io.xpulse._is_ntuples_2d_file(entry.path)
                for entry in entries
            )
            for entry in entries:
                if len(matches) >= max_matches:
                    return
                if entry.is_dir(follow_symlinks=False):
                    if has_ntuples_sibling and entry.name.lower() == "raw_slices":
                        continue
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
        found_label = QLabel(f"Found {len(matches)} spectra under {base_dir}:")
        found_label.setWordWrap(True)
        layout.addWidget(found_label)

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
            spectrum = self.session.add(path, vendor=vendor)
        except Exception as exc:  # noqa: BLE001 -- surfaced to the user, not swallowed
            QMessageBox.critical(self, "Failed to load spectrum", str(exc))
            return

        index = len(self.session) - 1
        # Loaded early (before the pseudo-2D default below) so a
        # previously-saved "Process Indirect Dimension" opt-in can
        # override that default rather than being silently reset by it
        # on every reopen -- see the pseudo_2d branch just below.
        saved_metadata = io_persistence.load_metadata_sidecar(spectrum.metadata.source_path)
        if spectrum.ndim == 2 and not spectrum.is_pseudo_2d:
            # Pseudo-2D is the default treatment for any 2D data,
            # regardless of what the vendor's own (known-unreliable for
            # at least Bruker T2/PFG data, see TODO.md) auto-detection
            # reports -- overridden here rather than trusting it, since
            # under-detecting pseudo-2D is the failure mode actually
            # seen on real data. Process Indirect Dimension (unchecked
            # by default after this) is how a genuine 2D-correlation
            # experiment opts back into a real F1 FFT -- or, if this
            # exact experiment already had that opt-in saved from a
            # previous session (see io.persistence), respect it here
            # instead of forcing pseudo-2D and making the user redo it
            # after every restart.
            # Only an explicit saved `False` overrides the default --
            # anything else (no saved sidecar, an older one predating
            # this field, or a saved 1D/pseudo-2D-only spectrum whose
            # own field is None) keeps today's unconditional "force
            # pseudo-2D" behaviour unchanged, rather than silently
            # switching to auto-detection.
            wants_pseudo_2d = (saved_metadata or {}).get("pseudo_2d") is not False
            try:
                spectrum = self.session.reload(index, pseudo_2d=wants_pseudo_2d)
            except Exception:  # noqa: BLE001 -- fall back to what actually loaded rather than failing the load
                pass
        # Snapshot *after* the pseudo-2D default above (its own
        # "start pseudo-2D unless told otherwise" convention is itself
        # the sensible "original" state to reset back to, not
        # whatever the vendor's own raw auto-detection produced) --
        # the Reset button's own target (see _on_reset_processing_clicked).
        self._original_load_kwargs[index] = dict(self.session.get_load_kwargs(index))
        # Baseline correction now defaults on for every vendor, not
        # just X-Pulse -- explicit user feedback that a well-phased
        # spectrum should generally land close to zero, and that this
        # wasn't reliably happening. For X-Pulse this records an
        # effect io.xpulse.read_1d already silently applied as a one-
        # off part of the initial load (linear, unless align_echo is
        # on -- see its own docstring; align_echo is always False at a
        # fresh GUI load, so that resolves to True unconditionally
        # here) -- without this, the correction already baked into
        # what's on screen would silently vanish the moment *any*
        # later control was touched, since every live control change
        # now recomputes from the raw FID via this GUI's own explicit
        # baseline toggle instead of read_1d itself. For every other
        # vendor, none of whose readers apply baseline correction at
        # load time at all, this is a genuine new default rather than
        # recording an already-applied effect -- a real, visible
        # change (a freshly-loaded spectrum now gets linear baseline
        # correction where it previously didn't), not just bookkeeping.
        spectrum.metadata.raw.setdefault("baseline_correct", True)
        spectrum.metadata.raw.setdefault("baseline_method", "linear")

        # Restore this experiment's own saved sidecar state (see
        # io.persistence), if any, *before* selecting it below -- so
        # _refresh_fid_processing_controls/_restore_analysis_state
        # (both run as part of _select_single_row's own cascade) see
        # the restored values immediately, the same way they already
        # pick up in-session-only state today. Explicit user request:
        # a saved processing recipe should survive an app restart, not
        # just a reselection within one running session. saved_metadata
        # itself was already loaded above (before the pseudo_2d default) --
        # re-used here rather than re-read, same file either way.
        if saved_metadata:
            if saved_metadata.get("mass_g") is not None:
                self.session.set_mass(index, saved_metadata["mass_g"])
                spectrum = self.session.spectra[index]
            if saved_metadata.get("nucleus"):
                self.session.set_nucleus(index, saved_metadata["nucleus"])
                spectrum = self.session.spectra[index]
            if saved_metadata.get("nucleus_indirect"):
                self.session.set_nucleus(index, saved_metadata["nucleus_indirect"], indirect=True)
                spectrum = self.session.spectra[index]
        saved_processing = io_persistence.load_processing_sidecar(spectrum.metadata.source_path)
        if saved_processing:
            spectrum.metadata.raw.update(saved_processing)

        self.spectrum_list.addItem(self.session.label(index))
        self._select_single_row(index)
        if saved_processing:
            # Seeding metadata.raw above only affects widget *values*
            # (via _refresh_fid_processing_controls, already run by
            # _select_single_row) and future persistence -- it doesn't
            # by itself touch spectrum.data, which is still whatever
            # the vendor's own default load just produced. Actually
            # apply the restored recipe now.
            self._apply_loaded_processing_sidecar(spectrum, index)

        shape_note = spectrum.metadata.raw.get("read_2d_shape_guess_failed")
        if shape_note:
            QMessageBox.information(
                self, "Loaded from processed data",
                f"{self.session.label(index)}:\n\n{shape_note}\n\nCouldn't determine the raw "
                f"ser file's shape, so this loaded from TopSpin's own already-processed data "
                f"instead -- raw-FID reprocessing (window, zero-fill, phase, ...) isn't "
                f"available for it.",
            )

        window_note = spectrum.metadata.raw.get("window_unsupported")
        if window_note:
            QMessageBox.information(
                self, "Window function not applied",
                f"{self.session.label(index)}:\n\n{window_note}\n\nThe FID was left "
                f"unwindowed rather than guessing at this window's parameters -- pick "
                f"a different one from the Processing tab and click Apply Processing, "
                f"or leave it unwindowed.",
            )

    def _apply_loaded_processing_sidecar(self, spectrum, index: int) -> None:
        """Actually apply a just-restored processing-parameters
        sidecar to ``spectrum.data`` itself -- called once, right
        after :meth:`_load` has already merged the saved dict into
        ``spectrum.metadata.raw`` and selected the row (so
        ``_refresh_fid_processing_controls`` has already set every
        live-control widget from it, and ``_capture_phase_reference``
        has already zeroed the Phase spin boxes against it). Merging
        into ``metadata.raw`` alone only changes widget *values* and
        future persistence -- ``spectrum.data`` itself is still
        whatever the vendor's own default load just produced until
        something actually recomputes it.

        Synchronous, not the usual debounced multi-row path
        (:meth:`_schedule_multi_row_reprocess`) -- this runs once at
        load time, not on a rapid run of live-edit ticks that
        debouncing exists to coalesce, and a synchronous recompute
        here means the SR-shift reapplication below (which needs the
        *already-recomputed* axis to rebase against, not a stale
        vendor-default one) doesn't have to guess when a debounced
        recompute will actually finish."""
        if not self._fid_reprocessable(spectrum):
            return
        if spectrum.ndim == 1:
            self._commit_processing_preview_as_spectrum(spectrum)
        else:
            self._recompute_multi_row_data(spectrum)
        self.spectrum_list.item(index).setText(self.session.label(index))
        sr_shift = spectrum.metadata.raw.get("sr_shift_hz")
        if sr_shift:
            self._apply_reference_shift_hz(sr_shift)  # also redraws
        else:
            self._draw_selected_spectrum(spectrum, row=index)

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
            self._update_compare_tab_visibility()

    def _on_clear_clicked(self) -> None:
        self.session.clear()
        self.spectrum_list.clear()
        for canvas in (
            self.canvas, self.fid_canvas, self.processing_preview_canvas,
            self.compare_canvas, self.fit_canvas, self.spectral_fit_canvas,
        ):
            canvas.clear()
        self._phase_mode = None
        self._set_processing_enabled(False)
        self.info_label.setText("—")
        self.process_indirect_button.setVisible(False)
        self.pseudo2d_view_combo.setVisible(False)
        self.dimension_box.setVisible(False)
        self.fid_processing_box.setVisible(False)
        self._update_fit_tab_visibility(None)
        self._update_processing_preview_tab_visibility(None)
        self.compare_status_label.setText("Select at least two 1D spectra to compare.")
        self.compare_list.clear()
        self._update_compare_tab_visibility()
        self.peak_table.setRowCount(0)
        self.analysis_results_label.setText("")
        self.integration_results_label.setText("")
        self.spectral_fit_params_label.setText("")
        self.analysis_status_label.setText("Select a pseudo-2D spectrum (e.g. a T1/T2 relaxation series) to fit.")
        self._current_analysis_index = None
        self._slice_fit_results = None
        self.view_slice_fit_spin.setEnabled(False)
        self.relaxation_fit_params_label.setText("")
        self._relaxation_fit_results = None
        self.view_relaxation_peak_spin.setEnabled(False)

    def _selected_index(self) -> int | None:
        row = self.spectrum_list.currentRow()
        return row if 0 <= row < len(self.session) else None

    def _on_spectrum_list_context_menu(self, pos) -> None:
        item = self.spectrum_list.itemAt(pos)
        if item is None:
            return
        index = self.spectrum_list.row(item)

        spectrum = self.session.spectra[index]

        menu = QMenu(self)
        override_action = menu.addAction("Override nucleus…")
        suggest_action = menu.addAction("Suggest nucleus (500 MHz console / 100 MHz magnet)…")
        mass_action = menu.addAction("Set sample mass…")
        procno_action = None
        if spectrum.metadata.vendor == "bruker":
            procno_action = menu.addAction("Change proc number…")
        fit_curve_action = None
        if spectrum.ndim == 2 and spectrum.is_pseudo_2d:
            menu.addSeparator()
            fit_curve_action = menu.addAction("Fit Curve (relaxation/diffusion)…")
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
        elif fit_curve_action is not None and chosen is fit_curve_action:
            self._on_fit_curve_pseudo2d_action(index)
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
        self._schedule_sidecar_save(index)

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
        # Entered/displayed in mg (typical solid-state sample sizes are
        # tens of mg -- entering "0.045" instead of "45" invites
        # mistakes), converted to/from metadata.mass's own stored unit
        # (grams, unchanged -- see its own docstring) right at this
        # boundary, so nothing downstream (comparison normalization,
        # session.set_mass) needs to know mg was ever involved.
        current_mg = (self.session.spectra[index].metadata.mass or 0.0) * 1000.0
        value_mg, ok = QInputDialog.getDouble(
            self, "Set sample mass", "Mass (mg) -- used for mass-normalized comparison:",
            current_mg, 0.0, 1.0e6, 4,
        )
        if ok and value_mg > 0:
            self.session.set_mass(index, value_mg / 1000.0)
            if index == self._selected_index():
                self._refresh_selection_display(index)
            self._refresh_compare_view_if_active()
            self._schedule_sidecar_save(index)

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

    def _on_process_indirect_toggled(self, checked: bool) -> None:
        """Checked = process the indirect (F1) dimension as a real
        frequency dimension (``pseudo_2d=False``); unchecked (the
        default -- see :meth:`_load`) = pseudo-2D, F1 left as an
        index/slice dimension with no FFT."""
        index = self._selected_index()
        if index is None:
            return
        spectrum = self.session.spectra[index]
        wants_pseudo_2d = not checked
        if spectrum.ndim != 2 or spectrum.is_pseudo_2d == wants_pseudo_2d:
            return
        try:
            self.session.reload(index, pseudo_2d=wants_pseudo_2d, **self._current_phase_kwargs(spectrum))
            self._reapply_phase_post_reload(self.session.spectra[index], previous_raw=spectrum.metadata.raw)
        except Exception as exc:  # noqa: BLE001 -- surfaced to the user, not swallowed
            QMessageBox.critical(self, "Failed to reprocess spectrum", str(exc))
            # Revert the button to match what's actually loaded.
            self.process_indirect_button.blockSignals(True)
            self.process_indirect_button.setChecked(not spectrum.is_pseudo_2d)
            self.process_indirect_button.blockSignals(False)
            return
        self._on_selection_changed(index)
        self._schedule_sidecar_save(index)

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
        self.canvas.show_compare(slices["rows"], mode="stack", normalize=True)
        self.canvas_tabs.setCurrentWidget(self.canvas)

    # ============================================================
    # Compare tab: multi-spectrum stack/overlay
    # ============================================================

    def _build_compare_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        compare_intro_label = QLabel(
            "Ctrl/Cmd-click (or Shift-click) several 1D spectra in the "
            "Data tab's list to start a comparison -- or just select "
            "a single pseudo-2D spectrum there, which fills this in "
            "with its own slices automatically, in order. Fine-tune "
            "either here: check/uncheck to include or exclude, Move Up/"
            "Down to reorder the plot. Right-click for Plot (jump to "
            "the Compare canvas tab) or Fit Curve (relaxation/"
            "diffusion) across whatever's checked here."
        )
        compare_intro_label.setWordWrap(True)
        layout.addWidget(compare_intro_label)

        self.compare_status_label = QLabel("Select at least two 1D spectra to compare.")
        self.compare_status_label.setWordWrap(True)
        layout.addWidget(self.compare_status_label)

        self.compare_list = QListWidget()
        self.compare_list.setToolTip(
            "The spectra currently in this comparison -- either "
            "manually multi-selected 1D spectra, or every slice of a "
            "single selected pseudo-2D spectrum. Uncheck one to drop "
            "it from the plot; select a row and use Move Up/Down to "
            "change the stack order. Right-click for Fit Curve."
        )
        # This list never actually holds keyboard focus (rows are set
        # current programmatically, e.g. after a rebuild, not by the
        # user clicking into the widget first) -- on macOS specifically,
        # Qt's native style renders an unfocused "current" row's
        # highlight in a much fainter inactive gray, easy to miss
        # entirely against a list of otherwise-identical checkable rows.
        # Forcing the inactive highlight colors to match the active ones
        # makes the current row look the same regardless of focus.
        compare_list_palette = self.compare_list.palette()
        for role in (QPalette.ColorRole.Highlight, QPalette.ColorRole.HighlightedText):
            compare_list_palette.setColor(QPalette.ColorGroup.Inactive, role,
                                           compare_list_palette.color(QPalette.ColorGroup.Active, role))
        self.compare_list.setPalette(compare_list_palette)
        self.compare_list.itemChanged.connect(lambda _item: self._refresh_compare_view_if_active())
        self.compare_list.currentRowChanged.connect(self._on_compare_list_current_row_changed)
        self.compare_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.compare_list.customContextMenuRequested.connect(self._on_compare_list_context_menu)
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

        compare_action_row = QHBoxLayout()
        compare_plot_button = QPushButton("Plot")
        compare_plot_button.setToolTip(
            "(Re-)draw the comparison from whatever's checked above and "
            "jump to the Compare canvas tab -- also happens automatically "
            "on every checkbox/control change; this is just an explicit, "
            "always-available way to do the same thing on demand."
        )
        compare_plot_button.clicked.connect(self._refresh_compare_view)
        compare_action_row.addWidget(compare_plot_button)
        compare_fit_curve_button = QPushButton("Fit Curve…")
        compare_fit_curve_button.setToolTip(
            "Fit a relaxation/diffusion model across whatever's checked "
            "above (needs at least two 1D spectra) -- fits each checked "
            "entry's peaks first if they haven't been already."
        )
        compare_fit_curve_button.clicked.connect(self._on_fit_curve_compare_stack_action)
        compare_action_row.addWidget(compare_fit_curve_button)
        layout.addLayout(compare_action_row)

        combine_row = QHBoxLayout()
        combine_row.addWidget(QLabel("Operation:"))
        self.combine_op_combo = QComboBox()
        self.combine_op_combo.addItem("Sum", "sum")
        self.combine_op_combo.addItem("Average", "average")
        self.combine_op_combo.addItem("Subtract from first", "subtract")
        self._narrow_combo(self.combine_op_combo)
        combine_row.addWidget(self.combine_op_combo)
        combine_button = QPushButton("Combine (Compare stack)")
        combine_button.setToolTip(
            "Add/subtract/average whatever's checked above (needs at "
            "least two 1D spectra), interpolated onto a common ppm "
            "axis over their overlapping range -- added back into the "
            "session as a new spectrum."
        )
        combine_button.clicked.connect(self._on_combine_spectra_clicked)
        combine_row.addWidget(combine_button)
        layout.addLayout(combine_row)

        layout.addStretch(1)
        return tab

    #: Where per-type figure-style defaults are persisted -- a plain
    #: JSON dict of {type_key: style_dict}, deliberately not tied to
    #: this repo (a user-level dotfile, like every other desktop app's
    #: own settings) so it survives an update/reinstall.
    _FIGURE_STYLES_PATH = os.path.expanduser("~/.opennmr/figure_styles.json")
    _FIGURE_STYLE_TYPES = (
        ("1D spectrum", "1d"), ("2D contour", "2d"), ("Pseudo-2D stack", "pseudo2d"), ("Compare", "compare"),
    )

    def _all_canvases(self) -> list:
        """Every ``MplCanvas`` instance this window owns -- core
        styling (linewidth/fontsize/ticks/grid/colormap) applies to
        all of them uniformly, so the app has one consistent look
        rather than a different one per canvas tab."""
        return [
            self.fid_canvas, self.processing_preview_canvas, self.canvas,
            self.spectral_fit_canvas, self.fit_canvas, self.compare_canvas,
        ]

    def _build_output_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        intro = QLabel(
            "Live style controls -- changes apply immediately to every "
            "plot in the app, not just a separate export preview."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        style_box = QGroupBox("Core styling")
        style_form = QFormLayout(style_box)

        self.output_linewidth_spin = QDoubleSpinBox()
        self.output_linewidth_spin.setRange(0.1, 10.0)
        self.output_linewidth_spin.setSingleStep(0.1)
        self.output_linewidth_spin.setValue(MplCanvas._AXIS_LINEWIDTH)
        self.output_linewidth_spin.valueChanged.connect(self._on_output_style_changed)
        style_form.addRow("Axis/spine linewidth:", self.output_linewidth_spin)

        self.output_fontsize_spin = QSpinBox()
        self.output_fontsize_spin.setRange(6, 40)
        self.output_fontsize_spin.setValue(MplCanvas._AXIS_LABELSIZE)
        self.output_fontsize_spin.valueChanged.connect(self._on_output_style_changed)
        style_form.addRow("Font size:", self.output_fontsize_spin)

        self.output_tick_length_spin = QDoubleSpinBox()
        self.output_tick_length_spin.setRange(0.0, 20.0)
        self.output_tick_length_spin.setSingleStep(0.5)
        self.output_tick_length_spin.setValue(MplCanvas._TICK_LENGTH)
        self.output_tick_length_spin.valueChanged.connect(self._on_output_style_changed)
        style_form.addRow("Tick length:", self.output_tick_length_spin)

        self.output_tick_width_spin = QDoubleSpinBox()
        self.output_tick_width_spin.setRange(0.1, 10.0)
        self.output_tick_width_spin.setSingleStep(0.1)
        self.output_tick_width_spin.setValue(MplCanvas._TICK_WIDTH)
        self.output_tick_width_spin.valueChanged.connect(self._on_output_style_changed)
        style_form.addRow("Tick width:", self.output_tick_width_spin)

        self.output_grid_checkbox = QCheckBox("Show grid")
        self.output_grid_checkbox.toggled.connect(self._on_output_style_changed)
        style_form.addRow(self.output_grid_checkbox)

        self.output_minor_ticks_checkbox = QCheckBox("Minor ticks")
        self.output_minor_ticks_checkbox.setChecked(True)
        self.output_minor_ticks_checkbox.toggled.connect(self._on_output_style_changed)
        style_form.addRow(self.output_minor_ticks_checkbox)

        self.output_legend_checkbox = QCheckBox("Show legend (Spectrum tab)")
        self.output_legend_checkbox.setToolTip(
            "Same on/off state as the Spectrum canvas's own right-click "
            "\"Show Legend\" -- a second, always-visible control for it, "
            "not a separate setting."
        )
        self.output_legend_checkbox.toggled.connect(self._on_output_legend_changed)
        style_form.addRow(self.output_legend_checkbox)

        self.output_legend_loc_combo = QComboBox()
        self.output_legend_loc_combo.addItems([
            "best", "upper right", "upper left", "lower left", "lower right", "center",
        ])
        self._narrow_combo(self.output_legend_loc_combo)
        self.output_legend_loc_combo.currentIndexChanged.connect(self._on_output_legend_changed)
        style_form.addRow("Legend position:", self.output_legend_loc_combo)

        self.output_colormap_combo = QComboBox()
        self.output_colormap_combo.addItems(plotting.CONTOUR_CMAP_NAMES)
        self.output_colormap_combo.setCurrentText("viridis")
        self._narrow_combo(self.output_colormap_combo)
        self.output_colormap_combo.setToolTip("Colour scheme for the Compare tab's stacked/overlaid traces.")
        self.output_colormap_combo.currentIndexChanged.connect(self._on_output_style_changed)
        style_form.addRow("Compare colour map:", self.output_colormap_combo)

        layout.addWidget(style_box)

        export_box = QGroupBox("Export settings")
        export_intro = QLabel(
            "Applies only to File > Save Figure As... -- never to the "
            "live view here, which Qt's own window layout already "
            "controls (changing a live figure's size out from under "
            "Qt fights it visibly)."
        )
        export_intro.setWordWrap(True)
        export_form = QFormLayout(export_box)
        export_form.addRow(export_intro)

        self.output_dpi_spin = QSpinBox()
        self.output_dpi_spin.setRange(50, 1200)
        self.output_dpi_spin.setValue(int(self.canvas.figure.get_dpi()))
        self.output_dpi_spin.valueChanged.connect(self._on_output_export_settings_changed)
        export_form.addRow("DPI:", self.output_dpi_spin)

        fig_w, fig_h = self.canvas.figure.get_size_inches()
        self.output_fig_width_spin = QDoubleSpinBox()
        self.output_fig_width_spin.setRange(1.0, 40.0)
        self.output_fig_width_spin.setValue(float(fig_w))
        self.output_fig_width_spin.setSuffix(" in")
        self.output_fig_width_spin.valueChanged.connect(self._on_output_export_settings_changed)
        export_form.addRow("Figure width:", self.output_fig_width_spin)

        self.output_fig_height_spin = QDoubleSpinBox()
        self.output_fig_height_spin.setRange(1.0, 40.0)
        self.output_fig_height_spin.setValue(float(fig_h))
        self.output_fig_height_spin.setSuffix(" in")
        self.output_fig_height_spin.valueChanged.connect(self._on_output_export_settings_changed)
        export_form.addRow("Figure height:", self.output_fig_height_spin)

        layout.addWidget(export_box)

        template_box = QGroupBox("Per-type defaults")
        template_layout = QVBoxLayout(template_box)
        template_intro = QLabel(
            "Save the current styling above as the default for one "
            "spectrum type, or load a previously-saved one back in."
        )
        template_intro.setWordWrap(True)
        template_layout.addWidget(template_intro)
        self.output_type_combo = QComboBox()
        for label, key in self._FIGURE_STYLE_TYPES:
            self.output_type_combo.addItem(label, key)
        self._narrow_combo(self.output_type_combo)
        template_layout.addWidget(self.output_type_combo)
        template_button_row = QHBoxLayout()
        save_template_button = QPushButton("Save as default")
        save_template_button.clicked.connect(self._on_save_figure_style_template)
        load_template_button = QPushButton("Load default")
        load_template_button.clicked.connect(self._on_load_figure_style_template)
        template_button_row.addWidget(save_template_button)
        template_button_row.addWidget(load_template_button)
        template_layout.addLayout(template_button_row)
        layout.addWidget(template_box)

        layout.addStretch(1)
        return tab

    def _current_output_style_kwargs(self) -> dict:
        return {
            "linewidth": self.output_linewidth_spin.value(),
            "labelsize": self.output_fontsize_spin.value(),
            "tick_length": self.output_tick_length_spin.value(),
            "tick_width": self.output_tick_width_spin.value(),
            "minor_ticks": self.output_minor_ticks_checkbox.isChecked(),
            "grid": self.output_grid_checkbox.isChecked(),
            "compare_colormap": self.output_colormap_combo.currentText(),
        }

    def _current_output_export_kwargs(self) -> dict:
        return {
            "dpi": self.output_dpi_spin.value(),
            "fig_width": self.output_fig_width_spin.value(),
            "fig_height": self.output_fig_height_spin.value(),
        }

    def _on_output_style_changed(self, *_args) -> None:
        kwargs = self._current_output_style_kwargs()
        for canvas in self._all_canvases():
            canvas.set_style(**kwargs)

    def _on_output_export_settings_changed(self, *_args) -> None:
        kwargs = self._current_output_export_kwargs()
        for canvas in self._all_canvases():
            canvas.set_export_settings(**kwargs)

    def _on_output_legend_changed(self, *_args) -> None:
        self.canvas.set_legend_visible(self.output_legend_checkbox.isChecked())
        self.canvas.set_style(legend_loc=self.output_legend_loc_combo.currentText())

    def _load_figure_styles(self) -> dict:
        try:
            with open(self._FIGURE_STYLES_PATH) as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _on_save_figure_style_template(self) -> None:
        styles = self._load_figure_styles()
        key = self.output_type_combo.currentData()
        styles[key] = {**self._current_output_style_kwargs(), **self._current_output_export_kwargs()}
        os.makedirs(os.path.dirname(self._FIGURE_STYLES_PATH), exist_ok=True)
        with open(self._FIGURE_STYLES_PATH, "w") as f:
            json.dump(styles, f, indent=2)
        QMessageBox.information(
            self, "Saved", f"Saved current styling as the default for '{self.output_type_combo.currentText()}'.",
        )

    def _on_load_figure_style_template(self) -> None:
        styles = self._load_figure_styles()
        key = self.output_type_combo.currentData()
        style = styles.get(key)
        if style is None:
            QMessageBox.information(
                self, "No saved default",
                f"No styling has been saved yet for '{self.output_type_combo.currentText()}'.",
            )
            return
        self.output_linewidth_spin.setValue(style.get("linewidth", MplCanvas._AXIS_LINEWIDTH))
        self.output_fontsize_spin.setValue(style.get("labelsize", MplCanvas._AXIS_LABELSIZE))
        self.output_tick_length_spin.setValue(style.get("tick_length", MplCanvas._TICK_LENGTH))
        self.output_tick_width_spin.setValue(style.get("tick_width", MplCanvas._TICK_WIDTH))
        self.output_minor_ticks_checkbox.setChecked(style.get("minor_ticks", True))
        self.output_grid_checkbox.setChecked(style.get("grid", False))
        self.output_colormap_combo.setCurrentText(style.get("compare_colormap", "viridis"))
        self.output_dpi_spin.setValue(style.get("dpi", 300))
        self.output_fig_width_spin.setValue(style.get("fig_width", 6.4))
        self.output_fig_height_spin.setValue(style.get("fig_height", 4.8))
        # Every spin/combo above already applies itself on change
        # (_on_output_style_changed/_on_output_export_settings_changed)
        # -- no separate "apply" step needed.

    def _selected_rows(self) -> list[int]:
        return sorted(self.spectrum_list.row(item) for item in self.spectrum_list.selectedItems())

    def _on_list_selection_changed(self) -> None:
        rows = self._selected_rows()
        if len(rows) > 1:
            self._populate_compare_list(rows)
            self._refresh_compare_view()
        else:
            # Multi- -> single-selection doesn't always change
            # currentRow() (the item that was already current can stay
            # current), so currentRowChanged wouldn't fire on its own
            # -- redraw the single-spectrum view explicitly.
            self._on_selection_changed(self.spectrum_list.currentRow())
            # Item D: selecting a single pseudo-2D spectrum fills the
            # Compare tab with its own slices automatically -- a
            # pseudo-2D series IS a stack of related 1D spectra, so it
            # gets the same select/deselect/reorder/Fit-Curve context
            # a manually multi-selected stack of separate files does,
            # with no extra step. Anything else about a single
            # selection deliberately doesn't touch compare_list -- a
            # multi-select gesture (e.g. clearSelection() followed by
            # several setSelected(True) calls) can pass through a
            # transient <=1-row state along the way, and clearing (or
            # replacing) on every such blip would wipe out check-state/
            # reordering already done in the Compare tab for no reason.
            #
            # Runs AFTER _on_selection_changed above, not before --
            # _on_selection_changed's own _draw_selected_spectrum/
            # _silo_other_canvas_tabs unconditionally redraws the
            # Spectrum canvas and clears compare_canvas (right, for a
            # single selection with nothing Compare-relevant to
            # restore), so populating Compare *before* that meant this
            # whole block's own effect -- the freshly-populated stack --
            # was silently undone one line later; running it last
            # instead means the *list* population is the thing left
            # standing. The canvas-tab switch is deliberately turned
            # off here (switch_tab=False), unlike a manual multi-select
            # (the branch above) -- explicit user feedback that loading
            # or selecting a pseudo-2D should land on the Spectrum tab,
            # not jump to Compare just because its slices got listed
            # there as a side effect; Compare is still fully populated
            # and correctly drawn in the background for whenever the
            # user actually clicks over to it themselves.
            if len(rows) == 1 and 0 <= rows[0] < len(self.session):
                spectrum = self.session.spectra[rows[0]]
                if spectrum.ndim == 2 and spectrum.is_pseudo_2d:
                    self._populate_compare_list_from_pseudo2d(rows[0], spectrum)
                    self._refresh_compare_view(switch_tab=False)

    def _populate_compare_list(self, rows: list[int]) -> None:
        """Rebuild the Compare tab's own list from the Data tab's
        current multi-selection (each row as a whole spectrum -- entry
        ``(row, None)``, see :meth:`_resolve_compare_entry`),
        preserving any check-state/reorder already done here for rows
        still selected -- so ctrl-clicking one more spectrum in the
        Data tab appends it rather than wiping out prior
        deselections/reordering, but ctrl-clicking a row *out* of the
        selection still drops it from here too."""
        # Filter out-of-range indices defensively -- a spectrum removal
        # can trigger this (directly, or reentrantly via the list
        # widget's own itemSelectionChanged) at a point where a stale
        # row number briefly doesn't match session.spectra's new,
        # shorter length.
        n = len(self.session)
        rows = [r for r in rows if 0 <= r < n]
        entries = [(r, None) for r in rows]
        self._rebuild_compare_list(entries)

    def _populate_compare_list_from_pseudo2d(self, row: int, spectrum) -> None:
        """Fill the Compare tab with every slice of ``spectrum`` (entry
        ``(row, slice_index)``, 1-based), in order, all checked --
        replaces whatever was there before (this is a fresh "look at
        this series" action, not a merge with an unrelated prior
        multi-select)."""
        n_slices = spectrum.data.shape[0]
        entries = [(row, slice_index) for slice_index in range(1, n_slices + 1)]
        self._rebuild_compare_list(entries, preserve_previous=False)

    def _rebuild_compare_list(self, entries: list[tuple[int, int | None]], preserve_previous: bool = True) -> None:
        previous_checked = {}
        previous_order = []
        previous_current_entry = None
        if preserve_previous:
            previous_order = [self.compare_list.item(i).data(Qt.ItemDataRole.UserRole)
                               for i in range(self.compare_list.count())]
            previous_checked = {self.compare_list.item(i).data(Qt.ItemDataRole.UserRole):
                                 self.compare_list.item(i).checkState()
                                 for i in range(self.compare_list.count())}
            current_item = self.compare_list.currentItem()
            if current_item is not None:
                previous_current_entry = current_item.data(Qt.ItemDataRole.UserRole)
        entries_set = set(entries)
        new_order = [e for e in previous_order if e in entries_set] + [e for e in entries if e not in previous_order]

        self.compare_list.blockSignals(True)
        self.compare_list.clear()
        for entry in new_order:
            item = QListWidgetItem(self._compare_entry_label(entry))
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(previous_checked.get(entry, Qt.CheckState.Checked))
            item.setData(Qt.ItemDataRole.UserRole, entry)
            self.compare_list.addItem(item)
        # Always leave *something* current/highlighted -- explicit user
        # feedback that it's "no longer obvious which spectrum is
        # selected" in this list. Nothing ever set an initial current
        # row before (only Move Up/Down did, and only after the user
        # had already clicked a row to move it), so a freshly
        # (re)populated list -- including the automatic pseudo-2D
        # auto-populate, which needs no click at all -- showed no
        # current-row highlight whatsoever. Keeps the same entry
        # current across a rebuild when it's still present (e.g. after
        # Move Up/Down or a check-state toggle), falling back to the
        # first row otherwise.
        if new_order:
            current_row = new_order.index(previous_current_entry) if previous_current_entry in new_order else 0
            self.compare_list.setCurrentRow(current_row)
        self.compare_list.blockSignals(False)
        self._update_compare_tab_visibility()

    def _compare_entry_label(self, entry: tuple[int, int | None]) -> str:
        session_index, slice_index = entry
        base = self.session.label(session_index)
        if slice_index is None:
            return base
        return f"{base} · {self._slice_label(self.session.spectra[session_index], slice_index)}"

    def _resolve_compare_entry(self, entry: tuple[int, int | None]):
        """The actual 1D ``NMRSpectrum`` a Compare-list entry refers
        to -- either a whole (already-1D) spectrum, or one extracted
        slice of a pseudo-2D series. Returns ``None`` for anything
        that's gone stale (a removed spectrum, a pseudo-2D reprocessed
        down to fewer slices) or isn't actually 1D/pseudo-2D at all,
        rather than raising -- the caller already reports a skip count
        for non-1D entries."""
        session_index, slice_index = entry
        if not (0 <= session_index < len(self.session)):
            return None
        spectrum = self.session.spectra[session_index]
        if slice_index is None:
            return spectrum
        if spectrum.ndim != 2 or not spectrum.is_pseudo_2d:
            return None
        if not (1 <= slice_index <= spectrum.data.shape[0]):
            return None
        return processing_slicing.get_trace(spectrum, axis="f1", index=slice_index)

    def _compare_checked_rows_and_entries(self) -> list[tuple[int, tuple[int, int | None]]]:
        result = []
        for i in range(self.compare_list.count()):
            item = self.compare_list.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                result.append((i, item.data(Qt.ItemDataRole.UserRole)))
        return result

    def _compare_checked_entries(self) -> list[tuple[int, int | None]]:
        return [entry for _row, entry in self._compare_checked_rows_and_entries()]

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

    def _on_compare_list_context_menu(self, pos) -> None:
        """Plot (an explicit, discoverable way to (re-)draw the
        comparison and jump to the Compare canvas tab, rather than
        only ever happening as a side effect of toggling a checkbox)
        and Fit Curve (relaxation/diffusion) across a "stack of 1D
        data" -- both gated on there being at least one checked entry,
        since neither makes sense before a stack actually exists."""
        if not self._compare_checked_entries():
            return
        menu = QMenu(self)
        plot_action = menu.addAction("Plot")
        fit_curve_action = menu.addAction("Fit Curve (relaxation/diffusion)…")
        chosen = menu.exec(self.compare_list.viewport().mapToGlobal(pos))
        if chosen is plot_action:
            self._refresh_compare_view()
        elif chosen is fit_curve_action:
            self._on_fit_curve_compare_stack_action()

    def _on_fit_curve_compare_stack_action(self) -> None:
        stack = self._compare_checked_1d_stack()
        if len(stack) < 2:
            QMessageBox.information(
                self, "Not enough spectra",
                "Fit Curve needs at least two 1D spectra checked in the Compare list.",
            )
            return
        try:
            slice_results = self._fit_peaks(stack)
        except Exception as exc:  # noqa: BLE001 -- surfaced to the user, not swallowed
            QMessageBox.critical(self, "Fit failed", str(exc))
            return
        # No single spectrum "owns" a Compare-stack fit the way a
        # pseudo-2D's own slices do -- reference_spectrum only
        # supplies vendor/path context for the Values-source combo
        # (Auto/vclist/relative-gradient), and persist_to is
        # deliberately None (see _run_curve_fit's own docstring):
        # switching away and back loses this result, matching the
        # Compare tab's own existing no-per-item-persistence character.
        self._run_curve_fit(slice_results, len(stack), stack[0], persist_to=None)

    def _on_combine_spectra_clicked(self) -> None:
        stack = self._compare_checked_1d_stack()
        if len(stack) < 2:
            QMessageBox.information(
                self, "Not enough spectra",
                "Combine needs at least two 1D spectra checked in the Compare list.",
            )
            return
        op = self.combine_op_combo.currentData()
        if op == "subtract":
            ops, scaling_factors, label_op = ["+"] + ["-"] * (len(stack) - 1), None, "diff"
        elif op == "average":
            ops, scaling_factors, label_op = ["+"] * len(stack), [1.0 / len(stack)] * len(stack), "avg"
        else:
            ops, scaling_factors, label_op = ["+"] * len(stack), None, "sum"
        try:
            combined = processing_arithmetic.combine_spectra(stack, ops=ops, scaling_factors=scaling_factors)
        except Exception as exc:  # noqa: BLE001 -- surfaced to the user, not swallowed
            QMessageBox.critical(self, "Combine failed", str(exc))
            return
        self._add_derived_spectrum(combined, f"{label_op}({len(stack)} spectra)")

    def _on_compare_controls_changed(self) -> None:
        self._refresh_compare_view_if_active()

    def _refresh_compare_view_if_active(self, switch_tab: bool = True) -> None:
        if self.compare_list.count() > 0:
            self._refresh_compare_view(switch_tab=switch_tab)

    def _refresh_compare_view(self, switch_tab: bool = True, force_render: bool = False) -> None:
        rows_and_entries = self._compare_checked_rows_and_entries()
        rows_and_candidates = [(row, self._resolve_compare_entry(e)) for row, e in rows_and_entries]
        # stale entries (removed/reprocessed) silently drop
        rows_and_candidates = [(row, c) for row, c in rows_and_candidates if c is not None]
        rows_and_1d = [(row, c) for row, c in rows_and_candidates if c.ndim == 1]
        skipped_ndim = len(rows_and_candidates) - len(rows_and_1d)

        norm_mode = ["none", "max", "mass"][self.compare_normalize_combo.currentIndex()]
        plot_rows: list[int] = []
        plot_spectra: list = []
        skipped_mass: list = []
        if norm_mode == "mass":
            # Only `data` (the real part actually plotted here) needs
            # scaling -- these copies are used for this one comparison
            # plot and discarded, never stored back into the session.
            for row, s in rows_and_1d:
                if not s.metadata.mass:
                    skipped_mass.append(s)
                    continue
                plot_rows.append(row)
                plot_spectra.append(dataclasses.replace(s, data=s.data / s.metadata.mass))
        else:
            plot_rows = [row for row, _s in rows_and_1d]
            plot_spectra = [s for _row, s in rows_and_1d]

        # compare_list's own row index -> position among what's actually
        # plotted (checked rows only, minus anything just filtered out
        # above) -- kept even on the "too few to plot" early return below,
        # so a stale mapping doesn't linger pointing at axes that no
        # longer exist.
        self._compare_plot_row_positions = {row: pos for pos, row in enumerate(plot_rows)}

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

        # Rendering a "Stacked" Compare view is one subplot per
        # spectrum -- fine for a typical handful, but a pseudo-2D
        # series can have hundreds/thousands of slices (a real
        # microimaging/HSQC-as-pseudo-2D dataset measured at ~25s just
        # for this render alone). Auto-populating Compare from such a
        # series (see _populate_compare_list_from_pseudo2d) already
        # calls this with switch_tab=False specifically so it doesn't
        # steal focus -- if the Compare canvas isn't even the one
        # currently showing either, there's no reason to pay that cost
        # now; defer the actual draw until Compare is genuinely
        # visited (see _render_compare_view_if_stale).
        if not force_render and not switch_tab and self.canvas_tabs.currentWidget() is not self.compare_canvas:
            self._compare_view_stale = True
            return
        self._compare_view_stale = False

        try:
            self.compare_canvas.show_compare(plot_spectra, mode=mode, normalize=(norm_mode == "max"))
            if switch_tab:
                self.canvas_tabs.setCurrentWidget(self.compare_canvas)
            # show_compare always resets the canvas's own selection
            # state -- reapply whatever compare_list's current row maps
            # to, so the plot's own highlight doesn't silently drop out
            # every time the comparison is rebuilt.
            self._on_compare_list_current_row_changed(self.compare_list.currentRow())
        except Exception as exc:  # noqa: BLE001 -- surfaced to the user, not swallowed
            QMessageBox.critical(self, "Failed to plot comparison", str(exc))

    def _on_compare_list_current_row_changed(self, row: int) -> None:
        """Highlight (tint the background of) whichever row of the
        Compare plot corresponds to compare_list's own current row --
        the same tint :meth:`MplCanvas._toggle_compare_selection`
        already applies on a direct canvas click, now also driven by
        list selection so it's obvious which spectrum is "selected"
        without needing to click the plot itself first. ``row`` can map
        to nothing plotted (unchecked, or filtered out as non-1D/no
        mass) -- clears the highlight in that case rather than leaving
        a stale one from whatever was selected before."""
        position = self._compare_plot_row_positions.get(row) if row >= 0 else None
        self.compare_canvas.set_compare_selection(position)

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

        self.fid_display_combo = QComboBox()
        self.fid_display_combo.addItem("Real", "real")
        self.fid_display_combo.addItem("Magnitude", "magnitude")
        self.fid_display_combo.setToolTip(
            "How the FID preview plots the (complex) time-domain data -- "
            "Magnitude (|FID|, a single clean decay/echo envelope, easiest "
            "for judging alignment/decay shape) or Real (one oscillating "
            "component, closer to what phase-sensitive detection actually "
            "recorded). A view setting only -- doesn't affect processing."
        )
        self.fid_display_combo.currentIndexChanged.connect(lambda _i: self._refresh_fid_display_preview())
        form.addRow("Preview:", self.fid_display_combo)

        self.fid_x_unit_combo = QComboBox()
        self.fid_x_unit_combo.addItem("Time (s)", "time")
        self.fid_x_unit_combo.addItem("Points", "points")
        self.fid_x_unit_combo.setToolTip(
            "FID plot's x-axis -- Time (s, the default) or raw sample "
            "index (Points)."
        )
        self._narrow_combo(self.fid_x_unit_combo)
        self.fid_x_unit_combo.currentIndexChanged.connect(lambda _i: self._refresh_fid_display_preview())
        form.addRow("FID x-axis:", self.fid_x_unit_combo)

        self.window_combo = QComboBox()
        self.window_combo.currentIndexChanged.connect(self._on_window_combo_changed)
        form.addRow("Window:", self.window_combo)

        self.lb_spin = QDoubleSpinBox()
        self.lb_spin.setRange(0.0, 1.0e5)
        self.lb_spin.setDecimals(2)
        self.lb_spin.setSuffix(" Hz")
        self.lb_spin.setToolTip("Line broadening, for the Exponential window only.")
        self.lb_spin.valueChanged.connect(lambda _v: self._on_fid_processing_control_changed())
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
        self.zero_fill_custom_spin.valueChanged.connect(lambda _v: self._on_fid_processing_control_changed())
        form.addRow("Custom size:", self.zero_fill_custom_spin)

        self.tdeff_spin = QSpinBox()
        self.tdeff_spin.setRange(1, 1 << 20)
        self.tdeff_spin.setSuffix(" pts")
        self.tdeff_spin.setToolTip(
            "Truncate the raw FID to this many points before any other "
            "processing (X-Pulse only) -- e.g. to cut off a recorded "
            "tail past the point the acquisition itself flagged as "
            "effective (TDEFF). Defaults to the full acquired FID "
            "(no truncation); step down from there to trim points off "
            "the end, rather than counting up from 1 to reach a large "
            "number. Prefilled from the file's own recorded TDEFF when "
            "it has one. Shown as a dashed red line on the FID plot."
        )
        self.tdeff_spin.valueChanged.connect(lambda _v: self._on_fid_processing_control_changed())
        form.addRow("Truncate to:", self.tdeff_spin)

        self.align_echo_button = QPushButton("Align Echo")
        self.align_echo_button.setCheckable(True)
        self.align_echo_button.setToolTip(
            "For full-echo acquisitions (rising then falling, e.g. "
            "imaging profiles) rather than a one-sided decaying FID -- "
            "rolls the FID so its peak sits at t=0, before windowing. "
            "Clicking immediately reprocesses and switches to the FID "
            "tab so you can see whether it actually worked (dashed "
            "green marker at the found echo top -- should land right "
            "at t=0 once aligned) rather than silently changing a "
            "setting for later. Toggle off to revert."
        )
        self.align_echo_button.toggled.connect(self._on_align_echo_toggled)
        form.addRow(self.align_echo_button)

        self.fid_shift_spin = QDoubleSpinBox()
        self.fid_shift_spin.setRange(-1.0e6, 1.0e6)
        self.fid_shift_spin.setDecimals(2)
        self.fid_shift_spin.setSuffix(" pts")
        self.fid_shift_spin.setToolTip(
            "Manual circular shift of the FID (samples, fractional "
            "allowed), applied before windowing -- independent of "
            "(and on top of) echo alignment."
        )
        self.fid_shift_spin.valueChanged.connect(lambda _v: self._on_fid_processing_control_changed())
        form.addRow("FID shift:", self.fid_shift_spin)

        self.linear_predict_points_spin = QSpinBox()
        self.linear_predict_points_spin.setRange(0, 1 << 16)
        self.linear_predict_points_spin.setToolTip(
            "Extend (Forward) or reconstruct (Backward, see the "
            "Direction combo below) the FID by this many linear-"
            "prediction points before windowing -- reduces truncation "
            "artefacts from a short acquisition, or recovers points "
            "lost to a dead-time window before acquisition began. "
            "0 = off."
        )
        self.linear_predict_points_spin.valueChanged.connect(lambda _v: self._on_fid_processing_control_changed())
        form.addRow("Linear-predict points:", self.linear_predict_points_spin)

        self.linear_predict_order_spin = QSpinBox()
        self.linear_predict_order_spin.setRange(1, 1000)
        self.linear_predict_order_spin.setValue(20)
        self.linear_predict_order_spin.setToolTip("Linear-prediction model order (only used when points > 0).")
        self.linear_predict_order_spin.valueChanged.connect(lambda _v: self._on_fid_processing_control_changed())
        form.addRow("LP order:", self.linear_predict_order_spin)

        self.linear_predict_direction_combo = QComboBox()
        self.linear_predict_direction_combo.addItem("Forward (extend past the end)", "forward")
        self.linear_predict_direction_combo.addItem("Backward (reconstruct before the start)", "backward")
        self._narrow_combo(self.linear_predict_direction_combo)
        self.linear_predict_direction_combo.setToolTip(
            "Forward: predicts points continuing past the FID's "
            "measured end -- the usual case, for a short acquisition. "
            "Backward: predicts points *before* the measured start "
            "instead, reconstructing ones lost to a fixed dead-time "
            "delay before acquisition began (only used when Linear-"
            "predict points > 0)."
        )
        self.linear_predict_direction_combo.currentIndexChanged.connect(
            lambda _i: self._on_fid_processing_control_changed()
        )
        form.addRow("LP direction:", self.linear_predict_direction_combo)

        # F1 (indirect dimension) TD/SI -- only meaningful for a
        # genuine 2D spectrum with a real F1 FFT (not pseudo-2D, where
        # "F1" is a slice index, not a frequency dimension); enabled
        # state is computed directly from spectrum.ndim/is_pseudo_2d in
        # _refresh_fid_processing_controls, NOT from self._phase_mode
        # -- _capture_phase_reference (which sets _phase_mode) runs
        # *after* _refresh_fid_processing_controls in
        # _refresh_selection_display's own call order, so _phase_mode
        # would always be one selection stale here. Explicit user
        # feedback: F2 already had both (Truncate to/Zero-fill to,
        # above); F1 had neither, for either vendor.
        self.tdeff_f1_spin = QSpinBox()
        self.tdeff_f1_spin.setRange(1, 1 << 20)
        self.tdeff_f1_spin.setSuffix(" pts")
        self.tdeff_f1_spin.setToolTip(
            "Truncate F1 (the indirect dimension) to this many "
            "increments before its own hypercomplex decode/FFT -- for "
            "States/States-TPPI/Echo-Antiecho data, a row *pair* per "
            "increment, so this never separates a real row from its "
            "matching imaginary one. Defaults to every increment "
            "actually present (no truncation)."
        )
        self.tdeff_f1_spin.valueChanged.connect(lambda _v: self._on_fid_processing_control_changed())
        form.addRow("Truncate F1 to:", self.tdeff_f1_spin)

        self.zero_fill_f1_spin = QSpinBox()
        self.zero_fill_f1_spin.setRange(1, 1 << 20)
        self.zero_fill_f1_spin.setSuffix(" pts")
        self.zero_fill_f1_spin.setToolTip(
            "F1 (the indirect dimension)'s own FFT size -- zero-fills "
            "(or truncates) F1 to this many points. Defaults to F1's "
            "own current length (no-op) -- unlike F2's own Zero-fill "
            "to above, there's no '2x acquired' convention for F1 "
            "here; opt in explicitly to a larger value."
        )
        self.zero_fill_f1_spin.valueChanged.connect(lambda _v: self._on_fid_processing_control_changed())
        form.addRow("Zero-fill F1 to:", self.zero_fill_f1_spin)

        reset_processing_button = QPushButton("Reset")
        reset_processing_button.setToolTip(
            "Discard every live processing choice above (window/"
            "zero-fill/shift/align-echo/linear-predict/truncate, F1 "
            "truncate/zero-fill, phase, baseline) and revert to "
            "exactly how this spectrum "
            "looked right when it was first loaded. Everything above "
            "is already live -- there's no separate 'Apply' step to "
            "commit it, so this is the opposite: throwing it away, "
            "not applying it."
        )
        reset_processing_button.clicked.connect(self._on_reset_processing_clicked)
        form.addRow(reset_processing_button)

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
        self._on_fid_processing_control_changed()

    def _set_window_combo_key(self, key: str) -> None:
        idx = self.window_combo.findData(key)
        self.window_combo.blockSignals(True)
        self.window_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.window_combo.blockSignals(False)
        self.lb_spin.setEnabled(self.window_combo.currentData() == "exponential")

    def _on_zero_fill_combo_changed(self, _index: int) -> None:
        self.zero_fill_custom_spin.setVisible(self.zero_fill_combo.currentData() is None)
        self._on_fid_processing_control_changed()

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
    def _fid_reprocessable(spectrum) -> bool:
        """Whether ``spectrum`` was loaded from a raw FID this GUI can
        re-read and reprocess with different parameters -- true for any
        X-Pulse, Varian, or JEOL spectrum (all three are always
        raw-FID, no vendor-processed fallback exists for any of them),
        or a Bruker experiment whose raw ``fid``/``ser`` is present on
        disk (if so, :func:`opennmr.io.bruker._read_auto` would have
        preferred it over the already-processed fallback in the first
        place). Still true even if ``metadata.raw["window_unsupported"]``
        is set -- that only means the *recorded* window wasn't applied
        (see :mod:`opennmr.io.bruker`'s module docstring); every other
        raw-FID reprocessing control, including picking a different
        window, still works. False, unlike the window case, when
        ``metadata.raw["read_2d_shape_guess_failed"]`` is set -- there
        the raw ``ser``'s shape couldn't be determined at all, so
        ``_read_auto`` already fell back to already-processed data (no
        ``fid_raw`` cached to reprocess from, unlike an unsupported
        window)."""
        if spectrum.metadata.raw.get("read_2d_shape_guess_failed"):
            return False
        if spectrum.metadata.vendor in ("xpulse", "varian", "jeol"):
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
        # Only worth showing while the feature is actually in play --
        # for an ordinary one-sided decaying FID this would just mark
        # the very first point (its magnitude peak is trivially at
        # t=0), which is a meaningless/cluttering line, not a useful
        # confirmation of anything.
        echo_top = processing_ft.find_echo_top(fid) if self.align_echo_button.isChecked() else None
        self.fid_canvas.show_fid(
            fid, dt, title=title, window_envelope=envelope, tdeff_index=tdeff or None, echo_top_index=echo_top,
            mode=self.fid_display_combo.currentData(), x_unit=self.fid_x_unit_combo.currentData(),
        )
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

    def _refresh_fid_display_preview(self) -> None:
        """The "Preview:" (Real/Magnitude) combo is a pure view setting
        -- doesn't affect processing -- so unlike the rest of the FID-
        processing box's controls, changing it should refresh whichever
        of the FID tab (the actually-loaded spectrum's own FID) or the
        Processing tab's own live preview happens to be showing right
        now; each of the two refreshers below is a no-op unless its
        own canvas is actually the current one, so calling both here
        is safe."""
        self._refresh_fid_preview_if_active()
        self._refresh_processing_preview_if_active()

    def _refresh_fid_preview_if_active(self) -> None:
        """Re-shows the FID tab's own already-loaded FID (only its
        display ``mode`` can change live here -- see
        :meth:`_refresh_fid_display_preview` -- everything else about
        it only changes via a real selection or "Apply Processing"),
        but only while that tab is actually the one showing --
        otherwise this would redraw and switch the user to a tab they
        didn't ask for. Clicking the FID tab itself (see
        :meth:`_on_canvas_tab_changed`) is what first shows it."""
        if self.canvas_tabs.currentWidget() is not self.fid_canvas:
            return
        index = self._selected_index()
        if index is None:
            return
        spectrum = self.session.spectra[index]
        if self._fid_preview_available(spectrum):
            self._show_fid_preview(spectrum, switch_tab=False)

    def _render_compare_view_if_stale(self) -> None:
        if self._compare_view_stale:
            self._refresh_compare_view(switch_tab=False, force_render=True)

    def _on_canvas_tab_changed(self, _index: int) -> None:
        if self.canvas_tabs.currentWidget() is self.processing_preview_canvas:
            self._refresh_processing_preview_if_active()
            return
        if self.canvas_tabs.currentWidget() is self.compare_canvas:
            self._render_compare_view_if_stale()
            return
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

    def _process_raw_fid_row(self, spectrum, raw_row, dt: float):
        """The FID-domain-through-FFT half of the live pipeline
        (shift/echo-align/truncate/linear-predict, window, zero-fill,
        FFT, NS-normalize) for exactly one raw FID row -- shared by
        :meth:`_compute_processing_preview` (one row: 1D, or a
        pseudo-2D's currently-selected slice) and
        :meth:`_recompute_multi_row_data` (every row: pseudo-2D
        "apply to every slice", or a conventional 2D's own F2
        dimension), so a live multi-row recompute can't silently
        drift from what the single-row preview (or, before this
        existed, "Apply Processing") would show for that same row.
        Uses :func:`opennmr.processing.ft.prepare_fid` -- the exact
        same function every vendor reader's own ``read_1d``/
        ``read_2d`` calls too. Returns ``(filled_fid, complex_spectrum,
        ppm)`` given a reference ppm axis of the row's own original
        (pre-zero-fill) length is reused for range only -- see below --
        or ``None`` on the same "can't compute this way" conditions
        :meth:`_compute_processing_preview` already documents (an
        unsupported Bruker WDW code hit with "Recorded" selected)."""
        is_xpulse = spectrum.metadata.vendor == "xpulse"
        truncate = self.tdeff_spin.value() or None if (is_xpulse and self.tdeff_spin.isEnabled()) else None
        try:
            fid = processing_ft.prepare_fid(
                raw_row.astype(complex), fid_shift=self.fid_shift_spin.value(),
                align_echo_enabled=self.align_echo_button.isChecked(), truncate=truncate,
                linear_predict_points=self.linear_predict_points_spin.value(),
                linear_predict_order=self.linear_predict_order_spin.value(),
                linear_predict_direction=self.linear_predict_direction_combo.currentData(),
            )
        except Exception:
            # Best-effort, matching _baseline_correct_if_enabled: a
            # genuinely degenerate combination of controls (e.g. LP with
            # too few points even at the smallest valid order) shouldn't
            # crash a live Qt signal handler -- fall back to the
            # unmodified raw row and keep going.
            fid = raw_row.astype(complex)

        window_key = self.window_combo.currentData()
        if window_key == "__recorded__":
            procs = spectrum.metadata.raw.get("procs")
            sw = float(procs["SW_p"]) if procs and "SW_p" in procs else None
            if sw is None:
                return None
            try:
                windowed = io_bruker._apply_bruker_window(fid, procs, sw)
            except NotImplementedError:
                windowed = fid  # matches read_1d/read_2d's own "leave unwindowed" fallback
        else:
            window_kwarg, window_kwargs_kwarg = self._resolve_window_kwargs()
            windowed = processing_ft.apodize(fid, dt, window=window_kwarg, **(window_kwargs_kwarg or {}))

        filled = processing_ft.zero_fill(windowed, self._zero_fill_value())
        freq_hz, data = processing_ft.fft_spectrum(filled, dt)
        # Every vendor reader's own read_1d/read_2d divides by NS
        # (scans) at load time (data = spectrum / mass / ns) -- missing
        # here meant this preview could show a spectrum hundreds or
        # thousands of times the amplitude of what's actually loaded
        # (found on real data: NS=2600, so a ~2600x mismatch), looking
        # nothing like the same phase/lineshape even though the phase
        # angle itself was correct throughout. mass isn't included --
        # the GUI never sends a nonzero load-time mass= to begin with,
        # unlike metadata.mass (a separate, Compare-only concept).
        data = data / (spectrum.metadata.ns or 1.0)
        return filled, data

    def _compute_processing_preview(self, spectrum):
        """Compute the Processing tab's live preview entirely from the
        already-cached raw FID (``metadata.raw["fid_raw"]``), in
        memory -- no disk I/O, and for a pseudo-2D, only the
        currently-viewed slice is actually processed, not the whole
        series (see :meth:`_recompute_multi_row_data` for "every
        row").

        Cheap enough to call on every phase-drag tick, unlike the
        ``Session.preview()``-based reload this replaces -- explicit
        user feedback that phasing wasn't updating this preview at all
        (only the Spectrum tab), because that reload was reserved for
        discrete actions only, never continuous dragging.

        Returns ``(fid, dt, preview_spectrum, title)``, or ``None`` if
        a preview can't be computed this way: no raw FID cached (an
        older on-disk session, or a vendor path that predates
        ``fid_raw``), or an unsupported Bruker WDW code hit with
        "Recorded" selected (mirrors ``read_1d``/``read_2d``'s own
        "leave unwindowed, don't fail the load" behaviour rather than
        raising)."""
        raw_fid = spectrum.metadata.raw.get("fid_raw")
        dt = spectrum.metadata.raw.get("fid_dt")
        if raw_fid is None or dt is None:
            return None

        title = ""
        slice_template = spectrum
        if raw_fid.ndim == 2:
            row = min(max(self.slice_spin.value() - 1, 0), raw_fid.shape[0] - 1)
            title = self._slice_label(spectrum, row + 1)
            raw_fid = raw_fid[row]
            slice_template = get_trace(spectrum, axis="f1", index=row + 1)

        computed = self._process_raw_fid_row(spectrum, raw_fid, dt)
        if computed is None:
            return None
        filled, data = computed

        # The ppm axis's *range* is fixed by acquisition (O1/SF/SW),
        # unaffected by window/zero-fill choice -- only the point
        # count changes with zero-fill size. Reused from the already-
        # loaded spectrum's own axis rather than re-deriving vendor-
        # specific O1/SF/SW axis math a second time here, which would
        # risk a second implementation quietly drifting from the
        # vendor's own.
        reference_ppm = slice_template.axes[0].ppm
        ppm = np.linspace(reference_ppm[0], reference_ppm[-1], len(data))
        hz = ppm * (spectrum.metadata.sfo1 or 1.0)

        # Total current phase: whatever was already baked in by the
        # last load/reprocess, plus the live delta still showing in
        # the Phase spin boxes -- same formula _current_phase_kwargs
        # uses for the dataset-wide case. A per-slice-autophased row's
        # own individually-different phase (from "Autophase (each
        # slice independently)") isn't tracked back to here -- a
        # known, minor display gap, not a correctness risk the way the
        # FID-domain steps above are.
        phase0 = float(spectrum.metadata.raw.get("phase0", 0.0)) + self.phase0_spin.value()
        phase1 = float(spectrum.metadata.raw.get("phase1", 0.0)) + self.phase1_spin.value()
        span = ppm.max() - ppm.min()
        pivot_frac = (self.pivot_spin.value() - ppm.min()) / span if span else 0.5
        data = processing_ft.apply_phase(data, ppm, phase0, phase1, pivot_frac=pivot_frac)
        data = self._baseline_correct_if_enabled(data, ppm, spectrum)

        axis = NMRAxis(kind="frequency", ppm=ppm, hz=hz)
        preview_spectrum = NMRSpectrum(
            data=np.real(data), axes=[axis], metadata=slice_template.metadata, data_imag=np.imag(data),
        )
        return filled, dt, preview_spectrum, title

    def _recompute_multi_row_data(self, spectrum) -> bool:
        """The whole-dataset analogue of :meth:`_compute_processing_preview`
        -- recomputes *every* row's FID-domain-through-phase(-through-
        baseline) processing live, in memory, and writes the result
        straight into ``spectrum.data``/``data_imag``/``axes``. This is
        what makes every FID-processing control (window/zero-fill/
        shift/align-echo/linear-predict/truncate) live for a pseudo-2D
        with "Apply to every slice" checked, or a conventional 2D --
        not just the phase/baseline controls, which already had their
        own live path. Explicit, repeated user feedback: "Apply
        Processing is still resetting everything" -- removed entirely
        in favour of this (see :meth:`_on_fid_processing_control_changed`),
        rather than patched again.

        For a conventional 2D, also redoes the F1 FFT -- F2's own
        reprocessing changes every row F1's FFT operates across, so F1
        must be redone too, using ``metadata.raw["fid_dt_f1"]`` (F1's
        own dt, stashed at load time) and F1's *own* already-loaded
        axis length (unaffected by an F2-only control, so zero-filling
        back to it exactly reproduces F1's own resolution) -- then
        reapplies whatever F1 phase is already durably committed
        (``metadata.raw["phase0_f1"/"phase1_f1"]``, see
        :meth:`_on_baseline_correct_toggled`'s own sibling mechanism
        for phase, :meth:`_commit_pending_phase_to_metadata`), since a
        fresh FFT has none. F2's own committed phase
        (``metadata.raw["phase0"/"phase1"]``) is reapplied the same
        way, unconditionally -- both baked-in totals, not whatever the
        Phase spin boxes happen to be showing right now (which may be
        approxied at a wholly different target, e.g. a single pseudo-2D
        slice) -- consistent, dimension-by-dimension.

        Returns ``False`` (spectrum left untouched) on the same
        "can't compute this way" conditions
        :meth:`_compute_processing_preview` documents; ``True`` on
        success."""
        raw_fid = spectrum.metadata.raw.get("fid_raw")
        dt = spectrum.metadata.raw.get("fid_dt")
        if raw_fid is None or dt is None or raw_fid.ndim != 2:
            return False

        rows = []
        filled_rows = []
        for row in raw_fid:
            computed = self._process_raw_fid_row(spectrum, row, dt)
            if computed is None:
                return False
            filled, row_data = computed
            filled_rows.append(filled)
            rows.append(row_data)
        data = np.array(rows)
        # Same reason as _commit_processing_preview_as_spectrum's own
        # update of this -- keep the FID preview tab's source in step,
        # not stale from whatever was last loaded/reset.
        spectrum.metadata.raw["fid"] = np.array(filled_rows)
        spectrum.metadata.raw["fid_dt"] = dt

        reference_ppm = spectrum.axes[-1].ppm
        ppm2 = np.linspace(reference_ppm[0], reference_ppm[-1], data.shape[1])
        hz2 = ppm2 * (spectrum.metadata.sfo1 or 1.0)

        phase0 = float(spectrum.metadata.raw.get("phase0", 0.0))
        phase1 = float(spectrum.metadata.raw.get("phase1", 0.0))
        span2 = ppm2.max() - ppm2.min()
        pivot_frac2 = (self.pivot_spin.value() - ppm2.min()) / span2 if span2 else 0.5
        data = processing_ft.apply_phase(data, ppm2, phase0, phase1, pivot_frac=pivot_frac2)

        dt_f1 = spectrum.metadata.raw.get("fid_dt_f1")
        f1_axis = spectrum.axes[0]
        if not spectrum.is_pseudo_2d and dt_f1:
            fn_mode = spectrum.metadata.raw.get("fn_mode", 1)
            # F1 truncate -- row-pair aware (a request never separates
            # a real row from its matching imaginary one), same
            # convention as io.bruker.read_2d/io.xpulse.read_2d's own
            # truncate_f1. `data` here still has the raw, undecoded row
            # count (every row of fid_raw processed independently
            # above), so this must happen *before* the hypercomplex
            # decode just below, same ordering read_2d itself uses.
            live_truncate_f1 = spectrum.metadata.raw.get("live_truncate_f1")
            if isinstance(live_truncate_f1, int) and not isinstance(live_truncate_f1, bool):
                n1_native = data.shape[0] // 2 if fn_mode in (4, 5, 6) else data.shape[0]
                if 0 < live_truncate_f1 < n1_native:
                    raw_row_count = 2 * live_truncate_f1 if fn_mode in (4, 5, 6) else live_truncate_f1
                    data = data[:raw_row_count]
            # Same hypercomplex row-pair decode read_2d itself does,
            # before its own F1 FFT -- skipping this would silently
            # feed States/States-TPPI/Echo-Antiecho data into zero_fill
            # at the wrong row count instead of a genuine hypercomplex
            # point per t1 increment, the moment any F2-only control
            # changes live -- exactly the "two pipelines quietly
            # diverge" bug this whole round's own architecture change
            # was about closing.
            data = processing_ft.decode_hypercomplex_f1(data, fn_mode)
            live_zero_fill_f1 = spectrum.metadata.raw.get("live_zero_fill_f1")
            if isinstance(live_zero_fill_f1, int) and not isinstance(live_zero_fill_f1, bool):
                si1 = live_zero_fill_f1
            else:
                # F1's own current (already-decoded) length -- unaffected
                # by F2-only controls, absent an explicit live F1 choice.
                si1 = spectrum.data.shape[0]
            f1_filled = np.apply_along_axis(lambda col: processing_ft.zero_fill(col, si1), 0, data)
            data = np.apply_along_axis(lambda col: processing_ft.fft_spectrum(col, dt_f1)[1], 0, f1_filled)
            # F1's ppm span (start/end) depends only on the acquisition
            # itself (transmitter offset + spectral width), never on
            # how many points it's sampled into -- same reasoning
            # ppm2/hz2 above already relies on for F2's own zero-fill.
            # Re-deriving from the existing axis's own endpoints (rather
            # than re-deriving o1_1/sf_1/obs_1/sw_1 from vendor-specific
            # raw params) keeps this vendor-agnostic, matching how this
            # whole method already avoids vendor-specific formulas.
            old_ppm1 = f1_axis.ppm
            ppm1 = np.linspace(old_ppm1[0], old_ppm1[-1], si1) if si1 != len(old_ppm1) else old_ppm1
            hz1 = ppm1 * (spectrum.metadata.sfo1_indirect or spectrum.metadata.sfo1 or 1.0)
            f1_axis = NMRAxis(kind="frequency", ppm=ppm1, hz=hz1)
            phase0_f1 = float(spectrum.metadata.raw.get("phase0_f1", 0.0))
            phase1_f1 = float(spectrum.metadata.raw.get("phase1_f1", 0.0))
            if phase0_f1 or phase1_f1:
                span1 = ppm1.max() - ppm1.min()
                pivot_frac1 = (self.pivot_spin.value() - ppm1.min()) / span1 if span1 else 0.5
                data = processing_ft.apply_phase(data.T, ppm1, phase0_f1, phase1_f1, pivot_frac=pivot_frac1).T

        data = self._baseline_correct_if_enabled_multi_row(data, ppm2, spectrum)
        spectrum.axes = [f1_axis, NMRAxis(kind="frequency", ppm=ppm2, hz=hz2)]
        spectrum.data = np.real(data)
        spectrum.data_imag = np.imag(data)
        return True

    @staticmethod
    def _baseline_correct_if_enabled_multi_row(data, freq_ppm, spectrum):
        """:meth:`_baseline_correct_if_enabled`, applied per row --
        baseline correction is always a per-row fit (each row's own
        baseline, not one shared across a whole 2D dataset)."""
        if not spectrum.metadata.raw.get("baseline_correct"):
            return data
        return np.apply_along_axis(
            lambda row: MainWindow._baseline_correct_if_enabled(row, freq_ppm, spectrum), 1, data,
        )

    def _recompute_multi_row_phase_and_baseline_only(self, spectrum) -> bool:
        """:meth:`_recompute_multi_row_data`'s fallback for a spectrum
        with no raw FID cached at all (a processed-only fallback
        load) -- re-derives phase+baseline from
        :attr:`_phase_reference_full` (the untouched-since-editing-
        started snapshot :meth:`_capture_phase_reference` took), the
        same computation :meth:`_apply_phase_preview`'s own
        (non-debounced) pseudo2d/2d "apply to every slice"/whole-
        dataset branches use, just callable from the debounced timer
        instead of inline on every tick. A no-op (returns ``False``)
        if the phase-editing context has moved on since this was
        scheduled (reselected, or the dimension/mode changed) --
        conservative rather than guessing at a target that may no
        longer be the right one."""
        if self._phase_mode not in ("pseudo2d", "2d"):
            return False
        if self._phase_reference_full is None or self._phase_freq_ppm is None:
            return False
        freq_ppm = self._phase_freq_ppm
        phase0 = float(spectrum.metadata.raw.get("phase0", 0.0)) + self.phase0_spin.value()
        phase1 = float(spectrum.metadata.raw.get("phase1", 0.0)) + self.phase1_spin.value()
        span = freq_ppm.max() - freq_ppm.min()
        pivot_frac = (self.pivot_spin.value() - freq_ppm.min()) / span if span else 0.5
        phased = processing_ft.apply_phase(self._phase_reference_full, freq_ppm, phase0, phase1, pivot_frac=pivot_frac)
        phased = self._baseline_correct_if_enabled_multi_row(phased, freq_ppm, spectrum)
        spectrum.data = np.real(phased)
        spectrum.data_imag = np.imag(phased)
        return True

    def _show_processing_preview(self, spectrum, index: int, switch_tab: bool = False) -> None:
        """Recompute and show the Processing tab's own live dual
        preview (FID top, downsampled spectrum bottom) for ``spectrum``
        -- exactly what "Apply Processing" would currently produce.
        Computed entirely in memory (see
        :meth:`_compute_processing_preview`), not via a disk re-read --
        cheap enough to call on every live control change, including a
        continuous phase-drag tick.

        Falls back to plainly mirroring the Spectrum tab's own live
        view (:meth:`MplCanvas.show_spectrum` against this same
        ``spectrum`` object -- the same source, not a second
        computation of it) for anything the FID+spectrum dual preview
        can't sensibly show: not FID-reprocessable at all, or a
        conventional (non-pseudo) 2D, whose F1 FID isn't captured
        either -- same rule :func:`_fid_preview_available` already
        uses for the real FID tab. Explicit user feedback: this tab
        going entirely blank for a real, F1-processed 2D (or whenever
        the raw-FID reprocess path itself failed) looked like the
        figure had "disappeared," when there was a perfectly good
        current spectrum to show all along -- just not one this
        method's own from-the-raw-FID preview knows how to rebuild."""
        if not self._fid_reprocessable(spectrum) or (spectrum.ndim == 2 and not spectrum.is_pseudo_2d):
            self.processing_preview_canvas.show_spectrum(spectrum)
        else:
            try:
                computed = self._compute_processing_preview(spectrum)
            except Exception:  # noqa: BLE001 -- best-effort live preview; a transiently-invalid
                # combination of controls mid-edit shouldn't pop an error dialog
                computed = None
            if computed is None:
                self.processing_preview_canvas.show_spectrum(spectrum)
            else:
                fid, dt, slice_spectrum, title = computed
                envelope = self._compute_window_envelope(dt, len(fid))
                tdeff = self.tdeff_spin.value() if self.tdeff_spin.isEnabled() else 0
                echo_top = processing_ft.find_echo_top(fid) if self.align_echo_button.isChecked() else None
                self.processing_preview_canvas.show_processing_preview(
                    fid, dt, slice_spectrum, title=title, window_envelope=envelope,
                    tdeff_index=tdeff or None, echo_top_index=echo_top,
                    fid_mode=self.fid_display_combo.currentData(),
                    fid_x_unit=self.fid_x_unit_combo.currentData(),
                )
        if switch_tab:
            self.canvas_tabs.setCurrentWidget(self.processing_preview_canvas)

    def _refresh_processing_preview_if_active(self) -> None:
        """Live-updates the Processing tab's own preview as a control
        changes, but only while that canvas tab is actually showing --
        same gating pattern as :meth:`_refresh_fid_preview_if_active`.
        A full rebuild (:meth:`_show_processing_preview`), so both
        panels' views reset -- correct for anything that can change
        the FID's own shape (window, zero-fill, shift, TDeff, LP,
        slice) or that's a discrete "just fix it" action where
        re-fitting the view to the new result makes more sense than
        preserving the old one (Autophase, Revert Autophase, Baseline
        Correct -- matches :meth:`MplCanvas.autoscale_y`'s own
        established reasoning for the same three actions on the
        Spectrum tab). For a live phase *drag*/typed-value change
        specifically, see :meth:`_refresh_processing_preview_phase_only`
        instead."""
        if self.canvas_tabs.currentWidget() is not self.processing_preview_canvas:
            return
        index = self._selected_index()
        if index is None:
            return
        self._show_processing_preview(self.session.spectra[index], index, switch_tab=False)

    def _refresh_processing_preview_phase_only(self) -> None:
        """Live-updates the Processing tab's own preview for a phase
        change specifically (drag, spin box, pivot) -- in place,
        preserving whatever zoom/pan is already on the spectrum panel,
        via :meth:`MplCanvas.update_processing_preview_spectrum`.
        Explicit user request: phasing should be possible while zoomed
        into a small peak, without the view snapping back out on every
        tick the way a full rebuild (:meth:`_refresh_processing_preview_if_active`)
        would. Phase never touches the FID, so the FID panel above is
        never touched either -- only the spectrum panel's data.

        Falls back to the full rebuild anyway when the in-place update
        can't be done (:meth:`MplCanvas.update_processing_preview_spectrum`
        returns ``False``) -- possible even here despite the "phase-only"
        framing, since the live recompute this pulls from also picks up
        window/zero-fill/etc. still sitting unapplied on the Processing
        tab, not just phase."""
        if self.canvas_tabs.currentWidget() is not self.processing_preview_canvas:
            return
        index = self._selected_index()
        if index is None:
            return
        spectrum = self.session.spectra[index]
        try:
            computed = self._compute_processing_preview(spectrum)
        except Exception:  # noqa: BLE001 -- best-effort live preview, see _show_processing_preview
            return
        if computed is None:
            return
        _fid, _dt, preview_spectrum, _title = computed
        updated = self.processing_preview_canvas.update_processing_preview_spectrum(preview_spectrum)
        if not updated:
            self._show_processing_preview(spectrum, index, switch_tab=False)
            return
        self.processing_preview_canvas.set_pivot_marker(self.pivot_spin.value())

    def _update_processing_preview_tab_visibility(self, spectrum) -> None:
        """Show the Processing Preview canvas tab only for a spectrum
        the FID-processing box itself would show controls for -- same
        gate as ``fid_processing_box`` (see
        :meth:`_refresh_fid_processing_controls`)."""
        index = self.canvas_tabs.indexOf(self.processing_preview_canvas)
        self.canvas_tabs.setTabVisible(index, spectrum is not None and self._fid_reprocessable(spectrum))

    def _on_side_tab_changed(self, index: int) -> None:
        """Entering the Processing (left) tab switches the canvas over
        to its own live preview automatically (see
        :meth:`_show_processing_preview`) -- "while in the processing
        tab, the right side should show" it, unprompted. Leaving
        switches back to the plain Spectrum view, but only if the
        preview was actually the one showing; if the user had manually
        navigated to e.g. Compare or Fit while still on the Processing
        tab, leaving shouldn't yank them away from that choice.

        Also hides the pivot marker on both canvases -- it's only
        meaningful while the Phase controls (on this same left tab)
        are the thing being actively worked with; left lingering as a
        stray dashed line once the user has moved on to Data/Analysis/
        Compare, it reads as leftover state rather than the live pivot
        point it was showing."""
        if index != self._processing_tab_index:
            self.canvas.set_pivot_marker(None)
            self.processing_preview_canvas.set_pivot_marker(None)
            if self.canvas_tabs.currentWidget() is self.processing_preview_canvas:
                self.canvas_tabs.setCurrentWidget(self.canvas)
            return
        selected = self._selected_index()
        spectrum = self.session.spectra[selected] if selected is not None else None
        if spectrum is not None and self._fid_reprocessable(spectrum):
            self._show_processing_preview(spectrum, selected, switch_tab=True)

    def _update_fit_tab_visibility(self, spectrum) -> None:
        """Show the Fit canvas tab only while there's something it
        could plausibly show: the currently selected spectrum is
        pseudo-2D (so a right-click Fit Curve on it is available), or
        ``spectrum`` is ``None`` (nothing selected -- hide). A
        Compare-stack fit (which doesn't depend on what's selected in
        the Data tab at all) force-shows this tab directly in
        :meth:`_run_curve_fit` instead of going through this check."""
        index = self.canvas_tabs.indexOf(self.fit_canvas)
        visible = spectrum is not None and spectrum.ndim == 2 and spectrum.is_pseudo_2d
        self.canvas_tabs.setTabVisible(index, visible)

    def _update_compare_tab_visibility(self) -> None:
        """Show the Compare canvas tab only once the (left-hand)
        Compare tab's own list actually has something in it."""
        index = self.canvas_tabs.indexOf(self.compare_canvas)
        self.canvas_tabs.setTabVisible(index, self.compare_list.count() > 0)

    def _refresh_fid_processing_controls(self, spectrum, index: int) -> None:
        reprocessable = self._fid_reprocessable(spectrum)
        self.fid_processing_box.setVisible(reprocessable)

        # Restore this spectrum's own last-used FID-processing settings
        # from its own metadata.raw (see
        # _persist_fid_processing_controls_to_metadata) rather than
        # always resetting to defaults, so switching away and back
        # doesn't lose an in-progress choice. Not session.get_load_
        # kwargs any more -- that only ever updated on an actual
        # session.reload call, which no longer happens for a live
        # control change now that there's no separate "Apply
        # Processing" step (explicit user report that processing was
        # "still reset when changing between experiments" once these
        # became live without this).
        raw = spectrum.metadata.raw if reprocessable else {}

        is_xpulse = spectrum.metadata.vendor == "xpulse"
        self.tdeff_spin.setEnabled(reprocessable and is_xpulse)
        self.tdeff_spin.blockSignals(True)
        # Max (and default, absent a saved/recorded choice) is this
        # spectrum's own actual raw-FID length -- lets the spin box's
        # step-down arrows/scroll trim points off the end starting from
        # a meaningful place, rather than needing to count up from 1 to
        # reach a large useful value first.
        raw_fid = spectrum.metadata.raw.get("fid_raw") if is_xpulse else None
        raw_length = len(raw_fid) if raw_fid is not None else (1 << 20)
        self.tdeff_spin.setMaximum(raw_length)
        saved_truncate = raw.get("live_truncate")
        if isinstance(saved_truncate, int) and not isinstance(saved_truncate, bool):
            default_tdeff = min(saved_truncate, raw_length)
        else:
            recorded_tdeff = int(spectrum.metadata.raw.get("exp_params", {}).get("TDEFF", 0) or 0) if is_xpulse else 0
            default_tdeff = recorded_tdeff if recorded_tdeff else raw_length
        self.tdeff_spin.setValue(max(1, default_tdeff))
        self.tdeff_spin.blockSignals(False)

        # F1 (indirect dimension) TD/SI -- vendor-agnostic (unlike F2's
        # tdeff_spin above, which only applies to X-Pulse), gated on
        # "genuine 2D with a real F1 FFT" -- not pseudo-2D, where "F1"
        # is a slice index, not a frequency dimension, and not 1D,
        # which has no F1 at all. Mirrors the same is_2d computation
        # _refresh_selection_display uses for the Dimension controls'
        # own visibility, computed directly from spectrum rather than
        # from self._phase_mode, which _capture_phase_reference doesn't
        # (re)set until *after* this method returns (see
        # _refresh_selection_display's own call order).
        is_genuine_2d = reprocessable and spectrum.ndim == 2 and not spectrum.is_pseudo_2d
        raw_fid_f1 = spectrum.metadata.raw.get("fid_raw") if is_genuine_2d else None
        fn_mode = spectrum.metadata.raw.get("fn_mode", 1)
        if raw_fid_f1 is not None and raw_fid_f1.ndim == 2:
            n1_native = raw_fid_f1.shape[0] // 2 if fn_mode in (4, 5, 6) else raw_fid_f1.shape[0]
        else:
            n1_native = spectrum.data.shape[0] if is_genuine_2d else (1 << 20)

        self.tdeff_f1_spin.setEnabled(is_genuine_2d)
        self.tdeff_f1_spin.blockSignals(True)
        self.tdeff_f1_spin.setMaximum(max(1, n1_native))
        saved_truncate_f1 = raw.get("live_truncate_f1")
        if isinstance(saved_truncate_f1, int) and not isinstance(saved_truncate_f1, bool):
            default_tdeff_f1 = min(saved_truncate_f1, n1_native)
        else:
            default_tdeff_f1 = n1_native
        self.tdeff_f1_spin.setValue(max(1, default_tdeff_f1))
        self.tdeff_f1_spin.blockSignals(False)

        self.zero_fill_f1_spin.setEnabled(is_genuine_2d)
        self.zero_fill_f1_spin.blockSignals(True)
        # F1's own current (already-decoded) length is the no-op
        # default -- matching _recompute_multi_row_data's own fallback
        # when no live choice has been made yet.
        default_si1 = spectrum.data.shape[0] if is_genuine_2d else 1
        saved_zero_fill_f1 = raw.get("live_zero_fill_f1")
        if isinstance(saved_zero_fill_f1, int) and not isinstance(saved_zero_fill_f1, bool):
            default_si1 = saved_zero_fill_f1
        self.zero_fill_f1_spin.setMaximum(max(default_si1, n1_native, 1) * 4)
        self.zero_fill_f1_spin.setValue(max(1, default_si1))
        self.zero_fill_f1_spin.blockSignals(False)

        if not reprocessable:
            return

        self._populate_window_combo(spectrum.metadata.vendor)
        self._set_window_combo_key(raw.get("live_window", "__recorded__"))

        for spin, value in (
            (self.lb_spin, raw.get("live_lb", 0.0)),
            (self.fid_shift_spin, raw.get("live_fid_shift", 0.0)),
            (self.linear_predict_points_spin, raw.get("live_linear_predict_points", 0)),
        ):
            spin.blockSignals(True)
            spin.setValue(value)
            spin.blockSignals(False)
        self.linear_predict_order_spin.blockSignals(True)
        self.linear_predict_order_spin.setValue(raw.get("live_linear_predict_order", 20))
        self.linear_predict_order_spin.blockSignals(False)
        self.linear_predict_direction_combo.blockSignals(True)
        direction_index = self.linear_predict_direction_combo.findData(raw.get("live_linear_predict_direction", "forward"))
        self.linear_predict_direction_combo.setCurrentIndex(max(0, direction_index))
        self.linear_predict_direction_combo.blockSignals(False)

        self.align_echo_button.blockSignals(True)
        self.align_echo_button.setChecked(raw.get("live_align_echo", False))
        self.align_echo_button.blockSignals(False)

        # Absent an explicit live-set override, default the combo to
        # this spectrum's own *actual current* length -- not the "2x
        # NP" conventional suggestion (_default_zero_fill_size), which
        # is a proposal for a *future* change, not necessarily what
        # the vendor reader's own default zero-fill actually produced
        # (it regularly doesn't -- e.g. X-Pulse's own default is no
        # zero-fill beyond the raw length at all). Showing the
        # suggestion here instead of reality meant every live recompute
        # that reads this combo silently zero-filled a freshly-loaded,
        # never-reprocessed spectrum the moment *any* control was
        # touched, even one that has nothing to do with zero-fill.
        # F2's own point count specifically -- for a 2D/pseudo-2D
        # spectrum, len(spectrum.data) is the *row* (F1) count, not
        # F2's, which is what zero-fill actually resizes (real bug,
        # found via a live-recompute round trip silently shrinking a
        # conventional 2D's own F2 dimension down to its F1 row count).
        f2_length = spectrum.data.shape[-1]
        default_zero_fill = raw.get("live_zero_fill", f2_length)
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

    def _persist_fid_processing_controls_to_metadata(self, spectrum) -> None:
        """Stash the FID-processing controls' own current values into
        ``spectrum.metadata.raw`` -- same "each experiment object has
        its own metadata locked in for when it's called upon"
        mechanism ``phase0``/``phase1``/``baseline_correct`` already
        use (see :meth:`_capture_phase_reference`'s own restore of
        those), extended to every other live control now that there's
        no separate "Apply Processing" commit step to persist them via
        ``session.get_load_kwargs`` (that only ever updated on an
        actual ``session.reload`` call, which no longer happens for a
        live control change -- explicit user report that processing
        state was "still reset when changing between experiments"
        once these became live without this). Restored by
        :meth:`_refresh_fid_processing_controls` on reselection."""
        raw = spectrum.metadata.raw
        raw["live_window"] = self.window_combo.currentData()
        raw["live_lb"] = self.lb_spin.value()
        raw["live_zero_fill"] = self._zero_fill_value()
        raw["live_fid_shift"] = self.fid_shift_spin.value()
        raw["live_align_echo"] = self.align_echo_button.isChecked()
        raw["live_linear_predict_points"] = self.linear_predict_points_spin.value()
        raw["live_linear_predict_order"] = self.linear_predict_order_spin.value()
        raw["live_linear_predict_direction"] = self.linear_predict_direction_combo.currentData()
        if self.tdeff_spin.isEnabled():  # X-Pulse only -- see tdeff_spin's own tooltip
            raw["live_truncate"] = self.tdeff_spin.value()
        if self.tdeff_f1_spin.isEnabled():  # genuine 2D only, any vendor -- see tdeff_f1_spin's own tooltip
            raw["live_truncate_f1"] = self.tdeff_f1_spin.value()
        if self.zero_fill_f1_spin.isEnabled():
            raw["live_zero_fill_f1"] = self.zero_fill_f1_spin.value()
        self._schedule_sidecar_save()

    def _on_fid_processing_control_changed(self) -> None:
        """Unified live-update handler for every FID-processing
        control (window/zero-fill/shift/align-echo/linear-predict/
        truncate) -- replaces "Apply Processing" entirely. Explicit,
        repeated user feedback ("Apply processing is still resetting
        everything... remove the button, we can have a reset button"):
        a separate commit step kept resetting state -- phase, baseline,
        whatever the last round's fix hadn't caught up to yet -- that
        had nothing to do with what was actually being changed, because
        it went through a full ``session.reload`` (a fresh disk read
        and reprocess) instead of updating the spectrum already in
        memory. This updates ``spectrum.data`` directly instead, the
        same "the live preview *is* the spectrum, not a separate draft"
        principle phase/baseline already established.

        Only 1D recomputes immediately
        (:meth:`_commit_processing_preview_as_spectrum`, already fast
        -- one row). Every multi-row case (pseudo-2D, conventional 2D)
        always goes through the debounced full recompute
        (:meth:`_schedule_multi_row_reprocess`), regardless of
        "Apply to every slice" -- that checkbox is a *phase*-targeting
        choice (which row(s) a phase drag currently affects), not a
        FID-domain-processing one: window/zero-fill/etc. were never a
        per-slice concept, so updating only the currently-visible row
        and leaving the rest stale until something else happened to
        touch them would reintroduce exactly the kind of inconsistent-
        between-views state this whole round is about removing (e.g. a
        Fit Curve run across every slice silently using stale,
        un-rewindowed data for every row except whichever one had been
        on screen most recently). Debouncing rather than running the
        full per-row cost on every intermediate keystroke/spin-box
        tick is what actually fixes the "hangs" report -- the single
        most expensive per-row step (spline baseline correction)
        firing on every tick, not just once after things settle, is
        indistinguishable from a hang once the Qt event queue backs up
        behind it."""
        index = self._selected_index()
        if index is None or self._phase_mode is None:
            return
        spectrum = self.session.spectra[index]
        if not self._fid_reprocessable(spectrum):
            return
        self._persist_fid_processing_controls_to_metadata(spectrum)

        if self._phase_mode == "1d":
            self._commit_processing_preview_as_spectrum(spectrum)
        else:
            self._schedule_multi_row_reprocess(index)
        if self.canvas_tabs.currentWidget() is self.fid_canvas and self._fid_preview_available(spectrum):
            self._show_fid_preview(spectrum, switch_tab=False)
        self._refresh_processing_preview_if_active()

    def _schedule_sidecar_save(self, index: int | None = None) -> None:
        """(Re)start the debounce timer for writing spectrum
        ``index``'s (default: whatever's currently selected) two
        on-disk sidecar files (see :mod:`opennmr.io.persistence`) --
        called after any live FID-processing control, phase commit,
        baseline toggle, analysis-state change (peaks/regions/fit
        settings), or metadata override (mass/nucleus/SR shift/
        imaging gradient) changes. A no-op if nothing is selected --
        e.g. a metadata change made via a right-click action on a
        *different* row than the one currently shown still passes its
        own ``index`` explicitly rather than relying on the default."""
        target = index if index is not None else self._selected_index()
        if target is None:
            return
        self._sidecar_save_pending_index = target
        self._sidecar_save_timer.start(800)

    def _run_debounced_sidecar_save(self) -> None:
        index = self._sidecar_save_pending_index
        self._sidecar_save_pending_index = None
        if index is None or index >= len(self.session):
            return
        spectrum = self.session.spectra[index]
        io_persistence.save_metadata_sidecar(spectrum)
        io_persistence.save_processing_sidecar(spectrum)

    def _schedule_multi_row_reprocess(self, index: int) -> None:
        """(Re)start the debounce timer for a full
        :meth:`_recompute_multi_row_data` recompute of spectrum
        ``index`` -- restarting on every call (rather than letting an
        already-running timer finish) is exactly what coalesces a
        rapid run of changes into one recompute a short idle period
        after the *last* one, not the first."""
        self._live_reprocess_pending_index = index
        self._live_reprocess_timer.start(200)

    def _run_debounced_multi_row_reprocess(self) -> None:
        index = self._live_reprocess_pending_index
        self._live_reprocess_pending_index = None
        if index is None or index >= len(self.session):
            return
        spectrum = self.session.spectra[index]
        if self._fid_reprocessable(spectrum):
            success = self._recompute_multi_row_data(spectrum)
        else:
            # No raw FID to redo the FID-domain steps from at all (a
            # processed-only fallback load) -- _recompute_multi_row_data
            # can't help there. Just phase+baseline, the same
            # computation the (non-debounced) inline pseudo2d/2d
            # branches of _apply_phase_preview already do for this
            # exact case, deferred here to settle-time instead so a
            # multi-row processed-only dataset gets the same "at most
            # one recompute per drag/edit session" protection.
            success = self._recompute_multi_row_phase_and_baseline_only(spectrum)
        if not success:
            return
        self.spectrum_list.item(index).setText(self.session.label(index))
        if index != self._selected_index():
            return  # recomputed and stored either way -- just not the currently-visible view
        if self._phase_mode == "pseudo2d":
            on_processing_preview = self.canvas_tabs.currentWidget() is self.processing_preview_canvas
            self._show_pseudo2d_slice(spectrum, reset_view=False, switch_tab=False)
            if self.canvas_tabs.currentWidget() is self.fid_canvas and self._fid_preview_available(spectrum):
                self._show_fid_preview(spectrum, switch_tab=False)
            if on_processing_preview:
                self._refresh_processing_preview_if_active()
        elif self._phase_mode == "2d":
            self.canvas.show_spectrum(spectrum)
            if self.canvas_tabs.currentWidget() is self.processing_preview_canvas:
                self.processing_preview_canvas.show_spectrum(spectrum)

    def _on_reset_processing_clicked(self) -> None:
        """Discard every live processing choice for the selected
        spectrum -- window/zero-fill/shift/align-echo/linear-predict/
        truncate, phase, baseline -- reverting to exactly how it
        looked right after it was first loaded (see ``_load``'s own
        snapshot into :attr:`_original_load_kwargs`). The opposite of
        the old "Apply Processing" button (which committed changes
        forward, unreliably); this discards them instead -- explicit
        user request: "remove the button, we can have a reset
        button"."""
        index = self._selected_index()
        if index is None:
            return
        original = self._original_load_kwargs.get(index, {})
        try:
            self.session.reset(index, **original)
        except Exception as exc:  # noqa: BLE001 -- surfaced to the user, not swallowed
            QMessageBox.critical(self, "Failed to reset spectrum", str(exc))
            return
        self.spectrum_list.item(index).setText(self.session.label(index))
        self._on_selection_changed(index)
        # Without this, Reset only discards the *in-memory* state --
        # an old saved processing sidecar (see io.persistence) would
        # still be sitting on disk and silently resurrect exactly what
        # was just reset the next time this experiment is reopened.
        self._schedule_sidecar_save(index)

    def _current_phase_kwargs(self, spectrum) -> dict:
        """The *total* current phase (whatever was already baked in
        by the last load/reprocess, plus the live delta still showing
        in the Phase spin boxes) as ``read_1d``/``read_2d``'s own
        ``phase0``/``phase1``/``pivot_frac``/``autophase=False``
        kwargs -- so a raw-FID reprocess reproduces the exact phase
        currently on screen instead of resetting to the vendor's own
        default (every vendor's ``read_1d``/``read_2d`` bakes its own
        applied ``phase0``/``phase1`` into ``metadata.raw``, which is
        exactly the "already baked in" half of this sum; the Phase
        spin boxes are deliberately *deltas* since editing started --
        see :meth:`_capture_phase_reference` -- not the total, so
        adding them on their own would under-correct).

        Returns ``{}`` (no override -- vendor default applies,
        unchanged) when nothing is phase-editable, when ``spectrum``
        isn't actually being re-read from a raw FID at all (e.g. a
        Bruker experiment with no ``fid``/``ser`` on disk -- routed
        through ``read_1d_processed``/``read_2d_processed``, which has
        no ``phase0``/``phase1`` concept at all, unlike ``read_1d``/
        ``read_2d``), or when the current Phase-controls target isn't
        a single scalar phase shared across the whole dataset along F2
        -- a pseudo-2D slice's own phase (not every slice's shared
        one), or a conventional 2D's F1/indirect phase -- since
        ``read_2d`` has no way to apply either of those on its own;
        see :meth:`_reapply_phase_post_reload` for how those two cases
        are instead carried through post-hoc."""
        if self._phase_mode is None or self._phase_freq_ppm is None:
            return {}
        if not self._fid_reprocessable(spectrum):
            return {}
        if self._phase_mode == "pseudo2d" and not self.apply_all_slices_checkbox.isChecked():
            return {}
        if self._phase_mode == "2d" and self.dim_combo.currentIndex() != 0:
            return {}
        freq_ppm = self._phase_freq_ppm
        span = freq_ppm.max() - freq_ppm.min()
        pivot_frac = (self.pivot_spin.value() - freq_ppm.min()) / span if span else 0.5
        baked_in = spectrum.metadata.raw
        return {
            "phase0": float(baked_in.get("phase0", 0.0)) + self.phase0_spin.value(),
            "phase1": float(baked_in.get("phase1", 0.0)) + self.phase1_spin.value(),
            "pivot_frac": pivot_frac,
            "autophase": False,
        }

    def _reapply_phase_post_reload(self, spectrum, previous_raw: dict | None = None) -> None:
        """Companion to :meth:`_current_phase_kwargs`: for the two
        cases it can't forward into the reload itself -- one pseudo-2D
        slice's own phase, or a conventional 2D's F1/indirect phase --
        reapply that same phase directly to the freshly-reloaded
        spectrum post-hoc, so at least what's actually being looked at
        keeps its phase instead of silently resetting. A no-op outside
        those two cases, or if the total is zero, or if the reload
        changed what the phase target even means (e.g. switching to
        pseudo-2D drops F1's frequency meaning entirely).

        ``previous_raw`` -- the *pre-reload* spectrum's own
        ``metadata.raw`` (the caller's own local ``spectrum`` variable,
        captured before ``session.reload`` replaced it) -- is where the
        F1 branch's own persisted ``"phase0_f1"/"phase1_f1"`` (see
        :meth:`_commit_pending_phase_to_metadata`) is read from: the
        *freshly*-reloaded spectrum's own ``metadata.raw`` never has
        these (unlike F2's ``"phase0"/"phase1"``, ``read_2d`` doesn't
        accept or record an F1 phase kwarg at all -- see
        :meth:`_current_phase_kwargs`'s own docstring), so without this
        the total would silently drop back to just the live spin-box
        delta, i.e. the exact loss this method exists to prevent."""
        if spectrum.ndim != 2 or spectrum.data_imag is None:
            return

        if self._phase_mode == "pseudo2d" and not self.apply_all_slices_checkbox.isChecked():
            phase0 = self.phase0_spin.value()
            phase1 = self.phase1_spin.value()
            if not phase0 and not phase1:
                return
            if not spectrum.is_pseudo_2d:
                return
            row = self.slice_spin.value() - 1
            if not (0 <= row < spectrum.data.shape[0]):
                return
            freq_ppm = spectrum.axis_values(-1, "ppm")
            span = freq_ppm.max() - freq_ppm.min()
            pivot_frac = (self.pivot_spin.value() - freq_ppm.min()) / span if span else 0.5
            complex_row = spectrum.data[row] + 1j * spectrum.data_imag[row]
            phased_row = processing_ft.apply_phase(complex_row, freq_ppm, phase0, phase1, pivot_frac=pivot_frac)
            spectrum.data[row] = np.real(phased_row)
            spectrum.data_imag[row] = np.imag(phased_row)

        elif self._phase_mode == "2d" and self.dim_combo.currentIndex() != 0:
            if spectrum.is_pseudo_2d:
                return  # F1 no longer a frequency axis -- nothing to reapply
            baked_in = previous_raw or {}
            phase0 = float(baked_in.get("phase0_f1", 0.0)) + self.phase0_spin.value()
            phase1 = float(baked_in.get("phase1_f1", 0.0)) + self.phase1_spin.value()
            if not phase0 and not phase1:
                return
            freq_ppm = spectrum.axis_values(0, "ppm")
            span = freq_ppm.max() - freq_ppm.min()
            pivot_frac = (self.pivot_spin.value() - freq_ppm.min()) / span if span else 0.5
            phased = processing_ft.apply_phase(
                spectrum.complex_data.T, freq_ppm, phase0, phase1, pivot_frac=pivot_frac,
            ).T
            spectrum.data = np.real(phased)
            spectrum.data_imag = np.imag(phased)

    def _on_align_echo_toggled(self, _checked: bool) -> None:
        """Align Echo is a push button, not a passive checkbox --
        toggling it immediately reprocesses (not just remembering the
        setting for a later "Reprocess" click) and switches to a view
        that shows the effect right away: a dashed green marker at
        :func:`opennmr.processing.ft.find_echo_top`'s result, which
        should land right at t=0 once genuinely aligned. Previously
        this only took effect silently on the next "Reprocess," with
        no way to tell from the UI alone whether it had actually done
        anything.

        Prefers the Processing-preview tab (already showing both FID
        and spectrum together) if that's what's already active, rather
        than always force-switching to the plain FID tab -- explicit
        user feedback that Align Echo pulled focus out of the
        Processing-preview context it should be the default view for.
        ``_on_fid_processing_control_changed`` above already refreshes
        that preview as a normal part of every control change, so
        there's nothing more to do in that case."""
        if self._selected_index() is None:
            return
        already_on_preview = self.canvas_tabs.currentWidget() is self.processing_preview_canvas
        self._on_fid_processing_control_changed()
        if already_on_preview:
            return
        index = self._selected_index()
        if index is None:
            return
        spectrum = self.session.spectra[index]
        if self._fid_preview_available(spectrum):
            self._show_fid_preview(spectrum, switch_tab=False)
        self.canvas_tabs.setCurrentWidget(self.fid_canvas)

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
            "numbering) the phase controls below act on -- also which "
            "slice the Processing tab's own live preview shows."
        )
        self.slice_spin.valueChanged.connect(self._on_slice_spin_changed)
        slice_row, self.slice_prev_button, self.slice_next_button = (
            self._spin_with_step_buttons(self.slice_spin)
        )
        self.slice_prev_button.setToolTip("Previous slice")
        self.slice_next_button.setToolTip("Next slice")
        self.slice_row_label = QLabel("Slice:")
        dim_form.addRow(self.slice_row_label, slice_row)

        self.apply_all_slices_checkbox = QCheckBox("Apply to every slice")
        self.apply_all_slices_checkbox.setToolTip(
            "Off: Autophase/phase drag/Baseline Correct above only act "
            "on the slice selected above. On: the same phase0/phase1 "
            "(or, for Baseline Correct, an independent baseline fit "
            "per slice) is applied to every slice at once."
        )
        self.apply_all_slices_checkbox.stateChanged.connect(self._on_apply_all_slices_toggled)
        dim_form.addRow(self.apply_all_slices_checkbox)

        self.autophase_slices_button = QPushButton("Autophase (each slice independently)")
        self.autophase_slices_button.setToolTip(
            "Zero-order autophase (grid-search maximizing the real "
            "part's integral) every slice on its own -- unlike the "
            "checkbox above, each slice gets its own phase0, not one "
            "shared value. Immediate, in place; the Phase controls "
            "above re-anchor to the result afterward."
        )
        self.autophase_slices_button.clicked.connect(self._on_autophase_slices_clicked)
        dim_form.addRow(self.autophase_slices_button)

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
        # Phase 1 (linear, applied across the whole spectrum) needs a
        # much larger per-step change than Phase 0 to have a visible
        # effect -- one degree of phase1 barely moves anything, unlike
        # one degree of phase0. ~20x (1-2 orders of magnitude) larger
        # step/drag-sensitivity, per explicit user feedback.
        self.phase1_handle, self.phase1_spin = self._make_phase_control(
            sensitivity=0.2 * _PHASE1_STEP_MULTIPLIER, step=1.0 * _PHASE1_STEP_MULTIPLIER,
        )
        form.addRow("Phase 0 (deg):", self._pair_widget(self.phase0_handle, self.phase0_spin))
        form.addRow("Phase 1 (deg):", self._pair_widget(self.phase1_handle, self.phase1_spin))

        quick_phase_column = QVBoxLayout()
        quick_phase_column.setSpacing(2)
        for delta in (180.0, 90.0, -90.0, -180.0):
            quick_button = QPushButton(f"{delta:+.0f}°")
            quick_button.setToolTip(
                f"Nudge Phase 0 by {delta:+.0f} degrees from its current value -- "
                "a quick fix for a spectrum that's fully inverted (180) or "
                "rotated a quarter-turn (90). Composes with whatever's already "
                "there (two +90 clicks equal one +180), same as typing/dragging "
                "Phase 0 by hand."
            )
            quick_button.clicked.connect(lambda _checked=False, d=delta: self._on_phase0_nudge_clicked(d))
            quick_phase_column.addWidget(quick_button)
        form.addRow("Quick phase:", quick_phase_column)

        self.pivot_spin = RelativeDoubleSpinBox()
        self.pivot_spin.setRange(-1.0e5, 1.0e5)
        self.pivot_spin.setDecimals(3)
        self.pivot_spin.setSuffix(" ppm")
        self.pivot_spin.setToolTip(
            "Phase 1 pivot -- the point held fixed (phase = Phase 0 exactly) "
            "while Phase 1 varies linearly away from it. Defaults to the "
            "centre of the spectrum on selection."
        )
        self.pivot_spin.valueChanged.connect(self._on_pivot_spin_changed)
        form.addRow("Pivot:", self.pivot_spin)

        # A slider covers the spectrum's own ppm range in 1000 steps
        # (mapped in _on_pivot_slider_changed/_sync_pivot_slider_from_spin
        # -- QSlider itself only knows integers) -- quicker for a rough
        # drag-to-position than typing/spinning a ppm value, with the
        # spin box still there for precise entry. The two stay in sync
        # via blockSignals on whichever side didn't originate the
        # change, so moving the pivot never re-triggers
        # _capture_phase_reference (which would zero Phase 0/Phase 1) --
        # only _apply_phase_preview, same as the spin box alone before.
        self.pivot_slider = QSlider(Qt.Orientation.Horizontal)
        self.pivot_slider.setRange(0, 1000)
        self.pivot_slider.setValue(500)
        self.pivot_slider.setToolTip("Drag to move the Phase 1 pivot across the spectrum's own ppm range.")
        self.pivot_slider.valueChanged.connect(self._on_pivot_slider_changed)
        form.addRow(self.pivot_slider)

        autophase_row = QHBoxLayout()
        autophase_button = QPushButton("Autophase")
        autophase_button.setToolTip(
            "Autophase (grid-search maximizing the real part's "
            "integral) for whatever the controls above currently "
            "target -- a plain 1D spectrum, the pseudo-2D slice "
            "selected below (or every slice at once, if 'Apply this "
            "phase to every slice' is checked), or the whole 2D "
            "dataset along the chosen dimension. Zero-order only "
            "(Phase 1 reset to 0) unless '1st-order' is checked; see "
            "'Autophase (each slice independently)' below for a "
            "per-slice-own-phase0 version."
        )
        autophase_button.clicked.connect(self._on_autophase_clicked)
        autophase_row.addWidget(autophase_button)

        self.autophase_joint_checkbox = QCheckBox("1st-order")
        self.autophase_joint_checkbox.setToolTip(
            "Also fit Phase 1 (processing.ft.autophase01's joint "
            "search), not just Phase 0 -- only kept if it "
            "demonstrably improves the lineshape over Phase 0 alone, "
            "same significance test used everywhere else in this app "
            "that offers joint phasing."
        )
        autophase_row.addWidget(self.autophase_joint_checkbox)
        form.addRow(autophase_row)

        self.revert_autophase_button = QPushButton("Revert Autophase")
        self.revert_autophase_button.setToolTip(
            "Undo the last Autophase click, restoring Phase 0/Phase 1 "
            "to what they were immediately before it -- in case the "
            "automatic result was worse than what you already had."
        )
        self.revert_autophase_button.setEnabled(False)
        self.revert_autophase_button.clicked.connect(self._on_revert_autophase_clicked)
        form.addRow(self.revert_autophase_button)

        self.baseline_correct_button = QPushButton("Baseline Correct")
        self.baseline_correct_button.setCheckable(True)
        self.baseline_correct_button.setToolTip(
            "Subtract a robust baseline fit to the real part "
            "(processing.ft.baseline_correct -- sigma-clipped against "
            "the spectrum's own edges so real peaks don't bias it) "
            "from whatever the controls above currently target -- "
            "same targeting rules as Autophase, including 'Apply to "
            "every slice' for a pseudo-2D series. A toggle, not a "
            "one-shot action -- uncheck to remove the correction again."
        )
        self.baseline_correct_button.toggled.connect(self._on_baseline_correct_toggled)
        form.addRow(self.baseline_correct_button)

        baseline_options_row = QHBoxLayout()
        self.baseline_method_combo = QComboBox()
        self.baseline_method_combo.addItem("Linear", "linear")
        self.baseline_method_combo.addItem("Polynomial", "polynomial")
        self.baseline_method_combo.addItem("Sinusoidal", "sinusoidal")
        self.baseline_method_combo.addItem("Spline", "spline")
        self.baseline_method_combo.setToolTip(
            "Linear: a single straight-line fit -- fast, but can't "
            "follow a curved/rolling baseline (common on wideline/"
            "solid-state spectra, and whenever a large recorded first-"
            "order phase leaves curvature concentrated at the edges), "
            "which it will under-correct no matter which points it's "
            "fit to. Polynomial: linear generalized to a higher degree "
            "(set below) -- follows real curvature directly. "
            "Sinusoidal: a slowly rolling, wave-like baseline. Spline: "
            "the most flexible option (a smooth cubic spline), for a "
            "shape none of the above capture -- but also the easiest "
            "to overfit to noise between sparse baseline points."
        )
        self._narrow_combo(self.baseline_method_combo)
        self.baseline_method_combo.currentIndexChanged.connect(self._on_baseline_correct_options_changed)
        baseline_options_row.addWidget(self.baseline_method_combo)

        self.baseline_degree_spin = QSpinBox()
        self.baseline_degree_spin.setRange(1, 8)
        self.baseline_degree_spin.setValue(2)
        self.baseline_degree_spin.setToolTip(
            "Polynomial degree (1 = linear, 2 = quadratic, ...) -- "
            "higher follows more curvature, but can start fitting "
            "noise between baseline points past what the spectrum's "
            "own baseline shape actually needs. Only used when "
            "Polynomial is selected above."
        )
        self.baseline_degree_spin.valueChanged.connect(self._on_baseline_correct_options_changed)
        baseline_options_row.addWidget(self.baseline_degree_spin)

        self.baseline_smoothing_spin = QDoubleSpinBox()
        self.baseline_smoothing_spin.setRange(0.0, 1.0e9)
        self.baseline_smoothing_spin.setDecimals(3)
        self.baseline_smoothing_spin.setSpecialValueText("Auto")  # 0.0 displays as "Auto" -- see below
        self.baseline_smoothing_spin.setValue(0.0)
        self.baseline_smoothing_spin.setToolTip(
            "Spline smoothness (UnivariateSpline's own s parameter) -- "
            "higher follows the baseline's own point-to-point noise "
            "less, lower follows it more closely (0/\"Auto\" picks a "
            "data-driven default from the baseline points' own noise "
            "level, rather than exactly interpolating through them, "
            "which would fit noise instead of the baseline's trend). "
            "Only used when Spline is selected above."
        )
        self.baseline_smoothing_spin.valueChanged.connect(self._on_baseline_correct_options_changed)
        baseline_options_row.addWidget(self.baseline_smoothing_spin)
        form.addRow("Baseline method:", baseline_options_row)
        self._refresh_baseline_options_visibility()

        reset_button = QPushButton("Reset")
        reset_button.clicked.connect(self._on_reset_phase_clicked)
        form.addRow(reset_button)

        layout.addWidget(phase_box)
        layout.addWidget(self._build_reference_box())
        layout.addStretch(1)
        return tab

    def _build_reference_box(self) -> QGroupBox:
        """Spectral reference (SR) correction -- an absolute Hz offset
        from the spectrum's own SFO1 base frequency, applied to the
        direct (F2) axis so a peak at a known position reads
        correctly, without needing to re-import/reprocess. Also
        available as a one-shot from the Spectrum plot's right-click
        menu (:meth:`_on_reference_shift_from_plot`), which is usually
        quicker than coming to this tab.

        This is a persistent *value*, not a one-shot delta: it shows
        (and is siloed per spectrum, like the FID-processing controls)
        the spectrum's current SR, and re-entering the same value is a
        no-op -- re-applying always recomputes from the spectrum's
        original, as-loaded axis (:meth:`_capture_original_axis`)
        rather than compounding onto whatever the axis currently is."""
        box = QGroupBox("Reference (SR)")
        box.setToolTip(
            "Spectral reference: an absolute Hz offset from this "
            "spectrum's own SFO1 base frequency, applied to the "
            "direct-dimension ppm/Hz axis -- e.g. the recorded SR "
            "correction from a calibration standard. Positive moves "
            "peaks to higher frequency/ppm. Persists as a value (not "
            "a one-shot delta) -- editing it recomputes the axis from "
            "scratch, so re-entering the same number twice is a no-op."
        )
        form = QFormLayout(box)

        self.sr_shift_spin = QDoubleSpinBox()
        self.sr_shift_spin.setRange(-1.0e6, 1.0e6)
        self.sr_shift_spin.setDecimals(4)
        self.sr_shift_spin.setSuffix(" Hz")
        form.addRow("SR:", self.sr_shift_spin)

        apply_button = QPushButton("Apply Reference Shift")
        apply_button.clicked.connect(self._on_apply_sr_clicked)
        form.addRow(apply_button)

        return box

    def _make_phase_control(self, sensitivity: float = 0.2, step: float = 1.0) -> tuple[DragHandle, QDoubleSpinBox]:
        handle = DragHandle(sensitivity=sensitivity, step=step)

        spin = RelativeDoubleSpinBox()
        spin.setRange(-_PHASE_SPIN_RANGE, _PHASE_SPIN_RANGE)
        spin.setDecimals(2)
        spin.setSingleStep(step)
        spin.setValue(0.0)
        # Qt's own sizeHint for a QDoubleSpinBox doesn't account for
        # -_PHASE_SPIN_RANGE being unbounded -- default width is too
        # narrow to show even a 3-4 digit value (e.g. "-1234.56")
        # without scrolling the text inside the box.
        spin.setMinimumWidth(80)

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

    def _on_pivot_spin_changed(self, value: float) -> None:
        """The pivot spin box changed (typed, or the slider moving it
        -- see :meth:`_on_pivot_slider_changed`, which sets it with
        signals blocked so this doesn't fire back). Keeps the slider's
        position in sync and previews the result -- deliberately
        does *not* call :meth:`_capture_phase_reference`, which would
        zero Phase 0/Phase 1 -- moving the pivot only changes where
        Phase 1 is measured from, not the correction itself (see
        :meth:`_compensate_phase0_for_pivot_change` for why Phase 0
        still needs adjusting, even though nothing here resets it)."""
        self._compensate_phase0_for_pivot_change(value)
        self._pivot_ppm_reference = value
        self._sync_pivot_slider_from_spin(value)
        self._apply_phase_preview()

    def _compensate_phase0_for_pivot_change(self, new_pivot_ppm: float) -> None:
        """``apply_phase`` computes ``phase0 + phase1 * (frac(x) -
        pivot_frac)`` -- so with Phase 1 nonzero, moving the pivot
        while leaving Phase 0/Phase 1's *numbers* unchanged still
        changes the correction actually applied at every point by a
        constant ``phase1 * (old_pivot_frac - new_pivot_frac)``, i.e.
        the whole spectrum visibly jumps even though neither spin box
        moved -- easy to mistake for "phasing reset". Folding that
        same offset into Phase 0 keeps the displayed correction
        continuous: only where *future* Phase 1 drags rotate around
        actually changes, matching what moving a pivot is supposed to
        mean."""
        if self._phase_freq_ppm is None or self._pivot_ppm_reference is None:
            return
        phase1 = self.phase1_spin.value()
        if not phase1:
            return
        span = self._phase_freq_ppm.max() - self._phase_freq_ppm.min()
        if not span:
            return
        old_frac = (self._pivot_ppm_reference - self._phase_freq_ppm.min()) / span
        new_frac = (new_pivot_ppm - self._phase_freq_ppm.min()) / span
        delta = phase1 * (new_frac - old_frac)
        if not delta:
            return
        self.phase0_spin.blockSignals(True)
        self.phase0_spin.setValue(self.phase0_spin.value() + delta)
        self.phase0_spin.blockSignals(False)
        self.phase0_handle.setValue(self.phase0_spin.value(), emit=False)

    def _sync_pivot_slider_from_spin(self, value: float) -> None:
        if self._phase_freq_ppm is None:
            return
        freq_ppm = self._phase_freq_ppm
        span = freq_ppm.max() - freq_ppm.min()
        frac = (value - freq_ppm.min()) / span if span else 0.5
        frac = min(max(frac, 0.0), 1.0)
        # The spectrum's own x-axis is plotted inverted (high ppm on
        # the left, per NMR convention -- see show_spectrum), so the
        # slider is inverted here to match: dragging it right should
        # move the pivot marker right on screen, which means *toward*
        # freq_ppm.min() (low value = slider 1000), not away from it.
        self.pivot_slider.blockSignals(True)
        self.pivot_slider.setValue(round((1.0 - frac) * 1000))
        self.pivot_slider.blockSignals(False)

    def _on_pivot_slider_changed(self, value: int) -> None:
        if self._phase_freq_ppm is None:
            return
        freq_ppm = self._phase_freq_ppm
        span = freq_ppm.max() - freq_ppm.min()
        ppm = float(freq_ppm.max() - (value / 1000.0) * span)  # inverted -- see _sync_pivot_slider_from_spin
        self.pivot_spin.blockSignals(True)
        self.pivot_spin.setValue(ppm)
        self.pivot_spin.blockSignals(False)
        # Bypasses _on_pivot_spin_changed (pivot_spin's own signals are
        # blocked above, to avoid re-entering this handler via the two
        # controls' mutual sync) -- so the same Phase 0 compensation it
        # would have done needs doing here too, or dragging the slider
        # specifically would still show the "phasing reset" jump.
        self._compensate_phase0_for_pivot_change(ppm)
        self._pivot_ppm_reference = ppm
        self._apply_phase_preview()

    def _on_phase0_nudge_clicked(self, delta: float) -> None:
        """+/-90/+/-180 degree quick-flip buttons -- nudges Phase 0 by a
        fixed amount *from its current value* (not a reset to that
        absolute value), through the same spin box a manual drag/type
        would change, so it composes naturally with whatever's already
        applied (e.g. two +90 clicks equal one +180) and goes through
        the same live-preview path as every other Phase 0 change."""
        self.phase0_spin.setValue(self.phase0_spin.value() + delta)

    def _on_reset_phase_clicked(self) -> None:
        for handle in (self.phase0_handle, self.phase1_handle):
            handle.setValue(0.0, emit=False)
        for spin in (self.phase0_spin, self.phase1_spin):
            spin.blockSignals(True)
            spin.setValue(0.0)
            spin.blockSignals(False)
        self._apply_phase_preview()

    def _on_autophase_clicked(self) -> None:
        """Autophase for whatever the Phase controls currently target
        -- unlike :meth:`_on_autophase_slices_clicked` (pseudo-2D
        only, each slice gets its own independent phase0 immediately
        applied), this always writes into the shared Phase 0/Phase 1
        spin boxes, same as dragging them by hand, so it works for a
        plain 1D spectrum too (which otherwise had no autophase option
        at all).

        Zero-order only (:func:`processing.ft.apc0`) by default,
        resetting Phase 1 to 0 (a stale nonzero Phase 1 from an
        earlier manual drag would otherwise silently combine with the
        freshly computed Phase 0 in a way apc0 never accounted for).
        With the "1st-order" checkbox on, uses
        :func:`processing.ft.autophase01`'s joint search instead,
        which only actually changes Phase 1 away from 0 when doing so
        demonstrably improves the lineshape.

        Snapshots the pre-autophase Phase 0/Phase 1 first, so a bad
        result can be undone via :meth:`_on_revert_autophase_clicked`."""
        if self._phase_mode is None or self._phase_reference_full is None:
            return

        freq_ppm = self._phase_freq_ppm
        if self._phase_mode == "pseudo2d" and not self.apply_all_slices_checkbox.isChecked():
            target = self._phase_reference_full[self.slice_spin.value() - 1, :]
        else:
            target = self._phase_reference_full

        span = freq_ppm.max() - freq_ppm.min()
        pivot_frac = (self.pivot_spin.value() - freq_ppm.min()) / span if span else 0.5
        if self.autophase_joint_checkbox.isChecked():
            phase0, phase1 = processing_ft.autophase01(target, freq_ppm, pivot_frac=pivot_frac)
        else:
            phase0 = processing_ft.apc0(target, freq_ppm)
            phase1 = 0.0

        self._last_autophase_revert = (self.phase0_spin.value(), self.phase1_spin.value())
        self.revert_autophase_button.setEnabled(True)

        self.phase1_handle.setValue(phase1, emit=False)
        self.phase1_spin.blockSignals(True)
        self.phase1_spin.setValue(phase1)
        self.phase1_spin.blockSignals(False)

        self.phase0_handle.setValue(phase0, emit=False)
        self.phase0_spin.blockSignals(True)
        self.phase0_spin.setValue(phase0)
        self.phase0_spin.blockSignals(False)

        self._apply_phase_preview()
        self.canvas.autoscale_y()
        # A discrete, one-shot action (not a per-tick drag) -- safe to
        # refresh the Processing-preview tab's own (potentially slow,
        # Session.preview()-based) content here, unlike a live phase
        # drag tick (see _apply_phase_preview's pseudo2d branch).
        self._refresh_processing_preview_if_active()

    def _on_revert_autophase_clicked(self) -> None:
        """Undo the last :meth:`_on_autophase_clicked`, restoring
        Phase 0/Phase 1 to their pre-autophase values."""
        if self._last_autophase_revert is None:
            return
        phase0, phase1 = self._last_autophase_revert

        self.phase0_handle.setValue(phase0, emit=False)
        self.phase0_spin.blockSignals(True)
        self.phase0_spin.setValue(phase0)
        self.phase0_spin.blockSignals(False)

        self.phase1_handle.setValue(phase1, emit=False)
        self.phase1_spin.blockSignals(True)
        self.phase1_spin.setValue(phase1)
        self.phase1_spin.blockSignals(False)

        self._apply_phase_preview()
        self.canvas.autoscale_y()
        self._refresh_processing_preview_if_active()
        self._last_autophase_revert = None
        self.revert_autophase_button.setEnabled(False)

    @staticmethod
    def _baseline_correct_if_enabled(data, freq_ppm, spectrum):
        """``data`` unchanged if ``spectrum.metadata.raw["baseline_
        correct"]`` isn't set, else baseline-corrected per its own
        recorded method/smoothing (see :meth:`_on_baseline_correct_
        toggled`) -- metadata.raw is the single source of truth for
        this state (not the live GUI widgets directly), matching
        :meth:`_compute_processing_preview`'s own read of it, so every
        phase-mode branch of :meth:`_apply_phase_preview` agrees with
        the Processing tab's preview on whether/how baseline correction
        is currently active.

        ``data`` unchanged (not raised) on a genuine baseline-fit
        failure too -- e.g. too few points for the estimator, which a
        live recompute can transiently hit on a very short FID/zero-
        fill combination -- best-effort, same as every other live
        preview step in this class; the user sees the uncorrected
        result rather than the whole live-update pipeline raising out
        of a signal handler over one edge case."""
        if not spectrum.metadata.raw.get("baseline_correct"):
            return data
        try:
            return processing_ft.baseline_correct(
                freq_ppm, data,
                method=spectrum.metadata.raw.get("baseline_method", "linear"),
                smoothing=spectrum.metadata.raw.get("baseline_smoothing"),
                degree=spectrum.metadata.raw.get("baseline_degree", 2),
            )
        except Exception:  # noqa: BLE001 -- best-effort live correction, see docstring above
            return data

    def _refresh_baseline_options_visibility(self) -> None:
        method = self.baseline_method_combo.currentData()
        self.baseline_smoothing_spin.setVisible(method == "spline")
        self.baseline_degree_spin.setVisible(method == "polynomial")

    def _on_baseline_correct_options_changed(self, *_args) -> None:
        """Method/smoothing changed -- re-anchor to whatever's
        currently displayed as this spectrum's own "before editing"
        state doesn't change (only how a checked baseline toggle would
        compute), then just re-show the current (already-corrected-or-
        not) state with the new choice recorded, same idea as
        :meth:`_on_baseline_correct_toggled`."""
        self._refresh_baseline_options_visibility()
        index = self._selected_index()
        if index is None:
            return
        spectrum = self.session.spectra[index]
        spectrum.metadata.raw["baseline_method"] = self.baseline_method_combo.currentData()
        smoothing = self.baseline_smoothing_spin.value()
        spectrum.metadata.raw["baseline_smoothing"] = smoothing if smoothing > 0.0 else None
        spectrum.metadata.raw["baseline_degree"] = self.baseline_degree_spin.value()
        self._schedule_sidecar_save(index)
        if self.baseline_correct_button.isChecked():
            self._apply_phase_preview()

    def _on_baseline_correct_toggled(self, checked: bool) -> None:
        """Subtract (or restore, if unchecked) a robust baseline
        (:func:`processing.ft.baseline_correct`, sigma-clipped so real
        peaks don't bias the fit; linear or spline, see the Baseline
        method controls) from whatever the Phase controls currently
        target -- same targeting rules as :meth:`_on_autophase_clicked`
        (including "Apply to every slice" for a pseudo-2D series, each
        slice fit independently). A toggle, not a one-shot action --
        explicit user feedback ("baseline correction should be applied
        to the processed spectrum but the correction should be able to
        be switched on and off") -- recomputed via
        :meth:`_apply_phase_preview` (same "1d" live-recompute pipeline
        phase dragging uses, for that mode; pseudo2d/2d re-derive from
        the same reference snapshot phase itself uses, with baseline
        applied on top when checked)."""
        index = self._selected_index()
        if index is None:
            return
        spectrum = self.session.spectra[index]
        spectrum.metadata.raw["baseline_correct"] = checked
        self._schedule_sidecar_save(index)
        self._apply_phase_preview()

    def _on_phase_target_changed(self) -> None:
        """The pseudo-2D slice or the 2D F1/F2 dimension picked in the
        Dimension box changed -- what the phase controls act on is now
        a different view, so re-anchor them to it (see
        ``_capture_phase_reference``) and redraw to show that view at
        its own current (un-previewed) phase."""
        self._capture_phase_reference()
        self._apply_phase_preview()

    def _on_slice_spin_changed(self, _value: int) -> None:
        """``slice_spin`` doubles as the Processing tab's own preview
        slice selector (see its tooltip) -- refresh that preview too,
        not just the phase controls' own target."""
        self._on_phase_target_changed()
        self._refresh_processing_preview_if_active()

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

    def _on_autophase_slices_clicked(self) -> None:
        """Zero-order autophase every slice of the selected pseudo-2D
        spectrum independently (:func:`opennmr.processing.ft.apc0` per
        row) -- unlike the shared phase0/phase1 controls above (even
        with "Apply to every slice" checked, which broadcasts
        one value to every row), each slice's own phase error is found
        and corrected on its own."""
        index = self._selected_index()
        if index is None:
            return
        spectrum = self.session.spectra[index]
        if not (spectrum.ndim == 2 and spectrum.is_pseudo_2d and spectrum.data_imag is not None):
            return

        freq_ppm = spectrum.axis_values(-1, "ppm")
        complex_data = spectrum.complex_data
        phased = np.empty_like(complex_data)
        for row in range(complex_data.shape[0]):
            # A coarser grid than apc0's own 0.1 deg default -- run
            # once per slice, so this stays quick across a series with
            # many slices rather than needing a progress indicator.
            phase0 = processing_ft.apc0(complex_data[row], freq_ppm, phase0_inc=0.5)
            phased[row] = processing_ft.apply_phase(complex_data[row], freq_ppm, phase0, 0.0)

        spectrum.data = np.real(phased)
        spectrum.data_imag = np.imag(phased)
        # The underlying data has genuinely changed -- re-anchor the
        # Phase controls to it (same as a fresh selection), rather than
        # leaving them pointing at the pre-autophase snapshot.
        self._capture_phase_reference()
        on_processing_preview = self.canvas_tabs.currentWidget() is self.processing_preview_canvas
        self._show_pseudo2d_slice(spectrum, reset_view=False, switch_tab=not on_processing_preview)
        self.canvas.set_pivot_marker(self.pivot_spin.value())
        if on_processing_preview:
            self._refresh_processing_preview_if_active()

    def _commit_processing_preview_as_spectrum(self, spectrum) -> bool:
        """Compute :meth:`_compute_processing_preview` and write the
        result straight into ``spectrum.data``/``data_imag``/``axes``,
        then refresh ``self.canvas`` in place -- the same source, not
        a second computation of it (factored out of
        :meth:`_apply_phase_preview`'s own "1d" branch so
        :meth:`_on_fid_processing_control_changed` can share it: a
        window/zero-fill/etc. change is just as much "the live preview
        *is* the spectrum now" as a phase change is). Returns
        ``False`` (spectrum left untouched) when there's no raw FID to
        compute this way from at all -- the caller decides what to do
        then (fall back to a narrower update, or nothing)."""
        computed = self._compute_processing_preview(spectrum) if self._fid_reprocessable(spectrum) else None
        if computed is None:
            return False
        fid, dt, preview_spectrum, _title = computed
        spectrum.data = preview_spectrum.data
        spectrum.data_imag = preview_spectrum.data_imag
        spectrum.axes = preview_spectrum.axes
        # Keep the FID preview tab's own source (metadata.raw["fid"])
        # in step too -- otherwise it would go stale the moment any
        # control changed, showing the FID from whatever was last
        # loaded/reset rather than what's actually now on screen.
        spectrum.metadata.raw["fid"] = fid
        spectrum.metadata.raw["fid_dt"] = dt
        self.canvas.update_data(spectrum)
        return True

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
            # Same source as the Processing tab's own live preview
            # (_compute_processing_preview), not a second, independent
            # phase-only computation against a snapshot taken back at
            # _capture_phase_reference time -- that snapshot predates
            # whatever window/zero-fill/etc. choice is *currently*
            # showing on the Processing tab but not yet "Applied",
            # so a phase drag used to silently re-diverge the two tabs
            # (explicit user report: could even differ in *point count*,
            # not just value, once zero-fill was live-but-unapplied).
            # Falls back to phasing the as-loaded snapshot directly only
            # when there's no raw FID to recompute from at all (a
            # processed-only fallback load -- _compute_processing_preview
            # returns None for exactly that case).
            if not self._commit_processing_preview_as_spectrum(spectrum):
                phased = processing_ft.apply_phase(self._phase_reference_full, freq_ppm, phase0, phase1, pivot_frac=pivot_frac)
                phased = self._baseline_correct_if_enabled(phased, freq_ppm, spectrum)
                spectrum.data = np.real(phased)
                spectrum.data_imag = np.imag(phased)
                # Kept up to date regardless of which canvas tab is
                # actually showing, so switching back to Spectrum later
                # (from the Processing-preview tab, see below) never
                # shows stale data -- only the tab *switch* itself is
                # guarded.
                self.canvas.update_data(spectrum)
            self.canvas.set_pivot_marker(self.pivot_spin.value())
            if self.canvas_tabs.currentWidget() is not self.processing_preview_canvas:
                self.canvas_tabs.setCurrentWidget(self.canvas)
            else:
                # In place, preserving whatever zoom is already on the
                # preview's spectrum panel -- see
                # _refresh_processing_preview_phase_only's own
                # docstring (explicit user request: phasing while
                # zoomed into a small peak shouldn't reset the view on
                # every tick).
                self._refresh_processing_preview_phase_only()

        elif self._phase_mode == "pseudo2d":
            if self.apply_all_slices_checkbox.isChecked():
                phased = processing_ft.apply_phase(
                    self._phase_reference_full, freq_ppm, phase0, phase1, pivot_frac=pivot_frac,
                )
                # Baseline correction is deliberately NOT applied here,
                # even when enabled -- a per-row spline fit across a
                # real multi-hundred-row dataset is genuinely slow
                # (~1-2ms/row, but that's seconds across a whole
                # dataset), and this runs on *every* drag tick with no
                # debouncing of its own. Recomputing it here on every
                # tick is exactly the "hangs" a rapid drag could
                # already cause -- deferred to a debounced full
                # recompute instead (_schedule_multi_row_reprocess),
                # which also picks it up from the FID-processing
                # controls' own live path now that there's no separate
                # "Apply Processing" step. The display briefly shows
                # phase-only during an active drag and catches up with
                # baseline shortly after it settles.
                if spectrum.metadata.raw.get("baseline_correct"):
                    self._schedule_multi_row_reprocess(index)
            else:
                row = self.slice_spin.value() - 1
                phased = self._phase_reference_full.copy()
                phased[row, :] = self._baseline_correct_if_enabled(
                    processing_ft.apply_phase(
                        self._phase_reference_full[row, :], freq_ppm, phase0, phase1, pivot_frac=pivot_frac,
                    ),
                    freq_ppm, spectrum,
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
            elif self.canvas_tabs.currentWidget() is self.processing_preview_canvas:
                # Keep the Spectrum tab's own content correct in the
                # background (switch_tab=False, reset_view=False -- so
                # switching back to it later shows the current phase,
                # with whatever zoom was already there preserved),
                # without stealing focus from the Processing tab's own
                # preview -- which itself also refreshes in place now
                # (see _refresh_processing_preview_phase_only), for the
                # same "don't reset a deliberate zoom on every drag
                # tick" reason.
                self._show_pseudo2d_slice(spectrum, reset_view=False, switch_tab=False)
                self._refresh_processing_preview_phase_only()
            else:
                self._show_pseudo2d_slice(spectrum, reset_view=False)
                self.canvas.set_pivot_marker(self.pivot_spin.value())

        elif self._phase_mode == "2d":
            dim = "f2" if self.dim_combo.currentIndex() == 0 else "f1"
            data_for_dim = self._phase_reference_full if dim == "f2" else self._phase_reference_full.T
            phased = processing_ft.apply_phase(data_for_dim, freq_ppm, phase0, phase1, pivot_frac=pivot_frac)
            if dim == "f1":
                phased = phased.T
            spectrum.data = np.real(phased)
            spectrum.data_imag = np.imag(phased)
            # Baseline correction deferred to a debounced recompute,
            # not applied inline on every drag tick -- see the
            # matching pseudo2d "apply to every slice" branch above
            # for why (the actual "hangs" mechanism: a per-row spline
            # fit re-run on every single mouse-move event during a
            # phase drag).
            if spectrum.metadata.raw.get("baseline_correct"):
                self._schedule_multi_row_reprocess(index)
            # Best-effort, same reasoning as _baseline_correct_if_enabled:
            # this runs on *every* drag tick, so a rendering exception
            # here must never raise out of the signal handler (freezing
            # the drag) or -- worse, since a dialog here would otherwise
            # pop once per mouse-move tick -- show a QMessageBox the way
            # _draw_selected_spectrum's own one-shot render failure does.
            # Skip this tick's redraw and let the next one (once
            # whatever's wrong resolves, e.g. the drag settles onto a
            # value that renders fine) catch up, rather than leaving the
            # whole live-phase-drag UX unusable over one bad frame. Found
            # via a real crash: a 2D contour render failure (a genuine
            # axis-shape bug, since fixed) reached this exact call site
            # unguarded and printed an unhandled traceback once per tick
            # for the whole duration of the drag.
            try:
                self.canvas.show_spectrum(spectrum)
                # A conventional 2D's Processing-preview tab mirrors this
                # same spectrum object directly (see _show_processing_preview's
                # own "same source, not a second computation" reasoning) --
                # keep it live during a drag too, rather than only catching
                # up the next time something else refreshes it.
                if self.canvas_tabs.currentWidget() is self.processing_preview_canvas:
                    self.processing_preview_canvas.show_spectrum(spectrum)
                else:
                    self.canvas_tabs.setCurrentWidget(self.canvas)
            except Exception:  # noqa: BLE001 -- see comment above
                pass

    def _show_pseudo2d_slice(self, spectrum, reset_view: bool, switch_tab: bool = True) -> None:
        """Preview a pseudo-2D spectrum as the single slice currently
        selected in the Dimension box -- a full contour redraw on every
        drag tick would be far too slow, and a line plot is what you
        actually need to see to judge a phase correction anyway.
        ``switch_tab=False`` (see :meth:`_show_fid_preview`'s own
        matching param) keeps the Spectrum tab's own content correct
        without stealing focus from whichever canvas tab is actually
        showing right now -- e.g. the Processing-preview tab, while
        that's the active context for phasing."""
        slice_index = self.slice_spin.value()
        trace = get_trace(spectrum, axis="f1", index=slice_index)
        if reset_view or not self.canvas.ax.lines:
            # No existing Line2D for update_data to refresh in place --
            # most commonly, phasing was just started from a pseudo-2D's
            # default contour view. X (ppm) means the same thing on a
            # contour and a slice line view (unlike y -- slice number
            # vs. intensity, genuinely different), so preserve whatever
            # ppm zoom was already there across the otherwise-
            # unavoidable rebuild, rather than silently resetting it
            # the moment the user's very first phase tick lands.
            preserve_xlim = not reset_view and self.canvas._contour_spectrum is not None
            xlim = self.canvas.ax.get_xlim() if preserve_xlim else None
            self.canvas.show_spectrum(trace)
            if xlim is not None:
                self.canvas.ax.set_xlim(xlim)
        else:
            self.canvas.update_data(trace)
        self.canvas.ax.set_title(self._slice_label(spectrum, slice_index))
        self.canvas.canvas.draw_idle()
        if switch_tab:
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

    def _commit_pending_phase_to_metadata(self) -> None:
        """Fold the phase delta currently showing in the Phase 0/Phase 1
        spin boxes into ``metadata.raw["phase0"/"phase1"]`` for whatever
        spectrum/mode this editing session was targeting
        (:attr:`_phase_target_index`/:attr:`_phase_mode`, both left over
        from the *previous* :meth:`_capture_phase_reference` call --
        this method always runs before that one re-anchors).

        Every live drag already writes the actual phased data straight
        into ``spectrum.data``/``data_imag`` (see
        :meth:`_apply_phase_preview`) -- that part was never the
        problem. But nothing ever updated ``metadata.raw`` to match,
        so anything that recomputes a phase from ``metadata.raw`` +
        the (about-to-be-zeroed) spin boxes instead of reading
        ``spectrum.data`` directly -- namely
        :meth:`_compute_processing_preview`, which rebuilds from the
        raw FID -- would show the *un*-phased result again the moment
        you switched slice/tab and came back, even though the actual
        spectrum was correctly phased the whole time.

        Skipped for one case ``metadata.raw`` can't actually represent
        (matches :meth:`_current_phase_kwargs`'s own gating): a
        pseudo-2D slice's *own* phase with "Apply to every slice"
        unchecked (metadata.raw describes the whole dataset, not one
        row -- :meth:`_reapply_phase_post_reload` already handles that
        case's own persistence-across-reprocess need directly). A
        conventional 2D's F1/indirect phase folds into its own
        ``"phase0_f1"/"phase1_f1"`` fields instead of ``"phase0"/
        "phase1"`` (which mean F2 specifically) -- explicit user
        feedback that F1 phasing was otherwise "impossible": every
        drag *did* write correctly into ``spectrum.data`` (never the
        problem, same as every other mode here), but with nothing
        durable ever recorded, any reload (Apply Processing, Align
        Echo, Process Indirect Dimension, ...) that landed after the
        spin boxes had already been re-zeroed by a later
        :meth:`_capture_phase_reference` call -- reselecting, changing
        slice/dimension -- silently reset F1 back to unphased, with no
        way to tell it had ever been touched."""
        if self._phase_mode is None or self._phase_target_index is None:
            return
        if self._phase_mode == "pseudo2d" and not self.apply_all_slices_checkbox.isChecked():
            return
        phase0_delta = self.phase0_spin.value()
        phase1_delta = self.phase1_spin.value()
        if not phase0_delta and not phase1_delta:
            return  # nothing pending -- avoid a needless metadata write
        index = self._phase_target_index
        if not (0 <= index < len(self.session)):
            return
        spectrum = self.session.spectra[index]
        is_f1 = self._phase_mode == "2d" and self.dim_combo.currentIndex() != 0
        key0, key1 = ("phase0_f1", "phase1_f1") if is_f1 else ("phase0", "phase1")
        spectrum.metadata.raw[key0] = float(spectrum.metadata.raw.get(key0, 0.0)) + phase0_delta
        spectrum.metadata.raw[key1] = float(spectrum.metadata.raw.get(key1, 0.0)) + phase1_delta
        self._schedule_sidecar_save(index)

    def _capture_phase_reference(self) -> None:
        """(Re)anchor the phase controls to whatever they're currently
        meant to act on -- a 1D spectrum, one pseudo-2D slice (or all of
        them), or a whole 2D dataset along F1/F2 -- and zero them, so
        dragging back to 0 always returns to how *that view* looked when
        you started editing it. Called on every spectrum selection and
        every slice/dimension change; never called mid-drag.

        Commits whatever phase is currently showing (baked-in +
        live delta) into the *previous* target's own metadata.raw
        first (see :meth:`_commit_pending_phase_to_metadata`) -- a live
        drag already writes the correction directly into
        ``spectrum.data``, but never into metadata.raw itself, so
        anything that recomputes from the raw FID + metadata instead
        of the live spectrum data (the Processing tab's own preview)
        would otherwise show a stale, seemingly "reset" phase the next
        time it's shown, even though the actual spectrum was never
        touched."""
        self._commit_pending_phase_to_metadata()
        for handle in (self.phase0_handle, self.phase1_handle):
            handle.setValue(0.0, emit=False)
        for spin in (self.phase0_spin, self.phase1_spin):
            spin.blockSignals(True)
            spin.setValue(0.0)
            spin.blockSignals(False)
        # A revert snapshot only makes sense against the view it was
        # taken from -- re-anchoring to a new spectrum/slice/dimension
        # invalidates it rather than leaving a stale "Revert" that
        # would restore the WRONG view's phase.
        self._last_autophase_revert = None
        self.revert_autophase_button.setEnabled(False)

        index = self._selected_index()
        self._phase_target_index = index
        if index is None:
            self._phase_mode = None
            self._phase_reference_full = None
            self._phase_freq_ppm = None
            return

        spectrum = self.session.spectra[index]

        # Baseline-correction state (on/off, method, smoothing) is
        # tracked per spectrum in metadata.raw -- same "each experiment
        # object should have its own metadata locked in for when it's
        # called upon" reasoning as phase0/phase1, restored here rather
        # than always resetting to "off" on every reselection.
        self.baseline_correct_button.blockSignals(True)
        self.baseline_correct_button.setChecked(bool(spectrum.metadata.raw.get("baseline_correct", False)))
        self.baseline_correct_button.blockSignals(False)
        self.baseline_method_combo.blockSignals(True)
        method_index = self.baseline_method_combo.findData(spectrum.metadata.raw.get("baseline_method", "linear"))
        self.baseline_method_combo.setCurrentIndex(max(0, method_index))
        self.baseline_method_combo.blockSignals(False)
        self.baseline_smoothing_spin.blockSignals(True)
        self.baseline_smoothing_spin.setValue(spectrum.metadata.raw.get("baseline_smoothing") or 0.0)
        self.baseline_smoothing_spin.blockSignals(False)
        self.baseline_degree_spin.blockSignals(True)
        self.baseline_degree_spin.setValue(spectrum.metadata.raw.get("baseline_degree", 2))
        self.baseline_degree_spin.blockSignals(False)
        self._refresh_baseline_options_visibility()

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

        centre_ppm = float((self._phase_freq_ppm.min() + self._phase_freq_ppm.max()) / 2)
        self.pivot_spin.blockSignals(True)
        self.pivot_spin.setValue(centre_ppm)
        self.pivot_spin.blockSignals(False)
        self.pivot_slider.blockSignals(True)
        self.pivot_slider.setValue(500)  # centre, matching the ppm value just set above
        self.pivot_slider.blockSignals(False)
        self._pivot_ppm_reference = centre_ppm

    def _set_processing_enabled(self, enabled: bool, status: str | None = None) -> None:
        for widget in (
            self.phase0_handle, self.phase0_spin, self.phase1_handle, self.phase1_spin,
            self.pivot_spin, self.pivot_slider, self.autophase_joint_checkbox,
            self.baseline_correct_button, self.baseline_method_combo, self.baseline_smoothing_spin,
        ):
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

    _PEAK_TABLE_COLUMNS = ("Shift (ppm)", "FWHM (ppm)", "Eta", "Fixed")
    _REGION_TABLE_COLUMNS = ("Low (ppm)", "High (ppm)")

    def _build_analysis_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        self.analysis_status_label = QLabel(
            "Fit Spectrum works on any 1D spectrum, or every slice of a "
            "pseudo-2D series. Fit Curve (relaxation/diffusion) needs a "
            "pseudo-2D series selected here (or right-click it in the "
            "Data tab), or a stack of 1D spectra in the Compare tab."
        )
        self.analysis_status_label.setWordWrap(True)
        layout.addWidget(self.analysis_status_label)

        # Sub-tabs -- Peak Picking (peak-shape fit against one
        # spectrum/slice), Integration (peak-fitting-free alternative
        # producing the same per-slice value), and Curve Fitting
        # (relaxation/diffusion fit against a peak's per-slice series)
        # -- were previously one long stacked panel (Peak Picking and
        # Integration were a single "Spectral Fitting" tab); split
        # further per explicit feedback that combined tab held too
        # much information at once for a casual user, leaving room for
        # future analysis modes as further sub-tabs without any one
        # panel growing without bound.
        self.analysis_sub_tabs = QTabWidget()
        self.analysis_sub_tabs.addTab(self._build_peak_picking_subtab(), "Peak Picking")
        self.analysis_sub_tabs.addTab(self._build_integration_subtab(), "Integration")
        self.analysis_sub_tabs.addTab(self._build_curve_fitting_subtab(), "Curve Fitting")
        self.analysis_sub_tabs.addTab(self._build_vocs_subtab(), "VOCS")
        self.analysis_sub_tabs.addTab(self._build_imaging_subtab(), "Imaging")
        layout.addWidget(self.analysis_sub_tabs)

        return tab

    def _build_peak_picking_subtab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        peak_box = QGroupBox("Peaks (pseudo-Voigt)")
        peak_layout = QVBoxLayout(peak_box)

        self.peak_table = QTableWidget(0, len(self._PEAK_TABLE_COLUMNS))
        self.peak_table.setHorizontalHeaderLabels(self._PEAK_TABLE_COLUMNS)
        self.peak_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        # Qt's own default QTableWidget size hint reserves generous
        # space regardless of actual row count -- capped so an empty
        # or few-peak table (the common case) doesn't spend more
        # vertical space than it needs, on a side panel where that
        # space is already tight (explicit user feedback: "the left
        # side tab is needing scrolling too often"). Comfortably shows
        # ~4 rows before it scrolls internally -- a typical peak count
        # for one spectrum -- rather than a fixed height chosen to
        # match some larger number that's rarely actually used.
        self.peak_table.setMaximumHeight(160)
        self.peak_table.setToolTip(
            "One row per peak to fit -- shift and FWHM are initial "
            "guesses (both, and the amplitude, are refined by the fit); "
            "eta is the Gaussian/Lorentzian mixing fraction (0 = pure "
            "Gaussian, 1 = pure Lorentzian), also refined. Check Fixed "
            "to lock a peak entirely at its current row values (fit "
            "only the others); right-click for Set Custom Bounds…"
        )
        self.peak_table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.peak_table.customContextMenuRequested.connect(self._on_peak_table_context_menu)
        # Same "table and canvas must always agree" reasoning as
        # region_table's own itemChanged connection just below --
        # editing a shift directly in the table (not just adding/
        # removing/auto-picking) should move its own triangle marker
        # too.
        self.peak_table.itemChanged.connect(lambda _item: self._refresh_peak_markers())
        peak_layout.addWidget(self.peak_table)

        peak_button_row = QHBoxLayout()
        add_peak_button = QPushButton("Add Peak")
        add_peak_button.clicked.connect(lambda: self._add_peak_row())
        remove_peak_button = QPushButton("Remove Peak")
        remove_peak_button.clicked.connect(self._on_remove_peak_clicked)
        clear_peaks_button = QPushButton("Clear All")
        clear_peaks_button.setToolTip("Remove every row from the peak table.")
        clear_peaks_button.clicked.connect(self._on_clear_peaks_clicked)
        peak_button_row.addWidget(add_peak_button)
        peak_button_row.addWidget(remove_peak_button)
        peak_button_row.addWidget(clear_peaks_button)
        peak_layout.addLayout(peak_button_row)

        auto_pick_row = QHBoxLayout()
        self.pick_peak_button = QPushButton("Pick from Plot")
        self.pick_peak_button.setCheckable(True)
        self.pick_peak_button.setToolTip(
            "While active, click a peak's position directly on the "
            "plot to add a row here at that shift, instead of typing "
            "it in."
        )
        self.pick_peak_button.toggled.connect(self._on_pick_peak_toggled)
        auto_pick_button = QPushButton("Auto-Pick Peaks")
        auto_pick_button.setToolTip(
            "Detect peaks automatically (scipy find_peaks on the "
            "current selection -- slice 1, for a pseudo-2D series) and "
            "replace the table above with them, as a starting point -- "
            "still just initial guesses, refined by the fit like any "
            "other row here. Sensitivity set by Min. prominence, below."
        )
        auto_pick_button.clicked.connect(self._on_auto_pick_peaks_clicked)
        auto_pick_row.addWidget(self.pick_peak_button)
        auto_pick_row.addWidget(auto_pick_button)
        peak_layout.addLayout(auto_pick_row)

        prominence_row = QHBoxLayout()
        prominence_row.addWidget(QLabel("Min. prominence:"))
        self.auto_pick_prominence_spin = QDoubleSpinBox()
        self.auto_pick_prominence_spin.setRange(0.01, 0.95)
        self.auto_pick_prominence_spin.setDecimals(2)
        self.auto_pick_prominence_spin.setSingleStep(0.05)
        self.auto_pick_prominence_spin.setValue(0.1)
        self.auto_pick_prominence_spin.setToolTip(
            "Minimum peak prominence for Auto-Pick Peaks, as a fraction "
            "of the spectrum's own peak-to-peak range (0-1). Raise this "
            "if auto-pick finds too many peaks (noise bumps, shoulders); "
            "lower it to find weaker ones."
        )
        prominence_row.addWidget(self.auto_pick_prominence_spin)
        peak_layout.addLayout(prominence_row)

        # Position tolerance / S-N gating are fit-quality tuning knobs,
        # not part of picking a peak itself -- a casual user's first
        # fit never needs them (both already have sensible defaults).
        # Collapsed by default so they don't add to what a first-time
        # user has to read past to find Fit Spectrum, below; still
        # exactly the same widgets/attributes, just tucked behind a
        # disclosure toggle rather than removed or renamed.
        advanced_content = QWidget()
        advanced_layout = QVBoxLayout(advanced_content)
        advanced_layout.setContentsMargins(0, 4, 0, 0)

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
        advanced_layout.addLayout(tolerance_row)

        noise_row = QHBoxLayout()
        noise_row.addWidget(QLabel("Noise region (ppm):"))
        self.sn_noise_region_edit = QLineEdit()
        self.sn_noise_region_edit.setPlaceholderText("optional, e.g. 8.5, 9.5")
        self.sn_noise_region_edit.setToolTip(
            "A ppm range with no real signal, 'low, high' -- when set, "
            "any peak whose fitted signal-to-noise (against this "
            "region) falls below the S/N threshold has its amplitude "
            "fixed to zero for that slice, instead of the fit chasing "
            "noise once a real peak has decayed away. Leave blank to "
            "disable (every peak fits normally regardless of S/N)."
        )
        noise_row.addWidget(self.sn_noise_region_edit)
        advanced_layout.addLayout(noise_row)

        sn_threshold_row = QHBoxLayout()
        sn_threshold_row.addWidget(QLabel("S/N threshold:"))
        self.sn_threshold_spin = QDoubleSpinBox()
        self.sn_threshold_spin.setRange(0.0, 1000.0)
        self.sn_threshold_spin.setDecimals(1)
        self.sn_threshold_spin.setSingleStep(0.5)
        self.sn_threshold_spin.setValue(3.0)
        self.sn_threshold_spin.setToolTip("Only used when Noise region above is set.")
        sn_threshold_row.addWidget(self.sn_threshold_spin)
        advanced_layout.addLayout(sn_threshold_row)

        advanced_section, self.peak_advanced_toggle = self._collapsible_section(
            "Advanced (fit tolerance and S/N)", advanced_content,
        )
        self.peak_advanced_content = advanced_content  # for tests -- see isHidden's own note in _collapsible_section
        peak_layout.addWidget(advanced_section)

        layout.addWidget(peak_box)

        fit_button = QPushButton("Fit Spectrum")
        fit_button.setToolTip(
            "Step 1: fit peaks against the current selection -- a plain 1D "
            "spectrum directly, or every slice of a pseudo-2D series."
        )
        fit_button.clicked.connect(self._on_fit_spectrum_clicked)
        layout.addWidget(fit_button)

        self.analysis_results_label = QLabel("")
        self.analysis_results_label.setWordWrap(True)
        layout.addWidget(self.analysis_results_label)

        # "Spectrum fit"/"curve fit" (not "slice fit"/"peak fit", the
        # original names) -- explicit user feedback that the two read
        # as confusingly similar, when what they actually browse is
        # different in kind: this one is Fit Spectrum's own per-slice
        # spectral (pseudo-Voigt) result; the "View curve fit" one
        # below is Fit Curve's per-peak relaxation/diffusion result.
        slice_fit_row = QHBoxLayout()
        slice_fit_row.addWidget(QLabel("Spectrum fit (slice #):"))
        self.view_slice_fit_spin = QSpinBox()
        self.view_slice_fit_spin.setMinimum(1)
        self.view_slice_fit_spin.setEnabled(False)
        self.view_slice_fit_spin.setToolTip(
            "Inspect any individual slice's spectral (pseudo-Voigt) "
            "fit from Fit Spectrum, in the Spectral Fit tab -- "
            "available once a fit has been run."
        )
        self.view_slice_fit_spin.valueChanged.connect(self._on_view_slice_fit_changed)
        slice_fit_spin_row, self.slice_fit_prev_button, self.slice_fit_next_button = (
            self._spin_with_step_buttons(self.view_slice_fit_spin)
        )
        slice_fit_row.addLayout(slice_fit_spin_row)
        layout.addLayout(slice_fit_row)

        self.spectral_fit_params_label = QLabel("")
        self.spectral_fit_params_label.setWordWrap(True)
        self.spectral_fit_params_label.setToolTip(
            "Fitted parameters (with 1-sigma uncertainties) for the "
            "slice shown above, one line per peak."
        )
        layout.addWidget(self.spectral_fit_params_label)

        layout.addStretch(1)
        return tab

    def _build_integration_subtab(self) -> QWidget:
        """Region integration -- the peak-fitting-free alternative to
        Peak Picking's own Fit Spectrum for producing a per-slice value
        Fit Curve can use. Split out into its own sub-tab (previously
        stacked directly under Peak Picking's own peak table/tuning
        controls) since the two are alternative "step 1"s, not part of
        the same flow -- explicit user feedback that the combined tab
        held too much at once for a casual user to take in."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.addWidget(self._build_region_box())

        self.integration_results_label = QLabel("")
        self.integration_results_label.setWordWrap(True)
        layout.addWidget(self.integration_results_label)

        layout.addStretch(1)
        return tab

    def _build_curve_fitting_subtab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        relax_box = QGroupBox("Fit Curve settings (relaxation / diffusion)")
        relax_box.setToolTip(
            "Configures the Fit Curve button below, and the Data "
            "tab's right-click 'Fit Curve' action on a pseudo-2D row, "
            "and the Compare tab's Fit Curve (button or right-click) "
            "on a checked stack of 1D spectra -- not used by Fit "
            "Spectrum, which lives in the Spectral Fitting sub-tab."
        )
        relax_form = QFormLayout(relax_box)

        self.fit_type_combo = QComboBox()
        self.fit_type_combo.addItems(["Relaxation (T1/T2)", "Diffusion (PFG)"])
        self._narrow_combo(self.fit_type_combo)
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
        self._narrow_combo(self.relaxation_model_combo)
        self.relaxation_model_combo.currentIndexChanged.connect(self._on_relaxation_model_changed)
        relax_form.addRow("Model:", self.relaxation_model_combo)

        self.custom_equation_row_label = QLabel("Equation:")
        self.custom_equation_edit = QLineEdit()
        self.custom_equation_edit.setPlaceholderText("e.g. A*exp(-x/T1)+C")
        self.custom_equation_edit.setToolTip(
            "A formula in 'x' (the delay/b-value) and any other names "
            "you choose -- every other name becomes a fitted "
            "parameter, in the order it first appears here. Parsed "
            "with sympy; standard functions (exp, log, sin, sqrt, "
            "...) are available by name."
        )
        self.custom_equation_edit.textChanged.connect(self._on_custom_equation_changed)
        relax_form.addRow(self.custom_equation_row_label, self.custom_equation_edit)

        self.custom_params_label = QLabel("")
        self.custom_params_label.setWordWrap(True)
        relax_form.addRow("", self.custom_params_label)

        self.custom_p0_row_label = QLabel("Initial guesses:")
        self.custom_p0_edit = QLineEdit()
        self.custom_p0_edit.setPlaceholderText("optional, e.g. 10, 0.5, 0 -- defaults to 1 each")
        self.custom_p0_edit.setToolTip(
            "Initial guesses for the parameters above, in the same "
            "order, comma-separated. Any left unspecified (including "
            "leaving this whole field blank) default to 1.0."
        )
        relax_form.addRow(self.custom_p0_row_label, self.custom_p0_edit)

        self.value_source_combo = QComboBox()
        self.value_source_combo.addItem("Integrated area", "integral")
        self.value_source_combo.addItem("Peak amplitude", "amplitude")
        self._narrow_combo(self.value_source_combo)
        self.value_source_combo.setToolTip(
            "What each slice's per-peak value is, for the curve fit: "
            "Integrated area (default -- sums over the whole fitted "
            "peak, more robust to a linewidth change across slices) or "
            "Peak amplitude (the fitted peak height alone -- simpler, "
            "but more sensitive to linewidth changes mimicking an "
            "intensity change)."
        )
        relax_form.addRow("Value source:", self.value_source_combo)

        self.delays_row_label = QLabel("Values:")
        self.delay_source_combo = QComboBox()
        self.delay_source_combo.addItems([
            "Auto", "Manual list", "Start + step", "vclist × factor", "Relative gradient (valueslist)",
        ])
        self._narrow_combo(self.delay_source_combo)
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

        # Stacked (one per row), not side by side -- crammed into one
        # row in this narrow side panel, the two boxes were too
        # cramped to read either value properly. Same fixed width so
        # neither looks arbitrarily bigger/smaller than the other.
        self.delay_start_spin = QDoubleSpinBox()
        self.delay_start_spin.setRange(-1.0e12, 1.0e12)
        self.delay_start_spin.setDecimals(6)
        self.delay_start_spin.setFixedWidth(120)
        self.delay_start_spin.setToolTip("Start")
        self.delay_step_spin = QDoubleSpinBox()
        self.delay_step_spin.setRange(-1.0e12, 1.0e12)
        self.delay_step_spin.setDecimals(6)
        self.delay_step_spin.setFixedWidth(120)
        self.delay_step_spin.setValue(0.1)
        self.delay_step_spin.setToolTip("Step")
        self.delay_start_row_label = QLabel("Start:")
        self.delay_step_row_label = QLabel("Step:")
        relax_form.addRow(self.delay_start_row_label, self.delay_start_spin)
        relax_form.addRow(self.delay_step_row_label, self.delay_step_spin)

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

        # Below every setting it actually depends on (model, value
        # source, delay/b-value source) -- it used to sit above this
        # whole box, reading as if it only depended on the model
        # choice right below it, when it really uses everything here.
        fit_curve_button = QPushButton("Fit Curve…")
        fit_curve_button.setToolTip(
            "Step 2: fit a relaxation/diffusion model to each peak's "
            "per-slice amplitude, across every slice of the selected "
            "pseudo-2D series -- reuses Step 1's results if already "
            "computed, or runs it now. For a stack of separately-loaded "
            "1D spectra instead of one pseudo-2D series, use the "
            "Compare tab's own Fit Curve button/context-menu item."
        )
        fit_curve_button.clicked.connect(self._on_fit_curve_button_clicked)
        relax_form.addRow(fit_curve_button)

        self._populate_fit_model_combo()
        self._on_delay_source_changed(0)

        layout.addWidget(relax_box)

        # Below Fit Curve (not above it, as these originally were) --
        # explicit user feedback: these browse Fit Curve's own result,
        # so they read as confusing/backwards sitting above the button
        # that produces what they show, the same "controls should
        # follow, not precede, what they configure/what's fit before
        # them" principle the Fit Curve button's own position above
        # already follows relative to its settings.
        relax_fit_row = QHBoxLayout()
        relax_fit_row.addWidget(QLabel("View curve fit (peak #):"))
        self.view_relaxation_peak_spin = QSpinBox()
        self.view_relaxation_peak_spin.setMinimum(1)
        self.view_relaxation_peak_spin.setEnabled(False)
        self.view_relaxation_peak_spin.setToolTip(
            "Which peak's relaxation/diffusion curve fit (from Fit "
            "Curve) is shown in the Fit tab -- available once a curve "
            "fit has been run."
        )
        self.view_relaxation_peak_spin.valueChanged.connect(self._on_view_relaxation_peak_changed)
        peak_spin_row, self.relaxation_peak_prev_button, self.relaxation_peak_next_button = (
            self._spin_with_step_buttons(self.view_relaxation_peak_spin)
        )
        relax_fit_row.addLayout(peak_spin_row)
        layout.addLayout(relax_fit_row)

        self.relaxation_uncertainty_checkbox = QCheckBox("Show uncertainty band")
        self.relaxation_uncertainty_checkbox.setToolTip(
            "Shade a +/-3-sigma uncertainty envelope around the fit "
            "curve (processing.relaxation.fit_curve_uncertainty, "
            "propagated from the fit's own parameter covariance) -- "
            "matches the shaded-region style of the user's own PFG "
            "diffusion-fitting plots."
        )
        self.relaxation_uncertainty_checkbox.stateChanged.connect(
            lambda _s: self._display_relaxation_peak(self.view_relaxation_peak_spin.value() - 1),
        )
        layout.addWidget(self.relaxation_uncertainty_checkbox)

        self.relaxation_fit_params_label = QLabel("")
        self.relaxation_fit_params_label.setWordWrap(True)
        layout.addWidget(self.relaxation_fit_params_label)

        layout.addStretch(1)
        return tab

    def _build_vocs_subtab(self) -> QWidget:
        """Variable Offset Cumulative Spectroscopy -- corrects a 1D
        spectrum's frequency-dependent excitation-profile roll-off
        from a finite pulse width (``processing.vocs``, fully built
        and tested; this is the first GUI wiring for it). Two
        operations, sharing the pulse-length/window-filter controls
        above them: correct one spectrum in place (as a new derived
        spectrum, since ``apply_vocs`` never mutates its input), or
        correct and sum/average several VOCS acquisitions -- offset
        1D spectra covering different, overlapping excitation windows
        -- from the Compare tab's checked stack into one composite."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        settings_box = QGroupBox("VOCS settings")
        settings_form = QFormLayout(settings_box)

        self.vocs_pulse_edit = QLineEdit("P1")
        self.vocs_pulse_edit.setToolTip(
            "The acquisition parameter (from this spectrum's recorded "
            "acqus table, e.g. Bruker's P1) giving the 90-degree pulse "
            "length in microseconds -- used to compute the RF "
            "nutation frequency the VOCS correction is centred on."
        )
        settings_form.addRow("Pulse param:", self.vocs_pulse_edit)

        self.vocs_window_filter_checkbox = QCheckBox("Limit to VOCS excitation bandwidth")
        self.vocs_window_filter_checkbox.setChecked(True)
        self.vocs_window_filter_checkbox.setToolTip(
            "Zero the correction outside the pulse's useful excitation "
            "bandwidth, instead of amplifying noise far from centre "
            "where the correction factor blows up."
        )
        settings_form.addRow(self.vocs_window_filter_checkbox)

        layout.addWidget(settings_box)

        apply_box = QGroupBox("Apply VOCS")
        apply_layout = QVBoxLayout(apply_box)
        apply_intro = QLabel("Corrects the currently selected spectrum, added as a new spectrum.")
        apply_intro.setWordWrap(True)
        apply_layout.addWidget(apply_intro)
        apply_button = QPushButton("Apply VOCS to Selected Spectrum")
        apply_button.clicked.connect(self._on_apply_vocs_clicked)
        apply_layout.addWidget(apply_button)
        layout.addWidget(apply_box)

        sum_box = QGroupBox("VOCS Sum")
        sum_layout = QVBoxLayout(sum_box)
        sum_intro = QLabel(
            "Corrects and combines every spectrum checked in the "
            "Compare tab onto one common ppm axis, added as a new "
            "spectrum."
        )
        sum_intro.setWordWrap(True)
        sum_layout.addWidget(sum_intro)
        self.vocs_average_checkbox = QCheckBox("Average instead of sum")
        sum_layout.addWidget(self.vocs_average_checkbox)
        sum_button = QPushButton("VOCS Sum (Compare stack)")
        sum_button.clicked.connect(self._on_vocs_sum_clicked)
        sum_layout.addWidget(sum_button)
        layout.addWidget(sum_box)

        layout.addStretch(1)
        return tab

    def _build_imaging_subtab(self) -> QWidget:
        """MRI/gradient-imaging position analysis -- a single new
        input (max gradient) turns the existing Hz axis into a real
        position axis (right-click "Axis Units" on the Spectrum tab
        gains "mm"), computed via
        :func:`opennmr.core.registry.find_gamma` for whichever
        spectrum is currently selected. No calibration table exists
        anywhere in the acquired data itself, so this is deliberately
        user-entered rather than looked up -- matches this project's
        own G/cm convention already used for PFG/diffusion
        (``io.bruker.read_diff_xml``).

        T2-by-integration across a pseudo-2D imaging series needs no
        new feature here at all: the Spectral Fitting sub-tab's
        Integration Regions -> Fit Curve pathway already does exactly
        this, generically -- select a pseudo-2D imaging spectrum's own
        slices, switch the Spectrum tab to mm via the toggle this
        gradient enables, drag out a region the same way as any other
        integration. Time-resolved uptake/exchange tracking is
        explicitly out of scope (not this sub-tab, not this project's
        current session)."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        intro = QLabel(
            "Set a max gradient to turn the Spectrum tab's Hz axis "
            "into a real position axis (right-click -> Axis Units -> "
            "mm). For T2-by-integration across an imaging series, "
            "switch to mm here, then use Integration Regions on the "
            "Spectral Fitting sub-tab as usual -- no separate workflow."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        gradient_box = QGroupBox("Gradient")
        gradient_form = QFormLayout(gradient_box)

        self.imaging_max_gradient_spin = QDoubleSpinBox()
        self.imaging_max_gradient_spin.setRange(0.0, 10000.0)
        self.imaging_max_gradient_spin.setDecimals(2)
        self.imaging_max_gradient_spin.setSuffix(" G/cm")
        self.imaging_max_gradient_spin.setToolTip(
            "Not recorded in the acquired data -- enter the actual "
            "gradient strength used for this experiment. 0 disables "
            "the mm axis option."
        )
        self.imaging_max_gradient_spin.valueChanged.connect(self._on_imaging_gradient_changed)
        gradient_form.addRow("Max gradient:", self.imaging_max_gradient_spin)

        self.imaging_gradient_status_label = QLabel("Select a spectrum to enable the mm axis.")
        self.imaging_gradient_status_label.setWordWrap(True)
        gradient_form.addRow(self.imaging_gradient_status_label)

        layout.addWidget(gradient_box)
        layout.addStretch(1)
        return tab

    def _on_imaging_gradient_changed(self, _value: float) -> None:
        index = self._selected_index()
        if index is not None:
            self.session.spectra[index].metadata.raw["imaging_max_gradient_g_per_cm"] = (
                self.imaging_max_gradient_spin.value() or None
            )
            self._schedule_sidecar_save(index)
        self._refresh_imaging_gradient()

    def _refresh_imaging_gradient(self) -> None:
        """Recompute (and hand to the canvas) the gamma/max-gradient
        pair the ``"mm"`` axis-unit option needs, for whichever
        spectrum is currently selected -- called on every reselection
        (:meth:`_refresh_selection_display`) as well as every max-
        gradient edit, so switching spectra never leaves the mm axis
        showing a stale nucleus's conversion."""
        index = self._selected_index()
        max_gradient = self.imaging_max_gradient_spin.value() or None
        gamma = None
        if index is not None and max_gradient:
            spectrum = self.session.spectra[index]
            if spectrum.metadata.nucleus:
                result = core_registry.find_gamma(spectrum.metadata.nucleus)
                gamma = result["gamma"] if result else None
        self.canvas.set_imaging_gradient_params(gamma, max_gradient)
        if self.canvas.axis_unit == "mm":
            if gamma and max_gradient:
                # Already showing the mm axis -- re-derive it with
                # whatever gamma/gradient just changed, rather than
                # leaving the plot showing a stale conversion until
                # the user happens to toggle the unit again.
                self.canvas.set_axis_unit("mm")
            else:
                # The gradient (or the selected spectrum's nucleus)
                # is no longer valid -- mm no longer means anything,
                # so don't leave the axis frozen on its last (now
                # stale) conversion; ppm is always available.
                self.canvas.set_axis_unit("ppm")
        if max_gradient and gamma:
            self.imaging_gradient_status_label.setText(
                f"{gamma * max_gradient * 1e-5:.4g} Hz/mm at this gradient, for {spectrum.metadata.nucleus}."
            )
        elif max_gradient:
            self.imaging_gradient_status_label.setText(
                "Selected spectrum's nucleus isn't recognised -- can't compute gamma for the mm axis."
            )
        else:
            self.imaging_gradient_status_label.setText("Set a max gradient to enable the mm axis.")

    def _add_derived_spectrum(self, spectrum: NMRSpectrum, label: str) -> int:
        """Append a spectrum computed from already-loaded data (not
        read from a file) into the session and the Data tab's own
        list, selecting it -- the derived-spectrum equivalent of
        :meth:`_load`'s own tail, shared by VOCS and spectrum
        arithmetic."""
        index = self.session.add_derived(spectrum, label)
        self.spectrum_list.addItem(self.session.label(index))
        self._select_single_row(index)
        return index

    def _compare_checked_1d_stack(self) -> list[NMRSpectrum]:
        """Every currently-checked Compare-list entry, resolved to its
        actual 1D spectrum -- stale (removed/reprocessed) or non-1D
        entries silently drop, matching :meth:`_refresh_compare_view`'s
        own tolerance for a Compare list that's drifted out of sync."""
        candidates = [self._resolve_compare_entry(e) for e in self._compare_checked_entries()]
        return [c for c in candidates if c is not None and c.ndim == 1]

    def _on_apply_vocs_clicked(self) -> None:
        index = self._selected_index()
        if index is None:
            QMessageBox.information(self, "No spectrum selected", "Select a spectrum first.")
            return
        spectrum = self.session.spectra[index]
        try:
            corrected = processing_vocs.apply_vocs(
                spectrum, pulse=self.vocs_pulse_edit.text().strip() or "P1",
                window_filter=self.vocs_window_filter_checkbox.isChecked(),
            )
        except Exception as exc:  # noqa: BLE001 -- surfaced to the user, not swallowed
            QMessageBox.critical(self, "VOCS correction failed", str(exc))
            return
        self._add_derived_spectrum(corrected, f"VOCS({self.session.label(index)})")

    def _on_vocs_sum_clicked(self) -> None:
        stack = self._compare_checked_1d_stack()
        if not stack:
            QMessageBox.information(
                self, "Nothing to combine",
                "Check at least one 1D spectrum in the Compare tab first.",
            )
            return
        try:
            result = processing_vocs.vocs_sum(
                stack, pulse=self.vocs_pulse_edit.text().strip() or "P1",
                average=self.vocs_average_checkbox.isChecked(),
                window_filter=self.vocs_window_filter_checkbox.isChecked(),
            )
        except Exception as exc:  # noqa: BLE001 -- surfaced to the user, not swallowed
            QMessageBox.critical(self, "VOCS Sum failed", str(exc))
            return
        ref = stack[0]
        axis = NMRAxis(kind="frequency", ppm=result["ppm"],
                        hz=result["ppm"] * ref.metadata.sfo1 if ref.metadata.sfo1 else None)
        metadata = dataclasses.replace(ref.metadata, raw={})
        combined = NMRSpectrum(data=result["vocs_sum"], axes=[axis], metadata=metadata)
        op = "average" if self.vocs_average_checkbox.isChecked() else "sum"
        self._add_derived_spectrum(combined, f"VOCS {op} ({len(stack)} spectra)")

    def _build_region_box(self) -> QGroupBox:
        """Integration regions -- the "most basic" way to turn a
        pseudo-2D series into a relaxation/diffusion curve, without
        needing a peak fit at all: pick one or more x-axis windows,
        integrate each across every slice (:meth:`_on_integrate_regions_clicked`
        -> :func:`opennmr.processing.relaxation.NMR2D_integrate_slices`),
        and feed the result straight into Fit Curve exactly like a
        peak fit's own per-slice integrals would. The same regions
        also bound the peak fit above when any are defined (passed as
        ``fit_regions`` -- see :meth:`_fit_peaks`), so a spectrum with
        other, unrelated signal elsewhere doesn't drag a peak fit
        toward it."""
        region_box = QGroupBox("Integration Regions")
        region_layout = QVBoxLayout(region_box)

        self.region_table = QTableWidget(0, len(self._REGION_TABLE_COLUMNS))
        self.region_table.setHorizontalHeaderLabels(self._REGION_TABLE_COLUMNS)
        self.region_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.region_table.setMaximumHeight(120)  # same reasoning as peak_table's own cap
        self.region_table.setToolTip(
            "x-axis regions to integrate -- a peak-fitting-free way to "
            "get a per-slice value for Fit Curve, and (when any are "
            "defined) also the window the peak fit above is bounded to."
        )
        self.region_table.itemChanged.connect(lambda _item: self._sync_region_shading())
        region_layout.addWidget(self.region_table)

        region_button_row = QHBoxLayout()
        add_region_button = QPushButton("Add Region")
        add_region_button.clicked.connect(lambda: self._add_region_row())
        remove_region_button = QPushButton("Remove Region")
        remove_region_button.clicked.connect(self._on_remove_region_clicked)
        clear_regions_button = QPushButton("Clear All")
        clear_regions_button.setToolTip("Remove every row from the region table.")
        clear_regions_button.clicked.connect(self._on_clear_regions_clicked)
        region_button_row.addWidget(add_region_button)
        region_button_row.addWidget(remove_region_button)
        region_button_row.addWidget(clear_regions_button)
        region_layout.addLayout(region_button_row)

        self.select_region_button = QPushButton("Select Region from Plot (⌘I)")
        self.select_region_button.setCheckable(True)
        self.select_region_button.setToolTip(
            "While active, drag across the Spectrum plot to add a row "
            "here spanning the dragged x-range, instead of typing it "
            "in. Toggle from anywhere with Cmd/Ctrl+I too."
        )
        self.select_region_button.toggled.connect(self._on_select_region_toggled)
        region_layout.addWidget(self.select_region_button)

        integrate_button = QPushButton("Integrate Regions")
        integrate_button.setToolTip(
            "Step 1, alternative to Fit Spectrum: integrate each "
            "region above across the current selection -- a plain 1D "
            "spectrum, or every slice of a pseudo-2D series -- ready "
            "for Fit Curve without needing a peak fit at all."
        )
        integrate_button.clicked.connect(self._on_integrate_regions_clicked)
        region_layout.addWidget(integrate_button)

        return region_box

    def _add_region_row(self, lo: float = 0.0, hi: float = 1.0) -> None:
        row = self.region_table.rowCount()
        self.region_table.insertRow(row)
        for col, value in enumerate((lo, hi)):
            self.region_table.setItem(row, col, QTableWidgetItem(f"{value:.4g}"))
        self._sync_region_shading()

    def _on_remove_region_clicked(self) -> None:
        row = self.region_table.currentRow()
        if row >= 0:
            self.region_table.removeRow(row)
        self._sync_region_shading()

    def _on_clear_regions_clicked(self) -> None:
        self.region_table.setRowCount(0)
        self._sync_region_shading()

    def _read_region_defs(self) -> list[tuple[float, float]]:
        regions = []
        for row in range(self.region_table.rowCount()):
            lo_item = self.region_table.item(row, 0)
            hi_item = self.region_table.item(row, 1)
            if lo_item is None or hi_item is None:
                continue
            try:
                lo, hi = float(lo_item.text()), float(hi_item.text())
            except ValueError:
                continue
            regions.append((min(lo, hi), max(lo, hi)))
        return regions

    def _sync_region_shading(self) -> None:
        """Reflect the region table's current contents as shaded spans
        on the Spectrum canvas -- called whenever the table itself
        changes (add/remove/clear/edit/drag-select), from
        :meth:`_restore_analysis_state` (needed there for a
        canvas-preserving refresh, e.g. a nucleus/mass override that
        doesn't redraw the canvas at all), and again from
        :meth:`_draw_selected_spectrum` itself (needed *there*
        specifically -- see its own comment -- since
        ``_restore_analysis_state``'s own call runs *before* that
        method's redraw in ``_on_selection_changed``'s call order and
        would otherwise be silently discarded by it). Not hooked into
        every possible Spectrum redraw (a phase drag, a slice switch)
        -- an acceptable, purely cosmetic gap given how many places
        redraw that canvas; the region *definitions* themselves are
        unaffected and stay correct for fitting/integration regardless."""
        self.canvas.set_regions(self._read_region_defs())

    def _refresh_peak_markers(self) -> None:
        """Reflect the peak table's current contents as triangle
        markers on the Spectrum canvas -- same "table and canvas must
        always agree" reasoning as :meth:`_sync_region_shading` just
        above, extended to peaks. Previously only ever called after
        Auto-Pick (which built its own marker list from the detection
        result directly, not the table), so a manually added/removed/
        cleared/edited peak (click-to-pick, the Add/Remove/Clear
        buttons, or a typed-in edit) left the canvas showing stale
        markers or -- for a fresh manual add -- none at all, even
        though the peak was really in the table. Called whenever the
        peak table itself changes, and (see :meth:`_draw_selected_spectrum`)
        after any canvas rebuild that would otherwise silently discard
        the previous marker `Line2D` references."""
        shifts = []
        for row in range(self.peak_table.rowCount()):
            item = self.peak_table.item(row, 0)
            if item is None:
                continue
            try:
                shifts.append(float(item.text()))
            except ValueError:
                continue
        self.canvas.set_peak_markers(shifts or None)

    def _on_select_region_toggled(self, checked: bool) -> None:
        # Mirrors _on_pick_peak_toggled: region-select only ever
        # listens on self.canvas (the Spectrum tab) -- force-switch to
        # it so turning this on always actually works, regardless of
        # which canvas tab happened to be showing.
        self.canvas.set_region_select_mode(checked)
        if checked:
            self.canvas_tabs.setCurrentWidget(self.canvas)

    def _on_region_selected(self, lo: float, hi: float) -> None:
        self._add_region_row(lo, hi)

    def _on_integrate_regions_clicked(self) -> None:
        index = self._selected_index()
        if index is None:
            return
        spectrum = self.session.spectra[index]
        if spectrum.ndim == 2 and not spectrum.is_pseudo_2d:
            QMessageBox.information(
                self, "Not supported",
                "Region integration isn't available for a conventional 2D spectrum yet -- "
                "select a 1D spectrum or a pseudo-2D series.",
            )
            return
        regions = self._read_region_defs()
        if not regions:
            QMessageBox.critical(self, "Integrate failed", "Add at least one region to integrate.")
            return

        try:
            if spectrum.ndim == 1:
                entries = [spectrum]
            else:
                n_slices = spectrum.data.shape[0]
                entries = processing_slicing.get_slices(spectrum, rows=list(range(1, n_slices + 1)))["rows"]
            # NMR2D_integrate_slices reads metadata.raw["slice"] off
            # every entry (get_slices already tags it for a pseudo-2D's
            # own slices; a plain 1D spectrum has no such thing yet --
            # same gap _fit_peaks already works around).
            prepared = [self._ensure_slice_info(s, i + 1) for i, s in enumerate(entries)]
            integrated = processing_relaxation.NMR2D_integrate_slices({"rows": prepared}, regions, mode="rows")
            results = integrated["rows"]
        except Exception as exc:  # noqa: BLE001 -- surfaced to the user, not swallowed
            QMessageBox.critical(self, "Integrate failed", str(exc))
            return

        self._show_integration_results(spectrum, results, len(regions))

    def _add_peak_row(self, shift: float = 0.0, fwhm: float = 1.0, eta: float = 0.5,
                       fixed: bool = False, bounds: dict | None = None) -> None:
        row = self.peak_table.rowCount()
        self.peak_table.insertRow(row)
        for col, value in enumerate((shift, fwhm, eta)):
            self.peak_table.setItem(row, col, QTableWidgetItem(f"{value:.4g}"))
        self.peak_table.setItem(row, 3, self._make_fixed_checkbox_item(fixed))
        if bounds:
            self.peak_table.item(row, 0).setData(Qt.ItemDataRole.UserRole, bounds)
        self._refresh_peak_markers()

    @staticmethod
    def _make_fixed_checkbox_item(checked: bool) -> QTableWidgetItem:
        item = QTableWidgetItem()
        item.setFlags((item.flags() | Qt.ItemFlag.ItemIsUserCheckable) & ~Qt.ItemFlag.ItemIsEditable)
        item.setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
        return item

    def _on_remove_peak_clicked(self) -> None:
        row = self.peak_table.currentRow()
        if row >= 0:
            self.peak_table.removeRow(row)
        self._refresh_peak_markers()

    def _on_clear_peaks_clicked(self) -> None:
        self.peak_table.setRowCount(0)
        self._refresh_peak_markers()

    def _on_peak_table_context_menu(self, pos) -> None:
        row = self.peak_table.currentRow()
        if row < 0:
            return
        menu = QMenu(self)
        bounds_action = menu.addAction("Set Custom Bounds…")
        chosen = menu.exec(self.peak_table.viewport().mapToGlobal(pos))
        if chosen is bounds_action:
            self._on_set_peak_bounds(row)

    #: Parameter keys a peak-fit custom-bounds dialog edits, paired
    #: with the key ``fitting.build_bounds`` actually reads for each
    #: (amplitude is "A" in ``fixed``, but "amplitude" in ``bounds`` --
    #: see fitting.py's own build_bounds/initialize_params_with_map).
    _PEAK_BOUNDS_FIELDS = (("Shift (ppm)", "shift"), ("FWHM (ppm)", "fwhm"),
                           ("Amplitude", "amplitude"), ("Eta", "eta"))

    def _on_set_peak_bounds(self, row: int) -> None:
        existing = self.peak_table.item(row, 0).data(Qt.ItemDataRole.UserRole) or {}
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Custom bounds -- peak {row + 1}")
        form = QFormLayout(dialog)
        edits = {}
        for label, key in self._PEAK_BOUNDS_FIELDS:
            edit = QLineEdit()
            if key in existing:
                edit.setText(f"{existing[key][0]:g}, {existing[key][1]:g}")
            edit.setPlaceholderText("optional, 'low, high'")
            form.addRow(f"{label}:", edit)
            edits[key] = edit
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        bounds = {}
        for key, edit in edits.items():
            text = edit.text().strip()
            if not text:
                continue
            try:
                lo_str, hi_str = text.split(",")
                bounds[key] = (float(lo_str), float(hi_str))
            except ValueError:
                QMessageBox.critical(self, "Invalid bounds", f"{key}: expected 'low, high', got {text!r}.")
                return
        self.peak_table.item(row, 0).setData(Qt.ItemDataRole.UserRole, bounds or None)

    def _on_peak_picked(self, shift: float) -> None:
        if self.pick_peak_button.isChecked():
            # A reasonable default width: 1% of the currently-visible
            # x-range, since there's no per-peak width information in
            # a single click -- the fit refines it regardless.
            xmin, xmax = self.canvas.ax.get_xlim()
            fwhm_guess = abs(xmax - xmin) * 0.01 or 1.0
            self._add_peak_row(shift=shift, fwhm=fwhm_guess, eta=0.5)

    def _on_pick_peak_toggled(self, checked: bool) -> None:
        # Pick mode only ever listens on self.canvas (the Spectrum
        # tab) -- clicking elsewhere (Compare, Fit, Spectral Fit, the
        # Processing-preview tab, all reachable without leaving the
        # Analysis tab this button lives on) silently did nothing,
        # which is almost certainly why this "doesn't seem to work"
        # was reported: nothing was wrong with picking itself, the
        # canvas being clicked just wasn't the one listening.
        self.canvas.set_pick_mode(checked)
        if checked:
            self.canvas_tabs.setCurrentWidget(self.canvas)

    def _on_auto_pick_peaks_clicked(self) -> None:
        """Replace the peak table with automatically-detected peaks
        (:func:`opennmr.processing.fitting.auto_pick_peaks`) from the
        current selection -- slice 1, for a pseudo-2D series, since
        auto-picking is a starting point, not a per-slice operation."""
        index = self._selected_index()
        if index is None:
            return
        spectrum = self.session.spectra[index]
        if spectrum.ndim == 2 and not spectrum.is_pseudo_2d:
            QMessageBox.information(
                self, "Not supported",
                "Auto-pick isn't available for a conventional 2D spectrum yet -- "
                "select a 1D spectrum or a pseudo-2D series.",
            )
            return
        try:
            target = spectrum if spectrum.ndim == 1 else processing_slicing.get_trace(spectrum, axis="f1", index=1)
            peaks = processing_fitting.auto_pick_peaks(target, prominence=self.auto_pick_prominence_spin.value())
        except Exception as exc:  # noqa: BLE001 -- surfaced to the user, not swallowed
            QMessageBox.critical(self, "Auto-pick failed", str(exc))
            return
        if not peaks:
            QMessageBox.information(
                self, "No peaks found",
                "No peaks found automatically -- try adding peaks manually, "
                "or check the spectrum has real signal above the noise floor.",
            )
            return
        self.peak_table.setRowCount(0)
        for peak in peaks:
            # Explicit user request: auto-pick should leave a visual
            # indication on the spectrum, not just fill the table -- a
            # small marker at each detected peak's own shift.
            # _add_peak_row's own _refresh_peak_markers call handles
            # this (reads the whole table, so the last iteration here
            # already reflects every peak added, not just its own row).
            self._add_peak_row(shift=peak["shift"], fwhm=peak["fwhm"], eta=peak["eta"])

    def _restore_sr_control(self, spectrum) -> None:
        self.sr_shift_spin.blockSignals(True)
        self.sr_shift_spin.setValue(spectrum.metadata.raw.get("sr_shift_hz", 0.0))
        self.sr_shift_spin.blockSignals(False)

    def _on_apply_sr_clicked(self) -> None:
        self._apply_reference_shift_hz(self.sr_shift_spin.value())

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
            self, "Reference shift", "SR (Hz):", current_hz, -1.0e6, 1.0e6, 4,
        )
        if ok:
            self._apply_reference_shift_hz(value)

    @staticmethod
    def _capture_original_axis(spectrum) -> None:
        """Stash the direct (F2) axis's hz/ppm arrays exactly as
        loaded, the first time SR is ever touched for this spectrum
        this session -- the stable baseline every SR value is
        recomputed from, so applying "the same SR twice" is a no-op
        instead of compounding, and applying "a different SR" replaces
        rather than adds to whatever's currently displayed. Reprocessing
        the FID naturally invalidates this (a fresh NMRSpectrum from
        the vendor reader has no ``raw`` carried over), which is the
        right behaviour -- SR starts over from the new axis."""
        if "sr_original_hz" in spectrum.metadata.raw:
            return
        axis = spectrum.axes[-1]
        spectrum.metadata.raw["sr_original_hz"] = axis.hz.copy() if axis.hz is not None else None
        spectrum.metadata.raw["sr_original_ppm"] = axis.ppm.copy() if axis.ppm is not None else None

    def _apply_reference_shift_hz(self, sr_hz: float) -> None:
        """Set the spectrum's SR to the absolute value ``sr_hz`` --
        the Hz offset from its own SFO1 base frequency -- recomputing
        the direct-dimension axis from the original, as-loaded arrays
        (:meth:`_capture_original_axis`) rather than adding onto
        whatever the axis currently is, so this is idempotent (calling
        it again with the same value is a no-op) rather than
        cumulative."""
        index = self._selected_index()
        if index is None:
            return
        spectrum = self.session.spectra[index]
        self._capture_original_axis(spectrum)

        axis = spectrum.axes[-1]  # direct (F2) dimension -- always frequency
        original_hz = spectrum.metadata.raw.get("sr_original_hz")
        original_ppm = spectrum.metadata.raw.get("sr_original_ppm")
        if original_hz is not None:
            axis.hz = original_hz + sr_hz
        if original_ppm is not None and spectrum.metadata.sfo1:
            axis.ppm = original_ppm + sr_hz / spectrum.metadata.sfo1

        spectrum.metadata.raw["sr_shift_hz"] = sr_hz
        self._restore_sr_control(spectrum)
        self._schedule_sidecar_save(index)

        # A full show_spectrum() redraw (via _draw_selected_spectrum)
        # always resets the view to ppm -- fine for an actual spectrum
        # *selection*, but SR is applied to whatever's already on
        # screen, so save/restore the axis-unit choice (ppm/Hz/kHz)
        # around it instead of silently dropping back to ppm.
        unit = self.canvas.axis_unit
        self._draw_selected_spectrum(spectrum)
        if unit != "ppm":
            self.canvas.set_axis_unit(unit)

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
            fixed_item = self.peak_table.item(row, 3)
            if fixed_item is not None and fixed_item.checkState() == Qt.CheckState.Checked:
                peak_def["fixed"] = {"shift": True, "fwhm": True, "A": True, "eta": True}
            custom_bounds = self.peak_table.item(row, 0).data(Qt.ItemDataRole.UserRole)
            if custom_bounds:
                peak_def["bounds"] = {**peak_def["bounds"], **custom_bounds}
            peak_defs.append(peak_def)
        return peak_defs

    def _read_sn_gating_kwargs(self) -> dict:
        text = self.sn_noise_region_edit.text().strip()
        if not text:
            return {}
        try:
            lo_str, hi_str = text.split(",")
            noise_region = (float(lo_str), float(hi_str))
        except ValueError as exc:
            raise ValueError("Noise region must be 'low, high' (ppm).") from exc
        return {"noise_region": noise_region, "sn_threshold": self.sn_threshold_spin.value()}

    def _is_diffusion_fit(self) -> bool:
        return self.fit_type_combo.currentIndex() == 1

    def _populate_fit_model_combo(self) -> None:
        items = _DIFFUSION_MODEL_ITEMS if self._is_diffusion_fit() else _RELAXATION_MODEL_ITEMS
        self.relaxation_model_combo.blockSignals(True)
        self.relaxation_model_combo.clear()
        for label, key in items:
            self.relaxation_model_combo.addItem(label, key)
        self.relaxation_model_combo.addItem("Custom equation…", "custom")
        self.relaxation_model_combo.blockSignals(False)
        self._on_relaxation_model_changed(self.relaxation_model_combo.currentIndex())

    def _on_fit_type_changed(self, _index: int) -> None:
        self._populate_fit_model_combo()
        self.delays_row_label.setText("b-values:" if self._is_diffusion_fit() else "Delays:")
        unit = "s/m²" if self._is_diffusion_fit() else "s"
        self.manual_delays_edit.setPlaceholderText(f"e.g. 1e8, 5e8, 2e9 ({unit})" if self._is_diffusion_fit()
                                                     else "e.g. 0.001, 0.01, 0.1, 1.0 (s)")

    def _on_relaxation_model_changed(self, _index: int) -> None:
        is_custom = self.relaxation_model_combo.currentData() == "custom"
        for widget in (
            self.custom_equation_row_label, self.custom_equation_edit,
            self.custom_params_label, self.custom_p0_row_label, self.custom_p0_edit,
        ):
            widget.setVisible(is_custom)
        if is_custom:
            self._on_custom_equation_changed(self.custom_equation_edit.text())

    def _on_custom_equation_changed(self, text: str) -> None:
        """Live preview of the parameters :func:`_build_custom_relaxation_model`
        would detect from the equation typed so far -- lets a typo or a
        missing independent variable be caught before Fit Curve is
        even clicked, rather than only surfacing as an error dialog."""
        text = text.strip()
        if not text:
            self.custom_params_label.setText("")
            return
        try:
            spec = processing_relaxation.build_custom_model(text)
        except ValueError as exc:
            self.custom_params_label.setText(f"Error: {exc}")
            return
        self.custom_params_label.setText(f"Parameters (in order): {', '.join(spec['params'])}")

    def _build_custom_relaxation_model(self) -> dict:
        equation = self.custom_equation_edit.text().strip()
        if not equation:
            raise ValueError("Enter an equation for the custom model (e.g. A*exp(-x/T1)+C).")
        p0_text = self.custom_p0_edit.text().strip()
        p0_overrides = [float(tok) for tok in p0_text.split(",") if tok.strip()] if p0_text else None
        return processing_relaxation.build_custom_model(equation, p0_overrides=p0_overrides)

    def _on_delay_source_changed(self, index: int) -> None:
        self.manual_delays_edit.setVisible(index == 1)
        self.delay_start_row_label.setVisible(index == 2)
        self.delay_start_spin.setVisible(index == 2)
        self.delay_step_row_label.setVisible(index == 2)
        self.delay_step_spin.setVisible(index == 2)
        self.vclist_factor_row_label.setVisible(index == 3)
        self.vclist_factor_spin.setVisible(index == 3)

    def _resolve_independent_values(self, spectrum, is_diffusion: bool, n_slices: int | None = None) -> np.ndarray | dict:
        """Resolve the x-axis values (delays in seconds, or b-values in
        s/m^2) to fit against, per the Values-row source combo.

        ``n_slices`` : overrides ``spectrum.data.shape[0]`` -- needed
        when ``spectrum`` is only supplying vendor/path *context*
        (Auto/vclist/relative-gradient source lookups) rather than
        being the actual series being fit, e.g. a Compare-tab stack of
        independently-loaded 1D spectra, whose own slice count has
        nothing to do with any single spectrum's own dimensions."""
        source = self.delay_source_combo.currentIndex()
        n_slices = n_slices if n_slices is not None else spectrum.data.shape[0]

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

    @staticmethod
    def _ensure_slice_info(spectrum, index: int):
        """``NMR2D_fit_slices`` requires ``metadata.raw["slice"]``
        (the way ``processing.slicing.get_slices`` already tags every
        row/column it extracts) -- a plain 1D spectrum, or a Compare-
        tab stack entry, has no such thing yet. Injects a minimal one
        rather than duplicating that function's fitting logic just to
        skip the lookup. Doesn't mutate ``spectrum`` itself (a fresh
        object, via ``dataclasses.replace``) -- this may run against
        the live session spectrum, and slice-tagging it in place would
        leak into every other consumer of its metadata."""
        if "slice" in spectrum.metadata.raw:
            return spectrum
        return dataclasses.replace(
            spectrum,
            metadata=dataclasses.replace(spectrum.metadata, raw={
                **spectrum.metadata.raw,
                "slice": {"slice_type": "row", "index": index, "shift": None, "shift_unit": None, "width": None},
            }),
        )

    def _fit_peaks(self, entries: list) -> list[dict]:
        """Spectral (pseudo-Voigt peak) fit -- step 1 of the analysis
        workflow, shared by a plain 1D spectrum (``entries`` is a
        single-item list), a pseudo-2D series (every slice), and a
        Compare-tab stack (every checked entry). Bounded to the
        Integration Regions table's own regions when any are defined
        (``fit_regions`` -- restricts which part of the spectrum the
        fit itself considers, so unrelated signal elsewhere can't drag
        it off) -- entirely optional, an empty region table fits the
        whole spectrum exactly as before. Also optionally S/N-gated
        (:meth:`_read_sn_gating_kwargs`) -- once a peak's signal-to-
        noise (against a user-marked noise region) falls below the
        threshold across a series, its amplitude is fixed to zero
        rather than the fit chasing noise once a real peak has decayed
        away."""
        peak_defs = self._read_peak_defs()
        if not peak_defs:
            raise ValueError("Add at least one peak to fit.")
        prepared = [self._ensure_slice_info(s, i + 1) for i, s in enumerate(entries)]
        fit_regions = self._read_region_defs() or None
        sn_kwargs = self._read_sn_gating_kwargs()
        fitted = processing_fitting.NMR2D_fit_slices(
            {"rows": prepared}, peak_defs, mode="rows", fit_regions=fit_regions,
            show_report=False, show_warnings=False, **sn_kwargs,
        )
        return fitted["rows"]

    def _store_slice_results(self, spectrum, results: list[dict]) -> None:
        self._slice_fit_results = results
        # Kept per-spectrum (not just in the live _slice_fit_results
        # attribute) so "Spectrum fit (slice #)"/Fit Curve survive a
        # reselection -- analysis_state alone only remembers the *text*
        # summary, not the underlying per-slice results.
        spectrum.metadata.raw["slice_fit_results"] = results

    def _show_spectral_fit_results(self, spectrum, results: list[dict]) -> None:
        self._store_slice_results(spectrum, results)
        if results:
            self._update_peak_table_from_fit(results[0])
        self.view_slice_fit_spin.blockSignals(True)
        self.view_slice_fit_spin.setEnabled(True)
        self.view_slice_fit_spin.setMaximum(max(1, len(results)))
        self.view_slice_fit_spin.setValue(1)
        self.view_slice_fit_spin.blockSignals(False)
        self._on_view_slice_fit_changed(1)  # show immediately, not just once the spinner is touched

    def _show_integration_results(self, spectrum, results: list[dict], n_regions: int) -> None:
        """Region-integration's own counterpart to
        :meth:`_show_spectral_fit_results` -- stored into the exact
        same slot (:attr:`_slice_fit_results`/``metadata.raw
        ["slice_fit_results"]``) so Fit Curve picks it up exactly like
        a peak fit's own results, with no separate code path needed
        there at all. Doesn't touch ``view_slice_fit_spin``/the
        Spectral Fit canvas, though -- :meth:`_display_slice_fit`
        needs a real peak-fit shape (``popt``/``pcov``/``peak_defs``/
        ``y_fit``), which a plain integral has no equivalent of.
        Writes into ``integration_results_label`` (the Integration
        sub-tab's own results display), not ``analysis_results_label``
        (Peak Picking's) -- the two used to share one label back when
        both lived on the same "Spectral Fitting" sub-tab, still
        cleared here for the same reason (a stale peak-fit params
        display from an earlier Fit Spectrum run would be misleading
        once results come from a plain integral instead)."""
        self._store_slice_results(spectrum, results)
        self.view_slice_fit_spin.blockSignals(True)
        self.view_slice_fit_spin.setEnabled(False)
        self.view_slice_fit_spin.blockSignals(False)
        self.spectral_fit_params_label.setText("")
        self.integration_results_label.setText(
            f"Integrated {len(results)} slice(s) across {n_regions} region(s) -- "
            f"ready for Fit Curve (Value source: Integrated area)."
        )
        integral_index = self.value_source_combo.findData("integral")
        if integral_index >= 0:
            self.value_source_combo.setCurrentIndex(integral_index)

    def _on_fit_spectrum_clicked(self) -> None:
        """Step 1 of the analysis workflow: fit peaks against the
        currently selected spectrum -- directly for a plain 1D
        spectrum, or across every slice for a pseudo-2D series.
        Always the prerequisite for "Fit Curve" (step 2, the
        relaxation/diffusion fit -- a spectrum-list/Compare-tab
        context menu item, not a button here), which reuses these
        results rather than re-fitting peaks itself when they're
        already available."""
        index = self._selected_index()
        if index is None:
            return
        spectrum = self.session.spectra[index]
        if spectrum.ndim == 2 and not spectrum.is_pseudo_2d:
            QMessageBox.information(
                self, "Not supported",
                "Spectral fitting isn't available for a conventional 2D spectrum yet -- "
                "select a 1D spectrum or a pseudo-2D series.",
            )
            return

        try:
            if spectrum.ndim == 1:
                entries = [spectrum]
            else:
                n_slices = spectrum.data.shape[0]
                entries = processing_slicing.get_slices(spectrum, rows=list(range(1, n_slices + 1)))["rows"]
            results = self._fit_peaks(entries)
        except Exception as exc:  # noqa: BLE001 -- surfaced to the user, not swallowed
            QMessageBox.critical(self, "Fit failed", str(exc))
            return

        self._show_spectral_fit_results(spectrum, results)

    def _run_curve_fit(self, slice_results: list[dict], n_slices: int, reference_spectrum,
                        persist_to=None) -> None:
        """Step 2 of the analysis workflow: fit a relaxation/diffusion
        model to each peak's per-slice amplitude series.

        ``slice_results`` : from :meth:`_fit_peaks` -- the caller
        supplies these (freshly computed, or reused from a prior "Fit
        Spectrum") rather than this method ever re-fitting peaks
        itself, so spectral fitting always genuinely comes first.
        ``reference_spectrum`` : supplies vendor/path context for the
        Values-source combo's Auto/vclist/relative-gradient options
        (see :meth:`_resolve_independent_values`'s own ``n_slices``
        override -- may not be the same length as this fit's actual
        series, e.g. a Compare-tab stack of independently-loaded
        files). ``persist_to`` : an ``NMRSpectrum`` to silo the result
        into (``None`` for a Compare-tab stack fit, which has no
        single natural owner to silo against -- matches the Compare
        tab's own existing "no per-spectrum persistence" character).
        """
        is_diffusion = self._is_diffusion_fit()
        try:
            independent_values = self._resolve_independent_values(reference_spectrum, is_diffusion, n_slices=n_slices)
            model_key = self.relaxation_model_combo.currentData()
            custom_model = self._build_custom_relaxation_model() if model_key == "custom" else None
            results = processing_relaxation.NMR2D_fit_relaxation(
                slice_results, model=model_key, vdlist=independent_values, show_report=False,
                value_source=self.value_source_combo.currentData(), custom_model=custom_model,
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

        self._relaxation_fit_results = results
        # A shallow copy of the list (each dict is still shared) is
        # enough -- _refit_relaxation_peak only ever reads from
        # _relaxation_fit_original, never mutates a dict in place, and
        # replaces _relaxation_fit_results[idx] wholesale on a refit.
        self._relaxation_fit_original = list(results)
        self._excluded_relaxation_indices = {}
        self._relaxation_fit_xlabel = xlabel
        self.view_relaxation_peak_spin.blockSignals(True)
        self.view_relaxation_peak_spin.setEnabled(True)
        self.view_relaxation_peak_spin.setMaximum(max(1, len(results)))
        self.view_relaxation_peak_spin.setValue(1)
        self.view_relaxation_peak_spin.blockSignals(False)
        self._display_relaxation_peak(0)
        # Force the tab visible: a Compare-stack fit has no single
        # "owning" spectrum, so it can run while the Data tab's own
        # selection isn't pseudo-2D (or is empty) -- the selection-
        # driven _update_fit_tab_visibility check alone wouldn't show it.
        self.canvas_tabs.setTabVisible(self.canvas_tabs.indexOf(self.fit_canvas), True)
        self.canvas_tabs.setCurrentWidget(self.fit_canvas)

        if persist_to is not None:
            persist_to.metadata.raw["relaxation_fit_results"] = results
            persist_to.metadata.raw["relaxation_fit_xlabel"] = xlabel

    def _on_view_relaxation_peak_changed(self, value: int) -> None:
        self._display_relaxation_peak(value - 1)

    def _display_relaxation_peak(self, index: int) -> None:
        """Show peak ``index`` (0-based) of :attr:`_relaxation_fit_results`
        in the Fit tab, with an uncertainty band if the checkbox for it
        is on -- shared by the spin box, the uncertainty checkbox, and
        anywhere else a fresh curve-fit result needs displaying."""
        if not self._relaxation_fit_results:
            return
        if not (0 <= index < len(self._relaxation_fit_results)):
            return
        result = self._relaxation_fit_results[index]
        uncertainty = None
        if self.relaxation_uncertainty_checkbox.isChecked():
            try:
                uncertainty = processing_relaxation.fit_curve_uncertainty(result)
            except Exception:  # noqa: BLE001 -- best-effort; fall back to the plain fit curve
                uncertainty = None

        excluded_points = None
        excluded_idx = self._excluded_relaxation_indices.get(index)
        if excluded_idx and self._relaxation_fit_original:
            original = self._relaxation_fit_original[index]
            idx_array = sorted(excluded_idx)
            excluded_points = (original["t"][idx_array], original["y"][idx_array])

        self.fit_canvas.show_relaxation_fit(
            result, uncertainty=uncertainty, xlabel=self._relaxation_fit_xlabel, excluded_points=excluded_points,
        )

        rate_param = result["rate_param"]
        if rate_param is not None:
            is_diffusion = self._is_diffusion_fit()
            formatter = processing_relaxation.format_diffusion_value if is_diffusion else processing_relaxation.format_time_value
            value_str = formatter(result["params"][rate_param], result["errors"].get(rate_param))
            self.relaxation_fit_params_label.setText(f"{result.get('peak_label', '')}: {rate_param} = {value_str}")
        else:
            params_str = ", ".join(f"{k}={v:.4g}" for k, v in result["params"].items())
            self.relaxation_fit_params_label.setText(f"{result.get('peak_label', '')}: {params_str}")

    def _on_fit_point_exclude_requested(self, x: float, y: float) -> None:
        """Right-click "Exclude Nearest Point" on the Fit tab -- finds
        the currently-included point of the currently-viewed peak
        closest to ``(x, y)`` (in axis-range-normalised distance, so
        the delay/b-value and intensity axes -- usually on very
        different numeric scales -- don't let one dominate "nearest"),
        excludes it, and re-fits that one peak without it. Real,
        user-reported motivation: the uncertainty band and the fit
        itself can both be thrown off by a single bad point (e.g. a
        slice with peak overlap or a corrupted intensity estimate),
        and this is the direct, standard way to deal with that in NMR
        curve fitting -- exclude the specific known-bad point, not
        guess at a more "robust" automatic fitting method instead."""
        if not self._relaxation_fit_original:
            return
        index = self.view_relaxation_peak_spin.value() - 1
        if not (0 <= index < len(self._relaxation_fit_original)):
            return
        original = self._relaxation_fit_original[index]
        t_orig, y_orig = original["t"], original["y"]
        excluded = self._excluded_relaxation_indices.setdefault(index, set())
        candidates = [i for i in range(len(t_orig)) if i not in excluded]
        if not candidates:
            return
        t_range = float(np.ptp(t_orig)) or 1.0
        y_range = float(np.ptp(y_orig)) or 1.0
        distances = [
            ((t_orig[i] - x) / t_range) ** 2 + ((y_orig[i] - y) / y_range) ** 2 for i in candidates
        ]
        nearest = candidates[int(np.argmin(distances))]
        excluded.add(nearest)
        if not self._refit_relaxation_peak(index):
            excluded.discard(nearest)  # not enough points left -- undo

    def _on_fit_points_reset_requested(self) -> None:
        """Right-click "Reset Excluded Points" -- re-includes every
        point of the currently-viewed peak and re-fits."""
        if not self._relaxation_fit_original:
            return
        index = self.view_relaxation_peak_spin.value() - 1
        if not (0 <= index < len(self._relaxation_fit_original)):
            return
        self._excluded_relaxation_indices.pop(index, None)
        self._refit_relaxation_peak(index)

    def _refit_relaxation_peak(self, index: int) -> bool:
        """Re-run :func:`processing.relaxation.fit_relaxation` for
        peak ``index`` alone, using :attr:`_relaxation_fit_original`'s
        full (t, y[, y_err]) minus whatever's in
        :attr:`_excluded_relaxation_indices` for it, and redisplay.
        Returns whether it actually ran (``False`` if there weren't
        enough points left, or the fit itself raised, in which case
        the caller is responsible for any state rollback -- this
        method only ever reads :attr:`_excluded_relaxation_indices`,
        never mutates it). Deliberately re-fits with plain defaults
        (``fit_relaxation``'s own automatic ``p0``/``bounds``) rather
        than reusing whatever custom fixed/bounds the original Fit
        Curve run might have used -- a real, accepted simplification
        for now (Fit Curve itself doesn't currently expose fixed/
        bounds from the GUI either, so nothing is actually lost yet)."""
        original = self._relaxation_fit_original[index]
        excluded = self._excluded_relaxation_indices.get(index, set())
        keep = [i for i in range(len(original["t"])) if i not in excluded]
        if len(keep) < 2:
            QMessageBox.information(
                self, "Not enough points", "At least 2 points are needed to fit -- re-include some first.",
            )
            return False
        t = original["t"][keep]
        y = original["y"][keep]
        y_err = original["y_err"][keep] if original.get("y_err") is not None else None
        try:
            result = processing_relaxation.fit_relaxation(t, y, model=original["model"], y_err=y_err)
        except Exception as exc:  # noqa: BLE001 -- surfaced to the user, not swallowed
            QMessageBox.critical(self, "Fit failed", str(exc))
            return False
        result["peak_label"] = original.get("peak_label")
        self._relaxation_fit_results[index] = result
        self._display_relaxation_peak(index)
        return True

    def _on_fit_curve_button_clicked(self) -> None:
        """"Fit Curve" as a real button in the Analysis tab -- the
        currently selected spectrum stands in for the ``index`` a
        right-click on a specific Spectra-tab row would otherwise
        supply (see :meth:`_on_fit_curve_pseudo2d_action`, still also
        reachable that way for a row that isn't the current selection)."""
        index = self._selected_index()
        if index is None:
            return
        spectrum = self.session.spectra[index]
        if not (spectrum.ndim == 2 and spectrum.is_pseudo_2d):
            QMessageBox.information(
                self, "Not supported",
                "Fit Curve needs a pseudo-2D series (e.g. a T1/T2/diffusion "
                "series) selected here. For a stack of separately-loaded 1D "
                "spectra instead, use the Compare tab's own Fit Curve button.",
            )
            return
        self._on_fit_curve_pseudo2d_action(index)

    def _on_fit_curve_pseudo2d_action(self, index: int) -> None:
        """"Fit Curve" from the Data tab's right-click menu, only
        offered for a pseudo-2D row (see
        :meth:`_on_spectrum_list_context_menu`). Selects that row
        first (so the peak table/Values-source/model shown are that
        spectrum's own siloed Analysis-tab state, not whatever was
        showing for a different selection), then reuses its spectral
        fit if one already exists (from "Fit Spectrum"), or runs one
        now with the peak_defs just brought into view."""
        self._select_single_row(index)
        spectrum = self.session.spectra[index]
        try:
            slice_results = spectrum.metadata.raw.get("slice_fit_results")
            if not slice_results:
                n_slices = spectrum.data.shape[0]
                entries = processing_slicing.get_slices(spectrum, rows=list(range(1, n_slices + 1)))["rows"]
                slice_results = self._fit_peaks(entries)
                self._show_spectral_fit_results(spectrum, slice_results)
        except Exception as exc:  # noqa: BLE001 -- surfaced to the user, not swallowed
            QMessageBox.critical(self, "Fit failed", str(exc))
            return
        self._run_curve_fit(slice_results, len(slice_results), spectrum, persist_to=spectrum)

    def _on_view_slice_fit_changed(self, value: int) -> None:
        if not self._slice_fit_results:
            return
        idx = value - 1
        if not (0 <= idx < len(self._slice_fit_results)):
            return
        result = self._slice_fit_results[idx]
        self._display_slice_fit(result, switch_tab=True)

    @staticmethod
    def _format_fit_params(per_peak: list[dict], peak_labels: list[str]) -> str:
        lines = []
        for label, peak in zip(peak_labels, per_peak):
            parts = [
                f"{name}={value:.4g}±{error:.2g}"
                for name, value, error in zip(peak["names"], peak["params"], peak["errors"])
            ]
            lines.append(f"{label}: " + ", ".join(parts))
        return "\n".join(lines)

    def _update_peak_table_from_fit(self, result: dict) -> None:
        """After a spectral fit, replace each peak row's starting
        values (shift/FWHM/eta) with this result's fitted values --
        so a re-fit, or a later Fit Curve (which reuses these same
        peak_defs), starts from the converged position rather than
        the original guess."""
        per_peak = processing_fitting.extract_params_and_errors(
            result["popt"], result["pcov"], result["peak_defs"], result["is_free"],
        )
        for row, peak in enumerate(per_peak):
            if row >= self.peak_table.rowCount():
                break
            params = dict(zip(peak["names"], peak["params"]))
            # Preserve any custom bounds already attached to this row's
            # column-0 item (see _on_set_peak_bounds) -- a fresh
            # QTableWidgetItem here would otherwise silently drop them.
            custom_bounds = self.peak_table.item(row, 0).data(Qt.ItemDataRole.UserRole) \
                if self.peak_table.item(row, 0) else None
            shift_item = QTableWidgetItem(f"{params['shift']:.4g}")
            if custom_bounds:
                shift_item.setData(Qt.ItemDataRole.UserRole, custom_bounds)
            self.peak_table.setItem(row, 0, shift_item)
            self.peak_table.setItem(row, 1, QTableWidgetItem(f"{params['fwhm']:.4g}"))
            if "eta" in params:
                self.peak_table.setItem(row, 2, QTableWidgetItem(f"{params['eta']:.4g}"))

    @staticmethod
    def _slice_label_from_fit_result(result: dict) -> str:
        return f"Slice {result['index']} spectral fit"

    def _display_slice_fit(self, result: dict, switch_tab: bool = False) -> None:
        """Draw one spectral-fit result (data, total fit, and each
        peak's own component) into the Spectral Fit canvas, and its
        numeric parameters into ``spectral_fit_params_label`` --
        shared by browsing slices via ``view_slice_fit_spin`` and by
        ``_silo_other_canvas_tabs`` restoring a reselected
        spectrum's already-computed fit."""
        title = self._slice_label_from_fit_result(result)
        per_peak = processing_fitting.extract_params_and_errors(
            result["popt"], result["pcov"], result["peak_defs"], result["is_free"],
        )
        peak_labels = [f"Peak {i + 1} ({p['params'][0]:.4g} ppm)" for i, p in enumerate(per_peak)]
        self.spectral_fit_canvas.show_slice_fit(
            result["x"], result["y"], result["y_fit"],
            components=result.get("components"), peak_labels=peak_labels, title=title,
            spectrum=result.get("slice"),
        )
        self.spectral_fit_params_label.setText(self._format_fit_params(per_peak, peak_labels))
        if switch_tab:
            self.canvas_tabs.setCurrentWidget(self.spectral_fit_canvas)

    def _refresh_analysis_tab(self, spectrum) -> None:
        is_pseudo2d = spectrum.ndim == 2 and spectrum.is_pseudo_2d
        if is_pseudo2d:
            self.analysis_status_label.setText(
                f"Fit Spectrum: peaks across all {spectrum.data.shape[0]} slices. Then Fit Curve "
                f"(a relaxation/diffusion curve per peak)."
            )
        elif spectrum.ndim == 1:
            self.analysis_status_label.setText(
                "Fit Spectrum: peaks in this spectrum. For a relaxation/diffusion curve, either select "
                "a pseudo-2D series instead, or build a stack of 1D spectra in the Compare tab."
            )
        else:
            self.analysis_status_label.setText(
                "Spectral fitting isn't available for a conventional 2D spectrum yet."
            )
        self._restore_analysis_state(spectrum)

    def _capture_analysis_state(self) -> dict:
        peaks = []
        peak_bounds = []
        for row in range(self.peak_table.rowCount()):
            row_values = [self.peak_table.item(row, col).text() if self.peak_table.item(row, col) else ""
                          for col in range(3)]  # Shift, FWHM, Eta -- plain text columns
            fixed_item = self.peak_table.item(row, 3)
            row_values.append("1" if fixed_item and fixed_item.checkState() == Qt.CheckState.Checked else "0")
            peaks.append(row_values)
            shift_item = self.peak_table.item(row, 0)
            peak_bounds.append((shift_item.data(Qt.ItemDataRole.UserRole) if shift_item else None) or {})
        regions = [
            [self.region_table.item(row, col).text() if self.region_table.item(row, col) else ""
             for col in range(len(self._REGION_TABLE_COLUMNS))]
            for row in range(self.region_table.rowCount())
        ]
        return {
            "peaks": peaks,
            "peak_bounds": peak_bounds,
            "regions": regions,
            "fit_type": self.fit_type_combo.currentIndex(),
            "model": self.relaxation_model_combo.currentIndex(),
            "custom_equation": self.custom_equation_edit.text(),
            "custom_p0": self.custom_p0_edit.text(),
            "value_source": self.value_source_combo.currentIndex(),
            "delay_source": self.delay_source_combo.currentIndex(),
            "manual_text": self.manual_delays_edit.text(),
            "start": self.delay_start_spin.value(),
            "step": self.delay_step_spin.value(), "vclist_factor": self.vclist_factor_spin.value(),
            "position_tolerance": self.position_tolerance_spin.value(),
            "sn_noise_region": self.sn_noise_region_edit.text(),
            "sn_threshold": self.sn_threshold_spin.value(),
            "results_text": self.analysis_results_label.text(),
            "integration_results_text": self.integration_results_label.text(),
        }

    def _save_current_analysis_state(self) -> None:
        index = self._current_analysis_index
        if index is None or not (0 <= index < len(self.session)):
            return
        self.session.spectra[index].metadata.raw["analysis_state"] = self._capture_analysis_state()
        self._schedule_sidecar_save(index)

    def _restore_analysis_state(self, spectrum) -> None:
        self.peak_table.setRowCount(0)
        self.region_table.setRowCount(0)
        self.analysis_results_label.setText("")
        self.integration_results_label.setText("")
        self.spectral_fit_params_label.setText("")

        state = spectrum.metadata.raw.get("analysis_state")
        if state is None:
            self.fit_type_combo.setCurrentIndex(0)
            self.custom_equation_edit.clear()
            self.custom_p0_edit.clear()
            self.value_source_combo.setCurrentIndex(0)
            self.delay_source_combo.setCurrentIndex(0)
            self.manual_delays_edit.clear()
            self.delay_start_spin.setValue(0.0)
            self.delay_step_spin.setValue(0.1)
            self.vclist_factor_spin.setValue(1.0)
            self.position_tolerance_spin.setValue(0.5)
            self.sn_noise_region_edit.clear()
            self.sn_threshold_spin.setValue(3.0)
        else:
            peak_bounds = state.get("peak_bounds", [])
            for i, peak_row in enumerate(state["peaks"]):
                shift, fwhm, eta = peak_row[0], peak_row[1], peak_row[2]
                fixed = len(peak_row) > 3 and peak_row[3] == "1"
                row = self.peak_table.rowCount()
                self.peak_table.insertRow(row)
                for col, value in enumerate((shift, fwhm, eta)):
                    self.peak_table.setItem(row, col, QTableWidgetItem(value))
                self.peak_table.setItem(row, 3, self._make_fixed_checkbox_item(fixed))
                if i < len(peak_bounds) and peak_bounds[i]:
                    self.peak_table.item(row, 0).setData(Qt.ItemDataRole.UserRole, peak_bounds[i])
            for lo, hi in state.get("regions", []):
                row = self.region_table.rowCount()
                self.region_table.insertRow(row)
                for col, value in enumerate((lo, hi)):
                    self.region_table.setItem(row, col, QTableWidgetItem(value))
            self.fit_type_combo.setCurrentIndex(state["fit_type"])
            self._populate_fit_model_combo()
            self.relaxation_model_combo.setCurrentIndex(state["model"])
            self.custom_equation_edit.setText(state.get("custom_equation", ""))
            self.custom_p0_edit.setText(state.get("custom_p0", ""))
            self.value_source_combo.setCurrentIndex(state.get("value_source", 0))
            self.delay_source_combo.setCurrentIndex(state["delay_source"])
            self.manual_delays_edit.setText(state["manual_text"])
            self.delay_start_spin.setValue(state["start"])
            self.delay_step_spin.setValue(state["step"])
            self.vclist_factor_spin.setValue(state["vclist_factor"])
            self.position_tolerance_spin.setValue(state.get("position_tolerance", 0.5))
            self.sn_noise_region_edit.setText(state.get("sn_noise_region", ""))
            self.sn_threshold_spin.setValue(state.get("sn_threshold", 3.0))
            self.analysis_results_label.setText(state["results_text"])
            self.integration_results_label.setText(state.get("integration_results_text", ""))

        self._slice_fit_results = spectrum.metadata.raw.get("slice_fit_results")
        # Only a real peak fit (popt/pcov/peak_defs/y_fit) can be shown
        # by _display_slice_fit -- a restored Integrate Regions result
        # (_show_integration_results, same storage slot) has none of
        # those, and enabling the spin box for it would crash the
        # moment it's touched.
        is_peak_fit = bool(self._slice_fit_results) and "popt" in self._slice_fit_results[0]
        self.view_slice_fit_spin.blockSignals(True)
        if is_peak_fit:
            self.view_slice_fit_spin.setEnabled(True)
            self.view_slice_fit_spin.setMaximum(max(1, len(self._slice_fit_results)))
            self.view_slice_fit_spin.setValue(1)
        else:
            self.view_slice_fit_spin.setEnabled(False)
        self.view_slice_fit_spin.blockSignals(False)
        self._sync_region_shading()
        self._refresh_peak_markers()

        self._relaxation_fit_results = spectrum.metadata.raw.get("relaxation_fit_results")
        self._relaxation_fit_xlabel = spectrum.metadata.raw.get("relaxation_fit_xlabel", "Delay (s)")
        self.view_relaxation_peak_spin.blockSignals(True)
        if self._relaxation_fit_results:
            self.view_relaxation_peak_spin.setEnabled(True)
            self.view_relaxation_peak_spin.setMaximum(max(1, len(self._relaxation_fit_results)))
            self.view_relaxation_peak_spin.setValue(1)
        else:
            self.view_relaxation_peak_spin.setEnabled(False)
        self.view_relaxation_peak_spin.blockSignals(False)

    # ============================================================
    # Selection handling (shared by both tabs)
    # ============================================================

    def _on_selection_changed(self, row: int) -> None:
        if row < 0 or row >= len(self.session):
            return
        self._save_current_analysis_state()
        # Flush any still-pending sidecar save for the *outgoing*
        # spectrum synchronously, rather than leaving it in the
        # debounce timer -- _schedule_sidecar_save only ever tracks
        # one pending index at a time (same limitation
        # _live_reprocess_timer already has), so scheduling one for
        # the *new* selection below would otherwise silently drop an
        # edit made to this spectrum just before switching away from
        # it, exactly the same "outgoing spectrum's own last edit"
        # concern _save_current_analysis_state right above exists for.
        if self._sidecar_save_timer.isActive():
            self._sidecar_save_timer.stop()
            self._run_debounced_sidecar_save()
        spectrum = self.session.spectra[row]
        # Refresh widget state (slice index range/default, etc.) before
        # drawing -- the FID-preview branch below reads e.g. slice_spin,
        # which must already reflect *this* spectrum, not whatever was
        # last selected.
        self._refresh_selection_display(row)
        self._draw_selected_spectrum(spectrum, row=row)
        self._current_analysis_index = row

    def _compare_shows_current_pseudo2d_stack(self, row: int, spectrum) -> bool:
        """True if the Compare tab's list is currently showing exactly
        ``spectrum``'s own auto-populated slice stack (see
        :meth:`_populate_compare_list_from_pseudo2d`) -- so a plain
        reselection of the same spectrum doesn't clobber it.

        Needed because ``spectrum_list`` fires *two* separate signals
        for a single-selection change (``itemSelectionChanged`` ->
        :meth:`_on_list_selection_changed`, and ``currentRowChanged``
        -> :meth:`_on_selection_changed` directly) -- confirmed
        empirically (not just assumed from the connect calls) to both
        actually fire for the exact "select a fresh pseudo-2D spectrum"
        gesture, with the direct one landing *after* whichever order
        the code runs its own population/redraw logic in. So this
        can't be fixed by reordering alone (tried first, didn't hold
        up under tracing) -- checking "is Compare already correctly
        showing this" is order-independent: no matter which handler
        fires last, it leaves an already-correct Compare view alone
        rather than redrawing over it."""
        if not (spectrum.ndim == 2 and spectrum.is_pseudo_2d):
            return False
        n = spectrum.data.shape[0]
        if self.compare_list.count() != n:
            return False
        return all(
            self.compare_list.item(i).data(Qt.ItemDataRole.UserRole) == (row, i + 1) for i in range(n)
        )

    def _draw_selected_spectrum(self, spectrum, row: int | None = None) -> None:
        """The actual draw step behind a selection change -- also used
        standalone (e.g. by the pseudo-2D view combo) when the
        selection itself hasn't changed and going through
        :func:`_refresh_selection_display` again would wrongly reset
        widgets (like that same combo) back to their per-selection
        defaults -- ``row`` isn't known in that case, hence optional
        (only needed for the Compare-stack check below, itself only
        relevant right after a real selection change)."""
        is_pseudo2d_stack = (
            spectrum.ndim == 2 and spectrum.is_pseudo_2d and self.pseudo2d_view_combo.currentIndex() == 1
        )
        # The FID tab is a live view of whatever's selected, not an
        # opt-in overlay (no more checkbox) -- if it's already the
        # active tab, a reselection keeps it active and refreshes it,
        # rather than force-switching away to the Spectrum tab. Same
        # idea for the Compare tab specifically when it's already
        # showing this exact pseudo-2D's own slice stack -- see
        # _compare_shows_current_pseudo2d_stack's own docstring for why
        # this check exists at all -- and for the Processing-preview
        # tab, which should be the default context for processing
        # steps (explicit user feedback) rather than something a
        # reselection/reprocess pulls focus away from.
        fid_tab_active = self.canvas_tabs.currentWidget() is self.fid_canvas
        processing_preview_active = self.canvas_tabs.currentWidget() is self.processing_preview_canvas
        # Two separate questions, not one: whether Compare already
        # correctly reflects this spectrum's own slices at all (used
        # below to decide whether _silo_other_canvas_tabs should leave
        # its content alone -- true regardless of which tab happens to
        # be showing, since Compare is meant to stay correctly drawn
        # in the background even while Spectrum is the visible one --
        # see _on_list_selection_changed's own "land on Spectrum, not
        # Compare" comment); and whether the Compare *tab* is already
        # the one actively showing (used to decide whether it's safe
        # to leave the canvas switch alone below, rather than
        # force-switching to Spectrum out from under someone who
        # deliberately navigated to Compare themselves, e.g. via a
        # manual multi-select).
        compare_matches_list = row is not None and self._compare_shows_current_pseudo2d_stack(row, spectrum)
        compare_tab_active_and_matches = (
            compare_matches_list and self.canvas_tabs.currentWidget() is self.compare_canvas
        )
        try:
            if is_pseudo2d_stack:
                self._show_pseudo2d_stack(spectrum)
            elif compare_tab_active_and_matches:
                pass
            else:
                # Kept up to date regardless of which canvas tab is
                # actually showing (see _apply_phase_preview's "1d"
                # branch for the same reasoning), so switching back to
                # Spectrum later never shows stale data -- only the tab
                # *switch* itself is guarded.
                self.canvas.show_spectrum(spectrum)
                if not fid_tab_active and not processing_preview_active:
                    self.canvas_tabs.setCurrentWidget(self.canvas)
            if fid_tab_active and self._fid_preview_available(spectrum):
                self._show_fid_preview(spectrum, switch_tab=False)
            self._refresh_processing_preview_if_active()
        except Exception as exc:  # noqa: BLE001 -- surfaced to the user, not swallowed
            QMessageBox.critical(self, "Failed to plot spectrum", str(exc))

        # Re-sync region shading/peak markers *after* whatever redraw
        # just happened above -- both are also called from
        # _restore_analysis_state (needed for a canvas-preserving
        # refresh, e.g. a nucleus/mass override that doesn't redraw at
        # all), but _restore_analysis_state itself runs *before*
        # _draw_selected_spectrum in _on_selection_changed's own call
        # order, so that earlier call silently gets discarded whenever
        # self.canvas.show_spectrum/_show_pseudo2d_stack just rebuilt
        # the figure (a real, confirmed-by-reproduction bug for region
        # shading specifically -- reselecting back to a spectrum with
        # a saved region lost its shading; peak markers had the same
        # gap plus never being drawn for a manually added peak at
        # all). Both calls are cheap and safely no-op via their own
        # canvas-side guards when there's nothing sensible to draw on
        # (a stack view, Compare-tab-active skip, etc.).
        self._sync_region_shading()
        self._refresh_peak_markers()

        self._silo_other_canvas_tabs(spectrum, skip_compare=compare_matches_list)

    def _silo_other_canvas_tabs(self, spectrum, skip_compare: bool = False) -> None:
        """Keep the Fit/Spectral Fit/Compare canvas tabs from showing a
        previously-selected spectrum's stale plot -- they aren't
        redrawn by the Spectrum/FID branches above, so without this a
        reselection would leave e.g. the Fit tab showing another
        experiment's relaxation curve. ``skip_compare`` -- see
        :meth:`_draw_selected_spectrum` -- leaves the Compare canvas
        alone when it's already correctly showing the just-selected
        pseudo-2D's own slice stack, instead of clearing it out from
        under itself."""
        fit_results = spectrum.metadata.raw.get("relaxation_fit_results")
        if fit_results:
            # matches view_relaxation_peak_spin resetting to 1 on reselect
            self._relaxation_fit_results = fit_results
            self._relaxation_fit_xlabel = spectrum.metadata.raw.get("relaxation_fit_xlabel", "Delay (s)")
            self._display_relaxation_peak(0)
        else:
            self.fit_canvas.clear()

        slice_fit_results = spectrum.metadata.raw.get("slice_fit_results")
        if slice_fit_results and "popt" in slice_fit_results[0]:
            # matches view_slice_fit_spin resetting to 1 on reselect --
            # guarded on shape since slice_fit_results can also hold an
            # Integrate Regions result (_show_integration_results, same
            # storage slot), which _display_slice_fit can't render.
            self._display_slice_fit(slice_fit_results[0], switch_tab=False)
        else:
            self.spectral_fit_canvas.clear()
            self.spectral_fit_params_label.setText("")

        # The Compare tab plots whatever is multi-selected in the
        # Spectra list, not this single spectrum -- it has no
        # per-spectrum state to restore, so just drop the stale plot
        # rather than show a comparison involving a spectrum that may
        # no longer even be part of the current selection. Unless
        # skip_compare says the Compare tab is actually already
        # showing exactly this spectrum's own auto-populated slices --
        # then it very much is per-spectrum state worth keeping.
        if not skip_compare:
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
            bits.append(f"Sample mass: {meta.mass * 1000.0:g} mg")
        if meta.probe:
            bits.append(f"Probe: {meta.probe}")
        self.info_label.setText("\n".join(bits))

        self.imaging_max_gradient_spin.blockSignals(True)
        self.imaging_max_gradient_spin.setValue(meta.raw.get("imaging_max_gradient_g_per_cm") or 0.0)
        self.imaging_max_gradient_spin.blockSignals(False)
        self._refresh_imaging_gradient()

        self.process_indirect_button.setVisible(spectrum.ndim == 2)
        self.process_indirect_button.blockSignals(True)
        self.process_indirect_button.setChecked(not spectrum.is_pseudo_2d)
        self.process_indirect_button.blockSignals(False)

        is_pseudo2d_now = spectrum.ndim == 2 and spectrum.is_pseudo_2d
        self.pseudo2d_view_combo.setVisible(is_pseudo2d_now)
        self.pseudo2d_view_combo.blockSignals(True)
        # Contour, not Stacked slices -- explicit user feedback: the
        # Spectrum tab is also where per-slice/whole-dataset phasing
        # happens (see the Phase group), which needs a plain single-
        # slice/contour view, not a multi-axes Compare-style stack;
        # "Stacked slices" stays available from the combo, just isn't
        # the default any more. Runs unconditionally on every
        # reselection (like the phase controls resetting below), so
        # it overrides a Stacked choice left over from a previous
        # spectrum rather than carrying it forward.
        self.pseudo2d_view_combo.setCurrentIndex(0)
        self.pseudo2d_view_combo.blockSignals(False)
        self._update_fit_tab_visibility(spectrum)
        self._update_processing_preview_tab_visibility(spectrum)

        self._refresh_fid_processing_controls(spectrum, row)
        self._restore_sr_control(spectrum)
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
        self.slice_prev_button.setVisible(is_pseudo2d)
        self.slice_next_button.setVisible(is_pseudo2d)
        self.apply_all_slices_checkbox.setVisible(is_pseudo2d)
        self.autophase_slices_button.setVisible(is_pseudo2d)
        self.dim_row_label.setVisible(is_2d)
        self.dim_combo.setVisible(is_2d)

        if is_pseudo2d:
            self.slice_spin.blockSignals(True)
            self.slice_spin.setMaximum(max(1, spectrum.data.shape[0]))
            self.slice_spin.setValue(1)
            self.slice_spin.blockSignals(False)
            self.apply_all_slices_checkbox.blockSignals(True)
            # Checked by default -- explicit user feedback: a shared
            # phase/baseline correction across the whole series is the
            # common case (and switching slices with it unchecked looks
            # like "processing resets" every time, since each slice
            # then keeps its own, usually-untouched phase). Per-slice
            # is still available by unchecking.
            self.apply_all_slices_checkbox.setChecked(True)
            self.apply_all_slices_checkbox.blockSignals(False)
        if is_2d:
            self.dim_combo.blockSignals(True)
            self.dim_combo.setCurrentIndex(0)
            self.dim_combo.blockSignals(False)

        self._capture_phase_reference()
