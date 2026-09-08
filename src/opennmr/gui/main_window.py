"""Main window: a tabbed side panel (spectrum management + live
processing controls) plus an embedded plotting canvas.

The side panel is a persistent, always-visible :class:`QTabWidget`
rather than transient popup dialogs -- opening a file is still a
(necessary) native file dialog, but everything else (list management,
nucleus override, phase adjustment) lives in the docked tabs and
updates the canvas live.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import os
import re
import time
from io import StringIO

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
        # 1-based row (dim="f2")/column (dim="f1") index of the
        # representative 1D slice shown while phasing a genuine 2D --
        # see _capture_phase_reference's own "2d" branch for how it's
        # picked, and _apply_phase_preview's "2d" branch for where it's
        # used. Only meaningful when _phase_mode == "2d".
        self._phase_2d_slice_index: int | None = None
        # Which session index _phase_mode/_phase_reference_full above
        # actually belong to -- needed by _commit_pending_phase_to_metadata,
        # called at the *start* of the next _capture_phase_reference,
        # by which point self._selected_index() already points at the
        # spectrum being switched *to*, not the one whose pending phase
        # delta needs committing.
        self._phase_target_index: int | None = None
        # Same reasoning as _phase_target_index just above, for a
        # genuine-2D session's own F2-vs-F1 target specifically (only
        # meaningful when _phase_mode == "2d") -- real bug this fixes:
        # dim_combo.currentIndexChanged fires *after* Qt has already
        # updated dim_combo.currentIndex() to the new value, so reading
        # it live inside _commit_pending_phase_to_metadata (itself
        # called from _capture_phase_reference, which that same signal
        # triggers via _on_phase_target_changed) would silently commit
        # a pending F2 delta under the F1 key (or vice versa) whenever
        # switching dimensions mid-edit -- confirmed directly against
        # real 2D data. Set once, in _capture_phase_reference's own
        # "2d" branch, at the point a session's target dimension is
        # actually established -- not re-derived from the (by then
        # possibly already-changed) live combo box.
        self._phase_target_dim: str | None = None
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
        # Same debounce reasoning as _live_reprocess_timer above, for
        # self.canvas's own 2D/pseudo-2D contour during a live phase
        # drag on a genuine 2D or "apply to every slice" pseudo-2D --
        # real user requirement, repeated across several rounds: "the
        # Spectrum tab should always show the 2D or pseudo 2D where
        # there is one," never switch to a single-slice line the way
        # phase-editing used to make it. A full contour redraw is
        # measured at ~0.35s on a real, large conventional-2D array
        # (see plotting.style.downsample_2d's own docstring) -- fine
        # settled, but redrawing on literally every mouse-move tick of
        # a drag would make the drag itself feel frozen. Debounced the
        # same way the multi-row recompute already is: the phase
        # correction itself is still applied to spectrum.data on every
        # tick (cheap, a plain array op), only the expensive on-screen
        # contour redraw waits for a short quiet period.
        self._canvas_2d_refresh_timer = QTimer(self)
        self._canvas_2d_refresh_timer.setSingleShot(True)
        self._canvas_2d_refresh_timer.timeout.connect(self._run_debounced_canvas_2d_refresh)
        self._canvas_2d_refresh_pending_index: int | None = None
        # How long a contour rebuild actually took, keyed by the
        # spectrum's own data shape -- what
        # _schedule_canvas_2d_refresh uses to decide between live
        # throttling and pure debounce (see its docstring). Keyed by
        # shape rather than session index because the cost is a
        # property of the data size, and a shape key stays valid across
        # the reloads and index shifts that replace a spectrum object.
        self._contour_rebuild_seconds: dict[tuple, float] = {}
        self._last_contour_rebuild_at: float = 0.0
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
        self.canvas.contourSlicesChanged.connect(self._on_contour_slices_changed)
        self.canvas.referenceCalibrationRequested.connect(self._on_reference_calibration_from_plot)
        self.canvas.sliceKeepRequested.connect(self._on_slice_keep_requested)
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

        # Every canvas, not just the Spectrum one: a label falling off
        # the figure is worth saying so wherever it happens, and each
        # canvas lays itself out independently. The handler names which
        # tab it came from, since only one is visible at a time.
        for name, canvas in (
            ("Spectrum", self.canvas), ("FID", self.fid_canvas),
            ("Compare", self.compare_canvas), ("Fit", self.fit_canvas),
            ("Spectral Fit", self.spectral_fit_canvas),
            ("Processing Preview", self.processing_preview_canvas),
        ):
            canvas.labelsClipped.connect(
                lambda labels, tab=name: self._on_labels_clipped(tab, labels)
            )

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

        # Navigation-mode shortcuts. matplotlib's own toolbar owns the
        # pan/zoom state, so these drive it rather than duplicating it --
        # the toolbar buttons stay in sync because they *are* the state.
        view_menu = menu_bar.addMenu("&View")
        pan_action = QAction("&Pan Mode", self)
        pan_action.setShortcut(QKeySequence("Ctrl+Shift+P"))
        pan_action.setToolTip("Drag to pan the spectrum. On by default when data is loaded.")
        pan_action.triggered.connect(lambda: self._set_navigation_mode("pan"))
        view_menu.addAction(pan_action)

        zoom_action = QAction("&Zoom Mode", self)
        zoom_action.setShortcut(QKeySequence("Ctrl+Shift+Z"))
        zoom_action.setToolTip("Drag a rectangle to zoom into it.")
        zoom_action.triggered.connect(lambda: self._set_navigation_mode("zoom"))
        view_menu.addAction(zoom_action)

        toggle_nav_action = QAction("&Toggle Pan/Zoom", self)
        # Ctrl+Z (Cmd+Z on macOS) as explicitly requested. There is no
        # undo stack in this application for it to collide with.
        toggle_nav_action.setShortcut(QKeySequence("Ctrl+Z"))
        toggle_nav_action.setToolTip("Switch between pan and zoom.")
        toggle_nav_action.triggered.connect(self._toggle_navigation_mode)
        view_menu.addAction(toggle_nav_action)

        view_menu.addSeparator()
        home_action = QAction("Reset &View", self)
        home_action.setShortcut(QKeySequence("Ctrl+0"))
        home_action.setToolTip("Reset pan/zoom back to the full spectrum.")
        home_action.triggered.connect(self._reset_navigation_view)
        view_menu.addAction(home_action)

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

    def _on_slice_keep_requested(self, payload) -> None:
        """Add an extracted slice to the session as its own 1D
        spectrum, and select it.

        Real user request: "a method to phase on an extracted slice".
        ``MplCanvas.extract_slice`` only ever *displayed* a trace -- it
        never reached the session, and every phase control acts on
        session spectra, so an extracted slice could be looked at but
        not worked on. Adding it makes phasing, fitting and Compare all
        apply to it through their existing paths, with no parallel
        machinery.

        Registered via :meth:`Session.add_derived`, which marks the
        vendor ``"derived"`` -- correct here: the slice has no raw FID
        of its own to reprocess from, so the FID-processing controls
        stay hidden for it (see ``_fid_reprocessable``) while the phase
        controls, which need only retained imaginary data, work
        normally. ``processing.slicing.get_trace`` carries ``data_imag``
        through, which is what makes that true -- and when it is absent
        (a magnitude-mode or processed-only source), the slice is still
        kept, but the user is told why phasing will be unavailable
        rather than left to discover a greyed-out control.
        """
        trace, label, parent, axis, slice_index = payload
        if trace is None:
            return
        if trace.data_imag is None:
            QMessageBox.information(
                self, "Slice kept without phase data",
                f"{label} has no retained imaginary data, so it can be fitted and compared "
                f"but not phased. That happens when the spectrum it came from was itself "
                f"loaded without an imaginary part (a magnitude-mode or processed-only load).",
            )
        index = self.session.add_derived(trace, label)
        kept = self.session.spectra[index]
        # Provenance, so the slice is a *working copy* of part of the 2D
        # rather than a dead end: "Apply Phase to Source 2D" uses these
        # to push a phase dialled in here back into the parent. Held as
        # a live object reference (not a session index, which shifts
        # when spectra are removed or reordered) and blocklisted from
        # the sidecar, since neither a spectrum object nor a
        # within-session relationship is meaningful once reloaded.
        kept.metadata.raw["slice_parent"] = parent
        kept.metadata.raw["slice_axis"] = axis
        kept.metadata.raw["slice_index"] = slice_index
        self.spectrum_list.addItem(self.session.label(index))
        self._select_single_row(index)
        self.statusBar().showMessage(
            f"Kept {label}. Phase it, then use \u201cApply Phase to Source 2D\u201d "
            f"to push that phase back into the 2D.", 8000,
        )

    def _on_labels_clipped(self, tab: str, labels) -> None:
        """Say so when a canvas cannot fit its own axis labels.

        The layout pass already shrinks the text as far as it usefully
        can (see ``MplCanvas._apply_layout``); reaching here means even
        the floor was not enough, which only the user can resolve -- a
        taller/wider window, or a narrower side panel. Silence would
        just leave a figure rendering with its labels off the edge,
        which is the thing being guarded against.

        An empty ``labels`` is the all-clear, emitted when a previously
        clipped canvas fits again -- clear the message rather than
        leaving a stale warning on screen."""
        if not labels:
            self.statusBar().clearMessage()
            return
        names = ", ".join(str(text) for text in labels[:3])
        if len(labels) > 3:
            names += ", ..."
        self.statusBar().showMessage(
            f"{tab}: not enough room for the axis labels ({names}) even at the "
            f"smallest font — make the window taller/wider, or narrow the side panel.",
            8000,
        )

    def _slice_source_of(self, spectrum):
        """``(parent, dim)`` for an extracted slice that still has a
        live link back to its 2D, else ``(None, None)``.

        ``dim`` is the parent dimension the slice's own phase belongs
        to, which is the *opposite* of the axis that was held fixed:
        extracting with ``axis="f1"`` fixes F1 and traces **along F2**,
        so phasing that trace is an F2 phase. Getting this backwards
        would silently phase the wrong dimension, so it is derived in
        one place rather than at each call site.
        """
        if spectrum is None:
            return None, None
        parent = spectrum.metadata.raw.get("slice_parent")
        axis = spectrum.metadata.raw.get("slice_axis")
        if parent is None or axis not in ("f1", "f2"):
            return None, None
        # Identity, not `in`/`.index()`: those compare with `==`, and
        # NMRSpectrum is a dataclass holding numpy arrays, so equality
        # returns an array and raises "truth value of an array with
        # more than one element is ambiguous". Caught by the test for
        # the removed-parent case.
        if not any(candidate is parent for candidate in self.session.spectra):
            # The 2D was removed from the session since; nothing to
            # apply to, and holding a reference to it would be worse
            # than saying so.
            return None, None
        return parent, ("f2" if axis == "f1" else "f1")

    def _on_apply_slice_phase_to_source(self) -> None:
        """Push the phase dialled in on an extracted slice back into
        the 2D it came from.

        Real user request: "I don't think the derived slice phasing is
        feeding back into the contour, it should be more like a
        temporary file and have some way to go back into the contour
        data." Without this the slice was a dead end -- you could phase
        it, but the result went nowhere.

        Deliberately adds into the parent's own *user* phase keys and
        then rebuilds it through :meth:`_recompute_multi_row_data`,
        rather than mutating the parent's data directly. That reuses
        the one pipeline everything else goes through, so the result is
        identical to having dialled the same phase on the 2D itself --
        and it survives the next rebuild, which a direct data mutation
        would not.

        Only the phase *added since extraction* is transferred: the
        slice inherited the parent's recorded phase when it was cut, so
        transferring that too would apply it twice.
        """
        index = self._selected_index()
        if index is None:
            return
        slice_spectrum = self.session.spectra[index]
        parent, dim = self._slice_source_of(slice_spectrum)
        if parent is None:
            QMessageBox.information(
                self, "No source 2D",
                "This spectrum isn't an extracted slice, or the 2D it came from is no "
                "longer loaded.",
            )
            return

        self._commit_pending_phase_to_metadata()
        raw = slice_spectrum.metadata.raw
        phase0 = float(raw.get("phase0", 0.0) or 0.0)
        phase1 = float(raw.get("phase1", 0.0) or 0.0)
        if not phase0 and not phase1:
            QMessageBox.information(
                self, "No phase to apply",
                "No phase has been dialled in on this slice yet, so there is nothing to "
                "apply to the 2D.",
            )
            return

        key0, key1 = io_bruker.USER_PHASE_KEYS[dim]
        parent_raw = parent.metadata.raw
        parent_raw[key0] = float(parent_raw.get(key0, 0.0) or 0.0) + phase0
        parent_raw[key1] = float(parent_raw.get(key1, 0.0) or 0.0) + phase1
        # Cleared on the slice so a second click can't apply the same
        # phase twice -- the slice's own displayed data keeps it (it is
        # already baked in), which is what makes the two agree.
        raw["phase0"] = 0.0
        raw["phase1"] = 0.0
        # The live spin boxes have to be zeroed too, and *before*
        # selecting the parent: _capture_phase_reference commits any
        # pending phase into the outgoing target on the way out, which
        # would write the same value straight back onto the slice and
        # undo the clearing above. Found by testing the round trip
        # rather than by reading the code.
        for spin, handle in ((self.phase0_spin, self.phase0_handle),
                             (self.phase1_spin, self.phase1_handle)):
            spin.blockSignals(True)
            spin.setValue(0.0)
            spin.blockSignals(False)
            handle.setValue(0.0, emit=False)

        # Identity search, for the same reason as in
        # _slice_source_of -- list.index() compares with `==`.
        parent_index = next(
            i for i, candidate in enumerate(self.session.spectra) if candidate is parent
        )
        self._select_single_row(parent_index)
        if not self._recompute_from_raw_if_possible(parent):
            self._draw_selected_spectrum(parent, row=parent_index)
        self.statusBar().showMessage(
            f"Applied phase0 {phase0:+.2f}\u00b0, phase1 {phase1:+.2f}\u00b0 to the "
            f"2D's {dim.upper()} dimension.", 8000,
        )

    def _set_navigation_mode(self, mode: str) -> None:
        """Put **every** canvas into ``"pan"`` or ``"zoom"``.

        Applied to all of them, not just the visible one: the mode is a
        per-toolbar flag, so leaving the others alone would mean
        switching canvas tab silently changed the interaction mode back.

        matplotlib's own toolbar owns this state and its ``pan()``/
        ``zoom()`` methods *toggle*, so each is called only when the
        toolbar is not already in the requested mode -- calling
        unconditionally would turn it back off.
        """
        for canvas in self._all_canvases():
            toolbar = getattr(canvas, "toolbar", None)
            if toolbar is None:
                continue
            current = toolbar.mode.name if hasattr(toolbar.mode, "name") else str(toolbar.mode)
            already = (mode == "pan" and "PAN" in current.upper()) or (
                mode == "zoom" and "ZOOM" in current.upper()
            )
            if already:
                continue
            if mode == "pan":
                toolbar.pan()
            else:
                toolbar.zoom()

    def _navigation_mode(self) -> str:
        """``"pan"``, ``"zoom"`` or ``""`` for the visible canvas."""
        current = self.canvas_tabs.currentWidget()
        if current is None or not hasattr(current, "toolbar"):
            return ""
        mode = current.toolbar.mode
        text = (mode.name if hasattr(mode, "name") else str(mode)).upper()
        if "PAN" in text:
            return "pan"
        if "ZOOM" in text:
            return "zoom"
        return ""

    def _toggle_navigation_mode(self) -> None:
        """Ctrl/Cmd+Z -- swap between pan and zoom. From neither mode
        (both toggled off via the toolbar), goes to pan, matching the
        default the app starts in."""
        self._set_navigation_mode("zoom" if self._navigation_mode() == "pan" else "pan")

    def _reset_navigation_view(self) -> None:
        """Reset the visible canvas's pan/zoom back to the full view --
        via :meth:`MplCanvas.reset_view`, which restores the *layout*
        as well as the limits (``toolbar.home()`` alone leaves a
        clipped axis label behind; see that method)."""
        current = self.canvas_tabs.currentWidget()
        if current is None:
            return
        if hasattr(current, "reset_view"):
            current.reset_view()
        elif hasattr(current, "toolbar"):
            current.toolbar.home()

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

        # One visible button (real user request: "there should just be
        # one button that can open files or folders"), but internally
        # still two separate *native* dialogs -- a single non-native
        # QFileDialog that tried to let one dialog session pick either
        # (an earlier version of this) turned out unreliable in
        # practice: real user report, its own restrictive default name
        # filter hid every extensionless Bruker/Varian folder and raw
        # fid/ser file, "basically only works for jeol", and even with
        # that fixed, folder selection itself "doesn't allow for just a
        # folder to be selected any more" -- the community-documented
        # "single click selects a folder too" behaviour for a
        # non-native ExistingFiles dialog didn't hold up on real usage.
        # A QToolButton with MenuButtonPopup keeps one visible button --
        # a plain click goes straight to the proven, reliable folder
        # picker (the common case, explicitly the more useful one per
        # that same feedback); the small dropdown arrow offers "Open
        # File(s)…" for the individual-file vendors (X-Pulse, JEOL)
        # via the other proven native dialog.
        self.open_button = QToolButton()
        self.open_button.setText("Open…")
        self.open_button.setToolTip(
            "Pick a folder -- either one experiment itself, or a parent "
            "folder holding several. If more than one spectrum is found "
            "inside, you'll get to choose which to open. Use the "
            "dropdown arrow to open individual files instead (e.g. "
            "X-Pulse, JEOL)."
        )
        self.open_button.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)
        self.open_button.clicked.connect(self._on_open_clicked)
        open_button_menu = QMenu(self.open_button)
        open_files_action_button = open_button_menu.addAction("Open File(s)…")
        open_files_action_button.triggered.connect(self._on_open_files_clicked)
        self.open_button.setMenu(open_button_menu)
        layout.addWidget(self.open_button)

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

        self.magnitude_button = QPushButton("Magnitude Mode")
        self.magnitude_button.setCheckable(True)
        self.magnitude_button.setToolTip(
            "Display the magnitude (absolute value) of the spectrum "
            "instead of its phase-sensitive real part -- |real + i*imag|, "
            "so every peak is positive and no phasing is needed (or "
            "possible: the phase controls switch off while this is on).\n\n"
            "Switched on automatically for any dataset whose own "
            "processing parameters ask for it (Bruker PH_mod = mc), "
            "which is common for gradient-COSY experiments. Turn it on "
            "yourself for anything that is hard to phase, or off to see "
            "the phase-sensitive spectrum a file recorded as magnitude. "
            "Reprocesses from the original file immediately."
        )
        self.magnitude_button.toggled.connect(self._on_magnitude_toggled)
        info_form.addRow(self.magnitude_button)

        self.pseudo2d_view_combo = QComboBox()
        self.pseudo2d_view_combo.addItems(["Contour", "Stacked slices", "Heatmap"])
        self.pseudo2d_view_combo.setToolTip(
            "How to display a pseudo-2D series: as a contour (F2 vs. "
            "slice number), every slice plotted as an offset 1D stack, "
            "or a heatmap (F2 vs. slice number, colour = intensity)."
        )
        self.pseudo2d_view_combo.currentIndexChanged.connect(self._on_pseudo2d_view_changed)
        info_form.addRow("View:", self.pseudo2d_view_combo)

        layout.addWidget(info_box)
        layout.addStretch(1)
        return tab

    def _on_open_clicked(self) -> None:
        """The Open… button's own plain-click action -- the reliable
        *folder-only* picker, same file-mode semantics this project
        always used before the (reverted) single-non-native-dialog
        attempt, but styled with Qt's own cross-platform dialog widget
        instead of the OS-native panel. See the button's own
        construction comment in :meth:`_build_spectra_tab` for why the
        original non-native attempt was reverted: real user reports
        that a non-native dialog in ``ExistingFiles`` mode (offering
        *files and folders together*) didn't actually let a folder be
        *selected* (only navigated into) in practice. This is
        deliberately narrower -- ``Directory`` mode only, matching
        :func:`QFileDialog.getExistingDirectory`'s own single-purpose
        semantics exactly, just with ``DontUseNativeDialog`` added --
        so the specific file-vs-folder click ambiguity that broke the
        earlier attempt doesn't apply here at all. Still, this couldn't
        be interactively verified in this headless dev environment (see
        this project's own standing lesson about that); worth a quick
        real check that folder selection still feels right. The
        restrictive-default-filter half of the original revert doesn't
        apply here either -- a plain directory picker has no name
        filter to restrict in the first place."""
        directory = QFileDialog.getExistingDirectory(
            self, "Open a data folder -- an experiment folder itself, or a "
                  "parent folder containing several",
            options=QFileDialog.Option.DontUseNativeDialog,
        )
        if not directory:
            return
        self._open_directory(directory)

    def _on_open_files_clicked(self) -> None:
        """The Open… button's dropdown action -- picks one or more
        individual spectrum files (X-Pulse, JEOL), styled the same way
        as :meth:`_on_open_clicked` above (Qt's own dialog widget, not
        the OS-native one) and for the same reason. "All files" is
        still the *default* selected filter (not the NMR-specific one)
        so a vendor's own extensionless raw files stay visible/
        selectable without the user needing to switch filters by hand
        -- the restrictive-filter half of the original revert, kept
        fixed regardless of dialog style."""
        paths, _selected_filter = QFileDialog.getOpenFileNames(
            self, "Open one or more spectrum files",
            filter="All files (*);;NMR spectrum files (*.jdx *.dx *.jdf)",
            options=QFileDialog.Option.DontUseNativeDialog,
        )
        if not paths:
            return
        self._open_paths(paths)

    def _open_paths(self, paths: list[str], report_unrecognised: bool = True) -> None:
        """Load each of ``paths`` -- a directory routes through the
        folder scan (:meth:`_open_directory`), a file loads directly
        via vendor auto-detection. ``report_unrecognised`` (default
        ``True``, a deliberate dialog pick) shows a message box listing
        anything not recognised; ``False`` (used by
        :meth:`_handle_dropped_paths`) silently skips it instead, since
        a drop can easily carry along OS clutter (e.g. ``.DS_Store``)
        that was never meant to be opened."""
        unrecognised: list[str] = []
        for path in paths:
            if os.path.isdir(path):
                self._open_directory(path)
                continue
            try:
                vendor = io.detect_vendor(path)
            except ValueError:
                unrecognised.append(path)
                continue
            self._load(path, vendor)

        if unrecognised and report_unrecognised:
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
        scan/prompt/load path (:meth:`_open_paths`, same routing the
        Open… dialog itself now uses). A dropped single file that's
        itself a recognised spectrum (e.g. one X-Pulse/Varian file, not
        inside a folder) loads directly; anything else unrecognised is
        silently skipped rather than reported, since a drop can easily
        include the OS's own clutter (a `.DS_Store`, etc.) alongside
        the folder actually meant. Split out from ``dropEvent`` itself
        so this is testable without constructing a real ``QDropEvent``."""
        self._open_paths([p for p in paths if os.path.isdir(p) or os.path.isfile(p)], report_unrecognised=False)

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
        # Decide the pseudo-2D treatment *before* the read, not after.
        # Every vendor reader sets metadata.source_path to exactly the
        # path handed to it, so this experiment's own saved preference
        # is readable straight from `path` -- which means a genuine 2D
        # can be read once with the right flag instead of being read in
        # full, inspected, and then read a second time (traced on the
        # real EXSY benchmark: a 5.19s genuine-2D reconstruction, F1 FFT
        # and all, computed and then discarded, before the 2.83s
        # pseudo-2D read whose result is what actually survives -- see
        # TODO.md's duplication-trace entry).
        #
        # Bruker only, deliberately: io.xpulse.read_2d has no pseudo_2d
        # parameter at all (see _refresh_selection_display's own guard),
        # and the other vendors keep the read-then-reload path below
        # unchanged rather than being switched on an assumption about
        # readers this trace never measured. io.bruker._read_auto drops
        # `pseudo_2d` for the 1D branches, so passing it here is safe
        # without knowing yet whether `path` is 1D or 2D.
        add_kwargs = {}
        if vendor == "bruker":
            presel = io_persistence.load_metadata_sidecar(path)
            add_kwargs["pseudo_2d"] = (presel or {}).get("pseudo_2d") is not False
            # Magnitude mode has to be decided here too, for the same
            # reason: it is applied by the *reader* (there is
            # deliberately no second in-memory implementation, see
            # _on_magnitude_toggled), so the sidecar-restore path --
            # which recomputes from the cached raw FID rather than
            # re-reading -- cannot apply it afterwards. Only an
            # explicit saved choice is honoured; with none, the reader
            # follows the file's own recorded PH_mod as usual.
            saved_processing = io_persistence.load_processing_sidecar(path) or {}
            if "magnitude_override" in saved_processing:
                add_kwargs["magnitude"] = bool(saved_processing["magnitude_override"])
        try:
            spectrum = self.session.add(path, vendor=vendor, **add_kwargs)
        except Exception as exc:  # noqa: BLE001 -- surfaced to the user, not swallowed
            QMessageBox.critical(self, "Failed to load spectrum", str(exc))
            return

        index = len(self.session) - 1
        # Loaded early (before the pseudo-2D default below) so a
        # previously-saved "Process Indirect Dimension" opt-in can
        # override that default rather than being silently reset by it
        # on every reopen -- see the pseudo_2d branch just below.
        saved_metadata = io_persistence.load_metadata_sidecar(spectrum.metadata.source_path)
        # X-Pulse's own read_2d has no pseudo_2d parameter at all --
        # every genuine 2D load from it is unconditionally the real
        # thing, so there is nothing to opt out of here (see
        # _refresh_selection_display's own matching guard on the
        # "Process Indirect Dimension" button for the full story). The
        # reload attempt below would always raise TypeError for it --
        # harmlessly swallowed by the except below today, but skipping
        # it outright avoids a guaranteed-to-fail call on every X-Pulse
        # 2D load.
        # Skipped entirely when the pre-read decision above already
        # applied this experiment's own preference (Bruker): the read
        # has already honoured it, so re-reading could only ever
        # reproduce the same result. Leaving this branch reachable in
        # that case was a real bug in the first draft of this fix,
        # caught by its own regression test rather than by inspection:
        # for a *saved genuine-2D* preference (pseudo_2d False), the
        # freshly-loaded spectrum is correctly not pseudo-2D, which
        # satisfied `not spectrum.is_pseudo_2d` and triggered a second,
        # byte-for-byte identical read -- reintroducing exactly the
        # duplicate this change exists to remove, for the one case that
        # doesn't take the common path.
        #
        # Still the live path for every other vendor, which deliberately
        # keeps the original read-then-reload behaviour (see above).
        if (
            "pseudo_2d" not in add_kwargs
            and spectrum.ndim == 2
            and not spectrum.is_pseudo_2d
            and spectrum.metadata.vendor != "xpulse"
        ):
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
        # Pan is the default interaction once there is data to pan --
        # explicit user request. Applied on load rather than at startup
        # so it lands on canvases that actually have something plotted,
        # and only for the first spectrum, so it never overrides a mode
        # the user has since chosen for themselves.
        if len(self.session) == 1:
            self._set_navigation_mode("pan")
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
                f"{self.session.label(index)}:\n\n{shape_note}\n\nCouldn't determine a "
                f"trustworthy shape for the raw ser file, so this loaded from TopSpin's own "
                f"already-processed data instead -- raw-FID reprocessing (window, zero-fill, "
                f"phase, ...) isn't available for it.",
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
        # Both dimensions -- a genuine 2D can carry an SR on each, and
        # reapplying only the direct one after a reprocess would
        # silently drop the indirect axis's referencing.
        applied = False
        for dim in ("f2", "f1"):
            sr_shift = spectrum.metadata.raw.get(self._SR_KEYS[dim][0])
            # Only a *user* value needs reapplying -- a fresh read has
            # already folded the file's own SR into its new axis.
            if sr_shift is not None and sr_shift != self._recorded_sr(spectrum, dim):
                self._apply_reference_shift_hz(sr_shift, dim)  # also redraws
                applied = True
        if not applied:
            self._draw_selected_spectrum(spectrum, row=index)

    def _select_single_row(self, row: int) -> None:
        """Select exactly ``row``, clearing any other selection, and
        refresh the display for it exactly once.

        Under ``ExtendedSelection`` mode, ``setCurrentRow`` alone moves
        keyboard/current focus *without* deselecting previously
        selected rows (by design, so ctrl/shift-click gestures compose)
        -- which would otherwise leave a stale multi-selection (and the
        Compare view still showing it) after loading a new spectrum or
        reordering one.

        The three list mutations below are done with the widget's own
        signals blocked, followed by one explicit
        :meth:`_on_list_selection_changed` call, because otherwise this
        method redraws the whole spectrum **twice**. Two of this
        widget's signals are connected to handlers that both end in a
        redraw (``currentRowChanged -> _on_selection_changed`` and
        ``itemSelectionChanged -> _on_list_selection_changed``, which
        itself calls ``_on_selection_changed``), and ``setCurrentRow``
        fires both -- traced directly on the real EXSY benchmark
        dataset, where each redraw is ~4.2s, so the duplicate was ~24%
        of a ~17.4s load (see TODO.md's duplication-trace entry).

        Deliberately fixed *here*, at the one call site that emits both
        signals for a single programmatic selection, rather than by
        removing ``_on_list_selection_changed``'s own explicit
        ``_on_selection_changed`` call -- that call is load-bearing for
        the case its own comment describes (a multi- to single-selection
        gesture doesn't necessarily change ``currentRow()``, so
        ``currentRowChanged`` genuinely doesn't fire on its own there),
        and every user-driven selection change still goes through the
        normal unblocked signal path unchanged.
        """
        self.spectrum_list.blockSignals(True)
        try:
            self.spectrum_list.clearSelection()
            self.spectrum_list.setCurrentRow(row)
            if 0 <= row < self.spectrum_list.count():
                self.spectrum_list.item(row).setSelected(True)
        finally:
            # finally, so a raised exception can't leave the list
            # permanently signal-deaf for the rest of the session.
            self.spectrum_list.blockSignals(False)
        # Exactly one dispatch, standing in for the (identical, but
        # duplicated) work the two blocked signals would have done.
        # _on_list_selection_changed is the right entry point rather
        # than _on_selection_changed directly: it dispatches to the
        # single- vs multi-selection branch itself, and its single-
        # selection branch already calls _on_selection_changed plus the
        # pseudo-2D Compare-population step that follows it.
        self._on_list_selection_changed()

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
        self._phase_target_dim = None
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
        index/slice dimension with no FFT.

        Commits any pending (not-yet-committed) phase delta *before*
        touching anything else -- real bug this fixes: without this,
        an uncommitted live delta sitting in the Phase spin boxes at
        the moment this button is clicked (e.g. mid-drag, released but
        not yet re-anchored by reselecting/switching dimension) would
        be silently dropped by the reload below, since neither
        :meth:`_current_phase_kwargs` nor :meth:`_reapply_phase_post_reload`
        reads the live spin boxes any more -- both now read only
        already-committed ``metadata.raw`` (see their own docstrings
        for why: whichever dimension *wasn't* currently focused used
        to lose its phase on every toggle regardless).

        F1's own ``"phase0_f1"/"phase1_f1"`` are carried over into the
        freshly-reloaded spectrum's own ``metadata.raw`` unconditionally,
        *before* :meth:`_reapply_phase_post_reload` runs -- real bug
        this fixes: ``read_2d``/``read_1d`` always build a brand new
        ``metadata.raw`` from scratch on every reload, with no F1-phase
        concept of its own (see :meth:`_current_phase_kwargs`'s own
        docstring) -- so without this, F1's phase survived exactly one
        toggle (baked into ``.data``, but never recorded anywhere) and
        was then gone for good, silently, on the next one: a genuine
        2D -> pseudo-2D -> genuine 2D round trip (e.g. briefly checking
        the pseudo-2D relaxation view and coming back) lost F1's phase
        entirely, even though a single toggle in isolation didn't.
        Carried over regardless of the direction switched *to* -- a
        dormant value sitting unread in a pseudo-2D spectrum's own
        metadata is harmless (nothing reads it in that state), and is
        exactly what makes the value available again the next time a
        toggle switches back to genuine 2D. Confirmed directly against
        real 2D data, including a full round trip."""
        index = self._selected_index()
        if index is None:
            return
        self._commit_pending_phase_to_metadata()
        spectrum = self.session.spectra[index]
        wants_pseudo_2d = not checked
        if spectrum.ndim != 2 or spectrum.is_pseudo_2d == wants_pseudo_2d:
            return
        phase0_f1 = spectrum.metadata.raw.get("phase0_f1")
        phase1_f1 = spectrum.metadata.raw.get("phase1_f1")
        try:
            self.session.reload(index, pseudo_2d=wants_pseudo_2d, **self._current_phase_kwargs(spectrum))
            new_spectrum = self.session.spectra[index]
            if phase0_f1 is not None:
                new_spectrum.metadata.raw["phase0_f1"] = phase0_f1
            if phase1_f1 is not None:
                new_spectrum.metadata.raw["phase1_f1"] = phase1_f1
            self._reapply_phase_post_reload(new_spectrum)
        except Exception as exc:  # noqa: BLE001 -- surfaced to the user, not swallowed
            QMessageBox.critical(self, "Failed to reprocess spectrum", str(exc))
            # Revert the button to match what's actually loaded.
            self.process_indirect_button.blockSignals(True)
            self.process_indirect_button.setChecked(not spectrum.is_pseudo_2d)
            self.process_indirect_button.blockSignals(False)
            return
        self._on_selection_changed(index)
        self._schedule_sidecar_save(index)

    def _on_magnitude_toggled(self, checked: bool) -> None:
        """Turn magnitude mode on/off for the selected spectrum,
        reprocessing from the original file.

        Deliberately a **reload** (like "Process Indirect Dimension"
        above, and unlike the window/zero-fill/phase controls, which
        recompute in memory from the cached raw FID): magnitude is a
        discrete on/off, not something dragged continuously, so there
        is no responsiveness argument for a second in-memory
        implementation -- and this project has repeatedly been bitten
        by exactly that ("two pipelines quietly diverge", see TODO.md).
        Reloading means ``io.bruker``'s own ``_apply_magnitude_mode``
        stays the single implementation.

        Passing ``magnitude=`` explicitly (rather than clearing it back
        to ``None``) is what makes "off" mean *off* even for a file
        whose own ``PH_mod`` asks for magnitude -- ``None`` would
        re-detect and switch it straight back on, which would make the
        button look broken for exactly the datasets it matters most
        for.
        """
        index = self._selected_index()
        if index is None:
            return
        self._commit_pending_phase_to_metadata()
        spectrum = self.session.spectra[index]
        if bool(spectrum.metadata.raw.get("magnitude_mode")) == checked:
            return
        try:
            self.session.reload(index, magnitude=checked)
        except TypeError:
            # A vendor reader without a `magnitude` parameter (only
            # Bruker has one today) -- surfaced plainly rather than
            # leaving the button stuck on a state that never applied.
            QMessageBox.information(
                self, "Magnitude mode not available",
                f"{self.session.label(index)} was loaded by a reader that doesn't support "
                f"magnitude mode yet -- it's currently Bruker-only.",
            )
            self.magnitude_button.blockSignals(True)
            self.magnitude_button.setChecked(not checked)
            self.magnitude_button.blockSignals(False)
            return
        except Exception as exc:  # noqa: BLE001 -- surfaced to the user, not swallowed
            QMessageBox.critical(self, "Failed to reprocess spectrum", str(exc))
            self.magnitude_button.blockSignals(True)
            self.magnitude_button.setChecked(not checked)
            self.magnitude_button.blockSignals(False)
            return
        # Record the *explicit* choice, separately from the reader's own
        # descriptive `magnitude_mode`. The two are not interchangeable:
        # `magnitude_mode` is absent both when the user turned this off
        # and when they never touched it, so it alone can't restore
        # "off" for a file whose own PH_mod asks for magnitude -- that
        # would silently switch back on at the next load. This flag is
        # what `_load` reads before the read to decide (see there).
        self.session.spectra[index].metadata.raw["magnitude_override"] = checked
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

    def _show_pseudo2d_heatmap(self, spectrum) -> None:
        """Plot every slice of a pseudo-2D series as a heatmap (F2 vs.
        slice number, colour = intensity) -- real roadmap ask:
        "pseudo-2D/stacked-1D data should be plottable as either a
        contour or a heatmap." Same slice extraction
        :meth:`_show_pseudo2d_stack` already uses; the rendering itself
        is :func:`opennmr.plotting.plot_1d_heatmap` (already ported
        from ``nmrtools2``'s own ``NMR1dheatmap``, just never called
        from GUI code before this)."""
        n_slices = spectrum.data.shape[0]
        slices = processing_slicing.get_slices(spectrum, rows=list(range(1, n_slices + 1)))
        self.canvas.show_1d_heatmap(slices["rows"])
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
    #: Same "user-level dotfile, survives an update/reinstall" reasoning
    #: as _FIGURE_STYLES_PATH -- a plain JSON dict of
    #: {probe_string: {config_key: value}}, explicit user request:
    #: "a configuration that each user can save directly into the app
    #: to store info like max gradient strength that can be associated
    #: automatically with the probe." Saving is an explicit action (the
    #: "Save as default for this probe" button); looking it up as a
    #: fallback when a freshly-loaded/reselected spectrum has no
    #: per-experiment value of its own is automatic -- matches "save...
    #: associated automatically" precisely: the user decides what to
    #: save and when, but never has to re-enter it per experiment
    #: afterward.
    _PROBE_CONFIG_PATH = os.path.expanduser("~/.opennmr/probe_config.json")

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

        self.output_contour_fraction_spin = QSpinBox()
        self.output_contour_fraction_spin.setRange(1, 100)
        self.output_contour_fraction_spin.setValue(25)  # matches plotting.plot_2d_contour's own default (0.25)
        self.output_contour_fraction_spin.setSuffix("%")
        self.output_contour_fraction_spin.setToolTip(
            "How much of a 2D contour's own resolution renders by "
            "default (lower = faster, coarser) -- doesn't affect "
            "a canvas currently set to Full Resolution (right-click "
            "the contour), or Save Figure As..., which always exports "
            "at true resolution regardless of this."
        )
        self.output_contour_fraction_spin.valueChanged.connect(self._on_contour_fraction_changed)
        style_form.addRow("2D downsample:", self.output_contour_fraction_spin)

        self.output_nq_line_checkbox = QCheckBox("Show diagonal (nQ) line")
        self.output_nq_line_checkbox.setToolTip(
            "Overlay a dashed y = n*x line on a 2D contour -- n = 1 for "
            "the main diagonal (e.g. an EXSY/NOESY/COSY-style homonuclear "
            "correlation), or a higher n for a multiple-quantum "
            "correlation. plotting.plot_2d_contour's own show_nQ_line/nQ "
            "parameters -- this is the GUI wiring for it."
        )
        self.output_nq_line_checkbox.toggled.connect(self._on_nq_line_changed)
        style_form.addRow(self.output_nq_line_checkbox)

        self.output_nq_value_spin = QSpinBox()
        self.output_nq_value_spin.setRange(1, 8)
        self.output_nq_value_spin.setValue(1)
        self.output_nq_value_spin.setEnabled(False)
        self.output_nq_value_spin.valueChanged.connect(self._on_nq_line_changed)
        style_form.addRow("Diagonal n:", self.output_nq_value_spin)

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

        # The canvas's own export default (8x8, or 7.5x10 once a 2D is
        # showing extracted slices), not the live embedded figure's
        # size -- that is whatever Qt's layout happened to give the
        # widget, which is not a sensible default for a saved figure.
        fig_w, fig_h = self.canvas.default_export_size()
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

    def _on_contour_fraction_changed(self, value: int) -> None:
        fraction = value / 100.0
        for canvas in self._all_canvases():
            canvas.set_contour_fraction(fraction)

    def _on_nq_line_changed(self, *_args) -> None:
        """The diagonal (nQ) overlay, stored against the spectrum it
        was set on rather than as one app-wide switch.

        Explicit user request: "plotting options like nQ line should be
        tied to the loaded spectrum". It genuinely belongs to the data,
        not the app -- a diagonal means something on a homonuclear
        correlation and nothing on a heteronuclear one, and n is a
        property of the experiment (1 for EXSY/NOESY/COSY, higher for a
        multiple-quantum correlation). As one global setting it followed
        the user onto every other spectrum they selected, including the
        ones it was meaningless on.

        Persisted with the rest of the processing state, so it survives
        a restart (:mod:`opennmr.io.persistence` writes every
        JSON-serialisable ``metadata.raw`` key it isn't told to skip)."""
        enabled = self.output_nq_line_checkbox.isChecked()
        n_value = self.output_nq_value_spin.value()
        self.output_nq_value_spin.setEnabled(enabled)
        for canvas in self._all_canvases():
            canvas.set_nq_line(enabled, [n_value])
        index = self._selected_index()
        if index is not None:
            raw = self.session.spectra[index].metadata.raw
            raw["nq_line"] = enabled
            raw["nq_value"] = n_value
            self._schedule_sidecar_save(index)

    def _on_contour_slices_changed(self, slices) -> None:
        """Store the canvas's extracted slices against the spectrum
        they were taken from -- same reasoning as the nQ line (see
        :meth:`_on_nq_line_changed`): a slice at F1 index 412 means
        nothing on any other dataset, so it cannot be canvas-local
        state that survives a reselection."""
        index = self._selected_index()
        if index is None:
            return
        self.session.spectra[index].metadata.raw["contour_slices"] = list(slices)
        self._schedule_sidecar_save(index)

    def _restore_contour_slices(self, spectrum) -> None:
        """Push ``spectrum``'s own stored slices back onto the canvas.
        Uses :meth:`MplCanvas.set_contour_slices`, which deliberately
        doesn't emit -- otherwise restoring would immediately write the
        same list back out as if the user had just added it."""
        self.canvas.set_contour_slices(spectrum.metadata.raw.get("contour_slices") or [])

    def _restore_nq_line_control(self, spectrum) -> None:
        """Put the nQ controls (and every canvas) back to whatever
        ``spectrum`` itself carries -- see :meth:`_on_nq_line_changed`.
        Signals are blocked so restoring doesn't write the value
        straight back out again as if the user had just set it."""
        enabled = bool(spectrum.metadata.raw.get("nq_line", False))
        n_value = int(spectrum.metadata.raw.get("nq_value", 1))
        self.output_nq_line_checkbox.blockSignals(True)
        self.output_nq_line_checkbox.setChecked(enabled)
        self.output_nq_line_checkbox.blockSignals(False)
        self.output_nq_value_spin.blockSignals(True)
        self.output_nq_value_spin.setValue(n_value)
        self.output_nq_value_spin.blockSignals(False)
        self.output_nq_value_spin.setEnabled(enabled)
        for canvas in self._all_canvases():
            canvas.set_nq_line(enabled, [n_value])

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

    def _load_probe_config(self) -> dict:
        try:
            with open(self._PROBE_CONFIG_PATH) as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _on_save_probe_gradient_default_clicked(self) -> None:
        index = self._selected_index()
        probe = self.session.spectra[index].metadata.probe if index is not None else None
        if not probe:
            QMessageBox.information(
                self, "No probe recorded",
                "This spectrum's own vendor file doesn't record a probe identifier, so there's "
                "nothing to associate this default with.",
            )
            return
        config = self._load_probe_config()
        config[probe] = {"max_gradient_g_per_cm": self.imaging_max_gradient_spin.value()}
        os.makedirs(os.path.dirname(self._PROBE_CONFIG_PATH), exist_ok=True)
        with open(self._PROBE_CONFIG_PATH, "w") as f:
            json.dump(config, f, indent=2)
        QMessageBox.information(
            self, "Saved",
            f"Saved {self.imaging_max_gradient_spin.value():g} G/cm as the default max gradient "
            f"for probe '{probe}' -- applied automatically to any spectrum recording this same "
            f"probe that doesn't already have its own value set.",
        )

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
        # Keep whatever the user had current across a rebuild (so Move
        # Up/Down and check-state toggles don't lose their place), but
        # never *pre*-select anything on a fresh populate. Explicit user
        # request: "in compare, none of the spectra should be
        # pre-selected like the last one is now."
        #
        # This narrows, rather than reverses, the earlier feedback that
        # it was "no longer obvious which spectrum is selected" here:
        # that was about the highlight being invisible once a row *had*
        # been chosen, which still works. What is gone is the automatic
        # choice made on the user's behalf before they have clicked
        # anything.
        if new_order and previous_current_entry in new_order:
            self.compare_list.setCurrentRow(new_order.index(previous_current_entry))
        else:
            self.compare_list.setCurrentRow(-1)
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
        self._warn_if_fit_near_region_edge(slice_results)
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

        # A way to overlay multiple 2D contours (real roadmap ask) --
        # reuses plotting.twod.overlay_plot (already ported from
        # nmrtools2's own NMR2D_overlay_plot, just never wired to the
        # GUI before this) via its own dedicated path, since its
        # compatibility rules (same F1/F2 nuclides, conventional vs.
        # pseudo-2D, 2-3 spectra only) and rendering (flat-colour line
        # contours, no mass-normalization/per-row-highlight concept)
        # are different enough from the 1D stack/superimpose view below
        # that folding them into the same code path would tangle two
        # genuinely different features together. A *mixed* 1D/2D, a
        # single 2D, or a >3-spectra 2D selection all fall through to
        # the existing 1D-only logic below unchanged (which already
        # skips non-1D entries and reports them as such).
        two_d_rows_and_candidates = [(row, c) for row, c in rows_and_candidates if c.ndim == 2]
        if len(rows_and_candidates) == len(two_d_rows_and_candidates) and 2 <= len(two_d_rows_and_candidates) <= 3:
            self._refresh_compare_2d_overlay(two_d_rows_and_candidates, switch_tab=switch_tab, force_render=force_render)
            return

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

    def _refresh_compare_2d_overlay(self, rows_and_candidates: list[tuple[int, object]],
                                     switch_tab: bool, force_render: bool) -> None:
        """:meth:`_refresh_compare_view`'s own 2-3-genuine-2D-spectra
        branch -- see its own comment for why this is a separate
        method rather than folded into the 1D logic. Mirrors that
        method's own "defer the draw until Compare is actually
        visited" staleness guard, but has no mass-normalization/
        per-row-highlight concept to carry over (an overlay's own
        legend, from each spectrum's ``Session.label``, already
        identifies which contour is which)."""
        if not force_render and not switch_tab and self.canvas_tabs.currentWidget() is not self.compare_canvas:
            self._compare_view_stale = True
            return
        self._compare_view_stale = False
        self._compare_plot_row_positions = {}

        labels = [self.session.label(row) for row, _c in rows_and_candidates]
        spectra = [c for _row, c in rows_and_candidates]
        try:
            self.compare_canvas.show_2d_overlay(spectra, labels=labels)
            self.compare_status_label.setText(f"Overlaying {len(spectra)} 2D spectra.")
            if switch_tab:
                self.canvas_tabs.setCurrentWidget(self.compare_canvas)
        except ValueError as exc:
            self.compare_status_label.setText(str(exc))
            self.compare_canvas.clear()
        except Exception as exc:  # noqa: BLE001 -- surfaced to the user, not swallowed
            QMessageBox.critical(self, "Failed to overlay spectra", str(exc))

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
        # Negative range needed for Gaussian (GM) -- real user report:
        # "Gaussian I think need[s] a negative lb" -- see
        # processing.ft._window_gaussian's own docstring for why.
        # Exponential itself only ever uses lb >= 0 in practice, but a
        # negative value there is harmless (just line-sharpening
        # instead of broadening), not worth a second spin box to
        # forbid it.
        self.lb_spin.setRange(-1.0e5, 1.0e5)
        self.lb_spin.setDecimals(2)
        self.lb_spin.setSuffix(" Hz")
        self.lb_spin.setToolTip(
            "Line broadening, for the Exponential window. For Gaussian, "
            "the line-sharpening term (typically negative, to cancel "
            "the FID's own decay before the Gaussian taper below)."
        )
        self.lb_spin.valueChanged.connect(lambda _v: self._on_fid_processing_control_changed())
        form.addRow("Line broadening:", self.lb_spin)

        self.gb_spin = QDoubleSpinBox()
        self.gb_spin.setRange(0.0, 1.0e5)
        self.gb_spin.setDecimals(2)
        self.gb_spin.setSuffix(" Hz")
        self.gb_spin.setToolTip(
            "Gaussian broadening width, for the Gaussian window only "
            "(see processing.ft._window_gaussian's own docstring -- "
            "not verified against real Bruker GM output). 0 disables "
            "the Gaussian term, leaving a plain exponential."
        )
        self.gb_spin.valueChanged.connect(lambda _v: self._on_fid_processing_control_changed())
        form.addRow("Gaussian broadening:", self.gb_spin)

        self.ssb_spin = QDoubleSpinBox()
        self.ssb_spin.setRange(0.0, 32.0)
        self.ssb_spin.setDecimals(2)
        self.ssb_spin.setValue(2.0)
        self.ssb_spin.setToolTip(
            "Sine-bell shift, Bruker's own SSB convention, for the Sine "
            "and QSine windows only. 2 = a pure cosine bell (no shift); "
            "0 or 1 = a plain sine bell starting at 0."
        )
        self.ssb_spin.valueChanged.connect(lambda _v: self._on_fid_processing_control_changed())
        form.addRow("Sine-bell shift (SSB):", self.ssb_spin)

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
        self._update_window_param_spin_enabled()

    def _update_window_param_spin_enabled(self) -> None:
        """Each window shape has its own parameters -- lb_spin (Exponential
        and Gaussian), gb_spin (Gaussian only), ssb_spin (Sine/QSine
        only) -- shown but disabled otherwise, same convention
        ``lb_spin`` alone already used before this existed."""
        key = self.window_combo.currentData()
        self.lb_spin.setEnabled(key in ("exponential", "gaussian"))
        self.gb_spin.setEnabled(key == "gaussian")
        self.ssb_spin.setEnabled(key in ("sine", "qsine"))

    def _on_window_combo_changed(self, _index: int) -> None:
        self._update_window_param_spin_enabled()
        self._on_fid_processing_control_changed()

    def _set_window_combo_key(self, key: str) -> None:
        idx = self.window_combo.findData(key)
        self.window_combo.blockSignals(True)
        self.window_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.window_combo.blockSignals(False)
        self._update_window_param_spin_enabled()

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
            # Note: _fid_preview_available (every real call site's own
            # gate) already excludes a genuine (non-pseudo) 2D
            # entirely -- this branch only ever runs for pseudo-2D in
            # practice.
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

    def _phase_2d_proportional_row(self, spectrum, raw_row_count: int) -> int:
        """Map :attr:`_phase_2d_slice_index` (an index into the
        *processed*, zero-filled F1 axis -- ``spectrum.data.shape[0]``
        long) onto an approximate row in a differently-sized raw array
        (``metadata.raw["fid_raw"]``/``["fid"]``, both pre-F1-zero-fill
        and, for a hypercomplex ``FnMODE``, pre-decode too -- so a
        genuine 2D's own raw row count essentially never equals its
        processed F1 length; confirmed on real data, 256 raw rows vs.
        1024 processed). There's no true inverse for a zero-filled
        index -- it doesn't correspond to any single real acquired
        point -- so this is deliberately proportional ("roughly the
        same relative position in the acquisition"), not exact, which
        is good enough for a preview but should not be treated as
        picking out a specific real data point."""
        processed_row_count = spectrum.data.shape[0]
        if processed_row_count <= 1 or raw_row_count <= 1:
            return 0
        fraction = (self._phase_2d_slice_index - 1) / (processed_row_count - 1)
        return int(round(fraction * (raw_row_count - 1)))

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
            if spectrum.is_pseudo_2d:
                row = min(max(self.slice_spin.value() - 1, 0), raw_fid.shape[0] - 1)
                spectrum_row = row + 1
            elif self._phase_2d_slice_index is not None and self.dim_combo.currentIndex() == 0:
                # This method only ever previews the F2 (direct)
                # dimension -- callers route the F1 (indirect) case to
                # _show_2d_indirect_processing_preview instead, since
                # an F1 trace can't be cheaply rebuilt from one raw row
                # the way an F2 trace can (F1 only exists once every
                # row's own F2 processing is already done -- see that
                # method's own docstring). Real user requirement,
                # repeated across two rounds of feedback: the Processing
                # tab must show a single representative slice for *any*
                # genuine 2D, not just while a phase drag happens to be
                # in progress -- so this always uses whichever F1 row
                # is currently the largest-magnitude one
                # (_phase_2d_slice_index, the same single source of
                # truth the Spectrum tab's own phase-preview slice
                # uses, see _capture_phase_reference), not gated behind
                # _phase_mode/an active drag. (_show_fid_preview has no
                # analogous need: it's always gated off for a genuine
                # 2D by _fid_preview_available, so its own
                # raw_fid.ndim == 2 branch never runs for this case.)
                row = self._phase_2d_proportional_row(spectrum, raw_fid.shape[0])
                # _phase_2d_slice_index is already the exact 1-based
                # index into spectrum's own (processed) F1 axis --
                # unlike `row`, which _phase_2d_proportional_row's own
                # docstring warns is only an approximate mapping into
                # the differently-sized *raw* array, "not... a specific
                # real data point". Using it directly here (rather than
                # row + 1) only happens to give the same value today
                # because _phase_2d_slice_index is hardcoded to 1 (see
                # _capture_phase_reference) -- keeping this explicit
                # avoids silently breaking again if that hardcoding is
                # ever revisited, the way it already changed once this
                # session (away from a largest-magnitude pick).
                spectrum_row = self._phase_2d_slice_index
            else:
                # No valid F2 slice index yet (e.g. dim_combo is on F1
                # -- shouldn't normally reach here, see above -- or
                # spectrum.data_imag is None so phasing, and therefore
                # _phase_2d_slice_index, was never set up at all).
                # "First" row, matching the user's own "first or
                # largest magnitude" framing for the default.
                row = 0
                spectrum_row = row + 1
            # title labels the FID panel specifically (show_processing_
            # preview sets it via fid_ax.set_title, not on the spectrum
            # axes below) -- that panel's own data is raw_fid[row]
            # (unchanged, still genuinely raw-row-based even after the
            # spectrum-panel fix below), so the title must keep
            # describing *that* row, i.e. row + 1, not spectrum_row.
            # _phase_2d_proportional_row's own job is picking a raw row
            # that approximately *is* the same acquisition slice as
            # spectrum_row, so this remains an accurate (if approximate)
            # label for the spectrum panel too whenever the two diverge
            # (e.g. after a live F1 zero-fill/truncate) -- see
            # test_main_window_processing_preview_slice_matches_spectrum_tab_across_zero_fill_mismatch.
            title = self._slice_label(spectrum, row + 1)
            raw_fid = raw_fid[row]
            # Echo-Antiecho (FnMODE 6): "fid_raw" already carries
            # whatever global conjugate read_2d applies to every row
            # (its own _CONJUGATE_QUADRATURE_NOTE step), but not its
            # own *extra* even-row conjugate-and-reversal, which is a
            # per-pair fix that isn't knowable until a specific row is
            # actually picked. No Echo-Antiecho raw-row transform
            # here either -- see _recompute_multi_row_data's own
            # matching note and io.bruker.read_2d's fn_mode == 6 one.
            slice_template = get_trace(spectrum, axis="f1", index=spectrum_row)

        computed = self._process_raw_fid_row(spectrum, raw_fid, dt)
        if computed is None:
            return None
        filled, data = computed

        # Real design requirement, stated explicitly: "2D processing
        # should use slice one of the xf2, whether or not the indirect
        # dimension has been processed, not index 1 of the fully
        # processed data." An earlier round of this exact method tried
        # making this branch return slice_template (spectrum.data's own
        # row, i.e. an F1 frequency bin) instead, reasoning that "raw
        # row reprocessed through F2 alone" and "F1's actual frequency
        # bin" are different quantities whenever real F1 modulation
        # exists (true, and still true) -- but that fix optimized for
        # "match the fully-processed contour" over "show what F2's own
        # controls, alone, would do to the raw acquisition," which is
        # this tab's actual, stated purpose (matching xf2/raw-row
        # semantics that pseudo-2D and 1D already use here without
        # controversy). Reverted once the user clarified the intended
        # design directly, rather than left as a guess either way --
        # `filled`/`data` (the raw-row reprocessing computed just above)
        # fall through to the same ppm/phase/NMRSpectrum-construction
        # code every other case already uses, unconditionally.

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
        # Rebuilt from the raw FID, so the file's own recorded phase has
        # to be re-applied here too -- the reader's own application of it
        # lives in the spectrum this replaces, not in `fid_raw`. Same
        # composition, same order, as io.bruker.read_1d/read_2d.
        data = io_bruker.apply_dimension_phase(
            data, ppm, io_bruker.recorded_phase(spectrum.metadata.raw, "f2"),
            (phase0, phase1), pivot_frac=pivot_frac,
        )
        data = self._baseline_correct_if_enabled(data, ppm, spectrum)

        axis = NMRAxis(kind="frequency", ppm=ppm, hz=hz)
        preview_spectrum = NMRSpectrum(
            data=np.real(data), axes=[axis], metadata=slice_template.metadata, data_imag=np.imag(data),
        )
        return filled, dt, preview_spectrum, title

    def _show_2d_indirect_processing_preview(self, spectrum) -> None:
        """The Processing tab's own preview for a genuine 2D while the
        Dimension box is on F1 (indirect) -- :meth:`_compute_processing_preview`'s
        sibling for the other dimension, but structurally different:
        an F1 trace only exists once every raw row has already been
        through its *own* F2-domain processing (window/zero-fill/FFT),
        so there's no cheap single-raw-row rebuild the way F2 has.
        That per-row work is exactly what the debounced whole-dataset
        recompute (:meth:`_recompute_multi_row_data`) already keeps
        ``spectrum.data`` current with (F1 zero-fill/truncate, *and*
        both dimensions' own durably-committed phase, included -- see
        that method's own docstring) -- so this just reads a trace
        straight off ``spectrum`` and displays it, deliberately
        applying *no* phase of its own -- ``spectrum.data`` is already
        the single source of truth for "current total phase," the same
        principle a genuine 2D's own Spectrum-tab contour now relies on
        too (self.canvas no longer renders its own 2D phase slice at
        all -- see :meth:`_apply_phase_preview`'s own docstring).

        That "no phase of its own" choice isn't an oversight: whichever
        dimension ``dim_combo`` currently points to already has its
        live phase-drag delta written straight into ``spectrum.data``
        by :meth:`_apply_phase_preview`'s own "2d" branch (uniformly,
        across the whole array) *before* this method ever runs -- a
        first version of this method re-applied ``phase0_f1``/
        ``phase1_f1`` plus the live spin-box delta on top of the
        already-phased trace, silently double-phasing it. Caught by
        cross-checking this method's own output numerically against a
        direct ``get_trace`` read of ``spectrum.data`` (the trusted
        reference, same source this method itself now uses), on real
        data (``test_data/20200804_NaMgMnO_Pristine_111218C_Check_23Na_1p3mm/31``)
        -- differed by a magnitude comparable to the data itself, not a
        rounding-sized gap.

        No FID-domain top panel either, for the same reason
        :meth:`_show_processing_preview`'s own docstring gives for
        skipping it elsewhere: F1 has no window/shift/echo/LP controls
        of its own to judge against a raw trace.

        Falls back to the plain :meth:`MplCanvas.show_spectrum` full-
        contour view when there's no slice index to show yet
        (``spectrum.data_imag`` is ``None``, so phasing -- and
        therefore :attr:`_phase_2d_slice_index` -- was never set up)."""
        if self._phase_2d_slice_index is None:
            self.processing_preview_canvas.show_spectrum(spectrum)
            return
        trace = get_trace(spectrum, axis="f2", index=self._phase_2d_slice_index)

        canvas = self.processing_preview_canvas
        # Prefer an in-place update, only rebuild when there's no
        # existing line to update -- a live phase drag shouldn't reset
        # the view on every tick, including preserving whatever ppm
        # zoom was already on screen the first time this replaces a
        # full contour.
        if not canvas.ax.lines:
            preserve_xlim = canvas._contour_spectrum is not None
            xlim = canvas.ax.get_xlim() if preserve_xlim else None
            canvas.show_spectrum(trace)
            if xlim is not None:
                canvas.ax.set_xlim(xlim)
        else:
            canvas.update_data(trace)
        canvas.ax.set_title(f"F2 column {self._phase_2d_slice_index} (phase preview)")
        canvas.set_pivot_marker(self.pivot_spin.value())
        canvas.canvas.draw_idle()

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

        # Whether the live phase-editing session (_phase_mode/dim_combo/
        # pivot_spin) is actually targeting *this* spectrum right now --
        # guards the F2/F1 pivot reuse below from a live pivot_spin
        # value that belongs to a completely different, unrelated
        # spectrum (this method's callers always operate on whatever's
        # currently selected, and _capture_phase_reference keeps
        # _phase_target_index in sync with that on every selection
        # change, so this should hold in practice -- checked explicitly
        # rather than assumed, since nothing else in this method
        # verifies it).
        is_phase_target = (
            self._phase_target_index is not None
            and 0 <= self._phase_target_index < len(self.session)
            and self.session.spectra[self._phase_target_index] is spectrum
        )

        fn_mode = spectrum.metadata.raw.get("fn_mode", 1)
        # No raw-row transform for Echo-Antiecho (fn_mode 6). This
        # method used to carry a copy of one, mirroring io.bruker.
        # read_2d's; that transform turned out to be destroying the
        # echo rows (it reversed each FID *before* the window and
        # zero-fill, so the window was applied backwards), and the
        # recombination it was propping up was wrong too. Both are
        # gone from the reader -- see its own fn_mode == 6 note and
        # processing.ft.decode_hypercomplex_f1. Kept here as a comment
        # rather than silence because the "two pipelines quietly
        # diverge" class this method's docstring warns about cuts both
        # ways: a copy that is *not* deleted alongside the original is
        # the same bug wearing the opposite sign.

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
        # pivot_spin is a single, shared widget -- only meaningful for
        # F2 specifically while F2 is the currently-focused dimension
        # (pseudo-2D phasing is always F2, so always meaningful there).
        # Real bug this fixes: reusing it unconditionally meant F2's
        # own phase1 got reapplied around whatever pivot F1 was last
        # centred on (in F1's own ppm units) any time this recompute
        # ran after switching dimensions -- a genuinely different,
        # wrong correction whenever phase1 is nonzero, not just a
        # cosmetic difference (pivot_frac defines *where* the phase1
        # ramp is zero). Centre (0.5) otherwise, matching
        # _current_phase_kwargs's own reasoning: no per-dimension pivot
        # is persisted anywhere, and centre is what a fresh
        # _capture_phase_reference always resets to, so it's what
        # phase0/phase1 were computed against unless the user
        # deliberately moved the pivot during that same session.
        f2_pivot_is_live = is_phase_target and (
            spectrum.is_pseudo_2d or (self._phase_mode == "2d" and self.dim_combo.currentIndex() == 0)
        )
        if f2_pivot_is_live:
            # Real bug, found and fixed: this method only ever read the
            # *committed* phase0/phase1, never the live, not-yet-
            # committed delta still sitting in the Phase spin boxes
            # mid-drag -- correct once a drag is released and committed
            # (_commit_pending_phase_to_metadata runs first), but this
            # method can also fire *during* an active drag: baseline
            # correction is deliberately skipped on every drag tick
            # (too slow to re-run per-tick) and instead deferred to
            # this same debounced recompute, restarted on every tick
            # while baseline_correct is on (see _apply_phase_preview's
            # own "2d"/pseudo2d branches). If the drag pauses for the
            # 200ms debounce window before being released, this method
            # fires mid-drag and rebuilt spectrum.data from the raw FID
            # using only the *old*, pre-drag committed phase -- silently
            # reverting the live-dragged display back to it. Real user
            # report: "I get a flicker as it saves and it seems to
            # reload something old." Confirmed directly: a live 77-
            # degree drag followed by this method firing (simulating
            # the debounce winning a race against the user releasing)
            # showed a ~1.35e9 magnitude difference between the
            # correctly-live-phased data and what this method rebuilt
            # -- not a rounding-level gap.
            phase0 += self.phase0_spin.value()
            phase1 += self.phase1_spin.value()
        pivot_frac2 = ((self.pivot_spin.value() - ppm2.min()) / span2 if span2 else 0.5) if f2_pivot_is_live else 0.5
        # Recorded phase re-applied for the same reason as in
        # _process_raw_fid_row: this rebuilds from `fid_raw`, which is
        # the FID *before* the reader phased anything.
        data = io_bruker.apply_dimension_phase(
            data, ppm2, io_bruker.recorded_phase(spectrum.metadata.raw, "f2"),
            (phase0, phase1), pivot_frac=pivot_frac2,
        )

        dt_f1 = spectrum.metadata.raw.get("fid_dt_f1")
        f1_axis = spectrum.axes[0]
        if not spectrum.is_pseudo_2d and dt_f1:
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
            # Same F1-quadrature normalisation io.bruker.read_2d does
            # before decoding -- this path rebuilds the same Bruker data,
            # so it must normalise it the same way.
            if fn_mode in (4, 5):
                data = data.copy()
                data[1::2] *= -1
            data = processing_ft.decode_hypercomplex_f1(data, fn_mode)
            # F1 linear prediction, same as io.bruker.read_2d -- without
            # it this rebuild would silently drop it and the spectrum
            # would jump (broader F1 lines) the moment any control was
            # touched. Deliberately follows the file's own ME_mod here,
            # matching the load, rather than being skipped for speed:
            # a live rebuild that differs from the load is exactly the
            # failure this project keeps hitting.
            proc2s_lp = spectrum.metadata.raw.get("proc2s")
            if proc2s_lp:
                data = io_bruker.apply_recorded_f1_linear_prediction(data, proc2s_lp)
            if spectrum.metadata.vendor == "bruker":
                # Real bug, found and fixed: this method never applied
                # any window to F1 at all -- read_2d always does
                # (defaulting to proc2s's own recorded WDW, the same
                # "__recorded__" convention F2's own window_combo
                # exposes, but with no equivalent live control for F1 at
                # all -- so "recorded" is the only correct choice here,
                # unconditionally). Confirmed directly: even a live
                # recompute with *no* phase or FID-processing change at
                # all (phase0=phase1=0, every other control untouched)
                # produced data ~80% different from the original,
                # freshly-loaded spectrum on a real COSY dataset (proc2s
                # WDW=4/QSINE) -- not a rounding-level gap, and not
                # explained by anything phase-related; isolated to this
                # exact missing step by confirming F2's own per-row
                # processing already matched read_2d's own primitives
                # exactly (0.0 diff) before looking at F1. X-Pulse's own
                # read_2d applies no F1 window either (confirmed by
                # reading it directly), so this stays Bruker-only.
                proc2s = spectrum.metadata.raw.get("proc2s")
                sw_1_for_window = spectrum.metadata.raw.get("proc2s", {}).get("SW_p") if proc2s else None
                if proc2s and sw_1_for_window:
                    try:
                        data = np.apply_along_axis(
                            lambda col: io_bruker._apply_bruker_window(col, proc2s, float(sw_1_for_window)),
                            0, data,
                        )
                    except NotImplementedError:
                        pass  # matches read_2d's own "leave unwindowed" fallback for an unsupported WDW
            live_zero_fill_f1 = spectrum.metadata.raw.get("live_zero_fill_f1")
            if isinstance(live_zero_fill_f1, int) and not isinstance(live_zero_fill_f1, bool):
                si1 = live_zero_fill_f1
            else:
                # F1's own current (already-decoded) length -- unaffected
                # by F2-only controls, absent an explicit live F1 choice.
                si1 = spectrum.data.shape[0]
            f1_filled = np.apply_along_axis(lambda col: processing_ft.zero_fill(col, si1), 0, data)
            data = np.apply_along_axis(lambda col: processing_ft.fft_spectrum(col, dt_f1)[1], 0, f1_filled)
            if fn_mode not in (4, 5, 6):
                # Mirrors io.bruker.read_2d's own matching fix -- same
                # "two pipelines quietly diverge" class this project has
                # hit before (e.g. the fn_mode==6 raw-row fix a few
                # lines above this method's own docstring already
                # covers): this method rebuilds F1's own FFT from
                # scratch, independently of read_2d, so it needs the
                # identical F1-mirror fix applied here too, not just
                # there. See read_2d's own comment on this exact branch
                # for the full "conjugating before an FFT reverses that
                # FFT's own frequency-bin order" story and the real COSY
                # data (PULPROG cosygpmfppqf) that caught it.
                data = data[::-1, :]
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
            # Mirror image of f2_pivot_is_live above: pivot_spin is
            # only meaningful for F1 while F1 is the currently-
            # focused dimension of the live-targeted spectrum.
            f1_pivot_is_live = is_phase_target and self._phase_mode == "2d" and self.dim_combo.currentIndex() == 1
            if f1_pivot_is_live:
                # Same live-drag-vs-debounced-recompute race as F2's
                # own fix just above -- see that comment for the full
                # story. Computed (and the "should we bother calling
                # apply_phase at all" check below moved after it)
                # *before* the old `if phase0_f1 or phase1_f1` guard,
                # which used to only ever see the committed total: a
                # live F1 drag that hadn't committed anything yet (both
                # committed values still exactly 0) skipped this whole
                # block entirely, silently dropping the live delta
                # rather than just reapplying a stale phase.
                phase0_f1 += self.phase0_spin.value()
                phase1_f1 += self.phase1_spin.value()
            recorded_f1 = io_bruker.recorded_phase(spectrum.metadata.raw, "f1")
            # The guard now includes the recorded phase: F1's own
            # recorded values can be nonzero while the user has dialled
            # nothing, and skipping the call then would drop the file's
            # own phase on every live rebuild.
            if phase0_f1 or phase1_f1 or any(recorded_f1):
                span1 = ppm1.max() - ppm1.min()
                pivot_frac1 = (
                    ((self.pivot_spin.value() - ppm1.min()) / span1 if span1 else 0.5) if f1_pivot_is_live else 0.5
                )
                data = io_bruker.apply_dimension_phase(
                    data.T, ppm1, recorded_f1, (phase0_f1, phase1_f1), pivot_frac=pivot_frac1,
                ).T

        data = self._baseline_correct_if_enabled_multi_row(data, ppm2, spectrum)
        spectrum.axes = [f1_axis, NMRAxis(kind="frequency", ppm=ppm2, hz=hz2)]
        # Magnitude/power last, exactly as the reader does it -- this
        # path rebuilds from the raw FID, so without it a magnitude-mode
        # spectrum silently reverted to phase-sensitive the moment any
        # control was touched (measured at 1.93 relative difference on
        # the real magnitude-mode COSY, against 0.000000 for every
        # phase-sensitive dataset once the recorded phase was wired in).
        magnitude_mode = spectrum.metadata.raw.get("magnitude_mode")
        if magnitude_mode:
            data = io_bruker._apply_magnitude_mode(data, magnitude_mode)
            spectrum.data = np.real(data)
            spectrum.data_imag = None
            return True
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
        computation of it) only when there's genuinely nothing else to
        show: not FID-reprocessable at all. A genuine (non-pseudo) 2D
        does *not* fall into this branch any more -- real, repeated
        user feedback that this tab kept showing a full contour
        (expensive to redraw on every tick, and visually inconsistent
        with the Spectrum tab's own single-slice phase preview) instead
        of "just one slice, like it already does for pseudo-2D" for
        *any* genuine 2D, not only while a phase drag happened to be in
        progress. F2 (direct) routes through :meth:`_compute_processing_preview`
        below, same as pseudo-2D; F1 (indirect) has no cheap from-raw
        rebuild (see :meth:`_show_2d_indirect_processing_preview`'s own
        docstring), so it's handled separately."""
        if not self._fid_reprocessable(spectrum):
            self.processing_preview_canvas.show_spectrum(spectrum)
        elif spectrum.ndim == 2 and not spectrum.is_pseudo_2d and self.dim_combo.currentIndex() != 0:
            self._show_2d_indirect_processing_preview(spectrum)
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
        tab, not just phase.

        A genuine 2D on F1 (indirect) has no dual FID+spectrum panel to
        update in place at all -- routed to
        :meth:`_show_2d_indirect_processing_preview` instead, which
        manages its own in-place-vs-rebuild choice."""
        if self.canvas_tabs.currentWidget() is not self.processing_preview_canvas:
            return
        index = self._selected_index()
        if index is None:
            return
        spectrum = self.session.spectra[index]
        if spectrum.ndim == 2 and not spectrum.is_pseudo_2d and self.dim_combo.currentIndex() != 0:
            self._show_2d_indirect_processing_preview(spectrum)
            return
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
        # reach a large useful value first. shape[-1] (not len(), which
        # reads axis 0 -- the *array-step* count for an X-Pulse arrayed/
        # pseudo-2D dataset's own 2D fid_raw, e.g. (16 steps, 2048
        # points) -- would silently cap this at 16 for a real 16-step
        # PFG/T1/T2 series, one point count short of true, same "len()
        # reads the wrong axis for 2D input" bug already found and
        # fixed once this session in processing.ft.fft_spectrum;
        # real user report, confirmed by direct reproduction: a
        # 16-step arrayed load capped Truncate's own maximum at 16
        # instead of the real per-FID point count) -- correct for both
        # a plain 1D fid_raw (single-file X-Pulse, shape[-1] == len())
        # and a 2D one (arrayed) alike.
        raw_fid = spectrum.metadata.raw.get("fid_raw") if is_xpulse else None
        raw_length = raw_fid.shape[-1] if raw_fid is not None else (1 << 20)
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
            (self.gb_spin, raw.get("live_gb", 0.0)),
            (self.ssb_spin, raw.get("live_ssb", 2.0)),
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
        if key == "gaussian":
            return "gaussian", {"lb": self.lb_spin.value(), "gb": self.gb_spin.value()}
        if key in ("sine", "qsine"):
            return key, {"ssb": self.ssb_spin.value()}
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
        raw["live_gb"] = self.gb_spin.value()
        raw["live_ssb"] = self.ssb_spin.value()
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
        shape_before = spectrum.data.shape
        rebuilt_from_raw = self._fid_reprocessable(spectrum)
        if rebuilt_from_raw:
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
        if self._phase_mode in ("pseudo2d", "2d") and spectrum.data.shape != shape_before:
            # Real crash, found via a user-reported traceback: a live
            # FID-processing change (e.g. zero-fill) that reaches here
            # resizes spectrum.data/axes via _recompute_multi_row_data
            # above, but _phase_reference_full/_phase_freq_ppm (this
            # editing session's own snapshot, taken once by
            # _capture_phase_reference and otherwise never refreshed --
            # see its own docstring, "never called mid-drag") stayed
            # at their *old* size. The next phase-drag/pivot tick then
            # wrote that stale-shaped snapshot straight into
            # spectrum.data (mismatched against the just-updated,
            # correctly-*new*-shaped spectrum.axes) -- ValueError: x
            # and y must have same first dimension, in plot_1d, on the
            # very next pivot/phase interaction after a zero-fill
            # change settled. Re-anchoring here, immediately after the
            # spectrum that snapshot describes actually changed shape
            # (and, for "2d", before the display block below, so
            # _phase_2d_slice_index reflects the fresh data too),
            # closes the gap -- same "commit previous, then re-anchor
            # to current state" mechanism _capture_phase_reference
            # already uses for a slice/dimension change, just also
            # triggered by a live reprocess settling, not only by user
            # navigation. Committing "pending" phase here is a safe
            # no-op in practice: the debounce that lands this method
            # only fires 200ms after the *last* control change, so no
            # drag can genuinely be in flight (the spin boxes hold
            # whatever they already held, most commonly 0).
            #
            # Gated on the shape actually changing, not just any
            # settled reprocess -- a *value*-only settle (e.g. baseline
            # correction alone, no zero-fill) must NOT re-anchor here:
            # _phase_reference_full is deliberately a clean, phase-only
            # snapshot that the baseline-correction path bypasses on
            # purpose (_recompute_multi_row_data writes straight into
            # spectrum.data from the raw FID, never touching
            # _phase_reference_full) -- re-capturing it from
            # spectrum.complex_data after a baseline-only settle would
            # bake baseline correction into what's supposed to be the
            # "undo back to this" reference, breaking the baseline
            # on/off toggle (found by a real regression test failure
            # this caused: baseline_correct_button.setChecked(False)
            # stopped visibly un-correcting anything, since the
            # "uncorrected" reference it recomputed from wasn't
            # actually uncorrected any more).
            self._capture_phase_reference()
        # Real, repeated user report: "the Spectrum tab should always
        # show the 2D or pseudo 2D where there is one" -- self.canvas
        # never switches to a single-slice line for phase-editing
        # purposes any more (see _apply_phase_preview's own matching
        # comment), so a settled FID-processing change just refreshes
        # whatever it's already showing -- always a contour/stack, not
        # conditionally so.
        # This rebuild's result is baseline-corrected and carries the
        # full live phase; make it the new reference so the drag
        # continues *from* it. Without this the next tick re-derived
        # the display from the pre-drag snapshot, throwing the baseline
        # away again -- the display then alternated between corrected
        # and not on every pause, which is the reported "bouncing".
        # Only for the multi-row modes, which are the ones that defer
        # baseline correction to this rebuild in the first place.
        # Only when the rebuild came from the raw FID. The
        # processed-only fallback (_recompute_multi_row_phase_and_
        # baseline_only) *derives* its result from this reference, so
        # overwriting it with that result would make the reference
        # self-referential -- baseline correction could then never be
        # turned back off, because the "uncorrected" source it would
        # revert to had been replaced by the corrected output. Caught
        # by the pseudo-2D apply-all-slices test, which toggles
        # baseline off and back on.
        if (rebuilt_from_raw and self._phase_mode in ("pseudo2d", "2d")
                and self._phase_reference_full is not None):
            refreshed = spectrum.complex_data
            if refreshed is not None and refreshed.shape == self._phase_reference_full.shape:
                self._phase_reference_full = refreshed
                self._phase_reference_delta_applied = (
                    self.phase0_spin.value(), self.phase1_spin.value(),
                )

        if self._phase_mode in ("pseudo2d", "2d"):
            on_processing_preview = self.canvas_tabs.currentWidget() is self.processing_preview_canvas
            self._redraw_2d_preserving_view(spectrum)
            # This *is* the settle redraw, so cancel the canvas-refresh
            # debounce rather than letting it fire straight afterwards
            # and rebuild the identical contour a second time. Both
            # timers are 200 ms and both restart on every drag tick, so
            # before this they always fired as a pair on a pause --
            # measured at 295 ms per contour rebuild for
            # Cyclosporine/203 and 1.34 s for the EXSY, i.e. the whole
            # cost of a settle was being paid twice for one frame.
            self._canvas_2d_refresh_timer.stop()
            self._canvas_2d_refresh_pending_index = None
            if self._phase_mode == "pseudo2d":
                if self.canvas_tabs.currentWidget() is self.fid_canvas and self._fid_preview_available(spectrum):
                    self._show_fid_preview(spectrum, switch_tab=False)
                if on_processing_preview:
                    self._refresh_processing_preview_if_active()
            elif on_processing_preview:
                self._refresh_processing_preview_phase_only()

    #: A contour rebuild at or under this (seconds) is cheap enough to
    #: run *during* a phase drag, giving live feedback; anything slower
    #: falls back to pure debounce, where nothing moves until the drag
    #: pauses. Between the two real measurements this was chosen from:
    #: 295 ms for a Cyclosporine/203 HSQC (live) and 1.34 s for the
    #: EXSY (debounce). Explicit user choice of adaptive throttling
    #: over always-debounce.
    _LIVE_CONTOUR_REBUILD_LIMIT_S = 0.4

    def _schedule_canvas_2d_refresh(self, index: int) -> None:
        """Schedule ``self.canvas``'s own 2D/pseudo-2D contour redraw
        during a live phase drag -- see the timer's own construction
        comment in ``__init__`` for why this isn't simply redrawn on
        every tick the way the phase correction itself is applied.

        **Adaptive, not a fixed debounce.** A contour this canvas has
        already rebuilt once and measured at or under
        :attr:`_LIVE_CONTOUR_REBUILD_LIMIT_S` is redrawn *during* the
        drag, throttled to one rebuild per its own measured duration --
        so an HSQC phases visibly, instead of the user dragging blind
        and only seeing the result on a pause. Anything slower (or not
        yet measured at all) keeps the original 200 ms debounce, since
        for the EXSY a live rebuild would be 1.34 s of frozen UI per
        frame and strictly worse than showing nothing until the pause.

        The first drag on a given shape always debounces: there is no
        measurement yet, and assuming "cheap" would be exactly the
        wrong guess on the datasets that most need protecting. The
        settle redraw that pause produces is what records the timing
        (see :meth:`_redraw_2d_preserving_view`), so a second drag on
        the same spectrum can go live.

        Either way this only ever *schedules*; the drag's own settle
        path still does a final authoritative redraw and cancels
        whatever is pending here, so a throttled frame can never be the
        last word on screen."""
        self._canvas_2d_refresh_pending_index = index
        measured = self._contour_rebuild_seconds.get(self._contour_shape_for(index))
        if measured is None or measured > self._LIVE_CONTOUR_REBUILD_LIMIT_S:
            self._canvas_2d_refresh_timer.start(200)
            return
        # One rebuild per measured-duration at most, so the drag's own
        # event handling keeps getting a share of the main thread
        # rather than the redraws saturating it.
        interval = max(0.05, measured)
        elapsed = time.perf_counter() - self._last_contour_rebuild_at
        if elapsed >= interval:
            self._canvas_2d_refresh_timer.stop()
            self._run_debounced_canvas_2d_refresh()
            return
        self._canvas_2d_refresh_timer.start(max(1, int((interval - elapsed) * 1000)))

    def _contour_shape_for(self, index: int) -> tuple | None:
        """The data shape :attr:`_contour_rebuild_seconds` is keyed by,
        for session entry ``index`` -- ``None`` when there is no such
        entry (an index left over from a removed spectrum), which
        simply reads as "never measured"."""
        if index is None or index < 0 or index >= len(self.session):
            return None
        return self.session.spectra[index].data.shape

    def _run_debounced_canvas_2d_refresh(self) -> None:
        index = self._canvas_2d_refresh_pending_index
        self._canvas_2d_refresh_pending_index = None
        if index is None or index != self._selected_index() or index >= len(self.session):
            return
        spectrum = self.session.spectra[index]
        if spectrum.ndim != 2:
            return
        self._redraw_2d_preserving_view(spectrum)

    def _redraw_2d_preserving_view(self, spectrum) -> None:
        """Rebuild ``self.canvas``'s contour for ``spectrum`` without
        throwing away whatever ppm/index zoom is currently on it.

        A full contour rebuild has no in-place update path (see
        ``MplCanvas.set_full_resolution``'s own docstring) and resets
        the view by default. Both debounced settle paths need the same
        "phase-editing shouldn't silently reset a deliberate zoom"
        treatment already applied to the 1D case in
        :meth:`_apply_phase_preview` -- real user report: "Something is
        up with the GUI phase control for the indirect and it's jumping
        during phasing". The jump was literal: with baseline correction
        on, both 200 ms debounce timers fire on every pause in a drag,
        and :meth:`_run_debounced_multi_row_reprocess` used a bare
        ``show_spectrum`` -- so a view zoomed to 4..2 ppm x 60..20 ppm
        snapped back to the full 12.7..-3.3 x 157.5..-7.5 every time
        the user paused.

        Best-effort by design: this runs from timer callbacks during a
        drag, where a render failure must never raise (same reasoning
        as :meth:`_apply_phase_preview`'s own "2d" branch) -- skip the
        frame and let the next one catch up."""
        started = time.perf_counter()
        try:
            xlim, ylim = self.canvas.ax.get_xlim(), self.canvas.ax.get_ylim()
            self.canvas.show_spectrum(spectrum)
            self.canvas.ax.set_xlim(xlim)
            self.canvas.ax.set_ylim(ylim)
            self.canvas.canvas.draw_idle()
        except Exception:  # noqa: BLE001 -- see docstring
            pass
        # Timed on every rebuild, including a failed one (a frame that
        # bailed early is not evidence the real thing is cheap, but it
        # also must not leave a stale fast measurement in place). This
        # is what makes the live-vs-debounce choice in
        # _schedule_canvas_2d_refresh self-calibrating rather than a
        # guess from the data's shape.
        self._last_contour_rebuild_at = time.perf_counter()
        self._contour_rebuild_seconds[spectrum.data.shape] = self._last_contour_rebuild_at - started

    def _on_reset_processing_clicked(self) -> None:
        """Discard every live *processing* choice for the selected
        spectrum -- window/zero-fill/shift/align-echo/linear-predict/
        truncate, phase, baseline -- reverting to exactly how it
        looked right after it was first loaded (see ``_load``'s own
        snapshot into :attr:`_original_load_kwargs`). The opposite of
        the old "Apply Processing" button (which committed changes
        forward, unreliably); this discards them instead -- explicit
        user request: "remove the button, we can have a reset
        button".

        Deliberately does *not* revert a 2D's own "Process Indirect
        Dimension" choice (genuine 2D vs. pseudo-2D) -- that's a
        structural interpretation of the data, not a processing
        recipe, so it's carried forward from whatever it's currently
        set to rather than reset to ``_load``'s own pseudo-2D-by-
        default snapshot. Real user report: Reset on a genuine-2D
        EXSY dataset was silently flattening it back to pseudo-2D
        along with discarding the window/zero-fill/phase/truncate
        choices it was actually meant to discard -- confirmed by
        direct reproduction (not a "shows as a literal 1D" bug, but a
        real, understandable point of confusion: "Reset... defaults it
        to a 1D" was describing pseudo-2D's own per-slice treatment,
        not a genuine dimensionality collapse)."""
        index = self._selected_index()
        if index is None:
            return
        spectrum = self.session.spectra[index]
        original = dict(self._original_load_kwargs.get(index, {}))
        if spectrum.ndim == 2:
            original["pseudo_2d"] = spectrum.is_pseudo_2d
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
        """``metadata.raw``'s own already-committed F2 phase
        (``"phase0"``/``"phase1"``), as ``read_1d``/``read_2d``'s own
        ``phase0``/``phase1``/``pivot_frac``/``autophase=False`` kwargs
        -- so a raw-FID reload (currently only
        :meth:`_on_process_indirect_toggled`, which always commits any
        pending live delta into ``metadata.raw`` via
        :meth:`_commit_pending_phase_to_metadata` immediately before
        calling this) reproduces the spectrum's own already-established
        F2 phase instead of resetting to the vendor's own default.

        Always reads ``metadata.raw`` directly, never a live UI control
        -- real bug this fixes: an earlier version only forwarded F2's
        phase when the Phase panel's own Dimension combo happened to be
        currently focused on F2, silently discarding it on every
        "Process Indirect Dimension" toggle whenever F1 was focused
        instead (a dataset's own established F2 phase has nothing to
        do with which dimension the UI happens to be showing right
        now). Confirmed directly against real 2D data.

        ``pivot_frac`` is always centre (0.5) -- this project doesn't
        persist *which* pivot ``phase0``/``phase1`` were originally
        computed against (only the live ``pivot_spin``, reset to
        centre on every :meth:`_capture_phase_reference`), so there's
        nothing more specific to reproduce here; only matters when
        ``phase1`` is nonzero and the pivot was deliberately moved
        off-centre for that correction.

        Returns ``{}`` (no override -- vendor default applies) when
        ``spectrum`` isn't actually being re-read from a raw FID at
        all (e.g. a Bruker experiment with no ``fid``/``ser`` on disk
        -- routed through ``read_1d_processed``/``read_2d_processed``,
        which has no ``phase0``/``phase1`` concept at all), or when
        F2's own phase is exactly zero (nothing to override)."""
        if not self._fid_reprocessable(spectrum):
            return {}
        baked_in = spectrum.metadata.raw
        phase0 = float(baked_in.get("phase0", 0.0))
        phase1 = float(baked_in.get("phase1", 0.0))
        if not phase0 and not phase1:
            return {}
        return {"phase0": phase0, "phase1": phase1, "pivot_frac": 0.5, "autophase": False}

    def _reapply_phase_post_reload(self, spectrum) -> None:
        """Companion to :meth:`_current_phase_kwargs`: for the two
        cases it can't forward into the reload itself -- one pseudo-2D
        slice's own phase, or a conventional 2D's F1/indirect phase --
        reapply that same phase directly to the freshly-reloaded
        spectrum post-hoc, so at least what's actually being looked at
        keeps its phase instead of silently resetting. Each of the two
        cases below is a self-contained, independent ``if`` (not
        ``elif``) -- both are no-ops when their own preconditions
        aren't met, e.g. when the reload changed what the phase target
        even means (switching to pseudo-2D drops F1's frequency
        meaning entirely, so the F1 block below never fires for that
        direction) -- rather than one sharing a single early ``return``
        that could skip a case still worth checking.

        The F1 block reads ``spectrum.metadata.raw["phase0_f1"/
        "phase1_f1"]`` directly -- ``read_2d`` always builds a brand
        new ``metadata.raw`` from scratch on every reload with no F1-
        phase concept of its own (see :meth:`_current_phase_kwargs`'s
        own docstring), so the caller (:meth:`_on_process_indirect_toggled`)
        must carry these two keys over from the pre-reload spectrum's
        own ``metadata.raw`` into this (post-reload) one *before*
        calling this method, or there is nothing here to read."""
        if spectrum.ndim != 2 or spectrum.data_imag is None:
            return

        # pseudo-2D single-slice phase: never durably committed to
        # metadata.raw at all (see _commit_pending_phase_to_metadata's
        # own docstring for why -- it describes one row, not the whole
        # dataset) -- can only be recovered from the live spin boxes,
        # which is safe here specifically because this branch only
        # ever fires when _phase_mode == "pseudo2d" was already the
        # active session immediately before this toggle (no risk of
        # picking up a stale value left over from editing some other
        # dimension, unlike the F1 case below).
        if (self._phase_mode == "pseudo2d" and not self.apply_all_slices_checkbox.isChecked()
                and spectrum.is_pseudo_2d):
            phase0 = self.phase0_spin.value()
            phase1 = self.phase1_spin.value()
            row = self.slice_spin.value() - 1
            if (phase0 or phase1) and 0 <= row < spectrum.data.shape[0]:
                freq_ppm = spectrum.axis_values(-1, "ppm")
                span = freq_ppm.max() - freq_ppm.min()
                pivot_frac = (self.pivot_spin.value() - freq_ppm.min()) / span if span else 0.5
                complex_row = spectrum.data[row] + 1j * spectrum.data_imag[row]
                phased_row = processing_ft.apply_phase(complex_row, freq_ppm, phase0, phase1, pivot_frac=pivot_frac)
                spectrum.data[row] = np.real(phased_row)
                spectrum.data_imag[row] = np.imag(phased_row)

        # F1/indirect phase: always reapplied for a genuine-2D target,
        # regardless of which dimension the Phase panel happens to be
        # currently focused on -- real bug this fixes (mirrors
        # _current_phase_kwargs's own F2 fix, see its docstring):
        # previously gated on dim_combo.currentIndex() != 0, silently
        # dropping F1's own already-established phase on every toggle
        # whenever F2 was the one currently focused instead. No live
        # spin-box delta added on top -- nothing is left pending by the
        # time this runs (see _on_process_indirect_toggled's own
        # up-front commit). pivot_frac is always centre, same reasoning
        # as _current_phase_kwargs's own docstring (no per-dimension
        # pivot is persisted anywhere; the live pivot_spin could
        # reflect whichever dimension was most recently active, not
        # necessarily F1).
        if not spectrum.is_pseudo_2d:
            baked_in = spectrum.metadata.raw
            phase0 = float(baked_in.get("phase0_f1", 0.0))
            phase1 = float(baked_in.get("phase1_f1", 0.0))
            if phase0 or phase1:
                freq_ppm = spectrum.axis_values(0, "ppm")
                phased = processing_ft.apply_phase(
                    spectrum.complex_data.T, freq_ppm, phase0, phase1, pivot_frac=0.5,
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

        self.phase0_handle, self.phase0_spin, self.phase0_current_display = self._make_phase_control()
        # Phase 1 (linear, applied across the whole spectrum) needs a
        # much larger per-step change than Phase 0 to have a visible
        # effect -- one degree of phase1 barely moves anything, unlike
        # one degree of phase0. ~20x (1-2 orders of magnitude) larger
        # step/drag-sensitivity, per explicit user feedback.
        self.phase1_handle, self.phase1_spin, self.phase1_current_display = self._make_phase_control(
            sensitivity=0.2 * _PHASE1_STEP_MULTIPLIER, step=1.0 * _PHASE1_STEP_MULTIPLIER,
        )
        form.addRow("Phase 0 (deg):", self._pair_widget(self.phase0_handle, self.phase0_current_display))
        form.addRow("Phase 1 (deg):", self._pair_widget(self.phase1_handle, self.phase1_current_display))

        quick_phase_column = QVBoxLayout()
        quick_phase_column.setSpacing(2)
        # +180 and -180 nudge by the identical amount (phase is 360-
        # degree periodic) -- real user feedback that having both was
        # pointless, one "180°" button (no sign, since neither is more
        # correct than the other) replaces them.
        for delta in (180.0, 90.0, -90.0):
            label = f"{delta:+.0f}°" if delta != 180.0 else "180°"
            quick_button = QPushButton(label)
            quick_button.setToolTip(
                f"Nudge Phase 0 by {delta:.0f} degrees from its current value -- "
                "a quick fix for a spectrum that's fully inverted (180) or "
                "rotated a quarter-turn (90). Composes with whatever's already "
                "there (two +90 clicks equal one +180), same as typing/dragging "
                "Phase 0 by hand."
            )
            quick_button.clicked.connect(lambda _checked=False, d=delta: self._on_phase0_nudge_clicked(d))
            quick_phase_column.addWidget(quick_button)
        form.addRow("Quick phase:", quick_phase_column)

        zero_phase_row = QHBoxLayout()
        zero_phase0_button = QPushButton("Zero Phase 0")
        zero_phase0_button.setToolTip("Zero just Phase 0 (0th order), leaving Phase 1 untouched.")
        zero_phase0_button.clicked.connect(lambda: self._on_zero_phase_clicked(0))
        zero_phase_row.addWidget(zero_phase0_button)
        zero_phase1_button = QPushButton("Zero Phase 1")
        zero_phase1_button.setToolTip("Zero just Phase 1 (1st order), leaving Phase 0 untouched.")
        zero_phase1_button.clicked.connect(lambda: self._on_zero_phase_clicked(1))
        zero_phase_row.addWidget(zero_phase1_button)
        form.addRow(zero_phase_row)

        reset_button = QPushButton("Reset")
        reset_button.setToolTip("Zero both Phase 0 and Phase 1 together.")
        reset_button.clicked.connect(self._on_reset_phase_clicked)
        form.addRow(reset_button)

        # Only meaningful for an extracted slice, so hidden otherwise
        # (see _refresh_selection_display). A slice is a working copy of
        # one row/column of a 2D: phase it on a single trace, where the
        # lineshape is actually readable, then push that phase into the
        # whole dataset.
        self.apply_slice_phase_button = QPushButton("Apply Phase to Source 2D")
        self.apply_slice_phase_button.setToolTip(
            "Add the phase dialled in on this slice to the 2D it was extracted "
            "from, along the matching dimension, and switch back to it. Phasing "
            "a single trace is far easier to judge than a contour."
        )
        self.apply_slice_phase_button.clicked.connect(self._on_apply_slice_phase_to_source)
        self.apply_slice_phase_button.setVisible(False)
        form.addRow(self.apply_slice_phase_button)

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
        form.addRow("SR (direct):", self.sr_shift_spin)

        # The indirect axis needs its own SR, not a share of F2's: it
        # has its own base frequency (metadata.sfo1_indirect) and, on a
        # heteronuclear 2D, its own nucleus entirely, so one Hz offset
        # cannot reference both. Shown only when there is an indirect
        # frequency axis to reference -- hidden for 1D, and for a
        # pseudo-2D, whose indirect axis is an index (slice number),
        # not a frequency.
        self.sr_shift_indirect_spin = QDoubleSpinBox()
        self.sr_shift_indirect_spin.setRange(-1.0e6, 1.0e6)
        self.sr_shift_indirect_spin.setDecimals(4)
        self.sr_shift_indirect_spin.setSuffix(" Hz")
        self.sr_indirect_label = QLabel("SR (indirect):")
        form.addRow(self.sr_indirect_label, self.sr_shift_indirect_spin)

        apply_button = QPushButton("Apply Reference Shift")
        apply_button.clicked.connect(self._on_apply_sr_clicked)
        form.addRow(apply_button)

        return box

    def _make_phase_control(self, sensitivity: float = 0.2, step: float = 1.0) -> tuple[DragHandle, QDoubleSpinBox, QLabel]:
        handle = DragHandle(sensitivity=sensitivity, step=step)

        # Never shown -- kept purely as the internal drag *delta*
        # value DragHandle stays synced with (_on_phase_handle_changed/
        # _on_phase_spin_changed, both otherwise unchanged), so every
        # existing computation that reads this spin box's own .value()
        # keeps working exactly as before. What the user actually sees
        # is current_display below instead: a plain non-editable
        # readout of the *absolute* phase (see
        # _refresh_current_phase_displays). Explicit user feedback:
        # "just display non-editable numbers with current phase
        # instead of that editable box... we don't need to see the
        # zeroed out phase values" -- the editable, always-0-after-
        # reselection number was the actual complaint, not the drag
        # handle itself, which keeps working exactly as before.
        spin = RelativeDoubleSpinBox()
        spin.setRange(-_PHASE_SPIN_RANGE, _PHASE_SPIN_RANGE)
        spin.setDecimals(2)
        spin.setSingleStep(step)
        spin.setValue(0.0)

        current_display = QLabel("—")
        # Same reasoning the old editable spin box's own minimumWidth
        # had -- real recorded phase1 values can run into 4 digits
        # (e.g. -1748.5), too wide for Qt's own default label sizing.
        current_display.setMinimumWidth(80)
        current_display.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        current_display.setToolTip(
            "The total phase actually applied right now -- drag the "
            "handle to adjust it. Always the real value, never a delta "
            "that resets to 0 on reselection."
        )

        handle.valueChanged.connect(lambda v: self._on_phase_handle_changed(handle, spin, v))
        spin.valueChanged.connect(lambda v: self._on_phase_spin_changed(handle, spin, v))
        return handle, spin, current_display

    @staticmethod
    def _pair_widget(handle: DragHandle, display: QWidget) -> QWidget:
        widget = QWidget()
        row = QHBoxLayout(widget)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(handle)
        row.addWidget(display)
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

    def _recompute_from_raw_if_possible(self, spectrum) -> bool:
        """Rebuild ``spectrum`` from its cached raw FID, so a change to
        something applied *during* reconstruction (the recorded phase)
        actually takes effect. ``False`` when there is no raw FID to
        rebuild from, leaving the caller to fall back to a plain
        preview."""
        if spectrum is None or not self._fid_reprocessable(spectrum):
            return False
        if spectrum.ndim == 1:
            self._commit_processing_preview_as_spectrum(spectrum)
        else:
            self._recompute_multi_row_data(spectrum)
        index = self._selected_index()
        if index is not None:
            self._draw_selected_spectrum(spectrum, row=index)
            self._schedule_sidecar_save(index)
        self._refresh_current_phase_displays()
        return True

    def _on_reset_phase_clicked(self) -> None:
        """Zero both Phase 0 and Phase 1 together -- the same "actually
        zero the applied phase, not just the drag delta" behaviour as
        :meth:`_on_zero_phase_clicked`, which see for why."""
        self._on_zero_phase_clicked(0)
        self._on_zero_phase_clicked(1)

    def _phase_target_dim_key(self) -> str:
        """``"f1"`` or ``"f2"`` -- which dimension the Phase controls
        are currently editing. One accessor, so every phase helper
        agrees about the target rather than each re-deriving it."""
        return "f1" if (self._phase_mode == "2d" and self.dim_combo.currentIndex() != 0) else "f2"

    def _phase_target_spectrum(self):
        """The spectrum the Phase controls act on, or ``None``."""
        index = self._phase_target_index
        if self._phase_mode is None or index is None or not (0 <= index < len(self.session)):
            return None
        return self.session.spectra[index]

    def _phase_committed_value(self, order: int) -> float:
        """The phase already applied to whatever the controls are
        currently targeting -- the other half of the total that
        :meth:`_refresh_current_phase_displays` shows (that total is
        this plus the live spin-box delta). ``0.0`` when there is no
        valid target, so callers don't have to re-check.

        Includes **both** halves: the file's own recorded phase and the
        user's own. They are stored separately and applied through
        different models (see ``io.bruker.apply_dimension_phase``), but
        what is "already applied" -- which is what the display shows and
        what the Zero buttons must cancel -- is their sum.
        """
        spectrum = self._phase_target_spectrum()
        if spectrum is None:
            return 0.0
        dim = self._phase_target_dim_key()
        raw = spectrum.metadata.raw
        user_key = io_bruker.USER_PHASE_KEYS[dim][order]
        return float(raw.get(user_key, 0.0) or 0.0) + io_bruker.recorded_phase(raw, dim)[order]

    def _on_zero_phase_clicked(self, order: int) -> None:
        """Zero just Phase 0 (``order=0``) or just Phase 1 (``order=1``),
        leaving the other untouched -- real user request, distinct from
        :meth:`_on_reset_phase_clicked` (the combined Reset button,
        which zeros both together).

        Zeros the phase **actually applied**, not merely the live drag
        delta. Real user report: "Zeroing phases should actually zero
        the values, not just reset to the loaded values." The spin box
        holds only the delta since the last
        :meth:`_capture_phase_reference`; the total in force is that
        delta *plus* whatever is committed in ``metadata.raw`` (which,
        on a freshly loaded Bruker spectrum, is the file's own recorded
        ``PHC0``/``PHC1``). Setting the delta to 0 therefore used to
        return to the *loaded* phase rather than to zero -- visibly
        doing nothing at all on a spectrum that hadn't been dragged
        yet, which is exactly what was reported.

        Setting the delta to ``-committed`` instead drives the total to
        0 while leaving every existing mechanism untouched: the same
        preview, commit and persistence paths run as for any other
        phase edit, rather than this reaching into ``metadata.raw``
        behind their back and having to re-derive what they already
        handle.
        """
        spectrum = self._phase_target_spectrum()
        if spectrum is not None:
            # Clear the recorded half outright rather than trying to
            # cancel it with an equal-and-opposite user phase: the two
            # use different models (index ramp vs pivot), so for
            # first-order phase an opposing user value would *not*
            # actually cancel it. Zeroing means "no phase", so remove
            # it at its source.
            dim = self._phase_target_dim_key()
            spectrum.metadata.raw[io_bruker.RECORDED_PHASE_KEYS[dim][order]] = 0.0
            spectrum.metadata.raw[io_bruker.USER_PHASE_KEYS[dim][order]] = 0.0
        handle, spin = (self.phase0_handle, self.phase0_spin) if order == 0 else (self.phase1_handle, self.phase1_spin)
        handle.setValue(0.0, emit=False)
        spin.blockSignals(True)
        spin.setValue(0.0)
        spin.blockSignals(False)
        # Rebuild rather than just re-previewing: the recorded phase is
        # applied during reconstruction from the raw FID, so clearing it
        # only takes effect once that rebuild runs.
        if not self._recompute_from_raw_if_possible(spectrum):
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
        # Discrete, one-shot action (a button click, not a continuous
        # drag) -- an immediate contour refresh is fine here, same as
        # _run_debounced_multi_row_reprocess's own settled-event
        # refresh; no need for _schedule_canvas_2d_refresh's own
        # per-tick debounce.
        on_processing_preview = self.canvas_tabs.currentWidget() is self.processing_preview_canvas
        self.canvas.show_spectrum(spectrum)
        if not on_processing_preview:
            self.canvas_tabs.setCurrentWidget(self.canvas)
        else:
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

    @staticmethod
    def _f2_quadrature_is_spent(spectrum) -> bool:
        """Whether ``spectrum``'s stored ``data``/``data_imag`` pair is
        F1's quadrature rather than F2's -- true for a genuine
        hypercomplex 2D (``FnMODE`` 4 States, 5 States-TPPI, 6
        Echo-Antiecho), false for everything else.

        The hypercomplex step *consumes* each row's F2 imaginary part
        to build F1's (see
        :func:`opennmr.processing.ft.decode_hypercomplex_f1`), so after
        the F1 transform what survives is TopSpin's ``2rr`` and
        ``2ri``: F1's pair. That is exactly what F1 phasing wants and
        exactly what F2 phasing cannot use.

        Why this needs saying at all: the *previous* decode combined
        whole complex rows (``a + 1j*b``), which is C-linear, so a
        global F2 phase commuted straight through it and through the
        F1 FFT -- phasing F2 on the finished array gave the same answer
        as phasing the rows first. That accidental equivalence is
        precisely what let F2's dispersion leak into F1 and produce the
        hourglass lineshape. The corrected decode is only R-linear, so
        the equivalence is gone -- correctly, but it means an F2 phase
        can no longer be previewed from the finished array and has to
        be applied before the decode, i.e. by a rebuild from the raw
        FID.

        Not done instead: keeping the second hypercomplex plane
        (TopSpin's ``2ir``/``2ii``) so F2 could be phased from four
        stored planes the way TopSpin does it. That is the textbook
        answer and it is rejected on memory -- the EXSY benchmark's
        decoded array is already ~1 GB, and this project caps contour
        points at a quarter of that same dataset for the same reason.
        """
        return (
            not spectrum.metadata.pseudo_2d
            and spectrum.ndim == 2
            and int(spectrum.metadata.raw.get("fn_mode", 1)) in (4, 5, 6)
        )

    def _apply_phase_preview(self) -> None:
        index = self._selected_index()
        if index is None or self._phase_mode is None:
            return

        spectrum = self.session.spectra[index]
        # Only the part not already baked into the reference. A
        # debounced rebuild that lands mid-drag replaces the reference
        # with its own (baseline-corrected) result and records the
        # phase that result already carries -- without subtracting it
        # here, the next drag tick would re-apply the whole phase to
        # data that already had it *and* silently discard the baseline,
        # so the display flipped between baselined and not on every
        # pause. Real user report: "The indirect phasing sort of
        # bounces like there is a mismatch to the current and read
        # phases."
        applied0, applied1 = self._phase_reference_delta_applied
        phase0 = self.phase0_spin.value() - applied0
        phase1 = self.phase1_spin.value() - applied1
        self._refresh_current_phase_displays()

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
            # Real, repeated user requirement: "the Spectrum tab should
            # always show the 2D or pseudo 2D where there is one" --
            # self.canvas no longer ever switches to a single-slice
            # line for phase editing (that was the "1D slice for
            # judging phase" idea an earlier round of this same
            # feedback asked for; superseded by this one, which asks
            # for the opposite). The Processing tab's own 1D preview
            # (already slice-based, see _compute_processing_preview/
            # _show_2d_indirect_processing_preview) is now the *only*
            # place a judgeable 1D slice is shown during 2D phase
            # editing -- "the processing tab should only have a 1D
            # that is matched to the fid tab" (for a pseudo-2D, both
            # already key off the same slice_spin row). self.canvas
            # itself just gets a debounced contour refresh -- see
            # _schedule_canvas_2d_refresh's own docstring for why not
            # an immediate one on every tick.
            if self.canvas_tabs.currentWidget() is self.fid_canvas and self._fid_preview_available(spectrum):
                self._show_fid_preview(spectrum, switch_tab=False)
            elif self.canvas_tabs.currentWidget() is self.processing_preview_canvas:
                self._refresh_processing_preview_phase_only()
            else:
                self.canvas_tabs.setCurrentWidget(self.canvas)
            self._schedule_canvas_2d_refresh(index)

        elif self._phase_mode == "2d":
            dim = "f2" if self.dim_combo.currentIndex() == 0 else "f1"
            if dim == "f2" and self._f2_quadrature_is_spent(spectrum):
                # Nothing to preview from: for a hypercomplex 2D the
                # stored complex pair is F1's quadrature, not F2's, so
                # rotating it by an F2 phase is not an F2 phase
                # correction. Defer the whole tick to the debounced
                # rebuild, which goes back to the raw FID and phases F2
                # in the one place it is still meaningful -- before the
                # hypercomplex decode. See _f2_quadrature_is_spent.
                self._schedule_multi_row_reprocess(index)
            else:
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
                # self.canvas never switches to a single-slice line for
                # phase editing any more -- see the matching comment in
                # the "pseudo2d" branch above for the full reasoning
                # ("the Spectrum tab should always show the 2D or
                # pseudo 2D where there is one"). Mirrors that branch's
                # own three-way canvas_tabs handling, including
                # deferring the FID/Processing-preview tabs to their own
                # already-dimension-generic refresh methods
                # (_show_fid_preview is a no-op for a genuine 2D either
                # way, gated off by _fid_preview_available;
                # _refresh_processing_preview_phase_only routes to
                # _compute_processing_preview or
                # _show_2d_indirect_processing_preview depending on
                # dim_combo -- see their own docstrings for why the two
                # dimensions can't share one implementation).
                if self.canvas_tabs.currentWidget() is self.fid_canvas and self._fid_preview_available(spectrum):
                    self._show_fid_preview(spectrum, switch_tab=False)
                elif self.canvas_tabs.currentWidget() is self.processing_preview_canvas:
                    self._refresh_processing_preview_phase_only()
                else:
                    self.canvas_tabs.setCurrentWidget(self.canvas)
                self._schedule_canvas_2d_refresh(index)
            except Exception:  # noqa: BLE001 -- see comment above
                pass

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
        way to tell it had ever been touched.

        Uses :attr:`_phase_target_dim` (captured once when this
        session's own target was established), not a live read of
        ``dim_combo.currentIndex()`` -- real bug, found and fixed:
        when this method runs because the *dimension* just changed
        (``dim_combo.currentIndexChanged`` -> ``_on_phase_target_changed``
        -> ``_capture_phase_reference`` -> here, all before this
        session's own new target is established), Qt has already
        updated ``dim_combo.currentIndex()`` to the *new* dimension by
        the time this runs -- a live read would silently commit a
        pending F2 delta under the F1 key (or vice versa) on every
        dimension switch. Confirmed directly against real 2D data."""
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
        is_f1 = self._phase_mode == "2d" and self._phase_target_dim == "f1"
        key0, key1 = ("phase0_f1", "phase1_f1") if is_f1 else ("phase0", "phase1")
        # Zero-order phase is cyclic (a full rotation, exp(i*phase0) --
        # 360deg away is physically identical) -- wrapped into [0, 360)
        # on commit so it doesn't grow unbounded across many separate
        # drag sessions. Phase 1 is deliberately left unwrapped: it's a
        # ramp *rate* across the spectrum, not a rotation, so "modulo
        # 360" has no equivalent physical meaning for it.
        spectrum.metadata.raw[key0] = (float(spectrum.metadata.raw.get(key0, 0.0)) + phase0_delta) % 360.0
        spectrum.metadata.raw[key1] = float(spectrum.metadata.raw.get(key1, 0.0)) + phase1_delta
        self._schedule_sidecar_save(index)

    def _refresh_current_phase_displays(self) -> None:
        """Show the *total* phase actually applied right now, in
        ``phase0_current_display``/``phase1_current_display`` (the
        non-editable labels next to each drag handle -- see
        :meth:`_make_phase_control`) -- whatever's already baked into
        ``metadata.raw`` for whatever this editing session is
        targeting, plus the live delta still held in the hidden
        ``phase0_spin``/``phase1_spin`` (0 right after a fresh
        :meth:`_capture_phase_reference`, non-zero mid-drag). Called
        from both, so these stay accurate whether nothing is being
        dragged right now or a drag is actively in progress."""
        index = self._phase_target_index
        if self._phase_mode is None or index is None or not (0 <= index < len(self.session)):
            self.phase0_current_display.setText("—")
            self.phase1_current_display.setText("—")
            return
        spectrum = self.session.spectra[index]
        is_f1 = self._phase_mode == "2d" and self.dim_combo.currentIndex() != 0
        key0, key1 = ("phase0_f1", "phase1_f1") if is_f1 else ("phase0", "phase1")
        # The file's own recorded phase counts as "already applied" too
        # -- see _phase_committed_value. Added here rather than merged
        # into the stored phase0/phase1, which mean the user's own.
        recorded0, recorded1 = io_bruker.recorded_phase(spectrum.metadata.raw, "f1" if is_f1 else "f2")
        # Wrapped the same way _commit_pending_phase_to_metadata wraps
        # the stored value -- keeps this display cyclic continuously
        # during an active drag too (e.g. a committed 350deg plus a
        # live +30deg delta shows as 20deg, not 380deg), not just after
        # the drag ends and commits.
        phase0 = (float(spectrum.metadata.raw.get(key0, 0.0)) + recorded0 + self.phase0_spin.value()) % 360.0
        phase1 = float(spectrum.metadata.raw.get(key1, 0.0)) + recorded1 + self.phase1_spin.value()
        self.phase0_current_display.setText(f"{phase0:.2f}°")
        self.phase1_current_display.setText(f"{phase1:.2f}°")

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
            self._phase_target_dim = None
            self._phase_reference_full = None
            self._phase_reference_delta_applied = (0.0, 0.0)
            self._phase_freq_ppm = None
            self._refresh_current_phase_displays()
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
            self._phase_target_dim = None
            self._phase_reference_full = None
            self._phase_reference_delta_applied = (0.0, 0.0)
            self._phase_freq_ppm = None
            self._set_processing_enabled(False, "Phase adjustment needs a spectrum with retained imaginary data.")
            self._refresh_current_phase_displays()
            return

        if spectrum.ndim == 1:
            self._phase_mode = "1d"
            self._phase_reference_full = spectrum.complex_data
            self._phase_reference_delta_applied = (0.0, 0.0)
            self._phase_freq_ppm = spectrum.axis_values(0, "ppm")
            self._phase_target_dim = None
            self._set_processing_enabled(True, "Adjust phase for the selected spectrum.")

        elif spectrum.ndim == 2 and spectrum.is_pseudo_2d:
            self._phase_mode = "pseudo2d"
            self._phase_reference_full = spectrum.complex_data
            self._phase_reference_delta_applied = (0.0, 0.0)
            self._phase_freq_ppm = spectrum.axis_values(-1, "ppm")
            self._phase_target_dim = None
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
            self._phase_reference_delta_applied = (0.0, 0.0)
            dim = "f2" if self.dim_combo.currentIndex() == 0 else "f1"
            self._phase_target_dim = dim
            self._phase_freq_ppm = spectrum.axis_values(-1 if dim == "f2" else 0, "ppm")
            # _phase_2d_slice_index picks which row/column the
            # Processing tab's own 1D preview shows while phasing a
            # genuine 2D (_compute_processing_preview for F2/direct,
            # _show_2d_indirect_processing_preview for F1/indirect) --
            # self.canvas itself no longer renders a slice of its own
            # at all any more (see _apply_phase_preview's own
            # docstring: "the Spectrum tab should always show the 2D or
            # pseudo 2D where there is one"). The correction itself
            # still applies uniformly to the *whole* array regardless
            # (a genuine-2D phase is one shared correction across every
            # row/column, not a per-slice one the way pseudo-2D's own
            # "not every slice" mode is) -- only which row/column gets
            # *shown* is what this index controls.
            #
            # Always slice 1 -- explicit request ("the processing
            # should just always use slice 1 of the 'xf2' (unprocessed
            # indirect)"), replacing an earlier largest-total-magnitude
            # pick. That pick needed the *processed* F1 axis to even
            # evaluate (magnitude.sum along the just-FFT'd array), which
            # is exactly what made the raw-vs-processed row-count
            # mismatch this session's own earlier fixes had to work
            # around (_phase_2d_proportional_row) -- always using index
            # 1 sidesteps that class of bug entirely: it's a valid
            # position in *both* the raw and the processed array (index
            # 0 either way), no mapping needed.
            self._phase_2d_slice_index = 1
            dim_label = "F2 (direct)" if dim == "f2" else "F1 (indirect)"
            self._set_processing_enabled(True, f"Adjust phase along {dim_label} for the whole 2D spectrum.")

        else:
            self._phase_mode = None
            self._phase_target_dim = None
            self._phase_reference_full = None
            self._phase_reference_delta_applied = (0.0, 0.0)
            self._phase_freq_ppm = None
            self._set_processing_enabled(False, "Phase adjustment needs a spectrum with retained imaginary data.")
            self._refresh_current_phase_displays()
            return

        self._refresh_current_phase_displays()
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
            self.phase0_handle, self.phase0_spin, self.phase0_current_display,
            self.phase1_handle, self.phase1_spin, self.phase1_current_display,
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

        # Which slice of a multi-row spectrum peaks are picked on.
        # Real user report: "peak picking on a pseudo 2D is currently
        # impossible" -- and it was: manual picking listens on the
        # Spectrum canvas, which shows the contour/stack for a
        # pseudo-2D, so a click there yields a 2D position with no
        # single trace to attach a peak to, and Auto-Pick was hard-wired
        # to slice 1 with no way to choose another.
        self.peak_slice_row = QWidget()
        peak_slice_layout = QHBoxLayout(self.peak_slice_row)
        peak_slice_layout.setContentsMargins(0, 0, 0, 0)
        peak_slice_layout.addWidget(QLabel("Pick on slice:"))
        self.peak_slice_spin = QSpinBox()
        self.peak_slice_spin.setRange(1, 1)
        # Wraps, per explicit user request: "if there's a box with a
        # number, pressing down on the lowest should cycle back to the
        # highest".
        self.peak_slice_spin.setWrapping(True)
        self.peak_slice_spin.setToolTip(
            "Which slice of a 2D/pseudo-2D series to pick peaks on -- "
            "both 'Pick from Plot' (which shows this slice on its own "
            "so there is a 1D trace to click) and 'Auto-Pick Peaks'. "
            "Wraps around at either end."
        )
        self.peak_slice_spin.valueChanged.connect(self._on_peak_slice_changed)
        peak_slice_layout.addWidget(self.peak_slice_spin)
        peak_slice_layout.addStretch(1)
        peak_layout.addWidget(self.peak_slice_row)

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
            "current selection, or on the chosen slice of a 2D/"
            "pseudo-2D series) and "
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

        seed_mode_row = QHBoxLayout()
        seed_mode_row.addWidget(QLabel("Seed each slice from:"))
        self.seed_mode_combo = QComboBox()
        self.seed_mode_combo.addItem("Table above (every slice independent)", "independent")
        self.seed_mode_combo.addItem("Previous slice's own result", "propagate")
        self.seed_mode_combo.addItem("First slice's result (shift/FWHM fixed)", "apply_first")
        self.seed_mode_combo.setToolTip(
            "How each slice after the first gets its own starting shift/"
            "FWHM/eta across a pseudo-2D series (never amplitude, which "
            "is expected to vary -- that's the point of a relaxation/"
            "diffusion series). 'Table above' (default): every slice "
            "starts from the same guess entered above, fit independently "
            "-- can fail to track a peak that drifts noticeably across "
            "the series. 'Previous slice's own result': each slice "
            "seeds the next with its own converged position/width -- "
            "follows a slow drift, or gives a weak later slice a much "
            "closer starting point. 'First slice's result': shift/FWHM "
            "(and eta) are fixed at the first slice's own converged "
            "values for every slice after it -- only amplitude is still "
            "fit; fastest, and most robust for a peak whose shape "
            "genuinely shouldn't change across the series."
        )
        seed_mode_row.addWidget(self.seed_mode_combo)
        advanced_layout.addLayout(seed_mode_row)

        self.amplitude_bound_regions_checkbox = QCheckBox("Bound amplitude using region integrals")
        self.amplitude_bound_regions_checkbox.setToolTip(
            "For each peak whose starting shift falls inside one of the "
            "Integration Regions above, cap its amplitude using that "
            "region's own raw (unfitted) integral -- converted via the "
            "peak's lineshape/FWHM -- instead of leaving it unbounded "
            "above. Recomputed fresh per slice, so it tracks a genuinely "
            "varying true amplitude across a relaxation/diffusion series. "
            "Deliberately loose (a whole region's own integral, not a "
            "tight per-peak estimate) -- mainly useful to keep an "
            "unconstrained fit from inflating one peak's amplitude to "
            "compensate for an unmodelled overlapping neighbour. Has no "
            "effect unless at least one region is defined above, and "
            "never overrides a peak's own explicit amplitude bound (Set "
            "Bounds... dialog)."
        )
        advanced_layout.addWidget(self.amplitude_bound_regions_checkbox)

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

        export_report_button = QPushButton("Export Fit Report…")
        export_report_button.setToolTip(
            "Save every slice's fitted peak parameters (shift/FWHM/"
            "amplitude/eta, with uncertainties) as a CSV file -- plus "
            "the curve-fit (Fit Curve) result per peak, if one has "
            "been run for this spectrum."
        )
        export_report_button.clicked.connect(self._on_export_fit_report_clicked)
        layout.addWidget(export_report_button)

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

        # The gradient a Bruker experiment actually ran at is a
        # *percentage* of the probe's maximum, recorded as GPZ1 in
        # acqus. Real user report: "the relative gradient strength
        # doesn't seem to be calculated correctly for my Bruker data
        # (should be gpz1 generally as a percentage value, e.g. gpz1=2
        # is 2%)". It was not being read at all -- the mm axis used the
        # full probe maximum, so at gpz1=2 the scale came out 50x too
        # small. Prefilled from GPZ1 but editable, since "generally"
        # is not "always": another pulse programme may put the imaging
        # gradient on a different GPZ index.
        self.imaging_gradient_percent_spin = QDoubleSpinBox()
        self.imaging_gradient_percent_spin.setRange(0.0, 100.0)
        self.imaging_gradient_percent_spin.setDecimals(3)
        self.imaging_gradient_percent_spin.setSuffix(" %")
        self.imaging_gradient_percent_spin.setToolTip(
            "Gradient actually used, as a percentage of the maximum "
            "above -- Bruker's own GPZ1 from acqus, which is read "
            "automatically. Override it if this experiment's pulse "
            "programme puts its imaging gradient on a different "
            "gradient channel."
        )
        self.imaging_gradient_percent_spin.valueChanged.connect(self._on_imaging_gradient_changed)
        gradient_form.addRow("Gradient used:", self.imaging_gradient_percent_spin)

        save_probe_gradient_button = QPushButton("Save as default for this probe")
        save_probe_gradient_button.setToolTip(
            "Remember this max gradient value for this spectrum's own recorded probe, "
            "app-wide -- any other spectrum (any experiment) recording the same probe "
            "and with no max gradient of its own already set will use it automatically."
        )
        save_probe_gradient_button.clicked.connect(self._on_save_probe_gradient_default_clicked)
        gradient_form.addRow(save_probe_gradient_button)

        self.imaging_gradient_status_label = QLabel("Select a spectrum to enable the mm axis.")
        self.imaging_gradient_status_label.setWordWrap(True)
        gradient_form.addRow(self.imaging_gradient_status_label)

        layout.addWidget(gradient_box)
        layout.addStretch(1)
        return tab

    def _on_imaging_gradient_changed(self, _value: float) -> None:
        index = self._selected_index()
        if index is not None:
            raw = self.session.spectra[index].metadata.raw
            raw["imaging_max_gradient_g_per_cm"] = self.imaging_max_gradient_spin.value() or None
            raw["imaging_gradient_percent"] = self.imaging_gradient_percent_spin.value()
            self._schedule_sidecar_save(index)
        self._refresh_imaging_gradient()

    @staticmethod
    def _recorded_gradient_percent(spectrum) -> float | None:
        """Bruker's ``GPZ1`` for ``spectrum``, as a percentage of the
        probe's maximum gradient -- ``acqus``'s ``GPZ`` array, index 1.
        ``None`` when there is no such array (a non-Bruker dataset, or
        one whose parameters weren't retained)."""
        gpz = (spectrum.metadata.raw.get("acqus") or {}).get("GPZ")
        if not isinstance(gpz, (list, tuple)) or len(gpz) < 2:
            return None
        try:
            return float(gpz[1])
        except (TypeError, ValueError):
            return None

    def _restore_imaging_gradient_percent(self, spectrum) -> None:
        """Put the gradient-percentage box back to this spectrum's own
        value -- its saved override, else ``GPZ1``, else 100% (the old
        behaviour, i.e. "the full maximum")."""
        stored = spectrum.metadata.raw.get("imaging_gradient_percent")
        if stored is None:
            recorded = self._recorded_gradient_percent(spectrum)
            stored = recorded if recorded else 100.0
        self.imaging_gradient_percent_spin.blockSignals(True)
        self.imaging_gradient_percent_spin.setValue(float(stored))
        self.imaging_gradient_percent_spin.blockSignals(False)

    def _refresh_imaging_gradient(self) -> None:
        """Recompute (and hand to the canvas) the gamma/max-gradient
        pair the ``"mm"`` axis-unit option needs, for whichever
        spectrum is currently selected -- called on every reselection
        (:meth:`_refresh_selection_display`) as well as every max-
        gradient edit, so switching spectra never leaves the mm axis
        showing a stale nucleus's conversion."""
        index = self._selected_index()
        # The *applied* gradient, not the probe's maximum: a Bruker
        # experiment runs at GPZ1 percent of it (see
        # _recorded_gradient_percent).
        percent = self.imaging_gradient_percent_spin.value()
        maximum = self.imaging_max_gradient_spin.value() or None
        max_gradient = (maximum * percent / 100.0) if (maximum and percent) else None
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
                       fixed: bool = False, bounds: dict | None = None,
                       per_param_fixed: dict | None = None) -> None:
        row = self.peak_table.rowCount()
        self.peak_table.insertRow(row)
        for col, value in enumerate((shift, fwhm, eta)):
            self.peak_table.setItem(row, col, QTableWidgetItem(f"{value:.4g}"))
        self.peak_table.setItem(row, 3, self._make_fixed_checkbox_item(fixed))
        if bounds:
            self.peak_table.item(row, 0).setData(Qt.ItemDataRole.UserRole, bounds)
        if per_param_fixed:
            self.peak_table.item(row, 0).setData(self._PEAK_FIXED_ITEM_ROLE, per_param_fixed)
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
        bounds_action = menu.addAction("Set Bounds / Fixed Parameters…")
        chosen = menu.exec(self.peak_table.viewport().mapToGlobal(pos))
        if chosen is bounds_action:
            self._on_set_peak_bounds(row)

    #: Parameter keys a peak-fit custom-bounds dialog edits, paired
    #: with the key ``fitting.build_bounds`` actually reads for each
    #: (amplitude is "A" in ``fixed``, but "amplitude" in ``bounds`` --
    #: see fitting.py's own build_bounds/initialize_params_with_map).
    _PEAK_BOUNDS_FIELDS = (("Shift (ppm)", "shift"), ("FWHM (ppm)", "fwhm"),
                           ("Amplitude", "amplitude"), ("Eta", "eta"))

    #: Second custom data role on a peak row's own column-0 item (see
    #: _on_set_peak_bounds) -- UserRole itself already holds this row's
    #: custom *bounds* dict; per-parameter *fixed* flags are stored
    #: alongside it here rather than folded into the same dict, so
    #: existing bounds-only code (persistence, _update_peak_table_from_fit)
    #: didn't need to change shape to gain this.
    _PEAK_FIXED_ITEM_ROLE = Qt.ItemDataRole.UserRole + 1

    def _on_set_peak_bounds(self, row: int) -> None:
        """A peak row's own "Fixed" checkbox (column 3) fixes *every*
        parameter together -- real user requirement: "individual
        parameters (not just an entire line) should be fixable." This
        dialog is the finer-grained tool for that: each field gets its
        own optional bounds text box *and* its own "Fixed" checkbox.
        The row's own all-or-nothing checkbox still wins outright when
        checked (see :meth:`_read_peak_defs`) -- leave it unchecked to
        let these per-parameter choices take effect instead."""
        existing_bounds = self.peak_table.item(row, 0).data(Qt.ItemDataRole.UserRole) or {}
        existing_fixed = self.peak_table.item(row, 0).data(self._PEAK_FIXED_ITEM_ROLE) or {}
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Bounds / fixed parameters -- peak {row + 1}")
        form = QFormLayout(dialog)
        edits = {}
        fixed_checks = {}
        for label, key in self._PEAK_BOUNDS_FIELDS:
            fixed_key = "A" if key == "amplitude" else key
            edit = QLineEdit()
            if key in existing_bounds:
                edit.setText(f"{existing_bounds[key][0]:g}, {existing_bounds[key][1]:g}")
            edit.setPlaceholderText("optional, 'low, high'")
            checkbox = QCheckBox("Fixed")
            checkbox.setChecked(bool(existing_fixed.get(fixed_key, False)))
            field_row = QWidget()
            field_layout = QHBoxLayout(field_row)
            field_layout.setContentsMargins(0, 0, 0, 0)
            field_layout.addWidget(edit)
            field_layout.addWidget(checkbox)
            form.addRow(f"{label}:", field_row)
            edits[key] = edit
            fixed_checks[fixed_key] = checkbox
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
        fixed = {key: True for key, checkbox in fixed_checks.items() if checkbox.isChecked()}
        self.peak_table.item(row, 0).setData(Qt.ItemDataRole.UserRole, bounds or None)
        self.peak_table.item(row, 0).setData(self._PEAK_FIXED_ITEM_ROLE, fixed or None)

    def _on_peak_picked(self, shift: float) -> None:
        if self.pick_peak_button.isChecked():
            # A reasonable default width: 1% of the currently-visible
            # x-range, since there's no per-peak width information in
            # a single click -- the fit refines it regardless.
            xmin, xmax = self.canvas.ax.get_xlim()
            fwhm_guess = abs(xmax - xmin) * 0.01 or 1.0
            self._add_peak_row(shift=shift, fwhm=fwhm_guess, eta=0.5)

    def _peak_pick_slice_index(self) -> int:
        return int(self.peak_slice_spin.value())

    def _refresh_peak_slice_control(self, spectrum) -> None:
        """Show the "Pick on slice" selector only where there is more
        than one row to choose between, and keep its range matched to
        the current data (a reprocess can change the row count)."""
        multi_row = spectrum is not None and spectrum.ndim == 2
        self.peak_slice_row.setVisible(bool(multi_row))
        if not multi_row:
            return
        n_rows = int(spectrum.data.shape[0])
        self.peak_slice_spin.blockSignals(True)
        self.peak_slice_spin.setRange(1, max(1, n_rows))
        if self.peak_slice_spin.value() > n_rows:
            self.peak_slice_spin.setValue(1)
        self.peak_slice_spin.blockSignals(False)

    def _peak_pick_target(self, spectrum):
        """The 1D spectrum peaks are picked on: the spectrum itself
        when it already is 1D, otherwise the selected slice. Replaces a
        hard-wired ``index=1``."""
        if spectrum.ndim == 1:
            return spectrum
        return processing_slicing.get_trace(spectrum, axis="f1", index=self._peak_pick_slice_index())

    def _on_peak_slice_changed(self, _value: int) -> None:
        # Only re-show the slice while picking is actually active --
        # otherwise changing the number would yank the user's contour
        # view away for no reason.
        if self.pick_peak_button.isChecked():
            self._show_peak_pick_slice()

    def _show_peak_pick_slice(self) -> None:
        """Put the chosen slice on the Spectrum canvas as its own 1D
        trace, so there is something to click. Reuses the canvas's
        existing extract-slice view (which brings its own "Back to
        Contour" bar), rather than inventing a second single-trace
        mode."""
        index = self._selected_index()
        if index is None:
            return
        spectrum = self.session.spectra[index]
        if spectrum.ndim != 2:
            return
        # Narrow, not a blanket swallow: only a stale/out-of-range
        # index is tolerated here (a reprocess can shrink the row count
        # between the spinner's range being set and this running). A
        # broad `except Exception` hid a real bug during development --
        # extract_slice raising because it could not find its source
        # spectrum at all -- and made peak-picking silently stick to
        # whichever slice it started on.
        try:
            self.canvas.extract_slice("f1", self._peak_pick_slice_index())
        except IndexError:
            pass

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
            # A 2D/pseudo-2D canvas has no 1D trace to click, which is
            # exactly why picking was impossible on one -- show the
            # chosen slice on its own for the duration.
            self._show_peak_pick_slice()

    def _on_auto_pick_peaks_clicked(self) -> None:
        """Replace the peak table with automatically-detected peaks
        (:func:`opennmr.processing.fitting.auto_pick_peaks`) from the
        current selection -- slice 1, for a pseudo-2D series, since
        auto-picking is a starting point, not a per-slice operation."""
        index = self._selected_index()
        if index is None:
            return
        spectrum = self.session.spectra[index]
        try:
            target = self._peak_pick_target(spectrum)
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
        for dim, spin in (("f2", self.sr_shift_spin), ("f1", self.sr_shift_indirect_spin)):
            spin.blockSignals(True)
            spin.setValue(self._current_sr(spectrum, dim))
            spin.blockSignals(False)
        # Nothing to reference against on a 1D or a pseudo-2D, whose
        # indirect axis is a slice index rather than a frequency.
        has_indirect = self._sr_axis(spectrum, "f1") is not None
        self.sr_indirect_label.setVisible(has_indirect)
        self.sr_shift_indirect_spin.setVisible(has_indirect)

    def _on_apply_sr_clicked(self) -> None:
        # Both dimensions from the one button. Each is idempotent
        # (recomputed from its own stashed original axis), so applying
        # the unchanged one costs nothing and re-entering the same
        # numbers twice stays a no-op.
        #
        # Both values are read up front, before either is applied:
        # _apply_reference_shift_hz ends by calling _restore_sr_control,
        # which rewrites *both* spin boxes from metadata -- so reading
        # the second one after applying the first handed back the
        # stored value (0) instead of what the user had just typed, and
        # the indirect shift silently did nothing.
        direct_hz = self.sr_shift_spin.value()
        indirect_hz = self.sr_shift_indirect_spin.value()
        self._apply_reference_shift_hz(direct_hz, "f2")
        self._apply_reference_shift_hz(indirect_hz, "f1")

    def _on_reference_calibration_from_plot(self, position) -> None:
        """Right-click "Set Reference from Cursor (ppm)…" on a 2D --
        reference the spectrum by saying where the clicked peak
        *should* be, rather than by working out an Hz offset by hand.

        Explicit user request: "I need a right click option on the 2D
        spectra to be able to set the sr (by setting ppm values)."
        Both dimensions at once for a genuine 2D, since a heteronuclear
        pair needs referencing on each; the direct one alone for a
        pseudo-2D, whose indirect axis is a slice index.

        The clicked position is read off the axis *as currently
        displayed*, which already carries whatever SR is applied -- so
        the new value is the existing one plus the correction, not the
        correction alone. (:meth:`_apply_reference_shift_hz` is
        idempotent against the original as-loaded axis, so a cumulative
        delta here would be wrong twice over.)"""
        index = self._selected_index()
        if index is None or position is None:
            return
        spectrum = self.session.spectra[index]
        if spectrum.ndim != 2:
            return
        clicked = {"f2": float(position[0]), "f1": float(position[1])}
        dims = ["f2"] + (["f1"] if self._sr_axis(spectrum, "f1") is not None else [])

        dialog = QDialog(self)
        dialog.setWindowTitle("Set Reference from Cursor")
        form = QFormLayout(dialog)
        form.addRow(QLabel(
            "Enter the true chemical shift of the peak under the cursor.\n"
            "The reference (SR) for each dimension is set to match."
        ))
        spins = {}
        labels = {"f2": f"Direct ({spectrum.metadata.nucleus or 'F2'}):",
                  "f1": f"Indirect ({spectrum.metadata.nucleus_indirect or 'F1'}):"}
        for dim in dims:
            spin = QDoubleSpinBox()
            spin.setRange(-1.0e5, 1.0e5)
            spin.setDecimals(4)
            spin.setSuffix(" ppm")
            spin.setValue(clicked[dim])
            spin.setToolTip(f"Cursor is currently at {clicked[dim]:.4f} ppm here.")
            form.addRow(labels[dim], spin)
            spins[dim] = spin

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        # Both values worked out before either is applied -- see
        # _on_apply_sr_clicked for why (applying one rewrites the other's
        # spin box from metadata).
        updates = {}
        for dim in dims:
            base_mhz = spectrum.metadata.sfo1 if dim == "f2" else spectrum.metadata.sfo1_indirect
            if not base_mhz:
                continue
            current = self._current_sr(spectrum, dim)
            updates[dim] = current + (spins[dim].value() - clicked[dim]) * base_mhz
        for dim, value in updates.items():
            self._apply_reference_shift_hz(value, dim)

    def _on_reference_shift_from_plot(self) -> None:
        """Handles :attr:`MplCanvas.referenceShiftRequested`, the
        Spectrum plot's right-click "Set Reference Shift (Hz)…" --
        a one-shot alternative to the Processing tab's Reference box
        for the common case of just entering a known SR correction."""
        index = self._selected_index()
        if index is None:
            return
        spectrum = self.session.spectra[index]
        current_hz = self._current_sr(spectrum, "f2")
        value, ok = QInputDialog.getDouble(
            self, "Reference shift", "SR (Hz):", current_hz, -1.0e6, 1.0e6, 4,
        )
        if ok:
            self._apply_reference_shift_hz(value)

    #: ``metadata.raw`` keys per dimension: (applied value, original
    #: Hz axis, original ppm axis). The direct dimension's keys are
    #: unsuffixed, exactly as they were before the indirect axis got
    #: its own SR, so a sidecar written by an older version still
    #: restores.
    _SR_KEYS = {
        "f2": ("sr_shift_hz", "sr_original_hz", "sr_original_ppm"),
        "f1": ("sr_shift_hz_f1", "sr_original_hz_f1", "sr_original_ppm_f1"),
    }

    #: Where the vendor's own already-applied SR is recorded -- see
    #: :data:`opennmr.io.bruker._SR_RECORDED_KEY`.
    _SR_RECORDED_KEYS = {"f2": "sr_recorded_hz", "f1": "sr_recorded_hz_f1"}

    @classmethod
    def _recorded_sr(cls, spectrum, dim: str) -> float:
        """The SR the *file* already carries for ``dim`` -- TopSpin's
        ``(SF - BF1) * 1e6``, which the reader has folded into the axis
        before this GUI ever sees it.

        Explicit user request: "I want any sr that was noted in the
        import files to be applied already and to show in the
        processing tab". It was always applied; what was missing is
        that nothing said so, and the SR box read 0 for a dataset that
        plainly was referenced -- 315.3 Hz on this project's own EXSY.

        Making the box show it means the number in it is the **total**
        SR in effect, not a user-only delta on top. So
        :meth:`_apply_reference_shift_hz` applies ``value - recorded``
        to the as-loaded axis: leaving the box alone is then a no-op,
        and typing a different number means "make the total this"."""
        value = spectrum.metadata.raw.get(cls._SR_RECORDED_KEYS[dim])
        return float(value) if value else 0.0

    @classmethod
    def _current_sr(cls, spectrum, dim: str) -> float:
        """The SR currently in effect for ``dim`` -- whatever the user
        has set, or the file's own value if they haven't."""
        stored = spectrum.metadata.raw.get(cls._SR_KEYS[dim][0])
        return float(stored) if stored is not None else cls._recorded_sr(spectrum, dim)

    @staticmethod
    def _sr_axis(spectrum, dim: str):
        """``spectrum``'s own axis for SR dimension ``dim``, or
        ``None`` when there isn't one to reference -- an indirect axis
        only counts if it's a genuine frequency axis with a base
        frequency to convert Hz to ppm with (a pseudo-2D's indirect
        axis is a slice index, and a 1D has no indirect axis at all)."""
        if dim == "f2":
            return spectrum.axes[-1]
        if spectrum.ndim != 2 or spectrum.metadata.pseudo_2d or not spectrum.metadata.sfo1_indirect:
            return None
        axis = spectrum.axes[0]
        return axis if axis.kind == "frequency" else None

    @classmethod
    def _capture_original_axis(cls, spectrum, dim: str = "f2") -> None:
        """Stash ``dim``'s hz/ppm arrays exactly as loaded, the first
        time SR is ever touched for this spectrum this session -- the
        stable baseline every SR value is recomputed from, so applying
        "the same SR twice" is a no-op instead of compounding, and
        applying "a different SR" replaces rather than adds to whatever's
        currently displayed. Reprocessing the FID naturally invalidates
        this (a fresh NMRSpectrum from the vendor reader has no ``raw``
        carried over), which is the right behaviour -- SR starts over
        from the new axis."""
        _, hz_key, ppm_key = cls._SR_KEYS[dim]
        if hz_key in spectrum.metadata.raw:
            return
        axis = cls._sr_axis(spectrum, dim)
        if axis is None:
            return
        spectrum.metadata.raw[hz_key] = axis.hz.copy() if axis.hz is not None else None
        spectrum.metadata.raw[ppm_key] = axis.ppm.copy() if axis.ppm is not None else None

    def _apply_reference_shift_hz(self, sr_hz: float, dim: str = "f2") -> None:
        """Set ``dim``'s SR to the absolute value ``sr_hz`` -- the Hz
        offset from that dimension's own base frequency -- recomputing
        its axis from the original, as-loaded arrays
        (:meth:`_capture_original_axis`) rather than adding onto
        whatever the axis currently is, so this is idempotent (calling
        it again with the same value is a no-op) rather than
        cumulative.

        ``dim`` : ``"f2"`` (direct) or ``"f1"`` (indirect). The two are
        independent values against independent base frequencies --
        ``sfo1`` and ``sfo1_indirect`` -- because on a heteronuclear 2D
        they are different nuclei, so a single Hz offset could not
        reference both. A no-op for ``"f1"`` when there is no indirect
        frequency axis (see :meth:`_sr_axis`)."""
        index = self._selected_index()
        if index is None:
            return
        spectrum = self.session.spectra[index]
        axis = self._sr_axis(spectrum, dim)
        if axis is None:
            return
        self._capture_original_axis(spectrum, dim)

        shift_key, hz_key, ppm_key = self._SR_KEYS[dim]
        base_mhz = spectrum.metadata.sfo1 if dim == "f2" else spectrum.metadata.sfo1_indirect
        # The stashed original axis is the one the reader produced, and
        # that already carries the file's own SR -- so what gets applied
        # on top of it is the *difference* from that, not the whole
        # value. See _recorded_sr.
        offset_hz = sr_hz - self._recorded_sr(spectrum, dim)
        original_hz = spectrum.metadata.raw.get(hz_key)
        original_ppm = spectrum.metadata.raw.get(ppm_key)
        if original_hz is not None:
            axis.hz = original_hz + offset_hz
        if original_ppm is not None and base_mhz:
            axis.ppm = original_ppm + offset_hz / base_mhz

        spectrum.metadata.raw[shift_key] = sr_hz
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
            # Per-parameter fixed choices (Set Bounds / Fixed
            # Parameters…) apply first; the row's own all-or-nothing
            # "Fixed" checkbox (column 3), when checked, still wins
            # outright over them -- same precedence _on_set_peak_bounds's
            # own docstring documents.
            per_param_fixed = self.peak_table.item(row, 0).data(self._PEAK_FIXED_ITEM_ROLE)
            fixed = dict(per_param_fixed) if per_param_fixed else {}
            fixed_item = self.peak_table.item(row, 3)
            if fixed_item is not None and fixed_item.checkState() == Qt.CheckState.Checked:
                fixed = {"shift": True, "fwhm": True, "A": True, "eta": True}
            if fixed:
                peak_def["fixed"] = fixed
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

    def _fit_equation_constants(self) -> dict:
        """The selected spectrum's own acquisition parameters, named
        the way a pulse programme names them (``d1``, ``p1``, ``cnst2``,
        ``l4``, ``ns``, ...), for use as known values in a custom
        fitting equation.

        Explicit user request: "it would be good if I can make fitting
        equations using values from the input data (e.g. delays and
        constants and L or whatever)". Every parameter is offered
        rather than a guessed subset -- ``build_custom_model``
        substitutes only the names an equation actually uses, so an
        unused one costs nothing, and a symbol nobody supplies stays a
        fitted parameter exactly as before."""
        index = self._selected_index()
        if index is None:
            return {}
        return io_bruker.acquisition_constants(self.session.spectra[index])

    def _on_custom_equation_changed(self, text: str) -> None:
        """Live preview of the parameters :func:`_build_custom_relaxation_model`
        would detect from the equation typed so far -- lets a typo or a
        missing independent variable be caught before Fit Curve is
        even clicked, rather than only surfacing as an error dialog.

        Also names which acquisition parameters the equation picked up,
        so "why is d1 not being fitted?" answers itself."""
        text = text.strip()
        if not text:
            self.custom_params_label.setText("")
            return
        try:
            spec = processing_relaxation.build_custom_model(
                text, constants=self._fit_equation_constants())
        except ValueError as exc:
            self.custom_params_label.setText(f"Error: {exc}")
            return
        message = f"Parameters (in order): {', '.join(spec['params'])}"
        used = spec.get("constants") or {}
        if used:
            message += "  |  from the data: " + ", ".join(
                f"{name}={value:g}" for name, value in sorted(used.items()))
        self.custom_params_label.setText(message)

    def _build_custom_relaxation_model(self) -> dict:
        equation = self.custom_equation_edit.text().strip()
        if not equation:
            raise ValueError("Enter an equation for the custom model (e.g. A*exp(-x/T1)+C).")
        p0_text = self.custom_p0_edit.text().strip()
        p0_overrides = [float(tok) for tok in p0_text.split(",") if tok.strip()] if p0_text else None
        return processing_relaxation.build_custom_model(
            equation, p0_overrides=p0_overrides, constants=self._fit_equation_constants())

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
        away. Also seeds each slice after the first per
        :attr:`seed_mode_combo` (see its own tooltip) rather than
        always fitting every slice independently from the same
        starting guess -- real user requirement: "fit one slice
        first, then either apply those same parameters to every
        other slice or use them as each next slice's own starting
        guess". And optionally caps each peak's amplitude from its own
        Integration Region's raw integral when
        :attr:`amplitude_bound_regions_checkbox` is checked (see its
        own tooltip) -- fitting-wizard-roadmap item "bounding possibly
        from integrals"."""
        peak_defs = self._read_peak_defs()
        if not peak_defs:
            raise ValueError("Add at least one peak to fit.")
        prepared = [self._ensure_slice_info(s, i + 1) for i, s in enumerate(entries)]
        fit_regions = self._read_region_defs() or None
        sn_kwargs = self._read_sn_gating_kwargs()
        amplitude_bound_regions = fit_regions if self.amplitude_bound_regions_checkbox.isChecked() else None
        fitted = processing_fitting.NMR2D_fit_slices(
            {"rows": prepared}, peak_defs, mode="rows", fit_regions=fit_regions,
            show_report=False, show_warnings=False, seed_mode=self.seed_mode_combo.currentData(),
            amplitude_bound_regions=amplitude_bound_regions,
            **sn_kwargs,
        )
        return fitted["rows"]

    #: Fraction of a region's own width, measured from either edge,
    #: that counts as "too close" for _warn_if_fit_near_region_edge --
    #: arbitrary but conservative (a peak fitted to within 1/20th of
    #: the window it was allowed to move in is a real sign the region
    #: itself may be cutting the true peak off, not usually a
    #: coincidence).
    _FIT_REGION_EDGE_WARNING_FRACTION = 0.05

    def _warn_if_fit_near_region_edge(self, results: list[dict]) -> None:
        """Real fitting-wizard-roadmap requirement: "a sensible warning
        when a fit result lands too close to its own region edge."
        Only meaningful when the Integration Regions table actually
        constrained the fit at all -- an unconstrained fit has no
        "edge" to land near. For each fitted peak, "its own region" is
        whichever region contained *that starting guess* (not the
        fitted result, which may already have drifted outside a region
        entirely -- a more serious problem this same check still
        catches, since a shift outside every region trivially counts
        as "not within the margin of the one it started in" here).
        Aggregated across every slice into one summary dialog rather
        than one popup per slice -- a pseudo-2D fit can run across
        hundreds of slices, and a popup per slice would make the whole
        feature unusable at that scale."""
        fit_regions = self._read_region_defs()
        if not fit_regions:
            return
        affected: dict[int, int] = {}
        for result in results:
            per_peak = processing_fitting.extract_params_and_errors(
                result["popt"], result["pcov"], result["peak_defs"], result["is_free"],
            )
            for i, peak in enumerate(per_peak):
                fitted_shift = dict(zip(peak["names"], peak["params"]))["shift"]
                start_shift = result["peak_defs"][i]["shift"]
                region = next(((lo, hi) for lo, hi in fit_regions if lo <= start_shift <= hi), None)
                if region is None:
                    continue
                lo, hi = region
                margin = self._FIT_REGION_EDGE_WARNING_FRACTION * (hi - lo)
                if margin > 0 and (fitted_shift - lo < margin or hi - fitted_shift < margin):
                    affected[i] = affected.get(i, 0) + 1
        if not affected:
            return
        n_slices = len(results)
        detail = "; ".join(f"peak {i + 1} in {count} of {n_slices} slice(s)"
                            for i, count in sorted(affected.items()))
        QMessageBox.warning(
            self, "Fit landed near a region edge",
            f"The fitted shift ended up within {self._FIT_REGION_EDGE_WARNING_FRACTION:.0%} of its own "
            f"region's edge for {detail} -- the true peak may extend beyond that region. Consider "
            f"widening the region in the Integration Regions table.",
        )

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
        self._warn_if_fit_near_region_edge(results)

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
                self._warn_if_fit_near_region_edge(slice_results)
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

    @staticmethod
    def _build_fit_report_csv(slice_results: list[dict], relaxation_results: list[dict] | None = None) -> str:
        """Fitting-wizard roadmap item: "a more readable, exportable
        fit report." A plain in-app label only ever shows one slice's
        own spectral-fit parameters at a time (:attr:`spectral_fit_params_label`)
        -- nothing aggregates across a whole pseudo-2D series, and
        nothing writes to a file at all. This builds one CSV (a
        genuinely readable table -- opens directly in Excel/Origin/
        pandas, not just a wall of text) covering *every* slice's
        fitted peak parameters, plus a second section for the curve
        fit (Fit Curve's own per-peak relaxation/diffusion result)
        when one has been run. A pure string-builder (no Qt, no file
        I/O) precisely so it can be tested directly against its own
        output, not just "did a file get written."""
        buffer = StringIO()
        writer = csv.writer(buffer)

        writer.writerow(["Spectral fit -- one row per slice per peak"])
        writer.writerow(["Slice", "Peak", "Type", "Shift", "Shift err", "FWHM", "FWHM err",
                          "Amplitude", "Amplitude err", "Eta", "Eta err"])
        for slice_idx, result in enumerate(slice_results, start=1):
            per_peak = processing_fitting.extract_params_and_errors(
                result["popt"], result["pcov"], result["peak_defs"], result["is_free"],
            )
            for peak_idx, peak in enumerate(per_peak, start=1):
                values = dict(zip(peak["names"], peak["params"]))
                errors = dict(zip(peak["names"], peak["errors"]))
                writer.writerow([
                    slice_idx, peak_idx, peak["type"],
                    f"{values['shift']:.6g}", f"{errors['shift']:.4g}",
                    f"{values['fwhm']:.6g}", f"{errors['fwhm']:.4g}",
                    f"{values['amplitude']:.6g}", f"{errors['amplitude']:.4g}",
                    f"{values['eta']:.6g}" if "eta" in values else "",
                    f"{errors['eta']:.4g}" if "eta" in errors else "",
                ])

        if relaxation_results:
            writer.writerow([])
            writer.writerow(["Curve fit -- one row per peak"])
            param_names = relaxation_results[0]["param_names"]
            header = ["Peak", "Model"]
            for name in param_names:
                header.extend([name, f"{name} err"])
            header.append("RMSE")
            writer.writerow(header)
            for result in relaxation_results:
                row = [result["peak_label"], result["model"]]
                for name in param_names:
                    row.extend([f"{result['params'][name]:.6g}", f"{result['errors'][name]:.4g}"])
                row.append(f"{result['fit_quality']['rmse']:.4g}")
                writer.writerow(row)

        return buffer.getvalue()

    def _on_export_fit_report_clicked(self) -> None:
        index = self._selected_index()
        if index is None:
            return
        spectrum = self.session.spectra[index]
        slice_results = spectrum.metadata.raw.get("slice_fit_results")
        if not slice_results:
            QMessageBox.information(self, "Nothing to export", "Run Fit Spectrum first.")
            return
        path, _selected_filter = QFileDialog.getSaveFileName(
            self, "Export fit report", "fit_report.csv", "CSV files (*.csv)",
            options=QFileDialog.Option.DontUseNativeDialog,
        )
        if not path:
            return
        relaxation_results = spectrum.metadata.raw.get("relaxation_fit_results")
        report = self._build_fit_report_csv(slice_results, relaxation_results)
        try:
            with open(path, "w", encoding="utf-8", newline="") as f:
                f.write(report)
        except OSError as exc:
            QMessageBox.critical(self, "Export failed", str(exc))

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
            # Preserve any custom bounds/per-parameter-fixed choices
            # already attached to this row's column-0 item (see
            # _on_set_peak_bounds) -- a fresh QTableWidgetItem here
            # would otherwise silently drop them.
            old_item = self.peak_table.item(row, 0)
            custom_bounds = old_item.data(Qt.ItemDataRole.UserRole) if old_item else None
            per_param_fixed = old_item.data(self._PEAK_FIXED_ITEM_ROLE) if old_item else None
            shift_item = QTableWidgetItem(f"{params['shift']:.4g}")
            if custom_bounds:
                shift_item.setData(Qt.ItemDataRole.UserRole, custom_bounds)
            if per_param_fixed:
                shift_item.setData(self._PEAK_FIXED_ITEM_ROLE, per_param_fixed)
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
        peak_fixed_params = []
        for row in range(self.peak_table.rowCount()):
            row_values = [self.peak_table.item(row, col).text() if self.peak_table.item(row, col) else ""
                          for col in range(3)]  # Shift, FWHM, Eta -- plain text columns
            fixed_item = self.peak_table.item(row, 3)
            row_values.append("1" if fixed_item and fixed_item.checkState() == Qt.CheckState.Checked else "0")
            peaks.append(row_values)
            shift_item = self.peak_table.item(row, 0)
            peak_bounds.append((shift_item.data(Qt.ItemDataRole.UserRole) if shift_item else None) or {})
            peak_fixed_params.append((shift_item.data(self._PEAK_FIXED_ITEM_ROLE) if shift_item else None) or {})
        regions = [
            [self.region_table.item(row, col).text() if self.region_table.item(row, col) else ""
             for col in range(len(self._REGION_TABLE_COLUMNS))]
            for row in range(self.region_table.rowCount())
        ]
        return {
            "peaks": peaks,
            "peak_bounds": peak_bounds,
            "peak_fixed_params": peak_fixed_params,
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
            "seed_mode": self.seed_mode_combo.currentIndex(),
            "amplitude_bound_regions": self.amplitude_bound_regions_checkbox.isChecked(),
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
            self.seed_mode_combo.setCurrentIndex(0)
            self.amplitude_bound_regions_checkbox.setChecked(False)
            self.sn_noise_region_edit.clear()
            self.sn_threshold_spin.setValue(3.0)
        else:
            peak_bounds = state.get("peak_bounds", [])
            peak_fixed_params = state.get("peak_fixed_params", [])
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
                if i < len(peak_fixed_params) and peak_fixed_params[i]:
                    self.peak_table.item(row, 0).setData(self._PEAK_FIXED_ITEM_ROLE, peak_fixed_params[i])
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
            self.seed_mode_combo.setCurrentIndex(state.get("seed_mode", 0))
            self.amplitude_bound_regions_checkbox.setChecked(state.get("amplitude_bound_regions", False))
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
        is_pseudo2d_heatmap = (
            spectrum.ndim == 2 and spectrum.is_pseudo_2d and self.pseudo2d_view_combo.currentIndex() == 2
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
            elif is_pseudo2d_heatmap:
                self._show_pseudo2d_heatmap(spectrum)
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
        if meta.pulse_program:
            bits.append(f"Pulse programme: {meta.pulse_program}")
        processed_reason = meta.raw.get("loaded_from_processed")
        if processed_reason:
            bits.append(f"Loaded from processed data, not the raw FID: {processed_reason}")
        self.info_label.setText("\n".join(bits))

        self.imaging_max_gradient_spin.blockSignals(True)
        own_gradient = meta.raw.get("imaging_max_gradient_g_per_cm")
        if own_gradient:
            gradient_default = own_gradient
        else:
            # No per-experiment value recorded yet -- fall back to
            # this probe's own saved app-wide default (see
            # _on_save_probe_gradient_default_clicked), if any. Not
            # written back into metadata.raw itself: an explicit edit
            # here should still take priority over a *later* probe-
            # config change, which a silent write-back would defeat by
            # locking this experiment onto whatever the probe default
            # happened to be the first time it was ever selected.
            probe_config = self._load_probe_config().get(meta.probe or "", {})
            gradient_default = probe_config.get("max_gradient_g_per_cm") or 0.0
        self.imaging_max_gradient_spin.setValue(gradient_default)
        self.imaging_max_gradient_spin.blockSignals(False)
        self._restore_imaging_gradient_percent(spectrum)
        self._refresh_imaging_gradient()

        # X-Pulse's own read_2d has no pseudo_2d concept at all (every
        # genuine 2D load is unconditionally the real thing, no
        # parameter to ask for pseudo-2D instead) -- unlike Bruker/
        # Varian, which both accept it. Offering this toggle anyway let
        # a user reach _on_process_indirect_toggled's own reload call,
        # which forwards pseudo_2d= straight into read_2d and raised
        # TypeError: read_2d() got an unexpected keyword argument
        # 'pseudo_2d' -- confirmed directly against a real X-Pulse HSQC
        # dataset. Hidden here the same way it already is for 1D data,
        # rather than shown-but-broken.
        supports_pseudo_2d_toggle = spectrum.metadata.vendor != "xpulse"
        self.process_indirect_button.setVisible(spectrum.ndim == 2 and supports_pseudo_2d_toggle)
        self.process_indirect_button.blockSignals(True)
        self.process_indirect_button.setChecked(not spectrum.is_pseudo_2d)
        self.process_indirect_button.blockSignals(False)

        # Bruker-only for now -- it's the one reader with a `magnitude`
        # parameter, and the only vendor whose files record the mode
        # themselves (PH_mod). Shown for 1D as well as 2D: a
        # hard-to-phase 1D is exactly what the manual override is for.
        # Only an extracted slice with a still-loaded parent can push
        # its phase back, so the button is hidden otherwise rather than
        # shown-but-failing.
        self.apply_slice_phase_button.setVisible(self._slice_source_of(spectrum)[0] is not None)

        self.magnitude_button.setVisible(spectrum.metadata.vendor == "bruker")
        self.magnitude_button.blockSignals(True)
        self.magnitude_button.setChecked(bool(spectrum.metadata.raw.get("magnitude_mode")))
        self.magnitude_button.blockSignals(False)

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
        self._restore_nq_line_control(spectrum)
        self._restore_contour_slices(spectrum)
        self._refresh_peak_slice_control(spectrum)
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
