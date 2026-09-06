"""
Matplotlib canvas embedded in the Qt GUI.

Plotting into this canvas always goes through the ``ax``-first
functions in :mod:`opennmr.plotting` (never
:func:`opennmr.plotting.style.setup_plotting`, which mutates global
matplotlib state) -- see that module's docstring for why: restyling
one canvas must never restyle every other open one.
"""

from __future__ import annotations

import os

import numpy as np
from matplotlib import colormaps
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout,
    QHBoxLayout, QInputDialog, QLabel, QLineEdit, QMenu, QPushButton, QSpinBox,
    QVBoxLayout, QWidget,
)

from ...core.registry import find_gamma, NUC_label
from ...core.spectrum import NMRSpectrum
from ... import plotting
from ...processing import slicing as processing_slicing


class _Toolbar(NavigationToolbar2QT):
    """The stock toolbar minus the two buttons that pop up a separate
    floating dialog (``Configure subplots``, ``Edit axis, curve and
    image parameters``) -- this app's own docked controls (the
    right-click menu's legend/aspect-ratio options, the Processing
    tab) cover the same ground without a popup window."""

    toolitems = [item for item in NavigationToolbar2QT.toolitems if item[0] not in ("Subplots", "Customize")]

    def save_figure(self, *args) -> None:
        """Overridden so a saved figure is never silently missing
        points the live view dropped purely for faster interactive
        rendering (see :meth:`MplCanvas.export_full_resolution`) --
        applies no matter how this is triggered (this toolbar's own
        Save icon, or programmatically, e.g. the File menu's Save
        Figure As…), since both end up calling this same method."""
        bound_save = super().save_figure
        owner = self.parent()
        if isinstance(owner, MplCanvas):
            owner.export_full_resolution(lambda: bound_save(*args))
        else:
            bound_save(*args)


class MplCanvas(QWidget):
    """A matplotlib figure + navigation toolbar (pan/zoom/save), ready
    to plot an :class:`NMRSpectrum` via :func:`show_spectrum`.

    Also always supports, without needing a toolbar tool selected
    first: mouse-wheel zoom (centred on the cursor) and left-click-drag
    pan. These stay out of the way of the toolbar's own explicit
    zoom-rectangle/pan tools -- the drag handler only acts when neither
    of those is active (``toolbar.mode == ""``); the toolbar's own
    button remains available for a precise rectangle-zoom.

    Right-click the plot for a legend toggle and an aspect-ratio lock
    (so the plot box keeps its shape when the window is resized).
    """

    #: Fraction the view shrinks/grows by per wheel notch.
    ZOOM_FACTOR = 1.15

    #: Fraction the contour threshold (as a fraction of peak intensity)
    #: scales by per wheel notch, while a 2D/pseudo-2D contour is shown.
    CONTOUR_ZOOM_FACTOR = 1.3
    _CONTOUR_FACTOR_MIN = 1e-6
    _CONTOUR_FACTOR_MAX = 0.5
    #: The lowest contour, as a fraction of the data's own peak --
    #: explicit user feedback: 1e-3 (0.1%) reached so far into the
    #: noise floor that, combined with 20 levels bunched toward it,
    #: real (weaker) signal was buried under a dense mass of
    #: near-noise-level rings rather than a few clean, readable ones.
    #: 7% is a reasonable "typical spectrum" middle of the requested
    #: 5-10% range -- still user-adjustable via the right-click "Set
    #: Contour Levels..." dialog for a specific dataset's own dynamic
    #: range.
    _CONTOUR_FACTOR_DEFAULT = 0.07

    #: Defaults for the right-click "Set Contour Levels…" dialog -- see
    #: opennmr.plotting.twod._prepare_contour for what each controls.
    _CONTOUR_CLEVELS_DEFAULT = 7
    _CONTOUR_LEVEL_MODE_DEFAULT = "power"
    _CONTOUR_POWER_DEFAULT = 1.25

    #: Shared axis styling for every 1D-line view (Spectrum, FID,
    #: Compare, Fit, Relaxation) -- applied via apply_spine_style/
    #: apply_tick_style at every (re)plot, and to axis-label/title
    #: fontsize directly. Deliberately heavier than
    #: opennmr.plotting.style.setup_plotting's own manuscript-figure
    #: defaults (axes.linewidth 1.0, fontsize 12): a GUI canvas is
    #: viewed at a larger on-screen size with more surrounding UI, so
    #: needs bolder axes/larger text to stay legible at a glance.
    #:
    #: These four are class-level *defaults* only -- __init__ copies
    #: them onto instance attributes of the same name, which is what
    #: every show_* method actually reads (self._AXIS_LINEWIDTH etc.),
    #: so :meth:`set_style` can override them per-instance (the
    #: Output tab's live style panel) without touching every show_*
    #: call site.
    _AXIS_LINEWIDTH = 1.75
    _AXIS_LABELSIZE = 16
    _TICK_LENGTH = 5.5
    _TICK_WIDTH = 1.75

    #: Default **export** figure size, inches. Not applied to the live
    #: embedded figure at any point -- see :meth:`set_export_settings`
    #: for why that fights Qt's own canvas resizing. A 2D carrying
    #: extracted slices gets its own pair, since it is a two-panel
    #: layout rather than one square contour.
    _EXPORT_FIG_SIZE = (8.0, 8.0)
    _EXPORT_FIG_SIZE_WITH_SLICES = (7.5, 10.0)

    #: Max pixel movement between a click and its release still counted
    #: as a click (row-select in a "Stacked" Compare view -- see
    #: _on_release) rather than a pan drag.
    _CLICK_DRAG_TOLERANCE_PX = 3

    #: Background tint for the selected row in a "Stacked" Compare view
    #: (see _apply_compare_selection_highlight) -- a light, unobtrusive
    #: highlight against the default white axes background.
    _COMPARE_SELECTED_FACECOLOR = "#fff3b0"

    #: Emitted with the clicked x-data value when pick mode is on (see
    #: :func:`set_pick_mode`) -- used by the Analysis tab's "Pick peaks
    #: by clicking" to add a peak at the clicked shift without needing
    #: to type it in.
    peakPicked = Signal(float)

    #: Emitted (with no value -- the caller prompts for it) when the
    #: right-click menu's "Set Reference Shift (Hz)…" is chosen, while
    #: a plain 1D spectrum is the current view. The canvas itself
    #: doesn't know about NMRSpectrum axes/session state, so it just
    #: asks the owner (MainWindow) to handle it.
    referenceShiftRequested = Signal()

    #: Emitted with the right-clicked (x, y) data position when
    #: "Exclude Nearest Point" is chosen from the right-click menu,
    #: while a relaxation/diffusion curve fit (show_relaxation_fit) is
    #: the current view -- the canvas only knows where the click
    #: landed, not which fit-result point that's closest to or how to
    #: re-fit without it, so it just asks the owner (MainWindow).
    fitPointExcludeRequested = Signal(float, float)

    #: Emitted (no value) when "Reset Excluded Points" is chosen --
    #: same division of responsibility as fitPointExcludeRequested.
    fitPointsResetRequested = Signal()

    #: Emitted with (lo, hi) x-data values -- lo < hi -- when a drag
    #: finishes while region-select mode is on (see
    #: :func:`set_region_select_mode`) -- used by the Analysis tab's
    #: integration-region picker to add a row spanning the dragged
    #: range without needing to type it in, the drag equivalent of
    #: peakPicked's single click.
    regionSelected = Signal(float, float)

    #: An extracted slice the user wants to work on properly (phase,
    #: fit, compare) rather than just look at. `extract_slice` only
    #: *displays* a trace -- it never reaches the session, so none of
    #: the phase controls (which act on session spectra) could touch
    #: it. Carries the trace itself; `MainWindow` adds it to the
    #: session and selects it. Real user request: "a method to phase
    #: on an extracted slice".
    sliceKeepRequested = Signal(object)

    #: Emitted with the full list of extracted-slice specs whenever
    #: the user adds or clears one, so MainWindow can store them
    #: against the spectrum (and persist them) rather than leaving
    #: them as canvas-local state that a reselection would drop.
    contourSlicesChanged = Signal(object)

    #: Tag applied to every persistent integration-region axvspan (see
    #: :func:`set_regions`) so they can be found and cleared by gid on
    #: the next call, without needing to track a separate reference
    #: list that would go stale across a figure.clear() (a brand new
    #: Axes each time, invalidating any previously held patch objects).
    _REGION_PATCH_GID = "integration_region"

    def __init__(self, parent=None, full_resolution: bool = False) -> None:
        super().__init__(parent)

        self.figure = Figure()
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = _Toolbar(self.canvas, self)

        layout = QVBoxLayout(self)
        layout.addWidget(self.toolbar)

        # A visible bar for acting on an extracted slice, shown only
        # while one is being viewed. Real user report: "There is no way
        # to actually phase the extracted slice as it stands. maybe the
        # extracted view should have new options to do something with
        # the slice". The actions already existed in the right-click
        # menu and worked -- but a context menu on a plot is not
        # somewhere anyone looks, so in practice the capability may as
        # well not have been there. Kept in the context menu too, since
        # removing it would break the habit of anyone who had found it.
        self._slice_bar = QWidget(self)
        slice_bar_layout = QHBoxLayout(self._slice_bar)
        slice_bar_layout.setContentsMargins(4, 0, 4, 0)
        self._slice_label = QLabel("")
        slice_bar_layout.addWidget(self._slice_label)
        slice_bar_layout.addStretch(1)

        self._keep_slice_button = QPushButton("Keep for Phasing/Fitting")
        self._keep_slice_button.setToolTip(
            "Add this slice to the spectrum list as its own 1D spectrum, so "
            "the Phase controls, fitting and Compare can act on it. Viewing "
            "a slice alone only displays it."
        )
        self._keep_slice_button.clicked.connect(self._emit_slice_keep_requested)
        slice_bar_layout.addWidget(self._keep_slice_button)

        self._back_to_contour_button = QPushButton("Back to Contour")
        self._back_to_contour_button.setToolTip("Return to the 2D contour this slice came from.")
        self._back_to_contour_button.clicked.connect(self._show_contour_again)
        slice_bar_layout.addWidget(self._back_to_contour_button)

        self._slice_bar.setVisible(False)
        layout.addWidget(self._slice_bar)

        layout.addWidget(self.canvas)

        self._pan_start = None  # (x_pixel, y_pixel, xlim, ylim) while dragging
        self._legend_visible = False
        self._legend_loc = "best"
        self._aspect_ratio: float | None = None  # None = unlocked

        # Per-instance style state (Output tab / MainWindow.set_style) --
        # copied from the class-level defaults above; see their own
        # comment for why every show_* method reading self._AXIS_* etc.
        # already picks these up with no further changes needed.
        self._AXIS_LINEWIDTH = type(self)._AXIS_LINEWIDTH
        self._AXIS_LABELSIZE = type(self)._AXIS_LABELSIZE
        self._TICK_LENGTH = type(self)._TICK_LENGTH
        self._TICK_WIDTH = type(self)._TICK_WIDTH
        self._grid_visible = False
        self._minor_ticks_visible = True
        self._compare_colormap = "viridis"
        # Export-only DPI/figure-size -- see set_export_settings's own
        # docstring for why these never touch the live figure directly.
        self._export_dpi: float | None = None
        self._export_fig_width: float | None = None
        self._export_fig_height: float | None = None
        # Persists across a rebuild the same way legend/aspect-lock do
        # (see show_spectrum's own docstring) -- an explicit opt-out of
        # plot_1d's/plot_2d_contour's default rendering-only
        # downsampling (see opennmr.plotting.style.downsample_indices/
        # downsample_2d), for on-screen viewing. Exporting/saving a
        # figure always uses full resolution regardless of this (see
        # export_full_resolution) -- this only controls what's
        # interactively shown. Defaults False (fast rendering) except
        # where a caller passes ``full_resolution=True`` at
        # construction -- MainWindow does this only for the main
        # Spectrum canvas (explicit user feedback: the Spectrum tab
        # was silently, severely downsampled by default with no way to
        # tell without already knowing to look for the toggle -- every
        # other canvas, e.g. the FID-processing live preview, is
        # exactly where fast/approximate rendering over exactness is
        # the right tradeoff, and keeps the old default).
        self._full_resolution = full_resolution
        # The 2D contour's own downsampling fraction (see
        # _contour_fraction_kwargs/plotting.plot_2d_contour's own
        # contour_fraction_x/contour_fraction_y) when _full_resolution
        # is off -- explicit user request that the "4x less" default
        # (0.25 -- plot_2d_contour's own hardcoded default, matched
        # here) be user-adjustable rather than fixed. Class-level, not
        # per-canvas-instance-defaulted the way full_resolution is --
        # every canvas shares one user preference, set uniformly by
        # MainWindow's Output tab, until changed again.
        self._contour_fraction = 0.25
        # Diagonal-line overlay (plotting.plot_2d_contour's own
        # show_nQ_line/nQ params -- already existed in the backend,
        # just never wired to the GUI until this) -- y = n*x per value
        # in _nq_values, the standard way to mark the main diagonal
        # (n=1, an EXSY/NOESY/COSY-style homonuclear correlation) or a
        # multiple-quantum correlation (n=2/3/...). Off by default,
        # same as show_nQ_line's own default.
        self._show_nq_line = False
        # Extracted slices drawn *beside* the contour (dashed line
        # across it + the trace in a panel to the right), as opposed
        # to extract_slice's own "replace the view with this trace".
        # Owned by the spectrum, not the canvas -- MainWindow stores
        # and restores them via contourSlicesChanged.
        self._contour_slices: list[dict] = []
        self._nq_values: list[int] = [1]
        self._pick_mode = False
        self._region_select_mode = False
        self._region_select_start: float | None = None  # x-data value of the drag's own start
        self._region_select_ax = None  # axes the current region-select drag started on
        self._region_select_patch = None  # the live rubber-band axvspan, redrawn every motion tick
        # Set only while a 2D/pseudo-2D contour is the current view --
        # scrolling then adjusts the contour threshold instead of
        # zooming (see _on_scroll). None whenever any other view
        # (1D line, FID, compare, fit) is shown.
        self._contour_spectrum: NMRSpectrum | None = None
        # Set only by extract_slice, right before it replaces a
        # contour with a single extracted row/column -- the contour
        # spectrum that slice came *from*, so a "Show Contour" menu
        # action can get back to it. Real gap this fixes: extract_slice
        # had no way back at all once used, short of reselecting the
        # spectrum in the list (which also loses anything else it was
        # mid-editing). Cleared unconditionally at the top of every
        # show_spectrum call (below) -- correct any time this canvas's
        # own content genuinely changes to something else, since
        # show_spectrum is the only entry point for this canvas.
        self._extracted_from_spectrum: NMRSpectrum | None = None
        # The extracted trace itself (and a label for it), so
        # "Keep Slice" can hand exactly what is on screen to the
        # session rather than recomputing it. Cleared alongside
        # _extracted_from_spectrum whenever the view changes.
        self._extracted_trace: NMRSpectrum | None = None
        self._extracted_label: str = ""
        self._extracted_axis: str = ""
        self._extracted_index: int = 0
        self._contour_factor: float | None = None
        # Number of contour levels, and how they're spaced between the
        # noise floor (_contour_factor) and the data's own peak -- see
        # opennmr.plotting.twod._prepare_contour. Reset to these
        # defaults on every fresh show_spectrum, same as _contour_factor.
        self._contour_clevels = self._CONTOUR_CLEVELS_DEFAULT
        self._contour_level_mode = self._CONTOUR_LEVEL_MODE_DEFAULT
        self._contour_power = self._CONTOUR_POWER_DEFAULT
        # The F1 (left)/F2 (top) projection panels' own axes, from the
        # current contour draw -- None (both) whenever a contour isn't
        # the current view, or a given side wasn't drawn. Lets scroll-
        # zoom over a projection adjust its own intensity scale
        # independently of the main contour (see _on_scroll).
        self._proj_axes: dict = {"f1": None, "f2": None}
        # Set only for a plain 1D spectrum/slice view -- lets the
        # right-click menu offer ppm/Hz/kHz switching by re-using the
        # spectrum's own stored axis arrays, no recomputation needed.
        # None for any other view (contour, FID, compare, fit), where
        # unit-switching isn't offered.
        self._current_1d_spectrum: NMRSpectrum | None = None
        self._axis_unit = "ppm"
        self._ppm_label = ""
        # Imaging Analysis sub-tab's gradient-to-mm conversion inputs
        # -- see set_imaging_gradient_params/set_axis_unit's "mm" case.
        self._imaging_gamma_hz_per_t: float | None = None
        self._imaging_max_gradient_g_per_cm: float | None = None
        # The Phase 1 pivot marker (see set_pivot_marker) -- a Line2D
        # kept around so it can be *moved* in place on every drag tick
        # instead of removed/recreated, but figure.clear() (every
        # show_* rebuild) invalidates it, so it's reset to None
        # wherever that happens (see _apply_view_options).
        self._pivot_line = None
        # Which axes (if any) the pivot marker above should draw on --
        # None wherever a pivot marker doesn't make sense (a 2D
        # contour, FID, Compare, fit view). Set explicitly by
        # show_spectrum (its 1D branch) and show_processing_preview
        # (its spectrum panel specifically, not the FID one) after
        # _apply_view_options resets it, same reset-then-opt-back-in
        # pattern as _freeform_zoom_axes.
        self._pivot_marker_ax = None
        # Small triangle markers left by set_peak_markers -- same
        # reset/reattach lifecycle as the pivot line above (dropped in
        # _apply_view_options, drawn onto _pivot_marker_ax).
        self._peak_marker_lines: list = []
        # The per-spectrum axes of a "Stacked" Compare view (one row
        # per spectrum, sharing x, each with its own independent
        # y-autoscale) -- None for every other view, including a
        # "Superimposed" Compare (one shared axes, same as any other
        # single-ax view). Each row keeps its own baseline fixed under
        # a y-zoom (a single shared axes can't do that at all -- see
        # show_compare's own docstring) while still letting scroll-zoom
        # scale every row together by default (see _on_scroll and
        # _selected_compare_index right below).
        self._compare_axes: list | None = None
        # Which row of a "Stacked" Compare view (an index into
        # _compare_axes) scroll-zoom targets on its own, decoupled from
        # the rest -- None (the default, and reset on every fresh
        # show_compare/clear) means scroll-zoom over ANY row scales
        # every row together instead, by the same relative factor,
        # each still pinned at its own baseline. Set/cleared by
        # clicking a row (see _on_release/_toggle_compare_selection);
        # _apply_compare_selection_highlight shows which one, if any,
        # via a tinted background.
        self._selected_compare_index: int | None = None
        # True only while show_relaxation_fit's plot is the current
        # view -- gates the right-click "Exclude Nearest Point"/"Reset
        # Excluded Points" actions (see _on_context_menu) and the
        # right-click position capture in _on_press that backs them.
        self._fit_mode = False
        # Axes whose y-zoom should be cursor-centred ("freeform")
        # rather than pinned at the current bottom -- an FID
        # oscillates around y=0 rather than rising from a baseline the
        # way a phased spectrum does, so "pin the bottom" doesn't mean
        # anything sensible there. Populated only by show_fid (the
        # whole view) and show_processing_preview (its FID panel
        # specifically, not the spectrum panel below it) -- empty for
        # every other view, where pin-bottom is still correct.
        self._freeform_zoom_axes: set = set()
        # show_processing_preview's own two axes, kept around (only
        # while that's the current view -- None otherwise) so
        # update_processing_preview_spectrum can update the spectrum
        # panel's data *in place* on a phase change, preserving
        # whatever zoom/pan is already there -- same idea as
        # update_data for the plain single-axes Spectrum view, and for
        # the same reason: dragging phase shouldn't reset a deliberate
        # zoom into a small peak on every tick.
        self._processing_preview_fid_ax = None
        self._processing_preview_spec_ax = None
        self._last_right_click_xy: tuple[float, float] | None = None

        self.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.canvas.mpl_connect("button_release_event", self._on_release)

        self.canvas.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.canvas.customContextMenuRequested.connect(self._on_context_menu)

    def set_pick_mode(self, enabled: bool) -> None:
        """Toggle click-to-pick: while on, a plain left-click emits
        :attr:`peakPicked` with the clicked x-data value instead of
        starting a pan drag."""
        self._pick_mode = enabled
        self.canvas.setCursor(Qt.CursorShape.CrossCursor if enabled else Qt.CursorShape.ArrowCursor)

    def set_region_select_mode(self, enabled: bool) -> None:
        """Toggle drag-to-select-a-region: while on, a left-click-drag
        emits :attr:`regionSelected` with the dragged (lo, hi) x-range
        on release, instead of starting a pan drag -- the range
        equivalent of :func:`set_pick_mode`'s single click."""
        self._region_select_mode = enabled
        self.canvas.setCursor(Qt.CursorShape.CrossCursor if enabled else Qt.CursorShape.ArrowCursor)
        if not enabled:
            self._clear_region_select_patch()
            self._region_select_start = None
            self._region_select_ax = None

    def set_regions(self, regions: list[tuple[float, float]]) -> None:
        """Persistently shade x-axis integration regions on the
        current Spectrum plot. Self-contained -- finds and removes any
        previously drawn region patches on ``self.ax`` by their own
        gid tag (see :attr:`_REGION_PATCH_GID`) rather than tracking a
        separate reference list, which would go stale the moment a
        redraw (``figure.clear()``) replaces ``self.ax`` with a new
        Axes and invalidates any previously held patch objects."""
        if self.ax is None:
            return
        for patch in list(self.ax.patches):
            if patch.get_gid() == self._REGION_PATCH_GID:
                patch.remove()
        for lo, hi in regions:
            span = self.ax.axvspan(lo, hi, color="tab:orange", alpha=0.15, zorder=0)
            span.set_gid(self._REGION_PATCH_GID)
        self.canvas.draw_idle()

    def _update_region_select_patch(self, x_current: float) -> None:
        self._clear_region_select_patch()
        lo, hi = sorted((self._region_select_start, x_current))
        self._region_select_patch = self._region_select_ax.axvspan(
            lo, hi, color="tab:blue", alpha=0.2, zorder=50,
        )
        self.canvas.draw_idle()

    def _clear_region_select_patch(self) -> None:
        if self._region_select_patch is not None:
            self._region_select_patch.remove()
            self._region_select_patch = None
            self.canvas.draw_idle()

    def _on_scroll(self, event) -> None:
        """Scroll hovering the x-axis (its tick-label strip, below the
        plotted data) zooms x only; scrolling over the plot body
        itself zooms y (intensity) only, pinning the bottom of the
        current view and only moving the top -- not a cursor-centred
        zoom, since the bottom (the spectrum's own baseline) is what
        should stay put while scaling how tall a peak looks, the same
        way any NMR viewer's own intensity slider works. Decoupled
        from x entirely, so one doesn't disturb the other. For a 2D/
        pseudo-2D contour specifically, the y-axis (F1/ppm, or slice
        index) is a *position* axis like x, not an intensity one --
        so it gets its own tick-label strip (on the right, matching
        ``plot_2d_contour``'s own ``tick_right()``) with the same
        cursor-centred zoom as the x-strip, rather than pin-bottom;
        the x-axis behaves the same way as the 1D case above;
        scrolling the contour's own body instead adjusts the contour
        threshold (see :meth:`_adjust_contour_levels`); scrolling one
        of its F1/F2 projection panels adjusts that panel's own
        intensity scale independently, rather than any of the above."""
        if event.x is None or event.y is None:
            return

        # The extracted-slice panel is in _proj_axes too, but it is not
        # a projection strip: its y axis is a normalised 0..1 stack of
        # evenly-spread traces, so both the intensity zoom below and
        # the generic fall-through further down would silently rescale
        # it and break the layout. It also sits exactly where the
        # contour's own right-hand "y strip" zoom region used to be
        # unoccupied. A no-op for now, rather than a wrong-op.
        if self._proj_axes.get("slices") is not None and event.inaxes is self._proj_axes["slices"]:
            return

        # Named explicitly rather than iterating _proj_axes, so anything
        # else that dict grows later cannot be mistaken for a
        # projection strip by this loop.
        for kind in ("f1", "f2"):
            proj_ax = self._proj_axes.get(kind)
            if proj_ax is not None and event.inaxes is proj_ax:
                self._zoom_projection_intensity(proj_ax, kind, event)
                return

        on_x_strip = self._hovering_x_axis(self.ax, event)
        on_y_strip = self._contour_factor is not None and self._hovering_y_axis(self.ax, event)
        ax = event.inaxes if event.inaxes is not None else (self.ax if (on_x_strip or on_y_strip) else None)
        if ax is None:
            return

        factor = 1 / self.ZOOM_FACTOR if event.button == "up" else self.ZOOM_FACTOR

        if on_x_strip:
            x = self._data_x_at(ax, event)
            if x is None:
                return
            xmin, xmax = ax.get_xlim()
            ax.set_xlim(x - (x - xmin) * factor, x + (xmax - x) * factor)
            self.canvas.draw_idle()
            return

        if on_y_strip:
            y = self._data_y_at(ax, event)
            if y is None:
                return
            ymin, ymax = ax.get_ylim()
            ax.set_ylim(y - (y - ymin) * factor, y + (ymax - y) * factor)
            self.canvas.draw_idle()
            return

        if self._contour_factor is not None and ax is self.ax:
            self._adjust_contour_levels(event)
            return

        if event.ydata is None:
            return

        if self._compare_axes:
            # Every row together by the same relative factor (each
            # still pinned at its own baseline) unless one row has
            # been clicked to select it -- see _selected_compare_index's
            # own docstring for why "together" is the right default
            # (a quick, expected "zoom the whole stack in/out"
            # gesture) with per-row as the opt-in exception, not the
            # other way around.
            target_axes = (
                [self._compare_axes[self._selected_compare_index]]
                if self._selected_compare_index is not None
                else self._compare_axes
            )
            for row_ax in target_axes:
                row_ymin, row_ymax = row_ax.get_ylim()
                row_ax.set_ylim(row_ymin, row_ymin + (row_ymax - row_ymin) * factor)
            self.canvas.draw_idle()
            return

        if ax in self._freeform_zoom_axes:
            # Cursor-centred, not pinned-bottom -- an FID oscillates
            # around y=0 rather than rising from a baseline, so there's
            # no sensible "bottom" to pin here (see
            # _freeform_zoom_axes's own docstring). Explicit user
            # request.
            y = event.ydata
            ymin, ymax = ax.get_ylim()
            ax.set_ylim(y - (y - ymin) * factor, y + (ymax - y) * factor)
            self.canvas.draw_idle()
            return

        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymin + (ymax - ymin) * factor)
        self.canvas.draw_idle()

    def _zoom_projection_intensity(self, proj_ax, kind: str, event) -> None:
        """Zoom one F1/F2 projection panel's own intensity axis,
        pinning its baseline end and moving only the peak end (see
        :meth:`_on_scroll`'s own docstring for why) -- ``"f2"`` (the
        top strip) plots intensity along y, low end is baseline
        (shared x with the main contour); ``"f1"`` (the left strip)
        plots it along x, rotated, on a *descending* axis so its peaks
        point away from the contour -- which leaves its baseline at the
        ``xmax`` (second) end of its own xlim, so pinning ``xmax`` is
        still pinning the baseline. That descending axis replaced an
        earlier ``-f1_projection`` negation, which pointed the peaks
        the same way but inverted the displayed polarity relative to
        both the contour and the F2 projection (a real, user-reported
        bug -- see ``plot_2d_contour``'s own comment there); this
        method needed no change, since it was already written in terms
        of whichever end ``xmax`` is. Independent of the main contour's
        own threshold/zoom (see :meth:`_on_scroll`)."""
        factor = 1 / self.ZOOM_FACTOR if event.button == "up" else self.ZOOM_FACTOR
        if kind == "f2":
            if event.ydata is None:
                return
            ymin, ymax = proj_ax.get_ylim()
            proj_ax.set_ylim(ymin, ymin + (ymax - ymin) * factor)
        else:
            if event.xdata is None:
                return
            xmin, xmax = proj_ax.get_xlim()
            proj_ax.set_xlim(xmax - (xmax - xmin) * factor, xmax)
        self.canvas.draw_idle()

    @staticmethod
    def _hovering_x_axis(ax, event) -> bool:
        """True if the cursor is below ``ax``'s plot area, within its
        horizontal span -- its x-tick-label strip -- rather than over
        the plotted data itself. ``event.inaxes`` alone can't detect
        this: matplotlib only sets it while the cursor is inside the
        axes' own data rectangle, not its margins."""
        bbox = ax.get_window_extent()
        return bbox.x0 <= event.x <= bbox.x1 and event.y < bbox.y0

    @staticmethod
    def _data_x_at(ax, event) -> float | None:
        """The data-x value under the cursor, even when
        ``event.xdata`` is ``None`` (cursor outside the axes' data
        rectangle, e.g. over its x-tick-label strip) -- matplotlib's
        own ``transData`` is still a valid (affine) mapping there for
        a standard linear axis."""
        try:
            x, _ = ax.transData.inverted().transform((event.x, event.y))
        except Exception:  # noqa: BLE001 -- e.g. a singular transform; just skip the zoom
            return None
        return float(x)

    @staticmethod
    def _hovering_y_axis(ax, event) -> bool:
        """True if the cursor is to the right of ``ax``'s plot area,
        within its vertical span -- its y-tick-label strip, matching
        ``plot_2d_contour``'s own ``tick_right()`` (the main contour
        panel's y-ticks are on the right, since the F1 projection
        panel occupies the left side) -- rather than over the plotted
        data itself. Only meaningful for a 2D/pseudo-2D contour (see
        the ``_contour_factor is not None`` gate at the call site);
        the plain 1D view has no y-ticks shown at all (intensity is
        zoomed by hovering the body instead -- see :meth:`_on_scroll`'s
        own docstring)."""
        bbox = ax.get_window_extent()
        return bbox.y0 <= event.y <= bbox.y1 and event.x > bbox.x1

    @staticmethod
    def _data_y_at(ax, event) -> float | None:
        """The data-y value under the cursor, even when
        ``event.ydata`` is ``None`` (cursor outside the axes' data
        rectangle, e.g. over its y-tick-label strip) -- same reasoning
        as :meth:`_data_x_at`."""
        try:
            _, y = ax.transData.inverted().transform((event.x, event.y))
        except Exception:  # noqa: BLE001 -- e.g. a singular transform; just skip the zoom
            return None
        return float(y)

    def _adjust_contour_levels(self, event) -> None:
        """Scroll-up raises the contour floor (fewer, stronger-only
        levels -- "zoom into the peaks"); scroll-down lowers it
        (reveals weaker signals/noise). Redraws the whole contour
        (~tens of ms, see plot_2d_contour) but preserves the current
        pan/zoom."""
        step = self.CONTOUR_ZOOM_FACTOR if event.button == "up" else 1 / self.CONTOUR_ZOOM_FACTOR
        self._contour_factor = min(
            max(self._contour_factor * step, self._CONTOUR_FACTOR_MIN), self._CONTOUR_FACTOR_MAX,
        )
        self._redraw_contour(preserve_view=True)

    def set_contour_levels(self, clevels: int, level_mode: str, power: float, factor: float | None = None) -> None:
        """Apply new contour-level settings from the right-click "Set
        Contour Levels…" dialog and redraw -- see
        :func:`opennmr.plotting.twod._prepare_contour` for what each
        parameter controls. ``factor`` (the lowest level, as a
        fraction of the data's own peak -- "start") is also normally
        adjustable by scrolling over the contour body; ``None`` (the
        default) leaves it at whatever that scroll last set it to."""
        if factor is not None:
            self._contour_factor = min(max(factor, self._CONTOUR_FACTOR_MIN), self._CONTOUR_FACTOR_MAX)
        self._contour_clevels = clevels
        self._contour_level_mode = level_mode
        self._contour_power = power
        self._redraw_contour(preserve_view=True)

    def _contour_fraction_kwargs(self) -> dict:
        """``contour_fraction_x``/``contour_fraction_y`` overrides for
        :func:`opennmr.plotting.plot_2d_contour`, both call sites --
        :attr:`_contour_fraction` (the user-adjustable downsample
        fraction, see :meth:`set_contour_fraction`) unless
        :attr:`_full_resolution` is on, in which case both are forced
        to ``None`` (no downsampling at all, see that function's own
        docstring)."""
        if self._full_resolution:
            return {"contour_fraction_x": None, "contour_fraction_y": None}
        return {"contour_fraction_x": self._contour_fraction, "contour_fraction_y": self._contour_fraction}

    def set_contour_fraction(self, fraction: float) -> None:
        """Change the 2D contour's own downsampling fraction (see
        :meth:`_contour_fraction_kwargs`) -- a no-op on the fraction
        actually used while :attr:`_full_resolution` is on (which
        already forces no downsampling regardless), but still recorded
        so it takes effect immediately if full resolution is turned
        back off. Redraws right away if a 2D contour is currently
        shown, same as :meth:`set_full_resolution`."""
        self._contour_fraction = fraction
        if self._contour_spectrum is not None and not self._full_resolution:
            self.show_spectrum(self._contour_spectrum)

    def _nq_line_kwargs(self) -> dict:
        """``show_nQ_line``/``nQ`` overrides for
        :func:`opennmr.plotting.plot_2d_contour`, both call sites --
        see :meth:`set_nq_line`."""
        return {"show_nQ_line": self._show_nq_line, "nQ": self._nq_values}

    def _slice_panel_kwargs(self) -> dict:
        """``slices`` override for
        :func:`opennmr.plotting.plot_2d_contour`, both call sites --
        see :meth:`set_contour_slices`."""
        return {"slices": self._contour_slices or None}

    def set_contour_slices(self, slices: list[dict] | None) -> None:
        """Replace the set of extracted slices drawn beside the contour
        -- each a ``{"axis", "index", "label"}`` dict, see
        :func:`opennmr.plotting.twod._draw_slice_panel`. Redraws right
        away if a 2D contour is currently shown, same as
        :meth:`set_nq_line`. Does *not* emit
        :attr:`contourSlicesChanged`: this is the restore direction
        (MainWindow pushing a spectrum's stored slices in), and
        emitting here would write them straight back out again."""
        self._contour_slices = list(slices or [])
        if self._contour_spectrum is not None:
            self.show_spectrum(self._contour_spectrum)

    def _add_contour_slice(self, axis: str, index: int, label: str) -> None:
        self._contour_slices.append({"axis": axis, "index": int(index), "label": label})
        self.contourSlicesChanged.emit(list(self._contour_slices))
        # preserve_view: adding a slice is an annotation, not a change
        # of subject -- explicit user request, "the scale/view shouldn't
        # reset when a slice is extracted". show_spectrum would rebuild
        # from scratch and drop whatever zoom was set.
        if self._contour_spectrum is not None:
            self._redraw_contour(preserve_view=True)

    def _clear_contour_slices(self) -> None:
        self._contour_slices = []
        self.contourSlicesChanged.emit([])
        if self._contour_spectrum is not None:
            self._redraw_contour(preserve_view=True)

    def set_nq_line(self, enabled: bool, values: list[int] | None = None) -> None:
        """Turn the diagonal-line overlay on/off, and/or change which
        ``n`` value(s) it draws (``y = n*x``, e.g. ``[1]`` for the main
        diagonal, ``[2]`` for a double-quantum correlation) -- redraws
        right away if a 2D contour is currently shown, same as
        :meth:`set_contour_fraction`."""
        self._show_nq_line = enabled
        if values is not None:
            self._nq_values = values
        if self._contour_spectrum is not None:
            self.show_spectrum(self._contour_spectrum)

    def _redraw_contour(self, preserve_view: bool) -> None:
        xlim, ylim = (self.ax.get_xlim(), self.ax.get_ylim()) if preserve_view else (None, None)
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        _, _, self._proj_axes = plotting.plot_2d_contour(
            self._contour_spectrum, ax=self.ax, fig=self.figure, factor=self._contour_factor,
            clevels=self._contour_clevels, level_mode=self._contour_level_mode, power=self._contour_power,
            square_aspect=self._aspect_ratio is None,
            **self._contour_fraction_kwargs(), **self._nq_line_kwargs(),
            **self._slice_panel_kwargs(), **self._contour_style_kwargs(),
        )
        self._apply_current_axis_style()
        if preserve_view:
            self.ax.set_xlim(xlim)
            self.ax.set_ylim(ylim)
        self._apply_view_options()
        self.canvas.draw_idle()

    def _on_press(self, event) -> None:
        # getattr, not a plain attribute access -- a real matplotlib
        # MouseEvent always has .xdata (possibly None), but this is no
        # longer gated on _fit_mode first (see below), so a fake event
        # missing the attribute entirely (some existing tests' own
        # minimal doubles, built before this needed to handle the
        # non-fit-mode case too) must not raise AttributeError here.
        if event.button == 3 and getattr(event, "xdata", None) is not None:
            # Captured here (a real matplotlib event, with xdata/ydata
            # already correctly computed) rather than re-derived from
            # the QPoint _on_context_menu's own customContextMenuRequested
            # signal hands over -- that's in Qt widget pixel coordinates,
            # not matplotlib data coordinates, and converting it back
            # correctly (accounting for the y-flip and any HiDPI scale
            # factor between the two coordinate systems) isn't needed
            # when the data position is already sitting right here.
            # Not gated on _fit_mode any more -- also used by
            # _prompt_extract_slice to default a right-clicked 2D
            # contour's Extract Slice… dialog to the cursor's own
            # position instead of a fixed index (real roadmap ask:
            # "row/slice extraction should default to the cursor's
            # current position rather than a fixed index").
            self._last_right_click_xy = (float(event.xdata), float(event.ydata))
            if self._fit_mode:
                return
        if event.inaxes is None or event.button != 1 or self.toolbar.mode != "":
            return
        if self._pick_mode:
            if event.xdata is not None:
                self.peakPicked.emit(float(event.xdata))
            return
        if self._region_select_mode:
            if event.xdata is not None:
                self._region_select_start = float(event.xdata)
                self._region_select_ax = event.inaxes
                self._update_region_select_patch(event.xdata)
            return
        self._pan_start = (event.x, event.y, event.inaxes.get_xlim(), event.inaxes.get_ylim(), event.inaxes)

    def _on_motion(self, event) -> None:
        if self._region_select_start is not None and event.x is not None and self._region_select_ax is not None:
            # Pixel coordinates, not event.xdata directly -- like the
            # pan drag below, this needs to keep tracking smoothly even
            # if the cursor briefly leaves the axes rectangle mid-drag
            # (event.xdata is None outside it; a pixel position isn't).
            inv = self._region_select_ax.transData.inverted()
            x_data, _y_data = inv.transform((event.x, event.y if event.y is not None else 0))
            self._update_region_select_patch(x_data)
            return
        if self._pan_start is None or event.x is None:
            return
        x0, y0, xlim, ylim, ax = self._pan_start
        inv = ax.transData.inverted()
        x0_data, y0_data = inv.transform((x0, y0))
        x1_data, y1_data = inv.transform((event.x, event.y))
        dx, dy = x1_data - x0_data, y1_data - y0_data
        ax.set_xlim(xlim[0] - dx, xlim[1] - dx)
        ax.set_ylim(ylim[0] - dy, ylim[1] - dy)
        self.canvas.draw_idle()

    def _on_release(self, event) -> None:
        if self._region_select_start is not None:
            end = self._region_select_start
            if event.xdata is not None:
                end = float(event.xdata)
            elif event.x is not None and self._region_select_ax is not None:
                inv = self._region_select_ax.transData.inverted()
                end, _y_data = inv.transform((event.x, event.y if event.y is not None else 0))
            lo, hi = sorted((self._region_select_start, end))
            self._clear_region_select_patch()
            self._region_select_start = None
            self._region_select_ax = None
            if hi > lo:
                self.regionSelected.emit(lo, hi)
            return
        if self._pan_start is not None and self._compare_axes:
            x0, y0, _xlim, _ylim, start_ax = self._pan_start
            if (
                event.x is not None and event.y is not None
                and abs(event.x - x0) <= self._CLICK_DRAG_TOLERANCE_PX
                and abs(event.y - y0) <= self._CLICK_DRAG_TOLERANCE_PX
                and start_ax in self._compare_axes
            ):
                self._toggle_compare_selection(start_ax)
        self._pan_start = None

    def set_compare_selection(self, index: int | None) -> None:
        """Public counterpart to :meth:`_toggle_compare_selection` --
        lets something other than a direct click on the plot itself
        (namely, :class:`~opennmr.gui.main_window.MainWindow`'s own
        Compare-list row selection) drive which row is tinted as
        "selected", so the highlight stays in sync with whichever row
        is current in the list too, not just the last row clicked
        directly in the plot. ``index`` : position among the rows
        currently plotted (matching :attr:`_compare_axes`'s own order),
        or ``None`` to clear the highlight -- out-of-range values are
        also treated as "nothing selected" rather than raising, since
        the list and the plot can briefly disagree on row count right
        after a rebuild."""
        if index is not None and not (self._compare_axes and 0 <= index < len(self._compare_axes)):
            index = None
        self._selected_compare_index = index
        self._apply_compare_selection_highlight()
        self.canvas.draw_idle()

    def _toggle_compare_selection(self, ax) -> None:
        """Click a row of a "Stacked" Compare view to make scroll-zoom
        target just that row from now on, decoupled from the rest (see
        _selected_compare_index) -- click the same row again to
        de-select and go back to zooming every row together; click a
        different row to switch which one is selected."""
        index = self._compare_axes.index(ax)
        self._selected_compare_index = None if self._selected_compare_index == index else index
        self._apply_compare_selection_highlight()
        self.canvas.draw_idle()

    def _apply_compare_selection_highlight(self) -> None:
        """Tint whichever row is selected (if any) so it's obvious at
        a glance which one scroll-zoom currently targets on its own."""
        if not self._compare_axes:
            return
        for i, row_ax in enumerate(self._compare_axes):
            selected = i == self._selected_compare_index
            row_ax.set_facecolor(self._COMPARE_SELECTED_FACECOLOR if selected else "none")
            # show_compare sets every row's patch alpha to 0 so a peak
            # bleeding in from a neighbouring row isn't hidden behind an
            # opaque background -- but set_facecolor's own color has no
            # alpha of its own, so that persisted 0 alpha would silently
            # swallow this tint too. Make the selected row's patch
            # opaque just enough to actually show; every other row stays
            # fully transparent for the bleed.
            row_ax.patch.set_alpha(1.0 if selected else 0.0)

    def _on_context_menu(self, pos) -> None:
        menu = QMenu(self)

        legend_action = menu.addAction("Show Legend")
        legend_action.setCheckable(True)
        legend_action.setChecked(self._legend_visible)

        lock_action = menu.addAction("Lock Aspect Ratio")
        lock_action.setCheckable(True)
        lock_action.setChecked(self._aspect_ratio is not None)

        set_ratio_action = menu.addAction("Set Aspect Ratio…")

        sr_action = None
        if self._current_1d_spectrum is not None:
            sr_action = menu.addAction("Set Reference Shift (Hz)…")

        unit_actions = {}
        if self._current_1d_spectrum is not None:
            unit_menu = menu.addMenu("Axis Units")
            for label, key in (("ppm", "ppm"), ("Hz", "hz"), ("kHz", "khz")):
                action = unit_menu.addAction(label)
                action.setCheckable(True)
                action.setChecked(self._axis_unit == key)
                unit_actions[action] = key
            mm_action = unit_menu.addAction("mm (Imaging)")
            mm_action.setCheckable(True)
            mm_action.setChecked(self._axis_unit == "mm")
            mm_action.setEnabled(
                bool(self._imaging_gamma_hz_per_t) and bool(self._imaging_max_gradient_g_per_cm)
            )
            mm_action.setToolTip(
                "Needs a max gradient set on the Imaging Analysis sub-tab."
            )
            unit_actions[mm_action] = "mm"

        contour_levels_action = None
        extract_slice_action = None
        clear_slices_action = None
        if self._contour_spectrum is not None:
            contour_levels_action = menu.addAction("Set Contour Levels…")
            extract_slice_action = menu.addAction("Extract Slice…")
            if self._contour_slices:
                clear_slices_action = menu.addAction(
                    f"Clear Extracted Slices ({len(self._contour_slices)})"
                )

        show_contour_action = None
        keep_slice_action = None
        if self._extracted_from_spectrum is not None:
            show_contour_action = menu.addAction("Show Contour")
            if getattr(self, "_extracted_trace", None) is not None:
                keep_slice_action = menu.addAction("Keep Slice (for phasing/fitting)")
                keep_slice_action.setToolTip(
                    "Add this slice to the spectrum list as its own 1D "
                    "spectrum, so the Phase controls, fitting and Compare "
                    "can act on it. Extracting alone only displays it."
                )

        exclude_point_action = None
        reset_points_action = None
        if self._fit_mode:
            exclude_point_action = menu.addAction("Exclude Nearest Point")
            exclude_point_action.setEnabled(self._last_right_click_xy is not None)
            reset_points_action = menu.addAction("Reset Excluded Points")

        full_res_action = None
        if self.has_downsampled_data():
            full_res_action = menu.addAction("Full Resolution")
            full_res_action.setCheckable(True)
            full_res_action.setChecked(self._full_resolution)
            full_res_action.setToolTip(
                "Render every point instead of the usual display-only "
                "cap (a saved/exported figure is always full "
                "resolution regardless of this)."
            )

        chosen = menu.exec(self.canvas.mapToGlobal(pos))
        if chosen in unit_actions:
            self.set_axis_unit(unit_actions[chosen])
        elif chosen is legend_action:
            self.set_legend_visible(not self._legend_visible)
        elif chosen is lock_action:
            self.set_aspect_ratio(None if self._aspect_ratio is not None else 1.0)
        elif chosen is set_ratio_action:
            value, ok = QInputDialog.getDouble(
                self, "Aspect ratio", "Width : height", self._aspect_ratio or 1.0, 0.01, 100.0, 3,
            )
            if ok:
                self.set_aspect_ratio(value)
        elif sr_action is not None and chosen is sr_action:
            self.referenceShiftRequested.emit()
        elif contour_levels_action is not None and chosen is contour_levels_action:
            self._prompt_contour_levels()
        elif extract_slice_action is not None and chosen is extract_slice_action:
            self._prompt_extract_slice()
        elif clear_slices_action is not None and chosen is clear_slices_action:
            self._clear_contour_slices()
        elif keep_slice_action is not None and chosen is keep_slice_action:
            self._emit_slice_keep_requested()
        elif show_contour_action is not None and chosen is show_contour_action:
            # Read out first -- show_spectrum unconditionally clears
            # _extracted_from_spectrum at its own top (correct for
            # every other caller, since this canvas's own content is
            # genuinely changing to something else), so grabbing the
            # value after that call would already see None.
            spectrum = self._extracted_from_spectrum
            self.show_spectrum(spectrum)
        elif exclude_point_action is not None and chosen is exclude_point_action:
            if self._last_right_click_xy is not None:
                self.fitPointExcludeRequested.emit(*self._last_right_click_xy)
        elif reset_points_action is not None and chosen is reset_points_action:
            self.fitPointsResetRequested.emit()
        elif full_res_action is not None and chosen is full_res_action:
            self.set_full_resolution(not self._full_resolution)

    def set_legend_visible(self, visible: bool) -> None:
        self._legend_visible = visible
        self._apply_view_options()
        self.canvas.draw_idle()

    def set_aspect_ratio(self, ratio: float | None) -> None:
        self._aspect_ratio = ratio
        self._apply_view_options()
        self.canvas.draw_idle()

    def default_export_size(self) -> tuple[float, float]:
        """The export figure size to use when the Output tab hasn't
        been given an explicit one -- :attr:`_EXPORT_FIG_SIZE`, or
        :attr:`_EXPORT_FIG_SIZE_WITH_SLICES` for a 2D contour that is
        currently showing extracted slices, which is a two-panel
        layout rather than one square contour."""
        if self._contour_spectrum is not None and self._contour_slices:
            return self._EXPORT_FIG_SIZE_WITH_SLICES
        return self._EXPORT_FIG_SIZE

    def _contour_style_kwargs(self) -> dict:
        """The Output tab's own live style, in the form
        :func:`opennmr.plotting.plot_2d_contour` takes it.

        Real user report: "scrolling the contours redraws the axis with
        defaults, not the ones set in output". Every contour rebuild --
        including the one a contour-level scroll triggers -- went
        through ``plot_2d_contour`` without these, so it fell back to
        that function's own manuscript defaults (fontsize 12, tick
        width 1.0) and silently undid the Output tab. The 1D views
        never had this problem because they read
        ``self._AXIS_LABELSIZE`` and friends directly."""
        return {
            "fontsize": self._AXIS_LABELSIZE,
            "tick_length": self._TICK_LENGTH,
            "tick_width": self._TICK_WIDTH,
        }

    def _apply_current_axis_style(self) -> None:
        """Apply the current linewidth/tick/label style to every axes
        on the figure. Called both by :meth:`set_style` (so a change is
        visible immediately) and after every contour rebuild (so a
        redraw doesn't revert to ``plot_2d_contour``'s own defaults --
        ``_contour_style_kwargs`` covers what that function accepts,
        this covers the rest, spine linewidth and minor ticks)."""
        for ax in self.figure.get_axes():
            for spine in ax.spines.values():
                spine.set_linewidth(self._AXIS_LINEWIDTH)
            ax.tick_params(which="major", length=self._TICK_LENGTH,
                           width=self._TICK_WIDTH, labelsize=self._AXIS_LABELSIZE)
            if self._minor_ticks_visible:
                ax.minorticks_on()
                ax.tick_params(which="minor", length=self._TICK_LENGTH * 0.5,
                               width=self._TICK_WIDTH * 0.75)
            else:
                ax.minorticks_off()
            ax.xaxis.label.set_fontsize(self._AXIS_LABELSIZE)
            ax.yaxis.label.set_fontsize(self._AXIS_LABELSIZE)
            if ax.get_title():
                ax.title.set_fontsize(self._AXIS_LABELSIZE)

    def set_style(self, *, linewidth=None, labelsize=None, tick_length=None, tick_width=None,
                  minor_ticks=None, grid=None, legend_loc=None, compare_colormap=None) -> None:
        """The Output tab's live style panel -- updates whichever of
        these aren't ``None``, both as the new per-instance default
        (every ``show_*`` method reads ``self._AXIS_LINEWIDTH`` etc.,
        so a future redraw picks these up automatically) and applied
        directly to every axes already on screen right now (so the
        change is visible immediately, not just after the next
        reselect/redraw). Applies across every axes in the figure
        (not just ``self.ax``), so it covers multi-axes views (a
        "Stacked" Compare, a 2D contour's projection panels) too --
        unlike :meth:`_apply_view_options`'s legend/grid/aspect, which
        are inherently single-axes concepts.

        Deliberately does **not** cover DPI/figure size -- see
        :meth:`set_export_settings` for why those are export-only,
        never applied to this live embedded figure directly.
        """
        if linewidth is not None:
            self._AXIS_LINEWIDTH = linewidth
        if labelsize is not None:
            self._AXIS_LABELSIZE = labelsize
        if tick_length is not None:
            self._TICK_LENGTH = tick_length
        if tick_width is not None:
            self._TICK_WIDTH = tick_width
        if minor_ticks is not None:
            self._minor_ticks_visible = minor_ticks
        if grid is not None:
            self._grid_visible = grid
        if legend_loc is not None:
            self._legend_loc = legend_loc
        if compare_colormap is not None:
            self._compare_colormap = compare_colormap

        # Everything except grid/legend is the same "apply the current
        # style to every axes" the contour rebuild needs, so it lives in
        # one place rather than being written twice. Applying all of it
        # unconditionally is equivalent to the old per-argument guards,
        # since anything not passed in is re-applied at its existing
        # value.
        self._apply_current_axis_style()
        for ax in self.figure.get_axes():
            if grid is not None:
                ax.grid(grid, **({"alpha": 0.3} if grid else {}))
            if legend_loc is not None and ax.get_legend() is not None:
                ax.legend(loc=legend_loc)

        # compare_colormap only takes effect on the next full
        # show_compare redraw (it re-derives every trace's colour from
        # scratch) -- nothing to update on the already-drawn axes here.

        self.canvas.draw_idle()

    def set_export_settings(self, *, dpi=None, fig_width=None, fig_height=None) -> None:
        """DPI/figure size (inches) for **exported** figures only --
        stored, never applied to this live embedded canvas's own
        figure. Deliberately not part of :meth:`set_style`: matplotlib
        figure-size handling and Qt's own automatic canvas-widget
        resizing fight each other if the embedded figure's size is
        changed out from under Qt (observed directly: the live plot
        visibly misbehaves the moment a Qt resize and a matplotlib
        ``set_size_inches`` disagree about how big the canvas is
        supposed to be). Applied only around the actual save call, by
        :meth:`export_full_resolution`, then restored immediately
        after."""
        if dpi is not None:
            self._export_dpi = dpi
        if fig_width is not None:
            self._export_fig_width = fig_width
        if fig_height is not None:
            self._export_fig_height = fig_height

    def _prompt_contour_levels(self) -> None:
        """Right-click "Set Contour Levels…" -- lets the user directly
        set the number of contour levels and how they're spaced
        (power-law, with a tunable exponent, or evenly linear)
        instead of only ever scrolling the noise floor up/down."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Set Contour Levels")
        form = QFormLayout(dialog)

        clevels_spin = QSpinBox()
        clevels_spin.setRange(2, 200)
        clevels_spin.setValue(self._contour_clevels)
        form.addRow("Number of contours:", clevels_spin)

        start_spin = QDoubleSpinBox()
        start_spin.setRange(self._CONTOUR_FACTOR_MIN, self._CONTOUR_FACTOR_MAX)
        start_spin.setDecimals(6)
        start_spin.setSingleStep(0.001)
        start_spin.setValue(self._contour_factor)
        start_spin.setToolTip(
            "Lowest contour level, as a fraction of the data's own peak "
            "intensity -- the noise floor. Same value scroll-over-the-"
            "contour adjusts; the top level is always the peak itself."
        )
        form.addRow("Start (fraction of peak):", start_spin)

        mode_combo = QComboBox()
        mode_combo.addItems(["Power", "Linear"])
        mode_combo.setCurrentIndex(0 if self._contour_level_mode == "power" else 1)
        form.addRow("Scaling:", mode_combo)

        power_spin = QDoubleSpinBox()
        power_spin.setRange(0.1, 10.0)
        power_spin.setSingleStep(0.05)
        power_spin.setValue(self._contour_power)
        power_spin.setToolTip(
            "Higher values bunch more levels near the noise floor -- "
            "good for finding weak peaks near a strong one without "
            "saturating the display. Only used for Power scaling."
        )
        power_spin.setEnabled(mode_combo.currentIndex() == 0)
        mode_combo.currentIndexChanged.connect(lambda i: power_spin.setEnabled(i == 0))
        form.addRow("Power exponent:", power_spin)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            level_mode = "power" if mode_combo.currentIndex() == 0 else "linear"
            self.set_contour_levels(clevels_spin.value(), level_mode, power_spin.value(), start_spin.value())

    @staticmethod
    def _nearest_slice_index(spectrum: NMRSpectrum, axis: str, cursor_value: float) -> int:
        """The 1-based row (``axis="f1"``) or column (``"f2"``) index
        nearest a cursor's data-coordinate value -- ``cursor_value`` is
        the Y position (F1) when fixing a row, or the X position (F2)
        when fixing a column, matching :func:`opennmr.processing.
        slicing.get_trace`'s own ``axis`` convention. A genuine
        frequency axis is matched by ppm (same nearest-value approach
        ``get_trace``'s own ``shift`` parameter already uses); a
        pseudo-2D's index-kind F1 axis has no ppm at all, but its own
        plotted Y value already *is* approximately the 1-based slice
        number, so just rounds and clips that instead."""
        spectrum_axis = spectrum.axes[0] if axis == "f1" else spectrum.axes[1]
        n = len(spectrum_axis)
        if spectrum_axis.kind == "frequency":
            values = spectrum_axis.values("ppm")
            return int(np.argmin(np.abs(values - cursor_value))) + 1
        return int(np.clip(round(cursor_value), 1, n))

    def _prompt_extract_slice(self) -> None:
        """Right-click "Extract Slice…" -- pulls a single row (fixed
        F1) or column (fixed F2) out of the current 2D/pseudo-2D
        spectrum as its own 1D trace and shows it in place of the
        contour (the same "temporarily replace this canvas's view"
        pattern the Processing tab's pseudo-2D "Stacked slices" view
        already uses) -- there was previously no way to isolate and
        look at just one row/column on its own, for either a
        conventional 2D spectrum or a pseudo-2D series without
        retained imaginary data (the phase-preview slice spinner needs
        that).

        The index defaults to wherever the user actually right-clicked
        (:attr:`_last_right_click_xy`, real roadmap ask: "row/slice
        extraction should default to the cursor's current position
        rather than a fixed index") -- falls back to 1 (the previous,
        fixed default) if the menu was somehow opened without a
        tracked right-click position (e.g. via the keyboard)."""
        spectrum = self._contour_spectrum
        if spectrum is None:
            return
        n_f1, n_f2 = spectrum.data.shape
        cursor_xy = self._last_right_click_xy

        dialog = QDialog(self)
        dialog.setWindowTitle("Extract Slice")
        form = QFormLayout(dialog)

        axis_combo = QComboBox()
        axis_combo.addItem("Row (fix F1, trace along F2)", "f1")
        axis_combo.addItem("Column (fix F2, trace along F1)", "f2")
        if self._contour_spectrum is not None and self._contour_spectrum.is_pseudo_2d:
            # Pre-existing crash, found while adding "Keep Slice": for a
            # pseudo-2D, F1 is a slice *index*, not a frequency, so a
            # trace along it has an index axis and plot_1d raises
            # "This axis has no 'ppm' values (kind='index')". Disabled
            # rather than removed, so the reason is visible instead of
            # the option silently not being there. (Tracing a pseudo-2D
            # along F1 is a decay curve, not a spectrum -- the Analysis
            # tab is where that belongs.)
            item = axis_combo.model().item(axis_combo.count() - 1)
            item.setEnabled(False)
            axis_combo.setItemData(
                axis_combo.count() - 1,
                "Not available for a pseudo-2D: F1 is a slice index, not a frequency, "
                "so a trace along it is a decay curve rather than a spectrum. Use the "
                "Analysis tab for that.",
                Qt.ItemDataRole.ToolTipRole,
            )
        form.addRow("Axis:", axis_combo)

        index_spin = QSpinBox()
        index_spin.setRange(1, n_f1)

        def _default_index(axis: str) -> int:
            if cursor_xy is None:
                return 1
            cursor_x, cursor_y = cursor_xy
            return self._nearest_slice_index(spectrum, axis, cursor_y if axis == "f1" else cursor_x)

        index_spin.setValue(_default_index("f1"))
        form.addRow("Index:", index_spin)

        # Two genuinely different things to do with a slice, so this is
        # a mode rather than a second menu entry: "replace" is the
        # original behaviour (swap the canvas over to the trace, to
        # phase or fit it); "panel" keeps the contour and draws the
        # slice beside it. User request: "the option to plot slices from
        # 2D contours with a dashed line across the contour slice and
        # then projecting out".
        mode_combo = QComboBox()
        mode_combo.addItem("Show beside the contour (keep contour)", "panel")
        mode_combo.addItem("Replace the view with this slice", "replace")
        form.addRow("Show as:", mode_combo)

        label_edit = QLineEdit()
        label_edit.setPlaceholderText("optional, e.g. Impurity")
        label_edit.setToolTip(
            "Labels the trace in the panel, in the same colour as its "
            "dashed line on the contour."
        )
        form.addRow("Label:", label_edit)

        def _sync_mode(i: int) -> None:
            label_edit.setEnabled(mode_combo.itemData(i) == "panel")
        mode_combo.currentIndexChanged.connect(_sync_mode)

        def _sync_range(i: int) -> None:
            axis = axis_combo.itemData(i)
            index_spin.setRange(1, n_f1 if axis == "f1" else n_f2)
            index_spin.setValue(_default_index(axis))
        axis_combo.currentIndexChanged.connect(_sync_range)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        if mode_combo.currentData() == "panel":
            self._add_contour_slice(
                axis_combo.currentData(), index_spin.value(), label_edit.text().strip(),
            )
        else:
            self.extract_slice(axis_combo.currentData(), index_spin.value())

    def _emit_slice_keep_requested(self) -> None:
        """Hand the displayed slice to whoever is listening (the main
        window adds it to the session). Shared by the toolbar button
        and the context-menu action so the two can't diverge."""
        if self._extracted_trace is not None:
            self.sliceKeepRequested.emit((
                self._extracted_trace, self._extracted_label,
                self._extracted_from_spectrum, self._extracted_axis, self._extracted_index,
            ))

    def _show_contour_again(self) -> None:
        """Back to the 2D this slice came from. Reads the source out
        *before* calling show_spectrum, which clears it at its own top
        (correct for every other caller, since the content genuinely
        changes) -- reading afterwards would already see ``None``."""
        source = self._extracted_from_spectrum
        if source is not None:
            self.show_spectrum(source)

    def _refresh_slice_bar(self) -> None:
        """Show the slice action bar exactly while a slice is being
        viewed. Driven from one place so the bar can't be left showing
        over an unrelated view."""
        showing = self._extracted_trace is not None
        self._slice_bar.setVisible(showing)
        if showing:
            self._slice_label.setText(f"Viewing {self._extracted_label} \u2014 not yet in the spectrum list")

    def extract_slice(self, axis: str, index: int) -> None:
        """Pull a single row (``axis="f1"``) or column (``"f2"``) out
        of the current 2D/pseudo-2D spectrum as its own 1D trace, via
        :func:`opennmr.processing.slicing.get_trace`, and show it in
        place of the contour -- right-click "Show Contour" (see
        ``_on_context_menu``) gets back to the contour it came from."""
        # Fall back to the spectrum a slice already on screen came
        # from, so extracting a *second* slice works while looking at
        # the first. show_spectrum(trace) clears _contour_spectrum (the
        # trace is 1D), so before this the source was None on any
        # re-slice and get_trace raised "expects a 2D NMRSpectrum" --
        # which made the right-click "Extract Slice..." a one-shot, and
        # silently pinned peak-picking to whichever slice it started on.
        source = self._contour_spectrum
        if source is None:
            source = self._extracted_from_spectrum
        if source is None:
            raise ValueError("No 2D spectrum to extract a slice from.")
        trace = processing_slicing.get_trace(source, axis=axis, index=index)
        self.show_spectrum(trace)
        self._extracted_from_spectrum = source
        # Kept so "Keep Slice (for phasing/fitting)" can hand the exact
        # trace being displayed to the session, rather than recomputing
        # it from arguments the context menu would have to carry.
        self._extracted_trace = trace
        self._extracted_label = f"{axis.upper()} slice {index}"
        # Provenance, so a kept slice can feed its phase back into the
        # 2D it came from rather than being a dead end.
        self._extracted_axis = axis
        self._extracted_index = index
        self._refresh_slice_bar()

    @property
    def axis_unit(self) -> str:
        """The current 1D view's x-axis unit (``"ppm"``/``"hz"``/
        ``"khz"``) -- lets a caller that needs to redraw the spectrum
        (e.g. after an in-place axis edit like a reference shift) save
        this first and restore it via :meth:`set_axis_unit` afterward,
        instead of a full :meth:`show_spectrum` silently resetting the
        view back to ppm."""
        return self._axis_unit

    def set_imaging_gradient_params(self, gamma_hz_per_t: float | None, max_gradient_g_per_cm: float | None) -> None:
        """The two numbers :meth:`set_axis_unit`'s ``"mm"`` option
        needs to convert a frequency offset into a position -- set by
        the Imaging Analysis sub-tab (gamma from the selected
        spectrum's own nucleus via
        :func:`opennmr.core.registry.find_gamma`, max gradient
        user-entered, since it isn't recorded anywhere in the acquired
        data itself). Either ``None`` disables the ``"mm"`` menu
        option entirely rather than silently producing a wrong result
        from a stale/missing value."""
        self._imaging_gamma_hz_per_t = gamma_hz_per_t
        self._imaging_max_gradient_g_per_cm = max_gradient_g_per_cm

    def set_axis_unit(self, unit: str) -> None:
        """Switch the current 1D view's x-axis between ``"ppm"``,
        ``"hz"``, ``"khz"``, and ``"mm"``, re-using the spectrum's own
        stored axis arrays (no recomputation, except for ``"mm"``,
        which is derived from the Hz axis) -- only available while a
        plain 1D spectrum/slice (not a contour, FID, compare, or fit
        view) is the current view (see :attr:`_current_1d_spectrum`).
        ``"mm"`` additionally needs :meth:`set_imaging_gradient_params`
        to have been called with real values -- silently ignored
        (falls through to whatever unit was already showing) if not,
        matching this method's existing "can't do this yet" behaviour
        for every other precondition it checks."""
        if self._current_1d_spectrum is None or unit not in ("ppm", "hz", "khz", "mm") or not self.ax.lines:
            return
        axis = self._current_1d_spectrum.axes[0]
        if unit == "ppm":
            x = axis.values("ppm")
            label = self._ppm_label
        elif unit == "hz":
            x = axis.values("hz")
            label = "Frequency / Hz"
        elif unit == "khz":
            x = axis.values("hz") / 1000.0
            label = "Frequency / kHz"
        else:
            gamma = self._imaging_gamma_hz_per_t
            gradient = self._imaging_max_gradient_g_per_cm
            if not gamma or not gradient:
                return
            # G/cm -> T/m (1 G = 1e-4 T, 1 cm = 1e-2 m) -> Hz/m (via
            # gamma) -> Hz/mm -- spelled out in separate steps rather
            # than one folded constant, so the unit conversion is
            # checkable at a glance rather than needing to be trusted.
            gradient_t_per_m = gradient * 1e-2
            hz_per_m = gamma * gradient_t_per_m
            hz_per_mm = hz_per_m * 1e-3
            x = axis.values("hz") / hz_per_mm
            label = "Position / mm"

        for line in self.ax.lines:
            line.set_xdata(x)
        self.ax.set_xlim(float(np.min(x)), float(np.max(x)))
        self.ax.invert_xaxis()
        self.ax.set_xlabel(label)
        self._axis_unit = unit
        self.canvas.draw_idle()

    def _apply_view_options(self) -> None:
        # Every show_* rebuild routes through here after its own
        # figure.clear() -- any pivot marker from before is gone with
        # it, so the stale Python-side reference must be dropped too
        # (set_pivot_marker would otherwise try to move a Line2D no
        # longer attached to any axes).
        self._pivot_line = None
        self._pivot_marker_ax = None
        self._peak_marker_lines = []
        if self._legend_visible and self.ax.get_legend_handles_labels()[0]:
            self.ax.legend(loc=self._legend_loc)
        else:
            existing = self.ax.get_legend()
            if existing is not None:
                existing.remove()
        # A 2D/pseudo-2D contour panel's own default square box (real
        # roadmap ask: "the contour panel itself should be square...
        # automatic/default for 2D") is applied by plot_2d_contour's own
        # square_aspect (via the same AxesDivider that lays out the F1/
        # F2 projections, see that function's own docstring for why --
        # box_aspect alone shrinks the main axes in isolation, stranding
        # any sibling projection axes at their pre-shrink position), not
        # here. This still applies the user's own general, explicit
        # _aspect_ratio lock (None when unlocked -- a no-op layered on
        # top of whatever plot_2d_contour's own divider already set).
        self.ax.set_box_aspect(self._aspect_ratio)
        # matplotlib's own Axes.grid(visible, **kwargs) quirk: passing
        # any style kwarg (even alongside visible=False) is treated as
        # an implicit "turn it on" and warns -- only pass alpha when
        # actually enabling it.
        self.ax.grid(self._grid_visible, **({"alpha": 0.3} if self._grid_visible else {}))

    def set_pivot_marker(self, ppm: float | None) -> None:
        """Show, move, or hide a dashed vertical line at ``ppm`` -- the
        Phase 1 pivot point, visible while phasing instead of only a
        number in a spin box. ``None`` hides it. A no-op when the
        current view has nowhere sensible to draw it (see
        :attr:`_pivot_marker_ax`'s own docstring -- a 2D contour, FID,
        or Compare view, or the Processing-preview's own FID panel)."""
        if self._pivot_marker_ax is None:
            return
        if ppm is None:
            if self._pivot_line is not None:
                self._pivot_line.remove()
                self._pivot_line = None
                self.canvas.draw_idle()
            return
        if self._pivot_line is None:
            self._pivot_line = self._pivot_marker_ax.axvline(
                ppm, linestyle="--", color="darkorange", linewidth=1.2, zorder=5,
            )
        else:
            self._pivot_line.set_xdata([ppm, ppm])
        self.canvas.draw_idle()

    def set_peak_markers(self, shifts: list[float] | None) -> None:
        """Show or clear a small downward-pointing triangle just above
        the trace at each ppm in ``shifts`` -- visual confirmation of
        what Auto-Pick Peaks (or the peak table generally) actually
        found on the spectrum itself, not just as numbers in the table.
        ``None`` or an empty list clears every marker. A no-op (markers
        just cleared) when the current view has nowhere sensible to
        draw them -- same gate as :meth:`set_pivot_marker` -- or when
        there's no known 1D trace to measure peak heights against."""
        for marker in self._peak_marker_lines:
            marker.remove()
        self._peak_marker_lines = []
        if not shifts or self._pivot_marker_ax is None or self._current_1d_spectrum is None:
            self.canvas.draw_idle()
            return
        ppm = self._current_1d_spectrum.axis_values(0, "ppm")
        y_data = self._current_1d_spectrum.normalized()
        ymin, ymax = self._pivot_marker_ax.get_ylim()
        offset = 0.04 * (ymax - ymin)
        for shift in shifts:
            idx = int(np.argmin(np.abs(ppm - shift)))
            marker = self._pivot_marker_ax.plot(
                [shift], [y_data[idx] + offset], marker="v", color="crimson",
                markersize=8, linestyle="none", zorder=6, clip_on=False,
            )[0]
            self._peak_marker_lines.append(marker)
        self.canvas.draw_idle()

    def clear(self) -> None:
        # figure.clear() below invalidates any existing pivot-line/peak-
        # marker Line2D objects the same way every show_* rebuild's own
        # figure.clear() does (see _apply_view_options) -- drop the
        # stale Python-side references too, or a later set_pivot_marker/
        # set_peak_markers call would try to .remove() an object no
        # longer attached to any axes.
        self._pivot_line = None
        self._pivot_marker_ax = None
        self._peak_marker_lines = []
        self._contour_spectrum = None
        self._contour_factor = None
        self._proj_axes = {"f1": None, "f2": None}
        self._compare_axes = None
        self._fit_mode = False
        self._selected_compare_index = None
        self._freeform_zoom_axes = set()
        self._processing_preview_fid_ax = None
        self._processing_preview_spec_ax = None
        self._current_1d_spectrum = None
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        self.canvas.draw_idle()

    def show_spectrum(self, spectrum: NMRSpectrum) -> None:
        """Plot ``spectrum`` into this canvas, replacing whatever was
        there -- 1D via :func:`opennmr.plotting.plot_1d`, 2D via
        :func:`opennmr.plotting.plot_2d_contour`. Preserves this
        canvas's legend-visible/aspect-lock choices across the
        rebuild."""
        self.figure.clear()
        self._extracted_from_spectrum = None
        self._extracted_trace = None
        self._extracted_label = ""
        self._extracted_axis = ""
        self._extracted_index = 0
        self._refresh_slice_bar()

        if spectrum.ndim == 1:
            self._contour_spectrum = None
            self._contour_factor = None
            self._proj_axes = {"f1": None, "f2": None}
            self._compare_axes = None
            self._fit_mode = False
            self._selected_compare_index = None
            self._freeform_zoom_axes = set()
            self._processing_preview_fid_ax = None
            self._processing_preview_spec_ax = None
            self._current_1d_spectrum = spectrum
            self._axis_unit = "ppm"  # a fresh spectrum always starts in ppm
            self.ax = self.figure.add_subplot(111)
            xmin, xmax, labeltext = plotting.plot_1d(self.ax, spectrum, full_resolution=self._full_resolution)
            self._ppm_label = labeltext
            self.ax.lines[0].set_label(_spectrum_label(spectrum))
            self.ax.set_xlim(xmin, xmax)
            self.ax.invert_xaxis()
            self.ax.set_xlabel(labeltext, fontsize=self._AXIS_LABELSIZE)
            self.ax.set_yticks([])
            plotting.apply_spine_style(self.ax, top=False, right=False, bottom=True, left=False,
                                        linewidth=self._AXIS_LINEWIDTH)
            plotting.apply_tick_style(self.ax, axis="x", length=self._TICK_LENGTH, width=self._TICK_WIDTH, minor=self._minor_ticks_visible,
                                       labelsize=self._AXIS_LABELSIZE)
        elif spectrum.ndim == 2:
            self._current_1d_spectrum = None
            self._contour_spectrum = spectrum
            self._contour_factor = self._CONTOUR_FACTOR_DEFAULT
            self._contour_clevels = self._CONTOUR_CLEVELS_DEFAULT
            self._contour_level_mode = self._CONTOUR_LEVEL_MODE_DEFAULT
            self._contour_power = self._CONTOUR_POWER_DEFAULT
            self.ax = self.figure.add_subplot(111)
            _, _, self._proj_axes = plotting.plot_2d_contour(
                spectrum, ax=self.ax, fig=self.figure, factor=self._contour_factor,
                clevels=self._contour_clevels, level_mode=self._contour_level_mode, power=self._contour_power,
                square_aspect=self._aspect_ratio is None,
                **self._contour_fraction_kwargs(), **self._nq_line_kwargs(),
                **self._slice_panel_kwargs(), **self._contour_style_kwargs(),
            )
            self._apply_current_axis_style()
        else:
            raise ValueError(f"Cannot display a spectrum with ndim={spectrum.ndim}")

        self._apply_view_options()
        if spectrum.ndim == 1:
            self._pivot_marker_ax = self.ax
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def show_fid(self, fid, dt: float, title: str = "", window_envelope=None, tdeff_index: int | None = None,
                 echo_top_index: float | None = None, mode: str = "magnitude", x_unit: str = "time") -> None:
        """Plot a raw/processed time-domain FID vs. time -- an
        alternate view (the FID canvas tab) for judging FID-domain
        processing directly, instead of only its effect after FFT. The
        frequency-domain spectrum (:func:`show_spectrum`) stays the
        default (Spectrum tab) view.

        ``x_unit`` : ``"time"`` (default, seconds) or ``"points"``
        (raw sample index) -- ``tdeff_index``/``echo_top_index`` are
        always given as sample indices either way, scaled by ``dt``
        only when ``x_unit="time"``.

        ``mode`` : ``"magnitude"`` (default, ``|fid|`` -- a single
        clean decay/echo envelope, easiest for judging alignment/decay
        shape) or ``"real"`` (one oscillating component, closer to
        what phase-sensitive detection actually recorded). Plots one
        line either way, not both real and imaginary together.

        ``window_envelope`` : the currently-selected apodization
        window's own shape (real-valued, same length as ``fid``, from
        applying it to an all-ones array) -- overlaid scaled to the
        FID's own peak magnitude, so a window/line-broadening choice
        can be judged directly against the data it will taper,
        without needing to click "Reprocess" first.

        ``tdeff_index`` : if given (and less than ``len(fid)``), draws
        a dashed vertical marker at that point in time -- where a
        TDeff truncation would cut the FID, previewed live as the
        control changes, before "Reprocess" is clicked.

        ``echo_top_index`` : if given, draws a dashed vertical marker
        at that point (a *fractional* sample index, from
        :func:`opennmr.processing.ft.find_echo_top` -- can be
        non-integer) -- visual confirmation of where "Align Echo" found
        the echo peak. Should land right at t=0 once actually aligned;
        if it doesn't, that's a direct, visible sign the alignment
        isn't doing what's expected, instead of a checkbox silently
        changing a setting with no feedback either way."""
        self._contour_spectrum = None
        self._contour_factor = None
        self._proj_axes = {"f1": None, "f2": None}
        self._compare_axes = None
        self._fit_mode = False
        self._selected_compare_index = None
        self._freeform_zoom_axes = set()
        self._processing_preview_fid_ax = None
        self._processing_preview_spec_ax = None
        self._current_1d_spectrum = None
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        self._freeform_zoom_axes = {self.ax}
        if x_unit not in ("time", "points"):
            raise ValueError("x_unit must be 'time' or 'points'")
        scale = dt if x_unit == "time" else 1.0
        t = np.arange(len(fid)) * scale
        if mode not in ("magnitude", "real"):
            raise ValueError("mode must be 'magnitude' or 'real'")
        y = np.real(fid) if mode == "real" else np.abs(fid)
        self.ax.plot(t, y, label="Real" if mode == "real" else "Magnitude")
        if window_envelope is not None:
            peak = np.max(np.abs(fid))
            self.ax.plot(t, np.asarray(window_envelope) * peak, "--", color="gray", label="Window", zorder=0)
            self._legend_visible = True
        if tdeff_index is not None and 0 < tdeff_index < len(fid):
            self.ax.axvline(tdeff_index * scale, linestyle="--", color="red", label="TDeff cutoff")
            self._legend_visible = True
        if echo_top_index is not None and 0 <= echo_top_index < len(fid):
            self.ax.axvline(echo_top_index * scale, linestyle="--", color="green", label="Echo top")
            self._legend_visible = True
        self.ax.set_xlabel("Time (s)" if x_unit == "time" else "Points", fontsize=self._AXIS_LABELSIZE)
        self.ax.set_ylabel("Amplitude", fontsize=self._AXIS_LABELSIZE)
        if title:
            self.ax.set_title(title, fontsize=self._AXIS_LABELSIZE)
        plotting.apply_spine_style(self.ax, top=False, right=False, bottom=True, left=True,
                                    linewidth=self._AXIS_LINEWIDTH)
        plotting.apply_tick_style(self.ax, length=self._TICK_LENGTH, width=self._TICK_WIDTH, minor=self._minor_ticks_visible,
                                   labelsize=self._AXIS_LABELSIZE)
        self._apply_view_options()
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def show_processing_preview(self, fid, dt: float, spectrum: NMRSpectrum, *, title: str = "",
                                 window_envelope=None, tdeff_index: int | None = None,
                                 echo_top_index: float | None = None, fid_mode: str = "magnitude",
                                 fid_x_unit: str = "time") -> None:
        """The Processing tab's own live dual view: the FID (top,
        exactly like :meth:`show_fid`'s own overlays) and a downsampled
        preview of what FFT'ing it with the currently-dialed-in
        settings would look like (bottom) -- a *speculative* preview of
        a not-yet-applied choice, always shown here, distinct from the
        FID and Spectrum canvas tabs (which only ever show the
        actually-loaded spectrum, i.e. the real result of the last
        "Apply Processing"/selection -- explicit user feedback that
        conflating the two, by having this live-tweak preview
        temporarily commandeer the FID tab, made those tabs unreliable
        for their own actual job, e.g. judging a live phase drag).

        Always downsampled (regardless of :attr:`_full_resolution`) --
        this view is about responsiveness while tweaking, not final
        inspection; the committed Spectrum tab view is unaffected and
        stays full-resolution as normal."""
        self._contour_spectrum = None
        self._contour_factor = None
        self._proj_axes = {"f1": None, "f2": None}
        self._compare_axes = None
        self._fit_mode = False
        self._selected_compare_index = None
        self._freeform_zoom_axes = set()
        self._processing_preview_fid_ax = None
        self._processing_preview_spec_ax = None
        self._current_1d_spectrum = None
        self.figure.clear()
        fid_ax, spec_ax = self.figure.subplots(2, 1)
        self.ax = spec_ax
        self._freeform_zoom_axes = {fid_ax}
        self._processing_preview_fid_ax = fid_ax
        self._processing_preview_spec_ax = spec_ax

        if fid_x_unit not in ("time", "points"):
            raise ValueError("fid_x_unit must be 'time' or 'points'")
        fid_scale = dt if fid_x_unit == "time" else 1.0
        t = np.arange(len(fid)) * fid_scale
        if fid_mode not in ("magnitude", "real"):
            raise ValueError("fid_mode must be 'magnitude' or 'real'")
        y = np.real(fid) if fid_mode == "real" else np.abs(fid)
        fid_ax.plot(t, y, color="black", linewidth=1.0)
        if window_envelope is not None:
            peak = np.max(np.abs(fid)) if len(fid) else 0.0
            fid_ax.plot(t, np.asarray(window_envelope) * peak, "--", color="gray", linewidth=1.0, zorder=0)
        if tdeff_index is not None and 0 < tdeff_index < len(fid):
            fid_ax.axvline(tdeff_index * fid_scale, linestyle="--", color="red")
        if echo_top_index is not None and 0 <= echo_top_index < len(fid):
            fid_ax.axvline(echo_top_index * fid_scale, linestyle="--", color="green")
        fid_ax.set_xlabel("Time (s)" if fid_x_unit == "time" else "Points", fontsize=self._AXIS_LABELSIZE)
        fid_ax.set_ylabel("FID Intensity", fontsize=self._AXIS_LABELSIZE)
        if title:
            fid_ax.set_title(title, fontsize=self._AXIS_LABELSIZE)
        plotting.apply_spine_style(fid_ax, top=False, right=False, bottom=True, left=True,
                                    linewidth=self._AXIS_LINEWIDTH)
        plotting.apply_tick_style(fid_ax, length=self._TICK_LENGTH, width=self._TICK_WIDTH, minor=self._minor_ticks_visible,
                                   labelsize=self._AXIS_LABELSIZE)

        xmin, xmax, labeltext = plotting.plot_1d(spec_ax, spectrum, full_resolution=False)
        spec_ax.set_xlim(xmin, xmax)
        spec_ax.invert_xaxis()
        spec_ax.set_xlabel(labeltext, fontsize=self._AXIS_LABELSIZE)
        spec_ax.set_yticks([])
        plotting.apply_spine_style(spec_ax, top=False, right=False, bottom=True, left=False,
                                    linewidth=self._AXIS_LINEWIDTH)
        plotting.apply_tick_style(spec_ax, axis="x", length=self._TICK_LENGTH, width=self._TICK_WIDTH, minor=self._minor_ticks_visible,
                                   labelsize=self._AXIS_LABELSIZE)

        self._apply_view_options()
        self._pivot_marker_ax = spec_ax
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def update_processing_preview_spectrum(self, spectrum: NMRSpectrum) -> None:
        """Refresh the Processing-preview's spectrum panel (bottom) in
        place, preserving whatever zoom/pan is already there -- unlike
        :meth:`show_processing_preview`, which resets both panels'
        views from scratch. For a phase change specifically (drag,
        spin box, autophase, pivot): phase never touches the FID, only
        the already-FFT'd spectrum, so only this panel's y-data
        (real part) ever needs updating -- the FID panel above is left
        completely untouched, same reasoning as :meth:`update_data`.
        Explicit user request: being able to zoom into a small peak
        and phase it without the view snapping back out on every tick.

        Falls back to a full :meth:`show_processing_preview`-style
        rebuild being the caller's job instead -- this method no-ops
        and returns ``False`` (rather than guessing) if there's no
        processing-preview layout currently up to update in place at
        all, or if ``spectrum``'s length no longer matches what's
        already plotted (possible even for a "phase-only" refresh,
        since :meth:`MainWindow._compute_processing_preview` recomputes
        the whole window/zero-fill/phase pipeline from whatever the
        Processing tab's *live* controls currently show, not just
        phase -- a zero-fill change sitting unapplied while phase is
        being dragged would otherwise plot new y-values against stale,
        wrong-length x positions). Returns ``True`` on a successful
        in-place update."""
        if self._processing_preview_spec_ax is None or not self._processing_preview_spec_ax.lines:
            return False
        y = spectrum.normalized()
        idx = plotting.downsample_indices(len(y))
        new_ydata = y if idx is None else y[idx]
        if len(new_ydata) != len(self._processing_preview_spec_ax.lines[0].get_ydata()):
            return False
        self._processing_preview_spec_ax.lines[0].set_ydata(new_ydata)
        self.canvas.draw_idle()
        return True

    def show_compare(self, spectra: list[NMRSpectrum], mode: str = "stack", normalize: bool = False) -> None:
        """Plot several 1D spectra together for comparison.

        ``mode="overlay"`` : every spectrum on one shared axes at y=0
        (via :func:`opennmr.plotting.plot_1d_stack`'s own offset-
        collapsing trick) -- a shared y-scale is exactly what direct
        overlay comparison needs, so a y-zoom was never a problem
        there (there's only one baseline, always at y=0).

        ``mode="stack"`` (more than one spectrum): one subplot per
        spectrum instead, sharing only the x-axis, each with its own
        independent y-autoscale -- *not* a single shared axes with
        every trace's data offset by a constant (what
        :func:`opennmr.plotting.plot_1d_stack` itself still does, and
        what this used to do too). That single-shared-axes approach
        has a real, structural problem a linear axis can't avoid:
        every trace shares one y-scale, so any y-zoom multiplies
        distances from wherever it's pinned by the same factor
        everywhere -- including the gaps *between* different traces'
        own baselines, which visibly spread apart or squeeze together
        as you zoom into any one trace's own intensity, even though no
        data actually moved (reported directly as "when I rescale on
        the compare plot, it should keep each baseline in place").
        Separate subplots don't have this problem: each one's own
        y-zoom only ever touches that one subplot's own range, so
        every other trace's baseline -- literally a different axes'
        own y=0 -- can't move as a side effect. By default
        :meth:`_on_scroll` still zooms every row *together* (each by
        the same relative factor, still individually pinned at its own
        baseline) -- a linked-unless-selected model, not fully
        independent: click a row first (see
        :meth:`_toggle_compare_selection`, wired from :meth:`_on_release`)
        to zoom just that one on its own instead.

        No ``y_offset``/gap parameter (there used to be one, wired to
        a "Stack gap" spin box in the GUI) -- per-spectrum vertical
        spacing was a meaningful, user-adjustable idea back when every
        trace shared one axes and a manual offset was literally what
        separated them, but once ``mode="stack"`` moved to one
        independent, auto-scaled axes per spectrum (the redesign
        above), that control no longer did anything for the actual
        multi-spectrum case -- each row's height is already entirely
        automatic (matplotlib's own equal-height subplot layout,
        ``hspace=0.0``), and there is no shared y-axis left for a
        numeric offset to even mean anything on. The GUI's own "Stack
        gap" spin box only still visibly affected a degenerate case
        (``mode="stack"`` with exactly one spectrum, via the ``else``
        branch below) where a vertical offset is meaningless anyway
        (nothing else to offset from) -- removed rather than kept as a
        control with no real effect, matching the explicit ask that
        stacked-spectrum spacing "should be automatic"."""
        self._contour_spectrum = None
        self._contour_factor = None
        self._proj_axes = {"f1": None, "f2": None}
        self._compare_axes = None
        self._fit_mode = False
        self._selected_compare_index = None
        self._freeform_zoom_axes = set()
        self._processing_preview_fid_ax = None
        self._processing_preview_spec_ax = None
        self._current_1d_spectrum = None
        self.figure.clear()

        if mode == "stack" and len(spectra) > 1:
            axes = list(np.atleast_1d(self.figure.subplots(len(spectra), 1, sharex=True)))
            colors = plotting.stack_colors(len(spectra), self._compare_colormap)
            xmin = xmax = labeltext = None
            for i, (row_ax, spectrum) in enumerate(zip(axes, spectra)):
                this_xmin, this_xmax, labeltext = plotting.plot_1d(
                    row_ax, spectrum, normalize=normalize, color=colors[i], full_resolution=self._full_resolution,
                )
                # Let a tall peak spill into a neighbouring row instead
                # of being hard-clipped at this row's own axes edge --
                # explicit user request, matching how a classic NMR
                # stacked plot looks (one trace's peak visibly
                # overlapping the next). Two parts, both needed: the
                # line itself must not be clipped to its own axes, and
                # every row's own background must be transparent (not
                # opaque white) -- otherwise whichever row happens to
                # be drawn *later* (rows are drawn top-to-bottom, so
                # any row below the one bleeding into it) would just
                # paint over the bleed with its own opaque background,
                # hiding it again.
                row_ax.lines[-1].set_clip_on(False)
                row_ax.patch.set_alpha(0.0)
                xmin = this_xmin if xmin is None else min(xmin, this_xmin)
                xmax = this_xmax if xmax is None else max(xmax, this_xmax)
                row_ax.set_yticks([])
                is_bottom = row_ax is axes[-1]
                plotting.apply_spine_style(row_ax, top=False, right=False, bottom=is_bottom, left=False,
                                            linewidth=self._AXIS_LINEWIDTH)
                if is_bottom:
                    plotting.apply_tick_style(row_ax, axis="x", length=self._TICK_LENGTH,
                                               width=self._TICK_WIDTH, minor=self._minor_ticks_visible, labelsize=self._AXIS_LABELSIZE)
                else:
                    row_ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
            axes[0].set_xlim(xmin, xmax)
            axes[0].invert_xaxis()
            axes[-1].set_xlabel(labeltext, fontsize=self._AXIS_LABELSIZE)
            self.figure.subplots_adjust(hspace=0.0)
            self._compare_axes = axes
            self.ax = axes[-1]
        else:
            self.ax = self.figure.add_subplot(111)
            # -1.0 for "overlay": plot_1d_stack's own offset-collapsing
            # trick (y_base += y_scale * (1 + y_offset)) puts every
            # trace back at y=0 when y_offset == -1.0. 0.2 for "stack"
            # -- only ever reached with exactly one spectrum (see the
            # len(spectra) > 1 branch above), where a vertical offset
            # is inherently meaningless (nothing else to offset from);
            # kept as a fixed constant rather than exposed as a GUI
            # control for exactly that reason.
            offset = -1.0 if mode == "overlay" else 0.2
            plotting.plot_1d_stack(spectra, ax=self.ax, fig=self.figure, y_offset=offset, normalize=normalize,
                                    color=self._compare_colormap, full_resolution=self._full_resolution)
            self.ax.set_yticks([])
            plotting.apply_spine_style(self.ax, top=False, right=False, bottom=True, left=False,
                                        linewidth=self._AXIS_LINEWIDTH)
            plotting.apply_tick_style(self.ax, axis="x", length=self._TICK_LENGTH, width=self._TICK_WIDTH, minor=self._minor_ticks_visible,
                                       labelsize=self._AXIS_LABELSIZE)
            self.ax.xaxis.label.set_fontsize(self._AXIS_LABELSIZE)

        self._apply_view_options()
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def show_2d_overlay(self, spectra: list[NMRSpectrum], labels: list[str] | None = None) -> None:
        """Overlay 2-3 2D/pseudo-2D spectra as flat-colour line contours
        on one set of axes -- real roadmap ask: "a way to overlay
        multiple 2D contours." Thin wrapper around
        :func:`opennmr.plotting.twod.overlay_plot` (already ported from
        ``nmrtools2``'s own ``NMR2D_overlay_plot``, just never called
        from GUI code before this) -- that function does the actual
        compatibility checking (same F1/F2 nuclides, conventional vs.
        pseudo-2D, 2-3 spectra only) and raises ``ValueError``
        otherwise; this method doesn't duplicate those checks, the
        caller decides how to surface that error.

        Unlike :meth:`show_spectrum`'s own single-2D-contour view, this
        is *not* set as :attr:`_contour_spectrum` -- scroll-to-
        threshold, Set Contour Levels…, and Extract Slice… all assume
        one specific spectrum's own data, which an overlay of several
        doesn't have."""
        self.clear()  # already leaves self.ax as a fresh, single subplot
        plotting.overlay_plot(spectra, ax=self.ax, fig=self.figure, labels=labels, **self._contour_fraction_kwargs())
        if labels is not None:
            self.ax.legend()
            self._legend_visible = True
        self._apply_view_options()
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def show_1d_heatmap(self, spectra: list[NMRSpectrum]) -> None:
        """Plot several 1D spectra (a pseudo-2D series' own slices) as
        rows of a heatmap -- real roadmap ask: "pseudo-2D/stacked-1D
        data should be plottable as either a contour or a heatmap."
        Thin wrapper around :func:`opennmr.plotting.plot_1d_heatmap`
        (already ported from ``nmrtools2``'s own ``NMR1dheatmap``, just
        never called from GUI code before this).

        Not a :attr:`_contour_spectrum` either, same reasoning as
        :meth:`show_2d_overlay` -- this is several interpolated-onto-
        one-axis 1D traces, not one spectrum's own 2D array."""
        self.clear()  # already leaves self.ax as a fresh, single subplot
        plotting.plot_1d_heatmap(spectra, ax=self.ax, fig=self.figure)
        self._apply_view_options()
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def show_slice_fit(self, x, y_data, y_fit, components=None, peak_labels=None, title: str = "",
                        spectrum: NMRSpectrum | None = None) -> None:
        """Plot one pseudo-2D slice's spectrum alongside its fitted
        (pseudo-Voigt) curve -- the intermediate per-slice spectral
        fit that feeds a relaxation/diffusion parameter fit
        (:func:`opennmr.processing.fitting.NMR2D_fit_slices`'s own
        per-slice result), so it can be inspected directly instead of
        only trusting the final T1/T2/D number.

        ``components``, if given, is the per-peak curve list from
        :func:`opennmr.processing.fitting.evaluate_fit` (last entry
        is the baseline) -- each is drawn individually alongside the
        total fit, so an overlapping multi-peak fit can be judged
        peak by peak rather than only as a summed curve. Styled after
        the user's own DMFit plots (``leonmr.dmfit``): each peak is a
        shaded ``fill_between`` region (not a solid line) so its
        visual weight tracks its actual amplitude instead of every
        component looking equally prominent regardless of size, drawn
        tallest-peak-first so a small peak's shading still shows on
        top rather than being buried under a larger neighbour's fill;
        the total fit is a dashed line drawn above every peak.

        ``spectrum`` : the slice's own 1D trace (``fit_result["slice"]``
        from :func:`opennmr.processing.fitting.NMR2D_fit_slices`), used
        only for its nucleus/SFO1 -- to label and orient the x-axis
        exactly like :meth:`show_spectrum` does (inverted, high-to-low
        ppm, with a nucleus-based label) rather than the plain
        unlabelled, un-inverted axis this used to have. ``x`` itself is
        always ppm (:func:`NMR2D_fit_slices`'s own default)."""
        self._contour_spectrum = None
        self._contour_factor = None
        self._proj_axes = {"f1": None, "f2": None}
        self._compare_axes = None
        self._fit_mode = False
        self._selected_compare_index = None
        self._freeform_zoom_axes = set()
        self._processing_preview_fid_ax = None
        self._processing_preview_spec_ax = None
        self._current_1d_spectrum = None
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        self.ax.plot(x, y_data, color="black", linewidth=1.5, label="Data", zorder=0)
        if components:
            n_peaks = len(components) - 1  # evaluate_fit appends the baseline last
            peak_curves = components[:n_peaks]
            if peak_curves:
                order = sorted(range(len(peak_curves)), key=lambda i: float(np.max(peak_curves[i])), reverse=True)
                # A qualitative palette, not a diverging one -- BrBG/RdBu-style
                # colormaps wash out to near-white at their midpoint, which made
                # a peak of middling rank nearly invisible once shaded.
                cmap = colormaps["tab10"]
                edge_colors = [cmap(i % 10) for i in range(len(peak_curves))]
                for rank, i in enumerate(order):
                    label = peak_labels[i] if peak_labels else f"Peak {i + 1}"
                    edge = edge_colors[rank]
                    face = (edge[0], edge[1], edge[2], 0.4)
                    self.ax.fill_between(x, peak_curves[i], color=face, edgecolor=edge,
                                          linewidth=1.0, label=label, zorder=1)
            self.ax.plot(x, components[-1], color="gray", linestyle=":", linewidth=1.0, label="Baseline", zorder=1)
        self.ax.plot(x, y_fit, color="red", linestyle="--", linewidth=2.0, label="Fit (total)", zorder=100)

        if spectrum is not None:
            nuclide = find_gamma(spectrum.metadata.nucleus, SFO1=spectrum.metadata.sfo1)["Nuclide"]
            xlabel = NUC_label(nuclide, x_unit="ppm")
        else:
            xlabel = "Shift (ppm)"
        xmin, xmax = float(np.min(x)), float(np.max(x))
        self.ax.set_xlim(xmin, xmax)
        self.ax.invert_xaxis()
        self.ax.set_xlabel(xlabel, fontsize=self._AXIS_LABELSIZE)
        self.ax.set_yticks([])
        if title:
            self.ax.set_title(title, fontsize=self._AXIS_LABELSIZE)
        plotting.apply_spine_style(self.ax, top=False, right=False, bottom=True, left=False,
                                    linewidth=self._AXIS_LINEWIDTH)
        plotting.apply_tick_style(self.ax, axis="x", length=self._TICK_LENGTH, width=self._TICK_WIDTH, minor=self._minor_ticks_visible,
                                   labelsize=self._AXIS_LABELSIZE)
        self._legend_visible = True
        self._apply_view_options()
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def show_relaxation_fit(self, result: dict, uncertainty: tuple | None = None,
                             xlabel: str = "Delay (s)", excluded_points: tuple | None = None) -> None:
        """Plot ONE peak's relaxation/diffusion series (data points)
        and fitted curve, from a single ``relaxation.fit_relaxation``-
        shaped dict (one entry of
        :func:`opennmr.processing.relaxation.NMR2D_fit_relaxation`'s
        own list output). Only one peak at a time -- when a fit covers
        several, :class:`~opennmr.gui.main_window.MainWindow` browses
        between them via a spin box, same pattern as the Spectral Fit
        tab's own slice browser, rather than plotting them all
        together (which got cluttered fast past 2-3 peaks, and doesn't
        leave room for a per-peak uncertainty band anyway). Styled
        after the user's own PFG-fitting plots: black circles for the
        data, a dashed red fit line, and (when ``uncertainty`` is
        given) a shaded grey uncertainty band -- not a color-per-peak
        legend, which stops meaning anything once only one peak is
        ever shown.

        ``uncertainty`` : ``(t_fine, y_fine, y_band)`` from
        :func:`opennmr.processing.relaxation.fit_curve_uncertainty` --
        a smooth fit curve plus a +/-``y_band`` envelope, shaded via
        ``ax.fill_between``. When ``None`` (default), the fit curve is
        instead drawn straight through the (coarser) experimental
        t-points, same as this method's behaviour before it supported
        an uncertainty band at all.

        ``excluded_points`` : ``(t_excluded, y_excluded)`` -- points
        the user right-clicked out of the fit (see
        :meth:`_on_context_menu`'s "Exclude Nearest Point"), drawn as
        faint grey X markers instead of just disappearing, so it stays
        visible *what* was excluded and *that* something was, not only
        that the fit curve changed. Sets :attr:`_fit_mode`, gating the
        right-click actions themselves."""
        self._contour_spectrum = None
        self._contour_factor = None
        self._proj_axes = {"f1": None, "f2": None}
        self._compare_axes = None
        self._fit_mode = True
        self._selected_compare_index = None
        self._freeform_zoom_axes = set()
        self._processing_preview_fid_ax = None
        self._processing_preview_spec_ax = None
        self._current_1d_spectrum = None
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)

        t, y, y_fit = result["t"], result["y"], result["y_fit"]
        if uncertainty is not None:
            t_fine, y_fine, y_band = uncertainty
            self.ax.fill_between(
                t_fine, y_fine - y_band, y_fine + y_band,
                color="grey", alpha=0.4, label="Uncertainty", zorder=1,
            )
            self.ax.plot(t_fine, y_fine, "--", color="red", linewidth=2.0, label="Fit", zorder=3)
        else:
            order = np.argsort(t)
            self.ax.plot(t[order], y_fit[order], "--", color="red", linewidth=2.0, label="Fit", zorder=3)
        self.ax.plot(t, y, "o", color="black", markersize=8.0,
                      label=result.get("peak_label", result.get("model", "Data")), zorder=2)
        if excluded_points is not None and len(excluded_points[0]):
            t_excluded, y_excluded = excluded_points
            self.ax.plot(t_excluded, y_excluded, "x", color="grey", markersize=8.0,
                          markeredgewidth=2.0, alpha=0.7, label="Excluded", zorder=2)

        self.ax.set_xlabel(xlabel, fontsize=self._AXIS_LABELSIZE)
        self.ax.set_ylabel("Intensity", fontsize=self._AXIS_LABELSIZE)
        plotting.apply_spine_style(self.ax, top=False, right=False, bottom=True, left=True,
                                    linewidth=self._AXIS_LINEWIDTH)
        plotting.apply_tick_style(self.ax, length=self._TICK_LENGTH, width=self._TICK_WIDTH, minor=self._minor_ticks_visible,
                                   labelsize=self._AXIS_LABELSIZE)
        self._legend_visible = True
        self._apply_view_options()
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def update_data(self, spectrum: NMRSpectrum) -> None:
        """Refresh the currently-displayed 1D spectrum's y-data in
        place, preserving the current view (zoom/pan) -- unlike
        :func:`show_spectrum`, which resets the view for a newly
        selected spectrum. For a live preview (e.g. dragging a phase
        control) where the spectrum's identity hasn't changed, only
        its values have.

        Re-derives the same rendering-only downsampling
        :func:`opennmr.plotting.plot_1d` applied to the x-data already
        on screen (see :func:`opennmr.plotting.style.downsample_indices`'s
        own docstring for why this reproduces the identical indices
        without needing to remember them) -- otherwise a high-point-
        count spectrum's full-length new y-data wouldn't match the
        already-downsampled x-data's length at all, let alone quickly.

        Falls back to a full :meth:`show_spectrum` rebuild (preserving
        ppm zoom across it -- the range doesn't change, only the point
        density) when ``spectrum``'s own length has actually changed
        since the last draw -- possible now that phase-dragging can
        recompute from the Processing tab's live window/zero-fill
        choice too, not just re-phase a fixed-length snapshot; the
        in-place path above assumes an unchanged length and would
        otherwise plot new y-values against stale, wrong-length x
        positions.
        """
        if spectrum.ndim != 1 or not self.ax.lines:
            self.show_spectrum(spectrum)
            return
        y = spectrum.normalized()
        idx = plotting.downsample_indices(len(y))
        new_ydata = y if idx is None else y[idx]
        if len(new_ydata) != len(self.ax.lines[0].get_ydata()):
            xlim = self.ax.get_xlim()
            self.show_spectrum(spectrum)
            self.ax.set_xlim(xlim)
            return
        self.ax.lines[0].set_ydata(new_ydata)
        self.canvas.draw_idle()

    def autoscale_y(self) -> None:
        """Refit the y (intensity) axis to the data actually plotted
        right now on the main line -- for after a *discrete* phase
        action (Autophase, Baseline Correct, Revert Autophase), not a
        continuous drag: a big one-shot phase/baseline change can
        leave the view's old y-limits (from before the change)
        showing mostly empty space or clipping the now-differently-
        scaled baseline, and there's no "current view" worth
        preserving through a single click the way there is through a
        drag gesture (see :meth:`update_data`'s own docstring, and the
        test that pins the view during a drag -- this is deliberately
        the opposite case). X stays exactly as it is; only y moves.

        Computed directly from the main data line's own y-values (not
        ``Axes.relim``/``autoscale_view``) since this canvas's own
        scroll-zoom/pan handlers call ``set_ylim`` directly, which
        matplotlib treats as the user having taken over the y-axis --
        ``autoscale_view`` then silently no-ops on it forever after,
        which isn't what a discrete "refit to what's on screen now"
        action should do."""
        if not self.ax.lines:
            return
        y = self.ax.lines[0].get_ydata()
        if len(y) == 0:
            return
        ymin, ymax = float(np.min(y)), float(np.max(y))
        pad = (ymax - ymin) * 0.05 or 1.0
        self.ax.set_ylim(ymin - pad, ymax + pad)
        self.canvas.draw_idle()

    def _lines_for_export(self) -> list:
        """Every currently-plotted Line2D across whichever axes this
        canvas is actually showing -- ``self.ax`` alone for every
        ordinary view, or all of :attr:`_compare_axes` for a "Stacked"
        Compare view (several real axes, not one shared by every
        line)."""
        axes = self._compare_axes if self._compare_axes is not None else [self.ax]
        return [line for ax in axes for line in ax.lines]

    def has_downsampled_data(self) -> bool:
        """Whether anything currently plotted was (or would be, if
        :attr:`_full_resolution` is already on) rendered below its own
        true resolution -- gates the right-click "Full Resolution"
        toggle, so it's not offered when there's nothing it would
        actually change. A 2D contour is always eligible (unlike the
        1D-line check below, this doesn't distinguish "actually
        reduced" from "configured to allow it" -- matching the same
        simplification the 1D stashing already makes; ``downsample_2d``
        itself is a safe no-op when the data's already smaller than
        the point cap)."""
        if self._contour_spectrum is not None:
            return True
        return any(hasattr(line, "_opennmr_full_res") for line in self._lines_for_export())

    def set_full_resolution(self, full: bool) -> None:
        """Toggle between the default rendering-only downsampling
        (see :func:`opennmr.plotting.plot_1d`) and true full
        resolution for whatever's plotted right now -- an explicit
        escape hatch for when the downsampling itself needs to be seen
        through, not just its usual visually-identical result (e.g.
        checking a specific point's exact value while zoomed in
        close). Persists across a rebuild the same way the legend/
        aspect-lock choices do (see :meth:`show_spectrum`'s own
        docstring): every subsequent (re)plot on this canvas keeps
        using whichever this was last set to, until changed again.

        Swaps each line's already-plotted data in place (never
        re-plots from a spectrum object, which this method doesn't
        even have access to) -- full resolution from the line's own
        stashed ``_opennmr_full_res``, or freshly re-derived
        downsampled data from that same full-resolution array
        (:func:`opennmr.plotting.style.downsample_indices` is
        deterministic in array length alone, so this reproduces
        exactly what the original plot call would have shown).

        A 2D contour has no equivalent in-place swap (a
        ``QuadContourSet``, unlike a ``Line2D``, has no reusable
        "already-plotted data" to reach back into) -- a full
        :meth:`show_spectrum` rebuild instead, which already reads
        :attr:`_full_resolution` (just set above) via
        :meth:`_contour_fraction_kwargs`."""
        self._full_resolution = full
        for line in self._lines_for_export():
            full_data = getattr(line, "_opennmr_full_res", None)
            if full_data is None:
                continue
            if full:
                line.set_data(*full_data)
            else:
                x_full, y_full = full_data
                idx = plotting.downsample_indices(len(y_full))
                line.set_data((x_full, y_full) if idx is None else (x_full[idx], y_full[idx]))
        if self._contour_spectrum is not None:
            self.show_spectrum(self._contour_spectrum)
        self.canvas.draw_idle()

    def export_full_resolution(self, save_fn) -> None:
        """Call ``save_fn()`` (e.g. ``self.toolbar.save_figure``) with
        every downsampled line's data temporarily swapped back to its
        full resolution first -- a saved/exported figure should never
        be silently missing points the live view dropped purely for
        faster interactive rendering, regardless of whether
        :attr:`_full_resolution` happens to be on already. Also
        applies the Output tab's own export DPI/figure-size (see
        :meth:`set_export_settings`), if set, for the same reason a
        saved figure shouldn't just inherit whatever size Qt's layout
        happened to give the live embedded widget -- **not** applied
        to the live figure at any other time (see that method's own
        docstring for why: matplotlib's own figure-size handling and
        Qt's automatic canvas-widget resizing fight each other if the
        embedded figure's size is changed out from under Qt). Restores
        the on-screen (possibly still-downsampled, still-interactive-
        sized) view again afterward either way, including if
        ``save_fn`` itself raises or the user cancels the save dialog
        inside it.

        A 2D contour has no in-place swap the way a ``Line2D`` does
        (see :meth:`set_full_resolution`'s own docstring) -- a full
        rebuild instead, with the current pan/zoom saved and restored
        around it so an export doesn't silently reset the view."""
        swapped = []
        for line in self._lines_for_export():
            full_data = getattr(line, "_opennmr_full_res", None)
            if full_data is None or self._full_resolution:
                continue  # already full resolution -- nothing to swap
            swapped.append((line, line.get_xdata(), line.get_ydata()))
            line.set_data(*full_data)
        if swapped:
            self.canvas.draw()

        contour_swapped = False
        contour_xlim = contour_ylim = None
        if self._contour_spectrum is not None and not self._full_resolution:
            contour_xlim, contour_ylim = self.ax.get_xlim(), self.ax.get_ylim()
            self._full_resolution = True
            self.show_spectrum(self._contour_spectrum)
            self.ax.set_xlim(contour_xlim)
            self.ax.set_ylim(contour_ylim)
            contour_swapped = True

        original_dpi = self.figure.get_dpi()
        original_size = self.figure.get_size_inches()
        if self._export_dpi is not None:
            self.figure.set_dpi(self._export_dpi)
        default_width, default_height = self.default_export_size()
        self.figure.set_size_inches(
            self._export_fig_width if self._export_fig_width is not None else default_width,
            self._export_fig_height if self._export_fig_height is not None else default_height,
        )

        try:
            save_fn()
        finally:
            if swapped:
                for line, x, y in swapped:
                    line.set_data(x, y)
            if contour_swapped:
                self._full_resolution = False
                self.show_spectrum(self._contour_spectrum)
                self.ax.set_xlim(contour_xlim)
                self.ax.set_ylim(contour_ylim)
            self.figure.set_dpi(original_dpi)
            self.figure.set_size_inches(original_size)
            self.canvas.draw_idle()


def _spectrum_label(spectrum: NMRSpectrum) -> str:
    meta = spectrum.metadata
    name = os.path.basename(meta.source_path.rstrip("/")) if meta.source_path else None
    if name and meta.nucleus:
        return f"{name} ({meta.nucleus})"
    return name or meta.nucleus or "spectrum"
