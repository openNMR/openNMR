"""
Matplotlib canvas embedded in the Qt GUI.

Plotting into this canvas always goes through the ``ax``-first
functions in :mod:`nmrsuite.plotting` (never
:func:`nmrsuite.plotting.style.setup_plotting`, which mutates global
matplotlib state) -- see that module's docstring for why: restyling
one canvas must never restyle every other open one.
"""

from __future__ import annotations

import os

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QInputDialog, QMenu, QVBoxLayout, QWidget

from ...core.spectrum import NMRSpectrum
from ... import plotting


class _Toolbar(NavigationToolbar2QT):
    """The stock toolbar minus the two buttons that pop up a separate
    floating dialog (``Configure subplots``, ``Edit axis, curve and
    image parameters``) -- this app's own docked controls (the
    right-click menu's legend/aspect-ratio options, the Processing
    tab) cover the same ground without a popup window."""

    toolitems = [item for item in NavigationToolbar2QT.toolitems if item[0] not in ("Subplots", "Customize")]


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
    _CONTOUR_FACTOR_DEFAULT = 1e-3

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

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        self.figure = Figure()
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = _Toolbar(self.canvas, self)

        layout = QVBoxLayout(self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        self._pan_start = None  # (x_pixel, y_pixel, xlim, ylim) while dragging
        self._legend_visible = False
        self._aspect_ratio: float | None = None  # None = unlocked
        self._pick_mode = False
        # Set only while a 2D/pseudo-2D contour is the current view --
        # scrolling then adjusts the contour threshold instead of
        # zooming (see _on_scroll). None whenever any other view
        # (1D line, FID, compare, fit) is shown.
        self._contour_spectrum: NMRSpectrum | None = None
        self._contour_factor: float | None = None
        # Set only for a plain 1D spectrum/slice view -- lets the
        # right-click menu offer ppm/Hz/kHz switching by re-using the
        # spectrum's own stored axis arrays, no recomputation needed.
        # None for any other view (contour, FID, compare, fit), where
        # unit-switching isn't offered.
        self._current_1d_spectrum: NMRSpectrum | None = None
        self._axis_unit = "ppm"
        self._ppm_label = ""

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

    def _on_scroll(self, event) -> None:
        if event.inaxes is None or event.xdata is None:
            return
        if self._contour_factor is not None and event.inaxes is self.ax:
            self._adjust_contour_levels(event)
            return

        ax = event.inaxes
        factor = 1 / self.ZOOM_FACTOR if event.button == "up" else self.ZOOM_FACTOR

        xmin, xmax = ax.get_xlim()
        ax.set_xlim(
            event.xdata - (event.xdata - xmin) * factor,
            event.xdata + (xmax - event.xdata) * factor,
        )
        if event.ydata is not None and ax.get_yticks().size:
            ymin, ymax = ax.get_ylim()
            ax.set_ylim(
                event.ydata - (event.ydata - ymin) * factor,
                event.ydata + (ymax - event.ydata) * factor,
            )
        self.canvas.draw_idle()

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
        xlim, ylim = self.ax.get_xlim(), self.ax.get_ylim()
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        plotting.plot_2d_contour(self._contour_spectrum, ax=self.ax, fig=self.figure, factor=self._contour_factor)
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self._apply_view_options()
        self.canvas.draw_idle()

    def _on_press(self, event) -> None:
        if event.inaxes is None or event.button != 1 or self.toolbar.mode != "":
            return
        if self._pick_mode:
            if event.xdata is not None:
                self.peakPicked.emit(float(event.xdata))
            return
        self._pan_start = (event.x, event.y, event.inaxes.get_xlim(), event.inaxes.get_ylim(), event.inaxes)

    def _on_motion(self, event) -> None:
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
        self._pan_start = None

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

    def set_legend_visible(self, visible: bool) -> None:
        self._legend_visible = visible
        self._apply_view_options()
        self.canvas.draw_idle()

    def set_aspect_ratio(self, ratio: float | None) -> None:
        self._aspect_ratio = ratio
        self._apply_view_options()
        self.canvas.draw_idle()

    def set_axis_unit(self, unit: str) -> None:
        """Switch the current 1D view's x-axis between ``"ppm"``,
        ``"hz"``, and ``"khz"``, re-using the spectrum's own stored
        axis arrays (no recomputation) -- only available while a plain
        1D spectrum/slice (not a contour, FID, compare, or fit view)
        is the current view (see :attr:`_current_1d_spectrum`)."""
        if self._current_1d_spectrum is None or unit not in ("ppm", "hz", "khz") or not self.ax.lines:
            return
        axis = self._current_1d_spectrum.axes[0]
        if unit == "ppm":
            x = axis.values("ppm")
            label = self._ppm_label
        elif unit == "hz":
            x = axis.values("hz")
            label = "Frequency / Hz"
        else:
            x = axis.values("hz") / 1000.0
            label = "Frequency / kHz"

        for line in self.ax.lines:
            line.set_xdata(x)
        self.ax.set_xlim(float(np.min(x)), float(np.max(x)))
        self.ax.invert_xaxis()
        self.ax.set_xlabel(label)
        self._axis_unit = unit
        self.canvas.draw_idle()

    def _apply_view_options(self) -> None:
        if self._legend_visible and self.ax.get_legend_handles_labels()[0]:
            self.ax.legend()
        else:
            existing = self.ax.get_legend()
            if existing is not None:
                existing.remove()
        self.ax.set_box_aspect(self._aspect_ratio)

    def clear(self) -> None:
        self._contour_spectrum = None
        self._contour_factor = None
        self._current_1d_spectrum = None
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        self.canvas.draw_idle()

    def show_spectrum(self, spectrum: NMRSpectrum) -> None:
        """Plot ``spectrum`` into this canvas, replacing whatever was
        there -- 1D via :func:`nmrsuite.plotting.plot_1d`, 2D via
        :func:`nmrsuite.plotting.plot_2d_contour`. Preserves this
        canvas's legend-visible/aspect-lock choices across the
        rebuild."""
        self.figure.clear()

        if spectrum.ndim == 1:
            self._contour_spectrum = None
            self._contour_factor = None
            self._current_1d_spectrum = spectrum
            self._axis_unit = "ppm"  # a fresh spectrum always starts in ppm
            self.ax = self.figure.add_subplot(111)
            xmin, xmax, labeltext = plotting.plot_1d(self.ax, spectrum)
            self._ppm_label = labeltext
            self.ax.lines[0].set_label(_spectrum_label(spectrum))
            self.ax.set_xlim(xmin, xmax)
            self.ax.invert_xaxis()
            self.ax.set_xlabel(labeltext)
            self.ax.set_yticks([])
            plotting.apply_spine_style(self.ax, top=False, right=False, bottom=True, left=False)
            plotting.apply_tick_style(self.ax, axis="x")
        elif spectrum.ndim == 2:
            self._current_1d_spectrum = None
            self._contour_spectrum = spectrum
            self._contour_factor = self._CONTOUR_FACTOR_DEFAULT
            self.ax = self.figure.add_subplot(111)
            plotting.plot_2d_contour(spectrum, ax=self.ax, fig=self.figure, factor=self._contour_factor)
        else:
            raise ValueError(f"Cannot display a spectrum with ndim={spectrum.ndim}")

        self._apply_view_options()
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def show_fid(self, fid, dt: float, title: str = "", window_envelope=None, tdeff_index: int | None = None) -> None:
        """Plot a raw/processed time-domain FID (real + imaginary vs.
        time) -- an alternate view (the FID canvas tab) for judging
        FID-domain processing directly, instead of only its effect
        after FFT. The frequency-domain spectrum (:func:`show_spectrum`)
        stays the default (Spectrum tab) view.

        ``window_envelope`` : the currently-selected apodization
        window's own shape (real-valued, same length as ``fid``, from
        applying it to an all-ones array) -- overlaid scaled to the
        FID's own peak magnitude, so a window/line-broadening choice
        can be judged directly against the data it will taper,
        without needing to click "Reprocess" first.

        ``tdeff_index`` : if given (and less than ``len(fid)``), draws
        a dashed vertical marker at that point in time -- where a
        TDeff truncation would cut the FID, previewed live as the
        control changes, before "Reprocess" is clicked."""
        self._contour_spectrum = None
        self._contour_factor = None
        self._current_1d_spectrum = None
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        t = np.arange(len(fid)) * dt
        self.ax.plot(t, np.real(fid), label="Real")
        self.ax.plot(t, np.imag(fid), label="Imag", alpha=0.7)
        if window_envelope is not None:
            peak = np.max(np.abs(fid))
            self.ax.plot(t, np.asarray(window_envelope) * peak, "--", color="gray", label="Window", zorder=0)
            self._legend_visible = True
        if tdeff_index is not None and 0 < tdeff_index < len(fid):
            self.ax.axvline(tdeff_index * dt, linestyle="--", color="red", label="TDeff cutoff")
            self._legend_visible = True
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Amplitude")
        if title:
            self.ax.set_title(title)
        self._apply_view_options()
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def show_compare(self, spectra: list[NMRSpectrum], mode: str = "stack",
                      y_offset: float = 0.2, normalize: bool = False) -> None:
        """Plot several 1D spectra together for comparison, via
        :func:`nmrsuite.plotting.plot_1d_stack` -- ``mode="stack"``
        (vertically offset by ``y_offset``, as a fraction of each
        spectrum's own height) or ``"overlay"`` (all at y=0): both are
        really the same call, since a large enough negative
        ``y_offset`` collapses the accumulated stack offset to zero
        without needing a second plotting function."""
        self._contour_spectrum = None
        self._contour_factor = None
        self._current_1d_spectrum = None
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        offset = y_offset if mode == "stack" else -1.0
        plotting.plot_1d_stack(spectra, ax=self.ax, fig=self.figure, y_offset=offset, normalize=normalize)
        self.ax.set_yticks([])
        plotting.apply_spine_style(self.ax, top=False, right=False, bottom=True, left=False)
        plotting.apply_tick_style(self.ax, axis="x")
        self._apply_view_options()
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def show_slice_fit(self, x, y_data, y_fit, title: str = "") -> None:
        """Plot one pseudo-2D slice's spectrum alongside its fitted
        (pseudo-Voigt) curve -- the intermediate per-slice spectral
        fit that feeds a relaxation/diffusion parameter fit
        (:func:`nmrsuite.processing.fitting.NMR2D_fit_slices`'s own
        per-slice result), so it can be inspected directly instead of
        only trusting the final T1/T2/D number."""
        self._contour_spectrum = None
        self._contour_factor = None
        self._current_1d_spectrum = None
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        self.ax.plot(x, y_data, color="black", linewidth=1.0, label="Data")
        self.ax.plot(x, y_fit, color="red", linestyle="--", linewidth=1.2, label="Fit")
        self.ax.set_yticks([])
        if title:
            self.ax.set_title(title)
        self._legend_visible = True
        self._apply_view_options()
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def show_relaxation_fit(self, results: list[dict]) -> None:
        """Plot each peak's relaxation series (points) and fitted curve
        (line), from a list of ``relaxation.fit_relaxation``-shaped
        dicts (as returned by
        :func:`nmrsuite.processing.relaxation.NMR2D_fit_relaxation`).

        The fit curve is drawn through ``t``/``y_fit`` (already sorted
        by delay) rather than a finely-resampled model curve -- keeps
        this module decoupled from ``processing.relaxation``'s model
        registry, and is visually adequate for a typical (5+ point)
        relaxation series."""
        self._contour_spectrum = None
        self._contour_factor = None
        self._current_1d_spectrum = None
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        for result in results:
            t, y, y_fit = result["t"], result["y"], result["y_fit"]
            label = result.get("peak_label", result.get("model", "fit"))
            line, = self.ax.plot(t, y, "o", label=label)
            self.ax.plot(t, y_fit, "-", color=line.get_color(), alpha=0.8)
        self.ax.set_xlabel("Delay (s)")
        self.ax.set_ylabel("Intensity")
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
        """
        if spectrum.ndim != 1 or not self.ax.lines:
            self.show_spectrum(spectrum)
            return
        self.ax.lines[0].set_ydata(spectrum.normalized())
        self.canvas.draw_idle()


def _spectrum_label(spectrum: NMRSpectrum) -> str:
    meta = spectrum.metadata
    name = os.path.basename(meta.source_path.rstrip("/")) if meta.source_path else None
    if name and meta.nucleus:
        return f"{name} ({meta.nucleus})"
    return name or meta.nucleus or "spectrum"
