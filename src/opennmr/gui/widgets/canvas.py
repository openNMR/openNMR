"""
Matplotlib canvas embedded in the Qt GUI.

Plotting into this canvas always goes through the ``ax``-first
functions in :mod:`opennmr.plotting` (never
:func:`opennmr.plotting.style.setup_plotting`, which mutates global
matplotlib state) -- see that module's docstring for why: restyling
one canvas must never restyle every other open one.
"""

from __future__ import annotations

import copy
import os
import re
import warnings

import numpy as np
import dataclasses

import matplotlib.collections
import matplotlib.colors
import matplotlib.container
import matplotlib.lines
import matplotlib.patches
import matplotlib.text
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtCore import QEvent, Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox, QColorDialog, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout,
    QHBoxLayout, QInputDialog, QLabel, QLineEdit, QMenu, QPushButton,
    QStyle, QToolButton, QVBoxLayout, QWidget,
)

from .typed_spinbox import TypedSpinBox
from ...core.registry import find_gamma, NUC_label
from ...notebook import fit_display
from ...core.spectrum import NMRAxis, NMRSpectrum
from ... import plotting
from ...plotting import inset as inset_module
from ...processing import slicing as processing_slicing


class _ContourCrosshair:
    """A blitted crosshair spanning a 2D contour and its own F1/F2
    projection strips: one vertical line across the contour + the F2
    (top) strip, one horizontal line across the contour + the F1
    (left) strip. Explicit user request: "when the input isn't
    panning or zooming, the cursor should be a cross with lines
    covering the entire contour space AND the projections".

    Blitted rather than a plain per-move redraw, which repaints the
    whole contour -- measured at 295 ms for a real HSQC and 1.34 s for
    the EXSY, nowhere near interactive.

    **Not two independent :class:`~matplotlib.widgets.MultiCursor`
    instances** (the first two attempts at this, both reverted -- see
    TODO.md). Two overlapping cursors bound to the same canvas each
    keep their *own* cached background, and matplotlib's own
    ``MultiCursor.onmove`` unconditionally does
    ``restore_region(own_background)`` before drawing its own line --
    so whichever cursor's ``motion_notify_event`` callback happens to
    run second (always the same one, by connection order) wipes the
    first one's just-drawn line before that frame is ever shown. One
    shared background here, restored once per move, with both lines
    drawn onto it before the single blit.

    **Torn down synchronously by the owning canvas, at the same
    moment its Axes are destroyed** (see ``MplCanvas._clear_figure``)
    -- never lazily, from a later mouse-move. The previous two
    attempts rebuilt/tore down from ``_on_motion``, which only runs on
    real mouse movement; a contour that rebuilds several times with no
    intervening move (e.g. repeated contour-level-adjust scrolls) left
    a stale instance connected for that whole stretch. Its Axes are
    gone from the figure, but matplotlib doesn't clear their
    ``.figure`` reference just because ``figure.clear()`` dropped
    them from ``figure.axes`` -- so a stale instance's ``onmove`` kept
    computing real ``ax.draw_artist`` calls against the *live*
    renderer, through the *old* Axes' now-meaningless transform. That
    is what "filling the contour and not going away" actually was:
    debris from an orphaned cursor, painted with a stale coordinate
    mapping, that a correctly-scoped redraw never had reason to erase.

    **A second, distinct source of the same symptom, found live after
    the fix above**: a freshly created instance starts with no cached
    background at all -- nothing has forced a real ``draw()`` (the
    only thing that fires ``draw_event`` -> ``_on_draw``) since
    creation, e.g. right after extracting or clearing a contour slice,
    which rebuilds the figure and only lazily recreates the crosshair
    on the next mouse move (see ``MplCanvas._ensure_crosshair``).
    ``onmove`` with no background can't ``restore_region`` first, so
    every move drew straight onto whatever the Agg buffer already held
    -- accumulating lines that only ever got wiped by the *next* real
    draw, i.e. whenever the cursor left the axes and ``_hide`` called
    ``draw_idle``. Fixed by forcing a synchronous ``draw()`` inside
    ``onmove`` itself the first time it finds no background, before
    touching the lines' position or visibility -- see there.
    """

    def __init__(self, canvas, ax, f1_ax, f2_ax, toolbar, stack_axes=None, panel_axes=None,
                 free_axes=None, **lineprops) -> None:
        """``stack_axes``: the rows of a stacked Compare (one shared x
        axis). The vertical line then runs through every row, whichever
        one the cursor is over, and the horizontal one marks the row under
        the cursor -- explicit user request (2026-10-08): "when hovering
        over one spectrum, have the crosshair continue along y; at
        present, it gets cut off and you can only see the crosshair on
        one of the n spectra being compared"."""
        self._canvas = canvas
        self._toolbar = toolbar
        stack = [a for a in (stack_axes or []) if a is not None]
        self._stacked = len(stack) > 1
        # Side-by-side 2D panels (Compare): the crosshair goes in every
        # panel, at the same position, while their x and y spans match,
        # and in the panel under the cursor alone when they do not.
        panels = [a for a in (panel_axes or []) if a is not None]
        self._panels = tuple(panels) if len(panels) > 1 else ()
        # A figure of several plots drawn elsewhere (Simulate: a SIMPSON
        # shape's amplitude/phase/offset rows, a Bloch simulation's pulse
        # and profile): the crosshair follows the cursor into whichever
        # plot it is over, and its vertical line runs through every plot
        # sharing that x axis -- as a stacked Compare does (explicit user
        # request, 2026-11-12: it was "either just not present or only
        # appears in one of the plots", being bound to the first alone).
        free = [a for a in (free_axes or []) if a is not None]
        self._free = tuple(free) if len(free) > 1 else ()
        self.axes = tuple(dict.fromkeys(
            a for a in (ax, f1_ax, f2_ax, *stack, *self._panels, *self._free) if a is not None))
        self._background = None
        self._vline_axes = list(dict.fromkeys(
            a for a in (ax, f2_ax, *stack, *self._panels, *self._free) if a is not None))
        self._vlines = [self._guide(a, "v", lineprops) for a in self._vline_axes]
        self._hline_axes = list(dict.fromkeys(
            a for a in (ax, f1_ax, *stack, *self._panels, *self._free) if a is not None))
        self._hlines = [self._guide(a, "h", lineprops) for a in self._hline_axes]
        self._cid_motion = canvas.mpl_connect("motion_notify_event", self._onmove)
        self._cid_draw = canvas.mpl_connect("draw_event", self._on_draw)

    @staticmethod
    def _guide(ax, orientation: str, lineprops):
        """One crosshair line, spanning ``ax`` -- **not** ``axvline``/
        ``axhline``. Those are data: created at 0 they put 0 into the
        Axes' data limits, so the first redraw after the crosshair was
        switched on autoscaled the plot out to include x = 0 and y = 0,
        and Home (which re-autoscales) could not undo it. Reported on
        the DNP field profile (fields near 9396 mT): "as soon as I cross
        into the axes it zooms out so I see (0,0) ... and I cannot reset
        the view". Any spectrum whose range excludes 0 ppm did the same.
        Drawn in axes-fraction coordinates along its length and never
        added to the Axes' children, it is not part of ``ax.lines``, the
        data limits or a relim; it is only ever drawn by blitting."""
        from matplotlib.lines import Line2D
        if orientation == "v":
            line = Line2D([0, 0], [0, 1], transform=ax.get_xaxis_transform(), **lineprops)
        else:
            line = Line2D([0, 1], [0, 0], transform=ax.get_yaxis_transform(), **lineprops)
        line.set_visible(False)
        line.set_animated(True)
        # Attached to the Axes (for its transforms and draw_artist) but
        # not added as a child: a child Line2D is in ax.lines, and relim
        # counts its data coordinate -- which along the line's length is
        # the Axes fraction 0..1 here, and across it the cursor's value.
        line.axes = ax
        line.set_figure(ax.figure)
        return line

    def disconnect(self) -> None:
        """Stop reacting to this canvas entirely. Must run before the
        Axes these lines belong to are destroyed -- see the class
        docstring's "torn down synchronously" note."""
        self._canvas.mpl_disconnect(self._cid_motion)
        self._canvas.mpl_disconnect(self._cid_draw)

    def remove(self) -> None:
        """:meth:`disconnect`, *and* take this crosshair's own Line2D
        artists back out of their Axes.

        For the usual teardown (``MplCanvas._clear_figure``) the Axes
        are destroyed in the same breath, so removing the lines first
        would be busywork -- ``disconnect`` alone is right there. This
        is for the other case: the Axes **survive** and something is
        about to walk ``ax.lines`` and rewrite every line it finds.
        :meth:`MplCanvas.set_axis_unit` does exactly that, and a
        crosshair left in place got its two-point ``axvline``/
        ``axhline`` overwritten with the spectrum's own full-length x
        array -- after which the next mouse move set a 2-element
        ``ydata`` against a 512-element ``xdata`` and matplotlib threw
        ``ValueError: shape mismatch`` out of ``Line2D.recache`` on
        every single move. That is what "the crosshair disappears when
        changing between axes" actually was.

        Tolerant of a line already detached (a rebuild that beat us to
        it): matplotlib raises if the artist has no Axes to be removed
        from, and there is nothing to do in that case anyway."""
        self.disconnect()
        for line in (*self._vlines, *self._hlines):
            try:
                line.remove()
            except (ValueError, NotImplementedError):
                pass
        self._vlines = []
        self._hlines = []
        self._vline_axes = []
        self._hline_axes = []

    def _on_draw(self, event) -> None:
        self._background = self._canvas.copy_from_bbox(self._canvas.figure.bbox)

    def panels_match(self) -> bool:
        """Whether every side-by-side panel shows the same x and y span
        right now (so one position means the same thing in each)."""
        if not self._panels:
            return True

        def same(a, b) -> bool:
            tol = 1e-9 * max(abs(a[1] - a[0]), abs(b[1] - b[0]), 1e-300)
            return abs(a[0] - b[0]) <= tol + 1e-6 * abs(a[1] - a[0]) \
                and abs(a[1] - b[1]) <= tol + 1e-6 * abs(a[1] - a[0])

        first = self._panels[0]
        fx, fy = first.get_xlim(), first.get_ylim()
        return all(same(p.get_xlim(), fx) and same(p.get_ylim(), fy) for p in self._panels[1:])

    @staticmethod
    def x_linked(a, b) -> bool:
        """Whether one x position means the same thing in Axes ``a`` and
        ``b``: they share their x axis (``sharex``, a twin), or they are
        labelled alike and show the same x range."""
        if a is None or b is None:
            return False
        if a is b or a.get_shared_x_axes().joined(a, b):
            return True
        label_a, label_b = a.xaxis.label.get_text(), b.xaxis.label.get_text()
        if not label_a or label_a != label_b:
            return False
        (a0, a1), (b0, b1) = a.get_xlim(), b.get_xlim()
        tol = 1e-6 * max(abs(a1 - a0), 1e-300)
        return abs(a0 - b0) <= tol and abs(a1 - b1) <= tol

    def _hide(self) -> None:
        """Take the crosshair off screen when the cursor leaves the
        plot -- **by blitting the cached background back, not by
        redrawing the figure.**

        ``draw_idle`` here was a full contour repaint on every single
        exit: measured at 295 ms for a real HSQC and 1.34 s for the
        EXSY, which is exactly the cost this class was blitted to
        avoid in the first place -- it was just being paid on the way
        *out* instead of on the way in. Reported as "when I move the
        mouse from the contour to the axis there is a very quick
        loading wheel", and it also ate the scroll cursor: the app is
        busy for that whole repaint, so the platform shows its own
        wait cursor over the canvas and the strip arrow only becomes
        visible once the redraw finishes -- by which time the pointer
        has usually moved on. Hence "I saw one very briefly once".

        The lines are ``animated=True``, so a normal draw never
        includes them and the cached background is already the clean
        frame -- restoring it *is* erasing them. ``set_visible(False)``
        still matters for whatever full redraw comes next.

        Falls back to ``draw_idle`` when there is no cached background
        yet (nothing has forced a real draw since this instance was
        created -- see the class docstring's second bug note), which is
        the one case where blitting would leave whatever the buffer
        happens to hold."""
        lines = (*self._vlines, *self._hlines)
        if not any(line.get_visible() for line in lines):
            return
        for line in lines:
            line.set_visible(False)
        if self._background is None:
            self._canvas.draw_idle()
            return
        self._canvas.restore_region(self._background)
        self._canvas.blit()

    def _onmove(self, event) -> None:
        if getattr(self._toolbar, "mode", "") or event.inaxes not in self.axes:
            self._hide()
            return
        if event.xdata is None or event.ydata is None:
            return
        if self._background is None:
            # Nothing has forced a real draw (the only thing that
            # fires draw_event -> _on_draw) since this instance was
            # created, so there is nothing to restore_region from --
            # see the class docstring's second bug note. Force one now
            # while every line is still invisible, so the captured
            # background is the correct "clean" frame to restore
            # before every future move; drawing this move's own
            # position onto an unset background instead is what
            # actually caused "filling the contour and not going away".
            self._canvas.draw()
        only_here = bool(self._panels) and not self.panels_match()
        here = event.inaxes
        for ax, line in zip(self._vline_axes, self._vlines):
            line.set_xdata((event.xdata, event.xdata))
            if self._free and ax in self._free:
                line.set_visible(ax is here or self.x_linked(ax, here))
            else:
                line.set_visible(not only_here or ax is here)
        for ax, line in zip(self._hline_axes, self._hlines):
            # In a stack each row has its own intensity scale: the
            # horizontal line belongs only to the row under the cursor.
            line.set_ydata((event.ydata, event.ydata))
            if self._free and ax in self._free:
                line.set_visible(ax is here)
            else:
                line.set_visible((not self._stacked or ax is here)
                                 and (not only_here or ax is here))
        self._canvas.restore_region(self._background)
        for ax, line in zip(self._vline_axes, self._vlines):
            ax.draw_artist(line)
        for ax, line in zip(self._hline_axes, self._hlines):
            if line.get_visible():
                ax.draw_artist(line)
        self._canvas.blit()


@dataclasses.dataclass
class SeriesStyle:
    """One addressable series on a plot: what it is called and what
    colour it is drawn in.

    **The single place a multi-series plot's per-series appearance
    lives.** Before this there were four: Compare derived colours from
    a colour map inside its draw loop, the DOSY overlay and the 2D
    overlay each did the same with a different palette, and only
    extracted slices stored anything. Four mechanisms meant four
    places to wire every new control, and -- because three of them
    re-derived the colour at draw time -- three places for it to reset
    on a redraw, which is precisely what kept happening.

    ``key`` addresses the series within its own view and is stable
    across redraws; ``label`` and ``colour`` are what the Plot Style
    dialog edits. ``can_rename`` is False where the name is not the
    plot's to change (a Compare trace is named by its spectrum, which
    has its own Rename action reaching every other view too).
    """

    key: str
    label: str
    colour: str
    can_rename: bool = True
    #: Whether ``colour`` was set deliberately rather than taken from
    #: the series palette. The distinction is what lets a redraw keep
    #: a colour *and* a palette change replace one: a derived colour is
    #: re-derived from whatever palette is now in force, an overridden
    #: one is left alone. Without it the reuse that stops colours
    #: resetting on a redraw also stopped the colour map from doing
    #: anything -- reported directly: "when I change the colour map
    #: from a plot, it should update the existing colours from that new
    #: series rather than me having to redraw the plot in some way".
    colour_overridden: bool = False
    #: Whether ``label`` was typed by the user rather than computed by
    #: the view that drew the series. **Exactly the distinction
    #: :attr:`colour_overridden` draws, and it was missing here.**
    #: Reuse keyed on ``can_rename`` alone meant a renameable series
    #: kept whatever label its key held last time -- so re-running
    #: SCORE redrew component 1 with the *previous* resolution's
    #: annotation (its D, its share, its ratio), stale but plausible,
    #: while the sidebar showed the new one. Reported directly: "the
    #: label on the first SCORE fit component plot doesn't match the
    #: output in the side bar... consistently showing 1.60e-10 m^2/s".
    label_overridden: bool = False


class _Toolbar(NavigationToolbar2QT):
    """The stock toolbar minus the two buttons that pop up a separate
    floating dialog (``Configure subplots``, ``Edit axis, curve and
    image parameters``) -- this app's own docked controls (the
    right-click menu's legend/aspect-ratio options, the Processing
    tab) cover the same ground without a popup window."""

    toolitems = [item for item in NavigationToolbar2QT.toolitems if item[0] not in ("Subplots", "Customize")]

    def home(self, *args) -> None:
        """The house icon resets through :meth:`MplCanvas.reset_view`,
        like View > Home. matplotlib's own home only replays its
        navigation stack, which the canvas's scroll-zoom and drag-pan
        never write to -- so after either, the house icon did nothing
        (reported on the FID and Processing views, true of every plot)."""
        owner = self.parent()
        if isinstance(owner, MplCanvas):
            owner.reset_view()
        else:
            super().home(*args)

    def pan(self, *args) -> None:
        super().pan(*args)
        self._notify_mode_changed()

    def zoom(self, *args) -> None:
        super().zoom(*args)
        self._notify_mode_changed()

    def _notify_mode_changed(self) -> None:
        owner = self.parent()
        if isinstance(owner, MplCanvas):
            owner._on_navigation_mode_changed()

    def _update_buttons_checked(self) -> None:
        """matplotlib checks Pan/Zoom only while its own tool is on. With
        neither on, a plain drag still pans or zooms (the canvas's
        default drag, :meth:`MplCanvas.set_drag_mode`), so that button is
        shown checked too -- "it isn't obvious it's pan" (explicit user
        report, 2026-10-14). A click mode (Pick, Select Region, Measure)
        owns the drag instead, and then neither is checked."""
        mode = getattr(self.mode, "name", str(self.mode or "")).upper()
        owner = self.parent()
        default = ""
        # getattr: matplotlib may call this while the canvas is still
        # being built, before its mode attributes exist.
        if not self.mode and isinstance(owner, MplCanvas) and hasattr(owner, "_drag_mode") \
                and not owner._click_mode_active():
            default = owner.drag_mode()
        if "pan" in self._actions:
            self._actions["pan"].setChecked("PAN" in mode or default == "pan")
        if "zoom" in self._actions:
            self._actions["zoom"].setChecked("ZOOM" in mode or default == "zoom")

    # A drag's motion callback reads ``_zoom_info`` / ``_pan_info`` on
    # every mouse move. matplotlib's own press overwrites an unfinished
    # info without disconnecting the first callback (a second button
    # pressed mid-drag -- Ctrl-click is the right button on a Mac
    # trackpad), and that orphan then raised AttributeError on every
    # later mouse move once the info was cleared (user report from a Mac,
    # 2026-10-15). So: finish an unfinished drag before a new press, and
    # make an orphaned callback a no-op that disconnects itself.

    def _cancel_unfinished(self) -> None:
        owner = self.parent()
        if isinstance(owner, MplCanvas) and (getattr(self, "_zoom_info", None) is not None
                                             or getattr(self, "_pan_info", None) is not None):
            owner._cancel_navigation_drag()

    def press_pan(self, event) -> None:
        owner = self.parent()
        if isinstance(owner, MplCanvas):
            owner.note_natural_view()
        self._cancel_unfinished()
        super().press_pan(event)

    def press_zoom(self, event) -> None:
        owner = self.parent()
        if isinstance(owner, MplCanvas):
            owner.note_natural_view()
        self._cancel_unfinished()
        super().press_zoom(event)

    def drag_zoom(self, event) -> None:
        if getattr(self, "_zoom_info", None) is None:
            self._drop_orphan_callback(self.drag_zoom)
            return
        super().drag_zoom(event)

    def drag_pan(self, event) -> None:
        if getattr(self, "_pan_info", None) is None:
            self._drop_orphan_callback(self.drag_pan)
            return
        super().drag_pan(event)

    def _drop_orphan_callback(self, method) -> None:
        """Disconnect every motion callback bound to ``method`` (an
        orphan whose drag already ended)."""
        try:
            callbacks = self.canvas.callbacks.callbacks.get("motion_notify_event", {})
            for cid, ref in list(callbacks.items()):
                func = ref()
                if func is not None and getattr(func, "__func__", None) is method.__func__ \
                        and getattr(func, "__self__", None) is self:
                    self.canvas.mpl_disconnect(cid)
        except Exception:  # noqa: BLE001 -- private matplotlib registry; a no-op is enough
            pass

    def save_figure(self, *args) -> None:
        """Overridden so a saved figure is never silently missing
        points the live view dropped purely for faster interactive
        rendering (see :meth:`MplCanvas.export_full_resolution`) --
        applies no matter how this is triggered (this toolbar's own
        Save icon, or programmatically, e.g. the File menu's Save
        Figure As…), since both end up calling this same method.

        Since 0.8.4 an :class:`MplCanvas` first shows a preview of the
        figure exactly as it will be written (explicit user request:
        "a preview that accurately reflects the position of the legend,
        text size relative to the figure size"); see
        :mod:`opennmr.gui.figure_preview_dialog`."""
        owner = self.parent()
        if isinstance(owner, MplCanvas):
            owner.open_export_preview()
            return
        super().save_figure(*args)


class _ProjectionStyleDialog(QDialog):
    """One projection strip's look: colour, line width, line style,
    size, shown or hidden -- opened by right-clicking the strip.

    Emits :attr:`changed` on every edit so the plot previews it live;
    the canvas undoes the preview on Cancel."""

    changed = Signal()

    def __init__(self, parent, dimension: str, style: dict, default_colour: str = "black") -> None:
        super().__init__(parent)
        name = "F2 (top)" if dimension == "f2" else "F1 (left)"
        self.setWindowTitle(f"{name} projection style")
        self._default_colour = default_colour
        self._colour = style.get("color")
        form = QFormLayout(self)

        colour_row = QHBoxLayout()
        self.colour_button = QPushButton()
        self.colour_button.setToolTip("Pick the line colour.")
        self.colour_button.clicked.connect(self._on_pick_colour)
        self.colour_default_button = QPushButton("Default")
        self.colour_default_button.setToolTip(
            "Black for a projection computed from the 2D, crimson for a dropped-in 1D.")
        self.colour_default_button.clicked.connect(lambda: self.set_colour(None))
        colour_row.addWidget(self.colour_button, 1)
        colour_row.addWidget(self.colour_default_button)
        form.addRow("Colour:", colour_row)

        import matplotlib as _mpl
        self.linewidth_spin = QDoubleSpinBox()
        self.linewidth_spin.setRange(0.1, 10.0)
        self.linewidth_spin.setSingleStep(0.1)
        self.linewidth_spin.setValue(float(style.get("linewidth", _mpl.rcParams["lines.linewidth"])))
        form.addRow("Line width:", self.linewidth_spin)

        self.linestyle_combo = QComboBox()
        for label, value in MplCanvas.PROJECTION_LINESTYLES:
            self.linestyle_combo.addItem(label, value)
        self.linestyle_combo.setCurrentIndex(max(0, self.linestyle_combo.findData(style.get("linestyle", "-"))))
        form.addRow("Line style:", self.linestyle_combo)

        self.size_spin = QDoubleSpinBox()
        self.size_spin.setRange(0.2, 4.0)
        self.size_spin.setSingleStep(0.1)
        self.size_spin.setSuffix(" in")
        self.size_spin.setValue(float(style.get("size", 1.0)))
        self.size_spin.setToolTip(
            "How deep the strip is. The contour takes whatever space the strips leave.")
        form.addRow("Strip size:", self.size_spin)

        self.visible_check = QCheckBox("Show this projection")
        self.visible_check.setChecked(style.get("visible", True) is not False)
        form.addRow(self.visible_check)

        self.both_check = QCheckBox("Apply to both projections")
        self.both_check.setToolTip("Give the F1 and F2 strips the same style.")
        form.addRow(self.both_check)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                   | QDialogButtonBox.StandardButton.Cancel
                                   | QDialogButtonBox.StandardButton.RestoreDefaults)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        buttons.button(QDialogButtonBox.StandardButton.RestoreDefaults).clicked.connect(self.restore_defaults)
        form.addRow(buttons)

        self._refresh_colour_button()
        for signal in (self.linewidth_spin.valueChanged, self.linestyle_combo.currentIndexChanged,
                       self.size_spin.valueChanged, self.visible_check.toggled, self.both_check.toggled):
            signal.connect(lambda *_: self.changed.emit())

    def _refresh_colour_button(self) -> None:
        shown = self._colour or self._default_colour
        self.colour_button.setText(shown if self._colour else f"{shown} (default)")
        self.colour_button.setStyleSheet(
            f"QPushButton {{ border-left: 14px solid {matplotlib.colors.to_hex(shown)}; }}")

    def set_colour(self, colour) -> None:
        self._colour = colour
        self._refresh_colour_button()
        self.changed.emit()

    def _on_pick_colour(self) -> None:
        from PySide6.QtGui import QColor
        from PySide6.QtWidgets import QColorDialog
        start = QColor(matplotlib.colors.to_hex(self._colour or self._default_colour))
        colour = QColorDialog.getColor(start, self, "Projection colour")
        if colour.isValid():
            self.set_colour(colour.name())

    def restore_defaults(self) -> None:
        import matplotlib as _mpl
        for widget in (self.linewidth_spin, self.linestyle_combo, self.size_spin, self.visible_check):
            widget.blockSignals(True)
        self.linewidth_spin.setValue(float(_mpl.rcParams["lines.linewidth"]))
        self.linestyle_combo.setCurrentIndex(0)
        self.size_spin.setValue(1.0)
        self.visible_check.setChecked(True)
        for widget in (self.linewidth_spin, self.linestyle_combo, self.size_spin, self.visible_check):
            widget.blockSignals(False)
        self._colour = None
        self._refresh_colour_button()
        self.changed.emit()

    def apply_to_both(self) -> bool:
        return self.both_check.isChecked()

    def style(self) -> dict:
        """The settings as ``projection_style`` takes them, left out
        where they equal the default so a later change of default
        still reaches them."""
        import matplotlib as _mpl
        style = {}
        if self._colour:
            style["color"] = self._colour
        if abs(self.linewidth_spin.value() - float(_mpl.rcParams["lines.linewidth"])) > 1e-9:
            style["linewidth"] = self.linewidth_spin.value()
        if self.linestyle_combo.currentData() != "-":
            style["linestyle"] = self.linestyle_combo.currentData()
        if abs(self.size_spin.value() - 1.0) > 1e-9:
            style["size"] = self.size_spin.value()
        if not self.visible_check.isChecked():
            style["visible"] = False
        return style


def _render_label(text: str, fontsize: float = 12.0, dpi: float = 100.0):
    """``(png_bytes, None)`` for ``text`` drawn exactly as matplotlib
    would draw an axis label, or ``(None, error message)`` when it
    cannot be drawn.

    Rendering, not just parsing, is the check: an unknown mathtext
    command (``$\\foo$``) is only found when matplotlib draws, and
    raised there it would take the whole plot down with it."""
    import io
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    figure = Figure(figsize=(0.1, 0.1), dpi=dpi)
    FigureCanvasAgg(figure)
    figure.text(0, 0, text, fontsize=fontsize)
    buffer = io.BytesIO()
    try:
        figure.savefig(buffer, format="png", bbox_inches="tight", pad_inches=0.02,
                       transparent=True)
    except Exception as exc:  # noqa: BLE001 -- any failure means "cannot be drawn"
        message = str(exc).strip().splitlines()
        return None, (message[-1] if message else type(exc).__name__)
    return buffer.getvalue(), None


class _AxisLabelsDialog(QDialog):
    """Edit a 2D's two axis labels separately -- explicit user request:
    "keep the name of the F2 axis but change F1".

    Blank = the automatic label (which keeps following Plot Style's
    label settings). Each axis shows a preview drawn by matplotlib
    itself, so mathtext is seen as it will print, and OK is refused
    while a label cannot be drawn."""

    def __init__(self, parent, *, f2_text: str, f1_text: str, f2_auto: str, f1_auto: str,
                 f1_enabled: bool = True) -> None:
        super().__init__(parent)
        self.setWindowTitle("Axis labels")
        self.resize(560, 0)
        self._auto = {"f2": f2_auto, "f1": f1_auto}
        layout = QVBoxLayout(self)
        self.edits, self.previews = {}, {}
        for dim, title, text in (("f2", "F2 (horizontal axis)", f2_text),
                                 ("f1", "F1 (vertical axis)", f1_text)):
            layout.addWidget(QLabel(f"<b>{title}</b>"))
            row = QHBoxLayout()
            edit = QLineEdit("" if text == self._auto[dim] else text)
            edit.setPlaceholderText("Automatic")
            edit.setClearButtonEnabled(True)
            copy_button = QPushButton("Start from automatic")
            copy_button.setToolTip(
                "Put the automatic label's own text in the box, to edit rather than retype.")
            copy_button.clicked.connect(lambda _checked=False, e=edit, d=dim: e.setText(self._auto[d]))
            row.addWidget(edit, 1)
            row.addWidget(copy_button)
            layout.addLayout(row)
            preview = QLabel()
            preview.setMinimumHeight(28)
            preview.setWordWrap(True)
            layout.addWidget(preview)
            if dim == "f1" and not f1_enabled:
                for widget in (edit, copy_button):
                    widget.setEnabled(False)
                edit.setPlaceholderText("A pseudo-2D's vertical axis is a slice number")
            self.edits[dim], self.previews[dim] = edit, preview
            edit.textChanged.connect(lambda _t, d=dim: self._refresh_preview(d))
        hint = QLabel(
            "Blank = automatic. Mathtext works, e.g. $\\delta$ $^{13}$C (ppm). "
            "{nuc}, {mass}, {element} and {unit} are filled in for that axis. "
            "A label without {unit} is used only in the unit it was typed in; "
            "in another unit the axis shows its automatic label.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: palette(mid);")
        # A wrapped QLabel reports a one-line height before the dialog
        # is laid out, which clipped this to two lines.
        hint.setMinimumHeight(hint.fontMetrics().lineSpacing() * 4 + 4)
        layout.addWidget(hint)
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                        | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self._errors = {}
        for dim in ("f2", "f1"):
            self._refresh_preview(dim)

    def _refresh_preview(self, dim: str) -> None:
        from PySide6.QtGui import QPixmap
        text = self.edits[dim].text().strip() or self._auto[dim]
        preview = self.previews[dim]
        png, error = _render_label(text) if text else (None, None)
        self._errors[dim] = error
        if error:
            preview.setPixmap(QPixmap())
            preview.setText(f"<span style='color:#b00020'>Cannot be drawn: {error}</span>")
        elif png:
            pixmap = QPixmap()
            pixmap.loadFromData(png, "PNG")
            preview.setText("")
            preview.setPixmap(pixmap)
        ok = self.buttons.button(QDialogButtonBox.StandardButton.Ok) if hasattr(self, "buttons") else None
        if ok is not None:
            ok.setEnabled(not any(self._errors.values()))

    def f2_text(self) -> str:
        return self.edits["f2"].text().strip() or self._auto["f2"]

    def f1_text(self) -> str:
        return self.edits["f1"].text().strip() or self._auto["f1"]


def _rows_from_list_drag(mime, fmt: str) -> list[int]:
    """Source rows of a QListWidget drag payload, in order, once each --
    the same decoding as the Compare list's own drop target."""
    from PySide6.QtCore import QDataStream, QIODevice
    # Held in a name: QDataStream reads from the QByteArray without
    # owning it, and a temporary would be freed underneath it.
    payload = mime.data(fmt)
    stream = QDataStream(payload, QIODevice.OpenModeFlag.ReadOnly)
    rows: list[int] = []
    while not stream.atEnd():
        row, _column, n_roles = stream.readInt32(), stream.readInt32(), stream.readInt32()
        for _ in range(n_roles):
            stream.readInt32()
            stream.readQVariant()
        if row not in rows:
            rows.append(row)
    return rows


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
    #: Which side of the axis line the ticks are drawn on: "in", "out"
    #: or "inout" (across it) -- matplotlib's own names. Global Plot
    #: Style option (explicit user request, 2026-10-01).
    _TICK_DIRECTION = "in"
    #: Space between a 1D spectrum's lowest point and the axis line, and
    #: above its highest point, as fractions of the trace's own height;
    #: and the space between neighbouring traces of a stacked Compare.
    #: Plot Style ▸ "Gap below spectra" / "Gap between stacked spectra"
    #: (explicit user request, 2026-10-15: a slice whose baseline sits
    #: below zero was drawn partly under the axis), measured from the
    #: *full-resolution* data (see _fit_y_to_data). The gap below is 10 %
    #: by default -- twice matplotlib's margin -- so the baseline always
    #: floats visibly clear of the axis line (follow-up the same day);
    #: the extracted-slice panel beside a contour uses both too.
    _BASELINE_GAP = 0.10
    _STACK_GAP = 0.05
    _TOP_GAP = 0.05
    #: The stacked-spectra gap may go **negative** -- spectra that overlap,
    #: each reaching into the one above by that fraction of its height
    #: (Compare ▸ Spacing slider, 2026-10-28). -0.9 keeps a spectrum's
    #: baseline 10 % of its height from the next one's.
    _STACK_GAP_MIN = -0.90
    TICK_DIRECTIONS = (("In", "in"), ("Out", "out"), ("In and out", "inout"))

    #: Default **export** figure size, inches. Not applied to the live
    #: embedded figure at any point -- see :meth:`set_export_settings`
    #: for why that fights Qt's own canvas resizing. A 2D carrying
    #: extracted slices gets its own pair, since it is a two-panel
    #: layout rather than one square contour.
    _EXPORT_FIG_SIZE = (8.0, 8.0)
    #: Landscape, not portrait: the sliced layout puts the trace panel
    #: *beside* the contour, so the figure is genuinely wider than tall
    #: and a portrait page squeezes the contour for no reason. (The
    #: original 7.5 x 10 was taken literally from a request that
    #: predated the side-by-side panel.)
    _EXPORT_FIG_SIZE_WITH_SLICES = (10.0, 7.5)

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

    #: Emitted with the right-clicked ``(x, y)`` position when the user
    #: asks to reference a 2D from the cursor. Carries the position
    #: rather than letting MainWindow read ``_last_right_click_xy``
    #: back out, so the handler cannot act on a *different* click than
    #: the one that opened the menu.
    referenceCalibrationRequested = Signal(object)

    #: Emitted with the right-clicked (x, y) data position when
    #: "Exclude Nearest Point" is chosen from the right-click menu,
    #: while a relaxation/diffusion curve fit (show_relaxation_fit) is
    #: the current view -- the canvas only knows where the click
    #: landed, not which fit-result point that's closest to or how to
    #: re-fit without it, so it just asks the owner (MainWindow).
    fitPointExcludeRequested = Signal(float, float)

    #: Emitted (no value) when "Restore Removed Points" is chosen --
    #: same division of responsibility as fitPointExcludeRequested.
    fitPointsResetRequested = Signal()

    #: Emitted with the right-clicked (x, y) when "Remove Slice from
    #: Every Fit" is chosen on a curve fit of a series (one peak or all
    #: peaks together): the whole slice nearest the click is to leave
    #: every fit, as unticking it in Dynamics' "Slices used in fits"
    #: does (explicit user request, 2026-11-10).
    fitSliceRemoveRequested = Signal(float, float)

    #: Emitted with the right-clicked x (data units) when "Calibrate
    #: Integral Here…" is chosen on a 1D spectrum with integration
    #: regions -- the owner finds the region and asks for its value.
    integralCalibrationRequested = Signal(float)
    #: "Reset Integral Calibration": back to the smallest integral = 1.
    integralCalibrationResetRequested = Signal()

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

    #: Emitted when the layout pass could not keep every axis label on
    #: the figure even after shrinking the text to its floor -- i.e.
    #: the panel is genuinely too small for what it has to show, and
    #: only the user can fix it (make the window bigger, or the side
    #: panel narrower). Carries the list of label texts still cut off.
    #: Emitted only on a *change* of state, so a phase drag that
    #: relayouts continuously doesn't repeat itself; a matching empty
    #: emission says the problem has cleared.
    labelsClipped = Signal(object)

    #: Emitted with the full list of extracted-slice specs whenever
    #: the user adds or clears one, so MainWindow can store them
    #: against the spectrum (and persist them) rather than leaving
    #: them as canvas-local state that a reselection would drop.
    contourSlicesChanged = Signal(object)
    #: A scroll over the contours of a Compare 2D plot (overlay or panels):
    #: the factor to multiply the lowest contour by (>1 fewer levels).
    compareContourScrolled = Signal(float)

    #: The DOSY contour floor, as a fraction of the map's own peak --
    #: the same quantity :attr:`_CONTOUR_FACTOR_DEFAULT` is for a 2D,
    #: and adjusted by the same scroll gesture. 2% keeps the weak
    #: components a DOSY exists to reveal.
    _DOSY_FACTOR_DEFAULT = 0.02
    _DOSY_FACTOR_MIN = 1e-4
    _DOSY_FACTOR_MAX = 0.9

    #: Contour levels drawn on a DOSY map. Matches ``plot_dosy``'s own
    #: default, so the panel control starts showing what is on screen.
    _DOSY_CLEVELS_DEFAULT = 8

    #: Default contour ring weight. Matches plot_2d_contour's and
    #: plot_dosy's own, so the control starts showing what is drawn.
    _CONTOUR_LINEWIDTH_DEFAULT = 0.8

    @classmethod
    def _contour_linewidth_default(cls) -> float:
        return cls._CONTOUR_LINEWIDTH_DEFAULT

    #: Emitted with a request to extract a trace from the DOSY map:
    #: ``(kind, value)`` where ``kind`` is ``"spectrum"`` (the map's
    #: row at one diffusion coefficient) or ``"profile"`` (its column
    #: at one chemical shift), and ``value`` is the D or ppm
    #: right-clicked. The canvas knows where the click landed; only
    #: MainWindow can put the result into the session.
    dosyExtractRequested = Signal(str, float)

    #: Emitted with the DOSY's contour floor (a fraction of the map's
    #: own peak) whenever the *scroll* gesture changes it, so a panel
    #: control showing the same number follows along instead of
    #: drifting out of step with the plot. Not emitted when the panel
    #: itself sets the floor -- that would be the control echoing back
    #: to itself.
    dosyLevelFloorChanged = Signal(float)

    #: Emitted with the finished measurement when the user completes one
    #: by clicking two points while measurement mode is active. Carries
    #: the two points in display units, the separation along each axis
    #: in every unit it converts to, and the already-formatted
    #: ``summary`` string -- see :meth:`_complete_measurement`.
    measurementCompleted = Signal(dict)

    #: Emitted when this canvas's own toolbar toggles measurement mode
    #: or the crosshair, so MainWindow can keep the matching menu item
    #: (and every other canvas) in step. Emitted on a real change only.
    measurementModeChanged = Signal(bool)
    #: A peak width (FWHM) measured by a click in Width mode: the
    #: :mod:`opennmr.notebook.peak_width` result plus ``summary``,
    #: ``kind`` ("1d"/"2d") and ``spectrum`` (its label); or ``{"error":
    #: message}`` when there is no peak to measure there.
    peakWidthMeasured = Signal(dict)
    widthModeChanged = Signal(bool)
    #: Inset mode (draw a box on the plot to magnify) switched on or off.
    insetModeChanged = Signal(bool)
    #: An inset was added from a box drawn on the plot (its index), or the
    #: insets on this plot were changed some other way.
    insetAdded = Signal(int)
    insetsChanged = Signal()
    #: The user asked for the inset options (right-click on the Inset button).
    insetOptionsRequested = Signal()
    #: The toolbar's × button was pressed: the owner asks for a factor and
    #: makes the scaled copy (see MainWindow._on_scale_spectrum_requested).
    scaleRequested = Signal()
    #: An inset could not be added (the reason).
    insetFailed = Signal(str)
    #: Click-to-pick / drag-to-select switched on or off -- including
    #: *by the canvas*, when another interaction mode takes over (see
    #: _release_other_modes), so the button that started it can uncheck.
    pickModeChanged = Signal(bool)
    regionSelectModeChanged = Signal(bool)
    crosshairEnabledChanged = Signal(bool)
    #: The crosshair's colour or line width was changed from its toolbar
    #: button's right-click menu -- MainWindow applies it to every canvas
    #: and saves it (Settings ▸ Interface holds the same two controls).
    crosshairStyleChanged = Signal(str, float)
    #: Measure's colour, line width and text size were changed from its
    #: toolbar button's menu: ``(colour, line width, text size)``, so the
    #: window can apply them to every plot and remember them.
    measurementStyleChanged = Signal(str, float, float)

    #: What the crosshair looks like unless the user has chosen otherwise.
    DEFAULT_CROSSHAIR_COLOR = "0.35"
    DEFAULT_CROSSHAIR_WIDTH = 0.8
    CROSSHAIR_WIDTH_RANGE = (0.1, 6.0)
    #: What Measure's arrows and labels look like unless the user has
    #: chosen otherwise (explicit user request, 2026-11-12: "change the
    #: size and colour of the Measure markers to make it easier to see").
    DEFAULT_MEASUREMENT_COLOR = "tab:red"
    DEFAULT_MEASUREMENT_WIDTH = 1.4
    DEFAULT_MEASUREMENT_FONTSIZE = 10.0
    MEASUREMENT_WIDTH_RANGE = (0.3, 8.0)
    MEASUREMENT_FONTSIZE_RANGE = (5.0, 40.0)
    #: The dashed zoom box drawn while click-dragging a region to zoom
    #: to: line width in screen pixels (matplotlib's own is a hairline of
    #: 1 px -- explicit user request: "too thin", adjustable in Settings).
    DEFAULT_ZOOM_BOX_WIDTH = 2.0
    ZOOM_BOX_WIDTH_RANGE = (0.5, 10.0)
    #: The contour right-click menu's "Scroll Up Shows More Contours"
    #: was toggled -- MainWindow applies it to every canvas and saves it
    #: (Settings ▸ Interface holds the same switch).
    contourScrollDirectionChanged = Signal(bool)

    #: The user picked a different result to show from this canvas's
    #: own view switcher (see :meth:`set_view_switcher`), carrying that
    #: entry's key. The canvas does not act on this itself: it holds no
    #: record of how any of those results were produced, so redrawing
    #: one is the owner's job.
    viewSwitchRequested = Signal(str)

    #: A row (or rows) of the Open experiments list was dropped on this
    #: canvas while it shows a 2D contour -- ``(session_rows, target)``,
    #: ``target`` being ``"f2"``/``"f1"`` when released over that
    #: projection strip and ``"auto"`` over the contour itself (the
    #: owner then picks the dimension from the nuclei). The canvas only
    #: reports the drop: checking the nucleus and asking the user are
    #: the owner's, which knows what the rows are.
    externalProjectionDropped = Signal(list, str)
    #: Rows of the Open experiments list dropped on the plot itself (not on a projection strip).
    experimentRowsDropped = Signal(list)
    #: Files / folders dragged in from the file explorer / Finder and dropped on the plot.
    pathsDropped = Signal(list)

    #: Qt's own payload type for a drag out of a QListWidget -- the
    #: Open experiments list's.
    _LIST_DRAG_MIME = "application/x-qabstractitemmodeldatalist"

    #: Tint of a projection strip while a drop there would replace it.
    _DROP_HIGHLIGHT_COLOUR = "#dbe8fb"

    #: Tag applied to every persistent integration-region axvspan (see
    #: :func:`set_regions`) so they can be found and cleared by gid on
    #: the next call, without needing to track a separate reference
    #: list that would go stale across a figure.clear() (a brand new
    #: Axes each time, invalidating any previously held patch objects).
    _REGION_PATCH_GID = "integration_region"
    _integral_label_items: list = []
    _integral_label_style: dict = {}

    #: Tag on the not-yet-committed baseline overlay, so it can be
    #: found and replaced without keeping a reference that a figure
    #: rebuild would invalidate -- same approach as the region patches
    #: above, and for the same reason.
    _BASELINE_PREVIEW_GID = "baseline_preview"

    #: Tag on the faint "before correction" trace drawn beneath the
    #: baseline overlay -- see :meth:`set_baseline_preview`.
    _BASELINE_GHOST_GID = "baseline_preview_uncorrected"

    def __init__(self, parent=None, full_resolution: bool = False) -> None:
        super().__init__(parent)

        self.figure = Figure()
        # Plot Style ▸ Lines: every line's width and style, applied as
        # the figure draws (and so on export too), whatever drew it --
        # see plotting.lines. Set before anything can draw.
        self._line_width_scale = 1.0
        self._line_overrides: dict[str, dict] = {}
        self._line_keys_seen: tuple = ()
        self.figure.draw = self._draw_with_line_styles
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasQTAgg(self.figure)
        # Every save (Save Figure, the toolbar, a style template preview)
        # goes through print_figure: vector formats are made safe for
        # Illustrator there (plotting.vector_export; user report,
        # 2026-10-17: a fit's components vanished from PDF/SVG opened in
        # Illustrator, which drops paths of more than ~32k points).
        self.canvas.print_figure = self._print_figure_vector_safe
        # The zoom box is drawn by the canvas's drawRectangle; replacing it
        # on this instance lets its line width be set (set_zoom_box_width).
        self.canvas.drawRectangle = self._draw_zoom_rectangle
        self.toolbar = _Toolbar(self.canvas, self)

        layout = QVBoxLayout(self)
        layout.addWidget(self.toolbar)

        # Crosshair/Measure live on the canvas's own toolbar, beside
        # pan/zoom, rather than only in the menu bar: both are things
        # you reach for *while looking at the plot*, and both are modes
        # you need to be able to see the state of. Explicit user
        # report that the crosshair button had gone missing.
        #
        # Plain QToolButtons rather than QActions added to the toolbar:
        # a QToolBar renders an icon-less QAction inconsistently across
        # platforms (macOS in particular can end up showing nothing at
        # all), while a QToolButton set to ToolButtonTextOnly always
        # shows its text. There are no icons for these two.
        self.toolbar.addSeparator()
        # "Plot style": the same menu as right-clicking the plot (Plot Style…, legend,
        # labels, insets ...), for those who do not know to right-click (user request,
        # 2026-11-23). Opens under the button; entries that need a clicked point are
        # simply unavailable, as the menu already handles.
        self._style_button = QToolButton(self.toolbar)
        self._style_button.setText("Plot style")
        self._style_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self._style_button.setToolTip(
            "The plot's style menu, the same as right-clicking the plot: Plot Style…, legend, "
            "labels, insets, export and more.")
        self._style_button.clicked.connect(lambda _checked=False: self._show_style_menu_from_button())
        self.toolbar.addWidget(self._style_button)
        self._crosshair_button = self._add_toolbar_toggle(
            "Crosshair",
            "Show a cross following the cursor across the plot (and, on a 2D, "
            "across both projection strips). Cmd/Ctrl+. toggles it. Right-click "
            "for its colour and line width.",
            self.set_crosshair_enabled,
            checked=True,
        )
        self._crosshair_button.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._crosshair_button.customContextMenuRequested.connect(
            self._on_crosshair_button_menu)
        self._measure_button = self._add_toolbar_toggle(
            "Measure",
            "Measure a separation: click two points on the plot and the "
            "distance between them is labelled in ppm and Hz at once (and in "
            "mm while the axis is displayed in mm). Cmd/Ctrl+D toggles "
            "this from anywhere.",
            self.set_measurement_mode,
            checked=False,
        )
        self._measure_button.setToolTip(self._measure_button.toolTip()
                                         + " Right-click for its colour, line width and text size.")
        self._measure_button.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._measure_button.customContextMenuRequested.connect(self._on_measure_button_menu)
        # Click a peak for its full width at half maximum (explicit user
        # request, 2026-10-16), 1D and 2D; an estimate, said so, where
        # peaks overlap -- notebook.peak_width.
        self._width_button = self._add_toolbar_toggle(
            "Width",
            "Peak width: click on a peak and its full width at half maximum is drawn and "
            "given in ppm and Hz (along F2 and F1 on a 2D). Where peaks overlap the value "
            "is marked as an estimate – fit the peaks for a reliable width. Cmd/Ctrl+L toggles "
            "this from anywhere.",
            self.set_width_mode,
            checked=False,
        )

        self._inset_button = self._add_toolbar_toggle(
            "Inset",
            "Magnify part of the plot: drag a box over the region and a zoomed copy is "
            "placed on the plot. Works on spectra, 2D contours, curves and profiles. "
            "Right-click this button, or the plot, for the inset's position, typeface, "
            "connecting lines and axes.",
            self.set_inset_mode,
            checked=False,
        )
        self._inset_button.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._inset_button.customContextMenuRequested.connect(
            lambda _pos: self.insetOptionsRequested.emit())

        # Multiply the spectrum on show by a factor of one's own choosing (user
        # request, 2026-11-19). A push button, not a mode: it asks, then acts.
        self._scale_button = QToolButton(self.toolbar)
        self._scale_button.setText("× Scale")
        self._scale_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self._scale_button.setToolTip(
            "Multiply this spectrum by a factor you choose (2, 0.5, 1/3, -1 …). A new spectrum is "
            "added to the list and the original is left as it was. To scale one trace in a "
            "comparison, right-click it in the Compare list.")
        self._scale_button.clicked.connect(lambda _checked=False: self.scaleRequested.emit())
        self._scale_button_action = self.toolbar.addWidget(self._scale_button)
        self._scale_button_action.setVisible(False)

        # "Showing:" -- which stored result this canvas is displaying,
        # when its owner has several to choose between (a curve fit, a
        # DOSY map, a SCORE fit, an eNMR fit all land on the same Fit
        # canvas and used to overwrite one another). Reported directly:
        # "I still don't have a clear button to alternate between e.g.
        # a curve fit, a DOSY ILT plot, or a SCORE fit".
        #
        # **On the toolbar rather than in a bar above the canvas.**
        # ``fit_canvas`` is referenced as a tab *widget* in sixteen
        # places and ninety-eight test sites; wrapping it in a
        # container to gain a header row would invalidate every
        # ``canvas_tabs.indexOf(self.fit_canvas)`` among them. The
        # toolbar is already this canvas's own place for controls that
        # belong to the plot (Crosshair, Measure), and costs no
        # re-parenting.
        #
        # Hidden until there is something to switch *between* -- a
        # one-entry switcher is a label pretending to be a control.
        self._view_switcher_label = QLabel(" Showing: ", self.toolbar)
        self._view_switcher = QComboBox(self.toolbar)
        self._view_switcher.setToolTip(
            "Every result stored for this spectrum. Pick one to bring it back "
            "without re-running it."
        )
        self._view_switcher.activated.connect(self._on_view_switcher_activated)
        self._view_switcher_label_action = self.toolbar.addWidget(self._view_switcher_label)
        self._view_switcher_action = self.toolbar.addWidget(self._view_switcher)
        self._view_switcher_label_action.setVisible(False)
        self._view_switcher_action.setVisible(False)

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
        # **Say loudly that the slice is only being viewed.** Explicit
        # user request: make "not yet saved" more obvious -- a warning
        # icon, larger bold text, the key words underlined. Phasing or
        # fitting what is on screen here acts on the 2D, not the slice,
        # until it is kept.
        self._slice_warning_icon = QLabel()
        icon = self.style().standardIcon(QStyle.StandardPixmap.SP_MessageBoxWarning)
        self._slice_warning_icon.setPixmap(icon.pixmap(22, 22))
        self._slice_warning_icon.setAccessibleName("Warning")
        slice_bar_layout.addWidget(self._slice_warning_icon)
        self._slice_label = QLabel("")
        self._slice_label.setTextFormat(Qt.TextFormat.RichText)
        slice_font = self._slice_label.font()
        if slice_font.pointSizeF() > 0:
            slice_font.setPointSizeF(slice_font.pointSizeF() * 1.25)
        self._slice_label.setFont(slice_font)
        # Wraps rather than being cut off on a narrow plot.
        self._slice_label.setWordWrap(True)
        slice_bar_layout.addWidget(self._slice_label, 1)

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

        # External projections: a 1D dropped onto a contour's
        # projection strip (see externalProjectionDropped). Keyed by the
        # 2D's id() with the 2D itself kept alongside, so switching to
        # another spectrum and back finds them again, and a recycled
        # id can never hand one 2D's projections to another.
        self._external_projections: dict[int, dict] = {}
        # Each projection strip's own look (right-click the strip):
        # color/linewidth/linestyle/size/visible, as
        # plotting.twod.projection_line_kwargs takes it. Per canvas,
        # like the rest of Plot Style -- it carries over to the next 2D.
        self._projection_style: dict[str, dict] = {"f1": {}, "f2": {}}
        # How the strips are computed from the 2D: "auto" (= maximum),
        # "max" or "sum" -- plot_2d_contour's projection_mode. Per
        # canvas, switched from the right-click menu.
        self._projection_mode = "auto"
        # Per-2D axis-label overrides, {"f2"/"f1": {"text", "unit"}} --
        # per spectrum, since a label says what *that* spectrum's axis
        # is. Same keying as _external_projections.
        self._axis_label_overrides: dict[int, dict] = {}
        self._last_right_click_axes = None
        self._drop_highlight = None  # (Axes, previous facecolor) while hovering a strip
        #: Set by the window for the plots that take dropped experiments (``experimentRowsDropped``)
        #: and files / folders (``pathsDropped``). Off by default: a canvas that does not take
        #: files leaves them to carry on up to the window's own folder drop.
        self.accept_row_drops = False
        self.accept_file_drops = False
        self.canvas.setAcceptDrops(True)
        self.canvas.installEventFilter(self)

        self._pan_start = None  # (x_pixel, y_pixel, xlim, ylim) while dragging
        #: What a plain left-drag does with no toolbar tool or click mode
        #: on: "pan" (this widget's default) or "zoom" (a box). MainWindow
        #: sets every canvas from Settings ▸ Interface (default "zoom").
        self._drag_mode = "pan"
        self._zoom_start = None  # (x_pixel, y_pixel, Axes) during a box-zoom drag
        # The view a fresh plot drew, {Axes: (xlim, ylim)} -- what Home
        # returns to. Recorded lazily, the first time the view is about
        # to change (see note_natural_view); cleared by _clear_figure.
        self._home_limits = None
        # Typed axis limits (see set_view_limits): (low, high) pairs, either
        # side None for "automatic", or None for no limit at all. Re-applied by
        # every draw that supports them; in the unit they were typed in.
        self._view_x = None
        self._view_y = None
        self._view_unit = "ppm"
        self._view_limits_note = ""
        # {Axes: (xlim, ylim)} the plot had before the typed limits were first
        # applied to it -- what clearing them returns to.
        self._natural_limits = None
        self._legend_visible = False
        self._legend_loc = "best"
        # Columns (0: whatever the legend's builder chose) and the
        # upper-left corner of a "custom" legend, in axes fractions --
        # plotting.legend_placement (user request, 2026-10-27).
        self._legend_columns = 0
        self._legend_x = 1.02
        self._legend_y = 1.0
        # --- Legend appearance ------------------------------------
        # **frameon=False and the axis label size by default.** A
        # framed legend at matplotlib's own default sits in an opaque
        # white box, which on a dense contour hides the data it is
        # labelling, and at its own default font size it disagrees
        # with every other piece of text on the figure. Explicit user
        # request: "default for legend should be the same font size as
        # axes and frameon=False, with an option to set the background
        # alpha/colour values or turn on the frame."
        #
        # ``_legend_fontsize`` of None means "follow _AXIS_LABELSIZE",
        # so the legend keeps up with the font-size control instead of
        # needing to be reset alongside it; a number pins it.
        self._legend_frameon = False
        self._legend_fontsize: float | None = None
        self._legend_framealpha = 0.8
        self._legend_facecolor: str | None = None
        # Width of the sample lines inside the legend, in points; None
        # draws them as wide as the traces (explicit user request,
        # 2026-09-28: thin spectra made the legend key hard to read).
        self._legend_linewidth: float | None = None
        # How traces are named when names are shown: "legend", or
        # "labels" -- each name written just above its own trace's noise,
        # at _trace_label_position (0 = plot's left edge, 1 = right).
        self._legend_style = "legend"
        self._trace_label_position = 0.02
        # Lift of the names above the noise, as a fraction of the plot height.
        self._trace_label_height = 0.0
        self._trace_label_artists: list = []
        #: True while the figure is one drawn by show_external_plot.
        self._external_plot = False
        #: Right-click "… Plot Options…" entries, by what is on show (see
        #: register_plot_options): owner -> (label, tooltip, callback).
        self._plot_options: dict = {}
        #: What show_external_plot was last given -- ``{"draw", "kind",
        #: "hz_per_ppm"}`` -- so a style change can redraw it (a colour
        #: map or a font size lives on artists only the draw makes).
        self._external_draw: dict | None = None
        #: Series colours on an external plot come from the figure's own
        #: draw until the series colour map is changed (or a colour set).
        self._external_palette = False
        #: The colour scale for maps and surfaces (tensor pictures, a
        #: Czjzek distribution); None = each plot's own default.
        self._map_colormap: str | None = None
        #: Points between a colour bar and its map or tensor surface
        #: (plotting.set_colorbar_gap); negative = each picture's own.
        self._colorbar_gap: float = -1.0
        #: How tensor pictures are drawn (Simulate ▸ ... ▸ Tensor
        #: picture): notebook.tensor_views.TENSOR_STYLE_DEFAULTS' keys,
        #: edited by right-click ▸ Tensor Plot Style… (explicit user
        #: request, 2026-10-01: "a specific tensor plot style").
        self._tensor_style: dict = {}
        #: True once the tensor style set the view, so a redraw takes the
        #: view from there instead of keeping the one dragged to.
        self._tensor_view_set = False
        # --- Title -------------------------------------------------
        # Off by default, on request. Several views set a title as
        # they draw (a slice number, a DOSY's own caption); this hides
        # the Artist rather than stopping them setting it, so turning
        # it back on needs no redraw and no view has to learn about
        # the setting.
        self._title_visible = False
        # Font family for every piece of text on the figure. None
        # leaves matplotlib's own default alone.
        self._font_family: str | None = None
        # Chemical-shift axis label -- the delta/nuclide/unit text
        # NUC_label builds. 'brackets' is opennmr's own default
        # (see core.registry.NUC_label); 'slash' is the older
        # "delta (^NUC) / ppm" form some people still prefer for
        # manuscripts. ``_axis_label_template``, when set, overrides
        # both entirely -- a literal custom label (see
        # _chemical_shift_label). Explicit user request: "people
        # should be allowed to swap between these or to even supply
        # their own custom label".
        self._axis_label_style: str = "brackets"
        self._axis_label_template: str | None = None
        self._aspect_ratio: float | None = None  # None = unlocked

        # Per-instance style state (Output tab / MainWindow.set_style) --
        # copied from the class-level defaults above; see their own
        # comment for why every show_* method reading self._AXIS_* etc.
        # already picks these up with no further changes needed.
        self._AXIS_LINEWIDTH = type(self)._AXIS_LINEWIDTH
        self._AXIS_LABELSIZE = type(self)._AXIS_LABELSIZE
        self._TICK_LENGTH = type(self)._TICK_LENGTH
        self._TICK_WIDTH = type(self)._TICK_WIDTH
        self._TICK_DIRECTION = type(self)._TICK_DIRECTION
        self._double_click_resets = True
        self._baseline_gap = type(self)._BASELINE_GAP
        self._stack_gap = type(self)._STACK_GAP
        self._grid_visible = False
        self._minor_ticks_visible = True
        #: Top/right spines on a "curve" plot (relaxation/diffusion
        #: fits, integral and parameter trends, DNP field profiles).
        #: Off by default -- the notebook plot functions' own
        #: ``show_top_spine=False, show_right_spine=False`` -- and
        #: switchable from Plot Style (explicit user request). Applied
        #: only to the Axes a curve view drew (``_curve_axes``), never
        #: to a spectrum, whose open-frame look is deliberate.
        self._top_spine_visible = False
        self._right_spine_visible = False
        self._curve_axes: list = []
        #: How a fit view draws its components -- shown, hidden or
        #: highlighted, filled or not, colours (notebook.fit_display).
        #: Kept per canvas and across refits, so a colour set for a
        #: site stays with it.
        self._fit_component_style = fit_display.component_style()
        self._slice_fit_args: dict | None = None
        self._slice_fit_active = False
        self._slice_fit_labels: list = []
        #: ``overlay(ax)`` drawn over the fit on show after every redraw
        #: of it (see show_slice_fit), or None.
        self._slice_fit_overlay = None
        self._curve_fit_shown: dict | None = None
        #: Every peak's curve fit, when they are shown together.
        self._curve_fits_shown: list = []
        #: Set by the owner while the integral labels on show are
        #: calibrated to a region's value (offers Reset Integral Calibration).
        self.integral_calibrated = False
        #: Set by the owner while the curve fit on show is a pseudo-2D's
        #: own slices, so a whole slice can be taken out of every fit
        #: from the right-click menu (False for a Compare stack or eNMR).
        self.series_slices_removable = False
        #: The palette every multi-series plot takes its default
        #: colours from -- Compare, extracted slices, the DOSY overlay
        #: and the 2D overlay. One setting rather than one per view, so
        #: two plots of the same samples agree without being set up
        #: twice. Individual series can override it; see SeriesStyle.
        self._series_colormap = "viridis"
        #: What kind of plot is showing -- see plotKindChanged.
        self._plot_kind: str | None = None
        #: Per-view series styling, keyed by the view that owns it.
        #: Written when a view is created and only ever *read* at draw
        #: time -- the rule that stops a colour resetting on a redraw.
        self._series: dict[str, list[SeriesStyle]] = {}
        #: The 1D trace's own colour. plot_1d has always taken one; the
        #: GUI simply never passed anything, so every Spectrum-tab
        #: trace was black. Explicit user request: "the colours should
        #: apply to the spectrum tab as well as the compare tab".
        self._trace_color = "black"
        #: Contour ring weight, for a 2D and a DOSY alike. Deliberately
        #: separate from _AXIS_LINEWIDTH, which styles axis furniture:
        #: at that setting's own 1.75 default the rings came out heavy
        #: enough to merge into blobs, which is why it was never
        #: forwarded. Explicit user request: "DOSY (all 2D?) line width
        #: should be editable on contour".
        self._contour_linewidth = self._CONTOUR_LINEWIDTH_DEFAULT
        #: Colour of the positive 2D contours: ``None`` (black), a colour
        #: map name (the colour follows the contour's height), a single
        #: colour, or a list of colours. See
        #: :func:`opennmr.plotting.style.resolve_contour_colouring`.
        self._contour_colour = None
        # Export-only DPI/figure-size -- see set_export_settings's own
        # docstring for why these never touch the live figure directly.
        self._export_dpi: float | None = None
        self._export_fig_width: float | None = None
        self._export_fig_height: float | None = None
        # Persists across a rebuild the same way legend/aspect-lock do
        # (see show_spectrum's own docstring) -- an explicit opt-out of
        # plot_1d's/plot_2d_contour's default rendering-only
        # downsampling (see opennmr.plotting.style.downsample_1d/
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
        # Which way a scroll over a contour body moves the levels. False
        # (the long-standing default): scroll up raises the floor, so
        # fewer contours ("zoom into the peaks"). True: scroll up lowers
        # it, so more contours. Asked for because some users find the
        # first counter-intuitive; one app-wide preference, set by
        # MainWindow from Settings ▸ Interface or the contour menu.
        self._contour_scroll_up_more = False
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
        # The DOSY equivalent: rows of the map drawn beside it (dashed
        # line across the map at each D + the spectrum at that D in a
        # panel to the right), each ``{"D", "label", "color"}``. Rows
        # only -- a DOSY's other extraction is a column over a log D
        # axis, which shares neither the panel's x axis nor its linear
        # stacking; see plot_dosy's own note.
        self._dosy_slices: list[dict] = []
        self._nq_values: list[int] = [1]
        self._pick_mode = False
        self._region_select_mode = False
        self._measurement_mode = False
        self._width_mode = False
        # Insets on any plot (see the "insets" section): the specs live
        # here, keyed by the plot they were drawn on, so they come back
        # whenever that plot is drawn again.
        self._inset_mode = False
        self._inset_start = None
        self._inset_store: dict = {}
        self._inset_context = None
        self._inset_target = 0
        self._inset_applying = False
        # The Qt cursor shape currently set on the canvas, so
        # _refresh_cursor can skip a setCursor that would change
        # nothing -- it runs on every mouse move, alongside a blitted
        # crosshair. None means "unknown", which forces the next call
        # to apply whatever it resolves; see _refresh_cursor for the
        # three places that deliberately reset it to None.
        self._last_cursor_shape = None
        # (axes, x_data, y_data) for each click of the current, not-yet-
        # complete measurement. The Axes is part of it so a second click
        # landing in a *different* Axes (a projection strip, another
        # Compare row) restarts the measurement there rather than
        # silently measuring across two unrelated coordinate systems.
        self._measurement_points: list[tuple[object, float, float]] = []
        # Every artist drawn for a measurement -- the pending first-point
        # marker, and the arrow + label of each completed one. Kept (so
        # several measurements can be placed at once, which is what a
        # J-coupling analysis actually needs) until measurement mode is
        # switched off or the figure is rebuilt.
        self._measurement_artists: list = []
        # Whether the crosshair is wanted at all -- the Crosshair toggle
        # on this canvas's own toolbar. On by default.
        self._crosshair_enabled = True
        # Crosshair appearance (see set_crosshair_style).
        self._crosshair_color = self.DEFAULT_CROSSHAIR_COLOR
        self._crosshair_width = self.DEFAULT_CROSSHAIR_WIDTH
        # Measure's appearance (see set_measurement_style).
        self._measurement_color = self.DEFAULT_MEASUREMENT_COLOR
        self._measurement_width = self.DEFAULT_MEASUREMENT_WIDTH
        self._measurement_fontsize = self.DEFAULT_MEASUREMENT_FONTSIZE
        self._zoom_box_width = self.DEFAULT_ZOOM_BOX_WIDTH
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
        # The crosshair (see _ContourCrosshair) -- None whenever a
        # contour isn't the current view. Always torn down through
        # _clear_figure, never left to notice for itself on a later
        # mouse-move -- see _ContourCrosshair's own docstring for why
        # that lazy approach was the actual bug in two earlier attempts.
        self._crosshair: _ContourCrosshair | None = None
        # Whether the *current* plot is one a crosshair belongs on.
        # Set by each show_* that draws one; cleared on every figure
        # rebuild, so a view that never opts in cannot inherit it from
        # whatever was on this canvas before.
        self._crosshair_ready = False
        # Set only for a plain 1D spectrum/slice view -- lets the
        # right-click menu offer ppm/Hz/kHz switching by re-using the
        # spectrum's own stored axis arrays, no recomputation needed.
        # None for any other view (contour, FID, compare, fit), where
        # unit-switching isn't offered.
        self._current_1d_spectrum: NMRSpectrum | None = None
        #: The Processing preview's own bottom-panel spectrum -- see
        #: :meth:`baseline_preview_source`. Only that layout sets it.
        self._preview_1d_spectrum: NMRSpectrum | None = None
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
        # A faint y = 0 line, shown while phasing / checking a baseline (user
        # request, 2026-11-19); see set_zero_line.
        self._zero_line = None
        self._zero_line_wanted = False
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
        # The spectra a Compare/component stack is currently showing --
        # needed by set_axis_unit, which has to rewrite every row.
        self._compare_spectra: list = []
        #: What show_compare was last called with, so a series restyle
        #: can redraw the same view rather than guessing at it.
        self._compare_kwargs: dict | None = None
        #: The 2D overlay's own spectra and labels, kept for the same
        #: reason.
        self._overlay_spectra: list = []
        self._overlay_kwargs: dict = {}
        self._panel_axes: list = []
        self._spectrum_panels: list = []         # Compare > Panels: one Axes per panel
        self._spectrum_panel_groups: list = []   # ... and the spectra (indices) it holds
        self._spectrum_panel_link = None
        self._compare_panels_kwargs = None
        #: Panel labels (a), (b)... : the ``add_panel_labels`` arguments last
        #: asked for (None: none), and a drag of one in progress.
        self._panel_label_args: dict | None = None
        self._label_drag: dict | None = None
        self._panel_home_limits: list = []
        self._panel_spectra: list = []
        #: Zoomed insets of the single 2D contour on show (Plot Style ▸ 2D ▸
        #: Inset…): :func:`opennmr.plotting.twod.draw_insets` specs, in the
        #: displayed axis units.
        self._contour_insets: list = []
        self._panels_linked = False
        self._panels_link_y = False
        self._panels_busy = False
        self._overlay_labels: list | None = None
        # The DOSY map currently shown (processing.ilt.dosy_map's own
        # output) and its contour floor, mirroring _contour_spectrum/
        # _contour_factor. Set only while a DOSY is the current view,
        # so the scroll and right-click machinery can tell "this is a
        # DOSY" apart from every other view without inspecting Axes.
        self._dosy_result: dict | None = None
        #: Every map when several are overlaid (``None`` for the
        #: ordinary single-map view) -- see :meth:`show_dosy_overlay`.
        self._dosy_overlay: list[dict] | None = None
        self._dosy_overlay_labels: list[str] | None = None
        self._dosy_factor = self._DOSY_FACTOR_DEFAULT
        self._dosy_title = ""
        #: How many contour levels to draw, and the D axis range to
        #: draw them in. ``None`` for the range means "fit the axis to
        #: the populated part of the grid" (``plot_dosy``'s own
        #: default); an explicit ``(low, high)`` is the user's own
        #: choice and is used as given.
        self._dosy_clevels = self._DOSY_CLEVELS_DEFAULT
        self._dosy_d_lims: tuple[float, float] | None = None
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
        # Pixel position of a left-press while a "Stacked" Compare view
        # is shown, resolved to a row on release (see _on_press/
        # _compare_row_index_at_y). Separate from _pan_start because a
        # press in a row's *margins* -- its tick strip, or the gap to
        # the next row -- has no `inaxes` at all and so never starts a
        # pan, but should still select the row it visually belongs to.
        self._compare_click_start: tuple[float, float] | None = None
        # Label texts reported by the last _apply_layout as still cut
        # off after shrinking -- kept so labelsClipped only fires on a
        # change of state rather than on every relayout.
        self._last_clipped_labels: list[str] = []
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

        # Re-fit the labels whenever Qt resizes the canvas -- see
        # _on_canvas_resized. The timer is what keeps a window drag
        # from paying for a layout pass per pixel.
        self._resize_layout_timer = QTimer(self)
        self._resize_layout_timer.setSingleShot(True)
        self._resize_layout_timer.timeout.connect(self._run_debounced_resize_layout)
        self.canvas.mpl_connect("resize_event", self._on_canvas_resized)
        self.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.canvas.mpl_connect("figure_leave_event", self._on_figure_leave)
        self.canvas.mpl_connect("button_release_event", self._on_release)
        self.canvas.mpl_connect("draw_event", self._on_draw_apply_insets)
        # Deliberately no matplotlib key_press_event binding for the
        # measurement toggle. It used to be a bare "d", which fired on
        # any keystroke over the canvas -- including the "d" of anything
        # typed nearby -- and was not the Cmd+D that was asked for:
        # matplotlib's own key names for the macOS Command key are
        # backend-dependent, whereas Qt's QKeySequence("Ctrl+D") maps to
        # Cmd+D on macOS and Ctrl+D elsewhere for free. It is therefore
        # a menu-action shortcut in MainWindow instead. Binding it in
        # both places would toggle twice per press and cancel itself out.

        self.canvas.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.canvas.customContextMenuRequested.connect(self._on_context_menu)

    def set_pick_mode(self, enabled: bool) -> None:
        """Toggle click-to-pick: while on, a plain left-click emits
        :attr:`peakPicked` with the clicked x-data value instead of
        starting a pan drag.

        One interaction mode at a time (explicit user request): turning
        this on releases Measure, Select Region and the toolbar's
        pan/zoom -- with pan still on, the toolbar consumed every click
        and picking silently did nothing."""
        enabled = bool(enabled)
        changed = enabled != self._pick_mode
        if enabled:
            self._release_other_modes(keep="pick")
        self._pick_mode = enabled
        self._refresh_cursor()
        if changed:
            self.pickModeChanged.emit(enabled)

    def set_region_select_mode(self, enabled: bool) -> None:
        """Toggle drag-to-select-a-region: while on, a left-click-drag
        emits :attr:`regionSelected` with the dragged (lo, hi) x-range
        on release, instead of starting a pan drag -- the range
        equivalent of :func:`set_pick_mode`'s single click. Releases the
        other interaction modes, as :meth:`set_pick_mode` does."""
        enabled = bool(enabled)
        changed = enabled != self._region_select_mode
        if enabled:
            self._release_other_modes(keep="region")
        self._region_select_mode = enabled
        if not enabled:
            self._clear_region_select_patch()
            self._region_select_start = None
            self._region_select_ax = None
        # *After* the half-finished drag is dropped, not before:
        # _refresh_cursor deliberately leaves the shape alone while a
        # drag is in flight, so refreshing first would be a no-op and
        # the cross would stay behind after the mode was switched off.
        self._refresh_cursor()
        if changed:
            self.regionSelectModeChanged.emit(enabled)

    def _release_other_modes(self, keep: str) -> None:
        """Switch off every interaction mode except ``keep`` -- one of
        ``"pick"``, ``"region"``, ``"measure"``, ``"nav"`` (the toolbar's
        pan/zoom). Only the most recently chosen mode stays active
        (explicit user request), so a click always means one thing.

        The Crosshair is a display overlay, not a click mode, and is
        left alone here; it is exclusive with Measure only (see
        :meth:`set_measurement_mode` / :meth:`set_crosshair_enabled`)."""
        if keep != "pick" and self._pick_mode:
            self.set_pick_mode(False)
        if keep != "region" and self._region_select_mode:
            self.set_region_select_mode(False)
        if keep != "measure" and self._measurement_mode:
            self.set_measurement_mode(False)
        if keep != "width" and getattr(self, "_width_mode", False):
            self.set_width_mode(False)
        if keep != "inset" and getattr(self, "_inset_mode", False):
            self.set_inset_mode(False)
        if keep != "nav" and self.toolbar.mode:
            # The base-class toggles, not _Toolbar's overrides: this is
            # the canvas releasing the tool, not the user choosing one.
            if self.toolbar.mode.name == "PAN":
                NavigationToolbar2QT.pan(self.toolbar)
            elif self.toolbar.mode.name == "ZOOM":
                NavigationToolbar2QT.zoom(self.toolbar)

    def _on_navigation_mode_changed(self) -> None:
        """The user picked the toolbar's pan or zoom tool: release the
        click modes, which that tool would otherwise leave armed but
        unreachable."""
        if self.toolbar.mode:
            self._release_other_modes(keep="nav")
        self._refresh_cursor()

    def set_scale_button_visible(self, visible: bool) -> None:
        """Show the × Scale button (only the canvas that shows a session spectrum has it)."""
        self._scale_button_action.setVisible(bool(visible))

    def _add_toolbar_toggle(self, text: str, tooltip: str, slot, checked: bool) -> QToolButton:
        """A checkable text-only button on this canvas's own matplotlib
        toolbar -- see the two call sites in ``__init__`` for why these
        are QToolButtons rather than QActions."""
        button = QToolButton(self.toolbar)
        button.setText(text)
        button.setToolTip(tooltip)
        button.setCheckable(True)
        button.setChecked(checked)
        button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        button.toggled.connect(slot)
        self.toolbar.addWidget(button)
        return button

    def set_crosshair_enabled(self, enabled: bool) -> None:
        """Turn the cursor crosshair on or off for this canvas.

        Turning it off removes the live instance outright (rather than
        just hiding its lines) so nothing is left connected to the
        canvas's motion/draw events, and nothing is left in
        ``ax.lines`` for the export/unit-switch paths that walk it."""
        enabled = bool(enabled)
        changed = enabled != self._crosshair_enabled
        if enabled and self._measurement_mode:
            # Crosshair and Measure are exclusive (explicit user
            # request): the most recently chosen one wins.
            self.set_measurement_mode(False)
        if enabled and getattr(self, "_width_mode", False):
            self.set_width_mode(False)
        if enabled and getattr(self, "_inset_mode", False):
            self.set_inset_mode(False)
        self._crosshair_enabled = enabled
        self._sync_toggle(getattr(self, "_crosshair_button", None), enabled)
        if not enabled:
            self._remove_crosshair()
            self.canvas.draw_idle()
        if changed:
            self.crosshairEnabledChanged.emit(enabled)

    def crosshair_style(self) -> tuple[str, float]:
        """``(colour, line width in points)`` of the crosshair."""
        return self._crosshair_color, self._crosshair_width

    def set_crosshair_style(self, color=None, linewidth=None) -> None:
        """Change the crosshair's colour and/or line width. ``None``
        leaves that one as it is. Any matplotlib colour is accepted
        (name, ``#rrggbb``, grey level as a string); a colour matplotlib
        cannot read, or a width outside :attr:`CROSSHAIR_WIDTH_RANGE`,
        is ignored rather than raised, so a stale saved setting can
        never stop a plot from drawing.

        A live crosshair is dropped, not edited: it is recreated, with
        the new style, by the next mouse move (see
        :meth:`_ensure_crosshair`), which is also how it survives every
        figure rebuild."""
        if color is not None:
            color = str(color)
            if matplotlib.colors.is_color_like(color):
                self._crosshair_color = color
        if linewidth is not None:
            try:
                width = float(linewidth)
            except (TypeError, ValueError):
                width = float("nan")
            lo, hi = self.CROSSHAIR_WIDTH_RANGE
            if lo <= width <= hi:
                self._crosshair_width = width
        if self._crosshair is not None:
            self._remove_crosshair()
            self.canvas.draw_idle()

    def zoom_box_width(self) -> float:
        """Line width, in screen pixels, of the dashed zoom box."""
        return self._zoom_box_width

    def set_zoom_box_width(self, width) -> bool:
        """Set the zoom box's line width (pixels). A value that is not a
        number or lies outside :attr:`ZOOM_BOX_WIDTH_RANGE` is ignored
        (a stale saved setting must never break a drag); returns whether
        it was accepted."""
        try:
            value = float(width)
        except (TypeError, ValueError):
            return False
        lo, hi = self.ZOOM_BOX_WIDTH_RANGE
        if not lo <= value <= hi:
            return False
        self._zoom_box_width = value
        return True

    def _draw_zoom_rectangle(self, rect) -> None:
        """``FigureCanvasQT.drawRectangle`` with an adjustable line width.

        Same drawing as matplotlib's (black and white dashes, so it shows
        on any background; drawn from the start corner so the dashes do
        not jump as the box grows), but the pen is
        :attr:`_zoom_box_width` pixels wide instead of 1. A Qt dash
        pattern is measured in pen widths, so the dashes stay in
        proportion at any thickness."""
        from PySide6.QtGui import QColor, QPen
        qt_canvas = self.canvas
        if rect is None:
            qt_canvas._draw_rect_callback = lambda painter: None
            qt_canvas.update()
            return
        ratio = qt_canvas.device_pixel_ratio
        x0, y0, w, h = (int(pt / ratio) for pt in rect)
        x1, y1 = x0 + w, y0 + h
        width = self._zoom_box_width

        def draw(painter) -> None:
            pen = QPen(QColor("black"), width)
            pen.setDashPattern([3, 3])
            for colour, offset in (("black", 0), ("white", 3)):
                pen.setDashOffset(offset)
                pen.setColor(QColor(colour))
                painter.setPen(pen)
                painter.drawLine(x0, y0, x0, y1)
                painter.drawLine(x0, y0, x1, y0)
                painter.drawLine(x0, y1, x1, y1)
                painter.drawLine(x1, y0, x1, y1)

        qt_canvas._draw_rect_callback = draw
        qt_canvas.update()

    def reset_crosshair_style(self) -> None:
        self.set_crosshair_style(self.DEFAULT_CROSSHAIR_COLOR, self.DEFAULT_CROSSHAIR_WIDTH)

    def _on_crosshair_button_menu(self, pos) -> None:
        """Right-click on the Crosshair button: colour, line width,
        reset. The choice is emitted (not applied here) so that every
        plot changes together and it is remembered."""
        menu = QMenu(self._crosshair_button)
        colour_action = menu.addAction("Colour…")
        width_action = menu.addAction("Line width…")
        menu.addSeparator()
        reset_action = menu.addAction("Reset to default")
        chosen = menu.exec(self._crosshair_button.mapToGlobal(pos))
        color, width = self.crosshair_style()
        if chosen is colour_action:
            from PySide6.QtGui import QColor
            picked = QColorDialog.getColor(
                QColor(matplotlib.colors.to_hex(color)), self, "Crosshair colour")
            if picked.isValid():
                color = picked.name()
        elif chosen is width_action:
            lo, hi = self.CROSSHAIR_WIDTH_RANGE
            value, ok = QInputDialog.getDouble(
                self, "Crosshair line width", "Line width (points):", width, lo, hi, 2)
            if ok:
                width = value
        elif chosen is reset_action:
            color, width = self.DEFAULT_CROSSHAIR_COLOR, self.DEFAULT_CROSSHAIR_WIDTH
        else:
            return
        self.set_crosshair_style(color, width)
        self.crosshairStyleChanged.emit(*self.crosshair_style())

    def measurement_style(self) -> tuple[str, float, float]:
        """``(colour, line width in points, text size in points)`` of
        Measure's arrows, markers and labels."""
        return self._measurement_color, self._measurement_width, self._measurement_fontsize

    def set_measurement_style(self, color=None, linewidth=None, fontsize=None) -> None:
        """Change Measure's colour, line width and/or text size; ``None``
        leaves that one as it is. As :meth:`set_crosshair_style`: a colour
        matplotlib cannot read, or a value outside its range, is ignored,
        so a stale saved setting never stops a plot drawing. Measurements
        already on the plot are restyled at once."""
        if color is not None:
            color = str(color)
            if matplotlib.colors.is_color_like(color):
                self._measurement_color = color
        for value, attr, (lo, hi) in ((linewidth, "_measurement_width", self.MEASUREMENT_WIDTH_RANGE),
                                      (fontsize, "_measurement_fontsize",
                                       self.MEASUREMENT_FONTSIZE_RANGE)):
            if value is None:
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if lo <= number <= hi:
                setattr(self, attr, number)
        self._restyle_measurements()

    def _measure_marker_size(self) -> float:
        """The first click's "+" grows with the line width."""
        return 11.0 * max(1.0, self._measurement_width / self.DEFAULT_MEASUREMENT_WIDTH)

    def _restyle_measurements(self) -> None:
        """Measure's own artists (not Width's, which share the list) in
        the current style."""
        colour, width, size = self.measurement_style()
        changed = False
        for artist in self._measurement_artists:
            if not getattr(artist, "_opennmr_measure", False):
                continue
            changed = True
            if isinstance(artist, matplotlib.text.Annotation) and artist.arrow_patch is not None:
                artist.arrow_patch.set_color(colour)
                artist.arrow_patch.set_linewidth(width)
                artist.arrow_patch.set_mutation_scale(self._measure_head_scale())
            elif isinstance(artist, matplotlib.text.Text):
                artist.set_color(colour)
                artist.set_fontsize(size)
                patch = artist.get_bbox_patch()
                if patch is not None:
                    patch.set_edgecolor(colour)
            elif isinstance(artist, matplotlib.lines.Line2D):
                artist.set_color(colour)
                artist.set_markeredgewidth(width + 0.2)
                artist.set_markersize(self._measure_marker_size())
            elif isinstance(artist, matplotlib.patches.FancyArrowPatch):
                artist.set_color(colour)
                artist.set_linewidth(width)
                artist.set_mutation_scale(self._measure_head_scale())
        if changed:
            self.canvas.draw_idle()

    def _measure_head_scale(self) -> float:
        """Arrow-head size, in proportion to the line (matplotlib's own
        default, 10, at the default width)."""
        return 10.0 * max(1.0, self._measurement_width / self.DEFAULT_MEASUREMENT_WIDTH)

    def _on_measure_button_menu(self, pos) -> None:
        """Right-click on the Measure button: colour, line width, text
        size, reset. Emitted (not applied here) so every plot changes
        together and the choice is remembered -- as for the Crosshair."""
        menu = QMenu(self._measure_button)
        colour_action = menu.addAction("Colour…")
        width_action = menu.addAction("Line width…")
        size_action = menu.addAction("Text size…")
        menu.addSeparator()
        reset_action = menu.addAction("Reset to default")
        chosen = menu.exec(self._measure_button.mapToGlobal(pos))
        style = self._measure_style_from_menu(chosen, colour_action, width_action, size_action,
                                              reset_action)
        if style is not None:
            self._apply_measure_menu_style(style)

    def _apply_measure_menu_style(self, style) -> None:
        """Apply a style chosen from the button menu here, and announce it
        so the window applies it to every plot and remembers it."""
        self.set_measurement_style(*style)
        self.measurementStyleChanged.emit(*self.measurement_style())

    def _measure_style_from_menu(self, chosen, colour_action, width_action, size_action,
                                 reset_action):
        """The style a menu choice asks for, or None (a seam for tests:
        the dialogs it opens are modal)."""
        colour, width, size = self.measurement_style()
        if chosen is colour_action:
            from PySide6.QtGui import QColor
            picked = QColorDialog.getColor(
                QColor(matplotlib.colors.to_hex(colour)), self, "Measure colour")
            if not picked.isValid():
                return None
            colour = picked.name()
        elif chosen is width_action:
            lo, hi = self.MEASUREMENT_WIDTH_RANGE
            value, ok = QInputDialog.getDouble(
                self, "Measure line width", "Line width (points):", width, lo, hi, 2)
            if not ok:
                return None
            width = value
        elif chosen is size_action:
            lo, hi = self.MEASUREMENT_FONTSIZE_RANGE
            value, ok = QInputDialog.getDouble(
                self, "Measure text size", "Text size (points):", size, lo, hi, 1)
            if not ok:
                return None
            size = value
        elif chosen is reset_action:
            colour, width, size = (self.DEFAULT_MEASUREMENT_COLOR, self.DEFAULT_MEASUREMENT_WIDTH,
                                   self.DEFAULT_MEASUREMENT_FONTSIZE)
        else:
            return None
        return colour, width, size

    def set_measurement_mode(self, enabled: bool) -> None:
        """Toggle measurement mode: while on, clicking two points on
        the plot draws the separation between them, labelled in every
        unit that applies, and emits :attr:`measurementCompleted`.

        Switching it off clears both a half-finished measurement and
        every completed one still drawn -- they are a transient
        annotation on the view, not part of the plot."""
        enabled = bool(enabled)
        changed = enabled != self._measurement_mode
        if enabled:
            self._release_other_modes(keep="measure")
            if self._crosshair_enabled:
                self.set_crosshair_enabled(False)
        self._measurement_mode = enabled
        self._sync_toggle(getattr(self, "_measure_button", None), enabled)
        self._refresh_cursor()
        if not enabled:
            self.clear_measurements()
        if changed:
            self.measurementModeChanged.emit(enabled)

    def set_width_mode(self, enabled: bool) -> None:
        """Toggle Width mode: while on, a click on a peak measures its
        FWHM (:meth:`_measure_width_at`). Releases the other click modes
        and the Crosshair, as Measure does; switching it off clears the
        width annotations."""
        enabled = bool(enabled)
        changed = enabled != self._width_mode
        if enabled:
            self._release_other_modes(keep="width")
            if self._crosshair_enabled:
                self.set_crosshair_enabled(False)
        self._width_mode = enabled
        self._sync_toggle(getattr(self, "_width_button", None), enabled)
        self._refresh_cursor()
        if not enabled:
            self.clear_measurements()
        if changed:
            self.widthModeChanged.emit(enabled)

    def set_inset_mode(self, enabled: bool) -> None:
        """Toggle Inset mode: while on, a left-drag draws a box over the
        part of the plot to magnify, and on release an inset of it is added
        (:meth:`add_inset_region`) and the mode switches itself off.
        Releases the other click modes and the Crosshair, as Width does."""
        enabled = bool(enabled)
        changed = enabled != self._inset_mode
        if enabled:
            self._release_other_modes(keep="inset")
            if self._crosshair_enabled:
                self.set_crosshair_enabled(False)
        self._inset_mode = enabled
        if not enabled:
            self._inset_start = None
            try:
                self.toolbar.remove_rubberband()
            except Exception:  # noqa: BLE001 -- a stale rubber band is harmless
                pass
        self._sync_toggle(getattr(self, "_inset_button", None), enabled)
        self._refresh_cursor()
        if changed:
            self.insetModeChanged.emit(enabled)

    def set_view_switcher(self, entries: list[tuple[str, str]], current: str | None = None) -> None:
        """Populate the "Showing:" switcher with ``(key, label)`` pairs
        and select ``current``.

        ``entries`` of fewer than two hides the control outright: with
        one result there is nothing to switch between, and a combo box
        holding a single fixed item reads as a control that has stopped
        working. Passing ``[]`` (no results at all) hides it too.

        The canvas stores the keys only to hand one back through
        :attr:`viewSwitchRequested`; it never interprets them.
        """
        entries = list(entries or [])
        self._view_switcher.blockSignals(True)
        try:
            self._view_switcher.clear()
            for key, label in entries:
                self._view_switcher.addItem(label, key)
            if current is not None:
                index = self._view_switcher.findData(current)
                if index >= 0:
                    self._view_switcher.setCurrentIndex(index)
        finally:
            self._view_switcher.blockSignals(False)
        visible = len(entries) > 1
        self._view_switcher_action.setVisible(visible)
        self._view_switcher_label_action.setVisible(visible)

    def current_view_key(self) -> str | None:
        """The key of the entry the switcher is showing, or ``None``
        when it holds nothing."""
        data = self._view_switcher.currentData()
        return str(data) if data is not None else None

    def _on_view_switcher_activated(self, _index: int) -> None:
        """``activated``, not ``currentIndexChanged``: the latter also
        fires when :meth:`set_view_switcher` repopulates the list,
        which would have every refresh ask the owner to redraw the
        thing it had just finished drawing."""
        key = self.current_view_key()
        if key is not None:
            self.viewSwitchRequested.emit(key)

    @staticmethod
    def _sync_toggle(button, checked: bool) -> None:
        """Push state onto a checkable button without its own
        ``toggled`` signal calling straight back into the setter that
        is already running."""
        if button is None or button.isChecked() == checked:
            return
        blocked = button.blockSignals(True)
        try:
            button.setChecked(checked)
        finally:
            button.blockSignals(blocked)

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

    def set_integral_labels(self, items, **style) -> None:
        """Write the integral of each region under its peak on the Spectrum
        plot (``items``: ``[(x, text), ...]``, from
        :mod:`opennmr.notebook.integral_labels`; ``None`` or empty removes
        them). The items are kept, so a redraw of the plot puts them back.

        ``style``: how the text looks -- ``fontsize`` (None: the tick-label
        size), ``color``, ``family``, ``bold``, ``italic``, ``rotation`` and
        ``offset_axes``, as :func:`add_integral_labels` takes them."""
        from ...notebook import peak_integral_labels as _labels
        self._integral_label_items = list(items or [])
        self._integral_label_style = dict(style)
        ax = self.ax
        if ax is None or ax not in self.figure.axes:
            return
        _labels.remove_integral_labels(ax)
        if self._integral_label_items:
            options = dict(self._integral_label_style)
            if not options.get("fontsize"):
                options["fontsize"] = None
                ticks = ax.xaxis.get_ticklabels()
                if ticks:
                    options["fontsize"] = ticks[0].get_fontsize()
            _labels.add_integral_labels(ax, self._integral_label_items, **options)
        self.canvas.draw_idle()

    def set_highlight_regions(self, regions, key: str = "highlight") -> None:
        """Shade named x ranges on whatever this canvas shows now -- every
        row of a Compare stack, or the one plot -- **without redrawing
        it**: the view, the plot style and the traces stay exactly as they
        are (explicit user report, 2026-10-16: CP Build-up's Drag on Plot
        redrew a plain figure, losing the styling and the zoom).

        ``regions``: ``[(name, lo, hi), ...]``; ``key`` tags this set, so a
        new call replaces it and leaves other shading alone. Each region
        has its own colour and its name along the top of the plot."""
        from ...plotting.accessibility import ordered_qualitative_colours
        gid = f"highlight:{key}"
        for ax in self.figure.axes:
            for artist in list(ax.patches) + list(ax.texts):
                if artist.get_gid() == gid:
                    artist.remove()
        axes = [a for a in (self._compare_axes or []) if a in self.figure.axes]
        if not axes and self.ax is not None and self.ax in self.figure.axes:
            axes = [self.ax]
        if not axes:
            return
        colours = ordered_qualitative_colours("okabe_ito")
        import matplotlib.transforms as mtransforms
        for k, (name, lo, hi) in enumerate(regions):
            colour = colours[k % len(colours)]
            for ax in axes:
                span = ax.axvspan(lo, hi, color=colour, alpha=0.2, lw=0, zorder=0)
                span.set_gid(gid)
            top = axes[0]
            label = top.text(0.5 * (lo + hi), 1.0, str(name), color=colour, fontsize=9,
                             ha="center", va="bottom", clip_on=False,
                             transform=mtransforms.blended_transform_factory(
                                 top.transData, top.transAxes))
            label.set_gid(gid)
        self.canvas.draw_idle()

    def has_spectrum_plot(self) -> bool:
        """Whether a spectrum (1D, or Compare rows) is on show here."""
        return bool(self._compare_axes or self._current_1d_spectrum is not None
                    or (self._compare_spectra and not getattr(self, "_external_plot", None)
                        and self._contour_spectrum is None))

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
        """Handle a scroll, then re-assert the cursor shape.

        **The cursor has to be re-applied after every scroll**, which
        is what :meth:`_dispatch_scroll` on its own does not do:

        * A contour- or DOSY-level scroll rebuilds the figure through
          ``_clear_figure``, which deliberately drops the cached shape
          (the bboxes it was resolved against are gone). Nothing then
          puts one back -- the pointer has not moved, so no
          ``motion_notify_event`` follows -- and the canvas sat on
          whatever the platform left until the user moved the mouse.
        * Every other scroll repaints too, and on a real contour that
          repaint is long enough for the platform's own busy cursor to
          cover ours. Re-asserting once the work is done is what brings
          it back without waiting for a mouse move.

        Reported directly: "the scroll cursor... only briefly appears
        again, and also it should remain while scrolling too if
        possible."

        In a ``finally`` so it covers every one of the dispatch's own
        early returns -- there are eleven, and one forgotten branch
        would be a cursor that sticks only on certain regions, which is
        exactly the symptom this is fixing.
        """
        self.note_natural_view()
        try:
            self._dispatch_scroll(event)
        finally:
            self._refresh_cursor(event)

    def _dispatch_scroll(self, event) -> None:
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
        # A 3D picture (Simulate's tensor views) rotates on drag and has
        # no axis strips; a scroll there means nothing here.
        if event.inaxes is not None and getattr(event.inaxes, "name", "") == "3d":
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

        # Which tick strip -- if any -- is under the cursor. Shared
        # with the cursor-shape resolver rather than hit-tested twice;
        # see :meth:`_scroll_strip_at`.
        strip = self._scroll_strip_at(event)
        on_x_strip = strip is not None and strip[0] == "x"
        on_y_strip = strip is not None and strip[0] == "y"
        strip_ax = strip[1] if strip is not None else self.ax
        if on_x_strip or (on_y_strip and self._panel_axes):
            ax = strip_ax
        else:
            ax = event.inaxes if event.inaxes is not None else (self.ax if on_y_strip else None)
        # A stacked Compare is the one view with no "nowhere": its rows
        # are thin and the gaps between them are not, so a scroll aimed
        # at a gap -- where there is no inaxes to resolve -- has an
        # obvious meaning (scale the stack) and used to have none. The
        # compare branch below reads no `ax`, so letting None through
        # to it is safe; every other branch tests `ax is <something>`
        # or `ax in <set>`, both False for None, and the bare
        # `ax.get_ylim()` fallback at the end is unreachable while
        # _compare_axes is set because that branch always returns.
        if ax is None and not self._compare_axes:
            return

        factor = 1 / self.ZOOM_FACTOR if event.button == "up" else self.ZOOM_FACTOR

        if self._panel_axes and not on_x_strip and not on_y_strip and ax in self._panel_axes:
            # The body of a Compare 2D panel has no intensity axis to rescale
            # (it used to squash F1); its two axes are zoomed from their strips.
            # What a scroll over the contours does is what it does on one 2D:
            # raise or lower the lowest contour. The window owns that setting
            # (Compare > 2D spectra > Lowest contour), so it is asked.
            if self._panel_spectra and all(getattr(s, "ndim", 0) == 2 for s in self._panel_spectra):
                self.compareContourScrolled.emit(float(self._contour_scroll_step(event)))
            return

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
            if self._dosy_result is not None and ax.get_yscale() == "log":
                # A log axis has to be zoomed in log space, or the
                # cursor-centred arithmetic below (which is linear)
                # walks the view off toward zero: half of a decade is
                # not half of the distance in data units.
                if y > 0:
                    low, high = ax.get_ylim()
                    log_y = np.log10(y)
                    log_low, log_high = np.log10(low), np.log10(high)
                    ax.set_ylim(
                        10.0 ** (log_y - (log_y - log_low) * factor),
                        10.0 ** (log_y + (log_high - log_y) * factor),
                    )
                    self.canvas.draw_idle()
                return
            ymin, ymax = ax.get_ylim()
            ax.set_ylim(y - (y - ymin) * factor, y + (ymax - y) * factor)
            self.canvas.draw_idle()
            return

        if self._dosy_result is not None and ax is self.ax:
            self._adjust_dosy_levels(event)
            return

        if self._contour_factor is not None and ax is self.ax:
            self._adjust_contour_levels(event)
            return

        if self._compare_axes:
            # Deliberately **above** the `event.ydata is None` guard
            # below, and reached whether or not the cursor is inside a
            # row's own data rectangle. Every row is rescaled from its
            # own ylim by `factor` -- no cursor y value is read at all
            # -- so the gaps between rows (where matplotlib reports no
            # ydata) can scale the stack too. They used to fall into
            # that guard and do nothing, which on a tall stack is most
            # of the places you would naturally aim: rows are thin, the
            # inter-row gaps are not, and half of "nothing happened"
            # was really "you hit the row above's x strip and zoomed x
            # instead" (see _x_strip_depth).
            #
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
                self._zoom_intensity_about_baseline(row_ax, factor)
            self.canvas.draw_idle()
            return

        if event.ydata is None:
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

        if self._plot_kind in self._BASELINE_ZOOM_KINDS:
            self._zoom_intensity_about_baseline(ax, factor)
        else:
            ymin, ymax = ax.get_ylim()
            ax.set_ylim(ymin, ymin + (ymax - ymin) * factor)
        self.canvas.draw_idle()

    #: Plot kinds whose body scroll scales intensity about the spectrum's
    #: own baseline (see _zoom_intensity_about_baseline) -- spectra. A
    #: curve (relaxation, DNP profile) keeps the pinned-bottom zoom.
    _BASELINE_ZOOM_KINDS = ("1d", "compare", "pseudo2d")

    #: Fraction of the visible points averaged at each end of the x-range
    #: to read the baseline there (at least a few points): one point
    #: would put the anchor wherever that point's noise happened to be.
    _BASELINE_END_FRACTION = 0.01

    @classmethod
    def baseline_anchor(cls, ax):
        """The y value a spectrum's intensity zoom keeps still: the
        midpoint of the spectrum's level at the two ends of the visible
        x-range -- explicit user request (2026-09-28): "find y at max x,
        find y at min x, then find the midpoint of these two y values and
        scroll relative to this". Each end is the median of the last 1 %
        of visible points there (at least 5), so noise does not move it.
        Averaged over every data line on ``ax`` (an overlay). ``None`` if
        the Axes has no line to read."""
        xlo, xhi = sorted(ax.get_xlim())
        anchors = []
        for line in ax.get_lines():
            if line.get_gid() in (cls._BASELINE_PREVIEW_GID, cls._BASELINE_GHOST_GID):
                continue
            label = line.get_label() or ""
            if label.startswith("_baseline"):
                continue
            x = np.asarray(line.get_xdata(), dtype=float)
            y = np.asarray(line.get_ydata(), dtype=float)
            if x.ndim != 1 or x.shape != y.shape or x.size < 10:
                continue
            visible = (x >= xlo) & (x <= xhi) & np.isfinite(y)
            if np.count_nonzero(visible) < 10:
                continue
            xv, yv = x[visible], y[visible]
            order = np.argsort(xv)
            yv = yv[order]
            n = max(5, int(round(yv.size * cls._BASELINE_END_FRACTION)))
            anchors.append(0.5 * (float(np.median(yv[:n])) + float(np.median(yv[-n:]))))
        return float(np.mean(anchors)) if anchors else None

    def _zoom_intensity_about_baseline(self, ax, factor: float) -> None:
        """Scale ``ax``'s intensity by ``factor`` keeping the spectrum's
        baseline where it is on screen, so zooming makes peaks taller or
        shorter without the baseline creeping up or down. Pinning the
        bottom of the view (what this replaced) only kept the baseline
        still when it sat exactly on that edge -- a baseline at 1 with the
        view starting at 0 rose on every scroll (user report). Falls back
        to the pinned bottom if no line can be read."""
        tagged = [ln for ln in ax.lines if getattr(ln, "_opennmr_baseline", None) is not None]
        if tagged:
            # Stacked traces (anisotropic-motion plots) share one axes, so rescaling
            # the y-limits would also change the spacing between the spectra (user
            # report). Scale each trace about its own baseline instead.
            g = 1.0 / float(factor)
            for ln in tagged:
                b = float(ln._opennmr_baseline)
                ln.set_ydata(b + (np.asarray(ln.get_ydata(), dtype=float) - b) * g)
            return
        ymin, ymax = ax.get_ylim()
        anchor = self.baseline_anchor(ax)
        if anchor is None or not np.isfinite(anchor):
            ax.set_ylim(ymin, ymin + (ymax - ymin) * factor)
            return
        ax.set_ylim(anchor - (anchor - ymin) * factor, anchor + (ymax - anchor) * factor)

    #: Cursor shown over each kind of scroll-zoomable tick strip -- a
    #: double-headed arrow along the axis the scroll would move.
    #: Explicit user request: "context cursors like some double-headed
    #: arrow where it is possible to scroll to zoom axes".
    #:
    #: **Strips only, never a plot body.** Scrolling a body does
    #: something too (intensity on a 1D/projection/Compare row, the
    #: contour or DOSY threshold on a map), but the crosshair is what
    #: belongs over data you are reading values off: "I would keep the
    #: crosshair as the main cursor on the canvas except for on the
    #: scrollable axes... While the projections, spectra, and contours
    #: (inc DOSY contours) are scrollable, these should stay as a
    #: (toggleable) crosshair."
    _STRIP_CURSORS = {
        "x": Qt.CursorShape.SizeHorCursor,
        "y": Qt.CursorShape.SizeVerCursor,
    }

    def _scroll_strip_at(self, event):
        """Which tick-label *strip* the cursor is over, as ``("x", ax)``
        or ``("y", ax)`` -- or ``None`` for anywhere else (a plot body,
        a dead margin).

        **The single hit-test behind both the scroll zoom and the
        cursor shape.** :meth:`_on_scroll` uses it to decide what a
        scroll does; :meth:`_refresh_cursor` uses it to decide what the
        cursor looks like. Written once because two copies of an
        eleven-branch region test would drift apart the first time a
        view was added -- the cursor would then promise a zoom that
        does not happen, or hide one that does.

        Reads only ``x``/``y``/``inaxes`` -- **never ``event.button``**,
        which means ``"up"``/``"down"`` on a scroll event but ``None``
        or an int on the motion event this also serves.

        The first two guards mirror early returns :meth:`_on_scroll`
        makes before it ever asks: the extracted-slice panel (a
        deliberate no-op) and the F1/F2 projection strips (which zoom
        their own intensity). ``_on_scroll`` still has to act on those
        itself, so those two ``if``s exist in both places -- but this
        one is what keeps a *cursor* off them, and keeping the
        precedence self-contained is what lets it be asked in isolation
        from a mouse move.
        """
        if event is None or event.x is None or event.y is None:
            return None
        if self._proj_axes.get("slices") is not None and event.inaxes is self._proj_axes["slices"]:
            return None
        for kind in ("f1", "f2"):
            proj_ax = self._proj_axes.get(kind)
            if proj_ax is not None and event.inaxes is proj_ax:
                return None

        # Whose x-tick strip is under the cursor, not just self.ax's.
        #
        # self.ax is the *last* Axes a multi-panel view created -- the
        # spectrum panel of the Processing tab's dual preview, the
        # bottom row of a stack. Checking only it meant the FID panel
        # on top could never have its own strip hovered, so its x axis
        # (time) could not be scroll-zoomed at all, however precisely
        # you aimed at its "Time (s)" labels. The panel does have a
        # real strip -- the dual preview is a plain subplots(2, 1) with
        # its own labels on both panels -- it was simply never
        # consulted.
        #
        # Only searched when the cursor is in a margin (`inaxes is
        # None`): inside a data rectangle the intensity zoom owns the
        # gesture, and one panel's strip region can overlap another
        # panel's plot area in a tight layout.
        # The *nearest* panel above the cursor, not the first match: in
        # a stack, a point below the bottom panel is also below every
        # panel over it, so taking the first hit zoomed the top panel
        # whenever the bottom one's own strip was aimed at.
        strip_ax = self.ax
        if event.inaxes is None:
            candidates = [ax for ax in self.figure.axes if self._hovering_x_axis(ax, event)]
            if candidates:
                # Ties broken toward self.ax -- two panels can share a
                # bottom edge (a full-figure Axes left under a stacked
                # layout, a twin), and picking whichever happened to be
                # created first would zoom an arbitrary one of them.
                strip_ax = min(
                    candidates,
                    key=lambda ax: (ax.get_window_extent().y0 - event.y, ax is not self.ax),
                )
        if strip_ax is not None and self._hovering_x_axis(strip_ax, event):
            return ("x", strip_ax)

        # A DOSY's y axis (D) is a *position* axis like a contour's F1,
        # not an intensity one, so it gets the same right-hand strip
        # zoom rather than the 1D pin-bottom intensity behaviour. A
        # plain 1D shows no y ticks at all, so it has no y strip and
        # falls through here to None -- which is correct both ways:
        # nothing to zoom, and so no arrow promising one.
        # Compare's 2D panels (and an overlay) have F1 strips of their own:
        # the one on the right of whichever panel the cursor is beside. Checked
        # before the single contour's, whose Axes would otherwise claim
        # everything to the right of the first panel.
        if self._panel_axes:
            if event.inaxes is not None:
                return None
            beside = [ax for ax in self._panel_axes if self._hovering_y_axis(ax, event)]
            if beside:
                return ("y", max(beside, key=lambda ax: ax.get_window_extent().x1))
            return None
        has_levels = self._contour_factor is not None or self._dosy_result is not None
        if has_levels and self.ax is not None and self._hovering_y_axis(self.ax, event):
            return ("y", self.ax)
        return None

    def _refresh_cursor(self, event=None) -> None:
        """The one place this canvas's cursor shape is decided.

        Four writers used to set it independently (pick, region-select
        and measurement mode, plus matplotlib's own toolbar), which is
        survivable while each is a simple on/off -- but a shape that
        also depends on *where the cursor is* has to be resolved
        against the others, or arming Pick Peak while hovering an axis
        strip would have the next mouse move quietly undo it.

        Precedence, highest first:

        1. **The toolbar's pan/zoom mode.** matplotlib sets its own
           MOVE/SELECT_REGION cursor there and caches what it set, so
           we leave it alone -- and forget our own shape, so that when
           the mode ends (the toolbar resetting to POINTER on its next
           mouse move) the call after that re-applies the context
           cursor instead of finding it unchanged and skipping.
           Ordering note: the toolbar is constructed before
           ``_on_motion`` is connected, so its ``mouse_move`` callback
           runs first and ours has the last word each move. Swapping
           those two lines in ``__init__`` would silently break this.
        2. **A drag in flight.** Changing shape mid-gesture reads as
           the gesture having ended.
        3. **A click mode** -- Pick Peak, Select Region, Measure. Each
           arms a click, and the cross is what says so.
        4. **A scroll-zoomable tick strip**, per :attr:`_STRIP_CURSORS`.
        5. Otherwise the plain arrow. Plot bodies land here: the
           crosshair is the cursor over data (see
           :attr:`_STRIP_CURSORS`), and the arrow on top of it stays
           out of the way of the cross the click modes use.

        ``event`` of ``None`` (every call from a mode setter) resolves
        as "no strip" rather than reusing the last known position:
        a stored event's Axes go stale on any rebuild, and this canvas
        has been bitten repeatedly by exactly that. The cost is that
        toggling a mode off while parked on a strip shows the arrow
        until the next mouse move, which is one frame nobody sees.
        """
        if event is None:
            # Every mode setter lands here: keep the Pan/Zoom check marks
            # on what a drag will now do.
            self.toolbar._update_buttons_checked()
        if self.toolbar.mode:
            self._last_cursor_shape = None
            return
        if (self._pan_start is not None or self._region_select_start is not None
                or self._zoom_start is not None):
            return
        if (self._pick_mode or self._region_select_mode or self._measurement_mode
                or getattr(self, "_width_mode", False) or self._inset_mode):
            shape = Qt.CursorShape.CrossCursor
        else:
            strip = self._scroll_strip_at(event)
            shape = (self._STRIP_CURSORS[strip[0]] if strip is not None
                     else Qt.CursorShape.ArrowCursor)
        if shape == self._last_cursor_shape:
            return
        self._last_cursor_shape = shape
        self.canvas.setCursor(shape)

    def _on_figure_leave(self, _event) -> None:
        """Forget the applied shape when the cursor leaves the canvas.

        Qt shows whichever widget the cursor is now over, so nothing is
        visibly stuck -- but the *cache* would still claim the strip
        arrow, and re-entering at the same pixel would find "no change
        needed" and leave whatever shape the canvas had drifted to."""
        self._last_cursor_shape = None

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

    def _x_strip_depth(self, ax) -> float:
        """How far below ``ax``'s data rectangle its x-tick strip
        reaches, in pixels.

        **Bounded, unlike the "everything below this axes" it
        replaces.** In a stacked Compare the region under one row *is*
        the next row down, so an unbounded strip handed more than half
        the plot to the x-zoom: measured on a 12-row stack, 52% of a
        vertical sweep resolved to the x strip against 42% to a row
        body, interleaving every few pixels, so scrolling to scale
        intensity zoomed x instead about half the time. Reported as
        "the x-axis scroll to zoom is a bit of an issue when there are
        lots of spectra stacked and you want to scale the y intensity."

        A panel with nothing below it keeps the old behaviour and runs
        to the bottom of the figure -- that is every single-panel view,
        where there is no other panel for the region to belong to and
        aiming below the axis label should still work. Otherwise the
        strip is only as deep as the decorations actually are, which
        for a stacked row that draws no tick labels at all (every row
        but the bottom, sharing one x axis) is zero: no strip, so the
        gap above it falls through to the intensity zoom.
        """
        bbox = ax.get_window_extent()
        below = any(other is not ax and other.get_window_extent().y0 < bbox.y0
                    for other in self.figure.axes)
        if not below:
            return float(bbox.y0)
        try:
            tight = ax.get_tightbbox()
        except Exception:  # noqa: BLE001 -- renderer not ready; fall back to no strip
            return 0.0
        if tight is None:
            return 0.0
        return float(max(bbox.y0 - tight.y0, 0.0))

    def _hovering_x_axis(self, ax, event) -> bool:
        """True if the cursor is in ``ax``'s x-tick-label strip --
        below its plot area, within its horizontal span, and no deeper
        than :meth:`_x_strip_depth`. ``event.inaxes`` alone can't
        detect this: matplotlib only sets it while the cursor is inside
        the axes' own data rectangle, not its margins."""
        bbox = ax.get_window_extent()
        if not (bbox.x0 <= event.x <= bbox.x1 and event.y < bbox.y0):
            return False
        return event.y >= bbox.y0 - self._x_strip_depth(ax)

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

    def set_contour_scroll_up_shows_more(self, up_shows_more: bool) -> None:
        """Choose which way a scroll over a contour (2D or DOSY) moves
        its levels: ``True`` -- scroll up shows *more* contours (lowers
        the floor); ``False`` (default) -- scroll up shows fewer."""
        self._contour_scroll_up_more = bool(up_shows_more)

    def contour_scroll_up_shows_more(self) -> bool:
        return self._contour_scroll_up_more

    def _contour_scroll_step(self, event) -> float:
        """Multiplier for the contour floor from one scroll click.
        Above 1 raises the floor (fewer levels), below 1 lowers it."""
        raise_floor = (event.button == "up") != self._contour_scroll_up_more
        return self.CONTOUR_ZOOM_FACTOR if raise_floor else 1 / self.CONTOUR_ZOOM_FACTOR

    def _adjust_contour_levels(self, event) -> None:
        """By default scroll-up raises the contour floor (fewer,
        stronger-only levels -- "zoom into the peaks") and scroll-down
        lowers it (reveals weaker signals/noise); reversed when
        :meth:`set_contour_scroll_up_shows_more` is on. Redraws the
        whole contour (~tens of ms, see plot_2d_contour) but preserves
        the current pan/zoom."""
        step = self._contour_scroll_step(event)
        self._contour_factor = min(
            max(self._contour_factor * step, self._CONTOUR_FACTOR_MIN), self._CONTOUR_FACTOR_MAX,
        )
        self._redraw_contour(preserve_view=True)

    def _on_contour_scroll_action_toggled(self, checked: bool) -> None:
        self.set_contour_scroll_up_shows_more(checked)
        self.contourScrollDirectionChanged.emit(bool(checked))

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

    # -- external projections -------------------------------------------

    def external_projection(self, dimension: str):
        """The 1D shown as the current contour's ``"f1"``/``"f2"``
        projection, as ``{"spectrum", "label"}``, or ``None`` when that
        strip shows the projection computed from the 2D itself."""
        entry = self._external_projection_entry()
        return None if entry is None else entry.get(dimension)

    def external_projections_for(self, spectrum) -> dict:
        """``{"f1"/"f2": {"spectrum", "label"}}`` set on the 2D
        ``spectrum``, whether or not it is the one shown right now."""
        entry = self._external_projections.get(id(spectrum))
        if entry is None or entry["owner"] is not spectrum:
            return {}
        return {d: entry[d] for d in ("f2", "f1") if d in entry}

    def _external_projection_entry(self, create: bool = False):
        return self._per_2d_entry(self._external_projections, create)

    def _per_2d_entry(self, store: dict, create: bool = False, spectrum=None):
        """The current (or given) 2D's entry in a per-2D ``store`` --
        keyed by ``id()``, with the 2D itself kept as ``"owner"`` so a
        recycled id can never hand one 2D's settings to another."""
        spectrum = self._contour_spectrum if spectrum is None else spectrum
        if spectrum is None:
            return None
        entry = store.get(id(spectrum))
        if entry is not None and entry["owner"] is not spectrum:
            entry = None
            del store[id(spectrum)]
        if entry is None and create:
            entry = store[id(spectrum)] = {"owner": spectrum}
        return entry

    def set_external_projection(self, dimension: str, spectrum: NMRSpectrum,
                                label: str | None = None) -> None:
        """Draw the 1D ``spectrum`` as the current contour's
        ``dimension`` (``"f2"`` top, ``"f1"`` left) projection, in
        place of the one computed from the 2D, until cleared.

        The caller is responsible for having checked that it belongs
        there (:func:`opennmr.notebook.check_external_projection`);
        this only refuses what cannot be drawn at all. The 1D is held
        by reference, so reprocessing it (phase, baseline) shows up at
        the next redraw of the contour."""
        if dimension not in ("f1", "f2"):
            raise ValueError("dimension must be 'f1' or 'f2'")
        if self._contour_spectrum is None:
            raise ValueError("No 2D contour is shown.")
        if spectrum is None or spectrum.ndim != 1:
            raise ValueError("An external projection must be a 1D spectrum.")
        if dimension == "f1" and self._contour_spectrum.is_pseudo_2d:
            raise ValueError("A pseudo-2D's F1 axis is a slice number, not a shift.")
        entry = self._external_projection_entry(create=True)
        entry[dimension] = {"spectrum": spectrum, "label": label or _spectrum_label(spectrum)}
        # A strip the user had hidden comes back: dropping onto it is
        # asking to see the 1D there.
        self._projection_style[dimension].pop("visible", None)
        self._redraw_contour(preserve_view=True, fresh_projections=(dimension,))

    def clear_external_projection(self, dimension: str | None = None) -> None:
        """Go back to the computed projection on ``dimension``, or on
        both strips when ``dimension`` is ``None``."""
        entry = self._external_projection_entry()
        if entry is None:
            return
        dims = ("f1", "f2") if dimension is None else (dimension,)
        changed = tuple(d for d in dims if entry.pop(d, None) is not None)
        if changed and self._contour_spectrum is not None:
            self._redraw_contour(preserve_view=True, fresh_projections=changed)

    def prune_external_projections(self, alive) -> None:
        """Forget external projections belonging to, or made from, a
        spectrum no longer in ``alive`` (the session's spectra) -- after
        a row is removed from Open experiments. Redraws if the current
        contour lost one."""
        alive_ids = {id(s) for s in alive}
        current_changed = False
        for key in list(self._external_projections):
            entry = self._external_projections[key]
            if id(entry["owner"]) not in alive_ids:
                del self._external_projections[key]
                continue
            for dim in ("f1", "f2"):
                held = entry.get(dim)
                if held is not None and id(held["spectrum"]) not in alive_ids:
                    del entry[dim]
                    current_changed |= entry["owner"] is self._contour_spectrum
        for key in list(self._axis_label_overrides):
            if id(self._axis_label_overrides[key]["owner"]) not in alive_ids:
                del self._axis_label_overrides[key]
        if current_changed and self._contour_spectrum is not None:
            self._redraw_contour(preserve_view=True, fresh_projections=("f1", "f2"))

    def _external_projection_kwargs(self) -> dict:
        """``f1_external``/``f2_external`` for
        :func:`opennmr.plotting.plot_2d_contour`, both call sites, in
        the contour's *displayed* unit.

        Converted through the 2D's own axis (every unit here is linear
        in ppm), not the 1D's Hz axis: a 1D recorded at another field
        has a different Hz axis for the same shifts, and only the ppm
        values are comparable between the two."""
        entry = self._external_projection_entry()
        if not entry:
            return {}
        kwargs = {}
        for dim, axis_index in (("f2", 1), ("f1", 0)):
            held = entry.get(dim)
            if held is None:
                continue
            spectrum = held["spectrum"]
            if self._axis_unit == "ppm":
                kwargs[f"{dim}_external"] = spectrum
                continue
            axis = self._contour_spectrum.axes[axis_index]
            values = self._axis_values_in_unit(axis, self._axis_unit)
            if values is None or axis.ppm is None or len(axis.ppm) < 2:
                continue  # not expressible in this unit: the computed one stays
            slope, intercept = np.polyfit(np.asarray(axis.ppm, float), np.asarray(values, float), 1)
            kwargs[f"{dim}_external"] = (slope * np.asarray(spectrum.axes[0].ppm, float) + intercept,
                                         spectrum.data)
        return kwargs

    def projection_drop_target(self, pos) -> str | None:
        """Where a drop at widget position ``pos`` (a ``QPointF``) would
        land: ``"f2"``/``"f1"`` over that projection strip, ``"auto"``
        anywhere else on a contour, ``None`` when no contour is shown
        (nothing to drop onto)."""
        if self._contour_spectrum is None:
            return None
        x, y = self.canvas.mouseEventCoords(pos)
        for dim in ("f2", "f1"):
            ax = self._proj_axes.get(dim)
            if ax is not None and ax.bbox.contains(x, y):
                return dim
        return "auto"

    def _set_drop_highlight(self, target) -> None:
        ax = self._proj_axes.get(target) if target in ("f1", "f2") else None
        if self._drop_highlight is not None and self._drop_highlight[0] is ax:
            return
        if self._drop_highlight is not None:
            old_ax, old_colour = self._drop_highlight
            old_ax.set_facecolor(old_colour)
            self._drop_highlight = None
        if ax is not None:
            self._drop_highlight = (ax, ax.get_facecolor())
            ax.set_facecolor(self._DROP_HIGHLIGHT_COLOUR)
        self.canvas.draw_idle()

    def eventFilter(self, obj, event) -> bool:  # noqa: N802 -- Qt override
        """Drags from the Open experiments list onto the plot. Anything
        else -- files from the desktop in particular -- is left alone
        (``return False`` with the event ignored), so it carries on up
        to the main window's own folder drop."""
        if obj is not self.canvas:
            return super().eventFilter(obj, event)
        kind = event.type()
        if kind not in (QEvent.Type.DragEnter, QEvent.Type.DragMove,
                        QEvent.Type.DragLeave, QEvent.Type.Drop):
            return False
        if kind == QEvent.Type.DragLeave:
            self._set_drop_highlight(None)
            return False
        mime = event.mimeData()
        if mime.hasUrls() and not mime.hasFormat(self._LIST_DRAG_MIME):
            if not self.accept_file_drops:
                event.ignore()
                return False
            # Files or folders from the file explorer / Finder: opened by the window.
            event.setDropAction(Qt.DropAction.CopyAction)
            event.accept()
            if kind == QEvent.Type.Drop:
                paths = [u.toLocalFile() for u in mime.urls() if u.isLocalFile()]
                if paths:
                    QTimer.singleShot(0, lambda: self._emit_safely(self.pathsDropped, paths))
            return True
        if not mime.hasFormat(self._LIST_DRAG_MIME):
            event.ignore()
            return False
        target = self.projection_drop_target(event.position())
        if target is None and not self.accept_row_drops:
            event.ignore()
            return True
        if target is None:
            # Not over a projection strip: the dragged experiments go onto the plot itself.
            event.setDropAction(Qt.DropAction.CopyAction)
            event.accept()
            if kind == QEvent.Type.Drop:
                rows = _rows_from_list_drag(mime, self._LIST_DRAG_MIME)
                if rows:
                    QTimer.singleShot(0, lambda: self._emit_safely(self.experimentRowsDropped, rows))
            return True
        event.setDropAction(Qt.DropAction.CopyAction)
        event.accept()
        if kind == QEvent.Type.Drop:
            self._set_drop_highlight(None)
            rows = _rows_from_list_drag(event.mimeData(), self._LIST_DRAG_MIME)
            if rows:
                # Deferred: the owner may ask a question, and a modal
                # dialog opened inside Qt's drag loop is unreliable
                # (Windows in particular leaves the drag cursor stuck).
                QTimer.singleShot(0, lambda: self.externalProjectionDropped.emit(rows, target))
        else:
            self._set_drop_highlight(target)
        return True

    @staticmethod
    def _emit_safely(signal, payload) -> None:
        """Emit a deferred drop signal, unless the canvas was deleted in the meantime."""
        try:
            signal.emit(payload)
        except RuntimeError:
            pass

    # -- projection strip style ----------------------------------------

    #: Line styles offered for a projection strip, (label, matplotlib).
    PROJECTION_LINESTYLES = (("Solid", "-"), ("Dashed", "--"), ("Dotted", ":"), ("Dash-dot", "-."))

    def projection_style(self, dimension: str) -> dict:
        """A copy of the ``"f1"``/``"f2"`` strip's own style settings
        (only what has been changed from the defaults)."""
        return dict(self._projection_style[dimension])

    def set_projection_style(self, dimension: str, **style) -> None:
        """Change one projection strip's look -- any of ``color``,
        ``linewidth``, ``linestyle``, ``size`` (inches), ``visible``.
        ``None`` puts that setting back to its default. Redraws a
        contour on screen, keeping the zoom."""
        if dimension not in ("f1", "f2"):
            raise ValueError("dimension must be 'f1' or 'f2'")
        from ...plotting.twod import PROJECTION_STYLE_KEYS
        unknown = set(style) - set(PROJECTION_STYLE_KEYS)
        if unknown:
            raise ValueError(f"Unknown projection style key(s): {sorted(unknown)}")
        current = self._projection_style[dimension]
        for key, value in style.items():
            if value is None:
                current.pop(key, None)
            else:
                current[key] = value
        if current.get("visible") is True:
            current.pop("visible")  # shown is the default: keep the dict minimal
        if self._contour_spectrum is not None:
            self._redraw_contour(preserve_view=True)

    def reset_projection_style(self, dimension: str | None = None) -> None:
        for dim in (("f1", "f2") if dimension is None else (dimension,)):
            self._projection_style[dim] = {}
        if self._contour_spectrum is not None:
            self._redraw_contour(preserve_view=True)

    def _projection_mode_kwargs(self) -> dict:
        return {"projection_mode": self._projection_mode}

    def projection_mode(self) -> str:
        """``"auto"``/``"max"``/``"sum"``: how the computed strips are built."""
        return self._projection_mode

    def set_projection_mode(self, mode: str) -> None:
        """Build the computed projections as the maximum (``"max"``, what
        ``"auto"`` means) or the sum (``"sum"``, which keeps the exact
        shear invariants but cancels positive against negative lobes) and
        redraw the current contour, keeping the zoom."""
        if mode not in ("auto", "max", "sum"):
            raise ValueError("projection mode must be 'auto', 'max' or 'sum'")
        if mode == self._projection_mode:
            return
        self._projection_mode = mode
        if self._contour_spectrum is not None:
            self._redraw_contour(preserve_view=True, fresh_projections=("f1", "f2"))

    def _projection_style_kwargs(self) -> dict:
        style = {d: dict(v) for d, v in self._projection_style.items() if v}
        return {"projection_style": style} if style else {}

    def _projection_default_colour(self, dimension: str) -> str:
        """What the strip is drawn in when no colour is set: crimson
        for an external 1D, black for the computed projection -- the
        style dialog shows this rather than a blank swatch."""
        return "crimson" if self.external_projection(dimension) is not None else "black"

    def _projection_dimension_of(self, ax) -> str | None:
        for dim in ("f2", "f1"):
            if ax is not None and ax is self._proj_axes.get(dim):
                return dim
        return None

    def _prompt_projection_style(self, dimension: str) -> None:
        """The right-click "Projection Style…" dialog for one strip.
        Changes show live; Cancel puts back what was there before."""
        before = {d: dict(v) for d, v in self._projection_style.items()}
        dialog = _ProjectionStyleDialog(
            self, dimension, self._projection_style[dimension],
            default_colour=self._projection_default_colour(dimension))

        def preview() -> None:
            style = dialog.style()
            dims = ("f1", "f2") if dialog.apply_to_both() else (dimension,)
            for dim in ("f1", "f2"):
                self._projection_style[dim] = dict(style if dim in dims else before[dim])
            if self._contour_spectrum is not None:
                self._redraw_contour(preserve_view=True)

        dialog.changed.connect(preview)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            self._projection_style = before
            if self._contour_spectrum is not None:
                self._redraw_contour(preserve_view=True)
            return
        preview()

    # -- per-axis labels on a 2D -----------------------------------------

    _UNIT_TEXT = {"ppm": "ppm", "hz": "Hz", "khz": "kHz", "mm": "mm"}

    def axis_label_override(self, dimension: str, spectrum=None):
        """The label typed for the current (or given) 2D's
        ``"f2"``/``"f1"`` axis, as ``{"text", "unit"}``, or ``None``
        when that axis uses the automatic label."""
        entry = self._per_2d_entry(self._axis_label_overrides, spectrum=spectrum)
        return None if entry is None else entry.get(dimension)

    def set_axis_label_override(self, dimension: str, text: str | None) -> None:
        """Label the current 2D's ``"f2"`` (horizontal) or ``"f1"``
        (vertical) axis with ``text`` instead of the automatic label;
        blank or ``None`` goes back to automatic. The other axis is not
        touched.

        ``{nuc}``, ``{mass}``, ``{element}`` and ``{unit}`` are filled
        in for *that* axis (so ``"δ {nuc} / {unit}"`` stays right
        through a unit change). Everything else is kept exactly as
        typed, braces included, so mathtext like ``$^{13}$C`` works.

        A label with no ``{unit}`` is tied to the unit it was typed in:
        switching the axis to Hz (say) shows the automatic Hz label
        rather than a label still saying "ppm", and switching back
        brings the typed one back."""
        if dimension not in ("f1", "f2"):
            raise ValueError("dimension must be 'f1' or 'f2'")
        if self._contour_spectrum is None:
            raise ValueError("No 2D contour is shown.")
        text = (text or "").strip()
        if text:
            _png, error = _render_label(text)
            if error:
                raise ValueError(f"This label cannot be drawn: {error}")
        entry = self._per_2d_entry(self._axis_label_overrides, create=True)
        if text:
            entry[dimension] = {"text": text, "unit": self._axis_unit}
        else:
            entry.pop(dimension, None)
        # A full rebuild rather than set_xlabel: the extracted-slice
        # panel carries its own copy of the axis label, and a longer
        # label needs the layout pass a rebuild ends with.
        self._redraw_contour(preserve_view=True)

    def _axis_nucleus(self, spectrum, dimension: str) -> str | None:
        meta = spectrum.metadata
        if dimension == "f2":
            raw, sfo1 = meta.nucleus, meta.sfo1
        else:
            raw = meta.nucleus_indirect or (meta.nucleus if meta.homonuclear else None)
            sfo1 = meta.sfo1_indirect if meta.nucleus_indirect else meta.sfo1
        try:
            return find_gamma(raw, SFO1=sfo1, quiet=True)["Nuclide"]
        except Exception:
            return raw

    def _apply_axis_label_override(self, spectrum, dimension: str, automatic):
        """``automatic`` unless this 2D's ``dimension`` has a typed
        label that applies in the current unit -- see
        :meth:`set_axis_label_override`."""
        held = self.axis_label_override(dimension, spectrum=spectrum)
        if held is None:
            return automatic
        text = held["text"]
        if "{unit}" not in text and held["unit"] != self._axis_unit:
            return automatic
        nuc = self._axis_nucleus(spectrum, dimension) or "?"
        element = nuc.strip("0123456789")
        mass = nuc[:len(nuc) - len(element)]
        for key, value in (("{nuc}", nuc), ("{mass}", mass), ("{element}", element),
                           ("{unit}", self._UNIT_TEXT.get(self._axis_unit, self._axis_unit))):
            text = text.replace(key, value)
        return text

    def _automatic_axis_labels(self):
        """``(f2_label, f1_label)`` the contour gets with no typed
        labels -- what the label dialog offers to go back to."""
        spectrum = self._contour_spectrum
        entry = self._axis_label_overrides.pop(id(spectrum), None)
        try:
            _spec, xlabel, ylabel = self._contour_for_display()
        finally:
            if entry is not None:
                self._axis_label_overrides[id(spectrum)] = entry
        return xlabel, ylabel

    def _prompt_axis_labels(self) -> None:
        """Right-click "Edit Axis Labels (F2 / F1)…": each axis on its
        own line, pre-filled with what it shows now."""
        spectrum = self._contour_spectrum
        if spectrum is None:
            return
        auto_x, auto_y = self._automatic_axis_labels()
        _spec, current_x, current_y = self._contour_for_display()
        dialog = _AxisLabelsDialog(
            self, f2_text=current_x or "", f1_text=current_y or "",
            f2_auto=auto_x or "", f1_auto=auto_y or "",
            f1_enabled=not spectrum.is_pseudo_2d)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        for dim, typed, auto in (("f2", dialog.f2_text(), auto_x), ("f1", dialog.f1_text(), auto_y)):
            if dim == "f1" and spectrum.is_pseudo_2d:
                continue
            # Unchanged from the automatic label = no override, so it
            # keeps following the Plot Style label settings.
            self.set_axis_label_override(dim, None if typed == (auto or "") else typed)

    def _slice_panel_kwargs(self) -> dict:
        """``slices`` override for
        :func:`opennmr.plotting.plot_2d_contour`, both call sites --
        see :meth:`set_contour_slices`. The traces float above the
        panel's axis line by Plot Style's "Gap below spectra" and are
        spaced by its "Gap between stacked spectra" (both fractions of a
        trace's height), as in a 1D and a stacked Compare."""
        return {"slices": self._contour_slices or None,
                "slice_bottom_gap": self._baseline_gap,
                "slice_offset_fraction": 1.0 / (1.0 + max(self._STACK_GAP_MIN, self._stack_gap))}

    def set_slice_colour(self, position: int, colour: str) -> None:
        """Recolour one extracted slice, by its position in the list.

        The trace and its dashed marker on the contour share the
        colour, so they stay paired -- that pairing is the only thing
        connecting them since the connector lines were dropped."""
        if not (0 <= position < len(self._contour_slices)):
            return
        # Delegated, not reimplemented: a slice is a series like any
        # other, and the shared setter is what records the colour as
        # deliberate so a palette change leaves it alone.
        self.set_series_style(str(position), colour=colour)

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

    #: How many distinct colours a slice's own colour is drawn from.
    #: Fixed rather than "one per slice so far", so adding a fourth
    #: slice does not recolour the first three.
    _SLICE_PALETTE_SIZE = 8

    def _sync_slice_series(self) -> None:
        """Register the extracted slices as series and write the
        resulting colours back onto their specs.

        The specs are the store that persists with the spectrum, and
        they are what ``_draw_slice_panel`` reads -- so the series
        layer decides and the specs follow. A colour set deliberately
        is marked on the spec too, so a palette change in a later
        session respects it."""
        entries = self._assign_series("slices", [
            (str(i), spec.get("label") or f"Slice {i + 1}", True)
            for i, spec in enumerate(self._contour_slices)
        ])
        for spec, entry in zip(self._contour_slices, entries):
            if spec.get("color_override") and spec.get("color"):
                entry.colour = spec["color"]
                entry.colour_overridden = True
            spec["color"] = entry.colour

    def _next_slice_colour(self) -> str:
        """The next extracted slice's colour, from this canvas's own
        Compare colour map.

        Explicit user request: extracted slices "should be able to
        either take a colour from the standard compare map, or be
        independently clickable". Taking it from the same map the
        Compare tab uses means the two views of the same data agree,
        and changing that map in the Plot Style dialog moves both."""
        palette = plotting.stack_colors(self._SLICE_PALETTE_SIZE, self._compare_colormap)
        return matplotlib.colors.to_hex(palette[len(self._contour_slices) % len(palette)])

    def _add_contour_slice(self, axis: str, index: int, label: str) -> None:
        # **The colour is stored on the slice, not chosen at draw time.**
        # It used to be assigned by position in the draw loop, so every
        # redraw re-derived it -- and a redraw that reordered or
        # rebuilt anything (a unit change, most visibly) handed the
        # slices different colours. Reported directly: "when I changed
        # the units, the colours reset". Stored here, it survives every
        # redraw and is what a per-slice colour picker edits.
        self._contour_slices.append({
            "axis": axis, "index": int(index), "label": label,
            "color": self._next_slice_colour(),
        })
        self.contourSlicesChanged.emit(list(self._contour_slices))
        # preserve_view: adding a slice is an annotation, not a change
        # of subject -- explicit user request, "the scale/view shouldn't
        # reset when a slice is extracted". show_spectrum would rebuild
        # from scratch and drop whatever zoom was set.
        if self._contour_spectrum is not None:
            self._redraw_contour(preserve_view=True)

    def contour_slice_label(self, position: int) -> str:
        """The name a slice is shown under: its own label, or the
        default ``Slice n`` (n counting from 1) when it has none."""
        spec = self._contour_slices[position]
        return spec.get("label") or f"Slice {position + 1}"

    def rename_contour_slice(self, position: int, label: str) -> bool:
        """Re-label an extracted slice after the fact.

        The label was previously settable only in the Extract Slice
        dialog. An empty ``label`` puts the default ``Slice n`` back.
        The spec is the store that persists with the spectrum, so this
        writes there, drops any "typed in the series list" override
        that would otherwise shadow it, emits
        :attr:`contourSlicesChanged` and redraws without resetting the
        zoom. Returns ``False`` for a position that does not exist."""
        if not (0 <= position < len(self._contour_slices)):
            return False
        text = (label or "").strip()
        self._contour_slices[position]["label"] = text
        for entry in self._series.get("slices", []):
            if entry.key == str(position):
                entry.label = text or f"Slice {position + 1}"
                entry.label_overridden = bool(text)
        self.contourSlicesChanged.emit(list(self._contour_slices))
        if self._contour_spectrum is not None:
            self._redraw_contour(preserve_view=True)
        return True

    def _prompt_rename_slice(self, position: int) -> None:
        if not (0 <= position < len(self._contour_slices)):
            return
        spec = self._contour_slices[position]
        text, ok = QInputDialog.getText(
            self, "Rename slice",
            f"Label for the {spec['axis'].upper()} slice at index {spec['index']} "
            "(blank for the default):",
            text=spec.get("label") or "",
        )
        if ok:
            self.rename_contour_slice(position, text)

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

    def _restore_configured_text_sizes(self) -> None:
        """Put every label, tick and title back to the size this canvas
        is *configured* for, undoing any shrinking a previous layout
        pass applied.

        :func:`opennmr.plotting.fit_labels_within_figure` only ever
        shrinks -- it has no way to know what size the text started at,
        so it cannot grow it back. That is right for a single layout
        pass, but it means a figure laid out once while its widget was
        small keeps that shrunken text forever, however large the
        widget later becomes. Re-laying out without this first would
        measure the already-shrunk text, find it fits, and leave it.

        Reported as "DOSY axes are still reverting to the small fonts
        sometimes" -- a DOSY is computed and drawn before its tab is
        made visible, so it is laid out at whatever size a
        not-yet-shown widget happens to have."""
        for ax in self.figure.get_axes():
            ax.xaxis.label.set_fontsize(self._AXIS_LABELSIZE)
            ax.yaxis.label.set_fontsize(self._AXIS_LABELSIZE)
            self._italicise_axis_labels(ax.xaxis, ax.yaxis)
            if ax.get_title():
                ax.title.set_fontsize(self._AXIS_LABELSIZE)
            ax.tick_params(axis="both", which="both", labelsize=self._AXIS_LABELSIZE)
            for label in (*ax.get_xticklabels(), *ax.get_yticklabels(),
                          *ax.get_xticklabels(minor=True), *ax.get_yticklabels(minor=True)):
                label.set_fontsize(self._AXIS_LABELSIZE)
            self._apply_title_and_font(ax)

    def _on_canvas_resized(self, _event=None) -> None:
        """Re-fit the labels after the widget's own size changes.

        A layout is only as good as the size it was computed at, and a
        GUI canvas is resized constantly -- by the window, by the
        splitter, and by a tab becoming visible for the first time.
        Nothing re-ran the layout on any of those, so a figure drawn
        small stayed laid out (and shrunk) for small.

        Debounced: a window drag fires a resize per pixel, and each
        layout pass is a real cost on a large contour."""
        self._resize_layout_timer.start(120)

    def _run_debounced_resize_layout(self) -> None:
        if not self.figure.get_axes():
            return
        self._restore_configured_text_sizes()
        self._apply_layout()
        self.canvas.draw_idle()

    def _apply_layout(self) -> None:
        """The layout pass every ``show_*``/rebuild ends with.

        Replaces a bare ``figure.tight_layout()``. ``tight_layout``
        reserves room for decorations *when the figure is big enough*
        and quietly gives up when it is not -- and a GUI canvas is
        sized by the window, so "not big enough" is an ordinary state.
        Real user report: "the x-axis label got cut off", and later a
        request for "better safeguards... maybe some kind of warning or
        push to resize the font".

        :func:`opennmr.plotting.fit_labels_within_figure` does both:
        shrink the text as far as needed (and no further -- a figure
        with room to spare is left exactly as it was), then report
        whether that was enough. When it wasn't, :attr:`labelsClipped`
        tells the owner so a person can be told, rather than the
        figure silently rendering with its labels off the edge."""
        # Tick direction is global (Plot Style), and every view ends
        # here -- including those that set their own tick_params, such
        # as the contour's projection strips -- so it is applied here,
        # before the label fitting measures the figure.
        for ax in self.figure.get_axes():
            if getattr(ax, "name", "") != "3d":
                ax.tick_params(which="both", direction=self._TICK_DIRECTION)
        self._reapply_panel_axis_styles()
        threes = [ax for ax in self.figure.get_axes() if getattr(ax, "name", "") == "3d"]
        if threes:
            # A 3D picture keeps the text at the size asked for (tick
            # labels the same size as axis labels, user request) and
            # makes room by shrinking the box instead -- shrinking the
            # text left some labels smaller than others.
            clipped = self._fit_3d_layout(threes)
        else:
            result = plotting.fit_labels_within_figure(self.figure)
            clipped = result["clipped"] if not result["fitted"] else []
            if self._settle_panel_labels():
                result = plotting.fit_labels_within_figure(self.figure)
                clipped = result["clipped"] if not result["fitted"] else []
                self._settle_panel_labels(final=True)
        if self._compare_axes:
            # A stack's rows touch: the layout pass may have opened gaps
            # between them, and the even pitch of the spacing control
            # (see _fit_compare_rows) assumes none.
            try:
                self.figure.subplots_adjust(hspace=0.0)
            except Exception:  # noqa: BLE001 -- spacing nicety: never break a redraw
                pass
        self._apply_colorbar_gap()
        # Only on a change: a phase drag relayouts continuously, and
        # re-announcing the same thing every frame is noise.
        if clipped != self._last_clipped_labels:
            self._last_clipped_labels = clipped
            self.labelsClipped.emit(list(clipped))

    # ---- panel labels (a), (b), ... -------------------------------------------

    def panel_labels(self):
        """The :class:`opennmr.notebook.panel_labels.PanelLabels` on show, or None."""
        labels = getattr(self.figure, "_opennmr_panel_labels", None)
        if labels is None or not any(t is not None for t in labels.texts):
            return None
        return labels

    def _settle_panel_labels(self, final: bool = False) -> bool:
        """Put the panel labels where the layout just made room for them.
        Returns True when they moved (so the layout must be redone to
        reserve that room). ``final``: only place, never ask for another pass."""
        labels = self.panel_labels()
        if labels is None:
            return False
        try:
            moved = labels.place()
        except Exception:  # noqa: BLE001 -- labels are cosmetic: never take a redraw down
            return False
        return bool(moved) and not final

    def _label_target_axes(self) -> list:
        """The panels a label set belongs to: the Compare panels, or the
        side-by-side 2D panels; none for any other view."""
        if self._spectrum_panels:
            return list(self._spectrum_panels)
        if len(self._panel_axes) > 1:
            return list(self._panel_axes)
        return []

    def set_panel_labels(self, args: dict | None) -> bool:
        """Show, change or remove the panel labels (a), (b)... on the panels
        showing now. ``args``: :func:`opennmr.notebook.panel_labels.add_panel_labels`
        keywords (``style``, ``start``, ``order``, ``placement``, ``positions``,
        ``fontsize``, ...), or None / empty for no labels. Kept, so the next
        redraw of the same view carries them. Returns whether labels are
        showing (False: no multi-panel view is on)."""
        from ...notebook.panel_labels import add_panel_labels
        self._panel_label_args = dict(args) if args else None
        old = getattr(self.figure, "_opennmr_panel_labels", None)
        if old is not None:
            old.remove()
        if self._compare_panels_kwargs is not None:
            self._compare_panels_kwargs["panel_labels"] = self._panel_label_args
        axes = self._label_target_axes()
        if not args or not axes:
            self._apply_layout()
            self.canvas.draw_idle()
            return False
        kwargs = dict(args)
        kwargs.setdefault("fontsize", self._AXIS_LABELSIZE * 1.15)
        add_panel_labels(self.figure, axes, **kwargs)
        self._apply_layout()
        self.canvas.draw_idle()
        return True

    # -- dragging a label with the mouse

    _LABEL_DRAG_BLOCKERS = ("_pick_mode", "_region_select_mode", "_measurement_mode", "_width_mode",
                            "_inset_mode")

    def _panel_label_at(self, event):
        """``(panel index, Text)`` of the panel label under the mouse, or None."""
        labels = self.panel_labels()
        if labels is None or event.x is None or event.y is None:
            return None
        if self.toolbar.mode != "" or any(getattr(self, name, False) for name in self._LABEL_DRAG_BLOCKERS):
            return None
        try:
            renderer = self.canvas.get_renderer()
        except AttributeError:
            return None
        for i, text in enumerate(labels.texts):
            if text is None or not text.get_visible():
                continue
            box = text.get_window_extent(renderer).expanded(1.15, 1.25)
            if box.contains(event.x, event.y):
                return i, text
        return None

    def _begin_label_drag(self, event) -> bool:
        if event.button != 1 or getattr(event, "dblclick", False):
            return False
        hit = self._panel_label_at(event)
        if hit is None:
            return False
        i, _text = hit
        labels = self.panel_labels()
        self._label_drag = {"panel": i, "x": float(event.x), "y": float(event.y),
                            "offset": tuple(labels.offsets[i]), "moved": False}
        return True

    def _drag_label(self, event) -> None:
        drag = self._label_drag
        labels = self.panel_labels()
        if drag is None or labels is None or event.x is None or event.y is None:
            return
        scale = 72.0 / self.figure.dpi
        dx = drag["offset"][0] + (event.x - drag["x"]) * scale
        dy = drag["offset"][1] + (event.y - drag["y"]) * scale
        if abs(event.x - drag["x"]) + abs(event.y - drag["y"]) > 2:
            drag["moved"] = True
        labels._apply(drag["panel"], (dx, dy))
        drag["current"] = (dx, dy)
        self.canvas.draw_idle()

    def _end_label_drag(self, event) -> None:
        drag, self._label_drag = self._label_drag, None
        if drag is None or not drag.get("moved") or "current" not in drag:
            return
        dx, dy = drag["current"]
        labels = self.panel_labels()
        if labels is not None:
            labels.offsets[drag["panel"]] = (dx, dy)
        self.panelLabelMoved.emit(int(drag["panel"]), float(dx), float(dy))

    def _cancel_navigation_drag(self) -> None:
        """Abandon any half-finished toolbar zoom/pan drag, and take
        its rubber-band rectangle off screen.

        Real user report: "the zoom dashed line box stays sometimes
        when switching around". ``NavigationToolbar2.release_zoom``
        returns immediately when ``_zoom_info`` is ``None`` and is the
        *only* caller of ``remove_rubberband`` -- so a zoom drag that
        never gets its release (the figure is rebuilt underneath it by
        a spectrum/tab switch mid-drag) leaves the Qt backend's
        ``_draw_rect_callback`` installed, and every later repaint
        draws the same dashed box again over unrelated content.
        ``_zoom_info`` is also left holding an Axes that no longer
        exists.

        Blitting made this far more visible than it used to be (the
        crosshair repaints on every mouse move, each one re-running
        that stale callback), but the leak predates it -- so this
        clears the state rather than papering over the repaints.

        Written defensively with ``getattr``: these are private
        matplotlib attributes, and a version that renames them should
        degrade to doing nothing rather than crashing a redraw.

        The press connected a motion callback (``drag_zoom`` /
        ``drag_pan``, its id in the info's ``cid``) that reads the info
        on every mouse move. Clearing the info without disconnecting it
        left that callback live: every later mouse move raised
        ``AttributeError: 'NoneType' object has no attribute 'button'``
        in ``drag_zoom`` (user report from a Mac, 2026-10-15, a
        traceback per mouse move on the command line). So this undoes
        exactly what the press did -- disconnect the drag callback and,
        for a pan, reconnect the toolbar's ordinary ``mouse_move`` as
        ``release_pan`` does."""
        toolbar = getattr(self, "toolbar", None)
        if toolbar is not None:
            mpl_canvas = getattr(toolbar, "canvas", None) or self.canvas
            for attr in ("_zoom_info", "_pan_info"):
                info = getattr(toolbar, attr, None)
                if info is None:
                    continue
                cid = getattr(info, "cid", None)
                if cid is not None:
                    mpl_canvas.mpl_disconnect(cid)
                if attr == "_pan_info" and hasattr(toolbar, "mouse_move"):
                    for ax in getattr(info, "axes", ()) or ():
                        try:
                            ax.end_pan()
                        except Exception:  # noqa: BLE001 -- an Axes already torn down
                            pass
                    toolbar._id_drag = mpl_canvas.mpl_connect(
                        "motion_notify_event", toolbar.mouse_move)
                setattr(toolbar, attr, None)
        # This canvas's own box-zoom drag (the default drag) too.
        self._zoom_start = None
        self._pan_start = None
        draw_rectangle = getattr(self.canvas, "drawRectangle", None)
        if draw_rectangle is not None:
            draw_rectangle(None)

    def _clear_figure(self) -> None:
        """``figure.clear()``, but tearing down the crosshair first,
        and cancelling any in-flight toolbar zoom/pan drag (see
        :meth:`_cancel_navigation_drag`) whose Axes this is about to
        destroy.

        Every one of this canvas's own rebuild paths goes through here
        rather than calling ``self.figure.clear()`` directly -- see
        :class:`_ContourCrosshair`'s own docstring for why teardown
        has to happen synchronously, in the same call that destroys
        the Axes a crosshair is bound to, rather than lazily on a
        later mouse-move."""
        if self._crosshair is not None:
            self._crosshair.disconnect()
            self._crosshair = None
        # Opt-in per view: a rebuild starts from "no crosshair" and the
        # next show_* says whether it wants one.
        self._crosshair_ready = False
        # Likewise the curve Axes whose top/right spines Plot Style
        # controls: only the next curve view re-declares them.
        self._curve_axes = []
        # And the fit view a component-style change would redraw, and the
        # curve fit a "Keep This Fit…" would keep.
        self._slice_fit_active = False
        self._slice_fit_overlay = None
        self._curve_fit_shown = None
        self._curve_fits_shown = []
        self.series_slices_removable = False
        self.simulation_spectrum = None
        # The Axes these were drawn on are about to go; the Python-side
        # list would otherwise keep handing out artists belonging to a
        # figure that no longer contains them.
        self._measurement_artists = []
        self._measurement_points = []
        # Same reasoning: the Axes these describe are about to go, and a
        # stale list would have set_axis_unit rewriting rows that no
        # longer exist. Cleared here, the one choke point every rebuild
        # path already goes through, rather than at each of the eight
        # places _compare_axes is reset.
        self._compare_spectra = []
        self._compare_kwargs = None
        self._spectrum_panels = []
        self._spectrum_panel_groups = []
        self._spectrum_panel_link = None
        self._compare_panels_kwargs = None
        # A heatmap's rows (for Measure's ppm/Hz, see _ppm_drawn_view).
        self._heatmap_spectra = []
        # The labels belonged to the Axes about to go
        self._label_drag = None
        self.figure._opennmr_panel_labels = None
        self._overlay_spectra = []
        self._overlay_labels = None
        self._overlay_kwargs = {}
        # Side-by-side 2D panels (Compare): their Axes and the limits
        # each was drawn at, which is how a zoom is recognised for
        # "Add inset from zoom".
        self._panel_axes = []
        self._panel_home_limits = []
        self._panel_spectra = []
        self._panels_linked = False
        # A DOSY's own state describes Axes about to be destroyed. The
        # redraw path re-sets it immediately afterwards; every other
        # view leaves it cleared, which is what makes "is this a DOSY?"
        # a reliable question for the scroll/right-click handlers.
        self._dosy_result = None
        self._dosy_overlay = None
        self._dosy_overlay_labels = None
        self._dosy_projection_ax = None
        # Rows drawn beside the map go with the map, for the same
        # reason: they describe Axes about to be destroyed, and a
        # canvas showing nothing has no rows. _draw_dosy carries them
        # across its own rebuild explicitly, alongside the result.
        self._dosy_slices = []
        # The shape was resolved against bboxes of Axes about to be
        # destroyed. Same choke point, same reasoning as the crosshair
        # teardown above.
        self._last_cursor_shape = None
        self._home_limits = None
        self._natural_limits = None
        self._trace_label_artists = []
        self._external_plot = False
        self._external_draw = None
        self.plot_owner = None
        self._cancel_navigation_drag()
        self.figure.clear()
        self.figureCleared.emit()

    def _remove_crosshair(self) -> None:
        """Tear the crosshair down while its Axes stay alive -- see
        :meth:`_ContourCrosshair.remove`. ``_clear_figure`` uses plain
        ``disconnect`` instead, since it destroys the Axes anyway."""
        if self._crosshair is not None:
            self._crosshair.remove()
            self._crosshair = None

    def _ensure_crosshair(self) -> None:
        """Create the crosshair the first time the mouse moves over a
        freshly (re)drawn contour -- not the moment it's drawn.

        Creating it eagerly, right after ``plot_2d_contour``, would add
        its (invisible-until-hovered) ``Line2D`` artists to ``ax.lines``
        for a view nobody has looked at yet -- breaking anything that
        counts lines on a just-built contour (e.g. the nQ-diagonal
        toggle asserting ``len(ax.lines) == 1``). Creation timing was
        never the bug; only teardown timing was -- see
        :class:`_ContourCrosshair`'s docstring and
        :meth:`_clear_figure`, which already tears down synchronously
        on every rebuild regardless of whether one was ever created."""
        if not self._crosshair_enabled or self._crosshair is not None or self.ax is None:
            return
        # 1D gets one too, on request ("ideally I'd also like crosshairs
        # on the 1D spectra"). The class already handles a lone Axes --
        # with no projection strips to span, `f1_ax`/`f2_ax` are None
        # and it draws a single vertical and horizontal line across the
        # plot.
        #
        # Gated on `_crosshair_ready` rather than on which *kind* of
        # spectrum is shown: it used to require a contour or a 1D
        # spectrum, which kept it off Compare, the spectral fit and the
        # curve fit -- all views where reading a value off the plot is
        # exactly what you are doing ("it would be good to have
        # crosshairs available on all plots including compare, spectral
        # fit, and fit"). Each view that wants one now says so; the FID
        # and processing-preview views deliberately do not.
        if not self._crosshair_ready:
            return
        self._crosshair = _ContourCrosshair(
            self.canvas, self.ax, self._proj_axes.get("f1"), self._proj_axes.get("f2"),
            self.toolbar, stack_axes=self._compare_axes,
            panel_axes=(self._panel_axes if len(self._panel_axes) > 1 else
                        [a for a in self._spectrum_panels
                         if getattr(a, "_opennmr_panel_kind", "") != "pulse"]
                        if len(self._spectrum_panels) > 1 else None),
            free_axes=self._crosshair_free_axes(),
            color=self._crosshair_color, linewidth=self._crosshair_width, linestyle="--",
        )

    def _crosshair_free_axes(self) -> list | None:
        """Every plot of a figure drawn elsewhere (show_external_plot),
        for the crosshair to follow the cursor into -- not a 3D view, a
        colour bar, or an Axes with its axis switched off (a legend or
        text panel)."""
        if not self._external_plot:
            return None
        axes = [ax for ax in self.figure.axes
                if getattr(ax, "name", "") != "3d" and ax.get_visible() and ax.axison
                and getattr(ax, "_colorbar", None) is None
                and not getattr(ax, "_colorbar_info", None)]
        return axes if len(axes) > 1 else None

    def _projection_intensity_limits(self) -> dict:
        """Each projection strip's own intensity limit -- the F2 (top)
        strip plots intensity along y, the F1 (left) strip along x.
        Captured around a rebuild so a scroll-zoom of a projection
        survives one."""
        limits = {}
        if self._proj_axes.get("f2") is not None:
            limits["f2"] = self._proj_axes["f2"].get_ylim()
        if self._proj_axes.get("f1") is not None:
            limits["f1"] = self._proj_axes["f1"].get_xlim()
        return limits

    def _restore_projection_intensity_limits(self, limits: dict) -> None:
        if "f2" in limits and self._proj_axes.get("f2") is not None:
            self._proj_axes["f2"].set_ylim(limits["f2"])
        if "f1" in limits and self._proj_axes.get("f1") is not None:
            self._proj_axes["f1"].set_xlim(limits["f1"])

    @staticmethod
    def _contour_display_limits(display_spectrum):
        """``(xlims, ylims)`` framing a contour on the *data's* own
        range rather than the axis's full extent -- an axis that is
        zero-filled, or that runs past the acquired region, otherwise
        draws F1 far larger than the signal in it.

        Used by **both** contour draw paths (the first draw in
        ``show_spectrum`` and every rebuild in ``_redraw_contour``).
        Applying it to only one of them is what made a contour visibly
        jump the first time anything rebuilt it -- a level scroll, a
        unit switch, adding a slice -- since the two then framed the
        same data differently.

        ``ylims`` is ``None`` for a pseudo-2D, whose indirect axis is a
        slice index that plot_2d_contour already frames correctly."""
        x_values = display_spectrum.axes[1].ppm
        xlims = (float(np.max(x_values)), float(np.min(x_values)))
        if display_spectrum.is_pseudo_2d:
            return xlims, None
        y_values = display_spectrum.axes[0].ppm
        return xlims, (float(np.min(y_values)), float(np.max(y_values)))

    def _redraw_contour(self, preserve_view: bool, fresh_projections=()) -> None:
        """Rebuild the contour. ``fresh_projections``: strips (``"f1"``/
        ``"f2"``) whose intensity zoom is *not* carried over, because
        what they show has just changed (an external projection added
        or removed) -- the old zoom belonged to the old trace."""
        xlim, ylim = (self.ax.get_xlim(), self.ax.get_ylim()) if preserve_view else (None, None)
        # The projection strips are their own Axes with their own
        # intensity zoom, and a rebuild threw it away -- so scrolling the
        # contour body to rescale the levels silently reset both
        # projections at the same time. Real user report: "the 2D
        # projections shouldn't reset when I scale the contours".
        projection_limits = self._projection_intensity_limits() if preserve_view else {}
        for dim in fresh_projections:
            projection_limits.pop(dim, None)
        self._drop_highlight = None  # its Axes is about to go
        self._clear_figure()
        # _clear_figure resets this; a contour rebuild is still a
        # contour, so it opts straight back in rather than waiting
        # for a show_spectrum that will not come.
        self._crosshair_ready = True
        self.ax = self.figure.add_subplot(111)
        # Before the draw: _draw_slice_panel reads each spec's own
        # "color", so the series layer has to have decided it first --
        # otherwise a palette change would update the series list and
        # leave the panel drawing last time's colours.
        self._sync_slice_series()
        display_spectrum, unit_xlabel, unit_ylabel = self._contour_for_display()
        xlims, ylims = self._contour_display_limits(display_spectrum)
        _, _, self._proj_axes = plotting.plot_2d_contour(
            display_spectrum, ax=self.ax, fig=self.figure, factor=self._contour_factor,
            xlabel=unit_xlabel, ylabel=unit_ylabel,
            clevels=self._contour_clevels, level_mode=self._contour_level_mode, power=self._contour_power,
            square_aspect=self._aspect_ratio is None,
            xlims=xlims, ylims=ylims,
            **self._contour_fraction_kwargs(), **self._nq_line_kwargs(),
            **self._slice_panel_kwargs(), **self._contour_style_kwargs(),
            colour_spectra=self._contour_colour, insets=self._contour_insets or None,
            **self._external_projection_kwargs(), **self._projection_style_kwargs(),
                **self._projection_mode_kwargs(),
        )
        self._apply_current_axis_style()
        # Typed limits first: with a zoom to keep, Home then returns to them,
        # and the zoom is put back on top just below.
        self._apply_view_limits()
        if preserve_view:
            self.note_natural_view()
            self.ax.set_xlim(xlim)
            self.ax.set_ylim(ylim)
            self._restore_projection_intensity_limits(projection_limits)
        self._apply_view_options()
        # Same layout pass show_spectrum ends with. Without it a rebuild
        # kept whatever subplot margins were last in force, and once the
        # slice panel (which brings its own x-axis label) changed what
        # the figure needs, the contour's own x label was clipped off
        # the bottom. Real user report: "the x-axis label got cut off".
        self._apply_layout()
        self.canvas.draw_idle()


    # ---- insets on any plot ---------------------------------------------------
    #
    # An inset is a magnified copy of part of a plot (opennmr.plotting.inset).
    # Its settings are kept here, against the plot they were drawn on, and
    # put back after every redraw: each redraw makes new axes, which would
    # otherwise lose them. A single 2D contour keeps its own list
    # (``_contour_insets``), because a contour is re-contoured from the data
    # in the region rather than copied.

    def set_inset_context(self, key) -> None:
        """Name the data on show (the selected spectrum, say), so insets
        drawn on one spectrum are not applied to another that happens to
        have the same axes."""
        if key != self._inset_context:
            self._inset_context = key
            self.canvas.draw_idle()

    def _plain_axes(self) -> list:
        return [a for a in self.figure.axes
                if a.get_gid() != inset_module.INSET_GID and getattr(a, "name", "") != "3d"]

    def insets_supported(self) -> bool:
        """Whether the plot on show can take an inset: any 2D plot made of
        lines, areas, points or images. Not a 3D plot."""
        if self._contour_spectrum is not None:
            return True
        return bool(self._plain_axes()) and not any(
            getattr(a, "name", "") == "3d" for a in self.figure.axes)

    @staticmethod
    def _inset_xrange(ax):
        """The x-extent of the data on an axes -- what tells two plots with
        the same labels apart, and unchanged when a long line is thinned for
        display or the plot is re-phased or zoomed."""
        lines = [l for l in ax.get_lines() if inset_module._copyable(l) and len(l.get_xdata()) > 1]
        if lines:
            x = np.asarray(max(lines, key=lambda l: len(l.get_xdata())).get_xdata(), dtype=float)
            if x.size and np.isfinite(x).any():
                return float(np.nanmin(x)), float(np.nanmax(x))
        if ax.images:
            extent = ax.images[0].get_extent()
            return float(min(extent[:2])), float(max(extent[:2]))
        return 0.0, 1.0

    @staticmethod
    def _inset_ranges_match(a, b) -> bool:
        span = max(abs(a[1] - a[0]), abs(b[1] - b[0]), 1e-12)
        return abs(a[0] - b[0]) <= 0.01 * span and abs(a[1] - b[1]) <= 0.01 * span

    def _inset_entry(self, index: int, ax, create: bool = False):
        """The stored insets of this plot -- ``{"xrange", "specs"}`` -- found
        by the data they were drawn on (spectrum, axes, label, x-extent)."""
        base = (self._inset_context, index, ax.get_xlabel())
        rng = self._inset_xrange(ax)
        for entry in self._inset_store.get(base, []):
            if self._inset_ranges_match(entry["xrange"], rng):
                return entry
        if not create:
            return None
        entry = {"xrange": rng, "specs": []}
        self._inset_store.setdefault(base, []).append(entry)
        return entry

    @staticmethod
    def _inset_stamp(ax):
        """Changes when the plotted values do, so an inset is redrawn."""
        lines = [l for l in ax.get_lines() if inset_module._copyable(l) and len(l.get_xdata()) > 1]
        if not lines:
            return None
        line = max(lines, key=lambda l: len(l.get_xdata()))
        y = np.asarray(line.get_ydata(), dtype=float)
        return (len(lines), int(y.size), float(np.nansum(y)))

    def _inset_target_axes(self):
        axes = self._plain_axes()
        if not axes:
            return None, -1
        index = self._inset_target if 0 <= self._inset_target < len(axes) else 0
        return axes[index], index

    def _on_draw_apply_insets(self, _event=None) -> None:
        """After a redraw, put back the insets that belong to what was
        drawn, and refresh any whose data has changed since."""
        if self._inset_applying or not self._inset_store or self._contour_spectrum is not None:
            return
        self._inset_applying = True
        try:
            changed = False
            for index, ax in enumerate(self._plain_axes()):
                entry = self._inset_entry(index, ax)
                specs = entry["specs"] if entry else None
                if not specs:
                    continue
                stamp = self._inset_stamp(ax)
                if (len(getattr(ax, "_opennmr_insets", [])) == len(specs)
                        and getattr(ax, "_opennmr_inset_stamp", None) == stamp):
                    continue
                try:
                    inset_module.draw_insets(ax, specs, resolve_host=self._inset_host_axes)
                except ValueError:
                    pass
                ax._opennmr_inset_stamp = stamp
                changed = True
            if changed:
                self.canvas.draw_idle()
        finally:
            self._inset_applying = False

    def inset_specs(self) -> list[dict]:
        """The insets on the plot on show, as complete specs
        (:data:`opennmr.plotting.inset.INSET_DEFAULTS`). For a 2D contour
        they carry ``factor`` (lowest contour) as well."""
        if self._contour_spectrum is not None:
            return [self._contour_to_generic(spec) for spec in self._contour_insets]
        ax, index = self._inset_target_axes()
        if ax is None:
            return []
        entry = self._inset_entry(index, ax)
        return [dict(spec) for spec in (entry["specs"] if entry else [])]

    def set_inset_specs(self, specs) -> None:
        """Replace the insets on the plot on show. ``ValueError`` (and the
        old insets kept) for a region with nothing in it."""
        specs = [dict(spec) for spec in (specs or [])]
        if self._contour_spectrum is not None:
            self.set_contour_insets([self._generic_to_contour(spec) for spec in specs])
            self.insetsChanged.emit()
            return
        ax, index = self._inset_target_axes()
        if ax is None:
            raise ValueError("There is no plot to put an inset on.")
        checked = [inset_module.normalise(spec) for spec in specs]
        entry = self._inset_entry(index, ax)
        old = list(entry["specs"]) if entry else None
        try:
            inset_module.draw_insets(ax, checked, resolve_host=self._inset_host_axes)
        except ValueError:
            if old is not None:
                inset_module.draw_insets(ax, old, resolve_host=self._inset_host_axes)
            self.canvas.draw_idle()
            raise
        if checked:
            self._inset_entry(index, ax, create=True)["specs"] = checked
        elif entry is not None:
            entry["specs"] = []
        ax._opennmr_inset_stamp = self._inset_stamp(ax)
        self.canvas.draw_idle()
        self.insetsChanged.emit()

    def clear_insets(self) -> None:
        self.set_inset_specs([])

    # -- every inset of a figure of several plots (Compare) ----------------
    #
    # Explicit user report, 2026-11-10: on a Compare stack, an inset made on
    # one spectrum could no longer be removed once another spectrum had been
    # selected and given an inset of its own -- the Insets window only ever
    # listed the insets of the *last* plot drawn on. These list (and set)
    # every plot's insets at once, each spec carrying the ``row`` it
    # magnifies; a spec's ``host`` places it in another row's area.

    def _inset_host_axes(self, index):
        """The plain axes ``index`` (an inset spec's ``host``), or None."""
        axes = self._plain_axes()
        return axes[index] if 0 <= int(index) < len(axes) else None

    def inset_rows(self) -> list[str]:
        """A name for each plot an inset can be drawn on, in order: the
        spectrum of each Compare row, else "Plot 1", "Plot 2"...; one name
        for a single plot or a 2D contour."""
        if self._contour_spectrum is not None:
            return ["Contour"]
        names = []
        for i, ax in enumerate(self._plain_axes()):
            label = next((line.get_label() for line in ax.get_lines()
                          if inset_module._copyable(line)
                          and not str(line.get_label()).startswith("_")), "")
            names.append(label or f"Plot {i + 1}")
        return names

    def all_inset_specs(self) -> list[dict]:
        """Every inset on the figure on show, each with ``row``: the index
        of the plot it magnifies (see :meth:`inset_rows`)."""
        if self._contour_spectrum is not None:
            return [{**spec, "row": 0} for spec in self.inset_specs()]
        out = []
        for index, ax in enumerate(self._plain_axes()):
            entry = self._inset_entry(index, ax)
            for spec in (entry["specs"] if entry else []):
                out.append({**spec, "row": index})
        return out

    def set_all_inset_specs(self, specs) -> None:
        """Replace every inset on the figure with ``specs`` (each with its
        ``row``; a spec without one goes on the plot last drawn on). Plots
        no spec names lose their insets. ``ValueError`` (with the plot's
        old insets kept) when a plot's regions have nothing in them."""
        specs = [dict(spec) for spec in (specs or [])]
        if self._contour_spectrum is not None:
            for spec in specs:
                spec.pop("row", None)
            self.set_inset_specs(specs)
            return
        axes = self._plain_axes()
        if not axes:
            if specs:
                raise ValueError("There is no plot to put an inset on.")
            return
        _ax, default_row = self._inset_target_axes()
        by_row: dict[int, list] = {}
        for spec in specs:
            row = spec.pop("row", None)
            row = default_row if row is None or not 0 <= int(row) < len(axes) else int(row)
            by_row.setdefault(row, []).append(spec)
        failures = []
        target = self._inset_target
        for index, ax in enumerate(axes):
            entry = self._inset_entry(index, ax)
            wanted = by_row.get(index, [])
            if not wanted and not (entry and entry["specs"]):
                continue
            self._inset_target = index
            try:
                self.set_inset_specs(wanted)
            except ValueError as exc:
                failures.append(str(exc))
        self._inset_target = target
        if failures:
            raise ValueError(failures[0])

    #: The 2D contour's own spec keys, and what they are called generically.
    @staticmethod
    def _contour_to_generic(spec: dict) -> dict:
        out = dict(spec)
        out["xlim"] = tuple(out.pop("xlims"))
        out["ylim"] = tuple(out.pop("ylims"))
        if "width" not in out:
            out["width"] = out["height"] = float(out.get("size", 0.4))
        out["connectors"] = bool(out.pop("mark_inset", True))
        out.pop("size", None)
        return out

    @staticmethod
    def _generic_to_contour(spec: dict) -> dict:
        out = dict(spec)
        out["xlims"] = tuple(sorted(out.pop("xlim"), reverse=True))
        out["ylims"] = tuple(sorted(out.pop("ylim")))
        out["mark_inset"] = bool(out.pop("connectors", True))
        out["size"] = float(out.get("width", 0.4))
        out.setdefault("factor", None)
        return out

    def inset_view_region(self):
        """``(xlim, ylim)`` of the target axes' current view, or ``None``."""
        if self._contour_spectrum is not None:
            return self.contour_view_region()
        ax, _index = self._inset_target_axes()
        if ax is None:
            return None
        return (tuple(float(v) for v in ax.get_xlim()), tuple(float(v) for v in ax.get_ylim()))

    @staticmethod
    def _free_inset_corner(ax, xlim, ylim, used) -> str:
        """The corner farthest from the region, and not already holding an
        inset, so a new inset never hides what it shows."""
        try:
            cx, cy = ax.transAxes.inverted().transform(
                ax.transData.transform((sum(xlim) / 2.0, sum(ylim) / 2.0)))
        except Exception:  # noqa: BLE001 -- only for placing the inset
            cx, cy = 0.5, 0.5
        centres = {"upper right": (0.75, 0.75), "upper left": (0.25, 0.75),
                   "lower left": (0.25, 0.25), "lower right": (0.75, 0.25)}
        return min(centres, key=lambda name: (
            name in used, -((centres[name][0] - cx) ** 2 + (centres[name][1] - cy) ** 2)))

    def add_inset_region(self, xlim, ylim, ax=None, **options) -> dict:
        """Add an inset magnifying ``xlim`` x ``ylim`` (data coordinates of
        the plot on show) and return its spec. Raises ``ValueError`` when
        the region holds nothing to show."""
        target = ax if ax is not None else self._inset_target_axes()[0]
        if self._contour_spectrum is None:
            axes = self._plain_axes()
            if target not in axes:
                raise ValueError("There is no plot to put an inset on.")
            self._inset_target = axes.index(target)
        current = self.inset_specs()
        used = {spec["corner"] for spec in current}
        spec = dict(inset_module.INSET_DEFAULTS)
        spec.update(options)
        spec["xlim"] = tuple(float(v) for v in xlim)
        spec["ylim"] = tuple(float(v) for v in ylim)
        if "corner" not in options and target is not None:
            spec["corner"] = self._free_inset_corner(target, spec["xlim"], spec["ylim"], used)
        if self._contour_spectrum is not None:
            spec.setdefault("factor", None)
        self.set_inset_specs(current + [spec])
        return spec

    def _finish_inset_box(self, event) -> None:
        x0, y0, ax = self._inset_start
        self._inset_start = None
        try:
            self.toolbar.remove_rubberband()
        except Exception:  # noqa: BLE001
            pass
        if event.x is None or event.y is None:
            return
        box = ax.bbox
        x1 = min(max(event.x, box.x0), box.x1)
        y1 = min(max(event.y, box.y0), box.y1)
        if abs(x1 - x0) < self._ZOOM_MIN_PX or abs(y1 - y0) < self._ZOOM_MIN_PX:
            return                      # a click, not a box: stay in the mode
        inv = ax.transData.inverted()
        (dx0, dy0), (dx1, dy1) = inv.transform((x0, y0)), inv.transform((x1, y1))
        xlim = ((min(dx0, dx1), max(dx0, dx1)) if ax.get_xlim()[0] <= ax.get_xlim()[1]
                else (max(dx0, dx1), min(dx0, dx1)))
        ylim = ((min(dy0, dy1), max(dy0, dy1)) if ax.get_ylim()[0] <= ax.get_ylim()[1]
                else (max(dy0, dy1), min(dy0, dy1)))
        try:
            self.add_inset_region(xlim, ylim, ax=ax)
        except ValueError as exc:
            self.set_inset_mode(False)
            self.insetFailed.emit(str(exc))
            return
        self.set_inset_mode(False)
        self.insetAdded.emit(len(self.inset_specs()) - 1)

    def contour_insets(self) -> list[dict]:
        """Copies of the insets drawn on the 2D contour on show."""
        return [dict(spec) for spec in self._contour_insets]

    def set_contour_insets(self, specs) -> None:
        """Draw zoomed insets (:func:`opennmr.plotting.twod.draw_insets`
        specs: ``xlims``, ``ylims``, ``corner``, ``size``, ``factor``,
        ``mark_inset``) on the single 2D contour on show, keeping its view.
        Raises ``ValueError`` (and keeps the old insets) for a region with
        no data in it."""
        old = self._contour_insets
        self._contour_insets = [dict(spec) for spec in (specs or [])]
        if self._contour_spectrum is None:
            return
        try:
            self._redraw_contour(preserve_view=True)
        except ValueError:
            self._contour_insets = old
            self._redraw_contour(preserve_view=True)
            raise

    def contour_view_region(self):
        """``(xlims, ylims)`` of the 2D contour's current view (displayed
        units), or ``None`` when no single 2D contour is on show."""
        if self._contour_spectrum is None or self.ax is None:
            return None
        return (tuple(float(v) for v in self.ax.get_xlim()),
                tuple(float(v) for v in self.ax.get_ylim()))

    def reset_view(self) -> None:
        """Back to the view a fresh load gives -- limits *and* layout.

        For a 2D this rebuilds the contour at its natural limits rather
        than going through ``toolbar.home()``: home only restores axis
        limits from the navigation stack, and that stack still holds
        Axes from before any of this canvas's own figure rebuilds, so it
        cannot restore a layout at all. Explicit user request: "be sure
        that the contour plot position resets back to the original load
        position"."""
        if self._contour_spectrum is not None:
            self._redraw_contour(preserve_view=False)
            return
        if (self._external_plot and self._external_draw is not None
                and any(getattr(ax, "name", "") == "3d" for ax in self.figure.axes)):
            # A 3D picture (Simulate's tensors): redrawn at its own
            # limits and box. The navigation stack restores x and y but
            # not z, so after a zoom Home left the box stretched along z
            # (user report, 2026-10-01). The rotation is kept.
            self._redraw_external_now(self._external_draw, keep_limits=False)
            self._home_limits = None
            return
        live = set(self.figure.axes)
        if self._home_limits and any(ax in live for ax in self._home_limits):
            for ax, (xlim, ylim, auto_x, auto_y) in self._home_limits.items():
                if ax not in live:
                    continue
                # An axis the plot left on autoscale is re-autoscaled,
                # not pinned to the number seen when recording: that
                # number can be provisional (matplotlib finishes
                # autoscaling at render, after later artists join), and
                # pinning it cut the baseline off a processing preview.
                if auto_x or auto_y:
                    ax.set_autoscalex_on(auto_x)
                    ax.set_autoscaley_on(auto_y)
                    ax.relim(visible_only=True)
                    ax.autoscale_view(scalex=auto_x, scaley=auto_y)
                if not auto_x:
                    ax.set_xlim(xlim)
                if not auto_y:
                    ax.set_ylim(ylim)
        else:
            NavigationToolbar2QT.home(self.toolbar)
        self._apply_layout()
        self.canvas.draw_idle()

    def note_natural_view(self) -> None:
        """Record every Axes' limits as the view Home returns to --
        once per fresh plot, the first time its view is about to change
        (a scroll-zoom, a drag-pan, a toolbar zoom/pan) or a redraw is
        about to re-apply a zoom the user already had.

        Public for callers outside the canvas that redraw and then put
        a kept zoom back themselves: call this between the two."""
        if self._home_limits is None and self.figure.axes:
            self._home_limits = {
                ax: (ax.get_xlim(), ax.get_ylim(), ax.get_autoscalex_on(), ax.get_autoscaley_on())
                for ax in self.figure.axes}

    # ---- exact axis limits (Plot Style ▸ Axis limits) -------------------------
    #
    # A typed x range (and, on a 2D contour, y range) that every draw re-applies,
    # so it survives a reprocess, a restyle and the next spectrum shown on the
    # same tab. Deliberately not part of style_settings(): a range belongs to the
    # data on show, not to the look, so it is neither saved as a per-type default
    # nor pushed by "Apply to all plots" (a 1H window saved as a default would
    # crop the next 13C spectrum to nothing). Home and a double-click return to
    # the typed range; clearing it returns to the full view.

    #: Pass as ``x``/``y`` of :meth:`set_view_limits` to leave that axis as it is.
    KEEP_LIMITS = object()

    @staticmethod
    def _clean_limit_pair(pair):
        """``(low, high)`` ordered, either side ``None`` for automatic; ``None``
        when both are (no limit). Order does not matter to the caller: the plot
        keeps its own direction (ppm still runs high to low)."""
        if pair is None:
            return None
        try:
            low, high = pair
        except (TypeError, ValueError):
            raise ValueError("an axis limit is a (low, high) pair; None on either side means automatic") from None
        cleaned = []
        for value in (low, high):
            if value is None:
                cleaned.append(None)
                continue
            value = float(value)
            if not np.isfinite(value):
                raise ValueError(f"axis limit {value!r} is not a finite number")
            cleaned.append(value)
        low, high = cleaned
        if low is not None and high is not None:
            if low == high:
                raise ValueError(f"the two axis limits are both {low:g}: the range would be empty")
            low, high = min(low, high), max(low, high)
        if low is None and high is None:
            return None
        return (low, high)

    def view_limits(self) -> dict:
        """The typed limits, ``{"x": pair-or-None, "y": pair-or-None}``."""
        return {"x": self._view_x, "y": self._view_y}

    def view_limit_support(self) -> tuple:
        """``(x, y)``: whether the plot on show takes a typed x range, and a typed
        y range. Both on a 2D contour, x only on a 1D spectrum or a Compare
        stack/overlay, neither on anything else (a FID, a fit, a curve...)."""
        if not self._view_limit_axes():
            return False, False
        return True, self._contour_spectrum is not None

    def view_limits_unit_text(self) -> str:
        """The unit the plot's axis is in now, as the limits are to be typed."""
        return self._UNIT_TEXT.get(self._axis_unit, self._axis_unit)

    def view_limits_note(self) -> str:
        """Why the typed limits are not doing what was asked, or ``""``."""
        if self._view_x is None and self._view_y is None:
            return ""
        if self._view_unit != self._axis_unit:
            was = self._UNIT_TEXT.get(self._view_unit, self._view_unit)
            return (f"These limits were typed for an axis in {was}; the plot is in "
                    f"{self.view_limits_unit_text()}, so they are not applied. Type them again "
                    f"in {self.view_limits_unit_text()}, or switch the axis back.")
        return self._view_limits_note

    def set_view_limits(self, x=KEEP_LIMITS, y=KEEP_LIMITS) -> None:
        """Fix the plot's x range (and, on a 2D contour, y range) exactly.

        ``x`` / ``y``: ``(low, high)`` in the unit the axis is showing now (ppm,
        Hz...), in either order -- the plot keeps its direction, so
        ``x=(-2, 12)`` still draws 12 on the left. ``None`` on either side leaves
        that end automatic; ``None`` for the whole pair clears that axis's limit
        and restores the full view. Omit an argument to leave that axis alone.
        Raises ``ValueError`` for a non-finite number or an empty range."""
        if x is not self.KEEP_LIMITS:
            x = self._clean_limit_pair(x)
        if y is not self.KEEP_LIMITS:
            y = self._clean_limit_pair(y)
        if x is not self.KEEP_LIMITS:
            self._view_x = x
        if y is not self.KEEP_LIMITS:
            self._view_y = y
        self._view_unit = self._axis_unit
        self._apply_view_limits()
        self.canvas.draw_idle()
        self.viewLimitsChanged.emit()

    def clear_view_limits(self) -> None:
        """Back to the full view on both axes."""
        self.set_view_limits(x=None, y=None)

    def forget_view_limits(self) -> bool:
        """Drop typed limits without redrawing: for a different spectrum about
        to be drawn, which must open on its own automatic range (typed limits
        belong to the spectrum they were set on). Returns whether there were
        any."""
        had = self._view_x is not None or self._view_y is not None
        self._view_x = None
        self._view_y = None
        self._natural_limits = None
        self._view_limits_note = ""
        if had:
            self._home_limits = None
        return had

    def _view_limit_axes(self) -> list:
        """The Axes a typed range goes on: the contour's main Axes (its
        projection strips share them), every row of a Compare stack, or the one
        Axes of a 1D spectrum / Compare overlay. Nothing for any other view."""
        live = self.figure.axes
        if self._contour_spectrum is not None:
            return [self.ax] if self.ax in live else []
        if self._compare_axes:
            return [ax for ax in self._compare_axes if ax in live]
        if self._spectrum_panels:
            # Compare > Panels: the panels that share the first one's x axis
            # (a 31P and a 195Pt panel cannot take the same typed range).
            panels = [ax for ax in self._spectrum_panels if ax in live]
            if not panels:
                return []
            label = panels[0].get_xlabel()
            return [ax for ax in panels if ax.get_xlabel() == label
                    and getattr(ax, "_opennmr_panel_kind", "1d") in ("1d", "2d", "contour")
                    or ax is panels[0]]
        if ((self._current_1d_spectrum is not None or self._plot_kind == "compare")
                and self.ax in live and self.ax.lines):
            return [self.ax]
        return []

    @staticmethod
    def _limits_in_orientation(current, natural, requested):
        """``requested`` (low, high; ``None`` = take the plot's own end) as limits
        for an axis currently running ``current``, keeping its direction."""
        low, high = requested
        natural_low, natural_high = sorted(natural)
        low = natural_low if low is None else low
        high = natural_high if high is None else high
        if low > high:
            low, high = high, low
        if low == high:
            return None
        return (high, low) if current[0] > current[1] else (low, high)

    def _apply_view_limits(self) -> None:
        """Put the typed limits on the plot on show (and take them off an axis
        whose limit was cleared). Called at the end of every draw that supports
        them, and by :meth:`set_view_limits`. Does nothing when there are no
        limits and none to undo."""
        if self._view_x is None and self._view_y is None and self._natural_limits is None:
            return
        self._view_limits_note = ""
        axes = self._view_limit_axes()
        if not axes:
            return
        if self._view_unit != self._axis_unit:
            return                      # view_limits_note() says so
        if self._natural_limits is None:
            # Before the first change: what a fresh draw gave (or, when a zoom is
            # already on, the view Home would return to).
            home = self._home_limits or {}
            self._natural_limits = {
                ax: ((home[ax][0], home[ax][1]) if ax in home else (ax.get_xlim(), ax.get_ylim()))
                for ax in axes}
        main = axes[0]
        natural = self._natural_limits
        # The crosshair's two-point lines live on this Axes and would be read as
        # data by the refit below; they are rebuilt on the next mouse move.
        self._remove_crosshair()

        for ax in axes:
            natural_x, _ = natural.get(ax, natural.get(main, (ax.get_xlim(), ax.get_ylim())))
            if self._view_x is None:
                target = natural_x
            else:
                target = self._limits_in_orientation(ax.get_xlim(), natural_x, self._view_x) or natural_x
            ax.set_xlim(target)
        if self._contour_spectrum is not None:
            _, natural_y = natural.get(main, (main.get_xlim(), main.get_ylim()))
            if self._view_y is None:
                target = natural_y
            else:
                target = self._limits_in_orientation(main.get_ylim(), natural_y, self._view_y) or natural_y
            main.set_ylim(target)
        elif self._spectrum_panels:
            linked_y = bool(self._spectrum_panel_link is not None and self._spectrum_panel_link.y)
            if not linked_y:
                for ax in axes:
                    if getattr(ax, "_opennmr_panel_kind", "1d") == "1d" and ax.lines:
                        if self._view_x is None:
                            self._fit_y_to_data(ax)
                        else:
                            self._fit_y_to_data(ax, y_range=self._data_y_range(
                                ax, x_window=tuple(sorted(ax.get_xlim()))))
        elif self._current_1d_spectrum is not None and self._compare_axes is None:
            if self._view_x is None:
                self._fit_y_to_data(main)
            else:
                # The intensity axis fits what is inside the window.
                self._fit_y_to_data(main, y_range=self._data_y_range(
                    main, x_window=tuple(sorted(main.get_xlim()))))

        # Said when a typed range misses the data altogether (the plot would be
        # an empty frame with no hint why).
        data_x = sorted(natural.get(main, (main.get_xlim(), None))[0])
        for name, pair, data in (("x", self._view_x, data_x),
                                  ("y", self._view_y,
                                   sorted(natural[main][1]) if self._contour_spectrum is not None
                                   and main in natural else None)):
            if pair is None or data is None:
                continue
            low = data[0] if pair[0] is None else pair[0]
            high = data[1] if pair[1] is None else pair[1]
            if high < data[0] or low > data[1]:
                self._view_limits_note = (f"The {name} range {min(low, high):g} to {max(low, high):g} "
                                          f"lies outside the data ({data[0]:g} to {data[1]:g}): "
                                          f"the plot is empty.")
        if self._view_x is None and self._view_y is None:
            # Everything cleared: back to a plain fresh view.
            self._natural_limits = None
            self._home_limits = None
            return
        # Home and a double-click return to the typed range.
        self._home_limits = {ax: (ax.get_xlim(), ax.get_ylim(), False, False)
                             for ax in self.figure.axes}

    def _on_press(self, event) -> None:
        if self._begin_label_drag(event):
            return
        # getattr, not a plain attribute access -- a real matplotlib
        # MouseEvent always has .xdata (possibly None), but this is no
        # longer gated on _fit_mode first (see below), so a fake event
        # missing the attribute entirely (some existing tests' own
        # minimal doubles, built before this needed to handle the
        # non-fit-mode case too) must not raise AttributeError here.
        # Also with the toolbar's Zoom or Pan button down (user report,
        # 2026-10-17: zoom, then double-click, and the view stayed) --
        # only the click modes (Measure, Width, picking, regions), where
        # two quick clicks are real input, keep a double-click.
        if (event.button == 1 and getattr(event, "dblclick", False)
                and self._double_click_resets
                and not self._click_mode_active() and event.inaxes is not None):
            # Double-click: back to the full view (Settings ▸ Double-click
            # resets the view; explicit user request, 2026-10-15). The
            # first click of the pair already started a (zero-size, so
            # no-op) zoom or pan; drop it. On a stacked Compare the first
            # click also toggled a row's selection -- clear it, so a
            # double-click never leaves a row tinted.
            self._zoom_start = None
            self._pan_start = None
            self._compare_click_start = None
            # The toolbar's own zoom/pan press from these clicks: drop it,
            # rubber band and all, so its release cannot zoom afterwards.
            if self.toolbar.mode != "":
                self._cancel_navigation_drag()
            if self._compare_axes and self._selected_compare_index is not None:
                self.set_compare_selection(None)
                self.compareSelectionCleared.emit()
            self.reset_view()
            return
        if event.button == 3:
            # Which Axes the right-click landed in, for the context menu
            # (a projection strip gets its own style entry). Reset on
            # every right-click, so a click outside every Axes never
            # inherits the previous one's.
            self._last_right_click_axes = getattr(event, "inaxes", None)
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
        # Captured *before* the inaxes guard below, so a click landing
        # in a stacked Compare row's own margins -- its y-tick strip,
        # or the gap to the next row -- still selects that row. Only
        # the data rectangle is "inaxes", but the whole horizontal band
        # belongs to the row visually, which is where people click
        # (explicit user request: the click boxes should "extend all
        # the way to the next stacked spectrum... to include both the
        # axis and the plotting spaces"). Resolved to a row by y band
        # on release, see _compare_row_index_at_y.
        if (
            self._compare_axes and event.button == 1 and self.toolbar.mode == ""
            and not self._pick_mode and not self._region_select_mode and not self._measurement_mode
            and not self._width_mode and not self._inset_mode
            and event.x is not None and event.y is not None
        ):
            self._compare_click_start = (event.x, event.y, event.inaxes)
        if event.inaxes is None or event.button != 1 or self.toolbar.mode != "":
            return
        if self._pick_mode:
            if event.xdata is not None:
                self.peakPicked.emit(float(event.xdata))
            return
        if self._inset_mode:
            if (event.xdata is not None and event.ydata is not None
                    and event.inaxes.get_gid() != inset_module.INSET_GID
                    and (self._contour_spectrum is None or event.inaxes is self.ax)):
                self._inset_start = (event.x, event.y, event.inaxes)
            return
        if self._region_select_mode:
            if event.xdata is not None:
                self._region_select_start = float(event.xdata)
                self._region_select_ax = event.inaxes
                self._update_region_select_patch(event.xdata)
            return
        if self._measurement_mode:
            if event.xdata is not None and event.ydata is not None:
                self._record_measurement_point(
                    event.inaxes, float(event.xdata), float(event.ydata)
                )
            return
        if self._width_mode:
            if event.xdata is not None and event.ydata is not None:
                self._measure_width_at(event.inaxes, float(event.xdata), float(event.ydata))
            return
        self.note_natural_view()
        if self._drag_mode == "zoom" and getattr(event.inaxes, "name", "") != "3d":
            self._zoom_start = (event.x, event.y, event.inaxes)
            return
        self._pan_start = (event.x, event.y, event.inaxes.get_xlim(), event.inaxes.get_ylim(), event.inaxes)

    # ---- the default drag: pan or box zoom ----------------------------------

    DRAG_MODES = ("zoom", "pan")
    #: A box narrower than this (pixels) along one axis zooms the other
    #: axis only -- matplotlib's own zoom tool does the same.
    _ZOOM_MIN_PX = 5

    def drag_mode(self) -> str:
        return self._drag_mode

    def set_double_click_resets(self, on: bool) -> None:
        """Whether a left double-click on a plot resets its view (Home)."""
        self._double_click_resets = bool(on)

    def double_click_resets(self) -> bool:
        return self._double_click_resets

    def set_drag_mode(self, mode: str) -> None:
        """What a plain left-drag does: ``"zoom"`` (draw a box, zoom to
        it) or ``"pan"``. The toolbar shows it by checking that button."""
        if mode not in self.DRAG_MODES:
            raise ValueError(f"drag mode must be one of {self.DRAG_MODES}, not {mode!r}")
        self._drag_mode = mode
        self.toolbar._update_buttons_checked()

    def _click_mode_active(self) -> bool:
        return bool(getattr(self, "_pick_mode", False) or getattr(self, "_region_select_mode", False)
                    or getattr(self, "_measurement_mode", False)
                    or getattr(self, "_width_mode", False)
                    or getattr(self, "_inset_mode", False))

    def _zoom_drag_to(self, event) -> None:
        x0, y0, ax = self._zoom_start
        self._zoom_start = None
        try:
            self.toolbar.remove_rubberband()
        except Exception:  # noqa: BLE001 -- a stale rubber band must never block the zoom
            pass
        if event.x is None or event.y is None:
            return
        box = ax.bbox
        x1 = min(max(event.x, box.x0), box.x1)
        y1 = min(max(event.y, box.y0), box.y1)
        wide, tall = abs(x1 - x0) >= self._ZOOM_MIN_PX, abs(y1 - y0) >= self._ZOOM_MIN_PX
        if not (wide or tall):
            return                      # a click, not a box
        inv = ax.transData.inverted()
        (dx0, dy0), (dx1, dy1) = inv.transform((x0, y0)), inv.transform((x1, y1))

        def keep_direction(lim, a, b):
            lo, hi = sorted((a, b))
            return (hi, lo) if lim[0] > lim[1] else (lo, hi)

        if wide:
            ax.set_xlim(keep_direction(ax.get_xlim(), dx0, dx1))
        if tall:
            ax.set_ylim(keep_direction(ax.get_ylim(), dy0, dy1))
        self.canvas.draw_idle()

    def _on_motion(self, event) -> None:
        if self._label_drag is not None:
            self._drag_label(event)
            return
        self._ensure_crosshair()
        # Before the drag branches below, not after: they return early,
        # and the cursor has to be resolved on every move, not only on
        # the ones that fall through to the end.
        self._refresh_cursor(event)
        if self._zoom_start is not None and event.x is not None and event.y is not None:
            x0, y0, ax = self._zoom_start
            # The box is clipped to the Axes it started in, as
            # matplotlib's own zoom tool clips it.
            box = ax.bbox
            x1 = min(max(event.x, box.x0), box.x1)
            y1 = min(max(event.y, box.y0), box.y1)
            self.toolbar.draw_rubberband(event, x0, y0, x1, y1)
            return
        if self._inset_start is not None and event.x is not None and event.y is not None:
            x0, y0, ax = self._inset_start
            box = ax.bbox
            self.toolbar.draw_rubberband(
                event, x0, y0, min(max(event.x, box.x0), box.x1), min(max(event.y, box.y0), box.y1))
            return
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

    #: How far the 3D box may shrink to keep its labels in the figure,
    #: and the step it shrinks by.
    _3D_MIN_ZOOM = 0.5
    _3D_ZOOM_STEP = 0.92

    def _fit_3d_layout(self, threes) -> list:
        """Shrink each 3D box until its axis labels lie inside the
        figure (down to :attr:`_3D_MIN_ZOOM`), refitting the ticks at
        every step. Returns the labels still outside.

        mplot3d places its axis labels only while drawing, so each
        check draws the three axes alone (ticks and labels: a few ms)
        rather than the whole picture (a frequency surface takes ~0.6 s)."""
        try:
            renderer = self.figure.canvas.get_renderer()
        except AttributeError:
            return []
        bounds, tol = self.figure.bbox, 2.0
        scale = 1.8294640721620434 * 25 / 24         # mplot3d's own box scale at zoom 1
        # A legend placed outside the box (Plot Style ▸ Legend position:
        # outside right / custom) gets its room first.
        from ...plotting.legend_place import make_room_for_legend
        for ax in threes:
            try:
                make_room_for_legend(ax, renderer=renderer)
            except Exception:  # never let a layout check break a redraw
                pass
        outside = []
        for _step in range(12):
            outside = []
            for ax in threes:
                try:
                    ax.M = ax.get_proj()
                    ax.invM = np.linalg.inv(ax.M)
                    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
                        axis.draw(renderer)
                except Exception:  # never let a layout check break a redraw
                    return []
                for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
                    label = axis.label
                    if not (label.get_text() and label.get_visible()):
                        continue
                    box = label.get_window_extent(renderer)
                    if (box.x0 < bounds.x0 - tol or box.y0 < bounds.y0 - tol
                            or box.x1 > bounds.x1 + tol or box.y1 > bounds.y1 + tol):
                        outside.append((ax, label.get_text()))
            if not outside:
                break
            shrunk = False
            for ax in {a for a, _t in outside}:
                aspect = np.asarray(ax._box_aspect, dtype=float)
                zoom = float(np.linalg.norm(aspect)) / scale
                if zoom * self._3D_ZOOM_STEP < self._3D_MIN_ZOOM:
                    continue
                ax.set_box_aspect(tuple(aspect), zoom=zoom * self._3D_ZOOM_STEP)
                self._fit_3d_ticks(ax)
                shrunk = True
            if not shrunk:
                break
        return sorted({text for _a, text in outside})

    def _fit_3d_ticks(self, ax) -> dict:
        """As many ticks on the 3D ``ax`` as fit at the current text size,
        view and limits (see tensor_views.fit_3d_ticks)."""
        from ...notebook.tensor_views import fit_3d_ticks
        try:
            return fit_3d_ticks(ax, self._AXIS_LABELSIZE, font_family=self._font_family)
        except Exception:  # never let tick fitting break a draw
            return {}

    def _refit_3d_ticks_after_release(self, event) -> None:
        """A rotation or toolbar zoom of a 3D picture changes how long
        each axis looks on screen, so how many tick labels fit."""
        threes = [ax for ax in self.figure.axes if getattr(ax, "name", "") == "3d"]
        if threes and getattr(event, "inaxes", None) in threes:
            for ax in threes:
                self._reset_3d_box(ax)
            self._fit_3d_layout(threes)
            self._apply_colorbar_gap()
            self.canvas.draw_idle()

    def _on_release(self, event) -> None:
        if self._label_drag is not None:
            self._end_label_drag(event)
            return
        self._refit_3d_ticks_after_release(event)
        if self._zoom_start is not None:
            self._zoom_drag_to(event)
        if self._inset_start is not None:
            self._finish_inset_box(event)
            return
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
        if self._compare_click_start is not None and self._compare_axes:
            x0, y0, start_ax = self._compare_click_start
            if (
                event.x is not None and event.y is not None
                and abs(event.x - x0) <= self._CLICK_DRAG_TOLERANCE_PX
                and abs(event.y - y0) <= self._CLICK_DRAG_TOLERANCE_PX
            ):
                # The Axes the press actually landed in wins whenever
                # there is one -- it is exact, and matplotlib already
                # did the hit-testing. The y-band lookup is only for
                # resolving a press with no `inaxes` at all, i.e. one
                # in a row's margins (its tick strip, or the gap to the
                # next row), which is the case this whole path exists
                # to add.
                target = start_ax if start_ax in self._compare_axes else None
                if target is None:
                    index = self._compare_row_index_at_y(y0)
                    target = self._compare_axes[index] if index is not None else None
                if target is not None:
                    self._toggle_compare_selection(target)
        self._compare_click_start = None
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

    def _compare_row_index_at_y(self, y: float) -> int | None:
        """Which "Stacked" Compare row owns display pixel row ``y``.

        Each row owns a full horizontal band, not just its own data
        rectangle: the band runs to the midpoint of the gap to the
        next row (so a click in a row's tick strip, or between two
        rows, lands on the one it visually belongs to), and the
        outermost bands run to the figure's own edges. Explicit user
        request -- the clickable area should "extend all the way to
        the next stacked spectrum... to include both the axis and the
        plotting spaces". With the stack's own ``hspace=0`` the gaps
        are usually zero and this reduces to the data rectangles
        themselves, but it stays correct if that ever changes.

        Returns ``None`` when no Compare stack is shown, or when ``y``
        is outside the whole stack (e.g. below the shared x-axis
        label), rather than snapping a far-away click onto the nearest
        row."""
        if not self._compare_axes:
            return None
        bounds = [(i, ax.get_window_extent()) for i, ax in enumerate(self._compare_axes)]
        bounds.sort(key=lambda pair: pair[1].y0)
        fig_bbox = self.figure.bbox
        if not (fig_bbox.y0 <= y <= fig_bbox.y1):
            return None
        for rank, (index, bbox) in enumerate(bounds):
            lower = fig_bbox.y0 if rank == 0 else 0.5 * (bounds[rank - 1][1].y1 + bbox.y0)
            upper = (
                fig_bbox.y1 if rank == len(bounds) - 1
                else 0.5 * (bbox.y1 + bounds[rank + 1][1].y0)
            )
            if lower <= y <= upper:
                return index
        return None

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
        if self._selected_compare_index is None:
            self.compareSelectionCleared.emit()     # the Compare list lets go too
        else:
            # The Compare list takes the same row as its current one, so
            # Move Up/Down (and the arrow keys) act on what was clicked.
            self.compareRowSelected.emit(int(self._selected_compare_index))

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

    def _exec_context_menu(self, menu, pos):
        """Show ``menu`` modally and return the chosen action.

        A one-line seam, so a test can inspect what the right-click menu
        offers without a real modal menu blocking the run. Patching
        ``QMenu.exec`` itself does not work -- PySide6 resolves it
        through the C++ binding, so the real menu opens anyway and the
        headless suite hangs rather than fails.
        """
        return menu.exec(self.canvas.mapToGlobal(pos))

    #: Emitted when the right-click menu's "Plot Style…" entry is
    #: chosen. The canvas knows the gesture happened; only MainWindow
    #: owns the dialog and the controls in it.
    plotStyleRequested = Signal()

    #: A double-click cleared a stacked Compare row's selection, so the
    #: Compare list can drop its current row too.
    compareSelectionCleared = Signal()
    #: A panel label was dragged to a new place: ``(panel, dx, dy)`` -- the
    #: 0-based panel and the label's offset from that panel's top-left
    #: corner, in points.
    panelLabelMoved = Signal(int, float, float)
    #: Emitted with the plotted-row position when a click on the stacked
    #: Compare plot selects a row (the list follows; see MainWindow).
    compareRowSelected = Signal(int)

    #: Emitted with the component labels after a fit view is drawn
    #: (:meth:`show_slice_fit`), so the component-display controls can
    #: list them.
    fitComponentsShown = Signal(list)

    #: The right-click "Components ▸ Colours and highlighting…" entry.
    fitDisplayOptionsRequested = Signal()

    #: The right-click "Keep This Fit…" entry, on any fit or curve fit.
    keepFitRequested = Signal()

    #: The right-click "Keep This Simulation in Open Experiments" entry,
    #: with the simulated spectrum on show (see :attr:`simulation_spectrum`).
    keepSimulationRequested = Signal(object)

    #: The simulated spectrum this plot shows, set by whoever drew it, so
    #: the right-click menu can keep it (explicit user request: simulations
    #: kept as Open experiments, to compare with data). Cleared by every
    #: rebuild.
    simulation_spectrum = None

    #: The figure was cleared for a new plot (emitted before it is
    #: drawn): a listener that depends on what is on show re-checks once
    #: the event loop is back (QTimer.singleShot(0, ...)).
    figureCleared = Signal()
    #: ``(dpi, width_in, height_in)`` chosen in the Save Figure preview
    #: with "Keep for later exports" ticked -- the main window copies it
    #: into Plot Style ▸ Export so every canvas follows.
    exportSettingsKept = Signal(float, float, float)
    #: The typed axis limits (Plot Style ▸ Axis limits) changed, were applied
    #: to a fresh draw, or stopped applying (the axis unit changed): the
    #: panel re-reads :meth:`view_limits` and :meth:`view_limits_note`.
    viewLimitsChanged = Signal()
    #: The display unit of the x-axis changed ("ppm", "hz", "khz", "mm"),
    #: by the Axis Units menu or :meth:`set_axis_unit` -- so a Units box
    #: elsewhere (Compare's) can follow it.
    axisUnitChanged = Signal(str)

    #: Who drew the figure on show, for its right-click plot options
    #: (``register_plot_options``): set by the drawer after drawing,
    #: cleared by every rebuild, kept through a restyle's redraw.
    plot_owner = None

    def register_plot_options(self, owner: str, label: str, callback, tooltip: str = "") -> None:
        """Offer ``label`` on the right-click menu, beside Plot Style…,
        while the plot on show is ``owner``'s -- the plot's own display
        options (explicit user request, 2026-10-16: formatting a plot is
        a right-click on it, not a section of the side panel).

        Owners set by the canvas itself: ``"curve_fit"`` (a relaxation/
        diffusion curve fit is on show) and ``"fit_components"`` (a
        spectral fit with components). Others are named by whoever draws
        (``canvas.plot_owner = owner`` after drawing)."""
        self._plot_options[owner] = (label, tooltip, callback)

    def current_plot_owners(self) -> list:
        """The owners whose plot options apply to what is on show."""
        owners = []
        if self.plot_owner:
            owners.append(self.plot_owner)
        if self._curve_fit_shown is not None or self._curve_fits_shown:
            owners.append("curve_fit")
        if self._slice_fit_active and self._slice_fit_labels:
            owners.append("fit_components")
        if self.insets_supported():
            owners.append("inset")
        return [o for o in owners if o in self._plot_options]

    #: Emitted after a style setting was changed from this canvas's own
    #: right-click menu (e.g. a curve plot's Top/Right Spine), so the
    #: Plot Style controls can re-read it.
    styleEditedFromMenu = Signal()

    #: Emitted with ``"1d"``, ``"2d"``, ``"pseudo2d"`` or ``"compare"``
    #: when this canvas is about to draw a *different kind* of plot
    #: from the last one -- before drawing it, so a listener can hand
    #: it that kind's saved default style (see
    #: :meth:`apply_style_defaults`) and the plot is drawn in it first
    #: time. Not emitted for another plot of the same kind: that would
    #: undo whatever the user has changed on this plot since.
    plotKindChanged = Signal(str)
    #: The set of lines on the plot changed (a redraw added or removed
    #: one), so a Lines list showing them should be rebuilt.
    linesChanged = Signal()

    #: The plot kinds :attr:`plotKindChanged` reports.
    PLOT_KINDS = ("1d", "2d", "pseudo2d", "compare", "curve", "tensor")

    #: Right-click ▸ Tensor Plot Style… on a tensor picture.
    tensorStyleRequested = Signal()
    #: The tensor-picture style changed (Tensor Plot Style, or Plot
    #: Style's colour maps / grid carried over to a tensor on show).
    tensorStyleChanged = Signal()

    def _show_style_menu_from_button(self) -> None:
        """The right-click menu, opened from the "Plot style" toolbar button."""
        self._last_right_click_axes = None
        self._last_right_click_xy = None
        button = self._style_button
        below = button.mapToGlobal(button.rect().bottomLeft())
        self._on_context_menu(self.canvas.mapFromGlobal(below))

    def _on_context_menu(self, pos) -> None:
        menu = QMenu(self)

        # Right-clicked on a projection strip: its own style first --
        # what the click was aimed at. Explicit user request: "control
        # the plot style of the projections by right clicking the
        # projection specifically".
        strip = (self._projection_dimension_of(self._last_right_click_axes)
                 if self._contour_spectrum is not None else None)
        if strip is not None:
            name = "F2 (top)" if strip == "f2" else "F1 (left)"
            strip_action = menu.addAction(f"{name} Projection Style…")
            strip_action.setToolTip(
                "Colour, line width and style, size, or hide this projection.")
            strip_action.triggered.connect(
                lambda _checked=False, d=strip: self._prompt_projection_style(d))
            hide_action = menu.addAction(f"Hide {name} Projection")
            hide_action.triggered.connect(
                lambda _checked=False, d=strip: self.set_projection_style(d, visible=False))
            menu.addSeparator()

        style_action = menu.addAction("Plot Style…")
        style_action.setToolTip("Line width, fonts, ticks, colours and export settings for this plot.")
        style_action.triggered.connect(self.plotStyleRequested.emit)
        for owner in self.current_plot_owners():
            label, tooltip, callback = self._plot_options[owner]
            options_entry = menu.addAction(label)
            if tooltip:
                options_entry.setToolTip(tooltip)
            options_entry.triggered.connect(lambda _c=False, cb=callback: cb())
        if self._external_plot and self._plot_kind == "tensor":
            tensor_action = menu.addAction("Tensor Plot Style…")
            tensor_action.setToolTip("Colour map, colour bar, principal axes, colours, opacity "
                                     "and view of this tensor picture.")
            tensor_action.triggered.connect(self.tensorStyleRequested.emit)
        menu.addSeparator()

        legend_action = menu.addAction("Show Legend")
        legend_action.setCheckable(True)
        legend_action.setChecked(self._legend_visible)

        if self.simulation_spectrum is not None:
            keep_sim = menu.addAction("Keep This Simulation in Open Experiments")
            keep_sim.setToolTip("List the simulated spectrum under Open experiments, to compare "
                                "it with data or other simulations (Compare), process or save it.")
            keep_sim.triggered.connect(
                lambda _c=False: self.keepSimulationRequested.emit(self.simulation_spectrum))
        if self.shown_fit() is not None:
            keep_action = menu.addAction("Keep This Fit…")
            keep_action.setToolTip("Save this fit with the spectrum, to reopen and compare "
                                   "with other fits later (Fit ▸ Kept fit models).")
            keep_action.triggered.connect(self.keepFitRequested.emit)
        # Fit views: quick component display, on the plot itself. The
        # full controls (highlight, colours) are right-click ▸ Fit Plot: Components and Colours…
        if self._slice_fit_active and self._slice_fit_labels:
            comp_menu = menu.addMenu("Components")
            style = self._fit_component_style
            for text, mode in (("Show automatically", "auto"), ("Show all", "all"),
                               ("Hide all", "none"), ("Highlight selected only", "highlight")):
                action = comp_menu.addAction(text)
                action.setCheckable(True)
                action.setChecked(style["show"] == mode)
                action.triggered.connect(
                    lambda _c=False, m=mode: (self.set_fit_component_style(show=m),
                                              self.styleEditedFromMenu.emit()))
            comp_menu.addSeparator()
            fill_action = comp_menu.addAction("Fill components")
            fill_action.setCheckable(True)
            fill_action.setChecked(bool(style["fill"]))
            fill_action.triggered.connect(
                lambda checked: (self.set_fit_component_style(fill=bool(checked)),
                                 self.styleEditedFromMenu.emit()))
            comp_menu.addSeparator()
            options_action = comp_menu.addAction("Colours and highlighting…")
            options_action.triggered.connect(self.fitDisplayOptionsRequested.emit)

        # Curves & profiles: top/right spines straight from the plot
        # (explicit user request -- the Plot Style checkboxes alone were
        # too far from the plot to be found).
        if [ax for ax in self._curve_axes if ax in self.figure.axes]:
            for text, key, value in (("Top Spine", "top_spine", self._top_spine_visible),
                                     ("Right Spine", "right_spine", self._right_spine_visible)):
                action = menu.addAction(text)
                action.setCheckable(True)
                action.setChecked(value)
                action.setToolTip("Draw this axis line (with ticks) on the curve plot.")
                action.triggered.connect(
                    lambda checked, k=key: (self.set_style(**{k: bool(checked)}),
                                            self.styleEditedFromMenu.emit()))

        lock_action = menu.addAction("Lock Aspect Ratio")
        lock_action.setCheckable(True)
        lock_action.setChecked(self._aspect_ratio is not None)

        set_ratio_action = menu.addAction("Set Aspect Ratio…")

        sr_action = None
        if self._current_1d_spectrum is not None:
            sr_action = menu.addAction("Set Reference Shift (Hz)…")

        # The Hz entry above is gated on a 1D view, so a 2D contour had
        # no referencing action at all -- and an Hz offset is the wrong
        # thing to ask for on one anyway, since what a user actually
        # knows is where a peak *should* be. This asks for that instead:
        # right-click the peak, type its true shift, and the SR follows.
        # The click needs no unit conversion because this is offered
        # only while ppm is displayed -- see the guard below.
        # ...and only while the contour is actually showing ppm: the
        # click is read in whatever unit is displayed, so asking for a
        # shift in ppm against an mm axis would be nonsense.
        calibrate_action = None
        if (self._contour_spectrum is not None and self._last_right_click_xy is not None
                and self._axis_unit == "ppm"):
            calibrate_action = menu.addAction("Set Reference from Cursor (ppm)…")
        elif (self._current_1d_spectrum is not None and self._last_right_click_xy is not None
                and self._axis_unit == "ppm"):
            # The same on a 1D (explicit user request, 2026-10-15): type
            # the true shift of the peak under the cursor; SR follows.
            calibrate_action = menu.addAction("Calibrate Shift Here (ppm)…")
            calibrate_action.setToolTip(
                "Type the true chemical shift of the peak under the cursor; "
                "the reference (SR) is set so it lands there.")

        # Relative integrals set on the plot (explicit user request,
        # 2026-11-10: "a way to calibrate relative integrals on the plot"):
        # right-click a region, type what it should read, and every other
        # region's label follows. Offered wherever integration regions are
        # shaded on a 1D spectrum (or their labels are on).
        if (self._current_1d_spectrum is not None and self._last_right_click_xy is not None
                and self.ax is not None
                and (self._integral_label_items
                     or any(p.get_gid() == self._REGION_PATCH_GID for p in self.ax.patches))):
            integral_action = menu.addAction("Calibrate Integral Here…")
            integral_action.setToolTip(
                "Set the integral of the region under the cursor (e.g. 3 for a CH₃); every "
                "other region's integral label is scaled to match. Turns the integral labels "
                "on.")
            integral_action.triggered.connect(
                lambda _c=False, x=float(self._last_right_click_xy[0]):
                self.integralCalibrationRequested.emit(x))
            if self.integral_calibrated:
                reset_integral_action = menu.addAction("Reset Integral Calibration")
                reset_integral_action.setToolTip("Back to the smallest integral = 1.")
                reset_integral_action.triggered.connect(
                    lambda _c=False: self.integralCalibrationResetRequested.emit())

        unit_actions = {}
        if (self._current_1d_spectrum is not None or self._contour_spectrum is not None
                or self._compare_spectra):
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

        dosy_spectrum_action = dosy_profile_action = dosy_levels_action = None
        dosy_panel_action = dosy_clear_slices_action = None
        if self._dosy_result is not None and self._last_right_click_xy is not None:
            shift, diffusion = self._last_right_click_xy
            dosy_spectrum_action = menu.addAction(
                f"Extract Spectrum at D = {diffusion:.3g} m²/s")
            dosy_spectrum_action.setToolTip(
                "The row of the map at this diffusion coefficient – the spectrum "
                "of whatever is diffusing at that rate, as its own 1D spectrum."
            )
            # The other of the two things worth doing with a row, and a
            # separate entry rather than a mode dialog: a 2D contour's
            # own Extract Slice… needs a dialog anyway (it has an axis
            # and an index to choose), where a DOSY row is fully
            # specified by the click that opened this menu. Explicit
            # user request: "when I extract a line from the DOSY, I
            # would also like the option to plot alongside with the
            # dashed line like in normal 2D contours".
            dosy_panel_action = menu.addAction(
                f"Show Spectrum at D = {diffusion:.3g} m²/s Beside the Map")
            dosy_panel_action.setToolTip(
                "Keeps the map, draws a dashed line across it at this diffusion "
                "coefficient, and stacks the spectrum there in a panel to the "
                "right – the same as a 2D contour's own extracted slices."
            )
            if self._dosy_slices:
                dosy_clear_slices_action = menu.addAction("Clear Rows Beside the Map")
            dosy_profile_action = menu.addAction(f"Extract Diffusion Profile at {shift:.4g} ppm")
            dosy_profile_action.setToolTip(
                "The column of the map at this shift – how this peak's intensity is "
                "distributed over D, which is what the ILT actually computed."
            )
            dosy_levels_action = menu.addAction("Reset Contour Levels")

        contour_levels_action = None
        extract_slice_action = None
        clear_slices_action = None
        if self._contour_spectrum is not None:
            contour_levels_action = menu.addAction("Set Contour Levels…")
            contour_scroll_action = menu.addAction("Scroll Up Shows More Contours")
            contour_scroll_action.setCheckable(True)
            contour_scroll_action.setChecked(self._contour_scroll_up_more)
            contour_scroll_action.setToolTip(
                "Off: scrolling up over the contour shows fewer levels (raises the "
                "lowest one). On: scrolling up shows more. Also in Settings ▸ Interface.")
            contour_scroll_action.toggled.connect(self._on_contour_scroll_action_toggled)
            extract_slice_action = menu.addAction("Extract Slice…")
            if self._contour_slices:
                clear_slices_action = menu.addAction(
                    f"Clear Extracted Slices ({len(self._contour_slices)})"
                )
                # One entry per slice, so there is nothing to hit-test: the
                # list shows each slice's current name and colour order.
                rename_menu = menu.addMenu("Rename Extracted Slice")
                rename_menu.setToolTip(
                    "Change the label of a slice already extracted. Also under "
                    "Plot Style, in the series list.")
                for position in range(len(self._contour_slices)):
                    rename_action = rename_menu.addAction(self.contour_slice_label(position))
                    rename_action.triggered.connect(
                        lambda _c=False, p=position: self._prompt_rename_slice(p))
            proj_menu = menu.addMenu("Projections computed as")
            for text, mode in (("Maximum (positive)", "max"), ("Sum", "sum")):
                act = proj_menu.addAction(text)
                act.setCheckable(True)
                act.setChecked((self._projection_mode == "auto" and mode == "max")
                               or self._projection_mode == mode)
                act.triggered.connect(lambda _c=False, m=mode: self.set_projection_mode(m))
            labels_action = menu.addAction("Edit Axis Labels (F2 / F1)…")
            labels_action.setToolTip(
                "Type your own label for either axis; the other keeps its own.")
            labels_action.triggered.connect(self._prompt_axis_labels)
            for dim in ("f2", "f1"):
                if self._projection_style[dim].get("visible") is False:
                    show_action = menu.addAction(f"Show {dim.upper()} Projection")
                    show_action.triggered.connect(
                        lambda _checked=False, d=dim: self.set_projection_style(d, visible=None))
            for dim in ("f2", "f1"):
                held = self.external_projection(dim)
                if held is None:
                    continue
                remove_action = menu.addAction(
                    f"Remove External {dim.upper()} Projection ({held['label']})")
                remove_action.setToolTip(
                    "Go back to the projection computed from the 2D itself.")
                remove_action.triggered.connect(
                    lambda _checked=False, d=dim: self.clear_external_projection(d))

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
        remove_slice_action = None
        # A single peak's fit (_fit_mode) or every peak's together
        # (_curve_fits_shown): either way the points are a series the
        # owner can refit without one of them. Explicit user request,
        # 2026-11-10: "to omit points from the integral profile for
        # pseudo-2d data, we should right-click and 'remove point'".
        if self._fit_mode or self._curve_fits_shown:
            exclude_point_action = menu.addAction("Remove Point")
            exclude_point_action.setToolTip(
                "Take the point nearest the click out of its peak's curve fit and refit "
                "that peak. It stays on the plot as a grey ×.")
            exclude_point_action.setEnabled(self._last_right_click_xy is not None)
            if self.series_slices_removable:
                remove_slice_action = menu.addAction("Remove Slice from Every Fit")
                remove_slice_action.setToolTip(
                    "Drop the whole slice nearest the click from every fit on this "
                    "spectrum – every peak's curve, SCORE and the DOSY ILT – and refit. "
                    "The same as unticking it in Dynamics ▸ Slices used in fits.")
                remove_slice_action.setEnabled(self._last_right_click_xy is not None)
            reset_points_action = menu.addAction("Restore Removed Points")

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

        chosen = self._exec_context_menu(menu, pos)
        if chosen is None:
            return      # dismissed
        # The guard above is load-bearing, not defensive tidiness. Every
        # action below is dispatched by identity, and the optional ones
        # are None when their view is not showing -- so a dismissed menu
        # (`chosen is None`) matched the first optional action and ran
        # it. Harmless for the always-created actions this file had
        # before; an immediate crash for the DOSY ones, which then
        # subscripted a `_last_right_click_xy` that was also None.
        if chosen is dosy_panel_action:
            self.dosyExtractRequested.emit("spectrum_panel", float(self._last_right_click_xy[1]))
            return
        if chosen is dosy_clear_slices_action:
            self.set_dosy_slices([])
            return
        if chosen is dosy_spectrum_action:
            self.dosyExtractRequested.emit("spectrum", float(self._last_right_click_xy[1]))
            return
        if chosen is dosy_profile_action:
            self.dosyExtractRequested.emit("profile", float(self._last_right_click_xy[0]))
            return
        if chosen is dosy_levels_action:
            self._dosy_factor = self._DOSY_FACTOR_DEFAULT
            self._draw_dosy(preserve_view=True)
            return

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
        elif calibrate_action is not None and chosen is calibrate_action:
            self.referenceCalibrationRequested.emit(self._last_right_click_xy)
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
        elif remove_slice_action is not None and chosen is remove_slice_action:
            if self._last_right_click_xy is not None:
                self.fitSliceRemoveRequested.emit(*self._last_right_click_xy)
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
            "tick_dir": self._TICK_DIRECTION,
            "contour_linewidth": self._contour_linewidth,
        }

    def _apply_current_axis_style(self) -> None:
        """Apply the current linewidth/tick/label style to every axes
        on the figure. Called both by :meth:`set_style` (so a change is
        visible immediately) and after every contour rebuild (so a
        redraw doesn't revert to ``plot_2d_contour``'s own defaults --
        ``_contour_style_kwargs`` covers what that function accepts,
        this covers the rest, spine linewidth and minor ticks)."""
        for ax in self.figure.get_axes():
            if getattr(ax, "name", "") == "3d":
                # A 3D picture (Simulate's tensor views) has no spines or
                # minor ticks in the 2D sense; its text follows the style.
                # Tick labels the same size as the axis labels (user
                # request, 2026-10-01), as on every other plot.
                ax.tick_params(labelsize=self._AXIS_LABELSIZE)
                self._italicise_axis_labels(ax.xaxis, ax.yaxis, ax.zaxis)
                for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
                    axis.label.set_fontsize(self._AXIS_LABELSIZE)
                    # Clear of the (now full-size) tick labels at any size.
                    axis.labelpad = 0.5 * self._AXIS_LABELSIZE
                    if self._font_family:
                        axis.label.set_fontfamily(self._font_family)
                self._reset_3d_box(ax)
                continue
            for spine in ax.spines.values():
                spine.set_linewidth(self._AXIS_LINEWIDTH)
            ax.tick_params(which="major", length=self._TICK_LENGTH,
                           width=self._TICK_WIDTH, labelsize=self._AXIS_LABELSIZE)
            ax.tick_params(which="both", direction=self._TICK_DIRECTION)
            if self._minor_ticks_visible:
                ax.minorticks_on()
                ax.tick_params(which="minor", length=self._TICK_LENGTH * 0.5,
                               width=self._TICK_WIDTH * 0.75)
            else:
                ax.minorticks_off()
            if getattr(ax, "_opennmr_categorical", False):
                # Rows and columns that are names, not numbers (a
                # correlation matrix): minor ticks and tick marks mean
                # nothing there.
                ax.minorticks_off()
                ax.tick_params(which="both", length=0)
            ax.xaxis.label.set_fontsize(self._AXIS_LABELSIZE)
            ax.yaxis.label.set_fontsize(self._AXIS_LABELSIZE)
            self._italicise_axis_labels(ax.xaxis, ax.yaxis)
            if ax.get_title():
                ax.title.set_fontsize(self._AXIS_LABELSIZE)
            self._apply_title_and_font(ax)
            # Secondary axes (e.g. the temperature axis on top of a
            # 1000/T plot) are children of their Axes, not in
            # figure.get_axes(); give them the same text and ticks.
            for child in getattr(ax, "child_axes", []):
                if type(child).__name__ != "SecondaryAxis":
                    continue
                child.tick_params(which="major", length=self._TICK_LENGTH,
                                  width=self._TICK_WIDTH, labelsize=self._AXIS_LABELSIZE)
                child.tick_params(which="both", direction=self._TICK_DIRECTION)
                self._italicise_axis_labels(child.xaxis, child.yaxis)
                for axis in (child.xaxis, child.yaxis):
                    axis.label.set_fontsize(self._AXIS_LABELSIZE)
                    if self._font_family:
                        axis.label.set_fontfamily(self._font_family)
                        for label in axis.get_ticklabels():
                            label.set_fontfamily(self._font_family)
                for spine in child.spines.values():
                    spine.set_linewidth(self._AXIS_LINEWIDTH)
        if self._external_plot:
            self._restyle_external_axis_labels()
        # A figure-level title (Simulate's Bloch and tensor pictures)
        # follows the text size like an Axes title.
        suptitle = getattr(self.figure, "_suptitle", None)
        if suptitle is not None and suptitle.get_text():
            suptitle.set_fontsize(self._AXIS_LABELSIZE)
            suptitle.set_visible(self._title_visible)
            if self._font_family:
                suptitle.set_fontfamily(self._font_family)
        self._apply_curve_spines()
        self._reapply_panel_axis_styles()
        self._restyle_existing_legends()
        threes = [ax for ax in self.figure.get_axes() if getattr(ax, "name", "") == "3d"]
        if threes:
            # After the relabelling above: a label's text sets its width.
            self._fit_3d_layout(threes)
        self._apply_colorbar_gap()

    def _reapply_panel_axis_styles(self) -> None:
        """Compare ▸ Panels: the user's per-panel axis options go on top of
        the global Plot Style, whenever that restyles the axes."""
        kw = self._compare_panels_kwargs
        if kw and self._spectrum_panels and kw.get("axis_styles"):
            from ...notebook.compare_panels import apply_axis_styles
            apply_axis_styles(self._spectrum_panels, kw["axis_styles"], self._AXIS_LABELSIZE)

    def _apply_colorbar_gap(self) -> None:
        """Plot Style's colour-bar gap, on Simulate's pictures (a Czjzek
        map, a tensor surface). After every layout pass, which re-spaces
        the plots, and every rotation, which moves a 3D box's corner."""
        if self._external_plot and self._colorbar_gap >= 0:
            try:
                plotting.set_colorbar_gap(self.figure, self._colorbar_gap)
            except Exception:  # a spacing nicety must never break a redraw
                pass

    def _reset_3d_box(self, ax) -> None:
        """The 3D box at the size the text asks for -- a little smaller
        as the text grows -- with its ticks fitted to it; then
        :meth:`_fit_3d_layout` shrinks it further only if a label would
        still fall outside the figure. Always from this base, so a box
        shrunk for one view grows back for the next."""
        zoom = min(1.0, max(0.75, 1.0 - 0.02 * (self._AXIS_LABELSIZE - 11.0)))
        aspect = getattr(ax, "_box_aspect", None)
        ax.set_box_aspect((1, 1, 1) if aspect is None else tuple(aspect), zoom=zoom)
        self._fit_3d_ticks(ax)

    #: "quantity (unit)" at the end of a label, and "quantity / unit".
    _BRACKET_LABEL = re.compile(r"^(?P<q>.*\S)\s*\((?P<u>[^()]+)\)\s*$")
    _SLASH_LABEL = re.compile(r"^(?P<q>.*\S)\s+/\s+(?P<u>\S.*)$")
    #: Parenthesised words that qualify a quantity rather than name its
    #: unit -- "Amplitude (normalised)" stays as it is.
    _NOT_A_UNIT = re.compile(r"^\s*(normali[sz]ed|relative|rel\.?|per\b.*|arb\.?)\s*$", re.I)

    def _tensor_style_carried_over(self, map_colormap, compare_colormap, grid) -> dict:
        """The tensor-style keys that a *change* of Plot Style's map or
        series colour map, or of its Grid box, sets while a tensor
        picture is on show (see set_style). A qualitative series palette
        (okabe_ito, tab10 ...) is a set of categories, not a scale, so
        it is not put on a surface."""
        if not (self._external_plot and self._plot_kind == "tensor"):
            return {}
        carried = {}
        if grid is not None and bool(grid) != bool(self._grid_visible):
            carried["grid"] = bool(grid)
        if compare_colormap is not None and compare_colormap != self._compare_colormap:
            from ...plotting.style import is_qualitative
            import matplotlib
            if compare_colormap in matplotlib.colormaps and not is_qualitative(compare_colormap):
                carried["colormap"] = compare_colormap
        if map_colormap is not None and (map_colormap or None) != self._map_colormap:
            carried["colormap"] = map_colormap or None
        return carried

    def _relabel_for_style(self, text: str) -> str:
        """``text`` in this canvas's axis-label style: ``"Q (unit)"``
        (brackets) or ``"Q / unit"`` (slash)."""
        if not text:
            return text
        if self._axis_label_style == "slash":
            match = self._BRACKET_LABEL.match(text)
            if match and not self._NOT_A_UNIT.match(match.group("u")):
                return f"{match.group('q')} / {match.group('u')}"
            return text
        match = self._SLASH_LABEL.match(text)
        if match and not self._NOT_A_UNIT.match(match.group("u")):
            return f"{match.group('q')} ({match.group('u')})"
        return text

    @staticmethod
    def _italicise_axis_labels(*axes) -> None:
        """Quantity symbols in axis labels italic, every plot the canvas
        shows (user request, 2026-10-17): "T (K)" -> "$T$ (K)", "T₁" ->
        "$T_1$" (plotting.labels.italicise_symbols; words, units and
        existing mathtext untouched)."""
        from ...plotting.labels import italicise_symbols
        for axis in axes:
            try:
                old = axis.label.get_text()
                new = italicise_symbols(old)
                if new != old:
                    axis.label.set_text(new)
            except Exception:  # noqa: BLE001 -- styling must never stop a plot drawing
                pass

    def _restyle_external_axis_labels(self) -> None:
        """A figure drawn elsewhere writes its own axis labels; Plot
        Style's "Axis label" choice (quantity (unit) or quantity / unit)
        is applied to them here, every axis including a 3D one's z and a
        colour bar's (user report, 2026-10-01: the setting did not reach
        Simulate's tensor plots)."""
        for ax in self.figure.get_axes():
            axes = [ax.xaxis, ax.yaxis] + ([ax.zaxis] if getattr(ax, "name", "") == "3d" else [])
            for axis in axes:
                old = axis.label.get_text()
                new = self._relabel_for_style(old)
                if new != old:
                    axis.label.set_text(new)

    def _apply_curve_spines(self) -> None:
        """Top/right spines on the curve Axes, as Plot Style says. With
        a spine shown, its ticks are drawn on it too (copies of the
        bottom/left ones), which is how a closed frame reads in a
        manuscript; hidden, they go with it."""
        axes = [ax for ax in self._curve_axes if ax in self.figure.axes]
        for ax in axes:
            if ax.yaxis.get_label_position() == "right":
                # A twin y axis (the Bloch pulse's phase/frequency): its
                # own axis *is* the right spine -- hiding it would leave
                # its ticks floating. Only the top follows the setting.
                ax.spines["top"].set_visible(self._top_spine_visible)
                ax.tick_params(which="both", top=self._top_spine_visible)
                continue
            ax.spines["top"].set_visible(self._top_spine_visible)
            ax.spines["right"].set_visible(self._right_spine_visible)
            ax.tick_params(which="both", top=self._top_spine_visible,
                           right=self._right_spine_visible)

    def _legend_kwargs(self) -> dict:
        """The keyword arguments every ``ax.legend()`` on this canvas is
        built with.

        **One builder, not a kwarg repeated at each call site.** There
        are legends on the Spectrum view, the Compare stack, the
        spectral fit and the DOSY overlay; before this each called
        ``ax.legend()`` bare, so matplotlib's own rcParams decided how
        they looked and none of them could be changed without finding
        all four.

        ``frameon`` off is the default (see ``__init__``), in which
        case the frame's own colour and alpha are not passed at all:
        matplotlib treats any ``framealpha``/``facecolor`` as an
        implicit "there is a frame", which would quietly undo the
        setting the user just turned off."""
        from ...plotting.legend_place import legend_placement
        try:
            kwargs = legend_placement(self._legend_loc, self._legend_columns,
                                      self._legend_x, self._legend_y)
        except ValueError:          # a saved position this version does not know
            kwargs = legend_placement("best", self._legend_columns)
        kwargs.update({
            "fontsize": self._legend_fontsize or self._AXIS_LABELSIZE,
            "frameon": self._legend_frameon,
        })
        if self._legend_frameon:
            kwargs["framealpha"] = self._legend_framealpha
            if self._legend_facecolor:
                kwargs["facecolor"] = self._legend_facecolor
        return kwargs

    def _refresh_trace_names(self, restyle: bool = True) -> None:
        """Name the traces the way the style says -- a legend, names
        written on the traces (``_legend_style == "labels"``), or
        neither when names are hidden. Only this part of the view: safe
        to call on a plot already drawn (a style change), unlike
        :meth:`_apply_view_options`, which is for a fresh rebuild."""
        self._remove_trace_labels()
        if self._external_plot:
            # A notebook figure (Integrate ▸ DNP) names its own traces;
            # only its legend's style is this canvas's business.
            if restyle:
                self._restyle_existing_legends()
            return
        if self._spectrum_panels:
            self._refresh_panel_trace_names(restyle)
            return
        handles, labels = self._legend_entries()
        # A stacked Compare has one Axes per trace; its legend (one entry
        # per trace) goes on the top row, and nowhere else.
        host = self._compare_axes[0] if self._compare_axes else self.ax
        if host is None:
            return
        for ax in self.figure.get_axes():
            if ax is not host and ax.get_legend() is not None and self._compare_axes \
                    and ax in self._compare_axes:
                ax.get_legend().remove()
        as_labels = self._legend_style == "labels"
        if self._legend_visible and handles and not as_labels:
            host.legend(handles, labels, **self._legend_kwargs())
        else:
            existing = host.get_legend()
            if existing is not None:
                existing.remove()
        if self._legend_visible and handles and as_labels:
            self._draw_trace_labels(handles, labels)
        if restyle:
            self._restyle_existing_legends()
            self.canvas.draw_idle()

    def _refresh_panel_trace_names(self, restyle: bool = True) -> None:
        """Compare ▸ Panels: each panel names its own traces (a legend per
        panel, or the names written on the traces)."""
        as_labels = self._legend_style == "labels"
        for ax in self._spectrum_panels:
            if getattr(ax, "_opennmr_panel_kind", "1d") != "1d":
                continue          # a 2D overlay and a pulse diagram draw their own legends
            handles, labels = ax.get_legend_handles_labels()
            pairs = [(h, plotting.safe_label(l) if hasattr(plotting, "safe_label") else l)
                     for h, l in zip(handles, labels) if not str(l).startswith("_")]
            old = ax.get_legend()
            if self._legend_visible and pairs and not as_labels:
                ax.legend([h for h, _l in pairs], [l for _h, l in pairs], **self._legend_kwargs())
            elif old is not None:
                old.remove()
            if self._legend_visible and pairs and as_labels:
                self._draw_trace_labels([h for h, _l in pairs], [l for _h, l in pairs])
        if restyle:
            self._restyle_existing_legends()
            self.canvas.draw_idle()

    def _draw_trace_labels(self, handles, labels) -> None:
        """Each named trace's name just above its noise -- see
        :func:`opennmr.plotting.label_traces`. Re-placed whenever that
        Axes' x-range changes, since the noise is read where the name
        sits."""
        from matplotlib.lines import Line2D
        by_axes: dict = {}
        for handle, label in zip(handles, labels):
            if isinstance(handle, Line2D) and handle.axes is not None:
                by_axes.setdefault(handle.axes, []).append((handle, label))
        fontsize = self._legend_fontsize or self._AXIS_LABELSIZE
        for ax, pairs in by_axes.items():
            texts = plotting.label_traces(
                ax, [h for h, _l in pairs], [l for _h, l in pairs],
                position=self._trace_label_position, fontsize=fontsize,
                fontfamily=self._font_family or None, height=self._trace_label_height)
            self._trace_label_artists.extend(texts)
            if not getattr(ax, "_opennmr_label_cid", None):
                ax._opennmr_label_cid = ax.callbacks.connect(
                    "xlim_changed", lambda _ax: self._on_trace_label_xlim_changed())

    def _on_trace_label_xlim_changed(self) -> None:
        if self._trace_label_artists and not getattr(self, "_placing_labels", False):
            self._placing_labels = True
            try:
                self._refresh_trace_names(restyle=False)
            finally:
                self._placing_labels = False

    def _remove_trace_labels(self) -> None:
        for text in self._trace_label_artists:
            try:
                text.remove()
            except (ValueError, NotImplementedError, AttributeError):
                pass            # its Axes already went with a figure rebuild
        self._trace_label_artists = []

    def trace_labels(self) -> list:
        """The names currently written on the traces (Text artists)."""
        return list(self._trace_label_artists)

    def _legend_entries(self) -> tuple[list, list]:
        """(handles, labels) for this plot's legend: every named line of
        the plot -- of every row, for a stacked Compare -- with each name
        made safe to draw (see :func:`opennmr.plotting.safe_label`).

        Collected here rather than by ``ax.legend()``'s own search, which
        only looks at one Axes (so a stack's legend listed only its
        bottom trace) and silently skips any name starting with "_"
        (a folder called "_run2" never appeared)."""
        axes = self._compare_axes if self._compare_axes else [self.ax]
        view = self._active_series_view()
        series_names = {e.label for e in self._series.get(view, [])} if view else set()
        handles, labels = [], []
        for ax in axes:
            h, l = ax.get_legend_handles_labels()
            handles += h
            labels += l
            for line in ax.lines:
                name = line.get_label()
                if name.startswith("_") and name in series_names and line not in handles:
                    handles.append(line)
                    labels.append(name)
        if self._compare_axes or (self._compare_spectra and self._active_series_view() == "compare"):
            # Kept fits in Compare repeat "Data", "Baseline" and each
            # component (every row of a stack, or every fit superimposed):
            # one legend entry per name and colour is enough.
            seen, kept_h, kept_l = set(), [], []
            for handle, name in zip(handles, labels):
                key = (name, self._handle_colour(handle))
                if name not in series_names and key in seen:
                    continue
                seen.add(key)
                kept_h.append(handle)
                kept_l.append(name)
            handles, labels = kept_h, kept_l
        return handles, [plotting.safe_label(name) for name in labels]

    @staticmethod
    def _handle_colour(handle):
        from matplotlib.colors import to_hex
        try:
            if hasattr(handle, "get_color"):
                return to_hex(handle.get_color())
            colours = handle.get_facecolor()
            return to_hex(colours[0], keep_alpha=True) if len(colours) else None
        except (ValueError, TypeError, AttributeError):
            return None

    def _restyle_existing_legends(self) -> None:
        """Re-issue every legend already on the figure with this
        canvas's own :meth:`_legend_kwargs`.

        **Why a sweep rather than a kwarg at each call site.** Three of
        the legends on this canvas are not built here at all -- the
        curve fit, the spectral fit and the relaxation plot are drawn
        by :mod:`opennmr.plotting.fit`, which calls a bare
        ``ax.legend()`` of its own. Plumbing a new parameter through
        those signatures would reach exactly today's three and miss the
        fourth the next time one is added; sweeping the figure after
        each rebuild reaches whatever is actually there.

        Rebuilt, not nudged: font size, frame, alpha and face colour
        are all fixed when a Legend is constructed."""
        kwargs = self._legend_kwargs()
        for ax in self.figure.get_axes():
            old = ax.get_legend()
            if old is None:
                continue
            # The same entries, restyled: re-collecting with a bare
            # ax.legend() would drop the entries gathered from other
            # rows of a stacked Compare (see _legend_entries).
            handles = list(getattr(old, "legend_handles", None) or old.legendHandles)
            labels = [t.get_text() for t in old.get_texts()]
            # Columns left to the legend's builder: as many as it had.
            own = dict(kwargs) if "ncols" in kwargs else {**kwargs,
                                                          "ncols": getattr(old, "_ncols", 1)}
            legend = ax.legend(handles, labels, **own)
            plotting.set_legend_linewidth(legend, self._legend_linewidth)
            # **After the rebuild, not before.** The font family is a
            # property of the Text artists inside the Legend, and
            # re-issuing the legend builds new ones -- applying it in
            # _apply_title_and_font first (which runs before this) set
            # it on artists that were then thrown away.
            if self._font_family:
                for text in legend.get_texts():
                    text.set_fontfamily(self._font_family)
            if self._external_plot:
                # A notebook figure's own legend follows Show Legend like
                # any other: it could not be switched off (user report on
                # the DNP field profile's "line not found" key).
                legend.set_visible(self._legend_visible)

    def _chemical_shift_label(self, spectrum, x_unit: str = "ppm") -> str:
        """This canvas's own axis-label style/template, applied to
        ``spectrum``'s own nucleus -- the single place every ppm axis
        label on this canvas is built, so the Plot Style panel's
        "Axis label" controls reach every view (Spectrum, 2D contour,
        the Fit tab, Compare, the Processing preview) without a
        ``style``/``template`` parameter threaded through each of
        :mod:`opennmr.plotting`'s own ``plot_1d``/``plot_2d_contour``
        calls, which build the same label with their own standalone
        (always-``'brackets'``) default for notebook use.

        Falls back to ``spectrum.metadata.nucleus`` verbatim (skipping
        :func:`~opennmr.core.registry.find_gamma`) if that lookup
        fails, e.g. an unrecognised or blank nucleus string -- a
        best-effort label beats no plot at all.
        """
        if spectrum.metadata.raw.get("axis_is_hz"):
            # e.g. SIMPSON output read without a nucleus and field: the
            # axis is in Hz and says so.
            return self._unit_label("Frequency", "Hz")
        try:
            nuclide = find_gamma(spectrum.metadata.nucleus, SFO1=spectrum.metadata.sfo1, quiet=True)["Nuclide"]
        except Exception:
            nuclide = spectrum.metadata.nucleus or "?"
        return NUC_label(nuclide, x_unit=x_unit, style=self._axis_label_style,
                          template=self._axis_label_template)

    def _unit_label(self, quantity: str, unit: str) -> str:
        """A non-nuclide axis label -- "Frequency"/Hz, "Position"/mm
        -- in this canvas's own brackets/slash style. No ``template``
        here: the custom-label box's placeholders (``{element}``,
        ``{mass}``, ...) are nuclide-specific and would not mean
        anything for a quantity that has neither."""
        if self._axis_label_style == "slash":
            return f"{quantity} / {unit}"
        return f"{quantity} ({unit})"

    def _refresh_axis_labels(self) -> None:
        """Recompute every ppm/unit axis label currently on screen
        from this canvas's own axis-label style/template -- called by
        :meth:`set_style` whenever either changes, so the effect is
        visible immediately rather than only on the view's next full
        redraw.

        Covers the views built from a *stored* spectrum reference:
        the Spectrum tab (``self.ax`` + ``_current_1d_spectrum``),
        the 2D contour (``self.ax``'s x/y, via ``_contour_for_display``),
        and a Compare stack's shared bottom label. The Fit tab and
        Processing preview rebuild their own axis label from scratch
        on their next redraw instead (a style-only change doesn't
        itself trigger one) -- reaching them here would need the same
        fit/preview state their own draw methods already hold.
        """
        if self._contour_spectrum is not None and self.ax is not None:
            _spec, xlabel, ylabel = self._contour_for_display()
            if xlabel is not None:
                self.ax.set_xlabel(xlabel, fontsize=self._AXIS_LABELSIZE)
            if ylabel is not None:
                self.ax.set_ylabel(ylabel, fontsize=self._AXIS_LABELSIZE)
            return
        if self._current_1d_spectrum is not None and self.ax is not None:
            if self._axis_unit == "ppm":
                label = self._chemical_shift_label(self._current_1d_spectrum, x_unit="ppm")
                self._ppm_label = label
            else:
                label = self._unit_axis_label(self._axis_unit)
            if label is not None:
                self.ax.set_xlabel(label, fontsize=self._AXIS_LABELSIZE)
            return
        if self._compare_axes and self._compare_spectra:
            if self._axis_unit == "ppm":
                label = self._chemical_shift_label(self._compare_spectra[-1], x_unit="ppm")
                self._ppm_label = label
            else:
                label = self._unit_axis_label(self._axis_unit)
            if label is not None:
                self._compare_axes[-1].set_xlabel(label, fontsize=self._AXIS_LABELSIZE)

    def _apply_title_and_font(self, ax) -> None:
        """One axes' title visibility and font family.

        Shared by the two style passes (:meth:`_apply_current_axis_style`
        and the resize-driven text refit) rather than written into
        each: they already both exist for the same reason, and a
        setting applied by only one of them comes back on the next
        window resize."""
        # Hidden, not blanked: the views that set a title keep setting
        # it, so turning the option back on shows it again with no
        # redraw and no view needing to know the setting exists.
        ax.title.set_visible(self._title_visible)
        if not self._font_family:
            return
        for text in (ax.title, ax.xaxis.label, ax.yaxis.label,
                     *ax.get_xticklabels(), *ax.get_yticklabels(),
                     *ax.get_xticklabels(minor=True), *ax.get_yticklabels(minor=True)):
            text.set_fontfamily(self._font_family)
        # The legend's own texts are handled by
        # _restyle_existing_legends, which runs after this and rebuilds
        # them -- see there.

    @property
    def _compare_colormap(self) -> str:
        """The old name for :attr:`_series_colormap`, from when this
        palette only served the Compare tab."""
        return self._series_colormap

    @_compare_colormap.setter
    def _compare_colormap(self, value: str) -> None:
        self._series_colormap = value

    #: Which view's series list is live, by what the canvas is showing.
    #: One place deciding this, so `series()` and every draw path agree
    #: about what is on screen.
    def _active_series_view(self) -> str | None:
        if self._external_plot and self._series.get("external"):
            return "external"
        if self._dosy_overlay:
            return "dosy_overlay"
        if self._overlay_spectra:
            return "overlay_2d"
        if self._contour_spectrum is not None and self._contour_slices:
            return "slices"
        if self._compare_spectra:
            return "compare"
        if self._current_1d_spectrum is not None and self._series.get("spectrum"):
            return "spectrum"
        return None

    def series(self) -> list[SeriesStyle]:
        """Every addressable series on the current plot.

        Empty for a view with no named series (a bare contour) -- a
        caller can offer per-series controls unconditionally and get
        nothing where there is nothing. A lone 1D spectrum has one: its
        legend name (and colour, which follows the trace colour until
        set here), so Plot Style can name it."""
        view = self._active_series_view()
        return list(self._series.get(view, [])) if view else []

    def set_series_style(self, key: str, *, label=None, colour=None) -> None:
        """Rename or recolour one series on the current plot, and
        redraw.

        The one entry point behind every per-series control, whatever
        kind of plot is showing -- the alternative was a separate
        setter and a separate dialog section per view, which is how the
        extracted-slice colour picker ended up not helping the DOSY
        overlay at all."""
        view = self._active_series_view()
        if view is None:
            return
        for entry in self._series.get(view, []):
            if entry.key != key:
                continue
            if label is not None and entry.can_rename:
                entry.label = label
                entry.label_overridden = True
            if colour is not None:
                entry.colour = colour
                entry.colour_overridden = True
            # Slices keep their own copy on the slice spec, which is
            # what persists with the spectrum; keep the two in step.
            if view == "slices":
                position = int(key)
                if 0 <= position < len(self._contour_slices):
                    self._contour_slices[position]["color"] = entry.colour
                    self._contour_slices[position]["label"] = entry.label
                    # Recorded on the spec too, since that is what
                    # persists with the spectrum -- a palette change in
                    # a later session must respect it the same way.
                    self._contour_slices[position]["color_override"] = entry.colour_overridden
                    self.contourSlicesChanged.emit(list(self._contour_slices))
            self._redraw_active_series_view()
            return

    def relabel_current_spectrum(self) -> None:
        """The spectrum on show was renamed: take its new name into the
        series and the legend, without redrawing the data."""
        spectrum = self._current_1d_spectrum
        entries = self._series.get("spectrum")
        if spectrum is None or not entries:
            return
        entries[0].label = _spectrum_label(spectrum)
        entries[0].label_overridden = False
        self._redraw_active_series_view()

    def _redraw_active_series_view(self) -> None:
        """Redraw whichever multi-series view is showing, preserving
        the view where that makes sense."""
        view = self._active_series_view()
        if view == "external":
            self._redraw_external()
        elif view == "dosy_overlay":
            self._draw_dosy(preserve_view=True)
        elif view == "overlay_2d":
            self.show_2d_overlay(self._overlay_spectra, self._overlay_labels, **self._overlay_kwargs)
        elif view == "slices":
            self._redraw_contour(preserve_view=True)
        elif view == "compare":
            if self._compare_panels_kwargs is not None and self._spectrum_panels:
                self.show_compare_panels(self._compare_spectra, self._spectrum_panel_groups,
                                         **self._compare_panels_kwargs)
            else:
                self.show_compare(self._compare_spectra, **(self._compare_kwargs or {}))
        elif view == "spectrum":
            # In place: a full show_spectrum would throw away the zoom
            # just to change a name or a colour.
            entry = self._series["spectrum"][0]
            if self.ax.lines:
                self.ax.lines[0].set_label(entry.label)
                self.ax.lines[0].set_color(entry.colour)
            self._apply_view_options()
            self.canvas.draw_idle()

    def _assign_series(self, view: str, entries) -> list[SeriesStyle]:
        """Build (or reuse) the series list for ``view``.

        Reuses a previous entry's label and colour where the key is the
        same, so a redraw -- a unit change, a level scroll, a restyle --
        keeps whatever was set rather than re-deriving it. That reuse
        *is* the fix for the resetting-colour bug, applied once here
        instead of per view."""
        previous = {entry.key: entry for entry in self._series.get(view, [])}
        palette = plotting.stack_colors(max(len(entries), 1), self._series_colormap)
        built = []
        for index, (key, label, can_rename) in enumerate(entries):
            was = previous.get(key)
            overridden = was.colour_overridden if was is not None else False
            built.append(SeriesStyle(
                key=key,
                # A name the user *typed* survives a redraw; a computed
                # one is recomputed. Same rule, and the same reasoning,
                # as the colour below: reusing on `can_rename` alone
                # could not tell "the user renamed this" from "a label
                # existed for this key last time", so a fresh SCORE run
                # kept the old run's numbers.
                label=(was.label if (was is not None and can_rename
                                     and was.label_overridden) else label),
                # Derived unless deliberately set: same palette gives
                # the same colour back on a redraw, a new palette gives
                # a new one, and an override survives both.
                colour=(was.colour if overridden
                        else matplotlib.colors.to_hex(palette[index % len(palette)])),
                can_rename=can_rename,
                colour_overridden=overridden,
                label_overridden=(was.label_overridden if was is not None else False),
            ))
        self._series[view] = built
        return built

    def reset_series_colours(self) -> None:
        """Drop every deliberate colour on the current plot, putting it
        back on the palette."""
        view = self._active_series_view()
        if view is None:
            return
        for entry in self._series.get(view, []):
            entry.colour_overridden = False
        if view == "external":
            self._external_palette = False
        if view == "slices":
            for spec in self._contour_slices:
                spec.pop("color", None)
                spec.pop("color_override", None)
            self.contourSlicesChanged.emit(list(self._contour_slices))
        self._redraw_active_series_view()

    def restyle_series(self) -> None:
        """Re-derive every non-overridden colour from the palette now in
        force, and redraw.

        What a colour-map change calls: the palette supplies defaults,
        so changing it has to actually reach the plot rather than
        waiting for the next redraw to happen for some other reason."""
        if self._active_series_view() == "external":
            self._external_palette = True
        if self._active_series_view() is not None:
            self._redraw_active_series_view()

    # -- Lines: widths and styles of every line on the plot -------------

    def _math_in_font_family(self):
        """A context for drawing in which the maths in labels ($^{13}$C,
        $T_1$) is set in Plot Style's font family too, not in matplotlib's
        DejaVu -- for a named face (user request, 2026-10-17: more
        typefaces); a generic family or Default leaves maths as it was."""
        import contextlib

        import matplotlib
        from matplotlib.text import Text

        from ...notebook import fonts as nb_fonts
        family = self._font_family
        named = bool(family) and family not in nb_fonts.GENERIC_FAMILIES
        if not named and not getattr(self, "_math_family_applied", False):
            return contextlib.nullcontext()
        default_math = matplotlib.rcParams["mathtext.fontset"]
        try:
            for text in self.figure.findobj(Text):
                if "$" in (text.get_text() or ""):
                    text.set_math_fontfamily("custom" if named else default_math)
        except Exception:  # noqa: BLE001 -- styling must never stop a plot drawing
            return contextlib.nullcontext()
        self._math_family_applied = named
        if not named:
            return contextlib.nullcontext()
        rc = {k: v for k, v in nb_fonts.font_rc(family).items() if k.startswith("mathtext.")}
        rc.pop("mathtext.fontset", None)
        return matplotlib.rc_context(rc)

    def _print_figure_vector_safe(self, filename, *args, **kwargs):
        from ...plotting import vector_export
        save = FigureCanvasQTAgg.print_figure.__get__(self.canvas)
        if vector_export.is_vector(filename, kwargs.get("format")):
            with vector_export.vector_safe(self.figure):
                return save(filename, *args, **kwargs)
        return save(filename, *args, **kwargs)

    def _draw_with_line_styles(self, renderer) -> None:
        """The figure's own draw, with Plot Style's line widths and
        styles applied first. A draw-time pass rather than a setting
        each plot's code reads, so it reaches every plot this canvas can
        show -- spectra, fits and their components, curve fits, Compare
        stacks, projection strips, notebook figures -- and every one
        added later, with nothing to wire per view."""
        try:
            plotting.lines.apply_line_styles(
                self.figure, self._line_width_scale, self._line_overrides,
                legend_widths=self._legend_linewidth is None)
            keys = tuple(info["key"] for info in plotting.lines.data_lines(self.figure))
        except Exception:  # noqa: BLE001 -- styling must never stop a plot drawing
            keys = self._line_keys_seen
        with self._math_in_font_family():
            Figure.draw(self.figure, renderer)
        if keys != self._line_keys_seen:
            self._line_keys_seen = keys
            # After this paint, not inside it: a listener rebuilds widgets.
            QTimer.singleShot(0, self.linesChanged.emit)

    @staticmethod
    def _clean_line_overrides(overrides) -> dict:
        """``{key: {"linewidth", "linestyle"}}`` checked and normalised
        (a JSON list dash pattern back to a tuple); empty entries dropped."""
        out = {}
        for key, entry in dict(overrides or {}).items():
            if not isinstance(entry, dict):
                continue
            clean = {}
            if entry.get("linewidth") is not None:
                width = float(entry["linewidth"])
                if width <= 0:
                    raise ValueError(f"Line width for {key!r} must be positive.")
                clean["linewidth"] = width
            if entry.get("linestyle") is not None:
                clean["linestyle"] = plotting.lines.normalise_linestyle(entry["linestyle"])
            if clean:
                out[str(key)] = clean
        return out

    def data_lines(self) -> list[dict]:
        """Every line on the current plot that Plot Style ▸ Lines can
        set, as :func:`opennmr.plotting.lines.data_lines` describes them,
        with their current (styled) width and style."""
        plotting.lines.apply_line_styles(self.figure, self._line_width_scale,
                                         self._line_overrides,
                                         legend_widths=self._legend_linewidth is None)
        lines = plotting.lines.data_lines(self.figure)
        for info in lines:
            info["overridden"] = info["key"] in self._line_overrides
        return lines

    def set_line_style(self, key: str, *, linewidth=None, linestyle=None,
                       clear: bool = False) -> None:
        """Width and/or style of one line (``key`` from :meth:`data_lines`),
        kept across redraws. ``clear`` drops its override."""
        if clear:
            self._line_overrides.pop(key, None)
        else:
            entry = dict(self._line_overrides.get(key, {}))
            if linewidth is not None:
                entry["linewidth"] = linewidth
            if linestyle is not None:
                entry["linestyle"] = linestyle
            self._line_overrides.update(self._clean_line_overrides({key: entry}))
        self.canvas.draw_idle()

    def reset_line_styles(self) -> None:
        """Every line back to its plot's own width and style."""
        self._line_overrides = {}
        self._line_width_scale = 1.0
        self.canvas.draw_idle()

    def style_settings(self) -> dict:
        """This canvas's own current style, in the form
        :meth:`set_style` takes it back.

        Style is per-canvas state (it always was -- ``__init__`` copies
        the class defaults onto instance attributes); what changed is
        that it is now *edited* per canvas rather than only ever being
        fanned out to all of them at once. A dialog editing one plot
        needs to show what that plot is currently set to, which is what
        this is for."""
        return {
            "linewidth": self._AXIS_LINEWIDTH,
            "labelsize": self._AXIS_LABELSIZE,
            "tick_length": self._TICK_LENGTH,
            "tick_width": self._TICK_WIDTH,
            "tick_direction": self._TICK_DIRECTION,
            "map_colormap": self._map_colormap,
            "colorbar_gap": self._colorbar_gap,
            "baseline_gap": self._baseline_gap,
            "stack_gap": self._stack_gap,
            "tensor_style": dict(self._tensor_style),
            "minor_ticks": self._minor_ticks_visible,
            "top_spine": self._top_spine_visible,
            "right_spine": self._right_spine_visible,
            "grid": self._grid_visible,
            "legend_loc": self._legend_loc,
            "legend_columns": self._legend_columns,
            "legend_x": self._legend_x,
            "legend_y": self._legend_y,
            "compare_colormap": self._compare_colormap,
            "trace_color": self._trace_color,
            "contour_linewidth": self._contour_linewidth,
            "contour_colour": self._contour_colour or "",
            "contour_clevels": self._contour_clevels,
            "contour_level_mode": self._contour_level_mode,
            "contour_power": self._contour_power,
            "dosy_clevels": self._dosy_clevels,
            "title_visible": self._title_visible,
            "font_family": self._font_family,
            "axis_label_style": self._axis_label_style,
            "axis_label_template": self._axis_label_template,
            "legend_frameon": self._legend_frameon,
            "legend_fontsize": self._legend_fontsize,
            "legend_framealpha": self._legend_framealpha,
            "legend_facecolor": self._legend_facecolor,
            "legend_linewidth": self._legend_linewidth,
            "legend_style": self._legend_style,
            "trace_label_position": self._trace_label_position,
            "trace_label_height": self._trace_label_height,
            "line_width_scale": self._line_width_scale,
            "line_overrides": copy.deepcopy(self._line_overrides),
        }

    def set_style(self, *, linewidth=None, labelsize=None, tick_length=None, tick_width=None,
                  tick_direction=None, map_colormap=None, tensor_style=None, colorbar_gap=None, minor_ticks=None, grid=None, legend_loc=None, legend_columns=None, legend_x=None, legend_y=None, compare_colormap=None,
                  trace_color=None, contour_linewidth=None, contour_colour=None, contour_clevels=None,
                  contour_level_mode=None, contour_power=None,
                  title_visible=None, font_family=None, axis_label_style=None,
                  axis_label_template=None, legend_frameon=None,
                  legend_fontsize=None, legend_framealpha=None, legend_facecolor=None,
                  legend_linewidth=None, legend_style=None,
                  trace_label_position=None, trace_label_height=None, top_spine=None, right_spine=None,
                  line_width_scale=None, line_overrides=None,
                  baseline_gap=None, stack_gap=None) -> None:
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
        if tick_direction is not None:
            if tick_direction not in ("in", "out", "inout"):
                raise ValueError(f"tick_direction must be 'in', 'out' or 'inout', not {tick_direction!r}")
            self._TICK_DIRECTION = tick_direction
        # A tensor picture on show takes Plot Style's colour maps and
        # grid when the user *changes* them (user report: "Changing
        # 'series colour map' did nothing for the tensor"). Only a
        # change counts: the panel re-sends every setting on any edit,
        # and a label-style edit re-sending grid=False switched the 3D
        # gridlines off. Last change wins; Tensor Plot Style shows it.
        carried = self._tensor_style_carried_over(map_colormap, compare_colormap, grid)
        if carried:
            tensor_style = {**carried, **(tensor_style or {})}
        if colorbar_gap is not None:
            self._colorbar_gap = float(colorbar_gap)
        gaps_changed = False
        if baseline_gap is not None and float(baseline_gap) != self._baseline_gap:
            self._baseline_gap = max(0.0, float(baseline_gap))
            gaps_changed = True
        if stack_gap is not None and float(stack_gap) != self._stack_gap:
            self._stack_gap = max(self._STACK_GAP_MIN, float(stack_gap))
            gaps_changed = True
        if gaps_changed:
            self._refit_spectrum_y_limits()
            if self._contour_spectrum is not None and self._contour_slices:
                self._redraw_contour(preserve_view=True)
        if map_colormap is not None:
            # "" from the combo is "each plot's own default".
            self._map_colormap = map_colormap or None
        if tensor_style is not None:
            if not isinstance(tensor_style, dict):
                raise ValueError("tensor_style must be a dict of tensor-picture settings.")
            changed_view = any(k in tensor_style and tensor_style[k] != self._tensor_style.get(k)
                               for k in ("elev", "azim"))
            self._tensor_style = {**self._tensor_style, **tensor_style}
            self._tensor_view_set = self._tensor_view_set or changed_view
            self.tensorStyleChanged.emit()
        if minor_ticks is not None:
            self._minor_ticks_visible = minor_ticks
        # Curve plots only (see _curve_axes); applied below by
        # _apply_current_axis_style like the rest of the axis furniture.
        if top_spine is not None:
            self._top_spine_visible = bool(top_spine)
        if right_spine is not None:
            self._right_spine_visible = bool(right_spine)
        if grid is not None:
            self._grid_visible = grid
        if legend_loc is not None:
            self._legend_loc = legend_loc
        if legend_columns is not None:
            self._legend_columns = max(int(legend_columns), 0)
        if legend_x is not None:
            self._legend_x = float(legend_x)
        if legend_y is not None:
            self._legend_y = float(legend_y)
        if compare_colormap is not None:
            self._compare_colormap = compare_colormap
        if title_visible is not None:
            self._title_visible = bool(title_visible)
        if font_family is not None:
            # "" from a combo meaning "matplotlib's own default" is
            # stored as None, so _apply_title_and_font's own truthiness
            # check is the only place that decision is made.
            self._font_family = font_family or None
        if axis_label_style is not None:
            if axis_label_style not in ("brackets", "slash"):
                raise ValueError("axis_label_style must be 'brackets' or 'slash'")
            self._axis_label_style = axis_label_style
        if axis_label_template is not None:
            # "" from the panel's line edit means "no custom template,
            # use axis_label_style instead" -- same None-vs-falsy-
            # sentinel convention as font_family just above.
            self._axis_label_template = axis_label_template or None
        if legend_frameon is not None:
            self._legend_frameon = bool(legend_frameon)
        if legend_fontsize is not None:
            # 0 is the panel's "follow the axis label size" sentinel --
            # see _legend_kwargs.
            self._legend_fontsize = legend_fontsize or None
        if legend_framealpha is not None:
            self._legend_framealpha = float(legend_framealpha)
        if legend_facecolor is not None:
            self._legend_facecolor = legend_facecolor or None
        if legend_linewidth is not None:
            # 0 is the panel's "match the traces" sentinel.
            self._legend_linewidth = float(legend_linewidth) or None
        if legend_style is not None:
            self._legend_style = "labels" if legend_style == "labels" else "legend"
        if trace_label_position is not None:
            self._trace_label_position = min(max(float(trace_label_position), 0.0), 1.0)
        if trace_label_height is not None:
            self._trace_label_height = min(max(float(trace_label_height), 0.0), 1.0)
        if line_width_scale is not None:
            if float(line_width_scale) <= 0:
                raise ValueError("line_width_scale must be positive.")
            self._line_width_scale = float(line_width_scale)
        if line_overrides is not None:
            self._line_overrides = self._clean_line_overrides(line_overrides)
        if contour_clevels is not None:
            # **The same number the right-click "Set Contour Levels…"
            # dialog and the DOSY display panel set**, not a parallel
            # one: a canvas shows a 2D contour or a DOSY map, never
            # both, so one control driving whichever is live is what
            # "linked to the other boxes where it might be used"
            # means. Both are stored so neither is lost by a switch
            # between the two views.
            clevels = max(2, int(contour_clevels))
            self._contour_clevels = clevels
            self._dosy_clevels = clevels
        # How those levels are spaced, alongside how many there are:
        # the count alone decides nothing about whether the weak peaks
        # beside a strong one are visible, which is what the spacing
        # is for. Same values the right-click "Set Contour Levels…"
        # dialog sets, so the two cannot disagree.
        if contour_level_mode is not None:
            self._contour_level_mode = "power" if contour_level_mode == "power" else "linear"
        if contour_power is not None:
            self._contour_power = float(contour_power)
        if contour_linewidth is not None:
            self._contour_linewidth = contour_linewidth
        colour_changed = False
        if contour_colour is not None:
            # "" = back to black (None = leave as it is). Validated here so a typo is an
            # error at the call, not a blank plot.
            from ...plotting.style import resolve_contour_colouring
            resolve_contour_colouring(contour_colour)
            if isinstance(contour_colour, list):
                contour_colour = list(contour_colour)
            new_colour = None if (isinstance(contour_colour, str) and contour_colour in ("", "default")) else contour_colour
            colour_changed = new_colour != self._contour_colour
            self._contour_colour = new_colour
        if (contour_linewidth is not None or colour_changed or contour_clevels is not None
                or contour_level_mode is not None or contour_power is not None):
            # Rebuild: a contour's line width and its levels both live
            # on the QuadContourSet matplotlib builds, so neither can
            # be set on what is already drawn the way a trace colour
            # can. One rebuild covers both, however many changed.
            if self._dosy_result is not None:
                self._draw_dosy(preserve_view=True)
            elif self._contour_spectrum is not None:
                self._redraw_contour(preserve_view=True)
        if trace_color is not None:
            self._trace_color = trace_color
            # Unlike the axis furniture below, a trace colour lives on
            # the Line2D itself -- applied to what is already drawn so
            # the change shows without waiting for a reselect, the same
            # way every other setting here does.
            for ax in self.figure.get_axes():
                for line in ax.lines:
                    if line.get_gid() not in (self._BASELINE_PREVIEW_GID,
                                               self._BASELINE_GHOST_GID):
                        line.set_color(trace_color)

        # A figure drawn by someone else (Simulate, Integrate ▸ DNP) reads
        # its colours and text sizes from style_settings() as it draws:
        # those live on artists only the draw makes, so it is redrawn.
        # The legend's size and style too: the S/N plot writes its values
        # and (optionally) its names at the legend's size (2026-10-15).
        if self._external_plot and self._external_draw is not None and any(
                value is not None for value in (labelsize, font_family, compare_colormap, colorbar_gap,
                                                map_colormap, tensor_style, trace_color,
                                                linewidth, legend_fontsize, legend_style)):
            self._redraw_external()

        # Everything except grid/legend is the same "apply the current
        # style to every axes" the contour rebuild needs, so it lives in
        # one place rather than being written twice. Applying all of it
        # unconditionally is equivalent to the old per-argument guards,
        # since anything not passed in is re-applied at its existing
        # value.
        # Ends with _restyle_existing_legends, which is what re-issues
        # every legend with the current kwargs -- deliberately *not*
        # repeated in the grid loop below. Doing it in both places
        # rebuilt each legend twice, and the second rebuild (with no
        # font pass after it) threw away the font family the first had
        # just applied to its Text artists.
        self._apply_current_axis_style()
        if (legend_style is not None or trace_label_position is not None
                or trace_label_height is not None
                or labelsize is not None or legend_fontsize is not None
                or font_family is not None):
            # Legend <-> names on the traces, or where/how big the names
            # are: the names are placed from the font size, so redo them.
            self._refresh_trace_names()
        if axis_label_style is not None or axis_label_template is not None:
            self._refresh_axis_labels()
        for ax in self.figure.get_axes():
            # A 3D axes' grid is the tensor style's "grid" (carried over
            # above when the Grid box is changed), redrawn with it.
            if grid is not None and getattr(ax, "name", "") != "3d":
                ax.grid(grid, **({"alpha": 0.3} if grid else {}))

        # compare_colormap only takes effect on the next full
        # show_compare redraw (it re-derives every trace's colour from
        # scratch) -- nothing to update on the already-drawn axes here.

        self.canvas.draw_idle()

    def set_stack_gap(self, gap: float) -> None:
        """Change the spacing between stacked spectra alone -- the Compare
        ▸ Spacing slider's fast path (:meth:`set_style` re-applies every axis
        setting, too slow to run on every step of a drag). ``gap`` is a
        fraction of a spectrum's height; negative: the spectra overlap."""
        gap = max(self._STACK_GAP_MIN, float(gap))
        if gap == self._stack_gap:
            return
        self._stack_gap = gap
        self._refit_spectrum_y_limits()
        if self._contour_spectrum is not None and self._contour_slices:
            self._redraw_contour(preserve_view=True)

    def stack_gap(self) -> float:
        """The current spacing between stacked spectra (see :meth:`set_stack_gap`)."""
        return float(self._stack_gap)

    def plot_kind(self) -> str | None:
        """The kind of plot showing -- ``"1d"``, ``"2d"``,
        ``"pseudo2d"``, ``"compare"``, ``"curve"`` (relaxation/diffusion
        fits, parameter trends, DNP profiles) -- or ``None`` before the first
        one (or for a view with no per-kind default, e.g. a FID)."""
        return self._plot_kind

    def _enter_plot_kind(self, kind: str) -> None:
        """Record that a ``kind`` plot is about to be drawn, announcing
        it if that is a change -- see :attr:`plotKindChanged`."""
        if kind == self._plot_kind:
            return
        self._plot_kind = kind
        self.plotKindChanged.emit(kind)

    #: set_style's keyword arguments, for filtering a saved style dict
    #: (which also carries export and bookkeeping keys) down to them.
    _STYLE_KEYS = (
        "linewidth", "labelsize", "tick_length", "tick_width", "tick_direction", "map_colormap",
        "tensor_style", "colorbar_gap", "minor_ticks", "grid",
        "legend_loc", "legend_columns", "legend_x", "legend_y", "compare_colormap", "trace_color", "contour_linewidth", "contour_colour",
        "contour_clevels", "contour_level_mode", "contour_power", "title_visible",
        "font_family", "axis_label_style", "axis_label_template", "legend_frameon",
        "legend_fontsize", "legend_framealpha", "legend_facecolor",
        "legend_linewidth", "legend_style", "trace_label_position", "trace_label_height",
        "top_spine", "right_spine", "line_width_scale", "line_overrides",
        "baseline_gap", "stack_gap",
    )

    def apply_style_defaults(self, style: dict) -> None:
        """Adopt ``style`` -- a saved per-plot-kind default, in
        :meth:`style_settings`' own keys (unknown ones ignored) -- as
        this canvas's style, **ahead of a redraw that is about to
        happen anyway**.

        Differs from :meth:`set_style` in one way: the contour
        settings (levels, spacing, exponent, line width) are stored
        without rebuilding whatever contour is on screen now, since it
        is about to be replaced -- and the level settings also become
        what this canvas resets to for each *new* 2D spectrum, which
        otherwise goes back to the built-in defaults (see
        :meth:`show_spectrum`). So a saved 2D default holds for every
        2D shown, not only the first."""
        values = {k: v for k, v in style.items() if k in self._STYLE_KEYS and v is not None}
        if "contour_clevels" in values:
            clevels = max(2, int(values["contour_clevels"]))
            self._contour_clevels = self._dosy_clevels = clevels
            self._CONTOUR_CLEVELS_DEFAULT = clevels
        if "contour_level_mode" in values:
            mode = "power" if values["contour_level_mode"] == "power" else "linear"
            self._contour_level_mode = self._CONTOUR_LEVEL_MODE_DEFAULT = mode
        if "contour_power" in values:
            self._contour_power = self._CONTOUR_POWER_DEFAULT = float(values["contour_power"])
        if "contour_linewidth" in values:
            self._contour_linewidth = float(values["contour_linewidth"])
        if "contour_colour" in values:
            colour = values["contour_colour"]
            self._contour_colour = None if colour in ("", "default") else colour
        for key in ("contour_clevels", "contour_level_mode", "contour_power", "contour_linewidth",
                    "contour_colour"):
            values.pop(key, None)
        if values.get("axis_label_style") not in (None, "brackets", "slash"):
            values.pop("axis_label_style")
        self.set_style(**values)

    #: Unit the Save Figure preview shows lengths in ("in" or "cm");
    #: set by Settings ▸ Interface ▸ Export size in.
    export_length_unit = "in"

    def open_export_preview(self) -> str | None:
        """File ▸ Save Figure As…: preview the figure as it will be
        written, then save it. Returns the path written, or ``None``."""
        from ..figure_preview_dialog import FigurePreviewDialog
        dialog = FigurePreviewDialog(self, parent=self.window(), unit=self.export_length_unit)
        dialog.settingsKept.connect(self.exportSettingsKept)
        dialog.exec()
        return dialog.saved_path

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

        clevels_spin = TypedSpinBox()
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
            "intensity – the noise floor. Same value scroll-over-the-"
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
            "Higher values bunch more levels near the noise floor – "
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

    def _nearest_slice_index(self, spectrum: NMRSpectrum, axis: str, cursor_value: float) -> int:
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
            # In whatever unit is on screen, not always ppm -- the
            # cursor value comes off the displayed axis, so comparing it
            # against ppm would pick the wrong row once the contour is
            # showing Hz, kHz or mm.
            values = self._axis_values_in_unit(spectrum_axis, self._axis_unit)
            if values is None:
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

        index_spin = TypedSpinBox()
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
            self._slice_label.setText(
                f"Viewing {self._extracted_label} \u2014 <b><u>not yet saved</u></b>. "
                f"Keep it to phase or fit it on its own \u2192")
            self._slice_label.setToolTip(
                "This slice is only being displayed. Until it is kept, the Phase controls "
                "and the fits act on the 2D it came from, and leaving this view discards it.")

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

    #: (quantity, unit) for each non-ppm display unit, fed to
    #: :meth:`_unit_label` for this canvas's own brackets/slash
    #: style. ppm's own label is nucleus-dependent and comes from
    #: :meth:`_chemical_shift_label`/``NUC_label`` instead.
    _UNIT_QUANTITIES = {"hz": ("Frequency", "Hz"), "khz": ("Frequency", "kHz"), "mm": ("Position", "mm")}

    def _unit_axis_label(self, unit: str) -> str | None:
        """The full axis label for a non-ppm display ``unit``
        ("hz"/"khz"/"mm"), in this canvas's own style -- ``None``
        for an unrecognised unit."""
        pair = self._UNIT_QUANTITIES.get(unit)
        return self._unit_label(*pair) if pair else None

    def _imaging_hz_per_mm(self) -> float | None:
        """Hz per mm for the current imaging gradient, or ``None`` when
        :meth:`set_imaging_gradient_params` hasn't been given real
        values. Spelled out in separate unit steps rather than one
        folded constant, so the conversion is checkable at a glance
        rather than needing to be trusted."""
        gamma = self._imaging_gamma_hz_per_t
        gradient = self._imaging_max_gradient_g_per_cm
        if not gamma or not gradient:
            return None
        gradient_t_per_m = gradient * 1e-2   # G/cm -> T/m (1 G = 1e-4 T, 1 cm = 1e-2 m)
        hz_per_m = gamma * gradient_t_per_m
        return hz_per_m * 1e-3

    def _axis_values_in_unit(self, axis, unit: str):
        """``axis``'s own values in ``unit``, or ``None`` if it cannot
        be expressed that way (an index axis, or ``"mm"`` with no
        gradient set)."""
        if axis is None or axis.kind != "frequency":
            return None
        if unit == "ppm":
            return axis.values("ppm")
        hz = axis.values("hz")
        if unit == "hz":
            return hz
        if unit == "khz":
            return hz / 1000.0
        if unit == "mm":
            hz_per_mm = self._imaging_hz_per_mm()
            return hz / hz_per_mm if hz_per_mm else None
        return None

    def _contour_for_display(self):
        """``(spectrum, xlabel, ylabel)`` to hand to
        :func:`plot_2d_contour` for the current display unit.

        For ppm this is the spectrum itself. For anything else it is a
        shallow copy whose frequency axes carry the *converted* values,
        because ``plot_2d_contour`` reads ``axes[i].ppm`` directly --
        so the copy's ``ppm`` field holds Hz/kHz/mm and the labels are
        passed explicitly to say so. A copy rather than a mutation:
        ``_contour_spectrum`` stays in ppm, which is what slicing,
        referencing and everything else still reason in.

        Real user report: "I am not able to right-click and change the
        axis on a 2D (for imaging)" -- the Axis Units menu was gated on
        a 1D view, and ``set_axis_unit`` only ever rewrote line data.
        """
        spectrum = self._contour_spectrum
        if spectrum is None:
            return spectrum, None, None
        if self._axis_unit == "ppm":
            # Still computed explicitly, in this canvas's own
            # style/template, rather than left for plot_2d_contour's
            # own (always-'brackets') NUC_label default -- see
            # _chemical_shift_label. Mirrors plot_2d_contour's own
            # homonuclear/pseudo-2D logic for the F1 (y) label so the
            # two only ever disagree on style, never on which axis
            # gets which nucleus.
            xlabel = self._chemical_shift_label(spectrum, x_unit="ppm")
            if spectrum.metadata.homonuclear:
                ylabel = xlabel
            elif not spectrum.is_pseudo_2d:
                indirect = copy.copy(spectrum)
                indirect.metadata = dataclasses.replace(
                    spectrum.metadata, nucleus=spectrum.metadata.nucleus_indirect)
                ylabel = self._chemical_shift_label(indirect, x_unit="ppm")
            else:
                ylabel = None  # plot_2d_contour's own 'Slice number' default
            return spectrum, *self._labels_with_overrides(spectrum, xlabel, ylabel)
        converted = copy.copy(spectrum)
        converted.axes = list(spectrum.axes)
        label = self._unit_axis_label(self._axis_unit)
        xlabel = ylabel = None
        for dim in (0, 1):
            values = self._axis_values_in_unit(spectrum.axes[dim], self._axis_unit)
            if values is None:
                continue
            axis = spectrum.axes[dim]
            converted.axes[dim] = NMRAxis(kind="frequency", ppm=values, hz=axis.hz,
                                          index=axis.index, label=axis.label)
            if dim == 1:
                xlabel = label
            else:
                ylabel = label
        return converted, *self._labels_with_overrides(spectrum, xlabel, ylabel)

    def _labels_with_overrides(self, spectrum, xlabel, ylabel):
        xlabel = self._apply_axis_label_override(spectrum, "f2", xlabel)
        if not spectrum.is_pseudo_2d:
            ylabel = self._apply_axis_label_override(spectrum, "f1", ylabel)
        return xlabel, ylabel

    def _contour_unit_available(self, unit: str) -> bool:
        """Whether the current contour can be shown in ``unit`` at all
        -- its direct axis has to be expressible that way."""
        if self._contour_spectrum is None:
            return False
        return self._axis_values_in_unit(self._contour_spectrum.axes[1], unit) is not None

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
        if unit not in ("ppm", "hz", "khz", "mm"):
            return
        previous_unit = self._axis_unit
        if unit != previous_unit:
            # Once the switch has happened (or been refused): see below.
            QTimer.singleShot(0, lambda: self._after_axis_unit_change(previous_unit))
        if self._contour_spectrum is not None:
            # A contour has no lines to rewrite -- its axes come from
            # the spectrum, so the unit change is a rebuild. Not
            # preserve_view: the old limits are in the *old* unit, so
            # keeping them would frame a meaningless region.
            if not self._contour_unit_available(unit):
                return
            self._axis_unit = unit
            self._redraw_contour(preserve_view=False)
            return
        if self._compare_spectra and self._compare_unit_groups():
            # Every Compare layout -- stack, overlay and separate panels
            # (the overlay and the panels could not change unit at all
            # before 2026-11-13, explicit user request).
            self._set_axis_unit_for_stack(unit)
            return
        if self._current_1d_spectrum is None or not self.ax.lines:
            return
        axis = self._current_1d_spectrum.axes[0]
        if unit == "ppm":
            x = axis.values("ppm")
            label = self._chemical_shift_label(self._current_1d_spectrum, x_unit="ppm")
        elif unit == "hz":
            x = axis.values("hz")
            label = self._unit_axis_label("hz")
        elif unit == "khz":
            x = axis.values("hz") / 1000.0
            label = self._unit_axis_label("khz")
        else:
            hz_per_mm = self._imaging_hz_per_mm()
            if not hz_per_mm:
                return
            x = axis.values("hz") / hz_per_mm
            label = self._unit_axis_label("mm")

        # Before the loop below, not after: it rewrites the x data of
        # *every* line on the Axes, and the crosshair's own axvline/
        # axhline are lines on this Axes too. Overwriting their
        # two-point data with the spectrum's full-length x array left
        # every subsequent mouse move raising ValueError: shape
        # mismatch out of Line2D.recache -- the reported "crosshair
        # disappears when changing between axes". Removed rather than
        # skipped, so `_ensure_crosshair` rebuilds it against the new
        # unit's coordinates on the next mouse move.
        self._remove_crosshair()
        # Same reasoning: a measurement's arrow and label are pinned to
        # data coordinates in the *old* unit, so they would silently
        # point somewhere else. A measurement is per-view; re-measure.
        self.clear_measurements(redraw=False)
        for line in self.ax.lines:
            line.set_xdata(x)
        self.ax.set_xlim(float(np.min(x)), float(np.max(x)))
        self.ax.invert_xaxis()
        self.ax.set_xlabel(label)
        if unit == "ppm":
            self._ppm_label = label
        self._axis_unit = unit
        self.canvas.draw_idle()

    def _after_axis_unit_change(self, previous_unit: str) -> None:
        """After :meth:`set_axis_unit`. If the unit did change, the view Home
        returns to and the plot before any typed limits were both in the old
        unit, so both are dropped; typed limits then go back on if the axis is
        again in the unit they were typed in, and the panel re-reads them."""
        if self._axis_unit == previous_unit:
            return
        try:
            self.axisUnitChanged.emit(self._axis_unit)
        except RuntimeError:    # the canvas was deleted before this queued call ran
            return
        self._home_limits = None
        self._natural_limits = None
        if self._view_x is None and self._view_y is None:
            return
        try:
            self._apply_view_limits()
            self.canvas.draw_idle()
        except RuntimeError:            # the canvas was closed in the meantime
            return
        self.viewLimitsChanged.emit()

    def _compare_unit_groups(self) -> list:
        """``[(axes, [spectrum indices drawn on it])]`` for the Compare on
        show: one row each in a stack, the one shared axes of an overlay,
        each 1D panel of a panel Compare (2D and pulse-sequence panels
        have no frequency axis to convert)."""
        spectra = self._compare_spectra
        if not spectra:
            return []
        if self._compare_axes:
            return [(ax, [i]) for i, ax in enumerate(self._compare_axes) if i < len(spectra)]
        if self._spectrum_panels:
            groups = []
            for ax, group in zip(self._spectrum_panels, self._spectrum_panel_groups):
                if getattr(ax, "_opennmr_panel_kind", "1d") != "1d":
                    continue
                indices = [i for i in group if 0 <= i < len(spectra)]
                if indices:
                    groups.append((ax, indices))
            return groups
        if self._compare_kwargs is not None and self.ax is not None:
            return [(self.ax, list(range(len(spectra))))]
        return []

    @staticmethod
    def _x_in_data(line, ax) -> bool:
        """Whether ``line``'s x values are data coordinates of ``ax`` (an
        offset or a y-only blend on top is fine; an axvline/axhline's
        axes-fraction x is not)."""
        transform = line.get_transform()
        if transform is ax.transData:
            return True
        try:
            return bool(transform.contains_branch_seperately(ax.transData)[0])
        except (AttributeError, ValueError):
            return False

    @staticmethod
    def _affine_between(from_values, to_values):
        """``(slope, intercept)`` with ``to = slope * from + intercept``
        for two samplings of one linear axis, or None."""
        if from_values is None or to_values is None:
            return None
        a = np.asarray(from_values, dtype=float)
        b = np.asarray(to_values, dtype=float)
        if a.size < 2 or a.size != b.size or a[-1] == a[0]:
            return None
        slope = (b[-1] - b[0]) / (a[-1] - a[0])
        return float(slope), float(b[0] - slope * a[0])

    @staticmethod
    def _line_owner(x, candidates, axes_values) -> int | None:
        """Which of ``candidates`` (spectrum indices) a line with x data
        ``x`` (in the current unit) was drawn from: the one whose axis
        holds its points -- a line drawn decimated keeps a subset of the
        axis's own points, and a kept fit's curve its own x. Unique match
        or None."""
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        if x.size == 0:
            return None
        probe = x[np.linspace(0, x.size - 1, min(x.size, 25)).astype(int)]
        owners = []
        for i in candidates:
            values = axes_values.get(i)
            if values is None or values.size < 2:
                continue
            ordered = np.sort(values)
            step = float(np.median(np.abs(np.diff(ordered)))) or 1.0
            where = np.clip(np.searchsorted(ordered, probe), 1, ordered.size - 1)
            nearest = np.minimum(np.abs(ordered[where] - probe), np.abs(ordered[where - 1] - probe))
            if np.all(nearest <= 1e-6 * step):
                owners.append(i)
        return owners[0] if len(owners) == 1 else None

    def _set_axis_unit_for_stack(self, unit: str) -> None:
        """:meth:`set_axis_unit` for a Compare -- a stack, an overlay or a
        panel Compare -- and for resolved components drawn the same way:
        rewrite **every** axes, not just ``self.ax``.

        Each spectrum is converted with its *own* axis, never a shared
        one: a Compare can hold spectra of different nuclei, fields or
        point counts, and its ppm-to-Hz conversion differs with each.
        Every line drawn from a spectrum (its data, a kept fit's curve,
        components and residual) goes with it, as does the view (zoom
        kept, in the new unit). On one shared axes (overlay, a panel
        of several) the lines of spectra whose conversions agree are
        converted together; where they disagree each line goes with the
        spectrum it is drawn from -- found by which axis holds its
        points -- and a line that cannot be told apart is left as it
        was. :meth:`compare_unit_note` says when a shared axes holds
        spectra whose Hz do not line up.

        An axes whose spectra cannot express the requested unit (an
        index axis, or mm with no gradient set) is left alone rather
        than the whole switch being refused -- one unconvertible row
        should not veto the others."""
        groups = self._compare_unit_groups()
        if not groups:
            return
        current = self._axis_unit
        spectra = self._compare_spectra
        now_values, maps = {}, {}
        for i, spectrum in enumerate(spectra):
            axis = spectrum.axes[0] if getattr(spectrum, "axes", None) else None
            if axis is None or getattr(spectrum, "ndim", 1) != 1:
                continue
            here = self._axis_values_in_unit(axis, current)
            there = self._axis_values_in_unit(axis, unit)
            mapping = self._affine_between(here, there)
            if mapping is not None:
                now_values[i] = np.asarray(here, dtype=float)
                maps[i] = mapping
        converted_any = False
        last_spectrum = None
        # New view limits are worked out for every axes first and set
        # afterwards: a stack's rows share their x-axis (and linked panels
        # follow each other), so setting one row's limits as we go would
        # have the next row's "current" limits already converted.
        new_limits = []
        for ax, indices in groups:
            usable = [i for i in indices if i in maps]
            if not usable:
                continue
            first = maps[usable[0]]
            agree = all(np.allclose(maps[i], first, rtol=1e-12, atol=1e-12) for i in usable)
            for target in (ax, *getattr(ax, "child_axes", ())):
                for line in target.lines:
                    if not self._x_in_data(line, target):
                        continue          # e.g. an axhline: x in axes coordinates
                    x = np.asarray(line.get_xdata(), dtype=float)
                    if x.size == 0:
                        continue
                    if agree:
                        mapping = first
                    else:
                        source = getattr(line, "_opennmr_source_spectrum", None)
                        owner = next((i for i in usable if spectra[i] is source), None) \
                            if source is not None else None
                        if owner is None:
                            owner = self._line_owner(x, usable, now_values)
                        if owner is None:
                            continue
                        mapping = maps[owner]
                    line.set_xdata(mapping[0] * x + mapping[1])
                lo, hi = target.get_xlim()
                slope, intercept = first
                new_limits.append((target, (slope * lo + intercept, slope * hi + intercept)))
            if ax.get_xlabel():
                if unit == "ppm":
                    label = self._chemical_shift_label(spectra[usable[-1]], x_unit="ppm")
                else:
                    label = self._unit_axis_label(unit)
                ax.set_xlabel(label, fontsize=self._AXIS_LABELSIZE)
            converted_any = True
            last_spectrum = spectra[usable[-1]]
        if not converted_any:
            return
        for target, limits in new_limits:
            target.set_xlim(*limits)
        if unit == "ppm" and last_spectrum is not None:
            self._ppm_label = self._chemical_shift_label(last_spectrum, x_unit="ppm")
        self._axis_unit = unit
        # Pinned to data coordinates in the old unit -- see set_axis_unit.
        self.clear_measurements(redraw=False)
        # The crosshair's lines live on these Axes too and would be
        # rewritten by the loop above -- removed, as on a 1D, so it
        # rebuilds against the new coordinates.
        self._remove_crosshair()
        self.canvas.draw_idle()

    def set_compare_unit(self, unit: str) -> None:
        """The Compare tile's Units box: convert the Compare on show, or
        -- with nothing drawn in a shift unit yet -- draw the next one in
        ``unit`` (show_compare and show_compare_panels convert to the unit
        in use)."""
        if unit not in ("ppm", "hz", "khz", "mm"):
            return
        if (self._compare_spectra and self._compare_unit_groups()) \
                or self._current_1d_spectrum is not None or self._contour_spectrum is not None:
            self.set_axis_unit(unit)
            return
        if unit != self._axis_unit:
            previous = self._axis_unit
            self._axis_unit = unit
            QTimer.singleShot(0, lambda: self._after_axis_unit_change(previous))

    def compare_unit_note(self, unit: str | None = None) -> str:
        """A warning for the Compare on show in ``unit`` (default: the one
        displayed), or "": in Hz, kHz or mm, spectra superimposed on one
        axes but recorded at different spectrometer frequencies (or
        referenced differently) each sit on their own frequency scale, so
        the same position on the axis is not the same chemical shift."""
        unit = unit or self._axis_unit
        if unit == "ppm":
            return ""
        spectra = self._compare_spectra
        for _ax, indices in self._compare_unit_groups():
            maps = []
            for i in indices:
                axis = spectra[i].axes[0] if getattr(spectra[i], "axes", None) else None
                mapping = self._affine_between(self._axis_values_in_unit(axis, "ppm"),
                                               self._axis_values_in_unit(axis, unit))
                if mapping is not None:
                    maps.append(mapping)
            if len(maps) > 1 and not all(np.allclose(m, maps[0], rtol=1e-9, atol=1e-9) for m in maps):
                return ("Spectra superimposed here have different spectrometer frequencies or "
                        "references: in this unit each is on its own frequency scale, so the "
                        "same position is not the same chemical shift. Use ppm to compare shifts.")
        return ""

    def _apply_view_options(self) -> None:
        # Every show_* rebuild routes through here after its own
        # figure.clear() -- any pivot marker from before is gone with
        # it, so the stale Python-side reference must be dropped too
        # (set_pivot_marker would otherwise try to move a Line2D no
        # longer attached to any axes).
        self._pivot_line = None
        self._zero_line = None
        self._pivot_marker_ax = None
        self._peak_marker_lines = []
        self._refresh_trace_names(restyle=False)
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
        # A stacked Compare has one axes per spectrum and ``self.ax`` is only
        # the last of them: locking just that one squeezed the bottom spectrum
        # into a small box while the rest kept the full width (user report,
        # 2026-11-23). Every row takes the same lock.
        for locked_ax in ([a for a in (self._compare_axes or []) if a in self.figure.axes] or [self.ax]):
            locked_ax.set_box_aspect(self._aspect_ratio)
        # matplotlib's own Axes.grid(visible, **kwargs) quirk: passing
        # any style kwarg (even alongside visible=False) is treated as
        # an implicit "turn it on" and warns -- only pass alpha when
        # actually enabling it.
        if getattr(self.ax, "name", "") != "3d":
            self.ax.grid(self._grid_visible, **({"alpha": 0.3} if self._grid_visible else {}))
        # Every rebuild, so a legend drawn by the plotting layer (which
        # knows nothing about these settings) still comes out matching
        # the ones drawn here -- see _restyle_existing_legends.
        self._restyle_existing_legends()
        # The title/font pass is in _apply_current_axis_style, which not
        # every rebuild reaches; a title set by a show_* that skips it
        # would otherwise appear despite the option being off.
        for ax in self.figure.get_axes():
            self._apply_title_and_font(ax)

    def set_pivot_marker(self, ppm: float | None) -> None:
        """Show, move, or hide a dashed vertical line at ``ppm`` -- the
        Phase 1 pivot point, visible while phasing instead of only a
        number in a spin box. ``None`` hides it. A no-op when the
        current view has nowhere sensible to draw it (see
        :attr:`_pivot_marker_ax`'s own docstring -- a 2D contour, FID,
        or Compare view, or the Processing-preview's own FID panel)."""
        if self._pivot_marker_ax is None:
            return
        self._sync_zero_line()
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

    def set_zero_line(self, on: bool) -> None:
        """Show or hide a thin y = 0 line across the spectrum -- the reference
        for judging a phase or a baseline. Not an ``axhline``: that is data, and
        would put 0 into the autoscale; this is pinned across the axes and
        added as an artist (the crosshair's guides are drawn the same way)."""
        self._zero_line_wanted = bool(on)
        self._sync_zero_line()

    def _sync_zero_line(self) -> None:
        from matplotlib.lines import Line2D
        ax = self._pivot_marker_ax
        wanted = self._zero_line_wanted and ax is not None
        if self._zero_line is not None and (not wanted or self._zero_line.axes is not ax):
            try:
                self._zero_line.remove()
            except (ValueError, NotImplementedError):
                pass
            self._zero_line = None
            self.canvas.draw_idle()
        if wanted and self._zero_line is None:
            line = Line2D([0.0, 1.0], [0.0, 0.0], transform=ax.get_yaxis_transform(),
                          color="0.45", linewidth=0.8, linestyle=(0, (4, 3)), zorder=1.5,
                          gid="opennmr_zero_line")
            ax.add_artist(line)
            self._zero_line = line
            self.canvas.draw_idle()

    def baseline_preview_source(self):
        """``(ppm, real data)`` of the 1D trace this canvas is
        currently showing, or ``None`` when it is not showing one.

        The scale a baseline overlay has to match is the *plotted*
        one, and only the canvas knows which spectrum it actually
        plotted -- a 2D contour has no trace to overlay at all, and the
        Processing preview plots a live-recomputed slice rather than
        anything in the session. Asking here keeps the caller from
        guessing, and from drawing a baseline computed off one array
        on top of another.
        """
        spectrum = self._preview_1d_spectrum or self._current_1d_spectrum
        if spectrum is None or self._pivot_marker_ax is None or spectrum.ndim != 1:
            return None
        return spectrum.axis_values(0, "ppm"), np.real(spectrum.data)

    def set_baseline_preview(self, x=None, baseline=None, uncorrected=None,
                             nodes=None) -> None:
        """Draw the baseline that *would* be subtracted over the trace
        it would be subtracted from, or clear it with no arguments.

        Explicit user request: "add a processing overlay to the plot
        showing the baseline corrections before they are committed".
        The point is to see whether the fit actually runs along the
        baseline -- or through the feet of the peaks, which is what
        every one of these estimators does when it goes wrong, and
        which is invisible once the correction has been applied and
        the evidence subtracted away.

        Drawn on :attr:`_pivot_marker_ax` rather than ``self.ax``, the
        same gate :meth:`set_peak_markers` uses: that is the spectrum
        panel in the Processing preview's two-panel layout and
        ``self.ax`` everywhere else, so one call does the right thing
        on both without the caller knowing which is showing.

        Found by gid rather than a stored reference, so a figure
        rebuild between calls cannot leave a stale Line2D behind.
        """
        stale = {self._BASELINE_PREVIEW_GID, self._BASELINE_GHOST_GID, self._BASELINE_NODES_GID}
        for ax in self.figure.get_axes():
            for line in list(ax.lines):
                if line.get_gid() in stale:
                    line.remove()
        target = self._pivot_marker_ax
        if x is None or baseline is None or target is None:
            self.canvas.draw_idle()
            return
        x = np.asarray(x, dtype=float)
        baseline = np.asarray(baseline, dtype=float)
        if x.size != baseline.size or x.size == 0:
            self.canvas.draw_idle()
            return
        if uncorrected is not None:
            # The trace the baseline was actually fitted to, faint and
            # behind everything. Without it, a correction that is
            # already applied leaves the overlay sitting on a flat line
            # with nothing to judge it against -- reported directly:
            # "seeing the baseline plotted against the already
            # corrected spectrum isn't so useful".
            ghost = np.asarray(uncorrected, dtype=float)
            if ghost.size == x.size:
                # Dashed, so it cannot be mistaken for a second
                # spectrum: it is the same data as the solid trace,
                # before a correction that has not been committed.
                # Explicit user request: "should use dashed lines".
                line = target.plot(
                    x, ghost, linewidth=1.0, color="0.55", linestyle=(0, (4, 2)),
                    zorder=1, label="_baseline_uncorrected",
                )[0]
                line.set_gid(self._BASELINE_GHOST_GID)
        # Dashed and coloured, over the trace rather than under it: the
        # whole job of this line is to be compared against the data, so
        # it has to stay visible where it crosses a peak.
        line = target.plot(
            x, baseline, linestyle="--", linewidth=1.4, color="tab:blue",
            zorder=5, label="_baseline_preview",
        )[0]
        line.set_gid(self._BASELINE_PREVIEW_GID)
        if nodes is not None:
            # The automatic spline's own nodes: the one thing to check
            # is that they sit on baseline, not on the foot of a peak.
            nx, ny = (np.asarray(v, dtype=float) for v in nodes)
            if nx.size and nx.size == ny.size:
                marks = target.plot(nx, ny, linestyle="none", marker="o", markersize=5,
                                    markerfacecolor="white", markeredgecolor="tab:blue",
                                    markeredgewidth=1.2, zorder=6,
                                    label="_baseline_nodes")[0]
                marks.set_gid(self._BASELINE_NODES_GID)
        self.canvas.draw_idle()

    #: gid of the automatic spline's node markers.
    _BASELINE_NODES_GID = "baseline_spline_nodes"

    #: gid of the manual-baseline region shading (see
    #: :meth:`set_baseline_regions`).
    _BASELINE_REGION_GID = "baseline_manual_region"

    def set_baseline_regions(self, regions=None) -> None:
        """Shade the ppm ranges a manual baseline is fitted to, on the
        same Axes the baseline overlay is drawn on; no argument clears
        them. Green-grey and behind everything, so they read as "these
        points are baseline" and not as integration regions (orange).
        Found by gid, like the overlay, so a figure rebuild cannot
        leave stale patches behind."""
        for ax in self.figure.get_axes():
            for patch in list(ax.patches):
                if patch.get_gid() == self._BASELINE_REGION_GID:
                    patch.remove()
        target = self._pivot_marker_ax
        if target is not None:
            for lo, hi in regions or []:
                span = target.axvspan(min(lo, hi), max(lo, hi), color="tab:green",
                                      alpha=0.12, lw=0, zorder=0)
                span.set_gid(self._BASELINE_REGION_GID)
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
        self._zero_line = None
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
        self._preview_1d_spectrum = None
        # Nothing on the canvas means nothing to switch between. Done
        # here rather than left to each caller: `clear` is already the
        # one thing every "this canvas now shows nothing" path goes
        # through, the same reasoning `_clear_figure` is the single
        # teardown point for the crosshair.
        self.set_view_switcher([])
        self._clear_figure()
        self.ax = self.figure.add_subplot(111)
        self.canvas.draw_idle()

    def show_spectrum(self, spectrum: NMRSpectrum) -> None:
        """Plot ``spectrum`` into this canvas, replacing whatever was
        there -- 1D via :func:`opennmr.plotting.plot_1d`, 2D via
        :func:`opennmr.plotting.plot_2d_contour`. Preserves this
        canvas's legend-visible/aspect-lock choices across the
        rebuild."""
        if spectrum.ndim in (1, 2):
            self._enter_plot_kind("1d" if spectrum.ndim == 1 else "2d")
        self._clear_figure()
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
            # One series, so the Plot Style dialog can name it (and
            # recolour it) like any other plot's series. Its colour is
            # the trace colour unless set here deliberately.
            # Keyed by the spectrum object (a reprocess redraws the same
            # one), so a colour set for one experiment does not follow
            # the view to the next.
            entry = self._assign_series(
                "spectrum", [(f"spectrum-{id(spectrum)}", _spectrum_label(spectrum), True)])[0]
            if not entry.colour_overridden:
                entry.colour = self._trace_color
            xmin, xmax, labeltext = plotting.plot_1d(
                self.ax, spectrum, full_resolution=self._full_resolution,
                color=entry.colour)
            # This canvas's own axis-label style/template, not
            # plot_1d's own (always-'brackets') default -- see
            # _chemical_shift_label.
            labeltext = self._chemical_shift_label(spectrum, x_unit="ppm")
            self._ppm_label = labeltext
            self.ax.lines[0].set_label(entry.label)
            plotting.safe_xlim(self.ax, xmin, xmax)
            self.ax.invert_xaxis()
            self._fit_y_to_data(self.ax)
            self.ax.set_xlabel(labeltext, fontsize=self._AXIS_LABELSIZE)
            self.ax.set_yticks([])
            plotting.apply_spine_style(self.ax, top=False, right=False, bottom=True, left=False,
                                        linewidth=self._AXIS_LINEWIDTH)
            plotting.apply_tick_style(self.ax, axis="x", length=self._TICK_LENGTH, width=self._TICK_WIDTH, minor=self._minor_ticks_visible,
                                       labelsize=self._AXIS_LABELSIZE)
        elif spectrum.ndim == 2:
            self._current_1d_spectrum = None
            # **Only reset the contour settings for a *different*
            # spectrum.** A live reprocess writes into the same
            # NMRSpectrum object and redraws it, so resetting here
            # unconditionally threw away the noise floor, level count,
            # scaling and exponent on every settle -- while
            # MainWindow._redraw_2d_preserving_view was carefully
            # preserving the zoom right beside it. User report: "the
            # contour levels and view shouldn't reset when processing".
            # Identity is the right test: a reprocess mutates in
            # place, a selection change hands over a new object.
            same_spectrum = self._contour_spectrum is spectrum
            self._contour_spectrum = spectrum
            if not same_spectrum:
                self._contour_insets = []        # a region of another spectrum
            if not same_spectrum or self._contour_factor is None:
                self._contour_factor = self._CONTOUR_FACTOR_DEFAULT
                self._contour_clevels = self._CONTOUR_CLEVELS_DEFAULT
                self._contour_level_mode = self._CONTOUR_LEVEL_MODE_DEFAULT
                self._contour_power = self._CONTOUR_POWER_DEFAULT
            self.ax = self.figure.add_subplot(111)
            display_spectrum, unit_xlabel, unit_ylabel = self._contour_for_display()
            xlims, ylims = self._contour_display_limits(display_spectrum)
            _, _, self._proj_axes = plotting.plot_2d_contour(
                display_spectrum, ax=self.ax, fig=self.figure, factor=self._contour_factor,
                xlabel=unit_xlabel, ylabel=unit_ylabel,
                clevels=self._contour_clevels, level_mode=self._contour_level_mode, power=self._contour_power,
                square_aspect=self._aspect_ratio is None,
                xlims=xlims, ylims=ylims,
                **self._contour_fraction_kwargs(), **self._nq_line_kwargs(),
                **self._slice_panel_kwargs(), **self._contour_style_kwargs(),
                colour_spectra=self._contour_colour, insets=self._contour_insets or None,
                **self._external_projection_kwargs(), **self._projection_style_kwargs(),
                **self._projection_mode_kwargs(),
            )
            self._apply_current_axis_style()
        else:
            raise ValueError(f"Cannot display a spectrum with ndim={spectrum.ndim}")

        self._apply_view_limits()
        self._apply_view_options()
        if spectrum.ndim == 1:
            self._pivot_marker_ax = self.ax
        # Crosshairs here too: reading a value off this plot is
        # exactly what it is for (1D spectra and 2D contours).
        self._crosshair_ready = True
        self._apply_layout()
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
        shape), ``"real"`` (one oscillating component, closer to what
        phase-sensitive detection actually recorded), or
        ``"real_imag"`` (both the real and imaginary components, each
        its own coloured line with a legend -- explicit user request:
        a raw FID's imaginary part carries real information (e.g. how
        well-balanced the quadrature channels are) that magnitude or
        real alone each throw away).

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
        self._preview_1d_spectrum = None
        self._clear_figure()
        self.ax = self.figure.add_subplot(111)
        self._freeform_zoom_axes = {self.ax}
        if x_unit not in ("time", "points"):
            raise ValueError("x_unit must be 'time' or 'points'")
        scale = dt if x_unit == "time" else 1.0
        t = np.arange(len(fid)) * scale
        if mode not in ("magnitude", "real", "real_imag"):
            raise ValueError("mode must be 'magnitude', 'real', or 'real_imag'")
        if mode == "real_imag":
            self.ax.plot(t, np.real(fid), color="black", label="Real")
            self.ax.plot(t, np.imag(fid), color=self._FID_IMAG_COLOR, label="Imaginary")
            self._legend_visible = True
        else:
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
        self._apply_layout()
        self.canvas.draw_idle()

    def processing_preview_limits(self):
        """The current ``((fid_xlim, fid_ylim), (spec_xlim, spec_ylim))``
        of the dual preview, or ``None`` when it is not showing one.

        Captured and restored around a rebuild by
        :meth:`show_processing_preview`'s own ``preserve_view`` -- see
        there for which rebuilds should use it.

        Gated on this canvas actually showing a dual preview, not just
        on it having two Axes: a contour with its two projection strips
        has three, and handing back *those* limits as "the preview's"
        would restore a contour's own framing onto a later FID panel."""
        if self._processing_preview_spec_ax is None:
            return None
        axes = self.figure.axes
        if len(axes) < 2:
            return None
        return tuple((ax.get_xlim(), ax.get_ylim()) for ax in axes[:2])

    def restore_processing_preview_limits(self, limits) -> None:
        """Put back what :meth:`processing_preview_limits` captured."""
        if not limits:
            return
        self.note_natural_view()  # the fresh draw's own view, before the kept zoom
        axes = self.figure.axes
        for ax, (xlim, ylim) in zip(axes[:2], limits):
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
        self.canvas.draw_idle()

    def show_processing_preview(self, fid, dt: float, spectrum: NMRSpectrum, *, title: str = "",
                                 window_envelope=None, tdeff_index: int | None = None,
                                 echo_top_index: float | None = None, fid_mode: str = "magnitude",
                                 fid_x_unit: str = "time", preserve_view: bool = False) -> None:
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
        stays full-resolution as normal. The downsampling itself is
        :func:`opennmr.plotting.style.downsample_1d`'s peak-preserving
        binning, not an arbitrary evenly-spaced pick, so this no longer
        trades correctness for speed -- a genuine peak stays visible
        here the same way it does on the (full-resolution) Spectrum
        tab, just possibly at a lower point density.

        ``fid_mode="real_imag"`` plots the FID's real and imaginary
        parts as two separate coloured lines with a small legend,
        instead of the usual single real/magnitude line -- see
        :meth:`show_fid`'s own ``mode`` docstring for the rationale.

        ``preserve_view`` : keep whatever zoom/pan the two panels
        already have, instead of refitting to the new data. Right for a
        rebuild that changes *what is plotted* without changing what
        the axes *mean* -- switching slice of a series, a periodic
        sidecar save's own refresh, or any FID-processing control
        (window, zero-fill, shift, TDeff, LP): none of those change
        what a ppm or a second on either axis is, so the region being
        looked at is still the region worth looking at.

        Wrong only where an axis genuinely changes meaning -- the FID
        panel's time/points unit switch -- where the old limits would
        frame a region that no longer exists."""
        previous_limits = self.processing_preview_limits() if preserve_view else None
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
        # The bottom panel's own spectrum, kept separately from
        # _current_1d_spectrum (which every other view uses and which
        # this layout deliberately leaves unset) so the baseline
        # overlay can find the trace it has to line up with here too.
        self._preview_1d_spectrum = spectrum
        self._clear_figure()
        fid_ax, spec_ax = self.figure.subplots(2, 1)
        self.ax = spec_ax
        self._freeform_zoom_axes = {fid_ax}
        self._processing_preview_fid_ax = fid_ax
        self._processing_preview_spec_ax = spec_ax

        if fid_x_unit not in ("time", "points"):
            raise ValueError("fid_x_unit must be 'time' or 'points'")
        fid_scale = dt if fid_x_unit == "time" else 1.0
        t = np.arange(len(fid)) * fid_scale
        if fid_mode not in ("magnitude", "real", "real_imag"):
            raise ValueError("fid_mode must be 'magnitude', 'real', or 'real_imag'")
        if fid_mode == "real_imag":
            fid_ax.plot(t, np.real(fid), color="black", linewidth=1.0, label="Real")
            fid_ax.plot(t, np.imag(fid), color=self._FID_IMAG_COLOR, linewidth=1.0, label="Imaginary")
        else:
            y = np.real(fid) if fid_mode == "real" else np.abs(fid)
            fid_ax.plot(t, y, color="black", linewidth=1.0)
        if window_envelope is not None:
            peak = np.max(np.abs(fid)) if len(fid) else 0.0
            fid_ax.plot(t, np.asarray(window_envelope) * peak, "--", color="gray", linewidth=1.0, zorder=0)
        if tdeff_index is not None and 0 < tdeff_index < len(fid):
            fid_ax.axvline(tdeff_index * fid_scale, linestyle="--", color="red")
        if echo_top_index is not None and 0 <= echo_top_index < len(fid):
            fid_ax.axvline(echo_top_index * fid_scale, linestyle="--", color="green")
        if fid_mode == "real_imag":
            fid_ax.legend(**self._legend_kwargs())
        fid_ax.set_xlabel("Time (s)" if fid_x_unit == "time" else "Points", fontsize=self._AXIS_LABELSIZE)
        fid_ax.set_ylabel("FID Intensity", fontsize=self._AXIS_LABELSIZE)
        if title:
            fid_ax.set_title(title, fontsize=self._AXIS_LABELSIZE)
        plotting.apply_spine_style(fid_ax, top=False, right=False, bottom=True, left=True,
                                    linewidth=self._AXIS_LINEWIDTH)
        plotting.apply_tick_style(fid_ax, length=self._TICK_LENGTH, width=self._TICK_WIDTH, minor=self._minor_ticks_visible,
                                   labelsize=self._AXIS_LABELSIZE)

        xmin, xmax, labeltext = plotting.plot_1d(spec_ax, spectrum, full_resolution=False)
        labeltext = self._chemical_shift_label(spectrum, x_unit="ppm")
        plotting.safe_xlim(spec_ax, xmin, xmax)
        spec_ax.invert_xaxis()
        spec_ax.set_xlabel(labeltext, fontsize=self._AXIS_LABELSIZE)
        spec_ax.set_yticks([])
        plotting.apply_spine_style(spec_ax, top=False, right=False, bottom=True, left=False,
                                    linewidth=self._AXIS_LINEWIDTH)
        plotting.apply_tick_style(spec_ax, axis="x", length=self._TICK_LENGTH, width=self._TICK_WIDTH, minor=self._minor_ticks_visible,
                                   labelsize=self._AXIS_LABELSIZE)

        self._apply_view_options()
        self._pivot_marker_ax = spec_ax
        self._apply_layout()
        if previous_limits:
            self.restore_processing_preview_limits(previous_limits)
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
        _, new_ydata = plotting.downsample_1d(None, y)
        if len(new_ydata) != len(self._processing_preview_spec_ax.lines[0].get_ydata()):
            return False
        self._processing_preview_spec_ax.lines[0].set_ydata(new_ydata)
        self.canvas.draw_idle()
        return True

    def show_compare(self, spectra: list[NMRSpectrum], mode: str = "stack", normalize: bool = False,
                      row_labels: list[str] | None = None, shared_scale: bool = False,
                      series_renameable: bool = False, plot_kind: str = "compare",
                      legend_labels: list[str] | None = None,
                      series_keys: list[str] | None = None) -> None:
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
        stacked-spectrum spacing "should be automatic".

        ``legend_labels`` -- each trace's name in the legend (without
        writing it onto the row, which is what ``row_labels`` does).
        ``series_keys`` -- a stable key per trace (default: its
        position), so a name or colour set on one trace stays with that
        trace when the list is reordered.

        ``plot_kind`` -- what :attr:`plotKindChanged` reports this as:
        ``"compare"``, or ``"pseudo2d"`` when the rows are the slices of
        one pseudo-2D spectrum, so each gets its own saved default
        style."""
        self._enter_plot_kind(plot_kind)
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
        self._clear_figure()
        # Every trace is drawn in ppm, so that is the display unit until
        # a stack is converted back to the one in use below. Left as it
        # was, a Compare drawn after switching a 1D to Hz was measured as
        # if its ppm were Hz (found 2026-11-12, checking Measure on every
        # plot).
        previous_unit = self._axis_unit
        self._axis_unit = "ppm"

        self._compare_kwargs = {"mode": mode, "normalize": normalize,
                                 "row_labels": row_labels, "shared_scale": shared_scale,
                                 "series_renameable": series_renameable,
                                 "plot_kind": plot_kind, "legend_labels": legend_labels,
                                 "series_keys": series_keys}
        # Both modes, not just "stack" -- the stack branch below set
        # this for its own unit-conversion needs, which left an overlay
        # Compare invisible to _active_series_view and so unstyleable.
        self._compare_spectra = list(spectra)
        # ``series_renameable`` because this renderer serves two quite
        # different things. The Compare tab plots session spectra, whose
        # names come from the spectrum and are changed by the Data tab's
        # Rename… (which reaches every other view too), so the plot must
        # not offer a competing local name. Resolved SCORE/OUTSCORE
        # components are drawn through the same renderer but are not
        # session spectra at all -- their labels are generated
        # annotations, plot-local, and theirs to rename.
        names = row_labels if row_labels else legend_labels
        entries = self._assign_series("compare", [
            ((series_keys[i] if series_keys and i < len(series_keys) else str(i)),
             (names[i] if names and i < len(names) else f"Trace {i + 1}"),
             series_renameable)
            for i in range(len(spectra))
        ])
        colors = [entry.colour for entry in entries]
        if row_labels is not None:
            row_labels = [entry.label for entry in entries]

        if mode == "stack" and len(spectra) > 1:
            axes = list(np.atleast_1d(self.figure.subplots(len(spectra), 1, sharex=True)))
            xmin = xmax = labeltext = None
            for i, (row_ax, spectrum) in enumerate(zip(axes, spectra)):
                this_xmin, this_xmax, labeltext = plotting.plot_1d(
                    row_ax, spectrum, normalize=normalize, color=colors[i], full_resolution=self._full_resolution,
                )
                # This canvas's own axis-label style/template -- see
                # _chemical_shift_label. Only the last row's own
                # label ends up used (set on axes[-1] below, since
                # the stack shares one x-axis), matching what was
                # already true of plot_1d's own returned labeltext.
                labeltext = self._chemical_shift_label(spectrum, x_unit="ppm")
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
                # Clip sideways to the row (so the zoom is what is seen
                # *and* what is exported -- an unclipped line made the
                # export's tight bounding box as wide as the whole
                # spectrum, reported 2026-10-22), but let a tall peak
                # spill up and down.
                self._clip_sideways_only(row_ax, row_ax.lines[-1])
                row_ax.lines[-1].set_label(entries[i].label)
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
            self._draw_compare_fit_overlays(spectra, axes, [ax.lines[0] for ax in axes],
                                            colors, [e.label for e in entries], normalize)
            # Each row's limits from its own full-resolution data (fit
            # overlays included), with Plot Style's gaps: below the
            # bottom row the axis gap, below every other row the gap
            # between stacked spectra.
            shared_range = None
            if shared_scale and not normalize:
                # One y-range across every row, so a component carrying
                # 10% of the signal is drawn a tenth as tall instead of
                # filling its own row exactly like a 100% one.
                #
                # Off by default and deliberately so: a Compare stack of
                # separately-acquired spectra has no shared intensity
                # scale to be meaningful *on* (different receiver gain,
                # different sample), and forcing one there would bury
                # every trace but the strongest. It is right for a set
                # of components resolved from one dataset, which do
                # share a scale by construction.
                # (A no-op when normalising: every trace already has the
                # same peak, so there are no relative heights to show.)
                ranges = [r for r in (self._data_y_range(a) for a in axes) if r is not None]
                if ranges:
                    shared_range = (min(r[0] for r in ranges), max(r[1] for r in ranges))
            self._compare_shared_range = shared_range
            self._fit_compare_rows(axes, shared_range)
            if row_labels:
                self._annotate_compare_rows(axes, row_labels, colors)
            plotting.safe_xlim(axes[0], xmin, xmax)
            axes[0].invert_xaxis()
            axes[-1].set_xlabel(labeltext, fontsize=self._AXIS_LABELSIZE)
            self.figure.subplots_adjust(hspace=0.0)
            self._compare_axes = axes
            # Kept so set_axis_unit can recompute every row's x data --
            # a stack has one Axes per spectrum, and rewriting only
            # self.ax (the last one) would silently convert one row and
            # leave the rest in the old unit.
            self._compare_spectra = list(spectra)
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
            # Each trace's name and colour from its series entry, as in
            # the stack -- the overlay used to take the colour map's
            # colours directly, so a colour set in Plot Style did not
            # reach it, and its lines had no names for a legend at all.
            data_lines = list(self.ax.lines[:len(spectra)])
            for line, entry, spectrum in zip(data_lines, entries, spectra):
                line.set_label(entry.label)
                line.set_color(entry.colour)
                # Which spectrum it shows -- see _set_axis_unit_for_stack.
                line._opennmr_source_spectrum = spectrum
            self._draw_compare_fit_overlays(spectra, [self.ax] * len(spectra), data_lines,
                                            colors, [e.label for e in entries], normalize)
            self._compare_shared_range = None
            self._fit_y_to_data(self.ax)
            self.ax.set_yticks([])
            plotting.apply_spine_style(self.ax, top=False, right=False, bottom=True, left=False,
                                        linewidth=self._AXIS_LINEWIDTH)
            plotting.apply_tick_style(self.ax, axis="x", length=self._TICK_LENGTH, width=self._TICK_WIDTH, minor=self._minor_ticks_visible,
                                       labelsize=self._AXIS_LABELSIZE)
            self.ax.xaxis.label.set_fontsize(self._AXIS_LABELSIZE)

        if previous_unit != "ppm" and self._compare_spectra:
            self._set_axis_unit_for_stack(previous_unit)
        self._apply_view_limits()
        self._apply_view_options()
        # Crosshairs here too: reading a value off this plot is
        # exactly what it is for (Compare).
        self._crosshair_ready = True
        self._apply_layout()
        self.canvas.draw_idle()

    # ---- Compare as separate panels (row / column / grid) -----------------

    def show_compare_panels(self, spectra: list, groups: list, *, layout: str = "row",
                            ncols: int | None = None, linked: bool = True, link_y: bool = False,
                            normalize: bool = False, series_keys: list | None = None,
                            legend_labels: list | None = None, axis_labels: str = "auto",
                            axis_styles=None, twod_options: dict | None = None,
                            pulse_panels: dict | None = None,
                            panel_labels: dict | None = None,
                            renameable: list | None = None) -> None:
        """Compare spectra as separate panels. ``renameable``: per spectrum, whether
        its legend name may be typed over on the plot (a slice of a pseudo-2D has no
        experiment of its own to rename; a whole spectrum is renamed in the list). ``groups`` says which of
        ``spectra`` (indices) share a panel: ``[[0, 1], [2]]`` is two panels,
        the first holding two superimposed spectra. Names and colours come
        from the series entries, so Plot Style and Rename reach every panel;
        a spectrum made from a kept fit brings its fit, components and
        residual. ``linked`` ties the panels' shift axes together (and
        ``link_y`` the intensity axes); see :meth:`set_spectrum_panels_linked`.

        A panel holds 1D spectra (superimposed) **or** 2D spectra (a contour,
        or an overlay of 2-6), so 1D and 2D can share one figure; ``twod_options``
        are extra :func:`opennmr.plotting.twod.plot_2d_contour` keywords for the
        2D panels (``factor``, ``clevels``, ``show_projections``...).
        ``pulse_panels`` -- ``{spectrum index: {"sequence": PulseSequence | None,
        "options": {...}, "message": "..."}}`` -- turns the (single-spectrum)
        panel of that index into the spectrum's pulse-sequence diagram
        (``sequence`` None: a note saying why there is none).
        ``panel_labels``: :func:`opennmr.notebook.panel_labels.add_panel_labels`
        keywords for (a), (b)... (None: no labels)."""
        from ...notebook.compare_panels import apply_axis_styles, plot_compare_panels
        groups = [list(g) for g in groups]
        pulse_panels = dict(pulse_panels or {})
        kwargs = {"layout": layout, "ncols": ncols, "linked": linked, "link_y": link_y,
                  "normalize": normalize, "series_keys": series_keys, "legend_labels": legend_labels,
                  "axis_labels": axis_labels, "axis_styles": axis_styles,
                  "twod_options": twod_options, "pulse_panels": pulse_panels,
                  "panel_labels": panel_labels, "renameable": renameable}
        self._enter_plot_kind("compare")
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
        was_panels = bool(self._spectrum_panels)
        # Drawn in ppm, then converted back to the unit in use (as
        # show_compare does) -- left as it was, a panel Compare drawn
        # after switching to Hz was read as if its ppm were Hz.
        previous_unit = self._axis_unit
        self._axis_unit = "ppm"
        self._clear_figure()
        if not was_panels:
            self._legend_visible = True         # panels hold different spectra: name them
        self._compare_panels_kwargs = kwargs
        self._panel_label_args = dict(panel_labels) if panel_labels else None
        self._compare_kwargs = None
        self._compare_spectra = list(spectra)
        entries = self._assign_series("compare", [
            ((series_keys[i] if series_keys and i < len(series_keys) else str(i)),
             (legend_labels[i] if legend_labels and i < len(legend_labels) else f"Trace {i + 1}"),
             bool(renameable[i]) if renameable and i < len(renameable) else False)
            for i in range(len(spectra))
        ])
        panels = []
        for g in groups:
            if len(g) == 1 and g[0] in pulse_panels:
                info = pulse_panels[g[0]]
                panels.append({"pulse_sequence": info.get("sequence"), "options": info.get("options") or {},
                               "message": info.get("message")})
            else:
                panels.append({"spectra": [spectra[i] for i in g],
                               "labels": [entries[i].label for i in g],
                               "colors": [entries[i].colour for i in g]})
        twod = {"tick_length": self._TICK_LENGTH, "tick_width": self._TICK_WIDTH,
                "contour_linewidth": self._contour_linewidth, "colour_spectra": self._contour_colour,
                "factor": self._CONTOUR_FACTOR_DEFAULT, "clevels": self._CONTOUR_CLEVELS_DEFAULT,
                **self._contour_fraction_kwargs(), **self._nq_line_kwargs()}
        twod.update({k: v for k, v in (twod_options or {}).items() if v is not None})
        label_args = dict(panel_labels) if panel_labels else None
        if label_args is not None:
            label_args.setdefault("fontsize", self._AXIS_LABELSIZE * 1.15)
        result = plot_compare_panels(
            panels, layout=layout, ncols=ncols, link=linked, link_y=link_y, normalize=normalize,
            full_resolution=self._full_resolution, fig=self.figure, fontsize=self._AXIS_LABELSIZE,
            tight=False, model_for=self.compare_fit_model, axis_labels=axis_labels,
            axis_styles=axis_styles, twod_options=twod, panel_labels=label_args)
        for ax in result.axes:
            ax.patch.set_alpha(0.0)
            if getattr(ax, "_opennmr_panel_kind", "1d") != "1d":
                continue                  # a 2D contour and a pulse diagram keep their own furniture
            plotting.apply_spine_style(ax, top=False, right=False, bottom=True, left=False,
                                       linewidth=self._AXIS_LINEWIDTH)
            plotting.apply_tick_style(ax, axis="x", length=self._TICK_LENGTH, width=self._TICK_WIDTH,
                                      minor=self._minor_ticks_visible, labelsize=self._AXIS_LABELSIZE)
        # the canvas's own tick defaults were just applied: the user's per-panel
        # choices go on top of them
        apply_axis_styles(result.axes, axis_styles, self._AXIS_LABELSIZE)
        self._spectrum_panels = list(result.axes)
        self._spectrum_panel_groups = groups
        self._spectrum_panel_link = result.link
        self._compare_shared_range = None
        self.ax = result.axes[0]
        if previous_unit != "ppm" and self._compare_spectra:
            self._set_axis_unit_for_stack(previous_unit)
        self._apply_view_options()
        self._crosshair_ready = True
        self._apply_layout()
        self.canvas.draw_idle()

    def spectrum_panels_linked(self) -> tuple:
        """``(x_linked, y_linked)`` of the current panel Compare, or (False, False)."""
        link = self._spectrum_panel_link
        return (bool(link.x), bool(link.y)) if link is not None else (False, False)

    def set_spectrum_panels_linked(self, linked: bool | None = None, link_y: bool | None = None) -> None:
        """Link or unlink the panels' shift (``linked``) / intensity (``link_y``)
        axes without redrawing; turning a link on puts every panel on the
        first one's current view."""
        link = self._spectrum_panel_link
        if link is None:
            return
        link.set(x=linked, y=link_y)
        if self._compare_panels_kwargs is not None:
            if linked is not None:
                self._compare_panels_kwargs["linked"] = bool(linked)
            if link_y is not None:
                self._compare_panels_kwargs["link_y"] = bool(link_y)
        self.canvas.draw_idle()

    # ---- kept fits in Compare --------------------------------------------

    _fit_model_cache: dict = {}

    @classmethod
    def compare_fit_model(cls, spectrum):
        """The kept fit a Compare entry was made from (decoded, cached),
        or None -- an Open experiment made by "Keep in Open Experiments"
        on a kept fit carries it as ``metadata.raw["fit_model"]``."""
        raw = getattr(getattr(spectrum, "metadata", None), "raw", None) or {}
        encoded = raw.get("fit_model")
        if not encoded:
            return None
        key = id(encoded)
        hit = cls._fit_model_cache.get(key)
        if hit is None or hit[0] is not encoded:
            from ...notebook.fit_models import decode_fit_model
            try:
                hit = (encoded, decode_fit_model(encoded))
            except Exception:  # noqa: BLE001 -- a damaged record draws as plain data
                return None
            if len(cls._fit_model_cache) > 64:
                cls._fit_model_cache.clear()
            cls._fit_model_cache[key] = hit
        return hit[1]

    def _draw_compare_fit_overlays(self, spectra, axes, data_lines, colors, names,
                                   normalize: bool) -> None:
        """Each Compare entry made from a kept fit gets its fit drawn over
        its data (see :func:`opennmr.notebook.compare_panels.draw_kept_fit_overlays`,
        which the notebook's panel Compare shares)."""
        from ...notebook.compare_panels import draw_kept_fit_overlays
        draw_kept_fit_overlays(spectra, axes, data_lines, colors, names, normalize,
                               model_for=self.compare_fit_model)

    @staticmethod
    def _apply_shared_row_scale(axes, spectra, normalize: bool) -> None:
        """Give every row of a stack the same y-range, so their heights
        are comparable rather than each row being autoscaled to fill
        itself.

        A no-op when ``normalize`` is on: that has already scaled every
        trace to the same peak, so there are no relative heights left
        to show and applying a shared range on top would just be
        confusing about which of the two settings is in force.
        """
        if normalize:
            return
        lows, highs = [], []
        for spectrum in spectra:
            data = np.real(np.asarray(spectrum.data, dtype=float))
            if data.size:
                lows.append(float(np.min(data)))
                highs.append(float(np.max(data)))
        if not highs:
            return
        low, high = min(lows), max(highs)
        if high <= low:
            return
        pad = (high - low) * 0.05
        for row_ax in axes:
            row_ax.set_ylim(low - pad, high + pad)

    def _annotate_compare_rows(self, axes, row_labels, colors) -> None:
        """Write each row's own label onto the row itself.

        Drawn in the row's own colour and pinned to the axes rather
        than the data, so it stays put through pan/zoom -- the point is
        to identify which trace is which without cross-referencing a
        list somewhere else, which stops working the moment the figure
        is exported."""
        for row_ax, label, color in zip(axes, row_labels, colors):
            if not label:
                continue
            row_ax.text(
                0.995, 0.92, label, transform=row_ax.transAxes,
                ha="right", va="top", color=color, fontsize=self._AXIS_LABELSIZE * 0.65,
                zorder=6,
                # Opaque, not translucent: with a shared scale the
                # tallest row's peaks reach the top of their axes, and
                # a see-through box let a peak spike straight through
                # the text. There is no corner of a spectrum that is
                # reliably empty, so the label wins over whatever is
                # under it rather than being placed around it.
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                           edgecolor="none", alpha=1.0),
            )

    def show_2d_overlay(self, spectra: list[NMRSpectrum], labels: list[str] | None = None,
                        insets=None, factor: float | None = None, clevels: int | None = None,
                        projections: bool = False) -> None:
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
        self._enter_plot_kind("2d")   # its saved default style (Plot Style ▸ Per-type)
        self.clear()  # already leaves self.ax as a fresh, single subplot
        self._overlay_spectra = list(spectra)
        self._overlay_labels = list(labels) if labels is not None else None
        self._overlay_kwargs = {"insets": insets, "factor": factor, "clevels": clevels}
        levels = {"factor": factor or self._CONTOUR_FACTOR_DEFAULT,
                  "clevels": clevels or self._CONTOUR_CLEVELS_DEFAULT}
        entries = self._assign_series("overlay_2d", [
            (str(i), (labels[i] if labels and i < len(labels) else f"Spectrum {i + 1}"), True)
            for i in range(len(spectra))
        ])
        plotting.overlay_plot(
            spectra, ax=self.ax, fig=self.figure,
            labels=[entry.label for entry in entries] if labels is not None else None,
            colours=[entry.colour for entry in entries], insets=insets,
            fontsize=self._AXIS_LABELSIZE, contour_linewidth=self._contour_linewidth,
            show_projections=bool(projections),
            **levels, **self._contour_fraction_kwargs())
        strips = getattr(self.ax, "opennmr_projection_axes", None) or {}
        self._proj_axes = {"f1": strips.get("f1"), "f2": strips.get("f2")}
        if labels is not None:
            self.ax.legend(**self._legend_kwargs())
            self._legend_visible = True
        self._apply_view_options()
        # Crosshairs here too: reading a value off this plot is
        # exactly what it is for (2D overlay).
        self._crosshair_ready = True
        self._apply_layout()
        self._panel_axes = [self.ax]
        self._panel_home_limits = [(self.ax.get_xlim(), self.ax.get_ylim())]
        self._panel_spectra = list(spectra)
        self.canvas.draw_idle()

    def show_2d_panels(self, spectra: list[NMRSpectrum], labels: list[str] | None = None,
                       layout: str = "row", projections: bool = False, link_axes: bool = False,
                       insets=None, factor: float | None = None, clevels: int | None = None,
                       panel_labels: dict | None = None) -> None:
        """2D spectra side by side -- one contour panel each, in a row,
        column or grid (:func:`opennmr.plotting.twod.multi_plot`, the
        GUI's port of the notebook's ``NMR2D_multi_plot``).

        ``insets`` : a list per panel (``None``/``[]`` for none) of
        :func:`opennmr.plotting.twod.draw_insets` specs.
        ``link_axes`` : every panel starts on the range they all span
        and zooming one zooms them all. Done with limit callbacks, not
        matplotlib's ``sharex``: each panel inverts its own axes, and
        inverting a shared axis once per panel flips it back and forth.

        ``panel_labels`` : :func:`opennmr.notebook.panel_labels.add_panel_labels`
        keywords -- (a), (b)... on the panels (None: none).
        """
        self._enter_plot_kind("2d")   # its saved default style (Plot Style ▸ Per-type)
        self.clear()
        self._clear_figure()
        n = len(spectra)
        insets = list(insets) if insets is not None else [None] * n
        levels = {"factor": factor or self._CONTOUR_FACTOR_DEFAULT,
                  "clevels": clevels or self._CONTOUR_CLEVELS_DEFAULT}
        panel_kwargs = [{"insets": insets[i] or None} for i in range(n)]
        _fig, axes = plotting.multi_plot(
            spectra, layout=layout, fig=self.figure, titles=labels, panel_kwargs=panel_kwargs,
            show_projections=projections, square_aspect=True,
            fontsize=self._AXIS_LABELSIZE, tick_length=self._TICK_LENGTH,
            tick_width=self._TICK_WIDTH, contour_linewidth=self._contour_linewidth,
            colour_spectra=self._contour_colour,
            **levels, **self._contour_fraction_kwargs(), **self._nq_line_kwargs())
        self.ax = axes[0]
        same_f1 = len({bool(sp.is_pseudo_2d) for sp in spectra}) == 1
        self._panel_spectra = list(spectra)
        self._panels_link_y = same_f1
        self._panels_linked = bool(link_axes and n > 1)
        if self._panels_linked:
            self._unify_panel_limits(axes, spectra)
        if n > 1:
            self._link_panel_limits(axes)
        self._panel_axes = list(axes)
        self._panel_home_limits = [(ax.get_xlim(), ax.get_ylim()) for ax in axes]
        self._panel_label_args = dict(panel_labels) if panel_labels else None
        if panel_labels and n > 1:
            from ...notebook.panel_labels import add_panel_labels
            label_args = dict(panel_labels)
            label_args.setdefault("fontsize", self._AXIS_LABELSIZE * 1.15)
            add_panel_labels(self.figure, list(axes), **label_args)
        # A crosshair here too (Compare 2D): in every panel while their
        # spans match, in the panel under the cursor alone otherwise.
        self._crosshair_ready = True
        self._apply_layout()
        self.canvas.draw_idle()

    @staticmethod
    def _unify_panel_limits(axes, spectra) -> None:
        """Every panel on the range they all span (x always; y when they
        share the kind of F1 axis)."""
        xs = [lim for ax in axes for lim in ax.get_xlim()]
        for ax in axes:
            ax.set_xlim(max(xs), min(xs))
        if len({bool(sp.is_pseudo_2d) for sp in spectra}) == 1:
            ys = [lim for ax in axes for lim in ax.get_ylim()]
            pseudo = spectra[0].is_pseudo_2d
            for ax in axes:
                ax.set_ylim((min(ys), max(ys)) if pseudo else (max(ys), min(ys)))

    def panels_linked(self) -> bool:
        """Whether zooming or panning one Compare 2D panel moves them all."""
        return bool(self._panels_linked)

    def set_panels_linked(self, linked: bool) -> None:
        """Link the side-by-side 2D panels (a zoom, pan or axis scroll on
        one is applied to all) or unlink them (only the one acted on
        moves). Takes effect at once, without redrawing; linking puts every
        panel on the first one's current view."""
        linked = bool(linked) and len(self._panel_axes) > 1
        if linked and not self._panels_linked and self._panel_axes:
            first = self._panel_axes[0]
            self._panels_linked = True
            self._follow_panel_limits(first, "x")
            if self._panels_link_y:
                self._follow_panel_limits(first, "y")
            self.canvas.draw_idle()
        self._panels_linked = linked

    def _follow_panel_limits(self, source, which: str) -> None:
        if not self._panels_linked or self._panels_busy:
            return
        self._panels_busy = True
        try:
            if which == "y" and not self._panels_link_y:
                return
            limits = source.get_xlim() if which == "x" else source.get_ylim()
            for other in self._panel_axes:
                if other is not source:
                    (other.set_xlim if which == "x" else other.set_ylim)(limits)
        finally:
            self._panels_busy = False

    def _link_panel_limits(self, axes) -> None:
        """Connect the panels' limit callbacks once per draw; whether they
        follow each other is :meth:`set_panels_linked`'s flag, so the link
        can be switched without redrawing."""
        for ax in axes:
            ax.callbacks.connect("xlim_changed", lambda a: self._follow_panel_limits(a, "x"))
            ax.callbacks.connect("ylim_changed", lambda a: self._follow_panel_limits(a, "y"))

    def zoomed_2d_region(self):
        """``(panel, xlims, ylims)`` of the 2D overlay/panel the user has
        zoomed into (0-based panel), or ``None`` if none is zoomed. With
        linked panels every panel moves together; the first is reported."""
        for i, (ax, (x0, y0)) in enumerate(zip(self._panel_axes, self._panel_home_limits)):
            x, y = ax.get_xlim(), ax.get_ylim()
            if not (np.allclose(x, x0) and np.allclose(y, y0)):
                return i, tuple(float(v) for v in x), tuple(float(v) for v in y)
        return None

    def show_1d_heatmap(self, spectra: list[NMRSpectrum], normalize: bool = True,
                        row_labels: list[str] | None = None,
                        colorbar_label: str | None = None,
                        first_row_on_top: bool = False) -> None:
        """Plot several 1D spectra (a pseudo-2D series' own slices) as
        rows of a heatmap -- real roadmap ask: "pseudo-2D/stacked-1D
        data should be plottable as either a contour or a heatmap."
        Thin wrapper around :func:`opennmr.plotting.plot_1d_heatmap`
        (already ported from ``nmrtools2``'s own ``NMR1dheatmap``, just
        never called from GUI code before this).

        Not a :attr:`_contour_spectrum` either, same reasoning as
        :meth:`show_2d_overlay` -- this is several interpolated-onto-
        one-axis 1D traces, not one spectrum's own 2D array.

        Used by the Compare tile's "Heatmap" layout too, which passes
        ``normalize`` (its Normalize choice), ``row_labels`` (one per
        row -- every row labelled up to 15, an even subset beyond),
        ``colorbar_label`` (a colour bar is drawn whenever one is given:
        a heatmap without a scale cannot be read quantitatively) and
        ``first_row_on_top`` (so rows read in the list's own order)."""
        self._enter_plot_kind("pseudo2d")   # its saved default style (Plot Style ▸ Per-type)
        self.clear()  # already leaves self.ax as a fresh, single subplot
        tick_mode = "none"
        if row_labels:
            tick_mode = "all" if len(row_labels) <= 15 else "auto"
        self._heatmap_spectra = list(spectra)
        plotting.plot_1d_heatmap(
            spectra, ax=self.ax, fig=self.figure, normalize=normalize,
            y_values=row_labels, y_ticks=tick_mode,
            colorbar=colorbar_label is not None, colorbar_label=colorbar_label)
        if first_row_on_top:
            self.ax.invert_yaxis()
        self._apply_view_options()
        # Crosshairs here too: reading a value off this plot is
        # exactly what it is for (heatmap).
        self._crosshair_ready = True
        self._apply_layout()
        self.canvas.draw_idle()

    def show_slice_fit(self, x, y_data, y_fit, components=None, peak_labels=None, title: str = "",
                        spectrum: NMRSpectrum | None = None, auto_hide_single: bool = False,
                        overlay=None, _keep_view: bool = False,
                        fit_label: str = "Fit (total)", keep_zoom: bool = False) -> None:
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

        How the components are drawn -- all, none, only highlighted
        ones, filled or as lines, and their colours -- is this canvas's
        component style (:meth:`set_fit_component_style`), drawn by
        :func:`opennmr.notebook.fit_display.draw_fit_components`, the
        same function a notebook figure uses. ``auto_hide_single``: a
        lone component only repeats the fit, so the default ("auto")
        leaves it out; "all" still shows it.

        ``spectrum`` : the slice's own 1D trace (``fit_result["slice"]``
        from :func:`opennmr.processing.fitting.NMR2D_fit_slices`), used
        only for its nucleus/SFO1 -- to label and orient the x-axis
        exactly like :meth:`show_spectrum` does (inverted, high-to-low
        ppm, with a nucleus-based label) rather than the plain
        unlabelled, un-inverted axis this used to have. ``x`` itself is
        always ppm (:func:`NMR2D_fit_slices`'s own default).

        ``overlay`` : optional ``overlay(ax)``, drawn on top once the fit
        is (annotations, markers -- e.g. pNMR environment labels). It is
        kept with the fit and drawn again whenever the fit is redrawn (a
        component-style or Plot Style change), and dropped when another
        plot replaces the fit. It never stops the fit itself being shown.

        ``keep_zoom``: a redraw of the same fit that is still being worked
        on (the live preview of a starting model, then the fit itself) keeps
        the user's zoom, if they zoomed or panned this fit and the zoomed
        range still overlaps the new data -- explicit user request,
        2026-10-13: "do not keep auto-zooming out on updating". Home still
        returns to the full fit window (the natural view is recorded
        again for the new Axes)."""
        previous_overlay = self._slice_fit_overlay
        # Only a redraw of *this same fit* (a component-style or Plot
        # Style change, which passes ``_keep_view`` itself) carries its
        # overlay over. A new fit that merely keeps the zoom
        # (``keep_zoom``) brings its own overlay -- or none. It used to
        # inherit the previous one whenever the zoom was kept, so the
        # grey "Not fitted" shading of a several-range fit stayed on
        # after the next fit used one range or the whole spectrum
        # (reported 2026-10-21).
        redraw_of_same_fit = bool(_keep_view)
        if keep_zoom and not _keep_view and self._slice_fit_zoom_to_keep(x):
            _keep_view = True
        self._enter_plot_kind("1d")   # its saved default style (Plot Style ▸ Per-type)
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
        self._preview_1d_spectrum = None
        view = (self.ax.get_xlim(), self.ax.get_ylim()) if _keep_view and self.ax is not None else None
        self._clear_figure()
        self.ax = self.figure.add_subplot(111)
        self._slice_fit_args = dict(x=x, y_data=y_data, y_fit=y_fit, components=components,
                                    peak_labels=peak_labels, title=title, spectrum=spectrum,
                                    auto_hide_single=auto_hide_single, fit_label=fit_label)
        self._slice_fit_active = True
        n_peaks = len(components) - 1 if components else 0
        # One name per component: extra names are dropped and missing
        # ones numbered, as this view always tolerated.
        given = [str(v) for v in (peak_labels or [])][:n_peaks]
        self._slice_fit_labels = given + [f"Peak {i + 1}" for i in range(len(given), n_peaks)]
        fit_display.draw_fit_components(
            self.ax, x, y_data, y_fit, components=components,
            labels=self._slice_fit_labels or None, style=self._fit_component_style,
            auto_hide_single=auto_hide_single, fit_label=fit_label)

        if spectrum is not None:
            xlabel = self._chemical_shift_label(spectrum, x_unit="ppm")
        else:
            xlabel = self._unit_label("Shift", "ppm")
        xmin, xmax = float(np.min(x)), float(np.max(x))
        plotting.safe_xlim(self.ax, xmin, xmax)
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
        if view is not None:
            # The new Axes' own limits are what Home goes back to.
            self.note_natural_view()
            self.ax.set_xlim(view[0])
            self.ax.set_ylim(view[1])
        # Crosshairs here too: reading a value off this plot is
        # exactly what it is for (spectral fit).
        self._crosshair_ready = True
        self._apply_layout()
        self._slice_fit_overlay = overlay if overlay is not None else (
            previous_overlay if redraw_of_same_fit else None)
        self.draw_slice_fit_overlay()
        self.canvas.draw_idle()
        self.fitComponentsShown.emit(list(self._slice_fit_labels))

    def _slice_fit_zoom_to_keep(self, x) -> bool:
        """Whether a redraw of the fit on show should keep the view: a
        spectral fit is on show, the user has zoomed or panned it (its
        natural view was recorded and differs from the view now), and the
        zoomed x-range still overlaps ``x``."""
        if not self._slice_fit_active or self.ax is None or not self._home_limits:
            return False
        natural = self._home_limits.get(self.ax)
        if natural is None:
            return False
        now = (self.ax.get_xlim(), self.ax.get_ylim())
        if np.allclose(natural[0], now[0]) and np.allclose(natural[1], now[1]):
            return False                      # back at Home: nothing to keep
        lo, hi = sorted(now[0])
        x = np.asarray(x, dtype=float)
        return bool(x.size and np.nanmax(x) >= lo and np.nanmin(x) <= hi)

    def slice_fit_overlay(self):
        """The overlay drawn over the fit on show (see show_slice_fit), or None."""
        return self._slice_fit_overlay if self._slice_fit_active else None

    def draw_slice_fit_overlay(self) -> bool:
        """Draw the fit's overlay on its Axes (once more). False if there is
        none, or it failed -- a broken overlay must not take the fit with it."""
        overlay = self.slice_fit_overlay()
        if overlay is None or self.ax is None:
            return False
        try:
            overlay(self.ax)
        except Exception as exc:  # noqa: BLE001 -- annotations only; the fit stays up
            warnings.warn(f"Fit overlay not drawn: {exc}", RuntimeWarning, stacklevel=2)
            return False
        return True

    def shown_fit(self) -> dict | None:
        """The fit on show, as ``{"kind": "spectrum" | "curve", ...}`` --
        a spectral fit (``show_slice_fit``'s arguments) or a curve fit
        (data, fit and axis labels) -- or None."""
        if self._slice_fit_active and self._slice_fit_args is not None:
            return {"kind": "spectrum", "labels": list(self._slice_fit_labels),
                    **self._slice_fit_args}
        if self._curve_fit_shown is not None:
            return {"kind": "curve", **self._curve_fit_shown}
        if self._curve_fits_shown:
            return {"kind": "curves", "fits": list(self._curve_fits_shown)}
        return None

    def fit_component_style(self) -> dict:
        """This canvas's component style (a copy)."""
        return fit_display.component_style(self._fit_component_style)

    def fit_component_labels(self) -> list:
        """The components of the fit on show ([] when none is)."""
        return list(self._slice_fit_labels) if self._slice_fit_active else []

    def set_fit_component_style(self, style: dict | None = None, **changes) -> None:
        """Change how fit components are drawn (keys as
        :data:`opennmr.notebook.fit_display.DEFAULT_COMPONENT_STYLE`) and
        redraw the fit on show, keeping the zoom."""
        self._fit_component_style = fit_display.component_style(
            style if style is not None else self._fit_component_style, **changes)
        if self._slice_fit_active and self._slice_fit_args is not None:
            self.show_slice_fit(**self._slice_fit_args, _keep_view=True)

    def show_dosy(self, dosy: dict, title: str = "", factor: float | None = None,
                   clevels: int | None = None, d_lims=None) -> None:
        """Draw a DOSY map (shift across, D down a log axis) from
        :func:`opennmr.processing.ilt.dosy_map`'s own output.

        Routed through :func:`opennmr.plotting.plot_dosy` rather than
        drawn here, so the on-screen DOSY and an exported one are the
        same figure -- the whole reason the ax-first plotting functions
        exist (see this module's own docstring).

        The map and its level floor are kept on the canvas so a DOSY
        behaves like a contour: scrolling its body scales the levels,
        scrolling either axis strip zooms that axis alone, and
        right-clicking offers to extract a trace. Deliberately *not*
        stored in ``_contour_spectrum``/``_contour_factor``: a DOSY is
        not an ``NMRSpectrum``, and its redraw goes through
        ``plot_dosy`` rather than ``plot_2d_contour``, so sharing that
        state would send every contour path down the wrong branch."""
        self._enter_plot_kind("2d")   # its saved default style (Plot Style ▸ Per-type)
        self._dosy_result = dosy
        self._dosy_overlay = None
        self._dosy_overlay_labels = None
        self._dosy_factor = self._DOSY_FACTOR_DEFAULT if factor is None else float(factor)
        self._dosy_clevels = self._DOSY_CLEVELS_DEFAULT if clevels is None else int(clevels)
        self._dosy_d_lims = None if d_lims is None else (float(d_lims[0]), float(d_lims[1]))
        self._dosy_title = title
        self._draw_dosy()

    def show_dosy_overlay(self, maps, labels=None, title: str = "", factor: float | None = None,
                           clevels: int | None = None, d_lims=None) -> None:
        """Draw several DOSY maps on one set of axes, each in its own
        colour, via :func:`opennmr.plotting.overlay_dosy`.

        Explicit user request: "I'd also like to be able to overlay
        DOSY plot ILT results". Reading whether a shift has moved up or
        down the diffusion axis between two samples off two separate
        figures -- each with its own independently fitted D axis -- is
        how that comparison goes wrong.

        Everything the single-map view supports still applies: the
        level floor and count come from the same panel controls, the
        scroll gesture still scales the levels, and the D axis can
        still be pinned. ``_dosy_result`` is left pointing at the first
        map so the right-click trace extraction still has one
        unambiguous map to extract from."""
        self._enter_plot_kind("2d")   # its saved default style (Plot Style ▸ Per-type)
        maps = list(maps)
        if len(maps) < 2:
            raise ValueError(f"show_dosy_overlay needs at least 2 maps, got {len(maps)}.")
        self._dosy_result = maps[0]
        self._dosy_overlay = maps
        self._dosy_overlay_labels = list(labels) if labels is not None else None
        if labels is not None:
            # On, not left to the canvas-wide toggle's default: the
            # legend is the only thing distinguishing one map from
            # another here, so an overlay without it is unreadable.
            # It is still the user's own toggle, so it can be switched
            # back off from the right-click menu afterwards.
            self._legend_visible = True
        self._dosy_factor = self._DOSY_FACTOR_DEFAULT if factor is None else float(factor)
        self._dosy_clevels = self._DOSY_CLEVELS_DEFAULT if clevels is None else int(clevels)
        self._dosy_d_lims = None if d_lims is None else (float(d_lims[0]), float(d_lims[1]))
        self._dosy_title = title
        self._draw_dosy()

    def set_dosy_display(self, factor=None, clevels=None, d_lims=...) -> None:
        """Change how the stored DOSY map is *drawn* -- its contour
        floor, how many levels, and the D axis range -- and redraw it,
        without re-running the inverse Laplace transform.

        The panel counterpart to the scroll gesture, which reaches the
        same floor (see :meth:`_adjust_dosy_levels`). Everything here
        is display-only: the map itself is whatever the last
        :meth:`show_dosy` computed.

        ``d_lims`` : ``(low, high)`` to pin the axis, or ``None`` to go
        back to fitting it to the populated part of the grid. Its
        default is the ``...`` sentinel rather than ``None`` precisely
        because ``None`` is a meaningful value here -- omitting the
        argument leaves the current range alone.

        No-op when no DOSY is stored, so a control can be wired up
        unconditionally.
        """
        if self._dosy_result is None:
            return
        if factor is not None:
            self._dosy_factor = min(max(float(factor), self._DOSY_FACTOR_MIN),
                                     self._DOSY_FACTOR_MAX)
        if clevels is not None:
            self._dosy_clevels = max(1, int(clevels))
        if d_lims is not ...:
            self._dosy_d_lims = None if d_lims is None else (float(d_lims[0]), float(d_lims[1]))
        # The D range is part of what is being set here, so it must not
        # be preserved from the view being replaced -- otherwise
        # setting it would visibly do nothing.
        self._draw_dosy(preserve_view=(d_lims is ...))

    def _draw_dosy(self, preserve_view: bool = False) -> None:
        """(Re)draw the stored DOSY map at its current level floor."""
        if self._dosy_result is None:
            return
        limits = (self.ax.get_xlim(), self.ax.get_ylim()) if (preserve_view and self.ax) else None
        dosy = self._dosy_result
        overlay = self._dosy_overlay
        overlay_labels = self._dosy_overlay_labels
        slices = list(self._dosy_slices)
        self._clear_figure()
        self._dosy_result = dosy      # _clear_figure resets view state
        self._dosy_overlay = overlay
        self._dosy_overlay_labels = overlay_labels
        self._dosy_slices = slices
        self._contour_spectrum = None
        self._current_1d_spectrum = None
        self._compare_axes = None
        self._proj_axes = {"f1": None, "f2": None}
        # A plain 111 subplot, exactly as _redraw_contour does for a 2D
        # -- plot_dosy now builds its own projection strip off the same
        # AxesDivider plot_2d_contour uses, so the panel proportions,
        # strip height and spacing match every other contour in the app
        # instead of being this one view's own gridspec. Explicit user
        # request: "the shape of the contour plot can just follow that
        # used for every other contour".
        self.ax = self.figure.add_subplot(111)
        draw = plotting.plot_dosy if overlay is None else plotting.overlay_dosy
        subject = dosy if overlay is None else overlay
        extra = {}
        if overlay is not None:
            labels = self._dosy_overlay_labels or []
            entries = self._assign_series("dosy_overlay", [
                (str(i), (labels[i] if i < len(labels) else f"Map {i + 1}"), True)
                for i in range(len(overlay))
            ])
            extra = {"labels": [e.label for e in entries],
                     "colours": [e.colour for e in entries]}
        _fig, (_contour_ax, projection_ax) = draw(
            subject, ax=self.ax, fig=self.figure, **extra,
            factor=self._dosy_factor, clevels=self._dosy_clevels,
            # Square panel, same as a 2D contour, unless the user has
            # locked their own aspect ratio on this canvas.
            square_aspect=self._aspect_ratio is None,
            # An explicit range from the panel, or None to let plot_dosy
            # fit the axis to the populated part of the grid.
            d_lims=self._dosy_d_lims,
            # This canvas's own style, not plot_dosy's standalone
            # defaults -- the DOSY is a contour panel like any other
            # here and has to answer to the Output tab's font/tick
            # settings the same way a 2D does. The same kwargs, from
            # the same place, for exactly that reason.
            #
            # _AXIS_LINEWIDTH is deliberately *not* forwarded as the
            # contour line width: it styles axis furniture (spines),
            # and at its 1.75 default the contour rings came out
            # heavy enough to merge into blobs. Spine widths are set
            # from it separately, by _apply_current_axis_style below.
            # Rows extracted from this map, drawn beside it. Only for
            # the single-map view: an overlay of several inversions has
            # no single map for a row to come from, and overlay_dosy
            # takes no slices.
            **({"slices": self._dosy_slices} if (overlay is None and self._dosy_slices) else {}),
            **self._contour_style_kwargs(),
        )
        self._dosy_projection_ax = projection_ax
        # plot_dosy cannot return the panel (it shares a fixed-arity
        # unpack with overlay_dosy), so it tags it -- see
        # plotting.twod._draw_traces_panel. Recorded in _proj_axes
        # under the same "slices" key a 2D contour's own panel uses, so
        # the scroll and cursor handlers that already special-case that
        # panel cover the DOSY's for free rather than learning a second
        # attribute.
        self._proj_axes["slices"] = next(
            (a for a in self.figure.get_axes()
             if a.get_gid() == plotting.twod.SLICE_PANEL_GID), None)
        if self._dosy_title and projection_ax is not None:
            projection_ax.set_title(self._dosy_title, fontsize=self._AXIS_LABELSIZE)
        self._apply_current_axis_style()
        self._apply_view_options()
        self._apply_layout()
        if limits:
            self.note_natural_view()
            self.ax.set_xlim(limits[0])
            self.ax.set_ylim(limits[1])
        # Crosshairs here too: a DOSY map is read off the plot.
        self._crosshair_ready = True
        self.canvas.draw_idle()

    def set_dosy_slices(self, slices: list[dict] | None) -> None:
        """Replace the set of rows drawn beside the DOSY map -- each a
        ``{"D", "label", "color"}`` dict. Redraws straight away if a
        map is currently shown, the same way :meth:`set_contour_slices`
        does for a 2D.

        The D values are kept as given rather than snapped to the
        grid: ``plot_dosy`` resolves each to its nearest row at draw
        time, so a map later re-inverted onto a finer grid still shows
        the row the user asked for rather than one frozen against the
        old grid."""
        self._dosy_slices = list(slices or [])
        if self._dosy_result is not None:
            self._draw_dosy(preserve_view=True)

    def _adjust_dosy_levels(self, event) -> None:
        """Scroll-up raises the DOSY's contour floor (fewer, stronger
        levels), scroll-down lowers it -- the same gesture and the same
        meaning as :meth:`_adjust_contour_levels` on a 2D, preserving
        the current pan/zoom. Follows the same scroll-direction choice."""
        step = self._contour_scroll_step(event)
        self._dosy_factor = min(
            max(self._dosy_factor * step, self._DOSY_FACTOR_MIN), self._DOSY_FACTOR_MAX,
        )
        self._draw_dosy(preserve_view=True)
        # So the panel's own floor control shows what the plot is
        # actually drawn at, rather than the two disagreeing as soon as
        # anyone scrolls.
        self.dosyLevelFloorChanged.emit(self._dosy_factor)

    #: The curve-fit line unless the user sets another (Dynamics ▸ Fit
    #: Curve ▸ Fit line).
    DEFAULT_FIT_LINE = {"color": "red", "linestyle": "--", "linewidth": 2.0}

    def show_relaxation_fits(self, results: list, xlabel: str = "Delay (s)",
                             x_scale: float = 1.0, ylabel: str = "Intensity",
                             fit_curves: list | None = None, fit_style: dict | None = None,
                             normalise: bool = False, excluded_points: list | None = None,
                             ylabel_suffix: str | None = None) -> None:
        """Every peak's series and fit on one plot (explicit user request,
        2026-09-30: see all fits together), each peak in its own
        colour-blind-safe colour -- its points and its fit line, drawn in
        the fit line style. ``fit_curves``: per peak ``(t_fine, y_fine)``
        or None (then ``y_fit`` point to point). ``normalise`` scales each
        peak by its own fitted maximum, so curves of very different
        intensity can be compared by shape (the y label says so).
        ``excluded_points``: per peak ``(t, y)`` of the points removed
        from its fit (right-click ▸ Remove Point), drawn as × in the
        peak's colour, or None. ``ylabel_suffix`` replaces the
        " (normalised)" the y label gets when normalising."""
        from ...plotting.accessibility import ordered_qualitative_colours
        style = {**self.DEFAULT_FIT_LINE, **(fit_style or {})}
        self._enter_plot_kind("curve")
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
        self._preview_1d_spectrum = None
        self._clear_figure()
        self.ax = self.figure.add_subplot(111)
        x_scale = float(x_scale) or 1.0
        colours = ordered_qualitative_colours("okabe_ito")
        shown = []
        for k, result in enumerate(results):
            colour = colours[k % len(colours)]
            t = np.asarray(result["t"], dtype=float) / x_scale
            y = np.asarray(result["y"], dtype=float)
            curve = (fit_curves or [None] * len(results))[k]
            if curve is not None:
                cx, cy = np.asarray(curve[0], dtype=float) / x_scale, np.asarray(curve[1], float)
            else:
                order = np.argsort(t)
                cx, cy = t[order], np.asarray(result["y_fit"], dtype=float)[order]
            scale = 1.0
            if normalise:
                scale = float(np.max(np.abs(cy))) or 1.0
            label = result.get("peak_label") or f"Peak {k + 1}"
            self.ax.plot(t, y / scale, "o", color=colour, markersize=6.0, label=label, zorder=2)
            self.ax.plot(cx, cy / scale, linestyle=style["linestyle"], color=colour,
                         linewidth=style["linewidth"], zorder=3, label="_nolegend_")
            removed = excluded_points[k] if excluded_points and k < len(excluded_points) else None
            if removed is not None and len(removed[0]):
                self.ax.plot(np.asarray(removed[0], dtype=float) / x_scale,
                             np.asarray(removed[1], dtype=float) / scale, "x", color=colour,
                             markersize=8.0, markeredgewidth=2.0, alpha=0.6,
                             label="_nolegend_", zorder=2)
            shown.append({"label": label, "x": t, "y": np.asarray(result["y"], dtype=float),
                          "y_fit": np.asarray(result["y_fit"], dtype=float),
                          "curve_x": cx, "curve_y": cy, "xlabel": xlabel, "ylabel": ylabel,
                          "result": result, "scale": scale})
        self.ax.set_xlabel(xlabel, fontsize=self._AXIS_LABELSIZE)
        suffix = ylabel_suffix if ylabel_suffix is not None else (
            " (normalised)" if normalise else "")
        self.ax.set_ylabel(ylabel + suffix, fontsize=self._AXIS_LABELSIZE)
        plotting.apply_spine_style(self.ax, top=False, right=False, bottom=True, left=True,
                                   linewidth=self._AXIS_LINEWIDTH)
        plotting.apply_tick_style(self.ax, length=self._TICK_LENGTH, width=self._TICK_WIDTH,
                                  minor=self._minor_ticks_visible,
                                  labelsize=self._AXIS_LABELSIZE)
        self._curve_axes = [self.ax]
        self._curve_fit_shown = None
        self._curve_fits_shown = shown          # (after _clear_figure reset it)
        self._legend_visible = True
        self._apply_current_axis_style()
        self._apply_view_options()
        self._crosshair_ready = True
        self._apply_layout()
        self.canvas.draw_idle()

    def show_relaxation_fit(self, result: dict, uncertainty: tuple | None = None,
                             xlabel: str = "Delay (s)", excluded_points: tuple | None = None,
                             x_scale: float = 1.0, ylabel: str = "Intensity",
                             extra_series: list | None = None, fit_curve: tuple | None = None,
                             fit_style: dict | None = None) -> None:
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
        right-click actions themselves.

        ``ylabel`` : what the y axis is measuring. Defaults to the
        intensity every relaxation and diffusion fit plots, and exists
        because the eNMR fit goes through this same method and plots a
        *phase* -- labelling that "Intensity" is simply wrong, and the
        one plotting path is worth keeping.

        ``extra_series`` : further series drawn faintly *behind* the
        main one, each a dict of ``x``, ``y``, ``label`` and optional
        ``marker``/``color``/``alpha``. For a result that is an
        average of several measurements: the inputs stay on the plot
        beside it, so how far apart they were is visible at the same
        time as what they averaged to. Drawn first and at a lower
        zorder, so the main series and its fit always read on top.

        ``x_scale`` : seconds per displayed x unit, so a fit done in
        seconds can be *drawn* in the millisecond (or microsecond) unit
        the Curve Fitting panel is set to. Every x on this axes goes
        through it -- data, fit curve, uncertainty band and excluded
        markers alike -- so nothing can end up plotted on a different
        scale from the rest. **The result dict itself is never
        modified**: this is a display transform, and the stored fit
        stays in seconds. A caller reading an x back off this axes (the
        right-click "Exclude Nearest Point") has to multiply by the
        same factor.

        ``fit_curve`` : ``(t_fine, y_fine)`` (x in seconds), the fitted
        function on a fine grid (``processing.relaxation.fit_curve``) --
        what is drawn as the fit when there is no uncertainty band.
        Without it the fit is ``y_fit`` joined point to point, which is
        not the fitted function (explicit user request, 2026-09-30: the
        fit only looked right with the band on).

        ``fit_style`` : the fit line's ``color``, ``linestyle`` and
        ``linewidth`` (user-set in Dynamics ▸ Fit Curve; default a red
        dashed line, 2 pt).

        A ``"curve"`` plot (see :meth:`_enter_plot_kind`): announced
        before drawing, so a saved default style for curves is in place
        when it is drawn. The data points take the trace colour."""
        style = {**self.DEFAULT_FIT_LINE, **(fit_style or {})}
        self._enter_plot_kind("curve")
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
        self._preview_1d_spectrum = None
        self._clear_figure()
        self.ax = self.figure.add_subplot(111)

        x_scale = float(x_scale) or 1.0
        for extra in (extra_series or []):
            self.ax.plot(
                np.asarray(extra["x"], dtype=float) / x_scale,
                np.asarray(extra["y"], dtype=float),
                extra.get("marker", "o"), color=extra.get("color", "grey"),
                markersize=6.0, alpha=extra.get("alpha", 0.45),
                markerfacecolor="none", label=extra.get("label"), zorder=1.5,
            )
        t = np.asarray(result["t"], dtype=float) / x_scale
        y, y_fit = result["y"], result["y_fit"]
        if uncertainty is not None:
            t_fine, y_fine, y_band = uncertainty
            t_fine = np.asarray(t_fine, dtype=float) / x_scale
            self.ax.fill_between(
                t_fine, y_fine - y_band, y_fine + y_band,
                color="grey", alpha=0.4, label="Uncertainty", zorder=1,
            )
            curve_x, curve_y = t_fine, np.asarray(y_fine, dtype=float)
        elif fit_curve is not None:
            curve_x = np.asarray(fit_curve[0], dtype=float) / x_scale
            curve_y = np.asarray(fit_curve[1], dtype=float)
        else:
            order = np.argsort(t)
            curve_x, curve_y = t[order], np.asarray(y_fit, dtype=float)[order]
        self.ax.plot(curve_x, curve_y, linestyle=style["linestyle"], color=style["color"],
                     linewidth=style["linewidth"], label="Fit", zorder=3)
        self.ax.plot(t, y, "o", color=self._trace_color, markersize=8.0,
                      label=result.get("peak_label", result.get("model", "Data")), zorder=2)
        if excluded_points is not None and len(excluded_points[0]):
            t_excluded, y_excluded = excluded_points
            self.ax.plot(np.asarray(t_excluded, dtype=float) / x_scale, y_excluded,
                          "x", color="grey", markersize=8.0,
                          markeredgewidth=2.0, alpha=0.7, label="Excluded", zorder=2)

        self.ax.set_xlabel(xlabel, fontsize=self._AXIS_LABELSIZE)
        self.ax.set_ylabel(ylabel, fontsize=self._AXIS_LABELSIZE)
        plotting.apply_spine_style(self.ax, top=False, right=False, bottom=True, left=True,
                                    linewidth=self._AXIS_LINEWIDTH)
        plotting.apply_tick_style(self.ax, length=self._TICK_LENGTH, width=self._TICK_WIDTH, minor=self._minor_ticks_visible,
                                   labelsize=self._AXIS_LABELSIZE)
        self._curve_axes = [self.ax]
        # What "Keep This Fit…" keeps: the curve as drawn (display units).
        self._curve_fit_shown = {
            "x": np.asarray(t, dtype=float), "y": np.asarray(y, dtype=float),
            "y_fit": np.asarray(y_fit, dtype=float), "xlabel": xlabel, "ylabel": ylabel,
            "label": result.get("peak_label", result.get("model", "")), "result": result,
            "curve_x": np.asarray(curve_x, dtype=float), "curve_y": np.asarray(curve_y, dtype=float)}
        self._legend_visible = True
        self._apply_current_axis_style()
        self._apply_view_options()
        # Crosshairs here too: reading a value off this plot is
        # exactly what it is for (curve fit).
        self._crosshair_ready = True
        self._apply_layout()
        self.canvas.draw_idle()

    def show_parameter_trend(self, x, y, yerr=None, point_labels=None, xlabel: str = "x",
                              ylabel: str = "", title: str = "") -> None:
        """One fitted parameter along a set of experiments (a multi-
        spectrum fit), with 1-sigma error bars: against ``x`` when every
        experiment has a numeric one, else in fitting order with each
        experiment named on the axis. A ``"curve"`` plot, like
        :meth:`show_relaxation_fit`."""
        self._enter_plot_kind("curve")
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
        self._preview_1d_spectrum = None
        self._clear_figure()
        self.ax = self.figure.add_subplot(111)
        y = np.asarray(y, dtype=float)
        err = None if yerr is None else np.nan_to_num(np.asarray(yerr, dtype=float))
        # ``list(...)``, not ``x or []``: a numpy array has no truth value,
        # so passing one raised (found 2026-11-12).
        x = [] if x is None else list(x)
        numeric = all(isinstance(v, (int, float, np.floating, np.integer)) for v in x)
        if numeric and len(x) == len(y):
            xs = np.asarray(x, dtype=float)
            order = np.argsort(xs)
            self.ax.errorbar(xs[order], y[order], yerr=None if err is None else err[order], fmt="o-",
                             color=self._trace_color, markersize=6.0, capsize=3, linewidth=1.2)
            self.ax.set_xlabel(xlabel, fontsize=self._AXIS_LABELSIZE)
        else:
            xs = np.arange(len(y))
            self.ax.errorbar(xs, y, yerr=err, fmt="o-", color=self._trace_color, markersize=6.0, capsize=3,
                             linewidth=1.2)
            self.ax.set_xticks(xs)
            self.ax.set_xticklabels(list(point_labels or [str(i + 1) for i in xs]),
                                    rotation=30, ha="right")
            self.ax.set_xlabel("Experiment (fitting order)", fontsize=self._AXIS_LABELSIZE)
        self.ax.set_ylabel(ylabel, fontsize=self._AXIS_LABELSIZE)
        if title:
            self.ax.set_title(title, fontsize=self._AXIS_LABELSIZE)
        plotting.apply_spine_style(self.ax, top=False, right=False, bottom=True, left=True,
                                    linewidth=self._AXIS_LINEWIDTH)
        plotting.apply_tick_style(self.ax, length=self._TICK_LENGTH, width=self._TICK_WIDTH,
                                   minor=self._minor_ticks_visible, labelsize=self._AXIS_LABELSIZE)
        self._curve_axes = [self.ax]
        self._legend_visible = False
        self._apply_current_axis_style()
        self._apply_view_options()
        self._crosshair_ready = True
        self._apply_layout()
        self.canvas.draw_idle()

    def show_external_plot(self, draw, kind: str | None = None, hz_per_ppm: float | None = None,
                           _keep_series: bool = False, series_group: str | None = None) -> None:
        """A figure drawn by a notebook plotting function -- ``draw(figure)``
        receives this canvas's own (cleared) figure -- so the GUI shows
        exactly the figure a notebook call produces, rather than a second
        drawing of the same thing that could drift from it. Used by
        Integrate ▸ DNP (``opennmr.notebook.dnp_workflow``'s plots).

        ``kind`` : the plot kind to announce first (``"curve"`` for a
        profile, ``"compare"`` for overlaid spectra), so its saved default
        style is in place. Whatever the kind, the canvas's current style
        -- label size, font, ticks, spine width, minor ticks, legend -- is
        applied to the finished figure, so it matches the rest of the app
        and follows the Plot Style panel. ``draw`` may read
        :meth:`style_settings` for what only it can apply (colours).

        View state is reset as for :meth:`show_parameter_trend`, so
        nothing from a previous spectrum/contour view survives; pick and
        region-select modes work on whatever Axes the click lands in, so
        they work on these plots too.

        ``hz_per_ppm`` : the spectrometer frequency (MHz) behind any ppm
        axis on the figure, so Measure can quote Hz as well as ppm (an
        axis is taken to be ppm, Hz or kHz from its label).

        Every named trace becomes a series (Plot Style ▸ names and
        colours): renamed or recoloured there, and coloured from the
        series colour map once that is changed. A style change that only
        the draw can apply (colours, text size, the map colour scale)
        redraws the figure, keeping the zoom.

        ``series_group``: a name for a family of figures that are redraws
        of the same thing (``"snr"``: every S/N calculation). Names and
        colours set on one survive the next of the same group, as they
        do a restyle (explicit user report, 2026-10-14: "when computing
        signal-to-noise, all plot styles disappear")."""
        same_group = series_group is not None and series_group == getattr(
            self, "_external_series_group", None)
        self._external_series_group = series_group
        # Not an external plot any more until this one is drawn: the
        # plot-kind defaults applied just below call set_style, which
        # would otherwise redraw the *previous* figure.
        self._external_plot = False
        self._external_draw = None
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
        self._preview_1d_spectrum = None
        if kind is not None:
            self._enter_plot_kind(kind)
        self._clear_figure()
        if not _keep_series and not same_group:
            self._series.pop("external", None)
            self._external_palette = False
        draw(self.figure)
        self._external_plot = True
        self._external_draw = {"draw": draw, "kind": kind, "hz_per_ppm": hz_per_ppm,
                               "series_group": series_group}
        self._apply_external_series()
        for ax in self.figure.axes:
            if ax.get_legend() is not None:
                ax.get_legend().set_visible(self._legend_visible)
        self.ax = self.figure.axes[0] if self.figure.axes else self.figure.add_subplot(111)
        if kind == "curve":
            self._curve_axes = list(self.figure.axes)
        self._apply_current_axis_style()
        if self._grid_visible:
            for ax in self.figure.axes:
                if getattr(ax, "name", "") != "3d":  # 3D: the tensor style's own
                    ax.grid(True, alpha=0.3)
        # _legend_visible is left alone: the figure names its own traces
        # (see _external_plot), and switching it off here turned the
        # Compare legend off for good after one look at a DNP plot.
        self._crosshair_ready = True
        self._apply_layout()
        self.canvas.draw_idle()

    def _redraw_external(self) -> None:
        """Redraw the external plot on show with the current style,
        keeping each 2D Axes' limits (the zoom) when the layout is the
        same, and what it offers to keep."""
        info = self._external_draw
        if info is None or getattr(self, "_external_redrawing", False):
            return
        self._external_redrawing = True
        try:
            self._redraw_external_now(info)
        finally:
            self._external_redrawing = False

    def _redraw_external_now(self, info: dict, keep_limits: bool = True) -> None:
        flat = [ax for ax in self.figure.axes if getattr(ax, "name", "") != "3d"]
        limits = [(ax.get_xlim(), ax.get_ylim()) for ax in flat] if keep_limits else []
        # A 3D picture keeps the view it was turned to -- unless the
        # tensor style has just set one.
        views = [(ax.elev, ax.azim) for ax in self.figure.axes if getattr(ax, "name", "") == "3d"]
        if self._tensor_view_set:
            views, self._tensor_view_set = [], False
        keep = self.simulation_spectrum
        owner = self.plot_owner
        self.show_external_plot(info["draw"], kind=info["kind"], hz_per_ppm=info["hz_per_ppm"],
                                _keep_series=True, series_group=info.get("series_group"))
        self.simulation_spectrum = keep
        self.plot_owner = owner
        threes = [ax for ax in self.figure.axes if getattr(ax, "name", "") == "3d"]
        if views and len(threes) == len(views):
            for ax, (elev, azim) in zip(threes, views):
                ax.view_init(elev=elev, azim=azim)
                self._reset_3d_box(ax)          # fitted for this view, not the default one
            self._fit_3d_layout(threes)
            self.canvas.draw_idle()
        flat_now = [ax for ax in self.figure.axes if getattr(ax, "name", "") != "3d"]
        if limits and len(flat_now) == len(limits):
            for ax, (xlim, ylim) in zip(flat_now, limits):
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
            self.canvas.draw_idle()

    @staticmethod
    def _handle_colour(handle):
        """The colour a legend handle is drawn in, as hex, or None."""
        try:
            if isinstance(handle, matplotlib.lines.Line2D):
                return matplotlib.colors.to_hex(handle.get_color())
            if isinstance(handle, matplotlib.container.Container):
                patches = list(getattr(handle, "patches", None) or [])
                if not patches:
                    return None
                handle = patches[0]
            if isinstance(handle, matplotlib.patches.Patch):
                return matplotlib.colors.to_hex(handle.get_facecolor())
            if isinstance(handle, matplotlib.collections.Collection):
                colours = handle.get_facecolor()
                if len(colours) == 0:
                    colours = handle.get_edgecolor()
                return matplotlib.colors.to_hex(colours[0]) if len(colours) else None
        except (ValueError, TypeError, IndexError):
            return None
        return None

    @staticmethod
    def _recolour_handle(ax, handle, colour) -> None:
        """Recolour what ``handle`` stands for. A histogram labels only
        the first bar of each data set, so a patch takes every patch in
        its Axes drawn in the same colour with it."""
        if isinstance(handle, matplotlib.lines.Line2D):
            handle.set_color(colour)
            return
        patches = None
        if isinstance(handle, matplotlib.container.Container):
            patches = list(getattr(handle, "patches", None) or [])
        elif isinstance(handle, matplotlib.patches.Patch):
            old = tuple(handle.get_facecolor())
            patches = [p for p in ax.patches if tuple(p.get_facecolor()) == old] or [handle]
        if patches is not None:
            for patch in patches:
                alpha = patch.get_alpha()
                patch.set_facecolor(colour)
                if alpha is not None:
                    patch.set_alpha(alpha)
            return
        if isinstance(handle, matplotlib.collections.Collection):
            alpha = handle.get_alpha()
            handle.set_facecolor(colour)
            if len(handle.get_edgecolor()):
                handle.set_edgecolor(colour)
            if alpha is not None:
                handle.set_alpha(alpha)

    def _apply_external_series(self) -> None:
        """Make every named trace of an external figure a series, and
        apply any name or colour set for it (see show_external_plot)."""
        per_axes = []
        order: list[str] = []
        first_colour: dict = {}
        for ax in self.figure.get_axes():
            if getattr(ax, "name", "") == "3d":
                continue
            handles, labels = ax.get_legend_handles_labels()
            per_axes.append((ax, list(zip(handles, labels)), ax.get_legend() is not None))
            for handle, label in zip(handles, labels):
                if label not in first_colour:
                    order.append(label)
                    first_colour[label] = self._handle_colour(handle)
        if not order:
            self._series.pop("external", None)
            return
        seen_before = {entry.key for entry in self._series.get("external", [])}
        entries = self._assign_series("external", [(label, label, True) for label in order])
        # A trace new to this figure takes the colour the Compare view
        # gives the same spectrum, when one was chosen there -- an S/N or
        # DNP plot of spectra styled in Compare keeps their colours.
        compare_colours = {entry.label: entry.colour for entry in self._series.get("compare", [])
                           if entry.colour_overridden}
        for entry in entries:
            if entry.key not in seen_before and entry.key in compare_colours:
                entry.colour = compare_colours[entry.key]
                entry.colour_overridden = True
        by_key = {entry.key: entry for entry in entries}
        for entry in entries:
            if not entry.colour_overridden and not self._external_palette and first_colour.get(entry.key):
                entry.colour = first_colour[entry.key]
        for ax, pairs, had_legend in per_axes:
            for handle, label in pairs:
                entry = by_key.get(label)
                if entry is None:
                    continue
                if entry.colour_overridden or self._external_palette:
                    self._recolour_handle(ax, handle, entry.colour)
                if entry.label != label:
                    handle.set_label(entry.label)
            if had_legend and pairs:
                ax.legend([h for h, _l in pairs],
                          [by_key[l].label if l in by_key else l for _h, l in pairs],
                          **self._legend_kwargs())

    def _external_axis_unit(self, axis) -> str | None:
        """``"ppm"``, ``"hz"`` or ``"khz"`` when an external figure's
        axis label says that is what it is in, else None."""
        text = axis.label.get_text() if axis is not None else ""
        if re.search(r"ppm", text):
            return "ppm"
        if re.search(r"\bkHz\b", text):
            return "khz"
        if re.search(r"\bHz\b", text):
            return "hz"
        return None

    def _external_delta(self, axis, delta: float) -> dict:
        """A measured separation on an external figure, in every unit
        its axis label and the plot's frequency allow."""
        unit = self._external_axis_unit(axis)
        if unit is None:
            return {}
        mhz = (self._external_draw or {}).get("hz_per_ppm")
        delta = abs(float(delta))
        hz = {"ppm": delta * mhz if mhz else None, "hz": delta, "khz": delta * 1000.0}[unit]
        out = {}
        if hz is not None:
            out["hz"] = hz
            out["khz"] = hz / 1000.0
        if unit == "ppm":
            out["ppm"] = delta
        elif mhz:
            out["ppm"] = hz / mhz
        return out

    def update_data(self, spectrum: NMRSpectrum) -> None:
        """Refresh the currently-displayed 1D spectrum's y-data in
        place, preserving the current view (zoom/pan) -- unlike
        :func:`show_spectrum`, which resets the view for a newly
        selected spectrum. For a live preview (e.g. dragging a phase
        control) where the spectrum's identity hasn't changed, only
        its values have.

        Re-derives the same rendering-only downsampling
        :func:`opennmr.plotting.plot_1d` applied to the y-data already
        on screen (see :func:`opennmr.plotting.style.downsample_1d`'s
        own docstring: the bin structure depends only on length, so
        this reproduces bins aligned to the x already plotted even
        though it's never given that x at all) -- otherwise a high-
        point-count spectrum's full-length new y-data wouldn't match
        the already-downsampled x-data's length at all, let alone
        quickly.

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
        _, new_ydata = plotting.downsample_1d(None, y)
        if len(new_ydata) != len(self.ax.lines[0].get_ydata()):
            # Both limits, not just x. The in-place branch below leaves
            # the y view exactly as it was (it only swaps ydata), so
            # restoring x alone here made the same edit reset the
            # intensity zoom or not depending purely on whether the
            # point count happened to change -- a zero-fill threw it
            # away, a window function did not. Reported as the view
            # still being reset "when the fid is being recomputed after
            # windowing etc.".
            xlim, ylim = self.ax.get_xlim(), self.ax.get_ylim()
            self.show_spectrum(spectrum)
            self.ax.set_xlim(xlim)
            self.ax.set_ylim(ylim)
            return
        self.ax.lines[0].set_ydata(new_ydata)
        self.canvas.draw_idle()

    @classmethod
    def _data_y_range(cls, ax, x_window=None):
        """``(lowest, highest)`` of every data line on ``ax``, read from
        each line's **full-resolution** arrays where it has them.

        The display downsampling keeps the largest-magnitude point of
        each bin, so a bin holding a tall positive peak drops the small
        negative dip beside it: an autoscale of the downsampled line
        then sat its bottom above the true minimum, and switching to
        Full Resolution (which swaps the data in place) drew that part
        under the axis. None if nothing finite is plotted."""
        lows, highs = [], []
        for line in ax.get_lines():
            if not line.get_visible() or line.get_gid() in (
                    cls._BASELINE_PREVIEW_GID, cls._BASELINE_GHOST_GID):
                continue
            full = getattr(line, "_opennmr_full_res", None)
            y = np.asarray(full[1] if full is not None else line.get_ydata(), dtype=float)
            if y.ndim != 1 or y.size < 2:
                continue
            if x_window is not None:
                # Only what is inside the typed x range: a solvent peak outside
                # the window must not set the height of the peaks within it.
                x = np.asarray(full[0] if full is not None else line.get_xdata(), dtype=float)
                if x.shape == y.shape:
                    y = y[(x >= x_window[0]) & (x <= x_window[1])]
            y = y[np.isfinite(y)]
            if y.size:
                lows.append(float(np.min(y)))
                highs.append(float(np.max(y)))
        if not lows:
            return None
        return min(lows), max(highs)

    def _fit_y_to_data(self, ax, bottom_gap: float | None = None, y_range=None,
                       top_gap: float | None = None) -> None:
        """Set ``ax``'s y-limits so the lowest point of what is plotted
        sits ``bottom_gap`` (a fraction of the traces' height; default
        Plot Style's "Gap below spectra") above the axis line, and the
        highest ``_TOP_GAP`` below the top. Never leaves part of a
        trace under the axis, whatever sign its baseline has."""
        if y_range is None:
            y_range = self._data_y_range(ax)
        if y_range is None:
            return
        low, high = y_range
        span = high - low
        if not np.isfinite(span) or span <= 0:
            span = abs(high) or 1.0
        gap = self._baseline_gap if bottom_gap is None else bottom_gap
        top = self._TOP_GAP if top_gap is None else top_gap
        ax.set_ylim(low - gap * span, high + top * span)
        ax._opennmr_fitted_ylim = ax.get_ylim()

    def _refit_spectrum_y_limits(self, only_if_untouched: bool = False) -> None:
        """Re-apply the gaps to whatever 1D spectrum / Compare is on
        show (a gap changed, or Full Resolution swapped the data).
        ``only_if_untouched``: leave an axes the user has zoomed alone.
        Otherwise a zoomed axes keeps its zoom -- the same crop of its row,
        carried over to the new limits -- instead of snapping back."""
        if self._plot_kind not in ("1d", "compare", "pseudo2d") or self._contour_spectrum is not None:
            return
        if self._compare_axes is not None:
            axes = list(self._compare_axes)
        elif self._current_1d_spectrum is not None or self._compare_kwargs_active():
            axes = [self.ax]
        else:
            return
        shared = None
        if self._compare_axes is not None and getattr(self, "_compare_shared_range", None) is not None:
            shared = self._compare_shared_range
        every = list(axes)
        zoomed = {}
        for ax in axes:
            fitted = getattr(ax, "_opennmr_fitted_ylim", None)
            if fitted is not None and tuple(ax.get_ylim()) != tuple(fitted):
                zoomed[ax] = (tuple(fitted), tuple(ax.get_ylim()))
        if only_if_untouched:
            axes = [ax for ax in axes if ax not in zoomed]
        if self._compare_axes is not None:
            self._fit_compare_rows(every, shared, bottom_row=self._compare_axes[-1], apply_to=axes)
        else:
            for ax in axes:
                self._fit_y_to_data(ax)
        if not only_if_untouched:
            for ax, (old_fit, view) in zoomed.items():
                new_fit = getattr(ax, "_opennmr_fitted_ylim", None)
                old_span = old_fit[1] - old_fit[0]
                if new_fit is None or not old_span:
                    continue
                lo = new_fit[0] + (view[0] - old_fit[0]) / old_span * (new_fit[1] - new_fit[0])
                hi = new_fit[0] + (view[1] - old_fit[0]) / old_span * (new_fit[1] - new_fit[0])
                ax.set_ylim(lo, hi)
        self.canvas.draw_idle()

    def _fit_compare_rows(self, axes, shared_range=None, bottom_row=None, apply_to=None) -> None:
        """Limits of the stacked rows, so that the spectra sit one above
        another at an **even pitch**: each baseline the same distance above
        the one below (``1 + stack gap + top margin`` spectrum heights),
        the bottom one ``baseline_gap`` above the axis. Every row is the
        same slice of the figure; each row's limits place its own spectrum
        where it belongs in the whole stack, so a gap changed -- including
        to a negative one, which overlaps the spectra -- moves every
        spectrum equally. A spectrum is one unit tall in its own scale
        (its own range, or the shared one), so rows of different heights
        keep their relative sizes on a shared scale. ``apply_to``: only
        set these rows' limits (the others are still part of the stack)."""
        if not axes:
            return
        n = len(axes)
        bg = self._baseline_gap
        top = self._TOP_GAP
        pitch = 1.0 + max(self._STACK_GAP_MIN, self._stack_gap) + top
        total = bg + (n - 1) * pitch + 1.0 + top
        window = total / n
        for index, ax in enumerate(axes):
            if apply_to is not None and ax not in apply_to:
                continue
            y_range = shared_range if shared_range is not None else self._data_y_range(ax)
            if y_range is None:
                continue
            low, high = y_range
            span = high - low
            if not np.isfinite(span) or span <= 0:
                span = abs(high) or 1.0
            k = n - 1 - index                  # 0 for the bottom row
            lo = low + (k * window - bg - k * pitch) * span
            ax.set_ylim(lo, lo + window * span)
            ax._opennmr_fitted_ylim = ax.get_ylim()

    def _compare_kwargs_active(self) -> bool:
        """An overlay Compare (one axes) is showing."""
        return (self._compare_axes is None and self._current_1d_spectrum is None
                and bool(getattr(self, "_compare_spectra", None))
                and self._plot_kind in ("compare", "pseudo2d"))

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
        line = self.ax.lines[0]
        full = getattr(line, "_opennmr_full_res", None)
        y = np.asarray(full[1] if full is not None else line.get_ydata(), dtype=float)
        if len(y) == 0:
            return
        self._fit_y_to_data(self.ax, y_range=(float(np.nanmin(y)), float(np.nanmax(y))))
        self.canvas.draw_idle()

    def _intensity_view_axes(self):
        """The axes whose y is the intensity of one 1D trace: the
        Processing preview's spectrum panel when that is up, else a
        plain 1D spectrum. ``None`` for anything else (a contour's y is
        a shift, a Compare has its own stacking)."""
        spec_ax = self._processing_preview_spec_ax
        if spec_ax is not None and spec_ax.lines:
            return spec_ax
        if (self._current_1d_spectrum is not None and self._contour_spectrum is None
                and self._compare_axes is None and self.ax is not None and self.ax.lines):
            return self.ax
        return None

    @staticmethod
    def _intensity_view_stats(ax):
        """``low``/``high`` of the trace (``ax.lines[0]``, at full
        resolution) inside the visible x range, its baseline ``level``
        (the median -- most points of a spectrum are baseline) and its
        ``reach``, the furthest any point gets from that level."""
        line = ax.lines[0]
        full = getattr(line, "_opennmr_full_res", None)
        x = np.asarray(full[0] if full is not None else line.get_xdata(), dtype=float)
        y = np.asarray(full[1] if full is not None else line.get_ydata(), dtype=float)
        if x.shape != y.shape or y.ndim != 1 or y.size < 2:
            return None
        lo, hi = sorted(ax.get_xlim())
        keep = (x >= lo) & (x <= hi) & np.isfinite(y)
        if np.count_nonzero(keep) < 2:
            keep = np.isfinite(y)
            if np.count_nonzero(keep) < 2:
                return None
        y = y[keep]
        level = float(np.median(y))
        return {"low": float(np.min(y)), "high": float(np.max(y)), "level": level,
                "reach": float(np.max(np.abs(y - level)))}

    def capture_intensity_view(self):
        """What :meth:`follow_intensity_view` needs to carry the y view
        across a change of the trace's *values* (a baseline correction):
        the view and the trace as they are now. ``None`` when there is
        no 1D intensity axes on show."""
        ax = self._intensity_view_axes()
        if ax is None:
            return None
        stats = self._intensity_view_stats(ax)
        if stats is None:
            return None
        v0, v1 = sorted(ax.get_ylim())
        stats["view"] = (float(v0), float(v1))
        # Everything visible was on screen: the view showed the whole
        # trace, rather than a deliberate intensity zoom.
        stats["whole"] = bool(v0 <= stats["low"] and v1 >= stats["high"])
        return stats

    def follow_intensity_view(self, captured) -> bool:
        """Re-place the y view after the trace's values changed under it.

        A baseline correction moves the whole trace up or down (and,
        since every trace is drawn scaled to its own largest point, can
        rescale it too), while a kept y view stays where it was -- so
        the baseline landed halfway up the plot with the peaks cut off.

        * The whole trace was on show: refit, so the whole corrected
          trace is on show, with the usual gaps.
        * The user had zoomed into the intensity: keep that zoom on the
          same part of the trace -- the baseline stays at the same
          height on screen and the tallest visible feature the same size
          -- rather than on the same raw numbers.

        ``captured`` comes from :meth:`capture_intensity_view` before the
        change. Computed from it alone, so calling this twice for the
        same data gives the same view. Returns whether anything moved."""
        if not captured:
            return False
        ax = self._intensity_view_axes()
        if ax is None:
            return False
        stats = self._intensity_view_stats(ax)
        if stats is None:
            return False
        if captured.get("whole", True):
            self._fit_y_to_data(ax, y_range=(stats["low"], stats["high"]))
        else:
            old_reach, new_reach = captured["reach"], stats["reach"]
            scale = new_reach / old_reach if old_reach > 0 and new_reach > 0 else 1.0
            if not np.isfinite(scale):
                scale = 1.0
            v0, v1 = captured["view"]
            ax.set_ylim(stats["level"] + (v0 - captured["level"]) * scale,
                        stats["level"] + (v1 - captured["level"]) * scale)
        self.canvas.draw_idle()
        return True

    @staticmethod
    def _clip_sideways_only(ax, line) -> None:
        """Clip ``line`` to ``ax``'s own x-range but not its height."""
        from matplotlib.transforms import Bbox, TransformedBbox
        line.set_clip_box(TransformedBbox(Bbox([[0.0, -50.0], [1.0, 50.0]]), ax.transAxes))
        line.set_clip_path(None)
        line.set_clip_on(True)

    def _lines_for_export(self) -> list:
        """Every currently-plotted Line2D across whichever axes this
        canvas is actually showing -- ``self.ax`` alone for every
        ordinary view, or all of :attr:`_compare_axes` for a "Stacked"
        Compare view (several real axes, not one shared by every
        line)."""
        axes = self._compare_axes if self._compare_axes is not None else (
            list(self._spectrum_panels) if self._spectrum_panels else [self.ax])
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
        downsampled data from that same full-resolution array via
        :func:`opennmr.plotting.style.downsample_1d`.

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
                line.set_data(*plotting.downsample_1d(x_full, y_full))
        if self._contour_spectrum is not None:
            self.show_spectrum(self._contour_spectrum)
        else:
            self._refit_spectrum_y_limits(only_if_untouched=True)
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
        # The crosshair is a view aid, never part of a figure: hide its
        # lines for the save, whatever they are drawn as (explicit user
        # report: an export made after hovering the plot contained it).
        crosshair_lines = []
        if self._crosshair is not None:
            crosshair_lines = [ln for ln in (*self._crosshair._vlines, *self._crosshair._hlines)
                               if ln.get_visible()]
            for ln in crosshair_lines:
                ln.set_visible(False)
        # Likewise the tint on a selected stacked-Compare row: it marks
        # which row the scroll-zoom targets, and is no part of a figure.
        selected_row = self._selected_compare_index if self._compare_axes else None
        if selected_row is not None:
            self._selected_compare_index = None
            self._apply_compare_selection_highlight()
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
        # While the figure is at its (possibly huge) export size, nothing
        # may repaint the live widget: a file dialog inside ``save_fn``
        # runs its own event loop, and each paint of an 8-inch figure at
        # 800 dpi is a 6400 x 6400 render. Re-enabled in ``finally``.
        self.canvas.setUpdatesEnabled(False)
        if self._export_dpi is not None:
            self.figure.set_dpi(self._export_dpi)
        default_width, default_height = self.default_export_size()
        self.figure.set_size_inches(
            self._export_fig_width if self._export_fig_width is not None else default_width,
            self._export_fig_height if self._export_fig_height is not None else default_height,
        )
        # **Re-lay-out at the size actually being written.** Subplot
        # margins are stored as *fractions* of the figure, and were
        # computed by tight_layout against whatever size Qt gave the
        # live widget -- but the text keeps its absolute point size. A
        # figure laid out while the widget was large has a small
        # fractional bottom margin, which at a smaller export size is
        # no longer enough room for the same x label, so it is written
        # off the bottom edge. Reported as "saved png/pdf cuts off
        # bottom of x-axis label below like 10x10 in".
        self._restore_configured_text_sizes()
        self._apply_layout()

        # **The save itself is told the DPI.** ``figure.set_dpi`` above is
        # not enough: matplotlib's ``savefig`` defaults to
        # ``savefig.dpi = "figure"``, and for a figure living in a Qt
        # canvas "figure" means ``figure._original_dpi`` -- the dpi the
        # figure was *created* with (screen resolution), which
        # ``set_dpi`` never touches. Every export therefore came out at
        # screen resolution whatever the Output tab said ("nominally 800
        # dpi, but it really does not look it"; Windows and macOS alike).
        # Saying the dpi explicitly, through rcParams (the toolbar's Save
        # calls ``figure.savefig(fname)`` with no arguments), is the one
        # route every save path shares.
        export_rc = {}
        if self._export_dpi is not None:
            export_rc["savefig.dpi"] = float(self._export_dpi)
        # And nothing may be left touching or crossing the edge: if the
        # layout pass could not make room for everything drawn (a legend
        # or annotation outside the axes, an inset, a colour-bar label),
        # crop to the content with a small margin rather than clip it.
        if plotting.edge_crowding(self.figure):
            export_rc.update({"savefig.bbox": "tight", "savefig.pad_inches": self._EXPORT_PAD_INCHES})

        try:
            with matplotlib.rc_context(export_rc):
                save_fn()
        finally:
            self.canvas.setUpdatesEnabled(True)
            if selected_row is not None:
                self._selected_compare_index = selected_row
                self._apply_compare_selection_highlight()
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
            # And back to a layout that suits the on-screen size, for
            # the same reason -- otherwise the export's own margins
            # stay behind on the live view.
            self._restore_configured_text_sizes()
            self._apply_layout()
            self.canvas.draw_idle()

    #: Margin kept around the content when an export has to be cropped
    #: to it (see :meth:`export_full_resolution`), in inches.
    _EXPORT_PAD_INCHES = 0.08

    #: How each reportable unit is formatted. Which of them a given
    #: measurement actually shows is :meth:`_measurement_unit_order`'s
    #: decision, not a fixed list -- see there.
    #: Explicit requirement from the plan: a separation is quoted in
    #: **both** ppm and Hz at once, because a position is ppm and a
    #: coupling is Hz, and switching the display unit to read one
    #: should not cost you the other.
    _MEASUREMENT_UNIT_FORMATS = {
        "mm": "{:.4f} mm", "ppm": "{:.4f} ppm", "hz": "{:.2f} Hz",
    }

    def _measurement_unit_order(self) -> tuple:
        """Which units a separation is quoted in, in order -- **a pair,
        always, and which pair depends on the axis**.

        ``ppm / Hz`` normally: a position is ppm and a coupling is Hz,
        so switching the display unit to read one should not cost you
        the other.

        ``mm / Hz`` on an imaging axis. The distance is the number
        wanted there, and ppm alongside it is the one that stops
        meaning anything -- a chemical shift is not what an mm axis
        measures. Explicit user request: "If the axis is mm, then show
        mm / Hz". It used to append mm to the front of the pair and
        report all three."""
        leading = "mm" if self._axis_unit == "mm" else "ppm"
        return tuple((unit, self._MEASUREMENT_UNIT_FORMATS[unit]) for unit in (leading, "hz"))

    #: Colour of every measurement annotation, and the gid marking the
    #: first-click marker so it can be cleared once the pair completes.
    _MEASUREMENT_COLOR = "tab:red"     # the default; the live one is _measurement_color
    _MEASUREMENT_PENDING_GID = "measurement_pending"

    #: Imaginary-part colour for the FID's "Real + Imaginary" display
    #: mode (:meth:`show_fid`, :meth:`show_processing_preview`) --
    #: real stays the plain default line colour (black); distinct from
    #: "tab:red"/"green" (already TDeff-cutoff/Echo-top markers on the
    #: same axes) and "gray" (the window-envelope overlay).
    _FID_IMAG_COLOR = "tab:blue"

    def _measurement_axes_for(self, ax):
        """``(x_axis, y_axis)`` -- the :class:`NMRAxis` objects the data
        coordinates of ``ax`` correspond to, or ``None`` for either
        where there is no spectral axis behind it (a 1D's intensity
        axis, a pseudo-2D's slice-index axis, any view that isn't a
        spectrum). Only these get converted to real units; anything
        else is reported as a plain difference in whatever is plotted.
        """
        # **A Compare row is a spectrum like any other.** Its axis was
        # not consulted here, so a measurement taken on the Compare tab
        # fell through to the "no spectral axis" branch and came back
        # as a bare difference in whatever the row happened to be
        # plotted in -- no ppm, no Hz. Reported directly: "when
        # measuring in compare, the value should show ppm / Hz like in
        # the spectrum view". Checked first, because a Compare stack
        # can be shown while _contour_spectrum/_current_1d_spectrum
        # still hold something from a previous view.
        if self._panel_axes and self._panel_spectra and ax in self._panel_axes:
            # A Compare 2D panel (or overlay, drawn on the first spectrum's axes).
            panel = self._panel_spectra[min(self._panel_axes.index(ax), len(self._panel_spectra) - 1)]
            return panel.axes[1], (None if panel.is_pseudo_2d else panel.axes[0])
        if self._compare_axes and self._compare_spectra and ax in self._compare_axes:
            position = self._compare_axes.index(ax)
            if position < len(self._compare_spectra):
                return self._compare_spectra[position].axes[0], None
        # **Compare ▸ Superimposed**: every trace on the one Axes, drawn
        # in the display unit like the stack. It fell through to "no
        # spectral axis" and came back as a bare Δx with no unit (user
        # report, 2026-11-12). The first trace's axis gives Hz per ppm.
        if (self._compare_spectra and not self._compare_axes and not self._spectrum_panels
                and ax is self.ax and self._contour_spectrum is None
                and self._current_1d_spectrum is None and not self._external_plot
                and getattr(self._compare_spectra[0], "ndim", 1) == 1):
            return self._compare_spectra[0].axes[0], None
        spectrum = self._contour_spectrum
        if spectrum is not None:
            if ax is self._proj_axes.get("f2"):
                return spectrum.axes[1], None   # top strip: x is F2, y is intensity
            if ax is self._proj_axes.get("f1"):
                return None, spectrum.axes[0]   # left strip: x is intensity, y is F1
            y_axis = None if spectrum.is_pseudo_2d else spectrum.axes[0]
            return spectrum.axes[1], y_axis
        if self._current_1d_spectrum is not None:
            return self._current_1d_spectrum.axes[0], None
        return None, None

    def _ppm_drawn_view(self, ax):
        """``(x_axis, y_axis)`` for a view drawn in **ppm whatever the
        display unit** -- Compare's 2D panels and 2D overlay, Compare ▸
        Panels, the heatmap, a DOSY map, the processing preview's
        spectrum -- or None for any other Axes. ``None`` in either place:
        no chemical-shift axis that way (a slice index, D, intensity).

        Each was its own gap in Measure: a bare number, or nothing in Hz
        (explicit user request, 2026-11-12: "go through all possible
        plots and ensure that the measure always works and gives a
        distance in ppm and Hz ... for both x and y in 2-d
        experiments")."""
        if self._panel_axes and self._panel_spectra and ax in self._panel_axes:
            panel = self._panel_spectra[min(self._panel_axes.index(ax), len(self._panel_spectra) - 1)]
            return panel.axes[1], (None if panel.is_pseudo_2d else panel.axes[0])
        if self._spectrum_panels and ax in self._spectrum_panels:
            if getattr(ax, "_opennmr_panel_kind", "1d") == "pulse":
                return None
            index = self._spectrum_panels.index(ax)
            groups = self._spectrum_panel_groups or []
            members = [self._compare_spectra[i] for i in (groups[index] if index < len(groups) else [])
                       if i < len(self._compare_spectra)]
            if not members:
                return None
            first = members[0]
            if first.ndim == 2:
                return first.axes[1], (None if first.is_pseudo_2d else first.axes[0])
            return first.axes[0], None
        if getattr(self, "_heatmap_spectra", None) and ax is self.ax:
            return self._heatmap_spectra[0].axes[0], None
        if self._dosy_result is not None and ax in (self.ax, self._dosy_projection_ax):
            dosy = self._dosy_result
            ppm = np.asarray(dosy.get("ppm", []), dtype=float)
            hz_per_ppm = dosy.get("hz_per_ppm")
            axis = NMRAxis(kind="frequency", ppm=ppm,
                           hz=ppm * float(hz_per_ppm) if hz_per_ppm else None)
            return axis, None
        spec_ax = getattr(self, "_processing_preview_spec_ax", None)
        preview = getattr(self, "_preview_1d_spectrum", None)
        if spec_ax is not None and ax is spec_ax and preview is not None:
            return preview.axes[0], None
        return None

    #: Units an axis label can name, as (pattern, key). Frequency units
    #: become ppm/Hz pairs; anything else is quoted as written.
    _LABEL_UNIT_PATTERNS = (
        (r"\bppm\b", "ppm"), (r"\bMHz\b", "mhz"), (r"\bkHz\b", "khz"), (r"\bHz\b", "hz"),
    )
    _INTENSITY_LABEL = re.compile(r"intensity|a\.\s*u\.|arb|signal|normali[sz]ed|amplitude \(a",
                                  re.IGNORECASE)

    @staticmethod
    def _plain_label(text: str) -> str:
        """An axis label without its mathtext markup: ``$\\mathit{t}$ / µs``
        -> ``t / µs``."""
        text = re.sub(r"\\math(?:it|rm|bf|sf|tt|cal)\{([^{}]*)\}", r"\1", text or "")
        text = re.sub(r"\\(?:mu)\b", "µ", text)
        text = re.sub(r"\\(?:circ)\b", "°", text)
        text = text.replace("^{2}", "²").replace("^2", "²").replace("^{-1}", "⁻¹")
        # mathtext spacing (\, \; \: \!), then any other command.
        text = re.sub(r"\\[,;:]", " ", text).replace("\\!", "")
        text = re.sub(r"\\[A-Za-z]+", "", text)
        text = re.sub(r"\s{2,}", " ", text)
        return text.replace("$", "").replace("{", "").replace("}", "").strip()

    @classmethod
    def _label_unit(cls, text: str) -> str:
        """The unit an axis label states: after its last " / ", else in
        its last brackets ("Delay (s)", "Offset / kHz"); "" if none."""
        text = cls._plain_label(text)
        if " / " in text:
            unit = text.rsplit(" / ", 1)[1].strip()
        else:
            match = re.search(r"[(\[]([^()\[\]]+)[)\]]\s*$", text)
            unit = match.group(1).strip() if match else ""
        return "" if unit.lower() in ("a.u.", "a. u.", "arb. units", "arb.") else unit

    def _axis_label_text(self, ax, which: str) -> str:
        """``ax``'s x or y label -- or, for a row of a stack with its x
        label only on the bottom row, the label of an Axes sharing its x."""
        axis = ax.xaxis if which == "x" else ax.yaxis
        text = axis.label.get_text()
        if text or which != "x":
            return text
        shared = ax.get_shared_x_axes()
        for other in self.figure.axes:
            if other is not ax and shared.joined(ax, other) and other.xaxis.label.get_text():
                return other.xaxis.label.get_text()
        return ""

    def _known_hz_per_ppm(self) -> float | None:
        """The spectrometer frequency behind whatever is on show, for a
        ppm axis that carries no Hz of its own."""
        external = (getattr(self, "_external_draw", None) or {}).get("hz_per_ppm")
        if external:
            return float(external)
        if getattr(self, "_slice_fit_active", False):
            mhz = self._slice_fit_hz_per_ppm()
            if mhz:
                return mhz
        if self._dosy_result is not None and self._dosy_result.get("hz_per_ppm"):
            return float(self._dosy_result["hz_per_ppm"])
        for spectrum in (self._current_1d_spectrum, self._contour_spectrum,
                         getattr(self, "_preview_1d_spectrum", None),
                         *(self._compare_spectra or []), *(self._panel_spectra or []),
                         *(getattr(self, "_heatmap_spectra", None) or [])):
            if spectrum is not None and getattr(spectrum.metadata, "sfo1", None):
                return float(spectrum.metadata.sfo1)
        return None

    def _label_delta(self, ax, which: str, a: float, b: float) -> dict:
        """A separation along an axis with no spectrum behind it, in the
        unit its label states: ppm/Hz (both, when the field is known) for
        a frequency axis, otherwise the label's own unit ("12.5 µs",
        "0.32 s"). A logarithmic axis also gets the ratio. ``{}`` when the
        label names no unit, or is an intensity (see _complete_measurement)."""
        text = self._axis_label_text(ax, which)
        if not text or (which == "y" and self._INTENSITY_LABEL.search(self._plain_label(text))):
            return {}
        delta = abs(float(b) - float(a))
        unit = self._label_unit(text)
        plain = self._plain_label(text)
        out: dict = {}
        frequency = next((key for pattern, key in self._LABEL_UNIT_PATTERNS
                          if re.search(pattern, unit or plain)), None)
        if frequency is not None:
            mhz = self._known_hz_per_ppm()
            if frequency == "ppm":
                out["ppm"] = delta
                if mhz:
                    out["hz"] = delta * mhz
            else:
                out["hz"] = delta * {"hz": 1.0, "khz": 1e3, "mhz": 1e6}[frequency]
                if mhz:
                    out["ppm"] = out["hz"] / mhz
            if "hz" in out:
                out["khz"] = out["hz"] / 1000.0
        elif unit:
            out["label_unit"] = unit
            out["value"] = delta
        else:
            return {}
        scale = ax.get_xscale() if which == "x" else ax.get_yscale()
        if scale == "log" and min(a, b) > 0:
            out["ratio"] = max(a, b) / min(a, b)
        return out

    def _delta_in_units(self, axis, delta: float) -> dict:
        """A separation ``delta``, given in whatever unit the axis is
        currently *displayed* in, expressed in every unit it can be.

        The display unit is the trap this exists to avoid: with the
        view in ppm, ``x2 - x1`` is a number of ppm, and with it in Hz
        the same gesture yields a number of Hz -- so the raw difference
        cannot be labelled with a fixed unit. Everything is taken to Hz
        first and derived from there, via the axis's own stored ppm and
        Hz arrays (rather than SFO1, which is the transmitter frequency
        and not the same thing once referencing has moved the axis), so
        there is exactly one conversion to get wrong.
        """
        out: dict = {}
        if axis is None or getattr(axis, "kind", None) != "frequency":
            return out
        hz_per_ppm = self._axis_hz_per_ppm(axis)
        if hz_per_ppm is None:
            # No Hz behind this axis (a simulation with no field): the
            # separation is still worth a unit in what is displayed.
            if self._axis_unit == "ppm":
                out["ppm"] = abs(float(delta))
            elif self._axis_unit in ("hz", "khz"):
                out["hz"] = abs(float(delta)) * (1000.0 if self._axis_unit == "khz" else 1.0)
                out["khz"] = out["hz"] / 1000.0
            return out
        hz_per_mm = self._imaging_hz_per_mm()
        hz_per_display = {
            "ppm": hz_per_ppm, "hz": 1.0, "khz": 1000.0, "mm": hz_per_mm,
        }.get(self._axis_unit)
        if not hz_per_display:
            return out
        delta_hz = abs(float(delta)) * hz_per_display
        out["hz"] = delta_hz
        out["khz"] = delta_hz / 1000.0
        out["ppm"] = delta_hz / hz_per_ppm
        # **Only when the view is actually in mm.** Having imaging
        # gradient parameters set is not enough: they are a property of
        # the acquisition, so once entered they stay set for every
        # spectrum in the session, and a chemical-shift measurement on
        # a ppm axis was being quoted with a distance beside it -- a
        # number with no meaning on that axis, inviting a position to
        # be read off an axis that is not one. Explicit user request:
        # "don't give a mm value unless the axis is set to mm".
        # ppm/Hz are still reported alongside it in the mm case: those
        # remain true of an imaging axis, and dropping them would cost
        # the coupling reading the pairing exists to preserve.
        if self._axis_unit == "mm":
            out["mm"] = delta_hz / hz_per_mm
        return out

    def _slice_fit_hz_per_ppm(self) -> float | None:
        """MHz (Hz per ppm) behind the fit on show: the fitted spectrum's
        own axis, else its SFO1, else the ``hz_per_ppm`` the caller gave."""
        args = getattr(self, "_slice_fit_args", None) or {}
        given = args.get("hz_per_ppm")
        spectrum = args.get("spectrum")
        if spectrum is not None:
            try:
                axis = spectrum.axes[-1]
                if axis.ppm is not None and axis.hz is not None and len(axis.ppm) > 1:
                    span_ppm = float(axis.ppm[-1] - axis.ppm[0])
                    span_hz = float(axis.hz[-1] - axis.hz[0])
                    if span_ppm and span_hz:
                        return abs(span_hz / span_ppm)
            except (AttributeError, IndexError, TypeError):
                pass
            if spectrum.metadata.sfo1:
                return float(spectrum.metadata.sfo1)
        return float(given) if given else None

    def _slice_fit_delta(self, ax, dx: float) -> dict | None:
        """A separation on a show_slice_fit plot in ppm and (when the field
        is known) Hz; None when ``ax`` is not such a plot."""
        if not getattr(self, "_slice_fit_active", False) or ax is not self.ax:
            return None
        out = {"ppm": abs(float(dx))}
        mhz = self._slice_fit_hz_per_ppm()
        if mhz:
            out["hz"] = out["ppm"] * mhz
            out["khz"] = out["hz"] / 1000.0
        return out

    # ---- peak width (FWHM) by a click -----------------------------------

    _WIDTH_COLOR = "tab:purple"
    _WIDTH_ESTIMATE_COLOR = "tab:orange"

    @staticmethod
    def _display_values(axis, unit: str):
        """``axis`` in the display ``unit`` (ppm, hz, khz), or None."""
        if axis is None or getattr(axis, "kind", None) != "frequency":
            return None
        if unit == "ppm":
            return None if axis.ppm is None else np.asarray(axis.ppm, dtype=float)
        if unit in ("hz", "khz") and axis.hz is not None:
            hz = np.asarray(axis.hz, dtype=float)
            return hz / 1000.0 if unit == "khz" else hz
        return None

    def _width_targets(self, ax):
        """What a Width click in ``ax`` measures: ``("2d", spectrum)``,
        ``("1d", [(label, ppm, y, hz_per_ppm, display_x)])`` or
        ``(None, reason)``."""
        unit = self._axis_unit
        if getattr(self, "_slice_fit_active", False) and ax is self.ax:
            args = self._slice_fit_args or {}
            x = np.asarray(args.get("x"), dtype=float)
            y = np.asarray(args.get("y_data"), dtype=float)
            return "1d", [("fit data", x, y, self._slice_fit_hz_per_ppm(), x)]
        if self._compare_axes and ax in self._compare_axes:
            i = self._compare_axes.index(ax)
            spectra = [(i, self._compare_spectra[i])] if i < len(self._compare_spectra) else []
        elif self._compare_spectra and self._contour_spectrum is None \
                and self._current_1d_spectrum is None and ax is self.ax \
                and not getattr(self, "_external_plot", None):
            spectra = list(enumerate(self._compare_spectra))         # overlay
        elif self._contour_spectrum is not None:
            spectrum = self._contour_spectrum
            if ax is not self.ax:
                return None, "Click on the 2D map itself (not a projection)."
            if spectrum.is_pseudo_2d:
                return None, ("A pseudo-2D's rows are not one peak in 2D: keep a slice and "
                              "measure it as a 1D.")
            return "2d", spectrum
        elif self._current_1d_spectrum is not None:
            spectra = [(0, self._current_1d_spectrum)]
        else:
            return None, "Width works on a spectrum, a Compare view or a spectral fit."
        labels = [entry.label for entry in self._series.get("compare", [])]
        out = []
        for i, spectrum in spectra:
            if spectrum.ndim != 1:
                continue
            axis = spectrum.axes[0]
            display = self._display_values(axis, unit)
            if display is None or axis.ppm is None:
                return None, ("Switch the axis to ppm or Hz to measure a width "
                              f"(it is in {unit}).")
            label = labels[i] if self._compare_spectra and i < len(labels) else \
                _spectrum_label(spectrum)
            out.append((label, np.asarray(axis.ppm, dtype=float),
                        np.asarray(spectrum.data, dtype=float),
                        self._axis_hz_per_ppm(axis), display))
        if not out:
            return None, "No 1D spectrum here to measure."
        return "1d", out

    def _plotted_scale(self, ax, display_x, y):
        """``(a, b, residual)`` with plotted ≈ a·y + b for the drawn line
        of this data (normalised or offset as it was drawn), or None."""
        order = np.argsort(display_x)
        dx, dy = display_x[order], y[order]
        best = None
        for line in ax.get_lines():
            if line.get_gid() or len(line.get_xdata()) < 5:
                continue
            lx = np.asarray(line.get_xdata(), dtype=float)
            ly = np.asarray(line.get_ydata(), dtype=float)
            ok = np.isfinite(lx) & np.isfinite(ly) & (lx >= dx[0]) & (lx <= dx[-1])
            if np.count_nonzero(ok) < 5:
                continue
            yi = np.interp(lx[ok], dx, dy)
            A = np.vstack([yi, np.ones_like(yi)]).T
            (a, b), *_ = np.linalg.lstsq(A, ly[ok], rcond=None)
            span = float(np.ptp(ly[ok])) or 1.0
            res = float(np.sqrt(np.mean((A @ (a, b) - ly[ok]) ** 2))) / span
            if best is None or res < best[2]:
                best = (float(a), float(b), res)
        return best

    def _to_display(self, ppm_value, ppm, display):
        order = np.argsort(ppm)
        return float(np.interp(ppm_value, ppm[order], display[order]))

    def _measure_width_at(self, ax, x: float, y: float) -> None:
        """Width mode's click: the FWHM of the peak there, drawn as a
        double arrow at half height (a cross on a 2D) and reported through
        :attr:`peakWidthMeasured`."""
        from ...notebook import peak_width

        kind, target = self._width_targets(ax)
        if kind is None:
            self.peakWidthMeasured.emit({"error": target})
            return
        try:
            if kind == "2d":
                result = self._measure_width_2d(ax, target, x, y)
            else:
                result = self._measure_width_1d(ax, target, x, y)
        except ValueError as exc:
            self.peakWidthMeasured.emit({"error": f"No width measured: {exc}"})
            return
        self.peakWidthMeasured.emit(result)

    def _measure_width_1d(self, ax, candidates, x, y) -> dict:
        from ...notebook import peak_width

        # The trace nearest the click (an overlay holds several).
        best = None
        for label, ppm, data, mhz, display in candidates:
            order = np.argsort(display)
            click_ppm = float(np.interp(x, display[order], ppm[order]))
            scale = self._plotted_scale(ax, display, data)
            if scale is None:
                distance = 0.0
            else:
                plotted = scale[0] * float(np.interp(x, display[order], data[order])) + scale[1]
                distance = abs(plotted - y)
            if best is None or distance < best[0]:
                best = (distance, label, ppm, data, mhz, display, click_ppm, scale)
        _d, label, ppm, data, mhz, display, click_ppm, scale = best
        result = peak_width.peak_fwhm(ppm, data, click_ppm, hz_per_ppm=mhz)
        left = result["left"] if result["left"] is not None else result["position"] - result["fwhm"] / 2
        right = result["right"] if result["right"] is not None else result["position"] + result["fwhm"] / 2
        if result["left"] is None and result["right"] is not None:
            left = result["right"] - result["fwhm"]
        if result["right"] is None and result["left"] is not None:
            right = result["left"] + result["fwhm"]
        x_l, x_r = (self._to_display(v, ppm, display) for v in (left, right))
        y_half = (scale[0] * result["half_height"] + scale[1]) if scale and scale[2] < 0.05 else y
        summary = peak_width.fwhm_summary(result, label)
        self._draw_width_annotation(ax, [((x_l, y_half), (x_r, y_half))],
                                    self._width_text(result), result["estimate"])
        result.update(kind="1d", spectrum=label, summary=summary)
        return result

    def _measure_width_2d(self, ax, spectrum, x, y) -> dict:
        from ...notebook import peak_width

        unit = self._axis_unit
        a1, a2 = spectrum.axes
        d1, d2 = self._display_values(a1, unit), self._display_values(a2, unit)
        if d1 is None:
            d1 = self._display_values(a1, "ppm")
        if d2 is None or d1 is None or a1.ppm is None or a2.ppm is None:
            raise ValueError(f"switch the axes to ppm or Hz (they are in {unit}).")
        p1, p2 = np.asarray(a1.ppm, dtype=float), np.asarray(a2.ppm, dtype=float)
        o1, o2 = np.argsort(d1), np.argsort(d2)
        f1_click = float(np.interp(y, d1[o1], p1[o1]))
        f2_click = float(np.interp(x, d2[o2], p2[o2]))
        result = peak_width.spectrum_peak_fwhm_2d(spectrum, f2_click, f1_click)
        r2, r1 = result["f2"], result["f1"]
        y0 = self._to_display(result["f1_position"], p1, d1)
        x0 = self._to_display(result["f2_position"], p2, d2)
        h2 = 0.5 * r2["fwhm"]
        h1 = 0.5 * r1["fwhm"]
        segments = [((self._to_display(r2["position"] - h2, p2, d2), y0),
                     (self._to_display(r2["position"] + h2, p2, d2), y0)),
                    ((x0, self._to_display(r1["position"] - h1, p1, d1)),
                     (x0, self._to_display(r1["position"] + h1, p1, d1)))]
        text = "F2 " + self._width_text(r2) + "\nF1 " + self._width_text(r1)
        self._draw_width_annotation(ax, segments, text, result["estimate"])
        label = _spectrum_label(spectrum)
        result.update(kind="2d", spectrum=label, summary=(
            f"{label}: F2 {peak_width.fwhm_summary(r2)}; F1 {peak_width.fwhm_summary(r1)}"))
        return result

    @staticmethod
    def _width_text(result) -> str:
        text = f"FWHM {result['fwhm']:.4g} {result.get('unit', 'ppm')}"
        if result.get("fwhm_hz") is not None:
            text += f" / {result['fwhm_hz']:.4g} Hz"
        if result.get("estimate"):
            text += " (estimate: overlap)"
        return text

    def _draw_width_annotation(self, ax, segments, text: str, estimate: bool) -> None:
        colour = self._WIDTH_ESTIMATE_COLOR if estimate else self._WIDTH_COLOR
        for (xa, ya), (xb, yb) in segments:
            arrow = ax.annotate(
                "", xy=(xb, yb), xytext=(xa, ya), xycoords="data", textcoords="data",
                arrowprops=dict(arrowstyle="|-|", color=colour, linewidth=1.4,
                                shrinkA=0, shrinkB=0, mutation_scale=4),
                annotation_clip=False, zorder=60)
            self._measurement_artists.append(arrow)
        (xa, ya), (xb, yb) = segments[0]
        label = ax.text(
            (xa + xb) / 2.0, max(ya, yb), text, color=colour, fontsize=10, ha="center",
            va="bottom", zorder=61, clip_on=False,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor=colour, alpha=0.9))
        self._measurement_artists.append(label)
        self.canvas.draw_idle()

    def _axis_hz_per_ppm(self, axis) -> float | None:
        """Hz per ppm on ``axis``: from its own ppm and Hz arrays when it
        has both, else from the SFO1 of the spectrum it belongs to (a
        simulated spectrum carries ppm only -- the crash reported on
        2026-10-01, "This axis has no 'hz' values"). None when neither
        is known."""
        if axis.ppm is not None and axis.hz is not None:
            ppm = np.asarray(axis.ppm, dtype=float)
            hz = np.asarray(axis.hz, dtype=float)
            if ppm.size >= 2 and hz.size >= 2:
                span_ppm = float(ppm[-1] - ppm[0])
                span_hz = float(hz[-1] - hz[0])
                if span_ppm and span_hz:
                    return abs(span_hz / span_ppm)
        candidates = [self._current_1d_spectrum, self._contour_spectrum,
                      *(self._compare_spectra or []), *(self._panel_spectra or [])]
        for spectrum in candidates:
            if spectrum is None:
                continue
            for position, own in enumerate(spectrum.axes):
                if own is axis:
                    meta = spectrum.metadata
                    indirect = spectrum.ndim == 2 and position == 0
                    mhz = meta.sfo1_indirect if indirect else meta.sfo1
                    return float(mhz) if mhz else None
        return None

    def _delta_ppm_hz(self, axis, delta_ppm: float) -> dict:
        """A separation given in ppm (how Compare's 2D panels are drawn),
        as ``{"ppm", "hz", "khz"}``; ``{}`` for an axis with no frequency
        scale (a pseudo-2D's slice index). Hz needs the axis's field."""
        if axis is None or getattr(axis, "kind", None) != "frequency":
            return {}
        out = {"ppm": abs(float(delta_ppm))}
        hz_per_ppm = self._axis_hz_per_ppm(axis)
        if hz_per_ppm:
            out["hz"] = out["ppm"] * hz_per_ppm
            out["khz"] = out["hz"] / 1000.0
        return out

    @staticmethod
    def _format_ppm_hz(label: str, deltas: dict, raw: float) -> str:
        """``Δx = 0.5000 ppm / 250.00 Hz`` -- always both units where the
        axis has them, never a bare number labelled as nothing."""
        parts = []
        if "ppm" in deltas:
            parts.append(f"{deltas['ppm']:.4f} ppm")
        if "hz" in deltas:
            parts.append(f"{deltas['hz']:.2f} Hz")
        if not parts and "label_unit" in deltas:
            parts.append(f"{deltas['value']:.4g} {deltas['label_unit']}")
        if not parts:
            return f"Δ{label} = {raw:.4g}" + (" slices" if label == "y" else "")
        return f"Δ{label} = " + " / ".join(parts)

    def _format_delta(self, label: str, deltas: dict, raw: float) -> str:
        """One axis's separation, every applicable unit in one string --
        never a bare number, and never a number carrying a unit it was
        not converted into (see :meth:`_measurement_unit_order`)."""
        parts = [fmt.format(deltas[unit])
                 for unit, fmt in self._measurement_unit_order() if unit in deltas]
        if not parts and "ppm" in deltas and self._axis_unit == "mm":
            parts = [self._MEASUREMENT_UNIT_FORMATS["ppm"].format(deltas["ppm"])]
        if not parts and "label_unit" in deltas:
            parts = [f"{deltas['value']:.4g} {deltas['label_unit']}"]
        if not parts:
            # No spectral axis behind this direction and no unit on its
            # label -- an intensity, or a slice index. Report the plain
            # difference, said plainly.
            return f"Δ{label} = {raw:.4g}"
        text = f"Δ{label} = " + " / ".join(parts)
        if deltas.get("ratio"):
            text += f" (×{deltas['ratio']:.3g})"
        return text

    def _record_measurement_point(self, ax, x: float, y: float) -> None:
        """Record one click of a measurement. The second click in the
        same Axes completes it; a second click somewhere else starts
        over from there, since a separation across two different
        coordinate systems is not a measurement of anything."""
        if not self._measurement_mode or ax is None or getattr(ax, "name", "") == "3d":
            return
        if self._measurement_points and self._measurement_points[0][0] is not ax \
                and not self._compare_rows_pair(self._measurement_points[0][0], ax):
            self._measurement_points = []
            self._clear_pending_marker()
        self._measurement_points.append((ax, x, y))
        if len(self._measurement_points) == 2:
            self._complete_measurement()
        else:
            self._draw_pending_marker(ax, x, y)

    def _compare_rows_pair(self, ax1, ax2) -> bool:
        """Both Axes are rows of the current Compare stack: they share one
        chemical-shift axis, so a horizontal distance between a point on
        one spectrum and a point on another is a real measurement (user
        request, 2026-10-06: "measure the horizontal distance between two
        different spectra")."""
        rows = self._compare_axes or []
        return ax1 is not ax2 and ax1 in rows and ax2 in rows

    def _draw_pending_marker(self, ax, x: float, y: float) -> None:
        """A marker on the first click, so it is visible that the
        measurement has started and where from -- without it, the first
        click of the pair does nothing observable at all, which reads
        as measurement mode being broken."""
        marker = ax.plot(
            [x], [y], marker="+", markersize=self._measure_marker_size(),
            markeredgewidth=self._measurement_width + 0.2,
            color=self._measurement_color, zorder=60, gid=self._MEASUREMENT_PENDING_GID,
        )[0]
        marker._opennmr_measure = True
        self._measurement_artists.append(marker)
        self.canvas.draw_idle()

    def _clear_pending_marker(self) -> None:
        keep = []
        for artist in self._measurement_artists:
            if artist.get_gid() == self._MEASUREMENT_PENDING_GID:
                self._remove_artist(artist)
            else:
                keep.append(artist)
        self._measurement_artists = keep

    def _complete_measurement(self) -> None:
        """Draw and report the finished measurement, then clear the
        pending pair so the next two clicks start a new one."""
        (ax, x1, y1), (ax2, x2, y2) = self._measurement_points
        self._measurement_points = []
        self._clear_pending_marker()
        if ax2 is not ax:
            self._complete_cross_measurement(ax, x1, y1, ax2, x2, y2)
            return

        ppm_view = self._ppm_drawn_view(ax)
        x_axis, y_axis = ppm_view if ppm_view is not None else self._measurement_axes_for(ax)
        dx_raw, dy_raw = abs(x2 - x1), abs(y2 - y1)
        slice_fit_deltas = self._slice_fit_delta(ax, dx_raw)
        if slice_fit_deltas is not None:
            # A fit drawn by show_slice_fit: x is always ppm (whatever the
            # display unit), with Hz from the fitted spectrum's field --
            # it used to come back as a bare number (explicit user report,
            # 2026-10-16: "does not give the units ... only ppm").
            dx_units, dy_units = slice_fit_deltas, {}
        elif ppm_view is not None:
            # Drawn in ppm whatever the display unit (Compare's panels,
            # the heatmap, DOSY, the processing preview): both
            # separations in ppm and Hz.
            dx_units = self._delta_ppm_hz(ppm_view[0], dx_raw)
            dy_units = self._delta_ppm_hz(ppm_view[1], dy_raw)
        elif self._external_plot and x_axis is None and y_axis is None:
            # A figure drawn elsewhere (Simulate): its axis labels say
            # what the axes are in.
            dx_units = self._external_delta(ax.xaxis, dx_raw)
            dy_units = self._external_delta(ax.yaxis, dy_raw)
        else:
            dx_units = self._delta_in_units(x_axis, dx_raw)
            dy_units = self._delta_in_units(y_axis, dy_raw)
        # Whatever is left without a unit -- a time, a delay, a D, a
        # field, an axis of a figure drawn elsewhere -- takes the unit
        # its label states, so no measurement is a bare number. Not the
        # y of a 1D spectrum: that is an intensity (see below).
        if not dx_units:
            dx_units = self._label_delta(ax, "x", x1, x2)
        if not dy_units and y_axis is None:
            dy_units = self._label_delta(ax, "y", y1, y2)

        if ppm_view is not None and self._panel_axes and ax in self._panel_axes:
            parts = [self._format_ppm_hz("x", dx_units, dx_raw)]
            if dy_raw:
                parts.append(self._format_ppm_hz("y", dy_units, dy_raw))
            summary = "   ".join(parts)
            self._draw_measurement_annotation(ax, x1, y1, x2, y2, summary)
            self.measurementCompleted.emit({
                "display_unit": "ppm",
                "x1": float(x1), "x2": float(x2), "y1": float(y1), "y2": float(y2),
                "dx_display": float(dx_raw), "dy_display": float(dy_raw),
                "dx": dx_units, "dy": dy_units, "summary": summary})
            return

        parts = [self._format_delta("x", dx_units, dx_raw)]
        # A y separation is only worth reporting when there is one: on
        # a 1D the second click's y is an intensity the user did not
        # aim at, and quoting it as a measurement invites reading it as
        # meaningful. Reported when it maps to a real axis (a 2D's F1),
        # or when it is a deliberate, non-trivial intensity difference.
        if dy_units or (dy_raw and y_axis is None and self._contour_spectrum is not None):
            parts.append(self._format_delta("y", dy_units, dy_raw))
        summary = "   ".join(parts)

        self._draw_measurement_annotation(ax, x1, y1, x2, y2, summary)

        result = {
            "display_unit": self._axis_unit,
            "x1": float(x1), "x2": float(x2), "y1": float(y1), "y2": float(y2),
            "dx_display": float(dx_raw), "dy_display": float(dy_raw),
            "dx": dx_units, "dy": dy_units,
            "summary": summary,
        }
        self.measurementCompleted.emit(result)

    def _complete_cross_measurement(self, ax1, x1, y1, ax2, x2, y2) -> None:
        """A measurement from a point on one Compare row to a point on
        another: the horizontal separation only (each row has its own
        intensity scale, so a vertical one means nothing), in every unit
        of the first row's axis -- with a note when the two spectra were
        recorded at different fields, whose Hz per ppm then differ."""
        rows = self._compare_axes
        i1, i2 = rows.index(ax1), rows.index(ax2)
        axis1 = self._compare_spectra[i1].axes[0] if i1 < len(self._compare_spectra) else None
        axis2 = self._compare_spectra[i2].axes[0] if i2 < len(self._compare_spectra) else None
        dx_raw = abs(x2 - x1)
        dx_units = self._delta_in_units(axis1, dx_raw)
        summary = self._format_delta("x", dx_units, dx_raw)
        note = ""
        h1 = self._axis_hz_per_ppm(axis1) if axis1 is not None else None
        h2 = self._axis_hz_per_ppm(axis2) if axis2 is not None else None
        if h1 and h2 and abs(h1 - h2) > 1e-3 * max(h1, h2) and "hz" in dx_units:
            note = " (Hz at the first spectrum's field; the two differ)"
            summary += note
        labels = [entry.label for entry in self._series.get("compare", [])]
        names = [labels[i] if i < len(labels) else f"row {i + 1}" for i in (i1, i2)]
        self._draw_cross_annotation(ax1, x1, y1, ax2, x2, y2, summary)
        self.measurementCompleted.emit({
            "display_unit": self._axis_unit, "x1": float(x1), "x2": float(x2),
            "y1": float(y1), "y2": float(y2), "dx_display": float(dx_raw), "dy_display": 0.0,
            "dx": dx_units, "dy": {}, "summary": summary, "across": (i1, i2),
            "spectra": names})

    def _draw_cross_annotation(self, ax1, x1, y1, ax2, x2, y2, summary: str) -> None:
        """A double-headed arrow from a point in one row to a point in
        another, labelled at its midpoint (in figure coordinates, since it
        belongs to neither row)."""
        from matplotlib.patches import ConnectionPatch
        arrow = ConnectionPatch(
            xyA=(x1, y1), coordsA=ax1.transData, xyB=(x2, y2), coordsB=ax2.transData,
            arrowstyle="<|-|>", color=self._measurement_color, linewidth=self._measurement_width,
            mutation_scale=self._measure_head_scale(), shrinkA=0, shrinkB=0, zorder=60)
        self.figure.add_artist(arrow)
        p1 = self.figure.transFigure.inverted().transform(ax1.transData.transform((x1, y1)))
        p2 = self.figure.transFigure.inverted().transform(ax2.transData.transform((x2, y2)))
        label = self.figure.text(
            (p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0, summary,
            color=self._measurement_color, fontsize=self._measurement_fontsize, ha="center",
            va="center", zorder=61,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor=self._measurement_color, alpha=0.9))
        arrow._opennmr_measure = label._opennmr_measure = True
        self._measurement_artists.extend((arrow, label))
        self.canvas.draw_idle()

    def _draw_measurement_annotation(self, ax, x1, y1, x2, y2, summary: str) -> None:
        """A double-headed arrow between the two points, labelled with
        the summary at its midpoint. Left on the plot until measurement
        mode is switched off (or the view rebuilds), so several
        measurements can stand at once -- a J-coupling analysis is
        rarely a single measurement."""
        arrow = ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1), xycoords="data", textcoords="data",
            arrowprops=dict(arrowstyle="<|-|>", color=self._measurement_color,
                            linewidth=self._measurement_width,
                            mutation_scale=self._measure_head_scale(), shrinkA=0, shrinkB=0),
            annotation_clip=False, zorder=60,
        )
        label = ax.text(
            (x1 + x2) / 2.0, max(y1, y2), summary,
            color=self._measurement_color, fontsize=self._measurement_fontsize, ha="center",
            va="bottom", zorder=61, clip_on=False,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor=self._measurement_color, alpha=0.9),
        )
        arrow._opennmr_measure = label._opennmr_measure = True
        self._measurement_artists.extend((arrow, label))
        self.canvas.draw_idle()

    @staticmethod
    def _remove_artist(artist) -> None:
        try:
            artist.remove()
        except (ValueError, NotImplementedError):
            pass   # already detached by a figure rebuild -- nothing to do

    def clear_measurements(self, redraw: bool = True) -> None:
        """Take every measurement annotation back off the plot, and
        drop any half-finished pair."""
        artists, self._measurement_artists = self._measurement_artists, []
        self._measurement_points = []
        for artist in artists:
            self._remove_artist(artist)
        if artists and redraw:
            self.canvas.draw_idle()


def _spectrum_label(spectrum: NMRSpectrum) -> str:
    meta = spectrum.metadata
    # A name the user gave the experiment (Open experiments > Rename…, or
    # the Plot Style name box) wins -- the same override Session.label
    # honours, so the legend says what the list says. This was missing:
    # a renamed spectrum still went into "Show Legend" under its folder
    # name (user report, 2026-09-28).
    override = (meta.raw or {}).get("display_label")
    if isinstance(override, str) and override.strip():
        return override.strip()
    if meta.vendor == "derived" and meta.source_path:
        # A derived spectrum's source_path is a *label*, not a path --
        # basename()ing it truncates at any "/" it contains (a unit
        # like "m^2/s", a ratio from spectrum arithmetic). Same fix as
        # Session.label; both read this field the same way -- including
        # dropping a suffix it already carries (strip_label_suffix).
        from ..session import strip_label_suffix
        name = strip_label_suffix(meta.source_path)
    else:
        name = os.path.basename(meta.source_path.rstrip("/")) if meta.source_path else None
    if name and meta.nucleus:
        return f"{name} ({meta.nucleus})"
    return name or meta.nucleus or "spectrum"
