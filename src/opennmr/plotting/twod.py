"""
2D spectrum plotting: contour + projections, multi-panel, and overlay.

``plot_2d_contour`` is ``ax``-first (``nmrtools2``'s ``NMR2Dplotter``
already accepted an external ``ax=``, so this ports over directly) and
never touches global matplotlib state when embedded that way -- see
:mod:`opennmr.plotting.style`'s docstring for why that matters for a
GUI canvas. Unlike the original, this only ever takes an already-
loaded :class:`NMRSpectrum` (not a raw path) -- keeping the plotting
layer decoupled from :mod:`opennmr.io`; load first, plot second.

Ported from ``nmrtools2``'s ``two_d.py``.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import MultipleLocator, MaxNLocator
from mpl_toolkits.axes_grid1 import axes_size
from mpl_toolkits.axes_grid1.axes_divider import AxesDivider

from ..core.registry import find_gamma, NUC_label
from ..core.spectrum import NMRSpectrum
from ..processing.slicing import get_trace
from .style import (setup_plotting, get_stack_colormap, get_contour_cmap, downsample_2d, save_figure,
                    draw_positive_contours)


# Absolute ceiling on how many points are handed to matplotlib's
# contour generator, applied on top of whatever resolution the caller
# asked for. Only very large datasets are affected at all.
#
# Why a ceiling exists: contour cost is linear in *levels x points*
# (measured -- and, notably, *not* in the number of `ax.contour` calls,
# so drawing the positive and negative level sets separately costs the
# same as one combined call), and there is no way to thread it.
# matplotlib never passes `thread_count` through to contourpy and
# offers no route to inject one, and contourpy does not release the
# GIL anyway (measured at 0.94x/0.97x across 4 threads), so resolution
# is the only lever that exists. On the project's real EXSY benchmark
# (1002 x 65536 = 65.7M points) full-resolution contouring costs ~4.0s,
# which was the single largest cost in a load once the duplicated work
# was removed.
#
# Why this value: full resolution in the Spectrum tab is an explicit
# standing user instruction, so this is deliberately a safety ceiling
# for outliers rather than a general downsampling policy -- it leaves
# every other real dataset in this project's own test_data untouched.
# For scale: the COSY (0.5M), bruker_T2 (2.1M) and bruker_PFG (4.2M)
# datasets are all far below it and render exactly as before.
#
# The value is defined as a *quarter of the EXSY benchmark's own point
# count*, on direct user instruction: "I stated that the maximum should
# be about that value for these EXSYs and should be applied to restrict
# any dataset to 0.25x the EXSY's number of points". The pseudo-2D EXSY
# is 1002 x 65536 = 65,667,072 points, so the ceiling is 16,416,768.
#
# (It was briefly raised to 2**25 on an earlier reading of "at least
# half", then corrected back here -- 0.25x is the preference.)
#
# At this ceiling the EXSY renders at 1002 x 16384, a quarter of its F2
# resolution, costing ~1.0s of contour time against ~4.0s at full.
_EXSY_BENCHMARK_POINTS = 1002 * 65536
_MAX_CONTOUR_POINTS = _EXSY_BENCHMARK_POINTS // 4

# Don't sacrifice the indirect dimension to save the direct one: F1 is
# usually far smaller (1002 vs 65536 on the benchmark) and far more
# sensitive to losing rows, so the cap comes out of F2 first and only
# touches F1 if F1 alone already exceeds the ceiling.
_MIN_CAPPED_CONTOUR_X = 4096

#: Colour for negative-going contours. Teal, not red -- see the
#: contour call in :func:`plot_2d_contour` for why.
NEGATIVE_CONTOUR_COLOUR = "teal"


def _cap_contour_points(max_x: int, max_y: int) -> tuple[int, int]:
    """Clamp a requested contour resolution to
    :data:`_MAX_CONTOUR_POINTS` total, reducing F2 (``max_x``) first.

    Returns the requested values unchanged whenever they are already
    within the ceiling, which is the case for every dataset in this
    project's own test data except the largest EXSY one.
    """
    if max_x <= 0 or max_y <= 0 or max_x * max_y <= _MAX_CONTOUR_POINTS:
        return max_x, max_y
    # Take it out of F2 first, but never below the floor (and never
    # *up*, for a caller that asked for less than the floor already).
    capped_x = min(max_x, max(_MIN_CAPPED_CONTOUR_X, _MAX_CONTOUR_POINTS // max_y))
    if capped_x * max_y <= _MAX_CONTOUR_POINTS:
        return capped_x, max_y
    # F2 is down at its floor and the total is *still* over, i.e. F1 is
    # genuinely oversized on its own -- reduce it too, rather than
    # returning something that silently breaches the ceiling anyway.
    # (Caught by the cap's own test: a hypothetical 65536x65536 request
    # came back 16x over the ceiling before this branch existed.)
    return capped_x, max(1, _MAX_CONTOUR_POINTS // capped_x)


def _prepare_contour(spectrum: NMRSpectrum, contour_fraction_x=0.25, contour_fraction_y=0.25,
                      ncontourpts_x=16384, ncontourpts_y=1024, factor=0.07, clevels=7,
                      level_mode="power", power=1.25):
    """Shared contour-preparation logic: resolve the F1 axis (ppm, or
    slice index for pseudo-2D), downsample for fast rendering, and
    compute the contour levels. Internal helper used by both
    :func:`plot_2d_contour` and :func:`overlay_plot`.

    ``level_mode`` : ``"power"`` (default) spaces ``clevels`` levels
    between ``factor * max(data)`` and ``max(data)`` on a power-law
    curve controlled by ``power`` (steeper/more levels bunched near
    the noise floor as ``power`` increases past 1) -- good for picking
    out weak peaks near a strong one without saturating the display.
    ``"linear"`` spaces them evenly instead, which better reflects
    relative peak heights at a glance.

    ``factor``/``clevels`` defaults (0.07, 7) are deliberately modest --
    a much lower ``factor`` (e.g. 1e-3) combined with many levels
    (e.g. 20) reaches so far into the noise floor, and bunches so many
    of those levels near it, that real but weaker signal ends up
    buried under a dense mass of near-noise-level rings rather than a
    few clean, readable contours (reported directly as "the contours
    seem to start out scaled up way too high").
    """
    data = spectrum.data
    x_ppm = spectrum.axes[1].ppm
    pseudo2d = spectrum.is_pseudo_2d
    y_axis = spectrum.axes[0].index.astype(float) if pseudo2d else spectrum.axes[0].ppm

    if contour_fraction_x is not None or contour_fraction_y is not None:
        max_x = min(
            int(len(x_ppm) * contour_fraction_x) if contour_fraction_x is not None else len(x_ppm),
            len(x_ppm), ncontourpts_x,
        )
        max_y = min(
            int(len(y_axis) * contour_fraction_y) if contour_fraction_y is not None else len(y_axis),
            len(y_axis), ncontourpts_y,
        )
    else:
        max_x, max_y = len(x_ppm), len(y_axis)

    max_x, max_y = _cap_contour_points(max_x, max_y)

    if (max_x, max_y) != (len(x_ppm), len(y_axis)):
        contour_data, contour_x, contour_y = downsample_2d(
            data, x_ppm, y_axis, max_points_x=max_x, max_points_y=max_y,
        )
    else:
        contour_x, contour_y, contour_data = x_ppm, y_axis, data

    if level_mode not in ("power", "linear"):
        raise ValueError("level_mode must be 'power' or 'linear'")

    # Both sides are scaled to the same reference -- the largest
    # *absolute* value in the data -- so a contour at a given level
    # means the same intensity whichever sign it has.
    #
    # Real user report: "the negative contours look really prominent vs
    # the positive. can the size of the contours be based on the
    # greatest absolute value, not the greatest negative value?" That is
    # exactly what was wrong: the negative side used to be scaled to
    # `amin(data)` independently, so on a spectrum whose negative
    # excursions are small (dispersive tails, artifacts) its lowest
    # level sat at `factor * |amin|` -- a far lower *absolute* threshold
    # than the positive side's `factor * amax`. The negative contours
    # therefore started much closer to the noise floor and drew far more
    # rings, reading as "prominent" when the underlying signal was
    # weaker, not stronger.
    peak = float(np.amax(np.abs(data)))
    thresvec = _contour_levels(factor * peak, peak, clevels, level_mode, power)

    # The negative levels are now the exact mirror of the positive ones,
    # drawn only when there is genuinely negative signal above the same
    # threshold the positive side starts at.
    neg_thresvec = [-level for level in thresvec] if float(np.amin(data)) < -factor * peak else None

    return contour_x, contour_y, contour_data, thresvec, y_axis, neg_thresvec


def _contour_levels(thresh_min, thresh_max, clevels, level_mode, power):
    """``clevels`` threshold levels between ``thresh_min`` and
    ``thresh_max`` (same sign as each other; either both positive, for
    the usual case, or both negative, for the mirrored negative-side
    levels ``_prepare_contour`` also computes) -- factored out of
    :func:`_prepare_contour` so the same spacing logic (and its own
    bug-fix history, see below) applies to both sides identically."""
    if level_mode == "linear":
        return list(np.linspace(thresh_min, thresh_max, clevels))
    # Real bug, found and fixed: the exponent used to be
    # `power * i / clevels`, which only reaches exactly `power`
    # (not 1) at the top level (i == clevels) -- for power > 1
    # (the default is 1.25), that overshoots thresh_max, so the
    # top several "contour levels" sit above every real data
    # point and are never actually drawn. The visible levels were
    # then squeezed into the remaining lower fraction of the
    # range, several coarser than intended -- moderate, real
    # signal often fell entirely between two of them, looking
    # like a 2D/pseudo-2D dataset "starts out scaled way too
    # high" (barely anything visible) until the threshold was
    # manually scrolled down a lot. `(i / clevels) ** power`
    # always reaches exactly 1 at i == clevels (so the top level
    # is exactly thresh_max, regardless of power), and for
    # power > 1 still bunches more levels toward thresh_min first
    # -- matching this function's own docstring ("steeper/more
    # levels bunched near the noise floor as power increases").
    cc2 = np.linspace(1, clevels, clevels)
    return [thresh_min * (thresh_max / thresh_min) ** ((i / clevels) ** power) for i in cc2]


#: Default colour cycle for extracted-slice lines and their traces.
#: Each slice's dashed line on the contour and its trace in the panel
#: share one colour -- that pairing is the whole point, so the cycle is
#: chosen to stay distinguishable both from the black positive contours
#: and from each other, including for the common red-green colour-vision
#: deficiencies.
#: gid on a slice/trace panel's own Axes, so a caller that cannot
#: receive it as a return value can find it on the figure -- see
#: :func:`_draw_traces_panel`.
SLICE_PANEL_GID = "slice_panel"

SLICE_COLOURS = ("#3b4cc0", "#b02ba0", "#e8703a", "#2a9d8f", "#7a4fbf", "#c0392b")

#: Dash pattern for the slice lines: a long-dash/dot alternation, so
#: they read as annotation rather than as another contour level, and
#: stay legible where they cross dense contours.
_SLICE_DASHES = (0, (7, 3, 1, 3))


def _right_label_clearance(ax, fontsize: float) -> float:
    """Inches of room the contour's **right-hand** y tick labels and
    axis label need, so a panel appended to the right can be padded
    clear of them.

    ``plot_2d_contour`` puts the y ticks and label on the right (the F1
    projection occupies the left), and the extracted-slice panel is
    appended to that same side -- so the gap between contour and panel
    is the pad *minus* whatever the tick labels take. With a fixed pad
    that gap shrinks as the numbers get wider, which is exactly what
    was reported: "I still see the extracted slices overlapping with
    the y-axis label of the contour, probably because the data I'm
    looking at has units in the 1000s because if I change to kHz (100s
    scale) it is less cut off."

    Measured on a 8 x 6 in figure: a +/-12.5 ppm axis left a 28 px gap,
    the same axis in Hz (+/-5000) left 10 px. Same pad, same figure --
    only the digit count changed.

    Estimated from the tick values and the font rather than measured
    off a renderer: this runs while the layout is being built, before
    anything has been drawn, and the estimate only has to be good
    enough to reserve space.
    """
    try:
        ticks = list(ax.get_yticks())
        formatter = ax.yaxis.get_major_formatter()
        widest = max((len(str(formatter.format_data_short(t)).strip()) for t in ticks),
                     default=4)
    except Exception:  # noqa: BLE001 -- an estimate that fails falls back to a plain one
        widest = 6
    # ~0.6 em per character for a proportional face, plus the rotated
    # axis label lying on its side (about one line's height) and the
    # tick marks themselves.
    return (0.6 * fontsize * widest + 1.8 * fontsize) / 72.0


def _draw_slice_panel(ax, divider, spectrum, slices, *,
                      size, pad, offset_fraction, colours,
                      font, fontsize, axis_label, linewidth, bottom_gap=0.05):
    """Draw extracted slices beside a contour: a dashed line across the
    contour at each slice's own position, and the extracted traces
    stacked in a panel to the right, one colour per slice.

    ``slices`` : one dict per slice, ``{"axis", "index", "label",
    "color"}``. ``axis`` follows
    :func:`opennmr.processing.slicing.get_trace`'s own convention --
    ``"f1"`` fixes a position along F1 and yields a *row* running along
    F2 (the common case). All slices in one call must share an
    ``axis``: the panel has a single x-axis, and rows and columns run
    along different dimensions, so mixing them would put two unrelated
    frequency axes on one plot. The first slice's axis wins and any
    other is skipped rather than silently mis-plotted.

    ``size`` : panel width as a multiple of the contour panel's own
    width (``axes_size.Scaled``), not inches. The same function draws
    both a standalone figure and the GUI's own embedded canvas, and a
    fixed inch width that looks right in one crushes the contour to a
    sliver in the other (measured -- 3.5 inches left the embedded
    contour a few pixels wide).

    **Traces are spread evenly over the panel's full height**, one slot
    each, and **each is normalised to its own height** ("the extracted
    projections should be normalised") so that a weak row can be
    compared with a strong one on lineshape rather than being flattened
    against it. ``offset_fraction`` is how much of its slot each trace
    fills.

    **No connector lines.** An earlier version continued each dashed
    line across the gap and joined it to its own trace's baseline. It
    looked right in a saved figure but was not worth its upkeep in the
    GUI, where zooming, panning and unit changes all move the two
    Axes independently: "drop the lines to the extracted slices, it
    seems too hard to maintain across different views." The shared
    colour is what pairs a line with its trace now.

    Returns the panel Axes, or ``None`` if there was nothing to draw.
    """
    slices = [s for s in slices if s is not None]
    if not slices:
        return None
    axis_kind = slices[0].get("axis", "f1")
    fixed_axis = spectrum.axes[0] if axis_kind == "f1" else spectrum.axes[1]
    fixed_positions = fixed_axis.ppm if fixed_axis.ppm is not None else fixed_axis.index

    entries = []
    for spec in slices:
        if spec.get("axis", "f1") != axis_kind:
            continue
        try:
            trace = get_trace(spectrum, axis=axis_kind, index=int(spec["index"]))
        except (ValueError, IndexError, KeyError):
            # A slice index left over from a dataset that has since been
            # reprocessed to a different size shouldn't take the whole
            # figure down with it -- skip it and draw the rest.
            continue
        trace_axis = trace.axes[0]
        trace_x = trace_axis.ppm if trace_axis.ppm is not None else trace_axis.index
        entries.append({
            "spec": spec,
            "x": trace_x,
            "y": np.asarray(trace.data, dtype=float),
            "position": float(fixed_positions[int(spec["index"]) - 1]),
        })
    return _draw_traces_panel(
        ax, divider, entries, axis_kind=axis_kind, size=size, pad=pad,
        offset_fraction=offset_fraction, colours=colours, font=font,
        fontsize=fontsize, axis_label=axis_label, linewidth=linewidth,
        bottom_gap=bottom_gap,
    )


def _draw_traces_panel(ax, divider, entries, *, axis_kind,
                       size, pad, offset_fraction, colours,
                       font, fontsize, axis_label, linewidth, bottom_gap=0.05):
    """The panel half of :func:`_draw_slice_panel`, over already-extracted
    traces rather than over an :class:`NMRSpectrum`.

    Split out so a **DOSY** can have the same thing. Its map is a plain
    ``{"Z", "D", "ppm"}`` dict from the inverse Laplace transform, not a
    spectrum with axes to slice, so it could not call the function
    above -- and copying the stacking, per-trace normalisation, label
    placement and dashed-marker code into :mod:`opennmr.plotting.fit`
    would have been the second place for all of it to drift. Explicit
    user request: extracting a line from a DOSY should offer "the
    option to plot alongside with the dashed line like in normal 2D
    contours".

    ``entries`` : one dict per trace, ``{"spec", "x", "y", "position"}``
    -- ``spec`` carrying the ``label``/``color`` the caller chose,
    ``position`` the coordinate on the *fixed* axis where the dashed
    marker goes.

    ``bottom_gap`` : space between the lowest trace's lowest point and
    the panel's axis line, as a fraction of that trace's height. Each
    trace starts at its own minimum, so without it the bottom trace's
    baseline lay *on* the axis line and could not be seen (user report,
    2026-10-15).
    """
    if not entries:
        return None
    traces = [(e["spec"], e) for e in entries]

    # sharex for a row panel, so the traces stay aligned under the
    # contour through every later zoom/pan rather than only at draw
    # time. Setting the limits once from ax.get_xlim() was not enough:
    # the GUI's own preserve-the-view redraw restores the contour's
    # limits *after* this function has already run, so a zoomed contour
    # got a full-range panel beside it. A column panel's x axis is the
    # contour's *y*, which matplotlib cannot share across orientations,
    # so that case still takes a one-time copy below.
    # Pad past whatever the contour's own right-hand labels occupy, not
    # a fixed distance that they eat into -- see _right_label_clearance.
    panel = divider.append_axes(
        "right", axes_size.Scaled(size),
        pad=max(pad, _right_label_clearance(ax, fontsize) + 0.15),
        **({"sharex": ax} if axis_kind == "f1" else {}),
    )
    # Span every row of the divider, not just the contour's own, so the
    # panel reaches the top of the F2 projection strip -- explicit user
    # request, "the extracted slice plot should be able to reach the top
    # height of the top projection". append_axes places it alongside the
    # main cell only; re-pointing its locator at rows 0..n gives it the
    # contour's height plus the pad and the strip above it. Self-
    # adapting: with no top projection there is one row and this is a
    # no-op.
    panel.set_axes_locator(divider.new_locator(
        nx=len(divider.get_horizontal()) - 1, ny=0, ny1=len(divider.get_vertical()),
    ))
    # Evenly spread: one slot per trace across a normalised 0..1 panel,
    # each trace scaled to fill its own slot.
    n = len(traces)
    slot = 1.0 / n
    heights = [float(np.max(e["y"]) - np.min(e["y"])) or 1.0 for _, e in traces]

    # Stack so the panel's vertical order matches the contour's: the
    # slice whose dashed line sits highest gets the highest trace.
    lo, hi = ax.get_ylim() if axis_kind == "f1" else ax.get_xlim()
    span = (hi - lo) or 1.0
    palette = tuple(colours) if colours else SLICE_COLOURS
    positions = [float(e["position"]) for _, e in traces]
    order = sorted(range(n), key=lambda i: (positions[i] - lo) / span)
    baselines = {slot_index: rank * slot for rank, slot_index in enumerate(order)}

    for i, (spec, entry) in enumerate(traces):
        colour = spec.get("color") or palette[i % len(palette)]
        trace_x = entry["x"]
        baseline = baselines[i]
        y = np.asarray(entry["y"], dtype=float)
        # Normalised per trace, not on one shared scale.
        scale = (slot * offset_fraction) / heights[i]
        panel.plot(trace_x, (y - float(np.min(y))) * scale + baseline,
                   color=colour, linewidth=linewidth)

        label = spec.get("label")
        if label:
            # Above the trace's own baseline noise rather than on it --
            # the left-hand end of a real row is rarely flat.
            panel.text(0.015, baseline + 0.45 * slot, str(label), color=colour,
                       transform=panel.get_yaxis_transform(which="grid"),
                       fontname=font, fontsize=fontsize - 2, va="bottom", ha="left",
                       clip_on=False)

        # Same weight as the trace the marker is paired with, rather
        # than the thinner annotation weight this used to have -- so a
        # marker and its own trace read as one object at a glance
        # ("linewidths of extracted slices should be 2" meant both).
        position = positions[i]
        if axis_kind == "f1":
            ax.axhline(position, color=colour, linestyle=_SLICE_DASHES, linewidth=linewidth)
        else:
            ax.axvline(position, color=colour, linestyle=_SLICE_DASHES, linewidth=linewidth)

    # Match the contour's own limits on the shared dimension, so a peak
    # in a trace sits directly under the same shift in the contour.
    # (Already automatic for a row panel via sharex, above.)
    if axis_kind != "f1":
        panel.set_xlim(ax.get_ylim())
    panel.set_ylim(-max(0.0, float(bottom_gap)) * slot * offset_fraction, 1.0)
    if axis_label:
        panel.set_xlabel(_italic_label(axis_label), fontsize=fontsize)
    panel.tick_params(axis="y", which="both", left=False, right=False,
                      labelleft=False, labelright=False)
    panel.tick_params(axis="x", which="both", labelsize=fontsize)
    for spine in ("top", "right", "left"):
        panel.spines[spine].set_visible(False)
    # Tagged so a caller that cannot take it as a return value can
    # still find it -- see plot_dosy, which shares a fixed-arity
    # unpack with overlay_dosy at every call site.
    panel.set_gid(SLICE_PANEL_GID)
    return panel


#: Most spectra :func:`overlay_plot` accepts. Raised from 3 on explicit
#: request ("allow up to 6"); past that, flat-colour contours stop
#: being told apart, and side-by-side panels are the better view.
MAX_OVERLAY_SPECTRA = 6

#: Where a GUI inset "corner" puts an inset, as ``(x0, y0)`` of its
#: lower-left corner given its size ``s`` and margin ``m`` (axes fraction).
INSET_CORNERS = {
    "upper right": lambda s, m: (1 - s - m, 1 - s - m),
    "upper left": lambda s, m: (m, 1 - s - m),
    "lower left": lambda s, m: (m, m),
    "lower right": lambda s, m: (1 - s - m, m),
}


def inset_bounds(corner: str = "upper right", size: float = 0.4, margin: float = 0.03):
    """``(x0, y0, width, height)`` in axes fraction for a square inset of
    ``size`` (fraction of the parent axes) in ``corner``."""
    if corner not in INSET_CORNERS:
        raise ValueError(f"corner must be one of {sorted(INSET_CORNERS)}, not {corner!r}")
    if not 0.05 <= size <= 0.9:
        raise ValueError("size must be between 0.05 and 0.9 of the parent axes")
    x0, y0 = INSET_CORNERS[corner](size, margin)
    return (x0, y0, size, size)


def crop_2d(spectrum: NMRSpectrum, xlims=None, ylims=None) -> NMRSpectrum:
    """``spectrum`` restricted to ``xlims`` (F2, ppm) and ``ylims`` (F1:
    ppm, or slice index for a pseudo-2D), either order. Keeps one point
    beyond each edge so contours run to the edge of an inset rather than
    stopping a point short. Raises if a range holds no points."""
    def _window(values, lims):
        if lims is None:
            return slice(None)
        lo, hi = sorted(float(v) for v in lims)
        inside = np.nonzero((values >= lo) & (values <= hi))[0]
        if inside.size == 0:
            raise ValueError(f"The range {lo:g} to {hi:g} contains no data points.")
        return slice(max(int(inside.min()) - 1, 0), int(inside.max()) + 2)

    f2, f1 = spectrum.axes[1], spectrum.axes[0]
    f1_values = f1.index.astype(float) if spectrum.is_pseudo_2d else f1.ppm
    cols = _window(f2.ppm, xlims)
    rows = _window(f1_values, ylims)

    def _axis(axis, window):
        return dataclasses.replace(axis, **{name: getattr(axis, name)[window]
                                            for name in ("ppm", "hz", "index")
                                            if getattr(axis, name) is not None})

    return dataclasses.replace(
        spectrum, data=spectrum.data[rows, cols],
        data_imag=None if spectrum.data_imag is None else spectrum.data_imag[rows, cols],
        axes=[_axis(f1, rows), _axis(f2, cols)])


def _levels_from_full(spectrum, factor, clevels, level_mode, power):
    """Contour levels from the *whole* spectrum's peak, so an inset's
    contours mean the same intensities as the plot it zooms into."""
    peak = float(np.amax(np.abs(spectrum.data)))
    if peak == 0:
        return [], None
    levels = _contour_levels(factor * peak, peak, clevels, level_mode, power)
    negative = ([-lv for lv in levels] if float(np.amin(spectrum.data)) < -factor * peak else None)
    return levels, negative


def draw_insets(parent_ax, spectra, insets, *, colours=None, factor=0.07, clevels=7,
                level_mode="power", power=1.25, contour_linewidth=0.8, fontsize=12,
                colour_spectra=None):
    """Draw zoomed insets of ``spectra`` (one spectrum, or the overlaid
    set) on ``parent_ax``.

    Each inset in ``insets`` is a dict with ``xlims`` (F2) and ``ylims``
    (F1) -- the region shown -- and either ``bounds``
    (``(x0, y0, w, h)``, axes fraction) or ``corner``/``size`` (see
    :func:`inset_bounds`). Optional: ``mark_inset`` (default True) draws
    the box and connectors to the region on the parent; ``factor``
    overrides the lowest contour level for that inset only.

    **The inset is drawn from the data in its region at full
    resolution** (the parent is usually downsampled for speed, which a
    zoom would expose), **with levels taken from each whole spectrum**,
    so the same contour means the same intensity in the inset and in
    the parent. Lower ``factor`` for an inset to bring out weak peaks.

    Unlike the notebook's ``NMR2Dplotter(inset=...)``, which recursed
    into the full plotter, this never builds an ``AxesDivider`` on the
    inset -- that replaces the inset's own placement and silently
    mislays it (the notebook refuses projections in insets for the same
    reason).

    Returns the inset axes, in order.
    """
    spectra = list(spectra) if isinstance(spectra, (list, tuple)) else [spectra]
    colours = colours or ["black"] * len(spectra)
    ref = spectra[0]
    created = []
    for i, spec in enumerate(insets or []):
        if "xlims" not in spec or "ylims" not in spec:
            raise ValueError(f"inset {i + 1} needs both xlims (F2) and ylims (F1).")
        bounds = spec.get("bounds") or inset_bounds(spec.get("corner", "upper right"), spec.get("size", 0.4))
        inset_factor = spec.get("factor") or factor
        ax = parent_ax.inset_axes(bounds)
        ax.set_zorder(parent_ax.get_zorder() + 5)
        for spectrum, colour in zip(spectra, colours):
            region = crop_2d(spectrum, spec["xlims"], spec["ylims"])
            levels, negative = _levels_from_full(spectrum, inset_factor, clevels, level_mode, power)
            if not levels:
                continue
            x = region.axes[1].ppm
            y = region.axes[0].index.astype(float) if region.is_pseudo_2d else region.axes[0].ppm
            max_x, max_y = _cap_contour_points(len(x), len(y))
            data = region.data
            if (max_x, max_y) != (len(x), len(y)):
                data, x, y = downsample_2d(data, x, y, max_points_x=max_x, max_points_y=max_y)
            draw_positive_contours(ax, x, y, data, levels,
                                   colour_spectra if len(spectra) == 1 else None,
                                   default_colour=colour, linewidths=contour_linewidth)
            if negative is not None and len(spectra) == 1:
                ax.contour(x, y, data, levels=sorted(negative), colors=NEGATIVE_CONTOUR_COLOUR,
                           linewidths=contour_linewidth)
        ax.set_xlim(max(spec["xlims"]), min(spec["xlims"]))       # high ppm on the left
        lo, hi = sorted(spec["ylims"])
        if ref.is_pseudo_2d:
            ax.set_ylim(lo, hi)
            ax.set_yticks([])
        else:
            ax.set_ylim(hi, lo)                                     # as the parent: high ppm at the bottom
        ax.tick_params(labelsize=0.75 * fontsize, direction="in", length=3, pad=2)
        # F1 labels on the side facing into the plot, so an inset in a
        # right-hand corner does not print over the parent's own
        # right-hand F1 labels.
        if bounds[0] + bounds[2] / 2 > 0.5:
            ax.yaxis.tick_left()
        else:
            ax.yaxis.tick_right()
        # Likewise F2 labels on top for an inset in a lower corner, clear
        # of the parent's own F2 labels.
        if bounds[1] + bounds[3] / 2 < 0.5:
            ax.xaxis.tick_top()
        ax.set_facecolor(parent_ax.get_facecolor())
        if spec.get("mark_inset", True):
            parent_ax.indicate_inset_zoom(ax, edgecolor=spec.get("edgecolor", "0.35"), alpha=0.9)
        created.append(ax)
    return created


#: What a projection strip's style may set -- see
#: :func:`projection_line_kwargs`.
PROJECTION_STYLE_KEYS = ("color", "linewidth", "linestyle", "size", "visible")


def projection_line_kwargs(projection_style, dimension: str) -> dict:
    """One strip's entry of a ``projection_style`` dict, checked: the
    keys :data:`PROJECTION_STYLE_KEYS` allows, ``None`` values dropped
    (``None`` = "leave the default"). Shared by
    :func:`plot_2d_contour` and the notebook's ``NMR2Dplotter`` so the
    two accept exactly the same thing.

    A misspelt key is an error rather than silently ignored -- a
    ``"colour"`` that did nothing would look like a bug in the plot."""
    if not projection_style:
        return {}
    style = projection_style.get(dimension) or {}
    unknown = set(style) - set(PROJECTION_STYLE_KEYS)
    if unknown:
        raise ValueError(
            f"Unknown projection_style key(s) for {dimension}: {sorted(unknown)}; "
            f"allowed: {list(PROJECTION_STYLE_KEYS)}")
    return {k: v for k, v in style.items() if v is not None}


def _external_projection_trace(source, limits):
    """``(x, intensity)`` of an external projection, cropped to the
    contour's displayed ``limits`` (either order) -- see
    :func:`plot_2d_contour`'s ``f1_external``/``f2_external``.

    Cropped so that a peak outside the contour's range (a solvent line,
    a spinning sideband) cannot set the strip's intensity scale."""
    if isinstance(source, NMRSpectrum):
        if source.ndim != 1:
            raise ValueError("An external projection must be a 1D spectrum.")
        x, y = source.axes[0].ppm, source.data
    elif isinstance(source, dict):
        x, y = source["x_ppm"], source["real_spectrum"]
    else:
        x, y = source
    x = np.asarray(x, dtype=np.float64)
    y = np.real(np.asarray(y)).astype(np.float64)
    lo, hi = min(limits), max(limits)
    mask = (x >= lo) & (x <= hi)
    if np.count_nonzero(mask) < 2:
        raise ValueError(
            f"The external projection ({x.min():.4g} to {x.max():.4g}) does not "
            f"overlap the contour's displayed range ({lo:.4g} to {hi:.4g}).")
    return x[mask], y[mask]


def _external_projection_limits(y, proj_scale=1.0):
    """``(bottom, top)`` intensity limits for an external projection:
    its own baseline at the bottom (2% headroom below the lowest point)
    and its tallest peak at ``proj_scale`` of the strip. Written out
    rather than copying the computed projection's ``min*0.9``, which
    puts the lower limit *above* the data whenever the trace dips below
    zero -- the usual case for a real, noisy 1D."""
    ymin, ymax = float(np.min(y)), float(np.max(y))
    span = (ymax - ymin) or 1.0
    return ymin - 0.02 * span, ymin + span * proj_scale


def plot_2d_contour(spectrum: NMRSpectrum, contour_fraction_x=0.25, contour_fraction_y=0.25,
                     ncontourpts_x=16384, ncontourpts_y=1024,
                     pl_height=7.0, pl_width=7.0, factor=0.07, clevels=7,
                     level_mode="power", power=1.25,
                     proj_scale=1.0, projection_mode="auto", show_projections=True,
                     projection_size=1.0, projection_pad=0.1, square_aspect=True,
                     field='', xlabel=None, ylabel=None, xlims=None, ylims=None,
                     font='Arial', fontsize=12, output_format='',
                     colour_spectra=None, equal_tickspacing=False, tick_number=None,
                     tick_dir='in', tick_length=4, tick_width=1.0, tick_pad=4,
                     show_nQ_line=False, nQ=1, nQ_colors=None,
                     slices=None, slice_panel_size=1.0, slice_panel_pad=0.95,
                     slice_offset_fraction=0.85, slice_colors=None, slice_linewidth=2.0,
                     slice_bottom_gap=0.05,
                     contour_linewidth: float = 0.8, insets=None,
                     f1_external=None, f2_external=None,
                     external_projection_color='crimson', projection_style=None,
                     ax=None, fig=None, save_fig=False, fig_name='nmr_2d_spectrum',
                     save_format='png'):
    """Plot a 2D NMR spectrum as a contour plot with F1/F2 projections.

    ``f1_external``/``f2_external`` : a separately acquired 1D spectrum
    to draw *instead of* the computed F1 (left)/F2 (top) projection --
    a 1D :class:`NMRSpectrum`, an ``NMR1dimport`` dict, or an
    ``(x, intensity)`` pair already in the contour's displayed unit.
    Cropped to the contour's displayed range, scaled on its own, and
    drawn in ``external_projection_color`` so it cannot be mistaken
    for the computed projection. Nothing here checks the nucleus: that
    is :func:`opennmr.notebook.check_external_projection`'s job, run by
    whoever chose the 1D. ``f1_external`` is refused for a pseudo-2D
    (its F1 is a slice number, not a shift). The same feature as the
    notebook's ``NMR2Dplotter(f1_external_projection=...)``.

    ``projection_style`` : per-strip appearance, ``{"f1": {...}, "f2":
    {...}}`` with any of ``color``, ``linewidth``, ``linestyle``,
    ``size`` (the strip's depth in inches, replacing
    ``projection_size`` for that strip) and ``visible`` (``False``
    hides that strip, as ``show_projections`` would). Anything not
    given keeps its default -- black for a computed projection,
    ``external_projection_color`` for an external one. See
    :func:`projection_line_kwargs`.

    ``insets`` : zoomed insets, see :func:`draw_insets` (drawn with this
    plot's own contour settings unless an inset overrides ``factor``).

    ``projection_mode`` : ``'sum'`` for conventional projections;
    ``'max'`` (of ``abs()``) for pseudo-2D relaxation/diffusion series,
    where a sum projection would wash out slice-to-slice changes.
    ``'auto'`` (default) picks ``'max'`` for pseudo-2D, ``'sum'``
    otherwise.

    ``show_projections`` : ``True`` (both), ``False`` (neither, e.g.
    embedding a compact panel), or ``'f1'``/``'f2'`` (one only).

    ``slices`` : extracted slices to show beside the contour -- one
    dict per slice, ``{"axis", "index", "label", "color"}``, drawn as a
    dashed line across the contour at that slice's own position plus
    its extracted trace, stacked in a panel to the right and joined to
    its line. See :func:`_draw_slice_panel` for the full behaviour and
    the remaining ``slice_*`` parameters.

    ``square_aspect`` : ``True`` (default) makes the main contour panel
    itself square, via the same ``AxesDivider`` that lays out the F1/F2
    projections (``divider.set_aspect(True)``) rather than
    ``ax.set_box_aspect`` -- the divider coordinates *all* of its cells
    (main panel + any appended projections) together when resizing for
    the aspect constraint, so the projections stay correctly sized/
    positioned against the now-square panel; ``ax.set_box_aspect``
    shrinks only the main axes in isolation, leaving sibling axes
    created via ``divider.append_axes`` stranded at their pre-shrink
    position/size (confirmed directly -- a real, visible misalignment,
    not a theoretical concern). Set ``False`` to leave the panel's
    aspect fully auto (e.g. a caller that will apply its own
    ``ax.set_box_aspect``/``set_aspect`` afterward -- doing so together
    with ``square_aspect=True`` is redundant, not additive, and doing
    so with ``square_aspect=True`` when projections are also shown
    reintroduces the exact misalignment this parameter exists to
    avoid).

    ``clevels``/``level_mode``/``power`` : see :func:`_prepare_contour`.

    ``ax`` : if given, plot into this axis instead of creating a new
    figure (for embedding, e.g. in a GUI canvas, or in
    :func:`multi_plot`) -- figure-level options (``pl_height``,
    ``pl_width``, ``font``, ``fontsize``, ``output_format``,
    ``save_fig``) are then ignored, since they'd affect the whole
    (shared) figure.

    Returns ``fig, ax, proj_axes`` -- ``proj_axes`` is
    ``{"f1": Axes | None, "f2": Axes | None}``, the F1 (left)/F2 (top)
    projection panels' own axes when drawn (``None`` for whichever
    ``show_projections`` excluded), so a caller can address their
    intensity scale independently of the main contour (e.g. its own
    scroll-to-zoom).
    """
    data = spectrum.data
    x_ppm = spectrum.axes[1].ppm
    pseudo2d = spectrum.is_pseudo_2d
    homonuclear = spectrum.metadata.homonuclear

    contour_x, contour_y, contour_data, thresvec, y_axis, neg_thresvec = _prepare_contour(
        spectrum, contour_fraction_x=contour_fraction_x, contour_fraction_y=contour_fraction_y,
        ncontourpts_x=ncontourpts_x, ncontourpts_y=ncontourpts_y, factor=factor, clevels=clevels,
        level_mode=level_mode, power=power,
    )

    nuc1 = find_gamma(spectrum.metadata.nucleus, field=field, SFO1=spectrum.metadata.sfo1)['Nuclide']
    labeltextx = NUC_label(nuc1, x_unit='ppm')

    if homonuclear:
        labeltexty = NUC_label(nuc1, x_unit='ppm')
    elif not pseudo2d:
        labeltexty = NUC_label(spectrum.metadata.nucleus_indirect, x_unit='ppm')
    else:
        labeltexty = 'Slice number'

    if ylabel is not None:
        labeltexty = ylabel
    if xlabel is not None:
        labeltextx = xlabel

    standalone = ax is None
    if standalone:
        style = setup_plotting(font=font, fontsize=fontsize, pl_width=pl_width,
                                pl_height=pl_height, output_format=output_format)
        pl_width, pl_height = style['pl_width'], style['pl_height']
        fig, ax = plt.subplots()
    elif fig is None:
        fig = ax.figure

    draw_positive_contours(ax, contour_x, contour_y, contour_data, thresvec, colour_spectra,
                           default_colour='black', linewidths=contour_linewidth)

    if neg_thresvec is not None:
        # Negative-going signal (dispersive phase error, an artifact,
        # or genuinely negative-intensity data) drawn in a contrasting
        # colour rather than invisible or lost among the positive
        # levels -- real user request, and a real practical need: it's
        # what makes an incoherent (no single correct phase) artifact
        # visually distinguishable from the genuine, phaseable peak
        # next to it.
        # Teal rather than red: red reads as an error/alarm colour and
        # dominated the black positive contours visually even at equal
        # intensity. Teal stays clearly distinguishable from black
        # (and remains so for the common red-green colour-vision
        # deficiencies, which red-vs-black does not) without
        # shouting. Real user request: "maybe also the colour needs to
        # be softened to something less stark like a teal".
        ax.contour(contour_x, contour_y, contour_data,
                   colors=NEGATIVE_CONTOUR_COLOUR, levels=sorted(neg_thresvec),
                   linewidths=contour_linewidth)

    # A plain make_axes_locatable(ax) divider's default reference sizes
    # (Size.AxesX/AxesY) are the axes' own *data-coordinate range*, not
    # its physical size -- fine for divider.set_aspect(False) (this
    # function's own non-square path, where the reference size choice
    # doesn't affect the result either way), but wrong for
    # square_aspect=True: equalising those would make 1 ppm on F2 equal
    # 1 ppm on F1 *physically* (data-aspect-locked), not make the panel
    # itself a square box -- and F1/F2 ppm spans are essentially never
    # numerically equal for real NMR data (confirmed directly: a real
    # HSQC's F2/F1 spans differed by ~10x, producing a panel roughly
    # 1 inch by 10 inches under the plain make_axes_locatable default,
    # not square at all, despite divider.set_aspect(True) being set).
    # Size.Scaled(1) is a plain, data-independent relative unit -- using
    # it for both directions makes divider.set_aspect(True) equalise
    # genuine physical box dimensions instead.
    divider = AxesDivider(ax, xref=axes_size.Scaled(1), yref=axes_size.Scaled(1))
    ax.set_axes_locator(divider.new_locator(nx=0, ny=0))
    divider.set_aspect(square_aspect)

    ax.set_xlabel(_italic_label(labeltextx), fontsize=fontsize)
    if pseudo2d:
        ax.set_yticks([])
    else:
        ax.set_ylabel(_italic_label(labeltexty), fontsize=fontsize)

    ax.tick_params(
        axis='both', which='both', direction=tick_dir, labelsize=fontsize,
        labelbottom=True, labelleft=not pseudo2d, bottom=True, left=True,
        length=tick_length, width=tick_width, pad=tick_pad,
    )
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()

    if xlims is not None:
        ax.set_xlim(min(xlims), max(xlims))
    if ylims is not None:
        ax.set_ylim(min(ylims), max(ylims))

    ax.invert_xaxis()

    if pseudo2d:
        if ylims is None:
            ax.set_ylim(y_axis.min() - 0.5, y_axis.max() + 0.5)
    else:
        # **Both frequency axes run the same way: values increase in
        # the negative screen direction.** F2 increases right-to-left
        # (`invert_xaxis` above, the universal NMR convention), and F1
        # increases top-to-bottom for exactly the same reason -- so
        # high ppm sits at the left *and* at the bottom, and the
        # nQ = 1 line through a homonuclear spectrum's
        # self-correlations runs bottom-left to top-right.
        #
        # This was briefly removed, on the mistaken reading that the
        # two axes should both increase toward the top-left. Restored
        # on the user's own statement of the convention: "the vertical
        # axis the wrong way round (low to high, bottom to top)
        # compared to the horizontal axis (NMR convention low to high,
        # right to left)". Either orientation puts the data at the same
        # *coordinates* -- the nQ line is drawn in data space and
        # passes through the self-correlations either way -- so nothing
        # but the axis direction itself distinguishes them, which is
        # why this has to follow the convention rather than an argument
        # about geometry.
        ax.invert_yaxis()

    if equal_tickspacing:
        x_range = abs(np.diff(ax.get_xlim())[0])
        y_range = abs(np.diff(ax.get_ylim())[0])
        max_range = max(x_range, y_range)
        locator = MaxNLocator(nbins=tick_number if tick_number is not None else 6)
        tick_values = locator.tick_values(0, max_range)
        tick_spacing = np.min(np.diff(tick_values))
        ax.xaxis.set_major_locator(MultipleLocator(tick_spacing))
        ax.yaxis.set_major_locator(MultipleLocator(tick_spacing))
    elif tick_number is not None:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=tick_number))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=tick_number))

    ax.tick_params(direction=tick_dir, labelsize=fontsize, length=tick_length,
                    width=tick_width, pad=tick_pad)
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()

    if show_nQ_line:
        nQ_list = nQ if isinstance(nQ, (list, tuple, np.ndarray)) else [nQ]
        nQ_list = [n for n in nQ_list if n != 0]
        if nQ_colors is None:
            nQ_colors = ['gray'] * len(nQ_list)
        elif len(nQ_colors) != len(nQ_list):
            raise ValueError("nQ_colors must have the same length as nQ.")

        xmin, xmax = ax.get_xlim()
        x_line = np.linspace(xmin, xmax, 500)
        for n, colour in zip(nQ_list, nQ_colors):
            ax.plot(x_line, n * x_line, linestyle='--', linewidth=1.5,
                    color=colour, alpha=0.8, zorder=10)

    if show_projections not in (True, False, 'f1', 'f2'):
        raise ValueError("show_projections must be True, False, 'f1', or 'f2'")
    f2_style = projection_line_kwargs(projection_style, "f2")
    f1_style = projection_line_kwargs(projection_style, "f1")
    show_f2_projection = show_projections in (True, 'f2') and f2_style.pop("visible", True)
    show_f1_projection = show_projections in (True, 'f1') and f1_style.pop("visible", True)
    f2_size = f2_style.pop("size", projection_size)
    f1_size = f1_style.pop("size", projection_size)

    proj_axes = {"f1": None, "f2": None}
    if f1_external is not None and pseudo2d:
        raise ValueError(
            "f1_external is not supported for pseudo-2D data (the F1 axis is a "
            "slice/experiment number, not a chemical shift).")

    if show_f2_projection or show_f1_projection:
        spec_float = np.asarray(data, dtype=np.float64)
        mode = projection_mode
        if mode == "auto":
            # "max" for *every* 2D now, not just pseudo-2D. A sum
            # projection of a phase-sensitive spectrum cancels its own
            # positive and negative lobes against each other and washes
            # the peaks out -- real user request: "the 2D projections
            # should probably just be the positive projections (not the
            # sum)". "sum" is still available explicitly.
            mode = "max"
        elif mode not in ("sum", "max"):
            raise ValueError("projection_mode must be 'auto', 'sum', or 'max'")

        if show_f2_projection:
            # projection_size halved from 2.0 on explicit request ("for
            # 2Ds, I'd rather the projection plots were about half their
            # current heights so that the contour plot can be generally
            # bigger") -- the divider gives the contour whatever the
            # projections don't take, so this is the one knob that
            # trades between them.
            ax_f2 = divider.append_axes("top", f2_size, pad=projection_pad, sharex=ax)
            ax_f2.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False, labeltop=False)
            ax_f2.tick_params(axis='y', which='both', left=False, right=False, labelleft=False, labelright=False)
            if f2_external is not None:
                ext_x, ext_y = _external_projection_trace(f2_external, ax.get_xlim())
                ax_f2.plot(ext_x, ext_y, gid="external_projection_f2",
                           **{"color": external_projection_color, **f2_style})
                ax_f2.set_ylim(_external_projection_limits(ext_y, proj_scale))
            else:
                f2_projection = np.sum(spec_float, axis=0) if mode == "sum" else np.max(spec_float, axis=0)
                ax_f2.plot(x_ppm, f2_projection + (0.02 * np.min(f2_projection)),
                           **{"color": "k", **f2_style})
                ax_f2.set_ylim([np.min(f2_projection) * 0.9, np.max(f2_projection) * proj_scale])
            proj_axes["f2"] = ax_f2

        if show_f1_projection:
            ax_f1 = divider.append_axes("left", f1_size, pad=projection_pad, sharey=ax)
            ax_f1.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False, labeltop=False)
            ax_f1.tick_params(axis='y', which='both', left=False, right=False, labelleft=False, labelright=False)
            f1_projection = np.sum(spec_float, axis=1) if mode == "sum" else np.max(spec_float, axis=1)
            # Real bug, user-reported ("the phase of the contours and
            # the projections are definitely inverted... it's more about
            # the matching of the contour phase to the projection
            # phase") and confirmed with a controlled synthetic fixture
            # carrying one known-positive and one known-negative peak:
            # this used to plot *negated* data (`-f1_projection`) while
            # the F2 projection above plots its own un-negated, so a
            # positive (black-contour) peak appeared as a
            # negative-going trace here and a negative (red-contour) one
            # as positive -- the two projections disagreed with each
            # other and the F1 one disagreed with the contour it sits
            # beside.
            #
            # The negation was doing real layout work, though, and that
            # has to be preserved: this panel sits to the *left* of the
            # contour with intensity along x, and negating made positive
            # peaks point leftward, away from the contour, rather than
            # into it. Fixed by reversing the *axis* instead of the
            # data -- plot the true values, then set a descending xlim,
            # which points the peaks the same way while leaving the
            # plotted numbers honest. The scroll-to-zoom handler
            # (`MplCanvas._zoom_projection_intensity`) still works
            # unchanged: it pins whichever end `set_xlim`'s second
            # element is, which is still the baseline end here.
            if f1_external is not None:
                # Same orientation as the computed one below it: true
                # values, descending xlim, peaks pointing away from the
                # contour.
                ext_x, ext_y = _external_projection_trace(f1_external, ax.get_ylim())
                ax_f1.plot(ext_y, ext_x, gid="external_projection_f1",
                           **{"color": external_projection_color, **f1_style})
                bottom, top = _external_projection_limits(ext_y, proj_scale)
                ax_f1.set_xlim([top, bottom])
            else:
                ax_f1.plot(f1_projection + (0.02 * np.min(f1_projection)), y_axis,
                           **{"color": "k", **f1_style})
                ax_f1.set_xlim([np.max(f1_projection) * proj_scale, np.min(f1_projection) * 0.99])
            proj_axes["f1"] = ax_f1

        for proj_ax in proj_axes.values():
            if proj_ax is None:
                continue
            for spine in ('top', 'right', 'bottom', 'left'):
                proj_ax.spines[spine].set_visible(False)

    # After the projections, so the panel is appended outside them and
    # the divider lays all the cells out together (see the AxesDivider
    # comment above for why its reference sizes matter here).
    if slices:
        proj_axes["slices"] = _draw_slice_panel(
            ax, divider, spectrum, slices,
            size=slice_panel_size, pad=slice_panel_pad,
            offset_fraction=slice_offset_fraction, colours=slice_colors,
            font=font, fontsize=fontsize, linewidth=slice_linewidth,
            bottom_gap=slice_bottom_gap,
            # The trace runs along the *other* dimension from the one
            # its slice fixes, so a row's panel carries the contour's
            # own x label and a column's its y label. User: "the axis
            # needs a matching axis label".
            axis_label=(labeltextx if slices[0].get("axis", "f1") == "f1" else labeltexty),
        )

    if insets:
        proj_axes["insets"] = draw_insets(
            ax, spectrum, insets, factor=factor, clevels=clevels, level_mode=level_mode,
            power=power, contour_linewidth=contour_linewidth, fontsize=fontsize,
            colour_spectra=colour_spectra)

    if standalone:
        fig.set_figheight(pl_height)
        fig.set_figwidth(pl_width)
        fig.subplots_adjust(left=0.15, right=0.88, bottom=0.12, top=0.88)
        if save_fig:
            save_figure(fig, fig_name=fig_name, save_format=save_format)

    return fig, ax, proj_axes


def multi_plot(spectra: list[NMRSpectrum], layout="row", panel_kwargs=None, titles=None,
               panel_width=5.0, panel_height=5.0, pl_width=None, pl_height=None,
               font='Arial', fontsize=12, output_format='',
               save_fig=False, fig_name='nmr_2d_multi', save_format='png',
               fig=None, **shared_kwargs):
    """Plot multiple 2D contour panels (:func:`plot_2d_contour`) on one
    figure, arranged in a row, column, or grid -- one panel per
    spectrum. Ported from ``nmrtools2``'s ``NMR2D_multi_plot``.

    ``panel_kwargs`` : per-panel keyword overrides forwarded to
    :func:`plot_2d_contour`, one dict per panel (``{}`` for none),
    taking precedence over the same key in ``**shared_kwargs``.

    Figure-level :func:`plot_2d_contour` arguments (``ax``, ``fig``,
    ``save_fig``, ``fig_name``, ``save_format``, ``pl_width``,
    ``pl_height``, ``font``, ``fontsize``, ``output_format``) are
    owned by this function at the figure level -- passing them in
    ``**shared_kwargs`` or ``panel_kwargs`` raises ``ValueError``.

    ``fig`` : draw into this (already cleared) figure instead of a new
    one -- for a GUI canvas. Figure size and ``setup_plotting`` are then
    left alone, since they would change the whole application's state.
    Per-panel ``insets`` go in ``panel_kwargs`` (see :func:`draw_insets`).

    Returns ``fig, axes`` (one ``Axes`` per panel, in ``spectra`` order).
    """
    if layout not in ("row", "column", "grid"):
        raise ValueError("layout must be 'row', 'column', or 'grid'")

    n = len(spectra)
    if n == 0:
        raise ValueError("spectra must contain at least one spectrum.")

    if panel_kwargs is not None and len(panel_kwargs) != n:
        raise ValueError("panel_kwargs must have the same length as spectra.")
    panel_kwargs = panel_kwargs if panel_kwargs is not None else [{}] * n

    if titles is not None and len(titles) != n:
        raise ValueError("titles must have the same length as spectra.")

    figure_level_keys = {
        "ax", "fig", "save_fig", "fig_name", "save_format",
        "pl_width", "pl_height", "font", "fontsize", "output_format",
    }
    bad = figure_level_keys & set(shared_kwargs)
    if bad:
        raise ValueError(
            f"Figure-level keys are not valid as shared per-panel kwargs: {sorted(bad)}; "
            f"use multi_plot's own same-named arguments instead."
        )
    for i, kw in enumerate(panel_kwargs):
        bad = figure_level_keys & set(kw)
        if bad:
            raise ValueError(f"panel_kwargs[{i}] contains figure-level keys not valid per panel: {sorted(bad)}")

    embedded = fig is not None
    if not embedded:
        setup_plotting(font=font, fontsize=fontsize, output_format=output_format)

    if layout == "row":
        nrows, ncols = 1, n
    elif layout == "column":
        nrows, ncols = n, 1
    else:
        ncols = int(np.ceil(np.sqrt(n)))
        nrows = int(np.ceil(n / ncols))

    if pl_width is None:
        pl_width = panel_width * ncols
    if pl_height is None:
        pl_height = panel_height * nrows

    if not embedded:
        fig = plt.figure(figsize=(pl_width, pl_height))
    gs = fig.add_gridspec(nrows, ncols)

    axes = []
    for i, spectrum in enumerate(spectra):
        r, c = divmod(i, ncols)
        ax = fig.add_subplot(gs[r, c])
        kw = dict(shared_kwargs)
        kw.update(panel_kwargs[i])
        plot_2d_contour(spectrum, ax=ax, fig=fig, fontsize=fontsize, **kw)
        if titles is not None:
            ax.set_title(titles[i], fontsize=fontsize)
        axes.append(ax)

    if not embedded:
        fig.tight_layout()
    if save_fig:
        save_figure(fig, fig_name=fig_name, save_format=save_format)

    return fig, axes


def _check_overlay_compatible(spectra: list[NMRSpectrum]):
    """Raise ``ValueError`` unless every spectrum shares the same
    F1/F2 nuclides and 2D experiment type (conventional vs pseudo-2D,
    homo- vs heteronuclear) -- i.e. their axes actually mean the same
    thing and can be overlaid directly. Deliberately does not compare
    SFO1 (transmitter frequency): ppm is field-independent, so the
    same experiment at two fields is still a valid overlay."""
    ref = spectra[0]
    for i, s in enumerate(spectra[1:], start=1):
        if (s.metadata.nucleus, s.metadata.nucleus_indirect) != (ref.metadata.nucleus, ref.metadata.nucleus_indirect):
            raise ValueError(
                f"Cannot overlay spectra with different F1/F2 nuclides: "
                f"spectrum 0 is {ref.metadata.nucleus}/{ref.metadata.nucleus_indirect}, "
                f"spectrum {i} is {s.metadata.nucleus}/{s.metadata.nucleus_indirect}."
            )
        if s.is_pseudo_2d != ref.is_pseudo_2d:
            raise ValueError(f"Cannot overlay a pseudo-2D spectrum with a conventional 2D spectrum (spectrum {i} vs 0).")
        if s.metadata.homonuclear != ref.metadata.homonuclear:
            raise ValueError(f"Cannot overlay a homonuclear spectrum with a heteronuclear spectrum (spectrum {i} vs 0).")


def overlay_plot(spectra: list[NMRSpectrum], contour_fraction_x=0.25, contour_fraction_y=0.25,
                  ncontourpts_x=16384, ncontourpts_y=1024,
                  pl_height=7.0, pl_width=7.0, factor=0.07, clevels=7,
                  contour_linewidth: float = 0.8,
                  colours='diverging', labels=None, legend_kwargs=None,
                  field='', xlabel=None, ylabel=None, xlims=None, ylims=None,
                  font='Arial', fontsize=12, output_format='',
                  equal_tickspacing=False, tick_number=None,
                  tick_dir='in', tick_length=4, tick_width=1.0, tick_pad=4,
                  show_nQ_line=False, nQ=1, nQ_colors=None, insets=None,
                  ax=None, fig=None, save_fig=False, fig_name='nmr_2d_overlay',
                  save_format='png'):
    """Overlay 2-6 2D NMR spectra as line contours on one set of axes,
    each in its own flat colour (cf. :func:`multi_plot`'s side-by-side
    panels). Ported from ``nmrtools2``'s ``NMR2D_overlay_plot``.

    Checks every spectrum shares the same F1/F2 nuclides and
    experiment type first (:func:`_check_overlay_compatible`) and
    raises ``ValueError`` otherwise.

    ``colours`` : ``'diverging'`` (default, from
    :func:`opennmr.plotting.style.get_stack_colormap`) or an explicit
    list of matplotlib colours, one per spectrum.

    ``insets`` : zoomed insets of the overlay, see :func:`draw_insets`.

    Returns ``fig, ax``.
    """
    if not (2 <= len(spectra) <= MAX_OVERLAY_SPECTRA):
        raise ValueError(
            f"overlay_plot accepts 2 to {MAX_OVERLAY_SPECTRA} spectra, got {len(spectra)}.")

    _check_overlay_compatible(spectra)
    n = len(spectra)

    if colours == 'diverging':
        cmap = get_stack_colormap('diverging')
        resolved_colours = [cmap(i / max(n - 1, 1)) for i in range(n)]
    elif isinstance(colours, (list, tuple)):
        if len(colours) != n:
            raise ValueError("colours must have the same length as spectra.")
        resolved_colours = list(colours)
    else:
        raise ValueError("colours must be 'diverging' or a list of colours, one per spectrum.")

    if labels is not None and len(labels) != n:
        raise ValueError("labels must have the same length as spectra.")

    ref = spectra[0]
    pseudo2d = ref.is_pseudo_2d
    homonuclear = ref.metadata.homonuclear

    nuc1 = find_gamma(ref.metadata.nucleus, field=field, SFO1=ref.metadata.sfo1)['Nuclide']
    labeltextx = NUC_label(nuc1, x_unit='ppm')
    if homonuclear:
        labeltexty = NUC_label(nuc1, x_unit='ppm')
    elif not pseudo2d:
        labeltexty = NUC_label(ref.metadata.nucleus_indirect, x_unit='ppm')
    else:
        labeltexty = 'Slice number'
    if ylabel is not None:
        labeltexty = ylabel
    if xlabel is not None:
        labeltextx = xlabel

    standalone = ax is None
    if standalone:
        style = setup_plotting(font=font, fontsize=fontsize, pl_width=pl_width,
                                pl_height=pl_height, output_format=output_format)
        pl_width, pl_height = style['pl_width'], style['pl_height']
        fig, ax = plt.subplots()
    elif fig is None:
        fig = ax.figure

    for spectrum, colour, label in zip(spectra, resolved_colours, labels if labels is not None else [None] * n):
        contour_x, contour_y, contour_data, thresvec, _, _ = _prepare_contour(
            spectrum, contour_fraction_x=contour_fraction_x, contour_fraction_y=contour_fraction_y,
            ncontourpts_x=ncontourpts_x, ncontourpts_y=ncontourpts_y, factor=factor, clevels=clevels,
        )
        ax.contour(contour_x, contour_y, contour_data, colors=[colour], levels=thresvec,
                   linewidths=contour_linewidth)
        if label is not None:
            ax.plot([], [], color=colour, label=label)

    ax.set_xlabel(_italic_label(labeltextx), fontsize=fontsize)
    if pseudo2d:
        ax.set_yticks([])
    else:
        ax.set_ylabel(_italic_label(labeltexty), fontsize=fontsize)

    ax.tick_params(
        axis='both', which='both', direction=tick_dir, labelsize=fontsize,
        labelbottom=True, labelleft=not pseudo2d, bottom=True, left=True,
        length=tick_length, width=tick_width, pad=tick_pad,
    )
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()

    if xlims is not None:
        ax.set_xlim(min(xlims), max(xlims))
    if ylims is not None:
        ax.set_ylim(min(ylims), max(ylims))

    ax.invert_xaxis()

    if pseudo2d:
        if ylims is None:
            y_axis = ref.axes[0].index
            ax.set_ylim(y_axis.min() - 0.5, y_axis.max() + 0.5)
    else:
        # **Both frequency axes run the same way: values increase in
        # the negative screen direction.** F2 increases right-to-left
        # (`invert_xaxis` above, the universal NMR convention), and F1
        # increases top-to-bottom for exactly the same reason -- so
        # high ppm sits at the left *and* at the bottom, and the
        # nQ = 1 line through a homonuclear spectrum's
        # self-correlations runs bottom-left to top-right.
        #
        # This was briefly removed, on the mistaken reading that the
        # two axes should both increase toward the top-left. Restored
        # on the user's own statement of the convention: "the vertical
        # axis the wrong way round (low to high, bottom to top)
        # compared to the horizontal axis (NMR convention low to high,
        # right to left)". Either orientation puts the data at the same
        # *coordinates* -- the nQ line is drawn in data space and
        # passes through the self-correlations either way -- so nothing
        # but the axis direction itself distinguishes them, which is
        # why this has to follow the convention rather than an argument
        # about geometry.
        ax.invert_yaxis()

    if equal_tickspacing:
        x_range = abs(np.diff(ax.get_xlim())[0])
        y_range = abs(np.diff(ax.get_ylim())[0])
        max_range = max(x_range, y_range)
        locator = MaxNLocator(nbins=tick_number if tick_number is not None else 6)
        tick_values = locator.tick_values(0, max_range)
        tick_spacing = np.min(np.diff(tick_values))
        ax.xaxis.set_major_locator(MultipleLocator(tick_spacing))
        ax.yaxis.set_major_locator(MultipleLocator(tick_spacing))
    elif tick_number is not None:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=tick_number))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=tick_number))

    ax.tick_params(direction=tick_dir, labelsize=fontsize, length=tick_length, width=tick_width, pad=tick_pad)
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()

    if show_nQ_line:
        nQ_list = nQ if isinstance(nQ, (list, tuple, np.ndarray)) else [nQ]
        nQ_list = [n_ for n_ in nQ_list if n_ != 0]
        if nQ_colors is None:
            nQ_colors = ['gray'] * len(nQ_list)
        elif len(nQ_colors) != len(nQ_list):
            raise ValueError("nQ_colors must have the same length as nQ.")

        xmin, xmax = ax.get_xlim()
        x_line = np.linspace(xmin, xmax, 500)
        for nq, colour in zip(nQ_list, nQ_colors):
            ax.plot(x_line, nq * x_line, linestyle='--', linewidth=1.5, color=colour, alpha=0.8, zorder=10)

    if labels is not None:
        ax.legend(**(legend_kwargs or {}))

    if insets:
        draw_insets(ax, spectra, insets, colours=resolved_colours, factor=factor, clevels=clevels,
                    contour_linewidth=contour_linewidth, fontsize=fontsize)

    if standalone:
        fig.set_figheight(pl_height)
        fig.set_figwidth(pl_width)
        fig.subplots_adjust(left=0.15, right=0.88, bottom=0.12, top=0.88)
        if save_fig:
            save_figure(fig, fig_name=fig_name, save_format=save_format)

    return fig, ax


def _italic_label(text):
    """Quantity symbols italic (plotting.labels.italicise_symbols)."""
    from opennmr.plotting.labels import italicise_symbols
    return italicise_symbols(text)
