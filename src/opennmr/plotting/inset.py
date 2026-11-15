"""Zoomed insets for any matplotlib plot: a 1D spectrum, overlaid or
stacked spectra, a fitted curve, a profile.

An inset is a second set of axes placed inside the first that shows the
same data over a smaller region, with optional lines joining it to the
region it magnifies. :func:`add_inset` copies the lines, filled areas,
points and images of the parent axes into the inset (so it needs no
knowledge of how the plot was made), then sets the inset's own limits.

For a 2D contour plot use :func:`opennmr.plotting.twod.draw_insets`
instead, which re-contours the data inside the region at full
resolution rather than copying lines that were thinned for display.

Everything about how an inset looks is a key of one dict (see
:data:`INSET_DEFAULTS`), so a GUI, a notebook call and a saved setting all
speak the same language::

    from opennmr.plotting.inset import add_inset
    add_inset(ax, xlim=(7.4, 7.1), ylim=(0, 0.2), corner="upper left",
              show_axes=False, connectors=True)
"""
from __future__ import annotations

import copy

import numpy as np

#: gid given to every artist an inset adds, so they can be found and
#: removed again (and are never themselves copied into another inset).
INSET_GID = "opennmr-inset"

#: Where an inset's lower-left corner goes for a named corner, given its
#: width ``w``, height ``h`` and margin ``m`` (axes fractions).
CORNERS = {
    "upper right": lambda w, h, m: (1 - w - m, 1 - h - m),
    "upper left": lambda w, h, m: (m, 1 - h - m),
    "lower left": lambda w, h, m: (m, m),
    "lower right": lambda w, h, m: (1 - w - m, m),
    "centre": lambda w, h, m: (0.5 - w / 2, 0.5 - h / 2),
}

#: Every key an inset spec understands, with its default.
INSET_DEFAULTS: dict = {
    # What is shown: limits in the parent's own data coordinates, in the
    # order the parent displays them (so an NMR axis stays reversed).
    "xlim": None,
    "ylim": None,
    # Where it sits: a named corner, or "custom" with x / y (lower-left,
    # axes fraction). width / height are fractions of the parent axes.
    "corner": "upper right",
    "x": 0.55,
    "y": 0.55,
    "width": 0.4,
    "height": 0.4,
    "margin": 0.03,
    # Appearance.
    "show_axes": True,          # ticks and tick labels; off = the peaks only. A line plot
                                # (1D spectrum, curve) never shows y-values; a 2D contour does.
    "show_xlabel": False,       # the x-axis label under the inset (off by default)
    "show_ylabel": False,       # the y-axis label beside it (off by default)
    "xlabel": None,             # the text; None copies the plot's own label
    "ylabel": None,
    "frame": True,              # the box around the inset
    "frame_colour": "0.2",
    "frame_width": 0.8,
    "connectors": True,         # the lines from the inset to the region it magnifies
    "connector_colour": "0.35",
    "connector_width": 0.8,
    "connector_style": "-",
    "family": None,             # typeface; None follows the plot
    "fontsize": None,           # tick-label size in points; None = 3/4 of the plot's
    "bold": False,
    "italic": False,
    "tick_direction": "in",     # "in", "out" or "inout"
    "tick_length": None,        # points; None = 3
    "tick_width": None,         # points; None = matplotlib's own
    "linewidth": None,          # width of the magnified data lines, points; None = the plot's
    "background": None,         # fill; None = the parent's
    "auto_ylim": False,         # fit the vertical range to the data in xlim
    "factor": None,             # 2D contour insets only: lowest contour, fraction of the maximum
    # Which plot's area the inset sits in, for a figure of several stacked
    # plots (Compare): None = the plot it magnifies; otherwise the index of
    # another plot, so one spectrum's inset can sit beside another's. The
    # position (corner / x, y) and size are then fractions of *that* plot.
    "host": None,
}

CORNER_NAMES = tuple(CORNERS)


def normalise(spec: dict | None = None, **overrides) -> dict:
    """A complete spec: :data:`INSET_DEFAULTS` overlaid with ``spec`` and
    ``overrides``. Raises ``ValueError`` for an unknown key or a size that
    cannot be drawn, rather than quietly ignoring a misspelling."""
    merged = dict(INSET_DEFAULTS)
    for source in (spec or {}, overrides):
        unknown = set(source) - set(INSET_DEFAULTS)
        if unknown:
            raise ValueError(
                f"Unknown inset option(s) {sorted(unknown)}; allowed: {sorted(INSET_DEFAULTS)}")
        merged.update(source)
    if merged["tick_direction"] not in ("in", "out", "inout"):
        raise ValueError(
            f"tick_direction must be 'in', 'out' or 'inout', not {merged['tick_direction']!r}")
    for key in ("tick_length", "tick_width", "linewidth"):
        if merged[key] is not None and float(merged[key]) < 0:
            raise ValueError(f"inset {key} cannot be negative")
    if merged["corner"] not in CORNERS and merged["corner"] != "custom":
        raise ValueError(
            f"corner must be one of {list(CORNERS) + ['custom']}, not {merged['corner']!r}")
    for key in ("width", "height"):
        if not 0.05 <= float(merged[key]) <= 0.95:
            raise ValueError(f"inset {key} must be between 0.05 and 0.95 of the plot")
    return merged


def inset_bounds(spec: dict) -> tuple[float, float, float, float]:
    """``(x0, y0, width, height)`` in parent-axes fractions."""
    w, h = float(spec["width"]), float(spec["height"])
    if spec["corner"] == "custom":
        x0, y0 = float(spec["x"]), float(spec["y"])
    else:
        x0, y0 = CORNERS[spec["corner"]](w, h, float(spec["margin"]))
    return (x0, y0, w, h)


def _copyable(artist) -> bool:
    return (artist.get_gid() != INSET_GID and not artist.get_animated()
            and artist.get_visible())


def _clone(artist, parent_ax, inset_ax):
    """A copy of ``artist`` that belongs to ``inset_ax``, with its data
    transforms pointed at the inset's coordinates."""
    new = copy.copy(artist)
    new._axes = None
    new._remove_method = None
    new.stale_callback = None
    if hasattr(new, "_parent_figure"):
        new._parent_figure = None
    for getter, setter in (("get_transform", "set_transform"),
                           ("get_offset_transform", "set_offset_transform")):
        if hasattr(new, getter) and hasattr(new, setter):
            try:
                if getattr(new, getter)() is parent_ax.transData:
                    getattr(new, setter)(inset_ax.transData)
            except Exception:  # noqa: BLE001 -- an artist without that transform keeps its own
                pass
    return new


def copy_artists(parent_ax, inset_ax) -> int:
    """Copy the lines, filled areas, points, patches and images of
    ``parent_ax`` into ``inset_ax``. Text is not copied: labels belong to
    the whole plot. Returns how many artists were copied."""
    from matplotlib.collections import Collection
    from matplotlib.image import AxesImage
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    count = 0
    ordered = sorted(
        (a for a in parent_ax.get_children()
         if isinstance(a, (Line2D, Collection, AxesImage, Patch)) and _copyable(a)
         and a is not parent_ax.patch),
        key=lambda a: a.get_zorder())
    for artist in ordered:
        try:
            new = _clone(artist, parent_ax, inset_ax)
            if isinstance(artist, Line2D):
                inset_ax.add_line(new)
            elif isinstance(artist, AxesImage):
                inset_ax.add_image(new)
            elif isinstance(artist, Collection):
                inset_ax.add_collection(new, autolim=False)
            else:
                inset_ax.add_patch(new)
            new.set_clip_path(inset_ax.patch)
            count += 1
        except Exception:  # noqa: BLE001 -- one artist that cannot be copied must not lose the rest
            continue
    return count


def _visible_y_range(parent_ax, xlim) -> tuple[float, float] | None:
    """The span of the parent's lines over ``xlim``, padded; ``None`` when
    no line has a point there."""
    lo, hi = sorted((float(xlim[0]), float(xlim[1])))
    low, high = np.inf, -np.inf
    for line in parent_ax.get_lines():
        if not _copyable(line):
            continue
        x = np.asarray(line.get_xdata(), dtype=float)
        y = np.asarray(line.get_ydata(), dtype=float)
        if x.size != y.size or x.size < 2:
            continue
        keep = (x >= lo) & (x <= hi) & np.isfinite(y)
        if keep.any():
            low, high = min(low, float(y[keep].min())), max(high, float(y[keep].max()))
    if not np.isfinite(low):
        return None
    pad = 0.05 * ((high - low) or abs(high) or 1.0)
    return low - pad, high + pad


def tick_kwargs(spec: dict) -> dict:
    """The ``tick_params`` keywords an inset spec's tick settings give:
    direction, length (3 pt unless set) and, when set, width."""
    kwargs = {"direction": spec.get("tick_direction") or "in",
              "length": float(spec["tick_length"]) if spec.get("tick_length") is not None else 3.0}
    if spec.get("tick_width") is not None:
        kwargs["width"] = float(spec["tick_width"])
    return kwargs


def set_data_linewidth(inset_ax, width) -> int:
    """Draw the lines copied into ``inset_ax`` at ``width`` points (``None``
    leaves them at the width they were copied with, the plot's). Returns how
    many lines were set."""
    if width is None:
        return 0
    count = 0
    for line in inset_ax.get_lines():
        line.set_linewidth(float(width))
        count += 1
    return count


def _text_size(parent_ax, spec: dict) -> float:
    """The inset's tick-label size: the spec's, else 3/4 of the plot's."""
    if spec.get("fontsize"):
        return float(spec["fontsize"])
    reference = parent_ax.xaxis.get_ticklabels() if parent_ax is not None else []
    base = float(reference[0].get_fontsize()) if reference else 10.0
    return 0.75 * base


def default_label(parent_ax, name: str = "x") -> str:
    """The text an inset's ``name`` ("x" or "y") axis label takes when the
    spec gives none: the plot's own label; else, on a figure of stacked
    plots (Compare), the label of the plots sharing that axis -- only the
    bottom row of a stack carries the x label -- and then any label on the
    figure; else "Intensity" for y, the quantity a spectrum's unlabelled
    vertical axis shows. Explicit user report, 2026-11-10: ticking "x-axis
    label" with no text of one's own drew nothing."""
    if parent_ax is None:
        return "Intensity" if name == "y" else ""
    getter = f"get_{name}label"
    text = getattr(parent_ax, getter)()
    if text:
        return text
    shared = (parent_ax.get_shared_x_axes() if name == "x"
              else parent_ax.get_shared_y_axes()).get_siblings(parent_ax)
    figure_axes = list(parent_ax.figure.axes) if parent_ax.figure is not None else []
    for other in [*shared, *figure_axes]:
        if other is parent_ax or other.get_gid() == INSET_GID:
            continue
        text = getattr(other, getter)()
        if text:
            return text
    return "Intensity" if name == "y" else ""


def _apply_labels(ax, parent_ax, spec: dict, bounds=None) -> None:
    """The x and y axis labels, when switched on: the spec's text, else the
    plot's own, in the inset's typeface and on the side facing into the plot
    (as the tick labels are). ``bounds`` is the inset's ``(x0, y0, w, h)``
    when it is not the spec's."""
    x0, y0, w, h = bounds if bounds is not None else inset_bounds(spec)
    ax.xaxis.set_label_position("top" if y0 + h / 2 < 0.5 else "bottom")
    ax.yaxis.set_label_position("left" if x0 + w / 2 > 0.5 else "right")
    size = _text_size(parent_ax, spec)
    for name, axis, setter in (("x", ax.xaxis, ax.set_xlabel), ("y", ax.yaxis, ax.set_ylabel)):
        if not spec.get(f"show_{name}label"):
            setter("")
            continue
        text = spec.get(f"{name}label")
        if text is None:
            text = default_label(parent_ax, name)
        setter(text or "")
        label = axis.get_label()
        label.set_fontsize(size)
        if spec.get("family"):
            label.set_fontfamily(spec["family"])
        label.set_fontweight("bold" if spec.get("bold") else "normal")
        label.set_fontstyle("italic" if spec.get("italic") else "normal")


def _style_axes(ax, parent_ax, spec: dict) -> None:
    ax.patch.set_facecolor(spec["background"] or parent_ax.get_facecolor())
    for side in ("left", "right", "top", "bottom"):
        ax.spines[side].set_visible(bool(spec["frame"]))
        ax.spines[side].set_edgecolor(spec["frame_colour"])
        ax.spines[side].set_linewidth(float(spec["frame_width"]))
    if not spec["show_axes"]:
        ax.set_xticks([])
        ax.set_yticks([])
        _apply_labels(ax, parent_ax, spec)
        return
    ax.tick_params(labelsize=_text_size(parent_ax, spec), pad=2, **tick_kwargs(spec))
    # Tick labels on the side facing into the plot, so an inset in a
    # corner does not print over the parent's own labels.
    x0, y0, w, h = inset_bounds(spec)
    if x0 + w / 2 > 0.5:
        ax.yaxis.tick_left()
    else:
        ax.yaxis.tick_right()
    if y0 + h / 2 < 0.5:
        ax.xaxis.tick_top()
    # A line plot has no y-values to read: the peaks' heights are
    # relative, and a curve's are in the plot itself. (A 2D contour's
    # F1 axis is a shift and keeps its numbers -- see restyle.)
    ax.set_yticks([])
    # Font properties act on the tick-label objects, which exist only
    # once the ticks have been placed.
    weight = "bold" if spec["bold"] else "normal"
    style = "italic" if spec["italic"] else "normal"
    for label in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        if spec["family"]:
            label.set_fontfamily(spec["family"])
        label.set_fontweight(weight)
        label.set_fontstyle(style)
    _apply_labels(ax, parent_ax, spec)


def _apply_fonts_late(ax, spec: dict) -> None:
    """Fonts again, once the limits (and so the tick labels) are final."""
    if not spec["show_axes"]:
        return
    weight = "bold" if spec["bold"] else "normal"
    style = "italic" if spec["italic"] else "normal"
    for axis in (ax.xaxis, ax.yaxis):
        for label in axis.get_ticklabels():
            if spec["family"]:
                label.set_fontfamily(spec["family"])
            label.set_fontweight(weight)
            label.set_fontstyle(style)


def _registry(parent_ax) -> list:
    if not hasattr(parent_ax, "_opennmr_insets"):
        parent_ax._opennmr_insets = []
    return parent_ax._opennmr_insets


def _indicator_artists(indicator):
    parts = []
    for name in ("rectangle", "connectors"):
        value = getattr(indicator, name, None)
        if value is None:
            continue
        parts.extend(value if isinstance(value, (list, tuple)) else [value])
    return [p for p in parts if p is not None and hasattr(p, "set_gid")]


def add_inset(parent_ax, xlim=None, ylim=None, spec: dict | None = None, host_ax=None,
              **options):
    """Add one inset to ``parent_ax`` and return its axes.

    ``xlim`` / ``ylim`` : the region magnified, in the parent's data
    coordinates, in display order (``(7.4, 7.1)`` keeps an NMR axis
    reversed). ``ylim=None`` fits the vertical range to the data in
    ``xlim``. Every other option is a key of :data:`INSET_DEFAULTS`.

    ``host_ax`` : another axes of the same figure to place the inset in
    (its corner / position and size are then fractions of ``host_ax``),
    so an inset of one stacked spectrum can sit beside another's. The
    copied data, the limits and the lines to the region still come from
    ``parent_ax``, and :func:`remove_insets` on ``parent_ax`` removes it.
    """
    spec = normalise(spec, **options)
    host = host_ax if host_ax is not None and host_ax is not parent_ax else parent_ax
    xlim = xlim if xlim is not None else spec["xlim"]
    ylim = ylim if ylim is not None else spec["ylim"]
    if xlim is None:
        raise ValueError("An inset needs xlim, the region to magnify.")
    if float(xlim[0]) == float(xlim[1]):
        raise ValueError("The inset region has no width.")
    inset = host.inset_axes(inset_bounds(spec))
    inset.set_gid(INSET_GID)
    inset.set_zorder(host.get_zorder() + 5)
    if not copy_artists(parent_ax, inset):
        inset.remove()
        raise ValueError("There is nothing on this plot to magnify.")
    set_data_linewidth(inset, spec["linewidth"])
    inset.set_xlim(*xlim)
    if ylim is None or spec["auto_ylim"]:
        fitted = _visible_y_range(parent_ax, xlim)
        if fitted is not None:
            descending = bool(ylim is not None and ylim[0] > ylim[1])
            ylim = fitted[::-1] if descending else fitted
    if ylim is None:
        ylim = parent_ax.get_ylim()
    inset.set_ylim(*ylim)
    _style_axes(inset, parent_ax, spec)
    _apply_fonts_late(inset, spec)
    record = {"axes": inset, "indicator": None, "spec": spec}
    if spec["connectors"]:
        indicator = parent_ax.indicate_inset_zoom(
            inset, edgecolor=spec["connector_colour"], alpha=0.9,
            linewidth=float(spec["connector_width"]), linestyle=spec["connector_style"])
        record["indicator"] = indicator
        for part in _indicator_artists(indicator):
            part.set_gid(INSET_GID)
    _registry(parent_ax).append(record)
    return inset


def insets_of(parent_ax) -> list:
    """The inset axes added to ``parent_ax`` by :func:`add_inset`."""
    return [r["axes"] for r in getattr(parent_ax, "_opennmr_insets", [])]


def remove_insets(parent_ax) -> int:
    """Remove every inset (and its connecting lines) from ``parent_ax``."""
    removed = 0
    for record in list(getattr(parent_ax, "_opennmr_insets", [])):
        indicator = record.get("indicator")
        if indicator is not None:
            try:
                indicator.remove()
            except Exception:  # noqa: BLE001 -- already gone with a cleared axes
                pass
        try:
            record["axes"].remove()
        except Exception:  # noqa: BLE001
            pass
        removed += 1
    parent_ax._opennmr_insets = []
    return removed


def draw_insets(parent_ax, specs, resolve_host=None) -> list:
    """Replace the insets on ``parent_ax`` with ``specs`` (a list of dicts
    as :func:`add_inset` takes) and return the new inset axes. A spec that
    cannot be drawn is skipped, so one bad region does not lose the others;
    ``ValueError`` is raised only when none could be drawn.

    ``resolve_host`` : ``index -> axes or None``, turning a spec's ``host``
    into the axes to place it in (see :data:`INSET_DEFAULTS`); without it,
    or for a host it cannot resolve, the inset sits on ``parent_ax``."""
    remove_insets(parent_ax)
    created, failures = [], []
    for spec in specs or []:
        host_ax = None
        host = (spec or {}).get("host")
        if host is not None and resolve_host is not None:
            try:
                host_ax = resolve_host(int(host))
            except (TypeError, ValueError, IndexError):
                host_ax = None
        try:
            created.append(add_inset(parent_ax, spec=spec, host_ax=host_ax))
        except ValueError as exc:
            failures.append(str(exc))
    if failures and not created:
        raise ValueError(failures[0])
    return created


def restyle(ax, spec: dict, parent_ax=None, bounds=None) -> None:
    """The appearance keys of ``spec`` (frame, axes on/off, typeface, tick
    size, axis labels) applied to an inset drawn elsewhere -- the 2D contour
    inset, whose contents :func:`opennmr.plotting.twod.draw_insets` draws
    itself. It keeps its y-values (F1 is a shift). ``parent_ax`` supplies
    the axis-label text when the spec has none; ``bounds`` is the inset's
    ``(x0, y0, w, h)`` in the parent."""
    for side in ("left", "right", "top", "bottom"):
        ax.spines[side].set_visible(bool(spec.get("frame", True)))
        ax.spines[side].set_edgecolor(spec.get("frame_colour", "0.2"))
        ax.spines[side].set_linewidth(float(spec.get("frame_width", 0.8)))
    if bounds is None:
        bounds = (0.55, 0.55, 0.4, 0.4)
    if not spec.get("show_axes", True):
        ax.set_xticks([])
        ax.set_yticks([])
        _apply_labels(ax, parent_ax, spec, bounds)
        return
    ax.tick_params(**tick_kwargs(spec))
    if spec.get("fontsize"):
        ax.tick_params(labelsize=float(spec["fontsize"]))
    weight = "bold" if spec.get("bold") else "normal"
    style = "italic" if spec.get("italic") else "normal"
    for axis in (ax.xaxis, ax.yaxis):
        for label in axis.get_ticklabels():
            if spec.get("family"):
                label.set_fontfamily(spec["family"])
            label.set_fontweight(weight)
            label.set_fontstyle(style)
    _apply_labels(ax, parent_ax, spec, bounds)
