"""Line widths and line styles for every line on a figure.

One mechanism for every kind of plot -- a spectrum, a Compare stack, a
spectral or curve fit and its components, a contour's projection
strips, a notebook figure shown in the GUI -- because it works on the
finished figure's ``Line2D`` artists rather than on any one plot's draw
code. The GUI's Plot Style ▸ Lines section and a notebook call the same
functions::

    from opennmr.plotting.lines import data_lines, apply_line_styles

    for info in data_lines(fig):          # what is there, and its key
        print(info["key"], info["name"], info["linewidth"], info["linestyle"])
    apply_line_styles(fig, scale=1.5,      # every line 1.5x as thick...
                      overrides={"label:Fit": {"linestyle": "--", "linewidth": 2}})

**Keys.** A line with a legend label is addressed as ``"label:<text>"``,
so a setting follows it across redraws and between plots of the same
kind (every spectral fit's "Fit" line, say). An unlabelled line is
addressed by position, ``"line:<axes>.<n>"`` (the n-th unlabelled line
of the axes-th Axes), which is stable for redraws of the same plot.

**Base values.** The first time a line is seen its own width and style
are recorded on it, so a scale or an override can always be undone
exactly, and repeated application does not compound.

Lines a user would not think of as plot lines are left alone:
``animated`` ones (the blitted crosshair), invisible ones, and any with
a ``gid`` starting ``"opennmr-ui"``.
"""

from __future__ import annotations

import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

__all__ = ["LINESTYLES", "linestyle_name", "normalise_linestyle", "data_lines",
           "apply_line_styles", "line_key"]

#: Line styles offered, (display name, matplotlib value). "None" keeps
#: only the markers (a scatter-style trace).
LINESTYLES = (
    ("Solid", "-"),
    ("Dashed", "--"),
    ("Dotted", ":"),
    ("Dash-dot", "-."),
    ("Long dash", (0, (8, 3))),
    ("Dash-dot-dot", (0, (6, 2, 1, 2, 1, 2))),
    ("Dense dots", (0, (1, 1))),
    ("Loose dots", (0, (1, 4))),
    ("None (markers only)", "None"),
)

_ALIASES = {"solid": "-", "dashed": "--", "dotted": ":", "dashdot": "-.", "dash-dot": "-.",
            "dot-dash": "-.", "none": "None", " ": "None", "": "None"}


def normalise_linestyle(style):
    """A linestyle as one of :data:`LINESTYLES`' values where it is one
    (``'dashed'`` -> ``'--'``; a JSON list ``[0, [8, 3]]`` -> the tuple),
    else unchanged. Raises ValueError for something matplotlib would not
    accept."""
    if isinstance(style, str):
        style = _ALIASES.get(style.lower(), style)
    elif isinstance(style, (list, tuple)) and len(style) == 2 and \
            isinstance(style[1], (list, tuple)):
        style = (float(style[0]) if style[0] else 0, tuple(float(v) for v in style[1]))
        style = (int(style[0]) if style[0] == int(style[0]) else style[0],
                 tuple(int(v) if v == int(v) else v for v in style[1]))
    try:
        Line2D([], [], linestyle=style)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Not a line style: {style!r}") from exc
    return style


def linestyle_name(style) -> str:
    """The :data:`LINESTYLES` name of ``style`` (or of the line's
    effective dash pattern), else 'Custom'."""
    try:
        style = normalise_linestyle(style)
    except ValueError:
        return "Custom"
    for name, value in LINESTYLES:
        if value == style:
            return name
    return "Custom"


def _is_ui_line(line) -> bool:
    gid = line.get_gid() or ""
    return bool(line.get_animated()) or not line.get_visible() or str(gid).startswith("opennmr-ui")


def _remember_base(line) -> None:
    if not hasattr(line, "_opennmr_base_lw"):
        line._opennmr_base_lw = float(line.get_linewidth())
        line._opennmr_base_ls = line.get_linestyle()
        # get_linestyle gives '-', '--', ':', '-.', 'None' for the named
        # styles; a custom dash comes back as '--' -- keep the dash
        # pattern itself so it is restored exactly.
        dashes = getattr(line, "_unscaled_dash_pattern", None)
        if line._opennmr_base_ls == "--" and dashes is not None and dashes[1] is not None:
            from matplotlib import rcParams
            default = tuple(rcParams["lines.dashed_pattern"])
            if tuple(dashes[1]) != default:
                line._opennmr_base_ls = (dashes[0], tuple(dashes[1]))


def line_key(label: str | None, axes_index: int, position: int) -> str:
    if label and not label.startswith("_"):
        return f"label:{label}"
    return f"line:{axes_index}.{position}"


def _axes_of(figure):
    """Every 2D Axes of ``figure`` in order (3D pictures have no lines a
    width setting means anything for)."""
    return [ax for ax in figure.get_axes() if getattr(ax, "name", "") != "3d"]


def data_lines(figure) -> list[dict]:
    """Every user-facing line on ``figure``: dicts with ``key``, ``name``
    (its label, or 'Line n (colour)'), ``line`` (the artist),
    ``linewidth``, ``linestyle``, ``base_linewidth``, ``base_linestyle``,
    ``colour`` (hex) and ``has_markers``. A label used by several lines
    (one legend entry drawn on several Axes) is listed once; its key
    reaches all of them."""
    out = []
    seen = {}
    n_unlabelled = 0
    for a, ax in enumerate(_axes_of(figure)):
        position = 0
        for line in ax.lines:
            if not isinstance(line, Line2D) or _is_ui_line(line):
                continue
            _remember_base(line)
            label = line.get_label()
            labelled = bool(label) and not label.startswith("_")
            key = line_key(label, a, position)
            if not labelled:
                position += 1
                n_unlabelled += 1
            if key in seen:
                seen[key]["lines"].append(line)
                continue
            try:
                colour = mcolors.to_hex(line.get_color())
            except (ValueError, TypeError):
                colour = "#000000"
            info = {
                "key": key,
                "name": label if labelled else f"Line {n_unlabelled}",
                "line": line, "lines": [line],
                "linewidth": float(line.get_linewidth()),
                "linestyle": line.get_linestyle(),
                "base_linewidth": line._opennmr_base_lw,
                "base_linestyle": line._opennmr_base_ls,
                "colour": colour,
                "has_markers": line.get_marker() not in (None, "None", "", " "),
                "axes_index": a,
            }
            seen[key] = info
            out.append(info)
    return out


def apply_line_styles(figure, scale: float = 1.0, overrides: dict | None = None,
                      legend_widths: bool = True) -> int:
    """Set every user-facing line's width to its own base width times
    ``scale``, then apply ``overrides`` (``{key: {"linewidth": w,
    "linestyle": s}}``, keys as :func:`data_lines` gives them; an absolute
    width wins over the scale). Lines without an override get their own
    base style back, so removing an override undoes it exactly.

    Legend entries follow their line's style (and width, unless
    ``legend_widths`` is False -- for a legend whose line width is set
    separately). Returns how many artists were changed; does not draw.
    """
    overrides = overrides or {}
    scale = float(scale) if scale else 1.0
    changed = 0
    by_label = {}
    for info in data_lines(figure):
        o = overrides.get(info["key"]) or {}
        width = (float(o["linewidth"]) if o.get("linewidth") is not None
                 else info["base_linewidth"] * scale)
        style = (normalise_linestyle(o["linestyle"]) if o.get("linestyle") is not None
                 else info["base_linestyle"])
        for line in info["lines"]:
            if abs(float(line.get_linewidth()) - width) > 1e-12:
                line.set_linewidth(width)
                changed += 1
            applied = getattr(line, "_opennmr_applied_ls", line._opennmr_base_ls)
            if applied != style:
                line.set_linestyle(style)
                line._opennmr_applied_ls = style
                changed += 1
        if info["key"].startswith("label:"):
            by_label[info["key"][6:]] = (width, style)
    legends = [ax.get_legend() for ax in _axes_of(figure)] + list(getattr(figure, "legends", []))
    for legend in legends:
        if legend is None:
            continue
        handles = list(getattr(legend, "legend_handles", None) or getattr(legend, "legendHandles", []))
        for handle, text in zip(handles, legend.get_texts()):
            target = by_label.get(text.get_text())
            if target is None or not isinstance(handle, Line2D):
                continue
            width, style = target
            if legend_widths and abs(float(handle.get_linewidth()) - width) > 1e-12:
                handle.set_linewidth(width)
                changed += 1
            if getattr(handle, "_opennmr_applied_ls", None) != style:
                handle.set_linestyle(style)
                handle._opennmr_applied_ls = style
                changed += 1
    return changed
