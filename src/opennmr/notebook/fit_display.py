"""
How a fit's components are drawn: shown, hidden or highlighted, filled or
not, and in which colours -- one function for the GUI's fit views and for
notebook figures, so the two agree.

    import matplotlib.pyplot as plt
    from opennmr.notebook.fit_display import draw_fit_components, component_style

    fig, ax = plt.subplots()
    draw_fit_components(ax, x, y, y_fit, components=[site1, site2, baseline],
                        labels=["AlO4", "AlO6"],
                        style=component_style(show="highlight", highlight=["AlO6"],
                                              fill=False,
                                              line_colours={"AlO6": "#0072B2"}))

``components`` follows the fitting modules' convention: one curve per
component, **the baseline last**. Each component is drawn from zero, as
the GUI always has, or -- ``on_baseline=True`` -- sitting on the
baseline (``component + baseline``), which is what it adds to the total
fit when the baseline is not flat.

Style keys (see :data:`DEFAULT_COMPONENT_STYLE`):

``show``
    ``"auto"`` (every component, except a lone one when the caller says
    it only duplicates the fit), ``"all"``, ``"none"``, or
    ``"highlight"`` -- the components named in ``highlight`` in colour,
    the others as faint grey outlines without a legend entry.
``fill`` / ``fill_alpha``
    Shade each component, or draw it as a line only.
``fill_colours`` / ``line_colours``
    ``{label: colour}`` overrides; anything not named takes the default
    qualitative palette (ordered tallest component first, as before).
``on_baseline``
    Draw each component on top of the baseline rather than from zero.
``show_baseline``, ``fit_colour``, ``fit_style``, ``fit_width``,
``data_colour``, ``data_width``, ``line_width``, ``dim_colour``
    The rest of the figure's lines.
"""

from __future__ import annotations

import copy

import numpy as np
from matplotlib.colors import to_rgba

DEFAULT_COMPONENT_STYLE = {
    "show": "auto",
    "highlight": [],
    "fill": True,
    "fill_alpha": 0.4,
    "fill_colours": {},
    "line_colours": {},
    "line_width": 1.0,
    "show_baseline": True,
    "on_baseline": False,
    "fit_colour": "red",
    "fit_style": "--",
    "fit_width": 2.0,
    "data_colour": "black",
    "data_width": 1.5,
    "dim_colour": "0.65",
}

SHOW_MODES = ("auto", "all", "none", "highlight")


def component_style(style: dict | None = None, **overrides) -> dict:
    """A complete, checked style: the defaults, then ``style``, then
    ``overrides``. Unknown keys raise, so a typo cannot be silently
    ignored in a figure."""
    out = copy.deepcopy(DEFAULT_COMPONENT_STYLE)
    for source in (style or {}), overrides:
        for key, value in source.items():
            if key not in out:
                raise KeyError(f"Unknown component-style key {key!r}; "
                               f"known: {', '.join(sorted(out))}.")
            out[key] = copy.deepcopy(value)
    if out["show"] not in SHOW_MODES:
        raise ValueError(f"show must be one of {SHOW_MODES}, got {out['show']!r}.")
    out["highlight"] = [str(h) for h in (out["highlight"] or [])]
    out["fill_alpha"] = float(min(max(float(out["fill_alpha"]), 0.0), 1.0))
    return out


def default_component_colours(curves, labels) -> dict:
    """``{label: rgb}`` -- the qualitative palette handed out tallest
    component first (so the dominant site always gets the first
    colour, whatever order the sites were entered in)."""
    from ..plotting import qualitative_colors
    palette = qualitative_colors(len(curves)) if len(curves) else []
    order = sorted(range(len(curves)), key=lambda i: float(np.max(curves[i])), reverse=True)
    return {labels[i]: tuple(palette[rank][:3]) for rank, i in enumerate(order)}


def visible_components(labels, style: dict, auto_hide_single: bool = False) -> list:
    """The labels drawn in colour under ``style``."""
    mode = style["show"]
    if mode == "none":
        return []
    if mode == "highlight":
        wanted = set(style["highlight"])
        return [label for label in labels if label in wanted]
    if mode == "auto" and auto_hide_single and len(labels) < 2:
        return []
    return list(labels)


def draw_fit_components(ax, x, y_data, y_fit, components=None, labels=None, style=None,
                        auto_hide_single: bool = False, data_label: str = "Data",
                        fit_label: str = "Fit (total)") -> dict:
    """Data, each component (per ``style``) and the total fit on ``ax``.

    ``components`` : per-component curves with the baseline **last**
    (``None`` to draw only data and fit). ``labels`` : one per
    component (the baseline excluded). ``auto_hide_single`` : under
    ``show="auto"``, skip a single component (it only repeats the fit).

    Returns ``{"data", "fit", "components": {label: artist},
    "dimmed": {label: artist}, "baseline"}`` for further styling.
    """
    style = component_style(style)
    x = np.asarray(x, dtype=float)
    out = {"components": {}, "dimmed": {}, "baseline": None}
    out["data"] = ax.plot(x, y_data, color=style["data_colour"], linewidth=style["data_width"],
                          label=data_label, zorder=0)[0]
    if components:
        curves = [np.asarray(c, dtype=float) for c in components[:-1]]
        baseline = np.asarray(components[-1], dtype=float)
        labels = list(labels) if labels else [f"Peak {i + 1}" for i in range(len(curves))]
        if len(labels) != len(curves):
            raise ValueError(f"{len(labels)} labels for {len(curves)} components.")
        labels = [str(label) for label in labels]
        defaults = default_component_colours(curves, labels)
        shown = set(visible_components(labels, style, auto_hide_single))
        # Tallest first, so a small component's shading lands on top.
        order = sorted(range(len(curves)), key=lambda i: float(np.max(curves[i])), reverse=True)
        for i in order:
            label = labels[i]
            base_curve = baseline if style["on_baseline"] else np.zeros_like(baseline)
            if label in shown:
                edge = to_rgba(style["line_colours"].get(label, defaults[label]))
                if style["fill"]:
                    face = to_rgba(style["fill_colours"].get(label, defaults[label]),
                                   alpha=style["fill_alpha"])
                    artist = ax.fill_between(x, base_curve, curves[i] + base_curve, color=face,
                                             edgecolor=edge, linewidth=style["line_width"],
                                             label=label, zorder=1)
                else:
                    artist = ax.plot(x, curves[i] + base_curve, color=edge,
                                     linewidth=style["line_width"] * 1.5, label=label,
                                     zorder=2)[0]
                out["components"][label] = artist
            elif style["show"] == "highlight":
                out["dimmed"][label] = ax.plot(
                    x, curves[i] + base_curve, color=style["dim_colour"],
                    linewidth=style["line_width"] * 0.8, alpha=0.8, label="_nolegend_",
                    zorder=1)[0]
        if style["show_baseline"] and style["show"] != "none":
            out["baseline"] = ax.plot(x, baseline, color="gray", linestyle=":", linewidth=1.0,
                                      label="Baseline", zorder=1)[0]
    out["fit"] = ax.plot(x, y_fit, color=style["fit_colour"], linestyle=style["fit_style"],
                         linewidth=style["fit_width"], label=fit_label, zorder=100)[0]
    return out
