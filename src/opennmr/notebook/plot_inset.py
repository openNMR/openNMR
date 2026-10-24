"""Zoomed insets for notebook figures -- the same drawing the GUI's Inset
tool uses (:mod:`opennmr.plotting.inset`).

::

    import opennmr.notebook as nb
    fig, ax = plt.subplots()
    ax.plot(ppm, intensity)
    ax.invert_xaxis()
    nb.add_inset(ax, xlim=(7.4, 7.1), corner="upper left", auto_ylim=True,
                 show_axes=False, connectors=True, family="Arial", italic=True)

``xlim`` / ``ylim`` are in the plot's own data coordinates, in display
order. Every other option is a key of ``nb.INSET_DEFAULTS``: position
(``corner`` or ``corner="custom"`` with ``x``, ``y``), ``width``,
``height``, ``show_axes``, ``frame``, ``connectors`` and their colour,
width and style, ``family``, ``fontsize``, ``bold``, ``italic``.

A 1D inset never shows y-values (``show_axes`` gives its x axis only); a 2D
contour inset keeps both. The axis labels are off unless asked for:
``show_xlabel=True`` / ``show_ylabel=True`` (the text is the plot's own, or
``xlabel=`` / ``ylabel=``).

For a 2D contour plot use ``opennmr.plotting.draw_insets`` (it re-contours
the data in the region instead of copying lines).
"""
from __future__ import annotations

from ..plotting.inset import (
    CORNER_NAMES as INSET_CORNERS,
    INSET_DEFAULTS,
    add_inset as _add_inset,

    insets_of,
    remove_insets,
)

__all__ = ["add_inset", "remove_insets", "insets_of", "INSET_DEFAULTS",
           "INSET_CORNERS"]


def add_inset(target, xlim=None, ylim=None, spec: dict | None = None, **options):
    """Add a magnified inset of ``xlim`` x ``ylim`` to ``target`` (an Axes,
    or a Figure, whose first axes is used) and return the inset's axes.

    ``ylim=None`` scales the inset to the data inside ``xlim``."""
    axes = target
    if hasattr(target, "axes") and not hasattr(target, "inset_axes"):
        figure_axes = list(target.axes)
        if not figure_axes:
            raise ValueError("The figure has no axes to put an inset on.")
        axes = figure_axes[0]
    if ylim is None and "auto_ylim" not in options and not (spec or {}).get("ylim"):
        options["auto_ylim"] = True
    return _add_inset(axes, xlim, ylim, spec, **options)
