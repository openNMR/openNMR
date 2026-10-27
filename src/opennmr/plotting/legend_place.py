"""Where a legend goes, and how many columns it has -- one place for
every plot (the GUI's Plot Style ▸ Legend and the notebook functions).

User request (2026-10-27): "plot style options to be able to move the
legend, or to make the legend multi-column".

    from opennmr.plotting import legend_placement
    ax.legend(**legend_placement("outside right", columns=2))

Positions
---------
* matplotlib's own, inside the axes: ``"best"``, ``"upper right"``,
  ``"upper left"``, ``"lower left"``, ``"lower right"``, ``"center"``,
  ``"upper center"``, ``"lower center"``, ``"center left"``,
  ``"center right"``.
* ``"outside right"``: beside the axes, centred on them. A 2D figure
  laid out by ``tight_layout`` (the GUI's layout pass) makes room for it
  itself; a 3D picture (whose layout is not ``tight_layout``) makes room
  with :func:`make_room_for_legend`.
* ``"custom"``: the legend's upper-left corner at ``(x, y)``, in axes
  fractions -- (0, 1) is the axes' top-left corner; values outside 0-1
  put it outside the axes (e.g. ``(0, 1.15)`` above them).

``columns`` : the number of columns; 0 (or ``None``) leaves the choice
to whoever builds the legend (one column unless they chose otherwise).
"""

from __future__ import annotations

__all__ = ["LEGEND_POSITIONS", "legend_placement", "make_room_for_legend"]

#: Every position :func:`legend_placement` accepts, in the order the GUI
#: offers them.
LEGEND_POSITIONS = (
    "best", "upper right", "upper left", "lower left", "lower right", "center",
    "upper center", "lower center", "center left", "center right",
    "outside right", "custom",
)

#: Gap between the axes and an "outside right" legend, in axes fraction.
_OUTSIDE_GAP = 0.02


def legend_placement(position: str = "best", columns: int | None = 0, x: float = 1.02,
                     y: float = 1.0) -> dict:
    """The keyword arguments of ``ax.legend`` that put it at
    ``position`` with ``columns`` columns (see the module docstring)."""
    position = str(position or "best").strip().lower()
    if position not in LEGEND_POSITIONS:
        raise ValueError(f"Unknown legend position {position!r}; one of {LEGEND_POSITIONS}.")
    if position == "outside right":
        kwargs = {"loc": "center left", "bbox_to_anchor": (1.0 + _OUTSIDE_GAP, 0.5)}
    elif position == "custom":
        kwargs = {"loc": "upper left", "bbox_to_anchor": (float(x), float(y))}
    else:
        kwargs = {"loc": position}
    if columns:
        columns = int(columns)
        if columns < 1:
            raise ValueError("columns must be 1 or more (0 for the default).")
        kwargs["ncols"] = columns
    return kwargs


def make_room_for_legend(ax, margin: float = 0.01, renderer=None) -> float:
    """Move ``ax`` (a 3D axes, or any whose figure is not laid out by
    ``tight_layout``) so that its legend lies inside the figure, by
    narrowing or lowering the axes on the side the legend sticks out.

    The first call remembers the axes' own position; every call starts
    from it, so a legend moved back inside gives the room back. Returns
    the fraction of the figure taken (0 when nothing was moved)."""
    figure = ax.figure
    # Only an axes this has moved is put back: one it never touched may
    # have been placed by someone else since (a colour bar's gap).
    base = getattr(ax, "_opennmr_legend_base", None)
    if base is not None:
        ax.set_position(base)
        del ax._opennmr_legend_base
    else:
        base = ax.get_position().frozen()
    legend = ax.get_legend()
    if legend is None or not legend.get_visible():
        return 0.0
    if renderer is None:
        try:
            renderer = figure.canvas.get_renderer()
        except AttributeError:
            return 0.0
    to_fig = figure.transFigure.inverted()
    x0, y0, x1, y1 = base.x0, base.y0, base.x1, base.y1
    taken = 0.0
    # A few passes: the legend is anchored to the axes, so it moves with
    # them; one pass is usually exact.
    for _ in range(4):
        box = legend.get_window_extent(renderer).transformed(to_fig)
        right = box.x1 - (1.0 - margin)
        left = margin - box.x0
        top = box.y1 - (1.0 - margin)
        bottom = margin - box.y0
        if max(right, left, top, bottom) <= 1e-4:
            break
        if right > 0:
            x1 -= right
        if left > 0:
            x0 += left
        if top > 0:
            y1 -= top
        if bottom > 0:
            y0 += bottom
        # Never below a fifth of the axes' own size: a legend too big for
        # the figure takes what it can and still overhangs (fewer
        # columns, or a smaller legend font, then).
        floor_w, floor_h = 0.2 * base.width, 0.2 * base.height
        if x1 - x0 < floor_w:
            x0, x1 = (x0, x0 + floor_w) if right > 0 else (x1 - floor_w, x1)
        if y1 - y0 < floor_h:
            y0, y1 = (y0, y0 + floor_h) if top > 0 else (y1 - floor_h, y1)
        ax.set_position([x0, y0, x1 - x0, y1 - y0])
        ax._opennmr_legend_base = base
    taken = 1.0 - ((x1 - x0) * (y1 - y0)) / max(base.width * base.height, 1e-12)
    return max(taken, 0.0)
