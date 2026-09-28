"""
Legend and series names that may contain LaTeX.

Explicit user request (2026-09-28): the names of datasets in a legend --
set by renaming an experiment, or in Plot Style -- "should be LaTeX
formattable". matplotlib already renders anything between a pair of
``$`` as mathematics (its built-in *mathtext*: sub/superscripts, Greek,
``\\mathrm``, fractions...; no LaTeX installation needed), so the work
here is not rendering but **safety**: a name whose ``$...$`` does not
parse (``$^{7$Li``) raises when the figure is drawn, which would blank
the whole plot. Every name that goes into a legend passes through
:func:`safe_label` first, and the GUI checks a typed name with
:func:`label_problem` before accepting it, so the user hears what is
wrong instead of seeing an empty canvas.

Examples of names that work::

    r"$^{7}$Li, 25 $^{\\circ}$C"
    r"LiCoO$_2$ after 10 cycles"
    r"$T_1$-filtered"
    r"$\\delta_{\\mathrm{iso}}$ = 12 ppm"
"""

from __future__ import annotations

from functools import lru_cache

from matplotlib.mathtext import MathTextParser


@lru_cache(maxsize=1)
def _parser() -> MathTextParser:
    return MathTextParser("path")


def _math_segments_balanced(text: str) -> bool:
    """matplotlib only treats text as mathematics when it holds an even,
    non-zero number of unescaped ``$``."""
    unescaped = text.count("$") - text.count(r"\$")
    return unescaped > 0 and unescaped % 2 == 0


def label_problem(text: str) -> str | None:
    """``None`` if ``text`` will draw; otherwise a one-line explanation.

    Plain text (no ``$``) always draws. An odd number of ``$`` is not an
    error for matplotlib -- it draws the text literally -- but it is
    almost always a typo, so it is reported too."""
    if text is None:
        return None
    text = str(text)
    unescaped = text.count("$") - text.count(r"\$")
    if unescaped == 0:
        return None
    if unescaped % 2:
        return ("Unmatched '$': LaTeX goes between a pair of dollar signs, e.g. $^{7}$Li. "
                "Write \\$ for a literal dollar sign.")
    try:
        _parser().parse(text, dpi=72)
    except ValueError as exc:
        import re
        lines = str(exc).strip().splitlines()
        detail = lines[-1] if lines else ""
        # "ParseFatalException: Unknown symbol: \foo, found '\'  (at char 0), (line:1, col:1)"
        detail = re.sub(r"^\w*Exception:\s*", "", detail)
        col = re.search(r"col:(\d+)", detail)
        detail = re.sub(r"\s*\(at char \d+\).*$", "", detail).strip().rstrip(",")
        where = f" (at character {col.group(1)})" if col else ""
        return f"The LaTeX could not be read: {detail}{where}." if detail else \
            "The LaTeX could not be read."
    return None


def safe_label(text) -> str:
    """``text`` as given if it draws; otherwise with every ``$`` escaped,
    so it draws literally rather than breaking the figure."""
    text = "" if text is None else str(text)
    if not _math_segments_balanced(text):
        return text
    return text if label_problem(text) is None else text.replace(r"\$", "$").replace("$", r"\$")


# ============================================================
# Legend line width, and names written on the traces themselves
# ============================================================

def set_legend_linewidth(legend, width: float | None) -> None:
    """Draw the sample lines in ``legend`` at ``width`` points, whatever
    the plotted lines' own width -- a thin spectrum's legend key is
    otherwise too faint to read the colour from. ``None`` or 0 leaves
    them as matplotlib drew them (matching the traces)."""
    if legend is None or not width:
        return
    for line in legend.get_lines():
        line.set_linewidth(float(width))


def _noise_level(x, y, xlo, xhi, x_at, n_sigma, window_fraction):
    """Baseline and noise of one trace near ``x_at``: the median of the
    points within ``window_fraction`` of the visible span around it, plus
    ``n_sigma`` times the noise (from the spread of point-to-point
    differences over the visible range, which a peak barely affects)."""
    import numpy as np

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    visible = (x >= xlo) & (x <= xhi) & np.isfinite(y)
    if np.count_nonzero(visible) < 3:
        visible = np.isfinite(y)
    xv, yv = x[visible], y[visible]
    if yv.size == 0:
        return None
    span = (xhi - xlo) or 1.0
    near = np.abs(xv - x_at) <= window_fraction * span
    if np.count_nonzero(near) < 3:
        near = np.argsort(np.abs(xv - x_at))[:min(5, xv.size)]
    baseline = float(np.median(yv[near]))
    diffs = np.diff(yv)
    sigma = 1.4826 * float(np.median(np.abs(diffs - np.median(diffs)))) / np.sqrt(2.0) \
        if diffs.size else 0.0
    return baseline + n_sigma * sigma


def label_traces(ax, lines=None, names=None, position: float = 0.02, fontsize=None,
                 n_sigma: float = 3.0, pad_points: float = 2.0, window_fraction: float = 0.04,
                 fontfamily=None):
    """Write each trace's name on the plot just above its own noise,
    instead of a legend (explicit user request, 2026-09-28).

    ``position`` : where along the visible x-range the names sit, as a
    fraction from the plot's left edge (0) to its right edge (1) --
    whatever the axis direction, so 0 is always the left of the figure.
    The name slides smoothly with it: at 0 it starts at the left edge,
    at 1 it ends at the right edge, at 0.5 it is centred -- so it never
    runs off the plot. The noise is read under the name's centre.

    Each name's height is its trace's baseline at that position plus
    ``n_sigma`` times the trace's noise, and ``pad_points`` above that --
    "just above the noise level". Names on the same axes (an overlay)
    that would collide are moved up, one line at a time, in the order
    of their heights. Drawn in the trace's colour; LaTeX allowed (see
    :func:`safe_label`).

    ``lines`` default to every named line on ``ax``; ``names`` default
    to their labels. Returns the Text artists (remove them to undo).
    The x position is in axes coordinates, so the names stay put when
    the plot is zoomed; call again after a zoom to re-read the noise at
    the new position (the GUI does this on every x-limit change)."""
    from matplotlib.transforms import blended_transform_factory, offset_copy

    if lines is None:
        lines = [ln for ln in ax.get_lines() if ln.get_label() and not ln.get_label().startswith("_")]
    lines = list(lines)
    if names is None:
        names = [ln.get_label() for ln in lines]
    position = min(max(float(position), 0.0), 1.0)
    xlo, xhi = sorted(ax.get_xlim())
    fontsize = fontsize or 10.0
    try:
        renderer = ax.figure.canvas.get_renderer()
    except AttributeError:
        renderer = None
    axes_width_px = ax.get_window_extent(renderer).width or 1.0

    def text_width(name):
        """The name's width as a fraction of the axes."""
        if renderer is None:
            return 0.0
        probe = ax.text(0, 0, safe_label(name), fontsize=fontsize, transform=ax.transAxes)
        if fontfamily:
            probe.set_fontfamily(fontfamily)
        try:
            return min(1.0, probe.get_window_extent(renderer).width / axes_width_px)
        finally:
            probe.remove()

    placed = []
    for line, name in zip(lines, names):
        if not name:
            continue
        width = text_width(name)
        left = position * (1.0 - width)            # 0: flush left, 1: flush right
        centre = left + width / 2.0
        x_at = ax.transData.inverted().transform(ax.transAxes.transform((centre, 0.5)))[0]
        y = _noise_level(line.get_xdata(), line.get_ydata(), xlo, xhi, x_at, n_sigma,
                         window_fraction)
        if y is None:
            continue
        placed.append((y, line, name, left, x_at))
    placed.sort(key=lambda item: item[0])

    # Collision handling in display space: each name at least one text
    # line above the one below it on the same axes.
    line_height_px = fontsize * 1.25 * ax.figure.dpi / 72.0
    pad_px = pad_points * ax.figure.dpi / 72.0
    texts = []
    previous_top = None
    for y, line, name, left, x_at in placed:
        y_px = ax.transData.transform((x_at, y))[1] + pad_px
        extra_px = 0.0
        if previous_top is not None and y_px < previous_top:
            extra_px = previous_top - y_px
        previous_top = y_px + extra_px + line_height_px
        transform = offset_copy(blended_transform_factory(ax.transAxes, ax.transData),
                                fig=ax.figure, x=0.0,
                                y=(pad_px + extra_px) * 72.0 / ax.figure.dpi, units="points")
        text = ax.text(left, y, safe_label(name), transform=transform, ha="left", va="bottom",
                       color=line.get_color(), fontsize=fontsize, zorder=6, clip_on=False)
        if fontfamily:
            text.set_fontfamily(fontfamily)
        text._opennmr_trace_label = True
        texts.append(text)
    return texts
