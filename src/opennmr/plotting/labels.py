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


#: Characters Arial (and most publication faces) do not have, with the
#: mathtext that draws them from the figure's maths font instead.
_NO_GLYPH = {"∥": r"$\parallel$", "⊥": r"$\perp$"}


def mathtext_safe(text) -> str:
    """``text`` with characters that common faces lack (∥, ⊥) written as
    mathtext, so matplotlib draws them instead of warning "Glyph 8741
    (PARALLEL TO) missing from font(s) Arial" and leaving a gap. For text
    drawn on a figure only: the same string stays as typed in logs and notes."""
    text = "" if text is None else str(text)
    for char, math in _NO_GLYPH.items():
        text = text.replace(char, math)
    return text


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
                 fontfamily=None, height: float = 0.0):
    """Write each trace's name on the plot just above its own noise,
    instead of a legend (explicit user request, 2026-09-28).

    ``position`` : where along the visible x-range the names sit, as a
    fraction from the plot's left edge (0) to its right edge (1) --
    whatever the axis direction, so 0 is always the left of the figure.
    The name slides smoothly with it: at 0 it starts at the left edge,
    at 1 it ends at the right edge, at 0.5 it is centred -- so it never
    runs off the plot. The noise is read under the name's centre.

    ``height`` : extra lift of every name above that noise level, as a
    fraction of the axes' height (0 = just above the noise, 0.25 = a
    quarter of the plot higher), so a name can be moved clear of a
    peak or a neighbouring trace. It is the vertical counterpart of
    ``position``.

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
    axes_height_px = ax.get_window_extent(renderer).height or 1.0
    height = min(max(float(height or 0.0), 0.0), 1.0)
    pad_px = pad_points * ax.figure.dpi / 72.0 + height * axes_height_px
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


# ----------------------------------------------------------------------
# Quantity symbols in italics (explicit user request, 2026-10-17: "any
# algebraic parameter name (T for temperature, T_1 ...) needs to have the
# letter italicised"). IUPAC: a quantity's symbol is italic; numbers,
# units and descriptive subscripts (iso, EFG, max) are upright.
# ----------------------------------------------------------------------

import re as _re

_GREEK = {"α": r"\alpha", "β": r"\beta", "γ": r"\gamma", "δ": r"\delta", "ε": r"\epsilon",
          "η": r"\eta", "θ": r"\theta", "κ": r"\kappa", "λ": r"\lambda",
          "ν": r"\nu", "ρ": r"\rho", "σ": r"\sigma", "τ": r"\tau", "φ": r"\phi",
          "χ": r"\chi", "ω": r"\omega", "Δ": r"\Delta", "Γ": r"\Gamma", "Ω": r"\Omega"}
_SUBSCRIPT_DIGITS = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
_SYMBOL_CHARS = "A-Za-z" + "".join(_GREEK)
_SUB = "₀₁₂₃₄₅₆₇₈₉"
#: A label that starts with a symbol: one letter (Δ may lead), Unicode
#: subscript digits and an optional ρ, then a unit in brackets, a slash,
#: "=", or nothing -- "T (K)", "t (min)", "T₁ρ (ms)", "δ / ppm", "D".
_LEADING = _re.compile(
    rf"^(?P<sym>Δ?[{_SYMBOL_CHARS}])(?P<sub>[{_SUB}]*ρ?)(?P<rest>(?:\s*[(/=].*)?)$")
#: A symbol with Unicode subscript digits anywhere: "ln(T₁)", "R₂ fit".
_INLINE = _re.compile(rf"(?<![{_SYMBOL_CHARS}\\$])(?P<sym>[A-Z])(?P<sub>[{_SUB}]+ρ?)"
                      rf"(?![{_SYMBOL_CHARS}])")
#: Upright text followed by a mathtext subscript: "x$_{EFG}$", "δ$_{xx}$".
_SPLIT = _re.compile(rf"(?<![{_SYMBOL_CHARS}\\$])(?P<sym>[{_SYMBOL_CHARS}])\$_\{{(?P<sub>[^{{}}$]*)\}}\$")


def _symbol(sym: str) -> str:
    return "".join(_GREEK.get(ch, ch) for ch in sym)


def _subscript(sub: str) -> str:
    """``"₁ρ"`` -> ``"_{1\\rho}"``; a word (iso, EFG, max) upright."""
    sub = sub.translate(_SUBSCRIPT_DIGITS).replace("ρ", r"\rho")
    if not sub:
        return ""
    if _re.fullmatch(r"[A-Za-z]{2,}", sub) and sub.lower() not in ("xx", "yy", "zz", "xy",
                                                                     "xz", "yz", "ij"):
        return rf"_{{\mathrm{{{sub}}}}}"
    return f"_{{{sub}}}"


def italicise_symbols(text) -> str:
    """``text`` with its quantity symbols in italic mathtext:

    * a label that *is* a symbol and unit -- ``"T (K)"`` -> ``"$T$ (K)"``,
      ``"T₁ρ (ms)"`` -> ``"$T_{1\\rho}$ (ms)"``, ``"δ (ppm)"`` ->
      ``"$\\delta$ (ppm)"``;
    * a symbol with Unicode subscripts anywhere -- ``"ln(T₁)"``;
    * an upright letter before a mathtext subscript -- ``"δ$_{iso}$"`` ->
      ``"$\\delta_{\\mathrm{iso}}$"``.

    Words, units and existing mathtext are left alone ("Delay (s)" stays
    as it is), so it is safe to pass any axis label through. Non-strings
    come back unchanged."""
    if not isinstance(text, str) or not text.strip():
        return text
    if "$" not in text:
        m = _LEADING.match(text)
        if m:
            return f"${_symbol(m['sym'])}{_subscript(m['sub'])}$" + _outside(m["rest"])
    out = _SPLIT.sub(lambda m: f"${_symbol(m['sym'])}{_subscript(m['sub'])}$", text)
    if out.count("$") % 2:
        return text                 # unbalanced: not ours to touch
    parts = out.split("$")
    first = _LEADING.match(parts[0]) if parts[0] else None
    if first and first["rest"]:
        # "D ($\mathregular{m^2 s^{-1}}$)": a symbol before a mathtext unit
        parts[0] = (f"${_symbol(first['sym'])}{_subscript(first['sub'])}$"
                    + _outside(first["rest"]))
        return "$".join([parts[0]] + [_outside(p) if i % 2 == 0 else p
                                      for i, p in enumerate(parts[1:], start=1)])
    return "$".join(_outside(p) if i % 2 == 0 else p for i, p in enumerate(parts))


#: A Greek letter standing alone in plain text: "δ − ...", "τ fit".
_LONE_GREEK = _re.compile(rf"(?<![{_SYMBOL_CHARS}\\])(?P<g>[{''.join(_GREEK)}])"
                          rf"(?![{_SYMBOL_CHARS}])")


#: Reciprocal temperature: "1000/T (K)", "1/T".
_RECIPROCAL = _re.compile(rf"(?P<pre>(?<![0-9.])(?:1000|100|10|1)/)(?P<sym>T)(?![{_SYMBOL_CHARS}])")


def _outside(segment: str) -> str:
    """Symbols in a stretch of plain (non-mathtext) label text."""
    segment = _RECIPROCAL.sub(lambda mm: f"{mm['pre']}${mm['sym']}$", segment)
    segment = _INLINE.sub(lambda mm: f"${_symbol(mm['sym'])}{_subscript(mm['sub'])}$", segment)
    pieces = segment.split("$")
    return "$".join(_LONE_GREEK.sub(lambda g: f"${_GREEK[g['g']]}$", p) if i % 2 == 0 else p
                    for i, p in enumerate(pieces))
