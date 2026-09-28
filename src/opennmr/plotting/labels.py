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
