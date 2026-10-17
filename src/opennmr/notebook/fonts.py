"""Typefaces for openNMR's figures: every font matplotlib can draw with
on this computer, with samples, and how to use one (explicit user
request, 2026-10-17 -- after "list all fonts available in matplotlib,
plus samples").

The GUI's Plot Style ▸ Font family offers the same list, so a notebook
and the GUI agree on what is available.

Notebook use::

    from opennmr.notebook import fonts
    fonts.available_fonts()              # ['Arial', 'Calibri', ...]
    fonts.font_samples()                 # an HTML table of samples, in Jupyter
    fonts.plot_font_samples(["Arial", "Times New Roman"])   # a figure instead
    with fonts.use_font("Times New Roman"):
        ...                               # plots made here use it, maths included

A figure saved as PDF/PS by :func:`opennmr.plotting.vector_export.save_vector_safe`
(or the GUI's Save Figure) embeds the font as TrueType, so the text stays
text in Illustrator or Inkscape; SVG keeps glyphs as outlines, so it looks
the same on a computer without the font.
"""
from __future__ import annotations

import contextlib
import html

#: matplotlib's generic families: always resolve to something.
GENERIC_FAMILIES = ("sans-serif", "serif", "monospace", "cursive", "fantasy")

#: Families matplotlib registers that are not text faces (symbol and
#: emoji fonts draw boxes or pictures for letters).
_NOT_TEXT = ("symbol", "webdings", "wingdings", "emoji", "marlett", "holomdl2",
             "segmdl2", "segoe fluent icons", "segoe mdl2", "dingbat", "cmex", "cmsy")


def available_fonts(text_only: bool = True) -> list[str]:
    """Every font family matplotlib can use on this computer, sorted,
    without duplicates (each family once, whatever its weights and
    styles). ``text_only`` drops symbol/emoji/icon fonts."""
    from matplotlib import font_manager
    names = set()
    for entry in font_manager.fontManager.ttflist:
        name = (entry.name or "").strip()
        if not name or name.startswith("."):
            continue
        if text_only and any(bad in name.lower() for bad in _NOT_TEXT):
            continue
        names.add(name)
    return sorted(names, key=str.casefold)


def font_available(name: str | None) -> bool:
    """Whether ``name`` is a generic family or an installed face."""
    if not name:
        return True
    if name in GENERIC_FAMILIES:
        return True
    from matplotlib import font_manager
    return any(entry.name == name for entry in font_manager.fontManager.ttflist)


def font_samples(fonts=None, sample: str = "δ(¹³C) = 128.4 ppm  Ea = 0.32 eV  ABCabc 0123",
                 columns: int = 2, size_px: int = 20):
    """An HTML table of each font's name written in that font (Jupyter
    shows it; elsewhere it is a string). ``fonts``: names, default all."""
    fonts = available_fonts() if fonts is None else list(fonts)
    rows = "\n".join(
        f"<p style='margin:2px 0'><span style='font-family:sans-serif;font-size:11px;"
        f"color:#666'>{html.escape(f)}</span><br><span style=\"font-family:'{html.escape(f)}';"
        f"font-size:{size_px}px\">{html.escape(sample)}</span></p>" for f in fonts)
    body = f"<div style='column-count:{int(columns)}'>{rows}</div>"
    try:
        from IPython.display import HTML
        return HTML(body)
    except ImportError:
        return body


def plot_font_samples(fonts=None, sample: str = "δ(¹³C) = 128.4 ppm  ABCabc 0123",
                      fontsize: float = 11.0):
    """The same samples drawn by matplotlib itself -- what a figure will
    actually look like (a browser may substitute a missing face; this
    shows matplotlib's own choice). Returns ``(fig, ax)``."""
    import matplotlib.pyplot as plt
    fonts = available_fonts() if fonts is None else list(fonts)
    height = max(1.0, 0.32 * len(fonts) + 0.3)
    fig, ax = plt.subplots(figsize=(7.0, height))
    ax.set_axis_off()
    for i, name in enumerate(fonts):
        y = 1.0 - (i + 0.5) / max(len(fonts), 1)
        ax.text(0.0, y, name, fontsize=fontsize * 0.75, color="0.45", va="center",
                transform=ax.transAxes, family="sans-serif")
        ax.text(0.32, y, sample, fontsize=fontsize, va="center", transform=ax.transAxes,
                family=name)
    return fig, ax


def font_rc(name: str | None) -> dict:
    """The rcParams that make ``name`` the font of a figure: its text and
    its mathematics (``$^{13}$C`` labels use the same face rather than
    DejaVu), and TrueType embedding in PDF/PS. Empty for ``None``."""
    if not name:
        return {}
    rc = {"pdf.fonttype": 42, "ps.fonttype": 42}
    if name in GENERIC_FAMILIES:
        rc["font.family"] = name
        return rc
    rc.update({"font.family": name, "mathtext.fontset": "custom", "mathtext.rm": name,
               "mathtext.it": f"{name}:italic", "mathtext.bf": f"{name}:bold",
               "mathtext.sf": name})
    return rc


@contextlib.contextmanager
def use_font(name: str | None):
    """``with use_font("Times New Roman"): ...`` -- figures made inside use
    that face for text and maths."""
    import matplotlib
    with matplotlib.rc_context(font_rc(name)):
        yield
