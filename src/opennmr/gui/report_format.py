"""Readable result read-outs: plain-text reports laid out as rich text.

Explicit user request (2026-10-13): the outputs under Fit and Simulate
"looked like a wall of text that is not very inviting or friendly".
The plain text every panel already writes (and logs, copies and saves)
is unchanged; this module only decides how it is *shown*:

* :func:`report_html` -- any plain report, laid out generically: the first
  line as a headline, "Head: a; b; c" lines as a bold head over its items,
  "x · y · z" lines as separate items, ⚠ lines as warnings, and space
  between paragraphs. Text that is already rich text passes through.
* :func:`sites_html` -- a set of CSA/quadrupolar/J sites as a table, one
  row per site, with whatever every site shares said once above it
  (seven sites with one CSA, one C_Q and one width were seven near-
  identical sentences).
* :func:`combined_simulation_html`, :func:`combined_fit_html`,
  :func:`combined_preview_html` -- Simulate's and Fit's CSA, Quad &
  J-coupling read-outs, from their structured results.

Pure functions of strings and dicts (no Qt), so they are testable and
usable from a notebook (``IPython.display.HTML(report_html(text))``).
"""

from __future__ import annotations

import html
import re

#: Default colours: secondary text and warnings. A widget passes colours
#: from its own palette (see widgets/report_label.py), so both themes work.
MUTED = "#7a7a7a"
WARN = "#c27a1a"

#: Rows a site table shows before summarising the rest.
MAX_TABLE_ROWS = 24

_RICH = re.compile(r"</?(b|i|u|p|br|div|span|table|tr|td|th|ul|ol|li|font|sub|sup|"
                   r"html|body|a|h\d|small|big|hr|img)\b[^<>]*>", re.IGNORECASE)

_P = '<p style="margin-top:0px; margin-bottom:6px;">'
_LINESHAPES = {"pseudo_voigt": "pseudo-Voigt", "gaussian": "Gaussian",
               "lorentzian": "Lorentzian", "voigt": "Voigt"}


def is_rich(text: str) -> bool:
    """Whether ``text`` already carries HTML tags (then it is shown as is)."""
    return bool(text) and bool(_RICH.search(text))


def _e(text) -> str:
    return html.escape(str(text), quote=False)


def _muted(text_html: str, muted: str) -> str:
    return f'<span style="color:{muted};">{text_html}</span>'


def _items(items, muted=None) -> str:
    """Items separated by a mid-dot; short items kept whole on a line."""
    out = []
    for item in items:
        item = item.strip()
        if not item:
            continue
        piece = _e(item)
        if len(item) <= 40:
            piece = piece.replace(" ", "&nbsp;")
        out.append(piece)
    dot = "&nbsp;&nbsp;·&nbsp; "
    joined = dot.join(out)
    return _muted(joined, muted) if muted else joined


def _warning(text: str, warn: str) -> str:
    body = text.lstrip("⚠").strip()
    return (f'{_P}<span style="color:{warn}; font-weight:600;">⚠&nbsp;</span>'
            f"{_e(body)}</p>")


def report_html(text: str, muted: str = MUTED, warn: str = WARN) -> str:
    """A plain report as readable rich text (see the module notes).
    Empty in, empty out; rich text in, unchanged out."""
    if not text:
        return ""
    if is_rich(text):
        return text
    lines = [line.rstrip() for line in str(text).split("\n")]
    lines = [line for line in lines if line.strip()]
    blocks = []
    for i, line in enumerate(lines):
        s = line.strip()
        indent = len(line) - len(line.lstrip())
        if s.startswith("⚠") or s.lower().startswith("warning:"):
            blocks.append(_warning(s, warn))
            continue
        head, sep, rest = s.partition(": ")
        if (sep and ";" in rest and len(head) <= 70 and ";" not in head
                and rest.count(";") >= 1):
            items = [b for b in rest.split(";")]
            pad = "&nbsp;" * 4 if indent else ""
            blocks.append(f"{_P}{pad}<b>{_e(head)}</b><br>{pad}{_items(items)}</p>")
            continue
        if "  ·  " in s or " · " in s:
            parts = re.split(r"\s+·\s+", s)
            weight = ' style="font-weight:600;"' if i == 0 and len(lines) > 1 else ""
            blocks.append(f"{_P}<span{weight}>{_items(parts)}</span></p>")
            continue
        if i == 0 and len(lines) > 1 and len(s) <= 90:
            blocks.append(f'{_P}<span style="font-weight:600;">{_e(s)}</span></p>')
            continue
        blocks.append(f"{_P}{_e(s)}</p>")
    return "".join(blocks)


# ---- CSA, Quad & J-coupling sites ---------------------------------------------------------------

def _g(value, digits=6) -> str:
    try:
        return f"{float(value):.{digits}g}"
    except (TypeError, ValueError):
        return str(value)


def site_cells(site: dict, show_nucleus: bool = False) -> dict:
    """{column: text} for one site dict (``Site.to_dict()`` or a panel's)."""
    cells = {}
    if show_nucleus:
        cells["Nucleus"] = str(site.get("nucleus") or site.get("isotope") or "")
    cells["δiso (ppm)"] = _g(site.get("delta_iso", 0.0))
    csa, quad, jays = [], [], []
    for inter in site.get("interactions") or []:
        kind = inter.get("type")
        if kind == "csa":
            names = (("Δ", "η") if inter.get("convention", "Haeberlen") == "Haeberlen"
                     else ("span", "skew"))
            text = (f"{names[0]} {_g(inter.get('aniso', 0))} ppm, "
                    f"{names[1]} {_g(inter.get('asym', 0), 4)}")
            euler = inter.get("euler") or ()
            if any(float(a) for a in euler):
                text += " at ({})°".format(", ".join(_g(a, 4) for a in euler))
            csa.append(text)
        elif kind == "quadrupolar":
            quad.append(f"C_Q {_g(inter.get('C_Q', 0), 5)} MHz, "
                        f"η_Q {_g(inter.get('eta_Q', 0), 4)}")
        elif kind == "jcoupling":
            jays.append(f"{_g(inter.get('J_Hz', 0), 5)} Hz to "
                        f"{inter.get('n', 1)}×{inter.get('nuclide', '?')}")
    if csa:
        cells["CSA"] = "; ".join(csa)
    if quad:
        cells["Quadrupolar"] = "; ".join(quad)
    if jays:
        cells["J"] = "; ".join(jays)
    cells["Amount"] = _g(site.get("amount", 1.0), 4)
    shape = str(site.get("lineshape", "") or "")
    width = f"{_g(site.get('fwhm', 0), 5)} ppm {_LINESHAPES.get(shape, shape)}".strip()
    if shape == "pseudo_voigt" and site.get("eta") not in (None, ""):
        width += f" (L {_g(site.get('eta'), 3)})"
    cells["Broadening"] = width
    link = site.get("link") or {}
    if link.get("to"):
        own = link.get("free") or []
        cells["Link"] = f"follows {link['to']}" + (f" (own {', '.join(own)})" if own else "")
    return cells


#: Columns a site table always keeps per row, even when every site agrees.
_ALWAYS_PER_ROW = ("δiso (ppm)", "Amount")
#: Columns that are numbers (right-aligned).
_NUMERIC = ("δiso (ppm)", "Amount")


def sites_html(site_dicts, muted: str = MUTED, max_rows: int = MAX_TABLE_ROWS) -> str:
    """Sites as a table, one row each; what every site shares is said once
    above it ("All 7 sites: …"). More than ``max_rows`` sites: the first
    ``max_rows`` and a line saying how many more."""
    sites = list(site_dicts or [])
    if not sites:
        return ""
    nuclei = {str(s.get("nucleus") or s.get("isotope") or "") for s in sites}
    rows = [(str(s.get("label", f"Site {i + 1}")), site_cells(s, show_nucleus=len(nuclei) > 1))
            for i, s in enumerate(sites)]
    columns: list[str] = []
    for _label, cells in rows:
        for c in cells:
            if c not in columns:
                columns.append(c)
    shared = {}
    if len(rows) > 1:
        for c in columns:
            if c in _ALWAYS_PER_ROW:
                continue
            values = {cells.get(c) for _l, cells in rows}
            if len(values) == 1 and None not in values:
                shared[c] = values.pop()
    per_row = [c for c in columns if c not in shared]
    out = []
    if shared:
        what = "Both sites" if len(rows) == 2 else f"All {len(rows)} sites"
        bits = [f"{c} {v}" if c != "Broadening" else f"FWHM {v}" for c, v in shared.items()]
        out.append(f"{_P}<b>{what}:</b> {_items(bits)}</p>")
    head = "".join(
        f'<th align="{"right" if c in _NUMERIC else "left"}" style="color:{muted}; '
        f'font-weight:600; padding-right:10px;">{_e(c)}</th>' for c in ["Site"] + per_row)
    body = []
    for label, cells in rows[:max_rows]:
        tds = [f'<td style="padding-right:10px;"><b>{_e(label)}</b></td>']
        for c in per_row:
            align = "right" if c in _NUMERIC else "left"
            tds.append(f'<td align="{align}" style="padding-right:10px;">'
                       f'{_e(cells.get(c, "—"))}</td>')
        body.append("<tr>" + "".join(tds) + "</tr>")
    out.append(f'<table cellspacing="0" cellpadding="2"><tr>{head}</tr>{"".join(body)}</table>')
    if len(rows) > max_rows:
        out.append(f"{_P}" + _muted(f"… and {len(rows) - max_rows} more sites (every site is in "
                                    "the saved simulation and Copy as Python).", muted) + "</p>")
    return "".join(out)


def _orientations(powder: dict) -> str:
    counts = {}
    for v in (powder or {}).values():
        key = (int(v.get("orientations", 0)), bool(v.get("octant")))
        counts[key] = counts.get(key, 0) + 1
    if not counts:
        return ""
    if len(counts) == 1:
        (n, octant), = counts
        return f"{n:,} orientations" + (" (octant)" if octant else "")
    return ", ".join(f"{c} sites {n:,}" + (" (octant)" if o else "")
                     for (n, o), c in counts.items()) + " orientations"


def conditions(settings: dict, powder: dict | None = None) -> list[str]:
    """The simulation/fit conditions as short items."""
    mas = float(settings.get("mas_Hz") or 0.0)
    items = ["static" if not mas else f"MAS {mas / 1e3:g} kHz"]
    if settings.get("larmor_Hz"):
        items.append(f"ν₀ {float(settings['larmor_Hz']) / 1e6:.6g} MHz")
    order = settings.get("order")
    if order:
        items.append("exact quadrupolar" if order == "exact" else f"{order}-order quadrupolar")
    transitions = settings.get("transitions")
    if transitions:
        items.append({"all": "all transitions", "ct": "central transition",
                      "st": "satellite transitions"}.get(transitions, f"transitions {transitions}"))
    text = _orientations(powder or {})
    if text:
        items.append(text)
    return items


def combined_simulation_html(result: dict, elapsed: float, warning: str = "",
                             muted: str = MUTED, warn: str = WARN) -> str:
    """Simulate ▸ CSA, Quad & J-coupling's read-out."""
    sites = result.get("site_dicts") or []
    n = len(sites)
    head = (f"{n} site{'s' if n != 1 else ''} simulated in {elapsed:.2f} s" if n
            else f"Simulated in {elapsed:.2f} s")
    out = [f'{_P}<span style="font-weight:600;">{_e(head)}</span><br>'
           f"{_items(conditions(result.get('settings') or {}, result.get('powder')), muted)}</p>"]
    out.append(sites_html(sites, muted=muted))
    for note in result.get("notes") or []:
        if str(note).startswith("⚠"):
            out.append(_warning(str(note), warn))
            continue
        out.append('<p style="margin-top:6px; margin-bottom:0px;">'
                   f'{_muted(_e(note), muted)}</p>')
    if warning:
        out.append(_warning(warning.strip(), warn))
    return "".join(out)


def combined_fit_html(lines: list[str], quality: dict, elapsed: float, warnings=(),
                      has_quadrupolar: bool = True, muted: str = MUTED, warn: str = WARN) -> str:
    """Fit ▸ CSA, Quad & J-coupling's read-out above its table.

    ``lines``: the fit log's first line ("CSA, Quad & J-coupling fit: <
    conditions>.") and any multi-start lines."""
    metrics = [("R²", f"{quality['r2']:.5f}"), ("RMSE", f"{quality['rmse']:.3g}"),
               ("reduced χ²", f"{quality['reduced_chi2']:.3g}"), ("time", f"{elapsed:.1f} s")]
    cells = "".join(
        f'<td style="padding-right:16px;"><span style="color:{muted};">{_e(k)}</span><br>'
        f'<span style="font-weight:600;">{_e(v)}</span></td>' for k, v in metrics)
    out = [f'<table cellspacing="0" cellpadding="0" style="margin-bottom:6px;">'
           f"<tr>{cells}</tr></table>"]
    first = lines[0] if lines else ""
    _title, sep, rest = first.partition(": ")
    if not sep:
        _title, sep, rest = first.partition(" — ")
    if sep:
        out.append(f"{_P}{_items(rest.rstrip('.').split(', '), muted)}</p>")
    elif first:
        out.append(f"{_P}{_muted(_e(first), muted)}</p>")
    for line in lines[1:]:
        out.append(f"{_P}{_e(line)}</p>")
    if has_quadrupolar:
        out.append(f"{_P}" + _muted("|C_Q| is reported: its sign is not in a lineshape.", muted)
                   + "</p>")
    for w in warnings or ():
        out.append(_warning(str(w), warn))
    return "".join(out)


def combined_preview_html(n_orientations: int, fit_orientations: int, amounts: dict,
                          estimated: bool, muted: str = MUTED) -> str:
    """The live preview's line under Fit ▸ CSA, Quad & J-coupling."""
    how = "estimated from the data" if estimated else "as typed"
    amount_text = ", ".join(f"{k} {v:.3g}" for k, v in amounts.items())
    items = [f"{n_orientations:,} orientations (the fit uses {fit_orientations:,})",
             f"amounts {how}"]
    return (f'{_P}<span style="font-weight:600;">Starting model shown on the Fit view</span>'
            f"<br>{_items(items, muted)}"
            + (f"<br>{_muted(_e(amount_text), muted)}" if amount_text else "") + "</p>")
