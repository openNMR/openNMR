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
* :func:`relaxation_report_html`, :func:`global_report_html`,
  :func:`peaks_summary_html`, :func:`curve_fit_summary_html` -- Dynamics'
  curve-fit read-outs (2026-11-09), from the fit dicts themselves.
* :func:`score_report_html` -- a SCORE/OUTSCORE resolution, laid out like
  the curve fit (explicit user request, 2026-11-10).
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


# ---- Solid-state NMR sites ---------------------------------------------------------------

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
    """Simulate ▸ Solid-state NMR's read-out."""
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
    """Fit ▸ Solid-state NMR's read-out above its table.

    ``lines``: the fit log's first line ("Solid-state NMR fit: <
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
    """The live preview's line under Fit ▸ Solid-state NMR."""
    how = "estimated from the data" if estimated else "as typed"
    amount_text = ", ".join(f"{k} {v:.3g}" for k, v in amounts.items())
    items = [f"{n_orientations:,} orientations (the fit uses {fit_orientations:,})",
             f"amounts {how}"]
    return (f'{_P}<span style="font-weight:600;">Starting model shown on the Fit view</span>'
            f"<br>{_items(items, muted)}"
            + (f"<br>{_muted(_e(amount_text), muted)}" if amount_text else "") + "</p>")


# ---- Dynamics: relaxation / diffusion curve fits -------------------------------------------
#
# Explicit user request, 2026-11-09: Dynamics' curve-fit report was the
# monospaced "====" text block; it should read like Fit's read-outs (a
# headline, bold and normal type, sizes, a quality row). These build that
# layout from the fit dict itself rather than by parsing the plain report,
# which (format_relaxation_report / format_global_report) is still what is
# logged, copied, exported and kept.

#: Model keys whose readable name is not just the key with spaces.
_MODEL_NAMES = {
    "stejskal_tanner": "Stejskal–Tanner",
    "stejskal_tanner_stretched": "Stejskal–Tanner, stretched",
    "stejskal_tanner_biexponential": "Stejskal–Tanner, biexponential",
    "diffusion_n": "N-component diffusion",
}
#: Parameter names that are a Greek letter, or a Greek letter with a subscript.
_GREEK = {"beta": "β", "alpha": "α", "tau": "τ", "gamma": "γ", "delta": "δ"}
_SYMBOL = re.compile(r"^([A-Za-z])(\d+[a-z]{0,3}|[a-z])?$")
_SUB = '<sub>{}</sub>'


def model_name(model) -> str:
    """A model key as words: ``inversion_recovery`` -> "Inversion recovery"."""
    key = str(model or "")
    if key in _MODEL_NAMES:
        return _MODEL_NAMES[key]
    words = key.replace("_", " ").strip()
    return words[:1].upper() + words[1:]


def param_html(name) -> str:
    """A parameter name as its symbol: italic letter, subscript label
    (T1 -> *T*₁, T2a -> *T*₂ₐ, M0 -> *M*₀, beta -> *β*, tau_c -> *τ*꜀,
    T1rho -> *T*₁ρ). Names that are not a symbol (a custom model's
    ``amp``) are shown as written, since guessing a subscript would
    misname them. Project rule: algebraic symbols are italic."""
    name = str(name)
    if name in _GREEK:
        return f"<i>{_GREEK[name]}</i>"
    if name == "T1rho":
        return "<i>T</i>" + _SUB.format("1ρ")
    head, sep, sub = name.partition("_")
    if sep and sub and (head in _GREEK or len(head) == 1) and sub.isalnum():
        letter = _GREEK.get(head, head)
        return f"<i>{_e(letter)}</i>" + _SUB.format(_e(sub))
    match = _SYMBOL.match(name)
    if match:
        letter, sub = match.groups()
        return f"<i>{letter}</i>" + (_SUB.format(sub) if sub else "")
    return _e(name)


def _value_html(text) -> str:
    """A formatted value ("68.9 ± 3.2 µs", "2.03e-09 ± … m^2/s") for HTML."""
    return _e(text).replace("^2", "²").replace("^3", "³")


def _metrics_html(metrics, muted: str) -> str:
    """Fit's quality row: each label small and muted over its value in bold."""
    cells = "".join(
        f'<td style="padding-right:16px;"><span style="color:{muted};">{label}</span><br>'
        f'<span style="font-weight:600;">{value}</span></td>' for label, value in metrics)
    return (f'<table cellspacing="0" cellpadding="0" style="margin-top:2px; margin-bottom:6px;">'
            f"<tr>{cells}</tr></table>")


def _section(title: str) -> str:
    # Qt's rich text drops a paragraph's top margin straight after a
    # table, so the gap above a section heading is a small empty line.
    return ('<p style="margin-top:0px; margin-bottom:0px; font-size:4pt;">&nbsp;</p>'
            f'<p style="margin-top:0px; margin-bottom:2px;"><b>{_e(title)}</b></p>')


def _finite(value) -> bool:
    try:
        return value is not None and bool(float(value) == float(value)) \
            and abs(float(value)) != float("inf")
    except (TypeError, ValueError):
        return False


def _quality_metrics(quality: dict, n_points=None) -> list[tuple[str, str]]:
    metrics = []
    if _finite(quality.get("r_squared")):
        metrics.append(("<i>R</i>²", f"{float(quality['r_squared']):.5f}"))
    if _finite(quality.get("rmse")):
        metrics.append(("RMSE", _e(f"{float(quality['rmse']):.4g}")))
    if _finite(quality.get("reduced_chi2")):
        metrics.append(("reduced χ²", _e(f"{float(quality['reduced_chi2']):.4g}")))
    points = quality.get("n_points", n_points)
    if points is not None:
        metrics.append(("points / d.o.f.", _e(f"{points} / {quality.get('dof', '?')}")))
    return metrics


def _uncertainty_html(uncertainty: dict, muted: str) -> str:
    if not uncertainty:
        return ""
    short = uncertainty.get("short", "")
    long = uncertainty.get("description", "")
    return (f'<p style="margin-top:4px; margin-bottom:6px;">'
            f'<span style="font-weight:600;">± is {_e(short)}.</span> '
            f"{_muted(_e(long), muted)}</p>")


def _title_html(title: str, subtitle_items, muted: str) -> str:
    out = f'{_P}<span style="font-size:large; font-weight:600;">{_e(title)}</span>'
    items = [i for i in subtitle_items if i]
    if items:
        out += f"<br>{_items(items, muted)}"
    return out + "</p>"


def relaxation_report_html(fit: dict, title: str | None = None, muted: str = MUTED,
                           warn: str = WARN) -> str:
    """One :func:`~opennmr.processing.relaxation.fit_relaxation` result
    laid out like Fit's read-outs; the same content as
    :func:`~opennmr.processing.relaxation.format_relaxation_report`
    (headline value, every parameter with its unit and kind of ±, the
    relative amplitudes, R², RMSE, reduced χ², points, final I/I₀)."""
    from opennmr.processing import relaxation as rx

    params, errors = fit.get("params") or {}, fit.get("errors") or {}
    names = list(fit.get("param_names") or params)
    quality = fit.get("fit_quality") or {}
    uncertainty = fit.get("uncertainty") or rx.describe_uncertainty(
        fit.get("y_err") is not None, quality.get("dof", 1))
    fractions = rx.amplitude_fractions(params)

    out = [_title_html(title or "Curve fit", [model_name(fit.get("model"))], muted)]

    rate = fit.get("rate_param")
    if rate and rate in params:
        value = rx.format_param_value(rate, params[rate], errors.get(rate))
        out.append(f'<p style="margin-top:2px; margin-bottom:6px;">'
                   f'<span style="font-size:x-large; font-weight:600;">'
                   f"{param_html(rate)} = {_value_html(value)}</span></p>")

    rows = []
    for name in names:
        value, error = params.get(name), errors.get(name)
        fixed = error is None or not _finite(error)
        shown = rx.format_param_value(name, value, None if fixed else error)
        notes = []
        if fixed:
            notes.append("fixed")
        if name in fractions:
            notes.append(f"{fractions[name] * 100:.1f}% of total")
        note = _muted(_e(", ".join(notes)), muted) if notes else ""
        rows.append(f'<tr><td style="padding-right:14px;">{param_html(name)}</td>'
                    f'<td style="padding-right:12px;">{_value_html(shown)}</td>'
                    f"<td>{note}</td></tr>")
    if rows:
        out.append(_section("Parameters"))
        out.append(f'<table cellspacing="0" cellpadding="1" style="margin-bottom:4px;">{"".join(rows)}</table>')

    if fractions:
        out.append(_section("Relative amplitudes (population weights)"))
        rows = []
        for name in [n for n in names if n in fractions]:
            paired = rx._paired_rate_param(name, names)
            rate_cell = ""
            if paired is not None:
                rate_cell = (f"{param_html(paired)} = "
                             f"{_value_html(rx.format_param_value(paired, params.get(paired)))}")
            rows.append(f'<tr><td style="padding-right:14px;">{param_html(name)}</td>'
                        f'<td align="right" style="padding-right:12px; font-weight:600;">'
                        f"{fractions[name] * 100:.2f}%</td><td>{rate_cell}</td></tr>")
        out.append(f'<table cellspacing="0" cellpadding="1" style="margin-bottom:4px;">{"".join(rows)}</table>')
        if any(share < 0 or share > 1 for share in fractions.values()):
            out.append(_warning("A share outside 0–100% means the components have opposite "
                                "signs, which usually indicates more components than the "
                                "data supports.", warn))

    out.append(_section("Fit quality"))
    t = fit.get("t")
    metrics = _quality_metrics(quality, None if t is None else len(t))
    fraction = quality.get("decay_fraction")
    if fraction is None:
        fraction = rx.decay_fraction(fit.get("y") if fit.get("y") is not None else [])
    if _finite(fraction):
        metrics.append(("final <i>I</i>/<i>I</i>₀",
                        _e(f"{float(fraction):.4g} ({float(fraction) * 100:.1f}%)")))
    out.append(_metrics_html(metrics, muted))
    out.append(_uncertainty_html(uncertainty, muted))
    return "".join(out)


def global_report_html(result: dict, muted: str = MUTED, warn: str = WARN) -> str:
    """A :func:`~opennmr.processing.relaxation.fit_relaxation_global`
    result: the shared parameters as the headline, a row per curve for
    the ones left local, then the joint quality row."""
    from opennmr.processing import relaxation as rx

    quality = result.get("fit_quality") or {}
    shared = result.get("shared") or {}
    sub = [model_name(result.get("model")),
           f"{quality.get('n_datasets', len(result.get('per_dataset') or []))} curves",
           f"{quality.get('n_points', '?')} points",
           f"{len(shared)} shared parameter{'s' if len(shared) != 1 else ''}"]
    out = [_title_html("Global curve fit", sub, muted)]

    if shared:
        errors = result.get("shared_errors") or {}
        lines = [f"{param_html(n)} = {_value_html(rx.format_param_value(n, v, errors.get(n)))}"
                 for n, v in shared.items()]
        out.append('<p style="margin-top:2px; margin-bottom:6px;">'
                   '<span style="font-size:x-large; font-weight:600;">'
                   + "<br>".join(lines) + "</span><br>"
                   + _muted("shared across every curve", muted) + "</p>")
    else:
        out.append(f"{_P}{_muted('No shared parameters — every curve fitted independently.', muted)}</p>")

    local = list(result.get("local_names") or [])
    per = list(result.get("per_dataset") or [])
    if local and per:
        out.append(_section("Per curve"))
        head = "".join(f'<th align="left" style="color:{muted}; font-weight:600; '
                       f'padding-right:12px;">{h}</th>'
                       for h in ["Curve"] + [param_html(n) for n in local]
                       + ["final <i>I</i>/<i>I</i>₀"])
        rows = []
        for k, entry in enumerate(per):
            shares = rx.amplitude_fractions(entry.get("params") or {})
            cells = [f'<td style="padding-right:12px;"><b>{_e(entry.get("label") or k + 1)}</b></td>']
            for n in local:
                if n in shares:
                    text = f"{shares[n] * 100:.1f}%"
                else:
                    text = rx.format_param_value(n, (entry.get("params") or {}).get(n),
                                                 (entry.get("errors") or {}).get(n))
                cells.append(f'<td style="padding-right:12px;">{_value_html(text)}</td>')
            frac = (entry.get("fit_quality") or {}).get("decay_fraction")
            cells.append(f"<td>{_e(f'{float(frac):.3g}') if _finite(frac) else '—'}</td>")
            rows.append("<tr>" + "".join(cells) + "</tr>")
        out.append(f'<table cellspacing="0" cellpadding="1"><tr>{head}</tr>{"".join(rows)}</table>')

    out.append(_section("Fit quality (joint, over every curve)"))
    out.append(_metrics_html(_quality_metrics(quality), muted))
    out.append(_uncertainty_html(result.get("uncertainty") or {}, muted))
    return "".join(out)


def peaks_summary_html(results: list[dict], muted: str = MUTED) -> str:
    """Several peaks' curve fits as one table: the rate (or, for a
    model with no single rate, every parameter, amplitudes as shares)
    and R² per peak."""
    from opennmr.processing import relaxation as rx

    if not results:
        return ""
    out = [_title_html(f"{len(results)} peaks", [model_name(results[0].get("model"))], muted)]
    rows = []
    for result in results:
        params, errors = result.get("params") or {}, result.get("errors") or {}
        rate = result.get("rate_param")
        if rate is not None and rate in params:
            value = (f"{param_html(rate)} = "
                     f"<b>{_value_html(rx.format_param_value(rate, params[rate], errors.get(rate)))}</b>")
        else:
            shares = rx.amplitude_fractions(params)
            value = ", ".join(
                f"{param_html(n)} = " + (f"{shares[n] * 100:.1f}%" if n in shares
                                         else _value_html(rx.format_param_value(n, params[n])))
                for n in result.get("param_names") or params)
        r2 = (result.get("fit_quality") or {}).get("r_squared")
        r2_cell = _muted(f"<i>R</i>² {float(r2):.4f}", muted) if _finite(r2) else ""
        rows.append(f'<tr><td style="padding-right:12px;"><b>{_e(result.get("peak_label", "?"))}</b>'
                    f'</td><td style="padding-right:12px;">{value}</td><td>{r2_cell}</td></tr>')
    out.append(f'<table cellspacing="0" cellpadding="1" style="margin-bottom:4px;">{"".join(rows)}</table>')
    short = (results[0].get("uncertainty") or {}).get("short")
    if short:
        out.append(f'<p style="margin-top:4px; margin-bottom:0px;">'
                   f"{_muted(_e(f'± is {short}.'), muted)}</p>")
    return "".join(out)


def curve_fit_summary_html(results: list[dict], global_result: dict | None = None,
                           note: str = "", provenance: str = "", muted: str = MUTED,
                           warn: str = WARN) -> str:
    """Dynamics' results read-out above the per-peak report: a warning
    about the x-values first if there is one, the global / single /
    several-peak layout, then where the x-values came from."""
    out = []
    if note:
        out.append(_warning(str(note), warn))
    if global_result is not None:
        out.append(global_report_html(global_result, muted=muted, warn=warn))
    elif len(results or []) == 1:
        out.append(relaxation_report_html(results[0], muted=muted, warn=warn))
    elif results:
        out.append(peaks_summary_html(results, muted=muted))
    if provenance:
        out.append(f'<p style="margin-top:6px; margin-bottom:0px;">{_muted(_e(provenance), muted)}</p>')
    return "".join(out)


def score_report_html(result: dict, muted: str = MUTED, warn: str = WARN,
                      footnote: str = "") -> str:
    """A :func:`~opennmr.processing.score.score` result laid out like the
    curve fit's read-out: the method and model as the title, a row per
    component (its share, its decay constant and any other parameter its
    model fits), the fit-quality row -- R², RMSE, the residual as a share
    of the signal, the worst slice, cross-talk -- then what the numbers
    can and cannot say, and any warning. The same content as
    :func:`~opennmr.processing.score.format_score_report`."""
    import numpy as np

    from opennmr.processing import relaxation as rx
    from opennmr.processing import score as sc

    method = result.get("method", "score")
    name = sc.RESOLUTION_METHODS[method]["label"].split(" (")[0]
    n = int(result["n_components"])
    sub = [model_name(result.get("model")), f"{n} component{'s' if n != 1 else ''}"]
    out = [_title_html(name, sub, muted)]
    if result.get("model_note"):
        out.append(f"{_P}{_muted(_e(result['model_note']), muted)}</p>")

    diffusion = result.get("kind") == "diffusion"
    rate_symbol = "<i>D</i>" if diffusion else "<i>T</i>"
    extra = [p for p in result.get("model_params", []) if p != result.get("rate_param")]
    head = "".join(f'<th align="left" style="color:{muted}; font-weight:600; '
                   f'padding-right:12px;">{h}</th>'
                   for h in ["Component", "Share", rate_symbol] + [param_html(p) for p in extra])
    rows = []
    fractions = result.get("fractions")
    params = result.get("component_params") or [{}] * n
    for index in range(n):
        if diffusion:
            value = rx.format_diffusion_value(result["constants"][index])
        else:
            value = rx.format_time_value(result["T"][index])
        share = float(fractions[index]) if fractions is not None else float("nan")
        cells = [f'<td style="padding-right:12px;"><b>{index + 1}</b></td>',
                 f'<td align="right" style="padding-right:12px; font-weight:600;">'
                 f"{share * 100:.1f}%</td>" if _finite(share) else "<td>—</td>",
                 f'<td style="padding-right:12px;">{_value_html(value)}</td>']
        for p in extra:
            v = (params[index] if index < len(params) else {}).get(p)
            cells.append(f'<td style="padding-right:12px;">'
                         f"{_e(f'{float(v):.4g}') if _finite(v) else '—'}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    out.append(_section("Components"))
    out.append(f'<table cellspacing="0" cellpadding="1" style="margin-bottom:4px;">'
               f"<tr>{head}</tr>{''.join(rows)}</table>")

    out.append(_section("Fit quality"))
    metrics = []
    if _finite(result.get("r_squared")):
        metrics.append(("<i>R</i>²", f"{float(result['r_squared']):.5f}"))
    if _finite(result.get("rmse")):
        metrics.append(("RMSE", _e(f"{float(result['rmse']):.4g}")))
    if _finite(result.get("relative_rmse")):
        metrics.append(("residual / signal", _e(f"{float(result['relative_rmse']) * 100:.2f}%")))
    by_level = np.asarray(result.get("residual_by_level", []), dtype=float)
    worst = result.get("worst_level")
    if by_level.size and worst is not None:
        rows_used = result.get("rows")
        number = (int(rows_used[worst]) + 1 if rows_used is not None and worst < len(rows_used)
                  else int(worst) + 1)
        metrics.append(("worst slice", _e(f"{number} ({by_level[worst] * 100:.2f}%)")))
    if _finite(result.get("crosstalk")):
        metrics.append(("cross-talk", _e(f"{float(result['crosstalk']):.4f}")))
    columns = result.get("columns")
    if columns is not None:
        metrics.append(("points fitted", _e(f"{len(columns)} (≥ {result['threshold']:.0%} of max)")))
    out.append(_metrics_html(metrics, muted))
    optimised = "residual" if method == "score" else "cross-talk"
    caveat = (f"This run minimised the {optimised}; the other numbers are reported for "
              "comparison, not as independent corroboration. The number of components was "
              "an input: a good R² says this many fit, never that it is the right number – "
              "compare one more and one fewer, and look at the residual (tick Residual in "
              "the component list).")
    out.append(f"{_P}{_muted(_e(caveat), muted)}</p>")
    for warning in result.get("warnings", []):
        out.append(_warning(str(warning), warn))
    if footnote:
        out.append(f"{_P}{_muted(_e(footnote), muted)}</p>")
    return "".join(out)
