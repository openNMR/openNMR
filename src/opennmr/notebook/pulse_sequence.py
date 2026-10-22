"""
Read and draw pulse sequences -- the notebook face of the GUI's
**Pulse sequence** viewer.

    from opennmr.notebook import pulse_sequence as ps

    seqs = ps.read_pulse_sequences("hahnecho3.enb")           # file on disk
    seqs = ps.pulse_sequence_for(spectrum)                    # an open experiment
    fig, ax = ps.plot_pulse_sequence(seqs[0])

Hard pulses are rectangles, shaped pulses a rounded half-ellipse,
decoupling a short rectangle, gradients and the acquisition their own
marks; every kind has its own colour (``colors=``). Pulse names
(``p1``, ``p12``) and delay names (``d1``, ``tau``) can be switched on and
off together (``show_pulse_names``, ``show_delay_names``) or one by one
(``show_names``). The drawing is a **schematic -- not to scale**: long
delays are compressed so every pulse stays visible.

Delays are drawn **among the pulses** (``delays_inline=True``, the default):
brackets just under the first channel, so one reads along the sequence in
time order. Housekeeping delays (recycle delay excepted, which is always
shown) are hidden by default and added one by one with ``show_minor_delays``
(``True`` for all, or a collection of their uids -- see
:func:`housekeeping_delays`). Calculated delays (``d20-p1``) are shown with
their expression. :func:`enable_zoom_pan` adds scroll-zoom / drag-pan to a
figure drawn outside the GUI.

Supported files: Bruker, Varian/Agilent, JEOL Delta (``.jxp``), X-Pulse and
Magritek (Prospa ``*_pp.mac``). See :mod:`opennmr.io.pulse_sequence`.
"""

from __future__ import annotations

import math
import os
import re

from ..io.pulse_sequence import (
    Event, Loop, PulseSequence, PulseSequenceError, compute_layout, detect_vendor,
    format_time, parse_pulse_sequence, read_pulse_sequences, shown_minor_uids,
    zgoptns_defines, KINDS,
)

#: Colour-blind-safe (Okabe-Ito) defaults, one per kind of event.
DEFAULT_COLORS = {
    "hard": "#0072B2",        # blue
    "shaped": "#D55E00",      # vermilion
    "decoupling": "#009E73",  # green
    "gradient": "#CC79A7",    # reddish purple
    "acquire": "#4D4D4D",     # grey
    "delay": "#333333",       # delay brackets
}

KIND_NAMES = {
    "hard": "Hard pulse", "shaped": "Shaped pulse",
    "decoupling": "Decoupling / recoupling", "gradient": "Gradient",
    "acquire": "Acquisition", "delay": "Delay",
}

_GREEK = {"tau": r"\tau", "delta": r"\delta", "Delta": r"\Delta", "gamma": r"\gamma",
          "theta": r"\theta", "alpha": r"\alpha", "beta": r"\beta", "phi": r"\phi"}
_SUPER = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")

__all__ = [
    "DEFAULT_COLORS", "KIND_NAMES", "PulseSequence", "PulseSequenceError",
    "read_pulse_sequences", "parse_pulse_sequence", "pulse_sequence_for",
    "find_sequence_file", "plot_pulse_sequence", "figure_size", "event_table", "channel_nuclei",
    "pulse_label", "enable_zoom_pan", "housekeeping_delays", "DELAY_STYLES",
]


# ---------------------------------------------------------------------------
# finding / reading the sequence of an open experiment
# ---------------------------------------------------------------------------

def find_sequence_file(spectrum) -> str | None:
    """Path of the pulse-sequence source belonging to ``spectrum``:
    a file the user chose (``metadata.raw['pulse_sequence_file']``) or,
    for Bruker data, the ``pulseprogram`` kept in the experiment folder.
    ``None`` when the data do not carry it (Varian, JEOL, X-Pulse and
    Spinsolve keep only the sequence *name*)."""
    meta = getattr(spectrum, "metadata", None)
    raw = getattr(meta, "raw", None) or {}
    chosen = raw.get("pulse_sequence_file")
    if chosen and os.path.isfile(chosen):
        return chosen
    sp = getattr(meta, "source_path", None)
    if sp:
        here = str(sp)
        for d in (here, os.path.dirname(here), os.path.dirname(os.path.dirname(here))):
            cand = os.path.join(d, "pulseprogram")
            if os.path.isfile(cand):
                return cand
    return None


def _nuc_label(nuc) -> str:
    m = re.fullmatch(r"(\d+)([A-Za-z]+)", str(nuc or "").strip())
    return f"{m.group(1).translate(_SUPER)}{m.group(2)}" if m else str(nuc or "")


def channel_nuclei(spectrum) -> dict:
    """``{'f1': '¹H', 'f2': '¹³C', ...}`` from the acquisition parameters
    (Bruker ``NUC1..NUC8``, Varian ``tn``/``dn``/``dn2``); empty if unknown."""
    raw = getattr(getattr(spectrum, "metadata", None), "raw", None) or {}
    out = {}
    acqus = raw.get("acqus") or {}
    for k in range(1, 5):
        v = acqus.get(f"NUC{k}")
        if v and str(v).lower() != "off":
            out[f"f{k}"] = _nuc_label(v)
    if not out:
        meta = getattr(spectrum, "metadata", None)
        nuc = getattr(meta, "nucleus", None)
        if nuc:
            out["f1"] = _nuc_label(nuc)
    return out


def pulse_sequence_for(spectrum, path: str | None = None, flags: dict | None = None) -> list:
    """The pulse sequence(s) of an open experiment, with its own parameter
    values (``d1``, ``p1``, ``cnst``...) filled in where the vendor records
    them. ``path`` overrides the lookup (see :func:`find_sequence_file`)."""
    from ..core.acquisition import acquisition_constants
    path = path or find_sequence_file(spectrum)
    if not path:
        name = getattr(getattr(spectrum, "metadata", None), "pulse_program", None)
        raise PulseSequenceError(
            f"This experiment does not carry its pulse-sequence source"
            f"{f' (only its name, {name!r})' if name else ''}. "
            "Open the sequence file with read_pulse_sequences(path).")
    raw = getattr(getattr(spectrum, "metadata", None), "raw", None) or {}
    defines = zgoptns_defines((raw.get("acqus") or {}).get("ZGOPTNS"))
    try:
        consts = acquisition_constants(spectrum)
    except Exception:
        consts = {}
    seqs = read_pulse_sequences(path, constants=consts, flags=flags, defines=defines)
    nuclei = channel_nuclei(spectrum)
    for s in seqs:
        s.channel_nuclei = dict(nuclei)
    return seqs


# ---------------------------------------------------------------------------
# labels / table
# ---------------------------------------------------------------------------

def pulse_label(name: str) -> str:
    """``'p1'`` -> ``'$p_{1}$'``, ``'2pw'`` -> ``'2$pw$'``, ``'tau'`` -> ``'$\\tau$'``
    (italic symbols with subscripts, as in a paper). A **calculated** value is
    set the same way piece by piece -- ``'d20-p1'`` -> ``'$d_{20}$−$p_{1}$'``,
    ``'p1*0.33'`` -> ``'$p_{1}$·0.33'``; anything else is returned as plain text."""
    if not name:
        return ""
    m = re.fullmatch(r"([0-9.]*)([A-Za-z]+)(\d*)", name)
    if m and len(m.group(2)) <= 6:
        pre, letters, digits = m.groups()
        sym = _GREEK.get(letters)
        body = sym if sym else rf"\mathit{{{letters}}}"
        if digits:
            body += f"_{{{digits}}}"
        return f"{pre}${body}$"
    if re.search(r"[-+*/]", name):
        out = []
        for part in re.split(r"((?<![\w.])[A-Za-z_]\w*)", name.replace("$", "")):
            if re.fullmatch(r"[A-Za-z]{1,6}\d*", part):
                out.append(pulse_label(part))
            else:
                out.append(part.replace("*", "·").replace("-", "−"))
        return "".join(out)
    return name.replace("$", "")


def housekeeping_delays(seq: PulseSequence) -> list:
    """The delays that only set the hardware up (``30u``, ``1m``, the dead
    time ``de``...), in time order -- each can be shown or hidden on its own
    via ``show_minor_delays=[uid, ...]`` of :func:`plot_pulse_sequence`."""
    return [e for e in seq.events if e.kind == "delay" and e.minor]


def event_table(seq: PulseSequence) -> list:
    """One dict per event -- for a log, a table or a CSV."""
    rows = []
    for e in seq.events:
        rows.append({"#": e.uid, "kind": KIND_NAMES.get(e.kind, e.kind), "channel": e.channel,
                     "name": e.label, "expression": e.expr, "duration": e.value_text(),
                     "shape": e.shape or "", "phase": e.phase or "",
                     "housekeeping": e.minor})
    return rows


# ---------------------------------------------------------------------------
# drawing
# ---------------------------------------------------------------------------

DELAY_STYLES = ("line", "brackets", "block")
_ROW_H = 1.9
_H_PULSE, _H_DEC, _H_GRAD, _H_ACQ = 1.0, 0.34, 0.8, 0.55


def figure_size(seq: PulseSequence, show_minor_delays=False,
                show_loops: bool = True) -> tuple:
    """A good ``figsize`` (inches) for ``seq`` -- wide enough for its
    pulses, tall enough for its channels, delay lanes and loop brackets."""
    lay = compute_layout(seq, show_minor_delays)
    nrows = max(len(seq.channels), 1)
    depth = (max([lp.depth for lp, _, _ in lay.loops] + [-1]) + 1) if show_loops else 0
    height = 0.55 * (nrows * _ROW_H + 0.95 * lay.n_lanes + 1.6 + 0.5 * depth) + 0.6
    return (min(max(6.0, 0.62 * lay.width + 2.0), 16.0), height)


def plot_pulse_sequence(seq: PulseSequence, ax=None, *, colors: dict | None = None,
                        show_pulse_names: bool = True, show_delay_names: bool = True,
                        show_names=None, show_values: bool = False,
                        show_minor_delays=False, show_loops: bool = True,
                        delay_style: str | None = None, delays_inline: bool | None = None,
                        legend: bool = True, title: bool = True,
                        figsize=None, fontsize: float = 10.0):
    """Draw ``seq`` as a pulse-sequence diagram. Returns ``(fig, ax)``.

    Parameters
    ----------
    colors
        ``{'hard': '#0072B2', 'shaped': ..., 'decoupling': ..., 'gradient': ...,
        'acquire': ..., 'delay': ...}`` -- any subset; see :data:`DEFAULT_COLORS`.
    show_pulse_names, show_delay_names
        Show every pulse / every delay name, or none.
    show_names
        Fine control: a collection of event ``uid`` values (``event.uid``,
        the ``#`` column of :func:`event_table`) whose names are shown;
        everything else is unlabelled. Overrides the two flags above.
    show_values
        Add the duration under a name where the file records it.
    show_minor_delays
        Kept for old code and ignored: every delay, however short, is an
        ordinary delay (switch single ones off with ``show_names``).
    delay_style
        ``"line"`` (the default): each delay's name is written **on the
        first channel's line**, in its gap between the pulses (``d6``,
        ``30u``, ``tau``), the way TopSpin does. ``"brackets"``: a bracket
        per delay on lanes right under that line. ``"block"``: the brackets
        in a block of their own below every channel. A delay whose name is
        switched off is not drawn at all -- with no delay names there is no
        delay row.
    delays_inline
        Old spelling: ``True`` is ``delay_style="brackets"``, ``False``
        is ``"block"``.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Rectangle
    import numpy as np

    col = {**DEFAULT_COLORS, **(colors or {})}
    style = delay_style or ("brackets" if delays_inline else "block" if delays_inline is False
                            else "line")
    if style not in DELAY_STYLES:
        raise ValueError(f"delay_style must be one of {DELAY_STYLES}, got {style!r}.")

    def named(e) -> bool:
        if show_names is not None:
            return e.uid in set(show_names)
        return show_delay_names if e.kind == "delay" else show_pulse_names

    def delay_text(e):
        """(drawn text, plain text for width) of a delay's name, or None."""
        name = e.name
        if not name or not named(e):
            return None
        txt = pulse_label(e.label) if e.label else name
        plain = name
        if style != "line" and show_values and e.value is not None and e.label:
            extra = f"  {format_time(e.value)}"
            txt, plain = txt + extra, plain + extra
        return txt, plain

    def text_width(plain: str, size: float = fontsize) -> float:
        return 0.115 * len(plain) * size / 10.0 + 0.2

    shown_delays = [e for e in seq.events if e.kind == "delay" and delay_text(e) is not None]
    # In "line" style the name sits in the delay's own gap: stretch short gaps
    # so it fits between the neighbouring pulses.
    min_widths = ({e.uid: text_width(delay_text(e)[1]) + 0.1 for e in shown_delays}
                  if style == "line" else None)
    lay = compute_layout(seq, False, min_widths)
    shown_uids = {e.uid for e in shown_delays}

    # Delay names are staggered onto up to three levels so neighbours never
    # overlap; the lane is made tall enough for the levels actually used.
    delay_level: dict = {}
    lane_ends: dict = {}
    for e in sorted((e for e in shown_delays), key=lambda ev: lay.x0[ev.uid]):
        t = delay_text(e)
        est = text_width(t[1])
        xm = 0.5 * (lay.x0[e.uid] + lay.x1[e.uid])
        ends = lane_ends.setdefault(lay.lane[e.uid] if style != "line" else 0, [-9e9, -9e9, -9e9])
        level = next((k for k in range(3) if ends[k] <= xm - est / 2), ends.index(min(ends)))
        ends[level] = xm + est / 2
        delay_level[e.uid] = level
    n_levels = max(delay_level.values(), default=0) + 1
    chans = seq.channels or ["f1"]
    nrows = len(chans)
    lane_h = 0.95 + 0.42 * max(0, n_levels - 2)
    has_delays = bool(shown_delays) and style != "line"
    n_lanes = (max(lay.lane[e.uid] for e in shown_delays) + 1) if has_delays else 0
    if style == "line":
        row_y = {c: -(i * _ROW_H) for i, c in enumerate(chans)}
        delay_top = row_y[chans[0]]
        bottom = row_y[chans[-1]] - 0.9
    elif style == "brackets":
        # The delay brackets sit right under the first channel's line, among
        # the pulses; the other channels are pushed down to make room.
        block = (n_lanes * lane_h + 0.6) if has_delays else 0.0
        row_y = {c: -(i * _ROW_H) - (block if i > 0 else 0.0) for i, c in enumerate(chans)}
        delay_top = row_y[chans[0]] - 0.9
        bottom = row_y[chans[-1]] - 0.9
    else:
        row_y = {c: -(i * _ROW_H) for i, c in enumerate(chans)}
        delay_top = -(nrows - 1) * _ROW_H - 0.9
        bottom = delay_top - n_lanes * lane_h - 0.5
    loop_depth = max([lp.depth for lp, _, _ in lay.loops] + [-1]) + 1 if show_loops else 0
    top = _H_PULSE + 0.9 + 0.5 * loop_depth

    if ax is None:
        if figsize is None:
            depth = (max([lp.depth for lp, _, _ in lay.loops] + [-1]) + 1) if show_loops else 0
            figsize = (min(max(6.0, 0.62 * lay.width + 2.0), 24.0),
                       0.55 * (nrows * _ROW_H + 0.95 * n_lanes + 1.6 + 0.5 * depth) + 0.6)
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    ax.set_axis_off()
    # room on the right for the longest loop label ("×count"), which hangs
    # off the end of its bracket
    right = lay.width + 0.6
    if show_loops:
        for lp, _lx0, lx1 in lay.loops:
            cnt = f"{lp.n}" if lp.n else (lp.count if lp.count else "n")
            right = max(right, lx1 + 0.12 + 0.115 * (len(cnt) + 1) * fontsize * 0.9 / 10.0)
    ax.set_xlim(-2.2, right + 0.1)
    ax.set_ylim(bottom - 0.4, top + (0.6 if title else 0.1))

    # channel baselines and labels
    nuc = getattr(seq, "channel_nuclei", {}) or {}
    for c in chans:
        y = row_y[c]
        ax.plot([0, lay.width], [y, y], color="#9a9a9a", lw=0.8, zorder=1, solid_capstyle="butt")
        if c == "G":
            lab = "Gradient"
        elif c == "ACQ":
            lab = "Acquire"
        else:
            lab = f"{nuc[c]}" if c in nuc else c
            lab = lab if c not in nuc else f"{lab}  ({c})"
        ax.text(-0.35, y + 0.18, lab, ha="right", va="center", fontsize=fontsize, color="#222")

    last_end: dict = {}

    def put_label(e, x_mid, y_base, h, fill):
        if not named(e) or not e.label or e.kind == "acquire":
            return
        txt = pulse_label(e.label)
        if show_values and e.value is not None:
            txt += "\n" + format_time(e.value)
        level = 0
        key = e.channel
        est = 0.115 * len(e.label) * fontsize / 10.0 + 0.1
        if last_end.get(key, (-9e9, 0))[0] > x_mid - est / 2:
            level = 1 - last_end[key][1]
        last_end[key] = (x_mid + est / 2, level)
        ax.text(x_mid, y_base + h + 0.1 + 0.5 * level, txt, ha="center", va="bottom",
                fontsize=fontsize, color=fill, zorder=5, linespacing=1.0)

    for e in seq.events:
        if e.kind == "delay":
            continue
        x0, x1 = lay.x0[e.uid], lay.x1[e.uid]
        y = row_y.get(e.channel, 0.0)
        c = col.get(e.kind, "#444")
        xm = 0.5 * (x0 + x1)
        if e.kind == "hard":
            ax.add_patch(Rectangle((x0, y), max(x1 - x0, 0.05), _H_PULSE, fc=c, ec=c, lw=0.8, zorder=3))
            put_label(e, xm, y, _H_PULSE, c)
        elif e.kind == "shaped":
            n = 60
            xs = np.linspace(x0, x1, n)
            a = max((x1 - x0) / 2, 1e-6)
            ys = y + _H_PULSE * np.sqrt(np.clip(1 - ((xs - xm) / a) ** 2, 0, 1))
            ax.add_patch(Polygon(np.column_stack([np.r_[xs, xs[0]], np.r_[ys, y]]),
                                 closed=True, fc=c, ec=c, lw=0.8, zorder=3, joinstyle="round"))
            put_label(e, xm, y, _H_PULSE, c)
        elif e.kind == "decoupling":
            ax.add_patch(Rectangle((x0, y), max(x1 - x0, 0.05), _H_DEC, fc=c, ec=c, lw=0.8,
                                   alpha=0.9, zorder=2))
            put_label(e, xm, y, _H_DEC, c)
        elif e.kind == "gradient":
            ax.add_patch(Rectangle((x0, y), max(x1 - x0, 0.05), _H_GRAD, fc=c, ec=c, lw=0.8,
                                   alpha=0.9, zorder=2))
            put_label(e, xm, y, _H_GRAD, c)
        elif e.kind == "acquire":
            xs = np.linspace(x0, x1, 240)
            f = max((x1 - x0) * 2.2, 3.0)
            ys = y + _H_ACQ * np.exp(-3.0 * (xs - x0) / max(x1 - x0, 1e-6)) * np.cos(
                2 * math.pi * f * (xs - x0) / max(x1 - x0, 1e-6) * 2.0)
            ax.plot(xs, ys, color=c, lw=1.3, zorder=3)
            put_label(e, xm, y, _H_ACQ + 0.2, c)

    # delays
    dcol = col["delay"]
    if style == "line":
        # the name sits on the first channel's line, in the gap between pulses
        y0 = row_y[chans[0]]
        for e in shown_delays:
            t = delay_text(e)
            xm = 0.5 * (lay.x0[e.uid] + lay.x1[e.uid])
            ax.text(xm, y0 + 0.12 + 0.42 * delay_level.get(e.uid, 0), t[0], ha="center",
                    va="bottom", fontsize=fontsize, color=dcol, zorder=5)
            if show_values and e.value is not None and e.label:
                ax.text(xm, y0 - 0.12, format_time(e.value), ha="center", va="top",
                        fontsize=fontsize * 0.8, color=dcol, zorder=5)
    else:
        # thin brackets on lanes under the first channel (or, as a block, under the last)
        for e in sorted(shown_delays, key=lambda ev: lay.x0[ev.uid]):
            x0, x1 = lay.x0[e.uid], lay.x1[e.uid]
            y = delay_top - lay.lane.get(e.uid, 0) * lane_h
            ax.plot([x0, x1], [y, y], color=dcol, lw=0.9, zorder=2, solid_capstyle="butt")
            for xx in (x0, x1):
                ax.plot([xx, xx], [y - 0.1, y + 0.1], color=dcol, lw=0.9, zorder=2)
            t = delay_text(e)
            xm = 0.5 * (x0 + x1)
            ax.text(xm, y - 0.14 - 0.42 * delay_level.get(e.uid, 0), t[0], ha="center", va="top",
                    fontsize=fontsize, color=dcol, zorder=5)
        if has_delays:
            ax.text(-0.35, delay_top, "Delays", ha="right", va="center", fontsize=fontsize,
                    color="#222")

    # loops
    loop_texts: list = []
    if show_loops:
        for lp, lx0, lx1 in lay.loops:
            y = top - 0.55 - 0.5 * lp.depth
            c = "#555"
            ax.plot([lx0, lx0, lx1, lx1], [y - 0.14, y, y, y - 0.14], color=c, lw=1.0, zorder=4)
            cnt = f"{lp.n}" if lp.n else (pulse_label(lp.count) if lp.count else "n")
            loop_texts.append(ax.text(lx1 + 0.06, y, f"×{cnt}", ha="left", va="center",
                                      fontsize=fontsize * 0.9, color=c))
        _fit_texts_inside_right_edge(ax, loop_texts)

    if legend:
        present = [k for k in ("hard", "shaped", "decoupling", "gradient", "acquire")
                   if any(e.kind == k for e in seq.events)]
        if present:
            from matplotlib.patches import Patch
            ax.legend(handles=[Patch(fc=col[k], ec=col[k], label=KIND_NAMES[k]) for k in present],
                      loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=len(present),
                      frameon=False, fontsize=fontsize * 0.85, handlelength=1.2,
                      bbox_transform=ax.transAxes)
    ax.text(lay.width + 0.5, bottom - 0.3, "schematic – not to scale", ha="right", va="bottom",
            fontsize=fontsize * 0.75, style="italic", color="#777")
    if title:
        ax.set_title(seq.name, fontsize=fontsize * 1.15, loc="left")
    # The full view, for "reset" after scrolling / dragging (enable_zoom_pan).
    ax._opennmr_home_view = (ax.get_xlim(), ax.get_ylim())
    return fig, ax


def _fit_texts_inside_right_edge(ax, texts, pad_px: float = 6.0) -> None:
    """Widen the right of ``ax`` until every text in ``texts`` is fully
    inside it -- measured, not estimated, so a long loop count ("×count")
    is never cut off whatever the figure's size. No-op without a renderer."""
    if not texts:
        return
    try:
        renderer = ax.figure.canvas.get_renderer()
    except Exception:  # noqa: BLE001 -- a backend without one: keep the estimate
        return
    for _ in range(4):
        edge = ax.get_window_extent(renderer).x1
        over = max(t.get_window_extent(renderer).x1 for t in texts) + pad_px - edge
        if over <= 0:
            return
        x0, x1 = ax.get_xlim()
        per_px = (x1 - x0) / max(ax.get_window_extent(renderer).width, 1.0)
        ax.set_xlim(x0, x1 + over * per_px * 1.1)



# ---------------------------------------------------------------------------
# scrolling / zooming
# ---------------------------------------------------------------------------

def enable_zoom_pan(fig, *, factor: float = 1.25, busy=None, max_out: float = 1.6):
    """Let the mouse zoom and pan a pulse-sequence figure.

    * **scroll** zooms in / out under the cursor along the time axis;
      **Ctrl + scroll** zooms the vertical axis; **Shift + scroll** pans
      along time.
    * **drag** (left button) pans; **double-click** returns to the full view.

    The full view is the one :func:`plot_pulse_sequence` drew (it is kept on
    the axes), and zooming out never goes beyond ``max_out`` times it. Works
    in a notebook with an interactive backend (``%matplotlib widget`` /
    ``qt``) and is what the GUI's Pulse sequence window uses. ``busy`` is an
    optional callable returning True while another tool (a toolbar's own
    pan / zoom) owns the mouse. Returns a function that switches it off."""
    state = {"drag": None}

    def home(ax):
        return getattr(ax, "_opennmr_home_view", None) or (ax.get_xlim(), ax.get_ylim())

    def clamp(lo, hi, home_lo, home_hi):
        span, full = hi - lo, home_hi - home_lo
        if span > full * max_out:
            mid = 0.5 * (lo + hi)
            lo, hi = mid - full * max_out / 2, mid + full * max_out / 2
        return lo, hi

    def zoom_axis(ax, which, centre, scale):
        get, put = (ax.get_xlim, ax.set_xlim) if which == "x" else (ax.get_ylim, ax.set_ylim)
        lo, hi = get()
        new_lo, new_hi = centre - (centre - lo) * scale, centre + (hi - centre) * scale
        hx, hy = home(ax)
        h_lo, h_hi = hx if which == "x" else hy
        put(clamp(new_lo, new_hi, h_lo, h_hi))

    def on_scroll(event):
        ax = event.inaxes
        if ax is None or (busy and busy()):
            return
        scale = 1 / factor if event.button == "up" else factor
        key = (event.key or "")
        if "shift" in key:
            lo, hi = ax.get_xlim()
            step = (hi - lo) * 0.1 * (-1 if event.button == "up" else 1)
            ax.set_xlim(lo + step, hi + step)
        elif "control" in key or "ctrl" in key:
            zoom_axis(ax, "y", event.ydata, scale)
        else:
            zoom_axis(ax, "x", event.xdata, scale)
        fig.canvas.draw_idle()

    def on_press(event):
        if event.inaxes is None or (busy and busy()) or event.button != 1:
            return
        if getattr(event, "dblclick", False):
            hx, hy = home(event.inaxes)
            event.inaxes.set_xlim(hx)
            event.inaxes.set_ylim(hy)
            state["drag"] = None
            fig.canvas.draw_idle()
            return
        state["drag"] = (event.inaxes, event.x, event.y, event.inaxes.get_xlim(),
                         event.inaxes.get_ylim())

    def on_move(event):
        d = state["drag"]
        if d is None or event.x is None or (busy and busy()):
            return
        ax, x0, y0, xlim, ylim = d
        inv = ax.transData.inverted()
        (a0, b0), (a1, b1) = inv.transform((x0, y0)), inv.transform((event.x, event.y))
        ax.set_xlim(xlim[0] - (a1 - a0), xlim[1] - (a1 - a0))
        ax.set_ylim(ylim[0] - (b1 - b0), ylim[1] - (b1 - b0))
        # the limits just changed, so the inverse used above is stale: re-anchor
        state["drag"] = (ax, event.x, event.y, ax.get_xlim(), ax.get_ylim())
        fig.canvas.draw_idle()

    def on_release(_event):
        state["drag"] = None

    cids = [fig.canvas.mpl_connect("scroll_event", on_scroll),
            fig.canvas.mpl_connect("button_press_event", on_press),
            fig.canvas.mpl_connect("motion_notify_event", on_move),
            fig.canvas.mpl_connect("button_release_event", on_release)]

    def disable():
        for cid in cids:
            fig.canvas.mpl_disconnect(cid)
    return disable
