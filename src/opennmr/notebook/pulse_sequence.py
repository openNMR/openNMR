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

Supported files: Bruker, Varian/Agilent, JEOL Delta (``.jxp``), X-Pulse and
Magritek (Prospa ``*_pp.mac``). See :mod:`opennmr.io.pulse_sequence`.
"""

from __future__ import annotations

import math
import os
import re

from ..io.pulse_sequence import (
    Event, Loop, PulseSequence, PulseSequenceError, compute_layout, detect_vendor,
    format_time, parse_pulse_sequence, read_pulse_sequences, zgoptns_defines, KINDS,
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
    "pulse_label",
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
    (italic symbols with subscripts, as in a paper); anything else is
    returned as plain text."""
    if not name:
        return ""
    m = re.fullmatch(r"([0-9.]*)([A-Za-z]+)(\d*)", name)
    if not m or len(m.group(2)) > 6:
        return name.replace("$", "")
    pre, letters, digits = m.groups()
    sym = _GREEK.get(letters)
    body = sym if sym else rf"\mathit{{{letters}}}"
    if digits:
        body += f"_{{{digits}}}"
    return f"{pre}${body}$"


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

_ROW_H = 1.9
_H_PULSE, _H_DEC, _H_GRAD, _H_ACQ = 1.0, 0.34, 0.8, 0.55


def figure_size(seq: PulseSequence, show_minor_delays: bool = False,
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
                        show_minor_delays: bool = False, show_loops: bool = True,
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
        Include housekeeping delays (``30u``, ``de``, hardware set-up).
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Rectangle
    import numpy as np

    col = {**DEFAULT_COLORS, **(colors or {})}
    lay = compute_layout(seq, show_minor_delays)
    chans = seq.channels or ["f1"]
    nrows = len(chans)
    row_y = {c: -(i * _ROW_H) for i, c in enumerate(chans)}
    delay_top = -(nrows - 1) * _ROW_H - 0.9
    lane_h = 0.95
    bottom = delay_top - lay.n_lanes * lane_h - 0.5
    loop_depth = max([lp.depth for lp, _, _ in lay.loops] + [-1]) + 1 if show_loops else 0
    top = _H_PULSE + 0.9 + 0.5 * loop_depth

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize or figure_size(seq, show_minor_delays, show_loops))
    else:
        fig = ax.figure
    ax.set_axis_off()
    ax.set_xlim(-2.2, lay.width + 0.6)
    ax.set_ylim(bottom - 0.4, top + (0.6 if title else 0.1))

    def named(e) -> bool:
        if show_names is not None:
            return e.uid in set(show_names)
        return show_delay_names if e.kind == "delay" else show_pulse_names

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

    # delays: thin brackets on lanes under the last channel
    dl_end: dict = {}
    for e in sorted(seq.events, key=lambda ev: lay.x0[ev.uid]):
        if e.kind != "delay":
            continue
        if e.minor and not show_minor_delays:
            continue
        x0, x1 = lay.x0[e.uid], lay.x1[e.uid]
        y = delay_top - lay.lane.get(e.uid, 0) * lane_h
        c = col["delay"]
        ax.plot([x0, x1], [y, y], color=c, lw=0.9, zorder=2, solid_capstyle="butt")
        for xx in (x0, x1):
            ax.plot([xx, xx], [y - 0.1, y + 0.1], color=c, lw=0.9, zorder=2)
        if named(e) and e.label:
            txt = pulse_label(e.label)
            if show_values and e.value is not None:
                txt += f"  {format_time(e.value)}"
            est = 0.115 * len(e.label) * fontsize / 10.0 + 0.1
            xm = 0.5 * (x0 + x1)
            lane = lay.lane.get(e.uid, 0)
            lvl = 0
            if dl_end.get(lane, (-9e9, 0))[0] > xm - est / 2:
                lvl = 1 - dl_end[lane][1]
            dl_end[lane] = (xm + est / 2, lvl)
            ax.text(xm, y - 0.14 - 0.42 * lvl, txt, ha="center", va="top",
                    fontsize=fontsize, color=c, zorder=5)
    if lay.n_lanes:
        ax.text(-0.35, delay_top, "Delays", ha="right", va="center", fontsize=fontsize, color="#222")

    # loops
    if show_loops:
        for lp, lx0, lx1 in lay.loops:
            y = top - 0.55 - 0.5 * lp.depth
            c = "#555"
            ax.plot([lx0, lx0, lx1, lx1], [y - 0.14, y, y, y - 0.14], color=c, lw=1.0, zorder=4)
            cnt = f"{lp.n}" if lp.n else (pulse_label(lp.count) if lp.count else "n")
            ax.text(lx1 + 0.06, y, f"×{cnt}", ha="left", va="center", fontsize=fontsize * 0.9, color=c)

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
    return fig, ax
