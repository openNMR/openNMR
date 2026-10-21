"""
Pulse-sequence source files -> a vendor-neutral event list.

The viewer (``notebook.pulse_sequence`` / the GUI pop-up) needs to know
*what the sequence does*, not how each vendor spells it. This module reads
the source of a pulse programme and returns a :class:`PulseSequence`: a
time-ordered list of :class:`Event` objects (hard pulses, shaped pulses,
decoupling, delays, gradients, acquisition) plus loop brackets.

Supported sources
-----------------
* **Bruker** TopSpin pulse programmes (``pulseprogram``, ``*.enb`` ...):
  the ``(p1 ph1):f1`` / ``(p6:sp6 ph3):f1`` syntax, delays (``d1``, ``1m``,
  ``30u``, ``vdlist``), ``cpd``/``do`` decoupling, ``p16:gp1`` gradients,
  ``go=`` acquisition, ``lo to N times lK`` loops and ``#ifdef`` blocks.
* **Varian / Agilent** ``pulsesequence()`` C files (``rgpulse``,
  ``dec2rgpulse``, ``sim3pulse``, ``shaped_pulse``, ``delay``, ``rgradient``,
  ``loop``/``endloop`` ...), with ``if(flag[0]=='y')`` branches.
* **JEOL Delta** ``.jxp`` experiment source (``begin ... end pulse``).
* **X-Pulse** Python sequences (``Transmit1``, ``Delay``, ``Gradient3`` ...,
  ``with sequential/parallel/repeat``), including a library file holding
  several ``# Name ~~~~`` chunks.
* **Magritek Spinsolve** Prospa ``*_pp.mac`` macros (``pulse``, ``wait``,
  ``delay``, ``Spoiler:*``, ``acquire``).

Everything here is standard-library only (no Qt, no matplotlib) so it can
be used from a notebook, a script or the GUI alike.

Timing
------
Events carry *nominal* start/end times (seconds). Where the real value is
known (from the file's own numbers, or from the ``constants`` passed in --
see :func:`opennmr.core.acquisition.acquisition_constants`) it is used;
otherwise a nominal duration for the kind of event. The drawing is a
schematic, so :func:`compute_layout` maps these onto roughly
logarithmic widths -- it is *not* to scale and says so.
"""

from __future__ import annotations

import ast
import math
import os
import re
from dataclasses import dataclass, field

KINDS = ("hard", "shaped", "decoupling", "delay", "gradient", "acquire")

#: Duration used for an event whose real length is not recorded (seconds).
NOMINAL = {"hard": 10e-6, "shaped": 1e-3, "decoupling": 1e-3, "delay": 1e-3,
           "gradient": 1e-3, "acquire": 50e-3}

#: Delays shorter than this that are *named* (e.g. ``de``) are treated as
#: housekeeping and hidden unless the user asks for them.
MINOR_NAMED_S = 20e-6
#: ... and literal delays (``30u``, ``1m``) up to this length.
MINOR_LITERAL_S = 1e-3


class PulseSequenceError(ValueError):
    """The file cannot be read as a pulse sequence (message is user-facing)."""


@dataclass
class Event:
    kind: str                     # one of KINDS
    channel: str                  # 'f1', 'f2', 'f3', 'G', 'ACQ' or 'delay'
    label: str = ""               # short name: p1, d1, tau ...
    expr: str = ""                # full expression as written
    t0: float = 0.0               # nominal start (s)
    t1: float = 0.0               # nominal end (s)
    value: float | None = None    # real duration (s) when known
    minor: bool = False           # housekeeping delay, hidden by default
    shape: str | None = None
    phase: str | None = None
    uid: int = -1

    @property
    def duration(self) -> float:
        return max(self.t1 - self.t0, 0.0)

    def value_text(self) -> str:
        return format_time(self.value)


@dataclass
class Loop:
    t0: float
    t1: float
    count: str = ""               # 'l20', '16', 'GARP_acq_blocks'
    n: int | None = None
    depth: int = 0


@dataclass
class PulseSequence:
    name: str
    vendor: str
    events: list = field(default_factory=list)
    loops: list = field(default_factory=list)
    source: str | None = None
    notes: list = field(default_factory=list)
    #: conditional blocks found in the source -> state used
    #: (``#ifdef dec`` / ``fsat == 'y'``); pass ``flags=`` to change them.
    flags: dict = field(default_factory=dict)
    channel_nuclei: dict = field(default_factory=dict)

    @property
    def channels(self) -> list:
        """Rows to draw, top to bottom (``f1`` first, ``G``, ``ACQ`` last)."""
        used = {e.channel for e in self.events if e.channel != "delay"}
        rf = sorted(c for c in used if re.fullmatch(r"f\d", c))
        out = rf + [c for c in ("G", "ACQ") if c in used]
        return out

    def of_kind(self, kind: str) -> list:
        return [e for e in self.events if e.kind == kind]

    def summary(self) -> str:
        counts = {k: len(self.of_kind(k)) for k in KINDS}
        parts = [f"{n} {k}" for k, n in counts.items() if n]
        return f"{self.name} ({self.vendor}): " + ", ".join(parts)


def format_time(value) -> str:
    """``1.5e-5`` -> ``'15 µs'``; ``None`` -> ``'—'``."""
    if value is None:
        return "—"
    if value == 0:
        return "0"
    a = abs(value)
    for scale, unit in ((1.0, "s"), (1e-3, "ms"), (1e-6, "µs"), (1e-9, "ns")):
        if a >= scale * 0.9995:
            return f"{value / scale:.4g} {unit}"
    return f"{value:.3g} s"


# ---------------------------------------------------------------------------
# timeline builder shared by every parser
# ---------------------------------------------------------------------------

class _Timeline:
    def __init__(self):
        self.t = 0.0
        self.events: list[Event] = []
        self.loops: list[Loop] = []
        self._open_dec: dict[str, Event] = {}
        self._open_grad: dict = {}
        self.recording = True

    def add(self, kind, channel, label="", *, expr="", dur=None, shape=None,
            phase=None, minor=False, advance=True) -> Event:
        d = dur if dur is not None else NOMINAL[kind]
        ev = Event(kind, channel, label, expr, self.t, self.t + d,
                   dur, minor, shape, phase)
        if self.recording:
            self.events.append(ev)
        if advance:
            self.t += d
        return ev

    def parallel(self, specs):
        """Start every spec at the current time; advance by the longest."""
        start, longest = self.t, 0.0
        for s in specs:
            self.t = start
            ev = self.add(advance=False, **s)
            longest = max(longest, ev.duration)
        self.t = start + longest

    def open_dec(self, channel, label="", expr="", at=None):
        if channel in self._open_dec:
            return
        ev = Event("decoupling", channel, label, expr,
                   self.t if at is None else at, math.nan)
        self._open_dec[channel] = ev
        if self.recording:
            self.events.append(ev)

    def close_dec(self, channel=None):
        for ch in ([channel] if channel else list(self._open_dec)):
            ev = self._open_dec.pop(ch, None)
            if ev is not None:
                ev.t1 = max(self.t, ev.t0 + NOMINAL["decoupling"] * 0.1)

    def loop(self, t0, count="", n=None):
        if self.recording and self.t > t0:
            self.loops.append(Loop(t0, self.t, str(count), n))

    def finish(self):
        self.close_dec()
        for ev in self.events:
            if isinstance(ev.t1, float) and math.isnan(ev.t1):
                ev.t1 = self.t


def _is_minor(label: str, value, literal: bool) -> bool:
    if value is None:
        return False
    return value <= MINOR_LITERAL_S if literal else value < MINOR_NAMED_S


def _label_of(expr: str) -> str:
    """Short display name for a duration/width expression:
    ``p1`` -> ``p1``; ``2*pw`` -> ``2pw``; ``taua - gt2 - 4e-6`` -> ``taua``;
    a bare number -> ``''`` (unnamed)."""
    e = expr.strip().strip("()")
    if not e:
        return ""
    m = re.fullmatch(r"([0-9.]+)\s*\*\s*([A-Za-z_][\w.]*)", e) or \
        re.fullmatch(r"([A-Za-z_][\w.]*)\s*\*\s*([0-9.]+)", e)
    if m:
        a, b = m.groups()
        if re.match(r"[0-9.]", a):
            k, name = a, b
        else:
            name, k = a, b
        k = k[:-2] if k.endswith(".0") else k
        return f"{k}{name.split('.')[-1]}"
    ids = re.findall(r"[A-Za-z_][\w.]*", re.sub(r"\d+\.?\d*[eE][-+]?\d+", "", e))
    ids = [i for i in ids if i not in ("e", "E")]
    return ids[0].split(".")[-1] if ids else ""


_SAFE_BIN = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b,
             ast.Mult: lambda a, b: a * b,
             ast.Div: lambda a, b: a / b if b else None,
             ast.Pow: lambda a, b: a ** b}


def _safe_eval(expr: str, names: dict):
    """Arithmetic over numbers and known names; ``None`` if anything is
    unknown (never raises, never executes code)."""
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError:
        return None

    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return float(n.value)
        if isinstance(n, ast.Name):
            v = names.get(n.id)
            return float(v) if isinstance(v, (int, float)) else None
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
            v = ev(n.operand)
            return None if v is None else (-v if isinstance(n.op, ast.USub) else v)
        if isinstance(n, ast.BinOp) and type(n.op) in _SAFE_BIN:
            a, b = ev(n.left), ev(n.right)
            if a is None or b is None:
                return None
            try:
                return _SAFE_BIN[type(n.op)](a, b)
            except (OverflowError, ZeroDivisionError):
                return None
        return None
    return ev(tree)


def _strip_c_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


# ---------------------------------------------------------------------------
# Bruker
# ---------------------------------------------------------------------------

_B_NUM = re.compile(r"^(\d*\.?\d+)\s*(u|m|n|s)?$", re.I)
_B_UNIT = {"u": 1e-6, "m": 1e-3, "n": 1e-9, "s": 1.0, None: 1.0, "": 1.0}
_B_LISTDELAY = re.compile(r"^(vd|vp|vc|vt|\w*list)$", re.I)


def zgoptns_defines(text) -> set:
    """``-Ddec -Dpresat`` (the ``ZGOPTNS`` parameter) -> ``{'dec','presat'}``."""
    return set(re.findall(r"-D(\w+)", text or ""))


def _bruker_value(tok, consts):
    """(seconds or None, is_literal) for a delay/pulse token."""
    m = _B_NUM.match(tok)
    if m:
        return float(m.group(1)) * _B_UNIT[(m.group(2) or "").lower() or None], True
    t = tok.lower()
    v = consts.get(t)
    if v is None:
        return None, False
    if re.fullmatch(r"p\d+|de|pw", t):
        return v * 1e-6, False
    return float(v), False


def _split_items(line: str):
    """A Bruker line -> [('group', inner, suffix), ('word', w)] in order."""
    items, i, n = [], 0, len(line)
    while i < n:
        c = line[i]
        if c.isspace():
            i += 1
        elif c == "(":
            depth, j = 1, i + 1
            while j < n and depth:
                depth += (line[j] == "(") - (line[j] == ")")
                j += 1
            inner = line[i + 1:j - 1]
            m = re.match(r":(f\d|gp\d*)", line[j:])
            suffix = ""
            if m:
                suffix = m.group(1)
                j += m.end()
            items.append(("group", inner, suffix))
            i = j
        else:
            j = i
            while j < n and not line[j].isspace() and line[j] != "(":
                j += 1
            items.append(("word", line[i:j]))
            i = j
    return items


def _bruker_group_specs(inner, suffix, consts):
    """Specs (for ``_Timeline.parallel``) for one ``( ... ):fN`` group."""
    specs = []
    if "(" in inner:                       # (center (p1 ph1):f1 (p2 ph2):f2 )
        for it in _split_items(inner):
            if it[0] == "group":
                specs += _bruker_group_specs(it[1], it[2] or suffix, consts)
        return specs
    toks = inner.split()
    if toks and toks[0].lower() in ("center", "start", "end", "aligned"):
        toks = toks[1:]
    if not toks:
        return specs
    head = toks[0]
    shape, name = None, head
    if ":" in head:
        name, shape = head.split(":", 1)
    ch = suffix if re.fullmatch(r"f\d", suffix or "") else "f1"
    kind = "hard"
    if suffix.startswith("gp") or (shape or "").lower().startswith("gp"):
        kind, ch = "gradient", "G"
    elif shape and not shape.lower().startswith("f"):
        kind = "shaped"
    value, _ = _bruker_value(name, consts)
    phase = next((t for t in toks[1:] if re.fullmatch(r"ph\d+\^?", t, re.I)), None)
    specs.append(dict(kind=kind, channel=ch, label=name, expr=name, dur=value,
                      shape=shape, phase=phase))
    return specs


def parse_bruker(text, *, constants=None, flags=None, defines=None, name=None,
                 source=None) -> PulseSequence:
    consts = {str(k).lower(): v for k, v in (constants or {}).items()}
    flags = dict(flags or {})
    active = set(defines or ())
    seq_flags: dict = {}
    tl = _Timeline()
    labels: dict = {}                       # '2' -> time at that label
    cond: list = []                         # stack of (active_here, was_true)
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)

    def on() -> bool:
        return all(c[0] for c in cond)

    for raw in text.splitlines():
        line = raw.strip()
        pp = re.match(r"#\s*(ifdef|ifndef|else|endif|define|undef|include|if|elif)\b(.*)", line)
        if pp:
            d, arg = pp.group(1), pp.group(2).strip()
            if d in ("ifdef", "ifndef"):
                nm = arg.split()[0] if arg else ""
                state = flags.get(nm, nm in active)
                seq_flags.setdefault(nm, bool(state))
                here = state if d == "ifdef" else not state
                cond.append((here, here))
            elif d == "if":
                cond.append((True, True))
            elif d == "else" and cond:
                cond[-1] = (not cond[-1][1], cond[-1][1])
            elif d == "endif" and cond:
                cond.pop()
            elif d == "define" and on():
                nm = arg.split()[0] if arg else ""
                if nm:
                    active.add(nm)
            elif d == "undef" and on():
                active.discard(arg.split()[0] if arg else "")
            continue
        if not on():
            continue
        line = line.split(";", 1)[0].strip()
        if not line or line.startswith('"') or re.match(r"(define|ph\d+\s*=|haltacqu)", line, re.I):
            continue
        # leading label ("1 ze", "2 d1")
        m = re.match(r"(\d+)(?:\s+|$)", line)
        line_t0 = tl.t
        if m and not _B_NUM.match(line.split()[0] + "x"):
            labels[m.group(1)] = tl.t
            line = line[m.end():]
        items = _split_items(line)
        housekeeping = any(
            it[0] == "word" and re.match(r"(pl\d*:|plw|fq\b|fq\d|fq=|sp\d*:|spnam|cpd|do:|setnmr|ctrlgrad|rppt?)", it[1], re.I)
            for it in items)
        par: list = []

        def flush():
            nonlocal par
            if par:
                tl.parallel(par)
                par = []

        k = 0
        stop = False
        while k < len(items) and not stop:
            it = items[k]
            k += 1
            if it[0] == "group":
                par += _bruker_group_specs(it[1], it[2], consts)
                continue
            w = it[1].rstrip(",")
            lw = w.lower()
            if lw.startswith("mc") and (lw == "mc" or lw == "mc#0"):
                stop = True
            elif lw == "lo":
                flush()
                rest = [x[1] for x in items[k:k + 4] if x[0] == "word"]
                if len(rest) >= 4 and rest[0].lower() == "to":
                    target, cnt = rest[1], rest[3]
                    t0 = labels.get(target)
                    if t0 is not None:
                        n = consts.get(cnt.lower())
                        tl.loop(t0, cnt, int(n) if isinstance(n, (int, float)) else None)
                k += 4
            elif lw.startswith("go="):
                flush()
                aq = consts.get("aq")
                tl.add("acquire", "ACQ", "ACQ", expr=w, dur=float(aq) if aq else None)
            elif re.fullmatch(r"cpd\w*:f\d|cpds?\d*:f\d", lw):
                tl.open_dec("f" + lw.rsplit("f", 1)[1], lw.split(":")[0], w, at=line_t0)
            elif re.fullmatch(r"do:f\d", lw):
                saved, tl.t = tl.t, line_t0
                tl.close_dec("f" + lw[-1])
                tl.t = saved
            elif re.fullmatch(r"p\d+:gp\d*", lw):
                flush()
                nm, sh = w.split(":")
                v, _ = _bruker_value(nm, consts)
                tl.add("gradient", "G", nm, expr=w, dur=v, shape=sh)
            elif lw in ("ze", "exit", "zd", "wr", "st", "rcyc", "if", "endif", "goto"):
                if lw == "exit":
                    flush()
                continue
            elif _B_NUM.match(w) or re.fullmatch(r"d\d+|de", lw) or _B_LISTDELAY.match(w):
                flush()
                if _B_LISTDELAY.match(w) and not re.fullmatch(r"vd|vp|vc|vt", lw):
                    lab, v, lit = "vd", None, False
                else:
                    v, lit = _bruker_value(w, consts)
                    lab = "" if lit else w
                minor = housekeeping or _is_minor(lab, v, lit)
                tl.add("delay", "delay", lab, expr=w, dur=v, minor=minor)
        flush()
    tl.close_dec()
    tl.finish()
    seq = PulseSequence(name or "Bruker pulse programme", "bruker", tl.events,
                        tl.loops, source, flags=seq_flags)
    return _finalise(seq)


# ---------------------------------------------------------------------------
# Varian / Agilent (C)
# ---------------------------------------------------------------------------

def _split_args(s: str) -> list:
    out, depth, cur = [], 0, ""
    for ch in s:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur.strip())
    return out


def _balanced(s: str, i: int, open_c="(", close_c=")"):
    """Index just past the group opening at ``s[i]``."""
    depth = 0
    for j in range(i, len(s)):
        if s[j] == open_c:
            depth += 1
        elif s[j] == close_c:
            depth -= 1
            if depth == 0:
                return j + 1
    return len(s)


def _c_statements(s: str):
    """Parse a C block body into nested nodes:
    ('call', name, [args]) | ('if', cond, then, else) | ('block', [...])."""
    nodes, i, n = [], 0, len(s)

    def stmt(i):
        while i < n and s[i].isspace():
            i += 1
        if i >= n:
            return None, i
        if s[i] == "{":
            j = _balanced(s, i, "{", "}")
            return ("block", _c_statements(s[i + 1:j - 1])), j
        m = re.match(r"(if|else|for|while)\b", s[i:])
        if m and m.group(1) == "if":
            p = s.index("(", i)
            q = _balanced(s, p)
            then, j = stmt(q)
            k = j
            while k < n and s[k].isspace():
                k += 1
            els = None
            if s[k:k + 4] == "else" and not re.match(r"\w", s[k + 4:k + 5] or " "):
                els, j = stmt(k + 4)
            return ("if", s[p + 1:q - 1], then, els), j
        if m and m.group(1) in ("for", "while"):
            p = s.index("(", i)
            q = _balanced(s, p)
            body, j = stmt(q)
            return ("skip",), j
        depth, j = 0, i
        while j < n:
            if s[j] in "([":
                depth += 1
            elif s[j] in ")]":
                depth -= 1
            elif s[j] == ";" and depth == 0:
                break
            j += 1
        text = s[i:j].strip()
        mc = re.fullmatch(r"([A-Za-z_]\w*)\s*\((.*)\)", text, re.S)
        if mc:
            return ("call", mc.group(1), _split_args(mc.group(2))), j + 1
        return ("skip",), j + 1

    while i < n:
        node, i = stmt(i)
        if node is None:
            break
        nodes.append(node)
    return nodes


def _varian_cond(cond: str, flags: dict, seen: dict):
    """Evaluate a Varian ``if`` condition from flag variables; unknown
    atoms count as true (so numeric guards keep their body)."""
    def atom(a):
        a = a.strip().strip("()").strip()
        m = re.fullmatch(r"(\w+)\s*\[\s*\w+\s*\]\s*(==|!=)\s*'(\w)'", a)
        if m:
            var, op, ch = m.groups()
            if var in ("dm", "dm2", "dm3", "dmm", "dmm2", "dseq", "homo"):
                state = "n"
            else:
                state = "y" if flags.get(var, seen.setdefault(var, False)) else "n"
                seen[var] = state == "y"
            res = state == ch
            return res if op == "==" else not res
        return True
    ors = re.split(r"\|\|", cond)
    return any(all(atom(x) for x in o.split("&&")) for o in ors)


_V_FUNCS = {
    # name: (channels of the pulses, index of first width argument)
    "rgpulse": (["f1"], 0), "pulse": (["f1"], 0), "obspulse": (["f1"], None),
    "decrgpulse": (["f2"], 0), "decpulse": (["f2"], 0),
    "dec2rgpulse": (["f3"], 0), "dec2pulse": (["f3"], 0),
    "dec3rgpulse": (["f4"], 0), "dec3pulse": (["f4"], 0),
    "simpulse": (["f1", "f2"], 0), "sim3pulse": (["f1", "f2", "f3"], 0),
    "sim4pulse": (["f1", "f2", "f3", "f4"], 0),
}
_V_SHAPED = {
    "shaped_pulse": (["f1"], 1), "obsshaped_pulse": (["f1"], 1),
    "decshaped_pulse": (["f2"], 1), "dec2shaped_pulse": (["f3"], 1),
    "simshaped_pulse": (["f1", "f2"], 2), "sim3shaped_pulse": (["f1", "f2", "f3"], 3),
}


def _varian_num(expr, consts):
    names = dict(consts)
    for k in ("pw", "p1", "pi", "tpwr"):
        if k in consts and k in ("pw", "p1", "pi"):
            names[k] = consts[k] * 1e-6        # procpar stores µs, code sees s
    v = _safe_eval(re.sub(r"(?<=\d)\.(?=[eE])", ".0", expr), names)
    return v


def parse_varian(text, *, constants=None, flags=None, name=None, source=None) -> PulseSequence:
    text = _strip_c_comments(text)
    m = re.search(r"\bpulsesequence\s*\(\s*\)\s*\{", text)
    if not m:
        raise PulseSequenceError("No pulsesequence() function found in this Varian file.")
    i = m.end() - 1
    body = text[i + 1:_balanced(text, i, "{", "}") - 1]
    nodes = _c_statements(body)
    consts = {str(k): v for k, v in (constants or {}).items()}
    flags = dict(flags or {})
    seen: dict = {}
    tl = _Timeline()
    inits: dict = {}
    loopstack: list = []
    state = {"acq": False}

    def lab(expr):
        return _label_of(expr)

    def add_pulses(chs, widths, phases, kind, shapes=None):
        specs = []
        for k, (ch, w) in enumerate(zip(chs, widths)):
            num = _varian_num(w, consts)
            if num == 0 or re.fullmatch(r"\s*0(\.0*)?([eE][-+]?\d+)?\s*", w):
                continue
            specs.append(dict(kind=kind, channel=ch, label=lab(w), expr=w, dur=num,
                              shape=(shapes[k] if shapes else None),
                              phase=(phases[k] if k < len(phases) else None)))
        if specs:
            tl.parallel(specs)

    def run(nodes):
        for nd in nodes:
            t = nd[0]
            if t == "block":
                run(nd[1])
            elif t == "if":
                branch = nd[2] if _varian_cond(nd[1], flags, seen) else nd[3]
                if branch:
                    run([branch])
            elif t == "call":
                call(nd[1], nd[2])

    def call(fn, a):
        if fn in _V_FUNCS and a:
            chs, _ = _V_FUNCS[fn]
            if fn == "obspulse":
                add_pulses(["f1"], ["pw"], ["oph"], "hard")
                return
            n = len(chs)
            add_pulses(chs, a[:n], a[n:2 * n] if n > 1 else a[1:2], "hard")
        elif fn in _V_SHAPED and a:
            chs, w0 = _V_SHAPED[fn]
            n = len(chs)
            shapes = a[:w0] if w0 > 1 else [a[0]]
            widths = a[w0:w0 + n]
            if len(shapes) < n:
                shapes = (shapes * n)[:n]
            add_pulses(chs, widths, a[w0 + n:w0 + 2 * n] if n > 1 else a[w0 + 1:w0 + 2],
                       "shaped", shapes)
        elif fn in ("delay", "hsdelay", "vdelay", "initdelay") and a:
            expr = a[0] if fn != "vdelay" else a[-1]
            num = _varian_num(expr, consts)
            literal = bool(re.fullmatch(r"[-+0-9.eE]+", expr))
            label = "" if literal else lab(expr)
            tl.add("delay", "delay", label, expr=expr, dur=num,
                   minor=_is_minor(label, num, literal))
        elif fn == "rgradient" and len(a) >= 2:
            level = a[1]
            key = a[0]
            is_zero = re.fullmatch(r"\s*0(\.0*)?\s*", level) is not None
            if is_zero:
                ev = tl._open_grad.pop(key, None)
                if ev is not None:
                    ev.t1 = max(tl.t, ev.t0 + NOMINAL["gradient"] * 0.1)
            elif key not in tl._open_grad:
                ev = Event("gradient", "G", lab(level), level, tl.t, math.nan)
                tl._open_grad[key] = ev
                tl.events.append(ev)
        elif fn in ("zgradpulse", "gradient") and a:
            lvl = a[0]
            if fn == "zgradpulse" and len(a) > 1:
                num = _varian_num(a[1], consts)
                tl.add("gradient", "G", lab(a[1]) or lab(lvl), expr=a[1], dur=num)
        elif fn == "initval" and len(a) == 2:
            inits[a[1].strip()] = a[0]
        elif fn == "loop" and len(a) == 2:
            cnt = inits.get(a[0].strip(), a[0])
            loopstack.append((a[1].strip(), tl.t, cnt))
        elif fn == "endloop" and a:
            for k in range(len(loopstack) - 1, -1, -1):
                if loopstack[k][0] == a[0].strip():
                    _, t0, cnt = loopstack.pop(k)
                    n = _varian_num(cnt, consts)
                    tl.loop(t0, _label_of(cnt) or cnt, int(n) if n else None)
                    break
        elif fn in ("decon", "dec2on", "dec3on", "decprgon", "dec2prgon"):
            ch = {"decon": "f2", "decprgon": "f2", "dec2on": "f3",
                  "dec2prgon": "f3", "dec3on": "f4"}[fn]
            tl.open_dec(ch, a[0] if a else "", fn)
        elif fn in ("decoff", "dec2off", "dec3off", "decprgoff", "dec2prgoff"):
            ch = {"decoff": "f2", "decprgoff": "f2", "dec2off": "f3",
                  "dec2prgoff": "f3", "dec3off": "f4"}[fn]
            tl.close_dec(ch)
        elif fn in ("setreceiver", "acquire") and not state["acq"]:
            state["acq"] = True
            tl.add("acquire", "ACQ", "ACQ", expr=fn)

    run(nodes)
    # open gradients left unclosed end with the sequence
    for ev in tl._open_grad.values():
        ev.t1 = max(tl.t, ev.t0 + NOMINAL["gradient"])
    if not state["acq"]:
        tl.add("acquire", "ACQ", "ACQ")
    tl.finish()
    fl = {k: v for k, v in seen.items()}
    seq = PulseSequence(name or "Varian pulse sequence", "varian", tl.events,
                        tl.loops, source, flags=fl)
    return _finalise(seq)


# ---------------------------------------------------------------------------
# JEOL Delta (.jxp)
# ---------------------------------------------------------------------------

_J_UNITS = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9, "%": None, "ppm": None,
            "khz": None, "hz": None, "mhz": None, "deg": None}


def _jeol_defs(pre: str):
    """name -> expression text from the declarations before ``begin``
    (statements end with ``;`` and may run over several lines)."""
    defs = {}
    pre = re.sub(r"--[^\n]*", "", pre)
    for st in pre.split(";"):
        m = re.match(r"\s*(\w+)\s*(?:=>|=\?|=)\s*(.*)$", st, re.S)
        if m:
            expr = re.split(r",\s*(?:help\b|[-+]?\d)", m.group(2))[0]
            expr = " ".join(expr.split()).strip().rstrip(",")
            defs.setdefault(m.group(1), expr)
    return defs


def _jeol_seconds(expr, defs, flags, depth=0):
    """Evaluate a JEOL delay expression to seconds (None if unknown)."""
    if depth > 8 or expr is None:
        return None
    e = expr.strip().strip('"')
    m = re.match(r"if\s+(\w+)\s+then\s+(.*?)\s+else\s+(.*)$", e, re.S)
    if m:
        cond = flags.get(m.group(1), defs.get(m.group(1), "True").strip().lower() == "true")
        return _jeol_seconds(m.group(2) if cond else m.group(3), defs, flags, depth + 1)
    m = re.fullmatch(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\[(\w+|%)\]", e)
    if m:
        scale = _J_UNITS.get(m.group(2).lower())
        return float(m.group(1)) * scale if scale else None
    # arithmetic over names / unit literals
    def sub_unit(mm):
        scale = _J_UNITS.get(mm.group(2).lower())
        return repr(float(mm.group(1)) * scale) if scale else "None"
    e2 = re.sub(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\[(\w+|%)\]", sub_unit, e)
    names = {}
    for nm in set(re.findall(r"[A-Za-z_]\w*", e2)):
        if nm in defs and defs[nm].strip() != e:
            v = _jeol_seconds(defs[nm], defs, flags, depth + 1)
            if v is not None:
                names[nm] = v
    return _safe_eval(e2, names)


def parse_jeol(text, *, flags=None, name=None, source=None) -> PulseSequence:
    text_nc = re.sub(r"--[^\n]*", "", text)
    m = re.search(r"^\s*pulse\b(.*?)^\s*begin\b(.*?)^\s*end\s+pulse\b", text_nc,
                  re.S | re.M)
    if not m:
        raise PulseSequenceError("No 'pulse ... begin ... end pulse' block found "
                                 "in this JEOL file.")
    defs = _jeol_defs(m.group(1))
    flags = dict(flags or {})
    seen = {}
    for nm, ex in defs.items():
        if ex.strip().lower() in ("true", "false") and not nm.endswith("_Setup"):
            seen[nm] = flags.get(nm, ex.strip().lower() == "true")
    fl = {**{k: v for k, v in seen.items()}}
    tl = _Timeline()
    chan = {"obs": "f1", "irr": "f2", "tnt": "f3"}
    for st in re.split(r";", m.group(2)):
        s = st.strip()
        if not s:
            continue
        if re.match(r"on\s*\(", s):
            ch = re.search(r"\b(obs|irr|tnt)\.", s)
            if ch:
                tl.open_dec(chan[ch.group(1)], "dec", s)
            continue
        if re.match(r"off\s*\(", s):
            ch = re.search(r"\b(obs|irr|tnt)\.", s)
            if ch:
                tl.close_dec(chan[ch.group(1)])
            continue
        if re.match(r"acq\s*\(", s):
            tl.add("acquire", "ACQ", "ACQ", expr=s)
            continue
        pm = re.match(r"(\w+)\s*,\s*\((.*)\)\s*$", s, re.S)
        if pm and re.search(r"\.gate", pm.group(2)):
            ch = re.search(r"\b(obs|irr|tnt)\.", pm.group(2))
            c = chan.get(ch.group(1), "f1") if ch else "f1"
            nm = pm.group(1)
            dur = _jeol_seconds(defs.get(nm), defs, flags)
            kind = "shaped" if re.search(r"shape|\.wave", pm.group(2)) else "hard"
            tl.add(kind, c, nm, expr=nm, dur=dur,
                   phase=(re.search(r"\.phs\.(\w+)", pm.group(2)) or [None, None])[1])
            continue
        if re.fullmatch(r"\w+", s):
            dur = _jeol_seconds(defs.get(s), defs, flags)
            tl.add("delay", "delay", s, expr=s, dur=dur,
                   minor=_is_minor(s, dur, False))
    tl.finish()
    nm = name
    fn = re.search(r'filename\s*=>\s*"([^"]*)"', text_nc)
    seq = PulseSequence(nm or (fn.group(1) if fn else "JEOL pulse sequence"),
                        "jeol", tl.events, tl.loops, source, flags=fl)
    return _finalise(seq)


# ---------------------------------------------------------------------------
# Magritek Spinsolve (Prospa macro)
# ---------------------------------------------------------------------------

def parse_magritek(text, *, constants=None, name=None, source=None) -> PulseSequence:
    consts = dict(constants or {})
    body = text
    m = re.search(r"initpp\s*\(.*?\)(.*?)endpp\s*\(", text, re.S)
    if m:
        body = m.group(1)
    tl = _Timeline()
    for ln in body.splitlines():
        ln = ln.split("#", 1)[0].strip()
        mm = re.match(r"(pulse|wait|delay|acquire|Spoiler:\w+|shapedpulse|rfpulse)\s*\((.*)\)\s*$", ln)
        if not mm:
            continue
        fn, args = mm.group(1), _split_args(mm.group(2))
        if fn == "pulse" and len(args) >= 4:
            ch = f"f{args[0].strip()}" if args[0].strip().isdigit() else "f1"
            nm = args[3]
            tl.add("hard", ch, nm, expr=nm, dur=_prospa_us(nm, consts), phase=args[2])
        elif fn in ("shapedpulse", "rfpulse") and len(args) >= 4:
            nm = args[3]
            tl.add("shaped", "f1", nm, expr=nm, dur=_prospa_us(nm, consts))
        elif fn in ("wait", "delay") and args:
            nm = args[0]
            v = _prospa_us(nm, consts)
            tl.add("delay", "delay", _label_of(nm), expr=nm, dur=v,
                   minor=fn == "delay" and (_is_minor(nm, v, False) or True) and v is None
                   and "stab" in nm.lower())
        elif fn.startswith("Spoiler") and len(args) >= 2:
            dspoil, dstab = (args[-2], args[-1])
            tl.add("gradient", "G", _label_of(dspoil), expr=dspoil, dur=_prospa_us(dspoil, consts))
            tl.add("delay", "delay", _label_of(dstab), expr=dstab,
                   dur=_prospa_us(dstab, consts), minor=True)
        elif fn == "acquire":
            tl.add("acquire", "ACQ", "ACQ", expr=ln)
    if not tl.events:
        raise PulseSequenceError("No pulse(...) / wait(...) / acquire(...) "
                                 "statements found in this Magritek macro.")
    tl.finish()
    nm = name or (re.sub(r"_pp$", "", os.path.splitext(os.path.basename(source))[0])
                  if source else "Magritek pulse programme")
    return _finalise(PulseSequence(nm, "magritek", tl.events, tl.loops, source,
                                   notes=["Durations are named in the macro; "
                                          "their values live in the experiment's parameters."]))


def _prospa_us(expr, consts):
    v = _safe_eval(expr, consts)
    return None if v is None else v * 1e-6


# ---------------------------------------------------------------------------
# X-Pulse (Python)
# ---------------------------------------------------------------------------

class _Val:
    """Value flowing through the symbolic X-Pulse interpreter."""
    __slots__ = ("num", "label", "text", "func", "items")

    def __init__(self, num=None, label="", text="", func=None, items=None):
        self.num, self.label, self.text, self.func, self.items = num, label, text, func, items


_XP_SPLIT = re.compile(r"^#\s*(.*?)\s*~{4,}\s*$", re.M)
_XP_HOUSEKEEPING = re.compile(r"txenable|dead|ringdown|blank|relay|settle", re.I)


def _xp_chunks(text: str):
    heads = list(_XP_SPLIT.finditer(text))
    if len(heads) < 2:
        return [(None, text)]
    out = []
    for k, h in enumerate(heads):
        end = heads[k + 1].start() if k + 1 < len(heads) else len(text)
        out.append((h.group(1) or f"Element {k + 1}", text[h.end():end]))
    return out


def _xp_fix(src: str) -> str:
    """Give an empty ``class X:`` / ``def f():`` body (library chunks) a
    ``pass`` so the chunk parses."""
    lines = src.split("\n")
    out = []
    for i, ln in enumerate(lines):
        out.append(ln)
        if re.match(r"(class|def)\b.*:\s*(#.*)?$", ln):
            nxt = next((x for x in lines[i + 1:] if x.strip() and not x.strip().startswith("#")), None)
            if nxt is None or not nxt.startswith((" ", "\t")):
                out.append("    pass")
    return "\n".join(out)


class _XPulse:
    def __init__(self, tree, src):
        self.funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
        self.params = {}
        for n in tree.body:
            if isinstance(n, ast.ClassDef) and n.name == "Parameters":
                for st in n.body:
                    if (isinstance(st, ast.Assign) and isinstance(st.value, ast.Call)
                            and getattr(st.value.func, "id", "") == "Parameter"
                            and st.value.args and isinstance(st.targets[0], ast.Name)):
                        a = st.value.args
                        pname = a[0].value if isinstance(a[0], ast.Constant) else st.targets[0].id
                        default = a[1].value if len(a) > 1 and isinstance(a[1], ast.Constant) else None
                        self.params[st.targets[0].id] = (str(pname), default)
        self.tl = _Timeline()
        self.has_main = "sequential_main" in src
        self.tl.recording = not self.has_main
        self.seq_name = next((v[1] for k, v in self.params.items() if k == "Sequence"), None)
        self.depth = 0

    # -- expressions
    def ev(self, n, env) -> _Val:
        text = ast.unparse(n) if hasattr(ast, "unparse") else ""
        if isinstance(n, ast.Constant):
            v = n.value
            return _Val(float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None,
                        "", text) if not isinstance(v, str) else _Val(None, "", text, items=v)
        if isinstance(n, ast.Name):
            if n.id in env:
                return env[n.id]
            return _Val(None, n.id, text, func=n.id)
        if isinstance(n, ast.Attribute):
            if isinstance(n.value, ast.Name) and n.value.id == "P":
                pname, default = self.params.get(n.attr, (n.attr, None))
                if isinstance(default, str):
                    return _Val(None, pname, text, items=default)
                return _Val(float(default) if isinstance(default, (int, float)) else None, pname, text)
            return _Val(None, n.attr, text)
        if isinstance(n, ast.UnaryOp):
            v = self.ev(n.operand, env)
            return _Val(-v.num if isinstance(n.op, ast.USub) and v.num is not None else v.num,
                        v.label, text)
        if isinstance(n, ast.BinOp):
            a, b = self.ev(n.left, env), self.ev(n.right, env)
            num = None
            if a.num is not None and b.num is not None and type(n.op) in _SAFE_BIN:
                try:
                    num = _SAFE_BIN[type(n.op)](a.num, b.num)
                except (OverflowError, ZeroDivisionError):
                    num = None
            label = a.label or b.label
            if isinstance(n.op, ast.Mult):
                for k, o in ((a, b), (b, a)):
                    if k.num is not None and not k.label and o.label:
                        kk = f"{k.num:g}"
                        label = f"{kk}{o.label}"
                        break
            return _Val(num, label, text)
        if isinstance(n, ast.Call):
            f = n.func
            fname = f.id if isinstance(f, ast.Name) else (f.attr if isinstance(f, ast.Attribute) else "")
            args = [self.ev(a, env) for a in n.args]
            if fname in ("abs", "int", "float", "round") and args:
                return _Val(abs(args[0].num) if fname == "abs" and args[0].num is not None else args[0].num,
                            args[0].label, text)
            if fname in self.funcs:
                return self.call_user(fname, args, env)
            return _Val(None, "", text)
        if isinstance(n, ast.Tuple):
            return _Val(None, "", text, items=[self.ev(e, env) for e in n.elts])
        if isinstance(n, ast.Dict):
            d = {}
            for k, v in zip(n.keys, n.values):
                if k is None:
                    continue
                kv = self.ev(k, env)
                d[kv.num if kv.num is not None else kv.items] = self.ev(v, env)
            return _Val(None, "", text, items=d)
        if isinstance(n, ast.Subscript):
            base = self.ev(n.value, env)
            if isinstance(base.items, dict) and base.items:
                idx = self.ev(n.slice, env)
                key = idx.num if idx.num is not None else idx.items
                return base.items.get(key, next(iter(base.items.values())))
            return _Val(None, "", text)
        return _Val(None, "", text)

    # -- calls
    def call_user(self, fname, args, env):
        if self.depth > 12:
            return _Val()
        fn = self.funcs[fname]
        loc = {}
        for p, a in zip(fn.args.args, args):
            loc[p.arg] = a
        self.depth += 1
        try:
            ret = self.block(fn.body, loc)
        finally:
            self.depth -= 1
        return ret if isinstance(ret, _Val) else _Val()

    def primitive(self, name, args, nodes_text=""):
        tl = self.tl
        m = re.fullmatch(r"Transmit(\d)", name)
        if m and args:
            a = args[0]
            tl.add("hard", f"f{m.group(1)}", a.label, expr=a.text,
                   dur=None if a.num is None else a.num * 1e-6)
            return
        m = re.fullmatch(r"Transmit(\d)ShapedPulse", name)
        if m and args:
            a = args[0]
            shape = args[2].label if len(args) > 2 else None
            tl.add("shaped", f"f{m.group(1)}", a.label, expr=a.text, shape=shape,
                   dur=None if a.num is None else a.num * 1e-6)
            return
        if name == "Delay" and args:
            a = args[0]
            dur = None if a.num is None else a.num * 1e-6
            lab = a.label if a.label else ""
            minor = bool(_XP_HOUSEKEEPING.search(lab)) or _is_minor(lab, dur, not lab)
            tl.add("delay", "delay", lab, expr=a.text, dur=dur, minor=minor)
            return
        m = re.fullmatch(r"Gradient(\d)(SlewRate|Matrix)?", name)
        if m and not m.group(2) and len(args) >= 2:
            key = m.group(1)
            t, v = args[0], args[1]
            dur = None if t.num is None else t.num * 1e-6
            zero = v.num == 0 if v.num is not None else (v.text.strip() == "0")
            if not zero and key not in tl._open_grad:
                ev = Event("gradient", "G", v.label, v.text, tl.t, math.nan)
                tl._open_grad[key] = ev
                if tl.recording:
                    tl.events.append(ev)
            tl.t += dur if dur is not None else 0.0
            if zero:
                ev = tl._open_grad.pop(key, None)
                if ev is not None:
                    ev.t1 = max(tl.t, ev.t0 + 1e-6)
            return
        m = re.fullmatch(r"Gradient(\d)ShapedPulse", name)
        if m and len(args) >= 3:
            a = args[0]
            tl.add("gradient", "G", args[2].label, expr=a.text, shape=args[1].label,
                   dur=None if a.num is None else a.num * 1e-6)
            return
        m = re.fullmatch(r"Receiver(\d)", name)
        if m and args and not any(e.kind == "acquire" for e in tl.events):
            tl.add("acquire", "ACQ", "ACQ", expr=name)

    # -- statements
    def block(self, stmts, env):
        ret = None
        for st in stmts:
            r = self.stmt(st, env)
            if isinstance(r, _Val):
                ret = r
        return ret

    def with_kind(self, item):
        c = item.context_expr
        if isinstance(c, ast.Name):
            return c.id, None
        if isinstance(c, ast.Call):
            f = c.func.id if isinstance(c.func, ast.Name) else getattr(c.func, "attr", "")
            return f, c
        return "", None

    def stmt(self, st, env):
        tl = self.tl
        if isinstance(st, ast.Expr) and isinstance(st.value, ast.Call):
            c = st.value
            f = c.func
            fname = f.id if isinstance(f, ast.Name) else getattr(f, "attr", "")
            args = [self.ev(a, env) for a in c.args]
            if isinstance(f, ast.Name) and f.id in env and env[f.id].func:
                fname = env[f.id].func
            if fname in self.funcs:
                self.call_user(fname, args, env)
            else:
                self.primitive(fname, args)
            return None
        if isinstance(st, ast.Assign):
            val = self.ev(st.value, env)
            for t in st.targets:
                if isinstance(t, ast.Name):
                    if not val.label and not isinstance(val.items, (list, dict)):
                        val = _Val(val.num, t.id, val.text, val.func, val.items)
                    env[t.id] = val
                elif isinstance(t, ast.Tuple) and isinstance(val.items, list):
                    for e, v in zip(t.elts, val.items):
                        if isinstance(e, ast.Name):
                            env[e.id] = v
            return None
        if isinstance(st, ast.Return):
            return self.ev(st.value, env) if st.value is not None else None
        if isinstance(st, ast.With):
            kind, call = self.with_kind(st.items[0])
            t_start = tl.t
            rec_before = tl.recording
            if kind == "sequential_main":
                tl.recording = True
            if kind == "parallel":
                ends, start = [], tl.t
                for child in st.body:
                    tl.t = start
                    n0 = len(tl.events)
                    self.stmt(child, env)
                    ends.append(tl.t)
                tl.t = max(ends) if ends else start
            else:
                self.block(st.body, env)
            if kind == "repeat" and call is not None and call.args:
                cnt = self.ev(call.args[0], env)
                tl.loop(t_start, cnt.label or "n", int(cnt.num) if cnt.num else None)
            tl.recording = rec_before if kind == "sequential_main" else tl.recording
            return None
        if isinstance(st, ast.For):
            loopvars = {}
            for nme in ast.walk(st.target):
                if isinstance(nme, ast.Name):
                    loopvars[nme.id] = _Val()
            env.update(loopvars)
            t_start = tl.t
            txt = ast.dump(st)
            self.block(st.body, env)
            if "sequential_main" not in txt and isinstance(st.iter, ast.Call) \
                    and getattr(st.iter.func, "id", "") == "range" and st.iter.args:
                cnt = self.ev(st.iter.args[-1], env)
                tl.loop(t_start, cnt.label or "n", int(cnt.num) if cnt.num and cnt.num > 1 else None)
            return None
        if isinstance(st, ast.If):
            take = self._decide(st.test, env)
            if take is None or take:
                return self.block(st.body, env)
            return self.block(st.orelse, env) if st.orelse else None
        return None

    def _decide(self, test, env):
        if isinstance(test, ast.Compare) and len(test.ops) == 1 and \
                isinstance(test.ops[0], (ast.Eq, ast.NotEq)):
            a, b = self.ev(test.left, env), self.ev(test.comparators[0], env)
            va = a.items if isinstance(a.items, str) else a.num
            vb = b.items if isinstance(b.items, str) else b.num
            if va is not None and vb is not None:
                same = va == vb
                return same if isinstance(test.ops[0], ast.Eq) else not same
        return None


def parse_xpulse(text, *, name=None, source=None) -> list:
    out = []
    chunks = _xp_chunks(text)
    for title, src in chunks:
        src = _xp_fix(src)
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        if not any(isinstance(n, ast.FunctionDef) and n.name == "run" for n in tree.body):
            continue
        xp = _XPulse(tree, src)
        run = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run")
        try:
            _xp_run(xp, run, tree)
        except RecursionError:
            continue
        xp.tl.finish()
        if not xp.tl.events:
            continue
        nm = title or xp.seq_name or name or "X-Pulse sequence"
        if title is None and xp.seq_name:
            nm = xp.seq_name
        seq = PulseSequence(nm, "xpulse", xp.tl.events, xp.tl.loops, source)
        out.append(_finalise(seq))
    if not out:
        raise PulseSequenceError("No X-Pulse sequence events (Transmit/Delay/Gradient ...) "
                                 "were found in this file.")
    return out


def _xp_run(xp, run, tree):
    # Decoupling / shaped-pulse detection needs the parallel structure, so
    # walk with a classifier wrapped around the interpreter's parallel blocks.
    orig = xp.stmt

    def stmt(st, env):
        if isinstance(st, ast.With) and xp.with_kind(st.items[0])[0] == "parallel":
            n0 = len(xp.tl.events)
            r = orig(st, env)
            _xp_classify_parallel(xp, st, xp.tl.events[n0:])
            return r
        return orig(st, env)
    xp.stmt = stmt
    xp.block(run.body, {})


def _call_names(node) -> set:
    out = set()
    for nd in ast.walk(node):
        if isinstance(nd, ast.Call):
            out.add(nd.func.id if isinstance(nd.func, ast.Name) else getattr(nd.func, "attr", ""))
    return out


def _xp_classify_parallel(xp, st, events):
    branches = list(st.body)
    sets = [_call_names(b) for b in branches]
    for s_i, names in enumerate(sets):
        phase_only = {n for n in names if re.fullmatch(r"Channel(\d)SetBasePhase", n)}
        freq = {n for n in names if re.fullmatch(r"Channel(\d)SetFrequency|Transmit(\d)SetScale", n)}
        has_tx = {n for n in names if re.fullmatch(r"Transmit(\d)", n)}
        # branch that only changes phase / frequency, beside one that transmits
        if (phase_only or freq) and not has_tx:
            for o_i, onames in enumerate(sets):
                if o_i == s_i:
                    continue
                for ev in events:
                    if ev.kind == "hard" and any(
                            n == f"Transmit{ev.channel[1:]}" for n in onames):
                        ev.kind = "decoupling" if phase_only and not freq else "shaped"
        # a transmit branch whose own siblings set scale/frequency -> shaped
    return events


# ---------------------------------------------------------------------------
# finishing, layout
# ---------------------------------------------------------------------------

def _finalise(seq: PulseSequence) -> PulseSequence:
    ev = [e for e in seq.events if e.t1 >= e.t0]
    ev.sort(key=lambda e: (e.t0, e.t1))
    # merge neighbouring housekeeping delays
    merged = []
    for e in ev:
        p = merged[-1] if merged else None
        if (p and e.kind == "delay" and p.kind == "delay" and e.minor and p.minor
                and abs(p.t1 - e.t0) < 1e-12):
            p.t1 = e.t1
            p.label = ""
            p.value = (p.value or 0.0) + (e.value or 0.0) if (p.value is not None or e.value is not None) else None
            continue
        merged.append(e)
    for i, e in enumerate(merged):
        e.uid = i
    seq.events = merged
    for lp in seq.loops:
        lp.depth = sum(1 for o in seq.loops
                       if o is not lp and o.t0 <= lp.t0 and o.t1 >= lp.t1
                       and (o.t0, o.t1) != (lp.t0, lp.t1))
    return seq


@dataclass
class Layout:
    x0: dict
    x1: dict
    width: float
    lane: dict            # delay uid -> lane index
    n_lanes: int
    loops: list           # (Loop, x0, x1)


def _interval_width(dur: float) -> float:
    return min(0.6 + 0.4 * math.log10(1.0 + max(dur, 0.0) / 1e-5), 4.2)


def compute_layout(seq: PulseSequence, show_minor: bool = False) -> Layout:
    """Schematic x position of every event (log-compressed time, not to
    scale). Housekeeping delays collapse to a thin gap unless
    ``show_minor``."""
    evs = seq.events
    pts = sorted({round(t, 12) for e in evs for t in (e.t0, e.t1)}
                 | {round(t, 12) for lp in seq.loops for t in (lp.t0, lp.t1)})
    if len(pts) < 2:
        pts = [0.0, 1e-3]
    visible_cover = [e for e in evs if not (e.kind == "delay" and e.minor)]
    xs = [0.0]
    for a, b in zip(pts[:-1], pts[1:]):
        mid = 0.5 * (a + b)
        covered = any(e.t0 <= mid <= e.t1 for e in visible_cover)
        in_minor = any(e.kind == "delay" and e.minor and e.t0 <= mid <= e.t1 for e in evs)
        if in_minor and not covered and not show_minor:
            w = 0.18
        else:
            w = _interval_width(b - a)
        xs.append(xs[-1] + w)

    def x(t):
        t = round(t, 12)
        if t <= pts[0]:
            return xs[0]
        if t >= pts[-1]:
            return xs[-1]
        import bisect
        k = bisect.bisect_right(pts, t) - 1
        if pts[k] == t:
            return xs[k]
        f = (t - pts[k]) / (pts[k + 1] - pts[k])
        return xs[k] + f * (xs[k + 1] - xs[k])

    x0 = {e.uid: x(e.t0) for e in evs}
    x1 = {e.uid: x(e.t1) for e in evs}
    # delays that overlap in time go on separate lanes
    lanes_end: list = []
    lane = {}
    for e in sorted((e for e in evs if e.kind == "delay"), key=lambda e: x0[e.uid]):
        for k, end in enumerate(lanes_end):
            if x0[e.uid] >= end - 1e-9:
                lane[e.uid] = k
                lanes_end[k] = x1[e.uid]
                break
        else:
            lane[e.uid] = len(lanes_end)
            lanes_end.append(x1[e.uid])
    loops = [(lp, x(lp.t0), x(lp.t1)) for lp in seq.loops]
    return Layout(x0, x1, xs[-1], lane, max(len(lanes_end), 1), loops)


# ---------------------------------------------------------------------------
# entry points
# ---------------------------------------------------------------------------

def detect_vendor(text: str, path: str | None = None) -> str:
    ext = os.path.splitext(path or "")[1].lower()
    base = os.path.basename(path or "").lower()
    if ext == ".jxp" or re.search(r"^\s*pulse\b.*?^\s*begin\b", text, re.S | re.M) \
            and "Delta NMR" in text[:600]:
        return "jeol"
    if re.search(r"\bpulsesequence\s*\(", text):
        return "varian"
    if "firebird" in text or "ParameterBlock" in text or re.search(r"def\s+run\s*\(\s*comms", text):
        return "xpulse"
    if "initpp(" in text or base.endswith("_pp.mac") or ext == ".mac":
        return "magritek"
    return "bruker"


def parse_pulse_sequence(text: str, vendor: str | None = None, *, name=None,
                         constants=None, flags=None, defines=None,
                         source=None) -> list:
    """Parse the *text* of a pulse programme -> a list of
    :class:`PulseSequence` (one, except for an X-Pulse library file).

    ``constants`` -- named acquisition values, e.g.
    ``acquisition_constants(spectrum)``; ``flags`` -- override the
    conditional blocks (``{'dec': True}``); ``defines`` -- Bruker ``-D``
    names (:func:`zgoptns_defines`)."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    vendor = (vendor or detect_vendor(text, source)).lower()
    if vendor == "bruker":
        return [parse_bruker(text, constants=constants, flags=flags,
                             defines=defines, name=name, source=source)]
    if vendor in ("varian", "agilent"):
        return [parse_varian(text, constants=constants, flags=flags, name=name,
                             source=source)]
    if vendor == "jeol":
        return [parse_jeol(text, flags=flags, name=name, source=source)]
    if vendor == "xpulse":
        return parse_xpulse(text, name=name, source=source)
    if vendor in ("magritek", "spinsolve"):
        return [parse_magritek(text, constants=constants, name=name, source=source)]
    raise PulseSequenceError(f"Unknown pulse-sequence format {vendor!r}.")


def read_pulse_sequences(path, vendor: str | None = None, **kw) -> list:
    """Read a pulse-sequence file from disk (see :func:`parse_pulse_sequence`)."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    except OSError as exc:
        raise PulseSequenceError(f"Cannot read {path!r}: {exc}") from exc
    kw.setdefault("name", None)
    seqs = parse_pulse_sequence(text, vendor, source=str(path), **kw)
    for s in seqs:
        if s.name in ("Bruker pulse programme",):
            s.name = os.path.splitext(os.path.basename(str(path)))[0]
    return seqs
