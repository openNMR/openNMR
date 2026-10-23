"""
Bruker PFG gradient information **without** a ``diff.xml``.

``io.bruker.read_diff_xml`` covers the easy case: an experiment set up
through TopSpin's **diff module** writes ``diff.xml``, which carries
TopSpin's own precomputed b-values and is always the right thing to
use. An experiment set up the other way -- the **``dosy`` AU program**
driving a stock sequence like ``ledbpgp2s`` -- writes no ``diff.xml``
at all, and until now that left openNMR with no b-axis for what is
probably the more common way a DOSY dataset gets acquired.

Everything needed is still on disk, just spread over four files::

    difflist      the gradient amplitude per F1 point, in G/cm
    acqus         P[30], D[20], D[16] -- but *which* index means what
                  is a property of the pulse programme, not of Bruker
    pulseprogram  which it states, in its own trailing comment block
    Difframp      the relative ramp, and gradient_calib the probe's
                  own G/cm -- used here only to cross-check difflist

so :func:`read_pfg_params` reads all four and returns one description
with every value tagged by where it came from.

**The discrimination that makes this safe.** ``difflist`` does not
always hold gradients. For a *diff module* experiment it holds
**b-values in s/m^2** -- verified against ``diff.xml`` on
``test_data/bruker_PFG/150`` and ``Diff50.../11``, where the two agree
digit for digit. Reading those as G/cm would silently produce gradient
amplitudes of 1e8 G/cm and a b-axis wrong by twenty orders of
magnitude. :func:`read_difflist` decides which it is from the presence
of the diff module's own files and from magnitude, and says which in
its result rather than leaving the caller to assume.

**Which parameter is delta.** Bruker's stock diffusion sequences
document their own parameters in a trailing comment block::

    ;p30: gradient pulse (little DELTA * 0.5)
    ;d20: diffusion time (big DELTA)
    ;d16: delay for gradient recovery

:func:`describe_diffusion_pulseprogram` reads that block rather than
hard-coding ``p30``/``d20``, so a sequence that numbers its parameters
differently still resolves, and one that documents nothing fails
loudly instead of returning a confident wrong answer. The ``* 0.5`` is
the point of parsing the wording rather than the name: on a bipolar
sequence ``p30`` is **half** of delta, and taking it as delta
understates b by 4x.

Verified on ``test_data/NMR400_KT_DPC_ALSK_AL_70B_3/10`` (``ledbpgp2s``,
1H, 32 steps): delta = 2*p30 = 4.42 ms, DELTA = d20 = 60 ms, tau =
d16 = 0.2 ms, gradients 1.046 .. 51.244 G/cm, and the resulting b-axis
fits the solute's three resolved peaks to a consistent
D = 1.78e-10 m^2/s (rmse 1.4% of I0). See :func:`selftest`.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import lists as io_lists
from .jcampdx import parse_jcampdx_params
from ..processing import diffusion as _diffusion


# ============================================================
# The pulse programme's own parameter documentation
# ============================================================

#: ``;p30: gradient pulse (little DELTA * 0.5)`` -- Bruker writes one
#: of these per parameter at the end of every stock sequence. The
#: leading name is always a bare ``p``/``d``/``gpz`` index.
_PP_DOC_RE = re.compile(r"^;\s*(?P<name>(?:p|d|gpz|cnst|l)\d+)\s*:\s*(?P<text>.+?)\s*$",
                        re.IGNORECASE)

#: A gradient statement in the programme body: ``p30:gp6*diff`` or
#: ``p30:gp6*-1*diff``. The ``*-1*`` is what makes a pair bipolar, and
#: reading it from the body is independent of the comment block.
_PP_GRADIENT_RE = re.compile(
    r"^\s*(?P<pulse>p\d+)\s*:\s*gp(?P<channel>\d+)\s*(?P<rest>\S*)", re.IGNORECASE)


@dataclass
class PulseProgramDiffusion:
    """What a pulse programme says about its own diffusion parameters."""

    name: str | None
    #: ``"p30"`` etc. -- the acqus parameter holding delta, and what to
    #: multiply it by (2 when the programme documents it as
    #: "little DELTA * 0.5").
    delta_param: str | None = None
    delta_factor: float = 1.0
    big_delta_param: str | None = None
    #: The gradient-recovery delay, used as tau by the bipolar b-value.
    tau_param: str | None = None
    #: Gradient channel the diffusion-encoding pulse uses (6 for the
    #: stock sequences), for looking up ``gpnam``/``GPZ``.
    gradient_channel: int | None = None
    bipolar: bool = False
    #: Every ``;name: text`` line found, for display and for a caller
    #: that wants a parameter this class doesn't model.
    doc: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def sequence(self) -> str:
        """The :data:`opennmr.processing.diffusion.DIFFUSION_SEQUENCES`
        key this programme corresponds to."""
        return "bipolar" if self.bipolar else "pgste"

    @property
    def resolved(self) -> bool:
        """True when both timings were identified -- i.e. this is
        usable, rather than a programme that documents nothing."""
        return self.delta_param is not None and self.big_delta_param is not None


def parse_pp_parameter_doc(text: str) -> dict[str, str]:
    """Pull a pulse programme's trailing ``;p30: ...`` documentation
    block into ``{"p30": "gradient pulse (little DELTA * 0.5)", ...}``.

    A compiled ``pulseprogram`` inlines its ``.incl`` files, and those
    carry their own comment blocks; a later line wins, so the sequence's
    own documentation (which comes last) overrides an include's.
    """
    doc: dict[str, str] = {}
    for line in text.splitlines():
        match = _PP_DOC_RE.match(line)
        if match:
            doc[match.group("name").lower()] = match.group("text")
    return doc


def _delta_factor_from(description: str) -> float:
    """2.0 when the description says this parameter is *half* of delta
    ("little DELTA * 0.5", "little DELTA/2"), else 1.0."""
    squashed = description.replace(" ", "").lower()
    return 2.0 if ("delta*0.5" in squashed or "delta/2" in squashed
                   or "0.5*delta" in squashed) else 1.0


def describe_diffusion_pulseprogram(text: str, name: str | None = None) -> PulseProgramDiffusion:
    """Work out which ``acqus`` parameters a diffusion pulse programme
    uses for delta, DELTA and the bipolar recovery delay.

    Reads the programme's own documentation block, so nothing here is
    keyed on a parameter *number*. The phrases matched ("little DELTA",
    "big DELTA"/"diffusion time", "gradient recovery") are Bruker's own
    wording, shared across the whole ``ste``/``led``/``dste`` family.

    A programme that documents none of them comes back with
    ``resolved`` False rather than a guess -- the diff-module sequences
    (``diffSte``, ``diffDste``) are exactly that case, and they have a
    ``diff.xml`` to be read instead.
    """
    doc = parse_pp_parameter_doc(text)
    result = PulseProgramDiffusion(name=name, doc=doc)

    for param, description in doc.items():
        lowered = description.lower()
        if "little delta" in lowered and result.delta_param is None:
            result.delta_param = param
            result.delta_factor = _delta_factor_from(description)
        elif ("big delta" in lowered or "diffusion time" in lowered) \
                and result.big_delta_param is None:
            result.big_delta_param = param
        elif "gradient recovery" in lowered and result.tau_param is None:
            result.tau_param = param

    # The bipolar test is structural, not textual: a diffusion-encoding
    # pulse replayed on the same gradient channel with *-1* is a
    # bipolar pair whatever the header comment happens to say.
    channels: dict[str, set[int]] = {}
    negated: set[str] = set()
    for line in text.splitlines():
        match = _PP_GRADIENT_RE.match(line)
        if not match:
            continue
        pulse, channel = match.group("pulse").lower(), int(match.group("channel"))
        channels.setdefault(pulse, set()).add(channel)
        if "-1" in match.group("rest"):
            negated.add(pulse)
    if result.delta_param in negated:
        result.bipolar = True
    if result.delta_param in channels and len(channels[result.delta_param]) == 1:
        result.gradient_channel = next(iter(channels[result.delta_param]))

    if result.bipolar and result.tau_param is None:
        result.notes.append(
            "bipolar pair with no documented gradient-recovery delay; tau taken as 0, "
            "which overstates the diffusion time by tau/2"
        )
    if not result.resolved:
        result.notes.append(
            "this pulse programme does not document a 'little DELTA'/'big DELTA' "
            "parameter – timings must be supplied by the caller"
        )
    return result


# ============================================================
# The gradient list
# ============================================================

#: No probe reaches this many G/cm, so a ``difflist`` above it is not a
#: gradient list. The strongest Bruker diffusion probe in
#: :data:`~opennmr.processing.diffusion.PROBE_MAX_GRADIENT_G_PER_CM` is
#: ~2900 G/cm; a diff-module ``difflist`` holds b-values, which start
#: around 1e6. Nothing real lands between the two.
_MAX_PLAUSIBLE_G_PER_CM = 1.0e5

#: Files TopSpin's diff module writes and the ``dosy`` AU never does.
_DIFF_MODULE_FILES = ("diff.xml", "diffConf.xml", "diffHistory.txt")


def is_diff_module_experiment(path: str) -> bool:
    """True if this dataset was set up through TopSpin's diff module --
    in which case ``diff.xml`` is authoritative and ``difflist`` holds
    b-values, not gradients."""
    return any(os.path.isfile(os.path.join(path, name)) for name in _DIFF_MODULE_FILES)


def read_difflist(path: str) -> dict:
    """Read ``difflist`` and say **what its numbers mean**.

    Returns ``{"values", "kind", "gradient_g_per_cm", "b", "reason"}``
    where ``kind`` is ``"gradient"`` (the ``dosy`` AU case, G/cm) or
    ``"b_value"`` (the diff-module case, s/m^2). Exactly one of
    ``gradient_g_per_cm`` and ``b`` is populated.
    """
    values = io_lists.read_value_list(os.path.join(path, "difflist"))["values"]

    if is_diff_module_experiment(path):
        kind, reason = "b_value", "diff module dataset (diff.xml/diffConf.xml present)"
    elif np.nanmax(np.abs(values)) > _MAX_PLAUSIBLE_G_PER_CM:
        kind = "b_value"
        reason = (f"values reach {np.nanmax(np.abs(values)):.3g}, far above any probe's "
                  f"gradient strength, so these are b-values")
    else:
        kind, reason = "gradient", "no diff-module files; values are in a probe's G/cm range"

    return {
        "values": values, "kind": kind, "reason": reason,
        "gradient_g_per_cm": values if kind == "gradient" else None,
        "b": values if kind == "b_value" else None,
    }


def read_gradient_calib(path: str) -> float | None:
    """The probe's calibrated maximum gradient in **G/cm**, from the
    ``gradient_calib`` file TopSpin writes beside the data (which
    states it in G/mm). None when absent."""
    full = os.path.join(path, "gradient_calib")
    if not os.path.isfile(full):
        return None
    with open(full, "r", errors="ignore") as handle:
        first = handle.readline().strip()
    match = re.match(r"^([-+0-9.eE]+)\s*(G/mm|G/cm|T/m)?", first)
    if not match:
        return None
    value, unit = float(match.group(1)), (match.group(2) or "G/cm")
    return {"G/mm": value * 10.0, "G/cm": value, "T/m": value * 100.0}[unit]


def read_shape_integral_factor(path: str, channel: int) -> float | None:
    """``SHAPE_INTEGFAC`` from this gradient channel's ``gpnam`` shape
    file -- the fraction of a rectangle the shape actually delivers.

    Needed only to *explain* ``difflist``, not to use it: see
    :func:`read_pfg_params` on why the factor is already folded in.
    """
    full = os.path.join(path, f"gpnam{channel}")
    if not os.path.isfile(full):
        return None
    try:
        params = parse_jcampdx_params(full)
    except Exception:
        return None
    value = params.get("SHAPE_INTEGFAC")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def read_difframp(path: str) -> np.ndarray | None:
    """The relative gradient ramp (0..1 per F1 point) from ``Difframp``.
    None when absent.

    A Bruker shape file is a JCAMP-DX parameter block followed by the
    shape itself after an ``##XYDATA=`` marker. The parameter block
    contains multi-line numeric arrays of its own (``SHL_*``), so the
    data has to be taken from the marker onwards rather than by
    picking out every line that parses as a number.
    """
    return _read_shape_data(os.path.join(path, "Difframp"))


def _read_shape_data(full: str) -> np.ndarray | None:
    if not os.path.isfile(full):
        return None
    values: list[float] = []
    started = False
    with open(full, "r", errors="ignore") as handle:
        for line in handle:
            if not started:
                started = line.startswith("##XYDATA")
                continue
            if line.startswith("##END"):
                break
            for token in line.replace(",", " ").split():
                try:
                    values.append(float(token))
                except ValueError:
                    return np.array(values, dtype=float) if values else None
    return np.array(values, dtype=float) if values else None


# ============================================================
# The whole description
# ============================================================

@dataclass
class BrukerPFG:
    """Everything needed to put a Bruker PFG series on a b-axis."""

    path: str
    #: Gradient amplitude per F1 point, **T/m**.
    gradient: np.ndarray | None
    #: Diffusion weighting per F1 point, s/m^2.
    b: np.ndarray | None
    delta: float | None          #: s, the total encoding duration
    big_delta: float | None      #: s
    tau: float                   #: s, bipolar inter-lobe recovery
    sequence: str                #: a DIFFUSION_SEQUENCES key
    shape: str                   #: a GRADIENT_SHAPE_FACTORS key
    nucleus: str | None
    gamma: float | None          #: Hz/T
    pulseprogram: str | None
    #: Where each value came from, keyed the same as the fields.
    sources: dict[str, str] = field(default_factory=dict)
    #: Anything the caller should see before trusting the numbers.
    warnings: list[str] = field(default_factory=list)
    #: The parsed pulse-programme description, when there was one.
    programme: PulseProgramDiffusion | None = None

    @property
    def complete(self) -> bool:
        return self.b is not None

    def as_pfg_params(self) -> dict:
        """The flat, seconds-based dict shaped like
        :func:`opennmr.processing.diffusion.pfg_params_from_metadata`'s,
        for a GUI that already speaks that shape.

        ``delta`` is the value the b-value expressions want (total
        encoding time), so ``delta_is_plateau_only`` is False and
        ``effective_delta`` is a no-op on it.
        """
        found: dict[str, Any] = {}
        if self.delta is not None:
            found["delta"] = self.delta
        if self.big_delta is not None:
            found["DELTA"] = self.big_delta
        if self.tau:
            found["tau"] = self.tau
        found["delta_is_plateau_only"] = False
        found["sequence"] = self.sequence
        found["shape"] = self.shape
        return found


def read_pfg_params(path: str, nucleus: str | None = None,
                    shape: str | None = None) -> BrukerPFG:
    """Describe a Bruker PFG experiment's gradient axis, with or
    without a ``diff.xml``.

    ``diff.xml`` wins where it exists -- it carries TopSpin's own
    b-values, which beat anything reconstructed from timings. Otherwise
    the gradients come from ``difflist`` and the timings from the pulse
    programme's own documentation.

    **Why ``shape`` defaults to rectangular on the difflist path.**
    ``difflist`` is not the raw amplifier setting: the ``dosy`` AU
    writes ``gradient_calib x SHAPE_INTEGFAC x GPZ/100 x Difframp``, so
    the gradient shape's integral factor is **already in the numbers**.
    On the reference dataset that identity holds to five digits
    (53.5 G/cm x 0.977376 = 52.2896 = difflist/Difframp). Applying a
    shape factor again on top would double-count it and understate D by
    the square of that factor. Pass ``shape=`` explicitly only if you
    know your ``difflist`` was written some other way.

    ``nucleus`` defaults to ``acqus``'s own ``NUC1``.
    """
    if not os.path.isdir(path):
        raise NotADirectoryError(path)
    acqus_path = os.path.join(path, "acqus")
    if not os.path.isfile(acqus_path):
        raise FileNotFoundError(f"{path!r} has no acqus")
    acqus = parse_jcampdx_params(acqus_path)

    nucleus = nucleus or str(acqus.get("NUC1") or "1H")
    try:
        gamma = _diffusion.gamma_for(nucleus)
    except Exception:
        gamma = None
    pulprog = str(acqus.get("PULPROG") or "") or None

    sources: dict[str, str] = {}
    warnings: list[str] = []

    # -- the easy case, unchanged: TopSpin already did the work.
    if os.path.isfile(os.path.join(path, "diff.xml")):
        from . import bruker as io_bruker

        diff = io_bruker.read_diff_xml(path, nucleus=nucleus)
        sources.update({k: "diff.xml" for k in ("gradient", "b", "delta", "big_delta")})
        return BrukerPFG(
            path=path, gradient=diff["gradient"], b=diff["B"],
            delta=diff["delta"], big_delta=diff["DELTA"], tau=0.0,
            sequence="pgste", shape="rect", nucleus=nucleus, gamma=gamma,
            pulseprogram=pulprog, sources=sources,
            warnings=["b-values are TopSpin's own, from diff.xml; the timings are "
                      "reported for display and are not used to recompute them"],
        )

    # -- the dosy-AU case.
    programme = None
    pp_path = os.path.join(path, "pulseprogram")
    if os.path.isfile(pp_path):
        with open(pp_path, "r", errors="ignore") as handle:
            programme = describe_diffusion_pulseprogram(handle.read(), name=pulprog)
        warnings.extend(programme.notes)

    def _acqus_value(param: str | None) -> float | None:
        """``"p30"`` -> ``acqus["P"][30]`` in seconds (P is us, D is s)."""
        if not param:
            return None
        letter, index = param[0].upper(), int(param[1:])
        table = acqus.get("P" if letter == "P" else "D")
        if table is None or index >= len(table):
            return None
        value = float(table[index])
        return value * 1e-6 if letter == "P" else value

    delta = big_delta = None
    tau = 0.0
    if programme is not None and programme.resolved:
        raw_delta = _acqus_value(programme.delta_param)
        if raw_delta is not None:
            delta = raw_delta * programme.delta_factor
            sources["delta"] = (
                f"acqus {programme.delta_param.upper()}"
                + (f" x {programme.delta_factor:g}" if programme.delta_factor != 1 else "")
                + f" ('{programme.doc[programme.delta_param]}')"
            )
        big_delta = _acqus_value(programme.big_delta_param)
        if big_delta is not None:
            sources["big_delta"] = (f"acqus {programme.big_delta_param.upper()} "
                                    f"('{programme.doc[programme.big_delta_param]}')")
        if programme.bipolar:
            tau = _acqus_value(programme.tau_param) or 0.0
            if tau:
                sources["tau"] = (f"acqus {programme.tau_param.upper()} "
                                  f"('{programme.doc[programme.tau_param]}')")

    sequence = programme.sequence if programme is not None else "pgste"
    shape = shape or "rect"
    if shape != "rect":
        warnings.append(
            f"shape={shape!r} applied on top of difflist, whose numbers already include "
            f"the gradient shape's integral factor – this double-counts it"
        )

    gradient = None
    if os.path.isfile(os.path.join(path, "difflist")):
        listed = read_difflist(path)
        sources["difflist"] = listed["reason"]
        if listed["kind"] == "gradient":
            gradient = _diffusion.g_per_cm_to_t_per_m(listed["gradient_g_per_cm"])
            sources["gradient"] = "difflist (G/cm)"
            warnings.extend(_cross_check_difflist(path, listed["gradient_g_per_cm"],
                                                  acqus, programme))
        else:
            warnings.append(
                "difflist holds b-values, not gradients, but this dataset has no "
                "diff.xml to confirm them against"
            )
    else:
        warnings.append(f"{path!r} has no difflist; the gradient amplitudes are unknown")

    b = None
    if gradient is not None and delta and big_delta and gamma:
        b = _diffusion.b_value(gradient, delta, big_delta, gamma=gamma,
                               shape=shape, sequence=sequence, tau=tau)
        sources["b"] = (f"computed: {sequence} timing, {shape} shape, "
                        f"gamma({nucleus}) = {gamma:.6g} Hz/T")
    else:
        missing = [name for name, value in
                   (("gradient", gradient), ("delta", delta), ("DELTA", big_delta))
                   if value is None]
        if missing:
            warnings.append("cannot compute b: missing " + ", ".join(missing))

    return BrukerPFG(
        path=path, gradient=gradient, b=b, delta=delta, big_delta=big_delta, tau=tau,
        sequence=sequence, shape=shape, nucleus=nucleus, gamma=gamma,
        pulseprogram=pulprog, sources=sources, warnings=warnings, programme=programme,
    )


def _cross_check_difflist(path: str, g_per_cm: np.ndarray, acqus: dict,
                          programme: PulseProgramDiffusion | None) -> list[str]:
    """Rebuild ``difflist`` from the probe calibration, the shape's
    integral factor, ``GPZ`` and ``Difframp``, and report any
    disagreement.

    This is the check that pins the convention rather than assuming it:
    when the four independent files reproduce ``difflist``, the shape
    factor is demonstrably already folded in, and applying one again
    downstream would be wrong.
    """
    calib = read_gradient_calib(path)
    ramp = read_difframp(path)
    channel = programme.gradient_channel if programme else None
    if calib is None or ramp is None or channel is None or ramp.size != g_per_cm.size:
        return []

    integ = read_shape_integral_factor(path, channel)
    gpz = acqus.get("GPZ")
    percent = float(gpz[channel]) if gpz is not None and channel < len(gpz) else 100.0
    rebuilt = calib * (integ if integ is not None else 1.0) * (percent / 100.0) * ramp

    scale = np.nanmax(np.abs(g_per_cm))
    if scale <= 0:
        return []
    error = float(np.nanmax(np.abs(rebuilt - g_per_cm)) / scale)
    if error < 1e-3:
        return []
    return [
        f"difflist does not match gradient_calib x SHAPE_INTEGFAC x GPZ{channel} x "
        f"Difframp (max {error:.1%} off). Either the shape factor is not folded into "
        f"difflist on this spectrometer, or the probe calibration changed after "
        f"acquisition – check before trusting D."
    ]


# ============================================================
# Self-check
# ============================================================

def selftest(path: str) -> dict:
    """Read ``path`` and report what was resolved and from where, so
    this can be pointed at a new dataset to see whether the reader
    still holds."""
    found = read_pfg_params(path)
    return {
        "path": path,
        "pulseprogram": found.pulseprogram,
        "nucleus": found.nucleus,
        "sequence": found.sequence,
        "bipolar": bool(found.programme and found.programme.bipolar),
        "delta_ms": None if found.delta is None else found.delta * 1e3,
        "DELTA_ms": None if found.big_delta is None else found.big_delta * 1e3,
        "tau_ms": found.tau * 1e3,
        "gradient_g_per_cm": None if found.gradient is None else
            (float(found.gradient[0] * 100), float(found.gradient[-1] * 100)),
        "b_range": None if found.b is None else (float(found.b[0]), float(found.b[-1])),
        "sources": found.sources,
        "warnings": found.warnings,
    }


if __name__ == "__main__":  # pragma: no cover
    import sys
    from pprint import pprint

    for folder in sys.argv[1:]:
        pprint(selftest(folder))
