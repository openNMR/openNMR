"""Processing provenance, written out as prose.

A spectrum carries everything that was done to it in
``metadata.raw`` -- the window and its parameters, zero-fill, linear
prediction, truncation, phase, baseline, referencing. That is enough
to write the sentence a paper's experimental section needs, and
writing it by hand from memory is exactly how a methods section ends
up describing a different processing chain than the one that produced
the figure.

Explicit user request: *"maybe a processing report for consistency
and/or for publication comparison, like spectra are reported using a
windowed EM function and 5 Hz of line broadening. 8 points of
backwards linear prediction are used. Things like that, as a
summary."*

Two renderings of the same facts:

* :func:`processing_summary` -- the structured facts, as a list of
  ``(label, value)`` pairs. For a table, a CSV, or a GUI panel.
* :func:`processing_prose` -- those facts as sentences, in the voice
  an experimental section uses.

Both deliberately report **only what was actually applied**. A step
left at its default is not mentioned, because "no linear prediction
was used" is not a thing anyone writes; and a step whose setting is
unknown is reported as unknown rather than guessed at.
"""

from __future__ import annotations

import numpy as np

#: How each window key is named in prose, and the conventional
#: abbreviation a reader will recognise. ``None`` where there is no
#: standard abbreviation worth quoting.
WINDOW_NAMES = {
    "none": ("no apodization", None),
    "exponential": ("exponential", "EM"),
    "gaussian": ("Gaussian", "GM"),
    "hann": ("Hann", None),
    "sine": ("sine-bell", "SINE"),
    "qsine": ("squared sine-bell", "QSINE"),
    "__recorded__": ("the window recorded in the acquisition file", None),
}

_BASELINE_NAMES = {
    "linear": "a linear fit",
    "polynomial": "a polynomial fit",
    "sinusoidal": "a sinusoidal fit",
    "spline": "a cubic smoothing spline",
    "whittaker": "an asymmetrically reweighted Whittaker smoother",
    "spline_auto": ("a natural cubic spline through automatically placed nodes "
                    "(baseline recognised by the Golotvin–Williams noise test)"),
    "spline_points": "a natural cubic spline through baseline points chosen by hand",
    "manual_polynomial": "a manual polynomial baseline",
    "manual_sine": "a manual sinusoidal baseline",
}


def _number(value: float, decimals: int = 2) -> str:
    """A number as a person would write it: no trailing zeros, no
    scientific notation for the ordinary magnitudes these are."""
    if value == int(value):
        return str(int(value))
    return f"{value:.{decimals}f}".rstrip("0").rstrip(".")


def _window_phrase(raw: dict) -> str | None:
    """The apodization clause, or ``None`` when none was applied."""
    key = raw.get("live_window")
    if key in (None, "none"):
        return None
    name, abbreviation = WINDOW_NAMES.get(key, (key, None))
    if key == "__recorded__":
        return name
    label = f"{name} ({abbreviation})" if abbreviation else name
    if key == "exponential":
        lb = float(raw.get("live_lb", 0.0) or 0.0)
        if lb:
            return f"{label} window function with {_number(lb)} Hz of line broadening"
        return f"{label} window function"
    if key == "gaussian":
        lb = float(raw.get("live_lb", 0.0) or 0.0)
        gb = float(raw.get("live_gb", 0.0) or 0.0)
        bits = []
        if lb:
            bits.append(f"{_number(lb)} Hz of line broadening")
        if gb:
            bits.append(f"a Gaussian factor of {_number(gb, 3)}")
        return f"{label} window function" + (f" with {' and '.join(bits)}" if bits else "")
    if key in ("sine", "qsine"):
        ssb = float(raw.get("live_ssb", 0.0) or 0.0)
        # SSB is Bruker's own convention: 2 is the shifted (pi/2)
        # sine-bell everyone means by "shifted sine-bell", 1 is
        # unshifted. Quoting the number alone would be meaningless to
        # a reader who does not use TopSpin.
        if ssb >= 2:
            return f"a {name} window function shifted by pi/{_number(ssb)} (SSB = {_number(ssb)})"
        return f"an unshifted {name} window function"
    return f"{label} window function"


def _linear_prediction_phrase(raw: dict) -> str | None:
    points = int(raw.get("live_linear_predict_points", 0) or 0)
    if points <= 0:
        return None
    direction = str(raw.get("live_linear_predict_direction", "forward"))
    order = int(raw.get("live_linear_predict_order", 0) or 0)
    # "backwards", not "backward": the user's own wording, and what a
    # methods section normally says.
    word = "backwards" if direction.startswith("back") else "forward"
    phrase = f"{points} points of {word} linear prediction"
    if order:
        phrase += f" (order {order})"
    return phrase


def _baseline_phrase(raw: dict) -> str | None:
    if not raw.get("baseline_correct"):
        return None
    method = str(raw.get("baseline_method", "linear"))
    described = _BASELINE_NAMES.get(method, method)
    if method == "polynomial":
        degree = int(raw.get("baseline_degree", 2) or 2)
        described = f"a degree-{degree} polynomial fit"
    if method in ("manual_polynomial", "manual_sine"):
        described = _manual_baseline_description(raw, method.split("_", 1)[1])
    if method == "spline_points":
        points = raw.get("baseline_spline_x") or []
        described += (" at " + ", ".join(f"{_number(p)}" for p in points) + " ppm") if points else ""
    return f"the baseline was corrected using {described}"


def _manual_baseline_description(raw: dict, model: str) -> str:
    """The model, its coefficients (or that it was refitted per trace)
    and the regions, as a methods section would state them."""
    from ..notebook import manual_baseline as nb_manual

    equation = nb_manual.MANUAL_BASELINE_EQUATIONS[model]
    regions = raw.get("baseline_manual_regions") or []
    where = ("the regions " + ", ".join(f"{_number(min(r))} to {_number(max(r))} ppm" for r in regions)
             if regions else "the first and last 5 % of the spectrum")
    params = raw.get(f"baseline_manual_{model}_params")
    names = nb_manual.MANUAL_BASELINE_PARAMS[model]
    if raw.get("baseline_manual_refit") or params is None or len(params) != len(names):
        return f"{equation} (x in ppm) least-squares fitted to {where} of each trace"
    return f"{nb_manual.describe_manual_baseline(model, params, 4)}, chosen against {where}"


def processing_summary(spectrum) -> list[tuple[str, str]]:
    """Every processing step actually applied to ``spectrum``, as
    ordered ``(label, value)`` pairs.

    Reads ``metadata.raw`` -- the same record the GUI's own controls
    are restored from and the sidecar persists, so this reports what
    the spectrum currently *is*, not what some earlier session asked
    for. Steps left at their defaults are omitted entirely: a table
    listing "linear prediction: none" for every experiment that never
    used it is noise, and this is meant to be pasted somewhere.
    """
    raw = getattr(spectrum.metadata, "raw", {}) or {}
    meta = spectrum.metadata
    rows: list[tuple[str, str]] = []

    if meta.nucleus:
        rows.append(("Nucleus", meta.nucleus))
    if meta.sfo1:
        rows.append(("Larmor frequency", f"{meta.sfo1:.4f} MHz"))
    if meta.pulse_program:
        rows.append(("Pulse programme", str(meta.pulse_program)))

    shift = float(raw.get("live_fid_shift", 0.0) or 0.0)
    if shift:
        rows.append(("FID left-shift", f"{_number(shift)} points"))
    if raw.get("live_align_echo"):
        offset = float(raw.get("live_echo_offset", 0.0) or 0.0)
        rows.append(("Echo alignment", "applied" + (f", offset {_number(offset)} points" if offset else "")))

    truncate = int(raw.get("live_truncate", 0) or 0)
    if truncate:
        rows.append(("Truncated to", f"{truncate} points"))

    linear_prediction = _linear_prediction_phrase(raw)
    if linear_prediction:
        rows.append(("Linear prediction", linear_prediction))

    window = _window_phrase(raw)
    rows.append(("Apodization", window or "none"))

    zero_fill = raw.get("live_zero_fill")
    if zero_fill:
        rows.append(("Zero-filled to", f"{int(zero_fill)} points"))
    zero_fill_f1 = raw.get("live_zero_fill_f1")
    if zero_fill_f1 and int(zero_fill_f1) > 1:
        rows.append(("Zero-filled to (F1)", f"{int(zero_fill_f1)} points"))

    phase0 = float(raw.get("phase0", 0.0) or 0.0)
    phase1 = float(raw.get("phase1", 0.0) or 0.0)
    if phase0 or phase1:
        rows.append(("Phase correction", f"PH0 = {_number(phase0, 1)}°, PH1 = {_number(phase1, 1)}°"))
    if raw.get("slice_phase0") and any(raw["slice_phase0"]):
        rows.append(("Phase correction", "each slice phased independently"))

    baseline = _baseline_phrase(raw)
    if baseline:
        method = str(raw.get("baseline_method", "linear"))
        rows.append(("Baseline correction", _BASELINE_NAMES.get(method, method)))

    sr = float(raw.get("sr_shift_hz", 0.0) or 0.0)
    if sr:
        rows.append(("Referencing", f"spectrum shifted by {_number(sr, 3)} Hz"))
    if raw.get("magnitude_mode"):
        rows.append(("Display mode", "magnitude"))

    return rows


def processing_prose(spectrum, subject: str = "Spectra") -> str:
    """The same facts as :func:`processing_summary`, written as the
    sentences an experimental section uses.

    ``subject`` : what the sentences are about ("Spectra", "The 1H
    spectrum", ...). Only the grammar of the first clause depends on
    it, and it is not pluralised automatically -- pass what you want
    to read.

    Returns ``""`` when nothing was applied at all, rather than a
    sentence saying so: an empty methods paragraph is easier to notice
    and fix than a confidently wrong one.
    """
    raw = getattr(spectrum.metadata, "raw", {}) or {}
    sentences: list[str] = []

    # Sentence 1: the FID-domain chain, in the order it is applied.
    processing: list[str] = []
    shift = float(raw.get("live_fid_shift", 0.0) or 0.0)
    if shift:
        processing.append(f"left-shifted by {_number(shift)} points")
    truncate = int(raw.get("live_truncate", 0) or 0)
    if truncate:
        processing.append(f"truncated to {truncate} points")
    linear_prediction = _linear_prediction_phrase(raw)
    if linear_prediction:
        processing.append(f"extended by {linear_prediction}")
    window = _window_phrase(raw)
    if window:
        processing.append(f"apodized with {'an' if window[0] in 'aeiou' else 'a'} {window}")
    zero_fill = raw.get("live_zero_fill")
    if zero_fill:
        processing.append(f"zero-filled to {int(zero_fill)} points")

    if processing:
        sentences.append(f"{subject} were processed by " + _join(processing) + ".")

    # Sentence 2: what happened after the transform.
    post: list[str] = []
    phase0 = float(raw.get("phase0", 0.0) or 0.0)
    phase1 = float(raw.get("phase1", 0.0) or 0.0)
    if raw.get("slice_phase0") and any(raw["slice_phase0"]):
        post.append("each slice was phased independently")
    elif phase0 or phase1:
        post.append(
            f"a phase correction of {_number(phase0, 1)}° (zero-order) and "
            f"{_number(phase1, 1)}° (first-order) was applied"
        )
    baseline = _baseline_phrase(raw)
    if baseline:
        post.append(baseline)
    sr = float(raw.get("sr_shift_hz", 0.0) or 0.0)
    if sr:
        post.append(f"the spectrum was referenced with a {_number(sr, 3)} Hz shift")
    if post:
        sentences.append(_capitalise(_join(post)) + ".")

    return " ".join(sentences)


def _join(parts: list[str]) -> str:
    """``"a, b and c"`` -- serial comma deliberately omitted, matching
    the house style of most chemistry journals."""
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + " and " + parts[-1]


def _capitalise(text: str) -> str:
    return text[:1].upper() + text[1:] if text else text


def format_processing_report(spectrum, label: str = "", subject: str = "Spectra") -> str:
    """A whole report: the prose paragraph, then the facts it was
    written from, as an aligned two-column table.

    Both, not one or the other -- the paragraph is what gets pasted
    into a manuscript, and the table is what makes it checkable
    against another dataset without re-reading the prose word by word.
    """
    lines: list[str] = []
    if label:
        lines += [f"Processing report — {label}", "=" * min(len(label) + 21, 60), ""]

    prose = processing_prose(spectrum, subject=subject)
    lines.append(prose if prose else "No processing has been applied to this spectrum.")
    lines.append("")

    rows = processing_summary(spectrum)
    if rows:
        width = max(len(name) for name, _value in rows)
        lines += [f"{name:<{width}}  {value}" for name, value in rows]
    return "\n".join(lines)


def compare_processing(spectra, labels=None) -> str:
    """Line up several spectra's processing side by side, so an
    inconsistency between datasets that are about to be compared is
    visible before the comparison is made rather than after.

    The explicit "for consistency" half of the request. Every setting
    any of them used gets a row; a spectrum that did not use it shows
    ``—``, and rows where they all agree are marked so the eye goes to
    the ones where they do not.
    """
    spectra = list(spectra)
    if not spectra:
        return "Nothing to compare."
    if labels is None:
        labels = [f"#{i + 1}" for i in range(len(spectra))]
    labels = [str(name) for name in labels]

    summaries = [dict(processing_summary(s)) for s in spectra]
    # Every setting any of them recorded, in the order the first
    # spectrum to mention it listed it -- which is processing order,
    # since processing_summary emits in processing order.
    names: list[str] = []
    for summary in summaries:
        for name in summary:
            if name not in names:
                names.append(name)

    label_width = max(len(name) for name in names)
    column_widths = [max(len(label), *(len(s.get(name, "—")) for name in names))
                     for label, s in zip(labels, summaries)]

    header = " " * (label_width + 2) + "  ".join(
        f"{label:<{w}}" for label, w in zip(labels, column_widths))
    lines = [header, "-" * len(header.rstrip())]
    for name in names:
        values = [s.get(name, "—") for s in summaries]
        agree = len(set(values)) == 1
        row = f"{name:<{label_width}}  " + "  ".join(
            f"{value:<{w}}" for value, w in zip(values, column_widths))
        # A marker, not colour: this gets pasted into plain text.
        lines.append(row.rstrip() + ("" if agree else "   <-- differs"))
    return "\n".join(lines)
