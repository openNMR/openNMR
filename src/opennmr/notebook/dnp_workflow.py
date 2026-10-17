"""
DNP, step by step: field profiles and enhancement factors from spectra
that are **aligned first, then measured** (explicit user request,
2026-09-30). The GUI's Integrate ▸ DNP sections call these functions in the
same order a notebook does::

    from opennmr.notebook import dnp_workflow as dw

    info = dw.spectrum_fields(spectra)                         # 1. B0 of each
    al = dw.align_spectra(spectra, reference=0, peak_ppm=250)  # 2. align on a peak
    dw.set_manual_shift(al, 3, -120.0)                         # 3. correct one by hand
    aligned = dw.aligned_spectra(spectra, al)                  #    ppm, own SR each
    m = dw.measure(aligned, peak_ppm=250, region=(240, 260))   # 4. height and integral
    fields = dw.alignment_fields(spectra, al)                  #    B0 at the line
    prof = dw.field_profile(m, fields, mode="ratio", off=0)    # 5. against B0
    dw.plot_field_profile(prof)
    tab = dw.enhancement_table(m, off=1)                       # or: enhancements

**1. Fields.** ``B0 = SFO1 / gamma`` for each spectrum
(:func:`spectrum_fields`): the field at which the nucleus resonates at
the carrier. That assumes the line sat *on* the carrier. It rarely does
-- SFO1 is set by hand, and in the first real 23Na sweep the line sat up
to 14 kHz (1.2 mT) off it, against 0.4 mT steps -- so once the spectra
are aligned, the line's own frequency in each spectrum is known, and
:func:`alignment_fields` gives ``B0 = nu_line / (gamma (1 + delta))``,
the direct measurement at the sample. Both are reported; the GUI offers
both and defaults to the line.

**2. Alignment** (:func:`align_spectra`). Nothing about a spectrum's
*position* can be trusted between experiments: the field moves the line,
and the carrier (receiver offset) is moved by hand to follow it. What
does survive is the line's **shape**. So the reference spectrum's line,
around the picked peak (+-3 FWHM by default), is the template, and each
spectrum is searched for the offset at which it matches that template
best -- the normalised cross-correlation (Pearson r) of the template
with the spectrum, both scaled to unit height in effect, over +-``window``
of the reference line's offset from *its* carrier. The search runs on the
carrier-offset axis (Hz), where a field-swept line stays near zero when
SFO1 follows it. ``|r|`` is maximised, so an inverted line (negative
enhancement) is found too, and its sign reported. A match is accepted
when ``|r| >= min_score`` *and* the matched signal is ``min_snr`` noise
deviations above the baseline -- near a zero-crossing of the profile
there is nothing to find, and a noise patch that happens to correlate is
not passed off as the line. Unmatched spectra keep the reference line's
carrier offset, flagged, for a check by eye. Each spectrum's FWHM is
reported against the reference's: a very different width is a warning
that the line changed (or a different line was matched).

**3. Manual correction.** :func:`set_manual_shift` adds a hand-set shift
(Hz) on top of the automatic one; :func:`shift_for_display_ppm`
converts a drag on the ppm axis to it.

**Aligned ppm axes: each spectrum its own SR.** Spectrum *i*'s ppm axis
is rebuilt so its line sits at the reference line's shift ``delta_ref``:
the reference frequency becomes ``nu0_i = nu_line,i / (1 + delta_ref)``
and ``ppm = (nu / nu0_i - 1) * 1e6`` (:func:`aligned_spectrum`) -- exactly
what setting a separate SR per spectrum does. The Hz axis is kept, so
integrals (always taken against Hz) are unaffected by the relabelling.

**4. Measurement** (:func:`measure`): the signed height at a picked
shift and the integral over a region, on every aligned spectrum, with
the local-baseline, noise-error and systematic (baseline window +-1
FWHM; region +-20%) conventions of :mod:`opennmr.notebook.dnp`. Doing
both is encouraged: they agree only if the line shape is unchanged.
Intensities are per scan (openNMR's importers divide by NS) and, by
default, scaled to the reference spectrum's receiver gain.

**5. Against the field** (:func:`field_profile`): absolute, the ratio to
an OFF (microwaves-off) spectrum, or one minus that ratio. Enhancements
of chosen ON spectra against an OFF one: :func:`enhancement_table`.
"""

from __future__ import annotations

import copy
import csv
import math

import numpy as np

from ..core.spectrum import NMRAxis, NMRSpectrum
from . import dnp

__all__ = [
    "spectrum_fields", "align_spectra", "set_manual_shift", "shift_for_display_ppm",
    "alignment_fields", "aligned_reference_mhz", "aligned_spectrum", "aligned_spectra",
    "measure", "field_profile", "enhancement_table", "alignment_report", "measurement_report",
    "profile_report", "enhancement_report", "profile_to_csv", "plot_aligned_spectra",
    "plot_field_profile", "PROFILE_MODES", "FIELD_SOURCES",
]

#: How the profile's y axis is expressed (:func:`field_profile`).
PROFILE_MODES = ("absolute", "ratio", "one_minus")
#: Where each spectrum's B0 comes from (:func:`alignment_fields`).
FIELD_SOURCES = ("line", "sfo1")

#: Default template half-width, in reference-line FWHMs.
MATCH_FWHMS = 3.0
#: Default acceptance: |Pearson r| of the matched line with the template.
MIN_SCORE = 0.6
#: A line wider or narrower than the reference's by more than this
#: factor is flagged.
WIDTH_FACTOR = 1.5


# ============================================================
# 1. Fields
# ============================================================

def spectrum_fields(spectra) -> dict:
    """Step 1: every spectrum's B0 from its nucleus and carrier (SFO1),
    ``B0 = SFO1 / gamma`` -- the field at which the nucleus resonates on
    the carrier. Returns ``{"field_T", "sfo1_MHz", "carrier_source",
    "nucleus", "gamma_Hz_per_T", "notes"}`` (arrays in the given order).
    Refuses a mix of nuclei (not one sweep)."""
    spectra = list(spectra)
    if not spectra:
        raise ValueError("Give at least one spectrum.")
    nuclei = []
    for k, s in enumerate(spectra):
        dnp._one_d(s, f"spectrum {k + 1}")
        nuclei.append(dnp._nucleus(s))
    if len(set(nuclei)) > 1:
        raise ValueError(f"The spectra observe different nuclei ({', '.join(sorted(set(nuclei)))}): "
                         "a field profile or enhancement needs one nucleus.")
    nucleus = nuclei[0]
    gamma = dnp._gamma_hz_per_t(nucleus)
    sfo1, sources, notes = [], [], []
    for k, s in enumerate(spectra):
        value, source = dnp.carrier_frequency_mhz(s)
        sfo1.append(value)
        sources.append(source)
        if source != "SFO1":
            notes.append(f"Spectrum {k + 1}: no SFO1 recorded; carrier from {source}.")
    sfo1 = np.asarray(sfo1, dtype=float)
    return {"field_T": sfo1 * 1e6 / gamma, "sfo1_MHz": sfo1, "carrier_source": sources,
            "nucleus": nucleus, "gamma_Hz_per_T": gamma, "notes": notes}


# ============================================================
# 2. Alignment
# ============================================================

def _sorted_xy(x, y):
    order = np.argsort(x)
    return np.asarray(x, dtype=float)[order], np.asarray(y, dtype=float)[order]


def _fwhm_at(x, y, j, hint: float | None = None) -> float:
    """FWHM (in x's units) of the line at index ``j`` (x ascending).

    With ``hint`` (an expected FWHM), the half height is taken above a
    straight line through the data at +-2.5 ``hint`` -- so a line on the
    flank of a broad feature, or on a rolled baseline, is not measured
    wide or narrow by the slope under it."""
    if hint is None or not np.isfinite(hint) or hint <= 0:
        lo, hi = dnp._line_window(x, y, j, fwhms=1.0)
        return float(0.5 * abs(hi - lo))
    step = float(np.median(np.diff(x))) if x.size > 1 else 1.0
    pad = max(0.25 * hint, 3 * step)
    anchors = []
    for side in (-2.5, 2.5):
        c = x[j] + side * hint
        sel = (x >= c - pad) & (x <= c + pad)
        if np.count_nonzero(sel):
            anchors.append((float(np.mean(x[sel])), float(np.median(y[sel]))))
    if len(anchors) == 2:
        (x1, y1), (x2, y2) = anchors
        base = y1 + (y2 - y1) * (x - x1) / (x2 - x1)
    else:
        base = np.full_like(y, float(np.median(y)))
    d = y - base
    top = d[j]
    half = 0.5 * abs(top)
    left = j
    while left > 0 and abs(d[left]) > half and np.sign(d[left]) == np.sign(top):
        left -= 1
    right = j
    while right < d.size - 1 and abs(d[right]) > half and np.sign(d[right]) == np.sign(top):
        right += 1

    def cross(a, b):
        # Linear interpolation of the half-height crossing between a and b.
        da, db = abs(d[a]), abs(d[b])
        if da == db:
            return x[a]
        return x[a] + (half - da) * (x[b] - x[a]) / (db - da)
    return float(abs(cross(right - 1, right) - cross(left + 1, left)))


def _detrended(values):
    """``values`` less its least-squares straight line (over its index)."""
    values = np.asarray(values, dtype=float)
    k = np.arange(values.size, dtype=float)
    if values.size < 3:
        return values - values.mean()
    b1, b0 = np.polyfit(k, values, 1)
    return values - (b0 + b1 * k)


def _line_centre(x, y, j, fwhm) -> float:
    """Centre of the line at index ``j`` (x ascending): the vertex of a
    parabola fitted, least squares, to its top (within +-FWHM/3). Many
    points, not three: at the top of a line the curvature between
    neighbouring points is comparable to the noise, and a three-point
    vertex wanders by half a point."""
    lo, hi = x[j] - fwhm / 3.0, x[j] + fwhm / 3.0
    sel = (x >= lo) & (x <= hi)
    if np.count_nonzero(sel) < 5:
        return float(x[j])
    xs, ys = x[sel] - x[j], y[sel]
    a, b, _c = np.polyfit(xs, ys, 2)
    if a == 0:
        return float(x[j])
    vertex = -b / (2 * a)
    return float(x[j] + vertex) if abs(vertex) <= fwhm / 3.0 else float(x[j])


def _sliding_pearson(template, signal):
    """Correlation of ``template`` (length m) with every length-m window
    of ``signal``, **each window less its own straight line**: a line
    sitting on the sloping flank of a broad feature (or a rolled
    baseline) still matches the template, which a plain Pearson r would
    mark down. ``template`` is detrended the same way here. r = 0 where a
    window is flat."""
    m = len(template)
    t = _detrended(template)
    t_norm = math.sqrt(float(np.dot(t, t)))
    if t_norm == 0 or len(signal) < m:
        return np.zeros(max(len(signal) - m + 1, 0))
    signal = np.asarray(signal, dtype=float)
    # t is orthogonal to a constant and to a line, so the numerator needs
    # no detrending of the signal; its norm does.
    num = np.correlate(signal, t, mode="valid")
    idx = np.arange(signal.size, dtype=float)
    c1 = np.concatenate(([0.0], np.cumsum(signal)))
    c2 = np.concatenate(([0.0], np.cumsum(signal * signal)))
    ci = np.concatenate(([0.0], np.cumsum(signal * idx)))
    s1 = c1[m:] - c1[:-m]
    s2 = c2[m:] - c2[:-m]
    starts = np.arange(s1.size, dtype=float)
    # sum over the window of S * (j - (m-1)/2), j the index inside it
    su = (ci[m:] - ci[:-m]) - starts * s1 - 0.5 * (m - 1) * s1
    u_norm2 = m * (m * m - 1) / 12.0
    var = np.clip(s2 - s1 * s1 / m - su * su / u_norm2, 0.0, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(var > 1e-300 * max(1.0, float(np.max(np.abs(s2)))), num / (t_norm * np.sqrt(var)),
                     0.0)
    return np.clip(r, -1.0, 1.0)


def _local_maxima(values, floor):
    """Indices of the local maxima of ``values`` at or above ``floor``."""
    v = np.asarray(values, dtype=float)
    if v.size < 3:
        return [int(np.argmax(v))] if v.size else []
    inner = np.flatnonzero((v[1:-1] >= v[:-2]) & (v[1:-1] >= v[2:]) & (v[1:-1] >= floor)) + 1
    return [int(k) for k in inner]


def _parabolic(values, k) -> float:
    """Sub-sample position of the extremum at ``k`` (parabola through
    k-1, k, k+1)."""
    if 0 < k < len(values) - 1:
        a, b, c = values[k - 1], values[k], values[k + 1]
        denom = a - 2 * b + c
        if denom != 0:
            return float(k + 0.5 * (a - c) / denom)
    return float(k)


def _line_frequency(entry) -> float:
    return entry["carrier_MHz"] * 1e6 + entry["line_offset_hz"]


def _refresh_entry(alignment: dict, i: int) -> None:
    e = alignment["entries"][i]
    e["shift_hz"] = e["auto_shift_hz"] + e["manual_shift_hz"]
    e["line_offset_hz"] = alignment["ref_offset_hz"] + e["shift_hz"]
    e["line_frequency_hz"] = _line_frequency(e)


def align_spectra(spectra, reference: int = 0, peak_ppm: float | None = None,
                  window_hz: float = dnp.DEFAULT_WINDOW_HZ, match_halfwidth_hz: float | None = None,
                  min_score: float = MIN_SCORE, min_snr: float = dnp.DEFAULT_MIN_SNR,
                  snap_ppm: float | None = None, labels=None) -> dict:
    """Step 2: find the reference line in every spectrum.

    ``reference`` : index of the reference spectrum. ``peak_ppm`` : the
    line to align on, on the reference's own ppm axis (the largest
    |signal| within ``snap_ppm`` of it is taken; default: the reference's
    largest |signal|). ``window_hz`` : search +- this far (Hz) from the
    reference line's offset from its carrier. ``match_halfwidth_hz`` :
    the template's half-width (default 3 FWHM of the reference line).

    Returns ``{"reference", "peak_ppm", "ref_offset_hz", "ref_fwhm_hz",
    "match_halfwidth_hz", "window_hz", "min_score", "min_snr",
    "entries": [...], "notes", "confirmed": False}``; each entry holds
    ``auto_shift_hz`` (the line's carrier offset minus the reference's),
    ``manual_shift_hz`` (0), ``shift_hz`` (their sum), ``score`` (|r|),
    ``sign`` (+1, or -1 for an inverted line), ``snr``, ``found``,
    ``fwhm_hz``, ``width_ratio``, ``carrier_MHz``, ``line_offset_hz``,
    ``line_frequency_hz``, ``label``, ``note``."""
    spectra = list(spectra)
    n = len(spectra)
    if n == 0:
        raise ValueError("Give at least one spectrum.")
    if not 0 <= reference < n:
        raise ValueError(f"reference must be 0..{n - 1}, not {reference}.")
    labels = list(labels) if labels is not None else [f"Spectrum {k + 1}" for k in range(n)]
    for k, s in enumerate(spectra):
        dnp._one_d(s, labels[k])
    ref = spectra[reference]
    ppm = np.asarray(ref.axes[-1].ppm, dtype=float)
    y_ref = np.asarray(ref.data, dtype=float)
    if peak_ppm is None:
        j = int(np.argmax(np.abs(y_ref - np.median(y_ref))))
    else:
        snap = 0.005 * float(np.ptp(ppm)) if snap_ppm is None else float(snap_ppm)
        j, _edge = dnp._find_extremum(ppm, y_ref, float(peak_ppm), snap)
    off_ref = dnp.carrier_offset_hz(ref)
    xr, yr = _sorted_xy(off_ref, y_ref)
    jr = int(np.argmin(np.abs(xr - float(off_ref[j]))))
    fwhm_ref = _fwhm_at(xr, yr, jr)
    fwhm_ref = _fwhm_at(xr, yr, jr, fwhm_ref)          # refined over its own baseline
    # The line's centre to a fraction of a point: every other spectrum is
    # placed relative to it, and one 12 Hz point is ~1 uT for 23Na.
    o_r = _line_centre(xr, yr, jr, fwhm_ref)
    ascending = off_ref[0] < off_ref[-1]
    peak = float(np.interp(o_r, off_ref if ascending else off_ref[::-1],
                           ppm if ascending else ppm[::-1]))
    step_ref = float(np.median(np.diff(xr)))
    h = float(match_halfwidth_hz) if match_halfwidth_hz else max(MATCH_FWHMS * fwhm_ref,
                                                                 15 * step_ref)
    w = float(window_hz)
    notes = []
    entries = []
    for i, s in enumerate(spectra):
        carrier, _src = dnp.carrier_frequency_mhz(s)
        entry = {"label": labels[i], "carrier_MHz": carrier, "auto_shift_hz": 0.0,
                 "manual_shift_hz": 0.0, "score": 1.0, "sign": 1, "snr": math.nan,
                 "found": True, "fwhm_hz": fwhm_ref, "width_ratio": 1.0, "note": ""}
        xi, yi = _sorted_xy(dnp.carrier_offset_hz(s), s.data)
        noise = dnp._noise_sigma(np.asarray(s.data, dtype=float))
        if i == reference:
            height = abs(dnp._height_at(xr, yr, jr, "local"))
            entry["snr"] = height / noise if noise > 0 else math.inf
            entry["note"] = "reference"
            entries.append(entry)
            continue
        dx = min(step_ref, float(np.median(np.diff(xi))))
        u = np.arange(-h, h + 0.5 * dx, dx)
        template = np.interp(o_r + u, xr, yr)
        grid = np.arange(o_r - w - h, o_r + w + h + 0.5 * dx, dx)
        inside = (grid >= xi[0]) & (grid <= xi[-1])
        if np.count_nonzero(inside) < len(u) + 2:
            entry.update(found=False, score=0.0,
                         note="the search window lies outside this spectrum")
            entries.append(entry)
            continue
        signal = np.interp(grid, xi, yi)
        r = _sliding_pearson(template, signal)
        # Windows reaching past the spectrum's own points match nothing.
        centres = grid[:len(r)] + h
        valid = (centres - h >= xi[0]) & (centres + h <= xi[-1])
        r = np.where(valid, r, 0.0)
        ar = np.abs(r)
        best = float(ar.max()) if ar.size else 0.0
        # Every good match, not only the best: a broad feature can
        # correlate nearly as well as the line itself, so among matches
        # within 0.15 of the best, the one whose width is closest to the
        # reference line's wins (shape first, then width).
        candidates = _local_maxima(ar, max(best - 0.15, 0.0)) or [int(np.argmax(ar))]
        chosen = None
        for k in candidates:
            kk = _parabolic(ar, k)
            centre_k = float(grid[0] + kk * dx + h)
            j_k, _edge = dnp._find_extremum(xi, yi, centre_k, max(0.5 * fwhm_ref, 2 * dx))
            height_k = abs(dnp._height_at(xi, yi, j_k, "local"))
            snr_k = height_k / noise if noise > 0 else math.inf
            fwhm_k = _fwhm_at(xi, yi, j_k, fwhm_ref) if snr_k >= min_snr else math.nan
            closeness = (min(fwhm_k, fwhm_ref) / max(fwhm_k, fwhm_ref)
                         if np.isfinite(fwhm_k) and fwhm_k > 0 else 0.5)
            merit = float(ar[k]) * closeness ** 0.5
            if chosen is None or merit > chosen[0]:
                chosen = (merit, k, centre_k, snr_k, fwhm_k)
        _merit, k, centre, snr, fwhm_i = chosen
        score = float(ar[k])
        sign = 1 if r[k] >= 0 else -1
        found = bool(score >= min_score and snr >= min_snr)
        entry.update(score=score, sign=sign, snr=float(snr), found=found,
                     fwhm_hz=fwhm_i, width_ratio=(fwhm_i / fwhm_ref if fwhm_ref else math.nan))
        if found:
            entry["auto_shift_hz"] = centre - o_r
            if abs(centre - o_r) > 0.95 * w:
                entry["note"] = "matched at the edge of the search window: widen it and check"
            elif np.isfinite(fwhm_i) and not (1 / WIDTH_FACTOR <= fwhm_i / fwhm_ref <= WIDTH_FACTOR):
                entry["note"] = (f"line {fwhm_i / fwhm_ref:.2f}x the reference's width: check "
                                 "it is the same line")
            elif sign < 0:
                entry["note"] = "inverted line (negative signal)"
        else:
            why = []
            if score < min_score:
                why.append(f"best |r| {score:.2f} < {min_score:g}")
            if snr < min_snr:
                why.append(f"S/N {snr:.1f} < {min_snr:g}")
            entry["note"] = ("no line matched (" + ", ".join(why) + "): left at the reference "
                             "line's carrier offset -- check it by eye")
        entries.append(entry)
    alignment = {"reference": int(reference), "peak_ppm": peak, "ref_offset_hz": o_r,
                 "ref_fwhm_hz": fwhm_ref, "match_halfwidth_hz": h, "window_hz": w,
                 "min_score": float(min_score), "min_snr": float(min_snr),
                 "entries": entries, "notes": notes, "confirmed": False}
    for i in range(n):
        _refresh_entry(alignment, i)
    missed = [e["label"] for e in entries if not e["found"]]
    if missed:
        notes.append(f"No line matched in {len(missed)} spectrum(s) ({', '.join(missed)}): "
                     "check them in the aligned view and move them by hand if needed.")
    return alignment


def set_manual_shift(alignment: dict, index: int, shift_hz: float) -> dict:
    """Step 3: a hand-set shift (Hz, on top of the automatic one) for
    spectrum ``index``. Un-confirms the alignment. Returns it."""
    alignment["entries"][index]["manual_shift_hz"] = float(shift_hz)
    _refresh_entry(alignment, index)
    alignment["confirmed"] = False
    return alignment


def aligned_reference_mhz(alignment: dict, index: int) -> float:
    """Spectrum ``index``'s own reference frequency (its "SR"): the
    frequency at which its line is at ``peak_ppm``, MHz."""
    e = alignment["entries"][index]
    return e["line_frequency_hz"] / (1.0 + alignment["peak_ppm"] * 1e-6) / 1e6


def shift_for_display_ppm(alignment: dict, index: int, delta_ppm: float) -> float:
    """The change in ``shift_hz`` that moves spectrum ``index``'s trace by
    ``delta_ppm`` on the aligned ppm axis (a drag to higher ppm lowers
    the frequency taken as the line's)."""
    return -float(delta_ppm) * 1e-6 * alignment["entries"][index]["line_frequency_hz"]


def aligned_spectrum(spectrum: NMRSpectrum, alignment: dict, index: int) -> NMRSpectrum:
    """Spectrum ``index`` on its aligned ppm axis (its line at the
    reference's ``peak_ppm``): a copy, with ``metadata.sfo1`` its own
    reference frequency and ``metadata.raw["dnp_alignment"]`` saying so.
    The Hz axis (relative to the original reference) is kept."""
    nu0 = aligned_reference_mhz(alignment, index) * 1e6
    absolute = dnp.absolute_frequency_hz(spectrum)
    ppm = (absolute / nu0 - 1.0) * 1e6
    axis = spectrum.axes[-1]
    hz = (np.asarray(axis.hz, dtype=float) if axis.hz is not None
          else absolute - dnp.reference_frequency_mhz(spectrum) * 1e6)
    metadata = copy.deepcopy(spectrum.metadata)
    e = alignment["entries"][index]
    metadata.raw = dict(metadata.raw or {})
    metadata.raw["dnp_alignment"] = {
        "reference": alignment["reference"], "peak_ppm": alignment["peak_ppm"],
        "shift_hz": e["shift_hz"], "auto_shift_hz": e["auto_shift_hz"],
        "manual_shift_hz": e["manual_shift_hz"], "own_reference_MHz": nu0 / 1e6,
        "original_reference_MHz": dnp.reference_frequency_mhz(spectrum)}
    metadata.sfo1 = nu0 / 1e6
    return NMRSpectrum(data=np.array(spectrum.data, dtype=float, copy=True),
                       axes=[NMRAxis(kind="frequency", ppm=ppm, hz=hz, label=axis.label)],
                       metadata=metadata)


def aligned_spectra(spectra, alignment: dict) -> list[NMRSpectrum]:
    """Every spectrum on its aligned ppm axis (:func:`aligned_spectrum`)."""
    return [aligned_spectrum(s, alignment, i) for i, s in enumerate(spectra)]


def alignment_fields(spectra, alignment: dict | None = None, source: str = "line") -> dict:
    """B0 of every spectrum: ``source="line"`` (default, needs an
    ``alignment``) -- from the line's own frequency, ``nu_line / (gamma (1
    + delta))`` with delta the reference line's shift; ``"sfo1"`` -- from
    the carrier, as :func:`spectrum_fields`. Returns ``{"field_T",
    "source", "field_sfo1_T", "field_line_T", "difference_mT"}``."""
    if source not in FIELD_SOURCES:
        raise ValueError(f"source must be one of {FIELD_SOURCES}, not {source!r}.")
    info = spectrum_fields(spectra)
    sfo1_field = info["field_T"]
    line_field = None
    if alignment is not None:
        line_hz = np.array([e["line_frequency_hz"] for e in alignment["entries"]], dtype=float)
        line_field = dnp.field_from_frequency(line_hz, info["nucleus"], alignment["peak_ppm"])
    if source == "line" and line_field is None:
        raise ValueError("B0 at the line needs the spectra aligned first (align_spectra).")
    field = line_field if source == "line" else sfo1_field
    diff = (line_field - sfo1_field) * 1e3 if line_field is not None else None
    return {"field_T": np.asarray(field, dtype=float), "source": source,
            "field_sfo1_T": sfo1_field, "field_line_T": line_field, "difference_mT": diff,
            "nucleus": info["nucleus"]}


# ============================================================
# 4. Measurement
# ============================================================

def measure(aligned, peak_ppm: float | None = None, region=None, baseline: str = "local",
            snap_ppm: float | None = None, correct_rg: bool = True, reference: int = 0,
            noise_region=None, labels=None) -> dict:
    """Step 4: the height at ``peak_ppm`` and/or the integral over
    ``region`` (ppm, on the aligned axes) of every aligned spectrum.

    Heights are the largest |signal| within ``snap_ppm`` (default 0.2% of
    the reference's width) of the pick, above a local baseline;
    integrals are taken against Hz. Errors are the noise; ``*_sys`` the
    spread from the baseline window (+-1 FWHM) or region (+-20%). With
    ``correct_rg`` every value is scaled to the receiver gain of spectrum
    ``reference``. Returns ``{"labels", "height": {...} | None, "area":
    {...} | None, "rg", "rg_scale", "notes", "settings"}``, each quantity
    holding ``value``, ``error``, ``systematic`` (arrays) and, for heights,
    ``position`` and ``snr``."""
    aligned = list(aligned)
    if peak_ppm is None and region is None:
        raise ValueError("Pick a shift, integrate a region, or (better) both.")
    dnp._check_baseline(baseline)
    n = len(aligned)
    labels = list(labels) if labels is not None else [f"Spectrum {k + 1}" for k in range(n)]
    notes = []
    rgs = [dnp._rg(s) for s in aligned]
    scale = np.ones(n)
    if correct_rg:
        rg_ref = rgs[reference]
        if rg_ref and all(rgs):
            scale = np.array([rg_ref / g for g in rgs], dtype=float)
            if np.any(np.abs(scale - 1) > 1e-12):
                notes.append("Receiver gains differ: every value is scaled to the reference "
                             f"spectrum's RG ({rg_ref:g}).")
        elif len(set(rgs)) > 1:
            notes.append("Receiver gain not recorded for every spectrum, so not corrected -- "
                         "check they were acquired with the same RG.")
    elif len({g for g in rgs if g}) > 1:
        notes.append("Receiver gains differ and were NOT corrected.")
    notes.append("Values are per scan (openNMR's importers divide by NS).")
    out = {"labels": labels, "height": None, "area": None, "rg": rgs, "rg_scale": scale,
           "notes": notes,
           "settings": {"peak_ppm": peak_ppm, "region": (None if region is None
                                                         else tuple(sorted(map(float, region)))),
                        "baseline": baseline, "snap_ppm": snap_ppm, "correct_rg": correct_rg,
                        "reference": reference}}
    if peak_ppm is not None:
        ref_ppm = np.asarray(aligned[reference].axes[-1].ppm, dtype=float)
        snap = 0.002 * float(np.ptp(ref_ppm)) if snap_ppm is None else float(snap_ppm)
        value, error, sys_, pos, snr = (np.full(n, np.nan) for _ in range(5))
        for i, s in enumerate(aligned):
            p = dnp.peak_height(s, peak_ppm, "ppm", snap, baseline, noise_region)
            value[i], error[i] = p["value"] * scale[i], p["error"] * scale[i]
            pos[i], snr[i] = p["position"], p["snr"]
            if baseline == "local":
                alts = [dnp.peak_height(s, peak_ppm, "ppm", snap, baseline, noise_region,
                                        k)["value"] * scale[i]
                        for k in (dnp.BASELINE_FWHMS - 1.0, dnp.BASELINE_FWHMS + 1.0)]
                sys_[i] = max(abs(a - value[i]) for a in alts)
            else:
                sys_[i] = 0.0
            if p["at_edge"]:
                notes.append(f"{labels[i]}: the largest signal is at the edge of the snap "
                             "window -- check the alignment there.")
        out["height"] = {"value": value, "error": error, "systematic": sys_, "position": pos,
                         "snr": snr}
    if region is not None:
        lo, hi = sorted(map(float, region))
        value, error, sys_ = (np.full(n, np.nan) for _ in range(3))
        centre, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
        for i, s in enumerate(aligned):
            a = dnp.region_integral(s, lo, hi, "ppm", baseline, noise_region)
            value[i], error[i] = a["value"] * scale[i], a["error"] * scale[i]
            spread = 0.0
            for f in (0.8, 1.2):
                try:
                    b = dnp.region_integral(s, centre - f * half, centre + f * half, "ppm",
                                            baseline, noise_region)["value"] * scale[i]
                except ValueError:
                    continue
                spread = max(spread, abs(b - value[i]))
            sys_[i] = spread
        out["area"] = {"value": value, "error": error, "systematic": sys_}
    if out["height"] is None:
        notes.append("Integrals only: pick the peak too, to check the heights say the same.")
    elif out["area"] is None:
        notes.append("Heights only: integrate a region too, to check the line shape is "
                     "unchanged.")
    return out


# ============================================================
# 5. Profiles and enhancements
# ============================================================

def _ratio_arrays(v, e, s, v0, e0, s0):
    with np.errstate(divide="ignore", invalid="ignore"):
        r = v / v0
        rel = np.sqrt(np.where(v != 0, (e / v) ** 2, 0.0) + (e0 / v0) ** 2)
        rel_s = np.sqrt(np.where(v != 0, (s / v) ** 2, 0.0) + (s0 / v0) ** 2)
    return r, np.abs(r) * rel, np.abs(r) * rel_s


def _expressed(q: dict, mode: str, off: int | None):
    v, e, s = (np.asarray(q[k], dtype=float) for k in ("value", "error", "systematic"))
    if mode == "absolute":
        return v, e, s
    if off is None:
        raise ValueError("An enhancement needs the OFF (microwaves-off) spectrum.")
    r, er, sr = _ratio_arrays(v, e, s, v[off], e[off], s[off])
    if mode == "ratio":
        return r, er, sr
    return 1.0 - r, er, sr


def field_profile(measurement: dict, fields, mode: str = "absolute", off: int | None = None,
                  exclude=()) -> dict:
    """Step 5: every measured quantity against the field, sorted by
    field. ``fields``: B0 (T) per spectrum, or :func:`alignment_fields`'s
    dict. ``mode``: 'absolute' (per scan, RG-corrected), 'ratio' (each /
    the OFF spectrum ``off``) or 'one_minus' (1 - that ratio).
    ``exclude``: indices to leave out (e.g. the OFF spectrum itself).
    Returns ``{"field_T", "labels", "index", "mode", "off", "height":
    {"value", "error", "systematic"} | None, "area": ... | None,
    "field_source"}``."""
    if mode not in PROFILE_MODES:
        raise ValueError(f"mode must be one of {PROFILE_MODES}, not {mode!r}.")
    source = None
    if isinstance(fields, dict):
        source = fields.get("source")
        fields = fields["field_T"]
    fields = np.asarray(fields, dtype=float)
    keep = [i for i in range(len(fields)) if i not in set(exclude)]
    order = sorted(keep, key=lambda i: fields[i])
    out = {"field_T": fields[order], "labels": [measurement["labels"][i] for i in order],
           "index": order, "mode": mode, "off": off, "height": None, "area": None,
           "field_source": source,
           "off_label": measurement["labels"][off] if off is not None else None}
    for key in ("height", "area"):
        q = measurement.get(key)
        if q is None:
            continue
        v, e, s = _expressed(q, mode, off)
        out[key] = {"value": v[order], "error": e[order], "systematic": s[order]}
    return out


def enhancement_table(measurement: dict, off: int, on=None) -> list[dict]:
    """Enhancement of each ON spectrum (default: every one but ``off``)
    against the OFF spectrum ``off``, by height and by integral: the ratio
    ON/OFF and 1 - ON/OFF, each +- noise and +- systematic, and whether
    height and integral agree (within twice their combined uncertainty).
    One dict per ON spectrum."""
    n = len(measurement["labels"])
    on = [i for i in range(n) if i != off] if on is None else list(on)
    rows = []
    for i in on:
        row = {"index": i, "label": measurement["labels"][i],
               "off_label": measurement["labels"][off]}
        for key in ("height", "area"):
            q = measurement.get(key)
            if q is None:
                row[key] = None
                continue
            r, er, sr = _ratio_arrays(*(np.asarray(q[k], dtype=float)[[i]] for k in
                                        ("value", "error", "systematic")),
                                      q["value"][off], q["error"][off], q["systematic"][off])
            row[key] = {"ratio": float(r[0]), "one_minus": float(1 - r[0]),
                        "error": float(er[0]), "systematic": float(sr[0])}
        if row["height"] and row["area"]:
            h, a = row["height"], row["area"]
            diff = h["ratio"] - a["ratio"]
            sigma = math.sqrt(h["error"] ** 2 + h["systematic"] ** 2 + a["error"] ** 2
                              + a["systematic"] ** 2)
            row["agreement"] = {"difference": diff, "sigma": sigma,
                                "consistent": bool(abs(diff) <= 2 * sigma if sigma > 0
                                                   else diff == 0)}
        else:
            row["agreement"] = None
        rows.append(row)
    return rows


# ============================================================
# Reports
# ============================================================

def _pm(v, e, s=None, digits=4) -> str:
    if v is None or not np.isfinite(v):
        return "—"
    text = f"{v:.{digits}g} ± {e:.2g}"
    if s is not None and np.isfinite(s) and s > 0:
        text += f" (± {s:.2g} sys.)"
    return text


def alignment_report(alignment: dict) -> str:
    ref = alignment["entries"][alignment["reference"]]["label"]
    lines = [f"Aligned on the line at {alignment['peak_ppm']:.4g} ppm in {ref} (reference; "
             f"{alignment['ref_offset_hz']:.1f} Hz from its carrier, FWHM "
             f"{alignment['ref_fwhm_hz']:.1f} Hz). Template ± {alignment['match_halfwidth_hz']:.0f} "
             f"Hz, searched ± {alignment['window_hz'] / 1e3:g} kHz; accepted at |r| >= "
             f"{alignment['min_score']:g} and S/N >= {alignment['min_snr']:g}."]
    for e in alignment["entries"]:
        manual = f" + {e['manual_shift_hz']:.1f} by hand" if e["manual_shift_hz"] else ""
        lines.append(f"  {e['label']}: shift {e['auto_shift_hz']:.1f} Hz{manual}; |r| "
                     f"{e['score']:.3f}, S/N {e['snr']:.0f}"
                     + ("" if e["found"] else " — NOT FOUND")
                     + (f" — {e['note']}" if e["note"] and e["note"] != "reference" else ""))
    lines.append("Confirmed." if alignment.get("confirmed") else "Not yet confirmed.")
    return "\n".join(lines)


def measurement_report(measurement: dict) -> str:
    st = measurement["settings"]
    what = []
    if measurement["height"] is not None:
        what.append(f"height at {st['peak_ppm']:.4g} ppm")
    if measurement["area"] is not None:
        what.append(f"integral over {st['region'][0]:.4g} to {st['region'][1]:.4g} ppm")
    lines = [f"Measured {' and '.join(what)}, {st['baseline']} baseline."]
    for i, label in enumerate(measurement["labels"]):
        bits = []
        for key, name in (("height", "height"), ("area", "integral")):
            q = measurement[key]
            if q is not None:
                bits.append(f"{name} {_pm(q['value'][i], q['error'][i], q['systematic'][i])}")
        lines.append(f"  {label}: " + "; ".join(bits))
    lines += measurement["notes"]
    return "\n".join(lines)


_MODE_TEXT = {"absolute": "absolute (per scan)", "ratio": "ON / OFF",
              "one_minus": "1 − ON / OFF"}


def profile_report(profile: dict) -> str:
    mode = _MODE_TEXT[profile["mode"]]
    if profile["mode"] != "absolute":
        mode += f" (OFF = {profile['off_label']})"
    src = {"line": "at the aligned line", "sfo1": "from SFO1 (line assumed on the carrier)"}.get(
        profile.get("field_source"), "as given")
    lines = [f"Field profile: {mode}; B0 {src}."]
    for k, label in enumerate(profile["labels"]):
        bits = [f"B0 {profile['field_T'][k]:.6f} T"]
        for key, name in (("height", "height"), ("area", "integral")):
            q = profile[key]
            if q is not None:
                bits.append(f"{name} {_pm(q['value'][k], q['error'][k], q['systematic'][k])}")
        lines.append(f"  {label}: " + "; ".join(bits))
    return "\n".join(lines)


def enhancement_report(rows: list[dict]) -> str:
    lines = []
    for row in rows:
        lines.append(f"{row['label']} vs {row['off_label']} (OFF):")
        for key, name in (("height", "by height"), ("area", "by integral")):
            q = row[key]
            if q is None:
                continue
            lines.append(f"  {name}: ε = ON/OFF = {_pm(q['ratio'], q['error'], q['systematic'])}; "
                         f"1 − ON/OFF = {_pm(q['one_minus'], q['error'], q['systematic'])}")
        ag = row["agreement"]
        if ag is not None:
            lines.append("  height and integral agree" if ag["consistent"] else
                         f"  height and integral DISAGREE by {ag['difference']:.3g} (> 2σ = "
                         f"{2 * ag['sigma']:.2g}): the line shape changed (broadening, heating), "
                         "the region misses part of the line, or the baseline differs.")
        elif row["height"] is None or row["area"] is None:
            lines.append("  measure both height and integral to check they agree.")
    return "\n".join(lines)


def profile_to_csv(profile: dict, path) -> None:
    """The profile as a CSV: label, B0 (T), then value/error/systematic
    per measured quantity."""
    header = ["label", "field_T"]
    for key in ("height", "area"):
        if profile[key] is not None:
            header += [f"{key}", f"{key}_error", f"{key}_systematic"]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for k, label in enumerate(profile["labels"]):
            row = [label, f"{profile['field_T'][k]:.9g}"]
            for key in ("height", "area"):
                q = profile[key]
                if q is not None:
                    row += [f"{q[c][k]:.9g}" for c in ("value", "error", "systematic")]
            writer.writerow(row)


# ============================================================
# Plots
# ============================================================

def plot_aligned_spectra(aligned, labels=None, mode: str = "stack", normalise: bool = True,
                         peak_ppm: float | None = None, region=None, xlims=None,
                         figure=None, cmap: str = "viridis", fields=None,
                         fontsize: float = 12.0, highlight: int | None = None,
                         aligned_axis: bool = True):
    """The aligned spectra on one ppm axis: ``mode`` 'stack' (offset, one
    row each) or 'overlay'. ``normalise`` scales each to unit height --
    what makes their shapes, and so the alignment, comparable. Colours
    run along ``fields`` when given. The pick and the region are marked.
    ``aligned_axis=False`` labels the axis as a spectrum's own (e.g. the
    reference alone, before aligning). Returns ``(figure, lines)`` -- ``lines[i]`` is spectrum i's trace."""
    import matplotlib.pyplot as plt
    from ..plotting.style import stack_colors

    aligned = list(aligned)
    n = len(aligned)
    labels = list(labels) if labels is not None else [f"Spectrum {k + 1}" for k in range(n)]
    if figure is None:
        figure = plt.figure(figsize=(3.375 * 1.6, 3.375 if mode == "overlay" else
                                     min(1.2 + 0.45 * n, 9.0)))
    else:
        figure.clear()
    ax = figure.add_subplot(111)
    palette = stack_colors(n, cmap)
    order = (np.argsort(np.asarray(fields, dtype=float), kind="stable") if fields is not None
             else np.arange(n))
    rank = np.empty(n, dtype=int)
    rank[order] = np.arange(n)
    lines = []
    if xlims is not None:
        lo_x, hi_x = min(xlims), max(xlims)
    traces = []
    for s in aligned:
        x = np.asarray(s.axes[-1].ppm, dtype=float)
        y = np.asarray(s.data, dtype=float)
        window = (x >= lo_x) & (x <= hi_x) if xlims is not None else np.ones(x.size, bool)
        if not np.any(window):
            window = np.ones(x.size, bool)
        if normalise:
            top = float(np.max(np.abs(y[window])))
            y = y / top if top > 0 else y
        traces.append((x, y, float(np.min(y[window])), float(np.max(y[window]))))
    # Stacked rows spaced by what each trace spans -- an inverted line
    # (negative enhancement) dips below its baseline, and must not run
    # into the row under it.
    offsets = np.zeros(n)
    if mode == "stack":
        span = max((hi - lo) for _x, _y, lo, hi in traces) or 1.0
        gap = 0.12 * span
        level = 0.0
        for k, i in enumerate(order):
            _x, _y, lo, hi = traces[i]
            if k:
                prev = traces[order[k - 1]]
                level += (prev[3] - 0.0) + (0.0 - lo) + gap
            offsets[i] = level
    for i, s in enumerate(aligned):
        x, y, _lo, _hi = traces[i]
        offset = float(offsets[i])
        line, = ax.plot(x, y + offset, color=palette[rank[i]],
                        linewidth=1.6 if highlight == i else 0.9, label=labels[i])
        line.dnp_index = i
        line.dnp_offset = offset
        lines.append(line)
        if mode == "stack":
            ax.annotate(labels[i], xy=(0.0, offset), xycoords=("axes fraction", "data"),
                        xytext=(2, 3), textcoords="offset points", fontsize=fontsize * 0.6,
                        color=palette[rank[i]], clip_on=True)
    if region is not None:
        ax.axvspan(min(region), max(region), color="0.88", zorder=0)
    if peak_ppm is not None:
        ax.axvline(peak_ppm, color="0.4", linewidth=0.8, linestyle="--", zorder=0)
    nucleus = dnp._nucleus_label(getattr(aligned[0].metadata, "nucleus", "")) if aligned else ""
    ax.set_xlabel(_italic_label((f"{nucleus} shift (ppm)" + (", aligned" if aligned_axis else "")).strip()),
                  fontsize=fontsize)
    if xlims is not None:
        ax.set_xlim(hi_x, lo_x)
        dnp._fit_y_to_visible(ax)
    else:
        ax.invert_xaxis()
    if mode == "overlay" and n <= 12:
        ax.legend(frameon=False, fontsize=fontsize * 0.7)
    dnp._spectrum_axes(ax, fontsize)
    figure.tight_layout()
    return figure, lines


def _field_axis(profile, relative):
    field = np.asarray(profile["field_T"], dtype=float)
    if relative:
        return (field - field[0]) * 1e3, "ΔB$_0$ (mT)"
    return field, "B$_0$ (T)"


def plot_field_profile(profile: dict, quantity: str = "both", relative: bool = False,
                       figure=None, colors=("black", "#d55e00"), fontsize: float = 12.0,
                       nucleus: str = "", show_top_spine: bool = False,
                       show_right_spine: bool = False):
    """Height and/or integral against B0 with 1σ (noise) error bars.
    ``quantity`` 'height', 'area' or 'both' (two panels sharing the field
    axis). ``relative`` plots B0 minus the lowest field, in mT. Returns
    the figure."""
    import matplotlib.pyplot as plt
    from ..plotting.style import apply_spine_style, apply_tick_style

    keys = [k for k in (("height", "area") if quantity == "both" else (quantity,))
            if profile.get(k) is not None]
    if not keys:
        raise ValueError(f"The profile holds no {quantity}: measure it first.")
    if figure is None:
        figure = plt.figure(figsize=(3.375 * 1.4, 2.8 * len(keys)))
    else:
        figure.clear()
    axes = figure.subplots(len(keys), 1, sharex=True, squeeze=False)[:, 0]
    x, xlabel = _field_axis(profile, relative)
    mode = profile["mode"]
    nuc = dnp._nucleus_label(nucleus)
    for ax, key, colour in zip(axes, keys, colors):
        q = profile[key]
        v, e = np.asarray(q["value"], float), np.asarray(q["error"], float)
        exponent = 0
        if mode == "absolute":
            biggest = np.nanmax(np.abs(v)) if v.size else 0.0
            if biggest > 0:
                exponent = int(math.floor(math.log10(biggest)))
                exponent = exponent if abs(exponent) >= 4 else 0
        if exponent:
            v, e = v / 10.0 ** exponent, e / 10.0 ** exponent
        ax.plot(x, v, "-", color=colour, linewidth=1.0, zorder=1)
        ax.errorbar(x, v, yerr=e, fmt="o", color=colour, markersize=5, capsize=2.5,
                    linewidth=1.0, zorder=2)
        if np.nanmin(v) < 0 < np.nanmax(v):
            ax.axhline(0, color="0.5", linewidth=0.8, zorder=0)
        name = "Intensity" if key == "height" else "Integral"
        if mode == "absolute":
            label = f"{nuc} {name.lower()} (" + (f"10$^{{{exponent}}}$ " if exponent else "") \
                + "a.u.)"
        elif mode == "ratio":
            label = f"ε ({name.lower()}, ON/OFF)"
        else:
            label = f"1 − ON/OFF ({name.lower()})"
        ax.set_ylabel(_italic_label(label.strip()), fontsize=fontsize)
        apply_spine_style(ax, top=show_top_spine, right=show_right_spine, bottom=True, left=True)
        apply_tick_style(ax, labelsize=fontsize * 0.9)
        ax.tick_params(which="both", top=show_top_spine, right=show_right_spine)
    from matplotlib.ticker import MaxNLocator
    axes[-1].set_xlabel(_italic_label(xlabel), fontsize=fontsize)
    axes[-1].ticklabel_format(axis="x", useOffset=False)
    # B0 in T carries six or seven digits: few ticks, or they collide.
    axes[-1].xaxis.set_major_locator(MaxNLocator(nbins=4 if not relative else 6))
    figure.tight_layout()
    return figure


def _italic_label(text):
    """Quantity symbols italic (plotting.labels.italicise_symbols)."""
    from opennmr.plotting.labels import italicise_symbols
    return italicise_symbols(text)
