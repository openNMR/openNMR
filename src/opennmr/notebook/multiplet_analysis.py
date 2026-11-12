"""Multiplets and J couplings from picked lines (explicit user request,
2026-11-12).

Pick the lines of a spectrum (Fit ▸ Pick Peak, or Auto-Pick Peaks) and
this sorts them into multiplets, names each (s, d, t, dd, td, …), gives
its centre and coupling constants, and draws the coupling tree. The
report is written the way a paper quotes it:
``¹H NMR (400 MHz) δ 7.26 (d, J = 8.1 Hz), 3.45 (dd, J = 10.1, 4.0 Hz)``.

**How a multiplet is read** (first order; Hoye, Hanson & Vyvyan, *J. Org.
Chem.* 1994, 59, 4096). Each line is given a whole number of intensity
units from its height -- the heights are scaled so the units add up to a
power of two, as a first-order multiplet's must (a triplet is 1:2:1, four
units). With the units as a list of positions measured from the first
line, the smallest spacing is the smallest J; the positions every J found
so far can explain (all sums of a subset of them) are struck off, and the
first position left is the next J; and so on until every position is
explained. The J values are then refined by least squares against every
line (not one spacing), and the root-mean-square misfit is reported, so
a multiplet that is not first order -- the misfit, or lines left over,
says so -- is called ``m`` with its range rather than given couplings
it does not have.

**What it cannot do.** Strongly coupled spins (Δν not ≫ J: roofing, extra
lines) are not first order; use the strongly coupled spin-system fit in
Simulate/Fit for those. Overlapping multiplets picked as one group give
``m``: narrow the grouping gap or pick fewer lines. The heights come from
the spectrum at each picked position, so a poorly phased or baseline-
distorted spectrum gives wrong units -- phase and baseline first.

**Finding the lines** (user report, 2026-11-12: Auto-Pick had taken only
the tallest line of each multiplet, so every one read as a singlet). Given
the spectrum, :func:`analyse_spectrum` finds every line itself: inside
each integration region, around each picked line, or across the whole
spectrum. A line counts when it stands :data:`DEFAULT_MIN_SNR` times the
noise above the local baseline and is at least
:data:`DEFAULT_MIN_FRACTION` of its multiplet's tallest line -- which
keeps a septet's outer lines (1/20 of the centre) and leaves out ¹³C
satellites (about 1/200). Heights are measured above a straight local
baseline drawn under each multiplet, so a raised or sloping baseline does
not distort the intensity units. (A straight baseline cannot follow the
curved tail of a far larger peak right beside a multiplet: draw a tight
region round the multiplet, or correct the baseline first.)

Works on any nucleus: give the spectrometer frequency (MHz) for the axis.
"""

from __future__ import annotations

import itertools

import numpy as np

from ..processing import multiplets as _mult

#: Lines further apart than this (Hz) start a new multiplet.
DEFAULT_MAX_GAP_HZ = 20.0
#: How far (Hz) a line may sit from where the couplings put it.
DEFAULT_TOLERANCE_HZ = 0.5
#: Most couplings sought for one multiplet (2**8 = 256 line units).
MAX_COUPLINGS = 8
#: A line must stand this many times the noise above the baseline ...
DEFAULT_MIN_SNR = 5.0
#: ... and be at least this fraction of its multiplet's tallest line.
DEFAULT_MIN_FRACTION = 0.02
#: How a multiplet is named in a report: the letters the peak table uses
#: for 4-6 neighbours (p, x, h) written out as usual.
_REPORT_NAMES = {"p": "quint", "x": "sext", "h": "sept"}


def report_name(pattern: str) -> str:
    """``"h"`` -> ``"sept"``, ``"dp"`` -> ``"dquint"``; others unchanged."""
    return "".join(_REPORT_NAMES.get(ch, ch) for ch in str(pattern))

_SUPERSCRIPT = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")


# ---- grouping ------------------------------------------------------------------------------

def group_lines(shifts_ppm, mhz: float, max_gap_hz: float = DEFAULT_MAX_GAP_HZ) -> list[list[int]]:
    """Indices of ``shifts_ppm`` grouped into multiplets: lines no more than
    ``max_gap_hz`` from their neighbour belong together. Groups run from
    high to low shift, each in its own high-to-low order."""
    shifts = np.asarray(shifts_ppm, dtype=float)
    if shifts.size == 0:
        return []
    if not mhz or mhz <= 0:
        raise ValueError("The spectrometer frequency (MHz) is needed to turn ppm into Hz.")
    order = list(np.argsort(-shifts))
    groups, current = [], [order[0]]
    for previous, index in zip(order, order[1:]):
        if (shifts[previous] - shifts[index]) * mhz > max_gap_hz:
            groups.append(current)
            current = []
        current.append(index)
    groups.append(current)
    return [[int(i) for i in g] for g in groups]


# ---- one multiplet -------------------------------------------------------------------------

def intensity_units(heights) -> tuple[np.ndarray, float]:
    """Whole intensity units for each line: the heights scaled so they add
    up to a power of two, rounding as little as possible. Returns ``(units,
    misfit)``, the misfit the largest rounding as a fraction of one unit.
    Heights of zero or below count as one unit."""
    h = np.abs(np.asarray(heights, dtype=float))
    if h.size == 0:
        return np.zeros(0, dtype=int), 0.0
    if not np.all(np.isfinite(h)) or np.all(h == 0):
        return np.ones(h.size, dtype=int), 0.0
    h = np.where(h > 0, h, h[h > 0].min())
    best = None
    for k in range(0, MAX_COUPLINGS + 1):
        total = 2 ** k
        if total < h.size:
            continue
        scaled = h * total / h.sum()
        units = np.maximum(1, np.rint(scaled)).astype(int)
        if units.sum() != total:
            continue
        misfit = float(np.max(np.abs(scaled - units)))
        # The simplest pattern wins unless a richer one fits much better.
        if best is None or misfit < best[1] - 0.15:
            best = (units, misfit)
    if best is None:
        units = np.maximum(1, np.rint(h / h.min())).astype(int)
        return units, float(np.max(np.abs(h / h.min() - units)))
    return best


def _subset_sums(js) -> list[tuple[float, tuple[int, ...]]]:
    """Every sum of a subset of ``js`` with the subset (as 0/1 flags)."""
    out = []
    for flags in itertools.product((0, 1), repeat=len(js)):
        out.append((float(np.dot(flags, js)) if js else 0.0, flags))
    return sorted(out)


def _match(offsets, sums, tolerance):
    """Assign each subset sum to an unused offset within ``tolerance``.
    Returns ``(assignment, unmatched_offsets)`` or ``None`` when a sum has
    no offset to go to."""
    used = [False] * len(offsets)
    assignment = []
    for value, flags in sums:
        best, best_d = None, None
        for i, off in enumerate(offsets):
            if used[i]:
                continue
            d = abs(off - value)
            if d <= tolerance and (best_d is None or d < best_d):
                best, best_d = i, d
        if best is None:
            return None
        used[best] = True
        assignment.append((best, flags))
    return assignment, [off for off, u in zip(offsets, used) if not u]


def decompose(offsets_hz, tolerance_hz: float = DEFAULT_TOLERANCE_HZ):
    """The couplings (Hz, smallest first) that explain ``offsets_hz`` --
    every line unit's position measured from the first line -- by the
    first-order method in the module notes; ``None`` if none do."""
    offsets = sorted(float(o) for o in offsets_hz)
    if not offsets:
        return None
    js: list[float] = []
    while len(js) <= MAX_COUPLINGS:
        matched = _match(offsets, _subset_sums(js), tolerance_hz)
        if matched is None:
            return None
        _assignment, left = matched
        if not left:
            return js
        nxt = min(left)
        if nxt <= tolerance_hz:
            return None
        js.append(nxt)
    return None


def _refine(line_hz, units, js, tolerance_hz):
    """Least-squares first line and couplings against every line unit;
    returns ``(first_hz, js, rms_hz)``."""
    positions = np.repeat(np.asarray(line_hz, dtype=float), units)
    origin = positions.min()
    matched = _match(list(positions - origin), _subset_sums(js), tolerance_hz)
    if matched is None:
        return origin, list(js), float("nan")
    assignment, _left = matched
    a = np.array([[1.0, *flags] for _i, flags in assignment])
    b = np.array([positions[i] for i, _flags in assignment])
    solution, *_ = np.linalg.lstsq(a, b, rcond=None)
    residual = b - a @ solution
    return float(solution[0]), [float(j) for j in solution[1:]], float(np.sqrt(np.mean(residual ** 2)))


def name_couplings(js_hz, tolerance_hz: float = DEFAULT_TOLERANCE_HZ):
    """``(pattern, J values)``: couplings equal within the tolerance are one
    J to several equivalent neighbours. Largest J first, as a multiplet is
    named and quoted: 7.0, 7.0, 2.0 Hz -> ``("td", [7.0, 2.0])``."""
    groups: list[list[float]] = []
    for j in sorted(js_hz, reverse=True):
        if groups and abs(groups[-1][0] - j) <= tolerance_hz:
            groups[-1].append(j)
        else:
            groups.append([j])
    counts = [len(g) for g in groups]
    return _mult.multiplet_name(counts), [float(np.mean(g)) for g in groups], counts


#: Heights further than this (RMS, as a fraction of the tallest line) from
#: the intensities a pattern predicts rule the pattern out.
MAX_INTENSITY_MISFIT = 0.3
#: Most candidate patterns tried for one multiplet.
_MAX_CANDIDATES = 20000


def _candidate_js(line_hz, tolerance_hz):
    """Every spacing between two lines (merged within the tolerance) no
    smaller than the first spacing -- the couplings a pattern can have."""
    diffs = sorted(abs(b - a) for i, a in enumerate(line_hz) for b in line_hz[i + 1:])
    first = line_hz[1] - line_hz[0]
    out: list[float] = []
    for d in diffs:
        if d < first - tolerance_hz:
            continue
        if out and d - out[-1] <= tolerance_hz:
            continue
        out.append(d)
    return out


def _stick_counts(offsets, js, tolerance_hz):
    """How many of the pattern ``js``'s 2**n lines fall on each observed
    line (offsets from the first); None if a line of the pattern falls on
    none, or an observed line gets none."""
    counts = np.zeros(len(offsets), dtype=int)
    for value, _flags in _subset_sums(js):
        k = int(np.argmin(np.abs(offsets - value)))
        if abs(offsets[k] - value) > tolerance_hz:
            return None
        counts[k] += 1
    return None if np.any(counts == 0) else counts


def best_pattern(line_hz, heights=None, tolerance_hz: float = DEFAULT_TOLERANCE_HZ):
    """The first-order pattern that best explains lines at ``line_hz``
    (sorted) with ``heights``: ``(js, counts, misfit)`` or None.

    Every candidate set of couplings must contain the first spacing (the
    smallest J of a first-order multiplet) and add up to the multiplet's
    width, and its lines must fall on the observed ones and leave none
    unexplained. Of those, the one whose predicted intensities (binomial
    counts) best match the heights wins -- the misfit is the RMS
    difference, as a fraction of the tallest line, at the best scale -- and
    of near-equal fits the one with fewer couplings. Without heights,
    the fewest couplings that explain the positions win."""
    pos = np.asarray(line_hz, dtype=float)
    if pos.size < 2:
        return None
    offsets = pos - pos[0]
    span = offsets[-1]
    candidates = _candidate_js(list(pos), tolerance_hz)
    first = offsets[1]
    h = None if heights is None else np.clip(np.asarray(heights, dtype=float), 0.0, None)
    min_n = int(np.ceil(np.log2(pos.size)))
    found = []
    tried = [0]

    def score(counts):
        if h is None or not np.any(h > 0):
            return 0.0
        c = counts.astype(float)
        scale = float(c @ h) / float(c @ c)
        return float(np.sqrt(np.mean((h - scale * c) ** 2)) / h.max())

    def search(js, total, start):
        if tried[0] > _MAX_CANDIDATES:
            return
        n = len(js)
        slack = tolerance_hz * max(n, 1)
        if abs(total - span) <= slack and n >= min_n:
            tried[0] += 1
            counts = _stick_counts(offsets, js, tolerance_hz)
            if counts is not None:
                found.append((score(counts), n, list(js), counts))
        if n >= MAX_COUPLINGS or total > span + slack:
            return
        for k in range(start, len(candidates)):
            j = candidates[k]
            if total + j > span + tolerance_hz * (n + 1):
                break
            search(js + [j], total + j, k)

    search([first], first, 0)
    if not found:
        return None
    best = min(s for s, *_ in found)
    if best > MAX_INTENSITY_MISFIT:
        return None
    # Of fits as good as the best (within 0.02), the simplest.
    misfit, _n, js, counts = min((f for f in found if f[0] <= best + 0.02),
                                 key=lambda f: (f[1], f[0]))
    return js, counts, misfit


def analyse_multiplet(shifts_ppm, mhz: float, heights=None,
                      tolerance_hz: float = DEFAULT_TOLERANCE_HZ) -> dict:
    """One multiplet from its lines' shifts (ppm) and heights.

    Returns a dict: ``pattern`` ("dd", "t", "s", or "m" when it is not
    first order), ``centre_ppm``, ``J_hz`` (one per distinct coupling,
    largest first), ``counts`` (equivalent neighbours for each J),
    ``lines_ppm``, ``units`` (intensity units per line), ``rms_hz`` (misfit
    of the line positions), ``range_ppm`` and ``notes``."""
    shifts = np.asarray(shifts_ppm, dtype=float)
    if shifts.size == 0:
        raise ValueError("No lines to analyse.")
    if not mhz or mhz <= 0:
        raise ValueError("The spectrometer frequency (MHz) is needed to turn ppm into Hz.")
    order = np.argsort(shifts)
    shifts = shifts[order]
    heights = None if heights is None else np.asarray(heights, dtype=float)[order]
    line_hz = shifts * mhz
    result = {"lines_ppm": [float(s) for s in shifts[::-1]], "range_ppm": (float(shifts.max()),
                                                                          float(shifts.min())),
              "mhz": float(mhz), "notes": []}
    if shifts.size == 1:
        result.update(pattern="s", centre_ppm=float(shifts[0]), J_hz=[], counts=[], units=[1],
                      rms_hz=0.0, first_order=True)
        return result
    found = best_pattern(line_hz, heights, tolerance_hz)
    if found is None:
        result.update(pattern="m", centre_ppm=float(np.average(
            shifts, weights=None if heights is None else np.clip(heights, 1e-12, None))),
            J_hz=[], counts=[], units=[1] * shifts.size, rms_hz=float("nan"),
            first_order=False, intensity_misfit=float("nan"))
        result["notes"].append(
            "Not a first-order pattern within the tolerance: reported as a multiplet (m) with "
            "its range. Try a larger tolerance, check the lines found and their heights (phase "
            "and baseline), or fit it as a strongly coupled spin system.")
        return result
    js, units, misfit = found
    result["units"] = [int(u) for u in units[::-1]]
    result["intensity_misfit"] = misfit
    if heights is None:
        result["notes"].append("No heights given: the pattern was read from the line positions "
                               "alone, so coinciding lines (a triplet's centre) are assumed.")
    elif misfit > MAX_INTENSITY_MISFIT / 2:
        result["notes"].append(f"The heights fit the pattern only roughly (off by "
                               f"{100 * misfit:.0f} % of the tallest line): check phase and "
                               "baseline; strong coupling also skews them.")
    first, js, rms = _refine(line_hz, np.asarray(units), js, max(tolerance_hz, 1e-9) * 4)
    pattern, j_values, counts = name_couplings(js, tolerance_hz)
    centre_hz = first + sum(js) / 2.0
    result.update(pattern=pattern, centre_ppm=float(centre_hz / mhz), J_hz=j_values, counts=counts,
                  rms_hz=rms, first_order=True, couplings_hz=sorted(js, reverse=True))
    if np.isfinite(rms) and rms > tolerance_hz:
        result["notes"].append(f"The lines sit {rms:.2f} Hz (RMS) from the first-order pattern: "
                               "more than the tolerance, so treat the J values with care.")
    return result


# ---- a spectrum's picked lines -------------------------------------------------------------

def spectrum_mhz(spectrum) -> float | None:
    """Hz per ppm of a 1D spectrum's axis (its own ppm and Hz), else SFO1."""
    axis = spectrum.axes[-1]
    if axis.ppm is not None and getattr(axis, "hz", None) is not None and len(axis.ppm) > 1:
        span = float(axis.ppm[-1] - axis.ppm[0])
        if span:
            return abs(float(axis.hz[-1] - axis.hz[0]) / span)
    return float(spectrum.metadata.sfo1) if spectrum.metadata.sfo1 else None


def snap_to_maxima(spectrum, shifts_ppm, window_hz: float = 2.0) -> np.ndarray:
    """Each shift moved to the top of the line it was picked on: the
    highest point within ``window_hz`` (and no further than halfway to the
    next picked line), refined between points by a parabola through the
    top three. A click is rarely on the very top of a line, and J values
    are differences of positions, so this matters. A shift with no
    maximum inside its window (a shoulder) is left as picked."""
    shifts = np.asarray(shifts_ppm, dtype=float)
    mhz = spectrum_mhz(spectrum)
    if not mhz or shifts.size == 0:
        return shifts.copy()
    ppm = np.asarray(spectrum.axes[-1].ppm, dtype=float)
    data = np.real(np.asarray(spectrum.data))
    if data.ndim > 1:
        raise ValueError("Multiplet analysis needs a 1D spectrum (or one slice of a series).")
    order = np.argsort(ppm)
    ppm, data = ppm[order], data[order]
    sorted_shifts = np.sort(shifts)
    out = shifts.copy()
    for k, x in enumerate(shifts):
        half = window_hz / mhz
        position = int(np.searchsorted(sorted_shifts, x))
        neighbours = [abs(x - sorted_shifts[i]) for i in (position - 1, position + 1)
                      if 0 <= i < sorted_shifts.size and sorted_shifts[i] != x]
        if neighbours:
            half = min(half, 0.5 * min(neighbours))
        lo, hi = np.searchsorted(ppm, x - half), np.searchsorted(ppm, x + half)
        if hi - lo < 3:
            continue
        i = lo + int(np.argmax(data[lo:hi]))
        if i in (lo, hi - 1) or i <= 0 or i >= ppm.size - 1:
            continue                       # the edge of the window: not a maximum
        y0, y1, y2 = data[i - 1], data[i], data[i + 1]
        denom = y0 - 2 * y1 + y2
        step = 0.5 * (y0 - y2) / denom if denom else 0.0
        out[k] = ppm[i] + float(np.clip(step, -0.5, 0.5)) * (ppm[i + 1] - ppm[i])
    return out


def heights_at(spectrum, shifts_ppm) -> np.ndarray:
    """The real spectrum's height at each shift (interpolated)."""
    ppm = np.asarray(spectrum.axes[-1].ppm, dtype=float)
    data = np.real(np.asarray(spectrum.data))
    if data.ndim > 1:
        raise ValueError("Multiplet analysis needs a 1D spectrum (or one slice of a series).")
    order = np.argsort(ppm)
    return np.interp(np.asarray(shifts_ppm, dtype=float), ppm[order], data[order])


def _sorted_axis(spectrum):
    ppm = np.asarray(spectrum.axes[-1].ppm, dtype=float)
    data = np.real(np.asarray(spectrum.data))
    if data.ndim > 1:
        raise ValueError("Multiplet analysis needs a 1D spectrum (or one slice of a series).")
    order = np.argsort(ppm)
    return ppm[order], data[order]


def noise_level(spectrum) -> float:
    """The noise's standard deviation, from the point-to-point differences
    (robust: the median absolute difference), so lines and a sloping
    baseline hardly affect it."""
    _ppm, y = _sorted_axis(spectrum)
    diff = np.diff(y)
    return float(np.median(np.abs(diff - np.median(diff))) / 0.6745 / np.sqrt(2.0))


def _baseline_level(ppm, y, lo, hi, margin):
    """``(slope, intercept)`` of a straight baseline under ``lo``..``hi``
    (ppm): through the lowest fifth of the points in a ``margin``-wide
    strip on each side (one side alone when the other runs off the axis)."""
    sides = []
    for a, b in ((lo - margin, lo), (hi, hi + margin)):
        m = (ppm >= a) & (ppm <= b)
        if np.count_nonzero(m) >= 3:
            yy = np.sort(y[m])
            sides.append((0.5 * (a + b), float(np.mean(yy[:max(1, yy.size // 5)]))))
    if not sides:
        return 0.0, 0.0
    if len(sides) == 1:
        return 0.0, sides[0][1]
    (x0, y0), (x1, y1) = sides
    slope = (y1 - y0) / (x1 - x0) if x1 != x0 else 0.0
    return slope, y0 - slope * x0


def heights_above_baseline(spectrum, shifts_ppm, max_gap_hz: float = DEFAULT_MAX_GAP_HZ):
    """The spectrum's height at each shift, above a straight baseline drawn
    under the group of lines (:func:`_baseline_level`, the strips one
    ``max_gap_hz`` wide beyond the outer lines)."""
    shifts = np.asarray(shifts_ppm, dtype=float)
    ppm, y = _sorted_axis(spectrum)
    mhz = spectrum_mhz(spectrum) or 1.0
    margin = max(max_gap_hz, 5.0) / mhz
    lo, hi = float(shifts.min()) - 0.5 * margin, float(shifts.max()) + 0.5 * margin
    slope, intercept = _baseline_level(ppm, y, lo, hi, margin)
    return np.interp(shifts, ppm, y) - (slope * shifts + intercept)


def find_lines(spectrum, low_ppm: float | None = None, high_ppm: float | None = None,
               min_snr: float = DEFAULT_MIN_SNR) -> np.ndarray:
    """Every line between ``low_ppm`` and ``high_ppm`` (the whole spectrum
    by default) standing ``min_snr`` times the noise above the local
    baseline -- each at the top of its peak (refined by a parabola) --
    high shift first."""
    from scipy.signal import find_peaks
    ppm, y = _sorted_axis(spectrum)
    lo = ppm[0] if low_ppm is None else min(low_ppm, high_ppm)
    hi = ppm[-1] if high_ppm is None else max(low_ppm, high_ppm)
    i0, i1 = np.searchsorted(ppm, lo), np.searchsorted(ppm, hi)
    if i1 - i0 < 3:
        return np.zeros(0)
    seg = y[i0:i1]
    noise = noise_level(spectrum) or float(np.std(seg)) or 1.0
    peaks, _props = find_peaks(seg, prominence=min_snr * noise)
    out = []
    for k in peaks:
        i = i0 + k
        if 0 < i < ppm.size - 1:
            y0, y1, y2 = y[i - 1], y[i], y[i + 1]
            denom = y0 - 2 * y1 + y2
            step = 0.5 * (y0 - y2) / denom if denom else 0.0
            out.append(ppm[i] + float(np.clip(step, -0.5, 0.5)) * (ppm[i + 1] - ppm[i]))
    return np.sort(np.asarray(out, dtype=float))[::-1]


#: A multiplet's tallest line must stand this many times the noise above
#: the baseline for a search to report it (a region, or the whole
#: spectrum, holds baseline wiggles that clear :data:`DEFAULT_MIN_SNR`).
DEFAULT_MIN_MULTIPLET_SNR = 20.0


def _grow(lines, heights, seed, mhz, max_gap_hz, min_fraction):
    """Indices of the multiplet around line ``seed``: neighbours no more
    than ``max_gap_hz`` apart, each at least ``min_fraction`` of the
    tallest line taken so far (lines sorted by shift)."""
    group = [seed]
    tallest = heights[seed]
    for step in (-1, 1):
        k = seed
        while 0 <= k + step < len(lines):
            nxt = k + step
            if abs(lines[nxt] - lines[k]) * mhz > max_gap_hz or heights[nxt] < min_fraction * tallest:
                break
            group.append(nxt)
            tallest = max(tallest, heights[nxt])
            k = nxt
    return sorted(group)


def analyse_spectrum(spectrum, regions=None, around=None, picked=None,
                     max_gap_hz: float = DEFAULT_MAX_GAP_HZ,
                     tolerance_hz: float = DEFAULT_TOLERANCE_HZ,
                     min_snr: float = DEFAULT_MIN_SNR,
                     min_fraction: float = DEFAULT_MIN_FRACTION,
                     min_multiplet_snr: float = DEFAULT_MIN_MULTIPLET_SNR) -> list[dict]:
    """Find the multiplets' lines in ``spectrum`` and analyse them.

    Where to look, in this order of preference:

    * ``picked`` -- shifts (ppm) to use as they are (each moved to the top
      of its peak), when every line has been picked by hand;
    * ``regions`` -- ``[(low, high), …]`` in ppm (the integration regions):
      the multiplets inside each;
    * ``around`` -- shifts (ppm), e.g. one picked line per multiplet: the
      multiplet each belongs to, grown outwards from it;
    * nothing given: every multiplet in the spectrum, grown from its
      strongest lines.

    A line must stand ``min_snr`` times the noise above the baseline and
    be at least ``min_fraction`` of its multiplet's tallest line; a
    multiplet whose tallest line is under ``min_multiplet_snr`` times the
    noise is not reported. Each result says how its lines were found
    (``found_by``)."""
    mhz = spectrum_mhz(spectrum)
    if not mhz:
        raise ValueError("The spectrometer frequency is unknown: set the field (or the nucleus "
                         "and ¹H frequency) for this spectrum.")
    if picked is not None and len(picked):
        results = analyse_multiplets(picked, spectrum=spectrum, max_gap_hz=max_gap_hz,
                                     tolerance_hz=tolerance_hz)
        for r in results:
            r["found_by"] = "picked lines"
        return results
    noise = noise_level(spectrum) or 1.0
    groups = []
    if regions:
        how = "integration region"
        for lo, hi in regions:
            lines = find_lines(spectrum, min(lo, hi), max(lo, hi), min_snr)
            for group in group_lines(lines, mhz, max_gap_hz):
                groups.append(lines[group])
    else:
        lines = np.sort(find_lines(spectrum, min_snr=min_snr))
        heights = np.array([heights_above_baseline(spectrum, [x], max_gap_hz)[0] for x in lines])
        if around is not None and len(around):
            how = "picked line"
            seeds = []
            for x in np.asarray(around, dtype=float):
                if lines.size:
                    k = int(np.argmin(np.abs(lines - x)))
                    if abs(lines[k] - x) * mhz <= max(max_gap_hz, 2.0):
                        seeds.append(k)
        else:
            how = "whole spectrum"
            seeds = [int(k) for k in np.argsort(-heights) if heights[k] >= min_multiplet_snr * noise]
        taken = set()
        for seed in seeds:
            if seed in taken:
                continue
            group = _grow(lines, heights, seed, mhz, max_gap_hz, min_fraction)
            taken.update(group)
            groups.append(lines[group])
    results = []
    for lines in groups:
        heights = heights_above_baseline(spectrum, lines, max_gap_hz)
        if heights.max() < min_multiplet_snr * noise:
            continue
        keep = heights >= min_fraction * heights.max()
        result = analyse_multiplet(lines[keep], mhz, heights[keep], tolerance_hz)
        result["found_by"] = how
        results.append(result)
    return sorted(results, key=lambda r: -r["centre_ppm"])


def analyse_multiplets(shifts_ppm, mhz: float | None = None, heights=None, spectrum=None,
                       max_gap_hz: float = DEFAULT_MAX_GAP_HZ,
                       tolerance_hz: float = DEFAULT_TOLERANCE_HZ,
                       snap: bool = True) -> list[dict]:
    """Group picked lines into multiplets and analyse each (see
    :func:`analyse_multiplet`). Give ``spectrum`` to take the frequency and
    the heights from it -- and, with ``snap``, to move each line to the top
    of its peak first (:func:`snap_to_maxima`) -- or ``mhz`` (and
    ``heights``) directly. High shift first."""
    shifts = np.asarray(shifts_ppm, dtype=float)
    picked = shifts.copy()
    from_spectrum = False
    if spectrum is not None:
        mhz = mhz or spectrum_mhz(spectrum)
        if snap:
            shifts = snap_to_maxima(spectrum, shifts)
        if heights is None:
            from_spectrum = True
    if not mhz:
        raise ValueError("The spectrometer frequency is unknown: set the field (or the nucleus "
                         "and ¹H frequency) for this spectrum, or pass mhz=.")
    heights = None if heights is None else np.asarray(heights, dtype=float)
    out = []
    for group in group_lines(shifts, mhz, max_gap_hz):
        if from_spectrum:
            group_heights = heights_above_baseline(spectrum, shifts[group], max_gap_hz)
        else:
            group_heights = None if heights is None else heights[group]
        result = analyse_multiplet(shifts[group], mhz, group_heights, tolerance_hz)
        # Which of the lines given went into it, and where they were given
        # (before any snapping): what a caller matches its own list against.
        result["indices"] = list(group)
        result["picked_ppm"] = [float(picked[i]) for i in group]
        out.append(result)
    return out


# ---- reporting -----------------------------------------------------------------------------

def nucleus_label(nucleus: str | None) -> str:
    """``"1H"`` -> ``"¹H"``."""
    text = str(nucleus or "").strip()
    digits = "".join(ch for ch in text if ch.isdigit())
    letters = "".join(ch for ch in text if ch.isalpha())
    return (digits.translate(_SUPERSCRIPT) + letters) if digits and letters else text


def format_multiplet(result: dict, decimals: int = 2, j_decimals: int = 1) -> str:
    """``"3.45 (dd, J = 10.1, 4.0 Hz)"``; ``"3.48–3.40 (m)"``; ``"1.20 (s)"``."""
    if result["pattern"] == "m":
        hi, lo = result["range_ppm"]
        return f"{hi:.{decimals}f}–{lo:.{decimals}f} (m)"
    centre = f"{result['centre_ppm']:.{decimals}f}"
    name = report_name(result["pattern"])
    if not result["J_hz"]:
        return f"{centre} ({name})"
    js = ", ".join(f"{j:.{j_decimals}f}" for j in result["J_hz"])
    return f"{centre} ({name}, J = {js} Hz)"


def format_report(results, nucleus: str | None = None, mhz: float | None = None,
                  decimals: int = 2, j_decimals: int = 1) -> str:
    """The line a paper quotes: ``¹H NMR (400 MHz) δ 7.26 (d, J = 8.1
    Hz), 3.45 (dd, J = 10.1, 4.0 Hz).`` -- high shift first. The frequency
    is rounded to the nearest whole MHz, as is usual."""
    head = []
    if nucleus:
        head.append(f"{nucleus_label(nucleus)} NMR")
    if mhz:
        head.append(f"({mhz:.0f} MHz)")
    ordered = sorted(results, key=lambda r: -r["centre_ppm"])
    body = ", ".join(format_multiplet(r, decimals, j_decimals) for r in ordered)
    return (" ".join(head) + " " if head else "") + f"δ {body}."


def report_rows(results) -> list[dict]:
    """One row per multiplet, for a table or a CSV."""
    rows = []
    for r in results:
        rows.append({"δ (ppm)": round(r["centre_ppm"], 4), "pattern": r["pattern"],
                     "J (Hz)": ", ".join(f"{j:.2f}" for j in r["J_hz"]),
                     "lines": len(r["lines_ppm"]),
                     "RMS (Hz)": None if not np.isfinite(r.get("rms_hz", np.nan))
                     else round(r["rms_hz"], 3),
                     "range (ppm)": f"{r['range_ppm'][0]:.4f}–{r['range_ppm'][1]:.4f}"})
    return rows


# ---- coupling trees ------------------------------------------------------------------------

def tree_segments(result: dict, top: float, bottom: float) -> list[tuple[tuple, tuple]]:
    """The coupling tree of one multiplet as line segments ``((x0, y0), (x1,
    y1))`` in ppm and in data y, from ``top`` down to ``bottom``: a stem at
    the centre, then one level per coupling (largest first), each node
    splitting into two J apart, then a drop to each line. Empty for a
    singlet or an ``m``."""
    js = result.get("couplings_hz") or []
    if not js or not result.get("first_order", False):
        return []
    mhz = result["mhz"]
    levels = len(js) + 1
    step = (top - bottom) / (levels + 1)
    segments = []
    nodes = [result["centre_ppm"]]
    y = top - step * 0.5
    segments.append(((nodes[0], top), (nodes[0], y)))
    for j in js:
        y_next = y - step
        children = []
        for x in nodes:
            for sign in (-0.5, 0.5):
                child = x + sign * j / mhz
                segments.append(((x, y), (child, y_next)))
                children.append(child)
        nodes = sorted(set(round(c, 9) for c in children))
        y = y_next
    for x in nodes:
        segments.append(((x, y), (x, bottom)))
    return segments


def plot_coupling_trees(ax, results, top: float | None = None, height: float | None = None,
                        color: str = "tab:blue", linewidth: float = 0.9, label: bool = True,
                        fontsize: float = 9.0):
    """Draw each multiplet's coupling tree above it on ``ax`` (a 1D
    spectrum in ppm), and its name and J values above the tree. ``top``
    and ``height`` are in data units; by default the trees sit just above
    the tallest data. Returns the artists drawn."""
    from matplotlib.collections import LineCollection
    if top is None or height is None:
        ylo, yhi = ax.get_ylim()
        span = yhi - ylo
        height = height or 0.12 * span
        top = top if top is not None else yhi - 0.08 * span
    artists = []
    for r in results:
        segs = tree_segments(r, top, top - height)
        if segs:
            lc = LineCollection(segs, colors=color, linewidths=linewidth, zorder=40)
            ax.add_collection(lc)
            artists.append(lc)
        if label:
            text = r["pattern"] if not r["J_hz"] else (
                f"{r['pattern']}\n" + ", ".join(f"{j:.1f}" for j in r["J_hz"]) + " Hz")
            x = r["centre_ppm"] if r["pattern"] != "m" else sum(r["range_ppm"]) / 2.0
            artists.append(ax.text(x, top + 0.01 * height, text, ha="center", va="bottom",
                                   color=color, fontsize=fontsize, zorder=41, clip_on=True))
    return artists
