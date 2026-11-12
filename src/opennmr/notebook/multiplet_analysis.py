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
    units, rounding = (intensity_units(heights) if heights is not None
                       else (np.ones(shifts.size, dtype=int), 0.0))
    result["units"] = [int(u) for u in units[::-1]]
    if heights is None:
        result["notes"].append("No heights given: every line counted as one unit, so lines of "
                               "a triplet or quartet that coincide cannot be told apart.")
    elif rounding > 0.35:
        result["notes"].append(f"The heights do not fall into whole units well (off by up to "
                               f"{rounding:.2f} of a unit): check phase and baseline, or the "
                               "multiplet may not be first order.")
    offsets = np.repeat(line_hz - line_hz[0], units)
    js = decompose(offsets, tolerance_hz)
    total = int(units.sum())
    if js is None or 2 ** len(js) != total:
        result.update(pattern="m", centre_ppm=float(np.average(shifts, weights=units)), J_hz=[],
                      counts=[], rms_hz=float("nan"), first_order=False)
        result["notes"].append(
            "Not a first-order pattern within the tolerance: reported as a multiplet (m) with "
            "its range. Try a larger tolerance, check the picked lines and their heights, or fit "
            "it as a strongly coupled spin system.")
        return result
    first, js, rms = _refine(line_hz, units, js, max(tolerance_hz, 1e-9) * 4)
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
    if spectrum is not None:
        mhz = mhz or spectrum_mhz(spectrum)
        if snap:
            shifts = snap_to_maxima(spectrum, shifts)
        if heights is None:
            heights = heights_at(spectrum, shifts)
    if not mhz:
        raise ValueError("The spectrometer frequency is unknown: set the field (or the nucleus "
                         "and ¹H frequency) for this spectrum, or pass mhz=.")
    heights = None if heights is None else np.asarray(heights, dtype=float)
    out = []
    for group in group_lines(shifts, mhz, max_gap_hz):
        result = analyse_multiplet(shifts[group], mhz,
                                   None if heights is None else heights[group], tolerance_hz)
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
    if not result["J_hz"]:
        return f"{centre} ({result['pattern']})"
    js = ", ".join(f"{j:.{j_decimals}f}" for j in result["J_hz"])
    return f"{centre} ({result['pattern']}, J = {js} Hz)"


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
