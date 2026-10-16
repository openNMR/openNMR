"""
Full width at half maximum (FWHM) of the peak at a clicked position, in
1D and 2D spectra -- model-free where the peak is resolved, and an
*estimate*, said so, where it is not.

Explicit user request (2026-10-16): "determine the peak widths (the full
width half max) by clicking on the peak itself ... care when dealing with
overlapping peaks ... a warning should be given that these values are
estimates due to overlap."

How a width is measured (:func:`peak_fwhm`):

1. **The peak.** From the clicked point, climb (on a lightly smoothed
   copy, so a noise spike does not stop the climb) to the local maximum
   of the signal's own sign, then refine the apex on the raw data with a
   three-point parabola.
2. **Half height** is half the apex height above ``baseline`` (0 by
   default: the spectrum is taken to be baseline-corrected; pass a number,
   or ``"local"`` for the lower of the two valleys either side).
3. **Each side** is walked outward until the signal falls to half
   height; the crossing is interpolated linearly between the two points
   either side of it. If the signal turns upward again *before* falling
   to half height (by more than three noise σ and 2 % of the height),
   that side runs into a neighbouring peak: it is **blocked**.
4. **Overlap** is reported when a side is blocked, when the valley
   beyond a crossing is above 10 % of the peak height (the neighbour's
   tail then lifts the half-height points and widens the peak), or when
   the two half-widths differ by more than 30 % (an unresolved shoulder).
   Then:

   * one side blocked: FWHM = 2 × the clean side's half width (a
     symmetric line assumed);
   * both blocked: a Lorentzian and a Gaussian are fitted to the top of
     the peak (above the higher valley) and the better one's FWHM is
     given;
   * in every overlap case the local-fit widths are reported as well, so
     the spread between the estimates shows how uncertain the number is.

   ``estimate`` is True and ``warnings`` say why. For overlapped peaks a
   deconvolution (Fit ▸ Spectral Fitting) is the right measurement.

Widths come back in the x unit given (ppm in the GUI) and in Hz when
``hz_per_ppm`` is given. In 2D (:func:`peak_fwhm_2d`) the widths are those
of the F2 row and F1 column through the apex: for a tilted (correlated)
lineshape these are cross-section widths, not principal-axis widths.
"""

from __future__ import annotations

import math

import numpy as np

#: A side is blocked when the signal rises again by this much (fraction of
#: the peak height, and at least 3 noise σ) before reaching half height.
RISE_FRACTION = 0.02
#: A valley beyond the half-height crossing above this fraction of the peak
#: height means the neighbour's tail reaches into the measured width.
VALLEY_FRACTION = 0.10
#: Half widths differing by more than this fraction: an unresolved shoulder.
ASYMMETRY_FRACTION = 0.30


def _noise_sigma(y: np.ndarray) -> float:
    """Noise σ from the median absolute first difference (robust to the
    peaks themselves): σ ≈ 1.4826 · MAD(Δy) / √2."""
    d = np.diff(np.asarray(y, dtype=float))
    if d.size < 3:
        return 0.0
    return float(1.4826 * np.median(np.abs(d - np.median(d))) / math.sqrt(2.0))


def _smooth(y: np.ndarray, n: int = 5) -> np.ndarray:
    if y.size < 3 * n:
        return y
    kernel = np.ones(n) / n
    return np.convolve(y, kernel, mode="same")


def _climb(yy: np.ndarray, i: int) -> int:
    """Index of the local maximum reached by walking uphill from ``i``."""
    n = yy.size
    while True:
        left = yy[i - 1] if i > 0 else -np.inf
        right = yy[i + 1] if i < n - 1 else -np.inf
        if left > yy[i] and left >= right:
            i -= 1
        elif right > yy[i]:
            i += 1
        else:
            return i


def _apex(x, yy, i):
    """Parabolic apex (position, height) from the three points at i."""
    if 0 < i < yy.size - 1:
        y0, y1, y2 = yy[i - 1], yy[i], yy[i + 1]
        denom = y0 - 2 * y1 + y2
        if denom < 0:
            p = 0.5 * (y0 - y2) / denom
            if abs(p) <= 1:
                dx = x[i + 1] - x[i] if p > 0 else x[i] - x[i - 1]
                return float(x[i] + p * dx), float(y1 - 0.25 * (y0 - y2) * p)
    return float(x[i]), float(yy[i])


def _walk(x, yy, i_top, half, step, rise):
    """Walk from the apex in direction ``step`` (+1/-1).

    Returns ``(crossing_x or None, valley_value, valley_index, blocked)``:
    the interpolated half-height crossing, the lowest value reached before
    the signal rises again (or the end), and whether the side was blocked
    by a neighbour before reaching half height."""
    n = yy.size
    j = i_top
    running_min, min_index = yy[i_top], i_top
    crossing = None
    while 0 <= j + step < n:
        k = j + step
        if crossing is None and yy[k] <= half:
            # interpolate between j (above) and k (at/below)
            frac = (yy[j] - half) / (yy[j] - yy[k]) if yy[j] != yy[k] else 0.0
            crossing = float(x[j] + frac * (x[k] - x[j]))
        if yy[k] < running_min:
            running_min, min_index = yy[k], k
        elif yy[k] - running_min > rise:
            # turned upward: a neighbouring peak
            return crossing, float(running_min), min_index, crossing is None
        j = k
    return crossing, float(running_min), min_index, crossing is None


def _shape_fit(x, yy, i_lo, i_hi, centre, height):
    """Lorentzian and Gaussian fitted to yy[i_lo:i_hi+1] (the top of the
    peak). Returns {shape: (fwhm, rms)}; empty when it cannot be done."""
    from scipy.optimize import curve_fit

    xs, ys = x[i_lo:i_hi + 1], yy[i_lo:i_hi + 1]
    if xs.size < 4:
        return {}
    span = float(abs(xs[-1] - xs[0])) or 1.0

    def lor(t, a, c, w):
        return a / (1.0 + ((t - c) / (0.5 * w)) ** 2)

    def gau(t, a, c, w):
        return a * np.exp(-4.0 * math.log(2.0) * ((t - c) / w) ** 2)

    out = {}
    for name, f in (("Lorentzian", lor), ("Gaussian", gau)):
        try:
            p, _ = curve_fit(f, xs, ys, p0=(height, centre, span), maxfev=4000,
                             bounds=([0, xs.min(), 1e-12 * span], [10 * height + 1e-300,
                                                                   xs.max(), 100 * span]))
        except Exception:  # noqa: BLE001 -- an estimate that fails is simply not offered
            continue
        rms = float(np.sqrt(np.mean((f(xs, *p) - ys) ** 2)))
        out[name] = (float(abs(p[2])), rms)
    return out


def peak_fwhm(x, y, x_click: float, baseline=0.0, hz_per_ppm: float | None = None,
              unit: str = "ppm") -> dict:
    """FWHM of the peak at ``x_click`` (see the module docstring).

    ``x``, ``y``: the spectrum (any order of x). ``baseline``: 0, a number,
    or ``"local"``. ``hz_per_ppm``: the spectrometer frequency in MHz, to
    give the width in Hz too (``unit`` must then be ppm).

    Returns a dict: ``position``, ``height`` (signed, above the baseline),
    ``baseline``, ``half_height`` (absolute y of the half-height line),
    ``left``/``right`` (the half-height x positions used, None where not
    measured), ``fwhm``, ``fwhm_hz``, ``method``, ``estimate``,
    ``overlap``, ``warnings``, ``alternatives`` ({name: fwhm}), ``points``
    (data points across the FWHM), ``unit``."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or x.shape != y.shape or x.size < 5:
        raise ValueError("peak_fwhm needs matching 1D x and y with at least 5 points.")
    order = np.argsort(x)
    x, y = x[order], y[order]
    if not (x[0] <= x_click <= x[-1]):
        raise ValueError(f"{x_click:g} {unit} is outside the spectrum.")
    base_value = 0.0 if isinstance(baseline, str) else float(baseline)
    i0 = int(np.argmin(np.abs(x - x_click)))
    sign = 1.0 if y[i0] - base_value >= 0 else -1.0
    yy = sign * (y - base_value)
    sigma = _noise_sigma(yy)
    i_top = _climb(_smooth(yy), i0)
    lo, hi = max(i_top - 3, 0), min(i_top + 4, yy.size)
    i_top = lo + int(np.argmax(yy[lo:hi]))      # back onto the raw maximum
    position, height = _apex(x, yy, i_top)
    warnings: list[str] = []
    if height <= 0:
        raise ValueError("No peak there: the signal is not above the baseline.")

    if baseline == "local":
        # the lower of the valleys either side, found with a provisional walk
        _, vl, _, _ = _walk(x, yy, i_top, -np.inf, -1, max(3 * sigma, RISE_FRACTION * height))
        _, vr, _, _ = _walk(x, yy, i_top, -np.inf, +1, max(3 * sigma, RISE_FRACTION * height))
        shift = min(vl, vr)
        yy = yy - shift
        base_value = base_value + sign * shift
        height -= shift

    if sigma > 0 and height < 5 * sigma:
        warnings.append(f"The peak is only {height / sigma:.1f}× the noise σ: its width is poorly "
                        "defined.")
    half = 0.5 * height
    rise = max(3 * sigma, RISE_FRACTION * height)
    left, vl, il, blocked_l = _walk(x, yy, i_top, half, -1, rise)
    right, vr, ir, blocked_r = _walk(x, yy, i_top, half, +1, rise)

    overlap = False
    hw_l = position - left if left is not None else None
    hw_r = right - position if right is not None else None
    if blocked_l or blocked_r:
        overlap = True
        sides = " and ".join(s for s, b in (("low-x", blocked_l), ("high-x", blocked_r)) if b)
        warnings.append(f"Overlap: the signal does not fall to half height on the {sides} side "
                        "before a neighbouring peak.")
    for side, v, edge, blocked in (("low-x", vl, il in (0,), blocked_l),
                                   ("high-x", vr, ir == yy.size - 1, blocked_r)):
        if not blocked and not edge and v > VALLEY_FRACTION * height:
            overlap = True
            warnings.append(f"Overlap: on the {side} side the valley to the next peak is at "
                            f"{100 * v / height:.0f} % of the height, so that peak's tail widens "
                            "this one at half height.")
    if left is None and il == 0 and not blocked_l:
        warnings.append("The low-x side reaches the end of the spectrum before half height.")
    if right is None and ir == yy.size - 1 and not blocked_r:
        warnings.append("The high-x side reaches the end of the spectrum before half height.")
    if hw_l is not None and hw_r is not None:
        mean = 0.5 * (hw_l + hw_r)
        if mean > 0 and abs(hw_l - hw_r) / mean > ASYMMETRY_FRACTION:
            overlap = True
            warnings.append(f"Asymmetric: the half widths differ by {100 * abs(hw_l - hw_r) / mean:.0f} %"
                            " -- possibly an unresolved shoulder (or a lineshape that is not "
                            "symmetric).")

    alternatives: dict = {}
    if overlap or hw_l is None or hw_r is None:
        top = max(vl, vr, 0.0) + 0.02 * height
        i_lo, i_hi = i_top, i_top
        while i_lo > 0 and yy[i_lo - 1] > top and i_lo - 1 > il - 1:
            i_lo -= 1
        while i_hi < yy.size - 1 and yy[i_hi + 1] > top and i_hi + 1 < ir + 1:
            i_hi += 1
        for name, (w, _rms) in sorted(_shape_fit(x, yy, i_lo, i_hi, position, height).items(),
                                      key=lambda kv: kv[1][1]):
            alternatives[f"{name} fit to the top"] = w

    if hw_l is not None and hw_r is not None:
        fwhm, method = hw_l + hw_r, "half height (both sides)"
    elif hw_l is not None or hw_r is not None:
        hw = hw_l if hw_l is not None else hw_r
        fwhm, method = 2.0 * hw, "2 × the clear side's half width (symmetric line assumed)"
    elif alternatives:
        name = next(iter(alternatives))
        fwhm, method = alternatives[name], name
    else:
        raise ValueError("No half-height point on either side, and no line shape could be "
                         "fitted to the top: the width cannot be estimated here.")
    estimate = bool(overlap or hw_l is None or hw_r is None)
    if estimate:
        warnings.append("This FWHM is an ESTIMATE because of overlap; fit the peaks "
                        "(Fit ▸ Spectral Fitting) for a reliable width.")

    step = float(np.median(np.diff(x)))
    points = fwhm / step if step > 0 else math.inf
    if points < 3:
        warnings.append(f"Only {points:.1f} data points across the FWHM: the width is limited by "
                        "the digital resolution (zero-fill, or reduce the line broadening).")
    out = {
        "position": position, "height": sign * height, "baseline": base_value,
        "half_height": base_value + sign * half, "left": left, "right": right,
        "fwhm": float(fwhm), "fwhm_hz": None, "method": method, "estimate": estimate,
        "overlap": overlap, "warnings": warnings, "alternatives": alternatives,
        "points": float(points), "unit": unit,
    }
    if hz_per_ppm:
        if unit != "ppm":
            raise ValueError("hz_per_ppm applies to an x axis in ppm.")
        out["fwhm_hz"] = float(fwhm) * float(hz_per_ppm)
        out["alternatives_hz"] = {k: (v * float(hz_per_ppm) if v is not None else None)
                                  for k, v in alternatives.items()}
    return out


def fwhm_summary(result: dict, label: str = "") -> str:
    """One line: 'FWHM = 0.123 ppm / 49.2 Hz at 3.45 ppm (estimate: overlap)'."""
    u = result.get("unit", "ppm")
    text = f"FWHM = {result['fwhm']:.4g} {u}"
    if result.get("fwhm_hz") is not None:
        text += f" / {result['fwhm_hz']:.4g} Hz"
    text += f" at {result['position']:.4g} {u}"
    if result.get("estimate"):
        text += " (ESTIMATE: overlap)"
    return (label + ": " if label else "") + text


def spectrum_peak_fwhm(spectrum, shift: float, baseline=0.0) -> dict:
    """:func:`peak_fwhm` on a 1D :class:`NMRSpectrum` at ``shift`` (ppm),
    with Hz from the axis's own ppm/Hz arrays (or SFO1)."""
    if spectrum.ndim != 1:
        raise ValueError("spectrum_peak_fwhm needs a 1D spectrum; use peak_fwhm_2d for a 2D.")
    axis = spectrum.axes[0]
    return peak_fwhm(axis.ppm, spectrum.data, shift, baseline=baseline,
                     hz_per_ppm=_hz_per_ppm(axis, spectrum.metadata.sfo1))


def _hz_per_ppm(axis, sfo1=None):
    ppm, hz = getattr(axis, "ppm", None), getattr(axis, "hz", None)
    if ppm is not None and hz is not None and len(ppm) > 1:
        dp = float(ppm[-1] - ppm[0])
        dh = float(hz[-1] - hz[0])
        if dp and dh:
            return abs(dh / dp)
    return float(sfo1) if sfo1 else None


def peak_fwhm_2d(f1, f2, data, f1_click: float, f2_click: float, baseline=0.0,
                 hz_per_ppm_f1: float | None = None, hz_per_ppm_f2: float | None = None) -> dict:
    """FWHM along F2 and F1 of the 2D peak nearest the click.

    ``f1``, ``f2``: axis values (ppm) for ``data``'s rows and columns.
    The apex is found by climbing on |data| of the clicked point's sign in
    2D; the widths are those of the row and the column through it
    (:func:`peak_fwhm` on each, with its overlap tests). Returns
    ``{"f1_position", "f2_position", "height", "f2": result, "f1": result,
    "estimate", "warnings"}``."""
    f1 = np.asarray(f1, dtype=float)
    f2 = np.asarray(f2, dtype=float)
    data = np.asarray(data, dtype=float)
    if data.shape != (f1.size, f2.size):
        raise ValueError("data must be (len(f1), len(f2)).")
    r = int(np.argmin(np.abs(f1 - f1_click)))
    c = int(np.argmin(np.abs(f2 - f2_click)))
    base = 0.0 if isinstance(baseline, str) else float(baseline)
    sign = 1.0 if data[r, c] - base >= 0 else -1.0
    z = sign * (data - base)
    while True:
        r0, r1 = max(r - 1, 0), min(r + 2, z.shape[0])
        c0, c1 = max(c - 1, 0), min(c + 2, z.shape[1])
        block = z[r0:r1, c0:c1]
        k = np.unravel_index(int(np.argmax(block)), block.shape)
        nr, nc = r0 + k[0], c0 + k[1]
        if (nr, nc) == (r, c) or z[nr, nc] <= z[r, c]:
            break
        r, c = nr, nc
    row = peak_fwhm(f2, data[r, :], float(f2[c]), baseline=baseline, hz_per_ppm=hz_per_ppm_f2)
    col = peak_fwhm(f1, data[:, c], float(f1[r]), baseline=baseline, hz_per_ppm=hz_per_ppm_f1)
    warnings = [f"F2: {w}" for w in row["warnings"]] + [f"F1: {w}" for w in col["warnings"]]
    return {"f1_position": col["position"], "f2_position": row["position"],
            "height": float(data[r, c]), "f2": row, "f1": col,
            "estimate": bool(row["estimate"] or col["estimate"]), "warnings": warnings}


def spectrum_peak_fwhm_2d(spectrum, f2_shift: float, f1_shift: float, baseline=0.0) -> dict:
    """:func:`peak_fwhm_2d` on a 2D :class:`NMRSpectrum` (ppm on both axes)."""
    if spectrum.ndim != 2:
        raise ValueError("spectrum_peak_fwhm_2d needs a 2D spectrum.")
    a1, a2 = spectrum.axes
    meta = spectrum.metadata
    return peak_fwhm_2d(a1.ppm, a2.ppm, spectrum.data, f1_shift, f2_shift, baseline=baseline,
                        hz_per_ppm_f1=_hz_per_ppm(a1, meta.sfo1_indirect),
                        hz_per_ppm_f2=_hz_per_ppm(a2, meta.sfo1))
