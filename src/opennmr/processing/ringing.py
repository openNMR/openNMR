"""
Truncation ringing: "wiggles" beside a line that look like a multiplet but are not.

A time-domain FID that has not decayed by the last point acquired (or kept) is cut off
abruptly, and its Fourier transform then carries a sinc ripple beside every strong line.
The lobes are spaced by ``1 / T`` (``T`` the length of the FID that was transformed, so
``sw / n_points`` in Hz) and decay slowly, which is why they read as a multiplet in a row
of a 2D spectrum -- most of all on its tallest peaks. It is not in the sample, and a smooth
window (a sine-bell, ``SINE`` / ``QSINE`` in TopSpin) removes it, at the price of broader
lines.

This module answers two questions with numbers instead of an impression:

* **Is the FID truncated?** :func:`diagnose_truncation` compares the amplitude left in the
  last points with the noise (measured from the spectrum's own quietest regions) and with
  the FID's start, and measures how much the spectrum would change if a smooth taper
  closed the edge -- the size of the wiggles, as a percentage of the tallest line.
* **Which window removes it for the least cost?** :func:`suggest_window` tries sine and
  squared-sine bells of increasing strength on the strongest row and returns the one with
  the narrowest noise bandwidth that brings the ripple below a threshold.

The GUI's Process ▸ Direct dimension ▸ "Check for truncation wiggles" and
:func:`opennmr.notebook.ringing.check_f2_ringing` call these exactly.

The check says what the *FID* does. Whether TopSpin's window was applied here (a recorded
``WDW`` openNMR cannot reproduce leaves F2 unwindowed) is the notebook wrapper's job.
"""

from __future__ import annotations

import numpy as np

from . import ft

#: Windows tried by :func:`suggest_window`, mildest first in the sense that matters (the
#: narrowest noise bandwidth that clears the threshold wins, however they are listed).
CANDIDATE_WINDOWS = (
    ("sine", 4.0), ("sine", 3.0), ("sine", 2.0),
    ("qsine", 4.0), ("qsine", 3.0), ("qsine", 2.0),
)

#: Ripple (largest change a closed edge makes, as a fraction of the tallest line) at which
#: the wiggles are called: negligible / mild / moderate / strong.
SEVERITY_LIMITS = (0.005, 0.02, 0.08)
SEVERITY_NAMES = ("none", "mild", "moderate", "strong")

#: A ripple under this is what :func:`suggest_window` aims for.
TARGET_RIPPLE = 0.005

#: Zero-fill used to look at the lobes: the unfilled grid samples a sinc at its nulls.
_ZERO_FILL_FACTOR = 8


def window_weights(window: str | None, n: int, dt: float, **params) -> np.ndarray:
    """The weights a named window gives ``n`` points -- the ones the readers apply.
    ``None`` / ``"none"``: no window (all ones)."""
    if window in (None, "none"):
        return np.ones(n)
    if window not in ft._WINDOWS:
        raise ValueError(f"Unknown window {window!r}; available: none, {', '.join(sorted(ft._WINDOWS))}")
    return np.asarray(ft._WINDOWS[window](n, dt, **params), dtype=float)


def noise_bandwidth(weights: np.ndarray) -> float:
    """Effective noise bandwidth of a window relative to no window (1.0): the factor by
    which it widens the noise, and, to a good approximation, what it costs in resolution."""
    w = np.asarray(weights, dtype=float)
    total = float(np.sum(w))
    if total <= 0:
        return float("inf")
    return float(w.size * np.sum(w * w) / total ** 2)


def _rows(fid) -> np.ndarray:
    rows = np.asarray(fid)
    if rows.ndim == 1:
        rows = rows[None, :]
    if rows.ndim != 2:
        raise ValueError("fid must be one FID or a 2D array with one FID per row")
    return rows.astype(complex)


def _spectral_noise(row: np.ndarray, block: int = 32) -> float:
    """Noise per real/imaginary component of ``fft(row)``: the median of the quietest quarter
    of block-wise scatter between neighbouring points, so lines and their ripple do not
    inflate it (an overestimate if the whole spectrum is lines -- the check then errs towards
    saying nothing is wrong)."""
    spectrum = np.fft.fft(row)
    parts = [np.real(spectrum), np.imag(spectrum)]
    n_blocks = spectrum.size // block
    if n_blocks < 4:
        diffs = np.concatenate([np.diff(part) for part in parts])
        return float(np.std(diffs) / np.sqrt(2.0))
    scatter = []
    for part in parts:
        for k in range(n_blocks):
            chunk = part[k * block:(k + 1) * block]
            scatter.append(np.std(np.diff(chunk)) / np.sqrt(2.0))
    scatter = np.sort(np.asarray(scatter))
    return float(np.median(scatter[:max(1, scatter.size // 4)]))


def _edge_taper(n: int, fraction: float = 0.25) -> np.ndarray:
    """1 for the first 1 - ``fraction`` of the points, then a cos^2 fall to 0."""
    taper = np.ones(n)
    k = max(2, int(round(fraction * n)))
    taper[-k:] = 0.5 * (1.0 + np.cos(np.pi * np.arange(k) / (k - 1)))
    return taper


def ripple_fraction(row: np.ndarray, weights: np.ndarray | None = None) -> float:
    """How much of the spectrum of one windowed FID is owed to the abrupt end of the FID,
    as a fraction of its tallest line: the largest difference between the spectrum and the
    spectrum of the same FID with its last quarter smoothly tapered to zero (both zero-filled
    eight-fold so the lobes are seen between their nulls).

    Measured away from the lines themselves (more than two points of the unfilled grid from
    anything over a quarter of the tallest line), where the lobes remain but the taper's
    slight narrowing of the lines is not mistaken for them; a spectrum that is nearly all
    line is measured everywhere."""
    row = np.asarray(row, dtype=complex)
    w = np.ones(row.size) if weights is None else np.asarray(weights, dtype=float)
    nfft = _ZERO_FILL_FACTOR * row.size
    windowed = row * w
    plain = np.fft.fft(windowed, nfft)
    closed = np.fft.fft(windowed * _edge_taper(row.size), nfft)
    tallest = float(np.max(np.abs(plain)))
    if tallest <= 0:
        return 0.0
    scale = tallest / max(float(np.max(np.abs(closed))), 1e-300)
    difference = np.abs(plain - closed * scale)
    strong = (np.abs(closed) * scale > 0.25 * tallest).astype(float)
    reach = 2 * _ZERO_FILL_FACTOR
    near = np.convolve(np.concatenate([strong[-reach:], strong, strong[:reach]]),
                       np.ones(2 * reach + 1), mode="same")[reach:-reach] > 0
    away = difference[~near]
    if away.size < 0.1 * difference.size:
        away = difference
    return float(np.max(away) / tallest)


def _severity(ripple: float) -> str:
    for limit, name in zip(SEVERITY_LIMITS, SEVERITY_NAMES):
        if ripple < limit:
            return name
    return SEVERITY_NAMES[-1]


def diagnose_truncation(fid, sw_hz: float, *, sf_mhz: float | None = None, truncate: int | None = None,
                        window_weights: np.ndarray | None = None, tail_fraction: float = 0.05,
                        strongest_rows: int = 8) -> dict:
    """Is the FID (or each row of a 2D's F2) cut off before it has decayed?

    ``fid``: one complex FID or a 2D array, one FID per row (a hypercomplex 2D's rows both
    count). ``sw_hz``: the spectral width. ``truncate``: use only this many points (TopSpin's
    TDeff). ``window_weights``: the window the spectrum is processed with, as weights for the
    points used (see :func:`window_weights`); ``None`` for no window. ``tail_fraction``: the
    last fraction of the points that is "the end".

    Returns a dict, among it:

    ``truncated``      bool -- signal is still present at the end of the FID, in at least
                       one of the strongest rows
    ``tail_snr``       the amplitude left in the last points over the noise amplitude there
    ``tail_to_start``  that amplitude over the FID's start amplitude (strongest row)
    ``ripple``         wiggle size without a window, as a fraction of the tallest line
    ``ripple_windowed`` the same with ``window_weights`` applied (equal to ``ripple`` if none)
    ``severity``       ``"none"``, ``"mild"``, ``"moderate"`` or ``"strong"`` (of the windowed)
    ``period_hz`` / ``period_ppm``  the spacing of the wiggles, ``sw / n_points`` (ppm needs
                       ``sf_mhz``) -- a splitting of exactly this size is this, not a coupling
    ``onset_index``    where the strongest row's amplitude peaks (a late peak: an echo or an
                       unremoved group delay, a different problem)
    ``suggestion``     :func:`suggest_window`'s answer, ``None`` when nothing is needed
    """
    rows = _rows(fid)
    if truncate:
        rows = rows[:, :int(truncate)]
    n = rows.shape[1]
    if n < 32:
        raise ValueError(f"{n} points are too few to judge truncation (need at least 32)")
    if not sw_hz or sw_hz <= 0:
        raise ValueError("sw_hz must be the positive spectral width in Hz")
    if window_weights is not None:
        window_weights = np.asarray(window_weights, dtype=float)
        if window_weights.shape != (n,):
            raise ValueError(f"window_weights has {window_weights.size} points; the FID used has {n}")
    m = max(4, int(round(tail_fraction * n)))
    head = max(4, int(round(0.1 * n)))
    strength = np.sqrt(np.mean(np.abs(rows[:, :head]) ** 2, axis=1))
    order = np.argsort(strength)[::-1][:max(1, strongest_rows)]
    strongest = int(order[0])
    if strength[strongest] <= 0:
        raise ValueError("the FID is all zeros")

    tail_snrs, tail_ratios = [], []
    for index in order:
        row = rows[index]
        sigma_component = _spectral_noise(row) / np.sqrt(n)
        end_power = float(np.mean(np.abs(row[-m:]) ** 2))
        excess = np.sqrt(max(end_power - 2.0 * sigma_component ** 2, 0.0))
        start = float(np.sqrt(np.mean(np.abs(row[:m]) ** 2)))
        tail_snrs.append(excess / (np.sqrt(2.0) * sigma_component) if sigma_component > 0
                         else (np.inf if excess > 0 else 0.0))
        tail_ratios.append(excess / start if start > 0 else 0.0)
    best_row = rows[strongest]
    ripple_plain = ripple_fraction(best_row, None)
    ripple_win = ripple_plain if window_weights is None else ripple_fraction(best_row, window_weights)
    ripple_gate = max(ripple_win, 0.0)
    tail_snr = float(max(tail_snrs))
    truncated = bool(tail_snr > 3.0 and max(tail_ratios) > 0.01)
    # A tail inside the noise cannot ring visibly whatever the arithmetic says.
    severity = _severity(ripple_gate) if truncated else "none"

    smooth = np.convolve(np.abs(best_row[:max(8, n // 4)]), np.ones(4) / 4.0, mode="same")
    onset = int(np.argmax(smooth))
    period_hz = float(sw_hz) / n
    result = {
        "n_points": int(n), "n_rows": int(rows.shape[0]), "rows_checked": [int(i) for i in order],
        "strongest_row": strongest, "acquisition_time_s": n / float(sw_hz),
        "tail_points": int(m), "tail_snr": tail_snr, "tail_to_start": float(max(tail_ratios)),
        "truncated": truncated, "ripple": float(ripple_plain), "ripple_windowed": float(ripple_win),
        "severity": severity, "period_hz": period_hz,
        "period_ppm": (period_hz / float(sf_mhz)) if sf_mhz else None,
        "onset_index": onset, "onset_late": bool(onset > max(8, 0.02 * n)),
        "window_noise_bandwidth": (noise_bandwidth(window_weights) if window_weights is not None else 1.0),
    }
    result["suggestion"] = suggest_window(rows[strongest], sw_hz, current_weights=window_weights) \
        if severity != "none" else None
    result["report"] = format_report(result)
    return result


def suggest_window(row, sw_hz: float, *, current_weights: np.ndarray | None = None,
                   target: float = TARGET_RIPPLE) -> dict | None:
    """The sine or squared-sine bell that clears the wiggles of ``row`` (one FID) for the
    narrowest noise bandwidth; the strongest candidate if none reaches ``target``. ``None``
    if the current window already does.

    Returns ``{"window", "ssb", "label", "ripple", "noise_bandwidth", "reaches_target"}``.
    ``window`` and ``ssb`` go straight into the readers' ``window=`` / ``window_kwargs=``
    (``{"ssb": ssb}``) or the Process tab's Window and Sine-bell shift."""
    row = np.asarray(row, dtype=complex)
    n = row.size
    dt = 1.0 / float(sw_hz)
    if current_weights is not None and ripple_fraction(row, current_weights) < target:
        return None
    tried = []
    for key, ssb in CANDIDATE_WINDOWS:
        weights = window_weights(key, n, dt, ssb=ssb)
        tried.append({"window": key, "ssb": ssb, "ripple": ripple_fraction(row, weights),
                      "noise_bandwidth": noise_bandwidth(weights)})
    good = [t for t in tried if t["ripple"] < target]
    pick = min(good, key=lambda t: t["noise_bandwidth"]) if good else min(tried, key=lambda t: t["ripple"])
    pick = dict(pick)
    pick["reaches_target"] = bool(good)
    pick["label"] = f"{pick['window'].upper()}, SSB {pick['ssb']:g}"
    return pick


def format_report(diagnosis: dict) -> str:
    """The diagnosis in sentences, for the GUI and a notebook."""
    d = diagnosis
    period = f"{d['period_hz']:.3g} Hz"
    if d.get("period_ppm"):
        period += f" ({d['period_ppm']:.3g} ppm)"
    lines = [f"{d['n_points']} points over {d['acquisition_time_s'] * 1000:.3g} ms; the "
             f"{len(d['rows_checked'])} strongest of {d['n_rows']} row(s) were checked."]
    if d["truncated"]:
        lines.append(
            f"The FID is cut off while it still carries signal: over the last {d['tail_points']} points "
            f"the amplitude is {d['tail_snr']:.3g} × the noise and {100 * d['tail_to_start']:.2g} % of the "
            f"FID's start.")
    else:
        lines.append("The FID has decayed into the noise by its end (or is within a factor of three of it).")
    lines.append(f"Without a window the transform would carry wiggles up to {100 * d['ripple']:.2g} % of the "
                 f"tallest line; with the window in use, {100 * d['ripple_windowed']:.2g} % "
                 f"({d['severity']}).")
    if d["severity"] != "none":
        lines.append(f"They repeat every {period} (1 / the length of the FID). A splitting of that size in "
                     f"a row is almost certainly this, not a coupling; one of a different size is not.")
    else:
        lines.append(f"Any wiggle repeating every {period} would be this; at this size it is too small to "
                     f"explain a multiplet.")
    if d.get("onset_late"):
        lines.append(f"The FID's amplitude peaks {d['onset_index']} points in, not at the start: an echo, or "
                     f"a digital-filter delay that was not removed. That is a different problem (try "
                     f"Align Echo, or check the digital filter) and does not ring like this.")
    suggestion = d.get("suggestion")
    if suggestion:
        lines.append(
            f"Suggested: {suggestion['label']}, which brings the wiggles to {100 * suggestion['ripple']:.2g} % "
            f"for a noise bandwidth of {suggestion['noise_bandwidth']:.2f} × (no window = 1; lines come out "
            f"about that much wider)."
            + ("" if suggestion["reaches_target"] else
               " It does not reach the target on its own: add forward linear prediction, or acquire longer."))
    elif d["severity"] != "none":
        lines.append("The window in use is already as good as the ones tried; use linear prediction or a "
                     "longer acquisition.")
    return "\n".join(lines)
