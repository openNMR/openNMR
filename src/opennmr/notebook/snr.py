"""
Signal-to-noise ratios of 1D spectra, with the statistics stated.

Apply ▸ Signal-to-Noise in the GUI calls these functions exactly as a
notebook does (explicit user request, 2026-10-12: "enable the
calculation of the signal to noise ... across one (or more) spectra ...
ensure that statistically reasonable statements are made, and that it
is clearly logged when the regions differed between spectra").

Two regions, both in ppm:

* the **signal region** -- where the line is. Its largest point (above
  the noise region's mean level) is the signal height ``H``; the area
  over the region is also measured, for broad lines.
* the **noise region** -- somewhere with no signal. A polynomial trend
  (order ``detrend``, linear by default) is removed from it and the
  standard deviation of what is left is the noise ``σ``. Several
  windows may be given (``[(40, 30), (-30, -40)]``).

What is reported, and why:

* ``snr = H / σ`` (RMS-noise convention), with a 1σ uncertainty and a
  95 % interval. Both are conservative: the interval combines the
  extremes of the height's and of σ's own intervals, and the height's
  uncertainty is taken as one σ although the maximum of several noisy
  points near the top of a line scatters a little less (and sits a
  little high -- most at low S/N, where the report says so). Two things limit it: the noise on the height itself
  (``σ``), and how well ``σ`` is known -- which depends on the number of
  *independent* noise points, not the number of points. Apodisation,
  zero-filling and digital filtering correlate neighbouring points; the
  correlation time ``τ`` is measured from the noise's own
  autocorrelation and ``n_eff = n / τ`` is used throughout.
* ``sino = H / (2σ)``, the convention of TopSpin's ``sino`` command
  (with a linear trend removed from the noise, as here) -- so a number
  can be compared with one from TopSpin.
* ``ph_eur = 2H / h``, the European Pharmacopoeia definition, ``h``
  the peak-to-peak range of the noise. ``h`` grows with the number of
  noise points, so this one depends on the noise region's width.
* ``integral_snr``: the region's area over the noise on that area
  (``σ · Δx · √(n · τ)``) -- the better measure for broad lines.
* A **detection test**: in a signal region of pure noise the largest
  point would still stand some way above zero. ``detection_threshold``
  is the height (in σ) that pure noise exceeds only 5 % of the time
  over that many independent points; below it the "line" cannot be
  told from noise.
* **Checks on the noise region**: outliers (a line or a spike in it),
  skew and excess kurtosis (it is not Gaussian noise), too few
  independent points, overlap with the signal region.
* For several spectra (:func:`signal_to_noise_many`): each one against
  a reference spectrum, as a ratio with its uncertainty and whether it
  differs significantly; whether that ratio is what √(number of scans)
  predicts; and **whether the regions differed between spectra**, which
  is stated in the result and in the report, spectrum by spectrum,
  because S/N measured over different regions are not strictly
  comparable.

Example::

    from opennmr.notebook import snr
    r = snr.signal_to_noise(spectrum, signal_region=(12, 8), noise_region=(60, 40))
    print(snr.snr_report(r))

    many = snr.signal_to_noise_many([s1, s2, s3], signal_region=(12, 8),
                                    noise_region=(60, 40),
                                    regions={2: {"noise": (-40, -60)}})
    print(snr.snr_report(many))
    snr.plot_snr([s1, s2, s3], many)

A spectrum may be an :class:`~opennmr.core.spectrum.NMRSpectrum`, a
notebook results dict (``"x_ppm"`` and ``"real_spectrum"``, from
``NMR1dimport``) or an ``(x_ppm, y)`` pair.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

#: The noise-region polynomial order removed before σ is taken (1 =
#: linear, as TopSpin's ``sino`` does).
DEFAULT_DETREND = 1
#: Fewer independent noise points than this and σ itself is uncertain by
#: more than ~13 % (1/√(2·30)): said so.
MIN_INDEPENDENT_NOISE = 30
#: Fewer noise points than this cannot give a noise estimate at all.
MIN_NOISE_POINTS = 8

_SIGNS = ("positive", "negative", "absolute")
_OFFSETS = ("mean", "trend", "none")


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def _xy(spectrum):
    """(x ppm, real y, info dict) from any accepted spectrum."""
    info = {"nucleus": None, "ns": None, "apodisation": None}
    if isinstance(spectrum, dict):
        x = np.asarray(spectrum["x_ppm"], dtype=float)
        y = np.real(np.asarray(spectrum["real_spectrum"]))
        info["nucleus"] = spectrum.get("nuclide") or spectrum.get("nucleus")
        for key in ("NS", "ns"):
            if spectrum.get(key) is not None:
                info["ns"] = _positive(spectrum.get(key))
    elif isinstance(spectrum, (tuple, list)) and len(spectrum) == 2:
        x = np.asarray(spectrum[0], dtype=float)
        y = np.real(np.asarray(spectrum[1]))
    else:
        if getattr(spectrum, "ndim", 1) != 1:
            raise ValueError("Signal-to-noise needs 1D spectra; extract a slice first.")
        x = np.asarray(spectrum.axes[-1].ppm, dtype=float)
        y = np.real(np.asarray(spectrum.data))
        meta = spectrum.metadata
        info["nucleus"] = getattr(meta, "nucleus", None)
        info["ns"] = _positive(getattr(meta, "ns", None))
        raw = getattr(meta, "raw", None) or {}
        if info["ns"] is None:
            acqus = raw.get("acqus") if isinstance(raw.get("acqus"), dict) else {}
            info["ns"] = _positive(acqus.get("NS"))
        info["apodisation"] = _apodisation(raw)
    y = np.asarray(y, dtype=float).ravel()
    if x.shape != y.shape:
        raise ValueError(f"The axis ({x.size} points) and the data ({y.size}) differ in length.")
    return x, y, info


def _positive(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value > 0 else None


def _apodisation(raw: dict) -> str | None:
    """A short description of the window function, when it is recorded
    (the GUI's live processing keys, else Bruker's procs)."""
    window = raw.get("live_window")
    if window and window != "__recorded__":
        if window == "none":
            return "none"
        if window == "exponential":
            return f"exponential, LB {float(raw.get('live_lb', 0.0)):g} Hz"
        if window == "gaussian":
            return (f"gaussian, LB {float(raw.get('live_lb', 0.0)):g} Hz, "
                    f"GB {float(raw.get('live_gb', 0.0)):g}")
        if window in ("sine", "qsine"):
            return f"{window}, SSB {float(raw.get('live_ssb', 0.0)):g}"
        return str(window)
    procs = raw.get("procs") if isinstance(raw.get("procs"), dict) else {}
    wdw = procs.get("WDW")
    if wdw is None:
        return None
    names = {0: "none", 1: "exponential", 2: "gaussian", 3: "sine", 4: "qsine"}
    try:
        name = names.get(int(wdw), str(wdw))
    except (TypeError, ValueError):
        name = str(wdw)
    if name == "exponential":
        return f"exponential, LB {float(procs.get('LB', 0.0)):g} Hz"
    if name == "gaussian":
        return f"gaussian, LB {float(procs.get('LB', 0.0)):g} Hz, GB {float(procs.get('GB', 0.0)):g}"
    if name in ("sine", "qsine"):
        return f"{name}, SSB {float(procs.get('SSB', 0.0)):g}"
    return name


def _window(region, what) -> tuple[float, float]:
    try:
        a, b = (float(v) for v in region)
    except (TypeError, ValueError):
        raise ValueError(f"The {what} region must be two numbers (ppm), not {region!r}.") from None
    if not (math.isfinite(a) and math.isfinite(b)) or a == b:
        raise ValueError(f"The {what} region needs two different, finite limits.")
    return (max(a, b), min(a, b))           # high ppm first, as spectra are drawn


def normalise_noise_region(region) -> list[tuple[float, float]]:
    """One window ``(a, b)`` or several ``[(a, b), (c, d)]`` -> a list of
    ``(high, low)`` windows."""
    if region is None:
        raise ValueError("Give a noise region.")
    region = list(region)
    if region and all(isinstance(v, (int, float, np.floating, np.integer)) for v in region):
        return [_window(region, "noise")]
    return [_window(w, "noise") for w in region]


def normalise_signal_region(region) -> tuple[float, float]:
    if region is None:
        raise ValueError("Give a signal region.")
    return _window(region, "signal")


def _mask(x, windows) -> np.ndarray:
    mask = np.zeros(x.shape, dtype=bool)
    for hi, lo in windows:
        mask |= (x >= lo) & (x <= hi)
    return mask


def _fmt_window(w) -> str:
    return f"{w[0]:.4g} to {w[1]:.4g} ppm"


def describe_regions(signal, noise) -> str:
    """'signal 12 to 8 ppm; noise 60 to 40 ppm (+ ...)' -- for logs."""
    noise = normalise_noise_region(noise)
    return (f"signal {_fmt_window(normalise_signal_region(signal))}; noise "
            + " + ".join(_fmt_window(w) for w in noise))


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _detrend(x, y, order):
    """Residual of ``y`` about a polynomial of ``order`` in ``x`` (scaled
    to ±1), and the polynomial's coefficients in that scaled variable."""
    centre = 0.5 * (x.max() + x.min())
    half = 0.5 * (x.max() - x.min()) or 1.0
    u = (x - centre) / half
    order = int(max(0, min(order, y.size - 2)))
    coef = np.polynomial.polynomial.polyfit(u, y, order)
    fitted = np.polynomial.polynomial.polyval(u, coef)
    return y - fitted, (coef, centre, half, order)


def _trend_at(trend, x):
    coef, centre, half, _order = trend
    return np.polynomial.polynomial.polyval((np.asarray(x, dtype=float) - centre) / half, coef)


def correlation_time(residual_segments: Sequence[np.ndarray]) -> float:
    """τ = 1 + 2 Σ ρ_k, the noise's integrated autocorrelation time in
    points, summed while ρ_k stays positive (the initial positive
    sequence) and pooled over the windows. 1 for white noise; zero-filling
    by 2 and line broadening raise it."""
    num = np.zeros(1)
    den = 0.0
    kmax = 0
    for r in residual_segments:
        r = np.asarray(r, dtype=float)
        if r.size < 4:
            continue
        kmax = max(kmax, max(1, r.size // 5))
    if kmax == 0:
        return 1.0
    num = np.zeros(kmax + 1)
    for r in residual_segments:
        r = np.asarray(r, dtype=float)
        if r.size < 4:
            continue
        r = r - r.mean()
        den += float(np.dot(r, r))
        for k in range(1, min(kmax, r.size - 1) + 1):
            num[k] += float(np.dot(r[:-k], r[k:]))
    if den <= 0:
        return 1.0
    tau = 1.0
    for k in range(1, kmax + 1):
        rho = num[k] / den
        if rho <= 0:
            break
        tau += 2.0 * rho
    return float(max(1.0, tau))


def detection_threshold(n_independent: float, sign: str = "positive",
                        confidence: float = 0.95) -> float:
    """The height, in σ, that the largest of ``n_independent`` Gaussian
    noise points exceeds with probability ``1 - confidence``: a signal
    region of pure noise reaches it 5 % of the time (by default)."""
    from scipy.stats import norm
    m = max(1.0, float(n_independent))
    p = confidence ** (1.0 / m)
    if sign == "absolute":
        return float(norm.ppf(0.5 * (1.0 + p)))
    return float(norm.ppf(p))


def signal_to_noise(spectrum, signal_region, noise_region, *, detrend: int = DEFAULT_DETREND,
                    sign: str = "positive", offset: str = "mean", label: str | None = None,
                    confidence: float = 0.95) -> dict:
    """The signal-to-noise ratio of one 1D spectrum (see the module
    docstring for every quantity and its convention).

    Parameters
    ----------
    spectrum : NMRSpectrum, results dict or (x_ppm, y)
    signal_region : (a, b) ppm
    noise_region : (a, b) ppm, or a list of such windows
    detrend : int
        Polynomial order removed from the noise before σ (0 = mean only,
        1 = linear, as TopSpin's ``sino``; 2 = quadratic).
    sign : 'positive' | 'negative' | 'absolute'
        Which extremum of the signal region is the line.
    offset : 'mean' | 'trend' | 'none'
        The level the height is measured from: the noise region's mean
        (default), its fitted trend extrapolated to the line, or zero.
    label : str, optional
        A name for the report.
    confidence : float
        For the interval and the detection threshold.

    Returns
    -------
    dict
        ``snr``, ``snr_error`` (1σ), ``snr_interval`` (``confidence``),
        ``sino``, ``ph_eur``, ``integral_snr``, ``height``,
        ``position_ppm``, ``noise_sigma``, ``noise_sigma_error``,
        ``n_noise``, ``n_noise_independent``, ``correlation_time``,
        ``n_signal``, ``detection_threshold``, ``detected``, ``ns``,
        ``snr_per_sqrt_scan``, ``signal_region``, ``noise_region``,
        ``warnings`` (plain-language), and the settings used.
    """
    from scipy.stats import chi2, kurtosis, norm, skew

    if sign not in _SIGNS:
        raise ValueError(f"sign must be one of {_SIGNS}, not {sign!r}")
    if offset not in _OFFSETS:
        raise ValueError(f"offset must be one of {_OFFSETS}, not {offset!r}")
    detrend = int(detrend)
    if detrend < 0 or detrend > 3:
        raise ValueError("detrend must be 0 (offset only), 1 (linear), 2 or 3.")
    x, y, info = _xy(spectrum)
    sig = normalise_signal_region(signal_region)
    noise = normalise_noise_region(noise_region)
    for w in noise:
        if w[1] < sig[0] and w[0] > sig[1]:
            raise ValueError(
                f"The noise region {_fmt_window(w)} overlaps the signal region "
                f"{_fmt_window(sig)}: noise must be measured where there is no signal.")
    warnings: list[str] = []

    # --- noise -----------------------------------------------------------
    noise_mask = _mask(x, noise)
    n_noise = int(noise_mask.sum())
    if n_noise < MIN_NOISE_POINTS:
        raise ValueError(f"The noise region holds {n_noise} point(s); at least "
                         f"{MIN_NOISE_POINTS} are needed (widen it).")
    xn, yn = x[noise_mask], y[noise_mask]
    residual, trend = _detrend(xn, yn, detrend)
    order = trend[3]
    dof = max(1, n_noise - (order + 1))
    sigma = float(np.sqrt(np.sum(residual ** 2) / dof))
    if not sigma > 0:
        raise ValueError("The noise region is flat (σ = 0): is it a simulated or zeroed region?")
    segments = [residual[_mask(xn, [w])] for w in noise]
    tau = correlation_time(segments)
    n_eff = max(2.0, n_noise / tau)
    dof_eff = max(1.0, n_eff - (order + 1))
    sigma_rel_err = 1.0 / math.sqrt(2.0 * dof_eff)
    q_lo, q_hi = 0.5 * (1 - confidence), 0.5 * (1 + confidence)
    sigma_lo = sigma * math.sqrt(dof_eff / chi2.ppf(q_hi, dof_eff))
    sigma_hi = sigma * math.sqrt(dof_eff / chi2.ppf(q_lo, dof_eff))
    noise_mean = float(np.mean(yn))
    z = residual / sigma
    peak_to_peak = float(residual.max() - residual.min())

    if n_eff < MIN_INDEPENDENT_NOISE:
        warnings.append(
            f"Only ~{n_eff:.0f} independent noise points (of {n_noise}; neighbours are "
            f"correlated, τ = {tau:.2g}): σ is uncertain by ±{100 * sigma_rel_err:.0f} %. "
            "Widen the noise region.")
    if n_eff >= 8:
        s = float(skew(z))
        k = float(kurtosis(z))           # excess
        se_s, se_k = math.sqrt(6.0 / n_eff), math.sqrt(24.0 / n_eff)
        if abs(s) > 3 * se_s or k > 3 * se_k:
            warnings.append(
                f"The noise region is not Gaussian noise (skew {s:.2g}, excess kurtosis "
                f"{k:.2g}): it may hold part of a line, a spinning sideband, a spike or a "
                "baseline roll. Choose a cleaner region or raise the detrend order.")
    else:
        s = k = float("nan")
    z_limit = float(norm.ppf(0.5 * (1 + 0.99 ** (1.0 / n_eff))))
    worst = int(np.argmax(np.abs(z)))
    if abs(z[worst]) > z_limit:
        warnings.append(
            f"A point in the noise region at {xn[worst]:.4g} ppm is {abs(z[worst]):.1f} σ from "
            f"the trend -- more than pure noise reaches 1 % of the time ({z_limit:.1f} σ). "
            "There may be signal or a spike there.")

    # --- signal ----------------------------------------------------------
    sig_mask = _mask(x, [sig])
    n_sig = int(sig_mask.sum())
    if n_sig < 1:
        raise ValueError(f"The signal region {_fmt_window(sig)} holds no points of this spectrum "
                         f"({x.min():.4g} to {x.max():.4g} ppm).")
    xs, ys = x[sig_mask], y[sig_mask]
    if offset == "mean":
        level = np.full(ys.shape, noise_mean)
    elif offset == "trend":
        level = _trend_at(trend, xs)
        span = xn.max() - xn.min()
        far = max(0.0, max(xn.min() - xs.max(), xs.min() - xn.max()))
        if order > 0 and far > span:
            warnings.append(
                "The height is measured from the noise region's trend, extrapolated further "
                "than the noise region is wide: a linear or curved trend extrapolated that far "
                "is unreliable. Prefer the noise region's mean level, or put noise on both sides.")
    else:
        level = np.zeros(ys.shape)
    above = ys - level
    if sign == "positive":
        j = int(np.argmax(above))
        height = float(above[j])
    elif sign == "negative":
        j = int(np.argmin(above))
        height = float(-above[j])
    else:
        j = int(np.argmax(np.abs(above)))
        height = float(abs(above[j]))
    position = float(xs[j])
    edge_k = int(max(3, round(2 * tau)))
    if (height > 3 * sigma and n_sig > 2 * edge_k
            and _line_continues_past_edge(x, y, sig, j, n_sig, level[j], sign, edge_k)):
        warnings.append(
            f"The largest point is at the edge of the signal region ({position:.4g} ppm): "
            "the line may continue beyond it. Widen the signal region.")

    snr = height / sigma
    snr_error = math.sqrt(1.0 + (snr * sigma_rel_err) ** 2)
    k95 = float(norm.ppf(q_hi))
    snr_interval = ((height - k95 * sigma) / sigma_hi, (height + k95 * sigma) / sigma_lo)
    n_sig_eff = max(1.0, n_sig / tau)
    threshold = detection_threshold(n_sig_eff, sign, confidence)
    detected = bool(snr > threshold)
    if not detected:
        warnings.append(
            f"S/N {snr:.2g} is below {threshold:.2g}, the height pure noise reaches in this signal "
            f"region {100 * (1 - confidence):.0f} % of the time (~{n_sig_eff:.0f} independent "
            "points): no line is detected here.")
    elif snr < 10:
        warnings.append(
            "At S/N below ~10 the largest point is biased upwards by the noise on it (the maximum "
            "is picked from noisy points); treat the value as an upper estimate.")

    # --- integral --------------------------------------------------------
    if n_sig >= 2:
        order_idx = np.argsort(xs)
        area = float(np.trapezoid(above[order_idx], xs[order_idx])) if hasattr(np, "trapezoid") \
            else float(np.trapz(above[order_idx], xs[order_idx]))
        if sign == "negative":
            area = -area
        dx = float(np.mean(np.abs(np.diff(xs[order_idx])))) if n_sig > 1 else 0.0
        area_sigma = sigma * dx * math.sqrt(n_sig * tau)
        integral_snr = area / area_sigma if area_sigma > 0 else float("nan")
        if sign == "absolute":
            integral_snr = abs(integral_snr)
    else:
        area = area_sigma = integral_snr = float("nan")

    ns = info["ns"]
    return {
        "label": label or "spectrum",
        "snr": float(snr), "snr_error": float(snr_error),
        "snr_interval": (float(snr_interval[0]), float(snr_interval[1])),
        "confidence": float(confidence),
        "sino": float(height / (2.0 * sigma)),
        "ph_eur": float(2.0 * height / peak_to_peak) if peak_to_peak > 0 else float("nan"),
        "integral_snr": float(integral_snr), "area": float(area), "area_sigma": float(area_sigma),
        "height": height, "position_ppm": position, "level": float(level[j]),
        "noise_sigma": sigma, "noise_sigma_error": sigma * sigma_rel_err,
        "noise_sigma_interval": (float(sigma_lo), float(sigma_hi)),
        "noise_mean": noise_mean, "noise_peak_to_peak": peak_to_peak,
        "noise_skew": s, "noise_excess_kurtosis": k,
        "n_noise": n_noise, "n_noise_independent": float(n_eff), "correlation_time": float(tau),
        "n_signal": n_sig, "n_signal_independent": float(n_sig_eff),
        "detection_threshold": threshold, "detected": detected,
        "ns": ns, "snr_per_sqrt_scan": float(snr / math.sqrt(ns)) if ns else None,
        "nucleus": info["nucleus"], "apodisation": info["apodisation"],
        "signal_region": sig, "noise_region": noise,
        "detrend": order, "sign": sign, "offset": offset,
        "warnings": warnings,
    }


def _line_continues_past_edge(x, y, sig, j, n_sig, level, sign, k) -> bool:
    """True when the extremum (index ``j`` of the signal region's points)
    is within ``k`` points of an edge and the ``k`` points just outside
    that edge average further from ``level`` (in the line's direction)
    than the ``k`` just inside it: the line goes on past the edge."""
    xs_idx = np.nonzero((x >= sig[1]) & (x <= sig[0]))[0]
    if xs_idx.size < 2:
        return False
    if j < k:
        inside = xs_idx[:k]
        start = xs_idx[0]
        outside = np.arange(max(0, start - k), start)
    elif j >= n_sig - k:
        inside = xs_idx[-k:]
        end = xs_idx[-1] + 1
        outside = np.arange(end, min(x.size, end + k))
    else:
        return False
    if outside.size == 0:
        return False
    a, b = float(np.mean(y[outside]) - level), float(np.mean(y[inside]) - level)
    if sign == "negative":
        return a < b
    if sign == "absolute":
        return abs(a) > abs(b)
    return a > b


# ---------------------------------------------------------------------------
# Several spectra
# ---------------------------------------------------------------------------

def _per_spectrum_regions(regions, n, labels):
    """``regions``: None, a list (one entry per spectrum, None = shared),
    or a dict keyed by index or label. Each entry {"signal": ..., "noise":
    ...}, either key optional."""
    out = [None] * n
    if regions is None:
        return out
    if isinstance(regions, dict):
        for key, value in regions.items():
            if isinstance(key, (int, np.integer)):
                i = int(key)
            elif key in labels:
                i = labels.index(key)
            else:
                raise ValueError(f"No spectrum {key!r} for its own regions.")
            if not 0 <= i < n:
                raise ValueError(f"No spectrum {key!r} for its own regions.")
            out[i] = value
    else:
        regions = list(regions)
        if len(regions) != n:
            raise ValueError("Give one regions entry per spectrum (None to use the shared ones).")
        out = regions
    return out


def signal_to_noise_many(spectra, signal_region=None, noise_region=None, *, labels=None,
                         regions=None, reference: int = 0, **kwargs) -> dict:
    """:func:`signal_to_noise` of each spectrum, by default with the same
    regions for all; ``regions`` gives any spectrum its own (``{2:
    {"noise": (-40, -60)}}``, by index or label; a missing key falls back
    to the shared region). Each row says whether its regions were the
    shared ones, and the result whether any differed.

    Each spectrum is compared with ``reference`` (index): the ratio of
    their S/N with its 1σ uncertainty, whether it differs from 1 at 95 %,
    and -- when both record a number of scans -- whether it is what
    √(NS ratio) predicts.

    Returns ``{"rows", "comparisons", "regions_differ", "differing",
    "shared_signal", "shared_noise", "notes", "reference"}``.
    """
    spectra = list(spectra)
    n = len(spectra)
    if n == 0:
        raise ValueError("No spectra.")
    labels = list(labels) if labels is not None else [f"Spectrum {k + 1}" for k in range(n)]
    if len(labels) != n:
        raise ValueError("Give one label per spectrum.")
    own = _per_spectrum_regions(regions, n, labels)
    shared_sig = normalise_signal_region(signal_region) if signal_region is not None else None
    shared_noise = normalise_noise_region(noise_region) if noise_region is not None else None
    rows = []
    for i, spectrum in enumerate(spectra):
        entry = own[i] or {}
        sig = entry.get("signal") if entry.get("signal") is not None else shared_sig
        noise = entry.get("noise") if entry.get("noise") is not None else shared_noise
        if sig is None or noise is None:
            raise ValueError(f"{labels[i]}: no signal or noise region (give shared ones, or its own).")
        row = signal_to_noise(spectrum, sig, noise, label=labels[i], **kwargs)
        row["own_signal"] = (shared_sig is None or
                             normalise_signal_region(sig) != shared_sig)
        row["own_noise"] = (shared_noise is None or
                            normalise_noise_region(noise) != shared_noise)
        rows.append(row)

    first = rows[0]
    regions_differ = any(r["signal_region"] != first["signal_region"]
                         or r["noise_region"] != first["noise_region"] for r in rows)
    if not regions_differ:
        differing = []
    elif shared_sig is not None and shared_noise is not None:
        # Those that set their own; the rest used the shared regions.
        differing = [r["label"] for r in rows if r["own_signal"] or r["own_noise"]]
    else:
        differing = [r["label"] for r in rows
                     if r["signal_region"] != first["signal_region"]
                     or r["noise_region"] != first["noise_region"]]

    notes = []
    if regions_differ:
        notes.append(
            "The regions differed between spectra (" + "; ".join(
                f"{r['label']}: {describe_regions(r['signal_region'], r['noise_region'])}"
                for r in rows if r["label"] in differing) + "). S/N measured over different "
            "regions are not strictly comparable: a different noise window samples different "
            "noise (and a baseline that differs), a different signal window can pick a "
            "different line.")
    nuclei = {r["nucleus"] for r in rows if r["nucleus"]}
    if len(nuclei) > 1:
        notes.append(f"The spectra are of different nuclei ({', '.join(sorted(nuclei))}).")
    apod = {r["apodisation"] for r in rows if r["apodisation"]}
    if len(apod) > 1:
        notes.append("The spectra were apodised differently (" + "; ".join(sorted(apod)) + "): "
                     "line broadening raises S/N, so the ratios include that, not only the "
                     "sample or the experiment.")
    taus = [r["correlation_time"] for r in rows]
    if max(taus) > 1.5 * min(taus):
        notes.append("The noise is correlated over different numbers of points "
                     f"(τ {min(taus):.2g} to {max(taus):.2g}): the spectra differ in zero-filling "
                     "or apodisation, which changes the noise per point.")

    if not 0 <= reference < n:
        raise ValueError(f"reference must be an index 0..{n - 1}.")
    ref = rows[reference]
    comparisons = []
    for i, r in enumerate(rows):
        if i == reference:
            continue
        comparisons.append(compare_snr(r, ref))
    return {"rows": rows, "comparisons": comparisons, "regions_differ": bool(regions_differ),
            "differing": differing,
            "shared_signal": shared_sig, "shared_noise": shared_noise,
            "notes": notes, "reference": reference}


def compare_snr(a: dict, b: dict, confidence: float = 0.95) -> dict:
    """S/N of ``a`` relative to ``b``: the ratio, its 1σ uncertainty
    (relative errors in quadrature), whether it differs from 1 at
    ``confidence``, and -- when both record a number of scans -- the
    ratio √(NS_a/NS_b) predicts and whether the measured one agrees."""
    from scipy.stats import norm
    k = float(norm.ppf(0.5 * (1 + confidence)))
    ratio = a["snr"] / b["snr"] if b["snr"] else float("nan")
    rel = math.sqrt((a["snr_error"] / a["snr"]) ** 2 + (b["snr_error"] / b["snr"]) ** 2) \
        if a["snr"] and b["snr"] else float("nan")
    err = abs(ratio) * rel
    out = {"label": a["label"], "reference": b["label"], "ratio": ratio, "ratio_error": err,
           "significant": bool(err > 0 and abs(ratio - 1.0) > k * err),
           "expected_from_scans": None, "consistent_with_scans": None,
           "regions_differ": a["signal_region"] != b["signal_region"]
           or a["noise_region"] != b["noise_region"]}
    if a.get("ns") and b.get("ns") and a["ns"] != b["ns"]:
        expected = math.sqrt(a["ns"] / b["ns"])
        out["expected_from_scans"] = expected
        out["consistent_with_scans"] = bool(abs(ratio - expected) <= k * err) if err > 0 else None
    return out


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _g(v, digits=3) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "—"
    return f"{v:.{digits}g}"


def _row_report(r: dict, indent: str = "") -> list[str]:
    lo, hi = r["snr_interval"]
    pct = f"{100 * r['confidence']:.0f} %"
    order = {0: "its mean removed", 1: "a linear trend removed",
             2: "a quadratic trend removed"}.get(r["detrend"], f"an order-{r['detrend']} trend removed")
    level = {"mean": "the noise region's mean level", "trend": "the noise region's trend",
             "none": "zero"}[r["offset"]]
    lines = [
        f"{indent}S/N {_g(r['snr'], 4)} ± {_g(r['snr_error'], 2)} ({pct} interval {_g(lo)} to "
        f"{_g(hi)}, conservative) = height / σ_noise.",
        f"{indent}  Signal: {_fmt_window(r['signal_region'])}; the line at "
        f"{r['position_ppm']:.4g} ppm, height {_g(r['height'], 4)} above {level}.",
        f"{indent}  Noise: " + " + ".join(_fmt_window(w) for w in r["noise_region"])
        + f", {r['n_noise']} points (~{r['n_noise_independent']:.0f} independent, τ = "
        f"{r['correlation_time']:.2g}), {order}; σ = {_g(r['noise_sigma'], 4)} ± "
        f"{100 * r['noise_sigma_error'] / r['noise_sigma']:.0f} %.",
        f"{indent}  Other conventions: TopSpin sino (H/2σ) {_g(r['sino'], 4)}; Ph. Eur. 2H/h "
        f"{_g(r['ph_eur'], 4)} (h peak-to-peak; depends on the noise width); integral S/N "
        f"{_g(r['integral_snr'], 4)}.",
        f"{indent}  Detection: pure noise in this signal region would reach "
        f"{_g(r['detection_threshold'])} σ {100 * (1 - r['confidence']):.0f} % of the time -- "
        + ("the line is detected." if r["detected"] else "NOT detected."),
    ]
    if r.get("ns"):
        lines.append(f"{indent}  NS = {r['ns']:g}: S/N per √scan {_g(r['snr_per_sqrt_scan'], 4)}.")
    if r.get("apodisation"):
        lines.append(f"{indent}  Apodisation: {r['apodisation']}.")
    for w in r["warnings"]:
        lines.append(f"{indent}  ⚠ {w}")
    return lines


def snr_report(result: dict) -> str:
    """Plain text: every number, the conventions, the checks -- and, for
    several spectra, the regions each used and the comparisons."""
    if "rows" not in result:
        return "\n".join([f"{result['label']}:"] + _row_report(result, "  "))
    lines = []
    rows = result["rows"]
    if result["regions_differ"]:
        lines.append("⚠ REGIONS DIFFERED BETWEEN SPECTRA: " + ", ".join(result["differing"])
                     + " used their own (marked †).")
    elif rows:
        lines.append("Same regions for every spectrum: "
                     + describe_regions(rows[0]["signal_region"], rows[0]["noise_region"]) + ".")
    for r in rows:
        mark = " †" if r["label"] in result["differing"] else ""
        lines.append(f"{r['label']}{mark}:")
        lines.extend(_row_report(r, "  "))
    if result["comparisons"]:
        ref = rows[result["reference"]]["label"]
        lines.append(f"Against {ref}:")
        for c in result["comparisons"]:
            text = (f"  {c['label']}: S/N ratio {_g(c['ratio'], 4)} ± {_g(c['ratio_error'], 2)} -- "
                    + ("differs significantly (95 %)." if c["significant"]
                       else "not significantly different (95 %)."))
            if c["expected_from_scans"] is not None:
                text += (f" √(NS ratio) predicts {_g(c['expected_from_scans'], 4)}: "
                         + ("consistent." if c["consistent_with_scans"] else
                            "NOT consistent -- something besides the number of scans changed."))
            if c["regions_differ"]:
                text += " (different regions)"
            lines.append(text)
    for note in result["notes"]:
        lines.append("Note: " + note)
    return "\n".join(lines)


def print_snr_report(result: dict) -> None:
    print(snr_report(result))


def snr_table(result: dict) -> list[dict]:
    """One flat dict per spectrum (for a CSV or a DataFrame)."""
    rows = result["rows"] if "rows" in result else [result]
    differing = set(result.get("differing") or [])
    out = []
    for r in rows:
        out.append({
            "spectrum": r["label"], "S/N": r["snr"], "S/N error (1σ)": r["snr_error"],
            "S/N low": r["snr_interval"][0], "S/N high": r["snr_interval"][1],
            "sino (H/2σ)": r["sino"], "Ph. Eur. 2H/h": r["ph_eur"],
            "integral S/N": r["integral_snr"], "height": r["height"],
            "line (ppm)": r["position_ppm"], "noise σ": r["noise_sigma"],
            "noise points": r["n_noise"], "independent noise points": r["n_noise_independent"],
            "detected": r["detected"], "NS": r["ns"], "S/N per √scan": r["snr_per_sqrt_scan"],
            "signal region (ppm)": _fmt_window(r["signal_region"]),
            "noise region (ppm)": " + ".join(_fmt_window(w) for w in r["noise_region"]),
            "own regions": r["label"] in differing,
            "warnings": " | ".join(r["warnings"]),
        })
    return out


def snr_to_csv(result: dict, path: str) -> None:
    import csv
    rows = snr_table(result)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_snr(spectra, result: dict | None = None, labels=None, figure=None, mode: str = "stack",
             xlims=None, cmap: str = "black", fontsize: float = 12.0,
             signal_colour: str = "#009e73", noise_colour: str = "0.6", regions=None):
    """The spectra with each one's signal region (green) and noise region
    (grey) shaded in its own row, the measured line marked and its S/N
    written beside it. ``mode`` 'stack' (one row each) or 'overlay' (the
    shading then shows the first spectrum's regions; differing regions
    are drawn as outlines). Without a ``result``, ``regions`` -- one
    ``{"signal": (a, b) or None, "noise": [(a, b), ...] or None}`` per
    spectrum -- are shaded as a preview. Returns the figure."""
    import matplotlib.pyplot as plt
    from ..plotting.style import stack_colors
    from .dnp import _nucleus_label, _spectrum_axes

    if result is not None:
        rows = result["rows"] if "rows" in result else [result]
        if "rows" not in result and not (isinstance(spectra, list) and len(spectra) == 1):
            spectra = [spectra]          # one spectrum (any accepted form), one result
    else:
        if not isinstance(spectra, list):
            spectra = [spectra]
        regions = list(regions) if regions is not None else [None] * len(spectra)
        rows = []
        for k, reg in enumerate(regions):
            reg = reg or {}
            rows.append({"label": f"Spectrum {k + 1}",
                         "signal_region": (normalise_signal_region(reg["signal"])
                                           if reg.get("signal") is not None else None),
                         "noise_region": (normalise_noise_region(reg["noise"])
                                          if reg.get("noise") is not None else [])})
    if len(rows) != len(spectra):
        raise ValueError("Give the spectra the result was measured on (one each).")
    n = len(rows)
    labels = list(labels) if labels is not None else [r["label"] for r in rows]
    if figure is None:
        figure = plt.figure(figsize=(3.375 * 1.6, 3.375 if n == 1 or mode == "overlay"
                                     else min(1.4 + 0.6 * n, 9.0)))
    else:
        figure.clear()
    ax = figure.add_subplot(111)
    palette = stack_colors(n, cmap)
    data = [_xy(s) for s in spectra]
    edges = []
    for r in rows:
        if r.get("signal_region") is not None:
            edges.extend(r["signal_region"])
        for w in r.get("noise_region") or []:
            edges.extend(w)
    if xlims is not None:
        lo_x, hi_x = min(xlims), max(xlims)
    elif len(edges) >= 2 and max(edges) > min(edges):
        span = max(edges) - min(edges)
        lo_x, hi_x = min(edges) - 0.1 * span, max(edges) + 0.1 * span
    else:
        lo_x = min(float(np.min(x)) for x, _y, _i in data)
        hi_x = max(float(np.max(x)) for x, _y, _i in data)
    traces = []
    for (x, y, _info), r in zip(data, rows):
        if "noise_mean" in r:
            base = r["noise_mean"]
        elif r.get("noise_region"):
            m = _mask(x, r["noise_region"])
            base = float(np.mean(y[m])) if np.any(m) else 0.0
        else:
            base = 0.0
        r_base = y - base
        inside = (x >= lo_x) & (x <= hi_x)
        if not np.any(inside):
            inside = np.ones(x.size, bool)
        traces.append((x, r_base, float(np.min(r_base[inside])), float(np.max(r_base[inside])),
                       base))
    offsets = np.zeros(n)
    if mode == "stack":
        span = max((t[3] - t[2]) for t in traces) or 1.0
        level = 0.0
        for i in range(1, n):
            level += traces[i - 1][3] - traces[i][2] + 0.12 * span
            offsets[i] = level
    differing = set((result or {}).get("differing") or [])
    for i, ((x, y, lo, hi, base), r) in enumerate(zip(traces, rows)):
        off = offsets[i]
        ax.plot(x, y + off, color=palette[i], linewidth=0.9, label=labels[i])
        windows = ([("signal", r["signal_region"])] if r.get("signal_region") is not None else [])
        windows += [("noise", w) for w in r.get("noise_region") or []]
        for kind, (w_hi, w_lo) in windows:
            colour = signal_colour if kind == "signal" else noise_colour
            if mode == "stack":
                ax.fill_between([w_lo, w_hi], off + lo, off + hi, color=colour, alpha=0.18,
                                linewidth=0, zorder=0)
            elif i == 0:
                ax.axvspan(w_lo, w_hi, color=colour, alpha=0.18, linewidth=0, zorder=0)
            elif labels[i] in differing or result is None:
                ax.axvspan(w_lo, w_hi, fill=False, edgecolor=colour, linestyle="--",
                           linewidth=0.8, zorder=0)
        if "snr" in r:
            ypk = r["height"] * (-1 if r["sign"] == "negative" else 1) + r["level"] - base
            ax.plot([r["position_ppm"]], [ypk + off], marker="v" if ypk >= 0 else "^",
                    color=palette[i], markersize=4, linestyle="none")
            mark = " †" if labels[i] in differing else ""
            ax.annotate(f"S/N {r['snr']:.3g}{mark}", xy=(r["position_ppm"], ypk + off),
                        xytext=(4, 2), textcoords="offset points", fontsize=fontsize * 0.6,
                        color=palette[i])
        if mode == "stack" and n > 1:
            # Above the row's noise (5σ when measured), not over the trace.
            above_noise = 5.0 * r["noise_sigma"] if "noise_sigma" in r else 0.15 * (hi - lo)
            ax.annotate(labels[i], xy=(0.0, off + min(above_noise, 0.5 * hi)),
                        xycoords=("axes fraction", "data"),
                        xytext=(2, 3), textcoords="offset points", fontsize=fontsize * 0.6,
                        color=palette[i], clip_on=True)
    nucleus = rows[0].get("nucleus") or (data[0][2].get("nucleus") if data else "")
    ax.set_xlabel(f"{_nucleus_label(nucleus or '')} shift (ppm)".strip(), fontsize=fontsize)
    ax.set_xlim(hi_x, lo_x)
    if mode == "overlay" and 1 < n <= 12:
        ax.legend(frameon=False, fontsize=fontsize * 0.7)
    _spectrum_axes(ax, fontsize)
    figure.tight_layout()
    return figure


def snr_python(signal_region, noise_region, regions=None, labels=None, **kwargs) -> str:
    """The notebook call that reproduces a GUI measurement (for "Copy
    notebook code")."""
    extras = "".join(f", {k}={v!r}" for k, v in kwargs.items())
    lines = ["from opennmr.notebook import snr",
             "# spectra: the same spectra, in the same order (NMRSpectrum, NMR1dimport dicts, "
             "or (x_ppm, y) pairs)"]
    if labels is not None:
        lines.append(f"labels = {list(labels)!r}")
    signal = tuple(signal_region) if signal_region is not None else None
    call = (f"result = snr.signal_to_noise_many(spectra, signal_region={signal!r}, "
            f"noise_region={noise_region!r}"
            + (", labels=labels" if labels is not None else "")
            + (f", regions={regions!r}" if regions else "") + extras + ")")
    lines.append(call)
    lines.append("print(snr.snr_report(result))")
    lines.append("snr.plot_snr(spectra, result" + (", labels=labels" if labels else "") + ")")
    return "\n".join(lines)
