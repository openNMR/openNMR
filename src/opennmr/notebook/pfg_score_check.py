"""SCORE next to the integral decay fit, with a consistency check (improvement notes §4).

SCORE resolves a PFG series into component spectra without choosing peaks, so it is a
valuable cross-check of the integral fit -- but it can fail (truncated FIDs, lineshape drift
between rows). :func:`score_with_check` runs SCORE for ``k = 2, 3`` and flags a result when:

* its water-like (fastest) D differs from the integral fit's by more than ``tol`` (5 %); or
* a second component takes more than ``split_fraction`` (40 %) of the *overlapping* signal
  **and** at least ``own_overlap`` (70 %) of each of the two components' own signal lies in the
  overlap -- the signature of one line drifting between rows being split in two. (The second
  condition keeps two genuinely different species that merely overlap in one region from being
  flagged; the notes' 40 % rule alone cannot tell them apart.)

:func:`align_rows` removes a frequency drift between rows before SCORE.
"""

from __future__ import annotations

import numpy as np

from ..processing.score import score as _score

__all__ = ["align_rows", "overlap_shares", "check_score_result", "score_with_check",
           "integral_fit_for_check", "compact_check", "format_check"]


def align_rows(ppm, data, reference: int = 0, max_shift_ppm: float | None = None):
    """Shift every row onto row ``reference`` (frequency drift between gradient levels).

    The shift of each row is the lag, refined by a parabola through the three highest
    cross-correlation points, that best matches the (magnitude-normalised) reference; rows
    are moved by linear interpolation. Returns ``(aligned, shifts_ppm)``."""
    x = np.asarray(ppm, dtype=float)
    Y = np.atleast_2d(np.asarray(data, dtype=float))
    order = np.argsort(x)
    xs, Ys = x[order], Y[:, order]
    step = float(np.median(np.diff(xs)))
    ref = Ys[reference] / (np.max(np.abs(Ys[reference])) or 1.0)
    max_lag = None if max_shift_ppm is None else int(round(abs(max_shift_ppm) / step))
    shifts = np.zeros(Y.shape[0])
    aligned = Ys.copy()
    n = xs.size
    for r in range(Y.shape[0]):
        if r == reference:
            continue
        row = Ys[r] / (np.max(np.abs(Ys[r])) or 1.0)
        corr = np.correlate(row, ref, mode="full")          # lag = index - (n - 1)
        lags = np.arange(-(n - 1), n)
        if max_lag is not None:
            corr = np.where(np.abs(lags) <= max_lag, corr, -np.inf)
        k = int(np.argmax(corr))
        frac = 0.0
        if 0 < k < corr.size - 1 and np.isfinite(corr[k - 1]) and np.isfinite(corr[k + 1]):
            denom = corr[k - 1] - 2.0 * corr[k] + corr[k + 1]
            frac = 0.5 * (corr[k - 1] - corr[k + 1]) / denom if denom else 0.0
        lag = lags[k] + frac                                  # row is displaced by +lag points
        shifts[r] = lag * step
        aligned[r] = np.interp(xs + lag * step, xs, Ys[r], left=0.0, right=0.0)
    back = np.empty_like(aligned)
    back[:, order] = aligned
    return back, shifts


def overlap_shares(spectra, threshold: float = 0.05) -> np.ndarray:
    """Each component's share of the signal where at least two components exceed
    ``threshold`` of the largest value (sums to 1; zeros when nothing overlaps)."""
    S = np.clip(np.asarray(spectra, dtype=float), 0.0, None)
    if S.size == 0 or S.max() <= 0:
        return np.zeros(S.shape[0])
    mask = np.sum(S > threshold * S.max(), axis=0) >= 2
    if not mask.any():
        return np.zeros(S.shape[0])
    part = S[:, mask].sum(axis=1)
    return part / part.sum()


def _fit_D(fit: dict) -> list[float]:
    names = fit["param_names"]
    if "D" in names:
        return [float(fit["params"]["D"])]
    return [float(fit["params"][n]) for n in names if n.startswith("D")]


def check_score_result(result: dict, integral_fit: dict | None = None, tol: float = 0.05,
                       split_fraction: float = 0.4, overlap_threshold: float = 0.05,
                       own_overlap: float = 0.7) -> dict:
    """Consistency flags for one SCORE result. Returns ``{"ok", "flags", "D_score",
    "D_integral", "water_difference", "overlap_shares"}``."""
    flags = []
    D_score = np.asarray(result.get("D", result.get("constants")), dtype=float).ravel()
    water_score = float(np.max(D_score)) if D_score.size else float("nan")
    D_int = _fit_D(integral_fit) if integral_fit is not None else []
    water_int = max(D_int) if D_int else float("nan")
    diff = abs(water_score - water_int) / water_int if D_int else float("nan")
    if D_int and diff > tol:
        flags.append(f"SCORE's water-like D ({water_score:.3g} m²/s) differs from the integral "
                     f"fit's ({water_int:.3g}) by {100 * diff:.0f} % (> {100 * tol:g} %).")
    S = np.clip(np.asarray(result["spectra"], dtype=float), 0.0, None)
    shares = overlap_shares(S, overlap_threshold)
    order = np.argsort(shares)[::-1]
    if order.size >= 2 and shares[order[1]] > split_fraction:
        # A split line has *both* parts sitting under each other; two real species that merely
        # share a region each keep most of their signal elsewhere.
        mask = np.sum(S > overlap_threshold * S.max(), axis=0) >= 2
        inside = [float(S[i, mask].sum() / S[i].sum()) if S[i].sum() > 0 else 0.0
                  for i in order[:2]]
        if min(inside) >= own_overlap:
            flags.append(f"A second component takes {100 * shares[order[1]]:.0f} % of the "
                         f"overlapping signal (> {100 * split_fraction:g} %) and both components "
                         f"lie {100 * min(inside):.0f} % or more inside the overlap: one line "
                         "drifting between rows can be split in two – align the rows or check "
                         "the FID truncation.")
    return {"ok": not flags, "flags": flags, "D_score": D_score, "D_integral": D_int,
            "water_difference": diff, "overlap_shares": shares}


def score_with_check(b, data, integral_fit: dict | None = None, ks=(2, 3), align_ppm=None,
                     tol: float = 0.05, split_fraction: float = 0.4, **score_kwargs) -> list[dict]:
    """Run SCORE for each ``k`` in ``ks`` next to an integral fit and check each.

    ``align_ppm``: pass the ppm axis to align the rows first (see :func:`align_rows`).
    Returns one ``{"k", "result", "check", "shifts_ppm"}`` per ``k``; the component spectra
    are ``result["spectra"]`` (plot them beside the decay)."""
    data = np.asarray(data, dtype=float)
    shifts = None
    if align_ppm is not None:
        data, shifts = align_rows(align_ppm, data)
    out = []
    for k in ks:
        result = _score(np.asarray(b, dtype=float), data, n_components=int(k), kind="diffusion",
                        **score_kwargs)
        out.append({"k": int(k), "result": result, "shifts_ppm": shifts,
                    "check": check_score_result(result, integral_fit, tol, split_fraction)})
    return out


def integral_fit_for_check(results) -> dict | None:
    """The curve-fit result :func:`check_score_result` should compare against: of a list of
    per-peak diffusion fits, the one with the fastest (water-like) D, or ``None``."""
    best, best_D = None, -np.inf
    for fit in results or []:
        try:
            ds = _fit_D(fit)
        except (KeyError, TypeError, ValueError):
            continue
        if ds and max(ds) > best_D:
            best, best_D = fit, max(ds)
    return best


def compact_check(check: dict, k: int | None = None) -> dict:
    """The parts of a :func:`check_score_result` dict worth storing with a result (plain
    floats and strings, so it survives saving a session)."""
    out = {"ok": bool(check["ok"]), "flags": list(check["flags"]),
           "water_difference": float(check["water_difference"]),
           "D_integral": [float(v) for v in check["D_integral"]]}
    if k is not None:
        out["k"] = int(k)
    return out


def format_check(result: dict) -> str:
    """One readable paragraph for a SCORE result carrying ``"check"`` (and, when SCORE was run
    next to the decay fit, ``"beside_fit"``); empty when there is nothing to say."""
    lines = []
    check = result.get("check")
    if check is not None:
        if check["flags"]:
            lines += ["⚠ " + f for f in check["flags"]]
        elif check["D_integral"]:
            lines.append(f"✓ Water-like D agrees with the integral fit "
                         f"({100 * check['water_difference']:.1f} % apart).")
        if not check["D_integral"]:
            lines.append("No integral decay fit to compare with: run Fit Curve for a cross-check.")
    shifts = result.get("alignment_shifts_ppm")
    if shifts is not None:
        s = np.abs(np.asarray(shifts, dtype=float))
        lines.append(f"Rows aligned before resolving (largest shift {s.max():.4f} ppm).")
    others = [c for c in result.get("beside_fit", []) if c.get("k") != result.get("n_components")]
    for c in others:
        lines.append(f"k = {c['k']}: " + ("consistent." if c["ok"] else "⚠ " + " ".join(c["flags"])))
    return "\n".join(lines)
