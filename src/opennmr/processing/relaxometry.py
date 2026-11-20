"""
Relaxometry extras (explicit user request, 2026-10-06): a three-exponential
relaxation model, integration regions found automatically from a threshold,
a choice of trapezoid or Simpson integration, and the parameter correlation
matrix of a fit.

Pure numpy/scipy, no plotting and no GUI, so both
:mod:`opennmr.processing.relaxation` (the GUI's fitter) and
:mod:`opennmr.notebook.relaxation` (the notebook's) can use it without an
import cycle. :mod:`opennmr.notebook.relaxometry` re-exports it with the
correlation-matrix figure.

Three exponentials
------------------
``triexponential_decay(t, A1, T2a, A2, T2b, A3, T2c)`` and
``triexponential_recovery(t, A1, T1a, A2, T1b, A3, T1c, C=2)``, the same
forms as the biexponential models with a third term. A sum of exponentials
is the textbook ill-conditioned fit, so two things are done that the
simpler models do without:

* **Starting point by a grid search** (:func:`multiexponential_start`):
  every combination of three time constants on a log grid spanning the
  delays is tried, with the amplitudes solved exactly by linear least
  squares for each (variable projection), and the best combination
  starts the non-linear fit. A guess from the median delay -- what the
  biexponential models use -- lands three components on top of each
  other, and the fit never separates them.
* **Components sorted by time constant** after the fit
  (:func:`sort_components`), so component a is always the fastest. A
  sum of exponentials is unchanged by relabelling, so without this the
  same data can come back with "a" and "c" swapped between two peaks or
  two runs.

Whether the data support three components at all is a separate question:
:func:`opennmr.notebook.fit_advice.advise_relaxation` compares the fit with
a two-component one (BIC) and says so.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

# ============================================================
# Three-exponential models
# ============================================================


def triexponential_decay(t, A1, T2a, A2, T2b, A3, T2c):
    """Three decaying components:
    ``A1 exp(-t/T2a) + A2 exp(-t/T2b) + A3 exp(-t/T2c)``."""
    return A1 * np.exp(-t / T2a) + A2 * np.exp(-t / T2b) + A3 * np.exp(-t / T2c)


def triexponential_recovery(t, A1, T1a, A2, T1b, A3, T1c, C=2.0):
    """Three recovering components sharing one inversion factor ``C``
    (2 = perfect inversion, 1 = saturation recovery):
    ``sum_i A_i (1 - C exp(-t/T1_i))`` -- the biexponential recovery with
    a third term."""
    return (A1 * (1.0 - C * np.exp(-t / T1a)) + A2 * (1.0 - C * np.exp(-t / T1b))
            + A3 * (1.0 - C * np.exp(-t / T1c)))


#: Minimum number of delays for a three-exponential fit: six (decay) or
#: seven (recovery) parameters, plus enough points left over to say
#: anything about the misfit.
TRIEXP_MIN_POINTS = 10


def _time_grid(t, n_grid=36):
    """Log-spaced trial time constants spanning the measured delays: from
    a third of the shortest non-zero delay to three times the longest."""
    t = np.asarray(t, dtype=float)
    pos = t[t > 0]
    if pos.size == 0:
        raise ValueError("The delays must include positive values.")
    lo = pos.min() / 3.0
    hi = 3.0 * t.max()
    if not hi > lo:
        hi = lo * 10.0
    return np.geomspace(lo, hi, int(n_grid))


def multiexponential_start(t, y, n_components=3, recovery=False, n_grid=36, weights=None,
                           n_starts=6):
    """The best starting point for a sum of ``n_components`` exponentials,
    by an exhaustive search over a log grid of time constants with the
    amplitudes solved by linear least squares at each (variable
    projection).

    Decay: ``y = sum_i A_i exp(-t/T_i)``. Recovery: ``y = M_inf - sum_i
    B_i exp(-t/T_i)``, converted to the model's ``A_i``, ``C`` as
    ``C = sum(B)/M_inf`` and ``A_i = B_i / C``.

    Combinations whose amplitudes all share the sign of the signal are
    preferred (a negative component in a decay is usually the fit using a
    spare term to bend the curve, not a population); if none does, the
    best unrestricted one is used.

    Returns ``{"T": [...] ascending, "A": [...], "C": float or None,
    "rss": float, "alternatives": [...]}`` -- ``alternatives`` being up to
    ``n_starts`` further, distinct good combinations in the same form,
    for multi-start fitting (:func:`triexp_starts`).
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.ones_like(y) if weights is None else np.asarray(weights, dtype=float)
    grid = _time_grid(t, n_grid)
    E = np.exp(-t[:, None] / grid[None, :])                 # n x G
    sw = np.sqrt(w)
    if recovery:
        basis = np.hstack([np.ones((t.size, 1)), -E])        # offset, then -exp
        offset = 1
    else:
        basis = E
        offset = 0
    Bw = basis * sw[:, None]
    yw = y * sw
    gram = Bw.T @ Bw
    rhs = Bw.T @ yw
    yy = float(yw @ yw)
    combos = np.array(list(combinations(range(grid.size), n_components)))
    cols = combos + offset
    if recovery:
        cols = np.hstack([np.zeros((cols.shape[0], 1), dtype=int), cols])
    G = gram[cols[:, :, None], cols[:, None, :]]             # K x m x m
    b = rhs[cols]                                            # K x m
    scale = np.trace(G, axis1=1, axis2=2)[:, None, None] / G.shape[1]
    ridge = 1e-12 * scale * np.eye(G.shape[1])[None]
    try:
        coef = np.linalg.solve(G + ridge, b[..., None])[..., 0]
    except np.linalg.LinAlgError:
        coef = np.array([np.linalg.lstsq(g, v, rcond=None)[0] for g, v in zip(G, b)])
    rss = yy - np.einsum("ki,ki->k", coef, b)
    amps = coef[:, offset:]
    if recovery:
        # Each B_i = C A_i must share the sign of the final level M_inf
        # (C > 0 and the components recover the same way).
        same = np.all(amps * np.sign(coef[:, :1]) > 0, axis=1)
    else:
        same = np.all(amps * (np.sign(np.sum(y)) or 1.0) > 0, axis=1)
    pick = np.flatnonzero(same) if same.any() else np.arange(rss.size)
    ranked = list(pick[np.argsort(rss[pick])])
    # A few more starts for multi-start fitting: the best combinations
    # that differ from those already taken in at least one time constant
    # by more than one grid step, then the best unrestricted one.
    chosen = []
    for k in ranked:
        if all(np.max(np.abs(combos[k] - combos[c])) > 1 for c in chosen):
            chosen.append(k)
        if len(chosen) >= n_starts:
            break
    free_best = int(np.argmin(rss))
    if free_best not in chosen:
        chosen.append(free_best)

    def describe(k):
        T = grid[combos[k]]
        a = amps[k]
        if recovery:
            m_inf = float(coef[k, 0])
            C = float(np.sum(a)) / m_inf if m_inf else 2.0
            if not np.isfinite(C) or C <= 0:
                C = 2.0
            return {"T": list(map(float, T)), "A": list(map(float, a / C)), "C": float(C),
                    "rss": float(rss[k])}
        return {"T": list(map(float, T)), "A": list(map(float, a)), "C": None,
                "rss": float(rss[k])}

    out = describe(chosen[0])
    out["alternatives"] = [describe(k) for k in chosen[1:]]
    return out


def _check_points(t):
    if np.asarray(t).size < TRIEXP_MIN_POINTS:
        raise ValueError(
            f"A three-exponential fit needs at least {TRIEXP_MIN_POINTS} delays "
            f"(it has 6-7 parameters); this series has {np.asarray(t).size}. Use a "
            "two-component or stretched model instead.")


def _vector(s, recovery):
    v = [s["A"][0], s["T"][0], s["A"][1], s["T"][1], s["A"][2], s["T"][2]]
    return v + [min(max(s["C"], 0.05), 3.95)] if recovery else v


def _median_start(t, y, recovery):
    """The biexponential models' kind of guess, extended to three: kept as
    one more start, never the only one."""
    t = np.asarray(t, dtype=float)
    M = float(np.max(np.abs(y))) or 1.0
    T = float(np.median(t[t > 0])) if np.any(t > 0) else 1.0
    v = [M / 3, T * 0.1, M / 3, T, M / 3, T * 10]
    return v + [2.0] if recovery else v


def triexp_starts(t, y, recovery=False):
    """Starting vectors for a three-exponential fit, best first: the grid
    search's best and its distinct runners-up, then a median-delay guess.
    :func:`fit_from_starts` fits from each and keeps the lowest χ²."""
    _check_points(t)
    s = multiexponential_start(t, y, 3, recovery=recovery)
    return ([_vector(s, recovery)] + [_vector(a, recovery) for a in s["alternatives"]]
            + [_median_start(t, y, recovery)])


def _p0(t, y, recovery):
    # The registry's p0 is also read to fill the GUI's starting-value
    # table and to probe a model's shape (SCORE), where too few points is
    # not an error; the fit itself checks the point count (triexp_starts).
    if np.asarray(t).size < TRIEXP_MIN_POINTS:
        return _median_start(t, y, recovery)
    try:
        return triexp_starts(t, y, recovery=recovery)[0]
    except (ValueError, np.linalg.LinAlgError):
        return _median_start(t, y, recovery)


def p0_triexp_decay(t, y):
    return _p0(t, y, recovery=False)


def bounds_triexp_decay(t, y):
    return ([-np.inf, 1e-9] * 3, [np.inf, np.inf] * 3)


def p0_triexp_recovery(t, y):
    return _p0(t, y, recovery=True)


def bounds_triexp_recovery(t, y):
    lo, hi = bounds_triexp_decay(t, y)
    return (lo + [0.0], hi + [4.0])


#: Model registry entries, merged into both RELAXATION_MODELS dicts.
#: ``components`` pairs each amplitude with its time constant, for
#: :func:`sort_components`; ``curve_fit_kwargs`` scales the parameters
#: from the Jacobian (amplitudes ~1e6 next to times ~1e-3 s otherwise
#: stall the trust-region steps).
TRIEXP_MODELS = {
    "triexponential_decay": {
        "func": triexponential_decay, "params": ["A1", "T2a", "A2", "T2b", "A3", "T2c"],
        "p0": p0_triexp_decay, "bounds": bounds_triexp_decay, "rate_param": None,
        "starts": lambda t, y: triexp_starts(t, y, recovery=False),
        "components": [("A1", "T2a"), ("A2", "T2b"), ("A3", "T2c")],
        "curve_fit_kwargs": {"x_scale": "jac"},
    },
    "triexponential_recovery": {
        "func": triexponential_recovery,
        "params": ["A1", "T1a", "A2", "T1b", "A3", "T1c", "C"],
        "p0": p0_triexp_recovery, "bounds": bounds_triexp_recovery, "rate_param": None,
        "starts": lambda t, y: triexp_starts(t, y, recovery=True),
        "components": [("A1", "T1a"), ("A2", "T1b"), ("A3", "T1c")],
        "curve_fit_kwargs": {"x_scale": "jac"},
    },
}


def candidate_starts(spec, t, y, p0_full, user_p0, free_idx):
    """The free-parameter starting vectors to try: the user's (or the
    spec's ``p0``) alone, unless the spec offers several ``starts`` and
    no start was given. Used by both relaxation fitters."""
    if user_p0 is not None or "starts" not in spec:
        return [[p0_full[i] for i in free_idx]]
    return [[full[i] for i in free_idx] for full in spec["starts"](t, y)]


def fit_from_starts(fit_once, starts, lo, hi):
    """Run ``fit_once(p0) -> (popt, pcov)`` from each start (clipped into
    the bounds) and keep the result with the lowest sum of squares,
    measured by ``fit_once``'s own third return value. Raises the last
    error if every start fails."""
    best, best_cost, last_error = None, np.inf, None
    for p0 in starts:
        p0 = np.clip(np.asarray(p0, dtype=float),
                     np.nextafter(np.asarray(lo, dtype=float), np.inf),
                     np.nextafter(np.asarray(hi, dtype=float), -np.inf))
        try:
            popt, pcov, cost = fit_once(list(p0))
        except (RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
            last_error = exc
            continue
        if cost < best_cost:
            best, best_cost = (popt, pcov), cost
    if best is None:
        raise last_error if last_error is not None else RuntimeError("No start converged.")
    return best


def sort_components(result: dict, components, descending: bool = False) -> dict:
    """Relabel a multi-exponential fit so its components run fastest
    first (``T..a < T..b < T..c``), carrying values, errors, ``popt`` and
    ``pcov`` with them. In place; returns ``result``.

    ``descending=True`` puts the *largest* value first instead (diffusion coefficients:
    component 1 is the fastest-diffusing).

    Not done when any of the components' parameters is fixed: a fixed
    value belongs to the component the user put it on."""
    if not components:
        return result
    params, errors = result["params"], result["errors"]
    free = list(result.get("free_names") or [])
    names = [n for pair in components for n in pair]
    if any(n not in free for n in names):
        return result
    times = [float(params[comp[1]]) for comp in components]
    order = list(np.argsort(-np.asarray(times) if descending else times))
    if order == list(range(len(components))):
        return result
    rename = {}
    for new_slot, old_slot in enumerate(order):
        for old_name, new_name in zip(components[old_slot], components[new_slot]):
            rename[old_name] = new_name
    new_params, new_errors = dict(params), dict(errors)
    for old, new in rename.items():
        new_params[new] = params[old]
        new_errors[new] = errors[old]
    result["params"], result["errors"] = new_params, new_errors
    # popt/pcov are in free_names order: permute so slot i now holds the
    # parameter whose new name is free[i].
    inverse = {new: old for old, new in rename.items()}
    perm = [free.index(inverse.get(name, name)) for name in free]
    if result.get("popt") is not None:
        result["popt"] = np.asarray(result["popt"])[perm]
    if result.get("pcov") is not None:
        pcov = np.asarray(result["pcov"])
        result["pcov"] = pcov[np.ix_(perm, perm)]
    if result.get("p0") is not None and len(result["p0"]) == len(result.get("param_names", [])):
        pn = list(result["param_names"])
        p0 = list(result["p0"])
        result["p0"] = [p0[pn.index(inverse.get(n, n))] for n in pn]
    result["components_sorted"] = True
    return result


# ============================================================
# Integration: trapezoid or Simpson, with exact noise propagation
# ============================================================

INTEGRATION_METHODS = ("trapezoid", "simpson")


def integration_weights(x, method: str = "trapezoid") -> np.ndarray:
    """Weights ``w`` such that ``sum(w * y)`` is the integral of ``y``
    over ``x`` (ascending) by the chosen rule. Being linear, the same
    weights give the noise exactly: ``sigma * sqrt(sum(w**2))``.

    ``"trapezoid"`` -- the composite trapezoid rule, any spacing.

    Which is better is decided by the region's edges: over a whole peak
    reaching the baseline at both ends the trapezoid rule is already all
    but exact (its Euler-Maclaurin error terms are the end derivatives),
    and Simpson's is no better; across a region edge that cuts through
    signal, Simpson's is the more accurate.

    ``"simpson"`` -- composite Simpson's rule for any spacing (each pair
    of intervals integrates the parabola through its three points); with
    an odd number of intervals the last one is integrated over the
    parabola through the last three points (Cartwright's correction, as
    :func:`scipy.integrate.simpson` does). Needs at least 3 points; with
    2 it is the trapezoid.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 2:
        raise ValueError("At least two points are needed to integrate.")
    if np.any(np.diff(x) <= 0):
        raise ValueError("x must be strictly ascending.")
    method = str(method).lower()
    if method not in INTEGRATION_METHODS:
        raise ValueError(f"Integration method must be one of {INTEGRATION_METHODS}.")
    h = np.diff(x)
    w = np.zeros(n)
    if method == "trapezoid" or n == 2:
        w[:-1] += 0.5 * h
        w[1:] += 0.5 * h
        return w
    n_int = n - 1
    pairs = n_int // 2
    for k in range(pairs):
        i = 2 * k
        h0, h1 = h[i], h[i + 1]
        s = h0 + h1
        w[i] += s / 6.0 * (2.0 - h1 / h0)
        w[i + 1] += s / 6.0 * s * s / (h0 * h1)
        w[i + 2] += s / 6.0 * (2.0 - h0 / h1)
    if n_int % 2:
        h0, h1 = h[-2], h[-1]
        w[-1] += (2 * h1 ** 2 + 3 * h0 * h1) / (6 * (h0 + h1))
        w[-2] += (h1 ** 2 + 3 * h0 * h1) / (6 * h0)
        w[-3] -= h1 ** 3 / (6 * h0 * (h0 + h1))
    return w


def integrate(x, y, method: str = "trapezoid", noise: float | None = None,
              zero_fill: float = 1.0):
    """``(integral, error)`` of ``y`` over ``x`` by ``method`` (any x
    order; sorted ascending first, so the sign follows the data, not the
    axis direction). ``error`` is ``noise * sqrt(sum(w**2))`` -- exact for
    independent Gaussian noise of standard deviation ``noise`` -- or nan
    when ``noise`` is None. A zero-filled spectrum has correlated neighbouring points, so
    the independent-noise error is too small by about ``sqrt(zero_fill)``: give the
    zero-fill factor (2-8x) to multiply it back."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    order = np.argsort(x)
    x, y = x[order], y[order]
    w = integration_weights(x, method)
    value = float(np.dot(w, y))
    err = float(noise * np.sqrt(np.sum(w ** 2)) * np.sqrt(max(float(zero_fill), 1.0))) \
        if noise is not None else float("nan")
    return value, err


# ============================================================
# Automatic integration regions from a threshold
# ============================================================

def _robust_noise(y, stride=1):
    """Noise σ from differences between points ``stride`` apart (MAD). ``stride="auto"`` widens
    the lag until the estimate stops growing: zero-filled spectra have *correlated* neighbours,
    so the point-to-point estimate (stride 1) is far too small (about 1/zero-fill)."""
    y = np.asarray(y, dtype=float)

    def at(lag):
        d = y[lag:] - y[:-lag]
        if d.size < 2:
            return 0.0
        return 1.4826 * float(np.median(np.abs(d - np.median(d)))) / np.sqrt(2.0)

    if stride == "auto":
        lag, sigma = 1, at(1)
        while lag * 2 <= max(1, y.size // 32):
            nxt = at(lag * 2)
            if nxt < 1.05 * sigma:
                break
            lag, sigma = lag * 2, nxt
        return sigma
    return at(max(int(stride), 1)) if y.size > max(int(stride), 1) else 0.0


def _baseline_corrected(x, y):
    from .fitting import estimate_baseline_robust
    b0, b1, x_ref = estimate_baseline_robust(x, y)
    return y - (b0 + b1 * (x - x_ref))


def find_integration_regions(x, y, threshold: float = 5.0, mode: str = "noise",
                             min_points: int = 3, extend_to: float = 1.0, pad_points: int = 2,
                             subtract_baseline: bool = True, noise: float | None = None,
                             noise_stride=1) -> dict:
    """Integration regions where a spectrum (or a series of them) holds
    signal, found from a threshold.

    ``y`` : one trace, or a 2D array (one trace per row, e.g. every slice
    of a pseudo-2D). For a series, signal in *any* row counts, so a peak
    that grows in or dies away along the series still gets a region; the
    regions are the same for every row, as an integral series needs.

    ``noise_stride`` : lag (in points) of the differences the noise is measured from; 1 is
    right for white noise. A zero-filled spectrum has correlated neighbours and needs its
    zero-fill factor (or ``"auto"``), otherwise σ comes out near zero and the whole axis is
    one region.

    ``mode`` / ``threshold`` : ``"noise"`` -- signal is where ``|y|``
    exceeds ``threshold`` × the noise σ (measured robustly from point-to-
    point differences, so peaks do not inflate it); ``"fraction"`` --
    where ``|y|`` exceeds ``threshold`` × the largest ``|y|`` anywhere in
    the series (e.g. 0.05) -- not each slice's own maximum, which on a
    fully decayed slice would be a share of its noise. Negative signal (an inverted slice) counts the same as
    positive.

    A region starts as a run of at least ``min_points`` consecutive
    points above the threshold (so a single noise spike is not a peak),
    is then widened on both sides while the lightly smoothed signal keeps
    its sign and stays above ``extend_to`` σ -- out to where the peak
    meets the noise, so the region holds its wings and not just its top
    -- plus ``pad_points``. Overlapping regions are merged.

    Returns a dict: ``regions`` (``[(lo, hi), ...]`` in x units,
    ascending), ``noise`` (σ, of the strongest row for a series),
    ``threshold`` (absolute), ``snr`` (each region's peak |y| over σ),
    ``mode``, and ``notes``.

    A threshold cannot see what is below it: two peaks whose wings meet
    above the noise come out as one region, and a broad weak feature
    under ``threshold`` σ is left out. For relaxation curves the region
    need only be the same on every slice -- a wing cut off at the noise
    is the same fraction of every slice's peak, so T₁/T₂ do not change,
    only the amplitude.
    """
    x = np.asarray(x, dtype=float)
    Y = np.atleast_2d(np.asarray(y, dtype=float))
    if Y.shape[-1] != x.size:
        raise ValueError("y must have one value per x point.")
    if mode not in ("noise", "fraction"):
        raise ValueError("mode must be 'noise' or 'fraction'.")
    if threshold <= 0:
        raise ValueError("The threshold must be positive.")
    order = np.argsort(x)
    xs = x[order]
    Y = Y[:, order]
    if subtract_baseline:
        Y = np.vstack([_baseline_corrected(xs, row) for row in Y])
    n = xs.size
    sigmas = np.array([noise if noise else _robust_noise(row, noise_stride) for row in Y])
    strongest = int(np.argmax(np.max(np.abs(Y), axis=1)))
    # "fraction" is of the largest signal anywhere in the series: a share
    # of each slice's own maximum would, on a fully decayed slice, be a
    # share of its noise.
    series_max = float(np.max(np.abs(Y)))
    sw = 5 if n >= 50 else 3
    kernel = np.ones(sw) / sw
    found = np.zeros(n, dtype=bool)
    abs_thr = None
    for row, sigma in zip(Y, sigmas):
        if not sigma > 0:
            continue
        thr = threshold * sigma if mode == "noise" else threshold * series_max
        if row is Y[strongest]:
            abs_thr = thr
        above = np.abs(row) > thr
        smooth = np.convolve(row, kernel, mode="same")
        floor = extend_to * sigma / np.sqrt(sw)
        edges = np.flatnonzero(np.diff(np.concatenate([[0], above.astype(int), [0]])))
        for a, b in zip(edges[0::2], edges[1::2]):
            if b - a < min_points:
                continue
            sign = np.sign(row[a:b][np.argmax(np.abs(row[a:b]))])
            lo, hi = a, b - 1
            while lo > 0 and smooth[lo - 1] * sign > floor:
                lo -= 1
            while hi < n - 1 and smooth[hi + 1] * sign > floor:
                hi += 1
            lo = max(0, lo - pad_points)
            hi = min(n - 1, hi + pad_points)
            found[lo:hi + 1] = True
    edges = np.flatnonzero(np.diff(np.concatenate([[0], found.astype(int), [0]])))
    regions, snr = [], []
    ref_sigma = float(sigmas[strongest]) if sigmas.size else 0.0
    for a, b in zip(edges[0::2], edges[1::2]):
        regions.append((float(xs[a]), float(xs[b - 1])))
        peak = float(np.max(np.abs(Y[:, a:b])))
        snr.append(peak / ref_sigma if ref_sigma > 0 else float("inf"))
    notes = []
    rule = (f"|signal| above {threshold:g} × the noise (σ = {ref_sigma:.3g})" if mode == "noise"
            else f"|signal| above {threshold:.3g} × the largest signal")
    if regions:
        notes.append(f"{len(regions)} region(s) where {rule}"
                     + (f", in any of the {Y.shape[0]} slices" if Y.shape[0] > 1 else "")
                     + ", each widened to where it meets the noise.")
    else:
        notes.append(f"No signal found: nothing has {rule}. Lower the threshold.")
    if mode == "fraction" and ref_sigma > 0 and abs_thr is not None \
            and abs_thr < 3 * ref_sigma:
        notes.append(f"That threshold is only {abs_thr / ref_sigma:.1f} σ: some regions may "
                     "be noise.")
    return {"regions": regions, "noise": ref_sigma,
            "threshold": float(abs_thr) if abs_thr is not None else float("nan"),
            "snr": snr, "mode": mode, "notes": notes}


# ============================================================
# Correlation matrix of a fit
# ============================================================

def correlation_matrix(fit: dict):
    """``(names, R)``: the free parameters of a fit (``free_names``) and
    their correlation matrix ``R_ij = cov_ij / (σ_i σ_j)`` from its
    covariance ``pcov``. Fixed parameters have no row: they did not vary.

    A parameter with no determinable variance (zero or non-finite) gets
    nan in its row and column, rather than a misleading 0."""
    names = list(fit.get("free_names") or [])
    pcov = fit.get("pcov")
    if pcov is None or not names:
        raise ValueError("This fit has no covariance matrix (no free parameters, or it "
                         "was not a least-squares fit).")
    pcov = np.asarray(pcov, dtype=float)
    if pcov.shape != (len(names), len(names)):
        raise ValueError("The covariance matrix does not match the fit's free parameters.")
    sd = np.sqrt(np.diag(pcov))
    ok = np.isfinite(sd) & (sd > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        R = pcov / np.outer(sd, sd)
    R[~ok, :] = np.nan
    R[:, ~ok] = np.nan
    R = np.clip(R, -1.0, 1.0)
    np.fill_diagonal(R, np.where(ok, 1.0, np.nan))
    return names, R


def strong_correlations(fit: dict, limit: float = 0.95):
    """``[(name_i, name_j, r), ...]`` with ``|r| >= limit``, strongest
    first."""
    names, R = correlation_matrix(fit)
    out = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            r = R[i, j]
            if np.isfinite(r) and abs(r) >= limit:
                out.append((names[i], names[j], float(r)))
    return sorted(out, key=lambda item: -abs(item[2]))
