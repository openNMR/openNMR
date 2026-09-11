"""
Inverse Laplace transform (ILT), and the DOSY map built from it.

A PFG series measures :math:`I(b) = \\int P(D)\\,e^{-bD}\\,\\mathrm{d}D`.
Fitting it with :mod:`opennmr.processing.relaxation`'s Stejskal-Tanner
models asks "how big is D, assuming there are exactly one or two
values of it". The ILT asks the other question -- "what does the
*distribution* of D look like" -- and answers it without being told
how many components to expect. That is what a DOSY plot displays: one
diffusion distribution per chemical shift, laid out as a 2D map of
shift against D.

**This problem is ill-posed, and that is not a detail.** Distinct
distributions produce almost identical decays; a modest amount of
noise admits a wide family of solutions that all fit the data within
error. Every ILT therefore has to impose something beyond the data,
and what it imposes shows up directly in the answer:

* **Non-negativity** (a population cannot be negative) -- the single
  most stabilising constraint, applied here always, via
  :func:`scipy.optimize.nnls`.
* **Regularisation** -- a penalty on roughness or magnitude that picks
  the smoothest distribution still consistent with the data. Tikhonov
  (ridge) here, weight ``alpha``.

Turning ``alpha`` down makes peaks sharper and eventually splits one
population into a spray of spurious spikes; turning it up merges
genuinely distinct populations into one broad hump. There is no value
that is simply correct, which is why :func:`l_curve_alpha` exists to
choose one defensibly and why every result carries the ``alpha`` it
used. **A sharp ILT peak is not evidence of a sharp distribution** --
treat peak *positions* as the result and peak *widths* as an artefact
of the regularisation until shown otherwise.

Where a discrete fit is applicable at all -- one or two known
components -- prefer it: :func:`opennmr.processing.relaxation.fit_relaxation`
gives a D with a meaningful uncertainty, which an ILT does not.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import nnls

#: Default number of points on the log-spaced D grid a distribution is
#: reconstructed on. Enough to resolve peaks a factor of ~1.1 apart
#: over four decades, and small enough that the NNLS solve stays
#: instant per column (which matters -- a DOSY map runs one per shift).
DEFAULT_N_GRID = 128

#: Default D range (m^2/s) when none is given: from a large aggregate
#: or restricted-diffusion regime up past a small molecule in a
#: non-viscous solvent. Deliberately wide, since a grid that clips the
#: true D silently piles the population up on the end point instead of
#: reporting nothing.
DEFAULT_D_RANGE = (1e-13, 1e-8)


def log_grid(low: float = DEFAULT_D_RANGE[0], high: float = DEFAULT_D_RANGE[1],
             n: int = DEFAULT_N_GRID) -> np.ndarray:
    """A log-spaced grid of diffusion coefficients. Log, not linear:
    diffusion coefficients of interest span five decades, and a linear
    grid fine enough to resolve 1e-12 would need ten thousand points
    to reach 1e-8, almost all of them wasted."""
    if low <= 0 or high <= 0 or high <= low:
        raise ValueError(f"Need 0 < low < high for a log grid, got low={low}, high={high}.")
    return np.logspace(np.log10(low), np.log10(high), int(n))


def kernel(b, d_grid) -> np.ndarray:
    """The exponential kernel ``K[i, j] = exp(-b_i * D_j)`` mapping a
    distribution over ``d_grid`` to the decay sampled at ``b``. The
    whole ill-posedness lives in this matrix: its singular values fall
    off exponentially, so inverting it directly amplifies noise without
    limit. That is why everything below solves a *penalised*,
    *constrained* problem rather than a linear system."""
    b = np.asarray(b, dtype=float)
    d_grid = np.asarray(d_grid, dtype=float)
    return np.exp(-np.outer(b, d_grid))


def ilt(b, y, d_grid=None, alpha: float = 1e-2, n_grid: int = DEFAULT_N_GRID,
        d_range=DEFAULT_D_RANGE, normalise: bool = True) -> dict:
    """Recover a non-negative distribution over diffusion coefficients
    from one decay ``y(b)``.

    Solves ``min ||K s - y||^2 + alpha^2 ||s||^2`` subject to
    ``s >= 0``, by stacking the penalty onto the system and handing the
    result to NNLS -- the standard augmented-matrix trick, and the
    reason non-negativity and regularisation can both be imposed by one
    solver rather than needing a bespoke optimiser.

    Parameters
    ----------
    b : array
        Diffusion weighting (s/m^2), one per point.
    y : array
        Signal at each ``b``. Normalised internally when
        ``normalise``, so the absolute intensity scale is irrelevant.
    d_grid : array, optional
        Diffusion coefficients to reconstruct on. Defaults to
        :func:`log_grid` over ``d_range`` with ``n_grid`` points.
    alpha : float
        Tikhonov weight. See the module docstring -- this is a choice,
        not a constant, and :func:`l_curve_alpha` picks one from the
        data.
    normalise : bool
        Scale ``y`` to a maximum of 1 first, so ``alpha`` means the
        same thing regardless of the data's own intensity units. On by
        default precisely so that a single ``alpha`` is transferable
        between datasets.

    Returns
    -------
    dict with ``D`` (the grid), ``P`` (the distribution, summing to 1
    when ``normalise``), ``y_fit``, ``residual_norm``,
    ``solution_norm``, ``alpha``, and the summary statistics
    :func:`distribution_moments` computes.
    """
    b = np.asarray(b, dtype=float)
    y = np.asarray(y, dtype=float)
    if b.size != y.size:
        raise ValueError(f"Got {b.size} b-values against {y.size} intensities.")
    if b.size < 3:
        raise ValueError("An ILT needs at least 3 points.")
    if alpha < 0:
        raise ValueError(f"alpha must be >= 0, got {alpha}.")

    d_grid = log_grid(*d_range, n=n_grid) if d_grid is None else np.asarray(d_grid, dtype=float)

    scale = float(np.nanmax(np.abs(y))) if normalise else 1.0
    scale = scale if scale > 0 else 1.0
    y_scaled = y / scale

    k = kernel(b, d_grid)
    n_d = d_grid.size
    # Augmented system: the bottom block contributes alpha^2 ||s||^2 to
    # the least-squares objective, so one plain NNLS solve delivers the
    # regularised, non-negative answer.
    k_aug = np.vstack([k, np.sqrt(alpha) * np.eye(n_d)])
    y_aug = np.concatenate([y_scaled, np.zeros(n_d)])

    solution, _residual = nnls(k_aug, y_aug)

    y_fit_scaled = k @ solution
    residual = y_scaled - y_fit_scaled
    total = float(np.sum(solution))
    distribution = solution / total if (normalise and total > 0) else solution

    result = {
        "D": d_grid, "P": distribution, "amplitude": total * scale,
        "b": b, "y": y_scaled, "y_fit": y_fit_scaled,
        "residual_norm": float(np.linalg.norm(residual)),
        "solution_norm": float(np.linalg.norm(solution)),
        "alpha": float(alpha), "scale": scale,
    }
    result.update(distribution_moments(d_grid, distribution))
    return result


def distribution_moments(d_grid, distribution) -> dict:
    """Summary numbers for a diffusion distribution, in the form these
    are actually quoted.

    ``D_mean`` is the population-weighted **geometric** mean, not the
    arithmetic one: D is a log-scale quantity, the grid is log-spaced,
    and an arithmetic mean over a distribution spanning decades is
    dominated by the largest value present rather than describing the
    bulk. ``D_mode`` is the grid point carrying most weight, and
    ``log10_width`` is the weighted standard deviation in decades,
    which is the honest way to state the spread of something that
    ranges over orders of magnitude.
    """
    d_grid = np.asarray(d_grid, dtype=float)
    distribution = np.asarray(distribution, dtype=float)
    total = float(np.sum(distribution))
    if total <= 0:
        return {"D_mean": float("nan"), "D_mode": float("nan"), "log10_width": float("nan")}
    weights = distribution / total
    log_d = np.log10(d_grid)
    mean_log = float(np.sum(weights * log_d))
    variance_log = float(np.sum(weights * (log_d - mean_log) ** 2))
    return {
        "D_mean": float(10.0 ** mean_log),
        "D_mode": float(d_grid[int(np.argmax(distribution))]),
        "log10_width": float(np.sqrt(max(variance_log, 0.0))),
    }


def l_curve_alpha(b, y, alphas=None, d_grid=None, n_grid: int = DEFAULT_N_GRID,
                  d_range=DEFAULT_D_RANGE) -> dict:
    """Choose the Tikhonov weight by the L-curve criterion: solve at a
    range of ``alpha``, plot residual norm against solution norm on
    log axes, and take the corner.

    The corner is where the two failure modes trade off. Left of it,
    reducing ``alpha`` buys almost no further reduction in residual
    while the solution norm climbs steeply -- that is the ILT fitting
    noise with spurious spikes. Right of it, the solution is smooth but
    the residual rises sharply -- real structure is being smoothed
    away. Located here by maximum curvature of the log-log curve,
    computed from finite differences.

    This makes the choice **defensible and reproducible**, not
    correct: the corner can be ill-defined when the data are very
    noisy or very clean, and the result still carries its own
    ``alpha`` so the reader can see what was used and vary it.

    Returns
    -------
    dict with ``alpha`` (the chosen value), ``alphas``,
    ``residual_norms``, ``solution_norms``, ``curvature`` and
    ``index`` -- everything needed to draw the L-curve itself and mark
    the chosen point on it.
    """
    alphas = np.logspace(-6, 1, 25) if alphas is None else np.asarray(alphas, dtype=float)
    d_grid = log_grid(*d_range, n=n_grid) if d_grid is None else np.asarray(d_grid, dtype=float)

    residual_norms, solution_norms = [], []
    for alpha in alphas:
        result = ilt(b, y, d_grid=d_grid, alpha=float(alpha), normalise=True)
        # A floor, not the raw value: a perfectly-fit or perfectly-zero
        # solution gives a zero norm, and log(0) would poison the whole
        # curvature calculation for every other point.
        residual_norms.append(max(result["residual_norm"], 1e-300))
        solution_norms.append(max(result["solution_norm"], 1e-300))

    log_rho = np.log(np.asarray(residual_norms))
    log_eta = np.log(np.asarray(solution_norms))

    if alphas.size < 3:
        index = int(np.argmin(log_rho + log_eta))
        curvature = np.zeros_like(log_rho)
    else:
        # Menger-style discrete curvature via first and second
        # differences with respect to the (log) alpha parameter.
        t = np.log(alphas)
        d_rho, d_eta = np.gradient(log_rho, t), np.gradient(log_eta, t)
        dd_rho, dd_eta = np.gradient(d_rho, t), np.gradient(d_eta, t)
        denominator = (d_rho ** 2 + d_eta ** 2) ** 1.5
        with np.errstate(divide="ignore", invalid="ignore"):
            curvature = np.where(denominator > 0,
                                 (d_rho * dd_eta - d_eta * dd_rho) / denominator, 0.0)
        curvature = np.nan_to_num(curvature)
        index = int(np.argmax(np.abs(curvature)))

    return {
        "alpha": float(alphas[index]), "alphas": alphas, "index": index,
        "residual_norms": np.asarray(residual_norms), "solution_norms": np.asarray(solution_norms),
        "curvature": curvature,
    }


def dosy_map(b, intensities, ppm, d_grid=None, alpha: float = 1e-2, n_grid: int = DEFAULT_N_GRID,
             d_range=DEFAULT_D_RANGE, threshold: float = 0.02, auto_alpha: bool = False) -> dict:
    """Build a DOSY map: one ILT per chemical shift, assembled into a
    ``(len(D), len(ppm))`` array of shift against diffusion
    coefficient.

    Parameters
    ----------
    b : array, shape (n_gradients,)
        Diffusion weighting per gradient step.
    intensities : array, shape (n_gradients, n_points)
        The PFG series itself -- one row per gradient step, each row a
        spectrum. This is exactly the shape a pseudo-2D's ``data``
        already has.
    ppm : array, shape (n_points,)
        Chemical shift of each column.
    threshold : float
        Columns whose first-row intensity is below this fraction of
        the spectrum's own maximum are skipped and left as zero.
        **This is the setting that decides whether a DOSY looks
        sensible**: an ILT of a noise-only column returns whatever the
        regularisation likes best, which paints a band of meaningless
        "population" clean across the map at some arbitrary D. The
        default of 2% keeps real but weak peaks while excluding
        baseline.
    auto_alpha : bool
        Choose ``alpha`` once, by :func:`l_curve_alpha` on the most
        intense column, and use it for every column. One value for the
        whole map, not one per column: a per-column alpha makes
        neighbouring shifts incomparable, since peak width in the D
        direction would then vary with the regularisation rather than
        with the sample.

    Returns
    -------
    dict with ``Z`` (the map, ``(n_D, n_ppm)``), ``D``, ``ppm``,
    ``projection`` (the b=0 row, i.e. the ordinary spectrum, for the
    top strip of a DOSY plot), ``alpha``, ``columns`` (which column
    indices were actually inverted) and ``per_column`` moments.
    """
    b = np.asarray(b, dtype=float)
    intensities = np.asarray(intensities, dtype=float)
    ppm = np.asarray(ppm, dtype=float)
    if intensities.ndim != 2:
        raise ValueError(f"intensities must be 2-D (gradient steps x points), got shape {intensities.shape}.")
    if intensities.shape[0] != b.size:
        raise ValueError(
            f"{intensities.shape[0]} gradient rows against {b.size} b-values -- "
            f"one b per row is required."
        )
    if intensities.shape[1] != ppm.size:
        raise ValueError(f"{intensities.shape[1]} points per row against {ppm.size} shifts.")

    d_grid = log_grid(*d_range, n=n_grid) if d_grid is None else np.asarray(d_grid, dtype=float)

    order = np.argsort(b)
    b, intensities = b[order], intensities[order]

    reference = intensities[0]
    peak = float(np.nanmax(np.abs(reference)))
    if not peak:
        raise ValueError("The first gradient row is entirely zero -- nothing to invert.")
    selected = np.flatnonzero(np.abs(reference) >= threshold * peak)

    if auto_alpha and selected.size:
        strongest = int(selected[int(np.argmax(np.abs(reference[selected])))])
        alpha = l_curve_alpha(b, intensities[:, strongest], d_grid=d_grid)["alpha"]

    z = np.zeros((d_grid.size, ppm.size), dtype=float)
    per_column = {}
    for column in selected:
        decay = intensities[:, column]
        if not np.all(np.isfinite(decay)) or np.nanmax(np.abs(decay)) <= 0:
            continue
        result = ilt(b, decay, d_grid=d_grid, alpha=alpha, normalise=True)
        # Weighted by the column's own intensity, so the map reads as a
        # spectrum smeared along D rather than every peak, however
        # small, being drawn at the same height -- which is what makes
        # a DOSY legible as a 2D contour at all.
        z[:, column] = result["P"] * abs(reference[column])
        per_column[int(column)] = {
            "D_mean": result["D_mean"], "D_mode": result["D_mode"],
            "log10_width": result["log10_width"], "ppm": float(ppm[column]),
        }

    return {
        "Z": z, "D": d_grid, "ppm": ppm, "projection": reference,
        "alpha": float(alpha), "columns": selected, "per_column": per_column,
        "threshold": float(threshold), "b": b,
    }
