"""Tikhonov regularisation of relaxation and diffusion data -- distributions
of T1, T2 or D (1D), and T1-T2, T2-T2 and D-T2 correlation maps (2D).

A Python port of the MATLAB scripts in ``MATLAB_RELAX_ALGORITHM``
(``Tikhonov_CPMG.m``, ``Tikhonov_T1_Inversion_Recovery.m``,
``Tikhonov_T1_Saturation_Recovery.m``, ``Tikhonov_Diffusion.m``,
``Tikhonov_Diffusion_PGSE.m``, ``Tikhonov_T1T2_IR.m``, ``Tikhonov_T1T2_SR.m``,
``Tikhonov_T2T2.m``, ``Tikhonov_DT2.m``) and the routines they call:

* ``csvd`` and ``tikhonov`` -- P. C. Hansen's *Regularization Tools*
  (compact SVD; standard-form Tikhonov with a prior ``x_0``);
* ``errestik`` with ``etanutik`` -- the regularisation parameter chosen by
  minimising an *error estimate*, C. Brezinski, G. Rodriguez and S. Seatzu,
  "Error estimates for the regularization of least squares problems",
  Numer. Algorithms 51 (2009) 61-76, eq. 3.14 (``eta_nu``, nu = 3), with
  the bisection of its section 3.2;
* ``nnls10`` -- non-negative least squares (Lawson & Hanson), here
  :func:`scipy.optimize.nnls`, kept as the unregularised comparison.

What every script does, and so what :func:`regularise` does:

1. The kernel ``K`` on a log-spaced grid of T (or D), plus one constant
   column for an offset (``A = [K, 1]``).
2. The compact SVD of ``A``.
3. ``lambda`` from the error estimate (``errestik(..., estype=3,
   opts=[1e-6, 2000, 1])``: bisection to a 1e-6 width in log10 lambda, at
   most 2000 solutions, the rectangular estimate forced).
4. Then ``n_iter`` times: ``x = |x|`` and ``x = tikhonov(U, s, V, y, lambda,
   x)`` -- Tikhonov with the last solution's magnitude as its prior, which
   pulls the distribution towards non-negative values. It does not make
   it non-negative: what is left negative is reported (``negative_share``).

The alternative ``solver="nonnegative"`` solves the same functional with
``x >= 0`` imposed exactly (NNLS on ``[A; lambda I]``), at the same lambda.

**This problem is ill-posed.** Many distributions fit the data equally
well; lambda decides which one is shown. Treat peak *positions* (and log
means) as the result and peak *widths* as largely set by lambda. Where a
model with one to three components fits, Dynamics ▸ Fit Curve gives
values with meaningful uncertainties; this does not.

Conventions: delays in seconds, b in s/m^2, D in m^2/s throughout.
"""

from __future__ import annotations

import numpy as np

# ----------------------------------------------------------------------
# Kernels
# ----------------------------------------------------------------------

#: key -> (label, axis name, the kernel K(x, grid)). The MATLAB scripts'
#: kernels, one each.
KERNELS = {
    "t2": ("T₂ decay (CPMG): exp(−t/T₂)", "T₂",
           lambda x, g: np.exp(-np.outer(x, 1.0 / g))),
    "t1_ir": ("T₁ inversion recovery: 1 − 2 exp(−t/T₁)", "T₁",
              lambda x, g: 1.0 - 2.0 * np.exp(-np.outer(x, 1.0 / g))),
    "t1_sr": ("T₁ saturation recovery: 1 − exp(−t/T₁)", "T₁",
              lambda x, g: 1.0 - np.exp(-np.outer(x, 1.0 / g))),
    "diffusion": ("Diffusion: exp(−b D)", "D",
                  lambda x, g: np.exp(-np.outer(x, g))),
}

#: Default grid ranges, in seconds (T) or m^2/s (D).
DEFAULT_RANGES = {"t2": (1e-5, 10.0), "t1_ir": (1e-4, 100.0), "t1_sr": (1e-4, 100.0),
                  "diffusion": (1e-13, 1e-8)}


def kernel_matrix(kernel: str, x, grid) -> np.ndarray:
    """``K[i, j]`` for the measured ``x`` (delays in s, or b in s/m^2) and
    the distribution's ``grid`` (T in s, or D in m^2/s)."""
    if kernel not in KERNELS:
        raise ValueError(f"Unknown kernel {kernel!r}; choose one of {sorted(KERNELS)}.")
    return KERNELS[kernel][2](np.asarray(x, dtype=float), np.asarray(grid, dtype=float))


def log_grid(low: float, high: float, n: int) -> np.ndarray:
    """The MATLAB scripts' ``logspace(sta, end, Npnts)``, given the end
    values themselves rather than their powers of ten."""
    low, high = float(low), float(high)
    if not (low > 0 and high > low):
        raise ValueError(f"Need 0 < low < high for the grid, got {low:g} and {high:g}.")
    if int(n) < 2:
        raise ValueError("The grid needs at least two points.")
    return np.logspace(np.log10(low), np.log10(high), int(n))


def suggest_range(kernel: str, x) -> tuple:
    """A grid range reaching past what the measured ``x`` can resolve:
    for T, from a third of the shortest non-zero delay to three times the
    longest; for D, from 1/(3 b_max) to 3/b_min (b_min the smallest
    non-zero b). Population outside the range would pile up at its ends,
    so err wide."""
    x = np.abs(np.asarray(x, dtype=float))
    positive = x[x > 0]
    if positive.size == 0:
        return DEFAULT_RANGES[kernel]
    lo, hi = float(positive.min()), float(positive.max())
    if kernel == "diffusion":
        return (1.0 / (3.0 * hi), 3.0 / lo)
    return (lo / 3.0, 3.0 * hi)


def static_gradient_b(tau, gradient, gamma=None, nucleus="1H") -> np.ndarray:
    """b (s/m^2) of a spin echo in a static field gradient (``Tikhonov_Diffusion.m``,
    ``Tikhonov_DT2.m``): ``b = gamma^2 G^2 (2 tau)^3 / 12``, with ``tau`` the
    half echo time (s), ``gradient`` in T/m and ``gamma`` in rad s^-1 T^-1
    (looked up from ``nucleus`` when not given).

    The scripts wrote ``(1/12) 7.16e-2 G^2 (2 tau)^3`` -- 7.16e16 being 1H's
    gamma^2 -- in their own mixed units; here everything is SI."""
    if gamma is None:
        from ..processing.diffusion import gamma_for
        gamma = 2.0 * np.pi * gamma_for(nucleus)          # Hz/T -> rad s^-1 T^-1
    tau = np.asarray(tau, dtype=float)
    return (float(gamma) ** 2) * float(gradient) ** 2 * (2.0 * tau) ** 3 / 12.0


# ----------------------------------------------------------------------
# Regularization Tools (Hansen) and errestools (Rodriguez)
# ----------------------------------------------------------------------

def csvd(A):
    """Compact SVD ``A = U diag(s) V^T`` (Hansen's ``csvd``): ``U`` m x p,
    ``s`` p, ``V`` n x p, p = min(m, n)."""
    U, s, Vt = np.linalg.svd(np.asarray(A, dtype=float), full_matrices=False)
    return U, s, Vt.T


def tikhonov(U, s, V, b, lam, x_0=None):
    """Hansen's ``tikhonov`` (standard form): the minimiser of
    ``||A x - b||^2 + lam^2 ||x - x_0||^2`` over the row space of ``A``,
    from its compact SVD. Returns ``(x, rho, eta)`` -- the solution, the
    residual norm and the solution norm. ``lam`` may be an array; then
    ``x`` has one column per value.

    As in the original, the part of ``x_0`` outside the row space of ``A``
    is not added back (it matters only when there are fewer data points
    than grid points), so a MATLAB result is reproduced exactly."""
    b = np.asarray(b, dtype=float)
    lams = np.atleast_1d(np.asarray(lam, dtype=float))
    if np.min(lams) < 0:
        raise ValueError("Illegal regularisation parameter lambda (negative).")
    p = s.size
    beta = U[:, :p].T @ b
    zeta = s * beta
    omega = None if x_0 is None else V.T @ np.asarray(x_0, dtype=float)
    X = np.zeros((V.shape[0], lams.size))
    rho = np.zeros(lams.size)
    eta = np.zeros(lams.size)
    for i, l2 in enumerate(lams ** 2):
        denom = s ** 2 + l2
        if omega is None:
            X[:, i] = V[:, :p] @ (zeta / denom)
            rho[i] = l2 * np.linalg.norm(beta / denom)
        else:
            X[:, i] = V[:, :p] @ ((zeta + l2 * omega) / denom)
            rho[i] = l2 * np.linalg.norm((beta - s * omega) / denom)
        eta[i] = np.linalg.norm(X[:, i])
    if U.shape[0] > p:
        rho = np.sqrt(rho ** 2 + np.linalg.norm(b - U @ beta) ** 2)
    if np.ndim(lam) == 0:
        return X[:, 0], float(rho[0]), float(eta[0])
    return X, rho, eta


def etanutik(norm_r, norm_Ex, norm_AEx, lam, nu=3.0):
    """Rodriguez's ``etanutik``: the error estimate eta_nu of eq. 3.14 of
    Brezinski, Rodriguez & Seatzu (2009), for Tikhonov solutions of a
    rectangular system."""
    norm_r, norm_Ex, norm_AEx, lam = (np.asarray(v, dtype=float)
                                      for v in (norm_r, norm_Ex, norm_AEx, lam))
    with np.errstate(divide="ignore", invalid="ignore"):
        return (norm_r ** (nu - 1) * norm_Ex ** (5 - 2 * nu) * norm_AEx ** (nu - 3)
                / lam ** (2 * nu - 4))


def errestik(A, b, U, s, V, lambda_range=(1e-30, 1e2), estype=3.0, tau=1e-6, kmax=2000):
    """Rodriguez's ``errestik`` through the SVD, with the identity as the
    regularisation matrix and the rectangular estimate (``etanutik``) --
    the options every MATLAB script passed (``[1e-6, 2000, 1]``).

    Five log-spaced lambdas over ``lambda_range``, then bisection either
    side of the current minimum until the bracket is narrower than
    ``tau`` (in log10 lambda) or ``kmax`` solutions have been computed.

    Returns ``{"lambda", "x", "lambdas", "errest", "evaluations",
    "hit_kmax"}``; ``lambdas``/``errest`` are every value scanned, in
    descending lambda."""
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)

    def evaluate(ll):
        x, _rho, _eta = tikhonov(U, s, V, b, 10.0 ** ll)
        r = np.linalg.norm(b - A @ x)
        phi = np.linalg.norm(x)              # ||H'H x||, H = I
        psi = np.linalg.norm(A @ x)          # ||A H'H x||
        return x, float(etanutik(r, phi, psi, 10.0 ** ll, estype))

    lo, hi = sorted(float(v) for v in lambda_range)
    if lo <= 0:
        raise ValueError("lambda_range must be positive.")
    llam = list(np.linspace(np.log10(hi), np.log10(lo), 5))
    X, err = [], []
    for ll in llam:
        x, e = evaluate(ll)
        X.append(x)
        err.append(e)
    nlambda = len(llam)
    index = int(np.nanargmin(err))
    width = abs(llam[0] - llam[1])
    while width > tau and nlambda < kmax:
        if index > 0:
            ll1 = 0.5 * (llam[index] + llam[index - 1])
            x1, e1 = evaluate(ll1)
            llam.insert(index, ll1)
            X.insert(index, x1)
            err.insert(index, e1)
            nlambda += 1
            index += 1
            width = min(width, abs(llam[index] - llam[index - 1]))
        if index < nlambda - 1:
            ll2 = 0.5 * (llam[index] + llam[index + 1])
            x2, e2 = evaluate(ll2)
            llam.insert(index + 1, ll2)
            X.insert(index + 1, x2)
            err.insert(index + 1, e2)
            nlambda += 1
            width = min(width, abs(llam[index] - llam[index + 1]))
        index = int(np.nanargmin(err))
    return {"lambda": float(10.0 ** llam[index]), "x": X[index],
            "lambdas": 10.0 ** np.asarray(llam), "errest": np.asarray(err),
            "evaluations": nlambda, "hit_kmax": nlambda >= kmax, "s_max": float(s[0])}


def nonnegative_tikhonov(A, b, lam, free_last=False):
    """``min ||A x - b||^2 + lam^2 ||x||^2`` subject to ``x >= 0`` (NNLS on
    the stacked system). ``free_last``: the last column (an offset) may
    take either sign -- it is split into a positive and a negative part."""
    from scipy.optimize import nnls
    A = np.asarray(A, dtype=float)
    if free_last:
        A = np.hstack([A, -A[:, -1:]])
    n = A.shape[1]
    big = np.vstack([A, float(lam) * np.eye(n)])
    rhs = np.concatenate([np.asarray(b, dtype=float), np.zeros(n)])
    x, _ = nnls(big, rhs, maxiter=50 * n)
    if free_last:
        x = np.concatenate([x[:-2], [x[-2] - x[-1]]])
    return x


def l_curve_nonnegative(A, b, lambdas=None, n=25, free_last=False):
    """lambda for :func:`nonnegative_tikhonov` at the corner of the L-curve
    (log residual norm against log solution norm, maximum curvature).

    The error estimate of :func:`errestik` is derived for *linear* Tikhonov
    solutions and is not used for the constrained one: at its lambda the
    non-negative solution is over-smoothed (it has no |x| iterations to
    relax it). Returns ``{"lambda", "lambdas", "residual_norms",
    "solution_norms", "index"}``."""
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    if lambdas is None:
        smax = float(np.linalg.norm(A, 2))
        lambdas = np.logspace(np.log10(smax) - 6, np.log10(smax), int(n))
    lambdas = np.asarray(lambdas, dtype=float)
    rho, eta = [], []
    for lam in lambdas:
        x = nonnegative_tikhonov(A, b, lam, free_last=free_last)
        rho.append(max(np.linalg.norm(A @ x - b), 1e-300))
        eta.append(max(np.linalg.norm(x), 1e-300))
    lr, le, t = np.log(rho), np.log(eta), np.log(lambdas)
    if lambdas.size < 3:
        index = int(np.argmin(lr + le))
    else:
        d_r, d_e = np.gradient(lr, t), np.gradient(le, t)
        dd_r, dd_e = np.gradient(d_r, t), np.gradient(d_e, t)
        denom = (d_r ** 2 + d_e ** 2) ** 1.5
        with np.errstate(divide="ignore", invalid="ignore"):
            curvature = np.where(denom > 0, (d_r * dd_e - d_e * dd_r) / denom, 0.0)
        curvature = np.nan_to_num(curvature)
        # Interior points only: the end points' one-sided differences are
        # not a corner.
        index = 1 + int(np.argmax(np.abs(curvature[1:-1])))
    return {"lambda": float(lambdas[index]), "lambdas": lambdas,
            "residual_norms": np.asarray(rho), "solution_norms": np.asarray(eta),
            "index": index}


# ----------------------------------------------------------------------
# The solve shared by 1D and 2D
# ----------------------------------------------------------------------

SOLVERS = {
    "matlab": "MATLAB scheme (|x|-prior iterations)",
    "nonnegative": "Non-negative Tikhonov (x ≥ 0)",
}


def _solve(A, y, lam=None, lambda_range=(1e-30, 1e2), n_iter=100, solver="matlab",
           tau=1e-6, kmax=2000, offset=True):
    if solver not in SOLVERS:
        raise ValueError(f"Unknown solver {solver!r}; choose one of {sorted(SOLVERS)}.")
    if not np.all(np.isfinite(y)):
        raise ValueError("The data contain NaN or infinite values.")
    if solver == "nonnegative":
        scan = None
        if lam is None:
            curve = l_curve_nonnegative(A, y, free_last=offset)
            lam = curve["lambda"]
            scan = {"lambdas": curve["lambdas"], "errest": None, "curve": curve,
                    "hit_kmax": False, "method": "l_curve"}
        lam = float(lam)
        if lam <= 0:
            raise ValueError("lambda must be positive.")
        return nonnegative_tikhonov(A, y, lam, free_last=offset), lam, scan
    U, s, V = csvd(A)
    scan = None
    if lam is None:
        scan = errestik(A, y, U, s, V, lambda_range=lambda_range, tau=tau, kmax=kmax)
        scan["method"] = "error_estimate"
        lam = scan["lambda"]
        x = scan["x"]
    else:
        lam = float(lam)
        if lam <= 0:
            raise ValueError("lambda must be positive.")
        x, _r, _e = tikhonov(U, s, V, y, lam)
    for _ in range(int(n_iter)):
        x, _r, _e = tikhonov(U, s, V, y, lam, np.abs(x))
    return x, float(lam), scan


def _cumulative(a):
    return np.cumsum(a)


def log_mean(grid, amplitudes) -> float:
    """``exp(sum a_i ln T_i / sum a_i)`` -- the scripts' "log mean" -- over
    the non-negative part of the distribution (a negative population has
    no meaning as a weight). NaN when nothing is positive."""
    a = np.clip(np.asarray(amplitudes, dtype=float), 0.0, None)
    total = a.sum()
    if total <= 0:
        return float("nan")
    return float(np.exp(np.sum(a * np.log(np.asarray(grid, dtype=float))) / total))


def distribution_peaks(grid, amplitudes, min_fraction=0.05) -> list:
    """The distribution's peaks: each local maximum above ``min_fraction`` of
    the largest, with its own log mean and share of the (non-negative)
    signal, split at the minima between peaks. Largest share first."""
    grid = np.asarray(grid, dtype=float)
    a = np.clip(np.asarray(amplitudes, dtype=float), 0.0, None)
    if a.size == 0 or a.max() <= 0:
        return []
    top = a.max()
    maxima = [i for i in range(a.size)
              if a[i] >= min_fraction * top
              and (i == 0 or a[i] > a[i - 1]) and (i == a.size - 1 or a[i] >= a[i + 1])]
    if not maxima:
        return []
    bounds = [0]
    for left, right in zip(maxima[:-1], maxima[1:]):
        bounds.append(left + int(np.argmin(a[left:right + 1])))
    bounds.append(a.size)
    total = a.sum()
    peaks = []
    for k, i in enumerate(maxima):
        lo, hi = bounds[k], bounds[k + 1]
        seg = a[lo:hi]
        share = seg.sum() / total
        peaks.append({"position": float(grid[i]), "log_mean": log_mean(grid[lo:hi], seg),
                      "share": float(share), "range": (float(grid[lo]), float(grid[hi - 1]))})
    peaks.sort(key=lambda p: -p["share"])
    return peaks


def regularise(x, y, kernel="t2", grid_range=None, n_grid=100, baseline=0.0,
               lam=None, lambda_range=(1e-30, 1e2), n_iter=100, solver="matlab",
               offset=True, nnls_comparison=True, label=None):
    """A distribution of T1, T2 or D from one decay/recovery/attenuation.

    Parameters
    ----------
    x : delays (s) or b-values (s/m^2).
    y : the measured intensities.
    kernel : 't2', 't1_ir', 't1_sr' or 'diffusion' (:data:`KERNELS`).
    grid_range : (low, high) of the grid, in s or m^2/s
        (default :data:`DEFAULT_RANGES`). Choose it to reach well past the
        shortest and longest values in the sample: population beyond the
        grid piles up at its ends.
    n_grid : grid points (the scripts' "Number of Inversion Points").
    baseline : subtracted from ``y`` first (the scripts' "Manual Baseline").
    lam : the regularisation parameter; None chooses it by the error
        estimate (:func:`errestik`).
    n_iter : iterations of the |x|-prior refinement (``solver='matlab'``).
    solver : 'matlab' or 'nonnegative' (:data:`SOLVERS`).
    offset : fit a constant offset alongside (the scripts' column of ones).
    nnls_comparison : also solve without regularisation (non-negative least
        squares, the scripts' ``nnls10``) for comparison.

    Returns a dict: ``grid``, ``amplitudes``, ``cumulative`` (the scripts'
    "porosity"), ``offset``, ``y_fit``, ``residuals``, ``lambda``,
    ``lambda_scan`` (lambdas and error estimates, or None), ``log_mean``,
    ``peaks``, ``negative_share`` (how much of the distribution's absolute
    sum is negative), ``rms_residual``, ``nnls_amplitudes``, the inputs,
    and ``notes`` (plain sentences for the log)."""
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel() - float(baseline)
    if x.size != y.size:
        raise ValueError(f"{x.size} x values but {y.size} intensities.")
    if x.size < 3:
        raise ValueError("Need at least three points.")
    if kernel not in KERNELS:
        raise ValueError(f"Unknown kernel {kernel!r}; choose one of {sorted(KERNELS)}.")
    lo, hi = grid_range or DEFAULT_RANGES[kernel]
    grid = log_grid(lo, hi, n_grid)
    K = kernel_matrix(kernel, x, grid)
    A = np.hstack([K, np.ones((x.size, 1))]) if offset else K
    sol, lam_used, scan = _solve(A, y, lam=lam, lambda_range=lambda_range, n_iter=n_iter,
                                 solver=solver, offset=offset)
    amplitudes = sol[:-1] if offset else sol
    off = float(sol[-1]) if offset else 0.0
    y_fit = A @ sol
    residuals = y - y_fit
    absolute = np.abs(amplitudes).sum()
    negative_share = float(-amplitudes[amplitudes < 0].sum() / absolute) if absolute else 0.0
    result = {
        "kind": "1d", "kernel": kernel, "kernel_label": KERNELS[kernel][0],
        "axis": KERNELS[kernel][1], "label": label, "x": x, "y": y, "baseline": float(baseline),
        "grid": grid, "amplitudes": amplitudes, "cumulative": _cumulative(amplitudes),
        "offset": off, "y_fit": y_fit, "residuals": residuals,
        "rms_residual": float(np.sqrt(np.mean(residuals ** 2))),
        "lambda": lam_used, "lambda_auto": lam is None,
        "lambda_scan": _scan_summary(scan),
        "solver": solver, "n_iter": int(n_iter) if solver == "matlab" else 0,
        "log_mean": log_mean(grid, amplitudes), "peaks": distribution_peaks(grid, amplitudes),
        "negative_share": negative_share, "nnls_amplitudes": None,
    }
    if nnls_comparison:
        from scipy.optimize import nnls
        x0, _ = nnls(A, y, maxiter=50 * A.shape[1])
        result["nnls_amplitudes"] = x0[:-1] if offset else x0
        result["nnls_offset"] = float(x0[-1]) if offset else 0.0
    result["notes"] = _notes(result, scan)
    return result


def _scan_summary(scan):
    if scan is None:
        return None
    out = {"method": scan["method"], "lambdas": scan["lambdas"], "errest": scan.get("errest"),
           "s_max": scan.get("s_max")}
    if scan.get("curve") is not None:
        out["residual_norms"] = scan["curve"]["residual_norms"]
        out["solution_norms"] = scan["curve"]["solution_norms"]
    return out


def _lambda_note(result, scan) -> list:
    if not result["lambda_auto"]:
        return [f"λ = {result['lambda']:.4g}, set by hand."]
    lams = scan["lambdas"]
    if scan["method"] == "l_curve":
        notes = [f"λ = {result['lambda']:.4g}, at the corner of the L-curve of the "
                 f"non-negative solutions ({lams.size} trial values)."]
    else:
        notes = [f"λ = {result['lambda']:.4g}, chosen by the Brezinski–Rodriguez–Seatzu "
                 f"error estimate (η₃) after {lams.size} trial solutions"
                 + (" — the 2000-solution limit was reached, so λ may not be the "
                    "minimum." if scan.get("hit_kmax") else ".")]
    if result["lambda"] in (lams.max(), lams.min()):
        notes.append("λ is at the end of the search range: there is no optimum inside it, "
                     "so this λ is not a choice the data support. Set λ by hand.")
    elif scan.get("s_max") and result["lambda"] < 1e-6 * scan["s_max"]:
        notes.append(f"λ is effectively zero (below 10⁻⁶ of the largest singular value, "
                     f"{scan['s_max']:.3g}): the error estimate found its minimum in "
                     "round-off, which happens with fewer data points than grid points. "
                     "The result is then unregularised and unreliable — set λ by hand "
                     "(try 10⁻³–10⁻¹ of that singular value) or use the non-negative solver.")
    return notes


def _notes(result, scan) -> list:
    unit = "m²/s" if result["kernel"] == "diffusion" else "s"
    grid = result["grid"]
    notes = [f"Kernel: {result['kernel_label']}; grid {grid[0]:.3g}–{grid[-1]:.3g} {unit}, "
             f"{grid.size} points, log-spaced."]
    if result.get("baseline"):
        notes.append(f"Baseline {result['baseline']:.6g} subtracted from the data first.")
    notes.extend(_lambda_note(result, scan))
    if result["solver"] == "matlab":
        notes.append(f"Refined {result['n_iter']} times with |x| as the Tikhonov prior "
                     "(the MATLAB scheme). Each refinement weakens the smoothing: the "
                     "number of iterations acts as a second regularisation parameter, so "
                     "keep it the same when comparing samples.")
    else:
        notes.append("Solved with x ≥ 0 imposed exactly (non-negative Tikhonov).")
    if result.get("negative_share", 0) > 0.01:
        notes.append(f"{100 * result['negative_share']:.1f}% of the distribution is negative "
                     "— not physical; the non-negative solver avoids it.")
    for edge, at in (("shortest", 0), ("longest", -1)):
        a = np.clip(result["amplitudes"], 0, None)
        if a.max() > 0 and a[at] > 0.2 * a.max():
            notes.append(f"The distribution is still large at the {edge} end of the grid: "
                         "extend the grid past it, or the population there is misplaced.")
    notes.append("Peak positions and log means are the result; peak widths are set largely "
                 "by λ, not by the sample.")
    return notes


def subsample(x, data, keep_first=0, step=1, axis=-1):
    """The 2D scripts' "Step after" / "Step": every point up to
    ``keep_first``, then every ``step``-th one after it -- a long echo train
    thinned so the problem stays a manageable size."""
    x = np.asarray(x, dtype=float)
    keep_first, step = max(int(keep_first), 0), max(int(step), 1)
    idx = np.concatenate([np.arange(min(keep_first, x.size)),
                          np.arange(min(keep_first, x.size), x.size, step)])
    return x[idx], np.take(np.asarray(data, dtype=float), idx, axis=axis)


#: Above this many entries in the full kernel matrix the 2D solve is refused
#: rather than left to exhaust memory -- thin the echo train (``subsample``)
#: or use fewer grid points.
MAX_2D_KERNEL_ENTRIES = 40_000_000


def regularise_2d(x1, x2, data, kernel1="t2", kernel2="t1_ir", range1=None, range2=None,
                  n1=40, n2=40, baseline=0.0, lam=None, lambda_range=(1e-30, 1e2),
                  n_iter=100, solver="matlab", offset=True, label=None):
    """A 2D distribution (T1-T2, T2-T2, D-T2) from a data matrix.

    ``data`` has one row per ``x2`` value (the indirect encoding: recovery
    delay, first T2 or b) and one column per ``x1`` value (the direct one,
    usually the echo train). The kernel is the Kronecker product of the
    two, as in the scripts (``A = kron(A2, A1)``, plus a column of ones).
    The default pairs are the scripts': ``t2`` x ``t1_ir``/``t1_sr``
    (T1-T2), ``t2`` x ``t2`` (T2-T2), ``t2`` x ``diffusion`` (D-T2).

    Returns ``map`` (``n2`` x ``n1``), ``grid1``/``grid2``,
    ``projection1``/``projection2`` (the map summed along the other axis),
    ``log_mean1``/``log_mean2``, ``fit`` (the fitted data matrix),
    ``residuals``, ``lambda``, ``lambda_scan``, ``notes``."""
    x1 = np.asarray(x1, dtype=float).ravel()
    x2 = np.asarray(x2, dtype=float).ravel()
    data = np.asarray(data, dtype=float) - float(baseline)
    for k in (kernel1, kernel2):
        if k not in KERNELS:
            raise ValueError(f"Unknown kernel {k!r}; choose one of {sorted(KERNELS)}.")
    if data.shape != (x2.size, x1.size):
        raise ValueError(f"The data are {data.shape[0]} × {data.shape[1]}; expected "
                         f"{x2.size} rows (x2) × {x1.size} columns (x1).")
    grid1 = log_grid(*(range1 or DEFAULT_RANGES[kernel1]), n1)
    grid2 = log_grid(*(range2 or DEFAULT_RANGES[kernel2]), n2)
    m, n = x1.size * x2.size, grid1.size * grid2.size + (1 if offset else 0)
    if m * n > MAX_2D_KERNEL_ENTRIES:
        raise ValueError(
            f"The kernel would be {m} × {n} ({m * n / 1e6:.0f} million entries), too large "
            "to solve here. Thin the echo train (keep the first points, then every k-th) "
            "or use fewer grid points.")
    A1 = kernel_matrix(kernel1, x1, grid1)
    A2 = kernel_matrix(kernel2, x2, grid2)
    A = np.kron(A2, A1)
    if offset:
        A = np.hstack([A, np.ones((m, 1))])
    y = data.ravel()          # row-major: y[n1*k + i] = data[k, i], the scripts' y_lex
    sol, lam_used, scan = _solve(A, y, lam=lam, lambda_range=lambda_range, n_iter=n_iter,
                                 solver=solver, offset=offset)
    amplitudes = sol[:-1] if offset else sol
    fmap = amplitudes.reshape(grid2.size, grid1.size)
    fit = (A @ sol).reshape(data.shape)
    result = {
        "kind": "2d", "label": label, "kernel1": kernel1, "kernel2": kernel2,
        "axis1": KERNELS[kernel1][1], "axis2": KERNELS[kernel2][1],
        "x1": x1, "x2": x2, "data": data, "baseline": float(baseline),
        "grid1": grid1, "grid2": grid2, "map": fmap, "offset": float(sol[-1]) if offset else 0.0,
        "fit": fit, "residuals": data - fit,
        "rms_residual": float(np.sqrt(np.mean((data - fit) ** 2))),
        "lambda": lam_used, "lambda_auto": lam is None,
        "lambda_scan": _scan_summary(scan),
        "solver": solver, "n_iter": int(n_iter) if solver == "matlab" else 0,
    }
    result.update(projections(result))
    notes = [f"{KERNELS[kernel2][1]}–{KERNELS[kernel1][1]} map: {x2.size} × {x1.size} data "
             f"points onto a {grid2.size} × {grid1.size} grid."]
    notes.extend(_lambda_note(result, scan))
    notes.append(f"Refined {result['n_iter']} times with |x| as the Tikhonov prior (the "
                 "MATLAB scheme)." if solver == "matlab"
                 else "Solved with x ≥ 0 imposed exactly (non-negative Tikhonov).")
    notes.append(f"Log means: {KERNELS[kernel1][1]} {result['log_mean1']:.4g}, "
                 f"{KERNELS[kernel2][1]} {result['log_mean2']:.4g}.")
    result["notes"] = notes
    return result


def projections(result, within1=None, within2=None) -> dict:
    """The 2D map's projections, each summed over the other axis -- the
    whole of it, or only over ``within2`` (low, high) for axis 1 and
    ``within1`` for axis 2 (the scripts' "proj limits") -- with their log
    means."""
    fmap = np.asarray(result["map"], dtype=float)
    g1, g2 = np.asarray(result["grid1"]), np.asarray(result["grid2"])
    rows = np.ones(g2.size, bool) if within2 is None else (g2 >= min(within2)) & (g2 <= max(within2))
    cols = np.ones(g1.size, bool) if within1 is None else (g1 >= min(within1)) & (g1 <= max(within1))
    p1 = fmap[rows].sum(axis=0)
    p2 = fmap[:, cols].sum(axis=1)
    return {"projection1": p1, "projection2": p2,
            "log_mean1": log_mean(g1, p1), "log_mean2": log_mean(g2, p2)}


# ----------------------------------------------------------------------
# Reporting, export and plots
# ----------------------------------------------------------------------

_TIME_UNITS = {"s": 1.0, "ms": 1e-3, "µs": 1e-6}


def _axis_unit(kernel, time_unit):
    if kernel == "diffusion":
        return 1.0, "m²/s"
    return _TIME_UNITS[time_unit], time_unit


def report(result, time_unit="ms") -> str:
    """Plain sentences: what was done and what came out."""
    lines = list(result.get("notes", []))
    if result["kind"] == "1d":
        scale, unit = _axis_unit(result["kernel"], time_unit)
        lines.append(f"Log mean {result['axis']} = {result['log_mean'] / scale:.4g} {unit}.")
        for k, p in enumerate(result["peaks"], 1):
            lines.append(f"Peak {k}: {result['axis']} ≈ {p['position'] / scale:.4g} {unit} "
                         f"(log mean {p['log_mean'] / scale:.4g} {unit}), "
                         f"{100 * p['share']:.1f}% of the signal.")
        lines.append(f"Offset {result['offset']:.4g}; RMS residual {result['rms_residual']:.4g}.")
    else:
        for axis in ("1", "2"):
            scale, unit = _axis_unit(result["kernel" + axis], time_unit)
            lines.append(f"Log mean {result['axis' + axis]} = "
                         f"{result['log_mean' + axis] / scale:.4g} {unit}.")
        lines.append(f"RMS residual {result['rms_residual']:.4g}.")
    return "\n".join(lines)


def to_csv(result, path, time_unit="s") -> None:
    """The distribution (and, for 1D, the data with the fit) as CSV."""
    import csv
    with open(path, "w", newline="", encoding="utf-8") as handle:
        w = csv.writer(handle)
        for line in report(result, time_unit).splitlines():
            w.writerow([f"# {line}"])
        if result["kind"] == "1d":
            scale, unit = _axis_unit(result["kernel"], time_unit)
            w.writerow([f"{result['axis']} ({unit})", "amplitude", "cumulative",
                        "nnls amplitude"])
            nnls = result.get("nnls_amplitudes")
            for i, g in enumerate(result["grid"]):
                w.writerow([g / scale, result["amplitudes"][i], result["cumulative"][i],
                            "" if nnls is None else nnls[i]])
            w.writerow([])
            xunit = "s/m²" if result["kernel"] == "diffusion" else unit
            xscale = 1.0 if result["kernel"] == "diffusion" else scale
            w.writerow([f"x ({xunit})", "data (baseline subtracted)", "fit", "residual"])
            for xi, yi, fi, ri in zip(result["x"], result["y"], result["y_fit"],
                                      result["residuals"]):
                w.writerow([xi / xscale, yi, fi, ri])
        else:
            s1, u1 = _axis_unit(result["kernel1"], time_unit)
            s2, u2 = _axis_unit(result["kernel2"], time_unit)
            w.writerow([f"{result['axis2']} ({u2}) \\ {result['axis1']} ({u1})"]
                       + [g / s1 for g in result["grid1"]])
            for k, g2 in enumerate(result["grid2"]):
                w.writerow([g2 / s2] + list(result["map"][k]))


def plot_regularisation(result, fig=None, time_unit="ms", colour="#0072B2",
                        fit_colour="#D55E00", show_nnls=False, show_cumulative=True,
                        show_residuals=True, linewidth=1.5, markersize=4.0):
    """The scripts' figures on one figure: the distribution (log axis), its
    cumulative sum, the data with the fit, and the residuals. Returns
    ``fig``."""
    import matplotlib.pyplot as plt
    if fig is None:
        fig = plt.figure(figsize=(7.0, 5.2))
    if result["kind"] != "1d":
        return plot_regularisation_2d(result, fig=fig, time_unit=time_unit)
    scale, unit = _axis_unit(result["kernel"], time_unit)
    panels = 2 + int(show_cumulative) + int(show_residuals)
    cols = 2 if panels > 2 else 1
    rows = int(np.ceil(panels / cols))
    if panels == 2:
        rows, cols = 1, 2
    axes = [fig.add_subplot(rows, cols, i + 1) for i in range(panels)]
    grid = result["grid"] / scale
    ax = axes[0]
    ax.semilogx(grid, result["amplitudes"], color=colour, linewidth=linewidth,
                label="Tikhonov")
    if show_nnls and result.get("nnls_amplitudes") is not None:
        ax.semilogx(grid, result["nnls_amplitudes"], color="0.5", linewidth=0.8,
                    linestyle=":", label="NNLS (no regularisation)")
        ax.legend()
    ax.set_xlabel(_italic_label(f"{result['axis']} ({unit})"))
    ax.set_ylabel(_italic_label("Amplitude"))
    ax.set_title(f"{result['axis']} distribution")
    i = 1
    if show_cumulative:
        ax = axes[i]
        ax.semilogx(grid, result["cumulative"], color=colour, linewidth=linewidth)
        ax.set_xlabel(_italic_label(f"{result['axis']} ({unit})"))
        ax.set_ylabel(_italic_label("Cumulative amplitude"))
        ax.set_title("Cumulative")
        i += 1
    xs = result["x"] if result["kernel"] == "diffusion" else result["x"] / scale
    xlabel = "b (s/m²)" if result["kernel"] == "diffusion" else f"t ({unit})"
    ax = axes[i]
    ax.plot(xs, result["y"], "o", color="black", markersize=markersize, label="Data")
    order = np.argsort(xs)
    ax.plot(xs[order], result["y_fit"][order], color=fit_colour, linewidth=linewidth,
            label="Fit")
    ax.set_xlabel(_italic_label(xlabel))
    ax.set_ylabel(_italic_label("Intensity"))
    ax.set_title("Data and fit")
    ax.legend()
    i += 1
    if show_residuals:
        ax = axes[i]
        ax.plot(xs, result["residuals"], "o", color=fit_colour, markersize=markersize * 0.8)
        ax.axhline(0.0, color="0.5", linewidth=0.8)
        ax.set_xlabel(_italic_label(xlabel))
        ax.set_ylabel(_italic_label("Residual"))
        ax.set_title("Residuals")
    fig.tight_layout()
    return fig


def plot_regularisation_2d(result, fig=None, time_unit="ms", cmap="viridis", levels=6,
                           show_projections=True, show_diagonal=None):
    """The 2D map (log axes) with filled contours, and its projections on
    the top and right, as the scripts' figures 3, 5 and 6. A T2-T2 or a
    T1-T2 map shows the T1 = T2 diagonal (``show_diagonal``, default on
    when both axes are times). Returns ``fig``."""
    import matplotlib.pyplot as plt
    if fig is None:
        fig = plt.figure(figsize=(5.6, 5.0))
    s1, u1 = _axis_unit(result["kernel1"], time_unit)
    s2, u2 = _axis_unit(result["kernel2"], time_unit)
    g1, g2 = result["grid1"] / s1, result["grid2"] / s2
    fmap = np.clip(result["map"], 0, None)
    if show_projections:
        gs = fig.add_gridspec(2, 2, width_ratios=(4, 1), height_ratios=(1, 4),
                              wspace=0.05, hspace=0.05)
        ax = fig.add_subplot(gs[1, 0])
        top = fig.add_subplot(gs[0, 0], sharex=ax)
        right = fig.add_subplot(gs[1, 1], sharey=ax)
    else:
        ax = fig.add_subplot(111)
    vmax = fmap.max() if fmap.max() > 0 else 1.0
    ax.contourf(g1, g2, fmap, levels=np.linspace(vmax / levels, vmax, levels), cmap=cmap)
    ax.contour(g1, g2, fmap, levels=np.linspace(vmax / levels, vmax, levels), colors="k",
               linewidths=0.4)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(_italic_label(f"{result['axis1']} ({u1})"))
    ax.set_ylabel(_italic_label(f"{result['axis2']} ({u2})"))
    times = result["kernel1"] != "diffusion" and result["kernel2"] != "diffusion"
    if show_diagonal is None:
        show_diagonal = times
    if show_diagonal and times:
        lo, hi = max(g1[0], g2[0]), min(g1[-1], g2[-1])
        if hi > lo:
            ax.plot([lo, hi], [lo, hi], color="0.4", linewidth=0.8, linestyle="--")
    if show_projections:
        top.plot(g1, result["projection1"], color="k", linewidth=1.0)
        top.set_xscale("log")
        top.axis("off")
        right.plot(result["projection2"], g2, color="k", linewidth=1.0)
        right.set_yscale("log")
        right.axis("off")
    return fig


def notebook_code(result_kwargs: dict) -> str:
    """The notebook call that reproduces a GUI run."""
    args = ", ".join(f"{k}={v!r}" for k, v in result_kwargs.items())
    return ("import opennmr.notebook as nmr\n"
            f"result = nmr.regularise(x, y, {args})\n"
            "print(nmr.regularisation_report(result))\n"
            "nmr.plot_regularisation(result)")


def _italic_label(text):
    """Quantity symbols italic (plotting.labels.italicise_symbols)."""
    from opennmr.plotting.labels import italicise_symbols
    return italicise_symbols(text)
