"""
SCORE and OUTSCORE -- multivariate resolution of a diffusion or
relaxation series into component spectra.

* **SCORE** (Speedy COmponent REsolution) -- Nilsson & Morris, *Anal.
  Chem.* **2008**, 80, 3777-3782; a variation on the earlier CORE
  method.
* **OUTSCORE** -- Colbourne, Meier, Morris & Nilsson, *Chem. Commun.*
  **2013**, 49, 10510, "Unmixing the NMR spectra of similar species --
  vive la difference".

Both fit the same bilinear model and share the same inner loop. They
differ in **what the outer loop optimises**, and that difference is the
whole reason to have both -- see :data:`RESOLUTION_METHODS`.

What makes this different from everything else in this package
--------------------------------------------------------------
:mod:`opennmr.processing.relaxation` fits **one number per spectrum**:
pick a peak, measure its intensity in each slice, fit a decay to that
series. Every peak is treated as an independent measurement, and a peak
that overlaps another simply cannot be separated -- the intensity you
measure is the sum, and no per-peak fit can undo that.

SCORE fits the **whole data matrix at once**. The model is bilinear:

.. math:: \\mathbf{X} = \\mathbf{C}\\,\\mathbf{S}^{\\mathsf{T}} + \\mathbf{E}

* :math:`\\mathbf{X}` -- the data, ``(m gradient levels x n spectral points)``
* :math:`\\mathbf{C}` -- ``(m x k)`` decays, column *j* being
  :math:`\\exp(-b_i D_j)`: **entirely determined by the k decay
  constants**
* :math:`\\mathbf{S}` -- ``(n x k)`` the pure spectrum of each component
* :math:`\\mathbf{E}` -- residual

The trick that makes it "speedy", and the whole reason it beats CORE:
for any trial set of diffusion coefficients, :math:`\\mathbf{C}` is
fixed, so the optimal :math:`\\mathbf{S}` follows from a **linear**
least-squares solve. The nonlinear search therefore runs over only the
*k* decay constants, however many thousands of spectral points there
are. This is separable nonlinear least squares (variable projection);
CORE optimised the spectra nonlinearly too, which is why it was slow
and guess-sensitive.

What you get out that a per-peak fit cannot give
------------------------------------------------
* **Resolved spectra of overlapping components.** Two species whose
  peaks sit on top of each other are separated by their *differing
  decay*, not by their positions.
* **Every point contributes to every D.** The decay constants are
  determined by the whole spectrum jointly, not by one peak's height.

What it demands in return
-------------------------
* **The number of components, k, is an input, not an output.** The
  paper gives no method for choosing it; :func:`estimate_components`
  here offers an SVD-based *suggestion* and is labelled as such. Ask
  for too many and SCORE will happily split one species into two
  near-identical spectra with near-identical D.
* **The model must be right.** A bilinear model says each component has
  one spectrum and one decay constant. A polydisperse component, or one
  whose lineshape changes across the series, breaks that assumption --
  and the fit will not tell you so, it will just distribute the misfit
  over the components you asked for.
* **Phase and baseline matter more than usual.** A dispersive or
  baseline-rolling spectrum is not a non-negative superposition of
  component spectra, so the constraint that most stabilises the
  solution stops applying.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares, minimize, nnls

#: The two objectives, and when each one is the right choice.
#:
#: ``score`` minimises the **residual** ``||X - C S^T||^2``: how well
#: the model reproduces the measured data. It is the obvious criterion
#: and it works whenever the components' decays differ enough to be
#: told apart.
#:
#: ``outscore`` minimises the **cross-talk** between the resolved
#: spectra instead -- the overlap between each pair, i.e. how much of
#: one component's signal has leaked into another's. It asks a
#: different question: not "does this fit?" but "are these actually
#: separate species?".
#:
#: **Why that matters, and it is not a preference.** Separating
#: overlapped spectra by residual typically needs the diffusion
#: coefficients to differ by ~30%; below that, the residual surface has
#: a large flat region around its minimum, so a wide range of
#: quite-different answers fit almost equally well and the optimiser
#: has nothing to steer by. The cross-talk surface has a much sharper
#: minimum over the same data, which is what lets OUTSCORE separate
#: species differing by closer to 3-4% -- almost an order of magnitude
#: better on similar-sized species.
#:
#: The trade: cross-talk is only a meaningful criterion when the
#: components genuinely *do* have different peaks. For components that
#: legitimately share most of their spectrum, minimising overlap is
#: pushing toward something untrue, and ``score`` is the honest choice.
RESOLUTION_METHODS = {
    "score": {
        "label": "SCORE (minimise residual)",
        "reference": "Nilsson & Morris, Anal. Chem. 2008, 80, 3777",
        "min_components": 1,
    },
    "outscore": {
        "label": "OUTSCORE (minimise cross-talk)",
        "reference": "Colbourne et al., Chem. Commun. 2013, 49, 10510",
        # Cross-talk is defined between *pairs*: with one component
        # there are no pairs and nothing to minimise.
        "min_components": 2,
    },
}

#: Decay kernels SCORE can resolve against, keyed the way the rest of
#: this package names them. Each maps ``(x, constant)`` to the
#: attenuation of a component at that gradient level or delay.
#:
#: The x variable is whatever the experiment actually varied: a b-value
#: in s/m^2 for diffusion (constant = D, m^2/s), a delay in seconds for
#: relaxation (constant = a rate, 1/s -- **not** a time constant, so
#: that every kernel here is a plain decaying exponential in its own
#: constant and the log-space search below behaves identically for all
#: of them; :func:`score` converts to T1/T2 on the way out).
DECAY_KERNELS = {
    "diffusion": {
        "func": lambda x, constant: np.exp(-np.outer(x, constant)),
        "label": "Diffusion", "constant": "D", "unit": "m^2/s",
    },
    "relaxation": {
        "func": lambda x, constant: np.exp(-np.outer(x, constant)),
        "label": "Relaxation (decay)", "constant": "rate", "unit": "1/s",
    },
}


def estimate_components(data, max_components: int = 8, noise_factor: float = 4.0) -> dict:
    """An SVD-based **suggestion** for how many components are present.

    A bilinear mixture of *k* components spans a rank-*k* subspace, so
    without noise the singular values drop to zero after the *k*-th.
    With real noise they drop to the *noise floor* instead, and the
    count is of how many rise clearly above it.

    The floor is estimated from the singular values themselves -- the
    median of the lower half, which for a rank-*k* matrix is entirely
    noise -- and a value counts as real at ``noise_factor`` times it.
    A **fixed fraction of the largest singular value was tried first
    and is not good enough**: it depends on how much total signal
    happens to be present, so the same data with one strong component
    instead of two crosses the threshold differently. Measured
    directly: a single-component series was reported as 8 components
    by a 1%-of-largest rule, and correctly as 1 by this one.

    **This is a suggestion, not a measurement**, and the paper offers no
    method at all. Two species with very similar diffusion coefficients
    span a nearly-degenerate subspace and will look like one component;
    a component whose lineshape drifts across the series will look like
    two. Treat the number as a starting point, run SCORE at *k* and
    *k±1*, and prefer the answer whose component spectra actually look
    like plausible, separate molecules.

    Returns a dict with ``n_components`` (the count), ``singular_values``
    (normalised to the largest) and ``max_components``.
    """
    matrix = np.asarray(data, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"data must be 2-D (levels x points), got shape {matrix.shape}.")
    singular = np.linalg.svd(matrix, compute_uv=False)
    peak = float(singular[0]) if singular.size else 0.0
    if peak <= 0:
        return {"n_components": 0, "singular_values": singular, "max_components": max_components,
                "noise_floor": 0.0, "noise_factor": float(noise_factor)}
    normalised = singular / peak
    # The lower half is noise for any realistic rank, so its median is
    # a robust floor -- robust specifically to the first few genuinely
    # large values, which a mean or a max would be dragged by.
    tail = normalised[max(1, normalised.size // 2):]
    floor = float(np.median(tail)) if tail.size else 0.0
    cutoff = max(floor * noise_factor, 1e-12)
    count = int(np.count_nonzero(normalised[:max_components] >= cutoff))
    return {
        "n_components": max(1, count),
        "singular_values": normalised,
        "max_components": max_components,
        "noise_floor": floor,
        "noise_factor": float(noise_factor),
        "cutoff": cutoff,
    }


def _signal_columns(matrix, threshold: float):
    """Indices of the spectral points worth fitting: those whose
    strongest value across the series clears ``threshold`` of the
    matrix maximum.

    Two reasons, and the first is not speed. **A noise-only column
    carries no information about any component's decay**, but it does
    contribute its noise to the residual that the decay constants are
    being optimised against -- with 16k points and a handful of real
    peaks, the objective can be dominated by baseline. Second, it is
    also what makes the fit interactive: the linear solve is over the
    kept columns only.
    """
    strongest = np.max(np.abs(matrix), axis=0)
    peak = float(np.max(strongest)) if strongest.size else 0.0
    if peak <= 0:
        raise ValueError("The data matrix is entirely zero -- nothing to resolve.")
    keep = np.flatnonzero(strongest >= threshold * peak)
    if keep.size < 1:
        raise ValueError(
            f"No spectral point reaches {threshold:.1%} of the maximum -- lower the threshold."
        )
    return keep


def crosstalk(spectra, normalise: bool = True) -> float:
    """Total overlap between every pair of component spectra -- what
    OUTSCORE minimises.

    For each pair, ``sum_k |S_i[k]| * |S_j[k]|``: nonzero wherever two
    components both claim intensity at the same point, which is exactly
    what leakage of one species into another's resolved spectrum looks
    like. Absolute values because a component that has picked up an
    inverted copy of another's peaks is just as contaminated as one
    that picked up an upright copy.

    **Normalised by each spectrum's own norm** (i.e. the cosine
    similarity), and that is a deliberate choice rather than the bare
    dot product the method is usually stated with. The bare product is
    degenerate at the edges of the search: a component whose decay
    constant runs off to where it contributes nothing has a spectrum
    near zero, and therefore near-zero overlap with everything -- a
    perfect score for having resolved nothing. The normalised form is
    scale-invariant, so shrinking a component buys nothing. A component
    that *has* collapsed to zero returns the worst possible value (1.0)
    instead of ``0/0``, which pushes the optimiser back out.
    """
    spectra = np.asarray(spectra, dtype=float)
    n_components = spectra.shape[0]
    total = 0.0
    for i in range(n_components):
        for j in range(i + 1, n_components):
            a, b = np.abs(spectra[i]), np.abs(spectra[j])
            overlap = float(a @ b)
            if normalise:
                norm_a, norm_b = float(np.linalg.norm(a)), float(np.linalg.norm(b))
                overlap = overlap / (norm_a * norm_b) if (norm_a > 0 and norm_b > 0) else 1.0
            total += overlap
    return total


def _solve_spectra(decays, matrix, non_negative: bool):
    """The inner, *linear* half of SCORE: the component spectra that
    best explain ``matrix`` given a fixed decay matrix.

    Unconstrained this is one least-squares solve for all points at
    once. With ``non_negative`` it is one NNLS per spectral point,
    which is what the method actually specifies but costs a solve per
    column -- see :func:`score` for why that is applied once at the end
    by default rather than on every iteration of the outer search."""
    if not non_negative:
        spectra, *_ = np.linalg.lstsq(decays, matrix, rcond=None)
        return spectra
    n_components = decays.shape[1]
    spectra = np.empty((n_components, matrix.shape[1]), dtype=float)
    for column in range(matrix.shape[1]):
        spectra[:, column], _residual = nnls(decays, matrix[:, column])
    return spectra


def score(x_values, data, n_components: int = 2, kind: str = "diffusion",
          method: str = "score", initial=None, bounds=None, threshold: float = 0.02,
          non_negative: bool = True, non_negative_during_fit: bool = False,
          max_nfev: int = 2000) -> dict:
    """Resolve a diffusion or relaxation series into ``n_components``
    component spectra and their decay constants, by SCORE or OUTSCORE.

    Parameters
    ----------
    method : {"score", "outscore"}
        Which objective the outer loop minimises -- the residual, or
        the cross-talk between resolved spectra. See
        :data:`RESOLUTION_METHODS`; ``"outscore"`` separates species
        whose decays differ by only a few percent, where ``"score"``
        needs about 30%, but is only meaningful when the components
        genuinely have different peaks.
    x_values : array, shape (m,)
        b-values (s/m^2) for ``kind="diffusion"``, delays (s) for
        ``kind="relaxation"`` -- one per row of ``data``.
    data : array, shape (m, n)
        The series itself: one spectrum per gradient level or delay.
        Exactly the shape a pseudo-2D's ``.data`` already has.
    n_components : int
        How many components to resolve. **An input, not an output** --
        see :func:`estimate_components`.
    initial : sequence, optional
        Starting decay constants, one per component. Omitted, they are
        spread over decades around the value that would decay by 1/e
        across the sampled range -- components started close together
        give the optimiser nothing to separate them by, and it will
        converge with two components sitting on the same constant.

        For ``method="outscore"`` the default is different and
        deliberate: a SCORE run is done first and its answer used as
        the starting point, because OUTSCORE cannot reliably find its
        own sharp minimum from a cold start. Supplying ``initial``
        explicitly skips that.
    bounds : (lo, hi), optional
        Bounds on the decay constants. Default spans four decades
        either side of the automatic guesses.
    threshold : float
        Skip spectral points below this fraction of the matrix maximum
        -- see :func:`_signal_columns` for why this affects the answer
        and not just the speed.
    non_negative : bool
        Apply the method's non-negativity constraint to the resolved
        spectra. A component cannot contribute negative signal, and
        this is the single most stabilising constraint available.
    non_negative_during_fit : bool
        Apply that constraint on *every* iteration rather than once at
        the end. Truer to the published method, and materially slower
        (one NNLS per spectral point per iteration against a single
        least-squares solve). Off by default: the decay constants come
        out essentially the same either way, and the unconstrained
        inner solve is exactly what makes SCORE "speedy".

    Returns
    -------
    dict with ``constants`` (per component, in the kernel's own unit),
    ``D`` or ``rates``/``T`` depending on ``kind``, ``spectra``
    ``(k x n)`` on the full point grid, ``fractions`` (each component's
    share of the total resolved signal), ``fit`` (the reconstructed
    matrix), ``residual_norm``, ``r_squared``, ``columns`` (which
    points were actually fitted) and ``n_components``.

    Raises
    ------
    ValueError
        On shape mismatches, an unknown ``kind``, fewer gradient levels
        than components (the model is then underdetermined), or an
        all-zero matrix.
    """
    if kind not in DECAY_KERNELS:
        raise ValueError(f"Unknown kind '{kind}'. Available: {sorted(DECAY_KERNELS)}")
    if method not in RESOLUTION_METHODS:
        raise ValueError(f"Unknown method '{method}'. Available: {sorted(RESOLUTION_METHODS)}")
    x_values = np.asarray(x_values, dtype=float)
    matrix = np.asarray(data, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"data must be 2-D (levels x points), got shape {matrix.shape}.")
    if matrix.shape[0] != x_values.size:
        raise ValueError(
            f"{matrix.shape[0]} rows against {x_values.size} x-values -- one per row is required."
        )
    if n_components < 1:
        raise ValueError(f"n_components must be >= 1, got {n_components}.")
    minimum = RESOLUTION_METHODS[method]["min_components"]
    if n_components < minimum:
        raise ValueError(
            f"method '{method}' needs at least {minimum} components (cross-talk is defined "
            f"between pairs, and there are none with {n_components}). Use method='score'."
        )
    if x_values.size < n_components + 1:
        raise ValueError(
            f"{x_values.size} gradient level(s)/delay(s) cannot determine {n_components} "
            f"component(s) -- the bilinear model is underdetermined."
        )

    order = np.argsort(x_values)
    x_values, matrix = x_values[order], matrix[order]

    columns = _signal_columns(matrix, threshold)
    fitted_block = matrix[:, columns]

    # OUTSCORE is seeded from a SCORE run unless the caller supplied a
    # starting point. **This is not an optimisation, it is required.**
    # The sharp cross-talk minimum that gives OUTSCORE its resolving
    # power is exactly what makes it hard to find from far away:
    # started from the automatic decade-spaced guesses, it converged to
    # 69-100% errors on cases SCORE got to within 5%. Seeded from
    # SCORE's answer it beats SCORE everywhere (measured: at a 1.03x
    # ratio in D, spectral correlation 0.981 against SCORE's 0.764).
    # SCORE's flat error surface is good at finding the right basin;
    # OUTSCORE's sharp one is good at finding the bottom of it.
    seeded_from = None
    if method == "outscore" and initial is None:
        seed_run = score(
            x_values, matrix, n_components=n_components, kind=kind, method="score",
            bounds=bounds, threshold=threshold, non_negative=non_negative,
            non_negative_during_fit=non_negative_during_fit, max_nfev=max_nfev,
        )
        initial = seed_run["constants"]
        seeded_from = seed_run

    guesses = _initial_constants(x_values, initial, n_components)
    if bounds is None:
        lo = [g * 1e-4 for g in guesses]
        hi = [g * 1e4 for g in guesses]
    else:
        lo, hi = (list(bounds[0]), list(bounds[1]))

    kernel = DECAY_KERNELS[kind]["func"]
    inner_non_negative = non_negative and non_negative_during_fit

    # Both methods share this inner half exactly: for a trial set of
    # decay constants, the best spectra follow from a *linear* solve.
    # Searched in log10 space for the same reason every diffusion
    # coefficient in this package is (see
    # opennmr.processing.relaxation.LOG_FITTED_PARAMS): the constants
    # span decades, and a linear search on them converges to the wrong
    # answer even from a good starting point.
    def spectra_for(log_constants):
        decays = kernel(x_values, 10.0 ** np.asarray(log_constants))
        return decays, _solve_spectra(decays, fitted_block, inner_non_negative)

    log_lo, log_hi = np.log10(lo), np.log10(hi)

    if method == "score":
        def residuals(log_constants):
            decays, spectra = spectra_for(log_constants)
            return (fitted_block - decays @ spectra).ravel()

        solution = least_squares(
            residuals, np.log10(guesses), bounds=(log_lo, log_hi), max_nfev=max_nfev,
        )
        log_constants, n_evaluations, success = solution.x, solution.nfev, solution.success
    else:
        def objective(log_constants):
            _decays, spectra = spectra_for(log_constants)
            return crosstalk(spectra)

        # Nelder-Mead, not least_squares: the cross-talk objective is a
        # single scalar (not a residual vector), and taking absolute
        # values makes it non-smooth, which a derivative-based method
        # handles badly. With only k parameters a simplex is cheap, and
        # the sharper minimum this objective is chosen for is exactly
        # what a simplex is good at descending into.
        solution = minimize(
            objective, np.log10(guesses), method="Nelder-Mead",
            bounds=list(zip(log_lo, log_hi)),
            options={"maxfev": max_nfev, "xatol": 1e-4, "fatol": 1e-10},
        )
        log_constants, n_evaluations, success = solution.x, solution.nfev, solution.success

    constants = 10.0 ** np.asarray(log_constants)
    # Slowest-decaying first, so component 1 is the largest/most
    # aggregated species -- the order these are quoted in, and stable
    # across runs regardless of where the optimiser happened to place
    # each one.
    ranking = np.argsort(constants)
    constants = constants[ranking]

    decays = kernel(x_values, constants)
    # The final spectra are solved on *every* point, not only the
    # fitted ones: the decay constants are what the threshold protects,
    # and a component spectrum with holes punched in it wherever the
    # signal was weak would be useless to look at.
    spectra = _solve_spectra(decays, matrix, non_negative)
    reconstructed = decays @ spectra

    residual = matrix - reconstructed
    ss_res = float(np.sum(residual ** 2))
    ss_tot = float(np.sum((matrix - matrix.mean()) ** 2))
    weights = np.sum(np.abs(spectra), axis=1)
    total = float(np.sum(weights))

    result = {
        "constants": constants,
        "spectra": spectra,
        "fractions": (weights / total) if total else np.full(n_components, np.nan),
        "fit": reconstructed,
        "residual_norm": float(np.linalg.norm(residual)),
        "r_squared": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "columns": columns,
        "n_components": n_components,
        "kind": kind,
        "x": x_values,
        "decays": decays,
        "threshold": float(threshold),
        "non_negative": bool(non_negative),
        "success": bool(success),
        "n_evaluations": int(n_evaluations),
        "method": method,
        # The SCORE run OUTSCORE started from, when it was made here --
        # so the pair can be compared without running SCORE twice, and
        # so it is visible what the refinement actually changed.
        "seed": seeded_from,
        # Reported for both methods, so the two are directly
        # comparable on the same data: SCORE optimised the first and
        # OUTSCORE the second, but each says something about the
        # answer either way.
        "crosstalk": crosstalk(spectra),
    }
    result["warnings"] = _diagnose(result, seeded_from)
    if kind == "diffusion":
        result["D"] = constants
    else:
        result["rates"] = constants
        # Reported as a time constant too, because that is what anyone
        # reads a relaxation result as -- the fit itself stays in rate
        # space so the log search behaves the same as for diffusion.
        with np.errstate(divide="ignore"):
            result["T"] = np.where(constants > 0, 1.0 / constants, np.inf)
    return result


#: Cross-talk above which OUTSCORE is reporting that it *could not*
#: separate the components -- measured against synthetic mixtures:
#: successful separations left 0.01-0.11, while a pair genuinely
#: sharing their strongest resonance never got below 0.37.
_CROSSTALK_UNSEPARATED = 0.25

#: Relative disagreement between OUTSCORE and its SCORE seed that
#: means the data does not actually determine the answer. Successful
#: runs agreed to under 1%; the failing case differed by 69%.
_SEED_DISAGREEMENT = 0.25


def _diagnose(result: dict, seed: dict | None) -> list[str]:
    """Warnings about a resolution that fits well but may still be
    wrong.

    **Fit quality cannot do this job here, and that is the point.**
    Measured on synthetic mixtures: an OUTSCORE run whose diffusion
    coefficients were 69% wrong reported R² = 0.99940, against 0.99937
    for the SCORE run that got them right. The error surface really is
    that flat -- wildly different answers reproduce the data equally
    well -- so a user reading R² alone would conclude a badly wrong
    answer was excellent. These two checks look at things R² cannot
    see.
    """
    warnings = []
    method = result.get("method", "score")

    if method == "outscore" and result["crosstalk"] > _CROSSTALK_UNSEPARATED:
        warnings.append(
            f"Cross-talk is still {result['crosstalk']:.2f} after minimising it "
            f"(≲0.1 when components separate cleanly). The components appear to "
            f"share resonances, which is the case OUTSCORE's assumption does not "
            f"cover -- minimising overlap then pulls away from the truth rather "
            f"than toward it. Prefer SCORE for this sample."
        )

    if seed is not None:
        relative = np.abs(result["constants"] - seed["constants"]) / np.abs(seed["constants"])
        worst = float(np.max(relative)) if relative.size else 0.0
        if worst > _SEED_DISAGREEMENT:
            warnings.append(
                f"OUTSCORE and SCORE disagree by up to {worst:.0%} on the decay "
                f"constants while fitting the data equally well (R² {result['r_squared']:.5f} "
                f"vs {seed['r_squared']:.5f}). The data does not determine the answer "
                f"to that precision -- treat neither as settled, and get more "
                f"gradient levels or a wider range before trusting either."
            )
    return warnings


def _initial_constants(x_values, initial, n_components: int) -> np.ndarray:
    """Starting decay constants: whatever the caller supplied, else a
    decade-spaced spread around the constant that would decay by 1/e
    across the sampled range.

    The *spread* is what matters. Two components started at similar
    values give the optimiser nothing to tell them apart, and it
    converges with both on the same constant and the spectra split
    arbitrarily between them -- a fit that looks converged and means
    nothing. Same failure, and the same fix, as the multi-component
    Stejskal-Tanner guesses in
    :func:`opennmr.processing.diffusion.build_diffusion_model`.
    """
    if initial is not None:
        values = np.atleast_1d(np.asarray(initial, dtype=float))
        if values.size != n_components:
            raise ValueError(
                f"initial has {values.size} entries for {n_components} component(s)."
            )
        if np.any(values <= 0):
            raise ValueError("Starting decay constants must be positive.")
        return values
    positive = x_values[x_values > 0]
    centre = 1.0 / float(np.max(positive)) if positive.size else 1.0
    return np.array([centre * 10.0 ** ((n_components - 1) / 2.0 - i)
                     for i in range(n_components)])


def components_to_csv(result: dict, ppm) -> str:
    """The resolved components as CSV: a header block naming each
    component's decay constant and share, then one column of shift and
    one column per component spectrum.

    A pure string builder (no file I/O), so it can be tested against
    its own output rather than against "did a file appear".

    The header carries the decay constants because a bare table of
    spectra is not a result -- the number that identifies each
    component is the one thing a spreadsheet column heading cannot
    hold, and separating them is how a resolved component ends up
    mislabelled later.
    """
    import csv
    from io import StringIO

    from .relaxation import format_diffusion_value, format_time_value

    ppm = np.asarray(ppm, dtype=float)
    spectra = np.asarray(result["spectra"], dtype=float)
    if ppm.size != spectra.shape[1]:
        raise ValueError(
            f"{ppm.size} shifts against {spectra.shape[1]} spectral points."
        )

    buffer = StringIO()
    writer = csv.writer(buffer)
    method = RESOLUTION_METHODS[result.get("method", "score")]
    writer.writerow([f"{method['label']} — {result['n_components']} components"])
    writer.writerow([method["reference"]])
    writer.writerow([])
    writer.writerow(["Component", "Decay constant", "Unit", "Share",
                      "R^2", "Cross-talk", "Non-negative"])
    unit = DECAY_KERNELS[result["kind"]]["unit"]
    for index in range(result["n_components"]):
        if result["kind"] == "diffusion":
            pretty = format_diffusion_value(result["constants"][index])
        else:
            pretty = format_time_value(result["T"][index])
        writer.writerow([
            index + 1, f"{result['constants'][index]:.6g}", unit,
            f"{result['fractions'][index]:.4f}",
            f"{result.get('r_squared', float('nan')):.6f}",
            f"{result.get('crosstalk', float('nan')):.4f}",
            "yes" if result["non_negative"] else "no",
            pretty,
        ])
    for warning in result.get("warnings", []):
        writer.writerow([])
        writer.writerow(["WARNING", warning])

    writer.writerow([])
    writer.writerow(["Component spectra"])
    writer.writerow(["ppm"] + [f"component_{i + 1}" for i in range(result["n_components"])])
    for point in range(ppm.size):
        writer.writerow([f"{ppm[point]:.6f}"]
                         + [f"{spectra[i, point]:.6g}" for i in range(spectra.shape[0])])
    return buffer.getvalue()


def format_score_report(result: dict, width: int = 66) -> str:
    """A readable summary of a :func:`score` result.

    Deliberately states what SCORE cannot tell you alongside what it
    can: the component count was an input, so a good R² is evidence
    that *k* components fit, never that *k* is the right number."""
    from .relaxation import format_diffusion_value, format_time_value

    rule = "=" * width
    name = RESOLUTION_METHODS[result.get("method", "score")]["label"].split(" (")[0]
    lines = [rule, f"{name} — {result['n_components']} components".center(width).rstrip(), rule, ""]
    lines.append("  Component        share    decay constant")
    lines.append("  " + "-" * (width - 4))
    for index in range(result["n_components"]):
        share = result["fractions"][index]
        if result["kind"] == "diffusion":
            value = format_diffusion_value(result["constants"][index])
        else:
            value = f"{format_time_value(result['T'][index])} (T)"
        lines.append(f"    {index + 1:<12} {share * 100:6.2f}%    {value}")

    lines.append("")
    lines.append("  Fit quality")
    lines.append("  " + "-" * (width - 4))
    if np.isfinite(result.get("r_squared", np.nan)):
        lines.append(f"    R²            = {result['r_squared']:.5f}")
    lines.append(f"    cross-talk    = {result.get('crosstalk', float('nan')):.4f}"
                  "   (0 = fully separated)")
    lines.append(f"    points fitted = {len(result['columns'])} (≥ {result['threshold']:.0%} of max)")
    lines.append(f"    non-negative  = {'yes' if result['non_negative'] else 'no'}")
    lines.append("")
    # Both numbers are shown for both methods, so it is visible which
    # one this run actually optimised -- a low cross-talk from OUTSCORE
    # is what it was told to achieve, not independent evidence.
    optimised = "residual (R²)" if result.get("method", "score") == "score" else "cross-talk"
    lines.append(f"  This run minimised the {optimised}; the other number is reported")
    lines.append("  for comparison, not as independent corroboration.")
    lines.append("")
    lines.append("  The number of components was an input, not a result: a good R²")
    lines.append("  says this many components fit, never that this is the right")
    lines.append("  number. Compare against one more and one fewer.")

    for warning in result.get("warnings", []):
        lines.append("")
        lines.append("  ⚠ " + "\n    ".join(_wrap_text(warning, width - 6)))
    lines.append(rule)
    return "\n".join(lines)


def _wrap_text(text: str, width: int) -> list[str]:
    """Greedy word wrap, local so the report's own width is the one
    thing controlling its layout."""
    words, lines, current = text.split(), [], ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) > width and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines
