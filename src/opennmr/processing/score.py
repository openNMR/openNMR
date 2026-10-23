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

#: Default curve-fit model per ``kind``, when :func:`score` is not
#: given one explicitly. Both are the plain single-exponential the
#: method was published with -- ``stejskal_tanner`` is exactly
#: ``exp(-b D)`` once its amplitude is absorbed into the component
#: spectrum, and ``exponential_decay`` exactly ``exp(-t/T)``.
DEFAULT_MODELS = {"diffusion": "stejskal_tanner", "relaxation": "exponential_decay"}

#: What each ``kind`` is measuring, for labels and reports. The decay
#: *function* is no longer defined here: it comes from
#: :data:`opennmr.processing.relaxation.RELAXATION_MODELS`, the same
#: registry the Fit Curve panel fits with -- see :func:`build_kernel`.
DECAY_KINDS = {
    "diffusion": {"label": "Diffusion", "unit": "m^2/s"},
    "relaxation": {"label": "Relaxation (decay)", "unit": "s"},
}

#: Smallest positive value a log-space bound is allowed to reach --
#: a bound of 0 has no logarithm, and 1e-20 is well below any physical
#: answer for either a diffusion coefficient or a relaxation time.
#: Matches :data:`opennmr.processing.relaxation._LOG_FLOOR`.
_LOG_FLOOR = 1e-20

#: How far either side of its automatic guess a component's rate
#: parameter is allowed to travel, when the model's own bounds are
#: open-ended (as they are for every model in the registry: a T2 is
#: bounded only by ``[1e-9, inf]``). Four decades each way, which is
#: the window this module has always searched.
_RATE_DECADES = 1e4


def build_kernel(model=None, kind: str = "diffusion", x_values=None, y_values=None) -> dict:
    """A SCORE decay kernel built from a **curve-fit model**.

    This is what makes SCORE and Fit Curve agree. Both now evaluate the
    same function from the same registry
    (:data:`opennmr.processing.relaxation.RELAXATION_MODELS`, or a spec
    built by :func:`~opennmr.processing.relaxation.build_custom_model`
    or :func:`~opennmr.processing.diffusion.build_diffusion_model`), so
    a Stejskal-Tanner fit and a SCORE resolution of the same series are
    the same model asked two different questions -- not two
    independently-written exponentials that could drift apart.

    **The amplitude parameter is dropped, not fitted.** SCORE's model is
    bilinear: each component contributes ``spectrum x decay``, and the
    component's overall size lives entirely in its spectrum, which is
    solved for linearly. A scale parameter in the decay as well would
    be exactly redundant with it -- the product ``I0 * S`` has infinitely
    many splits, and the optimiser would wander along that ridge. So the
    model's own amplitude is pinned at 1 and every *other* parameter is
    searched per component.

    Which parameter that is, is **measured, not assumed from its name**:
    each is doubled in turn and the one that doubles the whole curve is
    the amplitude. A model with no such parameter is rejected, and that
    rejection is the useful behaviour rather than a limitation -- a
    biexponential model has no single amplitude *because it already
    describes two populations*, which is the job SCORE's own component
    count is doing. Asking for a 2-component SCORE with a biexponential
    kernel would be asking for four.

    ``model`` : a key into ``RELAXATION_MODELS``, a spec dict, or
    ``None`` for ``DEFAULT_MODELS[kind]``.
    ``x_values``/``y_values`` : the series being fit, used only to
    evaluate the model's own ``p0``/``bounds`` callables (which are
    data-driven) and to probe for the amplitude parameter.

    Returns a dict with ``spec``, ``name``, ``amplitude`` (the pinned
    parameter), ``params`` (the free ones, in order), ``rate_param``
    (the one whose per-component spread separates the components),
    ``log_params`` (searched in log10 space), ``p0``/``bounds`` for the
    free parameters, and ``func(x, values)`` mapping an ``(k, p)``
    array of per-component free parameters to the ``(m, k)`` decay
    matrix.
    """
    from .relaxation import RELAXATION_MODELS

    if kind not in DECAY_KINDS:
        raise ValueError(f"Unknown kind '{kind}'. Available: {sorted(DECAY_KINDS)}")
    if model is None:
        model = DEFAULT_MODELS[kind]
    if isinstance(model, str):
        name = model
        if model not in RELAXATION_MODELS:
            raise ValueError(
                f"Unknown model '{model}'. Available: {sorted(RELAXATION_MODELS)}"
            )
        spec = RELAXATION_MODELS[model]
    else:
        spec = model
        name = spec.get("name") or "custom"

    param_names = list(spec["params"])
    func = spec["func"]

    x_probe = np.asarray(x_values, dtype=float) if x_values is not None else np.linspace(0.0, 1.0, 8)
    y_probe = np.asarray(y_values, dtype=float) if y_values is not None else np.ones_like(x_probe)
    start = _model_p0(spec, x_probe, y_probe, param_names)

    amplitude = _amplitude_parameter(func, param_names, start, x_probe)
    if amplitude is None:
        raise ValueError(
            f"The model '{name}' has no single amplitude parameter, so it cannot be a "
            f"SCORE kernel: its parameters are {', '.join(param_names)}. A model that "
            f"already describes several populations (a biexponential, an N-component "
            f"Stejskal-Tanner) is doing the job SCORE's component count does – use the "
            f"single-component form of it and ask SCORE for that many components."
        )

    free = [p for p in param_names if p != amplitude]
    if not free:
        raise ValueError(f"The model '{name}' has nothing left to fit once its amplitude is pinned.")

    rate_param = spec.get("rate_param") or free[0]
    if rate_param not in free:
        rate_param = free[0]

    declared_log = set(spec.get("log_params", ()))
    # The rate parameter is always searched in log10 space, declared or
    # not: it spans decades (a D from 1e-13 to 1e-8, a T from ms to
    # minutes) and a linear search on it converges to the wrong answer
    # even from a good starting point. That was already true of this
    # module's own hard-coded search; it is stated here now that the
    # parameter comes from the model.
    log_params = tuple(p for p in free if p in declared_log or p == rate_param)

    lo, hi = _model_bounds(spec, x_probe, y_probe, param_names)
    index_of = {p: param_names.index(p) for p in param_names}

    def evaluate(x, values):
        """``values`` : (k, p) free parameters per component -> (m, k)."""
        values = np.atleast_2d(np.asarray(values, dtype=float))
        columns = []
        for row in values:
            arguments = dict(zip(free, row))
            arguments[amplitude] = 1.0
            columns.append(np.asarray(func(x, *[arguments[p] for p in param_names]), dtype=float))
        return np.column_stack(columns)

    return {
        "spec": spec,
        "name": name,
        "amplitude": amplitude,
        "params": free,
        "rate_param": rate_param,
        "log_params": log_params,
        "p0": [float(start[index_of[p]]) for p in free],
        "bounds": ([float(lo[index_of[p]]) for p in free],
                   [float(hi[index_of[p]]) for p in free]),
        "func": evaluate,
        "kind": kind,
        "unit": DECAY_KINDS[kind]["unit"],
    }


def _model_p0(spec, x, y, param_names) -> list:
    """The model's own automatic starting guesses, or 1.0 each when it
    does not supply a ``p0`` callable (a custom equation carries a
    plain list instead)."""
    guess = spec.get("p0")
    if callable(guess):
        return [float(v) for v in guess(x, y)]
    if guess is not None:
        values = list(guess)
        return [float(values[i]) if i < len(values) else 1.0 for i in range(len(param_names))]
    return [1.0] * len(param_names)


def _model_bounds(spec, x, y, param_names):
    """The model's own bounds, or unbounded when it supplies none."""
    bounds = spec.get("bounds")
    if callable(bounds):
        lo, hi = bounds(x, y)
    elif bounds is not None:
        lo, hi = bounds
    else:
        lo = [-np.inf] * len(param_names)
        hi = [np.inf] * len(param_names)
    return list(lo), list(hi)


def _amplitude_parameter(func, param_names, start, x) -> str | None:
    """Which parameter scales the whole curve, found by doubling each
    one and checking whether the output doubles.

    Measured rather than matched against a list of names (``M0``,
    ``I0``, ``A``, ...), because a custom equation someone typed into
    the Fit Curve panel has whatever parameter names they chose, and a
    name-based guess there would pin the wrong parameter silently.
    """
    # A guess of zero (or a negative one) makes the doubling test
    # meaningless -- 0 doubled is still 0, and every parameter would
    # look like a scale. Probe from a strictly positive vector instead.
    probe = [value if value > 0 else 1.0 for value in start]
    baseline = np.asarray(func(x, *probe), dtype=float)
    scale = np.max(np.abs(baseline))
    if not np.isfinite(scale) or scale <= 0:
        return None
    for index, name in enumerate(param_names):
        doubled = list(probe)
        doubled[index] = probe[index] * 2.0
        try:
            candidate = np.asarray(func(x, *doubled), dtype=float)
        except Exception:  # noqa: BLE001 -- a parameter this model cannot take is simply not the amplitude
            continue
        if np.allclose(candidate, 2.0 * baseline, rtol=1e-9, atol=scale * 1e-12):
            return name
    return None


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
        raise ValueError("The data matrix is entirely zero – nothing to resolve.")
    keep = np.flatnonzero(strongest >= threshold * peak)
    if keep.size < 1:
        raise ValueError(
            f"No spectral point reaches {threshold:.1%} of the maximum – lower the threshold."
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
          max_nfev: int = 2000, model=None) -> dict:
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
    model : str or spec dict, optional
        The **curve-fit model** whose decay each component follows --
        a key into
        :data:`opennmr.processing.relaxation.RELAXATION_MODELS`, or a
        spec dict from ``build_custom_model``. Defaults to
        ``DEFAULT_MODELS[kind]``, the plain single
        exponential the method was published with. The model's
        amplitude parameter is pinned at 1 (SCORE's component spectra
        already carry the amplitudes) and every other parameter is
        fitted per component -- so a stretched model gives each
        component its own beta, and an inversion-recovery model its own
        inversion efficiency. See :func:`build_kernel`.
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
    if kind not in DECAY_KINDS:
        raise ValueError(f"Unknown kind '{kind}'. Available: {sorted(DECAY_KINDS)}")
    if method not in RESOLUTION_METHODS:
        raise ValueError(f"Unknown method '{method}'. Available: {sorted(RESOLUTION_METHODS)}")
    x_values = np.asarray(x_values, dtype=float)
    matrix = np.asarray(data, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"data must be 2-D (levels x points), got shape {matrix.shape}.")
    if matrix.shape[0] != x_values.size:
        raise ValueError(
            f"{matrix.shape[0]} rows against {x_values.size} x-values – one per row is required."
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
            f"component(s) – the bilinear model is underdetermined."
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
    # The kernel comes from the curve-fit model registry, so SCORE and
    # Fit Curve evaluate the same function -- see build_kernel. The
    # total signal per level is what the model's own data-driven p0 and
    # bounds callables are evaluated against: it is the decay of the
    # whole series, which is the best single curve available before any
    # component has been separated out.
    total_signal = np.sum(np.abs(matrix), axis=1)
    kernel = build_kernel(model, kind=kind, x_values=x_values, y_values=total_signal)

    seeded_from = None
    if method == "outscore" and initial is None:
        seed_run = score(
            x_values, matrix, n_components=n_components, kind=kind, method="score",
            bounds=bounds, threshold=threshold, non_negative=non_negative,
            non_negative_during_fit=non_negative_during_fit, max_nfev=max_nfev,
            model=model,
        )
        initial = seed_run["constants"]
        seeded_from = seed_run

    rate_index = kernel["params"].index(kernel["rate_param"])
    rate_guesses = _initial_constants(
        x_values, initial, n_components, centre=kernel["p0"][rate_index])

    # One row of free parameters per component. Every parameter starts
    # from the model's own automatic guess; only the rate parameter is
    # spread across decades, because that spread is what gives the
    # optimiser something to tell the components apart by (see
    # _initial_constants).
    guesses = np.tile(np.asarray(kernel["p0"], dtype=float), (n_components, 1))
    guesses[:, rate_index] = rate_guesses

    model_lo, model_hi = kernel["bounds"]
    lo = np.tile(np.asarray(model_lo, dtype=float), (n_components, 1))
    hi = np.tile(np.asarray(model_hi, dtype=float), (n_components, 1))
    if bounds is None:
        # The model's own bounds on a rate parameter are deliberately
        # open-ended (a T2 is bounded only by [1e-9, inf]), which is
        # right for a curve fit with real data behind each point and
        # wrong for a k-component search that has to stay in a region
        # where the components remain distinguishable. Window it around
        # each component's own guess, as this module always has, but
        # never outside what the model itself allows.
        lo[:, rate_index] = np.maximum(rate_guesses / _RATE_DECADES, model_lo[rate_index])
        hi[:, rate_index] = np.minimum(rate_guesses * _RATE_DECADES, model_hi[rate_index])
    else:
        lo[:, rate_index] = np.asarray(bounds[0], dtype=float)
        hi[:, rate_index] = np.asarray(bounds[1], dtype=float)

    log_mask = np.array([name in kernel["log_params"] for name in kernel["params"]], dtype=bool)
    inner_non_negative = non_negative and non_negative_during_fit

    def _pack(values):
        """(k, p) natural parameters -> flat search vector, with the
        log-searched columns in log10 space."""
        out = np.array(values, dtype=float, copy=True)
        out[:, log_mask] = np.log10(np.maximum(out[:, log_mask], _LOG_FLOOR))
        return out.ravel()

    def _unpack(vector):
        """Inverse of :func:`_pack`."""
        out = np.asarray(vector, dtype=float).reshape(n_components, -1).copy()
        out[:, log_mask] = 10.0 ** out[:, log_mask]
        return out

    # Both methods share this inner half exactly: for a trial set of
    # decay parameters, the best spectra follow from a *linear* solve.
    # The log-space columns are searched in log10 for the same reason
    # every diffusion coefficient in this package is (see
    # opennmr.processing.relaxation.LOG_FITTED_PARAMS): they span
    # decades, and a linear search on them converges to the wrong
    # answer even from a good starting point.
    def spectra_for(vector):
        decays = kernel["func"](x_values, _unpack(vector))
        return decays, _solve_spectra(decays, fitted_block, inner_non_negative)

    search_lo, search_hi = _pack(lo), _pack(hi)
    start = np.clip(_pack(guesses), search_lo, search_hi)

    if method == "score":
        def residuals(vector):
            decays, spectra = spectra_for(vector)
            return (fitted_block - decays @ spectra).ravel()

        solution = least_squares(
            residuals, start, bounds=(search_lo, search_hi), max_nfev=max_nfev,
        )
        vector, n_evaluations, success = solution.x, solution.nfev, solution.success
    else:
        def objective(vector):
            _decays, spectra = spectra_for(vector)
            return crosstalk(spectra)

        # Nelder-Mead, not least_squares: the cross-talk objective is a
        # single scalar (not a residual vector), and taking absolute
        # values makes it non-smooth, which a derivative-based method
        # handles badly. With only k parameters a simplex is cheap, and
        # the sharper minimum this objective is chosen for is exactly
        # what a simplex is good at descending into.
        solution = minimize(
            objective, start, method="Nelder-Mead",
            bounds=list(zip(search_lo, search_hi)),
            options={"maxfev": max_nfev, "xatol": 1e-4, "fatol": 1e-10},
        )
        vector, n_evaluations, success = solution.x, solution.nfev, solution.success

    values = _unpack(vector)
    # Slowest-decaying first, so component 1 is the largest/most
    # aggregated species -- the order these are quoted in, and stable
    # across runs regardless of where the optimiser happened to place
    # each one. Ranked by *how much each component has actually decayed*
    # at the largest x rather than by the raw constant, because the two
    # only agree for a rate-like parameter: a larger D decays faster, a
    # larger T2 decays slower, and the model now decides which of those
    # its constant is.
    attenuation = kernel["func"](x_values, values)[-1]
    ranking = np.argsort(-np.asarray(attenuation, dtype=float), kind="stable")
    values = values[ranking]
    constants = values[:, rate_index]

    decays = kernel["func"](x_values, values)
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
        # Which curve-fit model this resolution used, and every
        # component's full parameter set under it -- so a stretched or
        # inversion-recovery kernel's extra per-component parameters
        # (beta, C) are reported rather than silently discarded, and so
        # a result can say which model produced it.
        "model": kernel["name"],
        "model_params": list(kernel["params"]),
        "amplitude_param": kernel["amplitude"],
        "rate_param": kernel["rate_param"],
        "component_params": [dict(zip(kernel["params"], row)) for row in values],
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
        # The model's own rate parameter for a relaxation kernel is a
        # time constant (RELAXATION_MODELS is written in T1/T2, not in
        # rates), so `constants` are seconds and the rate is its
        # reciprocal. Both are reported, as they always were -- which
        # one `constants` itself holds now follows the model rather
        # than this module's own former convention.
        result["T"] = constants
        with np.errstate(divide="ignore"):
            result["rates"] = np.where(constants > 0, 1.0 / constants, np.inf)
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
            f"cover – minimising overlap then pulls away from the truth rather "
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
                f"to that precision – treat neither as settled, and get more "
                f"gradient levels or a wider range before trusting either."
            )
    return warnings


def _initial_constants(x_values, initial, n_components: int, centre=None) -> np.ndarray:
    """Starting decay constants: whatever the caller supplied, else a
    decade-spaced spread around ``centre`` -- the model's own automatic
    guess for its rate parameter (see
    :data:`opennmr.processing.relaxation.RELAXATION_MODELS`), falling
    back to the constant that would decay by 1/e across the sampled
    range when no model guess is available.

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
    if centre is None or not np.isfinite(centre) or centre <= 0:
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
    if result.get("model"):
        # Which curve-fit model produced these constants -- a column of
        # decay constants means something different under a stretched
        # kernel than under a plain exponential, and a spreadsheet has
        # nowhere else to record that.
        writer.writerow([f"Model: {result['model']}"])
    writer.writerow([])
    writer.writerow(["Component", "Decay constant", "Unit", "Share",
                      "R^2", "Cross-talk", "Non-negative"])
    # DECAY_KINDS, not a unit of this function's own: a relaxation
    # kernel's constant is the model's time constant (seconds), which
    # is what the "1/s" this line used to write no longer describes.
    unit = DECAY_KINDS[result["kind"]]["unit"]
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
    model = result.get("model")
    if model:
        # Named outright, because it is the same model the Fit Curve
        # panel is set to -- a resolution and a curve fit of the same
        # series are now the same function, and a report that did not
        # say which one would leave that invisible.
        amplitude = result.get("amplitude_param")
        note = f" (amplitude {amplitude} is the component spectrum)" if amplitude else ""
        lines.append(f"  Model: {model}{note}")
        lines.append("")
    lines.append("  Component        share    decay constant")
    lines.append("  " + "-" * (width - 4))
    extra = [p for p in result.get("model_params", []) if p != result.get("rate_param")]
    for index in range(result["n_components"]):
        share = result["fractions"][index]
        if result["kind"] == "diffusion":
            value = format_diffusion_value(result["constants"][index])
        else:
            value = f"{format_time_value(result['T'][index])} (T)"
        lines.append(f"    {index + 1:<12} {share * 100:6.2f}%    {value}")
        # A stretched or inversion-recovery kernel fits more than one
        # parameter per component; reporting only the rate would hide
        # the beta that is holding up the fit.
        if extra:
            values = result.get("component_params", [{}] * result["n_components"])[index]
            lines.append("                           "
                          + ",  ".join(f"{p} = {values.get(p, float('nan')):.4g}" for p in extra))

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
