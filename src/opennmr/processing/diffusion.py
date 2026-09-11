"""
PFG (pulsed-field-gradient) diffusion support for opennmr.

Three things live here, and nothing else does:

1. **b-value calculation** -- turning a gradient strength, a pair of
   gradient timings and a nucleus into the diffusion weighting
   :math:`b` (s/m^2) that the Stejskal-Tanner models in
   :mod:`opennmr.processing.relaxation` actually fit against.
2. **Gradient calibration** -- recovering a probe's absolute maximum
   gradient from a measurement of a *known* diffusion coefficient
   (the standard "run DI water and solve for G_max" experiment). This
   is the only way to get real b-values out of a spectrometer, like
   X-Pulse, that records gradient strengths as relative fractions of
   an uncalibrated maximum.
3. **N-component diffusion models** -- an arbitrary number of
   independently-diffusing populations, each optionally stretched,
   built on demand rather than enumerated one function per shape.

**Use the vendor's own b-values when they exist.** Bruker's
``diff.xml`` carries TopSpin's own precomputed b-values per gradient
step (see :func:`opennmr.io.bruker.read_diff_xml`), which already
account for the gradient shape, the sequence's own timing corrections
and any pre-emphasis calibration -- none of which are recorded
anywhere this module could read them. :func:`b_value` here is for the
case where there is no such file: X-Pulse, a hand-written sequence, a
vendor that records only gradient amplitudes. It is a *general
formula*, and its answer is exactly as good as the shape factor and
sequence correction it is told to use.

Units, stated once and used consistently
----------------------------------------
======================  ==========================================
Gradient strength       T/m  (1 G/cm = 0.01 T/m -- see
                        :func:`g_per_cm_to_t_per_m`)
delta, DELTA, tau       seconds
gamma                   Hz/T -- what
                        :func:`opennmr.core.registry.find_gamma`
                        returns. Converted to rad/s/T internally by
                        the 2*pi in :func:`b_value`; **do not**
                        pre-multiply it yourself.
b                       s/m^2
D                       m^2/s
======================  ==========================================
"""

from __future__ import annotations

import numpy as np

from ..core.registry import find_gamma

#: Effective-duration scaling for a shaped gradient pulse: a gradient
#: that is not a perfect rectangle dephases less than its nominal
#: amplitude and duration suggest, and the standard treatment folds
#: that into ``delta_effective = delta * shape_factor``.
#:
#: ``sine`` is the analytic value for a half-sine lobe (the integral of
#: ``sin`` over half a period, divided by its peak). ``smsq10`` is
#: Bruker's own published factor for ``SMSQ10.100``, the smoothed
#: square shape used by essentially every stock Bruker diffusion
#: sequence, and is a measured/tabulated number rather than something
#: derivable here. ``rect`` is the textbook Stejskal-Tanner case.
GRADIENT_SHAPE_FACTORS = {
    "rect": 1.0,
    "sine": 2.0 / np.pi,
    "smsq10": 0.9,
    "gauss": 0.6295,
    # Trapezoid does not *have* a single effective-duration factor --
    # its correction is additive rather than multiplicative (see
    # :func:`_timing_term`). 1.0 here so anything iterating this
    # mapping still gets a usable number; b_value itself never reads
    # it for this shape.
    "trapezoid": 1.0,
}

#: Human-readable names for the shapes above, for a GUI combo.
GRADIENT_SHAPE_LABELS = {
    "rect": "Rectangular (ideal)",
    "sine": "Half-sine (SINE.100)",
    "smsq10": "Smoothed square (SMSQ10.100)",
    "gauss": "Gaussian (GAUS)",
    "trapezoid": "Trapezoid (ramped)",
}

#: Shapes whose b-value depends on a **ramp time** as well as on
#: delta/DELTA. Only the trapezoid: everything else here is described
#: by scaling delta to an effective duration, which a ramp cannot be.
SHAPES_WITH_RAMP = frozenset({"trapezoid"})


def _correction_pgse(delta, DELTA, tau):
    """``DELTA - delta/3`` -- the diffusion time of a plain two-pulse
    spin-echo (PGSE) or stimulated-echo (PGSTE) sequence. The
    ``delta/3`` accounts for diffusion *during* each gradient pulse."""
    return DELTA - delta / 3.0


def _correction_bipolar(delta, DELTA, tau):
    """``DELTA - delta/3 - tau/2`` -- a bipolar gradient pair (Bruker's
    ``ledbpgp2s``/``dstebpgp3s``, and every "BPP-LED" variant). The
    extra ``tau/2`` is the recovery delay between the two lobes of
    each bipolar pair, during which no gradient is on but diffusion
    still proceeds. ``delta`` is the **total** gradient duration of the
    pair, not one lobe."""
    return DELTA - delta / 3.0 - tau / 2.0


def _correction_oneshot(delta, DELTA, tau):
    """``DELTA - delta/3 - tau/2`` as for a bipolar pair. Pelta/Morris
    "Oneshot" additionally applies an unbalancing factor to the two
    lobes, whose exact form depends on the ``alpha`` used; that
    refinement is deliberately **not** modelled here (it is a few
    percent, and getting it silently wrong is worse than not offering
    it), so this is the bipolar expression under its own name."""
    return DELTA - delta / 3.0 - tau / 2.0


#: Sequence -> the timing correction multiplying ``(gamma*g*delta)^2``.
#: Each takes ``(delta, DELTA, tau)`` in seconds and returns seconds.
DIFFUSION_SEQUENCES = {
    "pgse": {"label": "Spin echo (PGSE)", "correction": _correction_pgse, "uses_tau": False},
    "pgste": {"label": "Stimulated echo (PGSTE / LED)", "correction": _correction_pgse, "uses_tau": False},
    "bipolar": {"label": "Bipolar pair (BPP-LED, dstebpgp)", "correction": _correction_bipolar, "uses_tau": True},
    "oneshot": {"label": "Oneshot (bipolar, unbalancing ignored)", "correction": _correction_oneshot, "uses_tau": True},
}

#: Absolute maximum gradient (G/cm) of probes whose calibration is a
#: known constant rather than something to be measured. Both are
#: Bruker diffusion probes, carried over from the group's own
#: ``leonmr`` working code where they were inline magic numbers.
#: Anything not listed here has to be calibrated -- see
#: :func:`calibrate_max_gradient`.
PROBE_MAX_GRADIENT_G_PER_CM = {
    "diffBB": 1702.4136387358437,
    "diff50": 2897.6124790318013,
}

#: Literature self-diffusion coefficients (m^2/s) of the standard
#: calibration liquids, at the temperature named in the key. These are
#: the *reference* values a gradient calibration is solved against, so
#: the calibration is only ever as accurate as the sample temperature
#: is actually known -- water's D changes by about 2.4% per degree
#: near room temperature, which is why the temperature is in the key
#: rather than being an assumed 25 C.
CALIBRATION_LIQUIDS = {
    "H2O_25C": {"D": 2.299e-9, "label": "H2O, 25 °C", "nucleus": "1H"},
    "H2O_20C": {"D": 2.025e-9, "label": "H2O, 20 °C", "nucleus": "1H"},
    "D2O_25C": {"D": 1.872e-9, "label": "D2O (residual HDO), 25 °C", "nucleus": "1H"},
    "C6D6_25C": {"D": 2.207e-9, "label": "C6D6, 25 °C", "nucleus": "1H"},
}


# ============================================================
# Unit conversion
# ============================================================

def g_per_cm_to_t_per_m(value):
    """Gauss/cm -> Tesla/m. ``1 G = 1e-4 T`` and ``1 cm = 1e-2 m``, so
    the factor is ``1e-2``. Spelled out rather than folded into a bare
    constant at each call site, so it is checkable at a glance."""
    return np.asarray(value, dtype=float) * 1e-2


def t_per_m_to_g_per_cm(value):
    """Tesla/m -> Gauss/cm. Inverse of :func:`g_per_cm_to_t_per_m`."""
    return np.asarray(value, dtype=float) * 1e2


def gamma_for(nucleus: str) -> float:
    """Gyromagnetic ratio of ``nucleus`` in **Hz/T** -- the unit every
    function here expects. Thin wrapper over
    :func:`opennmr.core.registry.find_gamma` so callers don't have to
    remember which of the several conventions that returns (it is
    Hz/T, from a MHz/T table)."""
    return float(find_gamma(nucleus)["gamma"])


# ============================================================
# b-values
# ============================================================

def _timing_term(shape: str, delta: float, DELTA: float, tau: float, ramp: float,
                 correction) -> float:
    """The purely-temporal part of the b-value, in s^3 -- everything
    except ``(2*pi*gamma*g)^2``.

    Two different shapes of answer live here, and that is why this is
    a function rather than a lookup:

    * **Effective-duration shapes** (rectangular, sine, SMSQ10,
      Gaussian) scale ``delta`` by a factor and are otherwise the plain
      Stejskal-Tanner form: ``(delta*sigma)^2 * C``.
    * **The trapezoid** cannot be written that way. Its correction is
      *additive*: ``delta^2*C + ramp^3/30 - delta*ramp^2/6`` (Price,
      *Concepts Magn. Reson.* 1997, and the form every vendor
      implements). Folding a ramp into an effective duration would be
      an approximation to this, not a rewriting of it.

    ``delta`` for a trapezoid is the **flat top plus one ramp**, which
    is the convention the published expression uses; the term vanishes
    as ``ramp -> 0``, recovering the ideal rectangle exactly.

    ``correction`` is the sequence's own ``C(delta, DELTA, tau)`` (see
    :data:`DIFFUSION_SEQUENCES`). Composing it with the trapezoid's
    additive term is an extension: the published trapezoid expression
    is stated for a plain spin echo, where ``C = DELTA - delta/3``. For
    a bipolar pair the ``-tau/2`` enters the same diffusion-time factor
    the ``delta^2`` multiplies, which is where it belongs; the ramp
    terms are unaffected by it.
    """
    if shape == "trapezoid":
        return delta ** 2 * correction + ramp ** 3 / 30.0 - delta * ramp ** 2 / 6.0
    delta_eff = delta * GRADIENT_SHAPE_FACTORS[shape]
    return delta_eff ** 2 * correction


def b_value(gradient, delta, DELTA, gamma=None, nucleus=None, shape="rect",
            sequence="pgse", tau=0.0, ramp=0.0):
    """The Stejskal-Tanner diffusion weighting

    .. math:: b = (2\\pi\\,\\gamma\\, g\\, \\delta\\, \\sigma)^2 \\cdot C(\\delta, \\Delta, \\tau)

    in s/m^2, for one gradient strength or an array of them.

    Parameters
    ----------
    gradient : float or array
        Gradient amplitude(s) in **T/m**. Use
        :func:`g_per_cm_to_t_per_m` if you have G/cm.
    delta : float
        Gradient pulse duration (s). For a bipolar pair this is the
        **total** duration of the pair.
    DELTA : float
        Diffusion delay (s), gradient-pulse leading edge to leading
        edge.
    gamma : float, optional
        Gyromagnetic ratio in **Hz/T** (not rad/s/T -- the 2*pi is
        applied here). Either this or ``nucleus`` must be given.
    nucleus : str, optional
        Nucleus name (e.g. ``"1H"``) to look ``gamma`` up from, when
        it isn't supplied directly.
    shape : str
        Key into :data:`GRADIENT_SHAPE_FACTORS` -- scales ``delta`` to
        an effective duration. ``"rect"`` (the textbook case) is the
        default; **a stock Bruker sequence is almost always
        ``"smsq10"``**, and using ``"rect"`` for one overestimates b
        by ``1/0.9**2`` = 23%, which lands directly on D.
    sequence : str
        Key into :data:`DIFFUSION_SEQUENCES` -- chooses the timing
        correction. ``"bipolar"`` and ``"oneshot"`` additionally use
        ``tau``.
    tau : float
        Bipolar recovery delay (s). Ignored by the non-bipolar
        sequences.
    ramp : float
        Gradient ramp (rise) time (s), for ``shape="trapezoid"`` only.
        Its correction is additive rather than a scaling of ``delta``
        -- see :func:`_timing_term`. Zero recovers the ideal
        rectangle exactly. For a trapezoid, ``delta`` is the flat top
        **plus one ramp**.

    Returns
    -------
    float or ndarray, matching ``gradient``'s own shape.

    Raises
    ------
    ValueError
        On an unknown ``shape``/``sequence``, a missing gamma, or
        timings that make the correction term non-positive (i.e.
        ``DELTA`` shorter than the gradient pulses it is supposed to
        contain -- an input error, not a small b).
    """
    if shape not in GRADIENT_SHAPE_FACTORS:
        raise ValueError(f"Unknown gradient shape '{shape}'. Available: {sorted(GRADIENT_SHAPE_FACTORS)}")
    if sequence not in DIFFUSION_SEQUENCES:
        raise ValueError(f"Unknown sequence '{sequence}'. Available: {sorted(DIFFUSION_SEQUENCES)}")
    if gamma is None:
        if nucleus is None:
            raise ValueError("b_value needs either gamma (Hz/T) or nucleus.")
        gamma = gamma_for(nucleus)

    delta, DELTA, tau = float(delta), float(DELTA), float(tau)
    correction = DIFFUSION_SEQUENCES[sequence]["correction"](delta, DELTA, tau)
    if correction <= 0:
        raise ValueError(
            f"Diffusion time correction is {correction:.4g} s (<= 0) for delta={delta:g} s, "
            f"DELTA={DELTA:g} s, tau={tau:g} s on sequence '{sequence}'. DELTA must be long "
            f"enough to contain the gradient pulses."
        )

    ramp = float(ramp)
    if ramp < 0:
        raise ValueError(f"ramp must be >= 0, got {ramp:g} s.")
    if shape in SHAPES_WITH_RAMP and ramp > delta:
        raise ValueError(
            f"ramp ({ramp:g} s) cannot exceed delta ({delta:g} s) -- delta is the flat "
            f"top plus one ramp, so a longer ramp describes no real pulse."
        )
    timing = _timing_term(shape, delta, DELTA, tau, ramp, correction)
    if timing <= 0:
        raise ValueError(
            f"The timing term is {timing:.4g} s^3 (<= 0) for delta={delta:g} s, "
            f"DELTA={DELTA:g} s, ramp={ramp:g} s on shape '{shape}'. Check the timings."
        )
    gradient = np.asarray(gradient, dtype=float)
    b = (2.0 * np.pi * gamma * gradient) ** 2 * timing
    return float(b) if b.ndim == 0 else b


def b_values_from_relative(relative, max_gradient, delta, DELTA, gamma=None, nucleus=None,
                           shape="rect", sequence="pgse", tau=0.0, ramp=0.0,
                           max_gradient_unit="T/m"):
    """b-values from *relative* gradient strengths (0-1 fractions, or
    0-100 percentages) plus an absolute maximum.

    This is the X-Pulse case, and the reason this module exists: an
    X-Pulse PFG series records only ``G1_valueslist.txt``'s relative
    fractions (see
    :func:`opennmr.io.xpulse._read_gradient_relative`), with the
    probe's absolute maximum recorded nowhere in the data. Supply it
    from :data:`PROBE_MAX_GRADIENT_G_PER_CM`, from the vendor's
    specification, or -- the honest option -- from
    :func:`calibrate_max_gradient` against a known standard.

    ``relative`` values above 1.0 are read as percentages and divided
    by 100, so either convention works; a list mixing the two is not
    something this can detect and will be read as percentages.

    ``max_gradient_unit`` : ``"T/m"`` (default) or ``"G/cm"``.
    """
    relative = np.asarray(relative, dtype=float)
    if relative.size and np.nanmax(np.abs(relative)) > 1.0:
        relative = relative / 100.0
    if max_gradient_unit == "G/cm":
        max_gradient = g_per_cm_to_t_per_m(max_gradient)
    elif max_gradient_unit != "T/m":
        raise ValueError(f"max_gradient_unit must be 'T/m' or 'G/cm', got {max_gradient_unit!r}")
    return b_value(relative * float(max_gradient), delta, DELTA, gamma=gamma, nucleus=nucleus,
                   shape=shape, sequence=sequence, tau=tau, ramp=ramp)


# ============================================================
# Gradient calibration
# ============================================================

def calibrate_max_gradient(relative, intensities, known_D, delta, DELTA, gamma=None, nucleus="1H",
                           shape="rect", sequence="pgse", tau=0.0, ramp=0.0,
                           g_max_guess_t_per_m=1.0, stretched=False):
    """Solve for a probe's absolute maximum gradient from a diffusion
    series on a sample whose ``D`` is already known.

    The standard calibration experiment, and the missing piece for any
    spectrometer that reports gradients only as fractions of an
    uncalibrated maximum. Run e.g. DI water, hold ``D`` fixed at its
    literature value (:data:`CALIBRATION_LIQUIDS`), and let the fit
    find the one remaining unknown -- the ``G_max`` that makes the
    observed attenuation match.

    Note what is fixed and what is free: ``D`` is **fixed**, ``G_max``
    and the amplitude ``I0`` are free. Fitting D and G_max together is
    exactly degenerate -- they enter only as the product
    ``D * G_max^2`` -- so a calibration that leaves both free is not a
    calibration, it is an unconstrained curve through the data. That
    degeneracy is why this function exists as its own thing rather
    than being a normal fit with a different guess.

    Parameters
    ----------
    relative : array
        Relative gradient strengths (0-1, or 0-100 -- see
        :func:`b_values_from_relative`).
    intensities : array
        Signal at each gradient step. Normalised internally, so the
        absolute scale is irrelevant.
    known_D : float
        The standard's literature diffusion coefficient (m^2/s), e.g.
        ``CALIBRATION_LIQUIDS["H2O_25C"]["D"]``.
    delta, DELTA, tau, shape, sequence : as :func:`b_value`.
    gamma, nucleus : as :func:`b_value`; defaults to ``"1H"``, which
        is what every practical calibration standard is observed on.
    g_max_guess_t_per_m : float
        Starting guess for the maximum gradient (T/m). The default of
        1 T/m (100 G/cm) is a mid-range high-resolution probe.
    stretched : bool
        Fit a stretched exponential instead. Off by default: a clean
        calibration standard is monodisperse, and a free stretch
        exponent will happily absorb a gradient miscalibration into
        itself, which defeats the purpose.

    Returns
    -------
    dict with ``max_gradient_t_per_m``, ``max_gradient_g_per_cm``,
    their 1-sigma errors, the fitted ``I0``/``beta``, the implied
    ``b`` array, and ``fit`` -- the full
    :func:`opennmr.processing.relaxation.fit_relaxation`-shaped result
    (so :func:`~opennmr.processing.relaxation.format_relaxation_report`
    and the uncertainty machinery apply to it unchanged).
    """
    from scipy.optimize import curve_fit

    relative = np.asarray(relative, dtype=float)
    if relative.size and np.nanmax(np.abs(relative)) > 1.0:
        relative = relative / 100.0
    intensities = np.asarray(intensities, dtype=float)
    if relative.size != intensities.size:
        raise ValueError(
            f"Got {relative.size} gradient values and {intensities.size} intensities -- "
            f"a calibration needs one intensity per gradient step."
        )
    if relative.size < 3:
        raise ValueError("A gradient calibration needs at least 3 points.")

    peak = np.nanmax(np.abs(intensities))
    y = intensities / peak if peak else intensities

    if gamma is None:
        gamma = gamma_for(nucleus)

    # b per unit G_max^2, so the fit's only gradient-dependent unknown
    # is a single scalar multiplying a fixed array. Computed once via
    # b_value itself rather than re-deriving the formula here, so the
    # shape factor and sequence correction cannot drift apart from the
    # rest of this module.
    b_per_gmax_squared = b_value(relative, delta, DELTA, gamma=gamma, shape=shape,
                                 sequence=sequence, tau=tau, ramp=ramp)

    # Two separate model functions rather than one with beta bounded to
    # a single point: curve_fit cannot handle equal lower and upper
    # bounds, so "fixed at 1" has to mean "not a parameter at all".
    if stretched:
        def model(_x, I0, g_max, beta):
            b = b_per_gmax_squared * g_max ** 2
            return I0 * np.exp(-np.power(np.clip(known_D * b, 0.0, None), beta))

        p0 = [1.0, float(g_max_guess_t_per_m), 1.0]
        lo, hi = [0.0, 1e-4, 0.05], [np.inf, 1e4, 2.0]
    else:
        def model(_x, I0, g_max):
            b = b_per_gmax_squared * g_max ** 2
            return I0 * np.exp(-known_D * b)

        p0 = [1.0, float(g_max_guess_t_per_m)]
        lo, hi = [0.0, 1e-4], [np.inf, 1e4]

    popt, pcov = curve_fit(model, relative, y, p0=p0, bounds=(lo, hi), x_scale="jac", maxfev=20000)
    perr = np.sqrt(np.diag(pcov))

    i0, g_max = float(popt[0]), float(popt[1])
    beta = float(popt[2]) if stretched else 1.0
    g_max_err = float(perr[1])

    y_fit = model(relative, *popt)
    residual = y - y_fit
    ss_res = float(np.sum(residual ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))

    return {
        "max_gradient_t_per_m": g_max,
        "max_gradient_t_per_m_error": g_max_err,
        "max_gradient_g_per_cm": float(t_per_m_to_g_per_cm(g_max)),
        "max_gradient_g_per_cm_error": float(t_per_m_to_g_per_cm(g_max_err)),
        "I0": i0, "beta": beta, "known_D": float(known_D),
        "b": b_per_gmax_squared * g_max ** 2,
        "relative": relative, "y": y, "y_fit": y_fit,
        "r_squared": 1.0 - ss_res / ss_tot if ss_tot else float("nan"),
        "shape": shape, "sequence": sequence,
        "delta": float(delta), "DELTA": float(DELTA), "tau": float(tau),
    }


# ============================================================
# N-component diffusion models
# ============================================================

def diffusion_component_names(n_components: int, stretched=False) -> list[str]:
    """Parameter names of an ``n_components`` diffusion model, in fit
    order. One component is ``I0``/``D`` (matching the long-standing
    ``stejskal_tanner`` model's own names, so a single-component
    generalised fit is a drop-in for it); more are numbered from 1."""
    stretched = _stretch_flags(n_components, stretched)
    if n_components == 1:
        names = ["I0", "D"] + (["beta"] if stretched[0] else [])
        return names
    names = []
    for i in range(n_components):
        names.extend([f"I{i + 1}", f"D{i + 1}"])
        if stretched[i]:
            names.append(f"beta{i + 1}")
    return names


def _stretch_flags(n_components: int, stretched) -> list[bool]:
    """``stretched`` as one bool per component. Accepts a single bool
    (applied to every component) or a per-component sequence, matching
    the ``stretch=[...]`` convention of the group's own ``DiffFit``
    working code that this generalises."""
    if isinstance(stretched, (bool, np.bool_)):
        return [bool(stretched)] * n_components
    flags = [bool(s) for s in stretched]
    if len(flags) != n_components:
        raise ValueError(
            f"stretched has {len(flags)} entries for {n_components} component(s) -- "
            f"give one flag per component, or a single bool for all of them."
        )
    return flags


def build_diffusion_model(n_components: int = 1, stretched=False, expected_D=None,
                          d_bounds_factor: float = 100.0) -> dict:
    """A :data:`~opennmr.processing.relaxation.RELAXATION_MODELS`-shaped
    spec for a sum of ``n_components`` Stejskal-Tanner decays, each
    optionally stretched:

    .. math:: I(b) = \\sum_i I_i \\exp\\left[-(D_i b)^{\\beta_i}\\right]

    The generalisation of the three hand-written ``stejskal_tanner*``
    entries in that registry, which cover exactly (1 component), (1
    component, stretched) and (2 components, neither stretched). Those
    stay for compatibility; anything else -- three components, a
    two-component fit with only the slow one stretched -- comes from
    here. Feed the result to
    :func:`~opennmr.processing.relaxation.fit_relaxation` as
    ``custom_model``.

    Parameters
    ----------
    n_components : int
        Number of independently-diffusing populations (>= 1).
    stretched : bool or sequence of bool
        Whether each component carries its own stretch exponent
        ``beta``. A stretched component describes a *distribution* of
        diffusion coefficients around ``D``; a separate component
        describes a genuinely distinct population. They are different
        claims about the sample, which is why both are offered rather
        than one standing in for the other.
    expected_D : float or sequence of float, optional
        Starting guess(es) for each ``D`` (m^2/s). **This is the
        parameter worth supplying**: a multi-component diffusion fit is
        strongly guess-dependent, and the components have to start
        genuinely separated or the fit collapses them onto each other.
        When omitted, guesses are spread over decades around the value
        that would decay by ``1/e`` across the sampled b-range.
    d_bounds_factor : float
        Each ``D`` is bounded to ``[guess/factor, guess*factor]``.
        Anchoring the bounds to the guess (rather than leaving them
        unbounded) is what keeps two components from swapping places
        or converging to the same value; it is the convention the
        group's own working ``DiffFit`` code uses, widened here from
        its 100x to a symmetric factor.

    Returns
    -------
    dict with ``func``, ``params``, ``p0``, ``bounds``, ``rate_param``
    and ``n_components``/``stretched`` bookkeeping.
    """
    if n_components < 1:
        raise ValueError(f"n_components must be >= 1, got {n_components}")
    flags = _stretch_flags(n_components, stretched)
    param_names = diffusion_component_names(n_components, flags)
    # Validated here, not lazily inside p0/bounds: an expected_D of the
    # wrong length is a mistake in the *call*, and surfacing it from
    # inside a fit several steps later makes it far harder to place.
    _component_d_guesses(np.array([1.0]), expected_D, n_components)

    def func(b, *values):
        b = np.asarray(b, dtype=float)
        lookup = dict(zip(param_names, values))
        total = np.zeros_like(b, dtype=float)
        for i in range(n_components):
            if n_components == 1:
                amp, d = lookup["I0"], lookup["D"]
                beta = lookup.get("beta", 1.0)
            else:
                amp, d = lookup[f"I{i + 1}"], lookup[f"D{i + 1}"]
                beta = lookup.get(f"beta{i + 1}", 1.0)
            total = total + amp * np.exp(-np.power(np.clip(d * b, 0.0, None), beta))
        return total

    def guesses(b, y):
        b = np.asarray(b, dtype=float)
        amp = float(np.nanmax(np.abs(y))) if np.size(y) else 1.0
        amp = amp if amp > 0 else 1.0
        return _component_d_guesses(b, expected_D, n_components), amp / n_components

    def p0(b, y):
        d_guesses, amp = guesses(b, y)
        values = []
        for i in range(n_components):
            values.extend([amp, d_guesses[i]])
            if flags[i]:
                values.append(0.9)
        return values

    def bounds(b, y):
        d_guesses, _amp = guesses(b, y)
        lo, hi = [], []
        for i in range(n_components):
            lo.extend([0.0, d_guesses[i] / d_bounds_factor])
            hi.extend([np.inf, d_guesses[i] * d_bounds_factor])
            if flags[i]:
                lo.append(0.05)
                hi.append(2.0)
        return (lo, hi)

    return {
        "func": func, "params": param_names, "p0": p0, "bounds": bounds,
        # One component has an unambiguous "the" diffusion coefficient
        # to headline a report with; several do not, and naming one of
        # them would be picking a winner arbitrarily.
        "rate_param": "D" if n_components == 1 else None,
        # Every D is searched in log10 space -- see
        # opennmr.processing.relaxation.LOG_FITTED_PARAMS for the
        # measurements that made this non-optional for more than one
        # component. The bounds and guesses above stay in m^2/s.
        "log_params": tuple(name for name in param_names if name == "D" or
                            (name.startswith("D") and name[1:].isdigit())),
        "n_components": n_components, "stretched": flags,
        "kind": "diffusion",
    }


def _component_d_guesses(b, expected_D, n_components: int) -> list[float]:
    """One starting ``D`` per component, either from what the caller
    supplied or spread across decades around the b-range's own natural
    scale.

    The spread matters more than the centre. Two components started at
    similar values give ``curve_fit`` nothing to tell them apart, and
    it reliably converges with both sitting on the same D and the
    amplitudes split arbitrarily between them -- a fit that looks
    converged and means nothing. A decade of separation per component
    is enough for the residual to actually distinguish them."""
    def spread(centre: float) -> list[float]:
        # Descending, so component 1 is the fastest-diffusing: that is
        # the order a multi-component result is quoted in, and having
        # the numbering agree with it means a report reads top-to-bottom
        # from small/mobile to large/aggregated without reordering.
        return [centre * 10.0 ** ((n_components - 1) / 2.0 - i) for i in range(n_components)]

    if expected_D is not None:
        values = np.atleast_1d(np.asarray(expected_D, dtype=float))
        if values.size == n_components:
            return [float(v) for v in values]
        if values.size == 1:
            return spread(float(values[0]))
        raise ValueError(
            f"expected_D has {values.size} entries for {n_components} component(s) -- "
            f"give one per component, or a single value to spread around."
        )
    b = np.asarray(b, dtype=float)
    positive = b[b > 0]
    b_max = float(np.max(positive)) if positive.size else 1.0
    return spread(1.0 / b_max)


def component_weights(params: dict, n_components: int) -> list[float]:
    """Each component's amplitude as a fraction of the total, which is
    how a multi-component diffusion result is actually quoted ("70% at
    2.1e-10, 30% at 4.5e-12") -- the raw ``I`` values depend on
    whatever arbitrary intensity scale the spectra were on and are not
    comparable between datasets. Matches the normalisation the group's
    own ``DiffFit`` applies to its reported ``I``."""
    if n_components == 1:
        return [1.0]
    amplitudes = [float(params.get(f"I{i + 1}", 0.0)) for i in range(n_components)]
    total = sum(amplitudes)
    if not total:
        return [float("nan")] * n_components
    return [a / total for a in amplitudes]


# ============================================================
# Pulling the timings out of an acquisition file
# ============================================================

#: Where each PFG timing lives, per vendor, and what unit the file
#: records it in. Values are ``(metadata.raw sub-dict, key, seconds
#: per recorded unit)``.
#:
#: X-Pulse records these in its own ``.par`` file with self-describing
#: names -- ``delta`` ("Gradient Duration [us] -- Stejskal-Tanner
#: delta (plateau only, ramps separate)"), ``DELTA`` ("Diffusion Time
#: [us]") and ``D70`` ("Gradient Ramp Time [us]"). Bruker keeps the
#: equivalents in ``diff.xml``, which :func:`opennmr.io.bruker.read_diff_xml`
#: already reads -- and which additionally carries TopSpin's own
#: precomputed b-values, so nothing here needs to reconstruct them.
_PFG_PARAM_SOURCES = {
    "xpulse": {
        "delta": ("exp_params", "delta", 1e-6),
        "DELTA": ("exp_params", "DELTA", 1e-6),
        "ramp": ("exp_params", "D70", 1e-6),
        "gradient_fraction": ("exp_params", "G1", 1.0),
    },
}


def pfg_params_from_metadata(spectrum) -> dict:
    """The PFG timings this experiment's own acquisition file records,
    in **seconds**, or ``{}`` where the vendor records none.

    Returns any of ``delta``, ``DELTA``, ``ramp`` and
    ``gradient_fraction`` it can find, plus ``delta_is_plateau_only``
    saying how ``delta`` should be read.

    **That last flag is the trap this function exists to close.**
    X-Pulse records ``delta`` as the gradient *plateau*, with the ramps
    counted separately; the trapezoid b-value expression wants the
    plateau **plus one ramp** (see :func:`_timing_term`). Reading the
    file's number straight into a "delta" box would therefore
    understate the diffusion weighting on exactly the sequences whose
    ramps matter enough to model. :func:`effective_delta` does that
    arithmetic in one place rather than leaving it to each caller.

    Nothing is guessed: a value absent from the file is absent from the
    result, so a caller can tell "the file says 4 ms" from "the file
    does not say".
    """
    sources = _PFG_PARAM_SOURCES.get(getattr(spectrum.metadata, "vendor", None))
    if not sources:
        return {}
    raw = spectrum.metadata.raw or {}
    found: dict = {}
    for name, (bucket, key, scale) in sources.items():
        value = (raw.get(bucket) or {}).get(key)
        if value is None:
            continue
        try:
            found[name] = float(value) * scale
        except (TypeError, ValueError):
            continue
    if found:
        # Only X-Pulse is described here, and its own file says so in
        # the parameter's own comment. A vendor that records the total
        # instead would set this False.
        found["delta_is_plateau_only"] = True
    return found


def effective_delta(delta: float, ramp: float = 0.0, plateau_only: bool = False) -> float:
    """The ``delta`` the b-value expressions want, from what the file
    recorded.

    A trapezoid's ``delta`` in :func:`b_value` is the flat top **plus
    one ramp**. Where the acquisition file records the plateau alone
    (``plateau_only``), the ramp has to be added back; where it records
    the total already, it must not be, or the ramp is counted twice.

    One function so that arithmetic happens once, correctly, instead of
    at each place a timing is read.
    """
    return float(delta) + (float(ramp) if plateau_only else 0.0)
