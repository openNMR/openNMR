"""
Fit first-order J-coupled multiplets to experimental 1D NMR spectra,
using the multiplet simulator in :mod:`nmrtools.simulations`
(:func:`nmrtools.simulations.jcoupled_spectrum`) as the fit model and
the shared baseline/region-masking machinery from
:mod:`nmrtools.fitting`.

Workflow
--------

    NMR1dimport -> fit_jcoupling_spectrum -> NMR1dplot_with_jcoupling_fit
                                            -> print_jcoupling_fit_report

or, end-to-end, :func:`NMRJCouplingFitWorkflow`.

Like :mod:`nmrtools.csa_fitting`, and unlike the closed-form Gaussian/
Lorentzian/pseudo-Voigt peaks in :mod:`nmrtools.fitting`, a J-coupled
multiplet is only defined here via a stick pattern (built from the
coupling topology + current J values) convolved with a broadening
kernel -- see :func:`jcoupling_site_lineshape`. Every fit iteration
therefore re-simulates the full multiplet, not just evaluates a
closed-form peak.

What is (and isn't) fitted
---------------------------

For each chemical site you supply a fixed *coupling topology* -- which
partner nuclide(s) it's coupled to and how many equivalent nuclei of
each (``nuclide``, ``n``) -- plus a starting guess for each group's J
(Hz). The topology (``nuclide``, ``n``, and the partner's spin/natural
abundance, looked up from :func:`nmrtools.utils.find_isotope_info`
unless overridden) is chemical/structural information and is always
held fixed; only the *continuous* parameters -- each group's J (Hz),
the site's chemical shift, linewidth, and amplitude -- are fitted.
This mirrors how a real multiplet analysis works: you already know
(from the molecule) that a CH is coupled to, say, 2 equivalent 1H and
1 31P; what you're fitting is *how big* each of those couplings is.

Because :func:`nmrtools.simulations.spin_multiplet_pattern` places
lines at frequency offsets that are symmetric about zero (``m*J`` for
``m`` running symmetrically over ``-n*I ... +n*I``), a first-order
multiplet is identical for ``+J`` and ``-J`` -- the *sign* of J is not
identifiable from a simple multiplet's lineshape (only its magnitude,
i.e. the line spacing, is). ``J`` is therefore bounded ``>= 0`` by
default (see :func:`build_jcoupling_bounds`) to keep the optimiser
from wandering pointlessly across that sign-degenerate valley; pass
an explicit ``bounds_J`` on a coupling group if you have independent
reason to allow/require negative J.

Multi-site fits
----------------

``p0`` may describe more than one chemical site: pass a list of dicts
(one per site) instead of a single dict, e.g.::

    p0 = [
        {"label": "CH2", "center_ppm": 40.0, "fwhm": 1.5,
         "couplings": [{"nuclide": "1H", "n": 3, "J_Hz": 7.0}]},
        {"label": "CH",  "center_ppm": 100.0, "fwhm": 1.0,
         "couplings": [
             {"nuclide": "1H",  "n": 2, "J_Hz": 7.0},
             {"nuclide": "31P", "n": 1, "J_Hz": 140.0},
         ]},
    ]

Each site gets its own chemical shift, linewidth, amplitude, and one
J per coupling group; all sites share a single linear baseline. A bare
dict is still accepted for backward compatibility and is treated as a
one-site list. Per-parameter ``fixed``/``bounds`` overrides go inline
on the site (``site["fixed"]``/``site["bounds"]``, keyed by
"center_ppm"/"fwhm"/"amplitude") or on the individual coupling group
(``group["fixed_J"]``/``group["bounds_J"]``) -- see
:func:`initialise_jcoupling_params`/:func:`build_jcoupling_bounds`.

Note on cost and identifiability: each additional coupling group adds
another set of stick positions (the pattern grows combinatorially with
the number of groups and their ``n``), and overlapping multiplets from
different sites make amplitude/J parameters correlated, the same way
overlapping CSA sites do in :mod:`nmrtools.csa_fitting`. Good initial J
guesses (read off the peak spacing by eye first) matter a lot here,
more than for a CSA fit, since a multiplet pattern can alias onto a
similar-looking pattern at a very different J if the optimiser starts
too far off. A site with more than one coupling group is the case
most worth giving a tight ``bounds["center_ppm"]`` window (e.g. +/-1
ppm around the shift you read off the spectrum): with an unbounded
centre, a poor J starting guess can otherwise let the whole site drift
off to a completely different, unphysical shift while chasing a
marginally better residual elsewhere on the axis -- ordinary
curve-fitting risk, not something specific to J-coupling, but easy to
hit here because the model is cheap to re-evaluate anywhere on the axis.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from .utils import find_gamma, NUC_label, find_isotope_info
from .plotting import setup_plotting, apply_tick_style, apply_spine_style, save_figure
from .one_d import NMR1dimport
from .fitting import build_region_mask, estimate_baseline_robust, fit_quality, expand_pcov
from .simulations import jcoupled_spectrum, combine_jcouplings, multiplet_lineshape  # noqa: F401
from ..core import fit_progress as _fit_progress

# numpy >= 2.0 renamed trapz -> trapezoid
_trapz = getattr(np, "trapezoid", None) or np.trapz

_SITE_SCALAR_NAMES = ("center_ppm", "fwhm", "amplitude")


# ============================================================
# Peak-definition / per-site option handling
# ============================================================

def _resolve_group_topology(group):
    """
    Turn one coupling-group dict from ``p0`` into its fixed structural
    metadata (nuclide, n, spin, abundance) -- looked up via
    :func:`nmrtools.utils.find_isotope_info` unless ``spin``/
    ``abundance`` are given explicitly on the group -- plus an optional
    ``bond_label`` (e.g. ``"3J(F-H)"``) carried through unchanged for
    :func:`jcoupling_chemist_notation`.
    """
    if "nuclide" not in group or "n" not in group or "J_Hz" not in group:
        raise ValueError(
            "Each coupling group needs 'nuclide', 'n', and an initial "
            f"'J_Hz'; got {group!r}."
        )
    spin = group.get("spin")
    abundance = group.get("abundance")
    all_isotopes = bool(group.get("all_isotopes", False))
    if all_isotopes and abundance is not None:
        raise ValueError(f"Coupling group {group!r}: 'all_isotopes' uses natural abundances; "
                         f"do not also give 'abundance'.")
    if spin is None or abundance is None:
        info = find_isotope_info(group["nuclide"])
        spin = info["spin"] if spin is None else spin
        abundance = info["abundance"] if abundance is None else abundance
    return {
        "nuclide": group["nuclide"], "n": int(group["n"]),
        "spin": float(spin), "abundance": float(abundance),
        "bond_label": group.get("bond_label"),
        # J is to `nuclide`; the element's other magnetic isotopes follow
        # with gamma-scaled J (simulations.jcoupling_group_pattern).
        "all_isotopes": all_isotopes,
    }


def _normalise_jcoupling_sites(p0):
    """
    Accept ``p0`` as a single dict (one site) or a list/tuple of dicts
    (one per site) and return ``(sites, labels, topologies)``:

    - ``sites`` : the sites as given (still holding each group's
      ``J_Hz`` initial guess and any ``fixed``/``bounds`` overrides).
    - ``labels`` : unique, human-readable site labels.
    - ``topologies`` : per-site list of fixed structural coupling-group
      dicts (``nuclide``, ``n``, ``spin``, ``abundance``), in the same
      order as each site's ``couplings``.
    """
    if isinstance(p0, dict):
        p0 = [p0]
    if not isinstance(p0, (list, tuple)) or len(p0) == 0:
        raise TypeError(
            "p0 must be a dict (single site) or a non-empty list of dicts "
            "(one per site)."
        )

    sites, labels, topologies = [], [], []
    required = ("center_ppm", "fwhm", "couplings")
    for i, site in enumerate(p0):
        if not isinstance(site, dict):
            raise TypeError(f"p0[{i}] must be a dict of site parameters, got {type(site)}.")
        missing = [name for name in required if name not in site]
        if missing:
            raise ValueError(
                f"p0[{i}] (label={site.get('label')!r}) is missing required "
                f"key(s): {missing}."
            )
        if len(site["couplings"]) == 0:
            raise ValueError(
                f"p0[{i}] (label={site.get('label')!r}) has an empty "
                f"'couplings' list -- for an uncoupled singlet, this "
                f"module has nothing to do; use ordinary peak fitting "
                f"(nmrtools.fitting.fit_spectrum) instead."
            )
        sites.append(site)
        labels.append(site.get("label", f"Site {i + 1}"))
        topologies.append([_resolve_group_topology(g) for g in site["couplings"]])

    if len(set(labels)) != len(labels):
        raise ValueError(f"Site labels must be unique, got: {labels}")

    return sites, labels, topologies


def _site_param_names(label, topology):
    """Parameter names for one site's block: center, one J per
    coupling group (labelled with its partner nuclide/count for
    readability), fwhm, amplitude."""
    names = [f"{label}: center_ppm"]
    for i, group in enumerate(topology):
        names.append(f"{label}: J_{i + 1} ({group['n']}x{group['nuclide']})")
    names.append(f"{label}: fwhm")
    names.append(f"{label}: amplitude")
    return names


def _all_param_names(labels, topologies):
    names = []
    for label, topology in zip(labels, topologies):
        names.extend(_site_param_names(label, topology))
    names.extend(["baseline: b0", "baseline: b1"])
    return names


# ============================================================
# Model
# ============================================================

def jcoupling_site_lineshape(x_ppm_sorted, center_ppm, J_values, fwhm, larmor_Hz,
                              topology, lineshape="lorentzian", eta=0.0,
                              intensity_threshold=1e-8):
    """
    A single site's J-coupled multiplet on ``x_ppm_sorted``, scaled to
    height 1 at its tallest line -- the per-site building block
    :func:`build_jcoupling_model` sums (scaled by amplitude) across sites.

    Every line is an exact closed-form profile at its own position
    (:func:`nmrtools.simulations.multiplet_lineshape`), so the points
    ``x_ppm_sorted`` need not be sorted, evenly spaced or contiguous,
    and the result at a given ppm does not depend on which other points
    are evaluated. That makes "amplitude" mean the same thing wherever
    the model is evaluated: the multiplet's height at its tallest line
    position. (Before 2026-09-24 the sticks were placed on the axis grid
    and the result divided by its maximum *on that axis*, so the fitted
    amplitude, and every component or integral derived from it, changed
    with the axis the model was evaluated on -- the report, which
    evaluates on the fit region only, could disagree with the fit.)

    Parameters
    ----------
    center_ppm : float
        This site's uncoupled chemical shift.
    J_values : sequence of float
        One J (Hz) per entry in ``topology``, same order.
    fwhm : float
        Linewidth (ppm) of each individual multiplet line.
    larmor_Hz : float
        Observe frequency of the fitted nucleus, in Hz -- converts
        each line's J-coupling offset (Hz) to ppm.
    topology : list of dict
        Fixed structural coupling-group metadata for this site, as
        returned by :func:`_normalise_jcoupling_sites` (``nuclide``,
        ``n``, ``spin``, ``abundance`` per group).
    lineshape, eta, intensity_threshold :
        See :func:`nmrtools.simulations.jcoupled_spectrum`.

    Returns
    -------
    ndarray, same shape as ``x_ppm_sorted``.
    """
    couplings = [
        ({"nuclide": group["nuclide"], "n": group["n"], "J_Hz": J, "all_isotopes": True}
         if group.get("all_isotopes") else
         {"nuclide": group["nuclide"], "n": group["n"], "J_Hz": J,
          "spin": group["spin"], "abundance": group["abundance"]})
        for group, J in zip(topology, J_values)
    ]
    freqs_Hz, weights = combine_jcouplings(couplings, intensity_threshold=intensity_threshold)
    sticks = center_ppm + freqs_Hz / larmor_Hz * 1.0e6
    at_sticks = multiplet_lineshape(sticks, sticks, weights, fwhm, lineshape=lineshape,
                                    eta=eta, normalise="peak")
    height = float(np.max(at_sticks)) if at_sticks.size else 0.0
    shape = multiplet_lineshape(x_ppm_sorted, sticks, weights, fwhm, lineshape=lineshape,
                                eta=eta, normalise="peak")
    return shape / height if height > 0 else shape


def build_jcoupling_model(x_ppm_sorted, params, site_slices, topologies, larmor_Hz,
                           lineshape="lorentzian", eta=0.0, x_ref=None,
                           intensity_threshold=1e-8):
    """
    Full multi-site model:
    ``sum_i(amplitude_i * jcoupling_site_lineshape(..., site_i)) + b0 + b1*(x - x_ref)``.

    Parameters
    ----------
    params : ndarray
        Flat parameter vector: for each site (in order) ``(center_ppm,
        J_1, ..., J_k, fwhm, amplitude)`` -- ``k`` = that site's number
        of coupling groups, so unlike a CSA fit's fixed 6-per-site
        stride, each site's block length depends on its topology --
        followed by the shared baseline ``(b0, b1)``.
    site_slices : list of (int, int)
        ``(start, end)`` index into ``params`` for each site's block,
        as returned by :func:`initialise_jcoupling_params`.
    topologies : list of list of dict
        Per-site fixed coupling-group metadata, as returned by
        :func:`_normalise_jcoupling_sites`.
    larmor_Hz, lineshape, eta, intensity_threshold :
        Shared by every site -- see :func:`jcoupling_site_lineshape`.
    """
    params = np.asarray(params, dtype=float)
    x_ppm_sorted = np.asarray(x_ppm_sorted)
    x_ref = np.mean(x_ppm_sorted) if x_ref is None else x_ref

    total = np.zeros_like(x_ppm_sorted, dtype=float)
    for (start, end), topology in zip(site_slices, topologies):
        block = params[start:end]
        n_groups = len(topology)
        center_ppm = block[0]
        J_values = block[1:1 + n_groups]
        fwhm = block[1 + n_groups]
        amplitude = block[2 + n_groups]
        shape = jcoupling_site_lineshape(
            x_ppm_sorted, center_ppm, J_values, fwhm, larmor_Hz, topology,
            lineshape=lineshape, eta=eta, intensity_threshold=intensity_threshold,
        )
        total += amplitude * shape

    b0, b1 = params[-2], params[-1]
    return total + b0 + b1 * (x_ppm_sorted - x_ref)


# ============================================================
# Parameter initialisation / bounds
# ============================================================

def initialise_jcoupling_params(sites, labels, topologies, b0_init, b1_init,
                                 x=None, y=None,
                                 positive_intensity=True, allow_negative_intensity=False):
    """
    Build the initial parameter vector ``p0``, the boolean ``is_free``
    mask, and each site's ``(start, end)`` block slice.

    An ``amplitude`` not given explicitly on a site is estimated as
    ``max(y[nearest x to center_ppm], 0.0)`` when ``x``/``y`` are
    given (floored to 0 if the fit locks amplitudes positive), else 1.0.
    """
    amplitude_locked_positive = positive_intensity and not allow_negative_intensity

    p0, is_free, site_slices = [], [], []
    idx = 0

    for site, topology in zip(sites, topologies):

        start = idx
        fixed = site.get("fixed", {})

        p0.append(float(site["center_ppm"]))
        is_free.append(not fixed.get("center_ppm", False))
        idx += 1

        for group in site["couplings"]:
            p0.append(float(group["J_Hz"]))
            is_free.append(not group.get("fixed_J", False))
            idx += 1

        p0.append(float(site["fwhm"]))
        is_free.append(not fixed.get("fwhm", False))
        idx += 1

        if "amplitude" in site:
            amp = float(site["amplitude"])
        elif x is not None and y is not None:
            i_near = int(np.argmin(np.abs(x - site["center_ppm"])))
            amp = float(y[i_near])
            if amp < 0 and amplitude_locked_positive:
                amp = 0.0
        else:
            amp = 1.0
        p0.append(amp)
        is_free.append(not fixed.get("amplitude", False))
        idx += 1

        site_slices.append((start, idx))

    p0.extend([b0_init, b1_init])
    is_free.extend([True, True])

    return np.array(p0, dtype=float), np.array(is_free, dtype=bool), site_slices


def build_jcoupling_bounds(sites, topologies, positive_intensity=True,
                            allow_negative_intensity=False):
    """
    Construct lower/upper parameter bounds for ``scipy.optimize.curve_fit``.

    Defaults: ``center_ppm`` unbounded; each J bounded ``(0, inf)``
    (see the module docstring for why -- sign is not identifiable from
    a first-order multiplet); ``fwhm`` bounded ``(1e-6, inf)``;
    ``amplitude`` bounded ``(0, inf)`` unless negative intensities are
    allowed (mirrors :func:`nmrtools.fitting.build_bounds`). Override
    per site via ``site["bounds"]`` (keys "center_ppm"/"fwhm"/
    "amplitude") or per group via ``group["bounds_J"]``.
    """
    amp_lower = 0.0 if (positive_intensity and not allow_negative_intensity) else -np.inf

    lower, upper = [], []
    for site, topology in zip(sites, topologies):
        b = site.get("bounds", {})

        lo, hi = b.get("center_ppm", (-np.inf, np.inf))
        lower.append(lo); upper.append(hi)

        for group in site["couplings"]:
            lo, hi = group.get("bounds_J", (0.0, np.inf))
            lower.append(lo); upper.append(hi)

        lo, hi = b.get("fwhm", (1e-6, np.inf))
        lower.append(lo); upper.append(hi)

        lo, hi = b.get("amplitude", (amp_lower, np.inf))
        lower.append(lo); upper.append(hi)

    return np.array(lower, dtype=float), np.array(upper, dtype=float)


# ============================================================
# Fitting
# ============================================================

def _param_kinds(labels, topologies):
    """
    Tag each entry of the full parameter vector (site blocks, in
    order, then baseline) with what kind of quantity it is --
    'center_ppm', 'J', 'fwhm', 'amplitude', or 'baseline'. Used to
    size a sane finite-difference step per parameter for curve_fit
    (see :func:`_finite_diff_steps`).
    """
    kinds = []
    for label, topology in zip(labels, topologies):
        kinds.append("center_ppm")
        kinds.extend(["J"] * len(topology))
        kinds.append("fwhm")
        kinds.append("amplitude")
    kinds.extend(["baseline", "baseline"])
    return kinds


def _finite_diff_steps(p0_arr, kinds):
    """
    Absolute-floored finite-difference step for every parameter, to
    hand to ``scipy.optimize.curve_fit`` as ``diff_step``.

    Why this is needed: :func:`combine_jcouplings` (and the functions
    it calls) round stick-line frequencies to the nearest 1e-6 Hz so
    that lines from different coupling combinations can be merged by
    exact-key lookup. That is harmless for the physics, but it means
    two J values that differ by much less than ~1e-6 Hz produce a
    BIT-IDENTICAL simulated spectrum. ``curve_fit``'s default
    finite-difference step is *relative* (roughly ``sqrt(eps) * |x|``,
    ~1e-8 relative) -- for a typical J of a few Hz that is an absolute
    perturbation many orders of magnitude below that 1e-6 Hz floor, so
    the numerical derivative for every ``J`` parameter comes out
    exactly zero and the optimiser can never move it (this was caught
    by testing this module against synthetic data -- small-J
    parameters silently stayed pinned at their starting guess). Each
    parameter here instead gets a step no smaller than a fixed,
    parameter-kind-appropriate absolute floor (well above that 1e-6 Hz
    rounding grain for J; a small ppm step for shift/linewidth; scipy's
    own default-equivalent, unfloored, for amplitude, which has no
    such rounding pathology).

    Returns
    -------
    ndarray, same length as ``p0_arr`` -- pass ``[is_free]`` of this
    to curve_fit's ``diff_step``.
    """
    floors = {
        "center_ppm": 1e-4,   # ppm
        "J": 1e-3,             # Hz -- 1000x the 1e-6 Hz rounding grain
        "fwhm": 1e-4,          # ppm
        "amplitude": 0.0,      # no rounding pathology; use scipy's own default
        "baseline": 0.0,
    }
    p0_arr = np.asarray(p0_arr, dtype=float)
    steps = np.empty_like(p0_arr)
    for i, (val, kind) in enumerate(zip(p0_arr, kinds)):
        floor = floors[kind]
        if floor == 0.0:
            steps[i] = np.sqrt(np.finfo(float).eps)  # scipy's own default step
        else:
            # scipy's `diff_step` is relative: absolute step = diff_step * max(1, |x|).
            # Solve for the relative diff_step that yields exactly `floor` absolute.
            steps[i] = floor / max(1.0, abs(val))
    return steps


def fit_jcoupling_spectrum(results, p0, larmor_Hz=None, lineshape="lorentzian", eta=0.0,
                            fit_regions=None, use_normalised=True,
                            baseline_regions=None, baseline_sigma=2.5, baseline_max_iter=10,
                            positive_intensity=True, allow_negative_intensity=False,
                            intensity_threshold=1e-8):
    """
    Fit one or more J-coupled multiplet sites to an experimental 1D
    spectrum by nonlinear least squares.

    Mirrors :func:`nmrtools.csa_fitting.fit_csa_spectrum`'s workflow: a
    robust linear baseline is estimated once and held fixed, the
    spectrum is always optimised on a scale normalised to its maximum
    absolute intensity (for numerical conditioning -- everything
    returned is rescaled back to raw units), and the model is
    re-simulated on the full sorted ppm axis every iteration, then
    subset to ``fit_regions``.

    Parameters
    ----------
    results : dict
        Output of :func:`nmrtools.one_d.NMR1dimport`.
    p0 : dict or list of dict
        Initial guess -- one site, or a list of them. See the module
        docstring and :func:`_normalise_jcoupling_sites`.
    larmor_Hz : float, optional
        Observe frequency of the fitted nucleus, in Hz. If None, taken
        from ``results['SFO1']`` (MHz) * 1e6.
    lineshape : {'gaussian', 'lorentzian', 'pseudo_voigt'}, optional
        Broadening kernel convolved onto every site's stick pattern
        (shared by all sites). Default 'lorentzian'.
    eta : float, optional
        Gaussian/Lorentzian mixing fraction, only used when
        ``lineshape='pseudo_voigt'``; shared by all sites, held fixed
        (not a fitted parameter here -- unlike per-site CSA fitting,
        where each site gets its own).
    fit_regions : list of tuple, optional
        ``[(xmin, xmax), ...]`` in ppm. If None, the complete spectrum
        is fitted.
    use_normalised : bool, optional
        Kept for API symmetry with :func:`nmrtools.fitting.fit_spectrum`;
        the optimiser always works on the max-abs-normalised scale
        internally (see :func:`nmrtools.csa_fitting.fit_csa_spectrum`
        for why) and everything returned is in raw units regardless.
    baseline_regions, baseline_sigma, baseline_max_iter :
        See :func:`nmrtools.fitting.estimate_baseline_robust`. The
        baseline is estimated once, globally, and shared by all sites.
    positive_intensity, allow_negative_intensity : bool, optional
        See :func:`nmrtools.fitting.fit_spectrum`. Applies to every
        site's amplitude.
    intensity_threshold : float, optional
        Forwarded to :func:`jcoupling_site_lineshape` -- drops
        vanishingly small satellite combinations from very
        low-abundance coupling partners, keeping the simulation cheap.

    Returns
    -------
    popt : ndarray
        Per-site ``(center_ppm, J_1, ..., J_k, fwhm, amplitude)``
        blocks in site order, followed by ``(b0, b1)``.
    pcov_free : ndarray
        Covariance of the free parameters only -- expand with
        :func:`nmrtools.fitting.expand_pcov`. In raw units, like
        ``popt``.
    is_free : ndarray(bool)
    scale : float
    fit_info : dict
        ``{"site_slices", "topologies", "labels", "larmor_Hz",
        "lineshape", "eta", "x_ref", "nuclide", "fit_regions",
        "intensity_threshold", "warnings"}`` -- pass this whole dict
        on to :func:`evaluate_jcoupling_fit` /
        :func:`NMR1dplot_with_jcoupling_fit` /
        :func:`print_jcoupling_fit_report` so the fitted model is
        re-evaluated consistently. ``"nuclide"`` (from
        ``results.get('nuclide')``) is used only for
        :func:`jcoupling_chemist_notation`'s default labels.
    """
    sites, labels, topologies = _normalise_jcoupling_sites(p0)

    if larmor_Hz is None:
        larmor_Hz = results["SFO1"] * 1.0e6

    x_full = results["x_ppm"]
    y_raw = np.asarray(results["real_spectrum"], dtype=float)
    scale = float(np.max(np.abs(y_raw))) if y_raw.size else 1.0
    scale = scale if scale != 0 else 1.0
    y_full = y_raw / scale

    order = np.argsort(x_full)
    x_sorted = x_full[order]
    y_sorted = y_full[order]

    mask = build_region_mask(x_sorted, fit_regions)
    x_fit = x_sorted[mask]
    y_fit = y_sorted[mask]

    x_ref = np.mean(x_fit)

    b0_init, b1_init, _ = estimate_baseline_robust(
        x_sorted, y_sorted, baseline_regions=baseline_regions,
        sigma=baseline_sigma, max_iter=baseline_max_iter, x_ref=x_ref,
    )

    p0_arr, is_free, site_slices = initialise_jcoupling_params(
        sites, labels, topologies, b0_init, b1_init, x=x_fit, y=y_fit,
        positive_intensity=positive_intensity,
        allow_negative_intensity=allow_negative_intensity,
    )
    is_free[-2:] = False  # baseline held fixed, as in fit_spectrum

    # Amplitudes the caller supplied explicitly are in raw units; the
    # optimiser works in scaled ones.
    for (start, _), site, topology in zip(site_slices, sites, topologies):
        if "amplitude" in site:
            amp_idx = start + 1 + len(topology) + 1
            p0_arr[amp_idx] = float(site["amplitude"]) / scale

    lower, upper = build_jcoupling_bounds(
        sites, topologies, positive_intensity=positive_intensity,
        allow_negative_intensity=allow_negative_intensity,
    )
    lower = np.append(lower, [-np.inf, -np.inf])
    upper = np.append(upper, [np.inf, np.inf])
    for (start, _), topology in zip(site_slices, topologies):
        amp_idx = start + 1 + len(topology) + 1
        lower[amp_idx] = lower[amp_idx] / scale
        upper[amp_idx] = upper[amp_idx] / scale

    assert len(p0_arr) == len(is_free) == len(lower) == len(upper), "parameter length mismatch"

    all_names = _all_param_names(labels, topologies)
    warnings = []
    for i in np.flatnonzero(is_free):
        lo, hi = lower[i], upper[i]
        if not (lo <= p0_arr[i] <= hi):
            moved = float(np.clip(p0_arr[i], lo, hi))
            warnings.append(f"{all_names[i]}: starting value {p0_arr[i]:g} is outside its "
                             f"bounds ({lo:g}, {hi:g}); started at {moved:g} instead")
            print("Note: " + warnings[-1])
            p0_arr[i] = moved

    # The model is exact at any point (see jcoupling_site_lineshape), so
    # it is evaluated on the fitted points only -- not on the whole axis
    # and then masked.
    def model_wrapper(x, *free_params):
        full = p0_arr.copy()
        full[is_free] = free_params
        return build_jcoupling_model(
            x_fit, full, site_slices, topologies, larmor_Hz,
            lineshape=lineshape, eta=eta, x_ref=x_ref,
            intensity_threshold=intensity_threshold,
        )

    kinds = _param_kinds(labels, topologies)
    diff_step = _finite_diff_steps(p0_arr, kinds)[is_free]

    popt_free, pcov_free = _fit_progress.limited_curve_fit(
        _fit_progress.tracked(model_wrapper, y_fit, n_free=int(is_free.sum())), x_fit, y_fit,
        p0=p0_arr[is_free],
        bounds=(lower[is_free], upper[is_free]), label="The J-coupling fit",
        diff_step=diff_step,
    )

    popt = p0_arr.copy()
    popt[is_free] = popt_free

    unit = np.ones(len(popt))
    for (start, _), topology in zip(site_slices, topologies):
        unit[start + 1 + len(topology) + 1] = scale
    unit[-2:] = scale
    popt *= unit
    free_unit = unit[is_free]
    pcov_free = np.asarray(pcov_free) * np.outer(free_unit, free_unit)

    fit_info = {
        "site_slices": site_slices, "topologies": topologies, "labels": labels,
        "larmor_Hz": larmor_Hz, "lineshape": lineshape, "eta": eta,
        "x_ref": x_ref, "nuclide": results.get("nuclide"),
        "fit_regions": (None if fit_regions is None
                        else [(float(min(a, b)), float(max(a, b))) for a, b in fit_regions]),
        "intensity_threshold": intensity_threshold,
        "warnings": warnings,
    }

    return popt, pcov_free, is_free, scale, fit_info


def evaluate_jcoupling_fit(x_ppm, popt, fit_info, return_components=False):
    """
    Evaluate the fitted (possibly multi-site) J-coupling model at
    ``x_ppm``. If ``return_components=True``, also return each
    individual site's lineshape (scaled by its amplitude) plus the
    baseline (last), as a list.
    """
    site_slices = fit_info["site_slices"]
    topologies = fit_info["topologies"]
    larmor_Hz = fit_info["larmor_Hz"]
    lineshape = fit_info["lineshape"]
    eta = fit_info.get("eta", 0.0)
    x_ref = fit_info["x_ref"]
    threshold = fit_info.get("intensity_threshold", 1e-8)

    x_ppm = np.asarray(x_ppm)

    if not return_components:
        return build_jcoupling_model(
            x_ppm, popt, site_slices, topologies, larmor_Hz,
            lineshape=lineshape, eta=eta, x_ref=x_ref, intensity_threshold=threshold,
        )

    components = []
    for (start, end), topology in zip(site_slices, topologies):
        block = popt[start:end]
        n_groups = len(topology)
        center_ppm = block[0]
        J_values = block[1:1 + n_groups]
        fwhm = block[1 + n_groups]
        amplitude = block[2 + n_groups]
        shape = jcoupling_site_lineshape(
            x_ppm, center_ppm, J_values, fwhm, larmor_Hz, topology,
            lineshape=lineshape, eta=eta, intensity_threshold=threshold,
        )
        components.append(amplitude * shape)

    b0, b1 = popt[-2], popt[-1]
    baseline = b0 + b1 * (x_ppm - x_ref)
    components.append(baseline)

    y_fit = np.sum(components, axis=0)
    return y_fit, components


# ============================================================
# Reporting
# ============================================================

_MULTIPLICITY_LETTERS = {
    1: "s", 2: "d", 3: "t", 4: "q", 5: "quint", 6: "sext",
    7: "sept", 8: "oct", 9: "non", 10: "dec",
}


def _multiplicity_letter(n_lines):
    """Conventional organic-chemistry abbreviation for a sub-multiplet
    of ``n_lines`` lines (s/d/t/q/quint/sext/...); falls back to
    ``"(n_lines)-tet"`` beyond the commonly-named ones."""
    return _MULTIPLICITY_LETTERS.get(n_lines, f"{n_lines}-tet")


def _element_symbol(nuclide):
    """``'13C' -> 'C'``, ``'1H' -> 'H'``, ``'195Pt' -> 'Pt'``: strip the
    leading mass number off a nuclide string, for building a default
    coupling label when the group doesn't carry an explicit one."""
    i = 0
    while i < len(nuclide) and nuclide[i].isdigit():
        i += 1
    return nuclide[i:]


def jcoupling_chemist_notation(observed_nuclide, topology, J_values, sort_descending=True):
    """
    Format one site's fitted J-coupling pattern the way it's
    conventionally written in an experimental section, e.g. for a
    site coupled to one 19F and two inequivalent 1H (as three separate
    coupling groups, each n=1)::

        multiplicity, j_strings = jcoupling_chemist_notation(
            '13C', topology, [36.3, 11.0, 4.2],
        )
        # multiplicity == 'ddd'
        # j_strings == ['3J(C-F) = 36.3 Hz', '2J(C-H) = 11.0 Hz', '3J(C-H) = 4.2 Hz']
        # (bond counts only appear if you gave each group a `bond_label`
        # in p0 -- see below; without one, the default label just names
        # the two nuclides, with no bond count, since that's chemical
        # structure knowledge this module has no way to infer.)

    Each coupling GROUP in ``topology`` (see :func:`combine_jcouplings`)
    contributes one multiplicity letter -- from that group's own
    number of lines, ``2*n*spin + 1`` (so a group of 2 equivalent 1H
    gives 't', not 'dd'; two separate n=1 groups give 'dd'). Groups are
    listed largest-|J|-first by default, the usual convention.

    To get a bond-labelled string like ``"3J(F-H)"`` rather than the
    generic ``"J(C-F)"`` default, set ``bond_label`` on the
    corresponding coupling group in ``p0``, e.g.::

        {'nuclide': '19F', 'n': 1, 'J_Hz': 36.3, 'bond_label': '3J(C-F)'}

    Parameters
    ----------
    observed_nuclide : str or None
        The fitted (observed) nucleus, e.g. ``'13C'`` -- used only to
        build a default label (ignored for any group that has its own
        ``bond_label``). Pass ``fit_info.get('nuclide')`` after a
        :func:`fit_jcoupling_spectrum` call, or None to fall back to
        just the partner element (``"J(F)"`` instead of ``"J(C-F)"``).
    topology : list of dict
        One site's fixed coupling-group metadata (as returned by
        :func:`_normalise_jcoupling_sites` / stored in
        ``fit_info["topologies"]``).
    J_values : sequence of float
        That site's fitted (or any other) J value per group, same
        order as ``topology``.
    sort_descending : bool, optional
        List groups largest-|J|-first (the usual convention). Set
        False to keep them in ``topology``'s own order instead.

    Returns
    -------
    multiplicity : str
        e.g. ``'ddd'``, or ``'s'`` for an uncoupled site (empty
        topology).
    j_strings : list of str
        One ``"<label> = <J> Hz"`` entry per group, e.g.
        ``['3J(C-F) = 36.3 Hz', ...]``, in the same order as
        ``multiplicity``'s letters.
    """
    observed_element = _element_symbol(observed_nuclide) if observed_nuclide else None

    entries = []
    for group, J in zip(topology, J_values):
        n_lines = int(round(2 * group["n"] * group["spin"] + 1))
        letter = _multiplicity_letter(n_lines)
        label = group.get("bond_label")
        if not label:
            partner_element = _element_symbol(group["nuclide"])
            label = (f"J({observed_element}-{partner_element})" if observed_element
                      else f"J({partner_element})")
        entries.append((abs(J), letter, label, float(J)))

    if sort_descending:
        entries.sort(key=lambda e: e[0], reverse=True)

    multiplicity = "".join(letter for _, letter, _, _ in entries) or "s"
    j_strings = [f"{label} = {J:.1f} Hz" for _, _, label, J in entries]
    return multiplicity, j_strings


def print_jcoupling_fit_report(x, y, popt, pcov, is_free, fit_info, warnings=None,
                                save_txt=False, txt_filename="jcoupling_fit_report.txt",
                                chemist_notation=False):
    """Print (and optionally save) a human-readable report of fitted
    parameters, uncertainties, per-site integrals, and fit-quality
    metrics.

    Set ``chemist_notation=True`` to add one extra line per site
    formatted the way an experimental section conventionally reports
    a multiplet, via :func:`jcoupling_chemist_notation` -- e.g.
    ``Notation: ddd, 3J(C-F) = 36.3 Hz, 2J(C-H) = 11.0 Hz, 3J(C-H) = 4.2 Hz``.
    Give each coupling group a ``bond_label`` in ``p0`` for a proper
    bond-counted label (``"3J(C-F)"``); without one you still get a
    correct multiplicity letter and the fitted J, just with a generic
    label (``"J(C-F)"``, no bond count) -- see
    :func:`jcoupling_chemist_notation` for why bond count can't be
    inferred automatically.
    """

    labels = fit_info["labels"]
    site_slices = fit_info["site_slices"]
    topologies = fit_info["topologies"]

    errors = np.sqrt(np.maximum(np.diag(pcov), 0.0))

    y_fit, components = evaluate_jcoupling_fit(x, popt, fit_info, return_components=True)
    metrics = fit_quality(y, y_fit, len(popt))

    order = np.argsort(x)
    x_sorted = x[order]
    integrals = [_trapz(np.asarray(c)[order], x_sorted) for c in components[:-1]]
    total = np.sum(integrals)
    integrals_norm = [100 * a / total for a in integrals] if total != 0 else [np.nan] * len(integrals)

    warnings = list(fit_info.get("warnings", [])) + list(warnings or [])

    report = []

    def write(line=""):
        print(line)
        report.append(str(line))

    write("\n" + "=" * 60)
    write(f"J-COUPLING FIT REPORT ({len(labels)} site{'s' if len(labels) != 1 else ''}, "
          f"{fit_info['lineshape']} lineshape)")
    write("=" * 60)

    for (start, end), label, topology, integral, integral_pct in zip(
            site_slices, labels, topologies, integrals, integrals_norm):

        names = _site_param_names(label, topology)
        write(f"\n{label}:")
        write("-" * 60)
        for name, val, err, free in zip(names, popt[start:end], errors[start:end], is_free[start:end]):
            short_name = name.split(": ", 1)[1]
            if free:
                write(f"  {short_name:<28}= {val:.6f} \u00b1 {err:.6f}")
            else:
                write(f"  {short_name:<28}= {val:.6f} (fixed)")
        write(f"  {'integral (raw)':<28}= {integral:.4f}")
        write(f"  {'integral (%)':<28}= {integral_pct:.2f}")

        if chemist_notation:
            n_groups = len(topology)
            J_values = popt[start + 1:start + 1 + n_groups]
            multiplicity, j_strings = jcoupling_chemist_notation(
                fit_info.get("nuclide"), topology, J_values,
            )
            write(f"  Notation: ({multiplicity}, {', '.join(j_strings)})")

    write("\nFit quality:")
    write("-" * 60)
    write(f"RMSE          : {metrics['rmse']:.6e}")
    write(f"Chi^2         : {metrics['chi2']:.6e}")
    write(f"Reduced Chi^2 : {metrics['reduced_chi2']:.6e}")
    write("=" * 60 + "\n")

    if warnings:
        write("=" * 60 + "\n")
        write("Warnings:")
        write("-" * 60)
        for warning in warnings:
            write(warning)
            write("----")
        write("=" * 60 + "\n")

    if save_txt:
        with open(txt_filename, "w", encoding="utf-8") as f:
            f.write("\n".join(report))


# ============================================================
# 1D plotting with fit overlay
# ============================================================

def NMR1dplot_with_jcoupling_fit(results, popt, fit_info, fit_regions=None,
                                  xlims='default', xlabel=None, pl_width=3.375, pl_height=3.375,
                                  font='Arial', fontsize=12, output_format='',
                                  show_components=False, colour_data='black', colour_fit='red',
                                  colour_baseline='gray', colour_components=None,
                                  tick_dir='in', tick_length=4.0, tick_width=1.0, tick_pad=4.0,
                                  tick_number=None, field='', use_normalised=True,
                                  save_fig=False, fig_name='nmr_jcoupling_fit_plot', save_format='png'):
    """
    Plot a 1D spectrum together with its fitted (possibly multi-site)
    J-coupling model. See :func:`nmrtools.csa_fitting.NMR1dplot_with_csa_fit`
    for the (near-identical) parameter conventions this mirrors.

    Returns
    -------
    fig, ax
    """
    style = setup_plotting(font=font, fontsize=fontsize, pl_width=pl_width,
                            pl_height=pl_height, output_format=output_format)
    pl_width, pl_height = style['pl_width'], style['pl_height']

    labels = fit_info["labels"]

    x_full = results['x_ppm']
    order = np.argsort(x_full)
    x_sorted = x_full[order]

    if use_normalised:
        scale = np.max(np.abs(results['real_spectrum']))
        scale = scale if scale != 0 else 1.0
        y_sorted = (results['real_spectrum'] / scale)[order]
    else:
        scale = 1.0
        y_sorted = results['real_spectrum'][order]

    mask = build_region_mask(x_sorted, fit_regions) if fit_regions is not None \
        else np.ones_like(x_sorted, dtype=bool)
    x_fit = x_sorted[mask]
    y_data = y_sorted[mask]

    if show_components:
        y_model_full, components = evaluate_jcoupling_fit(x_sorted, popt, fit_info,
                                                            return_components=True)
        y_model = y_model_full[mask] / scale
        components = [c[mask] / scale for c in components]
    else:
        y_model = evaluate_jcoupling_fit(x_sorted, popt, fit_info)[mask] / scale
        components = None

    xmin, xmax = x_fit[0], x_fit[-1]

    fig, ax = plt.subplots()

    ax.plot(x_fit, y_data, color=colour_data, lw=1.5, label='Data')
    ax.plot(x_fit, y_model, color=colour_fit, lw=2, label='Fit')

    if show_components:
        n_sites = len(labels)
        site_components, baseline = components[:n_sites], components[n_sites]
        if colour_components is None:
            colour_components = plt.rcParams['axes.prop_cycle'].by_key()['color']
        for i, (comp, label) in enumerate(zip(site_components, labels)):
            colour = colour_components[i % len(colour_components)]
            ax.plot(x_fit, comp, color=colour, lw=1, ls='--', alpha=0.8, label=label)
        ax.plot(x_fit, baseline, color=colour_baseline, lw=1, alpha=0.6, label='Baseline')

    if xlims == 'default':
        ax.set_xlim(xmin, xmax)
    else:
        ax.set_xlim(min(xlims), max(xlims))

    ax.invert_xaxis()
    ax.set_yticks([])
    apply_spine_style(ax, top=False, right=False, bottom=True, left=False)
    apply_tick_style(ax, direction=tick_dir, length=tick_length, width=tick_width,
                      pad=tick_pad, num_ticks=tick_number, axis='x')

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    else:
        NUC = results.get('nuclide', 'unknown')
        SFO1 = results.get('SFO1', 0)
        nucgamma = find_gamma(NUC, field=field, SFO1=SFO1)
        ax.set_xlabel(NUC_label(nucgamma['Nuclide'], x_unit='ppm'))

    ax.legend()
    fig.set_figheight(pl_height)
    fig.set_figwidth(pl_width)
    fig.tight_layout()

    if save_fig:
        save_figure(fig, fig_name=fig_name, save_format=save_format)

    return fig, ax


# ============================================================
# 1D workflow
# ============================================================

def NMRJCouplingFitWorkflow(datapath, p0, procno=1, larmor_Hz=None,
                             lineshape="lorentzian", eta=0.0,
                             fit_regions=None, baseline_regions=None,
                             baseline_sigma=2.5, baseline_max_iter=10, normalise=True,
                             positive_intensity=True, allow_negative_intensity=False,
                             intensity_threshold=1e-8,
                             show_plot=True, show_components=False, show_report=True,
                             chemist_notation=False,
                             xlims='default', xlabel=None,
                             pl_width=3.375, pl_height=3.375, font='Arial', fontsize=12,
                             output_format='', tick_dir='in', tick_length=4.0,
                             tick_width=1.0, tick_pad=4.0, tick_number=None,
                             save_csv=False, csv_filename="jcoupling_fit_results.csv",
                             save_report=False, report_filename="jcoupling_fit_report.txt",
                             save_fig=False, fig_name='nmr_jcoupling_fit_plot', save_format='png'):
    """
    Fit one or more J-coupled multiplet sites to a processed 1D
    Bruker spectrum.

    The J-coupling analogue of :func:`nmrtools.csa_fitting.NMRCSAFitWorkflow`
    / :func:`nmrtools.fitting.NMRFitWorkflow`: imports the spectrum,
    fits each site's chemical shift, per-group J values, linewidth and
    amplitude (plus a shared fixed linear baseline) by nonlinear least
    squares against the multiplet simulation in
    :mod:`nmrtools.simulations`, and optionally plots and reports the
    result.

    Parameters
    ----------
    datapath : str
        Path to the Bruker experiment directory.
    p0 : dict or list of dict
        Initial guess for one site, or a list of them -- see the
        module docstring.
    procno : int, optional
    larmor_Hz, lineshape, eta, fit_regions, baseline_regions,
    baseline_sigma, baseline_max_iter, positive_intensity,
    allow_negative_intensity, intensity_threshold :
        See :func:`fit_jcoupling_spectrum`.
    normalise : bool, optional
        Fit (and plot) the spectrum normalised to its maximum absolute
        intensity.
    show_plot, show_components, show_report : bool, optional
    chemist_notation : bool, optional
        Forwarded to :func:`print_jcoupling_fit_report` -- add an
        organic-chemistry-style "(ddd, 3J(C-F) = 36.3 Hz, ...)" line
        per site. See that function and
        :func:`jcoupling_chemist_notation` for how to get proper
        bond-counted labels via ``bond_label`` on each coupling group
        in ``p0``.
    xlims, xlabel, pl_width, pl_height, font, fontsize, output_format,
    tick_dir, tick_length, tick_width, tick_pad, tick_number :
        Forwarded to :func:`NMR1dplot_with_jcoupling_fit`.
    save_csv, csv_filename, save_report, report_filename : optional
    save_fig, fig_name, save_format : optional

    Returns
    -------
    dict with keys ``popt``, ``pcov`` (full, expanded), ``fit_info``,
    ``labels``, ``x``, ``y``, ``y_fit``, ``components``, ``figure``,
    ``axes``.
    """

    setup_plotting()

    results = NMR1dimport(datapath, procno=procno)

    popt, pcov_free, is_free, scale, fit_info = fit_jcoupling_spectrum(
        results, p0, larmor_Hz=larmor_Hz, lineshape=lineshape, eta=eta,
        fit_regions=fit_regions, use_normalised=normalise,
        baseline_regions=baseline_regions, baseline_sigma=baseline_sigma,
        baseline_max_iter=baseline_max_iter,
        positive_intensity=positive_intensity,
        allow_negative_intensity=allow_negative_intensity,
        intensity_threshold=intensity_threshold,
    )

    labels = fit_info["labels"]
    pcov = expand_pcov(pcov_free, is_free, len(popt))

    x_full = results['x_ppm']
    order = np.argsort(x_full)
    x_sorted = x_full[order]
    y_sorted = results['real_spectrum'][order]

    mask = build_region_mask(x_sorted, fit_regions) if fit_regions is not None \
        else np.ones_like(x_sorted, dtype=bool)
    x_fit = x_sorted[mask]
    y_fit_data = y_sorted[mask]

    if show_components:
        y_model_full, components_full = evaluate_jcoupling_fit(
            x_sorted, popt, fit_info, return_components=True,
        )
        y_model = y_model_full[mask]
        components = [c[mask] for c in components_full]
    else:
        y_model = evaluate_jcoupling_fit(x_sorted, popt, fit_info)[mask]
        components = None

    fig, ax = None, None
    if show_plot:
        fig, ax = NMR1dplot_with_jcoupling_fit(
            results=results, popt=popt, fit_info=fit_info,
            fit_regions=fit_regions, show_components=show_components,
            use_normalised=normalise, xlims=xlims, xlabel=xlabel,
            pl_width=pl_width, pl_height=pl_height, font=font, fontsize=fontsize,
            output_format=output_format, tick_dir=tick_dir, tick_length=tick_length,
            tick_width=tick_width, tick_pad=tick_pad, tick_number=tick_number,
        )
        if save_fig:
            save_figure(fig, fig_name=fig_name, save_format=save_format)

    if show_report:
        print_jcoupling_fit_report(x_fit, y_fit_data, popt, pcov, is_free, fit_info,
                                    save_txt=save_report, txt_filename=report_filename,
                                    chemist_notation=chemist_notation)

    if save_csv:
        import csv

        columns = {"x": x_fit, "y": y_fit_data, "y_fit": y_model}
        if components is not None:
            for label, comp in zip(labels, components[:-1]):
                columns[f"{label}_component"] = comp
            columns["baseline"] = components[-1]

        with open(csv_filename, "w", newline="", encoding="utf-8") as outfile:
            writer = csv.writer(outfile)
            writer.writerow(columns.keys())
            for i in range(len(x_fit)):
                writer.writerow([columns[col][i] for col in columns])

    return {
        "popt": popt, "pcov": pcov, "fit_info": fit_info, "labels": labels,
        "x": x_fit, "y": y_fit_data, "y_fit": y_model,
        "components": components, "figure": fig, "axes": ax,
    }