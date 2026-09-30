"""
Fit static or finite-speed-MAS CSA powder lineshapes to experimental
1D NMR spectra, using the lineshape simulator in
:mod:`nmrtools.csa_quad_simulator` as the fit model and the shared
baseline/region-masking machinery from :mod:`nmrtools.fitting`.

Workflow
--------

    NMR1dimport -> fit_csa_spectrum -> NMR1dplot_with_csa_fit
                                     -> print_csa_fit_report

or, end-to-end, :func:`NMRCSAFitWorkflow`.

Unlike the Gaussian/Lorentzian/pseudo-Voigt models in
:mod:`nmrtools.fitting`, which are closed-form functions that can be
evaluated pointwise at any x, a CSA powder lineshape is only defined
here via a powder average that gets histogrammed onto a uniform ppm
axis (see :func:`nmrtools.csa_quad_simulator.static_csa`). So instead
of evaluating the model only at the fitted x-points, every fit
iteration below simulates the full spectrum on the (ascending-sorted)
experimental ppm axis and then selects the requested ``fit_regions``
subset -- see :func:`csa_model_full` and :func:`fit_csa_spectrum`. The
powder-average loop over crystallite orientations, not the number of
ppm points, dominates the simulation cost, so this is cheap relative
to a finer ppm grid but NOT cheap relative to ``n_theta * n_phi`` (or,
for the finite-MAS model, ``n_theta * n_phi * n_time``): keep those
coarse while fitting and only refine them for the final plot/report.

Multi-site fits
----------------

``p0`` may describe more than one CSA site: pass a list of dicts
(one per site) instead of a single dict, e.g.::

    p0 = [
        {"label": "Fe(1)", "d_iso": 40, "span_or_Delta": 300,
         "skew_or_eta": 0.3, "fwhm": 5},
        {"label": "Fe(2)", "d_iso": -10, "span_or_Delta": 250,
         "skew_or_eta": 0.6, "fwhm": 8},
    ]

Each site gets its own ``(d_iso, span_or_Delta, skew_or_eta, fwhm,
amplitude)``; all sites share a single linear baseline. A bare dict is
still accepted for backward compatibility and is treated as a
one-site list. See :func:`initialise_csa_params` for the full field
list, and :func:`build_csa_bounds`/:func:`fit_csa_spectrum` for how
``bounds``/``fixed`` can be given globally (applied to every site) or
per site (keyed by site index or label).

Note on cost and identifiability: each additional site adds another
full powder simulation per ``curve_fit`` iteration (roughly linear
cost in the number of sites), and if sites overlap substantially in
ppm their amplitude/span/skew parameters become correlated, which can
make Levenberg-Marquardt converge to a poor local minimum depending on
the initial guess. Tightening ``bounds``/``fixed`` per site, or
fitting sites individually first and combining the results as the
initial guess for the joint fit, are both worth trying if convergence
looks unstable.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from .utils import find_gamma, NUC_label
from .plotting import setup_plotting, apply_tick_style, apply_spine_style, save_figure
from .one_d import NMR1dimport
from .fitting import build_region_mask, estimate_baseline_robust, fit_quality, expand_pcov
from .simulations import csa_definition, make_static_csa_spectrum, finite_mas_csa_spectrum

_CSA_PARAM_NAMES = ["d_iso", "span_or_Delta", "skew_or_eta", "fwhm", "amplitude", "eta"]

# NOTE ON NAMING: `skew_or_eta` is the CSA tensor's own asymmetry
# parameter (Haeberlen eta_CS, or Herzfeld-Berger skew) -- an
# intrinsic, physical property of the tensor. `eta` (the 6th, newer
# per-site parameter below) is an unrelated quantity: the Gaussian/
# Lorentzian mixing fraction of the broadening kernel convolved onto
# the simulated powder pattern (see `nmrtools.simulations.
# pseudo_voigt_broadening`), only meaningful when `lineshape=
# 'pseudo_voigt'`. Both are conventionally called "eta" in the NMR
# literature; this package keeps them as two distinct fields
# (`skew_or_eta` vs `eta`) specifically to avoid conflating them, and
# `print_csa_fit_report` labels them differently for the same reason.
# Double-check which one you mean when reading or writing a p0/bounds/
# fixed dict.


# ============================================================
# Model
# ============================================================

def csa_model_full(x_ppm_sorted, d_iso, span_or_Delta, skew_or_eta, fwhm, eta=0.0,
                    convention="Haeberlen", static=True,
                    larmor_Hz=None, mas_Hz=None, rotor_angle="magic",
                    n_theta=60, n_phi=120, use_octant_symmetry=True,
                    n_time=1024, n_sidebands=15, subtract_residual_mean=True,
                    powder_scheme='rectangular', n_zcw=None,
                    lineshape="gaussian", normalised=True):
    """
    Simulate a unit-amplitude, normalised CSA lineshape for a SINGLE
    site on ``x_ppm_sorted`` (must be ascending and evenly spaced --
    the axis convention used by
    :func:`nmrtools.simulations.make_static_csa_spectrum` and
    :func:`nmrtools.simulations.finite_mas_csa_spectrum`).

    For multi-site spectra this is called once per site and the
    results are summed with per-site amplitudes -- see
    :func:`build_csa_model`.

    Parameters
    ----------
    static : bool, optional
        If True (default), simulate a static powder pattern via
        :func:`nmrtools.simulations.make_static_csa_spectrum`.
        If False, simulate a finite-speed-MAS sideband pattern via
        :func:`nmrtools.simulations.finite_mas_csa_spectrum`,
        which requires ``larmor_Hz`` and ``mas_Hz``.
    lineshape : {'gaussian', 'lorentzian', 'pseudo_voigt'}, optional
        Broadening kernel convolved onto the simulated powder pattern
        (applies to every site sharing this ``sim_kwargs``/fit -- see
        :func:`fit_csa_spectrum`). Default 'gaussian' (unchanged
        behaviour from earlier versions).
    eta : float, optional
        Gaussian/Lorentzian mixing fraction, 0 (pure Gaussian) to 1
        (pure Lorentzian). Only meaningful (and only fitted --
        see :func:`fit_csa_spectrum`) when ``lineshape='pseudo_voigt'``;
        ignored otherwise. NOT the CSA tensor's own asymmetry parameter
        -- see the note by ``_CSA_PARAM_NAMES`` above.
    n_theta, n_phi, use_octant_symmetry : optional
        Powder-grid size for the static case. Only used when
        ``powder_scheme='rectangular'``.
    n_theta, n_phi, n_time, n_sidebands, rotor_angle,
    subtract_residual_mean : optional
        Powder-grid/FFT size and gamma-averaging options for the
        finite-MAS case. See
        :func:`nmrtools.simulations.finite_mas_csa_spectrum`.
    powder_scheme : {'rectangular', 'zcw'}, optional
        Powder-averaging scheme, forwarded to the simulator (see
        :func:`nmrtools.simulations.get_powder_grid`). 'zcw'
        typically converges with far fewer orientations for the
        finite-MAS case (each orientation costs an FFT, and this
        function is called ~100x per fit via curve_fit's Jacobian,
        once per site) -- it's a smaller win for the static case,
        where ``use_octant_symmetry`` already gives an exact 8x
        reduction.
    n_zcw : int, optional
        Minimum ZCW orientation count; only used when
        ``powder_scheme='zcw'``. Defaults to ``n_theta * n_phi``.

    normalised : bool, optional
        True (default): the lineshape scaled to a maximum of 1, which is
        what the fit's amplitudes multiply. False: the simulator's own
        absolutely scaled ``real_spec`` (see csa_site_amounts).

    Returns
    -------
    ndarray
        Normalised lineshape (max absolute value 1), same shape as
        ``x_ppm_sorted``.
    """
    csa = csa_definition(d_iso, span_or_Delta, skew_or_eta, convention=convention)

    if static:
        spec = make_static_csa_spectrum(
            csa, x_ppm_sorted, fwhm=fwhm, n_theta=n_theta, n_phi=n_phi,
            use_octant_symmetry=use_octant_symmetry,
            powder_scheme=powder_scheme, n_zcw=n_zcw,
            lineshape=lineshape, eta=eta,
        )
    else:
        if larmor_Hz is None or mas_Hz is None:
            raise ValueError("larmor_Hz and mas_Hz are required when static=False.")
        spec = finite_mas_csa_spectrum(
            csa, x_ppm_sorted, larmor_Hz=larmor_Hz, mas_Hz=mas_Hz,
            rotor_angle=rotor_angle, n_theta=n_theta, n_phi=n_phi,
            n_time=n_time, n_sidebands=n_sidebands, fwhm=fwhm,
            subtract_residual_mean=subtract_residual_mean,
            powder_scheme=powder_scheme, n_zcw=n_zcw,
            lineshape=lineshape, eta=eta,
        )

    return spec["normalised_real_spec"] if normalised else spec["real_spec"]


def build_csa_model(x_ppm_sorted, params, n_sites, x_ref=None, **sim_kwargs):
    """
    Full multi-site model:
    ``sum_i(amplitude_i * csa_model_full(..., site_i)) + b0 + b1*(x - x_ref)``.

    Parameters
    ----------
    params : ndarray
        Flat parameter vector: for each of the ``n_sites`` sites (in
        order) ``(d_iso, span_or_Delta, skew_or_eta, fwhm, amplitude,
        eta)``, i.e. ``6 * n_sites`` values, followed by the shared
        baseline ``(b0, b1)``. Total length ``6 * n_sites + 2``.
    n_sites : int
        Number of CSA sites encoded in ``params``.
    sim_kwargs :
        Forwarded to :func:`csa_model_full` for every site
        (``convention``, ``static``, ``larmor_Hz``, ``mas_Hz``,
        ``lineshape``, grid sizes, etc). All sites share the same
        simulation settings (including ``lineshape``, the broadening
        kernel choice -- only each site's own ``eta`` mixing fraction,
        carried in ``params``, differs between sites).
    """
    params = np.asarray(params, dtype=float)
    x_ppm_sorted = np.asarray(x_ppm_sorted)

    expected_len = 6 * n_sites + 2
    if params.shape[0] != expected_len:
        raise ValueError(
            f"params has length {params.shape[0]}, expected {expected_len} "
            f"for n_sites={n_sites} (6 per site + 2 baseline)."
        )

    csa_params = params[: 6 * n_sites].reshape(n_sites, 6)
    b0, b1 = params[6 * n_sites], params[6 * n_sites + 1]
    x_ref = np.mean(x_ppm_sorted) if x_ref is None else x_ref

    total = np.zeros_like(x_ppm_sorted, dtype=float)
    for d_iso, span_or_Delta, skew_or_eta, fwhm, amplitude, eta in csa_params:
        shape = csa_model_full(x_ppm_sorted, d_iso, span_or_Delta, skew_or_eta,
                                fwhm, eta=eta, **sim_kwargs)
        total += amplitude * shape

    return total + b0 + b1 * (x_ppm_sorted - x_ref)


# ============================================================
# Peak-definition / per-site option handling
# ============================================================

#: Accepted spellings of each broadening kernel, in ``p0``'s ``"type"``
#: or the ``lineshape=`` argument.
_LINESHAPE_ALIASES = {
    "gaussian": "gaussian", "gauss": "gaussian", "g": "gaussian",
    "lorentzian": "lorentzian", "lorentz": "lorentzian", "l": "lorentzian",
    "pseudo_voigt": "pseudo_voigt", "pseudovoigt": "pseudo_voigt", "pv": "pseudo_voigt",
    "voigt": "pseudo_voigt",
}


def canonical_lineshape(name) -> str:
    """``'Pseudo-Voigt'`` -> ``'pseudo_voigt'`` etc.; raises on anything
    that is not Gaussian, Lorentzian or pseudo-Voigt."""
    key = str(name).strip().lower().replace("-", "_").replace(" ", "_")
    if key not in _LINESHAPE_ALIASES:
        raise ValueError(
            f"Unknown lineshape {name!r}; use 'gaussian', 'lorentzian', or 'pseudo_voigt'.")
    return _LINESHAPE_ALIASES[key]


def resolve_lineshape(p0, lineshape=None) -> str:
    """The broadening kernel a CSA fit uses.

    ``lineshape`` wins when given. Otherwise the sites' own ``"type"``
    (as written in an isotropic peak definition, e.g.
    ``"pseudo-voigt"``) -- previously ignored, so a ``p0`` saying
    pseudo-Voigt was silently fitted as a Gaussian. All sites share one
    kernel (the simulator broadens every site the same way), so
    differing types are an error rather than a guess. No type anywhere:
    Gaussian, the historical default."""
    sites = [p0] if isinstance(p0, dict) else list(p0 or [])
    types = {canonical_lineshape(site["type"]) for site in sites
             if isinstance(site, dict) and site.get("type") is not None}
    if lineshape is not None:
        chosen = canonical_lineshape(lineshape)
        if types and types != {chosen}:
            print(f"Note: lineshape={lineshape!r} overrides the sites' own "
                  f"type ({', '.join(sorted(types))}).")
        return chosen
    if len(types) > 1:
        raise ValueError(
            f"Sites ask for different lineshapes ({', '.join(sorted(types))}); a CSA fit "
            f"broadens every site with one kernel -- give them one type, or pass lineshape=.")
    return types.pop() if types else "gaussian"


def _normalise_peak_defs(p0):
    """
    Accept ``p0`` as a single dict (one site) or a list/tuple of dicts
    (one per site) and return ``(peak_defs, labels)`` -- a list of
    dicts and a matching list of unique, human-readable labels
    (``site["label"]`` if given, else ``"Site 1"``, ``"Site 2"``, ...).
    """
    if isinstance(p0, dict):
        p0 = [p0]
    if not isinstance(p0, (list, tuple)) or len(p0) == 0:
        raise TypeError(
            "p0 must be a dict (single site) or a non-empty list of dicts "
            "(one per site)."
        )

    peak_defs, labels = [], []
    required = ("d_iso", "span_or_Delta", "skew_or_eta", "fwhm")
    for i, site in enumerate(p0):
        if not isinstance(site, dict):
            raise TypeError(f"p0[{i}] must be a dict of CSA parameters, got {type(site)}.")
        missing = [name for name in required if name not in site]
        if missing:
            raise ValueError(
                f"p0[{i}] (label={site.get('label')!r}) is missing required "
                f"key(s): {missing}."
            )
        peak_defs.append(site)
        labels.append(site.get("label", f"Site {i + 1}"))

    if len(set(labels)) != len(labels):
        raise ValueError(f"Site labels must be unique, got: {labels}")

    return peak_defs, labels


def _resolve_site_index(key, labels):
    """Resolve a per-site key (int index or matching label) to an index."""
    if isinstance(key, (int, np.integer)) and not isinstance(key, bool):
        if 0 <= key < len(labels):
            return int(key)
        raise KeyError(f"Site index {key} out of range for {len(labels)} site(s).")
    if key in labels:
        return labels.index(key)
    raise KeyError(f"{key!r} is not a known site label or index; labels are {labels}.")


def _resolve_fixed_matrix(fixed, n_sites, labels):
    """
    Turn ``fixed`` into a list (one ``set`` of fixed parameter names
    per site). ``fixed`` may combine, in one dict:

    - global entries keyed by parameter name -> bool, applied to every
      site (also accepted as a bare list of names, all ``True``);
    - per-site entries keyed by site index or label -> a list of names
      or a ``{name: bool}`` dict, applied only to that site.

    e.g. ``fixed={"skew_or_eta": True, 0: ["d_iso"], "Fe(2)": {"fwhm": True}}``.
    """
    matrix = [set() for _ in range(n_sites)]
    if fixed is None:
        return matrix
    if isinstance(fixed, (list, tuple)):
        fixed = {name: True for name in fixed}
    if not isinstance(fixed, dict):
        raise TypeError(
            "fixed must be a list of parameter names or a dict; see "
            "initialise_csa_params / module docstring."
        )

    for key, val in fixed.items():
        if key not in _CSA_PARAM_NAMES:
            continue
        if not isinstance(val, bool):
            raise TypeError(f"Global fixed[{key!r}] must be a bool, got {type(val)}.")
        if val:
            for site_fixed in matrix:
                site_fixed.add(key)

    for key, val in fixed.items():
        if key in _CSA_PARAM_NAMES:
            continue
        idx = _resolve_site_index(key, labels)
        if isinstance(val, (list, tuple)):
            names = list(val)
        elif isinstance(val, dict):
            names = [name for name, is_fixed in val.items() if is_fixed]
        else:
            raise TypeError(
                f"Per-site fixed[{key!r}] must be a list of names or a dict, "
                f"got {type(val)}."
            )
        bad = [n for n in names if n not in _CSA_PARAM_NAMES]
        if bad:
            raise ValueError(f"Unknown parameter name(s) {bad} in fixed[{key!r}].")
        matrix[idx].update(names)

    return matrix


def _all_param_names(labels):
    """Flat, generic (non-convention-translated) parameter name per
    entry of the full parameter vector, e.g. ``"Fe(1): d_iso"``,
    ending in ``"b0", "b1"``. Used for diagnostics; see
    :func:`print_csa_fit_report` for the convention-aware display
    names."""
    names = []
    for label in labels:
        names.extend(f"{label}: {name}" for name in _CSA_PARAM_NAMES)
    names.extend(["b0", "b1"])
    return names


# ============================================================
# Model construction (initial parameters, bounds)
# ============================================================

def initialise_csa_params(p0, b0_init, b1_init, x=None, y=None, fixed=None):
    """
    Build the initial parameter vector and free-parameter mask for a
    (possibly multi-site) CSA fit.

    Parameters
    ----------
    p0 : dict or list of dict
        A single site's initial guess, or a list of them (one per
        site) -- see the module docstring. Each site dict is
        ``{"d_iso": ..., "span_or_Delta": ..., "skew_or_eta": ...,
        "fwhm": ..., ["amplitude": ...], ["eta": ...], ["label": ...]}``.
        If ``"amplitude"`` is omitted for a site, it is estimated as
        ``max(abs(y)) / n_sites`` (an equal-height guess -- ``x``/``y``
        must then be supplied, and you should supply ``"amplitude"``
        explicitly instead if sites differ substantially in
        intensity). If ``"eta"`` (the broadening-kernel Gaussian/
        Lorentzian mixing fraction, NOT the CSA asymmetry -- see the
        note by ``_CSA_PARAM_NAMES``) is omitted, it defaults to 0.0;
        only meaningful (and only actually fitted) when
        ``lineshape='pseudo_voigt'`` is passed to :func:`fit_csa_spectrum`
        -- see there for the automatic fixing that happens otherwise.
    fixed : list or dict, optional
        Names from
        ``["d_iso","span_or_Delta","skew_or_eta","fwhm","amplitude","eta"]``
        to hold fixed at their ``p0`` value, applied to every site; or
        a dict mixing such global names with per-site overrides keyed
        by site index or label -- see :func:`_resolve_fixed_matrix`.

    Returns
    -------
    params0 : ndarray
        Length ``6 * n_sites + 2`` (per-site CSA params, then
        ``b0, b1``).
    is_free : ndarray(bool)
        Same length as ``params0``.
    labels : list of str
        One label per site, in the same order as the per-site blocks
        in ``params0``.
    """
    peak_defs, labels = _normalise_peak_defs(p0)
    n_sites = len(peak_defs)

    values = []
    for site in peak_defs:
        if "amplitude" in site:
            amplitude = site["amplitude"]
        else:
            if x is None or y is None:
                raise ValueError(
                    'Every site missing "amplitude" in p0 needs x and y supplied '
                    'to estimate it as max(abs(y)) / n_sites, or supply '
                    '"amplitude" explicitly.'
                )
            amplitude = np.max(np.abs(y)) / n_sites
        values.extend([
            site["d_iso"], site["span_or_Delta"], site["skew_or_eta"],
            site["fwhm"], amplitude, site.get("eta", 0.0),
        ])

    fixed_matrix = _resolve_fixed_matrix(fixed, n_sites, labels)
    # A site's own "fixed" (as an isotropic peak definition writes it,
    # e.g. {"d_iso": True, "fwhm": False}) -- previously ignored, so a
    # parameter marked fixed inside p0 was silently fitted.
    for site_idx, site in enumerate(peak_defs):
        own = site.get("fixed")
        if not own:
            continue
        names = ([n for n, is_fixed in own.items() if is_fixed] if isinstance(own, dict)
                 else list(own))
        bad = [n for n in names if n not in _CSA_PARAM_NAMES]
        if bad:
            raise ValueError(f"Unknown parameter name(s) {bad} in p0[{site_idx}]['fixed'].")
        fixed_matrix[site_idx].update(names)
    is_free = []
    for site_idx in range(n_sites):
        is_free.extend(name not in fixed_matrix[site_idx] for name in _CSA_PARAM_NAMES)

    values.extend([b0_init, b1_init])
    is_free.extend([True, True])  # flipped to fixed by fit_csa_spectrum, cf. fit_spectrum

    return np.array(values, dtype=float), np.array(is_free, dtype=bool), labels


def sideband_window_warning(d_iso_ppm, mas_Hz, larmor_Hz, fit_regions=None, x_ppm=None):
    """A warning (str) when a finite-MAS fit window holds no first-order
    spinning sideband of a site, else ``None``.

    The sidebands, at ``d_iso +/- mas_Hz / larmor_Hz`` (ppm), are what
    pin the MAS rate *and* the anisotropy. Without them the fit only
    sees the centreband, which every MAS rate reproduces equally well:
    on a 79Br KBr spectrum at 20 kHz, a +/-40 ppm window gave R^2 0.9933
    at 20, 40 and 60 kHz alike, with Delta = -30, +240 and +485 ppm.

    ``fit_regions`` as :func:`fit_csa_spectrum` takes them (``None`` =
    the whole of ``x_ppm``)."""
    if not mas_Hz or not larmor_Hz:
        return None
    spacing = float(mas_Hz) / float(larmor_Hz) * 1.0e6
    if fit_regions is None:
        if x_ppm is None:
            return None
        windows = [(float(np.min(x_ppm)), float(np.max(x_ppm)))]
    else:
        windows = [(min(lo, hi), max(lo, hi)) for lo, hi in fit_regions]
    sidebands = (d_iso_ppm - spacing, d_iso_ppm + spacing)
    if any(lo <= sb <= hi for sb in sidebands for lo, hi in windows):
        return None
    return (f"The fit window holds no spinning sideband of the site at {d_iso_ppm:.4g} ppm "
            f"(first sidebands at {sidebands[1]:.4g} and {sidebands[0]:.4g} ppm for "
            f"{mas_Hz / 1e3:.4g} kHz). Without sidebands the MAS rate and the anisotropy are "
            f"not determined by this fit -- widen the window to include them.")


def merge_site_bounds(p0, bounds=None):
    """``bounds`` plus each site's own ``"bounds"`` (``{name: (lo, hi)}``
    inside its p0 dict, as an isotropic peak definition writes them),
    keyed by site label for :func:`build_csa_bounds`. A site's own entry
    wins over a ``bounds`` entry for the same site and parameter."""
    peak_defs, labels = _normalise_peak_defs(p0)
    merged = dict(bounds or {})
    for label, site in zip(labels, peak_defs):
        own = site.get("bounds")
        if own:
            merged[label] = {**dict(merged.get(label, {})), **dict(own)}
    return merged or None


def build_csa_bounds(n_sites, labels=None, convention="Haeberlen", bounds=None,
                      positive_intensity=True, allow_negative_intensity=False):
    """
    Construct lower/upper bounds for all sites' ``(d_iso,
    span_or_Delta, skew_or_eta, fwhm, amplitude)``, concatenated in
    site order (baseline bounds are appended separately by the
    caller, as before).

    Defaults (per site): ``d_iso`` and ``span_or_Delta`` unbounded
    (Haeberlen's ``Delta`` can be signed; pass ``bounds={"span_or_Delta":
    (0, np.inf)}`` for a Herzfeld-Berger ``span``, which is
    non-negative by definition). ``skew_or_eta`` defaults to (0, 1)
    for Haeberlen ``eta`` or (-1, 1) for Herzfeld-Berger ``skew``.
    ``fwhm`` defaults to (1e-6, inf). ``amplitude`` defaults to (0,
    inf) unless negative intensities are allowed (mirrors
    :func:`nmrtools.fitting.build_bounds`).

    Parameters
    ----------
    n_sites : int
    labels : list of str, optional
        Site labels, as returned by :func:`initialise_csa_params`.
        Required if ``bounds`` uses label keys for per-site overrides;
        otherwise only needed for warning messages (site index is
        used if omitted).
    bounds : dict, optional
        Global per-parameter ``(lower, upper)`` overrides, applied to
        every site, keyed by name from
        ``["d_iso","span_or_Delta","skew_or_eta","fwhm","amplitude"]``;
        may be combined in the same dict with per-site overrides keyed
        by site index or label, each mapping to a ``{name: (lo, hi)}``
        dict, e.g. ``bounds={"fwhm": (0.1, 50), 0: {"d_iso": (30, 50)}}``.
    """
    labels = list(labels) if labels is not None else list(range(n_sites))
    if len(labels) != n_sites:
        raise ValueError(f"labels has length {len(labels)}, expected n_sites={n_sites}.")

    default_skew_or_eta = (0.0, 1.0) if convention == "Haeberlen" else (-1.0, 1.0)
    amp_lower = 0.0 if (positive_intensity and not allow_negative_intensity) else -np.inf

    base_defaults = {
        "d_iso": (-np.inf, np.inf),
        "span_or_Delta": (-np.inf, np.inf),
        "skew_or_eta": default_skew_or_eta,
        "fwhm": (1e-6, np.inf),
        "amplitude": (amp_lower, np.inf),
        "eta": (0.0, 1.0),
    }

    per_site = [dict(base_defaults) for _ in range(n_sites)]
    bounds = bounds or {}

    for key, val in bounds.items():
        if key not in _CSA_PARAM_NAMES:
            continue
        for site_bounds in per_site:
            site_bounds[key] = val

    for key, val in bounds.items():
        if key in _CSA_PARAM_NAMES:
            continue
        idx = _resolve_site_index(key, labels)
        if not isinstance(val, dict):
            raise TypeError(f"Per-site bounds[{key!r}] must be a dict of {{name: (lo, hi)}}.")
        for name, bnd in val.items():
            if name not in _CSA_PARAM_NAMES:
                raise ValueError(f"Unknown parameter name {name!r} in bounds[{key!r}].")
            per_site[idx][name] = bnd

    lower, upper = [], []
    for i, site_bounds in enumerate(per_site):
        for name in _CSA_PARAM_NAMES:
            lo, hi = site_bounds[name]
            if lo > hi:
                print(f"Warning: swapped bounds for site {labels[i]!r} {name}: "
                      f"{(lo, hi)} -> fixing order")
                lo, hi = hi, lo
            lower.append(lo)
            upper.append(hi)

    return np.array(lower), np.array(upper)


# ============================================================
# Fitting
# ============================================================

def fit_csa_spectrum(results, p0, convention="Haeberlen", static=True,
                      larmor_Hz=None, mas_Hz=None, rotor_angle="magic",
                      n_theta=60, n_phi=120, use_octant_symmetry=True,
                      n_time=1024, n_sidebands=15, subtract_residual_mean=True,
                      powder_scheme='rectangular', n_zcw=None,
                      lineshape=None,
                      fit_regions=None, use_normalised=True,
                      baseline_regions=None, baseline_sigma=2.5, baseline_max_iter=10,
                      bounds=None, fixed=None,
                      positive_intensity=True, allow_negative_intensity=False):
    """
    Fit a static or finite-MAS CSA powder lineshape -- one or more
    sites -- to an experimental 1D spectrum by nonlinear least
    squares.

    Mirrors :func:`nmrtools.fitting.fit_spectrum`'s workflow (a robust
    linear baseline is estimated and held fixed, then ``curve_fit``
    optimises the remaining parameters), but the model is the CSA
    powder simulation in :mod:`nmrtools.csa_quad_simulator` rather
    than a closed-form peak shape -- see the module docstring for why
    that means the model is always evaluated on the full experimental
    ppm axis and then subset, rather than pointwise at arbitrary x.
    For multi-site ``p0``, every site is simulated independently and
    summed -- see :func:`build_csa_model`.

    Parameters
    ----------
    results : dict
        Output of :func:`nmrtools.one_d.NMR1dimport`.
    p0 : dict or list of dict
        Initial guess -- one site, or a list of them. See
        :func:`initialise_csa_params` and the module docstring.
    convention : {'Haeberlen', 'Herzfeld-Berger'}, optional
    static : bool, optional
        If False, fit a finite-speed-MAS sideband pattern instead of a
        static powder pattern (see :func:`csa_model_full`). Applies to
        every site.
    larmor_Hz : float, optional
        Only used if ``static=False``. If None, taken from
        ``results['SFO1']`` (MHz) * 1e6.
    mas_Hz : float, optional
        Required if ``static=False`` -- the spinning speed is not
        recorded in ``results`` and must be supplied explicitly.
    rotor_angle, n_time, n_sidebands, subtract_residual_mean : optional
        Only used if ``static=False``. See
        :func:`nmrtools.csa_quad_simulator.finite_mas_csa_spectrum`.
    n_theta, n_phi, use_octant_symmetry : optional
        Powder-grid size, shared by every site. Keep coarse for
        fitting (defaults 60/120) and refine for the final plot -- the
        powder-average loop, not the ppm-axis length, dominates the
        cost of each ``curve_fit`` iteration, and that cost is paid
        once per site per iteration.
    powder_scheme, n_zcw : optional
        Powder-averaging scheme -- see :func:`csa_model_full`. Worth
        trying ``powder_scheme='zcw'`` for finite-MAS fits (``static=
        False``) since that's where each ``curve_fit`` iteration is
        most expensive; check the fit result is unchanged when you
        raise ``n_zcw`` before trusting a low value.
    lineshape : {'gaussian', 'lorentzian', 'pseudo_voigt'}, optional
        Broadening kernel convolved onto every site's simulated
        powder pattern (shared by all sites in this fit). Default
        'gaussian' (unchanged behaviour from earlier versions). With
        ``lineshape='pseudo_voigt'``, each site's own ``eta`` (its
        Gaussian/Lorentzian mixing fraction, 0-1 -- see the note by
        ``_CSA_PARAM_NAMES``, NOT the CSA asymmetry ``skew_or_eta``)
        becomes a genuinely fittable per-site parameter, initialised
        from ``p0[i].get("eta", 0.0)`` and controllable via the usual
        per-site ``fixed``/``bounds`` machinery. With ``lineshape=
        'gaussian'`` or ``'lorentzian'``, ``eta`` has no broadening
        effect (those kernels don't mix), so it is forced fixed here
        (at 0.0 or 1.0 respectively) regardless of any ``fixed``
        setting -- a warning is printed if you had asked for it free.
    fit_regions : list of tuple, optional
        ``[(xmin, xmax), ...]`` in ppm. If None, the complete spectrum
        is fitted.
    use_normalised : bool, optional
        Kept for compatibility; no longer changes the fit. The optimiser
        **always** works on the spectrum scaled to its maximum absolute
        intensity (``scale``), because fitting raw intensities (~1e10 on
        a real spectrum) leaves it badly conditioned: on a 79Br KBr
        dataset ``use_normalised=False`` stopped at R^2 0.966 /
        Delta -117.8 ppm against 0.984 / -110.0 ppm scaled. Everything
        returned is in raw units either way.
    baseline_regions, baseline_sigma, baseline_max_iter :
        See :func:`nmrtools.fitting.estimate_baseline_robust`. The
        baseline is estimated once, globally, and shared by all sites.
    bounds : dict, optional
        See :func:`build_csa_bounds` -- global and/or per-site.
    fixed : list or dict, optional
        CSA/lineshape parameter names to hold fixed at their ``p0``
        value, global and/or per-site -- see
        :func:`initialise_csa_params`. The baseline is always held
        fixed, matching :func:`nmrtools.fitting.fit_spectrum`.
    positive_intensity, allow_negative_intensity : bool, optional
        See :func:`nmrtools.fitting.fit_spectrum`. Applies to every
        site's amplitude.

    Returns
    -------
    popt : ndarray
        Per-site ``(d_iso, span_or_Delta, skew_or_eta, fwhm,
        amplitude, eta)`` blocks in site order, followed by ``(b0, b1)``.
        Length ``6 * n_sites + 2``.
    pcov_free : ndarray
        Covariance of the free parameters only -- expand with
        :func:`nmrtools.fitting.expand_pcov`.
        In raw units, like ``popt``: the amplitude rows/columns are
        rescaled by ``scale``. (They were not, so amplitude
        uncertainties -- and the integral errors
        :mod:`csa_relaxation` derives from them -- came out too small
        by that factor. Shape parameters were unaffected.)
    is_free : ndarray(bool)
    scale : float
    fit_info : dict
        ``{"sim_kwargs": ..., "n_sites": ..., "labels": ..., "x_ref": ...}``.
        ``sim_kwargs`` (``convention``, ``static``, ``larmor_Hz``,
        ``mas_Hz``, grid sizes, ...) is forwarded to
        :func:`csa_model_full` per site; pass the whole ``fit_info``
        dict on to :func:`evaluate_csa_fit` /
        :func:`NMR1dplot_with_csa_fit` so the fitted model is
        re-evaluated consistently.
    """

    x_full = results["x_ppm"]

    # Always condition the optimiser (see use_normalised above); every
    # intensity-axis quantity is converted in and out of this scale.
    y_raw = np.asarray(results["real_spectrum"], dtype=float)
    scale = float(np.max(np.abs(y_raw))) if y_raw.size else 1.0
    scale = scale if scale != 0 else 1.0
    y_full = y_raw / scale
    lineshape = resolve_lineshape(p0, lineshape)

    # The CSA simulator bins its powder average onto a uniform,
    # ascending ppm axis (see the module docstring), so sort once here
    # and mask/fit on the sorted axis throughout.
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

    p0_arr, is_free, labels = initialise_csa_params(
        p0, b0_init, b1_init, x=x_fit, y=y_fit, fixed=fixed,
    )
    is_free[-2:] = False  # baseline held fixed, as in fit_spectrum
    n_sites = len(labels)
    amp_idx = np.arange(n_sites) * 6 + 4
    # An amplitude the caller supplied is in raw units; the optimiser
    # works in scaled ones. (Estimated amplitudes already are.)
    for site_i, site in enumerate(_normalise_peak_defs(p0)[0]):
        if "amplitude" in site:
            p0_arr[amp_idx[site_i]] = float(site["amplitude"]) / scale

    if not static and larmor_Hz is None:
        larmor_Hz = results["SFO1"] * 1.0e6

    # `eta` (per-site broadening-kernel mixing fraction) only has any
    # effect on the model when lineshape='pseudo_voigt' (see
    # csa_model_full/pseudo_voigt_broadening) -- for 'gaussian' or
    # 'lorentzian' it is degenerate (the model's derivative w.r.t. eta
    # is exactly zero), so leaving it free would waste curve_fit
    # iterations on an unidentifiable parameter. Force it fixed at the
    # value that matches the chosen kernel, regardless of `fixed`.
    lineshape_key = str(lineshape).lower().replace("-", "_")
    eta_idx = np.arange(n_sites) * 6 + 5
    if lineshape_key in ("gaussian", "lorentzian"):
        if np.any(is_free[eta_idx]):
            print(
                f"Note: lineshape={lineshape!r} has no Gaussian/Lorentzian "
                f"mixing to fit -- forcing every site's eta fixed "
                f"({'0.0' if lineshape_key == 'gaussian' else '1.0'})."
            )
        p0_arr[eta_idx] = 0.0 if lineshape_key == "gaussian" else 1.0
        is_free[eta_idx] = False
    elif lineshape_key not in ("pseudo_voigt", "pv"):
        raise ValueError(
            f"Unknown lineshape {lineshape!r}; use 'gaussian', 'lorentzian', "
            f"or 'pseudo_voigt'."
        )

    sim_kwargs = dict(
        convention=convention, static=static, larmor_Hz=larmor_Hz, mas_Hz=mas_Hz,
        rotor_angle=rotor_angle, n_theta=n_theta, n_phi=n_phi,
        use_octant_symmetry=use_octant_symmetry, n_time=n_time,
        n_sidebands=n_sidebands, subtract_residual_mean=subtract_residual_mean,
        powder_scheme=powder_scheme, n_zcw=n_zcw, lineshape=lineshape,
    )

    lower, upper = build_csa_bounds(
        n_sites, labels=labels, convention=convention,
        bounds=merge_site_bounds(p0, bounds),
        positive_intensity=positive_intensity,
        allow_negative_intensity=allow_negative_intensity,
    )
    lower = np.append(lower, [-np.inf, -np.inf])
    upper = np.append(upper, [np.inf, np.inf])
    # Amplitude bounds are given in raw units, like amplitudes.
    lower[amp_idx] = lower[amp_idx] / scale
    upper[amp_idx] = upper[amp_idx] / scale

    assert len(p0_arr) == len(is_free) == len(lower) == len(upper), "parameter length mismatch"

    all_names = _all_param_names(labels)
    free_idx = np.flatnonzero(is_free)
    start_notes = []
    for i in free_idx:
        lo, hi = lower[i], upper[i]
        if not (lo <= p0_arr[i] <= hi):
            # A start outside its own bounds (e.g. bounds set after the
            # starting value) made curve_fit fail with only "Initial
            # guess is outside of provided bounds". Start at the nearest
            # bound instead, and say so.
            moved = float(np.clip(p0_arr[i], lo, hi))
            start_notes.append(f"{all_names[i]}: starting value {p0_arr[i]:g} is outside its "
                               f"bounds ({lo:g}, {hi:g}); started at {moved:g} instead")
            print("Note: " + start_notes[-1])
            p0_arr[i] = moved

    def model_wrapper(x, *free_params):
        full = p0_arr.copy()
        full[is_free] = free_params
        # `x` here is always x_fit (== x_sorted[mask]), as passed by
        # curve_fit -- the model is simulated on the full sorted axis
        # and then masked, so the argument itself is not reused below.
        y_model_full = build_csa_model(x_sorted, full, n_sites, x_ref=x_ref, **sim_kwargs)
        return y_model_full[mask]

    popt_free, pcov_free = curve_fit(
        model_wrapper, x_fit, y_fit, p0=p0_arr[is_free],
        bounds=(lower[is_free], upper[is_free]), maxfev=20000,
    )

    popt = p0_arr.copy()
    popt[is_free] = popt_free

    # Every intensity-axis quantity back to raw units: amplitudes and
    # the baseline in popt, and the matching rows/columns of pcov.
    popt[amp_idx] *= scale   # amplitude, every site
    popt[-2:] *= scale       # baseline b0, b1
    unit = np.ones(len(popt))
    unit[amp_idx] = scale
    unit[-2:] = scale
    free_unit = unit[is_free]
    pcov_free = np.asarray(pcov_free) * np.outer(free_unit, free_unit)

    warnings = list(start_notes)
    if not static:
        for site_i in range(n_sites):
            text = sideband_window_warning(popt[site_i * 6], mas_Hz, larmor_Hz,
                                           fit_regions=fit_regions, x_ppm=x_sorted)
            if text and text not in warnings:
                warnings.append(text)

    fit_info = {
        "sim_kwargs": sim_kwargs,
        "n_sites": n_sites,
        "labels": labels,
        "x_ref": x_ref,
        # What the fit window was, so a report can say it.
        "fit_regions": (None if fit_regions is None
                        else [(float(min(a, b)), float(max(a, b))) for a, b in fit_regions]),
        # Things to read before trusting the numbers (see
        # sideband_window_warning).
        "warnings": warnings,
    }

    return popt, pcov_free, is_free, scale, fit_info


def csa_site_amounts(x_ppm, popt, fit_info, normalise=True):
    """Each fitted site's ``amount`` -- its relative number of nuclei,
    the weight Simulate > CSA (csa_multisite_spectrum) uses -- from the
    fit's amplitudes. A fitted amplitude scales a peak-normalised shape,
    so a narrow site and a broad one of equal amplitude hold very
    different numbers of nuclei; dividing by each shape's un-normalised
    peak height on the fit's own axis undoes that. With ``normalise``
    the largest amount is 1."""
    from .simulations import site_amounts_from_amplitudes
    n_sites = fit_info["n_sites"]
    x_sorted = np.sort(np.asarray(x_ppm, dtype=float))
    params = np.asarray(popt[: 6 * n_sites], dtype=float).reshape(n_sites, 6)
    heights = [float(np.max(csa_model_full(x_sorted, d, a, s, w, eta=e, normalised=False,
                                           **fit_info["sim_kwargs"])))
               for d, a, s, w, _amp, e in params]
    return site_amounts_from_amplitudes(params[:, 4], heights, normalise=normalise)


def evaluate_csa_fit(x_ppm, popt, fit_info, return_components=False):
    """
    Evaluate the fitted (possibly multi-site) CSA model at ``x_ppm``
    (any order/spacing -- it is sorted internally and un-sorted back
    to match the input order).

    Parameters
    ----------
    popt : ndarray
        As returned by :func:`fit_csa_spectrum`.
    fit_info : dict
        As returned by :func:`fit_csa_spectrum`.
    return_components : bool, optional
        If True, also return the per-site components followed by the
        baseline: ``[site_1, site_2, ..., site_N, baseline]``, in the
        same order as ``fit_info["labels"]``.
    """
    n_sites = fit_info["n_sites"]
    x_ref = fit_info["x_ref"]
    sim_kwargs = fit_info["sim_kwargs"]

    x_ppm = np.asarray(x_ppm)
    order = np.argsort(x_ppm)
    x_sorted = x_ppm[order]

    csa_params = np.asarray(popt[: 6 * n_sites]).reshape(n_sites, 6)
    b0, b1 = popt[6 * n_sites], popt[6 * n_sites + 1]

    site_components_sorted = []
    for d_iso, span_or_Delta, skew_or_eta, fwhm, amplitude, eta in csa_params:
        shape = csa_model_full(x_sorted, d_iso, span_or_Delta, skew_or_eta, fwhm,
                                eta=eta, **sim_kwargs)
        site_components_sorted.append(amplitude * shape)

    baseline_sorted = b0 + b1 * (x_sorted - x_ref)
    y_sorted = baseline_sorted.copy()
    for comp in site_components_sorted:
        y_sorted = y_sorted + comp

    y = np.empty_like(y_sorted)
    y[order] = y_sorted

    if not return_components:
        return y

    components = []
    for comp_sorted in site_components_sorted:
        comp = np.empty_like(comp_sorted)
        comp[order] = comp_sorted
        components.append(comp)
    baseline = np.empty_like(baseline_sorted)
    baseline[order] = baseline_sorted
    components.append(baseline)

    return y, components


def check_csa_bound_hits(popt, pcov, is_free, lower, upper, labels):
    """Flag any free CSA/lineshape parameter (any site) whose fitted
    value (plus or minus its uncertainty) reaches a bound -- a sign
    the bound may be too tight."""
    warnings = []
    for site_idx, label in enumerate(labels):
        for j, name in enumerate(_CSA_PARAM_NAMES):
            i = site_idx * 6 + j
            if not is_free[i]:
                continue
            val, lo, hi = popt[i], lower[i], upper[i]
            err = np.sqrt(max(pcov[i, i], 0.0))
            tag = f"{label}: {name}"
            if np.isfinite(hi) and val + err >= hi:
                warnings.append(f"{tag} near upper bound {hi}")
            if np.isfinite(lo) and val - err <= lo:
                warnings.append(f"{tag} near lower bound {lo}")
    return warnings


def print_csa_fit_report(x, y, y_fit, popt, pcov, is_free, labels, convention="Haeberlen",
                          warnings=None, save_txt=False, txt_filename="csa_fit_report.txt"):
    """
    Print (and optionally save) a human-readable report of the fitted
    CSA parameters (all sites), uncertainties, and fit-quality
    metrics.

    Parameters
    ----------
    y_fit : ndarray
        Model evaluated at ``x`` -- see :func:`evaluate_csa_fit`.
    labels : list of str
        Site labels, as returned by :func:`fit_csa_spectrum` (in
        ``fit_info["labels"]``).
    """
    errors = np.sqrt(np.diag(pcov))
    n_params = len(popt)
    metrics = fit_quality(y, y_fit, n_params)

    span_or_delta_label = "Delta" if convention == "Haeberlen" else "span"
    skew_or_eta_label = "eta" if convention == "Haeberlen" else "skew"
    # Deliberately NOT called "eta" -- under Haeberlen convention
    # `skew_or_eta_label` above is already "eta" (the CSA tensor's own
    # asymmetry parameter); this is the unrelated broadening-kernel
    # Gaussian/Lorentzian mixing fraction (see the note by
    # `_CSA_PARAM_NAMES`), and needs a visibly different label here or
    # the two are impossible to tell apart in the printed report.
    per_site_names = ["d_iso", span_or_delta_label, skew_or_eta_label, "fwhm",
                       "amplitude", "pv_mix_eta"]

    names = []
    for label in labels:
        names.extend(f"{label}: {name}" for name in per_site_names)
    names.extend(["b0", "b1"])

    report = []

    def write(line=""):
        print(line)
        report.append(str(line))

    n_sites = len(labels)
    write("\n" + "=" * 60)
    write(f"CSA FIT REPORT ({convention} convention, "
          f"{n_sites} site{'s' if n_sites != 1 else ''})")
    write("=" * 60)
    write("\nParameters:")
    write("-" * 60)

    name_width = max(12, max(len(n) for n in names) + 1)
    for name, val, err, is_f in zip(names, popt, errors, is_free):
        if is_f:
            write(f"  {name:<{name_width}}= {val:.6f} \u00b1 {err:.6f}")
        else:
            write(f"  {name:<{name_width}}= {val:.6f} (fixed)")

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

def NMR1dplot_with_csa_fit(results, popt, fit_info, fit_regions=None,
                            xlims='default', xlabel=None, pl_width=3.375, pl_height=3.375,
                            font='Arial', fontsize=12, output_format='',
                            show_components=False, colour_data='black', colour_fit='red',
                            colour_baseline='gray', colour_components=None,
                            tick_dir='in', tick_length=4.0, tick_width=1.0, tick_pad=4.0,
                            tick_number=None, field='', use_normalised=True,
                            save_fig=False, fig_name='nmr_csa_fit_plot', save_format='png'):
    """
    Plot a 1D spectrum together with its fitted (possibly multi-site)
    CSA model.

    Parameters
    ----------
    results : dict
        Output of :func:`nmrtools.one_d.NMR1dimport`.
    popt, fit_info :
        As returned by :func:`fit_csa_spectrum`.
    fit_regions : list of tuple, optional
        Region(s) actually shown/compared, ``[(xmin, xmax), ...]``. If
        None, the complete spectrum is shown.
    xlims : 'default' or [left, right], optional
    xlabel : str, optional
        Override the automatically-generated (nuclide-based) x-axis
        label.
    pl_width, pl_height : float, optional
        Figure size in inches.
    font, fontsize : optional
        Typeface settings, see :func:`nmrtools.plotting.setup_plotting`.
    show_components : bool, optional
        If True, also draw each site's individual lineshape (labelled
        in the legend by its site label) and the fixed linear
        baseline.
    colour_components : list of colour, optional
        Colours cycled across sites when ``show_components=True``.
        Defaults to matplotlib's current colour cycle.
    use_normalised : bool, optional
        Must match the ``use_normalised`` value used when fitting, so
        data and model are plotted on the same intensity scale.
    tick_dir : {'in', 'out', 'inout'}, optional
    tick_length, tick_width, tick_pad : float, optional
    tick_number : int, optional
    save_fig : bool, optional
        If True, save the figure (see :func:`nmrtools.plotting.save_figure`).
    fig_name, save_format : optional

    Returns
    -------
    fig, ax
    """
    style = setup_plotting(font=font, fontsize=fontsize, pl_width=pl_width,
                            pl_height=pl_height, output_format=output_format)
    pl_width, pl_height = style['pl_width'], style['pl_height']

    labels = fit_info["labels"]
    x_ref = fit_info["x_ref"]

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

    # popt (from fit_csa_spectrum) is always in raw intensity units,
    # regardless of use_normalised -- amplitudes and baseline are both
    # rescaled back after fitting (see fit_csa_spectrum). So the
    # evaluated model is in raw units too, and must be divided by the
    # same `scale` as the data above to land on the same axis.
    if show_components:
        y_model_full, components = evaluate_csa_fit(x_sorted, popt, fit_info,
                                                      return_components=True)
        y_model = y_model_full[mask] / scale
        components = [c[mask] / scale for c in components]
    else:
        y_model = evaluate_csa_fit(x_sorted, popt, fit_info)[mask] / scale
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

def NMRCSAFitWorkflow(datapath, p0, procno=1, convention="Haeberlen", static=True,
                       larmor_Hz=None, mas_Hz=None, rotor_angle="magic",
                       n_theta=60, n_phi=120, use_octant_symmetry=True,
                       n_time=1024, n_sidebands=15, subtract_residual_mean=True,
                       powder_scheme='rectangular', n_zcw=None, lineshape=None,
                       fit_regions=None, baseline_regions=None,
                       baseline_sigma=2.5, baseline_max_iter=10, normalise=True,
                       bounds=None, fixed=None,
                       positive_intensity=True, allow_negative_intensity=False,
                       show_plot=True, xlims='default', xlabel=None,
                       pl_width=3.375, pl_height=3.375, font='Arial', fontsize=12,
                       output_format='', tick_dir='in', tick_length=4.0,
                       tick_width=1.0, tick_pad=4.0, tick_number=None,
                       show_components=False, show_report=True,
                       save_csv=False, csv_filename="csa_fit_results.csv",
                       save_report=False, report_filename="csa_fit_report.txt",
                       save_fig=False, fig_name='nmr_csa_fit_plot', save_format='png'):
    """
    Fit a static or finite-MAS CSA powder lineshape -- one or more
    sites -- to a processed 1D Bruker spectrum.

    The CSA analogue of :func:`nmrtools.fitting.NMRFitWorkflow`:
    imports the spectrum, fits the CSA tensor parameters for every
    site (plus a shared linewidth-per-site, amplitude-per-site, and
    fixed linear baseline) by nonlinear least squares against the
    powder simulation in :mod:`nmrtools.csa_quad_simulator`, and
    optionally plots and reports the result.

    Parameters
    ----------
    datapath : str
        Path to the Bruker experiment directory.
    p0 : dict or list of dict
        Initial guess for one CSA site, or a list of such dicts (one
        per site), each fitted as an independent CSA lineshape sharing
        a single linear baseline. Required keys per site: "d_iso",
        "span_or_Delta", "skew_or_eta", "fwhm"; optional: "amplitude"
        (estimated as ``max(abs(y)) / n_sites`` if omitted -- a rough
        equal-height guess; supply it explicitly if sites differ a lot
        in intensity), "eta" (broadening-kernel Gaussian/Lorentzian
        mixing fraction, 0-1, only meaningful with ``lineshape=
        'pseudo_voigt'`` below -- defaults to 0.0; NOT the CSA
        asymmetry, which is "skew_or_eta"), and "label" (defaults to
        "Site 1", "Site 2", ... -- used in the legend, CSV columns,
        and fit report). See :func:`initialise_csa_params`.
    procno : int, optional
    convention, static, larmor_Hz, mas_Hz, rotor_angle, n_theta, n_phi,
    use_octant_symmetry, n_time, n_sidebands, subtract_residual_mean,
    powder_scheme, n_zcw, lineshape,
    fit_regions, baseline_regions, baseline_sigma, baseline_max_iter,
    bounds, fixed, positive_intensity, allow_negative_intensity :
        See :func:`fit_csa_spectrum`. ``bounds``/``fixed`` accept
        per-site overrides keyed by site index or label.
    normalise : bool, optional
        Fit (and plot) the spectrum normalised to its maximum absolute
        intensity.
    show_plot, show_components, show_report : bool, optional
    xlims, xlabel, pl_width, pl_height, font, fontsize, output_format,
    tick_dir, tick_length, tick_width, tick_pad, tick_number :
        Forwarded to :func:`NMR1dplot_with_csa_fit`.
    save_csv, csv_filename, save_report, report_filename : optional
    save_fig, fig_name, save_format : optional

    Returns
    -------
    dict with keys ``popt``, ``pcov`` (full, expanded), ``fit_info``,
    ``labels``, ``x``, ``y``, ``y_fit``, ``components``, ``figure``,
    ``axes``.

    Notes
    -----
    For each site, parameters are ordered ``d_iso, span_or_Delta,
    skew_or_eta, fwhm, amplitude, eta``; sites are concatenated in the
    order given in ``p0``, followed by the shared baseline ``b0, b1``.
    The fitted model is ``sum_i(amplitude_i * csa_lineshape_i) + b0 +
    b1 * (x - x_ref)``.
    """

    setup_plotting()

    results = NMR1dimport(datapath, procno=procno)

    popt, pcov_free, is_free, scale, fit_info = fit_csa_spectrum(
        results, p0, convention=convention, static=static,
        larmor_Hz=larmor_Hz, mas_Hz=mas_Hz, rotor_angle=rotor_angle,
        n_theta=n_theta, n_phi=n_phi, use_octant_symmetry=use_octant_symmetry,
        n_time=n_time, n_sidebands=n_sidebands,
        subtract_residual_mean=subtract_residual_mean,
        powder_scheme=powder_scheme, n_zcw=n_zcw, lineshape=lineshape,
        fit_regions=fit_regions, use_normalised=normalise,
        baseline_regions=baseline_regions, baseline_sigma=baseline_sigma,
        baseline_max_iter=baseline_max_iter, bounds=bounds, fixed=fixed,
        positive_intensity=positive_intensity,
        allow_negative_intensity=allow_negative_intensity,
    )

    n_sites = fit_info["n_sites"]
    labels = fit_info["labels"]

    pcov = expand_pcov(pcov_free, is_free, len(popt))

    lower, upper = build_csa_bounds(
        n_sites, labels=labels, convention=convention, bounds=merge_site_bounds(p0, bounds),
        positive_intensity=positive_intensity,
        allow_negative_intensity=allow_negative_intensity,
    )
    lower = np.append(lower, [-np.inf, -np.inf])
    upper = np.append(upper, [np.inf, np.inf])
    warnings = list(fit_info.get("warnings", [])) + check_csa_bound_hits(popt, pcov, is_free, lower, upper, labels)

    x_full = results['x_ppm']
    order = np.argsort(x_full)
    x_sorted = x_full[order]

    # popt/y_model (below) are always raw intensity units regardless
    # of `normalise` -- fit_csa_spectrum only uses use_normalised as
    # an internal numerical-conditioning trick and rescales amplitudes
    # and baseline back afterwards (see fit_csa_spectrum). So the data
    # used for the report and CSV export must also be raw units to
    # match popt/y_model, not divided by `scale`. `normalise` still
    # controls the *plotted* display scale, via use_normalised in
    # NMR1dplot_with_csa_fit below.
    y_sorted = results['real_spectrum'][order]

    mask = build_region_mask(x_sorted, fit_regions) if fit_regions is not None \
        else np.ones_like(x_sorted, dtype=bool)
    x_fit = x_sorted[mask]
    y_fit_data = y_sorted[mask]

    if show_components:
        y_model_full, components_full = evaluate_csa_fit(
            x_sorted, popt, fit_info, return_components=True,
        )
        y_model = y_model_full[mask]
        components = [c[mask] for c in components_full]
    else:
        y_model = evaluate_csa_fit(x_sorted, popt, fit_info)[mask]
        components = None

    fig, ax = None, None
    if show_plot:
        fig, ax = NMR1dplot_with_csa_fit(
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
        print_csa_fit_report(x_fit, y_fit_data, y_model, popt, pcov, is_free, labels,
                              convention=convention, warnings=warnings,
                              save_txt=save_report, txt_filename=report_filename)

    if save_csv:
        import csv

        columns = {"x": x_fit, "y": y_fit_data, "y_fit": y_model}
        if components is not None:
            for label, comp in zip(labels, components[:-1]):
                columns[f"{label}_component"] = comp
            columns["baseline"] = components[-1]

        with open(csv_filename, "w", newline="") as outfile:
            writer = csv.writer(outfile)
            writer.writerow(columns.keys())
            for i in range(len(x_fit)):
                writer.writerow([columns[col][i] for col in columns])

    return {
        "popt": popt, "pcov": pcov, "fit_info": fit_info, "labels": labels,
        "x": x_fit, "y": y_fit_data, "y_fit": y_model,
        "components": components, "figure": fig, "axes": ax,
    }
