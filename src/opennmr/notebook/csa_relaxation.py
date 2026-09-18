"""
CSA-lineshape-based T1/T2/T1rho relaxation fitting for nmrtools.

Bridges :mod:`nmrtools.csa_fitting` (static/finite-MAS CSA powder
lineshape fitting to a single 1D spectrum) and
:mod:`nmrtools.relaxation` (fitting a relaxation curve to a series of
per-slice integrals): fit a (possibly multi-site) CSA lineshape to
every slice of a pseudo-2D relaxation series, extract each site's
*integral* (area under its full simulated powder pattern -- NOT its
raw fitted "amplitude", see :func:`compute_csa_integrals`) as a
function of delay, and hand the resulting series straight to
:func:`nmrtools.relaxation.NMR2D_fit_relaxation`.

Workflow
--------

    NMR2dimport -> NMR2D_get_slices -> NMR2D_fit_csa_slices
                 -> nmrtools.relaxation.NMR2D_fit_relaxation
                 -> nmrtools.relaxation.NMR2D_plot_relaxation

or, end-to-end, :func:`NMR2D_csa_relaxation_workflow`.

Design note: freezing the CSA shape across the series
-------------------------------------------------------
A T1/T2/T1rho series is (absent exchange or dynamics) repeated
measurements of the *same* tensor -- only the population (amplitude)
changes with delay. Fitting the full nonlinear CSA lineshape
(``d_iso``, ``span_or_Delta``, ``skew_or_eta``, ``fwhm``)
independently on every slice is therefore both the wrong model (shape
parameters have no business varying with delay) and the wrong tool
statistically: shape parameters are poorly constrained on low-S/N
slices (near an inversion-recovery null point, or late in a T2
decay), and any resulting drift in the fitted centre/width biases the
very integral the relaxation curve is built from -- exactly the
failure mode :func:`nmrtools.fitting.NMR2D_fit_slices` already guards
against for closed-form peaks via ``sn_threshold`` amplitude gating.

``freeze_shape=True`` (the default here) fits the full nonlinear CSA
model (:func:`nmrtools.csa_fitting.fit_csa_spectrum`) only ONCE, on a
single high-S/N reference slice, then holds every site's shape
parameters fixed at those values for every other slice and fits only
the (per-site) amplitude and shared linear baseline -- which, with the
shape fixed, is a genuinely LINEAR problem, solved directly by
``scipy.optimize.lsq_linear`` rather than re-running ``curve_fit``
(and therefore the powder-average simulation) for every slice. Besides
being far cheaper, this is more robust for weak slices, since the
shape no longer has to be independently re-determined from a
vanishing signal.

This relies on every 'row' (or every 'column') in a given
:func:`nmrtools.two_d.NMR2D_get_slices` selection sharing an identical
x-axis, which is true by construction (see
:func:`nmrtools.two_d.NMR2d_get_trace`: a row always spans the same
F2 axis regardless of which F1 position it was taken at, and vice
versa for a column) -- :func:`NMR2D_fit_csa_slices` still checks this
per slice and falls back to re-evaluating the (still fixed, still
non-optimised) shape on that slice's own axis if it ever isn't true.

Set ``freeze_shape=False`` if you have a specific reason to expect the
tensor itself to evolve over the series (e.g. a temperature/exchange
process) -- but note this is both much slower (one full nonlinear,
multi-orientation powder fit per slice) and more fragile at low S/N.
"""

import numpy as np
from scipy.optimize import lsq_linear

from .fitting import build_region_mask, expand_pcov, fit_quality
from .two_d import NMR2dimport, NMR2D_get_slices, prepare_trace_for_fit, NMR2D_plot_slices
from .csa_fitting import (
    fit_csa_spectrum, csa_model_full, evaluate_csa_fit,
    check_csa_bound_hits, build_csa_bounds, print_csa_fit_report,
)
from .relaxation import NMR2D_fit_relaxation, NMR2D_plot_relaxation, print_relaxation_report, _select_series
from .plotting import save_figure

# numpy >= 2.0 renamed trapz -> trapezoid
_trapz = getattr(np, "trapezoid", None) or np.trapz


# ============================================================
# Integral of the full CSA pattern
# ============================================================

def compute_csa_integrals(x_sorted, popt, fit_info):
    """
    Integrate each fitted CSA site's *simulated lineshape* -- NOT the
    fitted "amplitude" -- over the full ppm axis, giving each site's
    total signed area under its powder pattern. The CSA analogue of
    :func:`nmrtools.fitting.compute_integrals`.

    This matters because :func:`nmrtools.csa_fitting.csa_model_full`
    returns a shape normalised to max(abs(shape)) == 1, not unit area
    -- "amplitude" alone is not a fair measure of total intensity
    between sites of different span/fwhm (a broad, short pattern and a
    narrow, tall one can share the same amplitude but very different
    total signal), and is not the right quantity to feed into a T1/T2
    fit.

    Parameters
    ----------
    x_sorted : ndarray
        The (ascending or not -- sorted internally) ppm axis the model
        was evaluated on.
    popt, fit_info :
        As returned by :func:`nmrtools.csa_fitting.fit_csa_spectrum`.

    Returns
    -------
    integrals : list of float
        One per site (baseline excluded), in ``fit_info["labels"]`` order.
    integrals_norm : list of float
        Each integral as a percentage of the (signed) sum across
        sites -- meaningful only when every site's integral shares
        the same sign (see :func:`nmrtools.fitting.compute_integrals`'s
        identical caveat).
    """
    x_sorted = np.asarray(x_sorted)
    order = np.argsort(x_sorted)
    x_s = x_sorted[order]

    _, components = evaluate_csa_fit(x_sorted, popt, fit_info, return_components=True)

    # components[:-1] are the per-site contributions, already
    # amplitude-scaled (components[-1] is the baseline) -- exactly the
    # same shape as fitting.compute_integrals' `components[:-1]`.
    integrals = [float(_trapz(np.asarray(c)[order], x_s)) for c in components[:-1]]

    total = np.sum(integrals)
    integrals_norm = [100 * a / total for a in integrals] if total != 0 else [np.nan] * len(integrals)

    return integrals, integrals_norm


def compute_csa_integral_errors(popt, pcov, fit_info, integrals):
    """
    Amplitude-uncertainty-only error propagation for
    :func:`compute_csa_integrals`, mirroring
    :func:`nmrtools.fitting.compute_integral_errors`: treats
    ``d_iso``/``span_or_Delta``/``skew_or_eta``/``fwhm`` as exact.
    This is only an approximation when those are independently fitted
    per slice (``freeze_shape=False``); it is EXACT when they are held
    fixed (``freeze_shape=True``, the default in
    :func:`NMR2D_fit_csa_slices`), since each site's integral is then
    strictly linear in its own amplitude.
    """
    n_sites = fit_info["n_sites"]
    csa_params = np.asarray(popt[: 5 * n_sites]).reshape(n_sites, 5)

    errors = []
    for i, row in enumerate(csa_params):
        amplitude = row[4]
        amp_idx = i * 5 + 4
        sigma_A = np.sqrt(max(pcov[amp_idx, amp_idx], 0.0))
        errors.append(np.nan if amplitude == 0 else abs(integrals[i]) * sigma_A / abs(amplitude))

    return errors


# ============================================================
# Linear amplitude-only fit (frozen shape)
# ============================================================

def _fit_frozen_shape_linear(x_masked, y_masked, shapes_masked, x_ref,
                              positive_intensity=True, allow_negative_intensity=False):
    """
    Fit per-site amplitudes + shared linear baseline by (bounded)
    linear least squares, given FIXED per-site shape arrays
    (``shapes_masked``, already evaluated on the masked/fit-region x).
    No powder simulation happens here -- this is the fast path used
    for every non-reference slice when ``freeze_shape=True``.

    Returns
    -------
    coeffs : ndarray, length ``n_sites + 2``
        ``[amplitude_1, ..., amplitude_N, b0, b1]``.
    pcov : ndarray, shape ``(n_sites + 2, n_sites + 2)``
        Covariance from the (unconstrained) normal equations -- an
        approximation when a bound is active, same spirit as
        ``curve_fit``'s own covariance estimate near an active bound.
    residual : ndarray
    """
    n_sites = len(shapes_masked)
    X = np.column_stack(list(shapes_masked) + [np.ones_like(x_masked), x_masked - x_ref])
    n_params = n_sites + 2

    amp_lower = 0.0 if (positive_intensity and not allow_negative_intensity) else -np.inf
    lower = np.array([amp_lower] * n_sites + [-np.inf, -np.inf])
    upper = np.full(n_params, np.inf)

    if np.isfinite(amp_lower):
        result = lsq_linear(X, y_masked, bounds=(lower, upper))
        coeffs = result.x
    else:
        coeffs, *_ = np.linalg.lstsq(X, y_masked, rcond=None)

    residual = y_masked - X @ coeffs
    dof = max(len(y_masked) - n_params, 1)
    sigma2 = np.sum(residual ** 2) / dof

    try:
        XtX_inv = np.linalg.pinv(X.T @ X)
    except np.linalg.LinAlgError:
        XtX_inv = np.linalg.pinv(X.T @ X + 1e-12 * np.eye(n_params))

    pcov = sigma2 * XtX_inv

    return coeffs, pcov, residual


def _select_reference_index(traces, reference):
    """Resolve ``reference`` ('max_amplitude', 'first', or an int) to
    a 0-based index into ``traces``."""
    if isinstance(reference, (int, np.integer)) and not isinstance(reference, bool):
        if 0 <= reference < len(traces):
            return int(reference)
        raise ValueError(f"reference index {reference} out of range for {len(traces)} slice(s).")
    if reference == "first":
        return 0
    if reference == "max_amplitude":
        return int(np.argmax([np.max(np.abs(tr["y"])) for tr in traces]))
    raise ValueError(f"Unknown reference {reference!r}; use 'max_amplitude', 'first', or an int.")


# ============================================================
# Per-slice CSA fitting
# ============================================================

def NMR2D_fit_csa_slices(slices, p0, mode="rows", convention="Haeberlen", static=True,
                          larmor_Hz=None, mas_Hz=None, rotor_angle="magic",
                          n_theta=60, n_phi=120, use_octant_symmetry=True,
                          n_time=1024, n_sidebands=15, subtract_residual_mean=True,
                          powder_scheme='rectangular', n_zcw=None,
                          fit_regions=None, normalise=True,
                          baseline_regions=None, baseline_sigma=2.5, baseline_max_iter=10,
                          bounds=None, fixed=None,
                          positive_intensity=True, allow_negative_intensity=False,
                          freeze_shape=True, reference="max_amplitude",
                          show_report=False, show_warnings=True):
    """
    Fit a (possibly multi-site) CSA powder lineshape to every extracted
    2D NMR slice (rows and/or columns) of a relaxation series -- the
    CSA analogue of :func:`nmrtools.fitting.NMR2D_fit_slices`.

    See the module docstring for why, by default (``freeze_shape=True``),
    only one slice (``reference``) gets a full nonlinear
    :func:`nmrtools.csa_fitting.fit_csa_spectrum` fit; every other slice
    then fits only amplitude(s) + baseline by linear least squares with
    the shape held fixed at the reference-slice values.

    Only ``mode="rows"`` is generally meaningful for a pseudo-2D
    relaxation series: a row spans F2, a genuine chemical-shift axis; a
    *pseudo-2D* column spans F1, which is a bare experiment/delay
    index, not a frequency axis a CSA lineshape can be fitted to.
    Passing ``mode="columns"`` or ``"both"`` with a pseudo-2D F1
    column selected raises ``ValueError``. (A genuine, non-pseudo 2D
    experiment's columns are a real chemical-shift axis and are fine.)

    Parameters
    ----------
    slices : dict
        Output of :func:`nmrtools.two_d.NMR2D_get_slices`.
    p0 : dict or list of dict
        Initial CSA guess -- one site, or a list of them. See
        :func:`nmrtools.csa_fitting.fit_csa_spectrum` /
        :mod:`nmrtools.csa_fitting`'s module docstring. Used for the
        reference-slice fit only (and for every slice when
        ``freeze_shape=False``).
    mode : {'rows', 'columns', 'both'}, optional
    convention, static, larmor_Hz, mas_Hz, rotor_angle, n_theta, n_phi,
    use_octant_symmetry, n_time, n_sidebands, subtract_residual_mean,
    powder_scheme, n_zcw, fit_regions, baseline_regions, baseline_sigma,
    baseline_max_iter, bounds, fixed :
        See :func:`nmrtools.csa_fitting.fit_csa_spectrum` -- used for
        the reference-slice fit (and, when ``freeze_shape=False``, for
        every slice).
    normalise : bool, optional
        Normalise each trace to its own max before fitting (see
        ``use_normalised`` in :func:`fit_csa_spectrum`).
    positive_intensity, allow_negative_intensity : bool, optional
        Applied to every site's amplitude, on both the reference
        nonlinear fit and (via bounded ``lsq_linear``) the frozen-shape
        linear fits.
    freeze_shape : bool, optional
        See module docstring. Default True.
    reference : {'max_amplitude', 'first'} or int, optional
        Which slice gets the full nonlinear fit when
        ``freeze_shape=True``: the slice with the largest peak absolute
        intensity (default -- typically the best-S/N slice), the first
        slice in ``slices``, or an explicit 0-based index (into the
        row or column group, sorted by slice position as
        :func:`nmrtools.two_d.NMR2D_get_slices` returns them). Ignored
        when ``freeze_shape=False``.
    show_report : bool, optional
        Print the reference slice's full CSA fit report (see
        :func:`nmrtools.csa_fitting.print_csa_fit_report`). Only used
        when ``freeze_shape=True`` (there is exactly one full nonlinear
        fit to report); when ``freeze_shape=False``, no automatic
        per-slice report is printed (to avoid flooding output for a
        long series) -- call
        :func:`nmrtools.csa_fitting.print_csa_fit_report` yourself
        using a given slice's returned ``popt``/``pcov``/``fit_info``
        if you need one.
    show_warnings : bool, optional
        Print bound-hit warnings for the reference (and, if
        ``freeze_shape=False``, every) slice.

    Returns
    -------
    dict with keys ``"rows"`` and ``"columns"``, each a list of
    per-slice result dicts, shaped closely enough to
    :func:`nmrtools.fitting.NMR2D_fit_slices`'s output to be used
    interchangeably with :func:`nmrtools.relaxation.NMR2D_fit_relaxation`:
    ``slice``, ``slice_type``, ``index``, ``shift``, ``x``, ``y``,
    ``popt``, ``pcov``, ``fit_info``, ``labels``, ``integrals``,
    ``integral_errors``, ``integrals_norm``, ``fit_quality``,
    ``is_reference``.
    """

    if mode not in ("rows", "columns", "both"):
        raise ValueError("mode must be 'rows', 'columns', or 'both'")

    selected_rows = list(slices.get("rows", [])) if mode in ("rows", "both") else []
    selected_columns = []
    if mode in ("columns", "both"):
        for s in slices.get("columns", []):
            if s.get("pseudo_2d") and not s.get("x_is_ppm", True):
                raise ValueError(
                    "A pseudo-2D column trace runs along the bare "
                    "experiment/delay index, not a chemical-shift axis "
                    "-- a CSA lineshape cannot be fitted to it. Use "
                    "mode='rows' (fitting each full spectrum at each "
                    "delay) instead."
                )
            selected_columns.append(s)

    if not selected_rows and not selected_columns:
        raise ValueError("No slices available for fitting.")

    fitted = {"rows": [], "columns": []}

    for slice_type, group_traces in (("row", selected_rows), ("column", selected_columns)):

        if not group_traces:
            continue

        ref_i = _select_reference_index(group_traces, reference) if freeze_shape else None

        # ----- reference (full nonlinear) fit -----
        ref_popt = ref_pcov = ref_fit_info = ref_x_sorted = ref_shapes_full = None

        if freeze_shape:
            ref_trace = group_traces[ref_i]
            results_ref = prepare_trace_for_fit(ref_trace, normalise=normalise)

            ref_popt, ref_pcov_free, ref_is_free, ref_scale, ref_fit_info = fit_csa_spectrum(
                results_ref, p0, convention=convention, static=static,
                larmor_Hz=larmor_Hz, mas_Hz=mas_Hz, rotor_angle=rotor_angle,
                n_theta=n_theta, n_phi=n_phi, use_octant_symmetry=use_octant_symmetry,
                n_time=n_time, n_sidebands=n_sidebands,
                subtract_residual_mean=subtract_residual_mean,
                powder_scheme=powder_scheme, n_zcw=n_zcw,
                fit_regions=fit_regions, use_normalised=normalise,
                baseline_regions=baseline_regions, baseline_sigma=baseline_sigma,
                baseline_max_iter=baseline_max_iter, bounds=bounds, fixed=fixed,
                positive_intensity=positive_intensity,
                allow_negative_intensity=allow_negative_intensity,
            )
            ref_pcov = expand_pcov(ref_pcov_free, ref_is_free, len(ref_popt))

            x_full = results_ref["x_ppm"]
            order = np.argsort(x_full)
            ref_x_sorted = x_full[order]

            n_sites = ref_fit_info["n_sites"]
            sim_kwargs = ref_fit_info["sim_kwargs"]
            ref_csa_params = np.asarray(ref_popt[: 5 * n_sites]).reshape(n_sites, 5)
            ref_shapes_full = [
                csa_model_full(ref_x_sorted, d_iso, span_or_Delta, skew_or_eta, fwhm, **sim_kwargs)
                for d_iso, span_or_Delta, skew_or_eta, fwhm, _ in ref_csa_params
            ]

            if show_warnings:
                lower, upper = build_csa_bounds(
                    n_sites, labels=ref_fit_info["labels"], convention=convention, bounds=bounds,
                    positive_intensity=positive_intensity,
                    allow_negative_intensity=allow_negative_intensity,
                )
                lower = np.append(lower, [-np.inf, -np.inf])
                upper = np.append(upper, [np.inf, np.inf])
                for w in check_csa_bound_hits(ref_popt, ref_pcov, ref_is_free, lower, upper,
                                               ref_fit_info["labels"]):
                    print(f"[reference slice] {w}")

            if show_report:
                y_ref_sorted = results_ref["real_spectrum"][order]
                mask = build_region_mask(ref_x_sorted, fit_regions) if fit_regions is not None \
                    else np.ones_like(ref_x_sorted, dtype=bool)
                y_model_ref = evaluate_csa_fit(ref_x_sorted, ref_popt, ref_fit_info)
                print_csa_fit_report(ref_x_sorted[mask], y_ref_sorted[mask], y_model_ref[mask],
                                      ref_popt, ref_pcov, ref_is_free, ref_fit_info["labels"],
                                      convention=convention)

        # ----- per-slice fit -----
        for i, s in enumerate(group_traces):

            results = prepare_trace_for_fit(s, normalise=normalise)
            x_full = results["x_ppm"]
            order = np.argsort(x_full)
            x_sorted = x_full[order]
            y_sorted = results["real_spectrum"][order]

            is_ref = freeze_shape and (i == ref_i)

            if is_ref:
                popt, pcov, fit_info = ref_popt, ref_pcov, ref_fit_info

            elif freeze_shape:
                # Fast path: reuse ref_shapes_full if this slice shares
                # the reference's x-axis exactly (true by construction
                # for every slice within one mode -- see the module
                # docstring); otherwise (defensively) re-evaluate the
                # fixed-shape functions on this slice's own axis. Either
                # way, no powder re-simulation/optimisation happens here.
                if x_sorted.shape == ref_x_sorted.shape and np.allclose(x_sorted, ref_x_sorted):
                    shapes_full = ref_shapes_full
                else:
                    sim_kwargs = ref_fit_info["sim_kwargs"]
                    shapes_full = [
                        csa_model_full(x_sorted, d_iso, span_or_Delta, skew_or_eta, fwhm, **sim_kwargs)
                        for d_iso, span_or_Delta, skew_or_eta, fwhm, _ in
                        np.asarray(ref_popt[: 5 * ref_fit_info["n_sites"]]).reshape(-1, 5)
                    ]

                mask = build_region_mask(x_sorted, fit_regions) if fit_regions is not None \
                    else np.ones_like(x_sorted, dtype=bool)
                x_masked = x_sorted[mask]
                y_masked = y_sorted[mask]
                shapes_masked = [shape[mask] for shape in shapes_full]

                coeffs, pcov_lin, _ = _fit_frozen_shape_linear(
                    x_masked, y_masked, shapes_masked, ref_fit_info["x_ref"],
                    positive_intensity=positive_intensity,
                    allow_negative_intensity=allow_negative_intensity,
                )

                n_sites = ref_fit_info["n_sites"]
                popt = ref_popt.copy()
                popt[np.arange(n_sites) * 5 + 4] = coeffs[:n_sites]      # amplitudes
                popt[5 * n_sites: 5 * n_sites + 2] = coeffs[n_sites:]    # b0, b1

                # Full-size pcov with only the (amplitude, b0, b1) block
                # populated from the linear fit -- shape-parameter rows/
                # columns are exactly zero, correctly reflecting that
                # they were held fixed (not "unknown"), matching
                # nmrtools.fitting.expand_pcov's convention for fixed
                # parameters elsewhere in this package.
                pcov = np.zeros_like(ref_pcov)
                lin_idx = list(np.arange(n_sites) * 5 + 4) + [5 * n_sites, 5 * n_sites + 1]
                for a, ia in enumerate(lin_idx):
                    for b, ib in enumerate(lin_idx):
                        pcov[ia, ib] = pcov_lin[a, b]

                fit_info = ref_fit_info

            else:
                popt, pcov_free, is_free, scale, fit_info = fit_csa_spectrum(
                    results, p0, convention=convention, static=static,
                    larmor_Hz=larmor_Hz, mas_Hz=mas_Hz, rotor_angle=rotor_angle,
                    n_theta=n_theta, n_phi=n_phi, use_octant_symmetry=use_octant_symmetry,
                    n_time=n_time, n_sidebands=n_sidebands,
                    subtract_residual_mean=subtract_residual_mean,
                    powder_scheme=powder_scheme, n_zcw=n_zcw,
                    fit_regions=fit_regions, use_normalised=normalise,
                    baseline_regions=baseline_regions, baseline_sigma=baseline_sigma,
                    baseline_max_iter=baseline_max_iter, bounds=bounds, fixed=fixed,
                    positive_intensity=positive_intensity,
                    allow_negative_intensity=allow_negative_intensity,
                )
                pcov = expand_pcov(pcov_free, is_free, len(popt))

                if show_warnings:
                    n_sites = fit_info["n_sites"]
                    lower, upper = build_csa_bounds(
                        n_sites, labels=fit_info["labels"], convention=convention, bounds=bounds,
                        positive_intensity=positive_intensity,
                        allow_negative_intensity=allow_negative_intensity,
                    )
                    lower = np.append(lower, [-np.inf, -np.inf])
                    upper = np.append(upper, [np.inf, np.inf])
                    for w in check_csa_bound_hits(popt, pcov, is_free, lower, upper, fit_info["labels"]):
                        print(f"[slice {s['index']}] {w}")

            integrals, integrals_norm = compute_csa_integrals(x_sorted, popt, fit_info)
            errors = compute_csa_integral_errors(popt, pcov, fit_info, integrals)

            mask_full = build_region_mask(x_sorted, fit_regions) if fit_regions is not None \
                else np.ones_like(x_sorted, dtype=bool)
            y_model = evaluate_csa_fit(x_sorted, popt, fit_info)
            metrics = fit_quality(y_sorted[mask_full], y_model[mask_full], len(popt))

            fit_result = {
                "slice": s, "slice_type": slice_type,
                "index": s["index"], "shift": s["shift"],
                "x": x_sorted, "y": y_sorted,
                "popt": popt, "pcov": pcov, "fit_info": fit_info,
                "labels": fit_info["labels"],
                "integrals": integrals, "integral_errors": errors, "integrals_norm": integrals_norm,
                "fit_quality": metrics, "is_reference": is_ref,
            }

            fitted["rows" if slice_type == "row" else "columns"].append(fit_result)

    return fitted


# ============================================================
# End-to-end CSA relaxation workflow
# ============================================================

def NMR2D_csa_relaxation_workflow(datapath, p0, procno=1,
                                   rows=None, row_shifts=None, row_width=1,
                                   convention="Haeberlen", static=True,
                                   larmor_Hz=None, mas_Hz=None, rotor_angle="magic",
                                   n_theta=60, n_phi=120, use_octant_symmetry=True,
                                   n_time=1024, n_sidebands=15, subtract_residual_mean=True,
                                   powder_scheme='rectangular', n_zcw=None,
                                   fit_regions=None, baseline_regions=None, baseline_sigma=2.5,
                                   baseline_max_iter=10, normalise=True,
                                   bounds=None, fixed=None,
                                   positive_intensity=True, allow_negative_intensity=False,
                                   freeze_shape=True, reference="max_amplitude",
                                   model="inversion_recovery", vdlist=None, relaxation_fixed=None,
                                   use_errors=True, normalised_integrals=False,
                                   show_series=None,
                                   show_slices=False, show_fit=True, annotate=True,
                                   show_report=True, show_warnings=True,
                                   colour_spectra="coolwarm", offset_scale=1.05, xlims=None,
                                   logplot=False,
                                   colours=None, fit_colours=None, fit_shade=0.45,
                                   annotate_position='end', annotate_offset=(5, 0),
                                   annotate_ha='left', annotate_va='center',
                                   font='Arial', fontsize=12, pl_width=3.375, pl_height=3.375,
                                   tick_dir='in', tick_length=4.0, tick_width=1.0, tick_pad=4.0,
                                   tick_number=None, show_top_spine=False, show_right_spine=False,
                                   save_fig=False, fig_name='nmr_csa_relaxation', save_format='png'):
    """
    High-level, end-to-end CSA-lineshape-based T1/T2/T1rho workflow for
    a Bruker pseudo-2D dataset: import once, extract every requested
    row, fit a (possibly multi-site) CSA lineshape to each (with the
    shape frozen after a single reference-slice fit by default -- see
    the module docstring and :func:`NMR2D_fit_csa_slices`), extract
    each site's integral as a function of delay, fit a relaxation
    model to each, and optionally report and plot the result.

    Only row extraction is supported here (``rows``/``row_shifts`` --
    a T1/T2/T1rho series lives along F1 with each row a full F2
    spectrum; see :func:`NMR2D_fit_csa_slices` for why a pseudo-2D
    column can't be CSA-fitted). Call :func:`NMR2D_fit_csa_slices`
    directly (with ``mode='columns'``) for a genuine, non-pseudo 2D
    dataset where the relaxation dimension runs along F2 instead.

    Workflow: ``NMR2dimport -> NMR2D_get_slices -> NMR2D_fit_csa_slices
    -> nmrtools.relaxation.NMR2D_fit_relaxation ->
    nmrtools.relaxation.NMR2D_plot_relaxation``.

    Parameters
    ----------
    datapath : str
    p0 : dict or list of dict
        See :func:`NMR2D_fit_csa_slices` / :mod:`nmrtools.csa_fitting`.
    procno : int, optional
    rows, row_shifts, row_width : optional
        Slice selection -- see :func:`nmrtools.two_d.NMR2D_get_slices`.
    convention, static, larmor_Hz, mas_Hz, rotor_angle, n_theta, n_phi,
    use_octant_symmetry, n_time, n_sidebands, subtract_residual_mean,
    powder_scheme, n_zcw, fit_regions, baseline_regions, baseline_sigma,
    baseline_max_iter, normalise, bounds, fixed, positive_intensity,
    allow_negative_intensity, freeze_shape, reference :
        See :func:`NMR2D_fit_csa_slices`.
    model, vdlist, use_errors, normalised_integrals :
        See :func:`nmrtools.relaxation.NMR2D_fit_relaxation`.
    relaxation_fixed : dict, optional
        Forwarded as ``fixed`` to
        :func:`nmrtools.relaxation.NMR2D_fit_relaxation` (renamed here
        to avoid clashing with the CSA-fit ``fixed`` above, which fixes
        CSA/lineshape parameters, not relaxation-model parameters).
    show_series, show_slices, show_fit, annotate, show_report,
    show_warnings, colour_spectra, offset_scale, xlims, logplot,
    colours, fit_colours, fit_shade, annotate_position, annotate_offset,
    annotate_ha, annotate_va, font, fontsize, pl_width, pl_height,
    tick_dir, tick_length, tick_width, tick_pad, tick_number,
    show_top_spine, show_right_spine :
        See :func:`nmrtools.relaxation.NMR2D_relaxation_workflow` --
        identical meaning here. Note ``show_report`` also controls
        whether :func:`NMR2D_fit_csa_slices` prints the reference
        slice's CSA fit report.
    save_fig, fig_name, save_format : optional

    Returns
    -------
    dict
        ``"dataset"``, ``"slices"``, ``"series"`` (output of
        :func:`NMR2D_fit_csa_slices`, ``["rows"]``),
        ``"relaxation_fits"`` (output of
        :func:`nmrtools.relaxation.NMR2D_fit_relaxation`, filtered to
        ``show_series`` if given), ``"figure"``, ``"axes"``,
        ``"slices_figure"``, ``"slices_axes"``.
    """

    if rows is None and row_shifts is None:
        raise ValueError("No rows were requested (rows or row_shifts).")

    dataset = NMR2dimport(datapath, procno=procno)

    slices = NMR2D_get_slices(dataset, rows=rows, row_shifts=row_shifts,
                               row_width=row_width, normalise=False)

    csa_fits = NMR2D_fit_csa_slices(
        slices, p0, mode="rows", convention=convention, static=static,
        larmor_Hz=larmor_Hz, mas_Hz=mas_Hz, rotor_angle=rotor_angle,
        n_theta=n_theta, n_phi=n_phi, use_octant_symmetry=use_octant_symmetry,
        n_time=n_time, n_sidebands=n_sidebands, subtract_residual_mean=subtract_residual_mean,
        powder_scheme=powder_scheme, n_zcw=n_zcw,
        fit_regions=fit_regions, normalise=normalise,
        baseline_regions=baseline_regions, baseline_sigma=baseline_sigma,
        baseline_max_iter=baseline_max_iter, bounds=bounds, fixed=fixed,
        positive_intensity=positive_intensity, allow_negative_intensity=allow_negative_intensity,
        freeze_shape=freeze_shape, reference=reference,
        show_report=show_report, show_warnings=show_warnings,
    )["rows"]

    labels = csa_fits[0]["labels"] if csa_fits else None

    relaxation_fits = NMR2D_fit_relaxation(
        csa_fits, model=model, vdlist=vdlist, peak_labels=labels,
        normalised=normalised_integrals, use_errors=use_errors,
        fixed=relaxation_fixed, show_report=False,
    )

    relaxation_fits_shown = _select_series(relaxation_fits, show_series)

    if show_report:
        for fit in relaxation_fits_shown:
            print(f"\n--- {fit['peak_label']} ---")
            print_relaxation_report(fit)

    fig_slices = ax_slices = None
    if show_slices:
        fig_slices, ax_slices = NMR2D_plot_slices(
            slices, mode="rows", colour_spectra=colour_spectra,
            offset_scale=offset_scale, xlims=xlims, font=font, fontsize=fontsize,
        )
        if save_fig:
            save_figure(fig_slices, fig_name=f"{fig_name}_slices", save_format=save_format)

    fig = ax = None
    if show_fit:
        fig, ax = NMR2D_plot_relaxation(
            relaxation_fits_shown, annotate=annotate, font=font, fontsize=fontsize, logplot=logplot,
            colours=colours, fit_colours=fit_colours, fit_shade=fit_shade,
            annotate_position=annotate_position, annotate_offset=annotate_offset,
            annotate_ha=annotate_ha, annotate_va=annotate_va,
            pl_width=pl_width, pl_height=pl_height, tick_dir=tick_dir, tick_length=tick_length,
            tick_width=tick_width, tick_pad=tick_pad, tick_number=tick_number,
            show_top_spine=show_top_spine, show_right_spine=show_right_spine,
        )
        if save_fig:
            save_figure(fig, fig_name=fig_name, save_format=save_format)

    return {
        "dataset": dataset, "slices": slices, "series": csa_fits,
        "relaxation_fits": relaxation_fits_shown,
        "figure": fig, "axes": ax,
        "slices_figure": fig_slices, "slices_axes": ax_slices,
    }
