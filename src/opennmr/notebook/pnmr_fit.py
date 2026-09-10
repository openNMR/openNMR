"""
pnmr_fit.py
===========

Fit an experimental 1D spectrum against the environments predicted for a
proposed CIF, using nmrtools.pnmr_simulator for the shift prediction and
nmrtools.fitting for the actual nonlinear least-squares fit.

This module does NOT reimplement peak fitting: every peak is fitted with
fitting.fit_spectrum, using the standard Gaussian/Lorentzian/pseudo-Voigt
models already defined there. What it adds is the CIF-specific bookkeeping
around that fit:

    1. predict_environments   - enumerate every (site, disorder-config) shift
                                 predicted from the CIF, un-broadened, each
                                 carrying a human-readable composition label
    2. group_environments     - merge predicted shifts that fall within a
                                 tolerance of one another into a single peak,
                                 while remembering *which* environments went
                                 into each group and their relative weight
                                 within it
    3. select_groups          - drop groups outside the requested fitting
                                 region, and drop groups that fall below a
                                 signal-to-noise threshold in the *experimental*
                                 spectrum (rather than assuming every
                                 CIF-predicted site is observable)
    4. groups_to_peak_defs    - turn surviving groups into fitting.py
                                 ``peak_defs``, with shift FIXED at the
                                 CIF-predicted value (only width/intensity/eta
                                 are floated, per the intended use of this
                                 tool -- see float_shift_window if you want
                                 to relax that)
    5. split_integrals_by_environment - after fitting, divide each fitted
                                 peak's integral back out across its member
                                 environments, in proportion to their CIF
                                 probabilities
    6. fit_predicted_spectrum - end-to-end driver wiring 1-5 together and
                                 handing off to fitting.fit_spectrum

Worked caveats (read before trusting the output)
-------------------------------------------------
* Grouping is inherently a modelling choice, not a fact about the physics.
  The default "sequential/gap" method can chain: if A-B and B-C are each
  within tolerance but A-C is not, all three still end up in one group,
  because grouping is decided pairwise along the sorted shift axis. This
  is the standard behaviour of single-linkage clustering, and
  ``method='hierarchical'`` exposed by group_environments is exactly
  single-linkage clustering (scipy) -- it will chain in the same way. If
  that's not what you want, cap group *span* (see ``max_group_span``) or
  post-hoc inspect ``members`` on any group you don't trust.
* The S/N based rejection in select_groups compares the local maximum of
  the experimental spectrum near a predicted shift to a global noise
  estimate. It cannot distinguish "there is genuinely no peak here" from
  "there is a peak here, but it overlaps a much larger one nearby and is
  swamped" -- an overlapping small peak will show a large local max (the
  neighbour's tail) and pass the S/N test even though its own signature is
  not resolved. Treat the automatic keep/drop list as a first pass, not a
  final answer -- inspect ``dropped`` before accepting it.
* Splitting a fitted peak's integral across its member environments by
  their *CIF probability* ratio assumes every environment in a group has
  the same lineshape and width -- i.e. that the reason they overlap is
  genuinely because their true shifts are close, not because the fit
  can't resolve them. This is likely fine when the grouping tolerance is
  small relative to the fitted FWHM, and increasingly shaky as tolerance
  approaches FWHM.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .pnmr_simulator import (
    read_cif_with_occupancies, get_neighbors_by_shell,
    enumerate_local_environments, build_shift_table,
)
from .fitting import (
    fit_spectrum, evaluate_fit, compute_integrals, compute_integral_errors,
    check_bound_hits, build_bounds, build_region_mask, expand_pcov,
    enforce_positive_intensity, estimate_noise_level, print_fit_report,
    NMR1dplot_with_fit,
)
from .one_d import NMR1dimport


# ===========================================================================
# 1. Predict every (site, config) shift from the CIF, unbroadened, labelled
# ===========================================================================
def predict_environments(cif_filename, atom_of_interest, shell_bins, shift_dict,
                          ignore_species=None, vacancy_tol=5,
                          warn_missing_shifts=True, verbose=False) -> pd.DataFrame:
    """
    Enumerate every disordered local environment for `atom_of_interest`
    sites in `cif_filename`, and the total shift/probability predicted for
    each -- the same underlying calculation as
    pnmr_simulator.generate_spectrum_dict, but returned as a flat,
    unbroadened table (no spectrum is built here), with an added
    human-readable `label` column describing the shell composition, e.g.
    "NN:5Fe+1Mn NNN:6Fe".

    Returns
    -------
    DataFrame with columns Site, Config, <shell>_<species> counts,
    Total_shift, p (probability, summed to 1 over the whole table), label.
    """
    structure, site_data = read_cif_with_occupancies(
        cif_filename, atom_of_interest=atom_of_interest,
        vacancy_tol=vacancy_tol, insert_vacancies=True, verbose=verbose,
    )
    shell_names = [name for name, _, _ in shell_bins]
    neighbors_info = get_neighbors_by_shell(
        structure, site_data, shell_bins, ignore_species=ignore_species,
    )
    local_envs = enumerate_local_environments(
        site_data, neighbors_info, atom_of_interest, shell_names,
    )
    shift_table = build_shift_table(
        local_envs, shift_dict, warn_missing=warn_missing_shifts, print_table=False,
    )
    if shift_table.empty:
        raise ValueError(
            f"No sites containing '{atom_of_interest}' were found (or none "
            f"had neighbors assigned to a shell) -- check atom_of_interest, "
            f"shell_bins and ignore_species."
        )

    non_count_cols = {"Site", "Config", "Total_shift", "p"}

    def _label(row):
        parts = []
        for shell in shell_names:
            comp = []
            prefix = f"{shell}_"
            for col in row.index:
                if col in non_count_cols or not col.startswith(prefix):
                    continue
                n = row[col]
                if n:
                    comp.append(f"{int(n)}{col[len(prefix):]}")
            if comp:
                parts.append(f"{shell}:" + "+".join(comp))
        return " ".join(parts) if parts else "bare"

    shift_table = shift_table.copy()
    shift_table["label"] = shift_table.apply(_label, axis=1)
    return shift_table


# ===========================================================================
# 2. Group shifts that fall within `tolerance` of one another
# ===========================================================================
def default_tolerance(shift_dict: Dict[str, Dict[str, float]]) -> float:
    """abs() of the smallest non-zero pathway contribution in shift_dict."""
    nonzero = [abs(v) for shell in shift_dict.values() for v in shell.values() if v != 0]
    if not nonzero:
        raise ValueError(
            "shift_dict contains no non-zero pathway contributions, so a "
            "default tolerance can't be inferred -- pass `tolerance` explicitly."
        )
    return min(nonzero)


def group_environments(shift_table: pd.DataFrame, tolerance: Optional[float] = None,
                        shift_dict: Optional[Dict] = None, method: str = "sequential",
                        max_group_span: Optional[float] = None) -> Tuple[pd.DataFrame, float]:
    """
    Merge rows of `shift_table` (as returned by predict_environments) whose
    Total_shift values are within `tolerance` ppm of one another into a
    single group/peak, remembering the member environments and their
    relative weight *within* the group.

    tolerance : float, optional
        Defaults to `default_tolerance(shift_dict)` -- the smallest
        non-zero pathway contribution supplied, per the brief ("group
        anything closer together than the smallest bond pathway can
        resolve").
    method : {'sequential', 'hierarchical'}
        'sequential' (default) walks the sorted shifts and starts a new
        group whenever the gap to the previous point exceeds tolerance.
        'hierarchical' is scipy single-linkage clustering with a distance
        threshold of `tolerance`. Both are single-linkage and can chain
        (see module docstring) -- they will usually agree, but
        'hierarchical' is a genuine second opinion since it doesn't rely
        on the data already being processed in shift order.
    max_group_span : float, optional
        If set, reject (raise) any group whose max-min shift exceeds this,
        as a guard against chaining silently producing an implausibly wide
        "single peak". Off by default.

    Returns
    -------
    (groups_df, tolerance_used)

    groups_df columns: group_shift (probability-weighted mean of members),
    group_p (summed probability), n_members, members (list of dicts with
    label/shift/p/fraction_in_group, the last being each member's
    probability as a fraction of the group total -- this is what later
    gets used to split a fitted peak's integral back out per environment).
    """
    if tolerance is None:
        if shift_dict is None:
            raise ValueError("Supply either `tolerance` or `shift_dict`.")
        tolerance = default_tolerance(shift_dict)

    df = shift_table.sort_values("Total_shift").reset_index(drop=True)

    if method == "sequential":
        group_id = [0]
        for i in range(1, len(df)):
            gap = df.loc[i, "Total_shift"] - df.loc[i - 1, "Total_shift"]
            group_id.append(group_id[-1] + (1 if gap > tolerance else 0))
        df["_group"] = group_id
    elif method == "hierarchical":
        from scipy.cluster.hierarchy import linkage, fcluster
        shifts = df["Total_shift"].to_numpy().reshape(-1, 1)
        Z = linkage(shifts, method="single")
        df["_group"] = fcluster(Z, t=tolerance, criterion="distance")
    else:
        raise ValueError("method must be 'sequential' or 'hierarchical'.")

    rows = []
    for _, sub in df.groupby("_group"):
        span = sub["Total_shift"].max() - sub["Total_shift"].min()
        if max_group_span is not None and span > max_group_span:
            raise ValueError(
                f"Group spanning {span:.4f} ppm exceeds max_group_span="
                f"{max_group_span} (members: {sub['label'].tolist()}). "
                f"Lower `tolerance`, or try method='hierarchical'."
            )
        total_p = sub["p"].sum()
        weighted_shift = (
            (sub["Total_shift"] * sub["p"]).sum() / total_p if total_p > 0
            else sub["Total_shift"].mean()
        )
        members = [
            {
                "label": r["label"], "shift": float(r["Total_shift"]),
                "p": float(r["p"]),
                "fraction_in_group": float(r["p"] / total_p) if total_p > 0 else np.nan,
            }
            for _, r in sub.iterrows()
        ]
        rows.append({
            "group_shift": float(weighted_shift), "group_p": float(total_p),
            "n_members": len(sub), "span": float(span), "members": members,
        })

    groups_df = pd.DataFrame(rows).sort_values("group_shift").reset_index(drop=True)
    return groups_df, tolerance


# ===========================================================================
# 3. Restrict to region of interest + reject groups below S/N in the DATA
# ===========================================================================
def select_groups(groups_df: pd.DataFrame, results: Dict, fit_regions=None,
                   x_unit: str = "ppm", snr_threshold: float = 3.0,
                   noise_region=None, window: Optional[float] = None,
                   tolerance: Optional[float] = None,
                   use_normalised: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    """
    Keep only groups that (a) fall inside `fit_regions` (default: the whole
    spectrum) and (b) show a local signal-to-noise ratio >= snr_threshold in
    the experimental spectrum `results` (an NMR1dimport/spectrum-dict).

    S/N is computed as (max |intensity| within +/- window of group_shift) /
    (noise std from estimate_noise_level). `window` defaults to
    `tolerance` (the same value used for grouping) if given, else to 1% of
    the spectrum's x-range.

    A group inside fit_regions but below threshold is dropped with reason
    'below S/N threshold'; a group outside fit_regions entirely is dropped
    with reason 'outside fit_regions', regardless of S/N.

    Returns (kept, dropped, noise_std). `kept` carries an added `A_init`
    column (the local max used for the S/N test) that groups_to_peak_defs
    uses as a starting amplitude guess, and an `snr` column throughout.
    """
    x = np.asarray(results["x_ppm"] if x_unit == "ppm" else results["x_Hz"], dtype=float)
    y = np.asarray(
        results["normalised_real_spectrum"] if use_normalised else results["real_spectrum"],
        dtype=float,
    )
    noise = estimate_noise_level(x, y, noise_region=noise_region)

    if window is None:
        window = tolerance if tolerance is not None else 0.01 * abs(x.max() - x.min())

    kept_rows, dropped_rows = [], []
    for _, row in groups_df.iterrows():
        shift = row["group_shift"]
        in_region = (
            fit_regions is None
            or any(min(lo, hi) <= shift <= max(lo, hi) for lo, hi in fit_regions)
        )
        r = row.to_dict()
        if not in_region:
            r["reason"] = "outside fit_regions"
            dropped_rows.append(r)
            continue

        local_mask = (x >= shift - window) & (x <= shift + window)
        local_max = float(np.max(np.abs(y[local_mask]))) if np.any(local_mask) else 0.0
        snr = local_max / noise if noise > 0 else np.inf
        r["snr"] = snr
        r["A_init"] = local_max
        if snr >= snr_threshold:
            kept_rows.append(r)
        else:
            r["reason"] = "below S/N threshold"
            dropped_rows.append(r)

    kept = pd.DataFrame(kept_rows).reset_index(drop=True)
    dropped = pd.DataFrame(dropped_rows).reset_index(drop=True)
    return kept, dropped, noise


# ===========================================================================
# 4. Surviving groups -> fitting.py peak_defs
# ===========================================================================
def groups_to_peak_defs(kept: pd.DataFrame, peak_type: str = "gaussian",
                         fwhm_init: float = 5.0, fwhm_bounds=(0.1, 100.0),
                         eta_init: float = 0.5, eta_bounds=(0.0, 1.0),
                         amplitude_bounds=(0.0, np.inf),
                         float_shift_window: float = 0.0) -> List[dict]:
    """
    Build a fitting.py-style peak_defs list, one peak per kept group, in
    the SAME row order as `kept` (required -- fit_spectrum/evaluate_fit
    return peaks/components in peak_defs order, and
    split_integrals_by_environment zips its integrals back against `kept`
    in that same order).

    Peak centre (`shift`) is FIXED at the CIF-predicted group_shift by
    default -- only fwhm, amplitude (and eta for pseudo_voigt) are fitted,
    per the brief. Set float_shift_window > 0 to instead allow `shift` to
    float within +/- that many ppm of the prediction (e.g. to absorb small
    referencing errors); shift then becomes a free parameter with those
    bounds.
    """
    if peak_type not in ("gaussian", "lorentzian", "pseudo_voigt"):
        raise ValueError("peak_type must be 'gaussian', 'lorentzian' or 'pseudo_voigt'.")

    peak_defs = []
    for _, row in kept.iterrows():
        shift = row["group_shift"]
        bounds = {"fwhm": fwhm_bounds, "amplitude": amplitude_bounds}
        fixed = {"shift": True}
        if float_shift_window > 0:
            bounds["shift"] = (shift - float_shift_window, shift + float_shift_window)
            fixed = {"shift": False}

        peak = {"type": peak_type, "shift": shift, "fwhm": fwhm_init, "fixed": fixed,
                "bounds": bounds}

        a_init = row.get("A_init", None)
        if a_init is not None and not (isinstance(a_init, float) and np.isnan(a_init)):
            peak["A_init"] = float(a_init)

        if peak_type == "pseudo_voigt":
            peak["eta"] = eta_init
            bounds["eta"] = eta_bounds

        peak_defs.append(peak)

    return peak_defs


# ===========================================================================
# 5. Fitted-peak integrals -> per-environment integrals
# ===========================================================================
def split_integrals_by_environment(kept: pd.DataFrame, integrals: Sequence[float],
                                    integrals_norm: Sequence[float],
                                    integral_errors: Optional[Sequence[float]] = None
                                    ) -> pd.DataFrame:
    """
    Divide each fitted peak's integral (raw and % of total fitted
    intensity) back out across the environments that were merged into it,
    in proportion to their `fraction_in_group` (i.e. their relative
    CIF-predicted probability -- see group_environments and the module
    docstring's caveat on this assumption).

    `kept`, `integrals`, `integrals_norm` (and `integral_errors`, if given)
    must all be in the SAME order as the peak_defs used for fitting (see
    groups_to_peak_defs).

    Returns a long-format DataFrame, one row per environment: peak_shift,
    environment_label, environment_shift, fraction_within_peak,
    integral_abs, integral_pct_of_total[, integral_abs_err].
    """
    if integral_errors is None:
        integral_errors = [np.nan] * len(integrals)

    rows = []
    for (_, grow), I, Ipct, Ierr in zip(kept.iterrows(), integrals, integrals_norm, integral_errors):
        for m in grow["members"]:
            rows.append({
                "peak_shift": grow["group_shift"],
                "environment_label": m["label"],
                "environment_shift": m["shift"],
                "fraction_within_peak": m["fraction_in_group"],
                "integral_abs": I * m["fraction_in_group"],
                "integral_pct_of_total": Ipct * m["fraction_in_group"],
                "integral_abs_err": Ierr * m["fraction_in_group"] if np.isfinite(Ierr) else np.nan,
            })
    return pd.DataFrame(rows)


def print_environment_report(env_df: pd.DataFrame, dropped: Optional[pd.DataFrame] = None):
    """Human-readable print of split_integrals_by_environment's output,
    plus (optionally) which predicted groups were excluded and why."""
    print("\n" + "=" * 70)
    print("PREDICTED ENVIRONMENTS CONTRIBUTING TO THE FIT")
    print("=" * 70)
    for peak_shift, sub in env_df.groupby("peak_shift"):
        print(f"\nPeak at {peak_shift:.3f} ppm "
              f"({sub['integral_pct_of_total'].sum():.2f}% of total, "
              f"{len(sub)} environment(s)):")
        for _, r in sub.sort_values("fraction_within_peak", ascending=False).iterrows():
            err = f" +/- {r['integral_abs_err']:.4g}" if np.isfinite(r["integral_abs_err"]) else ""
            print(f"    {r['environment_label']:<40s} "
                  f"{100 * r['fraction_within_peak']:5.1f}% of peak  ->  "
                  f"{r['integral_pct_of_total']:6.2f}% of total "
                  f"(abs {r['integral_abs']:.4g}{err})")

    if dropped is not None and len(dropped):
        print("\n" + "-" * 70)
        print(f"{len(dropped)} predicted group(s) excluded before fitting:")
        for _, r in dropped.iterrows():
            snr_txt = f", S/N={r['snr']:.2f}" if "snr" in r and pd.notna(r.get("snr")) else ""
            print(f"    {r['group_shift']:8.3f} ppm  ({r['n_members']} env., "
                  f"p={r['group_p']:.4g}{snr_txt})  -- {r['reason']}")
    print("=" * 70 + "\n")


# ===========================================================================
# 6. End-to-end driver
# ===========================================================================
def fit_predicted_spectrum(
    spectrum, cif_filename, atom_of_interest, shell_bins, shift_dict,
    ignore_species=None, fit_regions=None, tolerance=None, group_method="sequential",
    max_group_span=None, snr_threshold=3.0, noise_region=None, snr_window=None,
    peak_type="gaussian", fwhm_init=5.0, fwhm_bounds=(0.1, 100.0),
    eta_init=0.5, eta_bounds=(0.0, 1.0), amplitude_bounds=(0.0, np.inf),
    float_shift_window=0.0, x_unit="ppm", procno=1, use_normalised=True,
    positive_intensity=True, allow_negative_intensity=False,
    baseline_regions=None, baseline_sigma=2.5, baseline_max_iter=10,
    vacancy_tol=5, warn_missing_shifts=True, verbose=False,
    show_plot=True, show_components=True, show_report=True,
    xlims="default", xlabel=None,
):
    """
    End-to-end: predict environments from a CIF, group/filter them against
    an experimental spectrum, fit widths/intensities (+ eta) with their
    centres fixed at the CIF-predicted shifts, and report which
    environments contribute to each observed peak.

    Parameters
    ----------
    spectrum : str or dict
        Either a Bruker datapath (passed to NMR1dimport(spectrum,
        procno=procno)), or an already-loaded spectrum dict of the same
        shape (x_ppm/x_Hz/real_spectrum/... -- e.g. from NMR1dimport
        yourself, or from pnmr_simulator.generate_spectrum_dict if you
        want to fit a *simulated* spectrum as a sanity check).
    cif_filename, atom_of_interest, shell_bins, shift_dict, ignore_species,
    vacancy_tol, warn_missing_shifts, verbose :
        See predict_environments / pnmr_simulator.generate_spectrum_dict.
    fit_regions : list of (xmin, xmax), optional
        Region(s) of interest to fit. Default (None) is the whole
        spectrum -- matches the brief's "default is everything".
    tolerance, group_method, max_group_span : see group_environments.
    snr_threshold, noise_region, snr_window : see select_groups
        (`snr_window` is passed through as `window`).
    peak_type, fwhm_init, fwhm_bounds, eta_init, eta_bounds,
    amplitude_bounds, float_shift_window : see groups_to_peak_defs.
    x_unit, procno, use_normalised, positive_intensity,
    allow_negative_intensity, baseline_regions, baseline_sigma,
    baseline_max_iter : see fitting.fit_spectrum.
    show_plot, show_components, show_report, xlims, xlabel :
        Forwarded to fitting.NMR1dplot_with_fit / print_fit_report.

    Returns
    -------
    dict with keys:
        results        - the experimental spectrum dict that was fitted
        predicted       - raw predict_environments() table
        groups          - full group_environments() table (pre-filter)
        kept, dropped   - select_groups() output
        peak_defs       - what was actually fitted
        popt, pcov      - fitted parameters (full, expanded) + covariance
        is_free         - boolean mask of which parameters were floated
        bound_warnings  - from fitting.check_bound_hits
        integrals, integrals_norm, integral_errors - per PEAK (i.e. per
                          group, one entry per fitted line)
        environment_report - split_integrals_by_environment() output,
                          one row per individual CIF environment
        figure, axes    - if show_plot
    """
    # -- experimental spectrum --------------------------------------------
    if isinstance(spectrum, str):
        results = NMR1dimport(spectrum, procno=procno)
    else:
        results = spectrum

    # -- 1. predict, 2. group, 3. select ------------------------------------
    predicted = predict_environments(
        cif_filename, atom_of_interest, shell_bins, shift_dict,
        ignore_species=ignore_species, vacancy_tol=vacancy_tol,
        warn_missing_shifts=warn_missing_shifts, verbose=verbose,
    )
    groups, tolerance = group_environments(
        predicted, tolerance=tolerance, shift_dict=shift_dict,
        method=group_method, max_group_span=max_group_span,
    )
    kept, dropped, noise = select_groups(
        groups, results, fit_regions=fit_regions, x_unit=x_unit,
        snr_threshold=snr_threshold, noise_region=noise_region,
        window=snr_window, tolerance=tolerance, use_normalised=False,
    )
    if kept.empty:
        raise ValueError(
            "No predicted environments survived region/S:N filtering -- "
            "nothing to fit. Inspect `dropped` (call select_groups "
            "yourself) or loosen fit_regions/snr_threshold."
        )

    # -- 4. build peak_defs, fit --------------------------------------------
    peak_defs = groups_to_peak_defs(
        kept, peak_type=peak_type, fwhm_init=fwhm_init, fwhm_bounds=fwhm_bounds,
        eta_init=eta_init, eta_bounds=eta_bounds, amplitude_bounds=amplitude_bounds,
        float_shift_window=float_shift_window,
    )

    popt, pcov_free, is_free, scale, x_ref = fit_spectrum(
        results, peak_defs, fit_regions=fit_regions, use_normalised=use_normalised,
        x_unit=x_unit, baseline_regions=baseline_regions, baseline_sigma=baseline_sigma,
        baseline_max_iter=baseline_max_iter, positive_intensity=positive_intensity,
        allow_negative_intensity=allow_negative_intensity,
    )
    pcov = expand_pcov(pcov_free, is_free, len(popt))

    if positive_intensity and not allow_negative_intensity:
        popt = enforce_positive_intensity(popt, peak_defs)

    lower, upper = build_bounds(peak_defs, positive_intensity=positive_intensity,
                                 allow_negative_intensity=allow_negative_intensity)
    bound_warnings = check_bound_hits(popt, pcov, is_free, lower, upper, peak_defs)

    # -- 5. integrals, split by environment ----------------------------------
    x_full = results["x_ppm"] if x_unit == "ppm" else results["x_Hz"]
    mask = build_region_mask(np.asarray(x_full), fit_regions)
    x_fit = np.asarray(x_full)[mask]

    y_model, components = evaluate_fit(x_fit, popt, peak_defs, x_ref=x_ref, return_components=True)
    integrals, integrals_norm = compute_integrals(x_fit, components)
    integral_errors = compute_integral_errors(popt, pcov, peak_defs, integrals)

    environment_report = split_integrals_by_environment(
        kept, integrals, integrals_norm, integral_errors,
    )

    # -- optional plot / reports ---------------------------------------------
    fig = ax = None
    if show_plot:
        fig, ax = NMR1dplot_with_fit(
            results=results, popt=popt, peak_defs=peak_defs, fit_regions=fit_regions,
            x_unit=x_unit, show_components=show_components, use_normalised=use_normalised,
            xlims=xlims, xlabel=xlabel,
        )

    if show_report:
        y_full = results["normalised_real_spectrum"] if use_normalised else results["real_spectrum"]
        y_fit_data = np.asarray(y_full)[mask]
        print_fit_report(x_fit, y_fit_data, popt, pcov, peak_defs, bound_warnings, is_free,
                          components=components)
        print_environment_report(environment_report, dropped)

    return {
        "results": results, "predicted": predicted, "groups": groups,
        "kept": kept, "dropped": dropped, "noise": noise,
        "peak_defs": peak_defs, "popt": popt, "pcov": pcov, "is_free": is_free,
        "bound_warnings": bound_warnings, "integrals": integrals,
        "integrals_norm": integrals_norm, "integral_errors": integral_errors,
        "environment_report": environment_report, "figure": fig, "axes": ax,
    }
