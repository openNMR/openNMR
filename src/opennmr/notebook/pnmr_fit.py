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
                                 handing off to fitting.fit_spectrum, giving
                                 the "maximal fit" (one peak per surviving
                                 CIF-predicted environment group)
    7. merge_low_integral_peaks / merge_groups_to_picks - two ways to
                                 simplify a maximal fit into far fewer peaks:
                                 automatically, by absorbing low-integral
                                 peaks into their nearest neighbour, or
                                 manually, by naming the peak centres you
                                 want and merging everything onto them
    8. refit_merged_groups   - re-fit the simplified peak set from either
                                 of the above, this time with shift FREE to
                                 move (there are far fewer peaks now, so
                                 that's less likely to be degenerate), and
                                 report how far each fitted centre drifted
                                 from its CIF-predicted starting point

Worked caveats (read before trusting the output)
-------------------------------------------------
* Grouping is inherently a modelling choice, not a fact about the physics.
  The default method, ``method='probability'``, builds each group around
  the most probable remaining environment: it takes that environment,
  pulls in every remaining environment within +/- 0.5*tolerance of its
  shift, removes them from the pool, then repeats with whatever is most
  probable among what's left, until every environment has been assigned.
  Each group is therefore anchored to one seed shift and can't exceed
  `tolerance` in span. The two alternative methods, 'sequential' and
  'hierarchical', instead decide grouping pairwise along the sorted shift
  axis (single-linkage clustering) and so CAN chain: if A-B and B-C are
  each within tolerance but A-C is not, all three still end up in one
  group. ``method='hierarchical'`` is scipy single-linkage clustering and
  will chain the same way 'sequential' does. If you use one of those and
  chaining isn't what you want, cap group *span* (see ``max_group_span``)
  or post-hoc inspect ``members`` on any group you don't trust.
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
* merge_low_integral_peaks' cascade (a low peak merges into its nearest
  neighbour, which can then itself become a merge target) is a greedy,
  order-dependent simplification, not a unique answer -- inspect
  ``merge_log`` before trusting it, same as ``members``/``dropped``
  elsewhere. A low peak with nothing else within ``merge_tolerance`` is
  left alone rather than forced onto a distant peak.
* refit_merged_groups reads each fitted peak's centre off the fitted
  component itself (a parabola-refined argmax), not off ``popt``
  directly -- this makes it agnostic to fitting.py's internal parameter
  ordering, but it's still a point estimate on a finite grid and its
  precision is capped by the spectrum's digital resolution.
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
    enforce_positive_intensity, estimate_noise_level, estimate_baseline_robust,
    print_fit_report, NMR1dplot_with_fit,
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
                        shift_dict: Optional[Dict] = None, method: str = "probability",
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
    method : {'probability', 'sequential', 'hierarchical'}
        'probability' (default) is seed-based rather than chained: take
        the single most probable environment remaining, form a group out
        of every remaining environment within +/- 0.5*tolerance of *that*
        seed's shift, remove them from the pool, then repeat with the
        most probable environment still left, until none remain. Because
        each group is defined by its distance to one fixed seed (not to
        its neighbour along the sorted axis), it can't chain the way the
        other two methods can -- a group's full span is capped at
        `tolerance` by construction. The trade-off is that it's now
        order-dependent on `p`: a low-probability environment can end up
        in a different group than it would under sequential/hierarchical
        if it sits between two seeds.
        'sequential' walks the sorted shifts and starts a new group
        whenever the gap to the previous point exceeds tolerance.
        'hierarchical' is scipy single-linkage clustering with a distance
        threshold of `tolerance`. Both 'sequential' and 'hierarchical' are
        single-linkage and can chain (see module docstring) -- they will
        usually agree with each other, and 'hierarchical' is a genuine
        second opinion on 'sequential' since it doesn't rely on the data
        already being processed in shift order.
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

    if method == "probability":
        half_tol = 0.5 * tolerance
        group_id = pd.Series(-1, index=df.index)
        remaining = df.index.to_series()
        next_group = 0
        while len(remaining) > 0:
            # Most probable environment left becomes this group's seed.
            seed_idx = df.loc[remaining, "p"].idxmax()
            seed_shift = df.loc[seed_idx, "Total_shift"]
            in_group = remaining[
                (df.loc[remaining, "Total_shift"] >= seed_shift - half_tol)
                & (df.loc[remaining, "Total_shift"] <= seed_shift + half_tol)
            ]
            group_id.loc[in_group] = next_group
            remaining = remaining.drop(in_group)
            next_group += 1
        df["_group"] = group_id.to_numpy()
    elif method == "sequential":
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
        raise ValueError("method must be 'probability', 'sequential' or 'hierarchical'.")

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
# Shared low-level fit step (used by both fit_predicted_spectrum and
# refit_merged_groups): run fit_spectrum on a peak_defs list, expand and
# bound-check the result, integrate each peak, and split those integrals
# back out per-environment against `kept`. Pulled out so the "maximal fit"
# driver and the "refit a simplified peak set" driver below can't drift
# apart from each other.
# ===========================================================================
def _fit_peak_defs(results: Dict, peak_defs: List[dict], kept: pd.DataFrame,
                    fit_regions, x_unit: str, use_normalised: bool,
                    positive_intensity: bool, allow_negative_intensity: bool,
                    baseline_regions, baseline_sigma: float, baseline_max_iter: int) -> dict:
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

    x_full = results["x_ppm"] if x_unit == "ppm" else results["x_Hz"]
    mask = build_region_mask(np.asarray(x_full), fit_regions)
    x_fit = np.asarray(x_full)[mask]

    y_model, components = evaluate_fit(x_fit, popt, peak_defs, x_ref=x_ref, return_components=True)
    integrals, integrals_norm = compute_integrals(x_fit, components)
    integral_errors = compute_integral_errors(popt, pcov, peak_defs, integrals)

    environment_report = split_integrals_by_environment(
        kept, integrals, integrals_norm, integral_errors,
    )

    return {
        "popt": popt, "pcov": pcov, "is_free": is_free, "x_ref": x_ref,
        "bound_warnings": bound_warnings, "x_fit": x_fit, "mask": mask,
        "y_model": y_model, "components": components, "integrals": integrals,
        "integrals_norm": integrals_norm, "integral_errors": integral_errors,
        "environment_report": environment_report,
    }


# ===========================================================================
# 6. End-to-end driver
# ===========================================================================
def fit_predicted_spectrum(
    spectrum, cif_filename, atom_of_interest, shell_bins, shift_dict,
    ignore_species=None, fit_regions=None, tolerance=None, group_method="probability",
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

    fitted = _fit_peak_defs(
        results, peak_defs, kept, fit_regions=fit_regions, x_unit=x_unit,
        use_normalised=use_normalised, positive_intensity=positive_intensity,
        allow_negative_intensity=allow_negative_intensity,
        baseline_regions=baseline_regions, baseline_sigma=baseline_sigma,
        baseline_max_iter=baseline_max_iter,
    )
    popt, pcov = fitted["popt"], fitted["pcov"]
    is_free, bound_warnings = fitted["is_free"], fitted["bound_warnings"]
    x_fit, components = fitted["x_fit"], fitted["components"]
    integrals, integrals_norm = fitted["integrals"], fitted["integrals_norm"]
    integral_errors = fitted["integral_errors"]
    environment_report = fitted["environment_report"]

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
        y_fit_data = np.asarray(y_full)[fitted["mask"]]
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


# ===========================================================================
# 7. Simplify a maximal fit into far fewer peaks, two ways, before
#    refitting with shift free (see refit_merged_groups below). Both take
#    `kept` (the maximal fit's group table, e.g. fit_predicted_spectrum's
#    "kept") and return a new groups-shaped table -- same
#    group_shift/group_p/n_members/span/members columns as
#    group_environments' output, so it slots straight into
#    select_groups/groups_to_peak_defs/split_integrals_by_environment --
#    plus provenance columns recording which maximal-fit peak(s) fed into
#    each merged group.
# ===========================================================================
def _aggregate_members(rows: pd.DataFrame) -> dict:
    """Flatten the `members` lists of several `kept` rows into one merged
    group, recomputing group_shift/group_p/span/fraction_in_group exactly
    as group_environments would for a single group containing all of
    these environments together."""
    all_members = []
    for _, r in rows.iterrows():
        for m in r["members"]:
            all_members.append({"label": m["label"], "shift": m["shift"], "p": m["p"]})

    total_p = sum(m["p"] for m in all_members)
    shifts = [m["shift"] for m in all_members]
    if total_p > 0:
        weighted_shift = sum(m["shift"] * m["p"] for m in all_members) / total_p
        for m in all_members:
            m["fraction_in_group"] = m["p"] / total_p
    else:
        weighted_shift = float(np.mean(shifts)) if shifts else np.nan
        for m in all_members:
            m["fraction_in_group"] = np.nan

    return {
        "group_shift": float(weighted_shift), "group_p": float(total_p),
        "n_members": len(all_members),
        "span": float(max(shifts) - min(shifts)) if shifts else 0.0,
        "members": all_members,
    }


def merge_low_integral_peaks(
    kept: pd.DataFrame, integrals: Sequence[float],
    integral_errors: Optional[Sequence[float]] = None,
    integral_threshold: Optional[float] = None, relative_threshold: float = 0.05,
    merge_insignificant: bool = True, n_sigma: float = 2.0,
    merge_tolerance: Optional[float] = None, tolerance: Optional[float] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Automatically simplify a maximal fit by absorbing low-integral peaks
    into their nearest neighbour, so the result has far fewer peaks --
    intended to be re-fitted afterwards with shift free (see
    refit_merged_groups): unlike the CIF-locked maximal fit, a merged
    peak's centre is no longer forced to a single environment's predicted
    shift, so floating it lets the refit find where the data actually
    puts it.

    `kept` and `integrals` must be in the SAME order (i.e. exactly the
    peak_defs order used to produce `integrals` -- as returned together by
    fit_predicted_spectrum's "kept"/"integrals").

    A peak counts as "low integral" (a merge candidate) if ANY of:
      * `integral_threshold` is given and integral < integral_threshold
      * integral < relative_threshold * (largest integral in the fit)
        (default: below 5% of the biggest peak)
      * `merge_insignificant` (default True) and `integral_errors` is
        given and integral < n_sigma * that peak's integral_abs_err --
        i.e. statistically indistinguishable from zero. This check only
        ever applies to an original, un-merged peak's own error; once two
        peaks have merged there's no single error estimate to reuse, so
        only the threshold checks above apply to a merged bucket.

    Algorithm: repeatedly take the lowest-integral candidate peak still
    unmerged, and merge it into whichever OTHER peak (candidate or not) is
    nearest to it in shift -- provided that nearest peak is within
    `merge_tolerance` ppm. The merged peak's shift is the integral-weighted
    mean of the two and its integral is their sum; the merged peak then
    re-enters the pool, so merges can cascade (one peak absorbing several
    low neighbours in turn). A low peak with nothing else within
    `merge_tolerance` is left alone, rather than forced onto a distant
    peak.

    merge_tolerance : float, optional
        Defaults to `tolerance` (e.g. reuse the tolerance the maximal fit
        was grouped with) if given.

    Returns
    -------
    (merged_groups, merge_log)

    merged_groups : DataFrame, group_shift/group_p/n_members/span/members
    as in group_environments' output, PLUS n_source_peaks, source_shifts,
    source_integrals (which maximal-fit peaks/integrals fed into this
    group) and source_span (their shift spread, i.e. how far apart the
    merged peaks originally were).

    merge_log : DataFrame, one row per merge event actually performed --
    inspect this (like `members`/`dropped` elsewhere) before trusting the
    result; the cascade is a greedy, order-dependent simplification, not
    a unique answer.
    """
    if merge_tolerance is None:
        if tolerance is None:
            raise ValueError(
                "Supply merge_tolerance explicitly, or tolerance (e.g. the "
                "tolerance the maximal fit was grouped with) to use as a "
                "default merge_tolerance."
            )
        merge_tolerance = tolerance

    kept = kept.reset_index(drop=True)
    integrals = np.asarray(integrals, dtype=float)
    if len(integrals) != len(kept):
        raise ValueError("integrals must be the same length as kept (same order as peak_defs).")
    if integral_errors is not None:
        integral_errors = np.asarray(integral_errors, dtype=float)
    max_integral = integrals.max() if len(integrals) else 0.0

    def _is_low(bucket):
        low = False
        if integral_threshold is not None:
            low = low or bucket["integral"] < integral_threshold
        low = low or bucket["integral"] < relative_threshold * max_integral
        if merge_insignificant and bucket["error"] is not None:
            low = low or bucket["integral"] < n_sigma * bucket["error"]
        return low

    next_id = 0
    buckets = []
    for i in range(len(kept)):
        err_i = None
        if integral_errors is not None and np.isfinite(integral_errors[i]):
            err_i = float(integral_errors[i])
        buckets.append({
            "id": next_id, "shift": float(kept.loc[i, "group_shift"]),
            "integral": float(integrals[i]), "error": err_i, "row_indices": [i],
        })
        next_id += 1

    merge_log = []
    skip_ids = set()
    while True:
        low = [b for b in buckets if b["id"] not in skip_ids and _is_low(b)]
        if not low:
            break
        bi = min(low, key=lambda b: b["integral"])
        others = [b for b in buckets if b["id"] != bi["id"]]
        if not others:
            break
        bj = min(others, key=lambda b: abs(b["shift"] - bi["shift"]))
        if abs(bj["shift"] - bi["shift"]) > merge_tolerance:
            skip_ids.add(bi["id"])
            continue

        merged_integral = bi["integral"] + bj["integral"]
        merged_shift = (
            (bi["shift"] * bi["integral"] + bj["shift"] * bj["integral"]) / merged_integral
            if merged_integral > 0 else 0.5 * (bi["shift"] + bj["shift"])
        )
        merge_log.append({
            "absorbed_shift": bi["shift"], "absorbed_integral": bi["integral"],
            "into_shift": bj["shift"], "into_integral": bj["integral"],
            "merged_shift": merged_shift, "merged_integral": merged_integral,
        })
        buckets = [b for b in buckets if b["id"] not in (bi["id"], bj["id"])]
        buckets.append({
            "id": next_id, "shift": merged_shift, "integral": merged_integral,
            "error": None, "row_indices": bi["row_indices"] + bj["row_indices"],
        })
        next_id += 1
        skip_ids.clear()  # geometry changed -- a locked bucket may now have a new neighbour

    rows = []
    for b in sorted(buckets, key=lambda b: b["shift"]):
        sub = kept.loc[b["row_indices"]]
        agg = _aggregate_members(sub)
        source_shifts = [float(kept.loc[i, "group_shift"]) for i in b["row_indices"]]
        source_integrals = [float(integrals[i]) for i in b["row_indices"]]
        agg.update({
            "n_source_peaks": len(b["row_indices"]), "source_shifts": source_shifts,
            "source_integrals": source_integrals,
            "source_span": float(max(source_shifts) - min(source_shifts)),
        })
        rows.append(agg)

    merged_groups = pd.DataFrame(rows).sort_values("group_shift").reset_index(drop=True)
    return merged_groups, pd.DataFrame(merge_log)


def merge_groups_to_picks(
    kept: pd.DataFrame, picks: Sequence[float],
    integrals: Optional[Sequence[float]] = None,
    max_distance: Optional[float] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Manual alternative to merge_low_integral_peaks: instead of an
    automatic integral threshold, YOU name the peak centres you want
    (`picks`, a list of shifts -- e.g. read off the maximal fit or the raw
    spectrum by eye), and every maximal-fit peak in `kept` is merged into
    whichever pick is nearest to it. Intended to be re-fitted afterwards
    with shift free to move, exactly like merge_low_integral_peaks (see
    refit_merged_groups).

    integrals : optional, same order/length as `kept`
        Purely for provenance (recorded per pick in `source_integrals`, so
        you can see how much intensity ended up assigned to each pick) --
        it plays no role in the assignment, which is by shift only.
    max_distance : float, optional
        If given, a maximal-fit peak more than max_distance ppm from
        EVERY pick is excluded rather than force-assigned to the nearest
        (distant) pick -- it's recorded in `assignment_report` with
        assigned=False, not folded into any group. Default (None): always
        assign to the nearest pick, however far.

    Returns
    -------
    (merged_groups, assignment_report)

    merged_groups : same shape as merge_low_integral_peaks' output, one
    row per pick that received at least one peak, PLUS `pick_shift` (the
    value you supplied) alongside `group_shift` (the CIF-probability-
    weighted prediction for what got assigned to it -- compare the two
    before refitting: a pick far from its own group_shift is a sign you
    picked a point the CIF prediction doesn't actually support). Picks
    that received nothing are skipped (with a printed warning), not
    silently included as an empty peak.

    assignment_report : one row per input peak in `kept`, recording which
    pick (if any) it was assigned to and the distance.
    """
    if len(picks) == 0:
        raise ValueError("picks must be a non-empty list of shifts.")
    picks = list(picks)

    kept = kept.reset_index(drop=True)
    if integrals is not None:
        integrals = np.asarray(integrals, dtype=float)
        if len(integrals) != len(kept):
            raise ValueError("integrals must be the same length as kept.")

    assign_rows = []
    pick_members: Dict[int, List[int]] = {k: [] for k in range(len(picks))}
    for i, row in kept.iterrows():
        shift = float(row["group_shift"])
        dists = [abs(shift - p) for p in picks]
        k = int(np.argmin(dists))
        d = dists[k]
        assigned = max_distance is None or d <= max_distance
        assign_rows.append({
            "peak_shift": shift, "nearest_pick": picks[k], "distance": d,
            "assigned": assigned,
        })
        if assigned:
            pick_members[k].append(i)

    rows = []
    for k, pick in enumerate(picks):
        idxs = pick_members[k]
        if not idxs:
            print(f"merge_groups_to_picks: pick at {pick:.3f} ppm received no "
                  f"maximal-fit peaks -- skipped.")
            continue
        sub = kept.loc[idxs]
        agg = _aggregate_members(sub)
        source_shifts = [float(kept.loc[i, "group_shift"]) for i in idxs]
        source_integrals = [float(integrals[i]) for i in idxs] if integrals is not None else []
        agg.update({
            "pick_shift": float(pick), "n_source_peaks": len(idxs),
            "source_shifts": source_shifts, "source_integrals": source_integrals,
            "source_span": float(max(source_shifts) - min(source_shifts)),
        })
        rows.append(agg)

    if not rows:
        raise ValueError("None of the picks received any maximal-fit peaks -- nothing to merge.")

    merged_groups = pd.DataFrame(rows).sort_values("pick_shift").reset_index(drop=True)
    assignment_report = pd.DataFrame(assign_rows)
    return merged_groups, assignment_report


# ===========================================================================
# 8. Re-fit a simplified (merged) peak set with shift now FREE to move,
#    and report how far each fitted centre drifted from the CIF-predicted
#    shift it was built around.
# ===========================================================================
def _peak_center_from_component(x_fit: np.ndarray, component: np.ndarray) -> float:
    """Model-agnostic fitted-centre estimate for one fitted peak: a
    parabola-refined argmax of the fitted component itself, rather than
    reading a 'shift' parameter out of popt -- this way it works the same
    for gaussian/lorentzian/pseudo_voigt without needing to know
    fitting.py's internal parameter ordering."""
    y = np.abs(np.asarray(component))
    k = int(np.argmax(y))
    if k == 0 or k == len(y) - 1:
        return float(x_fit[k])
    y0, y1, y2 = y[k - 1], y[k], y[k + 1]
    denom = y0 - 2 * y1 + y2
    if denom == 0:
        return float(x_fit[k])
    delta = 0.5 * (y0 - y2) / denom
    dx = x_fit[k + 1] - x_fit[k]
    return float(x_fit[k] + delta * dx)


def _build_shift_report(kept: pd.DataFrame, x_fit: np.ndarray, components) -> pd.DataFrame:
    """One row per fitted peak: the CIF-predicted group_shift it started
    from vs. its fitted centre, and the difference between them."""
    rows = []
    for (_, row), component in zip(kept.iterrows(), components):
        predicted = float(row["group_shift"])
        fitted = _peak_center_from_component(x_fit, component)
        rows.append({
            "predicted_shift": predicted, "fitted_shift": fitted,
            "shift_delta": fitted - predicted,
            "n_members": row.get("n_members", np.nan),
            "n_source_peaks": row.get("n_source_peaks", np.nan),
        })
    return pd.DataFrame(rows)


def print_shift_drift_report(shift_report: pd.DataFrame):
    """Human-readable print of _build_shift_report's output: how far each
    merged/simplified peak's fitted centre moved from the CIF-predicted
    shift it was built around."""
    print("\n" + "=" * 70)
    print("FITTED SHIFT vs. CIF-PREDICTED SHIFT (merged/simplified peaks)")
    print("=" * 70)
    for _, r in shift_report.sort_values("predicted_shift").iterrows():
        n_src = r.get("n_source_peaks", np.nan)
        src_txt = f", merged from {int(n_src)} maximal-fit peak(s)" if pd.notna(n_src) else ""
        print(f"    predicted {r['predicted_shift']:8.3f} ppm  ->  "
              f"fitted {r['fitted_shift']:8.3f} ppm   "
              f"(delta {r['shift_delta']:+.3f} ppm{src_txt})")
    print("=" * 70 + "\n")


def refit_merged_groups(
    merged_groups: pd.DataFrame, results: Dict, shift_float_window: float,
    fit_regions=None, x_unit: str = "ppm", snr_threshold: float = 3.0,
    noise_region=None, window: Optional[float] = None,
    peak_type: str = "gaussian", fwhm_init: float = 5.0, fwhm_bounds=(0.1, 100.0),
    eta_init: float = 0.5, eta_bounds=(0.0, 1.0), amplitude_bounds=(0.0, np.inf),
    positive_intensity: bool = True, allow_negative_intensity: bool = False,
    baseline_regions=None, baseline_sigma: float = 2.5, baseline_max_iter: int = 10,
    use_normalised: bool = True, show_plot: bool = True, show_components: bool = True,
    show_report: bool = True, xlims="default", xlabel=None,
) -> dict:
    """
    Re-fit a simplified peak set (from merge_low_integral_peaks or
    merge_groups_to_picks) against the SAME experimental spectrum the
    maximal fit used -- this time with each peak's shift FREE to move by
    +/- shift_float_window ppm, since there are now far fewer, better-
    separated peaks and floating the centre is much less likely to
    produce a degenerate fit than it would have been on the maximal,
    one-peak-per-environment fit.

    shift_float_window : float
        Required, no default -- how far, in ppm, each peak's fitted
        centre is allowed to move from `group_shift` (the CIF-
        probability-weighted prediction for the environments merged into
        it). There's no universally sensible default; pick something you
        can justify (e.g. the tolerance/merge_tolerance used to build the
        merged groups, or your estimate of plausible referencing error).

    All other parameters mirror fit_predicted_spectrum's (see there) and
    are forwarded to select_groups/groups_to_peak_defs/fit_spectrum in the
    same way, just starting from `merged_groups` instead of a freshly
    predicted-and-grouped CIF table.

    Returns
    -------
    dict with the same keys as fit_predicted_spectrum's return (results,
    kept, dropped, noise, peak_defs, popt, pcov, is_free, bound_warnings,
    integrals, integrals_norm, integral_errors, environment_report,
    figure, axes) -- "predicted"/"groups" are replaced by the single
    `merged_groups` you started from -- PLUS `shift_report`: one row per
    fitted peak comparing its fitted centre to the CIF-predicted
    group_shift it started from. `shift_report['shift_delta']` is the
    number to quote when telling users how far this simplified fit moved
    a peak from its original CIF prediction.
    """
    if shift_float_window is None or shift_float_window <= 0:
        raise ValueError(
            "shift_float_window must be a positive ppm value -- the whole "
            "point of refitting a merged/simplified peak set is to let its "
            "centre move; pass e.g. the tolerance/merge_tolerance used to "
            "build the merged groups, or your estimate of referencing "
            "uncertainty."
        )

    kept, dropped, noise = select_groups(
        merged_groups, results, fit_regions=fit_regions, x_unit=x_unit,
        snr_threshold=snr_threshold, noise_region=noise_region,
        window=window, tolerance=None, use_normalised=False,
    )
    if kept.empty:
        raise ValueError(
            "No merged group survived region/S:N filtering -- nothing to "
            "fit. Inspect `dropped` or loosen fit_regions/snr_threshold."
        )

    peak_defs = groups_to_peak_defs(
        kept, peak_type=peak_type, fwhm_init=fwhm_init, fwhm_bounds=fwhm_bounds,
        eta_init=eta_init, eta_bounds=eta_bounds, amplitude_bounds=amplitude_bounds,
        float_shift_window=shift_float_window,
    )

    fitted = _fit_peak_defs(
        results, peak_defs, kept, fit_regions=fit_regions, x_unit=x_unit,
        use_normalised=use_normalised, positive_intensity=positive_intensity,
        allow_negative_intensity=allow_negative_intensity,
        baseline_regions=baseline_regions, baseline_sigma=baseline_sigma,
        baseline_max_iter=baseline_max_iter,
    )

    shift_report = _build_shift_report(kept, fitted["x_fit"], fitted["components"])

    fig = ax = None
    if show_plot:
        fig, ax = NMR1dplot_with_fit(
            results=results, popt=fitted["popt"], peak_defs=peak_defs, fit_regions=fit_regions,
            x_unit=x_unit, show_components=show_components, use_normalised=use_normalised,
            xlims=xlims, xlabel=xlabel,
        )

    if show_report:
        y_full = results["normalised_real_spectrum"] if use_normalised else results["real_spectrum"]
        y_fit_data = np.asarray(y_full)[fitted["mask"]]
        print_fit_report(fitted["x_fit"], y_fit_data, fitted["popt"], fitted["pcov"],
                          peak_defs, fitted["bound_warnings"], fitted["is_free"],
                          components=fitted["components"])
        print_environment_report(fitted["environment_report"], dropped)
        print_shift_drift_report(shift_report)

    return {
        "results": results, "merged_groups": merged_groups,
        "kept": kept, "dropped": dropped, "noise": noise,
        "peak_defs": peak_defs, "popt": fitted["popt"], "pcov": fitted["pcov"],
        "is_free": fitted["is_free"], "bound_warnings": fitted["bound_warnings"],
        "integrals": fitted["integrals"], "integrals_norm": fitted["integrals_norm"],
        "integral_errors": fitted["integral_errors"],
        "environment_report": fitted["environment_report"],
        "shift_report": shift_report, "figure": fig, "axes": ax,
    }


# ===========================================================================
# 9. Alternative to groups_to_peak_defs + fit_spectrum: fit the WHOLE
#    predicted envelope shape (no discretisation into separate peaks at
#    all). Appropriate when the CIF predicts a dense, quasi-continuous
#    comb/manifold of shifts (heavy combinatorial disorder, or pathway
#    constants with near-commensurate ratios -- see group_environments'
#    chaining caveat) rather than a handful of well-separated sites.
# ===========================================================================
def _envelope_model(x, shifts, p, shift_scale, x_offset, sigma, intensity_scale,
                     b0, b1, x_ref):
    """intensity_scale * sum_i p_i * Gaussian(x; shifts_i*shift_scale + x_offset, sigma)
    + linear baseline. Vectorised over all predicted sticks at once."""
    shifted = shifts * shift_scale + x_offset
    z = (x[:, None] - shifted[None, :]) / sigma
    y = intensity_scale * (p[None, :] * np.exp(-0.5 * z ** 2)).sum(axis=1)
    return y + b0 + b1 * (x - x_ref)


def fit_cif_envelope(
    spectrum, cif_filename, atom_of_interest, shell_bins, shift_dict,
    ignore_species=None, fit_regions=None, x_unit="ppm", procno=1,
    use_normalised=True, sigma_init=15.0, sigma_bounds=(1.0, 300.0),
    shift_scale_init=1.0, shift_scale_bounds=(0.5, 1.5),
    x_offset_init=0.0, x_offset_bounds=(-100.0, 100.0),
    baseline_regions=None, baseline_sigma=2.5, baseline_max_iter=10,
    vacancy_tol=5, warn_missing_shifts=True, verbose=False,
):
    """
    Fit an experimental spectrum to the CIF-predicted shift distribution
    as a single continuous envelope, rather than discretising it into
    separate peaks (see group_environments/select_groups/
    groups_to_peak_defs/fit_predicted_spectrum for that alternative).

    Free parameters: an inhomogeneous broadening sigma (ppm), a
    multiplicative scale on ALL shift_dict pathway constants at once
    (shift_scale -- corrects a systematically over/under-estimated
    contact-shift magnitude without needing you to re-guess every
    pathway constant by hand), a rigid referencing offset (x_offset),
    and an overall intensity scale. Baseline is estimated robustly and
    held fixed, exactly as in fitting.fit_spectrum.

    This sidesteps grouping entirely, so it has none of
    group_environments' chaining failure modes -- but it also can't tell
    you "this peak is environment X" the way fit_predicted_spectrum can,
    because there ARE no discrete peaks in this model. Use it to (a)
    sanity-check/calibrate shift_dict against real data, and/or (b) as
    the appropriate fit itself when the CIF predicts a dense manifold of
    shifts rather than a few resolvable sites (which
    group_environments/select_groups will tell you, via a small
    `n_groups` for a sane tolerance and/or a `max_group_span` error).

    Returns
    -------
    dict with keys: results, shifts, p (the raw, ungrouped sticks used),
    popt (dict of fitted values), pcov, x_fit, y_fit, y_model, rmse.
    """
    if isinstance(spectrum, str):
        results = NMR1dimport(spectrum, procno=procno)
    else:
        results = spectrum

    predicted = predict_environments(
        cif_filename, atom_of_interest, shell_bins, shift_dict,
        ignore_species=ignore_species, vacancy_tol=vacancy_tol,
        warn_missing_shifts=warn_missing_shifts, verbose=verbose,
    )
    shifts = predicted["Total_shift"].to_numpy()
    p = predicted["p"].to_numpy()

    x_full = np.asarray(results["x_ppm"] if x_unit == "ppm" else results["x_Hz"], dtype=float)
    y_full_raw = np.asarray(
        results["normalised_real_spectrum"] if use_normalised else results["real_spectrum"],
        dtype=float,
    )
    scale_norm = np.max(np.abs(results["real_spectrum"])) if use_normalised else 1.0
    scale_norm = scale_norm if scale_norm != 0 else 1.0

    mask = build_region_mask(x_full, fit_regions)
    x_fit = x_full[mask]
    y_fit = y_full_raw[mask]
    x_ref = np.mean(x_fit)

    b0_init, b1_init, _ = estimate_baseline_robust(
        x_full, y_full_raw, baseline_regions=baseline_regions,
        sigma=baseline_sigma, max_iter=baseline_max_iter, x_ref=x_ref,
    )

    # Rough intensity_scale starting guess: ratio of data amplitude to
    # unscaled-envelope amplitude at sigma_init.
    trial = _envelope_model(x_fit, shifts, p, shift_scale_init, x_offset_init,
                             sigma_init, 1.0, 0.0, 0.0, x_ref)
    trial_peak = np.max(np.abs(trial))
    data_peak = np.max(np.abs(y_fit - (b0_init + b1_init * (x_fit - x_ref))))
    intensity_scale_init = data_peak / trial_peak if trial_peak > 0 else 1.0

    from scipy.optimize import curve_fit

    def wrapper(x, shift_scale, x_offset, sigma, intensity_scale):
        return _envelope_model(x, shifts, p, shift_scale, x_offset, sigma,
                                intensity_scale, b0_init, b1_init, x_ref)

    p0 = [shift_scale_init, x_offset_init, sigma_init, intensity_scale_init]
    lower = [shift_scale_bounds[0], x_offset_bounds[0], sigma_bounds[0], 0.0]
    upper = [shift_scale_bounds[1], x_offset_bounds[1], sigma_bounds[1], np.inf]

    popt, pcov = curve_fit(wrapper, x_fit, y_fit, p0=p0, bounds=(lower, upper), maxfev=20000)

    y_model = wrapper(x_fit, *popt)
    rmse = float(np.sqrt(np.mean((y_fit - y_model) ** 2)))

    popt_named = {
        "shift_scale": popt[0], "x_offset": popt[1],
        "sigma": popt[2], "intensity_scale": popt[3],
        "b0": b0_init, "b1": b1_init,
    }

    return {
        "results": results, "shifts": shifts, "p": p,
        "popt": popt_named, "pcov": pcov,
        "x_fit": x_fit, "y_fit": y_fit, "y_model": y_model, "rmse": rmse,
    }
