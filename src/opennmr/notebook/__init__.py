"""
nmrtools
========

Tools for importing, plotting, and fitting Bruker NMR data (1D and 2D).

    from nmrtools import (
        NMR1dimport, NMR1dplot, VOCSSum, apply_VOCS,
        NMR2dimport, NMR2Dplotter, NMR2d_get_trace, NMR2D_get_slices,
        NMR2D_multi_plot, NMR2D_overlay_plot,
        NMR2D_plot_slices, NMR2D_summary_plot,
        fit_spectrum, NMRFitWorkflow,
        NMR2D_fit_slices, NMR2D_plot_integrals, NMR2D_workflow,
    )
"""

from .utils import (
    find_gamma, NUC_label, read_bruker_metadata, read_bruker_fid, read_vdlist,
    compute_BF1_SFO1,
)

from .plotting import (
    setup_plotting, get_stack_colormap, get_contour_cmap, downsample_2d,
    apply_tick_style, apply_spine_style, save_figure,
    plot_spectrum_dict, plot_comparison, plot_stacked_spectra, plot_quadrupolar_orders,
)
# Colour-blind-/print-friendly palette checks -- the same functions the
# GUI's Plot Style panel uses, so a notebook figure can be checked (and
# a palette chosen) exactly as the GUI does it.
# Legend/series names with LaTeX: check one, or make one safe to draw --
# the same checks the GUI's Rename… and Plot Style name boxes use.
from ..plotting.labels import label_problem, safe_label, label_traces, set_legend_linewidth
from ..plotting.accessibility import (
    friendly_palettes, max_safe_series, describe_accessibility, palette_distinctness,
    is_colourblind_safe, is_print_safe, trace_colour_presets, series_colours,
)

from .one_d import (
    NMR1dimport, NMR1dplot, NMR1dheatmap,
    vocs_f1, vocs_f2, vocs_power, pulse_to_w1, apply_VOCS, VOCSSum,
    CPMGprocess, CPMGplot, cpmg_echo_layout, cpmg_to_spectrum,
)

from .compare_panels import plot_compare_panels, NMR1D_multi_plot
from .panel_labels import add_panel_labels

from .two_d import (
    NMR2dimport, NMR2Dplotter, NMR2D_multi_plot, NMR2D_overlay_plot,
    NMR2d_get_trace, NMR2D_get_slices, prepare_trace_for_fit,
    NMR2D_plot_slices, NMR2D_summary_plot,
    canonical_nucleus, check_external_projection,
)

from .fitting import (
    gaussian, lorentzian, pseudo_voigt,
    estimate_baseline_from_edges, estimate_baseline_robust,
    fit_spectrum, evaluate_fit, compute_integrals, compute_integral_errors,
    fit_quality, print_fit_report, NMR1dplot_with_fit, NMRFitWorkflow,
    estimate_noise_level, select_2d_traces,
    NMR2D_fit_slices, NMR2D_print_fit_report, NMR2D_plot_integrals,
    NMR2D_workflow,
)

from .relaxation import (
    saturation_recovery, inversion_recovery, exponential_decay,
    biexponential_decay, biexponential_recovery,
    triexponential_decay, triexponential_recovery,
    stretched_decay, stretched_recovery, RELAXATION_MODELS,
    fit_relaxation, format_time_value, print_relaxation_report,
    NMR2D_integrate_slices,
    NMR2D_fit_relaxation, NMR2D_plot_relaxation,
    NMR2D_relaxation_workflow,
)
# Relaxometry extras (2026-10-06): threshold regions, trapezoid/Simpson,
# correlation matrices.
from . import relaxometry
from .relaxometry import find_integration_regions, correlation_matrix
# The correlation matrix of any fit -- relaxation, isotropic peaks,
# Solid-state NMR, multi-spectrum (2026-10-06).
# DFT-calculated tensors (CASTEP .magres, VASP OUTCAR) -> sites (2026-10-06).
from . import dft_nmr
from .dft_nmr import read_calculation, dft_sites, simulate_dft, plot_dft_tensors
from . import dft_structure
from .dft_structure import plot_dft_structure
from . import dft_calibration
from .dft_calibration import fit_calibration, reference_point, plot_calibration
# Paramagnetic NMR: g/ZFS/hyperfine shifts, pseudocontact shifts, lanthanide
# series, bond pathways (2026-11-06).
from . import paramagnetic_nmr, pseudocontact, lanthanides, bond_pathways
from . import fit_correlation
from .fit_correlation import plot_correlation_matrix, correlation_of, multi_fit_correlation

from .arithmetic import combine_spectra, combine_spectra_as_results
# Dynamics: relaxation against temperature (BPP / Arrhenius / Hendrickson-
# Bray) and exchange (ion-hopping) lineshapes -- Dynamics ▸ Relaxation vs
# Temperature and Dynamics ▸ Exchange Lineshape in the GUI.
from .vt_relaxation import (
    vt_dataset, load_vt_csv, fit_vt_relaxation, compare_vt_models, fit_arrhenius,
    hendrickson_bray, fit_hendrickson_bray, waugh_fedin_Ea, plot_arrhenius,
    plot_vs_temperature, plot_hendrickson_bray, fit_vt_regions, plot_arrhenius_regions,
)
from .exchange import (
    exchange_spectrum, exchange_spectra_vs_temperature, fit_exchange_spectra,
    plot_exchange_stack, plot_exchange_fit, coalescence_temperature,
)
# Tikhonov regularisation: T1/T2/D distributions and T1-T2, T2-T2, D-T2
# maps (a port of the MATLAB_RELAX_ALGORITHM scripts) -- Dynamics ▸
# Regularisation in the GUI.
from . import regularisation
from .regularisation import (
    regularise, regularise_2d, plot_regularisation, plot_regularisation_2d,
    static_gradient_b, subsample as regularisation_subsample,
    suggest_range as regularisation_grid_range,
    report as regularisation_report, to_csv as regularisation_to_csv,
)
# Referencing across nuclei, and linear field-drift correction
# (Najbauer & Andreas, J. Magn. Reson. 305 (2019) 1-4) -- Process ▸
# Reference and Process ▸ Field Drift Correction in the GUI.
from .referencing import transfer_sr, transfer_sr_between
# Phase each slice of a pseudo-2D on its own -- Process ▸ Phase ▸ "Autophase each
# slice independently" in the GUI.
from .slice_phase import autophase_slices
from .drift_correction import (
    correct_bruker_dataset, concatenate_nus_datasets, measure_drift_hz, drift_profile,
    correct_pseudo_2d_fids,
)
from .manual_baseline import (
    fit_manual_baseline, evaluate_manual_baseline, manual_baseline_correct,
    plot_manual_baseline, describe_manual_baseline, polynomial_baseline, sine_baseline,
    MANUAL_BASELINE_PARAMS, MANUAL_BASELINE_EQUATIONS,
)

from .simulations import (
    haeberlen_csa_tensor, herzfeld_berger_csa_tensor, csa_definition, csa_axis_limits,
    csa_frequency, powder_orientations, powder_orientations_octant, 
    static_csa, make_static_csa_spectrum, finite_mas_csa_spectrum,
    quadrupolar_static_spectrum, quadrupolar_mas_spectrum, quadrupolar_spectrum,
    quadrupolar_order_comparison, quadrupolar_transition_frequencies,
    quadrupolar_transition_table, quadrupolar_axis_limits, quadrupolar_nu_Q,
    quadrupolar_ct_isotropic_shift_Hz, larmor_frequency_Hz, zcw_angles, lebedev_angles, powder_grid_size,
    spin_multiplet_pattern, jcoupling_group_pattern, combine_jcouplings,
    multiplet_lineshape, jcoupled_spectrum, multi_site_jcoupled_spectrum, element_isotopes,
)

from .quadrupolar_fitting import (
    fit_quadrupolar_spectrum, evaluate_quad_fit, estimate_quadrupolar_start,
    quadrupolar_derived, print_quadrupolar_fit_report, NMR1dplot_with_quad_fit,
    NMRQuadFitWorkflow,
)

from .czjzek import (
    czjzek_pdf, czjzek_statistics, czjzek_spectrum, fit_czjzek_spectrum, evaluate_czjzek_fit,
    czjzek_fit_report, plot_czjzek_distribution, plot_czjzek_fit, CzjzekLibrary,
    czjzek_samples, extended_czjzek_weights,
)
from .fit_display import draw_fit_components, component_style
from .fit_models import (
    make_fit_model, fit_statistics, compare_fit_models, format_comparison, plot_fit_comparison,
    draw_fit_model, fit_view,
)
from .state import save_state, load_state, describe_state, saved_fit_models

from .combined_hamiltonian import (
    Site, simulate_combined, combined_axis_limits, combined_to_python, copy_interactions,
    site_transition_frequencies, plot_combined, sites_to_json, sites_from_json,
    resolve_links,
)
from . import isotopes
from .isotopes import frequency_ratio, to_observed_ppm, isotope_partners, partner_factors

from .combined_fitting import (
    fit_combined, fit_combined_multistart, print_combined_fit_report, plot_combined_fit,
    NMRCombinedFitWorkflow, site_parameter_names,
)

from .spin_system import (
    expand_spin_system, block_sizes, spin_system_transitions, spin_system_spectrum,
    coupling_strengths, fit_spin_system_spectrum, evaluate_spin_system_fit, fitted_system,
    equivalent_solutions,
)
# Weak (first-order) and strong (exact) J-coupled sets on one spectrum.
from .mixed_j import mixed_j_spectrum, split_spin_system

from .notes import experiment_notes, set_experiment_notes, notes_location, notes_first_line
from .stored_integrals import StoredIntegrals, get_stored_integrals
from . import field_drift
from .plot_inset import add_inset, remove_insets, insets_of, INSET_DEFAULTS, INSET_CORNERS
from .peak_integral_labels import (
    integral_labels, add_integral_labels, remove_integral_labels, normalised_integrals,
)
from .peak_integral_labels import reference_region as integral_reference_region
# The log / normalised plot of a curve fit (Stejskal-Tanner for PFG), and the
# model SCORE resolves with -- the same functions the GUI calls.
from ..processing.relaxation import attenuation, attenuation_values, ATTENUATION_MODES
from ..processing.score import kernel_model as score_kernel_model, residual_metrics
from .multi_fitting import (
    make_experiment, fit_multi, default_scopes, multi_fit_table, multi_fit_to_csv,
    parameter_trend, print_multi_fit_report, plot_multi_fit, plot_parameter_trend,
    MultiFitCancelled,
)
from .tensor_views import (
    frequency_surface, tensor_components, transition_labels, plot_tensors,
    plot_tensor_surface, plot_tensor_ellipsoids,
)
# The Bloch simulator (shaped and adiabatic pulses) lives in
# opennmr.simulate.bloch; exported here so a notebook has it with the rest.
from ..simulate.bloch import (
    simulate_pulse, adiabatic_shape, adiabaticity, peak_b1_for_adiabaticity,
    peak_b1_for_flip_angle, inverted_band_hz, plot_pulse_simulation, ADIABATIC_SHAPES,
)
from .dnp import (
    carrier_frequency_mhz, carrier_offset_hz, field_from_frequency, field_from_sfo1,
    peak_height, region_integral, enhancement, enhancement_report, print_enhancement_report,
    field_sweep, field_sweep_table, field_sweep_to_csv, field_sweep_report,
    print_field_sweep_report, plot_field_profile, plot_sweep_spectra, plot_enhancement_spectra,
)
# The step-by-step DNP workflow (align, measure, profile / enhancement):
# used as a module, dnp_workflow.align_spectra(...) etc.
from . import dnp_workflow
# Signal-to-noise with stated statistics, one or many spectra (Integrate ▸
# Signal-to-Noise; 2026-10-12): used as snr.signal_to_noise(...) etc.
from . import snr
from .snr import signal_to_noise, signal_to_noise_many, snr_report, plot_snr
# Truncation "wiggles" in F2 (2026-10-29): ringing.check_f2_ringing(spectrum) says whether
# the FID is cut off, how big the wiggles are, and which sine-bell removes them.
from . import ringing
from .ringing import check_f2_ringing
# 2026-10-16: intensities per scan / per gram / per RG (Compare ▸ Normalize),
# click-to-measure peak widths, several fit ranges, CP build-up curves.
from . import intensity_scaling, peak_width, fit_ranges, cp_buildup
from .intensity_scaling import scale_for_comparison
from .peak_width import peak_fwhm, peak_fwhm_2d, spectrum_peak_fwhm, spectrum_peak_fwhm_2d
from .fit_ranges import parse_ppm_ranges
from .cp_buildup import (
    contact_time_us, cp_series, fit_cp_buildup, cp_report, plot_cp_buildup,
    cp_series_regions, plot_cp_buildups,
)
# pNMR (CIF) isotropic peaks -> Solid-state NMR sites (2026-10-12).
from . import pnmr_export
from .pnmr_export import sites_from_pnmr_fit, sites_from_pnmr_table
from .progress import (
    processing_log, processing_history, list_save_points, save_point_state,
    save_progress, load_progress, log_event, load_saved_spectrum,
    saved_fits, simulation_log, simulation_history, log_simulation,
)

from .csa_fitting import (
    csa_model_full, build_csa_model, fit_csa_spectrum, evaluate_csa_fit,
    print_csa_fit_report, NMR1dplot_with_csa_fit, NMRCSAFitWorkflow
)

# A CSA / quadrupolar / Solid-state NMR fit of one slice, applied to
# every slice of a pseudo-2D series (T1, T2...) -- see series_fitting.
from .series_fitting import (
    model_components, fit_series_with_lineshapes, series_fit_report,
)

from .csa_relaxation import (
    compute_csa_integrals, compute_csa_integral_errors,
    NMR2D_fit_csa_slices, NMR2D_csa_relaxation_workflow,
)

from .pnmr_simulator import (
    read_cif_with_occupancies, get_neighbors_by_shell,
    local_environment_distribution, combine_labeled_shells, enumerate_local_environments,
    build_shift_table, _broadened_spectrum, generate_spectrum_dict,
    shell_distance_histogram,
    mark_sites_of_interest, species_element, candidate_nuclides, structure_species,
    site_label_table, contributing_species, neighbour_distance_table, shell_composition,
    validate_shell_bins, site_weights, shift_table_from_structure, suggest_shell_bins,
    estimate_configurations, simulation_parameters, save_simulation_parameters,
    load_simulation_parameters, EnumerationCancelled, TooManyConfigurations, PnmrWarning,
)

from .pnmr_fit import (
    predict_environments, default_tolerance, group_environments, 
    select_groups, groups_to_peak_defs, split_integrals_by_environment, 
    print_environment_report, dominant_environment, format_environment_label, 
    add_dominant_environment_labels, fit_predicted_spectrum, 
    merge_low_integral_peaks, merge_groups_to_picks, print_shift_drift_report, 
    refit_merged_groups, fit_cif_envelope,
    collapse_members, peak_weights_at_shift, peak_index_at_shift,
    environments_at_shift, format_environments, summarise_peaks, add_predicted_sticks,
)

from .simpson import (
    SIMPSONimport, CSDMimport, SIMPSONread_shape, SIMPSONapodization_window, SIMPSONread_fid,
    SIMPSONplot_fid, SIMPSONread_single_point_fid, SIMPSONread_fid_series, SIMPSONplot_1d_stack, SIMPSONplot_shape,
    SIMPSONplot_fid_series, SIMPSONload, SIMPSONread_table, SIMPSONread_script, SIMPSONplot_table,
    SIMPSONshape_profile, SIMPSONshape_duration_from_name, SIMPSONdescribe_shape,
)
from .simpson_series import (
    SIMPSONread_points, SIMPSONpoint_series, SIMPSONcombine_points, SIMPSONpoint_table,
    SIMPSONsave_point_table, SIMPSONplot_point_series, x_from_filenames,
    SIMPSONread_curve, SIMPSONcurve, SIMPSONplot_curve, SIMPSONcurve_table, SIMPSONsave_curve,
    stepped_values, SIMPSONread_scan, SIMPSONscan_from_table, SIMPSONscan_from_files,
    SIMPSONscan_table, SIMPSONplot_heatmap, scan_grid,
)
from .simpson_fit import fit_simpson_templates, SIMPSONfit_spectrum, SIMPSONplot_fit
from ..io.export import export_spectrum, read_jcamp_spectrum
from ..io.load_cache import read as load_cached

from .jcoupling_fitting import (
    jcoupling_site_lineshape, build_jcoupling_model, initialise_jcoupling_params, build_jcoupling_bounds,
    fit_jcoupling_spectrum, evaluate_jcoupling_fit, print_jcoupling_fit_report, NMR1dplot_with_jcoupling_fit,
    NMRJCouplingFitWorkflow, jcoupling_chemist_notation,
)

# NOTE: compute_BF1_SFO1, plot_spectrum_dict and plot_comparison used to be
# imported from here -- they now live in .utils / .plotting (single copy,
# shared with the BMS pipeline below) and are imported from there, above.
# pnmr_simulator.py still re-exports them itself for anyone importing
# `from nmrtools.pnmr_simulator import ...` directly, but nmr.compute_BF1_SFO1
# etc. below now resolve to the .utils/.plotting names, not pnmr_simulator's
# re-export -- same object either way, just naming the canonical source here.

from .diffusion import recorded_b_values, recorded_pfg_params
from .bms_spectrum import generate_bms_spectrum_dict
from .bms.io_utils import make_particle

# ---------------------------------------------------------------------------
# One-import convenience (2026-10-08): everything a notebook used to pull in
# from half a dozen sub-packages is reachable from `import opennmr.notebook
# as nmr` alone -- nmr.load(...), nmr.expand_pcov(...), nmr.pt.<...>, etc.
# The short module aliases match the names the tutorial notebooks used
# (pt, pf, dw, acc, bloch), so `pt.foo(...)` becomes `nmr.pt.foo(...)`, or
# `from opennmr.notebook import *` restores the bare names exactly.
# ---------------------------------------------------------------------------
from ..io import load
from .bridge import results_from_spectrum
from .fitting import expand_pcov
from .utils import find_isotope_info
from ..plotting.accessibility import ordered_qualitative_colours
from ..plotting import accessibility
from ..simulate import bloch
from . import periodic_table, pnmr_fit, bms
from .bms import run_bms, particles_to_array, build_box, fast_bms_spectrum

# Fitting limits and sideband guidance (2026-10-08): the app-wide maximum
# number of iterations (2000 to start with) and the 3-5 sideband rule.
from ..core.fit_progress import (
    DEFAULT_MAX_ITERATIONS, FitNotConverged, iteration_limit, max_iterations, set_max_iterations,
)
# Fitted values as readable JSON (2026-10-08, round 3).
from . import fit_values
from .fit_values import load_fitted_values, save_fitted_values
from .fit_advice import (
    GOOD_SIDEBANDS, MIN_SIDEBANDS, SIDEBAND_GUIDANCE, csa_span_ppm, sideband_count, sideband_note,
)

# Shearing and scaling of 2D spectra (MQMAS, STMAS, DAS, J-resolved,
# PASS/MAT; 2026-10-11).
from . import shearing
from .shearing import (
    make_shear_plan, shear_2d, estimate_shear, sideband_positions, isotropic_from_peak,
    plot_shear_scan,
)

# Offset Tracker (Integrate ▸ Offset Tracker; 2026-10-17, lineshapes
# 2026-10-18): integrals of an HMQC / J-resolved series against O2/O3, and
# what is drawn (bars, lines, peaks at the offsets) kept as a spectrum.
from . import offset_tracker
from .offset_tracker import (
    offset_series, offset_report, plot_offset_series, skyline_spectrum, sideband_pattern,
    sideband_spectrum, stick_spectrum, pattern_spectrum,
    offset_series_regions, plot_offset_series_regions,
)
# Typefaces for figures (2026-10-17).
from . import fonts
from .fonts import available_fonts, font_samples, plot_font_samples, use_font
from ..plotting.vector_export import save_vector_safe

pt = periodic_table
pf = pnmr_fit
dw = dnp_workflow
acc = accessibility


from .experiment_notebook import (
    experiment_notebook, write_experiment_notebook, export_experiment_notebook,
)

__all__ = [
    "add_inset", "remove_insets", "insets_of", "INSET_DEFAULTS", "INSET_CORNERS",
    "integral_labels", "add_integral_labels", "remove_integral_labels", "normalised_integrals",
    "integral_reference_region", "attenuation", "attenuation_values", "ATTENUATION_MODES",
    "score_kernel_model", "residual_metrics",
    # utils
    "find_gamma", "NUC_label", "read_bruker_metadata", "read_bruker_fid",
    "read_vdlist", "compute_BF1_SFO1",
    # plotting
    "setup_plotting", "get_stack_colormap", "get_contour_cmap", "downsample_2d",
    "apply_tick_style", "apply_spine_style", "save_figure",
    "plot_spectrum_dict", "plot_comparison", "plot_stacked_spectra", "plot_quadrupolar_orders",
    # colour accessibility
    "friendly_palettes", "max_safe_series", "describe_accessibility", "palette_distinctness",
    "is_colourblind_safe", "is_print_safe", "trace_colour_presets", "series_colours",
    # one_d
    "NMR1dimport", "NMR1dplot", "NMR1dheatmap",
    "vocs_f1", "vocs_f2", "vocs_power", "pulse_to_w1", "apply_VOCS", "VOCSSum",
    "CPMGprocess", "CPMGplot", "cpmg_echo_layout", "cpmg_to_spectrum",
    # compare_panels
    "plot_compare_panels", "NMR1D_multi_plot", "add_panel_labels",
    # two_d
    "NMR2dimport", "NMR2Dplotter", "NMR2D_multi_plot", "NMR2D_overlay_plot",
    "NMR2d_get_trace", "NMR2D_get_slices", "prepare_trace_for_fit",
    "NMR2D_plot_slices", "NMR2D_summary_plot", "canonical_nucleus", "check_external_projection",
    # diffusion: where a PFG series' b-values and timings come from
    "recorded_b_values", "recorded_pfg_params",
    # fitting
    "gaussian", "lorentzian", "pseudo_voigt",
    "estimate_baseline_from_edges", "estimate_baseline_robust",
    "fit_spectrum", "evaluate_fit", "compute_integrals", "compute_integral_errors",
    "fit_quality", "print_fit_report", "NMR1dplot_with_fit", "NMRFitWorkflow",
    "estimate_noise_level", "select_2d_traces",
    "NMR2D_fit_slices", "NMR2D_print_fit_report", "NMR2D_plot_integrals",
    "NMR2D_workflow",
    # relaxation
    "saturation_recovery", "inversion_recovery", "exponential_decay",
    "biexponential_decay", "biexponential_recovery",
    "triexponential_decay", "triexponential_recovery",
    "stretched_decay", "stretched_recovery", "RELAXATION_MODELS",
    "relaxometry", "find_integration_regions", "correlation_matrix",
    "dft_nmr", "read_calculation", "dft_sites", "simulate_dft", "plot_dft_tensors",
    "dft_structure", "plot_dft_structure",
    "dft_calibration", "fit_calibration", "reference_point", "plot_calibration",
    "paramagnetic_nmr", "pseudocontact", "lanthanides", "bond_pathways",
    "isotopes", "resolve_links", "frequency_ratio", "to_observed_ppm", "isotope_partners",
    "partner_factors",
    "plot_correlation_matrix", "fit_correlation", "correlation_of", "multi_fit_correlation",
    "fit_relaxation", "print_relaxation_report",
    "NMR2D_fit_relaxation", "NMR2D_plot_relaxation",
    "NMR2D_relaxation_workflow",
    # arithmetic
    "combine_spectra", "combine_spectra_as_results",
    # manual baseline
    "fit_manual_baseline", "evaluate_manual_baseline", "manual_baseline_correct",
    "plot_manual_baseline", "describe_manual_baseline", "polynomial_baseline", "sine_baseline",
    "MANUAL_BASELINE_PARAMS", "MANUAL_BASELINE_EQUATIONS",
    # Czjzek distributions
    "czjzek_pdf", "czjzek_statistics", "czjzek_spectrum", "fit_czjzek_spectrum",
    "evaluate_czjzek_fit", "czjzek_fit_report", "plot_czjzek_distribution", "plot_czjzek_fit",
    "CzjzekLibrary", "czjzek_samples", "extended_czjzek_weights",
    # fit display
    "draw_fit_components", "component_style",
    # kept fit models and openNMR states
    "make_fit_model", "fit_statistics", "compare_fit_models", "format_comparison",
    "plot_fit_comparison", "draw_fit_model", "fit_view", "save_state", "load_state", "describe_state", "saved_fit_models",
    # simulations
    "haeberlen_csa_tensor", "herzfeld_berger_csa_tensor", "csa_definition",
    "csa_frequency", "powder_orientations", "powder_orientations_octant", 
    "static_csa", "make_static_csa_spectrum", "finite_mas_csa_spectrum",
    "quadrupolar_static_spectrum", "quadrupolar_mas_spectrum", "quadrupolar_spectrum",
    "quadrupolar_order_comparison", "quadrupolar_transition_frequencies",
    "quadrupolar_transition_table", "quadrupolar_axis_limits", "quadrupolar_nu_Q",
    "quadrupolar_ct_isotropic_shift_Hz", "larmor_frequency_Hz", "zcw_angles", "lebedev_angles", "powder_grid_size",
    "spin_multiplet_pattern", "jcoupling_group_pattern", "combine_jcouplings",
    "multiplet_lineshape", "jcoupled_spectrum", "multi_site_jcoupled_spectrum", "element_isotopes",
    # quadrupolar fitting
    "fit_quadrupolar_spectrum", "evaluate_quad_fit", "estimate_quadrupolar_start",
    "quadrupolar_derived", "print_quadrupolar_fit_report", "NMR1dplot_with_quad_fit",
    "NMRQuadFitWorkflow",
    # combined Hamiltonian
    "Site", "simulate_combined", "combined_axis_limits", "combined_to_python", "copy_interactions",
    "site_transition_frequencies", "plot_combined", "sites_to_json", "sites_from_json",
    # combined-Hamiltonian fitting
    "fit_combined", "fit_combined_multistart", "print_combined_fit_report",
    "plot_combined_fit", "NMRCombinedFitWorkflow", "site_parameter_names",
    # spin_system (strong coupling)
    "expand_spin_system", "block_sizes", "spin_system_transitions", "spin_system_spectrum",
    "coupling_strengths", "fit_spin_system_spectrum", "evaluate_spin_system_fit",
    "fitted_system", "equivalent_solutions",
    # mixed_j (weak and strong J on one spectrum)
    "mixed_j_spectrum", "split_spin_system",
    # csa_fitting
    "csa_model_full", "build_csa_model", "fit_csa_spectrum", "evaluate_csa_fit",
    "print_csa_fit_report", "NMR1dplot_with_csa_fit", "NMRCSAFitWorkflow",
    # csa_relaxation
    "compute_csa_integrals", "compute_csa_integral_errors",
    "NMR2D_fit_csa_slices", "NMR2D_csa_relaxation_workflow",
    # pnmr_simulator
    "read_cif_with_occupancies", "get_neighbors_by_shell",
    "local_environment_distribution", "combine_labeled_shells", "enumerate_local_environments",
    "build_shift_table", "_broadened_spectrum", "generate_spectrum_dict",
    "shell_distance_histogram",
    "mark_sites_of_interest", "species_element", "candidate_nuclides", "structure_species",
    "site_label_table", "contributing_species", "neighbour_distance_table", "shell_composition",
    "validate_shell_bins", "site_weights", "shift_table_from_structure", "suggest_shell_bins",
    "estimate_configurations", "simulation_parameters", "save_simulation_parameters",
    "load_simulation_parameters", "EnumerationCancelled", "TooManyConfigurations", "PnmrWarning",
    # bms
    "generate_bms_spectrum_dict", "make_particle",
    # pnmr_fit
    "predict_environments", "default_tolerance", "group_environments", 
    "select_groups", "groups_to_peak_defs", "split_integrals_by_environment", 
    "print_environment_report", "dominant_environment", 
    "format_environment_label", "add_dominant_environment_labels", 
    "fit_predicted_spectrum", "merge_low_integral_peaks", 
    "merge_groups_to_picks", "print_shift_drift_report", 
    "refit_merged_groups", "fit_cif_envelope",
    "collapse_members", "peak_weights_at_shift", "peak_index_at_shift",
    "environments_at_shift", "format_environments", "summarise_peaks", "add_predicted_sticks",
    # simpson
    "SIMPSONimport", "CSDMimport", "SIMPSONread_shape", "SIMPSONapodization_window", "SIMPSONread_fid",
    "SIMPSONplot_fid", "SIMPSONread_single_point_fid", "SIMPSONread_fid_series", "SIMPSONplot_1d_stack", "SIMPSONplot_shape",
    "SIMPSONplot_fid_series", "SIMPSONload", "SIMPSONread_table", "SIMPSONread_script",
    "SIMPSONplot_table",
    "SIMPSONread_points", "SIMPSONpoint_series", "SIMPSONcombine_points", "SIMPSONpoint_table",
    "SIMPSONsave_point_table", "SIMPSONplot_point_series", "x_from_filenames",
    "SIMPSONshape_profile", "SIMPSONshape_duration_from_name", "SIMPSONdescribe_shape",
    "SIMPSONread_curve", "SIMPSONcurve", "SIMPSONplot_curve", "SIMPSONcurve_table",
    "SIMPSONsave_curve", "stepped_values", "SIMPSONread_scan", "SIMPSONscan_from_table",
    "SIMPSONscan_from_files", "SIMPSONscan_table", "SIMPSONplot_heatmap", "scan_grid",
    "fit_simpson_templates", "SIMPSONfit_spectrum", "SIMPSONplot_fit",
    # export (any vendor -> .csv, .xy, JCAMP-DX)
    "export_spectrum", "read_jcamp_spectrum", "load_cached",
    # jcoupling_fitting
    "jcoupling_site_lineshape", "build_jcoupling_model", "initialise_jcoupling_params", "build_jcoupling_bounds",
    "fit_jcoupling_spectrum", "evaluate_jcoupling_fit", "print_jcoupling_fit_report", "NMR1dplot_with_jcoupling_fit",
    "NMRJCouplingFitWorkflow", "jcoupling_chemist_notation",
    # notes
    "experiment_notes", "set_experiment_notes", "notes_location", "notes_first_line",
    # stored_integrals (integrals kept across spectra and sessions)
    "StoredIntegrals", "get_stored_integrals",
    # multi_fitting (one set of sites across several spectra)
    "make_experiment", "fit_multi", "default_scopes", "multi_fit_table", "multi_fit_to_csv",
    "parameter_trend", "print_multi_fit_report", "plot_multi_fit", "plot_parameter_trend",
    "MultiFitCancelled",
    # progress (processing log and save points, shared with the GUI)
    "processing_log", "processing_history", "list_save_points", "save_point_state",
    "save_progress", "load_progress", "log_event", "load_saved_spectrum",
    "saved_fits", "simulation_log", "simulation_history", "log_simulation",
    # DNP
    "carrier_frequency_mhz", "carrier_offset_hz", "field_from_frequency", "field_from_sfo1",
    "peak_height", "region_integral", "enhancement", "enhancement_report",
    "print_enhancement_report", "field_sweep", "field_sweep_table", "field_sweep_to_csv",
    "field_sweep_report", "print_field_sweep_report", "plot_field_profile", "plot_sweep_spectra",
    "plot_enhancement_spectra",
    # signal-to-noise; pNMR peaks -> Solid-state NMR sites
    "snr", "signal_to_noise", "signal_to_noise_many", "snr_report", "plot_snr",
    "ringing", "check_f2_ringing",
    "pnmr_export", "sites_from_pnmr_fit", "sites_from_pnmr_table",
    # comparison scaling, peak widths, fit ranges, CP build-up (2026-10-16)
    "intensity_scaling", "scale_for_comparison", "peak_width", "peak_fwhm", "peak_fwhm_2d",
    "spectrum_peak_fwhm", "spectrum_peak_fwhm_2d", "fit_ranges", "parse_ppm_ranges",
    "cp_buildup", "contact_time_us", "cp_series", "fit_cp_buildup", "cp_report",
    "plot_cp_buildup", "cp_series_regions", "plot_cp_buildups",
    # tensor pictures
    "frequency_surface", "tensor_components", "transition_labels", "plot_tensors",
    "plot_tensor_surface", "plot_tensor_ellipsoids",
    # Bloch simulator (opennmr.simulate.bloch)
    "simulate_pulse", "adiabatic_shape", "adiabaticity", "peak_b1_for_adiabaticity",
    "peak_b1_for_flip_angle", "inverted_band_hz", "plot_pulse_simulation", "ADIABATIC_SHAPES",
    # dynamics: relaxation vs temperature and exchange lineshapes
    "vt_dataset", "load_vt_csv", "fit_vt_relaxation", "compare_vt_models", "fit_arrhenius",
    "hendrickson_bray", "fit_hendrickson_bray", "waugh_fedin_Ea", "plot_arrhenius",
    "plot_vs_temperature", "plot_hendrickson_bray", "fit_vt_regions", "plot_arrhenius_regions",
    "exchange_spectrum", "exchange_spectra_vs_temperature", "fit_exchange_spectra",
    "plot_exchange_stack", "plot_exchange_fit", "coalescence_temperature",
    # dynamics: Tikhonov regularisation (distributions and 2D maps)
    "regularisation", "regularise", "regularise_2d", "plot_regularisation",
    "plot_regularisation_2d", "static_gradient_b", "regularisation_subsample",
    "regularisation_grid_range",
    "regularisation_report", "regularisation_to_csv",
    # one-import convenience: readers, helpers and module aliases
    "load", "results_from_spectrum", "expand_pcov", "find_isotope_info",
    "ordered_qualitative_colours", "run_bms", "particles_to_array", "build_box",
    "periodic_table", "pt", "pnmr_fit", "pf", "dnp_workflow", "dw",
    "accessibility", "acc", "bloch", "bms",
    # fitting limits and sideband guidance
    "DEFAULT_MAX_ITERATIONS", "FitNotConverged", "iteration_limit", "max_iterations",
    "set_max_iterations", "GOOD_SIDEBANDS", "MIN_SIDEBANDS", "SIDEBAND_GUIDANCE",
    "csa_span_ppm", "sideband_count", "sideband_note",
    # fitted values as readable JSON
    "fit_values", "save_fitted_values", "load_fitted_values",
    # shearing and scaling of 2D spectra
    "shearing", "make_shear_plan", "shear_2d", "estimate_shear", "sideband_positions",
    "isotropic_from_peak", "plot_shear_scan",
    # Offset Tracker (HMQC / J-resolved series against O2/O3)
    "offset_tracker", "offset_series", "offset_report", "plot_offset_series", "skyline_spectrum",
    "sideband_pattern", "sideband_spectrum", "stick_spectrum", "pattern_spectrum",
    "offset_series_regions", "plot_offset_series_regions",
    # typefaces and Illustrator-safe vector saves
    "fonts", "available_fonts", "font_samples", "plot_font_samples", "use_font",
    "save_vector_safe",
    # one experiment as a Jupyter notebook
    "experiment_notebook", "write_experiment_notebook", "export_experiment_notebook",
]
