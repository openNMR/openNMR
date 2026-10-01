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
    stretched_decay, stretched_recovery, RELAXATION_MODELS,
    fit_relaxation, format_time_value, print_relaxation_report,
    NMR2D_integrate_slices,
    NMR2D_fit_relaxation, NMR2D_plot_relaxation,
    NMR2D_relaxation_workflow,
)

from .arithmetic import combine_spectra, combine_spectra_as_results
# Referencing across nuclei, and linear field-drift correction
# (Najbauer & Andreas, J. Magn. Reson. 305 (2019) 1-4) -- Process ▸
# Reference and Process ▸ Field Drift Correction in the GUI.
from .referencing import transfer_sr, transfer_sr_between
from .drift_correction import (
    correct_bruker_dataset, concatenate_nus_datasets, measure_drift_hz, drift_profile,
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
    quadrupolar_ct_isotropic_shift_Hz, larmor_frequency_Hz, zcw_angles,
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
    Site, simulate_combined, combined_axis_limits, combined_to_python,
    site_transition_frequencies, plot_combined, sites_to_json, sites_from_json,
)

from .combined_fitting import (
    fit_combined, fit_combined_multistart, print_combined_fit_report, plot_combined_fit,
    NMRCombinedFitWorkflow, site_parameter_names,
)

from .spin_system import (
    expand_spin_system, block_sizes, spin_system_transitions, spin_system_spectrum,
    coupling_strengths, fit_spin_system_spectrum, evaluate_spin_system_fit, fitted_system,
    equivalent_solutions,
)

from .notes import experiment_notes, set_experiment_notes, notes_location, notes_first_line
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
from .progress import (
    processing_log, processing_history, list_save_points, save_point_state,
    save_progress, load_progress, log_event, load_saved_spectrum,
    saved_fits, simulation_log, simulation_history, log_simulation,
)

from .csa_fitting import (
    csa_model_full, build_csa_model, fit_csa_spectrum, evaluate_csa_fit,
    print_csa_fit_report, NMR1dplot_with_csa_fit, NMRCSAFitWorkflow
)

# A CSA / quadrupolar / Combined-Hamiltonian fit of one slice, applied to
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
)

from .simpson import (
    SIMPSONimport, CSDMimport, SIMPSONread_shape, SIMPSONapodization_window, SIMPSONread_fid,
    SIMPSONplot_fid, SIMPSONread_single_point_fid, SIMPSONread_fid_series, SIMPSONplot_1d_stack, SIMPSONplot_shape,
    SIMPSONplot_fid_series,
)

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

__all__ = [
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
    "stretched_decay", "stretched_recovery", "RELAXATION_MODELS",
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
    "quadrupolar_ct_isotropic_shift_Hz", "larmor_frequency_Hz", "zcw_angles",
    "spin_multiplet_pattern", "jcoupling_group_pattern", "combine_jcouplings",
    "multiplet_lineshape", "jcoupled_spectrum", "multi_site_jcoupled_spectrum", "element_isotopes",
    # quadrupolar fitting
    "fit_quadrupolar_spectrum", "evaluate_quad_fit", "estimate_quadrupolar_start",
    "quadrupolar_derived", "print_quadrupolar_fit_report", "NMR1dplot_with_quad_fit",
    "NMRQuadFitWorkflow",
    # combined Hamiltonian
    "Site", "simulate_combined", "combined_axis_limits", "combined_to_python",
    "site_transition_frequencies", "plot_combined", "sites_to_json", "sites_from_json",
    # combined-Hamiltonian fitting
    "fit_combined", "fit_combined_multistart", "print_combined_fit_report",
    "plot_combined_fit", "NMRCombinedFitWorkflow", "site_parameter_names",
    # spin_system (strong coupling)
    "expand_spin_system", "block_sizes", "spin_system_transitions", "spin_system_spectrum",
    "coupling_strengths", "fit_spin_system_spectrum", "evaluate_spin_system_fit",
    "fitted_system", "equivalent_solutions",
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
    # simpson
    "SIMPSONimport", "CSDMimport", "SIMPSONread_shape", "SIMPSONapodization_window", "SIMPSONread_fid",
    "SIMPSONplot_fid", "SIMPSONread_single_point_fid", "SIMPSONread_fid_series", "SIMPSONplot_1d_stack", "SIMPSONplot_shape",
    "SIMPSONplot_fid_series",
    # jcoupling_fitting
    "jcoupling_site_lineshape", "build_jcoupling_model", "initialise_jcoupling_params", "build_jcoupling_bounds",
    "fit_jcoupling_spectrum", "evaluate_jcoupling_fit", "print_jcoupling_fit_report", "NMR1dplot_with_jcoupling_fit",
    "NMRJCouplingFitWorkflow", "jcoupling_chemist_notation",
    # notes
    "experiment_notes", "set_experiment_notes", "notes_location", "notes_first_line",
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
    # tensor pictures
    "frequency_surface", "tensor_components", "transition_labels", "plot_tensors",
    "plot_tensor_surface", "plot_tensor_ellipsoids",
    # Bloch simulator (opennmr.simulate.bloch)
    "simulate_pulse", "adiabatic_shape", "adiabaticity", "peak_b1_for_adiabaticity",
    "peak_b1_for_flip_angle", "inverted_band_hz", "plot_pulse_simulation", "ADIABATIC_SHAPES",
]
