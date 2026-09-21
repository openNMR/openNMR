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
    find_gamma, NUC_label, read_bruker_metadata, read_vdlist, compute_BF1_SFO1,
)

from .plotting import (
    setup_plotting, get_stack_colormap, get_contour_cmap, downsample_2d,
    apply_tick_style, apply_spine_style, save_figure,
    plot_spectrum_dict, plot_comparison, plot_stacked_spectra,
)

from .one_d import (
    NMR1dimport, NMR1dplot, NMR1dheatmap,
    vocs_f1, vocs_f2, vocs_power, pulse_to_w1, apply_VOCS, VOCSSum,
)

from .two_d import (
    NMR2dimport, NMR2Dplotter, NMR2D_multi_plot, NMR2D_overlay_plot,
    NMR2d_get_trace, NMR2D_get_slices, prepare_trace_for_fit,
    NMR2D_plot_slices, NMR2D_summary_plot,
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

from .simulations import (
    haeberlen_csa_tensor, herzfeld_berger_csa_tensor, csa_definition,
    csa_frequency, powder_orientations, powder_orientations_octant, 
    static_csa, make_static_csa_spectrum, finite_mas_csa_spectrum,
    static_quadrupolar_spectrum_exact
)

from .csa_fitting import (
    csa_model_full, build_csa_model, fit_csa_spectrum, evaluate_csa_fit,
    print_csa_fit_report, NMR1dplot_with_csa_fit, NMRCSAFitWorkflow
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

# NOTE: compute_BF1_SFO1, plot_spectrum_dict and plot_comparison used to be
# imported from here -- they now live in .utils / .plotting (single copy,
# shared with the BMS pipeline below) and are imported from there, above.
# pnmr_simulator.py still re-exports them itself for anyone importing
# `from nmrtools.pnmr_simulator import ...` directly, but nmr.compute_BF1_SFO1
# etc. below now resolve to the .utils/.plotting names, not pnmr_simulator's
# re-export -- same object either way, just naming the canonical source here.

from .bms_spectrum import generate_bms_spectrum_dict
from .bms.io_utils import make_particle

__all__ = [
    # utils
    "find_gamma", "NUC_label", "read_bruker_metadata", "read_vdlist", "compute_BF1_SFO1",
    # plotting
    "setup_plotting", "get_stack_colormap", "get_contour_cmap", "downsample_2d",
    "apply_tick_style", "apply_spine_style", "save_figure",
    "plot_spectrum_dict", "plot_comparison", "plot_stacked_spectra",
    # one_d
    "NMR1dimport", "NMR1dplot", "NMR1dheatmap",
    "vocs_f1", "vocs_f2", "vocs_power", "pulse_to_w1", "apply_VOCS", "VOCSSum",
    # two_d
    "NMR2dimport", "NMR2Dplotter", "NMR2D_multi_plot", "NMR2D_overlay_plot",
    "NMR2d_get_trace", "NMR2D_get_slices", "prepare_trace_for_fit",
    "NMR2D_plot_slices", "NMR2D_summary_plot",
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
    # simulations
    "haeberlen_csa_tensor", "herzfeld_berger_csa_tensor", "csa_definition",
    "csa_frequency", "powder_orientations", "powder_orientations_octant", 
    "static_csa", "make_static_csa_spectrum", "finite_mas_csa_spectrum",
    "static_quadrupolar_spectrum_exact",
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
    "SIMPSONimport", "CSDMimport", "SIMPSONread_shape", "SIMPSONapodization_window", "SIMPSONread_fid"
    "SIMPSONplot_fid", "SIMPSONread_single_point_fid", "SIMPSONread_fid_series", "SIMPSONplot_1d_stack", "SIMPSONplot_shape",
    "SIMPSONplot_fid_series",
]
