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

from .utils import find_gamma, NUC_label, read_bruker_metadata, read_vdlist

from .plotting import (
    setup_plotting, get_stack_colormap, get_contour_cmap, downsample_2d,
    apply_tick_style, apply_spine_style, save_figure,
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

__all__ = [
    # utils
    "find_gamma", "NUC_label", "read_bruker_metadata", "read_vdlist",
    # plotting
    "setup_plotting", "get_stack_colormap", "get_contour_cmap", "downsample_2d",
    "apply_tick_style", "apply_spine_style", "save_figure",
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
]
