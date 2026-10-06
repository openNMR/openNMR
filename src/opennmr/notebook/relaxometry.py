"""
Relaxometry extras for notebooks (explicit user request, 2026-10-06): the
three-exponential models, threshold-found integration regions, trapezoid or
Simpson integration, and the correlation-matrix figure of a fit.

The maths lives in :mod:`opennmr.processing.relaxometry` (so the GUI's and
the notebook's fitters share it); this module re-exports it and adds the
figure.

    from opennmr.notebook import relaxation as rl, relaxometry as rx
    out = rl.NMR2D_relaxation_workflow(path, rows=range(1, 17),
                                       integration="regions", regions="auto",
                                       region_threshold=8, model="triexponential_decay")
    fit = out["relaxation_fits"][0]
    rx.plot_correlation_matrix(fit)
"""

from __future__ import annotations

from ..processing.relaxometry import (  # noqa: F401 -- re-exported
    INTEGRATION_METHODS, TRIEXP_MIN_POINTS, correlation_matrix, find_integration_regions,
    integrate, integration_weights, multiexponential_start, sort_components,
    strong_correlations, triexponential_decay, triexponential_recovery,
)

# The figure is shared with every other kind of fit (explicit user request,
# 2026-10-06): opennmr.notebook.fit_correlation.
from .fit_correlation import (  # noqa: F401,E402 -- re-exported
    DIVERGING, correlation_input, correlation_of, parameter_label, plot_correlation_matrix,
)
