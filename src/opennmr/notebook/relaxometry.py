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

import re

import numpy as np

from ..processing.relaxometry import (  # noqa: F401 -- re-exported
    INTEGRATION_METHODS, TRIEXP_MIN_POINTS, correlation_matrix, find_integration_regions,
    integrate, integration_weights, multiexponential_start, sort_components,
    strong_correlations, triexponential_decay, triexponential_recovery,
)

#: Blue (negative) to red (positive) through a neutral light grey: two
#: poles that read as opposite, and a midpoint that reads as "nothing".
DIVERGING = ("#2166ac", "#f0efec", "#b2182b")

_GREEK = {"beta": r"\beta", "tau": r"\tau", "tau_c": r"\tau_\mathrm{c}"}


def parameter_label(name: str) -> str:
    """A fit parameter's name as an IUPAC-style mathtext label: symbol
    italic, subscripts upright -- ``T2a`` → *T*₂ₐ, ``A1`` → *A*₁, ``M0``
    → *M*₀, ``beta`` → β. Unrecognised names are returned as they are."""
    if name in _GREEK:
        return f"${_GREEK[name]}$"
    m = re.fullmatch(r"(T1rho)", name)
    if m:
        return r"$\mathit{T}_{1\rho}$"
    m = re.fullmatch(r"([A-Za-z])(\d*)([a-z]?)", name)
    if not m:
        return name
    sym, num, sub = m.groups()
    tail = num + sub
    return f"$\\mathit{{{sym}}}_{{\\mathrm{{{tail}}}}}$" if tail else f"$\\mathit{{{sym}}}$"


def plot_correlation_matrix(fit, figure=None, ax=None, triangle="lower", annotate=True,
                            fontsize=12.0, title=None, colorbar=True, cmap=None):
    """The correlation matrix of a fit's free parameters as a heat map:
    ``r`` from −1 (blue) through 0 (grey) to +1 (red), with each value
    written in its cell.

    ``fit`` : any fit dict with ``free_names`` and ``pcov`` -- a
    :func:`~opennmr.notebook.relaxation.fit_relaxation` result, one entry
    of ``NMR2D_fit_relaxation``'s list, or anything shaped like them.
    ``figure`` / ``ax`` : draw into this figure (cleared) or these axes;
    otherwise a new figure.
    ``triangle`` : ``"lower"`` (each pair once, with the diagonal) or
    ``"full"``.
    ``title`` : defaults to the fit's ``peak_label``.

    |r| close to 1 means two parameters trade off against each other: the
    data fix a combination of them, not each one, and their separate
    errors understate how loosely they are pinned. A parameter with no
    determinable variance is drawn hatched (``nan``).

    Returns the figure; its axes are ``figure.correlation_axes`` =
    ``{"matrix", "colorbar"}`` and the numbers ``figure.correlation`` =
    ``(names, R)``.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    from ..plotting.style import apply_tick_style

    names, R = correlation_matrix(fit)
    n = len(names)
    if triangle not in ("lower", "full"):
        raise ValueError("triangle must be 'lower' or 'full'.")
    if ax is None:
        if figure is None:
            side = max(3.0, 0.55 * n + 1.6)
            figure = plt.figure(figsize=(side + 0.9, side))
        else:
            figure.clear()
        grid = figure.add_gridspec(1, 2 if colorbar else 1,
                                   width_ratios=[1.0, 0.05] if colorbar else None)
        ax = figure.add_subplot(grid[0, 0])
        cax = figure.add_subplot(grid[0, 1]) if colorbar else None
    else:
        figure = ax.figure
        cax = None
    cmap = cmap or LinearSegmentedColormap.from_list("correlation", DIVERGING)
    shown = R.copy()
    if triangle == "lower":
        shown[np.triu_indices(n, 1)] = np.nan
    # The diagonal is 1 by definition: neutral, so the eye goes to the pairs.
    np.fill_diagonal(shown, np.nan)
    masked = np.ma.masked_invalid(shown)
    face = figure.get_facecolor()
    mesh = ax.pcolormesh(np.arange(n + 1), np.arange(n + 1), masked, cmap=cmap,
                         vmin=-1.0, vmax=1.0, edgecolors=face, linewidth=1.5)
    for i in range(n):
        ax.add_patch(plt.Rectangle((i, i), 1, 1, facecolor="0.88", edgecolor=face,
                                   linewidth=1.5))
    # Undeterminable cells (not the hidden upper triangle) are hatched.
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if triangle == "lower" and j > i:
                continue
            if not np.isfinite(R[i, j]):
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, hatch="///",
                                           edgecolor="0.6", linewidth=0))
    ax.set_xlim(0, n)
    ax.set_ylim(n, 0)
    ax.set_aspect("equal")
    labels = [parameter_label(name) for name in names]
    ax.set_xticks(np.arange(n) + 0.5)
    ax.set_yticks(np.arange(n) + 0.5)
    ax.set_xticklabels(labels, fontsize=fontsize)
    ax.set_yticklabels(labels, fontsize=fontsize)
    ax.tick_params(which="both", length=0)
    ax.minorticks_off()
    ax._opennmr_categorical = True        # the GUI's style pass leaves it so
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)
    if annotate:
        size = fontsize * (0.8 if n <= 6 else 0.65)
        for i in range(n):
            for j in range(n):
                if triangle == "lower" and j > i:
                    continue
                r = R[i, j]
                if not np.isfinite(r):
                    continue
                ink = "white" if abs(r) > 0.6 and i != j else "0.15"
                text = "1" if i == j else f"{r:+.2f}".replace("-", "\u2212")
                ax.text(j + 0.5, i + 0.5, text, ha="center", va="center", fontsize=size,
                        color=ink)
    title = fit.get("peak_label") if title is None else title
    if title:
        ax.set_title(str(title), fontsize=fontsize)
    if cax is not None:
        bar = figure.colorbar(mesh, cax=cax, ticks=[-1, -0.5, 0, 0.5, 1])
        bar.set_label(r"Correlation coefficient, $\mathit{r}$", fontsize=fontsize * 0.9)
        bar.outline.set_visible(False)
        apply_tick_style(cax, labelsize=fontsize * 0.85)
    figure.correlation_axes = {"matrix": ax, "colorbar": cax}
    figure.correlation = (names, R)
    return figure
