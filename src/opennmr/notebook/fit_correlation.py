"""
The correlation matrix of any fit's parameters, as numbers and as a
publication-ready figure (explicit user requests, 2026-10-06: first for
relaxation curve fits, then "for normal spectral fits, too").

One function for every kind of fit openNMR makes:

* relaxation / diffusion curve fits
  (:func:`opennmr.notebook.relaxation.fit_relaxation`, ``NMR2D_fit_relaxation``);
* isotropic peak fits -- a 1D fit or one slice of
  :func:`opennmr.processing.fitting.NMR2D_fit_slices` (``popt``, ``pcov``,
  ``is_free``, ``peak_defs``);
* Solid-state NMR fits (:func:`opennmr.notebook.combined_fitting.fit_combined`);
* multi-spectrum fits (:func:`opennmr.notebook.multi_fitting.fit_multi`): the
  joint refinement's covariance, across every spectrum, or one spectrum's
  own fit in sequential mode.

    from opennmr.notebook import fit_correlation as fc
    fc.plot_correlation_matrix(fit)            # any of the above
    names, labels, R = fc.correlation_of(fit)  # the numbers

|r| near 1 means two parameters trade off: the data fix a combination of
them, not each one, and their separate ± understate how loosely each is
pinned. Overlapping peaks show it between their amplitudes and widths, a
CSA between Δ and η, a quadrupolar site between C_Q and δiso.
"""

from __future__ import annotations

import re

import numpy as np

from ..processing.relaxometry import correlation_matrix

#: Blue (negative) to red (positive) through a neutral light grey: two
#: poles that read as opposite, and a midpoint that reads as "nothing".
DIVERGING = ("#2166ac", "#f0efec", "#b2182b")

#: Values written in the cells up to this many parameters ("auto").
ANNOTATE_UP_TO = 12

_GREEK = {"beta": r"\beta", "tau": r"\tau", "tau_c": r"\tau_\mathrm{c}"}

#: Spectral-fit parameter names (Solid-state NMR, multi-spectrum) as
#: symbols: italic quantity, upright subscripts (IUPAC).
SPECTRAL_SYMBOLS = {
    "delta_iso": r"$\delta_\mathrm{iso}$",
    "shift": r"$\delta$",
    "fwhm": "FWHM",
    "width_ppm": "FWHM (ppm)",
    "width_Hz": "FWHM (Hz)",
    "amount": "amount",
    "amplitude": r"$\mathit{A}$",
    "eta": r"$\eta_\mathrm{L/G}$",
    "aniso": r"$\delta_\mathrm{aniso}$",
    "asym": r"$\eta_\mathrm{CSA}$",
    "C_Q": r"$\mathit{C}_\mathrm{Q}$",
    "eta_Q": r"$\eta_\mathrm{Q}$",
    "alpha": r"$\alpha$",
    "beta_euler": r"$\beta$",
    "gamma": r"$\gamma$",
    "mas_Hz": r"$\nu_\mathrm{r}$",
    "b0": r"$\mathit{b}_0$",
    "b1": r"$\mathit{b}_1$",
    "scale": "scale",
}
#: Names that are relaxation-model parameters when they stand alone, and
#: spectral ones only with a site label ("beta" is a stretch exponent on
#: its own, an Euler angle on a site).
_SITE_ONLY = {"beta": "beta_euler", "eta": "eta"}


def parameter_label(name: str) -> str:
    """A relaxation/diffusion parameter's name as an IUPAC-style mathtext
    label: symbol italic, subscripts upright -- ``T2a`` → *T*₂ₐ, ``A1`` →
    *A*₁, ``M0`` → *M*₀, ``beta`` → β. Unrecognised names are returned as
    they are."""
    if name in _GREEK:
        return f"${_GREEK[name]}$"
    if name == "T1rho":
        return r"$\mathit{T}_{1\rho}$"
    m = re.fullmatch(r"([A-Za-z])(\d*)([a-z]?)", name)
    if not m:
        return name
    sym, num, sub = m.groups()
    tail = num + sub
    return f"$\\mathit{{{sym}}}_{{\\mathrm{{{tail}}}}}$" if tail else f"$\\mathit{{{sym}}}$"


def _spectral_label(name: str, site: str = "") -> str:
    key = _SITE_ONLY.get(name, name) if site else name
    if key in SPECTRAL_SYMBOLS:
        symbol = SPECTRAL_SYMBOLS[key]
    elif re.fullmatch(r"J\d+", name):
        symbol = rf"$\mathit{{J}}_{{{name[1:]}}}$"
    else:
        symbol = parameter_label(name)
    return f"{symbol} ({site})" if site else symbol


def _split_name(free_name: str):
    """``"C1: aniso [spectrum 2]"`` → ``("C1", "aniso", "spectrum 2")``;
    ``"T1"`` → ``("", "T1", "")``."""
    tag = ""
    m = re.fullmatch(r"(.*) \[(.*)\]", free_name)
    if m:
        free_name, tag = m.groups()
    if ": " in free_name:
        site, name = free_name.split(": ", 1)
    else:
        site, name = "", free_name
    return site, name, tag


def correlation_input(fit: dict) -> dict:
    """Any supported fit as ``{"names", "labels", "groups", "pcov",
    "title"}``: its free parameters, their display labels, a group per
    parameter (the peak, site or spectrum it belongs to -- the figure draws
    a line between groups) and their covariance."""
    if fit.get("pcov") is not None and fit.get("free_names") is not None:
        names = [str(n) for n in fit["free_names"]]
        pcov = np.asarray(fit["pcov"], dtype=float)
        labels, groups = [], []
        spectral = any(": " in n for n in names) or any(
            _split_name(n)[1] in ("b0", "b1", "mas_Hz") for n in names)
        for n in names:
            site, name, tag = _split_name(n)
            if spectral:
                label = _spectral_label(name, site)
            else:
                label = parameter_label(name)
            if tag:
                label = f"{label} [{tag}]"
            labels.append(label)
            groups.append((tag, site) if spectral else _relaxation_group(name))
        title = fit.get("peak_label") or fit.get("title") or fit.get("label")
        return {"names": names, "labels": labels, "groups": groups, "pcov": pcov,
                "title": title}
    if all(k in fit for k in ("popt", "pcov", "is_free", "peak_defs")):
        return _isotropic_input(fit)
    raise ValueError("This fit has no covariance matrix to show (it needs 'pcov' with "
                     "'free_names', or a peak fit's 'popt', 'pcov', 'is_free' and "
                     "'peak_defs').")


def _relaxation_group(name: str) -> str:
    """Components of a multi-exponential fit: A1 with T..a, A2 with T..b."""
    m = re.fullmatch(r"[AI](\d)", name) or re.fullmatch(r"(?:T1|T2|D)([abc])", name)
    if not m:
        return ""
    key = m.group(1)
    return {"a": "1", "b": "2", "c": "3"}.get(key, key)


def _isotropic_input(fit: dict) -> dict:
    """An isotropic peak fit (``fit_spectrum`` / ``NMR2D_fit_slices``):
    every peak's shift, FWHM, amplitude (and η for a pseudo-Voigt), free
    ones only. ``pcov`` may be the free block or the expanded full
    matrix."""
    is_free = np.asarray(fit["is_free"], dtype=bool)
    pcov = np.asarray(fit["pcov"], dtype=float)
    names, labels, groups, index = [], [], [], []
    i = 0
    for k, peak in enumerate(fit["peak_defs"], start=1):
        params = ["shift", "fwhm", "amplitude"]
        if peak.get("type") not in ("gaussian", "lorentzian"):
            params.append("eta")
        for p in params:
            if i < is_free.size and is_free[i]:
                names.append(f"Peak {k}: {p}")
                sym = {"shift": r"$\delta_{%d}$" % k, "fwhm": r"FWHM$_{%d}$" % k,
                       "amplitude": r"$\mathit{A}_{%d}$" % k, "eta": r"$\eta_{%d}$" % k}[p]
                labels.append(sym)
                groups.append(k)
                index.append(i)
            i += 1
    # The J couplings of multiplets sit after every peak, before the baseline.
    for k, peak in enumerate(fit["peak_defs"], start=1):
        for j in range(len(peak.get("couplings") or ())):
            if i < is_free.size and is_free[i]:
                names.append(f"Peak {k}: J{j + 1}")
                labels.append(r"$\mathit{J}_{%d%s}$" % (k, "" if len(peak["couplings"]) == 1 else f",{j + 1}"))
                groups.append(k)
                index.append(i)
            i += 1
    n_free = int(is_free.sum())
    if pcov.shape == (is_free.size, is_free.size):
        sub = pcov[np.ix_(index, index)]
    elif pcov.shape == (n_free, n_free):
        free_pos = {full: j for j, full in enumerate(np.flatnonzero(is_free))}
        sel = [free_pos[i] for i in index]
        sub = pcov[np.ix_(sel, sel)]
    else:
        raise ValueError("The peak fit's covariance does not match its parameters.")
    title = fit.get("title")
    if title is None and fit.get("index") is not None and fit.get("slice_type"):
        title = f"Slice {fit['index']}"
    return {"names": names, "labels": labels, "groups": groups, "pcov": sub, "title": title}


def correlation_of(fit: dict):
    """``(names, labels, R)`` for any supported fit (see
    :func:`correlation_input`)."""
    data = correlation_input(fit)
    _names, R = correlation_matrix({"free_names": data["names"], "pcov": data["pcov"]})
    return data["names"], data["labels"], R


def multi_fit_correlation(result: dict, experiment=None) -> dict:
    """A multi-spectrum result in the shape :func:`plot_correlation_matrix`
    takes: the joint refinement's covariance across every spectrum
    (``experiment=None``), or experiment ``experiment``'s own fit (index or
    label; sequential mode)."""
    if experiment is None:
        if result.get("pcov") is None:
            raise ValueError("This multi-spectrum fit has no joint covariance (it was fitted "
                             "one spectrum at a time): choose one spectrum.")
        return {"pcov": result["pcov"], "free_names": result["free_names"],
                "title": "Joint fit, all spectra"}
    fits = result["experiments"]
    fit = (fits[experiment] if isinstance(experiment, int)
           else next(f for f in fits if f.get("label") == experiment))
    if fit.get("pcov") is None:
        raise ValueError(f"{fit.get('label', 'This spectrum')}'s parameters were refined "
                         "jointly: use the joint fit's matrix.")
    return {"pcov": fit["pcov"], "free_names": fit["free_names"], "title": fit.get("label")}


def plot_correlation_matrix(fit, figure=None, ax=None, triangle="lower", annotate="auto",
                            fontsize=12.0, title=None, colorbar=True, cmap=None, labels=None):
    """The correlation matrix of a fit's free parameters as a heat map:
    ``r`` from −1 (blue) through 0 (grey) to +1 (red), with each value
    written in its cell.

    ``fit`` : any fit :func:`correlation_input` takes -- a relaxation or
    diffusion curve fit, an isotropic peak fit (1D or one slice), a
    Solid-state NMR fit, or :func:`multi_fit_correlation` of a
    multi-spectrum one.
    ``figure`` / ``ax`` : draw into this figure (cleared) or these axes;
    otherwise a new figure.
    ``triangle`` : ``"lower"`` (each pair once, with the diagonal) or
    ``"full"``.
    ``annotate`` : write r in the cells -- ``"auto"`` does so up to
    :data:`ANNOTATE_UP_TO` parameters, beyond which they no longer fit.
    ``labels`` : axis labels to use instead of the generated ones.
    ``title`` : defaults to the fit's peak label / title.

    Parameters of different peaks, sites or spectra are separated by a
    line. A parameter with no determinable variance is hatched (``nan``).

    Returns the figure; its axes are ``figure.correlation_axes`` =
    ``{"matrix", "colorbar"}`` and the numbers ``figure.correlation`` =
    ``(names, R)``.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    from ..plotting.style import apply_tick_style

    data = correlation_input(fit)
    names, R = correlation_matrix({"free_names": data["names"], "pcov": data["pcov"]})
    n = len(names)
    if triangle not in ("lower", "full"):
        raise ValueError("triangle must be 'lower' or 'full'.")
    if annotate == "auto":
        annotate = n <= ANNOTATE_UP_TO
    if ax is None:
        if figure is None:
            side = min(max(3.0, 0.55 * n + 1.6), 9.0)
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
    face = figure.get_facecolor()
    gap = 1.5 if n <= 20 else 0.5
    mesh = ax.pcolormesh(np.arange(n + 1), np.arange(n + 1), np.ma.masked_invalid(shown),
                         cmap=cmap, vmin=-1.0, vmax=1.0, edgecolors=face, linewidth=gap)
    for i in range(n):
        ax.add_patch(plt.Rectangle((i, i), 1, 1, facecolor="0.88", edgecolor=face,
                                   linewidth=gap))
    # Undeterminable cells (not the hidden upper triangle) are hatched.
    for i in range(n):
        for j in range(n):
            if i == j or (triangle == "lower" and j > i):
                continue
            if not np.isfinite(R[i, j]):
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, hatch="///",
                                           edgecolor="0.6", linewidth=0))
    # A line between peaks / sites / spectra.
    groups = data["groups"]
    if len(set(map(str, groups))) > 1:
        for k in range(1, n):
            if groups[k] != groups[k - 1]:
                # Within the drawn triangle only.
                x_end = k if triangle == "lower" else n
                ax.plot([0, x_end], [k, k], color="0.35", linewidth=1.0, zorder=5)
                ax.plot([k, k], [k if triangle == "lower" else 0, n], color="0.35",
                        linewidth=1.0, zorder=5)
    ax.set_xlim(0, n)
    ax.set_ylim(n, 0)
    ax.set_aspect("equal")
    tick_labels = list(labels) if labels is not None else data["labels"]
    tick_size = fontsize if n <= 12 else fontsize * max(0.5, 12.0 / n)
    ax.set_xticks(np.arange(n) + 0.5)
    ax.set_yticks(np.arange(n) + 0.5)
    ax.set_xticklabels(tick_labels, fontsize=tick_size, rotation=90 if n > 6 else 0)
    ax.set_yticklabels(tick_labels, fontsize=tick_size)
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
                text = ("1" if i == j else "0.00" if abs(r) < 0.005
                        else f"{r:+.2f}".replace("-", "\u2212"))
                ax.text(j + 0.5, i + 0.5, text, ha="center", va="center", fontsize=size,
                        color=ink)
    title = data["title"] if title is None else title
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
