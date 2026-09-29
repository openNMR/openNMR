"""
Kept fit models: several fits of the same spectrum (a 2-, 3- and 4-site
model, a Czjzek against discrete sites, two relaxation models...) saved
with the spectrum, reopened later, and compared on one plot and by
information criteria.

A fit model is a plain dict (see :func:`make_fit_model`): what was fitted
(``x``, ``y``), the fitted curve and its components, how many parameters
were free, which points the fit used, its readable report, and -- when it
came from a GUI fit panel -- that panel's inputs and exact result, so it
can be put back. :func:`encode_fit_model` / :func:`decode_fit_model` make
it JSON-safe for the spectrum's saved state (``metadata.raw["fit_models"]``).

Comparing models fitted to the same points (:func:`compare_fit_models`)::

    AIC = n ln(RSS/n) + 2k        BIC = n ln(RSS/n) + k ln n

(least squares with Gaussian noise of unknown variance; n points, k free
parameters). Lower is better; only differences matter. For BIC, a
difference of 2-6 is positive, 6-10 strong and over 10 very strong
evidence (Kass & Raftery, J. Am. Stat. Assoc. 90, 773 (1995)). R^2 alone
always prefers more sites, which is why the criteria are reported.

Notebook::

    from opennmr.notebook.fit_models import make_fit_model, compare_fit_models, plot_fit_comparison
    m2 = make_fit_model("2 sites", "quadrupolar", x, y, fit2, components=comps2, n_free=11)
    m3 = make_fit_model("3 sites", "quadrupolar", x, y, fit3, components=comps3, n_free=17)
    print(format_comparison(compare_fit_models([m2, m3])))
    fig = plot_fit_comparison([m2, m3], residuals=True)
"""

from __future__ import annotations

import copy
import datetime as _dt
import math

import numpy as np

KIND_LABELS = {
    "isotropic": "Isotropic peaks", "integration": "Integration", "jcoupling": "J-coupling",
    "csa": "CSA/Dipolar", "quadrupolar": "Quadrupolar", "czjzek": "Czjzek",
    "combined": "Combined Hamiltonian", "curve": "Curve fit", "spectrum": "Spectral fit",
}

#: Evidence scale for a BIC difference (Kass & Raftery 1995).
BIC_EVIDENCE = ((2.0, "not worth more than a mention"), (6.0, "positive"), (10.0, "strong"),
                (math.inf, "very strong"))


def make_fit_model(name, kind, x, y, y_fit, components=None, labels=None, n_free=None,
                   mask=None, report="", target="", x_label="", y_label="", title="",
                   panel_inputs=None, panel_result=None, when=None) -> dict:
    """A fit model record.

    ``x``, ``y``, ``y_fit`` : the data and total fit (any x order).
    ``components`` : per-component curves, baseline last (optional).
    ``labels`` : one per component, baseline excluded. ``n_free`` : free
    parameters (needed for AIC/BIC). ``mask`` : the points the fit used
    (default: all). ``panel_inputs`` / ``panel_result`` : a GUI panel's
    ``get_state()`` and encoded ``result_state()``, to put it back.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    y_fit = np.asarray(y_fit, dtype=float)
    if not (x.shape == y.shape == y_fit.shape):
        raise ValueError("x, y and y_fit must have the same length.")
    mask = np.ones(x.size, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if mask.shape != x.shape:
        raise ValueError("mask must match x.")
    comps = [np.asarray(c, dtype=float) for c in (components or [])]
    if comps and any(c.shape != x.shape for c in comps):
        raise ValueError("Every component must match x.")
    n_comp = max(len(comps) - 1, 0)
    labels = [str(v) for v in (labels or [])][:n_comp]
    labels += [f"Component {i + 1}" for i in range(len(labels), n_comp)]
    return {
        "name": str(name), "kind": str(kind), "kind_label": KIND_LABELS.get(kind, str(kind)),
        "time": (when or _dt.datetime.now()).isoformat(timespec="seconds"),
        "target": str(target), "title": str(title or name),
        "x": x, "y": y, "y_fit": y_fit, "components": comps, "labels": labels,
        "mask": mask, "n_free": None if n_free is None else int(n_free),
        "report": str(report or ""), "x_label": str(x_label), "y_label": str(y_label),
        "panel_inputs": panel_inputs, "panel_result": panel_result,
    }


def encode_fit_model(model: dict) -> dict:
    """``model`` as JSON-safe data (arrays exact), for a saved state."""
    from ..io import serialize
    return serialize.encode(model)


def decode_fit_model(encoded: dict) -> dict:
    from ..io import serialize
    model = serialize.decode(copy.deepcopy(encoded))
    for key in ("x", "y", "y_fit", "mask"):
        if model.get(key) is not None:
            model[key] = np.asarray(model[key])
    model["components"] = [np.asarray(c) for c in (model.get("components") or [])]
    return model


def fit_statistics(model: dict) -> dict:
    """``n``, ``k``, ``rss``, ``r2``, ``reduced_chi2`` (RSS / (n - k)),
    ``aic``, ``bic`` over the points the fit used. AIC/BIC are None when
    the parameter count is unknown."""
    m = np.asarray(model["mask"], dtype=bool)
    y, f = np.asarray(model["y"])[m], np.asarray(model["y_fit"])[m]
    n = int(y.size)
    rss = float(np.sum((y - f) ** 2))
    tss = float(np.sum((y - y.mean()) ** 2)) if n else 0.0
    k = model.get("n_free")
    out = {"n": n, "k": k, "rss": rss, "r2": 1 - rss / tss if tss else float("nan"),
           "reduced_chi2": None, "aic": None, "bic": None}
    if k is not None and n > k:
        out["reduced_chi2"] = rss / (n - k)
    if k is not None and n > 0 and rss > 0:
        base = n * math.log(rss / n)
        out["aic"] = base + 2 * k
        out["bic"] = base + k * math.log(n)
    return out


def _same_points(models) -> bool:
    first = models[0]
    for m in models[1:]:
        if (np.asarray(m["x"]).shape != np.asarray(first["x"]).shape
                or not np.array_equal(np.asarray(m["mask"]), np.asarray(first["mask"]))
                or not np.allclose(np.asarray(m["x"])[np.asarray(m["mask"])],
                                   np.asarray(first["x"])[np.asarray(first["mask"])])
                or not np.allclose(np.asarray(m["y"])[np.asarray(m["mask"])],
                                   np.asarray(first["y"])[np.asarray(first["mask"])])):
            return False
    return True


def bic_evidence(delta: float) -> str:
    for limit, words in BIC_EVIDENCE:
        if delta < limit:
            return words
    return BIC_EVIDENCE[-1][1]


def compare_fit_models(models) -> dict:
    """Statistics for each model, and -- when they were all fitted to the
    same points -- ΔAIC and ΔBIC from the best (lowest). Returns
    ``{"rows": [...], "comparable": bool, "note": str}``; each row is
    :func:`fit_statistics` plus ``name``, ``kind_label``, ``delta_aic``,
    ``delta_bic`` and ``evidence`` (how strongly the best BIC beats it)."""
    models = list(models)
    if not models:
        return {"rows": [], "comparable": False, "note": "No models."}
    rows = []
    for m in models:
        row = {"name": m["name"], "kind_label": m.get("kind_label", m.get("kind", "")),
               **fit_statistics(m)}
        rows.append(row)
    comparable = len(models) > 1 and _same_points(models)
    note = ""
    if len(models) > 1 and not comparable:
        note = ("These fits used different points (window, regions or data), so AIC/BIC "
                "cannot rank them -- only fits to the same points can be compared.")
    for key in ("aic", "bic"):
        values = [r[key] for r in rows if r[key] is not None]
        best = min(values) if values and comparable else None
        for r in rows:
            r["delta_" + key] = (r[key] - best) if (best is not None and r[key] is not None) else None
    for r in rows:
        r["evidence"] = ("best" if r["delta_bic"] == 0 else
                         (f"{bic_evidence(r['delta_bic'])} evidence against"
                          if r["delta_bic"] is not None else ""))
    if comparable and any(r["k"] is None for r in rows):
        note = "Some models have no parameter count, so they have no AIC/BIC."
    return {"rows": rows, "comparable": comparable, "note": note}


def format_comparison(comparison: dict) -> str:
    """The comparison as a plain-text table."""
    def f(value, fmt):
        return "—" if value is None or (isinstance(value, float) and math.isnan(value)) \
            else format(value, fmt)
    lines = [f"{'Model':<28}{'k':>4}{'n':>7}{'R²':>9}{'red. χ²':>12}{'ΔAIC':>9}{'ΔBIC':>9}  evidence"]
    for r in comparison["rows"]:
        lines.append(f"{r['name'][:27]:<28}{f(r['k'], 'd'):>4}{r['n']:>7}{f(r['r2'], '.5f'):>9}"
                     f"{f(r['reduced_chi2'], '.4g'):>12}{f(r['delta_aic'], '.1f'):>9}"
                     f"{f(r['delta_bic'], '.1f'):>9}  {r['evidence']}")
    if comparison.get("note"):
        lines.append(comparison["note"])
    return "\n".join(lines)


#: How one kept fit is drawn beside others (the GUI keeps one per Open
#: experiment made from a kept fit; see :func:`draw_fit_model`).
DEFAULT_FIT_VIEW = {
    "show_data": True,        # the measured spectrum
    "show_fit": True,         # the total fit
    "show_residual": False,   # data - fit, offset below the spectrum
    "residual_gap": 0.08,     # gap above the residual, as a fraction of the data's range
    "fit_colour": None,       # None: the trace's own colour (Plot Style / ``colours``)
    # Component display (notebook.fit_display.component_style keys). No
    # baseline by default: beside other traces, a flat line at zero is
    # clutter (and one more legend entry per fit).
    "style": {"show_baseline": False},
}


def fit_view(view: dict | None = None, **overrides) -> dict:
    """A complete, checked fit view (see :data:`DEFAULT_FIT_VIEW`)."""
    from .fit_display import component_style
    out = copy.deepcopy(DEFAULT_FIT_VIEW)
    for source in (view or {}), overrides:
        for key, value in source.items():
            if key not in out:
                raise KeyError(f"Unknown fit-view key {key!r}; known: {', '.join(sorted(out))}.")
            out[key] = copy.deepcopy(value)
    for key in ("show_data", "show_fit", "show_residual"):
        out[key] = bool(out[key])
    out["residual_gap"] = float(max(0.0, out["residual_gap"]))
    out["style"] = component_style(out["style"])
    return out


def draw_fit_model(ax, model, view=None, scale: float = 1.0, fit_colour=None, name=None,
                   draw_data: bool = True, residual_index: int = 0, residual_step=None) -> dict:
    """One kept fit on ``ax``: its data, total fit and components (per
    ``view["style"]``), and optionally its residual, offset just below the
    spectrum. ``scale`` multiplies every curve (a normalised or per-mass
    plot). ``fit_colour`` : the total fit's colour unless the view sets
    one. ``residual_index`` / ``residual_step`` put several models'
    residuals one below another on a shared axes.

    Returns ``{"data", "fit", "components", "dimmed", "baseline",
    "residual", "residual_zero"}``."""
    from .fit_display import draw_fit_components
    view = fit_view(view)
    style = dict(view["style"])
    colour = view["fit_colour"] or fit_colour
    if colour is not None:
        style["fit_colour"] = colour
    x = np.asarray(model["x"], dtype=float)
    order = np.argsort(x)
    x = x[order]
    y = np.asarray(model["y"], dtype=float)[order] * scale
    y_fit = np.asarray(model["y_fit"], dtype=float)[order] * scale
    mask = np.asarray(model.get("mask", np.ones(x.size, dtype=bool)), dtype=bool)[order]
    comps = [np.asarray(c, dtype=float)[order] * scale for c in (model.get("components") or [])]
    name = str(name or model.get("name") or "Fit")
    out = draw_fit_components(ax, x, y, y_fit, components=comps or None,
                              labels=model.get("labels") or None, style=style,
                              auto_hide_single=True, data_label="Data", fit_label=name,
                              draw_data=draw_data and view["show_data"])
    if not view["show_fit"]:
        out["fit"].set_visible(False)
        out["fit"].set_label("_" + name)
    out["residual"] = out["residual_zero"] = None
    if view["show_residual"]:
        res = np.where(mask, y - y_fit, np.nan)
        span = float(np.nanmax(y) - np.nanmin(y)) if y.size else 1.0
        rspan = float(np.nanmax(res) - np.nanmin(res)) if np.any(mask) else 0.0
        step = float(residual_step) if residual_step is not None else 1.15 * rspan
        low = min(float(np.nanmin(y)), float(np.nanmin(y_fit)), 0.0)
        base = low - view["residual_gap"] * span - float(np.nanmax(res)) - residual_index * step
        out["residual_zero"] = ax.plot(x, np.full(x.size, base), color="0.75", linewidth=0.6,
                                       label="_residual zero", zorder=0)[0]
        out["residual"] = ax.plot(x, res + base, color=style["fit_colour"], linewidth=0.9,
                                  label=f"_{name} residual", zorder=3)[0]
    return out


def plot_fit_comparison(models, figure=None, residuals=True, show_components=None,
                        colours=None, fontsize=12.0, x_label=None, data_colour="black",
                        reverse_x=None, layout="superimposed", views=None):
    """Several fits of one spectrum.

    ``layout="superimposed"`` : the data once, each model's total fit in
    its own colour, and (with ``residuals``) each model's residual in a
    strip below, offset so they do not overlap. ``show_components`` : the
    name of one model whose components to shade as well.

    ``layout="stacked"`` : one row per model, each with the data, that
    model's fit and components (``views``: one :func:`fit_view` per model,
    or one for all) and, with ``residuals``, its residual just below --
    the data repeated in every row, which some prefer.

    ``colours`` : each model's fit colour. ``reverse_x``: default True
    for a ppm spectrum (any kind but "curve"). Returns the figure."""
    import matplotlib.pyplot as plt
    from ..plotting import qualitative_colors
    from ..plotting.style import apply_spine_style, apply_tick_style

    models = list(models)
    if not models:
        raise ValueError("No models to plot.")
    if layout not in ("superimposed", "stacked"):
        raise ValueError("layout must be 'superimposed' or 'stacked'.")
    if layout == "stacked":
        return _plot_fit_stack(models, figure, residuals, colours, fontsize, x_label,
                               data_colour, reverse_x, views)
    if figure is None:
        figure = plt.figure(figsize=(6.0, 4.8 if residuals else 3.8))
    else:
        figure.clear()
    colours = list(colours or qualitative_colors(len(models)))
    curve = all(m.get("kind") == "curve" for m in models)
    reverse_x = (not curve) if reverse_x is None else bool(reverse_x)
    if residuals:
        grid = figure.add_gridspec(2, 1, height_ratios=[3, 1.2], hspace=0.06)
        ax = figure.add_subplot(grid[0])
        rax = figure.add_subplot(grid[1], sharex=ax)
    else:
        ax, rax = figure.add_subplot(111), None
    first = models[0]
    x0 = np.asarray(first["x"])
    m0 = np.asarray(first["mask"])
    ax.plot(x0[m0] if curve else x0, np.asarray(first["y"])[m0] if curve else first["y"],
            "o" if curve else "-", color=data_colour, linewidth=1.2, markersize=5,
            label="Data", zorder=0)
    step = 0.0
    if rax is not None:
        spans = [float(np.ptp((np.asarray(m["y"]) - np.asarray(m["y_fit"]))[np.asarray(m["mask"])]))
                 for m in models]
        step = 1.2 * max(spans + [0.0])
    for i, m in enumerate(models):
        x = np.asarray(m["x"])
        order = np.argsort(x)
        colour = colours[i % len(colours)]
        ax.plot(x[order], np.asarray(m["y_fit"])[order], "--" if not curve else "-",
                color=colour, linewidth=1.8, label=m["name"], zorder=10 + i)
        if show_components and m["name"] == show_components and m.get("components"):
            for c, label in zip(m["components"][:-1], m.get("labels") or []):
                ax.fill_between(x[order], np.asarray(c)[order], color=colour, alpha=0.18,
                                linewidth=0.8, edgecolor=colour, zorder=1)
        if rax is not None:
            mask = np.asarray(m["mask"])
            res = (np.asarray(m["y"]) - np.asarray(m["y_fit"]))
            res = np.where(mask, res, np.nan)
            rax.plot(x[order], res[order] - i * step, color=colour, linewidth=0.9)
            rax.axhline(-i * step, color="0.75", linewidth=0.6)
    ax.legend(frameon=False, fontsize=fontsize * 0.8)
    label = x_label or first.get("x_label") or ("Shift (ppm)" if not curve else "x")
    last = rax if rax is not None else ax
    last.set_xlabel(label, fontsize=fontsize)
    if curve and first.get("y_label"):
        ax.set_ylabel(first["y_label"], fontsize=fontsize)
    for a in (ax, rax):
        if a is None:
            continue
        if not curve:
            a.set_yticks([])
            apply_spine_style(a, top=False, right=False, bottom=True, left=False)
        else:
            apply_spine_style(a, top=False, right=False, bottom=True, left=True)
        apply_tick_style(a, labelsize=fontsize * 0.9, axis="x" if not curve else "both")
    if rax is not None:
        rax.set_ylabel("Residuals", fontsize=fontsize * 0.8)
        ax.tick_params(labelbottom=False)
    if reverse_x:
        lo, hi = float(np.min(x0)), float(np.max(x0))
        ax.set_xlim(hi, lo)
    figure.fit_comparison_axes = {"fit": ax, "residuals": rax}
    _tight(figure)
    return figure


def _tight(figure) -> None:
    import warnings
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            figure.tight_layout()
    except Exception:  # noqa: BLE001 -- layout is cosmetic
        pass


def _plot_fit_stack(models, figure, residuals, colours, fontsize, x_label, data_colour,
                    reverse_x, views):
    import matplotlib.pyplot as plt
    from ..plotting import qualitative_colors
    from ..plotting.style import apply_spine_style, apply_tick_style

    n = len(models)
    if figure is None:
        figure = plt.figure(figsize=(6.0, 1.9 * n + 0.8))
    else:
        figure.clear()
    colours = list(colours or qualitative_colors(n))
    if views is None or isinstance(views, dict):
        views = [views] * n
    curve = all(m.get("kind") == "curve" for m in models)
    reverse_x = (not curve) if reverse_x is None else bool(reverse_x)
    axes = list(np.atleast_1d(figure.subplots(n, 1, sharex=True)))
    for i, (ax, m) in enumerate(zip(axes, models)):
        view = fit_view(views[i] if i < len(views) else None)
        if not views[i]:
            view["style"] = {**view["style"], "data_colour": data_colour}
            view["show_residual"] = bool(residuals)
        draw_fit_model(ax, m, view, fit_colour=colours[i % len(colours)])
        ax.text(0.995, 0.92, m["name"], transform=ax.transAxes, ha="right", va="top",
                color=view["fit_colour"] or colours[i % len(colours)], fontsize=fontsize * 0.8)
        ax.set_yticks([])
        apply_spine_style(ax, top=False, right=False, bottom=ax is axes[-1], left=False)
        if ax is axes[-1]:
            apply_tick_style(ax, axis="x", labelsize=fontsize * 0.9)
        else:
            ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    first = models[0]
    axes[-1].set_xlabel(x_label or first.get("x_label") or ("Shift (ppm)" if not curve else "x"),
                        fontsize=fontsize)
    if reverse_x:
        x0 = np.asarray(first["x"], dtype=float)
        axes[0].set_xlim(float(np.max(x0)), float(np.min(x0)))
    figure.subplots_adjust(hspace=0.05)
    figure.fit_comparison_axes = {"rows": axes}
    _tight(figure)
    return figure
