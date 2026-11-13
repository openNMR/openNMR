"""
Compare spectra as a **row, column or grid of separate panels**: 1D spectra,
2D spectra and pulse-sequence diagrams, in any mixture.

The 1D counterpart of :func:`opennmr.notebook.NMR2D_multi_plot`, and what the
GUI's Compare tile draws for *Layout ▸ Panels*::

    from opennmr.notebook import compare_panels as cp

    # one spectrum per panel, side by side, zoom linked
    fig, axes, link = cp.plot_compare_panels([s1, s2, s3], layout="row")

    # a grid; panel 1 holds two spectra, panel 2 a kept fit (data, fit,
    # components, residual), panel 3 one spectrum
    fig, axes, link = cp.plot_compare_panels(
        [[s1, s2], fit_experiment, s3], layout="grid", ncols=2)

    link.set(x=False)          # zoom each panel on its own
    link.set(y=True)           # ... or share the vertical scale too

    # 1D beside a 2D (several 2D in one panel are overlaid), a pulse sequence
    # in a panel of its own, and (a), (b)... on the panels
    fig, axes, link = cp.plot_compare_panels(
        [s1, hsqc, {"pulse_sequence": seq}], layout="row",
        panel_labels={"style": "(a)", "start": "a", "order": "row"})

Each **panel** is one Axes holding one 1D spectrum or many (superimposed, one
colour each, with a legend), one 2D spectrum (or 2-6 overlaid), or a pulse
sequence. Panel labels come from :mod:`opennmr.notebook.panel_labels`. A spectrum that carries a kept fit
(``metadata.raw["fit_model"]``, i.e. an experiment made by "Keep in Open
Experiments" on a kept fit) is drawn with its fit, components and residual as
set by its ``fit_model_view`` -- exactly as Compare draws it.

Linked panels show the same chemical-shift region: zooming, panning or
scrolling the axis of one does the same on all. The link is not matplotlib's
``sharex`` so that it can be switched on and off after drawing
(:meth:`PanelLink.set`), as the GUI's *Link panels* box does.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np

from ..core.spectrum import NMRSpectrum

LAYOUTS = ("row", "column", "grid")

__all__ = ["LAYOUTS", "ComparePanels", "PanelLink", "plot_compare_panels",
           "grid_shape", "panel_groups", "draw_kept_fit_overlays", "apply_axis_styles",
           "AXIS_STYLE_KEYS", "TICK_DIRECTIONS", "AXIS_LABEL_MODES"]


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------

def grid_shape(n: int, layout: str = "row", ncols: int | None = None) -> tuple:
    """``(nrows, ncols)`` for ``n`` panels. ``row``: 1 x n; ``column``:
    n x 1; ``grid``: ``ncols`` columns (default: as square as possible)."""
    n = max(int(n), 1)
    if layout == "row":
        return 1, n
    if layout == "column":
        return n, 1
    if layout != "grid":
        raise ValueError(f"layout must be one of {LAYOUTS}, got {layout!r}.")
    cols = int(ncols) if ncols else math.ceil(math.sqrt(n))
    cols = min(max(cols, 1), n)
    return math.ceil(n / cols), cols


def panel_groups(assignment) -> list:
    """Spectrum indices grouped by panel: ``[1, 1, 2, 3]`` (the panel each
    spectrum was given) -> ``[[0, 1], [2], [3]]``. Panel numbers need not be
    consecutive -- they are only used to say which spectra go together and in
    what order the panels come."""
    groups: dict = {}
    for i, p in enumerate(assignment):
        groups.setdefault(p, []).append(i)
    return [groups[k] for k in sorted(groups)]


# ---------------------------------------------------------------------------
# linking
# ---------------------------------------------------------------------------

class PanelLink:
    """Keeps the panels' views together. ``x`` (default True): the shift axis
    -- zooming or panning one panel moves them all. ``y`` (default False):
    the intensity axis too (only meaningful for spectra on one scale).
    Switch either at any time with :meth:`set`; turning one on puts every
    panel on the first panel's current view.

    ``x_keys`` / ``y_keys`` (one entry per panel) split the panels into link
    groups: panels follow only those with an equal key, and a ``None`` key
    never links. A figure mixing 1D and 2D spectra uses them so that a 1D and
    the F2 axis of a 2D of the same nucleus move together, while a pulse
    sequence (no shift axis) is left alone."""

    def __init__(self, axes, *, x: bool = True, y: bool = False, x_keys=None, y_keys=None) -> None:
        self.axes = list(axes)
        n = len(self.axes)
        self.x_keys = list(x_keys) if x_keys is not None else [0] * n
        self.y_keys = list(y_keys) if y_keys is not None else [0] * n
        if len(self.x_keys) != n or len(self.y_keys) != n:
            raise ValueError("x_keys and y_keys need one entry per panel.")
        self._busy = False
        self.x = False
        self.y = False
        for ax in self.axes:
            ax.callbacks.connect("xlim_changed", self._on_x)
            ax.callbacks.connect("ylim_changed", self._on_y)
        self.set(x=x, y=y)

    def set(self, x: bool | None = None, y: bool | None = None) -> None:
        many = len(self.axes) > 1
        if x is not None:
            turned_on = bool(x) and many and not self.x
            self.x = bool(x) and many
            if turned_on:
                self._align("x")
        if y is not None:
            turned_on = bool(y) and many and not self.y
            self.y = bool(y) and many
            if turned_on:
                self._align("y")

    def _align(self, which: str) -> None:
        """Every group onto its first panel's view."""
        keys = self.x_keys if which == "x" else self.y_keys
        seen = []
        for ax, key in zip(self.axes, keys):
            if key is None or key in seen:
                continue
            seen.append(key)
            self._follow(ax, which)

    def _on_x(self, source) -> None:
        self._follow(source, "x")

    def _on_y(self, source) -> None:
        self._follow(source, "y")

    def _follow(self, source, which: str) -> None:
        if self._busy or not (self.x if which == "x" else self.y):
            return
        keys = self.x_keys if which == "x" else self.y_keys
        try:
            key = keys[self.axes.index(source)]
        except ValueError:
            return
        if key is None:
            return
        self._busy = True
        try:
            limits = source.get_xlim() if which == "x" else source.get_ylim()
            for other, other_key in zip(self.axes, keys):
                if other is not source and other_key == key:
                    (other.set_xlim if which == "x" else other.set_ylim)(limits)
        finally:
            self._busy = False


class ComparePanels(NamedTuple):
    fig: object
    axes: list
    link: PanelLink

    @property
    def labels(self):
        """The panel labels (:class:`opennmr.notebook.panel_labels.PanelLabels`),
        or None when none were asked for."""
        return getattr(self.fig, "_opennmr_panel_labels", None)


# ---------------------------------------------------------------------------
# kept fits over their data (shared with the GUI's stacked / superimposed Compare)
# ---------------------------------------------------------------------------

_MODEL_CACHE: dict = {}


def _kept_fit_model(spectrum):
    """The kept fit ``spectrum`` was made from (decoded), or None."""
    from .fit_models import decode_fit_model
    raw = getattr(getattr(spectrum, "metadata", None), "raw", None) or {}
    encoded = raw.get("fit_model")
    if not encoded:
        return None
    hit = _MODEL_CACHE.get(id(encoded))
    if hit is None or hit[0] is not encoded:
        try:
            hit = (encoded, decode_fit_model(encoded))
        except Exception:  # noqa: BLE001 -- a damaged record draws as plain data
            return None
        if len(_MODEL_CACHE) > 64:
            _MODEL_CACHE.clear()
        _MODEL_CACHE[id(encoded)] = hit
    return hit[1]


def draw_kept_fit_overlays(spectra, axes, data_lines, colors, names, normalize: bool,
                           model_for=None) -> None:
    """Each entry made from a kept fit gets its fit drawn over its data: the
    total fit (in the trace's colour), its components and, if asked, its
    residual just below -- stacked (a row each, the data repeated) or
    superimposed (the data drawn once). How is the entry's own
    ``metadata.raw["fit_model_view"]`` (:func:`opennmr.notebook.fit_models.fit_view`).
    ``model_for(spectrum)`` finds the kept fit (default: decode it from the
    spectrum's metadata)."""
    from .fit_models import draw_fit_model, fit_view
    model_for = model_for or _kept_fit_model
    items = []
    for i, spectrum in enumerate(spectra):
        model = model_for(spectrum)
        if model is None:
            continue
        y = np.asarray(model["y"], dtype=float)
        data = np.real(np.asarray(spectrum.data, dtype=float))
        if data.shape != y.shape or not np.any(y):
            continue
        k = int(np.argmax(np.abs(y)))
        scale = float(data[k] / y[k]) if y[k] else 1.0
        if normalize:
            peak = float(np.max(np.abs(data))) or 1.0
            scale /= peak
        try:
            view = fit_view(spectrum.metadata.raw.get("fit_model_view"))
        except (KeyError, ValueError, TypeError):
            view = fit_view()
        items.append((i, model, view, scale))
    if not items:
        return
    shared = len(set(map(id, axes))) == 1
    step = None
    if shared:
        spans = [float(np.ptp((np.asarray(m["y"]) - np.asarray(m["y_fit"]))
                              [np.asarray(m["mask"], dtype=bool)])) * abs(sc)
                 for _i, m, v, sc in items if v["show_residual"]]
        step = 1.15 * max(spans) if spans else None
    drawn_data: set = set()
    residual_index = 0
    for i, model, view, scale in items:
        ax, line = axes[i], data_lines[i]
        data_colour = view["style"]["data_colour"]
        key = (np.asarray(model["y"], dtype=float).tobytes(), round(scale, 12))
        if not view["show_data"] or (shared and key in drawn_data):
            line.set_visible(False)
            line.set_label("_hidden data")
        else:
            drawn_data.add(key)
            line.set_color(data_colour)
            line.set_linewidth(view["style"]["data_width"])
            line.set_label("Data" if not shared or len(drawn_data) == 1 else "_data")
        draw_fit_model(ax, model, view, scale=scale, fit_colour=colors[i], name=names[i],
                       draw_data=False, residual_index=residual_index if shared else 0,
                       residual_step=step)
        if shared and view["show_residual"]:
            residual_index += 1


# ---------------------------------------------------------------------------
# the plot
# ---------------------------------------------------------------------------

def _spectrum_label(spectrum, fallback: str) -> str:
    raw = getattr(getattr(spectrum, "metadata", None), "raw", None) or {}
    return str(raw.get("display_label") or fallback)


#: Per-panel axis options (``axis_styles``): every key is optional.
AXIS_STYLE_KEYS = ("xlabel", "show_xlabel", "xlabel_fontsize", "tick_direction", "tick_length",
                   "tick_width", "tick_labelsize", "minor_ticks", "num_ticks", "show_tick_labels",
                   "spine_width")
TICK_DIRECTIONS = ("in", "out", "inout")
AXIS_LABEL_MODES = ("auto", "all", "bottom")


def _style_for(axis_styles, k: int) -> dict:
    """The options of panel ``k`` (0-based): the ``"all"`` entry, then the
    panel's own on top. ``axis_styles`` is ``{k or "all": {...}}`` or a
    list (one dict per panel; ``None`` for none)."""
    if not axis_styles:
        return {}
    if isinstance(axis_styles, dict):
        merged = dict(axis_styles.get("all") or {})
        merged.update(axis_styles.get(k) or {})
    else:
        merged = dict(axis_styles[k] or {}) if k < len(axis_styles) else {}
    unknown = set(merged) - set(AXIS_STYLE_KEYS)
    if unknown:
        raise ValueError(f"unknown axis option(s) {sorted(unknown)}; use {AXIS_STYLE_KEYS}")
    if merged.get("tick_direction") not in (None, *TICK_DIRECTIONS):
        raise ValueError(f"tick_direction must be one of {TICK_DIRECTIONS}")
    return merged


def apply_axis_styles(axes, axis_styles, fontsize: float = 12.0) -> None:
    """Apply ``axis_styles`` (see :func:`plot_compare_panels`) to each of
    ``axes``. Safe to call again after anything that restyles the ticks --
    the GUI does, so its own tick defaults never override yours."""
    from matplotlib.ticker import MaxNLocator
    for k, ax in enumerate(axes):
        st = _style_for(axis_styles, k)
        if not st or ax is None or getattr(ax, "_opennmr_panel_kind", "") == "pulse":
            continue
        kw = {}
        for key, name in (("tick_direction", "direction"), ("tick_length", "length"),
                          ("tick_width", "width"), ("tick_labelsize", "labelsize")):
            if st.get(key) is not None:
                kw[name] = st[key]
        if kw:
            ax.tick_params(axis="x", which="major", **kw)
            minor = {n: v for n, v in kw.items() if n != "labelsize"}
            if "length" in minor:
                minor["length"] = minor["length"] / 2
            if "width" in minor:
                minor["width"] = minor["width"] * 0.75
            if minor:
                ax.tick_params(axis="x", which="minor", **minor)
        if st.get("minor_ticks") is not None:
            ax.minorticks_on() if st["minor_ticks"] else ax.minorticks_off()
            if st["minor_ticks"] and kw:
                ax.tick_params(axis="x", which="minor", direction=kw.get("direction", "in"))
        if st.get("num_ticks"):
            ax.xaxis.set_major_locator(MaxNLocator(nbins=int(st["num_ticks"])))
        if st.get("show_tick_labels") is not None:
            ax.tick_params(axis="x", labelbottom=bool(st["show_tick_labels"]))
        if st.get("spine_width") is not None:
            ax.spines["bottom"].set_linewidth(st["spine_width"])
        text = st.get("xlabel")
        if text is not None:
            ax.set_xlabel(text, fontsize=st.get("xlabel_fontsize") or fontsize)
        elif st.get("xlabel_fontsize"):
            ax.xaxis.label.set_fontsize(st["xlabel_fontsize"])
        if st.get("show_xlabel") is False:
            ax.set_xlabel("")
        elif st.get("show_xlabel") and not ax.get_xlabel():
            ax.set_xlabel(getattr(ax, "_opennmr_auto_xlabel", ""), fontsize=st.get("xlabel_fontsize") or fontsize)


def _spectrum_kind(spectrum) -> str:
    return "2d" if getattr(spectrum, "ndim", 1) == 2 else "1d"


def _as_panels(panels) -> list:
    """Each panel -> ``{"kind": "1d" | "2d" | "pulse", "spectra": [...],
    "title": ..., "labels": [...], "colors": [...], ...}``. A panel may be a
    spectrum, a list of them, or such a dict (``title`` / ``labels`` /
    ``colors`` optional; a 2D panel may carry ``insets`` and ``options`` for
    :func:`opennmr.plotting.twod.plot_2d_contour`). A **pulse-sequence panel**
    is ``{"pulse_sequence": <PulseSequence | spectrum | file path>,
    "options": {...}}`` (``options``: :func:`plot_pulse_sequence` keywords).
    A panel holds spectra of one kind: 1D together (superimposed) or 2D
    together (an overlay, 2-6 of the same nuclei) -- never both."""
    if isinstance(panels, NMRSpectrum) or isinstance(panels, dict):
        panels = [panels]
    out = []
    for p in panels:
        if isinstance(p, dict) and "pulse_sequence" in p:
            if p.get("spectra"):
                raise ValueError("A pulse-sequence panel holds a sequence, not spectra.")
            out.append({"kind": "pulse", "spectra": [], "sequence": p["pulse_sequence"],
                        "title": p.get("title"), "labels": None, "colors": None,
                        "options": dict(p.get("options") or {}), "insets": None,
                        "message": p.get("message")})
        elif isinstance(p, dict):
            spectra = p["spectra"]
            spectra = [spectra] if isinstance(spectra, NMRSpectrum) else list(spectra)
            out.append({"kind": None, "spectra": spectra, "title": p.get("title"),
                        "labels": p.get("labels"), "colors": p.get("colors"),
                        "options": dict(p.get("options") or {}), "insets": p.get("insets")})
        elif isinstance(p, NMRSpectrum):
            out.append({"kind": None, "spectra": [p], "title": None, "labels": None,
                        "colors": None, "options": {}, "insets": None})
        else:
            out.append({"kind": None, "spectra": list(p), "title": None, "labels": None,
                        "colors": None, "options": {}, "insets": None})
    for k, p in enumerate(out):
        if p["kind"] == "pulse":
            continue
        if not p["spectra"]:
            raise ValueError(f"Panel {k + 1} holds no spectra.")
        for s in p["spectra"]:
            if not isinstance(s, NMRSpectrum) or s.ndim not in (1, 2):
                raise ValueError(f"Panel {k + 1}: every entry must be a 1D or 2D NMRSpectrum.")
        kinds = {_spectrum_kind(s) for s in p["spectra"]}
        if len(kinds) > 1:
            raise ValueError(f"Panel {k + 1} mixes 1D and 2D spectra; give each kind its own panel.")
        p["kind"] = kinds.pop()
    if not out:
        raise ValueError("Nothing to plot: give at least one panel.")
    return out


def _data_y_range(ax):
    lows, highs = [], []
    for line in ax.lines:
        if not line.get_visible() or str(line.get_label()).startswith("_hidden"):
            continue
        y = np.asarray(line.get_ydata(), dtype=float)
        y = y[np.isfinite(y)]
        if y.size > 2:
            lows.append(float(y.min()))
            highs.append(float(y.max()))
    return (min(lows), max(highs)) if lows else None


def _x_link_key(p: dict, x_unit: str):
    """Which panels' shift axes mean the same thing: ``(unit, nucleus)``. A
    pulse sequence has no shift axis."""
    if p["kind"] == "pulse":
        return None
    s = p["spectra"][0]
    unit = "ppm" if p["kind"] == "2d" else str(x_unit).lower()
    return (unit, getattr(getattr(s, "metadata", None), "nucleus", None))


def _y_link_key(p: dict):
    if p["kind"] == "pulse":
        return None
    if p["kind"] == "1d":
        return "intensity"
    s = p["spectra"][0]
    meta = getattr(s, "metadata", None)
    return ("2d", getattr(meta, "nucleus_indirect", None), bool(getattr(s, "is_pseudo_2d", False)))


def _call_with(func, **kwargs):
    """``func(**kwargs)`` keeping only the keywords it accepts."""
    import inspect
    accepted = inspect.signature(func).parameters
    if any(v.kind is inspect.Parameter.VAR_KEYWORD for v in accepted.values()):
        return func(**kwargs)
    return func(**{k: v for k, v in kwargs.items() if k in accepted})


def _draw_2d_panel(ax, fig, p: dict, names, cols, twod: dict, fontsize: float) -> None:
    """One 2D panel: a contour plot of one spectrum, or a flat-colour overlay of
    several (2-6 of the same nuclei). Settings, lowest first: this function's
    defaults (no projections, a panel that fills its cell so it lines up with a
    1D beside or above it), ``twod_options`` for every 2D panel, the panel's
    own ``options``."""
    from ..plotting import twod as _twod
    opts = {"show_projections": False, "square_aspect": False, "fontsize": fontsize}
    opts.update(twod)
    opts.update(p.get("options") or {})
    spectra = p["spectra"]
    if len(spectra) == 1:
        _twod.plot_2d_contour(spectra[0], ax=ax, fig=fig, insets=p.get("insets") or None, **opts)
        return
    if len(spectra) > _twod.MAX_OVERLAY_SPECTRA:
        raise ValueError(f"An overlay takes 2 to {_twod.MAX_OVERLAY_SPECTRA} spectra "
                         f"({len(spectra)} in one panel).")
    _twod._check_overlay_compatible(spectra)
    opts["labels"] = list(names)
    opts["colours"] = list(cols)
    opts["insets"] = p.get("insets") or None
    _call_with(_twod.overlay_plot, spectra=spectra, ax=ax, fig=fig, **opts)


def _draw_pulse_panel(ax, p: dict, defaults: dict, title) -> None:
    """A pulse-sequence panel: ``p["sequence"]`` is a ``PulseSequence``, an
    open spectrum (its own sequence) or the path of a sequence file."""
    import os
    from . import pulse_sequence as ps
    seq = p["sequence"]
    if seq is None:
        # no sequence to draw: say so in the panel rather than leave it blank
        ax.set_axis_off()
        ax.text(0.5, 0.5, p.get("message") or "No pulse sequence to show", ha="center", va="center",
                color="0.4", wrap=True, transform=ax.transAxes)
        return
    if isinstance(seq, NMRSpectrum):
        seq = ps.pulse_sequence_for(seq)[0]
    elif isinstance(seq, (str, os.PathLike)):
        seq = ps.read_pulse_sequences(str(seq))[0]
    opts = dict(defaults)
    opts.update(p.get("options") or {})
    if title and "title_text" not in opts:
        opts["title_text"] = title
    ps.plot_pulse_sequence(seq, ax, **opts)


def plot_compare_panels(panels, *, layout: str = "row", ncols: int | None = None,
                        link: bool = True, link_y: bool = False, normalize: bool = False,
                        titles=None, labels=None, colors=None, legend: bool = True,
                        x_unit: str = "ppm", full_resolution: bool = False,
                        linewidth: float = 1.5, fontsize: float = 12.0, fig=None,
                        figsize=None, tight: bool = True, gap_below: float = 0.05,
                        gap_above: float = 0.08, model_for=None, axis_labels: str = "auto",
                        axis_styles=None, twod_options: dict | None = None,
                        pulse_options: dict | None = None,
                        panel_labels=None) -> ComparePanels:
    """Draw ``panels`` side by side (``layout="row"``), one above another
    (``"column"``) or in a grid (``"grid"``, ``ncols`` columns).

    Parameters
    ----------
    panels
        One entry per panel. A panel is **one axes**:

        * a 1D :class:`NMRSpectrum`, or a list of them (superimposed, one
          colour each, with a legend) -- or ``{"spectra": [...], "title": ...,
          "labels": [...], "colors": [...]}``;
        * a 2D :class:`NMRSpectrum` (contours), or a list of 2-6 with the same
          nuclei (an overlay), optionally ``{"spectra": ..., "options": {...},
          "insets": [...]}`` -- see ``twod_options``;
        * a pulse sequence, ``{"pulse_sequence": <PulseSequence | spectrum |
          file path>, "options": {...}}`` (:func:`pulse_sequence.plot_pulse_sequence`
          keywords: colours, names, fonts ...).

        1D, 2D and pulse-sequence panels can be mixed freely; a panel never
        mixes 1D and 2D spectra. Whether each dataset has an axes of its own or
        shares one is just how you group the list.
    link, link_y
        Whether the panels share the shift axis (zoom and scroll together) /
        the intensity axis. Panels link only with panels whose axis means the
        same thing: a 1D and the F2 axis of a 2D of the same nucleus move
        together; a pulse sequence never links. Switch later with
        ``result.link.set(x=..., y=...)``.
    normalize
        Scale each 1D spectrum to its own maximum (a shape comparison).
    titles
        One title per panel. Default: the spectrum's name for a panel holding
        one spectrum, none (a legend instead) for a panel holding several.
    labels, colors
        Per panel, per spectrum: legend names and colours (default: the
        spectrum's own display name and a colour-blind-safe palette).
    legend
        Draw a legend in panels holding more than one trace.
    axis_labels
        Which panels carry a shift-axis label: ``"every"``/``"all"`` (each
        panel has its own), ``"bottom"`` (only the bottom panel of each
        column, as one shared axis), or ``"auto"`` (default): every panel
        when the panels' axes read differently -- different nuclei -- and
        the bottom ones when they all read the same. A figure with any 2D or
        pulse-sequence panel gives every panel its own label under ``"auto"``.
    axis_styles
        Per-panel axis options, ``{panel: {...}}`` (0-based; ``"all"`` for
        every panel; a list works too). Keys: ``xlabel`` (text, LaTeX
        allowed; ``""`` hides it), ``show_xlabel``, ``xlabel_fontsize``,
        ``tick_direction`` (``"in"``, ``"out"``, ``"inout"``), ``tick_length``,
        ``tick_width``, ``tick_labelsize``, ``minor_ticks``, ``num_ticks``,
        ``show_tick_labels``, ``spine_width``. See :func:`apply_axis_styles`.
    twod_options
        Keywords for every 2D panel, passed to
        :func:`opennmr.plotting.twod.plot_2d_contour` (``factor``, ``clevels``,
        ``contour_linewidth``, ``show_projections``, ``square_aspect`` ...).
        Defaults here: no projections and ``square_aspect=False`` (the panel
        fills its cell, so its F2 axis lines up with a 1D above or below it).
    pulse_options
        Keywords for every pulse-sequence panel (a panel's own ``options`` win).
    panel_labels
        Add (a), (b) ... to the panels: ``True``, a style (``"(a)"``, ``"A."``
        ...) or a dict of :func:`opennmr.notebook.panel_labels.add_panel_labels`
        arguments (``style``, ``start``, ``order``, ``placement``, ``positions``,
        ``fontsize`` ...). The labels are returned by ``result.labels``.
    fig, figsize, tight
        Draw into an existing figure (the GUI does); ``tight`` runs
        ``fig.tight_layout()`` afterwards.

    Returns ``ComparePanels(fig, axes, link)`` -- unpack as ``fig, axes, link``.
    """
    import matplotlib.pyplot as plt
    from ..plotting import apply_spine_style, apply_tick_style, plot_1d, qualitative_colors, safe_xlim

    if axis_labels == "every":
        axis_labels = "all"
    if axis_labels not in AXIS_LABEL_MODES:
        raise ValueError(f"axis_labels must be one of {AXIS_LABEL_MODES + ('every',)}, not {axis_labels!r}")
    spec = _as_panels(panels)
    for k in range(len(spec)):
        _style_for(axis_styles, k)                       # validate before drawing
    label_args = _label_arguments(panel_labels)          # validate before drawing
    n = len(spec)
    kinds = [p["kind"] for p in spec]
    mixed = any(kind != "1d" for kind in kinds)
    nrows, ncols_ = grid_shape(n, layout, ncols)
    if fig is None:
        cell_w, cell_h = 3.4, 2.6
        if "2d" in kinds:
            cell_w, cell_h = 3.8, 3.4
        if "pulse" in kinds:
            cell_w = max(cell_w, 5.5)
        fig = plt.figure(figsize=figsize or (min(cell_w * ncols_, 14.0),
                                             min(cell_h * nrows + 0.6, 11.0 if not mixed else 12.0)))
    else:
        fig.clear()
    axes_grid = fig.subplots(nrows, ncols_, squeeze=False)
    axes = [axes_grid[i // ncols_][i % ncols_] for i in range(n)]
    for i in range(n, nrows * ncols_):
        axes_grid[i // ncols_][i % ncols_].set_visible(False)

    twod = dict(twod_options or {})
    pulse_defaults = dict(pulse_options or {})
    ranges: list = [None] * n                 # (low, high) of each panel's shift axis
    label_texts = []
    for k, (ax, p) in enumerate(zip(axes, spec)):
        ax._opennmr_panel_kind = p["kind"]
        m = len(p["spectra"])
        title = (titles[k] if titles is not None and k < len(titles) else p["title"])
        if p["kind"] == "pulse":
            _draw_pulse_panel(ax, p, pulse_defaults, title)
            continue
        pal = qualitative_colors(m)
        cols = (p["colors"] or (colors[k] if colors is not None and k < len(colors) else None)
                or [pal[j] for j in range(m)])
        names = (p["labels"] or (labels[k] if labels is not None and k < len(labels) else None)
                 or [_spectrum_label(s, f"Spectrum {j + 1}") for j, s in enumerate(p["spectra"])])
        if p["kind"] == "2d":
            _draw_2d_panel(ax, fig, p, names, cols, twod, fontsize)
            if title is None and m == 1:
                title = names[0]
            if title:
                ax.set_title(title, fontsize=fontsize * 0.9)
            lo, hi = sorted(ax.get_xlim())
            ranges[k] = (lo, hi)
            ax._opennmr_auto_xlabel = ax.get_xlabel()
            label_texts.append(ax.get_xlabel())
            continue
        lines = []
        xmins, xmaxs = [], []
        for s, c in zip(p["spectra"], cols):
            lo, hi, label_text = plot_1d(ax, s, x_unit=x_unit, normalize=normalize, color=c,
                                         linewidth=linewidth, full_resolution=full_resolution)
            xmins.append(lo)
            xmaxs.append(hi)
            lines.append(ax.lines[-1])
            # Which spectrum the line shows, for a unit change: spectra on
            # one panel can share a ppm grid but not a ppm-to-Hz conversion.
            ax.lines[-1]._opennmr_source_spectrum = s
        ranges[k] = (min(xmins), max(xmaxs))
        for line, name in zip(lines, names):
            line.set_label(name)
        draw_kept_fit_overlays(p["spectra"], [ax] * m, lines, list(cols), list(names),
                               normalize, model_for)
        if title is None and m == 1:
            title = names[0]
        if title:
            ax.set_title(title, fontsize=fontsize * 0.9)
        multi = m > 1 or any(str(ln.get_label()) not in ("", "_hidden data")
                             and not str(ln.get_label()).startswith("_") for ln in ax.lines[m:])
        if legend and multi:
            ax.legend(frameon=False, fontsize=fontsize * 0.75, loc="upper left")
        ax.set_yticks([])
        apply_spine_style(ax, top=False, right=False, bottom=True, left=False)
        apply_tick_style(ax, axis="x", labelsize=fontsize * 0.85)
        ax.set_xlabel(label_text, fontsize=fontsize)
        ax._opennmr_auto_xlabel = label_text
        label_texts.append(label_text)

    # Which panels' axes mean the same thing. A figure of 1D panels only keeps the
    # original rule (every panel links); once 2D or pulse-sequence panels join,
    # only a 1D and a 2D of the same nucleus and unit share a shift axis.
    if mixed:
        x_keys = [_x_link_key(p, x_unit) for p in spec]
        y_keys = [_y_link_key(p) for p in spec]
    else:
        x_keys = [0] * n
        y_keys = [0] * n

    # shift axes: the union of the range of every panel in the link group when
    # linked, each panel's own otherwise
    group_range: dict = {}
    if link:
        for r, key in zip(ranges, x_keys):
            if r is None or key is None:
                continue
            lo_, hi_ = group_range.get(key, r)
            group_range[key] = (min(lo_, r[0]), max(hi_, r[1]))
    for ax, p, r, key in zip(axes, spec, ranges, x_keys):
        if r is None:
            continue
        lo, hi = group_range.get(key, r) if link else r
        if p["kind"] == "1d":
            safe_xlim(ax, lo, hi)
            ax.invert_xaxis()
        else:
            ax.set_xlim(hi, lo)

    # intensity axes of the 1D panels: each panel's own data (plus the gap Plot
    # Style calls "below spectra"), or one range for all when the vertical scale is linked
    one_d = [ax for ax, p in zip(axes, spec) if p["kind"] == "1d"]
    y_ranges = [_data_y_range(ax) for ax in one_d]
    known = [r for r in y_ranges if r is not None]
    shared = (min(r[0] for r in known), max(r[1] for r in known)) if known and link_y else None
    for ax, r in zip(one_d, y_ranges):
        r = shared or r
        if r is None:
            continue
        span = (r[1] - r[0]) or abs(r[1]) or 1.0
        ax.set_ylim(r[0] - gap_below * span, r[1] + gap_above * span)
        ax._opennmr_fitted_ylim = ax.get_ylim()
    if link_y:
        # 2D panels with the same kind of F1 axis share their vertical range
        by_key: dict = {}
        for ax, p, key in zip(axes, spec, y_keys):
            if p["kind"] == "2d":
                by_key.setdefault(key, []).append(ax)
        for group in by_key.values():
            if len(group) < 2:
                continue
            ys = [v for ax in group for v in ax.get_ylim()]
            for ax in group:
                inverted = ax.get_ylim()[0] > ax.get_ylim()[1]
                ax.set_ylim((max(ys), min(ys)) if inverted else (min(ys), max(ys)))

    # shared-axis convention: only the bottom panel of each column carries the
    # label -- unless the panels' axes read differently (different nuclei), a 2D
    # or pulse-sequence panel is among them, or every panel was asked for, when
    # each keeps its own
    each = (axis_labels == "all" or (axis_labels == "auto" and (mixed or len(set(label_texts)) > 1)))
    if layout in ("grid", "column") and not each:
        for i, ax in enumerate(axes):
            if i + ncols_ < n:
                ax.set_xlabel("")
    apply_axis_styles(axes, axis_styles, fontsize)
    controller = PanelLink(axes, x=link, y=link_y, x_keys=x_keys, y_keys=y_keys)
    fig._opennmr_panel_link = controller            # the callbacks hold only weak references
    labels_obj = None
    if label_args is not None:
        from .panel_labels import add_panel_labels
        label_args.setdefault("fontsize", fontsize * 1.15)
        labels_obj = add_panel_labels(fig, axes, **label_args)
    elif getattr(fig, "_opennmr_panel_labels", None) is not None:
        fig._opennmr_panel_labels = None
    if tight:
        def lay_out():
            try:
                fig.tight_layout()
            except Exception:  # noqa: BLE001 -- layout is cosmetic
                pass
        lay_out()
        if labels_obj is not None:
            labels_obj.settle(lay_out)
    return ComparePanels(fig, axes, controller)


def _label_arguments(panel_labels):
    """``panel_labels`` as the keyword dict for ``add_panel_labels`` (or None)."""
    if panel_labels in (None, False):
        return None
    if panel_labels is True:
        return {}
    if isinstance(panel_labels, str):
        return {"style": panel_labels}
    if isinstance(panel_labels, dict):
        return dict(panel_labels)
    raise ValueError("panel_labels must be True, a style such as '(a)', or a dict of options.")


NMR1D_multi_plot = plot_compare_panels
