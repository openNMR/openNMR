"""
Pictures of the interaction tensors of a site: CSA, quadrupolar (EFG),
their relative orientation, and what J does.

Two kinds of picture, because a tensor can be drawn two ways and they
answer different questions (explicit user request, 2026-09-27: "visualise
the different tensors (CSA, quadrupolar) being simulated -- as
ellipsoids? ... an isotropic shift would be a sphere of radius delta_iso"):

**Frequency surface** (:func:`plot_tensor_surface`) -- for every
direction of B0 in the site frame, the line's frequency, drawn as the
distance from the origin in that direction and coloured by its value.
This is *the* picture of what the spectrum is made of: a powder pattern
is the histogram of this surface's radius over all directions. It is
computed by :func:`opennmr.notebook.combined_hamiltonian.site_transition_frequencies`,
the same engine that simulates the spectrum, so it cannot disagree with
the simulated lineshape.

* Isotropic shift alone: a sphere of radius |delta_iso| (every direction
  gives the same frequency).
* CSA: delta(b) = delta_iso + b.D.b. This is **not** an ellipsoid -- it is
  the tensor's "ovaloid" (representation quadric), which touches the
  ellipsoid only along the principal axes. With ``relative_to='iso'`` only
  the anisotropic part b.D.b is drawn: a two-lobed (d_z2-like) shape,
  red where the shift is above delta_iso and blue where below.
* Quadrupolar, first order (satellite transitions, or integer spins):
  the same kind of rank-2 lobes, scaled by the transition's m. Central
  transition of a half-integer spin: first order vanishes and the
  second-order term has rank-4 symmetry -- more lobes.
* CSA and quadrupolar together: their frequencies add, direction by
  direction, with the CSA tensor rotated into the EFG frame by the site's
  Euler angles -- the surface shows the combined orientation dependence,
  which is exactly what the combined simulation powder-averages.
* **J** (isotropic, as the simulator treats it) does not depend on
  direction, so it does not change any shape. It splits every line: the
  surface is copied at each multiplet offset (radius + m J / nu0), all the
  same shape. At chemical-shift scale these copies are usually far too
  close to see, and the notes say so. (J does have an anisotropic part,
  but it has the form of the dipolar coupling and cannot be separated
  from it; the simulator does not include it.)

**Tensor glyphs** (:func:`plot_tensor_ellipsoids`) -- the conventional
picture of the tensors themselves, in one frame:

* CSA: the ellipsoid with semi-axes |delta_11|, |delta_22|, |delta_33|
  along its principal axes. It is a faithful glyph only when all three
  principal values have the same sign; otherwise the note says so and
  the negative axes are marked.
* EFG: traceless (V_xx + V_yy + V_zz = 0), so it can never be an
  ellipsoid -- its principal values always have mixed signs. It is drawn
  as its three principal axes, length proportional to |V_ii| (scaled to
  the picture, values in the legend, C_Q units), red positive, blue
  negative.
* Isotropic shift only: the sphere of radius |delta_iso|.
The CSA ellipsoid is drawn rotated by the site's Euler angles in the EFG
frame, so their relative orientation is what is seen.
"""

from __future__ import annotations

import numpy as np

from . import combined_hamiltonian as ch
from .simulations import _select_transitions, combine_jcouplings, csa_definition

__all__ = ["default_transition", "transition_labels", "frequency_surface", "tensor_components",
           "plot_tensor_surface", "plot_tensor_ellipsoids", "plot_tensors"]

#: Larmor frequency used when none is given and none is needed (a site
#: with no quadrupolar interaction: CSA, shift and the surface in ppm do
#: not depend on it).
_NOMINAL_LARMOR_HZ = 100e6

_POSITIVE, _NEGATIVE = "#b2182b", "#2166ac"          # RdBu ends: colour-blind distinguishable


def transition_labels(site):
    """The labels of the site's transitions, as the simulator names them."""
    site = ch.as_sites(site)[0]
    return [r["label"] for r in _select_transitions(site.spin, "all")]


def default_transition(site):
    """The transition a surface is drawn for when none is chosen: the
    central transition when there is one (spin-1/2, half-integer spins),
    otherwise the first single-quantum transition (integer spins)."""
    site = ch.as_sites(site)[0]
    rows = _select_transitions(site.spin, "ct") if site.spin % 1 else []
    return (rows or _select_transitions(site.spin, "all"))[0]["label"]


def _larmor(site, larmor_Hz):
    if larmor_Hz:
        return float(larmor_Hz)
    if site.get("quadrupolar"):
        raise ValueError("A quadrupolar frequency in ppm depends on the field: give larmor_Hz.")
    return _NOMINAL_LARMOR_HZ


def _grid(n_theta, n_phi):
    theta = np.linspace(0.0, np.pi, int(n_theta))
    phi = np.linspace(0.0, 2.0 * np.pi, int(n_phi))
    return np.meshgrid(theta, phi, indexing="ij")


def frequency_surface(site, larmor_Hz=None, transition=None, order="second", relative_to="zero",
                      n_theta=73, n_phi=145):
    """The line's frequency (ppm) for every direction of B0 in the site
    frame, for one transition.

    ``transition`` : a label from :func:`transition_labels` (default
    :func:`default_transition`). ``relative_to`` : 'zero' (the full
    frequency, delta_iso included) or 'iso' (minus delta_iso: the
    anisotropic part only). ``order`` : as the simulator's
    ('first', 'second', 'exact').

    Returns ``{"theta", "phi", "value" (ppm, n_theta x n_phi), "transition",
    "relative_to", "iso", "larmor_Hz"}``."""
    site = ch.as_sites(site)[0]
    if relative_to not in ("zero", "iso"):
        raise ValueError("relative_to must be 'zero' or 'iso'.")
    larmor = _larmor(site, larmor_Hz)
    labels = transition_labels(site)
    transition = transition or default_transition(site)
    if transition not in labels:
        raise ValueError(f"{site.label} (I = {site.spin:g}) has no transition {transition!r}; "
                         f"choose one of {labels}.")
    theta, phi = _grid(n_theta, n_phi)
    res = ch.site_transition_frequencies(site, larmor, theta.ravel(), phi.ravel(), order=order,
                                         transitions="all")
    column = res["label"].index(transition)
    value = np.asarray(res["offset_Hz"])[:, column].reshape(theta.shape) / larmor * 1e6
    if relative_to == "iso":
        value = value - site.delta_iso
    return {"theta": theta, "phi": phi, "value": value, "transition": transition,
            "relative_to": relative_to, "iso": site.delta_iso, "larmor_Hz": larmor}


def tensor_components(site):
    """The site's tensors in its frame (the EFG frame when there is a
    quadrupolar interaction, else the CSA frame), for drawing.

    ``{"iso": delta_iso (ppm),
       "csa": None | {"principal": (d_xx, d_yy, d_zz) ppm incl. delta_iso,
                      "axes": 3x3, columns the principal axes},
       "efg": None | {"principal": (V_xx, V_yy, V_zz) in C_Q units (MHz; V_zz = C_Q),
                      "axes": identity},
       "j": [{"nuclide", "n", "J_Hz"}, ...]}``"""
    site = ch.as_sites(site)[0]
    out = {"iso": site.delta_iso, "csa": None, "efg": None,
           "j": [{k: g[k] for k in ("nuclide", "n", "J_Hz")} for g in site.get("jcoupling")]}
    csa = site.get("csa")
    if csa:
        c = csa[0]
        principal = np.diag(csa_definition(site.delta_iso, c["aniso"], c["asym"],
                                           convention=c["convention"]))
        axes = ch._rotation_zyz(*c["euler"]) if site.get("quadrupolar") else np.eye(3)
        out["csa"] = {"principal": tuple(float(v) for v in principal), "axes": axes}
    quad = site.get("quadrupolar")
    if quad:
        cq, eta = quad[0]["C_Q"], quad[0]["eta_Q"]
        out["efg"] = {"principal": (-0.5 * cq * (1 - eta), -0.5 * cq * (1 + eta), cq),
                      "axes": np.eye(3)}
    return out


def _j_offsets_ppm(site, larmor):
    groups = [{k: v for k, v in g.items() if k != "type"} for g in site.get("jcoupling")]
    if not groups:
        return np.array([0.0]), np.array([1.0])
    f, w = combine_jcouplings(groups)
    return np.asarray(f, dtype=float) / larmor * 1e6, np.asarray(w, dtype=float)


#: How a tensor picture is drawn, unless told otherwise -- the keys
#: ``tensor_style`` takes (the GUI's Tensor Plot Style dialog edits them;
#: a notebook passes a dict with any of them). Every default colour is
#: colour-blind friendly: viridis, the diverging RdBu (ColorBrewer's
#: colour-blind-safe red-blue) and its two ends.
TENSOR_STYLE_DEFAULTS = {
    "colormap": None,          # None: RdBu_r for a surface of both signs, viridis otherwise
    "reverse": False,          # run the colour scale the other way
    "colorbar": True,          # the colour bar beside a frequency surface
    "show_axes": True,         # principal axes through the origin
    "grid": True,              # the 3D box's gridlines
    "surface_alpha": 1.0,      # opacity of the frequency surface
    "elev": 22.0,              # viewing elevation (degrees)
    "azim": 35.0,              # viewing azimuth (degrees)
    "positive_colour": "#b2182b",
    "negative_colour": "#2166ac",
    "ellipsoid_colour": "#8c8c8c",
    "ellipsoid_alpha": 0.35,
}


def tensor_style(overrides=None) -> dict:
    """:data:`TENSOR_STYLE_DEFAULTS` with ``overrides`` (unknown keys and
    ``None`` values ignored) laid over it."""
    style = dict(TENSOR_STYLE_DEFAULTS)
    for key, value in (overrides or {}).items():
        if key in style and (value is not None or key == "colormap"):
            style[key] = value
    return style


#: The merge behind :func:`tensor_style`, under a name the plotting
#: functions' own ``tensor_style`` argument does not shadow.
_merge_style = tensor_style


def _new_3d_axes(figure):
    import matplotlib.pyplot as plt
    if figure is None:
        figure = plt.figure(figsize=(5.0, 4.6))
    else:
        figure.clear()
    return figure, figure.add_subplot(111, projection="3d")


def _finish_3d(ax, extent, fontsize, unit_label, style=None):
    """Equal axes and room for the labels. Not tight_layout: it does not
    account for a 3D axes' labels and clipped the z label."""
    style = _merge_style(style)
    extent = float(extent) or 1.0
    ax.figure.subplots_adjust(left=0.08, right=0.86, bottom=0.04, top=0.92)
    for setter in (ax.set_xlim, ax.set_ylim, ax.set_zlim):
        setter(-extent, extent)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel(f"x ({unit_label})", fontsize=fontsize * 0.85, labelpad=2)
    ax.set_ylabel(f"y ({unit_label})", fontsize=fontsize * 0.85, labelpad=2)
    ax.set_zlabel(f"z ({unit_label})", fontsize=fontsize * 0.85, labelpad=2)
    # Tick labels the same size as the axis labels (user request, 2026-10-01).
    ax.tick_params(labelsize=fontsize * 0.85, pad=0)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_alpha(0.05)
    ax.grid(bool(style["grid"]))
    ax.view_init(elev=float(style["elev"]), azim=float(style["azim"]))
    fit_3d_ticks(ax, fontsize * 0.85)


#: Tick steps a thinned axis may use: the usual nice ones plus 4, so
#: thinning goes 25 -> 40 -> 50 (7, 5, 3 ticks on +-96) rather than
#: jumping from 7 ticks straight to 3.
_TICK_STEPS = (1, 2, 2.5, 4, 5, 10)


def fit_3d_ticks(ax, fontsize=None, max_ticks=8, font_family=None, label_pad=True):
    """As many ticks on each axis of the 3D ``ax`` as fit without their
    labels overlapping, at the current view, limits and text size.

    User request (2026-10-01): "the axis tick labels can end up
    overlapping each other when the font is large ... decreasing the
    number of ticks on increasing font size". For each axis this
    projects the axis onto the screen (its on-screen length and
    direction depend on the view), measures each candidate set of tick
    labels as they would be rendered (font, size, the formatter's own
    text), and keeps the most ticks whose neighbouring labels clear one
    another -- along the projected direction, plus a gap of 0.15 em. So
    the count also changes with the rotation: an axis seen end-on gets
    fewer ticks. End ticks too close to a corner of the box (where the
    neighbouring axis's end label sits) are left out. With ``label_pad``
    each axis label is moved clear of its tick labels as measured.
    Returns ``{"x": n, "y": n, "z": n}``, the number of ticks each axis
    now shows. The ticks are fixed values, so call again after rotating
    or zooming (the GUI does so on every mouse release).
    """
    from matplotlib.font_manager import FontProperties
    from matplotlib.ticker import FixedLocator, MaxNLocator
    from mpl_toolkits.mplot3d import proj3d

    figure = ax.figure
    if fontsize is None:
        labels = ax.xaxis.get_ticklabels()
        fontsize = labels[0].get_fontsize() if labels else 10.0
    fontsize = float(fontsize)
    prop = FontProperties(size=fontsize, family=font_family) if font_family else FontProperties(size=fontsize)
    renderer = figure.canvas.get_renderer() if hasattr(figure.canvas, "get_renderer") else None
    to_px = fontsize * figure.dpi / 72.0

    from matplotlib.text import Text

    def text_size(text):
        # Measured as a Text artist, the way a tick label is drawn: its
        # box includes the font's full line height, not only the glyphs.
        if renderer is not None:
            artist = Text(0, 0, text, fontproperties=prop)
            artist.set_figure(figure)
            box = artist.get_window_extent(renderer)
            return box.width, box.height
        return 0.62 * to_px * len(text), 1.4 * to_px    # no renderer: an em-based estimate

    M = ax.get_proj()
    limits = (ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d())

    def screen(point):
        x, y, _ = proj3d.proj_transform(*point, M)
        return ax.transData.transform((x, y))

    counts = {}
    for k, (name, axis) in enumerate(zip("xyz", (ax.xaxis, ax.yaxis, ax.zaxis))):
        lo, hi = sorted(float(v) for v in limits[k])
        if not hi > lo:
            continue
        # The four parallel box edges project to slightly different
        # lengths and angles (perspective, a few per cent); their mean
        # stands for the one the ticks are drawn on.
        others = [limits[j] for j in range(3) if j != k]
        vecs = []
        for a in others[0]:
            for b in others[1]:
                start, end = [0.0] * 3, [0.0] * 3
                start[k], end[k] = lo, hi
                rest = [j for j in range(3) if j != k]
                start[rest[0]] = end[rest[0]] = a
                start[rest[1]] = end[rest[1]] = b
                vecs.append(screen(end) - screen(start))
        lengths = [float(np.hypot(*v)) for v in vecs]
        length = float(np.mean(lengths))
        vec = np.mean([v / n for v, n in zip(vecs, lengths) if n > 0] or [np.zeros(2)], axis=0) * length
        if length <= 0:
            continue
        cos, sin = abs(vec[0]) / length, abs(vec[1]) / length
        gap = 0.15 * to_px
        formatter = axis.get_major_formatter()

        def clearance(w, h):
            """How far a label must move along the axis to clear an
            identical one: past it in x or in y, whichever comes first."""
            return min((w + gap) / cos if cos > 1e-6 else np.inf,
                       (h + gap) / sin if sin > 1e-6 else np.inf)

        fallback = partial = chosen = single = None
        for nbins in range(int(max_ticks), 0, -1):
            locator = MaxNLocator(nbins=nbins, steps=list(_TICK_STEPS))
            span = hi - lo
            ticks = [t for t in locator.tick_values(lo, hi) if lo - 1e-9 * span <= t <= hi + 1e-9 * span]
            if not ticks:
                continue
            if len(ticks) == 1:
                # One label (the middle one) when nothing else fits --
                # never two that collide.
                single = single or (ticks, [text_size(formatter.format_ticks(ticks)[0]
                                                      if hasattr(formatter, "format_ticks")
                                                      else f"{ticks[0]:g}")])
                continue
            try:
                texts = formatter.format_ticks(ticks)
            except Exception:  # a formatter not bound to this axis
                texts = [f"{t:g}" for t in ticks]
            sizes = [text_size(s) for s in texts]
            spacing = length * (ticks[1] - ticks[0]) / span
            # Neighbouring labels clear each other ...
            if any(spacing < clearance(0.5 * (w1 + w2), 0.5 * (h1 + h2))
                   for (w1, h1), (w2, h2) in zip(sizes, sizes[1:])):
                continue
            if fallback is None:
                fallback = (ticks, sizes)
            # ... and the end ones keep clear of the box corners, where
            # the next axis's end label sits.
            # A set that clears them whole is preferred to one with an
            # end tick dropped, which leaves the axis lopsided.
            kept = [(t, s) for t, s in zip(ticks, sizes)
                    if min(t - lo, hi - t) * length / span >= 0.5 * clearance(*s)]
            if len(kept) == len(ticks):
                chosen = (ticks, sizes)
                break
            if partial is None and len(kept) >= 2:
                partial = ([t for t, _s in kept], [s for _t, s in kept])
        if chosen is None and partial is None and fallback is None and lo < 0 < hi:
            # Before one lone label: a symmetric nice pair (+-50 on +-96),
            # which still shows the scale -- if it fits.
            edge = min(-lo, hi)
            for a in sorted({m * 10.0 ** e for m in _TICK_STEPS
                             for e in range(int(np.floor(np.log10(edge))) - 1,
                                            int(np.floor(np.log10(edge))) + 1)}, reverse=True):
                if not (0.3 * edge <= a <= edge):
                    continue
                pair = [-a, a]
                try:
                    texts = formatter.format_ticks(pair)
                except Exception:
                    texts = [f"{v:g}" for v in pair]
                sizes = [text_size(s) for s in texts]
                if (2 * a * length / span >= clearance(*np.max(sizes, axis=0))
                        and all((edge - a) * length / span >= 0.5 * clearance(*s) for s in sizes)):
                    chosen = (pair, sizes)
                    break
        chosen = chosen or partial or fallback or single
        if chosen is None:          # not even one nice value in range
            mid = 0.5 * (lo + hi)
            chosen = ([mid], [text_size(f"{mid:.3g}")])
        ticks, sizes = chosen
        axis.set_major_locator(FixedLocator(ticks))
        counts[name] = len(ticks)
        if label_pad:
            # The axis label clears its tick labels: their extent across
            # the axis on screen (a box w x h seen across a line at
            # angle (cos, sin) spans w*sin + h*cos), in points.
            # Half of it: mplot3d already offsets the label outward from
            # the tick-label row (found by rendering at 8-32 pt).
            across = max(w * sin + h * cos for w, h in sizes)
            axis.labelpad = 0.5 * across * 72.0 / figure.dpi + 0.2 * fontsize
    return counts


def _frame_rotation(frame):
    """Site frame → display frame (identity without a frame)."""
    if not frame or frame.get("rotation") is None:
        return np.eye(3)
    return np.asarray(frame["rotation"], dtype=float)


def _draw_frame_axes(ax, frame, extent, fontsize):
    """The display frame's reference axes (a, b, c of a cell), as arrows
    from the origin, equal length (directions only), in dark grey."""
    if not frame:
        return
    vectors = [(np.asarray(v, dtype=float), name) for v, name in frame.get("axes") or []]
    if not vectors:
        return
    for v, name in vectors:
        # Directions only, all the same length: the cell's lengths are in
        # Å and the picture in ppm or MHz, so a length would mean nothing.
        tip = v / (np.linalg.norm(v) or 1.0) * 0.95 * extent
        ax.quiver(0, 0, 0, *tip, color="0.2", linewidth=1.4, arrow_length_ratio=0.08)
        ax.text(*(1.07 * tip), name, color="0.2", fontsize=fontsize, fontweight="bold",
                fontstyle="italic")


def _draw_axes_lines(ax, axes, lengths, names, colours, linestyle="-", fontsize=10):
    for k in range(3):
        v = axes[:, k] * lengths[k]
        ax.plot([-v[0], v[0]], [-v[1], v[1]], [-v[2], v[2]], linestyle, color=colours[k],
                linewidth=1.6)
        ax.text(*(1.08 * v), names[k], color=colours[k], fontsize=fontsize)


def plot_tensor_surface(site, larmor_Hz=None, transition=None, order="second",
                        relative_to="zero", figure=None, fontsize=12.0, n_theta=73, n_phi=145,
                        show_axes=None, show_j=True, cmap=None, tensor_style=None, frame=None):
    """The frequency surface of one transition (see the module
    docstring): radius |frequency| in each direction of B0, coloured by
    the signed frequency (ppm) with a colour bar. Principal axes of the
    CSA (solid) and EFG (dashed) are drawn through the origin.

    ``cmap`` : the colour scale (any matplotlib name); default a
    diverging ``RdBu_r`` centred on zero for a surface of both signs,
    ``viridis`` otherwise.
    ``tensor_style`` : see :data:`TENSOR_STYLE_DEFAULTS` (colour map,
    reversed, colour bar, axes, opacity, view); ``cmap``/``show_axes``
    given explicitly win over it.
    ``frame`` : draw in another frame than the site's --
    ``{"rotation": R (site → display), "axes": [(vector, label), ...],
    "title": str}``, e.g. a crystal's cell (see
    :func:`opennmr.notebook.dft_nmr.display_frame`). The axes are drawn as
    arrows from the origin.

    Returns ``(figure, notes)`` -- ``notes`` a list of plain-language
    remarks (what is drawn, J, signs)."""
    import matplotlib.pyplot as plt
    from matplotlib import cm, colors

    look = _merge_style(tensor_style)
    cmap = cmap or look["colormap"]
    show_axes = look["show_axes"] if show_axes is None else show_axes
    site = ch.as_sites(site)[0]
    surf = frequency_surface(site, larmor_Hz, transition, order, relative_to, n_theta, n_phi)
    theta, phi, value = surf["theta"], surf["phi"], surf["value"]
    r = np.abs(value)
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    vmin, vmax = float(np.min(value)), float(np.max(value))
    signed = vmin < 0 < vmax or relative_to == "iso"
    if signed:
        lim = max(abs(vmin), abs(vmax)) or 1.0
        norm, cmap = colors.TwoSlopeNorm(0.0, -lim, lim), plt.get_cmap(cmap or "RdBu_r")
        if look["reverse"]:
            cmap = cmap.reversed()
    else:
        if vmax - vmin <= 1e-6 * max(abs(vmax), 1.0):
            # One value everywhere (isotropic): a readable colour bar
            # around it, not matplotlib's "1e-9 + 1.2e2" offset.
            pad = max(1.0, 0.01 * abs(vmax))
            vmin, vmax = vmin - pad, vmax + pad
        norm = colors.Normalize(vmin, vmax)
        cmap = plt.get_cmap(cmap or "viridis")
        if look["reverse"]:
            cmap = cmap.reversed()
    figure, ax = _new_3d_axes(figure)
    rot = _frame_rotation(frame)
    if frame:
        x, y, z = np.einsum("ij,j...->i...", rot, np.stack([x, y, z]))
    ax.plot_surface(x, y, z, facecolors=cmap(norm(value)), rstride=1, cstride=1, linewidth=0,
                    antialiased=True, shade=False, alpha=float(look["surface_alpha"]))
    extent = 1.2 * float(np.max(r))
    notes = []
    comps = tensor_components(site)
    larmor = surf["larmor_Hz"]

    if show_j and relative_to == "zero" and comps["j"]:
        offsets, weights = _j_offsets_ppm(site, larmor)
        spread = float(np.ptp(offsets))
        if spread > 0.01 * float(np.max(r)):
            for f, w in zip(offsets, weights):
                if abs(f) < 1e-12:
                    continue
                rr = np.abs(value + f)
                wx, wy, wz = np.einsum("ij,j...->i...", rot, np.stack([
                    rr * np.sin(theta) * np.cos(phi), rr * np.sin(theta) * np.sin(phi),
                    rr * np.cos(theta)]))
                ax.plot_wireframe(wx, wy, wz, rstride=6, cstride=6, color="0.3",
                                  linewidth=0.3, alpha=float(0.25 + 0.5 * w / weights.max()))
            notes.append(f"J: the surface is copied at each multiplet offset (spread "
                         f"{spread:.3g} ppm), drawn as wireframes -- same shape, J does not depend "
                         "on direction.")
        else:
            notes.append(f"J splits every line into {len(offsets)} copies of this surface, "
                         f"{spread:.2g} ppm apart at most ({spread * larmor * 1e-6:.3g} Hz) -- the "
                         "same shape, far too close to see here. J does not depend on direction, "
                         "so it changes no shape.")
    elif comps["j"] and relative_to == "iso":
        notes.append("J does not depend on direction: it shifts copies of the whole surface, "
                     "which the anisotropic-part view (δ − δiso) leaves out.")

    if show_axes:
        if comps["efg"] is not None:
            _draw_axes_lines(ax, rot @ comps["efg"]["axes"], [extent / 1.15] * 3,
                             ["x$_{EFG}$", "y$_{EFG}$", "z$_{EFG}$"], ["0.35"] * 3, "--",
                             fontsize * 0.75)
        if comps["csa"] is not None:
            _draw_axes_lines(ax, rot @ comps["csa"]["axes"], [extent / 1.1] * 3,
                             ["δ$_{xx}$", "δ$_{yy}$", "δ$_{zz}$"], ["black"] * 3, "-",
                             fontsize * 0.75)
    _draw_frame_axes(ax, frame, extent, fontsize * 0.85)

    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(value)
    # Kept on the figure even when the bar is not drawn: it is what says
    # which colour scale the surface uses.
    figure.tensor_mappable = mappable
    if look["colorbar"]:
        bar = figure.colorbar(mappable, ax=ax, shrink=0.65, pad=0.08)
        bar.set_label("δ − δ$_{iso}$ (ppm)" if relative_to == "iso" else "δ (ppm)",
                      fontsize=fontsize * 0.85)
        bar.ax.tick_params(labelsize=fontsize * 0.85)
        bar.formatter.set_useOffset(False)
        bar.update_ticks()
    _finish_3d(ax, extent, fontsize, "ppm", look)
    what = [f"{site.label}, {surf['transition']}"]
    if site.get("quadrupolar"):
        what.append(f"{larmor / 1e6:.4g} MHz, {order} order")
    if frame and frame.get("title"):
        what.append(frame["title"])
    figure.suptitle(" · ".join(what), fontsize=fontsize * 0.9)

    where = (frame.get("title") or "the chosen frame") if frame else "site frame"
    notes.insert(0, f"Radius = |frequency| for B0 along that direction ({where}); colour = the "
                    "signed value. A powder pattern is the distribution of this radius over all "
                    "directions.")
    if relative_to == "zero" and not site.interactions:
        notes.append(f"Isotropic shift only: a sphere of radius |δiso| = {abs(site.delta_iso):g} ppm.")
    if signed and relative_to == "zero":
        notes.append("The frequency changes sign across the surface: red and blue lobes have "
                     "opposite signs (radius is the magnitude).")
    return figure, notes


def _ellipsoid(ax, centre, semi_axes, axes, colour, alpha):
    u, v = np.meshgrid(np.linspace(0, 2 * np.pi, 60), np.linspace(0, np.pi, 30))
    pts = np.stack([np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v)], -1) * semi_axes
    pts = pts @ axes.T + centre
    ax.plot_surface(pts[..., 0], pts[..., 1], pts[..., 2], color=colour, alpha=alpha,
                    linewidth=0, shade=True)


def plot_tensor_ellipsoids(site, figure=None, fontsize=12.0, tensor_style=None, frame=None):
    """The tensors as glyphs in one frame (see the module docstring): the
    CSA ellipsoid (semi-axes |delta_ii|), the EFG's principal axes (length
    proportional to |V_ii|, red positive, blue negative), and the sphere of
    radius |delta_iso| for a site with neither. Returns ``(figure, notes)``.
    ``tensor_style`` : see :data:`TENSOR_STYLE_DEFAULTS` (the positive,
    negative and ellipsoid colours, the axes, the view). ``frame`` : as
    :func:`plot_tensor_surface` -- the glyphs drawn in another frame, with
    its axes."""
    from matplotlib.lines import Line2D

    look = _merge_style(tensor_style)
    positive, negative = look["positive_colour"], look["negative_colour"]
    shell, shell_alpha = look["ellipsoid_colour"], float(look["ellipsoid_alpha"])
    site = ch.as_sites(site)[0]
    comps = tensor_components(site)
    figure, ax = _new_3d_axes(figure)
    rot = _frame_rotation(frame)
    notes, handles = [], []
    extent = 1.0
    if comps["csa"] is not None:
        principal = np.array(comps["csa"]["principal"])
        axes = rot @ comps["csa"]["axes"]
        semi = np.maximum(np.abs(principal), 1e-9)
        _ellipsoid(ax, np.zeros(3), semi, axes, shell, shell_alpha)
        names = ["δ$_{xx}$", "δ$_{yy}$", "δ$_{zz}$"]
        for k in range(3):
            colour = positive if principal[k] >= 0 else negative
            v = axes[:, k] * semi[k]
            if look["show_axes"]:
                ax.plot([-v[0], v[0]], [-v[1], v[1]], [-v[2], v[2]], color=colour, linewidth=2.0)
            ax.text(*(1.1 * v), f"{names[k]} = {principal[k]:.4g}", color=colour,
                    fontsize=fontsize * 0.75)
        extent = 1.25 * float(semi.max())
        handles.append(Line2D([], [], color=shell, linewidth=6, alpha=0.5,
                              label="CSA ellipsoid (ppm)"))
        notes.append("CSA ellipsoid: semi-axes along the principal axes, δxx, δyy, δzz = "
                     + ", ".join(f"{v:.4g}" for v in principal) + " ppm.")
        if principal.min() < 0 < principal.max():
            notes.append("The CSA principal values have mixed signs, so no ellipsoid represents "
                         "the tensor exactly: the semi-axes are |δii|, and the negative ones are "
                         "drawn blue. The frequency surface shows the signs properly.")
        if abs(site.delta_iso) > 5 * float(np.ptp(principal)):
            notes.append("δiso dominates the anisotropy, so the ellipsoid is nearly a sphere; the "
                         "frequency surface with 'δ − δiso' shows the anisotropy itself.")
    elif comps["efg"] is None:
        r = abs(site.delta_iso) or 1.0
        _ellipsoid(ax, np.zeros(3), np.array([r, r, r]), np.eye(3), shell, shell_alpha)
        extent = 1.25 * r
        notes.append(f"Isotropic shift only: a sphere of radius |δiso| = {abs(site.delta_iso):g} ppm.")
        handles.append(Line2D([], [], color=shell, linewidth=6, alpha=0.5,
                              label="isotropic shift (ppm)"))
    if comps["efg"] is not None:
        principal = np.array(comps["efg"]["principal"])
        scale = (0.9 * extent / 1.25) / (np.max(np.abs(principal)) or 1.0)
        if comps["csa"] is None:
            scale, extent = 1.0 / (np.max(np.abs(principal)) or 1.0), 1.3
        names = ["V$_{xx}$", "V$_{yy}$", "V$_{zz}$"]
        for k in range(3):
            colour = positive if principal[k] >= 0 else negative
            v = (rot @ comps["efg"]["axes"])[:, k] * abs(principal[k]) * scale
            ax.plot([-v[0], v[0]], [-v[1], v[1]], [-v[2], v[2]], "--", color=colour, linewidth=2.2)
            if abs(principal[k]) > 0:
                ax.text(*(1.1 * v), f"{names[k]} = {principal[k]:.3g} MHz", color=colour,
                        fontsize=fontsize * 0.75)
        handles.append(Line2D([], [], color="0.3", linestyle="--",
                              label="EFG principal axes (C$_Q$ units)"))
        notes.append("The EFG is traceless (Vxx + Vyy + Vzz = 0): its principal values always have "
                     "mixed signs, so it is drawn as its axes (red positive, blue negative), not an "
                     "ellipsoid. Its length scale is separate from the CSA's (MHz, not ppm).")
        if comps["csa"] is not None:
            e = site.get("csa")[0]["euler"]
            where = "relative to the EFG" if frame else "in the EFG frame"
            notes.append(f"The CSA ellipsoid is shown {where}, rotated by the Euler angles "
                         "(α, β, γ) = ({:g}, {:g}, {:g})°.".format(*e))
    if comps["j"]:
        text = ", ".join(f"{g['n']}×{g['nuclide']} {g['J_Hz']:g} Hz" for g in comps["j"])
        notes.append(f"J ({text}) is isotropic in this simulation: it splits lines but has no "
                     "orientation, so it has no glyph.")
    _draw_frame_axes(ax, frame, extent, fontsize * 0.85)
    if frame and frame.get("axes"):
        handles.append(Line2D([], [], color="0.2", label="cell axes (direction only)"))
    if handles:
        ax.legend(handles=handles, loc="upper left", fontsize=fontsize * 0.7, frameon=False)
    _finish_3d(ax, extent, fontsize, "ppm" if comps["csa"] is not None or comps["efg"] is None
               else "rel.", look)
    if frame:
        figure.suptitle(f"{site.label}: tensors in the {frame.get('title') or 'chosen frame'}",
                        fontsize=fontsize * 0.9)
    else:
        figure.suptitle(f"{site.label}: tensors in the "
                        f"{'EFG' if comps['efg'] is not None else 'CSA'} frame",
                        fontsize=fontsize * 0.9)
    return figure, notes


def plot_tensors(site, larmor_Hz=None, style="surface", **kwargs):
    """``style`` 'surface' (:func:`plot_tensor_surface`) or 'ellipsoid'
    (:func:`plot_tensor_ellipsoids`). Returns ``(figure, notes)``."""
    if style == "surface":
        return plot_tensor_surface(site, larmor_Hz, **kwargs)
    if style == "ellipsoid":
        kwargs = {k: v for k, v in kwargs.items()
                  if k in ("figure", "fontsize", "tensor_style", "frame")}
        return plot_tensor_ellipsoids(site, **kwargs)
    raise ValueError("style must be 'surface' or 'ellipsoid'.")
