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


def _new_3d_axes(figure):
    import matplotlib.pyplot as plt
    if figure is None:
        figure = plt.figure(figsize=(5.0, 4.6))
    else:
        figure.clear()
    return figure, figure.add_subplot(111, projection="3d")


def _finish_3d(ax, extent, fontsize, unit_label):
    """Equal axes and room for the labels. Not tight_layout: it does not
    account for a 3D axes' labels and clipped the z label."""
    extent = float(extent) or 1.0
    ax.figure.subplots_adjust(left=0.08, right=0.86, bottom=0.04, top=0.92)
    for setter in (ax.set_xlim, ax.set_ylim, ax.set_zlim):
        setter(-extent, extent)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel(f"x ({unit_label})", fontsize=fontsize * 0.85, labelpad=2)
    ax.set_ylabel(f"y ({unit_label})", fontsize=fontsize * 0.85, labelpad=2)
    ax.set_zlabel(f"z ({unit_label})", fontsize=fontsize * 0.85, labelpad=2)
    ax.tick_params(labelsize=fontsize * 0.7, pad=0)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_alpha(0.05)
    ax.view_init(elev=22, azim=35)


def _draw_axes_lines(ax, axes, lengths, names, colours, linestyle="-", fontsize=10):
    for k in range(3):
        v = axes[:, k] * lengths[k]
        ax.plot([-v[0], v[0]], [-v[1], v[1]], [-v[2], v[2]], linestyle, color=colours[k],
                linewidth=1.6)
        ax.text(*(1.08 * v), names[k], color=colours[k], fontsize=fontsize)


def plot_tensor_surface(site, larmor_Hz=None, transition=None, order="second",
                        relative_to="zero", figure=None, fontsize=12.0, n_theta=73, n_phi=145,
                        show_axes=True, show_j=True):
    """The frequency surface of one transition (see the module
    docstring): radius |frequency| in each direction of B0, coloured by
    the signed frequency (ppm) with a colour bar. Principal axes of the
    CSA (solid) and EFG (dashed) are drawn through the origin.

    Returns ``(figure, notes)`` -- ``notes`` a list of plain-language
    remarks (what is drawn, J, signs)."""
    import matplotlib.pyplot as plt
    from matplotlib import cm, colors

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
        norm, cmap = colors.TwoSlopeNorm(0.0, -lim, lim), plt.get_cmap("RdBu_r")
    else:
        if vmax - vmin <= 1e-6 * max(abs(vmax), 1.0):
            # One value everywhere (isotropic): a readable colour bar
            # around it, not matplotlib's "1e-9 + 1.2e2" offset.
            pad = max(1.0, 0.01 * abs(vmax))
            vmin, vmax = vmin - pad, vmax + pad
        norm = colors.Normalize(vmin, vmax)
        cmap = plt.get_cmap("viridis")
    figure, ax = _new_3d_axes(figure)
    ax.plot_surface(x, y, z, facecolors=cmap(norm(value)), rstride=1, cstride=1, linewidth=0,
                    antialiased=True, shade=False)
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
                ax.plot_wireframe(rr * np.sin(theta) * np.cos(phi), rr * np.sin(theta) * np.sin(phi),
                                  rr * np.cos(theta), rstride=6, cstride=6, color="0.3",
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
            _draw_axes_lines(ax, comps["efg"]["axes"], [extent / 1.15] * 3,
                             ["x$_{EFG}$", "y$_{EFG}$", "z$_{EFG}$"], ["0.35"] * 3, "--",
                             fontsize * 0.75)
        if comps["csa"] is not None:
            _draw_axes_lines(ax, comps["csa"]["axes"], [extent / 1.1] * 3,
                             ["δ$_{xx}$", "δ$_{yy}$", "δ$_{zz}$"], ["black"] * 3, "-",
                             fontsize * 0.75)

    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(value)
    bar = figure.colorbar(mappable, ax=ax, shrink=0.65, pad=0.08)
    bar.set_label("δ − δ$_{iso}$ (ppm)" if relative_to == "iso" else "δ (ppm)", fontsize=fontsize * 0.85)
    bar.ax.tick_params(labelsize=fontsize * 0.7)
    bar.formatter.set_useOffset(False)
    bar.update_ticks()
    _finish_3d(ax, extent, fontsize, "ppm")
    what = [f"{site.label}, {surf['transition']}"]
    if site.get("quadrupolar"):
        what.append(f"{larmor / 1e6:.4g} MHz, {order} order")
    figure.suptitle(" · ".join(what), fontsize=fontsize * 0.9)

    notes.insert(0, "Radius = |frequency| for B0 along that direction (site frame); colour = the "
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


def plot_tensor_ellipsoids(site, figure=None, fontsize=12.0):
    """The tensors as glyphs in one frame (see the module docstring): the
    CSA ellipsoid (semi-axes |delta_ii|), the EFG's principal axes (length
    proportional to |V_ii|, red positive, blue negative), and the sphere of
    radius |delta_iso| for a site with neither. Returns ``(figure, notes)``."""
    from matplotlib.lines import Line2D

    site = ch.as_sites(site)[0]
    comps = tensor_components(site)
    figure, ax = _new_3d_axes(figure)
    notes, handles = [], []
    extent = 1.0
    if comps["csa"] is not None:
        principal = np.array(comps["csa"]["principal"])
        axes = comps["csa"]["axes"]
        semi = np.maximum(np.abs(principal), 1e-9)
        _ellipsoid(ax, np.zeros(3), semi, axes, "#8c8c8c", 0.35)
        names = ["δ$_{xx}$", "δ$_{yy}$", "δ$_{zz}$"]
        for k in range(3):
            colour = _POSITIVE if principal[k] >= 0 else _NEGATIVE
            v = axes[:, k] * semi[k]
            ax.plot([-v[0], v[0]], [-v[1], v[1]], [-v[2], v[2]], color=colour, linewidth=2.0)
            ax.text(*(1.1 * v), f"{names[k]} = {principal[k]:.4g}", color=colour,
                    fontsize=fontsize * 0.75)
        extent = 1.25 * float(semi.max())
        handles.append(Line2D([], [], color="#8c8c8c", linewidth=6, alpha=0.5,
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
        _ellipsoid(ax, np.zeros(3), np.array([r, r, r]), np.eye(3), "#8c8c8c", 0.35)
        extent = 1.25 * r
        notes.append(f"Isotropic shift only: a sphere of radius |δiso| = {abs(site.delta_iso):g} ppm.")
        handles.append(Line2D([], [], color="#8c8c8c", linewidth=6, alpha=0.5,
                              label="isotropic shift (ppm)"))
    if comps["efg"] is not None:
        principal = np.array(comps["efg"]["principal"])
        scale = (0.9 * extent / 1.25) / (np.max(np.abs(principal)) or 1.0)
        if comps["csa"] is None:
            scale, extent = 1.0 / (np.max(np.abs(principal)) or 1.0), 1.3
        names = ["V$_{xx}$", "V$_{yy}$", "V$_{zz}$"]
        for k in range(3):
            colour = _POSITIVE if principal[k] >= 0 else _NEGATIVE
            v = comps["efg"]["axes"][:, k] * abs(principal[k]) * scale
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
            notes.append("The CSA ellipsoid is shown in the EFG frame, rotated by the Euler angles "
                         "(α, β, γ) = ({:g}, {:g}, {:g})°.".format(*e))
    if comps["j"]:
        text = ", ".join(f"{g['n']}×{g['nuclide']} {g['J_Hz']:g} Hz" for g in comps["j"])
        notes.append(f"J ({text}) is isotropic in this simulation: it splits lines but has no "
                     "orientation, so it has no glyph.")
    if handles:
        ax.legend(handles=handles, loc="upper left", fontsize=fontsize * 0.7, frameon=False)
    _finish_3d(ax, extent, fontsize, "ppm" if comps["csa"] is not None or comps["efg"] is None
               else "rel.")
    figure.suptitle(f"{site.label}: tensors in the "
                    f"{'EFG' if comps['efg'] is not None else 'CSA'} frame", fontsize=fontsize * 0.9)
    return figure, notes


def plot_tensors(site, larmor_Hz=None, style="surface", **kwargs):
    """``style`` 'surface' (:func:`plot_tensor_surface`) or 'ellipsoid'
    (:func:`plot_tensor_ellipsoids`). Returns ``(figure, notes)``."""
    if style == "surface":
        return plot_tensor_surface(site, larmor_Hz, **kwargs)
    if style == "ellipsoid":
        kwargs = {k: v for k, v in kwargs.items() if k in ("figure", "fontsize")}
        return plot_tensor_ellipsoids(site, **kwargs)
    raise ValueError("style must be 'surface' or 'ellipsoid'.")
