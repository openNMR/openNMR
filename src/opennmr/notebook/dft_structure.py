"""
DFT tensors drawn on the structure they were calculated for (explicit user
request, 0.8.4): "see both the structure (periodic or molecular) with the
tensors of the atom(s) of interest centred on the corresponding atomic
positions".

    from opennmr.notebook import dft_nmr, dft_structure
    sample = dft_nmr.read_calculation("sample.magres")
    model = dft_nmr.dft_sites(sample, element="Na", delta_ref=7.21)
    fig, notes = dft_structure.plot_dft_structure(sample, model["atoms"],
                                                  tensor="shift_aniso", region="cell")

What is drawn
-------------
* **Atoms** at the calculation's own positions (Å), coloured by element
  (Jmol colours) and sized by covalent radius. A periodic calculation is
  shown either as its unit cell (``region="cell"``: every atom wrapped
  into the cell, with the copies on the faces, edges and corners that
  belong there, and the cell's edges) or as a cluster
  (``region="radius"``: every atom, periodic images included, within
  ``radius`` Å of an atom of interest), or as any block of cells in
  fractional coordinates (``region="range"``, ``frac_range=((-0.2,
  1.6), (0, 1), (0, 2))`` -- a supercell is ``((0, 2), (0, 2), (0,
  1))``; user request, 2026-10-27). A molecule is shown whole (or
  within ``radius`` of the atoms of interest).
* **Bonds** between atoms closer than ``bond_tolerance`` × the sum of
  their covalent radii (Cordero et al., Dalton Trans. 2008, 2832). Two
  metals are never bonded (it would join every cation of an ionic
  solid); turn bonds off, or change the tolerance, when the guess is
  wrong -- they are only a guide to the eye.
* **Tensors** of the atoms of interest, centred on their positions, in
  the same frame as the structure, so their orientation relative to the
  bonds is what is seen:

  - ``"shift"``: the shift tensor δ as an ellipsoid with semi-axes
    |δ11|, |δ22|, |δ33| along its principal axes (a faithful glyph only
    when all three have the same sign -- the notes say when not).
  - ``"shift_aniso"``: its anisotropic part δ − δiso·1 and
  - ``"efg"``: the electric field gradient V (traceless), both drawn as
    the tensor's *ovaloid*: in each direction n the distance
    |nᵀ T n|, red where nᵀ T n > 0 and blue where < 0. A traceless tensor
    always has both signs, so no ellipsoid can represent it.

  Every tensor in one picture shares **one length scale** (the largest
  is ``tensor_size`` Å), so their sizes compare directly; the scale is
  stated in the notes.

mplot3d draws each object (the atoms, each tensor) as a whole, not
pixel by pixel, so where two overlap the one drawn later is in front:
the tensors are drawn last and translucent, so the atom at their centre
shows through.
"""

from __future__ import annotations

import itertools

import numpy as np

from .tensor_views import _finish_3d, _merge_style, _new_3d_axes, fit_3d_ticks

__all__ = ["COVALENT_RADII", "ELEMENT_COLOURS", "structure_atoms", "find_bonds",
           "tensor_glyph", "plot_dft_structure", "frac_range_of", "MAX_ATOMS"]

#: Covalent radii (Å), Cordero et al. 2008 (low-spin values for Mn, Fe, Co;
#: sp³ C). Elements not listed fall back to 1.5 Å.
COVALENT_RADII = {
    "H": 0.31, "He": 0.28, "Li": 1.28, "Be": 0.96, "B": 0.84, "C": 0.76, "N": 0.71, "O": 0.66,
    "F": 0.57, "Ne": 0.58, "Na": 1.66, "Mg": 1.41, "Al": 1.21, "Si": 1.11, "P": 1.07, "S": 1.05,
    "Cl": 1.02, "Ar": 1.06, "K": 2.03, "Ca": 1.76, "Sc": 1.70, "Ti": 1.60, "V": 1.53,
    "Cr": 1.39, "Mn": 1.39, "Fe": 1.32, "Co": 1.26, "Ni": 1.24, "Cu": 1.32, "Zn": 1.22,
    "Ga": 1.22, "Ge": 1.20, "As": 1.19, "Se": 1.20, "Br": 1.20, "Kr": 1.16, "Rb": 2.20,
    "Sr": 1.95, "Y": 1.90, "Zr": 1.75, "Nb": 1.64, "Mo": 1.54, "Tc": 1.47, "Ru": 1.46,
    "Rh": 1.42, "Pd": 1.39, "Ag": 1.45, "Cd": 1.44, "In": 1.42, "Sn": 1.39, "Sb": 1.39,
    "Te": 1.38, "I": 1.39, "Xe": 1.40, "Cs": 2.44, "Ba": 2.15, "La": 2.07, "Ce": 2.04,
    "Pr": 2.03, "Nd": 2.01, "Pm": 1.99, "Sm": 1.98, "Eu": 1.98, "Gd": 1.96, "Tb": 1.94,
    "Dy": 1.92, "Ho": 1.92, "Er": 1.89, "Tm": 1.90, "Yb": 1.87, "Lu": 1.87, "Hf": 1.75,
    "Ta": 1.70, "W": 1.62, "Re": 1.51, "Os": 1.44, "Ir": 1.41, "Pt": 1.36, "Au": 1.36,
    "Hg": 1.32, "Tl": 1.45, "Pb": 1.46, "Bi": 1.48, "Po": 1.40, "At": 1.50, "Rn": 1.50,
    "Fr": 2.60, "Ra": 2.21, "Ac": 2.15, "Th": 2.06, "Pa": 2.00, "U": 1.96, "Np": 1.90,
    "Pu": 1.87,
}

#: Jmol element colours. Elements not listed are grey.
ELEMENT_COLOURS = {
    "H": "#FFFFFF", "He": "#D9FFFF", "Li": "#CC80FF", "Be": "#C2FF00", "B": "#FFB5B5",
    "C": "#909090", "N": "#3050F8", "O": "#FF0D0D", "F": "#90E050", "Ne": "#B3E3F5",
    "Na": "#AB5CF2", "Mg": "#8AFF00", "Al": "#BFA6A6", "Si": "#F0C8A0", "P": "#FF8000",
    "S": "#FFFF30", "Cl": "#1FF01F", "Ar": "#80D1E3", "K": "#8F40D4", "Ca": "#3DFF00",
    "Sc": "#E6E6E6", "Ti": "#BFC2C7", "V": "#A6A6AB", "Cr": "#8A99C7", "Mn": "#9C7AC7",
    "Fe": "#E06633", "Co": "#F090A0", "Ni": "#50D050", "Cu": "#C88033", "Zn": "#7D80B0",
    "Ga": "#C28F8F", "Ge": "#668F8F", "As": "#BD80E3", "Se": "#FFA100", "Br": "#A62929",
    "Kr": "#5CB8D1", "Rb": "#702EB0", "Sr": "#00FF00", "Y": "#94FFFF", "Zr": "#94E0E0",
    "Nb": "#73C2C9", "Mo": "#54B5B5", "Tc": "#3B9E9E", "Ru": "#248F8F", "Rh": "#0A7D8C",
    "Pd": "#006985", "Ag": "#C0C0C0", "Cd": "#FFD98F", "In": "#A67573", "Sn": "#668080",
    "Sb": "#9E63B5", "Te": "#D47A00", "I": "#940094", "Xe": "#429EB0", "Cs": "#57178F",
    "Ba": "#00C900", "La": "#70D4FF", "Ce": "#FFFFC7", "Pr": "#D9FFC7", "Nd": "#C7FFC7",
    "Pm": "#A3FFC7", "Sm": "#8FFFC7", "Eu": "#61FFC7", "Gd": "#45FFC7", "Tb": "#30FFC7",
    "Dy": "#1FFFC7", "Ho": "#00FF9C", "Er": "#00E675", "Tm": "#00D452", "Yb": "#00BF38",
    "Lu": "#00AB24", "Hf": "#4DC2FF", "Ta": "#4DA6FF", "W": "#2194D6", "Re": "#267DAB",
    "Os": "#266696", "Ir": "#175487", "Pt": "#D0D0E0", "Au": "#FFD123", "Hg": "#B8B8D0",
    "Tl": "#A6544D", "Pb": "#575961", "Bi": "#9E4FB5", "Po": "#AB5C00", "At": "#754F45",
    "Rn": "#428296", "Fr": "#420066", "Ra": "#007D00", "Ac": "#70ABFA", "Th": "#00BAFF",
    "Pa": "#00A1FF", "U": "#008FFF", "Np": "#0080FF", "Pu": "#006BFF",
}

#: Elements never bonded to one another (see the module docstring).
_NONMETALS = {"H", "He", "B", "C", "N", "O", "F", "Ne", "Si", "P", "S", "Cl", "Ar", "Ge", "As",
              "Se", "Br", "Kr", "Sb", "Te", "I", "Xe", "At", "Rn"}

TENSOR_KINDS = {"shift": "shift tensor δ (ellipsoid, semi-axes |δii|)",
                "shift_aniso": "anisotropic shift δ − δiso (ovaloid)",
                "efg": "EFG tensor V (ovaloid)"}

_TOL = 1e-4          # fractional-coordinate tolerance for "on a cell face"

#: The most atoms one picture draws: beyond this, finding the bonds and
#: drawing them takes long enough to look like a hang, and the picture is
#: unreadable anyway. Narrow the range (or the radius) instead.
MAX_ATOMS = 3000

#: Cell edges a picture can draw (``cell_edges``): every whole cell inside
#: the region, the cell at the origin only, the outline of the fractional
#: range, or none.
CELL_EDGES = ("cells", "origin", "range", "none")


def radius_of(species) -> float:
    return COVALENT_RADII.get(species, 1.5)


def colour_of(species, overrides=None) -> str:
    """The colour ``species`` is drawn in: ``overrides[species]`` when
    given (Tensor Plot Style's element colours), its Jmol colour
    otherwise, grey for an element without one."""
    if overrides and overrides.get(species):
        return overrides[species]
    return ELEMENT_COLOURS.get(species, "#A0A0A0")


def frac_range_of(frac_range):
    """``frac_range`` as a (3, 2) array of (from, to) fractional
    coordinates along a, b, c, checked: each from < to, and no more
    than 12 cells along any axis."""
    try:
        r = np.asarray(frac_range, dtype=float).reshape(3, 2)
    except (TypeError, ValueError):
        raise ValueError("frac_range must be three (from, to) pairs: ((a0, a1), (b0, b1), "
                         "(c0, c1)).") from None
    if not np.all(np.isfinite(r)):
        raise ValueError("frac_range must be finite numbers.")
    for (lo, hi), name in zip(r, "abc"):
        if not hi > lo:
            raise ValueError(f"The {name} range must run from a smaller to a larger value "
                             f"(got {lo:g} to {hi:g}).")
        if hi - lo > 12:
            raise ValueError(f"The {name} range spans {hi - lo:g} cells; at most 12.")
    return r


def _lattice(calc):
    lattice = calc.get("lattice")
    return None if lattice is None else np.asarray(lattice, dtype=float)


def _wrap(frac):
    """Fractional coordinates into [0, 1), with -0.0 and 1 - tiny to 0."""
    frac = np.mod(np.asarray(frac, dtype=float), 1.0)
    frac[np.abs(frac - 1.0) < _TOL] = 0.0
    return frac


def structure_atoms(calc, centres=None, region="cell", radius=6.0,
                    frac_range=None) -> list[dict]:
    """The atoms to draw: ``[{"species", "label", "position" (Cartesian
    Å), "image" (the lattice translation, integers)}, ...]``.

    ``region="cell"``: a periodic calculation's unit cell -- each atom
    wrapped into it, plus its copies on the faces/edges/corners at
    fractional coordinate 1 (a molecule: every atom). ``"radius"``: every
    atom (periodic images included) within ``radius`` Å of one of the
    Cartesian points ``centres``. ``"range"``: every atom (periodic
    images included) whose fractional coordinates lie within
    ``frac_range`` = ((a0, a1), (b0, b1), (c0, c1)), the ends included
    (so ``((0, 2), (0, 2), (0, 1))`` is a 2 × 2 × 1 supercell with the
    copies on its faces, as the unit cell has)."""
    atoms = calc.get("atoms") or []
    if not atoms:
        raise ValueError("This calculation has no atomic positions.")
    lattice = _lattice(calc)
    out = []
    if region == "cell":
        if lattice is None:
            return [{"species": a["species"], "label": a["label"],
                     "position": np.asarray(a["position"], dtype=float), "image": (0, 0, 0)}
                    for a in atoms]
        inv = np.linalg.inv(lattice)
        for a in atoms:
            frac0 = np.asarray(a["position"], dtype=float) @ inv
            frac = _wrap(frac0)
            shift0 = np.round(frac - frac0).astype(int)
            # Copies on the far faces: each coordinate at 0 is also at 1.
            options = [(0, 1) if f < _TOL else (0,) for f in frac]
            for extra in itertools.product(*options):
                f = frac + np.array(extra)
                out.append({"species": a["species"], "label": a["label"],
                            "position": f @ lattice,
                            "image": tuple(int(v) for v in shift0 + np.array(extra))})
        return out
    if region == "range":
        if lattice is None:
            raise ValueError("A fractional range needs a cell: this calculation is not "
                             "periodic.")
        r = frac_range_of(frac_range if frac_range is not None else ((0, 1),) * 3)
        inv = np.linalg.inv(lattice)
        lo, hi = r[:, 0] - _TOL, r[:, 1] + _TOL
        steps = [range(int(np.floor(r[k, 0])) - 1, int(np.ceil(r[k, 1])) + 1) for k in range(3)]
        shifts = np.array(list(itertools.product(*steps)), dtype=float)
        for a in atoms:
            frac0 = np.asarray(a["position"], dtype=float) @ inv
            frac = _wrap(frac0)
            shift0 = np.round(frac - frac0).astype(int)
            cand = frac[None, :] + shifts
            keep = np.all((cand >= lo) & (cand <= hi), axis=1)
            for f, n in zip(cand[keep], shifts[keep]):
                out.append({"species": a["species"], "label": a["label"],
                            "position": f @ lattice,
                            "image": tuple(int(v) for v in shift0 + n.astype(int))})
                if len(out) > MAX_ATOMS:
                    raise ValueError(f"More than {MAX_ATOMS} atoms in this range: narrow it.")
        return out
    if region != "radius":
        raise ValueError("region must be 'cell', 'range' or 'radius'.")
    centres = np.atleast_2d(np.asarray(centres if centres is not None else [], dtype=float))
    if centres.size == 0:
        raise ValueError("region='radius' needs the centres (the atoms of interest).")
    radius = float(radius)
    if lattice is None:
        images = [np.zeros(3, dtype=int)]
    else:
        # Enough images in each direction to reach ``radius``: the cell's
        # height along each reciprocal direction.
        volume = abs(np.linalg.det(lattice))
        heights = [volume / np.linalg.norm(np.cross(lattice[(i + 1) % 3], lattice[(i + 2) % 3]))
                   for i in range(3)]
        reach = [int(np.ceil(radius / h)) + 1 for h in heights]
        images = [np.array(n) for n in itertools.product(*(range(-r, r + 1) for r in reach))]
    seen = set()
    for idx, a in enumerate(atoms):
        p0 = np.asarray(a["position"], dtype=float)
        for n in images:
            p = p0 + (n @ lattice if lattice is not None else 0.0)
            if np.min(np.linalg.norm(centres - p, axis=1)) <= radius + 1e-9:
                key = (idx, tuple(int(v) for v in n))
                if key not in seen:
                    seen.add(key)
                    out.append({"species": a["species"], "label": a["label"], "position": p,
                                "image": key[1]})
                    if len(out) > MAX_ATOMS:
                        raise ValueError(f"More than {MAX_ATOMS} atoms within this radius: "
                                         "make it smaller.")
    return out


def find_bonds(atoms, tolerance=1.15, metal_metal=False) -> list[tuple[int, int]]:
    """Pairs ``(i, j)`` of ``atoms`` closer than ``tolerance`` × the sum of
    their covalent radii (and further apart than 0.1 Å). Two metals are
    left unbonded unless ``metal_metal``."""
    if len(atoms) < 2:
        return []
    pos = np.array([a["position"] for a in atoms], dtype=float)
    rad = np.array([radius_of(a["species"]) for a in atoms])
    metal = np.array([a["species"] not in _NONMETALS for a in atoms])
    diff = pos[:, None, :] - pos[None, :, :]
    dist = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))
    cutoff = float(tolerance) * (rad[:, None] + rad[None, :])
    ok = (dist < cutoff) & (dist > 0.1)
    if not metal_metal:
        ok &= ~(metal[:, None] & metal[None, :])
    i, j = np.nonzero(np.triu(ok, 1))
    return list(zip(i.tolist(), j.tolist()))


def tensor_of(record, kind):
    """The Cartesian tensor of ``record`` (a :func:`dft_nmr.dft_sites`
    atom) that ``kind`` draws, or ``None`` if it has none."""
    if kind == "shift":
        t = record.get("shift_tensor")
    elif kind == "shift_aniso":
        t = record.get("shift_tensor")
        if t is not None:
            t = np.asarray(t, dtype=float)
            t = t - np.trace(t) / 3.0 * np.eye(3)
    elif kind == "efg":
        t = record.get("efg_tensor")
        if t is not None:
            t = np.asarray(t, dtype=float)
            t = t - np.trace(t) / 3.0 * np.eye(3)
    else:
        raise ValueError(f"tensor must be one of {tuple(TENSOR_KINDS)}.")
    if t is None:
        return None
    t = np.asarray(t, dtype=float)
    return 0.5 * (t + t.T)


def tensor_glyph(tensor, kind, scale, centre, n_theta=31, n_phi=61):
    """Surface points of one glyph: ``(x, y, z, sign)``, each (n_theta,
    n_phi). An ellipsoid (``kind="shift"``, semi-axes ``scale`` × |λi|)
    or the ovaloid r(n) = ``scale`` × |nᵀ T n| (other kinds); ``sign`` is
    the sign of nᵀ T n in that direction (+1 everywhere for the
    ellipsoid)."""
    theta, phi = np.meshgrid(np.linspace(0.0, np.pi, n_theta), np.linspace(0, 2 * np.pi, n_phi),
                             indexing="ij")
    n = np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
    if kind == "shift":
        w, U = np.linalg.eigh(tensor)
        local = np.einsum("ij,j...->i...", U, n * np.abs(w)[:, None, None])
        sign = np.ones(theta.shape)
        pts = local * scale
    else:
        q = np.einsum("i...,ij,j...->...", n, tensor, n)
        sign = np.sign(q)
        pts = n * np.abs(q) * scale
    c = np.asarray(centre, dtype=float)
    return pts[0] + c[0], pts[1] + c[1], pts[2] + c[2], sign


def _cell_edges(lattice, origin=(0, 0, 0), size=(1, 1, 1)):
    """The 12 edges of the parallelepiped from fractional ``origin`` with
    sides ``size`` (in cells): one cell by default."""
    origin, size = np.asarray(origin, dtype=float), np.asarray(size, dtype=float)
    corners = {c: (origin + np.array(c, dtype=float) * size) @ lattice
               for c in itertools.product((0, 1), repeat=3)}
    edges = []
    for c in corners:
        for k in range(3):
            if c[k] == 0:
                d = list(c)
                d[k] = 1
                edges.append((corners[c], corners[tuple(d)]))
    return edges


def plot_dft_structure(calc, records, frame="cell", tensor="shift_aniso", region="cell",
                       radius=6.0, tensor_size=1.0, bonds=True, bond_tolerance=1.15,
                       atom_scale=None, labels=True, figure=None, fontsize=11.0,
                       tensor_style=None, frac_range=None, tensor_copies="one",
                       cell_edges="cells", legend_loc="outside right", legend_columns=0,
                       legend_xy=(1.02, 1.0)):
    """The structure of ``calc`` with the tensors of ``records`` (one
    :func:`opennmr.notebook.dft_nmr.dft_sites` atom record, or a list of
    them) centred on their atoms -- see the module docstring.

    ``frame`` : ``"cell"`` (periodic only; the standard setting, a along
    x, b in the xy plane, as :func:`dft_nmr.display_frame`) or
    ``"cartesian"`` (the file's own axes). ``tensor`` : ``"shift"``,
    ``"shift_aniso"`` or ``"efg"``. ``region`` / ``radius`` /
    ``frac_range`` : see :func:`structure_atoms` (``"cell"``,
    ``"radius"`` or ``"range"``). ``tensor_size`` : the largest tensor's
    largest extent, Å.

    ``tensor_copies`` (``region="range"``) : ``"one"`` -- each tensor on
    the copy of its atom nearest the middle of the range -- or ``"all"``,
    on every periodic copy drawn (the same tensor: a lattice translation
    does not turn it). ``cell_edges`` : ``"cells"`` (every whole cell in
    the region), ``"origin"`` (the cell at the origin), ``"range"`` (the
    outline of the fractional range) or ``"none"``.

    ``tensor_style`` : the keys of
    :data:`opennmr.notebook.tensor_views.TENSOR_STYLE_DEFAULTS` -- the
    tensor colours, opacity, grid and view; the 3D box's text (tick and
    axis labels, padding); and the atoms and bonds (size, sizing,
    opacity, outline, depth cueing, element colours, bond width and
    colour). ``atom_scale`` (if given) overrides the style's.

    ``legend_loc`` / ``legend_columns`` / ``legend_xy`` : see
    :func:`opennmr.plotting.legend_placement` (``"outside right"`` by
    default: beside the picture, which is narrowed to make room).

    Returns ``(figure, notes)``."""
    from matplotlib.lines import Line2D
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    from ..plotting.legend_place import legend_placement, make_room_for_legend

    if isinstance(records, dict):
        records = [records]
    records = list(records or [])
    if not records:
        raise ValueError("Choose at least one site to draw the tensors of.")
    if tensor_copies not in ("one", "all"):
        raise ValueError("tensor_copies must be 'one' or 'all'.")
    if cell_edges not in CELL_EDGES:
        raise ValueError(f"cell_edges must be one of {CELL_EDGES}.")
    look = _merge_style(tensor_style)
    if atom_scale is not None:
        look["atom_scale"] = float(atom_scale)
    lattice = _lattice(calc)
    if frame == "cell":
        if lattice is None:
            raise ValueError("This calculation has no cell (not periodic): use the Cartesian "
                             "frame.")
        from .dft_nmr import standard_cell_rotation
        Q = standard_cell_rotation(lattice)
        frame_title = "cell frame (a ∥ x, b in xy)"
    elif frame == "cartesian":
        Q = np.eye(3)
        frame_title = f"{calc.get('code', 'file')} Cartesian frame"
    else:
        raise ValueError("frame must be 'cell' or 'cartesian'.")
    if region not in ("cell", "radius", "range"):
        raise ValueError("region must be 'cell', 'radius' or 'range'.")
    if region == "cell" and lattice is None:
        region_used = "molecule"
    elif region == "range" and lattice is None:
        raise ValueError("A fractional range needs a cell: this calculation is not periodic.")
    else:
        region_used = region
    span = frac_range_of(frac_range if frac_range is not None else ((0, 1),) * 3) \
        if region_used == "range" else None
    notes = []

    # Where each tensor goes: its own atom (wrapped into the cell when the
    # cell is drawn), or the mean position of an averaged group; in a
    # fractional range, the copy (or copies) of it inside the range.
    inv = np.linalg.inv(lattice) if lattice is not None else None
    placed = []           # (record, [centres, Å]) -- the first one is labelled
    for rec in records:
        p = np.asarray(rec["position"], dtype=float)
        if region_used in ("cell", "range"):
            p = _wrap(p @ inv) @ lattice
        if region_used == "range":
            f = p @ inv
            steps = [range(int(np.floor(span[k, 0])) - 1, int(np.ceil(span[k, 1])) + 1)
                     for k in range(3)]
            cand = np.array([f + np.array(n) for n in itertools.product(*steps)])
            inside = cand[np.all((cand >= span[:, 0] - _TOL) & (cand <= span[:, 1] + _TOL),
                                 axis=1)]
            middle = span.mean(axis=1)
            if len(inside) == 0:
                inside = cand[[int(np.argmin(np.linalg.norm((cand - middle) @ lattice,
                                                            axis=1)))]]
                notes.append(f"{rec['label']}: no copy inside the range; its tensor is drawn "
                             "on the nearest one, outside it.")
            order = np.argsort(np.linalg.norm((inside - middle) @ lattice, axis=1),
                               kind="stable")
            inside = inside[order]
            chosen = inside if tensor_copies == "all" else inside[:1]
            placed.append((rec, [c @ lattice for c in chosen]))
        else:
            placed.append((rec, [p]))
        if int(rec.get("n_atoms", 1)) > 1:
            notes.append(f"{rec['label']}: an average over {rec['n_atoms']} sites, drawn at "
                         "their mean position.")
    first_centres = [c[0] for _r, c in placed]
    atoms = structure_atoms(calc, centres=first_centres,
                            region=region if region_used != "molecule" else "cell",
                            radius=radius, frac_range=span)
    pos = np.array([a["position"] for a in atoms]).reshape(-1, 3) @ Q.T
    if not atoms:
        notes.append("No atom lies inside this range.")

    figure, ax = _new_3d_axes(figure)
    # Drawn in the order given (bonds, atoms, tensors), not depth-sorted
    # per object: the translucent tensors are always on top, so the atom
    # at their centre shows through (see the module docstring).
    ax.computed_zorder = False

    # Cell edges.
    boxes = []                       # (fractional origin, size) of each parallelepiped
    if lattice is not None and region_used in ("cell", "range") and cell_edges != "none":
        if cell_edges == "origin" or region_used == "cell":
            boxes = [((0, 0, 0), (1, 1, 1))]
        elif cell_edges == "range":
            boxes = [(span[:, 0], span[:, 1] - span[:, 0])]
        else:
            whole = [range(int(np.ceil(span[k, 0] - _TOL)), int(np.floor(span[k, 1] + _TOL)))
                     for k in range(3)]
            boxes = [(n, (1, 1, 1)) for n in itertools.product(*whole)]
            if not boxes:
                notes.append("No whole cell inside the range: no cell edges drawn (choose "
                             "\"outline of the range\" to see its extent).")
    edge_points = []
    if boxes:
        segs = []
        for origin, size in boxes:
            for a, b in _cell_edges(lattice, origin, size):
                segs.append((a @ Q.T, b @ Q.T))
                edge_points += [a @ Q.T, b @ Q.T]
        ax.add_collection3d(Line3DCollection(segs, colors=look["cell_colour"],
                                             linewidths=float(look["cell_width"]), zorder=1))
        corner = np.zeros(3)
        for v, name in zip(lattice, "abc"):
            tip = (Q @ v)
            ax.text(*(corner + 0.55 * tip), name, fontsize=fontsize * 0.9,
                    color=look["cell_colour"], fontstyle="italic", fontweight="bold", zorder=6)

    element_colours = dict(look.get("atom_colours") or {})
    if bonds:
        pairs = find_bonds(atoms, tolerance=bond_tolerance)
        segs, colours = [], []
        for i, j in pairs:
            mid = 0.5 * (pos[i] + pos[j])
            segs += [(pos[i], mid), (mid, pos[j])]
            if look["bond_colour"]:
                colours += [look["bond_colour"]] * 2
            else:
                colours += [colour_of(atoms[i]["species"], element_colours),
                            colour_of(atoms[j]["species"], element_colours)]
        width = float(look["bond_width"])
        if segs and width > 0:
            # The bonds' own outline, so a white or pale element still shows.
            ax.add_collection3d(Line3DCollection(segs, colors="0.35", linewidths=width + 0.8,
                                                 zorder=1.5, alpha=0.5))
            ax.add_collection3d(Line3DCollection(segs, colors=colours, linewidths=width,
                                                 zorder=2))

    species = [a["species"] for a in atoms]
    scale_atoms = float(look["atom_scale"])
    if look["atom_size_by"] == "uniform":
        sizes = np.full(len(species), (1.2 * 9.0 * scale_atoms) ** 2)
    elif look["atom_size_by"] == "radius":
        sizes = np.array([(radius_of(s) * 9.0 * scale_atoms) ** 2 for s in species])
    else:
        raise ValueError("atom_size_by must be 'radius' or 'uniform'.")
    interest = {rec["label"] for rec in records} | {lab for rec in records
                                                    for lab in rec.get("members", [])}
    # The atoms of interest get a heavier outline.
    edge = float(look["atom_edge_width"])
    widths = [max(1.8, 2.5 * edge) if a["label"] in interest else edge for a in atoms]
    if scale_atoms > 0 and len(pos):
        ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=sizes,
                   c=[colour_of(s, element_colours) for s in species],
                   edgecolors=look["atom_edge_colour"], linewidths=widths,
                   depthshade=bool(look["atom_depthshade"]), alpha=float(look["atom_alpha"]),
                   zorder=3)

    # One length scale for every tensor drawn.
    tensors = [(rec, tensor_of(rec, tensor), centres) for rec, centres in placed]
    missing = [rec["label"] for rec, t, _c in tensors if t is None]
    if missing:
        notes.append(f"No {TENSOR_KINDS[tensor].split(' (')[0]} for " + ", ".join(missing)
                     + " – nothing drawn there.")
    sized = [np.max(np.abs(np.linalg.eigvalsh(t))) for _r, t, _c in tensors if t is not None]
    biggest = max(sized) if sized else 0.0
    if biggest <= 0:
        notes.append("Every tensor drawn is zero: nothing to show at this scale.")
    scale = float(tensor_size) / biggest if biggest > 0 else 0.0
    positive, negative = look["positive_colour"], look["negative_colour"]
    centres_d = []
    n_glyphs = 0
    for rec, t, centres in tensors:
        for k_copy, c0 in enumerate(centres):
            c = np.asarray(c0) @ Q.T
            centres_d.append(c)
            if t is None or scale == 0.0:
                continue
            td = Q @ t @ Q.T
            x, y, z, sign = tensor_glyph(td, tensor, scale, c)
            n_glyphs += 1
            if tensor == "shift":
                ax.plot_surface(x, y, z, color=look["ellipsoid_colour"],
                                alpha=max(float(look["ellipsoid_alpha"]), 0.25), linewidth=0,
                                shade=True, zorder=4)
            else:
                face = np.where(sign[..., None] >= 0, np.array(_rgba(positive)),
                                np.array(_rgba(negative)))
                face[..., 3] = 0.6
                ax.plot_surface(x, y, z, facecolors=face, linewidth=0, shade=False, zorder=4)
            if look["show_axes"]:
                w, U = np.linalg.eigh(td)
                for k in range(3):
                    v = U[:, k] * abs(w[k]) * scale * 1.15
                    ax.plot([c[0] - v[0], c[0] + v[0]], [c[1] - v[1], c[1] + v[1]],
                            [c[2] - v[2], c[2] + v[2]],
                            color=positive if w[k] >= 0 else negative, linewidth=1.2, zorder=5)
            if labels and k_copy == 0:
                ax.text(*(c + np.array([0.0, 0.0, 1.1 * float(tensor_size)])), rec["label"],
                        fontsize=fontsize * 0.75, color="black", zorder=7, ha="center")
        if tensor == "shift" and t is not None:
            w = np.linalg.eigvalsh(t)
            if w.min() < 0 < w.max():
                notes.append(f"{rec['label']}: its principal shifts have mixed signs, so the "
                             "ellipsoid (|δii|) does not show them; the anisotropic view does.")

    # Equal scaling in x, y, z: a sphere must look like a sphere.
    every = np.vstack([p for p in (pos, np.array(centres_d).reshape(-1, 3),
                                   np.array(edge_points).reshape(-1, 3)) if len(p)])
    pad = float(tensor_size) + 0.6
    lo, hi = every.min(axis=0) - pad, every.max(axis=0) + pad
    half = 0.5 * float(np.max(hi - lo))
    mid = 0.5 * (lo + hi)
    _finish_3d(ax, half, fontsize, "Å", look)
    ax.set_xlim(mid[0] - half, mid[0] + half)
    ax.set_ylim(mid[1] - half, mid[1] + half)
    ax.set_zlim(mid[2] - half, mid[2] + half)
    fit_3d_ticks(ax, fontsize * 0.85)

    shown = list(dict.fromkeys(species))
    handles = [Line2D([], [], linestyle="", marker="o", markersize=7,
                      markerfacecolor=colour_of(s, element_colours),
                      markeredgecolor=look["atom_edge_colour"], label=s)
               for s in shown[:12]]
    if tensor == "shift":
        handles.append(Line2D([], [], color=look["ellipsoid_colour"], linewidth=6, alpha=0.6,
                              label="δ ellipsoid"))
    else:
        handles += [Line2D([], [], color=positive, linewidth=6, alpha=0.7, label="> 0"),
                    Line2D([], [], color=negative, linewidth=6, alpha=0.7, label="< 0")]
    placement = legend_placement(legend_loc, legend_columns, *legend_xy)
    placement.setdefault("ncols", 1 if len(handles) < 8 else 2)
    ax.legend(handles=handles, fontsize=fontsize * 0.7, frameon=False, **placement)
    what = {"cell": "unit cell", "radius": f"within {float(radius):g} Å",
            "molecule": "molecule",
            "range": _describe_range(span) if span is not None else ""}[region_used]
    from ..plotting.labels import mathtext_safe
    figure.suptitle(mathtext_safe(
        f"{', '.join(r['label'] for r in records[:4])}"
        f"{' …' if len(records) > 4 else ''}: {TENSOR_KINDS[tensor].split(' (')[0]} "
        f"on the structure ({what}, {frame_title})"), fontsize=fontsize * 0.85)
    try:
        make_room_for_legend(ax)
    except Exception:  # a backend without a renderer: the legend may overhang
        pass

    notes.insert(0, f"{len(atoms)} atoms ({what}); tensors of "
                    f"{sum(t is not None for _r, t, _c in tensors)} site(s)"
                    + (f", {n_glyphs} drawn (every copy in the range)"
                       if tensor_copies == "all" and region_used == "range" else "")
                    + f": {TENSOR_KINDS[tensor]}.")
    if biggest > 0:
        notes.append(f"Scale: {float(tensor_size):g} Å = {_fmt(biggest)} "
                     f"{'V/m²' if tensor == 'efg' else 'ppm'} (the same for every tensor "
                     "drawn).")
    if tensor != "shift":
        notes.append("Ovaloid: in each direction n, the distance is |nᵀTn|, red where positive, "
                     "blue where negative.")
    if bonds:
        notes.append(f"Bonds: closer than {float(bond_tolerance):g} × the sum of covalent radii "
                     "(a guide to the eye; metal–metal pairs are not bonded).")
    notes.append("Turn the view by dragging. Atoms are drawn "
                 + ("to scale with their covalent radii only roughly (marker size)"
                    if look["atom_size_by"] == "radius" else "all the same size")
                 + "; positions are exact.")
    return figure, notes


def _describe_range(span) -> str:
    """"2 × 2 × 1 supercell" for whole cells from the origin, the
    fractional ranges otherwise."""
    span = np.asarray(span, dtype=float)
    if np.allclose(span[:, 0], 0) and np.allclose(span[:, 1], np.round(span[:, 1])):
        return " × ".join(f"{int(round(v))}" for v in span[:, 1]) + " supercell"
    return "fractional range " + " × ".join(f"[{a:g}, {b:g}]" for a, b in span)


def _rgba(colour):
    from matplotlib import colors
    return colors.to_rgba(colour)


def _fmt(v):
    return f"{v:.3g}" if abs(v) < 1e4 else f"{v:.3e}"
