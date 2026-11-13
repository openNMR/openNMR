"""Vector figures (PDF, SVG, EPS/PS) that open whole in Illustrator.

Adobe Illustrator drops any single path with more than about 32,000
anchor points when it opens a PDF or SVG -- silently. A fit's shaded
components are ``fill_between`` polygons with two points per spectrum
point (top and bottom), so on a 16k-point spectrum each one is ~32k
points and vanishes, while the data and fit lines (half as many points,
and thinned further by matplotlib's line simplification, which fills
never get) survive. Viewers, browsers and PNG are unaffected. User
report, 2026-10-17: "only the data and overall fit appear (but each site
remains in the legend)".

:func:`vector_safe` thins every over-long line and filled path for the
duration of a save, keeping each run's lowest and highest point (so no
peak or edge is lost: at any print resolution the result is the same
picture), then puts everything back. It also embeds fonts as TrueType
(``pdf.fonttype``/``ps.fonttype`` 42) rather than matplotlib's default
Type 3, which Illustrator cannot edit as text.

It also cuts every line and filled area at the edges of the box it is
clipped to (:func:`crop_to_axes`), so releasing the clipping masks in
Illustrator shows what the figure shows, not the whole spectrum running
past the axes (user report, 2026-11-13, stacked spectra).

Used by the GUI's Save Figure (``MplCanvas``) and by
:func:`opennmr.plotting.style.save_figure`; in a notebook::

    from opennmr.plotting.vector_export import save_vector_safe
    save_vector_safe(fig, "fit.pdf")
"""
from __future__ import annotations

import contextlib
import os

import numpy as np

#: The most points any one path keeps. Illustrator's limit is 32,767;
#: this leaves room, and is still ~4x the dots across a 7-inch figure at
#: 600 dpi, so the thinning is invisible.
MAX_PATH_POINTS = 20000

VECTOR_FORMATS = ("pdf", "svg", "svgz", "eps", "ps")


def thin_indices(values: np.ndarray, max_points: int = MAX_PATH_POINTS) -> np.ndarray:
    """Indices of ``values`` (in order) keeping each of ``max_points // 2``
    consecutive runs' smallest and largest value -- the envelope, so a
    narrow peak survives. All indices when already short enough; the
    first and last are always kept."""
    values = np.asarray(values, dtype=float)
    n = values.size
    if n <= max_points:
        return np.arange(n)
    bins = max(1, max_points // 2 - 1)
    edges = np.linspace(0, n, bins + 1).astype(int)
    keep = [0, n - 1]
    for a, b in zip(edges[:-1], edges[1:]):
        if b <= a:
            continue
        chunk = values[a:b]
        finite = np.isfinite(chunk)
        if not finite.any():
            keep.append(a)
            continue
        idx = np.flatnonzero(finite)
        keep.append(a + int(idx[np.argmin(chunk[finite])]))
        keep.append(a + int(idx[np.argmax(chunk[finite])]))
    return np.unique(np.asarray(keep, dtype=int))


def _monotonic(values) -> bool:
    d = np.diff(np.asarray(values, dtype=float)[np.isfinite(values)])
    return bool(d.size) and (np.all(d >= 0) or np.all(d <= 0))


def _thin_line(line, max_points):
    x = np.asarray(line.get_xdata(orig=True), dtype=float)
    y = np.asarray(line.get_ydata(orig=True), dtype=float)
    if y.size <= max_points or x.size != y.size:
        return None
    # The envelope of the values along the line: y for a spectrum
    # against ppm, x for a vertical trace (a projection strip).
    keep = thin_indices(x if _monotonic(y) and not _monotonic(x) else y, max_points)
    original = (line.get_xdata(orig=True), line.get_ydata(orig=True))
    line.set_data(x[keep], y[keep])
    return lambda: line.set_data(*original)


def _thin_path(path, max_points):
    from matplotlib.path import Path
    vertices = np.asarray(path.vertices)
    if len(vertices) <= max_points:
        return path
    codes = path.codes
    if codes is not None and np.any((codes != Path.MOVETO) & (codes != Path.LINETO)
                                     & (codes != Path.CLOSEPOLY)):
        return path            # curves: leave alone
    if codes is not None and np.count_nonzero(codes == Path.MOVETO) > 1:
        return path            # several sub-paths: leave alone
    body = vertices[:-1] if codes is not None and codes[-1] == Path.CLOSEPOLY else vertices
    # Along a filled-between polygon the outline runs along x, out and
    # back: the envelope of y within consecutive runs keeps its shape.
    keep = thin_indices(body[:, 1], max_points)
    return Path(body[keep], closed=True)


def _swap_paths(collection, paths) -> None:
    # PolyCollection.set_paths takes vertex arrays (and rebuilds every
    # path); these are finished Path objects, put back exactly as given.
    collection._paths = list(paths)
    collection.stale = True


# ---- cropping to the axes ------------------------------------------------------
#
# A PDF/SVG keeps every point of a line and lets the axes' clipping path
# hide what lies outside the view. Illustrator users routinely release
# that clipping mask (Object ▸ Clipping Mask ▸ Release, or ungroup) to edit
# the figure, and then the hidden parts reappear past the axes -- most
# visibly in a Compare/pseudo-2D stack, where every row's spectrum runs
# beyond the x-range (user report with a screenshot, 2026-11-13). The
# lines are therefore cut at the axes' edges for the save, at the exact
# crossing point, so the file holds only what is seen; the clipping path
# is still written, and the figure is restored afterwards.

#: The unit square: an axes' own rectangle in axes-fraction coordinates.
UNIT = (0.0, 1.0, 0.0, 1.0)


def _inside(xy, rect):
    x0, x1, y0, y1 = rect
    return (np.all(np.isfinite(xy), axis=1) & (xy[:, 0] >= x0) & (xy[:, 0] <= x1)
            & (xy[:, 1] >= y0) & (xy[:, 1] <= y1))


def _segment_clip(p, q, rect=UNIT):
    """Liang–Barsky for segments p[i] → q[i] (n×2) against the rectangle
    ``rect`` = (x0, x1, y0, y1): ``(t0, t1, visible)``."""
    x0, x1, y0, y1 = rect
    d = q - p
    t0 = np.zeros(len(p))
    t1 = np.ones(len(p))
    visible = np.all(np.isfinite(p), axis=1) & np.all(np.isfinite(q), axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        for pk, qk in ((-d[:, 0], p[:, 0] - x0), (d[:, 0], x1 - p[:, 0]),
                       (-d[:, 1], p[:, 1] - y0), (d[:, 1], y1 - p[:, 1])):
            r = qk / pk
            entering = pk < 0
            leaving = pk > 0
            t0 = np.where(entering, np.maximum(t0, r), t0)
            t1 = np.where(leaving, np.minimum(t1, r), t1)
            visible &= ~((pk == 0) & (qk < 0))
    visible &= t0 <= t1
    return t0, t1, visible


def clip_polyline(xy, rect=UNIT):
    """The parts of the polyline ``xy`` (n×2; NaN rows are gaps) inside
    the rectangle ``rect`` = (x0, x1, y0, y1), cut exactly where it
    crosses an edge, with NaN rows between separate pieces. ``None`` when
    nothing lies outside (no change needed)."""
    xy = np.asarray(xy, dtype=float)
    finite = np.all(np.isfinite(xy), axis=1)
    if len(xy) < 2 or np.array_equal(_inside(xy, rect), finite):
        return None
    p, q = xy[:-1], xy[1:]
    t0, t1, vis = _segment_clip(p, q, rect)
    k = np.flatnonzero(vis)
    if not k.size:
        return np.empty((0, 2))
    d = q[k] - p[k]
    start = p[k] + t0[k, None] * d
    end = p[k] + t1[k, None] * d
    # A visible segment continues the previous one when they meet at a
    # vertex inside the square.
    cont = np.zeros(k.size, dtype=bool)
    cont[1:] = (k[1:] == k[:-1] + 1) & (t1[k[:-1]] >= 1.0) & (t0[k[1:]] <= 0.0)
    slots = np.full((k.size, 3, 2), np.nan)
    slots[:, 1] = start
    slots[:, 2] = end
    keep = np.ones((k.size, 3), dtype=bool)
    keep[:, 0] = ~cont
    keep[0, 0] = False
    keep[:, 1] = ~cont
    return slots[keep]


def clip_polygon(xy, rect=UNIT):
    """Sutherland–Hodgman: the polygon ``xy`` (closed implicitly) cut to
    the rectangle ``rect`` = (x0, x1, y0, y1). ``None`` when it lies
    wholly inside."""
    poly = np.asarray(xy, dtype=float)
    if not len(poly) or not np.all(np.isfinite(poly)):
        return None
    if np.all(_inside(poly, rect)):
        return None
    x0, x1, y0, y1 = rect
    for axis, bound, keep_above in ((0, x0, True), (0, x1, False), (1, y0, True),
                                    (1, y1, False)):
        if not len(poly):
            break
        nxt = np.roll(poly, -1, axis=0)
        a, b = poly[:, axis], nxt[:, axis]
        in_a = a >= bound if keep_above else a <= bound
        in_b = b >= bound if keep_above else b <= bound
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(b != a, (bound - a) / (b - a), 0.0)
        cross = poly + t[:, None] * (nxt - poly)
        cross[:, axis] = bound
        slots = np.stack([cross, nxt], axis=1)          # per edge: crossing, next
        use = np.stack([in_a != in_b, in_b], axis=1)
        poly = slots[use]
    return poly


def _axes_clip(artist, ax):
    """``(transform, rect)``: artist coordinates → axes fraction, and the
    rectangle (x0, x1, y0, y1, in axes fractions) the artist is clipped
    to -- the axes' own, or a box of its own such as a Compare stack row's
    "sideways only" one (its lines may spill up and down into the next
    row). None when the artist is not clipped by a rectangle (clipping
    off, a clip path of another shape, a polar or 3D axes)."""
    if getattr(ax, "name", "") != "rectilinear" or not artist.get_clip_on():
        return None
    clip_path = artist.get_clip_path()
    if clip_path is not None and getattr(clip_path, "_patch", None) is not ax.patch:
        return None
    clip_box = artist.get_clip_box()
    if clip_box is None and clip_path is None:
        return None
    to_axes = ax.transAxes.inverted()
    if clip_box is None:
        rect = UNIT
    else:
        # add_line / add_collection clip to the axes patch, which
        # matplotlib stores as a box; a box of the artist's own is
        # turned into axes fractions the same way.
        (bx0, by0), (bx1, by1) = to_axes.transform(clip_box.get_points())
        rect = (min(bx0, bx1), max(bx0, bx1), min(by0, by1), max(by0, by1))
        if clip_path is not None:
            rect = (max(rect[0], 0.0), min(rect[1], 1.0), max(rect[2], 0.0), min(rect[3], 1.0))
    if not (rect[0] < rect[1] and rect[2] < rect[3]):
        return None
    return artist.get_transform() + to_axes, rect


def _crop_line(line, ax):
    if line.get_drawstyle() not in (None, "default"):
        return None
    marker = line.get_marker()
    has_marker = marker not in (None, "", "None", " ", "none")
    has_line = line.get_linestyle() not in ("None", "", " ", "none")
    clip = _axes_clip(line, ax)
    if clip is None:
        return None
    to_axes, rect = clip
    x = np.asarray(line.get_xdata(orig=True), dtype=float)
    y = np.asarray(line.get_ydata(orig=True), dtype=float)
    if x.size != y.size or x.size < 2:
        return None
    with np.errstate(invalid="ignore"):
        frac = to_axes.transform(np.column_stack([x, y]))
    if has_marker and has_line:
        return None            # new end points would gain markers
    if has_marker:
        inside = _inside(frac, rect)
        if inside.all():
            return None
        cut = np.column_stack([x, y])[inside]
    else:
        clipped = clip_polyline(frac, rect)
        if clipped is None:
            return None
        with np.errstate(invalid="ignore"):
            cut = to_axes.inverted().transform(clipped) if len(clipped) else clipped
    original = (line.get_xdata(orig=True), line.get_ydata(orig=True))
    line.set_data(cut[:, 0], cut[:, 1])
    return lambda: line.set_data(*original)


def _crop_poly_collection(coll, ax):
    from matplotlib.path import Path
    clip = _axes_clip(coll, ax)
    if clip is None or len(coll.get_offsets()) > 1:
        return None
    to_axes, rect = clip
    paths = coll.get_paths()
    new = []
    changed = False
    for path in paths:
        codes = path.codes
        if codes is not None and (np.any((codes != Path.MOVETO) & (codes != Path.LINETO)
                                          & (codes != Path.CLOSEPOLY))
                                  or np.count_nonzero(codes == Path.MOVETO) > 1):
            new.append(path)
            continue
        verts = np.asarray(path.vertices, dtype=float)
        body = verts[:-1] if codes is not None and codes[-1] == Path.CLOSEPOLY else verts
        clipped = clip_polygon(to_axes.transform(body), rect)
        if clipped is None:
            new.append(path)
            continue
        changed = True
        if len(clipped) < 3:
            new.append(Path(np.empty((0, 2))))
        else:
            new.append(Path(to_axes.inverted().transform(clipped), closed=True))
    if not changed:
        return None
    original = list(paths)
    _swap_paths(coll, new)
    return lambda: _swap_paths(coll, original)


def _all_axes(figure):
    """Every axes of ``figure``, insets (``ax.inset_axes`` children, which
    ``figure.get_axes()`` leaves out) included."""
    out = []
    todo = list(figure.get_axes())
    while todo:
        ax = todo.pop(0)
        if any(ax is a for a in out):
            continue
        out.append(ax)
        todo.extend(getattr(ax, "child_axes", []) or [])
    return out


def crop_to_axes(figure):
    """Cut every line and filled area of ``figure`` that is clipped to its
    axes at the axes' edges (in place). Returns a function that puts the
    figure back exactly as it was."""
    from matplotlib.collections import PathCollection, PolyCollection
    restore = []
    for ax in _all_axes(figure):
        try:
            ax.get_xlim(), ax.get_ylim()     # settle a pending autoscale first
        except Exception:  # noqa: BLE001 -- an axes without limits (3D, etc.)
            continue
        for line in list(ax.lines):
            undo = _crop_line(line, ax)
            if undo is not None:
                restore.append(undo)
        for coll in list(ax.collections):
            if not isinstance(coll, PolyCollection) or isinstance(coll, PathCollection):
                continue
            undo = _crop_poly_collection(coll, ax)
            if undo is not None:
                restore.append(undo)

    def put_back():
        for undo in reversed(restore):
            try:
                undo()
            except Exception:  # noqa: BLE001 -- best-effort restore
                pass
    return put_back


@contextlib.contextmanager
def vector_safe(figure, max_points: int = MAX_PATH_POINTS, crop: bool = True):
    """Within the block, every line and filled area of ``figure`` is cut
    at its axes' edges (``crop``, so nothing hidden by the clipping path
    reappears when it is released in Illustrator), every one still longer
    than ``max_points`` is thinned (envelope-preserving), and fonts are
    embedded as TrueType; on leaving, the figure is exactly as it was."""
    import matplotlib
    from matplotlib.collections import PathCollection, PolyCollection

    restore = []
    try:
        if crop:
            restore.append(crop_to_axes(figure))
        for ax in _all_axes(figure):
            for line in list(ax.lines):
                undo = _thin_line(line, max_points)
                if undo is not None:
                    restore.append(undo)
            for coll in list(ax.collections):
                if not isinstance(coll, PolyCollection) or isinstance(coll, PathCollection):
                    continue
                paths = coll.get_paths()
                thinned = [_thin_path(p, max_points) for p in paths]
                if any(t is not p for t, p in zip(thinned, paths)):
                    original = list(paths)
                    _swap_paths(coll, thinned)
                    restore.append(lambda c=coll, o=original: _swap_paths(c, o))
        with matplotlib.rc_context({"pdf.fonttype": 42, "ps.fonttype": 42}):
            yield
    finally:
        for undo in reversed(restore):
            try:
                undo()
            except Exception:  # noqa: BLE001 -- best-effort restore
                pass


def is_vector(fname=None, fmt=None) -> bool:
    """Whether a save to ``fname`` (or in ``fmt``) is a vector format."""
    if fmt:
        return str(fmt).lower().lstrip(".") in VECTOR_FORMATS
    if isinstance(fname, (str, os.PathLike)):
        return os.path.splitext(os.fspath(fname))[1].lower().lstrip(".") in VECTOR_FORMATS
    return False


def save_vector_safe(figure, fname, **kwargs):
    """``figure.savefig(fname, **kwargs)``, through :func:`vector_safe`
    when the format is a vector one."""
    if is_vector(fname, kwargs.get("format")):
        with vector_safe(figure):
            return figure.savefig(fname, **kwargs)
    return figure.savefig(fname, **kwargs)
