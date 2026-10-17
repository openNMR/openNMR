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


@contextlib.contextmanager
def vector_safe(figure, max_points: int = MAX_PATH_POINTS):
    """Within the block, every line and filled path of ``figure`` longer
    than ``max_points`` is thinned (envelope-preserving) and fonts are
    embedded as TrueType; on leaving, the figure is exactly as it was."""
    import matplotlib
    from matplotlib.collections import PathCollection, PolyCollection

    restore = []
    try:
        for ax in figure.get_axes():
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
