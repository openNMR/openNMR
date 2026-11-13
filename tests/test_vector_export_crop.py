"""Vector exports hold only what the figure shows (user report with a
screenshot, 2026-11-13): "when processing in Illustrator vector based images
in stacking mode, when I ungroup and get rid of the clipping mask the
spectra do exceed a bit the range".

A PDF/SVG keeps every point of a line and lets the clipping path hide the
rest; releasing the mask in Illustrator showed it again. During a vector
save, :func:`opennmr.plotting.vector_export.vector_safe` now cuts lines and
filled areas exactly at the edge of the box they are clipped to (the axes,
or a Compare stack row's sideways-only box), and restores the figure."""

import io
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib.transforms import Bbox, TransformedBbox  # noqa: E402

from opennmr.plotting import vector_export as ve  # noqa: E402
from test_gui import qapp  # noqa: E402, F401

# ---- the geometry ----------------------------------------------------------------------


def test_polyline_is_cut_exactly_at_the_edges():
    xy = np.array([[-1.0, 0.5], [0.5, 0.5], [2.0, 0.5]])
    out = ve.clip_polyline(xy)
    assert np.allclose(out, [[0.0, 0.5], [0.5, 0.5], [1.0, 0.5]])


def test_a_polyline_leaving_and_coming_back_is_split():
    xy = np.array([[0.2, 0.5], [0.5, 2.0], [0.8, 0.5]])
    out = ve.clip_polyline(xy)
    gaps = np.flatnonzero(np.isnan(out[:, 0]))
    assert gaps.size == 1
    first, second = out[:gaps[0]], out[gaps[0] + 1:]
    assert np.allclose(first[-1], [0.3, 1.0]) and np.allclose(second[0], [0.7, 1.0])
    assert np.allclose(first[0], [0.2, 0.5]) and np.allclose(second[-1], [0.8, 0.5])


def test_a_polyline_inside_is_left_alone_and_gaps_are_kept():
    assert ve.clip_polyline(np.array([[0.1, 0.1], [0.5, 0.9]])) is None
    # (a lone point between gaps draws nothing, so a gap needs a segment each side)
    xy = np.array([[0.1, 0.1], [0.2, 0.2], [np.nan, np.nan], [0.5, 0.9], [1.5, 0.9]])
    out = ve.clip_polyline(xy)
    assert np.isnan(out).any() and np.nanmax(out[:, 0]) == pytest.approx(1.0)
    assert np.allclose(out[0], [0.1, 0.1])


def test_a_polyline_wholly_outside_vanishes():
    out = ve.clip_polyline(np.array([[2.0, 0.5], [3.0, 0.5]]))
    assert out.shape == (0, 2)


def test_a_rectangle_of_its_own():
    rect = (0.0, 1.0, -50.0, 50.0)                 # a stack row: sideways only
    xy = np.array([[-1.0, 3.0], [2.0, 3.0]])
    assert np.allclose(ve.clip_polyline(xy, rect), [[0.0, 3.0], [1.0, 3.0]])


def test_polygon_is_cut_to_the_square():
    poly = np.array([[-1.0, -1.0], [2.0, -1.0], [2.0, 0.5], [-1.0, 0.5]])
    out = ve.clip_polygon(poly)
    assert out[:, 0].min() == pytest.approx(0.0) and out[:, 0].max() == pytest.approx(1.0)
    assert out[:, 1].min() == pytest.approx(0.0) and out[:, 1].max() == pytest.approx(0.5)
    assert ve.clip_polygon(np.array([[0.1, 0.1], [0.9, 0.1], [0.5, 0.9]])) is None


# ---- figures -------------------------------------------------------------------------------

def _stack(n=3, clip_sideways=True):
    fig, axes = plt.subplots(n, 1, sharex=True, figsize=(4, 4))
    fig.subplots_adjust(hspace=0)
    x = np.linspace(-50, 50, 5001)
    for i, ax in enumerate(axes):
        y = np.exp(-x ** 2 / 0.5) * (n - i) + 0.05 * np.sin(3 * x)
        line, = ax.plot(x, y)
        if clip_sideways:                       # as MplCanvas._clip_sideways_only
            line.set_clip_box(TransformedBbox(Bbox([[0.0, -50.0], [1.0, 50.0]]),
                                              ax.transAxes))
            line.set_clip_path(None)
        ax.fill_between(x, 0, 0.3 * y, alpha=0.3)
        ax.set_ylim(-0.2, 1.2)
    axes[0].set_xlim(10, -20)
    return fig, axes


def _svg_overshoot(fig):
    """How far (in SVG units) any clipped path's points lie outside its
    clipping rectangle sideways -- what releasing the mask would show."""
    buffer = io.StringIO()
    with ve.vector_safe(fig):
        fig.savefig(buffer, format="svg")
    text = buffer.getvalue()
    rects = {}
    for cid, body in re.findall(r'<clipPath id="([^"]+)">\s*(<rect[^>]*/>)', text):
        get = {k: float(v) for k, v in re.findall(r'(x|width)="([-\d.e]+)"', body)}
        rects[cid] = (get.get("x", 0.0), get.get("x", 0.0) + get["width"])
    worst = 0.0
    for d, cid in re.findall(r'<path d="([^"]+)"[^>]*clip-path="url\(#([^)]+)\)"', text):
        if cid not in rects:
            continue
        xs = np.array([float(a) for a, _b in re.findall(r"[ML] ?([-\d.e]+) ([-\d.e]+)", d)])
        if xs.size:
            lo, hi = rects[cid]
            worst = max(worst, lo - xs.min(), xs.max() - hi)
    return worst


def test_stacked_rows_stay_inside_their_x_range_when_unmasked():
    fig, _axes = _stack()
    plain = io.StringIO()
    fig.savefig(plain, format="svg")
    assert "clip-path" in plain.getvalue()
    assert _svg_overshoot(fig) < 0.01
    plt.close(fig)


def test_stacked_peaks_still_spill_up_and_down():
    fig, axes = _stack()
    with ve.vector_safe(fig):
        y = np.asarray(axes[0].lines[0].get_ydata(), dtype=float)
        x = np.asarray(axes[0].lines[0].get_xdata(), dtype=float)
        assert np.nanmax(y) > 2.5                       # beyond ylim: the row's bleed
        assert np.nanmin(x) == pytest.approx(-20) and np.nanmax(x) == pytest.approx(10)
    plt.close(fig)


def test_the_figure_is_put_back_exactly():
    fig, axes = _stack()
    before = [(np.array(l.get_xdata()), np.array(l.get_ydata())) for a in axes for l in a.lines]
    fills = [a.collections[0].get_paths()[0].vertices.copy() for a in axes]
    with ve.vector_safe(fig):
        fig.savefig(io.BytesIO(), format="pdf")
    after = [(np.array(l.get_xdata()), np.array(l.get_ydata())) for a in axes for l in a.lines]
    assert all(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
               for a, b in zip(before, after))
    assert all(np.array_equal(f, a.collections[0].get_paths()[0].vertices)
               for f, a in zip(fills, axes))
    plt.close(fig)


def test_the_cropped_figure_looks_the_same():
    fig, _axes = _stack()
    plain = io.BytesIO()
    fig.savefig(plain, format="png", dpi=80)
    with ve.vector_safe(fig):
        cropped = io.BytesIO()
        fig.savefig(cropped, format="png", dpi=80)
    a = plt.imread(io.BytesIO(plain.getvalue()))
    b = plt.imread(io.BytesIO(cropped.getvalue()))
    assert a.shape == b.shape and np.mean(np.abs(a - b)) < 2e-3
    plt.close(fig)


def test_fills_are_cut_to_the_axes():
    fig, axes = _stack()
    with ve.vector_safe(fig):
        v = axes[1].collections[0].get_paths()[0].vertices
        assert v[:, 0].min() >= -20 - 1e-9 and v[:, 0].max() <= 10 + 1e-9
    plt.close(fig)


def test_markers_outside_are_dropped_and_none_added():
    fig, ax = plt.subplots()
    ax.plot(np.arange(10.0), np.arange(10.0), "o")
    ax.set_xlim(2.5, 6.5)
    with ve.vector_safe(fig):
        assert list(ax.lines[0].get_xdata()) == [3.0, 4.0, 5.0, 6.0]
    assert len(ax.lines[0].get_xdata()) == 10
    plt.close(fig)


def test_unclipped_lines_are_left_alone():
    fig, ax = plt.subplots()
    line, = ax.plot([-10.0, 10.0], [0.0, 0.0], clip_on=False)
    ax.set_xlim(-1, 1)
    with ve.vector_safe(fig):
        assert list(line.get_xdata()) == [-10.0, 10.0]
    plt.close(fig)


def test_insets_are_cut_too():
    fig, ax = plt.subplots()
    x = np.linspace(0, 10, 1001)
    ax.plot(x, np.sin(x))
    inset = ax.inset_axes([0.6, 0.6, 0.35, 0.35])
    inset.plot(x, np.sin(x))
    inset.set_xlim(2, 3)
    with ve.vector_safe(fig):
        xs = np.asarray(inset.lines[0].get_xdata(), dtype=float)
        assert np.nanmin(xs) == pytest.approx(2) and np.nanmax(xs) == pytest.approx(3)
    plt.close(fig)


def test_crop_can_be_turned_off():
    fig, axes = _stack()
    with ve.vector_safe(fig, crop=False):
        assert np.nanmin(np.asarray(axes[0].lines[0].get_xdata(), dtype=float)) == -50
    plt.close(fig)


# ---- the GUI's Compare stack -----------------------------------------------------------

def test_gui_compare_stack_export_stays_in_range(qapp, tmp_path):
    from opennmr.gui.widgets.canvas import MplCanvas
    from test_view_limits import spec1d
    canvas = MplCanvas()
    canvas.show_compare([spec1d(), spec1d(), spec1d()], mode="stack")
    canvas.set_view_limits(x=(3, 7))
    path = tmp_path / "stack.svg"
    canvas.figure.savefig(str(path))                  # through print_figure, as the dialog
    text = path.read_text()
    rects = {}
    for cid, body in re.findall(r'<clipPath id="([^"]+)">\s*(<rect[^>]*/>)', text):
        get = {k: float(v) for k, v in re.findall(r'(x|width)="([-\d.e]+)"', body)}
        rects[cid] = (get.get("x", 0.0), get.get("x", 0.0) + get["width"])
    checked = 0
    for d, cid in re.findall(r'<path d="([^"]+)"[^>]*clip-path="url\(#([^)]+)\)"', text):
        xs = [float(a) for a, _b in re.findall(r"[ML] ?([-\d.e]+) ([-\d.e]+)", d)]
        if cid in rects and len(xs) > 20:                # the spectra, not ticks
            lo, hi = rects[cid]
            assert min(xs) >= lo - 0.01 and max(xs) <= hi + 0.01
            checked += 1
    assert checked >= 3
    for ax in canvas._compare_axes:                     # the canvas keeps its data
        assert np.nanmin(np.asarray(ax.lines[0].get_xdata(), dtype=float)) < 3
