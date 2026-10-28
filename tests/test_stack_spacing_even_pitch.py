"""Stacked Compare: the spacing control moves every spectrum equally (an even
pitch between baselines) and keeps a zoomed view. Synthetic data, canvas only."""
import numpy as np
import pytest

from opennmr.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
from opennmr.gui.widgets.canvas import MplCanvas


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def _spec(height, centre=5.0):
    ppm = np.linspace(10, 0, 500)
    data = height * np.exp(-((ppm - centre) / 0.2) ** 2) + 0.05 * height
    return NMRSpectrum(data=data, axes=[NMRAxis(kind="frequency", ppm=ppm, hz=ppm * 400)],
                       metadata=NMRMetadata(vendor="bruker", nucleus="1H", sfo1=400.0, ns=1))


def _baselines(canvas):
    """Figure-height fraction of every trace's lowest point, top row first, and
    the height of each trace in the same units."""
    fig = canvas.figure
    fig.canvas.draw()
    out, heights = [], []
    for ax in canvas._compare_axes:
        y = ax.lines[0].get_ydata()
        lo = ax.transData.transform((0, float(np.min(y))))[1]
        hi = ax.transData.transform((0, float(np.max(y))))[1]
        out.append(lo / fig.bbox.height)
        heights.append((hi - lo) / fig.bbox.height)
    return out, heights


def _stack(n=5, heights=None):
    canvas = MplCanvas()
    heights = heights or [1.0] * n
    canvas.show_compare([_spec(h, 3 + i * 0.5) for i, h in enumerate(heights)], mode="stack")
    return canvas


def test_baselines_are_evenly_spaced_at_any_gap(qapp):
    canvas = _stack(5, [1.0, 2.0, 3.0, 1.5, 4.0])
    for gap in (-0.8, -0.3, 0.0, 0.05, 0.5, 1.5):
        canvas.set_stack_gap(gap)
        base, _ = _baselines(canvas)
        steps = np.diff(base)
        assert np.allclose(steps, steps[0], atol=1e-6), (gap, steps)
        assert steps[0] < 0                                  # top row first: baselines fall down the list


def test_dragging_to_closer_moves_every_spectrum_not_just_the_bottom_pair(qapp):
    canvas = _stack(5)
    canvas.set_stack_gap(1.0)
    wide, _ = _baselines(canvas)
    canvas.set_stack_gap(-0.5)
    close, _ = _baselines(canvas)
    pitch_wide, pitch_close = -np.diff(wide), -np.diff(close)
    assert np.all(pitch_close < pitch_wide - 1e-3)           # every gap shrank


def test_pitch_is_in_spectrum_heights_and_overlap_is_possible(qapp):
    canvas = _stack(3)
    canvas.set_stack_gap(-0.5)
    base, heights = _baselines(canvas)
    pitch = base[0] - base[1]
    assert pitch / heights[0] == pytest.approx(1.0 - 0.5 + canvas._TOP_GAP, rel=1e-3)
    assert pitch < heights[0]                                # the spectra overlap


def test_each_spectrum_is_the_same_height_on_its_own_scale(qapp):
    canvas = _stack(4, [1.0, 5.0, 2.0, 3.0])
    _, heights = _baselines(canvas)
    assert np.allclose(heights, heights[0], rtol=1e-6)


def test_bottom_spectrum_keeps_the_axis_gap_whatever_the_stack_gap(qapp):
    canvas = _stack(3)
    canvas.set_style(baseline_gap=0.1)
    for gap in (-0.6, 0.05, 0.9):
        canvas.set_stack_gap(gap)
        base, heights = _baselines(canvas)
        axis_line = canvas._compare_axes[-1].get_position().y0
        assert (base[-1] - axis_line) / heights[-1] == pytest.approx(0.1, abs=1e-6)


def test_a_zoomed_view_survives_a_spacing_change(qapp):
    canvas = _stack(4)
    canvas.show_compare([_spec(1.0, 3 + i) for i in range(4)], mode="stack")
    top = canvas._compare_axes[0]
    top.set_xlim(6, 2)
    fitted = tuple(top._opennmr_fitted_ylim)
    lo, hi = fitted
    top.set_ylim(lo + 0.25 * (hi - lo), lo + 0.75 * (hi - lo))       # zoomed in y
    canvas.set_stack_gap(0.7)
    assert top.get_xlim() == pytest.approx((6, 2))                   # x untouched
    new_lo, new_hi = top._opennmr_fitted_ylim
    got_lo, got_hi = top.get_ylim()
    assert (got_lo - new_lo) / (new_hi - new_lo) == pytest.approx(0.25, abs=1e-9)
    assert (got_hi - new_lo) / (new_hi - new_lo) == pytest.approx(0.75, abs=1e-9)


def test_an_unzoomed_row_still_refits(qapp):
    canvas = _stack(3)
    before = tuple(canvas._compare_axes[0].get_ylim())
    canvas.set_stack_gap(0.9)
    assert tuple(canvas._compare_axes[0].get_ylim()) != before


def test_a_single_stacked_spectrum_and_extreme_gaps_do_not_fail(qapp):
    canvas = MplCanvas()
    canvas.show_compare([_spec(1.0), _spec(2.0)], mode="stack")
    canvas.set_stack_gap(-5.0)                                       # clamped
    canvas.set_stack_gap(50.0)
    base, _ = _baselines(canvas)
    assert np.all(np.isfinite(base))
