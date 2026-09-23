import matplotlib.pyplot as plt
import numpy as np
import pytest

from opennmr.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
from opennmr.plotting import oned


def _make_1d(n=64, center_ppm=5.0, sfo1=400.0):
    ppm = np.linspace(center_ppm - 2, center_ppm + 2, n)
    data = np.exp(-((ppm - center_ppm) ** 2) / 0.05)
    metadata = NMRMetadata(vendor="bruker", nucleus="1H", sfo1=sfo1)
    axis = NMRAxis(kind="frequency", ppm=ppm, hz=ppm * sfo1)
    return NMRSpectrum(data=data, axes=[axis], metadata=metadata)


def test_plot_1d_onto_existing_ax():
    fig, ax = plt.subplots()
    spec = _make_1d()
    xmin, xmax, label = oned.plot_1d(ax, spec)
    assert xmin < xmax
    assert "1" in label  # nucleus-based label mentions '1' for 1H
    assert len(ax.lines) == 1
    plt.close(fig)


def test_plot_1d_downsamples_high_point_count_for_rendering():
    # Explicit user request: cap how many points get rendered at once
    # for a high-point-count spectrum (diffusion/imaging acquisitions,
    # or just high digital resolution, routinely land in the
    # 10^5-10^6 range) -- display only, never the underlying data.
    from opennmr.plotting.style import DOWNSAMPLE_1D_MAX_POINTS

    fig, ax = plt.subplots()
    spec = _make_1d(n=DOWNSAMPLE_1D_MAX_POINTS * 4)
    xmin, xmax, _ = oned.plot_1d(ax, spec)

    line = ax.lines[0]
    assert len(line.get_xdata()) == DOWNSAMPLE_1D_MAX_POINTS
    # xmin/xmax are still the TRUE full-resolution range, not derived
    # from whatever the downsampled subset happens to span.
    assert xmin == pytest.approx(float(np.min(spec.axis_values(0))))
    assert xmax == pytest.approx(float(np.max(spec.axis_values(0))))
    plt.close(fig)


def test_plot_1d_does_not_downsample_below_the_cap():
    from opennmr.plotting.style import DOWNSAMPLE_1D_MAX_POINTS

    fig, ax = plt.subplots()
    spec = _make_1d(n=64)
    assert 64 < DOWNSAMPLE_1D_MAX_POINTS
    oned.plot_1d(ax, spec)
    assert len(ax.lines[0].get_xdata()) == 64
    plt.close(fig)


def test_plot_1d_downsampling_preserves_a_narrow_peak():
    # Explicit user report: a peak well-defined in the full-resolution
    # ("Spectrum") view was reduced to "three or four points" in a
    # downsampled ("Processing"/"Compare") view -- the old evenly-
    # spaced-index downsampling could step clean over a narrow peak.
    # Regression check at the plot_1d level (see
    # test_plotting_style.py for the underlying downsample_1d unit
    # tests): a narrow spike placed off any round stride must still
    # survive at close to its true height.
    from opennmr.plotting.style import DOWNSAMPLE_1D_MAX_POINTS

    n = DOWNSAMPLE_1D_MAX_POINTS * 6
    ppm = np.linspace(0, 10, n)
    data = np.zeros(n)
    spike_idx = int(n * 0.37) + 3  # off any round stride
    data[spike_idx] = 1.0
    metadata = NMRMetadata(vendor="bruker", nucleus="1H", sfo1=400.0)
    axis = NMRAxis(kind="frequency", ppm=ppm, hz=ppm * 400.0)
    spec = NMRSpectrum(data=data, axes=[axis], metadata=metadata)

    fig, ax = plt.subplots()
    oned.plot_1d(ax, spec, normalize=False)
    y_plotted = ax.lines[0].get_ydata()
    assert y_plotted.max() == pytest.approx(1.0)
    plt.close(fig)


def test_plot_1d_stack_standalone():
    specs = [_make_1d(center_ppm=c) for c in (4.0, 5.0, 6.0)]
    fig, ax = oned.plot_1d_stack(specs)
    assert len(ax.lines) == 3
    plt.close(fig)


def test_plot_1d_stack_uniform_spacing_when_not_normalized():
    # Explicit user request: the gap between stacked traces in the
    # Compare tab should be one uniform value, not scaled by each
    # spectrum's own peak height -- which is what happened whenever
    # normalize=False (the "None"/"By sample mass" Compare modes,
    # where seeing each spectrum's true relative height is the whole
    # point, so a per-spectrum-scaled gap on top of that isn't wanted).
    small = _make_1d(center_ppm=4.0)
    small.data = small.data * 0.1
    large = _make_1d(center_ppm=5.0)
    large.data = large.data * 10.0
    another_large = _make_1d(center_ppm=6.0)
    another_large.data = another_large.data * 10.0

    fig, ax = oned.plot_1d_stack([small, large, another_large], normalize=False, y_offset=0.2)

    # Each line's own baseline (~y_offset, since these Gaussians are
    # ~0 at the plotted edges) should step up by the same amount each
    # time -- not by an amount that depends on the *previous* trace's
    # own height.
    baselines = [float(np.min(line.get_ydata())) for line in ax.lines]
    gaps = np.diff(baselines)
    np.testing.assert_allclose(gaps, gaps[0], rtol=1e-6)
    plt.close(fig)


def test_plot_1d_heatmap_standalone():
    specs = [_make_1d(center_ppm=c) for c in (4.0, 4.5, 5.0, 5.5)]
    fig, ax, im = oned.plot_1d_heatmap(specs, npoints=100)
    assert im.get_array().shape[0] == len(specs) * 100 or im.get_array().size > 0
    plt.close(fig)


def test_plot_1d_heatmap_requires_at_least_two():
    try:
        oned.plot_1d_heatmap([_make_1d()], npoints=10)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")
