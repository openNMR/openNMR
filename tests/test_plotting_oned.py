import matplotlib.pyplot as plt
import numpy as np

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
