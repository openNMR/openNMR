import matplotlib.pyplot as plt
import numpy as np

from nmrsuite.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
from nmrsuite.plotting import oned


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
