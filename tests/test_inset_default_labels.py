"""Inset axis labels with no text of their own take a default (explicit user
report, 2026-11-10: ticking "x-axis label" drew nothing on a Compare stack,
whose rows other than the bottom carry no x label, and a spectrum has no y
label at all)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from opennmr.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum  # noqa: E402
from opennmr.gui.widgets.canvas import MplCanvas  # noqa: E402
from opennmr.plotting import inset  # noqa: E402

from test_gui import qapp  # noqa: E402,F401


def _two_rows():
    fig, (top, bottom) = plt.subplots(2, 1, sharex=True)
    x = np.linspace(0, 10, 500)
    for ax, x0 in ((top, 3.0), (bottom, 7.0)):
        ax.plot(x, np.exp(-((x - x0) / 0.1) ** 2))
    bottom.set_xlabel("δ (ppm)")
    return fig, top, bottom


def test_default_x_label_comes_from_the_shared_axis():
    fig, top, _bottom = _two_rows()
    assert top.get_xlabel() == ""
    assert inset.default_label(top, "x") == "δ (ppm)"
    ax = inset.add_inset(top, xlim=(2.5, 3.5), show_xlabel=True, show_ylabel=True)
    assert ax.get_xlabel() == "δ (ppm)"
    assert ax.get_ylabel() == "Intensity"           # nothing on the figure: the quantity
    plt.close(fig)


def test_the_plots_own_label_and_typed_text_still_win():
    fig, top, bottom = _two_rows()
    top.set_ylabel("Signal")
    ax = inset.add_inset(top, xlim=(2.5, 3.5), show_ylabel=True)
    assert ax.get_ylabel() == "Signal"
    ax = inset.add_inset(bottom, xlim=(6.5, 7.5), show_xlabel=True, xlabel="mine")
    assert ax.get_xlabel() == "mine"
    plt.close(fig)


def _trace(x0, name):
    ppm = np.linspace(10.0, 0.0, 2000)
    y = np.exp(-((ppm - x0) / 0.05) ** 2)
    return NMRSpectrum(data=y, axes=[NMRAxis(kind="frequency", ppm=ppm)],
                       metadata=NMRMetadata(vendor="derived", nucleus="1H", sfo1=400.0,
                                            source_path=name))


def test_a_compare_stack_inset_on_the_top_row_gets_the_x_label(qapp):
    canvas = MplCanvas()
    canvas.show_compare([_trace(5.0, "A"), _trace(6.0, "B")], mode="stack")
    rows = canvas._plain_axes()
    canvas.add_inset_region((5.3, 4.7), (0.0, 1.0), ax=rows[0],
                            show_xlabel=True, show_ylabel=True)
    canvas.canvas.draw()
    child = canvas._plain_axes()[0].child_axes[0]
    assert child.get_xlabel() == canvas._plain_axes()[-1].get_xlabel() != ""
    assert child.get_ylabel() != ""
