"""Extracted slices must never be clipped by the slice panel's own limits.

User report (2026-10-28): exporting a 2D with extracted slices could cut
the top slice(s) off, partly or wholly. Cause: the panel's top limit was a
fixed 1.0, yet each trace is ``slot * offset_fraction`` tall and the GUI
feeds ``1 / (1 + stack_gap)`` in, so a negative stack gap (overlapping
spectra) pushed the upper traces past the axes and they were clipped.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pytest

from opennmr.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
from opennmr.plotting.twod import SLICE_PANEL_GID, plot_2d_contour


def _spec(n1=48, n2=256):
    f2 = np.linspace(40, -40, n2)
    f1 = np.linspace(20, -20, n1)
    data = np.zeros((n1, n2))
    for r in range(n1):
        data[r] += np.exp(-((f2 - (0.2 * r - 5)) / 3) ** 2) * np.exp(-(f1[r] / 8) ** 2) * 1e6
    data += np.random.default_rng(0).normal(0, 1e3, data.shape)
    axes = [NMRAxis(kind="frequency", ppm=f1, hz=f1 * 400),
            NMRAxis(kind="frequency", ppm=f2, hz=f2 * 400)]
    md = NMRMetadata(vendor="bruker", nucleus="1H", sfo1=400.0,
                     nucleus_indirect="1H", sfo1_indirect=400.0)
    return NMRSpectrum(data=data, axes=axes, metadata=md)


def _panel(fig):
    return next(a for a in fig.axes if a.get_gid() == SLICE_PANEL_GID)


@pytest.mark.parametrize("offset_fraction", [0.5, 0.85, 1.0, 1.5, 4.0, 10.0])
@pytest.mark.parametrize("n_slices", [1, 3, 6])
def test_every_trace_sits_inside_the_panel(offset_fraction, n_slices):
    sl = [{"axis": "f1", "index": int(i), "label": f"{i}"}
          for i in np.linspace(6, 42, n_slices)]
    fig, _ax = plot_2d_contour(_spec(), slices=sl,
                               slice_offset_fraction=offset_fraction)[:2]
    panel = _panel(fig)
    lo, hi = panel.get_ylim()
    tops = [np.nanmax(line.get_ydata()) for line in panel.lines]
    bottoms = [np.nanmin(line.get_ydata()) for line in panel.lines]
    assert len(tops) == n_slices
    assert max(tops) <= hi + 1e-9, "the top trace pokes out of the panel"
    assert min(bottoms) >= lo - 1e-9
    matplotlib.pyplot.close(fig)


def test_stack_gap_negative_does_not_clip_slices_in_the_gui():
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    from opennmr.gui.widgets.canvas import MplCanvas
    c = MplCanvas()
    c.resize(900, 700)
    c.set_contour_slices([{"axis": "f1", "index": int(i), "label": f"{i}"}
                          for i in np.linspace(6, 42, 6)])
    c.set_stack_gap(-0.6)
    c.show_spectrum(_spec())
    app.processEvents()
    panel = next(a for a in c.figure.axes if a.get_gid() == SLICE_PANEL_GID)
    hi = panel.get_ylim()[1]
    assert max(np.nanmax(l.get_ydata()) for l in panel.lines) <= hi + 1e-9
