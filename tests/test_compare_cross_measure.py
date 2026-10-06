"""Measure in Compare: from a point on one spectrum to a point on another
(user report, 2026-10-06: "if I click on one point on one spectrum, then
click again on a second point on another spectrum, I cannot measure")."""

import numpy as np
import pytest
from matplotlib.patches import ConnectionPatch

from opennmr.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
from opennmr.gui.widgets.canvas import MplCanvas

from test_gui_canvas import qapp, _make_1d, _make_1d_at, _press_at  # noqa: F401 -- fixture


def _stack(*spectra):
    canvas = MplCanvas()
    canvas.figure.set_size_inches(8, 6)
    canvas.show_compare(list(spectra), mode="stack")
    canvas.canvas.draw()
    canvas.set_measurement_mode(True)
    got = []
    canvas.measurementCompleted.connect(got.append)
    return canvas, got


def test_a_point_on_one_row_to_a_point_on_another(qapp):
    canvas, got = _stack(_make_1d_at(4.0), _make_1d_at(6.0))
    top, bottom = canvas._compare_axes
    canvas._on_press(_press_at(canvas, top, xfrac=0.3, yfrac=0.5))
    canvas._on_press(_press_at(canvas, bottom, xfrac=0.7, yfrac=0.2))
    assert len(got) == 1
    result = got[0]
    assert result["across"] == (0, 1)
    assert result["dx"]["ppm"] == pytest.approx(4.0, rel=1e-3)
    assert result["dx"]["hz"] == pytest.approx(1600.0, rel=1e-3)
    assert "ppm" in result["summary"] and "Hz" in result["summary"]
    assert "Δy" not in result["summary"]       # rows have their own intensity scales
    assert len(result["spectra"]) == 2
    arrows = [a for a in canvas._measurement_artists if isinstance(a, ConnectionPatch)]
    assert len(arrows) == 1 and arrows[0] in canvas.figure.artists
    # Switching Measure off takes it all away.
    canvas.set_measurement_mode(False)
    assert arrows[0] not in canvas.figure.artists


def test_the_same_row_still_measures_as_before(qapp):
    canvas, got = _stack(_make_1d(), _make_1d())
    row = canvas._compare_axes[1]
    canvas._on_press(_press_at(canvas, row, xfrac=0.3, yfrac=0.5))
    canvas._on_press(_press_at(canvas, row, xfrac=0.7, yfrac=0.5))
    assert "across" not in got[0] and got[0]["dx"]["ppm"] == pytest.approx(4.0, rel=1e-3)


def test_different_fields_are_said(qapp):
    ppm = np.linspace(0, 10, 100)
    other = NMRSpectrum(data=np.exp(-((ppm - 5) ** 2)),
                        axes=[NMRAxis(kind="frequency", ppm=ppm, hz=ppm * 600.0)],
                        metadata=NMRMetadata(vendor="bruker", nucleus="1H", sfo1=600.0))
    canvas, got = _stack(_make_1d(), other)
    top, bottom = canvas._compare_axes
    canvas._on_press(_press_at(canvas, top, xfrac=0.3))
    canvas._on_press(_press_at(canvas, bottom, xfrac=0.7))
    assert got[0]["dx"]["hz"] == pytest.approx(1600.0, rel=1e-3)      # the first spectrum's
    assert "the two differ" in got[0]["summary"]


def test_a_click_off_the_stack_still_restarts(qapp):
    canvas, got = _stack(_make_1d(), _make_1d())
    other_ax = canvas.figure.add_axes([0.0, 0.0, 0.05, 0.05])
    canvas._record_measurement_point(canvas._compare_axes[0], 2.0, 0.5)
    canvas._record_measurement_point(other_ax, 3.0, 0.5)
    assert got == [] and len(canvas._measurement_points) == 1
