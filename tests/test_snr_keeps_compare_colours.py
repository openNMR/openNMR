"""Explicit user report, 2026-10-15: "When doing the s/n calculation,
the colour of the spectrum line changes; stop this from happening."

The S/N plot coloured its spectra from a fresh palette sized to the
spectra ticked, so measuring two of three gave the second one the
third's colour, and a single spectrum the plot's trace colour, whatever
Compare showed. It now takes each spectrum's colour from Compare.
"""

import matplotlib
import matplotlib.colors as mcolors
import numpy as np
import pytest
from PySide6.QtCore import Qt

from opennmr.notebook import snr

from test_gui import qapp  # noqa: F401 -- qapp is a fixture
from test_gui_snr import _load as _load_snr, _spectrum


def test_plot_snr_takes_given_colours():
    matplotlib.use("Agg")
    spectra = [_spectrum(40.0, 0), _spectrum(80.0, 1)]
    fig = snr.plot_snr(spectra, regions=[{"signal": (5, -5), "noise": (80, 40)}] * 2,
                       labels=["a", "b"], colours=["#ff0000", None], names="legend")
    by_label = {ln.get_label(): mcolors.to_hex(ln.get_color()) for ln in fig.axes[0].lines}
    assert by_label["a"] == "#ff0000"
    assert by_label["b"] != "#ff0000"                     # None: the palette's
    with pytest.raises(ValueError):
        snr.plot_snr(spectra, regions=[None, None], colours=["#ff0000"])


def _lines(canvas):
    return {ln.get_label()[:4]: mcolors.to_hex(ln.get_color())
            for ax in canvas.figure.axes for ln in ax.lines if not ln.get_label().startswith("_")}


@pytest.mark.parametrize("ticked", [[0, 1, 2], [0, 2], [1]])
def test_snr_keeps_each_spectrums_compare_colour(qapp, monkeypatch, ticked):
    from PySide6.QtWidgets import QMessageBox
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: None)
    from opennmr.gui.main_window import MainWindow
    window = MainWindow()
    panel = _load_snr(window, heights=(40.0, 80.0, 60.0), ns=(16, 64, 32))
    window._rebuild_compare_list([(i, None) for i in range(len(window.session))],
                                 preserve_previous=False)
    for i in range(window.compare_list.count()):
        window.compare_list.item(i).setCheckState(Qt.CheckState.Checked)
    window._refresh_compare_view(switch_tab=True, force_render=True)
    canvas = window.compare_canvas
    compare = _lines(canvas)
    labels = [e["label"] for e in panel._rows if e["label"].startswith("snr")]
    panel.set_checked([labels[i] for i in ticked])
    panel.set_signal_region(5, -5)
    panel.set_noise_region([(80, 40)])
    panel.calculate(explicit=True)
    shown = _lines(canvas)
    assert set(shown) == {f"snr{i}" for i in ticked}
    for name, colour in shown.items():
        assert colour == compare[name]
    # Recalculating keeps them too.
    panel.set_signal_region(4, -4)
    panel.calculate(explicit=True)
    assert _lines(canvas) == shown


def test_a_colour_set_in_compare_carries_over(qapp, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: None)
    from opennmr.gui.main_window import MainWindow
    window = MainWindow()
    panel = _load_snr(window)
    window._rebuild_compare_list([(i, None) for i in range(len(window.session))],
                                 preserve_previous=False)
    for i in range(window.compare_list.count()):
        window.compare_list.item(i).setCheckState(Qt.CheckState.Checked)
    window._refresh_compare_view(switch_tab=True, force_render=True)
    canvas = window.compare_canvas
    canvas.set_series_style(canvas._series["compare"][1].key, colour="#123456")
    labels = [e["label"] for e in panel._rows if e["label"].startswith("snr")]
    panel.set_checked([labels[1]])
    panel.show_spectra()
    assert _lines(canvas) == {"snr1": "#123456"}
    # Not in Compare: the palette, as before.
    assert window._compare_colours_for([{"index": 99}]) == [None]
