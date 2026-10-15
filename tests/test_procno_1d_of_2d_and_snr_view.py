"""Explicit user reports, 2026-10-15:

* "When loading a 2-d spectrum with several 1-d procnos, an error
  appears" -- [Errno 2] ... pdata\\99389\\proc2s. A 1D proc number of a
  2D (a row, column or projection TopSpin wrote there) has procs but no
  proc2s; the reader always took the 2D path for a ser. It now opens as
  TopSpin's processed 1D, on the axis of the dimension it lies along.
* "When computing s/n, all plot styles disappear ... if you click drag,
  the line thickness changes, then when you have selected a region, you
  zoom in automatically, rather than maintaining the current view." The
  S/N plot drew its traces at a fixed 0.9 (not the plot style's width)
  and always zoomed to the regions.
"""

import os
import shutil

import matplotlib
import numpy as np
import pytest
from PySide6.QtCore import Qt

from opennmr import io
from opennmr.io import bruker
from opennmr.notebook import snr

from test_gui import _write_params, qapp  # noqa: F401 -- qapp is a fixture
from test_gui_snr import _load as _load_snr


def _experiment(tmp_path):
    """A 2D (ser) with pdata/1 a 2D, pdata/2 an F2 row and pdata/3 an F1
    column, each as TopSpin writes them (procs + 1r only)."""
    expt = tmp_path / "2261"
    (expt / "pdata" / "1").mkdir(parents=True)
    (expt / "ser").write_bytes(b"\x00" * 16)
    _write_params(expt / "acqus", {"NUC1": "1H", "BF1": 800.15, "SFO1": 800.152, "O1": 1600.0,
                                   "NS": 4, "SW_h": 32679.7, "SW": 40.84, "TD": 1024})
    _write_params(expt / "acqu2s", {"NUC1": "13C", "BF1": 201.197, "SFO1": 201.2, "O1": 9000.0,
                                    "SW_h": 14084.5, "SW": 70.0, "TD": 256, "FnMODE": 5})
    _write_params(expt / "pdata" / "1" / "procs", {"SW_p": 32679.7, "SI": 64, "SF": 800.152,
                                                   "WDW": 4, "SSB": 3, "PHC0": 40.6})
    _write_params(expt / "pdata" / "1" / "proc2s", {"SW_p": 14084.5, "SI": 32, "SF": 201.2,
                                                    "WDW": 4, "SSB": 3})
    row = (np.arange(64) * 1000).astype("int32")
    col = (np.arange(32) * 500).astype("int32")
    for procno, params, data in (
            (2, {"SW_p": 32679.7, "SI": 64, "SF": 800.152, "WDW": 4, "SSB": 3}, row),
            (3, {"SW_p": 14084.5, "SI": 32, "SF": 201.2, "WDW": 4, "SSB": 3}, col)):
        folder = expt / "pdata" / str(procno)
        folder.mkdir()
        _write_params(folder / "procs", params)
        data.tofile(folder / "1r")
    return str(expt)


def test_dimension_of_each_procno(tmp_path):
    path = _experiment(tmp_path)
    assert [bruker.procno_dimension(path, n) for n in (1, 2, 3, 9)] == [2, 1, 1, None]
    assert bruker.slice_dimension_of_procno(path, 2) == 1      # 1H: along F2
    assert bruker.slice_dimension_of_procno(path, 3) == 2      # 13C: along F1
    entries = {e["procno"]: e for e in bruker.available_procnos(path)}
    assert entries[1]["ndim"] == 2 and entries[2]["ndim"] == 1
    assert entries[2]["summary"].startswith("1D of this 2D")
    assert "SI 64" in entries[2]["summary"] and "64.0" not in entries[2]["summary"]


def test_a_1d_procno_of_a_2d_opens_as_topspin_processed_it(tmp_path):
    path = _experiment(tmp_path)
    row = io.load(path, vendor="bruker", procno=2, pseudo_2d=False, window="qsine")
    assert row.ndim == 1 and row.data.shape == (64,) and row.metadata.nucleus == "1H"
    assert row.data_imag is None                                # no 1i written
    assert "1D processing of this 2D" in row.metadata.raw["loaded_from_processed"]
    np.testing.assert_allclose(row.data / row.data.max(), np.arange(64) / 63)
    col = io.load(path, vendor="bruker", procno=3, pseudo_2d=False)
    assert col.metadata.nucleus == "13C" and col.data.shape == (32,)
    assert "along F1" in col.metadata.raw["loaded_from_processed"]


def test_the_gui_opens_a_1d_procno_of_a_2d(qapp, tmp_path, monkeypatch):
    from opennmr.gui.main_window import MainWindow
    path = _experiment(tmp_path)
    window = MainWindow()

    def answer(self, dialog):
        dialog.set_checked([2, 3])
        return True
    monkeypatch.setattr(MainWindow, "_run_procno_dialog", answer)
    rows = window._choose_procnos(path, "bruker")
    assert [window.session.spectra[r].ndim for r in rows] == [1, 1]
    assert [window.session.procno(r) for r in rows] == [2, 3]


GOOD_2D = os.path.join(os.path.dirname(__file__), "..", "test_data", "Good_2D", "28")


@pytest.mark.skipif(not os.path.isdir(GOOD_2D), reason="test_data/Good_2D/28 not present")
def test_real_2d_with_a_written_row(tmp_path):
    expt = tmp_path / "28"
    shutil.copytree(GOOD_2D, expt, ignore=shutil.ignore_patterns("opennmr_*"))
    rr = np.fromfile(expt / "pdata" / "1" / "2rr", dtype="int32")
    (expt / "pdata" / "99389").mkdir()
    shutil.copy(expt / "pdata" / "1" / "procs", expt / "pdata" / "99389" / "procs")
    rr[:2048].tofile(expt / "pdata" / "99389" / "1r")
    two_d = bruker.read_2d_processed(str(expt))
    row = io.load(str(expt), vendor="bruker", procno=99389)
    assert row.ndim == 1
    np.testing.assert_allclose(row.axes[0].ppm, two_d.axes[1].ppm)


# ---- the S/N plot keeps the view and the plot style's line width -------------------

def test_plot_snr_uses_the_style_line_width_and_given_view():
    matplotlib.use("Agg")
    x = np.linspace(100, -100, 2001)
    y = np.exp(-(x / 0.5) ** 2) * 100 + np.random.default_rng(0).normal(size=x.size)
    with matplotlib.rc_context({"lines.linewidth": 2.25}):
        fig = snr.plot_snr([(x, y)], regions=[{"signal": (5, -5), "noise": (80, 40)}],
                           xlims=(90, -30))
    line = [ln for ln in fig.axes[0].lines if ln.get_label() == "Spectrum 1"][0]
    assert line.get_linewidth() == 2.25
    assert fig.axes[0].get_xlim() == (90, -30)


def test_choosing_regions_keeps_the_view_on_screen(qapp, monkeypatch):
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
    compare_width = [ln for ln in canvas.figure.axes[0].lines
                     if not ln.get_label().startswith("_")][0].get_linewidth()
    for ax in canvas.figure.axes:
        ax.set_xlim(90, -30)                       # the user's zoom
    panel.set_signal_region(5, -5)
    panel._armed = "noise"
    panel.on_canvas_region(80, 40)                 # a drag on the plot
    ax = canvas.figure.axes[0]
    assert ax.get_xlim() == pytest.approx((90, -30))
    widths = {ln.get_linewidth() for ln in ax.lines if not ln.get_label().startswith("_")}
    assert widths == {compare_width}
    panel.calculate(explicit=True)
    assert canvas.figure.axes[0].get_xlim() == pytest.approx((90, -30))
    # A region typed outside the view: the plot widens to show it.
    panel.set_noise_region([(-60, -80)])
    panel.show_spectra()
    left, right = canvas.figure.axes[0].get_xlim()
    assert min(left, right) < -60
