"""Explicit user reports, 2026-10-15:

* "The legend label size when computing s/n changes compared to other
  legend label sizes (e.g., in compare or when showing just one
  spectrum) ... Same goes for the text that describes the s/n value."
  The S/N plot named its rows and wrote its S/N values at fixed
  fractions of 12 pt (7.2 pt) whatever the plot style said.
* "the 'Result' table when computing the s/n is really hard to see ...
  Happy for this to be a table, but just make it easier to read and
  something where you do not have to scroll lots to see everything."
* A Mac user's command line filled with
  ``AttributeError: 'NoneType' object has no attribute 'button'`` from
  matplotlib's ``drag_zoom``: a zoom drag cancelled by a figure rebuild
  had its state cleared but its motion callback left connected.
"""

import matplotlib
import numpy as np
import pytest
from matplotlib.backend_bases import MouseButton, MouseEvent

from opennmr.notebook import snr

from test_gui import qapp  # noqa: F401 -- qapp is a fixture
from test_gui_snr import _load as _load_snr, _spectrum


# ---- notebook: text sizes and naming --------------------------------------------------

def _two():
    return [_spectrum(40.0, 0, 16), _spectrum(80.0, 1, 64)]


def test_plot_snr_text_sizes_and_names():
    matplotlib.use("Agg")
    spectra = _two()
    result = snr.signal_to_noise_many(spectra, (5, -5), [(80, 40)], labels=["a", "b"])
    fig = snr.plot_snr(spectra, result, names="legend", legend_fontsize=15,
                       annotation_fontsize=13)
    ax = fig.axes[0]
    assert [t.get_fontsize() for t in ax.get_legend().get_texts()] == [15, 15]
    assert {t.get_fontsize() for t in ax.texts} == {13}          # only the S/N values
    fig = snr.plot_snr(spectra, result, names="labels", legend_fontsize=15)
    ax = fig.axes[0]
    assert ax.get_legend() is None
    assert {t.get_text(): t.get_fontsize() for t in ax.texts}["a"] == 15
    # The notebook default is unchanged: names on the rows of a stack.
    fig = snr.plot_snr(spectra, result)
    assert fig.axes[0].get_legend() is None and len(fig.axes[0].texts) == 4
    with pytest.raises(ValueError):
        snr.plot_snr(spectra, result, names="sideways")


def test_snr_summary_is_one_row_per_quantity():
    spectra = _two()
    result = snr.signal_to_noise_many(spectra, (5, -5), [(80, 40)], labels=["a", "b"],
                                      reference=0)
    summary = snr.snr_summary(result)
    by_name = {row["short"]: row for row in summary}
    assert summary[0]["short"].startswith("S/N")
    assert all(len(row["values"]) == 2 and len(row["notes"]) == 2 for row in summary)
    r = result["rows"][1]
    assert float(by_name["H/2σ  (TopSpin ×½)"]["values"][1]) == pytest.approx(r["sino"],
                                                                            rel=1e-3)
    assert by_name["Ratio to reference"]["values"][0] == "reference"
    assert by_name["Ratio to reference"]["values"][1].endswith("*")      # 40 vs 80: differs
    assert by_name["Detected"]["values"] == ["yes", "yes"]
    assert by_name["Checks"]["values"] == ["OK", "OK"]
    # Every convention snr_conventions gives is in the table, RMS as the S/N row.
    names = {row["name"] for row in summary}
    assert {c["name"] for c in snr.snr_conventions(r) if c["key"] != "snr"} <= names
    # One spectrum: no comparison row.
    one = snr.signal_to_noise_many(spectra[:1], (5, -5), [(80, 40)])
    assert "Ratio to reference" not in {row["short"] for row in snr.snr_summary(one)}


# ---- the GUI: sizes follow the canvas's legend, and the result table ------------------

def _calculated(window):
    panel = _load_snr(window)
    panel.set_signal_region(5, -5)
    panel.set_noise_region([(80, 40)])
    panel.calculate(explicit=True)
    return panel


def test_snr_text_matches_the_canvas_legend(qapp, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: None)
    from opennmr.gui.main_window import MainWindow
    window = MainWindow()
    _calculated(window)
    canvas = window.compare_canvas
    size = canvas._legend_kwargs()["fontsize"]
    ax = canvas.figure.axes[0]
    assert {t.get_fontsize() for t in ax.get_legend().get_texts()} == {size}
    assert {t.get_fontsize() for t in ax.texts if t.get_text().startswith("S/N")} == {size}
    canvas.set_style(legend_fontsize=19)              # Plot Style: redraws the figure
    ax = canvas.figure.axes[0]
    assert {t.get_fontsize() for t in ax.get_legend().get_texts()} == {19}
    assert {t.get_fontsize() for t in ax.texts if t.get_text().startswith("S/N")} == {19}
    canvas.set_style(legend_style="labels")           # names on the traces instead
    ax = canvas.figure.axes[0]
    assert ax.get_legend() is None
    assert any(t.get_text().startswith("snr0") and t.get_fontsize() == 19 for t in ax.texts)


def test_result_table_shows_everything_without_scrolling(qapp, monkeypatch):
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QMessageBox
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: None)
    from opennmr.gui.main_window import MainWindow
    window = MainWindow()
    panel = _calculated(window)
    table = panel.result_table
    summary = snr.snr_summary(panel.result)
    assert not table.isHidden() and table.rowCount() == len(summary)
    assert table.columnCount() == 2
    assert table.verticalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAlwaysOff
    assert table.height() == table.content_height()
    assert table.height() >= sum(table.rowHeight(i) for i in range(table.rowCount()))
    assert table.item(0, 0).text() == summary[0]["values"][0]
    assert table.horizontalHeaderItem(0).toolTip().startswith("snr0")
    assert "Same regions for every spectrum" in panel.result_status.text()
    # The sentences moved to Full Report…, which repeats the table, unelided.
    assert "Same regions for every spectrum" in panel.result_label.text()
    assert panel.report_dialog.table.rowCount() == len(summary)
    assert panel.report_dialog.table.horizontalHeaderItem(0).text() == \
        panel.result["rows"][0]["label"]
    panel.show_report()
    assert panel.report_dialog.isVisible()
    # Changing a region hides the stale result.
    panel.set_signal_region(4, -4)
    assert table.isHidden() and "calculate again" in panel.result_status.text()


# ---- the Mac traceback: a cancelled drag leaves no motion callback behind ---------------

class _Errors(list):
    def __call__(self, exc):
        self.append(exc)


def _canvas_with_plot():
    from opennmr.gui.widgets.canvas import MplCanvas
    canvas = MplCanvas()
    canvas.resize(600, 400)
    ax = canvas.figure.add_subplot(111)
    ax.plot(np.arange(10), np.arange(10))
    canvas.canvas.draw()
    errors = _Errors()
    canvas.canvas.callbacks.exception_handler = errors
    return canvas, ax, errors


def _event(canvas, name, x, y, **kwargs):
    return MouseEvent(name, canvas.canvas, x, y, **kwargs)


def _press(canvas, ax, method):
    x, y = ax.transAxes.transform((0.5, 0.5))
    getattr(canvas.toolbar, method)(_event(canvas, "button_press_event", x, y,
                                           button=MouseButton.LEFT))
    return x, y


def _move(canvas, x, y):
    canvas.canvas.callbacks.process(
        "motion_notify_event",
        _event(canvas, "motion_notify_event", x + 20, y + 20, buttons={MouseButton.LEFT}))


def test_cancelled_zoom_drag_disconnects_its_callback(qapp):
    canvas, ax, errors = _canvas_with_plot()
    x, y = _press(canvas, ax, "press_zoom")
    assert canvas.toolbar._zoom_info is not None
    canvas._cancel_navigation_drag()                 # a figure rebuild mid-drag
    assert canvas.toolbar._zoom_info is None
    _move(canvas, x, y)
    assert errors == []


def test_cancelled_pan_drag_restores_the_toolbar_motion(qapp):
    canvas, ax, errors = _canvas_with_plot()
    x, y = _press(canvas, ax, "press_pan")
    assert canvas.toolbar._pan_info is not None
    canvas._cancel_navigation_drag()
    assert canvas.toolbar._pan_info is None
    _move(canvas, x, y)
    assert errors == []
    # The toolbar's own mouse_move (coordinates read-out) is connected again.
    registry = canvas.canvas.callbacks.callbacks["motion_notify_event"]
    assert canvas.toolbar._id_drag in registry


def test_a_second_press_mid_drag_leaves_no_orphan(qapp):
    """matplotlib's press overwrites an unfinished drag (a second button
    mid-drag; Ctrl-click is the right button on a Mac trackpad)."""
    canvas, ax, errors = _canvas_with_plot()
    x, y = _press(canvas, ax, "press_zoom")
    _press(canvas, ax, "press_zoom")
    canvas.toolbar.release_zoom(_event(canvas, "button_release_event", x, y,
                                       button=MouseButton.LEFT))
    assert canvas.toolbar._zoom_info is None
    _move(canvas, x, y)
    _move(canvas, x, y)
    assert errors == []


def test_an_orphaned_drag_callback_disconnects_itself(qapp):
    canvas, ax, errors = _canvas_with_plot()
    x, y = _press(canvas, ax, "press_zoom")
    canvas.toolbar._zoom_info = None                 # cleared by anything else
    _move(canvas, x, y)
    assert errors == []
    registry = canvas.canvas.callbacks.callbacks["motion_notify_event"]
    assert not any(getattr(ref(), "__func__", None) is type(canvas.toolbar).drag_zoom
                   for ref in registry.values())
