"""2026-10-21: the crosshair's colour and line width -- on a plot's Crosshair
button (right-click) and in Settings ▸ Interface ▸ Crosshair; remembered,
and applied to every plot together."""
import json

import pytest
from PySide6.QtWidgets import QApplication

from test_gui import _load_spectrum_directly, _make_1d_peak_spectrum_with_pulse  # noqa: F401


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(qapp):
    from opennmr.gui.main_window import MainWindow
    w = MainWindow()
    w.resize(1200, 800)
    yield w
    w.close()


def _live_crosshair(canvas):
    canvas._ensure_crosshair()
    assert canvas._crosshair is not None
    return canvas._crosshair


def test_default_style(window):
    from opennmr.gui.widgets.canvas import MplCanvas
    assert window.canvas.crosshair_style() == (MplCanvas.DEFAULT_CROSSHAIR_COLOR,
                                               MplCanvas.DEFAULT_CROSSHAIR_WIDTH)


def test_the_style_reaches_the_lines(window):
    _load_spectrum_directly(window, _make_1d_peak_spectrum_with_pulse())
    canvas = window.canvas
    canvas.set_crosshair_style("#ff0000", 2.5)
    lines = _live_crosshair(canvas)._vlines + _live_crosshair(canvas)._hlines
    assert lines
    for line in lines:
        assert line.get_linewidth() == pytest.approx(2.5)
        assert line.get_color() == "#ff0000"


def test_changing_the_style_replaces_a_live_crosshair(window):
    _load_spectrum_directly(window, _make_1d_peak_spectrum_with_pulse())
    canvas = window.canvas
    first = _live_crosshair(canvas)
    canvas.set_crosshair_style("blue", 1.5)
    assert canvas._crosshair is None                  # dropped, rebuilt on the next move
    second = _live_crosshair(canvas)
    assert second is not first
    assert second._vlines[0].get_color() == "blue"


def test_a_bad_colour_or_width_is_ignored(window):
    canvas = window.canvas
    canvas.set_crosshair_style("#00ff00", 1.0)
    canvas.set_crosshair_style("not a colour", 99.0)
    assert canvas.crosshair_style() == ("#00ff00", 1.0)
    canvas.set_crosshair_style(None, "wide")
    assert canvas.crosshair_style() == ("#00ff00", 1.0)
    canvas.set_crosshair_style(None, 0.0)
    assert canvas.crosshair_style() == ("#00ff00", 1.0)


def test_reset_goes_back_to_the_default(window):
    from opennmr.gui.widgets.canvas import MplCanvas
    window.canvas.set_crosshair_style("red", 3.0)
    window.canvas.reset_crosshair_style()
    assert window.canvas.crosshair_style() == (MplCanvas.DEFAULT_CROSSHAIR_COLOR,
                                               MplCanvas.DEFAULT_CROSSHAIR_WIDTH)


def test_the_window_applies_a_style_to_every_plot(window):
    window._set_crosshair_style("#123456", 2.0)
    canvases = window._all_canvases()
    assert len(canvases) > 3
    assert all(c.crosshair_style() == ("#123456", 2.0) for c in canvases)
    assert window.crosshair_width_spin.value() == pytest.approx(2.0)
    assert window.crosshair_colour_button.text() == "#123456"


def test_a_choice_made_on_one_plots_button_reaches_all_plots(window):
    window.compare_canvas.crosshairStyleChanged.emit("#abcdef", 3.5)
    assert all(c.crosshair_style() == ("#abcdef", 3.5) for c in window._all_canvases())


def test_the_settings_box_changes_every_plot(window):
    window.crosshair_width_spin.setValue(4.0)
    assert all(c.crosshair_style()[1] == pytest.approx(4.0) for c in window._all_canvases())


def test_the_style_is_remembered(qapp):
    from opennmr.gui.main_window import MainWindow
    first = MainWindow()
    first._set_crosshair_style("#336699", 1.75)
    with open(MainWindow._UI_SETTINGS_PATH, encoding="utf-8") as handle:
        saved = json.load(handle)
    assert saved["crosshair_color"] == "#336699"
    assert saved["crosshair_width"] == pytest.approx(1.75)
    first.close()

    second = MainWindow()
    try:
        assert all(c.crosshair_style() == ("#336699", 1.75) for c in second._all_canvases())
        assert second.crosshair_width_spin.value() == pytest.approx(1.75)
    finally:
        second.close()


def test_the_reset_button_restores_the_default(window):
    from opennmr.gui.widgets.canvas import MplCanvas
    window._set_crosshair_style("red", 5.0)
    window.crosshair_style_reset_button.click()
    assert window.canvas.crosshair_style() == (MplCanvas.DEFAULT_CROSSHAIR_COLOR,
                                               MplCanvas.DEFAULT_CROSSHAIR_WIDTH)


def test_a_saved_bad_value_does_not_break_start_up(qapp):
    import os
    from opennmr.gui.main_window import MainWindow
    os.makedirs(os.path.dirname(MainWindow._UI_SETTINGS_PATH), exist_ok=True)
    with open(MainWindow._UI_SETTINGS_PATH, "w", encoding="utf-8") as handle:
        json.dump({"crosshair_color": "banana", "crosshair_width": "thick"}, handle)
    window = MainWindow()
    try:
        assert window.canvas.crosshair_style()[0] == "0.35"
    finally:
        window.close()
