"""2026-10-21: the grey "Not fitted" box left on the fit plot after a
several-range fit; the Apply button when nothing is left in its menu."""
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(qapp):
    from opennmr.gui.main_window import MainWindow
    w = MainWindow()
    w.resize(1400, 900)
    yield w
    w.close()


def _gap_patches(ax, window):
    return [p for p in ax.patches if p.get_gid() == window.FIT_GAP_GID]


def _zoom(canvas, left, right):
    """What the user's zoom does: the natural (Home) view was recorded when
    the fit was drawn, and the axes now differ from it."""
    canvas.note_natural_view()
    canvas.ax.set_xlim(left, right)


def _combined_result(regions, n=400):
    x = np.linspace(40.0, -10.0, n)
    y = 1.0 / (1.0 + ((x - 12.0) / 2.0) ** 2)
    return {"x": x, "y": y, "y_fit": y * 0.98, "site_spectra": {"Site 1": y * 0.98},
            "baseline": np.zeros(n), "settings": {"fit_regions": regions}, "spectrum": None}


def test_gap_box_is_cleared_when_the_next_fit_has_one_range(window):
    """Several ranges -> the gap is shaded. Zoom, then fit one range with
    ``keep_zoom`` (the live preview does): the old shading must not come back."""
    canvas = window.fit_canvas
    window._draw_combined_fit(_combined_result([(5.0, 10.0), (12.0, 30.0)]))
    assert len(_gap_patches(canvas.ax, window)) == 1
    _zoom(canvas, 28.0, 6.0)                            # the user zooms in

    window._draw_combined_fit(_combined_result([(5.0, 30.0)]), keep_zoom=True)

    assert _gap_patches(canvas.ax, window) == []
    assert canvas.slice_fit_overlay() is None


def test_gap_box_is_cleared_when_the_next_fit_is_the_whole_spectrum(window):
    canvas = window.fit_canvas
    window._draw_combined_fit(_combined_result([(5.0, 10.0), (12.0, 30.0)]))
    _zoom(canvas, 28.0, 6.0)

    window._draw_combined_fit(_combined_result(None), keep_zoom=True)

    assert _gap_patches(canvas.ax, window) == []


def test_zoom_is_still_kept_when_the_gap_box_is_cleared(window):
    canvas = window.fit_canvas
    window._draw_combined_fit(_combined_result([(5.0, 10.0), (12.0, 30.0)]))
    _zoom(canvas, 28.0, 6.0)
    window._draw_combined_fit(_combined_result([(5.0, 30.0)]), keep_zoom=True)
    assert canvas.ax.get_xlim() == pytest.approx((28.0, 6.0))


def test_gap_box_survives_a_restyle_of_the_same_fit(window):
    """A component-style change redraws the *same* fit: its shading stays."""
    canvas = window.fit_canvas
    window._draw_combined_fit(_combined_result([(5.0, 10.0), (12.0, 30.0)]))
    canvas.redraw_shown_fit() if hasattr(canvas, "redraw_shown_fit") else canvas.show_slice_fit(
        **canvas._slice_fit_args, _keep_view=True)
    assert len(_gap_patches(canvas.ax, window)) == 1


# --- Apply tile -------------------------------------------------------------

def _apply_tile_visible(window) -> bool:
    for index in range(window.side_tabs.count()):
        if window.side_tabs.tabText(index) == "Apply":
            return window.side_tabs.isTabVisible(index)
    raise AssertionError("no Apply tile")


def test_apply_tile_is_there_by_default(window):
    assert _apply_tile_visible(window)
    assert window._window_panel_actions["Apply"].isVisible() or \
        window._window_panel_actions["Apply"].isEnabled()


def test_apply_tile_goes_when_enmr_and_imaging_are_both_off_and_returns(window):
    window._set_section_shown("enmr", False)
    assert _apply_tile_visible(window)             # Imaging is still on it
    window._set_section_shown("imaging", False)
    assert _apply_tile_visible(window)             # Electrochemistry is still on it
    window._set_section_shown("echem", False)
    assert not _apply_tile_visible(window)
    assert not window._window_panel_actions["Apply"].isEnabled()

    window._set_section_shown("imaging", True)     # either one brings it back
    assert _apply_tile_visible(window)
    assert window._window_panel_actions["Apply"].isEnabled()
    window._set_section_shown("imaging", False)
    window._set_section_shown("enmr", True)
    assert _apply_tile_visible(window)


def test_apply_tile_via_the_settings_checkboxes(window):
    window.section_shown_checkboxes["enmr"].setChecked(False)
    window.section_shown_checkboxes["imaging"].setChecked(False)
    window.section_shown_checkboxes["echem"].setChecked(False)
    assert not _apply_tile_visible(window)
    window.section_shown_checkboxes["enmr"].setChecked(True)
    assert _apply_tile_visible(window)


def test_apply_tile_stays_hidden_after_a_restart(qapp):
    import json
    import os
    from opennmr.gui.main_window import MainWindow
    path = MainWindow._UI_SETTINGS_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"hidden_sections": ["enmr", "imaging", "echem"]}, handle)
    window = MainWindow()
    try:
        assert not _apply_tile_visible(window)
    finally:
        window.close()


def test_apply_tile_with_one_section_hidden_stays_after_a_restart(qapp):
    import json
    import os
    from opennmr.gui.main_window import MainWindow
    path = MainWindow._UI_SETTINGS_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"hidden_sections": ["imaging"]}, handle)
    window = MainWindow()
    try:
        assert _apply_tile_visible(window)
    finally:
        window.close()
