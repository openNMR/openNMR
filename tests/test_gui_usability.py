"""Usability round, 2026-09-27 (explicit user requests):

* every collapsible box starts collapsed;
* the toolbar reads Open, Process, Compare, Fit, Simulate, Dynamics,
  Adv. Proc., Apply, Settings (and Ctrl+1..8 follow it);
* the default plot style reaches relaxation/diffusion curves, parameter
  trends and the DNP profiles ("Curves & profiles", falling back to the
  1D default).

The Title / notes drawer's tests are in test_experiment_notes.py.
"""
import json

import numpy as np
import pytest
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import QApplication

from opennmr.gui.main_window import MainWindow
from opennmr.gui.widgets.collapsible_section import CollapsibleSection


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


TOOLBAR = ["Open", "Process", "Compare", "Fit", "Simulate", "Dynamics", "Adv. Proc.", "Apply",
           "Settings"]


def test_toolbar_order(qapp):
    window = MainWindow()
    assert [window.side_tabs.tabText(i) for i in range(window.side_tabs.count())] == TOOLBAR
    # The stored indices follow the new order.
    for attr, name in (("_processing_tab_index", "Process"), ("_compare_tab_index", "Compare"),
                       ("_fitting_tab_index", "Fit"), ("_simulations_tab_index", "Simulate"),
                       ("_settings_tab_index", "Settings")):
        assert window.side_tabs.tabText(getattr(window, attr)) == name


def test_ctrl_number_opens_the_nth_toolbar_button(qapp):
    window = MainWindow()
    for n, (label, tile) in enumerate(MainWindow._WINDOW_MENU_PANELS, start=1):
        assert TOOLBAR[n - 1] == tile
        action = next(a for a in window.findChildren(QAction) if a.text() == label)
        assert action.shortcut() == QKeySequence(f"Ctrl+{n}")


def test_every_panel_opens_with_its_sections_collapsed(qapp):
    window = MainWindow()
    sections = window.findChildren(CollapsibleSection)
    assert len(sections) > 20
    for index in range(window.side_tabs.count()):
        window.side_tabs.setCurrentIndex(index)
        assert [s.title() for s in sections if s.isExpanded()] == []


def test_opening_a_section_is_kept_across_panel_switches(qapp):
    window = MainWindow()
    window._show_side_tab("Compare")
    window._dnp_section.setExpanded(True)
    window._show_side_tab("Fit")
    window._show_side_tab("Compare")
    assert window._dnp_section.isExpanded()


def test_code_that_sends_the_user_to_a_section_still_opens_it(qapp, monkeypatch):
    window = MainWindow()
    # The simulation itself is not what is being tested (and is slow).
    monkeypatch.setattr(window.combined_sim_panel, "simulate", lambda *a, **k: None)
    assert not window._combined_sim_section.isExpanded()
    window._send_combined_to_simulation(
        {"larmor_Hz": 105.9e6, "mas_Hz": 10000.0, "order": "second", "transitions": "all"},
        [], "23Na")
    assert window._combined_sim_section.isExpanded()


# -- default plot style on curves ------------------------------------------

def _save_styles(tmp_path, monkeypatch, styles):
    path = tmp_path / "figure_styles.json"
    path.write_text(json.dumps(styles))
    monkeypatch.setattr(MainWindow, "_FIGURE_STYLES_PATH", str(path))


def _relaxation_result():
    t = np.array([0.01, 0.1, 0.5, 1.0, 2.0, 5.0])
    y = 1 - np.exp(-t / 1.0)
    return {"t": t, "y": y, "y_fit": y, "model": "saturation_recovery", "peak_label": "peak 1"}


def test_curves_are_their_own_plot_kind_in_settings(qapp):
    assert ("Curves & profiles", "curve") in MainWindow._FIGURE_STYLE_TYPES
    window = MainWindow()
    assert "curve" in window.settings_type_save_buttons


def test_relaxation_curve_takes_the_1d_default_when_curves_have_none(qapp, tmp_path, monkeypatch):
    _save_styles(tmp_path, monkeypatch, {"1d": {"labelsize": 21, "tick_length": 9.0,
                                                "trace_color": "#1f77b4"}})
    window = MainWindow()
    window.fit_canvas.show_relaxation_fit(_relaxation_result())
    canvas = window.fit_canvas
    assert canvas.plot_kind() == "curve"
    assert canvas.ax.xaxis.label.get_fontsize() == 21
    assert canvas.ax.xaxis.get_major_ticks()[0].tick1line.get_markersize() == 9.0
    data = [line for line in canvas.ax.lines if line.get_marker() == "o"][0]
    assert data.get_color() == "#1f77b4"


def test_a_saved_curve_default_wins_over_the_1d_one(qapp, tmp_path, monkeypatch):
    _save_styles(tmp_path, monkeypatch, {"1d": {"labelsize": 21}, "curve": {"labelsize": 15}})
    window = MainWindow()
    window.fit_canvas.show_relaxation_fit(_relaxation_result())
    assert window.fit_canvas.ax.xaxis.label.get_fontsize() == 15


def test_parameter_trend_is_a_curve_too(qapp, tmp_path, monkeypatch):
    _save_styles(tmp_path, monkeypatch, {"curve": {"labelsize": 17}})
    window = MainWindow()
    window.fit_canvas.show_parameter_trend([1.0, 2.0, 3.0], [1.0, 4.0, 9.0], xlabel="T (K)",
                                           ylabel="x")
    assert window.fit_canvas.plot_kind() == "curve"
    assert window.fit_canvas.ax.xaxis.label.get_fontsize() == 17


def test_dnp_profile_follows_the_curve_style(qapp, tmp_path, monkeypatch):
    from test_dnp import ERRORS, FIELDS, _sweep
    _save_styles(tmp_path, monkeypatch, {"curve": {"labelsize": 19, "tick_width": 2.5,
                                                   "linewidth": 2.0, "trace_color": "#d62728"}})
    window = MainWindow()
    # A real window size: in a tiny one the canvas rightly shrinks labels
    # that would otherwise be cut off (fit_labels_within_figure).
    window.resize(1700, 1100)
    window.show()
    qapp.processEvents()
    for k, s in enumerate(_sweep(FIELDS, ERRORS)):
        window._add_derived_spectrum(s, f"s{k}")
    window.spectrum_list.selectAll()
    panel = window.dnp_panel
    panel.sweep_peak, panel.sweep_region = 0.0, (-3000.0, 3000.0)
    panel.sweep_quantity_combo.setCurrentIndex(panel.sweep_quantity_combo.findData("both"))
    panel.run_sweep()
    canvas = window.fit_canvas
    assert canvas.plot_kind() == "curve"
    for ax in canvas.figure.axes:
        assert ax.yaxis.label.get_fontsize() == 19
        assert ax.spines["left"].get_linewidth() == 2.0
        assert ax.yaxis.get_major_ticks()[0].tick1line.get_markeredgewidth() == 2.5
    points = [line for line in canvas.figure.axes[0].lines if line.get_marker() == "o"]
    assert points and all(line.get_color() == "#d62728" for line in points)


def test_dnp_sweep_spectra_use_the_compare_palette(qapp, tmp_path, monkeypatch):
    from matplotlib.colors import to_hex

    from opennmr.plotting.style import stack_colors
    from test_dnp import ERRORS, FIELDS, _sweep
    _save_styles(tmp_path, monkeypatch, {"compare": {"compare_colormap": "plasma"}})
    window = MainWindow()
    # Carriers exactly on the line: before tracking, spectra are coloured
    # by their SFO1 field, which carrier errors would reorder.
    for k, s in enumerate(_sweep(FIELDS, [0.0] * len(FIELDS))):
        window._add_derived_spectrum(s, f"s{k}")
    window.spectrum_list.selectAll()
    window.dnp_panel.show_sweep_spectra()
    lines = [line for line in window.compare_canvas.figure.axes[0].lines
             if len(line.get_xdata()) > 100]
    # FIELDS ascend, so the spectra take the palette in order.
    assert [to_hex(line.get_color()) for line in lines] == [
        to_hex(c) for c in stack_colors(len(FIELDS), "plasma")]
