"""2026-10-13 (explicit user requests):

* Fit's live preview (and the fit that follows it) keeps the user's zoom
  instead of zooming out on every update; Home still shows the whole fit.
* Result read-outs laid out to be read, not a wall of text
  (``gui/report_format.py``, ``widgets/report_label.py``); the plain report
  -- what is logged, copied and saved, and what ``text()`` returns -- is
  unchanged.
* Fit ▸ CSA, Quad & J-coupling's Parameters as one row per site, a column
  per parameter, with free/fixed and limits set by right-clicking.
"""

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMessageBox

from opennmr.gui import report_format as rf
from opennmr.gui.combined_panels import SPECTRUM_ROW
from opennmr.gui.main_window import MainWindow
from opennmr.gui.widgets.report_label import ReportLabel

from test_gui import qapp  # noqa: F401 -- fixture
from test_gui_combined_consolidation import _al_spectrum


@pytest.fixture(autouse=True)
def _dialogs(monkeypatch):
    shown = []
    for name in ("warning", "information", "critical"):
        monkeypatch.setattr(QMessageBox, name,
                            lambda *a, _n=name, **k: shown.append((_n, a[2] if len(a) > 2 else "")))
    return shown


# ---- report formatting (no Qt) -------------------------------------------------------------------

def test_report_html_lays_out_a_plain_report():
    text = ("Simulated in 0.59 s.\n"
            "Li_6 (7Li, I = 1.5): δiso 1320 ppm; CSA Δ 1300 ppm, η 0.78; amount 0.01\n"
            "R² 0.99  ·  RMSE 3.5  ·  0.2 s\n"
            "⚠ Intensity at the axis edge is 5% of the peak.\n"
            "a < b & c")
    out = rf.report_html(text, muted="#888888", warn="#aa6600")
    assert 'font-weight:600;">Simulated in 0.59 s.' in out          # a headline
    assert "<b>Li_6 (7Li, I = 1.5)</b><br>" in out                   # head over its items
    assert "·" in out and "δiso&nbsp;1320&nbsp;ppm" in out            # items kept whole
    assert 'color:#aa6600' in out and "Intensity at the axis edge" in out
    assert "a &lt; b &amp; c" in out                                 # escaped, not markup
    assert rf.report_html("") == ""


def test_rich_text_is_passed_through():
    rich = "<b>Done</b><br>3 peaks"
    assert rf.report_html(rich) == rich
    assert not rf.is_rich("R² < 0.9 and x > 2")


def _li_site(label, d, a):
    return {"label": label, "nucleus": "7Li", "delta_iso": d, "amount": a, "fwhm": 12,
            "lineshape": "pseudo_voigt", "eta": 0.5,
            "interactions": [{"type": "csa", "aniso": 1300, "asym": 0.78,
                              "convention": "Haeberlen", "euler": [0, 0, 0]},
                             {"type": "quadrupolar", "C_Q": 0.7, "eta_Q": 0.5}]}


def test_sites_table_says_what_every_site_shares_once():
    sites = [_li_site(f"Li_{i}", 1320 - 340 * i, 0.1 * (i + 1)) for i in range(7)]
    out = rf.sites_html(sites).replace("&nbsp;", " ")
    assert "All 7 sites:" in out
    assert out.count("Δ 1300 ppm") == 1                     # not once per site
    assert out.count("<tr>") == 8                           # header + 7 sites
    assert ">CSA<" not in out                               # shared: no column
    sites[3]["interactions"][0]["aniso"] = 900              # one differs: a column again
    out = rf.sites_html(sites).replace("&nbsp;", " ")
    assert ">CSA<" in out and "Δ 900 ppm" in out and "All 7 sites:" in out


def test_long_site_lists_are_summarised():
    sites = [_li_site(f"L{i}", i, 1.0) for i in range(30)]
    out = rf.sites_html(sites, max_rows=10)
    assert out.count("<tr>") == 11 and "… and 20 more sites" in out


def test_combined_fit_html_puts_quality_first():
    out = rf.combined_fit_html(
        ["CSA, Quad & J-coupling fit: MAS 50 kHz, nu0 = 200.482 MHz, window 1200 to -1200 ppm."],
        {"r2": 0.99837, "rmse": 5.42e7, "reduced_chi2": 2.94e15}, 16.0,
        warnings=["Site 1 is at its limit."], has_quadrupolar=False)
    assert out.index("0.99837") < out.index("MAS")
    assert "|C_Q|" not in out and "Site 1 is at its limit." in out


# ---- the label ------------------------------------------------------------------------------------

def test_report_label_keeps_the_plain_report(qapp):
    label = ReportLabel()
    label.setText("Fit done.\nA: x 1; y 2")
    assert label.text() == "Fit done.\nA: x 1; y 2"
    assert "<b>A</b>" in label.html()
    label.setReport("plain", "<b>shown</b>")
    assert label.text() == "plain" and label.html() == "<b>shown</b>"
    label.setText("")
    assert label.html() == ""


def test_simulate_read_out_is_a_table(qapp):
    window = MainWindow()
    panel = window.combined_sim_panel
    panel.nucleus_combo.setCurrentText("7Li")
    panel.regime_combo.setCurrentIndex(panel.regime_combo.findData("mas"))
    panel.mas_spin.setValue(50000)
    panel.orientations_spin.setValue(233)
    sites = [_li_site(f"Li_{i}", 300 * i, 0.2) for i in range(3)]
    for s in sites:
        s.pop("nucleus")
    panel.site_list.clear()
    panel.site_dicts = []
    panel.add_sites(sites)
    assert panel.simulate()
    assert panel.results_label.text().startswith("Simulated in")     # the plain log, as before
    shown = panel.results_label.html()
    assert "3 sites simulated in" in shown and "All 3 sites:" in shown and "<table" in shown
    assert "MAS&nbsp;50&nbsp;kHz" in shown


# ---- the live preview keeps the zoom --------------------------------------------------------------

def _fit_panel(window):
    _al_spectrum(window)
    panel = window.combined_fit_panel
    panel.seed_from_spectrum(window.session.spectra[0])
    panel.transitions_combo.setCurrentIndex(panel.transitions_combo.findData("ct"))
    panel.add_interaction("quadrupolar", C_Q=4.0, eta_Q=0.3)
    panel.diso_spin.setValue(10.0)
    return panel


def _zoom(canvas, xlim, ylim=None):
    canvas.note_natural_view()            # what a scroll/drag/toolbar zoom records first
    canvas.ax.set_xlim(*xlim)
    if ylim is not None:
        canvas.ax.set_ylim(*ylim)


def test_preview_updates_keep_the_users_zoom_and_home_resets(qapp):
    window = MainWindow()
    panel = _fit_panel(window)
    canvas = window.fit_canvas
    assert panel.preview_now()
    natural = canvas.ax.get_xlim()
    _zoom(canvas, (20.0, 0.0), (0.0, 500.0))
    panel.diso_spin.setValue(8.0)
    assert panel.preview_now()
    assert canvas.ax.get_xlim() == pytest.approx((20.0, 0.0))
    assert canvas.ax.get_ylim() == pytest.approx((0.0, 500.0))
    assert canvas.ax.get_title() == "Starting model (preview)"   # it did redraw
    canvas.toolbar.home()
    assert canvas.ax.get_xlim() == pytest.approx(natural)
    # Not zoomed: a preview is drawn at its own limits.
    panel.diso_spin.setValue(6.0)
    assert panel.preview_now()
    assert canvas.ax.get_xlim() == pytest.approx(natural)


def test_the_fit_after_a_preview_keeps_the_zoom_too(qapp):
    window = MainWindow()
    panel = _fit_panel(window)
    panel.orientations_spin.setValue(377)
    assert panel.preview_now()
    _zoom(window.fit_canvas, (25.0, -5.0))
    assert panel.fit()
    assert window.fit_canvas.ax.get_xlim() == pytest.approx((25.0, -5.0))


def test_a_zoom_outside_the_new_data_is_not_kept(qapp):
    window = MainWindow()
    canvas = window.fit_canvas
    x = np.linspace(-10, 10, 200)
    canvas.show_slice_fit(x, np.exp(-x ** 2), np.exp(-x ** 2), keep_zoom=True)
    _zoom(canvas, (5.0, 2.0))
    x2 = np.linspace(100, 120, 200)
    canvas.show_slice_fit(x2, np.exp(-(x2 - 110) ** 2), np.exp(-(x2 - 110) ** 2),
                          keep_zoom=True)
    assert sorted(canvas.ax.get_xlim()) == pytest.approx([100.0, 120.0])


def test_other_redraws_still_reset_the_view(qapp):
    window = MainWindow()
    canvas = window.fit_canvas
    x = np.linspace(-10, 10, 200)
    canvas.show_slice_fit(x, np.exp(-x ** 2), np.exp(-x ** 2))
    _zoom(canvas, (5.0, 2.0))
    canvas.show_slice_fit(x, np.exp(-x ** 2), np.exp(-x ** 2))      # no keep_zoom
    assert sorted(canvas.ax.get_xlim()) == pytest.approx([-10.0, 10.0])


def test_home_after_a_component_style_change_goes_to_the_whole_fit(qapp):
    # The same natural-view record now covers the style-change redraw.
    window = MainWindow()
    canvas = window.fit_canvas
    x = np.linspace(-10, 10, 200)
    canvas.show_slice_fit(x, np.exp(-x ** 2), np.exp(-x ** 2),
                          components=[np.exp(-x ** 2), np.exp(-x ** 2) * 0.5, 0 * x])
    _zoom(canvas, (5.0, 2.0))
    canvas.set_fit_component_style(fill=False)
    assert canvas.ax.get_xlim() == pytest.approx((5.0, 2.0))
    canvas.toolbar.home()
    assert sorted(canvas.ax.get_xlim()) == pytest.approx([-10.0, 10.0])


# ---- Parameters: one row per site ----------------------------------------------------------------

def _three_sites(window):
    panel = _fit_panel(window)
    panel.add_site()
    panel.add_site()
    return panel


def test_one_row_per_site_lists_every_parameter_in_columns(qapp):
    window = MainWindow()
    panel = _three_sites(window)
    panel.set_parameter_layout("sites")
    t = panel.site_parameter_table
    assert panel.parameter_stack.currentWidget() is t
    headers = [t.horizontalHeaderItem(j).text() for j in range(t.columnCount())]
    assert headers[0] == "Site" and "δiso (ppm)" in headers and "C_Q (MHz)" in headers
    assert t.rowCount() == 3
    assert [t.item(r, 0).text() for r in range(3)] == ["Site 1", "Site 2", "Site 3"]
    cq = headers.index("C_Q (MHz)")
    assert float(t.item(0, cq).text()) == pytest.approx(4.0)
    assert not (t.item(1, cq).flags() & Qt.ItemFlag.ItemIsEnabled)  # Site 2 has no C_Q
    # Typing a start value changes the site.
    diso = headers.index("δiso (ppm)")
    t.item(1, diso).setText("-12.5")
    assert panel.site_dicts[1]["delta_iso"] == pytest.approx(-12.5)


def test_right_click_constraints_reach_the_fit_and_the_other_layout(qapp, _dialogs):
    window = MainWindow()
    panel = _three_sites(window)
    panel.set_parameter_layout("sites")
    panel.select_parameter_column("fwhm")
    cells = panel.selected_parameter_cells()
    assert cells == [("Site 1", "fwhm"), ("Site 2", "fwhm"), ("Site 3", "fwhm")]
    panel.set_cell_constraints(cells, free=False)
    panel.set_cell_constraints([("Site 2", "delta_iso")], minimum="-5", maximum="5",
                               tries="-2, 2")
    args = panel.fit_arguments()
    assert all("fwhm" not in args["vary"][s] for s in ("Site 1", "Site 2", "Site 3"))
    assert args["bounds"]["Site 2"]["delta_iso"] == (-5.0, 5.0)
    assert args["starts"]["Site 2"]["delta_iso"][1:] == [-2.0, 2.0]
    t = panel.site_parameter_table
    headers = [t.horizontalHeaderItem(j).text() for j in range(t.columnCount())]
    fwhm_cell = t.item(0, headers.index("FWHM (ppm)"))
    assert fwhm_cell.font().italic() and "Fixed" in fwhm_cell.toolTip()
    assert "Min: -5" in t.item(1, headers.index("δiso (ppm)")).toolTip()
    with pytest.raises(ValueError, match="Min must be a number"):
        panel.set_cell_constraints(cells, minimum="abc")
    # The row-per-parameter layout shows the same choices.
    panel.set_parameter_layout("parameters")
    p = panel.parameter_table
    rows = {(p.item(r, 0).text(), p.item(r, 1).data(Qt.ItemDataRole.UserRole)): r
            for r in range(p.rowCount())}
    r = rows[("Site 2", "delta_iso")]
    assert (p.item(r, 4).text(), p.item(r, 5).text(), p.item(r, 6).text()) == ("-5", "5", "-2, 2")
    assert p.item(rows[("Site 1", "fwhm")], 3).checkState() == Qt.CheckState.Unchecked


def test_the_dialog_sets_limits_for_a_selection(qapp, monkeypatch):
    window = MainWindow()
    panel = _three_sites(window)
    panel.set_parameter_layout("sites")
    monkeypatch.setattr(panel, "_ask_cell_limits",
                        lambda cells: {"minimum": "0.1", "maximum": "9", "tries": ""})
    values = panel._ask_cell_limits([("Site 1", "amount")])
    panel.set_cell_constraints([("Site 1", "amount"), ("Site 3", "amount")], **values)
    assert panel.constraints["Site 3"]["amount"]["max"] == "9"


def test_mas_rate_has_its_own_row_and_the_layout_is_saved(qapp):
    window = MainWindow()
    panel = _three_sites(window)
    panel.regime_combo.setCurrentIndex(panel.regime_combo.findData("mas"))
    panel.mas_spin.setValue(12000)
    panel.set_parameter_layout("sites")
    t = panel.site_parameter_table
    assert t.item(t.rowCount() - 1, 0).text() == SPECTRUM_ROW
    panel.set_cell_constraints([(SPECTRUM_ROW, "mas_Hz")], free=True)
    assert panel.mas_fit_arguments()["fit_mas"] is True
    state = panel.get_state()
    assert "site_parameter_table" not in state["widgets"]
    panel.set_parameter_layout("parameters")
    panel.set_state(state)
    assert panel.parameter_layout() == "sites"
    assert panel.parameter_stack.currentWidget() is panel.site_parameter_table
