"""2026-09-29 (second round): Curves & profiles spines where they are
looked for (Settings, right-click), fit-component display options
(notebook.fit_display + Fit ▸ Fit plot), and Czjzek fitting/simulation in
the GUI. Synthetic data only."""
import json

import numpy as np
import pytest
from matplotlib.collections import PolyCollection

from opennmr.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
from opennmr.gui.widgets.canvas import MplCanvas
from opennmr.notebook import czjzek as cz
from opennmr.notebook import fit_display


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(qapp, tmp_path, monkeypatch):
    from opennmr.gui.main_window import MainWindow
    monkeypatch.setattr(MainWindow, "_FIGURE_STYLES_PATH", str(tmp_path / "figure_styles.json"))
    return MainWindow()


def _relax():
    t = np.geomspace(0.01, 1.0, 8)
    y = 1 - 2 * np.exp(-t / 0.2)
    return {"t": t, "y": y, "y_fit": y, "peak_label": "Peak 1", "model": "inversion_recovery"}


def _capture_menu(monkeypatch):
    # Through the canvas's own seam: patching QMenu.exec opens the real
    # modal menu anyway under PySide6 (see MplCanvas._exec_context_menu).
    menus = []
    monkeypatch.setattr(MplCanvas, "_exec_context_menu",
                        lambda self, menu, pos: menus.append(menu) or None)
    return menus


def _actions(menu):
    out = {}
    for action in menu.actions():
        out[action.text()] = action
        if action.menu() is not None:
            for sub in action.menu().actions():
                out[sub.text()] = sub
    return out


# ----------------------------------------------------------------------
# Curves & profiles spines
# ----------------------------------------------------------------------

def test_settings_curve_spines_save_and_apply(window):
    window.fit_canvas.show_relaxation_fit(_relax())
    assert not window.fit_canvas.ax.spines["top"].get_visible()
    window.settings_curve_top_spine_checkbox.setChecked(True)
    assert window.fit_canvas.ax.spines["top"].get_visible()
    saved = json.load(open(window._FIGURE_STYLES_PATH))
    assert saved["curve"] == {"top_spine": True, "right_spine": False}
    # Clearing the Curves & profiles default unticks it again.
    window._on_clear_figure_style_template("curve")
    assert not window.settings_curve_top_spine_checkbox.isChecked()


def test_a_spines_only_curve_default_still_takes_the_1d_default(window):
    window._save_figure_style("1d", {"linewidth": 2.5})
    window.settings_curve_right_spine_checkbox.setChecked(True)
    style = window._resolved_type_style("curve")
    assert style["linewidth"] == 2.5 and style["right_spine"] is True
    canvas = MplCanvas()
    window._on_canvas_plot_kind_changed(canvas, "curve")
    canvas.show_relaxation_fit(_relax())
    assert canvas.ax.spines["right"].get_visible()
    assert canvas.style_settings()["linewidth"] == 2.5


def test_right_click_toggles_curve_spines(window, monkeypatch):
    menus = _capture_menu(monkeypatch)
    canvas = window.fit_canvas
    canvas.show_relaxation_fit(_relax())
    canvas._on_context_menu(canvas.rect().center())
    actions = _actions(menus[-1])
    assert "Top Spine" in actions and "Right Spine" in actions
    actions["Right Spine"].trigger()
    assert canvas.ax.spines["right"].get_visible()
    # Not offered on a spectrum.
    ppm = np.linspace(10, 0, 200)
    canvas.show_spectrum(NMRSpectrum(np.exp(-(ppm - 5) ** 2), [NMRAxis("frequency", ppm=ppm, hz=ppm * 400)],
                                     NMRMetadata(vendor="bruker", nucleus="1H", sfo1=400.0)))
    canvas._on_context_menu(canvas.rect().center())
    assert "Top Spine" not in _actions(menus[-1])


# ----------------------------------------------------------------------
# Component display: notebook function
# ----------------------------------------------------------------------

def _curves():
    x = np.linspace(0, 10, 400)
    a = np.exp(-(x - 4) ** 2)
    b = 0.5 * np.exp(-(x - 6) ** 2)
    base = 0.05 + 0 * x
    return x, a + b + base, [a, b, base]


def _draw(style=None, **kw):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x, total, comps = _curves()
    fig, ax = plt.subplots()
    out = fit_display.draw_fit_components(ax, x, total, total, comps, ["A", "B"], style=style, **kw)
    return ax, out


def test_components_shown_filled_by_default_in_rank_colours():
    ax, out = _draw()
    assert set(out["components"]) == {"A", "B"}
    assert all(isinstance(a, PolyCollection) for a in out["components"].values())
    defaults = fit_display.default_component_colours(_curves()[2][:2], ["A", "B"])
    from opennmr.plotting import qualitative_colors
    assert defaults["A"] == tuple(qualitative_colors(2)[0][:3])   # tallest first


def test_none_highlight_and_line_only_modes():
    _ax, out = _draw({"show": "none"})
    assert out["components"] == {} and out["baseline"] is None
    _ax, out = _draw({"show": "highlight", "highlight": ["B"]})
    assert list(out["components"]) == ["B"] and list(out["dimmed"]) == ["A"]
    assert out["dimmed"]["A"].get_label() == "_nolegend_"
    _ax, out = _draw({"fill": False, "line_colours": {"A": "#123456"}})
    line = out["components"]["A"]
    assert not isinstance(line, PolyCollection)
    assert line.get_color() in ("#123456", (0x12 / 255, 0x34 / 255, 0x56 / 255, 1.0))


def test_fill_colour_alpha_and_on_baseline():
    x, total, comps = _curves()
    _ax, out = _draw({"fill_colours": {"B": "green"}, "fill_alpha": 0.25, "on_baseline": True})
    face = out["components"]["B"].get_facecolor()[0]
    np.testing.assert_allclose(face, (0, 0.50196, 0, 0.25), atol=1e-3)
    ys = out["components"]["B"].get_paths()[0].vertices[:, 1]
    assert ys.min() == pytest.approx(0.05, abs=1e-6)      # sits on the baseline


def test_single_component_auto_hidden_only_when_asked():
    import matplotlib.pyplot as plt
    x, _t, comps = _curves()
    fig, ax = plt.subplots()
    out = fit_display.draw_fit_components(ax, x, comps[0], comps[0], [comps[0], comps[2]], ["A"],
                                          auto_hide_single=True)
    assert out["components"] == {}
    out = fit_display.draw_fit_components(ax, x, comps[0], comps[0], [comps[0], comps[2]], ["A"],
                                          style={"show": "all"}, auto_hide_single=True)
    assert list(out["components"]) == ["A"]


def test_style_is_checked():
    with pytest.raises(KeyError):
        fit_display.component_style(colour="red")
    with pytest.raises(ValueError):
        fit_display.component_style(show="some")


# ----------------------------------------------------------------------
# Component display: GUI
# ----------------------------------------------------------------------

def test_fit_plot_panel_styles_both_fit_views_and_keeps_zoom(window):
    x, total, comps = _curves()
    canvas = window.fit_canvas
    canvas.show_slice_fit(x, total, total, components=comps, peak_labels=["A", "B"])
    panel = window.fit_display_panel
    assert panel._labels == ["A", "B"] and panel.table.rowCount() == 2
    canvas.ax.set_xlim(6, 3)
    panel.fill_check.setChecked(False)
    assert canvas.ax.get_xlim() == (6, 3)
    assert not [c for c in canvas.ax.collections if isinstance(c, PolyCollection)]
    assert window.spectral_fit_canvas.fit_component_style()["fill"] is False
    panel.set_component_colour("line_colours", "B", "#ff00ff")
    line = [l for l in canvas.ax.lines if l.get_label() == "B"][0]
    assert line.get_color() in ("#ff00ff", (1.0, 0.0, 1.0, 1.0))
    panel.table.cellWidget(0, 0).setChecked(True)        # tick A -> highlight mode
    assert canvas.fit_component_style()["show"] == "highlight"
    assert panel.show_combo.currentData() == "highlight"
    assert [l.get_label() for l in canvas.ax.lines].count("B") == 0


def test_right_click_components_menu(window, monkeypatch):
    menus = _capture_menu(monkeypatch)
    x, total, comps = _curves()
    canvas = window.fit_canvas
    canvas.show_slice_fit(x, total, total, components=comps, peak_labels=["A", "B"])
    canvas._on_context_menu(canvas.rect().center())
    actions = _actions(menus[-1])
    actions["Hide all"].trigger()
    assert canvas.fit_component_style()["show"] == "none"
    assert window.fit_display_panel.show_combo.currentData() == "none"
    actions["Colours and highlighting…"].trigger()
    assert window._fit_display_dialog.isVisible()       # a right-click window now (2026-10-16)
    window._fit_display_dialog.hide()
    actions["Fit Plot: Components and Colours…"].trigger()
    assert window._fit_display_dialog.isVisible()


def test_quadrupolar_single_site_now_shows_with_all(window):
    x, total, comps = _curves()
    result = {"x": x, "y": total, "y_fit": total, "components": [comps[0], comps[2]],
              "regions": None, "info": {"n_sites": 1, "labels": ["Site 1"]}}
    window._draw_quadrupolar_fit(result)
    assert not [c for c in window.fit_canvas.ax.collections if isinstance(c, PolyCollection)]
    window.fit_display_panel.show_combo.setCurrentIndex(
        window.fit_display_panel.show_combo.findData("all"))
    assert [c for c in window.fit_canvas.ax.collections if isinstance(c, PolyCollection)]


def test_notebook_quad_plot_takes_a_component_style():
    import matplotlib
    matplotlib.use("Agg")
    from opennmr.notebook import quadrupolar_fitting as qfit
    from opennmr.notebook.simulations import quadrupolar_spectrum
    x = np.linspace(100, -100, 1024)
    y = 0
    for d, cq in ((40, 3e6), (-10, 2e6)):
        y = y + quadrupolar_spectrum(2.5, cq, 0.3, 130.3e6, x, order="second", transitions="ct",
                                     delta_iso=d, fwhm=2, n_zcw=987)["normalised_real_spec"]
    res = {"x_ppm": x, "real_spectrum": y, "SFO1": 130.3, "nuclide": "27Al"}
    popt, _pc, _f, _s, info = qfit.fit_quadrupolar_spectrum(
        res, [{"d_iso": 40, "C_Q": 3, "eta_Q": 0.3, "fwhm": 2, "label": "A"},
              {"d_iso": -10, "C_Q": 2, "eta_Q": 0.3, "fwhm": 2, "label": "B"}],
        n_zcw=987, fixed=["eta_Q", "C_Q", "d_iso", "fwhm"])
    fig, ax = qfit.NMR1dplot_with_quad_fit(res, popt, info, show_residual=False, n_zcw_final=987,
                                           component_style={"show": "highlight", "highlight": ["A"]})
    labels = [c.get_label() for c in ax.collections]
    assert "A" in labels and "B" not in labels


# ----------------------------------------------------------------------
# Czjzek in the GUI
# ----------------------------------------------------------------------

def _glass(window):
    nu0 = 130.3e6
    x = np.linspace(200, -200, 2048)
    a = cz.czjzek_spectrum(2.5, 3.0, nu0, x, delta_iso=60, fwhm=4)["real_spec"]
    b = cz.czjzek_spectrum(2.5, 1.5, nu0, x, delta_iso=10, fwhm=6)["real_spec"]
    y = 1000 * a / a.max() + 400 * b / b.max() + np.random.default_rng(3).normal(0, 3, x.size)
    spectrum = NMRSpectrum(y, [NMRAxis("frequency", ppm=x, hz=x * 130.3)],
                           NMRMetadata(vendor="bruker", nucleus="27Al", sfo1=130.3))
    index = window._add_derived_spectrum(spectrum, "glass")
    return index


def test_czjzek_sections_are_where_expected(window):
    from opennmr.gui.widgets.collapsible_section import CollapsibleSection
    pages = {window.side_tabs.tabText(i): window.side_tabs.widget(i)
             for i in range(window.side_tabs.count())}
    fit_titles = [s.title() for s in pages["Fit"].findChildren(CollapsibleSection)]
    # 2026-10-06: right after Solid-state NMR, which replaced the
    # single-interaction sections (Quadrupolar among them).
    assert fit_titles.index("Quadrupolar (Czjzek distribution)") == \
        fit_titles.index("Solid-state NMR") + 1
    assert "Quadrupolar" not in fit_titles
    # The Fit plot options are a right-click window since 2026-10-16.
    assert "Fit plot: components and colours" not in fit_titles
    sim_titles = [s.title() for s in pages["Simulate"].findChildren(CollapsibleSection)]
    assert sim_titles.index("Czjzek distribution") == sim_titles.index("Solid-state NMR") + 1


def test_czjzek_fit_panel_fits_reports_maps_and_saves(window):
    index = _glass(window)
    panel = window.czjzek_fit_panel
    assert panel.nucleus_combo.currentText() == "27Al"
    assert panel.larmor_spin.value() == pytest.approx(130.3)
    # d starts fixed (italic), as the model says.
    _bounds, fixed = panel._constraints(0)
    assert fixed == {"d": True}
    panel.sites_table.item(0, 1).setText("50")
    panel.sites_table.item(0, 2).setText("2")
    panel.add_site(("AlO6", "15", "1", "5", "3", "0.5"))
    panel.grid_cq_spin.setValue(48)
    panel.fit()
    r = panel.result
    assert r["report"][0]["sigma"] == pytest.approx(3.0, rel=1e-2)
    assert r["report"][1]["d_iso"] == pytest.approx(10.0, abs=0.2)
    assert "Czjzek fit" in panel.results_label.text()
    headers = [panel.results_table.horizontalHeaderItem(c).text().replace("\n", " ")
               for c in range(panel.results_table.columnCount())]
    assert "⟨C_Q⟩ (MHz)" in headers and "⟨η_Q⟩" in headers
    # Drawn as a fit with both sites.
    labels = [c.get_label() for c in window.fit_canvas.ax.collections]
    assert "Site 1" in labels and "AlO6" in labels
    # The distribution map, spectrum above.
    panel.map_layout_combo.setCurrentIndex(panel.map_layout_combo.findData("top"))
    panel.map_site_combo.setCurrentIndex(panel.map_site_combo.findData(1))
    panel.show_map()
    axes = window.fit_canvas.figure.czjzek_axes
    assert axes["spectrum"].get_position().y0 > axes["map"].get_position().y1
    # Saved with the spectrum, and brought back on reselecting it.
    fits = window.session.spectra[index].metadata.raw["fits"]
    assert fits["czjzek"]["label"] == "Czjzek"
    panel.clear_result()
    window._restore_fit_panels(index)
    assert panel.results_table.rowCount() == 2
    assert panel.show_map_button.isEnabled()


def test_czjzek_send_and_simulate(window):
    sim = window.czjzek_sim_panel
    sim.set_from_fit({"nucleus": "27Al", "sigma": 2.5, "d": 5.0, "delta_iso": 30.0, "fwhm": 3.0,
                      "larmor_MHz": 130.3, "mas_Hz": 0.0, "transitions": "ct", "order": "second",
                      "ppm_range": (-150.0, 100.0)})
    sim.grid_cq_spin.setValue(32)
    # The spectrum alone is the default now; ask for the map.
    sim.layout_combo.setCurrentIndex(sim.layout_combo.findData("right"))
    result = sim.simulate()
    assert result["sigma"] == 2.5
    assert "⟨C_Q⟩" in sim.results_label.text()
    assert window.simulation_canvas.figure.czjzek_axes["map"] is not None
    n = len(window.session)
    sim.export()
    assert len(window.session) == n + 1
    assert window.session.spectra[-1].metadata.raw["simulation"]["model"] == "Czjzek"
    # Journalled like the other simulations.
    assert "czjzek" in window._simulation_panels()
