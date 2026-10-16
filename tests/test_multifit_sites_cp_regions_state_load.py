"""2026-10-16 round 2: every site's shift/width/... on one Multi-spectrum
graph; a state reopens each experiment with its saved load settings at the
first read; Apply ▸ CP Build-up's Drag on Plot shows the reference and adds
as many regions as wanted."""
import matplotlib.figure
import numpy as np
import pytest
from PySide6.QtWidgets import QMessageBox

from opennmr.gui.main_window import MainWindow
from opennmr.notebook import cp_buildup as cpb
from opennmr.notebook import multi_fitting as mf

from test_gui import qapp, _make_1d_dataset  # noqa: F401 -- qapp is a fixture
from test_multi_fit_quantities import series, _gui_fit  # noqa: F401 -- fixture
from test_compare_width_ranges_cp import _cp_spectrum, T


@pytest.fixture
def _dialogs(monkeypatch):
    shown = []
    for name in ("warning", "information", "critical"):
        monkeypatch.setattr(QMessageBox, name,
                            lambda *a, _n=name, **k: shown.append((_n, a[2] if len(a) > 2 else "")))
    return shown


# ---- Multi-spectrum: every site on one graph --------------------------------------

def test_notebook_plots_every_site_of_a_parameter_on_one_axes(series):
    assert mf.parameter_sites(series, "delta_iso") == ["A", "B"]
    assert "delta_iso" in mf.trend_parameter_names(series) and "fwhm" in mf.trend_parameter_names(series)
    ax = mf.plot_parameter_trends(series, "fwhm", ax=matplotlib.figure.Figure().add_subplot(111),
                                  xlabel="T (K)")
    labels = [c.get_label() for c in ax.containers]
    assert labels == ["A", "B"]
    a = ax.containers[0].lines[0]
    assert list(a.get_xdata()) == [280.0, 300.0, 320.0, 340.0]       # against x (the fit's own)
    assert list(a.get_ydata()) == pytest.approx([0.20, 0.16, 0.12, 0.08], rel=0.05)
    assert ax.get_xlabel() == "T (K)" and ax.get_ylabel() == "fwhm (ppm)"


def test_without_x_the_sites_are_plotted_in_fitting_order(series):
    ax = mf.plot_parameter_trends(series, "delta_iso", x_values=[None] * 4,
                                  ax=matplotlib.figure.Figure().add_subplot(111))
    assert [t.get_text() for t in ax.get_xticklabels()] == ["T0", "T1", "T2", "T3"]
    assert ax.get_xlabel() == "Experiment"


def test_gui_offers_all_sites_on_one_graph(qapp, _dialogs):
    window = MainWindow()
    panel = _gui_fit(window)
    entries = [panel.show_combo.itemText(i) for i in range(panel.show_combo.count())]
    assert any(e.startswith("Trend, all sites: ") and "FWHM" in e for e in entries), entries
    i = next(i for i, e in enumerate(entries) if e.startswith("Trend, all sites: ") and "FWHM" in e)
    panel.show_combo.setCurrentIndex(i)
    panel.show_selected()
    ax = window.fit_canvas.figure.axes[0]
    assert window.canvas_tabs.currentWidget() is window.fit_canvas
    assert sorted(c.get_label() for c in ax.containers) == ["A", "B"]
    assert _dialogs == []


# ---- Open openNMR State: the saved load settings at the first read -----------------

def test_open_state_reads_each_experiment_with_its_saved_load_settings(qapp, tmp_path, monkeypatch):
    window = MainWindow()
    path = _make_1d_dataset(tmp_path, n=64, name="one")
    window._load(path, vendor="bruker")
    folder = str(tmp_path / "work.opennmr")
    window._save_state(folder, "work", [0])
    from opennmr.io import state as io_state
    import json
    import os
    state_file = os.path.join(folder, io_state.STATE_FILE)
    info = json.load(open(state_file, encoding="utf-8"))
    info["entries"][0]["source"]["load_kwargs"] = {"procno": 1, "pseudo_2d": False}
    json.dump(info, open(state_file, "w", encoding="utf-8"))

    fresh = MainWindow()
    seen = []
    real = fresh._load
    monkeypatch.setattr(fresh, "_load", lambda p, v, preloaded=None, extra_kwargs=None:
                        (seen.append(extra_kwargs), real(p, v, preloaded, extra_kwargs))[1])
    opened, messages = fresh._open_state(folder)
    assert len(opened) == 1 and "reprocessed as saved" in messages[0]
    # The first (and only) read already asks for the saved proc number --
    # not the experiment's default, which for a 1D proc number of a 2D was
    # the whole 2D, read and thrown away.
    assert seen == [{"procno": 1, "pseudo_2d": False}]


# ---- CP Build-up: Drag on Plot, several regions -------------------------------------

def _cp_window():
    window = MainWindow()
    for k, t in enumerate(T):
        spectrum = _cp_spectrum(t)
        # a second line at 50 ppm, half the size, for a second region
        ppm = spectrum.axes[0].ppm
        amp = float(cpb.cp_is(np.array([t]), 50, 400, 4000)[0])
        spectrum.data = spectrum.data + amp / (1 + ((ppm - 50.0) / 1.0) ** 2)
        window._add_derived_spectrum(spectrum, f"cp{k}")
    window._show_side_tab("Apply")
    window._cp_buildup_section.setExpanded(True)
    panel = window.cp_buildup_panel
    panel.refresh()
    rows = [r for r, e in enumerate(panel._entries) if str(e["label"]).startswith("cp")]
    panel.set_checked(rows)
    panel.time_parameter_edit.setText("P15")
    assert panel.read_contact_times() == len(T)
    return window, panel, rows


def test_drag_on_plot_shows_the_reference_and_each_drag_adds_a_region(qapp, _dialogs):
    window, panel, rows = _cp_window()
    panel.set_reference(rows[-1])
    panel.drag_button.setChecked(True)
    # The reference is drawn on Compare, ready to be dragged on (it used to
    # be a blank Compare plot).
    assert window.canvas_tabs.currentWidget() is window.compare_canvas
    assert window.compare_canvas._region_select_mode
    lines = [l for ax in window.compare_canvas.figure.axes for l in ax.get_lines()
             if len(l.get_xdata()) > 100]
    assert lines, "the reference spectrum is on the plot"
    window.compare_canvas.regionSelected.emit(90.0, 110.0)
    window.compare_canvas.regionSelected.emit(45.0, 55.0)
    assert [(n, lo, hi) for n, lo, hi in panel.regions()] == [("Region 1", 90.0, 110.0),
                                                              ("Region 2", 45.0, 55.0)]
    # still armed for the next one, until switched off
    assert panel.armed and window.compare_canvas._region_select_mode
    panel.drag_button.setChecked(False)
    assert not window.compare_canvas._region_select_mode
    assert _dialogs == []


def test_several_regions_integrate_and_fit_separately(qapp, _dialogs):
    window, panel, _rows = _cp_window()
    panel.add_region(hi=110.0, lo=90.0, name="C=O")
    panel.add_region(hi=55.0, lo=45.0, name="CH2")
    assert panel.integrate()
    assert [s["name"] for s in panel.series_list] == ["C=O", "CH2"]
    ax = window.fit_canvas.figure.axes[0]
    assert {c.get_label() for c in ax.containers} == {"C=O", "CH2"}
    panel.model_combo.setCurrentIndex(panel.model_combo.findData("is"))
    assert panel.fit_all()
    assert panel.fits["C=O"]["params"]["T_IS"] == pytest.approx(150, rel=0.03)
    assert panel.fits["CH2"]["params"]["T_IS"] == pytest.approx(400, rel=0.03)
    panel.set_fit_region("CH2")
    assert "400" in panel.fit_results_label.text() or "T_IS" in panel.fit_results_label.text()
    kept = window.session.spectra[_rows[0]].metadata.raw["fits"]["cp_buildup"]
    assert "[C=O]" in kept["report"] and "[CH2]" in kept["report"]
    # one region fitted on its own, with a fixed T1ρ
    panel.set_parameter("T1rho_I", start=4000.0, fixed=True)
    assert panel.run_fit()
    assert panel.fits["CH2"]["params"]["T1rho_I"] == 4000.0


def test_regions_are_checked_and_changes_drop_stale_curves(qapp, _dialogs):
    _window, panel, _rows = _cp_window()
    assert panel.integrate() is False
    assert "Add at least one region" in _dialogs[-1][1]
    r = panel.add_region()
    assert panel.integrate() is False and "both limits" in _dialogs[-1][1]
    panel.region_table.item(r, 1).setText("110")
    panel.region_table.item(r, 2).setText("90")
    assert panel.integrate() and panel.series_list
    panel.add_region(hi=60.0, lo=40.0)
    assert panel.series_list == [] and "again" in panel.series_label.text()
    panel.region_table.selectRow(1)
    panel.remove_selected_regions()
    assert len(panel.regions()) == 1


def test_notebook_several_regions():
    spectra = [_cp_spectrum(t) for t in T]
    out = cpb.cp_series_regions(spectra, [(90, 110), (140, 160)], "P15", names=["peak", "empty"])
    assert [s["name"] for s in out] == ["peak", "empty"]
    assert np.max(out[0]["y"]) > 100 * np.max(np.abs(out[1]["y"]))
    ax = cpb.plot_cp_buildups(out, {"peak": cpb.fit_cp_buildup(out[0]["t_us"], out[0]["y"])},
                              ax=matplotlib.figure.Figure().add_subplot(111))
    assert len(ax.lines) >= 1 and {c.get_label() for c in ax.containers} == {"peak", "empty"}


# ---- Cmd/Ctrl+L: Width mode from the keyboard ---------------------------------------

def test_ctrl_l_toggles_width_mode_on_every_plot_and_stays_in_step(qapp):
    from PySide6.QtGui import QKeySequence
    window = MainWindow()
    action = window.width_menu_action
    assert action.shortcut() == QKeySequence("Ctrl+L")
    assert window.load_progress_action.shortcut() != action.shortcut()
    action.trigger()
    assert all(c._width_mode for c in window._measurement_canvases())
    # Measure takes over: Width lets go everywhere and the menu says so.
    window.measurement_menu_action.trigger()
    assert not any(c._width_mode for c in window._measurement_canvases())
    assert not action.isChecked()
    # The toolbar button on one plot drives the menu and the others too.
    window.canvas.set_width_mode(True)
    assert action.isChecked() and window.compare_canvas._width_mode
    assert not window.measurement_menu_action.isChecked()
    action.trigger()
    assert not any(c._width_mode for c in window._measurement_canvases())



def test_drag_on_plot_keeps_the_compare_plot_its_style_and_view(qapp, _dialogs):
    """Regions are shaded on the Compare plot as it is: no redraw, so the
    traces, styling and zoom stay (user report, 2026-10-16)."""
    window, panel, rows = _cp_window()
    window.spectrum_list.clearSelection()
    for i in rows[:2]:
        window.spectrum_list.item(i).setSelected(True)
    window._refresh_compare_view(switch_tab=True, force_render=True)
    canvas = window.compare_canvas
    axes_before = list(canvas.figure.axes)
    lines_before = [l for ax in axes_before for l in ax.get_lines()]
    canvas._compare_axes[0].set_xlim(130.0, 30.0)                  # the user's zoom
    panel.drag_button.setChecked(True)
    assert canvas._region_select_mode
    for lo, hi in ((90.0, 110.0), (45.0, 55.0), (60.0, 65.0)):
        canvas.regionSelected.emit(lo, hi)
    assert len(panel.regions()) == 3 and panel.armed and canvas._region_select_mode
    assert canvas.figure.axes == axes_before                        # nothing redrawn
    assert [l for ax in canvas.figure.axes for l in ax.get_lines()] == lines_before
    assert canvas._compare_axes[0].get_xlim() == pytest.approx((130.0, 30.0))
    shaded = [p for ax in canvas._compare_axes for p in ax.patches if p.get_gid() == "highlight:cp"]
    assert len(shaded) == 3 * len(canvas._compare_axes)
    # removing a region takes its shading away, still without a redraw
    panel.region_table.selectRow(2)
    panel.remove_selected_regions()
    shaded = [p for ax in canvas._compare_axes for p in ax.patches if p.get_gid() == "highlight:cp"]
    assert len(shaded) == 2 * len(canvas._compare_axes)
    assert canvas.figure.axes == axes_before
