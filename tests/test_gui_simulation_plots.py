"""2026-10-01: Simulate's plots on a full plot canvas (Plot Style,
right-click, scroll zoom, Measure, per-series names and colours), tick
direction in Plot Style, the Measure fixes (second click on a spectrum
with no Hz axis), and the Czjzek citations / measured time estimate."""
import types
import warnings

import matplotlib
import numpy as np
import pytest
from PySide6.QtWidgets import QMessageBox

from opennmr.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
from opennmr.gui.main_window import MainWindow
from opennmr.gui.widgets.canvas import MplCanvas
from opennmr.notebook import czjzek as cz

from test_gui import qapp  # noqa: F401 -- fixture


@pytest.fixture(autouse=True)
def _dialogs(monkeypatch, tmp_path):
    shown = []
    for name in ("warning", "information", "critical"):
        monkeypatch.setattr(QMessageBox, name,
                            lambda *a, _n=name, **k: shown.append((_n, a[2] if len(a) > 2 else "")))
    monkeypatch.setattr(MainWindow, "_FIGURE_STYLES_PATH", str(tmp_path / "figure_styles.json"))
    return shown


def _menu(canvas, monkeypatch):
    menus = []
    monkeypatch.setattr(MplCanvas, "_exec_context_menu", lambda self, m, p: menus.append(m))
    canvas._on_context_menu(canvas.rect().center())
    return {a.text(): a for a in menus[-1].actions()}


def _scroll_x_strip(canvas, ax, button="up"):
    """A scroll just below the axes: on its x tick-label strip."""
    bbox = ax.get_window_extent()
    x, y = bbox.x0 + 0.5 * bbox.width, bbox.y0 - 6
    event = types.SimpleNamespace(button=button, step=1, name="scroll_event",
                                  canvas=canvas.canvas, inaxes=None, x=x, y=y)
    event.xdata, event.ydata = None, None
    return event


def _press(canvas, ax, xfrac, yfrac):
    bbox = ax.get_window_extent()
    x, y = bbox.x0 + xfrac * bbox.width, bbox.y0 + yfrac * bbox.height
    event = types.SimpleNamespace(name="button_press_event", canvas=canvas.canvas,
                                  inaxes=ax, x=x, y=y, button=1, dblclick=False, key=None,
                                  guiEvent=None)
    event.xdata, event.ydata = ax.transData.inverted().transform((x, y))
    return event


def _csa_two_sites(window):
    window.sim_csa_nucleus_edit.setText("13C")
    window.sim_csa_orientations_spin.setValue(400)
    window.sim_csa_points_spin.setValue(1024)
    window.sim_csa_larmor_spin.setValue(100.6)
    window.sim_csa_diso_spin.setValue(20.0)
    window.sim_csa_sites.add_site()
    window.sim_csa_diso_spin.setValue(120.0)
    window._on_simulate_csa_clicked()
    return window.simulation_canvas


# ---------------------------------------------------------------------------
# Tick direction
# ---------------------------------------------------------------------------

def test_tick_direction_is_a_plot_style_option(qapp):  # noqa: F811
    window = MainWindow()
    combo = window.output_tick_direction_combo
    assert [combo.itemData(i) for i in range(combo.count())] == ["in", "out", "inout"]
    spectrum = NMRSpectrum(data=np.exp(-np.linspace(-3, 3, 200) ** 2),
                           axes=[NMRAxis(kind="frequency", ppm=np.linspace(10, 0, 200))],
                           metadata=NMRMetadata(vendor="derived", nucleus="1H", sfo1=400.0))
    window.canvas.show_spectrum(spectrum)
    window.canvas_tabs.setCurrentWidget(window.canvas)
    combo.setCurrentIndex(combo.findData("out"))
    assert window.canvas.style_settings()["tick_direction"] == "out"
    tick = window.canvas.ax.xaxis.get_major_ticks()[0]
    assert tick._tickdir == "out"
    # And it holds across a rebuild, and is part of the saved style.
    window.canvas.show_spectrum(spectrum)
    assert window.canvas.ax.xaxis.get_major_ticks()[0]._tickdir == "out"
    assert window._current_output_style_kwargs()["tick_direction"] == "out"
    with pytest.raises(ValueError):
        window.canvas.set_style(tick_direction="sideways")


def test_tick_direction_reaches_contours(qapp):  # noqa: F811
    canvas = MplCanvas()
    canvas.set_style(tick_direction="inout")
    assert canvas._contour_style_kwargs()["tick_dir"] == "inout"
    assert "tick_direction" in MplCanvas._STYLE_KEYS


# ---------------------------------------------------------------------------
# The Simulation plot is a full canvas
# ---------------------------------------------------------------------------

def test_simulation_plot_is_styled_like_every_other(qapp):  # noqa: F811
    window = MainWindow()
    assert isinstance(window.simulation_canvas, MplCanvas)
    assert window.simulation_canvas in window._all_canvases()
    assert window.simulation_canvas in window._measurement_canvases()


def test_multisite_plot_has_right_click_zoom_and_named_series(qapp, monkeypatch):  # noqa: F811
    window = MainWindow()
    canvas = _csa_two_sites(window)
    assert window.canvas_tabs.currentWidget() is canvas
    actions = _menu(canvas, monkeypatch)
    assert "Plot Style…" in actions and "Show Legend" in actions
    # Scroll on the x strip zooms x.
    canvas.canvas.draw()
    ax = canvas.figure.axes[0]
    before = ax.get_xlim()
    canvas._on_scroll(_scroll_x_strip(canvas, ax))
    after = ax.get_xlim()
    assert abs(after[1] - after[0]) < abs(before[1] - before[0])
    # Every site is a series Plot Style can rename and recolour.
    keys = [s.key for s in canvas.series()]
    assert keys == ["Total", "Site 1", "Site 2"]
    canvas.set_series_style("Site 2", label="Carbonyl", colour="#ff0000")
    ax = canvas.figure.axes[0]
    legend = [t.get_text() for t in ax.get_legend().get_texts()]
    assert legend == ["Total", "Site 1", "Carbonyl"]
    line = [l for l in ax.lines if l.get_label() == "Carbonyl"][0]
    assert matplotlib.colors.to_hex(line.get_color()) == "#ff0000"


def test_style_change_redraws_the_simulation_and_keeps_the_zoom(qapp, monkeypatch):  # noqa: F811
    window = MainWindow()
    canvas = _csa_two_sites(window)
    canvas.set_series_style("Site 1", label="Methyl")
    ax = canvas.figure.axes[0]
    ax.set_xlim(150.0, 50.0)
    canvas.set_style(labelsize=17)
    ax = canvas.figure.axes[0]
    assert ax.get_xlim() == pytest.approx((150.0, 50.0))
    assert ax.xaxis.label.get_fontsize() == pytest.approx(17)
    # The renamed series survives the redraw.
    assert "Methyl" in [t.get_text() for t in ax.get_legend().get_texts()]


def test_measure_on_the_simulation_plot_quotes_ppm_and_hz(qapp):  # noqa: F811
    window = MainWindow()
    canvas = _csa_two_sites(window)
    canvas.canvas.draw()
    results = []
    canvas.measurementCompleted.connect(results.append)
    canvas.set_measurement_mode(True)
    ax = canvas.figure.axes[0]
    canvas._on_press(_press(canvas, ax, 0.3, 0.5))
    canvas._on_press(_press(canvas, ax, 0.6, 0.5))
    assert results, "the second click completed a measurement"
    dx = results[-1]["dx"]
    assert dx["ppm"] > 0 and dx["hz"] == pytest.approx(dx["ppm"] * 100.6, rel=1e-6)


def test_tensor_picture_follows_the_map_colour_scale(qapp):  # noqa: F811
    window = MainWindow()
    window.sim_csa_nucleus_edit.setText("13C")
    window.sim_csa_aniso_spin.setValue(-80.0)
    controls = window.sim_csa_tensor_controls
    assert controls.show_tensors()
    canvas = window.simulation_canvas

    def surface_cmaps():
        return {getattr(im.cmap, "name", None) for ax in canvas.figure.axes for im in
                ax.collections + ax.images if getattr(im, "cmap", None) is not None}

    canvas.set_style(map_colormap="magma")
    assert "magma" in surface_cmaps()
    canvas.set_style(labelsize=15)
    ax3d = [ax for ax in canvas.figure.axes if ax.name == "3d"][0]
    assert ax3d.xaxis.label.get_fontsize() == pytest.approx(15)
    combo = window.output_map_colormap_combo
    assert combo.findData("RdBu_r") >= 0 and combo.itemData(0) == ""


def test_bloch_plot_has_plot_style_and_series(qapp, monkeypatch):  # noqa: F811
    window = MainWindow()
    window._run_bloch_simulation()
    canvas = window.simulation_canvas
    assert window.canvas_tabs.currentWidget() is canvas
    assert "Plot Style…" in _menu(canvas, monkeypatch)
    keys = [s.key for s in canvas.series()]
    assert "Mz" in keys and "Mx" in keys
    twin = [ax for ax in canvas.figure.axes if ax.yaxis.get_label_position() == "right"]
    assert twin and twin[0].spines["right"].get_visible()   # its own axis line stays
    canvas.set_style(legend_loc="lower left")
    legends = [ax.get_legend() for ax in canvas.figure.axes if ax.get_legend() is not None]
    assert legends and all(l._loc == 3 for l in legends)   # 3 = "lower left"


def test_diffusion_plot_has_plot_style(qapp, monkeypatch):  # noqa: F811
    window = MainWindow()
    window._run_diffusion_simulation()
    canvas = window.simulation_canvas
    assert "Plot Style…" in _menu(canvas, monkeypatch)
    ax = canvas.figure.axes[0]
    # Mathtext, not Unicode super/subscripts Arial lacks (the logged
    # "Glyph 8315 missing" warnings).
    assert "$" in ax.get_xlabel() and "⁻" not in ax.get_xlabel()
    assert "₀" not in ax.get_ylabel()


def test_stacked_histogram_species_recolour_together(qapp):  # noqa: F811
    """pNMR's neighbour histogram is a stacked histogram, which labels
    only the first bar of each species: recolouring the series must take
    every bar of it."""
    canvas = MplCanvas()

    def draw(figure):
        ax = figure.add_subplot(111)
        d = np.linspace(1, 5, 50)
        ax.hist([d, d], bins=10, stacked=True, label=["O", "Li"], color=["#111111", "#222222"])
        ax.legend()
    canvas.show_external_plot(draw, kind="curve")
    assert [s.key for s in canvas.series()] == ["O", "Li"]
    canvas.set_series_style("Li", colour="#00ff00")
    ax = canvas.figure.axes[0]
    colours = {matplotlib.colors.to_hex(p.get_facecolor()) for p in ax.patches}
    assert "#222222" not in colours and "#00ff00" in colours and "#111111" in colours


# ---------------------------------------------------------------------------
# Measure: a spectrum with no Hz axis (the logged crash)
# ---------------------------------------------------------------------------

def _ppm_only(sfo1):
    ppm = np.linspace(200, -100, 600)
    data = np.exp(-((ppm - 50) / 5) ** 2) + np.exp(-((ppm - 10) / 5) ** 2)
    meta = NMRMetadata(vendor="simulated", nucleus="13C", sfo1=sfo1)
    return NMRSpectrum(data=data, axes=[NMRAxis(kind="frequency", ppm=ppm)], metadata=meta)


@pytest.mark.parametrize("sfo1", [100.6, None])
def test_second_click_completes_a_measurement_without_an_hz_axis(qapp, sfo1):  # noqa: F811
    canvas = MplCanvas()
    canvas.figure.set_size_inches(8, 5)
    canvas.show_spectrum(_ppm_only(sfo1))
    canvas.canvas.draw()
    results = []
    canvas.measurementCompleted.connect(results.append)
    canvas.set_measurement_mode(True)
    canvas._on_press(_press(canvas, canvas.ax, 0.3, 0.5))
    canvas._on_press(_press(canvas, canvas.ax, 0.7, 0.5))
    assert len(results) == 1
    dx = results[0]["dx"]
    assert dx["ppm"] == pytest.approx(results[0]["dx_display"])
    if sfo1:
        assert dx["hz"] == pytest.approx(dx["ppm"] * sfo1)
        assert "Hz" in results[0]["summary"]
    else:
        assert "hz" not in dx and "ppm" in results[0]["summary"]


# ---------------------------------------------------------------------------
# Czjzek: citations, a measured estimate, a quick default
# ---------------------------------------------------------------------------

def test_czjzek_citations():
    plain = cz.citations(False)
    assert any("PhysRevB.23.2513" in c for c in plain)
    assert not any("Le Caër" in c for c in plain)
    extended = cz.citations(True)
    assert any("10.1088/0953-8984/10/47/020" in c for c in extended)
    assert any("Condens. Matter 25, 255402" in c for c in extended)
    x = np.linspace(150, -150, 2048)
    sim = cz.czjzek_spectrum(2.5, 2.0, 130.3e6, x, n_cq=12, n_eta=4)
    assert "PhysRevB.23.2513" in sim["log"] and sim["citations"] == plain


def test_estimate_is_measured_and_scales_with_the_grid():
    x = np.linspace(-150, 150, 4096)
    static = cz.estimate_library_seconds(2.5, 130.3e6, x, 12.0, 8.0, n_cq=32, n_eta=10)
    assert static["method"] == "scaling" and static["seconds"] == 0.0
    mas = cz.estimate_library_seconds(2.5, 130.3e6, x, 12.0, 8.6, n_cq=32, n_eta=10,
                                      mas_Hz=10000.0)
    assert mas["method"] == "simulated" and mas["cells"] == 320
    assert mas["seconds"] == pytest.approx(mas["per_cell"] * 320) and mas["per_cell"] > 0


def test_library_progress_is_reported_and_can_stop():
    calls = []
    x = np.linspace(150, -150, 1024)
    cz.czjzek_spectrum(2.5, 2.0, 130.3e6, x, n_cq=8, n_eta=3,
                       progress=lambda done, total: calls.append((done, total)))
    assert calls and calls[-1] == (24, 24)
    with pytest.raises(cz.SimulationCancelled):
        cz.czjzek_spectrum(2.5, 2.0, 130.3e6, x, n_cq=8, n_eta=3, progress=lambda d, t: False)


def test_czjzek_simulate_panel_defaults_and_cites(qapp):  # noqa: F811
    window = MainWindow()
    sim = window.czjzek_sim_panel
    assert (sim.grid_cq_spin.value(), sim.grid_eta_spin.value()) == (12, 4)
    assert "PhysRevB.23.2513" in sim.citation_label.text()
    assert "Le Caër" not in sim.citation_label.text()
    sim.model_combo.setCurrentIndex(sim.model_combo.findData("extended"))
    assert "Le Caër" in sim.citation_label.text()
    fit = window.czjzek_fit_panel
    assert "PhysRevB.23.2513" in fit.citation_label.text()


def test_slow_czjzek_asks_with_the_measured_time(qapp, monkeypatch):  # noqa: F811
    from opennmr.gui import czjzek_panels
    asked = []
    monkeypatch.setattr(QMessageBox, "question",
                        lambda *a, **k: asked.append(a[2]) or QMessageBox.StandardButton.No)
    monkeypatch.setattr(cz, "estimate_library_seconds", lambda *a, **k: {
        "method": "simulated", "reason": "finite spinning rate", "per_cell": 2.0,
        "seconds": 640.0, "cells": 320})
    window = MainWindow()
    sim = window.czjzek_sim_panel
    sim.regime_combo.setCurrentIndex(sim.regime_combo.findData("mas"))
    assert sim.simulate() is None
    assert asked and "on this computer" in asked[0] and "11 min" in asked[0]
    assert czjzek_panels._format_duration(30) == "30 s"


# ---------------------------------------------------------------------------
# Quiet nucleus checks
# ---------------------------------------------------------------------------

def test_isotope_check_is_quiet_and_the_warning_names_the_string():
    from opennmr.core import registry
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert registry.find_gamma("13X", quiet=True) is None
        assert not caught
        assert registry.find_gamma("13X") is None
    assert any("'13X'" in str(w.message) for w in caught)


def test_measure_on_a_2d_without_hz_axes_gives_both_dimensions(qapp):  # noqa: F811
    x = np.linspace(10, 0, 64)
    y = np.linspace(100, 90, 16)
    data = np.exp(-((x - 5) ** 2))[None, :] * np.exp(-((y - 95) ** 2))[:, None]
    meta = NMRMetadata(vendor="simulated", nucleus="1H", nucleus_indirect="13C",
                       sfo1=400.0, sfo1_indirect=100.0)
    spectrum = NMRSpectrum(data=data, axes=[NMRAxis(kind="frequency", ppm=y),
                                            NMRAxis(kind="frequency", ppm=x)], metadata=meta)
    canvas = MplCanvas()
    canvas.figure.set_size_inches(8, 6)
    canvas.show_spectrum(spectrum)
    canvas.canvas.draw()
    results = []
    canvas.measurementCompleted.connect(results.append)
    canvas.set_measurement_mode(True)
    canvas._on_press(_press(canvas, canvas.ax, 0.3, 0.3))
    canvas._on_press(_press(canvas, canvas.ax, 0.6, 0.7))
    assert results
    dx, dy = results[0]["dx"], results[0]["dy"]
    assert dx["hz"] == pytest.approx(dx["ppm"] * 400.0)
    assert dy["hz"] == pytest.approx(dy["ppm"] * 100.0)
