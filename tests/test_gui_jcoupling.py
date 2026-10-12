"""Simulate > J-coupling and Fit > J-coupling (gui/jcoupling_panels.py).

The fits run against exact synthetic multiplets whose parameters are
known, loaded into the window as ordinary spectra, so these check the
whole path: table -> notebook fitter -> results table."""
import numpy as np
import pytest
from PySide6.QtWidgets import QMessageBox

from opennmr.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
from opennmr.gui.main_window import MainWindow
from opennmr.gui.widgets.collapsible_section import CollapsibleSection
from opennmr.notebook import simulations as S
from PySide6.QtCore import Qt

from test_gui import qapp  # noqa: F401 -- fixture

L_1H = 400.13e6


@pytest.fixture(autouse=True)
def _dialogs(monkeypatch):
    shown = []
    for name in ("warning", "information", "critical"):
        monkeypatch.setattr(QMessageBox, name,
                            lambda *a, _n=name, **k: shown.append((_n, a[2] if len(a) > 2 else "")))
    return shown


def _titles(window, tab_name):
    names = [window.side_tabs.tabText(i) for i in range(window.side_tabs.count())]
    page = window.side_tabs.widget(names.index(tab_name))
    return [s.title() for s in page.findChildren(CollapsibleSection)]


def test_sections_are_where_asked(qapp):  # noqa: F811
    # 2026-10-06: no J-coupling section any more -- first-order J is an
    # interaction of a CSA, Quad & J-coupling peak, and these panels are
    # Combined's "Strongly coupled peaks" model, in spin-system mode.
    window = MainWindow()
    fit = _titles(window, "Fit")
    assert "J-coupling" not in fit and "CSA/Dipolar" not in fit
    assert fit.index("Integration") + 1 == fit.index("CSA, Quad & J-coupling")
    assert _titles(window, "Simulate")[0] == "CSA, Quad & J-coupling"
    for model, panel in ((window.combined_fit_model, window.jcoupling_fit_panel),
                         (window.combined_sim_model, window.jcoupling_sim_panel)):
        assert model.spin_widget is panel and panel.spin_mode()


# --- the shared editor -------------------------------------------------------

def _editor(qapp, mode="simulate"):  # noqa: F811
    """The first-order editor. Since 2026-10-06 the panels open in
    spin-system mode (first-order J is CSA, Quad & J-coupling's); their
    first-order mode remains, for kept fits and saved states."""
    window = MainWindow()
    panel = window.jcoupling_sim_panel if mode == "simulate" else window.jcoupling_fit_panel
    panel.model_combo.setCurrentIndex(panel.model_combo.findData("first_order"))
    return window, panel, panel.editor


def _set_coupling_cell(editor, row, col, text):
    editor.couplings_table.item(row, col).setText(str(text))


def test_editing_a_coupling_and_reselecting_keeps_it(qapp):  # noqa: F811
    _, _, ed = _editor(qapp)
    ed.sites_table.selectRow(0)
    _set_coupling_cell(ed, 0, 2, "7.25")          # stored as typed text
    ed.add_site(label="B", shift=3.0)
    ed.sites_table.selectRow(0)                   # regression: re-showing text crashed
    assert ed.couplings_table.item(0, 2).text() == "7.25"
    sites = ed.sites(L_1H)
    assert sites[0]["couplings"][0]["J_Hz"] == 7.25


def test_linewidth_is_entered_in_hz(qapp):  # noqa: F811
    _, _, ed = _editor(qapp)
    ed.sites_table.item(0, 2).setText("2.0")
    assert ed.sites(L_1H)[0]["fwhm"] == pytest.approx(2.0 / L_1H * 1e6)


def test_pattern_is_stated_as_it_is_built(qapp):  # noqa: F811
    _, _, ed = _editor(qapp)
    ed.sites_table.selectRow(0)
    ed.add_coupling("31P", 1, 140.0)
    text = ed.pattern_label.text()
    assert text.startswith("CH2: dq (") and "8 lines (+" in text   # + 1H's 0.01% 2H


def test_partner_notes_satellites_and_other_magnetic_isotopes(qapp):  # noqa: F811
    _, _, ed = _editor(qapp)
    ed.sites_table.selectRow(0)
    _set_coupling_cell(ed, 0, 0, "13C")
    assert "satellites" in ed.pattern_label.text()
    _set_coupling_cell(ed, 0, 0, "11B")
    assert "10B" in ed.pattern_label.text() and "non-magnetic" in ed.pattern_label.text()
    _set_coupling_cell(ed, 0, 3, "100")           # a pure isotope: nothing to warn
    assert "10B" not in ed.pattern_label.text()


@pytest.mark.parametrize("col, text, message", [
    (0, "1X", "not a nuclide"), (1, "two", "whole number"), (1, "0", "at least 1"),
    (3, "150", "abundance")])
def test_bad_coupling_cells_are_named(qapp, col, text, message):  # noqa: F811
    _, _, ed = _editor(qapp)
    ed.sites_table.selectRow(0)
    _set_coupling_cell(ed, 0, col, text)
    with pytest.raises(ValueError, match=message):
        ed.sites(L_1H)


# --- simulation --------------------------------------------------------------

def test_simulated_quartet_is_where_it_should_be(qapp):  # noqa: F811
    window, sim, _ed = _editor(qapp)
    assert sim.simulate()
    spec = sim.spectrum
    x, y = spec.axes[0].ppm, spec.data
    L = sim.larmor_Hz()
    assert L == pytest.approx(42.57746460132430e6 * 9.395, rel=1e-9)
    lines = 1.25 + np.array([-1.5, -0.5, 0.5, 1.5]) * 7.0 / L * 1e6
    heights = [y[np.argmin(abs(x - p))] for p in lines]
    assert np.allclose(np.array(heights) / max(heights), [1 / 3, 1, 1, 1 / 3], atol=0.01)
    assert spec.metadata.nucleus == "1H" and spec.metadata.sfo1 == pytest.approx(L / 1e6)
    assert "dq" not in sim.results_label.text() and "q," in sim.results_label.text()
    assert window.canvas_tabs.currentWidget() is window.fit_canvas


def test_export_and_staleness(qapp):  # noqa: F811
    window, sim, ed = _editor(qapp)
    assert not sim.export_button.isEnabled()
    sim.simulate()
    n = window.spectrum_list.count()
    sim.export()
    assert window.spectrum_list.count() == n + 1
    ed.sites_table.item(0, 1).setText("1.3")
    assert not sim.export_button.isEnabled()
    assert "inputs changed" in sim.results_label.text()


def test_undersampled_or_clipped_axis_is_flagged(qapp):  # noqa: F811
    _, sim, _ed = _editor(qapp)
    sim.auto_axis_check.setChecked(False)
    sim.ppm_min_spin.setValue(1.25)
    sim.ppm_max_spin.setValue(3.0)
    sim.points_spin.setValue(500)
    sim.simulate()
    text = sim.results_label.text()
    assert "edge of the axis" in text and "under-sampled" in text


def test_simulation_notebook_code_reruns(qapp):  # noqa: F811
    _, sim, _ed = _editor(qapp)
    sim.simulate()
    ns = {}
    exec(sim.notebook_code(), ns)  # noqa: S102 -- our own generated code
    assert np.allclose(ns["spec"]["real_spec"], sim.spectrum.data)


# --- fitting -----------------------------------------------------------------

def _synthetic(window, truth, n=32768, lo=0.0, hi=6.0, sfo1=True):
    """truth: [(centre ppm, [(J Hz, n), ...], fwhm Hz, height)]"""
    x = np.linspace(hi, lo, n)
    y = np.zeros(n)
    for c, groups, fwhm_hz, height in truth:
        freqs, w = S.combine_jcouplings([{"nuclide": "1H", "n": k, "J_Hz": J, "abundance": 100.0}
                                         for J, k in groups])
        sticks = c + freqs / L_1H * 1e6
        f = fwhm_hz / L_1H * 1e6
        top = S.multiplet_lineshape(sticks, sticks, w, f, normalise="peak").max()
        y += height * S.multiplet_lineshape(x, sticks, w, f, normalise="peak") / top
    spectrum = NMRSpectrum(data=y, axes=[NMRAxis(kind="frequency", ppm=x)],
                           metadata=NMRMetadata(vendor="simulated", nucleus="1H",
                                                sfo1=L_1H / 1e6 if sfo1 else None))
    window._add_derived_spectrum(spectrum, "synthetic")
    return spectrum


TRUTH = [(3.40, [(7.1, 2)], 0.9, 1000.0), (1.20, [(7.1, 3), (1.6, 1)], 1.3, 600.0)]


def _set_up_two_sites(ed):
    ed.clear()
    ed.add_site(label="CH2", shift=3.401, fwhm_hz=1.0,
                couplings=[{"nuclide": "1H", "n": 2, "J_Hz": 7.0, "abundance": 100.0}])
    ed.add_site(label="CH3", shift=1.199, fwhm_hz=1.0,
                couplings=[{"nuclide": "1H", "n": 3, "J_Hz": 7.0, "abundance": 100.0},
                           {"nuclide": "1H", "n": 1, "J_Hz": 1.5, "abundance": 100.0}])


def test_fit_recovers_known_multiplets(qapp):  # noqa: F811
    window, fit, ed = _editor(qapp, "fit")
    _synthetic(window, TRUTH)
    _set_up_two_sites(ed)
    assert fit.fit()
    got = fit.fitted_sites()
    assert got[0]["center_ppm"] == pytest.approx(3.40, abs=1e-6)
    assert got[0]["couplings"][0]["J_Hz"] == pytest.approx(7.1, abs=1e-3)
    assert got[0]["fwhm"] * L_1H / 1e6 == pytest.approx(0.9, abs=1e-3)
    assert got[0]["relative_intensity"] == pytest.approx(1000, rel=1e-3)
    assert [c["J_Hz"] for c in got[1]["couplings"]] == pytest.approx([7.1, 1.6], abs=1e-3)
    assert got[1]["fwhm"] * L_1H / 1e6 == pytest.approx(1.3, abs=1e-3)
    t = fit.results_table
    assert t.rowCount() == 2 and t.item(1, 5).text().startswith("qd,")
    assert "R² 1.0000" in fit.results_label.text()
    assert window.canvas_tabs.currentWidget() is window.fit_canvas
    assert "Site\tShift (ppm)" in fit.table_as_tsv()
    assert "J-COUPLING FIT REPORT" in fit.text_report()


def test_fitted_values_become_the_next_start(qapp):  # noqa: F811
    window, fit, ed = _editor(qapp, "fit")
    _synthetic(window, TRUTH)
    _set_up_two_sites(ed)
    fit.fit()
    fit.use_fitted_as_start()
    sites = ed.sites(L_1H)
    assert sites[1]["couplings"][1]["J_Hz"] == pytest.approx(1.6, abs=1e-3)
    assert sites[0]["center_ppm"] == pytest.approx(3.40, abs=1e-6)


def test_open_in_simulate_reproduces_the_fit(qapp):  # noqa: F811
    window, fit, ed = _editor(qapp, "fit")
    _synthetic(window, TRUTH)
    _set_up_two_sites(ed)
    fit.fit()
    fit.send_to_simulation()
    sim = window.jcoupling_sim_panel
    assert window.side_tabs.tabText(window.side_tabs.currentIndex()) == "Simulate"
    assert sim.larmor_Hz() == pytest.approx(L_1H, rel=1e-7)
    sim.simulate()
    # The simulation is the fitted model (baseline aside) up to one overall
    # scale -- both sites, although their linewidths differ.
    from opennmr.notebook import jcoupling_fitting as JF
    x, y = sim.spectrum.axes[0].ppm, sim.spectrum.data
    r = fit.result
    model = JF.evaluate_jcoupling_fit(x, r["popt"], r["info"])
    model -= r["popt"][-2] + r["popt"][-1] * (x - r["info"]["x_ref"])
    big = model > 0.05 * model.max()
    ratio = y[big] / model[big]
    assert ratio.std() / ratio.mean() < 1e-4


def test_fixed_j_is_held_and_marked(qapp):  # noqa: F811
    window, fit, ed = _editor(qapp, "fit")
    _synthetic(window, TRUTH[:1])
    ed.clear()
    ed.add_site(label="CH2", shift=3.40, fwhm_hz=1.0,
                couplings=[{"nuclide": "1H", "n": 2, "J_Hz": 7.0, "abundance": 100.0}])
    ed.set_coupling_constraint(0, 0, fixed=True)
    assert ed.couplings_table.item(0, 2).font().italic()
    fit.fit()
    assert fit.fitted_sites()[0]["couplings"][0]["J_Hz"] == 7.0
    assert "(fixed)" in fit.results_table.item(0, 1).text()


def test_shift_is_kept_near_its_start(qapp):  # noqa: F811
    # A start far from the truth may not leave the allowed window, and
    # the report says it stopped there.
    window, fit, ed = _editor(qapp, "fit")
    _synthetic(window, TRUTH[:1])
    ed.clear()
    ed.add_site(label="CH2", shift=3.30, fwhm_hz=1.0,
                couplings=[{"nuclide": "1H", "n": 2, "J_Hz": 7.0, "abundance": 100.0}])
    fit.shift_window_spin.setValue(0.02)
    fit.window_combo.setCurrentIndex(fit.window_combo.findData("range"))
    fit.window_lo_spin.setValue(3.2)
    fit.window_hi_spin.setValue(3.5)
    fit.fit()
    assert fit.fitted_sites()[0]["center_ppm"] <= 3.32 + 1e-9
    assert "edge of its allowed range" in fit.results_label.text()


def test_integration_regions_window_needs_regions(qapp, _dialogs):  # noqa: F811
    window, fit, _ed = _editor(qapp, "fit")
    _synthetic(window, TRUTH[:1])
    fit.window_combo.setCurrentIndex(fit.window_combo.findData("regions"))
    assert not fit.fit()
    assert "Integration regions" in _dialogs[-1][1]


def test_no_spectrum_or_no_frequency_is_explained(qapp, _dialogs):  # noqa: F811
    window, fit, _ed = _editor(qapp, "fit")
    assert not fit.fit()
    assert "Select a spectrum" in _dialogs[-1][1]
    _synthetic(window, TRUTH[:1], sfo1=False)
    assert not fit.fit()
    assert "observe frequency" in _dialogs[-1][1]


# --- all isotopes of a partner element ---------------------------------------

def test_all_isotopes_tick_adds_the_other_isotope(qapp):  # noqa: F811
    _, sim, ed = _editor(qapp)
    ed.sites_table.selectRow(0)
    _set_coupling_cell(ed, 0, 0, "11B")
    _set_coupling_cell(ed, 0, 1, "1")
    tick = ed.couplings_table.item(0, ed._ALL_ISOTOPES_COL)
    tick.setCheckState(Qt.CheckState.Checked)
    assert ed.sites(L_1H)[0]["couplings"][0]["all_isotopes"] is True
    text = ed.pattern_label.text()
    assert "10B 19.9%" in text and "11B 80.1%" in text and "⚠" not in text
    assert sim.simulate()
    _set_coupling_cell(ed, 0, 3, "50")                       # abundance and tick: refused
    with pytest.raises(ValueError, match="All isotopes"):
        ed.sites(L_1H)


# --- spin systems: simulate ----------------------------------------------------

def _spin_sim(qapp):  # noqa: F811
    window = MainWindow()
    sim = window.jcoupling_sim_panel
    sim.model_combo.setCurrentIndex(sim.model_combo.findData("spin_system"))
    return window, sim


def test_spin_system_simulates_an_exact_ab_quartet(qapp):  # noqa: F811
    from opennmr.notebook import spin_system as SS
    _, sim = _spin_sim(qapp)
    assert sim.model_stack.currentIndex() == 1
    sim.points_spin.setValue(200_000)
    assert sim.simulate()
    x, y = sim.spectrum.axes[0].ppm, sim.spectrum.data
    L = sim.larmor_Hz()
    t = SS.spin_system_transitions(sim.spin_editor.spins(), sim.spin_editor.couplings(),
                                   sim.field_spin.value())
    for p, w in zip(t["stick_ppm"], t["intensity"] / t["intensity"].max()):
        i = np.argmin(abs(x - p))
        assert y[i] / y.max() == pytest.approx(w, abs=0.02)   # roofed quartet as drawn
    assert "Strongly coupled" in sim.results_label.text()
    ns = {}
    exec(sim.notebook_code(), ns)  # noqa: S102
    assert np.allclose(ns["spec"]["real_spec"], sim.spectrum.data)
    sim.export()


def test_spin_editor_keeps_j_matrix_consistent(qapp):  # noqa: F811
    _, sim = _spin_sim(qapp)
    ed = sim.spin_editor
    ed.add_spin(label="X", nuclide="31P", shift=0.0)
    ed.set_J("A", "X", 12.0)
    ed.j_table.item(1, 2).setText("-3")                      # B-X, upper triangle
    assert ed.j_table.item(2, 1).text() == "-3"              # mirrored
    assert {(c["a"], c["b"]): c["J_Hz"] for c in ed.couplings()} == {
        ("A", "B"): 8.0, ("A", "X"): 12.0, ("B", "X"): -3.0}
    ed.spins_table.item(1, 0).setText("M")                   # rename keeps its Js
    assert ("A", "M") in {(c["a"], c["b"]) for c in ed.couplings()}
    ed.spins_table.selectRow(0)
    ed.remove_spin()
    assert ed.labels() == ["M", "X"]
    assert [(c["a"], c["b"], c["J_Hz"]) for c in ed.couplings()] == [("M", "X", -3.0)]


def test_spin_system_needs_the_observed_nucleus(qapp, _dialogs):  # noqa: F811
    _, sim = _spin_sim(qapp)
    sim.nucleus_combo.setCurrentText("19F")
    assert not sim.simulate()
    assert "no 19F spin" in _dialogs[-1][1]


def test_oversized_spin_system_is_explained(qapp, _dialogs):  # noqa: F811
    _, sim = _spin_sim(qapp)
    for k in range(13):
        sim.spin_editor.add_spin(nuclide="1H", shift=1.0 + 0.1 * k)
    assert "too large" in sim.spin_editor.info_label.text()
    assert not sim.simulate()
    assert "block" in _dialogs[-1][1]


# --- spin systems: fit ---------------------------------------------------------

ABX_TRUTH = ([{"label": "A", "nuclide": "1H", "shift_ppm": 7.30},
              {"label": "B", "nuclide": "1H", "shift_ppm": 7.28},
              {"label": "X", "nuclide": "31P", "shift_ppm": 20.0}],
             [{"a": "A", "b": "B", "J_Hz": 8.0}, {"a": "A", "b": "X", "J_Hz": 12.0},
              {"a": "B", "b": "X", "J_Hz": -3.0}])


def _abx_window(qapp):  # noqa: F811
    from opennmr.notebook import spin_system as SS
    window = MainWindow()
    x = np.linspace(8.0, 6.5, 65536)
    B0 = L_1H / abs(S.find_isotope_info("1H")["gamma"])
    y = 500 * SS.spin_system_spectrum(*ABX_TRUTH, B0, x, fwhm_Hz=0.7, scale="peak")["real_spec"]
    spectrum = NMRSpectrum(data=y, axes=[NMRAxis(kind="frequency", ppm=x)],
                           metadata=NMRMetadata(vendor="simulated", nucleus="1H",
                                                sfo1=L_1H / 1e6))
    window._add_derived_spectrum(spectrum, "ABX")
    fit = window.jcoupling_fit_panel
    fit.model_combo.setCurrentIndex(fit.model_combo.findData("spin_system"))
    start = [dict(s) for s in ABX_TRUTH[0]]
    start[0]["shift_ppm"] += 0.002
    start[1]["shift_ppm"] -= 0.002
    fit.spin_editor.set_system(start, [{"a": "A", "b": "B", "J_Hz": 7.5},
                                       {"a": "A", "b": "X", "J_Hz": 11.0},
                                       {"a": "B", "b": "X", "J_Hz": -2.0}])
    fit.spin_fwhm_spin.setValue(1.0)
    return window, fit


def test_spin_fit_reports_both_abx_solutions(qapp):  # noqa: F811
    window, fit = _abx_window(qapp)
    assert "A–B" in fit.spin_editor.info_label.text()        # field read from the spectrum
    assert fit.fit()
    assert fit.solution_combo.count() == 2 and fit.solution_combo.isVisibleTo(fit)
    assert "2 solutions fit equally well" in fit.results_label.text()
    assert "shift X (ppm) held" in fit.results_label.text()
    from opennmr.notebook import spin_system as SS
    truths = []
    for k in range(2):
        fit.solution_combo.setCurrentIndex(k)
        got = dict(zip(fit.result["info"]["names"], fit.result["popt"]))
        truths.append(abs(got["J A-X (Hz)"] - 12.0) < 1e-3 and abs(got["J B-X (Hz)"] + 3.0) < 1e-3
                      and abs(got["shift A (ppm)"] - 7.30) < 1e-6)
        assert "R² 1.0000" in fit.results_label.text()
    assert sum(truths) == 1
    assert window.canvas_tabs.currentWidget() is window.fit_canvas
    assert "Parameter\tValue" in fit.table_as_tsv()


def test_spin_fit_round_trips(qapp):  # noqa: F811
    window, fit = _abx_window(qapp)
    fit.ambiguity_check.setChecked(False)
    fit.fit()
    assert fit.solution_combo.count() == 1 and not fit.solution_combo.isVisibleTo(fit)
    fit.use_fitted_as_start()
    fitted = dict(zip(fit.result["info"]["names"], fit.result["popt"]))
    assert {(c["a"], c["b"]): c["J_Hz"] for c in fit.spin_editor.couplings()}[("A", "B")] == \
        pytest.approx(fitted["J A-B (Hz)"], abs=1e-4)
    fit.send_to_simulation()
    sim = window.jcoupling_sim_panel
    assert sim.spin_mode() and sim.spin_fwhm_spin.value() == pytest.approx(0.7, abs=1e-3)
    sim.points_spin.setValue(100_000)
    sim.simulate()
    from opennmr.notebook import spin_system as SS
    x, y = sim.spectrum.axes[0].ppm, sim.spectrum.data
    model = SS.evaluate_spin_system_fit(x, fit.result["popt"], fit.result["info"],
                                        include_baseline=False)
    big = model > 0.05 * model.max()
    ratio = y[big] / model[big]
    assert ratio.std() / ratio.mean() < 2e-3                  # fwhm rounded to 1e-3 Hz


def test_spin_fit_fixed_j(qapp):  # noqa: F811
    _, fit = _abx_window(qapp)
    fit.spin_editor.set_constraint("J A-B (Hz)", fixed=True)
    fit.ambiguity_check.setChecked(False)
    fit.fit()
    got = dict(zip(fit.result["info"]["names"], fit.result["popt"]))
    assert got["J A-B (Hz)"] == 7.5
    k = fit.result["info"]["names"].index("J A-B (Hz)")
    assert "(fixed)" in fit.results_table.item(k, 0).text()


def test_a_spin_system_fit_and_its_chosen_solution_come_back(qapp):  # noqa: F811
    # Saved with the spectrum (processing log / Load Progress): the
    # second of two equally good ABX solutions is still the one shown.
    window, fit = _abx_window(qapp)
    assert fit.fit()
    fit.solution_combo.setCurrentIndex(1)
    shown = (fit.table_as_tsv(), fit.results_label.text())
    saved = window.session.spectra[0].metadata.raw["fits"]["jcoupling"]
    assert serialize_decode(saved["result"])["chosen"] == 1
    window._add_derived_spectrum(NMRSpectrum(
        data=np.zeros(8), axes=[NMRAxis(kind="frequency", ppm=np.linspace(1, 0, 8))],
        metadata=NMRMetadata(vendor="simulated", nucleus="1H")), "blank")
    assert fit.result is None and not fit.solution_combo.isVisibleTo(fit)
    window._select_single_row(0)
    assert fit.solution_combo.currentIndex() == 1 and fit.solution_combo.isVisibleTo(fit)
    assert (fit.table_as_tsv(), fit.results_label.text()) == shown
    assert fit.spin_mode() and len(fit.spin_editor.labels()) == 3


def serialize_decode(value):
    from opennmr.io import serialize
    return serialize.decode(value)
