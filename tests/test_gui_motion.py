"""Dynamics ▸ Anisotropic motion (gui/motion_panel.py): the tables make a model, a sweep and an
overlay run through the notebook engine, Simulate is the source of the environments."""

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMessageBox

from opennmr.gui.motion_panel import AnisotropicMotionPanel, parse_dipolar, parse_values
from opennmr.notebook import motion as M
from opennmr.notebook.combined_hamiltonian import Site

from test_gui import qapp  # noqa: F401 -- fixture


@pytest.fixture
def said(monkeypatch):
    messages = []
    for name in ("warning", "information", "critical"):
        monkeypatch.setattr(QMessageBox, name,
                            lambda *a, _n=name, **k: messages.append(a[2] if len(a) > 2 else ""))
    return messages


def _site_2h(label="D", fwhm=3.0):
    return Site("2H", 0.0, label=label, fwhm=fwhm, lineshape="lorentzian").add_quadrupolar(0.17, 0.0)


# ---------------------------------------------------------------- parsing

def test_values_accept_lists_linear_and_log_ranges():
    assert parse_values("1, 2 3") == [1.0, 2.0, 3.0]
    assert parse_values("0:10:3") == [0.0, 5.0, 10.0]
    assert parse_values("1e2:1e4:3 log") == pytest.approx([1e2, 1e3, 1e4])
    for bad in ("", "a", "1:2", "1e2:1e4:1", "-1:10:3 log", "1:2:500"):
        with pytest.raises(ValueError):
            parse_values(bad)


def test_dipolar_partners_by_coupling_or_distance():
    assert parse_dipolar("1H:23.3, 1H@1.09") == [{"partner": "1H", "D_Hz": 23300.0},
                                                {"partner": "1H", "r_A": 1.09}]
    assert parse_dipolar("") == []
    with pytest.raises(ValueError):
        parse_dipolar("1H 23.3")


# ------------------------------------------------------------------ model

def test_the_example_is_a_three_site_cone_jump(qapp, said):
    panel = AnisotropicMotionPanel()
    model = panel.build_model()
    assert model.n_states == 3 and model.modes[0].n == 3
    assert model.sites[0].euler == (0.0, 70.5, 0.0)
    assert panel.nucleus() == "2H" and panel.larmor_MHz() == pytest.approx(61.4, rel=0.01)
    assert not said


def test_modes_combine_and_keep_their_order(qapp, said):
    panel = AnisotropicMotionPanel()
    panel.add_mode({"kind": "flip", "param": "180", "axis": "x", "k": "2e3"})
    model = panel.build_model()
    assert [m.n for m in model.modes] == [3, 2] and model.n_states == 6
    panel.modes_table.setCurrentCell(1, 1)
    panel.move_mode(-1)
    assert [m.n for m in panel.build_model().modes] == [2, 3]
    assert panel.sweep_combo.findData("all") >= 0


def test_a_mode_needs_exactly_one_of_k_and_ea(qapp, said):
    panel = AnisotropicMotionPanel()
    panel.modes_table.item(0, 4).setText("0.3")                   # k and Ea both given
    with pytest.raises(ValueError, match="either k or Ea"):
        panel.build_model()
    panel.modes_table.item(0, 3).setText("")
    assert "Ea_eV" in panel.build_model().modes[0].rate


def test_hops_join_environments_and_the_dipolar_column_is_read(qapp, said):
    panel = AnisotropicMotionPanel()
    panel.add_site(Site("2H", 5.0, label="E", fwhm=3.0, lineshape="lorentzian").add_quadrupolar(
        0.17, 0.0))
    panel.add_hop([1, 2, "1e3", "", "", ""])
    model = panel.build_model()
    assert model.n_states == 6 and model.hops[0]["k_Hz"] == 1e3
    panel.hops_table.item(0, 1).setText("1")
    with pytest.raises(ValueError, match="two different environments"):
        panel.build_model()
    panel.hops_table.item(0, 1).setText("2")


def test_dipolar_partner_makes_a_dipolar_tensor(qapp, said):
    panel = AnisotropicMotionPanel()
    panel.sites_table.setRowCount(0)
    panel.add_site(Site("13C", 0.0, label="C", fwhm=2.0, lineshape="lorentzian").add_csa(50.0, 0.0))
    panel.sites_table.item(0, 6).setText("1H:23.3")
    model = panel.build_model()
    assert model.sites[0].dipolar[0]["D_Hz"] == pytest.approx(23300.0)


# ------------------------------------------------------------- simulation

def test_simulate_stacks_a_rate_sweep_with_its_limits(qapp, said):
    panel = AnisotropicMotionPanel()
    panel.values_edit.setText("1e3, 1e5, 1e7")
    assert panel.simulate() is not False
    series = panel.series
    assert series["rates"] == [1e3, 1e5, 1e7] and "static" in series["limits"]
    assert "3 spectra" in panel.results_label.text()
    assert "Regime" in panel.status_log.toPlainText()
    assert not said


def test_temperature_sweep_needs_an_arrhenius_rate_and_runs_with_one(qapp, said):
    panel = AnisotropicMotionPanel()
    panel.sweep_combo.setCurrentIndex(panel.sweep_combo.findData("T"))
    panel._picked = "T"
    panel.values_edit.setText("200, 300")
    assert panel.simulate() is False and "Arrhenius" in said[-1]
    panel.modes_table.item(0, 3).setText("")
    panel.modes_table.item(0, 4).setText("0.3")
    panel.modes_table.item(0, 5).setText("1e12")
    panel.simulate()
    assert panel.series["which"] == "T" and panel.series["rates"] == [200.0, 300.0]
    panel._picked = None
    panel._refresh_sweep_choices()
    panel.values_edit.setText("1e3")
    assert panel.simulate() is False and "temperature" in said[-1]
    panel.temperature_edit.setText("300")
    panel.simulate()
    assert panel.series["which"] == "mode:0"


def test_mas_and_echo_reach_the_engine(qapp, said):
    panel = AnisotropicMotionPanel()
    panel.mas_spin.setValue(10.0)
    panel.echo_combo.setCurrentIndex(panel.echo_combo.findData("solid"))
    panel.tau_spin.setValue(200.0)
    panel.values_edit.setText("1e4")
    panel.simulate()
    log = panel.status_log.toPlainText()
    assert "time domain, MAS 10 kHz, solid echo" in log
    echo = panel.series["results"][0]["settings"]["echo"]
    assert echo["kind"] == "solid" and echo["tau_s"] == pytest.approx(2e-4)


def test_limits_report_the_averaged_tensor(qapp, said):
    panel = AnisotropicMotionPanel()
    panel.show_limits()
    log = panel.status_log.toPlainText()
    assert "Motionally averaged tensors" in log and "0.333 of the static value" in log
    assert {"static", "fast", "fast_tensor"} <= set(panel.limits)


def test_bad_values_are_said_not_raised(qapp, said):
    panel = AnisotropicMotionPanel()
    panel.values_edit.setText("nonsense")
    assert panel.simulate() is False and said
    panel.sites_table.setRowCount(0)
    assert panel.show_limits() is False and "environment" in said[-1]


# ---------------------------------------------------------------- overlay

def test_overlay_scales_the_simulation_onto_the_selected_spectrum(qapp, said):
    panel = AnisotropicMotionPanel()
    model = panel.build_model()
    kw = panel._kwargs(model)
    ppm = np.linspace(*M.motion_axis_limits(model, larmor_Hz=kw["larmor_Hz"]), 1024)
    sim = M.motion_spectrum(model, ppm, k_modes={0: 1e5}, **kw)["real_spec"]
    panel.get_spectra = lambda: [{"label": "exp", "ppm": ppm, "y": 4.0 * sim + 0.02, "T_K": None,
                                  "larmor_MHz": kw["larmor_Hz"] / 1e6, "nucleus": "2H"}]
    panel.points_spin.setValue(1024)
    panel.overlay_value_edit.setText("1e5")
    panel.overlay_selected()
    assert panel.overlay["scale"] == pytest.approx(4.0, rel=1e-3)
    assert "Overlay: scale" in panel.results_label.text()
    panel.get_spectra = lambda: []
    assert panel.overlay_selected() is False


def test_overlay_of_a_vt_series_uses_each_spectrum_temperature(qapp, said):
    panel = AnisotropicMotionPanel()
    panel.modes_table.item(0, 3).setText("")
    panel.modes_table.item(0, 4).setText("0.3")
    panel.modes_table.item(0, 5).setText("1e12")
    model = panel.build_model()
    kw = panel._kwargs(model)
    ppm = np.linspace(*M.motion_axis_limits(model, larmor_Hz=kw["larmor_Hz"]), 512)
    spectra = [{"label": f"{T:g} K", "ppm": ppm, "T_K": T, "larmor_MHz": kw["larmor_Hz"] / 1e6,
                "nucleus": "2H", "y": M.motion_spectrum(model, ppm, T_K=T, **kw)["real_spec"]}
               for T in (250.0, 350.0)]
    panel.get_spectra = lambda: spectra
    panel.points_spin.setValue(512)
    panel.overlay_selected()
    assert [o["scale"] for o in panel.overlay["overlays"]] == pytest.approx([1.0, 1.0], rel=1e-3)
    spectra[0]["T_K"] = None
    spectra[1]["T_K"] = None
    assert panel.overlay_selected() is False and "temperature" in said[-1]


# ----------------------------------------------------- notebook and state

def test_notebook_code_runs_and_reproduces_the_gui_sweep(qapp, said):
    panel = AnisotropicMotionPanel()
    panel.values_edit.setText("1e3, 1e5")
    panel.order_combo.setCurrentIndex(0)
    code = panel.notebook_code()
    scope = {}
    exec(compile(code.split("motion.plot_motion_series")[0], "code", "exec"), scope)
    panel.simulate()
    for a, b in zip(scope["series"]["spectra"], panel.series["spectra"]):
        assert np.allclose(a, b, rtol=1e-3, atol=1e-9 * np.max(b))


def test_state_round_trip(qapp, said):
    panel = AnisotropicMotionPanel()
    panel.add_mode({"kind": "flip", "param": "180", "axis": "1,0,1", "Ea": "0.2", "nu0": "1e12"})
    panel.mas_spin.setValue(7.5)
    panel.echo_combo.setCurrentIndex(panel.echo_combo.findData("hahn"))
    panel.temperature_edit.setText("310")
    panel.values_edit.setText("1e2:1e6:5 log")
    state = panel.get_state()
    other = AnisotropicMotionPanel()
    other.set_state(state)
    assert other.get_state() == state
    assert other.build_model().n_states == panel.build_model().n_states == 6


# ---------------------------------------------------- Simulate as the source

def test_sites_and_settings_are_taken_from_simulate(qapp, said):
    sites = [_site_2h("A"), _site_2h("B").add_jcoupling(1, "1H", 5.0) if False else _site_2h("B")]
    panel = AnisotropicMotionPanel(
        get_simulate_sites=lambda: sites,
        get_simulate_settings=lambda: {"larmor_Hz": 76.8e6, "mas_Hz": 12000.0,
                                       "order": "second", "transitions": "all"})
    assert panel.take_from_simulate()
    assert panel.sites_table.rowCount() == 2 and panel.sites_table.item(1, 0).text() == "B"
    assert panel.mas_spin.value() == pytest.approx(12.0) and panel.order_combo.currentData() == "second"
    assert panel.larmor_MHz() == pytest.approx(76.8, rel=1e-3)
    assert "C_Q 0.17 MHz" in panel.sites_table.item(0, 2).text()
    empty = AnisotropicMotionPanel(get_simulate_sites=lambda: [])
    assert empty.take_from_simulate() is False and said


def test_the_window_wires_the_panel_to_simulate_and_the_sections_list(qapp, said):
    from opennmr.gui.main_window import MainWindow
    from opennmr.gui.welcome_wizard import WORK_TYPES

    window = MainWindow()
    assert "motion" in window.OPTIONAL_SECTIONS and "motion" in WORK_TYPES["solution"][2]
    sim = window.combined_sim_panel
    sim.nucleus_combo.setCurrentText("23Na")
    sim.add_sites([Site("23Na", 5.0, label="Na1", fwhm=1.0, lineshape="lorentzian")
                   .add_quadrupolar(2.0, 0.3).to_dict()])
    sites = window._motion_simulate_sites()
    assert sites and all(s.nucleus == "23Na" for s in sites)
    assert window.motion_panel.take_from_simulate()
    assert window.motion_panel.nucleus() == "23Na"


# ------------------------------------------------------------ session journal

def test_a_motion_run_is_journalled_and_restored(qapp, said):
    import json
    from opennmr.gui.main_window import MainWindow
    from opennmr.io import history as io_history

    window = MainWindow()
    panel = window.motion_panel
    panel.values_edit.setText("1e3, 1e5")
    panel.mas_spin.setValue(0.0)
    assert panel.simulate() is not False
    entry = io_history.load_simulation_journal()["entries"][-1]
    assert entry["panel"] == "motion" and entry["label"] == "Anisotropic motion"
    text = "\n".join(entry["inputs_text"])
    assert "Environment 1" in text and "Mode 1" in text and "ring_jump" in text
    assert "2 spectra" in entry["summary"]
    json.dumps(entry)                                   # plain data

    panel.values_edit.setText("5")
    panel.series = None
    window._restore_simulation(entry)
    assert panel.values_edit.text() == "1e3, 1e5"
    assert panel.series is not None and len(panel.series["spectra"]) == 2
    assert io_history.load_simulation_journal()["entries"][-1]["runs"] == 1     # not re-logged


def test_the_limits_run_is_what_is_repeated_on_restore(qapp, said):
    from opennmr.gui.main_window import MainWindow
    from opennmr.io import history as io_history

    window = MainWindow()
    panel = window.motion_panel
    panel.show_limits()
    entry = io_history.load_simulation_journal()["entries"][-1]
    panel.limits = None
    window._restore_simulation(entry)
    assert panel.limits is not None and panel._last_run == "limits"


# ------------------------------------------------------------------- fitting

def _vt_panel(monkeypatch_said=None, temps=(230.0, 290.0, 350.0), arrhenius=True):
    """A motion panel with an Arrhenius cone jump and synthetic VT spectra, and its fit panel."""
    from opennmr.gui.motion_fit_panel import MotionFitPanel

    panel = AnisotropicMotionPanel()
    if arrhenius:
        panel.modes_table.item(0, 3).setText("")
        panel.modes_table.item(0, 4).setText("0.30")
        panel.modes_table.item(0, 5).setText("1e12")
    model = panel.build_model()
    kw = {k: v for k, v in panel._kwargs(model, "T").items() if k != "T_K"}
    x = np.linspace(2500, -2500, 256)
    rng = np.random.default_rng(0)
    spectra = []
    for T in temps:
        y = M.motion_spectrum(model, x, T_K=T, **kw)["real_spec"]
        spectra.append({"label": f"{T:g} K", "ppm": x, "T_K": T, "nucleus": "2H",
                        "larmor_MHz": kw["larmor_Hz"] / 1e6,
                        "y": y / y.max() + rng.normal(0, 0.01, x.size)})
    panel.get_spectra = lambda: spectra
    fit = MotionFitPanel(panel)
    fit.points_spin.setValue(256)
    return panel, fit


def test_the_parameter_table_lists_what_the_model_can_fit(qapp, said):
    panel, fit = _vt_panel()
    assert fit.refresh_parameters()
    names = [fit.table.item(r, 1).data(Qt.ItemDataRole.UserRole) for r in range(fit.table.rowCount())]
    assert {"mode1.Ea_eV", "mode1.log10_k_ref", "site1.beta", "gauss_fwhm_ppm"} <= set(names)
    assert set(fit._parameters()) == {"mode1.Ea_eV", "mode1.log10_k_ref"}
    fit.table.item(names.index("site1.beta"), 0).setCheckState(Qt.CheckState.Checked)
    fit.refresh_parameters()                              # the choice survives a refresh
    assert "site1.beta" in fit._parameters()
    for r in range(fit.table.rowCount()):
        fit.table.item(r, 0).setCheckState(Qt.CheckState.Unchecked)
    with pytest.raises(ValueError, match="at least one"):
        fit._parameters()


def test_the_rate_route_fits_each_spectrum_and_writes_the_line_back(qapp, said):
    panel, fit = _vt_panel()
    fit.route_combo.setCurrentIndex(fit.route_combo.findData("rates"))
    assert fit.run_fit() is not False and not said
    assert "3 of 3 rates determined" in fit.results_label.text() and "Arrhenius" in fit.results_label.text()
    assert fit.fit["arrhenius"]["Ea_eV"] == pytest.approx(0.30, abs=0.03)
    panel.modes_table.item(0, 4).setText("0.1")
    assert fit.use_fitted_values()
    assert float(panel._cell(panel.modes_table, 0, 4)) == pytest.approx(0.30, abs=0.03)
    assert float(panel._cell(panel.modes_table, 0, 3 + 2)) > 1e10


def test_the_global_route_fits_the_checked_parameters_and_fills_the_table(qapp, said):
    panel, fit = _vt_panel()
    panel.modes_table.item(0, 4).setText("0.22")
    panel.modes_table.item(0, 5).setText("1e11")
    fit.route_combo.setCurrentIndex(fit.route_combo.findData("global"))
    fit.refresh_parameters()
    assert fit.run_fit() is not False and not said
    assert "Fitted:" in fit.results_label.text() and "mode1.log10_nu0" in fit.results_label.text()
    row = next(r for r in range(fit.table.rowCount())
               if fit.table.item(r, 1).data(Qt.ItemDataRole.UserRole) == "mode1.Ea_eV")
    assert float(fit.table.item(row, 5).text()) == pytest.approx(0.30, abs=0.015)
    assert fit.table.item(row, 6).text() != ""
    assert fit.use_fitted_values()
    assert float(panel._cell(panel.modes_table, 0, 4)) == pytest.approx(0.30, abs=0.015)
    assert "mode1.Ea_eV" in fit.status_log.toPlainText()


def test_the_fit_says_what_is_missing(qapp, said):
    panel, fit = _vt_panel()
    panel.get_spectra = lambda: []
    assert fit.run_fit() is False and "Select one or more" in said[-1]
    panel2, fit2 = _vt_panel(arrhenius=False)
    fit2.route_combo.setCurrentIndex(fit2.route_combo.findData("global"))
    fit2.refresh_parameters()
    assert fit2.run_fit() is False and said                # a fixed rate has no Ea to fit
    fit2.route_combo.setCurrentIndex(fit2.route_combo.findData("rates"))
    fit2.kmin_edit.setText("1e6")
    fit2.kmax_edit.setText("1e2")
    assert fit2.run_fit() is False and "scan range" in said[-1]


def test_fit_state_and_notebook_code(qapp, said):
    panel, fit = _vt_panel()
    fit.refresh_parameters()
    fit.starts_spin.setValue(3)
    fit.tref_edit.setText("280")
    fit.route_combo.setCurrentIndex(fit.route_combo.findData("global"))
    state = fit.get_state()
    import json
    json.dumps(state)
    assert any("mode1.Ea_eV" in v for v in state["extra"].values())
    other_panel, other = _vt_panel()
    other.set_state(json.loads(json.dumps(state)))
    assert other.get_state() == state
    code = fit.notebook_code()
    compile(code, "code", "exec")
    assert "fit_motion(model, spectra, parameters, T_ref_K=280.0, n_starts=3" in code
    fit.route_combo.setCurrentIndex(fit.route_combo.findData("rates"))
    compile(fit.notebook_code(), "code", "exec")


def test_fitted_site_values_are_written_back_to_the_environment_table(qapp, said):
    panel = AnisotropicMotionPanel()
    d = panel.build_model().to_dict()
    d["sites"][0]["euler"] = [0.0, 33.0, 0.0]
    d["sites"][0]["site"]["interactions"][0]["C_Q"] = 0.2
    done = panel.apply_fitted_model(d, ["site1.beta", "site1.C_Q", "mode1.log10_k"])
    assert done == ["site1.beta", "site1.C_Q", "mode1.log10_k"]
    assert panel._cell(panel.sites_table, 0, 4) == "33"
    assert "C_Q 0.2 MHz" in panel.sites_table.item(0, 2).text()
    assert panel.build_model().sites[0].euler[1] == 33.0


def test_a_fit_is_journalled_and_the_window_wires_it(qapp, said):
    import json

    from opennmr.gui.main_window import MainWindow
    from opennmr.io import history as io_history

    window = MainWindow()
    assert "_motion_fit_section" in window.OPTIONAL_SECTIONS["motion"][1]
    section = window._simulation_panels()["motion_fit"]
    before = section.get_state()
    section.set_state(json.loads(json.dumps(before)))
    assert section.get_state() == before
    panel, _unused = _vt_panel()
    window.motion_panel.get_spectra = panel.get_spectra
    window.motion_panel.modes_table.item(0, 3).setText("")
    window.motion_panel.modes_table.item(0, 4).setText("0.30")
    window.motion_panel.modes_table.item(0, 5).setText("1e12")
    fit = window.motion_fit_panel
    fit.points_spin.setValue(256)
    fit.route_combo.setCurrentIndex(fit.route_combo.findData("rates"))
    assert fit.run_fit() is not False
    entry = io_history.load_simulation_journal()["entries"][-1]
    assert entry["panel"] == "motion_fit" and "rates determined" in entry["summary"]
    fit.results_label.setText("")
    window._restore_simulation(entry)                     # the inputs come back, no re-run
    assert fit.route_combo.currentData() == "rates"


def test_a_dipolar_coupling_is_fitted_and_written_back_in_khz(qapp, said):
    from opennmr.gui.motion_fit_panel import MotionFitPanel

    panel = AnisotropicMotionPanel()
    panel.sites_table.setRowCount(0)
    panel.add_site(Site("13C", 0.0, label="C", fwhm=15.0, lineshape="lorentzian").add_csa(40.0, 0.0),
                   euler=(0.0, 60.0, 0.0))
    panel.modes_table.item(0, 3).setText("2e4")
    panel.sites_table.item(0, 6).setText("1H@1.09, 1H:2.0")
    truth = panel.build_model()
    d0 = truth.sites[0].dipolar
    assert len(d0) == 2 and d0[0]["D_Hz"] > 1e4
    fit = MotionFitPanel(panel)
    fit.points_spin.setValue(256)
    fit.route_combo.setCurrentIndex(fit.route_combo.findData("global"))
    fit.refresh_parameters()
    names = [fit.table.item(r, 1).data(Qt.ItemDataRole.UserRole) for r in range(fit.table.rowCount())]
    assert {"site1.dipolar1.D_Hz", "site1.dipolar2.D_Hz"} <= set(names)
    assert "1H" in fit.table.item(names.index("site1.dipolar1.D_Hz"), 1).text()
    # data from the model with the first coupling 8 % larger
    d = truth.to_dict()
    d["sites"][0]["dipolar"][0]["D_Hz"] *= 1.08
    kw = {k: v for k, v in panel._kwargs(truth, "T").items() if k != "T_K"}
    x = np.linspace(400, -400, 256)
    y = M.motion_spectrum(M.MotionModel.from_dict(d), x, **kw)["real_spec"]
    panel.get_spectra = lambda: [{"label": "s", "ppm": x, "y": y / y.max(), "T_K": None,
                                  "larmor_MHz": kw["larmor_Hz"] / 1e6, "nucleus": "13C"}]
    for r, name in enumerate(names):
        on = name == "site1.dipolar1.D_Hz"
        fit.table.item(r, 0).setCheckState(Qt.CheckState.Checked if on else Qt.CheckState.Unchecked)
    assert fit.run_fit() is not False and not said
    assert fit.fit["params"]["site1.dipolar1.D_Hz"] == pytest.approx(d0[0]["D_Hz"] * 1.08, rel=0.01)
    assert fit.use_fitted_values()
    cell = panel._cell(panel.sites_table, 0, 6)
    assert cell.startswith("1H:") and cell.endswith("1H:2") and "@" not in cell
    assert panel.build_model().sites[0].dipolar[0]["D_Hz"] == pytest.approx(
        d["sites"][0]["dipolar"][0]["D_Hz"], rel=0.01)
