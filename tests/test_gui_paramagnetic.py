"""GUI for the paramagnetic NMR features (user request, 2026-11-06):
Simulate ▸ DFT tensors with an ORCA EPR output (g, ZFS, hyperfine), and
Simulate ▸ Paramagnetic NMR tools: shift vs T, fast BMS shifts (2026-11-07,
which replaced the lanthanide-series and pseudocontact sections), bond
pathways."""

import numpy as np
import pytest
from PySide6.QtWidgets import QMessageBox, QTableWidgetItem

from opennmr.gui.dft_panel import DftSimulationPanel
from opennmr.gui.paramagnetic_panel import ParamagneticToolsPanel
from opennmr.notebook import bond_pathways as bp
from test_bond_pathways import BINS, SCALING, _calc
from test_gui import qapp  # noqa: F401
from test_paramagnetic_nmr import MN_HFC, _mn_file


@pytest.fixture
def no_dialogs(monkeypatch):
    said = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: said.append(a[2]))
    return said


class _Plots:
    def __init__(self):
        self.drawn = []

    def __call__(self, draw, kind):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig = plt.figure()
        draw(fig)
        self.drawn.append((kind, fig))


def _dft_panel(tmp_path, plots=None):
    p = DftSimulationPanel(plot_spectrum=lambda s: None, add_to_open=lambda s: None,
                           draw_plot=plots)
    p.dft_sample_edit.setText(_mn_file(tmp_path))
    assert p.read_files(quiet=True), p.dft_files_label.text()
    p.dft_element_combo.setCurrentIndex(p.dft_element_combo.findText("Na"))
    p.larmor_spin.setValue(0)
    return p


def test_orca_epr_file_shows_its_box_and_sites(qapp, tmp_path, no_dialogs):
    p = _dft_panel(tmp_path)
    assert not p.dft_opm_box.isHidden() and p.dft_pm_box.isHidden()
    assert p.dft_opm_S_spin.value() == 2.5
    assert p.dft_site_list.count() == 1
    assert "g principal values" in p.dft_opm_label.text()


def test_calculate_gives_orca_shift_plus_orbital(qapp, tmp_path, no_dialogs):
    plots = _Plots()
    p = _dft_panel(tmp_path, plots)
    p.dft_opm_orbital_spin.setValue(3.0)
    assert p.calculate(simulate=False), no_dialogs
    assert p.model["atoms"][0]["iso"] == pytest.approx(-MN_HFC[35][5] + 3.0, abs=0.02)
    assert p.plot_contributions() and plots.drawn[-1][0] == "curve"
    ns = {}
    exec(p.dft_python(), ns)
    assert ns["model"]["atoms"][0]["iso"] == pytest.approx(p.model["atoms"][0]["iso"])


def test_model_change_recalculates(qapp, tmp_path, no_dialogs):
    p = _dft_panel(tmp_path)
    assert p.calculate(simulate=False)
    before = p.model["atoms"][0]["iso"]
    p.dft_opm_model_combo.setCurrentIndex(p.dft_opm_model_combo.findData("curie_weiss"))
    p.dft_opm_theta_spin.setValue(-100.0)
    p.dft_opm_theta_spin.editingFinished.emit()
    assert p.model["atoms"][0]["iso"] == pytest.approx(before * 298.0 / 398.0, rel=1e-6)


def test_state_round_trip_keeps_the_chosen_spin(qapp, tmp_path, no_dialogs):
    p = _dft_panel(tmp_path)
    p.dft_opm_S_spin.setValue(2.0)
    p.dft_opm_model_combo.setCurrentIndex(p.dft_opm_model_combo.findData("mu_eff"))
    state = p.get_state()
    q = DftSimulationPanel(plot_spectrum=lambda s: None, add_to_open=lambda s: None)
    q.set_state(state)
    assert q.dft_opm_S_spin.value() == 2.0
    assert q.dft_opm_model_combo.currentData() == "mu_eff"
    assert q.dft_site_list.count() == 1


def test_paramagnetic_source_for_the_tools(qapp, tmp_path, no_dialogs):
    p = _dft_panel(tmp_path)
    src = p.paramagnetic_source()
    assert src["labels"] == ["Na1"] and src["args"]["model"] == "curie"


# ---- the tools panel ----------------------------------------------------------------

def test_shift_against_temperature(qapp, tmp_path, no_dialogs):
    plots = _Plots()
    p = _dft_panel(tmp_path)
    t = ParamagneticToolsPanel(draw_plot=plots, dft_source=p.paramagnetic_source)
    t.pm_T_points_spin.setValue(5)
    assert t.calculate_temperature(), no_dialogs
    d = t.temperature_curves["sites"]["Na1"]["total"] * t.temperature_curves["T"]
    assert np.allclose(d, d[0])
    assert t.paste_temperature("250\t400\n300\t333.333333\n350\t285.714286\n") == 3
    assert t.fit_temperature(), no_dialogs
    assert t.temperature_fit["params"]["a"] == pytest.approx(1e5, rel=1e-4)
    assert "R²" in t.pm_T_result_label.text()
    ns = {}
    exec(t.temperature_python(), ns)
    assert ns["fit"]["params"]["a"] == pytest.approx(t.temperature_fit["params"]["a"])
    assert len(plots.drawn) >= 2


def test_temperature_without_a_dft_source_says_so(qapp, no_dialogs):
    t = ParamagneticToolsPanel()
    assert not t.calculate_temperature()
    assert "ORCA" in no_dialogs[-1]


def test_only_temperature_bms_and_pathways_remain(qapp):
    from opennmr.gui.widgets.collapsible_section import CollapsibleSection
    t = ParamagneticToolsPanel()
    titles = [sec.title() if callable(getattr(sec, "title", None)) else sec.title
              for sec in t.findChildren(CollapsibleSection)]
    assert t.pm_temperature_section is not None and t.pm_bms_section is not None
    assert t.pm_pathway_section is not None
    assert not hasattr(t, "pm_ln_table") and not hasattr(t, "pm_pcs_table")
    assert len(titles) == 3


@pytest.fixture
def bms_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENNMR_BMS_CACHE", str(tmp_path / "sidebands"))


def _bms_panel(plots=None, spectra=None):
    t = ParamagneticToolsPanel(draw_plot=plots,
                               plot_spectrum=spectra.append if spectra is not None else None)
    t.pm_bms_n_spin.setValue(15)
    t.pm_bms_box_spin.setValue(200.0)
    t.pm_bms_size_edit.setText("20")
    t.pm_bms_spread_edit.setText("2, 2, 2")
    t.pm_bms_grid_combo.setCurrentIndex(t.pm_bms_grid_combo.findData(64))
    t.pm_bms_chi_edit.setText("2.87e-3, 3.13e-3, 2.27e-3")
    return t


def test_bms_static_spectrum_and_python(qapp, no_dialogs, bms_cache):
    plots, spectra = _Plots(), []
    t = _bms_panel(plots, spectra)
    assert t.simulate_bms(), no_dialogs
    s = spectra[-1]
    sim = s.metadata.raw["simulation"]
    assert sim["kind"] == "bms" and sim["mode"] == "static" and sim["n_particles"] == 15
    assert s.metadata.nucleus == "7Li" and len(s.data) == len(s.axes[0].ppm)
    text = t.pm_bms_result_label.text()
    assert "15 particles" in text and "standard deviation" in text
    ns = {}
    exec(t.bms_python().replace("fig = ", "_fig = "), ns)
    assert np.allclose(ns["spectrum"]["real_spectrum"], t.bms_spectrum["real_spectrum"])
    assert t.plot_bms_slice() and plots.drawn[-1][0] == "curve"


def test_bms_packing_is_kept_until_its_settings_change(qapp, no_dialogs, bms_cache):
    t = _bms_panel()
    assert t.generate_bms_packing(), no_dialogs
    first = t.bms_particles
    assert "volume fraction" in t.pm_bms_result_label.text()
    assert t.simulate_bms() and t.bms_particles is first        # same packing re-used
    t.pm_bms_seed_spin.setValue(t.pm_bms_seed_spin.value() + 1)
    assert t.simulate_bms() and t.bms_particles is not first    # a new one for a new seed


def test_bms_mas_modes(qapp, no_dialogs, bms_cache):
    spectra = []
    t = _bms_panel(spectra=spectra)
    t.pm_bms_mode_combo.setCurrentIndex(t.pm_bms_mode_combo.findData("mas"))
    assert t._bms_spec_form.isRowVisible(t.pm_bms_rate_spin)
    t.pm_bms_rate_spin.setValue(20000.0)
    t.pm_bms_order_spin.setValue(40)
    assert t.simulate_bms(), no_dialogs
    assert "delta_tensor" in t.bms_field
    mean_mas = t.bms_spectrum["stats"]["mean_ppm"]
    t.pm_bms_mode_combo.setCurrentIndex(t.pm_bms_mode_combo.findData("mas_infinite"))
    assert not t._bms_spec_form.isRowVisible(t.pm_bms_rate_spin)
    assert t.simulate_bms(), no_dialogs
    assert t.bms_spectrum["stats"]["mean_ppm"] == pytest.approx(mean_mas, abs=1e-3)
    assert len(spectra) == 2


def test_bms_from_a_dump_file(qapp, tmp_path, no_dialogs, bms_cache):
    path = tmp_path / "pack.dump"
    path.write_text("ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n2\n"
                    "ITEM: BOX BOUNDS pp pp pp\n-50 50\n-50 50\n-50 50\n"
                    "ITEM: ATOMS id x y z v1 v2 v3 qi qj qk qr\n"
                    "1 -20 0 0 12 10 8 0 0 0 1\n2 22 5 -3 10 10 10 0.5 0.5 0.5 0.5\n")
    t = _bms_panel()
    t.pm_bms_source_combo.setCurrentIndex(t.pm_bms_source_combo.findData("dump"))
    assert not t._bms_particle_form.isRowVisible(t.pm_bms_n_spin)
    t.pm_bms_box_spin.setValue(100.0)
    t.pm_bms_grid_combo.setCurrentIndex(t.pm_bms_grid_combo.findData(64))
    assert not t.simulate_bms(quiet=True)                       # no file chosen yet
    t.pm_bms_dump_edit.setText(str(path))
    assert t.simulate_bms(), no_dialogs
    assert t.bms_field["n_particles"] == 2
    assert repr(str(path)) in t.bms_python()


def test_bms_bad_input_says_so(qapp, no_dialogs, bms_cache):
    t = _bms_panel()
    t.pm_bms_chi_edit.setText("1e-3, 2e-3")
    assert not t.simulate_bms()
    assert "χ" in no_dialogs[-1]
    t.pm_bms_chi_edit.setText("1e-3")
    t.pm_bms_size_edit.setText("20, abc")
    assert not t.simulate_bms()
    assert "Mean size" in no_dialogs[-1]


def test_bms_state_round_trip(qapp, no_dialogs):
    t = _bms_panel()
    t.pm_bms_mode_combo.setCurrentIndex(t.pm_bms_mode_combo.findData("mas"))
    t.pm_bms_observe_combo.setCurrentIndex(t.pm_bms_observe_combo.findData("outside"))
    t.pm_bms_boundary_combo.setCurrentIndex(t.pm_bms_boundary_combo.findData("isolated"))
    u = ParamagneticToolsPanel()
    u.set_state(t.get_state())
    assert u.pm_bms_chi_edit.text() == "2.87e-3, 3.13e-3, 2.27e-3"
    assert u.pm_bms_mode_combo.currentData() == "mas"
    assert u.pm_bms_observe_combo.currentData() == "outside"
    assert not u._bms_grid_form.isRowVisible(u.pm_bms_sample_combo)   # periodic only


def test_bond_pathways_and_send(qapp, monkeypatch, no_dialogs):
    plots = _Plots()
    ref = _calc([1] * 8, ["Mn", "Fe"] * 4, "fm.out")
    flips = {f"flip{j}.out": _calc([1 if i != j else -1 for i in range(8)], ["Mn", "Fe"] * 4,
                                   f"flip{j}.out") for j in range(2)}
    table = {"fm.out": ref, **flips}
    monkeypatch.setattr(bp, "_read", lambda c: table[c] if isinstance(c, str) else c)
    sent = []
    t = ParamagneticToolsPanel(draw_plot=plots,
                               send_pathways=lambda b, d: sent.append((b, d)) or ["note"])
    t.pm_bp_reference_edit.setText("fm.out")
    t._add_pathway_files(list(flips))
    t.pm_bp_shell_table.setRowCount(0)
    t.pm_bp_shell_table.insertRow(0)
    for c, text in enumerate(BINS[0]):
        t.pm_bp_shell_table.setItem(0, c, QTableWidgetItem(str(text)))
    t.pm_bp_route_combo.setCurrentIndex(t.pm_bp_route_combo.findData("phi"))
    t.pm_bp_phi_spin.setValue(SCALING["phi"])
    t.pm_bp_B0_spin.setValue(SCALING["B0"])
    assert t.calculate_pathways(), no_dialogs
    assert t.pathway_result["shift_dict"]["NN"]["Mn"] == pytest.approx(10.0)
    assert t.pathway_result["shift_dict"]["NN"]["Fe"] == pytest.approx(-4.0)
    assert t.pm_bp_result_table.rowCount() == 2
    assert t.send_pathways() and sent[0][1]["NN"]["Mn"] == pytest.approx(10.0)
    compile(t.pathways_python(), "<bp>", "exec")


def test_pnmr_panel_needs_its_structure_before_receiving_pathways(qapp):
    from opennmr.gui.main_window import MainWindow
    window = MainWindow()
    with pytest.raises(ValueError, match="Initialise"):
        window.pnmr_panel.set_shift_contributions([("NN", 0, 3)], {"NN": {"Mn": 1.0}})
    # the tools panel is wired to the DFT panel and to pNMR (CIF)
    assert window.paramagnetic_panel._dft_source() is None


def test_bms_runs_are_logged_and_rerun(qapp, monkeypatch, tmp_path):
    from opennmr.gui.main_window import MainWindow
    monkeypatch.setenv("OPENNMR_BMS_CACHE", str(tmp_path / "sidebands"))
    window = MainWindow()
    sections = window._simulation_panels()
    assert "bms" in sections and window._SIMULATION_LABELS["bms"] == "BMS shifts (fast)"
    logged = []
    monkeypatch.setattr(window, "_journal_simulation", lambda key, s=None: logged.append(key))
    t = window.paramagnetic_panel
    t.pm_bms_n_spin.setValue(10)
    t.pm_bms_grid_combo.setCurrentIndex(t.pm_bms_grid_combo.findData(64))
    assert t.simulate_bms(quiet=True)
    assert logged == ["bms"]
    assert window.fit_canvas.simulation_spectrum.metadata.raw["simulation"]["kind"] == "bms"
    assert "particles" in sections["bms"].summary()
