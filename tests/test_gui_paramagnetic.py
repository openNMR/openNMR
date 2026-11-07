"""GUI for the paramagnetic NMR features (user request, 2026-11-06):
Simulate ▸ DFT tensors with an ORCA EPR output (g, ZFS, hyperfine), and
Simulate ▸ Paramagnetic NMR tools: shift vs T, BMS shifts (2026-11-07,
which replaced the lanthanide-series and pseudocontact sections), bond
pathways. Second round (2026-11-07): shift vs T from CRYSTAL or typed-in
shifts; ion identification and neighbour distances for the pathways; the
BMS sample (rotor, plate), local interactions, views and time checks."""

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
    assert t.plot_bms_slice() and plots.drawn[-1][0] == "2d"


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
    assert u.pm_bms_sample_combo.currentData() == t.pm_bms_sample_combo.currentData()


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
    assert "bms" in sections and window._SIMULATION_LABELS["bms"] == "BMS shifts"
    logged = []
    monkeypatch.setattr(window, "_journal_simulation", lambda key, s=None: logged.append(key))
    t = window.paramagnetic_panel
    t.pm_bms_n_spin.setValue(10)
    t.pm_bms_grid_combo.setCurrentIndex(t.pm_bms_grid_combo.findData(64))
    assert t.simulate_bms(quiet=True)
    assert logged == ["bms"]
    assert window.fit_canvas.simulation_spectrum.metadata.raw["simulation"]["kind"] == "bms"
    assert "particles" in sections["bms"].summary()


# ---- second round (2026-11-07) ----------------------------------------------------------

def test_temperature_from_typed_in_shifts(qapp, no_dialogs):
    plots = _Plots()
    t = ParamagneticToolsPanel(draw_plot=plots)
    t.pm_T_source_combo.setCurrentIndex(t.pm_T_source_combo.findData("typed"))
    assert not t._T_typed_box.isHidden()
    assert t.paste_calculated("Na1\t3\t100\t20\nNa2, 1, -50, 0\n") == 2
    t.pm_T_ref_spin.setValue(300.0)
    t.pm_T_scaling_combo.setCurrentIndex(t.pm_T_scaling_combo.findData("curie_bleaney"))
    t.pm_T_min_spin.setValue(150.0)
    t.pm_T_max_spin.setValue(300.0)
    t.pm_T_points_spin.setValue(2)
    assert t.calculate_temperature(), no_dialogs
    na1 = t.temperature_curves["sites"]["Na1"]
    assert na1["total"][0] == pytest.approx(3 + 200 + 80)      # 1/T and 1/T² from 300 to 150 K
    assert na1["total"][1] == pytest.approx(123)
    assert "typed-in" in t.pm_T_source_label.text()
    ns = {}
    exec(t.temperature_python(), ns)
    assert np.allclose(ns["curves"]["sites"]["Na1"]["total"], na1["total"])
    assert plots.drawn


def test_temperature_from_a_crystal_calculation(qapp, no_dialogs):
    calc = _calc([1] * 8, tag="fm.out")
    args = {"theta": -50.0, "mu_eff": 5.9, "spin": 2.5, "T": 300.0, "B0": 9.4, "g": 2.0,
            "components": "both"}
    src = {"kind": "crystal", "calc": calc, "path": "fm.out", "args": args, "element": "Na",
           "labels": ["Na1"], "orbital": 2.0}
    t = ParamagneticToolsPanel(draw_plot=_Plots(), dft_source=lambda: src)
    t.pm_T_min_spin.setValue(250.0)
    t.pm_T_max_spin.setValue(350.0)
    t.pm_T_points_spin.setValue(3)
    assert t.calculate_temperature(), no_dialogs
    s = t.temperature_curves["sites"]["Na1"]["total"] - 2.0
    assert s[0] / s[1] == pytest.approx((300 + 50) / (250 + 50))
    assert "CRYSTAL" in t.pm_T_source_label.text()


def test_bond_pathway_ions_shown_and_overridden(qapp, monkeypatch, no_dialogs):
    ref = _calc([1] * 8, tag="fm.out")
    flips = {f"flip{j}.out": _calc([1 if i != j else -1 for i in range(8)], tag=f"flip{j}.out")
             for j in (2, 5)}
    table = {"fm.out": ref, **flips}
    monkeypatch.setattr(bp, "_read", lambda c: table[c] if isinstance(c, str) else c)
    t = ParamagneticToolsPanel(draw_plot=_Plots())
    t.pm_bp_reference_edit.setText("fm.out")
    t.pm_bp_magnetic_edit.setText("Mn")
    t._add_pathway_files(list(flips))
    assert t.identify_pathway_ions(), no_dialogs
    ions = [t.pm_bp_ion_table.item(r, 1).text() for r in range(2)]
    assert ions == ["Mn3", "Mn6"]
    assert "changed sign" in t.pm_bp_ion_table.item(0, 2).text()
    from PySide6.QtCore import Qt
    editable = Qt.ItemFlag.ItemIsEditable
    assert t.pm_bp_ion_table.item(0, 1).flags() & editable          # only the Ion column
    assert not t.pm_bp_ion_table.item(0, 2).flags() & editable
    # an override is used and kept
    t.pm_bp_ion_table.item(1, 1).setText("Mn7")
    assert t.pathway_flipped() == ["Mn3", "Mn7"]
    assert t.identify_pathway_ions()
    assert t.pm_bp_ion_table.item(1, 1).text() == "Mn7"
    assert t.pm_bp_ion_table.item(1, 2).text() == "given"
    state = t.get_state()
    u = ParamagneticToolsPanel()
    u.set_state(state)
    assert u.pathway_flipped() == ["Mn3", "Mn7"]


def test_bond_pathway_neighbour_distances_and_shells(qapp, monkeypatch, no_dialogs):
    ref = _calc([1] * 8, tag="fm.out")
    monkeypatch.setattr(bp, "_read", lambda c: {"fm.out": ref}[c] if isinstance(c, str) else c)
    plots = _Plots()
    t = ParamagneticToolsPanel(draw_plot=plots)
    t.pm_bp_reference_edit.setText("fm.out")
    t.pm_bp_magnetic_edit.setText("Mn")
    assert t.plot_neighbour_distances(), no_dialogs
    assert plots.drawn[-1][0] == "curve"
    assert t.suggest_pathway_shells(), no_dialogs
    bins = t.pathway_shells()
    assert bins[0][0] == "NN" and bins[0][1] < 3.464 < bins[0][2]
    t.pm_bp_reference_edit.setText("")
    t.pm_bp_mode_combo.setCurrentIndex(t.pm_bp_mode_combo.findData("single"))
    assert not t.plot_neighbour_distances()
    assert "Add the calculations" in no_dialogs[-1]


def test_bms_views_switch_without_simulating_again(qapp, no_dialogs, bms_cache):
    plots, logged, shown = _Plots(), [], []
    t = ParamagneticToolsPanel(draw_plot=plots, plot_spectrum=logged.append,
                               show_spectrum=shown.append)
    t.pm_bms_n_spin.setValue(12)
    t.pm_bms_box_spin.setValue(200.0)
    t.pm_bms_size_edit.setText("20")
    t.pm_bms_grid_combo.setCurrentIndex(t.pm_bms_grid_combo.findData(64))
    assert t.show_bms_view("packing"), no_dialogs          # the packing needs no simulation
    assert plots.drawn[-1][0] == "tensor" and not logged
    assert t.simulate_bms(), no_dialogs
    assert len(logged) == 1
    assert t.show_bms_view("slice") and plots.drawn[-1][0] == "2d"
    assert not t._bms_slice_row.isHidden()
    t.pm_bms_plane_combo.setCurrentIndex(t.pm_bms_plane_combo.findData("z"))
    assert plots.drawn[-1][0] == "2d"                         # redrawn for the new plane
    assert t.show_bms_view("spectrum")
    assert shown and shown[-1] is t.bms_nmr_spectrum and len(logged) == 1


def test_bms_field_is_kept_when_only_the_spectrum_changes(qapp, no_dialogs, bms_cache):
    t = _bms_panel()
    assert t.simulate_bms(), no_dialogs
    field = t.bms_field
    t.pm_bms_lw_spin.setValue(500.0)
    t.pm_bms_sample_combo.setCurrentIndex(t.pm_bms_sample_combo.findData("rotor"))
    t.pm_bms_samples_spin.setValue(5000)
    assert t.simulate_bms(), no_dialogs
    assert t.bms_field is field and "kept" in t.pm_bms_result_label.text()
    t.pm_bms_chi_edit.setText("1e-3")
    assert t.simulate_bms(), no_dialogs
    assert t.bms_field is not field


def test_bms_rotor_sample_and_local_interactions(qapp, no_dialogs, bms_cache):
    from opennmr.notebook.bms import sample as S
    t = _bms_panel()
    t.pm_bms_sample_combo.setCurrentIndex(t.pm_bms_sample_combo.findData("rotor"))
    t.pm_bms_rotor_combo.setCurrentIndex(t.pm_bms_rotor_combo.findData(1.3))
    t._rotor_chosen()
    assert (t.pm_bms_inner_spin.value(), t.pm_bms_length_spin.value()) == S.ROTORS[1.3]
    assert "µL" in t.pm_bms_sample_note.text()
    t.pm_bms_inner_spin.setValue(1.5)
    assert not t.simulate_bms()                              # inner ≥ outer diameter
    assert "inner diameter" in no_dialogs[-1]
    t.pm_bms_inner_spin.setValue(0.8)
    t.pm_bms_mode_combo.setCurrentIndex(t.pm_bms_mode_combo.findData("mas"))
    t.pm_bms_local_group.setChecked(True)
    t.pm_bms_contact_spin.setValue(100.0)
    t.pm_bms_cq_spin.setValue(0.05)
    assert t._bms_local_form.isRowVisible(t.pm_bms_transitions_combo)
    t.pm_bms_samples_spin.setValue(5000)
    assert t._bms_spec_form.isRowVisible(t.pm_bms_samples_spin)
    assert t.simulate_bms(), no_dialogs
    assert t.bms_spectrum["route"] == "sampled"
    assert t.bms_spectrum["stats"]["mean_ppm"] == pytest.approx(100.0, abs=10.0)
    text = t.pm_bms_result_label.text()
    assert "cylinder 0.8" in text and "C_Q 0.05" in text and "Monte Carlo noise" in text
    ns = {}
    exec(t.bms_python().replace("fig = ", "_f = ").replace("fig3d = ", "_g = "), ns)
    assert np.allclose(ns["spectrum"]["real_spectrum"], t.bms_spectrum["real_spectrum"])


def test_bms_asks_before_a_long_run(qapp, monkeypatch, no_dialogs, bms_cache):
    from opennmr.gui import long_run
    from opennmr.notebook.bms import views
    asked = []
    monkeypatch.setattr(long_run, "enabled", lambda: True)
    monkeypatch.setattr(long_run, "ask", lambda *a: asked.append(a[1:4]) or False)
    monkeypatch.setattr(views, "estimate_field_time",
                        lambda *a, **k: {"seconds": 600.0, "memory_GB": 0.1, "text": "about 10 min"})
    t = _bms_panel()
    assert not t.simulate_bms()                              # cancelled at the question
    assert asked and asked[0][0] == "BMS field" and asked[0][2] == "about 10 min"
    assert t.bms_spectrum is None
    monkeypatch.setattr(views, "estimate_field_time",
                        lambda *a, **k: {"seconds": 0.1, "memory_GB": 0.1, "text": "0.1 s"})
    monkeypatch.setattr(views, "estimate_sampling_time",
                        lambda *a, **k: {"seconds": 900.0, "text": "about 15 min"})
    assert not t.simulate_bms()                              # static: the sampled spectrum
    assert asked[-1][0] == "BMS spectrum" and asked[-1][2] == "about 15 min"


def test_bms_rotor_field_controls_and_keep(qapp, no_dialogs, bms_cache):
    kept = []
    t = ParamagneticToolsPanel(plot_spectrum=lambda s: None,
                               keep_spectrum=lambda s: kept.append(s) or 7)
    t.pm_bms_n_spin.setValue(12)
    t.pm_bms_box_spin.setValue(200.0)
    t.pm_bms_size_edit.setText("20")
    t.pm_bms_grid_combo.setCurrentIndex(t.pm_bms_grid_combo.findData(64))
    t.pm_bms_samples_spin.setValue(20000)
    assert t.pm_bms_sample_combo.currentData() == "rotor" and t.pm_bms_rotor_check.isChecked()
    assert t._bms_sample_form.isRowVisible(t.pm_bms_rotor_length_spin)
    assert not t.pm_bms_keep_button.isEnabled() and t.keep_bms_spectrum() is None
    sample = t.bms_sample()
    assert sample.rotor["chi_wall"] == pytest.approx(-8.5e-6)
    assert sample.rotor["chi_caps"] == pytest.approx(-9.02e-6)
    t.pm_bms_wall_chi_spin.setValue(-5.0)                       # typed: Custom
    assert t.pm_bms_wall_combo.currentData() == "custom"
    assert t.bms_sample().rotor["chi_wall"] == pytest.approx(-5e-6)
    assert t.simulate_bms(), no_dialogs
    assert "in a rotor 3.2" in t.pm_bms_result_label.text()
    assert t.pm_bms_keep_button.isEnabled()
    assert t.keep_bms_spectrum() == 7 and kept[-1] is t.bms_nmr_spectrum
    t.pm_bms_rotor_check.setChecked(False)
    assert not t._bms_sample_form.isRowVisible(t.pm_bms_rotor_length_spin)
    assert t.bms_sample().rotor is None
    t.pm_bms_rotor_combo.setCurrentIndex(t.pm_bms_rotor_combo.findData(4.0))
    t._rotor_chosen()
    assert t.pm_bms_rotor_length_spin.value() == pytest.approx(10.0 + 8.0)


def test_kept_bms_spectrum_lands_in_open_experiments(qapp, monkeypatch, tmp_path):
    from opennmr.gui.main_window import MainWindow
    monkeypatch.setenv("OPENNMR_BMS_CACHE", str(tmp_path / "sidebands"))
    window = MainWindow()
    t = window.paramagnetic_panel
    t.pm_bms_n_spin.setValue(10)
    t.pm_bms_grid_combo.setCurrentIndex(t.pm_bms_grid_combo.findData(64))
    t.pm_bms_sample_combo.setCurrentIndex(t.pm_bms_sample_combo.findData("sphere"))
    t.pm_bms_samples_spin.setValue(20000)
    before = len(window.session.spectra)
    assert t.simulate_bms(quiet=True)
    index = t.keep_bms_spectrum()
    assert index is not None and len(window.session.spectra) == before + 1
    assert t.keep_bms_spectrum() == index and len(window.session.spectra) == before + 1   # once


def test_bms_susceptibility_units(qapp, no_dialogs, bms_cache):
    from opennmr.notebook import susceptibility as X
    t = _bms_panel()
    g = t._bms_grid_form
    assert not g.isRowVisible(t.pm_bms_density_spin)                 # SI volume: nothing else
    t.pm_bms_chi_unit_combo.setCurrentIndex(t.pm_bms_chi_unit_combo.findData("mu_eff"))
    assert g.isRowVisible(t.pm_bms_chi_T_spin) and g.isRowVisible(t.pm_bms_molar_mass_spin)
    t.pm_bms_chi_edit.setText("5.92")
    assert "molar volume" in t.pm_bms_chi_si_label.text()          # says what is missing
    t.pm_bms_molar_mass_spin.setValue(157.85)
    t.pm_bms_density_spin.setValue(3.43)
    t.pm_bms_chi_T_spin.setValue(300.0)
    expect = X.to_si_volume(5.92, "mu_eff", molar_mass=157.85, density=3.43, T=300.0)
    assert t.bms_chi_si() == pytest.approx((expect,) * 3)
    assert f"{expect:.4g}" in t.pm_bms_chi_si_label.text()
    t.pm_bms_vm_combo.setCurrentIndex(t.pm_bms_vm_combo.findData("cell"))
    assert g.isRowVisible(t.pm_bms_Z_spin) and not g.isRowVisible(t.pm_bms_molar_mass_spin)
    t.pm_bms_cell_spin.setValue(157.85 / 3.43 * 4 / X.N_A * 1e24)
    t.pm_bms_Z_spin.setValue(4)
    assert t.bms_chi_si()[0] == pytest.approx(expect, rel=1e-6)
    t.pm_bms_samples_spin.setValue(20000)
    assert t.simulate_bms(), no_dialogs
    assert "μ_eff" in t.pm_bms_result_label.text()
    assert t.bms_nmr_spectrum.metadata.raw["simulation"]["chi_unit"] == "mu_eff"
    ns = {}
    exec(t.bms_python().replace("fig = ", "_f = ").replace("fig3d = ", "_g = "), ns)
    assert np.allclose(ns["spectrum"]["real_spectrum"], t.bms_spectrum["real_spectrum"])
    u = ParamagneticToolsPanel()
    u.set_state(t.get_state())
    assert u.bms_chi_si()[0] == pytest.approx(t.bms_chi_si()[0])
    assert u._bms_grid_form.isRowVisible(u.pm_bms_Z_spin)
