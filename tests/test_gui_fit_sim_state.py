"""Fits and simulations are logged and restorable: a fit is kept with the
spectrum it was run on (inputs, exact result, report) and comes back on
reselection, reopening and Load Progress; every simulation run is
journalled and can be restored into its panel."""

import json
import os

import numpy as np
import pytest
from PySide6.QtWidgets import QMessageBox

from opennmr import io
from opennmr.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
from opennmr.gui import panel_state
from opennmr.gui import progress as progress_dialogs
from opennmr.gui.main_window import MainWindow
from opennmr.io import history as io_history
from opennmr.io import serialize
from opennmr.notebook import csa_fitting as C
from opennmr.notebook import simulations as S

from test_gui import qapp, _settle_sidecar_save  # noqa: F401 -- qapp is a fixture

NU0 = 130.3e6
L_1H = 400.0e6


@pytest.fixture(autouse=True)
def _dialogs(monkeypatch):
    shown = []
    for name in ("warning", "information", "critical"):
        monkeypatch.setattr(QMessageBox, name,
                            lambda *a, _n=name, **k: shown.append((_n, a[2] if len(a) > 2 else "")))
    return shown


def _quad_spectrum(window, name="27Al synthetic"):
    x = np.linspace(60, -80, 2048)
    y = 1000 * S.quadrupolar_spectrum(2.5, 4.2e6, 0.35, NU0, x[::-1], transitions="ct",
                                      delta_iso=12.0, fwhm=0.6, n_zcw=6000)["normalised_real_spec"][::-1]
    y += np.random.default_rng(0).standard_normal(x.size) * 3
    spectrum = NMRSpectrum(data=y, axes=[NMRAxis(kind="frequency", ppm=x)],
                           metadata=NMRMetadata(vendor="simulated", nucleus="27Al", sfo1=NU0 / 1e6))
    return window._add_derived_spectrum(spectrum, name)


def _j_spectrum(window, name="1H synthetic"):
    x = np.linspace(6.0, 0.0, 16384)
    freqs, w = S.combine_jcouplings([{"nuclide": "1H", "n": 3, "J_Hz": 7.1, "abundance": 100.0}])
    sticks = 1.2 + freqs / L_1H * 1e6
    y = 500 * S.multiplet_lineshape(x, sticks, w, 1.0 / L_1H * 1e6, normalise="peak")
    spectrum = NMRSpectrum(data=y, axes=[NMRAxis(kind="frequency", ppm=x)],
                           metadata=NMRMetadata(vendor="simulated", nucleus="1H", sfo1=L_1H / 1e6))
    return window._add_derived_spectrum(spectrum, name)


def _fit_quad(window):
    panel = window.quad_fit_panel
    panel.orientations_spin.setValue(1500)
    panel.render_orientations_spin.setValue(3000)
    panel.estimate()
    panel.fit()
    return panel


def _table(table):
    return panel_state.table_text(table)


# ------------------------------------------------------------------
# Exact encoding of results
# ------------------------------------------------------------------

def test_serialize_round_trips_arrays_tuples_and_special_floats():
    value = {"a": np.linspace(0, 1, 7), "b": (1, 2.5, "x"), "c": [np.float32(1.5), np.int64(3)],
             "d": {1: "one"}, "e": complex(1, -2), "f": float("nan"), "g": np.eye(2, dtype=bool),
             "h": object()}
    back = serialize.decode(json.loads(json.dumps(serialize.encode(value))))
    np.testing.assert_array_equal(back["a"], value["a"])
    assert back["b"] == (1, 2.5, "x") and back["c"] == [1.5, 3] and back["d"] == {1: "one"}
    assert back["e"] == complex(1, -2) and np.isnan(back["f"])
    assert back["g"].dtype == bool and back["h"] is None
    assert serialize.unsupported_keys(value) == ["h"]


# ------------------------------------------------------------------
# Fits: kept with their spectrum, logged, and shown again
# ------------------------------------------------------------------

def test_a_quadrupolar_fit_is_stored_with_its_spectrum_exactly(qapp, _dialogs):
    window = MainWindow()
    index = _quad_spectrum(window)
    panel = _fit_quad(window)
    assert not _dialogs, _dialogs
    saved = window.session.spectra[index].metadata.raw["fits"]["quadrupolar"]
    assert saved["label"] == "Quadrupolar" and "27Al synthetic" in saved["target"]
    result = {k: v for k, v in panel.result.items() if k != "spectrum"}
    assert serialize.unsupported_keys(result) == []
    back = serialize.decode(saved["result"])
    np.testing.assert_array_equal(back["popt"], panel.result["popt"])
    np.testing.assert_array_equal(back["y_fit"], panel.result["y_fit"])
    assert "R²" in saved["report"] and "δiso" in saved["report"]
    assert any(line.startswith("Nucleus: 27Al") for line in saved["inputs_text"])


def test_reselecting_a_spectrum_shows_its_own_fit_and_not_another(qapp, _dialogs):
    window = MainWindow()
    first = _quad_spectrum(window, "first")
    panel = _fit_quad(window)
    table, label = _table(panel.results_table), panel.results_label.text()
    sites = _table(panel.sites_table)

    blank = MainWindow().quad_fit_panel
    second = _quad_spectrum(window, "second")  # selected; never fitted
    assert panel.result is None and panel.results_label.text() == ""
    # A new dataset starts from a blank fit, not the last one's inputs
    # (user request, 2026-11-13).
    assert _table(panel.sites_table) != sites
    assert _table(panel.sites_table) == _table(blank.sites_table)
    panel.add_site()
    panel.sites_table.item(0, 2).setText("9.9")
    draft = _table(panel.sites_table)

    window._select_single_row(first)
    assert _table(panel.results_table) == table and panel.results_label.text() == label
    assert _table(panel.sites_table) == sites  # the inputs it was fitted with
    assert panel.send_button.isEnabled()

    # The unfitted inputs set up for the second come back with it.
    window._select_single_row(second)
    assert panel.result is None and _table(panel.sites_table) == draft


def test_a_fit_is_logged_with_its_result_and_settings(qapp, _dialogs):
    window = MainWindow()
    index = _quad_spectrum(window)
    _fit_quad(window)
    _settle_sidecar_save(window)
    text = io_history.render_log(window.session.info(index)["history"])
    assert "Quadrupolar fit of 27Al synthetic" in text
    assert "Result:" in text and "R²" in text
    assert "Settings (sites are the starting values):" in text and "Nucleus: 27Al" in text


def test_j_coupling_first_order_and_spin_system_fits_come_back(qapp, _dialogs):
    window = MainWindow()
    fit = window.jcoupling_fit_panel
    # First order: the panel's restorable mode (it opens on spin systems).
    fit.model_combo.setCurrentIndex(fit.model_combo.findData("first_order"))
    index = _j_spectrum(window)
    fit.editor.clear()
    fit.editor.add_site(label="CH3", shift=1.201, fwhm_hz=1.2,
                        couplings=[{"nuclide": "1H", "n": 3, "J_Hz": 7.0, "abundance": 100.0}])
    assert fit.fit()
    first_order = (_table(fit.results_table), fit.results_label.text())
    result = {k: v for k, v in fit.result.items() if k != "spectrum"}
    assert serialize.unsupported_keys(result) == []

    _j_spectrum(window, "other")
    assert fit.result is None
    window._select_single_row(index)
    assert (_table(fit.results_table), fit.results_label.text()) == first_order
    assert fit.use_fitted_button.isEnabled()
    fit.use_fitted_as_start()  # works from a restored result
    assert fit.editor.sites(L_1H)[0]["couplings"][0]["J_Hz"] == pytest.approx(7.1, abs=1e-2)


def test_a_csa_fit_comes_back_on_reselection(qapp):
    window = MainWindow()
    x = np.linspace(-300.0, 300.0, 1024)
    y = C.csa_model_full(x, 0.0, -100.0, 0.5, 2.0, convention="Haeberlen", static=False,
                         larmor_Hz=1e8, mas_Hz=5000.0, powder_scheme="zcw", n_zcw=300)
    spectrum = NMRSpectrum(data=(1e10 * y)[::-1], axes=[NMRAxis(kind="frequency", ppm=x[::-1])],
                           metadata=NMRMetadata(vendor="bruker", nucleus="13C", sfo1=100.0,
                                                raw={"acqus": {"MASR": 5000.0}}))
    index = window._add_derived_spectrum(spectrum, "csa")
    window.csa_static_combo.setCurrentIndex(window.csa_static_combo.findData(False))
    window.csa_orientations_spin.setValue(200)
    window.csa_render_orientations_spin.setValue(300)
    window._add_csa_site(("A", "2", "-80", "0.4", "3"))
    window._on_csa_fit_clicked()
    shown = (_table(window.csa_results_table), window.csa_results_label.text())
    assert "CSA fit report" in shown[1]
    assert window.session.spectra[index].metadata.raw["fits"]["csa"]["label"] == "CSA"

    window._add_derived_spectrum(NMRSpectrum(
        data=np.zeros(16), axes=[NMRAxis(kind="frequency", ppm=np.linspace(1, 0, 16))],
        metadata=NMRMetadata(vendor="simulated", nucleus="13C")), "blank")
    assert window._csa_fit_result is None and not window.csa_results_table.isVisible()
    window._select_single_row(index)
    assert (_table(window.csa_results_table), window.csa_results_label.text()) == shown
    assert window._csa_fit_result["spectrum_index"] == index


def test_fits_survive_saving_and_reopening_and_follow_load_progress(qapp, tmp_path, _dialogs):
    window = MainWindow()
    index = _quad_spectrum(window)
    panel = _fit_quad(window)
    first = (_table(panel.results_table), panel.results_label.text())
    folder = str(tmp_path / "quad")
    window._save_progress(index, name="fitted", folder=folder)

    # A different fit, then back to the save point.
    panel.sites_table.item(0, 3).setText("0.9")
    panel.fit()
    assert (_table(panel.results_table), panel.results_label.text()) != first
    window._restore_progress(index, "fitted")
    assert (_table(panel.results_table), panel.results_label.text()) == first

    reopened = MainWindow()
    reopened._load(folder, vendor=io.detect_vendor(folder))
    again = reopened.quad_fit_panel
    assert (_table(again.results_table), again.results_label.text()) == first

    # ...and a notebook gets the same fit, exactly.
    from opennmr import notebook as nb
    fits = nb.saved_fits(folder)
    np.testing.assert_array_equal(fits["quadrupolar"]["result"]["popt"], again.result["popt"])
    assert fits["quadrupolar"]["report"] == window.session.spectra[index].metadata.raw[
        "fits"]["quadrupolar"]["report"]


def test_fit_panel_inputs_round_trip_exactly(qapp):
    window = MainWindow()
    for key, panel in window._fit_panels().items():
        before = panel.get_state()
        panel.set_state(json.loads(json.dumps(before)))
        assert panel.get_state() == before, key


# ------------------------------------------------------------------
# Simulations: journalled and restorable
# ------------------------------------------------------------------

def test_every_simulate_panel_round_trips_its_inputs(qapp):
    window = MainWindow()
    for key, section in window._simulation_panels().items():
        before = section.get_state()
        section.set_state(json.loads(json.dumps(before)))
        assert section.get_state() == before, key


def test_a_csa_simulation_is_journalled_and_restored(qapp):
    window = MainWindow()
    window.sim_csa_nucleus_edit.setText("13C")
    window.sim_csa_diso_spin.setValue(42.0)
    window.sim_csa_aniso_spin.setValue(-80.0)
    window.sim_csa_orientations_spin.setValue(500)
    window._on_simulate_csa_clicked()
    journal = io_history.load_simulation_journal()
    entry = journal["entries"][-1]
    assert entry["panel"] == "csa" and entry["runs"] == 1
    # Sites are logged whole (2026-09-30: several sites), one line each.
    assert "Site 1 (Site 1): δiso 42 ppm, Δ -80 ppm" in "\n".join(entry["inputs_text"])
    assert "δiso = 42 ppm" in entry["summary"]
    text = open(io_history.simulation_journal_paths()[1], encoding="utf-8").read()
    assert "#1 CSA" in text and "δiso 42 ppm" in text

    # A quick re-run is folded into the same entry.
    window.sim_csa_diso_spin.setValue(43.0)
    window._on_simulate_csa_clicked()
    journal = io_history.load_simulation_journal()
    assert len(journal["entries"]) == 1 and journal["entries"][0]["runs"] == 2

    window.sim_csa_diso_spin.setValue(0.0)
    window._restore_simulation(journal["entries"][0])
    assert window.sim_csa_diso_spin.value() == 43.0
    assert "δiso = 43 ppm" in window.sim_csa_results_label.text()  # re-run
    assert io_history.load_simulation_journal()["entries"][0]["runs"] == 2  # not re-logged


def test_panel_simulations_are_journalled_and_restore_their_model(qapp):
    window = MainWindow()
    combined = window.combined_sim_panel
    combined.nucleus_combo.setCurrentText("27Al")
    combined.site_dicts[0]["delta_iso"] = 17.0
    combined.orientations_spin.setValue(600)
    combined.points_spin.setValue(1024)
    combined.add_interaction("quadrupolar", C_Q=2.0, eta_Q=0.2)
    assert combined.simulate()
    entry = io_history.load_simulation_journal()["entries"][-1]
    assert entry["panel"] == "combined"
    assert any("C_Q 2 MHz" in line for line in entry["inputs_text"])

    combined.remove_site()
    assert combined.site_dicts == []
    window._restore_simulation(entry)
    assert combined.site_dicts[0]["delta_iso"] == 17.0
    assert combined.site_dicts[0]["interactions"][0]["C_Q"] == 2.0
    assert combined.spectrum is not None  # simulated again

    jsim = window.jcoupling_sim_panel
    assert jsim.simulate()
    entry = io_history.load_simulation_journal()["entries"][-1]
    assert entry["panel"] == "jcoupling"
    jsim.editor.clear()
    window._restore_simulation(entry)
    assert jsim.editor.sites_table.rowCount() == 1


def test_simulation_history_dialog_restores_the_chosen_run(qapp, monkeypatch):
    window = MainWindow()
    window.bloch_n_offsets_spin.setValue(51)
    window._run_bloch_simulation()
    window.sim_delta_spin.setValue(2.5)
    window._run_diffusion_simulation()
    window.bloch_n_offsets_spin.setValue(99)
    window.sim_delta_spin.setValue(9.0)

    shown = {}

    def pick_bloch(self):
        shown["items"] = [self.list.item(i).text() for i in range(self.list.count())]
        self.list.setCurrentRow(1)  # newest first: diffusion, then Bloch
        shown["preview"] = self.preview.toPlainText()
        return 1
    monkeypatch.setattr(progress_dialogs.SimulationHistoryDialog, "exec", pick_bloch)
    window._on_simulation_history_clicked()
    assert shown["items"][0].startswith("#2 Diffusion (PFG)")
    assert "Points: 51" in shown["preview"]
    assert window.bloch_n_offsets_spin.value() == 51
    assert window.sim_delta_spin.value() == 9.0  # the other panel is untouched


def test_the_simulation_journal_goes_to_the_simulations_folder(qapp, tmp_path):
    assert io_history.default_simulation_root() == str(tmp_path / "simulations")
    window = MainWindow()
    window._run_diffusion_simulation()
    assert os.path.exists(tmp_path / "simulations" / "opennmr_simulation_log.txt")


def test_notebook_simulations_share_the_journal_but_are_not_restored_in_the_gui(qapp):
    from opennmr import notebook as nb
    nb.log_simulation("CSA powder pattern", {"delta_iso": 12.0, "aniso": -40.0}, "made in a cell")
    entry = nb.simulation_history()[-1]
    assert entry["panel"] == "notebook" and "delta_iso: 12" in entry["inputs_text"]
    assert "CSA powder pattern (notebook)" in nb.simulation_log()
    with pytest.raises(ValueError, match="notebook"):
        MainWindow()._restore_simulation(entry)
