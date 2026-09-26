"""The per-spectrum processing history and its human-readable log
(opennmr.io.history) -- Qt-free, so usable from a notebook."""

import datetime as dt
import json
import os

import numpy as np
import pytest

from opennmr import io
from opennmr.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
from opennmr.io import history, persistence


def _spectrum(source_path=None, ndim=1, **raw):
    ppm = np.linspace(10.0, 0.0, 16)
    axes = [NMRAxis(kind="frequency", ppm=ppm, hz=ppm * 400.0)]
    data = np.arange(16.0)
    if ndim == 2:
        axes = [NMRAxis(kind="index", index=np.arange(1, 4))] + axes
        data = np.arange(48.0).reshape(3, 16)
    return NMRSpectrum(
        data=data, axes=axes, data_imag=data * 0.5,
        metadata=NMRMetadata(vendor="bruker", nucleus="13C", sfo1=100.6,
                             source_path=source_path, raw=dict(raw)),
    )


T0 = dt.datetime(2026, 9, 26, 15, 0, 0, tzinfo=dt.timezone(dt.timedelta(hours=2)))


def at(seconds):
    return T0 + dt.timedelta(seconds=seconds)


# ------------------------------------------------------------------
# The tracked state is the persisted state (one definition, not two)
# ------------------------------------------------------------------

def test_tracked_state_is_the_processing_sidecar_state_plus_metadata():
    spectrum = _spectrum(live_lb=5.0, phase0=12.0, acqus={"big": 1}, fid=np.zeros(4),
                         _read_used_saved_processing=True)
    spectrum.metadata.mass = 0.05
    state = history.tracked_state(spectrum)
    assert state["live_lb"] == 5.0 and state["phase0"] == 12.0
    assert state["meta.mass_g"] == 0.05 and state["meta.nucleus"] == "13C"
    # Same exclusions as the sidecar: vendor dumps, arrays, bookkeeping.
    for key in ("acqus", "fid", "_read_used_saved_processing"):
        assert key not in state
    processing, metadata = history.split_state(state)
    assert processing == persistence.processing_state(spectrum)
    assert metadata == {"mass_g": 0.05, "nucleus": "13C"}


def test_pseudo_2d_choice_is_tracked_only_for_2d():
    assert "meta.pseudo_2d" not in history.tracked_state(_spectrum())
    assert history.tracked_state(_spectrum(ndim=2))["meta.pseudo_2d"] is False


# ------------------------------------------------------------------
# Diffing and the words used
# ------------------------------------------------------------------

def test_a_change_is_written_from_old_to_new_with_units():
    changes = history.diff_states({"live_window": "none", "live_lb": 0.0},
                                  {"live_window": "exponential", "live_lb": 5.0})
    lines = history.change_lines(changes)
    assert "Apodization: no apodization → exponential (EM)" in lines
    assert "Line broadening (LB): 0 Hz → 5 Hz" in lines


def test_a_setting_first_appearing_at_a_blank_value_is_not_a_change():
    # A control persisting its default (False/0) is not something the
    # user did -- listing it would bury the real change.
    changes = history.diff_states({}, {"baseline_correct": False, "live_fid_shift": 0, "phase0": 30.0})
    assert [c["key"] for c in changes] == ["phase0"]


def test_changes_are_listed_in_processing_order():
    changes = history.diff_states({}, {"sr_shift_hz": 3.0, "phase0": 10.0, "live_lb": 2.0,
                                      "live_window": "exponential"})
    assert [c["key"] for c in changes] == ["live_window", "live_lb", "phase0", "sr_shift_hz"]


def test_analysis_state_changes_are_reported_setting_by_setting():
    before = {"analysis_state": {"peaks": [["1.20", "0.1", "0.5", "0"]], "fit_type": 0,
                                 "option_names": {"fit_type": "Relaxation (T1/T2)"},
                                 "pfg": {"delta": 1.0}}}
    after = {"analysis_state": {"peaks": [["1.20", "0.1", "0.5", "0"], ["4.52", "0.1", "0.5", "0"]],
                                "fit_type": 1, "option_names": {"fit_type": "Diffusion (PFG)"},
                                "pfg": {"delta": 2.0},
                                "results_text": "<b>Peak 1</b>: area 12.0<br>Peak 2: area 3.0"}}
    lines = history.change_lines(history.diff_states(before, after), before, after)
    assert "Peaks: 1 (1.20 ppm) → 2 (1.20, 4.52 ppm)" in lines
    assert "Curve-fit type: “Relaxation (T1/T2)” → “Diffusion (PFG)”" in lines
    assert "PFG delta: 1 → 2" in lines
    # Results are recorded as text, markup stripped, one line each.
    assert "Fit results:" in lines
    assert "    Peak 1: area 12.0" in lines and "    Peak 2: area 3.0" in lines
    # Plumbing never reaches the log.
    assert not any("option_names" in line or "Option names" in line for line in lines)


def test_per_slice_phases_list_only_the_slices_that_changed():
    lines = history.change_lines(history.diff_states(
        {"slice_phase0": [0.0, 0.0, 0.0]}, {"slice_phase0": [0.0, 15.0, 0.0]}))
    assert lines == ["Per-slice PH0: slice 2: 0° → 15°"]


def test_an_unknown_key_is_still_logged_under_its_own_name():
    # A control added later is logged without anyone editing a table.
    lines = history.change_lines(history.diff_states({}, {"brand_new_setting": 3}))
    assert lines == ["brand_new_setting: 3"]


# ------------------------------------------------------------------
# Recording, merging, save points
# ------------------------------------------------------------------

def test_record_state_merges_a_burst_of_edits_into_one_entry():
    h = history.new_history(_spectrum(), state={"live_lb": 0.0})
    history.record_state(h, {"live_lb": 1.0}, when=at(0))
    history.record_state(h, {"live_lb": 10.0}, when=at(5))
    history.record_state(h, {"live_lb": 10.0, "phase0": 3.0}, when=at(20))
    assert len(h["entries"]) == 1
    changes = {c["key"]: (c["old"], c["new"]) for c in h["entries"][0]["changes"]}
    assert changes == {"live_lb": (0.0, 10.0), "phase0": (None, 3.0)}


def test_record_state_starts_a_new_entry_after_a_pause():
    h = history.new_history(_spectrum(), state={"live_lb": 0.0})
    history.record_state(h, {"live_lb": 1.0}, when=at(0))
    history.record_state(h, {"live_lb": 2.0}, when=at(history.MERGE_WINDOW_S + 60))
    assert len(h["entries"]) == 2


def test_an_edit_undone_within_the_burst_leaves_no_entry():
    h = history.new_history(_spectrum(), state={"live_lb": 0.0})
    history.record_state(h, {"live_lb": 4.0}, when=at(0))
    history.record_state(h, {"live_lb": 0.0}, when=at(3))
    assert h["entries"] == []


def test_nothing_changed_records_nothing():
    h = history.new_history(_spectrum(), state={"live_lb": 1.0})
    assert history.record_state(h, {"live_lb": 1.0}, when=at(0)) == []
    assert h["entries"] == []


def test_savepoints_hold_the_full_state_and_are_found_by_number_or_name():
    h = history.new_history(_spectrum(), state={})
    first = history.add_savepoint(h, {"phase0": 10.0}, name="phased", when=at(0))
    history.add_savepoint(h, {"phase0": 10.0, "live_lb": 5.0}, when=at(200))
    assert first["id"] == 1 and first["state"] == {"phase0": 10.0}
    assert history.find_savepoint(h, 1)["name"] == "phased"
    assert history.find_savepoint(h, "phased")["id"] == 1
    assert history.find_savepoint(h, 2)["name"] == "Save point 2"
    # The change that led to the save point is logged before it.
    assert [e["kind"] for e in h["entries"]] == ["change", "savepoint", "change", "savepoint"]
    with pytest.raises(KeyError):
        history.find_savepoint(h, 7)


# ------------------------------------------------------------------
# Files: JSON is the record, the text is generated from it
# ------------------------------------------------------------------

def test_write_and_load_history_round_trip_next_to_the_data(tmp_path):
    expt = tmp_path / "10"
    expt.mkdir()
    spectrum = _spectrum(str(expt))
    h = history.new_history(spectrum, state={})
    history.add_event(h, "Opened in openNMR", kind="opened", when=at(0))
    history.record_state(h, {"live_window": "exponential", "live_lb": 5.0}, when=at(10))
    history.add_savepoint(h, {"live_window": "exponential", "live_lb": 5.0}, name="EM 5 Hz",
                          note="for the paper", when=at(20))
    json_path, text_path = history.write_history(str(expt), h, title="10 · 13C")

    assert json_path == str(expt / "opennmr_history.json")
    assert text_path == str(expt / "opennmr_log.txt")
    loaded = history.load_history(str(expt))
    assert history.find_savepoint(loaded, "EM 5 Hz")["state"] == {"live_window": "exponential", "live_lb": 5.0}
    assert loaded["next_savepoint"] == 2

    text = open(text_path, encoding="utf-8").read()
    assert "openNMR processing log" in text
    assert "Saturday 26 September 2026" in text
    assert "15:00:10  Changed" in text
    assert "Apodization: exponential (EM)" in text
    assert "★ SAVE POINT 1: EM 5 Hz" in text
    assert "Note: for the paper" in text
    assert "are not read back" in text


def test_a_file_scoped_experiment_gets_sibling_history_files(tmp_path):
    jdf = tmp_path / "proton.jdf"
    jdf.write_bytes(b"\x00")
    assert history.history_path(str(jdf)) == str(tmp_path / "proton.jdf.opennmr_history.json")
    assert history.log_path(str(jdf)) == str(tmp_path / "proton.jdf.opennmr_log.txt")


def test_editing_the_text_log_cannot_change_what_is_restored(tmp_path):
    expt = tmp_path / "11"
    expt.mkdir()
    h = history.new_history(_spectrum(str(expt)), state={})
    history.add_savepoint(h, {"phase0": 10.0}, when=at(0))
    _json, text_path = history.write_history(str(expt), h)
    with open(text_path, "w", encoding="utf-8") as f:
        f.write("PH0 = 99")
    assert history.find_savepoint(history.load_history(str(expt)), 1)["state"] == {"phase0": 10.0}


def test_load_history_is_none_for_missing_or_corrupt_files(tmp_path):
    assert history.load_history(str(tmp_path)) is None
    (tmp_path / "opennmr_history.json").write_text("{not json")
    assert history.load_history(str(tmp_path)) is None
    assert history.load_history(None) is None


def test_write_history_refuses_a_missing_folder(tmp_path):
    with pytest.raises(FileNotFoundError):
        history.write_history(str(tmp_path / "gone"), history.new_history(_spectrum(), state={}))


# ------------------------------------------------------------------
# Snapshots and openNMR dataset folders (simulations, derived spectra)
# ------------------------------------------------------------------

def test_snapshot_round_trips_data_and_axes_exactly(tmp_path):
    spectrum = _spectrum(ndim=2)
    spectrum.data = spectrum.data + 1e-13  # precision matters
    path = history.write_snapshot(str(tmp_path / "snap" / "1.npz"), spectrum)
    data, data_imag, axes = history.read_snapshot(path)
    np.testing.assert_array_equal(data, spectrum.data)
    np.testing.assert_array_equal(data_imag, spectrum.data_imag)
    assert [a.kind for a in axes] == ["index", "frequency"]
    np.testing.assert_array_equal(axes[1].ppm, spectrum.axes[1].ppm)
    np.testing.assert_array_equal(axes[0].index, spectrum.axes[0].index)


def test_a_saved_simulation_folder_reopens_through_io_load(tmp_path):
    ppm = np.linspace(200, -50, 32)
    sim = NMRSpectrum(data=np.exp(-ppm ** 2 / 100), axes=[NMRAxis(kind="frequency", ppm=ppm)],
                      metadata=NMRMetadata(vendor="simulated", nucleus="13C", sfo1=100.6,
                                           raw={"simulation": {"kind": "csa", "d_iso": 10.0}}))
    folder = history.save_spectrum_folder(sim, str(tmp_path / "CSA sim"), label="Simulated CSA")
    persistence.save_processing_sidecar(
        NMRSpectrum(data=sim.data, axes=sim.axes,
                    metadata=NMRMetadata(vendor="opennmr", source_path=folder, raw=sim.metadata.raw)))

    assert io.detect_vendor(folder) == "opennmr"
    loaded = io.load(folder)
    np.testing.assert_array_equal(loaded.data, sim.data)
    np.testing.assert_array_equal(loaded.axes[0].ppm, ppm)
    assert loaded.metadata.vendor == "opennmr"
    assert loaded.metadata.nucleus == "13C" and loaded.metadata.sfo1 == pytest.approx(100.6)
    assert loaded.metadata.raw["origin_vendor"] == "simulated"
    assert loaded.metadata.source_path == folder
    # The simulation's parameters travel in the processing sidecar.
    assert persistence.load_processing_sidecar(folder)["simulation"]["d_iso"] == 10.0
    assert io.peek(folder)["shape"] == (32,)


def test_default_folders_keep_derived_spectra_with_their_parent(tmp_path, monkeypatch):
    parent = tmp_path / "12"
    parent.mkdir()
    assert history.default_dataset_folder(_spectrum(), "CPMG 12", parent_path=str(parent)) == \
        str(parent / "opennmr_derived" / "CPMG 12")
    monkeypatch.setattr(history, "default_simulation_root", lambda: str(tmp_path / "sims"))
    folder = history.default_dataset_folder(_spectrum(), "Simulated CSA/δ", when=T0)
    assert folder == str(tmp_path / "sims" / "2026-09-26_150000_Simulated CSA")


def test_unique_folder_never_reuses_an_existing_one(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a (2)").mkdir()
    assert history.unique_folder(str(tmp_path / "a")) == str(tmp_path / "a (3)")
    assert history.unique_folder(str(tmp_path / "b")) == str(tmp_path / "b")


def test_a_summary_lists_only_settings_that_mean_something():
    state = {"live_window": "exponential", "live_lb": 5.0, "live_ssb": 2.0, "live_gb": 0.3,
             "live_linear_predict_points": 0, "live_linear_predict_order": 20,
             "live_truncate": 64, "fid_original_length": 64, "fid_dt": 1e-4,
             "baseline_correct": False, "baseline_method": "polynomial", "baseline_degree": 3,
             "meta.nucleus": "13C",
             "analysis_state": {"fit_within_regions": True, "n_components": 2, "regions": []}}
    lines = history.state_summary(state, source={"nucleus": "13C"})
    assert lines == ["Apodization: exponential (EM)", "Line broadening (LB): 5 Hz"]
    # A nucleus override is worth saying; the file's own is not.
    assert "Nucleus: 7Li" in history.state_summary({"meta.nucleus": "7Li"}, source={"nucleus": "13C"})
    state.update(baseline_correct=True, live_truncate=32)
    lines = history.state_summary(state, source={"nucleus": "13C"})
    assert "Baseline method: polynomial" in lines and "Baseline polynomial degree: 3" in lines
    assert "Truncation (TDeff): first 32 points" in lines


def test_a_large_result_is_fingerprinted_in_change_entries_not_copied():
    big = {"map": list(range(20000))}
    h = history.new_history(_spectrum(), state={})
    history.record_state(h, {"dosy_result": big}, when=at(0))
    change = h["entries"][0]["changes"][0]
    assert set(change["new"]) == {"_digest", "_size"}
    assert h["state"]["dosy_result"] == big  # the state itself keeps it, for save points
    assert history.change_lines(h["entries"][0]["changes"]) == ["DOSY transform: computed"]


def test_simulation_parameters_are_written_out_one_per_line():
    lines = history.state_summary({"simulation": {"kind": "csa", "d_iso": 12.5, "sites": [{"a": 1}]}})
    assert lines[0] == "Simulation parameters:"
    assert "    kind: csa" in lines and "    d_iso: 12.5" in lines and '    sites: [{"a": 1}]' in lines


# ------------------------------------------------------------------
# The notebook API (opennmr.notebook) -- same files as the GUI
# ------------------------------------------------------------------

def _sim():
    ppm = np.linspace(100.0, -100.0, 32)
    return NMRSpectrum(data=np.exp(-ppm ** 2 / 50.0), axes=[NMRAxis(kind="frequency", ppm=ppm)],
                       metadata=NMRMetadata(vendor="simulated", nucleus="13C",
                                            raw={"simulation": {"kind": "csa", "d_iso": 3.0}}))


def test_notebook_saves_a_simulation_as_a_dataset_folder_with_its_log(tmp_path):
    from opennmr import notebook as nb

    sim = _sim()
    nb.log_event(sim, "made for the tutorial")
    saved, entry = nb.save_progress(sim, "first", note="n", folder=str(tmp_path / "sim"))
    assert saved.metadata.vendor == "opennmr" and entry["snapshot"]
    folder = saved.metadata.source_path

    text = nb.processing_log(folder)
    assert "Created in openNMR" in text and "d_iso: 3" in text
    assert "made for the tutorial" in text and "★ SAVE POINT 1: first" in text
    assert nb.list_save_points(folder) == [
        {"id": 1, "name": "first", "time": entry["time"], "note": "n", "has_data": True}]
    assert nb.save_point_state(folder, 1)["processing"]["simulation"]["d_iso"] == 3.0

    # Change it, save again; the first save point still loads exactly.
    saved.data = saved.data * 2.0
    nb.save_progress(saved, "doubled")
    np.testing.assert_array_equal(nb.load_progress(folder, "first").data, sim.data)
    np.testing.assert_array_equal(nb.load_progress(folder, 2).data, sim.data * 2.0)
    np.testing.assert_array_equal(nb.load_saved_spectrum(folder).data, sim.data)


def test_notebook_load_progress_refuses_a_settings_only_save_point(tmp_path):
    from opennmr import notebook as nb

    expt = tmp_path / "10"
    expt.mkdir()
    h = history.new_history(_spectrum(str(expt)), state={})
    history.add_savepoint(h, {"live_lb": 5.0}, name="EM", when=at(0))
    history.write_history(str(expt), h)
    with pytest.raises(NotImplementedError, match="save_point_state"):
        nb.load_progress(str(expt), "EM")
    assert nb.save_point_state(str(expt), "EM")["summary"] == []  # no window set: LB means nothing
    assert "No openNMR processing log" in nb.processing_log(str(tmp_path / "nothing"))
