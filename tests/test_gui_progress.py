"""Save Progress / Load Progress / Processing Log in the GUI: the log is
written next to the data as changes happen, save points restore the
exact state (through the same path a reopen uses), and simulations or
derived spectra can be saved as openNMR dataset folders."""

import json
import os

import numpy as np
import pytest

from opennmr import io
from opennmr.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
from opennmr.gui import progress as progress_dialogs
from opennmr.gui.main_window import MainWindow
from opennmr.io import history as io_history
from opennmr.io import persistence as io_persistence

from test_gui import (  # noqa: F401 -- qapp is a fixture
    qapp, _make_1d_dataset, _make_raw_1d_dataset, _make_raw_2d_dataset,
    _settle_live_reprocess, _settle_sidecar_save,
)


def _log_text(path):
    with open(io_history.log_path(path), encoding="utf-8") as f:
        return f.read()


def _set_lb(window, value):
    window.window_combo.setCurrentIndex(window.window_combo.findData("exponential"))
    window.lb_spin.setValue(value)
    window._on_fid_processing_control_changed()
    _settle_sidecar_save(window)


def _set_zero_fill(window, value):
    window._set_zero_fill_value(value)
    window._on_fid_processing_control_changed()
    _settle_sidecar_save(window)


# ------------------------------------------------------------------
# The log follows the edits
# ------------------------------------------------------------------

def test_opening_alone_writes_nothing_into_the_data_folder(qapp, tmp_path, monkeypatch):
    path = _make_raw_1d_dataset(tmp_path, monkeypatch, n=64, name="quiet")
    window = MainWindow()
    window._load(path, vendor="bruker")
    window._on_selection_changed(0)
    assert not os.path.exists(io_history.log_path(path))
    assert not os.path.exists(io_history.history_path(path))
    # ...but the session already has its history, starting with the open.
    history = window.session.info(0)["history"]
    assert history["entries"][0]["kind"] == "opened"


def test_an_edit_is_logged_next_to_the_data_from_old_to_new(qapp, tmp_path, monkeypatch):
    path = _make_raw_1d_dataset(tmp_path, monkeypatch, n=64, name="logged")
    window = MainWindow()
    window._load(path, vendor="bruker")
    _set_lb(window, 5.0)

    text = _log_text(path)
    assert "Opened in openNMR" in text and "no earlier openNMR processing" in text
    assert "Line broadening (LB): 0 Hz → 5 Hz" in text
    # Only what was touched -- not every other control's default.
    assert "Zero-fill" not in text.split("Current state:")[0]
    saved = io_history.load_history(path)
    assert saved["state"]["live_lb"] == 5.0


def test_a_phase_edit_and_the_log_agree_with_the_sidecar(qapp, tmp_path, monkeypatch):
    path = _make_raw_1d_dataset(tmp_path, monkeypatch, n=64, name="phaselog")
    window = MainWindow()
    window._load(path, vendor="bruker")
    window.phase0_handle.setValue(45.0)
    window._capture_phase_reference()
    _settle_sidecar_save(window)

    assert io_persistence.load_processing_sidecar(path)["phase0"] == pytest.approx(45.0)
    assert "Phase PH0: 45°" in _log_text(path)


def test_closing_the_window_does_not_lose_the_last_edit(qapp, tmp_path, monkeypatch):
    # Previously a change made < 0.8 s before closing was never saved.
    path = _make_raw_1d_dataset(tmp_path, monkeypatch, n=64, name="closeflush")
    window = MainWindow()
    window._load(path, vendor="bruker")
    window._set_zero_fill_value(128)
    window._on_fid_processing_control_changed()
    assert window._sidecar_save_timer.isActive()
    window.close()
    assert io_persistence.load_processing_sidecar(path)["live_zero_fill"] == 128
    assert "Zero-fill: 64 points → 128 points" in _log_text(path)


def test_an_unwritable_folder_is_reported_not_raised(qapp, tmp_path, monkeypatch):
    path = _make_raw_1d_dataset(tmp_path, monkeypatch, n=64, name="readonly")
    window = MainWindow()
    window._load(path, vendor="bruker")

    def refuse(*_args, **_kwargs):
        raise PermissionError("read-only")
    monkeypatch.setattr(io_history, "write_history", refuse)
    _set_lb(window, 2.0)  # must not raise
    assert window.session.info(0)["history_error"] == "read-only"
    assert "processing log" in window.statusBar().currentMessage()


# ------------------------------------------------------------------
# Save points and restoring them
# ------------------------------------------------------------------

def test_save_progress_records_a_named_save_point_with_the_full_state(qapp, tmp_path, monkeypatch):
    path = _make_raw_1d_dataset(tmp_path, monkeypatch, n=64, name="savept")
    window = MainWindow()
    window._load(path, vendor="bruker")
    _set_lb(window, 3.0)
    # A phase still only in the controls is committed by the save.
    window.phase0_handle.setValue(30.0)

    entry = window._save_progress(0, name="EM 3 Hz, phased", note="check baseline next")

    assert entry["id"] == 1
    assert entry["state"]["live_lb"] == 3.0
    assert entry["state"]["phase0"] == pytest.approx(30.0)
    assert "snapshot" not in entry  # recomputable from the FID: the recipe is enough
    text = _log_text(path)
    assert "★ SAVE POINT 1: EM 3 Hz, phased" in text
    assert "Note: check baseline next" in text
    assert "Phase PH0: 30°" in text.split("SAVE POINT 1")[1]


def test_restoring_a_save_point_brings_back_the_data_exactly(qapp, tmp_path, monkeypatch):
    path = _make_raw_1d_dataset(tmp_path, monkeypatch, n=64, name="restore")
    window = MainWindow()
    window._load(path, vendor="bruker")
    _set_lb(window, 4.0)
    window.phase0_handle.setValue(20.0)
    window._save_progress(0, name="good")
    good = window.session.spectra[0].complex_data.copy()

    _set_zero_fill(window, 256)
    window.phase0_handle.setValue(70.0)
    window._capture_phase_reference()
    _settle_sidecar_save(window)
    assert len(window.session.spectra[0].data) == 256

    window._restore_progress(0, 1)

    spectrum = window.session.spectra[0]
    np.testing.assert_allclose(spectrum.complex_data, good, atol=1e-12)
    assert spectrum.metadata.raw["live_lb"] == 4.0
    assert spectrum.metadata.raw["phase0"] == pytest.approx(20.0)
    assert window.lb_spin.value() == 4.0
    assert window._zero_fill_value() == 64
    # The state being left was kept first, so nothing is lost.
    points = io_history.savepoints(io_history.load_history(path))
    assert [p["name"] for p in points] == ["good", "Automatic: before restoring save point 1 “good”"]
    assert points[1]["state"]["live_zero_fill"] == 256
    text = _log_text(path)
    assert "Restored save point 1 “good”" in text
    assert "Zero-fill: 256 points → 64 points" in text


def test_restoring_the_same_state_twice_adds_no_duplicate_automatic_save_point(qapp, tmp_path, monkeypatch):
    path = _make_raw_1d_dataset(tmp_path, monkeypatch, n=64, name="nodup")
    window = MainWindow()
    window._load(path, vendor="bruker")
    _set_lb(window, 4.0)
    window._save_progress(0, name="a")
    window._restore_progress(0, 1)  # already there: nothing to keep
    assert len(io_history.savepoints(window.session.info(0)["history"])) == 1


def test_restoring_the_original_matches_a_fresh_open(qapp, tmp_path, monkeypatch):
    path = _make_raw_1d_dataset(tmp_path, monkeypatch, n=64, name="orig")
    fresh = MainWindow()
    fresh._load(path, vendor="bruker")
    original = fresh.session.spectra[0].complex_data.copy()

    window = MainWindow()
    window._load(path, vendor="bruker")
    _set_lb(window, 10.0)
    _set_zero_fill(window, 128)
    window._restore_progress(0, progress_dialogs.ORIGINAL)

    np.testing.assert_allclose(window.session.spectra[0].complex_data, original, atol=1e-12)
    assert "live_lb" not in io_persistence.load_processing_sidecar(path)


def test_save_points_survive_a_restart_and_restore_there(qapp, tmp_path, monkeypatch):
    path = _make_raw_1d_dataset(tmp_path, monkeypatch, n=64, name="restart")
    window = MainWindow()
    window._load(path, vendor="bruker")
    _set_zero_fill(window, 128)
    window._save_progress(0, name="zf128")
    expected = window.session.spectra[0].complex_data.copy()
    _set_zero_fill(window, 512)
    window.close()

    reopened = MainWindow()
    reopened._load(path, vendor="bruker")
    assert len(reopened.session.spectra[0].data) == 512
    history = reopened.session.info(0)["history"]
    assert io_history.find_savepoint(history, "zf128")["state"]["live_zero_fill"] == 128
    reopened._restore_progress(0, "zf128")
    np.testing.assert_allclose(reopened.session.spectra[0].complex_data, expected, atol=1e-12)
    assert _log_text(path).count("Opened in openNMR") == 2


def test_a_state_changed_behind_the_logs_back_is_noted_on_opening(qapp, tmp_path, monkeypatch):
    path = _make_raw_1d_dataset(tmp_path, monkeypatch, n=64, name="drift")
    window = MainWindow()
    window._load(path, vendor="bruker")
    _set_lb(window, 1.0)
    window.close()
    sidecar = io_persistence.processing_sidecar_path(path)
    with open(sidecar) as f:
        payload = json.load(f)
    payload["state"]["live_lb"] = 7.0
    with open(sidecar, "w") as f:
        json.dump(payload, f)

    reopened = MainWindow()
    reopened._load(path, vendor="bruker")
    _set_zero_fill(reopened, 128)  # any change writes the log
    text = _log_text(path)
    assert "differed from this log's last record" in text
    assert "Line broadening (LB): 1 Hz → 7 Hz" in text


def test_a_2d_restore_keeps_the_saved_pseudo_2d_choice_and_f1_settings(qapp, tmp_path, monkeypatch):
    path = _make_raw_2d_dataset(tmp_path, monkeypatch, n_f1=8, n_f2=16, name="twod", f1_nuc="13C")
    window = MainWindow()
    window._load(path, vendor="bruker")
    window.phase0_handle.setValue(25.0)
    window._save_progress(0, name="p25")
    _settle_live_reprocess(window)
    saved = window.session.spectra[0].data.copy()
    pseudo = window.session.spectra[0].is_pseudo_2d

    window.phase0_handle.setValue(-60.0)
    window._capture_phase_reference()
    _settle_sidecar_save(window)
    window._restore_progress(0, "p25")
    _settle_live_reprocess(window)

    restored = window.session.spectra[0]
    assert restored.is_pseudo_2d == pseudo
    np.testing.assert_allclose(restored.data, saved, atol=1e-9)


# ------------------------------------------------------------------
# Spectra that cannot be recomputed from a raw FID keep their data
# ------------------------------------------------------------------

def test_a_processed_only_spectrum_keeps_a_data_snapshot_per_save_point(qapp, tmp_path):
    path = _make_1d_dataset(tmp_path, n=16, name="processed")
    window = MainWindow()
    window._load(path, vendor="bruker")
    assert not window._fid_reprocessable(window.session.spectra[0])
    before = window.session.spectra[0].data.copy()
    entry = window._save_progress(0, name="as read")
    assert entry["snapshot"].replace("\\", "/") == "opennmr_savepoints/savepoint_1.npz"
    assert os.path.exists(os.path.join(path, "opennmr_savepoints", "savepoint_1.npz"))

    window.session.spectra[0].data[:] = 0.0  # anything that changes the data
    window._restore_progress(0, 1)
    np.testing.assert_array_equal(window.session.spectra[0].data, before)


# ------------------------------------------------------------------
# Simulations and derived spectra
# ------------------------------------------------------------------

def _simulation():
    ppm = np.linspace(100.0, -100.0, 64)
    return NMRSpectrum(
        data=np.exp(-ppm ** 2 / 200.0), axes=[NMRAxis(kind="frequency", ppm=ppm)],
        data_imag=np.zeros(64),
        metadata=NMRMetadata(vendor="simulated", nucleus="13C", sfo1=100.6,
                             raw={"simulation": {"kind": "csa", "d_iso": 12.5, "aniso": -40.0}}))


def test_a_simulation_is_logged_in_memory_with_its_parameters(qapp):
    window = MainWindow()
    index = window._add_derived_spectrum(_simulation(), "Simulated CSA")
    history = window.session.info(index)["history"]
    created = history["entries"][0]
    assert created["kind"] == "created"
    assert "d_iso: 12.5" in created["text"] and "aniso: -40" in created["text"]
    assert io_history.savepoints(history)[0]["name"] == "As created"
    text = io_history.render_log(history)
    assert "Simulation parameters:" in text


def test_saving_a_simulation_writes_a_dataset_folder_that_reopens(qapp, tmp_path):
    window = MainWindow()
    index = window._add_derived_spectrum(_simulation(), "Simulated CSA")
    window.phase0_handle.setValue(15.0)
    folder = str(tmp_path / "sims" / "csa one")
    entry = window._save_progress(index, name="phased sim", folder=folder)

    spectrum = window.session.spectra[index]
    assert spectrum.metadata.vendor == "opennmr" and spectrum.metadata.source_path == folder
    assert window.session.label(index) == "Simulated CSA"
    assert entry["id"] == 2  # "As created" is 1
    for name in ("opennmr_spectrum.npz", "opennmr_spectrum.json", "opennmr_log.txt",
                 "opennmr_history.json", "opennmr_processing.json"):
        assert os.path.exists(os.path.join(folder, name)), name
    # The in-memory "As created" data went to disk with it.
    first = io_history.savepoints(io_history.load_history(folder))[0]
    assert os.path.exists(os.path.join(folder, first["snapshot"]))
    text = _log_text(folder)
    assert "Created in openNMR" in text and "Saved as an openNMR dataset folder" in text

    phased = spectrum.data.copy()
    reopened = MainWindow()
    reopened._load(folder, vendor=io.detect_vendor(folder))
    again = reopened.session.spectra[0]
    np.testing.assert_array_equal(again.data, phased)
    assert reopened.session.label(0) == "Simulated CSA"
    assert again.metadata.raw["simulation"]["d_iso"] == 12.5

    # ...and "As created" still restores exactly, after the restart.
    reopened._restore_progress(0, "As created")
    np.testing.assert_array_equal(reopened.session.spectra[0].data, _simulation().data)


def test_a_derived_spectrum_defaults_to_a_folder_inside_its_parent(qapp, tmp_path, monkeypatch):
    parent = tmp_path / "7"
    parent.mkdir()
    window = MainWindow()
    trace = _simulation()
    trace.metadata.source_path = str(parent)
    index = window._add_derived_spectrum(trace, "CPMG 7")
    assert window.session.info(index)["origin"] == str(parent)
    assert io_history.default_dataset_folder(
        window.session.spectra[index], "CPMG 7", parent_path=window.session.info(index)["origin"]
    ) == str(parent / "opennmr_derived" / "CPMG 7")


def test_save_progress_on_a_simulation_needs_a_folder(qapp):
    window = MainWindow()
    index = window._add_derived_spectrum(_simulation(), "Simulated CSA")
    with pytest.raises(ValueError):
        window._save_progress(index, name="x")


# ------------------------------------------------------------------
# Menus and dialogs
# ------------------------------------------------------------------

def test_file_menu_has_save_load_and_log(qapp):
    window = MainWindow()
    assert window.save_progress_action.shortcut().toString() == "Ctrl+Shift+S"
    assert window.load_progress_action.shortcut().toString() == "Ctrl+Shift+L"
    assert window.show_log_action.text() == "Processing &Log…"


def test_the_dialogs_drive_save_and_restore(qapp, tmp_path, monkeypatch):
    path = _make_raw_1d_dataset(tmp_path, monkeypatch, n=64, name="dialogs")
    window = MainWindow()
    window._load(path, vendor="bruker")
    _set_lb(window, 2.0)

    monkeypatch.setattr(progress_dialogs.SaveProgressDialog, "exec", lambda self: 1)
    monkeypatch.setattr(progress_dialogs.SaveProgressDialog, "values",
                        lambda self: ("from dialog", "a note", None))
    window._on_save_progress_clicked()
    assert io_history.find_savepoint(io_history.load_history(path), "from dialog")["note"] == "a note"

    _set_lb(window, 9.0)
    shown = {}

    def pick_first(self):
        shown["items"] = [self.list.item(i).text() for i in range(self.list.count())]
        shown["preview"] = self.preview.toPlainText()
        return 1
    monkeypatch.setattr(progress_dialogs.LoadProgressDialog, "exec", pick_first)
    window._on_load_progress_clicked()
    assert shown["items"][0].startswith("1. from dialog")
    assert shown["items"][-1].startswith("Original")
    assert "Line broadening (LB): 2 Hz" in shown["preview"]
    assert window.session.spectra[0].metadata.raw["live_lb"] == 2.0

    logged = {}
    monkeypatch.setattr(progress_dialogs.ProcessingLogDialog, "exec",
                        lambda self: logged.setdefault("text", self.text_edit.toPlainText()))
    window._on_show_log_clicked()
    assert "SAVE POINT 1: from dialog" in logged["text"]


def test_a_folder_scan_finds_spectra_saved_inside_their_parent_experiment(qapp, tmp_path):
    parent = _make_1d_dataset(tmp_path, n=16, name="20")
    window = MainWindow()
    index = window._add_derived_spectrum(_simulation(), "slice 3 of 20")
    folder = os.path.join(parent, "opennmr_derived", "slice 3 of 20")
    window._save_progress(index, name="kept", folder=folder)
    found = dict(MainWindow._scan_for_spectra(str(tmp_path)))
    assert found[parent] == "bruker"
    assert found[folder] == "opennmr"


def test_reset_and_title_edits_are_logged(qapp, tmp_path, monkeypatch):
    path = _make_raw_1d_dataset(tmp_path, monkeypatch, n=64, name="resetlog")
    window = MainWindow()
    window._load(path, vendor="bruker")
    _set_lb(window, 6.0)
    window._on_reset_processing_clicked()
    _settle_sidecar_save(window)
    spectrum = window.session.spectra[0]
    from opennmr.notebook import notes as nb_notes
    nb_notes.set_experiment_notes(spectrum, "sample 3, 80 C")
    window._on_notes_saved(spectrum, "title")
    nb_notes.set_experiment_notes(spectrum, "sample 3, 80 C, 2 h")
    window._on_notes_saved(spectrum, "title")
    text = _log_text(path)
    assert "Reset: processing discarded" in text
    assert "Line broadening (LB): 6 Hz → 0 Hz" in text or "Apodization: exponential (EM) →" in text
    assert text.count("Title/notes edited") == 1
    assert "Title/notes edited: “sample 3, 80 C, 2 h”" in text
