"""Autosave and crash recovery (explicit user request, 2026-11-12): the
open session is written every few minutes as an openNMR state (only when
something changed); a normal close removes it; a session that did not
close normally is moved aside at the next start-up and offered back, and
kept for File ▸ Recover Autosaved Session…."""

import os

import numpy as np
import pytest
from PySide6.QtWidgets import QMessageBox

from opennmr.io import recovery
from opennmr.io import state as io_state

from test_gui import qapp, _load_spectrum_directly  # noqa: F401
from test_state_and_fit_models import _derived


def _write(folder):
    io_state.save_state(folder, "Autosave", [{"label": "p", "kind": "derived",
                                              "spectrum": _derived()}], overwrite=True)


# ---- io.recovery ---------------------------------------------------------------------------

def test_the_recovery_folder_is_isolated_in_tests(tmp_path):
    assert recovery.recovery_root() == str(tmp_path / "recovery")


def test_a_clean_exit_leaves_nothing():
    assert recovery.take_crashed_session(pid=100) is None
    recovery.write_autosave(_write)
    assert io_state.is_state_folder(recovery.autosave_folder())
    recovery.clean_exit(pid=100)
    assert not os.path.exists(recovery.autosave_folder()) and recovery.read_lock() is None


def test_a_crash_is_moved_aside_and_kept():
    recovery.take_crashed_session(pid=100)
    recovery.write_autosave(_write)                   # then process 100 dies
    moved = recovery.take_crashed_session(pid=200)
    assert moved and io_state.is_state_folder(moved)
    assert os.path.basename(moved).startswith(recovery.PREVIOUS_PREFIX)
    assert not os.path.exists(recovery.autosave_folder())   # 200's autosaves cannot overwrite it
    assert recovery.previous_sessions() == [moved]
    assert recovery.read_lock()["pid"] == 200
    assert "1 spectrum" in recovery.describe(moved)
    recovery.clean_exit(pid=200)
    assert recovery.previous_sessions() == [moved]           # still there after a clean exit


def test_the_same_process_is_not_a_crash():
    recovery.take_crashed_session(pid=100)
    recovery.write_autosave(_write)
    assert recovery.take_crashed_session(pid=100) is None
    assert io_state.is_state_folder(recovery.autosave_folder())


def test_only_the_newest_few_are_kept():
    for k in range(recovery.KEEP_PREVIOUS + 2):
        recovery.take_crashed_session(pid=1000 + 2 * k)
        recovery.write_autosave(_write)
    recovery.take_crashed_session(pid=9999)
    assert len(recovery.previous_sessions()) == recovery.KEEP_PREVIOUS


def test_a_failed_autosave_leaves_the_last_one_intact():
    recovery.write_autosave(_write)

    def broken(folder):
        os.makedirs(folder)
        raise OSError("disk full")
    with pytest.raises(OSError):
        recovery.write_autosave(broken)
    assert io_state.is_state_folder(recovery.autosave_folder())


def test_another_process_lock_is_left_alone_by_clean_exit():
    recovery.write_lock(pid=100)
    recovery.clean_exit(pid=200)
    assert recovery.read_lock()["pid"] == 100


# ---- the window ----------------------------------------------------------------------------

def _window():
    from opennmr.gui.main_window import MainWindow
    return MainWindow()


def _spectrum():
    from opennmr.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
    ppm = np.linspace(10, 0, 256)
    return NMRSpectrum(np.exp(-(ppm - 5) ** 2), [NMRAxis("frequency", ppm=ppm, hz=ppm * 400)],
                       NMRMetadata(vendor="bruker", nucleus="1H", sfo1=400.0))


def test_the_window_autosaves_only_when_something_changed(qapp):
    window = _window()
    assert window._autosave_now() is None                 # nothing open: nothing written
    index = window._add_derived_spectrum(_spectrum(), "Sum")
    folder = window._autosave_now()
    assert folder and io_state.is_state_folder(folder)
    assert window._autosave_now() is None                 # unchanged
    window.session.spectra[index].metadata.raw["phase0"] = 30.0
    assert window._autosave_now() is not None             # changed
    labels = [e["label"] for e in io_state.read_state_info(folder)["entries"]]
    assert labels == ["Sum · 1H · derived"]
    # Autosaving adds nothing to the spectrum's own log.
    history = window._history_info(index).get("history") or {"entries": []}
    assert not any("Autosave" in e.get("text", "") for e in history["entries"])


def test_the_interval_setting_is_remembered_and_0_is_off(qapp):
    window = _window()
    assert window._autosave_timer.interval() == recovery.DEFAULT_INTERVAL_MIN * 60_000
    window.autosave_spin.setValue(0)
    assert not window._autosave_timer.isActive()
    assert window._ui_settings["autosave_minutes"] == 0
    window.autosave_spin.setValue(5)
    assert window._autosave_timer.isActive() and window._autosave_timer.interval() == 300_000
    assert _window().autosave_spin.value() == 5


def test_a_normal_close_removes_the_autosave(qapp):
    window = _window()
    window._add_derived_spectrum(_spectrum(), "Sum")
    assert window._autosave_now()
    window.close()
    assert not os.path.exists(recovery.autosave_folder())


def test_after_a_crash_the_session_is_offered_and_restored(qapp, monkeypatch):
    window = _window()
    window._add_derived_spectrum(_spectrum(), "Sum")
    assert window._autosave_now()
    # The process "dies" (no close); the next start is another process.
    lock = recovery.read_lock()
    recovery.write_lock(pid=lock["pid"] + 1)
    monkeypatch.setattr(QMessageBox, "question",
                        lambda *a, **k: QMessageBox.StandardButton.Yes)
    later = _window()                     # offers it as soon as the window is up
    assert recovery.previous_sessions()
    from PySide6.QtWidgets import QApplication
    QApplication.processEvents()
    assert len(later.session) == 1 and later.session.label(0) == "Sum · 1H · derived"
    later._state_opened_dialog.close()


def test_declined_it_stays_in_the_file_menu(qapp, monkeypatch):
    recovery.take_crashed_session(pid=100)
    recovery.write_autosave(_write)
    recovery.write_lock(pid=101)
    window = _window()
    folder = recovery.previous_sessions()[0]
    assert not window._offer_recovery(folder)            # conftest answers No
    assert recovery.previous_sessions() == [folder]
    assert window._on_recover_session_clicked() == folder
    assert len(window.session) == 1
    window._state_opened_dialog.close()
    titles = [a.text() for a in window.menuBar().actions()[0].menu().actions()]
    assert "Recover Autosaved Session…" in titles
