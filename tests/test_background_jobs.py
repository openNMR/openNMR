"""Long fits and simulations in the background, one at a time, the rest
waiting their turn (explicit user request, 2026-11-13) -- gui/jobs.py."""

import threading
import time

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication, QMessageBox

from opennmr.core import fit_progress
from opennmr.gui import jobs
from opennmr.gui.main_window import MainWindow

from test_gui import qapp  # noqa: F401 -- fixture
from test_gui_fit_sim_state import _fit_quad, _quad_spectrum


@pytest.fixture
def background(monkeypatch):
    monkeypatch.setenv(jobs.BACKGROUND_JOBS_ENV, "1")
    for name in ("warning", "information", "critical"):
        monkeypatch.setattr(QMessageBox, name, lambda *a, **k: None)


def _wait_for(condition, seconds=60.0):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        QApplication.processEvents()
        if condition():
            return True
        time.sleep(0.01)
    return False


def _job(window, title="Test fit", index=None, gate=None, value=1, applied=None, **kwargs):
    def compute():
        if gate is not None:
            assert gate.wait(30)
        return value

    def apply(result, stopped_early):
        if applied is not None:
            applied.append((title, result, threading.current_thread() is threading.main_thread()))
    return jobs.Job(title, compute, apply, spectrum_index=index, **kwargs)


def _idle(manager):
    return manager.current is None and not manager.queue


def test_one_runs_the_next_waits_its_turn_and_the_user_is_told(qapp, background):
    window = MainWindow()
    manager = window.job_manager
    try:
        gate, applied = threading.Event(), []
        assert manager.submit(_job(window, "First", gate=gate, value=1, applied=applied))
        assert manager.submit(_job(window, "Second", value=2, applied=applied))
        assert manager.current.title == "First" and [j.title for j in manager.queue] == ["Second"]
        assert "will start when First has finished" in window.statusBar().currentMessage()
        manager.indicator.refresh()
        assert manager.indicator.isVisibleTo(window) and "1 waiting" in manager.indicator.text()
        # The window is not blocked meanwhile.
        QApplication.processEvents()
        gate.set()
        assert _wait_for(lambda: _idle(manager) and len(applied) == 2)
        # In order, and each result shown on the GUI thread.
        assert applied == [("First", 1, True), ("Second", 2, True)]
        assert not manager.indicator.isVisibleTo(window)
    finally:
        manager.shutdown()


def test_the_queue_window_and_shortcut(qapp, background):
    window = MainWindow()
    manager = window.job_manager
    try:
        assert window.jobs_menu_action.shortcut().toString() == "Ctrl+R"
        gate = threading.Event()
        manager.submit(_job(window, "First", gate=gate))
        manager.submit(_job(window, "Second"))
        window.jobs_menu_action.trigger()
        dialog = manager.dialog
        assert dialog.isVisible() and dialog.queue_list.count() == 1
        dialog.queue_list.setCurrentRow(0)
        dialog._remove()                                # Remove from Queue
        assert not manager.queue
        gate.set()
        assert _wait_for(lambda: _idle(manager))
        dialog.close()
    finally:
        manager.shutdown()


def test_a_result_waits_for_its_own_spectrum(qapp, background):
    window = MainWindow()
    manager = window.job_manager
    try:
        first = _quad_spectrum(window, "first")
        second = _quad_spectrum(window, "second")      # selected
        gate, applied = threading.Event(), []
        manager.submit(_job(window, "Fit", index=first, gate=gate, applied=applied))
        gate.set()
        assert _wait_for(lambda: _idle(manager) and manager.waiting)
        assert applied == []                            # never shown over "second"
        message = window.statusBar().currentMessage()
        label = window.session.label(first)          # "first · 27Al · derived"
        assert f"has finished: select {label} to see the result" in message
        window._select_single_row(first)
        assert _wait_for(lambda: applied)
        assert not manager.waiting
        assert window.spectrum_list.currentRow() == first
        del second
    finally:
        manager.shutdown()


def test_a_removed_spectrum_gets_no_result(qapp, background):
    window = MainWindow()
    manager = window.job_manager
    try:
        index = _quad_spectrum(window, "gone")
        gate, applied = threading.Event(), []
        manager.submit(_job(window, "Fit", index=index, gate=gate, applied=applied))
        window._remove_rows([index])
        gate.set()
        assert _wait_for(lambda: _idle(manager))
        assert applied == [] and "removed" in window.statusBar().currentMessage()
    finally:
        manager.shutdown()


def test_discard_stops_a_running_fit(qapp, background):
    window = MainWindow()
    manager = window.job_manager
    applied = []

    def compute():
        model = fit_progress.tracked(lambda x, a: a * x, np.ones(3), n_free=1)
        for _ in range(3000):                      # stopped long before the end
            model(np.ones(3), 1.0)
            time.sleep(0.01)
        return "finished"

    try:
        manager.submit(jobs.Job("Slow fit", compute, lambda r, s: applied.append(r)))
        assert _wait_for(lambda: manager.current is not None and manager.current.status is not None)
        # The fit is watched on its own thread only.
        assert not fit_progress.active()
        manager.stop_current(keep=False)
        assert _wait_for(lambda: _idle(manager))
        assert applied == [] and "Nothing was changed" in window.statusBar().currentMessage()
    finally:
        manager.shutdown()


def test_a_failure_is_reported(qapp, background, monkeypatch):
    window = MainWindow()
    manager = window.job_manager
    failures = []

    def compute():
        raise ValueError("no good")
    try:
        manager.submit(jobs.Job("Bad fit", compute, lambda r, s: None,
                                fail=lambda exc: failures.append(str(exc))))
        assert _wait_for(lambda: _idle(manager) and failures)
        assert failures == ["no good"]
    finally:
        manager.shutdown()


def test_jobs_run_at_once_when_the_background_is_off(qapp, monkeypatch):
    monkeypatch.setenv(jobs.BACKGROUND_JOBS_ENV, "0")
    window = MainWindow()
    applied = []
    assert window.job_manager.submit(_job(window, value=5, applied=applied))
    assert applied == [("Test fit", 5, True)]


def test_a_quadrupolar_fit_runs_in_the_background_and_is_shown(qapp, background):
    window = MainWindow()
    manager = window.job_manager
    try:
        _quad_spectrum(window)
        panel = window.quad_fit_panel
        assert _fit_quad(window) is panel
        assert manager.current is not None or panel.result is not None
        assert _wait_for(lambda: _idle(manager) and panel.result is not None, seconds=300)
        assert panel.result["popt"][1] == pytest.approx(4.2, rel=0.05)   # C_Q in MHz
        # Stored on its spectrum, as a fit run on the GUI thread is.
        spectrum = window.session.spectra[panel.result["index"]]
        assert "quadrupolar" in (spectrum.metadata.raw.get("fits") or {})
    finally:
        manager.shutdown()


def test_inputs_changed_while_it_ran_are_put_back(qapp, background):
    window = MainWindow()
    manager = window.job_manager
    try:
        index = _quad_spectrum(window)
        panel = window.quad_fit_panel
        state = {"value": 1}
        gate, applied = threading.Event(), []
        job = _job(window, "Fit", index=index, gate=gate, applied=applied,
                   inputs={"value": 1}, current_inputs=lambda: dict(state),
                   restore_inputs=lambda s: state.update(s))
        manager.submit(job)
        state["value"] = 2                          # edited while it ran
        gate.set()
        assert _wait_for(lambda: _idle(manager) and applied)
        assert state == {"value": 1}
        assert "put back" in window.statusBar().currentMessage()
        del panel
    finally:
        manager.shutdown()
