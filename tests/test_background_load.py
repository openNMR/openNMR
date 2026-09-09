"""The background-load thread.

What it buys is responsiveness, not speed -- roughly half a large
load's time is matplotlib work that must stay on the GUI thread. These
tests pin the boundary: the read moves off, everything else does not.

No sleeps anywhere. A threaded test that waits on a timer passes on a
fast machine and flakes on a slow one, which is exactly the
platform-dependent failure this project's threading contract forbids.
Waiting is signal-driven with a generous timeout.
"""
import os

import numpy as np
import pytest
from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtWidgets import QApplication

from opennmr.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
from opennmr.gui import loader as loader_module
from opennmr.gui.loader import BackgroundLoader, SpectrumReader, background_load_enabled
from opennmr.gui.session import Session


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _wait_for(signal, timeout_ms=60000):
    """Block until ``signal`` fires, or the timeout. Returns its args."""
    loop = QEventLoop()
    received = []

    def capture(*args):
        received.append(args)
        loop.quit()

    signal.connect(capture)
    QTimer.singleShot(timeout_ms, loop.quit)
    loop.exec()
    signal.disconnect(capture)
    return received[0] if received else None


def test_background_load_can_be_disabled_by_environment(monkeypatch):
    # A serial path has to stay supported and testable -- for CI
    # determinism and for debugging a misbehaving load.
    monkeypatch.delenv(loader_module.BACKGROUND_LOAD_ENV, raising=False)
    assert background_load_enabled() is True

    for value in ("0", "false", "no", "off", "OFF"):
        monkeypatch.setenv(loader_module.BACKGROUND_LOAD_ENV, value)
        assert background_load_enabled() is False

    # An unparseable value must not stop a load -- same rule as
    # processing.ft.fft_workers.
    monkeypatch.setenv(loader_module.BACKGROUND_LOAD_ENV, "banana")
    assert background_load_enabled() is True


def test_reader_reports_failure_instead_of_raising(qapp):
    # An exception escaping a slot on a worker thread has nowhere
    # useful to go, so a failed read becomes a signal.
    reader = SpectrumReader()
    seen = []
    reader.failed.connect(lambda *a: seen.append(a))

    reader.read("/definitely/not/a/spectrum", "bruker", {})

    assert len(seen) == 1
    assert seen[0][0] == "/definitely/not/a/spectrum"


def test_reader_emits_nothing_once_cancelled(qapp):
    reader = SpectrumReader()
    seen = []
    reader.finished.connect(lambda *a: seen.append(a))
    reader.failed.connect(lambda *a: seen.append(a))

    reader.cancel()
    reader.read("/definitely/not/a/spectrum", "bruker", {})

    assert seen == []


@pytest.mark.skipif(
    not os.path.isdir(os.path.join("test_data", "bruker_PFG", "150")),
    reason="needs a real Bruker dataset",
)
def test_load_runs_off_the_gui_thread_and_returns_the_spectrum(qapp):
    import threading

    loader = BackgroundLoader()
    try:
        worker_thread_ids = []
        original = loader._reader.read

        def record_thread(path, vendor, kwargs=None):
            worker_thread_ids.append(threading.current_thread().ident)
            return original(path, vendor, kwargs)

        loader._reader.read = record_thread
        loader.load(os.path.join("test_data", "bruker_PFG", "150"), "bruker",
                    {"pseudo_2d": True})
        result = _wait_for(loader.finished)

        assert result is not None, "the read never completed"
        path, vendor, spectrum = result
        assert vendor == "bruker"
        assert spectrum.data.ndim == 2
        # The actual point of the whole module.
        assert worker_thread_ids
        assert worker_thread_ids[0] != threading.current_thread().ident
    finally:
        loader.shutdown()


def test_shutdown_is_idempotent(qapp):
    loader = BackgroundLoader()
    loader.shutdown()
    loader.shutdown()          # must not raise or hang


def test_session_adopt_records_load_info_like_add(qapp):
    # The worker hands a spectrum over; the GUI thread puts it in the
    # session. adopt() has to leave the session indistinguishable from
    # one built by add(), or reload/reset/preview break afterwards.
    session = Session()
    axis = NMRAxis(kind="frequency", ppm=np.linspace(10, 0, 8), hz=np.linspace(4000, 0, 8))
    spectrum = NMRSpectrum(data=np.zeros(8), axes=[axis],
                           metadata=NMRMetadata(vendor="bruker"))

    session.adopt("/some/path", "bruker", {"pseudo_2d": True}, spectrum)

    assert len(session) == 1
    assert session.spectra[0] is spectrum
    assert session.get_load_kwargs(0) == {"pseudo_2d": True}


def test_initial_load_kwargs_is_what_the_worker_reads_with(qapp, tmp_path):
    # The worker must read with exactly the kwargs _load would have
    # used, or the adopted spectrum and its recorded load info
    # disagree. Both call this one helper, so they agree by
    # construction.
    from opennmr.gui.main_window import MainWindow

    window = MainWindow()
    try:
        assert window._initial_load_kwargs(str(tmp_path), "bruker") == {"pseudo_2d": True}
        # Non-Bruker vendors take no pre-read decision at all.
        assert window._initial_load_kwargs(str(tmp_path), "xpulse") == {}
    finally:
        window.close()
