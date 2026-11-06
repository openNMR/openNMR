"""Opening large experiments (user request, 2026-11-06):

1. a copy of a slow read is kept (in the local cache folder, never beside
   the data) and used next time -- but only when nothing it depends on has
   changed: the experiment's files, the read settings, the openNMR code;
2. while an experiment opens, the status bar shows the stage the read is
   at, the time so far, and a Cancel that stops it between stages.

No sleeps: waiting is signal-driven, as in test_background_load.
"""

import os
import time

import numpy as np
import pytest
from PySide6.QtCore import QEventLoop, QTimer

from opennmr.core import load_progress
from opennmr.gui.loader import SpectrumReader
from opennmr.io import load_cache

import simpson_files as sf
from test_gui import qapp, _make_raw_2d_dataset  # noqa: F401 -- fixture


@pytest.fixture(autouse=True)
def _cache_settings():
    before = load_cache.configure()
    load_cache.configure(enabled=True, min_seconds=0.0, max_bytes=5 * 1024 ** 3)
    yield
    load_cache.configure(**{k: before[k] for k in ("enabled", "min_seconds", "max_bytes")})


def _experiment(tmp_path, name="expt"):
    folder = tmp_path / name
    (folder / "pdata" / "1").mkdir(parents=True)
    (folder / "acqus").write_text("##$TD= 8\n")
    (folder / "pdata" / "1" / "procs").write_text("##$SI= 8\n")
    (folder / "ser").write_bytes(b"\0" * 64)
    return str(folder)


class _Counting:
    """A stand-in reader: counts its calls, returns a fresh array each time."""

    def __init__(self):
        self.calls = 0

    def __call__(self, path, vendor, kwargs):
        self.calls += 1
        return {"data": np.arange(10.0) * (1 + kwargs.get("scale", 0)), "path": path}


# ---- the cache --------------------------------------------------------------------

def test_cache_lives_in_the_test_folder_not_the_users(tmp_path):
    assert load_cache.cache_dir().startswith(str(tmp_path))


def test_a_second_open_uses_the_copy(tmp_path):
    path, reader = _experiment(tmp_path), _Counting()
    first, info = load_cache.load(path, "bruker", {}, loader=reader)
    assert info["from_copy"] is False and info["kept"] is True
    second, info = load_cache.load(path, "bruker", {}, loader=reader)
    assert reader.calls == 1 and info["from_copy"] is True
    assert np.array_equal(first["data"], second["data"])
    assert second is not first                       # a fresh object each time


def test_a_changed_file_is_read_again(tmp_path):
    path, reader = _experiment(tmp_path), _Counting()
    load_cache.load(path, "bruker", {}, loader=reader)
    ser = os.path.join(path, "ser")
    with open(ser, "ab") as handle:
        handle.write(b"\0" * 8)                      # new size
    assert load_cache.load(path, "bruker", {}, loader=reader)[1]["from_copy"] is False
    stat = os.stat(os.path.join(path, "pdata", "1", "procs"))
    os.utime(os.path.join(path, "pdata", "1", "procs"),
             ns=(stat.st_atime_ns, stat.st_mtime_ns + 10 ** 9))   # same size, new time
    assert load_cache.load(path, "bruker", {}, loader=reader)[1]["from_copy"] is False
    assert reader.calls == 3


def test_other_read_settings_are_another_copy(tmp_path):
    path, reader = _experiment(tmp_path), _Counting()
    load_cache.load(path, "bruker", {"scale": 0}, loader=reader)
    got, info = load_cache.load(path, "bruker", {"scale": 1}, loader=reader)
    assert info["from_copy"] is False and got["data"][1] == 2.0
    assert load_cache.load(path, "bruker", {"scale": 0}, loader=reader)[1]["from_copy"]
    assert load_cache.load(path, "bruker", {"scale": 1}, loader=reader)[1]["from_copy"]
    assert reader.calls == 2


def test_array_settings_are_compared_by_value(tmp_path):
    path, reader = _experiment(tmp_path), _Counting()
    big = np.zeros(5000)
    load_cache.load(path, "bruker", {"weights": big}, loader=reader)
    changed = big.copy()
    changed[2500] = 1.0                              # repr() would not show this
    assert load_cache.load(path, "bruker", {"weights": changed}, loader=reader)[1][
        "from_copy"] is False


def test_saving_your_processing_does_not_void_the_copy(tmp_path):
    path, reader = _experiment(tmp_path), _Counting()
    load_cache.load(path, "bruker", {}, loader=reader)
    for name in ("opennmr_processing.json", "opennmr_history.json", "opennmr_log.txt",
                 "opennmr_metadata.json"):
        (tmp_path / "expt" / name).write_text("{}")
    os.makedirs(os.path.join(path, "opennmr_savepoints"))
    (tmp_path / "expt" / "opennmr_savepoints" / "1.npz").write_bytes(b"x")
    assert load_cache.load(path, "bruker", {}, loader=reader)[1]["from_copy"] is True


def test_a_new_data_file_does_void_it(tmp_path):
    path, reader = _experiment(tmp_path), _Counting()
    load_cache.load(path, "bruker", {}, loader=reader)
    (tmp_path / "expt" / "vdlist").write_text("1\n2\n")
    assert load_cache.load(path, "bruker", {}, loader=reader)[1]["from_copy"] is False


def test_changed_code_voids_every_copy(tmp_path, monkeypatch):
    path, reader = _experiment(tmp_path), _Counting()
    load_cache.load(path, "bruker", {}, loader=reader)
    monkeypatch.setattr(load_cache, "_code_fingerprint", "a different reader")
    assert load_cache.load(path, "bruker", {}, loader=reader)[1]["from_copy"] is False


def test_fast_reads_and_a_disabled_cache_keep_nothing(tmp_path, monkeypatch):
    path, reader = _experiment(tmp_path), _Counting()
    load_cache.configure(min_seconds=60.0)
    assert load_cache.load(path, "bruker", {}, loader=reader)[1]["kept"] is False
    load_cache.configure(min_seconds=0.0)
    monkeypatch.setenv("OPENNMR_LOAD_CACHE", "0")
    assert load_cache.load(path, "bruker", {}, loader=reader)[1]["kept"] is False
    assert load_cache.copies() == []


def test_a_damaged_copy_is_read_again(tmp_path):
    path, reader = _experiment(tmp_path), _Counting()
    load_cache.load(path, "bruker", {}, loader=reader)
    (copy_path, _size, _used), = load_cache.copies()
    with open(copy_path, "r+b") as handle:
        handle.truncate(20)
    got, info = load_cache.load(path, "bruker", {}, loader=reader)
    assert info["from_copy"] is False and reader.calls == 2


def test_the_oldest_copies_go_first(tmp_path):
    reader = _Counting()
    paths = [_experiment(tmp_path, f"e{k}") for k in range(3)]
    for k, path in enumerate(paths):
        load_cache.load(path, "bruker", {}, loader=reader)
        copy_path = next(p for p, _s, _u in load_cache.copies())
        os.utime(copy_path, (time.time() - 100 + k, time.time() - 100 + k))
    sizes = [size for _p, size, _u in load_cache.copies()]
    load_cache.prune(sum(sizes) - 1)                 # room for all but one
    assert len(load_cache.copies()) == 2
    assert load_cache.load(paths[0], "bruker", {}, loader=reader)[1]["from_copy"] is False
    assert load_cache.load(paths[2], "bruker", {}, loader=reader)[1]["from_copy"] is True
    assert load_cache.clear() > 0 and load_cache.copies() == []


def test_a_real_read_comes_back_identical(tmp_path):
    path = sf.write_spe(tmp_path / "s.spe")
    kwargs = {"nucleus": "13C", "field": 9.4}
    fresh, info = load_cache.load(path, "simpson", kwargs)
    assert info["kept"]
    copy, info = load_cache.load(path, "simpson", kwargs)
    assert info["from_copy"]
    assert np.array_equal(fresh.data, copy.data)
    assert np.array_equal(fresh.axes[0].ppm, copy.axes[0].ppm)
    assert fresh.metadata.nucleus == copy.metadata.nucleus
    assert fresh.metadata.sfo1 == copy.metadata.sfo1


# ---- stage reports ------------------------------------------------------------------

def test_stage_is_silent_outside_a_report():
    load_progress.stage("nothing listens")           # must not raise


def test_stages_reach_the_callback_and_can_cancel():
    seen = []
    with load_progress.reporting(seen.append):
        load_progress.stage("one")
    assert seen == ["one"]

    def refuse(_text):
        raise load_progress.LoadCancelled()

    with pytest.raises(load_progress.LoadCancelled):
        with load_progress.reporting(refuse):
            load_progress.stage("two")
    load_progress.stage("after")                     # the reporter is gone again


def test_a_2d_read_reports_its_stages(tmp_path, monkeypatch):
    from opennmr.io import bruker
    path = _make_raw_2d_dataset(tmp_path, monkeypatch, n_f1=4)
    seen = []
    with load_progress.reporting(seen.append):
        bruker.read_2d(path, pseudo_2d=False)
    text = " | ".join(seen)
    assert "Reading ser" in text and "digital filter" in text
    assert "Fourier transform (F2" in text and "Fourier transform (F1)" in text
    assert seen.index("Removing the digital filter") < next(
        k for k, t in enumerate(seen) if t.startswith("Fourier transform (F2"))


# ---- the reader thread's side -----------------------------------------------------------

def test_reader_reports_stages_and_timing_then_the_spectrum(qapp, tmp_path):  # noqa: F811
    path = sf.write_spe(tmp_path / "s.spe")
    reader = SpectrumReader()
    events = []
    reader.progress.connect(lambda p, t: events.append(("progress", t)))
    reader.timing.connect(lambda p, info: events.append(("timing", info["from_copy"])))
    reader.finished.connect(lambda *a: events.append(("finished",)))
    reader.read(path, "simpson", {})
    kinds = [e[0] for e in events]
    assert "progress" in kinds and kinds[-2:] == ["timing", "finished"]
    # Nothing saved yet, so no "Reading the saved copy" stage.
    assert ("progress", "Reading the saved copy") not in events


def test_a_dropped_request_stops_at_the_next_stage(qapp, tmp_path, monkeypatch):  # noqa: F811
    reader = SpectrumReader()
    stages_run = []

    def slow_load(path, vendor, kwargs):
        for name in ("first", "second", "third"):
            load_progress.stage(name)
            stages_run.append(name)
        return object(), {"from_copy": False, "seconds": 0.0}

    monkeypatch.setattr(load_cache, "load", slow_load)
    events = []
    reader.cancelled.connect(lambda *a: events.append("cancelled"))
    reader.finished.connect(lambda *a: events.append("finished"))
    reader.failed.connect(lambda *a: events.append("failed"))

    def drop_after_first(_path, text):
        if text == "first":
            reader._dropped.add(7)

    reader.progress.connect(drop_after_first)
    reader.read("/x", "bruker", {"_request_id": 7})
    assert events == ["cancelled"] and stages_run == ["first"]
    # A request that was dropped before it started never runs at all.
    stages_run.clear()
    reader.read("/x", "bruker", {"_request_id": 7})
    assert events == ["cancelled", "cancelled"] and stages_run == []


# ---- the window -----------------------------------------------------------------------

def _wait(predicate, timeout_ms=60000):
    loop = QEventLoop()
    timer = QTimer()
    timer.setInterval(20)
    timer.timeout.connect(lambda: predicate() and loop.quit())
    timer.start()
    QTimer.singleShot(timeout_ms, loop.quit)
    loop.exec()
    timer.stop()
    return predicate()


def test_window_opens_in_the_background_and_says_so(qapp, tmp_path, monkeypatch):  # noqa: F811
    from opennmr.gui.main_window import MainWindow
    monkeypatch.setenv("OPENNMR_BACKGROUND_LOAD", "1")
    window = MainWindow()
    try:
        path = sf.write_spe(tmp_path / "s.spe")
        window._open_one(path, "simpson")
        row = window._load_progress["row"]
        assert row.isVisibleTo(window)                    # shown straight away
        assert "s.spe" in window._load_progress["label"].text()
        assert _wait(lambda: len(window.session) == 1)
        assert not row.isVisibleTo(window)                # gone once it is drawn
        assert window._pending_loads == 0
    finally:
        window.close()


def test_window_cancel_stops_the_open(qapp, tmp_path, monkeypatch):  # noqa: F811
    from opennmr.gui.main_window import MainWindow
    monkeypatch.setenv("OPENNMR_BACKGROUND_LOAD", "1")

    def slow_load(path, vendor, kwargs):
        for k in range(200):
            load_progress.stage(f"step {k}")
            time.sleep(0.01)
        raise AssertionError("should have been stopped")

    monkeypatch.setattr(load_cache, "load", slow_load)
    window = MainWindow()
    (tmp_path / "big").mkdir()
    try:
        window._open_one(str(tmp_path / "big"), "bruker")
        assert _wait(lambda: "step" in window._load_progress["label"].text())
        window._load_progress["cancel"].click()
        assert _wait(lambda: window._pending_loads == 0)
        assert len(window.session) == 0
        assert "Stopped opening big" in window.statusBar().currentMessage()
        assert not window._load_progress["row"].isVisibleTo(window)
    finally:
        window.close()


def test_load_time_messages(qapp):  # noqa: F811
    from opennmr.gui.main_window import MainWindow
    from opennmr.gui.session import Session
    window = MainWindow()
    key = Session._path_key("/data/2037")
    window._load_infos[key] = {"from_copy": True, "seconds": 1.4, "first_seconds": 14.2}
    assert window._load_time_message("/data/2037") == (
        "Opened 2037 from its saved copy in 1.4 s (reading it took 14 s).")
    window._load_infos[key] = {"from_copy": False, "seconds": 14.2, "kept": True}
    assert "A copy is kept" in window._load_time_message("/data/2037")
    window._load_infos[key] = {"from_copy": False, "seconds": 0.3, "kept": False}
    assert window._load_time_message("/data/2037") is None


def test_settings_control_the_cache(qapp, tmp_path):  # noqa: F811
    from opennmr.gui.main_window import MainWindow
    window = MainWindow()
    assert window.load_cache_checkbox.isChecked()
    assert str(tmp_path) in window.load_cache_location_label.text()
    window.load_cache_checkbox.setChecked(False)
    assert load_cache.enabled() is False
    assert window._ui_settings["load_cache_enabled"] is False
    window.load_cache_checkbox.setChecked(True)
    window.load_cache_size_spin.setValue(2.0)
    window.load_cache_size_spin.editingFinished.emit()
    assert load_cache.configure()["max_bytes"] == 2 * 1024 ** 3
    path, reader = _experiment(tmp_path), _Counting()
    load_cache.configure(min_seconds=0.0)
    load_cache.load(path, "bruker", {}, loader=reader)
    window._refresh_load_cache_label()
    assert window.load_cache_clear_button.isEnabled()
    window.load_cache_clear_button.click()
    assert load_cache.copies() == []
    assert not window.load_cache_clear_button.isEnabled()
