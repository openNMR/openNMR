"""Folder scans that are fast, cloud-safe and remembered (explicit user
request, 2026-11-13: a large default data folder made openNMR take
minutes -- up to an hour on older machines -- to open)."""

import os
import threading
import time
import types

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from opennmr.io import base, cloud
from opennmr.io import scan as io_scan
from opennmr.gui.main_window import MainWindow
from opennmr.gui.widgets.data_browser import DataBrowser

from test_gui import qapp, _make_1d_dataset  # noqa: F401


OLD = 1_600_000_000  # a fixed past time for folder modification times


def _age(root):
    """Give every folder under ``root`` an old modification time, so a
    change made afterwards is unmistakable on any file system."""
    for folder, dirs, _files in os.walk(root):
        os.utime(folder, (OLD, OLD))


def _xpulse(path, text="##TITLE= t\n##$SFO1= 400.13\n"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return str(path)


def _mixed_tree(tmp_path):
    """Bruker sample folders, X-Pulse files with per-scan copies and a
    redundant raw_slices/, notes and images, and nested folders."""
    root = tmp_path / "data"
    for sample in ("sampleA", "sampleB"):
        for name in ("1", "2", "10"):
            _make_1d_dataset(root / sample, n=4, name=name)
    expt = root / "xpulse" / "run1"
    _xpulse(expt / "s_avg.jdx")
    for i in range(1, 4):
        _xpulse(expt / "raw" / f"s_{i}_raw.jdx")
    hsqc = root / "xpulse" / "hsqc"
    _xpulse(hsqc / "h_avg.jdx", "##TITLE= t\n##DATACLASS= NTUPLES\n##NUMDIM= 2\n##$SFO1= 400.13\n")
    for i in range(1, 4):
        _xpulse(hsqc / "raw_slices" / f"{i:04d}_slice.jdx")
    (root / "notes.txt").write_text("hello")
    (root / "photo.png").write_bytes(b"\x89PNG....")
    (root / "deep" / "er").mkdir(parents=True)
    _make_1d_dataset(root / "deep" / "er", n=4, name="5")
    return str(root)


# ------------------------------------------------------------------
# What is found
# ------------------------------------------------------------------

def test_the_scan_finds_the_same_experiments_in_natural_order(tmp_path):
    root = _mixed_tree(tmp_path)
    found = io_scan.scan_for_spectra(root)
    relative = [(os.path.relpath(p, root).replace(os.sep, "/"), v) for p, v in found]
    assert relative == [
        ("deep/er/5", "bruker"),
        ("sampleA/1", "bruker"), ("sampleA/2", "bruker"), ("sampleA/10", "bruker"),
        ("sampleB/1", "bruker"), ("sampleB/2", "bruker"), ("sampleB/10", "bruker"),
        ("xpulse/hsqc/h_avg.jdx", "xpulse"),
        ("xpulse/run1/s_avg.jdx", "xpulse"),
    ]
    assert found == MainWindow._scan_for_spectra(root)       # the window uses the same scan
    # Dates for sorting came with the scan.
    assert set(found.dates) == {path for path, _vendor in found}


def test_every_folder_reader_is_found_through_the_marker_check(tmp_path):
    """A folder is only handed to the readers when it holds a marker name
    (io.scan.DIR_MARKERS) -- each kind of folder experiment must still be
    found, with the reader detect_vendor itself gives it."""
    root = tmp_path / "kinds"
    bruker = _make_1d_dataset(root, n=4, name="bruker")
    varian = root / "varian.fid"
    varian.mkdir(parents=True)
    (varian / "procpar").write_text("x")
    (varian / "fid").write_bytes(b"\0" * 64)
    spinsolve = root / "spinsolve"
    spinsolve.mkdir()
    (spinsolve / "acqu.par").write_text("x")
    (spinsolve / "data.1d").write_bytes(b"\0" * 64)
    dataset = root / "dataset"
    dataset.mkdir()
    (dataset / "opennmr_spectrum.npz").write_bytes(b"x")
    (dataset / "opennmr_spectrum.json").write_text("{}")
    arrayed = root / "arrayed"
    (arrayed / "array1").mkdir(parents=True)
    expected = {str(p): base.detect_vendor(str(p))
                for p in (bruker, varian, spinsolve, dataset, arrayed)}
    assert set(expected.values()) == {"bruker", "varian", "spinsolve", "opennmr", "xpulse"}
    assert dict(io_scan.scan_for_spectra(str(root))) == expected


def test_a_plain_folder_and_unrelated_files_are_never_handed_to_the_readers(tmp_path, monkeypatch):
    root = _mixed_tree(tmp_path)
    asked = []
    real = base.detect_vendor
    monkeypatch.setattr(base, "detect_vendor", lambda path: asked.append(path) or real(path))
    io_scan.scan_for_spectra(root)
    names = {os.path.basename(p) for p in asked}
    # Notes and images are not opened (the SIMPSON reader used to read
    # the first line of every file), and plain folders are not asked.
    assert "notes.txt" not in names and "photo.png" not in names
    assert not any(os.path.basename(p) in ("sampleA", "deep", "er", "xpulse", "data")
                   for p in asked)


def test_cloud_only_files_are_never_opened(tmp_path, monkeypatch):
    root = tmp_path / "cloud"
    root.mkdir()
    for name in ("a.jdx", "b.jdf", "c.csv", "d.spe"):
        (root / name).write_text("##TITLE= x\n")
    monkeypatch.setattr(cloud, "is_dataless_stat", lambda info: True)
    monkeypatch.setattr(cloud, "is_dataless", lambda path: True)

    def refuse(path):
        raise AssertionError(f"opened {path}")
    monkeypatch.setattr(base, "detect_vendor", refuse)
    found = {os.path.basename(p): v for p, v in io_scan.scan_for_spectra(str(root))}
    # Recognised by extension; a cloud-only SIMPSON file is not listed.
    assert found == {"a.jdx": "xpulse", "b.jdf": "jeol", "c.csv": "text"}


def test_windows_online_only_files_are_recognised():
    online_only = types.SimpleNamespace(st_file_attributes=0x00400000, st_size=10)
    local = types.SimpleNamespace(st_file_attributes=0x20, st_size=10)
    assert cloud.is_dataless_stat(online_only)
    assert not cloud.is_dataless_stat(local)


def test_the_limit_and_stop(tmp_path):
    root = _mixed_tree(tmp_path)
    found = io_scan.scan_for_spectra(root, max_matches=3)
    assert len(found) == 3 and found.truncated and not found.cancelled
    stopped = io_scan.scan_for_spectra(root, cancel=lambda: True)
    assert stopped.cancelled and list(stopped) == []


# ------------------------------------------------------------------
# Remembered between sessions
# ------------------------------------------------------------------

def test_an_unchanged_folder_is_not_examined_again(tmp_path, monkeypatch):
    root = _mixed_tree(tmp_path)
    _age(root)
    index = io_scan.ScanIndex.load(root)
    first = io_scan.scan_for_spectra(root, index=index)
    index.save(first)
    assert first.listed > 0

    examined = []
    real = io_scan._list_dir
    monkeypatch.setattr(io_scan, "_list_dir", lambda p, m: examined.append(p) or real(p, m))
    again = io_scan.scan_for_spectra(root, index=io_scan.ScanIndex.load(root))
    assert examined == [] and again.listed == 0 and again.reused == first.listed
    assert again == first and again.dates == first.dates


def test_new_and_removed_experiments_are_found(tmp_path):
    root = _mixed_tree(tmp_path)
    _age(root)
    index = io_scan.ScanIndex.load(root)
    index.save(io_scan.scan_for_spectra(root, index=index))

    new = _make_1d_dataset(tmp_path / "data" / "sampleA", n=4, name="11")
    import shutil
    shutil.rmtree(os.path.join(root, "sampleB", "2"))
    index = io_scan.ScanIndex.load(root)
    found = io_scan.scan_for_spectra(root, index=index)
    index.save(found)
    paths = [p for p, _v in found]
    assert new in paths and os.path.join(root, "sampleB", "2") not in paths
    assert len(found) == 9
    # Only the folders that changed were examined.
    assert found.listed <= 4
    # And the remembered list is the new one, ready for next time.
    assert io_scan.ScanIndex.load(root).matches == list(found)


def test_the_index_survives_damage_and_is_kept_out_of_the_data(tmp_path):
    root = _mixed_tree(tmp_path)
    path = io_scan.ScanIndex.path_for(root)
    assert not path.startswith(root)            # never beside the data (a synced folder)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        handle.write("{ not json")
    assert io_scan.ScanIndex.load(root).matches == []
    index = io_scan.ScanIndex.load(root)
    index.save(io_scan.scan_for_spectra(root, index=index))
    assert len(io_scan.ScanIndex.load(root).matches) == 9
    io_scan.ScanIndex.forget(root)
    assert io_scan.ScanIndex.load(root).matches == []


# ------------------------------------------------------------------
# The Open panel
# ------------------------------------------------------------------

def test_updating_the_list_keeps_what_is_expanded_and_selected(qapp, tmp_path):
    root = _mixed_tree(tmp_path)
    found = io_scan.scan_for_spectra(root)
    browser = DataBrowser()
    browser.set_matches(root, found, dates=found.dates)
    folder = browser._folders["sampleA"]
    folder.setExpanded(True)
    leaf = browser._leaves[os.path.join(root, "sampleA", "2")]
    leaf.setSelected(True)
    new = (os.path.join(root, "sampleA", "11"), "bruker")
    gone = os.path.join(root, "deep", "er", "5")
    added, removed = browser.update_matches([m for m in found if m[0] != gone] + [new])
    assert (added, removed) == (1, 1)
    assert folder.isExpanded() and leaf.isSelected()
    assert browser.ordered_names(folder) == ["1", "2", "10", "11"]
    assert "deep" not in browser._folders            # the emptied folders went too
    assert len(browser.matches()) == 9 and "9 experiments" in browser._status.text()


def test_synchronous_browse_uses_and_updates_the_index(qapp, tmp_path):
    root = _mixed_tree(tmp_path)
    window = MainWindow()
    window._browse_folder(root)
    assert len(window.data_browser.matches()) == 9
    assert len(io_scan.ScanIndex.load(root).matches) == 9


def _wait_for(condition, seconds=20.0):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        QApplication.processEvents()
        if condition():
            return True
        time.sleep(0.01)
    return False


def test_the_window_shows_the_remembered_list_at_once_and_scans_in_the_background(
        qapp, tmp_path, monkeypatch):
    root = _mixed_tree(tmp_path)
    _age(root)
    index = io_scan.ScanIndex.load(root)
    index.save(io_scan.scan_for_spectra(root, index=index))
    new = _make_1d_dataset(tmp_path / "data" / "sampleB", n=4, name="12")

    monkeypatch.setenv("OPENNMR_BACKGROUND_SCAN", "1")
    release = threading.Event()
    real = io_scan.scan_for_spectra

    def held(*args, **kwargs):
        release.wait(10)
        return real(*args, **kwargs)
    monkeypatch.setattr(io_scan, "scan_for_spectra", held)

    window = MainWindow()
    try:
        started = time.monotonic()
        window._browse_folder(root)
        assert time.monotonic() - started < 2            # never waits for the scan
        browser = window.data_browser
        assert len(browser.matches()) == 9               # last time's list, at once
        assert browser._scan_row.isVisibleTo(browser)
        assert new not in dict(browser.matches())
        release.set()
        assert _wait_for(lambda: not window._folder_scanner.is_scanning())
        QApplication.processEvents()
        assert new in dict(browser.matches())            # found, and added in place
        assert not browser._scan_row.isVisibleTo(browser)
        assert "1 new experiment" in window.statusBar().currentMessage()
    finally:
        release.set()
        window._folder_scanner.shutdown()


def test_refresh_and_stop(qapp, tmp_path, monkeypatch):
    root = _mixed_tree(tmp_path)
    monkeypatch.setenv("OPENNMR_BACKGROUND_SCAN", "1")
    window = MainWindow()
    try:
        window._browse_folder(root)
        assert _wait_for(lambda: not window._folder_scanner.is_scanning())
        assert len(window.data_browser.matches()) == 9
        # Refresh looks again from scratch (the remembered scan ignored).
        calls = []
        real = io_scan.scan_for_spectra
        monkeypatch.setattr(io_scan, "scan_for_spectra",
                            lambda *a, **k: calls.append(dict(k["index"].records)) or real(*a, **k))
        window.data_browser.refreshRequested.emit()
        assert _wait_for(lambda: not window._folder_scanner.is_scanning())
        assert calls == [{}]
        # Stop: the list stays as it is.
        window._folder_scanner.cancel()
        assert len(window.data_browser.matches()) == 9
    finally:
        window._folder_scanner.shutdown()


def test_the_scan_is_in_the_notebook(tmp_path):
    from opennmr import notebook as nmr
    root = _mixed_tree(tmp_path)
    index = nmr.ScanIndex.load(root)
    found = nmr.find_experiments(root, index=index)
    index.save(found)
    assert len(found) == 9 and len(nmr.ScanIndex.load(root).matches) == 9
