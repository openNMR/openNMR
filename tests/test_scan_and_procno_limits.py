"""Folder-scan and proc-number limits (explicit user request: a head
folder with ~2000 experiments, and proc numbers above 999)."""
import pytest
from PySide6.QtWidgets import QInputDialog, QMessageBox

from opennmr.gui.main_window import MainWindow
from opennmr.gui.widgets.data_browser import DataBrowser

from test_gui import qapp, _make_1d_dataset, _make_raw_1d_dataset  # noqa: F401


def test_scan_lists_more_than_the_old_500_limit(tmp_path):
    for k in range(1, 1201):
        _make_1d_dataset(tmp_path, n=4, name=str(k))
    matches = MainWindow._scan_for_spectra(str(tmp_path))
    assert len(matches) == 1200
    assert not MainWindow._scan_was_truncated(matches)
    assert MainWindow.SCAN_LIMIT >= 2000


def test_a_truncated_scan_says_so(qapp, tmp_path, monkeypatch):  # noqa: F811
    for k in range(1, 6):
        _make_1d_dataset(tmp_path, n=4, name=str(k))
    monkeypatch.setattr(MainWindow, "SCAN_LIMIT", 3)
    matches = MainWindow._scan_for_spectra(str(tmp_path))
    assert len(matches) == 3 and MainWindow._scan_was_truncated(matches)

    browser = DataBrowser()
    browser.set_matches(str(tmp_path), matches, truncated_at=3)
    assert "stops at 3" in browser._status.text()

    window = MainWindow()
    told = []
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: told.append(a[2]))
    monkeypatch.setattr(window, "_prompt_select_spectra", lambda m, d: [])
    window._open_directory(str(tmp_path))
    assert told and "first 3" in told[0]


def test_proc_numbers_are_not_capped_at_999(qapp, tmp_path, monkeypatch):  # noqa: F811
    window = MainWindow()
    window._load(_make_raw_1d_dataset(tmp_path, monkeypatch), vendor="bruker")
    index = window._selected_index()
    seen = {}

    def fake_get_int(parent, title, label, value, minimum, maximum, *rest, **kw):
        seen.update(minimum=minimum, maximum=maximum)
        return value, False
    monkeypatch.setattr(QInputDialog, "getInt", fake_get_int)
    window._on_change_procno_action(index)
    assert seen == {"minimum": 1, "maximum": 2_147_483_647}
