"""Settings ▸ Updates (gui/update_panel.py over opennmr.update): find the
newest version from Settings, with the right next step for the app (a
download) or a pip install (a command). Explicit user request, 2026-10-15.

The network is never touched: ``update.check`` is patched throughout."""
import json
import time

import pytest

from opennmr import update
from opennmr.gui import update_panel


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(qapp):
    from opennmr.gui.main_window import MainWindow
    return MainWindow()


def _newer(version="9.9.9", url="https://dl/openNMR-setup.exe"):
    info = update.UpdateInfo(version=version, url=url, notes="Faster fits.")
    return update.UpdateCheck("newer", "0.1.0", info=info, latest_version=version,
                              release_page="https://github.com/o/r/releases/tag/v9.9.9")


def _patch_check(monkeypatch, result, calls=None):
    def fake(source=None, token=None, **_kwargs):
        if calls is not None:
            calls.append((source, token))
        return result
    monkeypatch.setattr(update, "check", fake)


def test_settings_has_updates_section(window):
    assert window.updates_section.title() == "Updates"
    assert window.updates_panel.isAncestorOf(window.updates_panel.check_button)
    panel = window.updates_panel
    assert update.__version__ in panel.version_label.text()
    assert panel.source() == update.default_source(update.install_kind())


def test_startup_check_is_opt_in(window):
    """A network call nobody asked for: off until the user turns it on."""
    panel = update_panel.UpdatesPanel({}, parent=window)
    assert not panel.startup_checkbox.isChecked()
    assert panel.startup_check_due() is False
    panel = update_panel.UpdatesPanel({"update_check_at_start": True}, parent=window)
    assert panel.startup_checkbox.isChecked()


def test_check_now_newer_app_shows_download(window, monkeypatch):
    monkeypatch.setattr(update_panel.update, "install_kind", lambda: "app")
    panel = update_panel.UpdatesPanel({}, parent=window)
    panel.background = False
    opened = []
    monkeypatch.setattr(panel, "_open_url", opened.append)
    _patch_check(monkeypatch, _newer())
    panel.check_now()
    assert "9.9.9 is available" in panel.status_label.text()
    assert "Faster fits." in panel.status_label.text()
    assert not panel.download_button.isHidden()
    assert panel.command_row.isHidden()          # the app has no pip command
    panel.download_button.click()
    assert opened == ["https://dl/openNMR-setup.exe"]


@pytest.mark.parametrize("kind, command", [
    ("pip", "pip install --upgrade opennmr"),
    ("git", "pip install --upgrade git+https://github.com/openNMR/openNMR.git"),
    ("source", "git pull"),
])
def test_check_now_newer_pip_shows_command(window, monkeypatch, kind, command):
    monkeypatch.setattr(update_panel.update, "install_kind", lambda: kind)
    panel = update_panel.UpdatesPanel({}, parent=window)
    panel.background = False
    _patch_check(monkeypatch, _newer())
    panel.check_now()
    assert panel.download_button.isHidden()      # a notebook install is not updated by an installer
    assert not panel.command_row.isHidden()
    assert panel.command_edit.text() == command
    panel.copy_command_button.click()
    from PySide6.QtGui import QGuiApplication
    assert QGuiApplication.clipboard().text() == command


def test_check_now_current_and_error(window, monkeypatch):
    panel = window.updates_panel
    panel.background = False
    _patch_check(monkeypatch, update.UpdateCheck("current", "0.1.0", latest_version="0.1.0"))
    panel.check_now()
    assert panel.status_label.text().startswith("Up to date")
    assert panel.download_button.isHidden()
    _patch_check(monkeypatch, update.UpdateCheck("error", "0.1.0", message="no network"))
    panel.check_now()
    assert "Could not check" in panel.status_label.text() and "no network" in panel.status_label.text()
    assert panel.check_button.isEnabled()


def test_source_and_token_are_used_and_remembered(window, monkeypatch):
    calls = []
    _patch_check(monkeypatch, update.UpdateCheck("current", "0.1.0"), calls)
    panel = window.updates_panel
    panel.background = False
    panel.source_edit.setText("someone/fork")
    panel._on_source_edited()
    panel.token_edit.setText("tok")
    panel._on_token_edited()
    panel.check_now()
    assert calls[-1] == ("someone/fork", "tok")
    saved = json.load(open(window._UI_SETTINGS_PATH, encoding="utf-8"))
    assert saved["update_source"] == "someone/fork"
    assert saved["update_token"] == "tok"
    assert saved["update_last_checked"] > 0
    panel._on_source_reset()
    saved = json.load(open(window._UI_SETTINGS_PATH, encoding="utf-8"))
    assert "update_source" not in saved           # the default is not stored


def test_startup_check_is_quiet_daily_and_switchable(window, monkeypatch):
    panel = update_panel.UpdatesPanel({"update_check_at_start": True}, parent=window)
    panel.background = False
    monkeypatch.delenv(update.STARTUP_CHECK_ENV, raising=False)
    assert panel.startup_check_due()
    _patch_check(monkeypatch, update.UpdateCheck("error", "0.1.0", message="offline"))
    messages = []
    panel._notify = messages.append
    assert panel.maybe_check_at_startup() is True
    assert messages == []                          # a failed start-up check is not announced
    assert panel.maybe_check_at_startup() is False  # once a day
    later = time.time() + update_panel.STARTUP_INTERVAL_S + 1
    assert panel.startup_check_due(now=later)
    panel.startup_checkbox.setChecked(False)
    assert not panel.startup_check_due(now=later)
    panel.startup_checkbox.setChecked(True)
    monkeypatch.setenv(update.STARTUP_CHECK_ENV, "0")
    assert not panel.startup_check_due(now=later)


def test_startup_check_announces_newer_in_status_bar(window, monkeypatch):
    panel = window.updates_panel
    panel.background = False
    panel.startup_checkbox.setChecked(True)               # opt-in
    monkeypatch.delenv(update.STARTUP_CHECK_ENV, raising=False)
    panel._settings.pop("update_last_checked", None)
    _patch_check(monkeypatch, _newer("2.0.0"))
    assert panel.maybe_check_at_startup()
    assert "2.0.0 is available" in window.statusBar().currentMessage()


def test_background_check_lands_on_gui_thread(window, monkeypatch, qapp):
    panel = update_panel.UpdatesPanel({}, parent=window)
    panel.background = True
    _patch_check(monkeypatch, update.UpdateCheck("current", "0.1.0"))
    seen = []
    panel.checkFinished.connect(seen.append)
    panel.check_now()
    deadline = time.time() + 5
    while not seen and time.time() < deadline:
        qapp.processEvents()
        time.sleep(0.01)
    assert seen and seen[0].status == "current"
    assert panel.check_button.isEnabled()


def test_no_startup_check_under_test_environment(window):
    # conftest sets OPENNMR_UPDATE_CHECK=0: building windows never touches the network.
    assert not window.updates_panel.startup_check_due()
