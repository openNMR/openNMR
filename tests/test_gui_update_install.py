"""Settings ▸ Updates ▸ Install (gui/update_panel.py over
opennmr.update_install), 2026-10-20: one click downloads, checks and
installs the update. Nothing real runs: the download, the hand-off, the
confirmation box, opening files and quitting are all stood in for."""
import pytest

from opennmr import update, update_install
from opennmr.gui import update_panel


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


class Spy:
    """Records what the panel did in place of the real thing."""

    def __init__(self):
        self.launched, self.opened, self.exited, self.asked = [], [], [], []


def _panel(qapp, monkeypatch, kind="app", answer=True):
    monkeypatch.setattr(update_panel.update, "install_kind", lambda: kind)
    panel = update_panel.UpdatesPanel({})
    panel.background = False
    spy = Spy()
    monkeypatch.setattr(panel, "_confirm", lambda title, text: spy.asked.append(text) or answer)
    monkeypatch.setattr(panel, "_launch", spy.launched.append)
    monkeypatch.setattr(panel, "_open_path", spy.opened.append)
    monkeypatch.setattr(panel, "_exit_app", lambda: spy.exited.append(True))
    info = update.UpdateInfo(version="9.9.9", url="https://dl/openNMR-9.9.9-windows-setup.exe",
                             asset_name="openNMR-9.9.9-windows-setup.exe", size=250_000_000,
                             sha256="a" * 64)
    panel.last_result = update.UpdateCheck("newer", "0.1.0", info=info, latest_version="9.9.9",
                                           release_page="https://github.com/o/r/releases/tag/v9.9.9")
    monkeypatch.setattr(update, "check", lambda *a, **k: panel.last_result)
    panel.check_now()
    return panel, spy


def test_the_app_gets_an_install_button_and_keeps_the_browser_download(qapp, monkeypatch):
    panel, _ = _panel(qapp, monkeypatch)
    assert not panel.install_button.isHidden()
    assert "9.9.9" in panel.install_button.text()
    assert not panel.download_button.isHidden()           # the fallback is still there


def test_install_downloads_then_hands_over_and_quits(qapp, monkeypatch):
    panel, spy = _panel(qapp, monkeypatch)
    plan = update_install.InstallPlan("windows", ["setup.exe", "/SILENT"])
    fetched = {}

    def fake_fetch(info, **kwargs):
        fetched["info"], fetched["token"] = info, kwargs.get("token")
        kwargs["progress"](50, 100)
        return "C:/tmp/setup.exe"
    monkeypatch.setattr(update_install, "fetch_installer", fake_fetch)
    monkeypatch.setattr(update_install, "plan_install", lambda path, **k: plan)
    panel.token_edit.setText("tok")
    panel.install_button.click()
    assert "250 MB" in spy.asked[0] and "9.9.9" in spy.asked[0]
    assert fetched["info"].version == "9.9.9" and fetched["token"] == "tok"
    assert spy.launched == [plan]
    assert spy.exited == [True]
    assert "close and start again" in panel.status_label.text()


def test_declining_the_confirmation_does_nothing(qapp, monkeypatch):
    panel, spy = _panel(qapp, monkeypatch, answer=False)
    monkeypatch.setattr(update_install, "fetch_installer",
                        lambda *a, **k: pytest.fail("must not download after a No"))
    panel.install_button.click()
    assert spy.launched == [] and spy.exited == []
    assert panel.install_button.isEnabled()


def test_an_unverifiable_release_shows_why_and_keeps_the_browser_download(qapp, monkeypatch):
    panel, spy = _panel(qapp, monkeypatch)

    def refuse(info, **kwargs):
        raise update_install.InstallError("This release does not publish a checksum.")
    monkeypatch.setattr(update_install, "fetch_installer", refuse)
    panel.install_button.click()
    assert "did not install" in panel.status_label.text() and "checksum" in panel.status_label.text()
    assert spy.launched == [] and spy.exited == []          # the app stays open
    assert panel.install_button.isEnabled() and panel.check_button.isEnabled()
    assert panel.progress_bar.isHidden()


def test_an_unexpected_error_is_shown_not_lost(qapp, monkeypatch):
    panel, spy = _panel(qapp, monkeypatch)
    monkeypatch.setattr(update_install, "fetch_installer",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    panel.install_button.click()
    assert "boom" in panel.status_label.text() and spy.exited == []


def test_a_manual_plan_opens_the_installer_and_keeps_openNMR_running(qapp, monkeypatch):
    panel, spy = _panel(qapp, monkeypatch)
    plan = update_install.InstallPlan("darwin", [], manual=True, open_path="/tmp/x.dmg",
                                      message="No permission to replace the app.")
    monkeypatch.setattr(update_install, "fetch_installer", lambda *a, **k: "/tmp/x.dmg")
    monkeypatch.setattr(update_install, "plan_install", lambda path, **k: plan)
    panel.install_button.click()
    assert spy.opened == ["/tmp/x.dmg"] and spy.launched == [] and spy.exited == []
    assert "No permission" in panel.status_label.text()


def test_a_helper_that_cannot_start_leaves_openNMR_running(qapp, monkeypatch):
    panel, spy = _panel(qapp, monkeypatch)
    monkeypatch.setattr(update_install, "fetch_installer", lambda *a, **k: "/tmp/x")
    monkeypatch.setattr(update_install, "plan_install",
                        lambda path, **k: update_install.InstallPlan("linux", ["/bin/sh"]))

    def cannot(plan):
        raise update_install.InstallError("Could not start the installer (denied).")
    monkeypatch.setattr(panel, "_launch", cannot)
    panel.install_button.click()
    assert "denied" in panel.status_label.text() and spy.exited == []


def test_progress_fills_the_bar_and_cancel_sets_the_flag(qapp, monkeypatch):
    panel, _ = _panel(qapp, monkeypatch)
    panel._on_install_progress(50_000_000, 250_000_000)
    assert (panel.progress_bar.maximum(), panel.progress_bar.value()) == (250_000_000, 50_000_000)
    assert "50 / 250 MB" in panel.progress_bar.format()
    panel._on_install_progress(10_000_000, 0)                # size unknown: busy bar
    assert panel.progress_bar.maximum() == 0
    assert not panel._cancel_event.is_set()
    panel.cancel_button.click()
    assert panel._cancel_event.is_set()


def test_pip_installs_update_with_pip_and_ask_for_a_restart(qapp, monkeypatch):
    panel, spy = _panel(qapp, monkeypatch, kind="git")
    assert "pip" in panel.install_button.text() and not panel.install_button.isHidden()
    assert panel.download_button.isHidden()
    ran = {}

    def fake_run(argv, **kwargs):
        ran["argv"] = argv
        kwargs["on_line"]("Collecting openNMR")
        return True, "Successfully installed opennmr\n"
    monkeypatch.setattr(update_install, "run_pip", fake_run)
    panel.install_button.click()
    assert ran["argv"][-1].startswith("git+https://github.com/")
    assert "Restart openNMR" in panel.status_label.text()
    assert spy.exited == [] and spy.launched == []           # the running copy is never killed


def test_a_failed_pip_update_says_what_pip_said(qapp, monkeypatch):
    panel, spy = _panel(qapp, monkeypatch, kind="git")
    monkeypatch.setattr(update_install, "run_pip",
                        lambda argv, **k: (False, "line1\nERROR: authentication failed\n"))
    panel.install_button.click()
    assert "authentication failed" in panel.status_label.text() and spy.exited == []


def test_a_source_folder_is_never_updated_by_the_button(qapp, monkeypatch):
    panel, _ = _panel(qapp, monkeypatch, kind="source")
    assert panel.install_button.isHidden()
    assert panel.command_edit.text() == "git pull"            # shown, never run


def test_the_install_button_is_off_while_installing(qapp, monkeypatch):
    panel, _ = _panel(qapp, monkeypatch)
    seen = []

    def fake_fetch(info, **kwargs):
        seen.append((panel.install_button.isEnabled(), panel.check_button.isEnabled(),
                     panel.progress_bar.isHidden()))
        raise update_install.InstallError("stop")
    monkeypatch.setattr(update_install, "fetch_installer", fake_fetch)
    panel.install_button.click()
    assert seen == [(False, False, False)]
    assert panel.install_button.isEnabled()                   # and back on afterwards


def test_a_release_without_an_installer_is_not_installable(qapp, monkeypatch):
    """A release that has only GitHub's source archives (no installer for this system) must
    not show Install, and must say why -- it used to offer the source archive as the installer."""
    panel, _ = _panel(qapp, monkeypatch)
    info = update.UpdateInfo(version="9.9.9", url="https://github.com/o/r/releases/tag/v9.9.9",
                             asset_name="")
    panel.last_result = update.UpdateCheck("newer", "0.1.0", info=info, latest_version="9.9.9",
                                           release_page="https://github.com/o/r/releases/tag/v9.9.9")
    panel.check_now()
    assert panel.install_button.isHidden() and panel.download_button.isHidden()
    assert not panel.release_page_button.isHidden()
    assert "no installer for your system" in panel.status_label.text()
