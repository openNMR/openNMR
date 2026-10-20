"""Settings ▸ Updates for a copy installed from PyPI (2026-10-20): one
button, one question, then openNMR closes, upgrades itself and opens again
-- with no dialog for an error. Nothing real runs: the helper is only
planned, never started; the confirmation box and quitting are stood in for."""
import json

import pytest

from opennmr import update, update_install
from opennmr.gui import update_panel


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def state_folder(tmp_path, monkeypatch):
    """Never the real ~/.opennmr: the result file is read (and removed) at start-up."""
    monkeypatch.setenv(update_install.STATE_DIR_ENV, str(tmp_path / "state"))
    return tmp_path / "state"


class Spy:
    def __init__(self):
        self.planned, self.launched, self.exited, self.asked, self.notified = [], [], [], [], []


def _panel(qapp, monkeypatch, blocker="", answer=True, launch_error=None):
    monkeypatch.setattr(update_panel.update, "install_kind", lambda: "pip")
    monkeypatch.setattr(update_install, "upgrade_blocker", lambda kind=None, python=None: blocker)
    spy = Spy()
    panel = update_panel.UpdatesPanel({}, notify=spy.notified.append)
    panel.background = False
    monkeypatch.setattr(panel, "_confirm", lambda title, text: spy.asked.append(text) or answer)
    plan = update_install.InstallPlan("linux", ["python", "helper.py", "config.json"])

    def fake_plan(version):
        spy.planned.append(version)
        return plan

    def fake_launch(given):
        if launch_error is not None:
            raise launch_error
        spy.launched.append(given)
    monkeypatch.setattr(panel, "_plan_pip_upgrade", fake_plan)
    monkeypatch.setattr(panel, "_launch_upgrade", fake_launch)
    monkeypatch.setattr(panel, "_exit_app", lambda: spy.exited.append(True))
    info = update.UpdateInfo(version="9.9.9", url="https://pypi.org/project/opennmr/9.9.9/")
    result = update.UpdateCheck("newer", "0.1.0", info=info, latest_version="9.9.9",
                                release_page="https://pypi.org/project/opennmr/9.9.9/")
    monkeypatch.setattr(update, "check", lambda *a, **k: result)
    panel.check_now()
    return panel, spy


def test_a_pip_install_asks_pypi_and_hides_the_repository_settings(qapp, monkeypatch):
    seen = []
    monkeypatch.setattr(update_panel.update, "install_kind", lambda: "pip")
    panel = update_panel.UpdatesPanel({})
    panel.background = False
    assert panel.source() == "pypi"
    assert panel.advanced_box.isHidden()                  # no repository or token to choose
    monkeypatch.setattr(update, "check", lambda source=None, token=None, **k: seen.append(source) or
                        update.UpdateCheck("current", "0.1.0"))
    panel.check_now()
    assert seen == ["pypi"]


def test_the_app_and_a_source_folder_keep_the_repository_settings(qapp, monkeypatch):
    for kind in ("app", "source", "git"):
        monkeypatch.setattr(update_panel.update, "install_kind", lambda kind=kind: kind)
        panel = update_panel.UpdatesPanel({})
        assert panel.source() == update.DEFAULT_SOURCE
        assert not panel.advanced_box.isHidden()


def test_a_newer_release_offers_one_update_button_and_the_command_as_a_fallback(qapp, monkeypatch):
    panel, _ = _panel(qapp, monkeypatch)
    assert not panel.install_button.isHidden()
    assert panel.install_button.text() == "Update to 9.9.9 and restart"
    assert panel.download_button.isHidden()                          # not an installer
    assert panel.command_edit.text() == "pip install --upgrade opennmr"
    assert "closes openNMR and opens it again" in panel.status_label.text()
    assert "pip install" not in panel.status_label.text()            # no jargon in the way


def test_the_button_asks_once_then_hands_over_and_quits(qapp, monkeypatch):
    panel, spy = _panel(qapp, monkeypatch)
    panel.install_button.click()
    assert len(spy.asked) == 1 and "9.9.9" in spy.asked[0] and "open again" in spy.asked[0]
    assert spy.planned == ["9.9.9"]
    assert len(spy.launched) == 1
    assert spy.exited == [True]
    assert "close and open again" in panel.status_label.text()


def test_declining_the_question_does_nothing(qapp, monkeypatch):
    panel, spy = _panel(qapp, monkeypatch, answer=False)
    panel.install_button.click()
    assert spy.planned == [] and spy.launched == [] and spy.exited == []


def test_pip_is_never_run_inside_the_running_app(qapp, monkeypatch):
    panel, spy = _panel(qapp, monkeypatch)
    monkeypatch.setattr(update_install, "run_pip",
                        lambda *a, **k: pytest.fail("the app must not run pip on itself"))
    panel.install_button.click()
    assert spy.exited == [True]


def test_where_pip_may_not_change_things_only_the_command_is_shown(qapp, monkeypatch):
    panel, spy = _panel(qapp, monkeypatch, blocker="This Python belongs to the operating system.")
    assert panel.install_button.isHidden()
    assert panel.command_edit.text() == "pip install --upgrade opennmr"
    assert "belongs to the operating system" in panel.status_label.text()


def test_a_helper_that_cannot_start_leaves_openNMR_open_with_a_calm_message(qapp, monkeypatch):
    error = update_install.InstallError("Could not start the update (no such program).")
    panel, spy = _panel(qapp, monkeypatch, launch_error=error)
    panel.install_button.click()                                      # must not raise
    assert spy.exited == []                                           # nothing was closed
    text = panel.status_label.text()
    assert "could not be started" in text and "left as it is" in text
    assert "Traceback" not in text and "Error" not in text
    assert panel.command_edit.text() == "pip install --upgrade opennmr" and not panel.command_row.isHidden()


def test_a_planning_failure_is_also_calm(qapp, monkeypatch):
    panel, spy = _panel(qapp, monkeypatch)

    def boom(version):
        raise OSError("disk full")
    monkeypatch.setattr(panel, "_plan_pip_upgrade", boom)
    panel.install_button.click()
    assert spy.exited == [] and "could not be started" in panel.status_label.text()


def test_the_first_start_after_an_upgrade_says_so_once(qapp, monkeypatch, state_folder):
    state_folder.mkdir(parents=True)
    (state_folder / update_install.RESULT_NAME).write_text(
        json.dumps({"status": "updated", "from": "0.1.0", "to": "0.1.1"}), encoding="utf-8")
    notes = []
    panel = update_panel.UpdatesPanel({}, notify=notes.append)
    assert panel.status_label.text() == "openNMR was updated to 0.1.1."
    assert notes == ["openNMR was updated to 0.1.1."]
    again = update_panel.UpdatesPanel({}, notify=notes.append)       # shown a single time
    assert again.status_label.text() == ""


def test_a_failed_upgrade_is_reported_as_a_line_not_a_dialog(qapp, monkeypatch, state_folder):
    state_folder.mkdir(parents=True)
    (state_folder / update_install.RESULT_NAME).write_text(
        json.dumps({"status": "failed", "from": "0.1.0", "to": "0.1.1", "detail": "pip said no"}),
        encoding="utf-8")
    from PySide6.QtWidgets import QMessageBox
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: pytest.fail("no dialog"))
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: pytest.fail("no dialog"))
    monkeypatch.setattr(QMessageBox, "critical", lambda *a, **k: pytest.fail("no dialog"))
    panel = update_panel.UpdatesPanel({})
    text = panel.status_label.text()
    assert "did not complete" in text and "Nothing was lost" in text
    assert "pip said no" not in text                                 # the detail is in the log


def test_a_cancelled_upgrade_leaves_no_message(qapp, state_folder):
    state_folder.mkdir(parents=True)
    (state_folder / update_install.RESULT_NAME).write_text(
        json.dumps({"status": "cancelled", "to": "0.1.1"}), encoding="utf-8")
    assert update_panel.UpdatesPanel({}).status_label.text() == ""
