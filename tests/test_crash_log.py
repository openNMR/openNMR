"""The crash log (src/opennmr_crashlog.py): unexpected errors recorded in
~/.opennmr/logs/crash.log, because the installed (windowed) app has no
console (stability audit / explicit user request, 2026-11-01)."""

import os
import subprocess
import sys
import textwrap
import threading

import pytest

import opennmr_crashlog as crashlog

from test_gui import qapp  # noqa: F401 -- fixture


@pytest.fixture
def log(tmp_path, monkeypatch):
    """The crash log installed into a temporary folder, no message box."""
    monkeypatch.setenv(crashlog.LOG_DIR_ENV, str(tmp_path))
    monkeypatch.setenv(crashlog.ENABLE_ENV, "1")
    monkeypatch.setenv(crashlog.DIALOG_ENV, "0")
    crashlog.uninstall()
    assert crashlog.install(qt=False)
    yield tmp_path / crashlog.LOG_NAME
    crashlog.uninstall()


def _raise_into_hook(exc):
    try:
        raise exc
    except type(exc):
        sys.excepthook(*sys.exc_info())


def test_disabled_under_the_test_suite_by_default():
    assert os.environ.get(crashlog.ENABLE_ENV) == "0"          # tests/conftest.py
    assert crashlog.install() is False
    assert sys.excepthook is not crashlog._excepthook


def test_an_unhandled_error_is_written_with_versions(log):
    _raise_into_hook(ValueError("bad phase table"))
    text = log.read_text(encoding="utf-8")
    assert "Unhandled error" in text
    assert "ValueError: bad phase table" in text
    assert "Traceback" in text
    assert "openNMR" in text and "Python" in text


def test_install_is_idempotent_and_uninstall_restores(log):
    assert crashlog.install(qt=False) is False
    crashlog.uninstall()
    assert sys.excepthook is not crashlog._excepthook
    assert threading.excepthook is not crashlog._threading_hook


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_errors_in_worker_threads_are_written(log):
    def work():
        raise RuntimeError("worker died")
    t = threading.Thread(target=work, name="opennmr-test-worker")
    t.start()
    t.join()
    text = log.read_text(encoding="utf-8")
    assert "thread 'opennmr-test-worker'" in text and "RuntimeError: worker died" in text


def test_nothing_is_written_when_nothing_goes_wrong(log):
    assert not log.exists() or log.read_text(encoding="utf-8") == ""


def test_large_log_is_rotated_at_start(tmp_path, monkeypatch):
    monkeypatch.setenv(crashlog.LOG_DIR_ENV, str(tmp_path))
    monkeypatch.setenv(crashlog.ENABLE_ENV, "1")
    big = tmp_path / crashlog.LOG_NAME
    big.write_text("x" * (crashlog.MAX_BYTES + 10), encoding="utf-8")
    crashlog.uninstall()
    try:
        crashlog.install(qt=False)
        assert (tmp_path / (crashlog.LOG_NAME + ".1")).stat().st_size > crashlog.MAX_BYTES
        assert big.stat().st_size == 0
    finally:
        crashlog.uninstall()


def test_unwritable_folder_never_raises(tmp_path, monkeypatch):
    blocker = tmp_path / "file"
    blocker.write_text("")
    monkeypatch.setenv(crashlog.LOG_DIR_ENV, str(blocker / "logs"))   # under a file
    monkeypatch.setenv(crashlog.ENABLE_ENV, "1")
    crashlog.uninstall()
    try:
        crashlog.install(qt=False)
        assert crashlog.write_entry("x", "y") is None
        _raise_into_hook(KeyError("still fine"))
    finally:
        crashlog.uninstall()


def test_a_slot_error_reaches_the_log(log, qapp):  # noqa: F811
    from PySide6.QtWidgets import QPushButton
    button = QPushButton()
    button.clicked.connect(lambda: 1 / 0)
    button.click()
    assert "ZeroDivisionError" in log.read_text(encoding="utf-8")


def test_the_user_is_told_once_per_burst(log, qapp, monkeypatch):  # noqa: F811
    monkeypatch.setenv(crashlog.DIALOG_ENV, "1")
    shown = []
    monkeypatch.setattr(crashlog, "show_error_message",
                        lambda *a: shown.append(a[3]) or None)
    for _ in range(5):
        _raise_into_hook(IndexError("redraw"))
    assert len(shown) == 1 and shown[0] == str(log)
    assert log.read_text(encoding="utf-8").count("IndexError: redraw") == 5


def test_the_message_box_is_modeless_and_offers_the_folder(log, qapp):  # noqa: F811
    box = crashlog.show_error_message(ValueError, ValueError("x"), "Traceback…", str(log))
    try:
        from PySide6.QtCore import Qt
        assert box.windowModality() == Qt.WindowModality.NonModal
        assert any(b.text() == "Open Log Folder" for b in box.buttons())
        assert str(log) in box.informativeText()
        assert box.detailedText() == "Traceback…"
    finally:
        box.close()


def test_a_hard_crash_leaves_a_stack_in_the_log(tmp_path):
    """faulthandler: a segmentation fault writes the Python stack before dying."""
    src = os.path.join(os.path.dirname(__file__), "..", "src")
    code = textwrap.dedent(f"""
        import sys; sys.path.insert(0, {os.path.abspath(src)!r})
        import opennmr_crashlog, faulthandler
        opennmr_crashlog.install(qt=False)
        def deep_in_a_fit():
            faulthandler._sigsegv()
        deep_in_a_fit()
    """)
    env = {**os.environ, crashlog.LOG_DIR_ENV: str(tmp_path), crashlog.ENABLE_ENV: "1"}
    proc = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True,
                          timeout=60)
    assert proc.returncode != 0
    text = (tmp_path / crashlog.LOG_NAME).read_text(encoding="utf-8", errors="replace")
    assert "Segmentation fault" in text or "Fatal Python error" in text
    assert "deep_in_a_fit" in text


def test_launcher_installs_the_crash_log():
    import inspect
    import opennmr_launch
    source = inspect.getsource(opennmr_launch.main)
    assert "opennmr_crashlog.install()" in source
    assert source.index("opennmr_crashlog.install()") < source.index("show_splash(")


def test_help_menu_opens_the_crash_log_folder(qapp, tmp_path, monkeypatch):  # noqa: F811
    from PySide6.QtGui import QDesktopServices
    from opennmr.gui.main_window import MainWindow
    monkeypatch.setenv(crashlog.LOG_DIR_ENV, str(tmp_path / "logs"))
    opened = []
    monkeypatch.setattr(QDesktopServices, "openUrl", lambda url: opened.append(url.toLocalFile()))
    window = MainWindow()
    try:
        assert window.crash_log_action in window.help_menu.actions()
        window.crash_log_action.trigger()
        # QUrl.toLocalFile gives forward slashes on Windows: compare the folders.
        assert [os.path.normcase(os.path.normpath(p)) for p in opened] == \
            [os.path.normcase(os.path.normpath(str(tmp_path / "logs")))]
        assert (tmp_path / "logs").is_dir()
    finally:
        window.close()


def test_launcher_still_starts_without_the_crash_log_module():
    """An editable install made before the module existed cannot import it;
    the launcher must start anyway."""
    import inspect
    import opennmr_launch
    source = inspect.getsource(opennmr_launch.main)
    block = source[source.index("try:\n        import opennmr_crashlog"):]
    assert block.index("except Exception") < block.index("guard = None")


def test_self_test_checks_the_crash_log_is_bundled():
    import inspect
    from opennmr import selftest
    assert '"opennmr_crashlog"' in inspect.getsource(selftest.run)
