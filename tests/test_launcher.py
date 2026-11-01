"""Start-up splash screen and one-window-only launch (2026-10-31).

Asked: show a "loading" logo when openNMR is opened, so a user does not click
again and end up with a crash or many copies.

What these pin:

* the launcher shows its splash *before* ``opennmr`` (numpy/scipy/matplotlib)
  is imported -- a splash after the imports would appear when the wait is
  over -- and imports nothing heavy itself;
* a second launch does not open a second window, and cannot lock the user out
  after a crash (stale lock from a dead process is taken over);
* ``--new-instance`` / ``OPENNMR_MULTI_INSTANCE`` still allow several windows;
* ``app.main`` hands the finished window to the splash and the guard;
* the build draws the pre-Python splash and keeps Tk when it does.
"""

import importlib.util
import os
import subprocess
import sys
import textwrap
import time

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:             # a new top-level module: an older editable install lacks it
    sys.path.insert(0, SRC)

import opennmr_launch as launch  # noqa: E402

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication, QWidget  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def tmp_lock_dir(tmp_path, monkeypatch):
    """Lock files in a private folder, and no inherited switches."""
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))
    monkeypatch.delenv(launch.MULTI_INSTANCE_ENV, raising=False)
    monkeypatch.delenv(launch.SPLASH_ENV, raising=False)
    return tmp_path


# -- the launcher imports nothing heavy ------------------------------------

def test_launcher_import_pulls_in_neither_opennmr_nor_numpy():
    """If it did, the splash could not appear before the multi-second wait."""
    code = ("import sys; sys.path.insert(0, %r); import opennmr_launch; "
            "bad = [m for m in ('opennmr', 'numpy', 'scipy', 'matplotlib') if m in sys.modules]; "
            "sys.exit(1 if bad else 0)" % SRC)
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_launcher_is_registered_as_the_console_script_and_packaged_module():
    tomllib = pytest.importorskip("tomllib")
    with open(os.path.join(ROOT, "pyproject.toml"), "rb") as handle:
        project = tomllib.load(handle)
    assert project["project"]["scripts"]["opennmr-gui"] == "opennmr_launch:main"
    assert "opennmr_launch" in project["tool"]["setuptools"]["py-modules"]


def test_packaged_entry_point_starts_through_the_launcher():
    text = open(os.path.join(ROOT, "packaging", "entrypoint.py"), encoding="utf-8").read()
    assert "from opennmr_launch import main" in text
    assert "opennmr.gui.app import main" not in text


# -- the splash picture ------------------------------------------------------

def test_splash_picture_has_the_agreed_size_and_changes_with_the_message(qapp):
    one = launch.render_splash_pixmap("1.2.3", "Loading…")
    two = launch.render_splash_pixmap("1.2.3", "Building the interface…")
    assert (one.width(), one.height()) == launch.SPLASH_SIZE
    assert not one.isNull()
    assert one.toImage() != two.toImage()


def test_show_splash_is_a_click_through_top_level_window(qapp, tmp_lock_dir):
    splash = launch.show_splash(qapp, "1.2.3")
    try:
        assert splash is not None and splash.isVisible()
        flags = splash.windowFlags()
        assert flags & Qt.WindowType.WindowTransparentForInput
        assert flags & Qt.WindowType.FramelessWindowHint
    finally:
        splash.close()


def test_splash_can_be_switched_off(qapp, tmp_lock_dir, monkeypatch):
    monkeypatch.setenv(launch.SPLASH_ENV, "0")
    assert launch.show_splash(qapp) is None
    launch.set_stage(qapp, None, "anything")        # no-op, no error


# -- one window only ---------------------------------------------------------

def test_second_launch_is_refused_and_asks_the_first_to_come_forward(qapp, tmp_lock_dir):
    first, second = launch.SingleInstance(), launch.SingleInstance()
    assert first.acquire() is True
    try:
        assert second.acquire() is False
        assert first.request_pending()

        class Window(QWidget):          # a real widget: QApplication.alert needs one
            def __init__(self):
                super().__init__()
                self.calls = []

            def isMinimized(self):
                return True

            def windowState(self):
                return Qt.WindowState.WindowMinimized

            def setWindowState(self, state):
                self.calls.append(("state", state))

            def show(self):
                self.calls.append("show")

            def raise_(self):
                self.calls.append("raise")

            def activateWindow(self):
                self.calls.append("activate")

        window = Window()
        assert first._answer(window) is True
        assert not first.request_pending()                    # answered once
        assert window.calls[0][0] == "state"                  # un-minimised
        assert not (window.calls[0][1] & Qt.WindowState.WindowMinimized)
        assert {"show", "raise", "activate"} <= set(window.calls)
        assert first._answer(window) is False                 # nothing more to do
    finally:
        first.release()


def test_released_lock_can_be_taken_again(qapp, tmp_lock_dir):
    first = launch.SingleInstance()
    assert first.acquire()
    first.release()
    again = launch.SingleInstance()
    assert again.acquire()
    again.release()


def test_a_request_left_by_a_finished_session_does_not_raise_the_next_one(qapp, tmp_lock_dir):
    stale = launch.SingleInstance()
    with open(stale.request_path, "w", encoding="utf-8") as handle:
        handle.write("raise\n")
    fresh = launch.SingleInstance()
    assert fresh.acquire()
    try:
        assert not fresh.request_pending()
    finally:
        fresh.release()


def _run_python(code, **kwargs):
    return subprocess.Popen([sys.executable, "-c", textwrap.dedent(code)],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, **kwargs)


def test_lock_left_by_a_crashed_process_is_taken_over(qapp, tmp_lock_dir):
    """A crash (or kill) must not lock the user out of their own program."""
    holder = _run_python(f"""
        import os, sys
        sys.path.insert(0, {SRC!r})
        import opennmr_launch as l
        from PySide6.QtCore import QCoreApplication
        app = QCoreApplication([])
        g = l.SingleInstance({str(tmp_lock_dir)!r})
        assert g.acquire()
        print("held", flush=True)
        os._exit(0)                 # dies without releasing
    """)
    out, err = holder.communicate(timeout=60)
    assert "held" in out, err
    survivor = launch.SingleInstance(str(tmp_lock_dir))
    assert survivor.acquire() is True
    survivor.release()


def test_main_exits_quietly_while_another_window_is_running(tmp_lock_dir, capfd):
    holder = _run_python(f"""
        import sys, time
        sys.path.insert(0, {SRC!r})
        import opennmr_launch as l
        from PySide6.QtCore import QCoreApplication
        app = QCoreApplication([])
        g = l.SingleInstance({str(tmp_lock_dir)!r})
        assert g.acquire()
        print("held", flush=True)
        time.sleep(60)
    """)
    try:
        assert holder.stdout.readline().strip() == "held"
        assert launch.main(["opennmr-gui"]) == 0          # no QApplication, no splash, no window
        assert "already running" in capfd.readouterr().err
        assert os.path.exists(launch.SingleInstance().request_path)   # the running one is asked
    finally:
        holder.kill()
        holder.wait()


# -- the launch sequence -----------------------------------------------------

@pytest.fixture
def fake_gui(monkeypatch, qapp):
    """Replace the heavy GUI module and QApplication; record what happens, in order."""
    import PySide6.QtWidgets as widgets
    import opennmr.gui as gui_package
    events = []

    class FakeApp:
        @staticmethod
        def main(app=None, splash=None, on_window=None):
            events.append(("gui_main", app is qapp, splash is not None,
                           on_window is not None))
            return 7

    real_show, real_stage = launch.show_splash, launch.set_stage
    monkeypatch.setattr(launch, "show_splash",
                        lambda app, version=None: events.append("splash") or real_show(app, version))
    monkeypatch.setattr(launch, "set_stage",
                        lambda app, splash, msg, version=None:
                        events.append(("stage", msg)) or real_stage(app, splash, msg, version))
    monkeypatch.setattr(gui_package, "app", FakeApp, raising=False)
    # Not monkeypatch: conftest's autouse cleanup runs *before* monkeypatch is
    # undone and calls QApplication.instance(), so the real class must be back
    # when this fixture is torn down.
    real_class = widgets.QApplication
    widgets.QApplication = lambda argv: events.append(("argv", list(argv))) or qapp
    try:
        yield events
    finally:
        widgets.QApplication = real_class


def test_splash_comes_before_the_gui_is_imported_and_the_guard_is_released(
        fake_gui, tmp_lock_dir):
    assert launch.main(["opennmr-gui"]) == 7              # the GUI's exit code is passed on
    kinds = [e if isinstance(e, str) else e[0] for e in fake_gui]
    assert kinds == ["argv", "splash", "stage", "stage", "gui_main"]
    assert fake_gui[-1] == ("gui_main", True, True, True)
    # The lock is gone again: the next launch is the first.
    again = launch.SingleInstance()
    assert again.acquire()
    again.release()


def test_new_instance_flag_allows_a_second_window_and_is_not_passed_on(fake_gui, tmp_lock_dir):
    holder = launch.SingleInstance()
    assert holder.acquire()
    try:
        assert launch.main(["opennmr-gui", "--new-instance"]) == 7
        assert ("argv", ["opennmr-gui"]) in fake_gui
        assert fake_gui[-1] == ("gui_main", True, True, False)     # no guard to watch with
    finally:
        holder.release()


def test_environment_switch_allows_several_windows(fake_gui, tmp_lock_dir, monkeypatch):
    holder = launch.SingleInstance()
    assert holder.acquire()
    try:
        monkeypatch.setenv(launch.MULTI_INSTANCE_ENV, "1")
        assert launch.main(["opennmr-gui"]) == 7
    finally:
        holder.release()


def test_splash_is_closed_and_lock_released_when_start_up_fails(
        monkeypatch, fake_gui, tmp_lock_dir):
    import opennmr.gui as gui_package

    class Broken:
        @staticmethod
        def main(**kwargs):
            raise RuntimeError("boom")

    monkeypatch.setattr(gui_package, "app", Broken, raising=False)
    with pytest.raises(RuntimeError):
        launch.main(["opennmr-gui"])
    again = launch.SingleInstance()
    assert again.acquire()                                # not locked out by the failure
    again.release()


# -- app.main hands over the finished window ---------------------------------

def test_app_main_closes_the_splash_on_the_window_and_starts_the_guard(qapp, monkeypatch):
    from opennmr.gui import app as app_module
    order = []

    class FakeWindow:
        def show(self):
            order.append("show")

        def maybe_show_welcome(self):
            pass

    class FakeSplash:
        def finish(self, window):
            order.append(("finish", isinstance(window, FakeWindow)))

    monkeypatch.setattr(app_module, "MainWindow", FakeWindow)
    monkeypatch.setattr(type(qapp), "exec", lambda self: 0)
    made = []
    monkeypatch.setattr(app_module, "QApplication", lambda argv: made.append(argv) or qapp)
    result = app_module.main(app=qapp, splash=FakeSplash(),
                             on_window=lambda w: order.append(("watch", isinstance(w, FakeWindow))))
    assert result == 0
    assert made == []                                     # used the one it was given
    assert order == ["show", ("finish", True), ("watch", True)]


# -- the build ------------------------------------------------------------------

def _load_build():
    spec = importlib.util.spec_from_file_location(
        "opennmr_build_splash", os.path.join(ROOT, "packaging", "build.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_boot_splash_is_skipped_without_tk_or_when_switched_off(monkeypatch):
    build = _load_build()
    monkeypatch.setattr(build, "_tk_available", lambda: False)
    assert build._make_boot_splash() is None
    monkeypatch.setattr(build, "_tk_available", lambda: True)
    monkeypatch.setenv("OPENNMR_BUILD_SPLASH", "0")
    assert build._make_boot_splash() is None


def test_boot_splash_image_is_drawn_when_tk_is_there(tmp_path, monkeypatch, qapp):
    build = _load_build()
    monkeypatch.setattr(build, "_tk_available", lambda: True)
    monkeypatch.setenv("OPENNMR_BUILD_SPLASH", "1")
    monkeypatch.setattr(build, "BUILD_DIR", str(tmp_path))
    path = build._make_boot_splash()
    assert path and os.path.exists(path)
    from PySide6.QtGui import QPixmap
    pix = QPixmap(path)
    assert (pix.width(), pix.height()) == launch.SPLASH_SIZE


@pytest.mark.parametrize("splash", ["/some/boot_splash.png", None])
def test_build_adds_the_splash_and_keeps_tk_only_then(monkeypatch, splash):
    build = _load_build()
    pyinstaller_main = pytest.importorskip("PyInstaller.__main__")
    captured = []
    monkeypatch.setattr(pyinstaller_main, "run", captured.append)
    monkeypatch.setattr(build, "_fix_qt_plugins", lambda: None)
    monkeypatch.setattr(build, "_stamp_macos_version", lambda: None)
    monkeypatch.setattr(build, "_make_icon", lambda: None)
    monkeypatch.setattr(build, "_make_boot_splash", lambda: splash)
    assert build.main() == 0
    args = captured[0]
    excluded = [args[i + 1] for i, a in enumerate(args) if a == "--exclude-module"]
    if splash:
        assert args[args.index("--splash") + 1] == splash
        assert "tkinter" not in excluded
    else:
        assert "--splash" not in args
        assert "tkinter" in excluded


def test_self_test_closes_the_boot_splash_before_running():
    text = open(os.path.join(ROOT, "packaging", "entrypoint.py"), encoding="utf-8").read()
    assert text.index("pyi_splash.close()") < text.index("selftest_main(sys.argv[2:])")


def test_boot_splash_is_off_by_default_on_windows_only(monkeypatch):
    """The Tk boot splash duplicated the Qt one on Windows (different size,
    font and position), so it needs an explicit opt-in there."""
    build = _load_build()
    monkeypatch.setattr(build, "_tk_available", lambda: True)
    monkeypatch.delenv("OPENNMR_BUILD_SPLASH", raising=False)
    monkeypatch.setattr(build, "_is_windows", lambda: True)
    assert build._make_boot_splash() is None


def test_splash_font_is_explicit_on_windows(qapp, monkeypatch):
    monkeypatch.setattr(launch, "_is_windows", lambda: True)
    font = launch.splash_font(44, bold=True)
    assert font.family() == "Segoe UI"
    assert font.pixelSize() == 44 and font.bold()
