"""Start-up shim for the desktop GUI: splash screen first, one window only.

Why this is a module of its own, *outside* the ``opennmr`` package
-------------------------------------------------------------------
``import opennmr`` runs ``opennmr/__init__.py``, which imports numpy,
scipy and matplotlib through ``core``/``io``/``processing``/``plotting``
(several seconds, most of the wait before a window appears). Anything
inside the package -- including ``opennmr.gui.app.main`` -- can only run
*after* that, so a splash screen placed there would appear when most of
the waiting is already over. This module imports nothing but the standard
library and PySide6 until the splash is on screen; the ``opennmr-gui``
console script, the packaged app (``packaging/entrypoint.py``) and
``python -m opennmr.gui.app`` all start here.

Single instance
---------------
Without feedback a user clicks again, and every click used to open another
copy. :class:`SingleInstance` lets only the first launch through: a later
one asks the running window to come to the front and exits. It is built on
``QLockFile`` plus a request file the running instance polls -- QtCore
only, because the packaged app deliberately leaves out QtNetwork (see
``packaging/build.py``), which ``QLocalServer`` would need.

Escape hatches: ``opennmr-gui --new-instance`` or ``OPENNMR_MULTI_INSTANCE=1``
(development, or comparing two sessions on purpose).

Threading note: while Python imports or builds the window the Qt event loop
cannot run, so the splash cannot animate. It shows a static message that is
updated at each stage instead (an animation that freezes looks like a hang).
"""

from __future__ import annotations

import os
import sys
import tempfile

#: Set to a non-empty, non-"0" value to allow several windows at once.
MULTI_INSTANCE_ENV = "OPENNMR_MULTI_INSTANCE"
#: Set to "0" to run without the splash screen (tests, embedding hosts).
SPLASH_ENV = "OPENNMR_SPLASH"
#: Command-line switch with the same effect as ``MULTI_INSTANCE_ENV``.
NEW_INSTANCE_FLAG = "--new-instance"

#: Seconds between looks for a "come to the front" request.
_POLL_MS = 400

SPLASH_SIZE = (520, 300)


# -- the splash picture -----------------------------------------------------

def installed_version() -> str:
    """The version for the splash, without importing ``opennmr`` (that is the
    slow part). Empty if it cannot be found."""
    try:
        from importlib import metadata
        return metadata.version("opennmr")
    except Exception:
        return ""


def _logo_path() -> str:
    """``logo.png`` beside ``opennmr/gui`` -- found without importing the
    package (``find_spec`` of a top-level name reads no ``__init__``)."""
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, "opennmr", "gui", "resources", "logo.png"),
        os.path.join(getattr(sys, "_MEIPASS", here), "opennmr", "gui", "resources", "logo.png"),
    ]
    try:
        import importlib.util
        spec = importlib.util.find_spec("opennmr")
        if spec and spec.submodule_search_locations:
            candidates.insert(0, os.path.join(
                list(spec.submodule_search_locations)[0], "gui", "resources", "logo.png"))
    except Exception:
        pass
    for path in candidates:
        if os.path.exists(path):
            return path
    return ""


def _is_windows() -> bool:
    return sys.platform.startswith("win")


def splash_font(pixel_size: int, bold: bool = False):
    """A splash font that does not depend on the Qt platform plugin.

    A bare ``QFont()`` is the *platform's* default: Segoe UI under the normal
    Windows plugin, but an arbitrary (condensed) face under ``offscreen``,
    which is what drew the pre-Python boot splash image at build time.
    Naming the Windows UI font explicitly makes both renders identical."""
    from PySide6.QtGui import QFont

    font = QFont()
    if _is_windows():
        font.setFamily("Segoe UI")
    font.setPixelSize(pixel_size)
    font.setBold(bold)
    return font


def render_splash_pixmap(version: str | None = None, message: str = "Loading…"):
    """The splash picture: logo, name, version and ``message``.

    A fixed light card whatever the system theme -- it is on screen for a
    few seconds, before the saved theme has been read. Also used by
    ``packaging/build.py`` to draw the image PyInstaller's boot splash shows
    before Python itself has started, so the two look like one."""
    from PySide6.QtCore import QRectF, Qt
    from PySide6.QtGui import QColor, QPainter, QPen, QPixmap

    width, height = SPLASH_SIZE
    version = installed_version() if version is None else version
    pix = QPixmap(width, height)
    pix.fill(QColor("#ffffff"))
    painter = QPainter(pix)
    try:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.setPen(QPen(QColor("#c9ced6"), 2))
        painter.drawRect(1, 1, width - 2, height - 2)
        logo_file = _logo_path()
        if logo_file:
            logo = QPixmap(logo_file)
            if not logo.isNull():
                side = 150
                logo = logo.scaled(side, side, Qt.AspectRatioMode.KeepAspectRatio,
                                   Qt.TransformationMode.SmoothTransformation)
                painter.drawPixmap(28, (height - logo.height()) // 2 - 10, logo)
        text_left = 205
        painter.setPen(QColor("#1f2933"))
        painter.setFont(splash_font(44, bold=True))
        painter.drawText(QRectF(text_left, 70, width - text_left - 20, 60),
                         int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
                         "openNMR")
        painter.setFont(splash_font(16))
        painter.setPen(QColor("#52606d"))
        if version:
            painter.drawText(QRectF(text_left, 132, width - text_left - 20, 26),
                             int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
                             f"Version {version}")
        painter.setPen(QColor("#1f2933"))
        painter.drawText(QRectF(text_left, 176, width - text_left - 20, 26),
                         int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
                         message)
    finally:
        painter.end()
    return pix


def show_splash(app, version: str | None = None):
    """Show the splash and paint it *now*. Returns the ``QSplashScreen``, or
    ``None`` when splashes are switched off (``OPENNMR_SPLASH=0``)."""
    if os.environ.get(SPLASH_ENV, "1").strip() == "0":
        return None
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QSplashScreen

    splash = QSplashScreen(render_splash_pixmap(version),
                           Qt.WindowType.WindowStaysOnTopHint
                           | Qt.WindowType.FramelessWindowHint
                           # Clicks pass through: a click on a window that is not
                           # pumping messages is what makes Windows grey it out as
                           # "Not responding" after five seconds.
                           | Qt.WindowType.WindowTransparentForInput)
    splash.show()
    app.processEvents()
    return splash


def set_stage(app, splash, message: str, version: str | None = None) -> None:
    """Redraw the splash with a new status line (no-op without a splash)."""
    if splash is None:
        return
    splash.setPixmap(render_splash_pixmap(version, message))
    app.processEvents()


def _close_boot_splash() -> None:
    """Close PyInstaller's pre-Python splash once ours is up (only exists in a
    frozen build made with ``--splash``)."""
    try:
        import pyi_splash  # type: ignore[import-not-found]
    except ImportError:
        return
    try:
        pyi_splash.close()
    except Exception:
        pass


# -- one window only -------------------------------------------------------

def _user_tag() -> str:
    try:
        import getpass
        name = getpass.getuser()
    except Exception:
        name = "user"
    return "".join(c if c.isalnum() else "_" for c in name) or "user"


class SingleInstance:
    """Let one openNMR window run per user.

    ``acquire()`` is ``True`` for the first launch. For any later one it is
    ``False`` after leaving a request file; the first instance, which called
    :meth:`watch`, sees it within a moment and raises its window.

    A crash cannot lock the user out: ``QLockFile`` recognises a lock whose
    process no longer exists as stale and takes it over. The age-based
    staleness rule is switched off (``setStaleLockTime(0)``) -- left on, a
    window open for more than 30 s would have its lock broken by the next
    launch.
    """

    def __init__(self, directory: str | None = None, tag: str | None = None):
        directory = directory or tempfile.gettempdir()
        tag = tag or _user_tag()
        self.lock_path = os.path.join(directory, f"opennmr-gui-{tag}.lock")
        self.request_path = os.path.join(directory, f"opennmr-gui-{tag}.raise")
        self._lock = None
        self._timer = None

    def acquire(self) -> bool:
        from PySide6.QtCore import QLockFile

        lock = QLockFile(self.lock_path)
        lock.setStaleLockTime(0)
        if lock.tryLock(0):
            self._lock = lock
            # A request left for an instance that has since quit must not
            # make this fresh one raise itself the moment it starts.
            self._remove_request()
            return True
        self._write_request()
        return False

    def release(self) -> None:
        if self._timer is not None:
            self._timer.stop()
            self._timer = None
        if self._lock is not None:
            self._lock.unlock()
            self._lock = None
        self._remove_request()

    def _write_request(self) -> None:
        try:
            with open(self.request_path, "w", encoding="utf-8") as handle:
                handle.write("raise\n")
        except OSError:
            pass            # nothing more can be done; still exit quietly

    def _remove_request(self) -> None:
        try:
            os.remove(self.request_path)
        except OSError:
            pass

    def request_pending(self) -> bool:
        return os.path.exists(self.request_path)

    def watch(self, window) -> None:
        """Start answering "come to the front" requests for ``window``."""
        from PySide6.QtCore import QTimer

        self._timer = QTimer(window)
        self._timer.setInterval(_POLL_MS)
        self._timer.timeout.connect(lambda: self._answer(window))
        self._timer.start()

    def _answer(self, window) -> bool:
        if not self.request_pending():
            return False
        self._remove_request()
        bring_to_front(window)
        return True


def bring_to_front(window) -> None:
    """Un-minimise, raise and focus ``window``; flash the taskbar entry where
    the system refuses to hand over focus (Windows)."""
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    if window.isMinimized():
        window.setWindowState(window.windowState() & ~Qt.WindowState.WindowMinimized
                              | Qt.WindowState.WindowActive)
    window.show()
    window.raise_()
    window.activateWindow()
    QApplication.alert(window, 2000)


# -- entry point --------------------------------------------------------------

def _wants_multiple(argv) -> bool:
    if os.environ.get(MULTI_INSTANCE_ENV, "").strip() not in ("", "0"):
        return True
    return NEW_INSTANCE_FLAG in argv


def main(argv=None) -> int:
    """Console-script entry point: splash, then the real start-up."""
    argv = list(sys.argv if argv is None else argv)
    multiple = _wants_multiple(argv)
    argv = [a for a in argv if a != NEW_INSTANCE_FLAG]

    # Qt must be told which plugins are PySide6's own before it loads
    # (see opennmr.gui.app for why).
    import PySide6
    os.environ.setdefault(
        "QT_PLUGIN_PATH", os.path.join(os.path.dirname(PySide6.__file__), "Qt", "plugins"))
    from PySide6.QtWidgets import QApplication

    guard = None
    if not multiple:
        guard = SingleInstance()
        if not guard.acquire():
            # Already running: it has been asked to come forward.
            print("openNMR is already running -- bringing its window to the front.",
                  file=sys.stderr)
            return 0

    app = QApplication(argv)
    version = installed_version()
    splash = show_splash(app, version)
    _close_boot_splash()

    try:
        set_stage(app, splash, "Loading numerical libraries…", version)
        # The slow imports happen here, behind the splash.
        from opennmr.gui import app as gui_app

        set_stage(app, splash, "Building the interface…", version)
        return gui_app.main(app=app, splash=splash,
                            on_window=guard.watch if guard is not None else None)
    finally:
        if splash is not None:
            splash.close()
        if guard is not None:
            guard.release()


if __name__ == "__main__":
    sys.exit(main())
