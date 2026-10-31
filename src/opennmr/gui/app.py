"""Entry point for the desktop GUI (``opennmr-gui`` console script)."""

from __future__ import annotations

import os
import sys

import PySide6

# When another Qt binding (e.g. PyQt6) is also installed in this
# environment, its Qt plugins can shadow PySide6's own -- at a
# different Qt version, which then fails to load at all ("Could not
# find the Qt platform plugin"). Point Qt at PySide6's own bundled
# plugins explicitly, before anything imports PySide6.QtWidgets/QtCore
# (which is what actually loads the Qt shared libraries).
os.environ.setdefault(
    "QT_PLUGIN_PATH", os.path.join(os.path.dirname(PySide6.__file__), "Qt", "plugins"),
)

from PySide6.QtCore import QTimer
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from .main_window import MainWindow
from .numeric_locale import use_decimal_point
from .style import remember_system_theme

_LOGO_PATH = os.path.join(os.path.dirname(__file__), "resources", "logo.png")


def main(app=None, splash=None, on_window=None) -> int:
    """Build and show the main window, then run the event loop.

    Called with no arguments (``python -m opennmr.gui.app``, tests) it makes
    its own ``QApplication``. The normal launch is ``opennmr_launch.main``,
    which has already made the application and a splash screen -- those come
    in as ``app`` and ``splash`` (closed once the window is up) -- and
    passes ``on_window``, called with the finished window (it starts
    answering "bring me to the front" requests from a second launch)."""
    # Before any widget: numbers are written 1.5, never 1,5, whatever the
    # computer's regional settings (see numeric_locale).
    use_decimal_point()
    if app is None:
        app = QApplication(sys.argv)
    # Before apply_theme, which overrides both the palette and the
    # colour scheme it would otherwise be read back from.
    remember_system_theme(app)
    # The theme itself is installed by MainWindow once its widgets
    # exist (much faster than before them -- see
    # MainWindow._apply_startup_ui); installing it here as well would
    # bring back the slow start.
    if os.path.exists(_LOGO_PATH):
        app.setWindowIcon(QIcon(_LOGO_PATH))
    window = MainWindow()
    window.show()
    if splash is not None:
        splash.finish(window)
    if on_window is not None:
        on_window(window)
    # First run of this version: the Welcome tour of the settings (skippable;
    # Help > Welcome to openNMR brings it back).
    QTimer.singleShot(0, window.maybe_show_welcome)
    return app.exec()


if __name__ == "__main__":
    # Through the launcher (splash, one window only) when it is installed;
    # the update helper relaunches the app with ``python -m opennmr.gui.app``.
    try:
        from opennmr_launch import main as _launch
    except ImportError:
        sys.exit(main())
    sys.exit(_launch())
