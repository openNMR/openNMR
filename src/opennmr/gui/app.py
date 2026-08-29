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

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from .main_window import MainWindow

_LOGO_PATH = os.path.join(os.path.dirname(__file__), "resources", "logo.png")


def main() -> int:
    app = QApplication(sys.argv)
    if os.path.exists(_LOGO_PATH):
        app.setWindowIcon(QIcon(_LOGO_PATH))
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
