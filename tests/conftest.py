import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# When another Qt binding (e.g. PyQt6) is also installed in this
# environment, its Qt plugins can shadow PySide6's own -- at a
# different Qt version, which then fails to load at all ("Could not
# find the Qt platform plugin"). Point Qt at PySide6's own bundled
# plugins explicitly, before anything imports PySide6.
import PySide6

os.environ.setdefault(
    "QT_PLUGIN_PATH", os.path.join(os.path.dirname(PySide6.__file__), "Qt", "plugins"),
)

import matplotlib

matplotlib.use("Agg")
