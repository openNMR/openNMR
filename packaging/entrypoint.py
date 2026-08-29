"""PyInstaller's actual entry point.

Deliberately not ``src/nmrsuite/gui/app.py`` itself: PyInstaller
analyzes whatever script it's pointed at as a standalone top-level
script, outside any package -- ``app.py``'s own ``from .main_window
import MainWindow`` (a relative import, valid only inside the
``nmrsuite.gui`` package) then fails at runtime with "attempted
relative import with no known parent package" the moment the frozen
app actually launches, even though the build itself completes without
error. Importing ``nmrsuite`` as an installed package here instead
(absolute import) sidesteps that entirely.
"""

from nmrsuite.gui.app import main

if __name__ == "__main__":
    raise SystemExit(main())
