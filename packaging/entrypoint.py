"""PyInstaller's actual entry point.

Deliberately not ``src/opennmr/gui/app.py`` itself: PyInstaller
analyzes whatever script it's pointed at as a standalone top-level
script, outside any package -- ``app.py``'s own ``from .main_window
import MainWindow`` (a relative import, valid only inside the
``opennmr.gui`` package) then fails at runtime with "attempted
relative import with no known parent package" the moment the frozen
app actually launches, even though the build itself completes without
error. Importing ``opennmr`` as an installed package here instead
(absolute import) sidesteps that entirely.

``--self-test OUT_DIR`` runs :mod:`opennmr.selftest` inside the frozen
app instead of opening the window -- how a build is checked (CI, and
``packaging/compare_selftest.py`` against a run from source).
"""

import multiprocessing
import sys

if __name__ == "__main__":
    # A frozen app that ever starts a process pool (the BMS simulator in
    # notebook/bms does) must call this first, or each child process
    # would launch another copy of the GUI on Windows and macOS.
    multiprocessing.freeze_support()
    if len(sys.argv) > 1 and sys.argv[1] == "--self-test":
        try:                      # the boot splash (build --splash) has no use here
            import pyi_splash
            pyi_splash.close()
        except ImportError:
            pass
        from opennmr.selftest import main as selftest_main
        raise SystemExit(selftest_main(sys.argv[2:]))
    # Not ``opennmr.gui.app`` directly: that imports numpy/scipy/matplotlib
    # (through ``opennmr/__init__``) before it can draw anything. The
    # launcher shows the splash first and keeps to one window.
    from opennmr_launch import main
    raise SystemExit(main())
