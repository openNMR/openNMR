"""
Cross-platform build script for a standalone opennmr desktop app.

Usage::

    pip install -e ".[build]"
    python packaging/build.py

The same script produces the right output on whichever OS it's run on
(macOS -> a ``.app`` bundle, Windows -> an ``.exe`` + folder, Linux ->
an executable + folder) -- "system agnostic" in the sense that
there's one build process, not three separate manual ones. Wrapping
that output in each platform's own native installer format (a DMG,
an MSI, a ``.deb``/AppImage) is a further, not-yet-built step -- see
``packaging/README.md``.
"""

from __future__ import annotations

import os
import posixpath
import re
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESOURCES_DIR = os.path.join(REPO_ROOT, "src", "opennmr", "gui", "resources")
LOGO_PNG = os.path.join(RESOURCES_DIR, "logo.png")
ENTRY_POINT = os.path.join(REPO_ROOT, "packaging", "entrypoint.py")
TOPSPIN_RESOURCES = os.path.join(REPO_ROOT, "src", "opennmr", "topspin", "resources")

#: The app's name as users see it (the .exe, the .app, the Start-menu
#: entry). Was "opennmr" until 2026-10-15; build.yml and
#: make_installer.py look for this name in dist/.
APP_NAME = "openNMR"
BUNDLE_ID = "org.opennmr.openNMR"
BUILD_DIR = os.path.join(REPO_ROOT, "build")
DIST_DIR = os.path.join(REPO_ROOT, "dist")

#: The matplotlib backends that write the vector formats Save Figure offers
#: (SVG, PDF, EPS/PS). matplotlib imports a backend by name at the moment of
#: saving (``importlib.import_module``), which PyInstaller's static analysis
#: cannot see, so without these a frozen app drew bitmaps fine and failed
#: every vector export with "No module named 'matplotlib.backends.backend_svg'"
#: (user report, 2026-10-29). Listed one by one rather than
#: ``--collect-submodules matplotlib.backends``, which would also pull in the
#: Qt5/Tk/GTK backends and the second Qt binding the build excludes.
MATPLOTLIB_EXPORT_BACKENDS = (
    "matplotlib.backends.backend_svg",
    "matplotlib.backends.backend_pdf",
    "matplotlib.backends.backend_ps",
    "matplotlib.backends.backend_agg",
)


def _is_windows() -> bool:
    return sys.platform.startswith("win")


def _tk_available() -> bool:
    """PyInstaller's boot splash is drawn by Tcl/Tk, which a few Python builds
    lack. (It is not supported on macOS at all.)"""
    if sys.platform == "darwin":
        return False
    try:
        import _tkinter  # noqa: F401
    except ImportError:
        return False
    return True


def _make_boot_splash() -> str | None:
    """Draw the image PyInstaller shows while the bundle unpacks and Python
    starts -- before any of our code can run -- with the same picture as the
    Qt splash that takes over (``opennmr_launch.render_splash_pixmap``).

    Returns its path, or ``None`` when it cannot or should not be made
    (no Tk, macOS, ``OPENNMR_BUILD_SPLASH=0``, or Windows unless
    ``OPENNMR_BUILD_SPLASH=1``); the build then goes ahead without it, and
    the Qt splash is all the user sees.

    Off by default on Windows: the app is a onedir build, so there is almost
    nothing to unpack before Python starts, and the Tk window is not
    DPI-aware -- it showed at a different size and position from the Qt
    splash, so users saw two splashes swap. Set ``OPENNMR_BUILD_SPLASH=1`` to
    force it (e.g. for a onefile build, where unpacking takes seconds)."""
    default = "0" if _is_windows() else "1"
    if os.environ.get("OPENNMR_BUILD_SPLASH", default).strip() == "0" or not _tk_available():
        return None
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        if os.name == "nt":      # offscreen finds no fonts on Windows (see tests/conftest.py)
            fonts = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")
            if os.path.isdir(fonts):
                os.environ.setdefault("QT_QPA_FONTDIR", fonts)
        src = os.path.join(REPO_ROOT, "src")
        if src not in sys.path:
            sys.path.insert(0, src)
        from PySide6.QtWidgets import QApplication
        import opennmr_launch
        app = QApplication.instance() or QApplication([])  # noqa: F841 - QPixmap needs it
        os.makedirs(BUILD_DIR, exist_ok=True)
        path = os.path.join(BUILD_DIR, "boot_splash.png")
        pix = opennmr_launch.render_splash_pixmap(app_version(), "Loading\u2026")
        return path if pix.save(path, "PNG") else None
    except Exception as exc:     # a missing splash must never fail a release build
        print(f"Boot splash skipped: {exc}", file=sys.stderr)
        return None


def _make_icon() -> str | None:
    """Convert the app's PNG logo to whatever icon format this
    platform's build wants (``.icns`` on macOS, ``.ico`` on Windows).
    Returns ``None`` on Linux (PyInstaller's own default icon is used
    instead -- desktop-entry icons are a separate, packaging-format-
    specific concern, not something PyInstaller itself sets), or if
    conversion fails for any reason -- icon polish is worth doing, not
    worth failing the whole build over.

    PNG, not the vector logo in ``assets/`` -- that SVG renders its
    "openNMR" text in a specific font (Bai Jamjuree) that isn't
    guaranteed to be installed wherever this build runs, so rasterizing
    it here risked baking in a silently wrong-looking (fallback-font)
    icon. The PNG is a pre-rendered, WYSIWYG export with no such
    dependency."""
    if not os.path.exists(LOGO_PNG):
        return None
    try:
        from PIL import Image
    except ImportError:
        return None

    os.makedirs(BUILD_DIR, exist_ok=True)
    img = Image.open(LOGO_PNG).convert("RGBA")

    try:
        if sys.platform.startswith("win"):
            path = os.path.join(BUILD_DIR, "icon.ico")
            img.save(path, format="ICO", sizes=[(16, 16), (32, 32), (48, 48), (128, 128), (256, 256)])
            return path
        if sys.platform.startswith("darwin"):
            path = os.path.join(BUILD_DIR, "icon.icns")
            img.save(path, format="ICNS")
            return path
    except Exception:  # noqa: BLE001 -- best-effort; fall through to no icon
        return None
    return None


#: Library directories of the distribution itself. A libexpat found anywhere else (conda,
#: a python.org build, pyenv) is private to the Python being frozen.
SYSTEM_LIB_PREFIXES = ("/lib/", "/lib64/", "/usr/lib/", "/usr/lib64/")


def private_expat_from_ldd(ldd_output: str) -> str | None:
    """The path of the ``libexpat`` that ``pyexpat`` resolved to in ``ldd`` output, when it is not
    one of the distribution's own (``None`` for a system one, or when nothing was found).

    PyInstaller leaves ``libexpat`` out of a Linux bundle as a "system" library, so a Python built
    against a newer expat than the machine running the app (conda's, say, on Ubuntu 22.04) fails at
    start-up: ``undefined symbol: XML_SetAllocTrackerActivationThreshold``. Bundling the library
    the Python was actually linked against fixes that; a distribution's own expat is the one the
    user's system already has, so it is left alone."""
    match = re.search(r"^\s*libexpat\.so[.\d]*\s*=>\s*(\S+)", ldd_output, flags=re.MULTILINE)
    if match is None:
        return None
    path = match.group(1)
    # posixpath as well as os.path: ldd output is a POSIX path, and on
    # Windows (Python 3.13+) os.path.isabs("/home/...") is False because the
    # path has no drive -- which failed the tests there.
    if path.startswith(SYSTEM_LIB_PREFIXES) or not (posixpath.isabs(path)
                                                    or os.path.isabs(path)):
        return None
    return path


def stage_expat(path: str, stage_dir: str) -> str:
    """Copy the library under the name ``ldd`` reported (its SONAME, ``libexpat.so.1``), not the
    versioned file the link points at: ``pyexpat`` asks for the SONAME, and a bundle holding only
    ``libexpat.so.1.12.2`` leaves the system's older copy to answer."""
    import shutil
    os.makedirs(stage_dir, exist_ok=True)
    staged = os.path.join(stage_dir, os.path.basename(path))
    shutil.copy2(os.path.realpath(path), staged)
    return staged


def _expat_args(platform: str | None = None, run=subprocess.run, stage_dir: str | None = None) -> list:
    """``--add-binary`` arguments bundling a private ``libexpat`` (Linux only; see above)."""
    platform = sys.platform if platform is None else platform
    if not platform.startswith("linux"):
        return []
    import importlib.util
    spec = importlib.util.find_spec("pyexpat")
    origin = getattr(spec, "origin", None)
    if not origin or not os.path.isfile(origin):
        return []
    try:
        output = run(["ldd", origin], capture_output=True, text=True, stdin=subprocess.DEVNULL).stdout
    except OSError:
        return []
    path = private_expat_from_ldd(output)
    if path is None:
        return []
    staged = stage_expat(path, stage_dir or os.path.join(BUILD_DIR, "expat"))
    print(f"Bundling the Python's own libexpat: {path} -> {os.path.basename(staged)}")
    return ["--add-binary", f"{staged}{os.pathsep}."]


def _fix_qt_plugins() -> None:
    """Overwrite every bundled ``PySide6/Qt/plugins`` directory with a
    fresh copy straight from the *installed* PySide6 package.

    Real bug found and fixed while building/verifying this on macOS: a
    dev environment with a conda-installed ``qtbase`` package
    alongside pip-installed PySide6 (e.g. pulled in by ``qtconsole``/
    ``PyQt6`` -- not something opennmr itself needs) can make
    PyInstaller collect *that* conda Qt's platform plugins
    (``libqcocoa.dylib`` etc., a different, older Qt version, built
    with a completely different linkage convention -- flat
    ``libQt6Core.6.dylib`` vs. PySide6's own ``QtCore.framework``)
    instead of PySide6's own matching ones. The core Qt libraries
    bundled alongside (``QtCore``/``QtGui``/...) are unaffected --
    only the *plugins* were ever seen to be substituted -- but the
    version mismatch is enough that Qt refuses to load the platform
    plugin at all ("Ignoring QPA plugin due to mismatching Qt
    versions"), and the app aborts on launch with no window ever
    shown. Copying PySide6's own plugins directory over whatever
    PyInstaller collected sidesteps the mismatch entirely, regardless
    of why its own collection picked the wrong one."""
    import PySide6

    source_plugins = os.path.join(os.path.dirname(PySide6.__file__), "Qt", "plugins")
    sync_qt_plugins(DIST_DIR, source_plugins)


def sync_qt_plugins(dist_dir: str, source_plugins: str) -> list[str]:
    """Replace every bundled ``PySide6/Qt/plugins`` under *dist_dir* with a
    copy of *source_plugins*; return the real directories that were synced.

    Symlinks are resolved rather than removed: inside a macOS ``.app``
    PyInstaller keeps the real files in ``Contents/Frameworks`` and
    leaves symlinks to them under ``Contents/Resources``.
    ``shutil.rmtree`` refuses to run on a symlink ("Cannot call rmtree
    on a symbolic link"), which broke the 0.3.0 macOS CI build and a Mac
    build again on 2026-10-24. Each real directory is synced once, so a
    link and its target are not handled twice, and a dangling link is
    skipped."""
    import shutil

    if not os.path.isdir(source_plugins):
        return []

    synced: list[str] = []
    seen: set[str] = set()
    for root, dirs, _files in os.walk(dist_dir):
        if os.path.basename(root) != "PySide6" or "Qt" not in dirs:
            continue
        bundled = os.path.join(root, "Qt", "plugins")
        if not os.path.isdir(bundled):  # follows links; False if dangling
            continue
        target = os.path.realpath(bundled)
        if target in seen:
            continue
        seen.add(target)
        print(f"Re-syncing Qt plugins: {bundled}"
              + (f" -> {target}" if target != os.path.abspath(bundled) else ""))
        shutil.rmtree(target)
        shutil.copytree(source_plugins, target, symlinks=True)
        synced.append(target)
    return synced


def _resign_macos_app() -> None:
    """Sign the .app again (ad hoc) after its files were changed: macOS
    refuses to run a bundle whose contents no longer match its signature
    (a hard "damaged" failure on Apple silicon, not the Gatekeeper prompt
    a user can click through). A real Developer ID, if there ever is one,
    replaces this step."""
    import subprocess

    app = os.path.join(DIST_DIR, f"{APP_NAME}.app")
    if os.path.isdir(app):
        print(f"Re-signing {app} (ad hoc)")
        subprocess.run(["codesign", "--force", "--deep", "--sign", "-", app], check=True)


def app_version() -> str:
    """The version in pyproject.toml -- the one the installers are named
    after and the update check compares."""
    try:
        import tomllib
    except ImportError:  # Python 3.10
        import tomli as tomllib  # type: ignore[no-redef]
    with open(os.path.join(REPO_ROOT, "pyproject.toml"), "rb") as handle:
        return tomllib.load(handle)["project"]["version"]


def _stamp_macos_version() -> None:
    """Write the version into the .app's Info.plist (Finder's Get Info,
    and macOS's own "is this newer" when replacing the app), which
    PyInstaller's command line leaves at 0.0.0."""
    plist_path = os.path.join(DIST_DIR, f"{APP_NAME}.app", "Contents", "Info.plist")
    if not os.path.exists(plist_path):
        return
    import plistlib
    with open(plist_path, "rb") as handle:
        plist = plistlib.load(handle)
    version = app_version()
    plist.update({"CFBundleShortVersionString": version, "CFBundleVersion": version,
                  "CFBundleDisplayName": APP_NAME, "NSHighResolutionCapable": True})
    with open(plist_path, "wb") as handle:
        plistlib.dump(plist, handle)


def main() -> int:
    try:
        import PyInstaller.__main__
    except ImportError:
        print(
            "PyInstaller isn't installed -- it's a build-time-only tool, not "
            "a runtime dependency of opennmr itself. Install it with:\n"
            "    pip install -e \".[build]\"",
            file=sys.stderr,
        )
        return 1

    args = [
        ENTRY_POINT,
        "--name", APP_NAME,
        "--windowed",  # no console window -- this is a GUI app
        "--noconfirm",
        "--distpath", DIST_DIR,
        "--workpath", os.path.join(BUILD_DIR, "pyinstaller"),
        "--specpath", BUILD_DIR,
        # opennmr doesn't use networking through Qt (update.py uses
        # plain urllib) -- excluded because PyInstaller's own
        # hook-PySide6.QtNetwork.py crashes (SIGABRT, in its isolated
        # OpenSSL-detection subprocess) in at least one real dev
        # environment this was built/verified in.
        "--exclude-module", "PySide6.QtNetwork",
        # opennmr uses PySide6, never PyQt6 -- but matplotlib's own
        # Qt-backend detection code imports both bindings if they're
        # both installed, and PyInstaller refuses to freeze an app
        # that pulls in two competing Qt bindings packages at once.
        # gui/app.py already has to work around the two coexisting at
        # runtime too (see its QT_PLUGIN_PATH comment) -- same root
        # cause, different failure mode.
        "--exclude-module", "PyQt6",
        # The window logo and the toolbar icons are loaded from files
        # beside gui/, which PyInstaller does not collect on its own.
        "--add-data", f"{RESOURCES_DIR}{os.pathsep}opennmr/gui/resources",
        # The TopSpin plugin template is a .py file read as data, which
        # PyInstaller would otherwise only compile, not ship as a file.
        "--add-data", f"{TOPSPIN_RESOURCES}{os.pathsep}opennmr/topspin/resources",
        # opennmr's own version metadata: without it importlib.metadata
        # finds nothing in the frozen app and _version.py silently falls
        # back to its hard-coded string (and the update check compares
        # against the wrong version). selftest's "version" step checks it.
        "--copy-metadata", "opennmr",
        # Every opennmr module, including ones only imported lazily
        # inside functions (the GUI defers heavy imports for start-up
        # time), which PyInstaller's static analysis can miss.
        "--collect-submodules", "opennmr",
        # pymatgen (Simulate/Fit > pNMR from a CIF) reads its periodic
        # table and other tables from JSON/YAML files at import.
        "--collect-data", "pymatgen",
        # Never used by the app; keep them (and their size) out.
        "--exclude-module", "PyQt5",
        "--exclude-module", "PySide2",
        "--exclude-module", "pytest",
        "--exclude-module", "nmrglue",
        "--osx-bundle-identifier", BUNDLE_ID,
    ]
    for backend in MATPLOTLIB_EXPORT_BACKENDS:
        args += ["--hidden-import", backend]
    args += _expat_args()
    boot_splash = _make_boot_splash()
    if boot_splash:
        args += ["--splash", boot_splash]       # needs tkinter, so it is not excluded
    else:
        args += ["--exclude-module", "tkinter"]
    icon = _make_icon()
    if icon:
        args += ["--icon", icon]

    print(f"Building opennmr for {sys.platform}...")
    PyInstaller.__main__.run(args)
    _fix_qt_plugins()
    _stamp_macos_version()
    # LAST: every change to the bundle above (plugins, Info.plist) breaks
    # its signature, and macOS then reports the app as "damaged" the
    # moment it is downloaded (found on a real Mac: "invalid Info.plist
    # (plist or signature have been modified)").
    if sys.platform == "darwin":
        _resign_macos_app()
    print(f"Done -- see {DIST_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
