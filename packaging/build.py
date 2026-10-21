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
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESOURCES_DIR = os.path.join(REPO_ROOT, "src", "opennmr", "gui", "resources")
LOGO_PNG = os.path.join(RESOURCES_DIR, "logo.png")
ENTRY_POINT = os.path.join(REPO_ROOT, "packaging", "entrypoint.py")
BUILD_DIR = os.path.join(REPO_ROOT, "build")
DIST_DIR = os.path.join(REPO_ROOT, "dist")


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
    on a symbolic link"), which broke the 0.3.0 macOS CI build. Each real
    directory is synced once, so a link and its target are not handled
    twice, and a dangling link is skipped."""
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
        shutil.copytree(source_plugins, target)
        synced.append(target)
    return synced


def _resign_macos_app() -> None:
    """Re-apply an ad-hoc signature to the ``.app`` after the plugin
    swap: PyInstaller signed the bundle before ``_fix_qt_plugins`` changed
    files inside it, and an invalid seal makes arm64 macOS kill the app."""
    if not sys.platform.startswith("darwin"):
        return
    import subprocess

    app = os.path.join(DIST_DIR, "opennmr.app")
    if os.path.isdir(app):
        subprocess.run(
            ["codesign", "--force", "--deep", "--sign", "-", app], check=False
        )


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
        "--name", "opennmr",
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
    ]
    icon = _make_icon()
    if icon:
        args += ["--icon", icon]

    print(f"Building opennmr for {sys.platform}...")
    PyInstaller.__main__.run(args)
    _fix_qt_plugins()
    _resign_macos_app()
    print(f"Done -- see {DIST_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
