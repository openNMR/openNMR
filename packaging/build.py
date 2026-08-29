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
LOGO_SVG = os.path.join(REPO_ROOT, "src", "opennmr", "gui", "resources", "logo.svg")
ENTRY_POINT = os.path.join(REPO_ROOT, "packaging", "entrypoint.py")
BUILD_DIR = os.path.join(REPO_ROOT, "build")
DIST_DIR = os.path.join(REPO_ROOT, "dist")


def _rasterize_svg(svg_path: str, size: int = 1024) -> str | None:
    """Render the SVG logo to a square PNG via Qt's own SVG renderer
    (PySide6/QtSvg -- already a dependency, so this avoids adding a new
    third-party SVG-rasterization library just for this one-time
    icon-conversion step). Pillow, used below for the actual ICO/ICNS
    conversion, can't read SVG directly. Returns ``None`` if QtSvg
    isn't available or the SVG fails to render."""
    try:
        import PySide6
    except ImportError:
        return None

    # Same fix as gui/app.py's own QT_PLUGIN_PATH comment -- if another
    # Qt binding (e.g. PyQt6) is also installed, its plugins can shadow
    # PySide6's own at a different Qt version, and the offscreen
    # platform plugin (needed since no display is required just to
    # rasterize into an in-memory QImage) then fails to load at all.
    # Must happen before anything imports PySide6.QtCore/QtGui.
    os.environ.setdefault(
        "QT_PLUGIN_PATH", os.path.join(os.path.dirname(PySide6.__file__), "Qt", "plugins"),
    )
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    try:
        from PySide6.QtCore import QSize
        from PySide6.QtGui import QGuiApplication, QImage, QPainter
        from PySide6.QtSvg import QSvgRenderer
    except ImportError:
        return None

    QGuiApplication.instance() or QGuiApplication([])

    renderer = QSvgRenderer(svg_path)
    if not renderer.isValid():
        return None

    image = QImage(QSize(size, size), QImage.Format.Format_ARGB32)
    image.fill(0)  # transparent background
    painter = QPainter(image)
    renderer.render(painter)
    painter.end()

    os.makedirs(BUILD_DIR, exist_ok=True)
    png_path = os.path.join(BUILD_DIR, "logo_rasterized.png")
    if not image.save(png_path):
        return None
    return png_path


def _make_icon() -> str | None:
    """Convert the app's SVG logo to whatever icon format this
    platform's build wants (``.icns`` on macOS, ``.ico`` on Windows).
    Returns ``None`` on Linux (PyInstaller's own default icon is used
    instead -- desktop-entry icons are a separate, packaging-format-
    specific concern, not something PyInstaller itself sets), or if
    conversion fails for any reason -- icon polish is worth doing, not
    worth failing the whole build over."""
    if not os.path.exists(LOGO_SVG):
        return None
    try:
        from PIL import Image
    except ImportError:
        return None

    rasterized = _rasterize_svg(LOGO_SVG)
    if rasterized is None:
        return None

    os.makedirs(BUILD_DIR, exist_ok=True)
    img = Image.open(rasterized).convert("RGBA")

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
    ]
    icon = _make_icon()
    if icon:
        args += ["--icon", icon]

    print(f"Building opennmr for {sys.platform}...")
    PyInstaller.__main__.run(args)
    print(f"Done -- see {DIST_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
