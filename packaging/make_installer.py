"""Wrap the app built by ``packaging/build.py`` in this platform's own
installer -- the step after the build, on the same machine:

=========  ===============================================  ==========================
Platform   Output (in ``dist/``)                            Needs
=========  ===============================================  ==========================
Windows    ``openNMR-<ver>-windows-setup.exe``              Inno Setup 6 (``iscc``)
macOS      ``openNMR-<ver>-macos-<arm64|x86_64>.dmg``       ``hdiutil`` (built in)
Linux      ``openNMR-<ver>-linux-x86_64.AppImage``          ``appimagetool``
=========  ===============================================  ==========================

Usage::

    python packaging/build.py
    python packaging/make_installer.py

The names are what :data:`opennmr.update.ASSET_PATTERNS` looks for in a
GitHub Release, so a release made by ``.github/workflows/build.yml`` is
found by Settings ▸ Updates without further configuration.

``appimagetool`` is looked for on ``PATH`` or in ``$APPIMAGETOOL``
(``.github/workflows/build.yml`` downloads it). ``iscc`` is looked for
on ``PATH``, then in Inno Setup's usual install folders.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build import APP_NAME, BUILD_DIR, DIST_DIR, LOGO_PNG, REPO_ROOT, app_version  # noqa: E402

PACKAGING = os.path.join(REPO_ROOT, "packaging")


def _arch() -> str:
    machine = platform.machine().lower()
    return {"amd64": "x86_64", "x64": "x86_64", "aarch64": "arm64"}.get(machine, machine)


def _find_iscc() -> str | None:
    found = shutil.which("iscc") or shutil.which("ISCC")
    if found:
        return found
    for base in (os.environ.get("ProgramFiles(x86)"), os.environ.get("ProgramFiles"),
                 os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs")):
        if base:
            candidate = os.path.join(base, "Inno Setup 6", "ISCC.exe")
            if os.path.exists(candidate):
                return candidate
    return None


def windows_installer(version: str) -> str:
    source = os.path.join(DIST_DIR, APP_NAME)
    if not os.path.exists(os.path.join(source, f"{APP_NAME}.exe")):
        raise SystemExit(f"No {APP_NAME}.exe in {source} -- run packaging/build.py first.")
    iscc = _find_iscc()
    if iscc is None:
        raise SystemExit("Inno Setup's iscc was not found. Install Inno Setup 6 "
                         "(https://jrsoftware.org/isinfo.php, or: choco install innosetup).")
    command = [iscc, f"/DAppVersion={version}", f"/DSourceDir={source}", f"/DOutputDir={DIST_DIR}"]
    icon = os.path.join(BUILD_DIR, "icon.ico")
    if os.path.exists(icon):
        command.append(f"/DIconFile={icon}")
    command.append(os.path.join(PACKAGING, "windows", "openNMR.iss"))
    subprocess.run(command, check=True)
    return os.path.join(DIST_DIR, f"openNMR-{version}-windows-setup.exe")


def macos_dmg(version: str) -> str:
    app = os.path.join(DIST_DIR, f"{APP_NAME}.app")
    if not os.path.isdir(app):
        raise SystemExit(f"No {app} -- run packaging/build.py first.")
    staging = os.path.join(BUILD_DIR, "dmg")
    shutil.rmtree(staging, ignore_errors=True)
    os.makedirs(staging)
    # symlinks=True: the .app's frameworks are full of symlinks, and
    # copying their targets instead breaks the code signature.
    shutil.copytree(app, os.path.join(staging, f"{APP_NAME}.app"), symlinks=True)
    os.symlink("/Applications", os.path.join(staging, "Applications"))
    out = os.path.join(DIST_DIR, f"openNMR-{version}-macos-{_arch()}.dmg")
    volume = f"openNMR {version}"
    _detach_volume(volume)
    # HFS+ rather than the default (APFS on recent macOS), whose image
    # creation from a folder is the usual source of "hdiutil: create
    # failed - Resource busy" (a friend's Mac, 2026-10-27); HFS+ also
    # opens on every macOS the app supports.
    command = ["hdiutil", "create", "-volname", volume, "-srcfolder", staging, "-fs", "HFS+",
               "-ov", "-format", "UDZO", out]
    _run_with_retries(command, out, volume)
    return out


#: How often, and how long apart (s), ``hdiutil create`` is tried. A busy
#: resource is usually Spotlight or the malware scan looking at the newly
#: signed app or the scratch image, which lets go within seconds.
DMG_ATTEMPTS = 5
DMG_PAUSE_S = 3.0


def _detach_volume(volume: str) -> None:
    """Eject ``/Volumes/<volume>`` if an earlier run (or a double-click on
    an earlier .dmg) left it mounted: a volume of the same name keeps the
    new image's from mounting ("Resource busy")."""
    mount = os.path.join("/Volumes", volume)
    if os.path.ismount(mount) or os.path.isdir(mount):
        subprocess.run(["hdiutil", "detach", mount, "-force"], check=False)


def _run_with_retries(command, out, volume, attempts=None, pause=None, sleep=None) -> None:
    """``command`` (``hdiutil create``), tried again after a pause when it
    fails -- each time with a stale volume ejected and the partial image
    removed. Raises the last failure after ``attempts`` tries."""
    import time
    attempts = DMG_ATTEMPTS if attempts is None else attempts
    pause = DMG_PAUSE_S if pause is None else pause
    sleep = time.sleep if sleep is None else sleep
    for attempt in range(1, attempts + 1):
        try:
            subprocess.run(command, check=True)
            return
        except subprocess.CalledProcessError:
            if attempt == attempts:
                raise
            print(f"hdiutil create failed (attempt {attempt} of {attempts}); trying again in "
                  f"{pause * attempt:g} s", flush=True)
            if os.path.exists(out):
                os.remove(out)
            _detach_volume(volume)
            sleep(pause * attempt)


def _copy_text_lf(source: str, target: str) -> None:
    """Copy a text file with Unix line endings. A Windows checkout may have
    turned these into CRLF, and a shell script with CRLF fails on Linux
    ("bad interpreter") while appimagetool rejects a CRLF .desktop file."""
    with open(source, "rb") as handle:
        data = handle.read().replace(b"\r\n", b"\n")
    with open(target, "wb") as handle:
        handle.write(data)


def linux_appimage(version: str) -> str:
    source = os.path.join(DIST_DIR, APP_NAME)
    if not os.path.exists(os.path.join(source, APP_NAME)):
        raise SystemExit(f"No {APP_NAME} in {source} -- run packaging/build.py first.")
    appdir = os.path.join(BUILD_DIR, "openNMR.AppDir")
    shutil.rmtree(appdir, ignore_errors=True)
    shutil.copytree(source, os.path.join(appdir, "usr", "lib", APP_NAME), symlinks=True)
    _copy_text_lf(os.path.join(PACKAGING, "linux", "AppRun"), os.path.join(appdir, "AppRun"))
    os.chmod(os.path.join(appdir, "AppRun"), 0o755)
    _copy_text_lf(os.path.join(PACKAGING, "linux", "openNMR.desktop"),
                  os.path.join(appdir, "openNMR.desktop"))
    shutil.copy2(LOGO_PNG, os.path.join(appdir, "openNMR.png"))
    tool = os.environ.get("APPIMAGETOOL") or shutil.which("appimagetool")
    out = os.path.join(DIST_DIR, f"openNMR-{version}-linux-{_arch()}.AppImage")
    if tool is None:
        raise SystemExit("appimagetool not found (set APPIMAGETOOL or put it on PATH): "
                         "https://github.com/AppImage/appimagetool/releases")
    env = dict(os.environ, ARCH=_arch(), APPIMAGE_EXTRACT_AND_RUN="1")
    subprocess.run([tool, "--no-appstream", appdir, out], check=True, env=env)
    return out


def main() -> int:
    version = app_version()
    if sys.platform.startswith("win"):
        out = windows_installer(version)
    elif sys.platform == "darwin":
        out = macos_dmg(version)
    else:
        out = linux_appimage(version)
    print(f"Installer: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
