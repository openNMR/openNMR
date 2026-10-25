# Packaging, installers and updates

openNMR ships two ways, built from the **same** `src/opennmr` package.
Neither replaces the other.

| | Who it's for | How it's installed | How it's updated |
|---|---|---|---|
| **Desktop app** | GUI users | `openNMR-<ver>-windows-setup.exe`, `…-macos-<arch>.dmg`, `…-linux-x86_64.AppImage` | Settings ▸ Updates ▸ Download, run the new installer |
| **Python package** | notebooks (Jupyter, Colab), scripts, developers | `pip install …` (below) | `pip install --upgrade …` / `git pull` |

The app is a frozen copy of the package with its own Python inside
(PyInstaller). A notebook cannot import from it, so notebook users pip
install the package; the GUI is in that too (`opennmr-gui`). The two are
separate copies, so updating one never changes the other.

## The package (notebooks)

```
pip install opennmr                          # from PyPI
pip install -e ".[test]"                     # from a clone (developers)
pip install "git+https://github.com/openNMR/openNMR.git"   # straight from GitHub
```

While the repository is **private**, the `git+https` forms need GitHub
credentials (a read-only token in the URL, kept out of shared notebooks).
`pip install opennmr` needs none.

## The app

### Build, check and package on your own machine

```
pip install -e ".[build]"
python packaging/build.py                 # -> dist/openNMR/ (or dist/openNMR.app)
python -m opennmr.selftest st-source      # the self-test, from source
dist/openNMR/openNMR --self-test st-app   # ... and inside the built app
python packaging/compare_selftest.py st-source st-app
python packaging/make_installer.py        # -> dist/openNMR-<ver>-<platform>...
```

On Windows the built app is `dist\openNMR\openNMR.exe`; on macOS
`dist/openNMR.app/Contents/MacOS/openNMR`. `make_installer.py` needs
Inno Setup 6 on Windows and `appimagetool` on Linux (see its docstring);
macOS's `hdiutil` is built in.

**The self-test** (`src/opennmr/selftest.py`) is how a build is judged,
not "it opened". Inside the built app it imports every module, checks
the version metadata, reads a synthetic Bruker 1D and 2D, runs a CSA
simulation and a least-squares fit, reads a CIF with pymatgen, saves the
1D/2D/CSA figures and captures the main window with a spectrum open.
`compare_selftest.py` then requires the same numbers and pixel-identical
images to a run from source on the same machine. A `diff_*.png` marks
any pixels that differ.

### Making a release (the whole workflow)

You edit the code and commit as usual. To release:

```
python packaging/release.py 0.8.2            # checks, bumps the version, builds + checks the wheel/sdist, commits, tags
python packaging/release.py 0.8.2 --publish  # ... and pushes the commit + tag and uploads to PyPI
```

`release.py` is the only place the version is changed (it writes
`pyproject.toml` and the fallback in `_version.py` together). It refuses
if the tree is dirty, the version is not newer, or the tag exists, and
`--tests` runs the suite first. Nothing leaves your machine without
`--publish` (or your own `git push`).

Pushing the tag starts `.github/workflows/build.yml` ("Release"), which:

1. builds the app on Windows, macOS (Apple silicon, Intel) and Linux,
   self-tests it three ways (from source, in the frozen app compared
   pixel for pixel, in the *installed* copy);
2. **only if all four pass**, publishes to PyPI (trusted publishing; a
   version already uploaded by `--publish` is skipped) and creates the
   GitHub Release with the installers and a `SHA256SUMS` file.

Settings ▸ Updates then offers the new version to every installed app,
downloads the installer for that platform, checks it against
`SHA256SUMS` and replaces the running copy.

**Run it by hand** (Actions ▸ Release ▸ Run workflow): builds and checks
everything and uploads the installers as artifacts; publishes nothing.

**No Actions minutes?** A private repository's allowance is monthly and
macOS jobs count ten times (Windows twice); a full release is roughly
400-500 billable minutes, so keep the long test workflow off every push.
A public repository runs Actions free. Without minutes: `release.py
--publish` still does PyPI, and an installer can be built on any machine
with `build.py` + `make_installer.py` (above) and attached to a GitHub
Release by hand together with a `SHA256SUMS` made by `sha256sum`.

### What users see (unsigned builds)

The builds are not code-signed yet. They work, but:

- **Windows**: SmartScreen says "Windows protected your PC" → *More info*
  → *Run anyway*. No administrator rights are needed (per-user install,
  `%LOCALAPPDATA%\Programs\openNMR`, Start-menu entry, uninstaller in
  Settings ▸ Apps).
- **macOS**: drag openNMR to Applications; the first launch is blocked
  ("cannot be opened"/"Apple could not verify") → System Settings ▸
  Privacy & Security ▸ *Open Anyway*. Download the `arm64` DMG for Apple
  silicon and `x86_64` for Intel Macs.
- **Linux**: `chmod +x openNMR-*.AppImage` and run it. Machines without
  FUSE 2 can run it with `--appimage-extract-and-run`.

Removing those warnings needs paid certificates: an Apple Developer ID
(notarisation, Apple Developer Program) and a Windows code-signing
certificate or Azure Trusted Signing. Both slot into `build.yml` as
extra steps; nothing else changes.

### Known limitations

- The **TopSpin button** (`opennmr-topspin-install`) is for pip
  installs. Its plugin starts the GUI with `python -m`, which the app
  does not have. It is not offered in the app's GUI.
- The app is ~440 MB unpacked (Qt, SciPy, pymatgen). Most of that is
  Qt and pymatgen; excluding unused Qt modules is the obvious next
  saving if it matters.

### Environment issues met while building (kept from 2026-08-29)

1. An obsolete **`pathlib` backport** package (`pip install pathlib`)
   makes PyInstaller refuse to run: `pip uninstall pathlib`.
2. **PySide6 + PyQt6** installed side by side: PyInstaller refuses to
   collect two Qt bindings; `build.py` excludes PyQt6.
3. A conda **`qtbase`** beside pip PySide6 can make PyInstaller bundle
   the wrong Qt plugins (the app aborts at launch); `build.py`'s
   `_fix_qt_plugins` copies PySide6's own over them.
4. `PySide6.QtNetwork` is excluded: PyInstaller's hook for it crashed in
   one environment, and openNMR does not use it (the update check uses
   `urllib`).

## Updates: `src/opennmr/update.py` and Settings ▸ Updates

Settings ▸ Updates shows the running version and how it was installed,
then checks for a newer one. For the app it offers a Download button
(this platform's installer). For a pip install it shows the
`pip`/`git pull` command with a Copy button. It can also check quietly
at start-up, at most once a day. It never downloads or installs
anything itself: the installers already upgrade in place, and pip
environments are the user's to change.

Where it looks (Settings ▸ Updates ▸ Releases from):

- **`owner/repo`** (default `openNMR/openNMR`): that repository's
  latest GitHub Release — exactly what a version tag publishes above.
  The installer is picked from the release's files by name
  (`update.ASSET_PATTERNS`, matching `make_installer.py`'s names).
- **an `https://` address**: a JSON manifest
  (`version_manifest.example.json`), for hosting releases elsewhere.

**While the repository is private**, GitHub answers "not found" to
anyone without access, and the panel says so. A collaborator can paste
a fine-grained GitHub token with *Contents: read-only* into **Access
token** (or set `OPENNMR_GITHUB_TOKEN`), and the browser download works
if they are logged in to GitHub. For users outside the project, the
usual pattern is a **public releases-only repository** (installers and
release notes, no source), set as the source here. When the main
repository goes public, nothing needs changing.

From a notebook:

```python
from opennmr import update
result = update.check()                      # never raises
print(result.status, result.message or result.info)
print(update.update_instructions(update.install_kind()))
```

`OPENNMR_UPDATE_CHECK=0` turns the start-up check off (the test suite
sets it).
