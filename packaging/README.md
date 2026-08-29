# Packaging / update framework

Not wired up yet, and not something you need to use day-to-day — this
exists so the mechanism is *built* ahead of actually needing it,
per an explicit request to have it ready rather than scrambling for it
later. Two independent pieces: **build** (this directory) and
**update-check** (`src/opennmr/update.py`).

## Build: a standalone app, the same way on every OS

```
pip install -e ".[build]"
python packaging/build.py
```

Produces a standalone app under `dist/` — a `.app` bundle on macOS, an
`.exe` + folder on Windows, an executable + folder on Linux — using
[PyInstaller](https://pyinstaller.org/). One script, not three
platform-specific processes: run it on whichever OS you're on, it
does the right thing there. **Verified end-to-end on macOS**
(2026-08-29): the built `.app` launches and runs correctly. Windows
and Linux haven't been tried yet — flag any issue that comes up there,
Qt-bundling problems in particular tend to be platform-specific.

### Two real environment issues found and worked around while verifying this

Both are `packaging/build.py`-level workarounds, not app bugs — noting
them here since either could resurface confusingly on a different
machine:

1. **An obsolete `pathlib` backport package** (`pip install pathlib` —
   a leftover from before Python 3.4, when `pathlib` wasn't in the
   standard library yet) makes PyInstaller refuse to run at all. If
   you hit `ERROR: The 'pathlib' package is an obsolete backport...`,
   `pip uninstall pathlib` (safe — nothing should actually import the
   backport once your Python is 3.4+, which is `opennmr`'s whole
   supported range).
2. **PySide6 + PyQt6 installed side by side** (as this dev environment
   has, apparently for unrelated reasons) makes PyInstaller abort with
   "attempt to collect multiple Qt bindings packages" — matplotlib's
   own Qt-backend detection imports both if both are present, even
   though `opennmr` only ever uses PySide6 (see `gui/app.py`'s own
   `QT_PLUGIN_PATH` comment for the *runtime* side of this same
   coexistence issue). `build.py` already passes
   `--exclude-module PyQt6` to work around it; if you *do* need PyQt6
   built in for some other reason, this'll need revisiting.

Also excludes `PySide6.QtNetwork` (`opennmr` doesn't use it —
`update.py` below uses plain `urllib`, not Qt networking) because
PyInstaller's own `hook-PySide6.QtNetwork.py` crashed outright
(`SIGABRT`) in its isolated OpenSSL-detection subprocess in this dev
environment. Worth revisiting if Qt networking is ever actually needed.

### Not yet built

- **Native installers** (a macOS DMG, a Windows MSI/installer exe, a
  Linux AppImage/`.deb`) wrapping the raw PyInstaller output above —
  what `build.py` produces today is a working standalone app folder,
  not yet a polished one-click installer.
- **Code signing / notarization** — the macOS build above is
  ad-hoc-signed only (PyInstaller's own `codesign -s -` step even
  half-failed with a benign "resource fork... not allowed" warning in
  this environment, worth a closer look before shipping anything).
  Unsigned/unnotarized builds will trip Gatekeeper warnings for anyone
  else who downloads one.
- **CI** — nothing runs `build.py` automatically; it's a manual local
  command right now.

## Update-check: `src/opennmr/update.py`

```python
from opennmr.update import check_for_update

info = check_for_update("https://your-host/manifest.json")
if info is not None:
    print(f"Update available: {info.version} -- {info.url}")
```

Fetches a small JSON manifest (see
`packaging/version_manifest.example.json`) and reports whether a
newer version is available for the current platform — nothing more.
**Deliberately read-only**: it never downloads or installs anything.
Actually replacing a running app's files safely is real,
platform-specific work (macOS app-bundle replacement, Windows
files-in-use locking, Linux package-manager-vs-standalone conventions
all differ) that depends on *where* builds actually end up being
hosted — not worth guessing at before there's a real distribution
channel to build it against. When that exists, the natural next step
is a GUI affordance (a menu item or startup check) calling this and
prompting the user to visit `info.url` — still manual, but wired up.

The manifest itself needs to be hosted somewhere reachable
(`check_for_update`'s only required input is that URL) — a GitHub raw
file, a static web host, whatever's convenient. Nothing about
`update.py` assumes a specific host.
