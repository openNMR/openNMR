"""
``opennmr-topspin-install`` -- put an OpenNMR button on TopSpin's toolbar.

Finds the TopSpin installations on this computer, asks which to use,
and copies the plugin ``opennmr.py`` into each one's
``exp/stan/nmr/py/user/`` folder. Typing ``opennmr`` in TopSpin then adds
the button, and ``opennmr -kill`` removes it. Optionally adds ``opennmr``
to the user's ``autostart.mac`` so the button appears on every start.

The plugin gets the exact Python running this installer (and, for conda,
the folders ``conda activate`` would put on PATH) written into it, so
TopSpin never has to find OpenNMR on PATH -- which fails for conda,
venvs, and apps started from the macOS Dock. Re-run the installer after
moving or recreating the environment. The icon is read straight from
the installed package.

TopSpin locations searched (``--topspin FOLDER`` names them instead):

* ``$XWINNMRHOME`` and ``topspin`` on PATH (both Linux conventions)
* macOS/Linux: ``/opt/topspin*``, ``/opt/*/topspin*``, and the folder each
  ``/Applications/TopSpin*.app`` launcher starts
* Windows: ``\\Bruker\\TopSpin*`` on every fixed drive, and TopSpin entries
  in the registry's list of installed programs
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path

from .._version import __version__

PLUGIN_NAME = "opennmr.py"
PLUGIN_SUBDIR = Path("exp", "stan", "nmr", "py", "user")
PLUGIN_TEMPLATE = Path(__file__).parent / "resources" / PLUGIN_NAME
# Also on the first line of the template's docstring, so plugins copied
# by hand before this installer existed are recognised too.
PLUGIN_MARK = "OpenNMR toolbar launcher for TopSpin"
ICON = Path(__file__).resolve().parents[1] / "gui" / "resources" / "logo.png"
GUI_MODULE = "opennmr.gui.app"

AUTOSTART_COMMAND = "opennmr"
AUTOSTART_BEGIN = "# BEGIN OPENNMR AUTOSTART"
AUTOSTART_END = "# END OPENNMR AUTOSTART"
_AUTOSTART_BLOCK = re.compile(
    r"(?m)^[ \t]*" + re.escape(AUTOSTART_BEGIN) + r".*?"
    + re.escape(AUTOSTART_END) + r"[ \t]*(\r?\n)?", re.S)
_SETTINGS_BLOCK = re.compile(
    r"(?ms)^# BEGIN INSTALLER SETTINGS$.*?^# END INSTALLER SETTINGS$")

IS_WINDOWS = os.name == "nt"
TESTED_FROM = (3, 5)  # button verified in TopSpin 3.5 pl7, 4.5 and 5.0


# -- TopSpin installations ---------------------------------------------------

@dataclass(frozen=True)
class TopSpin:
    """One TopSpin installation folder."""

    home: Path
    version: tuple[int, ...] = ()  # () when the folder name has no version

    @property
    def label(self) -> str:
        return "TopSpin " + (".".join(map(str, self.version)) or "(unknown version)")

    @property
    def plugin(self) -> Path:
        return self.home / PLUGIN_SUBDIR / PLUGIN_NAME

    @property
    def tested(self) -> bool:
        return self.version >= TESTED_FROM

    def has_plugin(self) -> bool:
        try:
            return PLUGIN_MARK in self.plugin.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return False

    def user_dir(self) -> Path | None:
        """The user's settings folder for this TopSpin (``~/.topspin1`` or
        similar), as TopSpin's own ``pathScript dot`` reports it."""
        for name in ("pathScript", "pathScript.exe", "pathScript.cmd", "pathScript.bat"):
            tool = self.home / name
            if not tool.is_file():
                continue
            env = dict(os.environ, HOME=str(user_home()))
            try:
                result = subprocess.run(
                    [str(tool), "dot"], capture_output=True, text=True,
                    timeout=20, env=env, cwd=str(self.home))
            except (OSError, subprocess.SubprocessError):
                return None
            lines = result.stdout.strip().splitlines()
            path = Path(lines[-1].strip()) if lines else None
            return path if path is not None and path.is_absolute() else None
        return None


def is_topspin_home(path: Path) -> bool:
    return (path / "exp" / "stan" / "nmr" / "py").is_dir() and (path / "prog").is_dir()


def version_from_name(name: str) -> tuple[int, ...]:
    """``"topspin4.5.0"`` / ``"TopSpin 5.0.0"`` -> ``(4, 5, 0)`` / ``(5, 0, 0)``."""
    match = re.search(r"topspin[\s_-]*(\d+(?:\.\d+)*)", name, re.I)
    return tuple(int(part) for part in match.group(1).split(".")) if match else ()


def find_topspin_installs() -> list[TopSpin]:
    """Every TopSpin installation found, newest first."""
    found: dict[Path, TopSpin] = {}
    for candidate in _candidate_dirs():
        try:
            home = Path(candidate).expanduser().resolve()
        except OSError:
            continue
        if home not in found and is_topspin_home(home):
            found[home] = TopSpin(home, version_from_name(home.name))
    return sorted(found.values(), key=lambda t: t.version, reverse=True)


def _candidate_dirs():
    if os.environ.get("XWINNMRHOME"):
        yield os.environ["XWINNMRHOME"]
    launcher = shutil.which("topspin")
    if launcher:
        yield os.path.dirname(os.path.realpath(launcher))
    if IS_WINDOWS:
        for drive in _windows_fixed_drives():
            yield from glob.glob(os.path.join(drive, "Bruker", "TopSpin*"))
        yield from _windows_registry_dirs()
        return
    for pattern in ("/opt/*", "/opt/*/*"):
        for path in glob.glob(pattern):
            if os.path.basename(path).lower().startswith("topspin"):
                yield path
    if sys.platform == "darwin":
        # Each app's launcher is a small shell script ending
        # ``exec /opt/topspin5.0.0/topspin``.
        for script in glob.glob("/Applications/TopSpin*.app/Contents/MacOS/topspin"):
            try:
                text = Path(script).read_text(errors="replace")
            except OSError:
                continue
            for target in re.findall(r"exec\s+\"?(/[^\"\n]*?)/topspin\"?\s*$", text, re.M):
                yield target


def _windows_fixed_drives():
    import ctypes
    import string

    kernel32 = ctypes.windll.kernel32
    mask = kernel32.GetLogicalDrives()
    for index, letter in enumerate(string.ascii_uppercase):
        root = letter + ":\\"
        # Fixed disks only: probing a disconnected network drive can hang.
        if mask >> index & 1 and kernel32.GetDriveTypeW(root) == 3:
            yield root


def _windows_registry_dirs():
    import winreg

    uninstall = r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"
    for hive in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
        for view in (winreg.KEY_WOW64_64KEY, winreg.KEY_WOW64_32KEY):
            try:
                key = winreg.OpenKey(hive, uninstall, 0, winreg.KEY_READ | view)
            except OSError:
                continue
            with key:
                for index in range(winreg.QueryInfoKey(key)[0]):
                    try:
                        with winreg.OpenKey(key, winreg.EnumKey(key, index)) as entry:
                            name = winreg.QueryValueEx(entry, "DisplayName")[0]
                            location = winreg.QueryValueEx(entry, "InstallLocation")[0]
                    except OSError:
                        continue
                    if "topspin" in str(name).lower() and location:
                        yield str(location)


# -- the plugin ----------------------------------------------------------------

def launch_settings() -> tuple[list[str], list[str]]:
    """``(command, path_prepend)`` that start the GUI with this Python."""
    # absolute(), not resolve(): a venv's python is a symlink to its base
    # interpreter, and resolving it would lose the venv.
    python = Path(sys.executable).absolute()
    if IS_WINDOWS and python.with_name("pythonw.exe").is_file():
        python = python.with_name("pythonw.exe")  # no console window
    prefix = Path(sys.prefix)
    if IS_WINDOWS:
        # What ``conda activate`` puts on PATH; without the Library folders
        # conda's Qt and numpy DLLs may not load. Missing ones (plain
        # Python, venvs) are skipped.
        folders = [prefix, prefix / "Library" / "mingw-w64" / "bin",
                   prefix / "Library" / "usr" / "bin", prefix / "Library" / "bin",
                   prefix / "Scripts", prefix / "bin"]
    else:
        folders = [prefix / "bin"]
    return ([str(python), "-m", GUI_MODULE],
            [str(folder) for folder in folders if folder.is_dir()])


def render_plugin(command: list[str], path_prepend: list[str], icon: Path) -> str:
    """The plugin template with its settings block filled in."""
    block = "\n".join([
        "# BEGIN INSTALLER SETTINGS",
        "# Written by opennmr-topspin-install %s; run it again to change these." % __version__,
        "COMMAND = %s" % _jython_list(command),
        "PATH_PREPEND = %s" % _jython_list(path_prepend),
        "ICON = %s" % _jython_str(str(icon)),
        "# END INSTALLER SETTINGS",
    ])
    template = PLUGIN_TEMPLATE.read_text(encoding="utf-8")
    text, count = _SETTINGS_BLOCK.subn(lambda _: block, template)
    if count != 1:
        raise RuntimeError("settings block not found in " + str(PLUGIN_TEMPLATE))
    return text


def _jython_str(value: str) -> str:
    # JSON string escapes are valid in a Python 2 unicode literal, so any
    # path (backslashes, quotes, non-ASCII user names) comes through intact.
    return "u" + json.dumps(value)


def _jython_list(values: list[str]) -> str:
    return "[" + ", ".join(_jython_str(v) for v in values) + "]"


# -- autostart.mac -----------------------------------------------------------

def set_autostart(macro: Path, enabled: bool) -> bool:
    """Add or remove the ``opennmr`` block in ``macro``, leaving everything
    else in the file untouched. Returns True if the file changed."""
    try:
        with open(macro, encoding="utf-8", errors="surrogateescape", newline="") as handle:
            text = handle.read()
    except FileNotFoundError:
        text = ""
    new = _AUTOSTART_BLOCK.sub("", text)
    eol = "\r\n" if "\r\n" in text else "\n"
    if new != text:
        new = new.rstrip("\r\n") + eol  # drop the blank line added with the block
    if enabled:
        if not new.strip():
            new = "# TopSpin runs this file as a macro after it starts." + eol
        new = new.rstrip("\r\n") + eol + eol
        new += eol.join([AUTOSTART_BEGIN, AUTOSTART_COMMAND, AUTOSTART_END]) + eol
    if new == text:
        return False
    _write_text(macro, new, errors="surrogateescape")
    _give_to_invoking_user(macro)
    return True


def has_autostart(macro: Path) -> bool:
    try:
        return AUTOSTART_BEGIN in macro.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False


# -- files and users -----------------------------------------------------------

def user_home() -> Path:
    """The home of the person running this, also under ``sudo``."""
    sudo_user = os.environ.get("SUDO_USER")
    if not IS_WINDOWS and sudo_user and os.geteuid() == 0:
        return Path(os.path.expanduser("~" + sudo_user))
    return Path.home()


def _give_to_invoking_user(path: Path) -> None:
    """Under ``sudo``, keep files in the user's home owned by the user."""
    if not IS_WINDOWS and os.geteuid() == 0 and os.environ.get("SUDO_UID"):
        try:
            os.chown(path, int(os.environ["SUDO_UID"]), int(os.environ.get("SUDO_GID", -1)))
        except OSError:
            pass


def _write_text(path: Path, text: str, errors: str = "strict") -> None:
    """Write via a temporary file and a rename, so a failure never leaves
    half a file, and an existing file owned by someone else can still be
    replaced when the folder is writable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp = tempfile.mkstemp(dir=str(path.parent), prefix="." + path.name)
    try:
        with open(handle, "w", encoding="utf-8", errors=errors, newline="") as out:
            out.write(text)
        os.chmod(temp, 0o644)
        os.replace(temp, path)
    except BaseException:
        os.unlink(temp)
        raise


# -- command line ----------------------------------------------------------------

def _ask(question: str, default: str, assume_default: bool) -> str:
    if assume_default or not sys.stdin.isatty():
        return default
    try:
        answer = input("%s [%s]: " % (question, default)).strip()
    except EOFError:
        answer = ""
    return answer or default


def _choose(installs: list[TopSpin], verb: str, assume_default: bool) -> list[TopSpin]:
    if len(installs) == 1:
        answer = _ask("%s %s?" % (verb, installs[0].label), "yes", assume_default)
        return installs if answer.lower().startswith("y") else []
    while True:
        answer = _ask("%s which? 'all', numbers such as 1,3, or 'none'" % verb,
                      "all", assume_default).lower()
        if answer == "all":
            return installs
        if answer in ("none", "n", "q"):
            return []
        try:
            picks = [int(part) for part in re.split(r"[\s,]+", answer) if part]
            return [installs[pick - 1] for pick in dict.fromkeys(picks)
                    if 1 <= pick <= len(installs)]
        except ValueError:
            print("  Please answer 'all', 'none', or numbers from the list.")


def _print_installs(installs: list[TopSpin]) -> None:
    print("TopSpin installations found:")
    for number, install in enumerate(installs, 1):
        notes = [note for note, on in (("plugin installed", install.has_plugin()),
                                       ("untested version", not install.tested)) if on]
        print("  %d) %-18s %s%s" % (number, install.label, install.home,
                                   "   [%s]" % ", ".join(notes) if notes else ""))


def _user_dirs(installs: list[TopSpin]) -> dict[Path, list[TopSpin]]:
    """Group installs by the user folder (and so autostart.mac) they use.
    Several TopSpin versions often share one. When TopSpin cannot say,
    every ``~/.topspin*`` folder with a ``prop`` subfolder is assumed."""
    groups: dict[Path, list[TopSpin]] = {}
    guessed = [d for d in sorted(user_home().glob(".topspin*")) if (d / "prop").is_dir()]
    for install in installs:
        folder = install.user_dir()
        for each in [folder] if folder is not None else guessed:
            groups.setdefault(each, []).append(install)
    return groups


def _update_autostart(all_installs: list[TopSpin], enabled: bool | None) -> None:
    """Turn the autostart block on or off in every autostart.mac these
    installs use. With ``enabled=None`` only remove it, from files no
    TopSpin with the plugin still uses."""
    groups = _user_dirs(all_installs)
    if not groups:
        print("Could not find your TopSpin settings folder, so autostart.mac "
              "was not changed. Start TopSpin once, then run this again.")
        return
    for folder, sharing in groups.items():
        macro = folder / "prop" / "autostart.mac"
        plugin_left = any(install.has_plugin() for install in sharing)
        keep = has_autostart(macro) if enabled is None else enabled
        on = keep and plugin_left
        if on and not macro.parent.is_dir():
            print("  %s does not exist yet: start TopSpin once, then run this "
                  "again to enable autostart." % macro.parent)
            continue
        try:
            changed = set_autostart(macro, on)
        except OSError as exc:
            print("  Could not update %s: %s" % (macro, exc))
            continue
        print("  %s %s" % (macro, ("updated" if changed else "unchanged")))
        missing = [install.label for install in sharing if not install.has_plugin()]
        if on and missing:
            print("    Note: %s also use%s this file but %s no plugin; TopSpin will "
                  "report 'opennmr' as unknown at startup there."
                  % (", ".join(missing), "s" if len(missing) == 1 else "",
                     "has" if len(missing) == 1 else "have"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="opennmr-topspin-install",
        description="Add an OpenNMR button to TopSpin's toolbar.")
    parser.add_argument("--topspin", metavar="FOLDER", action="append", type=Path,
                        default=[], help="a TopSpin installation folder to use "
                        "instead of searching (repeatable)")
    parser.add_argument("--autostart", action=argparse.BooleanOptionalAction,
                        help="show the button whenever TopSpin starts "
                             "(asked when not given)")
    parser.add_argument("--uninstall", action="store_true",
                        help="remove the plugin and its autostart entry")
    parser.add_argument("--list", action="store_true",
                        help="only list the TopSpin installations found")
    parser.add_argument("-y", "--yes", action="store_true",
                        help="accept the defaults without asking")
    args = parser.parse_args(argv)

    if args.topspin:
        homes = [path.expanduser().resolve() for path in args.topspin]
        for home in homes:
            if not is_topspin_home(home):
                print("Not a TopSpin installation folder: %s" % home)
        installs = [TopSpin(home, version_from_name(home.name))
                    for home in dict.fromkeys(homes) if is_topspin_home(home)]
    else:
        installs = find_topspin_installs()
    if not installs:
        print("No TopSpin installation found. Name its folder with --topspin, "
              "e.g. --topspin /opt/topspin5.0.0 or --topspin C:\\Bruker\\TopSpin5.0.0")
        return 1
    _print_installs(installs)
    if args.list:
        return 0

    if args.uninstall:
        chosen = _choose([i for i in installs if i.has_plugin()] or installs,
                         "Remove the plugin from", args.yes)
        for install in chosen:
            try:
                if install.has_plugin():
                    install.plugin.unlink()
                print("Removed from %s" % install.label)
            except OSError as exc:
                print("Could not remove %s: %s" % (install.plugin, exc))
        _update_autostart(installs, enabled=None)
        return 0

    if find_spec("PySide6") is None:
        print("Warning: PySide6 is missing from this Python, so the OpenNMR "
              "GUI will not start. Install it with: %s -m pip install PySide6"
              % sys.executable)

    chosen = _choose(installs, "Install the plugin into", args.yes)
    if not chosen:
        print("Nothing installed.")
        return 0

    command, path_prepend = launch_settings()
    plugin = render_plugin(command, path_prepend, ICON)
    failed = False
    for install in chosen:
        try:
            _write_text(install.plugin, plugin)
            print("Installed into %s: %s" % (install.label, install.plugin))
        except OSError as exc:
            failed = True
            print("Could not write %s: %s" % (install.plugin, exc))
    if failed:
        print("Folders inside TopSpin can need administrator rights: re-run as "
              "the TopSpin administrator (on macOS/Linux: sudo \"%s\" -m "
              "opennmr.topspin.install)." % sys.executable)

    enabled = args.autostart
    if enabled is None:
        enabled = _ask("Show the OpenNMR button whenever TopSpin starts? (yes/no)",
                       "yes", args.yes).lower().startswith("y")
    _update_autostart(installs, enabled)

    print("\nOpenNMR starts with:\n  %s" % " ".join(command))
    print("In a running TopSpin, type 'opennmr' to add the button "
          "('opennmr -kill' first if an older button is showing).")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
