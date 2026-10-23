"""Download, verify and install an update from inside openNMR, with no
terminal: the engine behind Settings ▸ Updates ▸ Install.

:mod:`opennmr.update` finds out that a newer version exists; this module
fetches it and hands over to the platform's own way of replacing a
running application. Nothing here needs Qt, so it is also usable from a
notebook (``update_install.fetch_installer`` / ``plan_install``).

The steps, in order, and why:

1. **Checksum first** (:func:`expected_sha256`). An installer is only
   downloaded if the release published its SHA-256 (the asset's own
   ``digest``, a ``SHA256SUMS`` file beside it, or the manifest's
   ``sha256``). A release without one is *not* installed automatically;
   the GUI offers the browser download instead. The checksum protects
   against a damaged or tampered download; it does **not** protect
   against a compromised release (whoever can replace the installer can
   replace its checksum) -- that needs code signing, which is not set up.
2. **Download** (:func:`download`) over https only, into a ``.part`` file,
   hashing as it streams; the file only gets its real name once the hash
   matches. A GitHub token is sent to ``api.github.com`` and dropped when
   a download is redirected elsewhere (:class:`_DropAuthOnRedirect`).
3. **Hand-off** (:func:`plan_install`, :func:`launch`). A running program
   cannot replace itself, so a small helper is started *detached* and the
   app then quits:

   * Windows: the Inno Setup installer, silent, per user or all users as
     the running copy was installed; it closes openNMR if still open and
     (``/RELAUNCH=1``, see ``packaging/windows/openNMR.iss``) starts it again.
   * macOS: a shell script waits for the app to exit, mounts the .dmg,
     swaps the ``.app`` bundle (keeping the old one until the new copy is
     in place, restoring it on failure) and opens it.
   * Linux: a shell script waits, then replaces the AppImage file
     (``$APPIMAGE``) and starts it.
   * If the app's folder is not writable (an ``/Applications`` copy owned by
     another user, say) the plan is *manual*: the installer is opened for
     the user, and no files are touched.

   Helpers log to ``~/.opennmr/update.log``.

A copy installed from PyPI is upgraded by :func:`plan_pip_upgrade` /
:func:`launch_upgrade`: a detached helper waits for the app to quit, runs
pip, checks the new version starts, goes back to the old one on any
failure, and reopens openNMR -- all silently. One installed from a git
URL is updated by :func:`pip_argv` / :func:`run_pip` (``git+https://…``).
A source folder (``pip install -e``) is never updated automatically: its
``git pull`` could overwrite the user's own work.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
import threading
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass

from . import update

CHUNK = 256 * 1024

#: The only host an access token is ever sent to.
TOKEN_HOSTS = {"api.github.com"}

LOG_PATH = os.path.join(os.path.expanduser("~"), ".opennmr", "update.log")


class InstallError(Exception):
    """Something stopped the update; the message is for the user."""


# ============================================================
# Downloading
# ============================================================

def _require_https(url: str) -> None:
    parts = urllib.parse.urlsplit(url)
    if parts.scheme == "https":
        return
    if parts.scheme == "http" and parts.hostname in ("localhost", "127.0.0.1", "::1"):
        return  # a local test server; never a real update source
    raise InstallError(f"Refusing to download from {url!r}: updates must come over https.")


class _DropAuthOnRedirect(urllib.request.HTTPRedirectHandler):
    """urllib copies every header (the token included) to the redirect's
    target. GitHub redirects an asset download to a signed storage URL,
    which rejects a second credential -- and a token has no business
    there -- so ``Authorization`` is removed when the host changes, and a
    redirect to plain http is refused."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _require_https(newurl)
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None:
            old_host = urllib.parse.urlsplit(req.full_url).netloc
            if urllib.parse.urlsplit(newurl).netloc != old_host:
                for store in (new.headers, new.unredirected_hdrs):
                    for key in [k for k in store if k.lower() == "authorization"]:
                        del store[key]
        return new


def _open(url: str, token: str | None = None, accept: str | None = None, timeout: float = 30.0):
    _require_https(url)
    headers = {"User-Agent": "openNMR-updater"}
    if accept:
        headers["Accept"] = accept
    if token and urllib.parse.urlsplit(url).hostname in TOKEN_HOSTS:
        headers["Authorization"] = f"Bearer {token}"
    opener = urllib.request.build_opener(
        _DropAuthOnRedirect, urllib.request.HTTPSHandler(context=update.ssl_context()))
    return opener.open(urllib.request.Request(url, headers=headers), timeout=timeout)


def asset_source(browser_url: str, api_url: str, token: str | None) -> tuple[str, str | None]:
    """(url, Accept header) to fetch a release file. A private
    repository's files can only be had through the API address with the
    token; a public one's browser address needs neither."""
    if token and api_url:
        return api_url, "application/octet-stream"
    return browser_url, None


def _network_message(exc: Exception) -> str:
    if isinstance(exc, urllib.error.HTTPError):
        return f"The download server answered {exc.code} ({exc.reason})."
    return f"Could not download the update ({getattr(exc, 'reason', exc)}). Check the internet connection."


def sha256_of(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, dest: str, *, token: str | None = None, accept: str | None = None,
             expected_sha256: str = "", progress=None, cancelled=None,
             timeout: float = 30.0) -> str:
    """Download ``url`` to ``dest`` and return its SHA-256.

    ``progress(done, total)`` (``total`` 0 when unknown) and
    ``cancelled()`` are called between chunks. The data goes to
    ``dest + ".part"`` and is renamed only if complete and, when
    ``expected_sha256`` is given, matching; otherwise the part file is
    removed and :class:`InstallError` raised."""
    part = dest + ".part"
    digest = hashlib.sha256()
    try:
        try:
            with _open(url, token, accept, timeout) as response, open(part, "wb") as out:
                total = int(response.headers.get("Content-Length") or 0)
                done = 0
                while True:
                    if cancelled is not None and cancelled():
                        raise InstallError("The update was cancelled.")
                    chunk = response.read(CHUNK)
                    if not chunk:
                        break
                    out.write(chunk)
                    digest.update(chunk)
                    done += len(chunk)
                    if progress is not None:
                        progress(done, total)
        except http.client.IncompleteRead as exc:
            raise InstallError("The download was cut short. Try again.") from exc
        except (urllib.error.URLError, http.client.HTTPException, OSError, TimeoutError) as exc:
            raise InstallError(_network_message(exc)) from exc
        if total and done != total:
            raise InstallError("The download was cut short. Try again.")
        result = digest.hexdigest()
        if expected_sha256 and result != expected_sha256.lower():
            raise InstallError("The downloaded file does not match its published checksum "
                               "(damaged or altered in transit), so it was discarded.")
        os.replace(part, dest)
        return result
    except BaseException:
        try:
            os.remove(part)
        except OSError:
            pass
        raise


def parse_checksums(text: str, name: str) -> str:
    """The SHA-256 listed for ``name`` in ``sha256sum`` output, or ``""``."""
    for line in text.splitlines():
        match = re.match(r"^([0-9a-fA-F]{64})\s+\*?(.+?)\s*$", line.strip())
        if match and os.path.basename(match.group(2).replace("\\", "/")) == name:
            return match.group(1).lower()
    return ""


def expected_sha256(info, token: str | None = None, timeout: float = 15.0) -> str:
    """The published SHA-256 of ``info``'s installer, or
    :class:`InstallError` -- an installer without one is never run."""
    if info.sha256:
        return info.sha256
    name = info.asset_name or os.path.basename(urllib.parse.urlsplit(info.url).path)
    if info.checksums_url or info.checksums_api_url:
        url, accept = asset_source(info.checksums_url, info.checksums_api_url, token)
        try:
            with _open(url, token, accept, timeout) as response:
                text = response.read(1_000_000).decode("utf-8", "replace")
        except (urllib.error.URLError, http.client.HTTPException, OSError, TimeoutError) as exc:
            raise InstallError(_network_message(exc)) from exc
        found = parse_checksums(text, name)
        if found:
            return found
    raise InstallError(
        f"This release does not publish a checksum for {name or 'the installer'}, so openNMR "
        "will not install it automatically. Use \"Download in browser\" instead.")


def safe_filename(info) -> str:
    name = info.asset_name or os.path.basename(urllib.parse.urlsplit(info.url).path)
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name).lstrip(".")
    if not name:
        raise InstallError("The release names no installer file.")
    return name


def download_dir(platform: str | None = None, environ=None) -> str:
    """Where to put the installer. A Linux AppImage is replaced by
    renaming, which only works on the same filesystem, so it is fetched
    beside the one that is running; everything else uses a fresh
    temporary folder."""
    platform = platform or update.current_platform()
    environ = os.environ if environ is None else environ
    if platform == "linux":
        target = environ.get("APPIMAGE")
        if target and os.access(os.path.dirname(target) or ".", os.W_OK):
            return os.path.dirname(target) or "."
    return tempfile.mkdtemp(prefix="opennmr-update-")


def fetch_installer(info, dest_dir: str | None = None, *, token: str | None = None,
                    progress=None, cancelled=None, platform: str | None = None) -> str:
    """Checksum, download and verify ``info``'s installer; returns its path.

    The checksum is resolved *before* the download so a release that
    cannot be verified fails at once instead of after hundreds of MB."""
    checksum = expected_sha256(info, token)
    dest_dir = dest_dir or download_dir(platform)
    dest = os.path.join(dest_dir, safe_filename(info))
    url, accept = asset_source(info.url, info.api_url, token)
    download(url, dest, token=token, accept=accept, expected_sha256=checksum,
             progress=progress, cancelled=cancelled)
    return dest


# ============================================================
# Hand-off to the platform's installer
# ============================================================

@dataclass
class InstallPlan:
    """What to run once openNMR has quit.

    ``argv`` is started detached by :func:`launch`. ``manual`` means the
    update could not be applied automatically: nothing is run, and
    ``open_path`` (the downloaded installer) is shown to the user
    instead, with ``message`` saying why."""

    platform: str
    argv: list
    message: str = ""
    manual: bool = False
    open_path: str = ""


_LINUX_SCRIPT = """#!/bin/sh
# openNMR update helper (Linux AppImage), started by the running app,
# which exits straight afterwards. Args: app pid, new file, target, log.
PID="$1"; NEW="$2"; TARGET="$3"; LOG="$4"
exec >>"$LOG" 2>&1
echo "$(date) updating $TARGET from $NEW"
n=0
while kill -0 "$PID" 2>/dev/null; do
  n=$((n+1))
  if [ "$n" -gt 240 ]; then echo "openNMR did not exit; giving up"; exit 1; fi
  sleep 0.5
done
chmod 755 "$NEW" || exit 1
mv -f "$NEW" "$TARGET" || { echo "could not replace $TARGET"; exit 1; }
echo "replaced; starting"
exec "$TARGET"
"""

_MACOS_SCRIPT = """#!/bin/sh
# openNMR update helper (macOS), started by the running app, which exits
# straight afterwards. Args: app pid, disk image, app bundle, log.
PID="$1"; DMG="$2"; APP="$3"; LOG="$4"
exec >>"$LOG" 2>&1
echo "$(date) updating $APP from $DMG"
n=0
while kill -0 "$PID" 2>/dev/null; do
  n=$((n+1))
  if [ "$n" -gt 240 ]; then echo "openNMR did not exit; giving up"; exit 1; fi
  sleep 0.5
done
MNT="$(mktemp -d /tmp/opennmr-update.XXXXXX)"
cleanup() { hdiutil detach "$MNT" -quiet -force 2>/dev/null; rmdir "$MNT" 2>/dev/null; }
if ! hdiutil attach -nobrowse -readonly -noverify -mountpoint "$MNT" "$DMG"; then
  echo "could not open the disk image"; open "$APP"; exit 1
fi
SRC="$(ls -d "$MNT"/*.app 2>/dev/null | head -n 1)"
if [ -z "$SRC" ]; then echo "no app in the disk image"; cleanup; open "$APP"; exit 1; fi
NEW="$APP.new"; OLD="$APP.old"
rm -rf "$NEW" "$OLD"
if ditto "$SRC" "$NEW" && mv "$APP" "$OLD" && mv "$NEW" "$APP"; then
  rm -rf "$OLD"; rm -f "$DMG"; echo "updated"
else
  echo "replacing failed; keeping the old version"
  if [ -d "$OLD" ] && [ ! -d "$APP" ]; then mv "$OLD" "$APP"; fi
  rm -rf "$NEW"
fi
cleanup
xattr -dr com.apple.quarantine "$APP" 2>/dev/null
open "$APP"
"""


def app_bundle_path(exe: str) -> str | None:
    """``/Applications/openNMR.app`` from the executable inside it."""
    exe = exe.replace("\\", "/")
    index = exe.find(".app/Contents")
    return exe[:index + 4] if index >= 0 else None


def _write_script(text: str, name: str, scripts_dir: str | None) -> str:
    folder = scripts_dir or tempfile.mkdtemp(prefix="opennmr-update-")
    path = os.path.join(folder, name)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    os.chmod(path, 0o700)
    return path


def plan_install(installer: str, platform: str | None = None, *, exe: str | None = None,
                 environ=None, pid: int | None = None, scripts_dir: str | None = None,
                 log_path: str | None = None) -> InstallPlan:
    """The hand-off for a downloaded ``installer`` (arguments after
    ``installer`` are only for tests)."""
    platform = platform or update.current_platform()
    exe = exe or sys.executable
    environ = os.environ if environ is None else environ
    pid = os.getpid() if pid is None else pid
    log_path = log_path or LOG_PATH
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)

    if platform == "windows":
        roots = [environ.get(k, "") for k in ("ProgramFiles", "ProgramFiles(x86)", "ProgramW6432")]
        norm = os.path.normcase(os.path.abspath(exe))
        machine_wide = any(r and norm.startswith(os.path.normcase(os.path.abspath(r)) + os.sep)
                           for r in roots)
        argv = [installer, "/SILENT", "/SUPPRESSMSGBOXES", "/NOCANCEL", "/CLOSEAPPLICATIONS",
                "/ALLUSERS" if machine_wide else "/CURRENTUSER", "/RELAUNCH=1"]
        return InstallPlan(platform, argv)

    if platform == "darwin":
        bundle = app_bundle_path(exe)
        if not bundle:
            return InstallPlan(platform, [], manual=True, open_path=installer,
                               message="This copy is not running from an app bundle, so it cannot "
                                       "replace itself. Open the downloaded disk image and drag "
                                       "openNMR to Applications.")
        if not os.access(os.path.dirname(bundle), os.W_OK):
            return InstallPlan(platform, [], manual=True, open_path=installer,
                               message=f"openNMR cannot replace itself in {os.path.dirname(bundle)} "
                                       "(no permission). Open the downloaded disk image and drag "
                                       "openNMR to Applications.")
        script = _write_script(_MACOS_SCRIPT, "update-macos.sh", scripts_dir)
        return InstallPlan(platform, ["/bin/sh", script, str(pid), installer, bundle, log_path])

    target = environ.get("APPIMAGE")
    if not target or not os.path.isfile(target):
        return InstallPlan(platform, [], manual=True, open_path=installer,
                           message="This copy is not running from an AppImage, so it cannot replace "
                                   "itself. Run the downloaded AppImage instead "
                                   "(chmod +x, then start it).")
    if not os.access(os.path.dirname(target) or ".", os.W_OK):
        return InstallPlan(platform, [], manual=True, open_path=installer,
                           message=f"openNMR cannot replace {target} (no permission).")
    script = _write_script(_LINUX_SCRIPT, "update-linux.sh", scripts_dir)
    return InstallPlan(platform, ["/bin/sh", script, str(pid), installer, target, log_path])


def clean_env(environ=None) -> dict:
    """The environment for a helper started from the frozen app:
    PyInstaller and AppImage point library paths into the (about to
    vanish) bundle, which the next openNMR must not inherit."""
    env = dict(os.environ if environ is None else environ)
    for var in ("LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH"):
        original = env.pop(var + "_ORIG", None)
        if original is not None:
            env[var] = original
        else:
            env.pop(var, None)
    for var in ("APPIMAGE", "APPDIR", "ARGV0", "OWD", "_MEIPASS2"):
        env.pop(var, None)
    return env


def launch(plan: InstallPlan, popen=subprocess.Popen) -> None:
    """Start ``plan`` detached (it outlives this process)."""
    if plan.manual or not plan.argv:
        raise InstallError(plan.message or "This update has to be installed by hand.")
    kwargs = {}
    if sys.platform.startswith("win"):
        kwargs["creationflags"] = 0x00000008 | 0x00000200  # DETACHED_PROCESS | NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    try:
        popen(plan.argv, env=clean_env(), stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
              stderr=subprocess.DEVNULL, close_fds=True, **kwargs)
    except OSError as exc:
        raise InstallError(f"Could not start the installer ({exc}).") from exc


# ============================================================
# pip installs
# ============================================================

def pip_argv(kind: str, source: str = update.DEFAULT_SOURCE) -> list | None:
    """The command that updates a pip/git install in place, or ``None``
    for the app and for a source folder (see the module docstring).
    A PyPI install updates from PyPI; one made from a git URL from that
    repository."""
    if kind == "pip":
        return [sys.executable, "-m", "pip", "install", "--upgrade", update.PYPI_PROJECT]
    if kind != "git":
        return None
    repo = source if update._is_repo(source) else update.DEFAULT_SOURCE
    return [sys.executable, "-m", "pip", "install", "--upgrade", f"git+https://github.com/{repo}.git"]


def run_pip(argv: list, on_line=None, timeout: float = 1800.0, cancelled=None) -> tuple:
    """Run ``argv`` and return ``(ok, output)``. git is told not to ask
    for a password in a terminal (there is none); a credential manager's
    own window still can. ``on_line(text)`` sees pip's output as it comes."""
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0", PIP_DISABLE_PIP_VERSION_CHECK="1")
    kwargs = {"creationflags": 0x08000000} if sys.platform.startswith("win") else {}
    try:
        proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                env=env, **kwargs)
    except OSError as exc:
        return False, f"Could not run pip ({exc})."
    watchdog = threading.Timer(timeout, proc.kill)
    watchdog.start()
    lines = []
    try:
        for line in proc.stdout:
            lines.append(line)
            if on_line is not None:
                on_line(line.rstrip())
            if cancelled is not None and cancelled():
                proc.kill()
                lines.append("Cancelled.\n")
                break
        proc.wait()
    finally:
        watchdog.cancel()
    return proc.returncode == 0, "".join(lines)


# ============================================================
# A PyPI install: close, upgrade, reopen -- with nothing shown
# ============================================================
#
# pip cannot replace files a running program has open (Windows locks
# loaded DLLs such as Qt's and numpy's), so the app never runs pip
# itself. It writes a small helper script, starts it detached and quits.
# The helper waits until the app has really exited, runs
# ``pip install opennmr==<new>``, checks the new version imports, goes
# back to the old version if anything failed, and opens openNMR again.
# It shows nothing (no console window, no dialog): the outcome goes to
# ``~/.opennmr/update.log`` and a small result file that Settings ▸
# Updates reads, once, at the next start (:func:`take_last_result`).

#: Test and rehearsal hook: extra arguments for pip, split like a shell
#: line (``--index-url https://test.pypi.org/simple/ --pre``). Not needed
#: in normal use.
PIP_ARGS_ENV = "OPENNMR_PIP_ARGS"

RESULT_NAME = "update_result.json"
LOG_NAME = "update.log"

#: What the helper runs, after the upgrade, to see that the new version
#: starts. ``import opennmr.gui.app`` loads Qt and every panel without
#: opening a window.
DEFAULT_VERIFY = "import opennmr, opennmr.gui.app"


#: Overrides :func:`state_dir` (the test suite points it at a temporary
#: folder, so a test never reads or removes a real user's update result).
STATE_DIR_ENV = "OPENNMR_STATE_DIR"


def state_dir() -> str:
    """openNMR's per-user folder (``~/.opennmr``): settings, the update
    log and the update result."""
    return os.environ.get(STATE_DIR_ENV) or os.path.join(os.path.expanduser("~"), ".opennmr")


def upgrade_blocker(kind: str | None = None, python: str | None = None) -> str:
    """Why this copy cannot be upgraded in place from the GUI: one plain
    sentence, or ``""`` when it can. Checked *before* the button is
    offered, so the user is never asked to do something that would fail:
    only a PyPI install qualifies, in a Python that exists, that is not
    managed by the operating system (PEP 668), in a folder this user may
    write to."""
    import sysconfig

    kind = kind or update.install_kind()
    if kind != "pip":
        return "Only a copy installed with pip is updated this way."
    python = python or sys.executable
    if not python or not os.path.exists(python):
        return "This Python's program could not be found."
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    stdlib = sysconfig.get_path("stdlib") or ""
    if not in_venv and os.path.exists(os.path.join(stdlib, "EXTERNALLY-MANAGED")):
        return "This Python belongs to the operating system, which does not let pip change it."
    for key in ("purelib", "scripts"):
        folder = sysconfig.get_path(key)
        if folder and os.path.isdir(folder) and not os.access(folder, os.W_OK):
            return "You do not have permission to change the folder openNMR is installed in."
    return ""


_UPGRADE_HELPER = r'''"""openNMR update helper – started by the running app, which quits at once.

Waits for the app to exit, upgrades openNMR with pip, checks that the new
version starts, goes back to the old version if anything failed, and opens
openNMR again. It shows nothing; the outcome goes to a log and a result file.
Standard library only. Argument: the path of its JSON settings.
"""
import json
import os
import shutil
import subprocess
import sys
import time

CONFIG_PATH = sys.argv[1]
with open(CONFIG_PATH, encoding="utf-8") as handle:
    CFG = json.load(handle)
WINDOWS = sys.platform.startswith("win")
NO_WINDOW = 0x08000000 if WINDOWS else 0
ENV = dict(os.environ, PIP_DISABLE_PIP_VERSION_CHECK="1", PIP_NO_INPUT="1")


def log(text):
    try:
        with open(CFG["log"], "a", encoding="utf-8") as handle:
            handle.write(time.strftime("%Y-%m-%d %H:%M:%S ") + text + "\n")
    except OSError:
        pass


def alive(pid):
    if WINDOWS:
        import ctypes
        kernel = ctypes.windll.kernel32
        handle = kernel.OpenProcess(0x00100000, False, int(pid))  # SYNCHRONIZE
        if not handle:
            return False
        try:
            return kernel.WaitForSingleObject(handle, 0) == 0x102  # still running
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return not zombie(pid)


def zombie(pid):
    """An exited process whose parent has not collected it yet still
    answers ``kill(pid, 0)``; it is gone as far as its files go."""
    try:
        with open("/proc/%d/stat" % int(pid), "rb") as handle:
            return handle.read().rsplit(b")", 1)[1].split()[0] == b"Z"
    except (OSError, IndexError):
        pass
    try:
        out = subprocess.run(["ps", "-o", "stat=", "-p", str(int(pid))], stdout=subprocess.PIPE,
                             stderr=subprocess.DEVNULL, timeout=10).stdout
        return out.strip().startswith(b"Z")
    except (OSError, subprocess.SubprocessError):
        return False


def wait_for_exit(pid, seconds):
    end = time.time() + seconds
    while alive(pid):
        if time.time() > end:
            return False
        time.sleep(0.25)
    time.sleep(CFG["settle"])  # Windows lets go of the files a moment later
    return True


def run(args, seconds):
    try:
        done = subprocess.run(args, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, errors="replace",
                              timeout=seconds, creationflags=NO_WINDOW, env=ENV)
    except subprocess.TimeoutExpired:
        return False, "timed out after %s seconds" % seconds
    except OSError as exc:
        return False, str(exc)
    return done.returncode == 0, done.stdout or ""


def tail(text, lines=8):
    return "\n".join((text or "").strip().splitlines()[-lines:])


def install(version):
    args = [CFG["python"], "-m", "pip", "install", "--upgrade", "--quiet",
            "opennmr==" + version] + list(CFG["pip_args"])
    return run(args, CFG["pip_timeout"])


def installed_version():
    ok, out = run([CFG["python"], "-c",
                   "import importlib.metadata as m; print(m.version('opennmr'))"], 120)
    return out.strip().splitlines()[-1].strip() if ok and out.strip() else ""


def starts_ok():
    ok, out = run([CFG["python"], "-c", CFG["verify"]], 600)
    if not ok:
        log("the new version did not start: " + tail(out))
    return ok


def finish(status, detail=""):
    try:
        with open(CFG["result"], "w", encoding="utf-8") as handle:
            json.dump({"status": status, "from": CFG["from"], "to": CFG["to"],
                       "time": time.time(), "detail": detail}, handle)
    except OSError:
        pass


def relaunch():
    if not CFG.get("relaunch"):
        return
    kwargs = {"creationflags": 0x00000008 | 0x00000200} if WINDOWS else {"start_new_session": True}
    try:
        subprocess.Popen(CFG["relaunch"], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, close_fds=True, cwd=CFG.get("cwd") or None,
                         **kwargs)
        log("started openNMR again")
    except OSError as exc:
        log("could not start openNMR again: %s" % exc)


def main():
    old, new = CFG["from"], CFG["to"]
    log("updating openNMR %s -> %s; waiting for it to close" % (old, new))
    if not wait_for_exit(CFG["pid"], CFG["wait"]):
        log("openNMR did not close; nothing was changed")
        finish("cancelled")
        return
    ok, out = install(new)
    if ok:
        found = installed_version()
        if found != new:
            ok, out = False, "pip finished but openNMR %s is installed, not %s" % (found or "?", new)
        elif not starts_ok():
            ok, out = False, "the new version does not start"
    if ok:
        log("updated to " + new)
        finish("updated")
        relaunch()
        return
    log("the update failed: " + tail(out))
    if installed_version() == old and starts_ok():
        log("the old version %s is intact" % old)
    else:
        good, out2 = install(old)
        log(("went back to " + old) if good else ("could not go back to %s: %s" % (old, tail(out2))))
    finish("failed", tail(out))
    relaunch()


try:
    main()
except Exception as exc:  # the helper must never leave the user without openNMR
    log("unexpected error: %r" % (exc,))
    finish("failed", repr(exc))
    relaunch()
finally:
    shutil.rmtree(os.path.dirname(CONFIG_PATH), ignore_errors=True)
'''


def _python_pair(executable: str | None = None) -> tuple:
    """``(console, windowless)`` Python programs for ``executable``. They
    differ only on Windows (``python.exe`` / ``pythonw.exe``): the helper
    runs pip under the console one (pipes need it), and openNMR is
    reopened under the windowless one so no terminal appears."""
    executable = executable or sys.executable
    folder, name = os.path.split(executable)
    if name.lower() in ("python.exe", "pythonw.exe"):
        console, windowless = (os.path.join(folder, "python.exe"),
                               os.path.join(folder, "pythonw.exe"))
        return (console if os.path.exists(console) else executable,
                windowless if os.path.exists(windowless) else executable)
    return executable, executable


def split_pip_args(text: str, windows: bool | None = None) -> list:
    """The words of a pip-arguments string such as
    ``--no-index --find-links "C:\\my wheels"``. On Windows backslashes
    are kept and the quotes around a word are removed (``shlex`` in its
    Windows mode keeps them, and pip would then look for a folder whose
    name contains quotation marks)."""
    windows = os.name == "nt" if windows is None else windows
    if not windows:
        return shlex.split(text)
    words = shlex.split(text, posix=False)
    return [w[1:-1] if len(w) >= 2 and w[0] == w[-1] and w[0] in "\"'" else w for w in words]


def plan_pip_upgrade(version: str, *, previous: str | None = None, pid: int | None = None,
                     python: str | None = None, relaunch: list | None = None,
                     pip_args: list | None = None, verify: str | None = None,
                     scripts_dir: str | None = None, settle: float = 1.5,
                     wait: float = 600.0, pip_timeout: float = 1800.0,
                     environ=None) -> InstallPlan:
    """The detached helper that upgrades this PyPI install to
    ``version`` once the running app has quit (see the section comment).

    Everything the helper needs is written to a JSON file beside it, in a
    new temporary folder the helper removes when done -- no quoting of
    paths on a command line, whatever the user's folder names. ``previous``
    is the version to go back to (default: the running one); ``relaunch``
    the command that reopens openNMR (default: this Python, windowless,
    ``-m opennmr.gui.app``; ``[]`` for none). ``pip_args`` (and the
    :data:`PIP_ARGS_ENV` variable) add arguments to pip, which is how an
    update is rehearsed against a local folder or TestPyPI. Nothing is
    started here: :func:`launch_upgrade` does that."""
    environ = os.environ if environ is None else environ
    console, windowless = _python_pair(python)
    extra = list(pip_args or []) + split_pip_args(environ.get(PIP_ARGS_ENV, ""), windows=os.name == "nt")
    folder = tempfile.mkdtemp(prefix="opennmr-update-", dir=scripts_dir)
    state = state_dir()
    try:
        os.makedirs(state, exist_ok=True)
    except OSError:
        state = tempfile.gettempdir()  # an unwritable home must not stop the update
    config = {
        "pid": os.getpid() if pid is None else int(pid),
        "python": console, "from": previous or update.__version__, "to": version,
        "pip_args": extra, "verify": verify or DEFAULT_VERIFY,
        "relaunch": ([windowless, "-m", "opennmr.gui.app"] if relaunch is None else list(relaunch)),
        "cwd": os.path.expanduser("~"), "settle": settle, "wait": wait,
        "pip_timeout": pip_timeout,
        "log": os.path.join(state, LOG_NAME), "result": os.path.join(state, RESULT_NAME),
    }
    config_path = os.path.join(folder, "config.json")
    helper_path = os.path.join(folder, "upgrade_helper.py")
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(config, handle)
    with open(helper_path, "w", encoding="utf-8") as handle:
        handle.write(_UPGRADE_HELPER)
    # The helper itself runs windowless: a console-less (detached) python.exe
    # -- and a virtual environment's python.exe is a launcher that starts the
    # real interpreter as a child -- opens a terminal window of its own.
    # pip and the checks it runs go through ``console``, hidden.
    return InstallPlan(platform=update.current_platform(), argv=[windowless, helper_path, config_path],
                       message=f"Updating openNMR to {version}.")


#: Windows process-creation flags for the helper: no console, its own
#: process group, and (first attempt) outside the launching terminal's job,
#: which would otherwise end it together with the app.
_DETACHED = 0x00000008
_NEW_GROUP = 0x00000200
_BREAKAWAY = 0x01000000


def launch_upgrade(plan: InstallPlan, popen=subprocess.Popen):
    """Start the helper of :func:`plan_pip_upgrade` detached, so it
    outlives this process. Returns the process. Raises
    :class:`InstallError` when it cannot be started -- the caller then
    keeps the app open and shows the pip command instead."""
    if plan.manual or not plan.argv:
        raise InstallError(plan.message or "This update has to be done by hand.")
    if sys.platform.startswith("win"):
        attempts = [{"creationflags": _DETACHED | _NEW_GROUP | _BREAKAWAY},
                    {"creationflags": _DETACHED | _NEW_GROUP}]
    else:
        attempts = [{"start_new_session": True}]
    failure = None
    for kwargs in attempts:
        try:
            return popen(plan.argv, env=clean_env(), stdin=subprocess.DEVNULL,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         close_fds=True, **kwargs)
        except OSError as exc:
            failure = exc
    raise InstallError(f"Could not start the update ({failure}).") from failure


def take_last_result(folder: str | None = None) -> dict | None:
    """What the last in-app upgrade did (``status`` ``"updated"``,
    ``"failed"`` or ``"cancelled"``, ``from``, ``to``, ``detail``), read
    once: the result file is removed, so it is shown a single time.
    ``None`` when there is none."""
    path = os.path.join(folder or state_dir(), RESULT_NAME)
    try:
        with open(path, encoding="utf-8") as handle:
            result = json.load(handle)
    except (OSError, ValueError):
        return None
    try:
        os.remove(path)
    except OSError:
        pass
    return result if isinstance(result, dict) else None


def describe_result(result: dict, running_version: str | None = None) -> str:
    """One calm line for the Updates panel about a :func:`take_last_result`.
    ``""`` for a cancelled upgrade (the user closed nothing; nothing to say)."""
    running = running_version or update.__version__
    status, target = result.get("status"), result.get("to") or "the new version"
    if status == "updated":
        return f"openNMR was updated to {target}."
    if status == "failed":
        return (f"The update to {target} did not complete, and openNMR {running} is still in use. "
                f"Nothing was lost; details are in {os.path.join('~', '.opennmr', LOG_NAME)}.")
    return ""
