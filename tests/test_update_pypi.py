"""A copy installed from PyPI (2026-10-20).

* ``update.check_pypi``: the newest release this Python can install, never
  a pre-release, a yanked file or one that needs a newer Python.
* ``update_install``: the upgrade that closes openNMR, runs pip in a hidden
  helper, goes back to the old version if anything fails, and reopens
  openNMR. The helper is really run, against throw-away wheels in a local
  folder and a real virtual environment (no network, no real openNMR).

What only a real Windows machine shows -- the helper's wait on a Windows
process handle, its DETACHED start -- is not covered here; the helper's
control flow and pip's behaviour are."""
import hashlib
import base64
import json
import os
import subprocess
import sys
import time
import urllib.error
import venv
import zipfile

import pytest

from opennmr import update, update_install
from opennmr.update_install import InstallError


# ============================================================
# check_pypi
# ============================================================

class _Response:
    def __init__(self, payload):
        self._body = payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8")

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _patch_pypi(monkeypatch, payload=None, error=None, seen=None):
    def fake_urlopen(request, timeout=None, **_kwargs):
        if seen is not None:
            seen.append(request)
        if error is not None:
            raise error
        return _Response(payload)
    monkeypatch.setattr(update.urllib.request, "urlopen", fake_urlopen)


def _file(requires=">=3.12", yanked=False):
    return {"filename": "opennmr.whl", "requires_python": requires, "yanked": yanked}


def _project(releases, latest=None, **info):
    """PyPI's JSON for a project: ``releases`` is ``{version: [files]}``."""
    return {"info": {"version": latest or sorted(releases)[-1], **info}, "releases": releases}


def test_a_newer_release_is_offered_with_its_pypi_page(monkeypatch):
    _patch_pypi(monkeypatch, _project({"0.1.0": [_file()], "0.1.1": [_file()]}))
    result = update.check_pypi("0.1.0", python_version="3.12.3")
    assert result.status == "newer"
    assert result.info.version == "0.1.1" == result.latest_version
    assert result.release_page == "https://pypi.org/project/opennmr/0.1.1/"


def test_the_same_version_is_current(monkeypatch):
    _patch_pypi(monkeypatch, _project({"0.1.0": [_file()]}))
    result = update.check_pypi("0.1.0", python_version="3.12.3")
    assert result.status == "current" and result.latest_version == "0.1.0"


def test_a_development_build_is_older_than_its_release(monkeypatch):
    # 0.1.0.dev2 (a TestPyPI rehearsal) must be offered 0.1.0; the old
    # tuple comparison read both as (0, 1, 0) and said "up to date".
    _patch_pypi(monkeypatch, _project({"0.1.0": [_file()]}))
    assert update.check_pypi("0.1.0.dev2", python_version="3.12.3").status == "newer"
    assert update.is_newer_release("0.1.0", "0.1.0.dev2") is True
    assert update.is_newer_release("0.1.1", "0.1.0") is True
    assert update.is_newer_release("0.10.0", "0.9.0") is True
    assert update.is_newer_release("0.1.0", "0.1.0") is False
    assert update.is_newer_release("not-a-version", "0.1.0") is False      # never raises


def test_prereleases_and_dev_builds_are_never_offered(monkeypatch):
    releases = {"0.1.0": [_file()], "0.2.0rc1": [_file()], "0.2.0.dev3": [_file()], "0.2.0a1": [_file()]}
    _patch_pypi(monkeypatch, _project(releases, latest="0.1.0"))
    assert update.check_pypi("0.1.0", python_version="3.12.3").status == "current"


def test_yanked_releases_are_skipped(monkeypatch):
    releases = {"0.1.0": [_file()], "0.1.1": [_file(yanked=True)],
                "0.1.2": [_file(), _file(yanked=True)]}
    assert update.pick_pypi_release(_project(releases), "3.12.3") == "0.1.2"   # one file left
    releases["0.1.2"] = [_file(yanked=True)]
    assert update.pick_pypi_release(_project(releases), "3.12.3") == "0.1.0"
    assert update.pick_pypi_release(_project({"0.1.0": []}), "3.12.3") == ""


def test_a_release_needing_a_newer_python_is_not_offered():
    releases = {"0.1.0": [_file(">=3.12")], "0.2.0": [_file(">=3.14")]}
    assert update.pick_pypi_release(_project(releases), "3.12.3") == "0.1.0"
    assert update.pick_pypi_release(_project(releases), "3.14.0") == "0.2.0"
    assert update.python_satisfies(None) and update.python_satisfies("")
    assert update.python_satisfies("garbage>>>") is True           # unreadable: pip decides
    assert update.python_satisfies(">=3.12", "3.12.0") and not update.python_satisfies(">=3.13", "3.12.9")


def test_without_the_releases_key_the_latest_release_is_used():
    # PyPI has deprecated "releases"; "info" and "urls" still describe the latest one.
    data = {"info": {"version": "0.3.0", "requires_python": ">=3.12"}, "urls": [{"filename": "a.whl"}]}
    assert update.pick_pypi_release(data, "3.12.1") == "0.3.0"
    assert update.pick_pypi_release(data, "3.11.9") == ""          # info's requires_python is used
    data["info"]["yanked"] = True
    assert update.pick_pypi_release(data, "3.12.1") == ""
    assert update.pick_pypi_release({}, "3.12.1") == ""


@pytest.mark.parametrize("error, text", [
    (urllib.error.HTTPError("u", 404, "Not Found", {}, None), "no project"),
    (urllib.error.HTTPError("u", 503, "Unavailable", {}, None), "503"),
    (urllib.error.URLError("offline"), "Could not reach PyPI"),
    (TimeoutError(), "Could not reach PyPI"),
])
def test_a_failed_check_is_a_readable_error_never_an_exception(monkeypatch, error, text):
    _patch_pypi(monkeypatch, error=error)
    result = update.check_pypi("0.1.0")
    assert result.status == "error" and text in result.message


def test_an_unreadable_answer_is_an_error(monkeypatch):
    _patch_pypi(monkeypatch, b"<html>not json</html>")
    assert "could not be read" in update.check_pypi("0.1.0").message
    _patch_pypi(monkeypatch, b"[1, 2]")
    assert update.check_pypi("0.1.0").status == "error"


def test_the_json_address_can_be_overridden_for_a_rehearsal(tmp_path, monkeypatch):
    # No network patch here: a file:// address goes through the real urlopen.
    path = tmp_path / "opennmr.json"
    path.write_text(json.dumps(_project({"0.1.0": [_file()], "0.5.0": [_file()]})), encoding="utf-8")
    monkeypatch.setenv(update.PYPI_JSON_ENV, path.as_uri())
    result = update.check_pypi("0.1.0", python_version="3.12.3")
    assert result.status == "newer" and result.info.version == "0.5.0"


def test_check_routes_the_pypi_source_to_pypi(monkeypatch):
    seen = []
    _patch_pypi(monkeypatch, _project({"0.1.0": [_file()]}), seen=seen)
    assert update.check("pypi", current_version="0.1.0").status == "current"
    assert update.check(" PyPI ", current_version="0.1.0").status == "current"
    assert seen[0].full_url == update.PYPI_JSON_URL == "https://pypi.org/pypi/opennmr/json"


def test_a_pip_install_asks_pypi_and_everything_else_asks_github():
    assert update.default_source("pip") == "pypi"
    for kind in ("app", "git", "source"):
        assert update.default_source(kind) == update.DEFAULT_SOURCE == "openNMR/openNMR"


# ============================================================
# Where an upgrade is allowed
# ============================================================

def test_only_a_pip_install_is_upgraded_in_place():
    assert "pip" in update_install.upgrade_blocker("app")
    assert update_install.upgrade_blocker("source") and update_install.upgrade_blocker("git")


def test_a_missing_python_blocks_the_upgrade(tmp_path):
    assert "could not be found" in update_install.upgrade_blocker("pip", python=str(tmp_path / "nope"))


def test_an_environment_pip_may_not_change_is_not_offered_the_button(tmp_path, monkeypatch):
    import sysconfig
    real = sysconfig.get_path
    (tmp_path / "EXTERNALLY-MANAGED").write_text("[externally-managed]", encoding="utf-8")
    monkeypatch.setattr(sysconfig, "get_path",
                        lambda key, *a, **k: str(tmp_path) if key == "stdlib" else real(key, *a, **k))
    monkeypatch.setattr(sys, "prefix", sys.base_prefix)             # not a virtual environment
    assert "operating system" in update_install.upgrade_blocker("pip")
    monkeypatch.setattr(sys, "prefix", sys.base_prefix + "-venv")  # a venv may be changed freely
    monkeypatch.setattr(os, "access", lambda *a, **k: True)
    assert update_install.upgrade_blocker("pip") == ""


def test_an_unwritable_install_folder_blocks_the_upgrade(tmp_path, monkeypatch):
    import sysconfig
    real = sysconfig.get_path
    monkeypatch.setattr(sysconfig, "get_path",
                        lambda key, *a, **k: str(tmp_path) if key in ("stdlib", "purelib", "scripts")
                        else real(key, *a, **k))
    monkeypatch.setattr(os, "access", lambda *a, **k: False)
    assert "permission" in update_install.upgrade_blocker("pip")
    monkeypatch.setattr(os, "access", lambda *a, **k: True)
    assert update_install.upgrade_blocker("pip") == ""


# ============================================================
# The plan: what is written, what is started
# ============================================================

def _plan(tmp_path, monkeypatch, **kwargs):
    monkeypatch.setenv(update_install.STATE_DIR_ENV, str(tmp_path / "state"))
    monkeypatch.delenv(update_install.PIP_ARGS_ENV, raising=False)
    return update_install.plan_pip_upgrade("9.9.9", scripts_dir=str(tmp_path), **kwargs)


def _config(plan):
    with open(plan.argv[2], encoding="utf-8") as handle:
        return json.load(handle)


def test_the_plan_writes_the_helper_and_its_settings_beside_each_other(tmp_path, monkeypatch):
    plan = _plan(tmp_path, monkeypatch)
    python, helper, config_path = plan.argv
    assert os.path.dirname(helper) == os.path.dirname(config_path)
    assert os.path.dirname(os.path.dirname(helper)) == str(tmp_path)
    compile(open(helper, encoding="utf-8").read(), helper, "exec")     # the helper is valid Python
    config = _config(plan)
    assert config["to"] == "9.9.9" and config["from"] == update.__version__
    assert config["pid"] == os.getpid()                                # it waits for *this* process
    # the helper runs windowless; pip runs through the console Python (the same file off Windows)
    assert os.path.dirname(config["python"]) == os.path.dirname(python)
    assert os.path.basename(python).lower().replace("pythonw", "python") == \
        os.path.basename(config["python"]).lower()
    assert config["relaunch"][-2:] == ["-m", "opennmr.gui.app"]
    assert config["verify"] == update_install.DEFAULT_VERIFY
    assert config["log"].endswith(update_install.LOG_NAME)
    assert os.path.dirname(config["log"]) == str(tmp_path / "state")
    assert not plan.manual and "9.9.9" in plan.message


def test_the_plan_takes_the_previous_version_and_pip_arguments(tmp_path, monkeypatch):
    monkeypatch.setenv(update_install.PIP_ARGS_ENV, "--pre --index-url https://test.pypi.org/simple/")
    plan = update_install.plan_pip_upgrade(
        "9.9.9", previous="1.2.3", pid=4242, pip_args=["--no-index"], relaunch=[], verify="import x",
        scripts_dir=str(tmp_path), settle=0.0, wait=5.0, pip_timeout=7.0)
    config = _config(plan)
    assert config["from"] == "1.2.3" and config["pid"] == 4242
    assert config["pip_args"] == ["--no-index", "--pre", "--index-url", "https://test.pypi.org/simple/"]
    assert config["relaunch"] == [] and config["verify"] == "import x"
    assert (config["settle"], config["wait"], config["pip_timeout"]) == (0.0, 5.0, 7.0)


def test_openNMR_is_reopened_without_a_console_window_on_windows(tmp_path):
    (tmp_path / "python.exe").write_text("", encoding="utf-8")
    (tmp_path / "pythonw.exe").write_text("", encoding="utf-8")
    console, windowless = update_install._python_pair(str(tmp_path / "pythonw.exe"))
    assert os.path.basename(console) == "python.exe"       # pip needs pipes: the console one
    assert os.path.basename(windowless) == "pythonw.exe"   # the GUI needs no terminal
    assert update_install._python_pair("/usr/bin/python3") == ("/usr/bin/python3", "/usr/bin/python3")
    only = tmp_path / "sub"
    only.mkdir()
    (only / "python.exe").write_text("", encoding="utf-8")
    assert update_install._python_pair(str(only / "python.exe"))[1].endswith("python.exe")


def test_the_helper_is_started_detached(tmp_path, monkeypatch):
    plan = _plan(tmp_path, monkeypatch)
    calls = []
    marker = object()
    update_install.launch_upgrade(plan, popen=lambda argv, **kw: calls.append((argv, kw)) or marker)
    argv, kwargs = calls[0]
    assert argv == plan.argv
    if sys.platform.startswith("win"):
        assert kwargs["creationflags"] & 0x00000008            # DETACHED_PROCESS
    else:
        assert kwargs["start_new_session"] is True
    assert kwargs["stdin"] == kwargs["stdout"] == kwargs["stderr"] == subprocess.DEVNULL


def test_on_windows_the_helper_first_breaks_away_from_the_launching_job(tmp_path, monkeypatch):
    plan = _plan(tmp_path, monkeypatch)
    monkeypatch.setattr(update_install.sys, "platform", "win32")
    attempts = []

    def popen(argv, **kwargs):
        attempts.append(kwargs["creationflags"])
        if kwargs["creationflags"] & 0x01000000:       # a job that forbids breakaway
            raise PermissionError("access denied")
        return "started"
    assert update_install.launch_upgrade(plan, popen=popen) == "started"
    assert attempts == [0x08 | 0x200 | 0x01000000, 0x08 | 0x200]


def test_a_helper_that_cannot_start_raises_one_clear_error(tmp_path, monkeypatch):
    plan = _plan(tmp_path, monkeypatch)

    def boom(*a, **k):
        raise OSError("no such program")
    with pytest.raises(InstallError, match="Could not start the update"):
        update_install.launch_upgrade(plan, popen=boom)
    with pytest.raises(InstallError):
        update_install.launch_upgrade(update_install.InstallPlan("linux", [], manual=True, message="by hand"))


# ============================================================
# What the next start says about it
# ============================================================

def test_the_result_is_read_once_then_gone(tmp_path):
    (tmp_path / update_install.RESULT_NAME).write_text(
        json.dumps({"status": "updated", "from": "0.1.0", "to": "0.1.1"}), encoding="utf-8")
    result = update_install.take_last_result(str(tmp_path))
    assert result["to"] == "0.1.1"
    assert update_install.take_last_result(str(tmp_path)) is None
    (tmp_path / update_install.RESULT_NAME).write_text("not json", encoding="utf-8")
    assert update_install.take_last_result(str(tmp_path)) is None
    (tmp_path / update_install.RESULT_NAME).write_text("[1]", encoding="utf-8")
    assert update_install.take_last_result(str(tmp_path)) is None


def test_the_message_after_an_upgrade_is_calm():
    assert update_install.describe_result({"status": "updated", "to": "0.1.1"}) == \
        "openNMR was updated to 0.1.1."
    failed = update_install.describe_result({"status": "failed", "to": "0.1.1"}, running_version="0.1.0")
    assert "did not complete" in failed and "0.1.0 is still in use" in failed and "Nothing was lost" in failed
    assert "error" not in failed.lower() and "traceback" not in failed.lower()
    assert update_install.describe_result({"status": "cancelled", "to": "0.1.1"}) == ""


# ============================================================
# The helper, really run: a virtual environment and local wheels
# ============================================================

def _wheel(folder, version, body=""):
    """A minimal ``opennmr`` wheel at ``version`` whose ``__init__`` runs ``body``."""
    dist = f"opennmr-{version}.dist-info"
    files = {
        "opennmr/__init__.py": f'__version__ = "{version}"\n{body}\n',
        f"{dist}/METADATA": f"Metadata-Version: 2.1\nName: opennmr\nVersion: {version}\n",
        f"{dist}/WHEEL": "Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
    }
    record = []
    for name, text in files.items():
        data = text.encode("utf-8")
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
        record.append(f"{name},sha256={digest},{len(data)}")
    record.append(f"{dist}/RECORD,,")
    files[f"{dist}/RECORD"] = "\n".join(record) + "\n"
    with zipfile.ZipFile(os.path.join(folder, f"opennmr-{version}-py3-none-any.whl"), "w") as archive:
        for name, text in files.items():
            archive.writestr(name, text)


class _Lab:
    """A throw-away environment: pip, local wheels, no network."""

    def __init__(self, root):
        self.wheels = os.path.join(str(root), "wheels")
        os.makedirs(self.wheels)
        _wheel(self.wheels, "0.1.0")
        _wheel(self.wheels, "0.1.1")
        _wheel(self.wheels, "0.1.2", body='raise RuntimeError("this release is broken")')
        env_dir = os.path.join(str(root), "env")
        venv.create(env_dir, with_pip=True)
        scripts = "Scripts" if sys.platform.startswith("win") else "bin"
        exe = "python.exe" if sys.platform.startswith("win") else "python"
        self.python = os.path.join(env_dir, scripts, exe)
        self.reset()

    def pip(self, *args):
        return subprocess.run([self.python, "-m", "pip", "--disable-pip-version-check", *args],
                              capture_output=True, text=True, timeout=300)

    def reset(self):
        """Put 0.1.0 back, whatever a test did to the environment."""
        done = self.pip("install", "--force-reinstall", "--no-deps", "--no-index",
                        "--find-links", self.wheels, "opennmr==0.1.0")
        assert done.returncode == 0, done.stdout + done.stderr
        assert self.version() == "0.1.0"

    def version(self):
        done = subprocess.run([self.python, "-c", "import importlib.metadata as m; print(m.version('opennmr'))"],
                              capture_output=True, text=True, timeout=120)
        return done.stdout.strip()

    def package_mtime(self):
        code = "import os, importlib.util as u; print(os.path.getmtime(u.find_spec('opennmr').origin))"
        done = subprocess.run([self.python, "-c", code], capture_output=True, text=True, timeout=120)
        return float(done.stdout.strip())


@pytest.fixture(scope="module")
def lab(tmp_path_factory):
    return _Lab(tmp_path_factory.mktemp("lab"))


@pytest.fixture
def fresh(lab, tmp_path, monkeypatch):
    lab.reset()
    monkeypatch.setenv(update_install.STATE_DIR_ENV, str(tmp_path / "state"))
    monkeypatch.delenv(update_install.PIP_ARGS_ENV, raising=False)
    return lab


def _finished_pid():
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


def _run_helper(lab, tmp_path, version, pid=None, wait=60.0, relaunch=True):
    marker = tmp_path / "reopened.txt"
    command = [lab.python, "-c", f"open({str(marker)!r}, 'w').write('reopened')"]
    plan = update_install.plan_pip_upgrade(
        version, previous="0.1.0", pid=_finished_pid() if pid is None else pid, python=lab.python,
        relaunch=command if relaunch else [], verify="import opennmr",
        pip_args=["--no-index", "--find-links", lab.wheels], scripts_dir=str(tmp_path),
        settle=0.0, wait=wait, pip_timeout=240.0)
    done = subprocess.run(plan.argv, capture_output=True, text=True, timeout=600)
    assert done.stdout == "" and done.stderr == ""          # nothing is ever printed
    deadline = time.time() + 15
    while relaunch and not marker.exists() and time.time() < deadline:
        time.sleep(0.1)                                      # the reopened app starts detached
    return plan, marker


def _state(tmp_path):
    folder = tmp_path / "state"
    result = json.loads((folder / update_install.RESULT_NAME).read_text(encoding="utf-8"))
    log = (folder / update_install.LOG_NAME).read_text(encoding="utf-8")
    return result, log


def test_the_helper_upgrades_reopens_and_cleans_up(fresh, tmp_path):
    plan, marker = _run_helper(fresh, tmp_path, "0.1.1")
    assert fresh.version() == "0.1.1"
    assert marker.read_text(encoding="utf-8") == "reopened"
    result, log = _state(tmp_path)
    assert result["status"] == "updated" and result["to"] == "0.1.1" and result["from"] == "0.1.0"
    assert "updated to 0.1.1" in log and "started openNMR again" in log
    assert not os.path.exists(os.path.dirname(plan.argv[1]))     # the helper removed its own folder


def test_the_helper_waits_for_the_app_to_exit_before_touching_files(fresh, tmp_path):
    done_file = tmp_path / "app_exited.txt"
    app = subprocess.Popen([sys.executable, "-c",
                            "import time; time.sleep(2.5); "
                            f"open({str(done_file)!r}, 'w').write(str(time.time()))"])
    try:
        _run_helper(fresh, tmp_path, "0.1.1", pid=app.pid)
    finally:
        app.wait(timeout=30)
    exited_at = float(done_file.read_text(encoding="utf-8"))
    assert fresh.version() == "0.1.1"
    assert fresh.package_mtime() >= exited_at - 0.05          # files changed only after the app was gone


def test_a_release_that_does_not_start_is_rolled_back(fresh, tmp_path):
    _, marker = _run_helper(fresh, tmp_path, "0.1.2")
    assert fresh.version() == "0.1.0"                         # back where it was, and it imports
    assert marker.exists()                                    # openNMR is reopened regardless
    result, log = _state(tmp_path)
    assert result["status"] == "failed" and result["to"] == "0.1.2"
    assert "did not start" in log and "went back to 0.1.0" in log


def test_a_version_pip_cannot_find_leaves_the_old_one_untouched(fresh, tmp_path):
    _, marker = _run_helper(fresh, tmp_path, "9.9.9")
    assert fresh.version() == "0.1.0"
    assert marker.exists()
    result, log = _state(tmp_path)
    assert result["status"] == "failed"
    assert "the update failed" in log and "old version 0.1.0 is intact" in log


def test_an_app_that_never_exits_is_left_alone(fresh, tmp_path):
    app = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        _, marker = _run_helper(fresh, tmp_path, "0.1.1", pid=app.pid, wait=1.0)
    finally:
        app.kill()
        app.wait(timeout=30)
    assert fresh.version() == "0.1.0"                         # nothing changed
    assert not marker.exists()                                # and it is not "reopened" over a running copy
    result, log = _state(tmp_path)
    assert result["status"] == "cancelled" and "did not close" in log
    assert update_install.describe_result(result) == ""       # and the panel stays silent about it


def test_pip_arguments_lose_their_quotes_on_windows():
    text = '--no-index --find-links "C:\\Users\\a b\\wheels" --pre'
    assert update_install.split_pip_args(text, windows=True) == [
        "--no-index", "--find-links", "C:\\Users\\a b\\wheels", "--pre"]
    assert update_install.split_pip_args("--find-links /tmp/a\\ b", windows=False) == [
        "--find-links", "/tmp/a b"]
    assert update_install.split_pip_args("", windows=True) == []
    assert update_install.split_pip_args("--index-url 'https://x/y'", windows=True)[-1] == "https://x/y"


def test_the_helper_itself_runs_windowless_and_pip_through_the_console_python(tmp_path, monkeypatch):
    exe = tmp_path / "python.exe"
    exe.write_text("")
    (tmp_path / "pythonw.exe").write_text("")
    plan = update_install.plan_pip_upgrade("9.9.9", python=str(exe), scripts_dir=str(tmp_path),
                                           environ={}, pid=1)
    assert os.path.basename(plan.argv[0]) == "pythonw.exe"     # no terminal window opens
    config = json.loads(open(plan.argv[2], encoding="utf-8").read())
    assert os.path.basename(config["python"]) == "python.exe"   # pip runs here, hidden
