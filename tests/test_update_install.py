"""opennmr.update_install: download + verify + hand over to the platform's
installer, with no terminal (Settings ▸ Updates ▸ Install; 2026-10-20).

No real network: downloads run against throw-away servers on 127.0.0.1.
The Linux and macOS helper *scripts* are really run (macOS with stand-in
hdiutil/ditto/open commands), so their control flow is tested; what only a
real Windows/macOS machine can show -- the Inno Setup installer, a real
disk image -- is not, and the packaging README says so."""
import hashlib
import http.server
import os
import subprocess
import sys
import threading

import pytest

from opennmr import update, update_install
from opennmr.update_install import InstallError

posix_only = pytest.mark.skipif(sys.platform.startswith("win"), reason="runs /bin/sh helpers")

# larger than two download chunks, so progress is reported several times
PAYLOAD = b"openNMR installer bytes " * 30000
PAYLOAD_SHA = hashlib.sha256(PAYLOAD).hexdigest()


# ---------------------------------------------------------------- a local server

class _Server:
    def __init__(self, routes):
        self.hits = []          # (path, headers dict) per request
        self.routes = routes
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                outer.hits.append((self.path, {k: v for k, v in self.headers.items()}))
                route = outer.routes.get(self.path)
                if route is None:
                    self.send_error(404)
                    return
                kind, value = route
                if kind == "redirect":
                    self.send_response(302)
                    self.send_header("Location", value)
                    self.end_headers()
                elif kind == "short":      # promises more than it sends
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(value) + 100))
                    self.end_headers()
                    self.wfile.write(value)
                else:
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(value)))
                    self.end_headers()
                    self.wfile.write(value)

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def url(self, path="/"):
        return f"http://127.0.0.1:{self.port}{path}"

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def serve():
    servers = []

    def make(routes):
        server = _Server(routes)
        servers.append(server)
        return server
    yield make
    for server in servers:
        server.close()


def _info(url, sha="", **kw):
    return update.UpdateInfo(version="9.9.9", url=url, asset_name="openNMR-9.9.9-linux-x86_64.AppImage",
                             sha256=sha, **kw)


# ---------------------------------------------------------------- checksums in the check

def test_clean_sha256_accepts_digest_forms_and_rejects_the_rest():
    assert update.clean_sha256("sha256:" + PAYLOAD_SHA.upper()) == PAYLOAD_SHA
    assert update.clean_sha256(PAYLOAD_SHA) == PAYLOAD_SHA
    for bad in ("", None, "sha1:abc", "abc", "z" * 64, PAYLOAD_SHA[:-1]):
        assert update.clean_sha256(bad) == ""


def test_github_release_carries_digest_size_and_api_address(monkeypatch):
    release = {"tag_name": "v9.9.9", "html_url": "https://github.com/o/r/releases/tag/v9.9.9",
               "assets": [
                   {"name": "openNMR-9.9.9-linux-x86_64.AppImage", "size": 123,
                    "url": "https://api.github.com/repos/o/r/releases/assets/1",
                    "browser_download_url": "https://github.com/o/r/releases/download/v9.9.9/a.AppImage",
                    "digest": "sha256:" + PAYLOAD_SHA},
                   {"name": "SHA256SUMS", "url": "https://api.github.com/repos/o/r/releases/assets/2",
                    "browser_download_url": "https://github.com/o/r/releases/download/v9.9.9/SHA256SUMS"}]}
    monkeypatch.setattr(update, "_github_request", lambda *a, **k: release)
    result = update.check("o/r", current_version="0.1.0", platform="linux")
    info = result.info
    assert (info.sha256, info.size) == (PAYLOAD_SHA, 123)
    assert info.api_url.endswith("/assets/1")
    assert info.checksums_url.endswith("/SHA256SUMS") and info.checksums_api_url.endswith("/assets/2")
    assert info.asset_name == "openNMR-9.9.9-linux-x86_64.AppImage"


def test_a_release_without_digest_has_no_sha256(monkeypatch):
    release = {"tag_name": "v9.9.9", "assets": [
        {"name": "openNMR-9.9.9-linux-x86_64.AppImage", "browser_download_url": "https://x/a"}]}
    monkeypatch.setattr(update, "_github_request", lambda *a, **k: release)
    info = update.check("o/r", current_version="0.1.0", platform="linux").info
    assert info.sha256 == "" and info.checksums_url == ""


def test_manifest_sha256_per_platform(monkeypatch):
    import io
    import json

    manifest = {"version": "9.9.9", "url": {"linux": "https://x/a.AppImage"},
                "sha256": {"linux": PAYLOAD_SHA, "windows": "0" * 64}}

    class _R(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False
    monkeypatch.setattr(update, "_urlopen", lambda *a, **k: _R(json.dumps(manifest).encode()))
    info = update.check_for_update("https://m/manifest.json", "0.1.0", "linux")
    assert info.sha256 == PAYLOAD_SHA and info.asset_name == "a.AppImage"


def test_parse_checksums_reads_sha256sum_output():
    text = (f"{PAYLOAD_SHA}  openNMR-9.9.9-linux-x86_64.AppImage\n"
            f"{'a' * 64} *other.dmg\nnot a checksum line\n")
    assert update_install.parse_checksums(text, "openNMR-9.9.9-linux-x86_64.AppImage") == PAYLOAD_SHA
    assert update_install.parse_checksums(text, "other.dmg") == "a" * 64
    assert update_install.parse_checksums(text, "missing.exe") == ""


# ---------------------------------------------------------------- downloading

def test_download_streams_verifies_and_renames(serve, tmp_path):
    server = serve({"/a": ("ok", PAYLOAD)})
    seen = []
    dest = str(tmp_path / "a.bin")
    digest = update_install.download(server.url("/a"), dest, expected_sha256=PAYLOAD_SHA,
                                     progress=lambda done, total: seen.append((done, total)))
    assert digest == PAYLOAD_SHA
    assert open(dest, "rb").read() == PAYLOAD
    assert not os.path.exists(dest + ".part")
    assert seen[-1] == (len(PAYLOAD), len(PAYLOAD)) and len(seen) > 1


def test_a_wrong_checksum_discards_the_file(serve, tmp_path):
    server = serve({"/a": ("ok", PAYLOAD)})
    dest = str(tmp_path / "a.bin")
    with pytest.raises(InstallError, match="checksum"):
        update_install.download(server.url("/a"), dest, expected_sha256="0" * 64)
    assert os.listdir(tmp_path) == []


def test_a_truncated_download_is_refused(serve, tmp_path):
    server = serve({"/a": ("short", PAYLOAD)})
    dest = str(tmp_path / "a.bin")
    with pytest.raises(InstallError, match="cut short"):
        update_install.download(server.url("/a"), dest)
    assert os.listdir(tmp_path) == []


def test_cancelling_stops_the_download_and_cleans_up(serve, tmp_path):
    server = serve({"/a": ("ok", PAYLOAD)})
    calls = []

    def cancelled():
        calls.append(1)
        return len(calls) > 1
    with pytest.raises(InstallError, match="cancelled"):
        update_install.download(server.url("/a"), str(tmp_path / "a.bin"), cancelled=cancelled)
    assert os.listdir(tmp_path) == []


def test_a_missing_file_is_a_readable_error(serve, tmp_path):
    server = serve({})
    with pytest.raises(InstallError, match="404"):
        update_install.download(server.url("/nothing"), str(tmp_path / "a.bin"))


def test_plain_http_is_refused_except_to_this_machine(tmp_path):
    with pytest.raises(InstallError, match="https"):
        update_install.download("http://example.com/a.exe", str(tmp_path / "a.exe"))
    with pytest.raises(InstallError, match="https"):
        update_install.download("ftp://example.com/a.exe", str(tmp_path / "a.exe"))


def test_the_token_is_sent_to_the_api_host_only_and_dropped_on_redirect(serve, tmp_path, monkeypatch):
    target = serve({"/file": ("ok", PAYLOAD)})
    api = serve({"/asset": ("redirect", target.url("/file"))})
    monkeypatch.setattr(update_install, "TOKEN_HOSTS", {"127.0.0.1"})
    update_install.download(api.url("/asset"), str(tmp_path / "a.bin"), token="s3cret",
                            accept="application/octet-stream")
    assert api.hits[0][1].get("Authorization") == "Bearer s3cret"
    assert api.hits[0][1].get("Accept") == "application/octet-stream"
    assert "Authorization" not in target.hits[0][1]      # the signed storage URL never sees it
    assert open(tmp_path / "a.bin", "rb").read() == PAYLOAD


def test_the_token_is_not_sent_to_other_hosts(serve, tmp_path):
    server = serve({"/a": ("ok", PAYLOAD)})            # 127.0.0.1 is not in TOKEN_HOSTS
    update_install.download(server.url("/a"), str(tmp_path / "a.bin"), token="s3cret")
    assert "Authorization" not in server.hits[0][1]


def test_asset_source_uses_the_api_only_with_a_token():
    assert update_install.asset_source("https://b", "https://api", None) == ("https://b", None)
    assert update_install.asset_source("https://b", "https://api", "t") == (
        "https://api", "application/octet-stream")
    assert update_install.asset_source("https://b", "", "t") == ("https://b", None)


# ---------------------------------------------------------------- the checksum rule

def test_expected_sha256_prefers_the_releases_own_digest():
    assert update_install.expected_sha256(_info("https://x/a", sha=PAYLOAD_SHA)) == PAYLOAD_SHA


def test_expected_sha256_reads_a_sums_file(serve):
    server = serve({"/SHA256SUMS": ("ok", f"{PAYLOAD_SHA}  openNMR-9.9.9-linux-x86_64.AppImage\n".encode())})
    info = _info("https://x/a", checksums_url=server.url("/SHA256SUMS"))
    assert update_install.expected_sha256(info) == PAYLOAD_SHA


def test_no_published_checksum_means_no_automatic_install(serve, tmp_path):
    server = serve({"/a": ("ok", PAYLOAD), "/SHA256SUMS": ("ok", b"deadbeef  other\n")})
    for info in (_info(server.url("/a")),
                 _info(server.url("/a"), checksums_url=server.url("/SHA256SUMS"))):
        with pytest.raises(InstallError, match="checksum"):
            update_install.fetch_installer(info, str(tmp_path))
    assert [p for p, _ in server.hits if p == "/a"] == []     # nothing was downloaded
    assert os.listdir(tmp_path) == []


def test_fetch_installer_end_to_end(serve, tmp_path):
    server = serve({"/a": ("ok", PAYLOAD)})
    path = update_install.fetch_installer(_info(server.url("/a"), sha=PAYLOAD_SHA), str(tmp_path))
    assert os.path.basename(path) == "openNMR-9.9.9-linux-x86_64.AppImage"
    assert update_install.sha256_of(path) == PAYLOAD_SHA


def test_a_tampered_download_is_not_kept(serve, tmp_path):
    server = serve({"/a": ("ok", PAYLOAD + b"x")})
    with pytest.raises(InstallError, match="checksum"):
        update_install.fetch_installer(_info(server.url("/a"), sha=PAYLOAD_SHA), str(tmp_path))
    assert os.listdir(tmp_path) == []


def test_safe_filename_cannot_escape_the_folder():
    info = update.UpdateInfo(version="1", url="https://x/y", asset_name="../../evil name.exe")
    name = update_install.safe_filename(info)
    assert "/" not in name and "\\" not in name and not name.startswith(".")
    with pytest.raises(InstallError):
        update_install.safe_filename(update.UpdateInfo(version="1", url="https://x/"))


# ---------------------------------------------------------------- the hand-off

def test_windows_plan_is_per_user_unless_installed_for_everyone(tmp_path):
    env = {"ProgramFiles": r"C:\Program Files"}
    user = update_install.plan_install(
        "setup.exe", "windows", exe=r"C:\Users\a\AppData\Local\Programs\openNMR\openNMR.exe",
        environ=env, log_path=str(tmp_path / "u.log"))
    assert "/CURRENTUSER" in user.argv and "/ALLUSERS" not in user.argv
    for flag in ("/SILENT", "/RELAUNCH=1", "/CLOSEAPPLICATIONS"):
        assert flag in user.argv
    assert user.argv[0] == "setup.exe" and not user.manual
    if os.sep == "\\":     # path prefix tests need Windows' path rules
        wide = update_install.plan_install(
            "setup.exe", "windows", exe=r"C:\Program Files\openNMR\openNMR.exe",
            environ=env, log_path=str(tmp_path / "u.log"))
        assert "/ALLUSERS" in wide.argv


_ISS = os.path.join(os.path.dirname(__file__), "..", "packaging", "windows", "openNMR.iss")


@pytest.mark.skipif(not os.path.exists(_ISS), reason="the installer packaging is not in this checkout")
def test_the_inno_script_relaunches_only_when_asked():
    iss = open(_ISS, encoding="utf-8").read()
    assert "RelaunchRequested" in iss and "{param:RELAUNCH|0}" in iss
    assert "Check: RelaunchRequested" in iss
    assert "postinstall skipifsilent" in iss          # CI's silent install must still not open a window


@pytest.mark.skipif(not os.path.exists(_ISS), reason="the installer packaging is not in this checkout")
def test_the_release_job_publishes_checksums():
    wf = open(os.path.join(os.path.dirname(__file__), "..", ".github", "workflows", "build.yml"),
              encoding="utf-8").read()
    assert "sha256sum" in wf and "SHA256SUMS" in wf


def _finished_pid():
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


@posix_only
def test_linux_helper_replaces_the_appimage_and_starts_it(tmp_path):
    target = tmp_path / "openNMR.AppImage"
    target.write_text("#!/bin/sh\necho old\n")
    new = tmp_path / "openNMR-9.9.9-linux-x86_64.AppImage"
    marker = tmp_path / "started"
    new.write_text(f"#!/bin/sh\necho started > '{marker}'\n")
    log = tmp_path / "update.log"
    plan = update_install.plan_install(str(new), "linux", environ={"APPIMAGE": str(target)},
                                       pid=_finished_pid(), scripts_dir=str(tmp_path),
                                       log_path=str(log))
    assert not plan.manual
    subprocess.run(plan.argv, check=True, timeout=30)
    assert marker.read_text().strip() == "started"          # the NEW file now lives at the target
    assert not new.exists()
    assert os.access(target, os.X_OK)
    assert "replaced" in log.read_text()


@posix_only
def test_linux_helper_waits_for_the_app_to_exit(tmp_path):
    target = tmp_path / "openNMR.AppImage"
    target.write_text("#!/bin/sh\nexit 0\n")
    new = tmp_path / "new.AppImage"
    new.write_text("#!/bin/sh\nexit 0\n")
    sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        plan = update_install.plan_install(str(new), "linux", environ={"APPIMAGE": str(target)},
                                           pid=sleeper.pid, scripts_dir=str(tmp_path),
                                           log_path=str(tmp_path / "l.log"))
        helper = subprocess.Popen(plan.argv)
        try:
            helper.wait(timeout=2)
            assert False, "the helper must not replace the file while the app is still running"
        except subprocess.TimeoutExpired:
            pass
        assert new.exists()
        sleeper.kill()
        sleeper.wait()
        assert helper.wait(timeout=30) == 0
        assert not new.exists()
    finally:
        sleeper.kill()
        helper.kill()


def test_linux_without_an_appimage_is_manual(tmp_path):
    plan = update_install.plan_install(str(tmp_path / "x.AppImage"), "linux", environ={},
                                       log_path=str(tmp_path / "l.log"))
    assert plan.manual and plan.open_path.endswith("x.AppImage") and plan.message
    with pytest.raises(InstallError):
        update_install.launch(plan)


@posix_only
@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root can write anywhere")
def test_linux_in_a_read_only_folder_is_manual(tmp_path):
    folder = tmp_path / "ro"
    folder.mkdir()
    target = folder / "openNMR.AppImage"
    target.write_text("x")
    folder.chmod(0o500)
    try:
        plan = update_install.plan_install(str(tmp_path / "n"), "linux", environ={"APPIMAGE": str(target)},
                                           log_path=str(tmp_path / "l.log"))
        assert plan.manual and "permission" in plan.message
    finally:
        folder.chmod(0o700)


def test_app_bundle_path():
    exe = "/Applications/openNMR.app/Contents/MacOS/openNMR"
    assert update_install.app_bundle_path(exe) == "/Applications/openNMR.app"
    assert update_install.app_bundle_path("/usr/bin/python3") is None


def _fake_macos(tmp_path, ditto_ok=True):
    """A bundle holding OLD, stand-in system commands, and a 'disk image'."""
    bundle = tmp_path / "Apps" / "openNMR.app"
    (bundle / "Contents" / "MacOS").mkdir(parents=True)
    (bundle / "Contents" / "MacOS" / "openNMR").write_text("OLD")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    opened = tmp_path / "opened"
    scripts = {
        "hdiutil": '#!/bin/sh\nif [ "$1" = attach ]; then\n'
                   '  while [ $# -gt 0 ]; do [ "$1" = -mountpoint ] && MP="$2"; shift; done\n'
                   '  mkdir -p "$MP/openNMR.app/Contents/MacOS"; echo NEW > "$MP/openNMR.app/Contents/MacOS/openNMR"\n'
                   'elif [ "$1" = detach ]; then rm -rf "$2"/*; fi\nexit 0\n',
        "ditto": '#!/bin/sh\n' + ('cp -R "$1" "$2"\n' if ditto_ok else 'exit 1\n'),
        "open": f'#!/bin/sh\necho "$@" >> "{opened}"\n',
        "xattr": "#!/bin/sh\nexit 0\n",
    }
    for name, text in scripts.items():
        path = bin_dir / name
        path.write_text(text)
        path.chmod(0o755)
    dmg = tmp_path / "openNMR-9.9.9-macos-arm64.dmg"
    dmg.write_text("not really a disk image")
    return bundle, bin_dir, opened, dmg


@posix_only
def test_macos_helper_swaps_the_bundle_and_opens_it(tmp_path):
    bundle, bin_dir, opened, dmg = _fake_macos(tmp_path)
    exe = str(bundle / "Contents" / "MacOS" / "openNMR")
    plan = update_install.plan_install(str(dmg), "darwin", exe=exe, pid=_finished_pid(),
                                       scripts_dir=str(tmp_path), log_path=str(tmp_path / "l.log"))
    assert not plan.manual and plan.argv[-2] == str(bundle)
    env = dict(os.environ, PATH=f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    subprocess.run(plan.argv, check=True, env=env, timeout=30)
    assert (bundle / "Contents" / "MacOS" / "openNMR").read_text().strip() == "NEW"
    assert not os.path.exists(str(bundle) + ".old") and not os.path.exists(str(bundle) + ".new")
    assert not dmg.exists()
    assert str(bundle) in opened.read_text()


@posix_only
def test_macos_helper_keeps_the_old_version_if_copying_fails(tmp_path):
    bundle, bin_dir, opened, dmg = _fake_macos(tmp_path, ditto_ok=False)
    exe = str(bundle / "Contents" / "MacOS" / "openNMR")
    plan = update_install.plan_install(str(dmg), "darwin", exe=exe, pid=_finished_pid(),
                                       scripts_dir=str(tmp_path), log_path=str(tmp_path / "l.log"))
    env = dict(os.environ, PATH=f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    subprocess.run(plan.argv, check=True, env=env, timeout=30)
    assert (bundle / "Contents" / "MacOS" / "openNMR").read_text() == "OLD"
    assert not os.path.exists(str(bundle) + ".new")
    assert dmg.exists()                                       # kept, so the user can still install it
    assert str(bundle) in opened.read_text()                  # the old app is reopened


def test_macos_outside_a_bundle_is_manual(tmp_path):
    plan = update_install.plan_install(str(tmp_path / "x.dmg"), "darwin", exe="/usr/bin/python3",
                                       log_path=str(tmp_path / "l.log"))
    assert plan.manual and plan.open_path.endswith("x.dmg")


def test_clean_env_removes_the_bundles_library_paths():
    env = update_install.clean_env({"PATH": "/bin", "LD_LIBRARY_PATH": "/tmp/_MEI1",
                                   "LD_LIBRARY_PATH_ORIG": "/usr/lib", "DYLD_LIBRARY_PATH": "/tmp/_MEI1",
                                   "APPIMAGE": "/x.AppImage", "APPDIR": "/tmp/.mount_x"})
    assert env["LD_LIBRARY_PATH"] == "/usr/lib"
    assert "DYLD_LIBRARY_PATH" not in env and "APPIMAGE" not in env and "APPDIR" not in env
    assert env["PATH"] == "/bin"


def test_launch_starts_the_helper_detached_with_a_clean_environment(monkeypatch):
    calls = []
    monkeypatch.setenv("APPIMAGE", "/x.AppImage")
    plan = update_install.InstallPlan("linux", ["/bin/sh", "helper.sh"])
    update_install.launch(plan, popen=lambda argv, **kw: calls.append((argv, kw)))
    argv, kw = calls[0]
    assert argv == ["/bin/sh", "helper.sh"]
    assert "APPIMAGE" not in kw["env"]
    assert kw.get("start_new_session") or kw.get("creationflags")


def test_launch_reports_a_helper_that_cannot_start():
    def boom(*a, **k):
        raise OSError("no such file")
    with pytest.raises(InstallError, match="Could not start"):
        update_install.launch(update_install.InstallPlan("linux", ["/nope"]), popen=boom)


# ---------------------------------------------------------------- pip installs

def test_pip_argv_only_for_pip_style_installs():
    argv = update_install.pip_argv("git", "o/r")
    assert argv[1:5] == ["-m", "pip", "install", "--upgrade"]
    assert argv[-1] == "git+https://github.com/o/r.git"
    assert update_install.pip_argv("pip", "o/r")[-1] == "opennmr"   # a PyPI install updates from PyPI
    assert update_install.pip_argv("app") is None
    assert update_install.pip_argv("source") is None          # never `git pull` a developer's checkout


def test_run_pip_reports_success_failure_and_streams_lines():
    lines = []
    ok, out = update_install.run_pip([sys.executable, "-c", "print('hello'); print('world')"],
                                     on_line=lines.append)
    assert ok and lines == ["hello", "world"] and "hello" in out
    ok, out = update_install.run_pip([sys.executable, "-c", "import sys; print('bad'); sys.exit(3)"])
    assert not ok and "bad" in out
    ok, out = update_install.run_pip(["/definitely/not/a/program"])
    assert not ok and "Could not run pip" in out


def test_run_pip_can_be_cancelled():
    code = "import time\nfor i in range(1000):\n    print(i, flush=True); time.sleep(0.05)"
    ok, out = update_install.run_pip([sys.executable, "-c", code], cancelled=lambda: True, timeout=30)
    assert not ok and "Cancelled" in out
