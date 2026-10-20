import json

from opennmr import update


# ============================================================
# Version parsing / comparison
# ============================================================

def test_parse_version_basic():
    assert update.parse_version("1.2.3") == (1, 2, 3)


def test_parse_version_tolerates_suffix():
    assert update.parse_version("1.2.3rc1") == (1, 2, 3)


def test_parse_version_missing_component_defaults_zero():
    assert update.parse_version("1.2") == (1, 2)


def test_is_newer_numeric_not_lexicographic():
    # A plain string comparison would get this wrong ("0.10.0" <
    # "0.9.0" lexicographically) -- the whole reason to parse first.
    assert update.is_newer("0.10.0", "0.9.0") is True


def test_is_newer_equal_is_not_newer():
    assert update.is_newer("1.0.0", "1.0.0") is False


def test_is_newer_older_is_not_newer():
    assert update.is_newer("1.0.0", "1.1.0") is False


# ============================================================
# current_platform
# ============================================================

def test_current_platform_returns_one_of_three(monkeypatch):
    for sys_platform, expected in [("darwin", "darwin"), ("win32", "windows"), ("linux", "linux")]:
        monkeypatch.setattr(update.sys, "platform", sys_platform)
        assert update.current_platform() == expected


# ============================================================
# check_for_update
# ============================================================

class _FakeResponse:
    def __init__(self, payload: dict):
        self._body = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self._body


def _patch_manifest(monkeypatch, payload):
    monkeypatch.setattr(update.urllib.request, "urlopen", lambda *a, **k: _FakeResponse(payload))


def test_check_for_update_finds_newer_version(monkeypatch):
    _patch_manifest(monkeypatch, {
        "version": "9.9.9", "notes": "big release",
        "url": {"darwin": "https://example.com/mac.dmg"},
    })
    result = update.check_for_update("https://example.com/manifest.json", current_version="0.1.0", platform="darwin")
    assert result == update.UpdateInfo(version="9.9.9", url="https://example.com/mac.dmg", notes="big release",
                                       asset_name="mac.dmg")


def test_check_for_update_returns_none_when_not_newer(monkeypatch):
    _patch_manifest(monkeypatch, {"version": "0.1.0", "url": {"darwin": "https://example.com/mac.dmg"}})
    result = update.check_for_update("https://example.com/manifest.json", current_version="0.1.0", platform="darwin")
    assert result is None


def test_check_for_update_returns_none_without_platform_url(monkeypatch):
    _patch_manifest(monkeypatch, {"version": "9.9.9", "url": {"windows": "https://example.com/win.zip"}})
    result = update.check_for_update("https://example.com/manifest.json", current_version="0.1.0", platform="darwin")
    assert result is None


def test_check_for_update_falls_back_to_wildcard_url(monkeypatch):
    _patch_manifest(monkeypatch, {"version": "9.9.9", "url": {"*": "https://example.com/any"}})
    result = update.check_for_update("https://example.com/manifest.json", current_version="0.1.0", platform="darwin")
    assert result.url == "https://example.com/any"


def test_check_for_update_uses_running_version_by_default(monkeypatch):
    from opennmr._version import __version__ as running_version
    _patch_manifest(monkeypatch, {"version": running_version, "url": {"darwin": "https://example.com/mac.dmg"}})
    result = update.check_for_update("https://example.com/manifest.json", platform="darwin")
    assert result is None  # manifest's version matches the running one -- not newer


# ============================================================
# GitHub Releases source, install kind, instructions (2026-10-15)
# ============================================================

import io
import urllib.error

import pytest


def _release(tag="v0.2.0", assets=None, body="What changed."):
    return {
        "tag_name": tag, "body": body,
        "html_url": f"https://github.com/o/r/releases/tag/{tag}",
        "assets": assets if assets is not None else [
            {"name": f"openNMR-{tag[1:]}-windows-setup.exe", "browser_download_url": "https://dl/win.exe"},
            {"name": f"openNMR-{tag[1:]}-macos-arm64.dmg", "browser_download_url": "https://dl/mac.dmg"},
            {"name": f"openNMR-{tag[1:]}-linux-x86_64.AppImage", "browser_download_url": "https://dl/lin.AppImage"},
        ],
    }


def _patch_api(monkeypatch, payload=None, error=None, seen=None):
    def fake_urlopen(request, timeout=None, **_kwargs):
        if seen is not None:
            seen.append(request)
        if error is not None:
            raise error
        return _FakeResponse(payload)
    monkeypatch.setattr(update.urllib.request, "urlopen", fake_urlopen)


def test_parse_version_strips_git_tag_v():
    # Without stripping, "v1.2.0" parsed as (0, 2, 0) and compared older than 0.9.0.
    assert update.parse_version("v1.2.0") == (1, 2, 0)
    assert update.is_newer("v1.0.0", "0.9.0") is True


@pytest.mark.parametrize("platform, url", [
    ("windows", "https://dl/win.exe"), ("darwin", "https://dl/mac.dmg"), ("linux", "https://dl/lin.AppImage"),
])
def test_github_release_picks_this_platforms_installer(monkeypatch, platform, url):
    _patch_api(monkeypatch, _release())
    result = update.check("o/r", current_version="0.1.0", platform=platform)
    assert result.status == "newer"
    assert result.info.version == "0.2.0"
    assert result.info.url == url
    assert result.info.notes == "What changed."


def test_github_release_prefers_installer_over_zip(monkeypatch):
    assets = [{"name": "openNMR-0.2.0-windows.zip", "browser_download_url": "https://dl/win.zip"},
              {"name": "openNMR-0.2.0-windows-setup.exe", "browser_download_url": "https://dl/win.exe"}]
    _patch_api(monkeypatch, _release(assets=assets))
    assert update.check("o/r", current_version="0.1.0", platform="windows").info.url == "https://dl/win.exe"


def test_github_release_without_platform_asset_still_reports_newer(monkeypatch):
    # A missing build must not hide an update: the release page is offered instead.
    _patch_api(monkeypatch, _release(assets=[]))
    result = update.check("o/r", current_version="0.1.0", platform="linux")
    assert result.status == "newer"
    assert result.info.url == "https://github.com/o/r/releases/tag/v0.2.0"


def test_github_release_same_version_is_current(monkeypatch):
    _patch_api(monkeypatch, _release(tag="v0.1.0"))
    result = update.check("o/r", current_version="0.1.0", platform="linux")
    assert result.status == "current"
    assert result.latest_version == "0.1.0"
    assert result.info is None


def test_private_repo_404_explains_token(monkeypatch):
    monkeypatch.delenv(update.TOKEN_ENV, raising=False)
    error = urllib.error.HTTPError("u", 404, "Not Found", {}, io.BytesIO(b""))
    _patch_api(monkeypatch, error=error)
    result = update.check("o/r", current_version="0.1.0")
    assert result.status == "error"
    assert "private" in result.message and "token" in result.message


def test_no_network_is_an_error_result_not_an_exception(monkeypatch):
    _patch_api(monkeypatch, error=urllib.error.URLError("no route"))
    result = update.check("o/r", current_version="0.1.0")
    assert result.status == "error"
    assert "no route" in result.message


def test_token_is_sent_as_bearer_header(monkeypatch):
    seen = []
    _patch_api(monkeypatch, _release(), seen=seen)
    update.check("o/r", current_version="0.1.0", token="abc123")
    assert seen[0].get_header("Authorization") == "Bearer abc123"
    assert seen[0].full_url == "https://api.github.com/repos/o/r/releases/latest"


def test_token_read_from_environment(monkeypatch):
    seen = []
    monkeypatch.setenv(update.TOKEN_ENV, "from-env")
    _patch_api(monkeypatch, _release(), seen=seen)
    update.check("o/r", current_version="0.1.0")
    assert seen[0].get_header("Authorization") == "Bearer from-env"


def test_check_with_manifest_url_still_works(monkeypatch):
    _patch_manifest(monkeypatch, {"version": "9.9.9", "url": {"linux": "https://x/l"}})
    result = update.check("https://example.com/m.json", current_version="0.1.0", platform="linux")
    assert result.status == "newer" and result.info.url == "https://x/l"


def test_check_rejects_nonsense_source():
    result = update.check("not a source", current_version="0.1.0")
    assert result.status == "error"


def test_install_kind_frozen_app(monkeypatch):
    monkeypatch.setattr(update.sys, "frozen", True, raising=False)
    assert update.install_kind() == "app"


@pytest.mark.parametrize("record, kind", [
    ({"dir_info": {"editable": True}, "url": "file:///x"}, "source"),
    ({"vcs_info": {"vcs": "git"}, "url": "https://github.com/o/r.git"}, "git"),
    ({"archive_info": {}, "url": "file:///x.whl"}, "pip"),
    (None, "pip"),
])
def test_install_kind_from_direct_url(monkeypatch, record, kind):
    class FakeDist:
        def read_text(self, name):
            return None if record is None else json.dumps(record)
    monkeypatch.delattr(update.sys, "frozen", raising=False)
    monkeypatch.setattr(update.metadata, "distribution", lambda name: FakeDist())
    assert update.install_kind() == kind


def test_update_instructions_differ_by_kind():
    assert "installer" in update.update_instructions("app", "0.2.0")
    assert "git pull" in update.update_instructions("source")
    assert "git+https://github.com/o/r.git" in update.update_instructions("git", source="o/r")
    # a PyPI install updates from PyPI, not from the repository (which may be private)
    assert "pip install --upgrade opennmr" in update.update_instructions("pip")
    assert "git+https" not in update.update_instructions("pip", source="o/r")
    assert update.pip_command("pip") == "pip install --upgrade opennmr"
    assert update.pip_command("app") is None
    assert update.pip_command("git", "o/r") == "pip install --upgrade git+https://github.com/o/r.git"


def test_startup_check_env(monkeypatch):
    monkeypatch.setenv(update.STARTUP_CHECK_ENV, "0")
    assert update.startup_check_enabled() is False
    monkeypatch.delenv(update.STARTUP_CHECK_ENV)
    assert update.startup_check_enabled() is True


def test_ssl_context_has_certificates_and_is_used(monkeypatch):
    # The frozen app on macOS has no OpenSSL certificate folder of its
    # own; certifi's bundle must be loaded into the check's context.
    context = update.ssl_context()
    assert context.cert_store_stats()["x509_ca"] > 0
    seen = []
    monkeypatch.setattr(update.urllib.request, "urlopen",
                        lambda req, timeout=None, context=None: seen.append(context) or _FakeResponse(_release()))
    update.check("o/r", current_version="0.1.0")
    assert seen and seen[0] is not None
