import json

from nmrsuite import update


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
    assert result == update.UpdateInfo(version="9.9.9", url="https://example.com/mac.dmg", notes="big release")


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
    from nmrsuite._version import __version__ as running_version
    _patch_manifest(monkeypatch, {"version": running_version, "url": {"darwin": "https://example.com/mac.dmg"}})
    result = update.check_for_update("https://example.com/manifest.json", platform="darwin")
    assert result is None  # manifest's version matches the running one -- not newer
