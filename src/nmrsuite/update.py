"""
Update checking -- the "good way to implement updates" half of the
packaging framework (see ``packaging/README.md`` for the build half).

Not wired into the running app yet (no menu item calls this, no
startup check runs it) -- this exists so the mechanism is *built*, per
an explicit request to have it ready ahead of actually needing it, not
because it's active today.

Deliberately **read-only**: this checks a version manifest and reports
whether something newer exists, but never downloads or installs
anything itself. Actually replacing a running application's files
safely is real, platform-specific work (macOS app bundles, Windows
files-in-use locking, Linux package-manager-vs-standalone conventions
all differ) -- building that without a concrete distribution channel
to test it against would be guessing at requirements that don't exist
yet. What's here is the reusable, testable part: given a manifest URL,
decide whether the user should be offered an update at all.
"""

from __future__ import annotations

import json
import sys
import urllib.request
from dataclasses import dataclass

from ._version import __version__


@dataclass
class UpdateInfo:
    """A newer version is available -- what :func:`check_for_update`
    returns when one is found."""

    version: str
    url: str
    notes: str = ""


def current_platform() -> str:
    """One of ``"darwin"``, ``"windows"``, ``"linux"`` -- the key a
    version manifest's ``"url"`` mapping is expected to use."""
    if sys.platform.startswith("darwin"):
        return "darwin"
    if sys.platform.startswith("win"):
        return "windows"
    return "linux"


def parse_version(version: str) -> tuple[int, ...]:
    """``"1.2.3"`` -> ``(1, 2, 3)``, tolerating a non-numeric suffix on
    any component (``"1.2.3rc1"`` -> ``(1, 2, 3)``) rather than raising
    -- a manifest is external, untrusted input, and a version string
    that doesn't parse cleanly should compare as "not newer" rather
    than crash the check."""
    parts = []
    for piece in version.strip().split("."):
        digits = ""
        for ch in piece:
            if ch.isdigit():
                digits += ch
            else:
                break
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def is_newer(candidate: str, current: str) -> bool:
    """Whether ``candidate`` is a newer version than ``current``,
    comparing parsed ``(major, minor, patch, ...)`` tuples (so
    ``"0.10.0" > "0.9.0"``, unlike a plain string comparison)."""
    return parse_version(candidate) > parse_version(current)


def check_for_update(manifest_url: str, current_version: str | None = None,
                      platform: str | None = None, timeout: float = 5.0) -> UpdateInfo | None:
    """Fetch a JSON version manifest from ``manifest_url`` and report
    whether a newer version is available for this platform.

    Manifest shape (hosted wherever the maintainer chooses -- a static
    web host, a GitHub raw file, etc.; see
    ``packaging/version_manifest.example.json``)::

        {
          "version": "0.2.0",
          "notes": "What changed in this release.",
          "url": {
            "darwin": "https://.../nmrsuite-0.2.0-macos.dmg",
            "windows": "https://.../nmrsuite-0.2.0-windows.zip",
            "linux": "https://.../nmrsuite-0.2.0-linux.tar.gz"
          }
        }

    Returns ``None`` if the manifest's version isn't newer than
    ``current_version`` (default: the running app's own
    :data:`nmrsuite._version.__version__`), or if it has no download
    URL for this platform (:func:`current_platform` by default).

    Raises whatever ``urllib.request.urlopen`` or ``json.loads`` raise
    on a genuine fetch/parse failure (a bad URL, no network, malformed
    JSON) -- deliberately not swallowed, so a caller can distinguish
    "checked, nothing newer" (``None``) from "the check itself failed"
    (an exception), which usually need different UI treatment.
    """
    current_version = current_version or __version__
    platform = platform or current_platform()

    with urllib.request.urlopen(manifest_url, timeout=timeout) as response:
        manifest = json.loads(response.read().decode("utf-8"))

    latest_version = manifest.get("version")
    if not latest_version or not is_newer(latest_version, current_version):
        return None

    urls = manifest.get("url", {})
    url = urls.get(platform) or urls.get("*")
    if not url:
        return None

    return UpdateInfo(version=latest_version, url=url, notes=manifest.get("notes", ""))
