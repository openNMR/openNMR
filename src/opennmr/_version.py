"""Single source of truth for the running app's version.

``pyproject.toml``'s ``[project] version`` is the actual source of
truth (what gets published to a wheel's metadata); this just exposes
it at runtime without hand-duplicating the string, which would risk
the two drifting apart. Falls back to a hardcoded value only for the
rare case metadata isn't resolvable at all (e.g. running straight from
a source checkout with no package installed, not even ``pip install
-e .``) -- keep this fallback in sync with ``pyproject.toml`` if it's
ever bumped.
"""

from __future__ import annotations

from importlib import metadata

_FALLBACK_VERSION = "0.9.10"

try:
    __version__ = metadata.version("opennmr")
except metadata.PackageNotFoundError:
    __version__ = _FALLBACK_VERSION
