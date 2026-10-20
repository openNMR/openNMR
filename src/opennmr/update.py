"""
Update checking -- Settings ▸ Updates in the GUI, and usable from a
notebook (``opennmr.update.check()``).

This module only **checks**: it finds out whether a newer version
exists and what to fetch, and never touches the network for anything
else or changes any file. Downloading, verifying and installing is
:mod:`opennmr.update_install`, which the GUI's "Install" button uses
(Settings ▸ Updates) -- replacing a running application's files is
platform-specific, so it lives apart from the check.

Three kinds of *source*, given as one string (:func:`check`):

* ``"pypi"`` -- the newest release on PyPI that this Python can install
  (:func:`check_pypi`). The default for a pip install (:func:`default_source`).
* ``"owner/repo"`` -- that GitHub repository's latest **Release** (the
  GitHub API's ``/releases/latest``; drafts and pre-releases are
  skipped by GitHub itself). The installer for this platform is picked
  from the release's assets by file name (:data:`ASSET_PATTERNS`), which
  is what ``.github/workflows/build.yml`` names them. A *private*
  repository answers 404 unless a GitHub token with read access is
  given -- see :func:`check`.
* ``"https://..."`` -- a JSON manifest (:func:`check_for_update`,
  ``packaging/version_manifest.example.json``), for hosting releases
  anywhere else (a lab web page, a public releases-only repository).
"""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from importlib import metadata

from ._version import __version__


@dataclass
class UpdateInfo:
    """A newer version is available -- what :func:`check_for_update`
    returns when one is found."""

    version: str
    url: str
    notes: str = ""
    #: What an in-app install needs beyond ``url`` (all optional -- a
    #: manifest or release that omits them still offers the browser
    #: download, it just cannot be installed in one click):
    asset_name: str = ""
    #: Lower-case hex SHA-256 of the installer ("" = none published).
    sha256: str = ""
    size: int = 0
    #: The GitHub API address of the asset: the only way to download a
    #: file from a *private* repository (``url`` needs a browser login).
    api_url: str = ""
    #: A ``SHA256SUMS`` file published beside the installers, used when
    #: the release carries no digest of its own.
    checksums_url: str = ""
    checksums_api_url: str = ""


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
    version = version.strip()
    # A git tag's "v" ("v1.2.0"): without this the first component would
    # parse as 0 and v1.2.0 would compare older than 0.9.0.
    if version[:1] in ("v", "V"):
        version = version[1:]
    for piece in version.split("."):
        digits = ""
        for ch in piece:
            if ch.isdigit():
                digits += ch
            else:
                break
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def clean_sha256(value) -> str:
    """A SHA-256 as 64 lower-case hex characters, from ``"sha256:ab12…"``
    or plain hex; ``""`` when ``value`` is anything else (a malformed
    digest must read as "no checksum", never be compared)."""
    text = str(value or "").strip().lower()
    if text.startswith("sha256:"):
        text = text[len("sha256:"):]
    return text if re.fullmatch(r"[0-9a-f]{64}", text) else ""


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
            "darwin": "https://.../opennmr-0.2.0-macos.dmg",
            "windows": "https://.../opennmr-0.2.0-windows.zip",
            "linux": "https://.../opennmr-0.2.0-linux.tar.gz"
          }
        }

    Returns ``None`` if the manifest's version isn't newer than
    ``current_version`` (default: the running app's own
    :data:`opennmr._version.__version__`), or if it has no download
    URL for this platform (:func:`current_platform` by default).

    Raises whatever ``urllib.request.urlopen`` or ``json.loads`` raise
    on a genuine fetch/parse failure (a bad URL, no network, malformed
    JSON) -- deliberately not swallowed, so a caller can distinguish
    "checked, nothing newer" (``None``) from "the check itself failed"
    (an exception), which usually need different UI treatment.
    """
    current_version = current_version or __version__
    platform = platform or current_platform()

    with _urlopen(manifest_url, timeout) as response:
        manifest = json.loads(response.read().decode("utf-8"))

    latest_version = manifest.get("version")
    if not latest_version or not is_newer(latest_version, current_version):
        return None

    urls = manifest.get("url", {})
    url = urls.get(platform) or urls.get("*")
    if not url:
        return None

    digest = manifest.get("sha256") or ""
    if isinstance(digest, dict):
        digest = digest.get(platform) or digest.get("*") or ""
    return UpdateInfo(version=latest_version, url=url, notes=manifest.get("notes", ""),
                      asset_name=url.rsplit("/", 1)[-1], sha256=clean_sha256(digest))


# ============================================================
# Where updates come from, and how this copy was installed
# ============================================================

#: The GitHub source: the project's own repository. Its Releases are
#: where ``.github/workflows/build.yml`` puts the installers on a ``v*``
#: tag (the standalone app is checked here). While the repository is
#: private, a check without a token says so (see :class:`UpdateCheck`)
#: rather than failing silently.
DEFAULT_SOURCE = "openNMR/openNMR"

#: The source name for a pip install: ask PyPI (:func:`check_pypi`)
#: instead of GitHub, which a PyPI user may not be able to read at all
#: (the repository can be private) and which is not where their copy
#: came from anyway.
PYPI_SOURCE = "pypi"

#: The project's name on PyPI and the address of its JSON description.
PYPI_PROJECT = "opennmr"
PYPI_JSON_URL = f"https://pypi.org/pypi/{PYPI_PROJECT}/json"

#: Overrides :data:`PYPI_JSON_URL` (an ``https://`` or ``file://``
#: address): for rehearsing an update against a local copy of the JSON
#: before a release exists. Not needed in normal use.
PYPI_JSON_ENV = "OPENNMR_PYPI_JSON"

#: Environment variable read for a GitHub token when none is passed --
#: handy for a notebook, and kept out of any settings file.
TOKEN_ENV = "OPENNMR_GITHUB_TOKEN"

#: Set to ``0`` to stop the GUI checking at start-up (the test suite
#: does; so can a lab machine with no internet). "Check now" still works.
STARTUP_CHECK_ENV = "OPENNMR_UPDATE_CHECK"

#: Release asset names per platform, as build.yml writes them
#: (``openNMR-0.2.0-windows-setup.exe`` ...). First match wins, so the
#: real installer is preferred over a plain archive.
ASSET_PATTERNS = {
    "windows": (r"windows.*setup.*\.exe$", r"windows.*\.zip$", r"\.exe$"),
    "darwin": (r"macos.*\.dmg$", r"\.dmg$", r"macos.*\.zip$"),
    "linux": (r"linux.*\.appimage$", r"\.appimage$", r"linux.*\.tar\.gz$"),
}

#: The four ways this package can be running, each updated differently
#: (:func:`update_instructions`).
INSTALL_KINDS = {
    "app": "the installed openNMR app",
    "source": "a source folder (pip install -e)",
    "git": "pip, from GitHub",
    "pip": "pip",
}


def startup_check_enabled() -> bool:
    """Whether the environment allows a start-up check (on unless
    :data:`STARTUP_CHECK_ENV` says ``0``/``false``/``no``/``off``)."""
    value = os.environ.get(STARTUP_CHECK_ENV)
    if value is None:
        return True
    return value.strip().lower() not in ("0", "false", "no", "off")


def install_kind() -> str:
    """How this copy of openNMR was installed -- a key of
    :data:`INSTALL_KINDS`.

    ``"app"`` for the frozen installer build (PyInstaller sets
    ``sys.frozen``); otherwise read from the package's own install
    record (PEP 610 ``direct_url.json``): an editable install is a
    ``"source"`` folder, one from a git URL is ``"git"``, anything else
    (PyPI, a wheel file) is ``"pip"``. No metadata at all (running
    straight from a checkout on ``sys.path``) counts as ``"source"``."""
    if getattr(sys, "frozen", False):
        return "app"
    try:
        dist = metadata.distribution("opennmr")
    except metadata.PackageNotFoundError:
        return "source"
    try:
        record = json.loads(dist.read_text("direct_url.json") or "null")
    except (OSError, ValueError):
        record = None
    if not isinstance(record, dict):
        return "pip"
    if (record.get("dir_info") or {}).get("editable"):
        return "source"
    if record.get("vcs_info"):
        return "git"
    return "pip"


def update_instructions(kind: str, version: str | None = None,
                        source: str = DEFAULT_SOURCE) -> str:
    """One plain sentence (and command) saying how to update an install
    of ``kind``. The notebook package and the app are updated
    separately: they are separate copies (the app carries its own
    Python), so updating one never changes the other."""
    target = f"openNMR {version}" if version else "the newest openNMR"
    repo = source if _is_repo(source) else DEFAULT_SOURCE
    if kind == "app":
        return (f"Download {target} and run the installer: it replaces this version "
                "and keeps your settings (~/.opennmr).")
    if kind == "source":
        return ("This copy runs from a source folder. In that folder run:\n"
                "    git pull\n"
                "(then  pip install -e .  if the dependencies changed).")
    if kind == "git":
        return ("In the Python environment you installed it into, run:\n"
                f"    pip install --upgrade git+https://github.com/{repo}.git")
    # "pip": installed from PyPI (or a wheel file), so PyPI is the source.
    return ("In the Python environment you installed it into, run:\n"
            f"    pip install --upgrade {PYPI_PROJECT}")


def default_source(kind: str | None = None) -> str:
    """Where a copy installed as ``kind`` looks for updates unless told
    otherwise: PyPI for a pip install, the GitHub repository for the
    app, a git install and a source folder."""
    return PYPI_SOURCE if (kind or install_kind()) == "pip" else DEFAULT_SOURCE


def pip_command(kind: str, source: str = DEFAULT_SOURCE) -> str | None:
    """The single command a pip/git/source install updates with (for a
    "Copy" button), or ``None`` for the app."""
    repo = source if _is_repo(source) else DEFAULT_SOURCE
    git_command = f"pip install --upgrade git+https://github.com/{repo}.git"
    return {"source": "git pull", "git": git_command,
            "pip": f"pip install --upgrade {PYPI_PROJECT}"}.get(kind)


# ============================================================
# Checking
# ============================================================

@dataclass
class UpdateCheck:
    """What one check found -- always returned, never raised, so the GUI
    can show it as it is.

    ``status`` is ``"newer"`` (``info`` set), ``"current"`` (nothing
    newer than ``current_version``) or ``"error"`` (``message`` says
    what went wrong in words a user can act on)."""

    status: str
    current_version: str
    info: UpdateInfo | None = None
    message: str = ""
    release_page: str = ""
    latest_version: str = ""
    assets: dict = field(default_factory=dict)


_REPO_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


def _is_repo(source: str | None) -> bool:
    return bool(source) and bool(_REPO_RE.match(source.strip()))


def pick_asset(assets: list[dict], platform: str) -> dict | None:
    """The release asset for ``platform`` (by :data:`ASSET_PATTERNS`,
    case-insensitively), or ``None``."""
    names = [(a.get("name") or "", a) for a in assets or []]
    for pattern in ASSET_PATTERNS.get(platform, ()):
        for name, asset in names:
            if re.search(pattern, name, flags=re.IGNORECASE):
                return asset
    return None


#: A release file listing the installers' SHA-256 sums (``sha256sum``'s
#: format), which build.yml publishes beside them.
CHECKSUM_PATTERN = r"^(sha256sums?|checksums?)(\.txt)?$"


def pick_checksums_asset(assets: list[dict]) -> dict | None:
    """The release's ``SHA256SUMS`` file, or ``None``."""
    for asset in assets or []:
        if re.search(CHECKSUM_PATTERN, asset.get("name") or "", flags=re.IGNORECASE):
            return asset
    return None


def ssl_context():
    """The TLS context for update checks: the system's certificates, plus
    certifi's bundle when it is installed (it comes with pymatgen's
    ``requests``).

    certifi matters in the frozen app: PyInstaller's Python looks for
    OpenSSL's certificate folder *of the build machine*, which does not
    exist on a user's Mac, so every https request failed with
    CERTIFICATE_VERIFY_FAILED. certifi's ``cacert.pem`` travels inside
    the app (PyInstaller collects it)."""
    import ssl

    context = ssl.create_default_context()
    try:
        import certifi
        context.load_verify_locations(cafile=certifi.where())
    except (ImportError, OSError):
        pass
    return context


def _urlopen(request, timeout: float):
    return urllib.request.urlopen(request, timeout=timeout, context=ssl_context())


def _github_request(url: str, token: str | None, timeout: float):
    request = urllib.request.Request(url, headers={
        "Accept": "application/vnd.github+json",
        "User-Agent": "openNMR-update-check",
        "X-GitHub-Api-Version": "2022-11-28",
    })
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    with _urlopen(request, timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def check_github_release(repo: str, current_version: str | None = None,
                         platform: str | None = None, token: str | None = None,
                         timeout: float = 5.0) -> UpdateCheck:
    """Compare the running version with ``repo``'s latest GitHub Release.

    The tag (``v0.2.0``) is the version. For the app the download is
    this platform's installer asset; for a pip install it is the
    release page (whose notes say what changed). A release without an
    installer for this platform still counts as newer -- the page is
    offered instead, so a missing build never hides an update."""
    current_version = current_version or __version__
    platform = platform or current_platform()
    token = token or os.environ.get(TOKEN_ENV) or None
    url = f"https://api.github.com/repos/{repo.strip()}/releases/latest"
    try:
        release = _github_request(url, token, timeout)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            if token:
                message = (f"No published release found in {repo} (or the token cannot read it). "
                           "A release appears once a version tag (e.g. v0.2.0) has been built.")
            else:
                message = (f"No published release found in {repo}. While the repository is "
                           "private, GitHub only answers with an access token (Settings ▸ "
                           "Updates ▸ Access token); otherwise no release has been made yet.")
        elif exc.code in (401, 403):
            message = (f"GitHub refused the check ({exc.code}). The access token may be wrong "
                       "or expired, or GitHub's hourly limit for unauthenticated checks was "
                       "reached -- try again later.")
        else:
            message = f"GitHub answered {exc.code} ({exc.reason})."
        return UpdateCheck("error", current_version, message=message)
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        reason = getattr(exc, "reason", exc)
        return UpdateCheck("error", current_version,
                           message=f"Could not reach GitHub ({reason}). Check the internet connection.")
    except ValueError:
        return UpdateCheck("error", current_version, message="GitHub's answer could not be read.")

    tag = str(release.get("tag_name") or "").strip()
    page = str(release.get("html_url") or f"https://github.com/{repo}/releases")
    assets = {a.get("name"): a.get("browser_download_url") for a in release.get("assets") or []}
    if not tag:
        return UpdateCheck("error", current_version, message="The latest release has no version tag.",
                           release_page=page)
    latest = tag[1:] if tag[:1] in ("v", "V") else tag
    if not is_newer(latest, current_version):
        return UpdateCheck("current", current_version, release_page=page,
                           latest_version=latest, assets=assets)
    asset = pick_asset(release.get("assets") or [], platform) or {}
    sums = pick_checksums_asset(release.get("assets") or []) or {}
    download = asset.get("browser_download_url") or page
    info = UpdateInfo(
        version=latest, url=download, notes=str(release.get("body") or "").strip(),
        asset_name=str(asset.get("name") or ""), sha256=clean_sha256(asset.get("digest")),
        size=int(asset.get("size") or 0), api_url=str(asset.get("url") or ""),
        checksums_url=str(sums.get("browser_download_url") or ""),
        checksums_api_url=str(sums.get("url") or ""))
    return UpdateCheck("newer", current_version, info=info, release_page=page,
                       latest_version=latest, assets=assets)


# ---- PyPI (a pip install) --------------------------------------------

def _pep440(version: str):
    """``version`` as a :class:`packaging.version.Version`, or ``None``
    when it is not PEP 440 (or ``packaging`` is missing, which it never
    is in a real install: matplotlib needs it)."""
    try:
        from packaging.version import Version
        return Version(str(version).strip())
    except Exception:  # InvalidVersion, ImportError
        return None


def is_newer_release(candidate: str, current: str) -> bool:
    """:func:`is_newer`, but PEP 440 aware where it can be: ``0.1.0`` is
    newer than ``0.1.0.dev2`` and ``0.1.1`` than ``0.1.0``. Falls back to
    :func:`is_newer` when either side is not a PEP 440 version."""
    a, b = _pep440(candidate), _pep440(current)
    if a is not None and b is not None:
        return a > b
    return is_newer(candidate, current)


def python_satisfies(requires_python: str | None, python_version: str | None = None) -> bool:
    """Whether the running Python (or ``python_version``, ``"3.12.4"``)
    meets a release's ``requires_python`` (``">=3.12"``). ``True`` when
    there is no requirement or it cannot be read: pip itself refuses an
    incompatible release, so an unreadable one is never hidden."""
    if not requires_python:
        return True
    try:
        from packaging.specifiers import SpecifierSet
        from packaging.version import Version
        running = python_version or "{}.{}.{}".format(*sys.version_info[:3])
        return SpecifierSet(requires_python).contains(Version(running), prereleases=True)
    except Exception:
        return True


def pick_pypi_release(data: dict, python_version: str | None = None) -> str:
    """The newest version in PyPI's JSON description of the project that
    this Python can install, or ``""``.

    Skipped: pre-releases and development versions (a ``0.2.0rc1`` is
    never offered), releases whose every file was yanked, and releases
    whose ``requires_python`` this Python does not meet (offering one
    that pip would then refuse is a broken update). Reads ``releases``;
    when PyPI no longer sends that (it is deprecated there) it falls
    back to ``info`` and ``urls``, which describe the latest release."""
    releases = data.get("releases")
    if not isinstance(releases, dict):
        info = data.get("info") or {}
        name = str(info.get("version") or "")
        files = [] if info.get("yanked") else [
            dict(f, requires_python=f.get("requires_python") or info.get("requires_python"))
            for f in data.get("urls") or []]
        releases = {name: files} if name else {}
    best, best_name = None, ""
    for name, files in releases.items():
        version = _pep440(name)
        if version is None or version.is_prerelease or version.is_devrelease:
            continue
        usable = [f for f in files or [] if not f.get("yanked")]
        if not usable:
            continue
        if not any(python_satisfies(f.get("requires_python"), python_version) for f in usable):
            continue
        if best is None or version > best:
            best, best_name = version, str(name)
    return best_name


def check_pypi(current_version: str | None = None, python_version: str | None = None,
               timeout: float = 5.0, url: str | None = None) -> UpdateCheck:
    """Compare the running version with the newest one on PyPI that this
    Python can install (:func:`pick_pypi_release`). Never raises. The
    result's ``release_page`` is the project's PyPI page; PyPI has no
    release notes to show."""
    current_version = current_version or __version__
    url = url or os.environ.get(PYPI_JSON_ENV) or PYPI_JSON_URL
    page = f"https://pypi.org/project/{PYPI_PROJECT}/"
    request = urllib.request.Request(url, headers={
        "Accept": "application/json", "User-Agent": "openNMR-update-check"})
    try:
        with _urlopen(request, timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        message = (f"PyPI has no project called {PYPI_PROJECT} yet." if exc.code == 404
                   else f"PyPI answered {exc.code} ({exc.reason}).")
        return UpdateCheck("error", current_version, message=message, release_page=page)
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        reason = getattr(exc, "reason", exc)
        return UpdateCheck("error", current_version, release_page=page,
                           message=f"Could not reach PyPI ({reason}). Check the internet connection.")
    except ValueError:
        return UpdateCheck("error", current_version, release_page=page,
                           message="PyPI's answer could not be read.")
    if not isinstance(data, dict):
        return UpdateCheck("error", current_version, release_page=page,
                           message="PyPI's answer could not be read.")
    latest = pick_pypi_release(data, python_version)
    if not latest or not is_newer_release(latest, current_version):
        return UpdateCheck("current", current_version, release_page=page,
                           latest_version=latest or current_version)
    release_page = f"{page}{latest}/"
    info = UpdateInfo(version=latest, url=release_page)
    return UpdateCheck("newer", current_version, info=info, release_page=release_page,
                       latest_version=latest)


def check(source: str | None = None, current_version: str | None = None,
          platform: str | None = None, token: str | None = None,
          timeout: float = 5.0) -> UpdateCheck:
    """Check ``source`` (default :data:`DEFAULT_SOURCE`) for a newer
    version. ``"pypi"`` asks PyPI (:func:`check_pypi`; what a pip install
    uses); ``"owner/repo"`` reads GitHub Releases
    (:func:`check_github_release`); a URL reads a JSON manifest
    (:func:`check_for_update`). Never raises: a failure comes back as
    ``status == "error"`` with a readable message.

    From a notebook::

        from opennmr import update
        result = update.check()
        print(result.status, result.message or result.info)
        print(update.update_instructions(update.install_kind()))
    """
    current_version = current_version or __version__
    source = (source or DEFAULT_SOURCE).strip()
    if source.lower() == PYPI_SOURCE:
        return check_pypi(current_version, timeout=timeout)
    if _is_repo(source):
        return check_github_release(source, current_version, platform, token, timeout)
    if not source.lower().startswith(("https://", "http://")):
        return UpdateCheck("error", current_version,
                           message=f"'{source}' is neither owner/repo nor an http(s) address.")
    try:
        info = check_for_update(source, current_version, platform, timeout)
    except urllib.error.HTTPError as exc:
        return UpdateCheck("error", current_version,
                           message=f"The update manifest answered {exc.code} ({exc.reason}).")
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        reason = getattr(exc, "reason", exc)
        return UpdateCheck("error", current_version, message=f"Could not reach {source} ({reason}).")
    except (ValueError, AttributeError):
        return UpdateCheck("error", current_version, message="The update manifest could not be read.")
    if info is None:
        return UpdateCheck("current", current_version)
    return UpdateCheck("newer", current_version, info=info, latest_version=info.version)
