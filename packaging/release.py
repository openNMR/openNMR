"""Make a release: ONE command from "the code is ready" to a tagged version.

    python packaging/release.py 0.8.2            # prepare, check, tag (nothing leaves your machine)
    python packaging/release.py 0.8.2 --publish  # ... and push + upload to PyPI

What it does, in order -- and stops at the first problem:

1. refuses unless the working tree is clean and the version is new
   (greater than the one in ``pyproject.toml`` and not already a tag);
2. sets the version in the two places that must agree
   (``pyproject.toml`` and ``_FALLBACK_VERSION`` in ``src/opennmr/_version.py``;
   ``tests/test_packaging.py`` fails if they ever differ);
3. builds the wheel and sdist (``python -m build``) and checks them
   (``twine check``) into ``dist/``;
4. commits "Release X.Y.Z" and tags ``vX.Y.Z``.

With ``--publish`` it then pushes the commit and the tag, and uploads the
wheel and sdist to PyPI with twine.

**The installers are not built here.** Pushing the tag starts
``.github/workflows/build.yml``, which builds and checks the Windows,
macOS and Linux apps, then publishes the GitHub Release with the
installers and ``SHA256SUMS`` (what Settings > Updates installs from).
That workflow also publishes to PyPI (skipping a version already there),
so with Actions minutes available a plain ``git push`` of the tag is
enough and ``--publish`` is only the fallback. Without minutes, build an
installer on your own machine with ``packaging/build.py`` +
``packaging/make_installer.py`` (see packaging/README.md).

The remote defaults to ``upstream`` (what this repository calls
``openNMR/openNMR``); change it with ``--remote``.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_VERSION_RE = re.compile(r"^\d+\.\d+\.\d+$")


def parse_version(text: str) -> tuple[int, int, int]:
    """``"0.8.2"`` -> ``(0, 8, 2)``; anything but X.Y.Z (no ``v``, no suffix) raises."""
    if not _VERSION_RE.match(text):
        raise ValueError(f"version must look like 0.8.2 (no 'v', no suffix), not {text!r}")
    major, minor, patch = (int(part) for part in text.split("."))
    return major, minor, patch


def read_version(repo: str = REPO_ROOT) -> str:
    with open(os.path.join(repo, "pyproject.toml"), encoding="utf-8") as handle:
        match = re.search(r'^version\s*=\s*"([^"]+)"', handle.read(), flags=re.MULTILINE)
    if match is None:
        raise ValueError("no version = \"...\" line in pyproject.toml")
    return match.group(1)


def _replace_once(path: str, pattern: str, replacement: str) -> None:
    with open(path, encoding="utf-8", newline="") as handle:
        text = handle.read()
    new, count = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE)
    if count != 1:
        raise ValueError(f"could not find the version line in {path}")
    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write(new)


def set_version(version: str, repo: str = REPO_ROOT) -> None:
    """Write ``version`` to pyproject.toml and the _version.py fallback."""
    parse_version(version)
    _replace_once(os.path.join(repo, "pyproject.toml"),
                  r'^version\s*=\s*"[^"]+"', f'version = "{version}"')
    _replace_once(os.path.join(repo, "src", "opennmr", "_version.py"),
                  r'^_FALLBACK_VERSION\s*=\s*"[^"]+"', f'_FALLBACK_VERSION = "{version}"')


def git(*args: str, repo: str = REPO_ROOT, check: bool = True) -> str:
    # stdin=DEVNULL: an inherited terminal handle (Git Bash's pty) cannot be
    # duplicated on Windows (WinError 50) when output is captured.
    done = subprocess.run(["git", *args], cwd=repo, stdin=subprocess.DEVNULL,
                          capture_output=True, text=True)
    if check and done.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed:\n{done.stderr.strip() or done.stdout.strip()}")
    return done.stdout.strip()


def check_ready(version: str, repo: str = REPO_ROOT) -> list[str]:
    """Reasons this release cannot be made, empty when it can."""
    problems = []
    try:
        new = parse_version(version)
    except ValueError as exc:
        return [str(exc)]
    current = read_version(repo)
    if new <= parse_version(current):
        problems.append(f"{version} is not newer than the current version {current}.")
    if git("status", "--porcelain", repo=repo):
        problems.append("The working tree has uncommitted changes: commit or stash them first "
                        "(the release commit must contain only the version bump).")
    if git("tag", "--list", f"v{version}", repo=repo):
        problems.append(f"The tag v{version} already exists.")
    return problems


def build_distributions(repo: str = REPO_ROOT) -> list[str]:
    """Wheel + sdist for the current version, checked; returns their paths."""
    dist = os.path.join(repo, "dist")
    version = read_version(repo)
    subprocess.run([sys.executable, "-m", "build", "--outdir", dist], cwd=repo, check=True)
    files = [os.path.join(dist, name) for name in sorted(os.listdir(dist))
             if name.startswith(f"opennmr-{version}") and name.endswith((".whl", ".tar.gz"))]
    if len(files) != 2:
        raise RuntimeError(f"expected a wheel and an sdist for {version} in {dist}, found {files}")
    subprocess.run([sys.executable, "-m", "twine", "check", *files], check=True)
    return files


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("version", help="the new version, e.g. 0.8.2")
    parser.add_argument("--publish", action="store_true",
                        help="push the commit and tag, and upload the wheel and sdist to PyPI")
    parser.add_argument("--remote", default="upstream", help="git remote to push to (default upstream)")
    parser.add_argument("--branch", default="main", help="branch the release is made from (default main)")
    parser.add_argument("--tests", action="store_true",
                        help="run the test suite (pytest -x -q -n auto) before releasing")
    args = parser.parse_args(argv)

    branch = git("rev-parse", "--abbrev-ref", "HEAD")
    if branch != args.branch:
        print(f"On branch {branch!r}; releases are made from {args.branch!r} (--branch to change).")
        return 1
    problems = check_ready(args.version)
    if problems:
        print("Not releasing:\n  " + "\n  ".join(problems))
        return 1
    if args.tests:
        if subprocess.run([sys.executable, "-m", "pytest", "-x", "-q", "-n", "auto"],
                          cwd=REPO_ROOT).returncode != 0:
            print("Tests failed: not releasing.")
            return 1

    old = read_version()
    set_version(args.version)
    try:
        files = build_distributions()
    except Exception:
        set_version(old)                      # leave the tree as it was
        raise
    git("add", "pyproject.toml", "src/opennmr/_version.py")
    git("commit", "-m", f"Release {args.version}")
    git("tag", "-a", f"v{args.version}", "-m", f"openNMR {args.version}")
    print(f"Committed and tagged v{args.version}; built:\n  " + "\n  ".join(files))

    if not args.publish:
        print(f"\nNothing has left this machine. To publish:\n"
              f"  git push {args.remote} {args.branch} v{args.version}     "
              f"(starts the installer build + GitHub Release + PyPI)\n"
              f"  python -m twine upload {' '.join(os.path.relpath(f, REPO_ROOT) for f in files)}"
              f"     (PyPI now, without waiting for the build)\n"
              f"or re-run with --publish to do both.")
        return 0

    git("push", args.remote, args.branch)
    git("push", args.remote, f"v{args.version}")
    subprocess.run([sys.executable, "-m", "twine", "upload", *files], check=True)
    print(f"Pushed v{args.version} and uploaded to PyPI. The installers appear on the GitHub "
          f"release once the Release workflow finishes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
