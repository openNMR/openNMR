"""The manual release route, one command per step (for when GitHub Actions is not available).

    python packaging/release_tools.py zip       # on the PC, after release.py: a clean source zip of the tag
    python packaging/release_tools.py build     # on EACH machine (Windows, Mac, Linux): build + test + collect
    python packaging/release_tools.py publish   # on the PC: checksums, GitHub release, optionally PyPI

Everything lands in ``release-X.Y.Z/`` inside the repository folder (git
ignores it), so there is one folder to carry between machines.

``build`` is the one to send to a friend with a Mac: unzip the source zip,
``cd`` into it and run ``python packaging/release_tools.py build``. It makes
its own virtual environment (``.release-venv``), installs the build tools,
builds the app and the installer, runs the self-test on the source and on the
frozen app, compares them, and copies the installer into ``release-X.Y.Z/``.
Send that one file back. Use ``--here`` to build in the Python you are already
using (your own PC, with everything installed).

``publish`` writes ``SHA256SUMS`` for the installers it finds, then creates (or
adds to) the GitHub release with the ``gh`` command-line tool if it is
installed (``winget install GitHub.cli``, then ``gh auth login``); without it,
it tells you what to drag onto the release page by hand. ``--pypi`` also builds
the wheel and sdist (into a temporary folder, so ``dist/`` is untouched), checks
them and uploads them to PyPI after you type the version to confirm: that
upload cannot be undone.

``--dry-run`` on any command prints what would be run and does nothing.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from release import read_version  # noqa: E402

DEFAULT_REPO = "openNMR/openNMR"
VENV_NAME = ".release-venv"
PLATFORMS = ("windows", "macos", "linux")

#: Installer file names as ``make_installer.py`` writes them (``{v}`` = the version).
INSTALLER_PATTERNS = {
    "windows": "openNMR-{v}-windows-setup.exe",
    "macos": "openNMR-{v}-macos-*.dmg",
    "linux": "openNMR-{v}-linux-*.AppImage",
}

#: Folders a build leaves behind that must not leak into the next one (a stale
#: ``dist`` once let a hand-patched library into an installer).
BUILD_LEFTOVERS = ("dist", "build", "st-src", "st-app", "st-appimage")


# ---- pure helpers (tested without running anything) --------------------------------------

def platform_key(system: str | None = None) -> str:
    """``"windows"``, ``"macos"`` or ``"linux"`` for ``sys.platform`` (or ``system``)."""
    system = sys.platform if system is None else system
    if system.startswith("win"):
        return "windows"
    return "macos" if system == "darwin" else "linux"


def installer_pattern(platform: str, version: str) -> str:
    return INSTALLER_PATTERNS[platform].format(v=version)


def find_installer(directory: str, platform: str, version: str) -> str | None:
    """The installer for ``platform``/``version`` in ``directory``, or ``None``."""
    hits = sorted(glob.glob(os.path.join(directory, installer_pattern(platform, version))))
    return hits[0] if hits else None


def find_installers(directory: str, version: str) -> dict[str, str | None]:
    return {p: find_installer(directory, p, version) for p in PLATFORMS}


def frozen_command(platform: str, dist: str) -> str:
    """The built app's executable inside ``dist``."""
    if platform == "windows":
        return os.path.join(dist, "openNMR", "openNMR.exe")
    if platform == "macos":
        return os.path.join(dist, "openNMR.app", "Contents", "MacOS", "openNMR")
    return os.path.join(dist, "openNMR", "openNMR")


def venv_python(venv: str) -> str:
    return (os.path.join(venv, "Scripts", "python.exe") if os.name == "nt"
            else os.path.join(venv, "bin", "python"))


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def checksum_text(files: list[str]) -> str:
    """``sha256sum``'s format (``<hash>  <name>``, LF line ends, sorted by name): what the
    in-app updater reads."""
    lines = [f"{sha256_file(path)}  {os.path.basename(path)}"
             for path in sorted(files, key=os.path.basename)]
    return "\n".join(lines) + "\n"


def write_checksums(files: list[str], dest: str) -> str:
    text = checksum_text(files)
    with open(dest, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    return text


@dataclass
class Step:
    name: str
    argv: list[str]
    env: dict = field(default_factory=dict)


def plan_build(platform: str, python: str, dist: str = "dist") -> list[Step]:
    """The commands that build and test this system's installer, in order. ``{installer}`` in
    an argument is replaced by the installer's path once it exists."""
    frozen = frozen_command(platform, dist)
    env = {"QT_QPA_PLATFORM": "offscreen"} if platform == "linux" else {}
    steps = [
        Step("install the build tools (editable, so the version is current)",
             [python, "-m", "pip", "install", "-e", ".[build]"]),
        Step("build the app", [python, os.path.join("packaging", "build.py")]),
        Step("build the installer", [python, os.path.join("packaging", "make_installer.py")]),
    ]
    if platform == "macos":
        steps.append(Step("check the app's signature",
                          ["codesign", "--verify", "--deep", "--strict",
                           os.path.join(dist, "openNMR.app")]))
    steps += [
        Step("self-test, from source", [python, "-m", "opennmr.selftest", "st-src"], env),
        Step("self-test, the built app", [frozen, "--self-test", "st-app"], env),
        Step("compare source and built app",
             [python, os.path.join("packaging", "compare_selftest.py"), "st-src", "st-app"], env),
    ]
    if platform == "linux":
        steps += [
            Step("self-test, the AppImage",
                 ["{installer}", "--appimage-extract-and-run", "--self-test", "st-appimage"], env),
            Step("compare source and AppImage",
                 [python, os.path.join("packaging", "compare_selftest.py"), "st-src", "st-appimage"],
                 env),
        ]
    return steps


def plan_zip(version: str, out_dir: str, tag: str | None = None) -> tuple[list[str], str]:
    """``git archive`` of the tag: only tracked files, Unix line ends, no ``.git``."""
    tag = tag or f"v{version}"
    target = os.path.join(out_dir, f"openNMR-{version}-src.zip")
    return ["git", "archive", "--format=zip", f"--prefix=openNMR-{version}/", "-o", target, tag], target


def gh_release_argv(version: str, repo: str, files: list[str], notes: str, exists: bool) -> list[str]:
    """``gh`` command that makes the release, or adds files to it if it exists already."""
    tag = f"v{version}"
    if exists:
        return ["gh", "release", "upload", tag, *files, "--clobber", "--repo", repo]
    return ["gh", "release", "create", tag, *files, "--repo", repo, "--verify-tag",
            "--title", f"openNMR {version}", "--notes", notes or f"openNMR {version}"]


def missing_installers(found: dict[str, str | None]) -> list[str]:
    return [p for p in PLATFORMS if not found.get(p)]


# ---- running -----------------------------------------------------------------------------

def show(argv: list[str]) -> str:
    return " ".join(f'"{a}"' if " " in a else a for a in argv)


def run_steps(steps: list[Step], cwd: str, *, installer: str = "", dry_run: bool = False,
              runner=subprocess.run) -> int:
    """Run ``steps``; stop at the first failure and say which."""
    for number, step in enumerate(steps, start=1):
        argv = [a.replace("{installer}", installer) for a in step.argv]
        print(f"\n==> [{number}/{len(steps)}] {step.name}\n    $ {show(argv)}", flush=True)
        if dry_run:
            continue
        env = {**os.environ, **step.env}
        result = runner(argv, cwd=cwd, env=env, stdin=subprocess.DEVNULL)
        if result.returncode != 0:
            print(f"\nFAILED at step {number}: {step.name} (exit code {result.returncode}). "
                  "Nothing after this step was run; fix it and run the command again.")
            return result.returncode or 1
    return 0


def cmd_zip(args, repo: str = REPO_ROOT) -> int:
    version = read_version(repo)
    out_dir = args.out or os.path.join(repo, f"release-{version}")
    tag = args.tag or f"v{version}"
    listed = subprocess.run(["git", "tag", "--list", tag], cwd=repo, capture_output=True,
                            text=True, stdin=subprocess.DEVNULL).stdout.strip()
    if not listed:
        print(f"There is no tag {tag}. Run  python packaging/release.py {version}  first "
              "(or pass --tag).")
        return 1
    argv, target = plan_zip(version, out_dir, tag)
    print(f"==> source of {tag}\n    $ {show(argv)}")
    if args.dry_run:
        return 0
    os.makedirs(out_dir, exist_ok=True)
    code = subprocess.run(argv, cwd=repo, stdin=subprocess.DEVNULL).returncode
    if code == 0:
        print(f"\nWrote {target}\nSend it to the other machines, unzip it and run, in the unzipped "
              "folder:\n    python packaging/release_tools.py build")
    return code


def cmd_build(args, repo: str = REPO_ROOT) -> int:
    version = read_version(repo)
    platform = platform_key()
    out_dir = args.out or os.path.join(repo, f"release-{version}")
    print(f"Building the openNMR {version} installer for {platform} ({sys.platform}).")
    python = sys.executable
    if not args.here:
        venv = os.path.join(repo, VENV_NAME)
        python = venv_python(venv)
        if not os.path.exists(python) and not args.dry_run:
            print(f"==> making a private virtual environment in {venv}")
            if subprocess.run([sys.executable, "-m", "venv", venv], stdin=subprocess.DEVNULL).returncode:
                print("Could not create the virtual environment.")
                return 1
            subprocess.run([python, "-m", "pip", "install", "--upgrade", "pip"],
                           stdin=subprocess.DEVNULL)
    if not args.dry_run:
        for name in BUILD_LEFTOVERS:
            shutil.rmtree(os.path.join(repo, name), ignore_errors=True)
    steps = plan_build(platform, python)
    # The AppImage test needs the installer, which only exists after the build: split the plan.
    split = next((i for i, s in enumerate(steps) if "{installer}" in " ".join(s.argv)), len(steps))
    first, later = steps[:split], steps[split:]
    code = run_steps(first, repo, dry_run=args.dry_run)
    installer = os.path.join(repo, "dist", installer_pattern(platform, version))
    if code == 0 and not args.dry_run:
        found = find_installer(os.path.join(repo, "dist"), platform, version)
        if not found:
            print(f"The build finished but no installer matching {installer_pattern(platform, version)} "
                  "is in dist/.")
            return 1
        installer = found
    if code == 0:
        code = run_steps(later, repo, installer=installer, dry_run=args.dry_run)
    if code != 0 or args.dry_run:
        return code
    os.makedirs(out_dir, exist_ok=True)
    target = os.path.join(out_dir, os.path.basename(installer))
    shutil.copy2(installer, target)
    print(f"\nAll checks passed.\n  installer: {target}\n  sha256:    {sha256_file(target)}")
    print("\nOne check is left for a person: install it, launch openNMR, and confirm About says "
          f"{version}.\nThen send {os.path.basename(target)} to whoever publishes the release.")
    return 0


def cmd_publish(args, repo: str = REPO_ROOT) -> int:
    version = read_version(repo)
    out_dir = args.out_dir or os.path.join(repo, f"release-{version}")
    found = find_installers(out_dir, version)
    missing = missing_installers(found)
    have = [path for path in found.values() if path]
    if not have:
        print(f"No installers for {version} in {out_dir}. Put them there (see `build`).")
        return 1
    if missing and not args.allow_missing:
        print(f"Missing in {out_dir}: " + ", ".join(INSTALLER_PATTERNS[p].format(v=version) for p in missing)
              + "\nAdd them, or pass --allow-missing to publish without (those users get "
              "'no installer for your system' in the app).")
        return 1
    sums = os.path.join(out_dir, "SHA256SUMS")
    print(f"==> checksums for {len(have)} installer(s) -> {sums}")
    if args.dry_run:
        print(checksum_text(have), end="")
    else:
        print(write_checksums(have, sums), end="")
    files = have + [sums]
    notes = args.notes
    if args.notes_file:
        with open(args.notes_file, encoding="utf-8") as handle:
            notes = handle.read()
    code = 0
    if shutil.which("gh"):
        exists = (not args.dry_run and subprocess.run(
            ["gh", "release", "view", f"v{version}", "--repo", args.repo],
            capture_output=True, stdin=subprocess.DEVNULL).returncode == 0)
        argv = gh_release_argv(version, args.repo, files, notes, exists)
        print(f"\n==> GitHub release v{version} ({'adding to the existing one' if exists else 'creating it'})"
              f"\n    $ {show(argv)}")
        if not args.dry_run:
            code = subprocess.run(argv, stdin=subprocess.DEVNULL).returncode
            if code:
                print("gh failed (are you logged in? try `gh auth login`).")
                return code
    else:
        print(f"\nThe GitHub CLI (`gh`) is not installed, so do this by hand: on GitHub open "
              f"Releases, Draft a new release, choose tag v{version}, drag in\n  "
              + "\n  ".join(files) + "\nand publish. (`winget install GitHub.cli` then `gh auth login` "
              "automates this next time.)")
    if args.pypi:
        code = _pypi(version, args)
    else:
        print("\nPyPI was not touched (add --pypi to upload the wheel and sdist).")
    return code


def _pypi(version: str, args) -> int:
    """Build into a temporary folder (``dist/`` may hold the app), check, confirm, upload."""
    tmp = tempfile.mkdtemp(prefix="opennmr-pypi-")
    steps = [[sys.executable, "-m", "build", "--outdir", tmp]]
    print(f"\n==> PyPI upload of {version}")
    for argv in steps:
        print(f"    $ {show(argv)}")
    if args.dry_run:
        print(f"    $ {show([sys.executable, '-m', 'twine', 'check', os.path.join(tmp, '*')])}\n"
              f"    $ {show([sys.executable, '-m', 'twine', 'upload', os.path.join(tmp, '*')])}")
        return 0
    if subprocess.run(steps[0], cwd=REPO_ROOT, stdin=subprocess.DEVNULL).returncode:
        return 1
    files = sorted(glob.glob(os.path.join(tmp, f"opennmr-{version}*")))
    if len(files) != 2:
        print(f"Expected a wheel and an sdist for {version} in {tmp}, found {files}.")
        return 1
    if subprocess.run([sys.executable, "-m", "twine", "check", *files],
                      stdin=subprocess.DEVNULL).returncode:
        return 1
    if not args.yes:
        answer = input(f"\nUploading {version} to PyPI cannot be undone. Type the version to continue: ")
        if answer.strip() != version:
            print("Not uploaded.")
            return 1
    return subprocess.run([sys.executable, "-m", "twine", "upload", *files]).returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p_zip = sub.add_parser("zip", help="source zip of the release tag, for other machines")
    p_zip.add_argument("--out", help="output folder (default release-X.Y.Z)")
    p_zip.add_argument("--tag", help="tag to archive (default vX.Y.Z)")
    p_zip.add_argument("--dry-run", action="store_true")
    p_build = sub.add_parser("build", help="build, test and collect this system's installer")
    p_build.add_argument("--out", help="output folder (default release-X.Y.Z)")
    p_build.add_argument("--here", action="store_true",
                         help="use the Python running this script instead of a private venv")
    p_build.add_argument("--dry-run", action="store_true")
    p_pub = sub.add_parser("publish", help="checksums + GitHub release (+ PyPI)")
    p_pub.add_argument("out_dir", nargs="?", help="folder with the installers (default release-X.Y.Z)")
    p_pub.add_argument("--repo", default=DEFAULT_REPO, help=f"GitHub repo (default {DEFAULT_REPO})")
    p_pub.add_argument("--notes", default="", help="release notes text")
    p_pub.add_argument("--notes-file", help="release notes from a file")
    p_pub.add_argument("--allow-missing", action="store_true", help="publish with some installers absent")
    p_pub.add_argument("--pypi", action="store_true", help="also upload the wheel and sdist to PyPI")
    p_pub.add_argument("--yes", action="store_true", help="skip the PyPI confirmation prompt")
    p_pub.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    return {"zip": cmd_zip, "build": cmd_build, "publish": cmd_publish}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
