#!/usr/bin/env python
"""Rehearse an in-app update end to end, before a real release exists.

It builds two wheels of this checkout -- the current version (A) and the
next patch version (B) -- into a temporary folder, installs A into a fresh
virtual environment (its dependencies come from PyPI), and points openNMR's
update check and pip at that folder instead of PyPI:

* ``OPENNMR_PYPI_JSON``  -- a local JSON file saying B is the newest release;
* ``OPENNMR_PIP_ARGS``   -- ``--no-index --find-links <the wheels>``;
* ``OPENNMR_STATE_DIR``  -- where the helper's log and result are written.

Then, by default, it opens the GUI of A: Settings > Updates > Check for
Updates should offer B; "Update to B and restart" should close the window,
upgrade, and open B, whose Updates panel says "openNMR was updated to B."
The script waits for the new version to appear and prints the log.

``--no-gui`` does the same without a window (the check, the helper, the
hidden upgrade): usable in a terminal or CI.

    python tools/rehearse_update.py            # watch it in the GUI
    python tools/rehearse_update.py --no-gui   # no window
    python tools/rehearse_update.py --keep     # keep the folder afterwards

Needs network access once, to install A's dependencies. Nothing is
uploaded and the real PyPI is never contacted by the update itself."""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(*args, **kwargs):
    return subprocess.run([str(a) for a in args], check=True, **kwargs)


def current_version() -> str:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    return re.search(r'^version\s*=\s*"([^"]+)"', text, re.M).group(1)


def next_patch(version: str) -> str:
    major, minor, patch = (int(n) for n in re.match(r"(\d+)\.(\d+)\.(\d+)", version).groups())
    return f"{major}.{minor}.{patch + 1}"


def build_wheel(version: str, wheels: Path, work: Path) -> None:
    """A wheel of this checkout labelled ``version``, built from a copy so
    the checkout gets no build/ or egg-info clutter."""
    copy = work / f"build-{version}"
    shutil.copytree(ROOT / "src", copy / "src",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info"))
    for name in ("README.md", "LICENSE"):
        if (ROOT / name).exists():
            shutil.copy2(ROOT / name, copy / name)
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    text = re.sub(r'^version\s*=\s*"[^"]+"', f'version = "{version}"', text, count=1, flags=re.M)
    (copy / "pyproject.toml").write_text(text, encoding="utf-8")
    run(sys.executable, "-m", "pip", "wheel", "--no-deps", "-q", "-w", wheels, copy)


def venv_python(env_dir: Path) -> Path:
    return env_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def installed_version(python: Path) -> str:
    done = subprocess.run(
        [str(python), "-c", "import importlib.metadata as m; print(m.version('opennmr'))"],
        capture_output=True, text=True)
    return done.stdout.strip()


HEADLESS = r'''
import subprocess, sys
from opennmr import update, update_install

print("running", update.__version__, "installed as", update.install_kind())
result = update.check("pypi")
print("check:", result.status, result.latest_version)
assert result.status == "newer", "the rehearsal's JSON should offer a newer version"
blocker = update_install.upgrade_blocker()
assert not blocker, blocker
gone = subprocess.Popen([sys.executable, "-c", "pass"]); gone.wait()   # an app that has exited
plan = update_install.plan_pip_upgrade(result.info.version, pid=gone.pid, relaunch=[], settle=0.0)
done = subprocess.run(plan.argv, capture_output=True, text=True)
assert done.stdout == "" and done.stderr == "", "the helper must print nothing"
print("helper finished")
'''


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--no-gui", action="store_true", help="drive the update without a window")
    parser.add_argument("--keep", action="store_true", help="keep the temporary folder")
    parser.add_argument("--wait", type=float, default=300.0, help="seconds to wait for the upgrade")
    args = parser.parse_args()

    work = Path(tempfile.mkdtemp(prefix="opennmr-rehearsal-"))
    old = current_version()
    new = next_patch(old)
    wheels, state = work / "wheels", work / "state"
    wheels.mkdir()
    ok = False
    try:
        print(f"Building openNMR {old} and {new} in {work} ...")
        build_wheel(old, wheels, work)
        build_wheel(new, wheels, work)
        print("Creating a virtual environment and installing", old, "...")
        run(sys.executable, "-m", "venv", work / "env")
        python = venv_python(work / "env")
        run(python, "-m", "pip", "install", "-q", "--disable-pip-version-check",
            next(wheels.glob(f"opennmr-{old}-*.whl")))
        json_path = work / "opennmr.json"
        files = [{"filename": "opennmr.whl", "requires_python": ">=3.12", "yanked": False}]
        json_path.write_text(json.dumps({"info": {"version": new},
                                         "releases": {old: files, new: files}}), encoding="utf-8")
        env = dict(os.environ,
                   OPENNMR_PYPI_JSON=json_path.as_uri(),
                   OPENNMR_PIP_ARGS=f'--no-index --find-links "{wheels}"',
                   OPENNMR_STATE_DIR=str(state))
        if args.no_gui:
            env.setdefault("QT_QPA_PLATFORM", "offscreen")
            run(python, "-c", HEADLESS, env=env)
        else:
            env.pop("QT_QPA_PLATFORM", None)
            print(f"\nOpening openNMR {old}. In Settings > Updates press 'Check for Updates', "
                  f"then 'Update to {new} and restart'.\n")
            subprocess.run([str(python), "-m", "opennmr.gui.app"], env=env)
            deadline = time.time() + args.wait
            while not (state / "update_result.json").exists() and time.time() < deadline:
                time.sleep(2)             # the helper writes this last, after checking the new version
            time.sleep(3)
        found = installed_version(python)
        log = state / "update.log"
        print("\nInstalled version now:", found)
        if log.exists():
            print("--- update.log ---\n" + log.read_text(encoding="utf-8").strip())
        ok = found == new
        print("\nRESULT:", "the update worked" if ok else f"still {found}: the update did NOT complete")
    finally:
        if args.keep:
            print("Kept", work)
        elif args.no_gui or not ok:
            shutil.rmtree(work, ignore_errors=True)
        else:
            print("The reopened openNMR still uses this folder; delete it once closed:", work)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
