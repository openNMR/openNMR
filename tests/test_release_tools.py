"""packaging/release_tools.py: the manual release route (zip / build / publish), without building anything."""

import hashlib
import importlib.util
import os
import subprocess
import sys
import zipfile

import pytest

PACKAGING = os.path.join(os.path.dirname(__file__), "..", "packaging")
_spec = importlib.util.spec_from_file_location("opennmr_release_tools",
                                               os.path.join(PACKAGING, "release_tools.py"))
tools = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = tools                     # dataclasses look their module up here
_spec.loader.exec_module(tools)

VERSION = "0.8.3"


def touch(directory, name, data=b"x"):
    path = os.path.join(directory, name)
    with open(path, "wb") as handle:
        handle.write(data)
    return path


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "pyproject.toml").write_text(f'[project]\nname = "opennmr"\nversion = "{VERSION}"\n')
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "module.py").write_text("x = 1\n")
    (tmp_path / "release-0.8.3").mkdir()                       # untracked work folder
    (tmp_path / ".gitignore").write_text("/release-*/\n")
    for cmd in (["init", "-q"], ["config", "user.email", "t@t"], ["config", "user.name", "t"],
                ["add", "."], ["commit", "-qm", "start"], ["tag", f"v{VERSION}"]):
        subprocess.run(["git", *cmd], cwd=tmp_path, check=True)
    return str(tmp_path)


def test_platform_key():
    assert tools.platform_key("win32") == "windows"
    assert tools.platform_key("darwin") == "macos"
    assert tools.platform_key("linux") == "linux"
    assert tools.platform_key() in tools.PLATFORMS


def test_installer_names_match_what_the_in_app_updater_looks_for():
    """The updater picks assets by pattern; the names built here must satisfy it."""
    import re
    patterns = {"windows": r"windows.*setup.*\.exe$", "macos": r"macos.*\.dmg$",
                "linux": r"linux.*\.appimage$"}
    examples = {"windows": "openNMR-0.8.3-windows-setup.exe", "macos": "openNMR-0.8.3-macos-arm64.dmg",
                "linux": "openNMR-0.8.3-linux-x86_64.AppImage"}
    for platform, name in examples.items():
        assert re.search(patterns[platform], name, flags=re.IGNORECASE)
        import fnmatch
        assert fnmatch.fnmatch(name, tools.installer_pattern(platform, "0.8.3"))


def test_find_installers_ignores_other_versions_and_reports_missing(tmp_path):
    touch(tmp_path, "openNMR-0.8.3-windows-setup.exe")
    touch(tmp_path, "openNMR-0.8.2-macos-arm64.dmg")             # an older release's file
    touch(tmp_path, "openNMR-0.8.3-linux-x86_64.AppImage")
    found = tools.find_installers(str(tmp_path), "0.8.3")
    assert found["windows"].endswith("windows-setup.exe") and found["linux"].endswith(".AppImage")
    assert found["macos"] is None
    assert tools.missing_installers(found) == ["macos"]


def test_frozen_command_per_platform():
    assert tools.frozen_command("windows", "dist").endswith(os.path.join("openNMR", "openNMR.exe"))
    assert tools.frozen_command("macos", "dist").endswith(os.path.join("Contents", "MacOS", "openNMR"))
    assert tools.frozen_command("linux", "dist").endswith(os.path.join("openNMR", "openNMR"))


def test_checksums_use_sha256sum_format_with_unix_line_ends(tmp_path):
    a = touch(tmp_path, "b.bin", b"hello")
    b = touch(tmp_path, "a.bin", b"world")
    dest = str(tmp_path / "SHA256SUMS")
    text = tools.write_checksums([a, b], dest)
    assert text == (f"{hashlib.sha256(b'world').hexdigest()}  a.bin\n"
                    f"{hashlib.sha256(b'hello').hexdigest()}  b.bin\n")       # sorted by name
    assert open(dest, "rb").read() == text.encode() and b"\r" not in open(dest, "rb").read()
    # ...and the updater's own parser reads it back.
    from opennmr import update_install
    assert update_install.parse_checksums(text, "a.bin") == hashlib.sha256(b"world").hexdigest()


def test_the_build_plan_is_ordered_and_platform_specific():
    win = tools.plan_build("windows", "py")
    names = [s.name for s in win]
    assert names[:3] == ["install the build tools (editable, so the version is current)",
                         "build the app", "build the installer"]
    assert win[0].argv == ["py", "-m", "pip", "install", "-e", ".[build]"]
    assert not any("codesign" in s.argv for s in win) and not any(s.env for s in win)
    assert win[4].argv[0].endswith("openNMR.exe") and "--self-test" in win[4].argv

    mac = tools.plan_build("macos", "py")
    assert any(s.argv[0] == "codesign" for s in mac)
    assert [s.argv[0] for s in mac].index("codesign") > [s.name for s in mac].index("build the installer")

    lin = tools.plan_build("linux", "py")
    assert all(s.env.get("QT_QPA_PLATFORM") == "offscreen" for s in lin[3:])
    appimage = [s for s in lin if "{installer}" in s.argv]
    assert len(appimage) == 1 and "--appimage-extract-and-run" in appimage[0].argv
    assert lin[-1].argv[-2:] == ["st-src", "st-appimage"]


def test_plan_zip_archives_only_the_tag_under_a_prefix():
    argv, target = tools.plan_zip(VERSION, "out")
    assert argv[:3] == ["git", "archive", "--format=zip"] and argv[-1] == f"v{VERSION}"
    assert f"--prefix=openNMR-{VERSION}/" in argv and target.endswith(f"openNMR-{VERSION}-src.zip")


def test_zip_command_makes_a_clean_source_zip(repo, tmp_path):
    out = str(tmp_path / "out")
    args = tools.argparse.Namespace(out=out, tag=None, dry_run=False)
    assert tools.cmd_zip(args, repo) == 0
    names = zipfile.ZipFile(os.path.join(out, f"openNMR-{VERSION}-src.zip")).namelist()
    assert f"openNMR-{VERSION}/pyproject.toml" in names and f"openNMR-{VERSION}/src/module.py" in names
    assert not any(".git/" in n or "release-" in n for n in names)


def test_zip_command_refuses_without_the_tag(repo):
    subprocess.run(["git", "tag", "-d", f"v{VERSION}"], cwd=repo, check=True)
    args = tools.argparse.Namespace(out=None, tag=None, dry_run=False)
    assert tools.cmd_zip(args, repo) == 1


def test_gh_release_command_creates_or_adds():
    create = tools.gh_release_argv(VERSION, "o/r", ["a", "SHA256SUMS"], "notes", exists=False)
    assert create[:4] == ["gh", "release", "create", f"v{VERSION}"] and "--verify-tag" in create
    assert create[create.index("--notes") + 1] == "notes" and "a" in create
    upload = tools.gh_release_argv(VERSION, "o/r", ["a"], "", exists=True)
    assert upload[:3] == ["gh", "release", "upload"] and "--clobber" in upload


def test_run_steps_stops_at_the_first_failure_and_substitutes_the_installer(capsys):
    seen = []

    class Result:
        def __init__(self, code):
            self.returncode = code

    def runner(argv, **kwargs):
        seen.append(argv)
        return Result(1 if argv[0] == "bad" else 0)

    steps = [tools.Step("one", ["{installer}", "--x"]), tools.Step("two", ["bad"]),
             tools.Step("three", ["never"])]
    assert tools.run_steps(steps, ".", installer="/p/app", runner=runner) == 1
    assert seen == [["/p/app", "--x"], ["bad"]]
    assert "FAILED at step 2" in capsys.readouterr().out


def test_dry_run_runs_nothing(capsys):
    def runner(*a, **k):
        raise AssertionError("must not run")
    assert tools.run_steps(tools.plan_build("windows", "py"), ".", dry_run=True, runner=runner) == 0
    assert "build the installer" in capsys.readouterr().out


def test_publish_needs_installers_and_all_three_unless_allowed(repo, capsys):
    out = os.path.join(repo, f"release-{VERSION}")
    args = tools.argparse.Namespace(out_dir=None, repo="o/r", notes="", notes_file=None,
                                    allow_missing=False, pypi=False, yes=False, dry_run=True)
    assert tools.cmd_publish(args, repo) == 1                    # nothing there at all
    touch(out, f"openNMR-{VERSION}-windows-setup.exe")
    assert tools.cmd_publish(args, repo) == 1                    # two missing
    assert "Missing" in capsys.readouterr().out
    args.allow_missing = True
    assert tools.cmd_publish(args, repo) == 0
    assert not os.path.exists(os.path.join(out, "SHA256SUMS"))   # dry run writes nothing


def test_the_appimage_comparison_comes_after_the_appimage_self_test(capsys):
    """The build splits its plan at the first step needing the installer; everything that reads
    ``st-appimage`` must be in the second half."""
    steps = tools.plan_build("linux", "py")
    split = next(i for i, s in enumerate(steps) if "{installer}" in " ".join(s.argv))
    assert not any("st-appimage" in " ".join(s.argv) for s in steps[:split])
    assert all("st-appimage" in " ".join(s.argv) for s in steps[split:])
