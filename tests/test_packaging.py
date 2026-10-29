"""Packaging: the installer build (packaging/build.py, make_installer.py),
the frozen-app self-test (opennmr.selftest) and its comparison with a
source run (packaging/compare_selftest.py). Explicit user request,
2026-10-15: installable apps for Windows, macOS and Linux, checked to
behave and look the same as the scripts.

Building with PyInstaller takes minutes and is done by
.github/workflows/build.yml; these tests check everything around it."""
import importlib.util
import json
import os
import re
import shutil
import sys

import pytest

from opennmr import update

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACKAGING = os.path.join(REPO, "packaging")


def _load(name):
    sys.path.insert(0, PACKAGING)
    try:
        spec = importlib.util.spec_from_file_location(name, os.path.join(PACKAGING, f"{name}.py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(PACKAGING)


def _pyproject_version():
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib
    with open(os.path.join(REPO, "pyproject.toml"), "rb") as handle:
        return tomllib.load(handle)["project"]["version"]


def test_version_is_stated_once_and_agrees():
    build = _load("build")
    from opennmr import _version
    assert build.app_version() == _pyproject_version()
    # The fallback is what a frozen app without metadata would report:
    # it must not drift from pyproject.toml.
    assert _version._FALLBACK_VERSION == _pyproject_version()


def test_build_passes_what_a_frozen_app_needs(monkeypatch):
    build = _load("build")
    pyinstaller_main = pytest.importorskip("PyInstaller.__main__")
    captured = []
    monkeypatch.setattr(pyinstaller_main, "run", captured.append)
    monkeypatch.setattr(build, "_fix_qt_plugins", lambda: None)
    monkeypatch.setattr(build, "_stamp_macos_version", lambda: None)
    monkeypatch.setattr(build, "_make_icon", lambda: None)
    assert build.main() == 0
    args = captured[0]

    def value_after(flag):
        return [args[i + 1] for i, a in enumerate(args) if a == flag]

    assert value_after("--name") == ["openNMR"]
    assert "opennmr" in value_after("--copy-metadata")
    assert "opennmr" in value_after("--collect-submodules")
    assert "pymatgen" in value_after("--collect-data")
    data = value_after("--add-data")
    assert any(d.endswith("opennmr/gui/resources") for d in data)
    assert any(d.endswith("opennmr/topspin/resources") for d in data)
    assert "--windowed" in args
    assert args[0].endswith("entrypoint.py")
    # Vector export (SVG/PDF/EPS) imports its matplotlib backend by name when saving,
    # which PyInstaller cannot see: the frozen app must be told (2026-10-29 report).
    hidden = value_after("--hidden-import")
    for backend in ("matplotlib.backends.backend_svg", "matplotlib.backends.backend_pdf",
                    "matplotlib.backends.backend_ps"):
        assert backend in hidden


def test_every_vector_backend_the_build_bundles_exists_and_saves(tmp_path):
    """The names in build.MATPLOTLIB_EXPORT_BACKENDS are real modules (a typo would
    only show up as a missing backend in a frozen app) and each writes its format."""
    import importlib

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    build = _load("build")
    for name in build.MATPLOTLIB_EXPORT_BACKENDS:
        importlib.import_module(name)
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    for extension in ("svg", "pdf", "eps", "png"):
        path = tmp_path / f"figure.{extension}"
        fig.savefig(path)
        assert path.stat().st_size > 500
    plt.close(fig)


@pytest.mark.parametrize("platform, name", [
    ("windows", "openNMR-{v}-windows-setup.exe"),
    ("darwin", "openNMR-{v}-macos-arm64.dmg"),
    ("darwin", "openNMR-{v}-macos-x86_64.dmg"),
    ("linux", "openNMR-{v}-linux-x86_64.AppImage"),
])
def test_installer_names_are_what_the_update_check_looks_for(platform, name):
    """make_installer.py names each installer; opennmr.update finds it in a
    release by that name. If either changes alone, the Download button
    silently falls back to the release page."""
    name = name.format(v="1.2.3")
    source = open(os.path.join(PACKAGING, "make_installer.py"), encoding="utf-8").read()
    stem = re.sub(r"\d+\.\d+\.\d+", "{version}", name)
    stem = stem.replace("arm64", "{_arch()}").replace("x86_64", "{_arch()}")
    if platform == "windows":
        iss = open(os.path.join(PACKAGING, "windows", "openNMR.iss"), encoding="utf-8").read()
        assert "OutputBaseFilename=openNMR-{#AppVersion}-windows-setup" in iss
        assert "openNMR-{version}-windows-setup.exe" in source
    else:
        assert stem in source
    others = [{"name": "openNMR-1.2.3-source.tar.gz", "browser_download_url": "x"}]
    asset = update.pick_asset(others + [{"name": name, "browser_download_url": "u"}], platform)
    assert asset is not None and asset["name"] == name


def test_windows_installer_keeps_its_app_id():
    # A changed AppId installs the next version *beside* the old one.
    iss = open(os.path.join(PACKAGING, "windows", "openNMR.iss"), encoding="utf-8").read()
    assert "AppId={{8F3C5B1E-6A2D-4C7E-9B41-2E5D7A0C9F13}" in iss
    assert "PrivilegesRequired=lowest" in iss   # no administrator needed


def test_entrypoint_runs_self_test_and_freeze_support():
    text = open(os.path.join(PACKAGING, "entrypoint.py"), encoding="utf-8").read()
    assert "multiprocessing.freeze_support()" in text
    assert '"--self-test"' in text


@pytest.fixture
def isolated_env(monkeypatch):
    """selftest.run sets these for its own process; give them back."""
    for key in ("HOME", "USERPROFILE", "OPENNMR_SIMULATIONS_DIR", "OPENNMR_UPDATE_CHECK",
                "OPENNMR_BACKGROUND_LOAD", "QT_QPA_PLATFORM"):
        if key in os.environ:
            monkeypatch.setenv(key, os.environ[key])
        else:
            monkeypatch.delenv(key, raising=False)


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_selftest_passes_from_source(tmp_path, isolated_env, qapp):
    from opennmr import selftest
    out = tmp_path / "run"
    assert selftest.main([str(out)]) == 0
    report = json.load(open(out / "report.json", encoding="utf-8"))
    assert report["ok"] and not report["frozen"]
    assert set(report["steps"]) == {"imports", "version", "read_process", "simulate_and_fit",
                                    "cif_structure", "tls", "figures", "gui"}
    for name in ("figure_1d.png", "figure_2d.png", "figure_csa.png", "gui_main_window.png"):
        assert (out / name).stat().st_size > 1000
    numbers = json.load(open(out / "numbers.json", encoding="utf-8"))
    assert numbers["lorentz_fit"] == pytest.approx([1.234, 0.8, 5.0])
    assert (out / "home").is_dir()


def _fake_run(folder, frozen, numbers, colour=(255, 255, 255)):
    from PIL import Image
    os.makedirs(folder, exist_ok=True)
    steps = {"gui": {"ok": True, "seconds": 0.1}}
    json.dump({"ok": True, "frozen": frozen, "platform": "x", "python": "3", "steps": steps},
              open(os.path.join(folder, "report.json"), "w"))
    json.dump(numbers, open(os.path.join(folder, "numbers.json"), "w"))
    for name in ("figure_1d.png", "figure_2d.png", "figure_csa.png", "gui_main_window.png"):
        Image.new("RGB", (20, 10), (255, 255, 255)).save(os.path.join(folder, name))
    image = Image.new("RGB", (20, 10), (255, 255, 255))
    image.putpixel((3, 3), colour)
    image.save(os.path.join(folder, "figure_1d.png"))


def test_compare_selftest_passes_identical_and_catches_differences(tmp_path, capsys):
    compare = _load("compare_selftest")
    src, app = str(tmp_path / "src"), str(tmp_path / "app")
    _fake_run(src, False, {"a": [1.0, 2.0]})
    _fake_run(app, True, {"a": [1.0, 2.0]})
    assert compare.main([src, app]) == 0

    _fake_run(app, True, {"a": [1.0, 2.1]})
    assert compare.main([src, app]) == 1
    assert "'a' differs" in capsys.readouterr().out

    _fake_run(app, True, {"a": [1.0, 2.0]}, colour=(0, 0, 0))
    assert compare.main([src, app]) == 1
    assert os.path.exists(os.path.join(app, "diff_figure_1d.png"))
    # 1 pixel in 200 is 0.5 %: allowed by a 1 % tolerance.
    assert compare.main([src, app, "--tolerance", "0.01"]) == 0

    _fake_run(app, False, {"a": [1.0, 2.0]})       # not a frozen build
    assert compare.main([src, app]) == 1


def test_make_installer_refuses_without_a_build(tmp_path, monkeypatch):
    mi = _load("make_installer")
    monkeypatch.setattr(mi, "DIST_DIR", str(tmp_path))
    for fn in (mi.windows_installer, mi.macos_dmg, mi.linux_appimage):
        with pytest.raises(SystemExit, match="build.py first"):
            fn("1.2.3")


@pytest.mark.skipif(not sys.platform.startswith("linux") or shutil.which("true") is None,
                    reason="AppDir layout check is Linux-only")
def test_linux_appdir_layout(tmp_path, monkeypatch):
    mi = _load("make_installer")
    dist = tmp_path / "dist"
    app = dist / "openNMR"
    app.mkdir(parents=True)
    (app / "openNMR").write_text("#!/bin/sh\n")
    monkeypatch.setattr(mi, "DIST_DIR", str(dist))
    monkeypatch.setattr(mi, "BUILD_DIR", str(tmp_path / "build"))
    calls = []
    monkeypatch.setattr(mi.subprocess, "run", lambda cmd, **k: calls.append(cmd))
    monkeypatch.setenv("APPIMAGETOOL", "/bin/true")
    out = mi.linux_appimage("1.2.3")
    appdir = tmp_path / "build" / "openNMR.AppDir"
    assert (appdir / "AppRun").stat().st_mode & 0o111
    assert (appdir / "usr" / "lib" / "openNMR" / "openNMR").exists()
    assert (appdir / "openNMR.png").exists()
    desktop = (appdir / "openNMR.desktop").read_text()
    assert "Exec=openNMR" in desktop and "Icon=openNMR" in desktop
    assert out.endswith("openNMR-1.2.3-linux-" + mi._arch() + ".AppImage")
    assert calls and calls[0][-1] == out


def test_macos_bundle_is_signed_last(monkeypatch):
    """Info.plist is rewritten after PyInstaller signs the app; signing before
    that left 'invalid Info.plist' and macOS called the download 'damaged'."""
    build = _load("build")
    pyinstaller_main = pytest.importorskip("PyInstaller.__main__")
    calls = []
    monkeypatch.setattr(pyinstaller_main, "run", lambda args: calls.append("pyinstaller"))
    monkeypatch.setattr(build, "_make_icon", lambda: None)
    monkeypatch.setattr(build, "_fix_qt_plugins", lambda: calls.append("plugins"))
    monkeypatch.setattr(build, "_stamp_macos_version", lambda: calls.append("plist"))
    monkeypatch.setattr(build, "_resign_macos_app", lambda: calls.append("sign"))
    monkeypatch.setattr(build.sys, "platform", "darwin")
    assert build.main() == 0
    assert calls == ["pyinstaller", "plugins", "plist", "sign"]
    calls.clear()
    monkeypatch.setattr(build.sys, "platform", "linux")
    build.main()
    assert "sign" not in calls


def test_appdir_text_files_have_unix_line_endings(tmp_path, monkeypatch):
    """A CRLF AppRun fails on Linux and a CRLF .desktop is rejected by appimagetool."""
    mi = _load("make_installer")
    src = tmp_path / "crlf.txt"
    src.write_bytes(b"#!/bin/sh\r\nexec x\r\n")
    dst = tmp_path / "out"
    mi._copy_text_lf(str(src), str(dst))
    assert dst.read_bytes() == b"#!/bin/sh\nexec x\n"
    desktop = open(os.path.join(PACKAGING, "linux", "openNMR.desktop"), "rb").read()
    assert b"Categories=Science;Chemistry;\n" in desktop.replace(b"\r\n", b"\n")


def test_dmg_is_hfs_plus_and_retried_when_busy(tmp_path, monkeypatch):
    """2026-10-27: "hdiutil: create failed - Resource busy" on a friend's
    Mac. The image is made as HFS+, a stale volume is ejected first, and a
    failed create is tried again (the partial image removed) before giving
    up."""
    import subprocess
    mi = _load("make_installer")
    dist = tmp_path / "dist"
    (dist / "openNMR.app" / "Contents").mkdir(parents=True)
    monkeypatch.setattr(mi, "DIST_DIR", str(dist))
    monkeypatch.setattr(mi, "BUILD_DIR", str(tmp_path / "build"))
    calls = []
    fails = {"left": 2}

    def fake_run(command, check=False, **_kw):
        calls.append(list(command))
        if command[:2] == ["hdiutil", "create"]:
            out = command[-1]
            open(out, "w").close()                       # a partial image
            if fails["left"]:
                fails["left"] -= 1
                raise subprocess.CalledProcessError(1, command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(mi.subprocess, "run", fake_run)
    monkeypatch.setattr(mi, "DMG_PAUSE_S", 0.0)
    out = mi.macos_dmg("1.2.3")
    creates = [c for c in calls if c[:2] == ["hdiutil", "create"]]
    assert len(creates) == 3
    assert creates[0][creates[0].index("-fs") + 1] == "HFS+"
    assert creates[0][creates[0].index("-volname") + 1] == "openNMR 1.2.3"
    assert out.endswith("openNMR-1.2.3-macos-" + mi._arch() + ".dmg")
    # Gives up after DMG_ATTEMPTS, with the last error.
    calls.clear()
    fails["left"] = 99
    with pytest.raises(subprocess.CalledProcessError):
        mi.macos_dmg("1.2.3")
    assert len([c for c in calls if c[:2] == ["hdiutil", "create"]]) == mi.DMG_ATTEMPTS
    # A partial image is removed before each new try.
    removed = []
    monkeypatch.setattr(mi.os, "remove", lambda p: removed.append(p))
    fails["left"] = 1
    mi.macos_dmg("1.2.3")
    assert removed == [out]
