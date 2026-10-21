"""Regression test: the macOS .app layout (symlinks into Frameworks) must
not crash the Qt-plugin re-sync in packaging/build.py (0.3.0 CI failure)."""
import importlib.util
import os
import sys

import pytest

_BUILD = os.path.join(os.path.dirname(__file__), "..", "packaging", "build.py")


def _load():
    spec = importlib.util.spec_from_file_location("opennmr_build", _BUILD)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _touch(path, text=""):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


def test_plain_onedir_layout(tmp_path):
    b = _load()
    src = tmp_path / "src"; _touch(str(src / "platforms" / "new.dylib"))
    plug = tmp_path / "dist" / "opennmr" / "_internal" / "PySide6" / "Qt" / "plugins"
    _touch(str(plug / "platforms" / "old.dylib"))
    out = b.sync_qt_plugins(str(tmp_path / "dist"), str(src))
    assert len(out) == 1
    assert (plug / "platforms" / "new.dylib").exists()
    assert not (plug / "platforms" / "old.dylib").exists()


@pytest.mark.skipif(sys.platform.startswith("win"), reason="needs symlinks")
def test_symlinked_app_layout(tmp_path):
    b = _load()
    src = tmp_path / "src"; _touch(str(src / "platforms" / "new.dylib"))
    app = tmp_path / "dist" / "opennmr.app" / "Contents"
    real = app / "Frameworks" / "PySide6" / "Qt" / "plugins"
    _touch(str(real / "platforms" / "old.dylib"))
    link = app / "Resources" / "PySide6" / "Qt" / "plugins"
    os.makedirs(str(link.parent))
    os.symlink(os.path.relpath(real, link.parent), str(link))
    out = b.sync_qt_plugins(str(tmp_path / "dist"), str(src))
    assert out == [os.path.realpath(real)]      # synced once, via the target
    assert link.is_symlink()                     # link preserved
    assert (link / "platforms" / "new.dylib").exists()
    assert not (real / "platforms" / "old.dylib").exists()


@pytest.mark.skipif(sys.platform.startswith("win"), reason="needs symlinks")
def test_dangling_link_skipped(tmp_path):
    b = _load()
    src = tmp_path / "src"; _touch(str(src / "x"))
    link = tmp_path / "dist" / "PySide6" / "Qt" / "plugins"
    os.makedirs(str(link.parent))
    os.symlink("nowhere", str(link))
    assert b.sync_qt_plugins(str(tmp_path / "dist"), str(src)) == []
