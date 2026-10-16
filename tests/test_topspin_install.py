"""The TopSpin plugin installer: finding TopSpin, filling in the plugin,
and editing autostart.mac without disturbing the user's own lines.
The plugin itself runs in TopSpin's Jython and is not exercised here."""

import ast
from pathlib import Path

from opennmr.topspin import install


def _fake_topspin(root: Path, name: str) -> Path:
    home = root / name
    (home / "exp" / "stan" / "nmr" / "py" / "user").mkdir(parents=True)
    (home / "prog").mkdir()
    return home


def test_version_from_name():
    assert install.version_from_name("topspin4.5.0") == (4, 5, 0)
    assert install.version_from_name("TopSpin 5.0.0") == (5, 0, 0)
    assert install.version_from_name("TopSpin3.6.5") == (3, 6, 5)
    assert install.version_from_name("topspin3.5pl7") == (3, 5)  # patch level
    assert install.version_from_name("nmr") == ()


def test_find_newest_first_and_skip_non_topspin(tmp_path, monkeypatch):
    old = _fake_topspin(tmp_path, "topspin4.5.0")
    new = _fake_topspin(tmp_path, "topspin5.0.0")
    (tmp_path / "topspin-notes").mkdir()  # named like one, but is not
    candidates = [old, new, tmp_path / "topspin-notes", new]  # duplicate too
    monkeypatch.setattr(install, "_candidate_dirs", lambda: iter(map(str, candidates)))

    found = install.find_topspin_installs()

    assert [t.version for t in found] == [(5, 0, 0), (4, 5, 0)]
    assert found[0].plugin == new.resolve() / "exp/stan/nmr/py/user/opennmr.py"


def test_rendered_plugin_holds_paths_exactly(tmp_path):
    odd = r"C:\Users\José O'Brien\miniforge3\pythonw.exe"
    text = install.render_plugin([odd, "-m", "opennmr.gui.app"], ["/a b/bin"],
                                 Path("/x/logo.png"))

    # Parse only the settings block: the rest imports Java classes.
    block = text.split("# BEGIN INSTALLER SETTINGS")[1].split("# END INSTALLER SETTINGS")[0]
    settings = {}
    for node in ast.parse(block).body:
        settings[node.targets[0].id] = ast.literal_eval(node.value)
    assert settings == {"COMMAND": [odd, "-m", "opennmr.gui.app"],
                        "PATH_PREPEND": ["/a b/bin"], "ICON": str(Path("/x/logo.png"))}
    assert install.PLUGIN_MARK in text
    assert text.isascii()  # safe for Jython whatever the file encoding


def test_launch_settings_use_this_python():
    import sys

    command, path_prepend = install.launch_settings()
    assert Path(command[0]).parent == Path(sys.executable).absolute().parent
    assert command[1:] == ["-m", "opennmr.gui.app"]
    assert all(Path(folder).is_dir() for folder in path_prepend)


def test_install_writes_plugin(tmp_path):
    home = _fake_topspin(tmp_path, "topspin5.0.0")
    topspin = install.TopSpin(home, (5, 0, 0))
    assert not topspin.has_plugin()

    install._write_text(topspin.plugin, install.render_plugin(["/py"], [], install.ICON))

    assert topspin.has_plugin()
    assert not list(topspin.plugin.parent.glob(".opennmr.py*"))  # no temp files left


def test_autostart_adds_once_and_removes_cleanly(tmp_path):
    macro = tmp_path / "prop" / "autostart.mac"
    macro.parent.mkdir()
    original = "# user comment\r\nre exam1d_13C 1 1 c:/bruker/topspin guest\r\n"
    macro.write_bytes(original.encode())

    assert install.set_autostart(macro, True)
    assert not install.set_autostart(macro, True)  # already there
    text = macro.read_bytes().decode()
    assert text.count("\nopennmr\r\n") == 1
    assert "\n" not in text.replace("\r\n", "")  # kept the file's CRLF
    assert install.has_autostart(macro)

    assert install.set_autostart(macro, False)
    assert macro.read_bytes().decode() == original


def test_autostart_keeps_other_blocks(tmp_path):
    macro = tmp_path / "autostart.mac"
    macro.write_text("# BEGIN TOPSPINTOOLS AUTOSTART\ntopspintools\n"
                     "# END TOPSPINTOOLS AUTOSTART\n")
    install.set_autostart(macro, True)
    install.set_autostart(macro, False)
    assert macro.read_text() == ("# BEGIN TOPSPINTOOLS AUTOSTART\ntopspintools\n"
                                 "# END TOPSPINTOOLS AUTOSTART\n")


def test_autostart_not_created_when_not_needed(tmp_path):
    macro = tmp_path / "autostart.mac"
    assert not install.set_autostart(macro, False)
    assert not macro.exists()
