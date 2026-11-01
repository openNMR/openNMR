"""Settings and other small JSON files are replaced atomically
(opennmr.core.atomic_write; stability audit / explicit user request,
2026-11-01): a failure part-way leaves the old file, never half a file."""

import json
import os

import numpy as np
import pytest

from opennmr.core import atomic_write as aw
from opennmr.core.atomic_write import write_json_atomic, write_text_atomic

from test_gui import qapp  # noqa: F401 -- fixture


def _leftovers(folder):
    return [n for n in os.listdir(folder) if n.endswith(".tmp")]


def test_writes_json_and_creates_the_folder(tmp_path):
    path = tmp_path / "deep" / "settings.json"
    out = write_json_atomic(path, {"b": 1, "a": [1, 2]}, indent=2, sort_keys=True)
    assert out == str(path)
    assert json.loads(path.read_text(encoding="utf-8")) == {"a": [1, 2], "b": 1}
    assert _leftovers(path.parent) == []


def test_an_unencodable_value_leaves_the_old_file_untouched(tmp_path):
    path = tmp_path / "ui_settings.json"
    write_json_atomic(path, {"ui_font_size": 10})
    before = path.read_text(encoding="utf-8")
    with pytest.raises(TypeError):
        write_json_atomic(path, {"ui_font_size": 11, "bad": object()})
    assert path.read_text(encoding="utf-8") == before
    assert _leftovers(tmp_path) == []


def test_in_place_json_dump_would_have_truncated(tmp_path):
    """What the old code did: json.dump streams, so a bad value mid-way
    leaves a truncated file -- the reason for this module."""
    path = tmp_path / "old_way.json"
    path.write_text('{"ui_font_size": 10}', encoding="utf-8")
    with pytest.raises(TypeError):
        with open(path, "w", encoding="utf-8") as handle:
            json.dump({"a": 1, "z": np.float32(2)}, handle)
    with pytest.raises(json.JSONDecodeError):
        json.loads(path.read_text(encoding="utf-8"))


def test_a_failed_rename_keeps_the_original_and_cleans_up(tmp_path, monkeypatch):
    path = tmp_path / "figure_styles.json"
    write_json_atomic(path, {"v": 1})
    monkeypatch.setattr(aw, "RETRY_SECONDS", 0.1)

    def locked(src, dst):
        raise PermissionError("locked by another program")
    monkeypatch.setattr(aw.os, "replace", locked)
    with pytest.raises(PermissionError):
        write_json_atomic(path, {"v": 2})
    monkeypatch.undo()
    assert json.loads(path.read_text(encoding="utf-8")) == {"v": 1}
    assert _leftovers(tmp_path) == []


def test_a_briefly_locked_target_is_retried(tmp_path, monkeypatch):
    """Windows: OneDrive or an antivirus scanner can hold the file for a moment."""
    path = tmp_path / "plot_style.json"
    write_json_atomic(path, {"v": 1})
    real, calls = os.replace, []

    def flaky(src, dst):
        calls.append(1)
        if len(calls) < 3:
            raise PermissionError("in use")
        real(src, dst)
    monkeypatch.setattr(aw.os, "replace", flaky)
    write_json_atomic(path, {"v": 2})
    assert len(calls) == 3
    assert json.loads(path.read_text(encoding="utf-8")) == {"v": 2}


def test_text_and_trailing_newline(tmp_path):
    p = write_text_atomic(tmp_path / "a.txt", "x\ny")
    assert open(p, encoding="utf-8").read() == "x\ny"
    write_json_atomic(tmp_path / "b.json", [1], trailing_newline=True)
    assert (tmp_path / "b.json").read_text(encoding="utf-8") == "[1]\n"


def test_expands_the_home_folder(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    out = write_json_atomic("~/.opennmr/x.json", {"ok": True})
    assert out == str(tmp_path / ".opennmr" / "x.json")


# ---- the callers -------------------------------------------------------------------------

def test_dft_calibration_library_uses_it(tmp_path):
    from opennmr.notebook import dft_calibration as dc
    import inspect
    src = inspect.getsource(dc.save_calibration) + inspect.getsource(dc.delete_calibration)
    assert "write_json_atomic" in src and 'open(path, "w"' not in src


def test_ui_settings_survive_an_unencodable_value(qapp, tmp_path, monkeypatch):  # noqa: F811
    from opennmr.gui.main_window import MainWindow
    path = tmp_path / "ui_settings.json"
    monkeypatch.setattr(MainWindow, "_UI_SETTINGS_PATH", str(path))
    window = MainWindow()
    try:
        window._ui_settings["ui_font_size"] = 12
        window._save_ui_settings()
        good = path.read_text(encoding="utf-8")
        window._ui_settings["oops"] = object()            # a value json cannot encode
        window._save_ui_settings()                        # reported, not raised
        assert path.read_text(encoding="utf-8") == good
        assert "Could not save settings" in window.statusBar().currentMessage()
    finally:
        window._ui_settings.pop("oops", None)
        window.close()


def test_every_settings_writer_in_the_window_is_atomic():
    import inspect
    from opennmr.gui import main_window
    src = inspect.getsource(main_window)
    for name in ("_PLOT_STYLE_CONFIG_PATH", "_FIGURE_STYLES_PATH", "_UI_SETTINGS_PATH",
                 "_PROBE_CONFIG_PATH"):
        assert f'open(self.{name}, "w"' not in src, name
