"""Experiment title/notes: Bruker pdata/<procno>/title read and written
in place (with a one-time title.orig backup), a sidecar ``notes`` entry
for every other format, and the editable box under Open experiments."""

import os

import numpy as np
import pytest

from opennmr import io
from opennmr import notebook as nb
from opennmr.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
from opennmr.gui.main_window import MainWindow
from opennmr.io import bruker as io_bruker
from opennmr.io import persistence as io_persistence

from test_gui import qapp, _make_1d_dataset  # noqa: F401 -- fixture


def _title(expt, procno=1):
    return os.path.join(expt, "pdata", str(procno), "title")


def _synthetic(vendor, source_path=None, **raw):
    return NMRSpectrum(
        data=np.zeros(4), axes=[NMRAxis(kind="frequency", ppm=np.linspace(0, 1, 4))],
        metadata=NMRMetadata(vendor=vendor, source_path=source_path, raw=dict(raw)))


# --- io.bruker ---------------------------------------------------------
def test_read_title_normalises_crlf_and_missing_file(tmp_path):
    expt = _make_1d_dataset(tmp_path)
    assert io_bruker.read_title(expt, 1) is None
    with open(_title(expt), "wb") as f:
        f.write(b"LiFePO4 7Li\r\nspin 60 kHz\r\n\r\n")
    assert io_bruker.read_title(expt, 1) == "LiFePO4 7Li\nspin 60 kHz"


def test_read_title_latin1_from_windows_topspin(tmp_path):
    expt = _make_1d_dataset(tmp_path)
    with open(_title(expt), "wb") as f:
        f.write("T = 80 °C".encode("latin-1"))
    assert io_bruker.read_title(expt, 1) == "T = 80 °C"


def test_write_title_backs_up_original_once_and_keeps_encoding(tmp_path):
    expt = _make_1d_dataset(tmp_path)
    original = "as acquired °".encode("latin-1")
    with open(_title(expt), "wb") as f:
        f.write(original)
    io_bruker.write_title(expt, 1, "first edit °C")
    io_bruker.write_title(expt, 1, "second edit")
    with open(_title(expt) + ".orig", "rb") as f:
        assert f.read() == original              # the acquired title, not the first edit
    with open(_title(expt), "rb") as f:
        assert f.read() == b"second edit\n"
    io_bruker.write_title(expt, 1, "third °")
    with open(_title(expt), "rb") as f:
        assert f.read() == "third °\n".encode("latin-1")   # still what TopSpin wrote
    assert not os.path.exists(_title(expt) + ".tmp")


def test_write_title_without_existing_file_makes_no_backup(tmp_path):
    expt = _make_1d_dataset(tmp_path)
    io_bruker.write_title(expt, 1, "new")
    assert io_bruker.read_title(expt, 1) == "new"
    assert not os.path.exists(_title(expt) + ".orig")


def test_write_title_refuses_missing_procno(tmp_path):
    expt = _make_1d_dataset(tmp_path)
    with pytest.raises(FileNotFoundError):
        io_bruker.write_title(expt, 7, "x")
    assert not os.path.exists(os.path.join(expt, "pdata", "7"))


def test_title_is_per_procno(tmp_path):
    expt = _make_1d_dataset(tmp_path)
    os.makedirs(os.path.join(expt, "pdata", "2"))
    io_bruker.write_title(expt, 1, "procno one")
    io_bruker.write_title(expt, 2, "procno two")
    assert io_bruker.read_title(expt, 1) == "procno one"
    assert io_bruker.read_title(expt, 2) == "procno two"


# --- notebook.notes ----------------------------------------------------
def test_notebook_bruker_round_trip(tmp_path):
    expt = _make_1d_dataset(tmp_path)
    spectrum = io.load(expt, vendor="bruker")
    assert nb.experiment_notes(spectrum) == ""
    where = nb.set_experiment_notes(spectrum, "sample 3\r\nafter 2 h  \n")
    assert where == _title(expt)
    assert nb.notes_location(spectrum) == _title(expt)
    assert nb.experiment_notes(spectrum) == "sample 3\nafter 2 h"
    # Bruker never goes to the sidecar.
    assert io_persistence.load_notes_sidecar(expt) is None


def test_notebook_non_bruker_uses_sidecar_and_keeps_other_keys(tmp_path):
    data_file = tmp_path / "run.jdf"
    data_file.write_bytes(b"")
    spectrum = _synthetic("jeol", str(data_file), title="JEOL header title", comment="a comment")
    # Until edited, the format's own title shows.
    assert nb.experiment_notes(spectrum) == "JEOL header title\na comment"
    sidecar = io_persistence.metadata_sidecar_path(str(data_file))
    assert not os.path.exists(sidecar)
    io_persistence._write_json(sidecar, {"mass_g": 0.01})
    assert nb.set_experiment_notes(spectrum, "my notes") == sidecar
    assert nb.experiment_notes(spectrum) == "my notes"
    assert io_persistence.load_metadata_sidecar(str(data_file))["mass_g"] == 0.01
    # An empty edit is a real choice: it does not fall back to the header.
    nb.set_experiment_notes(spectrum, "")
    assert nb.experiment_notes(spectrum) == ""


def test_notes_survive_a_metadata_sidecar_rewrite(tmp_path):
    data_file = tmp_path / "run.jdf"
    data_file.write_bytes(b"")
    spectrum = _synthetic("jeol", str(data_file))
    nb.set_experiment_notes(spectrum, "keep me")
    io_persistence.save_metadata_sidecar(spectrum)
    assert nb.experiment_notes(spectrum) == "keep me"


def test_varian_text_file_is_the_default_title(tmp_path):
    fid_dir = tmp_path / "s1.fid"
    fid_dir.mkdir()
    (fid_dir / "text").write_text("VnmrJ sample text\n")
    spectrum = _synthetic("varian", str(fid_dir))
    assert nb.experiment_notes(spectrum) == "VnmrJ sample text"
    nb.set_experiment_notes(spectrum, "edited")
    assert (fid_dir / "text").read_text() == "VnmrJ sample text\n"     # vendor file untouched
    assert nb.experiment_notes(spectrum) == "edited"


def test_spectrum_without_a_file_keeps_notes_in_session(tmp_path):
    spectrum = _synthetic("derived", "ratio a/b")
    assert "session" in nb.set_experiment_notes(spectrum, "derived note")
    assert nb.experiment_notes(spectrum) == "derived note"
    assert not any(tmp_path.iterdir())


def test_first_line():
    assert nb.notes_first_line("\n  first  \nsecond") == "first"
    assert nb.notes_first_line("") == ""
    assert nb.notes_first_line("x" * 100, limit=10) == "x" * 9 + "…"


# --- GUI ----------------------------------------------------------------
def test_notes_box_shows_and_saves_bruker_title(qapp, tmp_path):
    a = _make_1d_dataset(tmp_path, name="a")
    b = _make_1d_dataset(tmp_path, name="b")
    with open(_title(a), "w") as f:
        f.write("title of a\nline two\n")
    window = MainWindow()
    window._load(a, vendor="bruker")
    window._load(b, vendor="bruker")
    editor = window.notes_editor
    window._select_single_row(0)
    assert editor.text() == "title of a\nline two"
    assert window.spectrum_list.item(0).toolTip() == "title of a"
    assert "pdata" in editor.location_label.text()

    # Typing, then switching experiment, saves the edit to *a*.
    editor.edit.setPlainText("edited a")
    assert editor.is_dirty()
    window._select_single_row(1)
    assert io_bruker.read_title(a, 1) == "edited a"
    assert os.path.exists(_title(a) + ".orig")
    assert window.spectrum_list.item(0).toolTip() == "edited a"
    assert editor.text() == ""                       # b has no title

    # The idle-timer path saves too.
    editor.edit.setPlainText("title of b")
    editor._timer.timeout.emit()
    assert io_bruker.read_title(b, 1) == "title of b"


def test_notes_box_follows_removal_and_clear(qapp, tmp_path):
    a = _make_1d_dataset(tmp_path, name="a")
    window = MainWindow()
    window._load(a, vendor="bruker")
    window._select_single_row(0)
    window.notes_editor.edit.setPlainText("unsaved at removal")
    window._on_remove_clicked()
    assert io_bruker.read_title(a, 1) == "unsaved at removal"   # flushed, not lost
    assert window.notes_editor.spectrum is None
    assert not window.notes_editor.edit.isEnabled()

    window._load(a, vendor="bruker")
    window._select_single_row(0)
    assert window.notes_editor.text() == "unsaved at removal"
    window._on_clear_clicked()
    assert window.notes_editor.spectrum is None


def test_notes_save_failure_is_reported_not_raised(qapp, tmp_path, monkeypatch):
    a = _make_1d_dataset(tmp_path, name="a")
    window = MainWindow()
    window._load(a, vendor="bruker")
    window._select_single_row(0)

    def refuse(*_args, **_kwargs):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(io_bruker, "write_title", refuse)
    failures = []
    window.notes_editor.failed.connect(lambda _s, msg: failures.append(msg))
    window.notes_editor.edit.setPlainText("x")
    assert window.notes_editor.flush() is False
    assert failures and "Permission denied" in failures[0]
    assert "Couldn't save" in window.notes_editor.location_label.text()


def test_notes_box_hidden_with_open_experiments_on_simulate(qapp):
    window = MainWindow()
    assert window.notes_editor.parent() is not None
    # It lives inside the Open experiments box, so it shows and hides with it.
    widget = window.notes_editor
    while widget is not None and widget is not window.open_experiments_box:
        widget = widget.parentWidget()
    assert widget is window.open_experiments_box
