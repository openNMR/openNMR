"""User report, 2026-11-12: saving (or overwriting) an openNMR state failed
with "[Errno 2] No such file or directory: ...\\spectra\\41_VOCS sum (20
spectra) _ 23Na _ derived _ procno 1 _ 23Na _ derived _ procno 1 _ \\
opennmr_spectrum.json.tmp", whatever the state was called.

safe_folder_name trimmed a label's ends and *then* cut it to 80 characters,
so a long label could end in a space; Windows drops a trailing space (or
dot) from a folder name when it creates it, and writing into the name as
given then fails. The cut now comes first."""

import os

import pytest

from opennmr.io import history as io_history
from opennmr.io import state as io_state

from test_state_and_fit_models import _derived

LONG = ("VOCS sum (20 spectra) · 23Na · derived · procno 1 · 23Na · derived · "
        "procno 1 · 23Na · derived · procno 1")


@pytest.mark.parametrize("label", [
    LONG,
    "a" * 79 + " b",                 # the cut lands just after a space
    "a" * 79 + ". b",                # ... or a dot
    "x" * 200,
    "  ...  ",
])
def test_names_never_end_in_a_space_or_dot(label):
    name = io_history.safe_folder_name(label)
    assert name and len(name) <= io_history.SAFE_NAME_LENGTH
    assert name == name.strip(" .")
    assert not name.endswith("_")


def test_the_reported_label():
    assert io_history.safe_folder_name(LONG) == \
        "VOCS sum (20 spectra) _ 23Na _ derived _ procno 1 _ 23Na _ derived _ procno 1"


@pytest.mark.parametrize("label", ["CON", "nul", "Com1", "LPT9.txt", "aux.dat"])
def test_windows_device_names_are_not_used(label):
    assert io_history.safe_folder_name(label).endswith("_")


def test_ordinary_names_unchanged():
    assert io_history.safe_folder_name("231 · 23Na · bruker") == "231 _ 23Na _ bruker"
    assert io_history.safe_folder_name("console") == "console"


def test_a_state_with_a_long_label_saves_and_reopens(tmp_path):
    folder = str(tmp_path / "MnHCF_dry_Na_NMR.opennmr")
    entries = [{"label": LONG, "kind": "derived", "spectrum": _derived()}]
    io_state.save_state(folder, "MnHCF dry", entries)
    sub = os.listdir(os.path.join(folder, "spectra"))
    assert len(sub) == 1 and sub[0] == sub[0].rstrip(" .")
    # Overwriting it, as in the report, works too.
    io_state.save_state(folder, "MnHCF dry", entries, overwrite=True)
    info = io_state.load_state(folder)
    assert info["entries"][0]["label"] == LONG
    assert info["entries"][0]["spectrum"] is not None
