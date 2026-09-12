"""Vendor-neutral one-column value lists -- opennmr.io.lists."""

import os

import numpy as np
import pytest

from opennmr.io import bruker as io_bruker
from opennmr.io import lists as io_lists


def test_read_value_list_plain_numbers(tmp_path):
    path = tmp_path / "delays.txt"
    path.write_text("0.001\n0.01\n0.1\n1.0\n")
    parsed = io_lists.read_value_list(str(path))
    np.testing.assert_allclose(parsed["values"], [0.001, 0.01, 0.1, 1.0])
    # "seconds" is the key NMR2D_fit_relaxation and read_vdlist already
    # speak; it is the same array, not a converted one.
    np.testing.assert_allclose(parsed["seconds"], parsed["values"])
    assert parsed["name"] == "delays.txt"


def test_read_value_list_applies_bruker_time_suffixes(tmp_path):
    path = tmp_path / "vdlist"
    path.write_text("100u\n10m\n1s\n500n\n")
    parsed = io_lists.read_value_list(str(path))
    np.testing.assert_allclose(parsed["values"], [1e-4, 1e-2, 1.0, 5e-7])


def test_read_value_list_does_not_mistake_exponents_for_suffixes(tmp_path):
    # "1e5" ends in a letter that is not a unit; "5e-9" ends in a digit.
    # Getting this wrong would silently scale a b-value list by 1e-3.
    path = tmp_path / "bvalues.txt"
    path.write_text("1e5\n2.5e8\n5e-9\n")
    parsed = io_lists.read_value_list(str(path))
    np.testing.assert_allclose(parsed["values"], [1e5, 2.5e8, 5e-9])


def test_read_value_list_skips_blanks_and_comments(tmp_path):
    path = tmp_path / "list.txt"
    path.write_text("# gradient fractions\n\n0.1\n; another comment\n0.5\n$$ jcamp\n0.9\n")
    parsed = io_lists.read_value_list(str(path))
    np.testing.assert_allclose(parsed["values"], [0.1, 0.5, 0.9])
    assert parsed["raw"] == ["0.1", "0.5", "0.9"]


def test_read_value_list_takes_the_first_column(tmp_path):
    path = tmp_path / "list.txt"
    path.write_text("0.1  first step\n0.5  second step\n")
    parsed = io_lists.read_value_list(str(path))
    np.testing.assert_allclose(parsed["values"], [0.1, 0.5])


def test_read_value_list_rejects_an_unreadable_entry_rather_than_skipping_it(tmp_path):
    # A silently-short list is the failure mode that produces a
    # plausible-looking wrong fit, so this must raise, not shrug.
    path = tmp_path / "list.txt"
    path.write_text("0.1\nnot a number\n0.5\n")
    with pytest.raises(ValueError, match="line 2"):
        io_lists.read_value_list(str(path))


def test_read_value_list_rejects_an_empty_file(tmp_path):
    path = tmp_path / "list.txt"
    path.write_text("# nothing but a comment\n")
    with pytest.raises(ValueError, match="no numbers"):
        io_lists.read_value_list(str(path))


def test_read_value_list_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        io_lists.read_value_list(str(tmp_path / "nope.txt"))


def test_find_value_lists_finds_vendor_lists_in_the_experiment_folder(tmp_path):
    expt = tmp_path / "11"
    expt.mkdir()
    (expt / "vdlist").write_text("10m\n20m\n")
    (expt / "acqus").write_text("##$TD= 1024\n")
    found = io_lists.find_value_lists(str(expt))
    assert [os.path.basename(p) for p in found] == ["vdlist"]


def test_find_value_lists_finds_an_xpulse_parameters_valueslist(tmp_path):
    expt = tmp_path / "4"
    (expt / "parameters").mkdir(parents=True)
    (expt / "parameters" / "G1_valueslist.txt").write_text("0.05\n0.15\n0.95\n")
    found = io_lists.find_value_lists(str(expt))
    assert [os.path.basename(p) for p in found] == ["G1_valueslist.txt"]


def test_find_value_lists_finds_a_list_beside_the_experiment(tmp_path):
    # "usually co-located in the same parent folder as the data" -- a
    # hand-written list next to the experiment, not inside it.
    sample = tmp_path / "sample"
    expt = sample / "1"
    expt.mkdir(parents=True)
    (sample / "delays.txt").write_text("0.01\n0.02\n")
    found = io_lists.find_value_lists(str(expt))
    assert [os.path.basename(p) for p in found] == ["delays.txt"]
    assert io_lists.find_value_lists(str(expt), include_parent=False) == []


def test_find_value_lists_accepts_a_file_path_as_the_dataset(tmp_path):
    # X-Pulse/JEOL datasets are a file, not a folder.
    expt = tmp_path / "expt"
    expt.mkdir()
    (expt / "spectrum.jdx").write_text("##TITLE=x\n")
    (expt / "valueslist.txt").write_text("1\n2\n3\n")
    found = io_lists.find_value_lists(str(expt / "spectrum.jdx"), include_parent=False)
    assert [os.path.basename(p) for p in found] == ["valueslist.txt"]


def test_find_value_lists_ignores_extensionless_vendor_clutter(tmp_path):
    # Measured on real Bruker data: gpnam5, scon2, EA and a preemphasis
    # table all parse perfectly well as columns of numbers. Offering
    # them next to the difflist that is actually wanted turns a
    # one-click choice into a guessing game.
    expt = tmp_path / "150"
    expt.mkdir()
    (expt / "difflist").write_text("100\n200\n300\n")
    for clutter in ("gpnam5", "scon2", "EA"):
        (expt / clutter).write_text("1.0\n2.0\n3.0\n")
    found = io_lists.find_value_lists(str(expt))
    assert [os.path.basename(p) for p in found] == ["difflist"]


def test_find_value_lists_ignores_text_files_that_are_not_lists(tmp_path):
    expt = tmp_path / "1"
    expt.mkdir()
    (expt / "vdlist").write_text("10m\n20m\n")
    (expt / "README.txt").write_text("Acquired on the 400 on a Tuesday.\n")
    found = io_lists.find_value_lists(str(expt))
    assert [os.path.basename(p) for p in found] == ["vdlist"]


def test_find_value_lists_needs_more_than_one_entry(tmp_path):
    expt = tmp_path / "1"
    expt.mkdir()
    (expt / "single.txt").write_text("42\n")
    assert io_lists.find_value_lists(str(expt)) == []


def test_find_value_lists_deduplicates_bruker_list_copies(tmp_path):
    expt = tmp_path / "1"
    (expt / "lists" / "vd").mkdir(parents=True)
    (expt / "vdlist").write_text("10m\n20m\n")
    (expt / "lists" / "vd" / "vdlist").write_text("10m\n20m\n")
    found = io_lists.find_value_lists(str(expt))
    # Two distinct files with the same name: both are offered (they are
    # genuinely different paths and could differ), but each only once.
    assert len(found) == len(set(os.path.realpath(p) for p in found))


def test_find_value_lists_orders_known_names_first(tmp_path):
    expt = tmp_path / "1"
    expt.mkdir()
    (expt / "aaa.txt").write_text("1\n2\n")
    (expt / "vdlist").write_text("10m\n20m\n")
    (expt / "G1_valueslist.txt").write_text("0.1\n0.9\n")
    found = [os.path.basename(p) for p in io_lists.find_value_lists(str(expt))]
    assert found == ["vdlist", "G1_valueslist.txt", "aaa.txt"]


def test_find_value_lists_on_a_missing_path_is_empty():
    assert io_lists.find_value_lists("") == []
    assert io_lists.find_value_lists("/definitely/not/here") == []


def test_bruker_read_vdlist_still_returns_its_own_shape(tmp_path):
    # read_vdlist now delegates its parsing here; its contract (name,
    # raw strings, seconds) must be unchanged.
    expt = tmp_path / "1"
    expt.mkdir()
    (expt / "vdlist").write_text("10m\n100m\n1s\n")
    parsed = io_bruker.read_vdlist(str(expt))
    assert parsed["name"] == "vdlist"
    assert parsed["raw"] == ["10m", "100m", "1s"]
    np.testing.assert_allclose(parsed["seconds"], [0.01, 0.1, 1.0])


def test_read_value_list_reports_whether_the_file_named_its_own_units(tmp_path):
    # The difference between a list whose numbers are already absolute
    # times and one whose numbers mean whatever the caller decides.
    suffixed = tmp_path / "vdlist"
    suffixed.write_text("10m\n20m\n40m\n")
    assert io_lists.read_value_list(str(suffixed))["units_explicit"] is True

    bare = tmp_path / "delays.txt"
    bare.write_text("10\n20\n40\n")
    assert io_lists.read_value_list(str(bare))["units_explicit"] is False


def test_read_value_list_exponents_do_not_count_as_units(tmp_path):
    path = tmp_path / "bvalues.txt"
    path.write_text("1e5\n2.5e8\n")
    assert io_lists.read_value_list(str(path))["units_explicit"] is False


def test_has_unit_suffix():
    assert io_lists.has_unit_suffix("10m")
    assert io_lists.has_unit_suffix("500u")
    assert io_lists.has_unit_suffix("1.5s")
    assert not io_lists.has_unit_suffix("10")
    assert not io_lists.has_unit_suffix("1e5")
    assert not io_lists.has_unit_suffix("m")
    assert not io_lists.has_unit_suffix("")
