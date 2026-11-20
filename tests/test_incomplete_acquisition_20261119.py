"""Notes 8.11: empty, cut-off and over-long-list acquisitions."""
import numpy as np
import pytest

from opennmr.io import bruker


def _dic(td=256, td1=8):
    return {"acqus": {"TD": td, "DTYPA": 0}, "acqu2s": {"TD": td1}}


def _write_ser(path, rows, *, extra_bytes=0, zero_rows=0, td=256):
    """int32 ser with `rows` rows of 1024 bytes (td=256 -> one block), the last
    `zero_rows` of them all zero, plus `extra_bytes` of a partly written row."""
    data = np.arange(1, rows * td + 1, dtype="<i4").reshape(rows, td)
    if zero_rows:
        data[rows - zero_rows:] = 0
    with open(path / "ser", "wb") as fh:
        fh.write(data.tobytes())
        fh.write(b"\x01" * extra_bytes)


def test_empty_file_raises_a_clear_error(tmp_path):
    (tmp_path / "ser").write_bytes(b"")
    with pytest.raises(bruker.EmptyAcquisitionError) as err:
        bruker._check_not_empty(str(tmp_path / "ser"))
    assert bruker.EMPTY_ACQUISITION_MARKER in str(err.value)
    assert isinstance(err.value, ValueError)


def test_non_empty_file_passes(tmp_path):
    _write_ser(tmp_path, 2)
    bruker._check_not_empty(str(tmp_path / "ser"))


def test_cut_off_series_keeps_only_acquired_rows_and_says_so(tmp_path):
    _write_ser(tmp_path, 5, extra_bytes=300)             # 5 whole rows + a partial 6th
    data = bruker._read_raw_2d_data(str(tmp_path), _dic(td1=8))
    assert data.shape[0] == 5                            # partial row dropped
    info = bruker.acquisition_extent(_dic(td1=8), str(tmp_path / "ser"), data.shape[0])
    assert info["rows_acquired"] == 5 and info["rows_expected"] == 8
    assert not info["complete"] and info["td_disagrees"]
    assert info["partial_bytes"] == 300
    assert "5 of 8" in info["note"] and "partly written" in info["note"]


def test_trailing_zero_rows_of_a_stopped_acquisition_are_dropped(tmp_path):
    _write_ser(tmp_path, 6, zero_rows=2)                 # rows 5, 6 never written
    data = bruker._read_raw_2d_data(str(tmp_path), _dic(td1=8))
    assert data.shape[0] == 4


def test_complete_acquisition_is_untouched_and_silent(tmp_path):
    _write_ser(tmp_path, 8)
    data = bruker._read_raw_2d_data(str(tmp_path), _dic(td1=8))
    assert data.shape[0] == 8
    info = bruker.acquisition_extent(_dic(td1=8), str(tmp_path / "ser"), 8)
    assert info["complete"] and info["note"] == ""


def test_genuine_zero_row_in_a_complete_series_is_kept(tmp_path):
    _write_ser(tmp_path, 8, zero_rows=1)                 # all rows present: nothing is cut
    assert bruker._read_raw_2d_data(str(tmp_path), _dic(td1=8)).shape[0] == 8


def test_has_processed_data(tmp_path):
    assert not bruker._has_processed_data(str(tmp_path))
    (tmp_path / "pdata" / "1").mkdir(parents=True)
    (tmp_path / "pdata" / "1" / "1r").write_bytes(b"x")
    assert bruker._has_processed_data(str(tmp_path))


def test_value_list_longer_than_the_acquired_rows_is_cut_and_logged():
    from opennmr.gui.main_window import trim_values_to_rows
    values, note = trim_values_to_rows(np.arange(24.0), 16)
    assert values.size == 16 and values[-1] == 15.0
    assert "24 entries" in note and "16 rows" in note
    same, none = trim_values_to_rows(np.arange(16.0), 16)
    assert same.size == 16 and none == ""
    short, none = trim_values_to_rows(np.arange(10.0), 16)
    assert short.size == 10 and none == ""               # shorter is an error elsewhere, not a cut


def test_vdlist_dict_is_cut_consistently():
    from opennmr.gui.main_window import trim_values_to_rows
    vd = {"name": "vdlist", "raw": [str(i) for i in range(12)], "seconds": np.arange(12.0)}
    out, note = trim_values_to_rows(vd, 5)
    assert len(out["raw"]) == 5 and out["seconds"].size == 5 and out["name"] == "vdlist"
    assert note
