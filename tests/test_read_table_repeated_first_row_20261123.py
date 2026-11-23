"""2026-11-23: a headerless table whose first row repeats a value ("0  0") came back
as 0 and 0.1 (pandas renames repeated header names), drawing a point that is not in
the file (SH3_14A.txt)."""
import pytest

from opennmr.notebook.other_plot import read_table


def test_first_row_with_repeated_values_is_untouched():
    df = read_table("0\t0\n0.5\t2.5E-06\n1\t1.0E-05\n")
    assert list(df.columns) == ["col1", "col2"]
    assert df.iloc[0].tolist() == [0.0, 0.0] and len(df) == 3


def test_headers_still_detected_and_skip_lines():
    assert list(read_table("x,y\n1,2\n3,4").columns) == ["x", "y"]
    df = read_table("title\n\nt\tI\n0\t0\n1\t5", skip_lines=2)
    assert list(df.columns) == ["t", "I"] and df.iloc[0].tolist() == [0.0, 0.0]
    assert read_table("1 2\n3 4\n").shape == (2, 2)


def test_crlf_and_decimal_comma_rows():
    df = read_table("0\t0\r\n0.25\t1.5\r\n")
    assert df.shape == (2, 2) and df.iloc[1, 1] == pytest.approx(1.5)
