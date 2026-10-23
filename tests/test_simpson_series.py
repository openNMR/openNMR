"""notebook/simpson_series.py: points from FIDs, plotted against a named quantity."""
import numpy as np
import pytest

from opennmr.notebook import simpson_series as ss

import simpson_files as sf


@pytest.fixture
def files(tmp_path):
    """Five FIDs 'b1_10.fid' ... whose first three points are known."""
    paths, rows = [], []
    for k, amp in enumerate((10, 20, 30, 40, 50)):
        values = [amp * (1 + 0j), amp * 0.5 + 1j * amp * 0.25, amp * 0.1 - 1j * 0.3 * amp, 9, 9]
        rows.append(values[:3])
        paths.append(sf.write_point_fid(tmp_path / f"b1_{amp}.fid", values))
    return paths, np.array(rows)


def test_series_reads_the_leading_points(files):
    paths, rows = files
    series = ss.SIMPSONpoint_series(paths, x="filenames", xname="B1", n_points=3)
    assert series["letters"] == "ABC"
    assert np.allclose(series["x"], [10, 20, 30, 40, 50])
    assert np.allclose(series["points"], rows)


def test_default_number_of_points_is_what_every_file_has(files):
    series = ss.SIMPSONpoint_series(files[0])
    assert series["points"].shape == (5, 5)


def test_expression_combines_points(files):
    paths, rows = files
    series = ss.SIMPSONpoint_series(paths, x="filenames", n_points=3)
    got = ss.SIMPSONcombine_points(series, "A + B - 0.3*C")
    assert np.allclose(got, (rows[:, 0] + rows[:, 1] - 0.3 * rows[:, 2]).real)
    assert np.allclose(ss.SIMPSONcombine_points(series, "A", part="imag"), rows[:, 0].imag)
    assert np.allclose(ss.SIMPSONcombine_points(series, "B", part="magnitude"), np.abs(rows[:, 1]))
    assert np.allclose(ss.SIMPSONcombine_points(series, "(A - B)/2"),
                       ((rows[:, 0] - rows[:, 1]) / 2).real)


@pytest.mark.parametrize("bad", ["Z", "A +", "__import__('os')", "A.real", "A; B", "open('x')"] )
def test_expressions_are_checked_not_executed(files, bad):
    series = ss.SIMPSONpoint_series(files[0], x="filenames", n_points=3)
    assert ss.check_expression(series, bad) is not None
    with pytest.raises(ValueError):
        ss.SIMPSONcombine_points(series, bad)


def test_powers_functions_and_x_are_allowed(files):
    paths, rows = files
    series = ss.SIMPSONpoint_series(paths, x="filenames", n_points=3)
    assert np.allclose(ss.SIMPSONcombine_points(series, "A**2"), (rows[:, 0] ** 2).real)
    assert np.allclose(ss.SIMPSONcombine_points(series, "A/x"), (rows[:, 0] / series["x"]).real)
    assert np.allclose(ss.SIMPSONcombine_points(series, "2*A - 1j*0", part="real"),
                       2 * rows[:, 0].real)


def test_x_must_match_the_files(files):
    with pytest.raises(ValueError):
        ss.SIMPSONpoint_series(files[0], x=[1, 2, 3])


def test_table_and_csv(files, tmp_path):
    series = ss.SIMPSONpoint_series(files[0], x=[1, 2, 3, 4, 5], xname="tp", n_points=3)
    out = ss.SIMPSONsave_point_table(series, ["A", "A + B"], str(tmp_path / "t.csv"))
    lines = open(out).read().splitlines()
    assert lines[0].split(",")[0] == "tp" and len(lines) == 6


def test_plot_uses_the_user_s_name_and_label(files):
    import matplotlib
    matplotlib.use("Agg")
    series = ss.SIMPSONpoint_series(files[0], x="filenames", xname="B1",
                                    xlabel="RF amplitude (kHz)", n_points=3)
    fig, ax = ss.SIMPSONplot_point_series(series, ["A", "A + B - 0.3*C"])
    assert "RF amplitude" in ax.get_xlabel()
    assert len(ax.lines) == 2
    assert len(ax.lines[0].get_xdata()) == 5
