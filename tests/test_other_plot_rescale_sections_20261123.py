"""2026-11-23: multiply the data of a dataset by constants (2-D and 3-D plots), and read the
Dynamics curve-fit export (parameters, blank line, series) as its series."""
import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from PySide6.QtWidgets import QInputDialog

from opennmr.notebook import other_plot as op
from opennmr.processing import relaxation

from test_gui import qapp  # noqa: F401 -- fixture
from test_other_plot_datasets_20261123 import _fit, _panel  # noqa: F401


def test_dataset_scales_each_axis_and_the_error_bars_by_the_size_of_y():
    ds = op.dataset([1, 2, 3], [10, 20, 30], [1, 1, 2], yerr=[1, 2, 3], xscale=2, yscale=-0.5, zscale=10)
    assert list(ds["x"]) == [2, 4, 6] and list(ds["y"]) == [-5, -10, -15] and list(ds["z"]) == [10, 10, 20]
    assert list(ds["yerr"]) == [0.5, 1.0, 1.5]
    plain = op.dataset([1, 2], [1, 2])
    assert list(plain["x"]) == [1, 2] and list(plain["y"]) == [1, 2]
    with pytest.raises(ValueError, match="finite"):
        op.dataset([1], [1], yscale=float("nan"))


def test_the_plot_shows_the_scaled_numbers_in_2d_and_3d():
    ax = op.plot_datasets([op.dataset([1, 2, 3], [1, 2, 3], kind="line", yscale=1000)]).axes[0]
    assert list(ax.lines[0].get_ydata()) == [1000, 2000, 3000]
    fig = op.plot_datasets([op.dataset([1, 2], [1, 2], [3, 4], zscale=0.1)])
    assert fig.axes[0].name == "3d"
    assert fig.axes[0].get_zlim()[1] < 1                       # z in tenths, not 3-4


def test_rescaled_copies_a_ready_made_dataset_and_keeps_the_curve_over_its_points():
    sets = op.curve_fit_datasets([_fit()])
    scaled = [op.rescaled(d, y=100) for d in sets]
    for a, b in zip(sets, scaled):
        assert np.allclose(b["y"], 100 * a["y"]) and np.allclose(a["x"], b["x"])
    assert sets[0]["y"].max() < 2                             # the original is untouched
    with pytest.raises(ValueError):
        op.rescaled(sets[0], y=float("inf"))


def test_panel_multiplies_a_dataset_and_says_so_in_the_code(qapp):  # noqa: F811
    panel, drawn = _panel()
    panel.add_pasted("T\ta\n0\t1\n1\t2\n2\t3\n")
    panel.scale_edits["y"].setText("1e3")
    panel.scale_edits["y"].editingFinished.emit()
    panel.scale_edits["x"].setText("0,5")                     # a decimal comma is fine
    panel.scale_edits["x"].editingFinished.emit()
    sets = panel.build_datasets()
    assert list(sets[0]["y"]) == [1000, 2000, 3000] and list(sets[0]["x"]) == [0, 0.5, 1.0]
    assert "yscale=1000" in panel.notebook_code() and "xscale=0.5" in panel.notebook_code()
    panel.scale_edits["y"].setText("lots")
    panel.scale_edits["y"].editingFinished.emit()
    with pytest.raises(ValueError, match="multiplier"):
        panel.build_datasets()
    panel.plot()
    assert "multiplier" in panel.report_label.text()


def test_a_curve_fit_dataset_is_scaled_too(qapp, monkeypatch):  # noqa: F811
    panel, drawn = _panel(curve_fits=lambda: [("sat rec", [_fit()], 2)])
    monkeypatch.setattr(QInputDialog, "getItem", staticmethod(lambda *a, **k: (a[3][0], True)))
    panel.add_pasted("T\ta\n0\t1\n1\t2\n")
    panel.add_curve_fit()
    before = panel.build_datasets()[1]["y"].copy()
    panel.dataset_tree.setCurrentItem(panel.dataset_tree.topLevelItem(1))
    panel.scale_edits["y"].setText("10")
    panel.scale_edits["y"].editingFinished.emit()
    after = panel.build_datasets()[1]["y"]
    assert np.allclose(after, 10 * before)
    assert "op.rescaled(datasets[1]" in panel.notebook_code()


def test_the_curve_fit_export_is_read_as_its_series(tmp_path):
    t = np.array([0.01, 0.02, 0.05, 0.1, 0.2])
    y = 1 - 2 * np.exp(-t / 0.08)
    results = [{"t": t, "y": y, "y_fit": y, "peak_label": "1.2 ppm", "model": "inversion recovery",
                "params": {"M0": 1.0, "T1": 0.08}, "errors": {"M0": 0.01, "T1": 0.002},
                "fit_quality": {"r_squared": 0.999}}]
    path = tmp_path / "curve.csv"
    path.write_text(relaxation.curve_fit_to_csv(results, x_label="delay (ms)", x_scale=1e-3))
    table = op.read_table(str(path))
    assert list(table.columns) == ["delay (ms)", "1.2 ppm value", "1.2 ppm fit"]
    assert table.shape == (5, 3) and table.attrs["section"] == "Curve fit series"
    assert list(table["delay (ms)"]) == [10, 20, 50, 100, 200]
    # a plain file behaves as before
    assert op.read_table("a,b\n1,2\n3,4\n").shape == (2, 2)
    assert "section" not in op.read_table("a,b\n1,2\n3,4\n").attrs


def test_several_tables_without_a_series_title_give_the_one_with_most_numbers():
    table = op.read_table("first\na,b\n1,2\n\nsecond\nx,y\n1,2\n3,4\n5,6\n")
    assert list(table.columns) == ["x", "y"] and len(table) == 3 and table.attrs["section"] == "second"
