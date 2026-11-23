"""2026-11-22: Plot other data -- any table as a 2-D / 3-D scatter, line or bar chart
(notebook.other_plot) and its Apply panel (gui/other_plot_panel.py)."""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from opennmr.notebook import other_plot as op
from test_gui import qapp  # noqa: F401 -- qapp is a fixture

TEXT = "T_K,shift,width\n300,1.0,5\n310,1.5,6\n320,2.1,7\n330,2.4,9\n"


@pytest.fixture
def table():
    return op.read_table(TEXT)


def test_read_table_from_text_and_file(tmp_path):
    t = op.read_table(TEXT)
    assert list(t.columns) == ["T_K", "shift", "width"] and len(t) == 4
    f = tmp_path / "d.csv"
    f.write_text(TEXT)
    assert op.read_table(f).equals(t) and op.read_table(str(f)).equals(t)


def test_separators_are_found_and_headerless_tables_are_named():
    t = op.read_table("1\t2\n3\t4\n5\t6\n")
    assert list(t.columns) == ["col1", "col2"] and t.iloc[0].tolist() == [1, 2] and len(t) == 3
    assert op.read_table("a;b\n1;2\n3;4\n").shape == (2, 2)


def test_bad_tables_say_why(tmp_path):
    with pytest.raises(ValueError):
        op.read_table(str(tmp_path / "missing.csv"))
    with pytest.raises(ValueError, match="two numeric columns"):
        op.read_table("a\n1\n2\n")


@pytest.mark.parametrize("kind", op.KINDS)
def test_every_kind_draws_in_2d_and_3d(table, kind):
    fig = op.plot_other_data(table, x="T_K", y=["shift", "width"], kind=kind)
    assert len(fig.axes) >= 1 and not hasattr(fig.axes[0], "get_zlim")
    fig3 = op.plot_other_data(table, x="T_K", y="shift", z="width", kind=kind)
    assert hasattr(fig3.axes[0], "get_zlim")


def test_series_and_labels(table):
    fig = op.plot_other_data(table, x="T_K", y=["shift", "width"], kind="line", labels=["a", "b"])
    ax = fig.axes[0]
    assert [l.get_label() for l in ax.get_lines()] == ["a", "b"] and ax.get_xlabel() == "T_K"
    fig = op.plot_other_data(table, x="T_K", y="shift", xlabel="x!", ylabel="y!", title="t")
    ax = fig.axes[0]
    assert (ax.get_xlabel(), ax.get_ylabel(), ax.get_title()) == ("x!", "y!", "t")


def test_scatter_points_are_the_data(table):
    ax = op.plot_other_data(table, x="T_K", y="shift", kind="scatter").axes[0]
    assert np.allclose(ax.collections[0].get_offsets(), table[["T_K", "shift"]].to_numpy())


def test_bar_heights_are_the_data(table):
    ax = op.plot_other_data(table, x="T_K", y="shift", kind="bar").axes[0]
    assert np.allclose([r.get_height() for r in ax.patches], table["shift"])


def test_colour_by_adds_a_colourbar(table):
    fig = op.plot_other_data(table, x="T_K", y="shift", colour_by="width")
    assert len(fig.axes) == 2


def test_sequences_instead_of_columns():
    ax = op.plot_other_data(None, x=[1, 2, 3], y=[4, 5, 6], kind="line").axes[0]
    assert np.allclose(ax.get_lines()[0].get_ydata(), [4, 5, 6])


def test_nan_rows_are_skipped():
    ax = op.plot_other_data(None, x=[1, 2, 3], y=[4, np.nan, 6], kind="scatter").axes[0]
    assert len(ax.collections[0].get_offsets()) == 2


def test_errors(table):
    with pytest.raises(ValueError, match="No column"):
        op.plot_other_data(table, x="nope", y="shift")
    with pytest.raises(ValueError, match="kind"):
        op.plot_other_data(table, x="T_K", y="shift", kind="pie")
    with pytest.raises(ValueError, match="same length"):
        op.plot_other_data(None, x=[1, 2], y=[1, 2, 3])
    with pytest.raises(ValueError, match="one y column"):
        op.plot_other_data(table, x="T_K", y=["shift", "width"], z="width")


def test_draws_on_a_given_figure(table):
    fig = Figure()
    assert op.plot_other_data(table, x="T_K", y="shift", figure=fig) is fig


# ------------------------------------------------------------------ the panel
def test_panel_reads_plots_and_writes_code(qapp):
    from opennmr.gui.other_plot_panel import OtherPlotPanel
    drawn = []
    panel = OtherPlotPanel(draw=lambda fn: drawn.append(fn), fontsize=lambda: 11.0)
    assert panel.add_pasted(TEXT)
    assert len(panel.sources) == 1 and len(panel.datasets) == 1       # a first dataset comes with the file
    assert panel.x_combo.count() >= 3
    panel.plot()
    assert "Plotted" in panel.report_label.text() and len(drawn) == 1
    fig = Figure()
    drawn[0](fig)
    assert fig.axes
    panel.dim_combo.setCurrentIndex(panel.dim_combo.findData("3d"))
    panel.default_kind_combo.setCurrentIndex(panel.default_kind_combo.findData("line"))
    panel.datasets[0]["z"] = {"col": list(panel.sources[0]["table"].columns)[2]}
    panel.plot()
    fig = Figure()
    drawn[-1](fig)
    assert hasattr(fig.axes[0], "get_zlim")
    code = panel.notebook_code()
    assert "plot_datasets" in code and "'line'" in code


def test_panel_without_data_says_so(qapp):
    from opennmr.gui.other_plot_panel import OtherPlotPanel
    drawn = []
    panel = OtherPlotPanel(draw=drawn.append)
    panel.plot()
    assert not drawn and panel.report_label.text()


def test_section_is_registered_in_apply(qapp):
    from opennmr.gui.main_window import MainWindow
    w = MainWindow()
    assert w.OPTIONAL_SECTIONS["other_plot"][1] == ("_other_plot_section",)
    assert "other_plot" in w.APPLY_TILE_SECTIONS and hasattr(w, "other_plot_panel")
