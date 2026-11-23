"""2026-11-23: Plot other data -- skip any number of header lines, and add the top and right
spines (a 3-D figure: the remaining edges of its box)."""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pytest

from opennmr.notebook import other_plot as op
from test_gui import qapp  # noqa: F401 -- qapp is a fixture

BODY = "T_K,shift,width\n300,1.0,5\n310,1.5,6\n320,2.1,7\n330,2.4,9\n"
HEADER = "Instrument: XYZ\nSample: test\n\nsome, free, text\n"


def test_skip_lines_from_text_and_file(tmp_path):
    expected = op.read_table(BODY)
    assert op.read_table(HEADER + BODY, skip_lines=4).equals(expected)
    f = tmp_path / "d.csv"
    f.write_text(HEADER + BODY)
    assert op.read_table(f, skip_lines=4).equals(expected)
    assert op.read_table(str(f), skip_lines=4).equals(expected)


def test_skip_lines_before_a_headerless_table():
    t = op.read_table("title\n1\t2\n3\t4\n5\t6\n", skip_lines=1)
    assert list(t.columns) == ["col1", "col2"] and len(t) == 3 and t.iloc[0].tolist() == [1, 2]


def test_skipping_nothing_changes_nothing_and_negative_is_refused():
    assert op.read_table(BODY, skip_lines=0).equals(op.read_table(BODY))
    with pytest.raises(ValueError, match="negative"):
        op.read_table(BODY, skip_lines=-1)


@pytest.mark.parametrize("kind", op.KINDS)
def test_2d_spines_default_off_and_can_be_added(kind):
    table = op.read_table(BODY)
    ax = op.plot_other_data(table, x="T_K", y="shift", kind=kind).axes[0]
    assert not ax.spines["top"].get_visible() and not ax.spines["right"].get_visible()
    ax = op.plot_other_data(table, x="T_K", y="shift", kind=kind, top_spine=True,
                            right_spine=True).axes[0]
    assert ax.spines["top"].get_visible() and ax.spines["right"].get_visible()
    assert ax.spines["bottom"].get_visible() and ax.spines["left"].get_visible()
    ax = op.plot_other_data(table, x="T_K", y="shift", kind=kind, right_spine=True).axes[0]
    assert ax.spines["right"].get_visible() and not ax.spines["top"].get_visible()


def test_3d_box_edges_are_drawn_only_when_asked():
    table = op.read_table(BODY)
    plain = op.plot_other_data(table, x="T_K", y="shift", z="width", kind="scatter").axes[0]
    boxed = op.plot_other_data(table, x="T_K", y="shift", z="width", kind="scatter",
                               top_spine=True).axes[0]
    assert len(boxed.get_lines()) - len(plain.get_lines()) == 12
    # The axes limits are unchanged by the box.
    assert boxed.get_xlim3d() == plain.get_xlim3d() and boxed.get_zlim3d() == plain.get_zlim3d()


def test_the_panel_skips_lines_and_plot_style_adds_spines(qapp):
    from opennmr.gui.other_plot_panel import OtherPlotPanel
    drawn = []
    panel = OtherPlotPanel(lambda fn: drawn.append(fn),
                           style=lambda: dict(top_spine=True, right_spine=True))   # Plot Style's
    panel.skip_spin.setValue(4)
    assert panel.add_pasted(HEADER + BODY)
    assert list(panel.sources[0]["table"].columns) == ["T_K", "shift", "width"]
    panel.plot()
    assert drawn
    from matplotlib.figure import Figure
    fig = Figure()
    drawn[0](fig)
    ax = fig.axes[0]
    assert ax.spines["top"].get_visible() and ax.spines["right"].get_visible()
    code = panel.notebook_code()
    assert "skip_lines=4" in code and "top_spine=True" in code and "right_spine=True" in code
