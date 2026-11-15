"""Insets: data line width, tick length / width / direction and axis
thickness are controllable (explicit user request, 2026-11-15)."""
import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from opennmr.plotting import inset as ins

from test_gui import qapp  # noqa: F401 -- fixture


def _plot():
    fig, ax = plt.subplots()
    x = np.linspace(10, 0, 2000)
    ax.plot(x, np.exp(-0.5 * ((x - 5) / 0.05) ** 2), color="k", linewidth=1.5, label="s")
    ax.set_xlim(10, 0)
    return fig, ax


def test_defaults_leave_an_inset_as_it_was():
    fig, ax = _plot()
    inset = ins.add_inset(ax, (5.3, 4.7), auto_ylim=True)
    assert inset.get_lines()[0].get_linewidth() == pytest.approx(1.5)
    tick = inset.xaxis.get_major_ticks()[0].tick1line
    assert tick.get_markersize() == pytest.approx(3.0)
    plt.close(fig)


def test_data_line_width_ticks_and_axis_thickness_are_applied():
    fig, ax = _plot()
    inset = ins.add_inset(ax, (5.3, 4.7), auto_ylim=True, linewidth=3.0, tick_length=7.0,
                          tick_width=2.0, tick_direction="out", frame_width=2.5)
    assert inset.get_lines()[0].get_linewidth() == pytest.approx(3.0)
    assert ax.get_lines()[0].get_linewidth() == pytest.approx(1.5)      # the plot's own untouched
    tick = inset.xaxis.get_major_ticks()[0]
    assert tick.tick1line.get_markersize() == pytest.approx(7.0)
    assert tick.tick1line.get_markeredgewidth() == pytest.approx(2.0)
    assert all(s.get_linewidth() == pytest.approx(2.5) for s in inset.spines.values())
    plt.close(fig)


def test_an_inset_without_a_width_copies_the_plots():
    fig, ax = _plot()
    inset = ins.add_inset(ax, (5.3, 4.7), auto_ylim=True)
    assert inset.get_lines()[0].get_linewidth() == pytest.approx(1.5)
    plt.close(fig)


def test_bad_values_are_refused():
    fig, ax = _plot()
    with pytest.raises(ValueError):
        ins.add_inset(ax, (5.3, 4.7), tick_direction="sideways")
    with pytest.raises(ValueError):
        ins.add_inset(ax, (5.3, 4.7), linewidth=-1)
    plt.close(fig)


def test_the_panel_edits_reach_the_spec(qapp):  # noqa: F811
    from opennmr.gui.inset_dialog import InsetPanel
    panel = InsetPanel(None)
    panel._specs = [{"xlim": (5.3, 4.7), "ylim": (0.0, 1.0)}]
    panel._refresh_list()
    assert panel.data_width_spin.value() == 0.0          # "Same as plot"
    panel.data_width_spin.setValue(2.5)
    panel.tick_length_spin.setValue(6.0)
    panel.tick_width_spin.setValue(1.5)
    panel.frame_width.setValue(2.0)
    panel.tick_direction_combo.setCurrentIndex(panel.tick_direction_combo.findData("out"))
    spec = panel.specs()[0]
    assert spec["linewidth"] == 2.5 and spec["tick_length"] == 6.0 and spec["tick_width"] == 1.5
    assert spec["frame_width"] == 2.0 and spec["tick_direction"] == "out"
    panel.data_width_spin.setValue(0.0)                  # back to following the plot
    assert panel.specs()[0]["linewidth"] is None
    # Selecting it again shows what was set.
    panel._load_editor(0)
    assert panel.tick_length_spin.value() == 6.0
