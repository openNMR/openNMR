"""Cell beside spectra (opennmr.notebook.echem_plot)."""
import os
import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from opennmr.notebook import echem, echem_plot as ep
from test_echem_20261120 import _profile, MASS_MG


@pytest.fixture
def cell(tmp_path):
    path = tmp_path / "cell.csv"
    _profile().to_csv(path, index=False)
    return echem.load_echem(path, mass_mg=MASS_MG, molar_mass=100.0, x0=1.0)["data"]


def _series(n, npts=64):
    ppm = np.linspace(100, -100, npts)
    rows = [np.exp(-((ppm - (50 - 4 * k)) / 6.0) ** 2) for k in range(n)]
    return ppm, np.vstack(rows)


def test_experiment_numbers_range_and_errors():
    assert ep.experiment_numbers(11, 20, 3) == [11, 14, 17, 20]
    assert ep.experiment_numbers(5, 1, -2) == [5, 3, 1]
    with pytest.raises(ValueError):
        ep.experiment_numbers(1, 5, 0)
    with pytest.raises(ValueError):
        ep.experiment_numbers(1, 5, -1)


def test_experiment_number_from_path():
    assert ep.experiment_number("C:/data/sample/12") == 12
    assert ep.experiment_number("C:/data/sample/12/pdata/1") == 12
    assert ep.experiment_number("C:/data/sample/name") is None
    assert ep.experiment_number(None) is None


def test_operando_link_and_outside_note(cell):
    starts, ends = echem.operando_times(6, start_s=1000, per_spectrum_s=3600)
    link = ep.link_table(cell, "operando", starts_s=starts, ends_s=ends)
    assert len(link) == 6 and list(link["index"]) == [1, 2, 3, 4, 5, 6]
    assert np.all(np.isfinite(link["voltage_V"]))
    starts2, ends2 = echem.operando_times(3, start_s=1e9, per_spectrum_s=60)
    late = ep.link_table(cell, "operando", starts_s=starts2, ends_s=ends2)
    assert late.attrs["notes"]


def test_ex_situ_link_places_samples(cell):
    pts = [{"value": 3.5, "cycle": 1, "half": "charge", "label": "A"},
           {"value": 3.5, "cycle": 2, "half": "discharge", "label": "B"},
           {"value": 9.0, "cycle": 1, "half": "charge", "label": "bad"}]
    link = ep.link_table(cell, "ex_situ", points=pts)
    assert link["voltage_V"].iloc[0] == pytest.approx(3.5, abs=0.02)
    assert link["time_s"].iloc[0] < link["time_s"].iloc[1]
    assert np.isnan(link["voltage_V"].iloc[2])
    assert any("bad" in n for n in link.attrs["notes"])


def test_common_axis_interpolates():
    class Ax:
        def __init__(s, v): s.v = v
        def values(s, unit): return s.v
    class Sp:
        ndim = 1
        def __init__(s, x, y): s.axes = [Ax(x)]; s.data = y
    x1 = np.linspace(10, -10, 21); x2 = np.linspace(10, -10, 41)
    ppm, m, notes = ep.common_axis([Sp(x1, x1 ** 2), Sp(x2, x2 ** 2)])
    assert m.shape == (2, 21) and np.allclose(m[1], x1 ** 2, atol=1e-9) and notes


@pytest.mark.parametrize("side", ep.SIDES)
@pytest.mark.parametrize("display", ["stacked", "superimposed", "heatmap"])
def test_every_side_and_display_draws(cell, side, display):
    starts, ends = echem.operando_times(8, start_s=0, per_spectrum_s=7200)
    link = ep.link_table(cell, "operando", starts_s=starts, ends_s=ends)
    ppm, m = _series(8)
    fig = ep.plot_echem_nmr(cell, ppm, m, link, side=side, display=display, position="time_h")
    assert len(fig.axes) >= 2
    pos = {s: fig.axes_cell.get_position() for s in [side]}[side]
    sp = fig.axes_spectra.get_position()
    if side == "right": assert pos.x0 > sp.x0
    if side == "left": assert pos.x0 < sp.x0
    if side == "top": assert pos.y0 > sp.y0
    if side == "bottom": assert pos.y0 < sp.y0
    plt.close(fig)


def test_shared_axis_when_position_matches(cell):
    starts, ends = echem.operando_times(8, start_s=0, per_spectrum_s=7200)
    link = ep.link_table(cell, "operando", starts_s=starts, ends_s=ends)
    ppm, m = _series(8)
    fig = ep.plot_echem_nmr(cell, ppm, m, link, side="right", display="heatmap",
                            x="voltage_V", y="time_h", position="time_h")
    assert fig.axes_cell.get_shared_y_axes().joined(fig.axes_cell, fig.axes_spectra)
    plt.close(fig)


def test_differences_and_values_and_style_options(cell):
    starts, ends = echem.operando_times(8, start_s=0, per_spectrum_s=7200)
    link = ep.link_table(cell, "operando", starts_s=starts, ends_s=ends)
    ppm, m = _series(8)
    fig = ep.plot_echem_nmr(cell, ppm, m, link, display="heatmap", differences="successive",
                            cell_xlabel="E (V)", show_cell_ticklabels=False, position="time_h")
    assert fig.axes_cell.get_xlabel() == "E (V)"
    assert fig.axes_cell.xaxis._major_tick_kw["label1On"] is False
    assert fig.axes_cell.yaxis._major_tick_kw["label1On"] is False
    assert fig.axes_spectra.xaxis._major_tick_kw["label1On"] is True
    plt.close(fig)
    vals = pd.DataFrame({"shift": np.linspace(50, 20, 8), "area": np.linspace(1, 2, 8)})
    fig = ep.plot_echem_nmr(cell, None, None, link, display="values", values=vals,
                            value_columns=["shift"], position="time_h", side="top")
    assert fig.axes_spectra.lines
    plt.close(fig)
    with pytest.raises(ValueError):
        ep.plot_echem_nmr(cell, ppm, m[:5], link)
    with pytest.raises(ValueError):
        ep.plot_echem_nmr(cell, ppm, m, link, x="nope")


def test_crosshair_links_both_ways(cell):
    starts, ends = echem.operando_times(8, start_s=0, per_spectrum_s=7200)
    link = ep.link_table(cell, "operando", starts_s=starts, ends_s=ends)
    ppm, m = _series(8)
    fig = ep.plot_echem_nmr(cell, ppm, m, link, display="heatmap", x="voltage_V", y="time_h",
                            position="time_h")
    ch = fig._echem_crosshair
    i = ch.update_from_spectra(link["time_h"].iloc[3])
    assert i == 3 and ch._cell_h.get_visible()
    j = ch.update_from_cell(link["voltage_V"].iloc[5], link["time_h"].iloc[5])
    assert j == 5
    ch.show(None)
    assert not ch._cell_h.get_visible()
    old = ch
    fig2 = ep.plot_echem_nmr(cell, ppm, m, link, figure=fig, display="heatmap")
    assert fig2._echem_crosshair is not old and old._cids == []
    plt.close(fig)


_REAL = [r"C:\Users\quark\OneDrive\Documents\Electrochemistry\LIO\euanbassey_LIO003_cycle_test_10mAg-1_20230123_C06.mpr",
         r"C:\Users\quark\OneDrive\Documents\Electrochemistry\Alpha_NaMnO2_Cycling_Tests\eb599_Alpha_NaMnO2_002_cycle_test_10mAg-1_25_cycles.mpr"]


@pytest.mark.parametrize("path", _REAL)
def test_real_mpr_files_plot(path):
    if not os.path.exists(path):
        pytest.skip("real .mpr file not present")
    pytest.importorskip("navani")
    data = echem.load_echem(path, mass_mg=5.0, molar_mass=100.0, x0=1.0)["data"]
    assert data["cycle"].max() >= 2
    t0 = float(data["time_s"].iloc[0]); span = float(data["time_s"].iloc[-1]) - t0
    starts, ends = echem.operando_times(20, start_s=t0, per_spectrum_s=span / 20)
    link = ep.link_table(data, "operando", starts_s=starts, ends_s=ends)
    ppm, m = _series(20)
    fig = ep.plot_echem_nmr(data, ppm, m, link, display="heatmap", position="time_h",
                            x="voltage_V", y="time_h")
    plt.close(fig)


def _ex_link(cell, n=4):
    pts = [{"value": 3.3 + 0.1 * k, "cycle": 1, "half": "charge", "label": f"S{k + 1}"} for k in range(n)]
    return ep.link_table(cell, "ex_situ", points=pts)


def test_no_spectrum_placed_raises_a_clear_error_not_a_numpy_one(cell):
    pts = [{"value": 9.0, "cycle": 1, "half": "charge", "label": "a"},
           {"value": 9.5, "cycle": 1, "half": "charge", "label": "b"}]
    link = ep.link_table(cell, "ex_situ", points=pts)
    ppm, m = _series(2)
    with pytest.raises(ValueError, match="No spectrum could be placed"):
        ep.plot_echem_nmr(cell, ppm, m, link, display="stacked", position="time_h")
    # evenly spaced spectra do not need a place on the cell
    fig = ep.plot_echem_nmr(cell, ppm, m, link, display="stacked", distribute="even")
    assert fig.axes_spectra.lines
    plt.close(fig)


@pytest.mark.parametrize("distribute", ["even", "intensity"])
@pytest.mark.parametrize("display", ["stacked", "heatmap"])
def test_even_distribution_labels_each_spectrum(cell, distribute, display):
    link = _ex_link(cell)
    ppm, m = _series(4)
    m = m * np.array([[1.0], [0.5], [2.0], [1.0]])
    fig = ep.plot_echem_nmr(cell, ppm, m, link, ["S1", "S2", "S3", "S4"], display=display,
                            distribute=distribute, position="time_h", side="right")
    ax = fig.axes_spectra
    fig.canvas.draw()
    texts = [t.get_text() for t in ax.get_yticklabels()]
    assert texts == ["S1", "S2", "S3", "S4"]
    ticks = np.array(ax.get_yticks())
    gaps = np.diff(ticks)
    if distribute == "even":
        assert np.allclose(gaps, gaps[0])
    else:
        assert gaps[2] > gaps[1]          # the tall spectrum is given more room
    assert not ax.get_shared_y_axes().joined(ax, fig.axes_cell)
    plt.close(fig)


def test_colours_link_between_point_and_spectrum(cell):
    link = _ex_link(cell, 3)
    ppm, m = _series(3)
    fig = ep.plot_echem_nmr(cell, ppm, m, link, display="superimposed", position="time_h",
                            point_colours=["red", None, "blue"], link_colours=True, marker_colour="green")
    from matplotlib.colors import to_hex
    spectra = [to_hex(l.get_color()) for l in fig.axes_spectra.lines if l.get_linestyle() == "-"]
    assert spectra == [to_hex("red"), to_hex("green"), to_hex("blue")]
    points = [to_hex(l.get_color()) for l in fig.axes_cell.lines if l.get_marker() == "o" and len(l.get_xdata())]
    assert points == spectra
    plt.close(fig)
    fig = ep.plot_echem_nmr(cell, ppm, m, link, display="superimposed", position="time_h",
                            point_colours=["red", "red", "red"], spectrum_colours=["black", "gray", "k"],
                            link_colours=False)
    assert [to_hex(l.get_color()) for l in fig.axes_spectra.lines if l.get_linestyle() == "-"][1] == to_hex("gray")
    plt.close(fig)


@pytest.mark.parametrize("side", ["left", "right", "top", "bottom"])
def test_shared_axis_keeps_ticks_only_on_the_outer_plot(cell, side):
    starts, ends = echem.operando_times(8, start_s=0, per_spectrum_s=7200)
    link = ep.link_table(cell, "operando", starts_s=starts, ends_s=ends)
    ppm, m = _series(8)
    vertical = side in ("left", "right")
    kw = dict(x="voltage_V", y="time_h") if vertical else dict(x="time_h", y="voltage_V")
    fig = ep.plot_echem_nmr(cell, ppm, m, link, side=side, display="heatmap", position="time_h", **kw)
    axis = "y" if vertical else "x"
    inner = fig.axes_cell if side in ("right", "top") else fig.axes_spectra
    outer = fig.axes_spectra if inner is fig.axes_cell else fig.axes_cell
    tick_kw = lambda a: getattr(a, f"{axis}axis")._major_tick_kw
    first, second = ("label1On", "tick1On") if vertical else ("label1On", "tick1On")
    assert tick_kw(inner)["label1On"] is False and tick_kw(inner)["tick1On"] is False
    assert tick_kw(outer).get("label1On", True) is True
    plt.close(fig)
    # moving the cell round and plotting into the same figure leaves nothing behind
    fig = plt.figure()
    for s in ep.SIDES:
        ep.plot_echem_nmr(cell, ppm, m, link, figure=fig, side=s, display="heatmap", position="time_h")
    assert len(fig.axes) <= 3
    plt.close(fig)


def test_scaling_factors_mass_scans_and_manual_factor():
    class Md:
        mass = None; ns = 8; vendor = "jeol"; raw = {"ns": 8}
    class Sp:
        def __init__(s, ns): s.metadata = Md(); s.ndim = 1; s.data = np.ones(3)
    sp = [Sp(8), Sp(8)]
    f, notes = ep.scaling_factors(sp, masses_mg=[2.0, 4.0], factors=[1.0, 10.0])
    assert np.allclose(f, [0.5, 2.5])
    with pytest.raises(ValueError, match="sample mass"):
        ep.scaling_factors(sp, masses_mg=[2.0, np.nan])
    with pytest.raises(ValueError):
        ep.scaling_factors(sp, factors=[1.0, 2.0, 3.0])


def _operando(cell, n=6):
    starts, ends = echem.operando_times(n, start_s=0, per_spectrum_s=7200)
    return ep.link_table(cell, "operando", starts_s=starts, ends_s=ends)


def test_nucleus_names_the_shift_axis(cell):
    link = _operando(cell)
    ppm, m = _series(6)
    fig = ep.plot_echem_nmr(cell, ppm, m, link, display="superimposed", nucleus="23Na")
    from opennmr.core.registry import NUC_label
    assert fig.axes_spectra.get_xlabel() == NUC_label("23Na", x_unit="ppm")
    assert "mathregular{23}" in fig.axes_spectra.get_xlabel()          # upright mass number
    fig = ep.plot_echem_nmr(cell, ppm, m, link, display="superimposed")
    assert "delta" in fig.axes_spectra.get_xlabel() and "chemical" not in fig.axes_spectra.get_xlabel().lower()
    plt.close(fig)


def test_x_and_y_ticks_are_controlled_separately(cell):
    link = _operando(cell)
    ppm, m = _series(6)
    fig = ep.plot_echem_nmr(cell, ppm, m, link, display="superimposed", cell_xticks="none",
                            cell_yticks="both", spectra_xticks="ticks", spectra_yticks="none")
    kw = lambda a, axis: getattr(a, f"{axis}axis")._major_tick_kw
    assert kw(fig.axes_cell, "x")["label1On"] is False and kw(fig.axes_cell, "x")["tick1On"] is False
    assert kw(fig.axes_cell, "y").get("label1On", True) is True
    assert kw(fig.axes_spectra, "x")["label1On"] is False and kw(fig.axes_spectra, "x")["tick1On"] is True
    assert kw(fig.axes_spectra, "y")["label1On"] is False and kw(fig.axes_spectra, "y")["tick1On"] is False
    with pytest.raises(ValueError):
        ep.plot_echem_nmr(cell, ppm, m, link, display="superimposed", cell_xticks="some")
    plt.close(fig)


def test_top_and_right_spines_can_be_added_per_plot(cell):
    link = _operando(cell)
    ppm, m = _series(6)
    fig = ep.plot_echem_nmr(cell, ppm, m, link, display="superimposed", cell_top_spine=True,
                            spectra_right_spine=True)
    assert fig.axes_cell.spines["top"].get_visible() and not fig.axes_cell.spines["right"].get_visible()
    assert fig.axes_spectra.spines["right"].get_visible() and not fig.axes_spectra.spines["top"].get_visible()
    plt.close(fig)
    fig = ep.plot_echem_nmr(cell, ppm, m, link, display="superimposed")
    assert not any(fig.axes_cell.spines[k].get_visible() for k in ("top", "right"))
    plt.close(fig)


def test_an_explicit_spectrum_colour_beats_the_link(cell):
    link = _ex_link(cell, 3)
    ppm, m = _series(3)
    fig = ep.plot_echem_nmr(cell, ppm, m, link, display="superimposed", position="time_h",
                            point_colours=["red", "red", "red"], spectrum_colours=[None, "blue", None],
                            link_colours=True)
    from matplotlib.colors import to_hex
    got = [to_hex(l.get_color()) for l in fig.axes_spectra.lines if l.get_linestyle() == "-"]
    assert got == [to_hex("red"), to_hex("blue"), to_hex("red")]
    plt.close(fig)


def test_capacity_curve_has_no_line_between_half_cycles(cell):
    link = _operando(cell)
    ppm, m = _series(6)
    fig = ep.plot_echem_nmr(cell, ppm, m, link, display="superimposed", x="voltage_V",
                            y="capacity_spec_mAh_g", side="right")
    line = [l for l in fig.axes_cell.lines if l.get_label() == "cell"][0]
    ys = np.asarray(line.get_ydata(), float)
    assert np.isnan(ys).sum() == cell["half_cycle"].nunique() - 1      # one gap between halves
    plt.close(fig)


def test_labels_have_upright_superscripts():
    for name, text in echem.AXIS_LABELS.items():
        assert "^{-" not in text and "^{2" not in text, name          # only \mathregular{...} ones
    assert "mathregular{-1}" in echem.AXIS_LABELS["capacity_spec_mAh_g"]


def test_spectrum_and_point_share_a_series_name_when_linked(cell):
    link = _ex_link(cell, 3)
    ppm, m = _series(3)
    for linked, expect in ((True, "S1"), (False, "S1 (point)")):
        fig = ep.plot_echem_nmr(cell, ppm, m, link, ["S1", "S2", "S3"], display="superimposed",
                                position="time_h", link_colours=linked)
        assert "S1" in [l.get_label() for l in fig.axes_spectra.lines]
        assert expect in [l.get_label() for l in fig.axes_cell.lines]
        plt.close(fig)


@pytest.mark.parametrize("side", ep.SIDES)
@pytest.mark.parametrize("display", ["stacked", "superimposed", "heatmap"])
def test_the_spectra_are_always_the_first_axes(cell, side, display):
    link = _operando(cell)
    ppm, m = _series(6)
    fig = ep.plot_echem_nmr(cell, ppm, m, link, side=side, display=display, position="time_h")
    assert fig.axes[0] is fig.axes_spectra and fig.axes[1] is fig.axes_cell
    plt.close(fig)


def test_gap_between_the_plots_and_removing_the_spectra_left_spine(cell):
    link = _operando(cell)
    ppm, m = _series(6)
    tight = ep.plot_echem_nmr(cell, ppm, m, link, display="superimposed", gap=0.02, colourbar=False)
    wide = ep.plot_echem_nmr(cell, ppm, m, link, display="superimposed", gap=0.3, colourbar=False)
    sep = lambda f: f.axes_cell.get_position().x0 - f.axes_spectra.get_position().x1   # cell on the right
    assert sep(wide) > sep(tight) + 0.05
    fig = ep.plot_echem_nmr(cell, ppm, m, link, display="superimposed", spectra_left_spine=False)
    assert not fig.axes_spectra.spines["left"].get_visible()
    assert fig.axes_spectra.yaxis._major_tick_kw["label1On"] is False
    assert fig.axes_cell.spines["left"].get_visible()
    plt.close(fig)
