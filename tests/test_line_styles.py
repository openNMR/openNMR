"""Plot Style ▸ Lines: the width and style of every line on every plot.

plotting/lines.py does the work on a finished figure (so a notebook
figure gets it too); MplCanvas applies it as the figure draws, so every
kind of plot -- spectrum, fit, curve, Compare, notebook figure -- and
its export get it; MainWindow's Plot Style lists the lines."""
import io
import json

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from PySide6.QtWidgets import QApplication, QComboBox, QDoubleSpinBox  # noqa: E402

from opennmr.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum  # noqa: E402
from opennmr.gui.widgets.canvas import MplCanvas  # noqa: E402
from opennmr.plotting import lines as pl  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    yield QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


def _figure():
    fig, ax = plt.subplots()
    x = np.linspace(0, 1, 20)
    ax.plot(x, x, label="Data")
    ax.plot(x, x ** 2, "--", lw=2.0, label="Fit")
    ax.plot(x, 0 * x, color="0.5")                 # unlabelled
    ax.axvline(0.5, animated=True)                  # a crosshair-like UI line
    ax.legend()
    return fig, ax


def _spectrum(peak=5.0, name="1H"):
    ppm = np.linspace(0, 10, 200)
    metadata = NMRMetadata(vendor="bruker", nucleus=name, sfo1=400.0)
    axis = NMRAxis(kind="frequency", ppm=ppm, hz=ppm * 400.0)
    return NMRSpectrum(data=np.exp(-((ppm - peak) ** 2) / 0.1), axes=[axis], metadata=metadata)


def _render(canvas):
    canvas.canvas.draw()          # synchronous: runs the line-style pass


# --------------------------------------------------------- plotting.lines

def test_data_lines_keys_names_and_ui_lines_left_out():
    fig, _ax = _figure()
    infos = pl.data_lines(fig)
    assert [i["key"] for i in infos] == ["label:Data", "label:Fit", "line:0.0"]
    assert [i["name"] for i in infos] == ["Data", "Fit", "Line 1"]
    assert infos[1]["linewidth"] == 2.0 and infos[1]["linestyle"] == "--"
    assert infos[2]["colour"] == "#808080"


def test_scale_override_and_exact_undo():
    fig, ax = _figure()
    pl.apply_line_styles(fig, 2.0, {"label:Fit": {"linestyle": ":"},
                                    "line:0.0": {"linewidth": 0.5, "linestyle": "dashed"}})
    data, fit, flat = ax.lines[:3]
    assert data.get_linewidth() == pytest.approx(3.0)        # 1.5 x 2
    assert fit.get_linewidth() == pytest.approx(4.0)         # its own 2.0 x 2
    assert fit.get_linestyle() == ":"
    assert flat.get_linewidth() == pytest.approx(0.5) and flat.get_linestyle() == "--"
    # applying again does not compound
    pl.apply_line_styles(fig, 2.0, {"label:Fit": {"linestyle": ":"}})
    assert data.get_linewidth() == pytest.approx(3.0)
    pl.apply_line_styles(fig, 1.0, {})
    assert [l.get_linewidth() for l in ax.lines[:3]] == pytest.approx([1.5, 2.0, 1.5])
    assert [l.get_linestyle() for l in ax.lines[:3]] == ["-", "--", "-"]
    assert ax.lines[3].get_linewidth() == pytest.approx(1.5)  # the animated line untouched


def test_legend_entries_follow_their_line():
    fig, ax = _figure()
    pl.apply_line_styles(fig, 1.0, {"label:Data": {"linestyle": "-.", "linewidth": 3.0}})
    handle = ax.get_legend().legend_handles[0]
    assert handle.get_linestyle() == "-." and handle.get_linewidth() == pytest.approx(3.0)
    pl.apply_line_styles(fig, 1.0, {"label:Data": {"linewidth": 3.0}}, legend_widths=False)
    assert ax.get_legend().legend_handles[0].get_linestyle() == "-"


def test_custom_dash_pattern_is_restored_exactly():
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], linestyle=(0, (8, 3)), label="long")
    pl.apply_line_styles(fig, 1.0, {"label:long": {"linestyle": "-"}})
    pl.apply_line_styles(fig, 1.0, {})
    assert tuple(ax.lines[0]._unscaled_dash_pattern[1]) == (8, 3)


def test_every_offered_style_draws(tmp_path):
    fig, ax = plt.subplots()
    for i, (name, value) in enumerate(pl.LINESTYLES):
        ax.plot([0, 1], [i, i], label=name)
    pl.apply_line_styles(fig, 1.0, {f"label:{name}": {"linestyle": value}
                                    for name, value in pl.LINESTYLES})
    fig.savefig(tmp_path / "styles.png")
    for name, value in pl.LINESTYLES:
        assert pl.linestyle_name(value) == name


def test_normalise_linestyle():
    assert pl.normalise_linestyle("dashed") == "--"
    assert pl.normalise_linestyle("dot-dash") == "-."
    assert pl.normalise_linestyle([0, [8, 3]]) == (0, (8, 3))      # from JSON
    assert pl.linestyle_name("dotted") == "Dotted"
    assert pl.linestyle_name((0, (5, 7))) == "Custom"
    with pytest.raises(ValueError):
        pl.normalise_linestyle("wiggly")


def test_exported_from_plotting():
    from opennmr import plotting
    assert plotting.apply_line_styles is pl.apply_line_styles
    assert "LINESTYLES" in plotting.__all__


# ------------------------------------------------------------------ canvas

def test_canvas_scales_and_overrides_a_spectrum_and_keeps_it_on_redraw(qapp):
    canvas = MplCanvas()
    canvas.show_spectrum(_spectrum())
    _render(canvas)
    infos = canvas.data_lines()
    assert infos, "a spectrum has a line to style"
    key = infos[0]["key"]
    base = infos[0]["base_linewidth"]
    canvas.set_style(line_width_scale=2.0)
    _render(canvas)
    assert canvas.figure.axes[0].lines[0].get_linewidth() == pytest.approx(2 * base)
    canvas.set_line_style(key, linestyle="--", linewidth=0.7)
    _render(canvas)
    line = canvas.figure.axes[0].lines[0]
    assert line.get_linestyle() == "--" and line.get_linewidth() == pytest.approx(0.7)
    canvas.show_spectrum(_spectrum(peak=4.0))            # a redraw, new artists
    _render(canvas)
    line = canvas.figure.axes[0].lines[0]
    assert line.get_linestyle() == "--" and line.get_linewidth() == pytest.approx(0.7)
    canvas.set_line_style(key, clear=True)
    _render(canvas)
    assert canvas.figure.axes[0].lines[0].get_linewidth() == pytest.approx(2 * base)
    canvas.reset_line_styles()
    _render(canvas)
    assert canvas.figure.axes[0].lines[0].get_linewidth() == pytest.approx(base)


def test_canvas_styles_reach_the_exported_file(qapp, tmp_path):
    canvas = MplCanvas()
    canvas.show_spectrum(_spectrum())
    key = canvas.data_lines()[0]["key"]
    canvas.set_line_style(key, linewidth=5.0)
    buf = io.BytesIO()
    canvas.figure.savefig(buf, format="svg")
    assert "stroke-width: 5" in buf.getvalue().decode()


def test_canvas_styles_a_fit_and_a_notebook_figure(qapp):
    canvas = MplCanvas()
    x = np.linspace(0, 10, 200)
    y = np.exp(-((x - 5) ** 2))
    canvas.show_slice_fit(x, y, y * 0.98)
    _render(canvas)
    names = [i["name"] for i in canvas.data_lines()]
    assert len(names) >= 2                             # data and fit at least
    keys = [i["key"] for i in canvas.data_lines()]
    for k in keys:
        canvas.set_line_style(k, linestyle=":")
    _render(canvas)
    assert all(l.get_linestyle() == ":" for i in canvas.data_lines() for l in i["lines"])

    def draw(fig):
        ax = fig.add_subplot(111)
        ax.plot([0, 1], [0, 1], label="model")
        ax.plot([0, 1], [1, 0], "o", label="points")
    canvas.show_external_plot(draw, kind="curve")
    canvas.set_line_style("label:model", linestyle="-.", linewidth=2.5)
    _render(canvas)
    line = canvas.figure.axes[0].lines[0]
    assert line.get_linestyle() == "-." and line.get_linewidth() == pytest.approx(2.5)


def test_canvas_style_settings_round_trip_through_json(qapp):
    canvas = MplCanvas()
    canvas.set_style(line_width_scale=1.5, line_overrides={"label:Fit": {"linestyle": (0, (8, 3))}})
    settings = canvas.style_settings()
    assert settings["line_width_scale"] == 1.5
    saved = json.loads(json.dumps({"line_overrides": {"label:Fit": {"linestyle": [0, [8, 3]]}},
                                   "line_width_scale": 1.5}))
    other = MplCanvas()
    other.apply_style_defaults(saved)
    assert other.style_settings()["line_overrides"] == {"label:Fit": {"linestyle": (0, (8, 3))}}
    assert other.style_settings()["line_width_scale"] == 1.5
    with pytest.raises(ValueError):
        canvas.set_style(line_width_scale=0)
    with pytest.raises(ValueError):
        canvas.set_line_style("label:Fit", linewidth=-1)


def test_canvas_signals_when_its_lines_change(qapp):
    canvas = MplCanvas()
    seen = []
    canvas.linesChanged.connect(lambda: seen.append(1))
    canvas.show_spectrum(_spectrum())
    _render(canvas)
    QApplication.processEvents()
    assert seen


# ------------------------------------------------------------- Plot Style

def test_plot_style_lists_and_sets_lines(qapp):
    from opennmr.gui.main_window import MainWindow
    window = MainWindow()
    index = window._add_derived_spectrum(_spectrum(), "sample")
    window.spectrum_list.setCurrentRow(index)
    window.canvas_tabs.setCurrentWidget(window.canvas)
    _render(window.canvas)
    window._on_plot_style_clicked()
    assert window.lines_group.isVisibleTo(window._style_panel)
    rows = window._line_row_widgets
    assert rows
    key, (width, style, reset) = next(iter(rows.items()))
    assert isinstance(width, QDoubleSpinBox) and isinstance(style, QComboBox)
    assert not reset.isEnabled()
    width.setValue(3.0)
    style.setCurrentIndex(style.findText("Dash-dot"))
    _render(window.canvas)
    line = next(i for i in window.canvas.data_lines() if i["key"] == key)["line"]
    assert line.get_linewidth() == pytest.approx(3.0) and line.get_linestyle() == "-."
    assert reset.isEnabled()
    # saved with the plot type's default, as JSON
    payload = window._style_payload()
    assert json.loads(json.dumps(payload))["line_overrides"][key]["linestyle"] == "-."
    # the scale control reaches the canvas
    window.output_line_scale_spin.setValue(2.0)
    assert window.canvas.style_settings()["line_width_scale"] == 2.0
    window._on_reset_lines_clicked()
    assert window.canvas.style_settings()["line_overrides"] == {}
    assert window.output_line_scale_spin.value() == 1.0


def test_apply_to_all_pushes_named_line_styles(qapp):
    from opennmr.gui.main_window import MainWindow
    window = MainWindow()
    window.canvas_tabs.setCurrentWidget(window.canvas)
    window.canvas.set_line_style("label:Fit", linestyle="--")
    window._on_apply_style_to_all_clicked()
    assert window.fit_canvas.style_settings()["line_overrides"] == {"label:Fit": {"linestyle": "--"}}
