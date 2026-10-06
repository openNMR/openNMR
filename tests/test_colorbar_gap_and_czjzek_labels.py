"""2026-10-01: italic C_Q, eta_Q and P on Czjzek maps; a colour-bar gap
(Plot Style ▸ Colour bar gap; notebook ``plotting.set_colorbar_gap``)
for Czjzek maps and tensor surfaces."""
import numpy as np
import pytest
from PySide6.QtWidgets import QMessageBox

from opennmr import plotting
from opennmr.gui.main_window import MainWindow
from opennmr.gui.widgets.canvas import MplCanvas
from opennmr.notebook import czjzek as cz

from test_gui import qapp  # noqa: F401 -- fixture


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    for name in ("warning", "information", "critical"):
        monkeypatch.setattr(QMessageBox, name, lambda *a, **k: None)
    monkeypatch.setattr(MainWindow, "_FIGURE_STYLES_PATH", str(tmp_path / "figure_styles.json"))


_X = np.linspace(-50, 50, 300)


def _czjzek_draw(fig):
    cz.plot_czjzek_distribution({"sigma": 1.0, "CQ0": 2.0, "eta0": 0.3, "label": "A"},
                                spectrum={"x": _X, "y": np.exp(-_X ** 2 / 200)}, figure=fig,
                                n_cq=60, n_eta=30)


def _map_gap_pt(fig):
    m, bar = fig.czjzek_axes["map"], fig.czjzek_axes["colorbar"]
    return (bar.get_position().x0 - m.get_position().x1) * fig.get_figwidth() * 72.0


def test_czjzek_symbols_are_italic():
    # User request: "the C of C_Q should be italicised, as should the eta
    # of eta_Q, and the P of P(C_Q, eta_Q)".
    import matplotlib.pyplot as plt
    fig = plt.figure()
    _czjzek_draw(fig)
    axes = fig.czjzek_axes
    assert axes["map"].get_xlabel() == r"$\mathit{C}_\mathrm{Q}$ (MHz)"
    assert axes["map"].get_ylabel() == r"$\mathit{\eta}_\mathrm{Q}$"
    assert axes["colorbar"].get_ylabel().startswith(
        r"$\mathit{P}(\mathit{C}_\mathrm{Q},\,\mathit{\eta}_\mathrm{Q})$")
    legend_texts = [t.get_text() for t in axes["map"].get_legend().get_texts()]
    assert all(r"\mathit" in t for t in legend_texts)
    # \mathit really selects the italic face under the package's
    # mathtext settings (mathtext.default = "regular" sets letters upright).
    from matplotlib.mathtext import MathTextParser
    from opennmr.plotting.style import apply_default_font
    apply_default_font("DejaVu Sans")
    glyphs = MathTextParser("path").parse(r"$\mathit{C}_\mathrm{Q}\mathit{\eta}\mathit{P}$").glyphs
    # glyph tuples gained a field in newer matplotlib; the font is always first.
    styles = [glyph[0].style_name.lower() for glyph in glyphs]
    assert ["italic" in s or "oblique" in s for s in styles] == [True, False, True, True]
    plt.close(fig)


def test_notebook_colorbar_gap_on_a_czjzek_map():
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(9, 4))
    _czjzek_draw(fig)
    fig.canvas.draw()
    spec_before = fig.czjzek_axes["spectrum"].get_position().bounds
    bar_before = fig.czjzek_axes["colorbar"].get_position().bounds
    for gap in (0.0, 6.0, 24.0):
        assert plotting.set_colorbar_gap(fig, gap)
        assert _map_gap_pt(fig) == pytest.approx(gap, abs=0.01)
    # The bar and the spectrum beside it stay where they were.
    assert fig.czjzek_axes["colorbar"].get_position().bounds == pytest.approx(bar_before)
    assert fig.czjzek_axes["spectrum"].get_position().bounds == pytest.approx(spec_before)
    assert plotting.set_colorbar_gap(fig, None) == []       # None/negative: untouched
    plt.close(fig)


def test_plot_style_colorbar_gap_on_a_czjzek_map(qapp):  # noqa: F811
    canvas = MplCanvas(full_resolution=True)
    canvas.figure.set_size_inches(9, 4)
    canvas.show_external_plot(_czjzek_draw, kind="2d")
    auto = _map_gap_pt(canvas.figure)
    canvas.set_style(colorbar_gap=8.0)
    assert _map_gap_pt(canvas.figure) == pytest.approx(8.0, abs=0.05)
    # Kept through a restyle that redraws and re-lays-out the figure.
    canvas.set_style(labelsize=15.0)
    assert _map_gap_pt(canvas.figure) == pytest.approx(8.0, abs=0.05)
    assert canvas.style_settings()["colorbar_gap"] == 8.0
    canvas.set_style(colorbar_gap=-1.0)                      # Auto: the picture's own
    assert _map_gap_pt(canvas.figure) == pytest.approx(auto, rel=0.25)


def _tensor_gap_pt(canvas):
    from mpl_toolkits.mplot3d import proj3d
    ax = [a for a in canvas.figure.axes if a.name == "3d"][0]
    bar = [a for a in canvas.figure.axes if getattr(a, "_colorbar", None) is not None][0]
    M = ax.get_proj()
    xs = [ax.transData.transform(proj3d.proj_transform(x, y, z, M)[:2])[0]
          for x in ax.get_xlim3d() for y in ax.get_ylim3d() for z in ax.get_zlim3d()]
    fig = canvas.figure
    return (bar.get_position().x0 * fig.bbox.width - max(xs)) / fig.dpi * 72.0, bar


def test_plot_style_colorbar_gap_on_a_tensor_surface(qapp):  # noqa: F811
    window = MainWindow()
    window.sim_csa_nucleus_edit.setText("13C")
    window.sim_csa_aniso_spin.setValue(-80.0)
    assert window.sim_csa_tensor_controls.show_tensors()
    window.canvas_tabs.setCurrentWidget(window.simulation_canvas)
    canvas = window.simulation_canvas
    window.output_colorbar_gap_spin.setValue(12.0)
    gap, _bar = _tensor_gap_pt(canvas)
    assert gap == pytest.approx(12.0, abs=0.05)
    # Measured from the box at the current view: kept after rotating.
    ax = [a for a in canvas.figure.axes if a.name == "3d"][0]
    ax.view_init(elev=10.0, azim=80.0)
    canvas._refit_3d_ticks_after_release(type("E", (), {"inaxes": ax})())
    assert _tensor_gap_pt(canvas)[0] == pytest.approx(12.0, abs=0.05)
    # A gap too large for the figure stops the bar, labels included, at its edge.
    window.output_colorbar_gap_spin.setValue(144.0)
    canvas.canvas.draw()
    _gap, bar = _tensor_gap_pt(canvas)
    renderer = canvas.figure.canvas.get_renderer()
    assert bar.get_tightbbox(renderer).x1 <= canvas.figure.bbox.x1 + 0.5
    # "Auto" is shown for the special value, and saved with the per-type default.
    window.output_colorbar_gap_spin.setValue(-1.0)
    assert window.output_colorbar_gap_spin.text() == "Auto"
    window.output_colorbar_gap_spin.setValue(6.0)
    assert window._style_payload()["colorbar_gap"] == 6.0
