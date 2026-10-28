"""Shaped pulses in the pulse-sequence diagram: rounded (default), a rectangle with
a diagonal arrow, or that rectangle with one half shaded -- notebook, Compare panel
and the viewer's option (kept in the window's settings)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mc
import pytest
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle

from opennmr.io.pulse_sequence import parse_pulse_sequence
from opennmr.notebook import compare_panels as cp
from opennmr.notebook import pulse_sequence as ps
from test_pulse_sequence import BRUKER

SHAPED = "#445566"


@pytest.fixture
def seq():
    return parse_pulse_sequence(BRUKER, "bruker", defines={"dec"})[0]


def _shaped_artists(ax):
    rects = [p for p in ax.patches if isinstance(p, Rectangle) and mc.to_hex(p.get_edgecolor()) == SHAPED]
    polys = [p for p in ax.patches if isinstance(p, Polygon) and not isinstance(p, Rectangle)
             and mc.to_hex(p.get_facecolor()) == SHAPED]
    arrows = [c for c in ax.get_children() if isinstance(c, matplotlib.text.Annotation)]
    return rects, polys, arrows


def test_styles_are_listed_and_rounded_is_the_default():
    assert ps.SHAPED_STYLES == ("rounded", "arrow", "shaded")
    assert "SHAPED_STYLES" in ps.__all__


def test_rounded_is_unchanged(seq):
    fig, ax = ps.plot_pulse_sequence(seq, colors={"shaped": SHAPED})
    rects, polys, arrows = _shaped_artists(ax)
    assert len(polys) == 1 and not rects and not arrows


def test_arrow_is_an_open_rectangle_with_a_diagonal_arrow(seq):
    fig, ax = ps.plot_pulse_sequence(seq, colors={"shaped": SHAPED}, shaped_style="arrow")
    rects, polys, arrows = _shaped_artists(ax)
    assert len(rects) == 1 and not polys and len(arrows) == 1
    r = rects[0]
    assert r.get_facecolor()[3] == 0                         # not filled
    (x0, y0), (x1, y1) = arrows[0].xyann, arrows[0].xy      # bottom left -> top right
    assert (x0, y0) == pytest.approx((r.get_x(), r.get_y()))
    assert (x1, y1) == pytest.approx((r.get_x() + r.get_width(), r.get_y() + r.get_height()))
    assert arrows[0].arrowprops["arrowstyle"] == "-|>"


def test_shaded_fills_the_lower_right_half_only(seq):
    fig, ax = ps.plot_pulse_sequence(seq, colors={"shaped": SHAPED}, shaped_style="shaded")
    rects, polys, arrows = _shaped_artists(ax)
    assert len(rects) == 1 and len(polys) == 1 and len(arrows) == 1
    r = rects[0]
    xy = polys[0].get_xy()[:3]
    x0, y0, w, h = r.get_x(), r.get_y(), r.get_width(), r.get_height()
    assert set(map(tuple, xy.round(9))) == {(round(x0, 9), round(y0, 9)), (round(x0 + w, 9), round(y0, 9)),
                                             (round(x0 + w, 9), round(y0 + h, 9))}
    assert arrows[0].arrowprops["arrowstyle"] == "-"        # a plain diagonal, no head


def test_the_shaped_pulse_keeps_its_name_and_height(seq):
    for style in ps.SHAPED_STYLES:
        fig, ax = ps.plot_pulse_sequence(seq, shaped_style=style)
        assert any("6" in t.get_text() for t in ax.texts)    # p6 is still labelled
    a = ps.plot_pulse_sequence(seq, colors={"shaped": SHAPED}, shaped_style="arrow", pulse_height=2.0)[1]
    assert _shaped_artists(a)[0][0].get_height() > _shaped_artists(
        ps.plot_pulse_sequence(seq, colors={"shaped": SHAPED}, shaped_style="arrow")[1])[0][0].get_height()


def test_unknown_style_is_rejected(seq):
    with pytest.raises(ValueError, match="shaped_style"):
        ps.plot_pulse_sequence(seq, shaped_style="triangle")


def test_a_compare_panel_takes_the_style(seq):
    fig, axes, _ = cp.plot_compare_panels([{"pulse_sequence": seq}], layout="row", figsize=(6, 3),
                                          pulse_options={"shaped_style": "shaded",
                                                         "colors": {"shaped": SHAPED}})
    rects, polys, arrows = _shaped_artists(axes[0])
    assert rects and polys and arrows


def test_saved_viewer_option_reaches_compare_and_bad_values_fall_back():
    from opennmr.gui.pulse_sequence_dialog import saved_draw_options
    assert saved_draw_options({"pulse_sequence_options": {"shaped_style": "arrow"}})["shaped_style"] == "arrow"
    assert saved_draw_options({"pulse_sequence_options": {"shaped_style": "zigzag"}})["shaped_style"] == "rounded"
    assert saved_draw_options({})["shaped_style"] == "rounded"


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_viewer_has_the_choice_remembers_it_and_draws_it(qapp, seq):
    from opennmr.gui.pulse_sequence_dialog import PulseSequenceDialog
    settings = {}
    saved = []
    dialog = PulseSequenceDialog(None, settings=settings, save_settings=lambda: saved.append(1))
    assert [dialog.shaped_style_combo.itemData(i) for i in range(3)] == list(ps.SHAPED_STYLES)
    assert dialog.draw_options()["shaped_style"] == "rounded"
    dialog.shaped_style_combo.setCurrentIndex(dialog.shaped_style_combo.findData("shaded"))
    assert dialog.draw_options()["shaped_style"] == "shaded"
    assert settings["pulse_sequence_options"]["shaped_style"] == "shaded" and saved
    # a new viewer starts from what was saved
    again = PulseSequenceDialog(None, settings=settings)
    assert again.shaped_style_combo.currentData() == "shaded"
    options = {k: v for k, v in again.draw_options().items() if k != "show_names"}
    options["colors"] = {**options["colors"], "shaped": SHAPED}
    fig, ax = ps.plot_pulse_sequence(seq, **options)
    rects, polys, arrows = _shaped_artists(ax)
    assert rects and polys and arrows                       # drawn shaded, as saved
