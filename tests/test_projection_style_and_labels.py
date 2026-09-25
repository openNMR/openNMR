"""Right-click styling of a 2D's projection strips (each on its own), and
editing a 2D's F2 and F1 axis labels separately -- plotting, notebook
and GUI."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtWidgets import QDialog  # noqa: E402

from opennmr.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum  # noqa: E402
from opennmr.gui.widgets import canvas as canvas_module  # noqa: E402
from opennmr.notebook import two_d as td  # noqa: E402
from opennmr.plotting import twod  # noqa: E402

from test_gui import qapp  # noqa: F401,E402 -- fixture
from test_two_d import _make_dataset  # noqa: E402
from test_external_projection import _hsqc, _one_d, _window_with  # noqa: E402


def _colour(line):
    return matplotlib.colors.to_hex(line.get_color())


# --- plotting -------------------------------------------------------------
def test_projection_style_sets_each_strip_on_its_own():
    fig, ax, proj = twod.plot_2d_contour(
        _hsqc(), projection_style={"f1": {"color": "navy", "linewidth": 0.5, "linestyle": "--"}})
    f1, f2 = proj["f1"].lines[0], proj["f2"].lines[0]
    assert _colour(f1) == matplotlib.colors.to_hex("navy")
    assert f1.get_linewidth() == 0.5 and f1.get_linestyle() == "--"
    assert _colour(f2) == "#000000"  # untouched: the computed projection's black
    plt.close(fig)


def test_projection_style_colour_overrides_an_external_projections_crimson():
    fig, ax, proj = twod.plot_2d_contour(_hsqc(), f2_external=_one_d("1H"),
                                         projection_style={"f2": {"color": "black"}})
    assert _colour(proj["f2"].lines[0]) == "#000000"
    plt.close(fig)


def test_projection_style_size_and_visibility():
    fig, ax, proj = twod.plot_2d_contour(_hsqc(), square_aspect=False,
                                         projection_style={"f2": {"size": 2.0}})
    fig.canvas.draw()
    big = proj["f2"].get_window_extent().height
    plt.close(fig)
    fig, ax, proj = twod.plot_2d_contour(_hsqc(), square_aspect=False)
    fig.canvas.draw()
    assert big == pytest.approx(2 * proj["f2"].get_window_extent().height, rel=0.02)
    plt.close(fig)
    fig, ax, proj = twod.plot_2d_contour(_hsqc(), projection_style={"f1": {"visible": False}})
    assert proj["f1"] is None and proj["f2"] is not None
    plt.close(fig)


def test_projection_style_refuses_a_misspelt_key():
    with pytest.raises(ValueError, match="colour"):
        twod.plot_2d_contour(_hsqc(), projection_style={"f2": {"colour": "red"}})


def test_notebook_plotter_takes_the_same_projection_style():
    ds = _make_dataset(nuc1="1H", nuc2="13C")
    fig, ax = td.NMR2Dplotter(ds, field="", projection_style={
        "f2": {"color": "teal", "linewidth": 2.5}, "f1": {"visible": False}})
    strips = [a for a in fig.axes if a is not ax]
    assert len(strips) == 1  # F1 hidden
    [line] = strips[0].lines
    assert _colour(line) == matplotlib.colors.to_hex("teal") and line.get_linewidth() == 2.5
    plt.close(fig)


# --- GUI: projection style --------------------------------------------------
def test_projection_style_is_kept_through_redraws_and_the_next_2d(qapp):
    window = _window_with(qapp, [(_hsqc(), "HSQC"), (_hsqc(), "HSQC 2")])
    canvas = window.canvas
    canvas.set_projection_style("f1", color="#ff0000", linewidth=0.7)
    assert _colour(canvas._proj_axes["f1"].lines[0]) == "#ff0000"
    assert _colour(canvas._proj_axes["f2"].lines[0]) == "#000000"
    canvas._redraw_contour(preserve_view=True)
    window._select_single_row(1)
    qapp.processEvents()
    assert canvas._proj_axes["f1"].lines[0].get_linewidth() == 0.7
    canvas.set_projection_style("f1", color=None)
    assert _colour(canvas._proj_axes["f1"].lines[0]) == "#000000"
    assert canvas.projection_style("f1") == {"linewidth": 0.7}
    with pytest.raises(ValueError):
        canvas.set_projection_style("f1", colour="red")


def test_hiding_a_strip_and_dropping_onto_it_brings_it_back(qapp):
    window = _window_with(qapp, [(_hsqc(), "HSQC"), (_one_d("1H"), "1H")])
    canvas = window.canvas
    canvas.set_projection_style("f2", visible=False)
    assert canvas._proj_axes["f2"] is None and canvas._proj_axes["f1"] is not None
    assert canvas.projection_drop_target(canvas.canvas.rect().center().toPointF()) is not None
    canvas.set_external_projection("f2", window.session.spectra[1], "1H")
    assert canvas._proj_axes["f2"] is not None


def test_right_click_on_a_strip_is_recognised_as_that_strip(qapp):
    window = _window_with(qapp, [(_hsqc(), "HSQC")])
    canvas = window.canvas

    class _Press:
        button, xdata, ydata, x, y = 3, None, None, None, None

    for dim in ("f2", "f1"):
        press = _Press()
        press.inaxes = canvas._proj_axes[dim]
        canvas._on_press(press)
        assert canvas._projection_dimension_of(canvas._last_right_click_axes) == dim
    press = _Press()
    press.inaxes = None
    canvas._on_press(press)
    assert canvas._projection_dimension_of(canvas._last_right_click_axes) is None


def test_projection_style_dialog_reports_only_what_changed(qapp):
    dialog = canvas_module._ProjectionStyleDialog(None, "f1", {}, default_colour="crimson")
    assert dialog.style() == {}
    assert "crimson (default)" in dialog.colour_button.text()
    dialog.set_colour("#123456")
    dialog.linewidth_spin.setValue(0.4)
    dialog.linestyle_combo.setCurrentIndex(dialog.linestyle_combo.findData(":"))
    dialog.size_spin.setValue(1.5)
    dialog.visible_check.setChecked(False)
    assert dialog.style() == {"color": "#123456", "linewidth": 0.4, "linestyle": ":",
                              "size": 1.5, "visible": False}
    dialog.restore_defaults()
    assert dialog.style() == {}


def test_projection_style_dialog_previews_live_and_cancel_undoes_it(qapp, monkeypatch):
    window = _window_with(qapp, [(_hsqc(), "HSQC")])
    canvas = window.canvas
    seen = {}

    def run(dialog, accept, both=False):
        def exec_():
            dialog.both_check.setChecked(both)
            dialog.set_colour("#00aa00")
            seen["live"] = _colour(canvas._proj_axes["f2"].lines[0])
            return QDialog.DialogCode.Accepted if accept else QDialog.DialogCode.Rejected
        return exec_

    original = canvas_module._ProjectionStyleDialog.__init__

    for accept, both in ((False, False), (True, True)):
        def init(self, *a, _accept=accept, _both=both, **k):
            original(self, *a, **k)
            self.exec = run(self, _accept, _both)
        monkeypatch.setattr(canvas_module._ProjectionStyleDialog, "__init__", init)
        canvas._prompt_projection_style("f2")
        assert seen["live"] == "#00aa00"
        if not accept:
            assert canvas.projection_style("f2") == {}
            assert _colour(canvas._proj_axes["f2"].lines[0]) == "#000000"
        else:
            assert canvas.projection_style("f2") == {"color": "#00aa00"}
            assert canvas.projection_style("f1") == {"color": "#00aa00"}  # both


# --- GUI: axis labels -------------------------------------------------------
def test_changing_f1_leaves_f2_alone(qapp):
    window = _window_with(qapp, [(_hsqc(), "HSQC")])
    canvas = window.canvas
    f2_before = canvas.ax.get_xlabel()
    canvas.set_axis_label_override("f1", r"$\delta$ $^{13}$C (ppm) {nuc}")
    assert canvas.ax.get_xlabel() == f2_before
    # Braces are mathtext, kept; only the named placeholders are filled.
    assert canvas.ax.get_ylabel() == r"$\delta$ $^{13}$C (ppm) 13C"
    canvas.set_axis_label_override("f1", "")
    assert canvas.ax.get_ylabel() != "" and "13" in canvas.ax.get_ylabel()
    assert canvas.axis_label_override("f1") is None


def test_label_survives_redraw_export_and_reselection_and_is_per_spectrum(qapp):
    window = _window_with(qapp, [(_hsqc(), "HSQC"), (_hsqc(), "HSQC 2")])
    canvas = window.canvas
    canvas.set_axis_label_override("f2", "Proton shift (ppm)")
    canvas._redraw_contour(preserve_view=False)
    canvas.show_spectrum(canvas._contour_spectrum)  # what a full-resolution export does
    assert canvas.ax.get_xlabel() == "Proton shift (ppm)"
    window._select_single_row(1)
    qapp.processEvents()
    assert canvas.ax.get_xlabel() != "Proton shift (ppm)"
    window._select_single_row(0)
    qapp.processEvents()
    assert canvas.ax.get_xlabel() == "Proton shift (ppm)"


def test_a_label_without_unit_is_not_shown_in_another_unit(qapp):
    window = _window_with(qapp, [(_hsqc(), "HSQC")])
    canvas = window.canvas
    canvas.set_axis_label_override("f2", "Proton shift (ppm)")
    canvas.set_axis_label_override("f1", "Carbon ({unit})")
    canvas.set_axis_unit("hz")
    assert "ppm" not in canvas.ax.get_xlabel()  # automatic Hz label instead
    assert canvas.ax.get_ylabel() == "Carbon (Hz)"
    canvas.set_axis_unit("ppm")
    assert canvas.ax.get_xlabel() == "Proton shift (ppm)"
    assert canvas.ax.get_ylabel() == "Carbon (ppm)"


def test_axis_label_dialog_keeps_an_untouched_axis_automatic(qapp, monkeypatch):
    window = _window_with(qapp, [(_hsqc(), "HSQC")])
    canvas = window.canvas
    original = canvas_module._AxisLabelsDialog.__init__

    def init(self, *a, **k):
        original(self, *a, **k)
        assert self.edits["f2"].text() == ""  # automatic = blank
        assert self.previews["f2"].pixmap() is not None and not self.previews["f2"].pixmap().isNull()

        def exec_():
            self.edits["f1"].setText("Indirect")
            return QDialog.DialogCode.Accepted
        self.exec = exec_
    monkeypatch.setattr(canvas_module._AxisLabelsDialog, "__init__", init)
    canvas._prompt_axis_labels()
    assert canvas.axis_label_override("f2") is None  # untouched: still follows Plot Style
    assert canvas.axis_label_override("f1")["text"] == "Indirect"
    assert canvas.ax.get_ylabel() == "Indirect"


def test_pseudo_2d_has_no_f1_label_to_edit(qapp, monkeypatch):
    window = _window_with(qapp, [(_hsqc(pseudo=True), "T1")])
    canvas = window.canvas
    original = canvas_module._AxisLabelsDialog.__init__
    seen = {}

    def init(self, *a, **k):
        original(self, *a, **k)
        seen["f1_enabled"] = self.edits["f1"].isEnabled()
        self.exec = lambda: QDialog.DialogCode.Rejected
    monkeypatch.setattr(canvas_module._AxisLabelsDialog, "__init__", init)
    canvas._prompt_axis_labels()
    assert seen["f1_enabled"] is False


def test_removing_the_2d_forgets_its_labels(qapp):
    window = _window_with(qapp, [(_hsqc(), "HSQC"), (_one_d("1H"), "1H")])
    canvas = window.canvas
    canvas.set_axis_label_override("f1", "Indirect")
    hsqc = window.session.spectra[0]
    window.spectrum_list.setCurrentRow(0)
    window._on_remove_clicked()
    assert canvas.axis_label_override("f1", spectrum=hsqc) is None
    assert not canvas._axis_label_overrides


def test_a_label_matplotlib_cannot_draw_is_refused_before_it_reaches_the_plot(qapp):
    window = _window_with(qapp, [(_hsqc(), "HSQC")])
    canvas = window.canvas
    before = canvas.ax.get_ylabel()
    with pytest.raises(ValueError, match="cannot be drawn"):
        canvas.set_axis_label_override("f1", r"$\notacommand$ shift")
    assert canvas.ax.get_ylabel() == before
    canvas.canvas.draw()  # the plot still draws

    dialog = canvas_module._AxisLabelsDialog(None, f2_text="", f1_text="", f2_auto="x", f1_auto="y")
    ok = dialog.buttons.button(dialog.buttons.StandardButton.Ok)
    dialog.edits["f1"].setText(r"$\notacommand$")
    assert not ok.isEnabled() and "Cannot be drawn" in dialog.previews["f1"].text()
    dialog.edits["f1"].setText(r"$\delta$ $^{13}$C (ppm)")
    assert ok.isEnabled()


def test_right_click_menu_on_a_strip_offers_its_style_first(qapp, monkeypatch):
    window = _window_with(qapp, [(_hsqc(), "HSQC")])
    canvas = window.canvas
    menus = []
    # The canvas's own seam: patching QMenu.exec opens a real menu anyway.
    monkeypatch.setattr(canvas_module.MplCanvas, "_exec_context_menu",
                        lambda self, menu, pos: menus.append([a.text() for a in menu.actions()]))

    class _Press:
        button, xdata, ydata, x, y = 3, None, None, None, None
    press = _Press()
    press.inaxes = canvas._proj_axes["f1"]
    canvas._on_press(press)
    canvas._on_context_menu(canvas.canvas.rect().center())
    assert menus[-1][0] == "F1 (left) Projection Style…"
    assert "Hide F1 (left) Projection" in menus[-1]
    assert "Edit Axis Labels (F2 / F1)…" in menus[-1]
    press.inaxes = canvas.ax
    canvas._on_press(press)
    canvas.set_projection_style("f2", visible=False)
    canvas._on_context_menu(canvas.canvas.rect().center())
    assert not any("Projection Style" in t for t in menus[-1])
    assert "Show F2 Projection" in menus[-1]
