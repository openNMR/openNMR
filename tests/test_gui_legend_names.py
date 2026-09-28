"""Legend names follow the experiment's name, can be typed in Plot Style,
may contain LaTeX, and Compare shows a legend by default (explicit user
request, 2026-09-28)."""
import matplotlib
import pytest
from PySide6.QtWidgets import QApplication, QLineEdit, QMessageBox

from opennmr.gui.main_window import MainWindow
from opennmr.plotting import label_problem, safe_label
from tests.test_gui import _make_1d_dataset


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(qapp, tmp_path):
    w = MainWindow()
    w._load(_make_1d_dataset(tmp_path, name="alpha"), vendor="bruker")
    w._load(_make_1d_dataset(tmp_path, name="beta"), vendor="bruker")
    w.spectrum_list.setCurrentRow(0)
    return w


def _legend_texts(canvas):
    legends = [ax.get_legend() for ax in canvas.figure.get_axes() if ax.get_legend()]
    return [t.get_text() for legend in legends for t in legend.get_texts()]


def _draw(canvas):
    canvas.canvas.draw()          # raises if any label fails to render


# -- the label helpers ---------------------------------------------------------------

@pytest.mark.parametrize("text", ["plain", "folder_with_underscores", r"$^{7}$Li",
                                  r"LiCoO$_2$ at 25 $^{\circ}$C", r"$T_1$-filtered", r"costs \$5"])
def test_good_names_have_no_problem_and_pass_unchanged(text):
    assert label_problem(text) is None
    assert safe_label(text) == text


@pytest.mark.parametrize("text", [r"$^{7$Li", r"$\notacommand$", "one $ sign"])
def test_bad_names_are_explained(text):
    assert label_problem(text)


def test_unparseable_latex_is_drawn_literally_not_crashing():
    fig = matplotlib.figure.Figure()
    ax = fig.add_subplot()
    ax.plot([0, 1], label=safe_label(r"$^{7$Li"))
    ax.legend()
    fig.canvas.draw()                    # would raise with the raw label
    assert ax.get_legend().get_texts()[0].get_text() == r"\$^{7\$Li"


# -- Spectrum view -----------------------------------------------------------------

def test_renamed_experiment_is_the_legend_name(window):
    window._set_display_label(0, r"$^{7}$Li, sample A")
    window.canvas.set_legend_visible(True)
    assert _legend_texts(window.canvas) == [r"$^{7}$Li, sample A"]
    _draw(window.canvas)
    # Renaming again with the legend already up updates it in place.
    window._set_display_label(0, "sample B")
    assert _legend_texts(window.canvas) == ["sample B"]
    assert window.spectrum_list.item(0).text() == "sample B"
    # Blank restores the derived name.
    window._set_display_label(0, "")
    assert "alpha" in _legend_texts(window.canvas)[0]


def test_legend_uses_the_name_after_reselecting(window):
    window._set_display_label(1, "renamed beta")
    window.spectrum_list.setCurrentRow(1)
    window.canvas.set_legend_visible(True)
    assert _legend_texts(window.canvas) == ["renamed beta"]


def test_spectrum_view_offers_its_name_in_plot_style(window):
    window.canvas_tabs.setCurrentWidget(window.canvas)
    series = window.canvas.series()
    assert len(series) == 1 and series[0].can_rename
    window._on_series_renamed(series[0].key, r"LiCoO$_2$")
    # Typed in Plot Style = renamed the experiment: list and legend agree.
    assert window.session.label(0) == r"LiCoO$_2$"
    assert window.spectrum_list.item(0).text() == r"LiCoO$_2$"
    window.canvas.set_legend_visible(True)
    assert _legend_texts(window.canvas) == [r"LiCoO$_2$"]


def test_plot_style_refuses_broken_latex(window, monkeypatch):
    warned = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warned.append(a[2]))
    window.canvas_tabs.setCurrentWidget(window.canvas)
    key = window.canvas.series()[0].key
    before = window.session.label(0)
    window._on_series_renamed(key, r"$^{7$Li")
    assert warned and "LaTeX" in warned[0]
    assert window.session.label(0) == before


def test_rename_dialog_reprompts_on_broken_latex(window, monkeypatch):
    answers = iter([(r"$^{7$Li", True), (r"$^{7}$Li", True)])
    monkeypatch.setattr("opennmr.gui.main_window.QInputDialog.getText",
                        lambda *a, **k: next(answers))
    warned = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warned.append(a[2]))
    window._on_rename_spectrum_action(0)
    assert len(warned) == 1
    assert window.session.label(0) == r"$^{7}$Li"


def test_trace_colour_still_colours_a_single_spectrum(window):
    window.canvas.set_style(trace_color="#ff0000")
    window.canvas.show_spectrum(window.session.spectra[0])
    assert matplotlib.colors.to_hex(window.canvas.ax.lines[0].get_color()) == "#ff0000"


# -- Compare -----------------------------------------------------------------------

def _compare(window, mode_index=0):
    window._rebuild_compare_list([(0, None), (1, None)], preserve_previous=False)
    window.layout_mode_combo.setCurrentIndex(mode_index)
    window._refresh_compare_view(switch_tab=True, force_render=True)


@pytest.mark.parametrize("mode_index", [0, 1])   # stacked, overlaid
def test_compare_has_a_legend_by_default_naming_every_trace(window, mode_index):
    window._set_display_label(0, r"$^{7}$Li pristine")
    window._set_display_label(1, r"$^{7}$Li cycled")
    _compare(window, mode_index)
    assert _legend_texts(window.compare_canvas) == [r"$^{7}$Li pristine", r"$^{7}$Li cycled"]
    _draw(window.compare_canvas)
    # Exactly one legend, even with one axes per trace.
    assert sum(1 for ax in window.compare_canvas.figure.get_axes() if ax.get_legend()) == 1


def test_compare_legend_follows_a_rename(window):
    _compare(window)
    window._set_display_label(1, "after rename")
    window._refresh_compare_view(switch_tab=True, force_render=True)
    assert _legend_texts(window.compare_canvas)[1] == "after rename"
    assert window.compare_list.item(1).text() == "after rename"


def test_compare_name_typed_in_plot_style_renames_the_experiment(window):
    _compare(window)
    key = window.compare_canvas.series()[0].key
    assert window.compare_canvas.series()[0].can_rename
    window._on_series_renamed(key, "typed in Plot Style")
    assert window.session.label(0) == "typed in Plot Style"
    assert _legend_texts(window.compare_canvas)[0] == "typed in Plot Style"
    assert window.compare_list.item(0).text() == "typed in Plot Style"


def test_compare_legend_can_still_be_turned_off(window):
    _compare(window)
    window.compare_canvas.set_legend_visible(False)
    assert _legend_texts(window.compare_canvas) == []
    window._refresh_compare_view(switch_tab=True, force_render=True)
    assert _legend_texts(window.compare_canvas) == []


def test_compare_colour_set_in_plot_style_reaches_the_overlay(window):
    _compare(window, 1)
    key = window.compare_canvas.series()[1].key
    window.compare_canvas.set_series_style(key, colour="#00aa00")
    assert matplotlib.colors.to_hex(window.compare_canvas.ax.lines[1].get_color()) == "#00aa00"


def test_restyling_keeps_every_stack_entry(window):
    _compare(window)
    window.compare_canvas.set_style(legend_fontsize=14.0)
    texts = _legend_texts(window.compare_canvas)
    assert len(texts) == 2


def test_notebook_stacked_plot_accepts_latex_names():
    import numpy as np
    from opennmr.notebook import plot_stacked_spectra, label_problem as nb_problem
    spectra = [{"x_ppm": np.linspace(10, 0, 50), "normalised_real_spectrum": np.random.rand(50)}
               for _ in range(2)]
    ax = plot_stacked_spectra(spectra, labels=[r"$^{7}$Li", r"$^{7$Li"], fname=False)
    ax.legend()
    fig = ax.figure
    fig.canvas.draw()                  # the broken name would raise here
    assert [t.get_text() for t in ax.get_legend().get_texts()] == [r"$^{7}$Li", r"\$^{7\$Li"]
    assert nb_problem(r"$^{7$Li")
    matplotlib.pyplot.close(fig)
