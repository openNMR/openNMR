"""2026-11-12, explicit user requests:

* Derived spectra's names grew by " · 23Na · derived · procno 1" with
  every derivation and every reopening of a saved state ("VOCS sum (20
  spectra) · 23Na · derived · procno 1 · 23Na · derived · procno 1 …"),
  long enough to break saving a state on Windows. The suffix the list
  adds is now dropped from a name before it is added again.
* Measure's arrows, markers and labels take a colour, line width and
  text size: right-click any plot's Measure button, or Settings ▸
  Interface ▸ Measure. Every plot follows, and the choice is remembered.
"""

import matplotlib
import numpy as np
import pytest

from opennmr.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
from opennmr.gui.session import Session, strip_label_suffix
from opennmr.gui.widgets.canvas import MplCanvas

from test_gui_canvas import qapp, _make_1d, _press_at  # noqa: F401


# ---- derived labels ------------------------------------------------------------------------

@pytest.mark.parametrize("name, expected", [
    ("VOCS sum (20 spectra) · 23Na · derived · procno 1 · 23Na · derived · procno 1",
     "VOCS sum (20 spectra)"),
    ("VOCS(248 · 23Na · bruker · procno 1) · 23Na · derived · procno 1 · 23Na · derived · procno 1",
     "VOCS(248 · 23Na · bruker · procno 1)"),         # the parent, in brackets, stays
    ("CPMG 12 · 1H · derived", "CPMG 12"),
    ("Simulated CSA", "Simulated CSA"),
    ("D = 2.50e-10 m^2/s", "D = 2.50e-10 m^2/s"),
    ("derived", "derived"),                            # never stripped to nothing
    ("Ratio · bruker data", "Ratio · bruker data"),    # not a suffix
])
def test_strip_label_suffix(name, expected):
    assert strip_label_suffix(name) == expected


def _spectrum(nucleus="23Na"):
    ppm = np.linspace(10, 0, 64)
    return NMRSpectrum(np.ones(64), [NMRAxis("frequency", ppm=ppm, hz=ppm * 132.0)],
                       NMRMetadata(vendor="bruker", nucleus=nucleus, sfo1=132.0, procno=1))


def test_deriving_again_and_again_does_not_grow_the_label():
    session = Session()
    index = session.add_derived(_spectrum(), "VOCS sum (20 spectra)")
    first = session.label(index)
    assert first == "VOCS sum (20 spectra) · 23Na · derived · procno 1"
    for _ in range(4):                      # as a state reopened four times names it
        index = session.add_derived(_spectrum(), session.label(index))
    assert session.label(index) == first


def test_a_label_saved_before_the_fix_is_shown_cleanly():
    session = Session()
    index = session.add_derived(_spectrum(), "x")
    spectrum = session.spectra[index]
    # As an older session or state stored it, suffixes and all.
    spectrum.metadata.source_path = ("VOCS sum (20 spectra) · 23Na · derived · procno 1"
                                     " · 23Na · derived · procno 1")
    assert session.label(index) == "VOCS sum (20 spectra) · 23Na · derived · procno 1"


def test_a_user_name_still_wins():
    session = Session()
    index = session.add_derived(_spectrum(), "x · 23Na · derived · procno 1")
    session.spectra[index].metadata.raw["display_label"] = "My sum"
    assert session.label(index) == "My sum"


def test_the_plot_legend_name_matches(qapp):
    from opennmr.gui.widgets.canvas import _spectrum_label
    session = Session()
    index = session.add_derived(_spectrum(), "VOCS sum · 23Na · derived · procno 1")
    assert _spectrum_label(session.spectra[index]) == "VOCS sum (23Na)"


# ---- Measure style -------------------------------------------------------------------------

def _measured(canvas):
    canvas.figure.set_size_inches(8, 6)
    canvas.show_spectrum(_make_1d())
    canvas.canvas.draw()
    canvas.set_measurement_mode(True)
    canvas._on_press(_press_at(canvas, canvas.ax, xfrac=0.3, yfrac=0.5))
    pending = [a for a in canvas._measurement_artists if isinstance(a, matplotlib.lines.Line2D)]
    canvas._on_press(_press_at(canvas, canvas.ax, xfrac=0.7, yfrac=0.5))
    return pending


def _drawn(canvas):
    arrow = next(a for a in canvas._measurement_artists
                 if isinstance(a, matplotlib.text.Annotation) and a.arrow_patch is not None)
    label = next(a for a in canvas._measurement_artists
                 if isinstance(a, matplotlib.text.Text) and a.get_text())
    return arrow, label


def test_defaults_are_as_before(qapp):
    canvas = MplCanvas()
    assert canvas.measurement_style() == ("tab:red", 1.4, 10.0)
    _measured(canvas)
    arrow, label = _drawn(canvas)
    assert arrow.arrow_patch.get_linewidth() == pytest.approx(1.4)
    assert label.get_fontsize() == pytest.approx(10.0)


def test_new_measurements_take_the_style(qapp):
    canvas = MplCanvas()
    canvas.set_measurement_style("#00aaff", 3.0, 16.0)
    pending = _measured(canvas)
    arrow, label = _drawn(canvas)
    blue = matplotlib.colors.to_rgba("#00aaff")
    assert matplotlib.colors.to_rgba(arrow.arrow_patch.get_edgecolor()) == pytest.approx(blue)
    assert arrow.arrow_patch.get_linewidth() == pytest.approx(3.0)
    assert label.get_fontsize() == pytest.approx(16.0)
    assert matplotlib.colors.to_rgba(label.get_color()) == pytest.approx(blue)
    # The label's box is drawn at 90 % opacity: the colour, not the alpha.
    assert matplotlib.colors.to_rgba(label.get_bbox_patch().get_edgecolor())[:3] == \
        pytest.approx(blue[:3])
    assert pending and pending[0].get_markersize() > 11.0         # the "+" grows too


def test_measurements_on_the_plot_are_restyled_at_once(qapp):
    canvas = MplCanvas()
    _measured(canvas)
    canvas.set_measurement_style("green", 2.5, 20.0)
    arrow, label = _drawn(canvas)
    assert arrow.arrow_patch.get_linewidth() == pytest.approx(2.5)
    assert label.get_fontsize() == pytest.approx(20.0)


def test_bad_values_are_ignored(qapp):
    canvas = MplCanvas()
    canvas.set_measurement_style("not a colour", 999.0, "big")
    assert canvas.measurement_style() == ("tab:red", 1.4, 10.0)


def test_button_menu_choice_is_announced(qapp):
    # The step after the menu (which is modal, so not opened here).
    canvas = MplCanvas()
    heard = []
    canvas.measurementStyleChanged.connect(lambda *a: heard.append(a))
    canvas._apply_measure_menu_style(("#123456", 2.0, 12.0))
    assert heard == [("#123456", 2.0, 12.0)]
    assert canvas.measurement_style() == ("#123456", 2.0, 12.0)


def test_settings_apply_to_every_plot_and_are_remembered(qapp):
    from opennmr.gui.main_window import MainWindow
    window = MainWindow()
    window._set_measure_style("#ff00ff", 2.2, 14.0)
    assert all(c.measurement_style() == ("#ff00ff", 2.2, 14.0) for c in window._all_canvases())
    assert window._ui_settings["measure_color"] == "#ff00ff"
    assert window.measure_width_spin.value() == pytest.approx(2.2)
    # A plot's own button menu reaches the others and the settings.
    window.canvas.measurementStyleChanged.emit("#00ff00", 1.0, 9.0)
    assert window.fit_canvas.measurement_style() == ("#00ff00", 1.0, 9.0)
    later = MainWindow()                                       # same settings file (conftest)
    assert later.canvas.measurement_style() == ("#00ff00", 1.0, 9.0)
