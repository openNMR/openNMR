"""Open experiments: Remove removes every selected spectrum; and plot
style follows the kind of plot on show (explicit user requests,
2026-09-30)."""
import numpy as np
import pytest

from opennmr.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
from opennmr.gui.main_window import MainWindow
from opennmr.gui.widgets.canvas import MplCanvas

from test_gui import qapp  # noqa: F401 -- fixture


def _peak(x0):
    x = np.linspace(0, 10, 401)
    return NMRSpectrum(data=np.exp(-((x - x0) / 0.1) ** 2),
                       axes=[NMRAxis(kind="frequency", ppm=x, hz=x * 400.0)],
                       metadata=NMRMetadata(vendor="simulated", nucleus="1H", sfo1=400.0))


def _window(n=5):
    w = MainWindow()
    for k in range(n):
        w._add_derived_spectrum(_peak(1.0 + k), f"p{k}")
    return w


def test_remove_removes_every_selected_spectrum(qapp):  # noqa: F811
    w = _window()
    w.spectrum_list.clearSelection()
    for row in (1, 3, 4):
        w.spectrum_list.item(row).setSelected(True)
    kept = [w.session.spectra[0], w.session.spectra[2]]
    w._on_remove_clicked()
    assert len(w.session) == 2 and w.spectrum_list.count() == 2
    assert w.session.spectra[0] is kept[0] and w.session.spectra[1] is kept[1]
    assert "Removed 3 spectra" in w.statusBar().currentMessage()


def test_remove_with_one_selected_still_removes_just_it(qapp):  # noqa: F811
    w = _window(3)
    w.spectrum_list.clearSelection()
    w.spectrum_list.setCurrentRow(1)
    target = w.session.spectra[1]
    w._on_remove_clicked()
    assert len(w.session) == 2 and all(s is not target for s in w.session.spectra)


@pytest.mark.parametrize("method,args,kind", [
    ("show_slice_fit", lambda: (np.linspace(0, 10, 50), np.ones(50), np.ones(50)), "1d"),
    ("show_2d_overlay", None, "2d"),
])
def test_views_announce_their_kind(qapp, method, args, kind):  # noqa: F811
    canvas = MplCanvas()
    seen = []
    canvas.plotKindChanged.connect(seen.append)
    canvas._enter_plot_kind("curve")                 # something else was shown before
    if method == "show_2d_overlay":
        ppm1, ppm2 = np.linspace(0, 10, 20), np.linspace(0, 150, 30)
        two_d = NMRSpectrum(
            data=np.random.default_rng(0).random((20, 30)),
            axes=[NMRAxis(kind="frequency", ppm=ppm1, hz=ppm1 * 400),
                  NMRAxis(kind="frequency", ppm=ppm2, hz=ppm2 * 100)],
            metadata=NMRMetadata(vendor="simulated", nucleus="13C", sfo1=100.0))
        try:
            canvas.show_2d_overlay([two_d, two_d])
        except Exception:  # noqa: BLE001 -- the announcement comes first, before drawing
            pass
    else:
        getattr(canvas, method)(*args())
    assert seen[-1] == kind and canvas.plot_kind() == kind


def test_per_type_default_follows_the_plot_on_show(qapp):  # noqa: F811
    w = _window(1)
    w._on_plot_style_clicked()
    combo = w.output_type_combo
    # A spectrum on the Spectrum view: "1D spectrum".
    w.canvas_tabs.setCurrentWidget(w.canvas)
    w.canvas.show_spectrum(w.session.spectra[0])
    w._refresh_style_scope_label()
    assert combo.currentData() == "1d"
    assert "1D spectrum" in w.style_scope_label.text()
    # A curve on the Fit view: "Curves & profiles", as soon as it is drawn.
    w.canvas_tabs.setTabVisible(w.canvas_tabs.indexOf(w.fit_canvas), True)
    w.canvas_tabs.setCurrentWidget(w.fit_canvas)
    w.fit_canvas.show_external_plot(lambda fig: fig.add_subplot().plot([0, 1], [0, 1]),
                                    kind="curve")
    assert combo.currentData() == "curve"
    # A fit of a spectrum on the same view: back to "1D spectrum".
    w.fit_canvas.show_slice_fit(np.linspace(0, 10, 50), np.ones(50), np.ones(50))
    assert combo.currentData() == "1d"
