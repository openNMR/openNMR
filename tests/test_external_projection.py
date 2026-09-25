"""A separately acquired 1D spectrum as a 2D contour's F1/F2 projection:
the shared check (notebook.two_d.check_external_projection), the
drawing (plotting.twod.plot_2d_contour's f1_external/f2_external), and
the GUI's drag-and-drop from Open experiments onto the plot, with its
nucleus-mismatch warning."""
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtCore import QPointF, Qt  # noqa: E402
from PySide6.QtGui import QDropEvent  # noqa: E402
from PySide6.QtWidgets import QApplication, QInputDialog, QMessageBox  # noqa: E402

from opennmr.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum  # noqa: E402
from opennmr.gui.main_window import MainWindow  # noqa: E402
from opennmr.notebook import canonical_nucleus, check_external_projection  # noqa: E402
from opennmr.notebook import two_d as td  # noqa: E402
from opennmr.plotting import twod  # noqa: E402

from test_gui import qapp  # noqa: F401,E402 -- fixture


def _hsqc(nuc_direct="1H", nuc_indirect="13C", pseudo=False, homonuclear=False):
    x = np.linspace(10, 0, 256)
    y = np.linspace(150, 50, 128) if not pseudo else np.arange(1, 129, dtype=float)
    xx, yy = np.meshgrid(x, y)
    data = 1e6 * np.exp(-((xx - 5) ** 2 / 0.02 + (yy - (100 if not pseudo else 60)) ** 2 / 4))
    f1 = (NMRAxis("index", index=y) if pseudo else NMRAxis("frequency", ppm=y, hz=y * 100))
    return NMRSpectrum(data, [f1, NMRAxis("frequency", ppm=x, hz=x * 400)],
                       NMRMetadata(vendor="bruker", nucleus=nuc_direct,
                                   nucleus_indirect=None if homonuclear else nuc_indirect,
                                   sfo1=400.0, sfo1_indirect=100.0,
                                   pseudo_2d=pseudo, homonuclear=homonuclear))


def _one_d(nucleus="1H", lo=0.0, hi=10.0, peak=5.0, n=1024, height=1.0, sfo1=600.0):
    x = np.linspace(hi, lo, n)
    y = height * np.exp(-((x - peak) ** 2) / 0.001) - 0.01 * height  # dips below 0
    return NMRSpectrum(y, [NMRAxis("frequency", ppm=x, hz=x * sfo1)],
                       NMRMetadata(vendor="bruker", nucleus=nucleus, sfo1=sfo1))


# --- the shared check ----------------------------------------------------
@pytest.mark.parametrize("raw, expected", [
    ("13C", "13C"), ("C13", "13C"), ("<13C>", "13C"), (" 1h ", "1H"), ("Proton", "1H"),
    ("Carbon13", "13C"), ("27Al", "27Al"), ("AL27", "27Al"), ("off", None), (None, None), ("", None),
])
def test_canonical_nucleus_reads_every_vendor_spelling(raw, expected):
    assert canonical_nucleus(raw) == expected


def test_check_matches_each_dimension_on_its_own_nucleus():
    hsqc = _hsqc()
    ok = check_external_projection(hsqc, _one_d("1H"), "f2")
    assert ok["allowed"] and ok["nucleus_match"] is True and ok["problems"] == []
    wrong = check_external_projection(hsqc, _one_d("13C", lo=0, hi=160), "f2")
    assert wrong["allowed"] and wrong["nucleus_match"] is False
    assert "13C" in wrong["problems"][0] and "1H" in wrong["problems"][0]
    # A 1H 1D against the 13C axis: refused outright -- wrong nucleus
    # and no overlap, and both are reported.
    refused = check_external_projection(hsqc, _one_d("1H"), "f1")
    assert not refused["allowed"] and len(refused["problems"]) == 2
    carbon = check_external_projection(hsqc, _one_d("C13", lo=40, hi=160, peak=100), "f1")
    assert carbon["nucleus_match"] is True and carbon["problems"] == []


def test_check_uses_the_direct_nucleus_for_a_homonuclear_f1():
    cosy = _hsqc(homonuclear=True)
    assert check_external_projection(cosy, _one_d("1H"), "f1")["nucleus_match"] is True


def test_check_asks_when_a_nucleus_is_not_recorded():
    report = check_external_projection(_hsqc(), _one_d(nucleus=None), "f2")
    assert report["allowed"] and report["nucleus_match"] is None
    assert "Could not confirm" in report["problems"][0]


def test_check_refuses_a_pseudo_2d_f1_and_a_1d_with_no_overlap():
    pseudo = check_external_projection(_hsqc(pseudo=True), _one_d("1H"), "f1")
    assert not pseudo["allowed"] and "slice" in pseudo["problems"][0]
    far = check_external_projection(_hsqc(), _one_d("1H", lo=20, hi=30, peak=25), "f2")
    assert not far["allowed"] and "does not overlap" in far["problems"][-1]


def test_check_notes_partial_coverage_without_asking():
    report = check_external_projection(_hsqc(), _one_d("1H", lo=2.5, hi=7.5), "f2")
    assert report["problems"] == []
    assert report["coverage"] == pytest.approx(0.5, abs=0.01)
    assert "50%" in report["notes"][0]


def test_check_accepts_the_notebooks_own_dicts():
    ds = {"spectrum": np.zeros((4, 4)), "x_ppm": np.linspace(10, 0, 4), "y_ppm": np.linspace(150, 50, 4),
          "NUC1": "1H", "NUC2": "13C", "homonuclear": False, "pseudo_2d": False}
    carbon = {"x_ppm": np.linspace(160, 40, 50), "real_spectrum": np.ones(50), "nuclide": "13C"}
    assert check_external_projection(ds, carbon, "f1")["nucleus_match"] is True
    assert check_external_projection(ds, carbon, "f2")["nucleus_match"] is False


def test_notebook_plotter_warns_on_a_mismatched_external_projection():
    ds = {"spectrum": np.zeros((4, 4)), "x_ppm": np.linspace(10, 0, 4), "y_ppm": np.linspace(150, 50, 4),
          "NUC1": "1H", "NUC2": "13C", "homonuclear": False, "pseudo_2d": False}
    carbon = {"x_ppm": np.linspace(160, 0, 200), "real_spectrum": np.ones(200), "nuclide": "13C"}
    with pytest.warns(UserWarning, match="Nucleus mismatch"):
        td._warn_external_projection(ds, carbon, "f2")
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        td._warn_external_projection(ds, carbon, "f1")
        td._warn_external_projection(ds, "/some/bruker/path", "f2")  # never imported just to check


def test_notebook_plotter_runs_the_check():
    from test_two_d import _make_dataset
    ds = _make_dataset(nuc1="1H", nuc2="13C")
    carbon = {"x_ppm": np.linspace(210, -10, 400), "real_spectrum": np.ones(400), "nuclide": "13C"}
    with pytest.warns(UserWarning, match="Nucleus mismatch"):
        fig, _ = td.NMR2Dplotter(ds, f2_external_projection=carbon, field="")
    plt.close(fig)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        fig, _ = td.NMR2Dplotter(ds, f1_external_projection=carbon, field="")
    plt.close(fig)


# --- drawing -------------------------------------------------------------
def test_external_f2_replaces_the_computed_one_cropped_and_in_its_own_colour():
    one_d = _one_d("1H", lo=-5, hi=15, peak=5)
    one_d.data[0] = 1e3  # a huge line outside the 2D's range: must not set the scale
    fig, ax, proj = twod.plot_2d_contour(_hsqc(), f2_external=one_d, external_projection_color="crimson")
    [line] = proj["f2"].lines
    assert line.get_gid() == "external_projection_f2"
    assert matplotlib.colors.same_color(line.get_color(), "crimson")
    x = line.get_xdata()
    assert x.min() >= 0 and x.max() <= 10
    bottom, top = proj["f2"].get_ylim()
    assert bottom < line.get_ydata().min() and top == pytest.approx(line.get_ydata().max(), rel=0.05)
    # F1 is still the computed projection, in black.
    assert matplotlib.colors.same_color(proj["f1"].lines[0].get_color(), "k")
    plt.close(fig)


def test_external_f1_points_away_from_the_contour_like_the_computed_one():
    fig, ax, proj = twod.plot_2d_contour(_hsqc(), f1_external=_one_d("13C", lo=40, hi=160, peak=100))
    [line] = proj["f1"].lines
    left, right = proj["f1"].get_xlim()
    assert left > right  # descending: peaks point left, away from the contour
    assert line.get_ydata().min() >= 50 and line.get_ydata().max() <= 150
    plt.close(fig)


def test_external_f1_is_refused_on_a_pseudo_2d():
    with pytest.raises(ValueError, match="pseudo-2D"):
        twod.plot_2d_contour(_hsqc(pseudo=True), f1_external=_one_d("1H"))


# --- GUI -----------------------------------------------------------------
def _window_with(qapp, spectra):
    window = MainWindow()
    window.resize(1400, 900)
    window.show()
    for spectrum, label in spectra:
        window._add_derived_spectrum(spectrum, label)
    window.spectrum_list.setCurrentRow(0)
    window._select_single_row(0)
    qapp.processEvents()
    window.canvas.canvas.draw()
    return window


def _drop(qapp, window, row, where):
    """Drag Open-experiments ``row`` onto the plot, released over
    ``where``: ``"f2"``/``"f1"`` (that strip's centre) or ``"contour"``."""
    canvas = window.canvas
    ax = canvas._proj_axes[where] if where in ("f1", "f2") else canvas.ax
    bbox = ax.get_window_extent()
    dpr = canvas.canvas.device_pixel_ratio
    pos = QPointF((bbox.x0 + bbox.x1) / 2 / dpr,
                  canvas.canvas.height() - (bbox.y0 + bbox.y1) / 2 / dpr)
    mime = window.spectrum_list.model().mimeData([window.spectrum_list.model().index(row, 0)])
    event = QDropEvent(pos, Qt.DropAction.CopyAction, mime,
                       Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    # Straight into the canvas's drop filter: Qt 6 does not deliver a
    # synthetic QDropEvent through sendEvent without a real drag in
    # progress, and the filter is exactly what a real drop runs.
    assert canvas.eventFilter(canvas.canvas, event) is True
    qapp.processEvents()  # the owner's handler is deferred out of the drag loop
    return event


def test_dropping_a_matching_1d_on_the_top_strip_replaces_the_f2_projection(qapp, monkeypatch):
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: pytest.fail("asked on a match"))
    proton = _one_d("1H")
    window = _window_with(qapp, [(_hsqc(), "HSQC"), (proton, "1H reference")])
    assert window.canvas.projection_drop_target(QPointF(1, 1)) in ("auto", "f1", "f2")
    event = _drop(qapp, window, 1, "f2")
    assert event.isAccepted()
    held = window.canvas.external_projection("f2")
    assert held["spectrum"] is window.session.spectra[1] and held["label"] == window.session.label(1)
    assert window.canvas._proj_axes["f2"].lines[0].get_gid() == "external_projection_f2"
    assert window.canvas.external_projection("f1") is None
    assert "F2 (top)" in window.statusBar().currentMessage()


def test_a_nucleus_mismatch_asks_first_and_no_leaves_it_alone(qapp, monkeypatch):
    asked = {}

    def question(parent, title, text, *args, **kwargs):
        asked["text"] = text
        return QMessageBox.StandardButton.No
    monkeypatch.setattr(QMessageBox, "question", question)
    carbon = _one_d("13C", lo=0, hi=160, peak=100)
    window = _window_with(qapp, [(_hsqc(), "HSQC"), (carbon, "13C CP")])
    _drop(qapp, window, 1, "f2")
    assert "Nucleus mismatch" in asked["text"] and "13C" in asked["text"]
    assert "F1 (left)" in asked["text"]  # points at the strip it does match
    assert window.canvas.external_projection("f2") is None
    assert window.canvas._proj_axes["f2"].lines[0].get_gid() is None


def test_a_nucleus_mismatch_can_still_be_used_when_confirmed(qapp, monkeypatch):
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)
    fluorine = _one_d("19F")
    window = _window_with(qapp, [(_hsqc(), "HSQC"), (fluorine, "19F")])
    _drop(qapp, window, 1, "f2")
    assert window.canvas.external_projection("f2")["spectrum"] is window.session.spectra[1]


def test_dropping_on_the_contour_picks_the_dimension_from_the_nucleus(qapp, monkeypatch):
    monkeypatch.setattr(QInputDialog, "getItem", lambda *a, **k: pytest.fail("asked"))
    carbon = _one_d("13C", lo=40, hi=160, peak=100)
    window = _window_with(qapp, [(_hsqc(), "HSQC"), (carbon, "13C CP")])
    _drop(qapp, window, 1, "contour")
    assert window.canvas.external_projection("f1")["spectrum"] is window.session.spectra[1]
    assert window.canvas.external_projection("f2") is None


def test_homonuclear_drop_on_the_contour_asks_and_both_sets_both(qapp, monkeypatch):
    monkeypatch.setattr(QInputDialog, "getItem", lambda *a, **k: ("Both", True))
    proton = _one_d("1H", lo=-1, hi=11)
    cosy = _hsqc(homonuclear=True)
    cosy.axes[0] = NMRAxis("frequency", ppm=np.linspace(10, 0, 128), hz=np.linspace(10, 0, 128) * 400)
    window = _window_with(qapp, [(cosy, "COSY"), (proton, "1H")])
    _drop(qapp, window, 1, "contour")
    assert window.canvas.external_projection("f1")["spectrum"] is window.session.spectra[1]
    assert window.canvas.external_projection("f2")["spectrum"] is window.session.spectra[1]


def test_pseudo_2d_f1_drop_is_refused_with_a_warning(qapp, monkeypatch):
    warned = {}
    monkeypatch.setattr(QMessageBox, "warning", lambda p, t, text, *a, **k: warned.setdefault("t", text))
    window = _window_with(qapp, [(_hsqc(pseudo=True), "T1 series"), (_one_d("1H"), "1H")])
    _drop(qapp, window, 1, "f1")
    assert "slice" in warned["t"]
    assert window.canvas.external_projection("f1") is None


def test_dropping_a_2d_is_refused(qapp, monkeypatch):
    warned = {}
    monkeypatch.setattr(QMessageBox, "warning", lambda p, t, text, *a, **k: warned.setdefault("t", text))
    window = _window_with(qapp, [(_hsqc(), "HSQC"), (_hsqc(), "HSQC 2")])
    _drop(qapp, window, 1, "f2")
    assert "not a 1D" in warned["t"]
    assert window.canvas.external_projection("f2") is None


def test_file_drops_still_reach_the_main_window(qapp):
    from PySide6.QtCore import QMimeData, QUrl
    window = _window_with(qapp, [(_hsqc(), "HSQC")])
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile("/nowhere")])
    event = QDropEvent(QPointF(50, 50), Qt.DropAction.CopyAction, mime,
                       Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    assert window.canvas.eventFilter(window.canvas.canvas, event) is False
    assert not event.isAccepted()


def test_external_projection_survives_redraws_and_reselection_and_is_removable(qapp, monkeypatch):
    proton = _one_d("1H")
    hsqc = _hsqc()
    window = _window_with(qapp, [(hsqc, "HSQC"), (proton, "1H"), (_hsqc(), "other 2D")])
    _drop(qapp, window, 1, "f2")
    window.canvas._redraw_contour(preserve_view=True)  # e.g. a contour-level scroll
    assert window.canvas._proj_axes["f2"].lines[0].get_gid() == "external_projection_f2"
    window._select_single_row(2)
    qapp.processEvents()
    assert window.canvas.external_projection("f2") is None  # another 2D: its own projections
    window._select_single_row(0)
    qapp.processEvents()
    assert window.canvas.external_projection("f2")["spectrum"] is window.session.spectra[1]
    hsqc = window.session.spectra[0]  # the session holds its own copy
    assert "not computed from this 2D" in window._external_projection_report(hsqc)
    window.canvas.clear_external_projection("f2")
    assert window.canvas.external_projection("f2") is None
    assert window.canvas._proj_axes["f2"].lines[0].get_gid() is None
    assert window._external_projection_report(hsqc) == ""


def test_removing_the_1d_from_open_experiments_removes_its_projection(qapp):
    window = _window_with(qapp, [(_hsqc(), "HSQC"), (_one_d("1H"), "1H")])
    _drop(qapp, window, 1, "f2")
    window.spectrum_list.setCurrentRow(1)
    window._on_remove_clicked()
    window._select_single_row(0)
    qapp.processEvents()
    assert window.canvas.external_projection("f2") is None


def test_external_projection_follows_a_unit_change_through_the_2ds_own_axis(qapp):
    # The 1D was recorded at 600 MHz, the 2D at 400: in Hz they differ,
    # so the conversion must go through the 2D's axis, not the 1D's.
    proton = _one_d("1H", sfo1=600.0)
    window = _window_with(qapp, [(_hsqc(), "HSQC"), (proton, "1H 600")])
    _drop(qapp, window, 1, "f2")
    window.canvas.set_axis_unit("hz")
    line = window.canvas._proj_axes["f2"].lines[0]
    assert line.get_gid() == "external_projection_f2"
    peak_x = line.get_xdata()[np.argmax(line.get_ydata())]
    assert peak_x == pytest.approx(5.0 * 400, abs=15)  # 5 ppm on the 2D's 400 MHz axis


# --- the drag has to start from the list without changing the plot --------
def _mouse(kind, pos, buttons=Qt.MouseButton.LeftButton, button=Qt.MouseButton.LeftButton,
           modifiers=Qt.KeyboardModifier.NoModifier):
    from PySide6.QtCore import QEvent
    from PySide6.QtGui import QMouseEvent
    types = {"press": QEvent.Type.MouseButtonPress, "move": QEvent.Type.MouseMove,
             "release": QEvent.Type.MouseButtonRelease}
    return QMouseEvent(types[kind], QPointF(pos), QPointF(pos), button, buttons, modifiers)


def test_dragging_a_1d_out_of_open_experiments_leaves_the_2d_on_screen(qapp, monkeypatch):
    """The user-reported bug: pressing on the 1D to drag it selected it,
    which put the 1D on the Spectrum canvas -- so the drop found no
    contour. Driven through real mouse events on the list itself."""
    window = _window_with(qapp, [(_hsqc(), "HSQC"), (_one_d("1H"), "1H")])
    window._select_single_row(0)
    qapp.processEvents()
    hsqc, proton = window.session.spectra
    canvas, source = window.canvas, window.spectrum_list
    assert canvas._contour_spectrum is hsqc
    dragged = []
    monkeypatch.setattr(source, "_start_drag", dragged.append)

    viewport = source.viewport()
    start = source.visualItemRect(source.item(1)).center()
    QApplication.sendEvent(viewport, _mouse("press", start))
    assert canvas._contour_spectrum is hsqc  # nothing selected on press
    far = start + type(start)(QApplication.startDragDistance() + 5, 0)
    QApplication.sendEvent(viewport, _mouse("move", far))
    QApplication.sendEvent(viewport, _mouse("release", far, buttons=Qt.MouseButton.NoButton))
    qapp.processEvents()

    assert len(dragged) == 1
    assert canvas._contour_spectrum is hsqc and source.currentRow() == 0
    from opennmr.gui.widgets.canvas import _rows_from_list_drag
    assert _rows_from_list_drag(dragged[0], canvas._LIST_DRAG_MIME) == [1]

    # ...and that payload, dropped on the top strip, lands.
    bbox = canvas._proj_axes["f2"].get_window_extent()
    dpr = canvas.canvas.device_pixel_ratio
    pos = QPointF((bbox.x0 + bbox.x1) / 2 / dpr, canvas.canvas.height() - (bbox.y0 + bbox.y1) / 2 / dpr)
    for kind in ("enter", "drop"):
        from PySide6.QtGui import QDragEnterEvent
        event = (QDragEnterEvent(pos.toPoint(), Qt.DropAction.CopyAction, dragged[0],
                                 Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
                 if kind == "enter" else
                 QDropEvent(pos, Qt.DropAction.CopyAction, dragged[0],
                            Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier))
        assert canvas.eventFilter(canvas.canvas, event) is True and event.isAccepted()
    qapp.processEvents()
    assert canvas.external_projection("f2")["spectrum"] is proton


def test_a_plain_click_in_open_experiments_still_selects_and_shows(qapp):
    window = _window_with(qapp, [(_hsqc(), "HSQC"), (_one_d("1H"), "1H")])
    window._select_single_row(0)
    qapp.processEvents()
    source = window.spectrum_list
    at = source.visualItemRect(source.item(1)).center()
    QApplication.sendEvent(source.viewport(), _mouse("press", at))
    QApplication.sendEvent(source.viewport(), _mouse("release", at, buttons=Qt.MouseButton.NoButton))
    qapp.processEvents()
    assert source.currentRow() == 1
    assert window.canvas._contour_spectrum is None  # the 1D is shown now


def test_ctrl_click_still_extends_the_selection_straight_away(qapp):
    window = _window_with(qapp, [(_one_d("1H"), "A"), (_one_d("1H", peak=4), "B")])
    window._select_single_row(0)
    qapp.processEvents()
    source = window.spectrum_list
    at = source.visualItemRect(source.item(1)).center()
    QApplication.sendEvent(source.viewport(), _mouse(
        "press", at, modifiers=Qt.KeyboardModifier.ControlModifier))
    assert source.item(1).isSelected() and source.item(0).isSelected()
