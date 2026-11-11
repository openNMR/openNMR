"""2026-11-11, explicit user requests:

* the DOSY read-out laid out like the curve fit's and Fit's read-outs
  (a title, bold and normal type, a table) instead of the fixed-width
  block; the BMS simulation's information likewise;
* opening an openNMR state of many spectra said so in a QMessageBox
  that ran off the bottom of the screen -- now a scrolling message,
  grouped by outcome;
* the Browse panel's experiments sortable by name (A to Z, Z to A) or
  date (newest, oldest first), from a right-click menu.

The plain reports (text(): logged, copied, saved) are unchanged.
"""

import os
import time

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication, QDialog, QMessageBox

from opennmr.gui import report_format as rf
from opennmr.gui.widgets import message_dialog
from opennmr.gui.widgets.data_browser import (
    _PATH_ROLE, DEFAULT_SORT, SORT_MODES, DataBrowser, experiment_date, natural_key,
)
from opennmr.gui.widgets.report_label import ReportLabel

from test_gui import qapp  # noqa: F401 -- fixture


# ---- DOSY ----------------------------------------------------------------------------------

def _dosy(n_peaks=2):
    ppm = np.linspace(0.0, 10.0, 100)
    per_column = {i * 5: {"ppm": float(ppm[i * 5]), "D_mode": 1e-9 * (i + 1),
                          "log10_width": 0.1 * (i + 1)} for i in range(n_peaks)}
    projection = np.zeros(100)
    for i, column in enumerate(per_column):
        projection[column] = i + 1.0                 # the last peak is the largest
    return {"alpha": 0.02, "ppm": ppm, "threshold": 0.05, "projection": projection,
            "per_column": per_column}


def test_dosy_report_is_laid_out_like_the_curve_fit():
    # Short subtitle items are kept whole on a line with &nbsp;.
    html = rf.dosy_report_html(_dosy()).replace("&nbsp;", " ")
    assert "font-size:large" in html and "DOSY map" in html          # a title
    assert "regularisation α = 0.02" in html                          # α stated
    assert "2 of 100 points inverted" in html
    assert "<table" in html and "<i>D</i>" in html and "<i>δ</i> (ppm)" in html
    assert "set by <i>α</i>, not by the sample" in html               # the caveat kept
    # Largest peak first, as the plain summary orders them.
    assert html.index("0.505") < html.index("0.000")


def test_dosy_report_summarises_beyond_twelve_peaks():
    html = rf.dosy_report_html(_dosy(15))
    assert html.count("<tr>") == rf.DOSY_MAX_PEAKS + 1                # header + 12
    assert "… and 3 more." in html


def test_dosy_overlay_titles_each_map_by_its_label():
    html = rf.dosy_overlay_html([("first", _dosy()), ("second <b>", _dosy())])
    assert "first" in html and "second &lt;b&gt;" in html and html.count("<hr>") == 1


def test_main_window_dosy_label_shows_the_layout_and_keeps_the_plain_text(qapp, monkeypatch):
    from test_gui import _load_spectrum_directly
    from opennmr.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
    from opennmr.gui.main_window import MainWindow, _DosySettingsDialog
    from opennmr.processing.fitting import gaussian

    b_values = np.linspace(0.0, 4.0e9, 16)
    ppm = np.linspace(0.0, 10.0, 200)
    data = np.array([gaussian(ppm, 3.0, 0.1, np.exp(-1.5e-9 * b)) for b in b_values])
    spectrum = NMRSpectrum(
        data=data,
        axes=[NMRAxis(kind="index", index=np.arange(1, len(b_values) + 1)),
              NMRAxis(kind="frequency", ppm=ppm, hz=ppm * 400.0)],
        metadata=NMRMetadata(vendor="bruker", nucleus="1H", sfo1=400.0, pseudo_2d=True))
    window = MainWindow()
    _load_spectrum_directly(window, spectrum)
    window.fit_type_combo.setCurrentIndex(1)
    window.delay_source_combo.setCurrentIndex(1)
    window.manual_delays_edit.setText(", ".join(f"{b:.8g}" for b in b_values))
    monkeypatch.setattr(_DosySettingsDialog, "exec", lambda self: QDialog.DialogCode.Accepted)
    window._on_dosy_clicked()

    label = window.dosy_results_label
    assert isinstance(label, ReportLabel)
    assert label.text().startswith("DOSY map (inverse Laplace transform)")   # plain kept
    assert "font-size:large" in label.html() and "<table" in label.html()
    assert not label.font().fixedPitch()
    # Captured with the spectrum, so reselecting it brings the layout back.
    state = window._capture_analysis_state()
    assert rf.is_rich(state["dosy_results_html"])


# ---- BMS -----------------------------------------------------------------------------------

def _bms_parts(**changes):
    parts = {"n_particles": 15, "volume_fraction": 0.31, "grid": "64×64×64",
             "boundary": "periodic", "chi": "", "sample": "", "chi_v": None, "local": None,
             "quad_order": "second", "transitions": "", "mean_ppm": -1.25, "std_ppm": 3.5,
             "fwhm_ppm": 8.24, "isotropic": False, "timing": ["field 0.40 s"],
             "warnings": []}
    parts.update(changes)
    return parts


def test_bms_report_has_a_title_a_quality_style_row_and_sections():
    html = rf.bms_report_html(_bms_parts(
        sample="powder in a rotor 3.2 mm", chi_v=1.2e-4,
        local={"contact_ppm": 10, "aniso_ppm": 50, "eta": 0.2, "C_Q_MHz": 1.5, "eta_Q": 0.3},
        transitions="all", warnings=["the Monte Carlo noise is not small: raise Samples"]))
    assert "BMS shift simulation" in html and "font-size:large" in html
    assert "15&nbsp;particles" in html and "64×64×64&nbsp;grid" in html
    for label in ("mean", "standard deviation", "Gaussian FWHM equivalent"):
        assert f">{label}</span>" in html                            # muted over bold
    assert "-1.25 ppm" in html and "8.24 ppm" in html
    assert "<b>Sample:</b> powder in a rotor 3.2 mm" in html
    assert "<i>χ</i><sub>v</sub>" in html                            # italic symbols
    assert "<i>δ</i><sub>iso</sub> 10 ppm" in html and "<i>C</i><sub>Q</sub> 1.5 MHz" in html
    assert "all transitions" in html
    assert "⚠" in html and "The Monte Carlo noise is not small" in html


def test_bms_report_leaves_out_what_was_not_set():
    html = rf.bms_report_html(_bms_parts())
    assert "Susceptibility" not in html and "Local interactions" not in html and "⚠" not in html


def test_bms_packing_report():
    html = rf.bms_packing_html(12, 15, "200 µm", 0.3, asked=0.35, shrunk=2, mean_size=0.9)
    assert "Packing" in html and "12 of 15" in html and "200 µm" in html
    assert "(asked 0.35)" in html and "2 particles shrunk to fit" in html


@pytest.fixture
def bms_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENNMR_BMS_CACHE", str(tmp_path / "sidebands"))


def test_bms_panel_shows_the_layout_and_keeps_the_plain_summary(qapp, bms_cache, monkeypatch):
    from test_gui_paramagnetic import _bms_panel
    said = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: said.append(a[2]))
    t = _bms_panel(spectra=[])
    assert isinstance(t.pm_bms_result_label, ReportLabel)
    assert isinstance(t.pm_T_result_label, ReportLabel)
    assert isinstance(t.pm_bp_result_label, ReportLabel)

    assert t.generate_bms_packing(), said
    assert "volume fraction" in t.pm_bms_result_label.text()        # plain
    assert "particles placed" in t.pm_bms_result_label.html()      # laid out

    assert t.simulate_bms(), said
    plain = t.pm_bms_result_label.text()
    assert plain == t.bms_summary()
    assert plain.startswith("15 particles") and "standard deviation" in plain
    assert "Time: field" in plain
    html = t.pm_bms_result_label.html()
    assert "BMS shift simulation" in html and "Shift distribution" in html
    parts = t.bms_summary_parts()
    assert parts["n_particles"] == 15 and parts["grid"] == "64×64×64"


# ---- Opening a state: a scrolling message --------------------------------------------------

def _messages(n, folder="C:\\Users\\q\\NMR\\expt"):
    return [f"{231 + i} · 23Na · bruker · procno 1: reopened from {folder}\\{231 + i} and "
            "reprocessed as saved." for i in range(n)]


def test_state_message_groups_by_outcome_and_says_the_folder_once():
    messages = _messages(3) + ["kept fit: opened from the saved copy.",
                               "lost · 1H: NOT opened – its data are no longer at X and the "
                               "state holds no copy."]
    html = rf.state_opened_html("Na_dry", "2026-11-10 17:11:51", 4, messages, notes="dried")
    assert "Opened 4 spectra" in html and "“Na_dry”" in html and "<b>Notes:</b> dried" in html
    assert html.index("⚠") < html.index("Re-read from their data")   # failures first
    assert "Re-read from their data and reprocessed as saved (3)" in html
    assert html.count("C:\\Users\\q\\NMR\\expt") == 1                  # said once
    assert "Opened from the saved copy (1)" in html and "kept fit" in html
    # "231" is the folder and already the head of its label: not repeated.
    assert ">231</span>" not in html


def test_state_message_keeps_a_relative_path_that_differs_from_the_label():
    messages = ["first: reopened from /data/a/10 and reprocessed as saved.",
                "second: reopened from /data/b/20 and reprocessed as saved."]
    html = rf.state_opened_html("s", "", 2, messages)
    assert "in /data" in html and "a/10" in html and "b/20" in html


def test_long_message_scrolls_within_the_screen(qapp):
    plain = "\n".join(_messages(200))
    html = rf.state_opened_html("big", "", 200, _messages(200))
    dialog = message_dialog.show_message(None, "openNMR state opened", html, plain)
    try:
        QApplication.processEvents()
        avail = QApplication.primaryScreen().availableGeometry()
        assert dialog.height() <= int(avail.height() * 0.7) + 1
        assert dialog.browser.verticalScrollBar().maximum() > 0      # it scrolls
        dialog.copy()
        assert QApplication.clipboard().text() == plain
    finally:
        dialog.close()


def test_short_message_is_not_padded_out(qapp):
    html = rf.state_opened_html("small", "", 1, _messages(1))
    dialog = message_dialog.ScrollableMessageDialog(None, "t", html, "p")
    avail = QApplication.primaryScreen().availableGeometry()
    assert dialog.height() < int(avail.height() * 0.7)
    dialog.close()


def test_opening_a_state_uses_the_scrolling_message(qapp, tmp_path, monkeypatch):
    from opennmr.gui import main_window as mw

    shown = []
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: shown.append(a))
    window = mw.MainWindow()
    monkeypatch.setattr(mw.io_state, "is_state_folder", lambda folder: True)
    monkeypatch.setattr(mw.io_state, "read_state_info",
                        lambda folder: {"name": "Na_NMR_MnHCF_dry",
                                        "created": "2026-11-10T17:11:51", "notes": ""})
    messages = _messages(41)
    monkeypatch.setattr(window, "_open_state", lambda folder: (list(range(41)), messages))
    window._on_open_state_clicked(str(tmp_path))
    dialog = window._state_opened_dialog
    try:
        assert not shown                                       # no QMessageBox any more
        assert dialog.windowTitle() == "openNMR state opened"
        plain = dialog.plain_text()
        assert plain.startswith("Opened 41 spectra from “Na_NMR_MnHCF_dry”")
        assert all(m in plain for m in messages)               # Copy gives everything
        assert "Re-read from their data and reprocessed as saved (41)" in \
            dialog.browser.toPlainText()
    finally:
        dialog.close()


# ---- Browse panel: sorting -----------------------------------------------------------------

def test_natural_key_orders_numbers_as_numbers():
    names = ["10", "2", "b", "A", "1p3mm_2", "1p3mm_10"]
    assert sorted(names, key=natural_key) == ["1p3mm_2", "1p3mm_10", "2", "10", "A", "b"]


def _bruker_like(folder, name, mtime):
    expt = folder / name
    expt.mkdir(parents=True)
    fid = expt / "fid"
    fid.write_bytes(b"\0")
    os.utime(fid, (mtime, mtime))
    # The folder's own mtime is deliberately the reverse order: the date
    # must come from the data, not from the folder.
    os.utime(expt, (2e9 - mtime, 2e9 - mtime))
    return str(expt)


def _browser(tmp_path):
    now = time.time()
    paths = [_bruker_like(tmp_path, "2", now - 300),
             _bruker_like(tmp_path, "10", now - 100),
             _bruker_like(tmp_path, "1", now - 200),
             _bruker_like(tmp_path / "old_folder", "5", now - 900),
             _bruker_like(tmp_path / "new_folder", "7", now - 10)]
    browser = DataBrowser()
    # The scan's order: natural, by name.
    browser.set_matches(str(tmp_path), [(p, "bruker") for p in sorted(
        paths, key=lambda p: natural_key(os.path.relpath(p, tmp_path)))])
    return browser


def test_experiment_date_is_the_data_not_the_folder(tmp_path):
    path = _bruker_like(tmp_path, "3", 1.5e9)
    assert experiment_date(path) == pytest.approx(1.5e9)
    assert experiment_date(str(tmp_path / "missing")) == 0.0


def test_default_is_name_a_to_z(qapp, tmp_path):
    browser = _browser(tmp_path)
    assert browser.sort_mode() == DEFAULT_SORT == "name_asc"
    assert browser.ordered_names() == ["1", "2", "10", "new_folder", "old_folder"]


def test_sorting_by_name_and_date(qapp, tmp_path):
    browser = _browser(tmp_path)
    browser.set_sort_mode("name_desc")
    assert browser.ordered_names() == ["old_folder", "new_folder", "10", "2", "1"]
    browser.set_sort_mode("date_new")
    # A folder sorts by the newest experiment in it.
    assert browser.ordered_names() == ["new_folder", "10", "1", "2", "old_folder"]
    browser.set_sort_mode("date_old")
    assert browser.ordered_names() == ["old_folder", "2", "1", "10", "new_folder"]
    browser.set_sort_mode("not a mode")
    assert browser.sort_mode() == "name_asc"
    assert browser.ordered_names() == ["1", "2", "10", "new_folder", "old_folder"]


def test_sorting_keeps_what_was_expanded_and_selected(qapp, tmp_path):
    browser = _browser(tmp_path)
    tree = browser._tree
    folder = next(tree.topLevelItem(i) for i in range(tree.topLevelItemCount())
                  if tree.topLevelItem(i).text(0) == "new_folder")
    folder.setExpanded(True)
    leaf = next(tree.topLevelItem(i) for i in range(tree.topLevelItemCount())
                if tree.topLevelItem(i).data(0, _PATH_ROLE)
                and tree.topLevelItem(i).data(0, _PATH_ROLE).endswith(os.sep + "10"))
    tree.setCurrentItem(leaf)
    browser.set_sort_mode("date_old")
    assert folder.isExpanded()
    assert [p for p, _v in browser.selected_paths()] == [leaf.data(0, _PATH_ROLE)]


def test_a_new_folder_keeps_the_chosen_order(qapp, tmp_path):
    browser = _browser(tmp_path / "first")
    browser.set_sort_mode("name_desc")
    now = time.time()
    paths = [_bruker_like(tmp_path / "second", n, now) for n in ("3", "20", "4")]
    browser.set_matches(str(tmp_path / "second"), [(p, "bruker") for p in paths])
    assert browser.ordered_names() == ["20", "4", "3"]


def test_the_menu_choice_is_announced_and_remembered(qapp, tmp_path, monkeypatch):
    from opennmr.gui.main_window import MainWindow

    browser = _browser(tmp_path)
    heard = []
    browser.sortModeChanged.connect(heard.append)
    browser.set_sort_mode("date_new")                    # restoring: quiet
    browser.set_sort_mode("name_desc", emit=True)        # the menu: announced
    assert heard == ["name_desc"]
    assert set(SORT_MODES) == {"name_asc", "name_desc", "date_new", "date_old"}

    window = MainWindow()
    window.data_browser.set_sort_mode("date_old", emit=True)
    assert window._ui_settings["browser_sort"] == "date_old"
    later = MainWindow()                                 # same settings file (conftest)
    assert later.data_browser.sort_mode() == "date_old"
